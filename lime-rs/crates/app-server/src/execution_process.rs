use crate::processor::environment::EnvironmentRegistry;
use crate::processor::environment_exec::RemoteExecClient;
use app_server_protocol::protocol::v2::ThreadBackgroundTerminal;
use base64::Engine;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tool_runtime::execution_decision::{
    decide_tool_execution, ToolExecutionDecisionInput, ToolExecutionDecisionKind,
    ToolExecutionPolicyDecisionOptions,
};
use tool_runtime::execution_orchestrator::{
    RuntimeToolApprovalPolicy, RuntimeToolApprovalSource, RuntimeToolExecutionAttempt,
    RuntimeToolSandboxPolicy,
};
use tool_runtime::execution_policy::{
    ToolExecutionPolicy, ToolExecutionRestrictionProfile, ToolExecutionSandboxProfile,
    ToolExecutionWarningPolicy,
};
use tool_runtime::execution_policy_service::ToolExecutionResolverInput;
use tool_runtime::execution_process::{
    live::{LiveExecutionOutputBatch, LiveExecutionOutputQuery, LiveExecutionRequest},
    start_local_execution_process, ExecutionOutputDelta, ExecutionProcessSnapshot,
    LiveExecutionProcessRegistry, LocalExecutionProcessControlHandle, LocalExecutionRequest,
    LocalExecutionSandbox,
};
use tool_runtime::sandbox::{
    plan_sandbox_backend, SandboxBackendPlanInput, SandboxBackendPlatform,
};
use tool_runtime::shell::{is_shell_tool_name, shell_command_text_from_argv};
use tool_runtime::shell_permission::{check_shell_command_permission, ShellPermissionDecision};

const DEFAULT_DRAIN_LIMIT: usize = 128;
const MAX_DRAIN_LIMIT: usize = 1024;
const OUTPUT_EVENT_CAP: usize = 4096;
const OUTPUT_BYTE_CAP: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, Default)]
pub struct ExecutionProcessServer {
    inner: Arc<Mutex<ExecutionProcessState>>,
    environment_registry: Arc<std::sync::RwLock<Option<Arc<EnvironmentRegistry>>>>,
}

#[derive(Debug)]
struct ExecutionProcessState {
    processes: HashMap<String, ExecutionProcessEntry>,
    output: VecDeque<ExecutionOutputDelta>,
    output_bytes: usize,
    next_background_process_id: u64,
}

#[derive(Debug)]
struct ExecutionProcessEntry {
    handle: Option<LocalExecutionProcessControlHandle>,
    remote_control: Option<RemoteProcessControl>,
    snapshot: Option<ExecutionProcessSnapshot>,
    final_snapshot: Option<ExecutionProcessSnapshot>,
    background: Option<BackgroundTerminalEntry>,
}

#[derive(Debug, Clone)]
struct RemoteProcessControl {
    commands: mpsc::UnboundedSender<RemoteProcessCommand>,
}

#[derive(Debug)]
enum RemoteProcessCommand {
    Write(Vec<u8>),
    #[cfg(test)]
    Signal,
    Terminate,
}

#[derive(Debug, Clone)]
struct BackgroundTerminalEntry {
    thread_id: String,
    public_process_id: u64,
    item_id: String,
    command: String,
    cwd: String,
    listed: bool,
}

impl Default for ExecutionProcessState {
    fn default() -> Self {
        Self {
            processes: HashMap::new(),
            output: VecDeque::new(),
            output_bytes: 0,
            next_background_process_id: 1,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ExecutionProcessError {
    #[error("Execution process command must not be empty")]
    EmptyCommand,
    #[error("Execution process already exists: {0}")]
    ProcessExists(String),
    #[error("Execution process not found: {0}")]
    ProcessNotFound(String),
    #[error("Execution process working directory is invalid: {0}")]
    WorkingDirectory(String),
    #[error("Failed to start execution process: {0}")]
    Start(String),
    #[error("Execution process rejected by policy: {0}")]
    Policy(String),
    #[error("Execution process denied by sandbox: {message}")]
    SandboxDenied {
        reason_code: String,
        message: String,
    },
    #[error("Execution process denied managed network access: {message}")]
    ManagedNetworkDenied {
        reason_code: String,
        message: String,
        host: Option<String>,
    },
    #[error("Execution process was canceled: {0}")]
    Canceled(String),
    #[error("Execution process only supports shell tools")]
    UnsupportedTool,
    #[error("Execution environment is not available for process execution: {0}")]
    UnsupportedEnvironment(String),
    #[error("Failed to prepare sandboxed execution process: {0}")]
    Sandbox(String),
    #[error("Failed to control execution process: {0}")]
    Control(String),
    #[error("Execution process state is unavailable")]
    Lock,
}

impl ExecutionProcessServer {
    pub(crate) fn attach_environment_registry(&self, registry: Arc<EnvironmentRegistry>) {
        *self
            .environment_registry
            .write()
            .expect("Environment registry lock must not be poisoned") = Some(registry);
    }

    pub fn register_process_handle(
        &self,
        handle: LocalExecutionProcessControlHandle,
        snapshot: ExecutionProcessSnapshot,
    ) -> Result<(), ExecutionProcessError> {
        if handle.process_id() != snapshot.process_id {
            return Err(ExecutionProcessError::Control(format!(
                "control handle process id {} does not match snapshot process id {}",
                handle.process_id(),
                snapshot.process_id
            )));
        }
        let process_id = snapshot.process_id.clone();
        let is_terminal = snapshot.status.is_terminal();
        let mut state = self.inner.lock().map_err(|_| ExecutionProcessError::Lock)?;
        if state.processes.contains_key(&process_id) {
            return Err(ExecutionProcessError::ProcessExists(process_id));
        }
        state.processes.insert(
            process_id,
            ExecutionProcessEntry {
                handle: if is_terminal { None } else { Some(handle) },
                remote_control: None,
                snapshot: Some(snapshot.clone()),
                final_snapshot: if is_terminal { Some(snapshot) } else { None },
                background: None,
            },
        );
        Ok(())
    }

    pub fn record_process_output(
        &self,
        delta: ExecutionOutputDelta,
    ) -> Result<(), ExecutionProcessError> {
        let mut state = self.inner.lock().map_err(|_| ExecutionProcessError::Lock)?;
        if !state.processes.contains_key(&delta.process_id) {
            return Err(ExecutionProcessError::ProcessNotFound(delta.process_id));
        }
        push_output(&mut state, delta);
        Ok(())
    }

    pub fn finish_process(
        &self,
        snapshot: ExecutionProcessSnapshot,
    ) -> Result<(), ExecutionProcessError> {
        let process_id = snapshot.process_id.clone();
        let mut state = self.inner.lock().map_err(|_| ExecutionProcessError::Lock)?;
        let entry = state
            .processes
            .get_mut(&process_id)
            .ok_or_else(|| ExecutionProcessError::ProcessNotFound(process_id.clone()))?;
        entry.handle = None;
        entry.remote_control = None;
        entry.snapshot = Some(snapshot.clone());
        entry.final_snapshot = Some(snapshot);
        Ok(())
    }

    pub async fn start_thread_process(
        &self,
        thread_id: &str,
        display_command: &str,
        request: LiveExecutionRequest,
    ) -> Result<ExecutionProcessSnapshot, ExecutionProcessError> {
        let thread_id = thread_id.trim();
        if thread_id.is_empty() {
            return Err(ExecutionProcessError::Control(
                "background terminal thread id must not be empty".to_string(),
            ));
        }
        self.start_process_inner(Some((thread_id, display_command)), request)
            .await
    }

    async fn start_process_inner(
        &self,
        thread_scope: Option<(&str, &str)>,
        request: LiveExecutionRequest,
    ) -> Result<ExecutionProcessSnapshot, ExecutionProcessError> {
        if request.command.is_empty() {
            return Err(ExecutionProcessError::EmptyCommand);
        }
        if request.environment_id != "local" {
            return self.start_remote_process(thread_scope, request).await;
        }
        let requested_working_directory = request.working_directory;
        let working_directory =
            std::fs::canonicalize(&requested_working_directory).map_err(|error| {
                ExecutionProcessError::WorkingDirectory(format!(
                    "{}: {error}",
                    requested_working_directory.display()
                ))
            })?;
        if !working_directory.is_dir() {
            return Err(ExecutionProcessError::WorkingDirectory(format!(
                "{} is not a directory",
                working_directory.display()
            )));
        }
        let canonical_tool_name = canonical_shell_tool_name(&request.tool_name)
            .ok_or(ExecutionProcessError::UnsupportedTool)?;
        let command_text = shell_command_text_from_argv(&request.command);
        let sandbox = match request.attempt.as_ref() {
            Some(attempt) => {
                validate_orchestrated_attempt_identity(attempt, &request.tool_id)?;
                if attempt.is_cancelled() {
                    return Err(ExecutionProcessError::Canceled(
                        "cancellation was requested before process start".to_string(),
                    ));
                }
                validate_orchestrated_shell_command(
                    canonical_tool_name,
                    &command_text,
                    &working_directory,
                    attempt,
                )?;
                orchestrated_sandbox(attempt, request.runtime_metadata.as_ref())?
            }
            None => legacy_execution_sandbox(
                canonical_tool_name,
                &command_text,
                &working_directory,
                request.approval_policy.as_deref(),
                request.sandbox_policy.as_deref(),
                request.runtime_metadata.as_ref(),
            )?,
        };
        let process_id = request.process_id.clone();
        let background = thread_scope.map(|(thread_id, display_command)| BackgroundTerminalEntry {
            thread_id: thread_id.to_string(),
            public_process_id: 0,
            item_id: request.tool_id.clone(),
            command: display_command.trim().to_string(),
            cwd: working_directory.to_string_lossy().to_string(),
            listed: true,
        });
        {
            let mut state = self.inner.lock().map_err(|_| ExecutionProcessError::Lock)?;
            if state.processes.contains_key(&process_id) {
                return Err(ExecutionProcessError::ProcessExists(process_id));
            }
            let background = background.map(|mut background| {
                background.public_process_id = state.next_background_process_id;
                state.next_background_process_id =
                    state.next_background_process_id.saturating_add(1);
                background
            });
            state.processes.insert(
                process_id.clone(),
                ExecutionProcessEntry {
                    handle: None,
                    remote_control: None,
                    snapshot: None,
                    final_snapshot: None,
                    background,
                },
            );
        }
        let sandboxed = sandbox.is_some();
        let request = LocalExecutionRequest {
            process_id: process_id.clone(),
            tool_id: request.tool_id,
            tool_name: canonical_tool_name.to_string(),
            command: request.command,
            cwd: Some(working_directory),
            env: request.env,
            tty: request.tty,
            stdin: true,
            env_clear: false,
            pty_size: None,
            sandbox,
        };
        let mut handle = match start_local_execution_process(request) {
            Ok(handle) => handle,
            Err(error) => {
                if let Ok(mut state) = self.inner.lock() {
                    state.processes.remove(&process_id);
                }
                return Err(if sandboxed {
                    ExecutionProcessError::SandboxDenied {
                        reason_code: "sandbox_process_start_failed".to_string(),
                        message: error.to_string(),
                    }
                } else {
                    ExecutionProcessError::Start(error.to_string())
                });
            }
        };
        let snapshot = handle.status();
        let control_handle = handle.control_handle();
        let inner = Arc::clone(&self.inner);

        {
            let mut state = self.inner.lock().map_err(|_| ExecutionProcessError::Lock)?;
            let entry = state
                .processes
                .get_mut(&process_id)
                .ok_or_else(|| ExecutionProcessError::ProcessNotFound(process_id.clone()))?;
            entry.handle = Some(control_handle);
        }

        tokio::spawn(async move {
            while let Some(delta) = handle.recv_output().await {
                if let Ok(mut state) = inner.lock() {
                    push_output(&mut state, delta);
                }
            }
            let final_snapshot = handle.wait().await.ok();
            if let Ok(mut state) = inner.lock() {
                if let Some(entry) = state.processes.get_mut(&process_id) {
                    entry.handle = None;
                    entry.final_snapshot = final_snapshot;
                }
            }
        });

        Ok(snapshot)
    }

    async fn start_remote_process(
        &self,
        thread_scope: Option<(&str, &str)>,
        request: LiveExecutionRequest,
    ) -> Result<ExecutionProcessSnapshot, ExecutionProcessError> {
        let registry = self
            .environment_registry
            .read()
            .map_err(|_| ExecutionProcessError::Lock)?
            .clone()
            .ok_or_else(|| {
                ExecutionProcessError::UnsupportedEnvironment(request.environment_id.clone())
            })?;
        let client = registry
            .execution_client(&request.environment_id)
            .await
            .map_err(ExecutionProcessError::UnsupportedEnvironment)?;
        let cwd = remote_path_uri(&request.working_directory)
            .map_err(ExecutionProcessError::WorkingDirectory)?;
        let sandbox = remote_sandbox_context(&request, &cwd)?;
        let response: RemoteProcessStartResponse = client
            .request(
                "process/start",
                json!({
                    "processId": request.process_id.clone(),
                    "argv": request.command.clone(),
                    "cwd": cwd,
                    "env": request.env.clone(),
                    "tty": request.tty,
                    "pipeStdin": true,
                    "arg0": Value::Null,
                    "sandbox": sandbox,
                }),
            )
            .await
            .map_err(ExecutionProcessError::Start)?;
        if response.process_id != request.process_id {
            return Err(ExecutionProcessError::Start(format!(
                "exec-server returned process id `{}` for `{}`",
                response.process_id, request.process_id
            )));
        }

        let (commands, receiver) = mpsc::unbounded_channel();
        let process_id = request.process_id.clone();
        let snapshot = ExecutionProcessSnapshot {
            process_id: process_id.clone(),
            tool_id: request.tool_id.clone(),
            tool_name: request.tool_name.clone(),
            status: tool_runtime::execution_process::ExecutionProcessStatus::Running,
            exit_code: None,
            elapsed_ms: 0,
            output_bytes: 0,
            output_omitted_bytes: 0,
            output_truncated: false,
            retained_output: String::new(),
            failure: None,
        };
        let background = thread_scope.map(|(thread_id, display_command)| BackgroundTerminalEntry {
            thread_id: thread_id.to_string(),
            public_process_id: 0,
            item_id: request.tool_id.clone(),
            command: display_command.trim().to_string(),
            cwd: request.working_directory.to_string_lossy().to_string(),
            listed: true,
        });
        {
            let mut state = self.inner.lock().map_err(|_| ExecutionProcessError::Lock)?;
            if state.processes.contains_key(&process_id) {
                return Err(ExecutionProcessError::ProcessExists(process_id));
            }
            let background = background.map(|mut background| {
                background.public_process_id = state.next_background_process_id;
                state.next_background_process_id =
                    state.next_background_process_id.saturating_add(1);
                background
            });
            state.processes.insert(
                process_id.clone(),
                ExecutionProcessEntry {
                    handle: None,
                    remote_control: Some(RemoteProcessControl { commands }),
                    snapshot: Some(snapshot.clone()),
                    final_snapshot: None,
                    background,
                },
            );
        }
        let inner = Arc::clone(&self.inner);
        tokio::spawn(run_remote_process(
            inner,
            client,
            process_id,
            request.tool_id,
            request.tool_name,
            receiver,
        ));
        Ok(snapshot)
    }

    pub fn list_background_terminals(
        &self,
        thread_id: &str,
    ) -> Result<Vec<ThreadBackgroundTerminal>, ExecutionProcessError> {
        let state = self.inner.lock().map_err(|_| ExecutionProcessError::Lock)?;
        let mut terminals = state
            .processes
            .values()
            .filter_map(|entry| {
                let background = entry.background.as_ref()?;
                if background.thread_id != thread_id || !background.listed {
                    return None;
                }
                let active = entry
                    .handle
                    .as_ref()
                    .map(|handle| !handle.status().status.is_terminal())
                    .or_else(|| {
                        entry
                            .snapshot
                            .as_ref()
                            .map(|snapshot| !snapshot.status.is_terminal())
                    })
                    .unwrap_or(false);
                if !active {
                    return None;
                }
                Some((
                    background.public_process_id,
                    ThreadBackgroundTerminal {
                        item_id: background.item_id.clone(),
                        process_id: background.public_process_id.to_string(),
                        command: background.command.clone(),
                        cwd: background.cwd.clone(),
                        os_pid: None,
                        cpu_percent: None,
                        rss_kb: None,
                    },
                ))
            })
            .collect::<Vec<_>>();
        terminals.sort_by_key(|(process_id, _)| *process_id);
        Ok(terminals
            .into_iter()
            .map(|(_, terminal)| terminal)
            .collect())
    }

    pub fn terminate_background_terminal(
        &self,
        thread_id: &str,
        public_process_id: u64,
    ) -> Result<bool, ExecutionProcessError> {
        let process = {
            let mut state = self.inner.lock().map_err(|_| ExecutionProcessError::Lock)?;
            let Some((process_id, entry)) = state.processes.iter_mut().find(|(_, entry)| {
                entry.background.as_ref().is_some_and(|background| {
                    background.thread_id == thread_id
                        && background.public_process_id == public_process_id
                        && background.listed
                })
            }) else {
                return Ok(false);
            };
            let Some(background) = entry.background.as_mut() else {
                return Ok(false);
            };
            background.listed = false;
            (
                process_id.clone(),
                entry.handle.as_ref().and_then(|handle| {
                    (!handle.status().status.is_terminal()).then(|| handle.clone())
                }),
                entry.remote_control.clone(),
            )
        };
        tool_runtime::unified_exec::forget_process_session(&process.0);
        if let Some(control) = process.2 {
            return Ok(control
                .commands
                .send(RemoteProcessCommand::Terminate)
                .is_ok());
        }
        let Some(handle) = process.1 else {
            return Ok(false);
        };
        Ok(handle.terminate().is_ok())
    }

    pub fn clean_background_terminals(&self, thread_id: &str) -> Result<(), ExecutionProcessError> {
        let processes = {
            let mut state = self.inner.lock().map_err(|_| ExecutionProcessError::Lock)?;
            state
                .processes
                .iter_mut()
                .filter_map(|(process_id, entry)| {
                    let background = entry.background.as_mut()?;
                    if background.thread_id != thread_id || !background.listed {
                        return None;
                    }
                    background.listed = false;
                    Some((
                        process_id.clone(),
                        entry.handle.as_ref().and_then(|handle| {
                            (!handle.status().status.is_terminal()).then(|| handle.clone())
                        }),
                        entry.remote_control.clone(),
                    ))
                })
                .collect::<Vec<_>>()
        };
        for (process_id, handle, remote_control) in processes {
            tool_runtime::unified_exec::forget_process_session(&process_id);
            if let Some(control) = remote_control {
                let _ = control.commands.send(RemoteProcessCommand::Terminate);
            }
            if let Some(handle) = handle {
                let _ = handle.terminate();
            }
        }
        Ok(())
    }

    pub fn write_stdin(&self, process_id: &str, data: &[u8]) -> Result<(), ExecutionProcessError> {
        let state = self.inner.lock().map_err(|_| ExecutionProcessError::Lock)?;
        let entry = state
            .processes
            .get(process_id)
            .ok_or_else(|| ExecutionProcessError::ProcessNotFound(process_id.to_string()))?;
        if let Some(control) = entry.remote_control.as_ref() {
            control
                .commands
                .send(RemoteProcessCommand::Write(data.to_vec()))
                .map_err(|_| {
                    ExecutionProcessError::Control(
                        "remote process control channel closed".to_string(),
                    )
                })?;
            return Ok(());
        }
        let Some(handle) = entry.handle.as_ref() else {
            return Err(ExecutionProcessError::ProcessNotFound(
                process_id.to_string(),
            ));
        };
        handle
            .write_stdin(data)
            .map_err(|error| ExecutionProcessError::Control(format!("{error:?}")))?;
        Ok(())
    }

    pub fn terminate(
        &self,
        process_id: &str,
    ) -> Result<ExecutionProcessSnapshot, ExecutionProcessError> {
        let state = self.inner.lock().map_err(|_| ExecutionProcessError::Lock)?;
        let entry = state
            .processes
            .get(process_id)
            .ok_or_else(|| ExecutionProcessError::ProcessNotFound(process_id.to_string()))?;
        if let Some(control) = entry.remote_control.as_ref() {
            control
                .commands
                .send(RemoteProcessCommand::Terminate)
                .map_err(|_| {
                    ExecutionProcessError::Control(
                        "remote process control channel closed".to_string(),
                    )
                })?;
            return entry
                .snapshot
                .clone()
                .ok_or_else(|| ExecutionProcessError::ProcessNotFound(process_id.to_string()));
        }
        let Some(handle) = entry.handle.as_ref() else {
            return entry
                .final_snapshot
                .clone()
                .ok_or_else(|| ExecutionProcessError::ProcessNotFound(process_id.to_string()));
        };
        handle
            .terminate()
            .map_err(|error| ExecutionProcessError::Control(format!("{error:?}")))?;
        Ok(handle.status())
    }

    #[cfg(test)]
    pub fn signal(&self, process_id: &str) -> Result<(), ExecutionProcessError> {
        let state = self.inner.lock().map_err(|_| ExecutionProcessError::Lock)?;
        let entry = state
            .processes
            .get(process_id)
            .ok_or_else(|| ExecutionProcessError::ProcessNotFound(process_id.to_string()))?;
        if let Some(control) = entry.remote_control.as_ref() {
            control
                .commands
                .send(RemoteProcessCommand::Signal)
                .map_err(|_| {
                    ExecutionProcessError::Control(
                        "remote process control channel closed".to_string(),
                    )
                })?;
            return Ok(());
        }
        let Some(handle) = entry.handle.as_ref() else {
            return Err(ExecutionProcessError::ProcessNotFound(
                process_id.to_string(),
            ));
        };
        handle
            .write_stdin(&[3])
            .map_err(|error| ExecutionProcessError::Control(format!("{error:?}")))
    }

    pub fn status(
        &self,
        process_id: &str,
    ) -> Result<ExecutionProcessSnapshot, ExecutionProcessError> {
        let state = self.inner.lock().map_err(|_| ExecutionProcessError::Lock)?;
        let entry = state
            .processes
            .get(process_id)
            .ok_or_else(|| ExecutionProcessError::ProcessNotFound(process_id.to_string()))?;
        if let Some(snapshot) = &entry.final_snapshot {
            return Ok(snapshot.clone());
        }
        if entry.remote_control.is_some() {
            return entry
                .snapshot
                .clone()
                .ok_or_else(|| ExecutionProcessError::ProcessNotFound(process_id.to_string()));
        }
        let Some(handle) = entry.handle.as_ref() else {
            return Err(ExecutionProcessError::ProcessNotFound(
                process_id.to_string(),
            ));
        };
        Ok(handle.status())
    }

    pub fn drain_output(
        &self,
        query: LiveExecutionOutputQuery,
    ) -> Result<LiveExecutionOutputBatch, ExecutionProcessError> {
        let limit = query
            .limit
            .map(usize::from)
            .unwrap_or(DEFAULT_DRAIN_LIMIT)
            .min(MAX_DRAIN_LIMIT);
        let max_bytes = query
            .max_bytes
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(usize::MAX);
        let after_sequence = query.after_sequence.unwrap_or_default();
        let state = self.inner.lock().map_err(|_| ExecutionProcessError::Lock)?;
        let mut deltas = Vec::new();
        let mut bytes = 0usize;
        let mut next_sequence = query.after_sequence;

        for delta in state.output.iter() {
            if deltas.len() >= limit {
                break;
            }
            if query
                .process_id
                .as_ref()
                .is_some_and(|process_id| process_id != &delta.process_id)
            {
                continue;
            }
            if delta.sequence <= after_sequence {
                continue;
            }
            let delta_bytes = delta.delta.len();
            if !deltas.is_empty() && bytes.saturating_add(delta_bytes) > max_bytes {
                break;
            }
            bytes = bytes.saturating_add(delta_bytes);
            next_sequence = Some(next_sequence.unwrap_or_default().max(delta.sequence));
            deltas.push(delta.clone());
        }

        Ok(LiveExecutionOutputBatch {
            deltas,
            next_sequence,
        })
    }
}

fn validate_orchestrated_attempt_identity(
    attempt: &RuntimeToolExecutionAttempt,
    tool_id: &str,
) -> Result<(), ExecutionProcessError> {
    if attempt.identity().call_id() == tool_id {
        return Ok(());
    }
    Err(ExecutionProcessError::Policy(format!(
        "tool attempt call id '{}' does not match process tool id '{tool_id}'",
        attempt.identity().call_id()
    )))
}

fn validate_orchestrated_shell_command(
    tool_name: &str,
    command_text: &str,
    working_directory: &Path,
    attempt: &RuntimeToolExecutionAttempt,
) -> Result<(), ExecutionProcessError> {
    match check_shell_command_permission(tool_name, command_text, working_directory) {
        ShellPermissionDecision::Allow => Ok(()),
        ShellPermissionDecision::Deny(reason) => Err(ExecutionProcessError::Policy(reason)),
        ShellPermissionDecision::RequiresConfirmation(_)
            if attempt.approval_policy() == RuntimeToolApprovalPolicy::Never
                || attempt.approval_source() != RuntimeToolApprovalSource::Config
                || attempt.effective_sandbox_policy()
                    == RuntimeToolSandboxPolicy::DangerFullAccess =>
        {
            Ok(())
        }
        ShellPermissionDecision::RequiresConfirmation(reason) => {
            Err(ExecutionProcessError::Policy(reason))
        }
    }
}

fn orchestrated_sandbox(
    attempt: &RuntimeToolExecutionAttempt,
    runtime_metadata: Option<&serde_json::Value>,
) -> Result<Option<LocalExecutionSandbox>, ExecutionProcessError> {
    let effective_policy = attempt.effective_sandbox_policy();
    if matches!(
        effective_policy,
        RuntimeToolSandboxPolicy::None | RuntimeToolSandboxPolicy::DangerFullAccess
    ) {
        return Ok(None);
    }
    let plan = plan_sandbox_backend(SandboxBackendPlanInput {
        sandbox_profile: ToolExecutionSandboxProfile::WorkspaceCommand,
        requested_policy: effective_policy.label(),
        request_metadata: runtime_metadata,
        bypass_restrictions: false,
        platform: SandboxBackendPlatform::current(),
    });
    if plan.strict_fallback_blocks_execution() {
        return Err(ExecutionProcessError::SandboxDenied {
            reason_code: plan.reason_code.to_string(),
            message: plan.reason.to_string(),
        });
    }
    if let Some(host) = attempt.managed_network_host() {
        let network_granted = attempt
            .granted_permissions()
            .network
            .as_ref()
            .and_then(|network| network.enabled)
            .unwrap_or(false);
        if !network_granted {
            return Err(ExecutionProcessError::ManagedNetworkDenied {
                reason_code: "managed_network_denied".to_string(),
                message: format!("network access to '{host}' is blocked by managed policy"),
                host: Some(host.to_string()),
            });
        }
    }
    if !plan.can_run_with_backend() {
        return Ok(None);
    }
    Ok(Some(LocalExecutionSandbox {
        backend: plan.backend,
        requested_policy: effective_policy.label().map(str::to_string),
        granted_permissions: Some(attempt.granted_permissions().clone()),
    }))
}

fn legacy_execution_sandbox(
    tool_name: &str,
    command_text: &str,
    working_directory: &Path,
    approval_policy: Option<&str>,
    sandbox_policy: Option<&str>,
    runtime_metadata: Option<&serde_json::Value>,
) -> Result<Option<LocalExecutionSandbox>, ExecutionProcessError> {
    let decision = decide_tool_execution(
        ToolExecutionDecisionInput {
            tool_name,
            params: &json!({ "command": command_text }),
            working_directory,
            surface: "execution_process",
            auto_mode: false,
            bypass_restrictions: false,
            approval_policy,
            requested_sandbox_policy: sandbox_policy,
            resolver_input: ToolExecutionResolverInput {
                persisted_policy: None,
                request_metadata: runtime_metadata,
            },
        },
        app_server_tool_execution_policy_options(),
    );
    match decision.kind {
        ToolExecutionDecisionKind::Allow => {}
        ToolExecutionDecisionKind::RequiresApproval | ToolExecutionDecisionKind::Deny => {
            return Err(ExecutionProcessError::Policy(format!(
                "{}: {}",
                decision.reason_code, decision.reason
            )));
        }
        ToolExecutionDecisionKind::SandboxBlocked => {
            return Err(ExecutionProcessError::SandboxDenied {
                reason_code: decision.reason_code,
                message: decision.reason,
            });
        }
    }
    validate_shell_execution_process_command(
        tool_name,
        command_text,
        working_directory,
        approval_policy,
        sandbox_policy,
    )
    .map_err(ExecutionProcessError::Policy)?;

    if !decision.workspace_sandbox_backend_enforced() {
        return Ok(None);
    }
    Ok(Some(LocalExecutionSandbox {
        backend: decision.sandbox_backend().ok_or_else(|| {
            ExecutionProcessError::Sandbox(
                "execution decision did not identify a sandbox backend".to_string(),
            )
        })?,
        requested_policy: sandbox_policy.map(str::to_string),
        granted_permissions: None,
    }))
}

pub(crate) fn decide_command_execution(
    command: &[String],
    working_directory: &Path,
    sandbox_policy: Option<&str>,
) -> Result<(bool, Option<tool_runtime::sandbox::SandboxBackend>), String> {
    let command_text = shell_command_text_from_argv(command);
    let decision = decide_tool_execution(
        ToolExecutionDecisionInput {
            tool_name: tool_runtime::unified_exec::EXEC_COMMAND_TOOL_NAME,
            params: &json!({ "command": command_text }),
            working_directory,
            surface: "command_exec",
            auto_mode: false,
            bypass_restrictions: false,
            approval_policy: None,
            requested_sandbox_policy: sandbox_policy,
            resolver_input: ToolExecutionResolverInput {
                persisted_policy: None,
                request_metadata: None,
            },
        },
        app_server_tool_execution_policy_options(),
    );
    if !decision.allowed() {
        return Err(format!("{}: {}", decision.reason_code, decision.reason));
    }
    validate_shell_execution_process_command(
        tool_runtime::unified_exec::EXEC_COMMAND_TOOL_NAME,
        &command_text,
        working_directory,
        None,
        sandbox_policy,
    )?;
    Ok((
        decision.workspace_sandbox_backend_enforced(),
        decision.sandbox_backend(),
    ))
}

impl LiveExecutionProcessRegistry for ExecutionProcessServer {
    fn register_live_process(
        &self,
        handle: LocalExecutionProcessControlHandle,
        snapshot: ExecutionProcessSnapshot,
    ) -> Result<(), String> {
        self.register_process_handle(handle, snapshot)
            .map_err(|error| error.to_string())
    }

    fn record_live_process_output(&self, delta: ExecutionOutputDelta) -> Result<(), String> {
        self.record_process_output(delta)
            .map_err(|error| error.to_string())
    }

    fn finish_live_process(&self, snapshot: ExecutionProcessSnapshot) -> Result<(), String> {
        self.finish_process(snapshot)
            .map_err(|error| error.to_string())
    }
}

fn canonical_shell_tool_name(tool_name: &str) -> Option<&'static str> {
    is_shell_tool_name(tool_name).then_some(tool_runtime::unified_exec::EXEC_COMMAND_TOOL_NAME)
}

fn app_server_tool_execution_policy_options() -> ToolExecutionPolicyDecisionOptions {
    ToolExecutionPolicyDecisionOptions {
        default_policy_for_tool: app_server_default_tool_execution_policy,
        tool_names_match: app_server_tool_names_match,
    }
}

fn app_server_default_tool_execution_policy(tool_name: &str) -> ToolExecutionPolicy {
    if canonical_shell_tool_name(tool_name).is_some() {
        return ToolExecutionPolicy {
            warning_policy: ToolExecutionWarningPolicy::ShellCommandRisk,
            restriction_profile: ToolExecutionRestrictionProfile::WorkspaceShellCommand,
            sandbox_profile: ToolExecutionSandboxProfile::WorkspaceCommand,
        };
    }
    ToolExecutionPolicy::default()
}

fn app_server_tool_names_match(left: &str, right: &str) -> bool {
    normalized_tool_name(left) == normalized_tool_name(right)
}

fn validate_shell_execution_process_command(
    tool_name: &str,
    command_text: &str,
    working_directory: &Path,
    approval_policy: Option<&str>,
    sandbox_policy: Option<&str>,
) -> Result<(), String> {
    match check_shell_command_permission(tool_name, command_text, working_directory) {
        ShellPermissionDecision::Allow => Ok(()),
        ShellPermissionDecision::Deny(reason) => Err(reason),
        ShellPermissionDecision::RequiresConfirmation(_)
            if approval_policy.is_some_and(|policy| policy.eq_ignore_ascii_case("never"))
                || sandbox_policy
                    .is_some_and(|policy| policy.eq_ignore_ascii_case("danger-full-access")) =>
        {
            Ok(())
        }
        ShellPermissionDecision::RequiresConfirmation(message) => Err(message),
    }
}

fn normalized_tool_name(tool_name: &str) -> String {
    tool_name
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .map(|character| character.to_ascii_lowercase())
        .collect()
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RemoteProcessStartResponse {
    process_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RemoteProcessReadResponse {
    #[serde(default)]
    chunks: Vec<RemoteProcessOutputChunk>,
    #[serde(default)]
    next_seq: u64,
    #[serde(default)]
    exited: bool,
    exit_code: Option<i32>,
    #[serde(default)]
    closed: bool,
    failure: Option<String>,
    #[serde(default)]
    sandbox_denied: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RemoteProcessOutputChunk {
    seq: u64,
    stream: String,
    chunk: String,
}

fn remote_sandbox_context(
    request: &LiveExecutionRequest,
    cwd: &app_server_protocol::protocol::v2::PathUri,
) -> Result<Value, ExecutionProcessError> {
    let Some(label) = request
        .sandbox_policy
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(Value::Null);
    };
    let (file_system, network) = match label.to_ascii_lowercase().replace('_', "-").as_str() {
        "read-only" => (
            json!({
                "type": "restricted",
                "entries": [{
                    "path": {"type": "path", "path": cwd.as_str()},
                    "access": "read"
                }]
            }),
            "restricted",
        ),
        "workspace-write" => (
            json!({
                "type": "restricted",
                "entries": [{
                    "path": {"type": "path", "path": cwd.as_str()},
                    "access": "write"
                }]
            }),
            "restricted",
        ),
        "danger-full-access" => (json!({"type": "unrestricted"}), "enabled"),
        other => {
            return Err(ExecutionProcessError::SandboxDenied {
                reason_code: "remote_unknown_sandbox_policy".to_string(),
                message: format!("remote Environment cannot lower sandbox policy `{other}`"),
            });
        }
    };
    Ok(json!({
        "permissions": {
            "type": "managed",
            "fileSystem": file_system,
            "network": network,
        },
        "cwd": cwd.as_str(),
        "workspaceRoots": [cwd.as_str()],
        "windowsSandboxLevel": "disabled",
        "windowsSandboxPrivateDesktop": false,
        "useLegacyLandlock": false,
    }))
}

fn remote_path_uri(path: &Path) -> Result<app_server_protocol::protocol::v2::PathUri, String> {
    let rendered = path.to_string_lossy();
    if rendered.len() >= 3
        && rendered.as_bytes()[0].is_ascii_alphabetic()
        && rendered.as_bytes()[1] == b':'
        && matches!(rendered.as_bytes()[2], b'/' | b'\\')
    {
        let normalized = rendered.replace('\\', "/");
        return app_server_protocol::protocol::v2::PathUri::parse(&format!("file:///{normalized}"));
    }
    app_server_protocol::protocol::v2::PathUri::from_host_path(path)
}

async fn run_remote_process(
    inner: Arc<Mutex<ExecutionProcessState>>,
    client: Arc<RemoteExecClient>,
    process_id: String,
    tool_id: String,
    tool_name: String,
    mut commands: mpsc::UnboundedReceiver<RemoteProcessCommand>,
) {
    let started_at = std::time::Instant::now();
    let mut read_future = Box::pin(client.request::<RemoteProcessReadResponse>(
        "process/read",
        json!({
            "processId": process_id.clone(),
            "afterSeq": Value::Null,
            "maxBytes": 256 * 1024,
            "waitMs": 500,
        }),
    ));
    let terminal = loop {
        tokio::select! {
            command = commands.recv() => {
                let Some(command) = command else { break false };
                match command {
                    RemoteProcessCommand::Write(data) => {
                        let _ = client.request::<Value>("process/write", json!({
                            "processId": process_id.clone(),
                            "chunk": base64::engine::general_purpose::STANDARD.encode(data),
                            "writeId": uuid::Uuid::new_v4().to_string(),
                        })).await;
                    }
                    #[cfg(test)]
                    RemoteProcessCommand::Signal => {
                        let _ = client.request::<Value>("process/signal", json!({
                            "processId": process_id.clone(),
                            "signal": "interrupt",
                        })).await;
                    }
                    RemoteProcessCommand::Terminate => {
                        let _ = client.request::<Value>("process/terminate", json!({
                            "processId": process_id.clone(),
                        })).await;
                    }
                }
            }
            response = &mut read_future => {
                let response = match response {
                    Ok(response) => response,
                    Err(error) => {
                        finish_remote_process(&inner, &process_id, &tool_id, &tool_name, started_at.elapsed().as_millis() as u64, Some(error), false, None).await;
                        break true;
                    }
                };
                for chunk in response.chunks {
                    let raw_bytes = match base64::engine::general_purpose::STANDARD.decode(chunk.chunk) {
                        Ok(bytes) => bytes,
                        Err(error) => {
                            finish_remote_process(&inner, &process_id, &tool_id, &tool_name, started_at.elapsed().as_millis() as u64, Some(format!("invalid process/read output: {error}")), false, None).await;
                            return;
                        }
                    };
                    let kind = if chunk.stream.eq_ignore_ascii_case("stderr") {
                        tool_runtime::execution_process::ExecutionOutputKind::Stderr
                    } else {
                        tool_runtime::execution_process::ExecutionOutputKind::Stdout
                    };
                    let delta = tool_runtime::execution_process::ExecutionOutputDelta {
                        process_id: process_id.clone(),
                        tool_id: tool_id.clone(),
                        // Environment process cursors are zero-based; the local
                        // execution stream reserves sequence 0 as the initial cursor.
                        sequence: chunk.seq.saturating_add(1),
                        kind,
                        delta: String::from_utf8_lossy(&raw_bytes).into_owned(),
                        bytes: raw_bytes.len() as u64,
                        omitted_bytes: 0,
                        truncated: false,
                        raw_bytes,
                    };
                    if let Ok(mut state) = inner.lock() {
                        if let Some(entry) = state.processes.get_mut(&process_id) {
                            if let Some(snapshot) = entry.snapshot.as_mut() {
                                snapshot.output_bytes = snapshot.output_bytes.saturating_add(delta.bytes);
                                snapshot.retained_output.push_str(&delta.delta);
                                if snapshot.retained_output.len() > 128 * 1024 {
                                    let trim = snapshot.retained_output.len() - 128 * 1024;
                                    snapshot.retained_output.drain(..trim);
                                    snapshot.output_omitted_bytes = snapshot.output_omitted_bytes.saturating_add(trim as u64);
                                    snapshot.output_truncated = true;
                                }
                            }
                        }
                        push_output(&mut state, delta);
                    }
                }
                if response.exited || response.closed || response.failure.is_some() || response.sandbox_denied {
                    finish_remote_process(&inner, &process_id, &tool_id, &tool_name, started_at.elapsed().as_millis() as u64, response.failure, response.sandbox_denied, response.exit_code).await;
                    break true;
                }
                let after_seq = Some(response.next_seq);
                read_future = Box::pin(client.request::<RemoteProcessReadResponse>(
                    "process/read",
                    json!({
                        "processId": process_id.clone(),
                        "afterSeq": after_seq,
                        "maxBytes": 256 * 1024,
                        "waitMs": 500,
                    }),
                ));
            }
        }
    };
    if !terminal {
        finish_remote_process(
            &inner,
            &process_id,
            &tool_id,
            &tool_name,
            started_at.elapsed().as_millis() as u64,
            Some("remote process control channel closed".to_string()),
            false,
            None,
        )
        .await;
    }
}

async fn finish_remote_process(
    inner: &Arc<Mutex<ExecutionProcessState>>,
    process_id: &str,
    tool_id: &str,
    tool_name: &str,
    elapsed_ms: u64,
    failure: Option<String>,
    sandbox_denied: bool,
    exit_code: Option<i32>,
) {
    if let Ok(mut state) = inner.lock() {
        if let Some(entry) = state.processes.get_mut(process_id) {
            let mut snapshot = entry
                .snapshot
                .clone()
                .unwrap_or_else(|| ExecutionProcessSnapshot {
                    process_id: process_id.to_string(),
                    tool_id: tool_id.to_string(),
                    tool_name: tool_name.to_string(),
                    status: tool_runtime::execution_process::ExecutionProcessStatus::Running,
                    exit_code: None,
                    elapsed_ms: 0,
                    output_bytes: 0,
                    output_omitted_bytes: 0,
                    output_truncated: false,
                    retained_output: String::new(),
                    failure: None,
                });
            snapshot.status = if failure.is_some() || sandbox_denied {
                tool_runtime::execution_process::ExecutionProcessStatus::Failed
            } else {
                tool_runtime::execution_process::ExecutionProcessStatus::Exited
            };
            snapshot.exit_code = exit_code;
            snapshot.elapsed_ms = elapsed_ms;
            snapshot.failure = failure;
            entry.snapshot = Some(snapshot.clone());
            entry.remote_control = None;
            entry.final_snapshot = Some(snapshot);
        }
    }
}

fn push_output(state: &mut ExecutionProcessState, mut delta: ExecutionOutputDelta) {
    delta.raw_bytes.clear();
    state.output_bytes = state.output_bytes.saturating_add(delta.delta.len());
    state.output.push_back(delta);
    while state.output.len() > OUTPUT_EVENT_CAP || state.output_bytes > OUTPUT_BYTE_CAP {
        let Some(evicted) = state.output.pop_front() else {
            state.output_bytes = 0;
            break;
        };
        state.output_bytes = state.output_bytes.saturating_sub(evicted.delta.len());
    }
}

#[cfg(test)]
#[path = "execution_process/tests.rs"]
mod tests;
