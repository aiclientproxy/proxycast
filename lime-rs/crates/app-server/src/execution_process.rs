use app_server_protocol::protocol::v2::{GrantedPermissionProfile, ThreadBackgroundTerminal};
use serde_json::json;
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tool_runtime::execution_decision::{
    decide_tool_execution, ToolExecutionDecisionInput, ToolExecutionDecisionKind,
    ToolExecutionPolicyDecisionOptions,
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
};
use tool_runtime::sandbox::{prepare_sandbox_command, SandboxCommandRequest};
use tool_runtime::shell::{is_shell_tool_name, shell_command_text_from_argv};
use tool_runtime::shell_permission::{check_shell_command_permission, ShellPermissionDecision};

const DEFAULT_DRAIN_LIMIT: usize = 128;
const MAX_DRAIN_LIMIT: usize = 1024;
const OUTPUT_EVENT_CAP: usize = 4096;
const OUTPUT_BYTE_CAP: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, Default)]
pub struct ExecutionProcessServer {
    inner: Arc<Mutex<ExecutionProcessState>>,
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
    final_snapshot: Option<ExecutionProcessSnapshot>,
    background: Option<BackgroundTerminalEntry>,
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
    #[error("Execution process only supports shell tools")]
    UnsupportedTool,
    #[error("Failed to prepare sandboxed execution process: {0}")]
    Sandbox(String),
    #[error("Failed to control execution process: {0}")]
    Control(String),
    #[error("Execution process state is unavailable")]
    Lock,
}

impl ExecutionProcessServer {
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
        self.start_process_inner(Some((thread_id, display_command)), request, None)
            .await
    }

    pub(crate) async fn start_thread_process_with_permissions(
        &self,
        thread_id: &str,
        display_command: &str,
        request: LiveExecutionRequest,
        granted_permissions: Option<GrantedPermissionProfile>,
    ) -> Result<ExecutionProcessSnapshot, ExecutionProcessError> {
        let thread_id = thread_id.trim();
        if thread_id.is_empty() {
            return Err(ExecutionProcessError::Control(
                "background terminal thread id must not be empty".to_string(),
            ));
        }
        self.start_process_inner(
            Some((thread_id, display_command)),
            request,
            granted_permissions,
        )
        .await
    }

    async fn start_process_inner(
        &self,
        thread_scope: Option<(&str, &str)>,
        request: LiveExecutionRequest,
        granted_permissions: Option<GrantedPermissionProfile>,
    ) -> Result<ExecutionProcessSnapshot, ExecutionProcessError> {
        if request.command.is_empty() {
            return Err(ExecutionProcessError::EmptyCommand);
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
        let decision = decide_tool_execution(
            ToolExecutionDecisionInput {
                tool_name: canonical_tool_name,
                params: &json!({ "command": command_text }),
                working_directory: &working_directory,
                surface: "execution_process",
                auto_mode: false,
                bypass_restrictions: false,
                approval_policy: request.approval_policy.as_deref(),
                requested_sandbox_policy: request.sandbox_policy.as_deref(),
                resolver_input: ToolExecutionResolverInput {
                    persisted_policy: None,
                    request_metadata: request.runtime_metadata.as_ref(),
                },
            },
            app_server_tool_execution_policy_options(),
        );
        match decision.kind {
            ToolExecutionDecisionKind::Allow => {}
            ToolExecutionDecisionKind::RequiresApproval
            | ToolExecutionDecisionKind::Deny
            | ToolExecutionDecisionKind::SandboxBlocked => {
                return Err(ExecutionProcessError::Policy(format!(
                    "{}: {}",
                    decision.reason_code, decision.reason
                )));
            }
        }
        validate_shell_execution_process_command(
            canonical_tool_name,
            &command_text,
            &working_directory,
            request.approval_policy.as_deref(),
            request.sandbox_policy.as_deref(),
        )
        .map_err(ExecutionProcessError::Policy)?;

        let command = if decision.workspace_sandbox_backend_enforced() {
            prepare_sandbox_command(SandboxCommandRequest {
                backend: decision.sandbox_backend().ok_or_else(|| {
                    ExecutionProcessError::Sandbox(
                        "execution decision did not identify a sandbox backend".to_string(),
                    )
                })?,
                requested_policy: request.sandbox_policy.as_deref(),
                command: request.command,
                working_directory: &working_directory,
                granted_permissions: granted_permissions.as_ref(),
            })
            .map_err(|error| ExecutionProcessError::Sandbox(error.to_string()))?
        } else {
            request.command
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
                    final_snapshot: None,
                    background,
                },
            );
        }
        let request = LocalExecutionRequest {
            process_id: process_id.clone(),
            tool_id: request.tool_id,
            tool_name: canonical_tool_name.to_string(),
            command,
            cwd: Some(working_directory),
            env: request.env,
            tty: request.tty,
            stdin: true,
            env_clear: false,
            pty_size: None,
        };
        let mut handle = match start_local_execution_process(request) {
            Ok(handle) => handle,
            Err(error) => {
                if let Ok(mut state) = self.inner.lock() {
                    state.processes.remove(&process_id);
                }
                return Err(ExecutionProcessError::Start(error.to_string()));
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
                let handle = entry.handle.as_ref()?;
                if background.thread_id != thread_id
                    || !background.listed
                    || handle.status().status.is_terminal()
                {
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
            )
        };
        tool_runtime::unified_exec::forget_process_session(&process.0);
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
                    ))
                })
                .collect::<Vec<_>>()
        };
        for (process_id, handle) in processes {
            tool_runtime::unified_exec::forget_process_session(&process_id);
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
