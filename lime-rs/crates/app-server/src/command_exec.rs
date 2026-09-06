use crate::execution_process::ExecutionProcessServer;
use app_server_protocol::error_codes;
use app_server_protocol::protocol::v2::GrantedPermissionProfile;
use app_server_protocol::protocol::v2::{
    CommandExecOutputDeltaNotification, CommandExecOutputStream, CommandExecParams,
    CommandExecResizeParams, CommandExecResizeResponse, CommandExecResponse,
    CommandExecTerminateParams, CommandExecTerminateResponse, CommandExecWriteParams,
    CommandExecWriteResponse, ServerNotification,
};
use app_server_protocol::{JsonRpcError, JsonRpcNotification};
use app_server_transport::ConnectionId;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use futures::future::BoxFuture;
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tool_runtime::execution_process::{
    start_local_execution_process, ExecutionOutputDelta, ExecutionOutputKind,
    LocalExecutionProcessControlHandle, LocalExecutionProcessEvent, LocalExecutionRequest,
    LocalExecutionSandbox,
};

const DEFAULT_OUTPUT_BYTES_CAP: usize = 1024 * 1024;
const COMMAND_EXEC_TIMEOUT_EXIT_CODE: i32 = 124;

pub(crate) type CommandExecNotificationHook =
    Arc<dyn Fn(ConnectionId, JsonRpcNotification) -> BoxFuture<'static, ()> + Send + Sync>;

#[derive(Clone, Default)]
pub(crate) struct CommandExecServer {
    sessions: Arc<Mutex<HashMap<CommandExecKey, CommandExecSession>>>,
    notification_hook: Arc<Mutex<Option<CommandExecNotificationHook>>>,
    process_server: ExecutionProcessServer,
}

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
struct CommandExecKey {
    connection_id: ConnectionId,
    process_id: String,
}

#[derive(Clone)]
struct CommandExecSession {
    process_id: String,
    control: LocalExecutionProcessControlHandle,
    stream_stdin: bool,
    tty: bool,
    stdin_open: Arc<Mutex<bool>>,
}

impl CommandExecServer {
    pub(crate) fn with_process_server(mut self, process_server: ExecutionProcessServer) -> Self {
        self.process_server = process_server;
        self
    }

    pub(crate) fn with_notification_hook(mut self, hook: CommandExecNotificationHook) -> Self {
        self.set_notification_hook(hook);
        self
    }

    pub(crate) fn set_notification_hook(&mut self, hook: CommandExecNotificationHook) {
        *self
            .notification_hook
            .try_lock()
            .expect("command exec notification hook mutex poisoned") = Some(hook);
    }

    pub(crate) async fn exec(
        &self,
        connection_id: ConnectionId,
        params: CommandExecParams,
        granted_permissions: Option<GrantedPermissionProfile>,
    ) -> Result<CommandExecResponse, JsonRpcError> {
        validate_exec(&params)?;
        let process_id = params
            .process_id
            .clone()
            .unwrap_or_else(|| format!("command-{}", uuid::Uuid::new_v4()));
        let key = CommandExecKey {
            connection_id,
            process_id: process_id.clone(),
        };
        let owner_base_id = owner_process_id(connection_id, &process_id);
        let cwd = params
            .cwd
            .clone()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        if !cwd.is_absolute() {
            return Err(invalid_params("command/exec cwd must be an absolute path"));
        }
        let cwd = std::fs::canonicalize(&cwd)
            .map_err(|error| invalid_request(format!("command/exec cwd is invalid: {error}")))?;
        if !cwd.is_dir() {
            return Err(invalid_request("command/exec cwd must be a directory"));
        }
        let sandbox_policy = sandbox_policy_label(params.sandbox_policy.as_ref());
        let decision = crate::execution_process::decide_command_execution(
            &params.command,
            &cwd,
            sandbox_policy.as_deref(),
        )
        .map_err(invalid_runtime)?;
        let sandbox = if decision.0 {
            Some(LocalExecutionSandbox {
                backend: decision.1.ok_or_else(|| {
                    invalid_runtime("command/exec sandbox backend is unavailable")
                })?,
                requested_policy: sandbox_policy.clone(),
                granted_permissions,
                windows_mode: crate::execution_process::runtime_windows_sandbox_mode(
                    crate::runtime_backend::current_agent_runtime_config_metadata().as_ref(),
                ),
            })
        } else {
            None
        };
        let mut env = std::env::vars().collect::<HashMap<_, _>>();
        if let Some(overrides) = &params.env {
            for (key, value) in overrides {
                match value {
                    Some(value) => {
                        env.insert(key.clone(), value.clone());
                    }
                    None => {
                        env.remove(key);
                    }
                }
            }
        }
        let stream_stdin = params.tty || params.stream_stdin;
        let stream_output = params.tty || params.stream_stdout_stderr;
        let output_cap = if params.disable_output_cap {
            None
        } else {
            Some(params.output_bytes_cap.unwrap_or(DEFAULT_OUTPUT_BYTES_CAP))
        };
        let timeout = if params.disable_timeout {
            None
        } else {
            parse_timeout(params.timeout_ms)?
        };
        let mut sessions = self.sessions.lock().await;
        if sessions.contains_key(&key) {
            return Err(invalid_request(format!(
                "duplicate active command/exec process id: {process_id}"
            )));
        }
        let owner_id = if self.process_server.status(&owner_base_id).is_ok() {
            format!("{owner_base_id}-{}", uuid::Uuid::new_v4())
        } else {
            owner_base_id
        };
        let request = LocalExecutionRequest {
            process_id: owner_id.clone(),
            tool_id: owner_id.clone(),
            tool_name: "command/exec".to_string(),
            command: params.command.clone(),
            cwd: Some(cwd),
            env,
            tty: params.tty,
            stdin: stream_stdin,
            env_clear: false,
            pty_size: params.size.map(|size| (size.rows, size.cols)),
            sandbox,
        };
        let mut handle = start_local_execution_process(request)
            .map_err(|error| invalid_runtime(format!("failed to spawn command/exec: {error}")))?;
        if let Some(size) = params.size {
            if let Err(error) = handle.resize(size.rows, size.cols) {
                let _ = handle.terminate();
                return Err(invalid_runtime(error.to_string()));
            }
        }
        if let Err(error) = self
            .process_server
            .register_process_handle(handle.control_handle(), handle.status())
        {
            let _ = handle.terminate();
            return Err(invalid_runtime(format!(
                "failed to register command/exec process: {error}"
            )));
        }
        sessions.insert(
            key.clone(),
            CommandExecSession {
                process_id: owner_id.clone(),
                control: handle.control_handle(),
                stream_stdin,
                tty: params.tty,
                stdin_open: Arc::new(Mutex::new(stream_stdin)),
            },
        );
        drop(sessions);

        let mut stdout = Capture::new(output_cap);
        let mut stderr = Capture::new(output_cap);
        let mut timeout_sleep = timeout.map(tokio::time::sleep).map(Box::pin);
        let mut timed_out = false;
        let response = loop {
            tokio::select! {
                event = handle.next_event() => {
                    let Some(event) = event else {
                        break CommandExecResponse {
                            exit_code: if timed_out {
                                COMMAND_EXEC_TIMEOUT_EXIT_CODE
                            } else {
                                -1
                            },
                            stdout: stdout.text(),
                            stderr: stderr.text(),
                        };
                    };
                    match event {
                        LocalExecutionProcessEvent::Output(delta) => {
                            if let Err(error) = self.process_server.record_process_output(delta.clone())
                            {
                                tracing::warn!(
                                    process_id = %process_id,
                                    %error,
                                    "failed to record command/exec output in shared process owner"
                                );
                            }
                            self.handle_delta(connection_id, &process_id, delta, stream_output, &mut stdout, &mut stderr).await;
                        }
                        LocalExecutionProcessEvent::Exited(snapshot) => {
                            if let Err(error) = self.process_server.finish_process(snapshot.clone()) {
                                tracing::warn!(
                                    process_id = %process_id,
                                    %error,
                                    "failed to finalize command/exec in shared process owner"
                                );
                            }
                            break CommandExecResponse {
                                exit_code: if timed_out {
                                    COMMAND_EXEC_TIMEOUT_EXIT_CODE
                                } else {
                                    snapshot.exit_code.unwrap_or(-1)
                                },
                                stdout: if stream_output { String::new() } else { stdout.text() },
                                stderr: if stream_output { String::new() } else { stderr.text() },
                            };
                        }
                    }
                }
                _ = async { if let Some(sleep) = timeout_sleep.as_mut() { sleep.await } }, if timeout_sleep.is_some() => {
                    timed_out = true;
                    let _ = self.process_server.terminate(&owner_id);
                    timeout_sleep = None;
                }
            }
        };
        self.sessions.lock().await.remove(&key);
        Ok(response)
    }

    pub(crate) async fn write(
        &self,
        connection_id: ConnectionId,
        params: CommandExecWriteParams,
    ) -> Result<CommandExecWriteResponse, JsonRpcError> {
        if params.delta_base64.is_none() && !params.close_stdin {
            return Err(invalid_params(
                "command/exec/write requires deltaBase64 or closeStdin",
            ));
        }
        let session = self
            .session(&key(connection_id, &params.process_id))
            .await?;
        if !session.stream_stdin {
            return Err(invalid_request("command/exec stdin is not enabled"));
        }
        let bytes = params
            .delta_base64
            .map(|value| {
                STANDARD
                    .decode(value)
                    .map_err(|error| invalid_params(error.to_string()))
            })
            .transpose()?
            .unwrap_or_default();
        let mut open = session.stdin_open.lock().await;
        if !bytes.is_empty() {
            if !*open {
                return Err(invalid_request("command/exec stdin is already closed"));
            }
            self.process_server
                .write_stdin(&session.process_id, &bytes)
                .map_err(|error| invalid_runtime(error.to_string()))?;
        }
        if params.close_stdin {
            session
                .control
                .close_stdin()
                .map_err(|error| invalid_runtime(error.to_string()))?;
            *open = false;
        }
        Ok(CommandExecWriteResponse {})
    }

    pub(crate) async fn resize(
        &self,
        connection_id: ConnectionId,
        params: CommandExecResizeParams,
    ) -> Result<CommandExecResizeResponse, JsonRpcError> {
        let session = self
            .session(&key(connection_id, &params.process_id))
            .await?;
        if !session.tty {
            return Err(invalid_request(
                "command/exec/resize requires a tty process",
            ));
        }
        if params.size.rows == 0 || params.size.cols == 0 {
            return Err(invalid_params(
                "command/exec terminal size must be greater than 0",
            ));
        }
        session
            .control
            .resize(params.size.rows, params.size.cols)
            .map_err(|error| invalid_runtime(error.to_string()))?;
        Ok(CommandExecResizeResponse {})
    }

    pub(crate) async fn terminate(
        &self,
        connection_id: ConnectionId,
        params: CommandExecTerminateParams,
    ) -> Result<CommandExecTerminateResponse, JsonRpcError> {
        let session = self
            .session(&key(connection_id, &params.process_id))
            .await?;
        self.process_server
            .terminate(&session.process_id)
            .map_err(|error| invalid_runtime(error.to_string()))?;
        Ok(CommandExecTerminateResponse {})
    }

    pub(crate) async fn connection_closed(&self, connection_id: ConnectionId) {
        let sessions = {
            let mut guard = self.sessions.lock().await;
            let keys = guard
                .keys()
                .filter(|key| key.connection_id == connection_id)
                .cloned()
                .collect::<Vec<_>>();
            keys.into_iter()
                .filter_map(|key| guard.remove(&key))
                .collect::<Vec<_>>()
        };
        for session in sessions {
            let _ = self.process_server.terminate(&session.process_id);
        }
    }

    async fn session(&self, key: &CommandExecKey) -> Result<CommandExecSession, JsonRpcError> {
        self.sessions.lock().await.get(key).cloned().ok_or_else(|| {
            invalid_request(format!(
                "no active command/exec process: {}",
                key.process_id
            ))
        })
    }

    async fn handle_delta(
        &self,
        connection_id: ConnectionId,
        process_id: &str,
        delta: ExecutionOutputDelta,
        stream_output: bool,
        stdout: &mut Capture,
        stderr: &mut Capture,
    ) {
        let (stream, capture) = match delta.kind {
            ExecutionOutputKind::Stderr => (CommandExecOutputStream::Stderr, stderr),
            ExecutionOutputKind::Stdout | ExecutionOutputKind::Combined => {
                (CommandExecOutputStream::Stdout, stdout)
            }
        };
        let Some(bytes) = capture.accept(&delta.raw_bytes) else {
            return;
        };
        if stream_output && (!bytes.is_empty() || capture.cap_reached) {
            let hook = self.notification_hook.lock().await.clone();
            if let Some(hook) = hook {
                hook(
                    connection_id,
                    JsonRpcNotification::from(ServerNotification::CommandExecOutputDelta(
                        CommandExecOutputDeltaNotification {
                            process_id: process_id.to_string(),
                            stream,
                            delta_base64: STANDARD.encode(bytes),
                            cap_reached: capture.cap_reached,
                        },
                    )),
                )
                .await;
            }
        }
    }
}

#[derive(Default)]
struct Capture {
    cap: Option<usize>,
    bytes: Vec<u8>,
    seen: usize,
    cap_reached: bool,
}

impl Capture {
    fn new(cap: Option<usize>) -> Self {
        Self {
            cap,
            ..Self::default()
        }
    }
    fn accept(&mut self, bytes: &[u8]) -> Option<Vec<u8>> {
        if self.cap_reached {
            return None;
        }
        let available = self
            .cap
            .map(|cap| cap.saturating_sub(self.seen))
            .unwrap_or(bytes.len());
        let accepted_len = available.min(bytes.len());
        let accepted = bytes[..accepted_len].to_vec();
        self.seen = self.seen.saturating_add(accepted_len);
        self.bytes.extend_from_slice(&accepted);
        self.cap_reached = self.cap.is_some_and(|cap| self.seen == cap);
        Some(accepted)
    }
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.bytes).to_string()
    }
}

fn validate_exec(params: &CommandExecParams) -> Result<(), JsonRpcError> {
    if params.command.is_empty() {
        return Err(invalid_request("command/exec command must not be empty"));
    }
    if params
        .process_id
        .as_deref()
        .is_some_and(|id| id.trim().is_empty())
    {
        return Err(invalid_params("processId must not be empty"));
    }
    if params.process_id.is_none()
        && (params.tty || params.stream_stdin || params.stream_stdout_stderr)
    {
        return Err(invalid_request(
            "streaming command/exec requires a client-supplied processId",
        ));
    }
    if params.sandbox_policy.is_some() && params.permission_profile.is_some() {
        return Err(invalid_params(
            "permissionProfile cannot be combined with sandboxPolicy",
        ));
    }
    if params.disable_output_cap && params.output_bytes_cap.is_some() {
        return Err(invalid_params(
            "command/exec cannot set both outputBytesCap and disableOutputCap",
        ));
    }
    if params.disable_timeout && params.timeout_ms.is_some() {
        return Err(invalid_params(
            "command/exec cannot set both timeoutMs and disableTimeout",
        ));
    }
    if params.size.is_some() && !params.tty {
        return Err(invalid_params("command/exec size requires tty: true"));
    }
    if params
        .size
        .is_some_and(|size| size.rows == 0 || size.cols == 0)
    {
        return Err(invalid_params(
            "command/exec terminal size must be greater than 0",
        ));
    }
    Ok(())
}

fn parse_timeout(timeout: Option<i64>) -> Result<Option<std::time::Duration>, JsonRpcError> {
    match timeout {
        None => Ok(Some(std::time::Duration::from_secs(10))),
        Some(value) if value >= 0 => Ok(Some(std::time::Duration::from_millis(value as u64))),
        Some(value) => Err(invalid_params(format!(
            "timeoutMs must be non-negative, got {value}"
        ))),
    }
}

fn sandbox_policy_label(value: Option<&Value>) -> Option<String> {
    let value = value?;
    if let Some(label) = value.as_str() {
        return Some(label.to_string());
    }
    value.get("type").and_then(Value::as_str).map(|value| {
        let mut label = String::new();
        for (index, character) in value.chars().enumerate() {
            if character.is_uppercase() && index > 0 {
                label.push('-');
            }
            label.push(character.to_ascii_lowercase());
        }
        label
    })
}

fn key(connection_id: ConnectionId, process_id: &str) -> CommandExecKey {
    CommandExecKey {
        connection_id,
        process_id: process_id.to_string(),
    }
}

fn owner_process_id(connection_id: ConnectionId, process_id: &str) -> String {
    format!("command-exec-{}-{process_id}", connection_id.0)
}

fn invalid_request(message: impl Into<String>) -> JsonRpcError {
    JsonRpcError::new(error_codes::INVALID_REQUEST, message)
}
fn invalid_params(message: impl Into<String>) -> JsonRpcError {
    JsonRpcError::new(error_codes::INVALID_PARAMS, message)
}
fn invalid_runtime(message: impl Into<String>) -> JsonRpcError {
    JsonRpcError::new(error_codes::RUNTIME_ERROR, message)
}

#[cfg(test)]
mod tests;
