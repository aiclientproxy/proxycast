use app_server_protocol::error_codes;
use app_server_protocol::protocol::v2::{
    ProcessExitedNotification, ProcessKillParams, ProcessKillResponse,
    ProcessOutputDeltaNotification, ProcessOutputStream, ProcessResizePtyParams,
    ProcessResizePtyResponse, ProcessSpawnParams, ProcessSpawnResponse, ProcessWriteStdinParams,
    ProcessWriteStdinResponse, ServerNotification,
};
use app_server_protocol::{JsonRpcError, JsonRpcNotification};
use app_server_transport::ConnectionId;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use futures::future::BoxFuture;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{oneshot, Mutex};
use tool_runtime::execution_process::{
    start_local_execution_process, ExecutionOutputDelta, ExecutionOutputKind,
    LocalExecutionProcessControlHandle, LocalExecutionProcessEvent, LocalExecutionProcessHandle,
    LocalExecutionRequest,
};

const DEFAULT_OUTPUT_BYTES_CAP: usize = 1024 * 1024;
const PROCESS_TIMEOUT_EXIT_CODE: i32 = 124;

pub(crate) type ProcessNotificationHook =
    Arc<dyn Fn(ConnectionId, JsonRpcNotification) -> BoxFuture<'static, ()> + Send + Sync>;

#[derive(Clone, Default)]
pub(crate) struct ProcessServer {
    sessions: Arc<Mutex<HashMap<ProcessKey, ProcessSession>>>,
    notification_hook: Arc<Mutex<Option<ProcessNotificationHook>>>,
}

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
struct ProcessKey {
    connection_id: ConnectionId,
    process_handle: String,
}

#[derive(Clone)]
struct ProcessSession {
    control: LocalExecutionProcessControlHandle,
    stream_stdin: bool,
    tty: bool,
    stdin_open: Arc<Mutex<bool>>,
    activation: Arc<Mutex<Option<oneshot::Sender<()>>>>,
}

impl ProcessServer {
    pub(crate) fn with_notification_hook(mut self, hook: ProcessNotificationHook) -> Self {
        self.set_notification_hook(hook);
        self
    }

    pub(crate) fn set_notification_hook(&mut self, hook: ProcessNotificationHook) {
        *self
            .notification_hook
            .try_lock()
            .expect("process notification hook mutex poisoned") = Some(hook);
    }

    pub(crate) async fn spawn(
        &self,
        connection_id: ConnectionId,
        params: ProcessSpawnParams,
    ) -> Result<ProcessSpawnResponse, JsonRpcError> {
        validate_spawn(&params)?;
        let cwd = PathBuf::from(&params.cwd);
        let process_id = format!("process-{}-{}", connection_id.0, uuid::Uuid::new_v4());
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
        let output_cap = params
            .output_bytes_cap
            .unwrap_or(Some(DEFAULT_OUTPUT_BYTES_CAP));
        let timeout = parse_timeout(params.timeout_ms)?;
        let mut request =
            LocalExecutionRequest::new(process_id, "process", "process", params.command.clone());
        request.cwd = Some(cwd);
        request.env = env;
        request.tty = params.tty;
        request.stdin = stream_stdin;
        request.env_clear = true;
        request.pty_size = params.size.map(|size| (size.rows, size.cols));
        let mut handle = start_local_execution_process(request).map_err(|error| {
            JsonRpcError::new(
                error_codes::RUNTIME_ERROR,
                format!("failed to spawn process: {error}"),
            )
        })?;
        if let Some(size) = params.size {
            handle
                .resize(size.rows, size.cols)
                .map_err(|error| invalid_runtime(error.to_string()))?;
        }

        let key = ProcessKey {
            connection_id,
            process_handle: params.process_handle.clone(),
        };
        {
            let mut sessions = self.sessions.lock().await;
            if sessions.contains_key(&key) {
                let _ = handle.terminate();
                return Err(invalid_request(format!(
                    "duplicate active process handle: {:?}",
                    params.process_handle
                )));
            }
            let (activation_tx, activation_rx) = oneshot::channel();
            sessions.insert(
                key.clone(),
                ProcessSession {
                    control: handle.control_handle(),
                    stream_stdin,
                    tty: params.tty,
                    stdin_open: Arc::new(Mutex::new(stream_stdin)),
                    activation: Arc::new(Mutex::new(Some(activation_tx))),
                },
            );

            let server = self.clone();
            let process_handle = params.process_handle;
            tokio::spawn(async move {
                let _ = activation_rx.await;
                server
                    .run_session(
                        key,
                        process_handle,
                        &mut handle,
                        stream_output,
                        output_cap,
                        timeout,
                    )
                    .await;
            });
        }
        Ok(ProcessSpawnResponse {})
    }

    pub(crate) async fn activate(&self, connection_id: ConnectionId, process_handle: &str) {
        let activation = self
            .sessions
            .lock()
            .await
            .get(&key(connection_id, process_handle))
            .map(|session| session.activation.clone());
        if let Some(activation) = activation {
            if let Some(sender) = activation.lock().await.take() {
                let _ = sender.send(());
            }
        }
    }

    pub(crate) async fn write_stdin(
        &self,
        connection_id: ConnectionId,
        params: ProcessWriteStdinParams,
    ) -> Result<ProcessWriteStdinResponse, JsonRpcError> {
        if params.delta_base64.is_none() && !params.close_stdin {
            return Err(invalid_params(
                "process/writeStdin requires deltaBase64 or closeStdin",
            ));
        }
        let key = key(connection_id, &params.process_handle);
        let session = self.session(&key).await?;
        if !session.stream_stdin {
            return Err(invalid_request("process stdin is not enabled"));
        }
        let bytes = params
            .delta_base64
            .map(|delta| {
                STANDARD
                    .decode(delta)
                    .map_err(|error| invalid_params(format!("invalid deltaBase64: {error}")))
            })
            .transpose()?
            .unwrap_or_default();
        let mut stdin_open = session.stdin_open.lock().await;
        if !bytes.is_empty() {
            if !*stdin_open {
                return Err(invalid_request("process stdin is already closed"));
            }
            session
                .control
                .write_stdin(bytes)
                .map_err(|error| invalid_runtime(error.to_string()))?;
        }
        if params.close_stdin {
            session
                .control
                .close_stdin()
                .map_err(|error| invalid_runtime(error.to_string()))?;
            *stdin_open = false;
        }
        Ok(ProcessWriteStdinResponse {})
    }

    pub(crate) async fn resize_pty(
        &self,
        connection_id: ConnectionId,
        params: ProcessResizePtyParams,
    ) -> Result<ProcessResizePtyResponse, JsonRpcError> {
        let session = self
            .session(&key(connection_id, &params.process_handle))
            .await?;
        if !session.tty {
            return Err(invalid_request("process/resizePty requires a tty process"));
        }
        validate_terminal_size(params.size)?;
        session
            .control
            .resize(params.size.rows, params.size.cols)
            .map_err(|error| invalid_runtime(error.to_string()))?;
        Ok(ProcessResizePtyResponse {})
    }

    pub(crate) async fn kill(
        &self,
        connection_id: ConnectionId,
        params: ProcessKillParams,
    ) -> Result<ProcessKillResponse, JsonRpcError> {
        let session = self
            .session(&key(connection_id, &params.process_handle))
            .await?;
        session
            .control
            .terminate()
            .map_err(|error| invalid_runtime(error.to_string()))?;
        Ok(ProcessKillResponse {})
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
            if let Some(sender) = session.activation.lock().await.take() {
                let _ = sender.send(());
            }
            let _ = session.control.terminate();
        }
    }

    async fn session(&self, key: &ProcessKey) -> Result<ProcessSession, JsonRpcError> {
        self.sessions
            .lock()
            .await
            .get(key)
            .cloned()
            .ok_or_else(|| invalid_request(format!("no active process: {}", key.process_handle)))
    }

    async fn run_session(
        &self,
        key: ProcessKey,
        process_handle: String,
        handle: &mut LocalExecutionProcessHandle,
        stream_output: bool,
        output_cap: Option<usize>,
        timeout: Option<std::time::Duration>,
    ) {
        let mut stdout = Capture::new(output_cap);
        let mut stderr = Capture::new(output_cap);
        let timeout_sleep = timeout.map(tokio::time::sleep);
        let mut timeout_sleep = timeout_sleep.map(Box::pin);
        let mut timed_out = false;
        loop {
            tokio::select! {
                event = handle.next_event() => {
                    let Some(event) = event else { break };
                    let LocalExecutionProcessEvent::Exited(snapshot) = event else {
                        let LocalExecutionProcessEvent::Output(delta) = event else { unreachable!() };
                        self.handle_delta(key.connection_id, &process_handle, delta, stream_output, &mut stdout, &mut stderr).await;
                        continue;
                    };
                    let exit_code = snapshot.exit_code.unwrap_or(if timed_out { PROCESS_TIMEOUT_EXIT_CODE } else { -1 });
                    let notification = ProcessExitedNotification {
                        process_handle: process_handle.clone(),
                        exit_code,
                        stdout: if stream_output { String::new() } else { stdout.text() },
                        stdout_cap_reached: stdout.cap_reached,
                        stderr: if stream_output { String::new() } else { stderr.text() },
                        stderr_cap_reached: stderr.cap_reached,
                    };
                    self.notify(key.connection_id, ServerNotification::ProcessExited(notification)).await;
                    break;
                }
                _ = async { if let Some(sleep) = timeout_sleep.as_mut() { sleep.await } }, if timeout_sleep.is_some() => {
                    timed_out = true;
                    let _ = self.session(&key).await.map(|session| session.control.terminate());
                    timeout_sleep = None;
                }
            }
        }
        self.sessions.lock().await.remove(&key);
    }

    async fn handle_delta(
        &self,
        connection_id: ConnectionId,
        process_handle: &str,
        delta: ExecutionOutputDelta,
        stream_output: bool,
        stdout: &mut Capture,
        stderr: &mut Capture,
    ) {
        let (stream, capture) = match delta.kind {
            ExecutionOutputKind::Stderr => (ProcessOutputStream::Stderr, stderr),
            ExecutionOutputKind::Stdout | ExecutionOutputKind::Combined => {
                (ProcessOutputStream::Stdout, stdout)
            }
        };
        let Some(bytes) = capture.accept(&delta.raw_bytes) else {
            return;
        };
        if stream_output {
            if !bytes.is_empty() || capture.cap_reached {
                self.notify(
                    connection_id,
                    ServerNotification::ProcessOutputDelta(ProcessOutputDeltaNotification {
                        process_handle: process_handle.to_string(),
                        stream,
                        delta_base64: STANDARD.encode(bytes),
                        cap_reached: capture.cap_reached,
                    }),
                )
                .await;
            }
        }
    }

    async fn notify(&self, connection_id: ConnectionId, notification: ServerNotification) {
        let hook = self.notification_hook.lock().await.clone();
        if let Some(hook) = hook {
            hook(connection_id, JsonRpcNotification::from(notification)).await;
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

fn validate_spawn(params: &ProcessSpawnParams) -> Result<(), JsonRpcError> {
    if params.command.is_empty() {
        return Err(invalid_request("process/spawn command must not be empty"));
    }
    if params.process_handle.trim().is_empty() {
        return Err(invalid_request("processHandle must not be empty"));
    }
    if !std::path::Path::new(&params.cwd).is_absolute() {
        return Err(invalid_params("cwd must be an absolute path"));
    }
    if params.size.is_some() && !params.tty {
        return Err(invalid_params("process/spawn size requires tty: true"));
    }
    if let Some(size) = params.size {
        validate_terminal_size(size)?;
    }
    Ok(())
}

fn validate_terminal_size(
    size: app_server_protocol::protocol::v2::ProcessTerminalSize,
) -> Result<(), JsonRpcError> {
    if size.rows == 0 || size.cols == 0 {
        return Err(invalid_params(
            "process size rows and cols must be greater than 0",
        ));
    }
    Ok(())
}

fn parse_timeout(
    timeout: Option<Option<i64>>,
) -> Result<Option<std::time::Duration>, JsonRpcError> {
    match timeout {
        None => Ok(Some(std::time::Duration::from_secs(10))),
        Some(None) => Ok(None),
        Some(Some(value)) if value >= 0 => Ok(Some(std::time::Duration::from_millis(value as u64))),
        Some(Some(value)) => Err(invalid_params(format!(
            "timeoutMs must be non-negative, got {value}"
        ))),
    }
}

fn key(connection_id: ConnectionId, process_handle: &str) -> ProcessKey {
    ProcessKey {
        connection_id,
        process_handle: process_handle.to_string(),
    }
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
