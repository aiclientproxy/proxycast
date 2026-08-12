use super::client_state::{ConnectionState, PendingRequest};
use super::default_code_mode_host_path;
use super::protocol::{
    ClientHello, ClientToHost, FramedReader, FramedWriter, HostHello, HostRequest, HostResponse,
    HostToClient, WireExecuteRequest, PROTOCOL_VERSION, SESSION_LIMITS_CAPABILITY,
};
use crate::code_mode::{
    RuntimeCodeModeCellId, RuntimeCodeModeExecuteRequest, RuntimeCodeModeFuture,
    RuntimeCodeModeSession, RuntimeCodeModeSessionDelegate, RuntimeCodeModeSessionHandle,
    RuntimeCodeModeSessionLimits, RuntimeCodeModeSessionProvider,
    RuntimeCodeModeSessionProviderFuture, RuntimeCodeModeStartedCell, RuntimeCodeModeWaitOutcome,
    RuntimeCodeModeWaitRequest,
};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot, Mutex as AsyncMutex, Semaphore};
use tokio_util::sync::CancellationToken;

const HOST_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
const OUTGOING_QUEUE_CAPACITY: usize = 256;

pub struct ProcessCodeModeSessionProvider {
    host_path: PathBuf,
    connection: AsyncMutex<Option<Arc<ProcessConnection>>>,
    connect_permit: Semaphore,
    next_session_id: AtomicU64,
}

impl ProcessCodeModeSessionProvider {
    pub fn with_host_path(host_path: PathBuf) -> Self {
        Self {
            host_path,
            connection: AsyncMutex::new(None),
            connect_permit: Semaphore::new(1),
            next_session_id: AtomicU64::new(1),
        }
    }

    async fn connection(&self) -> Result<Arc<ProcessConnection>, String> {
        if let Some(connection) = self.live_connection().await {
            return Ok(connection);
        }

        let _permit = self
            .connect_permit
            .acquire()
            .await
            .map_err(|_| "code mode host connection coordinator closed".to_string())?;
        if let Some(connection) = self.live_connection().await {
            return Ok(connection);
        }

        let connection = Arc::new(ProcessConnection::spawn(&self.host_path).await?);
        self.connection
            .lock()
            .await
            .replace(Arc::clone(&connection));
        Ok(connection)
    }

    async fn live_connection(&self) -> Option<Arc<ProcessConnection>> {
        self.connection
            .lock()
            .await
            .as_ref()
            .filter(|connection| connection.is_alive())
            .cloned()
    }
}

impl Default for ProcessCodeModeSessionProvider {
    fn default() -> Self {
        Self::with_host_path(default_code_mode_host_path())
    }
}

impl RuntimeCodeModeSessionProvider for ProcessCodeModeSessionProvider {
    fn availability(&self) -> Result<(), String> {
        if self.host_path.is_file() {
            Ok(())
        } else {
            Err(format!(
                "code mode host executable was not found: {}",
                self.host_path.display()
            ))
        }
    }

    fn create_session<'a>(
        &'a self,
        delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
    ) -> RuntimeCodeModeSessionProviderFuture<'a> {
        self.create_session_with_limits(delegate, RuntimeCodeModeSessionLimits::default())
    }

    fn create_session_with_limits<'a>(
        &'a self,
        delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
        limits: RuntimeCodeModeSessionLimits,
    ) -> RuntimeCodeModeSessionProviderFuture<'a> {
        Box::pin(async move {
            self.availability()?;
            let connection = self.connection().await?;
            let session_id = format!(
                "session-{}",
                self.next_session_id.fetch_add(1, Ordering::Relaxed)
            );
            connection
                .open_session(session_id.clone(), Arc::clone(&delegate), limits)
                .await?;
            Ok(RuntimeCodeModeSessionHandle::new(Arc::new(
                ProcessCodeModeSession {
                    connection,
                    session_id,
                    closed: AtomicBool::new(false),
                },
            )))
        })
    }
}

struct ProcessCodeModeSession {
    connection: Arc<ProcessConnection>,
    session_id: String,
    closed: AtomicBool,
}

impl RuntimeCodeModeSession for ProcessCodeModeSession {
    fn execute<'a>(
        &'a self,
        request: RuntimeCodeModeExecuteRequest,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeStartedCell> {
        Box::pin(async move {
            if self.closed.load(Ordering::Acquire) {
                return Err("code mode session is closed".to_string());
            }
            self.connection
                .execute(self.session_id.clone(), request)
                .await
        })
    }

    fn wait<'a>(
        &'a self,
        request: RuntimeCodeModeWaitRequest,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeWaitOutcome> {
        Box::pin(async move {
            if self.closed.load(Ordering::Acquire) {
                return Err("code mode session is closed".to_string());
            }
            self.connection.wait(self.session_id.clone(), request).await
        })
    }

    fn terminate<'a>(
        &'a self,
        cell_id: RuntimeCodeModeCellId,
    ) -> RuntimeCodeModeFuture<'a, RuntimeCodeModeWaitOutcome> {
        Box::pin(async move {
            if self.closed.load(Ordering::Acquire) {
                return Err("code mode session is closed".to_string());
            }
            self.connection
                .terminate(self.session_id.clone(), cell_id)
                .await
        })
    }

    fn shutdown(&self) -> RuntimeCodeModeFuture<'_, ()> {
        Box::pin(async move {
            if self.closed.swap(true, Ordering::AcqRel) {
                return Ok(());
            }
            self.connection
                .shutdown_session(self.session_id.clone())
                .await
        })
    }
}

impl Drop for ProcessCodeModeSession {
    fn drop(&mut self) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        let connection = Arc::clone(&self.connection);
        let session_id = self.session_id.clone();
        if tokio::runtime::Handle::try_current().is_ok() {
            tokio::spawn(async move {
                let _ = connection.shutdown_session(session_id).await;
            });
        }
    }
}

struct ProcessConnection {
    outgoing: mpsc::Sender<ClientToHost>,
    state: Arc<ConnectionState>,
    next_request_id: AtomicU64,
    cancellation: CancellationToken,
}

impl ProcessConnection {
    async fn spawn(host_path: &Path) -> Result<Self, String> {
        let mut child = Command::new(host_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|error| {
                format!(
                    "failed to spawn code mode host {}: {error}",
                    host_path.display()
                )
            })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "code mode host stdin was unavailable".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "code mode host stdout was unavailable".to_string())?;
        let stderr = child.stderr.take();
        let cancellation = CancellationToken::new();
        let state = Arc::new(ConnectionState::new(cancellation.clone()));

        let mut reader = FramedReader::new(stdout);
        let mut writer = FramedWriter::new(stdin);
        writer
            .write(&ClientToHost::ClientHello(ClientHello {
                supported_versions: vec![PROTOCOL_VERSION],
                required_capabilities: vec![SESSION_LIMITS_CAPABILITY.to_string()],
                optional_capabilities: Vec::new(),
            }))
            .await
            .map_err(|error| format!("failed to write code mode host handshake: {error}"))?;
        let hello = tokio::time::timeout(HOST_HANDSHAKE_TIMEOUT, reader.read::<HostToClient>())
            .await
            .map_err(|_| "code mode host handshake timed out".to_string())?
            .map_err(|error| format!("failed to read code mode host handshake: {error}"))?
            .ok_or_else(|| "code mode host exited during handshake".to_string())?;
        let HostToClient::HostHello(HostHello {
            selected_version,
            capabilities,
            host_pid,
        }) = hello
        else {
            return Err(match hello {
                HostToClient::HandshakeRejected { reason } => {
                    format!("code mode host rejected handshake: {reason}")
                }
                _ => "code mode host returned an invalid handshake response".to_string(),
            });
        };
        if selected_version != PROTOCOL_VERSION {
            return Err(format!(
                "code mode host selected unsupported protocol version {selected_version}"
            ));
        }
        if !capabilities
            .iter()
            .any(|capability| capability == SESSION_LIMITS_CAPABILITY)
        {
            return Err(format!(
                "code mode host omitted required capability {SESSION_LIMITS_CAPABILITY}"
            ));
        }
        tracing::debug!(host_pid, "connected to standalone code mode host");

        let (outgoing, outgoing_rx) = mpsc::channel(OUTGOING_QUEUE_CAPACITY);
        spawn_writer(writer, outgoing_rx, Arc::clone(&state));
        spawn_reader(reader, outgoing.clone(), Arc::clone(&state));
        spawn_child_supervisor(child, Arc::clone(&state));
        if let Some(stderr) = stderr {
            spawn_stderr_reader(stderr);
        }

        Ok(Self {
            outgoing,
            state,
            next_request_id: AtomicU64::new(1),
            cancellation,
        })
    }

    fn is_alive(&self) -> bool {
        self.state.alive.load(Ordering::Acquire)
    }

    async fn open_session(
        &self,
        session_id: String,
        delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
        limits: RuntimeCodeModeSessionLimits,
    ) -> Result<(), String> {
        let response = self
            .request(HostRequest::OpenSession {
                session_id: session_id.clone(),
                cell_execution_limits: Some(limits),
            })
            .await?;
        match response {
            HostResponse::SessionReady {
                session_id: ready_id,
            } if ready_id == session_id => {
                self.state.insert_session(session_id, delegate);
                Ok(())
            }
            other => Err(format!(
                "unexpected code mode open-session response: {other:?}"
            )),
        }
    }

    async fn execute(
        &self,
        session_id: String,
        request: RuntimeCodeModeExecuteRequest,
    ) -> Result<RuntimeCodeModeStartedCell, String> {
        let cancellation_token = request.cancellation_token.clone();
        let wire_request = WireExecuteRequest::try_from(request)?;
        let id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let (started_tx, started_rx) = oneshot::channel();
        let (initial_tx, initial_rx) = oneshot::channel();
        self.state.register_pending(
            id,
            PendingRequest::Execute {
                started: started_tx,
                initial: initial_tx,
            },
        )?;
        if let Err(error) = self
            .send(ClientToHost::Request {
                id,
                request: HostRequest::Execute {
                    session_id,
                    request: wire_request,
                },
            })
            .await
        {
            self.state.remove_pending(id);
            return Err(error);
        }
        if let Some(token) = cancellation_token {
            let outgoing = self.outgoing.clone();
            let state = Arc::clone(&self.state);
            let finished = CancellationToken::new();
            state.register_caller_cancellation_watcher(id, finished.clone());
            tokio::spawn(async move {
                tokio::select! {
                    () = token.cancelled() => {
                        if state.has_pending(id) {
                            let _ = outgoing.send(ClientToHost::CancelRequest { id }).await;
                        }
                    }
                    () = finished.cancelled() => {}
                }
            });
        }
        let started = started_rx
            .await
            .map_err(|_| self.state.failure_message())??;
        let HostResponse::ExecutionStarted { cell_id } = started else {
            return Err(format!(
                "unexpected code mode execute response: {started:?}"
            ));
        };
        Ok(RuntimeCodeModeStartedCell::from_result_receiver(
            cell_id, initial_rx,
        ))
    }

    async fn wait(
        &self,
        session_id: String,
        request: RuntimeCodeModeWaitRequest,
    ) -> Result<RuntimeCodeModeWaitOutcome, String> {
        match self
            .request(HostRequest::Wait {
                session_id,
                request,
            })
            .await?
        {
            HostResponse::WaitCompleted { outcome } => Ok(outcome),
            other => Err(format!("unexpected code mode wait response: {other:?}")),
        }
    }

    async fn terminate(
        &self,
        session_id: String,
        cell_id: RuntimeCodeModeCellId,
    ) -> Result<RuntimeCodeModeWaitOutcome, String> {
        match self
            .request(HostRequest::Terminate {
                session_id,
                cell_id,
            })
            .await?
        {
            HostResponse::WaitCompleted { outcome } => Ok(outcome),
            other => Err(format!(
                "unexpected code mode terminate response: {other:?}"
            )),
        }
    }

    async fn shutdown_session(&self, session_id: String) -> Result<(), String> {
        let response = self
            .request(HostRequest::ShutdownSession {
                session_id: session_id.clone(),
            })
            .await;
        self.state.remove_session(&session_id);
        match response? {
            HostResponse::SessionClosed {
                session_id: closed_id,
            } if closed_id == session_id => Ok(()),
            other => Err(format!("unexpected code mode shutdown response: {other:?}")),
        }
    }

    async fn request(&self, request: HostRequest) -> Result<HostResponse, String> {
        let id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let (sender, receiver) = oneshot::channel();
        self.state
            .register_pending(id, PendingRequest::Standard(sender))?;
        if let Err(error) = self.send(ClientToHost::Request { id, request }).await {
            self.state.remove_pending(id);
            return Err(error);
        }
        receiver.await.map_err(|_| self.state.failure_message())?
    }

    async fn send(&self, message: ClientToHost) -> Result<(), String> {
        if !self.is_alive() {
            return Err(self.state.failure_message());
        }
        self.outgoing
            .send(message)
            .await
            .map_err(|_| self.state.failure_message())
    }
}

impl Drop for ProcessConnection {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

fn spawn_writer<W>(
    mut writer: FramedWriter<W>,
    mut outgoing: mpsc::Receiver<ClientToHost>,
    state: Arc<ConnectionState>,
) where
    W: tokio::io::AsyncWrite + Send + Unpin + 'static,
{
    tokio::spawn(async move {
        while let Some(message) = outgoing.recv().await {
            if let Err(error) = writer.write(&message).await {
                state.fail(format!("code mode host writer failed: {error}"));
                return;
            }
        }
        state.fail("code mode host writer stopped".to_string());
    });
}

fn spawn_reader<R>(
    mut reader: FramedReader<R>,
    outgoing: mpsc::Sender<ClientToHost>,
    state: Arc<ConnectionState>,
) where
    R: tokio::io::AsyncRead + Send + Unpin + 'static,
{
    tokio::spawn(async move {
        loop {
            match reader.read::<HostToClient>().await {
                Ok(Some(message)) => {
                    if let Err(error) = state.handle_host_message(message, &outgoing) {
                        state.fail(error);
                        return;
                    }
                }
                Ok(None) => {
                    state.fail("code mode host closed stdout".to_string());
                    return;
                }
                Err(error) => {
                    state.fail(format!("code mode host reader failed: {error}"));
                    return;
                }
            }
        }
    });
}

fn spawn_child_supervisor(mut child: Child, state: Arc<ConnectionState>) {
    tokio::spawn(async move {
        let result = tokio::select! {
            status = child.wait() => status.map_err(|error| error.to_string()),
            () = state.cancellation.cancelled() => {
                let _ = child.kill().await;
                child.wait().await.map_err(|error| error.to_string())
            }
        };
        let message = match result {
            Ok(status) => format!("code mode host exited with {status}"),
            Err(error) => format!("failed to wait for code mode host: {error}"),
        };
        state.fail(message);
    });
}

fn spawn_stderr_reader<R>(stderr: R)
where
    R: tokio::io::AsyncRead + Send + Unpin + 'static,
{
    tokio::spawn(async move {
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            tracing::warn!(target: "code_mode_host", "{line}");
        }
    });
}
