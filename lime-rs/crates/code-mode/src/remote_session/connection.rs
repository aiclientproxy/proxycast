use code_mode_protocol::host::{
    ClientHello, ClientToHost, FramedReader, FramedWriter, HostHello, HostRequest, HostResponse,
    HostToClient, WireExecuteRequest, PROTOCOL_VERSION, SESSION_LIMITS_CAPABILITY,
};
use code_mode_protocol::{
    RuntimeCodeModeCellId, RuntimeCodeModeExecuteRequest, RuntimeCodeModeSessionDelegate,
    RuntimeCodeModeSessionLimits, RuntimeCodeModeStartedCell, RuntimeCodeModeWaitOutcome,
    RuntimeCodeModeWaitRequest,
};
use std::path::Path;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

pub(super) mod driver;
mod reader;
use self::driver::{
    register_request, watch_caller_cancellation, ConnectionDriver, ConnectionState, PendingRequest,
};

const HOST_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
const OUTGOING_QUEUE_CAPACITY: usize = 256;

pub(super) struct ProcessConnection {
    outgoing: mpsc::Sender<ClientToHost>,
    state: Arc<ConnectionState>,
    next_request_id: AtomicU64,
    cancellation: CancellationToken,
}

impl ProcessConnection {
    pub(super) async fn spawn(host_path: &Path) -> Result<Self, String> {
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
        let (event_tx, event_rx) = mpsc::channel(OUTGOING_QUEUE_CAPACITY);
        spawn_writer(writer, outgoing_rx, Arc::clone(&state));
        reader::spawn(reader, event_tx);
        tokio::spawn(
            ConnectionDriver::new(
                Arc::clone(&state),
                event_rx,
                outgoing.clone(),
                cancellation.clone(),
            )
            .run(),
        );
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

    pub(super) fn is_alive(&self) -> bool {
        self.state.alive.load(Ordering::Acquire)
    }

    #[cfg(test)]
    pub(super) fn disconnect_for_test(&self) {
        self.state.fail("code mode host disconnected by test");
    }

    #[cfg(test)]
    pub(super) fn pending_request_count_for_test(&self) -> usize {
        self.state.pending_request_count()
    }

    pub(super) async fn open_session(
        &self,
        session_id: String,
        delegate: Arc<dyn RuntimeCodeModeSessionDelegate>,
        limits: RuntimeCodeModeSessionLimits,
    ) -> Result<(), String> {
        let cell_execution_limits =
            (limits != RuntimeCodeModeSessionLimits::default()).then_some(limits);
        let response = self
            .request(HostRequest::OpenSession {
                session_id: session_id.clone(),
                cell_execution_limits,
            })
            .await?;
        match response {
            HostResponse::SessionReady {
                session_id: ready_id,
            } if ready_id == session_id => {
                self.state.insert_session(session_id, delegate);
                Ok(())
            }
            other => self.protocol_error(format!(
                "unexpected code mode open-session response: {other:?}"
            )),
        }
    }

    pub(super) async fn execute(
        &self,
        session_id: String,
        request: RuntimeCodeModeExecuteRequest,
    ) -> Result<RuntimeCodeModeStartedCell, String> {
        let cancellation_token = request.cancellation_token.clone();
        let wire_request = WireExecuteRequest::try_from(request)?;
        let id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let (started_tx, started_rx) = oneshot::channel();
        let (initial_tx, initial_rx) = oneshot::channel();
        register_request(
            &self.state,
            id,
            PendingRequest::Execute {
                session_id: session_id.clone(),
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
            watch_caller_cancellation(state, id, token, outgoing);
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

    pub(super) async fn wait(
        &self,
        session_id: String,
        request: RuntimeCodeModeWaitRequest,
    ) -> Result<RuntimeCodeModeWaitOutcome, String> {
        let expected_cell_id = request.cell_id.clone();
        match self
            .request(HostRequest::Wait {
                session_id,
                request,
            })
            .await?
        {
            HostResponse::WaitCompleted { outcome }
                if wait_outcome_cell_id(&outcome) == &expected_cell_id =>
            {
                Ok(outcome)
            }
            HostResponse::WaitCompleted { outcome } => self.protocol_error(format!(
                "code mode host returned wait outcome for cell {} after waiting for {expected_cell_id}",
                wait_outcome_cell_id(&outcome)
            )),
            other => self.protocol_error(format!("unexpected code mode wait response: {other:?}")),
        }
    }

    pub(super) async fn terminate(
        &self,
        session_id: String,
        cell_id: RuntimeCodeModeCellId,
    ) -> Result<RuntimeCodeModeWaitOutcome, String> {
        let expected_cell_id = cell_id.clone();
        match self
            .request(HostRequest::Terminate {
                session_id,
                cell_id,
            })
            .await?
        {
            HostResponse::WaitCompleted { outcome }
                if wait_outcome_cell_id(&outcome) == &expected_cell_id =>
            {
                Ok(outcome)
            }
            HostResponse::WaitCompleted { outcome } => self.protocol_error(format!(
                "code mode host returned terminate outcome for cell {} after targeting {expected_cell_id}",
                wait_outcome_cell_id(&outcome)
            )),
            other => self.protocol_error(format!(
                "unexpected code mode terminate response: {other:?}"
            )),
        }
    }

    pub(super) async fn shutdown_session(&self, session_id: String) -> Result<(), String> {
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
            other => {
                self.protocol_error(format!("unexpected code mode shutdown response: {other:?}"))
            }
        }
    }

    async fn request(&self, request: HostRequest) -> Result<HostResponse, String> {
        let id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let (sender, receiver) = oneshot::channel();
        register_request(&self.state, id, PendingRequest::Standard(sender))?;
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

    fn protocol_error<T>(&self, error: String) -> Result<T, String> {
        self.state.fail(error.clone());
        Err(error)
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

fn wait_outcome_cell_id(outcome: &RuntimeCodeModeWaitOutcome) -> &RuntimeCodeModeCellId {
    match outcome {
        RuntimeCodeModeWaitOutcome::LiveCell(response)
        | RuntimeCodeModeWaitOutcome::MissingCell(response) => response.cell_id(),
    }
}
