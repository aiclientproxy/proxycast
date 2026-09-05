use code_mode_protocol::host::{
    ClientHello, ClientToHost, DelegateRequest, DelegateResponse, FramedReader, FramedWriter,
    HostHello, HostRequest, HostResponse, HostToClient, WireResult, MAX_IN_FLIGHT_REQUESTS,
    MAX_PENDING_DELEGATE_CALLS, PROTOCOL_VERSION, SESSION_LIMITS_CAPABILITY,
};
use code_mode_protocol::{
    RuntimeCodeModeCellId, RuntimeCodeModeFuture, RuntimeCodeModeNestedToolCall,
    RuntimeCodeModeSessionDelegate, RuntimeCodeModeSessionHandle, RuntimeCodeModeSessionProvider,
    RuntimeCodeModeWaitOutcome,
};
use code_mode_runtime::V8CodeModeSessionProvider;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::io::{stdin, stdout, AsyncRead, AsyncWrite};
use tokio::sync::{mpsc, oneshot, Semaphore};
use tokio_util::sync::CancellationToken;

const OUTGOING_QUEUE_CAPACITY: usize = 256;

pub async fn run_stdio() -> Result<(), String> {
    run_connection(stdin(), stdout()).await
}

async fn run_connection<R, W>(reader: R, writer: W) -> Result<(), String>
where
    R: AsyncRead + Send + Unpin + 'static,
    W: AsyncWrite + Send + Unpin + 'static,
{
    let mut reader = FramedReader::new(reader);
    let mut writer = FramedWriter::new(writer);
    let hello = reader
        .read::<ClientToHost>()
        .await
        .map_err(|error| format!("failed to read code mode client handshake: {error}"))?
        .ok_or_else(|| "code mode client closed before handshake".to_string())?;
    let ClientToHost::ClientHello(hello) = hello else {
        writer
            .write(&HostToClient::HandshakeRejected {
                reason: "the first message must be connection/hello".to_string(),
            })
            .await
            .map_err(|error| format!("failed to reject code mode handshake: {error}"))?;
        return Err("the first code mode client message was not a handshake".to_string());
    };
    if let Err(reason) = validate_hello(&hello) {
        writer
            .write(&HostToClient::HandshakeRejected {
                reason: reason.clone(),
            })
            .await
            .map_err(|error| format!("failed to reject code mode handshake: {error}"))?;
        return Err(reason);
    }
    V8CodeModeSessionProvider.availability()?;
    writer
        .write(&HostToClient::HostHello(HostHello {
            selected_version: PROTOCOL_VERSION,
            capabilities: vec![SESSION_LIMITS_CAPABILITY.to_string()],
            host_pid: std::process::id(),
        }))
        .await
        .map_err(|error| format!("failed to write code mode host handshake: {error}"))?;

    let (outgoing, mut outgoing_rx) = mpsc::channel(OUTGOING_QUEUE_CAPACITY);
    let state = Arc::new(HostState::new(outgoing.clone()));
    let writer_task = tokio::spawn(async move {
        while let Some(message) = outgoing_rx.recv().await {
            writer
                .write(&message)
                .await
                .map_err(|error| format!("code mode host writer failed: {error}"))?;
        }
        Ok::<(), String>(())
    });

    let read_result = loop {
        match reader.read::<ClientToHost>().await {
            Ok(Some(message)) => {
                if let Err(error) = state.handle_message(message) {
                    break Err(error);
                }
            }
            Ok(None) => break Ok(()),
            Err(error) => break Err(format!("code mode host reader failed: {error}")),
        }
    };
    state.shutdown_all().await;
    writer_task.abort();
    let _ = writer_task.await;
    read_result
}

fn validate_hello(hello: &ClientHello) -> Result<(), String> {
    if !hello.supported_versions.contains(&PROTOCOL_VERSION) {
        return Err(format!(
            "client does not support code mode protocol version {PROTOCOL_VERSION}"
        ));
    }
    let supported = [SESSION_LIMITS_CAPABILITY];
    if let Some(capability) = hello
        .required_capabilities
        .iter()
        .find(|capability| !supported.contains(&capability.as_str()))
    {
        return Err(format!("unsupported required capability {capability}"));
    }
    Ok(())
}

struct HostState {
    outgoing: mpsc::Sender<HostToClient>,
    sessions: Mutex<HashMap<String, RuntimeCodeModeSessionHandle>>,
    operations: Mutex<HashMap<u64, CancellationToken>>,
    pending_delegates: Mutex<HashMap<u64, oneshot::Sender<Result<DelegateResponse, String>>>>,
    cancelled_delegates: Mutex<HashSet<u64>>,
    cell_lifecycle: Mutex<CellLifecycle>,
    next_delegate_id: AtomicU64,
    operation_permits: Arc<Semaphore>,
}

impl HostState {
    fn new(outgoing: mpsc::Sender<HostToClient>) -> Self {
        Self {
            outgoing,
            sessions: Mutex::new(HashMap::new()),
            operations: Mutex::new(HashMap::new()),
            pending_delegates: Mutex::new(HashMap::new()),
            cancelled_delegates: Mutex::new(HashSet::new()),
            cell_lifecycle: Mutex::new(CellLifecycle::default()),
            next_delegate_id: AtomicU64::new(1),
            operation_permits: Arc::new(Semaphore::new(MAX_IN_FLIGHT_REQUESTS)),
        }
    }

    fn handle_message(self: &Arc<Self>, message: ClientToHost) -> Result<(), String> {
        match message {
            ClientToHost::Request { id, request } => self.spawn_request(id, request),
            ClientToHost::CancelRequest { id } => {
                if let Some(token) = self
                    .operations
                    .lock()
                    .expect("code mode host operations poisoned")
                    .get(&id)
                    .cloned()
                {
                    token.cancel();
                }
                Ok(())
            }
            ClientToHost::DelegateResponse { id, result } => {
                let sender = self
                    .pending_delegates
                    .lock()
                    .expect("code mode pending delegates poisoned")
                    .remove(&id);
                let Some(sender) = sender else {
                    if self
                        .cancelled_delegates
                        .lock()
                        .expect("code mode cancelled delegates poisoned")
                        .remove(&id)
                    {
                        return Ok(());
                    }
                    return Err(format!("unexpected code mode delegate response id {id}"));
                };
                let _ = sender.send(result.into_result());
                Ok(())
            }
            ClientToHost::ClientHello(_) => {
                Err("code mode client sent a second handshake".to_string())
            }
        }
    }

    fn spawn_request(self: &Arc<Self>, id: u64, request: HostRequest) -> Result<(), String> {
        let permit = Arc::clone(&self.operation_permits)
            .try_acquire_owned()
            .map_err(|_| {
                format!("code mode host request limit {MAX_IN_FLIGHT_REQUESTS} exceeded")
            })?;
        let cancellation = CancellationToken::new();
        if self
            .operations
            .lock()
            .expect("code mode host operations poisoned")
            .insert(id, cancellation.clone())
            .is_some()
        {
            return Err(format!("duplicate code mode host request id {id}"));
        }
        let state = Arc::clone(self);
        tokio::spawn(async move {
            let _permit = permit;
            match request {
                HostRequest::Execute {
                    session_id,
                    request,
                } => state.execute(id, session_id, request, cancellation).await,
                request => {
                    let result = state.request(request, cancellation).await;
                    let _ = state
                        .outgoing
                        .send(HostToClient::Response {
                            id,
                            result: WireResult::from_result(result),
                        })
                        .await;
                }
            }
            state
                .operations
                .lock()
                .expect("code mode host operations poisoned")
                .remove(&id);
        });
        Ok(())
    }

    async fn request(
        self: &Arc<Self>,
        request: HostRequest,
        cancellation: CancellationToken,
    ) -> Result<HostResponse, String> {
        match request {
            HostRequest::OpenSession {
                session_id,
                cell_execution_limits,
            } => {
                if session_id.trim().is_empty() {
                    return Err("code mode session id must not be empty".to_string());
                }
                if self
                    .sessions
                    .lock()
                    .expect("code mode host sessions poisoned")
                    .contains_key(&session_id)
                {
                    return Err(format!("code mode session {session_id} already exists"));
                }
                let delegate: Arc<dyn RuntimeCodeModeSessionDelegate> =
                    Arc::new(HostDelegate::new(session_id.clone(), Arc::clone(self)));
                let session = V8CodeModeSessionProvider
                    .create_session_with_limits(delegate, cell_execution_limits.unwrap_or_default())
                    .await?;
                self.sessions
                    .lock()
                    .expect("code mode host sessions poisoned")
                    .insert(session_id.clone(), session);
                Ok(HostResponse::SessionReady { session_id })
            }
            HostRequest::Wait {
                session_id,
                request,
            } => {
                let session = self.session(&session_id)?;
                let outcome = tokio::select! {
                    result = session.wait(request) => result?,
                    () = cancellation.cancelled() => return Err("code mode wait cancelled".to_string()),
                };
                Ok(HostResponse::WaitCompleted { outcome })
            }
            HostRequest::Terminate {
                session_id,
                cell_id,
            } => {
                let session = self.session(&session_id)?;
                let outcome = session.terminate(cell_id).await?;
                Ok(HostResponse::WaitCompleted { outcome })
            }
            HostRequest::ShutdownSession { session_id } => {
                let session = self
                    .sessions
                    .lock()
                    .expect("code mode host sessions poisoned")
                    .remove(&session_id)
                    .ok_or_else(|| format!("code mode session {session_id} not found"))?;
                session.shutdown().await?;
                Ok(HostResponse::SessionClosed { session_id })
            }
            HostRequest::Execute { .. } => unreachable!("execute uses the two-phase response"),
        }
    }

    async fn execute(
        self: &Arc<Self>,
        id: u64,
        session_id: String,
        request: code_mode_protocol::host::WireExecuteRequest,
        cancellation: CancellationToken,
    ) {
        let result = async {
            let session = self.session(&session_id)?;
            let mut request = code_mode_protocol::RuntimeCodeModeExecuteRequest::try_from(request)?;
            request.cancellation_token = Some(cancellation.clone());
            let started = tokio::select! {
                result = session.execute(request) => result?,
                () = cancellation.cancelled() => return Err("code mode execution cancelled".to_string()),
            };
            Ok::<_, String>((session, started))
        }
        .await;
        let (session, started) = match result {
            Ok(result) => result,
            Err(message) => {
                let _ = self
                    .outgoing
                    .send(HostToClient::Response {
                        id,
                        result: WireResult::Err { message },
                    })
                    .await;
                return;
            }
        };

        let cell_id = started.cell_id.clone();
        if self
            .outgoing
            .send(HostToClient::Response {
                id,
                result: WireResult::Ok {
                    value: HostResponse::ExecutionStarted {
                        cell_id: cell_id.clone(),
                    },
                },
            })
            .await
            .is_err()
        {
            return;
        }
        let initial = tokio::select! {
            result = started.initial_response() => result,
            () = cancellation.cancelled() => {
                session.terminate(cell_id.clone()).await.map(RuntimeCodeModeWaitOutcome::into_response)
            }
        };
        if self
            .outgoing
            .send(HostToClient::InitialResponse {
                id,
                result: WireResult::from_result(initial),
            })
            .await
            .is_ok()
        {
            self.announce_cell(session_id, cell_id);
        }
    }

    fn announce_cell(&self, session_id: String, cell_id: RuntimeCodeModeCellId) {
        let key = (session_id.clone(), cell_id.clone());
        let send_closed = {
            let mut lifecycle = self
                .cell_lifecycle
                .lock()
                .expect("code mode host cell lifecycle poisoned");
            if lifecycle.pending_closed.remove(&key) {
                true
            } else {
                lifecycle.announced.insert(key);
                false
            }
        };
        if send_closed {
            self.send_cell_closed(session_id, cell_id);
        }
    }

    fn close_cell(&self, session_id: String, cell_id: RuntimeCodeModeCellId) {
        let key = (session_id.clone(), cell_id.clone());
        let send_closed = {
            let mut lifecycle = self
                .cell_lifecycle
                .lock()
                .expect("code mode host cell lifecycle poisoned");
            if lifecycle.announced.remove(&key) {
                true
            } else {
                lifecycle.pending_closed.insert(key);
                false
            }
        };
        if send_closed {
            self.send_cell_closed(session_id, cell_id);
        }
    }

    fn send_cell_closed(&self, session_id: String, cell_id: RuntimeCodeModeCellId) {
        let outgoing = self.outgoing.clone();
        tokio::spawn(async move {
            let _ = outgoing
                .send(HostToClient::CellClosed {
                    session_id,
                    cell_id,
                })
                .await;
        });
    }

    fn session(&self, session_id: &str) -> Result<RuntimeCodeModeSessionHandle, String> {
        self.sessions
            .lock()
            .expect("code mode host sessions poisoned")
            .get(session_id)
            .cloned()
            .ok_or_else(|| format!("code mode session {session_id} not found"))
    }

    async fn delegate(
        &self,
        session_id: String,
        request: DelegateRequest,
        cancellation: CancellationToken,
    ) -> Result<DelegateResponse, String> {
        let id = self.next_delegate_id.fetch_add(1, Ordering::Relaxed);
        let (sender, receiver) = oneshot::channel();
        {
            let mut pending = self
                .pending_delegates
                .lock()
                .expect("code mode pending delegates poisoned");
            if pending.len() >= MAX_PENDING_DELEGATE_CALLS {
                return Err(format!(
                    "code mode delegate request limit {MAX_PENDING_DELEGATE_CALLS} exceeded"
                ));
            }
            pending.insert(id, sender);
        }
        if self
            .outgoing
            .send(HostToClient::DelegateRequest {
                id,
                session_id,
                request,
            })
            .await
            .is_err()
        {
            self.pending_delegates
                .lock()
                .expect("code mode pending delegates poisoned")
                .remove(&id);
            return Err("code mode client writer closed".to_string());
        }
        tokio::select! {
            result = receiver => result.map_err(|_| "code mode delegate response channel closed".to_string())?,
            () = cancellation.cancelled() => {
                if self.pending_delegates
                    .lock()
                    .expect("code mode pending delegates poisoned")
                    .remove(&id)
                    .is_some()
                {
                    self.cancelled_delegates
                        .lock()
                        .expect("code mode cancelled delegates poisoned")
                        .insert(id);
                }
                let _ = self.outgoing.send(HostToClient::CancelDelegateRequest { id }).await;
                Err("code mode delegate request cancelled".to_string())
            }
        }
    }

    async fn shutdown_all(&self) {
        let sessions = std::mem::take(
            &mut *self
                .sessions
                .lock()
                .expect("code mode host sessions poisoned"),
        );
        for session in sessions.into_values() {
            let _ = session.shutdown().await;
        }
        for token in self
            .operations
            .lock()
            .expect("code mode host operations poisoned")
            .drain()
            .map(|(_, token)| token)
        {
            token.cancel();
        }
        *self
            .cell_lifecycle
            .lock()
            .expect("code mode host cell lifecycle poisoned") = CellLifecycle::default();
    }
}

#[derive(Default)]
struct CellLifecycle {
    announced: HashSet<(String, RuntimeCodeModeCellId)>,
    pending_closed: HashSet<(String, RuntimeCodeModeCellId)>,
}

struct HostDelegate {
    session_id: String,
    state: Arc<HostState>,
}

impl HostDelegate {
    fn new(session_id: String, state: Arc<HostState>) -> Self {
        Self { session_id, state }
    }
}

impl RuntimeCodeModeSessionDelegate for HostDelegate {
    fn invoke_tool<'a>(
        &'a self,
        invocation: RuntimeCodeModeNestedToolCall,
        cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, Value> {
        Box::pin(async move {
            match self
                .state
                .delegate(
                    self.session_id.clone(),
                    DelegateRequest::InvokeTool { invocation },
                    cancellation_token,
                )
                .await?
            {
                DelegateResponse::ToolResult { result } => Ok(result),
                DelegateResponse::NotificationDelivered => Err(
                    "code mode client returned a notification response for a tool call".to_string(),
                ),
            }
        })
    }

    fn notify<'a>(
        &'a self,
        tool_call_id: String,
        cell_id: RuntimeCodeModeCellId,
        text: String,
        cancellation_token: CancellationToken,
    ) -> RuntimeCodeModeFuture<'a, ()> {
        Box::pin(async move {
            match self
                .state
                .delegate(
                    self.session_id.clone(),
                    DelegateRequest::Notify {
                        tool_call_id,
                        cell_id,
                        text,
                    },
                    cancellation_token,
                )
                .await?
            {
                DelegateResponse::NotificationDelivered => Ok(()),
                DelegateResponse::ToolResult { .. } => {
                    Err("code mode client returned a tool result for a notification".to_string())
                }
            }
        })
    }

    fn cell_closed(&self, cell_id: &RuntimeCodeModeCellId) {
        self.state
            .close_cell(self.session_id.clone(), cell_id.clone());
    }
}
