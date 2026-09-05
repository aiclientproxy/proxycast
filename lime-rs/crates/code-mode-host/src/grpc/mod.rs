//! gRPC transport for the stateful Code Mode host.

use code_mode_protocol::grpc::{self as proto, code_mode_host_server::CodeModeHost};
use code_mode_protocol::{
    RuntimeCodeModeCellId, RuntimeCodeModeExecuteRequest, RuntimeCodeModeSessionLimits,
    RuntimeCodeModeSessionProvider,
};
use code_mode_runtime::V8CodeModeSessionProvider;
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, Mutex, Notify, OwnedSemaphorePermit, Semaphore};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status};

mod conversions;
#[cfg(test)]
#[path = "conversions_tests.rs"]
mod conversions_tests;
pub(crate) mod events;
#[cfg(test)]
#[path = "robustness_tests.rs"]
mod robustness_tests;
pub(crate) mod routing;
#[cfg(test)]
#[path = "routing_tests.rs"]
mod routing_tests;
#[cfg(test)]
#[path = "service_tests.rs"]
mod service_tests;
pub(crate) mod session;
mod validation;
mod waits;

use crate::delegate::GrpcDelegate;
use conversions::{duration_ns, execution_outcome, tool_definition};
use session::{GrpcSession, Subscriber};
use validation::{require_identifier, require_uuid, tool_filters};
use waits::WaitControl;

type EventStream = ReceiverStream<Result<proto::SessionEvent, Status>>;
type ToolStream = ReceiverStream<Result<proto::ToolCall, Status>>;
type ExecuteStream = ReceiverStream<Result<proto::ExecuteEvent, Status>>;

const MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;
const MAX_IN_FLIGHT_REQUESTS: usize = 256;
const MAX_ACTIVE_CELLS: usize = 128;
const MAX_IN_FLIGHT_CONTROL_REQUESTS: usize = 256;
const MAX_SUBSCRIPTIONS: usize = 128;
const SESSION_EVENT_CAPACITY: usize =
    code_mode_protocol::host::MAX_PENDING_DELEGATE_CALLS * 2 + MAX_ACTIVE_CELLS;

#[derive(Clone)]
struct HostLimits {
    request_permits: Arc<Semaphore>,
    active_cell_permits: Arc<Semaphore>,
    control_permits: Arc<Semaphore>,
}

impl Default for HostLimits {
    fn default() -> Self {
        Self {
            request_permits: Arc::new(Semaphore::new(MAX_IN_FLIGHT_REQUESTS)),
            active_cell_permits: Arc::new(Semaphore::new(MAX_ACTIVE_CELLS)),
            control_permits: Arc::new(Semaphore::new(MAX_IN_FLIGHT_CONTROL_REQUESTS)),
        }
    }
}

impl HostLimits {
    fn request_permit(&self) -> Result<OwnedSemaphorePermit, Status> {
        self.request_permits
            .clone()
            .try_acquire_owned()
            .map_err(|_| Status::resource_exhausted("code-mode host request limit exceeded"))
    }

    fn active_cell_permit(&self) -> Result<OwnedSemaphorePermit, Status> {
        self.active_cell_permits
            .clone()
            .try_acquire_owned()
            .map_err(|_| Status::resource_exhausted("code-mode host active-cell limit exceeded"))
    }

    fn control_permit(&self) -> Result<OwnedSemaphorePermit, Status> {
        self.control_permits
            .clone()
            .try_acquire_owned()
            .map_err(|_| {
                Status::resource_exhausted("code-mode host control request limit exceeded")
            })
    }
}

pub(crate) async fn run_grpc_address(address: SocketAddr) -> Result<(), String> {
    tonic::transport::Server::builder()
        .add_service(
            proto::code_mode_host_server::CodeModeHostServer::new(GrpcCodeModeHost::new())
                .max_decoding_message_size(MAX_MESSAGE_BYTES)
                .max_encoding_message_size(MAX_MESSAGE_BYTES),
        )
        .serve(address)
        .await
        .map_err(|error| format!("code-mode gRPC host failed: {error}"))
}

#[derive(Clone, Default)]
pub struct GrpcCodeModeHost {
    sessions: Arc<Mutex<HashMap<String, Arc<GrpcSession>>>>,
    limits: HostLimits,
}

impl GrpcCodeModeHost {
    pub fn new() -> Self {
        Self::default()
    }

    async fn session(&self, id: &str) -> Result<Arc<GrpcSession>, Status> {
        self.sessions
            .lock()
            .await
            .get(id)
            .cloned()
            .ok_or_else(|| Status::not_found(format!("code-mode session {id} not found")))
    }

    async fn close_lease(&self, id: &str, expected: &Arc<GrpcSession>) {
        let session = {
            let mut sessions = self.sessions.lock().await;
            if sessions
                .get(id)
                .is_some_and(|session| Arc::ptr_eq(session, expected))
            {
                sessions.remove(id)
            } else {
                None
            }
        };
        if let Some(session) = session {
            session
                .close_pending("code-mode session lease closed")
                .await;
            if let Some(runtime) = session.runtime.lock().await.take() {
                let _ = runtime.shutdown().await;
            }
            session.clear_execution_state().await;
        }
    }
}

#[tonic::async_trait]
impl CodeModeHost for GrpcCodeModeHost {
    type OpenSessionStream = EventStream;
    type SubscribeToToolCallsStream = ToolStream;
    type ExecuteStream = ExecuteStream;

    async fn open_session(
        &self,
        request: Request<proto::OpenSessionRequest>,
    ) -> Result<Response<Self::OpenSessionStream>, Status> {
        let _request_permit = self.limits.request_permit()?;
        let (sender, receiver) = mpsc::channel(SESSION_EVENT_CAPACITY);
        let id = uuid::Uuid::new_v4().to_string();
        let session = Arc::new(GrpcSession {
            id: id.clone(),
            peer: crate::peer::PeerState::new(),
            closed: std::sync::atomic::AtomicBool::new(false),
            runtime: Mutex::new(None),
            subscribers: Mutex::new(Vec::new()),
            session_events: Mutex::new(Some(sender.clone())),
            event_shutdown: CancellationToken::new(),
            pending: Mutex::new(HashMap::new()),
            pending_notifications: Mutex::new(HashMap::new()),
            next_subscriber: AtomicU64::new(0),
            waits: Mutex::new(HashMap::new()),
            cancelled_waits: Mutex::new(HashSet::new()),
            pending_executions: Mutex::new(HashSet::new()),
            seen_executions: Mutex::new(Default::default()),
            execution_ids: Mutex::new(HashMap::new()),
            tool_call_sequences: Mutex::new(HashMap::new()),
            active_cells: Mutex::new(HashMap::new()),
        });
        let limits = match request.into_inner().cell_execution_limits {
            Some(limits) => RuntimeCodeModeSessionLimits {
                max_yield_time_ms: limits.max_yield_time_ms,
                max_heap_size_bytes: limits
                    .max_heap_size_bytes
                    .map(usize::try_from)
                    .transpose()
                    .map_err(|_| {
                        Status::invalid_argument("maximum heap size exceeds this platform")
                    })?,
            },
            None => RuntimeCodeModeSessionLimits::default(),
        };
        let runtime = V8CodeModeSessionProvider
            .create_session_with_limits(
                Arc::new(GrpcDelegate {
                    session: Arc::downgrade(&session),
                }),
                limits,
            )
            .await
            .map_err(Status::failed_precondition)?;
        *session.runtime.lock().await = Some(runtime);
        self.sessions
            .lock()
            .await
            .insert(id.clone(), Arc::clone(&session));
        sender
            .send(Ok(proto::SessionEvent {
                event: Some(proto::session_event::Event::Opened(proto::SessionOpened {
                    session_id: id.clone(),
                })),
            }))
            .await
            .map_err(|_| Status::internal("failed to publish code-mode session opening"))?;
        let host = self.clone();
        let session_weak = Arc::downgrade(&session);
        let event_shutdown = session.event_shutdown.clone();
        tokio::spawn(async move {
            tokio::select! {
                _ = sender.closed() => {}
                _ = event_shutdown.cancelled() => {}
            }
            if let Some(session) = session_weak.upgrade() {
                host.close_lease(&id, &session).await;
            }
        });
        Ok(Response::new(ReceiverStream::new(receiver)))
    }

    async fn close_session(
        &self,
        request: Request<proto::CloseSessionRequest>,
    ) -> Result<Response<proto::CloseSessionResponse>, Status> {
        let _control_permit = self.limits.control_permit()?;
        let id = request.into_inner().session_id;
        require_uuid(&id, "session_id")?;
        let session = self
            .sessions
            .lock()
            .await
            .remove(&id)
            .ok_or_else(|| Status::not_found("code-mode session not found"))?;
        session.close_pending("code-mode session closed").await;
        if let Some(runtime) = session.runtime.lock().await.take() {
            runtime.shutdown().await.map_err(Status::internal)?;
        }
        session.clear_execution_state().await;
        Ok(Response::new(proto::CloseSessionResponse {}))
    }

    async fn subscribe_to_tool_calls(
        &self,
        request: Request<proto::SubscribeToToolCallsRequest>,
    ) -> Result<Response<Self::SubscribeToToolCallsStream>, Status> {
        let _request_permit = self.limits.request_permit()?;
        require_uuid(&request.get_ref().session_id, "session_id")?;
        tool_filters(&request.get_ref().tool_names)?;
        let session = self.session(&request.get_ref().session_id).await?;
        let (sender, receiver) = mpsc::channel(64);
        let mut subscribers = session.subscribers.lock().await;
        if subscribers.len() >= MAX_SUBSCRIPTIONS {
            return Err(Status::resource_exhausted(
                "code-mode session subscription limit exceeded",
            ));
        }
        subscribers.push(Subscriber {
            tool_names: request.into_inner().tool_names,
            sender,
        });
        Ok(Response::new(ReceiverStream::new(receiver)))
    }

    async fn complete_tool_call(
        &self,
        request: Request<proto::CompleteToolCallRequest>,
    ) -> Result<Response<proto::CompleteToolCallResponse>, Status> {
        let _control_permit = self.limits.control_permit()?;
        let request = request.into_inner();
        require_uuid(&request.session_id, "session_id")?;
        require_uuid(&request.invocation_id, "invocation_id")?;
        let session = self.session(&request.session_id).await?;
        let result = match request.outcome {
            Some(proto::complete_tool_call_request::Outcome::Succeeded(value)) => {
                serde_json::from_slice(&value.output_json).map_err(|error| {
                    Status::invalid_argument(format!("invalid tool output JSON: {error}"))
                })?
            }
            Some(proto::complete_tool_call_request::Outcome::Failed(value)) => {
                let sender = session
                    .pending
                    .lock()
                    .await
                    .remove(&request.invocation_id)
                    .ok_or_else(|| Status::not_found("code-mode invocation not found"))?;
                sender
                    .send(Err(value.message))
                    .map_err(|_| Status::aborted("code-mode invocation already completed"))?;
                return Ok(Response::new(proto::CompleteToolCallResponse {}));
            }
            None => {
                return Err(Status::invalid_argument(
                    "tool completion outcome is required",
                ))
            }
        };
        let sender = session
            .pending
            .lock()
            .await
            .remove(&request.invocation_id)
            .ok_or_else(|| Status::not_found("code-mode invocation not found"))?;
        sender
            .send(Ok(result))
            .map_err(|_| Status::aborted("code-mode invocation already completed"))?;
        Ok(Response::new(proto::CompleteToolCallResponse {}))
    }

    async fn acknowledge_notification(
        &self,
        request: Request<proto::AcknowledgeNotificationRequest>,
    ) -> Result<Response<proto::AcknowledgeNotificationResponse>, Status> {
        let _control_permit = self.limits.control_permit()?;
        let request = request.into_inner();
        require_uuid(&request.session_id, "session_id")?;
        require_uuid(&request.notification_id, "notification_id")?;
        let session = self.session(&request.session_id).await?;
        session
            .acknowledge_notification(&request.notification_id)
            .await?;
        Ok(Response::new(proto::AcknowledgeNotificationResponse {}))
    }

    async fn execute(
        &self,
        request: Request<proto::ExecuteRequest>,
    ) -> Result<Response<Self::ExecuteStream>, Status> {
        let request_permit = self.limits.request_permit()?;
        let active_cell_permit = self.limits.active_cell_permit()?;
        let received_at = Instant::now();
        let request = request.into_inner();
        require_uuid(&request.session_id, "session_id")?;
        let session = self.session(&request.session_id).await?;
        require_uuid(&request.execution_id, "execution_id")?;
        require_identifier(&request.tool_call_id, "tool_call_id")?;
        let enabled_tools = request
            .enabled_tools
            .into_iter()
            .map(tool_definition)
            .collect::<Result<Vec<_>, _>>()?;
        let execute = RuntimeCodeModeExecuteRequest {
            tool_call_id: request.tool_call_id,
            source: request.source,
            enabled_tools,
            yield_time_ms: request.yield_time_ms,
            max_output_tokens: request
                .max_output_tokens
                .map(usize::try_from)
                .transpose()
                .map_err(|_| {
                    Status::invalid_argument("maximum output tokens exceeds this platform")
                })?,
            cancellation_token: None,
        };
        let runtime = session.runtime().await?;
        session.reserve_execution(&request.execution_id).await?;
        let started = match runtime.execute(execute).await {
            Ok(started) => started,
            Err(error) => {
                session.abandon_execution(&request.execution_id).await;
                return Err(Status::failed_precondition(error));
            }
        };
        let cell_id = started.cell_id.clone();
        if let Err(error) = session
            .register_execution(
                &cell_id.to_string(),
                &request.execution_id,
                active_cell_permit,
            )
            .await
        {
            session.abandon_execution(&request.execution_id).await;
            let _ = runtime.terminate(cell_id).await;
            return Err(Status::cancelled(error));
        }
        let (sender, receiver) = mpsc::channel(2);
        sender
            .send(Ok(proto::ExecuteEvent {
                event: Some(proto::execute_event::Event::Started(
                    proto::ExecutionStarted {
                        execution_id: request.execution_id,
                        cell_id: cell_id.to_string(),
                    },
                )),
            }))
            .await
            .map_err(|_| Status::internal("failed to publish code-mode execution start"))?;
        tokio::spawn(async move {
            let _request_permit = request_permit;
            tokio::select! {
                response = started.initial_response() => {
                    let event = match response {
                        Ok(response) => Ok(proto::ExecuteEvent {
                            event: Some(proto::execute_event::Event::Outcome(execution_outcome(
                                response,
                                duration_ns(received_at.elapsed()),
                            ))),
                        }),
                        Err(error) => Err(Status::internal(error)),
                    };
                    let _ = sender.send(event).await;
                }
                _ = sender.closed() => {
                    let _ = runtime.terminate(cell_id).await;
                }
            }
        });
        Ok(Response::new(ReceiverStream::new(receiver)))
    }

    async fn wait(
        &self,
        request: Request<proto::WaitRequest>,
    ) -> Result<Response<proto::WaitResponse>, Status> {
        let _request_permit = self.limits.request_permit()?;
        let received_at = Instant::now();
        let request = request.into_inner();
        require_uuid(&request.session_id, "session_id")?;
        let session = self.session(&request.session_id).await?;
        require_uuid(&request.wait_id, "wait_id")?;
        require_identifier(&request.cell_id, "cell_id")?;
        let cancellation = CancellationToken::new();
        let retired = Arc::new(Notify::new());
        let mut waits = session.waits.lock().await;
        if session.closed.load(std::sync::atomic::Ordering::Acquire) {
            return Err(Status::cancelled("code-mode session is closed"));
        }
        if session
            .cancelled_waits
            .lock()
            .await
            .remove(&request.wait_id)
        {
            return Err(Status::cancelled("code-mode wait cancelled"));
        }
        if waits
            .insert(
                request.wait_id.clone(),
                WaitControl {
                    cancellation: cancellation.clone(),
                    retired: Arc::clone(&retired),
                },
            )
            .is_some()
        {
            return Err(Status::already_exists(
                "code-mode wait_id is already active",
            ));
        }
        drop(waits);
        let wait_id = request.wait_id.clone();
        let cell_id = request.cell_id.clone();
        let yield_time_ms = request.yield_time_ms;
        let runtime = match session.runtime().await {
            Ok(runtime) => runtime,
            Err(error) => {
                if let Some(control) = session.waits.lock().await.remove(&wait_id) {
                    control.retired.notify_waiters();
                }
                return Err(error);
            }
        };
        let result = tokio::select! {
            outcome = async move {
                runtime.wait(code_mode_protocol::RuntimeCodeModeWaitRequest {
                    cell_id: RuntimeCodeModeCellId::new(cell_id),
                    yield_time_ms,
                }).await.map_err(Status::failed_precondition)
            } => outcome,
            _ = cancellation.cancelled() => Err(Status::cancelled("code-mode wait cancelled")),
        };
        if let Some(control) = session.waits.lock().await.remove(&wait_id) {
            control.retired.notify_waiters();
        }
        let outcome = result?;
        let duration_ns = duration_ns(received_at.elapsed());
        let state = match outcome {
            code_mode_protocol::RuntimeCodeModeWaitOutcome::LiveCell(response) => {
                proto::wait_response::State::LiveCell(execution_outcome(response, duration_ns))
            }
            code_mode_protocol::RuntimeCodeModeWaitOutcome::MissingCell(response) => {
                proto::wait_response::State::MissingCell(execution_outcome(response, duration_ns))
            }
        };
        Ok(Response::new(proto::WaitResponse { state: Some(state) }))
    }

    async fn cancel_wait(
        &self,
        request: Request<proto::CancelWaitRequest>,
    ) -> Result<Response<proto::CancelWaitResponse>, Status> {
        let _control_permit = self.limits.control_permit()?;
        let request = request.into_inner();
        require_uuid(&request.session_id, "session_id")?;
        require_uuid(&request.wait_id, "wait_id")?;
        let session = self.session(&request.session_id).await?;
        let control = session.waits.lock().await.get(&request.wait_id).cloned();
        let Some(control) = control else {
            session.cancelled_waits.lock().await.insert(request.wait_id);
            return Ok(Response::new(proto::CancelWaitResponse {}));
        };
        let retired = control.retired.notified();
        control.cancellation.cancel();
        retired.await;
        Ok(Response::new(proto::CancelWaitResponse {}))
    }

    async fn terminate(
        &self,
        request: Request<proto::TerminateRequest>,
    ) -> Result<Response<proto::WaitResponse>, Status> {
        let _request_permit = self.limits.request_permit()?;
        let received_at = Instant::now();
        let request = request.into_inner();
        require_uuid(&request.session_id, "session_id")?;
        require_identifier(&request.cell_id, "cell_id")?;
        let session = self.session(&request.session_id).await?;
        let response = session
            .runtime()
            .await?
            .terminate(RuntimeCodeModeCellId::new(request.cell_id))
            .await
            .map_err(Status::failed_precondition)?
            .into_response();
        Ok(Response::new(proto::WaitResponse {
            state: Some(proto::wait_response::State::LiveCell(execution_outcome(
                response,
                duration_ns(received_at.elapsed()),
            ))),
        }))
    }
}
