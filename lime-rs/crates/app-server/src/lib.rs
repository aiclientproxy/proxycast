mod agent_identity_store;
mod agent_mailbox_store;
mod agent_ui_event_schema;
mod agent_ui_sequence_verifier;
mod approval_server_request;
mod automation_execution;
mod capability;
mod execution_process;
mod external_backend;
mod file_checkpoint;
mod file_checkpoint_snapshot;
mod gateway_tunnel;
mod knowledge_builder_runtime;
mod local_data_source;
mod mcp_elicitation;
mod media_runtime_contract;
mod media_task;
mod media_task_payload;
mod media_task_worker;
mod memory_store;
mod model_route_assembly;
mod model_route_execution;
mod model_task_contract;
mod otel_trace;
mod plugin_packages;
mod processor;
mod project_shell;
mod runtime;
mod runtime_backend;
mod runtime_factory;
mod server_request;
mod skill_registry;
mod thread_listener;
#[cfg(test)]
mod thread_listener_tests;
mod thread_state;
mod trace_context;

pub use app_server_protocol::error_codes;
use app_server_protocol::AgentEvent;
pub use app_server_protocol::AgentInput;
pub use app_server_protocol::AgentSession;
pub use app_server_protocol::AgentSessionActionRespondParams;
pub use app_server_protocol::AgentSessionActionRespondResponse;
pub use app_server_protocol::AgentSessionActionScope;
pub use app_server_protocol::AgentSessionActionType;
pub use app_server_protocol::AgentSessionAnalysisHandoffExportParams;
pub use app_server_protocol::AgentSessionAnalysisHandoffExportResponse;
pub use app_server_protocol::AgentSessionApprovalDecision;
pub use app_server_protocol::AgentSessionHandoffArtifact;
pub use app_server_protocol::AgentSessionHandoffBundleExportParams;
pub use app_server_protocol::AgentSessionHandoffBundleExportResponse;
pub use app_server_protocol::AgentSessionMediaReadParams;
pub use app_server_protocol::AgentSessionMediaReadResponse;
pub use app_server_protocol::AgentSessionReplayCaseExportParams;
pub use app_server_protocol::AgentSessionReplayCaseExportResponse;
pub use app_server_protocol::AgentSessionReviewDecision;
pub use app_server_protocol::AgentSessionReviewDecisionSaveParams;
pub use app_server_protocol::AgentSessionReviewDecisionTemplateExportParams;
pub use app_server_protocol::AgentSessionReviewDecisionTemplateExportResponse;
pub use app_server_protocol::AgentSessionStartParams;
pub use app_server_protocol::AgentSessionStatus;
pub use app_server_protocol::AgentSessionTurnStartParams;
pub use app_server_protocol::AgentTurn;
pub use app_server_protocol::AgentTurnStatus;
pub use app_server_protocol::ArtifactContentStatus;
pub use app_server_protocol::ArtifactReadParams;
pub use app_server_protocol::ArtifactReadResponse;
pub use app_server_protocol::ArtifactSummary;
pub use app_server_protocol::CapabilityDescriptor;
pub use app_server_protocol::CapabilityListParams;
pub use app_server_protocol::CapabilityListResponse;
pub use app_server_protocol::EvidenceExportParams;
pub use app_server_protocol::EvidenceExportResponse;
pub use app_server_protocol::EvidencePackArtifact;
pub use app_server_protocol::EvidencePackSummary;
use app_server_protocol::InitializeParams;
use app_server_protocol::JsonRpcError;
pub use app_server_protocol::JsonRpcMessage;
pub use app_server_protocol::JsonRpcNotification;
pub use app_server_protocol::JsonRpcRequest;
pub use app_server_protocol::RequestId;
pub use app_server_protocol::RuntimeOptions;
pub use app_server_protocol::WorkflowReadParams;
pub use app_server_protocol::WorkflowReadResponse;
pub use app_server_protocol::METHOD_AGENT_SESSION_ACTION_RESPOND;
pub use app_server_protocol::METHOD_AGENT_SESSION_ANALYSIS_HANDOFF_EXPORT;
pub use app_server_protocol::METHOD_AGENT_SESSION_EVENT;
pub use app_server_protocol::METHOD_AGENT_SESSION_HANDOFF_BUNDLE_EXPORT;
pub use app_server_protocol::METHOD_AGENT_SESSION_MEDIA_READ;
pub use app_server_protocol::METHOD_AGENT_SESSION_REPLAY_CASE_EXPORT;
pub use app_server_protocol::METHOD_AGENT_SESSION_REVIEW_DECISION_SAVE;
pub use app_server_protocol::METHOD_AGENT_SESSION_REVIEW_DECISION_TEMPLATE_EXPORT;
pub use app_server_protocol::METHOD_ARTIFACT_READ;
pub use app_server_protocol::METHOD_CAPABILITY_LIST;
pub use app_server_protocol::METHOD_EVIDENCE_EXPORT;
pub use app_server_protocol::METHOD_INITIALIZE;
pub use app_server_protocol::METHOD_INITIALIZED;
pub use app_server_protocol::METHOD_THREAD_READ;
pub use app_server_protocol::METHOD_THREAD_START;
pub use app_server_protocol::METHOD_TURN_INTERRUPT;
pub use app_server_protocol::METHOD_TURN_START;
pub use app_server_protocol::METHOD_WORKFLOW_READ;
use app_server_transport::acquire_app_server_startup_lock;
use app_server_transport::app_server_startup_lock_path;
use app_server_transport::decode_message;
use app_server_transport::encode_message;
use app_server_transport::start_control_socket_acceptor;
use app_server_transport::start_stdio_connection;
use app_server_transport::start_websocket_acceptor;
use app_server_transport::AppServerTransport;
use app_server_transport::ConnectionId;
use app_server_transport::OutgoingMessage;
use app_server_transport::QueuedOutgoingMessage;
use app_server_transport::TransportError;
use app_server_transport::TransportEvent;
use app_server_transport::APP_SERVER_CONTROL_SOCKET_DIR_NAME;
pub use capability::capability_source_from_app_policy_json;
pub use capability::AppPolicyCapability;
pub use capability::AppPolicyLoadError;
pub use capability::AppPolicyManifest;
pub use capability::AppPolicyManifestError;
pub use capability::CapabilityInventoryRecord;
pub use capability::CapabilityInventorySource;
pub use capability::CapabilityListContext;
pub use capability::CapabilitySource;
pub use external_backend::ExternalBackend;
pub use external_backend::ExternalBackendConfig;
pub use external_backend::DEFAULT_EXTERNAL_BACKEND_TIMEOUT_MS;
pub use knowledge_builder_runtime::KnowledgeBuilderRuntimeExecutor;
pub use knowledge_builder_runtime::NativeKnowledgeBuilderRuntimeExecutor;
pub use local_data_source::LocalAppDataSource;
pub use memory_store::LocalMemoryBackend;
pub use memory_store::MemoryBackend;
pub use memory_store::RolloutSummaryWriteParams;
pub use otel_trace::init_app_server_otel_from_env;
pub use otel_trace::AppServerOtelGuard;
use processor::RequestProcessor;
pub(crate) use runtime::export_trace_events_from_store_to_path;
pub(crate) use runtime::summarize_trace_event_store;
pub use runtime::ActionRespondRequest;
pub use runtime::AppDataSource;
pub use runtime::ArtifactContentProvider;
pub use runtime::ArtifactContentRequest;
pub use runtime::AutomationManagementAppDataSource;
pub use runtime::AutomationOverviewAppDataSource;
pub use runtime::BasicEvidenceExportProvider;
pub use runtime::CancelExecutionRequest;
pub use runtime::ConnectAppDataSource;
pub use runtime::DiagnosticsAppDataSource;
pub use runtime::EventLogRecord;
pub use runtime::EventLogWriter;
pub use runtime::EvidenceExportProvider;
pub use runtime::EvidencePackRequest;
pub use runtime::ExecutionBackend;
pub use runtime::ExecutionRequest;
pub use runtime::FileCheckpointSnapshotReadRequest;
pub use runtime::FileCheckpointSnapshotRecord;
pub use runtime::FileCheckpointSnapshotSaveRequest;
pub use runtime::FileCheckpointSnapshotStore;
pub use runtime::FilesystemArtifactContentProvider;
pub use runtime::FilesystemFileCheckpointSnapshotStore;
pub use runtime::FilesystemOutputSnapshotStore;
pub use runtime::GatewayAppDataSource;
pub use runtime::InlineArtifactContentProvider;
pub use runtime::KnowledgeAppDataSource;
pub use runtime::McpAppDataSource;
pub use runtime::MediaAppDataSource;
pub use runtime::MemoryAppDataSource;
pub use runtime::MockBackend;
pub use runtime::ModelCatalogQuery;
pub use runtime::ModelProviderAppDataSource;
pub use runtime::NoopAppDataSource;
pub use runtime::NoopEvidenceExportProvider;
pub use runtime::NoopFileCheckpointSnapshotStore;
pub use runtime::NoopOutputSnapshotStore;
pub use runtime::OutputSnapshotReadRequest;
pub use runtime::OutputSnapshotRecord;
pub use runtime::OutputSnapshotSaveRequest;
pub use runtime::OutputSnapshotStore;
pub use runtime::PluginDataSource;
pub use runtime::ProjectionRepair;
pub use runtime::ProjectionStore;
pub use runtime::ProviderModelCatalog;
pub use runtime::RightSurfaceAppDataSource;
pub use runtime::RuntimeCore;
pub use runtime::RuntimeCoreError;
pub use runtime::RuntimeCoreEventAppender;
pub use runtime::RuntimeCoreOutput;
pub use runtime::RuntimeEvent;
pub use runtime::RuntimeEventHub;
pub use runtime::RuntimeEventSink;
pub use runtime::RuntimeHostContext;
pub use runtime::SessionAppDataSource;
pub use runtime::SidecarRef;
pub use runtime::SidecarStore;
pub use runtime::SidecarWriteRequest;
pub use runtime::SkillAppDataSource;
pub use runtime::StorageRoots;
pub use runtime::ToolInventoryReadRequest;
pub use runtime::TraceEventWriter;
pub use runtime::UnavailableBackend;
pub use runtime::UsageStatsAppDataSource;
pub use runtime::VoiceAppDataSource;
pub use runtime::WorkspaceAppDataSource;
pub use runtime::WorkspaceObjectCanvasReplayReadiness;
pub use runtime::WorkspaceObjectCanvasReplayReadinessListParams;
pub use runtime::WorkspaceObjectCanvasSnapshot;
pub use runtime::WorkspaceObjectCanvasSnapshotListParams;
pub use runtime::WorkspaceSkillBindingAppDataSource;
pub(crate) use runtime::TRACE_EVENT_MAX_FILES_PER_SESSION;
pub use runtime_backend::RuntimeBackend;
pub use runtime_factory::AppServerBackendMode;
pub use runtime_factory::AppServerRuntimeFactory;
pub use runtime_factory::UnsupportedBackendMode;
pub use server_request::ServerRequestError;
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;
use thiserror::Error;
use tokio::io;
use tokio::io::AsyncRead;
use tokio::io::AsyncWrite;
use tokio::sync::broadcast;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

const OUTBOUND_MESSAGE_CAPACITY: usize = 1024;
const DEFAULT_THREAD_UNLOADING_DELAY: Duration = Duration::from_secs(30 * 60);
const THREAD_UNLOAD_ACTIVITY_POLL_INTERVAL: Duration = Duration::from_millis(250);
type TransportWriter = mpsc::Sender<QueuedOutgoingMessage>;
type TransportWriters = Arc<Mutex<HashMap<ConnectionId, TransportWriter>>>;
type TransportDisconnects = Arc<Mutex<HashMap<ConnectionId, CancellationToken>>>;
type TransportInitialized = Arc<Mutex<HashSet<ConnectionId>>>;
type TransportNotificationOptOut = Arc<Mutex<HashMap<ConnectionId, HashSet<String>>>>;
type StreamedTransportMessage = Result<(ConnectionId, JsonRpcMessage), AppServerError>;

struct ServerRequestClient {
    connection_id: ConnectionId,
    writer: TransportWriter,
    disconnect_sender: Option<CancellationToken>,
}

#[derive(Debug, Error)]
pub enum AppServerError {
    #[error(transparent)]
    Transport(#[from] TransportError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("app-server transport connection {connection_id} is not open")]
    ConnectionUnavailable { connection_id: ConnectionId },
    #[error("app-server transport connection {connection_id} writer is closed")]
    ConnectionWriterClosed { connection_id: ConnectionId },
    #[error(transparent)]
    ServerRequest(#[from] ServerRequestError),
    #[error("app-server thread listener failed: {0}")]
    ThreadListener(String),
    #[error(
        "app-server transport is disabled; select stdio://, unix://PATH, or ws://127.0.0.1:PORT"
    )]
    TransportDisabled,
}

#[derive(Clone)]
pub struct AppServer {
    processor: RequestProcessor,
    thread_states: thread_state::ThreadStateManager,
    runtime_event_receiver: Arc<Mutex<Option<mpsc::UnboundedReceiver<AgentEvent>>>>,
    runtime_event_pump_started: Arc<AtomicBool>,
    outbound_messages: broadcast::Sender<JsonRpcMessage>,
    transport_writers: TransportWriters,
    transport_disconnects: TransportDisconnects,
    transport_initialized: TransportInitialized,
    transport_notification_opt_out: TransportNotificationOptOut,
    thread_unloading_delay: Duration,
    server_requests: server_request::ServerRequestRouter,
    mcp_elicitation_requests: mcp_elicitation::ElicitationRequestSource,
}

#[derive(Clone)]
pub struct AppServerEventBridge {
    runtime_events: RuntimeCoreEventAppender,
    thread_states: thread_state::ThreadStateManager,
    outbound_messages: broadcast::Sender<JsonRpcMessage>,
    transport_writers: TransportWriters,
    transport_disconnects: TransportDisconnects,
    transport_initialized: TransportInitialized,
    transport_notification_opt_out: TransportNotificationOptOut,
}

impl AppServer {
    #[cfg(test)]
    pub fn new() -> Self {
        Self::with_runtime(RuntimeCore::default())
    }

    pub fn with_runtime(runtime: RuntimeCore) -> Self {
        let (outbound_messages, _) = broadcast::channel(OUTBOUND_MESSAGE_CAPACITY);
        let runtime_event_receiver = Arc::new(Mutex::new(runtime.take_event_receiver()));
        let thread_states = thread_state::ThreadStateManager::new();
        let transport_writers = Arc::new(Mutex::new(HashMap::new()));
        let transport_disconnects = Arc::new(Mutex::new(HashMap::new()));
        let transport_initialized = Arc::new(Mutex::new(HashSet::new()));
        let transport_notification_opt_out = Arc::new(Mutex::new(HashMap::new()));
        let server_requests = server_request::ServerRequestRouter::default();
        let interrupt_bridge = AppServerEventBridge {
            runtime_events: runtime.event_appender(),
            thread_states: thread_states.clone(),
            outbound_messages: outbound_messages.clone(),
            transport_writers: transport_writers.clone(),
            transport_disconnects: transport_disconnects.clone(),
            transport_initialized: transport_initialized.clone(),
            transport_notification_opt_out: transport_notification_opt_out.clone(),
        };
        let interrupt_router = server_requests.clone();
        let turn_interrupt_hook: processor::TurnInterruptHook =
            Arc::new(move |thread_id, turn_id| {
                let bridge = interrupt_bridge.clone();
                let router = interrupt_router.clone();
                Box::pin(async move {
                    router
                        .abort_for_thread_turn(&bridge, &thread_id, &turn_id, "turn interrupted")
                        .await;
                })
            });
        Self {
            processor: RequestProcessor::new_with_thread_states(runtime, thread_states.clone())
                .with_turn_interrupt_hook(turn_interrupt_hook),
            thread_states,
            runtime_event_receiver,
            runtime_event_pump_started: Arc::new(AtomicBool::new(false)),
            outbound_messages,
            transport_writers,
            transport_disconnects,
            transport_initialized,
            transport_notification_opt_out,
            thread_unloading_delay: DEFAULT_THREAD_UNLOADING_DELAY,
            server_requests,
            mcp_elicitation_requests: mcp_elicitation::ElicitationRequestSource::default(),
        }
    }

    pub fn with_mcp_elicitation_router(
        mut self,
        router: lime_mcp::ElicitationRequestRouter,
    ) -> Result<Self, lime_mcp::ElicitationRouterError> {
        self.mcp_elicitation_requests =
            mcp_elicitation::ElicitationRequestSource::subscribe(router)?;
        Ok(self)
    }

    pub fn with_thread_unloading_delay(mut self, delay: Duration) -> Self {
        self.thread_unloading_delay = delay;
        self
    }

    pub fn subscribe_outbound_messages(&self) -> broadcast::Receiver<JsonRpcMessage> {
        self.outbound_messages.subscribe()
    }

    fn ensure_runtime_event_pump(&self) {
        if self.runtime_event_pump_started.swap(true, Ordering::AcqRel) {
            return;
        }
        let receiver = self
            .runtime_event_receiver
            .lock()
            .expect("app-server runtime event receiver mutex poisoned")
            .take();
        let Some(mut receiver) = receiver else {
            return;
        };
        let bridge = self.event_bridge();
        let server = self.clone();
        tokio::spawn(async move {
            while let Some(event) = receiver.recv().await {
                let Some(thread_id) = event
                    .thread_id
                    .as_deref()
                    .map(str::trim)
                    .filter(|thread_id| !thread_id.is_empty())
                    .map(agent_protocol::ThreadId::new)
                else {
                    tracing::error!(
                        event_id = %event.event_id,
                        event_type = %event.event_type,
                        "dropping background runtime event without canonical thread id"
                    );
                    continue;
                };
                let approval_event = (event.event_type == "action.required").then(|| event.clone());
                let (completion_tx, completion_rx) = if approval_event.is_some() {
                    let (sender, receiver) = oneshot::channel();
                    (Some(sender), Some(receiver))
                } else {
                    (None, None)
                };
                if let Err(error) = bridge
                    .send_thread_command(
                        thread_id,
                        thread_state::ThreadListenerCommand::PublishRuntimeEvent {
                            event,
                            completion_tx,
                        },
                    )
                    .await
                {
                    tracing::error!(%error, "failed to enqueue background runtime event");
                    continue;
                }
                if let (Some(event), Some(completion_rx)) = (approval_event, completion_rx) {
                    match completion_rx.await {
                        Ok(Ok(_)) => {}
                        Ok(Err(error)) => {
                            tracing::warn!(%error, "action.required projection failed");
                            continue;
                        }
                        Err(error) => {
                            tracing::warn!(%error, "action.required projection did not complete");
                            continue;
                        }
                    }
                    let server = server.clone();
                    tokio::spawn(async move {
                        server.handle_command_approval_request(event).await;
                    });
                }
            }
        });
    }

    pub fn event_bridge(&self) -> AppServerEventBridge {
        AppServerEventBridge {
            runtime_events: self.processor.runtime().event_appender(),
            thread_states: self.thread_states.clone(),
            outbound_messages: self.outbound_messages.clone(),
            transport_writers: self.transport_writers.clone(),
            transport_disconnects: self.transport_disconnects.clone(),
            transport_initialized: self.transport_initialized.clone(),
            transport_notification_opt_out: self.transport_notification_opt_out.clone(),
        }
    }

    fn schedule_thread_unload(&self, thread_id: agent_protocol::ThreadId) {
        let server = self.clone();
        tokio::spawn(async move {
            let Some(generation) = server
                .thread_states
                .schedule_unload_if_idle(&thread_id)
                .await
            else {
                return;
            };
            let poll_interval = server
                .thread_unloading_delay
                .min(THREAD_UNLOAD_ACTIVITY_POLL_INTERVAL)
                .max(Duration::from_millis(1));
            let mut idle_since = None;

            loop {
                if !server
                    .thread_states
                    .unload_ticket_is_current(&thread_id, generation)
                    .await
                {
                    return;
                }
                match server
                    .processor
                    .runtime()
                    .loaded_thread_is_active(thread_id.as_str())
                {
                    None => {
                        server.thread_states.remove_thread(&thread_id).await;
                        return;
                    }
                    Some(true) => idle_since = None,
                    Some(false) => {
                        let since = idle_since.get_or_insert_with(tokio::time::Instant::now);
                        if since.elapsed() >= server.thread_unloading_delay {
                            break;
                        }
                    }
                }
                tokio::time::sleep(poll_interval).await;
            }

            if !server
                .thread_states
                .begin_unload(&thread_id, generation)
                .await
            {
                return;
            }
            match server
                .processor
                .runtime()
                .unload_thread_if_idle(thread_id.as_str())
                .await
            {
                Ok(runtime::ThreadUnloadResult::Unloaded) => {
                    let bridge = server.event_bridge();
                    server
                        .server_requests
                        .abort_for_threads(&bridge, &[thread_id.to_string()], "thread closed")
                        .await;
                    server.thread_states.remove_thread(&thread_id).await;
                    let status: JsonRpcNotification =
                        app_server_protocol::protocol::v2::ServerNotification::ThreadStatusChanged(
                            app_server_protocol::protocol::v2::ThreadStatusChangedNotification {
                                thread_id: thread_id.to_string(),
                                status: app_server_protocol::protocol::v2::ThreadStatus::NotLoaded,
                            },
                        )
                        .into();
                    bridge
                        .broadcast_message(JsonRpcMessage::Notification(status))
                        .await;
                    let closed: JsonRpcNotification =
                        app_server_protocol::protocol::v2::ServerNotification::ThreadClosed(
                            app_server_protocol::protocol::v2::ThreadClosedNotification {
                                thread_id: thread_id.to_string(),
                            },
                        )
                        .into();
                    bridge
                        .broadcast_message(JsonRpcMessage::Notification(closed))
                        .await;
                }
                Ok(runtime::ThreadUnloadResult::NotLoaded) => {
                    server.thread_states.remove_thread(&thread_id).await;
                }
                Ok(runtime::ThreadUnloadResult::Active) => {
                    server
                        .thread_states
                        .cancel_unload(&thread_id, generation)
                        .await;
                    server.schedule_thread_unload(thread_id);
                }
                Err(error) => {
                    server
                        .thread_states
                        .cancel_unload(&thread_id, generation)
                        .await;
                    tracing::warn!(%thread_id, %error, "failed to unload idle thread");
                }
            }
        });
    }

    #[cfg(test)]
    fn runtime(&self) -> &RuntimeCore {
        self.processor.runtime()
    }

    pub async fn handle_json_line(&self, line: &str) -> Result<Vec<String>, AppServerError> {
        let message = decode_message(line)?;
        self.handle_message(message)
            .await?
            .iter()
            .map(|message| encode_message(message).map_err(AppServerError::from))
            .collect()
    }

    pub async fn handle_message(
        &self,
        message: JsonRpcMessage,
    ) -> Result<Vec<JsonRpcMessage>, AppServerError> {
        self.ensure_runtime_event_pump();
        match message {
            JsonRpcMessage::Request(request) => self.processor.handle_request(request).await,
            JsonRpcMessage::Notification(notification) => {
                self.processor.handle_notification(notification);
                Ok(Vec::new())
            }
            JsonRpcMessage::Response(_response) => {
                #[cfg(test)]
                self.resolve_test_server_request_response(_response.id, _response.result);
                Ok(Vec::new())
            }
            JsonRpcMessage::Error(_response) => {
                #[cfg(test)]
                self.resolve_test_server_request_error(_response.id, _response.error);
                Ok(Vec::new())
            }
        }
    }

    async fn handle_transport_message(
        &self,
        connection_id: ConnectionId,
        message: JsonRpcMessage,
    ) -> Result<Vec<JsonRpcMessage>, AppServerError> {
        self.ensure_runtime_event_pump();
        match message {
            JsonRpcMessage::Request(request) => {
                self.processor
                    .handle_transport_request(connection_id, request)
                    .await
            }
            JsonRpcMessage::Notification(notification) => {
                self.processor.handle_notification(notification);
                Ok(Vec::new())
            }
            JsonRpcMessage::Response(response) => {
                self.resolve_transport_server_request_response(
                    connection_id,
                    response.id,
                    response.result,
                );
                Ok(Vec::new())
            }
            JsonRpcMessage::Error(response) => {
                self.resolve_transport_server_request_error(
                    connection_id,
                    response.id,
                    response.error,
                );
                Ok(Vec::new())
            }
        }
    }

    pub async fn handle_message_streaming(
        &self,
        message: JsonRpcMessage,
        event_callback: &mut (dyn FnMut(JsonRpcMessage) + Send),
    ) -> Result<Vec<JsonRpcMessage>, AppServerError> {
        self.ensure_runtime_event_pump();
        match message {
            JsonRpcMessage::Request(request) => {
                self.processor
                    .handle_request_streaming(request, event_callback)
                    .await
            }
            JsonRpcMessage::Notification(notification) => {
                self.processor.handle_notification(notification);
                Ok(Vec::new())
            }
            JsonRpcMessage::Response(_response) => {
                #[cfg(test)]
                self.resolve_test_server_request_response(_response.id, _response.result);
                Ok(Vec::new())
            }
            JsonRpcMessage::Error(_response) => {
                #[cfg(test)]
                self.resolve_test_server_request_error(_response.id, _response.error);
                Ok(Vec::new())
            }
        }
    }

    async fn handle_transport_message_streaming(
        &self,
        connection_id: ConnectionId,
        message: JsonRpcMessage,
        event_callback: &mut (dyn FnMut(JsonRpcMessage) + Send),
    ) -> Result<Vec<JsonRpcMessage>, AppServerError> {
        self.ensure_runtime_event_pump();
        match message {
            JsonRpcMessage::Request(request) => {
                self.processor
                    .handle_transport_request_streaming(connection_id, request, event_callback)
                    .await
            }
            JsonRpcMessage::Notification(notification) => {
                self.processor.handle_notification(notification);
                Ok(Vec::new())
            }
            JsonRpcMessage::Response(response) => {
                self.resolve_transport_server_request_response(
                    connection_id,
                    response.id,
                    response.result,
                );
                Ok(Vec::new())
            }
            JsonRpcMessage::Error(response) => {
                self.resolve_transport_server_request_error(
                    connection_id,
                    response.id,
                    response.error,
                );
                Ok(Vec::new())
            }
        }
    }

    pub async fn append_external_runtime_events(
        &self,
        session_id: &str,
        turn_id: Option<&str>,
        runtime_events: Vec<RuntimeEvent>,
    ) -> Result<Vec<JsonRpcMessage>, JsonRpcError> {
        self.event_bridge()
            .append_external_runtime_events(session_id, turn_id, runtime_events)
            .await
    }

    fn register_transport_writer(
        &self,
        connection_id: ConnectionId,
        writer: TransportWriter,
        disconnect_sender: Option<CancellationToken>,
    ) {
        self.transport_writers
            .lock()
            .expect("app-server transport writer mutex poisoned")
            .insert(connection_id, writer);
        let mut disconnects = self
            .transport_disconnects
            .lock()
            .expect("app-server transport disconnect mutex poisoned");
        if let Some(disconnect_sender) = disconnect_sender {
            disconnects.insert(connection_id, disconnect_sender);
        } else {
            disconnects.remove(&connection_id);
        }
    }

    fn unregister_transport_writer(&self, connection_id: ConnectionId) {
        self.transport_writers
            .lock()
            .expect("app-server transport writer mutex poisoned")
            .remove(&connection_id);
        self.transport_disconnects
            .lock()
            .expect("app-server transport disconnect mutex poisoned")
            .remove(&connection_id);
        self.transport_initialized
            .lock()
            .expect("app-server transport initialization mutex poisoned")
            .remove(&connection_id);
        self.transport_notification_opt_out
            .lock()
            .expect("app-server transport notification opt-out mutex poisoned")
            .remove(&connection_id);
    }

    async fn clear_transport_writers(&self) {
        let connection_ids = self
            .transport_writers
            .lock()
            .expect("app-server transport writer mutex poisoned")
            .keys()
            .copied()
            .collect::<Vec<_>>();
        self.transport_writers
            .lock()
            .expect("app-server transport writer mutex poisoned")
            .clear();
        self.transport_disconnects
            .lock()
            .expect("app-server transport disconnect mutex poisoned")
            .clear();
        self.transport_initialized
            .lock()
            .expect("app-server transport initialization mutex poisoned")
            .clear();
        self.transport_notification_opt_out
            .lock()
            .expect("app-server transport notification opt-out mutex poisoned")
            .clear();
        for connection_id in connection_ids {
            self.thread_states
                .disconnect_connection(connection_id)
                .await;
        }
    }

    fn transport_writer(&self, connection_id: ConnectionId) -> Option<TransportWriter> {
        self.transport_writers
            .lock()
            .expect("app-server transport writer mutex poisoned")
            .get(&connection_id)
            .cloned()
    }

    fn transport_disconnect_sender(
        &self,
        connection_id: ConnectionId,
    ) -> Option<CancellationToken> {
        self.transport_disconnects
            .lock()
            .expect("app-server transport disconnect mutex poisoned")
            .get(&connection_id)
            .cloned()
    }

    fn mark_transport_initialized(&self, connection_id: ConnectionId) {
        self.transport_initialized
            .lock()
            .expect("app-server transport initialization mutex poisoned")
            .insert(connection_id);
    }

    fn set_transport_notification_opt_out(
        &self,
        connection_id: ConnectionId,
        methods: Option<Vec<String>>,
    ) {
        let methods = methods
            .unwrap_or_default()
            .into_iter()
            .map(|method| method.trim().to_string())
            .filter(|method| !method.is_empty())
            .collect();
        self.transport_notification_opt_out
            .lock()
            .expect("app-server transport notification opt-out mutex poisoned")
            .insert(connection_id, methods);
    }

    async fn send_to_transport_connection(
        &self,
        connection_id: ConnectionId,
        message: JsonRpcMessage,
    ) -> Result<(), AppServerError> {
        let writer = self
            .transport_writer(connection_id)
            .ok_or(AppServerError::ConnectionUnavailable { connection_id })?;
        let queued = QueuedOutgoingMessage::new(OutgoingMessage::from(message));
        if let Some(disconnect_sender) = self.transport_disconnect_sender(connection_id) {
            match writer.try_send(queued) {
                Ok(()) => Ok(()),
                Err(mpsc::error::TrySendError::Full(_)) => {
                    disconnect_sender.cancel();
                    Ok(())
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    disconnect_sender.cancel();
                    Err(AppServerError::ConnectionWriterClosed { connection_id })
                }
            }
        } else {
            writer
                .send(queued)
                .await
                .map_err(|_| AppServerError::ConnectionWriterClosed { connection_id })
        }
    }

    pub async fn send_server_request(
        &self,
        method: impl Into<String>,
        params: impl serde::Serialize,
    ) -> Result<serde_json::Value, AppServerError> {
        self.begin_server_request(method, params)
            .await?
            .wait()
            .await
            .map_err(AppServerError::from)
    }

    fn server_request_client(&self) -> Result<ServerRequestClient, AppServerError> {
        let writers = self
            .transport_writers
            .lock()
            .expect("app-server transport writer mutex poisoned")
            .iter()
            .map(|(connection_id, writer)| (*connection_id, writer.clone()))
            .collect::<Vec<_>>();
        let client_count = writers.len();
        if client_count == 0 {
            return Err(ServerRequestError::ClientUnavailable.into());
        }
        if client_count != 1 {
            return Err(ServerRequestError::ClientAmbiguous { client_count }.into());
        }

        let (connection_id, writer) = writers
            .into_iter()
            .next()
            .expect("one transport writer was counted");
        Ok(ServerRequestClient {
            connection_id,
            writer,
            disconnect_sender: self.transport_disconnect_sender(connection_id),
        })
    }

    async fn send_to_server_request_client(
        &self,
        client: ServerRequestClient,
        message: JsonRpcMessage,
    ) -> Result<(), AppServerError> {
        let queued = QueuedOutgoingMessage::new(OutgoingMessage::from(message));
        if let Some(disconnect_sender) = client.disconnect_sender {
            match client.writer.try_send(queued) {
                Ok(()) => Ok(()),
                Err(mpsc::error::TrySendError::Full(_)) => {
                    disconnect_sender.cancel();
                    Ok(())
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    disconnect_sender.cancel();
                    Err(AppServerError::ConnectionWriterClosed {
                        connection_id: client.connection_id,
                    })
                }
            }
        } else {
            client
                .writer
                .send(queued)
                .await
                .map_err(|_| AppServerError::ConnectionWriterClosed {
                    connection_id: client.connection_id,
                })
        }
    }

    fn cancel_pending_server_requests(&self, reason: &str) {
        self.server_requests.cancel_all(reason);
    }

    #[cfg(test)]
    fn resolve_test_server_request_response(&self, id: RequestId, result: serde_json::Value) {
        let request_id = id.clone();
        if let Err(error) = self.server_requests.resolve_response(id, result) {
            tracing::warn!(%request_id, %error, "dropping unmatched App Server test response");
        }
    }

    #[cfg(test)]
    fn resolve_test_server_request_error(&self, id: RequestId, response: JsonRpcError) {
        let request_id = id.clone();
        if let Err(error) = self.server_requests.resolve_error(id, response) {
            tracing::warn!(%request_id, %error, "dropping unmatched App Server test error");
        }
    }

    fn resolve_transport_server_request_response(
        &self,
        connection_id: ConnectionId,
        id: RequestId,
        result: serde_json::Value,
    ) {
        let request_id = id.clone();
        if let Err(error) =
            self.server_requests
                .resolve_transport_response(connection_id, id, result)
        {
            tracing::warn!(%request_id, %connection_id, %error, "dropping unmatched App Server transport response");
        }
    }

    fn resolve_transport_server_request_error(
        &self,
        connection_id: ConnectionId,
        id: RequestId,
        response: JsonRpcError,
    ) {
        let request_id = id.clone();
        if let Err(error) =
            self.server_requests
                .resolve_transport_error(connection_id, id, response)
        {
            tracing::warn!(%request_id, %connection_id, %error, "dropping unmatched App Server transport error");
        }
    }
}

pub fn spawn_image_task_worker_scheduler(
    db: lime_core::database::DbConnection,
    sidecar_store: Option<Arc<SidecarStore>>,
) -> tokio::task::JoinHandle<()> {
    media_task_worker::spawn_image_task_worker_scheduler(
        media_task_worker::ImageTaskWorkerContext::new(db).with_sidecar_store(sidecar_store),
    )
}

impl AppServerEventBridge {
    async fn broadcast_message(&self, message: JsonRpcMessage) {
        let _ = self.outbound_messages.send(message.clone());
        let connection_ids = self
            .transport_writers
            .lock()
            .expect("app-server transport writer mutex poisoned")
            .keys()
            .copied()
            .collect::<Vec<_>>();
        for connection_id in connection_ids {
            if let Err(error) = self
                .send_messages_to_connection(connection_id, std::slice::from_ref(&message))
                .await
            {
                tracing::warn!(
                    %connection_id,
                    %error,
                    "failed to broadcast App Server message"
                );
            }
        }
    }

    async fn send_messages_to_connection(
        &self,
        connection_id: ConnectionId,
        messages: &[JsonRpcMessage],
    ) -> Result<(), String> {
        let writer = self
            .transport_writers
            .lock()
            .expect("app-server transport writer mutex poisoned")
            .get(&connection_id)
            .cloned()
            .ok_or_else(|| format!("transport connection {connection_id} is not open"))?;
        if !self
            .transport_initialized
            .lock()
            .expect("app-server transport initialization mutex poisoned")
            .contains(&connection_id)
        {
            return Err(format!(
                "transport connection {connection_id} is not initialized"
            ));
        }
        let disconnect_sender = self
            .transport_disconnects
            .lock()
            .expect("app-server transport disconnect mutex poisoned")
            .get(&connection_id)
            .cloned();

        for message in messages {
            if notification_is_opted_out(
                &self.transport_notification_opt_out,
                connection_id,
                message,
            ) {
                continue;
            }
            let queued = QueuedOutgoingMessage::new(OutgoingMessage::from(message.clone()));
            if let Some(disconnect_sender) = disconnect_sender.as_ref() {
                match writer.try_send(queued) {
                    Ok(()) => {}
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        disconnect_sender.cancel();
                        return Err(format!(
                            "transport connection {connection_id} outbound queue is full"
                        ));
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        disconnect_sender.cancel();
                        return Err(format!(
                            "transport connection {connection_id} writer is closed"
                        ));
                    }
                }
            } else {
                writer.send(queued).await.map_err(|_| {
                    format!("transport connection {connection_id} writer is closed")
                })?;
            }
        }
        Ok(())
    }
}

pub async fn run_stdio(server: AppServer) -> Result<(), AppServerError> {
    run_json_lines(server, io::stdin(), io::stdout()).await
}

fn startup_lock_path_for_socket(socket_path: &Path) -> std::path::PathBuf {
    let socket_parent = socket_path.parent().unwrap_or_else(|| Path::new("."));
    let base_dir = if socket_parent.file_name()
        == Some(std::ffi::OsStr::new(APP_SERVER_CONTROL_SOCKET_DIR_NAME))
    {
        socket_parent.parent().unwrap_or(socket_parent)
    } else {
        socket_parent
    };
    app_server_startup_lock_path(base_dir)
}

/// Runs the App Server over the selected Codex-compatible transport.
///
/// Network transports share the same JSON-RPC event loop as stdio. This keeps
/// connection lifecycle, reverse requests, streaming responses, and backpressure
/// in one owner instead of maintaining a second server implementation.
pub async fn run_transport(
    server: AppServer,
    transport: AppServerTransport,
) -> Result<(), AppServerError> {
    let (transport_event_tx, transport_event_rx) = mpsc::channel(OUTBOUND_MESSAGE_CAPACITY);
    let shutdown_token = CancellationToken::new();
    let mut accept_handles = Vec::<JoinHandle<()>>::new();
    let single_client_mode = matches!(transport, AppServerTransport::Stdio);
    let _unix_socket_startup_lock = match &transport {
        AppServerTransport::UnixSocket { socket_path } => {
            Some(acquire_app_server_startup_lock(startup_lock_path_for_socket(socket_path)).await?)
        }
        _ => None,
    };

    match transport {
        AppServerTransport::Stdio => {
            accept_handles.extend(
                start_stdio_connection(transport_event_tx.clone(), io::stdin(), io::stdout())
                    .await?,
            );
        }
        AppServerTransport::UnixSocket { socket_path } => {
            accept_handles.push(
                start_control_socket_acceptor(
                    socket_path,
                    transport_event_tx.clone(),
                    shutdown_token.clone(),
                )
                .await?,
            );
        }
        AppServerTransport::WebSocket { bind_address } => {
            accept_handles.push(
                start_websocket_acceptor(
                    bind_address,
                    transport_event_tx.clone(),
                    shutdown_token.clone(),
                )
                .await?,
            );
        }
        AppServerTransport::Off => return Err(AppServerError::TransportDisabled),
    }
    drop(_unix_socket_startup_lock);
    drop(transport_event_tx);

    let result = run_transport_events(
        server,
        transport_event_rx,
        single_client_mode,
        shutdown_token.clone(),
    )
    .await;
    shutdown_token.cancel();
    for handle in accept_handles {
        let _ = handle.await;
    }
    result
}

pub async fn run_json_lines<R, W>(
    server: AppServer,
    reader: R,
    writer: W,
) -> Result<(), AppServerError>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    let (transport_event_tx, transport_event_rx) = mpsc::channel(OUTBOUND_MESSAGE_CAPACITY);
    let _stdio_handles = start_stdio_connection(transport_event_tx, reader, writer).await?;
    run_transport_events(server, transport_event_rx, true, CancellationToken::new()).await
}

async fn run_transport_events(
    server: AppServer,
    mut transport_event_rx: mpsc::Receiver<TransportEvent>,
    single_client_mode: bool,
    shutdown_token: CancellationToken,
) -> Result<(), AppServerError> {
    let (streamed_tx, mut streamed_rx) = mpsc::unbounded_channel::<StreamedTransportMessage>();
    let mcp_elicitation_shutdown = CancellationToken::new();
    let mcp_elicitation_pump = tokio::spawn(mcp_elicitation::run_request_pump(
        server.clone(),
        server.mcp_elicitation_requests.clone(),
        mcp_elicitation_shutdown.clone(),
    ));

    let transport_result = async {
        loop {
            tokio::select! {
                event = transport_event_rx.recv() => {
                    let Some(event) = event else {
                        break;
                    };
                    match event {
                        TransportEvent::ConnectionOpened {
                            connection_id,
                            writer,
                            disconnect_sender,
                            ..
                        } => {
                            server.register_transport_writer(
                                connection_id,
                                writer,
                                disconnect_sender,
                            );
                        }
                        TransportEvent::StdioClientInitialized { .. } => {}
                        TransportEvent::ConnectionClosed { connection_id } => {
                            let idle_thread_ids = server
                                .thread_states
                                .disconnect_connection(connection_id)
                                .await;
                            for thread_id in idle_thread_ids {
                                server.schedule_thread_unload(thread_id);
                            }
                            server.unregister_transport_writer(connection_id);
                            server.server_requests.cancel_owner(
                                server_request::ServerRequestOwner::Transport(connection_id),
                                "App Server transport disconnected",
                            );
                            if single_client_mode {
                                break;
                            }
                        }
                        TransportEvent::IncomingMessage {
                            connection_id,
                            message,
                        } => {
                            let is_initialize = matches!(
                                &message,
                                JsonRpcMessage::Request(request)
                                    if request.method == METHOD_INITIALIZE
                            );
                            let connection_initialized = transport_is_initialized(
                                &server.transport_initialized,
                                connection_id,
                            );
                            if is_initialize && connection_initialized {
                                let JsonRpcMessage::Request(request) = &message else {
                                    unreachable!("initialize must be a JSON-RPC request");
                                };
                                server
                                    .send_to_transport_connection(
                                        connection_id,
                                        JsonRpcMessage::Error(
                                            app_server_protocol::JsonRpcErrorResponse {
                                                id: request.id.clone(),
                                                error: JsonRpcError::new(
                                                    error_codes::ALREADY_INITIALIZED,
                                                    "initialize has already been accepted for this connection",
                                                ),
                                            },
                                        ),
                                    )
                                    .await?;
                                continue;
                            }
                            if !is_initialize && !connection_initialized {
                                if let JsonRpcMessage::Request(request) = &message {
                                    server
                                        .send_to_transport_connection(
                                            connection_id,
                                            JsonRpcMessage::Error(
                                                app_server_protocol::JsonRpcErrorResponse {
                                                    id: request.id.clone(),
                                                    error: JsonRpcError::new(
                                                        error_codes::NOT_INITIALIZED,
                                                        "initialize must complete for this connection before business methods",
                                                    ),
                                                },
                                            ),
                                        )
                                        .await?;
                                }
                                continue;
                            }
                            let notification_opt_out = if is_initialize {
                                initialize_notification_opt_out(&message)
                            } else {
                                None
                            };
                            if should_spawn_transport_request(&message) {
                                let prepared_resume = if transport_is_initialized(
                                    &server.transport_initialized,
                                    connection_id,
                                ) {
                                    if let Some((thread_id, request_id)) =
                                        thread_resume_request_context(&message)
                                    {
                                        let barrier = thread_state::ThreadResumeBarrier::new(
                                            connection_id,
                                            request_id,
                                        );
                                        server
                                            .event_bridge()
                                            .prepare_thread_resume(
                                                agent_protocol::ThreadId::new(thread_id.clone()),
                                                barrier.clone(),
                                            )
                                            .await
                                            .map_err(AppServerError::ThreadListener)?;
                                        Some((thread_id, barrier))
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                };
                                spawn_transport_request(
                                    server.clone(),
                                    connection_id,
                                    message,
                                    streamed_tx.clone(),
                                    prepared_resume,
                                );
                                continue;
                            }
                            let responses = server
                                .handle_transport_message(connection_id, message)
                                .await?;
                            let initialize_succeeded = is_initialize
                                && responses
                                    .iter()
                                    .any(|response| matches!(response, JsonRpcMessage::Response(_)));
                            for response in responses {
                                server
                                    .send_to_transport_connection(connection_id, response)
                                    .await?;
                            }
                            if initialize_succeeded {
                                server.set_transport_notification_opt_out(
                                    connection_id,
                                    notification_opt_out,
                                );
                                server.mark_transport_initialized(connection_id);
                                server
                                    .thread_states
                                    .connection_initialized(connection_id)
                                    .await;
                            }
                        }
                    }
                }
                streamed = streamed_rx.recv() => {
                    let Some(streamed) = streamed else {
                        break;
                    };
                    let (connection_id, message) = streamed?;
                    server
                        .send_to_transport_connection(connection_id, message)
                        .await?;
                }
                _ = shutdown_token.cancelled() => {
                    break;
                }
            }
        }
        Ok(())
    }
    .await;

    finish_json_lines(
        &server,
        mcp_elicitation_shutdown,
        mcp_elicitation_pump,
        transport_result,
    )
    .await
}

async fn finish_json_lines(
    server: &AppServer,
    mcp_elicitation_shutdown: tokio_util::sync::CancellationToken,
    mcp_elicitation_pump: tokio::task::JoinHandle<()>,
    transport_result: Result<(), AppServerError>,
) -> Result<(), AppServerError> {
    server.cancel_pending_server_requests("App Server transport stopped");
    mcp_elicitation_shutdown.cancel();
    let _ = mcp_elicitation_pump.await;
    server.thread_states.clear_all_listeners().await;
    server.clear_transport_writers().await;
    transport_result
}

fn response_thread_id(messages: &[JsonRpcMessage]) -> Option<String> {
    messages.iter().find_map(|message| {
        let JsonRpcMessage::Response(response) = message else {
            return None;
        };
        response
            .result
            .get("thread")
            .and_then(|thread| thread.get("id"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
    })
}

fn thread_unsubscribe_succeeded(messages: &[JsonRpcMessage]) -> bool {
    messages.iter().any(|message| {
        let JsonRpcMessage::Response(response) = message else {
            return false;
        };
        response
            .result
            .get("status")
            .and_then(serde_json::Value::as_str)
            == Some("unsubscribed")
    })
}

fn thread_token_usage_notification(
    thread_id: &str,
    snapshot: runtime::thread_usage::ThreadTokenUsageSnapshot,
) -> JsonRpcNotification {
    let breakdown = |usage: runtime::thread_usage::TokenUsageSnapshot| {
        app_server_protocol::protocol::v2::TokenUsageBreakdown {
            total_tokens: usage.total_tokens,
            input_tokens: usage.input_tokens,
            cached_input_tokens: usage.cached_input_tokens,
            cache_write_input_tokens: usage.cache_write_input_tokens,
            output_tokens: usage.output_tokens,
            reasoning_output_tokens: usage.reasoning_output_tokens,
        }
    };
    let notification =
        app_server_protocol::protocol::v2::ServerNotification::ThreadTokenUsageUpdated(
            app_server_protocol::protocol::v2::ThreadTokenUsageUpdatedNotification {
                thread_id: thread_id.to_string(),
                turn_id: snapshot.turn_id,
                token_usage: app_server_protocol::protocol::v2::ThreadTokenUsage {
                    total: breakdown(snapshot.total_token_usage),
                    last: breakdown(snapshot.last_token_usage),
                    model_context_window: snapshot.model_context_window,
                },
            },
        );
    notification.into()
}

fn transport_is_initialized(
    initialized: &TransportInitialized,
    connection_id: ConnectionId,
) -> bool {
    initialized
        .lock()
        .expect("app-server transport initialization mutex poisoned")
        .contains(&connection_id)
}

fn thread_resume_request_context(message: &JsonRpcMessage) -> Option<(String, RequestId)> {
    let JsonRpcMessage::Request(request) = message else {
        return None;
    };
    if request.method != app_server_protocol::protocol::v2::METHOD_THREAD_RESUME {
        return None;
    }
    let params = serde_json::from_value::<app_server_protocol::protocol::v2::ThreadResumeParams>(
        request.params.as_ref()?.clone(),
    )
    .ok()?;
    let thread_id = params.thread_id.trim();
    if thread_id.is_empty() {
        return None;
    }
    Some((thread_id.to_string(), request.id.clone()))
}

fn thread_goal_mutation_thread_id(message: &JsonRpcMessage) -> Option<String> {
    let JsonRpcMessage::Request(request) = message else {
        return None;
    };
    if request.method != app_server_protocol::protocol::v2::METHOD_THREAD_GOAL_SET
        && request.method != app_server_protocol::protocol::v2::METHOD_THREAD_GOAL_CLEAR
    {
        return None;
    }
    request
        .params
        .as_ref()?
        .get("threadId")?
        .as_str()
        .map(str::trim)
        .filter(|thread_id| !thread_id.is_empty())
        .map(str::to_string)
}

fn take_thread_goal_notification(
    messages: &mut Vec<JsonRpcMessage>,
    thread_id: &str,
) -> Option<JsonRpcNotification> {
    let index = messages.iter().position(|message| {
        let JsonRpcMessage::Notification(notification) = message else {
            return false;
        };
        let is_goal_notification = notification.method
            == app_server_protocol::protocol::v2::METHOD_THREAD_GOAL_UPDATED
            || notification.method == app_server_protocol::protocol::v2::METHOD_THREAD_GOAL_CLEARED;
        is_goal_notification
            && notification
                .params
                .as_ref()
                .and_then(|params| params.get("threadId"))
                .and_then(serde_json::Value::as_str)
                == Some(thread_id)
    })?;
    let JsonRpcMessage::Notification(notification) = messages.remove(index) else {
        unreachable!("goal notification position must contain a notification");
    };
    Some(notification)
}

fn take_thread_deleted_notifications(
    messages: &mut Vec<JsonRpcMessage>,
) -> Vec<(String, JsonRpcNotification)> {
    let mut retained = Vec::with_capacity(messages.len());
    let mut deleted = Vec::new();
    for message in std::mem::take(messages) {
        let JsonRpcMessage::Notification(notification) = &message else {
            retained.push(message);
            continue;
        };
        let Ok(app_server_protocol::protocol::v2::ServerNotification::ThreadDeleted(params)) =
            app_server_protocol::protocol::v2::ServerNotification::try_from(notification.clone())
        else {
            retained.push(message);
            continue;
        };
        deleted.push((params.thread_id, notification.clone()));
    }
    *messages = retained;
    deleted
}

async fn publish_thread_delete_transport_result(
    server: &AppServer,
    origin_connection_id: ConnectionId,
    messages: &mut Vec<JsonRpcMessage>,
) -> bool {
    let deleted = take_thread_deleted_notifications(messages);
    if deleted.is_empty() {
        return false;
    }

    let bridge = server.event_bridge();
    if let Err(error) = bridge
        .send_messages_to_connection(origin_connection_id, messages)
        .await
    {
        tracing::warn!(
            %origin_connection_id,
            %error,
            "failed to send thread/delete response"
        );
    }

    let deleted_thread_ids = deleted
        .iter()
        .map(|(thread_id, _)| thread_id.clone())
        .collect::<Vec<_>>();
    server
        .server_requests
        .abort_for_threads(&bridge, &deleted_thread_ids, "thread deleted")
        .await;

    for (thread_id, notification) in deleted {
        let canonical_thread_id = agent_protocol::ThreadId::new(&thread_id);
        let (completion_tx, completion_rx) = oneshot::channel();
        let publish_result = match bridge
            .send_thread_command(
                canonical_thread_id.clone(),
                thread_state::ThreadListenerCommand::PublishNotification {
                    notification,
                    origin_connection_id: Some(origin_connection_id),
                    completion_tx: Some(completion_tx),
                },
            )
            .await
        {
            Ok(()) => completion_rx
                .await
                .map_err(|error| error.to_string())
                .and_then(|result| result),
            Err(error) => Err(error),
        };
        if let Err(error) = publish_result {
            tracing::warn!(
                %origin_connection_id,
                %thread_id,
                %error,
                "failed to publish thread/deleted notification"
            );
        }
        server
            .thread_states
            .remove_thread(&canonical_thread_id)
            .await;
    }
    true
}

fn spawn_transport_request(
    server: AppServer,
    connection_id: ConnectionId,
    message: JsonRpcMessage,
    streamed_tx: mpsc::UnboundedSender<StreamedTransportMessage>,
    prepared_resume: Option<(String, thread_state::ThreadResumeBarrier)>,
) {
    tokio::spawn(async move {
        let is_thread_delete = matches!(
            &message,
            JsonRpcMessage::Request(request)
                if request.method == app_server_protocol::protocol::v2::METHOD_THREAD_DELETE
        );
        let thread_unsubscribe_id = match &message {
            JsonRpcMessage::Request(request)
                if request.method
                    == app_server_protocol::protocol::v2::METHOD_THREAD_UNSUBSCRIBE =>
            {
                request
                    .params
                    .as_ref()
                    .and_then(|params| params.get("threadId"))
                    .and_then(serde_json::Value::as_str)
                    .map(agent_protocol::ThreadId::new)
            }
            _ => None,
        };
        let goal_mutation_thread_id = thread_goal_mutation_thread_id(&message);
        let subscription_method = match &message {
            JsonRpcMessage::Request(request)
                if request.method == METHOD_THREAD_START
                    || request.method
                        == app_server_protocol::protocol::v2::METHOD_THREAD_RESUME =>
            {
                Some(request.method.clone())
            }
            _ => None,
        };
        if should_stream_transport_request(&message) {
            let mut event_callback = |message: JsonRpcMessage| {
                let _ = streamed_tx.send(Ok((connection_id, message)));
            };
            match server
                .handle_transport_message_streaming(connection_id, message, &mut event_callback)
                .await
            {
                Ok(messages) => {
                    for message in messages {
                        let _ = streamed_tx.send(Ok((connection_id, message)));
                    }
                }
                Err(error) => {
                    let _ = streamed_tx.send(Err(error));
                }
            }
            return;
        }

        match server
            .handle_transport_message(connection_id, message)
            .await
        {
            Ok(mut messages) => {
                if is_thread_delete
                    && publish_thread_delete_transport_result(&server, connection_id, &mut messages)
                        .await
                {
                    return;
                }
                if let Some((thread_id, barrier)) = prepared_resume {
                    let successful_resume = response_thread_id(&messages)
                        .as_deref()
                        .is_some_and(|response_thread_id| response_thread_id == thread_id);
                    let bridge = server.event_bridge();
                    let mut goal_outbox_through_id = None;
                    if successful_resume {
                        let owner = server_request::ServerRequestOwner::Transport(connection_id);
                        server.server_requests.claim_owner_thread(owner, &thread_id);
                        if let Some(snapshot) = server
                            .processor
                            .runtime()
                            .thread_token_usage_snapshot(&thread_id)
                        {
                            messages.push(JsonRpcMessage::Notification(
                                thread_token_usage_notification(&thread_id, snapshot),
                            ));
                        }
                        match bridge
                            .runtime_events
                            .latest_thread_goal_update_outbox_id(&thread_id)
                        {
                            Ok(outbox_id) => goal_outbox_through_id = outbox_id,
                            Err(error) => tracing::warn!(
                                thread_id = %thread_id,
                                "failed to read ThreadGoal resume outbox watermark: {error}"
                            ),
                        }
                        match server
                            .processor
                            .thread_goal_snapshot_notification(&thread_id)
                        {
                            Ok(notification) => {
                                messages.push(JsonRpcMessage::Notification(notification));
                            }
                            Err(error) => {
                                tracing::warn!(
                                    thread_id = %thread_id,
                                    "failed to read ThreadGoal resume snapshot: {error}"
                                );
                            }
                        }
                        messages.extend(
                            server
                                .server_requests
                                .snapshot_for_owner_thread(owner, &thread_id)
                                .into_iter()
                                .map(JsonRpcMessage::Request),
                        );
                    }
                    let (completion_tx, completion_rx) = oneshot::channel();
                    let command = thread_state::ThreadListenerCommand::CompleteResume {
                        barrier,
                        connection_id,
                        messages,
                        subscribe: successful_resume,
                        completion_tx,
                    };
                    let result = match bridge
                        .send_thread_command(agent_protocol::ThreadId::new(&thread_id), command)
                        .await
                    {
                        Ok(()) => completion_rx
                            .await
                            .map_err(|error| error.to_string())
                            .and_then(|result| result),
                        Err(error) => Err(error),
                    };
                    if result.is_ok() {
                        if let Some(outbox_id) = goal_outbox_through_id {
                            if let Err(error) = bridge
                                .runtime_events
                                .mark_thread_goal_snapshot_delivered(&thread_id, outbox_id)
                            {
                                tracing::warn!(
                                    thread_id = %thread_id,
                                    outbox_id,
                                    "ThreadGoal resume snapshot was sent but outbox acknowledgement failed: {error}"
                                );
                            }
                        }
                    }
                    if let Err(error) = result {
                        let _ = streamed_tx.send(Err(AppServerError::ThreadListener(error)));
                    }
                    return;
                }
                if let Some(thread_id) = goal_mutation_thread_id.as_deref() {
                    if let Some(notification) =
                        take_thread_goal_notification(&mut messages, thread_id)
                    {
                        let bridge = server.event_bridge();
                        if let Err(error) = bridge
                            .send_messages_to_connection(connection_id, &messages)
                            .await
                        {
                            tracing::warn!(
                                %connection_id,
                                %thread_id,
                                %error,
                                "failed to send ThreadGoal mutation response"
                            );
                        }
                        let (completion_tx, completion_rx) = oneshot::channel();
                        let command = thread_state::ThreadListenerCommand::PublishNotification {
                            notification,
                            origin_connection_id: Some(connection_id),
                            completion_tx: Some(completion_tx),
                        };
                        let publish_result = match bridge
                            .send_thread_command(agent_protocol::ThreadId::new(thread_id), command)
                            .await
                        {
                            Ok(()) => completion_rx
                                .await
                                .map_err(|error| error.to_string())
                                .and_then(|result| result),
                            Err(error) => Err(error),
                        };
                        if let Err(error) = publish_result {
                            tracing::warn!(
                                %connection_id,
                                %thread_id,
                                %error,
                                "failed to publish ThreadGoal notification"
                            );
                        } else if let Err(error) = server
                            .processor
                            .runtime()
                            .maybe_start_thread_goal_for_thread_if_idle(
                                thread_id,
                                server.processor.runtime_host_context(),
                            )
                            .await
                        {
                            tracing::warn!(
                                %connection_id,
                                %thread_id,
                                %error,
                                "failed to start idle ThreadGoal after mutation notification"
                            );
                        }
                        return;
                    }
                }
                if subscription_method.is_some() {
                    if let Some(thread_id) = response_thread_id(&messages) {
                        let (completion_tx, completion_rx) = oneshot::channel();
                        let command = thread_state::ThreadListenerCommand::SubscribeAndSend {
                            connection_id,
                            messages,
                            completion_tx,
                        };
                        let bridge = server.event_bridge();
                        let result = match bridge
                            .send_thread_command(agent_protocol::ThreadId::new(thread_id), command)
                            .await
                        {
                            Ok(()) => completion_rx
                                .await
                                .map_err(|error| error.to_string())
                                .and_then(|result| result),
                            Err(error) => Err(error),
                        };
                        if let Err(error) = result {
                            let _ = streamed_tx.send(Err(AppServerError::ThreadListener(error)));
                        }
                        return;
                    }
                }
                let schedule_unload =
                    thread_unsubscribe_id.filter(|_| thread_unsubscribe_succeeded(&messages));
                for message in messages {
                    let _ = streamed_tx.send(Ok((connection_id, message)));
                }
                if let Some(thread_id) = schedule_unload {
                    server.schedule_thread_unload(thread_id);
                }
            }
            Err(error) => {
                if let Some((thread_id, barrier)) = prepared_resume {
                    let (completion_tx, completion_rx) = oneshot::channel();
                    let command = thread_state::ThreadListenerCommand::CompleteResume {
                        barrier,
                        connection_id,
                        messages: Vec::new(),
                        subscribe: false,
                        completion_tx,
                    };
                    let bridge = server.event_bridge();
                    let cleanup_result = match bridge
                        .send_thread_command(agent_protocol::ThreadId::new(thread_id), command)
                        .await
                    {
                        Ok(()) => completion_rx
                            .await
                            .map_err(|error| error.to_string())
                            .and_then(|result| result),
                        Err(error) => Err(error),
                    };
                    if let Err(cleanup_error) = cleanup_result {
                        tracing::warn!(
                            %cleanup_error,
                            "failed to release thread resume barrier after request error"
                        );
                    }
                }
                let _ = streamed_tx.send(Err(error));
            }
        }
    });
}

#[cfg(test)]
fn enqueue_transport_outbound_message(
    writers: &TransportWriters,
    disconnects: &TransportDisconnects,
    initialized: &TransportInitialized,
    notification_opt_out: &TransportNotificationOptOut,
    message: JsonRpcMessage,
) {
    let writers = writers
        .lock()
        .expect("app-server transport writer mutex poisoned")
        .iter()
        .map(|(connection_id, writer)| (*connection_id, writer.clone()))
        .collect::<Vec<_>>();
    for (connection_id, writer) in writers {
        if !initialized
            .lock()
            .expect("app-server transport initialization mutex poisoned")
            .contains(&connection_id)
        {
            continue;
        }
        if notification_is_opted_out(notification_opt_out, connection_id, &message) {
            continue;
        }
        let disconnect_sender = disconnects
            .lock()
            .expect("app-server transport disconnect mutex poisoned")
            .get(&connection_id)
            .cloned();
        let queued = QueuedOutgoingMessage::new(OutgoingMessage::from(message.clone()));
        if let Some(disconnect_sender) = disconnect_sender {
            match writer.try_send(queued) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(_))
                | Err(mpsc::error::TrySendError::Closed(_)) => disconnect_sender.cancel(),
            }
        } else {
            tokio::spawn(async move {
                let _ = writer.send(queued).await;
            });
        }
    }
}

fn initialize_notification_opt_out(message: &JsonRpcMessage) -> Option<Vec<String>> {
    let JsonRpcMessage::Request(request) = message else {
        return None;
    };
    if request.method != METHOD_INITIALIZE {
        return None;
    }
    let params = request.params.as_ref()?.clone();
    serde_json::from_value::<InitializeParams>(params)
        .ok()
        .and_then(|params| params.capabilities.opt_out_notification_methods)
}

fn notification_is_opted_out(
    notification_opt_out: &TransportNotificationOptOut,
    connection_id: ConnectionId,
    message: &JsonRpcMessage,
) -> bool {
    let JsonRpcMessage::Notification(notification) = message else {
        return false;
    };
    notification_opt_out
        .lock()
        .expect("app-server transport notification opt-out mutex poisoned")
        .get(&connection_id)
        .is_some_and(|methods| methods.contains(&notification.method))
}

fn should_stream_transport_request(message: &JsonRpcMessage) -> bool {
    matches!(
        message,
        JsonRpcMessage::Request(request)
            if request.method == METHOD_AGENT_SESSION_MEDIA_READ
                    && request
                        .params
                        .as_ref()
                        .and_then(|params| params.get("stream"))
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false)
    )
}

fn should_spawn_transport_request(message: &JsonRpcMessage) -> bool {
    matches!(
        message,
        JsonRpcMessage::Request(request)
            if request.method != METHOD_INITIALIZE
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::ClientCapabilities;
    use app_server_protocol::ClientInfo;
    use app_server_protocol::InitializeParams;
    use app_server_protocol::RequestId;
    use app_server_protocol::METHOD_THREAD_LIST;
    use serde_json::json;
    use std::sync::Arc;
    use tokio::io::AsyncBufReadExt;
    use tokio::io::AsyncWriteExt;
    use tokio::io::BufReader;

    fn server_with_projection_store() -> (tempfile::TempDir, AppServer) {
        let temp = tempfile::tempdir().expect("projection tempdir");
        let store = ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store");
        let runtime = RuntimeCore::default().with_projection_store(Arc::new(store));
        (temp, AppServer::with_runtime(runtime))
    }

    #[test]
    fn default_control_socket_startup_lock_does_not_repeat_control_directory() {
        let base_dir = Path::new("state");
        let socket_path = app_server_transport::app_server_control_socket_path(base_dir);

        assert_eq!(
            startup_lock_path_for_socket(&socket_path),
            app_server_startup_lock_path(base_dir)
        );
    }

    #[tokio::test]
    async fn business_methods_require_initialized_notification() {
        let server = AppServer::new();

        let blocked_cases = [
            (1, METHOD_THREAD_START, json!({ "appId": "content-studio" })),
            (
                2,
                app_server_protocol::METHOD_WORKSPACE_UPDATE,
                json!({ "id": "workspace-main", "name": "Main" }),
            ),
            (
                3,
                app_server_protocol::METHOD_WORKSPACE_DELETE,
                json!({ "id": "workspace-main", "deleteDirectory": false }),
            ),
        ];

        for (id, method, params) in blocked_cases {
            let response = server
                .handle_message(JsonRpcMessage::Request(JsonRpcRequest::new(
                    RequestId::Integer(id),
                    method,
                    Some(params),
                )))
                .await
                .expect("handle")
                .remove(0);

            match response {
                JsonRpcMessage::Error(error) => {
                    assert_eq!(error.error.code, error_codes::NOT_INITIALIZED);
                }
                other => panic!("expected error for {method}, got {other:?}"),
            }
        }

        initialize(&server).await;

        let workspace_cases = [
            (
                4,
                app_server_protocol::METHOD_WORKSPACE_UPDATE,
                json!({ "id": "workspace-main", "name": "Main" }),
                "workspace/update is not available without an app data source",
            ),
            (
                5,
                app_server_protocol::METHOD_WORKSPACE_DELETE,
                json!({ "id": "workspace-main", "deleteDirectory": false }),
                "workspace/delete is not available without an app data source",
            ),
        ];

        for (id, method, params, message) in workspace_cases {
            let response = server
                .handle_message(JsonRpcMessage::Request(JsonRpcRequest::new(
                    RequestId::Integer(id),
                    method,
                    Some(params),
                )))
                .await
                .expect("handle")
                .remove(0);

            match response {
                JsonRpcMessage::Error(error) => {
                    assert_eq!(error.error.code, error_codes::RUNTIME_ERROR);
                    assert_eq!(error.error.message, message);
                }
                other => panic!("expected runtime error for {method}, got {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn session_and_turn_flow_runs_on_mock_runtime() {
        let (_temp, server) = server_with_projection_store();
        initialize(&server).await;

        let session_response = request(
            &server,
            2,
            METHOD_THREAD_START,
            json!({ "model": "gpt-4.1-mini", "modelProvider": "openai" }),
        )
        .await;
        let thread_id = session_response["thread"]["id"]
            .as_str()
            .expect("thread id")
            .to_string();
        let session_id = session_response["thread"]["sessionId"]
            .as_str()
            .expect("session id")
            .to_string();

        let turn_response = request(
            &server,
            3,
            METHOD_TURN_START,
            json!({
                "threadId": thread_id,
                "input": [{"type": "text", "text": "生成草稿"}]
            }),
        )
        .await;

        assert_eq!(turn_response["turn"]["id"].as_str().is_some(), true);
        assert_eq!(turn_response["turn"]["status"], "inProgress");

        let events = server
            .runtime()
            .events_for_session(&session_id)
            .expect("stored events");
        assert_eq!(
            events
                .iter()
                .map(|event| event.event_type.as_str())
                .collect::<Vec<_>>(),
            vec![
                "item.started",
                "message.created",
                "item.completed",
                "turn.accepted"
            ]
        );
    }

    #[tokio::test]
    async fn session_start_rejects_duplicate_caller_supplied_session_id() {
        let server = AppServer::new();
        initialize(&server).await;
        let params = json!({
            "sessionId": "sess_external",
            "threadId": "thread_external",
            "appId": "content-studio",
            "workspaceId": "default"
        });
        let messages = server
            .handle_message(JsonRpcMessage::Request(JsonRpcRequest::new(
                RequestId::Integer(2),
                METHOD_THREAD_START,
                Some(params),
            )))
            .await
            .expect("handle");

        match messages.first().expect("error response") {
            JsonRpcMessage::Error(error) => {
                assert_eq!(error.error.code, error_codes::INVALID_PARAMS);
                assert!(!error.error.message.is_empty());
            }
            other => panic!("expected current v2 contract rejection, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn capability_list_flows_through_json_rpc_router_with_scoped_inventory_source() {
        let runtime = AppServerRuntimeFactory::mock_runtime_core_with_capability_source(Arc::new(
            CapabilityInventorySource::new(vec![
                CapabilityInventoryRecord::new(CapabilityDescriptor {
                    id: "global.agent.session".to_string(),
                    title: "Global Agent Session".to_string(),
                    description: None,
                    methods: vec![METHOD_THREAD_START.to_string()],
                }),
                CapabilityInventoryRecord::new(CapabilityDescriptor {
                    id: "content.draft.generate".to_string(),
                    title: "Generate Draft".to_string(),
                    description: Some("Content Studio draft capability".to_string()),
                    methods: vec![METHOD_TURN_START.to_string()],
                })
                .for_apps(["content-studio"])
                .for_workspaces(["default"]),
                CapabilityInventoryRecord::new(CapabilityDescriptor {
                    id: "other.private".to_string(),
                    title: "Other Private".to_string(),
                    description: None,
                    methods: vec!["other/method".to_string()],
                })
                .for_apps(["other-app"]),
            ]),
        ));
        let server = AppServer::with_runtime(runtime);
        initialize(&server).await;

        let response = request(
            &server,
            2,
            METHOD_CAPABILITY_LIST,
            json!({
                "appId": "content-studio",
                "workspaceId": "default",
                "limit": 1
            }),
        )
        .await;

        let capabilities = response["capabilities"].as_array().expect("capabilities");
        let ids: Vec<&str> = capabilities
            .iter()
            .map(|capability| capability["id"].as_str().expect("id"))
            .collect();
        assert_eq!(ids, vec!["global.agent.session"]);
        assert_eq!(response["nextCursor"].as_str().expect("next cursor"), "1");

        let next_response = request(
            &server,
            3,
            METHOD_CAPABILITY_LIST,
            json!({
                "appId": "content-studio",
                "workspaceId": "default",
                "cursor": response["nextCursor"].as_str().expect("next cursor"),
                "limit": 1
            }),
        )
        .await;
        let next_capabilities = next_response["capabilities"]
            .as_array()
            .expect("next capabilities");
        assert_eq!(next_capabilities[0]["id"], "content.draft.generate");
        assert_eq!(next_capabilities[0]["methods"][0], METHOD_TURN_START);
        assert!(next_response.get("nextCursor").is_none());
    }

    #[tokio::test]
    async fn capability_list_with_session_id_uses_stored_session_scope() {
        let server = AppServer::new();
        initialize(&server).await;
        let messages = server
            .handle_message(JsonRpcMessage::Request(JsonRpcRequest::new(
                RequestId::Integer(2),
                METHOD_CAPABILITY_LIST,
                Some(json!({ "sessionId": "sess_capability_scope" })),
            )))
            .await
            .expect("current v2 contract response");
        match messages.first().expect("error response") {
            JsonRpcMessage::Error(error) => {
                assert_eq!(error.error.code, error_codes::SESSION_NOT_FOUND);
            }
            other => panic!("expected current v2 contract rejection, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn turn_start_rejects_legacy_session_shape_before_runtime_dispatch() {
        let server = AppServer::new();
        initialize(&server).await;
        let messages = server
            .handle_message(JsonRpcMessage::Request(JsonRpcRequest::new(
                RequestId::Integer(2),
                METHOD_TURN_START,
                Some(json!({
                    "sessionId": "sess_denied",
                    "input": {
                        "text": "draft"
                    },
                    "runtimeOptions": {
                        "capabilityId": "content.draft.generate"
                    }
                })),
            )))
            .await
            .expect("turn start");

        match messages.first().expect("error response") {
            JsonRpcMessage::Error(error) => {
                assert_eq!(error.error.code, error_codes::INVALID_PARAMS);
                assert!(error.error.message.contains("threadId"));
            }
            other => panic!("expected current v2 contract rejection, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn turn_start_rejects_parallel_active_turn_without_queue_flag() {
        let (_temp, server) = server_with_projection_store();
        initialize(&server).await;
        let thread = request(
            &server,
            2,
            METHOD_THREAD_START,
            json!({
                "model": "gpt-4.1-mini",
                "modelProvider": "openai"
            }),
        )
        .await;
        let thread_id = thread["thread"]["id"].as_str().expect("thread id");
        let first = request(
            &server,
            3,
            METHOD_TURN_START,
            json!({
                "threadId": thread_id,
                "input": [{"type": "text", "text": "running"}]
            }),
        )
        .await;
        let active_turn_id = first["turn"]["id"].as_str().expect("active turn id");

        let messages = server
            .handle_message(JsonRpcMessage::Request(JsonRpcRequest::new(
                RequestId::Integer(4),
                METHOD_TURN_START,
                Some(json!({
                    "threadId": thread_id,
                    "input": [{"type": "text", "text": "parallel"}]
                })),
            )))
            .await
            .expect("parallel turn start");

        match messages.first().expect("error response") {
            JsonRpcMessage::Error(error) => {
                assert_eq!(error.error.code, error_codes::TURN_ALREADY_ACTIVE);
                assert_eq!(
                    error.error.message,
                    format!("turn already active: {active_turn_id}")
                );
            }
            other => panic!("expected active turn error, got {other:?}"),
        }

        let read = request(
            &server,
            5,
            METHOD_THREAD_READ,
            json!({ "threadId": thread_id, "includeTurns": true }),
        )
        .await;
        let turns = read["thread"]["turns"].as_array().expect("turns");
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0]["id"], active_turn_id);
    }

    #[tokio::test]
    async fn turn_start_returns_response_and_event_notification() {
        let (_temp, server) = server_with_projection_store();
        initialize(&server).await;

        let session_response = request(
            &server,
            2,
            METHOD_THREAD_START,
            json!({ "model": "gpt-4.1-mini", "modelProvider": "openai" }),
        )
        .await;
        let thread_id = session_response["thread"]["id"]
            .as_str()
            .expect("thread id")
            .to_string();

        let messages = server
            .handle_message(JsonRpcMessage::Request(JsonRpcRequest::new(
                RequestId::Integer(3),
                METHOD_TURN_START,
                Some(json!({
                    "threadId": thread_id,
                    "input": [{"type": "text", "text": "生成草稿"}]
                })),
            )))
            .await
            .expect("handle");

        let [JsonRpcMessage::Response(response)] = messages.as_slice() else {
            panic!("expected current v2 turn response, got {messages:?}");
        };
        assert_eq!(response.result["turn"]["status"], "inProgress");
        assert!(response.result["turn"]["id"].as_str().is_some());
    }
    #[tokio::test]
    async fn append_external_runtime_events_returns_json_rpc_notifications() {
        let (_temp, server) = server_with_projection_store();
        initialize(&server).await;
        let thread_response = request(
            &server,
            2,
            METHOD_THREAD_START,
            json!({
                "model": "gpt-4.1-mini",
                "modelProvider": "openai"
            }),
        )
        .await;
        let thread_id = thread_response["thread"]["id"].as_str().expect("thread id");
        let turn_response = request(
            &server,
            3,
            METHOD_TURN_START,
            json!({
                "threadId": thread_id,
                "input": [{"type": "text", "text": "draft"}]
            }),
        )
        .await;
        let turn_id = turn_response["turn"]["id"].as_str().expect("turn id");

        let notifications = server
            .append_external_runtime_events(
                thread_response["thread"]["sessionId"]
                    .as_str()
                    .expect("session id"),
                Some(turn_id),
                vec![RuntimeEvent::new(
                    "message.delta",
                    json!({ "text": "delta" }),
                )],
            )
            .await
            .expect("notifications");

        assert_eq!(notifications.len(), 2);
        match &notifications[0] {
            JsonRpcMessage::Notification(notification) => {
                assert_eq!(notification.method, "item/started");
                let params = notification.params.as_ref().expect("params");
                assert_eq!(params["threadId"], thread_id);
                assert_eq!(params["turnId"], turn_id);
                assert_eq!(params["item"]["type"], "agentMessage");
            }
            other => panic!("expected notification, got {other:?}"),
        }
        match &notifications[1] {
            JsonRpcMessage::Notification(notification) => {
                assert_eq!(notification.method, "item/agentMessage/delta");
                let params = notification.params.as_ref().expect("params");
                assert_eq!(params["threadId"], thread_id);
                assert_eq!(params["turnId"], turn_id);
                assert_eq!(params["delta"], "delta");
            }
            other => panic!("expected notification, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn append_external_runtime_events_publishes_outbound_notification() {
        let (_temp, server) = server_with_projection_store();
        initialize(&server).await;
        let thread_response = request(
            &server,
            2,
            METHOD_THREAD_START,
            json!({
                "model": "gpt-4.1-mini",
                "modelProvider": "openai"
            }),
        )
        .await;
        let thread_id = thread_response["thread"]["id"].as_str().expect("thread id");
        let turn_response = request(
            &server,
            3,
            METHOD_TURN_START,
            json!({
                "threadId": thread_id,
                "input": [{"type": "text", "text": "draft"}]
            }),
        )
        .await;
        let turn_id = turn_response["turn"]["id"].as_str().expect("turn id");
        let mut outbound_messages = server.subscribe_outbound_messages();

        server
            .append_external_runtime_events(
                thread_response["thread"]["sessionId"]
                    .as_str()
                    .expect("session id"),
                Some(turn_id),
                vec![RuntimeEvent::new(
                    "message.delta",
                    json!({ "text": "async delta" }),
                )],
            )
            .await
            .expect("notifications");

        let lifecycle_message =
            tokio::time::timeout(std::time::Duration::from_secs(1), outbound_messages.recv())
                .await
                .expect("outbound lifecycle message")
                .expect("broadcast lifecycle message");
        match lifecycle_message {
            JsonRpcMessage::Notification(notification) => {
                assert_eq!(notification.method, "item/started");
                let params = notification.params.as_ref().expect("params");
                assert_eq!(params["threadId"], thread_id);
                assert_eq!(params["turnId"], turn_id);
                assert_eq!(params["item"]["type"], "agentMessage");
            }
            other => panic!("expected outbound lifecycle notification, got {other:?}"),
        }
        let delta_message = tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                let message = outbound_messages
                    .recv()
                    .await
                    .expect("broadcast delta message");
                if matches!(
                    &message,
                    JsonRpcMessage::Notification(notification)
                        if notification.method == "item/agentMessage/delta"
                ) {
                    break message;
                }
            }
        })
        .await
        .expect("outbound delta message");
        match delta_message {
            JsonRpcMessage::Notification(notification) => {
                assert_eq!(notification.method, "item/agentMessage/delta");
                let params = notification.params.as_ref().expect("params");
                assert_eq!(params["threadId"], thread_id);
                assert_eq!(params["turnId"], turn_id);
                assert_eq!(params["delta"], "async delta");
            }
            other => panic!("expected outbound notification, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn json_lines_loop_writes_external_outbound_notification() {
        let (_temp, server) = server_with_projection_store();
        let (mut input_client, input_server) = tokio::io::duplex(4096);
        let (output_server, output_client) = tokio::io::duplex(4096);
        let runner_server = server.clone();
        let runner = tokio::spawn(async move {
            run_json_lines(runner_server, input_server, output_server).await
        });
        let mut output_lines = BufReader::new(output_client).lines();

        write_json_message(
            &mut input_client,
            JsonRpcMessage::Request(JsonRpcRequest::new(
                RequestId::Integer(1),
                METHOD_INITIALIZE,
                Some(
                    serde_json::to_value(InitializeParams {
                        client_info: ClientInfo {
                            name: "json-lines-test".to_string(),
                            title: None,
                            version: None,
                        },
                        capabilities: ClientCapabilities::default(),
                    })
                    .expect("initialize params"),
                ),
            )),
        )
        .await;
        assert_response_id(
            next_json_message(&mut output_lines).await,
            RequestId::Integer(1),
        );

        write_json_message(
            &mut input_client,
            JsonRpcMessage::Notification(JsonRpcNotification::new(
                METHOD_INITIALIZED,
                Some(json!({})),
            )),
        )
        .await;
        write_json_message(
            &mut input_client,
            JsonRpcMessage::Request(JsonRpcRequest::new(
                RequestId::Integer(2),
                METHOD_THREAD_START,
                Some(json!({
                    "model": "fixture-model",
                    "modelProvider": "fixture-provider"
                })),
            )),
        )
        .await;
        let thread_start = next_json_message(&mut output_lines).await;
        let (thread_id, session_id) = match thread_start {
            JsonRpcMessage::Response(response) => (
                response.result["thread"]["id"]
                    .as_str()
                    .expect("thread id")
                    .to_string(),
                response.result["thread"]["sessionId"]
                    .as_str()
                    .expect("session id")
                    .to_string(),
            ),
            other => panic!("expected thread response, got {other:?}"),
        };

        write_json_message(
            &mut input_client,
            JsonRpcMessage::Request(JsonRpcRequest::new(
                RequestId::Integer(3),
                METHOD_TURN_START,
                Some(json!({
                    "threadId": thread_id,
                    "input": [{"type": "text", "text": "draft"}]
                })),
            )),
        )
        .await;
        let turn_result = next_json_response_result(&mut output_lines, RequestId::Integer(3)).await;
        let turn_id = turn_result["turn"]["id"]
            .as_str()
            .expect("turn id")
            .to_string();

        server
            .append_external_runtime_events(
                &session_id,
                Some(&turn_id),
                vec![RuntimeEvent::new(
                    "message.delta",
                    json!({ "text": "stdio async delta" }),
                )],
            )
            .await
            .expect("append external event");

        assert_next_direct_delta_notification(
            &mut output_lines,
            &thread_id,
            &turn_id,
            "stdio async delta",
        )
        .await;

        drop(input_client);
        runner.await.expect("runner join").expect("runner result");
    }

    #[tokio::test]
    async fn json_lines_loop_streams_external_backend_events_before_turn_response() {
        let Some(node) = node_binary() else {
            return;
        };
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let script_path = temp_dir.path().join("external-backend-stream.mjs");
        std::fs::write(
            &script_path,
            r#"
              import { setTimeout as delay } from 'node:timers/promises';
              console.log(JSON.stringify({
                type: 'message.delta',
                payload: { chunk: 1, text: 'hello' }
              }));
              await delay(10);
              console.log(JSON.stringify({
                type: 'message.delta',
                payload: { chunk: 2, text: 'world' }
              }));
            "#,
        )
        .expect("write backend script");

        let runtime = AppServerRuntimeFactory::external_runtime_core(
            ExternalBackendConfig::new(node)
                .with_args([script_path.to_string_lossy().to_string()])
                .with_timeout_ms(10_000),
        )
        .with_projection_store(Arc::new(
            ProjectionStore::initialize(temp_dir.path().join("projection.sqlite"))
                .expect("projection store"),
        ));
        let server = AppServer::with_runtime(runtime);
        let (mut input_client, input_server) = tokio::io::duplex(4096);
        let (output_server, output_client) = tokio::io::duplex(4096);
        let runner =
            tokio::spawn(async move { run_json_lines(server, input_server, output_server).await });
        let mut output_lines = BufReader::new(output_client).lines();

        write_json_message(
            &mut input_client,
            JsonRpcMessage::Request(JsonRpcRequest::new(
                RequestId::Integer(1),
                METHOD_INITIALIZE,
                Some(
                    serde_json::to_value(InitializeParams {
                        client_info: ClientInfo {
                            name: "content-studio".to_string(),
                            title: None,
                            version: Some("0.1.0".to_string()),
                        },
                        capabilities: ClientCapabilities::default(),
                    })
                    .expect("initialize params"),
                ),
            )),
        )
        .await;
        assert_response_id(
            next_json_message(&mut output_lines).await,
            RequestId::Integer(1),
        );

        write_json_message(
            &mut input_client,
            JsonRpcMessage::Notification(JsonRpcNotification::new(
                METHOD_INITIALIZED,
                Some(json!({})),
            )),
        )
        .await;
        write_json_message(
            &mut input_client,
            JsonRpcMessage::Request(JsonRpcRequest::new(
                RequestId::Integer(2),
                METHOD_THREAD_START,
                Some(json!({
                    "model": "fixture-model",
                    "modelProvider": "fixture-provider"
                })),
            )),
        )
        .await;
        let thread_id = match next_json_message(&mut output_lines).await {
            JsonRpcMessage::Response(response) => response.result["thread"]["id"]
                .as_str()
                .expect("thread id")
                .to_string(),
            other => panic!("expected thread response, got {other:?}"),
        };

        write_json_message(
            &mut input_client,
            JsonRpcMessage::Request(JsonRpcRequest::new(
                RequestId::Integer(3),
                METHOD_TURN_START,
                Some(json!({
                    "threadId": thread_id,
                    "input": [{"type": "text", "text": "draft"}]
                })),
            )),
        )
        .await;

        let turn_result = next_json_response_result(&mut output_lines, RequestId::Integer(3)).await;
        assert_eq!(turn_result["turn"]["status"], "inProgress");
        let turn_id = turn_result["turn"]["id"]
            .as_str()
            .expect("turn id")
            .to_string();
        assert_next_direct_delta_notification(&mut output_lines, &thread_id, &turn_id, "hello")
            .await;
        assert_next_direct_delta_notification(&mut output_lines, &thread_id, &turn_id, "world")
            .await;

        drop(input_client);
        runner.await.expect("runner join").expect("runner result");
    }

    #[tokio::test]
    async fn json_lines_loop_lists_sessions_while_external_turn_is_waiting_for_first_output() {
        let Some(node) = node_binary() else {
            return;
        };
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let script_path = temp_dir
            .path()
            .join("external-backend-waits-before-output.mjs");
        let trigger_path = temp_dir.path().join("release-backend-output");
        std::fs::write(
            &script_path,
            r#"
              import { access } from 'node:fs/promises';
              import { setTimeout as delay } from 'node:timers/promises';
              const triggerPath = process.argv[2];
              for (;;) {
                try {
                  await access(triggerPath);
                  break;
                } catch {
                  await delay(10);
                }
              }
              console.log(JSON.stringify({
                type: 'message.delta',
                payload: { text: 'late first output' }
              }));
            "#,
        )
        .expect("write backend script");

        let runtime = AppServerRuntimeFactory::external_runtime_core(
            ExternalBackendConfig::new(node)
                .with_args([
                    script_path.to_string_lossy().to_string(),
                    trigger_path.to_string_lossy().to_string(),
                ])
                .with_timeout_ms(10_000),
        )
        .with_projection_store(Arc::new(
            ProjectionStore::initialize(temp_dir.path().join("projection.sqlite"))
                .expect("projection store"),
        ));
        let server = AppServer::with_runtime(runtime);
        let (mut input_client, input_server) = tokio::io::duplex(4096);
        let (output_server, output_client) = tokio::io::duplex(4096);
        let runner =
            tokio::spawn(async move { run_json_lines(server, input_server, output_server).await });
        let mut output_lines = BufReader::new(output_client).lines();

        write_json_message(
            &mut input_client,
            JsonRpcMessage::Request(JsonRpcRequest::new(
                RequestId::Integer(1),
                METHOD_INITIALIZE,
                Some(
                    serde_json::to_value(InitializeParams {
                        client_info: ClientInfo {
                            name: "content-studio".to_string(),
                            title: None,
                            version: Some("0.1.0".to_string()),
                        },
                        capabilities: ClientCapabilities::default(),
                    })
                    .expect("initialize params"),
                ),
            )),
        )
        .await;
        assert_response_id(
            next_json_message(&mut output_lines).await,
            RequestId::Integer(1),
        );

        write_json_message(
            &mut input_client,
            JsonRpcMessage::Notification(JsonRpcNotification::new(
                METHOD_INITIALIZED,
                Some(json!({})),
            )),
        )
        .await;
        write_json_message(
            &mut input_client,
            JsonRpcMessage::Request(JsonRpcRequest::new(
                RequestId::Integer(2),
                METHOD_THREAD_START,
                Some(json!({
                    "model": "fixture-model",
                    "modelProvider": "fixture-provider"
                })),
            )),
        )
        .await;
        let (thread_id, session_id) = match next_json_message(&mut output_lines).await {
            JsonRpcMessage::Response(response) => (
                response.result["thread"]["id"]
                    .as_str()
                    .expect("thread id")
                    .to_string(),
                response.result["thread"]["sessionId"]
                    .as_str()
                    .expect("session id")
                    .to_string(),
            ),
            other => panic!("expected thread response, got {other:?}"),
        };

        write_json_message(
            &mut input_client,
            JsonRpcMessage::Request(JsonRpcRequest::new(
                RequestId::Integer(3),
                METHOD_TURN_START,
                Some(json!({
                    "threadId": thread_id,
                    "input": [{"type": "text", "text": "draft"}]
                })),
            )),
        )
        .await;

        write_json_message(
            &mut input_client,
            JsonRpcMessage::Request(JsonRpcRequest::new(
                RequestId::Integer(4),
                METHOD_THREAD_LIST,
                Some(json!({ "limit": 10 })),
            )),
        )
        .await;
        let mut turn_id = None;
        let list_response = loop {
            let message = next_json_message_with_timeout(
                &mut output_lines,
                std::time::Duration::from_millis(500),
            )
            .await;
            match message {
                JsonRpcMessage::Response(response) if response.id == RequestId::Integer(3) => {
                    turn_id = Some(
                        response.result["turn"]["id"]
                            .as_str()
                            .expect("turn id")
                            .to_string(),
                    );
                }
                JsonRpcMessage::Response(response) if response.id == RequestId::Integer(4) => {
                    break response;
                }
                _ => {}
            }
        };
        let turn_id = match turn_id {
            Some(turn_id) => turn_id,
            None => next_json_response_result(&mut output_lines, RequestId::Integer(3)).await
                ["turn"]["id"]
                .as_str()
                .expect("turn id")
                .to_string(),
        };
        let sessions = list_response.result["data"]
            .as_array()
            .expect("thread data array");
        assert!(
            sessions
                .iter()
                .any(|session| session["sessionId"] == session_id),
            "running session should be listed while backend waits"
        );

        std::fs::write(&trigger_path, b"continue").expect("release backend output");

        assert_next_direct_delta_notification(
            &mut output_lines,
            &thread_id,
            &turn_id,
            "late first output",
        )
        .await;

        drop(input_client);
        runner.await.expect("runner join").expect("runner result");
    }

    #[tokio::test]
    async fn json_lines_loop_streams_turn_failed_after_partial_external_backend_events() {
        let Some(node) = node_binary() else {
            return;
        };
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let script_path = temp_dir.path().join("external-backend-stream-fails.mjs");
        std::fs::write(
            &script_path,
            r#"
              console.log(JSON.stringify({
                type: 'message.delta',
                payload: { chunk: 1, text: 'partial' }
              }));
              console.error('external backend crashed after partial output');
              process.exit(7);
            "#,
        )
        .expect("write backend script");

        let runtime = AppServerRuntimeFactory::external_runtime_core(
            ExternalBackendConfig::new(node)
                .with_args([script_path.to_string_lossy().to_string()])
                .with_timeout_ms(10_000),
        )
        .with_projection_store(Arc::new(
            ProjectionStore::initialize(temp_dir.path().join("projection.sqlite"))
                .expect("projection store"),
        ));
        let server = AppServer::with_runtime(runtime);
        let (mut input_client, input_server) = tokio::io::duplex(4096);
        let (output_server, output_client) = tokio::io::duplex(4096);
        let runner =
            tokio::spawn(async move { run_json_lines(server, input_server, output_server).await });
        let mut output_lines = BufReader::new(output_client).lines();

        write_json_message(
            &mut input_client,
            JsonRpcMessage::Request(JsonRpcRequest::new(
                RequestId::Integer(1),
                METHOD_INITIALIZE,
                Some(
                    serde_json::to_value(InitializeParams {
                        client_info: ClientInfo {
                            name: "content-studio".to_string(),
                            title: None,
                            version: Some("0.1.0".to_string()),
                        },
                        capabilities: ClientCapabilities::default(),
                    })
                    .expect("initialize params"),
                ),
            )),
        )
        .await;
        assert_response_id(
            next_json_message(&mut output_lines).await,
            RequestId::Integer(1),
        );

        write_json_message(
            &mut input_client,
            JsonRpcMessage::Notification(JsonRpcNotification::new(
                METHOD_INITIALIZED,
                Some(json!({})),
            )),
        )
        .await;
        write_json_message(
            &mut input_client,
            JsonRpcMessage::Request(JsonRpcRequest::new(
                RequestId::Integer(2),
                METHOD_THREAD_START,
                Some(json!({
                    "model": "fixture-model",
                    "modelProvider": "fixture-provider"
                })),
            )),
        )
        .await;
        let (thread_id, session_id) = match next_json_message(&mut output_lines).await {
            JsonRpcMessage::Response(response) => (
                response.result["thread"]["id"]
                    .as_str()
                    .expect("thread id")
                    .to_string(),
                response.result["thread"]["sessionId"]
                    .as_str()
                    .expect("session id")
                    .to_string(),
            ),
            other => panic!("expected thread response, got {other:?}"),
        };

        write_json_message(
            &mut input_client,
            JsonRpcMessage::Request(JsonRpcRequest::new(
                RequestId::Integer(3),
                METHOD_TURN_START,
                Some(json!({
                    "threadId": thread_id,
                    "input": [{"type": "text", "text": "draft"}]
                })),
            )),
        )
        .await;

        let turn_result = next_json_response_result(&mut output_lines, RequestId::Integer(3)).await;
        let turn_id = turn_result["turn"]["id"]
            .as_str()
            .expect("turn id")
            .to_string();
        assert_next_direct_delta_notification(&mut output_lines, &thread_id, &turn_id, "partial")
            .await;
        loop {
            let message = next_json_message_with_timeout(
                &mut output_lines,
                std::time::Duration::from_secs(10),
            )
            .await;
            let JsonRpcMessage::Notification(notification) = message else {
                continue;
            };
            if notification.method != "turn/completed" {
                continue;
            }
            let params = notification.params.as_ref().expect("params");
            if params["threadId"] != thread_id || params["turn"]["id"] != turn_id {
                continue;
            }
            assert_eq!(params["turn"]["status"], "failed");
            break;
        }

        write_json_message(
            &mut input_client,
            JsonRpcMessage::Request(JsonRpcRequest::new(
                RequestId::Integer(4),
                METHOD_EVIDENCE_EXPORT,
                Some(json!({
                    "sessionId": session_id,
                    "turnId": turn_id,
                    "includeEvents": true,
                    "includeArtifacts": true
                })),
            )),
        )
        .await;
        let evidence = next_json_response_result(&mut output_lines, RequestId::Integer(4)).await;
        assert_eq!(evidence["session"]["sessionId"], session_id);
        assert_eq!(evidence["turns"][0]["turnId"], turn_id);
        let events = evidence["events"].as_array().expect("events");
        let message_created_index = events
            .iter()
            .position(|event| event["type"] == "message.created")
            .expect("message.created event");
        let message_delta_index = events
            .iter()
            .position(|event| event["type"] == "message.delta")
            .expect("message.delta event");
        let turn_failed_index = events
            .iter()
            .position(|event| event["type"] == "turn.failed")
            .expect("turn.failed event");
        assert!(
            message_created_index < message_delta_index && message_delta_index < turn_failed_index,
            "stored event order should preserve created -> delta -> failed: {events:?}"
        );
        assert!(events[turn_failed_index]["payload"]["message"]
            .as_str()
            .expect("evidence failure message")
            .contains("external backend crashed after partial output"));
        assert!(evidence["artifacts"]
            .as_array()
            .expect("artifacts")
            .is_empty());

        drop(input_client);
        runner.await.expect("runner join").expect("runner result");
    }

    async fn initialize(server: &AppServer) {
        request(
            server,
            1,
            METHOD_INITIALIZE,
            serde_json::to_value(InitializeParams {
                client_info: ClientInfo {
                    name: "test".to_string(),
                    title: None,
                    version: None,
                },
                capabilities: ClientCapabilities::default(),
            })
            .expect("params"),
        )
        .await;
        server
            .handle_message(JsonRpcMessage::Notification(JsonRpcNotification::new(
                METHOD_INITIALIZED,
                Some(json!({})),
            )))
            .await
            .expect("initialized");
    }

    async fn request(
        server: &AppServer,
        id: i64,
        method: &str,
        params: serde_json::Value,
    ) -> serde_json::Value {
        match server
            .handle_message(JsonRpcMessage::Request(JsonRpcRequest::new(
                RequestId::Integer(id),
                method,
                Some(params),
            )))
            .await
            .expect("handle")
            .remove(0)
        {
            JsonRpcMessage::Response(response) => response.result,
            JsonRpcMessage::Error(error) => panic!("unexpected error: {error:?}"),
            other => panic!("unexpected message: {other:?}"),
        }
    }

    async fn write_json_message(writer: &mut tokio::io::DuplexStream, message: JsonRpcMessage) {
        let line = encode_message(&message).expect("encode");
        writer.write_all(line.as_bytes()).await.expect("write");
        writer.flush().await.expect("flush");
    }

    async fn next_json_message(
        lines: &mut tokio::io::Lines<BufReader<tokio::io::DuplexStream>>,
    ) -> JsonRpcMessage {
        next_json_message_with_timeout(lines, std::time::Duration::from_secs(1)).await
    }

    async fn next_json_message_with_timeout(
        lines: &mut tokio::io::Lines<BufReader<tokio::io::DuplexStream>>,
        timeout: std::time::Duration,
    ) -> JsonRpcMessage {
        let line = tokio::time::timeout(timeout, lines.next_line())
            .await
            .expect("output line timeout")
            .expect("read line")
            .expect("output line");
        decode_message(&line).expect("decode")
    }

    async fn next_json_response_result(
        lines: &mut tokio::io::Lines<BufReader<tokio::io::DuplexStream>>,
        expected_id: RequestId,
    ) -> serde_json::Value {
        loop {
            match next_json_message_with_timeout(lines, std::time::Duration::from_secs(10)).await {
                JsonRpcMessage::Response(response) if response.id == expected_id => {
                    return response.result;
                }
                JsonRpcMessage::Error(error) if error.id == expected_id => {
                    panic!("request {expected_id:?} failed: {:?}", error.error);
                }
                _ => {}
            }
        }
    }

    async fn assert_next_direct_delta_notification(
        lines: &mut tokio::io::Lines<BufReader<tokio::io::DuplexStream>>,
        expected_thread_id: &str,
        expected_turn_id: &str,
        expected_delta: &str,
    ) {
        for _ in 0..16 {
            let message =
                next_json_message_with_timeout(lines, std::time::Duration::from_secs(10)).await;
            let JsonRpcMessage::Notification(notification) = &message else {
                continue;
            };
            if notification.method != "item/agentMessage/delta" {
                continue;
            }
            let params = notification.params.as_ref().expect("params");
            if params["threadId"] != expected_thread_id || params["turnId"] != expected_turn_id {
                continue;
            }
            assert_eq!(params["delta"], expected_delta);
            return;
        }
        panic!("expected direct delta notification for {expected_turn_id}");
    }
    fn node_binary() -> Option<String> {
        let candidates = std::env::var("NODE")
            .ok()
            .into_iter()
            .chain(["node".to_string()]);
        for candidate in candidates {
            if std::process::Command::new(&candidate)
                .arg("--version")
                .output()
                .is_ok_and(|output| output.status.success())
            {
                return Some(candidate);
            }
        }
        None
    }

    fn assert_response_id(message: JsonRpcMessage, expected_id: RequestId) {
        match message {
            JsonRpcMessage::Response(response) => assert_eq!(response.id, expected_id),
            other => panic!("expected response, got {other:?}"),
        }
    }

    #[test]
    fn initialized_requests_spawn_transport_tasks_without_racing_initialize() {
        assert!(should_spawn_transport_request(&JsonRpcMessage::Request(
            JsonRpcRequest::new(
                RequestId::Integer(1),
                METHOD_AGENT_SESSION_MEDIA_READ,
                Some(json!({})),
            ),
        )));
        assert!(should_spawn_transport_request(&JsonRpcMessage::Request(
            JsonRpcRequest::new(RequestId::Integer(2), METHOD_TURN_START, Some(json!({})),),
        )));
        assert!(should_spawn_transport_request(&JsonRpcMessage::Request(
            JsonRpcRequest::new(RequestId::Integer(3), METHOD_THREAD_START, Some(json!({}))),
        )));
        assert!(!should_spawn_transport_request(&JsonRpcMessage::Request(
            JsonRpcRequest::new(RequestId::Integer(4), METHOD_INITIALIZE, Some(json!({}))),
        )));
        assert!(!should_spawn_transport_request(
            &JsonRpcMessage::Notification(JsonRpcNotification::new(
                METHOD_INITIALIZED,
                Some(json!({})),
            )),
        ));
    }

    #[test]
    fn media_read_streaming_transport_requires_stream_flag() {
        assert!(should_stream_transport_request(&JsonRpcMessage::Request(
            JsonRpcRequest::new(
                RequestId::Integer(1),
                METHOD_AGENT_SESSION_MEDIA_READ,
                Some(json!({ "stream": true })),
            ),
        )));
        assert!(!should_stream_transport_request(&JsonRpcMessage::Request(
            JsonRpcRequest::new(
                RequestId::Integer(2),
                METHOD_AGENT_SESSION_MEDIA_READ,
                Some(json!({})),
            ),
        )));
    }

    #[tokio::test]
    async fn slow_websocket_broadcast_cancels_connection_when_outbound_queue_is_full() {
        let server = AppServer::new();
        let (writer, mut queued) = mpsc::channel(1);
        let first = JsonRpcMessage::Notification(JsonRpcNotification::new(
            "thread/status/changed",
            Some(json!({ "status": "busy" })),
        ));
        writer
            .send(QueuedOutgoingMessage::new(OutgoingMessage::from(
                first.clone(),
            )))
            .await
            .expect("fill websocket queue");
        let disconnect_sender = CancellationToken::new();
        server.register_transport_writer(ConnectionId(99), writer, Some(disconnect_sender.clone()));
        server.mark_transport_initialized(ConnectionId(99));

        enqueue_transport_outbound_message(
            &server.transport_writers,
            &server.transport_disconnects,
            &server.transport_initialized,
            &server.transport_notification_opt_out,
            JsonRpcMessage::Notification(JsonRpcNotification::new(
                "thread/status/changed",
                Some(json!({ "status": "idle" })),
            )),
        );

        assert!(disconnect_sender.is_cancelled());
        let queued_message = queued.try_recv().expect("original queued message");
        assert_eq!(queued_message.message.into_json_rpc_message(), first);
    }

    #[tokio::test]
    async fn broadcast_notifications_wait_for_connection_initialize() {
        let server = AppServer::new();
        let (writer, mut queued) = mpsc::channel(4);
        let connection_id = ConnectionId(100);
        server.register_transport_writer(connection_id, writer, None);
        let notification = JsonRpcMessage::Notification(JsonRpcNotification::new(
            "thread/status/changed",
            Some(json!({ "status": "busy" })),
        ));

        enqueue_transport_outbound_message(
            &server.transport_writers,
            &server.transport_disconnects,
            &server.transport_initialized,
            &server.transport_notification_opt_out,
            notification.clone(),
        );
        assert!(matches!(
            queued.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));

        server.mark_transport_initialized(connection_id);
        enqueue_transport_outbound_message(
            &server.transport_writers,
            &server.transport_disconnects,
            &server.transport_initialized,
            &server.transport_notification_opt_out,
            notification.clone(),
        );
        let queued_message = queued.recv().await.expect("initialized notification");
        assert_eq!(queued_message.message.into_json_rpc_message(), notification);
    }

    #[tokio::test]
    async fn broadcast_notifications_respect_connection_opt_out_methods() {
        let server = AppServer::new();
        let (writer, mut queued) = mpsc::channel(4);
        let connection_id = ConnectionId(101);
        server.register_transport_writer(connection_id, writer, None);
        server.set_transport_notification_opt_out(
            connection_id,
            Some(vec!["thread/status/changed".to_string()]),
        );
        server.mark_transport_initialized(connection_id);

        enqueue_transport_outbound_message(
            &server.transport_writers,
            &server.transport_disconnects,
            &server.transport_initialized,
            &server.transport_notification_opt_out,
            JsonRpcMessage::Notification(JsonRpcNotification::new(
                "thread/status/changed",
                Some(json!({ "status": "busy" })),
            )),
        );
        assert!(matches!(
            queued.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));

        enqueue_transport_outbound_message(
            &server.transport_writers,
            &server.transport_disconnects,
            &server.transport_initialized,
            &server.transport_notification_opt_out,
            JsonRpcMessage::Notification(JsonRpcNotification::new(
                "thread/started",
                Some(json!({ "threadId": "thread-1" })),
            )),
        );
        assert!(queued.recv().await.is_some());
    }

    #[tokio::test]
    async fn thread_listener_orders_resume_replay_before_scoped_live_event() {
        let server = AppServer::new();
        let subscribed_connection = ConnectionId(201);
        let unrelated_connection = ConnectionId(202);
        let (subscribed_writer, mut subscribed_messages) = mpsc::channel(8);
        let (unrelated_writer, mut unrelated_messages) = mpsc::channel(8);

        server.register_transport_writer(subscribed_connection, subscribed_writer, None);
        server.register_transport_writer(unrelated_connection, unrelated_writer, None);
        server.mark_transport_initialized(subscribed_connection);
        server.mark_transport_initialized(unrelated_connection);
        server
            .thread_states
            .connection_initialized(subscribed_connection)
            .await;
        server
            .thread_states
            .connection_initialized(unrelated_connection)
            .await;

        let thread_id = agent_protocol::ThreadId::new("thread-ordered");
        let response = JsonRpcMessage::Response(app_server_protocol::JsonRpcResponse {
            id: RequestId::String("resume-1".to_string()),
            result: json!({ "thread": { "id": thread_id.as_str() } }),
        });
        let pending_request = JsonRpcMessage::Request(JsonRpcRequest::new(
            RequestId::String("pending-1".to_string()),
            "item/tool/requestUserInput",
            Some(json!({ "threadId": thread_id.as_str() })),
        ));
        let usage = JsonRpcMessage::Notification(thread_token_usage_notification(
            thread_id.as_str(),
            runtime::thread_usage::ThreadTokenUsageSnapshot {
                turn_id: "turn-usage".to_string(),
                total_token_usage: runtime::thread_usage::TokenUsageSnapshot {
                    input_tokens: 90,
                    cached_input_tokens: 30,
                    cache_write_input_tokens: 12,
                    output_tokens: 30,
                    reasoning_output_tokens: 10,
                    total_tokens: 120,
                },
                last_token_usage: runtime::thread_usage::TokenUsageSnapshot {
                    input_tokens: 45,
                    cached_input_tokens: 15,
                    cache_write_input_tokens: 6,
                    output_tokens: 15,
                    reasoning_output_tokens: 5,
                    total_tokens: 60,
                },
                model_context_window: Some(128_000),
                source_sequence: 7,
            },
        ));
        let goal = JsonRpcMessage::Notification(
            app_server_protocol::protocol::v2::ServerNotification::ThreadGoalUpdated(
                app_server_protocol::protocol::v2::ThreadGoalUpdatedNotification {
                    thread_id: thread_id.as_str().to_string(),
                    turn_id: None,
                    goal: app_server_protocol::protocol::v2::ThreadGoal {
                        thread_id: thread_id.as_str().to_string(),
                        objective: "finish ordered resume".to_string(),
                        status: app_server_protocol::protocol::v2::ThreadGoalStatus::Active,
                        token_budget: Some(1_000),
                        tokens_used: 120,
                        time_used_seconds: 3,
                        created_at: 1,
                        updated_at: 2,
                    },
                },
            )
            .into(),
        );
        let bridge = server.event_bridge();
        let barrier = thread_state::ThreadResumeBarrier::new(
            subscribed_connection,
            RequestId::String("resume-1".to_string()),
        );
        bridge
            .prepare_thread_resume(thread_id.clone(), barrier.clone())
            .await
            .expect("prepare resume barrier");

        bridge
            .send_thread_command(
                thread_id.clone(),
                thread_state::ThreadListenerCommand::PublishRuntimeEvent {
                    event: AgentEvent {
                        event_id: "event-live-1".to_string(),
                        sequence: 1,
                        session_id: "session-ordered".to_string(),
                        thread_id: Some(thread_id.as_str().to_string()),
                        turn_id: Some("turn-1".to_string()),
                        event_type: "provider.step".to_string(),
                        timestamp: "2026-07-19T00:00:00Z".to_string(),
                        payload: json!({ "phase": "streaming" }),
                    },
                    completion_tx: None,
                },
            )
            .await
            .expect("queue live event behind resume barrier");

        let (completion_tx, completion_rx) = oneshot::channel();
        bridge
            .send_thread_command(
                thread_id.clone(),
                thread_state::ThreadListenerCommand::CompleteResume {
                    barrier,
                    connection_id: subscribed_connection,
                    messages: vec![
                        response.clone(),
                        usage.clone(),
                        goal.clone(),
                        pending_request.clone(),
                    ],
                    subscribe: true,
                    completion_tx,
                },
            )
            .await
            .expect("complete resume barrier");
        completion_rx
            .await
            .expect("listener completion")
            .expect("send resume response and live event");

        let first = subscribed_messages
            .recv()
            .await
            .expect("resume response")
            .message
            .into_json_rpc_message();
        let second = subscribed_messages
            .recv()
            .await
            .expect("token usage replay")
            .message
            .into_json_rpc_message();
        let third = subscribed_messages
            .recv()
            .await
            .expect("goal snapshot")
            .message
            .into_json_rpc_message();
        let fourth = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            subscribed_messages.recv(),
        )
        .await
        .expect("pending replay timeout")
        .expect("pending replay")
        .message
        .into_json_rpc_message();
        let fifth = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            subscribed_messages.recv(),
        )
        .await
        .expect("live event timeout")
        .expect("live event")
        .message
        .into_json_rpc_message();

        assert_eq!(first, response);
        assert_eq!(second, usage);
        let replayed_usage = serde_json::to_value(&second).expect("serialize usage replay");
        assert_eq!(
            replayed_usage.pointer("/params/tokenUsage/total/cacheWriteInputTokens"),
            Some(&json!(12))
        );
        assert_eq!(
            replayed_usage.pointer("/params/tokenUsage/last/cacheWriteInputTokens"),
            Some(&json!(6))
        );
        assert_eq!(third, goal);
        assert_eq!(fourth, pending_request);
        assert!(matches!(fifth, JsonRpcMessage::Notification(_)));
        assert!(matches!(
            unrelated_messages.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
    }
}
