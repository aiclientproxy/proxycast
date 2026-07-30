mod agent_control;
mod agent_control_gateway;
mod agent_control_gateway_support;
mod agent_mailbox_delivery;
mod agent_terminal_activity;
mod app_data;
pub(crate) mod approval_cache;
mod approval_decision_contract;
mod article_workspace_action_projection;
mod article_workspace_artifact_document_projection;
mod article_workspace_projection;
mod artifact_content;
mod artifact_document_versions;
mod artifact_projection;
mod artifact_reader;
mod artifact_sidecar;
mod automation;
mod backend;
mod background_terminal;
mod browser_session;
mod canonical_rollout;
mod canonical_thread_store;
#[cfg(test)]
mod canonical_thread_store_tests;
mod capabilities;
mod coding_activity_projection;
mod connect;
mod context_auto_compaction;
mod context_compaction;
mod context_media;
mod context_packet;
mod conversation_import;
mod diagnostics;
mod error;
mod event_log;
mod event_sink;
mod event_store;
mod evidence_provider;
mod execution_request;
mod expert_role_switch;
mod exports;
mod file_checkpoint_projection;
mod file_system;
mod gateway;
mod gateway_runner;
mod image_media;
mod input_media;
mod inter_agent_input;
mod knowledge;
mod load_context;
mod mcp;
mod media_task_read_model;
mod media_tasks;
mod memory;
pub(crate) mod memory_prompt;
mod model_providers;
mod output_media;
mod output_refs;
pub(crate) mod pending_action_descriptor;
mod permission_state_projection;
mod plugin_host_lifecycle;
mod plugin_task_runtime;
mod plugin_worker_orchestration;
mod plugin_worker_output_contract;
mod plugin_worker_runtime;
mod plugin_worker_streaming;
mod plugin_worker_turn;
mod plugin_worker_workflow;
mod plugin_worker_workflow_cancel;
mod plugin_worker_workflow_hooks;
mod plugin_worker_workflow_retry;
mod plugins;
mod project_git;
mod projection_item_events;
mod projection_payload_summary;
mod projection_protocol;
mod projection_rebuild;
mod projection_repair;
#[cfg(test)]
mod projection_repair_tests;
mod projection_schema;
mod projection_status;
mod projection_store;
#[cfg(test)]
mod projection_store_tests;
pub(crate) mod provider_history;
#[doc(hidden)]
pub use provider_history::ProviderTurnHistory;
mod queued_turn_intent;
mod read_model;
mod read_model_turn_usage;
mod right_surface;
mod service_projection;
mod session_control;
mod session_files;
mod session_lifecycle;
pub(crate) mod session_list_scope;
mod session_media_reader;
mod session_media_refs;
mod session_operations;
mod session_runtime_defaults;
mod session_shell;
mod session_submission;
pub(crate) mod session_title;
pub(crate) mod sidecar_store;
mod skills;
mod soul;
mod status;
mod storage_roots;
mod thread_delete;
mod thread_elicitation;
mod thread_fork;
mod thread_goal;
mod thread_goal_continuation;
mod thread_guardian;
mod thread_inject_items;
mod thread_item_projection;
mod thread_read;
mod thread_search;
pub(crate) use thread_read::ThreadUnloadResult;
pub(crate) mod thread_usage;
mod tool_item_projection;
mod tool_lifecycle;
mod trace;
mod trace_store;
mod turn_execution;
mod turn_input_events;
mod turn_start;
mod usage_stats;
mod value_fields;
mod voice;
pub(crate) mod workflow;
mod workspaces;
use crate::execution_process::ExecutionProcessServer;
pub use crate::file_checkpoint_snapshot::FileCheckpointSnapshotReadRequest;
pub use crate::file_checkpoint_snapshot::FileCheckpointSnapshotRecord;
pub use crate::file_checkpoint_snapshot::FileCheckpointSnapshotSaveRequest;
pub use crate::file_checkpoint_snapshot::FileCheckpointSnapshotStore;
pub use crate::file_checkpoint_snapshot::FilesystemFileCheckpointSnapshotStore;
pub use crate::file_checkpoint_snapshot::NoopFileCheckpointSnapshotStore;
pub use app_data::AppDataSource;
pub use app_data::AutomationManagementAppDataSource;
pub use app_data::AutomationOverviewAppDataSource;
pub use app_data::ConnectAppDataSource;
pub use app_data::DiagnosticsAppDataSource;
pub use app_data::GatewayAppDataSource;
pub use app_data::KnowledgeAppDataSource;
pub use app_data::McpAppDataSource;
pub use app_data::MediaAppDataSource;
pub use app_data::MemoryAppDataSource;
pub use app_data::NoopAppDataSource;
pub use app_data::PluginDataSource;
pub use app_data::RightSurfaceAppDataSource;
pub use app_data::SessionAppDataSource;
pub use app_data::SkillAppDataSource;
pub use app_data::UsageStatsAppDataSource;
pub use app_data::VoiceAppDataSource;
pub use app_data::WorkspaceAppDataSource;
pub use app_data::WorkspaceObjectCanvasSnapshot;
pub use app_data::WorkspaceObjectCanvasSnapshotListParams;
pub use app_data::WorkspaceSkillBindingAppDataSource;
pub use app_data::{ModelCatalogQuery, ModelProviderAppDataSource, ProviderModelCatalog};
pub use artifact_content::FilesystemArtifactContentProvider;
pub use artifact_content::InlineArtifactContentProvider;
pub use backend::MockBackend;
pub use backend::UnavailableBackend;
pub use error::RuntimeCoreError;
pub use event_log::EventLogRecord;
pub use event_log::EventLogWriter;
use event_sink::RuntimeEventCallback;
pub use event_sink::{RuntimeEvent, RuntimeEventHub, RuntimeEventSink};
pub use evidence_provider::BasicEvidenceExportProvider;
pub use evidence_provider::NoopEvidenceExportProvider;
pub use execution_request::ExecutionRequest;
pub use output_refs::FilesystemOutputSnapshotStore;
pub use output_refs::NoopOutputSnapshotStore;
pub use output_refs::OutputSnapshotReadRequest;
pub use output_refs::OutputSnapshotRecord;
pub use output_refs::OutputSnapshotSaveRequest;
pub use output_refs::OutputSnapshotStore;
pub(crate) use plugin_worker_streaming::ensure_workspace_patch_artifact_paths;
pub use projection_repair::ProjectionRepair;
pub use projection_store::ProjectionStore;
pub use right_surface::WorkspaceObjectCanvasReplayReadiness;
pub use right_surface::WorkspaceObjectCanvasReplayReadinessListParams;
pub use sidecar_store::SidecarRef;
pub use sidecar_store::SidecarStore;
pub use sidecar_store::SidecarWriteRequest;
pub use soul::install_style_pack_data_root;
pub use storage_roots::product_db_path_for_agent_root;
pub use storage_roots::StorageRoots;
pub(crate) use trace_store::export_trace_events_from_store_to_path;
pub(crate) use trace_store::summarize_trace_event_store;
pub use trace_store::TraceEventWriter;
pub(crate) use trace_store::TRACE_EVENT_MAX_FILES_PER_SESSION;
pub use turn_start::TurnStartRequest;
pub(super) use value_fields::event_request_id;

use crate::CapabilityInventorySource;
use crate::CapabilitySource;
use crate::KnowledgeBuilderRuntimeExecutor;
use crate::NativeKnowledgeBuilderRuntimeExecutor;
use agent_protocol::AgentInput;
use agent_runtime::session_loop::RuntimeSessionInputHandle;
use app_server_protocol::AgentEvent;
use app_server_protocol::AgentSession;
use app_server_protocol::AgentSessionActionScope;
use app_server_protocol::AgentSessionActionType;
use app_server_protocol::AgentSessionApprovalDecision;
use app_server_protocol::AgentTurn;
use app_server_protocol::ArtifactSummary;
use app_server_protocol::ClientInfo;
use app_server_protocol::EvidencePackSummary;
use app_server_protocol::RuntimeOptions;
use async_trait::async_trait;
use lime_browser_runtime::{BrowserProfileScope, BrowserRuntimeManager};
use lime_infra::telemetry::RequestLog;
use lime_infra::telemetry::TelemetryStore;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::Mutex;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use value_fields::{
    json_string, metadata_string, new_id, optional_id_or_new, raw_string_field, string_array_field,
    string_field, timestamp, timestamp_seconds,
};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeHostContext {
    pub client_name: Option<String>,
    pub client_version: Option<String>,
}

impl From<Option<ClientInfo>> for RuntimeHostContext {
    fn from(client_info: Option<ClientInfo>) -> Self {
        match client_info {
            Some(client_info) => Self {
                client_name: Some(client_info.name),
                client_version: client_info.version,
            },
            None => Self::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArtifactContentRequest {
    pub session: AgentSession,
    pub artifact: ArtifactSummary,
}

pub trait ArtifactContentProvider: Send + Sync {
    fn read_content(&self, request: &ArtifactContentRequest) -> Option<String>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvidencePackRequest {
    pub session: AgentSession,
    pub turns: Vec<AgentTurn>,
    pub events: Vec<AgentEvent>,
    pub artifacts: Vec<ArtifactSummary>,
    pub turn_runtime_metadata: BTreeMap<String, serde_json::Value>,
    pub request_logs: Vec<RequestLog>,
    pub workflow_audit_events: Vec<AgentEvent>,
}

#[async_trait]
pub trait EvidenceExportProvider: Send + Sync {
    async fn export_evidence_pack(
        &self,
        request: &EvidencePackRequest,
    ) -> Result<Option<EvidencePackSummary>, RuntimeCoreError>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeCoreOutput<T> {
    pub response: T,
    pub events: Vec<AgentEvent>,
}

#[derive(Debug, Clone)]
pub struct CancelExecutionRequest {
    pub host: RuntimeHostContext,
    pub session: AgentSession,
    pub turn: AgentTurn,
}

#[derive(Debug, Clone)]
pub struct ActionRespondRequest {
    pub host: RuntimeHostContext,
    pub session: AgentSession,
    pub turn: Option<AgentTurn>,
    pub request_id: String,
    pub action_type: AgentSessionActionType,
    pub decision: Option<AgentSessionApprovalDecision>,
    pub confirmed: bool,
    pub response: Option<String>,
    pub user_data: Option<serde_json::Value>,
    pub metadata: Option<serde_json::Value>,
    pub event_name: Option<String>,
    pub action_scope: Option<AgentSessionActionScope>,
    pub pending_action_descriptor: Option<agent_runtime::action_required::PendingActionDescriptor>,
}

#[derive(Debug, Clone)]
pub struct PermissionRespondRequest {
    pub session_id: String,
    pub thread_id: String,
    pub turn_id: String,
    pub request_id: String,
    pub response: app_server_protocol::protocol::v2::PermissionsRequestApprovalResponse,
}

#[derive(Debug, Clone)]
pub struct DynamicToolRespondRequest {
    pub session_id: String,
    pub thread_id: String,
    pub turn_id: String,
    pub call_id: String,
    pub response: app_server_protocol::protocol::v2::DynamicToolCallResponse,
}

impl ActionRespondRequest {
    pub fn runtime_metadata(&self) -> Option<&serde_json::Value> {
        self.metadata.as_ref()
    }
}

#[derive(Debug, Clone)]
pub struct ToolInventoryReadRequest {
    pub caller: Option<String>,
    pub workbench: bool,
    pub browser_assist: bool,
    pub metadata: Option<serde_json::Value>,
}

#[async_trait]
pub trait ExecutionBackend: Send + Sync {
    fn requires_provider_selection(&self) -> bool {
        false
    }

    fn has_live_session_responses(&self) -> bool {
        false
    }

    fn set_app_data_source(
        &self,
        _app_data_source: Arc<dyn AppDataSource>,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    fn set_current_time_gateway(
        &self,
        _gateway: Arc<dyn tool_runtime::current_time::CurrentTimeGateway>,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    fn effective_turn_runtime_options(
        &self,
        request: &ExecutionRequest,
        _first_sampling_turn: bool,
    ) -> Option<app_server_protocol::RuntimeOptions> {
        request.runtime_options.clone()
    }

    async fn preflight_turn(
        &self,
        _request: &ExecutionRequest,
        _first_sampling_turn: bool,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn preflight_thread_settings(
        &self,
        _session: &AgentSession,
        _settings: &app_server_protocol::protocol::v2::ThreadSettings,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn prepare_turn_runtime_options(
        &self,
        request: &ExecutionRequest,
        first_sampling_turn: bool,
    ) -> Result<Option<RuntimeOptions>, RuntimeCoreError> {
        self.preflight_turn(request, first_sampling_turn).await?;
        Ok(request.runtime_options.clone())
    }

    async fn start_turn(
        &self,
        request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError>;

    async fn start_turn_with_provider_history(
        &self,
        request: ExecutionRequest,
        _provider_history: ProviderTurnHistory,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.start_turn(request, sink).await
    }

    async fn start_turn_with_provider_history_and_session_input(
        &self,
        request: ExecutionRequest,
        provider_history: ProviderTurnHistory,
        _pending_input: Option<RuntimeSessionInputHandle>,
        _cancellation_token: Option<CancellationToken>,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.start_turn_with_provider_history(request, provider_history, sink)
            .await
    }

    async fn cancel_turn(
        &self,
        request: CancelExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError>;

    async fn close_session(
        &self,
        _session_id: &str,
        _thread_id: &str,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn respond_action(
        &self,
        request: ActionRespondRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError>;

    async fn resolve_permission_action(
        &self,
        _request: &PermissionRespondRequest,
    ) -> Result<(), RuntimeCoreError> {
        Err(RuntimeCoreError::Backend(
            "runtime backend does not support permission responses".to_string(),
        ))
    }

    async fn read_tool_inventory(
        &self,
        _request: ToolInventoryReadRequest,
    ) -> Result<serde_json::Value, RuntimeCoreError> {
        Err(RuntimeCoreError::Backend(
            "runtime backend does not expose tool inventory".to_string(),
        ))
    }

    async fn prepare_runtime_worker_artifact_events(
        &self,
        _request: &ExecutionRequest,
        _events: &mut Vec<RuntimeEvent>,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn prepare_plugin_worker_request(
        &self,
        _request: &ExecutionRequest,
        _worker_request: &mut serde_json::Value,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }
}

#[derive(Clone)]
pub struct RuntimeCore {
    pub(in crate::runtime) state: Arc<Mutex<RuntimeCoreState>>,
    pub(in crate::runtime) session_loops: agent_runtime::session_loop::RuntimeSessionRegistry,
    pub(in crate::runtime) turn_driver_completions: turn_execution::RuntimeTurnDriverCompletions,
    mailbox_trigger_flights: agent_mailbox_delivery::MailboxTriggerFlights,
    route_recovery: model_providers::RouteRecoveryCoordinator,
    catalog_refresh: model_providers::CatalogRefreshCoordinator,
    backend: Arc<dyn ExecutionBackend>,
    capability_source: Arc<dyn CapabilitySource>,
    pub(in crate::runtime) artifact_content_provider: Arc<dyn ArtifactContentProvider>,
    pub(in crate::runtime) file_checkpoint_snapshot_store: Arc<dyn FileCheckpointSnapshotStore>,
    pub(in crate::runtime) output_snapshot_store: Arc<dyn OutputSnapshotStore>,
    pub(in crate::runtime) sidecar_store: Option<Arc<SidecarStore>>,
    pub(in crate::runtime) event_log_writer: Option<Arc<EventLogWriter>>,
    pub(in crate::runtime) trace_event_writer: Option<Arc<TraceEventWriter>>,
    pub(in crate::runtime) projection_store: Option<Arc<ProjectionStore>>,
    pub(in crate::runtime) telemetry_store: Option<Arc<TelemetryStore>>,
    pub(in crate::runtime) event_hub: RuntimeEventHub,
    pub(in crate::runtime) browser_runtime: Arc<BrowserRuntimeManager>,
    evidence_export_provider: Arc<dyn EvidenceExportProvider>,
    knowledge_builder_runtime_executor: Arc<dyn KnowledgeBuilderRuntimeExecutor>,
    app_data_source: Arc<dyn AppDataSource>,
    pub(crate) execution_process_server: Option<ExecutionProcessServer>,
}

#[derive(Clone)]
pub struct RuntimeCoreEventAppender {
    state: Arc<Mutex<RuntimeCoreState>>,
    file_checkpoint_snapshot_store: Arc<dyn FileCheckpointSnapshotStore>,
    output_snapshot_store: Arc<dyn OutputSnapshotStore>,
    sidecar_store: Option<Arc<SidecarStore>>,
    event_log_writer: Option<Arc<EventLogWriter>>,
    trace_event_writer: Option<Arc<TraceEventWriter>>,
    projection_store: Option<Arc<ProjectionStore>>,
    session_loops: agent_runtime::session_loop::RuntimeSessionRegistry,
}

#[derive(Debug, Default)]
pub(in crate::runtime) struct RuntimeCoreState {
    pub(in crate::runtime) sessions: HashMap<String, StoredSession>,
    pub(in crate::runtime) thread_elicitation_counts: HashMap<String, i64>,
    pub(in crate::runtime) thread_goal_continuations: HashSet<String>,
    pub(in crate::runtime) import_jobs: HashMap<String, conversation_import::ImportJobRecord>,
    pub(in crate::runtime) session_approval_cache: approval_cache::SessionApprovalCache,
    pub(in crate::runtime) right_surface_pending:
        Vec<app_server_protocol::WorkspaceRightSurfacePendingRequest>,
    pub(in crate::runtime) browser_profile_scopes: Vec<BrowserProfileScope>,
    plugin_ui_runtimes: HashMap<String, plugins::PluginUiRuntimeProcess>,
}

#[derive(Debug, Clone)]
pub(in crate::runtime) struct StoredSession {
    pub(in crate::runtime) session: AgentSession,
    pub(in crate::runtime) turns: Vec<AgentTurn>,
    pub(in crate::runtime) turn_inputs: HashMap<String, Vec<AgentInput>>,
    pub(in crate::runtime) turn_runtime_options:
        HashMap<String, app_server_protocol::RuntimeOptions>,
    pub(in crate::runtime) events: Vec<AgentEvent>,
    pub(in crate::runtime) output_blobs: HashMap<String, output_refs::OutputBlobRecord>,
}

impl Default for RuntimeCore {
    fn default() -> Self {
        Self::with_backend(Arc::new(MockBackend))
    }
}

impl RuntimeCore {
    pub fn with_backend(backend: Arc<dyn ExecutionBackend>) -> Self {
        Self::with_backend_and_capability_source(
            backend,
            Arc::new(CapabilityInventorySource::default()),
        )
    }

    pub fn with_backend_and_capability_source(
        backend: Arc<dyn ExecutionBackend>,
        capability_source: Arc<dyn CapabilitySource>,
    ) -> Self {
        Self::with_backend_capability_source_and_artifact_content_provider(
            backend,
            capability_source,
            Arc::new(InlineArtifactContentProvider),
        )
    }

    pub fn with_backend_capability_source_and_artifact_content_provider(
        backend: Arc<dyn ExecutionBackend>,
        capability_source: Arc<dyn CapabilitySource>,
        artifact_content_provider: Arc<dyn ArtifactContentProvider>,
    ) -> Self {
        Self::with_backend_capability_source_artifact_content_provider_and_evidence_export_provider(
            backend,
            capability_source,
            artifact_content_provider,
            Arc::new(BasicEvidenceExportProvider),
        )
    }

    pub fn with_backend_capability_source_artifact_content_provider_and_evidence_export_provider(
        backend: Arc<dyn ExecutionBackend>,
        capability_source: Arc<dyn CapabilitySource>,
        artifact_content_provider: Arc<dyn ArtifactContentProvider>,
        evidence_export_provider: Arc<dyn EvidenceExportProvider>,
    ) -> Self {
        Self {
            state: Arc::new(Mutex::new(RuntimeCoreState::default())),
            session_loops: agent_runtime::session_loop::RuntimeSessionRegistry::default(),
            turn_driver_completions: turn_execution::RuntimeTurnDriverCompletions::default(),
            mailbox_trigger_flights: agent_mailbox_delivery::MailboxTriggerFlights::default(),
            route_recovery: model_providers::RouteRecoveryCoordinator::default(),
            catalog_refresh: model_providers::CatalogRefreshCoordinator::default(),
            backend,
            capability_source,
            artifact_content_provider,
            file_checkpoint_snapshot_store: Arc::new(NoopFileCheckpointSnapshotStore),
            output_snapshot_store: Arc::new(NoopOutputSnapshotStore),
            sidecar_store: None,
            event_log_writer: None,
            trace_event_writer: None,
            projection_store: None,
            telemetry_store: None,
            event_hub: RuntimeEventHub::new(),
            browser_runtime: Arc::new(BrowserRuntimeManager::new()),
            evidence_export_provider,
            knowledge_builder_runtime_executor: Arc::new(
                NativeKnowledgeBuilderRuntimeExecutor::new(),
            ),
            app_data_source: Arc::new(NoopAppDataSource),
            execution_process_server: None,
        }
    }

    pub(crate) fn take_event_receiver(&self) -> Option<mpsc::UnboundedReceiver<AgentEvent>> {
        self.event_hub.take_receiver()
    }

    pub(crate) fn with_execution_process_server(
        mut self,
        execution_process_server: ExecutionProcessServer,
    ) -> Self {
        self.execution_process_server = Some(execution_process_server);
        self
    }

    pub(crate) fn execution_process_server(&self) -> Option<ExecutionProcessServer> {
        self.execution_process_server.clone()
    }

    pub fn with_app_data_source(mut self, app_data_source: Arc<dyn AppDataSource>) -> Self {
        if let Err(error) = self.backend.set_app_data_source(app_data_source.clone()) {
            tracing::warn!(
                "runtime backend rejected app data source injection; app data source remains available to RuntimeCore: {error}"
            );
        }
        self.app_data_source = app_data_source;
        self
    }

    pub(crate) fn set_current_time_gateway(
        &self,
        gateway: Arc<dyn tool_runtime::current_time::CurrentTimeGateway>,
    ) -> Result<(), RuntimeCoreError> {
        self.backend.set_current_time_gateway(gateway)
    }

    pub fn with_output_snapshot_store(
        mut self,
        output_snapshot_store: Arc<dyn OutputSnapshotStore>,
    ) -> Self {
        self.output_snapshot_store = output_snapshot_store;
        self
    }

    pub fn with_file_checkpoint_snapshot_store(
        mut self,
        file_checkpoint_snapshot_store: Arc<dyn FileCheckpointSnapshotStore>,
    ) -> Self {
        self.file_checkpoint_snapshot_store = file_checkpoint_snapshot_store;
        self
    }

    pub fn with_sidecar_store(mut self, sidecar_store: Arc<SidecarStore>) -> Self {
        self.sidecar_store = Some(sidecar_store);
        self
    }

    pub fn with_event_log_writer(mut self, event_log_writer: Arc<EventLogWriter>) -> Self {
        self.event_log_writer = Some(event_log_writer);
        self
    }

    pub fn with_trace_event_writer(mut self, trace_event_writer: Arc<TraceEventWriter>) -> Self {
        self.trace_event_writer = Some(trace_event_writer);
        self
    }

    pub fn with_projection_store(mut self, projection_store: Arc<ProjectionStore>) -> Self {
        self.projection_store = Some(projection_store);
        self
    }

    pub fn with_telemetry_store(mut self, telemetry_store: Arc<TelemetryStore>) -> Self {
        self.telemetry_store = Some(telemetry_store);
        self
    }

    pub fn with_knowledge_builder_runtime_executor(
        mut self,
        executor: Arc<dyn KnowledgeBuilderRuntimeExecutor>,
    ) -> Self {
        self.knowledge_builder_runtime_executor = executor;
        self
    }

    pub fn event_appender(&self) -> RuntimeCoreEventAppender {
        RuntimeCoreEventAppender {
            state: self.state.clone(),
            file_checkpoint_snapshot_store: self.file_checkpoint_snapshot_store.clone(),
            output_snapshot_store: self.output_snapshot_store.clone(),
            sidecar_store: self.sidecar_store.clone(),
            event_log_writer: self.event_log_writer.clone(),
            trace_event_writer: self.trace_event_writer.clone(),
            projection_store: self.projection_store.clone(),
            session_loops: self.session_loops.clone(),
        }
    }
}

#[cfg(test)]
mod tests;
