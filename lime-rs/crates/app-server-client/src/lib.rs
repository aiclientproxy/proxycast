pub use app_server_protocol::app_server_method_catalog;
pub use app_server_protocol::is_app_server_notification_method;
pub use app_server_protocol::is_app_server_request_method;
pub use app_server_protocol::protocol::v2::ServerNotification;
pub use app_server_protocol::protocol::v2::METHOD_THREAD_ARCHIVE;
pub use app_server_protocol::protocol::v2::METHOD_THREAD_SHELL_COMMAND;
pub use app_server_protocol::protocol::v2::METHOD_THREAD_UNARCHIVE;
use app_server_protocol::protocol::v2::NOTIFICATION_METHODS;
pub use app_server_protocol::protocol::v2::{
    ThreadArchiveParams, ThreadArchiveResponse, ThreadShellCommandParams,
    ThreadShellCommandResponse, ThreadUnarchiveParams, ThreadUnarchiveResponse,
};
pub use app_server_protocol::AgentSessionAnalysisHandoffExportParams;
pub use app_server_protocol::AgentSessionAnalysisHandoffExportResponse;
pub use app_server_protocol::AgentSessionCompactParams;
pub use app_server_protocol::AgentSessionCompactResponse;
pub use app_server_protocol::AgentSessionFileCheckpointDetail;
pub use app_server_protocol::AgentSessionFileCheckpointDiffParams;
pub use app_server_protocol::AgentSessionFileCheckpointDiffResponse;
pub use app_server_protocol::AgentSessionFileCheckpointGetParams;
pub use app_server_protocol::AgentSessionFileCheckpointListParams;
pub use app_server_protocol::AgentSessionFileCheckpointListResponse;
pub use app_server_protocol::AgentSessionFileCheckpointRestoreParams;
pub use app_server_protocol::AgentSessionFileCheckpointRestoreResponse;
pub use app_server_protocol::AgentSessionFileCheckpointSummary;
pub use app_server_protocol::AgentSessionFileCheckpointThreadSummary;
pub use app_server_protocol::AgentSessionHandoffArtifact;
pub use app_server_protocol::AgentSessionHandoffBundleExportParams;
pub use app_server_protocol::AgentSessionHandoffBundleExportResponse;
pub use app_server_protocol::AgentSessionListParams;
pub use app_server_protocol::AgentSessionQueuedTurnPromoteParams;
pub use app_server_protocol::AgentSessionQueuedTurnPromoteResponse;
pub use app_server_protocol::AgentSessionQueuedTurnRemoveParams;
pub use app_server_protocol::AgentSessionQueuedTurnRemoveResponse;
pub use app_server_protocol::AgentSessionReadParams;
pub use app_server_protocol::AgentSessionReplayCaseExportParams;
pub use app_server_protocol::AgentSessionReplayCaseExportResponse;
pub use app_server_protocol::AgentSessionReviewDecision;
pub use app_server_protocol::AgentSessionReviewDecisionSaveParams;
pub use app_server_protocol::AgentSessionReviewDecisionTemplateExportParams;
pub use app_server_protocol::AgentSessionReviewDecisionTemplateExportResponse;
pub use app_server_protocol::AgentSessionStartParams;
pub use app_server_protocol::AgentSessionTurnCancelParams;
pub use app_server_protocol::AgentSessionTurnStartParams;
pub use app_server_protocol::AppServerMethodKind;
pub use app_server_protocol::AppServerMethodSpec;
pub use app_server_protocol::ArtifactReadParams;
pub use app_server_protocol::AutomationJobCreateParams;
pub use app_server_protocol::AutomationJobDeleteResponse;
pub use app_server_protocol::AutomationJobHealthParams;
pub use app_server_protocol::AutomationJobHealthResponse;
pub use app_server_protocol::AutomationJobIdParams;
pub use app_server_protocol::AutomationJobListResponse;
pub use app_server_protocol::AutomationJobReadResponse;
pub use app_server_protocol::AutomationJobRunHistoryParams;
pub use app_server_protocol::AutomationJobRunHistoryResponse;
pub use app_server_protocol::AutomationJobRunNowResponse;
pub use app_server_protocol::AutomationJobUpdateParams;
pub use app_server_protocol::AutomationJobWriteResponse;
pub use app_server_protocol::AutomationScheduleParams;
pub use app_server_protocol::AutomationSchedulePreviewResponse;
pub use app_server_protocol::AutomationScheduleValidateResponse;
pub use app_server_protocol::AutomationSchedulerConfigReadResponse;
pub use app_server_protocol::AutomationSchedulerConfigUpdateParams;
pub use app_server_protocol::AutomationSchedulerConfigUpdateResponse;
pub use app_server_protocol::AutomationSchedulerStatusResponse;
pub use app_server_protocol::BrowserSessionActionExecuteParams;
pub use app_server_protocol::BrowserSessionActionExecuteResponse;
pub use app_server_protocol::BrowserSessionCloseResponse;
pub use app_server_protocol::BrowserSessionEventItem;
pub use app_server_protocol::BrowserSessionEventListParams;
pub use app_server_protocol::BrowserSessionEventListResponse;
pub use app_server_protocol::BrowserSessionIdParams;
pub use app_server_protocol::BrowserSessionOpenParams;
pub use app_server_protocol::BrowserSessionOpenResponse;
pub use app_server_protocol::BrowserSessionPageInfo;
pub use app_server_protocol::BrowserSessionReadResponse;
pub use app_server_protocol::BrowserSessionState;
pub use app_server_protocol::BrowserSessionTargetInfo;
pub use app_server_protocol::BrowserSessionTargetListParams;
pub use app_server_protocol::BrowserSessionTargetListResponse;
pub use app_server_protocol::CapabilityListParams;
pub use app_server_protocol::DiagnosticsCapabilityRoutingMetricsSnapshot;
pub use app_server_protocol::DiagnosticsIdempotencyDiagnostics;
pub use app_server_protocol::DiagnosticsMetricConfig;
pub use app_server_protocol::DiagnosticsRequestDedupDiagnostics;
pub use app_server_protocol::DiagnosticsResponseCacheDiagnostics;
pub use app_server_protocol::DiagnosticsTelemetrySummary;
pub use app_server_protocol::EvidenceExportParams;
pub use app_server_protocol::FileSystemCreateDirectoryParams;
pub use app_server_protocol::FileSystemCreateFileParams;
pub use app_server_protocol::FileSystemDeleteFileParams;
pub use app_server_protocol::FileSystemDirectoryListing;
pub use app_server_protocol::FileSystemFileEntry;
pub use app_server_protocol::FileSystemFilePreview;
pub use app_server_protocol::FileSystemListDirectoryParams;
pub use app_server_protocol::FileSystemMutationResponse;
pub use app_server_protocol::FileSystemReadFilePreviewParams;
pub use app_server_protocol::FileSystemRenameFileParams;
pub use app_server_protocol::GatewayChannelStatusParams;
pub use app_server_protocol::GatewayChannelStatusResponse;
pub use app_server_protocol::GatewayTunnelCloudflaredInstallParams;
pub use app_server_protocol::GatewayTunnelCreateParams;
pub use app_server_protocol::GatewayTunnelSyncWebhookUrlParams;
pub use app_server_protocol::InitializeParams;
use app_server_protocol::JsonRpcErrorResponse;
use app_server_protocol::JsonRpcMessage;
use app_server_protocol::JsonRpcNotification;
use app_server_protocol::JsonRpcRequest;
use app_server_protocol::JsonRpcResponse;
pub use app_server_protocol::KnowledgeCompilePackParams;
pub use app_server_protocol::KnowledgeCompilePackResponse;
pub use app_server_protocol::KnowledgeContextResolutionResponse;
pub use app_server_protocol::KnowledgeImportSourceParams;
pub use app_server_protocol::KnowledgeImportSourceResponse;
pub use app_server_protocol::KnowledgeListPacksParams;
pub use app_server_protocol::KnowledgeListPacksResponse;
pub use app_server_protocol::KnowledgeReadPackParams;
pub use app_server_protocol::KnowledgeReadPackResponse;
pub use app_server_protocol::KnowledgeResolveContextParams;
pub use app_server_protocol::KnowledgeSetDefaultPackParams;
pub use app_server_protocol::KnowledgeSetDefaultPackResponse;
pub use app_server_protocol::KnowledgeUpdatePackStatusParams;
pub use app_server_protocol::KnowledgeUpdatePackStatusResponse;
pub use app_server_protocol::KnowledgeValidateContextRunParams;
pub use app_server_protocol::KnowledgeValidateContextRunResponse;
pub use app_server_protocol::LogClearResponse;
pub use app_server_protocol::LogEntry;
pub use app_server_protocol::LogListResponse;
pub use app_server_protocol::LogPersistedTailParams;
pub use app_server_protocol::LogPersistedTailResponse;
pub use app_server_protocol::LogStorageDiagnosticsResponse;
pub use app_server_protocol::McpPromptGetParams;
pub use app_server_protocol::McpPromptGetResponse;
pub use app_server_protocol::McpPromptListResponse;
pub use app_server_protocol::McpResourceListResponse;
pub use app_server_protocol::McpResourceReadParams;
pub use app_server_protocol::McpResourceReadResponse;
pub use app_server_protocol::McpResourceSubscribeParams;
pub use app_server_protocol::McpResourceSubscriptionResponse;
pub use app_server_protocol::McpResourceUnsubscribeParams;
pub use app_server_protocol::McpServerCreateParams;
pub use app_server_protocol::McpServerDeleteParams;
pub use app_server_protocol::McpServerEnabledSetParams;
pub use app_server_protocol::McpServerImportFromAppParams;
pub use app_server_protocol::McpServerImportFromAppResponse;
pub use app_server_protocol::McpServerLifecycleResponse;
pub use app_server_protocol::McpServerListResponse;
pub use app_server_protocol::McpServerStartParams;
pub use app_server_protocol::McpServerStatusListResponse;
pub use app_server_protocol::McpServerStopParams;
pub use app_server_protocol::McpServerUpdateParams;
pub use app_server_protocol::McpToolCallParams;
pub use app_server_protocol::McpToolCallResponse;
pub use app_server_protocol::McpToolCallWithCallerParams;
pub use app_server_protocol::McpToolListForContextParams;
pub use app_server_protocol::McpToolListResponse;
pub use app_server_protocol::McpToolSearchParams;
pub use app_server_protocol::MediaTaskArtifactAudioCompleteParams;
pub use app_server_protocol::MediaTaskArtifactAudioCreateParams;
pub use app_server_protocol::MediaTaskArtifactImageCreateParams;
pub use app_server_protocol::MediaTaskArtifactListParams;
pub use app_server_protocol::MediaTaskArtifactListResponse;
pub use app_server_protocol::MediaTaskArtifactLookupParams;
pub use app_server_protocol::MediaTaskArtifactResponse;
pub use app_server_protocol::MemoryStoreAddNoteParams;
pub use app_server_protocol::MemoryStoreAddNoteResponse;
pub use app_server_protocol::MemoryStoreConsolidateParams;
pub use app_server_protocol::MemoryStoreConsolidateResponse;
pub use app_server_protocol::MemoryStoreHealthResponse;
pub use app_server_protocol::MemoryStoreIndexRebuildResponse;
pub use app_server_protocol::MemoryStoreListParams;
pub use app_server_protocol::MemoryStoreListResponse;
pub use app_server_protocol::MemoryStoreReadParams;
pub use app_server_protocol::MemoryStoreReadResponse;
pub use app_server_protocol::MemoryStoreResetParams;
pub use app_server_protocol::MemoryStoreResetResponse;
pub use app_server_protocol::MemoryStoreReviewListParams;
pub use app_server_protocol::MemoryStoreReviewListResponse;
pub use app_server_protocol::MemoryStoreReviewNote;
pub use app_server_protocol::MemoryStoreReviewResolveAction;
pub use app_server_protocol::MemoryStoreReviewResolveParams;
pub use app_server_protocol::MemoryStoreReviewResolveResponse;
pub use app_server_protocol::MemoryStoreRootParams;
pub use app_server_protocol::MemoryStoreSearchParams;
pub use app_server_protocol::MemoryStoreSearchResponse;
pub use app_server_protocol::ModelListParams;
pub use app_server_protocol::ModelProviderAliasReadParams;
pub use app_server_protocol::PluginFetchCloudPackageParams;
pub use app_server_protocol::PluginInstalledDisabledSetParams;
pub use app_server_protocol::PluginInstalledListResponse;
pub use app_server_protocol::PluginInstalledSaveParams;
pub use app_server_protocol::PluginLocalPackageExportParams;
pub use app_server_protocol::PluginLocalPackageExportResponse;
pub use app_server_protocol::PluginLocalPackageInspectParams;
pub use app_server_protocol::PluginShellPrepareParams;
pub use app_server_protocol::PluginShellPrepareResponse;
pub use app_server_protocol::PluginUiRuntimeStartParams;
pub use app_server_protocol::PluginUiRuntimeStatusParams;
pub use app_server_protocol::PluginUiRuntimeStatusResponse;
pub use app_server_protocol::PluginUiRuntimeStopParams;
pub use app_server_protocol::PluginUninstallParams;
pub use app_server_protocol::PluginUninstallRehearsalParams;
pub use app_server_protocol::ProjectMemoryReadParams;
pub use app_server_protocol::ProjectMemoryReadResponse;
use app_server_protocol::RequestId;
pub use app_server_protocol::ServerDiagnosticsResponse;
pub use app_server_protocol::SkillDownloadInstallParams;
pub use app_server_protocol::SkillDownloadInstallResponse;
pub use app_server_protocol::SkillListResponse;
pub use app_server_protocol::SkillLocalDetailInspectParams;
pub use app_server_protocol::SkillLocalDetailInspectResponse;
pub use app_server_protocol::SkillLocalRenameParams;
pub use app_server_protocol::SkillLocalRenameResponse;
pub use app_server_protocol::SkillManagementListResponse;
pub use app_server_protocol::SkillMarketplaceInstallParams;
pub use app_server_protocol::SkillMarketplaceInstallResponse;
pub use app_server_protocol::SkillPackageExportParams;
pub use app_server_protocol::SkillPackageExportResponse;
pub use app_server_protocol::SkillPackageLocalInspectParams;
pub use app_server_protocol::SkillPackageLocalInspectResponse;
pub use app_server_protocol::SkillPackageLocalInstallParams;
pub use app_server_protocol::SkillPackageLocalInstallResponse;
pub use app_server_protocol::SkillPackageLocalReplaceParams;
pub use app_server_protocol::SkillPackageLocalReplaceResponse;
pub use app_server_protocol::SkillReadParams;
pub use app_server_protocol::SkillReadResponse;
pub use app_server_protocol::SupportBundleExportResponse;
pub use app_server_protocol::ThreadItemsListParams;
pub use app_server_protocol::ThreadItemsListResponse;
pub use app_server_protocol::ThreadListParams;
pub use app_server_protocol::ThreadListResponse;
pub use app_server_protocol::ThreadReadParams;
pub use app_server_protocol::ThreadReadResponse;
pub use app_server_protocol::ThreadTurnsListParams;
pub use app_server_protocol::ThreadTurnsListResponse;
pub use app_server_protocol::UsageStatsDailyTrendsListResponse;
pub use app_server_protocol::UsageStatsDailyUsage;
pub use app_server_protocol::UsageStatsModelRankingListResponse;
pub use app_server_protocol::UsageStatsModelUsage;
pub use app_server_protocol::UsageStatsRangeParams;
pub use app_server_protocol::UsageStatsReadResponse;
pub use app_server_protocol::UsageStatsSummary;
pub use app_server_protocol::WechatChannelAccountListResponse;
pub use app_server_protocol::WechatConfiguredAccount;
pub use app_server_protocol::WindowsStartupCheck;
pub use app_server_protocol::WindowsStartupDiagnosticsResponse;
pub use app_server_protocol::WorkflowCancelParams;
pub use app_server_protocol::WorkflowCancelResponse;
pub use app_server_protocol::WorkflowReadParams;
pub use app_server_protocol::WorkflowReadResponse;
pub use app_server_protocol::WorkflowRespondParams;
pub use app_server_protocol::WorkflowRespondResponse;
pub use app_server_protocol::WorkflowRetryParams;
pub use app_server_protocol::WorkflowRetryResponse;
pub use app_server_protocol::WorkspaceEnsureParams;
pub use app_server_protocol::WorkspaceEnsureProjectParams;
pub use app_server_protocol::WorkspaceEnsureProjectResponse;
pub use app_server_protocol::WorkspaceEnsureReadyResponse;
pub use app_server_protocol::WorkspaceListResponse;
pub use app_server_protocol::WorkspacePathReadParams;
pub use app_server_protocol::WorkspaceProjectPathResolveParams;
pub use app_server_protocol::WorkspaceProjectPathResolveResponse;
pub use app_server_protocol::WorkspaceProjectsRootReadResponse;
pub use app_server_protocol::WorkspaceReadParams;
pub use app_server_protocol::WorkspaceReadResponse;
pub use app_server_protocol::WorkspaceRightSurfacePendingChangedParams;
pub use app_server_protocol::WorkspaceRightSurfacePendingConsumeParams;
pub use app_server_protocol::WorkspaceRightSurfacePendingConsumeResponse;
pub use app_server_protocol::WorkspaceRightSurfacePendingDismissParams;
pub use app_server_protocol::WorkspaceRightSurfacePendingDismissResponse;
pub use app_server_protocol::WorkspaceRightSurfacePendingListParams;
pub use app_server_protocol::WorkspaceRightSurfacePendingListResponse;
pub use app_server_protocol::WorkspaceRightSurfacePendingRequest;
pub use app_server_protocol::WorkspaceRightSurfaceRequestParams;
pub use app_server_protocol::WorkspaceRightSurfaceRequestResponse;
pub use app_server_protocol::WorkspaceSkillBindingsListParams;
pub use app_server_protocol::WorkspaceSkillBindingsListResponse;
pub use app_server_protocol::METHOD_AGENT_SESSION_ANALYSIS_HANDOFF_EXPORT;
pub use app_server_protocol::METHOD_AGENT_SESSION_COMPACT;
pub use app_server_protocol::METHOD_AGENT_SESSION_EVENT;
pub use app_server_protocol::METHOD_AGENT_SESSION_FILE_CHECKPOINT_DIFF;
pub use app_server_protocol::METHOD_AGENT_SESSION_FILE_CHECKPOINT_GET;
pub use app_server_protocol::METHOD_AGENT_SESSION_FILE_CHECKPOINT_LIST;
pub use app_server_protocol::METHOD_AGENT_SESSION_FILE_CHECKPOINT_RESTORE;
pub use app_server_protocol::METHOD_AGENT_SESSION_HANDOFF_BUNDLE_EXPORT;
pub use app_server_protocol::METHOD_AGENT_SESSION_QUEUED_TURN_PROMOTE;
pub use app_server_protocol::METHOD_AGENT_SESSION_QUEUED_TURN_REMOVE;
pub use app_server_protocol::METHOD_AGENT_SESSION_REPLAY_CASE_EXPORT;
pub use app_server_protocol::METHOD_AGENT_SESSION_REVIEW_DECISION_SAVE;
pub use app_server_protocol::METHOD_AGENT_SESSION_REVIEW_DECISION_TEMPLATE_EXPORT;
pub use app_server_protocol::METHOD_ARTIFACT_READ;
pub use app_server_protocol::METHOD_AUTOMATION_JOB_CREATE;
pub use app_server_protocol::METHOD_AUTOMATION_JOB_DELETE;
pub use app_server_protocol::METHOD_AUTOMATION_JOB_HEALTH;
pub use app_server_protocol::METHOD_AUTOMATION_JOB_LIST;
pub use app_server_protocol::METHOD_AUTOMATION_JOB_READ;
pub use app_server_protocol::METHOD_AUTOMATION_JOB_RUN_HISTORY;
pub use app_server_protocol::METHOD_AUTOMATION_JOB_RUN_NOW;
pub use app_server_protocol::METHOD_AUTOMATION_JOB_UPDATE;
pub use app_server_protocol::METHOD_AUTOMATION_SCHEDULER_CONFIG_READ;
pub use app_server_protocol::METHOD_AUTOMATION_SCHEDULER_CONFIG_UPDATE;
pub use app_server_protocol::METHOD_AUTOMATION_SCHEDULER_STATUS;
pub use app_server_protocol::METHOD_AUTOMATION_SCHEDULE_PREVIEW;
pub use app_server_protocol::METHOD_AUTOMATION_SCHEDULE_VALIDATE;
pub use app_server_protocol::METHOD_BROWSER_SESSION_ACTION_EXECUTE;
pub use app_server_protocol::METHOD_BROWSER_SESSION_CLOSE;
pub use app_server_protocol::METHOD_BROWSER_SESSION_EVENT_LIST;
pub use app_server_protocol::METHOD_BROWSER_SESSION_OPEN;
pub use app_server_protocol::METHOD_BROWSER_SESSION_READ;
pub use app_server_protocol::METHOD_BROWSER_SESSION_TARGET_LIST;
pub use app_server_protocol::METHOD_CAPABILITY_LIST;
pub use app_server_protocol::METHOD_DIAGNOSTICS_LOG_STORAGE_READ;
pub use app_server_protocol::METHOD_DIAGNOSTICS_SERVER_READ;
pub use app_server_protocol::METHOD_DIAGNOSTICS_SUPPORT_BUNDLE_EXPORT;
pub use app_server_protocol::METHOD_DIAGNOSTICS_WINDOWS_STARTUP_READ;
pub use app_server_protocol::METHOD_EVIDENCE_EXPORT;
pub use app_server_protocol::METHOD_FILE_SYSTEM_CREATE_DIRECTORY;
pub use app_server_protocol::METHOD_FILE_SYSTEM_CREATE_FILE;
pub use app_server_protocol::METHOD_FILE_SYSTEM_DELETE_FILE;
pub use app_server_protocol::METHOD_FILE_SYSTEM_LIST_DIRECTORY;
pub use app_server_protocol::METHOD_FILE_SYSTEM_READ_FILE_PREVIEW;
pub use app_server_protocol::METHOD_FILE_SYSTEM_RENAME_FILE;
pub use app_server_protocol::METHOD_GATEWAY_CHANNEL_STATUS;
pub use app_server_protocol::METHOD_GATEWAY_TUNNEL_CLOUDFLARED_DETECT;
pub use app_server_protocol::METHOD_GATEWAY_TUNNEL_CLOUDFLARED_INSTALL;
pub use app_server_protocol::METHOD_GATEWAY_TUNNEL_CREATE;
pub use app_server_protocol::METHOD_GATEWAY_TUNNEL_PROBE;
pub use app_server_protocol::METHOD_GATEWAY_TUNNEL_RESTART;
pub use app_server_protocol::METHOD_GATEWAY_TUNNEL_START;
pub use app_server_protocol::METHOD_GATEWAY_TUNNEL_STATUS;
pub use app_server_protocol::METHOD_GATEWAY_TUNNEL_STOP;
pub use app_server_protocol::METHOD_GATEWAY_TUNNEL_SYNC_WEBHOOK_URL;
pub use app_server_protocol::METHOD_INITIALIZE;
pub use app_server_protocol::METHOD_INITIALIZED;
pub use app_server_protocol::METHOD_KNOWLEDGE_CONTEXT_RESOLVE;
pub use app_server_protocol::METHOD_KNOWLEDGE_CONTEXT_RUN_VALIDATE;
pub use app_server_protocol::METHOD_KNOWLEDGE_PACK_COMPILE;
pub use app_server_protocol::METHOD_KNOWLEDGE_PACK_DEFAULT_SET;
pub use app_server_protocol::METHOD_KNOWLEDGE_PACK_LIST;
pub use app_server_protocol::METHOD_KNOWLEDGE_PACK_READ;
pub use app_server_protocol::METHOD_KNOWLEDGE_PACK_STATUS_UPDATE;
pub use app_server_protocol::METHOD_KNOWLEDGE_SOURCE_IMPORT;
pub use app_server_protocol::METHOD_LOG_CLEAR;
pub use app_server_protocol::METHOD_LOG_DIAGNOSTIC_HISTORY_CLEAR;
pub use app_server_protocol::METHOD_LOG_LIST;
pub use app_server_protocol::METHOD_LOG_PERSISTED_TAIL;
pub use app_server_protocol::METHOD_MCP_PROMPT_GET;
pub use app_server_protocol::METHOD_MCP_PROMPT_LIST;
pub use app_server_protocol::METHOD_MCP_RESOURCE_LIST;
pub use app_server_protocol::METHOD_MCP_RESOURCE_READ;
pub use app_server_protocol::METHOD_MCP_RESOURCE_SUBSCRIBE;
pub use app_server_protocol::METHOD_MCP_RESOURCE_UNSUBSCRIBE;
pub use app_server_protocol::METHOD_MCP_SERVER_CREATE;
pub use app_server_protocol::METHOD_MCP_SERVER_DELETE;
pub use app_server_protocol::METHOD_MCP_SERVER_ENABLED_SET;
pub use app_server_protocol::METHOD_MCP_SERVER_IMPORT_FROM_APP;
pub use app_server_protocol::METHOD_MCP_SERVER_LIST;
pub use app_server_protocol::METHOD_MCP_SERVER_START;
pub use app_server_protocol::METHOD_MCP_SERVER_STATUS_LIST;
pub use app_server_protocol::METHOD_MCP_SERVER_STOP;
pub use app_server_protocol::METHOD_MCP_SERVER_SYNC_ALL_TO_LIVE;
pub use app_server_protocol::METHOD_MCP_SERVER_UPDATE;
pub use app_server_protocol::METHOD_MCP_TOOL_CALL;
pub use app_server_protocol::METHOD_MCP_TOOL_CALL_WITH_CALLER;
pub use app_server_protocol::METHOD_MCP_TOOL_LIST;
pub use app_server_protocol::METHOD_MCP_TOOL_LIST_FOR_CONTEXT;
pub use app_server_protocol::METHOD_MCP_TOOL_SEARCH;
pub use app_server_protocol::METHOD_MEDIA_TASK_ARTIFACT_AUDIO_COMPLETE;
pub use app_server_protocol::METHOD_MEDIA_TASK_ARTIFACT_AUDIO_CREATE;
pub use app_server_protocol::METHOD_MEDIA_TASK_ARTIFACT_CANCEL;
pub use app_server_protocol::METHOD_MEDIA_TASK_ARTIFACT_GET;
pub use app_server_protocol::METHOD_MEDIA_TASK_ARTIFACT_IMAGE_CREATE;
pub use app_server_protocol::METHOD_MEDIA_TASK_ARTIFACT_LIST;
pub use app_server_protocol::METHOD_MEMORY_STORE_ADD_NOTE;
pub use app_server_protocol::METHOD_MEMORY_STORE_CONSOLIDATE;
pub use app_server_protocol::METHOD_MEMORY_STORE_HEALTH;
pub use app_server_protocol::METHOD_MEMORY_STORE_INDEX_REBUILD;
pub use app_server_protocol::METHOD_MEMORY_STORE_LIST;
pub use app_server_protocol::METHOD_MEMORY_STORE_READ;
pub use app_server_protocol::METHOD_MEMORY_STORE_RESET;
pub use app_server_protocol::METHOD_MEMORY_STORE_REVIEW_LIST;
pub use app_server_protocol::METHOD_MEMORY_STORE_REVIEW_RESOLVE;
pub use app_server_protocol::METHOD_MEMORY_STORE_SEARCH;
pub use app_server_protocol::METHOD_MODEL_LIST;
pub use app_server_protocol::METHOD_MODEL_PREFERENCES_LIST;
pub use app_server_protocol::METHOD_MODEL_PROVIDER_ALIAS_LIST;
pub use app_server_protocol::METHOD_MODEL_PROVIDER_ALIAS_READ;
pub use app_server_protocol::METHOD_MODEL_PROVIDER_CATALOG_LIST;
pub use app_server_protocol::METHOD_MODEL_PROVIDER_LIST;
pub use app_server_protocol::METHOD_MODEL_SYNC_STATE_READ;
pub use app_server_protocol::METHOD_PLUGIN_INSTALLED_DISABLED_SET;
pub use app_server_protocol::METHOD_PLUGIN_INSTALLED_LIST;
pub use app_server_protocol::METHOD_PLUGIN_INSTALLED_SAVE;
pub use app_server_protocol::METHOD_PLUGIN_INSTALLED_UNINSTALL;
pub use app_server_protocol::METHOD_PLUGIN_INSTALLED_UNINSTALL_REHEARSAL;
pub use app_server_protocol::METHOD_PLUGIN_LOCAL_PACKAGE_EXPORT;
pub use app_server_protocol::METHOD_PLUGIN_LOCAL_PACKAGE_INSPECT;
pub use app_server_protocol::METHOD_PLUGIN_PACKAGE_FETCH_CLOUD;
pub use app_server_protocol::METHOD_PLUGIN_SHELL_PREPARE;
pub use app_server_protocol::METHOD_PLUGIN_UI_RUNTIME_START;
pub use app_server_protocol::METHOD_PLUGIN_UI_RUNTIME_STATUS;
pub use app_server_protocol::METHOD_PLUGIN_UI_RUNTIME_STOP;
pub use app_server_protocol::METHOD_PROJECT_MEMORY_READ;
pub use app_server_protocol::METHOD_SKILL_LIST;
pub use app_server_protocol::METHOD_SKILL_LOCAL_DETAIL_INSPECT;
pub use app_server_protocol::METHOD_SKILL_LOCAL_RENAME;
pub use app_server_protocol::METHOD_SKILL_MARKETPLACE_INSTALL;
pub use app_server_protocol::METHOD_SKILL_PACKAGE_DOWNLOAD_INSTALL;
pub use app_server_protocol::METHOD_SKILL_PACKAGE_EXPORT;
pub use app_server_protocol::METHOD_SKILL_PACKAGE_LOCAL_INSPECT;
pub use app_server_protocol::METHOD_SKILL_PACKAGE_LOCAL_INSTALL;
pub use app_server_protocol::METHOD_SKILL_PACKAGE_LOCAL_REPLACE;
pub use app_server_protocol::METHOD_SKILL_READ;
pub use app_server_protocol::METHOD_THREAD_ITEMS_LIST;
pub use app_server_protocol::METHOD_THREAD_LIST;
pub use app_server_protocol::METHOD_THREAD_READ;
pub use app_server_protocol::METHOD_THREAD_RESUME;
pub use app_server_protocol::METHOD_THREAD_START;
pub use app_server_protocol::METHOD_THREAD_TURNS_LIST;
pub use app_server_protocol::METHOD_TURN_INTERRUPT;
pub use app_server_protocol::METHOD_TURN_START;
pub use app_server_protocol::METHOD_USAGE_STATS_DAILY_TRENDS_LIST;
pub use app_server_protocol::METHOD_USAGE_STATS_MODEL_RANKING_LIST;
pub use app_server_protocol::METHOD_USAGE_STATS_READ;
pub use app_server_protocol::METHOD_WECHAT_CHANNEL_ACCOUNT_LIST;
pub use app_server_protocol::METHOD_WORKFLOW_CANCEL;
pub use app_server_protocol::METHOD_WORKFLOW_READ;
pub use app_server_protocol::METHOD_WORKFLOW_RESPOND;
pub use app_server_protocol::METHOD_WORKFLOW_RETRY;
pub use app_server_protocol::METHOD_WORKSPACE_BY_PATH_READ;
pub use app_server_protocol::METHOD_WORKSPACE_DEFAULT_ENSURE;
pub use app_server_protocol::METHOD_WORKSPACE_DEFAULT_READ;
pub use app_server_protocol::METHOD_WORKSPACE_ENSURE;
pub use app_server_protocol::METHOD_WORKSPACE_ENSURE_READY;
pub use app_server_protocol::METHOD_WORKSPACE_LIST;
pub use app_server_protocol::METHOD_WORKSPACE_PROJECTS_ROOT_READ;
pub use app_server_protocol::METHOD_WORKSPACE_PROJECT_PATH_RESOLVE;
pub use app_server_protocol::METHOD_WORKSPACE_READ;
pub use app_server_protocol::METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CHANGED;
pub use app_server_protocol::METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CONSUME;
pub use app_server_protocol::METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_DISMISS;
pub use app_server_protocol::METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_LIST;
pub use app_server_protocol::METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST;
pub use app_server_protocol::METHOD_WORKSPACE_SKILL_BINDINGS_LIST;
use app_server_transport::encode_message;
use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("invalid request params: {0}")]
    InvalidParams(String),
    #[error("invalid server notification: {0}")]
    InvalidNotification(String),
    #[error("failed to serialize request params: {0}")]
    Serialize(#[from] serde_json::Error),
    #[error(transparent)]
    Transport(#[from] app_server_transport::TransportError),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedRequest<P> {
    method: &'static str,
    params: P,
}

impl<P> TypedRequest<P> {
    pub fn new(method: &'static str, params: P) -> Self {
        Self { method, params }
    }

    pub fn method(&self) -> &'static str {
        self.method
    }

    pub fn params(&self) -> &P {
        &self.params
    }

    pub fn into_parts(self) -> (&'static str, P) {
        (self.method, self.params)
    }
}

fn validate_mcp_resource_target(server: &str, uri: &str) -> Result<(), ClientError> {
    if server.trim().is_empty() {
        return Err(ClientError::InvalidParams(
            "MCP resource server cannot be empty".to_string(),
        ));
    }
    if uri.trim().is_empty() {
        return Err(ClientError::InvalidParams(
            "MCP resource URI cannot be empty".to_string(),
        ));
    }
    Ok(())
}

fn validate_mcp_prompt_target(server: &str, name: &str) -> Result<(), ClientError> {
    if server.trim().is_empty() {
        return Err(ClientError::InvalidParams(
            "MCP prompt server cannot be empty".to_string(),
        ));
    }
    if name.trim().is_empty() {
        return Err(ClientError::InvalidParams(
            "MCP prompt name cannot be empty".to_string(),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum ClientEvent {
    Lifecycle(ServerNotification),
    AgentSession(JsonRpcNotification),
    Notification(JsonRpcNotification),
    Request(JsonRpcRequest),
    Response(JsonRpcResponse),
    Error(JsonRpcErrorResponse),
}

impl TryFrom<JsonRpcMessage> for ClientEvent {
    type Error = ClientError;

    fn try_from(message: JsonRpcMessage) -> Result<Self, ClientError> {
        match message {
            JsonRpcMessage::Notification(notification)
                if notification.method == METHOD_AGENT_SESSION_EVENT =>
            {
                let event_type = read_agent_session_event_type(&notification).ok_or_else(|| {
                    ClientError::InvalidNotification(
                        "agentSession/event is missing event.type".to_string(),
                    )
                })?;
                if is_agent_session_side_channel(event_type) {
                    Ok(Self::AgentSession(notification))
                } else {
                    Err(ClientError::InvalidNotification(format!(
                        "agentSession/event type '{event_type}' is not an allowed raw side-channel"
                    )))
                }
            }
            JsonRpcMessage::Notification(notification)
                if NOTIFICATION_METHODS.contains(&notification.method.as_str()) =>
            {
                ServerNotification::try_from(notification)
                    .map(Self::Lifecycle)
                    .map_err(ClientError::InvalidNotification)
            }
            JsonRpcMessage::Notification(notification) => Ok(Self::Notification(notification)),
            JsonRpcMessage::Request(request) => Ok(Self::Request(request)),
            JsonRpcMessage::Response(response) => Ok(Self::Response(response)),
            JsonRpcMessage::Error(error) => Ok(Self::Error(error)),
        }
    }
}

fn read_agent_session_event_type(notification: &JsonRpcNotification) -> Option<&str> {
    notification
        .params
        .as_ref()?
        .get("event")?
        .get("type")?
        .as_str()
}

fn is_agent_session_side_channel(event_type: &str) -> bool {
    event_type.starts_with("action.")
        || event_type.starts_with("approval.")
        || event_type.starts_with("provider.")
        || event_type.starts_with("image_task.")
        || event_type.starts_with("image_task_")
        || event_type.starts_with("media.")
        || event_type.starts_with("runtime.")
}

#[derive(Debug, Clone)]
pub struct AppServerClient {
    next_request_id: i64,
}

impl Default for AppServerClient {
    fn default() -> Self {
        Self { next_request_id: 1 }
    }
}

impl AppServerClient {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn initialize(&mut self, params: InitializeParams) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::initialize(params))
    }

    pub fn initialized(&self) -> JsonRpcNotification {
        JsonRpcNotification::new(METHOD_INITIALIZED, Some(serde_json::json!({})))
    }

    pub fn list_capabilities(
        &mut self,
        params: CapabilityListParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_capabilities(params))
    }

    pub fn list_capabilities_default(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.list_capabilities(CapabilityListParams::default())
    }

    pub fn list_sessions(
        &mut self,
        params: AgentSessionListParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_sessions(params))
    }

    pub fn read_thread(&mut self, params: ThreadReadParams) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_thread(params))
    }

    pub fn list_threads(
        &mut self,
        params: ThreadListParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_threads(params))
    }

    pub fn archive_thread(
        &mut self,
        params: ThreadArchiveParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::archive_thread(params))
    }

    pub fn unarchive_thread(
        &mut self,
        params: ThreadUnarchiveParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::unarchive_thread(params))
    }

    pub fn run_thread_shell_command(
        &mut self,
        params: ThreadShellCommandParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::run_thread_shell_command(params))
    }

    pub fn list_thread_turns(
        &mut self,
        params: ThreadTurnsListParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_thread_turns(params))
    }

    pub fn list_thread_items(
        &mut self,
        params: ThreadItemsListParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_thread_items(params))
    }

    pub fn start_session(
        &mut self,
        params: AgentSessionStartParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::start_session(params))
    }

    pub fn read_session(
        &mut self,
        params: AgentSessionReadParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_session(params))
    }

    pub fn read_workflow(
        &mut self,
        params: WorkflowReadParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_workflow(params))
    }

    pub fn cancel_workflow(
        &mut self,
        params: WorkflowCancelParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::cancel_workflow(params))
    }

    pub fn retry_workflow(
        &mut self,
        params: WorkflowRetryParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::retry_workflow(params))
    }

    pub fn respond_workflow(
        &mut self,
        params: WorkflowRespondParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::respond_workflow(params))
    }

    pub fn list_workspaces(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_workspaces())
    }

    pub fn read_workspace(
        &mut self,
        params: WorkspaceReadParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_workspace(params))
    }

    pub fn read_workspace_by_path(
        &mut self,
        params: WorkspacePathReadParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_workspace_by_path(params))
    }

    pub fn ensure_workspace(
        &mut self,
        params: WorkspaceEnsureProjectParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::ensure_workspace(params))
    }

    pub fn read_default_workspace(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_default_workspace())
    }

    pub fn ensure_default_workspace(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::ensure_default_workspace())
    }

    pub fn read_workspace_projects_root(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_workspace_projects_root())
    }

    pub fn resolve_workspace_project_path(
        &mut self,
        params: WorkspaceProjectPathResolveParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::resolve_workspace_project_path(params))
    }

    pub fn ensure_workspace_ready(
        &mut self,
        params: WorkspaceEnsureParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::ensure_workspace_ready(params))
    }

    pub fn request_workspace_right_surface(
        &mut self,
        params: WorkspaceRightSurfaceRequestParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::request_workspace_right_surface(params))
    }

    pub fn list_workspace_right_surface_pending(
        &mut self,
        params: WorkspaceRightSurfacePendingListParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_workspace_right_surface_pending(params))
    }

    pub fn consume_workspace_right_surface_pending(
        &mut self,
        params: WorkspaceRightSurfacePendingConsumeParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::consume_workspace_right_surface_pending(params))
    }

    pub fn dismiss_workspace_right_surface_pending(
        &mut self,
        params: WorkspaceRightSurfacePendingDismissParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::dismiss_workspace_right_surface_pending(params))
    }

    pub fn list_browser_session_targets(
        &mut self,
        params: BrowserSessionTargetListParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_browser_session_targets(params))
    }

    pub fn open_browser_session(
        &mut self,
        params: BrowserSessionOpenParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::open_browser_session(params))
    }

    pub fn read_browser_session(
        &mut self,
        params: BrowserSessionIdParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_browser_session(params))
    }

    pub fn close_browser_session(
        &mut self,
        params: BrowserSessionIdParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::close_browser_session(params))
    }

    pub fn list_browser_session_events(
        &mut self,
        params: BrowserSessionEventListParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_browser_session_events(params))
    }

    pub fn execute_browser_session_action(
        &mut self,
        params: BrowserSessionActionExecuteParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::execute_browser_session_action(params))
    }

    pub fn list_skills(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_skills())
    }

    pub fn read_skill(&mut self, params: SkillReadParams) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_skill(params))
    }

    pub fn inspect_local_skill_detail(
        &mut self,
        params: SkillLocalDetailInspectParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::inspect_local_skill_detail(params))
    }

    pub fn rename_local_skill(
        &mut self,
        params: SkillLocalRenameParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::rename_local_skill(params))
    }

    pub fn inspect_local_skill_package(
        &mut self,
        params: SkillPackageLocalInspectParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::inspect_local_skill_package(params))
    }

    pub fn install_local_skill_package(
        &mut self,
        params: SkillPackageLocalInstallParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::install_local_skill_package(params))
    }

    pub fn replace_local_skill_package(
        &mut self,
        params: SkillPackageLocalReplaceParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::replace_local_skill_package(params))
    }

    pub fn export_skill_package(
        &mut self,
        params: SkillPackageExportParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::export_skill_package(params))
    }

    pub fn install_marketplace_skill(
        &mut self,
        params: SkillMarketplaceInstallParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::install_marketplace_skill(params))
    }

    pub fn install_skill_from_download_url(
        &mut self,
        params: SkillDownloadInstallParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::install_skill_from_download_url(params))
    }

    pub fn read_gateway_channel_status(
        &mut self,
        params: GatewayChannelStatusParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_gateway_channel_status(params))
    }

    pub fn probe_gateway_tunnel(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::probe_gateway_tunnel())
    }

    pub fn detect_gateway_tunnel_cloudflared(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::detect_gateway_tunnel_cloudflared())
    }

    pub fn install_gateway_tunnel_cloudflared(
        &mut self,
        params: GatewayTunnelCloudflaredInstallParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::install_gateway_tunnel_cloudflared(params))
    }

    pub fn create_gateway_tunnel(
        &mut self,
        params: GatewayTunnelCreateParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::create_gateway_tunnel(params))
    }

    pub fn start_gateway_tunnel(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::start_gateway_tunnel())
    }

    pub fn stop_gateway_tunnel(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::stop_gateway_tunnel())
    }

    pub fn restart_gateway_tunnel(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::restart_gateway_tunnel())
    }

    pub fn read_gateway_tunnel_status(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_gateway_tunnel_status())
    }

    pub fn sync_gateway_tunnel_webhook_url(
        &mut self,
        params: GatewayTunnelSyncWebhookUrlParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::sync_gateway_tunnel_webhook_url(params))
    }

    pub fn list_wechat_channel_accounts(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_wechat_channel_accounts())
    }

    pub fn create_image_media_task_artifact(
        &mut self,
        params: MediaTaskArtifactImageCreateParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::create_image_media_task_artifact(params))
    }

    pub fn create_audio_media_task_artifact(
        &mut self,
        params: MediaTaskArtifactAudioCreateParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::create_audio_media_task_artifact(params))
    }

    pub fn complete_audio_media_task_artifact(
        &mut self,
        params: MediaTaskArtifactAudioCompleteParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::complete_audio_media_task_artifact(params))
    }

    pub fn get_media_task_artifact(
        &mut self,
        params: MediaTaskArtifactLookupParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::get_media_task_artifact(params))
    }

    pub fn list_media_task_artifacts(
        &mut self,
        params: MediaTaskArtifactListParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_media_task_artifacts(params))
    }

    pub fn cancel_media_task_artifact(
        &mut self,
        params: MediaTaskArtifactLookupParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::cancel_media_task_artifact(params))
    }

    pub fn read_server_diagnostics(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_server_diagnostics())
    }

    pub fn read_log_storage_diagnostics(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_log_storage_diagnostics())
    }

    pub fn export_support_bundle(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::export_support_bundle())
    }

    pub fn read_windows_startup_diagnostics(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_windows_startup_diagnostics())
    }

    pub fn list_workspace_skill_bindings(
        &mut self,
        params: WorkspaceSkillBindingsListParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_workspace_skill_bindings(params))
    }

    pub fn inspect_plugin_local_package(
        &mut self,
        params: PluginLocalPackageInspectParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::inspect_plugin_local_package(params))
    }

    pub fn export_plugin_local_package(
        &mut self,
        params: PluginLocalPackageExportParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::export_plugin_local_package(params))
    }

    pub fn fetch_plugin_cloud_package(
        &mut self,
        params: PluginFetchCloudPackageParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::fetch_plugin_cloud_package(params))
    }

    pub fn save_plugin_installed(
        &mut self,
        params: PluginInstalledSaveParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::save_plugin_installed(params))
    }

    pub fn list_plugin_installed(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_plugin_installed())
    }

    pub fn set_plugin_installed_disabled(
        &mut self,
        params: PluginInstalledDisabledSetParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::set_plugin_installed_disabled(params))
    }

    pub fn preview_plugin_uninstall(
        &mut self,
        params: PluginUninstallRehearsalParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::preview_plugin_uninstall(params))
    }

    pub fn uninstall_plugin(
        &mut self,
        params: PluginUninstallParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::uninstall_plugin(params))
    }

    pub fn prepare_plugin_shell(
        &mut self,
        params: PluginShellPrepareParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::prepare_plugin_shell(params))
    }

    pub fn start_plugin_ui_runtime(
        &mut self,
        params: PluginUiRuntimeStartParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::start_plugin_ui_runtime(params))
    }

    pub fn plugin_ui_runtime_status(
        &mut self,
        params: PluginUiRuntimeStatusParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::plugin_ui_runtime_status(params))
    }

    pub fn stop_plugin_ui_runtime(
        &mut self,
        params: PluginUiRuntimeStopParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::stop_plugin_ui_runtime(params))
    }

    pub fn list_knowledge_packs(
        &mut self,
        params: KnowledgeListPacksParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_knowledge_packs(params))
    }

    pub fn read_knowledge_pack(
        &mut self,
        params: KnowledgeReadPackParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_knowledge_pack(params))
    }

    pub fn import_knowledge_source(
        &mut self,
        params: KnowledgeImportSourceParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::import_knowledge_source(params))
    }

    pub fn compile_knowledge_pack(
        &mut self,
        params: KnowledgeCompilePackParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::compile_knowledge_pack(params))
    }

    pub fn set_default_knowledge_pack(
        &mut self,
        params: KnowledgeSetDefaultPackParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::set_default_knowledge_pack(params))
    }

    pub fn update_knowledge_pack_status(
        &mut self,
        params: KnowledgeUpdatePackStatusParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::update_knowledge_pack_status(params))
    }

    pub fn resolve_knowledge_context(
        &mut self,
        params: KnowledgeResolveContextParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::resolve_knowledge_context(params))
    }

    pub fn validate_knowledge_context_run(
        &mut self,
        params: KnowledgeValidateContextRunParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::validate_knowledge_context_run(params))
    }

    pub fn list_automation_jobs(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_automation_jobs())
    }

    pub fn read_automation_scheduler_config(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_automation_scheduler_config())
    }

    pub fn update_automation_scheduler_config(
        &mut self,
        params: AutomationSchedulerConfigUpdateParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::update_automation_scheduler_config(params))
    }

    pub fn read_automation_scheduler_status(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_automation_scheduler_status())
    }

    pub fn read_automation_job(
        &mut self,
        params: AutomationJobIdParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_automation_job(params))
    }

    pub fn create_automation_job(
        &mut self,
        params: AutomationJobCreateParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::create_automation_job(params))
    }

    pub fn update_automation_job(
        &mut self,
        params: AutomationJobUpdateParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::update_automation_job(params))
    }

    pub fn delete_automation_job(
        &mut self,
        params: AutomationJobIdParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::delete_automation_job(params))
    }

    pub fn run_automation_job_now(
        &mut self,
        params: AutomationJobIdParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::run_automation_job_now(params))
    }

    pub fn read_automation_health(
        &mut self,
        params: AutomationJobHealthParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_automation_health(params))
    }

    pub fn read_automation_run_history(
        &mut self,
        params: AutomationJobRunHistoryParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_automation_run_history(params))
    }

    pub fn preview_automation_schedule(
        &mut self,
        params: AutomationScheduleParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::preview_automation_schedule(params))
    }

    pub fn validate_automation_schedule(
        &mut self,
        params: AutomationScheduleParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::validate_automation_schedule(params))
    }

    pub fn list_mcp_servers(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_mcp_servers())
    }

    pub fn list_mcp_servers_with_status(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_mcp_servers_with_status())
    }

    pub fn create_mcp_server(
        &mut self,
        params: McpServerCreateParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::create_mcp_server(params))
    }

    pub fn update_mcp_server(
        &mut self,
        params: McpServerUpdateParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::update_mcp_server(params))
    }

    pub fn delete_mcp_server(
        &mut self,
        params: McpServerDeleteParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::delete_mcp_server(params))
    }

    pub fn set_mcp_server_enabled(
        &mut self,
        params: McpServerEnabledSetParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::set_mcp_server_enabled(params))
    }

    pub fn import_mcp_servers_from_app(
        &mut self,
        params: McpServerImportFromAppParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::import_mcp_servers_from_app(params))
    }

    pub fn sync_all_mcp_servers_to_live(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::sync_all_mcp_servers_to_live())
    }

    pub fn start_mcp_server(
        &mut self,
        params: McpServerStartParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::start_mcp_server(params))
    }

    pub fn stop_mcp_server(
        &mut self,
        params: McpServerStopParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::stop_mcp_server(params))
    }

    pub fn list_mcp_tools(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_mcp_tools())
    }

    pub fn list_mcp_tools_for_context(
        &mut self,
        params: McpToolListForContextParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_mcp_tools_for_context(params))
    }

    pub fn search_mcp_tools(
        &mut self,
        params: McpToolSearchParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::search_mcp_tools(params))
    }

    pub fn call_mcp_tool(
        &mut self,
        params: McpToolCallParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::call_mcp_tool(params))
    }

    pub fn call_mcp_tool_with_caller(
        &mut self,
        params: McpToolCallWithCallerParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::call_mcp_tool_with_caller(params))
    }

    pub fn list_mcp_prompts(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_mcp_prompts())
    }

    pub fn get_mcp_prompt(
        &mut self,
        params: McpPromptGetParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::get_mcp_prompt(params)?)
    }

    pub fn list_mcp_resources(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_mcp_resources())
    }

    pub fn read_mcp_resource(
        &mut self,
        params: McpResourceReadParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_mcp_resource(params)?)
    }

    pub fn subscribe_mcp_resource(
        &mut self,
        params: McpResourceSubscribeParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::subscribe_mcp_resource(params)?)
    }

    pub fn unsubscribe_mcp_resource(
        &mut self,
        params: McpResourceUnsubscribeParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::unsubscribe_mcp_resource(params)?)
    }

    pub fn read_project_memory(
        &mut self,
        params: ProjectMemoryReadParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_project_memory(params))
    }

    pub fn list_memory_store(
        &mut self,
        params: MemoryStoreListParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_memory_store(params))
    }

    pub fn read_memory_store(
        &mut self,
        params: MemoryStoreReadParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_memory_store(params))
    }

    pub fn search_memory_store(
        &mut self,
        params: MemoryStoreSearchParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::search_memory_store(params))
    }

    pub fn add_memory_store_note(
        &mut self,
        params: MemoryStoreAddNoteParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::add_memory_store_note(params))
    }

    pub fn consolidate_memory_store(
        &mut self,
        params: MemoryStoreConsolidateParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::consolidate_memory_store(params))
    }

    pub fn list_memory_store_review_notes(
        &mut self,
        params: MemoryStoreReviewListParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_memory_store_review_notes(params))
    }

    pub fn resolve_memory_store_review_note(
        &mut self,
        params: MemoryStoreReviewResolveParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::resolve_memory_store_review_note(params))
    }

    pub fn health_memory_store(
        &mut self,
        params: MemoryStoreRootParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::health_memory_store(params))
    }

    pub fn reset_memory_store(
        &mut self,
        params: MemoryStoreResetParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::reset_memory_store(params))
    }

    pub fn rebuild_memory_store_index(
        &mut self,
        params: MemoryStoreRootParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::rebuild_memory_store_index(params))
    }

    pub fn list_logs(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_logs())
    }

    pub fn read_persisted_log_tail(
        &mut self,
        params: LogPersistedTailParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_persisted_log_tail(params))
    }

    pub fn clear_logs(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::clear_logs())
    }

    pub fn clear_diagnostic_log_history(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::clear_diagnostic_log_history())
    }

    pub fn read_usage_stats(
        &mut self,
        params: UsageStatsRangeParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_usage_stats(params))
    }

    pub fn list_usage_stats_model_ranking(
        &mut self,
        params: UsageStatsRangeParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_usage_stats_model_ranking(params))
    }

    pub fn list_usage_stats_daily_trends(
        &mut self,
        params: UsageStatsRangeParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_usage_stats_daily_trends(params))
    }

    pub fn list_models(&mut self, params: ModelListParams) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_models(params))
    }

    pub fn list_model_preferences(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_model_preferences())
    }

    pub fn read_model_sync_state(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_model_sync_state())
    }

    pub fn list_model_providers(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_model_providers())
    }

    pub fn list_model_provider_catalog(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_model_provider_catalog())
    }

    pub fn read_model_provider_alias(
        &mut self,
        params: ModelProviderAliasReadParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_model_provider_alias(params))
    }

    pub fn list_model_provider_aliases(&mut self) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_model_provider_aliases())
    }

    pub fn read_artifacts(
        &mut self,
        params: ArtifactReadParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_artifacts(params))
    }

    pub fn list_directory(
        &mut self,
        params: FileSystemListDirectoryParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::list_directory(params))
    }

    pub fn read_file_preview(
        &mut self,
        params: FileSystemReadFilePreviewParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::read_file_preview(params))
    }

    pub fn create_file(
        &mut self,
        params: FileSystemCreateFileParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::create_file(params))
    }

    pub fn create_directory(
        &mut self,
        params: FileSystemCreateDirectoryParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::create_directory(params))
    }

    pub fn rename_file(
        &mut self,
        params: FileSystemRenameFileParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::rename_file(params))
    }

    pub fn delete_file(
        &mut self,
        params: FileSystemDeleteFileParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::delete_file(params))
    }

    pub fn export_evidence(
        &mut self,
        params: EvidenceExportParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::export_evidence(params))
    }

    pub fn export_handoff_bundle(
        &mut self,
        params: AgentSessionHandoffBundleExportParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::export_handoff_bundle(params))
    }

    pub fn export_replay_case(
        &mut self,
        params: AgentSessionReplayCaseExportParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::export_replay_case(params))
    }

    pub fn export_analysis_handoff(
        &mut self,
        params: AgentSessionAnalysisHandoffExportParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::export_analysis_handoff(params))
    }

    pub fn export_review_decision_template(
        &mut self,
        params: AgentSessionReviewDecisionTemplateExportParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::export_review_decision_template(params))
    }

    pub fn save_review_decision(
        &mut self,
        params: AgentSessionReviewDecisionSaveParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::save_review_decision(params))
    }

    pub fn start_turn(
        &mut self,
        params: AgentSessionTurnStartParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::start_turn(params))
    }

    pub fn cancel_turn(
        &mut self,
        params: AgentSessionTurnCancelParams,
    ) -> Result<JsonRpcRequest, ClientError> {
        self.typed_request(typed::cancel_turn(params))
    }

    pub fn typed_request<P: Serialize>(
        &mut self,
        request: TypedRequest<P>,
    ) -> Result<JsonRpcRequest, ClientError> {
        let (method, params) = request.into_parts();
        self.request(method, params)
    }

    pub fn request(
        &mut self,
        method: impl Into<String>,
        params: impl Serialize,
    ) -> Result<JsonRpcRequest, ClientError> {
        let id = self.next_id();
        Ok(JsonRpcRequest::new(
            id,
            method,
            Some(serde_json::to_value(params)?),
        ))
    }

    pub fn encode_request(request: JsonRpcRequest) -> Result<String, ClientError> {
        Ok(encode_message(&JsonRpcMessage::Request(request))?)
    }

    pub fn encode_notification(notification: JsonRpcNotification) -> Result<String, ClientError> {
        Ok(encode_message(&JsonRpcMessage::Notification(notification))?)
    }

    pub fn event(message: JsonRpcMessage) -> Result<ClientEvent, ClientError> {
        ClientEvent::try_from(message)
    }

    fn next_id(&mut self) -> RequestId {
        let id = self.next_request_id;
        self.next_request_id += 1;
        RequestId::Integer(id)
    }
}

pub mod typed {
    use super::*;

    pub fn initialize(params: InitializeParams) -> TypedRequest<InitializeParams> {
        TypedRequest::new(METHOD_INITIALIZE, params)
    }

    pub fn list_capabilities(params: CapabilityListParams) -> TypedRequest<CapabilityListParams> {
        TypedRequest::new(METHOD_CAPABILITY_LIST, params)
    }

    pub fn list_sessions(params: AgentSessionListParams) -> TypedRequest<AgentSessionListParams> {
        TypedRequest::new(METHOD_THREAD_LIST, params)
    }

    pub fn read_thread(params: ThreadReadParams) -> TypedRequest<ThreadReadParams> {
        TypedRequest::new(METHOD_THREAD_READ, params)
    }

    pub fn list_threads(params: ThreadListParams) -> TypedRequest<ThreadListParams> {
        TypedRequest::new(METHOD_THREAD_LIST, params)
    }

    pub fn archive_thread(params: ThreadArchiveParams) -> TypedRequest<ThreadArchiveParams> {
        TypedRequest::new(METHOD_THREAD_ARCHIVE, params)
    }

    pub fn unarchive_thread(params: ThreadUnarchiveParams) -> TypedRequest<ThreadUnarchiveParams> {
        TypedRequest::new(METHOD_THREAD_UNARCHIVE, params)
    }

    pub fn run_thread_shell_command(
        params: ThreadShellCommandParams,
    ) -> TypedRequest<ThreadShellCommandParams> {
        TypedRequest::new(METHOD_THREAD_SHELL_COMMAND, params)
    }

    pub fn list_thread_turns(params: ThreadTurnsListParams) -> TypedRequest<ThreadTurnsListParams> {
        TypedRequest::new(METHOD_THREAD_TURNS_LIST, params)
    }

    pub fn list_thread_items(params: ThreadItemsListParams) -> TypedRequest<ThreadItemsListParams> {
        TypedRequest::new(METHOD_THREAD_ITEMS_LIST, params)
    }

    pub fn start_session(params: AgentSessionStartParams) -> TypedRequest<AgentSessionStartParams> {
        TypedRequest::new(METHOD_THREAD_START, params)
    }

    pub fn read_session(params: AgentSessionReadParams) -> TypedRequest<AgentSessionReadParams> {
        TypedRequest::new(METHOD_THREAD_READ, params)
    }

    pub fn read_workflow(params: WorkflowReadParams) -> TypedRequest<WorkflowReadParams> {
        TypedRequest::new(METHOD_WORKFLOW_READ, params)
    }

    pub fn cancel_workflow(params: WorkflowCancelParams) -> TypedRequest<WorkflowCancelParams> {
        TypedRequest::new(METHOD_WORKFLOW_CANCEL, params)
    }

    pub fn retry_workflow(params: WorkflowRetryParams) -> TypedRequest<WorkflowRetryParams> {
        TypedRequest::new(METHOD_WORKFLOW_RETRY, params)
    }

    pub fn respond_workflow(params: WorkflowRespondParams) -> TypedRequest<WorkflowRespondParams> {
        TypedRequest::new(METHOD_WORKFLOW_RESPOND, params)
    }

    pub fn list_workspaces() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_WORKSPACE_LIST, serde_json::json!({}))
    }

    pub fn read_workspace(params: WorkspaceReadParams) -> TypedRequest<WorkspaceReadParams> {
        TypedRequest::new(METHOD_WORKSPACE_READ, params)
    }

    pub fn read_workspace_by_path(
        params: WorkspacePathReadParams,
    ) -> TypedRequest<WorkspacePathReadParams> {
        TypedRequest::new(METHOD_WORKSPACE_BY_PATH_READ, params)
    }

    pub fn ensure_workspace(
        params: WorkspaceEnsureProjectParams,
    ) -> TypedRequest<WorkspaceEnsureProjectParams> {
        TypedRequest::new(METHOD_WORKSPACE_ENSURE, params)
    }

    pub fn read_default_workspace() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_WORKSPACE_DEFAULT_READ, serde_json::json!({}))
    }

    pub fn ensure_default_workspace() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_WORKSPACE_DEFAULT_ENSURE, serde_json::json!({}))
    }

    pub fn read_workspace_projects_root() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_WORKSPACE_PROJECTS_ROOT_READ, serde_json::json!({}))
    }

    pub fn resolve_workspace_project_path(
        params: WorkspaceProjectPathResolveParams,
    ) -> TypedRequest<WorkspaceProjectPathResolveParams> {
        TypedRequest::new(METHOD_WORKSPACE_PROJECT_PATH_RESOLVE, params)
    }

    pub fn ensure_workspace_ready(
        params: WorkspaceEnsureParams,
    ) -> TypedRequest<WorkspaceEnsureParams> {
        TypedRequest::new(METHOD_WORKSPACE_ENSURE_READY, params)
    }

    pub fn request_workspace_right_surface(
        params: WorkspaceRightSurfaceRequestParams,
    ) -> TypedRequest<WorkspaceRightSurfaceRequestParams> {
        TypedRequest::new(METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST, params)
    }

    pub fn list_workspace_right_surface_pending(
        params: WorkspaceRightSurfacePendingListParams,
    ) -> TypedRequest<WorkspaceRightSurfacePendingListParams> {
        TypedRequest::new(METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_LIST, params)
    }

    pub fn consume_workspace_right_surface_pending(
        params: WorkspaceRightSurfacePendingConsumeParams,
    ) -> TypedRequest<WorkspaceRightSurfacePendingConsumeParams> {
        TypedRequest::new(METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CONSUME, params)
    }

    pub fn dismiss_workspace_right_surface_pending(
        params: WorkspaceRightSurfacePendingDismissParams,
    ) -> TypedRequest<WorkspaceRightSurfacePendingDismissParams> {
        TypedRequest::new(METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_DISMISS, params)
    }

    pub fn list_browser_session_targets(
        params: BrowserSessionTargetListParams,
    ) -> TypedRequest<BrowserSessionTargetListParams> {
        TypedRequest::new(METHOD_BROWSER_SESSION_TARGET_LIST, params)
    }

    pub fn open_browser_session(
        params: BrowserSessionOpenParams,
    ) -> TypedRequest<BrowserSessionOpenParams> {
        TypedRequest::new(METHOD_BROWSER_SESSION_OPEN, params)
    }

    pub fn read_browser_session(
        params: BrowserSessionIdParams,
    ) -> TypedRequest<BrowserSessionIdParams> {
        TypedRequest::new(METHOD_BROWSER_SESSION_READ, params)
    }

    pub fn close_browser_session(
        params: BrowserSessionIdParams,
    ) -> TypedRequest<BrowserSessionIdParams> {
        TypedRequest::new(METHOD_BROWSER_SESSION_CLOSE, params)
    }

    pub fn list_browser_session_events(
        params: BrowserSessionEventListParams,
    ) -> TypedRequest<BrowserSessionEventListParams> {
        TypedRequest::new(METHOD_BROWSER_SESSION_EVENT_LIST, params)
    }

    pub fn execute_browser_session_action(
        params: BrowserSessionActionExecuteParams,
    ) -> TypedRequest<BrowserSessionActionExecuteParams> {
        TypedRequest::new(METHOD_BROWSER_SESSION_ACTION_EXECUTE, params)
    }

    pub fn list_skills() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_SKILL_LIST, serde_json::json!({}))
    }

    pub fn read_skill(params: SkillReadParams) -> TypedRequest<SkillReadParams> {
        TypedRequest::new(METHOD_SKILL_READ, params)
    }

    pub fn inspect_local_skill_detail(
        params: SkillLocalDetailInspectParams,
    ) -> TypedRequest<SkillLocalDetailInspectParams> {
        TypedRequest::new(METHOD_SKILL_LOCAL_DETAIL_INSPECT, params)
    }

    pub fn rename_local_skill(
        params: SkillLocalRenameParams,
    ) -> TypedRequest<SkillLocalRenameParams> {
        TypedRequest::new(METHOD_SKILL_LOCAL_RENAME, params)
    }

    pub fn inspect_local_skill_package(
        params: SkillPackageLocalInspectParams,
    ) -> TypedRequest<SkillPackageLocalInspectParams> {
        TypedRequest::new(METHOD_SKILL_PACKAGE_LOCAL_INSPECT, params)
    }

    pub fn install_local_skill_package(
        params: SkillPackageLocalInstallParams,
    ) -> TypedRequest<SkillPackageLocalInstallParams> {
        TypedRequest::new(METHOD_SKILL_PACKAGE_LOCAL_INSTALL, params)
    }

    pub fn replace_local_skill_package(
        params: SkillPackageLocalReplaceParams,
    ) -> TypedRequest<SkillPackageLocalReplaceParams> {
        TypedRequest::new(METHOD_SKILL_PACKAGE_LOCAL_REPLACE, params)
    }

    pub fn export_skill_package(
        params: SkillPackageExportParams,
    ) -> TypedRequest<SkillPackageExportParams> {
        TypedRequest::new(METHOD_SKILL_PACKAGE_EXPORT, params)
    }

    pub fn install_marketplace_skill(
        params: SkillMarketplaceInstallParams,
    ) -> TypedRequest<SkillMarketplaceInstallParams> {
        TypedRequest::new(METHOD_SKILL_MARKETPLACE_INSTALL, params)
    }

    pub fn install_skill_from_download_url(
        params: SkillDownloadInstallParams,
    ) -> TypedRequest<SkillDownloadInstallParams> {
        TypedRequest::new(METHOD_SKILL_PACKAGE_DOWNLOAD_INSTALL, params)
    }

    pub fn read_gateway_channel_status(
        params: GatewayChannelStatusParams,
    ) -> TypedRequest<GatewayChannelStatusParams> {
        TypedRequest::new(METHOD_GATEWAY_CHANNEL_STATUS, params)
    }

    pub fn probe_gateway_tunnel() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_GATEWAY_TUNNEL_PROBE, serde_json::json!({}))
    }

    pub fn detect_gateway_tunnel_cloudflared() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(
            METHOD_GATEWAY_TUNNEL_CLOUDFLARED_DETECT,
            serde_json::json!({}),
        )
    }

    pub fn install_gateway_tunnel_cloudflared(
        params: GatewayTunnelCloudflaredInstallParams,
    ) -> TypedRequest<GatewayTunnelCloudflaredInstallParams> {
        TypedRequest::new(METHOD_GATEWAY_TUNNEL_CLOUDFLARED_INSTALL, params)
    }

    pub fn create_gateway_tunnel(
        params: GatewayTunnelCreateParams,
    ) -> TypedRequest<GatewayTunnelCreateParams> {
        TypedRequest::new(METHOD_GATEWAY_TUNNEL_CREATE, params)
    }

    pub fn start_gateway_tunnel() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_GATEWAY_TUNNEL_START, serde_json::json!({}))
    }

    pub fn stop_gateway_tunnel() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_GATEWAY_TUNNEL_STOP, serde_json::json!({}))
    }

    pub fn restart_gateway_tunnel() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_GATEWAY_TUNNEL_RESTART, serde_json::json!({}))
    }

    pub fn read_gateway_tunnel_status() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_GATEWAY_TUNNEL_STATUS, serde_json::json!({}))
    }

    pub fn sync_gateway_tunnel_webhook_url(
        params: GatewayTunnelSyncWebhookUrlParams,
    ) -> TypedRequest<GatewayTunnelSyncWebhookUrlParams> {
        TypedRequest::new(METHOD_GATEWAY_TUNNEL_SYNC_WEBHOOK_URL, params)
    }

    pub fn list_wechat_channel_accounts() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_WECHAT_CHANNEL_ACCOUNT_LIST, serde_json::json!({}))
    }

    pub fn create_image_media_task_artifact(
        params: MediaTaskArtifactImageCreateParams,
    ) -> TypedRequest<MediaTaskArtifactImageCreateParams> {
        TypedRequest::new(METHOD_MEDIA_TASK_ARTIFACT_IMAGE_CREATE, params)
    }

    pub fn create_audio_media_task_artifact(
        params: MediaTaskArtifactAudioCreateParams,
    ) -> TypedRequest<MediaTaskArtifactAudioCreateParams> {
        TypedRequest::new(METHOD_MEDIA_TASK_ARTIFACT_AUDIO_CREATE, params)
    }

    pub fn complete_audio_media_task_artifact(
        params: MediaTaskArtifactAudioCompleteParams,
    ) -> TypedRequest<MediaTaskArtifactAudioCompleteParams> {
        TypedRequest::new(METHOD_MEDIA_TASK_ARTIFACT_AUDIO_COMPLETE, params)
    }

    pub fn get_media_task_artifact(
        params: MediaTaskArtifactLookupParams,
    ) -> TypedRequest<MediaTaskArtifactLookupParams> {
        TypedRequest::new(METHOD_MEDIA_TASK_ARTIFACT_GET, params)
    }

    pub fn list_media_task_artifacts(
        params: MediaTaskArtifactListParams,
    ) -> TypedRequest<MediaTaskArtifactListParams> {
        TypedRequest::new(METHOD_MEDIA_TASK_ARTIFACT_LIST, params)
    }

    pub fn cancel_media_task_artifact(
        params: MediaTaskArtifactLookupParams,
    ) -> TypedRequest<MediaTaskArtifactLookupParams> {
        TypedRequest::new(METHOD_MEDIA_TASK_ARTIFACT_CANCEL, params)
    }

    pub fn list_workspace_skill_bindings(
        params: WorkspaceSkillBindingsListParams,
    ) -> TypedRequest<WorkspaceSkillBindingsListParams> {
        TypedRequest::new(METHOD_WORKSPACE_SKILL_BINDINGS_LIST, params)
    }

    pub fn inspect_plugin_local_package(
        params: PluginLocalPackageInspectParams,
    ) -> TypedRequest<PluginLocalPackageInspectParams> {
        TypedRequest::new(METHOD_PLUGIN_LOCAL_PACKAGE_INSPECT, params)
    }

    pub fn export_plugin_local_package(
        params: PluginLocalPackageExportParams,
    ) -> TypedRequest<PluginLocalPackageExportParams> {
        TypedRequest::new(METHOD_PLUGIN_LOCAL_PACKAGE_EXPORT, params)
    }

    pub fn fetch_plugin_cloud_package(
        params: PluginFetchCloudPackageParams,
    ) -> TypedRequest<PluginFetchCloudPackageParams> {
        TypedRequest::new(METHOD_PLUGIN_PACKAGE_FETCH_CLOUD, params)
    }

    pub fn save_plugin_installed(
        params: PluginInstalledSaveParams,
    ) -> TypedRequest<PluginInstalledSaveParams> {
        TypedRequest::new(METHOD_PLUGIN_INSTALLED_SAVE, params)
    }

    pub fn list_plugin_installed() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_PLUGIN_INSTALLED_LIST, serde_json::json!({}))
    }

    pub fn set_plugin_installed_disabled(
        params: PluginInstalledDisabledSetParams,
    ) -> TypedRequest<PluginInstalledDisabledSetParams> {
        TypedRequest::new(METHOD_PLUGIN_INSTALLED_DISABLED_SET, params)
    }

    pub fn preview_plugin_uninstall(
        params: PluginUninstallRehearsalParams,
    ) -> TypedRequest<PluginUninstallRehearsalParams> {
        TypedRequest::new(METHOD_PLUGIN_INSTALLED_UNINSTALL_REHEARSAL, params)
    }

    pub fn uninstall_plugin(params: PluginUninstallParams) -> TypedRequest<PluginUninstallParams> {
        TypedRequest::new(METHOD_PLUGIN_INSTALLED_UNINSTALL, params)
    }

    pub fn prepare_plugin_shell(
        params: PluginShellPrepareParams,
    ) -> TypedRequest<PluginShellPrepareParams> {
        TypedRequest::new(METHOD_PLUGIN_SHELL_PREPARE, params)
    }

    pub fn start_plugin_ui_runtime(
        params: PluginUiRuntimeStartParams,
    ) -> TypedRequest<PluginUiRuntimeStartParams> {
        TypedRequest::new(METHOD_PLUGIN_UI_RUNTIME_START, params)
    }

    pub fn plugin_ui_runtime_status(
        params: PluginUiRuntimeStatusParams,
    ) -> TypedRequest<PluginUiRuntimeStatusParams> {
        TypedRequest::new(METHOD_PLUGIN_UI_RUNTIME_STATUS, params)
    }

    pub fn stop_plugin_ui_runtime(
        params: PluginUiRuntimeStopParams,
    ) -> TypedRequest<PluginUiRuntimeStopParams> {
        TypedRequest::new(METHOD_PLUGIN_UI_RUNTIME_STOP, params)
    }

    pub fn list_knowledge_packs(
        params: KnowledgeListPacksParams,
    ) -> TypedRequest<KnowledgeListPacksParams> {
        TypedRequest::new(METHOD_KNOWLEDGE_PACK_LIST, params)
    }

    pub fn read_knowledge_pack(
        params: KnowledgeReadPackParams,
    ) -> TypedRequest<KnowledgeReadPackParams> {
        TypedRequest::new(METHOD_KNOWLEDGE_PACK_READ, params)
    }

    pub fn import_knowledge_source(
        params: KnowledgeImportSourceParams,
    ) -> TypedRequest<KnowledgeImportSourceParams> {
        TypedRequest::new(METHOD_KNOWLEDGE_SOURCE_IMPORT, params)
    }

    pub fn compile_knowledge_pack(
        params: KnowledgeCompilePackParams,
    ) -> TypedRequest<KnowledgeCompilePackParams> {
        TypedRequest::new(METHOD_KNOWLEDGE_PACK_COMPILE, params)
    }

    pub fn set_default_knowledge_pack(
        params: KnowledgeSetDefaultPackParams,
    ) -> TypedRequest<KnowledgeSetDefaultPackParams> {
        TypedRequest::new(METHOD_KNOWLEDGE_PACK_DEFAULT_SET, params)
    }

    pub fn update_knowledge_pack_status(
        params: KnowledgeUpdatePackStatusParams,
    ) -> TypedRequest<KnowledgeUpdatePackStatusParams> {
        TypedRequest::new(METHOD_KNOWLEDGE_PACK_STATUS_UPDATE, params)
    }

    pub fn resolve_knowledge_context(
        params: KnowledgeResolveContextParams,
    ) -> TypedRequest<KnowledgeResolveContextParams> {
        TypedRequest::new(METHOD_KNOWLEDGE_CONTEXT_RESOLVE, params)
    }

    pub fn validate_knowledge_context_run(
        params: KnowledgeValidateContextRunParams,
    ) -> TypedRequest<KnowledgeValidateContextRunParams> {
        TypedRequest::new(METHOD_KNOWLEDGE_CONTEXT_RUN_VALIDATE, params)
    }

    pub fn list_automation_jobs() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_AUTOMATION_JOB_LIST, serde_json::json!({}))
    }

    pub fn read_automation_scheduler_config() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(
            METHOD_AUTOMATION_SCHEDULER_CONFIG_READ,
            serde_json::json!({}),
        )
    }

    pub fn update_automation_scheduler_config(
        params: AutomationSchedulerConfigUpdateParams,
    ) -> TypedRequest<AutomationSchedulerConfigUpdateParams> {
        TypedRequest::new(METHOD_AUTOMATION_SCHEDULER_CONFIG_UPDATE, params)
    }

    pub fn read_automation_scheduler_status() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_AUTOMATION_SCHEDULER_STATUS, serde_json::json!({}))
    }

    pub fn read_automation_job(
        params: AutomationJobIdParams,
    ) -> TypedRequest<AutomationJobIdParams> {
        TypedRequest::new(METHOD_AUTOMATION_JOB_READ, params)
    }

    pub fn create_automation_job(
        params: AutomationJobCreateParams,
    ) -> TypedRequest<AutomationJobCreateParams> {
        TypedRequest::new(METHOD_AUTOMATION_JOB_CREATE, params)
    }

    pub fn update_automation_job(
        params: AutomationJobUpdateParams,
    ) -> TypedRequest<AutomationJobUpdateParams> {
        TypedRequest::new(METHOD_AUTOMATION_JOB_UPDATE, params)
    }

    pub fn delete_automation_job(
        params: AutomationJobIdParams,
    ) -> TypedRequest<AutomationJobIdParams> {
        TypedRequest::new(METHOD_AUTOMATION_JOB_DELETE, params)
    }

    pub fn run_automation_job_now(
        params: AutomationJobIdParams,
    ) -> TypedRequest<AutomationJobIdParams> {
        TypedRequest::new(METHOD_AUTOMATION_JOB_RUN_NOW, params)
    }

    pub fn read_automation_health(
        params: AutomationJobHealthParams,
    ) -> TypedRequest<AutomationJobHealthParams> {
        TypedRequest::new(METHOD_AUTOMATION_JOB_HEALTH, params)
    }

    pub fn read_automation_run_history(
        params: AutomationJobRunHistoryParams,
    ) -> TypedRequest<AutomationJobRunHistoryParams> {
        TypedRequest::new(METHOD_AUTOMATION_JOB_RUN_HISTORY, params)
    }

    pub fn preview_automation_schedule(
        params: AutomationScheduleParams,
    ) -> TypedRequest<AutomationScheduleParams> {
        TypedRequest::new(METHOD_AUTOMATION_SCHEDULE_PREVIEW, params)
    }

    pub fn validate_automation_schedule(
        params: AutomationScheduleParams,
    ) -> TypedRequest<AutomationScheduleParams> {
        TypedRequest::new(METHOD_AUTOMATION_SCHEDULE_VALIDATE, params)
    }

    pub fn list_mcp_servers() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_MCP_SERVER_LIST, serde_json::json!({}))
    }

    pub fn list_mcp_servers_with_status() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_MCP_SERVER_STATUS_LIST, serde_json::json!({}))
    }

    pub fn create_mcp_server(params: McpServerCreateParams) -> TypedRequest<McpServerCreateParams> {
        TypedRequest::new(METHOD_MCP_SERVER_CREATE, params)
    }

    pub fn update_mcp_server(params: McpServerUpdateParams) -> TypedRequest<McpServerUpdateParams> {
        TypedRequest::new(METHOD_MCP_SERVER_UPDATE, params)
    }

    pub fn delete_mcp_server(params: McpServerDeleteParams) -> TypedRequest<McpServerDeleteParams> {
        TypedRequest::new(METHOD_MCP_SERVER_DELETE, params)
    }

    pub fn set_mcp_server_enabled(
        params: McpServerEnabledSetParams,
    ) -> TypedRequest<McpServerEnabledSetParams> {
        TypedRequest::new(METHOD_MCP_SERVER_ENABLED_SET, params)
    }

    pub fn import_mcp_servers_from_app(
        params: McpServerImportFromAppParams,
    ) -> TypedRequest<McpServerImportFromAppParams> {
        TypedRequest::new(METHOD_MCP_SERVER_IMPORT_FROM_APP, params)
    }

    pub fn sync_all_mcp_servers_to_live() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_MCP_SERVER_SYNC_ALL_TO_LIVE, serde_json::json!({}))
    }

    pub fn start_mcp_server(params: McpServerStartParams) -> TypedRequest<McpServerStartParams> {
        TypedRequest::new(METHOD_MCP_SERVER_START, params)
    }

    pub fn stop_mcp_server(params: McpServerStopParams) -> TypedRequest<McpServerStopParams> {
        TypedRequest::new(METHOD_MCP_SERVER_STOP, params)
    }

    pub fn list_mcp_tools() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_MCP_TOOL_LIST, serde_json::json!({}))
    }

    pub fn list_mcp_tools_for_context(
        params: McpToolListForContextParams,
    ) -> TypedRequest<McpToolListForContextParams> {
        TypedRequest::new(METHOD_MCP_TOOL_LIST_FOR_CONTEXT, params)
    }

    pub fn search_mcp_tools(params: McpToolSearchParams) -> TypedRequest<McpToolSearchParams> {
        TypedRequest::new(METHOD_MCP_TOOL_SEARCH, params)
    }

    pub fn call_mcp_tool(params: McpToolCallParams) -> TypedRequest<McpToolCallParams> {
        TypedRequest::new(METHOD_MCP_TOOL_CALL, params)
    }

    pub fn call_mcp_tool_with_caller(
        params: McpToolCallWithCallerParams,
    ) -> TypedRequest<McpToolCallWithCallerParams> {
        TypedRequest::new(METHOD_MCP_TOOL_CALL_WITH_CALLER, params)
    }

    pub fn list_mcp_prompts() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_MCP_PROMPT_LIST, serde_json::json!({}))
    }

    pub fn get_mcp_prompt(
        params: McpPromptGetParams,
    ) -> Result<TypedRequest<McpPromptGetParams>, ClientError> {
        validate_mcp_prompt_target(&params.server, &params.name)?;
        Ok(TypedRequest::new(METHOD_MCP_PROMPT_GET, params))
    }

    pub fn list_mcp_resources() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_MCP_RESOURCE_LIST, serde_json::json!({}))
    }

    pub fn read_mcp_resource(
        params: McpResourceReadParams,
    ) -> Result<TypedRequest<McpResourceReadParams>, ClientError> {
        validate_mcp_resource_target(&params.server, &params.uri)?;
        Ok(TypedRequest::new(METHOD_MCP_RESOURCE_READ, params))
    }

    pub fn subscribe_mcp_resource(
        params: McpResourceSubscribeParams,
    ) -> Result<TypedRequest<McpResourceSubscribeParams>, ClientError> {
        validate_mcp_resource_target(&params.server, &params.uri)?;
        Ok(TypedRequest::new(METHOD_MCP_RESOURCE_SUBSCRIBE, params))
    }

    pub fn unsubscribe_mcp_resource(
        params: McpResourceUnsubscribeParams,
    ) -> Result<TypedRequest<McpResourceUnsubscribeParams>, ClientError> {
        validate_mcp_resource_target(&params.server, &params.uri)?;
        Ok(TypedRequest::new(METHOD_MCP_RESOURCE_UNSUBSCRIBE, params))
    }

    pub fn read_project_memory(
        params: ProjectMemoryReadParams,
    ) -> TypedRequest<ProjectMemoryReadParams> {
        TypedRequest::new(METHOD_PROJECT_MEMORY_READ, params)
    }

    pub fn list_memory_store(params: MemoryStoreListParams) -> TypedRequest<MemoryStoreListParams> {
        TypedRequest::new(METHOD_MEMORY_STORE_LIST, params)
    }

    pub fn read_memory_store(params: MemoryStoreReadParams) -> TypedRequest<MemoryStoreReadParams> {
        TypedRequest::new(METHOD_MEMORY_STORE_READ, params)
    }

    pub fn search_memory_store(
        params: MemoryStoreSearchParams,
    ) -> TypedRequest<MemoryStoreSearchParams> {
        TypedRequest::new(METHOD_MEMORY_STORE_SEARCH, params)
    }

    pub fn add_memory_store_note(
        params: MemoryStoreAddNoteParams,
    ) -> TypedRequest<MemoryStoreAddNoteParams> {
        TypedRequest::new(METHOD_MEMORY_STORE_ADD_NOTE, params)
    }

    pub fn consolidate_memory_store(
        params: MemoryStoreConsolidateParams,
    ) -> TypedRequest<MemoryStoreConsolidateParams> {
        TypedRequest::new(METHOD_MEMORY_STORE_CONSOLIDATE, params)
    }

    pub fn list_memory_store_review_notes(
        params: MemoryStoreReviewListParams,
    ) -> TypedRequest<MemoryStoreReviewListParams> {
        TypedRequest::new(METHOD_MEMORY_STORE_REVIEW_LIST, params)
    }

    pub fn resolve_memory_store_review_note(
        params: MemoryStoreReviewResolveParams,
    ) -> TypedRequest<MemoryStoreReviewResolveParams> {
        TypedRequest::new(METHOD_MEMORY_STORE_REVIEW_RESOLVE, params)
    }

    pub fn health_memory_store(
        params: MemoryStoreRootParams,
    ) -> TypedRequest<MemoryStoreRootParams> {
        TypedRequest::new(METHOD_MEMORY_STORE_HEALTH, params)
    }

    pub fn reset_memory_store(
        params: MemoryStoreResetParams,
    ) -> TypedRequest<MemoryStoreResetParams> {
        TypedRequest::new(METHOD_MEMORY_STORE_RESET, params)
    }

    pub fn rebuild_memory_store_index(
        params: MemoryStoreRootParams,
    ) -> TypedRequest<MemoryStoreRootParams> {
        TypedRequest::new(METHOD_MEMORY_STORE_INDEX_REBUILD, params)
    }

    pub fn list_logs() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_LOG_LIST, serde_json::json!({}))
    }

    pub fn read_persisted_log_tail(
        params: LogPersistedTailParams,
    ) -> TypedRequest<LogPersistedTailParams> {
        TypedRequest::new(METHOD_LOG_PERSISTED_TAIL, params)
    }

    pub fn clear_logs() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_LOG_CLEAR, serde_json::json!({}))
    }

    pub fn clear_diagnostic_log_history() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_LOG_DIAGNOSTIC_HISTORY_CLEAR, serde_json::json!({}))
    }

    pub fn read_server_diagnostics() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_DIAGNOSTICS_SERVER_READ, serde_json::json!({}))
    }

    pub fn read_log_storage_diagnostics() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_DIAGNOSTICS_LOG_STORAGE_READ, serde_json::json!({}))
    }

    pub fn export_support_bundle() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(
            METHOD_DIAGNOSTICS_SUPPORT_BUNDLE_EXPORT,
            serde_json::json!({}),
        )
    }

    pub fn read_windows_startup_diagnostics() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(
            METHOD_DIAGNOSTICS_WINDOWS_STARTUP_READ,
            serde_json::json!({}),
        )
    }

    pub fn read_usage_stats(params: UsageStatsRangeParams) -> TypedRequest<UsageStatsRangeParams> {
        TypedRequest::new(METHOD_USAGE_STATS_READ, params)
    }

    pub fn list_usage_stats_model_ranking(
        params: UsageStatsRangeParams,
    ) -> TypedRequest<UsageStatsRangeParams> {
        TypedRequest::new(METHOD_USAGE_STATS_MODEL_RANKING_LIST, params)
    }

    pub fn list_usage_stats_daily_trends(
        params: UsageStatsRangeParams,
    ) -> TypedRequest<UsageStatsRangeParams> {
        TypedRequest::new(METHOD_USAGE_STATS_DAILY_TRENDS_LIST, params)
    }

    pub fn list_models(params: ModelListParams) -> TypedRequest<ModelListParams> {
        TypedRequest::new(METHOD_MODEL_LIST, params)
    }

    pub fn list_model_preferences() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_MODEL_PREFERENCES_LIST, serde_json::json!({}))
    }

    pub fn read_model_sync_state() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_MODEL_SYNC_STATE_READ, serde_json::json!({}))
    }

    pub fn list_model_providers() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_MODEL_PROVIDER_LIST, serde_json::json!({}))
    }

    pub fn list_model_provider_catalog() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_MODEL_PROVIDER_CATALOG_LIST, serde_json::json!({}))
    }

    pub fn read_model_provider_alias(
        params: ModelProviderAliasReadParams,
    ) -> TypedRequest<ModelProviderAliasReadParams> {
        TypedRequest::new(METHOD_MODEL_PROVIDER_ALIAS_READ, params)
    }

    pub fn list_model_provider_aliases() -> TypedRequest<serde_json::Value> {
        TypedRequest::new(METHOD_MODEL_PROVIDER_ALIAS_LIST, serde_json::json!({}))
    }

    pub fn read_artifacts(params: ArtifactReadParams) -> TypedRequest<ArtifactReadParams> {
        TypedRequest::new(METHOD_ARTIFACT_READ, params)
    }

    pub fn list_directory(
        params: FileSystemListDirectoryParams,
    ) -> TypedRequest<FileSystemListDirectoryParams> {
        TypedRequest::new(METHOD_FILE_SYSTEM_LIST_DIRECTORY, params)
    }

    pub fn read_file_preview(
        params: FileSystemReadFilePreviewParams,
    ) -> TypedRequest<FileSystemReadFilePreviewParams> {
        TypedRequest::new(METHOD_FILE_SYSTEM_READ_FILE_PREVIEW, params)
    }

    pub fn create_file(
        params: FileSystemCreateFileParams,
    ) -> TypedRequest<FileSystemCreateFileParams> {
        TypedRequest::new(METHOD_FILE_SYSTEM_CREATE_FILE, params)
    }

    pub fn create_directory(
        params: FileSystemCreateDirectoryParams,
    ) -> TypedRequest<FileSystemCreateDirectoryParams> {
        TypedRequest::new(METHOD_FILE_SYSTEM_CREATE_DIRECTORY, params)
    }

    pub fn rename_file(
        params: FileSystemRenameFileParams,
    ) -> TypedRequest<FileSystemRenameFileParams> {
        TypedRequest::new(METHOD_FILE_SYSTEM_RENAME_FILE, params)
    }

    pub fn delete_file(
        params: FileSystemDeleteFileParams,
    ) -> TypedRequest<FileSystemDeleteFileParams> {
        TypedRequest::new(METHOD_FILE_SYSTEM_DELETE_FILE, params)
    }

    pub fn export_evidence(params: EvidenceExportParams) -> TypedRequest<EvidenceExportParams> {
        TypedRequest::new(METHOD_EVIDENCE_EXPORT, params)
    }

    pub fn export_handoff_bundle(
        params: AgentSessionHandoffBundleExportParams,
    ) -> TypedRequest<AgentSessionHandoffBundleExportParams> {
        TypedRequest::new(METHOD_AGENT_SESSION_HANDOFF_BUNDLE_EXPORT, params)
    }

    pub fn export_replay_case(
        params: AgentSessionReplayCaseExportParams,
    ) -> TypedRequest<AgentSessionReplayCaseExportParams> {
        TypedRequest::new(METHOD_AGENT_SESSION_REPLAY_CASE_EXPORT, params)
    }

    pub fn export_analysis_handoff(
        params: AgentSessionAnalysisHandoffExportParams,
    ) -> TypedRequest<AgentSessionAnalysisHandoffExportParams> {
        TypedRequest::new(METHOD_AGENT_SESSION_ANALYSIS_HANDOFF_EXPORT, params)
    }

    pub fn export_review_decision_template(
        params: AgentSessionReviewDecisionTemplateExportParams,
    ) -> TypedRequest<AgentSessionReviewDecisionTemplateExportParams> {
        TypedRequest::new(METHOD_AGENT_SESSION_REVIEW_DECISION_TEMPLATE_EXPORT, params)
    }

    pub fn save_review_decision(
        params: AgentSessionReviewDecisionSaveParams,
    ) -> TypedRequest<AgentSessionReviewDecisionSaveParams> {
        TypedRequest::new(METHOD_AGENT_SESSION_REVIEW_DECISION_SAVE, params)
    }

    pub fn start_turn(
        params: AgentSessionTurnStartParams,
    ) -> TypedRequest<AgentSessionTurnStartParams> {
        TypedRequest::new(METHOD_TURN_START, params)
    }

    pub fn cancel_turn(
        params: AgentSessionTurnCancelParams,
    ) -> TypedRequest<AgentSessionTurnCancelParams> {
        TypedRequest::new(METHOD_TURN_INTERRUPT, params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::AgentEvent;
    use app_server_protocol::AgentSessionActionType;
    use app_server_protocol::AgentSessionEventParams;
    use app_server_protocol::ClientCapabilities;
    use app_server_protocol::ClientInfo;
    use app_server_protocol::JsonRpcError;
    use serde_json::json;

    #[test]
    fn initialize_request_uses_stable_method_and_incrementing_id() {
        let mut client = AppServerClient::new();

        let request = client
            .initialize(InitializeParams {
                client_info: ClientInfo {
                    name: "content-studio".to_string(),
                    title: None,
                    version: Some("0.1.0".to_string()),
                },
                capabilities: ClientCapabilities::default(),
            })
            .expect("request");

        assert_eq!(request.id, RequestId::Integer(1));
        assert_eq!(request.method, METHOD_INITIALIZE);
        assert!(AppServerClient::encode_request(request)
            .expect("line")
            .ends_with('\n'));
    }

    #[test]
    fn typed_request_helper_binds_method_to_protocol_params() {
        let typed = typed::read_session(AgentSessionReadParams {
            session_id: "sess_1".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        });

        assert_eq!(typed.method(), METHOD_THREAD_READ);
        assert_eq!(typed.params().session_id, "sess_1");

        let mut client = AppServerClient::new();
        let request = client.typed_request(typed).expect("request");

        assert_eq!(request.id, RequestId::Integer(1));
        assert_eq!(request.method, METHOD_THREAD_READ);
        assert_eq!(request.params.expect("params")["sessionId"], "sess_1");

        let typed = typed::read_workflow(WorkflowReadParams {
            session_id: "sess_1".to_string(),
        });

        assert_eq!(typed.method(), METHOD_WORKFLOW_READ);
        assert_eq!(typed.params().session_id, "sess_1");

        let request = client.typed_request(typed).expect("request");

        assert_eq!(request.id, RequestId::Integer(2));
        assert_eq!(request.method, METHOD_WORKFLOW_READ);
        assert_eq!(request.params.expect("params")["sessionId"], "sess_1");

        let typed = typed::cancel_workflow(WorkflowCancelParams {
            session_id: "sess_1".to_string(),
            workflow_run_id: "run_1".to_string(),
            step_id: Some("step_1".to_string()),
            reason_code: Some("user_requested".to_string()),
            reason: None,
        });

        assert_eq!(typed.method(), METHOD_WORKFLOW_CANCEL);
        assert_eq!(typed.params().workflow_run_id, "run_1");

        let typed = typed::retry_workflow(WorkflowRetryParams {
            session_id: "sess_1".to_string(),
            workflow_run_id: "run_1".to_string(),
            step_id: None,
            reason_code: None,
            reason: Some("retry".to_string()),
        });

        assert_eq!(typed.method(), METHOD_WORKFLOW_RETRY);
        assert_eq!(typed.params().session_id, "sess_1");

        let typed = typed::respond_workflow(WorkflowRespondParams {
            session_id: "sess_1".to_string(),
            workflow_run_id: "run_1".to_string(),
            step_id: Some("approval".to_string()),
            request_id: Some("ask-1".to_string()),
            action_type: Some(AgentSessionActionType::AskUser),
            confirmed: Some(true),
            response: Some(json!({ "approved": true })),
        });

        assert_eq!(typed.method(), METHOD_WORKFLOW_RESPOND);
        assert_eq!(typed.params().step_id.as_deref(), Some("approval"));
    }

    #[test]
    fn canonical_thread_read_helpers_bind_current_methods() {
        let mut client = AppServerClient::new();
        let cases = [
            client
                .read_thread(
                    serde_json::from_value(json!({
                        "threadId": "thread_1",
                        "turnsView": "full"
                    }))
                    .expect("thread/read params"),
                )
                .expect("thread/read request"),
            client
                .list_threads(
                    serde_json::from_value(json!({
                        "cursor": "opaque:thread:2",
                        "includeArchived": true,
                        "turnsView": "summary"
                    }))
                    .expect("thread/list params"),
                )
                .expect("thread/list request"),
            client
                .list_thread_turns(
                    serde_json::from_value(json!({
                        "threadId": "thread_1",
                        "itemsView": "summary"
                    }))
                    .expect("thread/turns/list params"),
                )
                .expect("thread/turns/list request"),
            client
                .list_thread_items(
                    serde_json::from_value(json!({
                        "threadId": "thread_1",
                        "turnId": "turn_1"
                    }))
                    .expect("thread/items/list params"),
                )
                .expect("thread/items/list request"),
        ];

        assert_eq!(
            cases.map(|request| request.method),
            [
                METHOD_THREAD_READ,
                METHOD_THREAD_LIST,
                METHOD_THREAD_TURNS_LIST,
                METHOD_THREAD_ITEMS_LIST,
            ]
        );
    }

    #[test]
    fn list_capabilities_uses_default_params_and_stable_method() {
        let mut client = AppServerClient::new();

        let request = client.list_capabilities_default().expect("request");

        assert_eq!(request.id, RequestId::Integer(1));
        assert_eq!(request.method, METHOD_CAPABILITY_LIST);
        assert_eq!(request.params.expect("params"), json!({}));
    }

    #[test]
    fn mcp_helpers_use_current_methods_and_params() {
        let mut client = AppServerClient::new();

        let servers = client.list_mcp_servers().expect("servers");
        let status = client
            .list_mcp_servers_with_status()
            .expect("server status");
        let server = json!({
            "id": "server-1",
            "name": "filesystem",
            "server_config": { "command": "node", "args": ["server.js"] },
            "enabled_lime": true,
            "enabled_claude": false,
            "enabled_codex": true,
            "enabled_gemini": false,
        });
        let create = client
            .create_mcp_server(McpServerCreateParams {
                server: server.clone(),
            })
            .expect("create server");
        let update = client
            .update_mcp_server(McpServerUpdateParams {
                server: server.clone(),
            })
            .expect("update server");
        let delete = client
            .delete_mcp_server(McpServerDeleteParams {
                id: "server-1".to_string(),
            })
            .expect("delete server");
        let enabled = client
            .set_mcp_server_enabled(McpServerEnabledSetParams {
                id: "server-1".to_string(),
                app_type: "codex".to_string(),
                enabled: true,
            })
            .expect("set server enabled");
        let imported = client
            .import_mcp_servers_from_app(McpServerImportFromAppParams {
                app_type: "codex".to_string(),
            })
            .expect("import servers");
        let synced = client.sync_all_mcp_servers_to_live().expect("sync servers");
        let start = client
            .start_mcp_server(McpServerStartParams {
                name: "filesystem".to_string(),
            })
            .expect("start server");
        let stop = client
            .stop_mcp_server(McpServerStopParams {
                name: "filesystem".to_string(),
            })
            .expect("stop server");
        let tools = client.list_mcp_tools().expect("tools");
        let context_tools = client
            .list_mcp_tools_for_context(McpToolListForContextParams {
                caller: Some("agent-chat".to_string()),
                include_deferred: true,
            })
            .expect("context tools");
        let searched_tools = client
            .search_mcp_tools(McpToolSearchParams {
                query: "file".to_string(),
                caller: Some("agent-chat".to_string()),
                limit: 5,
            })
            .expect("searched tools");
        let tool_call = client
            .call_mcp_tool(McpToolCallParams {
                tool_name: "filesystem.read".to_string(),
                arguments: json!({ "path": "/workspace/README.md" }),
            })
            .expect("tool call");
        let caller_tool_call = client
            .call_mcp_tool_with_caller(McpToolCallWithCallerParams {
                tool_name: "filesystem.read".to_string(),
                arguments: json!({ "path": "/workspace/README.md" }),
                caller: Some("agent-chat".to_string()),
            })
            .expect("caller tool call");
        let prompts = client.list_mcp_prompts().expect("prompts");
        let prompt = client
            .get_mcp_prompt(McpPromptGetParams {
                server: "filesystem".to_string(),
                name: "summarize".to_string(),
                arguments: serde_json::Map::from_iter([(
                    "topic".to_string(),
                    json!("release notes"),
                )]),
            })
            .expect("prompt");
        let resources = client.list_mcp_resources().expect("resources");
        let resource = client
            .read_mcp_resource(McpResourceReadParams {
                server: "filesystem".to_string(),
                uri: "file:///workspace/README.md".to_string(),
            })
            .expect("resource");
        let resource_subscription = client
            .subscribe_mcp_resource(McpResourceSubscribeParams {
                server: "filesystem".to_string(),
                uri: "file:///workspace/README.md".to_string(),
            })
            .expect("subscribe resource");
        let resource_unsubscription = client
            .unsubscribe_mcp_resource(McpResourceUnsubscribeParams {
                server: "filesystem".to_string(),
                uri: "file:///workspace/README.md".to_string(),
            })
            .expect("unsubscribe resource");

        assert_eq!(servers.method, METHOD_MCP_SERVER_LIST);
        assert_eq!(servers.params.expect("params"), json!({}));
        assert_eq!(status.method, METHOD_MCP_SERVER_STATUS_LIST);
        assert_eq!(status.params.expect("params"), json!({}));
        assert_eq!(create.method, METHOD_MCP_SERVER_CREATE);
        assert_eq!(
            create.params.expect("params"),
            json!({ "server": server.clone() })
        );
        assert_eq!(update.method, METHOD_MCP_SERVER_UPDATE);
        assert_eq!(update.params.expect("params"), json!({ "server": server }));
        assert_eq!(delete.method, METHOD_MCP_SERVER_DELETE);
        assert_eq!(delete.params.expect("params"), json!({ "id": "server-1" }));
        assert_eq!(enabled.method, METHOD_MCP_SERVER_ENABLED_SET);
        assert_eq!(
            enabled.params.expect("params"),
            json!({ "id": "server-1", "appType": "codex", "enabled": true })
        );
        assert_eq!(imported.method, METHOD_MCP_SERVER_IMPORT_FROM_APP);
        assert_eq!(
            imported.params.expect("params"),
            json!({ "appType": "codex" })
        );
        assert_eq!(synced.method, METHOD_MCP_SERVER_SYNC_ALL_TO_LIVE);
        assert_eq!(synced.params.expect("params"), json!({}));
        assert_eq!(start.method, METHOD_MCP_SERVER_START);
        assert_eq!(
            start.params.expect("params"),
            json!({ "name": "filesystem" })
        );
        assert_eq!(stop.method, METHOD_MCP_SERVER_STOP);
        assert_eq!(
            stop.params.expect("params"),
            json!({ "name": "filesystem" })
        );
        assert_eq!(tools.method, METHOD_MCP_TOOL_LIST);
        assert_eq!(tools.params.expect("params"), json!({}));
        assert_eq!(context_tools.method, METHOD_MCP_TOOL_LIST_FOR_CONTEXT);
        assert_eq!(
            context_tools.params.expect("params"),
            json!({ "caller": "agent-chat", "includeDeferred": true })
        );
        assert_eq!(searched_tools.method, METHOD_MCP_TOOL_SEARCH);
        assert_eq!(
            searched_tools.params.expect("params"),
            json!({ "query": "file", "caller": "agent-chat", "limit": 5 })
        );
        assert_eq!(tool_call.method, METHOD_MCP_TOOL_CALL);
        assert_eq!(
            tool_call.params.expect("params"),
            json!({
                "toolName": "filesystem.read",
                "arguments": { "path": "/workspace/README.md" },
            })
        );
        assert_eq!(caller_tool_call.method, METHOD_MCP_TOOL_CALL_WITH_CALLER);
        assert_eq!(
            caller_tool_call.params.expect("params"),
            json!({
                "toolName": "filesystem.read",
                "arguments": { "path": "/workspace/README.md" },
                "caller": "agent-chat",
            })
        );
        assert_eq!(prompts.method, METHOD_MCP_PROMPT_LIST);
        assert_eq!(prompts.params.expect("params"), json!({}));
        assert_eq!(prompt.method, METHOD_MCP_PROMPT_GET);
        assert_eq!(
            prompt.params.expect("params"),
            json!({
                "server": "filesystem",
                "name": "summarize",
                "arguments": { "topic": "release notes" },
            })
        );
        assert_eq!(resources.method, METHOD_MCP_RESOURCE_LIST);
        assert_eq!(resources.params.expect("params"), json!({}));
        assert_eq!(resource.method, METHOD_MCP_RESOURCE_READ);
        assert_eq!(
            resource.params.expect("params"),
            json!({
                "server": "filesystem",
                "uri": "file:///workspace/README.md"
            })
        );
        assert_eq!(resource_subscription.method, METHOD_MCP_RESOURCE_SUBSCRIBE);
        assert_eq!(
            resource_subscription.params.expect("params"),
            json!({
                "server": "filesystem",
                "uri": "file:///workspace/README.md"
            })
        );
        assert_eq!(
            resource_unsubscription.method,
            METHOD_MCP_RESOURCE_UNSUBSCRIBE
        );
        assert_eq!(
            resource_unsubscription.params.expect("params"),
            json!({
                "server": "filesystem",
                "uri": "file:///workspace/README.md"
            })
        );
    }

    #[test]
    fn mcp_resource_helpers_reject_empty_exact_target() {
        let mut client = AppServerClient::new();

        let read = client.read_mcp_resource(McpResourceReadParams {
            server: " ".to_string(),
            uri: "docs://readme".to_string(),
        });
        let subscribe = client.subscribe_mcp_resource(McpResourceSubscribeParams {
            server: "docs".to_string(),
            uri: " ".to_string(),
        });
        let unsubscribe = client.unsubscribe_mcp_resource(McpResourceUnsubscribeParams {
            server: " ".to_string(),
            uri: " ".to_string(),
        });

        assert!(matches!(read, Err(ClientError::InvalidParams(_))));
        assert!(matches!(subscribe, Err(ClientError::InvalidParams(_))));
        assert!(matches!(unsubscribe, Err(ClientError::InvalidParams(_))));
    }

    #[test]
    fn mcp_prompt_helper_rejects_empty_exact_target() {
        let mut client = AppServerClient::new();

        let missing_server = client.get_mcp_prompt(McpPromptGetParams {
            server: " ".to_string(),
            name: "summarize".to_string(),
            arguments: serde_json::Map::new(),
        });
        let missing_name = client.get_mcp_prompt(McpPromptGetParams {
            server: "docs".to_string(),
            name: " ".to_string(),
            arguments: serde_json::Map::new(),
        });

        assert!(matches!(missing_server, Err(ClientError::InvalidParams(_))));
        assert!(matches!(missing_name, Err(ClientError::InvalidParams(_))));
    }

    #[test]
    fn list_capabilities_preserves_optional_scope_params() {
        let mut client = AppServerClient::new();

        let request = client
            .list_capabilities(CapabilityListParams {
                app_id: Some("content-studio".to_string()),
                workspace_id: Some("workspace-main".to_string()),
                session_id: Some("sess_1".to_string()),
                cursor: Some("2".to_string()),
                limit: Some(25),
            })
            .expect("request");

        assert_eq!(request.id, RequestId::Integer(1));
        assert_eq!(request.method, METHOD_CAPABILITY_LIST);
        assert_eq!(
            request.params.expect("params"),
            json!({
                "appId": "content-studio",
                "workspaceId": "workspace-main",
                "sessionId": "sess_1",
                "cursor": "2",
                "limit": 25,
            })
        );
    }

    #[test]
    fn list_sessions_preserves_filters_and_stable_method() {
        let mut client = AppServerClient::new();

        let request = client
            .list_sessions(AgentSessionListParams {
                include_archived: Some(true),
                archived_only: Some(false),
                workspace_id: Some("workspace-main".to_string()),
                cwd: None,
                limit: Some(50),
            })
            .expect("request");

        assert_eq!(request.id, RequestId::Integer(1));
        assert_eq!(request.method, METHOD_THREAD_LIST);
        assert_eq!(
            request.params.expect("params"),
            json!({
                "includeArchived": true,
                "archivedOnly": false,
                "workspaceId": "workspace-main",
                "limit": 50,
            })
        );
    }

    #[test]
    fn model_read_helpers_use_current_methods() {
        let mut client = AppServerClient::new();

        let models = client
            .list_models(ModelListParams {
                provider_id: Some("openai".to_string()),
                tier: None,
            })
            .expect("models");
        let preferences = client.list_model_preferences().expect("preferences");
        let sync_state = client.read_model_sync_state().expect("sync state");
        let providers = client.list_model_providers().expect("providers");
        let catalog = client
            .list_model_provider_catalog()
            .expect("provider catalog");
        let alias = client
            .read_model_provider_alias(ModelProviderAliasReadParams {
                provider: "openai".to_string(),
            })
            .expect("alias");
        let aliases = client.list_model_provider_aliases().expect("aliases");

        assert_eq!(models.method, METHOD_MODEL_LIST);
        assert_eq!(
            models.params.expect("params"),
            json!({ "providerId": "openai" })
        );
        assert_eq!(preferences.method, METHOD_MODEL_PREFERENCES_LIST);
        assert_eq!(sync_state.method, METHOD_MODEL_SYNC_STATE_READ);
        assert_eq!(providers.method, METHOD_MODEL_PROVIDER_LIST);
        assert_eq!(catalog.method, METHOD_MODEL_PROVIDER_CATALOG_LIST);
        assert_eq!(alias.method, METHOD_MODEL_PROVIDER_ALIAS_READ);
        assert_eq!(
            alias.params.expect("params"),
            json!({ "provider": "openai" })
        );
        assert_eq!(aliases.method, METHOD_MODEL_PROVIDER_ALIAS_LIST);
    }

    #[test]
    fn workspace_helpers_use_current_methods() {
        let mut client = AppServerClient::new();

        let workspaces = client.list_workspaces().expect("workspaces");
        let workspace = client
            .read_workspace(WorkspaceReadParams {
                id: "workspace-main".to_string(),
            })
            .expect("workspace");
        let workspace_by_path = client
            .read_workspace_by_path(WorkspacePathReadParams {
                root_path: "/workspace/project".to_string(),
            })
            .expect("workspace by path");
        let ensured_workspace = client
            .ensure_workspace(WorkspaceEnsureProjectParams {
                name: "content-studio".to_string(),
                root_path: "/workspace/content-studio".to_string(),
                workspace_type: Some("general".to_string()),
            })
            .expect("ensure workspace");
        let default_workspace = client.read_default_workspace().expect("default workspace");
        let ensured_default = client
            .ensure_default_workspace()
            .expect("ensure default workspace");
        let projects_root = client
            .read_workspace_projects_root()
            .expect("projects root");
        let project_path = client
            .resolve_workspace_project_path(WorkspaceProjectPathResolveParams {
                name: "content-studio".to_string(),
                parent_root_path: Some("/workspace".to_string()),
            })
            .expect("project path");
        let ready = client
            .ensure_workspace_ready(WorkspaceEnsureParams {
                id: "workspace-main".to_string(),
            })
            .expect("ready");
        let right_surface_request = client
            .request_workspace_right_surface(WorkspaceRightSurfaceRequestParams {
                workspace_id: Some("workspace-main".to_string()),
                workspace_root: Some("/workspace/project".to_string()),
                session_id: Some("sess-main".to_string()),
                surface_kind: "objectCanvas".to_string(),
                origin: "mcp:browser".to_string(),
                reason: Some("Browser candidate".to_string()),
                priority: Some("high".to_string()),
                candidate_id: Some("candidate-1".to_string()),
                ttl_ms: Some(60_000),
                metadata: Some(json!({ "source": "browser-assist" })),
            })
            .expect("right surface request");
        let right_surface_pending = client
            .list_workspace_right_surface_pending(WorkspaceRightSurfacePendingListParams {
                workspace_id: Some("workspace-main".to_string()),
                workspace_root: Some("/workspace/project".to_string()),
                session_id: Some("sess-main".to_string()),
                surface_kind: Some("objectCanvas".to_string()),
                limit: Some(10),
            })
            .expect("right surface pending");
        let right_surface_consume = client
            .consume_workspace_right_surface_pending(WorkspaceRightSurfacePendingConsumeParams {
                request_id: Some("right-surface:req-1".to_string()),
                request_ids: vec!["right-surface:req-2".to_string()],
            })
            .expect("right surface consume");
        let right_surface_dismiss = client
            .dismiss_workspace_right_surface_pending(WorkspaceRightSurfacePendingDismissParams {
                request_id: Some("right-surface:req-3".to_string()),
                request_ids: vec!["right-surface:req-4".to_string()],
                reason: Some("user_closed_surface".to_string()),
            })
            .expect("right surface dismiss");
        let browser_targets = client
            .list_browser_session_targets(BrowserSessionTargetListParams {
                remote_debugging_port: 9222,
            })
            .expect("browser targets");
        let browser_open = client
            .open_browser_session(BrowserSessionOpenParams {
                profile_key: "task-profile".to_string(),
                remote_debugging_port: 9222,
                target_id: Some("target-1".to_string()),
                launch_url: Some("https://example.com".to_string()),
                environment_preset_id: None,
                environment_preset_name: None,
            })
            .expect("browser open");
        let browser_read = client
            .read_browser_session(BrowserSessionIdParams {
                session_id: "browser-session-1".to_string(),
            })
            .expect("browser read");
        let browser_close = client
            .close_browser_session(BrowserSessionIdParams {
                session_id: "browser-session-1".to_string(),
            })
            .expect("browser close");
        let browser_events = client
            .list_browser_session_events(BrowserSessionEventListParams {
                session_id: "browser-session-1".to_string(),
                cursor: Some(3),
            })
            .expect("browser events");
        let browser_action = client
            .execute_browser_session_action(BrowserSessionActionExecuteParams {
                session_id: "browser-session-1".to_string(),
                action: "get_page_info".to_string(),
                args: Some(json!({ "includeMarkdown": true })),
            })
            .expect("browser action");

        assert_eq!(workspaces.method, METHOD_WORKSPACE_LIST);
        assert_eq!(workspaces.params.expect("params"), json!({}));
        assert_eq!(workspace.method, METHOD_WORKSPACE_READ);
        assert_eq!(
            workspace.params.expect("params"),
            json!({ "id": "workspace-main" })
        );
        assert_eq!(workspace_by_path.method, METHOD_WORKSPACE_BY_PATH_READ);
        assert_eq!(
            workspace_by_path.params.expect("params"),
            json!({ "rootPath": "/workspace/project" })
        );
        assert_eq!(ensured_workspace.method, METHOD_WORKSPACE_ENSURE);
        assert_eq!(
            ensured_workspace.params.expect("params"),
            json!({
                "name": "content-studio",
                "rootPath": "/workspace/content-studio",
                "workspaceType": "general",
            })
        );
        assert_eq!(default_workspace.method, METHOD_WORKSPACE_DEFAULT_READ);
        assert_eq!(ensured_default.method, METHOD_WORKSPACE_DEFAULT_ENSURE);
        assert_eq!(projects_root.method, METHOD_WORKSPACE_PROJECTS_ROOT_READ);
        assert_eq!(project_path.method, METHOD_WORKSPACE_PROJECT_PATH_RESOLVE);
        assert_eq!(
            project_path.params.expect("params"),
            json!({
                "name": "content-studio",
                "parentRootPath": "/workspace",
            })
        );
        assert_eq!(ready.method, METHOD_WORKSPACE_ENSURE_READY);
        assert_eq!(
            ready.params.expect("params"),
            json!({ "id": "workspace-main" })
        );
        assert_eq!(
            right_surface_request.method,
            METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST
        );
        assert_eq!(
            right_surface_request.params.expect("params"),
            json!({
                "workspaceId": "workspace-main",
                "workspaceRoot": "/workspace/project",
                "sessionId": "sess-main",
                "surfaceKind": "objectCanvas",
                "origin": "mcp:browser",
                "reason": "Browser candidate",
                "priority": "high",
                "candidateId": "candidate-1",
                "ttlMs": 60000,
                "metadata": { "source": "browser-assist" },
            })
        );
        assert_eq!(
            right_surface_pending.method,
            METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_LIST
        );
        assert_eq!(
            right_surface_pending.params.expect("params"),
            json!({
                "workspaceId": "workspace-main",
                "workspaceRoot": "/workspace/project",
                "sessionId": "sess-main",
                "surfaceKind": "objectCanvas",
                "limit": 10,
            })
        );
        assert_eq!(
            right_surface_consume.method,
            METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CONSUME
        );
        assert_eq!(
            right_surface_consume.params.expect("params"),
            json!({
                "requestId": "right-surface:req-1",
                "requestIds": ["right-surface:req-2"],
            })
        );
        assert_eq!(
            right_surface_dismiss.method,
            METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_DISMISS
        );
        assert_eq!(
            right_surface_dismiss.params.expect("params"),
            json!({
                "requestId": "right-surface:req-3",
                "requestIds": ["right-surface:req-4"],
                "reason": "user_closed_surface",
            })
        );
        assert_eq!(browser_targets.method, METHOD_BROWSER_SESSION_TARGET_LIST);
        assert_eq!(
            browser_targets.params.expect("params"),
            json!({ "remoteDebuggingPort": 9222 })
        );
        assert_eq!(browser_open.method, METHOD_BROWSER_SESSION_OPEN);
        assert_eq!(
            browser_open.params.expect("params"),
            json!({
                "profileKey": "task-profile",
                "remoteDebuggingPort": 9222,
                "targetId": "target-1",
                "launchUrl": "https://example.com",
            })
        );
        assert_eq!(browser_read.method, METHOD_BROWSER_SESSION_READ);
        assert_eq!(
            browser_read.params.expect("params"),
            json!({ "sessionId": "browser-session-1" })
        );
        assert_eq!(browser_close.method, METHOD_BROWSER_SESSION_CLOSE);
        assert_eq!(
            browser_close.params.expect("params"),
            json!({ "sessionId": "browser-session-1" })
        );
        assert_eq!(browser_events.method, METHOD_BROWSER_SESSION_EVENT_LIST);
        assert_eq!(
            browser_events.params.expect("params"),
            json!({
                "sessionId": "browser-session-1",
                "cursor": 3,
            })
        );
        assert_eq!(browser_action.method, METHOD_BROWSER_SESSION_ACTION_EXECUTE);
        assert_eq!(
            browser_action.params.expect("params"),
            json!({
                "sessionId": "browser-session-1",
                "action": "get_page_info",
                "args": { "includeMarkdown": true },
            })
        );
    }

    #[test]
    fn skill_helpers_use_current_methods() {
        let mut client = AppServerClient::new();

        let skills = client.list_skills().expect("skills");
        let skill = client
            .read_skill(SkillReadParams {
                skill_id: "project:article-writer".to_string(),
            })
            .expect("skill");
        let inspect_detail = client
            .inspect_local_skill_detail(SkillLocalDetailInspectParams {
                app: "lime".to_string(),
                directory: "article-writer".to_string(),
            })
            .expect("inspect detail");
        let rename_skill = client
            .rename_local_skill(SkillLocalRenameParams {
                app: "lime".to_string(),
                directory: "article-writer".to_string(),
                new_directory: "article-writer-renamed".to_string(),
            })
            .expect("rename skill");
        let inspect_package = client
            .inspect_local_skill_package(SkillPackageLocalInspectParams {
                app: "lime".to_string(),
                source_path: "/tmp/article-writer.skill".to_string(),
            })
            .expect("inspect package");
        let install_package = client
            .install_local_skill_package(SkillPackageLocalInstallParams {
                app: "lime".to_string(),
                source_path: "/tmp/article-writer.skill".to_string(),
                skill_name: Some("article-writer".to_string()),
            })
            .expect("install package");
        let replace_package = client
            .replace_local_skill_package(SkillPackageLocalReplaceParams {
                app: "lime".to_string(),
                directory: "article-writer".to_string(),
                source_path: "/tmp/article-writer.skill".to_string(),
            })
            .expect("replace package");
        let export_package = client
            .export_skill_package(SkillPackageExportParams {
                app: "lime".to_string(),
                directory: "article-writer".to_string(),
                target_path: "/tmp/article-writer.skills".to_string(),
            })
            .expect("export package");
        let install_marketplace = client
            .install_marketplace_skill(SkillMarketplaceInstallParams {
                app: "lime".to_string(),
                manifest_version: "agentskills.v1".to_string(),
                name: "article-writer".to_string(),
                aliases: vec!["writer".to_string()],
                version: "1.0.0".to_string(),
                content_hash: "sha256-demo".to_string(),
                file_count: 1,
                files: vec![app_server_protocol::SkillMarketplaceBundleFile {
                    path: "SKILL.md".to_string(),
                    content: "# Writer".to_string(),
                    encoding: None,
                    sha256: None,
                }],
            })
            .expect("install marketplace");
        let install_download = client
            .install_skill_from_download_url(SkillDownloadInstallParams {
                app: "lime".to_string(),
                skill_name: "article-writer".to_string(),
                download_url: "https://example.com/article-writer.skill".to_string(),
            })
            .expect("install download");
        let bindings = client
            .list_workspace_skill_bindings(WorkspaceSkillBindingsListParams {
                workspace_root: "/workspace/project".to_string(),
                caller: Some("agent-chat".to_string()),
                workbench: true,
                browser_assist: false,
            })
            .expect("bindings");

        assert_eq!(skills.method, METHOD_SKILL_LIST);
        assert_eq!(skills.params.expect("params"), json!({}));
        assert_eq!(skill.method, METHOD_SKILL_READ);
        assert_eq!(
            skill.params.expect("params"),
            json!({ "skillId": "project:article-writer" })
        );
        assert_eq!(inspect_detail.method, METHOD_SKILL_LOCAL_DETAIL_INSPECT);
        assert_eq!(
            inspect_detail.params.expect("params"),
            json!({ "app": "lime", "directory": "article-writer" })
        );
        assert_eq!(rename_skill.method, METHOD_SKILL_LOCAL_RENAME);
        assert_eq!(
            rename_skill.params.expect("params"),
            json!({
                "app": "lime",
                "directory": "article-writer",
                "newDirectory": "article-writer-renamed",
            })
        );
        assert_eq!(inspect_package.method, METHOD_SKILL_PACKAGE_LOCAL_INSPECT);
        assert_eq!(
            inspect_package.params.expect("params"),
            json!({ "app": "lime", "sourcePath": "/tmp/article-writer.skill" })
        );
        assert_eq!(install_package.method, METHOD_SKILL_PACKAGE_LOCAL_INSTALL);
        assert_eq!(
            install_package.params.expect("params"),
            json!({
                "app": "lime",
                "sourcePath": "/tmp/article-writer.skill",
                "skillName": "article-writer",
            })
        );
        assert_eq!(replace_package.method, METHOD_SKILL_PACKAGE_LOCAL_REPLACE);
        assert_eq!(
            replace_package.params.expect("params"),
            json!({
                "app": "lime",
                "directory": "article-writer",
                "sourcePath": "/tmp/article-writer.skill",
            })
        );
        assert_eq!(export_package.method, METHOD_SKILL_PACKAGE_EXPORT);
        assert_eq!(
            export_package.params.expect("params"),
            json!({
                "app": "lime",
                "directory": "article-writer",
                "targetPath": "/tmp/article-writer.skills",
            })
        );
        assert_eq!(install_marketplace.method, METHOD_SKILL_MARKETPLACE_INSTALL);
        assert_eq!(
            install_marketplace.params.expect("params"),
            json!({
                "app": "lime",
                "manifestVersion": "agentskills.v1",
                "name": "article-writer",
                "aliases": ["writer"],
                "version": "1.0.0",
                "contentHash": "sha256-demo",
                "fileCount": 1,
                "files": [{
                    "path": "SKILL.md",
                    "content": "# Writer",
                }],
            })
        );
        assert_eq!(
            install_download.method,
            METHOD_SKILL_PACKAGE_DOWNLOAD_INSTALL
        );
        assert_eq!(
            install_download.params.expect("params"),
            json!({
                "app": "lime",
                "skillName": "article-writer",
                "downloadUrl": "https://example.com/article-writer.skill",
            })
        );
        assert_eq!(bindings.method, METHOD_WORKSPACE_SKILL_BINDINGS_LIST);
        assert_eq!(
            bindings.params.expect("params"),
            json!({
                "workspaceRoot": "/workspace/project",
                "caller": "agent-chat",
                "workbench": true,
                "browserAssist": false,
            })
        );
    }

    #[test]
    fn app_data_surface_helpers_use_current_methods() {
        let mut client = AppServerClient::new();

        let installed = client.list_plugin_installed().expect("installed plugins");
        let shell_prepare = client
            .prepare_plugin_shell(PluginShellPrepareParams {
                descriptor: json!({
                    "appId": "content-factory-app",
                }),
            })
            .expect("plugin shell prepare");
        let knowledge = client
            .list_knowledge_packs(KnowledgeListPacksParams {
                working_dir: "/workspace/project".to_string(),
                include_archived: true,
            })
            .expect("knowledge packs");
        let knowledge_detail = client
            .read_knowledge_pack(KnowledgeReadPackParams {
                working_dir: "/workspace/project".to_string(),
                name: "sample-product".to_string(),
            })
            .expect("knowledge pack detail");
        let imported_knowledge_source = client
            .import_knowledge_source(KnowledgeImportSourceParams {
                working_dir: "/workspace/project".to_string(),
                pack_name: "sample-product".to_string(),
                description: None,
                pack_type: None,
                language: None,
                source_file_name: None,
                source_text: Some("示例产品事实".to_string()),
            })
            .expect("knowledge source import");
        let compiled_knowledge_pack = client
            .compile_knowledge_pack(KnowledgeCompilePackParams {
                working_dir: "/workspace/project".to_string(),
                name: "sample-product".to_string(),
                builder_runtime: Some(json!({ "enabled": true })),
            })
            .expect("knowledge pack compile");
        let default_knowledge_pack = client
            .set_default_knowledge_pack(KnowledgeSetDefaultPackParams {
                working_dir: "/workspace/project".to_string(),
                name: "sample-product".to_string(),
            })
            .expect("knowledge pack default");
        let updated_knowledge_pack_status = client
            .update_knowledge_pack_status(KnowledgeUpdatePackStatusParams {
                working_dir: "/workspace/project".to_string(),
                name: "sample-product".to_string(),
                status: "ready".to_string(),
            })
            .expect("knowledge pack status");
        let knowledge_context = client
            .resolve_knowledge_context(KnowledgeResolveContextParams {
                working_dir: "/workspace/project".to_string(),
                name: "sample-product".to_string(),
                packs: Vec::new(),
                task: Some("写产品介绍".to_string()),
                max_chars: None,
                activation: None,
                write_run: true,
                run_reason: None,
            })
            .expect("knowledge context");
        let knowledge_context_validation = client
            .validate_knowledge_context_run(KnowledgeValidateContextRunParams {
                working_dir: "/workspace/project".to_string(),
                name: "sample-product".to_string(),
                run_path: "runs/context.json".to_string(),
            })
            .expect("knowledge context validation");
        let scheduler_config = client
            .read_automation_scheduler_config()
            .expect("automation scheduler config");
        let scheduler_config_update = client
            .update_automation_scheduler_config(AutomationSchedulerConfigUpdateParams {
                config: json!({
                    "enabled": true,
                    "poll_interval_secs": 60,
                    "enable_history": true,
                }),
            })
            .expect("automation scheduler config update");
        let scheduler_status = client
            .read_automation_scheduler_status()
            .expect("automation scheduler status");
        let jobs = client.list_automation_jobs().expect("automation jobs");
        let job = client
            .read_automation_job(AutomationJobIdParams {
                id: "job-1".to_string(),
            })
            .expect("automation job");
        let created_job = client
            .create_automation_job(AutomationJobCreateParams {
                request: json!({
                    "name": "每日简报",
                    "workspace_id": "workspace-main",
                    "schedule": {
                        "kind": "every",
                        "every_secs": 3600,
                    },
                    "payload": {
                        "kind": "agent_turn",
                        "prompt": "总结今天重点",
                        "web_search": false,
                    },
                }),
            })
            .expect("automation job create");
        let updated_job = client
            .update_automation_job(AutomationJobUpdateParams {
                id: "job-1".to_string(),
                request: json!({
                    "enabled": false,
                }),
            })
            .expect("automation job update");
        let deleted_job = client
            .delete_automation_job(AutomationJobIdParams {
                id: "job-1".to_string(),
            })
            .expect("automation job delete");
        let run_now = client
            .run_automation_job_now(AutomationJobIdParams {
                id: "job-1".to_string(),
            })
            .expect("automation job run now");
        let health = client
            .read_automation_health(AutomationJobHealthParams {
                query: Some(json!({
                    "top_limit": 3,
                })),
            })
            .expect("automation health");
        let history = client
            .read_automation_run_history(AutomationJobRunHistoryParams {
                id: "job-1".to_string(),
                limit: Some(10),
            })
            .expect("automation run history");
        let preview = client
            .preview_automation_schedule(AutomationScheduleParams {
                schedule: json!({
                    "kind": "every",
                    "every_secs": 3600,
                }),
            })
            .expect("automation schedule preview");
        let validate = client
            .validate_automation_schedule(AutomationScheduleParams {
                schedule: json!({
                    "kind": "every",
                    "every_secs": 3600,
                }),
            })
            .expect("automation schedule validate");
        let memory = client
            .read_project_memory(ProjectMemoryReadParams {
                project_id: "workspace-main".to_string(),
            })
            .expect("project memory");
        let memory_store_list = client
            .list_memory_store(MemoryStoreListParams {
                root: MemoryStoreRootParams {
                    scope: app_server_protocol::MemoryStoreScope::Workspace,
                    workspace_root: Some("/workspace/project".to_string()),
                },
                path: Some("skills".to_string()),
                cursor: None,
                max_results: Some(20),
            })
            .expect("memory store list");
        let memory_store_read = client
            .read_memory_store(MemoryStoreReadParams {
                root: MemoryStoreRootParams {
                    scope: app_server_protocol::MemoryStoreScope::Workspace,
                    workspace_root: Some("/workspace/project".to_string()),
                },
                path: "MEMORY.md".to_string(),
                line_offset: None,
                max_lines: Some(40),
                max_tokens: None,
            })
            .expect("memory store read");
        let memory_store_search = client
            .search_memory_store(MemoryStoreSearchParams {
                root: MemoryStoreRootParams {
                    scope: app_server_protocol::MemoryStoreScope::Workspace,
                    workspace_root: Some("/workspace/project".to_string()),
                },
                queries: vec!["voice".to_string(), "preference".to_string()],
                match_mode: app_server_protocol::MemoryStoreSearchMatchMode::AllWithinLines,
                within_lines: Some(4),
                case_sensitive: false,
                normalized: false,
                context_lines: 0,
                cursor: None,
                max_results: None,
            })
            .expect("memory store search");
        let memory_store_add_note = client
            .add_memory_store_note(MemoryStoreAddNoteParams {
                root: MemoryStoreRootParams {
                    scope: app_server_protocol::MemoryStoreScope::Workspace,
                    workspace_root: Some("/workspace/project".to_string()),
                },
                content: "Prefer concise answers.".to_string(),
                title: Some("Tone note".to_string()),
                slug: None,
            })
            .expect("memory store add note");
        let memory_store_consolidate = client
            .consolidate_memory_store(MemoryStoreConsolidateParams {
                root: MemoryStoreRootParams {
                    scope: app_server_protocol::MemoryStoreScope::Workspace,
                    workspace_root: Some("/workspace/project".to_string()),
                },
                max_notes: Some(10),
            })
            .expect("memory store consolidate");
        let memory_store_review_list = client
            .list_memory_store_review_notes(MemoryStoreReviewListParams {
                root: MemoryStoreRootParams {
                    scope: app_server_protocol::MemoryStoreScope::Workspace,
                    workspace_root: Some("/workspace/project".to_string()),
                },
                cursor: None,
                max_results: Some(10),
            })
            .expect("memory store review list");
        let memory_store_review_resolve = client
            .resolve_memory_store_review_note(MemoryStoreReviewResolveParams {
                root: MemoryStoreRootParams {
                    scope: app_server_protocol::MemoryStoreScope::Workspace,
                    workspace_root: Some("/workspace/project".to_string()),
                },
                path: "extensions/ad_hoc/review/secret.md".to_string(),
                action: MemoryStoreReviewResolveAction::Reject,
            })
            .expect("memory store review resolve");
        let memory_store_health = client
            .health_memory_store(MemoryStoreRootParams {
                scope: app_server_protocol::MemoryStoreScope::Workspace,
                workspace_root: Some("/workspace/project".to_string()),
            })
            .expect("memory store health");
        let memory_store_reset = client
            .reset_memory_store(MemoryStoreResetParams {
                root: MemoryStoreRootParams {
                    scope: app_server_protocol::MemoryStoreScope::Workspace,
                    workspace_root: Some("/workspace/project".to_string()),
                },
            })
            .expect("memory store reset");
        let memory_store_index_rebuild = client
            .rebuild_memory_store_index(MemoryStoreRootParams {
                scope: app_server_protocol::MemoryStoreScope::Workspace,
                workspace_root: Some("/workspace/project".to_string()),
            })
            .expect("memory store index rebuild");
        let logs = client.list_logs().expect("logs");
        let persisted_tail = client
            .read_persisted_log_tail(LogPersistedTailParams { lines: Some(250) })
            .expect("persisted log tail");
        let cleared_logs = client.clear_logs().expect("clear logs");
        let cleared_diagnostic_history = client
            .clear_diagnostic_log_history()
            .expect("clear diagnostic history");

        assert_eq!(installed.method, METHOD_PLUGIN_INSTALLED_LIST);
        assert_eq!(installed.params.expect("params"), json!({}));
        assert_eq!(shell_prepare.method, METHOD_PLUGIN_SHELL_PREPARE);
        assert_eq!(
            shell_prepare.params.expect("params"),
            json!({
                "descriptor": {
                    "appId": "content-factory-app",
                },
            })
        );
        assert_eq!(knowledge.method, METHOD_KNOWLEDGE_PACK_LIST);
        assert_eq!(
            knowledge.params.expect("params"),
            json!({
                "workingDir": "/workspace/project",
                "includeArchived": true,
            })
        );
        assert_eq!(knowledge_detail.method, METHOD_KNOWLEDGE_PACK_READ);
        assert_eq!(
            knowledge_detail.params.expect("params"),
            json!({
                "workingDir": "/workspace/project",
                "name": "sample-product",
            })
        );
        assert_eq!(
            imported_knowledge_source.method,
            METHOD_KNOWLEDGE_SOURCE_IMPORT
        );
        assert_eq!(
            imported_knowledge_source.params.expect("params"),
            json!({
                "workingDir": "/workspace/project",
                "packName": "sample-product",
                "sourceText": "示例产品事实",
            })
        );
        assert_eq!(
            compiled_knowledge_pack.method,
            METHOD_KNOWLEDGE_PACK_COMPILE
        );
        assert_eq!(
            compiled_knowledge_pack.params.expect("params"),
            json!({
                "workingDir": "/workspace/project",
                "name": "sample-product",
                "builderRuntime": {
                    "enabled": true,
                },
            })
        );
        assert_eq!(
            default_knowledge_pack.method,
            METHOD_KNOWLEDGE_PACK_DEFAULT_SET
        );
        assert_eq!(
            default_knowledge_pack.params.expect("params"),
            json!({
                "workingDir": "/workspace/project",
                "name": "sample-product",
            })
        );
        assert_eq!(
            updated_knowledge_pack_status.method,
            METHOD_KNOWLEDGE_PACK_STATUS_UPDATE
        );
        assert_eq!(
            updated_knowledge_pack_status.params.expect("params"),
            json!({
                "workingDir": "/workspace/project",
                "name": "sample-product",
                "status": "ready",
            })
        );
        assert_eq!(knowledge_context.method, METHOD_KNOWLEDGE_CONTEXT_RESOLVE);
        assert_eq!(
            knowledge_context.params.expect("params"),
            json!({
                "workingDir": "/workspace/project",
                "name": "sample-product",
                "task": "写产品介绍",
                "writeRun": true,
            })
        );
        assert_eq!(
            knowledge_context_validation.method,
            METHOD_KNOWLEDGE_CONTEXT_RUN_VALIDATE
        );
        assert_eq!(
            knowledge_context_validation.params.expect("params"),
            json!({
                "workingDir": "/workspace/project",
                "name": "sample-product",
                "runPath": "runs/context.json",
            })
        );
        assert_eq!(
            scheduler_config.method,
            METHOD_AUTOMATION_SCHEDULER_CONFIG_READ
        );
        assert_eq!(
            scheduler_config_update.method,
            METHOD_AUTOMATION_SCHEDULER_CONFIG_UPDATE
        );
        assert_eq!(
            scheduler_config_update.params.expect("params"),
            json!({
                "config": {
                    "enabled": true,
                    "poll_interval_secs": 60,
                    "enable_history": true,
                },
            })
        );
        assert_eq!(scheduler_status.method, METHOD_AUTOMATION_SCHEDULER_STATUS);
        assert_eq!(jobs.method, METHOD_AUTOMATION_JOB_LIST);
        assert_eq!(jobs.params.expect("params"), json!({}));
        assert_eq!(job.method, METHOD_AUTOMATION_JOB_READ);
        assert_eq!(job.params.expect("params"), json!({ "id": "job-1" }));
        assert_eq!(created_job.method, METHOD_AUTOMATION_JOB_CREATE);
        assert_eq!(
            created_job.params.expect("params"),
            json!({
                "request": {
                    "name": "每日简报",
                    "workspace_id": "workspace-main",
                    "schedule": {
                        "kind": "every",
                        "every_secs": 3600,
                    },
                    "payload": {
                        "kind": "agent_turn",
                        "prompt": "总结今天重点",
                        "web_search": false,
                    },
                },
            })
        );
        assert_eq!(updated_job.method, METHOD_AUTOMATION_JOB_UPDATE);
        assert_eq!(
            updated_job.params.expect("params"),
            json!({
                "id": "job-1",
                "request": {
                    "enabled": false,
                },
            })
        );
        assert_eq!(deleted_job.method, METHOD_AUTOMATION_JOB_DELETE);
        assert_eq!(run_now.method, METHOD_AUTOMATION_JOB_RUN_NOW);
        assert_eq!(health.method, METHOD_AUTOMATION_JOB_HEALTH);
        assert_eq!(
            health.params.expect("params"),
            json!({
                "query": {
                    "top_limit": 3,
                },
            })
        );
        assert_eq!(history.method, METHOD_AUTOMATION_JOB_RUN_HISTORY);
        assert_eq!(
            history.params.expect("params"),
            json!({
                "id": "job-1",
                "limit": 10,
            })
        );
        assert_eq!(preview.method, METHOD_AUTOMATION_SCHEDULE_PREVIEW);
        assert_eq!(
            preview.params.expect("params"),
            json!({
                "schedule": {
                    "kind": "every",
                    "every_secs": 3600,
                },
            })
        );
        assert_eq!(validate.method, METHOD_AUTOMATION_SCHEDULE_VALIDATE);
        assert_eq!(memory.method, METHOD_PROJECT_MEMORY_READ);
        assert_eq!(
            memory.params.expect("params"),
            json!({ "projectId": "workspace-main" })
        );
        assert_eq!(memory_store_list.method, METHOD_MEMORY_STORE_LIST);
        assert_eq!(
            memory_store_list.params.expect("params"),
            json!({
                "scope": "workspace",
                "workspaceRoot": "/workspace/project",
                "path": "skills",
                "maxResults": 20,
            })
        );
        assert_eq!(memory_store_read.method, METHOD_MEMORY_STORE_READ);
        assert_eq!(
            memory_store_read.params.expect("params"),
            json!({
                "scope": "workspace",
                "workspaceRoot": "/workspace/project",
                "path": "MEMORY.md",
                "maxLines": 40,
            })
        );
        assert_eq!(memory_store_search.method, METHOD_MEMORY_STORE_SEARCH);
        assert_eq!(
            memory_store_search.params.expect("params"),
            json!({
                "scope": "workspace",
                "workspaceRoot": "/workspace/project",
                "queries": ["voice", "preference"],
                "matchMode": "allWithinLines",
                "withinLines": 4,
                "caseSensitive": false,
                "normalized": false,
                "contextLines": 0,
            })
        );
        assert_eq!(memory_store_add_note.method, METHOD_MEMORY_STORE_ADD_NOTE);
        assert_eq!(
            memory_store_add_note.params.expect("params"),
            json!({
                "scope": "workspace",
                "workspaceRoot": "/workspace/project",
                "content": "Prefer concise answers.",
                "title": "Tone note",
            })
        );
        assert_eq!(
            memory_store_consolidate.method,
            METHOD_MEMORY_STORE_CONSOLIDATE
        );
        assert_eq!(
            memory_store_consolidate.params.expect("params"),
            json!({
                "scope": "workspace",
                "workspaceRoot": "/workspace/project",
                "maxNotes": 10,
            })
        );
        assert_eq!(
            memory_store_review_list.method,
            METHOD_MEMORY_STORE_REVIEW_LIST
        );
        assert_eq!(
            memory_store_review_list.params.expect("params"),
            json!({
                "scope": "workspace",
                "workspaceRoot": "/workspace/project",
                "maxResults": 10,
            })
        );
        assert_eq!(
            memory_store_review_resolve.method,
            METHOD_MEMORY_STORE_REVIEW_RESOLVE
        );
        assert_eq!(
            memory_store_review_resolve.params.expect("params"),
            json!({
                "scope": "workspace",
                "workspaceRoot": "/workspace/project",
                "path": "extensions/ad_hoc/review/secret.md",
                "action": "reject",
            })
        );
        assert_eq!(memory_store_health.method, METHOD_MEMORY_STORE_HEALTH);
        assert_eq!(
            memory_store_health.params.expect("params"),
            json!({
                "scope": "workspace",
                "workspaceRoot": "/workspace/project",
            })
        );
        assert_eq!(memory_store_reset.method, METHOD_MEMORY_STORE_RESET);
        assert_eq!(
            memory_store_reset.params.expect("params"),
            json!({
                "scope": "workspace",
                "workspaceRoot": "/workspace/project",
            })
        );
        assert_eq!(
            memory_store_index_rebuild.method,
            METHOD_MEMORY_STORE_INDEX_REBUILD
        );
        assert_eq!(
            memory_store_index_rebuild.params.expect("params"),
            json!({
                "scope": "workspace",
                "workspaceRoot": "/workspace/project",
            })
        );
        assert_eq!(logs.method, METHOD_LOG_LIST);
        assert_eq!(logs.params.expect("params"), json!({}));
        assert_eq!(persisted_tail.method, METHOD_LOG_PERSISTED_TAIL);
        assert_eq!(
            persisted_tail.params.expect("params"),
            json!({ "lines": 250 })
        );
        assert_eq!(cleared_logs.method, METHOD_LOG_CLEAR);
        assert_eq!(cleared_logs.params.expect("params"), json!({}));
        assert_eq!(
            cleared_diagnostic_history.method,
            METHOD_LOG_DIAGNOSTIC_HISTORY_CLEAR
        );
        assert_eq!(
            cleared_diagnostic_history.params.expect("params"),
            json!({})
        );
    }

    #[test]
    fn read_artifacts_preserves_filter_and_stable_method() {
        let mut client = AppServerClient::new();

        let request = client
            .read_artifacts(ArtifactReadParams {
                session_id: "sess_1".to_string(),
                turn_id: Some("turn_1".to_string()),
                artifact_ref: Some("artifact-report".to_string()),
                include_content: Some(true),
                cursor: Some("1".to_string()),
                limit: Some(5),
            })
            .expect("request");

        assert_eq!(request.id, RequestId::Integer(1));
        assert_eq!(request.method, METHOD_ARTIFACT_READ);
        assert_eq!(
            request.params.expect("params"),
            json!({
                "sessionId": "sess_1",
                "turnId": "turn_1",
                "artifactRef": "artifact-report",
                "includeContent": true,
                "cursor": "1",
                "limit": 5,
            })
        );
    }

    #[test]
    fn file_system_helpers_use_current_methods() {
        let mut client = AppServerClient::new();

        let listing = client
            .list_directory(FileSystemListDirectoryParams {
                path: "/workspace".to_string(),
            })
            .expect("listing");
        let preview = client
            .read_file_preview(FileSystemReadFilePreviewParams {
                path: "/workspace/README.md".to_string(),
                max_size: Some(1024),
            })
            .expect("preview");
        let create_file = client
            .create_file(FileSystemCreateFileParams {
                path: "/workspace/new.md".to_string(),
            })
            .expect("create file");
        let create_directory = client
            .create_directory(FileSystemCreateDirectoryParams {
                path: "/workspace/new-dir".to_string(),
            })
            .expect("create directory");
        let rename_file = client
            .rename_file(FileSystemRenameFileParams {
                old_path: "/workspace/new.md".to_string(),
                new_path: "/workspace/renamed.md".to_string(),
            })
            .expect("rename file");
        let delete_file = client
            .delete_file(FileSystemDeleteFileParams {
                path: "/workspace/renamed.md".to_string(),
                recursive: Some(false),
            })
            .expect("delete file");

        assert_eq!(listing.id, RequestId::Integer(1));
        assert_eq!(listing.method, METHOD_FILE_SYSTEM_LIST_DIRECTORY);
        assert_eq!(
            listing.params.expect("params"),
            json!({
                "path": "/workspace",
            })
        );
        assert_eq!(preview.id, RequestId::Integer(2));
        assert_eq!(preview.method, METHOD_FILE_SYSTEM_READ_FILE_PREVIEW);
        assert_eq!(
            preview.params.expect("params"),
            json!({
                "path": "/workspace/README.md",
                "maxSize": 1024,
            })
        );
        assert_eq!(create_file.id, RequestId::Integer(3));
        assert_eq!(create_file.method, METHOD_FILE_SYSTEM_CREATE_FILE);
        assert_eq!(
            create_file.params.expect("params"),
            json!({
                "path": "/workspace/new.md",
            })
        );
        assert_eq!(create_directory.id, RequestId::Integer(4));
        assert_eq!(create_directory.method, METHOD_FILE_SYSTEM_CREATE_DIRECTORY);
        assert_eq!(
            create_directory.params.expect("params"),
            json!({
                "path": "/workspace/new-dir",
            })
        );
        assert_eq!(rename_file.id, RequestId::Integer(5));
        assert_eq!(rename_file.method, METHOD_FILE_SYSTEM_RENAME_FILE);
        assert_eq!(
            rename_file.params.expect("params"),
            json!({
                "oldPath": "/workspace/new.md",
                "newPath": "/workspace/renamed.md",
            })
        );
        assert_eq!(delete_file.id, RequestId::Integer(6));
        assert_eq!(delete_file.method, METHOD_FILE_SYSTEM_DELETE_FILE);
        assert_eq!(
            delete_file.params.expect("params"),
            json!({
                "path": "/workspace/renamed.md",
                "recursive": false,
            })
        );
    }

    #[test]
    fn export_evidence_preserves_scope_and_stable_method() {
        let mut client = AppServerClient::new();

        let request = client
            .export_evidence(EvidenceExportParams {
                session_id: "sess_1".to_string(),
                turn_id: Some("turn_1".to_string()),
                include_events: Some(true),
                include_artifacts: Some(false),
                include_evidence_pack: Some(false),
            })
            .expect("request");

        assert_eq!(request.id, RequestId::Integer(1));
        assert_eq!(request.method, METHOD_EVIDENCE_EXPORT);
        assert_eq!(
            request.params.expect("params"),
            json!({
                "sessionId": "sess_1",
                "turnId": "turn_1",
                "includeEvents": true,
                "includeArtifacts": false,
                "includeEvidencePack": false,
            })
        );
    }

    #[test]
    fn export_handoff_bundle_preserves_scope_and_stable_method() {
        let mut client = AppServerClient::new();

        let request = client
            .export_handoff_bundle(AgentSessionHandoffBundleExportParams {
                session_id: "sess_handoff".to_string(),
                locale: Some("zh-CN".to_string()),
            })
            .expect("request");

        assert_eq!(request.id, RequestId::Integer(1));
        assert_eq!(request.method, METHOD_AGENT_SESSION_HANDOFF_BUNDLE_EXPORT);
        assert_eq!(
            request.params.expect("params"),
            json!({
                "sessionId": "sess_handoff",
                "locale": "zh-CN",
            })
        );
    }

    #[test]
    fn runtime_export_residual_methods_preserve_scope_and_stable_method() {
        let mut client = AppServerClient::new();

        let replay = client
            .export_replay_case(AgentSessionReplayCaseExportParams {
                session_id: "sess_replay".to_string(),
                locale: Some("en-US".to_string()),
            })
            .expect("replay request");
        let analysis = client
            .export_analysis_handoff(AgentSessionAnalysisHandoffExportParams {
                session_id: "sess_analysis".to_string(),
                locale: None,
            })
            .expect("analysis request");
        let review = client
            .export_review_decision_template(AgentSessionReviewDecisionTemplateExportParams {
                session_id: "sess_review".to_string(),
                locale: None,
            })
            .expect("review request");
        let save = client
            .save_review_decision(AgentSessionReviewDecisionSaveParams {
                session_id: "sess_review".to_string(),
                decision_status: "accepted".to_string(),
                decision_summary: "ok".to_string(),
                chosen_fix_strategy: "current".to_string(),
                risk_level: "low".to_string(),
                risk_tags: vec!["runtime".to_string()],
                human_reviewer: "reviewer".to_string(),
                followup_actions: Vec::new(),
                regression_requirements: vec!["npm run test:contracts".to_string()],
                notes: String::new(),
                locale: Some("zh-CN".to_string()),
            })
            .expect("save request");

        assert_eq!(replay.id, RequestId::Integer(1));
        assert_eq!(replay.method, METHOD_AGENT_SESSION_REPLAY_CASE_EXPORT);
        assert_eq!(analysis.id, RequestId::Integer(2));
        assert_eq!(
            analysis.method,
            METHOD_AGENT_SESSION_ANALYSIS_HANDOFF_EXPORT
        );
        assert_eq!(review.id, RequestId::Integer(3));
        assert_eq!(
            review.method,
            METHOD_AGENT_SESSION_REVIEW_DECISION_TEMPLATE_EXPORT
        );
        assert_eq!(save.id, RequestId::Integer(4));
        assert_eq!(save.method, METHOD_AGENT_SESSION_REVIEW_DECISION_SAVE);
        assert_eq!(
            save.params.expect("params"),
            json!({
                "sessionId": "sess_review",
                "decisionStatus": "accepted",
                "decisionSummary": "ok",
                "chosenFixStrategy": "current",
                "riskLevel": "low",
                "riskTags": ["runtime"],
                "humanReviewer": "reviewer",
                "followupActions": [],
                "regressionRequirements": ["npm run test:contracts"],
                "notes": "",
                "locale": "zh-CN",
            })
        );
    }

    #[test]
    fn typed_facade_preserves_request_id_sequence() {
        let mut client = AppServerClient::new();

        let first = client
            .read_session(AgentSessionReadParams {
                session_id: "sess_1".to_string(),
                history_limit: None,
                history_offset: None,
                history_before_message_id: None,
            })
            .expect("first");
        let second = client
            .cancel_turn(AgentSessionTurnCancelParams {
                session_id: "sess_1".to_string(),
                turn_id: "turn_1".to_string(),
            })
            .expect("second");
        let third = client
            .run_thread_shell_command(ThreadShellCommandParams {
                thread_id: "thread_1".to_string(),
                command: "printf ready".to_string(),
            })
            .expect("third");

        assert_eq!(first.id, RequestId::Integer(1));
        assert_eq!(second.id, RequestId::Integer(2));
        assert_eq!(third.id, RequestId::Integer(3));
        assert_eq!(second.method, METHOD_TURN_INTERRUPT);
        assert_eq!(third.method, METHOD_THREAD_SHELL_COMMAND);
        assert_eq!(
            third.params.expect("shell params"),
            json!({
                "threadId": "thread_1",
                "command": "printf ready"
            })
        );
    }

    #[test]
    fn event_classifies_allowed_agent_session_raw_side_channel() {
        for event_type in [
            "action.required",
            "approval.required",
            "provider.first_event.received",
            "image_task.created",
            "media.read.chunk",
            "runtime.status",
        ] {
            let notification = JsonRpcNotification::new(
                METHOD_AGENT_SESSION_EVENT,
                Some(
                    serde_json::to_value(AgentSessionEventParams {
                        event: AgentEvent {
                            event_id: format!("evt_{event_type}"),
                            sequence: 1,
                            session_id: "sess_1".to_string(),
                            thread_id: Some("thread_1".to_string()),
                            turn_id: Some("turn_1".to_string()),
                            event_type: event_type.to_string(),
                            timestamp: "2026-06-04T00:00:00Z".to_string(),
                            payload: json!({ "source": "fixture" }),
                        },
                    })
                    .expect("params"),
                ),
            );

            let event = AppServerClient::event(JsonRpcMessage::Notification(notification.clone()))
                .expect("raw side-channel event");

            assert_eq!(event, ClientEvent::AgentSession(notification));
        }
    }

    #[test]
    fn event_rejects_unowned_agent_session_raw_event() {
        let notification = JsonRpcNotification::new(
            METHOD_AGENT_SESSION_EVENT,
            Some(
                serde_json::to_value(AgentSessionEventParams {
                    event: AgentEvent {
                        event_id: "evt_1".to_string(),
                        sequence: 1,
                        session_id: "sess_1".to_string(),
                        thread_id: Some("thread_1".to_string()),
                        turn_id: Some("turn_1".to_string()),
                        event_type: "workflow.step.started".to_string(),
                        timestamp: "2026-06-04T00:00:00Z".to_string(),
                        payload: json!({}),
                    },
                })
                .expect("params"),
            ),
        );

        let error = AppServerClient::event(JsonRpcMessage::Notification(notification))
            .expect_err("unowned raw event must fail closed");

        assert!(matches!(
            error,
            ClientError::InvalidNotification(message)
                if message.contains("workflow.step.started")
        ));
    }

    #[test]
    fn event_decodes_all_direct_v2_lifecycle_notifications() {
        let thread = json!({
            "id": "thread_1",
            "sessionId": "session_1",
            "preview": "",
            "ephemeral": false,
            "historyMode": "legacy",
            "modelProvider": "openai",
            "createdAt": 10,
            "updatedAt": 11,
            "cwd": "/workspace",
            "cliVersion": "1.0.0",
            "source": "appServer",
            "turns": []
        });
        let turn = json!({
            "id": "turn_1",
            "items": [],
            "itemsView": "full",
            "status": "inProgress"
        });
        let item = json!({
            "type": "agentMessage",
            "id": "item_1",
            "text": "hello"
        });
        let notifications = [
            json!({
                "method": "thread/started",
                "params": { "thread": thread }
            }),
            json!({
                "method": "turn/started",
                "params": { "threadId": "thread_1", "turn": turn }
            }),
            json!({
                "method": "turn/completed",
                "params": {
                    "threadId": "thread_1",
                    "turn": {
                        "id": "turn_1",
                        "items": [],
                        "itemsView": "full",
                        "status": "completed"
                    }
                }
            }),
            json!({
                "method": "item/started",
                "params": {
                    "item": item,
                    "threadId": "thread_1",
                    "turnId": "turn_1",
                    "startedAtMs": 12
                }
            }),
            json!({
                "method": "item/completed",
                "params": {
                    "item": {
                        "type": "agentMessage",
                        "id": "item_1",
                        "text": "hello"
                    },
                    "threadId": "thread_1",
                    "turnId": "turn_1",
                    "completedAtMs": 13
                }
            }),
            json!({
                "method": "item/agentMessage/delta",
                "params": {
                    "threadId": "thread_1",
                    "turnId": "turn_1",
                    "itemId": "item_1",
                    "delta": "hello"
                }
            }),
        ];

        for notification in notifications {
            let method = notification["method"]
                .as_str()
                .expect("notification method")
                .to_string();
            let raw =
                JsonRpcNotification::new(method.clone(), Some(notification["params"].clone()));
            let event = AppServerClient::event(JsonRpcMessage::Notification(raw))
                .expect("direct lifecycle event");

            assert!(matches!(
                event,
                ClientEvent::Lifecycle(ref lifecycle) if lifecycle.method() == method
            ));
        }
    }

    #[test]
    fn event_rejects_agent_session_lifecycle_wrapper() {
        let notification = JsonRpcNotification::new(
            METHOD_AGENT_SESSION_EVENT,
            Some(
                serde_json::to_value(AgentSessionEventParams {
                    event: AgentEvent {
                        event_id: "evt_legacy_turn".to_string(),
                        sequence: 1,
                        session_id: "sess_1".to_string(),
                        thread_id: Some("thread_1".to_string()),
                        turn_id: Some("turn_1".to_string()),
                        event_type: "turn.started".to_string(),
                        timestamp: "2026-06-04T00:00:00Z".to_string(),
                        payload: json!({ "status": "running" }),
                    },
                })
                .expect("params"),
            ),
        );

        let error = AppServerClient::event(JsonRpcMessage::Notification(notification))
            .expect_err("wrapper lifecycle must fail closed");

        assert!(matches!(
            error,
            ClientError::InvalidNotification(message)
                if message.contains("turn.started")
        ));
    }

    #[test]
    fn event_rejects_malformed_direct_lifecycle_notification() {
        let notification = JsonRpcNotification::new(
            "turn/started",
            Some(json!({
                "threadId": "thread_1",
                "turn": { "status": "inProgress" }
            })),
        );

        let error = AppServerClient::event(JsonRpcMessage::Notification(notification))
            .expect_err("malformed direct lifecycle must fail closed");

        assert!(matches!(error, ClientError::InvalidNotification(_)));
    }

    #[test]
    fn event_keeps_non_session_messages_available_to_caller() {
        let response =
            JsonRpcResponse::new(RequestId::Integer(1), json!({ "ok": true })).expect("response");
        let error = JsonRpcErrorResponse {
            id: RequestId::Integer(2),
            error: JsonRpcError::new(-32000, "runtime error"),
        };

        assert_eq!(
            AppServerClient::event(JsonRpcMessage::Response(response.clone())).expect("response"),
            ClientEvent::Response(response)
        );
        assert_eq!(
            AppServerClient::event(JsonRpcMessage::Error(error.clone())).expect("error"),
            ClientEvent::Error(error)
        );
    }

    #[test]
    fn reexports_protocol_method_catalog_for_consumers() {
        let catalog = app_server_method_catalog();
        let methods: Vec<&str> = catalog.iter().map(|spec| spec.method).collect();

        assert!(methods.contains(&METHOD_INITIALIZE));
        assert!(methods.contains(&METHOD_ARTIFACT_READ));
        assert!(methods.contains(&METHOD_FILE_SYSTEM_LIST_DIRECTORY));
        assert!(methods.contains(&METHOD_FILE_SYSTEM_READ_FILE_PREVIEW));
        assert!(methods.contains(&METHOD_FILE_SYSTEM_CREATE_FILE));
        assert!(methods.contains(&METHOD_FILE_SYSTEM_CREATE_DIRECTORY));
        assert!(methods.contains(&METHOD_FILE_SYSTEM_RENAME_FILE));
        assert!(methods.contains(&METHOD_FILE_SYSTEM_DELETE_FILE));
        assert!(methods.contains(&METHOD_EVIDENCE_EXPORT));
        assert!(methods.contains(&METHOD_TURN_START));
        assert!(methods.contains(&METHOD_WORKSPACE_LIST));
        assert!(methods.contains(&METHOD_PLUGIN_SHELL_PREPARE));
        assert!(methods.contains(&METHOD_WORKSPACE_READ));
        assert!(methods.contains(&METHOD_WORKSPACE_BY_PATH_READ));
        assert!(methods.contains(&METHOD_WORKSPACE_ENSURE));
        assert!(methods.contains(&METHOD_WORKSPACE_DEFAULT_READ));
        assert!(methods.contains(&METHOD_WORKSPACE_DEFAULT_ENSURE));
        assert!(methods.contains(&METHOD_WORKSPACE_PROJECTS_ROOT_READ));
        assert!(methods.contains(&METHOD_WORKSPACE_PROJECT_PATH_RESOLVE));
        assert!(methods.contains(&METHOD_WORKSPACE_ENSURE_READY));
        assert!(methods.contains(&METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST));
        assert!(methods.contains(&METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_LIST));
        assert!(methods.contains(&METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CONSUME));
        assert!(methods.contains(&METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_DISMISS));
        assert!(methods.contains(&METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CHANGED));
        assert!(methods.contains(&METHOD_BROWSER_SESSION_TARGET_LIST));
        assert!(methods.contains(&METHOD_BROWSER_SESSION_OPEN));
        assert!(methods.contains(&METHOD_BROWSER_SESSION_READ));
        assert!(methods.contains(&METHOD_BROWSER_SESSION_CLOSE));
        assert!(methods.contains(&METHOD_BROWSER_SESSION_EVENT_LIST));
        assert!(methods.contains(&METHOD_BROWSER_SESSION_ACTION_EXECUTE));
        assert!(methods.contains(&METHOD_SKILL_LIST));
        assert!(methods.contains(&METHOD_SKILL_READ));
        assert!(methods.contains(&METHOD_SKILL_PACKAGE_LOCAL_INSPECT));
        assert!(methods.contains(&METHOD_SKILL_PACKAGE_LOCAL_INSTALL));
        assert!(methods.contains(&METHOD_SKILL_PACKAGE_EXPORT));
        assert!(methods.contains(&METHOD_WORKSPACE_SKILL_BINDINGS_LIST));
        assert!(methods.contains(&METHOD_PLUGIN_INSTALLED_LIST));
        assert!(methods.contains(&METHOD_KNOWLEDGE_PACK_LIST));
        assert!(methods.contains(&METHOD_KNOWLEDGE_PACK_READ));
        assert!(methods.contains(&METHOD_AUTOMATION_SCHEDULER_CONFIG_READ));
        assert!(methods.contains(&METHOD_AUTOMATION_SCHEDULER_CONFIG_UPDATE));
        assert!(methods.contains(&METHOD_AUTOMATION_SCHEDULER_STATUS));
        assert!(methods.contains(&METHOD_AUTOMATION_JOB_LIST));
        assert!(methods.contains(&METHOD_AUTOMATION_JOB_READ));
        assert!(methods.contains(&METHOD_AUTOMATION_JOB_CREATE));
        assert!(methods.contains(&METHOD_AUTOMATION_JOB_UPDATE));
        assert!(methods.contains(&METHOD_AUTOMATION_JOB_DELETE));
        assert!(methods.contains(&METHOD_AUTOMATION_JOB_RUN_NOW));
        assert!(methods.contains(&METHOD_AUTOMATION_JOB_HEALTH));
        assert!(methods.contains(&METHOD_AUTOMATION_JOB_RUN_HISTORY));
        assert!(methods.contains(&METHOD_AUTOMATION_SCHEDULE_PREVIEW));
        assert!(methods.contains(&METHOD_AUTOMATION_SCHEDULE_VALIDATE));
        assert!(methods.contains(&METHOD_PROJECT_MEMORY_READ));
        assert!(methods.contains(&METHOD_AGENT_SESSION_EVENT));
        assert!(is_app_server_request_method(METHOD_CAPABILITY_LIST));
        assert!(is_app_server_request_method(METHOD_ARTIFACT_READ));
        assert!(is_app_server_request_method(
            METHOD_FILE_SYSTEM_LIST_DIRECTORY
        ));
        assert!(is_app_server_request_method(
            METHOD_FILE_SYSTEM_READ_FILE_PREVIEW
        ));
        assert!(is_app_server_request_method(METHOD_FILE_SYSTEM_CREATE_FILE));
        assert!(is_app_server_request_method(
            METHOD_FILE_SYSTEM_CREATE_DIRECTORY
        ));
        assert!(is_app_server_request_method(METHOD_FILE_SYSTEM_RENAME_FILE));
        assert!(is_app_server_request_method(METHOD_FILE_SYSTEM_DELETE_FILE));
        assert!(is_app_server_request_method(METHOD_EVIDENCE_EXPORT));
        assert!(is_app_server_request_method(METHOD_WORKSPACE_LIST));
        assert!(is_app_server_request_method(
            METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST
        ));
        assert!(is_app_server_request_method(
            METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_LIST
        ));
        assert!(is_app_server_request_method(
            METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CONSUME
        ));
        assert!(is_app_server_request_method(
            METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_DISMISS
        ));
        assert!(!is_app_server_request_method(
            METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CHANGED
        ));
        assert!(is_app_server_request_method(
            METHOD_BROWSER_SESSION_TARGET_LIST
        ));
        assert!(is_app_server_request_method(METHOD_BROWSER_SESSION_OPEN));
        assert!(is_app_server_request_method(METHOD_BROWSER_SESSION_READ));
        assert!(is_app_server_request_method(METHOD_BROWSER_SESSION_CLOSE));
        assert!(is_app_server_request_method(
            METHOD_BROWSER_SESSION_EVENT_LIST
        ));
        assert!(is_app_server_request_method(
            METHOD_BROWSER_SESSION_ACTION_EXECUTE
        ));
        assert!(is_app_server_request_method(METHOD_SKILL_LIST));
        assert!(is_app_server_request_method(METHOD_PLUGIN_INSTALLED_LIST));
        assert!(is_app_server_request_method(METHOD_KNOWLEDGE_PACK_LIST));
        assert!(is_app_server_request_method(METHOD_KNOWLEDGE_PACK_READ));
        assert!(is_app_server_request_method(
            METHOD_AUTOMATION_SCHEDULER_CONFIG_READ
        ));
        assert!(is_app_server_request_method(
            METHOD_AUTOMATION_SCHEDULER_CONFIG_UPDATE
        ));
        assert!(is_app_server_request_method(
            METHOD_AUTOMATION_SCHEDULER_STATUS
        ));
        assert!(is_app_server_request_method(METHOD_AUTOMATION_JOB_LIST));
        assert!(is_app_server_request_method(METHOD_AUTOMATION_JOB_CREATE));
        assert!(is_app_server_request_method(
            METHOD_AUTOMATION_SCHEDULE_VALIDATE
        ));
        assert!(is_app_server_request_method(METHOD_PROJECT_MEMORY_READ));
        assert!(is_app_server_request_method(METHOD_TURN_INTERRUPT));
        assert!(!is_app_server_request_method(METHOD_INITIALIZED));
        assert!(is_app_server_notification_method(METHOD_INITIALIZED));
        assert!(is_app_server_notification_method(
            METHOD_AGENT_SESSION_EVENT
        ));
        assert!(is_app_server_notification_method(
            METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CHANGED
        ));
    }
}
