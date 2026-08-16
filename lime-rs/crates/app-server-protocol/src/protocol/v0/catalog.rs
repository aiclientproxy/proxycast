use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::protocol::v2::METHOD_THREAD_RESUME;

#[cfg(test)]
use crate::{JsonRpcRequest, RequestId};

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum AppServerMethodKind {
    Request,
    Notification,
    ServerRequest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AppServerMethodSpec {
    pub method: &'static str,
    pub kind: AppServerMethodKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum AppServerRequestSerializationScope {
    Thread,
    McpOauth,
    McpResourceSubscription,
    BrowserSession,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AppServerRequestSerializationScopeSpec {
    pub method: &'static str,
    pub scope: AppServerRequestSerializationScope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum AppServerRequestAccess {
    Exclusive,
    SharedRead,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AppServerRequestAccessSpec {
    pub method: &'static str,
    pub access: AppServerRequestAccess,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub enum AppServerRequestMethod {
    #[serde(rename = "initialize")]
    Initialize,
    #[serde(rename = "capability/list")]
    CapabilityList,
    #[serde(rename = "artifact/read")]
    ArtifactRead,
    #[serde(rename = "projectGit/status")]
    ProjectGitStatus,
    #[serde(rename = "projectGit/diff")]
    ProjectGitDiff,
    #[serde(rename = "projectGit/commits/list")]
    ProjectGitCommitsList,
    #[serde(rename = "projectGit/branch/checkout")]
    ProjectGitBranchCheckout,
    #[serde(rename = "projectGit/branch/create")]
    ProjectGitBranchCreate,
    #[serde(rename = "projectGit/worktree/create")]
    ProjectGitWorktreeCreate,
    #[serde(rename = "agentSession/handoffBundle/export")]
    AgentSessionHandoffBundleExport,
    #[serde(rename = "agentSession/replayCase/export")]
    AgentSessionReplayCaseExport,
    #[serde(rename = "agentSession/analysisHandoff/export")]
    AgentSessionAnalysisHandoffExport,
    #[serde(rename = "agentSession/reviewDecisionTemplate/export")]
    AgentSessionReviewDecisionTemplateExport,
    #[serde(rename = "agentSession/reviewDecision/save")]
    AgentSessionReviewDecisionSave,
    #[serde(rename = "agentSession/fileCheckpoint/list")]
    AgentSessionFileCheckpointList,
    #[serde(rename = "agentSession/fileCheckpoint/get")]
    AgentSessionFileCheckpointGet,
    #[serde(rename = "agentSession/fileCheckpoint/diff")]
    AgentSessionFileCheckpointDiff,
    #[serde(rename = "agentSession/fileCheckpoint/restore")]
    AgentSessionFileCheckpointRestore,
    #[serde(rename = "agentSession/toolInventory/read")]
    AgentSessionToolInventoryRead,
    #[serde(rename = "sessionFile/getOrCreate")]
    SessionFileGetOrCreate,
    #[serde(rename = "sessionFile/updateMeta")]
    SessionFileUpdateMeta,
    #[serde(rename = "sessionFile/save")]
    SessionFileSave,
    #[serde(rename = "sessionFile/read")]
    SessionFileRead,
    #[serde(rename = "sessionFile/resolvePath")]
    SessionFileResolvePath,
    #[serde(rename = "sessionFile/delete")]
    SessionFileDelete,
    #[serde(rename = "sessionFile/list")]
    SessionFileList,
    #[serde(rename = "workspace/list")]
    WorkspaceList,
    #[serde(rename = "workspace/read")]
    WorkspaceRead,
    #[serde(rename = "workspace/update")]
    WorkspaceUpdate,
    #[serde(rename = "workspace/delete")]
    WorkspaceDelete,
    #[serde(rename = "workspace/ensure")]
    WorkspaceEnsure,
    #[serde(rename = "workspace/byPath/read")]
    WorkspaceByPathRead,
    #[serde(rename = "workspace/default/read")]
    WorkspaceDefaultRead,
    #[serde(rename = "workspace/default/ensure")]
    WorkspaceDefaultEnsure,
    #[serde(rename = "workspace/projectsRoot/read")]
    WorkspaceProjectsRootRead,
    #[serde(rename = "workspace/projectPath/resolve")]
    WorkspaceProjectPathResolve,
    #[serde(rename = "workspace/ensureReady")]
    WorkspaceEnsureReady,
    #[serde(rename = "skill/read")]
    SkillRead,
    #[serde(rename = "skillManagement/list")]
    SkillManagementList,
    #[serde(rename = "skillManagement/install")]
    SkillManagementInstall,
    #[serde(rename = "skillManagement/uninstall")]
    SkillManagementUninstall,
    #[serde(rename = "skillRepository/list")]
    SkillRepositoryList,
    #[serde(rename = "skillRepository/save")]
    SkillRepositorySave,
    #[serde(rename = "skillRepository/delete")]
    SkillRepositoryDelete,
    #[serde(rename = "skillCache/refresh")]
    SkillCacheRefresh,
    #[serde(rename = "skillInstalledDirectories/list")]
    SkillInstalledDirectoriesList,
    #[serde(rename = "skillLocal/inspect")]
    SkillLocalInspect,
    #[serde(rename = "skillLocal/detail/inspect")]
    SkillLocalDetailInspect,
    #[serde(rename = "skillLocal/scaffold/create")]
    SkillLocalScaffoldCreate,
    #[serde(rename = "skillLocal/import")]
    SkillLocalImport,
    #[serde(rename = "skillLocal/rename")]
    SkillLocalRename,
    #[serde(rename = "skillRemote/inspect")]
    SkillRemoteInspect,
    #[serde(rename = "skillPackage/local/inspect")]
    SkillPackageLocalInspect,
    #[serde(rename = "skillPackage/local/install")]
    SkillPackageLocalInstall,
    #[serde(rename = "skillPackage/local/replace")]
    SkillPackageLocalReplace,
    #[serde(rename = "skillPackage/export")]
    SkillPackageExport,
    #[serde(rename = "skillMarketplace/install")]
    SkillMarketplaceInstall,
    #[serde(rename = "skillPackage/download/install")]
    SkillPackageDownloadInstall,
    #[serde(rename = "gatewayChannel/start")]
    GatewayChannelStart,
    #[serde(rename = "gatewayChannel/stop")]
    GatewayChannelStop,
    #[serde(rename = "gatewayChannel/status")]
    GatewayChannelStatus,
    #[serde(rename = "telegramChannel/probe")]
    TelegramChannelProbe,
    #[serde(rename = "feishuChannel/probe")]
    FeishuChannelProbe,
    #[serde(rename = "discordChannel/probe")]
    DiscordChannelProbe,
    #[serde(rename = "wechatChannel/probe")]
    WechatChannelProbe,
    #[serde(rename = "wechatChannel/login/start")]
    WechatChannelLoginStart,
    #[serde(rename = "wechatChannel/login/wait")]
    WechatChannelLoginWait,
    #[serde(rename = "wechatChannel/accounts/list")]
    WechatChannelAccountList,
    #[serde(rename = "wechatChannel/account/remove")]
    WechatChannelAccountRemove,
    #[serde(rename = "wechatChannel/runtimeModel/set")]
    WechatChannelRuntimeModelSet,
    #[serde(rename = "gatewayTunnel/probe")]
    GatewayTunnelProbe,
    #[serde(rename = "gatewayTunnel/cloudflared/detect")]
    GatewayTunnelCloudflaredDetect,
    #[serde(rename = "gatewayTunnel/cloudflared/install")]
    GatewayTunnelCloudflaredInstall,
    #[serde(rename = "gatewayTunnel/create")]
    GatewayTunnelCreate,
    #[serde(rename = "gatewayTunnel/start")]
    GatewayTunnelStart,
    #[serde(rename = "gatewayTunnel/stop")]
    GatewayTunnelStop,
    #[serde(rename = "gatewayTunnel/restart")]
    GatewayTunnelRestart,
    #[serde(rename = "gatewayTunnel/status")]
    GatewayTunnelStatus,
    #[serde(rename = "gatewayTunnel/syncWebhookUrl")]
    GatewayTunnelSyncWebhookUrl,
    #[serde(rename = "mediaTaskArtifact/image/create")]
    MediaTaskArtifactImageCreate,
    #[serde(rename = "mediaTaskArtifact/audio/create")]
    MediaTaskArtifactAudioCreate,
    #[serde(rename = "mediaTaskArtifact/transcription/create")]
    MediaTaskArtifactTranscriptionCreate,
    #[serde(rename = "mediaTaskArtifact/video/create")]
    MediaTaskArtifactVideoCreate,
    #[serde(rename = "mediaTaskArtifact/image/complete")]
    MediaTaskArtifactImageComplete,
    #[serde(rename = "mediaTaskArtifact/audio/complete")]
    MediaTaskArtifactAudioComplete,
    #[serde(rename = "mediaTaskArtifact/get")]
    MediaTaskArtifactGet,
    #[serde(rename = "mediaTaskArtifact/list")]
    MediaTaskArtifactList,
    #[serde(rename = "mediaTaskArtifact/cancel")]
    MediaTaskArtifactCancel,
    #[serde(rename = "galleryMaterial/get")]
    GalleryMaterialGet,
    #[serde(rename = "galleryMaterialMetadata/create")]
    GalleryMaterialMetadataCreate,
    #[serde(rename = "galleryMaterialMetadata/get")]
    GalleryMaterialMetadataGet,
    #[serde(rename = "galleryMaterialMetadata/update")]
    GalleryMaterialMetadataUpdate,
    #[serde(rename = "galleryMaterialMetadata/delete")]
    GalleryMaterialMetadataDelete,
    #[serde(rename = "galleryMaterial/listByImageCategory")]
    GalleryMaterialListByImageCategory,
    #[serde(rename = "galleryMaterial/listByLayoutCategory")]
    GalleryMaterialListByLayoutCategory,
    #[serde(rename = "galleryMaterial/listByMood")]
    GalleryMaterialListByMood,
    #[serde(rename = "projectMaterial/list")]
    ProjectMaterialList,
    #[serde(rename = "projectMaterial/get")]
    ProjectMaterialGet,
    #[serde(rename = "projectMaterial/count")]
    ProjectMaterialCount,
    #[serde(rename = "projectMaterial/upload")]
    ProjectMaterialUpload,
    #[serde(rename = "projectMaterial/importFromUrl")]
    ProjectMaterialImportFromUrl,
    #[serde(rename = "projectMaterial/update")]
    ProjectMaterialUpdate,
    #[serde(rename = "projectMaterial/delete")]
    ProjectMaterialDelete,
    #[serde(rename = "projectMaterial/content")]
    ProjectMaterialContent,
    #[serde(rename = "voiceAsrCredential/list")]
    VoiceAsrCredentialList,
    #[serde(rename = "voiceAsrCredential/create")]
    VoiceAsrCredentialCreate,
    #[serde(rename = "voiceAsrCredential/update")]
    VoiceAsrCredentialUpdate,
    #[serde(rename = "voiceAsrCredential/delete")]
    VoiceAsrCredentialDelete,
    #[serde(rename = "voiceAsrCredential/default/set")]
    VoiceAsrCredentialDefaultSet,
    #[serde(rename = "voiceAsrCredential/test")]
    VoiceAsrCredentialTest,
    #[serde(rename = "voiceInstruction/list")]
    VoiceInstructionList,
    #[serde(rename = "voiceInstruction/save")]
    VoiceInstructionSave,
    #[serde(rename = "voiceInstruction/delete")]
    VoiceInstructionDelete,
    #[serde(rename = "voiceModel/default/set")]
    VoiceModelDefaultSet,
    #[serde(rename = "voiceModel/testTranscribeFile")]
    VoiceModelTestTranscribeFile,
    #[serde(rename = "voiceTranscription/transcribeAudio")]
    VoiceTranscriptionTranscribeAudio,
    #[serde(rename = "voiceTranscription/polishText")]
    VoiceTranscriptionPolishText,
    #[serde(rename = "workspaceSkillBindings/list")]
    WorkspaceSkillBindingsList,
    #[serde(rename = "workspaceRegisteredSkills/list")]
    WorkspaceRegisteredSkillsList,
    #[serde(rename = "workspaceRightSurface/request")]
    WorkspaceRightSurfaceRequest,
    #[serde(rename = "workspaceRightSurface/pending/list")]
    WorkspaceRightSurfacePendingList,
    #[serde(rename = "workspaceRightSurface/pending/consume")]
    WorkspaceRightSurfacePendingConsume,
    #[serde(rename = "workspaceRightSurface/pending/dismiss")]
    WorkspaceRightSurfacePendingDismiss,
    #[serde(rename = "browserSession/target/list")]
    BrowserSessionTargetList,
    #[serde(rename = "browserSession/open")]
    BrowserSessionOpen,
    #[serde(rename = "browserSession/read")]
    BrowserSessionRead,
    #[serde(rename = "browserSession/close")]
    BrowserSessionClose,
    #[serde(rename = "browserSession/event/list")]
    BrowserSessionEventList,
    #[serde(rename = "browserSession/action/execute")]
    BrowserSessionActionExecute,
    #[serde(rename = "soulStylePack/install")]
    SoulStylePackInstall,
    #[serde(rename = "soulStylePack/list")]
    SoulStylePackList,
    #[serde(rename = "soulStylePack/status/set")]
    SoulStylePackStatusSet,
    #[serde(rename = "soulStylePack/uninstall")]
    SoulStylePackUninstall,
    #[serde(rename = "knowledgePack/list")]
    KnowledgePackList,
    #[serde(rename = "knowledgePack/read")]
    KnowledgePackRead,
    #[serde(rename = "knowledgePack/source/import")]
    KnowledgeSourceImport,
    #[serde(rename = "knowledgePack/compile")]
    KnowledgePackCompile,
    #[serde(rename = "knowledgePack/default/set")]
    KnowledgePackDefaultSet,
    #[serde(rename = "knowledgePack/status/update")]
    KnowledgePackStatusUpdate,
    #[serde(rename = "knowledgeContext/resolve")]
    KnowledgeContextResolve,
    #[serde(rename = "knowledgeContextRun/validate")]
    KnowledgeContextRunValidate,
    #[serde(rename = "automationScheduler/config/read")]
    AutomationSchedulerConfigRead,
    #[serde(rename = "automationScheduler/config/update")]
    AutomationSchedulerConfigUpdate,
    #[serde(rename = "automationScheduler/status")]
    AutomationSchedulerStatus,
    #[serde(rename = "automationJob/list")]
    AutomationJobList,
    #[serde(rename = "automationJob/read")]
    AutomationJobRead,
    #[serde(rename = "automationJob/create")]
    AutomationJobCreate,
    #[serde(rename = "automationJob/update")]
    AutomationJobUpdate,
    #[serde(rename = "automationJob/delete")]
    AutomationJobDelete,
    #[serde(rename = "automationJob/runNow")]
    AutomationJobRunNow,
    #[serde(rename = "automationJob/health")]
    AutomationJobHealth,
    #[serde(rename = "automationJob/runHistory")]
    AutomationJobRunHistory,
    #[serde(rename = "automationSchedule/preview")]
    AutomationSchedulePreview,
    #[serde(rename = "automationSchedule/validate")]
    AutomationScheduleValidate,
    #[serde(rename = "scheduledTask/list")]
    ScheduledTaskList,
    #[serde(rename = "scheduledTask/read")]
    ScheduledTaskRead,
    #[serde(rename = "scheduledTask/create")]
    ScheduledTaskCreate,
    #[serde(rename = "scheduledTask/update")]
    ScheduledTaskUpdate,
    #[serde(rename = "scheduledTask/delete")]
    ScheduledTaskDelete,
    #[serde(rename = "scheduledTask/enabled/set")]
    ScheduledTaskEnabledSet,
    #[serde(rename = "scheduledTask/run/start")]
    ScheduledTaskRunStart,
    #[serde(rename = "scheduledTask/run/list")]
    ScheduledTaskRunList,
    #[serde(rename = "scheduledTask/schedule/preview")]
    ScheduledTaskSchedulePreview,
    #[serde(rename = "mcpServer/list")]
    McpServerList,
    #[serde(rename = "mcpServerStatus/list")]
    McpServerStatusList,
    #[serde(rename = "mcpServer/create")]
    McpServerCreate,
    #[serde(rename = "mcpServer/update")]
    McpServerUpdate,
    #[serde(rename = "mcpServer/delete")]
    McpServerDelete,
    #[serde(rename = "mcpServer/enabled/set")]
    McpServerEnabledSet,
    #[serde(rename = "mcpServer/importFromApp")]
    McpServerImportFromApp,
    #[serde(rename = "mcpServer/syncAllToLive")]
    McpServerSyncAllToLive,
    #[serde(rename = "mcpServer/oauth/login")]
    McpServerOauthLogin,
    #[serde(rename = "mcpServer/start")]
    McpServerStart,
    #[serde(rename = "mcpServer/stop")]
    McpServerStop,
    #[serde(rename = "mcpTool/list")]
    McpToolList,
    #[serde(rename = "mcpTool/listForContext")]
    McpToolListForContext,
    #[serde(rename = "mcpTool/search")]
    McpToolSearch,
    #[serde(rename = "mcpPrompt/list")]
    McpPromptList,
    #[serde(rename = "mcpPrompt/get")]
    McpPromptGet,
    #[serde(rename = "mcpResource/list")]
    McpResourceList,
    #[serde(rename = "mcpResource/subscribe")]
    McpResourceSubscribe,
    #[serde(rename = "mcpResource/unsubscribe")]
    McpResourceUnsubscribe,
    #[serde(rename = "projectMemory/read")]
    ProjectMemoryRead,
    #[serde(rename = "memoryStore/list")]
    MemoryStoreList,
    #[serde(rename = "memoryStore/read")]
    MemoryStoreRead,
    #[serde(rename = "memoryStore/search")]
    MemoryStoreSearch,
    #[serde(rename = "memoryStore/addNote")]
    MemoryStoreAddNote,
    #[serde(rename = "memoryStore/consolidate")]
    MemoryStoreConsolidate,
    #[serde(rename = "memoryStore/review/list")]
    MemoryStoreReviewList,
    #[serde(rename = "memoryStore/review/resolve")]
    MemoryStoreReviewResolve,
    #[serde(rename = "memoryStore/health")]
    MemoryStoreHealth,
    #[serde(rename = "memoryStore/index/rebuild")]
    MemoryStoreIndexRebuild,
    #[serde(rename = "log/list")]
    LogList,
    #[serde(rename = "log/persistedTail")]
    LogPersistedTail,
    #[serde(rename = "log/clear")]
    LogClear,
    #[serde(rename = "log/diagnosticHistory/clear")]
    LogDiagnosticHistoryClear,
    #[serde(rename = "diagnostics/logStorage/read")]
    DiagnosticsLogStorageRead,
    #[serde(rename = "diagnostics/supportBundle/export")]
    DiagnosticsSupportBundleExport,
    #[serde(rename = "diagnostics/server/read")]
    DiagnosticsServerRead,
    #[serde(rename = "diagnostics/windowsStartup/read")]
    DiagnosticsWindowsStartupRead,
    #[serde(rename = "diagnostics/trace/list")]
    DiagnosticsTraceList,
    #[serde(rename = "diagnostics/trace/read")]
    DiagnosticsTraceRead,
    #[serde(rename = "diagnostics/trace/export")]
    DiagnosticsTraceExport,
    #[serde(rename = "usageStats/read")]
    UsageStatsRead,
    #[serde(rename = "usageStats/modelRanking/list")]
    UsageStatsModelRankingList,
    #[serde(rename = "usageStats/dailyTrends/list")]
    UsageStatsDailyTrendsList,
    #[serde(rename = "modelPreferences/list")]
    ModelPreferencesList,
    #[serde(rename = "modelSyncState/read")]
    ModelSyncStateRead,
    #[serde(rename = "modelProvider/list")]
    ModelProviderList,
    #[serde(rename = "modelProvider/catalog/list")]
    ModelProviderCatalogList,
    #[serde(rename = "modelProvider/read")]
    ModelProviderRead,
    #[serde(rename = "modelProvider/create")]
    ModelProviderCreate,
    #[serde(rename = "modelProvider/update")]
    ModelProviderUpdate,
    #[serde(rename = "modelProvider/delete")]
    ModelProviderDelete,
    #[serde(rename = "modelProvider/sortOrders/update")]
    ModelProviderSortOrdersUpdate,
    #[serde(rename = "modelProviderConfig/export")]
    ModelProviderConfigExport,
    #[serde(rename = "modelProviderConfig/import")]
    ModelProviderConfigImport,
    #[serde(rename = "modelProvider/testConnection")]
    ModelProviderTestConnection,
    #[serde(rename = "modelProvider/testChat")]
    ModelProviderTestChat,
    #[serde(rename = "modelProvider/fetchModels")]
    ModelProviderFetchModels,
    #[serde(rename = "modelProviderKey/create")]
    ModelProviderKeyCreate,
    #[serde(rename = "modelProviderKey/update")]
    ModelProviderKeyUpdate,
    #[serde(rename = "modelProviderKey/delete")]
    ModelProviderKeyDelete,
    #[serde(rename = "modelProviderUiState/read")]
    ModelProviderUiStateRead,
    #[serde(rename = "modelProviderUiState/write")]
    ModelProviderUiStateWrite,
    #[serde(rename = "modelProviderAlias/read")]
    ModelProviderAliasRead,
    #[serde(rename = "modelProviderAlias/list")]
    ModelProviderAliasList,
    #[serde(rename = "connectDeepLink/resolve")]
    ConnectDeepLinkResolve,
    #[serde(rename = "connectOpenDeepLink/resolve")]
    ConnectOpenDeepLinkResolve,
    #[serde(rename = "connectRelayApiKey/save")]
    ConnectRelayApiKeySave,
    #[serde(rename = "connectCallback/send")]
    ConnectCallbackSend,
    #[serde(rename = "conversationImport/source/scan")]
    ConversationImportSourceScan,
    #[serde(rename = "conversationImport/thread/preview")]
    ConversationImportThreadPreview,
    #[serde(rename = "conversationImport/thread/commit")]
    ConversationImportThreadCommit,
    #[serde(rename = "conversationImport/job/read")]
    ConversationImportJobRead,
    #[serde(rename = "agentSession/action/respond")]
    AgentSessionActionRespond,
    #[serde(rename = "workflow/read")]
    WorkflowRead,
    #[serde(rename = "workflow/cancel")]
    WorkflowCancel,
    #[serde(rename = "workflow/retry")]
    WorkflowRetry,
    #[serde(rename = "workflow/respond")]
    WorkflowRespond,
}

impl AppServerRequestMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Initialize => METHOD_INITIALIZE,
            Self::CapabilityList => METHOD_CAPABILITY_LIST,
            Self::ArtifactRead => METHOD_ARTIFACT_READ,
            Self::ProjectGitStatus => METHOD_PROJECT_GIT_STATUS,
            Self::ProjectGitDiff => METHOD_PROJECT_GIT_DIFF,
            Self::ProjectGitCommitsList => METHOD_PROJECT_GIT_COMMITS_LIST,
            Self::ProjectGitBranchCheckout => METHOD_PROJECT_GIT_BRANCH_CHECKOUT,
            Self::ProjectGitBranchCreate => METHOD_PROJECT_GIT_BRANCH_CREATE,
            Self::ProjectGitWorktreeCreate => METHOD_PROJECT_GIT_WORKTREE_CREATE,
            Self::AgentSessionHandoffBundleExport => METHOD_AGENT_SESSION_HANDOFF_BUNDLE_EXPORT,
            Self::AgentSessionReplayCaseExport => METHOD_AGENT_SESSION_REPLAY_CASE_EXPORT,
            Self::AgentSessionAnalysisHandoffExport => METHOD_AGENT_SESSION_ANALYSIS_HANDOFF_EXPORT,
            Self::AgentSessionReviewDecisionTemplateExport => {
                METHOD_AGENT_SESSION_REVIEW_DECISION_TEMPLATE_EXPORT
            }
            Self::AgentSessionReviewDecisionSave => METHOD_AGENT_SESSION_REVIEW_DECISION_SAVE,
            Self::AgentSessionFileCheckpointList => METHOD_AGENT_SESSION_FILE_CHECKPOINT_LIST,
            Self::AgentSessionFileCheckpointGet => METHOD_AGENT_SESSION_FILE_CHECKPOINT_GET,
            Self::AgentSessionFileCheckpointDiff => METHOD_AGENT_SESSION_FILE_CHECKPOINT_DIFF,
            Self::AgentSessionFileCheckpointRestore => METHOD_AGENT_SESSION_FILE_CHECKPOINT_RESTORE,
            Self::AgentSessionToolInventoryRead => METHOD_AGENT_SESSION_TOOL_INVENTORY_READ,
            Self::SessionFileGetOrCreate => METHOD_SESSION_FILE_GET_OR_CREATE,
            Self::SessionFileUpdateMeta => METHOD_SESSION_FILE_UPDATE_META,
            Self::SessionFileSave => METHOD_SESSION_FILE_SAVE,
            Self::SessionFileRead => METHOD_SESSION_FILE_READ,
            Self::SessionFileResolvePath => METHOD_SESSION_FILE_RESOLVE_PATH,
            Self::SessionFileDelete => METHOD_SESSION_FILE_DELETE,
            Self::SessionFileList => METHOD_SESSION_FILE_LIST,
            Self::WorkspaceList => METHOD_WORKSPACE_LIST,
            Self::WorkspaceRead => METHOD_WORKSPACE_READ,
            Self::WorkspaceUpdate => METHOD_WORKSPACE_UPDATE,
            Self::WorkspaceDelete => METHOD_WORKSPACE_DELETE,
            Self::WorkspaceEnsure => METHOD_WORKSPACE_ENSURE,
            Self::WorkspaceByPathRead => METHOD_WORKSPACE_BY_PATH_READ,
            Self::WorkspaceDefaultRead => METHOD_WORKSPACE_DEFAULT_READ,
            Self::WorkspaceDefaultEnsure => METHOD_WORKSPACE_DEFAULT_ENSURE,
            Self::WorkspaceProjectsRootRead => METHOD_WORKSPACE_PROJECTS_ROOT_READ,
            Self::WorkspaceProjectPathResolve => METHOD_WORKSPACE_PROJECT_PATH_RESOLVE,
            Self::WorkspaceEnsureReady => METHOD_WORKSPACE_ENSURE_READY,
            Self::SkillRead => METHOD_SKILL_READ,
            Self::SkillManagementList => METHOD_SKILL_MANAGEMENT_LIST,
            Self::SkillManagementInstall => METHOD_SKILL_MANAGEMENT_INSTALL,
            Self::SkillManagementUninstall => METHOD_SKILL_MANAGEMENT_UNINSTALL,
            Self::SkillRepositoryList => METHOD_SKILL_REPOSITORY_LIST,
            Self::SkillRepositorySave => METHOD_SKILL_REPOSITORY_SAVE,
            Self::SkillRepositoryDelete => METHOD_SKILL_REPOSITORY_DELETE,
            Self::SkillCacheRefresh => METHOD_SKILL_CACHE_REFRESH,
            Self::SkillInstalledDirectoriesList => METHOD_SKILL_INSTALLED_DIRECTORIES_LIST,
            Self::SkillLocalInspect => METHOD_SKILL_LOCAL_INSPECT,
            Self::SkillLocalDetailInspect => METHOD_SKILL_LOCAL_DETAIL_INSPECT,
            Self::SkillLocalScaffoldCreate => METHOD_SKILL_LOCAL_SCAFFOLD_CREATE,
            Self::SkillLocalImport => METHOD_SKILL_LOCAL_IMPORT,
            Self::SkillLocalRename => METHOD_SKILL_LOCAL_RENAME,
            Self::SkillRemoteInspect => METHOD_SKILL_REMOTE_INSPECT,
            Self::SkillPackageLocalInspect => METHOD_SKILL_PACKAGE_LOCAL_INSPECT,
            Self::SkillPackageLocalInstall => METHOD_SKILL_PACKAGE_LOCAL_INSTALL,
            Self::SkillPackageLocalReplace => METHOD_SKILL_PACKAGE_LOCAL_REPLACE,
            Self::SkillPackageExport => METHOD_SKILL_PACKAGE_EXPORT,
            Self::SkillMarketplaceInstall => METHOD_SKILL_MARKETPLACE_INSTALL,
            Self::SkillPackageDownloadInstall => METHOD_SKILL_PACKAGE_DOWNLOAD_INSTALL,
            Self::GatewayChannelStart => METHOD_GATEWAY_CHANNEL_START,
            Self::GatewayChannelStop => METHOD_GATEWAY_CHANNEL_STOP,
            Self::GatewayChannelStatus => METHOD_GATEWAY_CHANNEL_STATUS,
            Self::TelegramChannelProbe => METHOD_TELEGRAM_CHANNEL_PROBE,
            Self::FeishuChannelProbe => METHOD_FEISHU_CHANNEL_PROBE,
            Self::DiscordChannelProbe => METHOD_DISCORD_CHANNEL_PROBE,
            Self::WechatChannelProbe => METHOD_WECHAT_CHANNEL_PROBE,
            Self::WechatChannelLoginStart => METHOD_WECHAT_CHANNEL_LOGIN_START,
            Self::WechatChannelLoginWait => METHOD_WECHAT_CHANNEL_LOGIN_WAIT,
            Self::WechatChannelAccountList => METHOD_WECHAT_CHANNEL_ACCOUNT_LIST,
            Self::WechatChannelAccountRemove => METHOD_WECHAT_CHANNEL_ACCOUNT_REMOVE,
            Self::WechatChannelRuntimeModelSet => METHOD_WECHAT_CHANNEL_RUNTIME_MODEL_SET,
            Self::GatewayTunnelProbe => METHOD_GATEWAY_TUNNEL_PROBE,
            Self::GatewayTunnelCloudflaredDetect => METHOD_GATEWAY_TUNNEL_CLOUDFLARED_DETECT,
            Self::GatewayTunnelCloudflaredInstall => METHOD_GATEWAY_TUNNEL_CLOUDFLARED_INSTALL,
            Self::GatewayTunnelCreate => METHOD_GATEWAY_TUNNEL_CREATE,
            Self::GatewayTunnelStart => METHOD_GATEWAY_TUNNEL_START,
            Self::GatewayTunnelStop => METHOD_GATEWAY_TUNNEL_STOP,
            Self::GatewayTunnelRestart => METHOD_GATEWAY_TUNNEL_RESTART,
            Self::GatewayTunnelStatus => METHOD_GATEWAY_TUNNEL_STATUS,
            Self::GatewayTunnelSyncWebhookUrl => METHOD_GATEWAY_TUNNEL_SYNC_WEBHOOK_URL,
            Self::MediaTaskArtifactImageCreate => METHOD_MEDIA_TASK_ARTIFACT_IMAGE_CREATE,
            Self::MediaTaskArtifactAudioCreate => METHOD_MEDIA_TASK_ARTIFACT_AUDIO_CREATE,
            Self::MediaTaskArtifactTranscriptionCreate => {
                METHOD_MEDIA_TASK_ARTIFACT_TRANSCRIPTION_CREATE
            }
            Self::MediaTaskArtifactVideoCreate => METHOD_MEDIA_TASK_ARTIFACT_VIDEO_CREATE,
            Self::MediaTaskArtifactImageComplete => METHOD_MEDIA_TASK_ARTIFACT_IMAGE_COMPLETE,
            Self::MediaTaskArtifactAudioComplete => METHOD_MEDIA_TASK_ARTIFACT_AUDIO_COMPLETE,
            Self::MediaTaskArtifactGet => METHOD_MEDIA_TASK_ARTIFACT_GET,
            Self::MediaTaskArtifactList => METHOD_MEDIA_TASK_ARTIFACT_LIST,
            Self::MediaTaskArtifactCancel => METHOD_MEDIA_TASK_ARTIFACT_CANCEL,
            Self::GalleryMaterialGet => METHOD_GALLERY_MATERIAL_GET,
            Self::GalleryMaterialMetadataCreate => METHOD_GALLERY_MATERIAL_METADATA_CREATE,
            Self::GalleryMaterialMetadataGet => METHOD_GALLERY_MATERIAL_METADATA_GET,
            Self::GalleryMaterialMetadataUpdate => METHOD_GALLERY_MATERIAL_METADATA_UPDATE,
            Self::GalleryMaterialMetadataDelete => METHOD_GALLERY_MATERIAL_METADATA_DELETE,
            Self::GalleryMaterialListByImageCategory => {
                METHOD_GALLERY_MATERIAL_LIST_BY_IMAGE_CATEGORY
            }
            Self::GalleryMaterialListByLayoutCategory => {
                METHOD_GALLERY_MATERIAL_LIST_BY_LAYOUT_CATEGORY
            }
            Self::GalleryMaterialListByMood => METHOD_GALLERY_MATERIAL_LIST_BY_MOOD,
            Self::ProjectMaterialList => METHOD_PROJECT_MATERIAL_LIST,
            Self::ProjectMaterialGet => METHOD_PROJECT_MATERIAL_GET,
            Self::ProjectMaterialCount => METHOD_PROJECT_MATERIAL_COUNT,
            Self::ProjectMaterialUpload => METHOD_PROJECT_MATERIAL_UPLOAD,
            Self::ProjectMaterialImportFromUrl => METHOD_PROJECT_MATERIAL_IMPORT_FROM_URL,
            Self::ProjectMaterialUpdate => METHOD_PROJECT_MATERIAL_UPDATE,
            Self::ProjectMaterialDelete => METHOD_PROJECT_MATERIAL_DELETE,
            Self::ProjectMaterialContent => METHOD_PROJECT_MATERIAL_CONTENT,
            Self::VoiceAsrCredentialList => METHOD_VOICE_ASR_CREDENTIAL_LIST,
            Self::VoiceAsrCredentialCreate => METHOD_VOICE_ASR_CREDENTIAL_CREATE,
            Self::VoiceAsrCredentialUpdate => METHOD_VOICE_ASR_CREDENTIAL_UPDATE,
            Self::VoiceAsrCredentialDelete => METHOD_VOICE_ASR_CREDENTIAL_DELETE,
            Self::VoiceAsrCredentialDefaultSet => METHOD_VOICE_ASR_CREDENTIAL_DEFAULT_SET,
            Self::VoiceAsrCredentialTest => METHOD_VOICE_ASR_CREDENTIAL_TEST,
            Self::VoiceInstructionList => METHOD_VOICE_INSTRUCTION_LIST,
            Self::VoiceInstructionSave => METHOD_VOICE_INSTRUCTION_SAVE,
            Self::VoiceInstructionDelete => METHOD_VOICE_INSTRUCTION_DELETE,
            Self::VoiceModelDefaultSet => METHOD_VOICE_MODEL_DEFAULT_SET,
            Self::VoiceModelTestTranscribeFile => METHOD_VOICE_MODEL_TEST_TRANSCRIBE_FILE,
            Self::VoiceTranscriptionTranscribeAudio => METHOD_VOICE_TRANSCRIPTION_TRANSCRIBE_AUDIO,
            Self::VoiceTranscriptionPolishText => METHOD_VOICE_TRANSCRIPTION_POLISH_TEXT,
            Self::WorkspaceSkillBindingsList => METHOD_WORKSPACE_SKILL_BINDINGS_LIST,
            Self::WorkspaceRegisteredSkillsList => METHOD_WORKSPACE_REGISTERED_SKILLS_LIST,
            Self::WorkspaceRightSurfaceRequest => METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST,
            Self::WorkspaceRightSurfacePendingList => METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_LIST,
            Self::WorkspaceRightSurfacePendingConsume => {
                METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CONSUME
            }
            Self::WorkspaceRightSurfacePendingDismiss => {
                METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_DISMISS
            }
            Self::BrowserSessionTargetList => METHOD_BROWSER_SESSION_TARGET_LIST,
            Self::BrowserSessionOpen => METHOD_BROWSER_SESSION_OPEN,
            Self::BrowserSessionRead => METHOD_BROWSER_SESSION_READ,
            Self::BrowserSessionClose => METHOD_BROWSER_SESSION_CLOSE,
            Self::BrowserSessionEventList => METHOD_BROWSER_SESSION_EVENT_LIST,
            Self::BrowserSessionActionExecute => METHOD_BROWSER_SESSION_ACTION_EXECUTE,
            Self::SoulStylePackInstall => METHOD_SOUL_STYLE_PACK_INSTALL,
            Self::SoulStylePackList => METHOD_SOUL_STYLE_PACK_LIST,
            Self::SoulStylePackStatusSet => METHOD_SOUL_STYLE_PACK_STATUS_SET,
            Self::SoulStylePackUninstall => METHOD_SOUL_STYLE_PACK_UNINSTALL,
            Self::KnowledgePackList => METHOD_KNOWLEDGE_PACK_LIST,
            Self::KnowledgePackRead => METHOD_KNOWLEDGE_PACK_READ,
            Self::KnowledgeSourceImport => METHOD_KNOWLEDGE_SOURCE_IMPORT,
            Self::KnowledgePackCompile => METHOD_KNOWLEDGE_PACK_COMPILE,
            Self::KnowledgePackDefaultSet => METHOD_KNOWLEDGE_PACK_DEFAULT_SET,
            Self::KnowledgePackStatusUpdate => METHOD_KNOWLEDGE_PACK_STATUS_UPDATE,
            Self::KnowledgeContextResolve => METHOD_KNOWLEDGE_CONTEXT_RESOLVE,
            Self::KnowledgeContextRunValidate => METHOD_KNOWLEDGE_CONTEXT_RUN_VALIDATE,
            Self::AutomationSchedulerConfigRead => METHOD_AUTOMATION_SCHEDULER_CONFIG_READ,
            Self::AutomationSchedulerConfigUpdate => METHOD_AUTOMATION_SCHEDULER_CONFIG_UPDATE,
            Self::AutomationSchedulerStatus => METHOD_AUTOMATION_SCHEDULER_STATUS,
            Self::AutomationJobList => METHOD_AUTOMATION_JOB_LIST,
            Self::AutomationJobRead => METHOD_AUTOMATION_JOB_READ,
            Self::AutomationJobCreate => METHOD_AUTOMATION_JOB_CREATE,
            Self::AutomationJobUpdate => METHOD_AUTOMATION_JOB_UPDATE,
            Self::AutomationJobDelete => METHOD_AUTOMATION_JOB_DELETE,
            Self::AutomationJobRunNow => METHOD_AUTOMATION_JOB_RUN_NOW,
            Self::AutomationJobHealth => METHOD_AUTOMATION_JOB_HEALTH,
            Self::AutomationJobRunHistory => METHOD_AUTOMATION_JOB_RUN_HISTORY,
            Self::AutomationSchedulePreview => METHOD_AUTOMATION_SCHEDULE_PREVIEW,
            Self::AutomationScheduleValidate => METHOD_AUTOMATION_SCHEDULE_VALIDATE,
            Self::ScheduledTaskList => METHOD_SCHEDULED_TASK_LIST,
            Self::ScheduledTaskRead => METHOD_SCHEDULED_TASK_READ,
            Self::ScheduledTaskCreate => METHOD_SCHEDULED_TASK_CREATE,
            Self::ScheduledTaskUpdate => METHOD_SCHEDULED_TASK_UPDATE,
            Self::ScheduledTaskDelete => METHOD_SCHEDULED_TASK_DELETE,
            Self::ScheduledTaskEnabledSet => METHOD_SCHEDULED_TASK_ENABLED_SET,
            Self::ScheduledTaskRunStart => METHOD_SCHEDULED_TASK_RUN_START,
            Self::ScheduledTaskRunList => METHOD_SCHEDULED_TASK_RUN_LIST,
            Self::ScheduledTaskSchedulePreview => METHOD_SCHEDULED_TASK_SCHEDULE_PREVIEW,
            Self::McpServerList => METHOD_MCP_SERVER_LIST,
            Self::McpServerStatusList => METHOD_MCP_SERVER_STATUS_LIST,
            Self::McpServerCreate => METHOD_MCP_SERVER_CREATE,
            Self::McpServerUpdate => METHOD_MCP_SERVER_UPDATE,
            Self::McpServerDelete => METHOD_MCP_SERVER_DELETE,
            Self::McpServerEnabledSet => METHOD_MCP_SERVER_ENABLED_SET,
            Self::McpServerImportFromApp => METHOD_MCP_SERVER_IMPORT_FROM_APP,
            Self::McpServerSyncAllToLive => METHOD_MCP_SERVER_SYNC_ALL_TO_LIVE,
            Self::McpServerOauthLogin => METHOD_MCP_SERVER_OAUTH_LOGIN,
            Self::McpServerStart => METHOD_MCP_SERVER_START,
            Self::McpServerStop => METHOD_MCP_SERVER_STOP,
            Self::McpToolList => METHOD_MCP_TOOL_LIST,
            Self::McpToolListForContext => METHOD_MCP_TOOL_LIST_FOR_CONTEXT,
            Self::McpToolSearch => METHOD_MCP_TOOL_SEARCH,
            Self::McpPromptList => METHOD_MCP_PROMPT_LIST,
            Self::McpPromptGet => METHOD_MCP_PROMPT_GET,
            Self::McpResourceList => METHOD_MCP_RESOURCE_LIST,
            Self::McpResourceSubscribe => METHOD_MCP_RESOURCE_SUBSCRIBE,
            Self::McpResourceUnsubscribe => METHOD_MCP_RESOURCE_UNSUBSCRIBE,
            Self::ProjectMemoryRead => METHOD_PROJECT_MEMORY_READ,
            Self::MemoryStoreList => METHOD_MEMORY_STORE_LIST,
            Self::MemoryStoreRead => METHOD_MEMORY_STORE_READ,
            Self::MemoryStoreSearch => METHOD_MEMORY_STORE_SEARCH,
            Self::MemoryStoreAddNote => METHOD_MEMORY_STORE_ADD_NOTE,
            Self::MemoryStoreConsolidate => METHOD_MEMORY_STORE_CONSOLIDATE,
            Self::MemoryStoreReviewList => METHOD_MEMORY_STORE_REVIEW_LIST,
            Self::MemoryStoreReviewResolve => METHOD_MEMORY_STORE_REVIEW_RESOLVE,
            Self::MemoryStoreHealth => METHOD_MEMORY_STORE_HEALTH,
            Self::MemoryStoreIndexRebuild => METHOD_MEMORY_STORE_INDEX_REBUILD,
            Self::LogList => METHOD_LOG_LIST,
            Self::LogPersistedTail => METHOD_LOG_PERSISTED_TAIL,
            Self::LogClear => METHOD_LOG_CLEAR,
            Self::LogDiagnosticHistoryClear => METHOD_LOG_DIAGNOSTIC_HISTORY_CLEAR,
            Self::DiagnosticsLogStorageRead => METHOD_DIAGNOSTICS_LOG_STORAGE_READ,
            Self::DiagnosticsSupportBundleExport => METHOD_DIAGNOSTICS_SUPPORT_BUNDLE_EXPORT,
            Self::DiagnosticsServerRead => METHOD_DIAGNOSTICS_SERVER_READ,
            Self::DiagnosticsWindowsStartupRead => METHOD_DIAGNOSTICS_WINDOWS_STARTUP_READ,
            Self::DiagnosticsTraceList => METHOD_DIAGNOSTICS_TRACE_LIST,
            Self::DiagnosticsTraceRead => METHOD_DIAGNOSTICS_TRACE_READ,
            Self::DiagnosticsTraceExport => METHOD_DIAGNOSTICS_TRACE_EXPORT,
            Self::UsageStatsRead => METHOD_USAGE_STATS_READ,
            Self::UsageStatsModelRankingList => METHOD_USAGE_STATS_MODEL_RANKING_LIST,
            Self::UsageStatsDailyTrendsList => METHOD_USAGE_STATS_DAILY_TRENDS_LIST,
            Self::ModelPreferencesList => METHOD_MODEL_PREFERENCES_LIST,
            Self::ModelSyncStateRead => METHOD_MODEL_SYNC_STATE_READ,
            Self::ModelProviderList => METHOD_MODEL_PROVIDER_LIST,
            Self::ModelProviderCatalogList => METHOD_MODEL_PROVIDER_CATALOG_LIST,
            Self::ModelProviderRead => METHOD_MODEL_PROVIDER_READ,
            Self::ModelProviderCreate => METHOD_MODEL_PROVIDER_CREATE,
            Self::ModelProviderUpdate => METHOD_MODEL_PROVIDER_UPDATE,
            Self::ModelProviderDelete => METHOD_MODEL_PROVIDER_DELETE,
            Self::ModelProviderSortOrdersUpdate => METHOD_MODEL_PROVIDER_SORT_ORDERS_UPDATE,
            Self::ModelProviderConfigExport => METHOD_MODEL_PROVIDER_CONFIG_EXPORT,
            Self::ModelProviderConfigImport => METHOD_MODEL_PROVIDER_CONFIG_IMPORT,
            Self::ModelProviderTestConnection => METHOD_MODEL_PROVIDER_TEST_CONNECTION,
            Self::ModelProviderTestChat => METHOD_MODEL_PROVIDER_TEST_CHAT,
            Self::ModelProviderFetchModels => METHOD_MODEL_PROVIDER_FETCH_MODELS,
            Self::ModelProviderKeyCreate => METHOD_MODEL_PROVIDER_KEY_CREATE,
            Self::ModelProviderKeyUpdate => METHOD_MODEL_PROVIDER_KEY_UPDATE,
            Self::ModelProviderKeyDelete => METHOD_MODEL_PROVIDER_KEY_DELETE,
            Self::ModelProviderUiStateRead => METHOD_MODEL_PROVIDER_UI_STATE_READ,
            Self::ModelProviderUiStateWrite => METHOD_MODEL_PROVIDER_UI_STATE_WRITE,
            Self::ModelProviderAliasRead => METHOD_MODEL_PROVIDER_ALIAS_READ,
            Self::ModelProviderAliasList => METHOD_MODEL_PROVIDER_ALIAS_LIST,
            Self::ConnectDeepLinkResolve => METHOD_CONNECT_DEEP_LINK_RESOLVE,
            Self::ConnectOpenDeepLinkResolve => METHOD_CONNECT_OPEN_DEEP_LINK_RESOLVE,
            Self::ConnectRelayApiKeySave => METHOD_CONNECT_RELAY_API_KEY_SAVE,
            Self::ConnectCallbackSend => METHOD_CONNECT_CALLBACK_SEND,
            Self::ConversationImportSourceScan => METHOD_CONVERSATION_IMPORT_SOURCE_SCAN,
            Self::ConversationImportThreadPreview => METHOD_CONVERSATION_IMPORT_THREAD_PREVIEW,
            Self::ConversationImportThreadCommit => METHOD_CONVERSATION_IMPORT_THREAD_COMMIT,
            Self::ConversationImportJobRead => METHOD_CONVERSATION_IMPORT_JOB_READ,
            Self::AgentSessionActionRespond => METHOD_AGENT_SESSION_ACTION_RESPOND,
            Self::WorkflowRead => METHOD_WORKFLOW_READ,
            Self::WorkflowCancel => METHOD_WORKFLOW_CANCEL,
            Self::WorkflowRetry => METHOD_WORKFLOW_RETRY,
            Self::WorkflowRespond => METHOD_WORKFLOW_RESPOND,
        }
    }

    pub fn parse(method: &str) -> Option<Self> {
        match method {
            METHOD_INITIALIZE => Some(Self::Initialize),
            METHOD_CAPABILITY_LIST => Some(Self::CapabilityList),
            METHOD_ARTIFACT_READ => Some(Self::ArtifactRead),
            METHOD_PROJECT_GIT_STATUS => Some(Self::ProjectGitStatus),
            METHOD_PROJECT_GIT_DIFF => Some(Self::ProjectGitDiff),
            METHOD_PROJECT_GIT_COMMITS_LIST => Some(Self::ProjectGitCommitsList),
            METHOD_PROJECT_GIT_BRANCH_CHECKOUT => Some(Self::ProjectGitBranchCheckout),
            METHOD_PROJECT_GIT_BRANCH_CREATE => Some(Self::ProjectGitBranchCreate),
            METHOD_PROJECT_GIT_WORKTREE_CREATE => Some(Self::ProjectGitWorktreeCreate),
            METHOD_AGENT_SESSION_HANDOFF_BUNDLE_EXPORT => {
                Some(Self::AgentSessionHandoffBundleExport)
            }
            METHOD_AGENT_SESSION_REPLAY_CASE_EXPORT => Some(Self::AgentSessionReplayCaseExport),
            METHOD_AGENT_SESSION_ANALYSIS_HANDOFF_EXPORT => {
                Some(Self::AgentSessionAnalysisHandoffExport)
            }
            METHOD_AGENT_SESSION_REVIEW_DECISION_TEMPLATE_EXPORT => {
                Some(Self::AgentSessionReviewDecisionTemplateExport)
            }
            METHOD_AGENT_SESSION_REVIEW_DECISION_SAVE => Some(Self::AgentSessionReviewDecisionSave),
            METHOD_AGENT_SESSION_FILE_CHECKPOINT_LIST => Some(Self::AgentSessionFileCheckpointList),
            METHOD_AGENT_SESSION_FILE_CHECKPOINT_GET => Some(Self::AgentSessionFileCheckpointGet),
            METHOD_AGENT_SESSION_FILE_CHECKPOINT_DIFF => Some(Self::AgentSessionFileCheckpointDiff),
            METHOD_AGENT_SESSION_FILE_CHECKPOINT_RESTORE => {
                Some(Self::AgentSessionFileCheckpointRestore)
            }
            METHOD_AGENT_SESSION_TOOL_INVENTORY_READ => Some(Self::AgentSessionToolInventoryRead),
            METHOD_SESSION_FILE_GET_OR_CREATE => Some(Self::SessionFileGetOrCreate),
            METHOD_SESSION_FILE_UPDATE_META => Some(Self::SessionFileUpdateMeta),
            METHOD_SESSION_FILE_SAVE => Some(Self::SessionFileSave),
            METHOD_SESSION_FILE_READ => Some(Self::SessionFileRead),
            METHOD_SESSION_FILE_RESOLVE_PATH => Some(Self::SessionFileResolvePath),
            METHOD_SESSION_FILE_DELETE => Some(Self::SessionFileDelete),
            METHOD_SESSION_FILE_LIST => Some(Self::SessionFileList),
            METHOD_WORKSPACE_LIST => Some(Self::WorkspaceList),
            METHOD_WORKSPACE_READ => Some(Self::WorkspaceRead),
            METHOD_WORKSPACE_UPDATE => Some(Self::WorkspaceUpdate),
            METHOD_WORKSPACE_DELETE => Some(Self::WorkspaceDelete),
            METHOD_WORKSPACE_ENSURE => Some(Self::WorkspaceEnsure),
            METHOD_WORKSPACE_BY_PATH_READ => Some(Self::WorkspaceByPathRead),
            METHOD_WORKSPACE_DEFAULT_READ => Some(Self::WorkspaceDefaultRead),
            METHOD_WORKSPACE_DEFAULT_ENSURE => Some(Self::WorkspaceDefaultEnsure),
            METHOD_WORKSPACE_PROJECTS_ROOT_READ => Some(Self::WorkspaceProjectsRootRead),
            METHOD_WORKSPACE_PROJECT_PATH_RESOLVE => Some(Self::WorkspaceProjectPathResolve),
            METHOD_WORKSPACE_ENSURE_READY => Some(Self::WorkspaceEnsureReady),
            METHOD_SKILL_READ => Some(Self::SkillRead),
            METHOD_SKILL_MANAGEMENT_LIST => Some(Self::SkillManagementList),
            METHOD_SKILL_MANAGEMENT_INSTALL => Some(Self::SkillManagementInstall),
            METHOD_SKILL_MANAGEMENT_UNINSTALL => Some(Self::SkillManagementUninstall),
            METHOD_SKILL_REPOSITORY_LIST => Some(Self::SkillRepositoryList),
            METHOD_SKILL_REPOSITORY_SAVE => Some(Self::SkillRepositorySave),
            METHOD_SKILL_REPOSITORY_DELETE => Some(Self::SkillRepositoryDelete),
            METHOD_SKILL_CACHE_REFRESH => Some(Self::SkillCacheRefresh),
            METHOD_SKILL_INSTALLED_DIRECTORIES_LIST => Some(Self::SkillInstalledDirectoriesList),
            METHOD_SKILL_LOCAL_INSPECT => Some(Self::SkillLocalInspect),
            METHOD_SKILL_LOCAL_DETAIL_INSPECT => Some(Self::SkillLocalDetailInspect),
            METHOD_SKILL_LOCAL_SCAFFOLD_CREATE => Some(Self::SkillLocalScaffoldCreate),
            METHOD_SKILL_LOCAL_IMPORT => Some(Self::SkillLocalImport),
            METHOD_SKILL_LOCAL_RENAME => Some(Self::SkillLocalRename),
            METHOD_SKILL_REMOTE_INSPECT => Some(Self::SkillRemoteInspect),
            METHOD_SKILL_PACKAGE_LOCAL_INSPECT => Some(Self::SkillPackageLocalInspect),
            METHOD_SKILL_PACKAGE_LOCAL_INSTALL => Some(Self::SkillPackageLocalInstall),
            METHOD_SKILL_PACKAGE_LOCAL_REPLACE => Some(Self::SkillPackageLocalReplace),
            METHOD_SKILL_PACKAGE_EXPORT => Some(Self::SkillPackageExport),
            METHOD_SKILL_MARKETPLACE_INSTALL => Some(Self::SkillMarketplaceInstall),
            METHOD_SKILL_PACKAGE_DOWNLOAD_INSTALL => Some(Self::SkillPackageDownloadInstall),
            METHOD_GATEWAY_CHANNEL_START => Some(Self::GatewayChannelStart),
            METHOD_GATEWAY_CHANNEL_STOP => Some(Self::GatewayChannelStop),
            METHOD_GATEWAY_CHANNEL_STATUS => Some(Self::GatewayChannelStatus),
            METHOD_TELEGRAM_CHANNEL_PROBE => Some(Self::TelegramChannelProbe),
            METHOD_FEISHU_CHANNEL_PROBE => Some(Self::FeishuChannelProbe),
            METHOD_DISCORD_CHANNEL_PROBE => Some(Self::DiscordChannelProbe),
            METHOD_WECHAT_CHANNEL_PROBE => Some(Self::WechatChannelProbe),
            METHOD_WECHAT_CHANNEL_LOGIN_START => Some(Self::WechatChannelLoginStart),
            METHOD_WECHAT_CHANNEL_LOGIN_WAIT => Some(Self::WechatChannelLoginWait),
            METHOD_WECHAT_CHANNEL_ACCOUNT_LIST => Some(Self::WechatChannelAccountList),
            METHOD_WECHAT_CHANNEL_ACCOUNT_REMOVE => Some(Self::WechatChannelAccountRemove),
            METHOD_WECHAT_CHANNEL_RUNTIME_MODEL_SET => Some(Self::WechatChannelRuntimeModelSet),
            METHOD_GATEWAY_TUNNEL_PROBE => Some(Self::GatewayTunnelProbe),
            METHOD_GATEWAY_TUNNEL_CLOUDFLARED_DETECT => Some(Self::GatewayTunnelCloudflaredDetect),
            METHOD_GATEWAY_TUNNEL_CLOUDFLARED_INSTALL => {
                Some(Self::GatewayTunnelCloudflaredInstall)
            }
            METHOD_GATEWAY_TUNNEL_CREATE => Some(Self::GatewayTunnelCreate),
            METHOD_GATEWAY_TUNNEL_START => Some(Self::GatewayTunnelStart),
            METHOD_GATEWAY_TUNNEL_STOP => Some(Self::GatewayTunnelStop),
            METHOD_GATEWAY_TUNNEL_RESTART => Some(Self::GatewayTunnelRestart),
            METHOD_GATEWAY_TUNNEL_STATUS => Some(Self::GatewayTunnelStatus),
            METHOD_GATEWAY_TUNNEL_SYNC_WEBHOOK_URL => Some(Self::GatewayTunnelSyncWebhookUrl),
            METHOD_MEDIA_TASK_ARTIFACT_IMAGE_CREATE => Some(Self::MediaTaskArtifactImageCreate),
            METHOD_MEDIA_TASK_ARTIFACT_AUDIO_CREATE => Some(Self::MediaTaskArtifactAudioCreate),
            METHOD_MEDIA_TASK_ARTIFACT_TRANSCRIPTION_CREATE => {
                Some(Self::MediaTaskArtifactTranscriptionCreate)
            }
            METHOD_MEDIA_TASK_ARTIFACT_VIDEO_CREATE => Some(Self::MediaTaskArtifactVideoCreate),
            METHOD_MEDIA_TASK_ARTIFACT_IMAGE_COMPLETE => Some(Self::MediaTaskArtifactImageComplete),
            METHOD_MEDIA_TASK_ARTIFACT_AUDIO_COMPLETE => Some(Self::MediaTaskArtifactAudioComplete),
            METHOD_MEDIA_TASK_ARTIFACT_GET => Some(Self::MediaTaskArtifactGet),
            METHOD_MEDIA_TASK_ARTIFACT_LIST => Some(Self::MediaTaskArtifactList),
            METHOD_MEDIA_TASK_ARTIFACT_CANCEL => Some(Self::MediaTaskArtifactCancel),
            METHOD_GALLERY_MATERIAL_GET => Some(Self::GalleryMaterialGet),
            METHOD_GALLERY_MATERIAL_METADATA_CREATE => Some(Self::GalleryMaterialMetadataCreate),
            METHOD_GALLERY_MATERIAL_METADATA_GET => Some(Self::GalleryMaterialMetadataGet),
            METHOD_GALLERY_MATERIAL_METADATA_UPDATE => Some(Self::GalleryMaterialMetadataUpdate),
            METHOD_GALLERY_MATERIAL_METADATA_DELETE => Some(Self::GalleryMaterialMetadataDelete),
            METHOD_GALLERY_MATERIAL_LIST_BY_IMAGE_CATEGORY => {
                Some(Self::GalleryMaterialListByImageCategory)
            }
            METHOD_GALLERY_MATERIAL_LIST_BY_LAYOUT_CATEGORY => {
                Some(Self::GalleryMaterialListByLayoutCategory)
            }
            METHOD_GALLERY_MATERIAL_LIST_BY_MOOD => Some(Self::GalleryMaterialListByMood),
            METHOD_PROJECT_MATERIAL_LIST => Some(Self::ProjectMaterialList),
            METHOD_PROJECT_MATERIAL_GET => Some(Self::ProjectMaterialGet),
            METHOD_PROJECT_MATERIAL_COUNT => Some(Self::ProjectMaterialCount),
            METHOD_PROJECT_MATERIAL_UPLOAD => Some(Self::ProjectMaterialUpload),
            METHOD_PROJECT_MATERIAL_IMPORT_FROM_URL => Some(Self::ProjectMaterialImportFromUrl),
            METHOD_PROJECT_MATERIAL_UPDATE => Some(Self::ProjectMaterialUpdate),
            METHOD_PROJECT_MATERIAL_DELETE => Some(Self::ProjectMaterialDelete),
            METHOD_PROJECT_MATERIAL_CONTENT => Some(Self::ProjectMaterialContent),
            METHOD_VOICE_ASR_CREDENTIAL_LIST => Some(Self::VoiceAsrCredentialList),
            METHOD_VOICE_ASR_CREDENTIAL_CREATE => Some(Self::VoiceAsrCredentialCreate),
            METHOD_VOICE_ASR_CREDENTIAL_UPDATE => Some(Self::VoiceAsrCredentialUpdate),
            METHOD_VOICE_ASR_CREDENTIAL_DELETE => Some(Self::VoiceAsrCredentialDelete),
            METHOD_VOICE_ASR_CREDENTIAL_DEFAULT_SET => Some(Self::VoiceAsrCredentialDefaultSet),
            METHOD_VOICE_ASR_CREDENTIAL_TEST => Some(Self::VoiceAsrCredentialTest),
            METHOD_VOICE_INSTRUCTION_LIST => Some(Self::VoiceInstructionList),
            METHOD_VOICE_INSTRUCTION_SAVE => Some(Self::VoiceInstructionSave),
            METHOD_VOICE_INSTRUCTION_DELETE => Some(Self::VoiceInstructionDelete),
            METHOD_VOICE_MODEL_DEFAULT_SET => Some(Self::VoiceModelDefaultSet),
            METHOD_VOICE_MODEL_TEST_TRANSCRIBE_FILE => Some(Self::VoiceModelTestTranscribeFile),
            METHOD_VOICE_TRANSCRIPTION_TRANSCRIBE_AUDIO => {
                Some(Self::VoiceTranscriptionTranscribeAudio)
            }
            METHOD_VOICE_TRANSCRIPTION_POLISH_TEXT => Some(Self::VoiceTranscriptionPolishText),
            METHOD_WORKSPACE_SKILL_BINDINGS_LIST => Some(Self::WorkspaceSkillBindingsList),
            METHOD_WORKSPACE_REGISTERED_SKILLS_LIST => Some(Self::WorkspaceRegisteredSkillsList),
            METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST => Some(Self::WorkspaceRightSurfaceRequest),
            METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_LIST => {
                Some(Self::WorkspaceRightSurfacePendingList)
            }
            METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CONSUME => {
                Some(Self::WorkspaceRightSurfacePendingConsume)
            }
            METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_DISMISS => {
                Some(Self::WorkspaceRightSurfacePendingDismiss)
            }
            METHOD_BROWSER_SESSION_TARGET_LIST => Some(Self::BrowserSessionTargetList),
            METHOD_BROWSER_SESSION_OPEN => Some(Self::BrowserSessionOpen),
            METHOD_BROWSER_SESSION_READ => Some(Self::BrowserSessionRead),
            METHOD_BROWSER_SESSION_CLOSE => Some(Self::BrowserSessionClose),
            METHOD_BROWSER_SESSION_EVENT_LIST => Some(Self::BrowserSessionEventList),
            METHOD_BROWSER_SESSION_ACTION_EXECUTE => Some(Self::BrowserSessionActionExecute),
            METHOD_SOUL_STYLE_PACK_INSTALL => Some(Self::SoulStylePackInstall),
            METHOD_SOUL_STYLE_PACK_LIST => Some(Self::SoulStylePackList),
            METHOD_SOUL_STYLE_PACK_STATUS_SET => Some(Self::SoulStylePackStatusSet),
            METHOD_SOUL_STYLE_PACK_UNINSTALL => Some(Self::SoulStylePackUninstall),
            METHOD_KNOWLEDGE_PACK_LIST => Some(Self::KnowledgePackList),
            METHOD_KNOWLEDGE_PACK_READ => Some(Self::KnowledgePackRead),
            METHOD_KNOWLEDGE_SOURCE_IMPORT => Some(Self::KnowledgeSourceImport),
            METHOD_KNOWLEDGE_PACK_COMPILE => Some(Self::KnowledgePackCompile),
            METHOD_KNOWLEDGE_PACK_DEFAULT_SET => Some(Self::KnowledgePackDefaultSet),
            METHOD_KNOWLEDGE_PACK_STATUS_UPDATE => Some(Self::KnowledgePackStatusUpdate),
            METHOD_KNOWLEDGE_CONTEXT_RESOLVE => Some(Self::KnowledgeContextResolve),
            METHOD_KNOWLEDGE_CONTEXT_RUN_VALIDATE => Some(Self::KnowledgeContextRunValidate),
            METHOD_AUTOMATION_SCHEDULER_CONFIG_READ => Some(Self::AutomationSchedulerConfigRead),
            METHOD_AUTOMATION_SCHEDULER_CONFIG_UPDATE => {
                Some(Self::AutomationSchedulerConfigUpdate)
            }
            METHOD_AUTOMATION_SCHEDULER_STATUS => Some(Self::AutomationSchedulerStatus),
            METHOD_AUTOMATION_JOB_LIST => Some(Self::AutomationJobList),
            METHOD_AUTOMATION_JOB_READ => Some(Self::AutomationJobRead),
            METHOD_AUTOMATION_JOB_CREATE => Some(Self::AutomationJobCreate),
            METHOD_AUTOMATION_JOB_UPDATE => Some(Self::AutomationJobUpdate),
            METHOD_AUTOMATION_JOB_DELETE => Some(Self::AutomationJobDelete),
            METHOD_AUTOMATION_JOB_RUN_NOW => Some(Self::AutomationJobRunNow),
            METHOD_AUTOMATION_JOB_HEALTH => Some(Self::AutomationJobHealth),
            METHOD_AUTOMATION_JOB_RUN_HISTORY => Some(Self::AutomationJobRunHistory),
            METHOD_AUTOMATION_SCHEDULE_PREVIEW => Some(Self::AutomationSchedulePreview),
            METHOD_AUTOMATION_SCHEDULE_VALIDATE => Some(Self::AutomationScheduleValidate),
            METHOD_SCHEDULED_TASK_LIST => Some(Self::ScheduledTaskList),
            METHOD_SCHEDULED_TASK_READ => Some(Self::ScheduledTaskRead),
            METHOD_SCHEDULED_TASK_CREATE => Some(Self::ScheduledTaskCreate),
            METHOD_SCHEDULED_TASK_UPDATE => Some(Self::ScheduledTaskUpdate),
            METHOD_SCHEDULED_TASK_DELETE => Some(Self::ScheduledTaskDelete),
            METHOD_SCHEDULED_TASK_ENABLED_SET => Some(Self::ScheduledTaskEnabledSet),
            METHOD_SCHEDULED_TASK_RUN_START => Some(Self::ScheduledTaskRunStart),
            METHOD_SCHEDULED_TASK_RUN_LIST => Some(Self::ScheduledTaskRunList),
            METHOD_SCHEDULED_TASK_SCHEDULE_PREVIEW => Some(Self::ScheduledTaskSchedulePreview),
            METHOD_MCP_SERVER_LIST => Some(Self::McpServerList),
            METHOD_MCP_SERVER_STATUS_LIST => Some(Self::McpServerStatusList),
            METHOD_MCP_SERVER_CREATE => Some(Self::McpServerCreate),
            METHOD_MCP_SERVER_UPDATE => Some(Self::McpServerUpdate),
            METHOD_MCP_SERVER_DELETE => Some(Self::McpServerDelete),
            METHOD_MCP_SERVER_ENABLED_SET => Some(Self::McpServerEnabledSet),
            METHOD_MCP_SERVER_IMPORT_FROM_APP => Some(Self::McpServerImportFromApp),
            METHOD_MCP_SERVER_SYNC_ALL_TO_LIVE => Some(Self::McpServerSyncAllToLive),
            METHOD_MCP_SERVER_OAUTH_LOGIN => Some(Self::McpServerOauthLogin),
            METHOD_MCP_SERVER_START => Some(Self::McpServerStart),
            METHOD_MCP_SERVER_STOP => Some(Self::McpServerStop),
            METHOD_MCP_TOOL_LIST => Some(Self::McpToolList),
            METHOD_MCP_TOOL_LIST_FOR_CONTEXT => Some(Self::McpToolListForContext),
            METHOD_MCP_TOOL_SEARCH => Some(Self::McpToolSearch),
            METHOD_MCP_PROMPT_LIST => Some(Self::McpPromptList),
            METHOD_MCP_PROMPT_GET => Some(Self::McpPromptGet),
            METHOD_MCP_RESOURCE_LIST => Some(Self::McpResourceList),
            METHOD_MCP_RESOURCE_SUBSCRIBE => Some(Self::McpResourceSubscribe),
            METHOD_MCP_RESOURCE_UNSUBSCRIBE => Some(Self::McpResourceUnsubscribe),
            METHOD_PROJECT_MEMORY_READ => Some(Self::ProjectMemoryRead),
            METHOD_MEMORY_STORE_LIST => Some(Self::MemoryStoreList),
            METHOD_MEMORY_STORE_READ => Some(Self::MemoryStoreRead),
            METHOD_MEMORY_STORE_SEARCH => Some(Self::MemoryStoreSearch),
            METHOD_MEMORY_STORE_ADD_NOTE => Some(Self::MemoryStoreAddNote),
            METHOD_MEMORY_STORE_CONSOLIDATE => Some(Self::MemoryStoreConsolidate),
            METHOD_MEMORY_STORE_REVIEW_LIST => Some(Self::MemoryStoreReviewList),
            METHOD_MEMORY_STORE_REVIEW_RESOLVE => Some(Self::MemoryStoreReviewResolve),
            METHOD_MEMORY_STORE_HEALTH => Some(Self::MemoryStoreHealth),
            METHOD_MEMORY_STORE_INDEX_REBUILD => Some(Self::MemoryStoreIndexRebuild),
            METHOD_LOG_LIST => Some(Self::LogList),
            METHOD_LOG_PERSISTED_TAIL => Some(Self::LogPersistedTail),
            METHOD_LOG_CLEAR => Some(Self::LogClear),
            METHOD_LOG_DIAGNOSTIC_HISTORY_CLEAR => Some(Self::LogDiagnosticHistoryClear),
            METHOD_DIAGNOSTICS_LOG_STORAGE_READ => Some(Self::DiagnosticsLogStorageRead),
            METHOD_DIAGNOSTICS_SUPPORT_BUNDLE_EXPORT => Some(Self::DiagnosticsSupportBundleExport),
            METHOD_DIAGNOSTICS_SERVER_READ => Some(Self::DiagnosticsServerRead),
            METHOD_DIAGNOSTICS_WINDOWS_STARTUP_READ => Some(Self::DiagnosticsWindowsStartupRead),
            METHOD_DIAGNOSTICS_TRACE_LIST => Some(Self::DiagnosticsTraceList),
            METHOD_DIAGNOSTICS_TRACE_READ => Some(Self::DiagnosticsTraceRead),
            METHOD_DIAGNOSTICS_TRACE_EXPORT => Some(Self::DiagnosticsTraceExport),
            METHOD_USAGE_STATS_READ => Some(Self::UsageStatsRead),
            METHOD_USAGE_STATS_MODEL_RANKING_LIST => Some(Self::UsageStatsModelRankingList),
            METHOD_USAGE_STATS_DAILY_TRENDS_LIST => Some(Self::UsageStatsDailyTrendsList),
            METHOD_MODEL_PREFERENCES_LIST => Some(Self::ModelPreferencesList),
            METHOD_MODEL_SYNC_STATE_READ => Some(Self::ModelSyncStateRead),
            METHOD_MODEL_PROVIDER_LIST => Some(Self::ModelProviderList),
            METHOD_MODEL_PROVIDER_CATALOG_LIST => Some(Self::ModelProviderCatalogList),
            METHOD_MODEL_PROVIDER_READ => Some(Self::ModelProviderRead),
            METHOD_MODEL_PROVIDER_CREATE => Some(Self::ModelProviderCreate),
            METHOD_MODEL_PROVIDER_UPDATE => Some(Self::ModelProviderUpdate),
            METHOD_MODEL_PROVIDER_DELETE => Some(Self::ModelProviderDelete),
            METHOD_MODEL_PROVIDER_SORT_ORDERS_UPDATE => Some(Self::ModelProviderSortOrdersUpdate),
            METHOD_MODEL_PROVIDER_CONFIG_EXPORT => Some(Self::ModelProviderConfigExport),
            METHOD_MODEL_PROVIDER_CONFIG_IMPORT => Some(Self::ModelProviderConfigImport),
            METHOD_MODEL_PROVIDER_TEST_CONNECTION => Some(Self::ModelProviderTestConnection),
            METHOD_MODEL_PROVIDER_TEST_CHAT => Some(Self::ModelProviderTestChat),
            METHOD_MODEL_PROVIDER_FETCH_MODELS => Some(Self::ModelProviderFetchModels),
            METHOD_MODEL_PROVIDER_KEY_CREATE => Some(Self::ModelProviderKeyCreate),
            METHOD_MODEL_PROVIDER_KEY_UPDATE => Some(Self::ModelProviderKeyUpdate),
            METHOD_MODEL_PROVIDER_KEY_DELETE => Some(Self::ModelProviderKeyDelete),
            METHOD_MODEL_PROVIDER_UI_STATE_READ => Some(Self::ModelProviderUiStateRead),
            METHOD_MODEL_PROVIDER_UI_STATE_WRITE => Some(Self::ModelProviderUiStateWrite),
            METHOD_MODEL_PROVIDER_ALIAS_READ => Some(Self::ModelProviderAliasRead),
            METHOD_MODEL_PROVIDER_ALIAS_LIST => Some(Self::ModelProviderAliasList),
            METHOD_CONNECT_DEEP_LINK_RESOLVE => Some(Self::ConnectDeepLinkResolve),
            METHOD_CONNECT_OPEN_DEEP_LINK_RESOLVE => Some(Self::ConnectOpenDeepLinkResolve),
            METHOD_CONNECT_RELAY_API_KEY_SAVE => Some(Self::ConnectRelayApiKeySave),
            METHOD_CONNECT_CALLBACK_SEND => Some(Self::ConnectCallbackSend),
            METHOD_CONVERSATION_IMPORT_SOURCE_SCAN => Some(Self::ConversationImportSourceScan),
            METHOD_CONVERSATION_IMPORT_THREAD_PREVIEW => {
                Some(Self::ConversationImportThreadPreview)
            }
            METHOD_CONVERSATION_IMPORT_THREAD_COMMIT => Some(Self::ConversationImportThreadCommit),
            METHOD_CONVERSATION_IMPORT_JOB_READ => Some(Self::ConversationImportJobRead),
            METHOD_AGENT_SESSION_ACTION_RESPOND => Some(Self::AgentSessionActionRespond),
            METHOD_WORKFLOW_READ => Some(Self::WorkflowRead),
            METHOD_WORKFLOW_CANCEL => Some(Self::WorkflowCancel),
            METHOD_WORKFLOW_RETRY => Some(Self::WorkflowRetry),
            METHOD_WORKFLOW_RESPOND => Some(Self::WorkflowRespond),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub enum AppServerNotificationMethod {
    #[serde(rename = "initialized")]
    Initialized,
    #[serde(rename = "workspaceRightSurface/pendingChanged")]
    WorkspaceRightSurfacePendingChanged,
    #[serde(rename = "agentSession/event")]
    AgentSessionEvent,
}

impl AppServerNotificationMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Initialized => METHOD_INITIALIZED,
            Self::WorkspaceRightSurfacePendingChanged => {
                METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CHANGED
            }
            Self::AgentSessionEvent => METHOD_AGENT_SESSION_EVENT,
        }
    }

    pub fn parse(method: &str) -> Option<Self> {
        match method {
            METHOD_INITIALIZED => Some(Self::Initialized),
            METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CHANGED => {
                Some(Self::WorkspaceRightSurfacePendingChanged)
            }
            METHOD_AGENT_SESSION_EVENT => Some(Self::AgentSessionEvent),
            _ => None,
        }
    }
}

pub const APP_SERVER_METHODS: &[AppServerMethodSpec] = &[
    AppServerMethodSpec {
        method: METHOD_INITIALIZE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_INITIALIZED,
        kind: AppServerMethodKind::Notification,
    },
    AppServerMethodSpec {
        method: METHOD_CAPABILITY_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_ARTIFACT_READ,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_PROJECT_GIT_STATUS,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_PROJECT_GIT_DIFF,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_PROJECT_GIT_COMMITS_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_PROJECT_GIT_BRANCH_CHECKOUT,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_PROJECT_GIT_BRANCH_CREATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_PROJECT_GIT_WORKTREE_CREATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_AGENT_SESSION_HANDOFF_BUNDLE_EXPORT,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_AGENT_SESSION_REPLAY_CASE_EXPORT,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_AGENT_SESSION_ANALYSIS_HANDOFF_EXPORT,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_AGENT_SESSION_REVIEW_DECISION_TEMPLATE_EXPORT,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_AGENT_SESSION_REVIEW_DECISION_SAVE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_AGENT_SESSION_FILE_CHECKPOINT_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_AGENT_SESSION_FILE_CHECKPOINT_GET,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_AGENT_SESSION_FILE_CHECKPOINT_DIFF,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_AGENT_SESSION_FILE_CHECKPOINT_RESTORE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_AGENT_SESSION_TOOL_INVENTORY_READ,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SESSION_FILE_GET_OR_CREATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SESSION_FILE_UPDATE_META,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SESSION_FILE_SAVE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SESSION_FILE_READ,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SESSION_FILE_RESOLVE_PATH,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SESSION_FILE_DELETE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SESSION_FILE_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WORKSPACE_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WORKSPACE_READ,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WORKSPACE_UPDATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WORKSPACE_DELETE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WORKSPACE_ENSURE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WORKSPACE_BY_PATH_READ,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WORKSPACE_DEFAULT_READ,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WORKSPACE_DEFAULT_ENSURE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WORKSPACE_PROJECTS_ROOT_READ,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WORKSPACE_PROJECT_PATH_RESOLVE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WORKSPACE_ENSURE_READY,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SKILL_READ,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SKILL_MANAGEMENT_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SKILL_MANAGEMENT_INSTALL,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SKILL_MANAGEMENT_UNINSTALL,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SKILL_REPOSITORY_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SKILL_REPOSITORY_SAVE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SKILL_REPOSITORY_DELETE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SKILL_CACHE_REFRESH,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SKILL_INSTALLED_DIRECTORIES_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SKILL_LOCAL_INSPECT,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SKILL_LOCAL_DETAIL_INSPECT,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SKILL_LOCAL_SCAFFOLD_CREATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SKILL_LOCAL_IMPORT,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SKILL_LOCAL_RENAME,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SKILL_REMOTE_INSPECT,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SKILL_PACKAGE_LOCAL_INSPECT,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SKILL_PACKAGE_LOCAL_INSTALL,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SKILL_PACKAGE_LOCAL_REPLACE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SKILL_PACKAGE_EXPORT,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SKILL_MARKETPLACE_INSTALL,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SKILL_PACKAGE_DOWNLOAD_INSTALL,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_GATEWAY_CHANNEL_START,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_GATEWAY_CHANNEL_STOP,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_GATEWAY_CHANNEL_STATUS,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_TELEGRAM_CHANNEL_PROBE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_FEISHU_CHANNEL_PROBE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_DISCORD_CHANNEL_PROBE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WECHAT_CHANNEL_PROBE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WECHAT_CHANNEL_LOGIN_START,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WECHAT_CHANNEL_LOGIN_WAIT,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WECHAT_CHANNEL_ACCOUNT_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WECHAT_CHANNEL_ACCOUNT_REMOVE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WECHAT_CHANNEL_RUNTIME_MODEL_SET,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_GATEWAY_TUNNEL_PROBE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_GATEWAY_TUNNEL_CLOUDFLARED_DETECT,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_GATEWAY_TUNNEL_CLOUDFLARED_INSTALL,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_GATEWAY_TUNNEL_CREATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_GATEWAY_TUNNEL_START,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_GATEWAY_TUNNEL_STOP,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_GATEWAY_TUNNEL_RESTART,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_GATEWAY_TUNNEL_STATUS,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_GATEWAY_TUNNEL_SYNC_WEBHOOK_URL,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MEDIA_TASK_ARTIFACT_IMAGE_CREATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MEDIA_TASK_ARTIFACT_AUDIO_CREATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MEDIA_TASK_ARTIFACT_TRANSCRIPTION_CREATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MEDIA_TASK_ARTIFACT_VIDEO_CREATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MEDIA_TASK_ARTIFACT_IMAGE_COMPLETE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MEDIA_TASK_ARTIFACT_AUDIO_COMPLETE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MEDIA_TASK_ARTIFACT_GET,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MEDIA_TASK_ARTIFACT_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MEDIA_TASK_ARTIFACT_CANCEL,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_GALLERY_MATERIAL_GET,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_GALLERY_MATERIAL_METADATA_CREATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_GALLERY_MATERIAL_METADATA_GET,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_GALLERY_MATERIAL_METADATA_UPDATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_GALLERY_MATERIAL_METADATA_DELETE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_GALLERY_MATERIAL_LIST_BY_IMAGE_CATEGORY,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_GALLERY_MATERIAL_LIST_BY_LAYOUT_CATEGORY,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_GALLERY_MATERIAL_LIST_BY_MOOD,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_PROJECT_MATERIAL_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_PROJECT_MATERIAL_GET,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_PROJECT_MATERIAL_COUNT,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_PROJECT_MATERIAL_UPLOAD,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_PROJECT_MATERIAL_IMPORT_FROM_URL,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_PROJECT_MATERIAL_UPDATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_PROJECT_MATERIAL_DELETE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_PROJECT_MATERIAL_CONTENT,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_VOICE_ASR_CREDENTIAL_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_VOICE_ASR_CREDENTIAL_CREATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_VOICE_ASR_CREDENTIAL_UPDATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_VOICE_ASR_CREDENTIAL_DELETE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_VOICE_ASR_CREDENTIAL_DEFAULT_SET,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_VOICE_ASR_CREDENTIAL_TEST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_VOICE_INSTRUCTION_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_VOICE_INSTRUCTION_SAVE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_VOICE_INSTRUCTION_DELETE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_VOICE_MODEL_DEFAULT_SET,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_VOICE_MODEL_TEST_TRANSCRIBE_FILE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_VOICE_TRANSCRIPTION_TRANSCRIBE_AUDIO,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_VOICE_TRANSCRIPTION_POLISH_TEXT,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WORKSPACE_SKILL_BINDINGS_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WORKSPACE_REGISTERED_SKILLS_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CONSUME,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_DISMISS,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CHANGED,
        kind: AppServerMethodKind::Notification,
    },
    AppServerMethodSpec {
        method: METHOD_BROWSER_SESSION_TARGET_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_BROWSER_SESSION_OPEN,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_BROWSER_SESSION_READ,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_BROWSER_SESSION_CLOSE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_BROWSER_SESSION_EVENT_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_BROWSER_SESSION_ACTION_EXECUTE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SOUL_STYLE_PACK_INSTALL,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SOUL_STYLE_PACK_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SOUL_STYLE_PACK_STATUS_SET,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SOUL_STYLE_PACK_UNINSTALL,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_KNOWLEDGE_PACK_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_KNOWLEDGE_PACK_READ,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_KNOWLEDGE_SOURCE_IMPORT,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_KNOWLEDGE_PACK_COMPILE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_KNOWLEDGE_PACK_DEFAULT_SET,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_KNOWLEDGE_PACK_STATUS_UPDATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_KNOWLEDGE_CONTEXT_RESOLVE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_KNOWLEDGE_CONTEXT_RUN_VALIDATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_AUTOMATION_SCHEDULER_CONFIG_READ,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_AUTOMATION_SCHEDULER_CONFIG_UPDATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_AUTOMATION_SCHEDULER_STATUS,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_AUTOMATION_JOB_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_AUTOMATION_JOB_READ,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_AUTOMATION_JOB_CREATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_AUTOMATION_JOB_UPDATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_AUTOMATION_JOB_DELETE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_AUTOMATION_JOB_RUN_NOW,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_AUTOMATION_JOB_HEALTH,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_AUTOMATION_JOB_RUN_HISTORY,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_AUTOMATION_SCHEDULE_PREVIEW,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_AUTOMATION_SCHEDULE_VALIDATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SCHEDULED_TASK_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SCHEDULED_TASK_READ,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SCHEDULED_TASK_CREATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SCHEDULED_TASK_UPDATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SCHEDULED_TASK_DELETE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SCHEDULED_TASK_ENABLED_SET,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SCHEDULED_TASK_RUN_START,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SCHEDULED_TASK_RUN_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_SCHEDULED_TASK_SCHEDULE_PREVIEW,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MCP_SERVER_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MCP_SERVER_STATUS_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MCP_SERVER_CREATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MCP_SERVER_UPDATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MCP_SERVER_DELETE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MCP_SERVER_ENABLED_SET,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MCP_SERVER_IMPORT_FROM_APP,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MCP_SERVER_SYNC_ALL_TO_LIVE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MCP_SERVER_OAUTH_LOGIN,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MCP_SERVER_START,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MCP_SERVER_STOP,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MCP_TOOL_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MCP_TOOL_LIST_FOR_CONTEXT,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MCP_TOOL_SEARCH,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MCP_PROMPT_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MCP_PROMPT_GET,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MCP_RESOURCE_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MCP_RESOURCE_SUBSCRIBE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MCP_RESOURCE_UNSUBSCRIBE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_PROJECT_MEMORY_READ,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MEMORY_STORE_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MEMORY_STORE_READ,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MEMORY_STORE_SEARCH,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MEMORY_STORE_ADD_NOTE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MEMORY_STORE_CONSOLIDATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MEMORY_STORE_REVIEW_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MEMORY_STORE_REVIEW_RESOLVE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MEMORY_STORE_HEALTH,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MEMORY_STORE_INDEX_REBUILD,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_LOG_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_LOG_PERSISTED_TAIL,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_LOG_CLEAR,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_LOG_DIAGNOSTIC_HISTORY_CLEAR,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_DIAGNOSTICS_LOG_STORAGE_READ,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_DIAGNOSTICS_SUPPORT_BUNDLE_EXPORT,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_DIAGNOSTICS_SERVER_READ,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_DIAGNOSTICS_WINDOWS_STARTUP_READ,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_DIAGNOSTICS_TRACE_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_DIAGNOSTICS_TRACE_READ,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_DIAGNOSTICS_TRACE_EXPORT,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_USAGE_STATS_READ,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_USAGE_STATS_MODEL_RANKING_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_USAGE_STATS_DAILY_TRENDS_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MODEL_PREFERENCES_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MODEL_SYNC_STATE_READ,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MODEL_PROVIDER_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MODEL_PROVIDER_CATALOG_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MODEL_PROVIDER_READ,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MODEL_PROVIDER_CREATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MODEL_PROVIDER_UPDATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MODEL_PROVIDER_DELETE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MODEL_PROVIDER_SORT_ORDERS_UPDATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MODEL_PROVIDER_CONFIG_EXPORT,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MODEL_PROVIDER_CONFIG_IMPORT,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MODEL_PROVIDER_TEST_CONNECTION,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MODEL_PROVIDER_TEST_CHAT,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MODEL_PROVIDER_FETCH_MODELS,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MODEL_PROVIDER_KEY_CREATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MODEL_PROVIDER_KEY_UPDATE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MODEL_PROVIDER_KEY_DELETE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MODEL_PROVIDER_UI_STATE_READ,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MODEL_PROVIDER_UI_STATE_WRITE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MODEL_PROVIDER_ALIAS_READ,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_MODEL_PROVIDER_ALIAS_LIST,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_CONNECT_DEEP_LINK_RESOLVE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_CONNECT_OPEN_DEEP_LINK_RESOLVE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_CONNECT_RELAY_API_KEY_SAVE,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_CONNECT_CALLBACK_SEND,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_CONVERSATION_IMPORT_SOURCE_SCAN,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_CONVERSATION_IMPORT_THREAD_PREVIEW,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_CONVERSATION_IMPORT_THREAD_COMMIT,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_CONVERSATION_IMPORT_JOB_READ,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_AGENT_SESSION_ACTION_RESPOND,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WORKFLOW_READ,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WORKFLOW_CANCEL,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WORKFLOW_RETRY,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_WORKFLOW_RESPOND,
        kind: AppServerMethodKind::Request,
    },
    AppServerMethodSpec {
        method: METHOD_AGENT_SESSION_EVENT,
        kind: AppServerMethodKind::Notification,
    },
];

pub const APP_SERVER_REQUEST_SERIALIZATION_SCOPES: &[AppServerRequestSerializationScopeSpec] = &[
    AppServerRequestSerializationScopeSpec {
        method: METHOD_THREAD_READ,
        scope: AppServerRequestSerializationScope::Thread,
    },
    AppServerRequestSerializationScopeSpec {
        method: crate::protocol::v2::METHOD_THREAD_ARCHIVE,
        scope: AppServerRequestSerializationScope::Thread,
    },
    AppServerRequestSerializationScopeSpec {
        method: crate::protocol::v2::METHOD_THREAD_DELETE,
        scope: AppServerRequestSerializationScope::Thread,
    },
    AppServerRequestSerializationScopeSpec {
        method: crate::protocol::v2::METHOD_THREAD_UNSUBSCRIBE,
        scope: AppServerRequestSerializationScope::Thread,
    },
    AppServerRequestSerializationScopeSpec {
        method: crate::protocol::v2::METHOD_THREAD_INCREMENT_ELICITATION,
        scope: AppServerRequestSerializationScope::Thread,
    },
    AppServerRequestSerializationScopeSpec {
        method: crate::protocol::v2::METHOD_THREAD_DECREMENT_ELICITATION,
        scope: AppServerRequestSerializationScope::Thread,
    },
    AppServerRequestSerializationScopeSpec {
        method: crate::protocol::v2::METHOD_THREAD_UNARCHIVE,
        scope: AppServerRequestSerializationScope::Thread,
    },
    AppServerRequestSerializationScopeSpec {
        method: crate::protocol::v2::METHOD_THREAD_COMPACT_START,
        scope: AppServerRequestSerializationScope::Thread,
    },
    AppServerRequestSerializationScopeSpec {
        method: METHOD_THREAD_TURNS_LIST,
        scope: AppServerRequestSerializationScope::Thread,
    },
    AppServerRequestSerializationScopeSpec {
        method: METHOD_THREAD_ITEMS_LIST,
        scope: AppServerRequestSerializationScope::Thread,
    },
    AppServerRequestSerializationScopeSpec {
        method: crate::protocol::v2::METHOD_THREAD_INJECT_ITEMS,
        scope: AppServerRequestSerializationScope::Thread,
    },
    AppServerRequestSerializationScopeSpec {
        method: crate::protocol::v2::METHOD_THREAD_SETTINGS_UPDATE,
        scope: AppServerRequestSerializationScope::Thread,
    },
    AppServerRequestSerializationScopeSpec {
        method: crate::protocol::v2::METHOD_THREAD_MEMORY_MODE_SET,
        scope: AppServerRequestSerializationScope::Thread,
    },
    AppServerRequestSerializationScopeSpec {
        method: crate::protocol::v2::METHOD_THREAD_SHELL_COMMAND,
        scope: AppServerRequestSerializationScope::Thread,
    },
    AppServerRequestSerializationScopeSpec {
        method: crate::protocol::v2::METHOD_THREAD_APPROVE_GUARDIAN_DENIED_ACTION,
        scope: AppServerRequestSerializationScope::Thread,
    },
    AppServerRequestSerializationScopeSpec {
        method: crate::protocol::v2::METHOD_THREAD_BACKGROUND_TERMINALS_CLEAN,
        scope: AppServerRequestSerializationScope::Thread,
    },
    AppServerRequestSerializationScopeSpec {
        method: crate::protocol::v2::METHOD_THREAD_BACKGROUND_TERMINALS_LIST,
        scope: AppServerRequestSerializationScope::Thread,
    },
    AppServerRequestSerializationScopeSpec {
        method: crate::protocol::v2::METHOD_THREAD_BACKGROUND_TERMINALS_TERMINATE,
        scope: AppServerRequestSerializationScope::Thread,
    },
    AppServerRequestSerializationScopeSpec {
        method: crate::protocol::v2::METHOD_THREAD_GOAL_SET,
        scope: AppServerRequestSerializationScope::Thread,
    },
    AppServerRequestSerializationScopeSpec {
        method: crate::protocol::v2::METHOD_THREAD_GOAL_GET,
        scope: AppServerRequestSerializationScope::Thread,
    },
    AppServerRequestSerializationScopeSpec {
        method: crate::protocol::v2::METHOD_THREAD_GOAL_CLEAR,
        scope: AppServerRequestSerializationScope::Thread,
    },
    AppServerRequestSerializationScopeSpec {
        method: crate::protocol::v2::METHOD_ARTIFACT_WRITE,
        scope: AppServerRequestSerializationScope::Thread,
    },
    AppServerRequestSerializationScopeSpec {
        method: METHOD_TURN_START,
        scope: AppServerRequestSerializationScope::Thread,
    },
    AppServerRequestSerializationScopeSpec {
        method: METHOD_TURN_STEER,
        scope: AppServerRequestSerializationScope::Thread,
    },
    AppServerRequestSerializationScopeSpec {
        method: METHOD_TURN_INTERRUPT,
        scope: AppServerRequestSerializationScope::Thread,
    },
    AppServerRequestSerializationScopeSpec {
        method: METHOD_THREAD_RESUME,
        scope: AppServerRequestSerializationScope::Thread,
    },
    AppServerRequestSerializationScopeSpec {
        method: METHOD_MCP_SERVER_OAUTH_LOGIN,
        scope: AppServerRequestSerializationScope::McpOauth,
    },
    AppServerRequestSerializationScopeSpec {
        method: METHOD_MCP_RESOURCE_SUBSCRIBE,
        scope: AppServerRequestSerializationScope::McpResourceSubscription,
    },
    AppServerRequestSerializationScopeSpec {
        method: METHOD_MCP_RESOURCE_UNSUBSCRIBE,
        scope: AppServerRequestSerializationScope::McpResourceSubscription,
    },
    AppServerRequestSerializationScopeSpec {
        method: METHOD_BROWSER_SESSION_OPEN,
        scope: AppServerRequestSerializationScope::BrowserSession,
    },
    AppServerRequestSerializationScopeSpec {
        method: METHOD_BROWSER_SESSION_READ,
        scope: AppServerRequestSerializationScope::BrowserSession,
    },
    AppServerRequestSerializationScopeSpec {
        method: METHOD_BROWSER_SESSION_CLOSE,
        scope: AppServerRequestSerializationScope::BrowserSession,
    },
    AppServerRequestSerializationScopeSpec {
        method: METHOD_BROWSER_SESSION_ACTION_EXECUTE,
        scope: AppServerRequestSerializationScope::BrowserSession,
    },
];

pub const APP_SERVER_REQUEST_ACCESSES: &[AppServerRequestAccessSpec] = &[
    AppServerRequestAccessSpec {
        method: METHOD_BROWSER_SESSION_READ,
        access: AppServerRequestAccess::SharedRead,
    },
    AppServerRequestAccessSpec {
        method: METHOD_THREAD_READ,
        access: AppServerRequestAccess::SharedRead,
    },
    AppServerRequestAccessSpec {
        method: METHOD_THREAD_LIST,
        access: AppServerRequestAccess::SharedRead,
    },
    AppServerRequestAccessSpec {
        method: crate::protocol::v2::METHOD_THREAD_LOADED_LIST,
        access: AppServerRequestAccess::SharedRead,
    },
    AppServerRequestAccessSpec {
        method: METHOD_THREAD_TURNS_LIST,
        access: AppServerRequestAccess::SharedRead,
    },
    AppServerRequestAccessSpec {
        method: METHOD_THREAD_ITEMS_LIST,
        access: AppServerRequestAccess::SharedRead,
    },
    AppServerRequestAccessSpec {
        method: crate::protocol::v2::METHOD_THREAD_BACKGROUND_TERMINALS_LIST,
        access: AppServerRequestAccess::SharedRead,
    },
    AppServerRequestAccessSpec {
        method: crate::protocol::v2::METHOD_THREAD_GOAL_GET,
        access: AppServerRequestAccess::SharedRead,
    },
];

pub fn app_server_request_access(method: &str) -> AppServerRequestAccess {
    APP_SERVER_REQUEST_ACCESSES
        .iter()
        .find(|spec| spec.method == method)
        .map(|spec| spec.access)
        .unwrap_or(AppServerRequestAccess::Exclusive)
}

pub fn app_server_request_serialization_scope(
    method: &str,
) -> Option<AppServerRequestSerializationScope> {
    APP_SERVER_REQUEST_SERIALIZATION_SCOPES
        .iter()
        .find(|spec| spec.method == method)
        .map(|spec| spec.scope)
}

pub fn is_app_server_request_method(method: &str) -> bool {
    crate::protocol::v2::Method::parse(method).is_some()
        || AppServerRequestMethod::parse(method).is_some()
}

pub fn is_app_server_notification_method(method: &str) -> bool {
    crate::protocol::v2::NOTIFICATION_METHODS.contains(&method)
        || APP_SERVER_METHODS
            .iter()
            .any(|spec| spec.kind == AppServerMethodKind::Notification && spec.method == method)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn client_request_rejects_notification_method() {
        let request = JsonRpcRequest::new(RequestId::Integer(1), METHOD_AGENT_SESSION_EVENT, None);

        let error = AppServerClientRequest::try_from(request).expect_err("reject notification");

        assert!(error.contains("method not found: agentSession/event"));
    }

    #[test]
    fn client_request_round_trips_v0_request_method() {
        let request = AppServerClientRequest::try_from(JsonRpcRequest::new(
            RequestId::Integer(7),
            METHOD_CAPABILITY_LIST,
            Some(json!({})),
        ))
        .expect("typed request");

        assert_eq!(request.method(), AppServerRequestMethod::CapabilityList);
        let raw: JsonRpcRequest = request.into();
        assert_eq!(raw.method, METHOD_CAPABILITY_LIST);
        assert_eq!(raw.id, RequestId::Integer(7));
    }

    #[test]
    fn v0_client_request_rejects_v2_request_method() {
        let error = AppServerClientRequest::try_from(JsonRpcRequest::new(
            RequestId::Integer(7),
            METHOD_TURN_START,
            Some(json!({ "threadId": "thread-1", "input": [] })),
        ))
        .expect_err("v2 request must use the typed v2 envelope");

        assert_eq!(
            error,
            "v2 method requires protocol::v2::ClientRequest: turn/start"
        );
    }

    #[test]
    fn canonical_thread_read_methods_are_shared_reads() {
        for method in [
            METHOD_THREAD_READ,
            METHOD_THREAD_LIST,
            crate::protocol::v2::METHOD_THREAD_LOADED_LIST,
            METHOD_THREAD_TURNS_LIST,
            METHOD_THREAD_ITEMS_LIST,
        ] {
            assert!(is_app_server_request_method(method));
            assert_eq!(
                app_server_request_access(method),
                AppServerRequestAccess::SharedRead
            );
        }
        for method in [
            METHOD_THREAD_READ,
            METHOD_THREAD_TURNS_LIST,
            METHOD_THREAD_ITEMS_LIST,
        ] {
            assert_eq!(
                app_server_request_serialization_scope(method),
                Some(AppServerRequestSerializationScope::Thread)
            );
        }
        assert_eq!(
            app_server_request_serialization_scope(METHOD_THREAD_LIST),
            None
        );
    }

    #[test]
    fn client_notification_accepts_initialized_payload_shape() {
        let notification = crate::JsonRpcNotification::new(METHOD_INITIALIZED, Some(json!({})));

        let decoded = ClientNotification::try_from(notification).expect("decode initialized");

        assert_eq!(decoded, ClientNotification::Initialized);
    }

    #[test]
    fn notification_round_trips_agent_session_event_payload() {
        let event = AgentEvent {
            event_id: "evt_1".to_string(),
            sequence: 1,
            session_id: "sess_1".to_string(),
            thread_id: Some("thread_1".to_string()),
            turn_id: Some("turn_1".to_string()),
            event_type: "turn.completed".to_string(),
            timestamp: "2026-07-05T00:00:00Z".to_string(),
            payload: json!({}),
        };
        let notification = ServerNotification::AgentSessionEvent(AgentSessionEventParams {
            event: event.clone(),
        });

        let raw: crate::JsonRpcNotification = notification.into();
        assert_eq!(raw.method, METHOD_AGENT_SESSION_EVENT);
        let decoded = ServerNotification::try_from(raw).expect("decode");

        assert_eq!(
            decoded,
            ServerNotification::AgentSessionEvent(AgentSessionEventParams { event })
        );
    }
}
