use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{JsonRpcRequest, RequestId};

use super::AppServerRequestMethod;

macro_rules! app_server_client_request_definitions {
    ($($variant:ident => $wire:literal),* $(,)?) => {
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
        #[serde(tag = "method")]
        pub enum AppServerClientRequest {
            $(
                #[serde(rename = $wire)]
                $variant {
                    #[serde(rename = "id")]
                    id: RequestId,
                    #[serde(default)]
                    params: Value,
                },
            )*
        }

        impl AppServerClientRequest {
            pub fn id(&self) -> &RequestId {
                match self {
                    $(Self::$variant { id, .. } => id,)*
                }
            }

            pub fn method(&self) -> AppServerRequestMethod {
                match self {
                    $(Self::$variant { .. } => AppServerRequestMethod::$variant,)*
                }
            }

            pub fn into_jsonrpc_parts(self) -> (RequestId, AppServerRequestMethod, Option<Value>) {
                match self {
                    $(
                        Self::$variant { id, params } => {
                            (id, AppServerRequestMethod::$variant, Some(params))
                        }
                    )*
                }
            }
        }

        impl TryFrom<JsonRpcRequest> for AppServerClientRequest {
            type Error = String;

            fn try_from(request: JsonRpcRequest) -> Result<Self, Self::Error> {
                let JsonRpcRequest { id, method, params } = request;
                let params = params.unwrap_or_else(|| serde_json::json!({}));
                if crate::protocol::v2::Method::parse(&method).is_some() {
                    return Err(format!(
                        "v2 method requires protocol::v2::ClientRequest: {method}"
                    ));
                }
                match AppServerRequestMethod::parse(&method) {
                    $(
                        Some(AppServerRequestMethod::$variant) => {
                            Ok(Self::$variant { id, params })
                        }
                    )*
                    None => Err(format!("method not found: {method}")),
                }
            }
        }

        impl From<AppServerClientRequest> for JsonRpcRequest {
            fn from(request: AppServerClientRequest) -> Self {
                let (id, method, params) = request.into_jsonrpc_parts();
                JsonRpcRequest::new(id, method.as_str(), params)
            }
        }
    };
}

app_server_client_request_definitions! {
    Initialize => "initialize",
    CapabilityList => "capability/list",
    ArtifactRead => "artifact/read",
    ProjectGitStatus => "projectGit/status",
    ProjectGitDiff => "projectGit/diff",
    ProjectGitCommitsList => "projectGit/commits/list",
    ProjectGitBranchCheckout => "projectGit/branch/checkout",
    ProjectGitBranchCreate => "projectGit/branch/create",
    ProjectGitWorktreeCreate => "projectGit/worktree/create",
    AgentSessionHandoffBundleExport => "agentSession/handoffBundle/export",
    AgentSessionReplayCaseExport => "agentSession/replayCase/export",
    AgentSessionAnalysisHandoffExport => "agentSession/analysisHandoff/export",
    AgentSessionReviewDecisionTemplateExport => "agentSession/reviewDecisionTemplate/export",
    AgentSessionReviewDecisionSave => "agentSession/reviewDecision/save",
    AgentSessionFileCheckpointList => "agentSession/fileCheckpoint/list",
    AgentSessionFileCheckpointGet => "agentSession/fileCheckpoint/get",
    AgentSessionFileCheckpointDiff => "agentSession/fileCheckpoint/diff",
    AgentSessionFileCheckpointRestore => "agentSession/fileCheckpoint/restore",
    AgentSessionToolInventoryRead => "agentSession/toolInventory/read",
    SessionFileGetOrCreate => "sessionFile/getOrCreate",
    SessionFileUpdateMeta => "sessionFile/updateMeta",
    SessionFileSave => "sessionFile/save",
    SessionFileRead => "sessionFile/read",
    SessionFileResolvePath => "sessionFile/resolvePath",
    SessionFileDelete => "sessionFile/delete",
    SessionFileList => "sessionFile/list",
    WorkspaceList => "workspace/list",
    WorkspaceRead => "workspace/read",
    WorkspaceUpdate => "workspace/update",
    WorkspaceDelete => "workspace/delete",
    WorkspaceEnsure => "workspace/ensure",
    WorkspaceByPathRead => "workspace/byPath/read",
    WorkspaceDefaultRead => "workspace/default/read",
    WorkspaceDefaultEnsure => "workspace/default/ensure",
    WorkspaceProjectsRootRead => "workspace/projectsRoot/read",
    WorkspaceProjectPathResolve => "workspace/projectPath/resolve",
    WorkspaceEnsureReady => "workspace/ensureReady",
    SkillRead => "skill/read",
    SkillManagementList => "skillManagement/list",
    SkillManagementInstall => "skillManagement/install",
    SkillManagementUninstall => "skillManagement/uninstall",
    SkillRepositoryList => "skillRepository/list",
    SkillRepositorySave => "skillRepository/save",
    SkillRepositoryDelete => "skillRepository/delete",
    SkillCacheRefresh => "skillCache/refresh",
    SkillInstalledDirectoriesList => "skillInstalledDirectories/list",
    SkillLocalInspect => "skillLocal/inspect",
    SkillLocalDetailInspect => "skillLocal/detail/inspect",
    SkillLocalScaffoldCreate => "skillLocal/scaffold/create",
    SkillLocalImport => "skillLocal/import",
    SkillLocalRename => "skillLocal/rename",
    SkillRemoteInspect => "skillRemote/inspect",
    SkillPackageLocalInspect => "skillPackage/local/inspect",
    SkillPackageLocalInstall => "skillPackage/local/install",
    SkillPackageLocalReplace => "skillPackage/local/replace",
    SkillPackageExport => "skillPackage/export",
    SkillMarketplaceInstall => "skillMarketplace/install",
    SkillPackageDownloadInstall => "skillPackage/download/install",
    GatewayChannelStart => "gatewayChannel/start",
    GatewayChannelStop => "gatewayChannel/stop",
    GatewayChannelStatus => "gatewayChannel/status",
    TelegramChannelProbe => "telegramChannel/probe",
    FeishuChannelProbe => "feishuChannel/probe",
    DiscordChannelProbe => "discordChannel/probe",
    WechatChannelProbe => "wechatChannel/probe",
    WechatChannelLoginStart => "wechatChannel/login/start",
    WechatChannelLoginWait => "wechatChannel/login/wait",
    WechatChannelAccountList => "wechatChannel/accounts/list",
    WechatChannelAccountRemove => "wechatChannel/account/remove",
    WechatChannelRuntimeModelSet => "wechatChannel/runtimeModel/set",
    GatewayTunnelProbe => "gatewayTunnel/probe",
    GatewayTunnelCloudflaredDetect => "gatewayTunnel/cloudflared/detect",
    GatewayTunnelCloudflaredInstall => "gatewayTunnel/cloudflared/install",
    GatewayTunnelCreate => "gatewayTunnel/create",
    GatewayTunnelStart => "gatewayTunnel/start",
    GatewayTunnelStop => "gatewayTunnel/stop",
    GatewayTunnelRestart => "gatewayTunnel/restart",
    GatewayTunnelStatus => "gatewayTunnel/status",
    GatewayTunnelSyncWebhookUrl => "gatewayTunnel/syncWebhookUrl",
    MediaTaskArtifactImageCreate => "mediaTaskArtifact/image/create",
    MediaTaskArtifactAudioCreate => "mediaTaskArtifact/audio/create",
    MediaTaskArtifactTranscriptionCreate => "mediaTaskArtifact/transcription/create",
    MediaTaskArtifactVideoCreate => "mediaTaskArtifact/video/create",
    MediaTaskArtifactImageComplete => "mediaTaskArtifact/image/complete",
    MediaTaskArtifactAudioComplete => "mediaTaskArtifact/audio/complete",
    MediaTaskArtifactGet => "mediaTaskArtifact/get",
    MediaTaskArtifactList => "mediaTaskArtifact/list",
    MediaTaskArtifactCancel => "mediaTaskArtifact/cancel",
    GalleryMaterialGet => "galleryMaterial/get",
    GalleryMaterialMetadataCreate => "galleryMaterialMetadata/create",
    GalleryMaterialMetadataGet => "galleryMaterialMetadata/get",
    GalleryMaterialMetadataUpdate => "galleryMaterialMetadata/update",
    GalleryMaterialMetadataDelete => "galleryMaterialMetadata/delete",
    GalleryMaterialListByImageCategory => "galleryMaterial/listByImageCategory",
    GalleryMaterialListByLayoutCategory => "galleryMaterial/listByLayoutCategory",
    GalleryMaterialListByMood => "galleryMaterial/listByMood",
    ProjectMaterialList => "projectMaterial/list",
    ProjectMaterialGet => "projectMaterial/get",
    ProjectMaterialCount => "projectMaterial/count",
    ProjectMaterialUpload => "projectMaterial/upload",
    ProjectMaterialImportFromUrl => "projectMaterial/importFromUrl",
    ProjectMaterialUpdate => "projectMaterial/update",
    ProjectMaterialDelete => "projectMaterial/delete",
    ProjectMaterialContent => "projectMaterial/content",
    VoiceAsrCredentialList => "voiceAsrCredential/list",
    VoiceAsrCredentialCreate => "voiceAsrCredential/create",
    VoiceAsrCredentialUpdate => "voiceAsrCredential/update",
    VoiceAsrCredentialDelete => "voiceAsrCredential/delete",
    VoiceAsrCredentialDefaultSet => "voiceAsrCredential/default/set",
    VoiceAsrCredentialTest => "voiceAsrCredential/test",
    VoiceInstructionList => "voiceInstruction/list",
    VoiceInstructionSave => "voiceInstruction/save",
    VoiceInstructionDelete => "voiceInstruction/delete",
    VoiceModelDefaultSet => "voiceModel/default/set",
    VoiceModelTestTranscribeFile => "voiceModel/testTranscribeFile",
    VoiceTranscriptionTranscribeAudio => "voiceTranscription/transcribeAudio",
    VoiceTranscriptionPolishText => "voiceTranscription/polishText",
    WorkspaceSkillBindingsList => "workspaceSkillBindings/list",
    WorkspaceRegisteredSkillsList => "workspaceRegisteredSkills/list",
    WorkspaceRightSurfaceRequest => "workspaceRightSurface/request",
    WorkspaceRightSurfacePendingList => "workspaceRightSurface/pending/list",
    WorkspaceRightSurfacePendingConsume => "workspaceRightSurface/pending/consume",
    WorkspaceRightSurfacePendingDismiss => "workspaceRightSurface/pending/dismiss",
    BrowserSessionTargetList => "browserSession/target/list",
    BrowserSessionOpen => "browserSession/open",
    BrowserSessionRead => "browserSession/read",
    BrowserSessionClose => "browserSession/close",
    BrowserSessionEventList => "browserSession/event/list",
    BrowserSessionActionExecute => "browserSession/action/execute",
    SoulStylePackInstall => "soulStylePack/install",
    SoulStylePackList => "soulStylePack/list",
    SoulStylePackStatusSet => "soulStylePack/status/set",
    SoulStylePackUninstall => "soulStylePack/uninstall",
    KnowledgePackList => "knowledgePack/list",
    KnowledgePackRead => "knowledgePack/read",
    KnowledgeSourceImport => "knowledgePack/source/import",
    KnowledgePackCompile => "knowledgePack/compile",
    KnowledgePackDefaultSet => "knowledgePack/default/set",
    KnowledgePackStatusUpdate => "knowledgePack/status/update",
    KnowledgeContextResolve => "knowledgeContext/resolve",
    KnowledgeContextRunValidate => "knowledgeContextRun/validate",
    AutomationSchedulerConfigRead => "automationScheduler/config/read",
    AutomationSchedulerConfigUpdate => "automationScheduler/config/update",
    AutomationSchedulerStatus => "automationScheduler/status",
    AutomationJobList => "automationJob/list",
    AutomationJobRead => "automationJob/read",
    AutomationJobCreate => "automationJob/create",
    AutomationJobUpdate => "automationJob/update",
    AutomationJobDelete => "automationJob/delete",
    AutomationJobRunNow => "automationJob/runNow",
    AutomationJobHealth => "automationJob/health",
    AutomationJobRunHistory => "automationJob/runHistory",
    AutomationSchedulePreview => "automationSchedule/preview",
    AutomationScheduleValidate => "automationSchedule/validate",
    ScheduledTaskList => "scheduledTask/list",
    ScheduledTaskRead => "scheduledTask/read",
    ScheduledTaskCreate => "scheduledTask/create",
    ScheduledTaskUpdate => "scheduledTask/update",
    ScheduledTaskDelete => "scheduledTask/delete",
    ScheduledTaskEnabledSet => "scheduledTask/enabled/set",
    ScheduledTaskRunStart => "scheduledTask/run/start",
    ScheduledTaskRunList => "scheduledTask/run/list",
    ScheduledTaskSchedulePreview => "scheduledTask/schedule/preview",
    McpServerList => "mcpServer/list",
    McpServerStatusList => "mcpServerStatus/list",
    McpServerCreate => "mcpServer/create",
    McpServerUpdate => "mcpServer/update",
    McpServerDelete => "mcpServer/delete",
    McpServerEnabledSet => "mcpServer/enabled/set",
    McpServerImportFromApp => "mcpServer/importFromApp",
    McpServerSyncAllToLive => "mcpServer/syncAllToLive",
    McpServerOauthLogin => "mcpServer/oauth/login",
    McpServerStart => "mcpServer/start",
    McpServerStop => "mcpServer/stop",
    McpToolList => "mcpTool/list",
    McpToolListForContext => "mcpTool/listForContext",
    McpToolSearch => "mcpTool/search",
    McpPromptList => "mcpPrompt/list",
    McpPromptGet => "mcpPrompt/get",
    McpResourceList => "mcpResource/list",
    McpResourceSubscribe => "mcpResource/subscribe",
    McpResourceUnsubscribe => "mcpResource/unsubscribe",
    ProjectMemoryRead => "projectMemory/read",
    MemoryStoreList => "memoryStore/list",
    MemoryStoreRead => "memoryStore/read",
    MemoryStoreSearch => "memoryStore/search",
    MemoryStoreAddNote => "memoryStore/addNote",
    MemoryStoreConsolidate => "memoryStore/consolidate",
    MemoryStoreReviewList => "memoryStore/review/list",
    MemoryStoreReviewResolve => "memoryStore/review/resolve",
    MemoryStoreHealth => "memoryStore/health",
    MemoryStoreIndexRebuild => "memoryStore/index/rebuild",
    LogList => "log/list",
    LogPersistedTail => "log/persistedTail",
    LogClear => "log/clear",
    LogDiagnosticHistoryClear => "log/diagnosticHistory/clear",
    DiagnosticsLogStorageRead => "diagnostics/logStorage/read",
    DiagnosticsSupportBundleExport => "diagnostics/supportBundle/export",
    DiagnosticsServerRead => "diagnostics/server/read",
    DiagnosticsWindowsStartupRead => "diagnostics/windowsStartup/read",
    DiagnosticsTraceList => "diagnostics/trace/list",
    DiagnosticsTraceRead => "diagnostics/trace/read",
    DiagnosticsTraceExport => "diagnostics/trace/export",
    UsageStatsRead => "usageStats/read",
    UsageStatsModelRankingList => "usageStats/modelRanking/list",
    UsageStatsDailyTrendsList => "usageStats/dailyTrends/list",
    ModelPreferencesList => "modelPreferences/list",
    ModelSyncStateRead => "modelSyncState/read",
    ModelProviderList => "modelProvider/list",
    ModelProviderCatalogList => "modelProvider/catalog/list",
    ModelProviderRead => "modelProvider/read",
    ModelProviderCreate => "modelProvider/create",
    ModelProviderUpdate => "modelProvider/update",
    ModelProviderDelete => "modelProvider/delete",
    ModelProviderSortOrdersUpdate => "modelProvider/sortOrders/update",
    ModelProviderConfigExport => "modelProviderConfig/export",
    ModelProviderConfigImport => "modelProviderConfig/import",
    ModelProviderTestConnection => "modelProvider/testConnection",
    ModelProviderTestChat => "modelProvider/testChat",
    ModelProviderFetchModels => "modelProvider/fetchModels",
    ModelProviderKeyCreate => "modelProviderKey/create",
    ModelProviderKeyUpdate => "modelProviderKey/update",
    ModelProviderKeyDelete => "modelProviderKey/delete",
    ModelProviderUiStateRead => "modelProviderUiState/read",
    ModelProviderUiStateWrite => "modelProviderUiState/write",
    ModelProviderAliasRead => "modelProviderAlias/read",
    ModelProviderAliasList => "modelProviderAlias/list",
    ConnectDeepLinkResolve => "connectDeepLink/resolve",
    ConnectOpenDeepLinkResolve => "connectOpenDeepLink/resolve",
    ConnectRelayApiKeySave => "connectRelayApiKey/save",
    ConnectCallbackSend => "connectCallback/send",
    ConversationImportSourceScan => "conversationImport/source/scan",
    ConversationImportThreadPreview => "conversationImport/thread/preview",
    ConversationImportThreadCommit => "conversationImport/thread/commit",
    ConversationImportJobRead => "conversationImport/job/read",
    AgentSessionActionRespond => "agentSession/action/respond",
    WorkflowRead => "workflow/read",
    WorkflowCancel => "workflow/cancel",
    WorkflowRetry => "workflow/retry",
    WorkflowRespond => "workflow/respond",
}
