use crate::gateway_tunnel;
use crate::plugin_packages;
use crate::AutomationManagementAppDataSource;
use crate::AutomationOverviewAppDataSource;
use crate::ConnectAppDataSource;
use crate::DiagnosticsAppDataSource;
use crate::GatewayAppDataSource;
use crate::KnowledgeAppDataSource;
use crate::LocalMemoryBackend;
use crate::McpAppDataSource;
use crate::MediaAppDataSource;
use crate::MemoryAppDataSource;
use crate::MemoryBackend;
use crate::ModelProviderAppDataSource;
use crate::PluginDataSource;
use crate::RightSurfaceAppDataSource;
use crate::RolloutSummaryWriteParams;
use crate::RuntimeCoreError;
use crate::SessionAppDataSource;
use crate::SkillAppDataSource;
use crate::UsageStatsAppDataSource;
use crate::VoiceAppDataSource;
use crate::WorkspaceAppDataSource;
use crate::WorkspaceSkillBindingAppDataSource;
mod automation;
mod channels;
mod connect;
mod diagnostics;
mod gallery_materials;
mod impls;
mod knowledge;
mod mcp;
mod media_tasks;
mod model_projection;
mod model_providers;
pub(super) mod plugin_catalog;
mod plugins;
mod project_materials;
mod right_surface;
mod session_files;
mod skills;
mod usage_stats;
mod voice_asr_credentials;
mod voice_instructions;
mod voice_text_processing;
mod workspaces;
use app_server_protocol::protocol::v2::{
    McpServerResourceReadParams, McpServerResourceReadResponse,
};
use app_server_protocol::AutomationJobCreateParams;
use app_server_protocol::AutomationJobDeleteResponse;
use app_server_protocol::AutomationJobHealthParams;
use app_server_protocol::AutomationJobHealthResponse;
use app_server_protocol::AutomationJobIdParams;
use app_server_protocol::AutomationJobListResponse;
use app_server_protocol::AutomationJobReadResponse;
use app_server_protocol::AutomationJobRunHistoryParams;
use app_server_protocol::AutomationJobRunHistoryResponse;
use app_server_protocol::AutomationJobUpdateParams;
use app_server_protocol::AutomationJobWriteResponse;
use app_server_protocol::AutomationScheduleParams;
use app_server_protocol::AutomationSchedulePreviewResponse;
use app_server_protocol::AutomationScheduleValidateResponse;
use app_server_protocol::AutomationSchedulerConfigReadResponse;
use app_server_protocol::AutomationSchedulerConfigUpdateParams;
use app_server_protocol::AutomationSchedulerConfigUpdateResponse;
use app_server_protocol::AutomationSchedulerStatusResponse;
use app_server_protocol::ChannelProbeParams;
use app_server_protocol::ChannelProbeResponse;
use app_server_protocol::ConnectCallbackSendParams;
use app_server_protocol::ConnectCallbackSendResponse;
use app_server_protocol::ConnectDeepLinkResolveParams;
use app_server_protocol::ConnectDeepLinkResolveResponse;
use app_server_protocol::ConnectOpenDeepLinkResolveParams;
use app_server_protocol::ConnectOpenDeepLinkResolveResponse;
use app_server_protocol::ConnectRelayApiKeySaveParams;
use app_server_protocol::ConnectRelayApiKeySaveResponse;
use app_server_protocol::GalleryMaterialDeleteResponse;
use app_server_protocol::GalleryMaterialFilterParams;
use app_server_protocol::GalleryMaterialListResponse;
use app_server_protocol::GalleryMaterialLookupParams;
use app_server_protocol::GalleryMaterialMetadataCreateParams;
use app_server_protocol::GalleryMaterialMetadataResponse;
use app_server_protocol::GalleryMaterialMetadataUpdateParams;
use app_server_protocol::GalleryMaterialResponse;
use app_server_protocol::GatewayChannelStartParams;
use app_server_protocol::GatewayChannelStatusParams;
use app_server_protocol::GatewayChannelStatusResponse;
use app_server_protocol::GatewayChannelStopParams;
use app_server_protocol::GatewayTunnelCloudflaredDetectResponse;
use app_server_protocol::GatewayTunnelCloudflaredInstallParams;
use app_server_protocol::GatewayTunnelCloudflaredInstallResponse;
use app_server_protocol::GatewayTunnelCreateParams;
use app_server_protocol::GatewayTunnelCreateResponse;
use app_server_protocol::GatewayTunnelProbeResponse;
use app_server_protocol::GatewayTunnelStatusResponse;
use app_server_protocol::GatewayTunnelSyncWebhookUrlParams;
use app_server_protocol::GatewayTunnelSyncWebhookUrlResponse;
use app_server_protocol::KnowledgeCompilePackResponse;
use app_server_protocol::KnowledgeContextResolutionResponse;
use app_server_protocol::KnowledgeImportSourceParams;
use app_server_protocol::KnowledgeImportSourceResponse;
use app_server_protocol::KnowledgeListPacksParams;
use app_server_protocol::KnowledgeListPacksResponse;
use app_server_protocol::KnowledgeReadPackParams;
use app_server_protocol::KnowledgeReadPackResponse;
use app_server_protocol::KnowledgeResolveContextParams;
use app_server_protocol::KnowledgeSetDefaultPackParams;
use app_server_protocol::KnowledgeSetDefaultPackResponse;
use app_server_protocol::KnowledgeUpdatePackStatusParams;
use app_server_protocol::KnowledgeUpdatePackStatusResponse;
use app_server_protocol::KnowledgeValidateContextRunParams;
use app_server_protocol::KnowledgeValidateContextRunResponse;
use app_server_protocol::LogClearResponse;
use app_server_protocol::LogListResponse;
use app_server_protocol::LogPersistedTailParams;
use app_server_protocol::LogPersistedTailResponse;
use app_server_protocol::LogStorageDiagnosticsResponse;
use app_server_protocol::McpPromptGetParams;
use app_server_protocol::McpPromptGetResponse;
use app_server_protocol::McpPromptListResponse;
use app_server_protocol::McpResourceListResponse;
use app_server_protocol::McpResourceSubscribeParams;
use app_server_protocol::McpResourceSubscriptionResponse;
use app_server_protocol::McpResourceUnsubscribeParams;
use app_server_protocol::McpServerCreateParams;
use app_server_protocol::McpServerDeleteParams;
use app_server_protocol::McpServerEnabledSetParams;
use app_server_protocol::McpServerImportFromAppParams;
use app_server_protocol::McpServerImportFromAppResponse;
use app_server_protocol::McpServerLifecycleResponse;
use app_server_protocol::McpServerListResponse;
use app_server_protocol::McpServerStartParams;
use app_server_protocol::McpServerStatusListResponse;
use app_server_protocol::McpServerStopParams;
use app_server_protocol::McpServerUpdateParams;
use app_server_protocol::McpToolListForContextParams;
use app_server_protocol::McpToolListResponse;
use app_server_protocol::McpToolSearchParams;
use app_server_protocol::MediaTaskArtifactAudioCompleteParams;
use app_server_protocol::MediaTaskArtifactAudioCreateParams;
use app_server_protocol::MediaTaskArtifactImageCompleteParams;
use app_server_protocol::MediaTaskArtifactImageCreateParams;
use app_server_protocol::MediaTaskArtifactListParams;
use app_server_protocol::MediaTaskArtifactListResponse;
use app_server_protocol::MediaTaskArtifactLookupParams;
use app_server_protocol::MediaTaskArtifactResponse;
use app_server_protocol::MediaTaskArtifactVideoCreateParams;
use app_server_protocol::MemoryStoreAddNoteParams;
use app_server_protocol::MemoryStoreAddNoteResponse;
use app_server_protocol::MemoryStoreConsolidateParams;
use app_server_protocol::MemoryStoreConsolidateResponse;
use app_server_protocol::MemoryStoreHealthResponse;
use app_server_protocol::MemoryStoreIndexRebuildResponse;
use app_server_protocol::MemoryStoreListParams;
use app_server_protocol::MemoryStoreListResponse;
use app_server_protocol::MemoryStoreReadParams;
use app_server_protocol::MemoryStoreReadResponse;
use app_server_protocol::MemoryStoreResetParams;
use app_server_protocol::MemoryStoreResetResponse;
use app_server_protocol::MemoryStoreReviewListParams;
use app_server_protocol::MemoryStoreReviewListResponse;
use app_server_protocol::MemoryStoreReviewResolveParams;
use app_server_protocol::MemoryStoreReviewResolveResponse;
use app_server_protocol::MemoryStoreRootParams;
use app_server_protocol::MemoryStoreSearchParams;
use app_server_protocol::MemoryStoreSearchResponse;
use app_server_protocol::ModelPreferencesListResponse;
use app_server_protocol::ModelProviderAliasListResponse;
use app_server_protocol::ModelProviderAliasReadParams;
use app_server_protocol::ModelProviderAliasReadResponse;
use app_server_protocol::ModelProviderCatalogListResponse;
use app_server_protocol::ModelProviderConfigExportParams;
use app_server_protocol::ModelProviderConfigExportResponse;
use app_server_protocol::ModelProviderConfigImportParams;
use app_server_protocol::ModelProviderConfigImportResponse;
use app_server_protocol::ModelProviderCreateParams;
use app_server_protocol::ModelProviderDeleteParams;
use app_server_protocol::ModelProviderDeleteResponse;
use app_server_protocol::ModelProviderFetchModelsParams;
use app_server_protocol::ModelProviderFetchModelsResponse;
use app_server_protocol::ModelProviderKeyCreateParams;
use app_server_protocol::ModelProviderKeyDeleteParams;
use app_server_protocol::ModelProviderKeyDeleteResponse;
use app_server_protocol::ModelProviderKeyUpdateParams;
use app_server_protocol::ModelProviderKeyWriteResponse;
use app_server_protocol::ModelProviderListResponse;
use app_server_protocol::ModelProviderMutationResponse;
use app_server_protocol::ModelProviderReadParams;
use app_server_protocol::ModelProviderReadResponse;
use app_server_protocol::ModelProviderSortOrdersUpdateParams;
use app_server_protocol::ModelProviderTestChatParams;
use app_server_protocol::ModelProviderTestChatResponse;
use app_server_protocol::ModelProviderTestConnectionParams;
use app_server_protocol::ModelProviderTestConnectionResponse;
use app_server_protocol::ModelProviderUiStateReadParams;
use app_server_protocol::ModelProviderUiStateReadResponse;
use app_server_protocol::ModelProviderUiStateWriteParams;
use app_server_protocol::ModelProviderUpdateParams;
use app_server_protocol::ModelProviderWriteResponse;
use app_server_protocol::ModelSyncStateReadResponse;
use app_server_protocol::PluginFetchCloudPackageParams;
use app_server_protocol::PluginInstalledDisabledSetParams;
use app_server_protocol::PluginInstalledListResponse;
use app_server_protocol::PluginInstalledSaveParams;
use app_server_protocol::PluginLocalPackageExportParams;
use app_server_protocol::PluginLocalPackageExportResponse;
use app_server_protocol::PluginLocalPackageInspectParams;
use app_server_protocol::PluginLocalPackageInspectResponse;
use app_server_protocol::PluginPackageCacheEntry;
use app_server_protocol::PluginUninstallParams;
use app_server_protocol::PluginUninstallRehearsalParams;
use app_server_protocol::PluginUninstallRehearsalResponse;
use app_server_protocol::PluginUninstallResponse;
use app_server_protocol::ProjectMaterialContentResponse;
use app_server_protocol::ProjectMaterialCountResponse;
use app_server_protocol::ProjectMaterialDeleteResponse;
use app_server_protocol::ProjectMaterialImportFromUrlParams;
use app_server_protocol::ProjectMaterialListParams;
use app_server_protocol::ProjectMaterialListResponse;
use app_server_protocol::ProjectMaterialLookupParams;
use app_server_protocol::ProjectMaterialResponse;
use app_server_protocol::ProjectMaterialUpdateParams;
use app_server_protocol::ProjectMaterialUploadParams;
use app_server_protocol::ProjectMemoryReadParams;
use app_server_protocol::ProjectMemoryReadResponse;
use app_server_protocol::SessionFileEntryResponse;
use app_server_protocol::SessionFileGetOrCreateParams;
use app_server_protocol::SessionFileIdParams;
use app_server_protocol::SessionFileListResponse;
use app_server_protocol::SessionFileMetaResponse;
use app_server_protocol::SessionFileMutationResponse;
use app_server_protocol::SessionFileReadResponse;
use app_server_protocol::SessionFileResolvePathResponse;
use app_server_protocol::SessionFileSaveParams;
use app_server_protocol::SessionFileUpdateMetaParams;
use app_server_protocol::SkillDownloadInstallParams;
use app_server_protocol::SkillDownloadInstallResponse;
use app_server_protocol::SkillInstalledDirectoriesListResponse;
use app_server_protocol::SkillLocalDetailInspectParams;
use app_server_protocol::SkillLocalDetailInspectResponse;
use app_server_protocol::SkillLocalImportParams;
use app_server_protocol::SkillLocalImportResponse;
use app_server_protocol::SkillLocalInspectParams;
use app_server_protocol::SkillLocalInspectResponse;
use app_server_protocol::SkillLocalRenameParams;
use app_server_protocol::SkillLocalRenameResponse;
use app_server_protocol::SkillManagementInstallParams;
use app_server_protocol::SkillManagementListParams;
use app_server_protocol::SkillManagementListResponse;
use app_server_protocol::SkillManagementUninstallParams;
use app_server_protocol::SkillManagementWriteResponse;
use app_server_protocol::SkillMarketplaceInstallParams;
use app_server_protocol::SkillMarketplaceInstallResponse;
use app_server_protocol::SkillPackageExportParams;
use app_server_protocol::SkillPackageExportResponse;
use app_server_protocol::SkillPackageLocalInspectParams;
use app_server_protocol::SkillPackageLocalInspectResponse;
use app_server_protocol::SkillPackageLocalInstallParams;
use app_server_protocol::SkillPackageLocalInstallResponse;
use app_server_protocol::SkillPackageLocalReplaceParams;
use app_server_protocol::SkillPackageLocalReplaceResponse;
use app_server_protocol::SkillReadParams;
use app_server_protocol::SkillReadResponse;
use app_server_protocol::SkillRemoteInspectParams;
use app_server_protocol::SkillRemoteInspectResponse;
use app_server_protocol::SkillRepositoryDeleteParams;
use app_server_protocol::SkillRepositoryListResponse;
use app_server_protocol::SkillRepositorySaveParams;
use app_server_protocol::SkillScaffoldCreateParams;
use app_server_protocol::SkillScaffoldCreateResponse;
use app_server_protocol::SupportBundleExportResponse;
use app_server_protocol::UsageStatsDailyTrendsListResponse;
use app_server_protocol::UsageStatsModelRankingListResponse;
use app_server_protocol::UsageStatsRangeParams;
use app_server_protocol::UsageStatsReadResponse;
use app_server_protocol::VoiceAsrCredentialCreateParams;
use app_server_protocol::VoiceAsrCredentialIdParams;
use app_server_protocol::VoiceAsrCredentialListResponse;
use app_server_protocol::VoiceAsrCredentialMutationResponse;
use app_server_protocol::VoiceAsrCredentialTestResponse;
use app_server_protocol::VoiceAsrCredentialUpdateParams;
use app_server_protocol::VoiceAsrCredentialWriteResponse;
use app_server_protocol::VoiceInstructionIdParams;
use app_server_protocol::VoiceInstructionListResponse;
use app_server_protocol::VoiceInstructionMutationResponse;
use app_server_protocol::VoiceInstructionSaveParams;
use app_server_protocol::VoiceModelDefaultSetParams;
use app_server_protocol::VoiceModelDefaultSetResponse;
use app_server_protocol::VoiceModelTestTranscribeFileParams;
use app_server_protocol::VoiceModelTestTranscribeFileResponse;
use app_server_protocol::VoiceTranscriptionPolishTextParams;
use app_server_protocol::VoiceTranscriptionPolishTextResponse;
use app_server_protocol::VoiceTranscriptionTranscribeAudioParams;
use app_server_protocol::VoiceTranscriptionTranscribeAudioResponse;
use app_server_protocol::WechatChannelAccountListResponse;
use app_server_protocol::WechatChannelAccountRemoveParams;
use app_server_protocol::WechatChannelAccountRemoveResponse;
use app_server_protocol::WechatLoginStartParams;
use app_server_protocol::WechatLoginStartResponse;
use app_server_protocol::WechatLoginWaitParams;
use app_server_protocol::WechatLoginWaitResponse;
use app_server_protocol::WechatRuntimeModelSetParams;
use app_server_protocol::WechatRuntimeModelSetResponse;
use app_server_protocol::WindowsStartupDiagnosticsResponse;
use app_server_protocol::WorkspaceDeleteParams;
use app_server_protocol::WorkspaceDeleteResponse;
use app_server_protocol::WorkspaceEnsureParams;
use app_server_protocol::WorkspaceEnsureProjectParams;
use app_server_protocol::WorkspaceEnsureProjectResponse;
use app_server_protocol::WorkspaceEnsureReadyResponse;
use app_server_protocol::WorkspaceListResponse;
use app_server_protocol::WorkspacePathReadParams;
use app_server_protocol::WorkspaceProjectPathResolveParams;
use app_server_protocol::WorkspaceProjectPathResolveResponse;
use app_server_protocol::WorkspaceProjectsRootReadResponse;
use app_server_protocol::WorkspaceReadParams;
use app_server_protocol::WorkspaceReadResponse;
use app_server_protocol::WorkspaceRegisteredSkillsListParams;
use app_server_protocol::WorkspaceRegisteredSkillsListResponse;
use app_server_protocol::WorkspaceRightSurfacePendingListParams;
use app_server_protocol::WorkspaceRightSurfacePendingRequest;
use app_server_protocol::WorkspaceSkillBindingsListParams;
use app_server_protocol::WorkspaceSkillBindingsListResponse;
use app_server_protocol::WorkspaceUpdateParams;
use app_server_protocol::WorkspaceUpdateResponse;
use lime_core::config::load_config;
use lime_core::database::DbConnection;
use lime_core::logger;
use lime_core::session_files::SessionFileStorage;
use lime_gateway::discord::DiscordGatewayState;
use lime_gateway::feishu::FeishuGatewayState;
use lime_gateway::telegram::TelegramGatewayState;
use lime_gateway::tunnel::GatewayTunnelState;
use lime_gateway::wechat::WechatGatewayState;
use lime_gateway::wechat::WechatLoginState;
use lime_mcp::McpClientManager;
use lime_mcp::McpManagerState;
use lime_services::api_key_provider_service::ApiKeyProviderService;
use lime_services::model_registry_service::ModelRegistryService;
use lime_services::skill_service::SkillService;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;

pub struct LocalAppDataSource {
    db: DbConnection,
    /// 由组合根注入的机器数据根。诊断与产品资产 owner 只能消费它，不得自行解析平台根。
    app_data_root: std::path::PathBuf,
    /// 实际打开的 Product DB 路径。诊断只报告它，不重新解析。
    product_db_path: std::path::PathBuf,
    plugin_data_root: std::path::PathBuf,
    session_files_root: std::path::PathBuf,
    connect_registry_cache_path: std::path::PathBuf,
    logs: std::sync::Arc<tokio::sync::RwLock<lime_core::logger::LogStore>>,
    api_key_provider_service: ApiKeyProviderService,
    model_registry_service: ModelRegistryService,
    mcp_manager: McpManagerState,
    mcp_elicitation_router: lime_mcp::ElicitationRequestRouter,
    telegram_gateway_state: TelegramGatewayState,
    feishu_gateway_state: FeishuGatewayState,
    discord_gateway_state: DiscordGatewayState,
    wechat_gateway_state: WechatGatewayState,
    gateway_tunnel_state: GatewayTunnelState,
    wechat_login_state: WechatLoginState,
    memory_backend: Arc<dyn MemoryBackend>,
    sidecar_store: Option<Arc<crate::runtime::SidecarStore>>,
}

impl LocalAppDataSource {
    /// AppDataRoot 与 AgentRoot 都必须由组合根显式注入。
    /// 这里不解析平台默认根，也不从 AgentRoot 反推 AppDataRoot。
    pub async fn initialize_with_roots(
        db: DbConnection,
        app_data_root: impl Into<std::path::PathBuf>,
        data_root: impl Into<std::path::PathBuf>,
    ) -> Result<Self, String> {
        let app_data_root = app_data_root.into();
        let data_root = data_root.into();
        let product_db_path = crate::runtime::product_db_path_for_agent_root(&data_root);
        let plugin_data_root = plugin_packages::plugin_data_dir_for_agent_root(&data_root);
        let session_files_root = SessionFileStorage::base_dir_for_agent_root(&data_root);
        let connect_registry_cache_path =
            connect::connect_registry_cache_path_for_agent_root(&data_root);
        let config = load_config().map_err(|error| error.to_string())?;
        let logs = std::sync::Arc::new(tokio::sync::RwLock::new(
            logger::create_log_store_from_config(&data_root, &config.logging)?,
        ));
        let api_key_provider_service = ApiKeyProviderService::new();
        let model_registry_service = ModelRegistryService::new(db.clone());
        model_registry_service.initialize().await?;
        let gateway_tunnel_state = GatewayTunnelState::default();
        gateway_tunnel::spawn_gateway_tunnel_daemon(gateway_tunnel_state.clone(), logs.clone());
        let memory_backend = Arc::new(LocalMemoryBackend::new(data_root));
        let mcp_oauth_store_root =
            lime_mcp::oauth_store::oauth_store_root_for_app_data_root(&app_data_root);
        Ok(Self {
            db,
            app_data_root,
            product_db_path,
            plugin_data_root,
            session_files_root,
            connect_registry_cache_path,
            logs,
            api_key_provider_service,
            model_registry_service,
            mcp_manager: Arc::new(TokioMutex::new(
                McpClientManager::new(None).with_oauth_store_root(mcp_oauth_store_root),
            )),
            mcp_elicitation_router: lime_mcp::ElicitationRequestRouter::default(),
            telegram_gateway_state: TelegramGatewayState::default(),
            feishu_gateway_state: FeishuGatewayState::default(),
            discord_gateway_state: DiscordGatewayState::default(),
            wechat_gateway_state: WechatGatewayState::default(),
            gateway_tunnel_state,
            wechat_login_state: WechatLoginState::default(),
            memory_backend,
            sidecar_store: None,
        })
    }

    pub fn with_sidecar_store(mut self, sidecar_store: Arc<crate::runtime::SidecarStore>) -> Self {
        self.sidecar_store = Some(sidecar_store);
        self
    }

    pub(crate) fn app_data_root(&self) -> &std::path::Path {
        &self.app_data_root
    }

    pub(crate) fn product_db_path(&self) -> &std::path::Path {
        &self.product_db_path
    }

    pub fn mcp_elicitation_router(&self) -> lime_mcp::ElicitationRequestRouter {
        self.mcp_elicitation_router.clone()
    }
}

fn values_from_serializable_vec<T: serde::Serialize>(
    values: Vec<T>,
) -> Result<Vec<Value>, RuntimeCoreError> {
    values
        .into_iter()
        .map(|value| serde_json::to_value(value).map_err(data_error))
        .collect()
}

fn value_from_serializable<T: serde::Serialize>(value: T) -> Result<Value, RuntimeCoreError> {
    serde_json::to_value(value).map_err(data_error)
}

fn data_error(error: impl std::fmt::Display) -> RuntimeCoreError {
    RuntimeCoreError::Backend(error.to_string())
}

#[cfg(test)]
mod tests;
