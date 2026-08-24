import type * as protocol from "../../../packages/app-server-client/src/protocol";

export type AppServerHandleJsonLinesRequest = {
  lines: string[];
};

export type AppServerHandleJsonLinesResult = {
  lines: string[];
};

export type AppServerDrainEventsRequest = {
  includeRecent?: boolean;
  limit?: number;
};

export type AppServerDrainEventsResult = {
  lines: string[];
};

export type AppServerRequestId = protocol.RequestId;
export type AppServerJsonValue = protocol.JsonValue;
export type AppServerJsonRpcRequest = protocol.JsonRpcRequest;
export type AppServerJsonRpcNotification = protocol.JsonRpcNotification;
export type AppServerConfigWarningNotification =
  protocol.ConfigWarningNotification;
export type AppServerConfigWarningJsonRpcNotification = {
  method: typeof protocol.METHOD_CONFIG_WARNING;
  params: AppServerConfigWarningNotification;
};
export type AppServerJsonRpcResponse<T = unknown> = protocol.JsonRpcResponse<T>;
export type AppServerJsonRpcError = protocol.JsonRpcError;
export type AppServerJsonRpcErrorResponse = protocol.JsonRpcErrorResponse;
export type AppServerJsonRpcMessage<T = unknown> = protocol.JsonRpcMessage<T>;
export type AppServerClientInfo = protocol.ClientInfo;
export type AppServerClientCapabilities = protocol.ClientCapabilities;
export type AppServerInitializeParams = protocol.InitializeParams;
export type AppServerInitializeResponse = protocol.InitializeResponse;
export type AppServerBusinessObjectRef = protocol.BusinessObjectRef;
export type AppServerCapabilityListParams = protocol.CapabilityListParams;
export type AppServerCapabilityDescriptor = protocol.CapabilityDescriptor;
export type AppServerCapabilityListResponse = protocol.CapabilityListResponse;
export type AppServerRuntimeCapabilityManifest =
  protocol.RuntimeCapabilityManifest;
export type AppServerArtifactReadParams = protocol.ArtifactReadParams;
export type AppServerArtifactContentStatus = protocol.ArtifactContentStatus;
export type AppServerArtifactSummary = protocol.ArtifactSummary;
export type AppServerArtifactReadResponse = protocol.ArtifactReadResponse;
export type AppServerArtifactWriteParams = protocol.ArtifactWriteParams;
export type AppServerArtifactWriteResponse = protocol.ArtifactWriteResponse;
export type AppServerFuzzyFileSearchParams = protocol.FuzzyFileSearchParams;
export type AppServerFuzzyFileSearchResponse = protocol.FuzzyFileSearchResponse;
export type AppServerFuzzyFileSearchResult = protocol.FuzzyFileSearchResult;
export type AppServerFuzzyFileSearchMatchType =
  protocol.FuzzyFileSearchMatchType;
export type AppServerFsReadFileParams = protocol.FsReadFileParams;
export type AppServerFsReadFileResponse = protocol.FsReadFileResponse;
export type AppServerFsWriteFileParams = protocol.FsWriteFileParams;
export type AppServerFsWriteFileResponse = protocol.FsWriteFileResponse;
export type AppServerFsCreateDirectoryParams = protocol.FsCreateDirectoryParams;
export type AppServerFsCreateDirectoryResponse =
  protocol.FsCreateDirectoryResponse;
export type AppServerFsGetMetadataParams = protocol.FsGetMetadataParams;
export type AppServerFsGetMetadataResponse = protocol.FsGetMetadataResponse;
export type AppServerFsReadDirectoryParams = protocol.FsReadDirectoryParams;
export type AppServerFsReadDirectoryResponse = protocol.FsReadDirectoryResponse;
export type AppServerFsReadDirectoryEntry = protocol.FsReadDirectoryEntry;
export type AppServerFsRemoveParams = protocol.FsRemoveParams;
export type AppServerFsRemoveResponse = protocol.FsRemoveResponse;
export type AppServerFsCopyParams = protocol.FsCopyParams;
export type AppServerFsCopyResponse = protocol.FsCopyResponse;
export type AppServerFsWatchParams = protocol.FsWatchParams;
export type AppServerFsWatchResponse = protocol.FsWatchResponse;
export type AppServerFsUnwatchParams = protocol.FsUnwatchParams;
export type AppServerFsUnwatchResponse = protocol.FsUnwatchResponse;
export type AppServerFsChangedNotification = protocol.FsChangedNotification;
export type AppServerProjectGitStatusParams = protocol.ProjectGitStatusParams;
export type AppServerProjectGitDiffBase = protocol.ProjectGitDiffBase;
export type AppServerProjectGitDiffParams = protocol.ProjectGitDiffParams;
export type AppServerProjectGitDiffResponse = protocol.ProjectGitDiffResponse;
export type AppServerProjectGitCommitListParams =
  protocol.ProjectGitCommitListParams;
export type AppServerProjectGitCommitListResponse =
  protocol.ProjectGitCommitListResponse;
export type AppServerProjectGitBranchCheckoutParams =
  protocol.ProjectGitBranchCheckoutParams;
export type AppServerProjectGitBranchCheckoutResponse =
  protocol.ProjectGitBranchCheckoutResponse;
export type AppServerProjectGitBranchCreateParams =
  protocol.ProjectGitBranchCreateParams;
export type AppServerProjectGitBranchCreateResponse =
  protocol.ProjectGitBranchCreateResponse;
export type AppServerProjectGitStatusResponse =
  protocol.ProjectGitStatusResponse;
export type AppServerProjectGitWorktreeCreateParams =
  protocol.ProjectGitWorktreeCreateParams;
export type AppServerProjectGitWorktreeCreateResponse =
  protocol.ProjectGitWorktreeCreateResponse;
export type AppServerAgentSessionHandoffBundleExportParams =
  protocol.AgentSessionHandoffBundleExportParams;
export type AppServerAgentSessionHandoffBundleExportResponse =
  protocol.AgentSessionHandoffBundleExportResponse;
export type AppServerAgentSessionHandoffArtifact =
  protocol.AgentSessionHandoffArtifact;
export type AppServerAgentSessionReplayCaseExportParams =
  protocol.AgentSessionReplayCaseExportParams;
export type AppServerAgentSessionReplayCaseExportResponse =
  protocol.AgentSessionReplayCaseExportResponse;
export type AppServerAgentSessionAnalysisHandoffExportParams =
  protocol.AgentSessionAnalysisHandoffExportParams;
export type AppServerAgentSessionAnalysisHandoffExportResponse =
  protocol.AgentSessionAnalysisHandoffExportResponse;
export type AppServerAgentSessionReviewDecisionTemplateExportParams =
  protocol.AgentSessionReviewDecisionTemplateExportParams;
export type AppServerAgentSessionReviewDecisionTemplateExportResponse =
  protocol.AgentSessionReviewDecisionTemplateExportResponse;
export type AppServerAgentSessionReviewDecisionSaveParams =
  protocol.AgentSessionReviewDecisionSaveParams;
export type AppServerAgentSessionReviewDecision =
  protocol.AgentSessionReviewDecision;
export type AppServerThreadStartParams = protocol.ThreadStartParams;
export type AppServerThreadForkParams = protocol.ThreadForkParams;
export type AppServerThreadForkResponse = protocol.ThreadForkResponse;
export type AppServerAgentSessionListParams = protocol.AgentSessionListParams;
export type AppServerAgentSessionListResponse =
  protocol.AgentSessionListResponse;
export type AppServerAgentSessionReadParams = protocol.AgentSessionReadParams;
export type AppServerMediaReadParams = protocol.MediaReadParams;
export type AppServerMediaReadResponse = protocol.MediaReadResponse;
export type AppServerAppsListParams = protocol.AppsListParams;
export type AppServerAppsListResponse = protocol.AppsListResponse;
export type AppServerAppsInstalledParams = protocol.AppsInstalledParams;
export type AppServerAppsInstalledResponse = protocol.AppsInstalledResponse;
export type AppServerAppsReadParams = protocol.AppsReadParams;
export type AppServerAppsReadResponse = protocol.AppsReadResponse;
export type AppServerAppInfo = protocol.AppInfo;
export type AppServerInstalledApp = protocol.InstalledApp;
export type AppServerConnectorMetadata = protocol.ConnectorMetadata;
export type AppServerAppListUpdatedNotification =
  protocol.AppListUpdatedNotification;
export type AppServerSkillsListParams = protocol.SkillsListParams;
export type AppServerSkillsListResponse = protocol.SkillsListResponse;
export type AppServerSkillsExtraRootsSetParams =
  protocol.SkillsExtraRootsSetParams;
export type AppServerSkillsExtraRootsSetResponse =
  protocol.SkillsExtraRootsSetResponse;
export type AppServerSkillsConfigWriteParams = protocol.SkillsConfigWriteParams;
export type AppServerSkillsConfigWriteResponse =
  protocol.SkillsConfigWriteResponse;
export type AppServerPluginCatalogListParams = protocol.PluginCatalogListParams;
export type AppServerPluginCatalogListResponse =
  protocol.PluginCatalogListResponse;
export type AppServerPluginCatalogSummary = protocol.PluginCatalogSummary;
export type AppServerPluginSearchParams = protocol.PluginSearchParams;
export type AppServerPluginSearchResponse = protocol.PluginSearchResponse;
export type AppServerPluginSearchResult = protocol.PluginSearchResult;
export type AppServerPluginSummary = protocol.PluginSummary;
export type AppServerPluginCatalogDetail = protocol.PluginCatalogDetail;
export type AppServerPluginCatalogCapability = protocol.PluginCatalogCapability;
export type AppServerPluginCatalogHook = protocol.PluginCatalogHook;
export type AppServerPluginCatalogUiResource = protocol.PluginCatalogUiResource;
export type AppServerPluginCatalogReadParams = protocol.PluginCatalogReadParams;
export type AppServerPluginCatalogReadResponse =
  protocol.PluginCatalogReadResponse;
export type AppServerPluginCatalogInstallParams =
  protocol.PluginCatalogInstallParams;
export type AppServerPluginCatalogInstallResponse =
  protocol.PluginCatalogInstallResponse;
export type AppServerPluginCatalogUninstallParams =
  protocol.PluginCatalogUninstallParams;
export type AppServerPluginCatalogUninstallResponse =
  protocol.PluginCatalogUninstallResponse;
export type AppServerPluginCatalogInstalledParams =
  protocol.PluginCatalogInstalledParams;
export type AppServerPluginCatalogEnabledSetParams =
  protocol.PluginCatalogEnabledSetParams;
export type AppServerPluginCatalogEnabledSetResponse =
  protocol.PluginCatalogEnabledSetResponse;
export type AppServerAgentSessionToolInventoryReadParams =
  protocol.AgentSessionToolInventoryReadParams;
export type AppServerAgentInput = protocol.AgentInput;
export type AppServerAgentAttachment = protocol.AgentAttachment;
export type AppServerRuntimeOptions = protocol.RuntimeOptions;
export type AppServerAgentSessionTurnStartParams =
  protocol.AgentSessionTurnStartParams;
export type AppServerReviewStartParams = protocol.ReviewStartParams;
export type AppServerReviewStartResponse = protocol.ReviewStartResponse;
export type AppServerReviewDelivery = protocol.ReviewDelivery;
export type AppServerReviewTarget = protocol.ReviewTarget;
export type AppServerAgentSessionTurnCancelParams =
  protocol.AgentSessionTurnCancelParams;
export type AppServerTurnSteerParams = protocol.TurnSteerParams;
export type AppServerTurnSteerResponse = protocol.TurnSteerResponse;
export type AppServerAgentSessionActionType = protocol.AgentSessionActionType;
export type AppServerAgentSessionActionScope = protocol.AgentSessionActionScope;
export type AppServerAgentSessionActionRespondParams =
  protocol.AgentSessionActionRespondParams;
export type AppServerAgentSessionStatus = protocol.AgentSessionStatus;
export type AppServerAgentSession = protocol.AgentSession;
export type AppServerAgentTurnStatus = protocol.AgentTurnStatus;
export type AppServerAgentTurn = protocol.AgentTurn;
export type AppServerAgentEvent = protocol.AgentEvent;
export type AppServerThreadStartResponse = protocol.ThreadStartResponse;
export type AppServerAgentSessionReadResponse =
  protocol.AgentSessionReadResponse;
export type AppServerThread = protocol.Thread;
export type AppServerThreadListParams = protocol.ThreadListParams;
export type AppServerThreadListResponse = protocol.ThreadListResponse;
export type AppServerThreadSection = protocol.ThreadSection;
export type AppServerThreadSectionListParams = protocol.ThreadSectionListParams;
export type AppServerThreadSectionListResponse =
  protocol.ThreadSectionListResponse;
export type AppServerThreadSectionCreateParams =
  protocol.ThreadSectionCreateParams;
export type AppServerThreadSectionCreateResponse =
  protocol.ThreadSectionCreateResponse;
export type AppServerThreadSectionUpdateParams =
  protocol.ThreadSectionUpdateParams;
export type AppServerThreadSectionUpdateResponse =
  protocol.ThreadSectionUpdateResponse;
export type AppServerThreadSectionDeleteParams =
  protocol.ThreadSectionDeleteParams;
export type AppServerThreadSectionDeleteResponse =
  protocol.ThreadSectionDeleteResponse;
export type AppServerThreadSectionMoveParams = protocol.ThreadSectionMoveParams;
export type AppServerThreadSectionMoveResponse =
  protocol.ThreadSectionMoveResponse;
export type AppServerThreadReadParams = protocol.ThreadReadParams;
export type AppServerThreadReadResponse = protocol.ThreadReadResponse;
export type AppServerThreadSetNameParams = protocol.ThreadSetNameParams;
export type AppServerThreadSetNameResponse = protocol.ThreadSetNameResponse;
export type AppServerThreadSettingsUpdateParams =
  protocol.ThreadSettingsUpdateParams;
export type AppServerThreadSettingsUpdateResponse =
  protocol.ThreadSettingsUpdateResponse;
export type AppServerThreadMemoryModeSetParams =
  protocol.ThreadMemoryModeSetParams;
export type AppServerThreadMemoryModeSetResponse =
  protocol.ThreadMemoryModeSetResponse;
export type AppServerThreadShellCommandParams =
  protocol.ThreadShellCommandParams;
export type AppServerThreadShellCommandResponse =
  protocol.ThreadShellCommandResponse;
export type AppServerCommandExecParams = protocol.CommandExecParams;
export type AppServerCommandExecResponse = protocol.CommandExecResponse;
export type AppServerCommandExecWriteParams = protocol.CommandExecWriteParams;
export type AppServerCommandExecWriteResponse =
  protocol.CommandExecWriteResponse;
export type AppServerCommandExecResizeParams = protocol.CommandExecResizeParams;
export type AppServerCommandExecResizeResponse =
  protocol.CommandExecResizeResponse;
export type AppServerCommandExecTerminateParams =
  protocol.CommandExecTerminateParams;
export type AppServerCommandExecTerminateResponse =
  protocol.CommandExecTerminateResponse;
export type AppServerCommandExecOutputDeltaNotification =
  protocol.CommandExecOutputDeltaNotification;
export type AppServerThreadBackgroundTerminal =
  protocol.ThreadBackgroundTerminal;
export type AppServerThreadBackgroundTerminalsListParams =
  protocol.ThreadBackgroundTerminalsListParams;
export type AppServerThreadBackgroundTerminalsListResponse =
  protocol.ThreadBackgroundTerminalsListResponse;
export type AppServerThreadBackgroundTerminalsTerminateParams =
  protocol.ThreadBackgroundTerminalsTerminateParams;
export type AppServerThreadBackgroundTerminalsTerminateResponse =
  protocol.ThreadBackgroundTerminalsTerminateResponse;
export type AppServerThreadArchiveParams = protocol.ThreadArchiveParams;
export type AppServerThreadArchiveResponse = protocol.ThreadArchiveResponse;
export type AppServerThreadUnarchiveParams = protocol.ThreadUnarchiveParams;
export type AppServerThreadUnarchiveResponse = protocol.ThreadUnarchiveResponse;
export type AppServerAgentSessionToolInventoryReadResponse =
  protocol.AgentSessionToolInventoryReadResponse;
export type AppServerThreadDeleteParams = protocol.ThreadDeleteParams;
export type AppServerThreadDeleteResponse = protocol.ThreadDeleteResponse;
export type AppServerThreadCompactStartParams =
  protocol.ThreadCompactStartParams;
export type AppServerThreadCompactStartResponse =
  protocol.ThreadCompactStartResponse;
export type AppServerThreadResumeParams = protocol.ThreadResumeParams;
export type AppServerThreadResumeResponse = protocol.ThreadResumeResponse;
export type AppServerAgentSessionFileCheckpointListParams =
  protocol.AgentSessionFileCheckpointListParams;
export type AppServerAgentSessionFileCheckpointGetParams =
  protocol.AgentSessionFileCheckpointGetParams;
export type AppServerAgentSessionFileCheckpointDiffParams =
  protocol.AgentSessionFileCheckpointDiffParams;
export type AppServerAgentSessionFileCheckpointRestoreParams =
  protocol.AgentSessionFileCheckpointRestoreParams;
export type AppServerAgentSessionFileCheckpointSummary =
  protocol.AgentSessionFileCheckpointSummary;
export type AppServerAgentSessionFileCheckpointThreadSummary =
  protocol.AgentSessionFileCheckpointThreadSummary;
export type AppServerAgentSessionFileCheckpointListResponse =
  protocol.AgentSessionFileCheckpointListResponse;
export type AppServerAgentSessionFileCheckpointDetail =
  protocol.AgentSessionFileCheckpointDetail;
export type AppServerAgentSessionFileCheckpointDiffResponse =
  protocol.AgentSessionFileCheckpointDiffResponse;
export type AppServerAgentSessionFileCheckpointRestoreResponse =
  protocol.AgentSessionFileCheckpointRestoreResponse;
export type AppServerSessionFileIdParams = protocol.SessionFileIdParams;
export type AppServerSessionFileGetOrCreateParams =
  protocol.SessionFileGetOrCreateParams;
export type AppServerSessionFileUpdateMetaParams =
  protocol.SessionFileUpdateMetaParams;
export type AppServerSessionFileSaveParams = protocol.SessionFileSaveParams;
export type AppServerSessionFileMeta = protocol.SessionFileMeta;
export type AppServerSessionFileEntry = protocol.SessionFileEntry;
export type AppServerSessionFileMetaResponse = protocol.SessionFileMetaResponse;
export type AppServerSessionFileEntryResponse =
  protocol.SessionFileEntryResponse;
export type AppServerSessionFileReadResponse = protocol.SessionFileReadResponse;
export type AppServerSessionFileResolvePathResponse =
  protocol.SessionFileResolvePathResponse;
export type AppServerSessionFileListResponse = protocol.SessionFileListResponse;
export type AppServerSessionFileMutationResponse =
  protocol.SessionFileMutationResponse;
export type AppServerAgentSessionTurnStartResponse =
  protocol.AgentSessionTurnStartResponse;
export type AppServerAgentSessionTurnCancelResponse =
  protocol.AgentSessionTurnCancelResponse;
export type AppServerAgentSessionActionRespondResponse =
  protocol.AgentSessionActionRespondResponse;
export type AppServerWorkflowReadParams = protocol.WorkflowReadParams;
export type AppServerWorkflowReadResponse = protocol.WorkflowReadResponse;
export type AppServerWorkflowCancelParams = protocol.WorkflowCancelParams;
export type AppServerWorkflowCancelResponse = protocol.WorkflowCancelResponse;
export type AppServerWorkflowRetryParams = protocol.WorkflowRetryParams;
export type AppServerWorkflowRetryResponse = protocol.WorkflowRetryResponse;
export type AppServerWorkflowRespondParams = protocol.WorkflowRespondParams;
export type AppServerWorkflowRespondResponse = protocol.WorkflowRespondResponse;
export type AppServerGatewayChannelStatusParams =
  protocol.GatewayChannelStatusParams;
export type AppServerGatewayChannelStartParams =
  protocol.GatewayChannelStartParams;
export type AppServerGatewayChannelStopParams =
  protocol.GatewayChannelStopParams;
export type AppServerGatewayChannelStatusResponse =
  protocol.GatewayChannelStatusResponse;
export type AppServerGatewayTunnelCloudflaredDetectResponse =
  protocol.GatewayTunnelCloudflaredDetectResponse;
export type AppServerGatewayTunnelCloudflaredInstallParams =
  protocol.GatewayTunnelCloudflaredInstallParams;
export type AppServerGatewayTunnelCloudflaredInstallResponse =
  protocol.GatewayTunnelCloudflaredInstallResponse;
export type AppServerGatewayTunnelCreateParams =
  protocol.GatewayTunnelCreateParams;
export type AppServerGatewayTunnelCreateResponse =
  protocol.GatewayTunnelCreateResponse;
export type AppServerGatewayTunnelProbeResponse =
  protocol.GatewayTunnelProbeResponse;
export type AppServerGatewayTunnelStatusResponse =
  protocol.GatewayTunnelStatusResponse;
export type AppServerGatewayTunnelSyncWebhookUrlParams =
  protocol.GatewayTunnelSyncWebhookUrlParams;
export type AppServerGatewayTunnelSyncWebhookUrlResponse =
  protocol.GatewayTunnelSyncWebhookUrlResponse;
export type AppServerChannelProbeParams = protocol.ChannelProbeParams;
export type AppServerChannelProbeResponse = protocol.ChannelProbeResponse;
export type AppServerWechatConfiguredAccount = protocol.WechatConfiguredAccount;
export type AppServerWechatChannelAccountListResponse =
  protocol.WechatChannelAccountListResponse;
export type AppServerWechatLoginStartParams = protocol.WechatLoginStartParams;
export type AppServerWechatLoginStartResponse =
  protocol.WechatLoginStartResponse;
export type AppServerWechatLoginWaitParams = protocol.WechatLoginWaitParams;
export type AppServerWechatLoginWaitResponse = protocol.WechatLoginWaitResponse;
export type AppServerWechatChannelAccountRemoveParams =
  protocol.WechatChannelAccountRemoveParams;
export type AppServerWechatChannelAccountRemoveResponse =
  protocol.WechatChannelAccountRemoveResponse;
export type AppServerWechatRuntimeModelSetParams =
  protocol.WechatRuntimeModelSetParams;
export type AppServerWechatRuntimeModelSetResponse =
  protocol.WechatRuntimeModelSetResponse;
export type AppServerLogEntry = protocol.LogEntry;
export type AppServerLogListResponse = protocol.LogListResponse;
export type AppServerLogPersistedTailParams = protocol.LogPersistedTailParams;
export type AppServerLogPersistedTailResponse =
  protocol.LogPersistedTailResponse;
export type AppServerLogClearResponse = protocol.LogClearResponse;
export type AppServerLogStorageDiagnosticsResponse =
  protocol.LogStorageDiagnosticsResponse;
export type AppServerSupportBundleExportParams =
  protocol.SupportBundleExportParams;
export type AppServerSupportBundleExportResponse =
  protocol.SupportBundleExportResponse;
export type AppServerServerDiagnosticsResponse =
  protocol.ServerDiagnosticsResponse;
export type AppServerWindowsStartupDiagnosticsResponse =
  protocol.WindowsStartupDiagnosticsResponse;
export type AppServerDiagnosticsTraceListParams =
  protocol.DiagnosticsTraceListParams;
export type AppServerDiagnosticsTraceReadParams =
  protocol.DiagnosticsTraceReadParams;
export type AppServerDiagnosticsTraceExportParams =
  protocol.DiagnosticsTraceExportParams;
export type AppServerDiagnosticsTraceListResponse =
  protocol.DiagnosticsTraceListResponse;
export type AppServerDiagnosticsTraceReadResponse =
  protocol.DiagnosticsTraceReadResponse;
export type AppServerDiagnosticsTraceExportResponse =
  protocol.DiagnosticsTraceExportResponse;
export type AppServerDiagnosticsTraceSummary = protocol.DiagnosticsTraceSummary;
export type AppServerDiagnosticsTraceEvent = protocol.DiagnosticsTraceEvent;
export type AppServerDiagnosticsTraceRedactionPolicy =
  protocol.DiagnosticsTraceRedactionPolicy;
export type AppServerMediaTaskArtifactImageCreateParams =
  protocol.MediaTaskArtifactImageCreateParams;
export type AppServerMediaTaskArtifactAudioCreateParams =
  protocol.MediaTaskArtifactAudioCreateParams;
export type AppServerMediaTaskArtifactTranscriptionCreateParams =
  protocol.MediaTaskArtifactTranscriptionCreateParams;
export type AppServerMediaTaskArtifactVideoCreateParams =
  protocol.MediaTaskArtifactVideoCreateParams;
export type AppServerMediaTaskArtifactImageCompleteParams =
  protocol.MediaTaskArtifactImageCompleteParams;
export type AppServerMediaTaskArtifactAudioCompleteParams =
  protocol.MediaTaskArtifactAudioCompleteParams;
export type AppServerMediaTaskArtifactLookupParams =
  protocol.MediaTaskArtifactLookupParams;
export type AppServerMediaTaskArtifactListParams =
  protocol.MediaTaskArtifactListParams;
export type AppServerMediaTaskArtifactResponse =
  protocol.MediaTaskArtifactResponse;
export type AppServerMediaTaskArtifactListResponse =
  protocol.MediaTaskArtifactListResponse;
export type AppServerGalleryMaterialLookupParams =
  protocol.GalleryMaterialLookupParams;
export type AppServerGalleryMaterialMetadataCreateParams =
  protocol.GalleryMaterialMetadataCreateParams;
export type AppServerGalleryMaterialMetadataUpdateParams =
  protocol.GalleryMaterialMetadataUpdateParams;
export type AppServerGalleryMaterialFilterParams =
  protocol.GalleryMaterialFilterParams;
export type AppServerGalleryMaterialResponse = protocol.GalleryMaterialResponse;
export type AppServerGalleryMaterialMetadataResponse =
  protocol.GalleryMaterialMetadataResponse;
export type AppServerGalleryMaterialListResponse =
  protocol.GalleryMaterialListResponse;
export type AppServerGalleryMaterialDeleteResponse =
  protocol.GalleryMaterialDeleteResponse;
export type AppServerProjectMaterial = protocol.ProjectMaterial;
export type AppServerProjectMaterialListParams =
  protocol.ProjectMaterialListParams;
export type AppServerProjectMaterialLookupParams =
  protocol.ProjectMaterialLookupParams;
export type AppServerProjectMaterialUploadParams =
  protocol.ProjectMaterialUploadParams;
export type AppServerProjectMaterialImportFromUrlParams =
  protocol.ProjectMaterialImportFromUrlParams;
export type AppServerProjectMaterialUpdateParams =
  protocol.ProjectMaterialUpdateParams;
export type AppServerProjectMaterialListResponse =
  protocol.ProjectMaterialListResponse;
export type AppServerProjectMaterialResponse = protocol.ProjectMaterialResponse;
export type AppServerProjectMaterialCountResponse =
  protocol.ProjectMaterialCountResponse;
export type AppServerProjectMaterialContentResponse =
  protocol.ProjectMaterialContentResponse;
export type AppServerProjectMaterialDeleteResponse =
  protocol.ProjectMaterialDeleteResponse;
export type AppServerVoiceAsrProviderType = protocol.VoiceAsrProviderType;
export type AppServerVoiceAsrCredential = protocol.VoiceAsrCredential;
export type AppServerVoiceAsrCredentialCreateParams =
  protocol.VoiceAsrCredentialCreateParams;
export type AppServerVoiceAsrCredentialUpdateParams =
  protocol.VoiceAsrCredentialUpdateParams;
export type AppServerVoiceAsrCredentialIdParams =
  protocol.VoiceAsrCredentialIdParams;
export type AppServerVoiceAsrCredentialListResponse =
  protocol.VoiceAsrCredentialListResponse;
export type AppServerVoiceAsrCredentialWriteResponse =
  protocol.VoiceAsrCredentialWriteResponse;
export type AppServerVoiceAsrCredentialMutationResponse =
  protocol.VoiceAsrCredentialMutationResponse;
export type AppServerVoiceAsrCredentialTestResponse =
  protocol.VoiceAsrCredentialTestResponse;
export type AppServerVoiceInstruction = protocol.VoiceInstruction;
export type AppServerVoiceInstructionSaveParams =
  protocol.VoiceInstructionSaveParams;
export type AppServerVoiceInstructionIdParams =
  protocol.VoiceInstructionIdParams;
export type AppServerVoiceInstructionListResponse =
  protocol.VoiceInstructionListResponse;
export type AppServerVoiceInstructionMutationResponse =
  protocol.VoiceInstructionMutationResponse;
export type AppServerVoiceModelDefaultSetParams =
  protocol.VoiceModelDefaultSetParams;
export type AppServerVoiceModelDefaultSetResponse =
  protocol.VoiceModelDefaultSetResponse;
export type AppServerVoiceModelTestTranscribeFileParams =
  protocol.VoiceModelTestTranscribeFileParams;
export type AppServerVoiceModelTestTranscribeFileResponse =
  protocol.VoiceModelTestTranscribeFileResponse;
export type AppServerVoiceTranscriptionTranscribeAudioParams =
  protocol.VoiceTranscriptionTranscribeAudioParams;
export type AppServerVoiceTranscriptionTranscribeAudioResponse =
  protocol.VoiceTranscriptionTranscribeAudioResponse;
export type AppServerVoiceTranscriptionPolishTextParams =
  protocol.VoiceTranscriptionPolishTextParams;
export type AppServerVoiceTranscriptionPolishTextResponse =
  protocol.VoiceTranscriptionPolishTextResponse;
export type AppServerWorkspaceRightSurfaceRequestParams =
  protocol.WorkspaceRightSurfaceRequestParams;
export type AppServerWorkspaceRightSurfaceRequestResponse =
  protocol.WorkspaceRightSurfaceRequestResponse;
export type AppServerWorkspaceRightSurfacePendingListParams =
  protocol.WorkspaceRightSurfacePendingListParams;
export type AppServerWorkspaceRightSurfacePendingListResponse =
  protocol.WorkspaceRightSurfacePendingListResponse;
export type AppServerWorkspaceRightSurfacePendingConsumeParams =
  protocol.WorkspaceRightSurfacePendingConsumeParams;
export type AppServerWorkspaceRightSurfacePendingConsumeResponse =
  protocol.WorkspaceRightSurfacePendingConsumeResponse;
export type AppServerWorkspaceRightSurfacePendingDismissParams =
  protocol.WorkspaceRightSurfacePendingDismissParams;
export type AppServerWorkspaceRightSurfacePendingDismissResponse =
  protocol.WorkspaceRightSurfacePendingDismissResponse;
export type AppServerWorkspaceRightSurfacePendingChangedParams =
  protocol.WorkspaceRightSurfacePendingChangedParams;
export type AppServerWorkspaceRightSurfacePendingChangedNotification =
  protocol.WorkspaceRightSurfacePendingChangedNotification;
export type AppServerUsageStatsRangeParams = protocol.UsageStatsRangeParams;
export type AppServerUsageStatsReadResponse = protocol.UsageStatsReadResponse;
export type AppServerUsageStatsModelRankingListResponse =
  protocol.UsageStatsModelRankingListResponse;
export type AppServerUsageStatsDailyTrendsListResponse =
  protocol.UsageStatsDailyTrendsListResponse;
export type AppServerWindowsSandboxReadinessParams =
  protocol.WindowsSandboxReadinessParams;
export type AppServerWindowsSandboxReadinessResponse =
  protocol.WindowsSandboxReadinessResponse;
export type AppServerWindowsSandboxSetupStartParams =
  protocol.WindowsSandboxSetupStartParams;
export type AppServerWindowsSandboxSetupStartResponse =
  protocol.WindowsSandboxSetupStartResponse;
export type AppServerModelProviderCapabilitiesReadParams =
  protocol.ModelProviderCapabilitiesReadParams;
export type AppServerModelProviderCapabilitiesReadResponse =
  protocol.ModelProviderCapabilitiesReadResponse;
export type AppServerEnvironmentAddParams = protocol.EnvironmentAddParams;
export type AppServerEnvironmentAddResponse = protocol.EnvironmentAddResponse;
export type AppServerEnvironmentInfoParams = protocol.EnvironmentInfoParams;
export type AppServerEnvironmentInfoResponse = protocol.EnvironmentInfoResponse;
export type AppServerEnvironmentStatusParams = protocol.EnvironmentStatusParams;
export type AppServerEnvironmentStatusResponse =
  protocol.EnvironmentStatusResponse;
export type AppServerThreadQueueAddParams = protocol.ThreadQueueAddParams;
export type AppServerThreadQueueAddResponse = protocol.ThreadQueueAddResponse;
export type AppServerThreadQueueListParams = protocol.ThreadQueueListParams;
export type AppServerThreadQueueListResponse =
  protocol.ThreadQueueListResponse;
export type AppServerThreadQueueUpdateParams = protocol.ThreadQueueUpdateParams;
export type AppServerThreadQueueUpdateResponse =
  protocol.ThreadQueueUpdateResponse;
export type AppServerThreadQueueDeleteParams = protocol.ThreadQueueDeleteParams;
export type AppServerThreadQueueDeleteResponse =
  protocol.ThreadQueueDeleteResponse;
export type AppServerThreadQueueReorderParams =
  protocol.ThreadQueueReorderParams;
export type AppServerThreadQueueReorderResponse =
  protocol.ThreadQueueReorderResponse;
export type AppServerThreadQueueStartParams = protocol.ThreadQueueStartParams;
export type AppServerThreadQueueStartResponse =
  protocol.ThreadQueueStartResponse;
export type AppServerThreadQueueChangedNotification =
  protocol.ThreadQueueChangedNotification;

export type AppServerRequestResult<T> = {
  id: AppServerRequestId;
  result: T;
  response: AppServerJsonRpcResponse<T>;
  notifications: AppServerJsonRpcNotification[];
  configWarnings: AppServerConfigWarningNotification[];
  messages: AppServerJsonRpcMessage[];
};

export type AppServerRequestOptions = {
  signal?: AbortSignal;
};
