import { APP_SERVER_CLIENT_METHODS } from "./appServerClientMethodSpecs";
import type * as appServer from "./appServerTypes";

type AppServerClientRequestRunner = {
  request<T>(
    method: string,
    params?: unknown,
    options?: appServer.AppServerRequestOptions,
  ): Promise<appServer.AppServerRequestResult<T>>;
};

declare module "./appServerClient" {
  interface AppServerClient {
    startSession(
      params: appServer.AppServerThreadStartParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadStartResponse>
    >;
    readWindowsSandboxReadiness(
      params?: appServer.AppServerWindowsSandboxReadinessParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerWindowsSandboxReadinessResponse>
    >;
    startWindowsSandboxSetup(
      params: appServer.AppServerWindowsSandboxSetupStartParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerWindowsSandboxSetupStartResponse>
    >;
    readModelProviderCapabilities(
      params?: appServer.AppServerModelProviderCapabilitiesReadParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerModelProviderCapabilitiesReadResponse>
    >;
    addEnvironment(
      params: appServer.AppServerEnvironmentAddParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerEnvironmentAddResponse>
    >;
    readEnvironmentInfo(
      params: appServer.AppServerEnvironmentInfoParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerEnvironmentInfoResponse>
    >;
    readEnvironmentStatus(
      params: appServer.AppServerEnvironmentStatusParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerEnvironmentStatusResponse>
    >;
    forkThread(
      params: appServer.AppServerThreadForkParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadForkResponse>
    >;
    revertThread(
      params: appServer.AppServerThreadRevertParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadRevertResponse>
    >;
    addThreadQueue(
      params: appServer.AppServerThreadQueueAddParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadQueueAddResponse>
    >;
    listThreadQueue(
      params: appServer.AppServerThreadQueueListParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadQueueListResponse>
    >;
    updateThreadQueue(
      params: appServer.AppServerThreadQueueUpdateParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadQueueUpdateResponse>
    >;
    deleteThreadQueue(
      params: appServer.AppServerThreadQueueDeleteParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadQueueDeleteResponse>
    >;
    reorderThreadQueue(
      params: appServer.AppServerThreadQueueReorderParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadQueueReorderResponse>
    >;
    startThreadQueue(
      params: appServer.AppServerThreadQueueStartParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadQueueStartResponse>
    >;
    readPromptHistory(
      params: appServer.AppServerPromptHistoryReadParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerPromptHistoryReadResponse>
    >;
    appendPromptHistory(
      params: appServer.AppServerPromptHistoryAppendParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerPromptHistoryAppendResponse>
    >;
    listSessions(
      params?: appServer.AppServerAgentSessionListParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerAgentSessionListResponse>
    >;
    listCapabilities(
      params?: appServer.AppServerCapabilityListParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerCapabilityListResponse>
    >;
    requestWorkspaceRightSurface(
      params: appServer.AppServerWorkspaceRightSurfaceRequestParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerWorkspaceRightSurfaceRequestResponse>
    >;
    listWorkspaceRightSurfacePending(
      params?: appServer.AppServerWorkspaceRightSurfacePendingListParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerWorkspaceRightSurfacePendingListResponse>
    >;
    consumeWorkspaceRightSurfacePending(
      params: appServer.AppServerWorkspaceRightSurfacePendingConsumeParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerWorkspaceRightSurfacePendingConsumeResponse>
    >;
    dismissWorkspaceRightSurfacePending(
      params: appServer.AppServerWorkspaceRightSurfacePendingDismissParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerWorkspaceRightSurfacePendingDismissResponse>
    >;
    readArtifacts(
      params: appServer.AppServerArtifactReadParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerArtifactReadResponse>
    >;
    writeArtifact(
      params: appServer.AppServerArtifactWriteParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerArtifactWriteResponse>
    >;
    searchFiles(
      params: appServer.AppServerFuzzyFileSearchParams,
      options?: appServer.AppServerRequestOptions,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerFuzzyFileSearchResponse>
    >;
    listPluginCatalog(
      params?: appServer.AppServerPluginCatalogListParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerPluginCatalogListResponse>
    >;
    searchPlugins(
      params: appServer.AppServerPluginSearchParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerPluginSearchResponse>
    >;
    setPluginCatalogEnabled(
      params: appServer.AppServerPluginCatalogEnabledSetParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerPluginCatalogEnabledSetResponse>
    >;
    readPluginCatalog(
      params: appServer.AppServerPluginCatalogReadParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerPluginCatalogReadResponse>
    >;
    installPluginCatalog(
      params: appServer.AppServerPluginCatalogInstallParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerPluginCatalogInstallResponse>
    >;
    uninstallPluginCatalog(
      params: appServer.AppServerPluginCatalogUninstallParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerPluginCatalogUninstallResponse>
    >;
    listInstalledPluginCatalog(
      params?: appServer.AppServerPluginCatalogInstalledParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerPluginCatalogListResponse>
    >;
    readFile(
      params: appServer.AppServerFsReadFileParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerFsReadFileResponse>
    >;
    writeFile(
      params: appServer.AppServerFsWriteFileParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerFsWriteFileResponse>
    >;
    createDirectory(
      params: appServer.AppServerFsCreateDirectoryParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerFsCreateDirectoryResponse>
    >;
    getMetadata(
      params: appServer.AppServerFsGetMetadataParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerFsGetMetadataResponse>
    >;
    readDirectory(
      params: appServer.AppServerFsReadDirectoryParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerFsReadDirectoryResponse>
    >;
    remove(
      params: appServer.AppServerFsRemoveParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerFsRemoveResponse>
    >;
    copy(
      params: appServer.AppServerFsCopyParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerFsCopyResponse>
    >;
    watch(
      params: appServer.AppServerFsWatchParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerFsWatchResponse>
    >;
    unwatch(
      params: appServer.AppServerFsUnwatchParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerFsUnwatchResponse>
    >;
    readProjectGitStatus(
      params: appServer.AppServerProjectGitStatusParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerProjectGitStatusResponse>
    >;
    readProjectGitDiff(
      params: appServer.AppServerProjectGitDiffParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerProjectGitDiffResponse>
    >;
    listProjectGitCommits(
      params: appServer.AppServerProjectGitCommitListParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerProjectGitCommitListResponse>
    >;
    checkoutProjectGitBranch(
      params: appServer.AppServerProjectGitBranchCheckoutParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerProjectGitBranchCheckoutResponse>
    >;
    createProjectGitBranch(
      params: appServer.AppServerProjectGitBranchCreateParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerProjectGitBranchCreateResponse>
    >;
    createProjectGitWorktree(
      params: appServer.AppServerProjectGitWorktreeCreateParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerProjectGitWorktreeCreateResponse>
    >;
    exportHandoffBundle(
      params: appServer.AppServerAgentSessionHandoffBundleExportParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerAgentSessionHandoffBundleExportResponse>
    >;
    exportReplayCase(
      params: appServer.AppServerAgentSessionReplayCaseExportParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerAgentSessionReplayCaseExportResponse>
    >;
    exportAnalysisHandoff(
      params: appServer.AppServerAgentSessionAnalysisHandoffExportParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerAgentSessionAnalysisHandoffExportResponse>
    >;
    exportReviewDecisionTemplate(
      params: appServer.AppServerAgentSessionReviewDecisionTemplateExportParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerAgentSessionReviewDecisionTemplateExportResponse>
    >;
    saveReviewDecision(
      params: appServer.AppServerAgentSessionReviewDecisionSaveParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerAgentSessionReviewDecisionTemplateExportResponse>
    >;
    readSession(
      params: appServer.AppServerAgentSessionReadParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerAgentSessionReadResponse>
    >;
    listThreads(
      params?: appServer.AppServerThreadListParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadListResponse>
    >;
    moveThreadToSection(
      params: appServer.AppServerThreadSectionMoveParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadSectionMoveResponse>
    >;
    listThreadSections(
      params?: appServer.AppServerThreadSectionListParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadSectionListResponse>
    >;
    createThreadSection(
      params: appServer.AppServerThreadSectionCreateParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadSectionCreateResponse>
    >;
    updateThreadSection(
      params: appServer.AppServerThreadSectionUpdateParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadSectionUpdateResponse>
    >;
    deleteThreadSection(
      params: appServer.AppServerThreadSectionDeleteParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadSectionDeleteResponse>
    >;
    readThread(
      params: appServer.AppServerThreadReadParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadReadResponse>
    >;
    setThreadName(
      params: appServer.AppServerThreadSetNameParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadSetNameResponse>
    >;
    updateThreadSettings(
      params: appServer.AppServerThreadSettingsUpdateParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadSettingsUpdateResponse>
    >;
    setThreadMemoryMode(
      params: appServer.AppServerThreadMemoryModeSetParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadMemoryModeSetResponse>
    >;
    runThreadShellCommand(
      params: appServer.AppServerThreadShellCommandParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadShellCommandResponse>
    >;
    execCommand(
      params: appServer.AppServerCommandExecParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerCommandExecResponse>
    >;
    writeCommandExec(
      params: appServer.AppServerCommandExecWriteParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerCommandExecWriteResponse>
    >;
    resizeCommandExec(
      params: appServer.AppServerCommandExecResizeParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerCommandExecResizeResponse>
    >;
    terminateCommandExec(
      params: appServer.AppServerCommandExecTerminateParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerCommandExecTerminateResponse>
    >;
    listThreadBackgroundTerminals(
      params: appServer.AppServerThreadBackgroundTerminalsListParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadBackgroundTerminalsListResponse>
    >;
    terminateThreadBackgroundTerminal(
      params: appServer.AppServerThreadBackgroundTerminalsTerminateParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadBackgroundTerminalsTerminateResponse>
    >;
    archiveThread(
      params: appServer.AppServerThreadArchiveParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadArchiveResponse>
    >;
    unarchiveThread(
      params: appServer.AppServerThreadUnarchiveParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadUnarchiveResponse>
    >;
    readMedia(
      params: appServer.AppServerMediaReadParams,
      options?: appServer.AppServerRequestOptions,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerMediaReadResponse>
    >;
    listApps(
      params?: appServer.AppServerAppsListParams,
      options?: appServer.AppServerRequestOptions,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerAppsListResponse>
    >;
    readApps(
      params: appServer.AppServerAppsReadParams,
      options?: appServer.AppServerRequestOptions,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerAppsReadResponse>
    >;
    listInstalledApps(
      params?: appServer.AppServerAppsInstalledParams,
      options?: appServer.AppServerRequestOptions,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerAppsInstalledResponse>
    >;
    listSkills(
      params?: appServer.AppServerSkillsListParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerSkillsListResponse>
    >;
    setSkillsExtraRoots(
      params: appServer.AppServerSkillsExtraRootsSetParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerSkillsExtraRootsSetResponse>
    >;
    writeSkillsConfig(
      params: appServer.AppServerSkillsConfigWriteParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerSkillsConfigWriteResponse>
    >;
    readAgentSessionToolInventory(
      params?: appServer.AppServerAgentSessionToolInventoryReadParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerAgentSessionToolInventoryReadResponse>
    >;
    deleteThread(
      params: appServer.AppServerThreadDeleteParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadDeleteResponse>
    >;
    startThreadCompaction(
      params: appServer.AppServerThreadCompactStartParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadCompactStartResponse>
    >;
    resumeThread(
      params: appServer.AppServerThreadResumeParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadResumeResponse>
    >;
    listAgentSessionFileCheckpoints(
      params: appServer.AppServerAgentSessionFileCheckpointListParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerAgentSessionFileCheckpointListResponse>
    >;
    getAgentSessionFileCheckpoint(
      params: appServer.AppServerAgentSessionFileCheckpointGetParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerAgentSessionFileCheckpointDetail>
    >;
    diffAgentSessionFileCheckpoint(
      params: appServer.AppServerAgentSessionFileCheckpointDiffParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerAgentSessionFileCheckpointDiffResponse>
    >;
    restoreAgentSessionFileCheckpoint(
      params: appServer.AppServerAgentSessionFileCheckpointRestoreParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerAgentSessionFileCheckpointRestoreResponse>
    >;
    getOrCreateSessionFile(
      params: appServer.AppServerSessionFileGetOrCreateParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerSessionFileMetaResponse>
    >;
    updateSessionFileMeta(
      params: appServer.AppServerSessionFileUpdateMetaParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerSessionFileMetaResponse>
    >;
    saveSessionFile(
      params: appServer.AppServerSessionFileSaveParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerSessionFileEntryResponse>
    >;
    readSessionFile(
      params: appServer.AppServerSessionFileIdParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerSessionFileReadResponse>
    >;
    resolveSessionFilePath(
      params: appServer.AppServerSessionFileIdParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerSessionFileResolvePathResponse>
    >;
    deleteSessionFile(
      params: appServer.AppServerSessionFileIdParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerSessionFileMutationResponse>
    >;
    listSessionFiles(
      params: appServer.AppServerSessionFileGetOrCreateParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerSessionFileListResponse>
    >;
    startTurn(
      params: appServer.AppServerAgentSessionTurnStartParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerAgentSessionTurnStartResponse>
    >;
    startReview(
      params: appServer.AppServerReviewStartParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerReviewStartResponse>
    >;
    cancelTurn(
      params: appServer.AppServerAgentSessionTurnCancelParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerAgentSessionTurnCancelResponse>
    >;
    steerTurn(
      params: appServer.AppServerTurnSteerParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerTurnSteerResponse>
    >;
    respondAction(
      params: appServer.AppServerAgentSessionActionRespondParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerAgentSessionActionRespondResponse>
    >;
    readWorkflow(
      params: appServer.AppServerWorkflowReadParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerWorkflowReadResponse>
    >;
    cancelWorkflow(
      params: appServer.AppServerWorkflowCancelParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerWorkflowCancelResponse>
    >;
    retryWorkflow(
      params: appServer.AppServerWorkflowRetryParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerWorkflowRetryResponse>
    >;
    respondWorkflow(
      params: appServer.AppServerWorkflowRespondParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerWorkflowRespondResponse>
    >;
    listLogs(): Promise<
      appServer.AppServerRequestResult<appServer.AppServerLogListResponse>
    >;
    readPersistedLogTail(
      params: appServer.AppServerLogPersistedTailParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerLogPersistedTailResponse>
    >;
    clearLogs(): Promise<
      appServer.AppServerRequestResult<appServer.AppServerLogClearResponse>
    >;
    clearDiagnosticLogHistory(): Promise<
      appServer.AppServerRequestResult<appServer.AppServerLogClearResponse>
    >;
    readLogStorageDiagnostics(): Promise<
      appServer.AppServerRequestResult<appServer.AppServerLogStorageDiagnosticsResponse>
    >;
    exportSupportBundle(
      params?: appServer.AppServerSupportBundleExportParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerSupportBundleExportResponse>
    >;
    readServerDiagnostics(): Promise<
      appServer.AppServerRequestResult<appServer.AppServerServerDiagnosticsResponse>
    >;
    readWindowsStartupDiagnostics(): Promise<
      appServer.AppServerRequestResult<appServer.AppServerWindowsStartupDiagnosticsResponse>
    >;
    listDiagnosticsTraces(
      params: appServer.AppServerDiagnosticsTraceListParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerDiagnosticsTraceListResponse>
    >;
    readDiagnosticsTrace(
      params: appServer.AppServerDiagnosticsTraceReadParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerDiagnosticsTraceReadResponse>
    >;
    exportDiagnosticsTrace(
      params: appServer.AppServerDiagnosticsTraceExportParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerDiagnosticsTraceExportResponse>
    >;
    readGatewayChannelStatus(
      params: appServer.AppServerGatewayChannelStatusParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerGatewayChannelStatusResponse>
    >;
    startGatewayChannel(
      params: appServer.AppServerGatewayChannelStartParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerGatewayChannelStatusResponse>
    >;
    stopGatewayChannel(
      params: appServer.AppServerGatewayChannelStopParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerGatewayChannelStatusResponse>
    >;
    probeTelegramChannel(
      params?: appServer.AppServerChannelProbeParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerChannelProbeResponse>
    >;
    probeFeishuChannel(
      params?: appServer.AppServerChannelProbeParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerChannelProbeResponse>
    >;
    probeDiscordChannel(
      params?: appServer.AppServerChannelProbeParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerChannelProbeResponse>
    >;
    probeWechatChannel(
      params?: appServer.AppServerChannelProbeParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerChannelProbeResponse>
    >;
    startWechatChannelLogin(
      params?: appServer.AppServerWechatLoginStartParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerWechatLoginStartResponse>
    >;
    waitWechatChannelLogin(
      params: appServer.AppServerWechatLoginWaitParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerWechatLoginWaitResponse>
    >;
    listWechatChannelAccounts(): Promise<
      appServer.AppServerRequestResult<appServer.AppServerWechatChannelAccountListResponse>
    >;
    removeWechatChannelAccount(
      params: appServer.AppServerWechatChannelAccountRemoveParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerWechatChannelAccountRemoveResponse>
    >;
    setWechatChannelRuntimeModel(
      params: appServer.AppServerWechatRuntimeModelSetParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerWechatRuntimeModelSetResponse>
    >;
    probeGatewayTunnel(): Promise<
      appServer.AppServerRequestResult<appServer.AppServerGatewayTunnelProbeResponse>
    >;
    detectGatewayTunnelCloudflared(): Promise<
      appServer.AppServerRequestResult<appServer.AppServerGatewayTunnelCloudflaredDetectResponse>
    >;
    installGatewayTunnelCloudflared(
      params: appServer.AppServerGatewayTunnelCloudflaredInstallParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerGatewayTunnelCloudflaredInstallResponse>
    >;
    createGatewayTunnel(
      params: appServer.AppServerGatewayTunnelCreateParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerGatewayTunnelCreateResponse>
    >;
    startGatewayTunnel(): Promise<
      appServer.AppServerRequestResult<appServer.AppServerGatewayTunnelStatusResponse>
    >;
    stopGatewayTunnel(): Promise<
      appServer.AppServerRequestResult<appServer.AppServerGatewayTunnelStatusResponse>
    >;
    restartGatewayTunnel(): Promise<
      appServer.AppServerRequestResult<appServer.AppServerGatewayTunnelStatusResponse>
    >;
    readGatewayTunnelStatus(): Promise<
      appServer.AppServerRequestResult<appServer.AppServerGatewayTunnelStatusResponse>
    >;
    syncGatewayTunnelWebhookUrl(
      params: appServer.AppServerGatewayTunnelSyncWebhookUrlParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerGatewayTunnelSyncWebhookUrlResponse>
    >;
    createImageMediaTaskArtifact(
      params: appServer.AppServerMediaTaskArtifactImageCreateParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerMediaTaskArtifactResponse>
    >;
    createAudioMediaTaskArtifact(
      params: appServer.AppServerMediaTaskArtifactAudioCreateParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerMediaTaskArtifactResponse>
    >;
    createTranscriptionMediaTaskArtifact(
      params: appServer.AppServerMediaTaskArtifactTranscriptionCreateParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerMediaTaskArtifactResponse>
    >;
    createVideoMediaTaskArtifact(
      params: appServer.AppServerMediaTaskArtifactVideoCreateParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerMediaTaskArtifactResponse>
    >;
    completeImageMediaTaskArtifact(
      params: appServer.AppServerMediaTaskArtifactImageCompleteParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerMediaTaskArtifactResponse>
    >;
    completeAudioMediaTaskArtifact(
      params: appServer.AppServerMediaTaskArtifactAudioCompleteParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerMediaTaskArtifactResponse>
    >;
    getMediaTaskArtifact(
      params: appServer.AppServerMediaTaskArtifactLookupParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerMediaTaskArtifactResponse>
    >;
    listMediaTaskArtifacts(
      params: appServer.AppServerMediaTaskArtifactListParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerMediaTaskArtifactListResponse>
    >;
    cancelMediaTaskArtifact(
      params: appServer.AppServerMediaTaskArtifactLookupParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerMediaTaskArtifactResponse>
    >;
    getGalleryMaterial(
      params: appServer.AppServerGalleryMaterialLookupParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerGalleryMaterialResponse>
    >;
    createGalleryMaterialMetadata(
      params: appServer.AppServerGalleryMaterialMetadataCreateParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerGalleryMaterialMetadataResponse>
    >;
    getGalleryMaterialMetadata(
      params: appServer.AppServerGalleryMaterialLookupParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerGalleryMaterialMetadataResponse>
    >;
    updateGalleryMaterialMetadata(
      params: appServer.AppServerGalleryMaterialMetadataUpdateParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerGalleryMaterialMetadataResponse>
    >;
    deleteGalleryMaterialMetadata(
      params: appServer.AppServerGalleryMaterialLookupParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerGalleryMaterialDeleteResponse>
    >;
    listGalleryMaterialsByImageCategory(
      params: appServer.AppServerGalleryMaterialFilterParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerGalleryMaterialListResponse>
    >;
    listGalleryMaterialsByLayoutCategory(
      params: appServer.AppServerGalleryMaterialFilterParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerGalleryMaterialListResponse>
    >;
    listGalleryMaterialsByMood(
      params: appServer.AppServerGalleryMaterialFilterParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerGalleryMaterialListResponse>
    >;
    listProjectMaterials(
      params: appServer.AppServerProjectMaterialListParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerProjectMaterialListResponse>
    >;
    getProjectMaterial(
      params: appServer.AppServerProjectMaterialLookupParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerProjectMaterialResponse>
    >;
    countProjectMaterials(
      params: appServer.AppServerProjectMaterialListParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerProjectMaterialCountResponse>
    >;
    uploadProjectMaterial(
      params: appServer.AppServerProjectMaterialUploadParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerProjectMaterialResponse>
    >;
    importProjectMaterialFromUrl(
      params: appServer.AppServerProjectMaterialImportFromUrlParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerProjectMaterialResponse>
    >;
    updateProjectMaterial(
      params: appServer.AppServerProjectMaterialUpdateParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerProjectMaterialResponse>
    >;
    deleteProjectMaterial(
      params: appServer.AppServerProjectMaterialLookupParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerProjectMaterialDeleteResponse>
    >;
    readProjectMaterialContent(
      params: appServer.AppServerProjectMaterialLookupParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerProjectMaterialContentResponse>
    >;
    listVoiceAsrCredentials(): Promise<
      appServer.AppServerRequestResult<appServer.AppServerVoiceAsrCredentialListResponse>
    >;
    createVoiceAsrCredential(
      params: appServer.AppServerVoiceAsrCredentialCreateParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerVoiceAsrCredentialWriteResponse>
    >;
    updateVoiceAsrCredential(
      params: appServer.AppServerVoiceAsrCredentialUpdateParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerVoiceAsrCredentialMutationResponse>
    >;
    deleteVoiceAsrCredential(
      params: appServer.AppServerVoiceAsrCredentialIdParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerVoiceAsrCredentialMutationResponse>
    >;
    setDefaultVoiceAsrCredential(
      params: appServer.AppServerVoiceAsrCredentialIdParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerVoiceAsrCredentialMutationResponse>
    >;
    testVoiceAsrCredential(
      params: appServer.AppServerVoiceAsrCredentialIdParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerVoiceAsrCredentialTestResponse>
    >;
    listVoiceInstructions(): Promise<
      appServer.AppServerRequestResult<appServer.AppServerVoiceInstructionListResponse>
    >;
    saveVoiceInstruction(
      params: appServer.AppServerVoiceInstructionSaveParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerVoiceInstructionMutationResponse>
    >;
    deleteVoiceInstruction(
      params: appServer.AppServerVoiceInstructionIdParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerVoiceInstructionMutationResponse>
    >;
    setDefaultVoiceModel(
      params: appServer.AppServerVoiceModelDefaultSetParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerVoiceModelDefaultSetResponse>
    >;
    testTranscribeVoiceModelFile(
      params: appServer.AppServerVoiceModelTestTranscribeFileParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerVoiceModelTestTranscribeFileResponse>
    >;
    transcribeVoiceAudio(
      params: appServer.AppServerVoiceTranscriptionTranscribeAudioParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerVoiceTranscriptionTranscribeAudioResponse>
    >;
    polishVoiceText(
      params: appServer.AppServerVoiceTranscriptionPolishTextParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerVoiceTranscriptionPolishTextResponse>
    >;
    readUsageStats(
      params: appServer.AppServerUsageStatsRangeParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerUsageStatsReadResponse>
    >;
    listUsageStatsModelRanking(
      params: appServer.AppServerUsageStatsRangeParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerUsageStatsModelRankingListResponse>
    >;
    listUsageStatsDailyTrends(
      params: appServer.AppServerUsageStatsRangeParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerUsageStatsDailyTrendsListResponse>
    >;
  }
}

export function installAppServerClientMethods(prototype: object): void {
  for (const spec of APP_SERVER_CLIENT_METHODS) {
    Object.defineProperty(prototype, spec.name, {
      configurable: true,
      value: function (
        this: AppServerClientRequestRunner,
        params?: unknown,
        options?: appServer.AppServerRequestOptions,
      ) {
        if (spec.params === "none") {
          return this.request(
            spec.method,
            {},
            params as appServer.AppServerRequestOptions | undefined,
          );
        }
        if (spec.params === "optional-empty") {
          return this.request(spec.method, params ?? {}, options);
        }
        return this.request(spec.method, params, options);
      },
    });
  }
}
