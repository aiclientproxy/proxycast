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
    forkThread(
      params: appServer.AppServerThreadForkParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadForkResponse>
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
    listDirectory(
      params: appServer.AppServerFileSystemListDirectoryParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerFileSystemDirectoryListing>
    >;
    readFilePreview(
      params: appServer.AppServerFileSystemReadFilePreviewParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerFileSystemFilePreview>
    >;
    createFile(
      params: appServer.AppServerFileSystemCreateFileParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerFileSystemMutationResponse>
    >;
    createDirectory(
      params: appServer.AppServerFileSystemCreateDirectoryParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerFileSystemMutationResponse>
    >;
    renameFile(
      params: appServer.AppServerFileSystemRenameFileParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerFileSystemMutationResponse>
    >;
    deleteFile(
      params: appServer.AppServerFileSystemDeleteFileParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerFileSystemMutationResponse>
    >;
    startExecutionProcess(
      params: appServer.AppServerExecutionProcessStartParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerExecutionProcessStartResponse>
    >;
    writeExecutionProcessStdin(
      params: appServer.AppServerExecutionProcessWriteStdinParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerExecutionProcessEmptyResponse>
    >;
    interruptExecutionProcess(
      params: appServer.AppServerExecutionProcessIdParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerExecutionProcessStatusResponse>
    >;
    terminateExecutionProcess(
      params: appServer.AppServerExecutionProcessIdParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerExecutionProcessStatusResponse>
    >;
    readExecutionProcessStatus(
      params: appServer.AppServerExecutionProcessIdParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerExecutionProcessStatusResponse>
    >;
    drainExecutionProcessOutput(
      params?: appServer.AppServerExecutionProcessDrainOutputParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerExecutionProcessDrainOutputResponse>
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
    exportEvidence(
      params: appServer.AppServerEvidenceExportParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerEvidenceExportResponse>
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
    readThread(
      params: appServer.AppServerThreadReadParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadReadResponse>
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
    readAgentSessionMedia(
      params: appServer.AppServerAgentSessionMediaReadParams,
      options?: appServer.AppServerRequestOptions,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerAgentSessionMediaReadResponse>
    >;
    readAgentSessionToolInventory(
      params?: appServer.AppServerAgentSessionToolInventoryReadParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerAgentSessionToolInventoryReadResponse>
    >;
    updateSession(
      params: appServer.AppServerAgentSessionUpdateParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerAgentSessionUpdateResponse>
    >;
    deleteThread(
      params: appServer.AppServerThreadDeleteParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadDeleteResponse>
    >;
    compactAgentSession(
      params: appServer.AppServerAgentSessionCompactParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerAgentSessionCompactResponse>
    >;
    resumeThread(
      params: appServer.AppServerThreadResumeParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerThreadResumeResponse>
    >;
    removeAgentSessionQueuedTurn(
      params: appServer.AppServerAgentSessionQueuedTurnRemoveParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerAgentSessionQueuedTurnRemoveResponse>
    >;
    promoteAgentSessionQueuedTurn(
      params: appServer.AppServerAgentSessionQueuedTurnPromoteParams,
    ): Promise<
      appServer.AppServerRequestResult<appServer.AppServerAgentSessionQueuedTurnPromoteResponse>
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
