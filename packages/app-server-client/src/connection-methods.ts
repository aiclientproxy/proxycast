import * as protocol from "./protocol.js";
import type { AppServerClient } from "./request-client.js";
import type {
  AppServerConnection,
  AppServerRequestOptions,
  AppServerRequestResult,
} from "./connection.js";
import {
  APP_SERVER_REQUEST_CLIENT_METHODS,
  type AppServerRequestClientMethodSpec,
  type RequestClientParamsMode,
} from "./request-client-methods.js";

type ConnectionParamsMode = RequestClientParamsMode;

type ConnectionMethodSpec = Pick<
  AppServerRequestClientMethodSpec,
  "name" | "method" | "params"
> & {
  clientMethod: string;
};

declare module "./connection.js" {
  interface AppServerConnection {
    startSession(
      params: protocol.ThreadStartParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ThreadStartResponse>>;
    forkThread(
      params: protocol.ThreadForkParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ThreadForkResponse>>;
    listCapabilities(
      params?: protocol.CapabilityListParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.CapabilityListResponse>>;
    listSessions(
      params?: protocol.AgentSessionListParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.AgentSessionListResponse>>;
    readThread(
      params: protocol.ThreadReadParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ThreadReadResponse>>;
    listThreads(
      params?: protocol.ThreadListParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ThreadListResponse>>;
    listLoadedThreads(
      params?: protocol.ThreadLoadedListParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ThreadLoadedListResponse>>;
    incrementThreadElicitation(
      params: protocol.ThreadIncrementElicitationParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.ThreadIncrementElicitationResponse>
    >;
    decrementThreadElicitation(
      params: protocol.ThreadDecrementElicitationParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.ThreadDecrementElicitationResponse>
    >;
    approveGuardianDeniedAction(
      params: protocol.ThreadApproveGuardianDeniedActionParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.ThreadApproveGuardianDeniedActionResponse>
    >;
    injectThreadItems(
      params: protocol.ThreadInjectItemsParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ThreadInjectItemsResponse>>;
    archiveThread(
      params: protocol.ThreadArchiveParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ThreadArchiveResponse>>;
    unarchiveThread(
      params: protocol.ThreadUnarchiveParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ThreadUnarchiveResponse>>;
    setThreadName(
      params: protocol.ThreadSetNameParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ThreadSetNameResponse>>;
    updateThreadMetadata(
      params: protocol.ThreadMetadataUpdateParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ThreadMetadataUpdateResponse>>;
    listThreadTurns(
      params: protocol.ThreadTurnsListParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ThreadTurnsListResponse>>;
    listThreadItems(
      params: protocol.ThreadItemsListParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ThreadItemsListResponse>>;
    searchThreads(
      params: protocol.ThreadSearchParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ThreadSearchResponse>>;
    searchThreadOccurrences(
      params: protocol.ThreadSearchOccurrencesParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.ThreadSearchOccurrencesResponse>
    >;
    cleanThreadBackgroundTerminals(
      params: protocol.ThreadBackgroundTerminalsCleanParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.ThreadBackgroundTerminalsCleanResponse>
    >;
    listThreadBackgroundTerminals(
      params: protocol.ThreadBackgroundTerminalsListParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.ThreadBackgroundTerminalsListResponse>
    >;
    terminateThreadBackgroundTerminal(
      params: protocol.ThreadBackgroundTerminalsTerminateParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.ThreadBackgroundTerminalsTerminateResponse>
    >;
    updateThreadSettings(
      params: protocol.ThreadSettingsUpdateParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ThreadSettingsUpdateResponse>>;
    setThreadMemoryMode(
      params: protocol.ThreadMemoryModeSetParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ThreadMemoryModeSetResponse>>;
    deleteThread(
      params: protocol.ThreadDeleteParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ThreadDeleteResponse>>;
    startThreadCompaction(
      params: protocol.ThreadCompactStartParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ThreadCompactStartResponse>>;
    resumeThread(
      params: protocol.ThreadResumeParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ThreadResumeResponse>>;
    listAgentSessionFileCheckpoints(
      params: protocol.AgentSessionFileCheckpointListParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.AgentSessionFileCheckpointListResponse>
    >;
    getAgentSessionFileCheckpoint(
      params: protocol.AgentSessionFileCheckpointGetParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.AgentSessionFileCheckpointDetail>
    >;
    diffAgentSessionFileCheckpoint(
      params: protocol.AgentSessionFileCheckpointDiffParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.AgentSessionFileCheckpointDiffResponse>
    >;
    restoreAgentSessionFileCheckpoint(
      params: protocol.AgentSessionFileCheckpointRestoreParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.AgentSessionFileCheckpointRestoreResponse>
    >;
    getOrCreateSessionFile(
      params: protocol.SessionFileGetOrCreateParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.SessionFileMetaResponse>>;
    updateSessionFileMeta(
      params: protocol.SessionFileUpdateMetaParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.SessionFileMetaResponse>>;
    saveSessionFile(
      params: protocol.SessionFileSaveParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.SessionFileEntryResponse>>;
    readSessionFile(
      params: protocol.SessionFileIdParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.SessionFileReadResponse>>;
    resolveSessionFilePath(
      params: protocol.SessionFileIdParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.SessionFileResolvePathResponse>>;
    deleteSessionFile(
      params: protocol.SessionFileIdParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.SessionFileMutationResponse>>;
    listSessionFiles(
      params: protocol.SessionFileGetOrCreateParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.SessionFileListResponse>>;
    listWorkspaces(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.WorkspaceListResponse>>;
    readWorkspace(
      params: protocol.WorkspaceReadParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.WorkspaceReadResponse>>;
    updateWorkspace(
      params: protocol.WorkspaceUpdateParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.WorkspaceUpdateResponse>>;
    deleteWorkspace(
      params: protocol.WorkspaceDeleteParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.WorkspaceDeleteResponse>>;
    ensureWorkspace(
      params: protocol.WorkspaceEnsureProjectParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.WorkspaceEnsureProjectResponse>>;
    readWorkspaceByPath(
      params: protocol.WorkspacePathReadParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.WorkspaceReadResponse>>;
    readDefaultWorkspace(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.WorkspaceReadResponse>>;
    ensureDefaultWorkspace(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.WorkspaceReadResponse>>;
    readWorkspaceProjectsRoot(
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.WorkspaceProjectsRootReadResponse>
    >;
    resolveWorkspaceProjectPath(
      params: protocol.WorkspaceProjectPathResolveParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.WorkspaceProjectPathResolveResponse>
    >;
    ensureWorkspaceReady(
      params: protocol.WorkspaceEnsureParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.WorkspaceEnsureReadyResponse>>;
    requestWorkspaceRightSurface(
      params: protocol.WorkspaceRightSurfaceRequestParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.WorkspaceRightSurfaceRequestResponse>
    >;
    listWorkspaceRightSurfacePending(
      params?: protocol.WorkspaceRightSurfacePendingListParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.WorkspaceRightSurfacePendingListResponse>
    >;
    consumeWorkspaceRightSurfacePending(
      params: protocol.WorkspaceRightSurfacePendingConsumeParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.WorkspaceRightSurfacePendingConsumeResponse>
    >;
    dismissWorkspaceRightSurfacePending(
      params: protocol.WorkspaceRightSurfacePendingDismissParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.WorkspaceRightSurfacePendingDismissResponse>
    >;
    listSkills(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.SkillListResponse>>;
    readSkill(
      params: protocol.SkillReadParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.SkillReadResponse>>;
    listManagementSkills(
      params: protocol.SkillManagementListParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.SkillManagementListResponse>>;
    installManagementSkill(
      params: protocol.SkillManagementInstallParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.SkillManagementWriteResponse>>;
    uninstallManagementSkill(
      params: protocol.SkillManagementUninstallParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.SkillManagementWriteResponse>>;
    listSkillRepositories(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.SkillRepositoryListResponse>>;
    saveSkillRepository(
      params: protocol.SkillRepositorySaveParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.SkillManagementWriteResponse>>;
    deleteSkillRepository(
      params: protocol.SkillRepositoryDeleteParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.SkillManagementWriteResponse>>;
    refreshSkillCache(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.SkillManagementWriteResponse>>;
    listInstalledSkillDirectories(
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.SkillInstalledDirectoriesListResponse>
    >;
    inspectLocalSkill(
      params: protocol.SkillLocalInspectParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.SkillLocalInspectResponse>>;
    inspectLocalSkillPackage(
      params: protocol.SkillPackageLocalInspectParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.SkillPackageLocalInspectResponse>
    >;
    inspectLocalSkillDetail(
      params: protocol.SkillLocalDetailInspectParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.SkillLocalDetailInspectResponse>
    >;
    createSkillScaffold(
      params: protocol.SkillScaffoldCreateParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.SkillScaffoldCreateResponse>>;
    importLocalSkill(
      params: protocol.SkillLocalImportParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.SkillLocalImportResponse>>;
    renameLocalSkill(
      params: protocol.SkillLocalRenameParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.SkillLocalRenameResponse>>;
    inspectRemoteSkill(
      params: protocol.SkillRemoteInspectParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.SkillRemoteInspectResponse>>;
    installLocalSkillPackage(
      params: protocol.SkillPackageLocalInstallParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.SkillPackageLocalInstallResponse>
    >;
    replaceLocalSkillPackage(
      params: protocol.SkillPackageLocalReplaceParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.SkillPackageLocalReplaceResponse>
    >;
    exportSkillPackage(
      params: protocol.SkillPackageExportParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.SkillPackageExportResponse>>;
    installMarketplaceSkill(
      params: protocol.SkillMarketplaceInstallParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.SkillMarketplaceInstallResponse>
    >;
    installSkillFromDownload(
      params: protocol.SkillDownloadInstallParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.SkillDownloadInstallResponse>>;
    listWorkspaceSkillBindings(
      params: protocol.WorkspaceSkillBindingsListParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.WorkspaceSkillBindingsListResponse>
    >;
    listWorkspaceRegisteredSkills(
      params: protocol.WorkspaceRegisteredSkillsListParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.WorkspaceRegisteredSkillsListResponse>
    >;
    inspectPluginLocalPackage(
      params: protocol.PluginLocalPackageInspectParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.PluginLocalPackageInspectResponse>
    >;
    exportPluginLocalPackage(
      params: protocol.PluginLocalPackageExportParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.PluginLocalPackageExportResponse>
    >;
    fetchPluginCloudPackage(
      params: protocol.PluginFetchCloudPackageParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.PluginPackageCacheEntry>>;
    savePluginInstalled(
      params: protocol.PluginInstalledSaveParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<unknown>>;
    listPluginInstalled(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.PluginInstalledListResponse>>;
    setPluginInstalledDisabled(
      params: protocol.PluginInstalledDisabledSetParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.PluginInstalledListResponse>>;
    previewPluginUninstall(
      params: protocol.PluginUninstallRehearsalParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.PluginUninstallRehearsalResponse>
    >;
    uninstallPlugin(
      params: protocol.PluginUninstallParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.PluginUninstallResponse>>;
    preparePluginShell(
      params: protocol.PluginShellPrepareParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.PluginShellPrepareResponse>>;
    startPluginUiRuntime(
      params: protocol.PluginUiRuntimeStartParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.PluginUiRuntimeStatusResponse>>;
    getPluginUiRuntimeStatus(
      params: protocol.PluginUiRuntimeStatusParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.PluginUiRuntimeStatusResponse>>;
    stopPluginUiRuntime(
      params: protocol.PluginUiRuntimeStopParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.PluginUiRuntimeStatusResponse>>;
    listKnowledgePacks(
      params: protocol.KnowledgeListPacksParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.KnowledgeListPacksResponse>>;
    readKnowledgePack(
      params: protocol.KnowledgeReadPackParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.KnowledgeReadPackResponse>>;
    importKnowledgeSource(
      params: protocol.KnowledgeImportSourceParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.KnowledgeImportSourceResponse>>;
    compileKnowledgePack(
      params: protocol.KnowledgeCompilePackParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.KnowledgeCompilePackResponse>>;
    setDefaultKnowledgePack(
      params: protocol.KnowledgeSetDefaultPackParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.KnowledgeSetDefaultPackResponse>
    >;
    updateKnowledgePackStatus(
      params: protocol.KnowledgeUpdatePackStatusParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.KnowledgeUpdatePackStatusResponse>
    >;
    resolveKnowledgeContext(
      params: protocol.KnowledgeResolveContextParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.KnowledgeContextResolutionResponse>
    >;
    validateKnowledgeContextRun(
      params: protocol.KnowledgeValidateContextRunParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.KnowledgeValidateContextRunResponse>
    >;
    listAutomationJobs(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.AutomationJobListResponse>>;
    readAutomationSchedulerConfig(
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.AutomationSchedulerConfigReadResponse>
    >;
    updateAutomationSchedulerConfig(
      params: protocol.AutomationSchedulerConfigUpdateParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.AutomationSchedulerConfigUpdateResponse>
    >;
    readAutomationSchedulerStatus(
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.AutomationSchedulerStatusResponse>
    >;
    readAutomationJob(
      params: protocol.AutomationJobIdParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.AutomationJobReadResponse>>;
    createAutomationJob(
      params: protocol.AutomationJobCreateParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.AutomationJobWriteResponse>>;
    updateAutomationJob(
      params: protocol.AutomationJobUpdateParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.AutomationJobWriteResponse>>;
    deleteAutomationJob(
      params: protocol.AutomationJobIdParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.AutomationJobDeleteResponse>>;
    runAutomationJobNow(
      params: protocol.AutomationJobIdParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.AutomationJobRunNowResponse>>;
    readAutomationHealth(
      params?: protocol.AutomationJobHealthParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.AutomationJobHealthResponse>>;
    readAutomationRunHistory(
      params: protocol.AutomationJobRunHistoryParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.AutomationJobRunHistoryResponse>
    >;
    previewAutomationSchedule(
      params: protocol.AutomationScheduleParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.AutomationSchedulePreviewResponse>
    >;
    validateAutomationSchedule(
      params: protocol.AutomationScheduleParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.AutomationScheduleValidateResponse>
    >;
    listMcpServers(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.McpServerListResponse>>;
    listMcpServersWithStatus(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.McpServerStatusListResponse>>;
    createMcpServer(
      params: protocol.McpServerCreateParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.McpServerListResponse>>;
    updateMcpServer(
      params: protocol.McpServerUpdateParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.McpServerListResponse>>;
    deleteMcpServer(
      params: protocol.McpServerDeleteParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.McpServerListResponse>>;
    setMcpServerEnabled(
      params: protocol.McpServerEnabledSetParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.McpServerListResponse>>;
    importMcpServersFromApp(
      params: protocol.McpServerImportFromAppParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.McpServerImportFromAppResponse>>;
    syncAllMcpServersToLive(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.McpServerListResponse>>;
    loginMcpServerOauth(
      params: protocol.McpServerOauthLoginParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.McpServerOauthLoginResponse>>;
    startMcpServer(
      params: protocol.McpServerStartParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.McpServerLifecycleResponse>>;
    stopMcpServer(
      params: protocol.McpServerStopParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.McpServerLifecycleResponse>>;
    listMcpTools(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.McpToolListResponse>>;
    listMcpToolsForContext(
      params: protocol.McpToolListForContextParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.McpToolListResponse>>;
    searchMcpTools(
      params: protocol.McpToolSearchParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.McpToolListResponse>>;
    callMcpTool(
      params: protocol.McpToolCallParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.McpToolCallResponse>>;
    callMcpToolWithCaller(
      params: protocol.McpToolCallWithCallerParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.McpToolCallResponse>>;
    listMcpPrompts(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.McpPromptListResponse>>;
    getMcpPrompt(
      params: protocol.McpPromptGetParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.McpPromptGetResponse>>;
    listMcpResources(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.McpResourceListResponse>>;
    readMcpResource(
      params: protocol.McpResourceReadParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.McpResourceReadResponse>>;
    subscribeMcpResource(
      params: protocol.McpResourceSubscribeParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.McpResourceSubscriptionResponse>
    >;
    unsubscribeMcpResource(
      params: protocol.McpResourceUnsubscribeParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.McpResourceSubscriptionResponse>
    >;
    readProjectMemory(
      params: protocol.ProjectMemoryReadParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ProjectMemoryReadResponse>>;
    listMemoryStore(
      params: protocol.MemoryStoreListParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.MemoryStoreListResponse>>;
    readMemoryStore(
      params: protocol.MemoryStoreReadParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.MemoryStoreReadResponse>>;
    searchMemoryStore(
      params: protocol.MemoryStoreSearchParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.MemoryStoreSearchResponse>>;
    addMemoryStoreNote(
      params: protocol.MemoryStoreAddNoteParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.MemoryStoreAddNoteResponse>>;
    consolidateMemoryStore(
      params: protocol.MemoryStoreConsolidateParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.MemoryStoreConsolidateResponse>>;
    listMemoryStoreReviewNotes(
      params: protocol.MemoryStoreReviewListParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.MemoryStoreReviewListResponse>>;
    resolveMemoryStoreReviewNote(
      params: protocol.MemoryStoreReviewResolveParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.MemoryStoreReviewResolveResponse>
    >;
    healthMemoryStore(
      params: protocol.MemoryStoreRootParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.MemoryStoreHealthResponse>>;
    resetMemoryStore(
      params: protocol.MemoryStoreResetParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.MemoryStoreResetResponse>>;
    rebuildMemoryStoreIndex(
      params: protocol.MemoryStoreRootParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.MemoryStoreIndexRebuildResponse>
    >;
    listLogs(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.LogListResponse>>;
    readPersistedLogTail(
      params: protocol.LogPersistedTailParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.LogPersistedTailResponse>>;
    clearLogs(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.LogClearResponse>>;
    clearDiagnosticLogHistory(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.LogClearResponse>>;
    readLogStorageDiagnostics(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.LogStorageDiagnosticsResponse>>;
    exportSupportBundle(
      params?: protocol.SupportBundleExportParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.SupportBundleExportResponse>>;
    readServerDiagnostics(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ServerDiagnosticsResponse>>;
    readWindowsStartupDiagnostics(
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.WindowsStartupDiagnosticsResponse>
    >;
    listDiagnosticsTraces(
      params: protocol.DiagnosticsTraceListParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.DiagnosticsTraceListResponse>>;
    readDiagnosticsTrace(
      params: protocol.DiagnosticsTraceReadParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.DiagnosticsTraceReadResponse>>;
    exportDiagnosticsTrace(
      params: protocol.DiagnosticsTraceExportParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.DiagnosticsTraceExportResponse>>;
    readGatewayChannelStatus(
      params: protocol.GatewayChannelStatusParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.GatewayChannelStatusResponse>>;
    startGatewayChannel(
      params: protocol.GatewayChannelStartParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.GatewayChannelStatusResponse>>;
    stopGatewayChannel(
      params: protocol.GatewayChannelStopParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.GatewayChannelStatusResponse>>;
    probeTelegramChannel(
      params?: protocol.ChannelProbeParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ChannelProbeResponse>>;
    probeFeishuChannel(
      params?: protocol.ChannelProbeParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ChannelProbeResponse>>;
    probeDiscordChannel(
      params?: protocol.ChannelProbeParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ChannelProbeResponse>>;
    probeWechatChannel(
      params?: protocol.ChannelProbeParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ChannelProbeResponse>>;
    startWechatChannelLogin(
      params?: protocol.WechatLoginStartParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.WechatLoginStartResponse>>;
    waitWechatChannelLogin(
      params: protocol.WechatLoginWaitParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.WechatLoginWaitResponse>>;
    listWechatChannelAccounts(
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.WechatChannelAccountListResponse>
    >;
    removeWechatChannelAccount(
      params: protocol.WechatChannelAccountRemoveParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.WechatChannelAccountRemoveResponse>
    >;
    setWechatChannelRuntimeModel(
      params: protocol.WechatRuntimeModelSetParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.WechatRuntimeModelSetResponse>>;
    probeGatewayTunnel(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.GatewayTunnelProbeResponse>>;
    detectGatewayTunnelCloudflared(
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.GatewayTunnelCloudflaredDetectResponse>
    >;
    installGatewayTunnelCloudflared(
      params: protocol.GatewayTunnelCloudflaredInstallParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.GatewayTunnelCloudflaredInstallResponse>
    >;
    createGatewayTunnel(
      params: protocol.GatewayTunnelCreateParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.GatewayTunnelCreateResponse>>;
    startGatewayTunnel(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.GatewayTunnelStatusResponse>>;
    stopGatewayTunnel(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.GatewayTunnelStatusResponse>>;
    restartGatewayTunnel(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.GatewayTunnelStatusResponse>>;
    readGatewayTunnelStatus(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.GatewayTunnelStatusResponse>>;
    syncGatewayTunnelWebhookUrl(
      params: protocol.GatewayTunnelSyncWebhookUrlParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.GatewayTunnelSyncWebhookUrlResponse>
    >;
    createImageMediaTaskArtifact(
      params: protocol.MediaTaskArtifactImageCreateParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.MediaTaskArtifactResponse>>;
    createAudioMediaTaskArtifact(
      params: protocol.MediaTaskArtifactAudioCreateParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.MediaTaskArtifactResponse>>;
    createVideoMediaTaskArtifact(
      params: protocol.MediaTaskArtifactVideoCreateParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.MediaTaskArtifactResponse>>;
    completeImageMediaTaskArtifact(
      params: protocol.MediaTaskArtifactImageCompleteParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.MediaTaskArtifactResponse>>;
    completeAudioMediaTaskArtifact(
      params: protocol.MediaTaskArtifactAudioCompleteParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.MediaTaskArtifactResponse>>;
    getMediaTaskArtifact(
      params: protocol.MediaTaskArtifactLookupParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.MediaTaskArtifactResponse>>;
    listMediaTaskArtifacts(
      params: protocol.MediaTaskArtifactListParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.MediaTaskArtifactListResponse>>;
    cancelMediaTaskArtifact(
      params: protocol.MediaTaskArtifactLookupParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.MediaTaskArtifactResponse>>;
    getGalleryMaterial(
      params: protocol.GalleryMaterialLookupParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.GalleryMaterialResponse>>;
    createGalleryMaterialMetadata(
      params: protocol.GalleryMaterialMetadataCreateParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.GalleryMaterialMetadataResponse>
    >;
    getGalleryMaterialMetadata(
      params: protocol.GalleryMaterialLookupParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.GalleryMaterialMetadataResponse>
    >;
    updateGalleryMaterialMetadata(
      params: protocol.GalleryMaterialMetadataUpdateParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.GalleryMaterialMetadataResponse>
    >;
    deleteGalleryMaterialMetadata(
      params: protocol.GalleryMaterialLookupParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.GalleryMaterialDeleteResponse>>;
    listGalleryMaterialsByImageCategory(
      params: protocol.GalleryMaterialFilterParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.GalleryMaterialListResponse>>;
    listGalleryMaterialsByLayoutCategory(
      params: protocol.GalleryMaterialFilterParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.GalleryMaterialListResponse>>;
    listGalleryMaterialsByMood(
      params: protocol.GalleryMaterialFilterParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.GalleryMaterialListResponse>>;
    listProjectMaterials(
      params: protocol.ProjectMaterialListParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ProjectMaterialListResponse>>;
    getProjectMaterial(
      params: protocol.ProjectMaterialLookupParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ProjectMaterialResponse>>;
    countProjectMaterials(
      params: protocol.ProjectMaterialListParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ProjectMaterialCountResponse>>;
    uploadProjectMaterial(
      params: protocol.ProjectMaterialUploadParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ProjectMaterialResponse>>;
    importProjectMaterialFromUrl(
      params: protocol.ProjectMaterialImportFromUrlParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ProjectMaterialResponse>>;
    updateProjectMaterial(
      params: protocol.ProjectMaterialUpdateParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ProjectMaterialResponse>>;
    deleteProjectMaterial(
      params: protocol.ProjectMaterialLookupParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ProjectMaterialDeleteResponse>>;
    readProjectMaterialContent(
      params: protocol.ProjectMaterialLookupParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ProjectMaterialContentResponse>>;
    listVoiceAsrCredentials(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.VoiceAsrCredentialListResponse>>;
    createVoiceAsrCredential(
      params: protocol.VoiceAsrCredentialCreateParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.VoiceAsrCredentialWriteResponse>
    >;
    updateVoiceAsrCredential(
      params: protocol.VoiceAsrCredentialUpdateParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.VoiceAsrCredentialMutationResponse>
    >;
    deleteVoiceAsrCredential(
      params: protocol.VoiceAsrCredentialIdParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.VoiceAsrCredentialMutationResponse>
    >;
    setDefaultVoiceAsrCredential(
      params: protocol.VoiceAsrCredentialIdParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.VoiceAsrCredentialMutationResponse>
    >;
    testVoiceAsrCredential(
      params: protocol.VoiceAsrCredentialIdParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.VoiceAsrCredentialTestResponse>>;
    listVoiceInstructions(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.VoiceInstructionListResponse>>;
    saveVoiceInstruction(
      params: protocol.VoiceInstructionSaveParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.VoiceInstructionMutationResponse>
    >;
    deleteVoiceInstruction(
      params: protocol.VoiceInstructionIdParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.VoiceInstructionMutationResponse>
    >;
    setDefaultVoiceModel(
      params: protocol.VoiceModelDefaultSetParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.VoiceModelDefaultSetResponse>>;
    testTranscribeVoiceModelFile(
      params: protocol.VoiceModelTestTranscribeFileParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.VoiceModelTestTranscribeFileResponse>
    >;
    transcribeVoiceAudio(
      params: protocol.VoiceTranscriptionTranscribeAudioParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.VoiceTranscriptionTranscribeAudioResponse>
    >;
    polishVoiceText(
      params: protocol.VoiceTranscriptionPolishTextParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.VoiceTranscriptionPolishTextResponse>
    >;
    readUsageStats(
      params: protocol.UsageStatsRangeParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.UsageStatsReadResponse>>;
    listUsageStatsModelRanking(
      params: protocol.UsageStatsRangeParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.UsageStatsModelRankingListResponse>
    >;
    listUsageStatsDailyTrends(
      params: protocol.UsageStatsRangeParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.UsageStatsDailyTrendsListResponse>
    >;
    readArtifacts(
      params: protocol.ArtifactReadParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ArtifactReadResponse>>;
    writeArtifact(
      params: protocol.ArtifactWriteParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ArtifactWriteResponse>>;
    listDirectory(
      params: protocol.FileSystemListDirectoryParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.FileSystemDirectoryListing>>;
    readFilePreview(
      params: protocol.FileSystemReadFilePreviewParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.FileSystemFilePreview>>;
    createFile(
      params: protocol.FileSystemCreateFileParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.FileSystemMutationResponse>>;
    createDirectory(
      params: protocol.FileSystemCreateDirectoryParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.FileSystemMutationResponse>>;
    renameFile(
      params: protocol.FileSystemRenameFileParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.FileSystemMutationResponse>>;
    deleteFile(
      params: protocol.FileSystemDeleteFileParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.FileSystemMutationResponse>>;
    readProjectGitStatus(
      params: protocol.ProjectGitStatusParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ProjectGitStatusResponse>>;
    readProjectGitDiff(
      params: protocol.ProjectGitDiffParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ProjectGitDiffResponse>>;
    listProjectGitCommits(
      params: protocol.ProjectGitCommitListParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ProjectGitCommitListResponse>>;
    checkoutProjectGitBranch(
      params: protocol.ProjectGitBranchCheckoutParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.ProjectGitBranchCheckoutResponse>
    >;
    createProjectGitBranch(
      params: protocol.ProjectGitBranchCreateParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ProjectGitBranchCreateResponse>>;
    createProjectGitWorktree(
      params: protocol.ProjectGitWorktreeCreateParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.ProjectGitWorktreeCreateResponse>
    >;
    startProjectShellSession(
      params: protocol.ProjectShellSessionStartParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.ProjectShellSessionStartResponse>
    >;
    writeProjectShellSession(
      params: protocol.ProjectShellSessionWriteParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ProjectShellEmptyResponse>>;
    resizeProjectShellSession(
      params: protocol.ProjectShellSessionResizeParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ProjectShellEmptyResponse>>;
    killProjectShellSession(
      params: protocol.ProjectShellSessionKillParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ProjectShellEmptyResponse>>;
    drainProjectShellSessionEvents(
      params?: protocol.ProjectShellSessionDrainEventsParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.ProjectShellSessionDrainEventsResponse>
    >;
    startExecutionProcess(
      params: protocol.ExecutionProcessStartParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ExecutionProcessStartResponse>>;
    writeExecutionProcessStdin(
      params: protocol.ExecutionProcessWriteStdinParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ExecutionProcessEmptyResponse>>;
    interruptExecutionProcess(
      params: protocol.ExecutionProcessIdParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ExecutionProcessStatusResponse>>;
    terminateExecutionProcess(
      params: protocol.ExecutionProcessIdParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ExecutionProcessStatusResponse>>;
    readExecutionProcessStatus(
      params: protocol.ExecutionProcessIdParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ExecutionProcessStatusResponse>>;
    drainExecutionProcessOutput(
      params?: protocol.ExecutionProcessDrainOutputParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.ExecutionProcessDrainOutputResponse>
    >;
    exportEvidence(
      params: protocol.EvidenceExportParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.EvidenceExportResponse>>;
    exportHandoffBundle(
      params: protocol.AgentSessionHandoffBundleExportParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.AgentSessionHandoffBundleExportResponse>
    >;
    exportReplayCase(
      params: protocol.AgentSessionReplayCaseExportParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.AgentSessionReplayCaseExportResponse>
    >;
    exportAnalysisHandoff(
      params: protocol.AgentSessionAnalysisHandoffExportParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.AgentSessionAnalysisHandoffExportResponse>
    >;
    exportReviewDecisionTemplate(
      params: protocol.AgentSessionReviewDecisionTemplateExportParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.AgentSessionReviewDecisionTemplateExportResponse>
    >;
    saveReviewDecision(
      params: protocol.AgentSessionReviewDecisionSaveParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.AgentSessionReviewDecisionTemplateExportResponse>
    >;
    readSession(
      params: protocol.AgentSessionReadParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.AgentSessionReadResponse>>;
    readAgentSessionMedia(
      params: protocol.AgentSessionMediaReadParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.AgentSessionMediaReadResponse>>;
    readAgentSessionToolInventory(
      params?: protocol.AgentSessionToolInventoryReadParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.AgentSessionToolInventoryReadResponse>
    >;
    listModels(
      params?: protocol.ModelListParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ModelListResponse>>;
    listModelPreferences(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ModelPreferencesListResponse>>;
    readModelSyncState(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ModelSyncStateReadResponse>>;
    listModelProviders(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ModelProviderListResponse>>;
    listModelProviderCatalog(
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.ModelProviderCatalogListResponse>
    >;
    readModelProviderAlias(
      params: protocol.ModelProviderAliasReadParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ModelProviderAliasReadResponse>>;
    listModelProviderAliases(
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ModelProviderAliasListResponse>>;
    resolveConnectDeepLink(
      params: protocol.ConnectDeepLinkResolveParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ConnectDeepLinkResolveResponse>>;
    resolveConnectOpenDeepLink(
      params: protocol.ConnectOpenDeepLinkResolveParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.ConnectOpenDeepLinkResolveResponse>
    >;
    saveConnectRelayApiKey(
      params: protocol.ConnectRelayApiKeySaveParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ConnectRelayApiKeySaveResponse>>;
    sendConnectCallback(
      params: protocol.ConnectCallbackSendParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.ConnectCallbackSendResponse>>;
    scanConversationImportSource(
      params?: protocol.ConversationImportSourceScanParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.ConversationImportSourceScanResponse>
    >;
    previewConversationImportThread(
      params: protocol.ConversationImportThreadPreviewParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.ConversationImportThreadPreviewResponse>
    >;
    commitConversationImportThread(
      params: protocol.ConversationImportThreadCommitParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.ConversationImportThreadCommitStartResponse>
    >;
    readConversationImportJob(
      params: protocol.ConversationImportJobReadParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.ConversationImportJobReadResponse>
    >;
    startTurn(
      params: protocol.TurnStartParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.TurnStartResponse>>;
    steerTurn(
      params: protocol.TurnSteerParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.TurnSteerResponse>>;
    cancelTurn(
      params: protocol.TurnInterruptParams,
      options?: AppServerRequestOptions,
    ): Promise<AppServerRequestResult<protocol.TurnInterruptResponse>>;
    respondAction(
      params: protocol.AgentSessionActionRespondParams,
      options?: AppServerRequestOptions,
    ): Promise<
      AppServerRequestResult<protocol.AgentSessionActionRespondResponse>
    >;
  }
}

const CONNECTION_CLIENT_METHOD_EXCLUSIONS = new Set<string>([
  "initialize",
  "listBrowserSessionTargets",
  "openBrowserSession",
  "readBrowserSession",
  "closeBrowserSession",
  "listBrowserSessionEvents",
  "executeBrowserSessionAction",
  "readWorkflow",
  "cancelWorkflow",
  "retryWorkflow",
  "respondWorkflow",
  "readModelProvider",
  "createModelProvider",
  "updateModelProvider",
  "deleteModelProvider",
  "updateModelProviderSortOrders",
  "exportModelProviderConfig",
  "importModelProviderConfig",
  "testModelProviderConnection",
  "testModelProviderChat",
  "fetchModelProviderModels",
  "createModelProviderKey",
  "updateModelProviderKey",
  "deleteModelProviderKey",
  "readModelProviderUiState",
  "writeModelProviderUiState",
]);

const CONNECTION_METHODS: readonly ConnectionMethodSpec[] =
  APP_SERVER_REQUEST_CLIENT_METHODS.filter(
    (spec) =>
      spec.kind === "request" &&
      !CONNECTION_CLIENT_METHOD_EXCLUSIONS.has(spec.name),
  ).map((spec) => ({
    name: spec.name,
    clientMethod: spec.name,
    method: spec.method,
    params: spec.params,
  }));

type ConnectionRuntime = AppServerConnection & {
  readonly client: AppServerClient;
  request<T>(
    requestMessage: protocol.JsonRpcRequest,
    method?: string,
    options?: AppServerRequestOptions,
  ): Promise<AppServerRequestResult<T>>;
};

type ResolvedArgs = {
  clientArgs: unknown[];
  options: AppServerRequestOptions;
};

function resolveArgs(
  mode: ConnectionParamsMode,
  args: IArguments,
): ResolvedArgs {
  if (mode === "none") {
    return {
      clientArgs: [],
      options: (args[0] as AppServerRequestOptions | undefined) ?? {},
    };
  }
  if (mode === "optional-empty") {
    return {
      clientArgs: [args.length === 0 || args[0] === undefined ? {} : args[0]],
      options: (args[1] as AppServerRequestOptions | undefined) ?? {},
    };
  }
  return {
    clientArgs: [args[0]],
    options: (args[1] as AppServerRequestOptions | undefined) ?? {},
  };
}

export function installAppServerConnectionMethods(
  prototype: AppServerConnection,
): void {
  for (const spec of CONNECTION_METHODS) {
    Object.defineProperty(prototype, spec.name, {
      configurable: true,
      writable: true,
      value: async function (
        this: ConnectionRuntime,
      ): Promise<AppServerRequestResult<unknown>> {
        const { clientArgs, options } = resolveArgs(spec.params, arguments);
        const client = this.client as unknown as Record<
          string,
          (...args: unknown[]) => protocol.JsonRpcRequest
        >;
        const requestMessage = client[spec.clientMethod](...clientArgs);
        return await this.request(requestMessage, spec.method, options);
      },
    });
  }
}
