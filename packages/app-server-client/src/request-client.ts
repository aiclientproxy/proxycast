import * as protocol from "./protocol.js";
import { installAppServerRequestClientMethods } from "./request-client-methods.js";

export class AppServerClient {
  #nextRequestId: number;

  constructor(options: { initialRequestId?: number } = {}) {
    this.#nextRequestId = options.initialRequestId ?? 1;
  }

  request(method: string, params: unknown): protocol.JsonRpcRequest {
    return protocol.request(this.nextId(), method, params);
  }

  nextId(): protocol.RequestId {
    const id = this.#nextRequestId;
    this.#nextRequestId += 1;
    return id;
  }
}

export interface AppServerClient {
  initialize(params: protocol.InitializeParams): protocol.JsonRpcRequest;
  initialized(): protocol.JsonRpcNotification;
  listCapabilities(
    params?: protocol.CapabilityListParams,
  ): protocol.JsonRpcRequest;
  listSessions(
    params?: protocol.AgentSessionListParams,
  ): protocol.JsonRpcRequest;
  readThread(params: protocol.ThreadReadParams): protocol.JsonRpcRequest;
  listThreads(params?: protocol.ThreadListParams): protocol.JsonRpcRequest;
  moveThreadToSection(
    params: protocol.ThreadSectionMoveParams,
  ): protocol.JsonRpcRequest;
  listThreadSections(
    params?: protocol.ThreadSectionListParams,
  ): protocol.JsonRpcRequest;
  createThreadSection(
    params: protocol.ThreadSectionCreateParams,
  ): protocol.JsonRpcRequest;
  updateThreadSection(
    params: protocol.ThreadSectionUpdateParams,
  ): protocol.JsonRpcRequest;
  deleteThreadSection(
    params: protocol.ThreadSectionDeleteParams,
  ): protocol.JsonRpcRequest;
  listLoadedThreads(
    params?: protocol.ThreadLoadedListParams,
  ): protocol.JsonRpcRequest;
  unsubscribeThread(
    params: protocol.ThreadUnsubscribeParams,
  ): protocol.JsonRpcRequest;
  incrementThreadElicitation(
    params: protocol.ThreadIncrementElicitationParams,
  ): protocol.JsonRpcRequest;
  decrementThreadElicitation(
    params: protocol.ThreadDecrementElicitationParams,
  ): protocol.JsonRpcRequest;
  approveGuardianDeniedAction(
    params: protocol.ThreadApproveGuardianDeniedActionParams,
  ): protocol.JsonRpcRequest;
  injectThreadItems(
    params: protocol.ThreadInjectItemsParams,
  ): protocol.JsonRpcRequest;
  archiveThread(params: protocol.ThreadArchiveParams): protocol.JsonRpcRequest;
  unarchiveThread(
    params: protocol.ThreadUnarchiveParams,
  ): protocol.JsonRpcRequest;
  setThreadName(params: protocol.ThreadSetNameParams): protocol.JsonRpcRequest;
  updateThreadMetadata(
    params: protocol.ThreadMetadataUpdateParams,
  ): protocol.JsonRpcRequest;
  listThreadTurns(
    params: protocol.ThreadTurnsListParams,
  ): protocol.JsonRpcRequest;
  listThreadItems(
    params: protocol.ThreadItemsListParams,
  ): protocol.JsonRpcRequest;
  searchThreads(params: protocol.ThreadSearchParams): protocol.JsonRpcRequest;
  searchThreadOccurrences(
    params: protocol.ThreadSearchOccurrencesParams,
  ): protocol.JsonRpcRequest;
  cleanThreadBackgroundTerminals(
    params: protocol.ThreadBackgroundTerminalsCleanParams,
  ): protocol.JsonRpcRequest;
  listThreadBackgroundTerminals(
    params: protocol.ThreadBackgroundTerminalsListParams,
  ): protocol.JsonRpcRequest;
  terminateThreadBackgroundTerminal(
    params: protocol.ThreadBackgroundTerminalsTerminateParams,
  ): protocol.JsonRpcRequest;
  updateThreadSettings(
    params: protocol.ThreadSettingsUpdateParams,
  ): protocol.JsonRpcRequest;
  setThreadMemoryMode(
    params: protocol.ThreadMemoryModeSetParams,
  ): protocol.JsonRpcRequest;
  runThreadShellCommand(
    params: protocol.ThreadShellCommandParams,
  ): protocol.JsonRpcRequest;
  deleteThread(params: protocol.ThreadDeleteParams): protocol.JsonRpcRequest;
  startThreadCompaction(
    params: protocol.ThreadCompactStartParams,
  ): protocol.JsonRpcRequest;
  resumeThread(params: protocol.ThreadResumeParams): protocol.JsonRpcRequest;
  listAgentSessionFileCheckpoints(
    params: protocol.AgentSessionFileCheckpointListParams,
  ): protocol.JsonRpcRequest;
  getAgentSessionFileCheckpoint(
    params: protocol.AgentSessionFileCheckpointGetParams,
  ): protocol.JsonRpcRequest;
  diffAgentSessionFileCheckpoint(
    params: protocol.AgentSessionFileCheckpointDiffParams,
  ): protocol.JsonRpcRequest;
  restoreAgentSessionFileCheckpoint(
    params: protocol.AgentSessionFileCheckpointRestoreParams,
  ): protocol.JsonRpcRequest;
  getOrCreateSessionFile(
    params: protocol.SessionFileGetOrCreateParams,
  ): protocol.JsonRpcRequest;
  updateSessionFileMeta(
    params: protocol.SessionFileUpdateMetaParams,
  ): protocol.JsonRpcRequest;
  saveSessionFile(
    params: protocol.SessionFileSaveParams,
  ): protocol.JsonRpcRequest;
  readSessionFile(
    params: protocol.SessionFileIdParams,
  ): protocol.JsonRpcRequest;
  resolveSessionFilePath(
    params: protocol.SessionFileIdParams,
  ): protocol.JsonRpcRequest;
  deleteSessionFile(
    params: protocol.SessionFileIdParams,
  ): protocol.JsonRpcRequest;
  listSessionFiles(
    params: protocol.SessionFileGetOrCreateParams,
  ): protocol.JsonRpcRequest;
  listWorkspaces(): protocol.JsonRpcRequest;
  readWorkspace(params: protocol.WorkspaceReadParams): protocol.JsonRpcRequest;
  updateWorkspace(
    params: protocol.WorkspaceUpdateParams,
  ): protocol.JsonRpcRequest;
  deleteWorkspace(
    params: protocol.WorkspaceDeleteParams,
  ): protocol.JsonRpcRequest;
  ensureWorkspace(
    params: protocol.WorkspaceEnsureProjectParams,
  ): protocol.JsonRpcRequest;
  readWorkspaceByPath(
    params: protocol.WorkspacePathReadParams,
  ): protocol.JsonRpcRequest;
  readDefaultWorkspace(): protocol.JsonRpcRequest;
  ensureDefaultWorkspace(): protocol.JsonRpcRequest;
  readWorkspaceProjectsRoot(): protocol.JsonRpcRequest;
  resolveWorkspaceProjectPath(
    params: protocol.WorkspaceProjectPathResolveParams,
  ): protocol.JsonRpcRequest;
  ensureWorkspaceReady(
    params: protocol.WorkspaceEnsureParams,
  ): protocol.JsonRpcRequest;
  requestWorkspaceRightSurface(
    params: protocol.WorkspaceRightSurfaceRequestParams,
  ): protocol.JsonRpcRequest;
  listWorkspaceRightSurfacePending(
    params?: protocol.WorkspaceRightSurfacePendingListParams,
  ): protocol.JsonRpcRequest;
  consumeWorkspaceRightSurfacePending(
    params: protocol.WorkspaceRightSurfacePendingConsumeParams,
  ): protocol.JsonRpcRequest;
  dismissWorkspaceRightSurfacePending(
    params: protocol.WorkspaceRightSurfacePendingDismissParams,
  ): protocol.JsonRpcRequest;
  listBrowserSessionTargets(
    params: protocol.BrowserSessionTargetListParams,
  ): protocol.JsonRpcRequest;
  openBrowserSession(
    params: protocol.BrowserSessionOpenParams,
  ): protocol.JsonRpcRequest;
  readBrowserSession(
    params: protocol.BrowserSessionIdParams,
  ): protocol.JsonRpcRequest;
  closeBrowserSession(
    params: protocol.BrowserSessionIdParams,
  ): protocol.JsonRpcRequest;
  listBrowserSessionEvents(
    params: protocol.BrowserSessionEventListParams,
  ): protocol.JsonRpcRequest;
  executeBrowserSessionAction(
    params: protocol.BrowserSessionActionExecuteParams,
  ): protocol.JsonRpcRequest;
  listApps(params?: protocol.AppsListParams): protocol.JsonRpcRequest;
  readApps(params: protocol.AppsReadParams): protocol.JsonRpcRequest;
  listInstalledApps(
    params?: protocol.AppsInstalledParams,
  ): protocol.JsonRpcRequest;
  listPluginCatalog(
    params?: protocol.PluginCatalogListParams,
  ): protocol.JsonRpcRequest;
  searchPlugins(params: protocol.PluginSearchParams): protocol.JsonRpcRequest;
  setPluginCatalogEnabled(
    params: protocol.PluginCatalogEnabledSetParams,
  ): protocol.JsonRpcRequest;
  readPluginCatalog(
    params: protocol.PluginCatalogReadParams,
  ): protocol.JsonRpcRequest;
  installPluginCatalog(
    params: protocol.PluginCatalogInstallParams,
  ): protocol.JsonRpcRequest;
  uninstallPluginCatalog(
    params: protocol.PluginCatalogUninstallParams,
  ): protocol.JsonRpcRequest;
  listInstalledPluginCatalog(
    params?: protocol.PluginCatalogInstalledParams,
  ): protocol.JsonRpcRequest;
  listSkills(params?: protocol.SkillsListParams): protocol.JsonRpcRequest;
  setSkillsExtraRoots(
    params: protocol.SkillsExtraRootsSetParams,
  ): protocol.JsonRpcRequest;
  writeSkillsConfig(
    params: protocol.SkillsConfigWriteParams,
  ): protocol.JsonRpcRequest;
  listHooks(params?: protocol.HooksListParams): protocol.JsonRpcRequest;
  readSkill(params: protocol.SkillReadParams): protocol.JsonRpcRequest;
  listManagementSkills(
    params: protocol.SkillManagementListParams,
  ): protocol.JsonRpcRequest;
  installManagementSkill(
    params: protocol.SkillManagementInstallParams,
  ): protocol.JsonRpcRequest;
  uninstallManagementSkill(
    params: protocol.SkillManagementUninstallParams,
  ): protocol.JsonRpcRequest;
  listSkillRepositories(): protocol.JsonRpcRequest;
  saveSkillRepository(
    params: protocol.SkillRepositorySaveParams,
  ): protocol.JsonRpcRequest;
  deleteSkillRepository(
    params: protocol.SkillRepositoryDeleteParams,
  ): protocol.JsonRpcRequest;
  refreshSkillCache(): protocol.JsonRpcRequest;
  listInstalledSkillDirectories(): protocol.JsonRpcRequest;
  inspectLocalSkill(
    params: protocol.SkillLocalInspectParams,
  ): protocol.JsonRpcRequest;
  inspectLocalSkillDetail(
    params: protocol.SkillLocalDetailInspectParams,
  ): protocol.JsonRpcRequest;
  createSkillScaffold(
    params: protocol.SkillScaffoldCreateParams,
  ): protocol.JsonRpcRequest;
  importLocalSkill(
    params: protocol.SkillLocalImportParams,
  ): protocol.JsonRpcRequest;
  renameLocalSkill(
    params: protocol.SkillLocalRenameParams,
  ): protocol.JsonRpcRequest;
  inspectRemoteSkill(
    params: protocol.SkillRemoteInspectParams,
  ): protocol.JsonRpcRequest;
  inspectLocalSkillPackage(
    params: protocol.SkillPackageLocalInspectParams,
  ): protocol.JsonRpcRequest;
  installLocalSkillPackage(
    params: protocol.SkillPackageLocalInstallParams,
  ): protocol.JsonRpcRequest;
  replaceLocalSkillPackage(
    params: protocol.SkillPackageLocalReplaceParams,
  ): protocol.JsonRpcRequest;
  exportSkillPackage(
    params: protocol.SkillPackageExportParams,
  ): protocol.JsonRpcRequest;
  installMarketplaceSkill(
    params: protocol.SkillMarketplaceInstallParams,
  ): protocol.JsonRpcRequest;
  installSkillFromDownload(
    params: protocol.SkillDownloadInstallParams,
  ): protocol.JsonRpcRequest;
  listWorkspaceSkillBindings(
    params: protocol.WorkspaceSkillBindingsListParams,
  ): protocol.JsonRpcRequest;
  listWorkspaceRegisteredSkills(
    params: protocol.WorkspaceRegisteredSkillsListParams,
  ): protocol.JsonRpcRequest;
  listKnowledgePacks(
    params: protocol.KnowledgeListPacksParams,
  ): protocol.JsonRpcRequest;
  readKnowledgePack(
    params: protocol.KnowledgeReadPackParams,
  ): protocol.JsonRpcRequest;
  importKnowledgeSource(
    params: protocol.KnowledgeImportSourceParams,
  ): protocol.JsonRpcRequest;
  compileKnowledgePack(
    params: protocol.KnowledgeCompilePackParams,
  ): protocol.JsonRpcRequest;
  setDefaultKnowledgePack(
    params: protocol.KnowledgeSetDefaultPackParams,
  ): protocol.JsonRpcRequest;
  updateKnowledgePackStatus(
    params: protocol.KnowledgeUpdatePackStatusParams,
  ): protocol.JsonRpcRequest;
  resolveKnowledgeContext(
    params: protocol.KnowledgeResolveContextParams,
  ): protocol.JsonRpcRequest;
  validateKnowledgeContextRun(
    params: protocol.KnowledgeValidateContextRunParams,
  ): protocol.JsonRpcRequest;
  listAutomationJobs(): protocol.JsonRpcRequest;
  readAutomationSchedulerConfig(): protocol.JsonRpcRequest;
  updateAutomationSchedulerConfig(
    params: protocol.AutomationSchedulerConfigUpdateParams,
  ): protocol.JsonRpcRequest;
  readAutomationSchedulerStatus(): protocol.JsonRpcRequest;
  readAutomationJob(
    params: protocol.AutomationJobIdParams,
  ): protocol.JsonRpcRequest;
  createAutomationJob(
    params: protocol.AutomationJobCreateParams,
  ): protocol.JsonRpcRequest;
  updateAutomationJob(
    params: protocol.AutomationJobUpdateParams,
  ): protocol.JsonRpcRequest;
  deleteAutomationJob(
    params: protocol.AutomationJobIdParams,
  ): protocol.JsonRpcRequest;
  runAutomationJobNow(
    params: protocol.AutomationJobIdParams,
  ): protocol.JsonRpcRequest;
  readAutomationHealth(
    params?: protocol.AutomationJobHealthParams,
  ): protocol.JsonRpcRequest;
  readAutomationRunHistory(
    params: protocol.AutomationJobRunHistoryParams,
  ): protocol.JsonRpcRequest;
  previewAutomationSchedule(
    params: protocol.AutomationScheduleParams,
  ): protocol.JsonRpcRequest;
  validateAutomationSchedule(
    params: protocol.AutomationScheduleParams,
  ): protocol.JsonRpcRequest;
  listMcpServers(): protocol.JsonRpcRequest;
  listMcpServersWithStatus(): protocol.JsonRpcRequest;
  createMcpServer(
    params: protocol.McpServerCreateParams,
  ): protocol.JsonRpcRequest;
  updateMcpServer(
    params: protocol.McpServerUpdateParams,
  ): protocol.JsonRpcRequest;
  deleteMcpServer(
    params: protocol.McpServerDeleteParams,
  ): protocol.JsonRpcRequest;
  setMcpServerEnabled(
    params: protocol.McpServerEnabledSetParams,
  ): protocol.JsonRpcRequest;
  importMcpServersFromApp(
    params: protocol.McpServerImportFromAppParams,
  ): protocol.JsonRpcRequest;
  syncAllMcpServersToLive(): protocol.JsonRpcRequest;
  loginMcpServerOauth(
    params: protocol.McpServerOauthLoginParams,
  ): protocol.JsonRpcRequest;
  startMcpServer(
    params: protocol.McpServerStartParams,
  ): protocol.JsonRpcRequest;
  stopMcpServer(params: protocol.McpServerStopParams): protocol.JsonRpcRequest;
  listMcpTools(): protocol.JsonRpcRequest;
  listMcpToolsForContext(
    params: protocol.McpToolListForContextParams,
  ): protocol.JsonRpcRequest;
  searchMcpTools(params: protocol.McpToolSearchParams): protocol.JsonRpcRequest;
  callMcpServerTool(
    params: protocol.McpServerToolCallParams,
  ): protocol.JsonRpcRequest;
  listMcpPrompts(): protocol.JsonRpcRequest;
  getMcpPrompt(params: protocol.McpPromptGetParams): protocol.JsonRpcRequest;
  listMcpResources(): protocol.JsonRpcRequest;
  readMcpServerResource(
    params: protocol.McpServerResourceReadParams,
  ): protocol.JsonRpcRequest;
  subscribeMcpResource(
    params: protocol.McpResourceSubscribeParams,
  ): protocol.JsonRpcRequest;
  unsubscribeMcpResource(
    params: protocol.McpResourceUnsubscribeParams,
  ): protocol.JsonRpcRequest;
  readProjectMemory(
    params: protocol.ProjectMemoryReadParams,
  ): protocol.JsonRpcRequest;
  listMemoryStore(
    params: protocol.MemoryStoreListParams,
  ): protocol.JsonRpcRequest;
  readMemoryStore(
    params: protocol.MemoryStoreReadParams,
  ): protocol.JsonRpcRequest;
  searchMemoryStore(
    params: protocol.MemoryStoreSearchParams,
  ): protocol.JsonRpcRequest;
  addMemoryStoreNote(
    params: protocol.MemoryStoreAddNoteParams,
  ): protocol.JsonRpcRequest;
  consolidateMemoryStore(
    params: protocol.MemoryStoreConsolidateParams,
  ): protocol.JsonRpcRequest;
  listMemoryStoreReviewNotes(
    params: protocol.MemoryStoreReviewListParams,
  ): protocol.JsonRpcRequest;
  resolveMemoryStoreReviewNote(
    params: protocol.MemoryStoreReviewResolveParams,
  ): protocol.JsonRpcRequest;
  healthMemoryStore(
    params: protocol.MemoryStoreRootParams,
  ): protocol.JsonRpcRequest;
  resetMemory(): protocol.JsonRpcRequest;
  rebuildMemoryStoreIndex(
    params: protocol.MemoryStoreRootParams,
  ): protocol.JsonRpcRequest;
  listLogs(): protocol.JsonRpcRequest;
  readPersistedLogTail(
    params: protocol.LogPersistedTailParams,
  ): protocol.JsonRpcRequest;
  clearLogs(): protocol.JsonRpcRequest;
  clearDiagnosticLogHistory(): protocol.JsonRpcRequest;
  readLogStorageDiagnostics(): protocol.JsonRpcRequest;
  exportSupportBundle(
    params?: protocol.SupportBundleExportParams,
  ): protocol.JsonRpcRequest;
  readServerDiagnostics(): protocol.JsonRpcRequest;
  readWindowsStartupDiagnostics(): protocol.JsonRpcRequest;
  listDiagnosticsTraces(
    params: protocol.DiagnosticsTraceListParams,
  ): protocol.JsonRpcRequest;
  readDiagnosticsTrace(
    params: protocol.DiagnosticsTraceReadParams,
  ): protocol.JsonRpcRequest;
  exportDiagnosticsTrace(
    params: protocol.DiagnosticsTraceExportParams,
  ): protocol.JsonRpcRequest;
  readGatewayChannelStatus(
    params: protocol.GatewayChannelStatusParams,
  ): protocol.JsonRpcRequest;
  startGatewayChannel(
    params: protocol.GatewayChannelStartParams,
  ): protocol.JsonRpcRequest;
  stopGatewayChannel(
    params: protocol.GatewayChannelStopParams,
  ): protocol.JsonRpcRequest;
  probeTelegramChannel(
    params?: protocol.ChannelProbeParams,
  ): protocol.JsonRpcRequest;
  probeFeishuChannel(
    params?: protocol.ChannelProbeParams,
  ): protocol.JsonRpcRequest;
  probeDiscordChannel(
    params?: protocol.ChannelProbeParams,
  ): protocol.JsonRpcRequest;
  probeWechatChannel(
    params?: protocol.ChannelProbeParams,
  ): protocol.JsonRpcRequest;
  startWechatChannelLogin(
    params?: protocol.WechatLoginStartParams,
  ): protocol.JsonRpcRequest;
  waitWechatChannelLogin(
    params: protocol.WechatLoginWaitParams,
  ): protocol.JsonRpcRequest;
  listWechatChannelAccounts(): protocol.JsonRpcRequest;
  removeWechatChannelAccount(
    params: protocol.WechatChannelAccountRemoveParams,
  ): protocol.JsonRpcRequest;
  setWechatChannelRuntimeModel(
    params: protocol.WechatRuntimeModelSetParams,
  ): protocol.JsonRpcRequest;
  probeGatewayTunnel(): protocol.JsonRpcRequest;
  detectGatewayTunnelCloudflared(): protocol.JsonRpcRequest;
  installGatewayTunnelCloudflared(
    params: protocol.GatewayTunnelCloudflaredInstallParams,
  ): protocol.JsonRpcRequest;
  createGatewayTunnel(
    params: protocol.GatewayTunnelCreateParams,
  ): protocol.JsonRpcRequest;
  startGatewayTunnel(): protocol.JsonRpcRequest;
  stopGatewayTunnel(): protocol.JsonRpcRequest;
  restartGatewayTunnel(): protocol.JsonRpcRequest;
  readGatewayTunnelStatus(): protocol.JsonRpcRequest;
  syncGatewayTunnelWebhookUrl(
    params: protocol.GatewayTunnelSyncWebhookUrlParams,
  ): protocol.JsonRpcRequest;
  createImageMediaTaskArtifact(
    params: protocol.MediaTaskArtifactImageCreateParams,
  ): protocol.JsonRpcRequest;
  createAudioMediaTaskArtifact(
    params: protocol.MediaTaskArtifactAudioCreateParams,
  ): protocol.JsonRpcRequest;
  createVideoMediaTaskArtifact(
    params: protocol.MediaTaskArtifactVideoCreateParams,
  ): protocol.JsonRpcRequest;
  completeImageMediaTaskArtifact(
    params: protocol.MediaTaskArtifactImageCompleteParams,
  ): protocol.JsonRpcRequest;
  completeAudioMediaTaskArtifact(
    params: protocol.MediaTaskArtifactAudioCompleteParams,
  ): protocol.JsonRpcRequest;
  getMediaTaskArtifact(
    params: protocol.MediaTaskArtifactLookupParams,
  ): protocol.JsonRpcRequest;
  listMediaTaskArtifacts(
    params: protocol.MediaTaskArtifactListParams,
  ): protocol.JsonRpcRequest;
  cancelMediaTaskArtifact(
    params: protocol.MediaTaskArtifactLookupParams,
  ): protocol.JsonRpcRequest;
  getGalleryMaterial(
    params: protocol.GalleryMaterialLookupParams,
  ): protocol.JsonRpcRequest;
  createGalleryMaterialMetadata(
    params: protocol.GalleryMaterialMetadataCreateParams,
  ): protocol.JsonRpcRequest;
  getGalleryMaterialMetadata(
    params: protocol.GalleryMaterialLookupParams,
  ): protocol.JsonRpcRequest;
  updateGalleryMaterialMetadata(
    params: protocol.GalleryMaterialMetadataUpdateParams,
  ): protocol.JsonRpcRequest;
  deleteGalleryMaterialMetadata(
    params: protocol.GalleryMaterialLookupParams,
  ): protocol.JsonRpcRequest;
  listGalleryMaterialsByImageCategory(
    params: protocol.GalleryMaterialFilterParams,
  ): protocol.JsonRpcRequest;
  listGalleryMaterialsByLayoutCategory(
    params: protocol.GalleryMaterialFilterParams,
  ): protocol.JsonRpcRequest;
  listGalleryMaterialsByMood(
    params: protocol.GalleryMaterialFilterParams,
  ): protocol.JsonRpcRequest;
  listProjectMaterials(
    params: protocol.ProjectMaterialListParams,
  ): protocol.JsonRpcRequest;
  getProjectMaterial(
    params: protocol.ProjectMaterialLookupParams,
  ): protocol.JsonRpcRequest;
  countProjectMaterials(
    params: protocol.ProjectMaterialListParams,
  ): protocol.JsonRpcRequest;
  uploadProjectMaterial(
    params: protocol.ProjectMaterialUploadParams,
  ): protocol.JsonRpcRequest;
  importProjectMaterialFromUrl(
    params: protocol.ProjectMaterialImportFromUrlParams,
  ): protocol.JsonRpcRequest;
  updateProjectMaterial(
    params: protocol.ProjectMaterialUpdateParams,
  ): protocol.JsonRpcRequest;
  deleteProjectMaterial(
    params: protocol.ProjectMaterialLookupParams,
  ): protocol.JsonRpcRequest;
  readProjectMaterialContent(
    params: protocol.ProjectMaterialLookupParams,
  ): protocol.JsonRpcRequest;
  listVoiceAsrCredentials(): protocol.JsonRpcRequest;
  createVoiceAsrCredential(
    params: protocol.VoiceAsrCredentialCreateParams,
  ): protocol.JsonRpcRequest;
  updateVoiceAsrCredential(
    params: protocol.VoiceAsrCredentialUpdateParams,
  ): protocol.JsonRpcRequest;
  deleteVoiceAsrCredential(
    params: protocol.VoiceAsrCredentialIdParams,
  ): protocol.JsonRpcRequest;
  setDefaultVoiceAsrCredential(
    params: protocol.VoiceAsrCredentialIdParams,
  ): protocol.JsonRpcRequest;
  testVoiceAsrCredential(
    params: protocol.VoiceAsrCredentialIdParams,
  ): protocol.JsonRpcRequest;
  listVoiceInstructions(): protocol.JsonRpcRequest;
  saveVoiceInstruction(
    params: protocol.VoiceInstructionSaveParams,
  ): protocol.JsonRpcRequest;
  deleteVoiceInstruction(
    params: protocol.VoiceInstructionIdParams,
  ): protocol.JsonRpcRequest;
  setDefaultVoiceModel(
    params: protocol.VoiceModelDefaultSetParams,
  ): protocol.JsonRpcRequest;
  testTranscribeVoiceModelFile(
    params: protocol.VoiceModelTestTranscribeFileParams,
  ): protocol.JsonRpcRequest;
  transcribeVoiceAudio(
    params: protocol.VoiceTranscriptionTranscribeAudioParams,
  ): protocol.JsonRpcRequest;
  polishVoiceText(
    params: protocol.VoiceTranscriptionPolishTextParams,
  ): protocol.JsonRpcRequest;
  readUsageStats(
    params: protocol.UsageStatsRangeParams,
  ): protocol.JsonRpcRequest;
  listUsageStatsModelRanking(
    params: protocol.UsageStatsRangeParams,
  ): protocol.JsonRpcRequest;
  listUsageStatsDailyTrends(
    params: protocol.UsageStatsRangeParams,
  ): protocol.JsonRpcRequest;
  readArtifacts(params: protocol.ArtifactReadParams): protocol.JsonRpcRequest;
  writeArtifact(params: protocol.ArtifactWriteParams): protocol.JsonRpcRequest;
  readFile(params: protocol.FsReadFileParams): protocol.JsonRpcRequest;
  writeFile(params: protocol.FsWriteFileParams): protocol.JsonRpcRequest;
  createDirectory(
    params: protocol.FsCreateDirectoryParams,
  ): protocol.JsonRpcRequest;
  getMetadata(params: protocol.FsGetMetadataParams): protocol.JsonRpcRequest;
  readDirectory(
    params: protocol.FsReadDirectoryParams,
  ): protocol.JsonRpcRequest;
  remove(params: protocol.FsRemoveParams): protocol.JsonRpcRequest;
  copy(params: protocol.FsCopyParams): protocol.JsonRpcRequest;
  watch(params: protocol.FsWatchParams): protocol.JsonRpcRequest;
  unwatch(params: protocol.FsUnwatchParams): protocol.JsonRpcRequest;
  readProjectGitStatus(
    params: protocol.ProjectGitStatusParams,
  ): protocol.JsonRpcRequest;
  readProjectGitDiff(
    params: protocol.ProjectGitDiffParams,
  ): protocol.JsonRpcRequest;
  listProjectGitCommits(
    params: protocol.ProjectGitCommitListParams,
  ): protocol.JsonRpcRequest;
  checkoutProjectGitBranch(
    params: protocol.ProjectGitBranchCheckoutParams,
  ): protocol.JsonRpcRequest;
  createProjectGitBranch(
    params: protocol.ProjectGitBranchCreateParams,
  ): protocol.JsonRpcRequest;
  createProjectGitWorktree(
    params: protocol.ProjectGitWorktreeCreateParams,
  ): protocol.JsonRpcRequest;
  exportEvidence(
    params: protocol.EvidenceExportParams,
  ): protocol.JsonRpcRequest;
  exportHandoffBundle(
    params: protocol.AgentSessionHandoffBundleExportParams,
  ): protocol.JsonRpcRequest;
  exportReplayCase(
    params: protocol.AgentSessionReplayCaseExportParams,
  ): protocol.JsonRpcRequest;
  exportAnalysisHandoff(
    params: protocol.AgentSessionAnalysisHandoffExportParams,
  ): protocol.JsonRpcRequest;
  exportReviewDecisionTemplate(
    params: protocol.AgentSessionReviewDecisionTemplateExportParams,
  ): protocol.JsonRpcRequest;
  saveReviewDecision(
    params: protocol.AgentSessionReviewDecisionSaveParams,
  ): protocol.JsonRpcRequest;
  startSession(params: protocol.ThreadStartParams): protocol.JsonRpcRequest;
  forkThread(params: protocol.ThreadForkParams): protocol.JsonRpcRequest;
  readSession(params: protocol.AgentSessionReadParams): protocol.JsonRpcRequest;
  readMedia(params: protocol.MediaReadParams): protocol.JsonRpcRequest;
  readWorkflow(params: protocol.WorkflowReadParams): protocol.JsonRpcRequest;
  cancelWorkflow(
    params: protocol.WorkflowCancelParams,
  ): protocol.JsonRpcRequest;
  retryWorkflow(params: protocol.WorkflowRetryParams): protocol.JsonRpcRequest;
  respondWorkflow(
    params: protocol.WorkflowRespondParams,
  ): protocol.JsonRpcRequest;
  readAgentSessionToolInventory(
    params?: protocol.AgentSessionToolInventoryReadParams,
  ): protocol.JsonRpcRequest;
  listModels(params?: protocol.ModelListParams): protocol.JsonRpcRequest;
  listModelPreferences(): protocol.JsonRpcRequest;
  readModelSyncState(): protocol.JsonRpcRequest;
  listModelProviders(): protocol.JsonRpcRequest;
  listModelProviderCatalog(): protocol.JsonRpcRequest;
  readModelProvider(
    params: protocol.ModelProviderReadParams,
  ): protocol.JsonRpcRequest;
  createModelProvider(
    params: protocol.ModelProviderCreateParams,
  ): protocol.JsonRpcRequest;
  updateModelProvider(
    params: protocol.ModelProviderUpdateParams,
  ): protocol.JsonRpcRequest;
  deleteModelProvider(
    params: protocol.ModelProviderDeleteParams,
  ): protocol.JsonRpcRequest;
  updateModelProviderSortOrders(
    params: protocol.ModelProviderSortOrdersUpdateParams,
  ): protocol.JsonRpcRequest;
  exportModelProviderConfig(
    params?: protocol.ModelProviderConfigExportParams,
  ): protocol.JsonRpcRequest;
  importModelProviderConfig(
    params: protocol.ModelProviderConfigImportParams,
  ): protocol.JsonRpcRequest;
  testModelProviderConnection(
    params: protocol.ModelProviderTestConnectionParams,
  ): protocol.JsonRpcRequest;
  testModelProviderChat(
    params: protocol.ModelProviderTestChatParams,
  ): protocol.JsonRpcRequest;
  fetchModelProviderModels(
    params: protocol.ModelProviderFetchModelsParams,
  ): protocol.JsonRpcRequest;
  createModelProviderKey(
    params: protocol.ModelProviderKeyCreateParams,
  ): protocol.JsonRpcRequest;
  updateModelProviderKey(
    params: protocol.ModelProviderKeyUpdateParams,
  ): protocol.JsonRpcRequest;
  deleteModelProviderKey(
    params: protocol.ModelProviderKeyDeleteParams,
  ): protocol.JsonRpcRequest;
  readModelProviderUiState(
    params: protocol.ModelProviderUiStateReadParams,
  ): protocol.JsonRpcRequest;
  writeModelProviderUiState(
    params: protocol.ModelProviderUiStateWriteParams,
  ): protocol.JsonRpcRequest;
  readModelProviderAlias(
    params: protocol.ModelProviderAliasReadParams,
  ): protocol.JsonRpcRequest;
  listModelProviderAliases(): protocol.JsonRpcRequest;
  resolveConnectDeepLink(
    params: protocol.ConnectDeepLinkResolveParams,
  ): protocol.JsonRpcRequest;
  resolveConnectOpenDeepLink(
    params: protocol.ConnectOpenDeepLinkResolveParams,
  ): protocol.JsonRpcRequest;
  saveConnectRelayApiKey(
    params: protocol.ConnectRelayApiKeySaveParams,
  ): protocol.JsonRpcRequest;
  sendConnectCallback(
    params: protocol.ConnectCallbackSendParams,
  ): protocol.JsonRpcRequest;
  scanConversationImportSource(
    params?: protocol.ConversationImportSourceScanParams,
  ): protocol.JsonRpcRequest;
  previewConversationImportThread(
    params: protocol.ConversationImportThreadPreviewParams,
  ): protocol.JsonRpcRequest;
  commitConversationImportThread(
    params: protocol.ConversationImportThreadCommitParams,
  ): protocol.JsonRpcRequest;
  readConversationImportJob(
    params: protocol.ConversationImportJobReadParams,
  ): protocol.JsonRpcRequest;
  spawnProcess(params: protocol.ProcessSpawnParams): protocol.JsonRpcRequest;
  writeProcessStdin(
    params: protocol.ProcessWriteStdinParams,
  ): protocol.JsonRpcRequest;
  resizeProcessPty(
    params: protocol.ProcessResizePtyParams,
  ): protocol.JsonRpcRequest;
  killProcess(params: protocol.ProcessKillParams): protocol.JsonRpcRequest;
  execCommand(params: protocol.CommandExecParams): protocol.JsonRpcRequest;
  writeCommandExec(
    params: protocol.CommandExecWriteParams,
  ): protocol.JsonRpcRequest;
  resizeCommandExec(
    params: protocol.CommandExecResizeParams,
  ): protocol.JsonRpcRequest;
  terminateCommandExec(
    params: protocol.CommandExecTerminateParams,
  ): protocol.JsonRpcRequest;
  startTurn(params: protocol.TurnStartParams): protocol.JsonRpcRequest;
  startReview(params: protocol.ReviewStartParams): protocol.JsonRpcRequest;
  steerTurn(params: protocol.TurnSteerParams): protocol.JsonRpcRequest;
  cancelTurn(params: protocol.TurnInterruptParams): protocol.JsonRpcRequest;
  respondAction(
    params: protocol.AgentSessionActionRespondParams,
  ): protocol.JsonRpcRequest;
}

installAppServerRequestClientMethods(AppServerClient.prototype);
