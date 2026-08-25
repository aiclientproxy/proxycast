import * as constants from "./appServerConstants";
export type AppServerClientMethodParamsMode =
  | "none"
  | "required"
  | "optional-empty";

export type AppServerClientMethodSpec = {
  name: string;
  method: string;
  params: AppServerClientMethodParamsMode;
};

export const APP_SERVER_CLIENT_METHODS: readonly AppServerClientMethodSpec[] = [
  {
    name: "startSession",
    method: constants.APP_SERVER_METHOD_THREAD_START,
    params: "required",
  },
  {
    name: "readWindowsSandboxReadiness",
    method: constants.APP_SERVER_METHOD_WINDOWS_SANDBOX_READINESS,
    params: "optional-empty",
  },
  {
    name: "startWindowsSandboxSetup",
    method: constants.APP_SERVER_METHOD_WINDOWS_SANDBOX_SETUP_START,
    params: "required",
  },
  {
    name: "readModelProviderCapabilities",
    method: constants.APP_SERVER_METHOD_MODEL_PROVIDER_CAPABILITIES_READ,
    params: "optional-empty",
  },
  {
    name: "addEnvironment",
    method: constants.APP_SERVER_METHOD_ENVIRONMENT_ADD,
    params: "required",
  },
  {
    name: "readEnvironmentInfo",
    method: constants.APP_SERVER_METHOD_ENVIRONMENT_INFO,
    params: "required",
  },
  {
    name: "readEnvironmentStatus",
    method: constants.APP_SERVER_METHOD_ENVIRONMENT_STATUS,
    params: "required",
  },
  {
    name: "forkThread",
    method: constants.APP_SERVER_METHOD_THREAD_FORK,
    params: "required",
  },
  {
    name: "revertThread",
    method: constants.APP_SERVER_METHOD_THREAD_REVERT,
    params: "required",
  },
  {
    name: "addThreadQueue",
    method: constants.APP_SERVER_METHOD_THREAD_QUEUE_ADD,
    params: "required",
  },
  {
    name: "listThreadQueue",
    method: constants.APP_SERVER_METHOD_THREAD_QUEUE_LIST,
    params: "required",
  },
  {
    name: "updateThreadQueue",
    method: constants.APP_SERVER_METHOD_THREAD_QUEUE_UPDATE,
    params: "required",
  },
  {
    name: "deleteThreadQueue",
    method: constants.APP_SERVER_METHOD_THREAD_QUEUE_DELETE,
    params: "required",
  },
  {
    name: "reorderThreadQueue",
    method: constants.APP_SERVER_METHOD_THREAD_QUEUE_REORDER,
    params: "required",
  },
  {
    name: "startThreadQueue",
    method: constants.APP_SERVER_METHOD_THREAD_QUEUE_START,
    params: "required",
  },
  {
    name: "listSessions",
    method: constants.APP_SERVER_METHOD_THREAD_LIST,
    params: "optional-empty",
  },
  {
    name: "moveThreadToSection",
    method: constants.APP_SERVER_METHOD_THREAD_SECTION_MOVE,
    params: "required",
  },
  {
    name: "listThreadSections",
    method: constants.APP_SERVER_METHOD_THREAD_SECTION_LIST,
    params: "optional-empty",
  },
  {
    name: "createThreadSection",
    method: constants.APP_SERVER_METHOD_THREAD_SECTION_CREATE,
    params: "required",
  },
  {
    name: "updateThreadSection",
    method: constants.APP_SERVER_METHOD_THREAD_SECTION_UPDATE,
    params: "required",
  },
  {
    name: "deleteThreadSection",
    method: constants.APP_SERVER_METHOD_THREAD_SECTION_DELETE,
    params: "required",
  },
  {
    name: "listThreadBackgroundTerminals",
    method: constants.APP_SERVER_METHOD_THREAD_BACKGROUND_TERMINALS_LIST,
    params: "required",
  },
  {
    name: "terminateThreadBackgroundTerminal",
    method: constants.APP_SERVER_METHOD_THREAD_BACKGROUND_TERMINALS_TERMINATE,
    params: "required",
  },
  {
    name: "listCapabilities",
    method: constants.APP_SERVER_METHOD_CAPABILITY_LIST,
    params: "optional-empty",
  },
  {
    name: "requestWorkspaceRightSurface",
    method: constants.APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST,
    params: "required",
  },
  {
    name: "listWorkspaceRightSurfacePending",
    method: constants.APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_LIST,
    params: "optional-empty",
  },
  {
    name: "consumeWorkspaceRightSurfacePending",
    method: constants.APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CONSUME,
    params: "required",
  },
  {
    name: "dismissWorkspaceRightSurfacePending",
    method: constants.APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_DISMISS,
    params: "required",
  },
  {
    name: "readArtifacts",
    method: constants.APP_SERVER_METHOD_ARTIFACT_READ,
    params: "required",
  },
  {
    name: "writeArtifact",
    method: constants.APP_SERVER_METHOD_ARTIFACT_WRITE,
    params: "required",
  },
  {
    name: "searchFiles",
    method: constants.APP_SERVER_METHOD_FUZZY_FILE_SEARCH,
    params: "required",
  },
  {
    name: "readFile",
    method: constants.APP_SERVER_METHOD_FS_READ_FILE,
    params: "required",
  },
  {
    name: "writeFile",
    method: constants.APP_SERVER_METHOD_FS_WRITE_FILE,
    params: "required",
  },
  {
    name: "createDirectory",
    method: constants.APP_SERVER_METHOD_FS_CREATE_DIRECTORY,
    params: "required",
  },
  {
    name: "getMetadata",
    method: constants.APP_SERVER_METHOD_FS_GET_METADATA,
    params: "required",
  },
  {
    name: "readDirectory",
    method: constants.APP_SERVER_METHOD_FS_READ_DIRECTORY,
    params: "required",
  },
  {
    name: "remove",
    method: constants.APP_SERVER_METHOD_FS_REMOVE,
    params: "required",
  },
  {
    name: "copy",
    method: constants.APP_SERVER_METHOD_FS_COPY,
    params: "required",
  },
  {
    name: "watch",
    method: constants.APP_SERVER_METHOD_FS_WATCH,
    params: "required",
  },
  {
    name: "unwatch",
    method: constants.APP_SERVER_METHOD_FS_UNWATCH,
    params: "required",
  },
  {
    name: "readProjectGitStatus",
    method: constants.APP_SERVER_METHOD_PROJECT_GIT_STATUS,
    params: "required",
  },
  {
    name: "readProjectGitDiff",
    method: constants.APP_SERVER_METHOD_PROJECT_GIT_DIFF,
    params: "required",
  },
  {
    name: "listProjectGitCommits",
    method: constants.APP_SERVER_METHOD_PROJECT_GIT_COMMITS_LIST,
    params: "required",
  },
  {
    name: "checkoutProjectGitBranch",
    method: constants.APP_SERVER_METHOD_PROJECT_GIT_BRANCH_CHECKOUT,
    params: "required",
  },
  {
    name: "createProjectGitBranch",
    method: constants.APP_SERVER_METHOD_PROJECT_GIT_BRANCH_CREATE,
    params: "required",
  },
  {
    name: "createProjectGitWorktree",
    method: constants.APP_SERVER_METHOD_PROJECT_GIT_WORKTREE_CREATE,
    params: "required",
  },
  {
    name: "exportHandoffBundle",
    method: constants.APP_SERVER_METHOD_AGENT_SESSION_HANDOFF_BUNDLE_EXPORT,
    params: "required",
  },
  {
    name: "exportReplayCase",
    method: constants.APP_SERVER_METHOD_AGENT_SESSION_REPLAY_CASE_EXPORT,
    params: "required",
  },
  {
    name: "exportAnalysisHandoff",
    method: constants.APP_SERVER_METHOD_AGENT_SESSION_ANALYSIS_HANDOFF_EXPORT,
    params: "required",
  },
  {
    name: "exportReviewDecisionTemplate",
    method:
      constants.APP_SERVER_METHOD_AGENT_SESSION_REVIEW_DECISION_TEMPLATE_EXPORT,
    params: "required",
  },
  {
    name: "saveReviewDecision",
    method: constants.APP_SERVER_METHOD_AGENT_SESSION_REVIEW_DECISION_SAVE,
    params: "required",
  },
  {
    name: "readSession",
    method: constants.APP_SERVER_METHOD_THREAD_READ,
    params: "required",
  },
  {
    name: "listThreads",
    method: constants.APP_SERVER_METHOD_THREAD_LIST,
    params: "optional-empty",
  },
  {
    name: "readThread",
    method: constants.APP_SERVER_METHOD_THREAD_READ,
    params: "required",
  },
  {
    name: "setThreadName",
    method: constants.APP_SERVER_METHOD_THREAD_NAME_SET,
    params: "required",
  },
  {
    name: "updateThreadSettings",
    method: constants.APP_SERVER_METHOD_THREAD_SETTINGS_UPDATE,
    params: "required",
  },
  {
    name: "setThreadMemoryMode",
    method: constants.APP_SERVER_METHOD_THREAD_MEMORY_MODE_SET,
    params: "required",
  },
  {
    name: "runThreadShellCommand",
    method: constants.APP_SERVER_METHOD_THREAD_SHELL_COMMAND,
    params: "required",
  },
  {
    name: "execCommand",
    method: constants.APP_SERVER_METHOD_COMMAND_EXEC,
    params: "required",
  },
  {
    name: "writeCommandExec",
    method: constants.APP_SERVER_METHOD_COMMAND_EXEC_WRITE,
    params: "required",
  },
  {
    name: "resizeCommandExec",
    method: constants.APP_SERVER_METHOD_COMMAND_EXEC_RESIZE,
    params: "required",
  },
  {
    name: "terminateCommandExec",
    method: constants.APP_SERVER_METHOD_COMMAND_EXEC_TERMINATE,
    params: "required",
  },
  {
    name: "archiveThread",
    method: constants.APP_SERVER_METHOD_THREAD_ARCHIVE,
    params: "required",
  },
  {
    name: "unarchiveThread",
    method: constants.APP_SERVER_METHOD_THREAD_UNARCHIVE,
    params: "required",
  },
  {
    name: "readMedia",
    method: constants.APP_SERVER_METHOD_MEDIA_READ,
    params: "required",
  },
  {
    name: "listApps",
    method: constants.APP_SERVER_METHOD_APP_LIST,
    params: "optional-empty",
  },
  {
    name: "readApps",
    method: constants.APP_SERVER_METHOD_APP_READ,
    params: "required",
  },
  {
    name: "listInstalledApps",
    method: constants.APP_SERVER_METHOD_APP_INSTALLED,
    params: "optional-empty",
  },
  {
    name: "listSkills",
    method: constants.APP_SERVER_METHOD_SKILLS_LIST,
    params: "optional-empty",
  },
  {
    name: "setSkillsExtraRoots",
    method: constants.APP_SERVER_METHOD_SKILLS_EXTRA_ROOTS_SET,
    params: "required",
  },
  {
    name: "writeSkillsConfig",
    method: constants.APP_SERVER_METHOD_SKILLS_CONFIG_WRITE,
    params: "required",
  },
  {
    name: "setPluginCatalogEnabled",
    method: constants.APP_SERVER_METHOD_PLUGIN_ENABLED_SET,
    params: "required",
  },
  {
    name: "listPluginCatalog",
    method: constants.APP_SERVER_METHOD_PLUGIN_LIST,
    params: "optional-empty",
  },
  {
    name: "searchPlugins",
    method: constants.APP_SERVER_METHOD_PLUGIN_SEARCH,
    params: "required",
  },
  {
    name: "readPluginCatalog",
    method: constants.APP_SERVER_METHOD_PLUGIN_READ,
    params: "required",
  },
  {
    name: "installPluginCatalog",
    method: constants.APP_SERVER_METHOD_PLUGIN_INSTALL,
    params: "required",
  },
  {
    name: "uninstallPluginCatalog",
    method: constants.APP_SERVER_METHOD_PLUGIN_UNINSTALL,
    params: "required",
  },
  {
    name: "listInstalledPluginCatalog",
    method: constants.APP_SERVER_METHOD_PLUGIN_INSTALLED,
    params: "optional-empty",
  },
  {
    name: "readAgentSessionToolInventory",
    method: constants.APP_SERVER_METHOD_AGENT_SESSION_TOOL_INVENTORY_READ,
    params: "optional-empty",
  },
  {
    name: "deleteThread",
    method: constants.APP_SERVER_METHOD_THREAD_DELETE,
    params: "required",
  },
  {
    name: "startThreadCompaction",
    method: constants.APP_SERVER_METHOD_THREAD_COMPACT_START,
    params: "required",
  },
  {
    name: "resumeThread",
    method: constants.APP_SERVER_METHOD_THREAD_RESUME,
    params: "required",
  },
  {
    name: "listAgentSessionFileCheckpoints",
    method: constants.APP_SERVER_METHOD_AGENT_SESSION_FILE_CHECKPOINT_LIST,
    params: "required",
  },
  {
    name: "getAgentSessionFileCheckpoint",
    method: constants.APP_SERVER_METHOD_AGENT_SESSION_FILE_CHECKPOINT_GET,
    params: "required",
  },
  {
    name: "diffAgentSessionFileCheckpoint",
    method: constants.APP_SERVER_METHOD_AGENT_SESSION_FILE_CHECKPOINT_DIFF,
    params: "required",
  },
  {
    name: "restoreAgentSessionFileCheckpoint",
    method: constants.APP_SERVER_METHOD_AGENT_SESSION_FILE_CHECKPOINT_RESTORE,
    params: "required",
  },
  {
    name: "getOrCreateSessionFile",
    method: constants.APP_SERVER_METHOD_SESSION_FILE_GET_OR_CREATE,
    params: "required",
  },
  {
    name: "updateSessionFileMeta",
    method: constants.APP_SERVER_METHOD_SESSION_FILE_UPDATE_META,
    params: "required",
  },
  {
    name: "saveSessionFile",
    method: constants.APP_SERVER_METHOD_SESSION_FILE_SAVE,
    params: "required",
  },
  {
    name: "readSessionFile",
    method: constants.APP_SERVER_METHOD_SESSION_FILE_READ,
    params: "required",
  },
  {
    name: "resolveSessionFilePath",
    method: constants.APP_SERVER_METHOD_SESSION_FILE_RESOLVE_PATH,
    params: "required",
  },
  {
    name: "deleteSessionFile",
    method: constants.APP_SERVER_METHOD_SESSION_FILE_DELETE,
    params: "required",
  },
  {
    name: "listSessionFiles",
    method: constants.APP_SERVER_METHOD_SESSION_FILE_LIST,
    params: "required",
  },
  {
    name: "startTurn",
    method: constants.APP_SERVER_METHOD_TURN_START,
    params: "required",
  },
  {
    name: "startReview",
    method: constants.APP_SERVER_METHOD_REVIEW_START,
    params: "required",
  },
  {
    name: "cancelTurn",
    method: constants.APP_SERVER_METHOD_TURN_INTERRUPT,
    params: "required",
  },
  {
    name: "steerTurn",
    method: constants.APP_SERVER_METHOD_TURN_STEER,
    params: "required",
  },
  {
    name: "respondAction",
    method: constants.APP_SERVER_METHOD_AGENT_SESSION_ACTION_RESPOND,
    params: "required",
  },
  {
    name: "readWorkflow",
    method: constants.APP_SERVER_METHOD_WORKFLOW_READ,
    params: "required",
  },
  {
    name: "cancelWorkflow",
    method: constants.APP_SERVER_METHOD_WORKFLOW_CANCEL,
    params: "required",
  },
  {
    name: "retryWorkflow",
    method: constants.APP_SERVER_METHOD_WORKFLOW_RETRY,
    params: "required",
  },
  {
    name: "respondWorkflow",
    method: constants.APP_SERVER_METHOD_WORKFLOW_RESPOND,
    params: "required",
  },
  {
    name: "listLogs",
    method: constants.APP_SERVER_METHOD_LOG_LIST,
    params: "none",
  },
  {
    name: "readPersistedLogTail",
    method: constants.APP_SERVER_METHOD_LOG_PERSISTED_TAIL,
    params: "required",
  },
  {
    name: "clearLogs",
    method: constants.APP_SERVER_METHOD_LOG_CLEAR,
    params: "none",
  },
  {
    name: "clearDiagnosticLogHistory",
    method: constants.APP_SERVER_METHOD_LOG_DIAGNOSTIC_HISTORY_CLEAR,
    params: "none",
  },
  {
    name: "readLogStorageDiagnostics",
    method: constants.APP_SERVER_METHOD_DIAGNOSTICS_LOG_STORAGE_READ,
    params: "none",
  },
  {
    name: "exportSupportBundle",
    method: constants.APP_SERVER_METHOD_DIAGNOSTICS_SUPPORT_BUNDLE_EXPORT,
    params: "optional-empty",
  },
  {
    name: "readServerDiagnostics",
    method: constants.APP_SERVER_METHOD_DIAGNOSTICS_SERVER_READ,
    params: "none",
  },
  {
    name: "readWindowsStartupDiagnostics",
    method: constants.APP_SERVER_METHOD_DIAGNOSTICS_WINDOWS_STARTUP_READ,
    params: "none",
  },
  {
    name: "listDiagnosticsTraces",
    method: constants.APP_SERVER_METHOD_DIAGNOSTICS_TRACE_LIST,
    params: "required",
  },
  {
    name: "readDiagnosticsTrace",
    method: constants.APP_SERVER_METHOD_DIAGNOSTICS_TRACE_READ,
    params: "required",
  },
  {
    name: "exportDiagnosticsTrace",
    method: constants.APP_SERVER_METHOD_DIAGNOSTICS_TRACE_EXPORT,
    params: "required",
  },
  {
    name: "readGatewayChannelStatus",
    method: constants.APP_SERVER_METHOD_GATEWAY_CHANNEL_STATUS,
    params: "required",
  },
  {
    name: "startGatewayChannel",
    method: constants.APP_SERVER_METHOD_GATEWAY_CHANNEL_START,
    params: "required",
  },
  {
    name: "stopGatewayChannel",
    method: constants.APP_SERVER_METHOD_GATEWAY_CHANNEL_STOP,
    params: "required",
  },
  {
    name: "probeTelegramChannel",
    method: constants.APP_SERVER_METHOD_TELEGRAM_CHANNEL_PROBE,
    params: "optional-empty",
  },
  {
    name: "probeFeishuChannel",
    method: constants.APP_SERVER_METHOD_FEISHU_CHANNEL_PROBE,
    params: "optional-empty",
  },
  {
    name: "probeDiscordChannel",
    method: constants.APP_SERVER_METHOD_DISCORD_CHANNEL_PROBE,
    params: "optional-empty",
  },
  {
    name: "probeWechatChannel",
    method: constants.APP_SERVER_METHOD_WECHAT_CHANNEL_PROBE,
    params: "optional-empty",
  },
  {
    name: "startWechatChannelLogin",
    method: constants.APP_SERVER_METHOD_WECHAT_CHANNEL_LOGIN_START,
    params: "optional-empty",
  },
  {
    name: "waitWechatChannelLogin",
    method: constants.APP_SERVER_METHOD_WECHAT_CHANNEL_LOGIN_WAIT,
    params: "required",
  },
  {
    name: "listWechatChannelAccounts",
    method: constants.APP_SERVER_METHOD_WECHAT_CHANNEL_ACCOUNT_LIST,
    params: "none",
  },
  {
    name: "removeWechatChannelAccount",
    method: constants.APP_SERVER_METHOD_WECHAT_CHANNEL_ACCOUNT_REMOVE,
    params: "required",
  },
  {
    name: "setWechatChannelRuntimeModel",
    method: constants.APP_SERVER_METHOD_WECHAT_CHANNEL_RUNTIME_MODEL_SET,
    params: "required",
  },
  {
    name: "probeGatewayTunnel",
    method: constants.APP_SERVER_METHOD_GATEWAY_TUNNEL_PROBE,
    params: "none",
  },
  {
    name: "detectGatewayTunnelCloudflared",
    method: constants.APP_SERVER_METHOD_GATEWAY_TUNNEL_CLOUDFLARED_DETECT,
    params: "none",
  },
  {
    name: "installGatewayTunnelCloudflared",
    method: constants.APP_SERVER_METHOD_GATEWAY_TUNNEL_CLOUDFLARED_INSTALL,
    params: "required",
  },
  {
    name: "createGatewayTunnel",
    method: constants.APP_SERVER_METHOD_GATEWAY_TUNNEL_CREATE,
    params: "required",
  },
  {
    name: "startGatewayTunnel",
    method: constants.APP_SERVER_METHOD_GATEWAY_TUNNEL_START,
    params: "none",
  },
  {
    name: "stopGatewayTunnel",
    method: constants.APP_SERVER_METHOD_GATEWAY_TUNNEL_STOP,
    params: "none",
  },
  {
    name: "restartGatewayTunnel",
    method: constants.APP_SERVER_METHOD_GATEWAY_TUNNEL_RESTART,
    params: "none",
  },
  {
    name: "readGatewayTunnelStatus",
    method: constants.APP_SERVER_METHOD_GATEWAY_TUNNEL_STATUS,
    params: "none",
  },
  {
    name: "syncGatewayTunnelWebhookUrl",
    method: constants.APP_SERVER_METHOD_GATEWAY_TUNNEL_SYNC_WEBHOOK_URL,
    params: "required",
  },
  {
    name: "createImageMediaTaskArtifact",
    method: constants.APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_IMAGE_CREATE,
    params: "required",
  },
  {
    name: "createAudioMediaTaskArtifact",
    method: constants.APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_AUDIO_CREATE,
    params: "required",
  },
  {
    name: "createTranscriptionMediaTaskArtifact",
    method:
      constants.APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_TRANSCRIPTION_CREATE,
    params: "required",
  },
  {
    name: "createVideoMediaTaskArtifact",
    method: constants.APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_VIDEO_CREATE,
    params: "required",
  },
  {
    name: "completeImageMediaTaskArtifact",
    method: constants.APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_IMAGE_COMPLETE,
    params: "required",
  },
  {
    name: "completeAudioMediaTaskArtifact",
    method: constants.APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_AUDIO_COMPLETE,
    params: "required",
  },
  {
    name: "getMediaTaskArtifact",
    method: constants.APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_GET,
    params: "required",
  },
  {
    name: "listMediaTaskArtifacts",
    method: constants.APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_LIST,
    params: "required",
  },
  {
    name: "cancelMediaTaskArtifact",
    method: constants.APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_CANCEL,
    params: "required",
  },
  {
    name: "getGalleryMaterial",
    method: constants.APP_SERVER_METHOD_GALLERY_MATERIAL_GET,
    params: "required",
  },
  {
    name: "createGalleryMaterialMetadata",
    method: constants.APP_SERVER_METHOD_GALLERY_MATERIAL_METADATA_CREATE,
    params: "required",
  },
  {
    name: "getGalleryMaterialMetadata",
    method: constants.APP_SERVER_METHOD_GALLERY_MATERIAL_METADATA_GET,
    params: "required",
  },
  {
    name: "updateGalleryMaterialMetadata",
    method: constants.APP_SERVER_METHOD_GALLERY_MATERIAL_METADATA_UPDATE,
    params: "required",
  },
  {
    name: "deleteGalleryMaterialMetadata",
    method: constants.APP_SERVER_METHOD_GALLERY_MATERIAL_METADATA_DELETE,
    params: "required",
  },
  {
    name: "listGalleryMaterialsByImageCategory",
    method: constants.APP_SERVER_METHOD_GALLERY_MATERIAL_LIST_BY_IMAGE_CATEGORY,
    params: "required",
  },
  {
    name: "listGalleryMaterialsByLayoutCategory",
    method:
      constants.APP_SERVER_METHOD_GALLERY_MATERIAL_LIST_BY_LAYOUT_CATEGORY,
    params: "required",
  },
  {
    name: "listGalleryMaterialsByMood",
    method: constants.APP_SERVER_METHOD_GALLERY_MATERIAL_LIST_BY_MOOD,
    params: "required",
  },
  {
    name: "listProjectMaterials",
    method: constants.APP_SERVER_METHOD_PROJECT_MATERIAL_LIST,
    params: "required",
  },
  {
    name: "getProjectMaterial",
    method: constants.APP_SERVER_METHOD_PROJECT_MATERIAL_GET,
    params: "required",
  },
  {
    name: "countProjectMaterials",
    method: constants.APP_SERVER_METHOD_PROJECT_MATERIAL_COUNT,
    params: "required",
  },
  {
    name: "uploadProjectMaterial",
    method: constants.APP_SERVER_METHOD_PROJECT_MATERIAL_UPLOAD,
    params: "required",
  },
  {
    name: "importProjectMaterialFromUrl",
    method: constants.APP_SERVER_METHOD_PROJECT_MATERIAL_IMPORT_FROM_URL,
    params: "required",
  },
  {
    name: "updateProjectMaterial",
    method: constants.APP_SERVER_METHOD_PROJECT_MATERIAL_UPDATE,
    params: "required",
  },
  {
    name: "deleteProjectMaterial",
    method: constants.APP_SERVER_METHOD_PROJECT_MATERIAL_DELETE,
    params: "required",
  },
  {
    name: "readProjectMaterialContent",
    method: constants.APP_SERVER_METHOD_PROJECT_MATERIAL_CONTENT,
    params: "required",
  },
  {
    name: "listVoiceAsrCredentials",
    method: constants.APP_SERVER_METHOD_VOICE_ASR_CREDENTIAL_LIST,
    params: "none",
  },
  {
    name: "createVoiceAsrCredential",
    method: constants.APP_SERVER_METHOD_VOICE_ASR_CREDENTIAL_CREATE,
    params: "required",
  },
  {
    name: "updateVoiceAsrCredential",
    method: constants.APP_SERVER_METHOD_VOICE_ASR_CREDENTIAL_UPDATE,
    params: "required",
  },
  {
    name: "deleteVoiceAsrCredential",
    method: constants.APP_SERVER_METHOD_VOICE_ASR_CREDENTIAL_DELETE,
    params: "required",
  },
  {
    name: "setDefaultVoiceAsrCredential",
    method: constants.APP_SERVER_METHOD_VOICE_ASR_CREDENTIAL_DEFAULT_SET,
    params: "required",
  },
  {
    name: "testVoiceAsrCredential",
    method: constants.APP_SERVER_METHOD_VOICE_ASR_CREDENTIAL_TEST,
    params: "required",
  },
  {
    name: "listVoiceInstructions",
    method: constants.APP_SERVER_METHOD_VOICE_INSTRUCTION_LIST,
    params: "none",
  },
  {
    name: "saveVoiceInstruction",
    method: constants.APP_SERVER_METHOD_VOICE_INSTRUCTION_SAVE,
    params: "required",
  },
  {
    name: "deleteVoiceInstruction",
    method: constants.APP_SERVER_METHOD_VOICE_INSTRUCTION_DELETE,
    params: "required",
  },
  {
    name: "setDefaultVoiceModel",
    method: constants.APP_SERVER_METHOD_VOICE_MODEL_DEFAULT_SET,
    params: "required",
  },
  {
    name: "testTranscribeVoiceModelFile",
    method: constants.APP_SERVER_METHOD_VOICE_MODEL_TEST_TRANSCRIBE_FILE,
    params: "required",
  },
  {
    name: "transcribeVoiceAudio",
    method: constants.APP_SERVER_METHOD_VOICE_TRANSCRIPTION_TRANSCRIBE_AUDIO,
    params: "required",
  },
  {
    name: "polishVoiceText",
    method: constants.APP_SERVER_METHOD_VOICE_TRANSCRIPTION_POLISH_TEXT,
    params: "required",
  },
  {
    name: "readUsageStats",
    method: constants.APP_SERVER_METHOD_USAGE_STATS_READ,
    params: "required",
  },
  {
    name: "listUsageStatsModelRanking",
    method: constants.APP_SERVER_METHOD_USAGE_STATS_MODEL_RANKING_LIST,
    params: "required",
  },
  {
    name: "listUsageStatsDailyTrends",
    method: constants.APP_SERVER_METHOD_USAGE_STATS_DAILY_TRENDS_LIST,
    params: "required",
  },
];
