import js from "@eslint/js";
import tseslint from "@typescript-eslint/eslint-plugin";
import tsparser from "@typescript-eslint/parser";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import globals from "globals";
import agentCommandCatalog from "./src/lib/governance/agentCommandCatalog.json" with { type: "json" };

const legacyChatRestrictedPatterns = [
  "@/components/chat",
  "@/components/chat/*",
  "@/components/chat/**",
  "**/components/chat",
  "**/components/chat/*",
  "**/components/chat/**",
];

const agentCompatRestrictedPatterns = [
  {
    group: [
      "@/components/agent/chat/hooks/useAgentChat",
      "@/components/agent/chat/hooks/useAgentChat.*",
      "**/useAgentChat",
      "**/useAgentChat.*",
    ],
    message:
      "useAgentChat 属于零引用 compat Hook，请改用 useAgentChatUnified 或 useAgentChat；仅兼容回归测试允许引用。",
  },
  {
    group: [
      "@/stores/agentStore",
      "@/stores/agentStore.*",
      "**/agentStore",
      "**/agentStore.*",
    ],
    message:
      "agentStore 属于零引用遗留状态容器，请改用 useAgentChatUnified / useAgentChat；仅兼容回归测试允许引用。",
  },
];

const sceneAppDeadRestrictedPatterns = [
  {
    group: [
      "@/lib/sceneapp/types-runtime-context",
      "@/lib/sceneapp/types-runtime-context.*",
      "**/types-runtime-context",
      "**/types-runtime-context.*",
    ],
    message:
      "types-runtime-context.ts 历史转发文件已删除；请统一改用 @/lib/sceneapp/types 或当前 sceneapp barrel，避免 dead 类型副本重新扩散。",
  },
];

const deprecatedAgentRuntimeHelperNames = Object.keys(
  agentCommandCatalog.deprecatedHelperReplacements,
);
const LEGACY_DESKTOP_HOST_AGGREGATE_HOOK_PATH = `@/hooks/${[
  "use",
  "Ta",
  "uri",
].join("")}`;

const generalChatRestrictedPaths = [
  {
    name: "@/components/general-chat",
    importNames: ["useChat"],
    message:
      "general-chat 的 useChat 属于旧路径，请优先使用 useAgentChatUnified / useAgentChat 或当前现役聊天入口。",
  },
  {
    name: "@/components/general-chat",
    importNames: ["useSession", "useStreaming"],
    message:
      "general-chat 当前属于兼容链路，请不要在新代码中继续引入页面入口或旧 Hook。",
  },
  {
    name: "@/components/general-chat",
    importNames: ["GeneralChatPage"],
    message:
      "general-chat 当前属于兼容链路，请不要在新代码中继续引入页面入口或旧 Hook。",
  },
  {
    name: "@/components/general-chat/hooks",
    importNames: ["useChat"],
    message:
      "general-chat/hooks/useChat 属于旧路径，请优先使用 useAgentChatUnified / useAgentChat 或当前现役聊天入口。",
  },
  {
    name: "@/components/general-chat/hooks",
    importNames: ["useSession", "useStreaming"],
    message:
      "general-chat/hooks 下的 useSession/useStreaming 属于兼容实现，请优先接入 useAgentChatUnified / useAgentChat / agent_runtime_*。",
  },
  {
    name: "@/components/general-chat/hooks/useChat",
    message:
      "general-chat/hooks/useChat 属于旧路径，请优先使用 useAgentChatUnified / useAgentChat 或当前现役聊天入口。",
  },
  {
    name: "@/components/general-chat/GeneralChatPage",
    message:
      "GeneralChatPage 属于旧版 general-chat 入口，请不要在新代码中继续引入。",
  },
  {
    name: "@/components/general-chat/hooks/useSession",
    message:
      "general-chat/hooks/useSession 属于旧版会话兼容 Hook，请优先接入统一对话链路。",
  },
  {
    name: "@/components/general-chat/hooks/useStreaming",
    message:
      "general-chat/hooks/useStreaming 依赖旧流事件协议，请优先接入统一对话链路。",
  },
  {
    name: "@/components/general-chat/canvas",
    message:
      "请不要直接深导入 general-chat/canvas；跨模块复用请改用 @/components/general-chat/bridge。",
  },
  {
    name: "@/components/general-chat/types",
    message:
      "请不要直接深导入 general-chat/types；跨模块复用请改用 @/components/general-chat/bridge 或现役共享类型。",
  },
  {
    name: "@/components/general-chat/store/useGeneralChatStore",
    message:
      "请不要直接深导入 general-chat 内部 store；如需兼容桥接，请显式放在 compat 层。",
  },
  {
    name: "@/lib/api/generalChatCompat",
    message:
      "generalChatCompat 属于兼容网关，请仅在 general-chat store 中消费，避免 compat 逻辑再次向业务层扩散。",
  },
  {
    name: "@/lib/api/compat",
    message:
      "api/compat.ts 属于历史记忆兼容层，请不要在新代码中接入；记忆主链请使用当前 memory store / MemoryBackend / memory tools / Soul 配置入口。",
  },
  {
    name: "@/lib/api/agent",
    message:
      "agent.ts 与 agentRuntime root aggregate 已删除；请直接使用 @/lib/api/agentRuntime/<domain> client 或 @/lib/api/agentStream。",
  },
  {
    name: LEGACY_DESKTOP_HOST_AGGREGATE_HOOK_PATH,
    message:
      "已删除的桌面宿主聚合 Hook 现在只是兼容聚合层，禁止新增依赖；请直接接入对应的 @/lib/api/* 网关。",
  },
  {
    name: LEGACY_DESKTOP_HOST_AGGREGATE_HOOK_PATH,
    importNames: [
      "startServer",
      "stopServer",
      "getServerStatus",
      "getServerDiagnostics",
      "getLogStorageDiagnostics",
      "exportSupportBundle",
      "getWindowsStartupDiagnostics",
      "ServerStatus",
      "ServerDiagnostics",
      "LogStorageDiagnostics",
      "SupportBundleExportResult",
      "WindowsStartupDiagnostics",
    ],
    message:
      "server/diagnostics 相关能力已迁移到 @/lib/api/serverRuntime，请不要继续从 已删除的桌面宿主聚合 Hook 聚合层引入。",
  },
  {
    name: LEGACY_DESKTOP_HOST_AGGREGATE_HOOK_PATH,
    importNames: [
      "getLogs",
      "getPersistedLogsTail",
      "clearLogs",
      "clearDiagnosticLogHistory",
      "LogEntry",
    ],
    message:
      "日志相关能力已迁移到 @/lib/api/logs，请不要继续从 已删除的桌面宿主聚合 Hook 聚合层引入。",
  },
  {
    name: LEGACY_DESKTOP_HOST_AGGREGATE_HOOK_PATH,
    importNames: [
      "checkApiCompatibility",
      "ApiCheckResult",
      "ApiCompatibilityResult",
    ],
    message:
      "API 兼容性检查能力已迁移到 @/lib/api/apiCompatibility，请不要继续从 已删除的桌面宿主聚合 Hook 聚合层引入。",
  },
  {
    name: LEGACY_DESKTOP_HOST_AGGREGATE_HOOK_PATH,
    importNames: [
      "getEndpointProviders",
      "setEndpointProvider",
      "EndpointProvidersConfig",
    ],
    message:
      "端点 Provider 配置能力已迁移到 @/lib/api/endpointProviders，请不要继续从 已删除的桌面宿主聚合 Hook 聚合层引入。",
  },
  {
    name: LEGACY_DESKTOP_HOST_AGGREGATE_HOOK_PATH,
    importNames: [
      "getMemoryStats",
      "requestMemoryAnalysis",
      "cleanupMemory",
      "CleanupMemoryResult",
      "MemoryAnalysisResult",
      "MemoryStatsResponse",
    ],
    message:
      "旧记忆分析/清理能力已下线；记忆整理、审阅和重建索引请走 src/lib/api/memoryStore.ts -> App Server memoryStore/*，不要继续从已删除的桌面宿主聚合 Hook 聚合层引入。",
  },
  {
    name: LEGACY_DESKTOP_HOST_AGGREGATE_HOOK_PATH,
    importNames: [
      "testTts",
      "getAvailableVoices",
      "TtsTestResult",
      "VoiceOption",
    ],
    message:
      "语音测试能力已迁移到 @/lib/api/voiceTools，请不要继续从 已删除的桌面宿主聚合 Hook 聚合层引入。",
  },
  {
    name: LEGACY_DESKTOP_HOST_AGGREGATE_HOOK_PATH,
    importNames: ["uploadAvatar", "deleteAvatar", "UploadResult"],
    message:
      "头像上传/删除能力已迁移到 @/lib/api/profileAssets，请不要继续从 已删除的桌面宿主聚合 Hook 聚合层引入。",
  },
  {
    name: LEGACY_DESKTOP_HOST_AGGREGATE_HOOK_PATH,
    importNames: [
      "getConfig",
      "saveConfig",
      "getEnvironmentPreview",
      "getDefaultProvider",
      "setDefaultProvider",
      "updateProviderEnvVars",
      "Config",
      "EnvironmentPreview",
    ],
    message:
      "配置/环境预览相关能力已迁移到 @/lib/api/appConfig，请不要继续从 已删除的桌面宿主聚合 Hook 聚合层引入。",
  },
  {
    name: LEGACY_DESKTOP_HOST_AGGREGATE_HOOK_PATH,
    importNames: [
      "ChannelsConfig",
      "GatewayConfig",
      "TelegramBotConfig",
      "DiscordBotConfig",
      "FeishuBotConfig",
      "GatewayChannelStatusResponse",
      "TelegramProbeResult",
      "FeishuProbeResult",
      "DiscordProbeResult",
      "GatewayTunnelStatus",
      "GatewayTunnelProbeResult",
      "CloudflaredInstallStatus",
      "CloudflaredInstallResult",
      "GatewayTunnelCreateResponse",
      "GatewayTunnelSyncWebhookResponse",
      "gatewayChannelStart",
      "gatewayChannelStop",
      "gatewayChannelStatus",
      "telegramChannelProbe",
      "feishuChannelProbe",
      "discordChannelProbe",
      "gatewayTunnelProbe",
      "gatewayTunnelDetectCloudflared",
      "gatewayTunnelInstallCloudflared",
      "gatewayTunnelCreate",
      "gatewayTunnelStart",
      "gatewayTunnelStop",
      "gatewayTunnelRestart",
      "gatewayTunnelStatus",
      "gatewayTunnelSyncWebhookUrl",
    ],
    message:
      "channels/gateway 相关能力已迁移到 @/lib/api/channelsRuntime，请不要继续从 已删除的桌面宿主聚合 Hook 聚合层引入。",
  },
  {
    name: LEGACY_DESKTOP_HOST_AGGREGATE_HOOK_PATH,
    importNames: [
      "getExperimentalConfig",
      "saveExperimentalConfig",
      "validateShortcut",
      "updateScreenshotShortcut",
      "ExperimentalFeatures",
      "SmartInputConfig",
    ],
    message:
      "实验室配置/截图快捷键相关能力已迁移到 @/lib/api/experimentalFeatures，请不要继续从 已删除的桌面宿主聚合 Hook 聚合层引入。",
  },
  {
    name: LEGACY_DESKTOP_HOST_AGGREGATE_HOOK_PATH,
    importNames: [
      "getMemoryOverview",
      "getMemoryEffectiveSources",
      "getMemoryAutoIndex",
      "toggleMemoryAuto",
      "updateMemoryAutoNote",
      "MemoryOverviewResponse",
      "EffectiveMemorySourcesResponse",
      "AutoMemoryIndexResponse",
      "MemoryAutoConfig",
      "MemoryAutoToggleResponse",
      "MemoryConfig",
      "MemoryProfileConfig",
      "MemoryResolveConfig",
      "MemorySourcesConfig",
    ],
    message:
      "旧记忆运行时相关能力已下线；记忆设置和显式整理请走 src/lib/api/memoryStore.ts 与 memory.soul current 配置，不要继续从已删除的桌面宿主聚合 Hook 聚合层引入。",
  },
  {
    name: LEGACY_DESKTOP_HOST_AGGREGATE_HOOK_PATH,
    importNames: ["getAvailableModels", "ModelInfo"],
    message:
      "模型列表查询已迁移到 @/lib/api/modelCatalog，请不要继续从 已删除的桌面宿主聚合 Hook 聚合层引入。",
  },
  {
    name: LEGACY_DESKTOP_HOST_AGGREGATE_HOOK_PATH,
    importNames: [
      "getUsageStats",
      "getModelUsageRanking",
      "getDailyUsageTrends",
      "UsageStatsResponse",
      "ModelUsage",
      "DailyUsage",
    ],
    message:
      "使用统计查询已迁移到 @/lib/api/usageStats，请不要继续从 已删除的桌面宿主聚合 Hook 聚合层引入。",
  },
  {
    name: LEGACY_DESKTOP_HOST_AGGREGATE_HOOK_PATH,
    importNames: [
      "reloadCredentials",
      "refreshKiroToken",
      "getKiroCredentials",
      "getEnvVariables",
      "getTokenFileHash",
      "checkAndReloadCredentials",
      "getGeminiCredentials",
      "reloadGeminiCredentials",
      "refreshGeminiToken",
      "getGeminiEnvVariables",
      "getGeminiTokenFileHash",
      "checkAndReloadGeminiCredentials",
      "getQwenCredentials",
      "reloadQwenCredentials",
      "refreshQwenToken",
      "getQwenEnvVariables",
      "getQwenTokenFileHash",
      "checkAndReloadQwenCredentials",
      "getOpenAICustomStatus",
      "setOpenAICustomConfig",
      "getClaudeCustomStatus",
      "setClaudeCustomConfig",
      "KiroCredentialStatus",
      "EnvVariable",
      "CheckResult",
      "GeminiCredentialStatus",
      "QwenCredentialStatus",
      "OpenAICustomStatus",
      "ClaudeCustomStatus",
      "CredentialEntry",
      "GeminiApiKeyEntry",
      "VertexApiKeyEntry",
      "VertexModelAlias",
      "AmpConfig",
      "AmpModelMapping",
    ],
    message:
      "provider 凭证/自定义状态相关能力已迁移到 @/lib/api/providerRuntime，请不要继续从 已删除的桌面宿主聚合 Hook 聚合层引入。",
  },
  {
    name: LEGACY_DESKTOP_HOST_AGGREGATE_HOOK_PATH,
    importNames: ["testApi", "TestResult", "getNetworkInfo", "NetworkInfo"],
    message:
      "API 测试/网络信息相关能力已迁移到 @/lib/api/serverTools，请不要继续从 已删除的桌面宿主聚合 Hook 聚合层引入。",
  },
  {
    name: "@/stores/agentStore",
    message:
      "agentStore 属于零引用遗留状态容器，请改用现役 useAgentChatUnified / useAgentChat 链路。",
  },
  {
    name: "@/stores",
    importNames: [
      "useAgentStore",
      "useAgentMessages",
      "useAgentStreaming",
      "useAgentSessions",
      "usePendingActions",
    ],
    message:
      "agentStore 相关导出属于零引用遗留状态容器，请改用现役 useAgentChatUnified / useAgentChat 链路。",
  },
  {
    name: "@/lib/api/agentCompat",
    message:
      "agentCompat 属于遗留兼容层，请仅在历史桥接或兼容测试中使用，避免继续向业务层扩散。",
  },
  {
    name: "@/lib/api/agent",
    importNames: ["sendAgentMessage", "sendAgentMessageStream"],
    message:
      "agent.ts 已删除；旧 Agent 发送 API 请改用 submitAgentRuntimeTurn() 或 useAgentChat().sendMessage()。",
  },
  {
    name: "@/lib/api/agentRuntime",
    importNames: deprecatedAgentRuntimeHelperNames,
    message:
      "agentRuntime root aggregate 已删除；业务层请直连对应分域 client 或 useAgentChat。",
  },
  {
    name: "@/hooks/useUnifiedChat",
    message:
      "useUnifiedChat 已删除；Agent 工作台请改用 useAgentChatUnified / useAgentChat，旧 General/Creator 路径如需恢复，必须基于 agent_runtime_* 重新设计。",
  },
  {
    name: "@/lib/api/agent",
    importNames: [
      "initagent",
      "getagentStatus",
      "resetagent",
      "listruntimeProviders",
    ],
    message:
      "agent.ts 已删除；旧 agent 命名 API 请改用现役 Agent API 或 Provider 配置流程。",
  },
];

const generalChatRestrictedPathsWithoutCompatApi =
  generalChatRestrictedPaths.filter(
    (entry) => entry.name !== "@/lib/api/generalChatCompat",
  );

const createLegacyChatImportRule = (
  paths,
  { includeAgentCompatPatterns = true } = {},
) => {
  const patterns = [
    {
      group: legacyChatRestrictedPatterns,
      message:
        "components/chat 为遗留聊天模块，禁止新增依赖；请优先使用现役聊天入口。",
    },
    ...sceneAppDeadRestrictedPatterns,
  ];

  if (includeAgentCompatPatterns) {
    patterns.push(...agentCompatRestrictedPatterns);
  }

  return [
    "error",
    {
      paths,
      patterns,
    },
  ];
};

const generalChatCompatCommandSelectors = [
  "general_chat_get_session",
  "general_chat_list_sessions",
  "general_chat_create_session",
  "general_chat_delete_session",
  "general_chat_rename_session",
  "general_chat_get_messages",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "general_chat_* compat 命令只允许集中放在 src/lib/api/generalChatCompat.ts，禁止在业务层直接扩散。",
}));

const agentRuntimeCommandSelectors = [
  ...new Set([
    ...agentCommandCatalog.runtimeGatewayCommands,
    ...Object.keys(agentCommandCatalog.deprecatedCommandReplacements),
  ]),
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "agent runtime 命令只允许集中放在 `src/lib/api/agentRuntime` 网关目录；旧 agent_/agent_ 命令已废弃，禁止在业务模块直接扩散。",
}));

const projectGatewayCommandSelectors = [
  "workspace_create",
  "workspace_get_projects_root",
  "workspace_resolve_project_path",
  "workspace_list",
  "workspace_get_default",
  "workspace_ensure_default_ready",
  "workspace_set_default",
  "workspace_get_by_path",
  "workspace_get",
  "workspace_update",
  "workspace_delete",
  "workspace_ensure_ready",
  "get_or_create_default_project",
  "content_create",
  "content_get",
  "content_get_theme_workbench_document_state",
  "content_list",
  "content_update",
  "content_delete",
  "content_reorder",
  "content_stats",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "workspace/project 相关后端命令请统一通过 `src/lib/api/project.ts` 暴露的网关函数调用，避免业务层继续拼接命令名并扩散兼容逻辑。",
}));

const materialGatewayCommandSelectors = [
  "list_materials",
  "get_material_count",
  "upload_material",
  "update_material",
  "delete_material",
  "get_material_content",
  "import_material_from_url",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "素材相关后端命令请统一通过 `src/lib/api/materials.ts` 暴露的网关函数调用，避免在 Hook / 组件中继续直接拼接命令名。",
}));

const templateGatewayCommandSelectors = [
  "list_templates",
  "get_default_template",
  "create_template",
  "update_template",
  "delete_template",
  "set_default_template",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "模板相关后端命令请统一通过 `src/lib/api/templates.ts` 暴露的网关函数调用，避免在 Hook / 组件中继续直接拼接命令名。",
}));

const personaGatewayCommandSelectors = [
  "list_personas",
  "get_default_persona",
  "create_persona",
  "update_persona",
  "delete_persona",
  "set_default_persona",
  "list_persona_templates",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "人设相关后端命令请统一通过 `src/lib/api/personas.ts` 暴露的网关函数调用，避免在 Hook / 组件中继续直接拼接命令名。",
}));

const brandPersonaGatewayCommandSelectors = [
  "get_brand_persona",
  "get_brand_extension",
  "save_brand_extension",
  "update_brand_extension",
  "delete_brand_extension",
  "list_brand_persona_templates",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "品牌人设相关后端命令请统一通过 `src/lib/api/brandPersona.ts` 暴露的网关函数调用，避免在 Hook / 组件中继续直接拼接命令名。",
}));

const posterMaterialGatewayCommandSelectors = [
  "get_poster_material",
  "create_poster_metadata",
  "update_poster_metadata",
  "delete_poster_metadata",
  "list_by_image_category",
  "list_by_layout_category",
  "list_by_mood",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "海报素材相关后端命令请统一通过 `src/lib/api/posterMaterials.ts` 暴露的网关函数调用，避免在 Hook / 组件中继续直接拼接命令名。",
}));

const fileSystemCommandSelectors = [
  "reveal_in_finder",
  "open_with_default_app",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "文件打开/定位相关后端命令请统一通过 `src/lib/api/fileSystem.ts` 暴露的网关函数调用，避免在 Hook / 组件中继续直接拼接命令名。",
}));

const fileBrowserCommandSelectors = [
  "list_dir",
  "read_file_preview_cmd",
  "create_file",
  "create_directory",
  "rename_file",
  "delete_file",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "文件浏览/文件树相关后端命令请统一通过 `src/lib/api/fileBrowser.ts` 暴露的网关函数调用，避免在组件中继续直接拼接命令名。",
}));

const appUpdateCommandSelectors = [
  "check_for_updates",
  "download_update",
  "close_update_window",
  "dismiss_update_notification",
  "record_update_notification_action",
  "remind_update_later",
  "skip_update_version",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "更新检查/更新提醒相关后端命令请统一通过 `src/lib/api/appUpdate.ts` 暴露的网关函数调用，避免在页面中继续直接拼接命令名。",
}));

const screenshotChatCommandSelectors = [
  "send_screenshot_chat",
  "close_screenshot_chat_window",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "截图聊天窗口相关后端命令请统一通过 `src/lib/api/screenshotChat.ts` 暴露的网关函数调用，避免在页面/组件中继续直接拼接命令名。",
}));

const systemSupportCommandSelectors = [
  "show_notification",
  "auto_fix_configuration",
  "report_frontend_crash",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "系统支持类后端命令请统一通过对应 API 网关（`src/lib/api/notification.ts` / `autoFix.ts` / `frontendCrash.ts`）调用，避免在 lib / hook 中继续直接拼接命令名。",
}));

const serverRuntimeCommandSelectors = [
  "start_server",
  "stop_server",
  "get_server_status",
  "get_server_diagnostics",
  "get_log_storage_diagnostics",
  "export_support_bundle",
  "get_windows_startup_diagnostics",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "服务控制/诊断相关后端命令请统一通过 `src/lib/api/serverRuntime.ts` 暴露的网关函数调用，避免继续在其他模块中直接拼接命令名。",
}));

const logCommandSelectors = [
  "get_logs",
  "get_persisted_logs_tail",
  "clear_logs",
  "clear_diagnostic_log_history",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "日志相关后端命令请统一通过 `src/lib/api/logs.ts` 暴露的网关函数调用，避免继续在其他模块中直接拼接命令名。",
}));

const appConfigCommandSelectors = [
  "get_config",
  "save_config",
  "get_environment_preview",
  "get_default_provider",
  "set_default_provider",
  "update_provider_env_vars",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "配置/环境预览相关后端命令请统一通过 `src/lib/api/appConfig.ts` 暴露的网关函数调用，避免继续在其他模块中直接拼接命令名。",
}));

const channelsRuntimeCommandSelectors = [
  "gateway_channel_start",
  "gateway_channel_stop",
  "gateway_channel_status",
  "telegram_channel_probe",
  "feishu_channel_probe",
  "discord_channel_probe",
  "gateway_tunnel_probe",
  "gateway_tunnel_detect_cloudflared",
  "gateway_tunnel_install_cloudflared",
  "gateway_tunnel_create",
  "gateway_tunnel_start",
  "gateway_tunnel_stop",
  "gateway_tunnel_restart",
  "gateway_tunnel_status",
  "gateway_tunnel_sync_webhook_url",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "channels/gateway 相关后端命令请统一通过 `src/lib/api/channelsRuntime.ts` 暴露的网关函数调用，避免继续在其他模块中直接拼接命令名。",
}));

const experimentalFeaturesCommandSelectors = [
  "validate_shortcut",
  "update_screenshot_shortcut",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "实验室配置/截图快捷键相关后端命令请统一通过 `src/lib/api/experimentalFeatures.ts` 暴露的网关函数调用，避免继续在其他模块中直接拼接命令名。",
}));

const memoryRuntimeCommandSelectors = [
  "get_conversation_memory_stats",
  "request_conversation_memory_analysis",
  "cleanup_conversation_memory",
  "get_conversation_memory_overview",
  "memory_get_effective_sources",
  "memory_get_auto_index",
  "memory_toggle_auto",
  "memory_update_auto_note",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "旧记忆运行时命令已下线；记忆主链请走 `src/lib/api/memoryStore.ts`、MemoryBackend 和 memory tools，不要恢复旧 runtime 网关。",
}));

const projectMemoryCommandSelectors = [
  "character_list",
  "character_get",
  "character_create",
  "character_update",
  "character_delete",
  "world_building_get",
  "world_building_update",
  "style_guide_get",
  "style_guide_update",
  "outline_node_list",
  "outline_node_get",
  "outline_node_create",
  "outline_node_update",
  "outline_node_delete",
  "project_memory_get",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "旧项目记忆 CRUD 命令已下线；项目上下文读取请走 `src/lib/api/projectMemory.ts` -> `projectMemory/read`，不要恢复旧聚合网关。",
}));

const toolHooksCommandSelectors = [
  "execute_hooks",
  "add_hook_rule",
  "remove_hook_rule",
  "toggle_hook_rule",
  "get_hook_rules",
  "get_hook_execution_stats",
  "clear_hook_execution_stats",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "工具钩子相关命令请统一通过 `src/lib/api/toolHooks.ts` 暴露的网关函数调用，避免继续在其他模块中直接拼接命令名。",
}));

const a2uiFormCommandSelectors = [
  "create_a2ui_form",
  "get_a2ui_form",
  "get_a2ui_forms_by_message",
  "get_a2ui_forms_by_session",
  "save_a2ui_form_data",
  "submit_a2ui_form",
  "delete_a2ui_form",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "A2UI 表单持久化命令请统一通过 `src/lib/api/a2uiForm.ts` 暴露的网关函数调用，避免继续在其他模块中直接拼接命令名。",
}));

const memoryFeedbackCommandSelectors = [
  "unified_memory_feedback",
  "get_memory_feedback_stats",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "旧记忆反馈命令已下线；记忆整理、审阅和证据展示请走 memory store current 主链，不要恢复旧反馈网关。",
}));

const contextMemoryCommandSelectors = [
  "save_memory_entry",
  "get_session_memories",
  "get_memory_context",
  "record_error",
  "should_avoid_operation",
  "mark_error_resolved",
  "get_memory_stats",
  "cleanup_expired_memories",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "旧上下文记忆命令已下线；长期记忆走 memory store，session / evidence 走 App Server current 方法，不要恢复旧上下文记忆网关。",
}));

const asrProviderCommandSelectors = [
  "list_audio_devices",
  "get_voice_input_config",
  "save_voice_input_config",
  "get_voice_instructions",
  "save_voice_instruction",
  "delete_voice_instruction",
  "transcribe_audio",
  "polish_voice_text",
  "open_voice_window",
  "close_voice_window",
  "output_voice_text",
  "start_recording",
  "stop_recording",
  "cancel_recording",
  "get_recording_status",
  "open_input_with_text",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "语音输入/ASR 相关命令请统一通过 `src/lib/api/asrProvider.ts` 暴露的网关函数调用，避免继续在其他模块中直接拼接命令名。",
}));

const contentWorkflowCommandSelectors = [
  "content_workflow_create",
  "content_workflow_get",
  "content_workflow_get_by_content",
  "content_workflow_advance",
  "content_workflow_retry",
  "content_workflow_cancel",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "旧 content_workflow_* 命令已下线，禁止恢复旧 content-workflow 网关；Workflow 控制面必须走 App Server current method：workflow/read、workflow/cancel、workflow/retry、workflow/respond。",
}));

const novelCommandSelectors = [
  "novel_create_project",
  "novel_update_settings",
  "novel_generate_outline",
  "novel_generate_characters",
  "novel_generate_chapter",
  "novel_continue_chapter",
  "novel_rewrite_chapter",
  "novel_polish_chapter",
  "novel_check_consistency",
  "novel_get_project_snapshot",
  "novel_list_runs",
  "novel_delete_character",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "小说编排命令请统一通过 `src/lib/api/novel.ts` 暴露的网关函数调用，避免继续在其他模块中直接拼接命令名。",
}));

const unifiedChatCommandSelectors = [
  "chat_create_session",
  "chat_list_sessions",
  "chat_get_session",
  "chat_delete_session",
  "chat_rename_session",
  "chat_get_messages",
  "chat_send_message",
  "chat_stop_generation",
  "chat_configure_provider",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "`chat_*` 兼容命令已停用；不要继续直接拼接旧命令名，现役链路请统一收敛到 agent_runtime_*。",
}));

const unifiedMemoryCommandSelectors = [
  "unified_memory_list",
  "unified_memory_search",
  "unified_memory_get",
  "unified_memory_create",
  "unified_memory_update",
  "unified_memory_delete",
  "unified_memory_stats",
  "unified_memory_analyze",
  "unified_memory_semantic_search",
  "unified_memory_hybrid_search",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "旧 unified_memory_* 命令已下线；不要恢复旧统一记忆网关，记忆主链请收敛到当前 memory store / MemoryBackend / memory tools。",
}));

const apiCompatibilityCommandSelectors = ["check_api_compatibility"].map(
  (command) => ({
    selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
    message:
      "API 兼容性检查命令请统一通过 `src/lib/api/apiCompatibility.ts` 暴露的网关函数调用，避免继续在其他模块中直接拼接命令名。",
  }),
);

const sceneAppCurrentProjectionAliasImportSelectors = [
  {
    selector:
      "ImportSpecifier[imported.name='SceneAppCurrentDescriptor'], ImportSpecifier[imported.name='SceneAppCurrentPlanResult'], ImportSpecifier[imported.name='SceneAppCurrentCatalog']",
    message:
      "SceneApp base type 现已是 current 事实源；业务层请直接改用 SceneAppDescriptor / SceneAppPlanResult / SceneAppCatalog，避免继续依赖 SceneAppCurrent* 过渡别名。",
  },
];

const sceneAppWideExecutionTypeImportSelectors = [
  {
    selector:
      "ImportSpecifier[imported.name='SceneAppBindingFamily'], ImportSpecifier[imported.name='SceneAppRuntimeAction']",
    message:
      "SceneApp 宽执行类型只允许停留在 API 归一化边界；业务层与测试夹具请改用 SceneAppExecutorBindingFamily / SceneAppExecutionRuntimeAction。",
  },
];

const sceneAppCompatInputTypeImportSelectors = [
  {
    selector:
      "ImportSpecifier[imported.name='SceneAppCompatDescriptorInput'], ImportSpecifier[imported.name='SceneAppCompatCatalogInput'], ImportSpecifier[imported.name='SceneAppCompatExecutionPlanInput'], ImportSpecifier[imported.name='SceneAppCompatPlanResultInput']",
    message:
      "SceneApp raw compat 输入类型只允许停留在 API 归一化边界；业务层与测试夹具请改用 SceneAppCurrent* 或 current 业务投影。",
  },
];

const sceneAppCompatTokenSelectors = [
  {
    selector:
      "Literal[value='cloud_managed'], Literal[value='cloud_runtime'], Literal[value='cloud_session'], Literal[value='cloud_scene'], Literal[value='launch_cloud_scene'], Literal[value='cloudSessionReady'], Literal[value='cloud_session_ready']",
    message:
      "SceneApp compat token 只允许停留在 API/目录编译/compat 展示边界；业务层请统一使用 current 语义。",
  },
  {
    selector:
      "Property[key.name='cloudSessionReady'], TSPropertySignature[key.name='cloudSessionReady'], MemberExpression[property.name='cloudSessionReady']",
    message:
      "cloudSessionReady 只允许停留在 SceneApp compat input 边界；业务层请统一改用 directorySessionReadyCompat 或 current 语义。",
  },
  {
    selector:
      "Property[key.name='cloud_session_ready'], TSPropertySignature[key.name='cloud_session_ready'], MemberExpression[property.name='cloud_session_ready']",
    message:
      "cloud_session_ready 只允许停留在 SceneApp compat input 边界；业务层请统一改用 directorySessionReadyCompat 或 current 语义。",
  },
];

const endpointProvidersCommandSelectors = [
  "get_endpoint_providers",
  "set_endpoint_provider",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "端点 Provider 配置命令请统一通过 `src/lib/api/endpointProviders.ts` 暴露的网关函数调用，避免继续在其他模块中直接拼接命令名。",
}));

const voiceToolsCommandSelectors = ["test_tts", "get_available_voices"].map(
  (command) => ({
    selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
    message:
      "语音测试命令请统一通过 `src/lib/api/voiceTools.ts` 暴露的网关函数调用，避免继续在其他模块中直接拼接命令名。",
  }),
);

const profileAssetsCommandSelectors = ["upload_avatar", "delete_avatar"].map(
  (command) => ({
    selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
    message:
      "头像资产命令请统一通过 `src/lib/api/profileAssets.ts` 暴露的网关函数调用，避免继续在其他模块中直接拼接命令名。",
  }),
);

const modelCatalogCommandSelectors = ["get_available_models"].map(
  (command) => ({
    selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
    message:
      "模型列表查询命令请统一通过 `src/lib/api/modelCatalog.ts` 暴露的网关函数调用，避免继续在其他模块中直接拼接命令名。",
  }),
);

const usageStatsCommandSelectors = [
  "get_usage_stats",
  "get_model_usage_ranking",
  "get_daily_usage_trends",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "使用统计命令请统一通过 `src/lib/api/usageStats.ts` 暴露的网关函数调用，避免继续在其他模块中直接拼接命令名。",
}));

const providerRuntimeCommandSelectors = [
  "refresh_kiro_token",
  "reload_credentials",
  "get_kiro_credentials",
  "get_env_variables",
  "get_token_file_hash",
  "check_and_reload_credentials",
  "get_gemini_credentials",
  "reload_gemini_credentials",
  "refresh_gemini_token",
  "get_gemini_env_variables",
  "get_gemini_token_file_hash",
  "check_and_reload_gemini_credentials",
  "get_qwen_credentials",
  "reload_qwen_credentials",
  "refresh_qwen_token",
  "get_qwen_env_variables",
  "get_qwen_token_file_hash",
  "check_and_reload_qwen_credentials",
  "get_openai_custom_status",
  "set_openai_custom_config",
  "get_claude_custom_status",
  "set_claude_custom_config",
].map((command) => ({
  selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
  message:
    "provider 凭证/自定义状态相关后端命令请统一通过 `src/lib/api/providerRuntime.ts` 暴露的网关函数调用，避免继续在其他模块中直接拼接命令名。",
}));

const serverToolsCommandSelectors = ["test_api", "get_network_info"].map(
  (command) => ({
    selector: `CallExpression[callee.name='safeInvoke'][arguments.0.value='${command}'], CallExpression[callee.name='invoke'][arguments.0.value='${command}']`,
    message:
      "API 测试/网络信息相关后端命令请统一通过 `src/lib/api/serverTools.ts` 暴露的网关函数调用，避免继续在其他模块中直接拼接命令名。",
  }),
);

export default [
  { ignores: ["dist", "lime-rs", "node_modules", "**/*.d.ts"] },
  {
    files: ["**/*.{ts,tsx}"],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
      parser: tsparser,
      parserOptions: {
        ecmaFeatures: { jsx: true },
      },
    },
    plugins: {
      "@typescript-eslint": tseslint,
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    rules: {
      ...js.configs.recommended.rules,
      ...tseslint.configs.recommended.rules,
      ...reactHooks.configs.recommended.rules,
      "react-refresh/only-export-components": [
        "warn",
        {
          allowConstantExport: true,
          allowExportNames: [
            // AddCustomProviderModal.tsx
            "validateCustomProviderForm",
            "isFormValid",
            "hasRequiredFields",
            // ApiKeyItem.tsx
            "extractApiKeyDisplayInfo",
            // ApiKeyList.tsx
            "getApiKeyListStats",
            // ApiKeyProviderSection.tsx
            "verifyProviderSelectionSync",
            "extractSelectionState",
            // ConnectionTestButton.tsx
            "getConnectionTestStatusInfo",
            // DeleteProviderDialog.tsx
            "canDeleteProvider",
            "isSystemProvider",
            // ProviderConfigForm.tsx
            "getFieldsForProviderType",
            "providerTypeRequiresField",
            // ProviderGroup.tsx
            "getGroupLabel",
            "isProviderInGroup",
            "getGroupOrder",
            // ProviderList.tsx
            "filterProviders",
            "groupProviders",
            "matchesSearchQuery",
            // ProviderListItem.tsx
            "extractListItemDisplayInfo",
            "getApiKeyCount",
            // ProviderSetting.tsx
            "extractProviderSettingInfo",
            // icons/providers/index.tsx
            "iconComponents",
          ],
        },
      ],
      "no-restricted-imports": createLegacyChatImportRule(
        generalChatRestrictedPaths,
      ),
      "no-restricted-syntax": [
        "error",
        ...generalChatCompatCommandSelectors,
        ...agentRuntimeCommandSelectors,
        ...projectGatewayCommandSelectors,
        ...materialGatewayCommandSelectors,
        ...templateGatewayCommandSelectors,
        ...personaGatewayCommandSelectors,
        ...brandPersonaGatewayCommandSelectors,
        ...posterMaterialGatewayCommandSelectors,
        ...fileSystemCommandSelectors,
        ...fileBrowserCommandSelectors,
        ...appUpdateCommandSelectors,
        ...screenshotChatCommandSelectors,
        ...systemSupportCommandSelectors,
        ...serverRuntimeCommandSelectors,
        ...logCommandSelectors,
        ...appConfigCommandSelectors,
        ...channelsRuntimeCommandSelectors,
        ...experimentalFeaturesCommandSelectors,
        ...memoryRuntimeCommandSelectors,
        ...projectMemoryCommandSelectors,
        ...toolHooksCommandSelectors,
        ...a2uiFormCommandSelectors,
        ...memoryFeedbackCommandSelectors,
        ...contextMemoryCommandSelectors,
        ...asrProviderCommandSelectors,
        ...contentWorkflowCommandSelectors,
        ...novelCommandSelectors,
        ...unifiedChatCommandSelectors,
        ...unifiedMemoryCommandSelectors,
        ...apiCompatibilityCommandSelectors,
        ...endpointProvidersCommandSelectors,
        ...modelCatalogCommandSelectors,
        ...profileAssetsCommandSelectors,
        ...usageStatsCommandSelectors,
        ...providerRuntimeCommandSelectors,
        ...serverToolsCommandSelectors,
        ...voiceToolsCommandSelectors,
      ],
      "@typescript-eslint/no-unused-vars": [
        "error",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          caughtErrorsIgnorePattern: "^_",
        },
      ],
      "@typescript-eslint/no-explicit-any": "off",
    },
  },
  {
    files: ["src/components/general-chat/store/useGeneralChatStore.ts"],
    rules: {
      "no-restricted-imports": createLegacyChatImportRule(
        generalChatRestrictedPathsWithoutCompatApi,
      ),
    },
  },
  {
    files: ["src/lib/api/generalChatCompat.ts"],
    rules: {
      "no-restricted-syntax": "off",
    },
  },
  {
    files: [
      "src/lib/api/agent.test.ts",
      "src/components/agent/chat/hooks/useAgentChat.test.tsx",
      "src/stores/agentStore.test.ts",
    ],
    rules: {
      "no-restricted-imports": createLegacyChatImportRule(
        generalChatRestrictedPaths,
        { includeAgentCompatPatterns: false },
      ),
    },
  },
  {
    files: [
      "src/lib/api/agentRuntime.ts",
      "src/lib/api/agentRuntime/**/*.ts",
      "src/lib/api/agentCompat.ts",
      "src/lib/api/project.ts",
      "src/lib/api/materials.ts",
      "src/lib/api/templates.ts",
      "src/lib/api/personas.ts",
      "src/lib/api/brandPersona.ts",
      "src/lib/api/posterMaterials.ts",
      "src/lib/api/subAgentScheduler.ts",
      "src/lib/api/fileSystem.ts",
      "src/lib/api/fileBrowser.ts",
      "src/lib/api/a2uiForm.ts",
      "src/lib/api/appUpdate.ts",
      "src/lib/api/asrProvider.ts",
      "src/lib/api/screenshotChat.ts",
      "src/lib/api/notification.ts",
      "src/lib/api/novel.ts",
      "src/lib/api/autoFix.ts",
      "src/lib/api/frontendCrash.ts",
      "src/lib/api/toolHooks.ts",
      "src/lib/api/serverRuntime.ts",
      "src/lib/api/logs.ts",
      "src/lib/api/apiCompatibility.ts",
      "src/lib/api/appConfig.ts",
      "src/lib/api/channelsRuntime.ts",
      "src/lib/api/endpointProviders.ts",
      "src/lib/api/experimentalFeatures.ts",
      "src/lib/api/modelCatalog.ts",
      "src/lib/api/profileAssets.ts",
      "src/lib/api/usageStats.ts",
      "src/lib/api/providerRuntime.ts",
      "src/lib/api/serverTools.ts",
      "src/lib/api/voiceTools.ts",
      "src/lib/api/sceneapp.ts",
    ],
    rules: {
      "no-restricted-syntax": "off",
    },
  },
  {
    files: ["src/lib/api/agentRuntime/**/*.ts"],
    rules: {
      "no-restricted-imports": [
        "error",
        {
          paths: [
            {
              name: "../agentRuntime",
              message:
                "agentRuntime 目录内部必须直连 session/request/evidence/media/tool/execution owner；禁止回绕已退役根入口 `../agentRuntime`。",
            },
            {
              name: "@/lib/api/agentRuntime",
              message:
                "agentRuntime 目录内部必须直连分域 owner；禁止通过 alias 回绕已退役根入口。",
            },
            {
              name: "./types",
              message:
                "agentRuntime/types 已退役；请直接依赖 sessionTypes、requestTypes、evidenceTypes、mediaTaskTypes、toolInventoryTypes 或 agentExecutionRuntime。",
            },
          ],
        },
      ],
    },
  },
  {
    files: ["src/lib/api/*.ts"],
    ignores: ["src/lib/api/agent.test.ts"],
    rules: {
      "no-restricted-imports": [
        "error",
        {
          paths: [
            {
              name: "./agentRuntime",
              message:
                "src/lib/api 根层如需复用 agent runtime，请直接依赖具体分域 client/type owner；禁止恢复已删除的 `./agentRuntime` root aggregate。",
            },
            {
              name: "./agentRuntime/types",
              message:
                "agentRuntime/types 已退役；请直接依赖 sessionTypes、requestTypes、evidenceTypes、mediaTaskTypes、toolInventoryTypes 或 agentExecutionRuntime。",
            },
          ],
        },
      ],
    },
  },
  {
    files: ["src/**/*.{ts,tsx}"],
    ignores: ["src/lib/api/sceneapp.ts"],
    rules: {
      "no-restricted-syntax": [
        "error",
        ...sceneAppCurrentProjectionAliasImportSelectors,
        ...sceneAppWideExecutionTypeImportSelectors,
        ...sceneAppCompatInputTypeImportSelectors,
      ],
    },
  },
  {
    files: [
      "src/lib/sceneapp/**/*.{ts,tsx}",
      "src/components/sceneapps/**/*.{ts,tsx}",
      "src/components/settings-v2/system/automation/**/*.{ts,tsx}",
      "src/components/agent/chat/workspace/**/*.{ts,tsx}",
      "src/components/agent/chat/AgentChatWorkspace.tsx",
    ],
    ignores: [
      "**/*.test.ts",
      "**/*.test.tsx",
      "src/lib/sceneapp/types.ts",
      "src/lib/sceneapp/catalog.ts",
      "src/lib/sceneapp/presentation.ts",
    ],
    rules: {
      "no-restricted-syntax": [
        "error",
        ...sceneAppCurrentProjectionAliasImportSelectors,
        ...sceneAppCompatTokenSelectors,
      ],
    },
  },
  // === 分层 import 边界守卫（R-30）===
  // lib/** 禁止 import components/features/pages（反向依赖）
  {
    files: ["src/lib/**/*.{ts,tsx}"],
    ignores: ["**/*.test.ts", "**/*.test.tsx", "**/*.d.ts"],
    rules: {
      "no-restricted-imports": [
        "error",
        {
          patterns: [
            {
              group: ["@/components/**"],
              message:
                "lib 层禁止 import components 层。将纯逻辑下沉到 lib，或把 UI 留在 components 层消费 lib。详见 internal/refactor/progressive-refactor-plan.md R-30。",
            },
            {
              group: ["@/features/**"],
              message:
                "lib 层禁止 import features 层。将共享逻辑下沉到 lib，或用参数注入/类型契约解耦。详见 internal/refactor/progressive-refactor-plan.md R-30。",
            },
            {
              group: ["@/pages/**"],
              message:
                "lib 层禁止 import pages 层。详见 internal/refactor/progressive-refactor-plan.md R-30。",
            },
          ],
        },
      ],
    },
  },
  // features/** 禁止 import components（UI 组件不应被 features 直接引用）
  {
    files: ["src/features/**/*.{ts,tsx}"],
    ignores: ["**/*.test.ts", "**/*.test.tsx", "**/*.d.ts"],
    rules: {
      "no-restricted-imports": [
        "error",
        {
          patterns: [
            {
              group: ["@/components/**"],
              message:
                "features 层禁止 import components 层。将 UI 组件下沉到共享层，或通过 contracts/类型契约解耦。详见 internal/refactor/progressive-refactor-plan.md R-30。",
            },
          ],
        },
      ],
    },
  },
  // === R-40：业务代码禁止直接 import dev-bridge ===
  {
    files: [
      "src/components/**/*.{ts,tsx}",
      "src/features/**/*.{ts,tsx}",
      "src/hooks/**/*.ts",
    ],
    ignores: ["**/*.test.ts", "**/*.test.tsx", "**/*.d.ts", "src/lib/**"],
    rules: {
      "no-restricted-imports": [
        "error",
        {
          patterns: [
            {
              group: ["@/lib/dev-bridge/**", "@/lib/dev-bridge"],
              message:
                "业务代码禁止直接 import dev-bridge。请改用 lib/api/ 统一入口（如 lib/api/bridgeEvents.ts）。详见 internal/refactor/progressive-refactor-plan.md R-40。",
            },
          ],
        },
      ],
    },
  },
];
