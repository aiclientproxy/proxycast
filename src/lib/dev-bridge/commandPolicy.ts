import { isElectronHostCommandAvailable } from "@/lib/electron-host";

export type DevBridgeCommandTimeoutProfile =
  | "startup-truth"
  | "agent-session-get"
  | "agent-session-list"
  | "agent-session-patch"
  | "agent-session-create"
  | "app-server-turn-start"
  | "app-server-import"
  | "app-server-long-running"
  | "app-server-provider-network"
  | "app-server-read"
  | "agent-runtime"
  | "plugin-installed-write"
  | "plugin-package-inspect"
  | "plugin-ui-runtime-start"
  | "plugin-package"
  | "desktop-user-interaction"
  | "knowledge-compile"
  | "voice-model-download"
  | "layered-design-project"
  | "truth"
  | "default";

/**
 * 前端命令策略的唯一分类入口。
 *
 * current 主链命令必须走真实 Electron/Desktop Host 或 DevBridge；
 * 测试夹具只能通过 invokeMockOnly，不能由生产 invoke 自动回退 mock。
 */
const bridgeTruthCommands = new Set<string>([
  "open_external_url",
  "open_update_window",
  "start_oem_cloud_oauth_callback_bridge",
  "get_or_create_default_project",
  "workspace_list",
  "workspace_get_default",
  "workspace_get",
  "workspace_ensure",
  "workspace_ensure_ready",
  "plugin_start_ui_runtime",
  "plugin_get_ui_runtime_status",
  "plugin_stop_ui_runtime",
  "get_default_provider",
  "app_server_handle_json_lines",
  "app_server_drain_events",
  "get_file_name",
]);

const noMockFallbackCompatCommands = new Set<string>([
  "plugin_runtime_start_task",
  "plugin_runtime_cancel_task",
  "plugin_runtime_get_task",
  "plugin_runtime_submit_host_response",
]);

const electronHostNoMockFallbackCommands = new Set([
  "plugin_select_directory",
  "open_file_preview_window",
  "open_resource_manager_window",
  "open_system_settings_url",
  "save_layered_design_project_export",
  "read_layered_design_project_export",
  "recognize_layered_design_text",
  "analyze_layered_design_flat_image",
  "voice_models_delete",
  "voice_models_download",
]);

const optionalLegacyUxCommands = new Set<string>(["get_hint_routes"]);

const devBridgePluginUiRuntimeStartCommands = new Set([
  "plugin_start_ui_runtime",
]);

const devBridgePluginPackageCommands = new Set([
  "pluginLocalPackage/inspect",
  "pluginLocalPackage/export",
]);

const electronHostLayeredDesignProjectCommands = new Set([
  "save_layered_design_project_export",
  "read_layered_design_project_export",
]);

const electronHostUserInteractionCommands = new Set([
  "plugin_select_directory",
]);

const devBridgeCooldownBypassCommands = new Set([
  "get_or_create_default_project",
  "workspace_get",
  "workspace_get_default",
  "workspace_list",
  "workspace_ensure",
  "workspace_ensure_ready",
  "workspace_ensure_default_ready",
]);

const devBridgeReadRetryCommands = new Set<string>([
  "workspace_get",
  "workspace_get_default",
  "workspace_list",
  "workspace_ensure",
  "workspace_ensure_ready",
  "workspace_ensure_default_ready",
]);

const devBridgeStartupTruthCommands = new Set([
  "workspace_ensure_ready",
  "workspace_ensure_default_ready",
]);

const APP_SERVER_HANDLE_JSON_LINES_COMMAND = "app_server_handle_json_lines";
const APP_SERVER_DRAIN_EVENTS_COMMAND = "app_server_drain_events";
const APP_SERVER_THREAD_LIST_METHOD = "thread/list";
const APP_SERVER_TURN_START_METHOD = "turn/start";
const APP_SERVER_CONVERSATION_IMPORT_METHODS = new Set([
  "conversationImport/job/read",
  "conversationImport/source/scan",
  "conversationImport/thread/preview",
  "conversationImport/thread/commit",
]);
const APP_SERVER_PLUGIN_UI_RUNTIME_START_METHOD = "pluginUiRuntime/start";
const APP_SERVER_KNOWLEDGE_COMPILE_METHOD = "knowledgePack/compile";
const APP_SERVER_LONG_RUNNING_METHODS = new Set(["automationJob/runNow"]);
const APP_SERVER_PLUGIN_INSTALLED_WRITE_METHODS = new Set([
  "pluginInstalled/save",
]);
const APP_SERVER_PROVIDER_NETWORK_METHODS = new Set([
  "modelProvider/testConnection",
  "modelProvider/testChat",
  "modelProvider/fetchModels",
]);
const APP_SERVER_MODEL_CONTROL_METHODS = new Set([
  "modelProvider/list",
  "modelProvider/catalog/list",
  "modelProvider/read",
  "modelProvider/create",
  "modelProvider/update",
  "modelProvider/delete",
  "modelProvider/sortOrders/update",
  "modelProviderConfig/export",
  "modelProviderConfig/import",
  ...APP_SERVER_PROVIDER_NETWORK_METHODS,
  "modelProviderKey/create",
  "modelProviderKey/update",
  "modelProviderKey/delete",
  "modelProviderUiState/read",
  "modelProviderUiState/write",
  "modelProviderAlias/read",
  "modelProviderAlias/list",
]);
const APP_SERVER_CURRENT_METHODS = new Set([
  "capability/list",
  "artifact/read",
  "fileSystem/listDirectory",
  "fileSystem/readFilePreview",
  "fileSystem/createFile",
  "fileSystem/createDirectory",
  "fileSystem/renameFile",
  "fileSystem/deleteFile",
  "thread/start",
  "thread/read",
  "workflow/read",
  "workflow/cancel",
  "workflow/retry",
  "workflow/respond",
  "thread/settings/update",
  "thread/archive",
  "thread/unarchive",
  "skill/list",
  "skill/read",
  "skillManagement/list",
  "skillManagement/install",
  "skillManagement/uninstall",
  "skillRepository/list",
  "skillRepository/save",
  "skillRepository/delete",
  "skillCache/refresh",
  "skillInstalledDirectories/list",
  "skillLocal/inspect",
  "skillLocal/scaffold/create",
  "skillLocal/import",
  "skillRemote/inspect",
  "workspaceSkillBindings/list",
  "workspaceRegisteredSkills/list",
  "pluginLocalPackage/inspect",
  "pluginLocalPackage/export",
  "pluginPackage/fetchCloud",
  "pluginInstalled/save",
  "pluginInstalled/list",
  "pluginInstalled/disabled/set",
  "pluginInstalled/uninstall/rehearsal",
  "pluginInstalled/uninstall",
  "pluginShell/prepare",
  "pluginUiRuntime/status",
  "knowledgePack/list",
  "knowledgePack/read",
  "knowledgePack/source/import",
  "knowledgePack/compile",
  "knowledgePack/default/set",
  "knowledgePack/status/update",
  "knowledgeContext/resolve",
  "knowledgeContextRun/validate",
  "automationScheduler/config/read",
  "automationScheduler/config/update",
  "automationScheduler/status",
  "automationJob/list",
  "automationJob/read",
  "automationJob/create",
  "automationJob/update",
  "automationJob/delete",
  "automationJob/runNow",
  "automationJob/health",
  "automationJob/runHistory",
  "automationSchedule/preview",
  "automationSchedule/validate",
  "projectMemory/read",
  "gatewayChannel/status",
  "wechatChannel/accounts/list",
  "mediaTaskArtifact/image/create",
  "mediaTaskArtifact/audio/create",
  "mediaTaskArtifact/image/complete",
  "mediaTaskArtifact/audio/complete",
  "mediaTaskArtifact/get",
  "mediaTaskArtifact/list",
  "mediaTaskArtifact/cancel",
  "workspaceRightSurface/request",
  "workspaceRightSurface/pending/list",
  "workspaceRightSurface/pending/consume",
  "workspaceRightSurface/pending/dismiss",
  "sessionFile/getOrCreate",
  "sessionFile/updateMeta",
  "sessionFile/save",
  "sessionFile/list",
  "sessionFile/read",
  "sessionFile/resolvePath",
  "sessionFile/delete",
  "model/list",
  "modelPreferences/list",
  "modelSyncState/read",
  ...APP_SERVER_MODEL_CONTROL_METHODS,
  "mcpServer/list",
  "mcpServerStatus/list",
  "mcpServer/create",
  "mcpServer/update",
  "mcpServer/delete",
  "mcpServer/enabled/set",
  "mcpServer/importFromApp",
  "mcpServer/syncAllToLive",
  "mcpServer/oauth/login",
  "mcpServer/start",
  "mcpServer/stop",
  "mcpTool/list",
  "mcpTool/listForContext",
  "mcpTool/search",
  "mcpTool/call",
  "mcpTool/callWithCaller",
  "mcpPrompt/list",
  "mcpPrompt/get",
  "mcpResource/list",
  "mcpResource/read",
  "mcpResource/subscribe",
  "mcpResource/unsubscribe",
  "connectDeepLink/resolve",
  "connectOpenDeepLink/resolve",
  "voiceAsrCredential/list",
  "voiceAsrCredential/create",
  "voiceAsrCredential/update",
  "voiceAsrCredential/delete",
  "voiceAsrCredential/default/set",
  "voiceAsrCredential/test",
  "voiceInstruction/list",
  "voiceInstruction/save",
  "voiceInstruction/delete",
  "voiceModel/default/set",
  "voiceModel/testTranscribeFile",
  "voiceTranscription/transcribeAudio",
  "voiceTranscription/polishText",
  "conversationImport/source/scan",
  "conversationImport/job/read",
  "conversationImport/thread/preview",
  "conversationImport/thread/commit",
  "projectGit/status",
  "projectGit/diff",
  "projectGit/commits/list",
  "projectGit/branch/checkout",
  "projectGit/branch/create",
  "projectGit/worktree/create",
]);
const APP_SERVER_STARTUP_TRUTH_METHODS = new Set([
  "workspace/default/read",
  "workspace/default/ensure",
  "workspace/list",
  "workspace/read",
  "workspace/update",
  "workspace/delete",
  "workspace/byPath/read",
  "workspace/ensure",
  "workspace/projectsRoot/read",
  "workspace/projectPath/resolve",
  "workspace/ensureReady",
]);

const bridgeTruthEventPrefixes = [
  "voice-model-download-progress",
  "agent_stream_",
  "embedded-browser-view-",
  "mcp:",
];

export function isBridgeTruthCommand(command: string): boolean {
  return bridgeTruthCommands.has(command);
}

export function shouldDisallowMockFallbackCommand(command: string): boolean {
  return (
    isBridgeTruthCommand(command) ||
    noMockFallbackCompatCommands.has(command) ||
    electronHostNoMockFallbackCommands.has(command)
  );
}

export function isOptionalLegacyUxCommand(command: string): boolean {
  return optionalLegacyUxCommands.has(command);
}

export function isOptionalLegacyUxCommandAvailable(command: string): boolean {
  return (
    isOptionalLegacyUxCommand(command) &&
    isElectronHostCommandAvailable(command)
  );
}

export function areOptionalLegacyUxCommandsAvailable(
  commands: string[],
): boolean {
  return commands.every(isOptionalLegacyUxCommandAvailable);
}

export function isBridgeTruthEvent(eventName: string): boolean {
  const normalizedEventName = eventName.trim();
  if (!normalizedEventName) {
    return false;
  }
  return bridgeTruthEventPrefixes.some((prefix) =>
    normalizedEventName.startsWith(prefix),
  );
}

export function shouldBypassDevBridgeCooldown(command: string): boolean {
  return devBridgeCooldownBypassCommands.has(command);
}

export function shouldRetryDevBridgeReadCommand(command: string): boolean {
  return devBridgeReadRetryCommands.has(command);
}

export function resolveDevBridgeCommandTimeoutProfile(
  command: string,
  args?: unknown,
): DevBridgeCommandTimeoutProfile {
  if (devBridgeStartupTruthCommands.has(command)) {
    return "startup-truth";
  }
  if (isAppServerTurnStartCommand(command, args)) {
    return "app-server-turn-start";
  }
  if (isAppServerConversationImportCommand(command, args)) {
    return "app-server-import";
  }
  if (isAppServerPluginUiRuntimeStartCommand(command, args)) {
    return "plugin-ui-runtime-start";
  }
  if (isAppServerLongRunningCommand(command, args)) {
    return "app-server-long-running";
  }
  if (isAppServerThreadListCommand(command, args)) {
    return "agent-session-list";
  }
  if (isAppServerStartupTruthCommand(command, args)) {
    return "startup-truth";
  }
  if (isAppServerKnowledgeCompileCommand(command, args)) {
    return "knowledge-compile";
  }
  if (isAppServerPluginPackageInspectCommand(command, args)) {
    return "plugin-package-inspect";
  }
  if (isAppServerPluginInstalledWriteCommand(command, args)) {
    return "plugin-installed-write";
  }
  if (isAppServerProviderNetworkCommand(command, args)) {
    return "app-server-provider-network";
  }
  if (isAppServerCurrentMethodCommand(command, args)) {
    return "app-server-read";
  }
  if (command === APP_SERVER_DRAIN_EVENTS_COMMAND) {
    return "app-server-read";
  }
  if (command.startsWith("plugin_runtime_")) {
    return "agent-runtime";
  }
  if (devBridgePluginUiRuntimeStartCommands.has(command)) {
    return "plugin-ui-runtime-start";
  }
  if (devBridgePluginPackageCommands.has(command)) {
    return "plugin-package";
  }
  if (electronHostUserInteractionCommands.has(command)) {
    return "desktop-user-interaction";
  }
  if (command === "voice_models_download") {
    return "voice-model-download";
  }
  if (electronHostLayeredDesignProjectCommands.has(command)) {
    return "layered-design-project";
  }
  if (isBridgeTruthCommand(command)) {
    return "truth";
  }
  return "default";
}

function isAppServerThreadListCommand(command: string, args: unknown): boolean {
  if (command !== APP_SERVER_HANDLE_JSON_LINES_COMMAND) {
    return false;
  }
  return extractAppServerJsonLines(args).some((line) =>
    jsonRpcLineHasMethod(line, APP_SERVER_THREAD_LIST_METHOD),
  );
}

function isAppServerTurnStartCommand(command: string, args: unknown): boolean {
  if (command !== APP_SERVER_HANDLE_JSON_LINES_COMMAND) {
    return false;
  }
  return extractAppServerJsonLines(args).some((line) =>
    jsonRpcLineHasMethod(line, APP_SERVER_TURN_START_METHOD),
  );
}

function isAppServerConversationImportCommand(
  command: string,
  args: unknown,
): boolean {
  if (command !== APP_SERVER_HANDLE_JSON_LINES_COMMAND) {
    return false;
  }
  return extractAppServerJsonLines(args).some((line) =>
    jsonRpcLineHasAnyMethod(line, APP_SERVER_CONVERSATION_IMPORT_METHODS),
  );
}

function isAppServerPluginUiRuntimeStartCommand(
  command: string,
  args: unknown,
): boolean {
  if (command !== APP_SERVER_HANDLE_JSON_LINES_COMMAND) {
    return false;
  }
  return extractAppServerJsonLines(args).some((line) =>
    jsonRpcLineHasMethod(line, APP_SERVER_PLUGIN_UI_RUNTIME_START_METHOD),
  );
}

function isAppServerStartupTruthCommand(
  command: string,
  args: unknown,
): boolean {
  if (command !== APP_SERVER_HANDLE_JSON_LINES_COMMAND) {
    return false;
  }
  return extractAppServerJsonLines(args).some((line) =>
    jsonRpcLineHasAnyMethod(line, APP_SERVER_STARTUP_TRUTH_METHODS),
  );
}

function isAppServerKnowledgeCompileCommand(
  command: string,
  args: unknown,
): boolean {
  if (command !== APP_SERVER_HANDLE_JSON_LINES_COMMAND) {
    return false;
  }
  return extractAppServerJsonLines(args).some((line) =>
    jsonRpcLineHasMethod(line, APP_SERVER_KNOWLEDGE_COMPILE_METHOD),
  );
}

function isAppServerLongRunningCommand(
  command: string,
  args: unknown,
): boolean {
  if (command !== APP_SERVER_HANDLE_JSON_LINES_COMMAND) {
    return false;
  }
  return extractAppServerJsonLines(args).some((line) =>
    jsonRpcLineHasAnyMethod(line, APP_SERVER_LONG_RUNNING_METHODS),
  );
}

function isAppServerCurrentMethodCommand(
  command: string,
  args: unknown,
): boolean {
  if (command !== APP_SERVER_HANDLE_JSON_LINES_COMMAND) {
    return false;
  }
  return extractAppServerJsonLines(args).some((line) =>
    jsonRpcLineHasAnyMethod(line, APP_SERVER_CURRENT_METHODS),
  );
}

function isAppServerPluginPackageInspectCommand(
  command: string,
  args: unknown,
): boolean {
  if (command !== APP_SERVER_HANDLE_JSON_LINES_COMMAND) {
    return false;
  }
  return extractAppServerJsonLines(args).some(
    (line) =>
      jsonRpcLineHasMethod(line, "pluginLocalPackage/inspect") ||
      jsonRpcLineHasMethod(line, "pluginLocalPackage/export"),
  );
}

function isAppServerPluginInstalledWriteCommand(
  command: string,
  args: unknown,
): boolean {
  if (command !== APP_SERVER_HANDLE_JSON_LINES_COMMAND) {
    return false;
  }
  return extractAppServerJsonLines(args).some((line) =>
    jsonRpcLineHasAnyMethod(line, APP_SERVER_PLUGIN_INSTALLED_WRITE_METHODS),
  );
}

function isAppServerProviderNetworkCommand(
  command: string,
  args: unknown,
): boolean {
  if (command !== APP_SERVER_HANDLE_JSON_LINES_COMMAND) {
    return false;
  }
  return extractAppServerJsonLines(args).some((line) =>
    jsonRpcLineHasAnyMethod(line, APP_SERVER_PROVIDER_NETWORK_METHODS),
  );
}

function extractAppServerJsonLines(args: unknown): string[] {
  if (!args || typeof args !== "object" || Array.isArray(args)) {
    return [];
  }
  const request = (args as { request?: unknown }).request;
  if (!request || typeof request !== "object" || Array.isArray(request)) {
    return [];
  }
  const lines = (request as { lines?: unknown }).lines;
  if (!Array.isArray(lines)) {
    return [];
  }
  return lines.filter((line): line is string => typeof line === "string");
}

function jsonRpcLineHasMethod(line: string, method: string): boolean {
  return jsonRpcLineHasAnyMethod(line, new Set([method]));
}

function jsonRpcLineHasAnyMethod(
  line: string,
  methods: ReadonlySet<string>,
): boolean {
  try {
    const parsed = JSON.parse(line.trim()) as { method?: unknown } | null;
    return Boolean(
      parsed &&
      typeof parsed === "object" &&
      typeof parsed.method === "string" &&
      methods.has(parsed.method),
    );
  } catch {
    return false;
  }
}
