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
  "get_default_provider",
  "app_server_handle_json_lines",
  "app_server_drain_events",
  "get_file_name",
]);

const electronHostNoMockFallbackCommands = new Set([
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

const electronHostLayeredDesignProjectCommands = new Set([
  "save_layered_design_project_export",
  "read_layered_design_project_export",
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
const APP_SERVER_KNOWLEDGE_COMPILE_METHOD = "knowledgePack/compile";
const APP_SERVER_LONG_RUNNING_METHODS = new Set(["scheduledTask/run/start"]);
const APP_SERVER_PROVIDER_NETWORK_METHODS = new Set([
  "modelProvider/testConnection",
  "modelProvider/testChat",
  "modelProvider/fetchModels",
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
  if (isAppServerProviderNetworkCommand(command, args)) {
    return "app-server-provider-network";
  }
  if (command === APP_SERVER_HANDLE_JSON_LINES_COMMAND) {
    return "app-server-read";
  }
  if (command === APP_SERVER_DRAIN_EVENTS_COMMAND) {
    return "app-server-read";
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
