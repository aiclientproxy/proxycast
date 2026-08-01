import fs from "node:fs";
import path from "node:path";

const appServerClientIndexFile = "packages/app-server-client/src/index.ts";
const appServerGeneratedProtocolFile =
  "packages/app-server-client/src/generated/protocol-types.ts";
const appServerClientSplitSourceFiles = [
  appServerClientIndexFile,
  "packages/app-server-client/src/request-client.ts",
  "packages/app-server-client/src/request-client-methods.ts",
  "packages/app-server-client/src/connection.ts",
  "packages/app-server-client/src/connection-methods.ts",
  "packages/app-server-client/src/sidecar.ts",
  "packages/app-server-client/src/sidecar-types.ts",
  "packages/app-server-client/src/sidecar-manifest.ts",
  "packages/app-server-client/src/sidecar-process.ts",
  "packages/app-server-client/src/sidecar-lifecycle.ts",
  "packages/app-server-client/src/agent-runtime.ts",
];

function normalizeContractSnippet(value) {
  return value
    .replace(/\b(?:protocol|appServer|constants)\./gu, "")
    .replace(/(\w+)\s*:\s*([A-Za-z0-9_<>,\[\]\s|&]+)\s*=\s*\{\}/gu, "$1?: $2")
    .replace(/\basync\s+(?=[A-Za-z_$][\w$]*\()/gu, "")
    .replace(/\.await\b/gu, "")
    .replace(/\.boxed\(\)/gu, "")
    .replace(/\bready\(/gu, "")
    .replace(/,\s*\)/gu, ")")
    .replace(/\s+/gu, "");
}

function contractContentIncludes(content, snippet) {
  if (content.includes(snippet)) {
    return true;
  }
  return normalizeContractSnippet(content).includes(
    normalizeContractSnippet(snippet),
  );
}

function expandContractFiles(files) {
  return [
    ...new Set(
      files.flatMap((file) =>
        file === appServerClientIndexFile
          ? appServerClientSplitSourceFiles
          : [file],
      ),
    ),
  ];
}

function requiredContractContent(repoRoot, files, content) {
  if (!files.includes("packages/app-server-client/src/protocol.ts")) {
    return content;
  }
  const generatedPath = path.join(repoRoot, appServerGeneratedProtocolFile);
  if (!fs.existsSync(generatedPath)) {
    return content;
  }
  return `${content}\n${fs.readFileSync(generatedPath, "utf8")}`;
}

function collectSourceFiles(root) {
  if (!fs.existsSync(root)) {
    return [];
  }
  const ignoredDirectories = new Set([
    ".git",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "target",
  ]);
  const sourceExtensions = new Set([
    ".cjs",
    ".js",
    ".jsx",
    ".json",
    ".mjs",
    ".rs",
    ".ts",
    ".tsx",
  ]);
  const files = [];
  const pending = [root];
  while (pending.length > 0) {
    const current = pending.pop();
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const absolutePath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        if (!ignoredDirectories.has(entry.name)) {
          pending.push(absolutePath);
        }
      } else if (sourceExtensions.has(path.extname(entry.name))) {
        files.push(absolutePath);
      }
    }
  }
  return files;
}

export function checkMcpRuntimeCurrentContracts({ repoRoot, failures }) {
  const requiredByFile = new Map([
    [
      "lime-rs/crates/app-server-protocol/src/protocol/v0/method_names.rs",
      [
        'pub const METHOD_MCP_SERVER_CREATE: &str = "mcpServer/create"',
        'pub const METHOD_MCP_SERVER_UPDATE: &str = "mcpServer/update"',
        'pub const METHOD_MCP_SERVER_DELETE: &str = "mcpServer/delete"',
        'pub const METHOD_MCP_SERVER_ENABLED_SET: &str = "mcpServer/enabled/set"',
        'pub const METHOD_MCP_SERVER_IMPORT_FROM_APP: &str = "mcpServer/importFromApp"',
        'pub const METHOD_MCP_SERVER_SYNC_ALL_TO_LIVE: &str = "mcpServer/syncAllToLive"',
        'pub const METHOD_MCP_SERVER_OAUTH_LOGIN: &str = "mcpServer/oauth/login"',
        'pub const METHOD_MCP_SERVER_START: &str = "mcpServer/start"',
        'pub const METHOD_MCP_SERVER_STOP: &str = "mcpServer/stop"',
        'pub const METHOD_MCP_TOOL_LIST_FOR_CONTEXT: &str = "mcpTool/listForContext"',
        'pub const METHOD_MCP_TOOL_SEARCH: &str = "mcpTool/search"',
        'pub const METHOD_MCP_TOOL_CALL: &str = "mcpTool/call"',
        'pub const METHOD_MCP_TOOL_CALL_WITH_CALLER: &str = "mcpTool/callWithCaller"',
        'pub const METHOD_MCP_PROMPT_GET: &str = "mcpPrompt/get"',
        'pub const METHOD_MCP_RESOURCE_LIST: &str = "mcpResource/list"',
        'pub const METHOD_MCP_RESOURCE_READ: &str = "mcpResource/read"',
        'pub const METHOD_MCP_RESOURCE_SUBSCRIBE: &str = "mcpResource/subscribe"',
        'pub const METHOD_MCP_RESOURCE_UNSUBSCRIBE: &str = "mcpResource/unsubscribe"',
      ],
    ],
    [
      "lime-rs/crates/app-server-protocol/src/protocol/v0/mcp.rs",
      [
        "pub struct McpServerCreateParams",
        "pub struct McpServerUpdateParams",
        "pub struct McpServerDeleteParams",
        "pub struct McpServerEnabledSetParams",
        "pub struct McpServerImportFromAppParams",
        "pub struct McpServerImportFromAppResponse",
        "pub struct McpServerOauthLoginParams",
        "pub struct McpServerOauthLoginResponse",
        "pub struct McpToolCallParams",
        "pub struct McpToolCallResponse",
        "pub struct McpPromptGetParams { #[schemars(length(min = 1))] pub server: String, #[schemars(length(min = 1))] pub name: String",
        "pub struct McpResourceListResponse",
        "pub resource_templates: Vec<serde_json::Value>",
        "pub struct McpResourceReadParams { #[schemars(length(min = 1))] pub server: String, #[schemars(length(min = 1))] pub uri: String",
        "pub struct McpResourceSubscribeParams { #[schemars(length(min = 1))] pub server: String, #[schemars(length(min = 1))] pub uri: String",
        "pub struct McpResourceUnsubscribeParams { #[schemars(length(min = 1))] pub server: String, #[schemars(length(min = 1))] pub uri: String",
        "pub struct McpResourceSubscriptionResponse",
      ],
    ],
    [
      [
        "lime-rs/crates/app-server-protocol/src/protocol/v2/mcp.rs",
        "lime-rs/crates/app-server-protocol/src/protocol/v2/methods.rs",
        "lime-rs/crates/app-server-protocol/src/protocol/v2/envelopes.rs",
        "lime-rs/crates/app-server/src/processor/mcp.rs",
      ],
      [
        'pub const METHOD_MCP_SERVER_OAUTH_LOGIN_COMPLETED: &str = "mcpServer/oauthLogin/completed"',
        'pub const METHOD_MCP_SERVER_STARTUP_STATUS_UPDATED: &str = "mcpServer/startupStatus/updated"',
        "pub struct McpServerOauthLoginCompletedNotification",
        "pub struct McpServerStatusUpdatedNotification",
        "McpServerOauthLoginCompleted(McpServerOauthLoginCompletedNotification)",
        "McpServerStatusUpdated(McpServerStatusUpdatedNotification)",
        "let (success, error) = match handle.wait().await",
        "publish_server_notification(ServerNotification::McpServerOauthLoginCompleted(",
        "publish_server_notification(ServerNotification::McpServerStatusUpdated(",
      ],
    ],
    [
      [
        "lime-rs/crates/app-server/src/runtime.rs",
        "lime-rs/crates/app-server/src/runtime/mcp.rs",
      ],
      [
        "pub async fn create_mcp_server(",
        "pub async fn update_mcp_server(",
        "pub async fn delete_mcp_server(",
        "pub async fn set_mcp_server_enabled(",
        "pub async fn import_mcp_servers_from_app(",
        "pub async fn sync_all_mcp_servers_to_live(",
        "pub async fn login_mcp_server_oauth(",
        "pub async fn call_mcp_tool(",
        "pub async fn call_mcp_tool_with_caller(",
        "pub async fn get_mcp_prompt(",
        "pub async fn list_mcp_resources(",
        "pub async fn read_mcp_resource(",
        "pub async fn subscribe_mcp_resource(",
        "pub async fn unsubscribe_mcp_resource(",
        "self.app_data_source.create_mcp_server(params).await",
        "self.app_data_source.update_mcp_server(params).await",
        "self.app_data_source.delete_mcp_server(params).await",
        "self.app_data_source.set_mcp_server_enabled(params).await",
        ".import_mcp_servers_from_app(params)",
        "self.app_data_source.sync_all_mcp_servers_to_live().await",
        "self.app_data_source.login_mcp_server_oauth(params).await",
        "self.app_data_source.call_mcp_tool(params).await",
        "self.app_data_source.call_mcp_tool_with_caller(params).await",
        "self.app_data_source.get_mcp_prompt(params).await",
        "self.app_data_source.list_mcp_resources().await",
        "self.app_data_source.read_mcp_resource(params).await",
        "self.app_data_source.subscribe_mcp_resource(params).await",
        "self.app_data_source.unsubscribe_mcp_resource(params).await",
      ],
    ],
    [
      [
        "lime-rs/crates/app-server/src/local_data_source.rs",
        "lime-rs/crates/app-server/src/local_data_source/mcp.rs",
      ],
      [
        "mcp_manager: McpManagerState",
        "McpClientManager::new(None)",
        "fn create_mcp_server(",
        "McpService::add(db, server)",
        "fn update_mcp_server(",
        "McpService::update(db, server)",
        "fn delete_mcp_server(",
        "McpService::delete(db, &params.id)",
        "fn set_mcp_server_enabled(",
        "McpService::toggle_enabled(",
        "fn import_mcp_servers_from_app(",
        "McpService::import_from_app(db, &params.app_type)",
        "fn sync_all_mcp_servers_to_live(",
        "McpService::sync_all_to_live(db)",
        "async fn login_mcp_server_oauth(",
        ".start_oauth_login(",
        "async fn call_mcp_tool(",
        "async fn call_mcp_tool_with_caller(",
        ".call_tool(&params.tool_name, params.arguments)",
        ".call_tool_with_caller(",
        ".get_prompt(&params.server, &params.name, params.arguments)",
        "async fn list_mcp_resources(",
        "manager.list_resources()",
        "list_resource_templates()",
        "resource_templates:",
        ".read_resource(&params.server, &params.uri)",
        ".subscribe_resource(&params.server, &params.uri)",
        ".unsubscribe_resource(&params.server, &params.uri)",
      ],
    ],
    [
      "lime-rs/crates/app-server-client/src/lib.rs",
      [
        "pub fn read_mcp_resource(",
        "pub fn subscribe_mcp_resource(",
        "pub fn unsubscribe_mcp_resource(",
        "validate_mcp_resource_target(&params.server, &params.uri)?",
        "METHOD_MCP_RESOURCE_READ",
        "METHOD_MCP_RESOURCE_SUBSCRIBE",
        "METHOD_MCP_RESOURCE_UNSUBSCRIBE",
      ],
    ],
    [
      [
        "lime-rs/crates/app-server/src/processor/mod.rs",
        "lime-rs/crates/app-server/src/processor/dispatch.rs",
        "lime-rs/crates/app-server/src/processor/mcp.rs",
      ],
      [
        "METHOD_MCP_SERVER_CREATE => self.handle_mcp_server_create_impl(params).await",
        "METHOD_MCP_SERVER_UPDATE => self.handle_mcp_server_update_impl(params).await",
        "METHOD_MCP_SERVER_DELETE => self.handle_mcp_server_delete_impl(params).await",
        "METHOD_MCP_SERVER_ENABLED_SET =>",
        "self.handle_mcp_server_enabled_set_impl(params).boxed()",
        "METHOD_MCP_SERVER_IMPORT_FROM_APP =>",
        "METHOD_MCP_SERVER_SYNC_ALL_TO_LIVE =>",
        "self.handle_mcp_server_sync_all_to_live_impl().await",
        "METHOD_MCP_SERVER_OAUTH_LOGIN =>",
        "self.handle_mcp_server_oauth_login_impl(params).await",
        "METHOD_MCP_TOOL_CALL => self.handle_mcp_tool_call_impl(params).await",
        "METHOD_MCP_TOOL_CALL_WITH_CALLER =>",
        "METHOD_MCP_PROMPT_GET => self.handle_mcp_prompt_get_impl(params).await",
        "METHOD_MCP_RESOURCE_LIST => self.handle_mcp_resource_list_impl().await",
        "METHOD_MCP_RESOURCE_READ => self.handle_mcp_resource_read_impl(params).await",
        "METHOD_MCP_RESOURCE_SUBSCRIBE =>",
        "METHOD_MCP_RESOURCE_UNSUBSCRIBE =>",
        "let params: McpServerCreateParams = parse_params(params)?",
        "let params: McpServerUpdateParams = parse_params(params)?",
        "let params: McpServerDeleteParams = parse_params(params)?",
        "let params: McpServerEnabledSetParams = parse_params(params)?",
        "let params: McpServerImportFromAppParams = parse_params(params)?",
        "let params: McpServerOauthLoginParams = parse_params(params)?",
        "let params: McpToolCallParams = parse_params(params)?",
        "let params: McpPromptGetParams = parse_params(params)?",
        "pub(super) async fn handle_mcp_resource_list_impl(",
        "let params: McpResourceReadParams = parse_params(params)?",
        "let params: McpResourceSubscribeParams = parse_params(params)?",
        "let params: McpResourceUnsubscribeParams = parse_params(params)?",
      ],
    ],
    [
      "packages/app-server-client/src/protocol.ts",
      [
        'export const METHOD_MCP_SERVER_CREATE = "mcpServer/create"',
        'export const METHOD_MCP_SERVER_UPDATE = "mcpServer/update"',
        'export const METHOD_MCP_SERVER_DELETE = "mcpServer/delete"',
        'export const METHOD_MCP_SERVER_ENABLED_SET = "mcpServer/enabled/set"',
        'export const METHOD_MCP_SERVER_IMPORT_FROM_APP = "mcpServer/importFromApp"',
        'export const METHOD_MCP_SERVER_SYNC_ALL_TO_LIVE = "mcpServer/syncAllToLive"',
        'export const METHOD_MCP_SERVER_OAUTH_LOGIN = "mcpServer/oauth/login"',
        'export const METHOD_MCP_TOOL_CALL = "mcpTool/call"',
        'export const METHOD_MCP_TOOL_CALL_WITH_CALLER = "mcpTool/callWithCaller"',
        'export const METHOD_MCP_PROMPT_GET = "mcpPrompt/get"',
        'export const METHOD_MCP_RESOURCE_LIST = "mcpResource/list"',
        'export const METHOD_MCP_RESOURCE_READ = "mcpResource/read"',
        'export const METHOD_MCP_RESOURCE_SUBSCRIBE = "mcpResource/subscribe"',
        'export const METHOD_MCP_RESOURCE_UNSUBSCRIBE = "mcpResource/unsubscribe"',
        "export type McpServerCreateParams",
        "export type McpServerUpdateParams",
        "export type McpServerDeleteParams",
        "export type McpServerEnabledSetParams",
        "export type McpServerImportFromAppParams",
        "export type McpServerImportFromAppResponse",
        "export type McpServerOauthLoginParams",
        "export type McpServerOauthLoginResponse",
        "export type McpToolCallParams",
        "export type McpToolCallResponse",
        "export type McpPromptGetParams = { server: string; name: string",
        "export type McpResourceListResponse",
        "resourceTemplates?: unknown[]",
        "export type McpResourceReadParams",
        "export type McpResourceSubscribeParams",
        "export type McpResourceUnsubscribeParams",
        "export type McpResourceSubscriptionResponse",
      ],
    ],
    [
      "packages/app-server-client/src/index.ts",
      [
        "createMcpServer(params: McpServerCreateParams): JsonRpcRequest",
        "updateMcpServer(params: McpServerUpdateParams): JsonRpcRequest",
        "deleteMcpServer(params: McpServerDeleteParams): JsonRpcRequest",
        "setMcpServerEnabled(params: McpServerEnabledSetParams): JsonRpcRequest",
        "importMcpServersFromApp(",
        "syncAllMcpServersToLive(): JsonRpcRequest",
        "loginMcpServerOauth(params: McpServerOauthLoginParams): JsonRpcRequest",
        "callMcpTool(params: McpToolCallParams): JsonRpcRequest",
        "callMcpToolWithCaller(",
        "getMcpPrompt(params: McpPromptGetParams): JsonRpcRequest",
        "listMcpResources(): JsonRpcRequest",
        "readMcpResource(params: McpResourceReadParams): JsonRpcRequest",
        "subscribeMcpResource(params: McpResourceSubscribeParams): JsonRpcRequest",
        "unsubscribeMcpResource(params: McpResourceUnsubscribeParams): JsonRpcRequest",
        "async listMcpResources(",
        "async subscribeMcpResource(",
        "async unsubscribeMcpResource(",
      ],
    ],
    [
      "src/lib/api/mcp.ts",
      [
        'import { AppServerClient } from "@/lib/api/appServer"',
        "METHOD_MCP_SERVER_CREATE",
        "METHOD_MCP_SERVER_UPDATE",
        "METHOD_MCP_SERVER_DELETE",
        "METHOD_MCP_SERVER_ENABLED_SET",
        "METHOD_MCP_SERVER_IMPORT_FROM_APP",
        "METHOD_MCP_SERVER_SYNC_ALL_TO_LIVE",
        "METHOD_MCP_SERVER_OAUTH_LOGIN",
        "METHOD_MCP_TOOL_CALL",
        "METHOD_MCP_TOOL_CALL_WITH_CALLER",
        "METHOD_MCP_PROMPT_GET",
        "METHOD_MCP_RESOURCE_LIST",
        "METHOD_MCP_RESOURCE_READ",
        "METHOD_MCP_RESOURCE_SUBSCRIBE",
        "METHOD_MCP_RESOURCE_UNSUBSCRIBE",
        'export * from "./mcpTypes"',
        'from "./mcpTypes"',
        'from "./mcpResponseGuards"',
        "requestMcpAppServer",
        "McpResourceListResult",
        "listResourcesWithTemplates",
        "subscribeResource",
        "unsubscribeResource",
        "addServer: (server: McpServer): Promise<void> =>",
        "updateServer: (server: McpServer): Promise<void> =>",
        "deleteServer: (id: string): Promise<void> =>",
        "toggleServer: (",
        "importFromApp: (appType: string): Promise<number> =>",
        "syncAllToLive: (): Promise<void> =>",
        "loginOAuthServer: (",
      ],
    ],
    [
      "src/lib/api/mcpResponseGuards.ts",
      [
        "export function assertArrayField",
        "export function assertServerListResponse",
        "export function assertLifecycleResponse",
        "export function assertEmptyResponse",
        "export function assertOAuthLoginResponse",
        "export function assertMcpToolResult",
        "export function assertMcpPromptResult",
        "export function assertMcpResourceContent",
        "export function assertMcpResourceListResponse",
        "McpResourceTemplateDefinition",
        "resourceTemplates",
      ],
    ],
    [
      "src/lib/api/mcpTypes.ts",
      [
        "export interface McpServer",
        "export type McpServerConfig",
        "export interface McpServerRuntimeStatus",
        "export interface McpServerAuthStatus",
        "export interface McpServerOAuthLoginOptions",
        "export interface McpToolDefinition",
        "export type McpContent",
        "export interface McpToolResult",
        "export function getMcpInnerToolName(",
        "export interface McpPromptDefinition",
        "export interface McpPromptResult",
        "export interface McpResourceDefinition",
        "export interface McpResourceTemplateDefinition",
        "export interface McpResourceListResult",
        "export interface McpResourceContent",
      ],
    ],
    [
      "packages/app-server-client/src/server-notifications.ts",
      [
        "export type McpServerOauthLoginCompletedServerNotification = Extract<",
        '{ method: "mcpServer/oauthLogin/completed" }',
        "export function mcpServerOauthLoginCompletedServerNotification(",
        'message.method !== "mcpServer/oauthLogin/completed"',
        '!hasOnlyKeys(params, ["error", "name", "success", "threadId"])',
        '!hasRequiredNullableString(params, "threadId")',
        "export type McpServerStatusUpdatedServerNotification = Extract<",
        '{ method: "mcpServer/startupStatus/updated" }',
        "export function mcpServerStatusUpdatedServerNotification(",
        'message.method !== "mcpServer/startupStatus/updated"',
        '!hasRequiredNullableString(params, "error")',
      ],
    ],
    [
      "src/hooks/useMcpEvents.ts",
      [
        'import { safeListen } from "@/lib/api/bridgeEvents"',
        'import { subscribeAppServerNotifications } from "@/lib/api/appServerEventBus"',
        "mcpServerOauthLoginCompletedServerNotification",
        "mcpServerStatusUpdatedServerNotification",
        "export async function setupMcpEventListeners(",
        '"mcp:tools_updated"',
        '"mcp:resources_updated"',
        '"mcp:resource_updated"',
        "subscribeAppServerNotifications({",
      ],
    ],
    [
      "src/lib/api/mcp.test.ts",
      [
        "appServerRequestMock",
        '"mcpServer/create"',
        '"mcpServer/update"',
        '"mcpServer/delete"',
        '"mcpServer/enabled/set"',
        '"mcpServer/importFromApp"',
        '"mcpServer/syncAllToLive"',
        '"mcpServer/oauth/login"',
        '"mcpTool/call"',
        '"mcpTool/callWithCaller"',
        '"mcpPrompt/get"',
        '"mcpResource/list"',
        "listResourcesWithTemplates",
        "resourceTemplates",
        '"mcpResource/read"',
        '"mcpResource/subscribe"',
        '"mcpResource/unsubscribe"',
        "expect(safeInvoke).not.toHaveBeenCalled()",
      ],
    ],
    [
      "src/lib/api/mcp.failClosed.test.ts",
      [
        "App Server unavailable",
        '"mcpServer/create"',
        '"mcpServer/update"',
        '"mcpServer/delete"',
        '"mcpServer/enabled/set"',
        '"mcpServer/importFromApp"',
        '"mcpServer/syncAllToLive"',
        '"mcpServer/oauth/login"',
        '"mcpTool/call"',
        '"mcpPrompt/get"',
        '"mcpResource/read"',
        '"mcpResource/subscribe"',
        '"mcpResource/unsubscribe"',
        "expect(safeInvoke).not.toHaveBeenCalled()",
      ],
    ],
    [
      [
        "scripts/mcp/current-smoke.mjs",
        "scripts/mcp/lib/current-smoke-core.mjs",
        "scripts/mcp/lib/current-smoke-fixture.mjs",
        "scripts/mcp/lib/current-smoke-transport.mjs",
        "scripts/mcp/live-provider-smoke.mjs",
      ],
      [
        "--allow-live-provider",
        "LIME_MCP_LIVE_SERVER_URL",
        "runMcpLiveProviderSmoke",
        "shouldSummarizeUriField",
        "summarizeUriField",
        "parseLiveProviderUrl",
        "must not include username, password, query, or hash",
        "must reference an environment variable name, not an inline secret",
        '"mcpTool/listForContext"',
        '"mcpTool/search"',
        '"mcpTool/call"',
        '"mcpResource/read"',
        "resourceUriProvided",
        "resourceUriSummary",
        "uriMatchesExpected",
        "outputSchemaStructuredContentSeen",
        "structuredContentEcho",
        '"resources/templates/list"',
        "resourceTemplatesSeen",
        "resourceTemplateUriTemplate",
        "structuredContent: {",
        "outputSchema: {",
        "summary.legacyMcpCommandsSeen.length === 0",
      ],
    ],
    [
      "internal/aiprompts/commands.md",
      [
        "## MCP 控制面主链",
        "`src/lib/api/mcp.ts -> AppServerClient.request(...) -> app_server_handle_json_lines -> App Server JSON-RPC -> lime-rs/crates/mcp`",
        "`mcpServer/list`",
        "`mcpServerStatus/list`",
        "`mcpServer/create`",
        "`mcpServer/update`",
        "`mcpServer/delete`",
        "`mcpServer/enabled/set`",
        "`mcpServer/importFromApp`",
        "`mcpServer/syncAllToLive`",
        "`mcpServer/oauth/login`",
        "`mcpServer/oauthLogin/completed`",
        "`mcpServer/start`",
        "`mcpServer/stop`",
        "`mcpTool/list`",
        "`mcpTool/listForContext`",
        "`mcpTool/search`",
        "`mcpTool/call`",
        "`mcpTool/callWithCaller`",
        "`mcpPrompt/list`",
        "`mcpPrompt/get`",
        "`mcpResource/list`",
        "`mcpResource/read`",
        "`mcpResource/subscribe`",
        "`mcpResource/unsubscribe`",
        "`mcp:resources_updated`",
        "`mcp:resource_updated`",
        "浏览器模式不得静默退回 mock event fallback",
        "`smoke:mcp-current -- --allow-live-provider`",
        "`LIME_MCP_LIVE_SERVER_URL`",
        "不得包含 username、password、query 或 hash",
        "不允许 inline secret",
        "`network-invoke.json`",
        "旧 MCP Desktop facade 已统一归类为 `dead / retired guard-only`",
        "`get_mcp_servers`",
        "`mcp_list_servers_with_status`",
        "`mcp_list_tools`",
        "`mcp_list_prompts`",
        "`mcp_list_resources`",
        "`mcp_call_tool`",
        "`mcp_start_server`",
        "`sync_all_mcp_to_live`",
      ],
    ],
  ]);

  for (const [relativePath, snippets] of requiredByFile.entries()) {
    const paths = expandContractFiles(
      Array.isArray(relativePath) ? relativePath : [relativePath],
    );
    const content = paths
      .map((item) => fs.readFileSync(path.join(repoRoot, item), "utf8"))
      .join("\n");
    const requiredContent = requiredContractContent(repoRoot, paths, content);
    const location = paths.join(", ");
    for (const snippet of snippets) {
      if (!contractContentIncludes(requiredContent, snippet)) {
        failures.push(
          `MCP runtime current contract: missing ${JSON.stringify(
            snippet,
          )} in ${location}`,
        );
      }
    }
  }

  const rendererGateway = [
    "src/lib/api/mcp.ts",
    "src/lib/api/mcpResponseGuards.ts",
  ]
    .map((file) => fs.readFileSync(path.join(repoRoot, file), "utf8"))
    .join("\n");
  const forbiddenRendererSnippets = [
    "safeInvoke",
    '"get_mcp_servers"',
    '"mcp_list_servers_with_status"',
    '"mcp_list_tools"',
    '"mcp_call_tool"',
    '"mcp_call_tool_with_caller"',
    '"mcp_list_prompts"',
    '"mcp_get_prompt"',
    '"mcp_list_resources"',
    '"mcp_read_resource"',
    '"mcp_list_tools_for_context"',
    '"mcp_search_tools"',
    '"mcp_start_server"',
    '"mcp_stop_server"',
    '"add_mcp_server"',
    '"update_mcp_server"',
    '"delete_mcp_server"',
    '"toggle_mcp_server"',
    '"import_mcp_from_app"',
    '"sync_all_mcp_to_live"',
  ];
  for (const snippet of forbiddenRendererSnippets) {
    if (rendererGateway.includes(snippet)) {
      failures.push(
        `MCP runtime renderer gateway must not use legacy command path: forbidden ${JSON.stringify(
          snippet,
        )}`,
      );
    }
  }

  const eventBridgeContent = fs.readFileSync(
    path.join(repoRoot, "src/hooks/useMcpEvents.ts"),
    "utf8",
  );
  const forbiddenEventBridgeSnippets = [
    "defaultMocks",
    "mockPriorityCommands",
    "invokeMockOnly",
    "explicitMockFallback",
    "listenExplicitMock",
  ];
  for (const snippet of forbiddenEventBridgeSnippets) {
    if (eventBridgeContent.includes(snippet)) {
      failures.push(
        `MCP runtime events must stay on safeListen current bridge without mock fallback: forbidden ${JSON.stringify(
          snippet,
        )}`,
      );
    }
  }

  const legacyMcpEvents = [
    "mcp:oauth_completed",
    "mcp:server_started",
    "mcp:server_stopped",
    "mcp:server_error",
  ];
  const currentSourceFiles = ["electron", "lime-rs", "packages", "src"].flatMap(
    (root) => collectSourceFiles(path.join(repoRoot, root)),
  );
  for (const legacyEvent of legacyMcpEvents) {
    const legacyEventFiles = currentSourceFiles.filter((file) =>
      fs.readFileSync(file, "utf8").includes(legacyEvent),
    );
    for (const file of legacyEventFiles) {
      failures.push(
        `MCP lifecycle must use typed App Server notifications: forbidden ${JSON.stringify(
          legacyEvent,
        )} in ${path.relative(repoRoot, file)}`,
      );
    }
  }

  const appServerCurrentContent = [
    "lime-rs/crates/app-server/src/runtime.rs",
    "lime-rs/crates/app-server/src/local_data_source.rs",
    "lime-rs/crates/app-server/src/processor/mod.rs",
    "lime-rs/crates/app-server/src/processor/mcp.rs",
  ]
    .map((file) => fs.readFileSync(path.join(repoRoot, file), "utf8"))
    .join("\n");
  for (const snippet of ["mcp_cmd", "lime-rs/src/commands"]) {
    if (appServerCurrentContent.includes(snippet)) {
      failures.push(
        `MCP runtime App Server current path must not reference legacy command cleanup area: forbidden ${JSON.stringify(
          snippet,
        )}`,
      );
    }
  }
}

export function checkWorkspaceRightSurfaceCurrentContracts({
  repoRoot,
  failures,
}) {
  const requiredByFile = new Map([
    [
      [
        "lime-rs/crates/app-server-protocol/src/protocol/v0/right_surface.rs",
        "lime-rs/crates/app-server-protocol/src/protocol/v0/method_names.rs",
        "lime-rs/crates/app-server-protocol/src/protocol/v0/catalog.rs",
        "lime-rs/crates/app-server/src/runtime/right_surface.rs",
        "lime-rs/crates/app-server/src/processor/right_surface.rs",
        "lime-rs/crates/app-server-client/src/lib.rs",
        "packages/app-server-client/src/protocol.ts",
        "packages/app-server-client/src/request-client.ts",
        "packages/app-server-client/src/request-client-methods.ts",
        "packages/app-server-client/src/connection-methods.ts",
        "src/lib/api/workspaceRightSurface.ts",
        "src/lib/api/appServerClientMethods.ts",
        "src/lib/api/appServerClientMethodSpecs.ts",
        "src/components/agent/chat/workspace/useWorkspaceRightSurfacePendingRuntime.ts",
        "src/lib/governance/agentCommandCatalog.json",
      ],
      [
        '"workspaceRightSurface/request"',
        '"workspaceRightSurface/pending/list"',
        '"workspaceRightSurface/pending/consume"',
        '"workspaceRightSurface/pending/dismiss"',
        '"workspaceRightSurface/pendingChanged"',
        "WorkspaceRightSurfacePendingDismissParams",
        "WorkspaceRightSurfacePendingDismissResponse",
        "WorkspaceRightSurfacePendingChangedParams",
        "workspaceRightSurfacePendingChangedNotification(",
        "dismiss_workspace_right_surface_pending(",
        "dismissWorkspaceRightSurfacePending(",
        "dismissPendingRequestsForSurface",
        "dismissedRequestIds",
        "App Server workspaceRightSurface/pending/dismiss did not return dismissed request ids",
      ],
    ],
  ]);

  for (const [relativePath, snippets] of requiredByFile.entries()) {
    const paths = expandContractFiles(
      Array.isArray(relativePath) ? relativePath : [relativePath],
    );
    const content = paths
      .map((item) => fs.readFileSync(path.join(repoRoot, item), "utf8"))
      .join("\n");
    const location = paths.join(", ");
    for (const snippet of snippets) {
      if (!contractContentIncludes(content, snippet)) {
        failures.push(
          `Workspace Right Surface current contract: missing ${JSON.stringify(
            snippet,
          )} in ${location}`,
        );
      }
    }
  }
}
