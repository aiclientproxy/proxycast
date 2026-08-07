// ============================================================================
// 基础类型定义
// ============================================================================

export interface McpServer {
  id: string;
  name: string;
  server_config: McpServerConfig;
  description?: string;
  enabled_lime: boolean;
  enabled_claude: boolean;
  enabled_codex: boolean;
  enabled_gemini: boolean;
  created_at?: number;
}

export type McpServerConfig =
  | {
      transport?: "stdio";
      type?: "stdio";
      command: string;
      args?: string[];
      env?: Record<string, string>;
      cwd?: string;
      timeout?: number;
      startup_timeout?: number;
      startupTimeout?: number;
      tool_timeout?: number;
      toolTimeout?: number;
      enabled?: boolean;
      enabled_tools?: string[];
      enabledTools?: string[];
      disabled_tools?: string[];
      disabledTools?: string[];
      required?: boolean;
      supports_parallel_tool_calls?: boolean;
      supportsParallelToolCalls?: boolean;
    }
  | {
      transport?: "streamable_http" | "streamable-http" | "http";
      type?: "streamable_http" | "streamable-http" | "http";
      url: string;
      bearer_token_env_var?: string;
      bearerTokenEnvVar?: string;
      http_headers?: Record<string, string>;
      httpHeaders?: Record<string, string>;
      env_http_headers?: Record<string, string>;
      envHttpHeaders?: Record<string, string>;
      timeout?: number;
      startup_timeout?: number;
      startupTimeout?: number;
      tool_timeout?: number;
      toolTimeout?: number;
      enabled?: boolean;
      enabled_tools?: string[];
      enabledTools?: string[];
      disabled_tools?: string[];
      disabledTools?: string[];
      required?: boolean;
      supports_parallel_tool_calls?: boolean;
      supportsParallelToolCalls?: boolean;
      scopes?: string[];
      oauth?: McpServerOAuthConfig;
      oauth_resource?: string;
      oauthResource?: string;
    };

export interface McpServerOAuthConfig {
  client_id?: string;
  clientId?: string;
}

/** MCP 服务器能力信息 */
export interface McpServerCapabilities {
  name: string;
  version: string;
  supports_tools: boolean;
  supports_prompts: boolean;
  supports_resources: boolean;
}

/** MCP 服务器信息（包含运行状态） */
export interface McpServerInfo {
  id: string;
  name: string;
  description?: string;
  config: McpServerConfig;
  is_running: boolean;
  server_info?: McpServerCapabilities;
  runtime_status?: McpServerRuntimeStatus;
  enabled_lime: boolean;
  enabled_claude: boolean;
  enabled_codex: boolean;
  enabled_gemini: boolean;
}

export interface McpServerRuntimeStatus {
  name: string;
  transport: "stdio" | "streamable_http" | string;
  enabled: boolean;
  is_running: boolean;
  required: boolean;
  supports_parallel_tool_calls: boolean;
  startup_timeout: number;
  tool_timeout: number;
  enabled_tools?: string[];
  disabled_tools: string[];
  server_info?: McpServerCapabilities;
  auth_status: McpServerAuthStatus;
}

export interface McpServerAuthStatus {
  mode: "none" | "static_headers" | "oauth" | string;
  available: boolean;
  reason_code?:
    | "oauth_login_required"
    | "oauth_runtime_not_implemented"
    | string;
  action_plan?: McpServerAuthActionPlan;
}

export interface McpServerAuthActionPlan {
  kind: "oauth_login" | "oauth_elicitation" | string;
  state: "login_required" | "runtime_not_connected" | string;
  required_runtime?:
    | "mcp_server_oauth_login"
    | "mcp_elicitation_approval"
    | string;
  scopes?: string[];
  oauth_resource?: string;
  client_id?: string;
}

export interface McpServerOAuthLoginOptions {
  scopes?: string[];
  timeoutSecs?: number;
}

export interface McpServerOAuthLoginResponse {
  authorizationUrl: string;
  state: string;
}

// ============================================================================
// 工具类型
// ============================================================================

/** MCP 工具定义。`name` 当前格式固定为 `mcp__<server>__<tool>`。 */
export interface McpToolDefinition {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
  output_schema?: Record<string, unknown>;
  server_name: string;
  deferred_loading?: boolean;
  always_visible?: boolean;
  allowed_callers?: string[];
  input_examples?: unknown[];
  tags?: string[];
}

/** MCP 内容类型 */
export type McpContent =
  | { type: "text"; text: string }
  | { type: "image"; data: string; mime_type: string }
  | { type: "resource"; uri: string; text?: string; blob?: string };

/** MCP 工具调用结果 */
export interface McpToolResult {
  content: McpContent[];
  structuredContent?: unknown;
  is_error: boolean;
}

export type McpPrepareRequestMethod =
  | "mcpServer/importFromApp"
  | "mcpServer/start"
  | "mcpTool/listForContext";

export interface McpPrepareRequest {
  method: McpPrepareRequestMethod | string;
  params?: Record<string, unknown>;
  reason?: string;
  status?: "candidate" | string;
}

export interface McpPrepareResult {
  method: McpPrepareRequestMethod;
  status: "completed";
  importedCount?: number;
  toolCount?: number;
  tools?: McpToolDefinition[];
}

export type McpCallProofRequestMethod = "mcpServer/tool/call";

export interface McpCallProofRequest {
  method: McpCallProofRequestMethod | string;
  params?: Record<string, unknown>;
  reason?: string;
  status?: "candidate" | string;
}

export interface McpCallProofResult {
  method: McpCallProofRequestMethod;
  status: "completed";
  result: McpToolResult;
}

/** 从 runtime 名 `mcp__<server>__<tool>` 提取 inner tool 名。 */
export function getMcpInnerToolName(
  toolName: string,
  serverName?: string,
): string {
  if (!toolName) return toolName;

  if (serverName) {
    const prefixedName = `mcp__${serverName}__`;
    if (toolName.startsWith(prefixedName)) {
      return toolName.slice(prefixedName.length);
    }
  }

  if (!toolName.startsWith("mcp__")) {
    return toolName;
  }

  const parts = toolName.split("__");
  return parts.length >= 3 ? parts.slice(2).join("__") : toolName;
}

// ============================================================================
// 提示词类型
// ============================================================================

/** MCP 提示词参数 */
export interface McpPromptArgument {
  name: string;
  description?: string;
  required: boolean;
}

/** MCP 提示词定义 */
export interface McpPromptDefinition {
  name: string;
  description?: string;
  arguments: McpPromptArgument[];
  server_name: string;
}

/** MCP 提示词消息 */
export interface McpPromptMessage {
  role: string;
  content: McpContent;
}

/** MCP 提示词结果 */
export interface McpPromptResult {
  description?: string;
  messages: McpPromptMessage[];
}

// ============================================================================
// 资源类型
// ============================================================================

/** MCP 资源定义 */
export interface McpResourceDefinition {
  uri: string;
  name: string;
  description?: string;
  mime_type?: string;
  server_name: string;
}

/** MCP 资源模板定义 */
export interface McpResourceTemplateDefinition {
  uri_template: string;
  name: string;
  title?: string;
  description?: string;
  mime_type?: string;
  server_name: string;
}

export interface McpResourceListResult {
  resources: McpResourceDefinition[];
  resourceTemplates: McpResourceTemplateDefinition[];
}

/** MCP 资源内容 */
export interface McpResourceContent {
  uri: string;
  mime_type?: string;
  text?: string;
  blob?: string;
  meta?: Record<string, unknown>;
}
