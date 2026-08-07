import { AppServerClient } from "@/lib/api/appServer";
import {
  METHOD_MCP_PROMPT_GET,
  METHOD_MCP_PROMPT_LIST,
  METHOD_MCP_RESOURCE_LIST,
  METHOD_MCP_RESOURCE_SUBSCRIBE,
  METHOD_MCP_RESOURCE_UNSUBSCRIBE,
  METHOD_MCP_SERVER_CREATE,
  METHOD_MCP_SERVER_DELETE,
  METHOD_MCP_SERVER_ENABLED_SET,
  METHOD_MCP_SERVER_IMPORT_FROM_APP,
  METHOD_MCP_SERVER_LIST,
  METHOD_MCP_SERVER_OAUTH_LOGIN,
  METHOD_MCP_SERVER_SYNC_ALL_TO_LIVE,
  METHOD_MCP_SERVER_START,
  METHOD_MCP_SERVER_STATUS_LIST,
  METHOD_MCP_SERVER_STOP,
  METHOD_MCP_SERVER_UPDATE,
  METHOD_MCP_SERVER_RESOURCE_READ,
  METHOD_MCP_SERVER_TOOL_CALL,
  METHOD_MCP_TOOL_LIST,
  METHOD_MCP_TOOL_LIST_FOR_CONTEXT,
  METHOD_MCP_TOOL_SEARCH,
  type McpPromptGetResponse as AppServerMcpPromptGetResponse,
  type McpPromptListResponse as AppServerMcpPromptListResponse,
  type McpResourceListResponse as AppServerMcpResourceListResponse,
  type McpServerResourceReadResponse as AppServerMcpServerResourceReadResponse,
  type McpResourceSubscriptionResponse as AppServerMcpResourceSubscriptionResponse,
  type McpServerImportFromAppResponse as AppServerMcpServerImportFromAppResponse,
  type McpServerLifecycleResponse as AppServerMcpServerLifecycleResponse,
  type McpServerListResponse as AppServerMcpServerListResponse,
  type McpServerOauthLoginResponse as AppServerMcpServerOauthLoginResponse,
  type McpServerStatusListResponse as AppServerMcpServerStatusListResponse,
  type McpServerToolCallResponse as AppServerMcpServerToolCallResponse,
  type McpToolListResponse as AppServerMcpToolListResponse,
} from "../../../packages/app-server-client/src/protocol";
import type {
  McpPromptDefinition,
  McpPromptResult,
  McpCallProofRequest,
  McpCallProofResult,
  McpPrepareRequest,
  McpPrepareResult,
  McpResourceContent,
  McpResourceDefinition,
  McpResourceListResult,
  McpServer,
  McpServerInfo,
  McpServerOAuthLoginOptions,
  McpServerOAuthLoginResponse,
  McpToolDefinition,
  McpToolResult,
} from "./mcpTypes";
import {
  assertArrayField,
  assertEmptyResponse,
  assertLifecycleResponse,
  assertMcpPromptResult,
  assertMcpServerResourceContent,
  assertMcpServerToolResult,
  assertMcpResourceListResponse,
  assertOAuthLoginResponse,
  assertServerListResponse,
} from "./mcpResponseGuards";
export * from "./mcpTypes";

type McpAppServerClient = Pick<AppServerClient, "request">;

async function requestMcpAppServer<T>(
  method: string,
  params: unknown = {},
  appServerClient: McpAppServerClient = new AppServerClient(),
): Promise<T> {
  const response = await appServerClient.request<T>(method, params);
  return response.result;
}

function requireMcpResourceTarget(server: string, uri: string) {
  const normalizedServer = server.trim();
  const normalizedUri = uri.trim();
  if (!normalizedServer) {
    throw new Error("MCP resource server cannot be empty");
  }
  if (!normalizedUri) {
    throw new Error("MCP resource URI cannot be empty");
  }
  return { server: normalizedServer, uri: normalizedUri };
}

function requireMcpPromptTarget(server: string, name: string) {
  const normalizedServer = server.trim();
  const normalizedName = name.trim();
  if (!normalizedServer) {
    throw new Error("MCP prompt server cannot be empty");
  }
  if (!normalizedName) {
    throw new Error("MCP prompt name cannot be empty");
  }
  return { server: normalizedServer, name: normalizedName };
}

// ============================================================================
// API 封装
// ============================================================================

export const mcpApi = {
  // --------------------------------------------------------------------------
  // 配置管理 API
  // --------------------------------------------------------------------------

  getServers: (): Promise<McpServer[]> =>
    requestMcpAppServer<AppServerMcpServerListResponse>(
      METHOD_MCP_SERVER_LIST,
    ).then((response) =>
      assertArrayField<McpServer>(METHOD_MCP_SERVER_LIST, response, "servers"),
    ),

  addServer: (server: McpServer): Promise<void> =>
    requestMcpAppServer<AppServerMcpServerListResponse>(
      METHOD_MCP_SERVER_CREATE,
      { server },
    ).then((response) => {
      assertServerListResponse(METHOD_MCP_SERVER_CREATE, response);
      return undefined;
    }),

  updateServer: (server: McpServer): Promise<void> =>
    requestMcpAppServer<AppServerMcpServerListResponse>(
      METHOD_MCP_SERVER_UPDATE,
      { server },
    ).then((response) => {
      assertServerListResponse(METHOD_MCP_SERVER_UPDATE, response);
      return undefined;
    }),

  deleteServer: (id: string): Promise<void> =>
    requestMcpAppServer<AppServerMcpServerListResponse>(
      METHOD_MCP_SERVER_DELETE,
      { id },
    ).then((response) => {
      assertServerListResponse(METHOD_MCP_SERVER_DELETE, response);
      return undefined;
    }),

  toggleServer: (
    id: string,
    appType: string,
    enabled: boolean,
  ): Promise<void> =>
    requestMcpAppServer<AppServerMcpServerListResponse>(
      METHOD_MCP_SERVER_ENABLED_SET,
      {
        id,
        appType,
        enabled,
      },
    ).then((response) => {
      assertServerListResponse(METHOD_MCP_SERVER_ENABLED_SET, response);
      return undefined;
    }),

  /** 从外部应用导入 MCP 配置 */
  importFromApp: (appType: string): Promise<number> =>
    requestMcpAppServer<AppServerMcpServerImportFromAppResponse>(
      METHOD_MCP_SERVER_IMPORT_FROM_APP,
      { appType },
    ).then((response) => {
      if (typeof response.importedCount !== "number") {
        throw new Error(
          `${METHOD_MCP_SERVER_IMPORT_FROM_APP} did not return importedCount`,
        );
      }
      assertServerListResponse(METHOD_MCP_SERVER_IMPORT_FROM_APP, response);
      return response.importedCount;
    }),

  /** 同步所有 MCP 配置到实际配置文件 */
  syncAllToLive: (): Promise<void> =>
    requestMcpAppServer<AppServerMcpServerListResponse>(
      METHOD_MCP_SERVER_SYNC_ALL_TO_LIVE,
    ).then((response) => {
      assertServerListResponse(METHOD_MCP_SERVER_SYNC_ALL_TO_LIVE, response);
      return undefined;
    }),

  // --------------------------------------------------------------------------
  // 生命周期管理 API
  // --------------------------------------------------------------------------

  /** 获取所有服务器及其运行状态 */
  listServersWithStatus: (): Promise<McpServerInfo[]> =>
    requestMcpAppServer<AppServerMcpServerStatusListResponse>(
      METHOD_MCP_SERVER_STATUS_LIST,
    ).then((response) =>
      assertArrayField<McpServerInfo>(
        METHOD_MCP_SERVER_STATUS_LIST,
        response,
        "servers",
      ),
    ),

  /** 启动 MCP 服务器 */
  startServer: (name: string): Promise<void> =>
    requestMcpAppServer<AppServerMcpServerLifecycleResponse>(
      METHOD_MCP_SERVER_START,
      { name },
    ).then((response) => {
      assertLifecycleResponse(METHOD_MCP_SERVER_START, response);
      return undefined;
    }),

  /** 停止 MCP 服务器 */
  stopServer: (name: string): Promise<void> =>
    requestMcpAppServer<AppServerMcpServerLifecycleResponse>(
      METHOD_MCP_SERVER_STOP,
      { name },
    ).then((response) => {
      assertLifecycleResponse(METHOD_MCP_SERVER_STOP, response);
      return undefined;
    }),

  /** 启动 streamable HTTP MCP OAuth 授权登录 */
  loginOAuthServer: (
    name: string,
    options: McpServerOAuthLoginOptions = {},
  ): Promise<McpServerOAuthLoginResponse> =>
    requestMcpAppServer<AppServerMcpServerOauthLoginResponse>(
      METHOD_MCP_SERVER_OAUTH_LOGIN,
      {
        name,
        ...(options.scopes ? { scopes: options.scopes } : {}),
        ...(options.timeoutSecs ? { timeoutSecs: options.timeoutSecs } : {}),
      },
    ).then((response) =>
      assertOAuthLoginResponse(METHOD_MCP_SERVER_OAUTH_LOGIN, response),
    ),

  // --------------------------------------------------------------------------
  // 工具管理 API
  // --------------------------------------------------------------------------

  /** 获取所有可用工具，返回名格式为 `mcp__<server>__<tool>`。 */
  listTools: (): Promise<McpToolDefinition[]> =>
    requestMcpAppServer<AppServerMcpToolListResponse>(
      METHOD_MCP_TOOL_LIST,
    ).then((response) =>
      assertArrayField<McpToolDefinition>(
        METHOD_MCP_TOOL_LIST,
        response,
        "tools",
      ),
    ),

  /** 按调用上下文获取可见工具（支持 deferred_loading） */
  listToolsForContext: (
    caller?: string,
    includeDeferred = false,
  ): Promise<McpToolDefinition[]> =>
    requestMcpAppServer<AppServerMcpToolListResponse>(
      METHOD_MCP_TOOL_LIST_FOR_CONTEXT,
      {
        caller,
        includeDeferred,
      },
    ).then((response) =>
      assertArrayField<McpToolDefinition>(
        METHOD_MCP_TOOL_LIST_FOR_CONTEXT,
        response,
        "tools",
      ),
    ),

  /** 工具搜索（Tool Search），返回名格式为 `mcp__<server>__<tool>`。 */
  searchTools: (
    query: string,
    caller?: string,
    limit = 10,
  ): Promise<McpToolDefinition[]> =>
    requestMcpAppServer<AppServerMcpToolListResponse>(METHOD_MCP_TOOL_SEARCH, {
      query,
      caller,
      limit,
    }).then((response) =>
      assertArrayField<McpToolDefinition>(
        METHOD_MCP_TOOL_SEARCH,
        response,
        "tools",
      ),
    ),

  /** Codex exact MCP tool call，必须由真实 Thread owner 发起。 */
  callServerTool: async (params: {
    threadId: string;
    server: string;
    tool: string;
    arguments?: Record<string, unknown>;
    meta?: unknown;
  }): Promise<McpToolResult> => {
    const threadId = params.threadId.trim();
    const server = params.server.trim();
    const tool = params.tool.trim();
    if (!threadId) {
      throw new Error("MCP tool call requires a threadId");
    }
    if (!server || !tool) {
      throw new Error("MCP tool call requires a server and tool");
    }
    const response =
      await requestMcpAppServer<AppServerMcpServerToolCallResponse>(
        METHOD_MCP_SERVER_TOOL_CALL,
        {
          threadId,
          server,
          tool,
          ...(params.arguments === undefined
            ? {}
            : { arguments: params.arguments }),
          ...(params.meta === undefined ? {} : { _meta: params.meta }),
        },
      );
    return assertMcpServerToolResult(METHOD_MCP_SERVER_TOOL_CALL, response);
  },

  // --------------------------------------------------------------------------
  // 提示词管理 API
  // --------------------------------------------------------------------------

  /** 获取所有可用提示词 */
  listPrompts: (): Promise<McpPromptDefinition[]> =>
    requestMcpAppServer<AppServerMcpPromptListResponse>(
      METHOD_MCP_PROMPT_LIST,
    ).then((response) =>
      assertArrayField<McpPromptDefinition>(
        METHOD_MCP_PROMPT_LIST,
        response,
        "prompts",
      ),
    ),

  /** 获取提示词内容 */
  getPrompt: async (
    server: string,
    name: string,
    args: Record<string, unknown>,
  ): Promise<McpPromptResult> => {
    const target = requireMcpPromptTarget(server, name);
    const response = await requestMcpAppServer<AppServerMcpPromptGetResponse>(
      METHOD_MCP_PROMPT_GET,
      {
        ...target,
        arguments: args,
      },
    );
    return assertMcpPromptResult(METHOD_MCP_PROMPT_GET, response);
  },

  // --------------------------------------------------------------------------
  // 资源管理 API
  // --------------------------------------------------------------------------

  /** 获取所有可用资源 */
  listResources: (): Promise<McpResourceDefinition[]> =>
    requestMcpAppServer<AppServerMcpResourceListResponse>(
      METHOD_MCP_RESOURCE_LIST,
    )
      .then((response) =>
        assertMcpResourceListResponse(METHOD_MCP_RESOURCE_LIST, response),
      )
      .then((response) => response.resources),

  /** 获取所有可用资源及资源模板 */
  listResourcesWithTemplates: (): Promise<McpResourceListResult> =>
    requestMcpAppServer<AppServerMcpResourceListResponse>(
      METHOD_MCP_RESOURCE_LIST,
    ).then((response) =>
      assertMcpResourceListResponse(METHOD_MCP_RESOURCE_LIST, response),
    ),

  /** 读取资源内容 */
  readResource: async (
    server: string,
    uri: string,
    runtimeOwner?: { sessionId?: string; threadId: string },
  ): Promise<McpResourceContent> => {
    const target = requireMcpResourceTarget(server, uri);
    const threadId = runtimeOwner?.threadId?.trim();
    if (runtimeOwner && !threadId) {
      throw new Error("MCP runtime threadId cannot be empty");
    }
    const response =
      await requestMcpAppServer<AppServerMcpServerResourceReadResponse>(
        METHOD_MCP_SERVER_RESOURCE_READ,
        {
          ...target,
          ...(threadId ? { threadId } : {}),
        },
      );
    return assertMcpServerResourceContent(
      METHOD_MCP_SERVER_RESOURCE_READ,
      response,
    );
  },

  /** 订阅资源更新 */
  subscribeResource: async (server: string, uri: string): Promise<void> => {
    const target = requireMcpResourceTarget(server, uri);
    const response =
      await requestMcpAppServer<AppServerMcpResourceSubscriptionResponse>(
        METHOD_MCP_RESOURCE_SUBSCRIBE,
        target,
      );
    assertEmptyResponse(METHOD_MCP_RESOURCE_SUBSCRIBE, response);
  },

  /** 取消订阅资源更新 */
  unsubscribeResource: async (server: string, uri: string): Promise<void> => {
    const target = requireMcpResourceTarget(server, uri);
    const response =
      await requestMcpAppServer<AppServerMcpResourceSubscriptionResponse>(
        METHOD_MCP_RESOURCE_UNSUBSCRIBE,
        target,
      );
    assertEmptyResponse(METHOD_MCP_RESOURCE_UNSUBSCRIBE, response);
  },

  executePrepareRequests: async (
    requests: McpPrepareRequest[],
  ): Promise<McpPrepareResult[]> => {
    const results: McpPrepareResult[] = [];
    for (const request of requests) {
      results.push(await executeMcpPrepareRequest(request));
    }
    return results;
  },

  executeCallProofRequests: async (
    requests: McpCallProofRequest[],
    threadId: string,
  ): Promise<McpCallProofResult[]> => {
    const results: McpCallProofResult[] = [];
    for (const request of requests) {
      results.push(await executeMcpCallProofRequest(request, threadId));
    }
    return results;
  },
};

function assertPrepareCandidate(request: McpPrepareRequest): void {
  if (request.status !== "candidate") {
    throw new Error("MCP prepare request must be candidate");
  }
}

function getPrepareParams(
  method: string,
  params: McpPrepareRequest["params"],
): Record<string, unknown> {
  if (!params || typeof params !== "object" || Array.isArray(params)) {
    throw new Error(`${method} did not provide prepare params`);
  }
  return params;
}

function readStringPrepareParam(
  method: string,
  params: Record<string, unknown>,
  field: string,
): string {
  const value = params[field];
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${method} prepare params require ${field}`);
  }
  return value;
}

function readOptionalStringPrepareParam(
  method: string,
  params: Record<string, unknown>,
  field: string,
): string | undefined {
  const value = params[field];
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "string") {
    throw new Error(`${method} prepare params require ${field} string`);
  }
  return value;
}

function readOptionalBooleanPrepareParam(
  method: string,
  params: Record<string, unknown>,
  field: string,
): boolean {
  const value = params[field];
  if (value === undefined) {
    return false;
  }
  if (typeof value !== "boolean") {
    throw new Error(`${method} prepare params require ${field} boolean`);
  }
  return value;
}

function readRecordPrepareParam(
  method: string,
  params: Record<string, unknown>,
  field: string,
): Record<string, unknown> {
  const value = params[field];
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${method} prepare params require ${field} object`);
  }
  return value as Record<string, unknown>;
}

async function executeMcpPrepareRequest(
  request: McpPrepareRequest,
): Promise<McpPrepareResult> {
  assertPrepareCandidate(request);
  if (request.method === METHOD_MCP_SERVER_IMPORT_FROM_APP) {
    const params = getPrepareParams(request.method, request.params);
    const appType = readStringPrepareParam(request.method, params, "appType");
    const importedCount = await mcpApi.importFromApp(appType);
    return {
      method: METHOD_MCP_SERVER_IMPORT_FROM_APP,
      status: "completed",
      importedCount,
    };
  }
  if (request.method === METHOD_MCP_SERVER_START) {
    const params = getPrepareParams(request.method, request.params);
    const name = readStringPrepareParam(request.method, params, "name");
    await mcpApi.startServer(name);
    return {
      method: METHOD_MCP_SERVER_START,
      status: "completed",
    };
  }
  if (request.method === METHOD_MCP_TOOL_LIST_FOR_CONTEXT) {
    const params = getPrepareParams(request.method, request.params);
    const caller = readOptionalStringPrepareParam(
      request.method,
      params,
      "caller",
    );
    const includeDeferred = readOptionalBooleanPrepareParam(
      request.method,
      params,
      "includeDeferred",
    );
    const tools = await mcpApi.listToolsForContext(caller, includeDeferred);
    return {
      method: METHOD_MCP_TOOL_LIST_FOR_CONTEXT,
      status: "completed",
      toolCount: tools.length,
      tools,
    };
  }
  throw new Error(
    `Unsupported MCP prepare request method: ${String(request.method)}`,
  );
}

function assertCallProofCandidate(request: McpCallProofRequest): void {
  if (request.status !== "candidate") {
    throw new Error("MCP call proof request must be candidate");
  }
}

async function executeMcpCallProofRequest(
  request: McpCallProofRequest,
  threadId: string,
): Promise<McpCallProofResult> {
  assertCallProofCandidate(request);
  if (request.method !== METHOD_MCP_SERVER_TOOL_CALL) {
    throw new Error(
      `Unsupported MCP call proof request method: ${String(request.method)}`,
    );
  }

  const params = getPrepareParams(request.method, request.params);
  const server = readStringPrepareParam(request.method, params, "server");
  const tool = readStringPrepareParam(request.method, params, "tool");
  const args = readRecordPrepareParam(request.method, params, "arguments");
  const result = await mcpApi.callServerTool({
    threadId,
    server,
    tool,
    arguments: args,
  });
  if (result.is_error) {
    throw new Error("MCP call proof returned tool error");
  }
  return {
    method: METHOD_MCP_SERVER_TOOL_CALL,
    status: "completed",
    result,
  };
}
