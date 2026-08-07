const LIVE_SERVER_URL_ENV = "LIME_MCP_LIVE_SERVER_URL";
const LIVE_SERVER_NAME_ENV = "LIME_MCP_LIVE_SERVER_NAME";
const LIVE_BEARER_TOKEN_ENV_VAR_ENV = "LIME_MCP_LIVE_BEARER_TOKEN_ENV_VAR";
const LIVE_ENV_HTTP_HEADERS_JSON_ENV = "LIME_MCP_LIVE_ENV_HTTP_HEADERS_JSON";
const LIVE_SCOPES_ENV = "LIME_MCP_LIVE_SCOPES";
const LIVE_TOOL_NAME_ENV = "LIME_MCP_LIVE_TOOL_NAME";
const LIVE_TOOL_ARGS_JSON_ENV = "LIME_MCP_LIVE_TOOL_ARGS_JSON";
const LIVE_RESOURCE_URI_ENV = "LIME_MCP_LIVE_RESOURCE_URI";
const ENV_VAR_NAME_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/;

export const LIVE_PROVIDER_METHODS = [
  "mcpServer/create",
  "mcpServer/start",
  "mcpServerStatus/list",
  "mcpTool/list",
  "mcpTool/listForContext",
  "mcpTool/search",
  "mcpResource/list",
  "mcpServer/stop",
  "mcpServer/delete",
];

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function readTrimmedEnv(env, name) {
  const value = env[name];
  return typeof value === "string" ? value.trim() : "";
}

function parseJsonEnv(env, name) {
  const raw = readTrimmedEnv(env, name);
  if (!raw) {
    return undefined;
  }
  try {
    return JSON.parse(raw);
  } catch (error) {
    throw new Error(
      `${name} must be valid JSON: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
}

function assertEnvVarName(value, source) {
  if (!ENV_VAR_NAME_PATTERN.test(value)) {
    throw new Error(
      `${source} must reference an environment variable name, not an inline secret`,
    );
  }
}

function parseStringRecordEnv(env, name) {
  const parsed = parseJsonEnv(env, name);
  if (parsed === undefined) {
    return undefined;
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`${name} must be a JSON object`);
  }
  const entries = Object.entries(parsed);
  for (const [key, value] of entries) {
    if (!key.trim() || typeof value !== "string" || !value.trim()) {
      throw new Error(
        `${name} must map non-empty header names to env var names`,
      );
    }
    assertEnvVarName(value.trim(), `${name}.${key}`);
  }
  return Object.fromEntries(entries.map(([key, value]) => [key, value.trim()]));
}

function parseScopes(env) {
  const raw = readTrimmedEnv(env, LIVE_SCOPES_ENV);
  return raw
    ? raw
        .split(",")
        .map((scope) => scope.trim())
        .filter(Boolean)
    : undefined;
}

function parseToolArgs(env) {
  const parsed = parseJsonEnv(env, LIVE_TOOL_ARGS_JSON_ENV);
  if (parsed === undefined) {
    return {};
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`${LIVE_TOOL_ARGS_JSON_ENV} must be a JSON object`);
  }
  return parsed;
}

function sanitizeServerName(value) {
  return value.replace(/[^a-zA-Z0-9_-]/g, "-").replace(/^-+|-+$/g, "");
}

function runtimeToolName(serverName, toolName) {
  if (toolName.startsWith("mcp__")) {
    return toolName;
  }
  return `mcp__${serverName}__${toolName}`;
}

function summarizeUri(uri) {
  if (!uri) {
    return null;
  }
  try {
    const parsed = new URL(uri);
    return {
      scheme: parsed.protocol.replace(/:$/, "") || null,
      host: parsed.host || null,
      hasPath: parsed.pathname !== "" && parsed.pathname !== "/",
      pathDepth: parsed.pathname.split("/").filter(Boolean).length,
      hasQuery: Boolean(parsed.search),
      hasHash: Boolean(parsed.hash),
    };
  } catch {
    return {
      scheme: null,
      host: null,
      hasPath: false,
      pathDepth: 0,
      hasQuery: false,
      hasHash: false,
      opaqueLength: uri.length,
    };
  }
}

function contentSummary(result) {
  const content = Array.isArray(result?.content) ? result.content : [];
  const structuredContent =
    result?.structuredContent ?? result?.structured_content;
  return {
    contentCount: content.length,
    contentTypes: Array.from(
      new Set(content.map((item) => item?.type).filter(Boolean)),
    ).sort(),
    structuredContentKeys:
      structuredContent && typeof structuredContent === "object"
        ? Object.keys(structuredContent).sort()
        : [],
    isError: result?.is_error ?? result?.isError ?? null,
  };
}

function parseLiveProviderUrl(rawUrl) {
  let parsedUrl = null;
  try {
    parsedUrl = new URL(rawUrl);
  } catch (error) {
    throw new Error(
      `${LIVE_SERVER_URL_ENV} must be a valid URL: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
  if (!["http:", "https:"].includes(parsedUrl.protocol)) {
    throw new Error(`${LIVE_SERVER_URL_ENV} must use http or https`);
  }
  if (
    parsedUrl.username ||
    parsedUrl.password ||
    parsedUrl.search ||
    parsedUrl.hash
  ) {
    throw new Error(
      `${LIVE_SERVER_URL_ENV} must not include username, password, query, or hash; pass secrets through ${LIVE_BEARER_TOKEN_ENV_VAR_ENV} or ${LIVE_ENV_HTTP_HEADERS_JSON_ENV}`,
    );
  }
  return parsedUrl;
}

export function readLiveProviderConfig(env = process.env) {
  const url = readTrimmedEnv(env, LIVE_SERVER_URL_ENV);
  if (!url) {
    throw new Error(
      `--allow-live-provider requires ${LIVE_SERVER_URL_ENV} to point at a real streamable HTTP MCP server`,
    );
  }

  const parsedUrl = parseLiveProviderUrl(url);
  const serverName =
    sanitizeServerName(readTrimmedEnv(env, LIVE_SERVER_NAME_ENV)) ||
    `live-${sanitizeServerName(parsedUrl.hostname || "provider")}`;
  const bearerTokenEnvVar = readTrimmedEnv(env, LIVE_BEARER_TOKEN_ENV_VAR_ENV);
  if (bearerTokenEnvVar) {
    assertEnvVarName(bearerTokenEnvVar, LIVE_BEARER_TOKEN_ENV_VAR_ENV);
  }
  const envHttpHeaders = parseStringRecordEnv(
    env,
    LIVE_ENV_HTTP_HEADERS_JSON_ENV,
  );
  const scopes = parseScopes(env);
  const toolName = readTrimmedEnv(env, LIVE_TOOL_NAME_ENV);
  const resourceUri = readTrimmedEnv(env, LIVE_RESOURCE_URI_ENV);

  return {
    serverName,
    serverConfig: {
      transport: "streamable_http",
      url: parsedUrl.toString(),
      timeout: 10,
      ...(bearerTokenEnvVar ? { bearer_token_env_var: bearerTokenEnvVar } : {}),
      ...(envHttpHeaders ? { env_http_headers: envHttpHeaders } : {}),
      ...(scopes && scopes.length > 0 ? { scopes } : {}),
    },
    expected: {
      toolName,
      toolArguments: toolName ? parseToolArgs(env) : {},
      resourceUri,
    },
    evidence: {
      urlHost: parsedUrl.host,
      bearerTokenEnvVar: bearerTokenEnvVar || null,
      envHttpHeaderNames: envHttpHeaders
        ? Object.keys(envHttpHeaders).sort()
        : [],
      scopes: scopes ?? [],
      toolName: toolName || null,
      resourceUriProvided: Boolean(resourceUri),
      resourceUriSummary: summarizeUri(resourceUri),
    },
  };
}

async function cleanupLiveServer({
  options,
  entries,
  invokeAppServerMethod,
  serverId,
  serverName,
}) {
  await invokeAppServerMethod(
    options,
    "mcpServer/stop",
    { name: serverName },
    entries,
  ).catch((error) => {
    console.warn(
      `[smoke:mcp-current] live provider stop failed: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  });
  await invokeAppServerMethod(
    options,
    "mcpServer/delete",
    { id: serverId },
    entries,
  ).catch((error) => {
    console.warn(
      `[smoke:mcp-current] live provider delete failed: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  });
}

export function describeMcpLiveProviderEnv() {
  return [
    LIVE_SERVER_URL_ENV,
    LIVE_SERVER_NAME_ENV,
    LIVE_BEARER_TOKEN_ENV_VAR_ENV,
    LIVE_ENV_HTTP_HEADERS_JSON_ENV,
    LIVE_SCOPES_ENV,
    LIVE_TOOL_NAME_ENV,
    LIVE_TOOL_ARGS_JSON_ENV,
    LIVE_RESOURCE_URI_ENV,
  ];
}

export async function runMcpLiveProviderSmoke({
  options,
  entries,
  invokeAppServerMethod,
}) {
  const live = readLiveProviderConfig();
  const serverId = `mcp-live-current-${Date.now()}`;
  const serverName = sanitizeServerName(`${live.serverName}-${Date.now()}`);

  try {
    const createResult = await invokeAppServerMethod(
      options,
      "mcpServer/create",
      {
        server: {
          id: serverId,
          name: serverName,
          description: "Current MCP live provider smoke",
          server_config: live.serverConfig,
          enabled_lime: true,
          enabled_claude: false,
          enabled_codex: false,
          enabled_gemini: false,
          created_at: Date.now(),
        },
      },
      entries,
    );
    assert(
      Array.isArray(createResult?.servers),
      "mcpServer/create did not return servers",
    );

    await invokeAppServerMethod(
      options,
      "mcpServer/start",
      { name: serverName },
      entries,
    );

    const statusResult = await invokeAppServerMethod(
      options,
      "mcpServerStatus/list",
      {},
      entries,
    );
    const statusServer = Array.isArray(statusResult?.servers)
      ? statusResult.servers.find((server) => server?.name === serverName)
      : null;
    assert(statusServer, "mcpServerStatus/list did not return live provider");
    assert(
      statusServer.is_running === true,
      "live provider did not reach running status",
    );

    const toolsResult = await invokeAppServerMethod(
      options,
      "mcpTool/list",
      {},
      entries,
    );
    const tools = Array.isArray(toolsResult?.tools) ? toolsResult.tools : [];
    const liveTools = tools.filter((tool) =>
      String(tool?.name || "").startsWith(`mcp__${serverName}__`),
    );
    const toolsForContextResult = await invokeAppServerMethod(
      options,
      "mcpTool/listForContext",
      { caller: "assistant", includeDeferred: true },
      entries,
    );
    const toolsForContext = Array.isArray(toolsForContextResult?.tools)
      ? toolsForContextResult.tools
      : [];
    const liveToolsForContext = toolsForContext.filter((tool) =>
      String(tool?.name || "").startsWith(`mcp__${serverName}__`),
    );
    const searchQuery = live.expected.toolName || serverName;
    const toolSearchResult = await invokeAppServerMethod(
      options,
      "mcpTool/search",
      { query: searchQuery, caller: "tool_search", limit: 20 },
      entries,
    );
    const searchedTools = Array.isArray(toolSearchResult?.tools)
      ? toolSearchResult.tools
      : [];
    const liveSearchedTools = searchedTools.filter((tool) =>
      String(tool?.name || "").startsWith(`mcp__${serverName}__`),
    );

    let calledTool = null;
    let targetToolName = null;
    if (live.expected.toolName) {
      targetToolName = runtimeToolName(serverName, live.expected.toolName);
      assert(
        liveTools.some((tool) => tool?.name === targetToolName),
        `mcpTool/list did not return requested live tool ${targetToolName}`,
      );
      assert(
        liveToolsForContext.some((tool) => tool?.name === targetToolName),
        `mcpTool/listForContext did not return requested live tool ${targetToolName}`,
      );
      assert(
        liveSearchedTools.some((tool) => tool?.name === targetToolName),
        `mcpTool/search did not return requested live tool ${targetToolName}`,
      );
      const thread = await invokeAppServerMethod(
        options,
        "thread/start",
        { ephemeral: true },
        entries,
      );
      const threadId = thread?.thread?.id;
      assert(
        typeof threadId === "string" && threadId.length > 0,
        "thread/start did not return a thread id for live MCP call",
      );
      const toolResult = await invokeAppServerMethod(
        options,
        "mcpServer/tool/call",
        {
          threadId,
          server: serverName,
          tool: targetToolName.slice(`mcp__${serverName}__`.length),
          arguments: live.expected.toolArguments,
        },
        entries,
      );
      const toolSummary = contentSummary(toolResult);
      assert(
        toolSummary.isError !== true,
        `live provider tool call returned isError=true for ${targetToolName}`,
      );
      calledTool = {
        toolName: targetToolName,
        ...toolSummary,
      };
    }

    const resourceResult = await invokeAppServerMethod(
      options,
      "mcpResource/list",
      {},
      entries,
    );
    const resources = Array.isArray(resourceResult?.resources)
      ? resourceResult.resources
      : [];
    const resourceTemplates = Array.isArray(resourceResult?.resourceTemplates)
      ? resourceResult.resourceTemplates
      : [];

    let readResource = null;
    if (live.expected.resourceUri) {
      assert(
        resources.some(
          (resource) =>
            resource?.uri === live.expected.resourceUri &&
            (resource?.server_name ?? resource?.serverName) === serverName,
        ),
        `mcpResource/list did not return requested live resource ${live.expected.resourceUri}`,
      );
      const readResult = await invokeAppServerMethod(
        options,
        "mcpServer/resource/read",
        { server: serverName, uri: live.expected.resourceUri },
        entries,
      );
      const content = Array.isArray(readResult?.contents)
        ? readResult.contents[0]
        : null;
      readResource = {
        uriMatchesExpected: content?.uri === live.expected.resourceUri,
        uriSummary: summarizeUri(content?.uri),
        mimeType: content?.mime_type ?? content?.mimeType ?? null,
        hasText: typeof content?.text === "string",
        hasBlob: typeof content?.blob === "string",
      };
    }

    return {
      serverId,
      serverName,
      provider: live.evidence,
      status: {
        transport: statusServer.runtime_status?.transport ?? null,
        supportsTools:
          statusServer.server_info?.supports_tools ??
          statusServer.runtime_status?.server_info?.supports_tools ??
          null,
        supportsResources:
          statusServer.server_info?.supports_resources ??
          statusServer.runtime_status?.server_info?.supports_resources ??
          null,
      },
      toolCount: liveTools.length,
      toolsForContextCount: liveToolsForContext.length,
      toolSearch: {
        query: searchQuery,
        resultCount: liveSearchedTools.length,
        requestedToolSeen:
          targetToolName === null
            ? null
            : liveSearchedTools.some((tool) => tool?.name === targetToolName),
      },
      resourceCount: resources.length,
      resourceTemplateCount: resourceTemplates.length,
      calledTool,
      readResource,
    };
  } finally {
    await cleanupLiveServer({
      options,
      entries,
      invokeAppServerMethod,
      serverId,
      serverName,
    });
  }
}
