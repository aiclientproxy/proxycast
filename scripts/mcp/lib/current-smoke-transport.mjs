import fs from "node:fs";

export const APP_SERVER_HANDLE_JSON_LINES_COMMAND =
  "app_server_handle_json_lines";

export const LEGACY_MCP_COMMANDS = [
  "get_mcp_servers",
  "mcp_list_servers_with_status",
  "mcp_list_tools",
  "mcp_list_tools_for_context",
  "mcp_search_tools",
  "mcp_call_tool",
  "mcp_call_tool_with_caller",
  "mcp_list_prompts",
  "mcp_get_prompt",
  "mcp_list_resources",
  "mcp_read_resource",
  "mcp_start_server",
  "mcp_stop_server",
  "add_mcp_server",
  "update_mcp_server",
  "delete_mcp_server",
  "toggle_mcp_server",
  "import_mcp_from_app",
  "sync_all_mcp_to_live",
];

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function parseJson(value) {
  if (!value) {
    return null;
  }
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function parseJsonRpcLine(line) {
  const trimmed = String(line || "").trim();
  if (!trimmed) {
    return null;
  }
  try {
    const parsed = JSON.parse(trimmed);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
}

function decodeJsonRpcLines(lines) {
  return Array.isArray(lines)
    ? lines.map(parseJsonRpcLine).filter(Boolean)
    : [];
}

function sanitizeText(value) {
  const sanitized = String(value ?? "")
    .replace(
      /((?:api[_-]?key|authorization|password|secret|session|token)[^=\s]*=)(["']?)[^\s"']+/gi,
      "$1$2[redacted]",
    )
    .replace(/(Bearer\s+)[A-Za-z0-9._~+/=-]+/gi, "$1[redacted]");
  return sanitized.length > 1_200
    ? `${sanitized.slice(0, 1_200)}... [truncated ${sanitized.length - 1_200} chars]`
    : sanitized;
}

function shouldSummarizeUriField(key) {
  const normalized = String(key || "").toLowerCase();
  return (
    normalized === "url" ||
    normalized.endsWith("url") ||
    normalized === "uri" ||
    normalized.endsWith("uri")
  );
}

function summarizeUriField(value) {
  const text = sanitizeText(value);
  try {
    const parsed = new URL(text);
    return {
      scheme: parsed.protocol.replace(/:$/, "") || null,
      host: parsed.host || null,
      hasPath: parsed.pathname !== "" && parsed.pathname !== "/",
      pathDepth: parsed.pathname.split("/").filter(Boolean).length,
      hasQuery: Boolean(parsed.search),
      hasHash: Boolean(parsed.hash),
    };
  } catch {
    return text;
  }
}

export function sanitizeJson(value, depth = 0, key = "") {
  if (depth > 5) {
    return "[truncated-depth]";
  }
  if (typeof value === "string") {
    return shouldSummarizeUriField(key)
      ? summarizeUriField(value)
      : sanitizeText(value);
  }
  if (
    value === null ||
    value === undefined ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value ?? null;
  }
  if (Array.isArray(value)) {
    return value.slice(0, 40).map((item) => sanitizeJson(item, depth + 1));
  }
  if (typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .slice(0, 80)
        .map(([entryKey, item]) => [
          entryKey,
          sanitizeJson(item, depth + 1, entryKey),
        ]),
    );
  }
  return sanitizeText(String(value));
}

export function writeJsonFile(filePath, value) {
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

export async function waitForHealth(options) {
  const startedAt = Date.now();
  let lastError = null;

  while (Date.now() - startedAt < options.timeoutMs) {
    try {
      const response = await fetch(options.healthUrl, {
        signal: AbortSignal.timeout(Math.min(5_000, options.timeoutMs)),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      console.log(
        `[smoke:mcp-current] DevBridge 已就绪 (${Date.now() - startedAt}ms) status=${payload?.status ?? response.status}`,
      );
      return payload;
    } catch (error) {
      lastError = error;
      await sleep(options.intervalMs);
    }
  }

  const detail =
    lastError instanceof Error
      ? lastError.message
      : String(lastError || "unknown error");
  throw new Error(
    `[smoke:mcp-current] DevBridge 未就绪，请先启动 npm run electron:dev。最后错误: ${detail}`,
  );
}

function collectInvokeEntry(requestPayload, responsePayload, url) {
  const requestLines =
    requestPayload?.request?.lines ??
    requestPayload?.args?.request?.lines ??
    requestPayload?.payload?.lines ??
    requestPayload?.lines;
  const responseLines =
    responsePayload?.result?.result?.lines ??
    responsePayload?.result?.lines ??
    responsePayload?.request?.lines ??
    responsePayload?.lines;
  const requestMessages = decodeJsonRpcLines(requestLines);
  const responseMessages = decodeJsonRpcLines(responseLines);
  const responseById = new Map(
    responseMessages
      .filter((message) => message && message.id !== undefined)
      .map((message) => [message.id, message]),
  );

  return {
    url: sanitizeText(url),
    cmd: requestPayload?.cmd ?? null,
    appServerRequests: requestMessages
      .filter((message) => typeof message?.method === "string")
      .map((message) => ({
        id: message.id ?? null,
        method: message.method,
        params: sanitizeJson(message.params ?? {}),
        response: sanitizeJson(responseById.get(message.id) ?? null),
      })),
    responseMessageCount: responseMessages.length,
    responseMessages: responseMessages.map(sanitizeJson),
  };
}

function parseAppServerResponseMessages(responsePayload) {
  const responseLines =
    responsePayload?.result?.result?.lines ??
    responsePayload?.result?.lines ??
    responsePayload?.lines;
  return decodeJsonRpcLines(responseLines);
}

export async function invokeBridgeCommand(options, cmd, args, entries) {
  const requestPayload = { cmd, args };
  const response = await fetch(options.invokeUrl, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(requestPayload),
    signal: AbortSignal.timeout(options.timeoutMs),
  });
  const text = await response.text();
  const responsePayload = parseJson(text);
  if (responsePayload) {
    entries.push(
      collectInvokeEntry(requestPayload, responsePayload, options.invokeUrl),
    );
  }
  if (!response.ok) {
    throw new Error(`${cmd} HTTP ${response.status}: ${sanitizeText(text)}`);
  }
  if (responsePayload?.error) {
    throw new Error(`${cmd} error: ${sanitizeText(responsePayload.error)}`);
  }
  if (!responsePayload) {
    throw new Error(`${cmd} returned non-JSON response`);
  }
  return responsePayload;
}

let appServerRequestId = 1;

export async function invokeAppServerMethod(options, method, params, entries) {
  const id = `mcp-current-${appServerRequestId++}`;
  const request =
    params === undefined ? { id, method } : { id, method, params };
  const responsePayload = await invokeBridgeCommand(
    options,
    APP_SERVER_HANDLE_JSON_LINES_COMMAND,
    { request: { lines: [`${JSON.stringify(request)}\n`] } },
    entries,
  );
  const messages = parseAppServerResponseMessages(responsePayload);
  const error = messages.find((message) => message.id === id && message.error);
  if (error) {
    throw new Error(
      `${method} error: ${error.error?.message || "App Server JSON-RPC error"}`,
    );
  }
  const response = messages.find(
    (message) => message.id === id && Object.hasOwn(message, "result"),
  );
  if (!response) {
    throw new Error(`${method} missing App Server response`);
  }
  return response.result;
}

export function summarizeInvokeEntries(
  entries,
  {
    requiredReadMethods,
    fixtureMethods,
    oauthFixtureMethods,
  },
) {
  const appServerRequests = entries.flatMap((entry) => entry.appServerRequests);
  const appServerMethodsSeen = Array.from(
    new Set(appServerRequests.map((request) => request.method)),
  ).sort();
  const legacyMcpCommandsSeen = Array.from(
    new Set(
      entries
        .map((entry) =>
          typeof entry.cmd === "string" &&
          LEGACY_MCP_COMMANDS.includes(entry.cmd)
            ? entry.cmd
            : null,
        )
        .filter(Boolean),
    ),
  ).sort();
  const appServerHandleJsonLinesSeen = entries.some(
    (entry) => entry.cmd === APP_SERVER_HANDLE_JSON_LINES_COMMAND,
  );
  const openExternalUrlSeen = entries.some(
    (entry) => entry.cmd === "open_external_url",
  );
  const responses = new Map();
  for (const request of appServerRequests) {
    responses.set(request.method, request.response);
  }

  return {
    appServerHandleJsonLinesSeen,
    openExternalUrlSeen,
    appServerMethodsSeen,
    legacyMcpCommandsSeen,
    missingReadMethods: requiredReadMethods.filter(
      (method) => !appServerMethodsSeen.includes(method),
    ),
    missingFixtureMethods: fixtureMethods.filter(
      (method) => !appServerMethodsSeen.includes(method),
    ),
    missingOAuthFixtureMethods: oauthFixtureMethods.filter(
      (method) => !appServerMethodsSeen.includes(method),
    ),
    mcpCounts: {
      servers: Array.isArray(responses.get("mcpServer/list")?.result?.servers)
        ? responses.get("mcpServer/list").result.servers.length
        : null,
      statusServers: Array.isArray(
        responses.get("mcpServerStatus/list")?.result?.servers,
      )
        ? responses.get("mcpServerStatus/list").result.servers.length
        : null,
      tools: Array.isArray(responses.get("mcpTool/list")?.result?.tools)
        ? responses.get("mcpTool/list").result.tools.length
        : null,
      prompts: Array.isArray(responses.get("mcpPrompt/list")?.result?.prompts)
        ? responses.get("mcpPrompt/list").result.prompts.length
        : null,
      resources: Array.isArray(
        responses.get("mcpResource/list")?.result?.resources,
      )
        ? responses.get("mcpResource/list").result.resources.length
        : null,
      resourceTemplates: Array.isArray(
        responses.get("mcpResource/list")?.result?.resourceTemplates,
      )
        ? responses.get("mcpResource/list").result.resourceTemplates.length
        : null,
    },
  };
}
