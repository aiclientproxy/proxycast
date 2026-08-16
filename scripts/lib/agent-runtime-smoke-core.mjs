const PROVIDER_PICK_ORDER = ["deepseek", "doubao", "lime-hub", "openai"];
const TERMINAL_THREAD_STATUSES = new Set([
  "completed",
  "failed",
  "aborted",
  "idle",
  "waiting_request",
]);
const RUNNING_THREAD_STATUSES = new Set(["running", "queued", "interrupting"]);
const APP_SERVER_HANDLE_JSON_LINES_COMMAND = "app_server_handle_json_lines";
const APP_SERVER_DRAIN_EVENTS_COMMAND = "app_server_drain_events";
const APP_SERVER_METHOD_MODEL_PROVIDER_LIST = "modelProvider/list";
const APP_SERVER_METHOD_MODEL_PROVIDER_READ = "modelProvider/read";
const APP_SERVER_METHOD_THREAD_START = "thread/start";
const APP_SERVER_METHOD_THREAD_SETTINGS_UPDATE = "thread/settings/update";
const APP_SERVER_METHOD_THREAD_READ = "thread/read";
const APP_SERVER_METHOD_TURN_START = "turn/start";
const APP_SERVER_METHOD_TURN_INTERRUPT = "turn/interrupt";
const APP_SERVER_REQUEST_METHODS = new Set([
  "item/commandExecution/requestApproval",
  "item/fileChange/requestApproval",
  "item/tool/requestUserInput",
  "mcpServer/elicitation/request",
]);

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function assertSmoke(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

export function normalizeProviderId(provider) {
  return String(
    provider?.id || provider?.provider_id || provider?.providerId || "",
  ).trim();
}

export function providerEnabled(provider) {
  return provider?.enabled !== false;
}

export function providerHasUsableKey(provider) {
  const keys = Array.isArray(provider?.api_keys)
    ? provider.api_keys
    : Array.isArray(provider?.apiKeys)
      ? provider.apiKeys
      : [];
  return keys.some((key) => key?.enabled !== false);
}

export function pickModelPreference(provider) {
  const candidates = [
    ...(Array.isArray(provider?.models) ? provider.models : []),
  ]
    .map((value) =>
      typeof value === "string"
        ? value
        : String(value?.name || value?.id || value?.model || "").trim(),
    )
    .filter(Boolean);

  return (
    candidates.find((value) => /flash|mini|lite|small/i.test(value)) ||
    candidates[0] ||
    ""
  );
}

export function pickProvider(providers, preferredProviderId = "") {
  const enabled = providers.filter((provider) => providerEnabled(provider));
  const keyed = enabled.filter((provider) => providerHasUsableKey(provider));
  const pool = keyed.length > 0 ? keyed : enabled;

  if (preferredProviderId) {
    return (
      pool.find(
        (provider) => normalizeProviderId(provider) === preferredProviderId,
      ) ||
      enabled.find(
        (provider) => normalizeProviderId(provider) === preferredProviderId,
      ) ||
      providers.find(
        (provider) => normalizeProviderId(provider) === preferredProviderId,
      ) ||
      null
    );
  }

  for (const providerId of PROVIDER_PICK_ORDER) {
    const match = pool.find(
      (provider) => normalizeProviderId(provider) === providerId,
    );
    if (match) {
      return match;
    }
  }

  return pool[0] || enabled[0] || null;
}

export function providerRuntimeName(provider) {
  return String(
    provider?.runtime_provider_name ||
      provider?.runtimeProviderName ||
      provider?.type ||
      provider?.provider_type ||
      provider?.providerType ||
      provider?.id ||
      "",
  ).trim();
}

export async function waitForHealth({
  healthUrl,
  timeoutMs,
  intervalMs,
  logPrefix,
}) {
  const startedAt = Date.now();
  let lastError = null;

  while (Date.now() - startedAt < timeoutMs) {
    try {
      const response = await fetch(healthUrl, {
        method: "GET",
        signal: AbortSignal.timeout(Math.min(intervalMs, 5_000)),
      });
      const text = await response.text();
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const payload = text ? JSON.parse(text) : {};
      console.log(
        `${logPrefix} DevBridge ready elapsedMs=${Date.now() - startedAt}${
          payload?.status ? ` status=${payload.status}` : ""
        }`,
      );
      return payload;
    } catch (error) {
      lastError = error;
      await sleep(intervalMs);
    }
  }

  const detail =
    lastError instanceof Error
      ? lastError.message
      : String(lastError || "unknown");
  throw new Error(`${logPrefix} DevBridge health timeout: ${detail}`);
}

export async function invokeDevBridge(
  options,
  cmd,
  args = {},
  timeoutMs = options.timeoutMs,
) {
  const response = await fetch(options.invokeUrl, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ cmd, args }),
    signal: AbortSignal.timeout(timeoutMs),
  });
  const text = await response.text();
  if (!response.ok) {
    throw new Error(`${cmd} HTTP ${response.status}: ${text}`);
  }
  const payload = text ? JSON.parse(text) : null;
  if (payload?.error) {
    throw new Error(`${cmd} error: ${payload.error}`);
  }
  return payload?.result;
}

let appServerRequestId = 1;

export async function invokeAppServerMethod(
  options,
  method,
  params,
  timeoutMs = options.timeoutMs,
) {
  const id = `agent-runtime-smoke-${appServerRequestId++}`;
  const request =
    params === undefined ? { id, method } : { id, method, params };
  const result = await invokeDevBridge(
    options,
    APP_SERVER_HANDLE_JSON_LINES_COMMAND,
    { request: { lines: [`${JSON.stringify(request)}\n`] } },
    timeoutMs,
  );
  const responseLines = result?.result?.lines ?? result?.lines;
  const messages = (Array.isArray(responseLines) ? responseLines : [])
    .map((line) => {
      try {
        return JSON.parse(String(line));
      } catch {
        return null;
      }
    })
    .filter(Boolean);
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

function decodeAppServerLines(result) {
  const lines = result?.result?.lines ?? result?.lines;
  return (Array.isArray(lines) ? lines : [])
    .map((line) => {
      try {
        return JSON.parse(String(line));
      } catch {
        return null;
      }
    })
    .filter(Boolean);
}

function normalizedScope(actionScope) {
  const scope =
    actionScope &&
    typeof actionScope === "object" &&
    !Array.isArray(actionScope)
      ? actionScope
      : {};
  return {
    threadId: String(scope.threadId ?? scope.thread_id ?? "").trim(),
    turnId: String(scope.turnId ?? scope.turn_id ?? "").trim(),
  };
}

export function findPendingAppServerRequest(messages, request) {
  const requestId = String(request?.requestId ?? "").trim();
  const scope = normalizedScope(request?.actionScope);
  return (
    [...messages].reverse().find((message) => {
      if (
        !APP_SERVER_REQUEST_METHODS.has(message?.method) ||
        (typeof message?.id !== "string" && typeof message?.id !== "number")
      ) {
        return false;
      }
      const params =
        message.params &&
        typeof message.params === "object" &&
        !Array.isArray(message.params)
          ? message.params
          : {};
      const semanticIds = [
        params.approvalId,
        params.approval_id,
        params.itemId,
        params.item_id,
        params.requestId,
        params.request_id,
      ]
        .map((value) => String(value ?? "").trim())
        .filter(Boolean);
      if (requestId && semanticIds.includes(requestId)) {
        return true;
      }
      if (semanticIds.length > 0) {
        return false;
      }
      const threadId = String(params.threadId ?? params.thread_id ?? "").trim();
      const turnId = String(params.turnId ?? params.turn_id ?? "").trim();
      return (
        Boolean(scope.threadId) &&
        threadId === scope.threadId &&
        (!scope.turnId || !turnId || turnId === scope.turnId)
      );
    }) ?? null
  );
}

export function buildAppServerRequestResponse(serverRequest, request) {
  const decision = request?.decision;
  switch (serverRequest.method) {
    case "item/commandExecution/requestApproval":
    case "item/fileChange/requestApproval":
      return {
        decision:
          decision === "allow_for_session"
            ? "acceptForSession"
            : decision === "cancel"
              ? "cancel"
              : decision === "decline" || request?.confirmed === false
                ? "decline"
                : "accept",
      };
    case "item/tool/requestUserInput":
      return {
        answers:
          request?.userData?.answers &&
          typeof request.userData.answers === "object"
            ? request.userData.answers
            : (request?.userData ?? {}),
      };
    case "mcpServer/elicitation/request":
      return {
        action:
          decision === "cancel"
            ? "cancel"
            : decision === "decline" || request?.confirmed === false
              ? "decline"
              : "accept",
        ...(request?.confirmed === false
          ? {}
          : { content: request?.userData ?? request?.response ?? {} }),
      };
    default:
      throw new Error(
        `unsupported App Server server request: ${serverRequest.method}`,
      );
  }
}

async function waitForPendingAppServerRequest(options, request) {
  const startedAt = Date.now();
  const timeoutMs = Math.min(options.timeoutMs, 30_000);
  while (Date.now() - startedAt < timeoutMs) {
    const result = await invokeDevBridge(
      options,
      APP_SERVER_DRAIN_EVENTS_COMMAND,
      { request: { includeRecent: true, limit: 500 } },
      timeoutMs,
    );
    const pending = findPendingAppServerRequest(
      decodeAppServerLines(result),
      request,
    );
    if (pending) {
      return pending;
    }
    await sleep(Math.min(options.intervalMs ?? 250, 250));
  }
  throw new Error(
    `typed App Server server request not found: ${request?.requestId ?? "unknown"}`,
  );
}

export async function createAgentSessionCurrent(
  options,
  { workspaceId, title, executionStrategy = "react", metadata = {} },
) {
  const metadataRecord =
    metadata && typeof metadata === "object" && !Array.isArray(metadata)
      ? metadata
      : {};
  const harnessMetadata =
    metadataRecord.harness &&
    typeof metadataRecord.harness === "object" &&
    !Array.isArray(metadataRecord.harness)
      ? metadataRecord.harness
      : {};
  const response = await invokeAppServerMethod(
    options,
    APP_SERVER_METHOD_THREAD_START,
    {
      appId: "desktop",
      workspaceId,
      businessObjectRef: {
        kind: "agent.session",
        id: `agent-session:${workspaceId}:${Date.now()}`,
        title,
        metadata: {
          ...metadataRecord,
          title,
          executionStrategy,
          runStartHooks: false,
          harness: {
            source: "smoke:agent-runtime",
            ...harnessMetadata,
          },
        },
      },
    },
    30_000,
  );
  const sessionId = String(response?.session?.sessionId || "").trim();
  if (!sessionId) {
    throw new Error("thread/start 未返回 sessionId");
  }
  return sessionId;
}

export async function updateAgentThreadSettingsCurrent(
  options,
  { threadId, provider },
  invoke = invokeAppServerMethod,
) {
  await invoke(
    options,
    APP_SERVER_METHOD_THREAD_SETTINGS_UPDATE,
    {
      threadId,
      modelProvider: provider.providerPreference,
      model: provider.modelPreference,
    },
    30_000,
  );
}

export async function readAgentSessionDetailCurrent(
  options,
  sessionId,
  { historyLimit = 20 } = {},
) {
  const response = await invokeAppServerMethod(
    options,
    APP_SERVER_METHOD_THREAD_READ,
    {
      sessionId,
      historyLimit,
    },
  );
  const detail = response?.detail;
  if (detail && typeof detail === "object" && !Array.isArray(detail)) {
    return detail;
  }
  return {
    id: response?.session?.sessionId || sessionId,
    thread_id: response?.session?.threadId || sessionId,
    workspace_id: response?.session?.workspaceId || null,
    turns: Array.isArray(response?.turns) ? response.turns : [],
    messages: [],
    items: [],
  };
}

function compactRecord(record) {
  return Object.fromEntries(
    Object.entries(record).filter(([, value]) => value !== undefined),
  );
}

export async function startAgentSessionTurnCurrent(
  options,
  {
    sessionId,
    workspaceId,
    message,
    eventName,
    turnId,
    runtimeRequest = {},
    queueIfBusy,
    queuedTurnId,
    skipPreSubmitResume = true,
  },
  invoke = invokeAppServerMethod,
) {
  const runtimeOptions = compactRecord({
    stream: true,
    eventName,
    queuedTurnId,
    runtimeRequest: compactRecord({
      workspaceId,
      ...runtimeRequest,
    }),
  });
  return invoke(
    options,
    APP_SERVER_METHOD_TURN_START,
    compactRecord({
      sessionId,
      turnId,
      input: {
        text: message,
      },
      runtimeOptions,
      queueIfBusy,
      skipPreSubmitResume,
    }),
  );
}

export async function cancelAgentSessionTurnCurrent(
  options,
  { sessionId, turnId },
) {
  return invokeAppServerMethod(options, APP_SERVER_METHOD_TURN_INTERRUPT, {
    sessionId,
    turnId,
  });
}

export async function respondAgentServerRequestCurrent(options, request) {
  const capturedRequest = request?.serverRequest;
  const serverRequest =
    capturedRequest &&
    APP_SERVER_REQUEST_METHODS.has(capturedRequest.method) &&
    (typeof capturedRequest.id === "string" ||
      typeof capturedRequest.id === "number")
      ? capturedRequest
      : await waitForPendingAppServerRequest(options, request);
  const result = buildAppServerRequestResponse(serverRequest, request);
  return invokeDevBridge(options, APP_SERVER_HANDLE_JSON_LINES_COMMAND, {
    request: {
      lines: [`${JSON.stringify({ id: serverRequest.id, result })}\n`],
    },
  });
}

export async function listPendingAppServerRequestsCurrent(
  options,
  { threadId, turnId },
) {
  const result = await invokeDevBridge(
    options,
    APP_SERVER_DRAIN_EVENTS_COMMAND,
    { request: { includeRecent: true, limit: 500 } },
    Math.min(options.timeoutMs, 30_000),
  );
  return decodeAppServerLines(result).filter((message) => {
    if (!APP_SERVER_REQUEST_METHODS.has(message?.method)) {
      return false;
    }
    const params = message?.params ?? {};
    const requestThreadId = String(
      params.threadId ?? params.thread_id ?? "",
    ).trim();
    const requestTurnId = String(params.turnId ?? params.turn_id ?? "").trim();
    return (
      (!threadId || requestThreadId === threadId) &&
      (!turnId || !requestTurnId || requestTurnId === turnId)
    );
  });
}

export async function readAgentRuntimeThreadCurrent(
  options,
  sessionId,
  { historyLimit } = {},
  invoke = invokeAppServerMethod,
) {
  const response = await invoke(
    options,
    APP_SERVER_METHOD_THREAD_READ,
    compactRecord({
      sessionId,
      historyLimit,
    }),
  );
  const threadRead =
    response?.detail?.thread_read || response?.detail?.threadRead;
  if (
    threadRead &&
    typeof threadRead === "object" &&
    !Array.isArray(threadRead)
  ) {
    return threadRead;
  }
  const turns = Array.isArray(response?.turns) ? response.turns : [];
  const latestTurn = turns[turns.length - 1] || null;
  return {
    thread_id: response?.session?.threadId || sessionId,
    status: response?.session?.status || latestTurn?.status || "idle",
    active_turn_id:
      response?.session?.activeTurnId ||
      response?.session?.active_turn_id ||
      latestTurn?.turnId ||
      null,
    turns,
    queued_turns: [],
    pending_requests: [],
  };
}

export async function resolveProviderPreference(
  options,
  invoke = invokeAppServerMethod,
) {
  const explicitProvider = String(options.providerPreference || "").trim();
  const explicitModel = String(options.modelPreference || "").trim();
  if (explicitProvider && explicitModel) {
    return {
      providerPreference: explicitProvider,
      providerName: explicitProvider,
      modelPreference: explicitModel,
      source: "explicit",
    };
  }

  const providerList = await invoke(
    options,
    APP_SERVER_METHOD_MODEL_PROVIDER_LIST,
    {},
    30_000,
  );
  const providers = providerList?.providers;
  const selected = pickProvider(
    Array.isArray(providers) ? providers : [],
    explicitProvider,
  );
  const providerId = normalizeProviderId(selected);
  if (!providerId) {
    throw new Error(
      `${options.logPrefix} no usable provider found; pass --provider-preference and --model-preference`,
    );
  }

  let providerDetail = selected;
  try {
    providerDetail =
      (
        await invoke(
          options,
          APP_SERVER_METHOD_MODEL_PROVIDER_READ,
          { providerId },
          30_000,
        )
      )?.provider || selected;
  } catch (error) {
    console.warn(
      `${options.logPrefix} provider detail failed, using list item: ${error.message}`,
    );
  }

  const modelPreference = explicitModel || pickModelPreference(providerDetail);
  if (!modelPreference) {
    throw new Error(
      `${options.logPrefix} provider ${providerId} has no configured model`,
    );
  }

  return {
    providerPreference: providerId,
    providerName: providerRuntimeName(providerDetail) || providerId,
    modelPreference,
    source:
      explicitProvider || explicitModel
        ? "partial-explicit"
        : "auto-enabled-provider",
  };
}

export function latestTurnStatus(threadRead) {
  return (
    threadRead?.diagnostics?.latest_turn_status ||
    threadRead?.diagnostics?.latestTurnStatus ||
    threadRead?.runtime_summary?.latestTurnStatus ||
    threadRead?.runtimeSummary?.latestTurnStatus ||
    threadRead?.status ||
    null
  );
}

export function summarizeThreadRead(threadRead) {
  const pendingRequests = Array.isArray(threadRead?.pending_requests)
    ? threadRead.pending_requests
    : Array.isArray(threadRead?.pendingRequests)
      ? threadRead.pendingRequests
      : [];
  const queuedTurns = Array.isArray(threadRead?.queued_turns)
    ? threadRead.queued_turns
    : Array.isArray(threadRead?.queuedTurns)
      ? threadRead.queuedTurns
      : [];
  return {
    threadStatus: threadRead?.status || null,
    latestTurnStatus: latestTurnStatus(threadRead),
    activeTurnId:
      threadRead?.active_turn_id || threadRead?.activeTurnId || null,
    turnCount: Array.isArray(threadRead?.turns) ? threadRead.turns.length : 0,
    queuedTurnCount: queuedTurns.length,
    pendingRequestCount: pendingRequests.length,
  };
}

export function threadSettled(threadRead) {
  const status = String(threadRead?.status || "").toLowerCase();
  if (RUNNING_THREAD_STATUSES.has(status)) {
    return false;
  }
  if (TERMINAL_THREAD_STATUSES.has(status)) {
    return true;
  }
  return !threadRead?.active_turn_id && !threadRead?.activeTurnId;
}
