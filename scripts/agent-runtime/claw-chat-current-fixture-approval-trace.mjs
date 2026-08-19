import {
  APP_SERVER_HANDLE_JSON_LINES_COMMAND,
} from "./claw-chat-current-fixture-constants.mjs";
import {
  decodeJsonRpcLines,
  readTraceMessages,
} from "./claw-chat-current-fixture-rpc.mjs";
import { sanitizeJson, sleep } from "./claw-chat-current-fixture-utils.mjs";

const METHOD_SERVER_REQUEST_RESOLVED = "serverRequest/resolved";
const SERVER_REQUEST_LIFECYCLE_TRACE_KEY =
  "lime:debug:app-server-server-request-lifecycle:v1";
const RUNTIME_TERMINAL_METHODS = new Set(["item/completed", "turn/completed"]);

function collectApprovalServerRequestResponses(traceMessages) {
  return traceMessages
    .filter((entry) => entry?.command === APP_SERVER_HANDLE_JSON_LINES_COMMAND)
    .flatMap((entry) =>
      decodeJsonRpcLines(entry?.args_preview?.request?.lines).map(
        (message) => ({
          id: message.id ?? null,
          decision: message.result?.decision ?? null,
          timestamp: entry.timestamp ?? null,
          transport: entry.transport ?? null,
          status: entry.status ?? null,
        }),
      ),
    )
    .filter(
      (message) =>
        (typeof message.id === "string" || typeof message.id === "number") &&
        typeof message.decision === "string",
    );
}

function messageMatchesTurn(message, threadId, turnId) {
  const params = message?.params;
  if (!params || typeof params !== "object" || Array.isArray(params)) {
    return false;
  }
  const messageThreadId =
    params.threadId ?? params.item?.threadId ?? params.turn?.threadId ?? null;
  const messageTurnId =
    params.turnId ?? params.item?.turnId ?? params.turn?.id ?? null;
  return messageThreadId === threadId && messageTurnId === turnId;
}

export function summarizeApprovalServerRequestLifecycle({
  traceMessages,
  lifecycleEntries = [],
  notifications,
  wireDecision,
  threadId,
  turnId,
}) {
  const lifecycleResponseIndex = lifecycleEntries.findLastIndex(
    (entry) =>
      entry?.kind === "response" &&
      entry?.decision === wireDecision &&
      lifecycleEntries.some(
        (request) =>
          request?.kind === "request" &&
          request?.id === entry?.id &&
          request?.threadId === threadId &&
          request?.turnId === turnId,
      ),
  );
  const lifecycleResponse =
    lifecycleResponseIndex >= 0
      ? lifecycleEntries[lifecycleResponseIndex]
      : null;
  const response =
    lifecycleResponse ??
    collectApprovalServerRequestResponses(traceMessages)
      .filter((candidate) => candidate.decision === wireDecision)
      .at(-1);
  const request = lifecycleEntries.find(
    (entry) => entry?.kind === "request" && entry?.id === response?.id,
  );
  const lifecycleResolvedIndex = lifecycleEntries.findIndex(
    (entry) => entry?.kind === "resolved" && entry?.id === response?.id,
  );
  const lifecycleTerminalIndex = lifecycleEntries.findIndex(
    (entry, index) =>
      index > lifecycleResolvedIndex &&
      entry?.kind === "terminal" &&
      entry?.threadId === threadId &&
      entry?.turnId === turnId,
  );
  const resolvedIndex = notifications.findIndex(
    (notification) =>
      notification?.method === METHOD_SERVER_REQUEST_RESOLVED &&
      notification?.params?.requestId === response?.id,
  );
  const terminalIndex = notifications.findIndex(
    (notification) =>
      RUNTIME_TERMINAL_METHODS.has(notification?.method) &&
      messageMatchesTurn(notification, threadId, turnId),
  );
  const resolved =
    lifecycleResolvedIndex >= 0
      ? lifecycleEntries[lifecycleResolvedIndex]
      : resolvedIndex >= 0
        ? notifications[resolvedIndex]
        : null;
  const runtimeTerminal =
    lifecycleTerminalIndex >= 0
      ? lifecycleEntries[lifecycleTerminalIndex]
      : terminalIndex >= 0
        ? notifications[terminalIndex]
        : null;
  return sanitizeJson({
    request: request ?? null,
    response: response ?? null,
    responseMatchesRequest:
      request != null &&
      response != null &&
      request.id === response.id &&
      request.threadId === threadId &&
      request.turnId === turnId,
    resolved: resolved
      ? {
          method: resolved.method,
          requestId:
            resolved.requestId ?? resolved.params?.requestId ?? resolved.id,
          threadId: resolved.threadId ?? resolved.params?.threadId ?? null,
        }
      : null,
    responseMatchesResolved:
      response != null &&
      (resolved?.requestId ?? resolved?.params?.requestId ?? resolved?.id) ===
        response.id,
    responseBeforeResolved:
      lifecycleResponseIndex >= 0 &&
      lifecycleResolvedIndex > lifecycleResponseIndex,
    runtimeTerminalMethod: runtimeTerminal?.method ?? null,
    resolvedBeforeRuntimeTerminal:
      lifecycleResolvedIndex >= 0 &&
      lifecycleTerminalIndex > lifecycleResolvedIndex,
  });
}

async function readApprovalServerRequestLifecycleEntries(page) {
  return await page.evaluate((key) => {
    try {
      const parsed = JSON.parse(window.localStorage.getItem(key) || "[]");
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }, SERVER_REQUEST_LIFECYCLE_TRACE_KEY);
}

export async function readApprovalServerRequestLifecycleSnapshot(
  page,
  { wireDecision, threadId, turnId },
) {
  const [traceRaw, lifecycleEntries] = await Promise.all([
    page.evaluate(() =>
      window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
    ),
    readApprovalServerRequestLifecycleEntries(page),
  ]);
  return summarizeApprovalServerRequestLifecycle({
    traceMessages: readTraceMessages(traceRaw),
    lifecycleEntries,
    notifications: [],
    wireDecision,
    threadId,
    turnId,
  });
}

export async function waitForApprovalServerRequestResponse(
  page,
  options,
  identity,
) {
  const startedAt = Date.now();
  let lastSummary = null;
  while (Date.now() - startedAt < Math.min(options.timeoutMs, 30_000)) {
    lastSummary = await readApprovalServerRequestLifecycleSnapshot(
      page,
      identity,
    );
    if (lastSummary.responseMatchesRequest === true) {
      return lastSummary;
    }
    await sleep(options.intervalMs);
  }
  return sanitizeJson(lastSummary);
}

export function summarizeApprovalHostInterruptLifecycle({
  lifecycleEntries = [],
  threadId,
  turnId,
}) {
  const requestIndex = lifecycleEntries.findIndex(
    (entry) =>
      entry?.kind === "request" &&
      entry?.threadId === threadId &&
      entry?.turnId === turnId,
  );
  const request = requestIndex >= 0 ? lifecycleEntries[requestIndex] : null;
  const resolvedIndex = lifecycleEntries.findIndex(
    (entry, index) =>
      index > requestIndex &&
      entry?.kind === "resolved" &&
      entry?.id === request?.id &&
      entry?.threadId === threadId,
  );
  const terminalIndex = lifecycleEntries.findIndex(
    (entry, index) =>
      index > resolvedIndex &&
      entry?.kind === "terminal" &&
      entry?.threadId === threadId &&
      entry?.turnId === turnId,
  );
  const responseCount = request
    ? lifecycleEntries.filter(
        (entry) => entry?.kind === "response" && entry?.id === request.id,
      ).length
    : 0;
  const resolved = resolvedIndex >= 0 ? lifecycleEntries[resolvedIndex] : null;
  const terminal = terminalIndex >= 0 ? lifecycleEntries[terminalIndex] : null;
  return sanitizeJson({
    request,
    resolved: resolved
      ? {
          method: resolved.method,
          requestId: resolved.id ?? resolved.requestId ?? null,
          threadId: resolved.threadId ?? null,
        }
      : null,
    responseCount,
    noRendererResponse: responseCount === 0,
    runtimeTerminalMethod: terminal?.method ?? null,
    resolvedBeforeRuntimeTerminal:
      resolvedIndex > requestIndex && terminalIndex > resolvedIndex,
  });
}

export async function waitForApprovalHostInterruptLifecycle(
  page,
  options,
  { threadId, turnId },
) {
  const startedAt = Date.now();
  let lastSummary = null;
  while (Date.now() - startedAt < Math.min(options.timeoutMs, 30_000)) {
    const lifecycleEntries = await page.evaluate((key) => {
      try {
        const parsed = JSON.parse(window.localStorage.getItem(key) || "[]");
        return Array.isArray(parsed) ? parsed : [];
      } catch {
        return [];
      }
    }, SERVER_REQUEST_LIFECYCLE_TRACE_KEY);
    lastSummary = summarizeApprovalHostInterruptLifecycle({
      lifecycleEntries,
      threadId,
      turnId,
    });
    if (
      lastSummary.noRendererResponse === true &&
      lastSummary.resolvedBeforeRuntimeTerminal === true
    ) {
      return lastSummary;
    }
    await sleep(options.intervalMs);
  }
  return sanitizeJson(lastSummary);
}

export async function waitForApprovalServerRequestLifecycle(
  page,
  options,
  { wireDecision, threadId, turnId },
) {
  const startedAt = Date.now();
  let lastSummary = null;
  while (Date.now() - startedAt < Math.min(options.timeoutMs, 30_000)) {
    lastSummary = await readApprovalServerRequestLifecycleSnapshot(page, {
      wireDecision,
      threadId,
      turnId,
    });
    if (
      lastSummary.responseMatchesResolved === true &&
      lastSummary.responseBeforeResolved === true &&
      lastSummary.resolvedBeforeRuntimeTerminal === true
    ) {
      return lastSummary;
    }
    await sleep(options.intervalMs);
  }
  return sanitizeJson(lastSummary);
}
