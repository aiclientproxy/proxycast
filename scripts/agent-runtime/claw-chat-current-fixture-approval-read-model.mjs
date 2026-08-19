import {
  APP_SERVER_METHOD_SESSION_READ,
  APPROVAL_REQUEST_CANCEL_DONE_TEXT,
  APPROVAL_REQUEST_RESUME_APPROVAL_PROMPT,
  APPROVAL_REQUEST_RESUME_COMMAND,
  APPROVAL_REQUEST_RESUME_DONE_TEXT,
  APPROVAL_REQUEST_RESUME_PROMPT,
  APPROVAL_REQUEST_RESUME_REQUEST_ID,
  APPROVAL_REQUEST_RESUME_RESULT_TEXT,
  APPROVAL_REQUEST_DECLINE_DONE_TEXT,
  APPROVAL_REQUEST_DECLINE_RESULT_TEXT,
  APPROVAL_REQUEST_RESUME_TOOL_CALL_ID,
  SESSION_ID,
} from "./claw-chat-current-fixture-constants.mjs";
import { invokeAppServerFromPage } from "./claw-chat-current-fixture-rpc.mjs";
import {
  collectReadModelItems,
  collectReadModelTurns,
  readModelTurnId,
} from "./claw-chat-current-fixture-read-model-core.mjs";
import { sanitizeJson, sleep } from "./claw-chat-current-fixture-utils.mjs";

const METHOD_COMMAND_EXECUTION_REQUEST_APPROVAL =
  "item/commandExecution/requestApproval";
const SERVER_REQUEST_LIFECYCLE_TRACE_KEY =
  "lime:debug:app-server-server-request-lifecycle:v1";

function collectPendingRequests(readModel) {
  const detail = readModel?.detail ?? readModel ?? {};
  const threadRead = detail?.thread_read ?? detail?.threadRead ?? {};
  return [
    ...(Array.isArray(readModel?.pending_requests)
      ? readModel.pending_requests
      : []),
    ...(Array.isArray(readModel?.pendingRequests)
      ? readModel.pendingRequests
      : []),
    ...(Array.isArray(detail?.pending_requests) ? detail.pending_requests : []),
    ...(Array.isArray(detail?.pendingRequests) ? detail.pendingRequests : []),
    ...(Array.isArray(threadRead?.pending_requests)
      ? threadRead.pending_requests
      : []),
    ...(Array.isArray(threadRead?.pendingRequests)
      ? threadRead.pendingRequests
      : []),
  ].filter(Boolean);
}

function readLatestTurnStatus(readModel) {
  const canonicalTurns = Array.isArray(readModel?.thread?.turns)
    ? readModel.thread.turns
    : [];
  return (
    readModel?.detail?.thread_read?.runtime_summary?.latestTurnStatus ??
    readModel?.detail?.threadRead?.runtimeSummary?.latestTurnStatus ??
    readModel?.detail?.thread_read?.status ??
    readModel?.detail?.threadRead?.status ??
    readModel?.detail?.status ??
    canonicalTurns.at(-1)?.status ??
    null
  );
}

function findPendingApprovalServerRequest(lifecycleEntries) {
  const entries = Array.isArray(lifecycleEntries) ? lifecycleEntries : [];
  const requestIndex = entries.findLastIndex(
    (entry) =>
      entry?.kind === "request" &&
      entry?.method === METHOD_COMMAND_EXECUTION_REQUEST_APPROVAL &&
      entry?.approvalId === APPROVAL_REQUEST_RESUME_REQUEST_ID,
  );
  if (requestIndex < 0) {
    return null;
  }
  const request = entries[requestIndex];
  const settled = entries
    .slice(requestIndex + 1)
    .some(
      (entry) =>
        entry?.id === request.id &&
        (entry?.kind === "response" || entry?.kind === "resolved"),
    );
  return settled ? null : request;
}

function readCanonicalToolIdentity(readModel, request) {
  if (!request?.itemId) {
    return null;
  }
  for (const turn of collectReadModelTurns(readModel)) {
    const item = (Array.isArray(turn?.items) ? turn.items : []).find(
      (candidate) =>
        (candidate?.id ?? candidate?.item_id ?? candidate?.itemId) ===
        request.itemId,
    );
    if (item) {
      return {
        item,
        turnId: readModelTurnId(turn),
      };
    }
  }
  const item = collectReadModelItems(readModel).find(
    (candidate) =>
      (candidate?.id ?? candidate?.item_id ?? candidate?.itemId) ===
      request.itemId,
  );
  return item
    ? {
        item,
        turnId: item?.turn_id ?? item?.turnId ?? null,
      }
    : null;
}

function containsStringValue(value, expected) {
  if (typeof value === "string") {
    return value.includes(expected);
  }
  if (Array.isArray(value)) {
    return value.some((item) => containsStringValue(item, expected));
  }
  if (value && typeof value === "object") {
    return Object.values(value).some((item) =>
      containsStringValue(item, expected),
    );
  }
  return false;
}

export function summarizeApprovalPendingReadModel(readModel, lifecycleEntries) {
  const readModelPendingRequests = collectPendingRequests(readModel);
  const request = findPendingApprovalServerRequest(lifecycleEntries);
  const canonicalTool = readCanonicalToolIdentity(readModel, request);
  const canonicalToolItem = canonicalTool?.item ?? null;
  const canonicalToolName =
    canonicalToolItem?.tool ??
    canonicalToolItem?.tool_name ??
    canonicalToolItem?.toolName ??
    canonicalToolItem?.name ??
    (canonicalToolItem?.type === "commandExecution"
      ? "exec_command"
      : null) ??
    null;
  return sanitizeJson({
    pendingRequestCount: request ? 1 : 0,
    readModelPendingRequestCount: readModelPendingRequests.length,
    latestTurnStatus: readLatestTurnStatus(readModel),
    hasPendingRequest: Boolean(request),
    outerRequestId: request?.id ?? null,
    requestId: request?.approvalId ?? null,
    requestType: request?.method ?? null,
    requestStatus: request ? "pending" : null,
    threadId: request?.threadId ?? null,
    turnId: request?.turnId ?? null,
    itemId: request?.itemId ?? null,
    payloadActionType: request ? "tool_confirmation" : null,
    payloadToolName: canonicalToolName,
    hasCanonicalToolItem: Boolean(canonicalToolItem),
    canonicalToolItemType:
      canonicalToolItem?.type ?? canonicalToolItem?.item_type ?? null,
    canonicalToolItemStatus: canonicalToolItem?.status ?? null,
    canonicalToolItemScoped:
      Boolean(request?.turnId) && canonicalTool?.turnId === request?.turnId,
    includesPrompt: containsStringValue(
      readModel,
      APPROVAL_REQUEST_RESUME_PROMPT,
    ),
    includesApprovalPrompt: containsStringValue(
      readModel,
      APPROVAL_REQUEST_RESUME_APPROVAL_PROMPT,
    ),
    includesCommand: containsStringValue(
      readModel,
      APPROVAL_REQUEST_RESUME_COMMAND,
    ),
    includesRequestId:
      request?.approvalId === APPROVAL_REQUEST_RESUME_REQUEST_ID,
    includesToolCallId:
      containsStringValue(readModel, APPROVAL_REQUEST_RESUME_TOOL_CALL_ID) &&
      request?.itemId === APPROVAL_REQUEST_RESUME_TOOL_CALL_ID,
  });
}

export function summarizeApprovalCompletedReadModel(readModel) {
  const serialized = JSON.stringify(readModel || {});
  const pendingRequests = collectPendingRequests(readModel);
  return sanitizeJson({
    pendingRequestCount: pendingRequests.length,
    latestTurnStatus: readLatestTurnStatus(readModel),
    includesPrompt: serialized.includes(APPROVAL_REQUEST_RESUME_PROMPT),
    includesApprovalPrompt: serialized.includes(
      APPROVAL_REQUEST_RESUME_APPROVAL_PROMPT,
    ),
    includesRequestId: serialized.includes(APPROVAL_REQUEST_RESUME_REQUEST_ID),
    includesToolCallId: serialized.includes(
      APPROVAL_REQUEST_RESUME_TOOL_CALL_ID,
    ),
    includesToolResult: serialized.includes(
      APPROVAL_REQUEST_RESUME_RESULT_TEXT,
    ),
    includesActionResolved:
      serialized.includes("action.resolved") ||
      serialized.includes("action_resolved") ||
      serialized.includes('"decision":"allow_for_session"') ||
      serialized.includes('"decision":"approve"') ||
      serialized.includes('"decision":"approved"'),
    includesAssistantDone: serialized.includes(
      APPROVAL_REQUEST_RESUME_DONE_TEXT,
    ),
    includesAssistantSummary: serialized.includes(
      APPROVAL_REQUEST_RESUME_RESULT_TEXT,
    ),
  });
}

export function summarizeApprovalDecisionReadModel(readModel, decision) {
  const serialized = JSON.stringify(readModel || {});
  const pendingRequests = collectPendingRequests(readModel);
  const latestTurnStatus = readLatestTurnStatus(readModel);
  const expectedCanonicalDecision =
    decision === "decline"
      ? "denied"
      : decision === "cancel"
        ? "abort"
        : decision;
  const approvalEvents = collectEventLikeRecords(readModel).filter(
    (event) =>
      event.eventType.toLowerCase() === "approval" &&
      eventPayloadRequestId(event.payload) ===
        APPROVAL_REQUEST_RESUME_REQUEST_ID,
  );
  const includesCanonicalTerminalApproval = approvalEvents.some((event) =>
    ["completed", "failed", "interrupted", "cancelled"].includes(
      String(event.status || "").toLowerCase(),
    ),
  );
  const includesCanonicalDecision = approvalEvents.some(
    (event) => event.payload?.decision === expectedCanonicalDecision,
  );
  const canonicalApprovalItems = collectReadModelItems(readModel).filter(
    (item) =>
      (item?.type === "approval_request" || item?.kind === "approval") &&
      String(item?.request_id ?? item?.requestId ?? "") ===
        APPROVAL_REQUEST_RESUME_REQUEST_ID,
  );
  const includesCanonicalTerminalItem = canonicalApprovalItems.some((item) =>
    ["completed", "failed", "interrupted", "cancelled"].includes(
      String(item?.status || "").toLowerCase(),
    ),
  );
  const includesCanonicalItemDecision = canonicalApprovalItems.some((item) => {
    const response = item?.response;
    const itemDecision =
      typeof response === "string"
        ? response
        : (response?.decision ?? item?.decision);
    return itemDecision === expectedCanonicalDecision;
  });
  return sanitizeJson({
    decision,
    pendingRequestCount: pendingRequests.length,
    latestTurnStatus,
    latestTurnCanceled:
      latestTurnStatus === "canceled" ||
      latestTurnStatus === "cancelled" ||
      latestTurnStatus === "interrupted",
    includesPrompt: serialized.includes(APPROVAL_REQUEST_RESUME_PROMPT),
    includesApprovalPrompt: serialized.includes(
      APPROVAL_REQUEST_RESUME_APPROVAL_PROMPT,
    ),
    includesRequestId: serialized.includes(APPROVAL_REQUEST_RESUME_REQUEST_ID),
    includesToolCallId: serialized.includes(
      APPROVAL_REQUEST_RESUME_TOOL_CALL_ID,
    ),
    includesToolResult: serialized.includes(
      APPROVAL_REQUEST_RESUME_RESULT_TEXT,
    ),
    includesActionResolved:
      serialized.includes("action.resolved") ||
      serialized.includes("action_resolved") ||
      includesCanonicalTerminalApproval ||
      includesCanonicalTerminalItem,
    includesDecision:
      serialized.includes(`"decision":"${decision}"`) ||
      serialized.includes(`"decision": "${decision}"`) ||
      includesCanonicalDecision ||
      includesCanonicalItemDecision,
    includesDeclineResult: serialized.includes(
      APPROVAL_REQUEST_DECLINE_RESULT_TEXT,
    ),
    includesDeclineDone: serialized.includes(
      APPROVAL_REQUEST_DECLINE_DONE_TEXT,
    ),
    includesCancelDone: serialized.includes(APPROVAL_REQUEST_CANCEL_DONE_TEXT),
    includesCanceled:
      serialized.includes('"status":"canceled"') ||
      serialized.includes('"status": "canceled"') ||
      serialized.includes('"status":"interrupted"') ||
      serialized.includes('"status": "interrupted"') ||
      serialized.includes("turn.canceled") ||
      serialized.includes("turn_canceled") ||
      approvalEvents.some((event) => event.payload?.decision === "abort") ||
      canonicalApprovalItems.some((item) => item?.response === "abort"),
  });
}

function collectEventLikeRecords(value, output = []) {
  if (!value || typeof value !== "object") {
    return output;
  }
  if (Array.isArray(value)) {
    value.forEach((item) => collectEventLikeRecords(item, output));
    return output;
  }
  const eventType =
    value.event_type ?? value.eventType ?? value.type ?? value.kind;
  const payload = value.payload;
  if (typeof eventType === "string" && payload && typeof payload === "object") {
    output.push({ eventType, payload, status: value.status });
  }
  Object.values(value).forEach((item) => collectEventLikeRecords(item, output));
  return output;
}

function eventPayloadRequestId(payload) {
  return (
    payload?.requestId ??
    payload?.request_id ??
    payload?.actionId ??
    payload?.action_id ??
    null
  );
}

export async function waitForApprovalPendingReadModel(
  page,
  options,
  requestLog,
) {
  const startedAt = Date.now();
  let lastSummary = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const read = await invokeAppServerFromPage(
      page,
      APP_SERVER_METHOD_SESSION_READ,
      {
        threadId: options.threadId,
        includeTurns: true,
      },
      requestLog,
    );
    const lifecycleEntries = await page.evaluate((key) => {
      try {
        const parsed = JSON.parse(window.localStorage.getItem(key) || "[]");
        return Array.isArray(parsed) ? parsed : [];
      } catch {
        return [];
      }
    }, SERVER_REQUEST_LIFECYCLE_TRACE_KEY);
    lastSummary = summarizeApprovalPendingReadModel(
      read.result,
      lifecycleEntries,
    );
    if (
      lastSummary.hasPendingRequest === true &&
      lastSummary.readModelPendingRequestCount === 0 &&
      lastSummary.payloadActionType === "tool_confirmation" &&
      lastSummary.payloadToolName === "exec_command" &&
      lastSummary.hasCanonicalToolItem === true &&
      lastSummary.canonicalToolItemScoped === true &&
      lastSummary.threadId === options.threadId &&
      typeof lastSummary.turnId === "string" &&
      lastSummary.turnId.length > 0 &&
      lastSummary.itemId === APPROVAL_REQUEST_RESUME_TOOL_CALL_ID &&
      (typeof lastSummary.outerRequestId === "string" ||
        typeof lastSummary.outerRequestId === "number")
    ) {
      return lastSummary;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `审批 pending read model 未出现: ${JSON.stringify(
      sanitizeJson(lastSummary),
    )}`,
  );
}
