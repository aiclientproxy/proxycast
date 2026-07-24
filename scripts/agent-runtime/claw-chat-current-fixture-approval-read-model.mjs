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
  APPROVAL_REQUEST_RESUME_SECOND_DONE_TEXT,
  APPROVAL_REQUEST_RESUME_SECOND_PROMPT,
  APPROVAL_REQUEST_RESUME_SECOND_RESULT_TEXT,
  APPROVAL_REQUEST_RESUME_TOOL_CALL_ID,
  SESSION_ID,
} from "./claw-chat-current-fixture-constants.mjs";
import { invokeAppServerFromPage } from "./claw-chat-current-fixture-rpc.mjs";
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

function collectReadModelItems(readModel) {
  const detail = readModel?.detail ?? readModel ?? {};
  const threadRead = detail?.thread_read ?? detail?.threadRead ?? {};
  return [
    ...(Array.isArray(readModel?.items) ? readModel.items : []),
    ...(Array.isArray(readModel?.thread_items) ? readModel.thread_items : []),
    ...(Array.isArray(readModel?.threadItems) ? readModel.threadItems : []),
    ...(Array.isArray(detail?.items) ? detail.items : []),
    ...(Array.isArray(detail?.thread_items) ? detail.thread_items : []),
    ...(Array.isArray(detail?.threadItems) ? detail.threadItems : []),
    ...(Array.isArray(threadRead?.items) ? threadRead.items : []),
    ...(Array.isArray(threadRead?.thread_items) ? threadRead.thread_items : []),
    ...(Array.isArray(threadRead?.threadItems) ? threadRead.threadItems : []),
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
  const settled = entries.slice(requestIndex + 1).some(
    (entry) =>
      entry?.id === request.id &&
      (entry?.kind === "response" || entry?.kind === "resolved"),
  );
  return settled ? null : request;
}

export function summarizeApprovalPendingReadModel(
  readModel,
  lifecycleEntries,
) {
  const readModelPendingRequests = collectPendingRequests(readModel);
  const request = findPendingApprovalServerRequest(lifecycleEntries);
  const readModelSerialized = JSON.stringify(readModel || {});
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
    payloadToolName: request ? "exec_command" : null,
    includesPrompt: readModelSerialized.includes(
      APPROVAL_REQUEST_RESUME_PROMPT,
    ),
    includesApprovalPrompt: readModelSerialized.includes(
      APPROVAL_REQUEST_RESUME_APPROVAL_PROMPT,
    ),
    includesCommand: readModelSerialized.includes(
      APPROVAL_REQUEST_RESUME_COMMAND,
    ),
    includesRequestId:
      request?.approvalId === APPROVAL_REQUEST_RESUME_REQUEST_ID,
    includesToolCallId:
      readModelSerialized.includes(APPROVAL_REQUEST_RESUME_TOOL_CALL_ID) &&
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

export function summarizeApprovalSessionCacheReadModel(
  readModel,
  secondTurnId,
) {
  const serialized = JSON.stringify(readModel || {});
  const pendingRequests = collectPendingRequests(readModel);
  const secondPermissionRequestId = secondTurnId
    ? `permission-${secondTurnId}`
    : null;
  const secondPermissionEvents = collectEventLikeRecords(readModel).filter(
    (event) =>
      secondPermissionRequestId &&
      eventPayloadRequestId(event.payload) === secondPermissionRequestId,
  );
  const includesApprovalSessionCacheMetadata = serialized.includes(
    "approval_session_cache",
  );
  const includesCacheResolvedSource = serialized.includes(
    "approval_session_cache",
  );
  const includesAllowForSession =
    serialized.includes("allow_for_session") ||
    serialized.includes("approvedForSession");
  const includesSecondPermissionRequestId = secondPermissionRequestId
    ? secondPermissionEvents.length > 0 ||
      serialized.includes(secondPermissionRequestId)
    : false;
  const includesActionRequiredForSecondPermission = secondPermissionEvents.some(
    (event) =>
      event.eventType === "action.required" ||
      event.eventType === "action_required",
  );
  const includesActionResolvedForSecondPermission =
    secondPermissionEvents.some(
      (event) =>
        ((event.eventType === "action.resolved" ||
          event.eventType === "action_resolved") &&
          (event.payload?.source === "approval_session_cache" ||
            event.payload?.decision === "allow_for_session")) ||
        (event.eventType === "approval" &&
          ["completed", "failed", "interrupted", "cancelled"].includes(
            event.status,
          ) &&
          event.payload?.decision === "approvedForSession"),
    ) ||
    (includesSecondPermissionRequestId &&
      includesAllowForSession &&
      !includesActionRequiredForSecondPermission);
  return sanitizeJson({
    pendingRequestCount: pendingRequests.length,
    latestTurnStatus: readLatestTurnStatus(readModel),
    includesSecondPrompt: serialized.includes(
      APPROVAL_REQUEST_RESUME_SECOND_PROMPT,
    ),
    includesSecondResult: serialized.includes(
      APPROVAL_REQUEST_RESUME_SECOND_RESULT_TEXT,
    ),
    includesSecondDone: serialized.includes(
      APPROVAL_REQUEST_RESUME_SECOND_DONE_TEXT,
    ),
    includesApprovalSessionCacheHit:
      serialized.includes("approval.session_cache.hit") ||
      includesActionResolvedForSecondPermission,
    includesApprovalSessionCacheMetadata,
    includesCacheResolvedSource,
    includesAllowForSession,
    includesSecondPermissionRequestId,
    includesActionRequiredForSecondPermission,
    includesActionResolvedForSecondPermission,
  });
}

function approvalCacheHarnessFromTurnStart(entry) {
  const harness = entry?.runtimeRequest?.metadata?.harness ?? {};
  return harness && typeof harness === "object" ? harness : {};
}

export function summarizeApprovalSecondTurnStart(entry) {
  const harness = approvalCacheHarnessFromTurnStart(entry);
  const browserAssist = harness.browser_assist ?? harness.browserAssist ?? {};
  const runtimeContract =
    browserAssist?.runtime_contract ?? browserAssist?.runtimeContract ?? {};
  const cache =
    harness.approval_session_cache ?? harness.approvalSessionCache ?? null;
  const cacheKey = cache?.key ?? {};
  const runtimeRequest = entry?.runtimeRequest ?? {};
  return sanitizeJson({
    sessionId: entry?.sessionId ?? null,
    threadId: entry?.threadId ?? null,
    turnId: entry?.turnId ?? null,
    inputText: entry?.inputText ?? null,
    approvalPolicy:
      runtimeRequest?.approval_policy ?? runtimeRequest?.approvalPolicy,
    sandboxPolicy:
      runtimeRequest?.sandbox_policy ?? runtimeRequest?.sandboxPolicy,
    browserAssistEnabled: browserAssist?.enabled ?? null,
    browserAssistProfileKey:
      browserAssist?.profile_key ?? browserAssist?.profileKey ?? null,
    browserAssistContractKey:
      runtimeContract?.contract_key ?? runtimeContract?.contractKey ?? null,
    approvalSessionCacheDecision: cache?.decision ?? null,
    approvalSessionCacheDecisionScope:
      cache?.decisionScope ?? cache?.decision_scope ?? null,
    approvalSessionCacheSourceRequestId:
      cache?.sourceRequestId ?? cache?.source_request_id ?? null,
    approvalSessionCacheKey: {
      actionKind: cacheKey?.actionKind ?? cacheKey?.action_kind ?? null,
      toolFamily: cacheKey?.toolFamily ?? cacheKey?.tool_family ?? null,
      approvalPolicy:
        cacheKey?.approvalPolicy ?? cacheKey?.approval_policy ?? null,
      sandboxPolicy:
        cacheKey?.sandboxPolicy ?? cacheKey?.sandbox_policy ?? null,
      contractKey: cacheKey?.contractKey ?? cacheKey?.contract_key ?? null,
    },
  });
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
      lastSummary.includesCommand === true &&
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
