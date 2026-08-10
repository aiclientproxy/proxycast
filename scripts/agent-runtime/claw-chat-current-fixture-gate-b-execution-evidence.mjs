import { decodeJsonRpcLines } from "./claw-chat-current-fixture-rpc.mjs";

const TURN_START_METHOD = "turn/start";
const SESSION_READ_METHOD = "thread/read";
const COLLABORATION_MODE_LIST_METHOD = "collaborationMode/list";
const PERMISSION_PROFILE_LIST_METHOD = "permissionProfile/list";
const PERMISSION_PROFILE_SANDBOX_POLICIES = new Map([
  [":read-only", "read-only"],
  [":workspace", "workspace-write"],
  [":danger-full-access", "danger-full-access"],
]);
const TERMINAL_STATUSES = new Set([
  "canceled",
  "cancelled",
  "completed",
  "failed",
]);
const PENDING_STATUSES = new Set([
  "accepted",
  "in_progress",
  "pending",
  "queued",
  "requires_action",
  "running",
  "waiting",
  "waiting_for_approval",
]);
const TERMINAL_EVENTS = new Set([
  "turn.canceled",
  "turn.completed",
  "turn.failed",
]);

export async function collectGateBGuiEvidence(page, expectedIdentity = null) {
  return await page.evaluate((expectedTurnId) => {
    const isVisible = (element) => {
      if (!(element instanceof HTMLElement)) {
        return false;
      }
      const rect = element.getBoundingClientRect();
      const style = window.getComputedStyle(element);
      return (
        rect.width > 0 &&
        rect.height > 0 &&
        style.display !== "none" &&
        style.visibility !== "hidden"
      );
    };
    const textarea = Array.from(
      document.querySelectorAll('textarea[name="agent-chat-message"]'),
    )
      .filter(isVisible)
      .at(-1);
    const turnGroups = Array.from(
      document.querySelectorAll('[data-testid="message-turn-group"]'),
    ).filter(isVisible);
    const assistantGroups = turnGroups.filter((group) =>
      Array.from(
        group.querySelectorAll('[data-message-role="assistant"]'),
      ).some(isVisible),
    );
    const canonicalAssistantGroups = assistantGroups.filter((group) =>
      Array.from(
        group.querySelectorAll('[data-message-role="assistant"]'),
      ).some((element) => {
        if (!isVisible(element)) {
          return false;
        }
        const itemId = element.getAttribute("data-thread-item-id") || "";
        const timelineItems = element.getAttribute("data-timeline-items") || "";
        return (
          itemId.startsWith("item_agent-message") ||
          timelineItems
            .split("|")
            .some((entry) => entry.trim().startsWith("agent_message:"))
        );
      }),
    );
    const pendingAssistantGroups = assistantGroups.filter((group) =>
      Array.from(
        group.querySelectorAll(
          '[data-testid="assistant-first-token-runtime-status"]',
        ),
      ).some(isVisible),
    );
    const expectedTurnGroup = expectedTurnId
      ? [...turnGroups]
          .reverse()
          .find(
            (group) =>
              group.getAttribute("data-runtime-turn-id") === expectedTurnId,
          )
      : null;
    const turnGroup =
      expectedTurnGroup ??
      pendingAssistantGroups.at(-1) ??
      canonicalAssistantGroups.at(-1) ??
      assistantGroups.at(-1) ??
      turnGroups.at(-1) ??
      null;
    const assistant = Array.from(
      turnGroup?.querySelectorAll('[data-message-role="assistant"]') ?? [],
    )
      .filter(isVisible)
      .at(-1);
    const stopButtonVisible = Array.from(
      document.querySelectorAll("button"),
    ).some((button) => {
      if (!isVisible(button) || button.disabled) {
        return false;
      }
      const label = [
        button.getAttribute("title") || "",
        button.getAttribute("aria-label") || "",
        button.textContent || "",
      ].join("\n");
      return (
        label.includes("停止") ||
        label.includes("终止") ||
        /\bStop\b/i.test(label)
      );
    });
    const canonicalThreadItemIds = Array.from(
      turnGroup?.querySelectorAll("[data-thread-item-id]") ?? [],
    )
      .map((element) => element.getAttribute("data-thread-item-id"))
      .filter((value) => typeof value === "string" && value.trim().length > 0);
    const timelineItemIds = String(
      assistant?.getAttribute("data-timeline-items") ||
        turnGroup?.getAttribute("data-timeline-items") ||
        "",
    )
      .split("|")
      .map((entry) => entry.trim())
      .filter((entry) => entry.startsWith("agent_message:"))
      .map((entry) => entry.slice("agent_message:".length).trim())
      .filter((value) => value.length > 0);
    canonicalThreadItemIds.push(...timelineItemIds);
    const turnId =
      assistant?.getAttribute("data-runtime-turn-id") ||
      turnGroup?.getAttribute("data-runtime-turn-id") ||
      null;
    const turnStatus = String(
      turnGroup?.getAttribute("data-runtime-turn-status") || "",
    )
      .trim()
      .toLowerCase();
    const turnIsTerminal = [
      "aborted",
      "canceled",
      "cancelled",
      "completed",
      "failed",
      "interrupted",
    ].includes(turnStatus);
    const turnIsPending = [
      "accepted",
      "in_progress",
      "pending",
      "queued",
      "running",
      "waiting",
    ].includes(turnStatus);

    return {
      sessionId: textarea?.dataset.sessionId || null,
      turnId,
      itemId: canonicalThreadItemIds.at(-1) || null,
      state: turnIsTerminal
        ? "terminal"
        : turnIsPending || stopButtonVisible
          ? "pending"
          : turnId
            ? "terminal"
            : "idle",
      turnStatus: turnStatus || null,
      stopButtonVisible,
      turnGroupCount: turnGroups.length,
    };
  }, expectedIdentity?.turnId ?? null);
}

export function buildGateBExecutionEvidence({
  traceMessages,
  appServerRequests,
  backendLedger,
  guiEvidence,
  expectedIdentity,
}) {
  const requests = Array.isArray(appServerRequests) ? appServerRequests : [];
  const ledger = Array.isArray(backendLedger) ? backendLedger : [];
  const turnStartRequests = requests.filter(
    (entry) => entry?.method === TURN_START_METHOD,
  );
  const readResponses = requests
    .filter((entry) => entry?.method === SESSION_READ_METHOD)
    .map((entry) => entry?.response)
    .filter(Boolean);
  const backendTurns = ledger.filter((entry) => entry?.kind === "turnStart");
  const traceTurnStarts = collectTraceTurnStarts(traceMessages);
  const traceRequests = collectTraceJsonRpcRequests(traceMessages);
  const primaryTurnId = firstString(
    guiEvidence?.turnId,
    turnStartRequests.at(-1)?.response?.turnId,
    backendTurns.at(-1)?.turnId,
    readResponses.at(-1)?.activeTurnId,
    readResponses.at(-1)?.turns?.at(-1)?.turnId,
  );
  const turnStartRequest = findByTurnId(
    turnStartRequests,
    primaryTurnId,
    (entry) => entry?.response?.turnId,
  );
  const backendTurn = findByTurnId(
    backendTurns,
    primaryTurnId,
    (entry) => entry?.turnId,
  );
  const readModel =
    [...readResponses]
      .reverse()
      .find((entry) => readResponseContainsTurn(entry, primaryTurnId)) ??
    readResponses.at(-1) ??
    null;
  const readModelTurn = primaryTurnId
    ? (readModel?.turns?.find((turn) => turn?.turnId === primaryTurnId) ?? null)
    : null;
  const traceTurnStart = findTraceTurnStart(traceTurnStarts, {
    sessionId: firstString(
      guiEvidence?.sessionId,
      expectedIdentity?.sessionId,
      turnStartRequest?.response?.sessionId,
    ),
    turnId: primaryTurnId,
  });
  const readModelItem = guiEvidence?.itemId
    ? (readModel?.items?.find((item) => item?.itemId === guiEvidence.itemId) ??
      null)
    : null;
  const sources = {
    expected: compactIdentity(expectedIdentity),
    renderer: compactIdentity(guiEvidence),
    trace: compactIdentity(traceTurnStart),
    appServer: compactIdentity(turnStartRequest?.response),
    runtime: compactIdentity(backendTurn),
    readModel: compactIdentity({
      sessionId: readModel?.sessionId,
      threadId: readModel?.threadId,
      turnId: readModelTurn?.turnId ?? readModel?.activeTurnId,
      itemId: readModelItem?.itemId,
    }),
  };
  const identity = buildIdentitySummary(sources, primaryTurnId);
  const readModelStatus = normalizeStatus(
    readModelTurn?.status ?? readModel?.latestTurnStatus,
  );
  const backendTerminalEvents = ledger
    .filter((entry) => !primaryTurnId || entry?.turnId === primaryTurnId)
    .flatMap((entry) =>
      Array.isArray(entry?.eventTypes) ? entry.eventTypes : [entry?.eventType],
    )
    .filter((eventType) => TERMINAL_EVENTS.has(eventType));
  const guiState = firstString(guiEvidence?.state) ?? "unknown";
  const terminalObserved =
    TERMINAL_STATUSES.has(readModelStatus) || backendTerminalEvents.length > 0;
  const pendingObserved = PENDING_STATUSES.has(readModelStatus);
  const kind =
    guiState === "terminal" && terminalObserved
      ? "terminal"
      : guiState === "pending" && pendingObserved
        ? "pending"
        : guiState === "idle" && !primaryTurnId
          ? "terminal"
          : "unknown";
  const collaborationMode = buildCollaborationModeEvidence({
    traceRequests,
    backendTurn,
  });
  const permissionProfile = buildPermissionProfileEvidence({
    traceRequests,
    backendTurn,
  });

  return {
    identity,
    outcome: {
      kind,
      guiState,
      readModelStatus: readModelStatus || null,
      backendTerminalEvents: [...new Set(backendTerminalEvents)].sort(),
      turnId: primaryTurnId,
      explicit: kind === "terminal" || kind === "pending",
    },
    collaborationMode,
    permissionProfile,
  };
}

function buildPermissionProfileEvidence({ traceRequests, backendTurn }) {
  const turnStartIndex = findLastRequestIndex(
    traceRequests,
    (entry) => entry.method === TURN_START_METHOD,
  );
  const catalogIndexes = traceRequests
    .map((entry, index) =>
      entry.method === PERMISSION_PROFILE_LIST_METHOD ? index : -1,
    )
    .filter((index) => index >= 0);
  const latestCatalogIndex = catalogIndexes.at(-1) ?? -1;
  const turnStart = turnStartIndex >= 0 ? traceRequests[turnStartIndex] : null;
  const params = turnStart?.params ?? {};
  const runtimeRequest = backendTurn?.runtimeRequest ?? {};
  const metadata = runtimeRequest.metadata ?? {};
  const profileId = normalizedString(params.permissions);
  const runtimeSandboxPolicy = normalizedString(
    runtimeRequest.sandboxPolicy ?? runtimeRequest.sandbox_policy,
  );
  const activePermissionProfileId = normalizedString(
    metadata.activePermissionProfile?.id ??
      metadata.active_permission_profile?.id,
  );
  const expectedSandboxPolicy =
    PERMISSION_PROFILE_SANDBOX_POLICIES.get(profileId) ?? null;

  return {
    catalogMethod: PERMISSION_PROFILE_LIST_METHOD,
    catalogRequestCount: catalogIndexes.length,
    catalogBeforeTurnStart:
      latestCatalogIndex >= 0 && turnStartIndex > latestCatalogIndex,
    wire: {
      profileId,
      legacySandboxPolicyPresent: Object.prototype.hasOwnProperty.call(
        params,
        "sandboxPolicy",
      ),
    },
    runtime: {
      matchedPrimaryTurn: backendTurn != null,
      turnId: normalizedString(backendTurn?.turnId),
      sandboxPolicy: runtimeSandboxPolicy,
      activePermissionProfileId,
    },
    wireRuntimeConsistent:
      profileId != null &&
      !Object.prototype.hasOwnProperty.call(params, "sandboxPolicy") &&
      expectedSandboxPolicy === runtimeSandboxPolicy &&
      activePermissionProfileId === profileId,
  };
}

function buildCollaborationModeEvidence({ traceRequests, backendTurn }) {
  const turnStartIndex = findLastRequestIndex(
    traceRequests,
    (entry) => entry.method === TURN_START_METHOD,
  );
  const catalogIndexes = traceRequests
    .map((entry, index) =>
      entry.method === COLLABORATION_MODE_LIST_METHOD ? index : -1,
    )
    .filter((index) => index >= 0);
  const latestCatalogIndex = catalogIndexes.at(-1) ?? -1;
  const turnStart = turnStartIndex >= 0 ? traceRequests[turnStartIndex] : null;
  const params = turnStart?.params ?? {};
  const runtimeRequest = backendTurn?.runtimeRequest ?? {};
  const wireMode = compactCollaborationMode(params.collaborationMode);
  const runtimeMode = compactCollaborationMode(
    runtimeRequest.collaborationMode,
  );
  const wire = {
    model: normalizedString(params.model),
    effort: normalizedString(params.effort),
    collaborationMode: wireMode,
  };
  const runtime = {
    modelPreference: normalizedString(runtimeRequest.modelPreference),
    reasoningEffort: normalizedString(runtimeRequest.reasoningEffort),
    collaborationMode: runtimeMode,
  };

  return {
    catalogMethod: COLLABORATION_MODE_LIST_METHOD,
    catalogRequestCount: catalogIndexes.length,
    catalogBeforeTurnStart:
      latestCatalogIndex >= 0 && turnStartIndex > latestCatalogIndex,
    wire,
    runtime,
    wireRuntimeConsistent:
      wireMode.mode != null &&
      JSON.stringify(wireMode) === JSON.stringify(runtimeMode) &&
      wire.effort === runtime.reasoningEffort &&
      wireMode.settings.model === runtime.modelPreference,
  };
}

function collectTraceJsonRpcRequests(traceMessages) {
  return (Array.isArray(traceMessages) ? traceMessages : [])
    .filter(
      (entry) =>
        entry?.command === "app_server_handle_json_lines" &&
        entry?.transport === "electron-ipc" &&
        entry?.status === "success",
    )
    .flatMap((entry) =>
      decodeJsonRpcLines(entry?.args_preview?.request?.lines)
        .filter((message) => typeof message?.method === "string")
        .map((message) => ({
          method: message.method,
          params: message.params ?? {},
        })),
    );
}

function findLastRequestIndex(entries, predicate) {
  for (let index = entries.length - 1; index >= 0; index -= 1) {
    if (predicate(entries[index])) {
      return index;
    }
  }
  return -1;
}

function compactCollaborationMode(value) {
  const settings = value?.settings ?? {};
  return {
    mode: normalizedString(value?.mode),
    settings: {
      model: normalizedString(settings.model),
      reasoning_effort: normalizedString(settings.reasoning_effort),
      developer_instructions:
        typeof settings.developer_instructions === "string"
          ? settings.developer_instructions
          : null,
    },
  };
}

function normalizedString(value) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function collectTraceTurnStarts(traceMessages) {
  return (Array.isArray(traceMessages) ? traceMessages : [])
    .filter((entry) => entry?.command === "app_server_handle_json_lines")
    .flatMap((entry) => decodeJsonRpcLines(entry?.args_preview?.request?.lines))
    .filter((message) => message?.method === TURN_START_METHOD)
    .map((message) => ({
      sessionId: firstString(
        message.params?.sessionId,
        message.params?.session_id,
      ),
      threadId: firstString(
        message.params?.threadId,
        message.params?.thread_id,
      ),
      turnId: firstString(message.params?.turnId, message.params?.turn_id),
    }));
}

function findTraceTurnStart(entries, identity) {
  const reversed = [...entries].reverse();
  if (identity.turnId) {
    const exactTurn = reversed.find(
      (entry) => entry.turnId === identity.turnId,
    );
    if (exactTurn) {
      return exactTurn;
    }
  }
  return (
    reversed.find(
      (entry) => identity.sessionId && entry.sessionId === identity.sessionId,
    ) ??
    entries.at(-1) ??
    null
  );
}

function findByTurnId(entries, turnId, readTurnId) {
  if (turnId) {
    return (
      [...entries].reverse().find((entry) => readTurnId(entry) === turnId) ??
      null
    );
  }
  return entries.at(-1) ?? null;
}

function readResponseContainsTurn(response, turnId) {
  if (!turnId) {
    return false;
  }
  return (
    response?.activeTurnId === turnId ||
    response?.turns?.some((turn) => turn?.turnId === turnId) === true
  );
}

function buildIdentitySummary(sources, primaryTurnId) {
  const sessionIds = collectSourceValues(sources, "sessionId");
  const threadIds = collectSourceValues(sources, "threadId");
  const turnIds = collectSourceValues(sources, "turnId");
  const itemIds = collectSourceValues(sources, "itemId");
  const sessionConsistent =
    sessionIds.length >= 3 && new Set(sessionIds).size === 1;
  const threadConsistent =
    threadIds.length < 2 || new Set(threadIds).size === 1;
  const turnConsistent = primaryTurnId
    ? sources.renderer.turnId === primaryTurnId &&
      turnIds.length >= 2 &&
      new Set(turnIds).size === 1
    : turnIds.length === 0;
  const itemConsistent = sources.renderer.itemId
    ? sources.readModel.itemId === sources.renderer.itemId &&
      new Set(itemIds).size === 1
    : itemIds.length < 2 || new Set(itemIds).size === 1;

  return {
    sources,
    sessionConsistent,
    threadConsistent,
    turnConsistent,
    itemConsistent,
    consistent:
      sessionConsistent && threadConsistent && turnConsistent && itemConsistent,
  };
}

function collectSourceValues(sources, key) {
  return Object.values(sources)
    .map((source) => source[key])
    .filter(Boolean);
}

function compactIdentity(value) {
  return {
    sessionId: firstString(value?.sessionId, value?.session_id),
    threadId: firstString(value?.threadId, value?.thread_id),
    turnId: firstString(value?.turnId, value?.turn_id),
    itemId: firstString(value?.itemId, value?.item_id),
  };
}

function normalizeStatus(value) {
  return typeof value === "string" ? value.trim().toLowerCase() : "";
}

function firstString(...values) {
  return (
    values
      .find((value) => typeof value === "string" && value.trim().length > 0)
      ?.trim() ?? null
  );
}
