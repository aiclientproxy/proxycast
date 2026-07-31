import {
  readJsonl,
  readString,
  sanitizeJson,
  sleep,
} from "./claw-chat-current-fixture-utils.mjs";

export function sanitizeBackendLedgerForEvidence(backendLedger) {
  return backendLedger.map((entry) => {
    if (entry?.kind === "turnStart") {
      const runtimeHarness = entry.runtimeRequest?.metadata?.harness ?? {};
      const expert = entry.runtimeRequest?.metadata?.expert ?? {};
      const harnessExpert =
        runtimeHarness?.expert && typeof runtimeHarness.expert === "object"
          ? runtimeHarness.expert
          : {};
      const runtimeEnable =
        runtimeHarness?.workspace_skill_runtime_enable ??
        runtimeHarness?.workspaceSkillRuntimeEnable ??
        null;
      return sanitizeJson({
        kind: entry.kind,
        sessionId: entry.sessionId,
        turnId: entry.turnId,
        inputText: entry.inputText,
        providerPreference: entry.providerPreference,
        modelPreference: entry.modelPreference,
        runtimeMetadataHarnessSource: runtimeHarness?.source ?? null,
        workspaceSkillRuntimeEnable: runtimeEnable,
        expert: {
          expertId: expert?.expertId ?? expert?.expert_id ?? null,
          title: expert?.title ?? null,
          skillRefs: Array.isArray(expert?.skillRefs)
            ? expert.skillRefs
            : Array.isArray(expert?.skill_refs)
              ? expert.skill_refs
              : [],
        },
        harnessExpert: {
          expertId: harnessExpert?.expert_id ?? harnessExpert?.expertId ?? null,
          title: harnessExpert?.title ?? null,
          skillRefs: Array.isArray(harnessExpert?.skill_refs)
            ? harnessExpert.skill_refs
            : Array.isArray(harnessExpert?.skillRefs)
              ? harnessExpert.skillRefs
              : [],
        },
        recordedAt: entry.recordedAt,
      });
    }
    if (entry?.kind === "backendEmit") {
      return sanitizeJson({
        kind: entry.kind,
        sessionId: entry.sessionId,
        turnId: entry.turnId,
        eventCount: entry.eventCount,
        eventTypes: entry.eventTypes,
        recordedAt: entry.recordedAt,
      });
    }
    if (
      entry?.kind === "typedErrorRetryingAwaitingSignal" ||
      entry?.kind === "typedErrorAwaitingTerminalSignal" ||
      entry?.kind === "typedErrorTerminalEmitted"
    ) {
      return sanitizeJson({
        kind: entry.kind,
        scenario: entry.scenario,
        sessionId: entry.sessionId,
        threadId: entry.threadId,
        turnId: entry.turnId,
        eventType: entry.eventType,
        eventTypes: entry.eventTypes,
        retryCount: entry.retryCount,
        expectedWillRetry: entry.expectedWillRetry,
        retryable: entry.retryable,
        signalStage: entry.signalStage,
        terminalStatus: entry.terminalStatus,
        recordedAt: entry.recordedAt,
      });
    }
    if (
      entry?.kind === "actionRespond" ||
      entry?.kind === "approvalRequestResumeActionRespond" ||
      entry?.kind === "approvalRequestResumeActionRespondIgnored"
    ) {
      return sanitizeJson({
        kind: entry.kind,
        sessionId: entry.sessionId,
        threadId: entry.threadId,
        turnId: entry.turnId,
        requestId: entry.requestId,
        actionType: entry.actionType,
        decision: entry.decision,
        decisionScope: entry.decisionScope,
        confirmed: entry.confirmed,
        requestKeys: Array.isArray(entry.requestKeys) ? entry.requestKeys : [],
        actionScope: entry.actionScope,
        recordedAt: entry.recordedAt,
      });
    }
    return sanitizeJson({
      kind: entry?.kind ?? null,
      sessionId: entry?.sessionId ?? null,
      turnId: entry?.turnId ?? null,
      recordedAt: entry?.recordedAt ?? null,
    });
  });
}

export async function waitForBackendLedgerEntry(filePath, predicate, options) {
  const startedAt = Date.now();
  let lastLedger = [];
  const timeoutMs = Math.min(options.timeoutMs, 10_000);
  while (Date.now() - startedAt < timeoutMs) {
    lastLedger = readJsonl(filePath);
    const matched = lastLedger.find(predicate);
    if (matched) {
      return { entry: matched, ledger: lastLedger };
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `external backend ledger 未记录预期事件: ${JSON.stringify(
      sanitizeBackendLedgerForEvidence(lastLedger),
    )}`,
  );
}

export async function waitForBackendLedgerTurnStart(
  filePath,
  inputText,
  options,
) {
  return waitForBackendLedgerEntry(
    filePath,
    (entry) => entry.kind === "turnStart" && entry.inputText === inputText,
    options,
  );
}

export async function waitForBackendLedgerTurnStartContaining(
  filePath,
  inputText,
  options,
) {
  return waitForBackendLedgerEntry(
    filePath,
    (entry) =>
      entry.kind === "turnStart" &&
      String(entry.inputText || "").includes(inputText),
    options,
  );
}

export async function waitForBackendLedgerTurnStartOrNull(
  filePath,
  inputText,
  options,
) {
  const startedAt = Date.now();
  const timeoutMs = Math.min(options.timeoutMs, 10_000);
  while (Date.now() - startedAt < timeoutMs) {
    const ledger = readJsonl(filePath);
    const matched = ledger.find(
      (entry) => entry.kind === "turnStart" && entry.inputText === inputText,
    );
    if (matched) {
      return { entry: matched, ledger };
    }
    await sleep(options.intervalMs);
  }
  return null;
}

export function summarizeBackendLedger(backendLedger) {
  const turnStartEntries = backendLedger.filter(
    (entry) => entry.kind === "turnStart",
  );
  const turnCancelEntries = backendLedger.filter(
    (entry) => entry.kind === "turnCancel",
  );
  const backendEmitEntries = backendLedger.filter(
    (entry) => entry.kind === "backendEmit",
  );
  const latestTurnStart = turnStartEntries.at(-1) ?? null;
  const latestTurnCancel = turnCancelEntries.at(-1) ?? null;
  const latestTurnEmitEntries =
    latestTurnStart?.turnId == null
      ? []
      : backendEmitEntries.filter(
          (entry) => entry.turnId === latestTurnStart.turnId,
        );
  const latestTurnEmitTimes = latestTurnEmitEntries
    .map((entry) => Date.parse(entry.recordedAt))
    .filter((timestamp) => Number.isFinite(timestamp));
  const latestTurnEmitSpanMs =
    latestTurnEmitTimes.length >= 2
      ? Math.max(...latestTurnEmitTimes) - Math.min(...latestTurnEmitTimes)
      : 0;
  const runtimeRequest = latestTurnStart?.runtimeRequest ?? null;
  const collaborationMode = runtimeRequest?.collaborationMode?.mode ?? null;
  return {
    kinds: backendLedger.map((entry) => entry.kind),
    turnStartCount: turnStartEntries.length,
    turnCancelCount: turnCancelEntries.length,
    backendEmitCount: backendEmitEntries.length,
    latestTurnBackendEmitCount: latestTurnEmitEntries.length,
    latestTurnBackendEmitSpanMs: latestTurnEmitSpanMs,
    latestTurnBackendEmitTypes: latestTurnEmitEntries.map(
      (entry) => entry.eventTypes,
    ),
    latestTurnStart: latestTurnStart
      ? sanitizeJson({
          sessionId: latestTurnStart.sessionId,
          turnId: latestTurnStart.turnId,
          inputText: latestTurnStart.inputText,
          providerPreference: latestTurnStart.providerPreference,
          modelPreference: latestTurnStart.modelPreference,
          searchMode: runtimeRequest?.searchMode ?? null,
          webSearch: Object.prototype.hasOwnProperty.call(
            runtimeRequest || {},
            "webSearch",
          )
            ? runtimeRequest.webSearch
            : null,
          collaborationMode,
        })
      : null,
    latestTurnCancel: latestTurnCancel
      ? sanitizeJson({
          sessionId: latestTurnCancel.sessionId,
          turnId: latestTurnCancel.turnId,
        })
      : null,
  };
}

export function readHarnessMetadataFromTurnStart(turnStart) {
  const runtimeRequest = turnStart?.runtimeRequest ?? {};
  return runtimeRequest?.metadata?.harness ?? {};
}

export function readWorkspaceSkillRuntimeEnableFromTurnStart(turnStart) {
  const harness = readHarnessMetadataFromTurnStart(turnStart);
  return (
    harness?.workspace_skill_runtime_enable ??
    harness?.workspaceSkillRuntimeEnable ??
    null
  );
}

export function readObjectiveTextFromHarness(harness) {
  return (
    harness?.thread_goal?.set?.objective ??
    harness?.threadGoal?.set?.objective ??
    harness?.goal?.set?.objective ??
    null
  );
}
