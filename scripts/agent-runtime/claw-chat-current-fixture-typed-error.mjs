import fs from "node:fs";
import {
  TYPED_ERROR_RETRY_FAILURE_ERROR_TEXT,
  TYPED_ERROR_RETRY_FAILURE_PARTIAL_TEXT,
  TYPED_ERROR_RETRY_FAILURE_PROMPT,
  TYPED_ERROR_RETRY_FAILURE_SCENARIO,
  TYPED_ERROR_RETRY_SUCCESS_DONE_TEXT,
  TYPED_ERROR_RETRY_SUCCESS_PROMPT,
  TYPED_ERROR_RETRY_SUCCESS_SCENARIO,
  TYPED_ERROR_RETRY_SUCCESS_TEXT,
} from "./claw-chat-current-fixture-constants.mjs";
import { sendPromptFromGui } from "./claw-chat-current-fixture-gui-actions.mjs";
import { waitForGuiChatCompleted } from "./claw-chat-current-fixture-gui-completion-waits.mjs";
import {
  waitForSessionReadCompleted,
  waitForSessionReadFailedAfterAnswer,
} from "./claw-chat-current-fixture-read-model-waits.mjs";
import { readModelLatestTurnStatus } from "./claw-chat-current-fixture-read-model-core.mjs";
import {
  evaluatePageSnapshot,
  invokeAppServerFromPage,
} from "./claw-chat-current-fixture-rpc.mjs";
import { waitForBackendLedgerEntry } from "./claw-chat-current-fixture-backend-ledger.mjs";
import { sanitizeJson, sleep } from "./claw-chat-current-fixture-utils.mjs";

function appendSignal(signalPath, stage) {
  fs.appendFileSync(
    signalPath,
    `${JSON.stringify({ stage, recordedAt: new Date().toISOString() })}\n`,
  );
}

function typedErrorStatusSnapshotFromDom({ prompt, phase }) {
  const text = document.body?.innerText || "";
  const turnGroups = Array.from(
    document.querySelectorAll('[data-testid="message-turn-group"]'),
  );
  const group =
    [...turnGroups]
      .reverse()
      .find((candidate) => (candidate.innerText || "").includes(prompt)) ??
    null;
  const statusNodes = Array.from(
    document.querySelectorAll(
      [
        '[data-testid="message-runtime-status-pill"]',
        '[data-testid="assistant-first-token-runtime-status"]',
        '[data-testid="inputbar-runtime-status-line"]',
      ].join(","),
    ),
  );
  const statusEntries = statusNodes.map((node) => ({
    testId: node.getAttribute("data-testid") || "",
    text: node.textContent || "",
    ariaLabel: node.getAttribute("aria-label") || "",
    status: node.getAttribute("data-status") || "",
  }));
  const statusText = statusEntries
    .map((entry) => [entry.text, entry.ariaLabel, entry.status].join(" "))
    .join("\n");
  const hasExpectedPhase = statusEntries.some(
    (entry) => entry.status === phase,
  );
  const textarea = document.querySelector(
    'textarea[name="agent-chat-message"]',
  );
  const textareaRect = textarea?.getBoundingClientRect();
  const textareaStyle = textarea ? window.getComputedStyle(textarea) : null;
  const textareaVisible = Boolean(
    textarea &&
    textareaRect &&
    textareaRect.width > 16 &&
    textareaRect.height > 16 &&
    textareaStyle?.visibility !== "hidden" &&
    textareaStyle?.display !== "none",
  );
  const stopButtonVisible = Array.from(
    document.querySelectorAll("button"),
  ).some((button) => {
    const label = [
      button.getAttribute("title") || "",
      button.textContent || "",
      button.getAttribute("aria-label") || "",
    ].join(" ");
    return (
      !button.disabled &&
      (label.includes("停止") ||
        label.includes("终止") ||
        /\bStop\b/i.test(label))
    );
  });
  return {
    url: window.location.href,
    hasPrompt: text.includes(prompt),
    hasExpectedPhase,
    statusEntries,
    statusText,
    stopButtonVisible,
    textareaVisible,
    textareaDisabled:
      textarea instanceof HTMLTextAreaElement ? textarea.disabled : null,
    bodyText: text,
    scopedText: group?.innerText || text,
  };
}

async function waitForTypedErrorGuiStatus(page, options, { prompt, phase }) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(
      page,
      typedErrorStatusSnapshotFromDom,
      { prompt, phase },
    );
    lastSnapshot = snapshot;
    if (snapshot?.hasPrompt && snapshot.hasExpectedPhase) {
      return sanitizeJson(snapshot);
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `Claw GUI 未观察到 typed error 状态: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

async function waitForReadModelTypedErrorPending(
  page,
  options,
  requestLog,
  { prompt },
) {
  const startedAt = Date.now();
  let lastRead = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const response = await invokeAppServerFromPage(
      page,
      "thread/read",
      { threadId: options.threadId, includeTurns: true },
      requestLog,
    );
    const read = response.result;
    lastRead = read;
    const serialized = JSON.stringify(read || {});
    const status = String(readModelLatestTurnStatus(read) || "").toLowerCase();
    if (
      serialized.includes(prompt) &&
      !["completed", "failed", "interrupted", "canceled", "cancelled"].includes(
        status,
      )
    ) {
      return sanitizeJson({
        latestTurnStatus: readModelLatestTurnStatus(read),
        includesPrompt: true,
        terminal: false,
      });
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `App Server read model 在 typed error 后提前进入终态或未保留输入: ${JSON.stringify(
      sanitizeJson(lastRead),
    )}`,
  );
}

export function buildTypedErrorScenarioAssertions({
  isTypedErrorRetryFailureScenario,
  isTypedErrorRetrySuccessScenario,
  summary,
  typedErrorTurnStart,
}) {
  const retryBackend = summary.typedErrorRetryingBackend ?? {};
  const terminalBackend = summary.typedErrorTerminalBackend ?? {};
  const retryGui = summary.typedErrorRetryingGui ?? {};
  const awaitingGui = summary.typedErrorAwaitingTerminalGui ?? {};
  const terminalGui = summary.guiTypedErrorTerminal ?? {};
  const terminalRead = summary.readModelTypedErrorTerminal ?? {};
  const pendingRead = summary.readModelTypedErrorPending ?? {};
  const retryTypes = retryBackend.eventTypes ?? [];
  const expectedPrompt = isTypedErrorRetrySuccessScenario
    ? TYPED_ERROR_RETRY_SUCCESS_PROMPT
    : TYPED_ERROR_RETRY_FAILURE_PROMPT;
  return {
    typedErrorRetryPromptReachedBackend:
      typedErrorTurnStart?.inputText === expectedPrompt &&
      typeof typedErrorTurnStart?.sessionId === "string" &&
      typeof typedErrorTurnStart?.turnId === "string",
    typedErrorRetryBackendEventOrder:
      retryTypes.length === (isTypedErrorRetryFailureScenario ? 2 : 1) &&
      retryTypes.every((eventType) => eventType === "plugin_worker.retry") &&
      retryBackend.expectedWillRetry === true,
    typedErrorRetryGuiRetryingStatusVisible:
      retryGui.hasPrompt === true && retryGui.hasExpectedPhase === true,
    typedErrorRetryNoPrematureTerminal:
      retryGui.stopButtonVisible === true &&
      retryGui.textareaDisabled === false,
    typedErrorRetrySuccessGuiCompleted: isTypedErrorRetrySuccessScenario
      ? terminalGui.hasPrompt === true &&
        terminalGui.bodyText?.includes(TYPED_ERROR_RETRY_SUCCESS_TEXT) ===
          true &&
        terminalGui.bodyText?.includes(TYPED_ERROR_RETRY_SUCCESS_DONE_TEXT) ===
          true &&
        terminalGui.stopButtonVisible === false &&
        terminalGui.textareaDisabled === false
      : true,
    typedErrorRetrySuccessReadModelCompleted: isTypedErrorRetrySuccessScenario
      ? terminalRead.latestTurnStatus === "completed" &&
        terminalRead.includesPrompt === true &&
        terminalRead.includesSuccessText === true
      : true,
    typedErrorRetryFailureGuiAwaitedTerminal: isTypedErrorRetryFailureScenario
      ? awaitingGui.hasPrompt === true &&
        awaitingGui.hasExpectedPhase === true &&
        awaitingGui.stopButtonVisible === true
      : true,
    typedErrorRetryFailureGuiCompleted: isTypedErrorRetryFailureScenario
      ? terminalGui.hasPrompt === true &&
        terminalGui.bodyText?.includes(
          TYPED_ERROR_RETRY_FAILURE_PARTIAL_TEXT,
        ) === true &&
        terminalGui.stopButtonVisible === false &&
        terminalGui.textareaDisabled === false
      : true,
    typedErrorRetryFailureReadModelFailed: isTypedErrorRetryFailureScenario
      ? terminalRead.latestTurnStatus === "failed" &&
        terminalRead.includesPrompt === true &&
        terminalRead.includesPartialText === true &&
        terminalRead.includesFailureText === true
      : true,
    typedErrorRetryFailureNoPrematureTerminal: isTypedErrorRetryFailureScenario
      ? terminalBackend.eventType === "turn.failed" &&
        summary.typedErrorAwaitingTerminalBackend?.eventType ===
          "runtime.error" &&
        summary.typedErrorAwaitingTerminalBackend?.expectedWillRetry === false &&
        pendingRead.includesPrompt === true &&
        pendingRead.terminal === false
      : true,
    typedErrorRetryIdentityConsistent:
      typedErrorTurnStart?.sessionId === summary.sessionId &&
      typedErrorTurnStart?.turnId === terminalBackend.turnId &&
      terminalGui.completionScope?.runtimeTurnId ===
        typedErrorTurnStart?.turnId,
  };
}

export async function runTypedErrorScenario({
  page,
  options,
  appServerRequests,
  runtimeEnv,
  logStage,
}) {
  const isSuccess = options.scenario === TYPED_ERROR_RETRY_SUCCESS_SCENARIO;
  const prompt = isSuccess
    ? TYPED_ERROR_RETRY_SUCCESS_PROMPT
    : TYPED_ERROR_RETRY_FAILURE_PROMPT;
  const result = {};

  logStage("send-typed-error-prompt-from-gui");
  result.typedErrorRetryInputSend = sanitizeJson(
    await sendPromptFromGui(page, options, prompt),
  );

  logStage("wait-typed-error-retrying-backend");
  const retryingBackend = await waitForBackendLedgerEntry(
    runtimeEnv.backendLedgerPath,
    (entry) => entry.kind === "typedErrorRetryingAwaitingSignal",
    options,
  );
  result.typedErrorRetryingBackend = sanitizeJson(retryingBackend.entry);
  result.typedErrorTurnStart = {
    sessionId: retryingBackend.entry.sessionId,
    turnId: retryingBackend.entry.turnId,
    inputText: prompt,
    requireBackendTurn: true,
  };

  logStage("wait-typed-error-retrying-gui");
  result.typedErrorRetryingGui = sanitizeJson(
    await waitForTypedErrorGuiStatus(page, options, {
      prompt,
      phase: "retrying",
    }),
  );
  appendSignal(runtimeEnv.typedErrorSignalPath, "retry-visible");

  if (isSuccess) {
    logStage("wait-typed-error-success-gui-completed");
    result.guiTypedErrorTerminal = sanitizeJson(
      await waitForGuiChatCompleted(page, options, {
        prompt,
        summaryText: TYPED_ERROR_RETRY_SUCCESS_TEXT,
        doneText: TYPED_ERROR_RETRY_SUCCESS_DONE_TEXT,
      }),
    );
    logStage("wait-typed-error-success-read-model-completed");
    const readModel = await waitForSessionReadCompleted(
      page,
      options,
      appServerRequests,
      {
        prompt,
        summaryText: TYPED_ERROR_RETRY_SUCCESS_TEXT,
        doneText: TYPED_ERROR_RETRY_SUCCESS_DONE_TEXT,
      },
    );
    const serialized = JSON.stringify(readModel || {});
    result.readModelTypedErrorTerminal = sanitizeJson({
      latestTurnStatus: readModelLatestTurnStatus(readModel),
      includesPrompt: serialized.includes(prompt),
      includesSuccessText: serialized.includes(TYPED_ERROR_RETRY_SUCCESS_TEXT),
      includesDoneText: serialized.includes(
        TYPED_ERROR_RETRY_SUCCESS_DONE_TEXT,
      ),
    });
  } else {
    logStage("wait-typed-error-awaiting-terminal-backend");
    const awaitingTerminal = await waitForBackendLedgerEntry(
      runtimeEnv.backendLedgerPath,
      (entry) => entry.kind === "typedErrorAwaitingTerminalSignal",
      options,
    );
    result.typedErrorAwaitingTerminalBackend = sanitizeJson(
      awaitingTerminal.entry,
    );
    logStage("wait-typed-error-awaiting-terminal-gui");
    result.typedErrorAwaitingTerminalGui = sanitizeJson(
      await waitForTypedErrorGuiStatus(page, options, {
        prompt,
        phase: "failed",
      }),
    );
    result.readModelTypedErrorPending =
      await waitForReadModelTypedErrorPending(page, options, appServerRequests, {
        prompt,
      });
    appendSignal(runtimeEnv.typedErrorSignalPath, "terminal-visible");

    logStage("wait-typed-error-failure-gui-completed");
    result.guiTypedErrorTerminal = sanitizeJson(
      await waitForGuiChatCompleted(page, options, {
        prompt,
        summaryText: TYPED_ERROR_RETRY_FAILURE_PARTIAL_TEXT,
        dedupeGuardTexts: [TYPED_ERROR_RETRY_FAILURE_PARTIAL_TEXT],
      }),
    );
    logStage("wait-typed-error-failure-read-model-failed");
    const readModel = await waitForSessionReadFailedAfterAnswer(
      page,
      options,
      appServerRequests,
      {
        prompt,
        partialText: TYPED_ERROR_RETRY_FAILURE_PARTIAL_TEXT,
        failureText: TYPED_ERROR_RETRY_FAILURE_ERROR_TEXT,
      },
    );
    const serialized = JSON.stringify(readModel || {});
    result.readModelTypedErrorTerminal = sanitizeJson({
      latestTurnStatus: readModelLatestTurnStatus(readModel),
      includesPrompt: serialized.includes(prompt),
      includesPartialText: serialized.includes(
        TYPED_ERROR_RETRY_FAILURE_PARTIAL_TEXT,
      ),
      includesFailureText: serialized.includes(
        TYPED_ERROR_RETRY_FAILURE_ERROR_TEXT,
      ),
    });
  }

  logStage("wait-typed-error-terminal-backend");
  const terminal = await waitForBackendLedgerEntry(
    runtimeEnv.backendLedgerPath,
    (entry) => entry.kind === "typedErrorTerminalEmitted",
    options,
  );
  result.typedErrorTerminalBackend = sanitizeJson(terminal.entry);
  return result;
}
