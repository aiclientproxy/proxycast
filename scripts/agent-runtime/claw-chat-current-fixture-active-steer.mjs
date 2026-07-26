import { APP_SERVER_METHOD_SESSION_READ } from "./claw-chat-current-fixture-constants.mjs";
import { sendPromptFromGui } from "./claw-chat-current-fixture-gui-actions.mjs";
import {
  collectReadModelItems,
  collectReadModelTurns,
  isReadModelTerminalTurnStatus,
  readModelLatestTurnStatus,
  readModelScopedTurnId,
  readModelTurnId,
  readModelTurnStatus,
  summarizeReadModelQueueState,
} from "./claw-chat-current-fixture-read-model-core.mjs";
import {
  decodeJsonRpcLines,
  evaluatePageSnapshot,
  invokeAppServerFromPage,
} from "./claw-chat-current-fixture-rpc.mjs";
import { waitForSessionReadCompleted } from "./claw-chat-current-fixture-read-model-waits.mjs";
import {
  assert,
  sanitizeJson,
  sleep,
} from "./claw-chat-current-fixture-utils.mjs";

export const ACTIVE_STEER_SCENARIO = "inputbar-active-steer";
export const ACTIVE_STEER_INITIAL_PROMPT =
  "验证 active turn steer：先保持当前回合运行";
export const ACTIVE_STEER_INPUT = "请在同一回合中转向并确认 steer 已生效";
export const ACTIVE_STEER_FIRST_TEXT =
  "ACTIVE_STEER_FIRST_VISIBLE: 当前回合仍在运行。";
export const ACTIVE_STEER_FINAL_TEXT =
  "ACTIVE_STEER_APPLIED: 已在同一回合接收 steer。";
export const ACTIVE_STEER_DONE_TEXT = "ACTIVE_STEER_DONE";

export const ACTIVE_STEER_ASSERTION_KEYS = [
  "activeSteerUsedThreadReadBeforeSteer",
  "activeSteerExpectedTurnIdMatched",
  "activeSteerTypedInputReachedJsonRpc",
  "activeSteerResponsePreservedTurnIdentity",
  "activeSteerProviderReceivedSecondStep",
  "activeSteerSingleTurnInReadModel",
  "activeSteerSingleTurnInGui",
  "activeSteerNoSecondTurnStart",
  "activeSteerNoPublicQueueMethod",
  "activeSteerNoQueuedTurnGui",
];

const PUBLIC_QUEUE_METHODS = new Set([
  "agentSession/queuedTurn/promote",
  "agentSession/queuedTurn/remove",
  "turn/queue/promote",
]);

function readInputText(input) {
  if (typeof input === "string") {
    return input;
  }
  if (Array.isArray(input)) {
    return input
      .map((part) => readInputText(part))
      .filter(Boolean)
      .join("\n");
  }
  if (!input || typeof input !== "object") {
    return "";
  }
  if (typeof input.text === "string") {
    return input.text;
  }
  if (typeof input.content === "string") {
    return input.content;
  }
  if (typeof input.displayContent === "string") {
    return input.displayContent;
  }
  if (input.Text && typeof input.Text.text === "string") {
    return input.Text.text;
  }
  return readInputText(input.parts);
}

async function readInvokeTrace(page) {
  return await page.evaluate(() => {
    try {
      const parsed = JSON.parse(
        window.localStorage.getItem("lime_invoke_trace_buffer_v1") || "[]",
      );
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  });
}

function flattenTraceRequests(traceEntries) {
  return traceEntries.flatMap((entry, entryIndex) =>
    decodeJsonRpcLines(entry?.args_preview?.request?.lines).map((message) => ({
      entryIndex,
      timestamp: entry?.timestamp ?? null,
      timestampMs: Date.parse(entry?.timestamp ?? ""),
      transport: entry?.transport ?? null,
      status: entry?.status ?? null,
      method: message?.method ?? null,
      params: message?.params ?? {},
    })),
  );
}

async function waitForInitialActiveTurn({ page, options, appServerRequests }) {
  const startedAt = Date.now();
  let lastSummary = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const read = await invokeAppServerFromPage(
      page,
      APP_SERVER_METHOD_SESSION_READ,
      { threadId: options.threadId, includeTurns: true },
      appServerRequests,
    );
    const turns = collectReadModelTurns(read.result);
    const queueState = summarizeReadModelQueueState(read.result);
    const activeTurn = [...turns]
      .reverse()
      .find(
        (turn) => !isReadModelTerminalTurnStatus(readModelTurnStatus(turn)),
      );
    const activeTurnId =
      queueState.activeTurnId ?? readModelTurnId(activeTurn) ?? null;
    const serialized = JSON.stringify(read.result ?? {});
    lastSummary = sanitizeJson({
      ...queueState,
      includesInitialPrompt: serialized.includes(ACTIVE_STEER_INITIAL_PROMPT),
      includesFirstText: serialized.includes(ACTIVE_STEER_FIRST_TEXT),
    });
    if (
      activeTurnId &&
      lastSummary.includesInitialPrompt &&
      lastSummary.includesFirstText
    ) {
      return {
        turnId: activeTurnId,
        summary: lastSummary,
      };
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `active steer 首段未进入 running read model: ${JSON.stringify(lastSummary)}`,
  );
}

async function waitForActiveSteerFirstVisible(page, options) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(page, () => {
      const textarea = document.querySelector(
        'textarea[name="agent-chat-message"]',
      );
      const inputbar = textarea?.closest(
        '[data-testid="inputbar-core-container"]',
      );
      const sendButton = inputbar?.querySelector('[data-testid="send-btn"]');
      const buttons = Array.from(document.querySelectorAll("button"));
      const stopButtonVisible = buttons.some((button) => {
        const label = [
          button.getAttribute("title") || "",
          button.getAttribute("aria-label") || "",
          button.textContent || "",
        ].join("\n");
        return (
          !button.disabled &&
          (label.includes("停止") ||
            label.includes("终止") ||
            /\bStop\b/iu.test(label))
        );
      });
      const bodyText = document.body?.innerText || "";
      return {
        hasInitialPrompt: bodyText.includes(
          "验证 active turn steer：先保持当前回合运行",
        ),
        hasFirstText: bodyText.includes(
          "ACTIVE_STEER_FIRST_VISIBLE: 当前回合仍在运行。",
        ),
        stopButtonVisible,
        textareaVisible:
          textarea instanceof HTMLTextAreaElement &&
          textarea.getClientRects().length > 0,
        textareaDisabled:
          textarea instanceof HTMLTextAreaElement ? textarea.disabled : null,
        sendButtonDisabled:
          sendButton instanceof HTMLButtonElement ? sendButton.disabled : null,
      };
    });
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;
    if (
      snapshot.hasInitialPrompt &&
      snapshot.hasFirstText &&
      snapshot.stopButtonVisible &&
      snapshot.textareaVisible &&
      snapshot.textareaDisabled === false
    ) {
      return sanitizeJson(snapshot);
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `active steer 首段未在 GUI 保持运行: ${JSON.stringify(lastSnapshot)}`,
  );
}

async function waitForSteerTrace({
  page,
  options,
  baselineTimestampMs,
  activeTurnId,
}) {
  const startedAt = Date.now();
  let lastEvidence = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const traceEntries = await readInvokeTrace(page);
    const requests = flattenTraceRequests(traceEntries).filter(
      (request) =>
        Number.isFinite(request.timestampMs) &&
        request.timestampMs >= baselineTimestampMs,
    );
    const steer = [...requests]
      .reverse()
      .find(
        (request) =>
          request.method === "turn/steer" &&
          request.params?.threadId === options.threadId &&
          readInputText(request.params?.input).includes(ACTIVE_STEER_INPUT),
      );
    const precedingThreadRead = steer
      ? requests.find(
          (request) =>
            request.entryIndex < steer.entryIndex &&
            request.method === "thread/read" &&
            request.params?.threadId === options.threadId,
        )
      : null;
    const turnStartAfterBaselineCount = requests.filter(
      (request) => request.method === "turn/start",
    ).length;
    const publicQueueMethodHits = requests
      .map((request) => request.method)
      .filter((method) => PUBLIC_QUEUE_METHODS.has(method));
    lastEvidence = sanitizeJson({
      baselineTimestamp: new Date(baselineTimestampMs).toISOString(),
      traceCount: traceEntries.length,
      threadReadBeforeSteer: Boolean(precedingThreadRead),
      turnStartAfterBaselineCount,
      publicQueueMethodHits,
      steer: steer
        ? {
            transport: steer.transport,
            status: steer.status,
            threadId: steer.params?.threadId ?? null,
            expectedTurnId: steer.params?.expectedTurnId ?? null,
            inputText: readInputText(steer.params?.input),
            inputIsTyped: Array.isArray(steer.params?.input),
          }
        : null,
    });
    if (
      precedingThreadRead &&
      steer?.transport === "electron-ipc" &&
      steer?.status === "success" &&
      steer?.params?.expectedTurnId === activeTurnId
    ) {
      return lastEvidence;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `未观察到 renderer thread/read -> turn/steer: ${JSON.stringify(lastEvidence)}`,
  );
}

async function waitForProviderSecondStep(readTextProviderRequests, options) {
  const startedAt = Date.now();
  let lastSummary = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const requests = readTextProviderRequests();
    const chatRequests = requests.filter(
      (request) =>
        request?.method === "POST" &&
        ["/chat/completions", "/v1/chat/completions"].includes(request?.url),
    );
    lastSummary = sanitizeJson({
      chatRequestCount: chatRequests.length,
      initialRequestObserved: chatRequests.some(
        (request) =>
          request?.bodySummary?.bodyIncludesActiveSteerInitialPrompt === true,
      ),
      steerRequestObserved: chatRequests.some(
        (request) =>
          request?.bodySummary?.bodyIncludesActiveSteerInput === true,
      ),
    });
    if (
      lastSummary.chatRequestCount >= 2 &&
      lastSummary.initialRequestObserved &&
      lastSummary.steerRequestObserved
    ) {
      return lastSummary;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `Provider 未收到 active steer 第二步请求: ${JSON.stringify(lastSummary)}`,
  );
}

async function collectFinalGuiIdentity(page, activeTurnId) {
  return sanitizeJson(
    await page.evaluate(
      ({ activeTurnId, markers }) => {
        const groups = Array.from(
          document.querySelectorAll('[data-testid="message-turn-group"]'),
        );
        const scenarioGroups = groups.filter((group) =>
          markers.some((marker) => (group.innerText || "").includes(marker)),
        );
        const scenarioRuntimeTurnIds = Array.from(
          new Set(
            scenarioGroups
              .map((group) => group.getAttribute("data-runtime-turn-id"))
              .filter(Boolean),
          ),
        );
        const queuedTurnNodes = Array.from(
          document.querySelectorAll(
            [
              '[data-testid*="queued-turn"]',
              '[data-testid*="pending-turn"]',
              '[data-testid*="QueuedTurn"]',
              '[data-testid*="PendingTurn"]',
            ].join(","),
          ),
        );
        const textarea = document.querySelector(
          'textarea[name="agent-chat-message"]',
        );
        const stopButtonVisible = Array.from(
          document.querySelectorAll("button"),
        ).some((button) => {
          const label = [
            button.getAttribute("title") || "",
            button.getAttribute("aria-label") || "",
            button.textContent || "",
          ].join("\n");
          return (
            !button.disabled &&
            (label.includes("停止") ||
              label.includes("终止") ||
              /\bStop\b/iu.test(label))
          );
        });
        return {
          activeTurnId,
          scenarioGroupCount: scenarioGroups.length,
          scenarioRuntimeTurnIds,
          includesInitialPrompt: scenarioGroups.some((group) =>
            (group.innerText || "").includes(markers[0]),
          ),
          includesSteerInput: scenarioGroups.some((group) =>
            (group.innerText || "").includes(markers[1]),
          ),
          includesFirstText: scenarioGroups.some((group) =>
            (group.innerText || "").includes(markers[2]),
          ),
          includesFinalText: scenarioGroups.some((group) =>
            (group.innerText || "").includes(markers[3]),
          ),
          includesDoneText: scenarioGroups.some((group) =>
            (group.innerText || "").includes(markers[4]),
          ),
          hasCompletedGroup: scenarioGroups.some(
            (group) =>
              group.getAttribute("data-runtime-turn-status") === "completed",
          ),
          scenarioGroups: scenarioGroups.map((group) => ({
            runtimeTurnId: group.getAttribute("data-runtime-turn-id") || null,
            runtimeTurnStatus:
              group.getAttribute("data-runtime-turn-status") || null,
            includesInitialPrompt: (group.innerText || "").includes(markers[0]),
            includesSteerInput: (group.innerText || "").includes(markers[1]),
            includesFirstText: (group.innerText || "").includes(markers[2]),
            includesFinalText: (group.innerText || "").includes(markers[3]),
            includesDoneText: (group.innerText || "").includes(markers[4]),
          })),
          queuedTurnGuiCount: queuedTurnNodes.length,
          textareaVisible:
            textarea instanceof HTMLTextAreaElement &&
            textarea.getClientRects().length > 0,
          textareaDisabled:
            textarea instanceof HTMLTextAreaElement ? textarea.disabled : null,
          stopButtonVisible,
        };
      },
      {
        activeTurnId,
        markers: [
          ACTIVE_STEER_INITIAL_PROMPT,
          ACTIVE_STEER_INPUT,
          ACTIVE_STEER_FIRST_TEXT,
          ACTIVE_STEER_FINAL_TEXT,
          ACTIVE_STEER_DONE_TEXT,
        ],
      },
    ),
  );
}

async function waitForActiveSteerGuiCompleted(page, options, activeTurnId) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await collectFinalGuiIdentity(page, activeTurnId);
    lastSnapshot = snapshot;
    if (
      snapshot?.scenarioRuntimeTurnIds?.length === 1 &&
      snapshot.scenarioRuntimeTurnIds[0] === activeTurnId &&
      snapshot.includesInitialPrompt === true &&
      snapshot.includesSteerInput === true &&
      snapshot.includesFirstText === true &&
      snapshot.includesFinalText === true &&
      snapshot.includesDoneText === true &&
      snapshot.hasCompletedGroup === true &&
      snapshot.textareaVisible === true &&
      snapshot.textareaDisabled === false &&
      snapshot.stopButtonVisible === false
    ) {
      return {
        gui: snapshot,
        guiCompleted: sanitizeJson({
          hasPrompt: true,
          hasAssistantSummary: true,
          hasDoneText: true,
          textareaVisible: snapshot.textareaVisible,
          textareaDisabled: snapshot.textareaDisabled,
          stopButtonVisible: snapshot.stopButtonVisible,
          completionScope: {
            foundTurnGroup: true,
            runtimeTurnId: activeTurnId,
            assistantRuntimeTurnId: activeTurnId,
          },
        }),
      };
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `active steer GUI 未按同一 Turn identity 收敛: ${JSON.stringify(lastSnapshot)}`,
  );
}

function summarizeFinalReadModel(readModel, activeTurnId) {
  const serialized = JSON.stringify(readModel ?? {});
  const turns = collectReadModelTurns(readModel);
  const items = collectReadModelItems(readModel);
  const turnIds = Array.from(
    new Set(turns.map((turn) => readModelTurnId(turn)).filter(Boolean)),
  );
  const relevantItemTurnIds = Array.from(
    new Set(
      items
        .filter((item) =>
          [
            ACTIVE_STEER_INITIAL_PROMPT,
            ACTIVE_STEER_INPUT,
            ACTIVE_STEER_FIRST_TEXT,
            ACTIVE_STEER_FINAL_TEXT,
            ACTIVE_STEER_DONE_TEXT,
          ].some((marker) => JSON.stringify(item ?? {}).includes(marker)),
        )
        .map((item) => readModelScopedTurnId(item))
        .filter(Boolean),
    ),
  );
  return sanitizeJson({
    activeTurnId,
    latestTurnStatus: readModelLatestTurnStatus(readModel),
    turnIds,
    turnStatuses: turns.map((turn) => ({
      turnId: readModelTurnId(turn),
      status: readModelTurnStatus(turn),
    })),
    relevantItemTurnIds,
    includesInitialPrompt: serialized.includes(ACTIVE_STEER_INITIAL_PROMPT),
    includesSteerInput: serialized.includes(ACTIVE_STEER_INPUT),
    includesFirstText: serialized.includes(ACTIVE_STEER_FIRST_TEXT),
    includesFinalText: serialized.includes(ACTIVE_STEER_FINAL_TEXT),
    includesDoneText: serialized.includes(ACTIVE_STEER_DONE_TEXT),
    includesTurnSteerSource: serialized.includes("turn/steer"),
  });
}

function summarizeScenarioTrace(traceEntries) {
  const requests = flattenTraceRequests(traceEntries);
  const scenarioTurnStarts = requests.filter((request) => {
    if (request.method !== "turn/start") {
      return false;
    }
    const inputText = readInputText(request.params?.input);
    return (
      inputText.includes(ACTIVE_STEER_INITIAL_PROMPT) ||
      inputText.includes(ACTIVE_STEER_INPUT)
    );
  });
  const publicQueueMethodHits = requests
    .map((request) => request.method)
    .filter((method) => PUBLIC_QUEUE_METHODS.has(method));
  return sanitizeJson({
    scenarioTurnStartCount: scenarioTurnStarts.length,
    initialTurnStartCount: scenarioTurnStarts.filter((request) =>
      readInputText(request.params?.input).includes(
        ACTIVE_STEER_INITIAL_PROMPT,
      ),
    ).length,
    steerInputTurnStartCount: scenarioTurnStarts.filter((request) =>
      readInputText(request.params?.input).includes(ACTIVE_STEER_INPUT),
    ).length,
    publicQueueMethodHits,
  });
}

export async function runActiveSteerScenario({
  page,
  options,
  summary,
  appServerRequests,
  readTextProviderRequests,
}) {
  assert(
    typeof readTextProviderRequests === "function",
    "active steer 场景缺少文本 Provider request reader",
  );

  const inputSend = sanitizeJson(
    await sendPromptFromGui(page, options, ACTIVE_STEER_INITIAL_PROMPT, {
      expectedSessionId: summary.sessionId,
    }),
  );
  const firstVisible = await waitForActiveSteerFirstVisible(page, options);
  const activeRead = await waitForInitialActiveTurn({
    page,
    options,
    appServerRequests,
  });
  const initialTrace = summarizeScenarioTrace(await readInvokeTrace(page));
  const baselineTimestampMs = Date.now();

  const steerInputSend = sanitizeJson(
    await sendPromptFromGui(page, options, ACTIVE_STEER_INPUT, {
      expectedSessionId: summary.sessionId,
      submitWithEnter: true,
    }),
  );
  const steerTrace = await waitForSteerTrace({
    page,
    options,
    baselineTimestampMs,
    activeTurnId: activeRead.turnId,
  });
  const provider = await waitForProviderSecondStep(
    readTextProviderRequests,
    options,
  );
  const guiCompletion = await waitForActiveSteerGuiCompleted(
    page,
    options,
    activeRead.turnId,
  );
  const guiCompleted = guiCompletion.guiCompleted;
  const finalReadModel = await waitForSessionReadCompleted(
    page,
    options,
    appServerRequests,
    {
      prompt: ACTIVE_STEER_INITIAL_PROMPT,
      doneText: ACTIVE_STEER_DONE_TEXT,
      summaryText: ACTIVE_STEER_FINAL_TEXT,
    },
  );
  const readModel = summarizeFinalReadModel(finalReadModel, activeRead.turnId);
  const gui = guiCompletion.gui;
  const trace = sanitizeJson({
    ...initialTrace,
    postBaselineTurnStartCount: steerTrace.turnStartAfterBaselineCount,
    publicQueueMethodHits: Array.from(
      new Set([
        ...(initialTrace.publicQueueMethodHits ?? []),
        ...(steerTrace.publicQueueMethodHits ?? []),
      ]),
    ),
  });

  return {
    guiCompleted,
    readModelCompleted: {
      latestTurnStatus: readModel.latestTurnStatus,
      includesPrompt: readModel.includesInitialPrompt,
      includesAssistantDone: readModel.includesDoneText,
      includesAssistantSummary: readModel.includesFinalText,
    },
    activeSteer: sanitizeJson({
      activeTurnId: activeRead.turnId,
      inputSend,
      firstVisible,
      activeRead: activeRead.summary,
      steerInputSend,
      steerTrace,
      provider,
      gui,
      readModel,
      trace,
    }),
  };
}

export function buildActiveSteerScenarioAssertions({ summary }) {
  const evidence = summary.activeSteer ?? {};
  const steer = evidence.steerTrace?.steer ?? {};
  return {
    activeSteerUsedThreadReadBeforeSteer:
      evidence.steerTrace?.threadReadBeforeSteer === true,
    activeSteerExpectedTurnIdMatched:
      typeof evidence.activeTurnId === "string" &&
      evidence.activeTurnId.length > 0 &&
      steer.expectedTurnId === evidence.activeTurnId,
    activeSteerTypedInputReachedJsonRpc:
      steer.transport === "electron-ipc" &&
      steer.status === "success" &&
      steer.inputIsTyped === true &&
      steer.inputText?.includes(ACTIVE_STEER_INPUT) === true,
    activeSteerResponsePreservedTurnIdentity:
      summary.guiCompleted?.completionScope?.runtimeTurnId ===
        evidence.activeTurnId &&
      summary.guiCompleted?.completionScope?.assistantRuntimeTurnId ===
        evidence.activeTurnId,
    activeSteerProviderReceivedSecondStep:
      evidence.provider?.chatRequestCount >= 2 &&
      evidence.provider?.initialRequestObserved === true &&
      evidence.provider?.steerRequestObserved === true,
    activeSteerSingleTurnInReadModel:
      evidence.readModel?.turnIds?.length === 1 &&
      evidence.readModel?.turnIds?.[0] === evidence.activeTurnId &&
      evidence.readModel?.includesInitialPrompt === true &&
      evidence.readModel?.includesSteerInput === true &&
      evidence.readModel?.includesFirstText === true &&
      evidence.readModel?.includesFinalText === true &&
      evidence.readModel?.includesDoneText === true,
    activeSteerSingleTurnInGui:
      evidence.gui?.scenarioRuntimeTurnIds?.length === 1 &&
      evidence.gui?.scenarioRuntimeTurnIds?.[0] === evidence.activeTurnId &&
      evidence.gui?.includesInitialPrompt === true &&
      evidence.gui?.includesSteerInput === true &&
      evidence.gui?.includesFirstText === true &&
      evidence.gui?.includesFinalText === true &&
      evidence.gui?.includesDoneText === true &&
      evidence.gui?.textareaVisible === true &&
      evidence.gui?.textareaDisabled === false &&
      evidence.gui?.stopButtonVisible === false,
    activeSteerNoSecondTurnStart:
      evidence.trace?.scenarioTurnStartCount === 1 &&
      evidence.trace?.initialTurnStartCount === 1 &&
      evidence.trace?.steerInputTurnStartCount === 0 &&
      evidence.trace?.postBaselineTurnStartCount === 0,
    activeSteerNoPublicQueueMethod:
      evidence.trace?.publicQueueMethodHits?.length === 0,
    activeSteerNoQueuedTurnGui: evidence.gui?.queuedTurnGuiCount === 0,
  };
}
