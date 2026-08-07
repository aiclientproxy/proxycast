import {
  APP_SERVER_METHOD_SESSION_LIST,
  APP_SERVER_METHOD_SESSION_READ,
  APP_SERVER_METHOD_SESSION_START,
  APP_SERVER_METHOD_SESSION_TURN_START,
  SESSION_TITLE,
  TURN_PLAN_UPDATE_DONE_TEXT,
  TURN_PLAN_UPDATE_EXPLANATION,
  TURN_PLAN_UPDATE_PROMPT,
  TURN_PLAN_UPDATE_SCENARIO,
  TURN_PLAN_UPDATE_STEPS,
} from "./claw-chat-current-fixture-constants.mjs";
import { sendPromptFromGui } from "./claw-chat-current-fixture-gui-actions.mjs";
import { waitForGuiChatCompleted } from "./claw-chat-current-fixture-gui-completion-waits.mjs";
import {
  collectReadModelItems,
  collectReadModelTurns,
  readModelLatestTurnStatus,
  readModelTurnId,
} from "./claw-chat-current-fixture-read-model-core.mjs";
import { waitForSessionReadCompleted } from "./claw-chat-current-fixture-read-model-waits.mjs";
import {
  collectRuntimeEvents,
  drainAppServerEventsFromPage,
  evaluatePageSnapshot,
  invokeAppServerFromPage,
  reloadRendererDocument,
  waitForRendererReady,
} from "./claw-chat-current-fixture-rpc.mjs";
import {
  openSessionFromSidebar,
  waitForGuiSessionVisible,
} from "./claw-chat-current-fixture-session.mjs";
import { sanitizeJson, sleep } from "./claw-chat-current-fixture-utils.mjs";

export async function runTurnPlanUpdateScenario({
  page,
  options,
  summary,
  appServerRequests,
  readTextProviderRequests,
  logStage,
}) {
  logStage("send-turn-plan-update-prompt-from-gui");
  const turnPlanUpdateInputSend = await sendPromptFromGui(
    page,
    options,
    TURN_PLAN_UPDATE_PROMPT,
  );
  summary.turnPlanUpdateInputSend = sanitizeJson(turnPlanUpdateInputSend);

  logStage("wait-gui-turn-plan-update-completed");
  const guiTurnPlanUpdateCompleted = await waitForGuiChatCompleted(
    page,
    options,
    {
      prompt: TURN_PLAN_UPDATE_PROMPT,
      doneText: TURN_PLAN_UPDATE_DONE_TEXT,
      summaryText: "执行清单已更新",
    },
  );
  summary.guiTurnPlanUpdateCompleted = sanitizeJson(guiTurnPlanUpdateCompleted);

  logStage("wait-read-model-turn-plan-update-completed");
  const readModel = await waitForSessionReadCompleted(
    page,
    options,
    appServerRequests,
    {
      prompt: TURN_PLAN_UPDATE_PROMPT,
      doneText: TURN_PLAN_UPDATE_DONE_TEXT,
      summaryText: "执行清单已更新",
    },
  );
  const readModelTurnPlanUpdateCompleted =
    summarizeTurnPlanReadModel(readModel);
  summary.readModelTurnPlanUpdateCompleted = sanitizeJson(
    readModelTurnPlanUpdateCompleted,
  );

  logStage("drain-turn-plan-updated-notification");
  const drained = await drainAppServerEventsFromPage(page, 200);
  const turnPlanUpdateNotification = summarizeTurnPlanNotification(
    drained.messages,
    readModelTurnPlanUpdateCompleted.latestTurnId,
  );
  summary.turnPlanUpdateDrain = summarizeTurnPlanDrain(drained.messages);
  summary.turnPlanUpdateNotification = sanitizeJson(turnPlanUpdateNotification);

  const turnPlanUpdateProviderRequests = summarizeProviderRequests(
    readTextProviderRequests?.() ?? [],
  );
  if (options.hookFixture) {
    return sanitizeJson({
      turnPlanUpdateInputSend,
      guiTurnPlanUpdateCompleted,
      readModelTurnPlanUpdateCompleted,
      turnPlanUpdateNotification,
      turnPlanUpdateProviderRequests,
      hookFocusedGateB: true,
    });
  }

  logStage("wait-live-turn-plan-checklist");
  const guiTurnPlanUpdateChecklist = await waitForTurnPlanChecklist(
    page,
    options,
  );
  summary.guiTurnPlanUpdateChecklist = sanitizeJson(guiTurnPlanUpdateChecklist);

  logStage("reload-turn-plan-update-session");
  const turnPlanUpdateReload = await reloadRendererDocument(page, options);
  const turnPlanUpdateRendererReady = await waitForRendererReady(page, options);
  const turnPlanUpdateSessionVisible = await waitForGuiSessionVisible(
    page,
    options,
    SESSION_TITLE,
  );
  const turnPlanUpdateSessionOpened = await openSessionFromSidebar(
    page,
    options,
    appServerRequests,
    {
      sessionId: options.sessionId,
      threadId: options.threadId,
      title: SESSION_TITLE,
    },
  );
  const guiTurnPlanUpdateHydrated = await waitForTurnPlanChecklist(
    page,
    options,
  );
  const hydratedRead = await invokeAppServerFromPage(
    page,
    APP_SERVER_METHOD_SESSION_READ,
    { threadId: options.threadId, includeTurns: true },
    appServerRequests,
  );

  return sanitizeJson({
    turnPlanUpdateInputSend,
    guiTurnPlanUpdateCompleted,
    guiTurnPlanUpdateChecklist,
    readModelTurnPlanUpdateCompleted,
    turnPlanUpdateNotification,
    turnPlanUpdateProviderRequests,
    turnPlanUpdateReload,
    turnPlanUpdateRendererReady,
    turnPlanUpdateSessionVisible,
    turnPlanUpdateSessionOpened,
    guiTurnPlanUpdateHydrated,
    readModelTurnPlanUpdateHydrated: summarizeTurnPlanReadModel(
      hydratedRead.result,
    ),
  });
}

export function buildTurnPlanUpdateScenarioAssertions(context) {
  const { summary, traceTurnStarts, appServerRequestMethods } = context;
  const live = summary.guiTurnPlanUpdateChecklist;
  const hydrated = summary.guiTurnPlanUpdateHydrated;
  const readModel = summary.readModelTurnPlanUpdateCompleted;
  const hydratedReadModel = summary.readModelTurnPlanUpdateHydrated;
  const notification = summary.turnPlanUpdateNotification;
  const provider = summary.turnPlanUpdateProviderRequests;
  const traceTurnStart = traceTurnStarts.find(
    (entry) =>
      entry.inputText === TURN_PLAN_UPDATE_PROMPT &&
      entry.transport === "electron-ipc" &&
      entry.status === "success",
  );

  return {
    turnPlanUpdateUsesRuntimeBackend: summary.backendMode === "runtime",
    turnPlanUpdateUsesCurrentSessionMethods:
      appServerRequestMethods.includes(APP_SERVER_METHOD_SESSION_START) &&
      appServerRequestMethods.includes(APP_SERVER_METHOD_SESSION_LIST) &&
      appServerRequestMethods.includes(APP_SERVER_METHOD_SESSION_READ),
    turnPlanUpdateUsesCurrentTurnStart:
      appServerRequestMethods.includes(APP_SERVER_METHOD_SESSION_TURN_START) &&
      Boolean(traceTurnStart),
    turnPlanUpdateProviderReceivedToolDefinition:
      provider.initialRequest?.hasUpdatePlanTool === true,
    turnPlanUpdateProviderReturnedAfterToolResult:
      provider.followupRequest?.hasPlanUpdatedToolResult === true &&
      provider.chatRequestCount === 2,
    turnPlanUpdateNotificationObserved:
      notification.methodCount >= 1 &&
      notification.normalizedEventType === "turn.plan.updated" &&
      notification.turnId === readModel.latestTurnId &&
      notification.threadId === summary.threadId &&
      notification.explanation === TURN_PLAN_UPDATE_EXPLANATION &&
      notification.hasExpectedPlan === true,
    turnPlanUpdateLiveChecklistVisible: checklistSnapshotPassed(live),
    turnPlanUpdateDoesNotCreatePlanUi:
      live?.visiblePlanBlockCount === 0 &&
      live?.visiblePlanDecisionCount === 0 &&
      live?.visibleUpdatePlanToolRowCount === 0,
    turnPlanUpdateReadModelCanonical:
      readModel.latestTurnStatus === "completed" &&
      readModel.updatePlanToolCount === 1 &&
      readModel.completedUpdatePlanToolCount === 1 &&
      readModel.planItemCount === 0 &&
      readModel.includesExpectedArguments === true &&
      readModel.includesFinalAnswer === true,
    turnPlanUpdateHydratesAfterReload:
      checklistSnapshotPassed(hydrated) &&
      hydratedReadModel.latestTurnStatus === "completed" &&
      hydratedReadModel.updatePlanToolCount === 1 &&
      hydratedReadModel.planItemCount === 0 &&
      hydratedReadModel.includesExpectedArguments === true,
    turnPlanUpdateInputReadyAfterCompletion:
      hydrated?.textareaVisible === true &&
      hydrated?.textareaDisabled === false &&
      hydrated?.stopButtonVisible === false,
    turnPlanUpdateUserAndAssistantVisible:
      summary.guiTurnPlanUpdateCompleted?.hasPrompt === true &&
      (summary.guiTurnPlanUpdateCompleted?.hasAssistantSummary === true ||
        summary.guiTurnPlanUpdateCompleted?.hasDoneText === true),
    turnPlanUpdateNoInvokeOrConsoleErrors:
      !context.errorRaw && context.actionableConsoleErrors.length === 0,
  };
}

export function summarizeTurnPlanReadModel(readModel) {
  const items = dedupeItems(collectReadModelItems(readModel));
  const planItems = items.filter((item) => item?.type === "plan");
  const updatePlanTools = items.filter(
    (item) => item?.type === "dynamicToolCall" && item?.tool === "update_plan",
  );
  const serializedTools = JSON.stringify(updatePlanTools);
  const turns = collectReadModelTurns(readModel);
  return sanitizeJson({
    latestTurnId: readModelTurnId(turns.at(-1)),
    latestTurnStatus: readModelLatestTurnStatus(readModel),
    itemCount: items.length,
    planItemCount: planItems.length,
    updatePlanToolCount: updatePlanTools.length,
    completedUpdatePlanToolCount: updatePlanTools.filter(
      (item) => item?.status === "completed" && item?.success === true,
    ).length,
    includesExpectedArguments: TURN_PLAN_UPDATE_STEPS.every(
      (step) =>
        serializedTools.includes(step.step) &&
        serializedTools.includes(step.status),
    ),
    includesPlanUpdatedOutput: serializedTools.includes("Plan updated"),
    includesFinalAnswer: JSON.stringify(items).includes(
      TURN_PLAN_UPDATE_DONE_TEXT,
    ),
  });
}

function dedupeItems(items) {
  const byId = new Map();
  for (const item of items) {
    const key = typeof item?.id === "string" ? item.id : JSON.stringify(item);
    byId.set(key, item);
  }
  return [...byId.values()];
}

function summarizeProviderRequests(requests) {
  const chatRequests = requests.filter(
    (request) =>
      request?.method === "POST" &&
      ["/chat/completions", "/v1/chat/completions"].includes(request?.url),
  );
  const summarize = (request) => ({
    authorized: request?.authorization === "present",
    hasPrompt: request?.bodySummary?.bodyIncludesTurnPlanUpdatePrompt === true,
    hasUpdatePlanTool:
      request?.bodySummary?.bodyIncludesUpdatePlanToolDefinition === true,
    hasPlanUpdatedToolResult:
      request?.bodySummary?.bodyIncludesPlanUpdatedToolResult === true,
  });
  return {
    chatRequestCount: chatRequests.length,
    initialRequest: summarize(chatRequests[0]),
    followupRequest: summarize(chatRequests[1]),
  };
}

function summarizeTurnPlanNotification(messages, expectedTurnId) {
  const matching = messages.filter(
    (message) =>
      message?.method === "turn/plan/updated" &&
      (!expectedTurnId || message?.params?.turnId === expectedTurnId),
  );
  const latest = matching.at(-1);
  const normalized = collectRuntimeEvents(matching).at(-1);
  const serializedPlan = JSON.stringify(latest?.params?.plan ?? []);
  return sanitizeJson({
    methodCount: matching.length,
    normalizedEventType: normalized?.type ?? null,
    threadId: latest?.params?.threadId ?? null,
    turnId: latest?.params?.turnId ?? null,
    explanation: latest?.params?.explanation ?? null,
    hasExpectedPlan: TURN_PLAN_UPDATE_STEPS.every(
      (step) =>
        serializedPlan.includes(step.step) &&
        serializedPlan.includes(
          step.status === "in_progress" ? "inProgress" : step.status,
        ),
    ),
  });
}

function summarizeTurnPlanDrain(messages) {
  const methods = messages
    .map((message) => message?.method)
    .filter((method) => typeof method === "string");
  const hookMessages = messages.filter(
    (message) =>
      message?.method === "hook/started" ||
      message?.method === "hook/completed",
  );
  const hookRunIds = (method) =>
    hookMessages
      .filter((message) => message.method === method)
      .map((message) => message?.params?.run?.id)
      .filter((runId) => typeof runId === "string" && runId.length > 0);
  const startedRunIds = hookRunIds("hook/started");
  const completedRunIds = hookRunIds("hook/completed");
  return sanitizeJson({
    messageCount: messages.length,
    methods,
    planIndex: methods.lastIndexOf("turn/plan/updated"),
    terminalIndex: methods.lastIndexOf("turn/completed"),
    hookLifecycle: {
      startedRunIds,
      completedRunIds,
      pairedRunIds: startedRunIds.filter((runId) =>
        completedRunIds.includes(runId),
      ),
    },
  });
}

async function waitForTurnPlanChecklist(page, options) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(
      page,
      ({ prompt, doneText, steps }) => {
        const visible = (node) => {
          if (!(node instanceof HTMLElement)) return false;
          const rect = node.getBoundingClientRect();
          const style = window.getComputedStyle(node);
          return (
            rect.width > 0 &&
            rect.height > 0 &&
            style.display !== "none" &&
            style.visibility !== "hidden"
          );
        };
        const ownerDefinitions = [
          {
            kind: "run-control",
            root: '[data-testid="task-center-run-control-plan"]',
            item: '[data-testid="task-center-run-control-plan-item"]',
            revision: '[data-testid="task-center-run-control-plan-revision"]',
          },
          {
            kind: "task-rail",
            root: '[data-testid="task-center-task-rail-plan"]',
            item: '[data-testid="task-center-task-rail-plan-item"]',
            revision: '[data-testid="task-center-task-rail-plan-revision"]',
          },
        ];
        const owners = ownerDefinitions.flatMap((definition) => {
          const root = document.querySelector(definition.root);
          if (!visible(root)) return [];
          const revision = root.querySelector(definition.revision);
          return [
            {
              kind: definition.kind,
              items: Array.from(root.querySelectorAll(definition.item)).map(
                (item) => ({
                  text: item.textContent || "",
                  status: item.getAttribute("data-status"),
                }),
              ),
              revisionSource:
                revision?.getAttribute("data-plan-source") || null,
            },
          ];
        });
        const bodyText = document.body?.innerText || "";
        const expectedStatus = new Map([
          ["completed", "completed"],
          ["in_progress", "running"],
          ["pending", "pending"],
        ]);
        const completeOwners = owners.filter((owner) =>
          steps.every((step) =>
            owner.items.some(
              (item) =>
                item.text.includes(step.step) &&
                item.status === expectedStatus.get(step.status),
            ),
          ),
        );
        const textarea = Array.from(
          document.querySelectorAll('textarea[name="agent-chat-message"]'),
        ).find(visible);
        const stopButtons = Array.from(
          document.querySelectorAll(
            '[data-testid="stop-btn"], [data-testid="stop-button"]',
          ),
        ).filter(visible);
        return {
          url: window.location.href,
          hasPrompt: bodyText.includes(prompt),
          hasDoneText: bodyText.includes(doneText),
          ownerCount: owners.length,
          completeOwnerCount: completeOwners.length,
          owners,
          visiblePlanBlockCount: Array.from(
            document.querySelectorAll('[data-testid="agent-plan-block"]'),
          ).filter(visible).length,
          visiblePlanDecisionCount: Array.from(
            document.querySelectorAll(
              '[data-testid="plan-composer-decision-panel"]',
            ),
          ).filter(visible).length,
          visibleUpdatePlanToolRowCount: Array.from(
            document.querySelectorAll(
              '[data-testid="tool-call-row"][data-tool-name="update_plan"]',
            ),
          ).filter(visible).length,
          textareaVisible: Boolean(textarea),
          textareaDisabled:
            textarea instanceof HTMLTextAreaElement ? textarea.disabled : null,
          stopButtonVisible: stopButtons.length > 0,
        };
      },
      {
        prompt: TURN_PLAN_UPDATE_PROMPT,
        doneText: TURN_PLAN_UPDATE_DONE_TEXT,
        steps: TURN_PLAN_UPDATE_STEPS,
      },
    );
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;
    if (checklistSnapshotPassed(snapshot)) {
      return sanitizeJson(snapshot);
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `GUI 未显示 update_plan checklist: ${JSON.stringify(sanitizeJson(lastSnapshot))}`,
  );
}

function checklistSnapshotPassed(snapshot) {
  return (
    snapshot?.hasPrompt === true &&
    snapshot?.hasDoneText === true &&
    snapshot?.completeOwnerCount > 0 &&
    snapshot?.visiblePlanBlockCount === 0 &&
    snapshot?.visiblePlanDecisionCount === 0 &&
    snapshot?.visibleUpdatePlanToolRowCount === 0
  );
}

export {
  TURN_PLAN_UPDATE_DONE_TEXT,
  TURN_PLAN_UPDATE_PROMPT,
  TURN_PLAN_UPDATE_SCENARIO,
  TURN_PLAN_UPDATE_STEPS,
};
