import fs from "node:fs";
import {
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_GET,
  APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_LIST,
  APP_SERVER_METHOD_SESSION_TURN_CANCEL,
  APP_SERVER_METHOD_THREAD_SHELL_COMMAND,
  APP_SERVER_METHOD_WORKFLOW_READ,
  APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_LIST,
  APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST,
  CONTINUE_PROMPT,
  CONTENT_FACTORY_INLINE_IMAGE_URL,
  EVENT_READ_PROBE_TOOL_NAME,
  GOAL_PROMPT,
  HOME_HOTPATH_GREETING_SCENARIO,
  HOME_HOTPATH_SCENARIO,
  IMAGE_COMMAND_CREATE_TASK_TOOL_NAME,
  IMAGE_COMMAND_PROMPT,
  IMAGE_FIXTURE_MODEL,
  INPUTBAR_RICH_RESTORE_PATH_NAME,
  INPUTBAR_RICH_RESTORE_PROMPT,
  NEWS_PROMPT,
  PLAN_PROMPT,
  PLAN_STEPS,
  SESSION_ID,
  SOUL_STYLE_SCENARIO,
} from "./claw-chat-current-fixture-constants.mjs";
import { buildContentFactoryArticleWorkspaceScenarioAssertions } from "./claw-chat-current-fixture-content-factory-assertions.mjs";
import {
  buildApprovalRequestDecisionScenarioAssertions,
  buildApprovalRequestFullAccessScenarioAssertions,
  buildApprovalRequestHostInterruptScenarioAssertions,
  buildApprovalRequestResumeScenarioAssertions,
} from "./claw-chat-current-fixture-approval-assertions.mjs";
import {
  buildLiveTailCommitScenarioAssertions,
  buildMcpStructuredContentScenarioAssertions,
  buildMediaReferenceScenarioAssertions,
  buildReasoningFirstVisibleScenarioAssertions,
} from "./claw-chat-current-fixture-runtime-surface-assertions.mjs";
import {
  buildExpertSkillsRuntimeScenarioAssertions,
  buildSkillsRuntimeScenarioAssertions,
} from "./claw-chat-current-fixture-skills-runtime-assertions.mjs";
import { buildElectronResizeReflowScenarioAssertions } from "./claw-chat-current-fixture-resize-reflow-assertions.mjs";
import { buildTerminalScenarioAssertions } from "./claw-chat-current-fixture-terminal-assertions.mjs";
import { buildTypedErrorScenarioAssertions } from "./claw-chat-current-fixture-typed-error.mjs";
import { buildWebToolsRenderingScenarioAssertions } from "./claw-chat-current-fixture-web-tools-assertions.mjs";
import { buildSoulStyleScenarioAssertions } from "./claw-chat-current-fixture-soul-style.mjs";
import { buildActiveSteerScenarioAssertions } from "./claw-chat-current-fixture-active-steer.mjs";
import { buildUnknownItemScenarioAssertions } from "./claw-chat-current-fixture-unknown-item.mjs";

function readImageCommandTaskFromHarness(harness) {
  const launch =
    harness?.image_command_intent ?? harness?.imageCommandIntent ?? null;
  const requestContext = launch?.request_context ?? launch?.requestContext;
  return (
    launch?.image_task ??
    launch?.imageTask ??
    requestContext?.image_task ??
    requestContext?.imageTask ??
    null
  );
}

function flattenBackendEmitTypesForPrompt(backendLedger, prompt) {
  const startIndex = backendLedger.findIndex(
    (entry) => entry?.kind === "turnStart" && entry?.inputText === prompt,
  );
  if (startIndex < 0) {
    return [];
  }
  const emitTypes = [];
  for (const entry of backendLedger.slice(startIndex + 1)) {
    if (entry?.kind === "turnStart") {
      break;
    }
    if (entry?.kind !== "backendEmit" || !Array.isArray(entry.eventTypes)) {
      continue;
    }
    emitTypes.push(...entry.eventTypes.filter(Boolean));
  }
  return emitTypes;
}

function includesOrderedEventTypes(eventTypes, expectedTypes) {
  let cursor = 0;
  for (const eventType of eventTypes) {
    if (eventType === expectedTypes[cursor]) {
      cursor += 1;
    }
    if (cursor >= expectedTypes.length) {
      return true;
    }
  }
  return false;
}

function streamParserBoundaryBackendObserved(backendLedger) {
  const emitTypes = flattenBackendEmitTypesForPrompt(
    backendLedger,
    NEWS_PROMPT,
  );
  return (
    emitTypes.includes("message.completed") &&
    includesOrderedEventTypes(emitTypes, [
      "message.delta",
      "message.completed",
      "turn.completed",
    ])
  );
}

function homeHotpathTraceSession(summary) {
  const sessions = Array.isArray(summary.agentUiPerformanceTrace?.sessions)
    ? summary.agentUiPerformanceTrace.sessions
    : [];
  return (
    sessions.find(
      (session) =>
        typeof session?.metrics?.homeInputToPendingPreviewPaintMs ===
          "number" ||
        typeof session?.metrics?.inputbarTriggerToPendingPreviewPaintMs ===
          "number",
    ) ?? null
  );
}

function metricWithin(traceSession, key, maxMs) {
  const value = traceSession?.metrics?.[key];
  return typeof value === "number" && Number.isFinite(value) && value <= maxMs;
}

const HOME_HOTPATH_PENDING_PREVIEW_PAINT_BUDGET_MS = 750;
const HOME_HOTPATH_PENDING_PROJECTION_VISIBLE_BUDGET_MS = 250;
const HOME_HOTPATH_SEND_DISPATCH_BUDGET_MS = 250;
const HOME_HOTPATH_SUBMIT_ACCEPTED_BUDGET_MS = 1800;
const HOME_HOTPATH_TEXT_DELTA_TO_PAINT_BUDGET_MS = 250;

function buildHomeHotpathScenarioAssertions({
  homeHotpathPrompt,
  homeHotpathTurnStart,
  summary,
}) {
  const hotpath = summary.homeHotpath ?? {};
  const traceSession = homeHotpathTraceSession(summary);
  const postSubmitProjection = hotpath.postSubmitProjection ?? {};
  const completedProjection = hotpath.completedProjection ?? {};
  const expectedPrompt = hotpath.prompt ?? homeHotpathPrompt ?? NEWS_PROMPT;
  const preTurnTraceAvailable =
    typeof hotpath.preTurnTrace?.clickAt === "string" &&
    typeof hotpath.preTurnTrace?.turnStartAt === "string";
  return {
    homeHotpathScenarioRegistered:
      HOME_HOTPATH_SCENARIO === "home-hotpath" &&
      HOME_HOTPATH_GREETING_SCENARIO === "home-hotpath-greeting",
    homeHotpathStartedFromEmptyState:
      hotpath.homeOpened?.after?.hasEmptyStateFirstScreen === true &&
      hotpath.homeOpened?.after?.hasConnectedComposer === true &&
      hotpath.homeOpened?.after?.textareaVisible === true &&
      hotpath.homeOpened?.after?.textareaDisabled === false,
    homeHotpathPromptReachedBackend:
      homeHotpathTurnStart?.inputText === expectedPrompt &&
      typeof hotpath.backendTurnStart?.sessionId === "string" &&
      hotpath.backendTurnStart.sessionId.length > 0,
    homeHotpathUserMessageVisibleAfterSubmit:
      hotpath.inputSend?.afterClick?.promptInBody === true ||
      postSubmitProjection.promptInBody === true,
    homeHotpathNoBlankConversationAfterSubmit:
      postSubmitProjection.hasEmptyConversationText === false &&
      postSubmitProjection.hasNoAvailableModelText === false,
    homeHotpathNoTransientFallbackAfterInputFill:
      hotpath.inputSend?.afterFillStability?.stable === true,
    homeHotpathNoFlickerBetweenSubmitAndConversation:
      hotpath.submitToConversationStability?.stable === true,
    homeHotpathPendingProjectionVisibleWithinBudget:
      typeof hotpath.submitToConversationStability?.conversationStartedAtMs ===
        "number" &&
      hotpath.submitToConversationStability.conversationStartedAtMs <=
        HOME_HOTPATH_PENDING_PROJECTION_VISIBLE_BUDGET_MS,
    homeHotpathNoImperativePendingShell:
      hotpath.submitToConversationStability?.noImperativePendingShell ===
        true &&
      postSubmitProjection.imperativePendingShellCount === 0 &&
      completedProjection.imperativePendingShellCount === 0,
    homeHotpathMainAreaBoundsStable:
      hotpath.submitToConversationStability?.mainAreaBounds?.stable === true,
    homeHotpathPreTurnTraceWindowAvailable: preTurnTraceAvailable,
    homeHotpathNoAuxiliaryAppServerBeforeTurnStart:
      Array.isArray(
        hotpath.preTurnTrace?.blockedAuxiliaryMethodsBeforeTurnStart,
      ) &&
      hotpath.preTurnTrace.blockedAuxiliaryMethodsBeforeTurnStart.length === 0,
    homeHotpathCompletedInGui:
      hotpath.guiCompleted?.hasPrompt === true &&
      (hotpath.guiCompleted?.hasAssistantSummary === true ||
        hotpath.guiCompleted?.hasDoneText === true) &&
      hotpath.guiCompleted?.textareaVisible === true &&
      hotpath.guiCompleted?.textareaDisabled === false &&
      hotpath.guiCompleted?.stopButtonVisible === false,
    homeHotpathReadModelCompleted:
      hotpath.readModelCompleted?.available === true &&
      hotpath.readModelCompleted?.includesPrompt === true &&
      hotpath.readModelCompleted?.includesDone === true,
    homeHotpathNoHomeOrModelFallbackAfterComplete:
      completedProjection.hasTaskCenterHomeText === false &&
      completedProjection.hasEmptyConversationText === false &&
      completedProjection.hasNoAvailableModelText === false,
    homeHotpathNoTransientHomeOrBlankDuringRun:
      hotpath.stability?.stable === true,
    homeHotpathNoPostCompletionRefreshFlicker:
      hotpath.postCompletionStability?.stable === true,
    homeHotpathTraceHasPendingPreviewPaint:
      Boolean(traceSession) &&
      (typeof traceSession?.metrics?.homeInputToPendingPreviewPaintMs ===
        "number" ||
        typeof traceSession?.metrics?.inputbarTriggerToPendingPreviewPaintMs ===
          "number"),
    homeHotpathPendingPreviewPaintWithinBudget:
      metricWithin(
        traceSession,
        "homeInputToPendingPreviewPaintMs",
        HOME_HOTPATH_PENDING_PREVIEW_PAINT_BUDGET_MS,
      ) ||
      metricWithin(
        traceSession,
        "inputbarTriggerToPendingPreviewPaintMs",
        HOME_HOTPATH_PENDING_PREVIEW_PAINT_BUDGET_MS,
      ),
    homeHotpathTraceHasSendDispatch:
      typeof traceSession?.metrics?.homeInputToSendDispatchMs === "number" ||
      typeof traceSession?.metrics?.inputbarTriggerToSendDispatchMs ===
        "number",
    homeHotpathSendDispatchWithinBudget:
      metricWithin(
        traceSession,
        "homeInputToSendDispatchMs",
        HOME_HOTPATH_SEND_DISPATCH_BUDGET_MS,
      ) ||
      metricWithin(
        traceSession,
        "inputbarTriggerToSendDispatchMs",
        HOME_HOTPATH_SEND_DISPATCH_BUDGET_MS,
      ),
    homeHotpathTraceHasSubmitAccepted:
      typeof traceSession?.metrics?.homeInputToSubmitAcceptedMs === "number" ||
      typeof traceSession?.metrics?.inputbarTriggerToSubmitAcceptedMs ===
        "number",
    homeHotpathSubmitAcceptedWithinBudget:
      metricWithin(
        traceSession,
        "homeInputToSubmitAcceptedMs",
        HOME_HOTPATH_SUBMIT_ACCEPTED_BUDGET_MS,
      ) ||
      metricWithin(
        traceSession,
        "inputbarTriggerToSubmitAcceptedMs",
        HOME_HOTPATH_SUBMIT_ACCEPTED_BUDGET_MS,
      ),
    homeHotpathTraceHasFirstTextPaint:
      typeof traceSession?.metrics?.homeInputToFirstTextPaintMs === "number" ||
      typeof traceSession?.metrics?.submitAcceptedToFirstTextPaintMs ===
        "number",
    homeHotpathTextDeltaToPaintWithinBudget: metricWithin(
      traceSession,
      "firstTextDeltaToFirstTextPaintMs",
      HOME_HOTPATH_TEXT_DELTA_TO_PAINT_BUDGET_MS,
    ),
  };
}

export function buildScenarioAssertions(context) {
  const {
    appServerRequestMethods,
    backendLedger,
    collaborationMode,
    continueTurnStart,
    expectedExpertHarnessSkillRef,
    expertHarnessMetadata,
    expertHarnessSkillRefs,
    expertRuntimeMetadata,
    expertPanelSkillsRuntimeTurnStart,
    expertSkillsRuntimeTurnStart,
    expectedImageIntentRoutedPrompt,
    goalObjectiveText,
    goalTurnStart,
    homeHotpathPrompt,
    homeHotpathTurnStart,
    isHomeHotpathScenario,
    imageCommandHarness,
    imageCommandTurnStart,
    guiTurnStartReachedBackend,
    hasCancelPhase,
    inputbarRichRestoreTurnStart,
    isAnyExpertSkillsRuntimeScenario,
    isApprovalRequestCancelScenario,
    isApprovalRequestDeclineScenario,
    isApprovalRequestDecisionScenario,
    isApprovalRequestFullAccessScenario,
    isApprovalRequestHostInterruptScenario,
    isApprovalRequestResumeScenario,
    isCancelThenContinueScenario,
    isContentFactoryArticleWorkspaceScenario,
    isContentFactoryInlineImageArticleWorkspaceScenario,
    isExpertPanelSkillsRuntimeScenario,
    isExpertPlazaSkillsRuntimeScenario,
    isElectronResizeReflowScenario,
    isGoalScenario,
    isImageCommandScenario,
    isLiveTailCommitScenario,
    isInputbarRichRestoreScenario,
    isActiveSteerScenario,
    isMcpStructuredContentScenario,
    isMediaReferenceScenario,
    isPlanScenario,
    isReasoningFirstVisibleScenario,
    isTerminalCanceledAfterAnswerScenario,
    isTerminalFailedAfterAnswerScenario,
    isTypedErrorRetryFailureScenario,
    isTypedErrorRetrySuccessScenario,
    isTypedErrorRetryScenario,
    isUnknownItemScenario,
    isRightSurfaceVisualMatrixScenario,
    isSkillsRuntimeScenario,
    isSoulStyleScenario,
    isTerminalStaleGuardScenario,
    isWebToolsRenderingScenario,
    latestTurnCancel,
    electronResizeReflowTurnStart,
    liveTailCommitTurnStart,
    manualEnableRuntimeBinding,
    manualEnableRuntimeMetadata,
    mcpStructuredContentTurnStart,
    mediaReferenceTurnStart,
    newsTurnStart,
    pageText,
    planImplementationTurnStart,
    planTurnStart,
    approvalRequestResumeTurnStart,
    approvalRequestFullAccessTurnStart,
    skillsRuntimeTurnStart,
    explicitSkillsRuntimeTurnStart,
    manualEnableSkillsRuntimeTurnStart,
    summary,
    terminalCanceledAfterAnswerTurnStart,
    terminalFailedAfterAnswerTurnStart,
    terminalStaleGuardFirstTurnStart,
    typedErrorTurnStart,
    terminalStaleGuardSecondTurnStart,
    webToolsRenderingTurnStart,
    reasoningFirstVisibleTurnStart,
    workspace,
  } = context;
  const rightSurfaceVisualMatrix = summary.rightSurfaceVisualMatrix ?? {};
  const rightSurfaceVisualCaptures = rightSurfaceVisualMatrix.captures ?? {};
  const rightSurfaceVisualAppSurface =
    rightSurfaceVisualCaptures.appSurface?.stable?.pluginSurface ?? {};
  const imageCommandTask = readImageCommandTaskFromHarness(imageCommandHarness);
  const imageCommandRuntimeContract =
    imageCommandTask?.runtime_contract ?? imageCommandTask?.runtimeContract;
  const imageCommandContractKey =
    imageCommandTask?.modality_contract_key ??
    imageCommandTask?.modalityContractKey ??
    imageCommandRuntimeContract?.contract_key ??
    imageCommandRuntimeContract?.contractKey;
  const scenarioAssertions = isUnknownItemScenario
    ? buildUnknownItemScenarioAssertions({
        appServerRequestMethods,
        backendLedger,
        summary,
      })
    : isActiveSteerScenario
      ? buildActiveSteerScenarioAssertions({ summary })
      : isSoulStyleScenario
        ? buildSoulStyleScenarioAssertions({
            summary,
            guiTurnStartReachedBackend,
          })
        : isHomeHotpathScenario
          ? buildHomeHotpathScenarioAssertions({
              homeHotpathPrompt,
              homeHotpathTurnStart,
              summary,
            })
          : isContentFactoryArticleWorkspaceScenario
            ? {
                ...buildContentFactoryArticleWorkspaceScenarioAssertions({
                  appServerRequestMethods,
                  backendLedger,
                  pageText,
                  summary,
                }),
                ...(isContentFactoryInlineImageArticleWorkspaceScenario
                  ? {
                      contentFactoryInlineImageTaskSlotPersisted:
                        summary.contentFactoryInlineImageTaskCreated
                          ?.payloadUsage === "document-inline" &&
                        summary.contentFactoryInlineImageTaskCreated
                          ?.relationshipSlotId ===
                          "article-inline-image-slot-e2e",
                      contentFactoryInlineImageTaskEventEmitted:
                        summary.contentFactoryInlineImageTaskSubmittedEvent
                          ?.emitted === true &&
                        summary.contentFactoryInlineImageTaskSubmittedEvent
                          ?.slotId === "article-inline-image-slot-e2e",
                      contentFactoryInlineImageTaskCompleted:
                        summary.contentFactoryInlineImageTaskCompleted
                          ?.normalizedStatus === "succeeded" &&
                        summary.contentFactoryInlineImageTaskCompleted
                          ?.relationshipSlotId ===
                          "article-inline-image-slot-e2e" &&
                        summary.contentFactoryInlineImageTaskCompleted
                          ?.firstImageUrl === CONTENT_FACTORY_INLINE_IMAGE_URL,
                      contentFactoryInlineImageReadModelRestored:
                        summary.contentFactoryInlineImageReadModel
                          ?.hasInlineTitle === true &&
                        summary.contentFactoryInlineImageReadModel
                          ?.hasAnchorText === true &&
                        summary.contentFactoryInlineImageReadModel
                          ?.hasImageUrl === true &&
                        summary.contentFactoryInlineImageReadModel
                          ?.hasPendingProtocol === false,
                      contentFactoryInlineImageArticleRestored:
                        summary.contentFactoryInlineImageCanvas?.hasImageUrl ===
                          true &&
                        summary.contentFactoryInlineImageCanvas
                          ?.hasRenderedImage === true &&
                        summary.contentFactoryInlineImageCanvas
                          ?.hasRenderedImageNearAnchor === true &&
                        summary.contentFactoryInlineImageCanvas
                          ?.hasUnavailablePlaceholderForExpected === false &&
                        summary.contentFactoryInlineImageCanvas
                          ?.hasPendingProtocol === false,
                    }
                  : {}),
              }
            : isRightSurfaceVisualMatrixScenario
              ? {
                  rightSurfaceVisualMatrixRequestedThroughAppServer:
                    appServerRequestMethods.includes(
                      APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST,
                    ) &&
                    appServerRequestMethods.includes(
                      APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_LIST,
                    ) &&
                    Boolean(
                      rightSurfaceVisualMatrix.requests?.files?.requestId,
                    ) &&
                    Boolean(
                      rightSurfaceVisualMatrix.requests?.objectCanvas
                        ?.requestId,
                    ) &&
                    Boolean(
                      rightSurfaceVisualMatrix.requests?.browser?.requestId,
                    ) &&
                    Boolean(
                      rightSurfaceVisualMatrix.requests
                        ?.appSurfaceContentFactory?.requestId,
                    ) &&
                    Boolean(
                      rightSurfaceVisualMatrix.requests?.appSurfacePromptLab
                        ?.requestId,
                    ),
                  rightSurfaceVisualMatrixFilesSurfaceVisible:
                    rightSurfaceVisualCaptures.files?.stable?.activeSurface ===
                      "files" &&
                    rightSurfaceVisualCaptures.files?.stable?.rootVisible ===
                      true,
                  rightSurfaceVisualMatrixObjectCanvasSurfaceVisible:
                    rightSurfaceVisualCaptures.objectCanvas?.stable
                      ?.activeSurface === "objectCanvas" &&
                    rightSurfaceVisualCaptures.objectCanvas?.stable
                      ?.expectedActiveSurface === "objectCanvas" &&
                    rightSurfaceVisualCaptures.objectCanvas?.stable
                      ?.rootVisible === true &&
                    rightSurfaceVisualCaptures.objectCanvas?.stable
                      ?.visibleRootKinds?.[0] === "objectCanvas",
                  rightSurfaceVisualMatrixExpertSurfaceVisible:
                    rightSurfaceVisualCaptures.expertInfo?.stable
                      ?.activeSurface === "expertInfo" &&
                    rightSurfaceVisualCaptures.expertInfo?.stable
                      ?.rootVisible === true,
                  rightSurfaceVisualMatrixBrowserSurfaceVisible:
                    rightSurfaceVisualCaptures.browser?.stable
                      ?.activeSurface === "browser" &&
                    rightSurfaceVisualCaptures.browser?.stable?.rootVisible ===
                      true &&
                    typeof rightSurfaceVisualCaptures.browser?.screenshot ===
                      "string" &&
                    rightSurfaceVisualCaptures.browser.screenshot.endsWith(
                      "-right-surface-browser.png",
                    ) &&
                    fs.existsSync(
                      rightSurfaceVisualCaptures.browser.screenshot,
                    ) &&
                    rightSurfaceVisualCaptures.browser?.stable?.browserSurface
                      ?.sessionId === "fixture-browser-session" &&
                    rightSurfaceVisualCaptures.browser?.stable?.browserSurface
                      ?.profileKey === "fixture-profile" &&
                    rightSurfaceVisualCaptures.browser?.stable?.browserSurface
                      ?.controlOwner === "shared",
                  rightSurfaceVisualMatrixAppSurfaceVisible:
                    rightSurfaceVisualCaptures.appSurface?.stable
                      ?.activeSurface === "appSurface" &&
                    rightSurfaceVisualCaptures.appSurface?.stable
                      ?.rootVisible === true,
                  rightSurfaceVisualMatrixAppSurfaceMultiInstanceTabs:
                    rightSurfaceVisualAppSurface.tabs?.visible === true &&
                    rightSurfaceVisualAppSurface.tabCount >= 2 &&
                    rightSurfaceVisualAppSurface.visibleFrameCount === 1 &&
                    rightSurfaceVisualAppSurface.visibleViewportCount === 1 &&
                    rightSurfaceVisualAppSurface.tabLabels?.includes(
                      "内容工厂",
                    ) === true &&
                    rightSurfaceVisualAppSurface.tabLabels?.includes(
                      "Prompt Lab",
                    ) === true,
                  rightSurfaceVisualMatrixSurfacesMutuallyExclusive: [
                    rightSurfaceVisualCaptures.files?.stable,
                    rightSurfaceVisualCaptures.objectCanvas?.stable,
                    rightSurfaceVisualCaptures.expertInfo?.stable,
                    rightSurfaceVisualCaptures.browser?.stable,
                    rightSurfaceVisualCaptures.appSurface?.stable,
                  ].every(
                    (capture) =>
                      Array.isArray(capture?.visibleRootKinds) &&
                      capture.visibleRootKinds.length === 1 &&
                      capture.visibleRootKinds[0] ===
                        (capture.expectedSurface ?? capture.activeSurface),
                  ),
                  rightSurfaceVisualMatrixHostsFillRightSide: [
                    rightSurfaceVisualCaptures.files?.stable,
                    rightSurfaceVisualCaptures.objectCanvas?.stable,
                    rightSurfaceVisualCaptures.expertInfo?.stable,
                    rightSurfaceVisualCaptures.browser?.stable,
                    rightSurfaceVisualCaptures.appSurface?.stable,
                  ].every(
                    (capture) =>
                      capture?.geometry?.hostFillsCanvasPanel === true &&
                      capture?.geometry?.rootFillsSurfaceViewport === true,
                  ),
                  rightSurfaceVisualMatrixObjectCanvasRailVisible:
                    rightSurfaceVisualCaptures.objectCanvas?.stable
                      ?.activeSurface === "objectCanvas" &&
                    rightSurfaceVisualCaptures.objectCanvas?.stable?.geometry
                      ?.rootFillsSurfaceViewport === true,
                  rightSurfaceVisualMatrixPendingConsumeKeepsSurfaceOpen:
                    rightSurfaceVisualCaptures.files?.opened?.rootVisible ===
                      true &&
                    rightSurfaceVisualCaptures.files?.stable?.rootVisible ===
                      true &&
                    rightSurfaceVisualCaptures.objectCanvas?.opened
                      ?.rootVisible === true &&
                    rightSurfaceVisualCaptures.objectCanvas?.stable
                      ?.rootVisible === true &&
                    rightSurfaceVisualCaptures.browser?.opened?.rootVisible ===
                      true &&
                    rightSurfaceVisualCaptures.browser?.stable?.rootVisible ===
                      true &&
                    rightSurfaceVisualCaptures.appSurface?.opened
                      ?.rootVisible === true &&
                    rightSurfaceVisualCaptures.appSurface?.stable
                      ?.rootVisible === true,
                  rightSurfaceVisualMatrixDoesNotUseModelTurn:
                    backendLedger.every((entry) => entry.kind !== "turnStart"),
                }
              : isPlanScenario
                ? {
                    planModeEnabledInGui:
                      summary.planModeEnabled?.statusChipVisible === true,
                    planPromptReachedBackend:
                      planTurnStart?.inputText === PLAN_PROMPT,
                    planCollaborationModeReachedBackend:
                      collaborationMode === "plan",
                    guiPlanRailVisible:
                      summary.guiPlanCompleted?.hasPlanSection === true &&
                      summary.guiPlanCompleted?.planOwnerHasAllSteps === true &&
                      Array.isArray(
                        summary.guiPlanCompleted?.planOwnerKindsWithAllSteps,
                      ) &&
                      summary.guiPlanCompleted.planOwnerKindsWithAllSteps
                        .length > 0,
                    guiPlanStepsVisible:
                      summary.guiPlanCompleted?.planOwnerHasAllSteps === true &&
                      (summary.guiPlanCompleted?.planStepHits ?? []).every(
                        (hit) =>
                          hit.visible === true &&
                          Array.isArray(hit.owners) &&
                          hit.owners.length > 0,
                      ),
                    guiPlanDecisionDrawerVisible:
                      summary.guiPlanCompleted?.planDecisionVisible === true &&
                      summary.guiPlanCompleted?.planDecisionHasTitle === true &&
                      summary.guiPlanCompleted?.planDecisionHasAcceptOption ===
                        true &&
                      summary.guiPlanCompleted?.planDecisionHasAdjustInput ===
                        true &&
                      summary.guiPlanCompleted?.planDecisionRevisionBound ===
                        true,
                    guiPlanDidNotAutoImplement: !planImplementationTurnStart,
                    readModelPlanCompleted:
                      summary.readModelPlanCompleted?.includesPrompt === true &&
                      summary.readModelPlanCompleted
                        ?.includesProposedPlanBlock === true &&
                      summary.readModelPlanCompleted?.includesPlanItem ===
                        true &&
                      summary.readModelPlanCompleted?.includesAllPlanSteps ===
                        true &&
                      summary.readModelPlanCompleted?.latestTurnCompleted ===
                        true,
                    readModelPlanThreadItemRevisioned:
                      summary.readModelPlanThreadItem
                        ?.hasCompletedPlanThreadItem === true &&
                      summary.readModelPlanThreadItem?.hasRevisionId === true &&
                      summary.readModelPlanThreadItem?.source ===
                        "proposed_plan" &&
                      summary.readModelPlanThreadItem?.includesAllPlanSteps ===
                        true,
                    guiPlanHistoryHydrateCompleted:
                      summary.guiPlanHistoryHydrateCompleted?.hasPrompt ===
                        true &&
                      summary.guiPlanHistoryHydrateCompleted
                        ?.planOwnerHasAllSteps === true &&
                      summary.guiPlanHistoryHydrateCompleted
                        ?.planDecisionVisible === true &&
                      summary.guiPlanHistoryHydrateCompleted
                        ?.planDecisionHasTitle === true &&
                      summary.guiPlanHistoryHydrateCompleted
                        ?.planDecisionRevisionBound === true &&
                      summary.guiPlanHistoryHydrateCompleted
                        ?.legacyUpdatePlanToolVisible === false,
                    readModelPlanHistoryHydratePreserved:
                      summary.readModelPlanHistoryHydrate
                        ?.hasCompletedPlanThreadItem === true &&
                      summary.readModelPlanHistoryHydrate?.hasRevisionId ===
                        true &&
                      summary.readModelPlanHistoryHydrate?.source ===
                        "proposed_plan" &&
                      summary.readModelPlanHistoryHydrate
                        ?.includesAllPlanSteps === true,
                    legacyUpdatePlanToolHidden:
                      summary.guiPlanHistoryHydrateCompleted
                        ?.legacyUpdatePlanToolVisible === false &&
                      summary.readModelPlanThreadItem
                        ?.legacyUpdatePlanToolItemCount === 0,
                    proposedPlanVisible:
                      summary.guiPlanCompleted?.planOwnerHasAllSteps === true &&
                      summary.guiPlanHistoryHydrateCompleted
                        ?.planOwnerHasAllSteps === true,
                  }
                : isGoalScenario
                  ? {
                      goalModeEnabledInGui:
                        summary.goalModeEnabled?.statusChipVisible === true &&
                        summary.goalModeEnabled?.statusText?.includes(
                          "追求目标",
                        ) === true,
                      goalPromptReachedBackend:
                        goalTurnStart?.inputText === GOAL_PROMPT,
                      goalObjectiveTextReachedBackend:
                        goalObjectiveText === GOAL_PROMPT,
                      guiGoalCompleted:
                        summary.guiGoalCompleted?.hasPrompt === true &&
                        (summary.guiGoalCompleted?.hasAssistantSummary ===
                          true ||
                          summary.guiGoalCompleted?.hasDoneText === true) &&
                        summary.guiGoalCompleted?.textareaVisible === true &&
                        summary.guiGoalCompleted?.textareaDisabled === false &&
                        summary.guiGoalCompleted?.stopButtonVisible === false,
                      readModelGoalCompleted:
                        summary.readModelGoalCompleted?.includesPrompt ===
                          true &&
                        (summary.readModelGoalCompleted
                          ?.includesAssistantDone === true ||
                          summary.readModelGoalCompleted
                            ?.includesAssistantSummary === true),
                    }
                  : isImageCommandScenario
                    ? {
                        imageCommandPromptReachedBackend:
                          imageCommandTurnStart?.inputText ===
                          expectedImageIntentRoutedPrompt,
                        imageCommandMetadataReachedBackend:
                          imageCommandContractKey === "image_generation" &&
                          imageCommandTurnStart?.providerPreference == null &&
                          imageCommandTurnStart?.modelPreference == null,
                        imageCommandLegacySkillLaunchNotSubmitted:
                          imageCommandHarness?.image_skill_launch == null &&
                          imageCommandHarness?.imageSkillLaunch == null,
                        imageCommandUsedCurrentMediaTaskArtifactMethods:
                          summary.imageCommandWorkflowUsed === true &&
                          appServerRequestMethods.includes(
                            APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_GET,
                          ) &&
                          appServerRequestMethods.includes(
                            APP_SERVER_METHOD_MEDIA_TASK_ARTIFACT_LIST,
                          ),
                        imageCommandTaskArtifactWritten:
                          summary.imageCommandTaskArtifact?.exists === true &&
                          summary.imageCommandTaskArtifact
                            ?.pathIncludesImageGenerate === true &&
                          typeof summary.imageCommandTaskArtifact?.taskId ===
                            "string" &&
                          summary.imageCommandTaskArtifact.taskId.length > 0,
                        imageCommandTaskArtifactReadable:
                          summary.imageCommandTaskArtifact?.getReturned ===
                            true &&
                          summary.imageCommandTaskArtifact?.listReturned ===
                            true &&
                          summary.imageCommandTaskArtifact?.listContainsTask ===
                            true &&
                          summary.imageCommandTaskArtifact
                            ?.listContainsImageGenerate === true,
                        imageCommandTaskArtifactTerminal:
                          summary.imageCommandTaskArtifactTerminalPatch
                            ?.status === "succeeded" &&
                          summary.imageCommandTaskArtifactTerminalPatch
                            ?.completeMethodUsed === "media_runtime_worker" &&
                          summary.imageCommandTaskArtifactTerminalPatch
                            ?.completeReturned === true &&
                          summary.imageCommandTaskArtifactTerminalPatch
                            ?.normalizedStatus === "succeeded" &&
                          summary.imageCommandTaskArtifactTerminalPatch
                            ?.resultImageCount === 1 &&
                          summary.imageCommandTaskArtifactTerminal
                            ?.fileNormalizedStatus === "succeeded" &&
                          summary.imageCommandTaskArtifactTerminal
                            ?.fileResultImageCount === 1 &&
                          summary.imageCommandTaskArtifactAfterReload
                            ?.fileNormalizedStatus === "succeeded" &&
                          summary.imageCommandTaskArtifactAfterReload
                            ?.listContainsTask === true,
                        imageCommandTaskArtifactSameTaskUpdated:
                          summary.imageCommandTaskArtifactTerminalPatch
                            ?.sameTaskFileUpdated === true &&
                          summary.imageCommandTaskArtifactTerminalPatch
                            ?.taskId ===
                            summary.imageCommandTaskArtifact?.taskId &&
                          summary.imageCommandTaskArtifactTerminalPatch
                            ?.taskPath ===
                            summary.imageCommandTaskArtifact?.taskPath &&
                          summary.imageCommandTaskArtifactTerminalPatch
                            ?.currentAttemptStatus === "succeeded" &&
                          summary.imageCommandTaskArtifactTerminalPatch
                            ?.currentAttemptWorkerId ===
                            "lime-image-api-worker" &&
                          summary.imageCommandTaskArtifactTerminalPatch
                            ?.currentAttemptHasResultSnapshot === true,
                        imageCommandTaskAuditLogWritten:
                          summary.imageCommandTaskAuditLog?.exists === true &&
                          summary.imageCommandTaskAuditLog?.lineCount >= 6 &&
                          summary.imageCommandTaskAuditLog
                            ?.logsRefLooksLikeTaskLog === true,
                        imageCommandTaskAuditLogEventSequence:
                          summary.imageCommandTaskAuditLog
                            ?.hasExpectedEventSequence === true &&
                          summary.imageCommandTaskAuditLog?.events?.[0] ===
                            "worker_loaded" &&
                          summary.imageCommandTaskAuditLog?.events?.at(-1) ===
                            "task_succeeded" &&
                          summary.imageCommandTaskAuditLog
                            ?.allEventTaskIdsMatch === true &&
                          summary.imageCommandTaskAuditLog?.parseError == null,
                        imageCommandTaskAuditLogNoSensitiveTokens:
                          summary.imageCommandTaskAuditLog
                            ?.hasNoSensitiveTokenMarkers === true &&
                          Array.isArray(
                            summary.imageCommandTaskAuditLog
                              ?.forbiddenMarkerHits,
                          ) &&
                          summary.imageCommandTaskAuditLog.forbiddenMarkerHits
                            .length === 0,
                        imageCommandWorkflowAuditReadModelProjected:
                          appServerRequestMethods.includes(
                            APP_SERVER_METHOD_WORKFLOW_READ,
                          ) &&
                          summary.imageCommandWorkflowRead?.sessionId ===
                            summary.sessionId &&
                          summary.imageCommandWorkflowRead?.matchedRun
                            ?.workflowKey === "image_command_workflow" &&
                          summary.imageCommandWorkflowRead?.matchedRun
                            ?.status === "completed" &&
                          summary.imageCommandWorkflowRead?.matchedRun
                            ?.stepCounts?.total === 5 &&
                          summary.imageCommandWorkflowRead
                            ?.activeWorkflowRunId === "",
                        imageCommandWorkflowAuditStepsProjected:
                          summary.imageCommandWorkflowRead?.hasExpectedSteps ===
                            true &&
                          summary.imageCommandWorkflowRead?.matchedStepIds?.includes(
                            "intent",
                          ) === true &&
                          summary.imageCommandWorkflowRead?.matchedStepIds?.includes(
                            "create_tasks",
                          ) === true &&
                          summary.imageCommandWorkflowRead?.completedStepIds?.includes(
                            "intent",
                          ) === true &&
                          summary.imageCommandWorkflowRead?.completedStepIds?.includes(
                            "create_tasks",
                          ) === true &&
                          summary.imageCommandWorkflowRead?.createTasksStep
                            ?.status === "completed",
                        imageCommandWorkflowAuditSummaryRedacted:
                          summary.imageCommandWorkflowRead?.containsPrompt ===
                            false &&
                          summary.imageCommandWorkflowRead?.containsTaskPath ===
                            false,
                        imageCommandWorkerUsedFixtureProviderAndModel:
                          (summary.imageCommandTaskCreateRequest?.provider_id ??
                            summary.imageCommandTaskCreateRequest
                              ?.providerId) ===
                            summary.imageFixtureProvider?.providerId &&
                          (summary.imageCommandTaskCreateRequest?.model ??
                            null) === IMAGE_FIXTURE_MODEL &&
                          summary.imageProviderFixtureServer?.requests?.some(
                            (request) =>
                              request?.method === "GET" &&
                              request?.url === "/v1/models" &&
                              request?.authorization === "present",
                          ) === true &&
                          summary.imageProviderFixtureServer?.requests?.find(
                            (request) =>
                              request?.method === "POST" &&
                              request?.url === "/v1/images/generations",
                          )?.headerProviderId ===
                            summary.imageFixtureProvider?.providerId &&
                          summary.imageProviderFixtureServer?.requests?.find(
                            (request) =>
                              request?.method === "POST" &&
                              request?.url === "/v1/images/generations",
                          )?.model === IMAGE_FIXTURE_MODEL &&
                          summary.imageProviderFixtureServer?.requests?.find(
                            (request) =>
                              request?.method === "POST" &&
                              request?.url === "/v1/images/generations",
                          )?.bodyIncludesModel === true,
                        imageCommandWorkflowToolObserved:
                          summary.readModelImageCommandCompleted
                            ?.includesWorkflowSource === true ||
                          pageText.includes("image_command_workflow") ||
                          (summary.imageCommandTaskCreateRequest
                            ?.runtimeContract?.executor_adapter_key ===
                            "workflow:image_command" &&
                            summary.imageCommandTaskCreateRequest
                              ?.runtimeContract?.executor_binding_key ===
                              "image_command"),
                        imageCommandCreateTaskToolObserved:
                          summary.readModelImageCommandCompleted
                            ?.includesCreateTaskTool === true,
                        guiImageCommandInputSubmitted:
                          summary.imageCommandInputSend?.afterFill
                            ?.promptVisibleInTextarea === true &&
                          summary.imageCommandInputSend?.clicked?.clicked ===
                            true,
                        guiImageCommandToolProcessVisible:
                          summary.guiImageCommandCompleted
                            ?.hasVisibleImageTaskProcess === true,
                        guiImageCommandTaskCardVisible:
                          summary.guiImageCommandCompleted
                            ?.imageTaskCardVisible === true,
                        guiImageCommandTaskCardTerminal:
                          summary.guiImageCommandTerminal?.cardCount === 1 &&
                          summary.guiImageCommandTerminal?.mediaCount >= 1 &&
                          summary.guiImageCommandTerminal
                            ?.hasPresentationIntro === true &&
                          summary.guiImageCommandTerminal
                            ?.hasPresentationIntroInAssistantText === true &&
                          summary.guiImageCommandTerminal?.hasToolStripLabel ===
                            true &&
                          summary.guiImageCommandTerminal
                            ?.hasImageModelLabel === true &&
                          summary.guiImageCommandTerminal?.hasTokenUsage ===
                            true &&
                          summary.guiImageCommandTerminal?.hasPreviewImage ===
                            true &&
                          summary.guiImageCommandTerminal
                            ?.hasLoadedVisiblePreviewImage === true &&
                          summary.guiImageCommandTerminal
                            ?.hasPresentationCaptionAfterCard === true &&
                          summary.guiImageCommandTerminal
                            ?.cardHasPresentationCaption === false &&
                          summary.guiImageCommandTerminal?.introBeforeCard ===
                            true &&
                          summary.guiImageCommandTerminal
                            ?.completionAfterCard === true &&
                          summary.guiImageCommandTerminal
                            ?.visiblePendingStatus === false,
                        guiImageCommandSingleTaskCard:
                          summary.guiImageCommandTerminal?.cardCount === 1 &&
                          summary.guiImageCommandRestoredAfterReload
                            ?.cardCount === 1,
                        guiImageCommandRestoredAfterReload:
                          summary.guiImageCommandReload?.renderer?.electron ===
                            true &&
                          summary.guiImageCommandReload?.session?.inputReady
                            ?.hasMessageList === true &&
                          summary.guiImageCommandRestoredAfterReload
                            ?.cardCount === 1 &&
                          summary.guiImageCommandRestoredAfterReload
                            ?.mediaCount >= 1 &&
                          summary.guiImageCommandRestoredAfterReload
                            ?.hasPresentationIntro === true &&
                          summary.guiImageCommandRestoredAfterReload
                            ?.hasPresentationIntroInAssistantText === true &&
                          summary.guiImageCommandRestoredAfterReload
                            ?.hasToolStripLabel === true &&
                          summary.guiImageCommandRestoredAfterReload
                            ?.hasImageModelLabel === true &&
                          summary.guiImageCommandRestoredAfterReload
                            ?.hasTokenUsage === true &&
                          summary.guiImageCommandRestoredAfterReload
                            ?.hasPreviewImage === true &&
                          summary.guiImageCommandRestoredAfterReload
                            ?.hasLoadedVisiblePreviewImage === true &&
                          summary.guiImageCommandRestoredAfterReload
                            ?.hasPresentationCaptionAfterCard === true &&
                          summary.guiImageCommandRestoredAfterReload
                            ?.cardHasPresentationCaption === false &&
                          summary.guiImageCommandRestoredAfterReload
                            ?.introBeforeCard === true &&
                          summary.guiImageCommandRestoredAfterReload
                            ?.completionAfterCard === true &&
                          summary.guiImageCommandRestoredAfterReload
                            ?.visiblePendingStatus === false,
                        guiImageCommandNoDraftCard:
                          summary.guiImageCommandCompleted
                            ?.draftImageVisible === false &&
                          summary.guiImageCommandTerminal?.draftImageVisible ===
                            false &&
                          summary.guiImageCommandRestoredAfterReload
                            ?.draftImageVisible === false &&
                          summary.readModelImageCommandCompleted
                            ?.includesDraftTask === false,
                        guiImageCommandNoTemplateTaskId:
                          summary.guiImageCommandCompleted
                            ?.templateTaskIdVisible === false &&
                          summary.guiImageCommandTerminal
                            ?.templateTaskIdVisible === false &&
                          summary.guiImageCommandRestoredAfterReload
                            ?.templateTaskIdVisible === false &&
                          summary.readModelImageCommandCompleted
                            ?.includesTaskIdPlaceholder === false,
                        readModelImageCommandCompleted:
                          summary.readModelImageCommandCompleted
                            ?.includesPrompt === true &&
                          summary.readModelImageCommandCompleted
                            ?.latestTurnStatus === "completed" &&
                          summary.readModelImageCommandCompleted
                            ?.includesCreateTaskTool === true,
                        readModelImageCommandTaskPreviewObserved:
                          summary.readModelImageCommandCompleted
                            ?.createTaskOutputContainsTaskId === true &&
                          summary.readModelImageCommandCompleted
                            ?.createTaskOutputContainsTaskFile === true,
                      }
                    : isWebToolsRenderingScenario
                      ? buildWebToolsRenderingScenarioAssertions({
                          summary,
                          webToolsRenderingTurnStart,
                        })
                      : isReasoningFirstVisibleScenario
                        ? buildReasoningFirstVisibleScenarioAssertions({
                            reasoningFirstVisibleTurnStart,
                            summary,
                          })
                        : isElectronResizeReflowScenario
                          ? buildElectronResizeReflowScenarioAssertions({
                              electronResizeReflowTurnStart,
                              summary,
                            })
                          : isLiveTailCommitScenario
                            ? buildLiveTailCommitScenarioAssertions({
                                liveTailCommitTurnStart,
                                summary,
                              })
                            : isApprovalRequestFullAccessScenario
                              ? buildApprovalRequestFullAccessScenarioAssertions(
                                  {
                                    appServerRequestMethods,
                                    approvalRequestFullAccessTurnStart,
                                    pageText,
                                    summary,
                                  },
                                )
                              : isApprovalRequestHostInterruptScenario
                                ? buildApprovalRequestHostInterruptScenarioAssertions(
                                    {
                                      appServerRequestMethods,
                                      approvalRequestResumeTurnStart,
                                      summary,
                                    },
                                  )
                                : isApprovalRequestResumeScenario
                                  ? buildApprovalRequestResumeScenarioAssertions(
                                      {
                                        appServerRequestMethods,
                                        approvalRequestResumeTurnStart,
                                        backendLedger,
                                        pageText,
                                        summary,
                                      },
                                    )
                                  : isApprovalRequestDecisionScenario
                                    ? buildApprovalRequestDecisionScenarioAssertions(
                                        {
                                          appServerRequestMethods,
                                          approvalRequestResumeTurnStart,
                                          backendLedger,
                                          isApprovalRequestCancelScenario,
                                          isApprovalRequestDeclineScenario,
                                          pageText,
                                          summary,
                                        },
                                      )
                                    : isTypedErrorRetryScenario
                                      ? buildTypedErrorScenarioAssertions({
                                          isTypedErrorRetryFailureScenario,
                                          isTypedErrorRetrySuccessScenario,
                                          summary,
                                          typedErrorTurnStart,
                                        })
                                      : isTerminalStaleGuardScenario ||
                                          isTerminalCanceledAfterAnswerScenario ||
                                          isTerminalFailedAfterAnswerScenario
                                        ? buildTerminalScenarioAssertions({
                                            isTerminalCanceledAfterAnswerScenario,
                                            isTerminalFailedAfterAnswerScenario,
                                            isTerminalStaleGuardScenario,
                                            summary,
                                            terminalCanceledAfterAnswerTurnStart,
                                            terminalFailedAfterAnswerTurnStart,
                                            terminalStaleGuardFirstTurnStart,
                                            terminalStaleGuardSecondTurnStart,
                                          })
                                        : isMcpStructuredContentScenario
                                          ? buildMcpStructuredContentScenarioAssertions(
                                              {
                                                mcpStructuredContentTurnStart,
                                                summary,
                                              },
                                            )
                                          : isMediaReferenceScenario
                                            ? buildMediaReferenceScenarioAssertions(
                                                {
                                                  mediaReferenceTurnStart,
                                                  pageText,
                                                  summary,
                                                },
                                              )
                                            : isSkillsRuntimeScenario
                                              ? buildSkillsRuntimeScenarioAssertions(
                                                  {
                                                    explicitSkillsRuntimeTurnStart,
                                                    manualEnableRuntimeBinding,
                                                    manualEnableRuntimeMetadata,
                                                    manualEnableSkillsRuntimeTurnStart,
                                                    skillsRuntimeTurnStart,
                                                    summary,
                                                    workspace,
                                                  },
                                                )
                                              : isAnyExpertSkillsRuntimeScenario
                                                ? buildExpertSkillsRuntimeScenarioAssertions(
                                                    {
                                                      expectedExpertHarnessSkillRef,
                                                      expertHarnessMetadata,
                                                      expertHarnessSkillRefs,
                                                      expertPanelSkillsRuntimeTurnStart,
                                                      expertRuntimeMetadata,
                                                      expertSkillsRuntimeTurnStart,
                                                      isExpertPanelSkillsRuntimeScenario,
                                                      isExpertPlazaSkillsRuntimeScenario,
                                                      summary,
                                                    },
                                                  )
                                                : hasCancelPhase
                                                  ? {
                                                      usedCurrentTurnCancel:
                                                        appServerRequestMethods.includes(
                                                          APP_SERVER_METHOD_SESSION_TURN_CANCEL,
                                                        ),
                                                      externalFixtureCancelUsed:
                                                        backendLedger.some(
                                                          (entry) =>
                                                            entry.kind ===
                                                            "turnCancel",
                                                        ),
                                                      fixtureCancelReachedBackend:
                                                        latestTurnCancel?.sessionId ===
                                                          summary.sessionId &&
                                                        typeof latestTurnCancel?.turnId ===
                                                          "string" &&
                                                        latestTurnCancel.turnId.trim()
                                                          .length > 0,
                                                      guiStopClicked:
                                                        summary.stopClick
                                                          ?.clicked?.clicked ===
                                                        true,
                                                      readModelCanceled:
                                                        summary
                                                          .readModelCanceled
                                                          ?.includesPrompt ===
                                                          true &&
                                                        summary
                                                          .readModelCanceled
                                                          ?.hasInterruptedTurn ===
                                                          true,
                                                      ...(isCancelThenContinueScenario
                                                        ? {
                                                            continuePromptReachedBackend:
                                                              continueTurnStart?.inputText ===
                                                              CONTINUE_PROMPT,
                                                            guiContinueInputSubmitted:
                                                              summary
                                                                .continueInputSend
                                                                ?.afterFill
                                                                ?.promptVisibleInTextarea ===
                                                                true &&
                                                              summary
                                                                .continueInputSend
                                                                ?.clicked
                                                                ?.clicked ===
                                                                true,
                                                            guiContinueCompleted:
                                                              summary
                                                                .guiContinueCompleted
                                                                ?.hasPrompt ===
                                                                true &&
                                                              (summary
                                                                .guiContinueCompleted
                                                                ?.hasAssistantSummary ===
                                                                true ||
                                                                summary
                                                                  .guiContinueCompleted
                                                                  ?.hasDoneText ===
                                                                  true) &&
                                                              summary
                                                                .guiContinueCompleted
                                                                ?.textareaVisible ===
                                                                true &&
                                                              summary
                                                                .guiContinueCompleted
                                                                ?.textareaDisabled ===
                                                                false &&
                                                              summary
                                                                .guiContinueCompleted
                                                                ?.stopButtonVisible ===
                                                                false,
                                                            readModelContinueCompleted:
                                                              summary
                                                                .readModelContinueCompleted
                                                                ?.includesPrompt ===
                                                                true &&
                                                              (summary
                                                                .readModelContinueCompleted
                                                                ?.includesAssistantDone ===
                                                                true ||
                                                                summary
                                                                  .readModelContinueCompleted
                                                                  ?.includesAssistantSummary ===
                                                                  true),
                                                            backendRecordedCancelThenContinue:
                                                              backendLedger.filter(
                                                                (entry) =>
                                                                  entry.kind ===
                                                                  "turnStart",
                                                              ).length >= 2 &&
                                                              backendLedger.some(
                                                                (entry) =>
                                                                  entry.kind ===
                                                                  "turnCancel",
                                                              ),
                                                          }
                                                        : {}),
                                                    }
                                                  : isInputbarRichRestoreScenario
                                                    ? {
                                                        inputbarRichRestorePromptReachedBackend:
                                                          String(
                                                            inputbarRichRestoreTurnStart?.inputText ||
                                                              "",
                                                          ).includes(
                                                            INPUTBAR_RICH_RESTORE_PROMPT,
                                                          ),
                                                        inputbarRichRestoreDraftPrepared:
                                                          summary
                                                            .inputbarRichRestoreDraftPrepared
                                                            ?.prepared
                                                            ?.imageRestored ===
                                                            true &&
                                                          summary
                                                            .inputbarRichRestoreDraftPrepared
                                                            ?.prepared
                                                            ?.pathRestored ===
                                                            true &&
                                                          summary
                                                            .inputbarRichRestoreDraftPrepared
                                                            ?.prepared
                                                            ?.skillRestored ===
                                                            true,
                                                        inputbarRichRestoreInputSubmitted:
                                                          summary
                                                            .inputbarRichRestoreInputSend
                                                            ?.afterFill
                                                            ?.promptVisibleInTextarea ===
                                                            true &&
                                                          summary
                                                            .inputbarRichRestoreInputSend
                                                            ?.clicked
                                                            ?.clicked === true,
                                                        inputbarRichRestoreBackendInputSummaryReached:
                                                          summary
                                                            .inputbarRichRestoreBackendTurnStart
                                                            ?.inputSummary
                                                            ?.imageAttachmentCount >=
                                                            1 &&
                                                          summary
                                                            .inputbarRichRestoreBackendTurnStart
                                                            ?.inputSummary
                                                            ?.fileReferenceCount >=
                                                            1 &&
                                                          summary.inputbarRichRestoreBackendTurnStart?.inputSummary?.fileReferenceNames?.includes(
                                                            INPUTBAR_RICH_RESTORE_PATH_NAME,
                                                          ) === true,
                                                        inputbarRichRestoreUsedCurrentTurnCancel:
                                                          appServerRequestMethods.includes(
                                                            APP_SERVER_METHOD_SESSION_TURN_CANCEL,
                                                          ),
                                                        inputbarRichRestoreBackendCanceled:
                                                          latestTurnCancel?.sessionId ===
                                                            inputbarRichRestoreTurnStart?.sessionId &&
                                                          latestTurnCancel?.turnId ===
                                                            inputbarRichRestoreTurnStart?.turnId &&
                                                          typeof latestTurnCancel?.turnId ===
                                                            "string" &&
                                                          latestTurnCancel.turnId.trim()
                                                            .length > 0,
                                                        inputbarRichRestoreGuiCanceled:
                                                          summary
                                                            .inputbarRichRestoreGuiCanceled
                                                            ?.stopButtonVisible ===
                                                            false &&
                                                          summary
                                                            .inputbarRichRestoreGuiCanceled
                                                            ?.textareaDisabled ===
                                                            false,
                                                        inputbarRichRestoreTextRestored:
                                                          summary
                                                            .inputbarRichRestoreGuiCanceled
                                                            ?.textareaValue ===
                                                          INPUTBAR_RICH_RESTORE_PROMPT,
                                                        inputbarRichRestoreImageRestored:
                                                          summary
                                                            .inputbarRichRestoreGuiCanceled
                                                            ?.imageRestored ===
                                                          true,
                                                        inputbarRichRestorePathRestored:
                                                          summary
                                                            .inputbarRichRestoreGuiCanceled
                                                            ?.pathRestored ===
                                                          true,
                                                        inputbarRichRestoreSkillRestored:
                                                          summary
                                                            .inputbarRichRestoreGuiCanceled
                                                            ?.skillRestored ===
                                                          true,
                                                        inputbarRichRestoreNoVisibleAssistantOutput:
                                                          summary
                                                            .inputbarRichRestoreGuiCanceled
                                                            ?.noVisibleAssistantOutput ===
                                                          true,
                                                        inputbarRichRestoreReadModelCanceled:
                                                          summary
                                                            .inputbarRichRestoreReadModelCanceled
                                                            ?.includesPrompt ===
                                                            true &&
                                                          summary
                                                            .inputbarRichRestoreReadModelCanceled
                                                            ?.includesCanceled ===
                                                            true &&
                                                          summary
                                                            .inputbarRichRestoreReadModelCanceled
                                                            ?.forbiddenAssistantOutput ===
                                                            false,
                                                      }
                                                    : {
                                                        noEpochFallbackTitle:
                                                          summary.guiCompleted
                                                            ?.hasEpochFallbackTitle ===
                                                          false,
                                                        readModelCompleted:
                                                          summary
                                                            .readModelCompleted
                                                            ?.includesPrompt ===
                                                            true &&
                                                          (summary
                                                            .readModelCompleted
                                                            ?.includesAssistantDone ===
                                                            true ||
                                                            summary
                                                              .readModelCompleted
                                                              ?.includesAssistantSummary ===
                                                              true),
                                                        eventReadProbeObserved:
                                                          summary.eventReadProbe
                                                            ?.events
                                                            ?.hasTextDelta ===
                                                            true &&
                                                          summary.eventReadProbe
                                                            ?.events
                                                            ?.hasToolStarted ===
                                                            true &&
                                                          summary.eventReadProbe
                                                            ?.events
                                                            ?.hasToolResult ===
                                                            true &&
                                                          summary.eventReadProbe
                                                            ?.events
                                                            ?.hasTerminal ===
                                                            true &&
                                                          summary.eventReadProbe
                                                            ?.events
                                                            ?.eventTurnIds
                                                            ?.length === 1 &&
                                                          summary.eventReadProbe
                                                            ?.events
                                                            ?.eventTurnIds?.[0] ===
                                                            summary
                                                              .eventReadProbe
                                                              ?.turnId,
                                                        readModelEventReadAligned:
                                                          summary.eventReadProbe
                                                            ?.readModel
                                                            ?.containsTurnId ===
                                                            true &&
                                                          summary.eventReadProbe
                                                            ?.readModel
                                                            ?.containsReadText ===
                                                            true,
                                                        readModelToolCallAligned:
                                                          summary.eventReadProbe
                                                            ?.readModel
                                                            ?.containsToolCall ===
                                                            true &&
                                                          summary.eventReadProbe
                                                            ?.readModel
                                                            ?.toolName ===
                                                            EVENT_READ_PROBE_TOOL_NAME &&
                                                          summary.eventReadProbe
                                                            ?.readModel
                                                            ?.toolStatus ===
                                                            "completed" &&
                                                          summary.eventReadProbe
                                                            ?.readModel
                                                            ?.containsToolOutput ===
                                                            true &&
                                                          summary.eventReadProbe
                                                            ?.readModel
                                                            ?.toolTurnId ===
                                                            summary
                                                              .eventReadProbe
                                                              ?.turnId,
                                                        guiNoPlanUiWithoutProposedPlan:
                                                          summary.guiCompleted
                                                            ?.planUiAbsentWithoutProposedPlan ===
                                                            true &&
                                                          summary.guiCompleted
                                                            ?.planUiAbsence
                                                            ?.planOwnerCount ===
                                                            0 &&
                                                          summary.guiCompleted
                                                            ?.planUiAbsence
                                                            ?.planDecisionVisible ===
                                                            false &&
                                                          (summary.guiCompleted
                                                            ?.planUiAbsence
                                                            ?.legacyUpdatePlanVisibleHits
                                                            ?.length ?? 0) ===
                                                            0,
                                                        streamParserCompletedFullTextObserved:
                                                          streamParserBoundaryBackendObserved(
                                                            backendLedger,
                                                          ),
                                                        guiStreamParserNoDuplicateFinalText:
                                                          summary.guiCompleted
                                                            ?.assistantScopeSummaryOccurrences ===
                                                            1 &&
                                                          (
                                                            summary.guiCompleted
                                                              ?.assistantScopeDedupeGuardHits ??
                                                            []
                                                          ).every(
                                                            (hit) =>
                                                              hit.occurrences ===
                                                              1,
                                                          ),
                                                        readModelStreamParserNoDuplicateFinalText:
                                                          summary
                                                            .readModelCompleted
                                                            ?.streamParserBoundary
                                                            ?.noDuplicateFinalText ===
                                                          true,
                                                      };
  const userShellGate = summary.userShellGate;
  const userShellAssertions = userShellGate
    ? {
        userShellGuiInputSubmitted:
          userShellGate.assertions?.guiInputSubmitted === true,
        userShellUsesCurrentMethod: appServerRequestMethods.includes(
          APP_SERVER_METHOD_THREAD_SHELL_COMMAND,
        ),
        userShellCommandItemStarted:
          userShellGate.assertions?.commandItemStarted === true,
        userShellCommandItemCompleted:
          userShellGate.assertions?.commandItemCompleted === true,
        userShellCommandOutputVisible:
          userShellGate.assertions?.commandOutputVisible === true,
        userShellCommandOutputPersisted:
          userShellGate.assertions?.commandOutputPersisted === true,
        userShellCommandExitCodeZero:
          userShellGate.assertions?.commandExitCodeZero === true,
        userShellCommandSourceCurrent:
          userShellGate.assertions?.commandSourceUserShell === true,
        userShellIdentityConsistent:
          userShellGate.assertions?.identityConsistent === true,
        userShellInputReadyAfterCompletion:
          userShellGate.assertions?.inputReadyAfterCompletion === true,
      }
    : {};
  return { ...scenarioAssertions, ...userShellAssertions };
}
