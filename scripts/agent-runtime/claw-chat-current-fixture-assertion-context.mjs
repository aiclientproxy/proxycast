import {
  APPROVAL_REQUEST_CANCEL_SCENARIO,
  APPROVAL_REQUEST_DECLINE_SCENARIO,
  APPROVAL_REQUEST_FULL_ACCESS_PROMPT,
  APPROVAL_REQUEST_FULL_ACCESS_SCENARIO,
  APPROVAL_REQUEST_HOST_INTERRUPT_SCENARIO,
  APPROVAL_REQUEST_RESUME_PROMPT,
  APPROVAL_REQUEST_RESUME_SCENARIO,
  CONTINUE_PROMPT,
  CONTENT_FACTORY_ARTICLE_WORKSPACE_SCENARIO,
  CONTENT_FACTORY_INLINE_IMAGE_ARTICLE_WORKSPACE_SCENARIO,
  EXPERT_SKILLS_RUNTIME_PANEL_PROMPT,
  EXPERT_SKILLS_RUNTIME_PROMPT,
  EXPERT_SKILLS_RUNTIME_SKILL_REF,
  ELECTRON_RESIZE_REFLOW_SCENARIO,
  GOAL_PROMPT,
  HOME_HOTPATH_GREETING_SCENARIO,
  HOME_HOTPATH_SCENARIO,
  IMAGE_COMMAND_PROMPT,
  IMAGE_COMMAND_SCENARIO,
  INPUTBAR_RICH_RESTORE_PROMPT,
  INPUTBAR_RICH_RESTORE_SCENARIO,
  LIVE_TAIL_COMMIT_PROMPT,
  LIVE_TAIL_COMMIT_SCENARIO,
  MCP_STRUCTURED_CONTENT_PROMPT,
  NEWS_PROMPT,
  PLAIN_IMAGE_INTENT_ROUTED_PROMPT,
  PLAIN_IMAGE_INTENT_SCENARIO,
  PLAN_PROMPT,
  REASONING_FIRST_VISIBLE_PROMPT,
  REASONING_FIRST_VISIBLE_SCENARIO,
  RIGHT_SURFACE_VISUAL_MATRIX_SCENARIO,
  SOUL_STYLE_SCENARIO,
  SKILLS_RUNTIME_EXPLICIT_PROMPT,
  SKILLS_RUNTIME_MANUAL_ENABLE_PROMPT,
  SKILLS_RUNTIME_PROMPT,
  TERMINAL_CANCELED_AFTER_ANSWER_PROMPT,
  TERMINAL_CANCELED_AFTER_ANSWER_SCENARIO,
  TERMINAL_FAILED_AFTER_ANSWER_PROMPT,
  TERMINAL_FAILED_AFTER_ANSWER_SCENARIO,
  TERMINAL_STALE_GUARD_FIRST_PROMPT,
  TERMINAL_STALE_GUARD_SCENARIO,
  TERMINAL_STALE_GUARD_SECOND_PROMPT,
  TYPED_ERROR_RETRY_FAILURE_PROMPT,
  TYPED_ERROR_RETRY_FAILURE_SCENARIO,
  TYPED_ERROR_RETRY_SUCCESS_PROMPT,
  TYPED_ERROR_RETRY_SUCCESS_SCENARIO,
  WEB_TOOLS_RENDERING_PROMPT,
} from "./claw-chat-current-fixture-constants.mjs";
import { EXPERT_PANEL_SKILLS_RUNTIME_UI_SKILL_REF } from "./claw-chat-current-fixture-expert-actions.mjs";
import {
  collectTraceRequestMethods,
  decodeJsonRpcLines,
} from "./claw-chat-current-fixture-rpc.mjs";
import { buildGateBContractEvidence } from "./claw-chat-current-fixture-gate-b-contract.mjs";
import {
  readHarnessMetadataFromTurnStart,
  readObjectiveTextFromHarness,
  readWorkspaceSkillRuntimeEnableFromTurnStart,
} from "./claw-chat-current-fixture-backend-ledger.mjs";
import {
  MEDIA_REFERENCE_PROMPT,
  MEDIA_REFERENCE_SCENARIO,
} from "./claw-chat-current-fixture-media-reference.mjs";
import {
  ACTIVE_STEER_INITIAL_PROMPT,
  ACTIVE_STEER_SCENARIO,
} from "./claw-chat-current-fixture-active-steer.mjs";
import { UNKNOWN_ITEM_SCENARIO } from "./claw-chat-current-fixture-unknown-item.mjs";

export function resolveGateBExpectedIdentity({
  summary,
  options,
  backendLedger,
  appServerRequests,
}) {
  if (
    options?.scenario === CONTENT_FACTORY_ARTICLE_WORKSPACE_SCENARIO ||
    options?.scenario === CONTENT_FACTORY_INLINE_IMAGE_ARTICLE_WORKSPACE_SCENARIO
  ) {
    const identity =
      summary?.contentFactoryArticleWorkspaceSessionCreation?.identity;
    const sessionId =
      typeof identity?.sessionId === "string" ? identity.sessionId.trim() : "";
    const threadId =
      typeof identity?.threadId === "string" ? identity.threadId.trim() : "";
    if (!sessionId || !threadId) {
      throw new Error(
        "Gate B Content Factory identity requires the scenario-created sessionId and threadId",
      );
    }
    return { sessionId, threadId };
  }
  if (options?.scenario === ACTIVE_STEER_SCENARIO) {
    const turnId = summary?.activeSteer?.activeTurnId ?? null;
    if (!summary?.sessionId || !summary?.threadId || !turnId) {
      throw new Error(
        "Gate B active steer identity requires sessionId, threadId and activeTurnId",
      );
    }
    return {
      sessionId: summary.sessionId,
      threadId: summary.threadId,
      turnId,
    };
  }
  const isHomeHotpathScenario =
    options?.scenario === HOME_HOTPATH_SCENARIO ||
    options?.scenario === HOME_HOTPATH_GREETING_SCENARIO;
  if (!isHomeHotpathScenario) {
    const scenarioTurnStart = resolveScenarioTurnStart(summary, options);
    if (scenarioTurnStart) {
      return resolveTurnIdentity({
        sessionId: scenarioTurnStart.sessionId,
        turnId: scenarioTurnStart.turnId,
        requireBackendTurn: scenarioTurnStart.requireBackendTurn,
        backendLedger,
        appServerRequests,
      });
    }
    return {
      sessionId: summary?.sessionId ?? null,
      threadId: summary?.threadId ?? null,
    };
  }

  const sessionId = summary?.homeHotpath?.backendTurnStart?.sessionId ?? null;
  const turnId = summary?.homeHotpath?.backendTurnStart?.turnId ?? null;
  return resolveTurnIdentity({
    sessionId,
    turnId,
    backendLedger,
    appServerRequests,
  });
}

function resolveScenarioTurnStart(summary, options) {
  if (
    options?.scenario === TYPED_ERROR_RETRY_SUCCESS_SCENARIO ||
    options?.scenario === TYPED_ERROR_RETRY_FAILURE_SCENARIO
  ) {
    return summary?.typedErrorTurnStart ?? null;
  }
  if (options?.scenario === "skills-runtime") {
    return summary?.manualEnableSkillsRuntimeTurnStart?.backend ?? null;
  }
  if (options?.scenario === "expert-panel-skills-runtime") {
    return summary?.expertPanelSkillsRuntimeTurnStart ?? null;
  }
  if (options?.scenario === "expert-plaza-skills-runtime") {
    return summary?.expertSkillsRuntimeTurnStart ?? null;
  }
  return null;
}

function resolveTurnIdentity({
  sessionId,
  turnId,
  requireBackendTurn = true,
  backendLedger,
  appServerRequests,
}) {
  const matchingBackendTurnStart = (
    Array.isArray(backendLedger) ? backendLedger : []
  ).find(
    (entry) =>
      entry?.kind === "turnStart" &&
      entry?.sessionId === sessionId &&
      entry?.turnId === turnId,
  );
  const matchingSessionRead = (
    Array.isArray(appServerRequests) ? appServerRequests : []
  ).find(
    (request) =>
      request?.method === "thread/read" &&
      request?.params?.threadId === request?.response?.threadId &&
      request?.response?.sessionId === sessionId &&
      request?.response?.turns?.some((turn) => turn?.turnId === turnId) ===
        true &&
      typeof request?.response?.threadId === "string" &&
      request.response.threadId.trim().length > 0,
  );
  const matchingAppServerTurnStart = (
    Array.isArray(appServerRequests) ? appServerRequests : []
  ).find(
    (request) =>
      request?.method === "turn/start" &&
      request?.params?.threadId === matchingSessionRead?.response?.threadId &&
      request?.response?.turnId === turnId,
  );
  const turnStartObserved = requireBackendTurn
    ? Boolean(matchingBackendTurnStart)
    : Boolean(matchingBackendTurnStart || matchingAppServerTurnStart);
  if (!sessionId || !turnId || !turnStartObserved || !matchingSessionRead) {
    throw new Error(
      "Gate B scenario identity requires matching backend turnStart and thread/read evidence for one session and turn",
    );
  }

  return {
    sessionId,
    threadId: matchingSessionRead.response.threadId,
    turnId,
  };
}

export function readTraceTurnStartInputText(input) {
  if (input && typeof input === "object" && !Array.isArray(input)) {
    if (typeof input.text === "string") {
      return input.text;
    }
    if (typeof input.content === "string") {
      return input.content;
    }
  }

  if (!Array.isArray(input)) {
    return null;
  }

  const text = input
    .filter((part) => part && typeof part === "object")
    .map((part) => (typeof part.text === "string" ? part.text : ""))
    .filter(Boolean)
    .join("\n");
  return text || null;
}

export function buildAssertionContext({
  backendLedger,
  traceMessages,
  appServerRequests,
  rendererSnapshot,
  summary,
  pageText,
  errorRaw,
  actionableConsoleErrors,
  workspace,
  options,
}) {
  const traceTurnStarts = traceMessages
    .filter((entry) => entry?.command === "app_server_handle_json_lines")
    .flatMap((entry) =>
      decodeJsonRpcLines(entry?.args_preview?.request?.lines)
        .filter((message) => message?.method === "turn/start")
        .map((message) => ({
          transport: entry.transport ?? null,
          status: entry.status ?? null,
          sessionId: message.params?.sessionId ?? message.params?.session_id,
          turnId: message.params?.turnId ?? message.params?.turn_id,
          inputText: readTraceTurnStartInputText(message.params?.input),
        })),
    );
  const appServerRequestMethods = Array.from(
    new Set(
      [
        ...appServerRequests.map((request) => request.method),
        ...collectTraceRequestMethods(traceMessages),
      ].filter(Boolean),
    ),
  );
  const gateBExpectedIdentity = resolveGateBExpectedIdentity({
    summary,
    options,
    backendLedger,
    appServerRequests,
  });
  const gateBContract = buildGateBContractEvidence({
    traceMessages,
    rendererSnapshot,
    pageErrors: summary.pageErrors,
    pageLifecycleEvents: summary.pageLifecycleEvents,
    runId: summary.runId,
    artifacts: {
      summary: summary.summary,
      backendLedger: summary.backendLedger,
      screenshot: summary.screenshot,
    },
    appServerRequests,
    backendLedger,
    guiEvidence: summary.gateBGuiEvidence,
    expectedIdentity: gateBExpectedIdentity,
  });
  const latestTurnStart = backendLedger
    .filter((entry) => entry.kind === "turnStart")
    .at(-1);
  const planImplementationTurnStart = backendLedger.find(
    (entry) =>
      entry.kind === "turnStart" && entry.inputText === "Implement the plan.",
  );
  const newsTurnStart = backendLedger.find(
    (entry) => entry.kind === "turnStart" && entry.inputText === NEWS_PROMPT,
  );
  const newsTraceTurnStart = traceTurnStarts.find(
    (entry) =>
      entry.inputText === NEWS_PROMPT &&
      entry.transport === "electron-ipc" &&
      entry.status === "success",
  );
  const homeHotpathPrompt =
    typeof summary?.homeHotpath?.prompt === "string" &&
    summary.homeHotpath.prompt.trim()
      ? summary.homeHotpath.prompt
      : NEWS_PROMPT;
  const homeHotpathTurnStart = backendLedger.find(
    (entry) =>
      entry.kind === "turnStart" && entry.inputText === homeHotpathPrompt,
  );
  const homeHotpathTraceTurnStart = traceTurnStarts.find(
    (entry) =>
      entry.inputText === homeHotpathPrompt &&
      entry.transport === "electron-ipc" &&
      entry.status === "success",
  );
  const planTurnStart = backendLedger.find(
    (entry) => entry.kind === "turnStart" && entry.inputText === PLAN_PROMPT,
  );
  const goalTurnStart = backendLedger.find(
    (entry) => entry.kind === "turnStart" && entry.inputText === GOAL_PROMPT,
  );
  const inputbarRichRestoreTurnStart = backendLedger.find(
    (entry) =>
      entry.kind === "turnStart" &&
      String(entry.inputText || "").includes(INPUTBAR_RICH_RESTORE_PROMPT),
  );
  const expectedImageIntentRoutedPrompt =
    options.scenario === PLAIN_IMAGE_INTENT_SCENARIO
      ? PLAIN_IMAGE_INTENT_ROUTED_PROMPT
      : IMAGE_COMMAND_PROMPT;
  const imageCommandWorkflowTurnStart =
    summary.imageCommandWorkflowTurnStart ??
    summary.imageCommandBackendTurnStart ??
    null;
  const imageCommandTurnStart =
    backendLedger.find(
      (entry) =>
        entry.kind === "turnStart" &&
        entry.inputText === expectedImageIntentRoutedPrompt,
    ) ?? imageCommandWorkflowTurnStart;
  const webToolsRenderingTurnStart = backendLedger.find(
    (entry) =>
      entry.kind === "turnStart" &&
      entry.inputText === WEB_TOOLS_RENDERING_PROMPT,
  );
  const reasoningFirstVisibleTurnStart = backendLedger.find(
    (entry) =>
      entry.kind === "turnStart" &&
      entry.inputText === REASONING_FIRST_VISIBLE_PROMPT,
  );
  const liveTailCommitTurnStart = backendLedger.find(
    (entry) =>
      entry.kind === "turnStart" && entry.inputText === LIVE_TAIL_COMMIT_PROMPT,
  );
  const electronResizeReflowTurnStart = liveTailCommitTurnStart;
  const approvalRequestResumeTurnStart = backendLedger.find(
    (entry) =>
      entry.kind === "turnStart" &&
      entry.inputText === APPROVAL_REQUEST_RESUME_PROMPT,
  );
  const approvalRequestFullAccessTurnStart = backendLedger.find(
    (entry) =>
      entry.kind === "turnStart" &&
      entry.inputText === APPROVAL_REQUEST_FULL_ACCESS_PROMPT,
  );
  const terminalCanceledAfterAnswerTurnStart = backendLedger.find(
    (entry) =>
      entry.kind === "turnStart" &&
      entry.inputText === TERMINAL_CANCELED_AFTER_ANSWER_PROMPT,
  );
  const terminalFailedAfterAnswerTurnStart = backendLedger.find(
    (entry) =>
      entry.kind === "turnStart" &&
      entry.inputText === TERMINAL_FAILED_AFTER_ANSWER_PROMPT,
  );
  const terminalStaleGuardFirstTurnStart = backendLedger.find(
    (entry) =>
      entry.kind === "turnStart" &&
      entry.inputText === TERMINAL_STALE_GUARD_FIRST_PROMPT,
  );
  const terminalStaleGuardSecondTurnStart = backendLedger.find(
    (entry) =>
      entry.kind === "turnStart" &&
      entry.inputText === TERMINAL_STALE_GUARD_SECOND_PROMPT,
  );
  const typedErrorTurnStart = backendLedger.find(
    (entry) =>
      entry.kind === "turnStart" &&
      (entry.inputText === TYPED_ERROR_RETRY_SUCCESS_PROMPT ||
        entry.inputText === TYPED_ERROR_RETRY_FAILURE_PROMPT),
  );
  const mcpStructuredContentTurnStart = backendLedger.find(
    (entry) =>
      entry.kind === "turnStart" &&
      entry.inputText === MCP_STRUCTURED_CONTENT_PROMPT,
  );
  const mediaReferenceTurnStart = backendLedger.find(
    (entry) =>
      entry.kind === "turnStart" && entry.inputText === MEDIA_REFERENCE_PROMPT,
  );
  const skillsRuntimeTurnStart = backendLedger.find(
    (entry) =>
      entry.kind === "turnStart" && entry.inputText === SKILLS_RUNTIME_PROMPT,
  );
  const explicitSkillsRuntimeTurnStart = backendLedger.find(
    (entry) =>
      entry.kind === "turnStart" &&
      entry.inputText === SKILLS_RUNTIME_EXPLICIT_PROMPT,
  );
  const manualEnableSkillsRuntimeTurnStart = backendLedger.find(
    (entry) =>
      entry.kind === "turnStart" &&
      entry.inputText === SKILLS_RUNTIME_MANUAL_ENABLE_PROMPT,
  );
  const expertSkillsRuntimeTurnStart = backendLedger.find(
    (entry) =>
      entry.kind === "turnStart" &&
      String(entry.inputText || "").includes(EXPERT_SKILLS_RUNTIME_PROMPT),
  );
  const expertPanelSkillsRuntimeTurnStart = backendLedger.find(
    (entry) =>
      entry.kind === "turnStart" &&
      entry.inputText === EXPERT_SKILLS_RUNTIME_PANEL_PROMPT,
  );
  const continueTurnStart = backendLedger.find(
    (entry) =>
      entry.kind === "turnStart" && entry.inputText === CONTINUE_PROMPT,
  );
  const latestTurnCancel = backendLedger
    .filter((entry) => entry.kind === "turnCancel")
    .at(-1);
  const isCancelOnlyScenario = options.scenario === "cancel";
  const isCancelThenContinueScenario =
    options.scenario === "cancel-then-continue";
  const isPlanScenario = options.scenario === "plan";
  const isGoalScenario = options.scenario === "goal";
  const isHomeHotpathScenario =
    options.scenario === HOME_HOTPATH_SCENARIO ||
    options.scenario === HOME_HOTPATH_GREETING_SCENARIO;
  const isImageCommandScenario =
    options.scenario === IMAGE_COMMAND_SCENARIO ||
    options.scenario === PLAIN_IMAGE_INTENT_SCENARIO;
  const isInputbarRichRestoreScenario =
    options.scenario === INPUTBAR_RICH_RESTORE_SCENARIO;
  const isActiveSteerScenario = options.scenario === ACTIVE_STEER_SCENARIO;
  const isWebToolsRenderingScenario =
    options.scenario === "web-tools-rendering";
  const isReasoningFirstVisibleScenario =
    options.scenario === REASONING_FIRST_VISIBLE_SCENARIO;
  const isLiveTailCommitScenario =
    options.scenario === LIVE_TAIL_COMMIT_SCENARIO;
  const isElectronResizeReflowScenario =
    options.scenario === ELECTRON_RESIZE_REFLOW_SCENARIO;
  const isApprovalRequestResumeScenario =
    options.scenario === APPROVAL_REQUEST_RESUME_SCENARIO;
  const isApprovalRequestDeclineScenario =
    options.scenario === APPROVAL_REQUEST_DECLINE_SCENARIO;
  const isApprovalRequestHostInterruptScenario =
    options.scenario === APPROVAL_REQUEST_HOST_INTERRUPT_SCENARIO;
  const isApprovalRequestCancelScenario =
    options.scenario === APPROVAL_REQUEST_CANCEL_SCENARIO ||
    isApprovalRequestHostInterruptScenario;
  const isApprovalRequestFullAccessScenario =
    options.scenario === APPROVAL_REQUEST_FULL_ACCESS_SCENARIO;
  const isApprovalRequestDecisionScenario =
    isApprovalRequestDeclineScenario || isApprovalRequestCancelScenario;
  const isTerminalCanceledAfterAnswerScenario =
    options.scenario === TERMINAL_CANCELED_AFTER_ANSWER_SCENARIO;
  const isTerminalFailedAfterAnswerScenario =
    options.scenario === TERMINAL_FAILED_AFTER_ANSWER_SCENARIO;
  const isTerminalStaleGuardScenario =
    options.scenario === TERMINAL_STALE_GUARD_SCENARIO;
  const isTypedErrorRetrySuccessScenario =
    options.scenario === TYPED_ERROR_RETRY_SUCCESS_SCENARIO;
  const isTypedErrorRetryFailureScenario =
    options.scenario === TYPED_ERROR_RETRY_FAILURE_SCENARIO;
  const isTypedErrorRetryScenario =
    isTypedErrorRetrySuccessScenario || isTypedErrorRetryFailureScenario;
  const isMcpStructuredContentScenario =
    options.scenario === "mcp-structured-content";
  const isMediaReferenceScenario =
    options.scenario === MEDIA_REFERENCE_SCENARIO;
  const isSkillsRuntimeScenario = options.scenario === "skills-runtime";
  const isUnknownItemScenario = options.scenario === UNKNOWN_ITEM_SCENARIO;
  const isSoulStyleScenario = options.scenario === SOUL_STYLE_SCENARIO;
  const isExpertPlazaSkillsRuntimeScenario =
    options.scenario === "expert-plaza-skills-runtime";
  const isExpertPanelSkillsRuntimeScenario =
    options.scenario === "expert-panel-skills-runtime";
  const isRightSurfaceVisualMatrixScenario =
    options.scenario === RIGHT_SURFACE_VISUAL_MATRIX_SCENARIO;
  const isContentFactoryInlineImageArticleWorkspaceScenario =
    options.scenario ===
    CONTENT_FACTORY_INLINE_IMAGE_ARTICLE_WORKSPACE_SCENARIO;
  const isContentFactoryArticleWorkspaceScenario =
    options.scenario === CONTENT_FACTORY_ARTICLE_WORKSPACE_SCENARIO ||
    isContentFactoryInlineImageArticleWorkspaceScenario;
  const isAnyExpertSkillsRuntimeScenario =
    isExpertPlazaSkillsRuntimeScenario || isExpertPanelSkillsRuntimeScenario;
  const expertRuntimeTurnStartForAssertions = isExpertPanelSkillsRuntimeScenario
    ? expertPanelSkillsRuntimeTurnStart
    : expertSkillsRuntimeTurnStart;
  const runtimeRequest =
    (isActiveSteerScenario
      ? {}
      : isPlanScenario
        ? planTurnStart?.runtimeRequest
        : isGoalScenario
          ? goalTurnStart?.runtimeRequest
          : isImageCommandScenario
            ? imageCommandTurnStart?.runtimeRequest
            : isInputbarRichRestoreScenario
              ? inputbarRichRestoreTurnStart?.runtimeRequest
              : isWebToolsRenderingScenario
                ? webToolsRenderingTurnStart?.runtimeRequest
                : isReasoningFirstVisibleScenario
                  ? reasoningFirstVisibleTurnStart?.runtimeRequest
                  : isLiveTailCommitScenario
                    ? liveTailCommitTurnStart?.runtimeRequest
                    : isElectronResizeReflowScenario
                      ? electronResizeReflowTurnStart?.runtimeRequest
                      : isApprovalRequestResumeScenario ||
                          isApprovalRequestDecisionScenario
                        ? approvalRequestResumeTurnStart?.runtimeRequest
                        : isApprovalRequestFullAccessScenario
                          ? approvalRequestFullAccessTurnStart?.runtimeRequest
                          : isTerminalCanceledAfterAnswerScenario
                            ? terminalCanceledAfterAnswerTurnStart?.runtimeRequest
                            : isTerminalFailedAfterAnswerScenario
                              ? terminalFailedAfterAnswerTurnStart?.runtimeRequest
                              : isTypedErrorRetryScenario
                                ? typedErrorTurnStart?.runtimeRequest
                                : isMcpStructuredContentScenario
                                  ? mcpStructuredContentTurnStart?.runtimeRequest
                                  : isMediaReferenceScenario
                                    ? mediaReferenceTurnStart?.runtimeRequest
                                    : isSkillsRuntimeScenario
                                      ? skillsRuntimeTurnStart?.runtimeRequest
                                      : isAnyExpertSkillsRuntimeScenario
                                        ? expertRuntimeTurnStartForAssertions?.runtimeRequest
                                        : isContentFactoryArticleWorkspaceScenario
                                          ? {}
                                          : isHomeHotpathScenario
                                            ? homeHotpathTurnStart?.runtimeRequest
                                            : newsTurnStart?.runtimeRequest) ??
    {};
  const hasCancelPhase = isCancelOnlyScenario || isCancelThenContinueScenario;
  const goalHarness = readHarnessMetadataFromTurnStart(goalTurnStart);
  const goalObjectiveText = readObjectiveTextFromHarness(goalHarness);
  const imageCommandHarness = readHarnessMetadataFromTurnStart(
    imageCommandTurnStart,
  );
  const manualEnableRuntimeMetadata =
    readWorkspaceSkillRuntimeEnableFromTurnStart(
      manualEnableSkillsRuntimeTurnStart,
    );
  const manualEnableRuntimeBinding = Array.isArray(
    manualEnableRuntimeMetadata?.bindings,
  )
    ? manualEnableRuntimeMetadata.bindings[0]
    : null;
  const expertRuntimeMetadata =
    expertRuntimeTurnStartForAssertions?.runtimeRequest?.metadata?.expert ?? {};
  const expertHarnessMetadata =
    expertRuntimeTurnStartForAssertions?.runtimeRequest?.metadata?.harness
      ?.expert ?? {};
  const rawExpertHarnessSkillRefs =
    expertHarnessMetadata?.skill_refs ?? expertHarnessMetadata?.skillRefs ?? [];
  const expertHarnessSkillRefs = Array.isArray(rawExpertHarnessSkillRefs)
    ? rawExpertHarnessSkillRefs
    : [];
  const expectedExpertHarnessSkillRef = isExpertPanelSkillsRuntimeScenario
    ? EXPERT_PANEL_SKILLS_RUNTIME_UI_SKILL_REF
    : EXPERT_SKILLS_RUNTIME_SKILL_REF;
  const collaborationMode = runtimeRequest?.collaborationMode?.mode ?? null;
  const guiTurnStartReachedBackend = isActiveSteerScenario
    ? summary.activeSteer?.provider?.initialRequestObserved === true &&
      summary.activeSteer?.activeRead?.includesInitialPrompt === true &&
      summary.activeSteer?.inputSend?.clicked?.clicked === true
    : isPlanScenario
      ? planTurnStart?.inputText === PLAN_PROMPT
      : isGoalScenario
        ? goalTurnStart?.inputText === GOAL_PROMPT
        : isHomeHotpathScenario
          ? homeHotpathTurnStart?.inputText === homeHotpathPrompt ||
            homeHotpathTraceTurnStart?.inputText === homeHotpathPrompt
          : isImageCommandScenario
            ? imageCommandTurnStart?.inputText ===
              expectedImageIntentRoutedPrompt
            : isInputbarRichRestoreScenario
              ? String(inputbarRichRestoreTurnStart?.inputText || "").includes(
                  INPUTBAR_RICH_RESTORE_PROMPT,
                )
              : isWebToolsRenderingScenario
                ? webToolsRenderingTurnStart?.inputText ===
                  WEB_TOOLS_RENDERING_PROMPT
                : isReasoningFirstVisibleScenario
                  ? reasoningFirstVisibleTurnStart?.inputText ===
                    REASONING_FIRST_VISIBLE_PROMPT
                  : isLiveTailCommitScenario
                    ? liveTailCommitTurnStart?.inputText ===
                      LIVE_TAIL_COMMIT_PROMPT
                    : isElectronResizeReflowScenario
                      ? electronResizeReflowTurnStart?.inputText ===
                        LIVE_TAIL_COMMIT_PROMPT
                      : isApprovalRequestResumeScenario ||
                          isApprovalRequestDecisionScenario
                        ? approvalRequestResumeTurnStart?.inputText ===
                          APPROVAL_REQUEST_RESUME_PROMPT
                        : isApprovalRequestFullAccessScenario
                          ? approvalRequestFullAccessTurnStart?.inputText ===
                            APPROVAL_REQUEST_FULL_ACCESS_PROMPT
                          : isTerminalCanceledAfterAnswerScenario
                            ? terminalCanceledAfterAnswerTurnStart?.inputText ===
                              TERMINAL_CANCELED_AFTER_ANSWER_PROMPT
                            : isTerminalFailedAfterAnswerScenario
                              ? terminalFailedAfterAnswerTurnStart?.inputText ===
                                TERMINAL_FAILED_AFTER_ANSWER_PROMPT
                              : isTypedErrorRetryScenario
                                ? typedErrorTurnStart?.inputText ===
                                  (isTypedErrorRetrySuccessScenario
                                    ? TYPED_ERROR_RETRY_SUCCESS_PROMPT
                                    : TYPED_ERROR_RETRY_FAILURE_PROMPT)
                                : isTerminalStaleGuardScenario
                                  ? terminalStaleGuardFirstTurnStart?.inputText ===
                                      TERMINAL_STALE_GUARD_FIRST_PROMPT &&
                                    terminalStaleGuardSecondTurnStart?.inputText ===
                                      TERMINAL_STALE_GUARD_SECOND_PROMPT
                                  : isMcpStructuredContentScenario
                                    ? mcpStructuredContentTurnStart?.inputText ===
                                      MCP_STRUCTURED_CONTENT_PROMPT
                                    : isMediaReferenceScenario
                                      ? mediaReferenceTurnStart?.inputText ===
                                        MEDIA_REFERENCE_PROMPT
                                      : isSkillsRuntimeScenario
                                        ? skillsRuntimeTurnStart?.inputText ===
                                            SKILLS_RUNTIME_PROMPT &&
                                          explicitSkillsRuntimeTurnStart?.inputText ===
                                            SKILLS_RUNTIME_EXPLICIT_PROMPT &&
                                          manualEnableSkillsRuntimeTurnStart?.inputText ===
                                            SKILLS_RUNTIME_MANUAL_ENABLE_PROMPT
                                        : isAnyExpertSkillsRuntimeScenario
                                          ? isExpertPanelSkillsRuntimeScenario
                                            ? expertPanelSkillsRuntimeTurnStart?.inputText ===
                                              EXPERT_SKILLS_RUNTIME_PANEL_PROMPT
                                            : expertSkillsRuntimeTurnStart?.inputText?.includes(
                                                EXPERT_SKILLS_RUNTIME_PROMPT,
                                              ) === true
                                          : isContentFactoryArticleWorkspaceScenario
                                            ? true
                                            : isSoulStyleScenario
                                              ? newsTraceTurnStart?.inputText ===
                                                NEWS_PROMPT
                                              : newsTurnStart?.inputText ===
                                                NEWS_PROMPT;
  return {
    backendLedger,
    traceMessages,
    appServerRequests,
    rendererSnapshot,
    summary,
    pageText,
    errorRaw,
    actionableConsoleErrors,
    workspace,
    options,
    appServerRequestMethods,
    gateBContract,
    latestTurnStart,
    traceTurnStarts,
    planImplementationTurnStart,
    newsTurnStart,
    newsTraceTurnStart,
    homeHotpathPrompt,
    homeHotpathTurnStart,
    homeHotpathTraceTurnStart,
    planTurnStart,
    goalTurnStart,
    inputbarRichRestoreTurnStart,
    imageCommandTurnStart,
    expectedImageIntentRoutedPrompt,
    webToolsRenderingTurnStart,
    reasoningFirstVisibleTurnStart,
    liveTailCommitTurnStart,
    electronResizeReflowTurnStart,
    approvalRequestResumeTurnStart,
    approvalRequestFullAccessTurnStart,
    terminalCanceledAfterAnswerTurnStart,
    terminalFailedAfterAnswerTurnStart,
    terminalStaleGuardFirstTurnStart,
    terminalStaleGuardSecondTurnStart,
    typedErrorTurnStart,
    mcpStructuredContentTurnStart,
    mediaReferenceTurnStart,
    skillsRuntimeTurnStart,
    explicitSkillsRuntimeTurnStart,
    manualEnableSkillsRuntimeTurnStart,
    expertSkillsRuntimeTurnStart,
    expertPanelSkillsRuntimeTurnStart,
    continueTurnStart,
    latestTurnCancel,
    isCancelOnlyScenario,
    isCancelThenContinueScenario,
    isPlanScenario,
    isGoalScenario,
    isHomeHotpathScenario,
    isImageCommandScenario,
    isInputbarRichRestoreScenario,
    isActiveSteerScenario,
    isWebToolsRenderingScenario,
    isReasoningFirstVisibleScenario,
    isLiveTailCommitScenario,
    isElectronResizeReflowScenario,
    isApprovalRequestResumeScenario,
    isApprovalRequestDeclineScenario,
    isApprovalRequestHostInterruptScenario,
    isApprovalRequestCancelScenario,
    isApprovalRequestFullAccessScenario,
    isApprovalRequestDecisionScenario,
    isTerminalCanceledAfterAnswerScenario,
    isTerminalFailedAfterAnswerScenario,
    isTerminalStaleGuardScenario,
    isTypedErrorRetrySuccessScenario,
    isTypedErrorRetryFailureScenario,
    isTypedErrorRetryScenario,
    isMcpStructuredContentScenario,
    isMediaReferenceScenario,
    isSkillsRuntimeScenario,
    isUnknownItemScenario,
    isSoulStyleScenario,
    isExpertPlazaSkillsRuntimeScenario,
    isExpertPanelSkillsRuntimeScenario,
    isRightSurfaceVisualMatrixScenario,
    isContentFactoryArticleWorkspaceScenario,
    isContentFactoryInlineImageArticleWorkspaceScenario,
    isAnyExpertSkillsRuntimeScenario,
    expertRuntimeTurnStartForAssertions,
    runtimeRequest,
    hasCancelPhase,
    goalObjectiveText,
    imageCommandHarness,
    manualEnableRuntimeMetadata,
    manualEnableRuntimeBinding,
    expertRuntimeMetadata,
    expertHarnessMetadata,
    rawExpertHarnessSkillRefs,
    expertHarnessSkillRefs,
    expectedExpertHarnessSkillRef,
    collaborationMode,
    guiTurnStartReachedBackend,
  };
}
