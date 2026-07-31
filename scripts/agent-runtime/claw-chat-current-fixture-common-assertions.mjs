import {
  APP_SERVER_METHOD_SESSION_LIST,
  APP_SERVER_METHOD_SESSION_READ,
  APP_SERVER_METHOD_SESSION_START,
  APP_SERVER_METHOD_SESSION_TURN_START,
  APP_SERVER_METHOD_ARTIFACT_WRITE,
  APP_SERVER_METHOD_DIAGNOSTICS_SUPPORT_BUNDLE_EXPORT,
  APP_SERVER_METHOD_DIAGNOSTICS_TRACE_EXPORT,
  APP_SERVER_METHOD_DIAGNOSTICS_TRACE_LIST,
  APP_SERVER_METHOD_DIAGNOSTICS_TRACE_READ,
  APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST,
  APPROVAL_REQUEST_CANCEL_DONE_TEXT,
  APPROVAL_REQUEST_DECLINE_DONE_TEXT,
  APPROVAL_REQUEST_DECLINE_RESULT_TEXT,
  APPROVAL_REQUEST_FULL_ACCESS_DONE_TEXT,
  APPROVAL_REQUEST_FULL_ACCESS_PROMPT,
  APPROVAL_REQUEST_FULL_ACCESS_RESULT_TEXT,
  APPROVAL_REQUEST_RESUME_DONE_TEXT,
  APPROVAL_REQUEST_RESUME_PROMPT,
  APPROVAL_REQUEST_RESUME_RESULT_TEXT,
  ASSISTANT_DONE_TEXT,
  CONTINUE_DONE_TEXT,
  CONTINUE_PROMPT,
  EXPERT_SKILLS_RUNTIME_TITLE,
  FIXTURE_MODEL,
  FIXTURE_PROVIDER,
  GOAL_DONE_TEXT,
  GOAL_PROMPT,
  IMAGE_COMMAND_DONE_TEXT,
  INPUTBAR_RICH_RESTORE_PROMPT,
  MCP_STRUCTURED_CONTENT_DONE_TEXT,
  NEWS_PROMPT,
  PLAN_DONE_TEXT,
  PLAN_PROMPT,
  PLAN_STEPS,
  REASONING_FIRST_VISIBLE_DONE_TEXT,
  REASONING_FIRST_VISIBLE_FINAL_TEXT,
  REASONING_FIRST_VISIBLE_PROMPT,
  REASONING_FIRST_VISIBLE_TEXT,
  TERMINAL_CANCELED_AFTER_ANSWER_PARTIAL_TEXT,
  TERMINAL_CANCELED_AFTER_ANSWER_PROMPT,
  TERMINAL_FAILED_AFTER_ANSWER_FAILURE_TEXT,
  TERMINAL_FAILED_AFTER_ANSWER_PARTIAL_TEXT,
  TERMINAL_FAILED_AFTER_ANSWER_PROMPT,
  TERMINAL_STALE_GUARD_DONE_TEXT,
  TERMINAL_STALE_GUARD_FIRST_DONE_TEXT,
  TERMINAL_STALE_GUARD_FIRST_PROMPT,
  TERMINAL_STALE_GUARD_SECOND_PROMPT,
  TERMINAL_STALE_GUARD_SECOND_TEXT,
  TERMINAL_STALE_GUARD_STALE_DONE_TEXT,
  TYPED_ERROR_RETRY_FAILURE_PARTIAL_TEXT,
  TYPED_ERROR_RETRY_FAILURE_PROMPT,
  TYPED_ERROR_RETRY_SUCCESS_PROMPT,
  TYPED_ERROR_RETRY_SUCCESS_TEXT,
} from "./claw-chat-current-fixture-constants.mjs";
import {
  MEDIA_REFERENCE_PROMPT,
  MEDIA_REFERENCE_SUMMARY_TEXT,
  MEDIA_REFERENCE_URI,
} from "./claw-chat-current-fixture-media-reference.mjs";
import {
  ACTIVE_STEER_DONE_TEXT,
  ACTIVE_STEER_INITIAL_PROMPT,
  ACTIVE_STEER_INPUT,
} from "./claw-chat-current-fixture-active-steer.mjs";

const LEGACY_RESPOND_ACTION_METHOD = [
  "agent",
  "runtime",
  "respond",
  "action",
].join("_");

export function buildCommonAssertions(context) {
  const {
    rendererSnapshot,
    appServerRequestMethods,
    guiTurnStartReachedBackend,
    backendLedger,
    runtimeRequest,
    isCancelOnlyScenario,
    isCancelThenContinueScenario,
    isPlanScenario,
    isGoalScenario,
    isHomeHotpathScenario,
    isImageCommandScenario,
    isInputbarRichRestoreScenario,
    isActiveSteerScenario,
    inputbarRichRestoreTurnStart,
    isWebToolsRenderingScenario,
    isLiveTailCommitScenario,
    isElectronResizeReflowScenario,
    isMcpStructuredContentScenario,
    isMediaReferenceScenario,
    isReasoningFirstVisibleScenario,
    isTerminalCanceledAfterAnswerScenario,
    isTerminalFailedAfterAnswerScenario,
    isTerminalStaleGuardScenario,
    isTypedErrorRetryScenario,
    isTypedErrorRetrySuccessScenario,
    isSkillsRuntimeScenario,
    isSoulStyleScenario,
    isRightSurfaceVisualMatrixScenario,
    isContentFactoryArticleWorkspaceScenario,
    isAnyExpertSkillsRuntimeScenario,
    isApprovalRequestResumeScenario,
    isApprovalRequestDeclineScenario,
    isApprovalRequestCancelScenario,
    isApprovalRequestDecisionScenario,
    isApprovalRequestFullAccessScenario,
    isExpertPlazaSkillsRuntimeScenario,
    isExpertPanelSkillsRuntimeScenario,
    hasCancelPhase,
    summary,
    pageText,
    errorRaw,
    actionableConsoleErrors,
  } = context;
  const shouldRequireAgentUiTraceEvidence =
    !isRightSurfaceVisualMatrixScenario &&
    !isContentFactoryArticleWorkspaceScenario &&
    !isExpertPlazaSkillsRuntimeScenario;
  const isLiveTailLayoutScenario =
    isLiveTailCommitScenario || isElectronResizeReflowScenario;
  const shouldRequireTextStreamTraceSeparation =
    shouldRequireAgentUiTraceEvidence &&
    !isPlanScenario &&
    !isImageCommandScenario &&
    !isMediaReferenceScenario &&
    !isSoulStyleScenario &&
    !isInputbarRichRestoreScenario &&
    !isApprovalRequestResumeScenario &&
    !isApprovalRequestDecisionScenario &&
    !isApprovalRequestFullAccessScenario;
  const shouldRequireProviderTextTraceEvidence =
    shouldRequireTextStreamTraceSeparation && !isTypedErrorRetryScenario;
  const agentUiPerformanceTrace = summary.agentUiPerformanceTrace;
  const appServerTraceEvidence = summary.appServerTraceEvidence;
  const appServerTraceEvidenceDeferred =
    summary.appServerTraceEvidenceDeferred != null;
  const approvalRequestDecisionCompleted = isApprovalRequestDeclineScenario
    ? summary.guiApprovalRequestDeclineCompleted
    : isApprovalRequestCancelScenario
      ? summary.guiApprovalRequestCancelCompleted
      : null;
  const approvalRequestDecisionPageText =
    approvalRequestDecisionCompleted?.bodyText ?? "";
  const approvalRequestDecisionCanceledReadModel =
    summary.readModelApprovalRequestCancelCanceled;
  const approvalRequestDecisionDeclinedReadModel =
    summary.readModelApprovalRequestDeclineCompleted;
  const approvalRequestDecisionHistoricalDetailsHidden =
    Number(approvalRequestDecisionCompleted?.compactTimelinePreviewCount || 0) >
      0 &&
    approvalRequestDecisionCompleted?.approvalRecordShape?.recordCount === 0 &&
    Array.isArray(
      approvalRequestDecisionCompleted?.approvalRecordShape?.texts,
    ) &&
    approvalRequestDecisionCompleted.approvalRecordShape.texts.length === 0;
  const homeHotpath = summary.homeHotpath ?? {};
  const homeHotpathGuiCompleted = homeHotpath.guiCompleted ?? {};
  const homeHotpathPostSubmitProjection =
    homeHotpath.postSubmitProjection ?? {};
  const homeHotpathCompletedProjection = homeHotpath.completedProjection ?? {};
  const homeHotpathInputAfterClick = homeHotpath.inputSend?.afterClick ?? {};
  const homeHotpathReadModelCompleted = homeHotpath.readModelCompleted ?? {};
  const homeHotpathExpectedPrompt =
    typeof homeHotpath.prompt === "string" && homeHotpath.prompt.trim()
      ? homeHotpath.prompt
      : NEWS_PROMPT;
  const homeHotpathExpectedDoneText =
    typeof homeHotpath.doneText === "string" && homeHotpath.doneText.trim()
      ? homeHotpath.doneText
      : ASSISTANT_DONE_TEXT;
  const homeHotpathExpectedSummaryText =
    typeof homeHotpath.summaryText === "string" &&
    homeHotpath.summaryText.trim()
      ? homeHotpath.summaryText
      : "今日国际新闻简要整理";
  const homeHotpathPageText = [
    pageText,
    homeHotpathGuiCompleted.bodyText,
    homeHotpathGuiCompleted.mainText,
    homeHotpathCompletedProjection.bodyText,
    homeHotpathCompletedProjection.mainText,
  ]
    .filter((value) => typeof value === "string")
    .join("\n");
  const commonAssertions = {
    electronPreloadBridge: rendererSnapshot.electron === true,
    appServerJsonRpcUsed:
      appServerRequestMethods.includes(APP_SERVER_METHOD_SESSION_TURN_START) ||
      guiTurnStartReachedBackend ||
      (isRightSurfaceVisualMatrixScenario &&
        appServerRequestMethods.includes(
          APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST,
        )) ||
      (isContentFactoryArticleWorkspaceScenario &&
        appServerRequestMethods.includes(APP_SERVER_METHOD_ARTIFACT_WRITE)),
    usedCurrentSessionStart: appServerRequestMethods.includes(
      APP_SERVER_METHOD_SESSION_START,
    ),
    usedCurrentSessionRead: appServerRequestMethods.includes(
      APP_SERVER_METHOD_SESSION_READ,
    ),
    usedCurrentSessionList: appServerRequestMethods.includes(
      APP_SERVER_METHOD_SESSION_LIST,
    ),
    externalFixtureBackendUsed: isActiveSteerScenario
      ? summary.textProviderFixtureServer?.requestCount >= 3 &&
        summary.activeSteer?.provider?.chatRequestCount >= 2
      : isImageCommandScenario
        ? summary.imageCommandWorkflowUsed === true
        : isRightSurfaceVisualMatrixScenario ||
            isContentFactoryArticleWorkspaceScenario
          ? true
          : isSoulStyleScenario
            ? summary.textProviderFixtureServer?.requestCount >= 1
            : backendLedger.some((entry) => entry.kind === "turnStart"),
    fixturePromptReachedBackend: isActiveSteerScenario
      ? summary.activeSteer?.provider?.initialRequestObserved === true &&
        summary.activeSteer?.provider?.steerRequestObserved === true
      : isRightSurfaceVisualMatrixScenario ||
          isContentFactoryArticleWorkspaceScenario
        ? true
        : isInputbarRichRestoreScenario
          ? Boolean(inputbarRichRestoreTurnStart)
          : guiTurnStartReachedBackend,
    liveProviderNotUsed: backendLedger.every((entry) => {
      if (entry.kind !== "turnStart") {
        return true;
      }
      if (isInputbarRichRestoreScenario) {
        return (
          (!entry.providerPreference ||
            entry.providerPreference ===
              summary.textFixtureProvider?.providerId ||
            entry.providerPreference === FIXTURE_PROVIDER) &&
          (!entry.modelPreference ||
            entry.modelPreference === summary.textFixtureProvider?.modelId ||
            entry.modelPreference === FIXTURE_MODEL)
        );
      }
      return (
        (!entry.providerPreference ||
          entry.providerPreference === FIXTURE_PROVIDER) &&
        (!entry.modelPreference || entry.modelPreference === FIXTURE_MODEL)
      );
    }),
    newsRequestDidNotForceRequiredSearch:
      runtimeRequest?.search_mode !== "required",
    newsRequestDidNotPassLegacyWebSearchFlag:
      !Object.prototype.hasOwnProperty.call(runtimeRequest || {}, "web_search"),
    guiUserMessageVisible: isCancelOnlyScenario
      ? summary.guiCanceled?.hasPrompt === true
      : isCancelThenContinueScenario
        ? summary.guiContinueCompleted?.hasPrompt === true &&
          summary.guiContinueCompleted?.bodyText?.includes(NEWS_PROMPT) === true
        : isInputbarRichRestoreScenario
          ? summary.inputbarRichRestoreGuiCanceled?.hasPrompt === true
          : isRightSurfaceVisualMatrixScenario
            ? summary.rightSurfaceVisualMatrix?.captures?.expertInfo?.stable
                ?.rootVisible === true
            : isContentFactoryArticleWorkspaceScenario
              ? summary.contentFactoryArticleWorkspaceGui?.rootVisible ===
                  true &&
                summary.contentFactoryArticleWorkspaceGui
                  ?.hasArticleDraftObject === true
              : isPlanScenario
                ? summary.guiPlanCompleted?.hasPrompt === true
                : isGoalScenario
                  ? summary.guiGoalCompleted?.hasPrompt === true
                  : isHomeHotpathScenario
                    ? homeHotpathGuiCompleted.hasPrompt === true ||
                      homeHotpathPostSubmitProjection.promptInBody === true ||
                      homeHotpathInputAfterClick.promptInBody === true
                    : isImageCommandScenario
                      ? summary.guiImageCommandCompleted?.hasPrompt === true
                      : isWebToolsRenderingScenario
                        ? summary.guiWebToolsRenderingCompleted?.hasPrompt ===
                          true
                        : isReasoningFirstVisibleScenario
                          ? summary.guiReasoningFirstVisibleCompleted
                              ?.hasPrompt === true
                          : isLiveTailLayoutScenario
                            ? summary.guiLiveTailCompleted?.hasPrompt ===
                                true ||
                              summary.guiElectronResizeReflowCompleted
                                ?.hasPrompt === true
                            : isApprovalRequestResumeScenario
                              ? summary.guiApprovalRequestResumeCompleted
                                  ?.hasPrompt === true
                              : isApprovalRequestFullAccessScenario
                                ? summary.guiApprovalRequestFullAccessCompleted
                                    ?.hasPrompt === true
                                : isApprovalRequestDecisionScenario
                                  ? approvalRequestDecisionCompleted?.hasPrompt ===
                                    true
                                  : isTerminalCanceledAfterAnswerScenario
                                    ? summary
                                        .guiTerminalCanceledAfterAnswerCanceled
                                        ?.hasPrompt === true
                                    : isTerminalFailedAfterAnswerScenario
                                      ? summary
                                          .guiTerminalFailedAfterAnswerCompleted
                                          ?.hasPrompt === true
                                      : isTerminalStaleGuardScenario
                                        ? summary
                                            .guiTerminalStaleGuardFirstCompleted
                                            ?.hasPrompt === true &&
                                          summary
                                            .guiTerminalStaleGuardSecondCompleted
                                            ?.hasPrompt === true
                                        : isTypedErrorRetryScenario
                                          ? summary.guiTypedErrorTerminal
                                              ?.hasPrompt === true
                                          : isMcpStructuredContentScenario
                                            ? summary
                                                .guiMcpStructuredContentCompleted
                                                ?.hasPrompt === true
                                            : isMediaReferenceScenario
                                              ? summary
                                                  .guiMediaReferenceCompleted
                                                  ?.hasPrompt === true
                                              : isSkillsRuntimeScenario
                                                ? summary
                                                    .guiSkillsRuntimeCompleted
                                                    ?.hasPrompt === true &&
                                                  summary
                                                    .guiExplicitSkillsRuntimeCompleted
                                                    ?.hasPrompt === true &&
                                                  summary
                                                    .guiManualEnableSkillsRuntimeCompleted
                                                    ?.hasPrompt === true
                                                : isAnyExpertSkillsRuntimeScenario
                                                  ? isExpertPanelSkillsRuntimeScenario
                                                    ? summary
                                                        .guiExpertPanelSkillsRuntimeCompleted
                                                        ?.hasPrompt === true
                                                    : summary
                                                        .guiExpertSkillsRuntimeCompleted
                                                        ?.hasPrompt === true
                                                  : summary.guiCompleted
                                                      ?.hasPrompt === true,
    guiAssistantOutputVisible: isCancelOnlyScenario
      ? summary.guiCanceled?.hasStoppedCopy === true
      : isCancelThenContinueScenario
        ? summary.guiContinueCompleted?.hasAssistantSummary === true ||
          summary.guiContinueCompleted?.hasDoneText === true
        : isInputbarRichRestoreScenario
          ? summary.inputbarRichRestoreGuiCanceled?.noVisibleAssistantOutput ===
            true
          : isRightSurfaceVisualMatrixScenario
            ? summary.rightSurfaceVisualMatrix?.captures?.files?.stable
                ?.rootVisible === true &&
              summary.rightSurfaceVisualMatrix?.captures?.objectCanvas?.stable
                ?.rootVisible === true &&
              summary.rightSurfaceVisualMatrix?.captures?.browser?.stable
                ?.rootVisible === true
            : isContentFactoryArticleWorkspaceScenario
              ? summary.contentFactoryArticleWorkspaceArtifactFrame?.visible ===
                  true &&
                summary.contentFactoryArticleWorkspaceArtifactFrame
                  ?.hasArticlePreviewContent === true &&
                summary.contentFactoryArticleWorkspaceGui
                  ?.hasArticleCanvasContent === true
              : isPlanScenario
                ? summary.guiPlanCompleted?.hasPlanIntro === true ||
                  summary.guiPlanCompleted?.hasDoneText === true ||
                  (summary.guiPlanCompleted?.hasPlanSection === true &&
                    summary.guiPlanCompleted?.hasAllPlanSteps === true)
                : isGoalScenario
                  ? summary.guiGoalCompleted?.hasAssistantSummary === true ||
                    summary.guiGoalCompleted?.hasDoneText === true
                  : isHomeHotpathScenario
                    ? homeHotpathGuiCompleted.hasAssistantSummary === true ||
                      homeHotpathGuiCompleted.hasDoneText === true ||
                      homeHotpathReadModelCompleted.includesDone === true
                    : isImageCommandScenario
                      ? summary.guiImageCommandCompleted
                          ?.hasAssistantSummary === true ||
                        summary.guiImageCommandCompleted?.hasDoneText ===
                          true ||
                        summary.guiImageCommandCompleted
                          ?.imageTaskCardVisible === true
                      : isWebToolsRenderingScenario
                        ? summary.guiWebToolsRenderingCompleted
                            ?.hasAssistantSummary === true ||
                          summary.guiWebToolsRenderingCompleted?.hasDoneText ===
                            true
                        : isReasoningFirstVisibleScenario
                          ? summary.guiReasoningFirstVisibleBeforeAnswer
                              ?.hasReasoningText === true &&
                            summary.guiReasoningFirstVisibleCompleted
                              ?.hasFinalText === true
                          : isLiveTailLayoutScenario
                            ? summary.guiLiveTailFirstVisibleBeforeCommit
                                ?.hasFirstText === true ||
                              (summary.guiElectronResizeReflowCompleted
                                ?.hasAssistantSummary === true &&
                                summary.electronResizeReflowLayout?.snapshots
                                  ?.restored?.hasTableTail === true)
                            : isApprovalRequestResumeScenario
                              ? summary.guiApprovalRequestResumeCompleted
                                  ?.hasAssistantSummary === true ||
                                summary.guiApprovalRequestResumeCompleted
                                  ?.hasDoneText === true
                              : isApprovalRequestFullAccessScenario
                                ? summary.guiApprovalRequestFullAccessCompleted
                                    ?.hasAssistantSummary === true ||
                                  summary.guiApprovalRequestFullAccessCompleted
                                    ?.hasDoneText === true
                                : isApprovalRequestDecisionScenario
                                  ? isApprovalRequestDeclineScenario
                                    ? approvalRequestDecisionCompleted?.hasAssistantSummary ===
                                        true ||
                                      approvalRequestDecisionCompleted?.hasDoneText ===
                                        true
                                    : approvalRequestDecisionCanceledReadModel?.latestTurnCanceled ===
                                        true &&
                                      approvalRequestDecisionHistoricalDetailsHidden
                                  : isTerminalCanceledAfterAnswerScenario
                                    ? summary
                                        .guiTerminalCanceledAfterAnswerCanceled
                                        ?.hasPartialText === true
                                    : isTerminalFailedAfterAnswerScenario
                                      ? summary
                                          .guiTerminalFailedAfterAnswerCompleted
                                          ?.hasAssistantSummary === true
                                      : isTerminalStaleGuardScenario
                                        ? summary
                                            .guiTerminalStaleGuardSecondCompleted
                                            ?.hasAssistantSummary === true ||
                                          summary
                                            .guiTerminalStaleGuardSecondCompleted
                                            ?.hasDoneText === true
                                        : isTypedErrorRetryScenario
                                          ? summary.guiTypedErrorTerminal
                                              ?.hasAssistantSummary === true ||
                                            summary.guiTypedErrorTerminal
                                              ?.hasDoneText === true
                                          : isMcpStructuredContentScenario
                                            ? (summary
                                                .guiMcpStructuredContentCompleted
                                                ?.hasStructuredAnswer ===
                                                true &&
                                                summary
                                                  .guiMcpStructuredContentCompleted
                                                  ?.hasReferenceId === true) ||
                                              summary
                                                .guiMcpStructuredContentCompleted
                                                ?.hasDoneText === true ||
                                              summary
                                                .guiMcpStructuredContentCompleted
                                                ?.terminalDetailsCompacted ===
                                                true
                                            : isMediaReferenceScenario
                                              ? summary
                                                  .guiMediaReferenceSnapshot
                                                  ?.hasCard === true &&
                                                summary
                                                  .guiMediaReferenceSnapshot
                                                  ?.hasUri === true
                                              : isSkillsRuntimeScenario
                                                ? summary
                                                    .guiSkillsRuntimeCompleted
                                                    ?.hasAssistantSummary ===
                                                    true ||
                                                  summary
                                                    .guiSkillsRuntimeCompleted
                                                    ?.hasDoneText === true ||
                                                  summary
                                                    .guiExplicitSkillsRuntimeCompleted
                                                    ?.hasAssistantSummary ===
                                                    true ||
                                                  summary
                                                    .guiExplicitSkillsRuntimeCompleted
                                                    ?.hasDoneText === true ||
                                                  summary
                                                    .guiManualEnableSkillsRuntimeCompleted
                                                    ?.hasAssistantSummary ===
                                                    true ||
                                                  summary
                                                    .guiManualEnableSkillsRuntimeCompleted
                                                    ?.hasDoneText === true
                                                : isAnyExpertSkillsRuntimeScenario
                                                  ? isExpertPanelSkillsRuntimeScenario
                                                    ? summary
                                                        .guiExpertPanelSkillsRuntimeCompleted
                                                        ?.hasAssistantSummary ===
                                                        true ||
                                                      summary
                                                        .guiExpertPanelSkillsRuntimeCompleted
                                                        ?.hasDoneText === true
                                                    : summary
                                                        .guiExpertSkillsRuntimeCompleted
                                                        ?.hasAssistantSummary ===
                                                        true ||
                                                      summary
                                                        .guiExpertSkillsRuntimeCompleted
                                                        ?.hasDoneText === true
                                                  : summary.guiCompleted
                                                      ?.hasAssistantSummary ===
                                                      true ||
                                                    summary.guiCompleted
                                                      ?.hasDoneText === true,
    guiInputRemainsReady: isCancelOnlyScenario
      ? summary.guiCanceled?.textareaVisible === true &&
        summary.guiCanceled?.textareaDisabled === false
      : isCancelThenContinueScenario
        ? summary.guiContinueCompleted?.textareaVisible === true &&
          summary.guiContinueCompleted?.textareaDisabled === false
        : isInputbarRichRestoreScenario
          ? summary.inputbarRichRestoreGuiCanceled?.textareaVisible === true &&
            summary.inputbarRichRestoreGuiCanceled?.textareaDisabled ===
              false &&
            summary.inputbarRichRestoreGuiCanceled?.textareaValue ===
              INPUTBAR_RICH_RESTORE_PROMPT
          : isRightSurfaceVisualMatrixScenario
            ? summary.guiRightSurfaceVisualMatrixSessionOpened?.inputReady
                ?.textareaDisabled === false
            : isContentFactoryArticleWorkspaceScenario
              ? summary.guiContentFactoryArticleWorkspaceSessionOpened
                  ?.inputReady?.textareaDisabled === false
              : isPlanScenario
                ? summary.guiPlanCompleted?.planDecisionVisible === true &&
                  summary.guiPlanCompleted?.textareaVisible === false
                : isGoalScenario
                  ? summary.guiGoalCompleted?.textareaVisible === true &&
                    summary.guiGoalCompleted?.textareaDisabled === false
                  : isHomeHotpathScenario
                    ? homeHotpathGuiCompleted.textareaVisible === true &&
                      homeHotpathGuiCompleted.textareaDisabled === false
                    : isImageCommandScenario
                      ? summary.guiImageCommandCompleted?.textareaVisible ===
                          true &&
                        summary.guiImageCommandCompleted?.textareaDisabled ===
                          false
                      : isWebToolsRenderingScenario
                        ? summary.guiWebToolsRenderingCompleted
                            ?.textareaVisible === true &&
                          summary.guiWebToolsRenderingCompleted
                            ?.textareaDisabled === false
                        : isReasoningFirstVisibleScenario
                          ? summary.guiReasoningFirstVisibleCompleted
                              ?.textareaVisible === true &&
                            summary.guiReasoningFirstVisibleCompleted
                              ?.textareaDisabled === false
                          : isLiveTailLayoutScenario
                            ? summary.guiLiveTailCompleted?.textareaVisible ===
                                true ||
                              (summary.guiElectronResizeReflowCompleted
                                ?.textareaVisible === true &&
                                summary.guiElectronResizeReflowCompleted
                                  ?.textareaDisabled === false)
                            : isApprovalRequestResumeScenario
                              ? summary.guiApprovalRequestResumeCompleted
                                  ?.textareaVisible === true &&
                                summary.guiApprovalRequestResumeCompleted
                                  ?.textareaDisabled === false
                              : isApprovalRequestFullAccessScenario
                                ? summary.guiApprovalRequestFullAccessCompleted
                                    ?.textareaVisible === true &&
                                  summary.guiApprovalRequestFullAccessCompleted
                                    ?.textareaDisabled === false
                                : isApprovalRequestDecisionScenario
                                  ? approvalRequestDecisionCompleted?.textareaVisible ===
                                      true &&
                                    approvalRequestDecisionCompleted?.textareaDisabled ===
                                      false
                                  : isTerminalCanceledAfterAnswerScenario
                                    ? summary
                                        .guiTerminalCanceledAfterAnswerCanceled
                                        ?.textareaVisible === true &&
                                      summary
                                        .guiTerminalCanceledAfterAnswerCanceled
                                        ?.textareaDisabled === false
                                    : isTerminalFailedAfterAnswerScenario
                                      ? summary
                                          .guiTerminalFailedAfterAnswerCompleted
                                          ?.textareaVisible === true &&
                                        summary
                                          .guiTerminalFailedAfterAnswerCompleted
                                          ?.textareaDisabled === false
                                      : isTerminalStaleGuardScenario
                                        ? summary
                                            .guiTerminalStaleGuardSecondCompleted
                                            ?.textareaVisible === true &&
                                          summary
                                            .guiTerminalStaleGuardSecondCompleted
                                            ?.textareaDisabled === false
                                        : isTypedErrorRetryScenario
                                          ? summary.guiTypedErrorTerminal
                                              ?.textareaVisible === true &&
                                            summary.guiTypedErrorTerminal
                                              ?.textareaDisabled === false
                                          : isMcpStructuredContentScenario
                                            ? summary
                                                .guiMcpStructuredContentCompleted
                                                ?.textareaVisible === true &&
                                              summary
                                                .guiMcpStructuredContentCompleted
                                                ?.textareaDisabled === false
                                            : isMediaReferenceScenario
                                              ? summary
                                                  .guiMediaReferenceCompleted
                                                  ?.textareaVisible === true &&
                                                summary
                                                  .guiMediaReferenceCompleted
                                                  ?.textareaDisabled === false
                                              : isSkillsRuntimeScenario
                                                ? summary
                                                    .guiSkillsRuntimeCompleted
                                                    ?.textareaVisible ===
                                                    true &&
                                                  summary
                                                    .guiSkillsRuntimeCompleted
                                                    ?.textareaDisabled ===
                                                    false &&
                                                  summary
                                                    .guiExplicitSkillsRuntimeCompleted
                                                    ?.textareaVisible ===
                                                    true &&
                                                  summary
                                                    .guiExplicitSkillsRuntimeCompleted
                                                    ?.textareaDisabled ===
                                                    false &&
                                                  summary
                                                    .guiManualEnableSkillsRuntimeCompleted
                                                    ?.textareaVisible ===
                                                    true &&
                                                  summary
                                                    .guiManualEnableSkillsRuntimeCompleted
                                                    ?.textareaDisabled === false
                                                : isAnyExpertSkillsRuntimeScenario
                                                  ? isExpertPanelSkillsRuntimeScenario
                                                    ? summary
                                                        .guiExpertPanelSkillsRuntimeCompleted
                                                        ?.textareaVisible ===
                                                        true &&
                                                      summary
                                                        .guiExpertPanelSkillsRuntimeCompleted
                                                        ?.textareaDisabled ===
                                                        false
                                                    : summary
                                                        .guiExpertSkillsRuntimeCompleted
                                                        ?.textareaVisible ===
                                                        true &&
                                                      summary
                                                        .guiExpertSkillsRuntimeCompleted
                                                        ?.textareaDisabled ===
                                                        false
                                                  : summary.guiCompleted
                                                      ?.textareaVisible ===
                                                      true &&
                                                    summary.guiCompleted
                                                      ?.textareaDisabled ===
                                                      false,
    guiNotStuckStreaming: isCancelOnlyScenario
      ? summary.guiCanceled?.stopButtonVisible === false
      : isCancelThenContinueScenario
        ? summary.guiContinueCompleted?.stopButtonVisible === false
        : isInputbarRichRestoreScenario
          ? summary.inputbarRichRestoreGuiCanceled?.stopButtonVisible === false
          : isRightSurfaceVisualMatrixScenario
            ? true
            : isContentFactoryArticleWorkspaceScenario
              ? true
              : isPlanScenario
                ? summary.guiPlanCompleted?.stopButtonVisible === false
                : isGoalScenario
                  ? summary.guiGoalCompleted?.stopButtonVisible === false
                  : isHomeHotpathScenario
                    ? homeHotpathGuiCompleted.stopButtonVisible === false
                    : isImageCommandScenario
                      ? summary.guiImageCommandCompleted?.stopButtonVisible ===
                        false
                      : isWebToolsRenderingScenario
                        ? summary.guiWebToolsRenderingCompleted
                            ?.stopButtonVisible === false
                        : isReasoningFirstVisibleScenario
                          ? summary.guiReasoningFirstVisibleCompleted
                              ?.stopButtonVisible === false
                          : isLiveTailLayoutScenario
                            ? summary.guiLiveTailCompleted
                                ?.stopButtonVisible === false ||
                              summary.guiElectronResizeReflowCompleted
                                ?.stopButtonVisible === false
                            : isApprovalRequestResumeScenario
                              ? summary.guiApprovalRequestResumeCompleted
                                  ?.stopButtonVisible === false
                              : isApprovalRequestFullAccessScenario
                                ? summary.guiApprovalRequestFullAccessCompleted
                                    ?.stopButtonVisible === false
                                : isApprovalRequestDecisionScenario
                                  ? approvalRequestDecisionCompleted?.stopButtonVisible ===
                                    false
                                  : isTerminalCanceledAfterAnswerScenario
                                    ? summary
                                        .guiTerminalCanceledAfterAnswerCanceled
                                        ?.stopButtonVisible === false
                                    : isTerminalFailedAfterAnswerScenario
                                      ? summary
                                          .guiTerminalFailedAfterAnswerCompleted
                                          ?.stopButtonVisible === false
                                      : isTerminalStaleGuardScenario
                                        ? summary
                                            .guiTerminalStaleGuardSecondCompleted
                                            ?.stopButtonVisible === false
                                        : isTypedErrorRetryScenario
                                          ? summary.guiTypedErrorTerminal
                                              ?.stopButtonVisible === false
                                          : isMcpStructuredContentScenario
                                            ? summary
                                                .guiMcpStructuredContentCompleted
                                                ?.stopButtonVisible === false
                                            : isMediaReferenceScenario
                                              ? summary
                                                  .guiMediaReferenceCompleted
                                                  ?.stopButtonVisible === false
                                              : isSkillsRuntimeScenario
                                                ? summary
                                                    .guiSkillsRuntimeCompleted
                                                    ?.stopButtonVisible ===
                                                    false &&
                                                  summary
                                                    .guiExplicitSkillsRuntimeCompleted
                                                    ?.stopButtonVisible ===
                                                    false &&
                                                  summary
                                                    .guiManualEnableSkillsRuntimeCompleted
                                                    ?.stopButtonVisible ===
                                                    false
                                                : isAnyExpertSkillsRuntimeScenario
                                                  ? isExpertPanelSkillsRuntimeScenario
                                                    ? summary
                                                        .guiExpertPanelSkillsRuntimeCompleted
                                                        ?.stopButtonVisible ===
                                                      false
                                                    : summary
                                                        .guiExpertSkillsRuntimeCompleted
                                                        ?.stopButtonVisible ===
                                                      false
                                                  : summary.guiCompleted
                                                      ?.stopButtonVisible ===
                                                    false,
    guiRunningStatusPreservedBeforeStop:
      !hasCancelPhase ||
      (summary.stopClick?.beforeClick?.hasVisibleAssistantOutput === true &&
        summary.stopClick?.beforeClick?.hasRunningStatus === true &&
        summary.stopClick?.beforeClick?.startupNoteVisible === false &&
        Array.isArray(summary.stopClick?.beforeClick?.stopButtons) &&
        summary.stopClick.beforeClick.stopButtons.length > 0),
    pageMentionsPromptAndAssistant: isActiveSteerScenario
      ? pageText.includes(ACTIVE_STEER_INITIAL_PROMPT) &&
        pageText.includes(ACTIVE_STEER_INPUT) &&
        pageText.includes(ACTIVE_STEER_DONE_TEXT)
      : isCancelOnlyScenario
        ? pageText.includes(NEWS_PROMPT) &&
          (pageText.includes("已停止") ||
            pageText.includes("本轮已中止") ||
            /\bStopped\b/i.test(pageText) ||
            /\bCanceled\b/i.test(pageText))
        : isCancelThenContinueScenario
          ? pageText.includes(NEWS_PROMPT) &&
            pageText.includes(CONTINUE_PROMPT) &&
            (pageText.includes("继续输出已恢复") ||
              pageText.includes(CONTINUE_DONE_TEXT))
          : isInputbarRichRestoreScenario
            ? pageText.includes(INPUTBAR_RICH_RESTORE_PROMPT) &&
              summary.inputbarRichRestoreGuiCanceled
                ?.noVisibleAssistantOutput === true
            : isRightSurfaceVisualMatrixScenario
              ? summary.rightSurfaceVisualMatrix?.captures?.files?.stable
                  ?.activeSurface === "files" &&
                summary.rightSurfaceVisualMatrix?.captures?.objectCanvas?.stable
                  ?.activeSurface === "objectCanvas" &&
                summary.rightSurfaceVisualMatrix?.captures?.expertInfo?.stable
                  ?.activeSurface === "expertInfo" &&
                summary.rightSurfaceVisualMatrix?.captures?.browser?.stable
                  ?.activeSurface === "browser" &&
                summary.rightSurfaceVisualMatrix?.captures?.appSurface?.stable
                  ?.activeSurface === "appSurface"
              : isContentFactoryArticleWorkspaceScenario
                ? summary.contentFactoryArticleWorkspaceGui?.activeSurface ===
                    "articleWorkspace" &&
                  summary.contentFactoryArticleWorkspaceGui
                    ?.hasArticleDraftObject === true &&
                  summary.contentFactoryArticleWorkspaceGui
                    ?.hasArticleCanvasContent === true &&
                  summary.contentFactoryArticleWorkspaceArtifactFrame
                    ?.hasArticlePreviewContent === true &&
                  summary.contentFactoryArticleWorkspaceReadModel
                    ?.hasImageSetObject === true &&
                  summary.contentFactoryArticleWorkspaceReadModel
                    ?.hasStoryboardObject === true &&
                  summary.contentFactoryArticleWorkspaceReadModel
                    ?.hasChecklistObject === true
                : isPlanScenario
                  ? pageText.includes(PLAN_PROMPT) &&
                    PLAN_STEPS.every((step) => pageText.includes(step.step))
                  : isGoalScenario
                    ? pageText.includes(GOAL_PROMPT) &&
                      (pageText.includes("目标已绑定到本轮请求") ||
                        pageText.includes(GOAL_DONE_TEXT))
                    : isHomeHotpathScenario
                      ? homeHotpathPageText.includes(
                          homeHotpathExpectedPrompt,
                        ) &&
                        (homeHotpathPageText.includes(
                          homeHotpathExpectedSummaryText,
                        ) ||
                          homeHotpathPageText.includes(
                            homeHotpathExpectedDoneText,
                          ))
                      : isImageCommandScenario
                        ? summary.guiImageCommandCompleted?.hasPrompt ===
                            true &&
                          (summary.guiImageCommandCompleted
                            ?.imageTaskCardVisible === true ||
                            summary.guiImageCommandCompleted?.hasDoneText ===
                              true ||
                            pageText.includes(IMAGE_COMMAND_DONE_TEXT)) &&
                          (summary.guiImageCommandCompleted?.hasSkillName ===
                            true ||
                            summary.guiImageCommandCompleted
                              ?.hasCreateTaskTool === true)
                        : isWebToolsRenderingScenario
                          ? summary.guiWebToolsRenderingCompleted?.hasPrompt ===
                              true &&
                            summary.guiWebToolsRenderingCompleted
                              ?.historicalTimelinePreviewVisible === true &&
                            summary.guiWebToolsRenderingCompleted
                              ?.hasAssistantSummary === true
                          : isReasoningFirstVisibleScenario
                            ? pageText.includes(
                                REASONING_FIRST_VISIBLE_PROMPT,
                              ) &&
                              pageText.includes(REASONING_FIRST_VISIBLE_TEXT) &&
                              (pageText.includes(
                                REASONING_FIRST_VISIBLE_FINAL_TEXT,
                              ) ||
                                pageText.includes(
                                  REASONING_FIRST_VISIBLE_DONE_TEXT,
                                ))
                            : isLiveTailLayoutScenario
                              ? isLiveTailCommitScenario
                                ? summary.guiLiveTailVisualOracle?.hasPrompt ===
                                    true &&
                                  summary.guiLiveTailVisualOracle
                                    ?.hasFirstText === true &&
                                  summary.guiLiveTailVisualOracle
                                    ?.hasOverflowMarker === true &&
                                  summary.guiLiveTailVisualOracle
                                    ?.hasTableTail === true &&
                                  summary.guiLiveTailVisualOracle
                                    ?.hasDoneText === true
                                : summary.guiElectronResizeReflowCompleted
                                    ?.hasPrompt === true &&
                                  summary.electronResizeReflowLayout?.snapshots
                                    ?.restored?.hasTableTail === true
                              : isApprovalRequestResumeScenario
                                ? pageText.includes(
                                    APPROVAL_REQUEST_RESUME_PROMPT,
                                  ) &&
                                  pageText.includes(
                                    APPROVAL_REQUEST_RESUME_RESULT_TEXT,
                                  ) &&
                                  pageText.includes(
                                    APPROVAL_REQUEST_RESUME_DONE_TEXT,
                                  ) &&
                                  !pageText.includes(
                                    LEGACY_RESPOND_ACTION_METHOD,
                                  )
                                : isApprovalRequestDecisionScenario
                                  ? approvalRequestDecisionPageText.includes(
                                      APPROVAL_REQUEST_RESUME_PROMPT,
                                    ) &&
                                    !approvalRequestDecisionPageText.includes(
                                      LEGACY_RESPOND_ACTION_METHOD,
                                    ) &&
                                    approvalRequestDecisionHistoricalDetailsHidden &&
                                    (isApprovalRequestDeclineScenario
                                      ? approvalRequestDecisionPageText.includes(
                                          APPROVAL_REQUEST_DECLINE_RESULT_TEXT,
                                        ) &&
                                        approvalRequestDecisionPageText.includes(
                                          APPROVAL_REQUEST_DECLINE_DONE_TEXT,
                                        ) &&
                                        approvalRequestDecisionDeclinedReadModel?.includesToolResult ===
                                          false
                                      : (approvalRequestDecisionPageText.includes(
                                          APPROVAL_REQUEST_CANCEL_DONE_TEXT,
                                        ) ||
                                          approvalRequestDecisionCanceledReadModel?.latestTurnCanceled ===
                                            true) &&
                                        approvalRequestDecisionCanceledReadModel?.includesToolResult ===
                                          false)
                                  : isApprovalRequestFullAccessScenario
                                    ? pageText.includes(
                                        APPROVAL_REQUEST_FULL_ACCESS_PROMPT,
                                      ) &&
                                      pageText.includes(
                                        APPROVAL_REQUEST_FULL_ACCESS_RESULT_TEXT,
                                      ) &&
                                      pageText.includes(
                                        APPROVAL_REQUEST_FULL_ACCESS_DONE_TEXT,
                                      ) &&
                                      !pageText.includes(
                                        LEGACY_RESPOND_ACTION_METHOD,
                                      )
                                    : isTerminalCanceledAfterAnswerScenario
                                      ? pageText.includes(
                                          TERMINAL_CANCELED_AFTER_ANSWER_PROMPT,
                                        ) &&
                                        pageText.includes(
                                          TERMINAL_CANCELED_AFTER_ANSWER_PARTIAL_TEXT,
                                        )
                                      : isTerminalFailedAfterAnswerScenario
                                        ? pageText.includes(
                                            TERMINAL_FAILED_AFTER_ANSWER_PROMPT,
                                          ) &&
                                          pageText.includes(
                                            TERMINAL_FAILED_AFTER_ANSWER_PARTIAL_TEXT,
                                          )
                                        : isTerminalStaleGuardScenario
                                          ? pageText.includes(
                                              TERMINAL_STALE_GUARD_FIRST_PROMPT,
                                            ) &&
                                            pageText.includes(
                                              TERMINAL_STALE_GUARD_SECOND_PROMPT,
                                            ) &&
                                            pageText.includes(
                                              TERMINAL_STALE_GUARD_SECOND_TEXT,
                                            ) &&
                                            pageText.includes(
                                              TERMINAL_STALE_GUARD_DONE_TEXT,
                                            ) &&
                                            pageText.includes(
                                              TERMINAL_STALE_GUARD_FIRST_DONE_TEXT,
                                            ) &&
                                            !pageText.includes(
                                              TERMINAL_STALE_GUARD_STALE_DONE_TEXT,
                                            )
                                          : isTypedErrorRetryScenario
                                            ? pageText.includes(
                                                isTypedErrorRetrySuccessScenario
                                                  ? TYPED_ERROR_RETRY_SUCCESS_PROMPT
                                                  : TYPED_ERROR_RETRY_FAILURE_PROMPT,
                                              ) &&
                                              pageText.includes(
                                                isTypedErrorRetrySuccessScenario
                                                  ? TYPED_ERROR_RETRY_SUCCESS_TEXT
                                                  : TYPED_ERROR_RETRY_FAILURE_PARTIAL_TEXT,
                                              )
                                            : isMcpStructuredContentScenario
                                              ? summary
                                                  .guiMcpStructuredContentCompleted
                                                  ?.hasPrompt === true &&
                                                ((summary
                                                  .guiMcpStructuredContentCompleted
                                                  ?.hasStructuredAnswer ===
                                                  true &&
                                                  summary
                                                    .guiMcpStructuredContentCompleted
                                                    ?.hasReferenceId ===
                                                    true) ||
                                                  summary
                                                    .guiMcpStructuredContentCompleted
                                                    ?.hasDoneText === true ||
                                                  summary
                                                    .guiMcpStructuredContentCompleted
                                                    ?.terminalDetailsCompacted ===
                                                    true) &&
                                                summary
                                                  .guiMcpStructuredContentCompleted
                                                  ?.envelopeVisible === false
                                              : isMediaReferenceScenario
                                                ? pageText.includes(
                                                    MEDIA_REFERENCE_PROMPT,
                                                  ) &&
                                                  pageText.includes(
                                                    MEDIA_REFERENCE_SUMMARY_TEXT,
                                                  ) &&
                                                  pageText.includes(
                                                    MEDIA_REFERENCE_URI,
                                                  )
                                                : isSkillsRuntimeScenario
                                                  ? summary
                                                      .guiSkillsRuntimeCompleted
                                                      ?.hasPrompt === true &&
                                                    (summary
                                                      .guiSkillsRuntimeCompleted
                                                      ?.hasAssistantSummary ===
                                                      true ||
                                                      summary
                                                        .guiSkillsRuntimeCompleted
                                                        ?.hasDoneText ===
                                                        true) &&
                                                    summary
                                                      .guiExplicitSkillsRuntimeCompleted
                                                      ?.hasPrompt === true &&
                                                    (summary
                                                      .guiExplicitSkillsRuntimeCompleted
                                                      ?.hasAssistantSummary ===
                                                      true ||
                                                      summary
                                                        .guiExplicitSkillsRuntimeCompleted
                                                        ?.hasDoneText ===
                                                        true) &&
                                                    summary
                                                      .guiManualEnableSkillsRuntimeCompleted
                                                      ?.hasPrompt === true &&
                                                    (summary
                                                      .guiManualEnableSkillsRuntimeCompleted
                                                      ?.hasAssistantSummary ===
                                                      true ||
                                                      summary
                                                        .guiManualEnableSkillsRuntimeCompleted
                                                        ?.hasDoneText === true)
                                                  : isAnyExpertSkillsRuntimeScenario
                                                    ? isExpertPanelSkillsRuntimeScenario
                                                      ? summary
                                                          .guiExpertPanelSkillsRuntimeCompleted
                                                          ?.hasPrompt ===
                                                          true &&
                                                        (summary
                                                          .guiExpertPanelSkillsRuntimeCompleted
                                                          ?.hasAssistantSummary ===
                                                          true ||
                                                          summary
                                                            .guiExpertPanelSkillsRuntimeCompleted
                                                            ?.hasDoneText ===
                                                            true) &&
                                                        pageText.includes(
                                                          EXPERT_SKILLS_RUNTIME_TITLE,
                                                        )
                                                      : summary
                                                          .guiExpertSkillsRuntimeCompleted
                                                          ?.hasPrompt ===
                                                          true &&
                                                        (summary
                                                          .guiExpertSkillsRuntimeCompleted
                                                          ?.hasAssistantSummary ===
                                                          true ||
                                                          summary
                                                            .guiExpertSkillsRuntimeCompleted
                                                            ?.hasDoneText ===
                                                            true) &&
                                                        pageText.includes(
                                                          EXPERT_SKILLS_RUNTIME_TITLE,
                                                        )
                                                    : pageText.includes(
                                                        NEWS_PROMPT,
                                                      ) &&
                                                      (pageText.includes(
                                                        "今日国际新闻简要整理",
                                                      ) ||
                                                        pageText.includes(
                                                          ASSISTANT_DONE_TEXT,
                                                        )),
    noInvokeErrors: !errorRaw,
    noConsoleErrors: actionableConsoleErrors.length === 0,
    agentUiPerformanceTraceEvidenceAvailable:
      !shouldRequireAgentUiTraceEvidence ||
      agentUiPerformanceTrace?.available === true,
    agentUiPerformanceTraceSeparatesProviderAndClient:
      !shouldRequireTextStreamTraceSeparation ||
      summary.agentUiPerformanceTraceDeferred != null ||
      ((!shouldRequireProviderTextTraceEvidence ||
        agentUiPerformanceTrace?.hasProviderWaitMs === true) &&
        agentUiPerformanceTrace?.hasClientLocalOutputMs === true),
    agentUiPerformanceTraceHasFirstVisibleTextPaint:
      !shouldRequireTextStreamTraceSeparation ||
      summary.agentUiPerformanceTraceDeferred != null ||
      (agentUiPerformanceTrace?.hasFirstVisibleOutputMs === true &&
        agentUiPerformanceTrace?.hasFirstTextDeltaToFirstTextPaintMs === true),
    agentUiPerformanceTraceNoRawPayload:
      agentUiPerformanceTrace == null ||
      (agentUiPerformanceTrace.rawEntriesExported === false &&
        agentUiPerformanceTrace.forbiddenFragmentPresent === false),
    appServerTraceEvidenceAvailable:
      !shouldRequireAgentUiTraceEvidence ||
      appServerTraceEvidence?.available === true,
    appServerTraceEvidenceUsesCurrentMethods:
      !shouldRequireAgentUiTraceEvidence ||
      appServerTraceEvidenceDeferred ||
      (appServerRequestMethods.includes(
        APP_SERVER_METHOD_DIAGNOSTICS_TRACE_LIST,
      ) &&
        appServerRequestMethods.includes(
          APP_SERVER_METHOD_DIAGNOSTICS_TRACE_READ,
        ) &&
        appServerRequestMethods.includes(
          APP_SERVER_METHOD_DIAGNOSTICS_TRACE_EXPORT,
        )),
    appServerTraceSupportBundleOptInUsesCurrentMethod:
      !shouldRequireAgentUiTraceEvidence ||
      appServerTraceEvidenceDeferred ||
      appServerRequestMethods.includes(
        APP_SERVER_METHOD_DIAGNOSTICS_SUPPORT_BUNDLE_EXPORT,
      ),
    appServerTraceEvidenceSeparatesProviderAndServer:
      !shouldRequireTextStreamTraceSeparation ||
      appServerTraceEvidenceDeferred ||
      ((!shouldRequireProviderTextTraceEvidence ||
        appServerTraceEvidence?.hasProviderFirstTextDelta === true) &&
        appServerTraceEvidence?.hasAppServerMessageDelta === true),
    appServerTraceEvidenceHasProviderFirstTextCheckpoint:
      !shouldRequireProviderTextTraceEvidence ||
      appServerTraceEvidenceDeferred ||
      appServerTraceEvidence?.hasProviderFirstTextDelta === true,
    appServerTraceEvidenceHasMessageDeltaCheckpoint:
      !shouldRequireTextStreamTraceSeparation ||
      appServerTraceEvidenceDeferred ||
      appServerTraceEvidence?.hasAppServerMessageDelta === true,
    appServerTraceEvidenceHasProviderWaitMs:
      !shouldRequireProviderTextTraceEvidence ||
      appServerTraceEvidenceDeferred ||
      appServerTraceEvidence?.hasProviderWaitMs === true,
    appServerTraceEvidenceHasServerEmissionTimestamp:
      !shouldRequireTextStreamTraceSeparation ||
      appServerTraceEvidenceDeferred ||
      appServerTraceEvidence?.hasServerEventEmittedAt === true,
    appServerTraceEvidenceHasW3cCarrier:
      !shouldRequireAgentUiTraceEvidence ||
      appServerTraceEvidenceDeferred ||
      appServerTraceEvidence?.hasW3cTraceContext === true,
    appServerTraceEvidenceExportedSummaryOnly:
      !shouldRequireAgentUiTraceEvidence ||
      appServerTraceEvidenceDeferred ||
      (appServerTraceEvidence?.export?.exported === true &&
        appServerTraceEvidence?.export?.includedSections?.includes(
          "trace/events.jsonl",
        ) === true &&
        appServerTraceEvidence?.export?.omittedSections?.includes(
          "assistant delta text",
        ) === true &&
        appServerTraceEvidence?.export?.redactionMode === "summary_only"),
    appServerTraceSupportBundleOptInSummaryOnly:
      !shouldRequireAgentUiTraceEvidence ||
      appServerTraceEvidenceDeferred ||
      (appServerTraceEvidence?.supportBundleWithTrace?.bundlePathExists ===
        true &&
        appServerTraceEvidence?.supportBundleWithTrace?.traceExportIncluded ===
          true &&
        appServerTraceEvidence?.supportBundleWithTrace?.rawTraceJsonlOmitted ===
          true),
    appServerTraceEvidenceNoRawPayload:
      appServerTraceEvidence == null ||
      appServerTraceEvidence.forbiddenFragmentPresent === false,
  };
  return commonAssertions;
}
