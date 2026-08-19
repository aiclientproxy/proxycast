import {
  APP_SERVER_METHOD_AGENT_SESSION_ACTION_RESPOND,
  APP_SERVER_METHOD_SESSION_TURN_CANCEL,
  APPROVAL_REQUEST_DECLINE_RESULT_TEXT,
  APPROVAL_REQUEST_FULL_ACCESS_DONE_TEXT,
  APPROVAL_REQUEST_FULL_ACCESS_PROMPT,
  APPROVAL_REQUEST_FULL_ACCESS_RESULT_TEXT,
  APPROVAL_REQUEST_RESUME_PROMPT,
  APPROVAL_REQUEST_RESUME_REQUEST_ID,
  APPROVAL_REQUEST_RESUME_RESULT_TEXT,
  APPROVAL_REQUEST_RESUME_TOOL_CALL_ID,
} from "./claw-chat-current-fixture-constants.mjs";

const LEGACY_RESPOND_ACTION_METHOD = [
  "agent",
  "runtime",
  "respond",
  "action",
].join("_");

function emitEventTypesForTurn(backendLedger, turnId) {
  return (Array.isArray(backendLedger) ? backendLedger : [])
    .filter((entry) => entry?.kind === "backendEmit" && entry.turnId === turnId)
    .flatMap((entry) =>
      Array.isArray(entry.eventTypes) ? entry.eventTypes : [],
    );
}

function decisionServerRequestResponseScoped(
  summary,
  threadId,
  turnId,
  expectedDecision,
) {
  const lifecycle = summary.approvalRequestDecisionServerRequestResponse;
  return (
    lifecycle?.request?.threadId === threadId &&
    lifecycle?.request?.turnId === turnId &&
    lifecycle?.response?.id === lifecycle?.request?.id &&
    lifecycle?.response?.threadId === threadId &&
    lifecycle?.response?.turnId === turnId &&
    lifecycle?.response?.decision === expectedDecision &&
    lifecycle?.responseMatchesRequest === true
  );
}

function serverRequestLifecycleResolved(
  summary,
  expectedWireDecision,
  threadId,
  turnId,
) {
  return (
    summary?.request?.method === "item/commandExecution/requestApproval" &&
    summary?.request?.threadId === threadId &&
    summary?.request?.turnId === turnId &&
    summary?.response?.decision === expectedWireDecision &&
    summary?.resolved?.method === "serverRequest/resolved" &&
    summary?.resolved?.threadId === threadId &&
    summary?.responseMatchesResolved === true &&
    summary?.responseBeforeResolved === true &&
    summary?.resolvedBeforeRuntimeTerminal === true
  );
}

function hostInterruptLifecycleResolved(summary, threadId, turnId) {
  return (
    summary?.request?.method === "item/commandExecution/requestApproval" &&
    summary?.request?.threadId === threadId &&
    summary?.request?.turnId === turnId &&
    summary?.resolved?.method === "serverRequest/resolved" &&
    summary?.resolved?.threadId === threadId &&
    summary?.noRendererResponse === true &&
    summary?.resolvedBeforeRuntimeTerminal === true
  );
}

function noLegacyRuntimeRespond(appServerRequestMethods) {
  return !appServerRequestMethods.includes(LEGACY_RESPOND_ACTION_METHOD);
}

function historicalApprovalDetailsHidden(snapshot) {
  const shape = snapshot?.approvalRecordShape;
  return (
    shape?.recordCount === 0 &&
    Array.isArray(shape?.texts) &&
    shape.texts.length === 0
  );
}

function currentFailureStatusHidden(snapshot) {
  return !String(snapshot?.completionScope?.assistantText ?? "").includes(
    "当前处理失败",
  );
}

function turnStartApprovalPolicy(turnStart) {
  return (
    turnStart?.runtimeRequest?.approval_policy ??
    turnStart?.runtimeRequest?.approvalPolicy ??
    turnStart?.approvalPolicy ??
    turnStart?.approval_policy ??
    null
  );
}

function turnStartSandboxPolicy(turnStart) {
  return (
    turnStart?.runtimeRequest?.sandbox_policy ??
    turnStart?.runtimeRequest?.sandboxPolicy ??
    turnStart?.sandboxPolicy ??
    turnStart?.sandbox_policy ??
    null
  );
}

export function buildApprovalRequestFullAccessScenarioAssertions({
  appServerRequestMethods,
  approvalRequestFullAccessTurnStart,
  pageText,
  summary,
}) {
  return {
    approvalRequestFullAccessPromptReachedBackend:
      approvalRequestFullAccessTurnStart?.inputText ===
      APPROVAL_REQUEST_FULL_ACCESS_PROMPT,
    approvalRequestFullAccessUsesFullAccessPolicy:
      turnStartApprovalPolicy(approvalRequestFullAccessTurnStart) === "never" &&
      turnStartSandboxPolicy(approvalRequestFullAccessTurnStart) ===
        "danger-full-access",
    guiApprovalRequestFullAccessInputSubmitted:
      summary.approvalRequestFullAccessInputSend?.afterFill
        ?.promptVisibleInTextarea === true &&
      summary.approvalRequestFullAccessInputSend?.clicked?.clicked === true,
    guiApprovalRequestFullAccessCompleted:
      summary.guiApprovalRequestFullAccessCompleted?.hasPrompt === true &&
      (summary.guiApprovalRequestFullAccessCompleted?.hasAssistantSummary ===
        true ||
        summary.guiApprovalRequestFullAccessCompleted?.hasDoneText === true) &&
      summary.guiApprovalRequestFullAccessCompleted?.textareaVisible === true &&
      summary.guiApprovalRequestFullAccessCompleted?.textareaDisabled ===
        false &&
      summary.guiApprovalRequestFullAccessCompleted?.stopButtonVisible ===
        false &&
      pageText.includes(APPROVAL_REQUEST_FULL_ACCESS_RESULT_TEXT) &&
      pageText.includes(APPROVAL_REQUEST_FULL_ACCESS_DONE_TEXT),
    guiApprovalRequestFullAccessNoApprovalPrompt:
      summary.guiApprovalRequestFullAccessNoApproval?.approvalPromptVisible ===
        false &&
      summary.guiApprovalRequestFullAccessNoApproval
        ?.includesRuntimePermissionPrompt === false &&
      summary.guiApprovalRequestFullAccessNoApproval
        ?.includesRuntimeApprovalPrompt === false &&
      summary.guiApprovalRequestFullAccessNoApproval?.textareaVisible ===
        true &&
      summary.guiApprovalRequestFullAccessNoApproval?.textareaDisabled ===
        false,
    guiApprovalRequestFullAccessNoApprovalRecord:
      summary.guiApprovalRequestFullAccessCompleted?.approvalRecordShape
        ?.recordCount === 0 &&
      summary.guiApprovalRequestFullAccessNoApproval?.approvalRecordCount === 0,
    readModelApprovalRequestFullAccessCompleted:
      summary.readModelApprovalRequestFullAccessCompleted?.latestTurnStatus ===
        "completed" &&
      summary.readModelApprovalRequestFullAccessCompleted?.includesPrompt ===
        true &&
      summary.readModelApprovalRequestFullAccessCompleted
        ?.includesAssistantSummary === true &&
      summary.readModelApprovalRequestFullAccessCompleted
        ?.includesAssistantDone === true,
    readModelApprovalRequestFullAccessNoApprovalRequest:
      summary.readModelApprovalRequestFullAccessCompleted
        ?.pendingRequestCount === 0 &&
      summary.readModelApprovalRequestFullAccessCompleted
        ?.includesApprovalRequest === false &&
      summary.readModelApprovalRequestFullAccessCompleted
        ?.includesActionRequired === false &&
      summary.readModelApprovalRequestFullAccessCompleted
        ?.includesActionResolved === false &&
      summary.readModelApprovalRequestFullAccessCompleted
        ?.includesApprovalPrompt === false,
    approvalRequestFullAccessNoActionRespond: !appServerRequestMethods.includes(
      APP_SERVER_METHOD_AGENT_SESSION_ACTION_RESPOND,
    ),
    approvalRequestFullAccessNoLegacyRuntimeRespond: noLegacyRuntimeRespond(
      appServerRequestMethods,
    ),
  };
}

export function buildApprovalRequestResumeScenarioAssertions({
  appServerRequestMethods,
  approvalRequestResumeTurnStart,
  backendLedger,
  pageText,
  summary,
}) {
  return {
    approvalRequestResumePromptReachedBackend:
      approvalRequestResumeTurnStart?.inputText ===
      APPROVAL_REQUEST_RESUME_PROMPT,
    guiApprovalRequestResumeInputSubmitted:
      summary.approvalRequestResumeInputSend?.afterFill
        ?.promptVisibleInTextarea === true &&
      summary.approvalRequestResumeInputSend?.clicked?.clicked === true,
    guiApprovalRequestResumePendingVisible:
      summary.approvalRequestResumePendingGui?.hasSection === true &&
      summary.approvalRequestResumePendingGui?.hasCurrentInteractionLayer ===
        true &&
      summary.approvalRequestResumePendingGui?.interactionKind ===
        "approval" &&
      summary.approvalRequestResumePendingGui?.hasApprovalContent === true &&
      summary.approvalRequestResumePendingGui?.hasPrompt === true &&
      summary.approvalRequestResumePendingGui?.hasToolName === false &&
      summary.approvalRequestResumePendingGui?.hasCommand === false &&
      summary.approvalRequestResumePendingGui?.hasDetails === false &&
      summary.approvalRequestResumePendingGui?.hasPreformattedArguments ===
        false &&
      summary.approvalRequestResumePendingGui?.textareaVisible === true &&
      summary.approvalRequestResumePendingGui?.textareaDisabled === false &&
      summary.approvalRequestResumePendingGui?.singleLine === true &&
      summary.approvalRequestResumePendingGui?.approveButtonVisible === true &&
      summary.approvalRequestResumePendingGui?.approveButtonDisabled === false,
    readModelApprovalRequestResumePending:
      summary.approvalRequestResumePendingReadModel?.hasPendingRequest ===
        true &&
      summary.approvalRequestResumePendingReadModel?.payloadActionType ===
        "tool_confirmation" &&
      summary.approvalRequestResumePendingReadModel?.includesRequestId ===
        true &&
      summary.approvalRequestResumePendingReadModel?.includesToolCallId ===
        true,
    approvalRequestResumeUsedCurrentServerRequestResponse:
      summary.approvalRequestResumeServerRequestResponse?.request?.method ===
        "item/commandExecution/requestApproval" &&
      summary.approvalRequestResumeServerRequestResponse?.request?.threadId ===
        summary.threadId &&
      summary.approvalRequestResumeServerRequestResponse?.request?.turnId ===
        approvalRequestResumeTurnStart?.turnId &&
      summary.approvalRequestResumeServerRequestResponse?.response?.decision ===
        "acceptForSession" &&
      summary.approvalRequestResumeServerRequestResponse
        ?.responseMatchesRequest === true,
    approvalRequestResumeBackendResponseScoped:
      summary.approvalRequestResumeBackendActionRespond?.sessionId ===
        summary.sessionId &&
      summary.approvalRequestResumeBackendActionRespond?.sessionId ===
        approvalRequestResumeTurnStart?.sessionId &&
      summary.approvalRequestResumeBackendActionRespond?.threadId ===
        summary.threadId &&
      summary.approvalRequestResumeBackendActionRespond?.actionScope
        ?.threadId === summary.threadId &&
      summary.approvalRequestResumeBackendActionRespond?.actionScope?.turnId ===
        approvalRequestResumeTurnStart?.turnId,
    approvalRequestResumeServerRequestResolved: serverRequestLifecycleResolved(
      summary.approvalRequestResumeServerRequestLifecycle,
      "acceptForSession",
      summary.threadId,
      approvalRequestResumeTurnStart?.turnId,
    ),
    approvalRequestResumeBackendActionRespondObserved:
      summary.approvalRequestResumeBackendActionRespond?.requestId ===
        APPROVAL_REQUEST_RESUME_REQUEST_ID &&
      summary.approvalRequestResumeBackendActionRespond?.actionType ===
        "tool_confirmation" &&
      summary.approvalRequestResumeBackendActionRespond?.decision ===
        "allow_for_session" &&
      summary.approvalRequestResumeBackendActionRespond?.decisionScope ===
        "session" &&
      summary.approvalRequestResumeBackendActionRespond?.turnId ===
        approvalRequestResumeTurnStart?.turnId,
    approvalRequestResumePendingCleared:
      summary.readModelApprovalRequestResumeCompleted?.pendingRequestCount ===
      0,
    guiApprovalRequestResumeCompleted:
      summary.guiApprovalRequestResumeCompleted?.hasPrompt === true &&
      (summary.guiApprovalRequestResumeCompleted?.hasAssistantSummary ===
        true ||
        summary.guiApprovalRequestResumeCompleted?.hasDoneText === true) &&
      summary.guiApprovalRequestResumeCompleted?.textareaVisible === true &&
      summary.guiApprovalRequestResumeCompleted?.textareaDisabled === false &&
      summary.guiApprovalRequestResumeCompleted?.stopButtonVisible === false &&
      currentFailureStatusHidden(summary.guiApprovalRequestResumeCompleted),
    guiApprovalRequestResumeHistoricalDetailsHidden:
      historicalApprovalDetailsHidden(
        summary.guiApprovalRequestResumeCompleted,
      ),
    readModelApprovalRequestResumeCompleted:
      summary.readModelApprovalRequestResumeCompleted?.latestTurnStatus ===
        "completed" &&
      summary.readModelApprovalRequestResumeCompleted?.includesPrompt ===
        true &&
      summary.readModelApprovalRequestResumeCompleted?.includesRequestId ===
        false &&
      summary.readModelApprovalRequestResumeCompleted
        ?.includesApprovalPrompt === false &&
      summary.readModelApprovalRequestResumeCompleted
        ?.includesActionResolved === false &&
      summary.readModelApprovalRequestResumeCompleted?.includesToolCallId ===
        true &&
      summary.readModelApprovalRequestResumeCompleted?.includesToolResult ===
        true &&
      summary.readModelApprovalRequestResumeCompleted
        ?.includesAssistantSummary === true,
    approvalRequestResumeNoLegacyRuntimeRespond:
      noLegacyRuntimeRespond(appServerRequestMethods) &&
      !appServerRequestMethods.includes(
        APP_SERVER_METHOD_AGENT_SESSION_ACTION_RESPOND,
      ) &&
      pageText.includes(APPROVAL_REQUEST_RESUME_RESULT_TEXT),
  };
}

export function buildApprovalRequestDecisionScenarioAssertions({
  appServerRequestMethods,
  approvalRequestResumeTurnStart,
  backendLedger,
  isApprovalRequestCancelScenario,
  isApprovalRequestDeclineScenario,
  summary,
}) {
  const expectedDecision = isApprovalRequestCancelScenario
    ? "cancel"
    : "decline";
  const turnId = approvalRequestResumeTurnStart?.turnId;
  const emitTypes = emitEventTypesForTurn(backendLedger, turnId);
  const noToolResult =
    summary.readModelApprovalRequestDeclineCompleted?.includesToolResult !==
      true &&
    summary.readModelApprovalRequestCancelCanceled?.includesToolResult !== true;
  return {
    approvalRequestDecisionPromptReachedBackend:
      approvalRequestResumeTurnStart?.inputText ===
      APPROVAL_REQUEST_RESUME_PROMPT,
    guiApprovalRequestDecisionInputSubmitted:
      summary.approvalRequestDecisionInputSend?.afterFill
        ?.promptVisibleInTextarea === true &&
      summary.approvalRequestDecisionInputSend?.clicked?.clicked === true,
    guiApprovalRequestDecisionPendingVisible:
      summary.approvalRequestDecisionPendingGui?.hasSection === true &&
      summary.approvalRequestDecisionPendingGui?.hasApprovalContent === true &&
      summary.approvalRequestDecisionPendingGui?.hasPrompt === true &&
      summary.approvalRequestDecisionPendingGui?.hasToolName === false &&
      summary.approvalRequestDecisionPendingGui?.hasCommand === false &&
      summary.approvalRequestDecisionPendingGui?.hasDetails === false &&
      summary.approvalRequestDecisionPendingGui?.hasPreformattedArguments ===
        false &&
      summary.approvalRequestDecisionPendingGui?.textareaVisible === true &&
      summary.approvalRequestDecisionPendingGui?.textareaDisabled === false &&
      summary.approvalRequestDecisionPendingGui?.singleLine === true &&
      (expectedDecision === "cancel"
        ? summary.approvalRequestDecisionPendingGui?.cancelButtonVisible ===
          true
        : summary.approvalRequestDecisionPendingGui?.declineButtonVisible ===
          true),
    readModelApprovalRequestDecisionPending:
      summary.approvalRequestDecisionPendingReadModel?.hasPendingRequest ===
        true &&
      summary.approvalRequestDecisionPendingReadModel?.payloadActionType ===
        "tool_confirmation" &&
      summary.approvalRequestDecisionPendingReadModel?.includesRequestId ===
        true &&
      summary.approvalRequestDecisionPendingReadModel?.includesToolCallId ===
        true,
    approvalRequestDecisionUsedCurrentServerRequestResponse:
      summary.approvalRequestDecisionServerRequestResponse?.request?.method ===
        "item/commandExecution/requestApproval" &&
      summary.approvalRequestDecisionServerRequestResponse?.response
        ?.decision === expectedDecision &&
      summary.approvalRequestDecisionServerRequestResponse
        ?.responseMatchesRequest === true,
    approvalRequestDecisionServerRequestResponseScoped:
      decisionServerRequestResponseScoped(
        summary,
        summary.threadId,
        turnId,
        expectedDecision,
      ),
    approvalRequestDecisionServerRequestResolved:
      serverRequestLifecycleResolved(
        summary.approvalRequestDecisionServerRequestLifecycle,
        expectedDecision,
        summary.threadId,
        turnId,
      ),
    approvalRequestDecisionBackendActionRespondObserved:
      summary.approvalRequestDecisionBackendActionRespond?.requestId ===
        APPROVAL_REQUEST_RESUME_REQUEST_ID &&
      summary.approvalRequestDecisionBackendActionRespond?.actionType ===
        "tool_confirmation" &&
      summary.approvalRequestDecisionBackendActionRespond?.decision ===
        expectedDecision &&
      summary.approvalRequestDecisionBackendActionRespond?.decisionScope ===
        "once" &&
      summary.approvalRequestDecisionBackendActionRespond?.turnId === turnId,
    approvalRequestDecisionPendingCleared:
      (isApprovalRequestDeclineScenario
        ? summary.readModelApprovalRequestDeclineCompleted?.pendingRequestCount
        : summary.readModelApprovalRequestCancelCanceled
            ?.pendingRequestCount) === 0,
    ...(isApprovalRequestDeclineScenario
      ? {
          approvalRequestDeclineNoToolExecuted:
            noToolResult &&
            emitTypes.includes("turn.completed") &&
            !emitTypes.includes("turn.canceled"),
          guiApprovalRequestDeclineCompleted:
            summary.guiApprovalRequestDeclineCompleted?.hasPrompt === true &&
            summary.guiApprovalRequestDeclineCompleted?.hasAssistantSummary ===
              true &&
            summary.guiApprovalRequestDeclineCompleted?.textareaVisible ===
              true &&
            summary.guiApprovalRequestDeclineCompleted?.textareaDisabled ===
              false &&
            summary.guiApprovalRequestDeclineCompleted?.stopButtonVisible ===
              false &&
            currentFailureStatusHidden(
              summary.guiApprovalRequestDeclineCompleted,
            ),
          guiApprovalRequestDeclineHistoricalDetailsHidden:
            historicalApprovalDetailsHidden(
              summary.guiApprovalRequestDeclineCompleted,
            ),
          readModelApprovalRequestDeclineCompleted:
            summary.readModelApprovalRequestDeclineCompleted
              ?.latestTurnStatus === "completed" &&
            summary.readModelApprovalRequestDeclineCompleted?.includesPrompt ===
              true &&
            summary.readModelApprovalRequestDeclineCompleted
              ?.includesRequestId === false &&
            summary.readModelApprovalRequestDeclineCompleted
              ?.includesApprovalPrompt === false &&
            summary.readModelApprovalRequestDeclineCompleted
              ?.includesToolCallId === true &&
            summary.readModelApprovalRequestDeclineCompleted
              ?.includesActionResolved === false &&
            summary.readModelApprovalRequestDeclineCompleted
              ?.includesDecision === false &&
            summary.readModelApprovalRequestDeclineCompleted
              ?.includesDeclineResult === true &&
            summary.readModelApprovalRequestDeclineCompleted
              ?.includesDeclineDone === true &&
            summary.readModelApprovalRequestDeclineCompleted
              ?.includesToolResult === false &&
            summary.guiApprovalRequestDeclineCompleted?.hasAssistantSummary ===
              true,
        }
      : {
          approvalRequestCancelNoToolExecuted:
            noToolResult &&
            emitTypes.includes("turn.canceled") &&
            !emitTypes.includes("turn.completed"),
          guiApprovalRequestCancelCompleted:
            summary.guiApprovalRequestCancelCompleted?.hasPrompt === true &&
            summary.guiApprovalRequestCancelCompleted?.textareaVisible ===
              true &&
            summary.guiApprovalRequestCancelCompleted?.textareaDisabled ===
              false &&
            summary.guiApprovalRequestCancelCompleted?.stopButtonVisible ===
              false &&
            currentFailureStatusHidden(
              summary.guiApprovalRequestCancelCompleted,
            ),
          guiApprovalRequestCancelHistoricalDetailsHidden:
            historicalApprovalDetailsHidden(
              summary.guiApprovalRequestCancelCompleted,
            ),
          readModelApprovalRequestCancelCanceled:
            summary.readModelApprovalRequestCancelCanceled
              ?.latestTurnCanceled === true &&
            summary.readModelApprovalRequestCancelCanceled?.includesPrompt ===
              true &&
            summary.readModelApprovalRequestCancelCanceled
              ?.includesToolCallId === true &&
            summary.readModelApprovalRequestCancelCanceled?.includesCanceled ===
              true &&
            summary.readModelApprovalRequestCancelCanceled
              ?.includesToolResult === false,
        }),
    approvalRequestDecisionNoRendererActionRespond:
      noLegacyRuntimeRespond(appServerRequestMethods) &&
      !appServerRequestMethods.includes(
        APP_SERVER_METHOD_AGENT_SESSION_ACTION_RESPOND,
      ),
  };
}

export function buildApprovalRequestHostInterruptScenarioAssertions({
  appServerRequestMethods,
  approvalRequestResumeTurnStart,
  summary,
}) {
  const turnId = approvalRequestResumeTurnStart?.turnId;
  const pendingGui = summary.approvalRequestDecisionPendingGui;
  const pendingReadModel = summary.approvalRequestDecisionPendingReadModel;
  const guiCanceled = summary.guiApprovalRequestCancelCompleted;
  const readModelCanceled = summary.readModelApprovalRequestCancelCanceled;
  const interruptRequest = summary.approvalRequestHostInterruptRequest;
  return {
    approvalRequestHostInterruptPromptReachedBackend:
      approvalRequestResumeTurnStart?.inputText ===
      APPROVAL_REQUEST_RESUME_PROMPT,
    guiApprovalRequestHostInterruptInputSubmitted:
      summary.approvalRequestDecisionInputSend?.afterFill
        ?.promptVisibleInTextarea === true &&
      summary.approvalRequestDecisionInputSend?.clicked?.clicked === true,
    guiApprovalRequestHostInterruptPendingVisible:
      pendingGui?.hasSection === true &&
      pendingGui?.hasApprovalContent === true &&
      pendingGui?.hasPrompt === true &&
      pendingGui?.textareaVisible === true &&
      pendingGui?.textareaDisabled === false &&
      pendingGui?.singleLine === true,
    readModelApprovalRequestHostInterruptPending:
      pendingReadModel?.hasPendingRequest === true &&
      pendingReadModel?.payloadActionType === "tool_confirmation" &&
      pendingReadModel?.includesRequestId === true &&
      pendingReadModel?.includesToolCallId === true,
    approvalRequestHostInterruptUsedCurrentMethod:
      appServerRequestMethods.includes(APP_SERVER_METHOD_SESSION_TURN_CANCEL) &&
      interruptRequest?.method === APP_SERVER_METHOD_SESSION_TURN_CANCEL,
    approvalRequestHostInterruptPayloadScoped:
      interruptRequest?.params?.threadId === summary.threadId &&
      interruptRequest?.params?.turnId === turnId,
    approvalRequestHostInterruptServerRequestResolved:
      hostInterruptLifecycleResolved(
        summary.approvalRequestHostInterruptLifecycle,
        summary.threadId,
        turnId,
      ),
    approvalRequestHostInterruptNoRendererResponse:
      summary.approvalRequestHostInterruptLifecycle?.responseCount === 0,
    approvalRequestHostInterruptNoBackendActionRespond:
      summary.approvalRequestHostInterruptBackend?.actionRespondCount === 0 &&
      !appServerRequestMethods.includes(
        APP_SERVER_METHOD_AGENT_SESSION_ACTION_RESPOND,
      ),
    approvalRequestHostInterruptPendingCleared:
      readModelCanceled?.pendingRequestCount === 0,
    guiApprovalRequestHostInterruptCanceled:
      guiCanceled?.hasPrompt === true &&
      guiCanceled?.textareaVisible === true &&
      guiCanceled?.textareaDisabled === false &&
      guiCanceled?.stopButtonVisible === false &&
      currentFailureStatusHidden(guiCanceled),
    readModelApprovalRequestHostInterruptCanceled:
      readModelCanceled?.latestTurnCanceled === true &&
      readModelCanceled?.includesPrompt === true &&
      readModelCanceled?.includesToolCallId === true &&
      readModelCanceled?.includesCanceled === true &&
      readModelCanceled?.includesToolResult === false,
    approvalRequestHostInterruptCanonicalEventOrder:
      summary.approvalRequestHostInterruptCanonicalEvents?.ordered === true,
  };
}
