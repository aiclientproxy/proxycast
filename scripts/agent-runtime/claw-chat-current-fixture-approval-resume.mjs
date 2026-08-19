import fs from "node:fs";
import path from "node:path";

import {
  APP_SERVER_METHOD_SESSION_TURN_CANCEL,
  APPROVAL_REQUEST_CANCEL_DONE_TEXT,
  APPROVAL_REQUEST_DECLINE_DONE_TEXT,
  APPROVAL_REQUEST_DECLINE_RESULT_TEXT,
  APPROVAL_REQUEST_FULL_ACCESS_DONE_TEXT,
  APPROVAL_REQUEST_FULL_ACCESS_PROMPT,
  APPROVAL_REQUEST_FULL_ACCESS_RESULT_TEXT,
  APPROVAL_REQUEST_RESUME_DONE_TEXT,
  APPROVAL_REQUEST_RESUME_PROMPT,
  APPROVAL_REQUEST_RESUME_REQUEST_ID,
  APPROVAL_REQUEST_RESUME_RESULT_TEXT,
} from "./claw-chat-current-fixture-constants.mjs";
import {
  waitForBackendLedgerEntry,
  waitForBackendLedgerTurnStart,
} from "./claw-chat-current-fixture-backend-ledger.mjs";
import {
  clickApprovalDecisionButton,
  clickApprovalApproveButton,
  waitForGuiApprovalPromptAbsent,
  waitForGuiApprovalPending,
} from "./claw-chat-current-fixture-approval-gui.mjs";
import {
  summarizeApprovalCompletedReadModel,
  summarizeApprovalDecisionReadModel,
  waitForApprovalPendingReadModel,
} from "./claw-chat-current-fixture-approval-read-model.mjs";
import {
  readApprovalServerRequestLifecycleSnapshot,
  waitForApprovalHostInterruptLifecycle,
  waitForApprovalServerRequestResponse,
  waitForApprovalServerRequestLifecycle,
} from "./claw-chat-current-fixture-approval-trace.mjs";
import {
  waitForGuiChatCanceled,
  waitForGuiChatCompleted,
} from "./claw-chat-current-fixture-gui-completion-waits.mjs";
import {
  sendPromptFromGui,
  setInputbarAccessMode,
} from "./claw-chat-current-fixture-gui-actions.mjs";
import { readModelLatestTurnStatus } from "./claw-chat-current-fixture-read-model-core.mjs";
import {
  waitForSessionReadCanceled,
  waitForSessionReadCompleted,
} from "./claw-chat-current-fixture-read-model-waits.mjs";
import { invokeAppServerFromPage } from "./claw-chat-current-fixture-rpc.mjs";
import { readJsonl, sanitizeJson } from "./claw-chat-current-fixture-utils.mjs";

function summarizeFullAccessReadModel(readModel) {
  const detail = readModel?.detail ?? readModel ?? {};
  const threadRead = detail?.thread_read ?? detail?.threadRead ?? {};
  const pendingRequests = [
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
  const serialized = JSON.stringify(readModel || {});
  return sanitizeJson({
    pendingRequestCount: pendingRequests.length,
    latestTurnStatus: readModelLatestTurnStatus(readModel),
    includesPrompt: serialized.includes(APPROVAL_REQUEST_FULL_ACCESS_PROMPT),
    includesAssistantSummary: serialized.includes(
      APPROVAL_REQUEST_FULL_ACCESS_RESULT_TEXT,
    ),
    includesAssistantDone: serialized.includes(
      APPROVAL_REQUEST_FULL_ACCESS_DONE_TEXT,
    ),
    includesApprovalRequest: serialized.includes("approval_request"),
    includesActionRequired:
      serialized.includes("action.required") ||
      serialized.includes("action_required"),
    includesActionResolved:
      serialized.includes("action.resolved") ||
      serialized.includes("action_resolved"),
    includesApprovalPrompt: serialized.includes("需要确认浏览器控制权限"),
  });
}

function summarizeBackendActionRespond(entry) {
  return {
    sessionId: entry.sessionId ?? null,
    threadId: entry.threadId ?? null,
    turnId: entry.turnId ?? null,
    requestId: entry.requestId ?? null,
    actionType: entry.actionType ?? null,
    decision: entry.decision ?? null,
    decisionScope: entry.decisionScope ?? null,
    confirmed: entry.confirmed ?? null,
    response: entry.response ?? null,
    actionScope: entry.actionScope ?? null,
  };
}

function summarizeHostInterruptCanonicalEvents({
  agentRoot,
  sessionId,
  turnId,
}) {
  const eventLogPath = path.join(
    agentRoot,
    "runtime",
    "events",
    "sessions",
    `session_${sessionId}.jsonl`,
  );
  const events = fs.existsSync(eventLogPath)
    ? readJsonl(eventLogPath).filter(
        (event) => (event?.turnId ?? event?.turn_id) === turnId,
      )
    : [];
  return summarizeHostInterruptCanonicalEventSequence(events);
}

export function summarizeHostInterruptCanonicalEventSequence(events) {
  const eventTypes = events.map(
    (event) => event?.type ?? event?.eventType ?? event?.event_type ?? null,
  );
  const actionCanceledIndex = eventTypes.findIndex(
    (eventType) => eventType === "action.canceled",
  );
  const itemCompletedCanceledIndex = events.findIndex(
    (event) =>
      (event?.type ?? event?.eventType ?? event?.event_type) ===
        "item.completed" &&
      ["cancelled", "canceled"].includes(
        String(event?.payload?.item?.status ?? "").toLowerCase(),
      ),
  );
  const turnCanceledIndex = eventTypes.findIndex(
    (eventType) => eventType === "turn.canceled",
  );
  return sanitizeJson({
    eventCount: events.length,
    eventTypes,
    actionCanceledIndex,
    itemCompletedCanceledIndex,
    turnCanceledIndex,
    ordered:
      actionCanceledIndex >= 0 &&
      itemCompletedCanceledIndex > actionCanceledIndex &&
      turnCanceledIndex > itemCompletedCanceledIndex,
  });
}

export async function runApprovalRequestHostInterruptScenario({
  page,
  options,
  appServerRequests,
  runtimeEnv,
  logStage,
}) {
  logStage("send-approval-request-host-interrupt-prompt-from-gui");
  const inputSend = sanitizeJson(
    await sendPromptFromGui(page, options, APPROVAL_REQUEST_RESUME_PROMPT),
  );

  logStage("wait-approval-request-host-interrupt-backend-turn-start");
  const backendTurnStart = await waitForBackendLedgerTurnStart(
    runtimeEnv.backendLedgerPath,
    APPROVAL_REQUEST_RESUME_PROMPT,
    options,
  );
  const turnId = backendTurnStart.entry.turnId;

  logStage("wait-gui-approval-request-host-interrupt-pending");
  const pendingGui = await waitForGuiApprovalPending(page, options);

  logStage("wait-read-model-approval-request-host-interrupt-pending");
  const pendingReadModel = await waitForApprovalPendingReadModel(
    page,
    options,
    appServerRequests,
  );

  logStage("invoke-approval-request-host-interrupt");
  await invokeAppServerFromPage(
    page,
    APP_SERVER_METHOD_SESSION_TURN_CANCEL,
    { threadId: options.threadId, turnId },
    appServerRequests,
  );

  logStage("wait-gui-approval-request-host-interrupt-canceled");
  const guiCanceled = sanitizeJson(
    await waitForGuiChatCanceled(page, options, {
      prompt: APPROVAL_REQUEST_RESUME_PROMPT,
    }),
  );

  logStage("wait-read-model-approval-request-host-interrupt-canceled");
  const canceledReadModel = await waitForSessionReadCanceled(
    page,
    options,
    appServerRequests,
    {
      threadId: options.threadId,
      prompt: APPROVAL_REQUEST_RESUME_PROMPT,
    },
  );

  logStage("wait-approval-request-host-interrupt-lifecycle");
  const lifecycle = await waitForApprovalHostInterruptLifecycle(page, options, {
    threadId: options.threadId,
    turnId,
  });
  const backendLedger = readJsonl(runtimeEnv.backendLedgerPath);
  const backendActionRespondCount = backendLedger.filter(
    (entry) =>
      ["actionRespond", "approvalRequestResumeActionRespond"].includes(
        entry?.kind,
      ) && entry?.turnId === turnId,
  ).length;
  const canonicalEvents = summarizeHostInterruptCanonicalEvents({
    agentRoot: runtimeEnv.agentRoot,
    sessionId: backendTurnStart.entry.sessionId,
    turnId,
  });
  const summarizedReadModel = summarizeApprovalDecisionReadModel(
    canceledReadModel,
    "cancel",
  );

  return sanitizeJson({
    approvalRequestDecisionInputSend: inputSend,
    approvalRequestDecisionBackendTurnStart: {
      sessionId: backendTurnStart.entry.sessionId ?? null,
      threadId: backendTurnStart.entry.threadId ?? null,
      turnId,
      inputText: backendTurnStart.entry.inputText ?? null,
    },
    approvalRequestDecisionPendingGui: pendingGui,
    approvalRequestDecisionPendingReadModel: pendingReadModel,
    guiApprovalRequestCancelCompleted: guiCanceled,
    readModelApprovalRequestCancelCanceled: summarizedReadModel,
    approvalRequestHostInterruptRequest: {
      method: APP_SERVER_METHOD_SESSION_TURN_CANCEL,
      params: { threadId: options.threadId, turnId },
    },
    approvalRequestHostInterruptLifecycle: lifecycle,
    approvalRequestHostInterruptBackend: {
      actionRespondCount: backendActionRespondCount,
    },
    approvalRequestHostInterruptCanonicalEvents: canonicalEvents,
  });
}

export async function runApprovalRequestFullAccessScenario({
  page,
  options,
  appServerRequests,
  runtimeEnv,
  logStage,
}) {
  logStage("set-approval-request-full-access-mode");
  const accessModeSet = await setInputbarAccessMode(
    page,
    options,
    "full-access",
  );

  logStage("send-approval-request-full-access-prompt-from-gui");
  const inputSend = sanitizeJson(
    await sendPromptFromGui(page, options, APPROVAL_REQUEST_FULL_ACCESS_PROMPT),
  );

  logStage("wait-approval-request-full-access-backend-turn-start");
  const backendTurnStart = await waitForBackendLedgerTurnStart(
    runtimeEnv.backendLedgerPath,
    APPROVAL_REQUEST_FULL_ACCESS_PROMPT,
    options,
  );

  logStage("wait-gui-approval-request-full-access-completed");
  const guiCompleted = sanitizeJson(
    await waitForGuiChatCompleted(page, options, {
      prompt: APPROVAL_REQUEST_FULL_ACCESS_PROMPT,
      doneText: APPROVAL_REQUEST_FULL_ACCESS_DONE_TEXT,
      summaryText: APPROVAL_REQUEST_FULL_ACCESS_RESULT_TEXT,
    }),
  );

  logStage("wait-gui-approval-request-full-access-no-prompt");
  const noApprovalPrompt = await waitForGuiApprovalPromptAbsent(page, options, {
    requiredText: APPROVAL_REQUEST_FULL_ACCESS_DONE_TEXT,
  });

  logStage("wait-read-model-approval-request-full-access-completed");
  const completedReadModel = await waitForSessionReadCompleted(
    page,
    options,
    appServerRequests,
    {
      prompt: APPROVAL_REQUEST_FULL_ACCESS_PROMPT,
      doneText: APPROVAL_REQUEST_FULL_ACCESS_DONE_TEXT,
      summaryText: APPROVAL_REQUEST_FULL_ACCESS_RESULT_TEXT,
    },
  );

  return sanitizeJson({
    approvalRequestFullAccessAccessModeSet: accessModeSet,
    approvalRequestFullAccessInputSend: inputSend,
    approvalRequestFullAccessBackendTurnStart: {
      sessionId: backendTurnStart.entry.sessionId ?? null,
      threadId: backendTurnStart.entry.threadId ?? null,
      turnId: backendTurnStart.entry.turnId ?? null,
      inputText: backendTurnStart.entry.inputText ?? null,
      approvalPolicy:
        backendTurnStart.entry.runtimeRequest?.approval_policy ??
        backendTurnStart.entry.runtimeRequest?.approvalPolicy ??
        null,
      sandboxPolicy:
        backendTurnStart.entry.runtimeRequest?.sandbox_policy ??
        backendTurnStart.entry.runtimeRequest?.sandboxPolicy ??
        null,
    },
    guiApprovalRequestFullAccessCompleted: guiCompleted,
    guiApprovalRequestFullAccessNoApproval: noApprovalPrompt,
    readModelApprovalRequestFullAccessCompleted:
      summarizeFullAccessReadModel(completedReadModel),
  });
}

export async function runApprovalRequestDecisionScenario({
  page,
  options,
  appServerRequests,
  runtimeEnv,
  logStage,
  decision,
}) {
  const scenarioStage = `approval-request-${decision}`;
  logStage(`send-${scenarioStage}-prompt-from-gui`);
  const inputSend = sanitizeJson(
    await sendPromptFromGui(page, options, APPROVAL_REQUEST_RESUME_PROMPT),
  );

  logStage(`wait-${scenarioStage}-backend-turn-start`);
  const backendTurnStart = await waitForBackendLedgerTurnStart(
    runtimeEnv.backendLedgerPath,
    APPROVAL_REQUEST_RESUME_PROMPT,
    options,
  );

  logStage(`wait-gui-${scenarioStage}-pending`);
  const pendingGui = await waitForGuiApprovalPending(page, options);

  logStage(`wait-read-model-${scenarioStage}-pending`);
  const pendingReadModel = await waitForApprovalPendingReadModel(
    page,
    options,
    appServerRequests,
  );

  logStage(`click-${scenarioStage}`);
  const decisionClick = await clickApprovalDecisionButton(
    page,
    options,
    decision,
  );

  logStage(`wait-${scenarioStage}-server-request-response`);
  const serverRequestResponse = await waitForApprovalServerRequestResponse(
    page,
    options,
    {
      wireDecision: decision,
      threadId: options.threadId,
      turnId: backendTurnStart.entry.turnId,
    },
  );
  if (serverRequestResponse.responseMatchesRequest !== true) {
    throw new Error(
      `审批 ${decision} reverse server request 未记录匹配响应: ${JSON.stringify(
        serverRequestResponse,
      )}`,
    );
  }

  logStage(`wait-${scenarioStage}-backend-action-respond`);
  let backendActionRespond;
  try {
    backendActionRespond = await waitForBackendLedgerEntry(
      runtimeEnv.backendLedgerPath,
      (entry) =>
        entry.kind === "approvalRequestResumeActionRespond" &&
        entry.requestId === APPROVAL_REQUEST_RESUME_REQUEST_ID &&
        entry.decision === decision,
      options,
    );
  } catch (error) {
    const lifecycle = await readApprovalServerRequestLifecycleSnapshot(page, {
      wireDecision: decision,
      threadId: options.threadId,
      turnId: backendTurnStart.entry.turnId,
    });
    throw new Error(
      `${error instanceof Error ? error.message : String(error)}; ` +
        `serverRequestLifecycle=${JSON.stringify(lifecycle)}`,
      { cause: error },
    );
  }

  if (decision === "cancel") {
    logStage(`wait-gui-${scenarioStage}-canceled`);
    const guiCanceled = sanitizeJson(
      await waitForGuiChatCanceled(page, options, {
        prompt: APPROVAL_REQUEST_RESUME_PROMPT,
      }),
    );

    logStage(`wait-read-model-${scenarioStage}-canceled`);
    const canceledReadModel = await waitForSessionReadCanceled(
      page,
      options,
      appServerRequests,
      {
        prompt: APPROVAL_REQUEST_RESUME_PROMPT,
      },
    );

    logStage(`wait-${scenarioStage}-server-request-lifecycle`);
    const serverRequestLifecycle = await waitForApprovalServerRequestLifecycle(
      page,
      options,
      {
        wireDecision: "cancel",
        threadId: options.threadId,
        turnId: backendTurnStart.entry.turnId,
      },
    );

    return sanitizeJson({
      approvalRequestDecisionExpectedDecision: decision,
      approvalRequestDecisionInputSend: inputSend,
      approvalRequestDecisionBackendTurnStart: {
        sessionId: backendTurnStart.entry.sessionId ?? null,
        threadId: backendTurnStart.entry.threadId ?? null,
        turnId: backendTurnStart.entry.turnId ?? null,
        inputText: backendTurnStart.entry.inputText ?? null,
      },
      approvalRequestDecisionPendingGui: pendingGui,
      approvalRequestDecisionPendingReadModel: pendingReadModel,
      approvalRequestDecisionClick: decisionClick,
      approvalRequestDecisionServerRequestResponse: serverRequestResponse,
      approvalRequestDecisionBackendActionRespond:
        summarizeBackendActionRespond(backendActionRespond.entry),
      approvalRequestDecisionServerRequestLifecycle: serverRequestLifecycle,
      guiApprovalRequestCancelCompleted: {
        ...guiCanceled,
        expectedDoneText: APPROVAL_REQUEST_CANCEL_DONE_TEXT,
      },
      readModelApprovalRequestCancelCanceled:
        summarizeApprovalDecisionReadModel(canceledReadModel, decision),
    });
  }

  logStage(`wait-gui-${scenarioStage}-completed`);
  const guiCompleted = sanitizeJson(
    await waitForGuiChatCompleted(page, options, {
      prompt: APPROVAL_REQUEST_RESUME_PROMPT,
      doneText: APPROVAL_REQUEST_DECLINE_DONE_TEXT,
      summaryText: APPROVAL_REQUEST_DECLINE_RESULT_TEXT,
    }),
  );

  logStage(`wait-read-model-${scenarioStage}-completed`);
  const completedReadModel = await waitForSessionReadCompleted(
    page,
    options,
    appServerRequests,
    {
      prompt: APPROVAL_REQUEST_RESUME_PROMPT,
      doneText: APPROVAL_REQUEST_DECLINE_DONE_TEXT,
      summaryText: APPROVAL_REQUEST_DECLINE_RESULT_TEXT,
    },
  );

  logStage(`wait-${scenarioStage}-server-request-lifecycle`);
  const serverRequestLifecycle = await waitForApprovalServerRequestLifecycle(
    page,
    options,
    {
      wireDecision: "decline",
      threadId: options.threadId,
      turnId: backendTurnStart.entry.turnId,
    },
  );

  return sanitizeJson({
    approvalRequestDecisionExpectedDecision: decision,
    approvalRequestDecisionInputSend: inputSend,
    approvalRequestDecisionBackendTurnStart: {
      sessionId: backendTurnStart.entry.sessionId ?? null,
      threadId: backendTurnStart.entry.threadId ?? null,
      turnId: backendTurnStart.entry.turnId ?? null,
      inputText: backendTurnStart.entry.inputText ?? null,
    },
    approvalRequestDecisionPendingGui: pendingGui,
    approvalRequestDecisionPendingReadModel: pendingReadModel,
    approvalRequestDecisionClick: decisionClick,
    approvalRequestDecisionServerRequestResponse: serverRequestResponse,
    approvalRequestDecisionBackendActionRespond: summarizeBackendActionRespond(
      backendActionRespond.entry,
    ),
    approvalRequestDecisionServerRequestLifecycle: serverRequestLifecycle,
    guiApprovalRequestDeclineCompleted: guiCompleted,
    readModelApprovalRequestDeclineCompleted:
      summarizeApprovalDecisionReadModel(completedReadModel, decision),
  });
}

export async function runApprovalRequestResumeScenario({
  page,
  options,
  appServerRequests,
  runtimeEnv,
  logStage,
}) {
  logStage("send-approval-request-resume-prompt-from-gui");
  const inputSend = sanitizeJson(
    await sendPromptFromGui(page, options, APPROVAL_REQUEST_RESUME_PROMPT),
  );

  logStage("wait-approval-request-resume-backend-turn-start");
  const backendTurnStart = await waitForBackendLedgerTurnStart(
    runtimeEnv.backendLedgerPath,
    APPROVAL_REQUEST_RESUME_PROMPT,
    options,
  );

  logStage("wait-gui-approval-request-resume-pending");
  const pendingGui = await waitForGuiApprovalPending(page, options);

  logStage("wait-read-model-approval-request-resume-pending");
  const pendingReadModel = await waitForApprovalPendingReadModel(
    page,
    options,
    appServerRequests,
  );

  logStage("click-approval-request-resume-approve");
  const approveClick = await clickApprovalApproveButton(page, options);

  logStage("wait-approval-request-resume-server-request-response");
  const serverRequestResponse = await waitForApprovalServerRequestResponse(
    page,
    options,
    {
      wireDecision: "acceptForSession",
      threadId: options.threadId,
      turnId: backendTurnStart.entry.turnId,
    },
  );
  if (serverRequestResponse.responseMatchesRequest !== true) {
    throw new Error(
      `审批 reverse server request 未记录匹配响应: ${JSON.stringify(
        serverRequestResponse,
      )}`,
    );
  }

  logStage("wait-approval-request-resume-backend-action-respond");
  let backendActionRespond;
  try {
    backendActionRespond = await waitForBackendLedgerEntry(
      runtimeEnv.backendLedgerPath,
      (entry) =>
        entry.kind === "approvalRequestResumeActionRespond" &&
        entry.requestId === APPROVAL_REQUEST_RESUME_REQUEST_ID,
      options,
    );
  } catch (error) {
    const lifecycle = await readApprovalServerRequestLifecycleSnapshot(page, {
      wireDecision: "acceptForSession",
      threadId: options.threadId,
      turnId: backendTurnStart.entry.turnId,
    });
    throw new Error(
      `${error instanceof Error ? error.message : String(error)}; ` +
        `serverRequestLifecycle=${JSON.stringify(lifecycle)}`,
      { cause: error },
    );
  }

  logStage("wait-gui-approval-request-resume-completed");
  const guiCompleted = sanitizeJson(
    await waitForGuiChatCompleted(page, options, {
      prompt: APPROVAL_REQUEST_RESUME_PROMPT,
      doneText: APPROVAL_REQUEST_RESUME_DONE_TEXT,
      summaryText: APPROVAL_REQUEST_RESUME_RESULT_TEXT,
    }),
  );

  logStage("wait-read-model-approval-request-resume-completed");
  const completedReadModel = await waitForSessionReadCompleted(
    page,
    options,
    appServerRequests,
    {
      prompt: APPROVAL_REQUEST_RESUME_PROMPT,
      doneText: APPROVAL_REQUEST_RESUME_DONE_TEXT,
      summaryText: APPROVAL_REQUEST_RESUME_RESULT_TEXT,
    },
  );

  logStage("wait-approval-request-resume-server-request-lifecycle");
  const serverRequestLifecycle = await waitForApprovalServerRequestLifecycle(
    page,
    options,
    {
      wireDecision: "acceptForSession",
      threadId: options.threadId,
      turnId: backendTurnStart.entry.turnId,
    },
  );

  return sanitizeJson({
    approvalRequestResumeInputSend: inputSend,
    approvalRequestResumeBackendTurnStart: {
      sessionId: backendTurnStart.entry.sessionId ?? null,
      threadId: backendTurnStart.entry.threadId ?? null,
      turnId: backendTurnStart.entry.turnId ?? null,
      inputText: backendTurnStart.entry.inputText ?? null,
    },
    approvalRequestResumePendingGui: pendingGui,
    approvalRequestResumePendingReadModel: pendingReadModel,
    approvalRequestResumeApproveClick: approveClick,
    approvalRequestResumeServerRequestResponse: serverRequestResponse,
    approvalRequestResumeBackendActionRespond: summarizeBackendActionRespond(
      backendActionRespond.entry,
    ),
    approvalRequestResumeServerRequestLifecycle: serverRequestLifecycle,
    guiApprovalRequestResumeCompleted: guiCompleted,
    readModelApprovalRequestResumeCompleted:
      summarizeApprovalCompletedReadModel(completedReadModel),
  });
}
