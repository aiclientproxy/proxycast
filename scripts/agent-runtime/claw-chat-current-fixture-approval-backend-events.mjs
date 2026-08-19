import {
  APPROVAL_REQUEST_CANCEL_DONE_TEXT,
  APPROVAL_REQUEST_DECLINE_DONE_TEXT,
  APPROVAL_REQUEST_DECLINE_RESULT_TEXT,
  APPROVAL_REQUEST_RESUME_APPROVAL_PROMPT,
  APPROVAL_REQUEST_RESUME_COMMAND,
  APPROVAL_REQUEST_RESUME_DONE_TEXT,
  APPROVAL_REQUEST_RESUME_REQUEST_ID,
  APPROVAL_REQUEST_RESUME_RESULT_TEXT,
  APPROVAL_REQUEST_RESUME_TOOL_CALL_ID,
  APPROVAL_REQUEST_RESUME_TOOL_NAME,
  FIXTURE_MODEL,
  FIXTURE_PROVIDER,
} from "./claw-chat-current-fixture-constants.mjs";

function js(value) {
  return JSON.stringify(value);
}

export function renderApprovalRequestResumeActionRespondScript() {
  return `
if (input.kind === "actionRespond") {
  const requestId =
    input.request?.requestId ||
    input.request?.request_id ||
    input.request?.actionId ||
    input.request?.action_id;
  const actionType =
    input.request?.actionType ||
    input.request?.action_type ||
    "tool_confirmation";
  const actionScope = input.request?.actionScope || input.request?.action_scope || {};
  const rawApprovalDecision = input.request?.decision;
  const rawApprovalDecisionScope =
    input.request?.decisionScope || input.request?.decision_scope;
  const approvalDecision =
    rawApprovalDecision ||
    (rawApprovalDecisionScope === "session"
      ? "allow_for_session"
      : "decline");
  const approvalDecisionScope =
    rawApprovalDecisionScope ||
    (approvalDecision === "allow_for_session" ? "session" : "once");
  const actionScopeSessionId = actionScope.sessionId || actionScope.session_id;
  const actionScopeThreadId = actionScope.threadId || actionScope.thread_id;
  const actionScopeTurnId = actionScope.turnId || actionScope.turn_id;
  const turnId = currentTurnId();
  const threadId = currentThreadId();
  const isApprovalRequestResumeAction =
    requestId === ${js(APPROVAL_REQUEST_RESUME_REQUEST_ID)} &&
    actionType === "tool_confirmation";
  if (!isApprovalRequestResumeAction) {
    appendLedgerEntry({
      kind: "approvalRequestResumeActionRespondIgnored",
      sessionId: input.request?.session?.sessionId,
      threadId,
      turnId,
      requestId,
      actionType,
      decision: input.request?.decision,
      decisionScope: input.request?.decisionScope || input.request?.decision_scope,
      requestKeys: Object.keys(input.request || {}).sort(),
      actionScope: {
        sessionId: actionScopeSessionId,
        session_id: actionScopeSessionId,
        threadId: actionScopeThreadId,
        thread_id: actionScopeThreadId,
        turnId: actionScopeTurnId,
        turn_id: actionScopeTurnId
      }
    });
    emitEvents([]);
    process.exit(0);
  }
  const approvalAllowed =
    approvalDecision === "allow_once" ||
    approvalDecision === "allow_for_session";
  const approvalCanceled = approvalDecision === "cancel";
  const resolvedResponse =
    input.request?.response ||
    (approvalCanceled ? "canceled" : approvalAllowed ? "approved" : "declined");
  appendLedgerEntry({
    kind: "approvalRequestResumeActionRespond",
    sessionId: input.request?.session?.sessionId,
    threadId,
    turnId,
    requestId,
    actionType,
    decision: approvalDecision,
    decisionScope: approvalDecisionScope,
    confirmed: input.request?.confirmed,
    response: input.request?.response,
    actionScope: {
      sessionId: actionScopeSessionId,
      session_id: actionScopeSessionId,
      threadId: actionScopeThreadId,
      thread_id: actionScopeThreadId,
      turnId: actionScopeTurnId,
      turn_id: actionScopeTurnId
    }
  });
  const actionResolvedEvent = {
      type: "action.resolved",
      payload: {
        requestId: ${js(APPROVAL_REQUEST_RESUME_REQUEST_ID)},
        request_id: ${js(APPROVAL_REQUEST_RESUME_REQUEST_ID)},
        actionId: ${js(APPROVAL_REQUEST_RESUME_REQUEST_ID)},
        action_id: ${js(APPROVAL_REQUEST_RESUME_REQUEST_ID)},
        actionType: "tool_confirmation",
        action_type: "tool_confirmation",
        actionKind: "permission_preflight",
        action_kind: "permission_preflight",
        confirmed: approvalAllowed,
        decision: approvalDecision,
        decisionScope: approvalDecisionScope,
        decision_scope: approvalDecisionScope,
        response: resolvedResponse,
        toolCallId: ${js(APPROVAL_REQUEST_RESUME_TOOL_CALL_ID)},
        tool_call_id: ${js(APPROVAL_REQUEST_RESUME_TOOL_CALL_ID)},
        toolName: ${js(APPROVAL_REQUEST_RESUME_TOOL_NAME)},
        tool_name: ${js(APPROVAL_REQUEST_RESUME_TOOL_NAME)},
        approvalPolicy: "on-request",
        approval_policy: "on-request",
        sandboxPolicy: "workspace-write",
        sandbox_policy: "workspace-write",
        scope: {
          sessionId: actionScopeSessionId || input.request?.session?.sessionId,
          session_id: actionScopeSessionId || input.request?.session?.sessionId,
          threadId: actionScopeThreadId || threadId,
          thread_id: actionScopeThreadId || threadId,
          turnId: actionScopeTurnId || turnId,
          turn_id: actionScopeTurnId || turnId
        }
      }
    };
  const completionEvents = approvalCanceled
    ? [
        {
          type: "item.completed",
          payload: buildCanonicalToolItem({
            sessionId: input.request?.session?.sessionId,
            threadId,
            turnId,
            itemId: ${js(APPROVAL_REQUEST_RESUME_TOOL_CALL_ID)},
            ordinal: 2,
            callId: ${js(APPROVAL_REQUEST_RESUME_TOOL_CALL_ID)},
            name: ${js(APPROVAL_REQUEST_RESUME_TOOL_NAME)},
            status: "failed",
            output: { error: ${js(APPROVAL_REQUEST_CANCEL_DONE_TEXT)} }
          })
        },
        {
          type: "turn.canceled",
          payload: {
            status: "canceled",
            reason: "approval_request_cancelled",
            text: ${js(APPROVAL_REQUEST_CANCEL_DONE_TEXT)}
          }
        }
      ]
    : approvalAllowed
      ? [
          {
            type: "item.completed",
            payload: buildCanonicalToolItem({
              sessionId: input.request?.session?.sessionId,
              threadId,
              turnId,
              itemId: ${js(APPROVAL_REQUEST_RESUME_TOOL_CALL_ID)},
              ordinal: 2,
              callId: ${js(APPROVAL_REQUEST_RESUME_TOOL_CALL_ID)},
              name: ${js(APPROVAL_REQUEST_RESUME_TOOL_NAME)},
              status: "completed",
              output: { text: ${js(APPROVAL_REQUEST_RESUME_RESULT_TEXT)} }
            })
          },
          {
            type: "provider.first_text_delta.received",
            payload: {
              stage: "first_text_delta_received",
              provider: ${js(FIXTURE_PROVIDER)},
              model: ${js(FIXTURE_MODEL)},
              attempt: 1,
              elapsed_ms: 90,
              elapsedMs: 90,
              status: "running",
              text_chars: ${js(APPROVAL_REQUEST_RESUME_RESULT_TEXT)}.length,
              textChars: ${js(APPROVAL_REQUEST_RESUME_RESULT_TEXT)}.length
            }
          },
          {
            type: "message.delta",
            payload: {
              text: ${js(`${APPROVAL_REQUEST_RESUME_RESULT_TEXT}\n${APPROVAL_REQUEST_RESUME_DONE_TEXT}\n`)},
              item_id: "agent-message-final-" + (turnId || "turn"),
              itemId: "agent-message-final-" + (turnId || "turn"),
              phase: "final_answer",
              thread_id: threadId,
              threadId,
              turn_id: turnId,
              turnId
            }
          },
          {
            type: "turn.completed",
            payload: {
              status: "completed",
              text: ${js(APPROVAL_REQUEST_RESUME_DONE_TEXT)}
            }
          }
        ]
      : [
          {
            type: "item.completed",
            payload: buildCanonicalToolItem({
              sessionId: input.request?.session?.sessionId,
              threadId,
              turnId,
              itemId: ${js(APPROVAL_REQUEST_RESUME_TOOL_CALL_ID)},
              ordinal: 2,
              callId: ${js(APPROVAL_REQUEST_RESUME_TOOL_CALL_ID)},
              name: ${js(APPROVAL_REQUEST_RESUME_TOOL_NAME)},
              status: "failed",
              output: { error: ${js(APPROVAL_REQUEST_DECLINE_RESULT_TEXT)} }
            })
          },
          {
            type: "provider.first_text_delta.received",
            payload: {
              stage: "first_text_delta_received",
              provider: ${js(FIXTURE_PROVIDER)},
              model: ${js(FIXTURE_MODEL)},
              attempt: 1,
              elapsed_ms: 90,
              elapsedMs: 90,
              status: "running",
              text_chars: ${js(APPROVAL_REQUEST_DECLINE_RESULT_TEXT)}.length,
              textChars: ${js(APPROVAL_REQUEST_DECLINE_RESULT_TEXT)}.length
            }
          },
          {
            type: "message.delta",
            payload: {
              text: ${js(`${APPROVAL_REQUEST_DECLINE_RESULT_TEXT}\n${APPROVAL_REQUEST_DECLINE_DONE_TEXT}\n`)},
              item_id: "agent-message-final-" + (turnId || "turn"),
              itemId: "agent-message-final-" + (turnId || "turn"),
              phase: "final_answer",
              thread_id: threadId,
              threadId,
              turn_id: turnId,
              turnId
            }
          },
          {
            type: "turn.completed",
            payload: {
              status: "completed",
              text: ${js(APPROVAL_REQUEST_DECLINE_DONE_TEXT)}
            }
          }
        ];
  emitEvents([actionResolvedEvent, ...completionEvents]);
  process.exit(0);
}
`;
}

export function renderApprovalRequestResumeTurnStartScript() {
  return `
  if (isApprovalRequestResumePrompt) {
    const turnId = currentTurnId();
    const threadId = currentThreadId();
    emitEvents([
      {
        type: "provider.request.started",
        payload: providerTracePayload("request_started", 0, "running")
      },
      {
        type: "provider.first_event.received",
        payload: providerTracePayload("first_event_received", 40, "running")
      },
      {
        type: "item.started",
        payload: buildCanonicalToolItem({
          sessionId: input.request?.session?.sessionId,
          threadId,
          turnId,
          itemId: ${js(APPROVAL_REQUEST_RESUME_TOOL_CALL_ID)},
          ordinal: 2,
          callId: ${js(APPROVAL_REQUEST_RESUME_TOOL_CALL_ID)},
          name: ${js(APPROVAL_REQUEST_RESUME_TOOL_NAME)},
          arguments: {
            command: ${js(APPROVAL_REQUEST_RESUME_COMMAND)}
          },
          status: "inProgress",
          metadata: { commandExecutionSource: "agent" }
        })
      },
      {
        type: "action.required",
        payload: {
          requestId: ${js(APPROVAL_REQUEST_RESUME_REQUEST_ID)},
          request_id: ${js(APPROVAL_REQUEST_RESUME_REQUEST_ID)},
          actionId: ${js(APPROVAL_REQUEST_RESUME_REQUEST_ID)},
          action_id: ${js(APPROVAL_REQUEST_RESUME_REQUEST_ID)},
          actionType: "tool_confirmation",
          action_type: "tool_confirmation",
          actionKind: "permission_preflight",
          action_kind: "permission_preflight",
          availableDecisions: ["allow_once", "allow_for_session", "decline", "cancel"],
          available_decisions: ["allow_once", "allow_for_session", "decline", "cancel"],
          toolCallId: ${js(APPROVAL_REQUEST_RESUME_TOOL_CALL_ID)},
          tool_call_id: ${js(APPROVAL_REQUEST_RESUME_TOOL_CALL_ID)},
          toolName: ${js(APPROVAL_REQUEST_RESUME_TOOL_NAME)},
          tool_name: ${js(APPROVAL_REQUEST_RESUME_TOOL_NAME)},
          prompt: ${js(APPROVAL_REQUEST_RESUME_APPROVAL_PROMPT)},
          message: ${js(APPROVAL_REQUEST_RESUME_APPROVAL_PROMPT)},
          approvalPolicy: "on-request",
          approval_policy: "on-request",
          sandboxPolicy: "workspace-write",
          sandbox_policy: "workspace-write",
          runtime_contract: {
            session_cache_supported: true
          },
          arguments: {
            command: ${js(APPROVAL_REQUEST_RESUME_COMMAND)}
          },
          data: {
            prompt: ${js(APPROVAL_REQUEST_RESUME_APPROVAL_PROMPT)},
            availableDecisions: ["allow_once", "allow_for_session", "decline", "cancel"],
            available_decisions: ["allow_once", "allow_for_session", "decline", "cancel"],
            toolName: ${js(APPROVAL_REQUEST_RESUME_TOOL_NAME)},
            tool_name: ${js(APPROVAL_REQUEST_RESUME_TOOL_NAME)},
            approvalPolicy: "on-request",
            approval_policy: "on-request",
            sandboxPolicy: "workspace-write",
            sandbox_policy: "workspace-write",
            runtime_contract: {
              session_cache_supported: true
            },
            arguments: {
              command: ${js(APPROVAL_REQUEST_RESUME_COMMAND)}
            }
          },
          scope: {
            sessionId: input.request?.session?.sessionId,
            session_id: input.request?.session?.sessionId,
            threadId,
            thread_id: threadId,
            turnId,
            turn_id: turnId,
          }
        }
      }
    ]);
    process.exit(0);
  }
`;
}
