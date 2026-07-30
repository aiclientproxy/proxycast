import type { ActionRequired, ConfirmResponse } from "../agentActionTypes";
import {
  actionFromPendingInteraction,
  findPendingActionInteraction,
  getDefaultPendingInteractionController,
} from "./pendingInteractionController";
import type {
  AgentRuntimeRespondActionRequest,
  AgentRuntimeReplayedActionRequiredView,
} from "./requestTypes";

export function findPendingTypedServerRequestAction(
  sessionId: string,
  requestId: string,
): ActionRequired | null {
  const pendingActions = getDefaultPendingInteractionController()
    .getSnapshot()
    .map(actionFromPendingInteraction)
    .filter((action): action is ActionRequired => Boolean(action));
  return findPendingTypedAction(pendingActions, sessionId, requestId);
}

/**
 * Settle a typed approval/AskUser request without re-emitting the retired
 * agentSession/action/respond command. Callers must fail closed when the
 * matching typed request is no longer pending.
 */
export function respondPendingTypedServerRequest(
  request: AgentRuntimeRespondActionRequest,
): boolean {
  const action = findPendingTypedServerRequestAction(
    request.session_id,
    request.request_id,
  );
  if (!action || action.actionType !== request.action_type) {
    return false;
  }
  if (!actionScopeMatches(action, request)) {
    return false;
  }

  const response: ConfirmResponse = {
    requestId: request.request_id,
    actionType: request.action_type,
    confirmed: request.confirmed,
    decision: request.decision,
    response: request.response,
    userData: request.user_data,
  };
  const controller = getDefaultPendingInteractionController();
  const interaction = findPendingActionInteraction(
    controller.getSnapshot(),
    request.action_type,
    request.request_id,
  );
  if (!interaction) {
    return false;
  }
  if (interaction.kind === "approval") {
    return controller.respond({
      interactionId: interaction.id,
      kind: "approval",
      response,
    }).accepted;
  }
  return controller.respond({
    confirmed: response.confirmed !== false,
    interactionId: interaction.id,
    kind: "request_user_input",
    response: response.response,
    userData: response.userData,
  }).accepted;
}

export function findPendingTypedAction(
  pendingActions: readonly ActionRequired[],
  sessionId: string,
  requestId: string,
): ActionRequired | null {
  const normalizedSessionId = sessionId.trim();
  const normalizedRequestId = requestId.trim();
  if (!normalizedSessionId || !normalizedRequestId) {
    return null;
  }

  return (
    pendingActions.find(
      (action) =>
        action.requestId === normalizedRequestId &&
        actionBelongsToSession(action, normalizedSessionId),
    ) ?? null
  );
}

export function replayedActionViewFromPendingAction(
  action: ActionRequired,
): AgentRuntimeReplayedActionRequiredView {
  return omitUndefined({
    type: "action_required" as const,
    request_id: action.requestId,
    action_type: action.actionType,
    tool_name: action.toolName,
    arguments: action.arguments,
    prompt: action.prompt,
    questions: action.questions,
    requested_schema: action.requestedSchema,
    available_decisions: action.availableDecisions,
    scope: action.scope
      ? omitUndefined({
          session_id: action.scope.sessionId,
          thread_id: action.scope.threadId,
          turn_id: action.scope.turnId,
        })
      : undefined,
  });
}

function actionBelongsToSession(
  action: ActionRequired,
  sessionId: string,
): boolean {
  const scope = action.scope;
  if (!scope) {
    return false;
  }
  return scope.sessionId === sessionId || scope.threadId === sessionId;
}

function actionScopeMatches(
  action: ActionRequired,
  request: AgentRuntimeRespondActionRequest,
): boolean {
  const requestedScope = request.action_scope;
  if (!requestedScope) {
    return true;
  }
  const actionScope = action.scope;
  if (!actionScope) {
    return false;
  }
  return (
    (requestedScope.session_id === undefined ||
      requestedScope.session_id === actionScope.sessionId) &&
    (requestedScope.thread_id === undefined ||
      requestedScope.thread_id === actionScope.threadId) &&
    (requestedScope.turn_id === undefined ||
      requestedScope.turn_id === actionScope.turnId)
  );
}

function omitUndefined<T extends Record<string, unknown>>(value: T): T {
  return Object.fromEntries(
    Object.entries(value).filter(([, entry]) => entry !== undefined),
  ) as T;
}
