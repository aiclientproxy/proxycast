import {
  AgentRuntimeEventPipeline,
  type AgentRuntimeSequenceVerifierMode,
} from "@limecloud/agent-runtime-client/sessionGateway";
import {
  APP_SERVER_METHOD_AGENT_SESSION_EVENT,
  type AppServerAgentEvent,
  type AppServerJsonRpcNotification,
} from "@/lib/api/appServer";
import { providerTraceStageFromEventType } from "./appServerEventPayloadUtils";
import {
  isAppServerV2NotificationMethod,
  readAppServerV2NotificationRoute,
  resetAppServerV2NotificationProjectionState,
} from "./appServerV2Notification";
import { readAppServerNotificationDriftRoute } from "./appServerNotificationDrift";

type GateResult =
  | { kind: "accepted"; notifications: AppServerJsonRpcNotification[] }
  | { kind: "ignored" }
  | { kind: "blocked"; codes: string[] };

const gates = new Map<string, AgentRuntimeEventPipeline>();
const directItemStates = new Map<
  string,
  Map<string, { completed: boolean; type: string }>
>();
const DIRECT_ITEM_STREAM_GUARDS = new Map<
  string,
  { codePrefix: string; expectedItemType: string }
>([
  [
    "item/commandExecution/outputDelta",
    {
      codePrefix: "command_output_delta",
      expectedItemType: "commandExecution",
    },
  ],
  [
    "item/fileChange/patchUpdated",
    {
      codePrefix: "file_change_patch_updated",
      expectedItemType: "fileChange",
    },
  ],
  [
    "item/mcpToolCall/progress",
    { codePrefix: "mcp_tool_call_progress", expectedItemType: "mcpToolCall" },
  ],
  ["item/plan/delta", { codePrefix: "plan_delta", expectedItemType: "plan" }],
]);
type SequenceGateNotification = Parameters<
  AgentRuntimeEventPipeline["processSync"]
>[0];

export function resetAgentRuntimeEventSequenceGatesForTests(): void {
  gates.clear();
  directItemStates.clear();
  resetAppServerV2NotificationProjectionState();
}

export function agentRuntimeSequenceGateAllowsNotification(
  eventName: string,
  notification: AppServerJsonRpcNotification,
  mode: AgentRuntimeSequenceVerifierMode = "fail-closed",
  gateScope = "shared",
): boolean {
  const result = processNotification(eventName, notification, mode, gateScope);
  reportBlockedSequence(eventName, result);
  return result.kind !== "blocked";
}

export function agentRuntimeSequenceGateAllowsPayload(
  eventName: string,
  payload: unknown,
  mode: AgentRuntimeSequenceVerifierMode = "fail-closed",
  gateScope = "shared",
): boolean {
  const notification = notificationFromPublishedPayload(payload);
  if (!notification) {
    return true;
  }
  const result = processNotification(eventName, notification, mode, gateScope);
  reportBlockedSequence(eventName, result);
  return result.kind !== "blocked";
}

export function projectAgentRuntimeSequenceGateNotifications(
  eventName: string,
  notification: AppServerJsonRpcNotification,
  mode: AgentRuntimeSequenceVerifierMode = "fail-closed",
  gateScope = "shared",
): AppServerJsonRpcNotification[] {
  const result = processNotification(eventName, notification, mode, gateScope);
  reportBlockedSequence(eventName, result);
  return result.kind === "accepted" ? result.notifications : [];
}

export function projectAgentRuntimeSequenceGatePayloads(
  eventName: string,
  payload: unknown,
  mode: AgentRuntimeSequenceVerifierMode = "fail-closed",
  gateScope = "shared",
): unknown[] {
  const notification = notificationFromPublishedPayload(payload);
  if (!notification) {
    return [payload];
  }
  return projectAgentRuntimeSequenceGateNotifications(
    eventName,
    notification,
    mode,
    gateScope,
  ).map((notification) => {
    const event = readAppServerAgentEvent(notification.params);
    return event?.payload ?? payload;
  });
}

function processNotification(
  eventName: string,
  notification: AppServerJsonRpcNotification,
  mode: AgentRuntimeSequenceVerifierMode,
  gateScope: string,
): GateResult {
  const directRoute = readAppServerV2NotificationRoute(notification);
  if (directRoute) {
    if (
      notification.method === "warning" ||
      notification.method === "guardianWarning" ||
      notification.method === "autoApprovalReview/strictReviewRequired" ||
      notification.method === "error" ||
      notification.method === "item/autoApprovalReview/started" ||
      notification.method === "item/autoApprovalReview/completed" ||
      notification.method === "turn/diff/updated" ||
      notification.method === "turn/moderationMetadata" ||
      notification.method === "turn/plan/updated"
    ) {
      return { kind: "accepted", notifications: [notification] };
    }
    if (isDirectStreamingNotification(notification.method)) {
      const guard = DIRECT_ITEM_STREAM_GUARDS.get(notification.method);
      if (guard) {
        return processDirectItemStreamingNotification(
          eventName,
          notification,
          directRoute.threadId,
          directRoute.itemId,
          guard,
          mode,
          gateScope,
        );
      }
      return { kind: "accepted", notifications: [notification] };
    }
    return processDirectLifecycleNotification(
      eventName,
      notification,
      directRoute.threadId,
      mode,
      gateScope,
    );
  }
  const event = readAppServerAgentEvent(notification.params);
  if (!event) {
    if (
      !isAppServerV2NotificationMethod(notification.method) &&
      readAppServerNotificationDriftRoute(notification)
    ) {
      return { kind: "accepted", notifications: [notification] };
    }
    return { kind: "ignored" };
  }
  if (isCurrentNonThreadSideChannelEvent(event.type)) {
    return { kind: "accepted", notifications: [notification] };
  }
  return { kind: "ignored" };
}

function isDirectStreamingNotification(method: string): boolean {
  return (
    method === "item/agentMessage/delta" ||
    method === "item/commandExecution/outputDelta" ||
    method === "item/fileChange/patchUpdated" ||
    method === "item/mcpToolCall/progress" ||
    method === "item/plan/delta" ||
    method === "item/reasoning/summaryTextDelta" ||
    method === "item/reasoning/summaryPartAdded" ||
    method === "item/reasoning/textDelta" ||
    method === "thread/tokenUsage/updated"
  );
}

function processDirectLifecycleNotification(
  eventName: string,
  notification: AppServerJsonRpcNotification,
  threadId: string,
  mode: AgentRuntimeSequenceVerifierMode,
  gateScope: string,
): GateResult {
  const gate = gateFor(gateScope, eventName, threadId, mode);
  let result: ReturnType<AgentRuntimeEventPipeline["processSync"]>;
  try {
    result = gate.processSync(
      notification as unknown as SequenceGateNotification,
    );
  } catch {
    return { kind: "blocked", codes: ["invalid_lifecycle_notification"] };
  }
  if (result.accepted) {
    if (
      notification.method === "item/started" ||
      notification.method === "item/completed"
    ) {
      observeDirectItemLifecycle(
        directItemStateFor(gateScope, eventName, threadId, mode),
        notification,
      );
    } else if (notification.method === "turn/completed") {
      directItemStates.delete(gateKey(gateScope, eventName, threadId, mode));
    }
    return {
      kind: "accepted",
      notifications:
        result.notifications as unknown as AppServerJsonRpcNotification[],
    };
  }
  if (result.reason === "dropped") {
    return { kind: "ignored" };
  }
  return {
    kind: "blocked",
    codes: gate.getViolations().map((violation) => violation.code),
  };
}

function processDirectItemStreamingNotification(
  eventName: string,
  notification: AppServerJsonRpcNotification,
  threadId: string,
  itemId: string | undefined,
  guard: { codePrefix: string; expectedItemType: string },
  mode: AgentRuntimeSequenceVerifierMode,
  gateScope: string,
): GateResult {
  if (!itemId) {
    return {
      kind: "blocked",
      codes: [`${guard.codePrefix}_missing_item`],
    };
  }
  const state = directItemStateFor(gateScope, eventName, threadId, mode).get(
    itemId,
  );
  if (!state) {
    return {
      kind: "blocked",
      codes: [`${guard.codePrefix}_without_started`],
    };
  }
  if (state.type !== guard.expectedItemType) {
    return {
      kind: "blocked",
      codes: [`${guard.codePrefix}_item_type_mismatch`],
    };
  }
  if (state.completed) {
    return {
      kind: "blocked",
      codes: [`${guard.codePrefix}_after_completed`],
    };
  }
  return { kind: "accepted", notifications: [notification] };
}

function observeDirectItemLifecycle(
  states: Map<string, { completed: boolean; type: string }>,
  notification: AppServerJsonRpcNotification,
): void {
  if (
    notification.method !== "item/started" &&
    notification.method !== "item/completed"
  ) {
    return;
  }
  const params = normalizeRecord(notification.params);
  const item = normalizeRecord(params?.item);
  const itemId = item ? readString(item, "id") : undefined;
  const itemType = item ? readString(item, "type") : undefined;
  if (!itemId || !itemType) {
    return;
  }
  states.set(itemId, {
    completed: notification.method === "item/completed",
    type: itemType,
  });
}

function directItemStateFor(
  gateScope: string,
  eventName: string,
  sessionId: string,
  mode: AgentRuntimeSequenceVerifierMode,
): Map<string, { completed: boolean; type: string }> {
  const key = gateKey(gateScope, eventName, sessionId, mode);
  let states = directItemStates.get(key);
  if (!states) {
    states = new Map();
    directItemStates.set(key, states);
  }
  return states;
}

function isCurrentNonThreadSideChannelEvent(eventType: string): boolean {
  return (
    providerTraceStageFromEventType(eventType) !== undefined ||
    eventType === "runtime.status" ||
    eventType === "image_task.presentation.generated" ||
    eventType === "image_task.created" ||
    eventType === "image_task.parameters.required" ||
    eventType === "image_task_parameters_required"
  );
}

function gateFor(
  gateScope: string,
  eventName: string,
  sessionId: string,
  mode: AgentRuntimeSequenceVerifierMode,
): AgentRuntimeEventPipeline {
  const key = gateKey(gateScope, eventName, sessionId, mode);
  let gate = gates.get(key);
  if (!gate) {
    gate = new AgentRuntimeEventPipeline({
      sequenceVerifierMode: mode,
    });
    gates.set(key, gate);
  }
  return gate;
}

function gateKey(
  gateScope: string,
  eventName: string,
  sessionId: string,
  mode: AgentRuntimeSequenceVerifierMode,
): string {
  return `${gateScope}\u0000${eventName}\u0000${sessionId}\u0000${mode}`;
}

function notificationFromPublishedPayload(
  payload: unknown,
): AppServerJsonRpcNotification | null {
  const record = normalizeRecord(payload);
  if (!record) {
    return null;
  }
  const eventId = readString(record, "event_id", "eventId");
  const sessionId = readString(record, "session_id", "sessionId");
  const timestamp = readString(record, "timestamp");
  const sequence = record.sequence;
  const type = appServerEventTypeFromPublishedPayload(record);
  if (
    !eventId ||
    !sessionId ||
    !timestamp ||
    typeof sequence !== "number" ||
    !type
  ) {
    return null;
  }
  const event: AppServerAgentEvent = {
    eventId,
    sequence,
    sessionId,
    threadId: readString(record, "thread_id", "threadId"),
    turnId: readString(record, "turn_id", "turnId"),
    type,
    timestamp,
    payload: {
      ...record,
      toolCallId: readString(
        record,
        "tool_id",
        "tool_call_id",
        "toolId",
        "toolCallId",
      ),
      actionId: readString(
        record,
        "request_id",
        "requestId",
        "action_id",
        "actionId",
      ),
    },
  };
  return {
    method: APP_SERVER_METHOD_AGENT_SESSION_EVENT,
    params: { event },
  };
}

function appServerEventTypeFromPublishedPayload(
  payload: Record<string, unknown>,
): string | undefined {
  const type = readString(payload, "runtime_event_type", "event_type");
  if (type) {
    return type;
  }
  switch (readString(payload, "type")) {
    case "text_delta":
      return "message.delta";
    case "text_delta_batch":
      return "message.delta_batch";
    case "thinking_delta":
      return "thinking.delta";
    case "reasoning_started":
      return "reasoning.started";
    case "reasoning_delta":
      return "reasoning.delta";
    case "reasoning_final":
      return "reasoning.final";
    case "reasoning_ended":
      return "reasoning.ended";
    case "plan_delta":
      return "plan.delta";
    case "plan_final":
      return "plan.final";
    case "model_effective":
      return "model.effective";
    case "tool_start":
    case "tool_started":
      return "tool.started";
    case "tool_end":
    case "tool_result":
      return "tool.result";
    case "tool_complete":
    case "tool_completed":
      return "tool.completed";
    case "item_started":
      return "item.started";
    case "item_updated":
      return "item.updated";
    case "item_completed":
      return "item.completed";
    case "action_required":
      return "action.required";
    case "action_resolved":
      return "action.resolved";
    case "artifact_snapshot":
      return "artifact.snapshot";
    case "turn_completed":
      return "turn.completed";
    case "turn_failed":
      return "turn.failed";
    case "turn_canceled":
      return "turn.canceled";
    default:
      return undefined;
  }
}

function readAppServerAgentEvent(params: unknown): AppServerAgentEvent | null {
  const record = normalizeRecord(params);
  const event = normalizeRecord(record?.event);
  if (!event) {
    return null;
  }
  const eventId = readString(event, "eventId", "event_id");
  const sessionId = readString(event, "sessionId", "session_id");
  const type = readString(event, "type");
  const timestamp = readString(event, "timestamp");
  const sequence = event.sequence;
  if (
    !eventId ||
    !sessionId ||
    !type ||
    !timestamp ||
    typeof sequence !== "number"
  ) {
    return null;
  }
  return {
    eventId,
    sequence,
    sessionId,
    threadId: readString(event, "threadId", "thread_id"),
    turnId: readString(event, "turnId", "turn_id"),
    type,
    timestamp,
    payload: event.payload,
  };
}

function reportBlockedSequence(eventName: string, result: GateResult): void {
  if (result.kind !== "blocked") {
    return;
  }
  console.warn(
    `[agentRuntime] blocked invalid runtime event sequence for ${eventName}: ${result.codes.join(", ")}`,
  );
}

function normalizeRecord(value: unknown): Record<string, unknown> | undefined {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function readString(
  record: Record<string, unknown>,
  ...keys: string[]
): string | undefined {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "string" && value.length > 0) {
      return value;
    }
  }
  return undefined;
}
