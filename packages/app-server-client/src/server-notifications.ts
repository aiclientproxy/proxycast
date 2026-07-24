import {
  isJsonRpcNotification,
  type JsonRpcMessage,
  type ServerNotification,
} from "./protocol.js";

export type RuntimeServerNotification = Extract<
  ServerNotification,
  {
    method:
      | "thread/started"
      | "turn/started"
      | "turn/completed"
      | "item/started"
      | "item/completed"
      | "item/agentMessage/delta"
      | "item/commandExecution/outputDelta"
      | "item/fileChange/patchUpdated"
      | "item/mcpToolCall/progress"
      | "item/plan/delta"
      | "item/reasoning/summaryTextDelta"
      | "item/reasoning/summaryPartAdded"
      | "item/reasoning/textDelta"
      | "thread/settings/updated";
  }
>;

export type ServerNotificationFor<
  Method extends RuntimeServerNotification["method"],
> = Extract<RuntimeServerNotification, { method: Method }>;

export function serverNotification(
  message: JsonRpcMessage,
): RuntimeServerNotification | undefined {
  if (!isJsonRpcNotification(message)) {
    return undefined;
  }

  switch (message.method) {
    case "thread/started":
      return hasEntityId(record(message.params)?.thread)
        ? (message as ServerNotificationFor<"thread/started">)
        : undefined;
    case "turn/started":
      return hasTurnNotification(message.params, ["inProgress"])
        ? (message as ServerNotificationFor<"turn/started">)
        : undefined;
    case "turn/completed":
      return hasTurnNotification(message.params, [
        "completed",
        "failed",
        "interrupted",
      ])
        ? (message as ServerNotificationFor<"turn/completed">)
        : undefined;
    case "item/started":
      return hasItemNotification(message.params, "startedAtMs")
        ? (message as ServerNotificationFor<"item/started">)
        : undefined;
    case "item/completed":
      return hasItemNotification(message.params, "completedAtMs")
        ? (message as ServerNotificationFor<"item/completed">)
        : undefined;
    case "item/agentMessage/delta":
      return hasItemTextDelta(message.params)
        ? (message as ServerNotificationFor<"item/agentMessage/delta">)
        : undefined;
    case "item/commandExecution/outputDelta":
      return hasItemTextDelta(message.params)
        ? (message as ServerNotificationFor<"item/commandExecution/outputDelta">)
        : undefined;
    case "item/fileChange/patchUpdated":
      return hasFileChangePatchUpdated(message.params)
        ? (message as ServerNotificationFor<"item/fileChange/patchUpdated">)
        : undefined;
    case "item/mcpToolCall/progress":
      return hasMcpToolCallProgress(message.params)
        ? (message as ServerNotificationFor<"item/mcpToolCall/progress">)
        : undefined;
    case "item/plan/delta":
      return hasItemTextDelta(message.params)
        ? (message as ServerNotificationFor<"item/plan/delta">)
        : undefined;
    case "item/reasoning/summaryTextDelta":
      return hasReasoningDelta(message.params, "summaryIndex")
        ? (message as ServerNotificationFor<"item/reasoning/summaryTextDelta">)
        : undefined;
    case "item/reasoning/summaryPartAdded":
      return hasReasoningIdentity(message.params, "summaryIndex")
        ? (message as ServerNotificationFor<"item/reasoning/summaryPartAdded">)
        : undefined;
    case "item/reasoning/textDelta":
      return hasReasoningDelta(message.params, "contentIndex")
        ? (message as ServerNotificationFor<"item/reasoning/textDelta">)
        : undefined;
    case "thread/settings/updated":
      return hasThreadSettings(message.params)
        ? (message as ServerNotificationFor<"thread/settings/updated">)
        : undefined;
    default:
      return undefined;
  }
}

export function isServerNotification(
  message: JsonRpcMessage,
): message is RuntimeServerNotification {
  return serverNotification(message) !== undefined;
}

export function isThreadStartedNotification(
  message: JsonRpcMessage,
): message is ServerNotificationFor<"thread/started"> {
  return serverNotification(message)?.method === "thread/started";
}

export function isTurnStartedNotification(
  message: JsonRpcMessage,
): message is ServerNotificationFor<"turn/started"> {
  return serverNotification(message)?.method === "turn/started";
}

export function isTurnCompletedNotification(
  message: JsonRpcMessage,
): message is ServerNotificationFor<"turn/completed"> {
  return serverNotification(message)?.method === "turn/completed";
}

export function isItemStartedNotification(
  message: JsonRpcMessage,
): message is ServerNotificationFor<"item/started"> {
  return serverNotification(message)?.method === "item/started";
}

export function isItemCompletedNotification(
  message: JsonRpcMessage,
): message is ServerNotificationFor<"item/completed"> {
  return serverNotification(message)?.method === "item/completed";
}

export function isAgentMessageDeltaNotification(
  message: JsonRpcMessage,
): message is ServerNotificationFor<"item/agentMessage/delta"> {
  return serverNotification(message)?.method === "item/agentMessage/delta";
}

export function isCommandExecutionOutputDeltaNotification(
  message: JsonRpcMessage,
): message is ServerNotificationFor<"item/commandExecution/outputDelta"> {
  return (
    serverNotification(message)?.method === "item/commandExecution/outputDelta"
  );
}

export function isFileChangePatchUpdatedNotification(
  message: JsonRpcMessage,
): message is ServerNotificationFor<"item/fileChange/patchUpdated"> {
  return serverNotification(message)?.method === "item/fileChange/patchUpdated";
}

export function isMcpToolCallProgressNotification(
  message: JsonRpcMessage,
): message is ServerNotificationFor<"item/mcpToolCall/progress"> {
  return serverNotification(message)?.method === "item/mcpToolCall/progress";
}

export function isPlanDeltaNotification(
  message: JsonRpcMessage,
): message is ServerNotificationFor<"item/plan/delta"> {
  return serverNotification(message)?.method === "item/plan/delta";
}

export function isReasoningSummaryTextDeltaNotification(
  message: JsonRpcMessage,
): message is ServerNotificationFor<"item/reasoning/summaryTextDelta"> {
  return (
    serverNotification(message)?.method === "item/reasoning/summaryTextDelta"
  );
}

export function isReasoningSummaryPartAddedNotification(
  message: JsonRpcMessage,
): message is ServerNotificationFor<"item/reasoning/summaryPartAdded"> {
  return (
    serverNotification(message)?.method === "item/reasoning/summaryPartAdded"
  );
}

export function isReasoningTextDeltaNotification(
  message: JsonRpcMessage,
): message is ServerNotificationFor<"item/reasoning/textDelta"> {
  return serverNotification(message)?.method === "item/reasoning/textDelta";
}

export function isThreadSettingsUpdatedNotification(
  message: JsonRpcMessage,
): message is ServerNotificationFor<"thread/settings/updated"> {
  return serverNotification(message)?.method === "thread/settings/updated";
}

function hasTurnNotification(value: unknown, statuses: string[]): boolean {
  const params = record(value);
  const turn = record(params?.turn);
  return (
    hasString(params, "threadId") &&
    hasEntityId(turn) &&
    statuses.includes(readString(turn, "status") ?? "")
  );
}

function hasItemNotification(
  value: unknown,
  timestampKey: "startedAtMs" | "completedAtMs",
): boolean {
  const params = record(value);
  const item = record(params?.item);
  return (
    hasString(params, "threadId") &&
    hasString(params, "turnId") &&
    hasEntityId(item) &&
    hasString(item, "type") &&
    hasFiniteNumber(params, timestampKey)
  );
}

function hasItemTextDelta(value: unknown): boolean {
  const params = record(value);
  return (
    hasString(params, "threadId") &&
    hasString(params, "turnId") &&
    hasString(params, "itemId") &&
    typeof params?.delta === "string"
  );
}

function hasFileChangePatchUpdated(value: unknown): boolean {
  const params = record(value);
  return (
    hasString(params, "threadId") &&
    hasString(params, "turnId") &&
    hasString(params, "itemId") &&
    Array.isArray(params?.changes) &&
    params.changes.every((change) => {
      const entry = record(change);
      const kind = record(entry?.kind);
      const kindType = readString(kind, "type");
      const movePath = kind?.move_path;
      return (
        hasString(entry, "path") &&
        typeof entry?.diff === "string" &&
        (kindType === "add" ||
          kindType === "delete" ||
          kindType === "update") &&
        (movePath === undefined ||
          (kindType === "update" &&
            typeof movePath === "string" &&
            movePath.length > 0))
      );
    })
  );
}

function hasMcpToolCallProgress(value: unknown): boolean {
  const params = record(value);
  return (
    hasString(params, "threadId") &&
    hasString(params, "turnId") &&
    hasString(params, "itemId") &&
    hasString(params, "message")
  );
}

function hasReasoningIdentity(
  value: unknown,
  indexKey: "summaryIndex" | "contentIndex",
): boolean {
  const params = record(value);
  return (
    hasString(params, "threadId") &&
    hasString(params, "turnId") &&
    hasString(params, "itemId") &&
    hasFiniteNumber(params, indexKey)
  );
}

function hasReasoningDelta(
  value: unknown,
  indexKey: "summaryIndex" | "contentIndex",
): boolean {
  const params = record(value);
  return (
    hasReasoningIdentity(params, indexKey) && typeof params?.delta === "string"
  );
}

function hasThreadSettings(value: unknown): boolean {
  const params = record(value);
  const settings = record(params?.threadSettings);
  return (
    hasString(params, "threadId") &&
    hasString(settings, "model") &&
    hasString(settings, "modelProvider") &&
    typeof settings?.cwd === "string"
  );
}

function hasEntityId(value: unknown): boolean {
  return hasString(record(value), "id");
}

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function readString(
  value: Record<string, unknown> | undefined,
  key: string,
): string | undefined {
  const field = value?.[key];
  return typeof field === "string" && field.trim().length > 0
    ? field
    : undefined;
}

function hasString(
  value: Record<string, unknown> | undefined,
  key: string,
): boolean {
  return readString(value, key) !== undefined;
}

function hasFiniteNumber(
  value: Record<string, unknown> | undefined,
  key: string,
): boolean {
  const field = value?.[key];
  return typeof field === "number" && Number.isFinite(field);
}
