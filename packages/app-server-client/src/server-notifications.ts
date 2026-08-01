import {
  isJsonRpcNotification,
  type JsonRpcMessage,
  type ServerNotification,
} from "./protocol.js";

export type RuntimeServerNotification = Extract<
  ServerNotification,
  {
    method:
      | "error"
      | "turn/plan/updated"
      | "thread/started"
      | "turn/started"
      | "turn/completed"
      | "item/started"
      | "item/completed"
      | "item/agentMessage/delta"
      | "item/commandExecution/outputDelta"
      | "item/commandExecution/terminalInteraction"
      | "item/fileChange/patchUpdated"
      | "item/mcpToolCall/progress"
      | "item/plan/delta"
      | "item/reasoning/summaryTextDelta"
      | "item/reasoning/summaryPartAdded"
      | "item/reasoning/textDelta"
      | "thread/settings/updated";
  }
>;

export type ModelListUpdatedServerNotification = Extract<
  ServerNotification,
  { method: "model/list/updated" }
>;

export type SkillsChangedServerNotification = Extract<
  ServerNotification,
  { method: "skills/changed" }
>;

export type McpServerOauthLoginCompletedServerNotification = Extract<
  ServerNotification,
  { method: "mcpServer/oauthLogin/completed" }
>;

export type McpServerStatusUpdatedServerNotification = Extract<
  ServerNotification,
  { method: "mcpServer/startupStatus/updated" }
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
    case "error":
      return hasErrorNotification(message.params)
        ? (message as ServerNotificationFor<"error">)
        : undefined;
    case "turn/plan/updated":
      return hasTurnPlanUpdatedNotification(message.params)
        ? (message as ServerNotificationFor<"turn/plan/updated">)
        : undefined;
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
    case "item/commandExecution/terminalInteraction":
      return hasTerminalInteraction(message.params)
        ? (message as ServerNotificationFor<"item/commandExecution/terminalInteraction">)
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

export function modelListUpdatedServerNotification(
  message: JsonRpcMessage,
): ModelListUpdatedServerNotification | undefined {
  if (
    !isJsonRpcNotification(message) ||
    message.method !== "model/list/updated"
  ) {
    return undefined;
  }
  const params = record(message.params);
  if (
    !params ||
    !Number.isSafeInteger(params.generation) ||
    (params.generation as number) < 0
  ) {
    return undefined;
  }
  if (
    params.providerId !== undefined &&
    params.providerId !== null &&
    !hasString(params, "providerId")
  ) {
    return undefined;
  }
  return message as ModelListUpdatedServerNotification;
}

export function isModelListUpdatedNotification(
  message: JsonRpcMessage,
): message is ModelListUpdatedServerNotification {
  return modelListUpdatedServerNotification(message) !== undefined;
}

export function skillsChangedServerNotification(
  message: JsonRpcMessage,
): SkillsChangedServerNotification | undefined {
  if (!isJsonRpcNotification(message) || message.method !== "skills/changed") {
    return undefined;
  }
  const params = record(message.params);
  if (!params || Object.keys(params).length !== 0) {
    return undefined;
  }
  return message as SkillsChangedServerNotification;
}

export function isSkillsChangedNotification(
  message: JsonRpcMessage,
): message is SkillsChangedServerNotification {
  return skillsChangedServerNotification(message) !== undefined;
}

export function mcpServerOauthLoginCompletedServerNotification(
  message: JsonRpcMessage,
): McpServerOauthLoginCompletedServerNotification | undefined {
  if (
    !isJsonRpcNotification(message) ||
    message.method !== "mcpServer/oauthLogin/completed"
  ) {
    return undefined;
  }
  const params = record(message.params);
  if (
    !hasOnlyKeys(params, ["error", "name", "success", "threadId"]) ||
    !hasString(params, "name") ||
    typeof params?.success !== "boolean" ||
    !hasRequiredNullableString(params, "threadId") ||
    !hasOptionalString(params, "error")
  ) {
    return undefined;
  }
  return message as McpServerOauthLoginCompletedServerNotification;
}

export function mcpServerStatusUpdatedServerNotification(
  message: JsonRpcMessage,
): McpServerStatusUpdatedServerNotification | undefined {
  if (
    !isJsonRpcNotification(message) ||
    message.method !== "mcpServer/startupStatus/updated"
  ) {
    return undefined;
  }
  const params = record(message.params);
  const status = params?.status;
  const failureReason = params?.failureReason;
  if (
    !hasOnlyKeys(params, [
      "error",
      "failureReason",
      "name",
      "status",
      "threadId",
    ]) ||
    !hasRequiredNullableString(params, "threadId") ||
    !hasString(params, "name") ||
    (status !== "starting" &&
      status !== "ready" &&
      status !== "failed" &&
      status !== "cancelled") ||
    !hasRequiredNullableString(params, "error") ||
    !Object.prototype.hasOwnProperty.call(params, "failureReason") ||
    (failureReason !== null && failureReason !== "reauthenticationRequired")
  ) {
    return undefined;
  }
  return message as McpServerStatusUpdatedServerNotification;
}

export function isThreadStartedNotification(
  message: JsonRpcMessage,
): message is ServerNotificationFor<"thread/started"> {
  return serverNotification(message)?.method === "thread/started";
}

export function isErrorNotification(
  message: JsonRpcMessage,
): message is ServerNotificationFor<"error"> {
  return serverNotification(message)?.method === "error";
}

export function isTurnPlanUpdatedNotification(
  message: JsonRpcMessage,
): message is ServerNotificationFor<"turn/plan/updated"> {
  return serverNotification(message)?.method === "turn/plan/updated";
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

export function isCommandExecutionTerminalInteractionNotification(
  message: JsonRpcMessage,
): message is ServerNotificationFor<"item/commandExecution/terminalInteraction"> {
  return (
    serverNotification(message)?.method ===
    "item/commandExecution/terminalInteraction"
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

function hasErrorNotification(value: unknown): boolean {
  const params = record(value);
  const error = record(params?.error);
  const additionalDetails = error?.additionalDetails;
  const codexErrorInfo = error?.codexErrorInfo;
  return (
    hasOnlyKeys(params, ["error", "threadId", "turnId", "willRetry"]) &&
    hasOnlyKeys(error, ["additionalDetails", "codexErrorInfo", "message"]) &&
    hasString(params, "threadId") &&
    hasString(params, "turnId") &&
    hasString(error, "message") &&
    typeof params?.willRetry === "boolean" &&
    hasCodexErrorInfo(codexErrorInfo) &&
    (additionalDetails === undefined ||
      additionalDetails === null ||
      typeof additionalDetails === "string")
  );
}

function hasTurnPlanUpdatedNotification(value: unknown): boolean {
  const params = record(value);
  if (
    !hasOnlyKeys(params, ["explanation", "plan", "threadId", "turnId"]) ||
    !hasString(params, "threadId") ||
    !hasString(params, "turnId") ||
    !Array.isArray(params?.plan) ||
    (params.explanation !== undefined &&
      params.explanation !== null &&
      typeof params.explanation !== "string")
  ) {
    return false;
  }
  return params.plan.every((value) => {
    const step = record(value);
    return (
      hasOnlyKeys(step, ["status", "step"]) &&
      hasString(step, "step") &&
      (step?.status === "pending" ||
        step?.status === "inProgress" ||
        step?.status === "completed")
    );
  });
}

function hasOnlyKeys(
  value: Record<string, unknown> | undefined,
  allowedKeys: string[],
): boolean {
  return Boolean(
    value && Object.keys(value).every((key) => allowedKeys.includes(key)),
  );
}

function hasRequiredNullableString(
  value: Record<string, unknown> | undefined,
  key: string,
): boolean {
  return Boolean(
    value &&
    Object.prototype.hasOwnProperty.call(value, key) &&
    (value[key] === null || hasString(value, key)),
  );
}

function hasOptionalString(
  value: Record<string, unknown> | undefined,
  key: string,
): boolean {
  return Boolean(
    value &&
    (!Object.prototype.hasOwnProperty.call(value, key) ||
      typeof value[key] === "string"),
  );
}

function hasCodexErrorInfo(value: unknown): boolean {
  if (value === undefined || value === null) {
    return true;
  }
  if (typeof value === "string") {
    return [
      "badRequest",
      "contextWindowExceeded",
      "cyberPolicy",
      "internalServerError",
      "other",
      "sandboxError",
      "serverOverloaded",
      "sessionBudgetExceeded",
      "threadRollbackFailed",
      "unauthorized",
      "usageLimitExceeded",
    ].includes(value);
  }

  const variant = record(value);
  if (!variant || Object.keys(variant).length !== 1) {
    return false;
  }
  const variantName = Object.keys(variant)[0];
  const details = record(variant[variantName]);
  if (!details) {
    return false;
  }
  if (variantName === "activeTurnNotSteerable") {
    return details.turnKind === "review" || details.turnKind === "compact";
  }
  if (
    ![
      "httpConnectionFailed",
      "responseStreamConnectionFailed",
      "responseStreamDisconnected",
      "responseTooManyFailedAttempts",
    ].includes(variantName)
  ) {
    return false;
  }
  const status = details.httpStatusCode;
  return (
    status === undefined ||
    status === null ||
    (Number.isInteger(status) &&
      (status as number) >= 0 &&
      (status as number) <= 65_535)
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

function hasTerminalInteraction(value: unknown): boolean {
  const params = record(value);
  return (
    hasOnlyKeys(params, [
      "itemId",
      "processId",
      "stdin",
      "threadId",
      "turnId",
    ]) &&
    hasString(params, "threadId") &&
    hasString(params, "turnId") &&
    hasString(params, "itemId") &&
    hasString(params, "processId") &&
    isTerminalInteractionSummary(params?.stdin)
  );
}

function isTerminalInteractionSummary(value: unknown): value is string {
  return (
    value === "(poll)" ||
    value === "(interrupt)" ||
    (typeof value === "string" && /^sent [0-9]+ chars$/u.test(value))
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
