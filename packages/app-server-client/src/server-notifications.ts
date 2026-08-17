import {
  isJsonRpcNotification,
  type JsonRpcMessage,
  type AppInfo,
  type ServerNotification,
} from "./protocol.js";

export type RuntimeServerNotification = Extract<
  ServerNotification,
  {
    method:
      | "error"
      | "guardianWarning"
      | "turn/diff/updated"
      | "turn/moderationMetadata"
      | "turn/plan/updated"
      | "thread/started"
      | "turn/started"
      | "turn/completed"
      | "item/started"
      | "item/completed"
      | "item/autoApprovalReview/started"
      | "item/autoApprovalReview/completed"
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

export type AppListUpdatedServerNotification = Extract<
  ServerNotification,
  { method: "app/list/updated" }
>;

export type ScheduledTaskChangedServerNotification = Extract<
  ServerNotification,
  { method: "scheduledTask/changed" }
>;

export type ScheduledTaskRunUpdatedServerNotification = Extract<
  ServerNotification,
  { method: "scheduledTask/run/updated" }
>;

export type McpServerOauthLoginCompletedServerNotification = Extract<
  ServerNotification,
  { method: "mcpServer/oauthLogin/completed" }
>;

export type McpServerStatusUpdatedServerNotification = Extract<
  ServerNotification,
  { method: "mcpServer/startupStatus/updated" }
>;

export type CommandExecOutputDeltaServerNotification = Extract<
  ServerNotification,
  { method: "command/exec/outputDelta" }
>;

export type GuardianReviewStartedServerNotification = Extract<
  ServerNotification,
  { method: "item/autoApprovalReview/started" }
>;

export type GuardianReviewCompletedServerNotification = Extract<
  ServerNotification,
  { method: "item/autoApprovalReview/completed" }
>;

export type GuardianWarningServerNotification = Extract<
  ServerNotification,
  { method: "guardianWarning" }
>;

export function commandExecOutputDeltaServerNotification(
  message: JsonRpcMessage,
): CommandExecOutputDeltaServerNotification | undefined {
  if (
    !isJsonRpcNotification(message) ||
    message.method !== "command/exec/outputDelta"
  ) {
    return undefined;
  }
  const params = record(message.params);
  if (
    !params ||
    typeof params.processId !== "string" ||
    !params.processId.trim() ||
    (params.stream !== "stdout" && params.stream !== "stderr") ||
    typeof params.deltaBase64 !== "string" ||
    typeof params.capReached !== "boolean"
  ) {
    return undefined;
  }
  return message as CommandExecOutputDeltaServerNotification;
}

export function isCommandExecOutputDeltaServerNotification(
  message: JsonRpcMessage,
): message is CommandExecOutputDeltaServerNotification {
  return commandExecOutputDeltaServerNotification(message) !== undefined;
}

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
    case "guardianWarning":
      return hasGuardianWarningNotification(message.params)
        ? (message as ServerNotificationFor<"guardianWarning">)
        : undefined;
    case "turn/diff/updated":
      return hasTurnDiffUpdatedNotification(message.params)
        ? (message as ServerNotificationFor<"turn/diff/updated">)
        : undefined;
    case "turn/moderationMetadata":
      return hasTurnModerationMetadataNotification(message.params)
        ? (message as ServerNotificationFor<"turn/moderationMetadata">)
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
    case "item/autoApprovalReview/started":
      return hasGuardianReviewStartedNotification(message.params)
        ? (message as ServerNotificationFor<"item/autoApprovalReview/started">)
        : undefined;
    case "item/autoApprovalReview/completed":
      return hasGuardianReviewCompletedNotification(message.params)
        ? (message as ServerNotificationFor<"item/autoApprovalReview/completed">)
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

export function appListUpdatedServerNotification(
  message: JsonRpcMessage,
): AppListUpdatedServerNotification | undefined {
  if (
    !isJsonRpcNotification(message) ||
    message.method !== "app/list/updated"
  ) {
    return undefined;
  }
  const params = record(message.params);
  if (
    !params ||
    !hasOnlyKeys(params, ["data"]) ||
    !Array.isArray(params.data) ||
    !params.data.every(isAppInfo)
  ) {
    return undefined;
  }
  return message as AppListUpdatedServerNotification;
}

export function isAppListUpdatedNotification(
  message: JsonRpcMessage,
): message is AppListUpdatedServerNotification {
  return appListUpdatedServerNotification(message) !== undefined;
}

export function scheduledTaskChangedServerNotification(
  message: JsonRpcMessage,
): ScheduledTaskChangedServerNotification | undefined {
  if (
    !isJsonRpcNotification(message) ||
    message.method !== "scheduledTask/changed"
  ) {
    return undefined;
  }
  const params = record(message.params);
  if (
    !hasOnlyKeys(params, ["change", "taskId"]) ||
    !hasString(params, "taskId") ||
    (params?.change !== "created" &&
      params?.change !== "deleted" &&
      params?.change !== "enabled" &&
      params?.change !== "updated")
  ) {
    return undefined;
  }
  return message as ScheduledTaskChangedServerNotification;
}

export function isScheduledTaskChangedNotification(
  message: JsonRpcMessage,
): message is ScheduledTaskChangedServerNotification {
  return scheduledTaskChangedServerNotification(message) !== undefined;
}

export function scheduledTaskRunUpdatedServerNotification(
  message: JsonRpcMessage,
): ScheduledTaskRunUpdatedServerNotification | undefined {
  if (
    !isJsonRpcNotification(message) ||
    message.method !== "scheduledTask/run/updated"
  ) {
    return undefined;
  }
  const params = record(message.params);
  if (
    !hasOnlyKeys(params, [
      "attention",
      "error",
      "notificationPolicy",
      "runId",
      "status",
      "taskId",
      "threadId",
      "title",
      "turnId",
    ]) ||
    !hasString(params, "taskId") ||
    !hasString(params, "runId") ||
    !hasString(params, "status") ||
    (params?.status !== "success" &&
      params?.status !== "error" &&
      params?.status !== "canceled" &&
      params?.status !== "timeout" &&
      params?.status !== "missed") ||
    typeof params?.attention !== "boolean" ||
    (params?.notificationPolicy !== "all_runs" &&
      params?.notificationPolicy !== "failures" &&
      params?.notificationPolicy !== "none") ||
    !hasOptionalNullableStringValue(params, "title") ||
    !hasOptionalNullableStringValue(params, "threadId") ||
    !hasOptionalNullableStringValue(params, "turnId") ||
    !hasOptionalNullableStringValue(params, "error")
  ) {
    return undefined;
  }
  return message as ScheduledTaskRunUpdatedServerNotification;
}

export function isScheduledTaskRunUpdatedNotification(
  message: JsonRpcMessage,
): message is ScheduledTaskRunUpdatedServerNotification {
  return scheduledTaskRunUpdatedServerNotification(message) !== undefined;
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

export function isTurnDiffUpdatedNotification(
  message: JsonRpcMessage,
): message is ServerNotificationFor<"turn/diff/updated"> {
  return serverNotification(message)?.method === "turn/diff/updated";
}

export function isTurnModerationMetadataNotification(
  message: JsonRpcMessage,
): message is ServerNotificationFor<"turn/moderationMetadata"> {
  return serverNotification(message)?.method === "turn/moderationMetadata";
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

export function isGuardianReviewStartedNotification(
  message: JsonRpcMessage,
): message is GuardianReviewStartedServerNotification {
  return (
    serverNotification(message)?.method === "item/autoApprovalReview/started"
  );
}

export function isGuardianReviewCompletedNotification(
  message: JsonRpcMessage,
): message is GuardianReviewCompletedServerNotification {
  return (
    serverNotification(message)?.method === "item/autoApprovalReview/completed"
  );
}

export function isGuardianWarningNotification(
  message: JsonRpcMessage,
): message is GuardianWarningServerNotification {
  return serverNotification(message)?.method === "guardianWarning";
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

function hasGuardianWarningNotification(value: unknown): boolean {
  const params = record(value);
  return (
    hasOnlyKeys(params, ["message", "threadId"]) &&
    hasString(params, "threadId") &&
    hasString(params, "message")
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

function hasTurnDiffUpdatedNotification(value: unknown): boolean {
  const params = record(value);
  return (
    hasOnlyKeys(params, ["diff", "threadId", "turnId"]) &&
    hasString(params, "threadId") &&
    hasString(params, "turnId") &&
    typeof params?.diff === "string"
  );
}

function hasTurnModerationMetadataNotification(value: unknown): boolean {
  const params = record(value);
  return (
    hasOnlyKeys(params, ["metadata", "threadId", "turnId"]) &&
    hasString(params, "threadId") &&
    hasString(params, "turnId") &&
    Object.prototype.hasOwnProperty.call(params, "metadata") &&
    isJsonValue(params?.metadata)
  );
}

function isJsonValue(value: unknown): boolean {
  if (
    value === null ||
    typeof value === "boolean" ||
    typeof value === "string"
  ) {
    return true;
  }
  if (typeof value === "number") {
    return Number.isFinite(value);
  }
  if (Array.isArray(value)) {
    return value.every(isJsonValue);
  }
  const object = record(value);
  return Boolean(object && Object.values(object).every(isJsonValue));
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

function hasGuardianReviewStartedNotification(value: unknown): boolean {
  const params = record(value);
  const review = record(params?.review);
  return (
    hasOnlyKeys(params, [
      "action",
      "review",
      "reviewId",
      "startedAtMs",
      "targetItemId",
      "threadId",
      "turnId",
    ]) &&
    hasString(params, "threadId") &&
    hasString(params, "turnId") &&
    hasString(params, "reviewId") &&
    hasFiniteNumber(params, "startedAtMs") &&
    isGuardianAction(params?.action) &&
    isGuardianReview(review, "inProgress")
  );
}

function hasGuardianReviewCompletedNotification(value: unknown): boolean {
  const params = record(value);
  const review = record(params?.review);
  return (
    hasOnlyKeys(params, [
      "action",
      "completedAtMs",
      "decisionSource",
      "review",
      "reviewId",
      "startedAtMs",
      "targetItemId",
      "threadId",
      "turnId",
    ]) &&
    hasString(params, "threadId") &&
    hasString(params, "turnId") &&
    hasString(params, "reviewId") &&
    hasFiniteNumber(params, "startedAtMs") &&
    hasFiniteNumber(params, "completedAtMs") &&
    params?.decisionSource === "agent" &&
    isGuardianAction(params?.action) &&
    isGuardianReview(review, "terminal")
  );
}

function isGuardianReview(
  value: Record<string, unknown> | undefined,
  status: "inProgress" | "terminal",
): boolean {
  if (
    !hasOnlyKeys(value, [
      "rationale",
      "riskLevel",
      "status",
      "userAuthorization",
    ])
  ) {
    return false;
  }
  const allowedStatuses =
    status === "inProgress"
      ? ["inProgress"]
      : ["approved", "denied", "timedOut", "aborted"];
  return (
    typeof value?.status === "string" &&
    allowedStatuses.includes(value.status) &&
    hasOptionalNullableStringValue(value, "rationale") &&
    hasOptionalNullableEnum(value, "riskLevel", [
      "low",
      "medium",
      "high",
      "critical",
    ]) &&
    hasOptionalNullableEnum(value, "userAuthorization", [
      "unknown",
      "low",
      "medium",
      "high",
    ])
  );
}

function isGuardianAction(value: unknown): boolean {
  const action = record(value);
  const type = readString(action, "type");
  if (!action || !type) return false;
  switch (type) {
    case "command":
      return (
        hasOnlyKeys(action, ["command", "cwd", "source", "type"]) &&
        hasString(action, "source") &&
        hasString(action, "command") &&
        hasString(action, "cwd")
      );
    case "execve":
      return (
        hasOnlyKeys(action, ["argv", "cwd", "program", "source", "type"]) &&
        hasString(action, "source") &&
        hasString(action, "program") &&
        hasString(action, "cwd") &&
        Array.isArray(action.argv) &&
        action.argv.every((value) => typeof value === "string")
      );
    case "applyPatch":
      return (
        hasOnlyKeys(action, ["cwd", "files", "type"]) &&
        hasString(action, "cwd") &&
        Array.isArray(action.files) &&
        action.files.every((value) => typeof value === "string")
      );
    case "networkAccess":
      return (
        hasOnlyKeys(action, ["host", "port", "protocol", "target", "type"]) &&
        hasString(action, "target") &&
        hasString(action, "host") &&
        hasString(action, "protocol") &&
        Number.isSafeInteger(action.port) &&
        (action.port as number) >= 0 &&
        (action.port as number) <= 65_535
      );
    case "mcpToolCall":
      return (
        hasOnlyKeys(action, [
          "connectorId",
          "connectorName",
          "server",
          "toolName",
          "toolTitle",
          "type",
        ]) &&
        hasString(action, "server") &&
        hasString(action, "toolName") &&
        hasOptionalNullableStringValue(action, "connectorId") &&
        hasOptionalNullableStringValue(action, "connectorName") &&
        hasOptionalNullableStringValue(action, "toolTitle")
      );
    case "requestPermissions":
      return (
        hasOnlyKeys(action, ["permissions", "reason", "type"]) &&
        Object.prototype.hasOwnProperty.call(action, "permissions") &&
        isJsonValue(action.permissions) &&
        hasOptionalNullableStringValue(action, "reason")
      );
    default:
      return false;
  }
}

function hasOptionalNullableStringValue(
  value: Record<string, unknown>,
  key: string,
): boolean {
  return (
    !Object.prototype.hasOwnProperty.call(value, key) ||
    value[key] === null ||
    typeof value[key] === "string"
  );
}

function hasOptionalNullableEnum(
  value: Record<string, unknown>,
  key: string,
  allowed: readonly string[],
): boolean {
  return (
    !Object.prototype.hasOwnProperty.call(value, key) ||
    value[key] === null ||
    (typeof value[key] === "string" && allowed.includes(value[key]))
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

function isAppInfo(value: unknown): value is AppInfo {
  const app = record(value);
  return Boolean(
    app &&
    hasString(app, "id") &&
    hasString(app, "name") &&
    typeof app.isAccessible === "boolean" &&
    typeof app.isEnabled === "boolean" &&
    Array.isArray(app.pluginDisplayNames) &&
    app.pluginDisplayNames.every((name) => typeof name === "string"),
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
