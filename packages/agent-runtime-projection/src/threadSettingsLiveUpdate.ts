import type {
  AgentUiProjectionContext,
  AgentUiProjectionEvent,
  AgentUiRuntimeStatus,
} from "@limecloud/agent-ui-contracts";

import {
  compactProjectionFields,
  definedString,
  readBooleanField,
  readRecord,
  readStringField,
} from "./normalization.js";

export type AgentUiThreadSettingsLiveUpdateIssueCode =
  | "missing_thread_id"
  | "missing_settings_notification"
  | "settings_notification_wrong_thread"
  | "notification_settings_missing_expected_fields"
  | "settings_only_started_model_request"
  | "active_turn_settings_polluted"
  | "future_turn_missing_updated_model"
  | "future_turn_missing_updated_service_tier"
  | "future_turn_missing_updated_cwd"
  | "settings_rendered_as_transcript_item"
  | "settings_persisted_as_read_model_item"
  | "ack_updated_cached_session_without_notification"
  | "cached_session_not_updated_from_notification"
  | "sandbox_policy_combined_with_permissions"
  | "turn_override_missing_settings_notification";

export interface AgentUiThreadSettingsLiveUpdateIssue {
  code: AgentUiThreadSettingsLiveUpdateIssueCode;
  path: string;
  message: string;
}

export interface AgentUiThreadSettingsSnapshot {
  model?: string;
  serviceTier?: string | null;
  cwd?: string;
  approvalPolicy?: string;
  approvalsReviewer?: string;
  permissions?: string;
  sandboxPolicy?: string;
  effort?: string;
  summary?: string;
  collaborationMode?: string;
  personality?: string;
}

export interface AgentUiThreadSettingsLiveUpdateProjectionInput {
  threadId?: string | null;
  settingsUpdateRequest?: unknown;
  turnStartOverride?: unknown;
  expectedSettings?: unknown;
  settingsNotification?: unknown;
  notifications?: unknown;
  modelRequestsDuringSettingsUpdate?: unknown;
  activeTurnBefore?: unknown;
  activeTurnAfter?: unknown;
  futureTurnRequired?: boolean;
  futureTurnRequest?: unknown;
  futureEnvironmentContext?: unknown;
  transcriptItems?: unknown;
  readModelItems?: unknown;
  cachedSessionBeforeAck?: unknown;
  cachedSessionAfterAck?: unknown;
  cachedSessionAfterNotification?: unknown;
  invalidSettingsUpdateError?: unknown;
  timestamp?: string | null;
}

export interface AgentUiThreadSettingsNotificationSnapshot {
  index: number;
  threadId?: string;
  method?: string;
  settings: AgentUiThreadSettingsSnapshot;
}

export interface AgentUiThreadSettingsLiveUpdateSnapshot {
  threadId?: string;
  expectedSettings: AgentUiThreadSettingsSnapshot;
  settingsNotifications: AgentUiThreadSettingsNotificationSnapshot[];
  settingsNotificationSeen: boolean;
  notificationMatchesExpected: boolean;
  settingsOnlyModelRequestCount: number;
  activeTurnSettingsStable: boolean;
  futureTurnUsesUpdatedModel: boolean;
  futureTurnUsesUpdatedServiceTier: boolean;
  futureTurnUsesUpdatedCwd: boolean;
  transcriptClean: boolean;
  readModelClean: boolean;
  ackDoesNotMutateCache: boolean;
  notificationUpdatesCache: boolean;
  invalidSandboxPermissionsRejected: boolean;
  validationIssues: AgentUiThreadSettingsLiveUpdateIssue[];
}

function issue(
  code: AgentUiThreadSettingsLiveUpdateIssueCode,
  path: string,
  message: string,
): AgentUiThreadSettingsLiveUpdateIssue {
  return { code, path, message };
}

function recordArray(value: unknown): Record<string, unknown>[] {
  if (Array.isArray(value)) {
    return value
      .map((item) => readRecord(item))
      .filter((item): item is Record<string, unknown> => Boolean(item));
  }
  const record = readRecord(value);
  if (!record) return [];
  const nested =
    record.data ?? record.items ?? record.notifications ?? record.requests;
  if (nested !== undefined) return recordArray(nested);
  return [record];
}

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableValue);
  const record = readRecord(value);
  if (!record) return value;
  return Object.fromEntries(
    Object.entries(record)
      .filter(([, entry]) => entry !== undefined)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, entry]) => [key, stableValue(entry)]),
  );
}

function stableEqual(left: unknown, right: unknown): boolean {
  if (left === undefined || right === undefined) return true;
  return (
    JSON.stringify(stableValue(left)) === JSON.stringify(stableValue(right))
  );
}

function readOptionalStringField(
  record: Record<string, unknown> | undefined,
  keys: string[],
): string | null | undefined {
  if (!record) return undefined;
  for (const key of keys) {
    const value = record[key];
    if (value === null) return null;
    if (typeof value === "string") return definedString(value) ?? undefined;
  }
  return undefined;
}

function readMethod(record: Record<string, unknown>): string | undefined {
  return readStringField(record, ["method"]);
}

function readThreadId(value: unknown): string | undefined {
  const record = readRecord(value);
  const params = readRecord(record?.params);
  return readStringField(params ?? record, ["threadId", "thread_id", "id"]);
}

function unwrapSettingsRecord(
  value: unknown,
): Record<string, unknown> | undefined {
  const record = readRecord(value);
  if (!record) return undefined;
  const params = readRecord(record.params);
  return (
    readRecord(params?.threadSettings) ??
    readRecord(params?.thread_settings) ??
    readRecord(record.threadSettings) ??
    readRecord(record.thread_settings) ??
    readRecord(record.settings) ??
    readRecord(record.thread) ??
    record
  );
}

function normalizeMode(value: unknown): string | undefined {
  if (typeof value === "string") return definedString(value);
  const record = readRecord(value);
  return readStringField(record, ["mode", "kind", "type"]);
}

function normalizeSandboxPolicy(value: unknown): string | undefined {
  if (typeof value === "string") return definedString(value);
  const record = readRecord(value);
  return (
    readStringField(record, ["type", "policy", "kind"]) ??
    (record ? "object" : undefined)
  );
}

function readSettingsSnapshot(value: unknown): AgentUiThreadSettingsSnapshot {
  const record = unwrapSettingsRecord(value);
  const activePermissionProfile = readRecord(record?.activePermissionProfile);
  return compactProjectionFields({
    model: readStringField(record, ["model", "modelId", "model_id"]),
    serviceTier: readOptionalStringField(record, [
      "serviceTier",
      "service_tier",
    ]),
    cwd: readStringField(record, [
      "cwd",
      "path",
      "workspaceRoot",
      "workspace_root",
    ]),
    approvalPolicy: readStringField(record, [
      "approvalPolicy",
      "approval_policy",
    ]),
    approvalsReviewer: readStringField(record, [
      "approvalsReviewer",
      "approvals_reviewer",
    ]),
    permissions:
      readStringField(record, ["permissions", "permissionProfile"]) ??
      readStringField(activePermissionProfile, ["id"]),
    sandboxPolicy: normalizeSandboxPolicy(
      record?.sandboxPolicy ?? record?.sandbox_policy,
    ),
    effort: readStringField(record, [
      "effort",
      "reasoningEffort",
      "reasoning_effort",
    ]),
    summary: readStringField(record, ["summary"]),
    collaborationMode: normalizeMode(record?.collaborationMode),
    personality: readStringField(record, ["personality"]),
  } satisfies AgentUiThreadSettingsSnapshot);
}

function settingsKeys(settings: AgentUiThreadSettingsSnapshot): string[] {
  return Object.entries(settings)
    .filter(([, value]) => value !== undefined)
    .map(([key]) => key);
}

function valuesMatch(expected: unknown, actual: unknown): boolean {
  if (expected === undefined) return true;
  if (expected === null) return actual === null || actual === undefined;
  return expected === actual;
}

function isSettingsNotification(record: Record<string, unknown>): boolean {
  const method = readMethod(record);
  const params = readRecord(record.params);
  return Boolean(
    method === "thread/settings/updated" ||
    method === "thread_settings_updated" ||
    method === "thread.settings.updated" ||
    params?.threadSettings ||
    params?.thread_settings,
  );
}

function readSettingsNotifications(
  input: AgentUiThreadSettingsLiveUpdateProjectionInput,
): AgentUiThreadSettingsNotificationSnapshot[] {
  const records = [
    ...recordArray(input.settingsNotification),
    ...recordArray(input.notifications),
  ].filter(isSettingsNotification);
  return records.map((record, index) => ({
    index,
    threadId: readThreadId(record),
    method: readMethod(record),
    settings: readSettingsSnapshot(record),
  }));
}

function readModelRequestCount(value: unknown): number {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  return recordArray(value).length;
}

function isSettingsProjectionRecord(record: Record<string, unknown>): boolean {
  const type = readStringField(record, [
    "type",
    "kind",
    "sourceType",
    "source_type",
  ]);
  const method = readMethod(record);
  return Boolean(
    isSettingsNotification(record) ||
    method === "thread/settings/update" ||
    type === "thread_settings" ||
    type === "thread_settings_updated" ||
    type === "settings_update" ||
    type === "thread_settings_live_update_projection",
  );
}

function leakIndexes(value: unknown): number[] {
  return recordArray(value)
    .map((record, index) => ({ record, index }))
    .filter(({ record }) => isSettingsProjectionRecord(record))
    .map(({ index }) => index);
}

function notificationMatchesExpectedFields(
  notification: AgentUiThreadSettingsSnapshot | undefined,
  expected: AgentUiThreadSettingsSnapshot,
): boolean {
  if (!notification) return false;
  for (const key of settingsKeys(expected) as Array<
    keyof AgentUiThreadSettingsSnapshot
  >) {
    if (key === "serviceTier" && expected[key] === null) continue;
    if (!valuesMatch(expected[key], notification[key])) return false;
  }
  return true;
}

function futureRequestRecord(
  value: unknown,
): Record<string, unknown> | undefined {
  return recordArray(value)[0];
}

function futureRequestUsesModel(
  record: Record<string, unknown> | undefined,
  expected: AgentUiThreadSettingsSnapshot,
  required: boolean,
): boolean {
  if (expected.model === undefined) return true;
  if (!record) return !required;
  return (
    readStringField(record, ["model", "model_id", "modelId"]) === expected.model
  );
}

function futureRequestUsesServiceTier(
  record: Record<string, unknown> | undefined,
  expected: AgentUiThreadSettingsSnapshot,
  required: boolean,
): boolean {
  if (expected.serviceTier === undefined) return true;
  if (!record) return !required;
  const actual = readOptionalStringField(record, [
    "service_tier",
    "serviceTier",
  ]);
  return valuesMatch(expected.serviceTier, actual);
}

function futureRequestUsesCwd(
  record: Record<string, unknown> | undefined,
  environmentContext: unknown,
  expected: AgentUiThreadSettingsSnapshot,
  required: boolean,
): boolean {
  if (expected.cwd === undefined) return true;
  if (!record && environmentContext === undefined) return !required;
  const actual =
    readStringField(record, ["cwd", "workspaceRoot", "workspace_root"]) ??
    readStringField(record, ["environmentContext", "environment_context"]);
  if (actual?.includes(expected.cwd)) return true;
  if (typeof environmentContext === "string")
    return environmentContext.includes(expected.cwd);
  return false;
}

function activeTurnSettingsStable(
  before: unknown,
  after: unknown,
  expected: AgentUiThreadSettingsSnapshot,
): boolean {
  if (before === undefined || after === undefined) return true;
  const beforeSettings = readSettingsSnapshot(before);
  const afterSettings = readSettingsSnapshot(after);
  for (const key of settingsKeys(expected) as Array<
    keyof AgentUiThreadSettingsSnapshot
  >) {
    const beforeValue = beforeSettings[key];
    const afterValue = afterSettings[key];
    if (beforeValue !== undefined && !valuesMatch(beforeValue, afterValue)) {
      return false;
    }
  }
  return true;
}

function updateCombinesSandboxPolicyAndPermissions(value: unknown): boolean {
  const settings = readSettingsSnapshot(value);
  return (
    settings.sandboxPolicy !== undefined && settings.permissions !== undefined
  );
}

function errorRejected(value: unknown): boolean {
  const record = readRecord(value);
  if (!record) return false;
  if (readBooleanField(record, ["blocked", "failed", "rejected"]) === true)
    return true;
  const status = readStringField(record, ["status", "error", "code"]);
  return Boolean(
    status &&
    ["error", "failed", "rejected", "invalid_request", "-32600"].includes(
      status.trim().toLowerCase(),
    ),
  );
}

function notificationUpdatesCache(
  before: unknown,
  afterNotification: unknown,
): boolean {
  if (before === undefined || afterNotification === undefined) return true;
  return !stableEqual(before, afterNotification);
}

function validateSnapshot(
  input: AgentUiThreadSettingsLiveUpdateProjectionInput,
  snapshot: Omit<AgentUiThreadSettingsLiveUpdateSnapshot, "validationIssues">,
): AgentUiThreadSettingsLiveUpdateIssue[] {
  const issues: AgentUiThreadSettingsLiveUpdateIssue[] = [];

  if (!snapshot.threadId) {
    issues.push(
      issue(
        "missing_thread_id",
        "$.threadId",
        "thread/settings/updated projection requires a thread id.",
      ),
    );
  }
  if (!snapshot.settingsNotificationSeen) {
    issues.push(
      issue(
        "missing_settings_notification",
        "$.settingsNotification",
        "Settings changes must be observed through thread/settings/updated.",
      ),
    );
  }
  for (const notification of snapshot.settingsNotifications) {
    if (snapshot.threadId && notification.threadId !== snapshot.threadId) {
      issues.push(
        issue(
          "settings_notification_wrong_thread",
          `$.settingsNotification[${notification.index}].params.threadId`,
          "thread/settings/updated must be scoped to the updated thread.",
        ),
      );
    }
  }
  if (
    snapshot.settingsNotificationSeen &&
    !snapshot.notificationMatchesExpected
  ) {
    issues.push(
      issue(
        "notification_settings_missing_expected_fields",
        "$.settingsNotification[].params.threadSettings",
        "thread/settings/updated must carry the effective updated settings.",
      ),
    );
  }
  if (snapshot.settingsOnlyModelRequestCount > 0) {
    issues.push(
      issue(
        "settings_only_started_model_request",
        "$.modelRequestsDuringSettingsUpdate",
        "thread/settings/update must not start a model request by itself.",
      ),
    );
  }
  if (!snapshot.activeTurnSettingsStable) {
    issues.push(
      issue(
        "active_turn_settings_polluted",
        "$.activeTurnAfter",
        "Settings changes must not rewrite the already active turn settings.",
      ),
    );
  }
  if (!snapshot.futureTurnUsesUpdatedModel) {
    issues.push(
      issue(
        "future_turn_missing_updated_model",
        "$.futureTurnRequest.model",
        "The next turn must use the updated thread model.",
      ),
    );
  }
  if (!snapshot.futureTurnUsesUpdatedServiceTier) {
    issues.push(
      issue(
        "future_turn_missing_updated_service_tier",
        "$.futureTurnRequest.service_tier",
        "The next turn must use the updated or cleared service tier.",
      ),
    );
  }
  if (!snapshot.futureTurnUsesUpdatedCwd) {
    issues.push(
      issue(
        "future_turn_missing_updated_cwd",
        "$.futureTurnRequest.environment_context",
        "The next turn environment context must use the updated cwd.",
      ),
    );
  }
  for (const index of leakIndexes(input.transcriptItems)) {
    issues.push(
      issue(
        "settings_rendered_as_transcript_item",
        `$.transcriptItems[${index}]`,
        "thread/settings/updated must update runtime settings surfaces, not transcript messages.",
      ),
    );
  }
  for (const index of leakIndexes(input.readModelItems)) {
    issues.push(
      issue(
        "settings_persisted_as_read_model_item",
        `$.readModelItems[${index}]`,
        "thread/settings/updated must not persist as a normal thread item.",
      ),
    );
  }
  if (!snapshot.ackDoesNotMutateCache) {
    issues.push(
      issue(
        "ack_updated_cached_session_without_notification",
        "$.cachedSessionAfterAck",
        "The settings update response is only an ack; cached UI state changes on notification.",
      ),
    );
  }
  if (!snapshot.notificationUpdatesCache) {
    issues.push(
      issue(
        "cached_session_not_updated_from_notification",
        "$.cachedSessionAfterNotification",
        "thread/settings/updated must update the cached session state.",
      ),
    );
  }
  if (!snapshot.invalidSandboxPermissionsRejected) {
    issues.push(
      issue(
        "sandbox_policy_combined_with_permissions",
        "$.settingsUpdateRequest",
        "`permissions` cannot be combined with `sandboxPolicy`.",
      ),
    );
  }
  if (
    settingsKeys(readSettingsSnapshot(input.turnStartOverride)).length > 0 &&
    !snapshot.settingsNotificationSeen
  ) {
    issues.push(
      issue(
        "turn_override_missing_settings_notification",
        "$.turnStartOverride",
        "turn/start settings overrides must also emit thread/settings/updated.",
      ),
    );
  }
  return issues;
}

export function extractThreadSettingsLiveUpdateSnapshot(
  input: AgentUiThreadSettingsLiveUpdateProjectionInput,
): AgentUiThreadSettingsLiveUpdateSnapshot {
  const notifications = readSettingsNotifications(input);
  const threadId =
    definedString(input.threadId ?? undefined) ??
    notifications.find((notification) => notification.threadId)?.threadId ??
    readThreadId(input.settingsUpdateRequest) ??
    readThreadId(input.turnStartOverride);
  const scopedNotifications = threadId
    ? notifications.filter((notification) => notification.threadId === threadId)
    : notifications;
  const expectedSettings = readSettingsSnapshot(
    input.expectedSettings ??
      input.settingsUpdateRequest ??
      input.turnStartOverride ??
      scopedNotifications[scopedNotifications.length - 1]?.settings,
  );
  const latestNotification =
    scopedNotifications[scopedNotifications.length - 1];
  const futureRequest = futureRequestRecord(input.futureTurnRequest);
  const invalidSandboxPermissionsRejected =
    !updateCombinesSandboxPolicyAndPermissions(input.settingsUpdateRequest) ||
    errorRejected(input.invalidSettingsUpdateError);
  const base = {
    threadId,
    expectedSettings,
    settingsNotifications: notifications,
    settingsNotificationSeen: scopedNotifications.length > 0,
    notificationMatchesExpected: notificationMatchesExpectedFields(
      latestNotification?.settings,
      expectedSettings,
    ),
    settingsOnlyModelRequestCount: readModelRequestCount(
      input.modelRequestsDuringSettingsUpdate,
    ),
    activeTurnSettingsStable: activeTurnSettingsStable(
      input.activeTurnBefore,
      input.activeTurnAfter,
      expectedSettings,
    ),
    futureTurnUsesUpdatedModel: futureRequestUsesModel(
      futureRequest,
      expectedSettings,
      input.futureTurnRequired === true,
    ),
    futureTurnUsesUpdatedServiceTier: futureRequestUsesServiceTier(
      futureRequest,
      expectedSettings,
      input.futureTurnRequired === true,
    ),
    futureTurnUsesUpdatedCwd: futureRequestUsesCwd(
      futureRequest,
      input.futureEnvironmentContext,
      expectedSettings,
      input.futureTurnRequired === true,
    ),
    transcriptClean: leakIndexes(input.transcriptItems).length === 0,
    readModelClean: leakIndexes(input.readModelItems).length === 0,
    ackDoesNotMutateCache: stableEqual(
      input.cachedSessionBeforeAck,
      input.cachedSessionAfterAck,
    ),
    notificationUpdatesCache: notificationUpdatesCache(
      input.cachedSessionBeforeAck,
      input.cachedSessionAfterNotification,
    ),
    invalidSandboxPermissionsRejected,
  };
  return {
    ...base,
    validationIssues: validateSnapshot(input, base),
  };
}

function runtimeStatus(
  issues: readonly AgentUiThreadSettingsLiveUpdateIssue[],
): AgentUiRuntimeStatus {
  return issues.length > 0 ? "failed" : "completed";
}

export function buildThreadSettingsLiveUpdateProjectionEvent(
  input: AgentUiThreadSettingsLiveUpdateProjectionInput,
  context: AgentUiProjectionContext = {},
): AgentUiProjectionEvent {
  const snapshot = extractThreadSettingsLiveUpdateSnapshot(input);
  const status = runtimeStatus(snapshot.validationIssues);
  return compactProjectionFields({
    type: "context.changed",
    sourceType: "thread_settings_live_update_projection",
    sequence: context.sequence,
    timestamp: definedString(input.timestamp ?? undefined) ?? context.timestamp,
    sessionId: definedString(context.sessionId ?? undefined),
    threadId: snapshot.threadId ?? definedString(context.threadId ?? undefined),
    runId: definedString(context.runId ?? undefined),
    turnId: definedString(context.turnId ?? undefined),
    owner: "context",
    scope: "thread",
    phase: status === "failed" ? "failed" : "completed",
    surface: "runtime_status",
    persistence: "snapshot",
    control: "none",
    runtimeEntity: "agent_turn",
    runtimeStatus: status,
    latestTurnStatus: status,
    payload: {
      threadSettingsLiveUpdateEvent: "thread_settings_updated",
      settingsNotificationSeen: snapshot.settingsNotificationSeen,
      notificationMatchesExpected: snapshot.notificationMatchesExpected,
      settingsOnlyModelRequestCount: snapshot.settingsOnlyModelRequestCount,
      activeTurnSettingsStable: snapshot.activeTurnSettingsStable,
      futureTurnUsesUpdatedModel: snapshot.futureTurnUsesUpdatedModel,
      futureTurnUsesUpdatedServiceTier:
        snapshot.futureTurnUsesUpdatedServiceTier,
      futureTurnUsesUpdatedCwd: snapshot.futureTurnUsesUpdatedCwd,
      transcriptClean: snapshot.transcriptClean,
      readModelClean: snapshot.readModelClean,
      ackDoesNotMutateCache: snapshot.ackDoesNotMutateCache,
      notificationUpdatesCache: snapshot.notificationUpdatesCache,
      invalidSandboxPermissionsRejected:
        snapshot.invalidSandboxPermissionsRejected,
      threadSettingsLiveUpdate: snapshot,
      validationIssues: snapshot.validationIssues,
    },
  } satisfies AgentUiProjectionEvent);
}
