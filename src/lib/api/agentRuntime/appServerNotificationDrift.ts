import { logAgentDebug } from "@/lib/agentDebug";
import type { AppServerJsonRpcNotification } from "@/lib/api/appServer";
import { RENDER_PROJECTION_REFERENCE_REVISION } from "./conversationProjection";

const CODEX_V2_NOTIFICATION_METHODS = new Set([
  "account/login/completed",
  "account/rateLimits/updated",
  "account/updated",
  "app/list/updated",
  "command/exec/outputDelta",
  "configWarning",
  "deprecationNotice",
  "error",
  "externalAgentConfig/import/completed",
  "externalAgentConfig/import/progress",
  "fs/changed",
  "fuzzyFileSearch/sessionCompleted",
  "fuzzyFileSearch/sessionUpdated",
  "guardianWarning",
  "hook/completed",
  "hook/started",
  "item/agentMessage/delta",
  "item/autoApprovalReview/completed",
  "item/autoApprovalReview/started",
  "item/commandExecution/outputDelta",
  "item/commandExecution/terminalInteraction",
  "item/completed",
  "item/fileChange/outputDelta",
  "item/fileChange/patchUpdated",
  "item/mcpToolCall/progress",
  "item/plan/delta",
  "item/reasoning/summaryPartAdded",
  "item/reasoning/summaryTextDelta",
  "item/reasoning/textDelta",
  "item/started",
  "mcpServer/oauthLogin/completed",
  "mcpServer/startupStatus/updated",
  "model/rerouted",
  "model/safetyBuffering/updated",
  "model/verification",
  "process/exited",
  "process/outputDelta",
  "rawResponse/completed",
  "rawResponseItem/completed",
  "remoteControl/status/changed",
  "serverRequest/resolved",
  "skills/changed",
  "thread/archived",
  "thread/closed",
  "thread/compacted",
  "thread/deleted",
  "thread/environment/connected",
  "thread/environment/disconnected",
  "thread/goal/cleared",
  "thread/goal/updated",
  "thread/name/updated",
  "thread/realtime/closed",
  "thread/realtime/error",
  "thread/realtime/itemAdded",
  "thread/realtime/outputAudio/delta",
  "thread/realtime/sdp",
  "thread/realtime/started",
  "thread/realtime/transcript/delta",
  "thread/realtime/transcript/done",
  "thread/settings/updated",
  "thread/started",
  "thread/status/changed",
  "thread/tokenUsage/updated",
  "thread/unarchived",
  "turn/completed",
  "turn/diff/updated",
  "turn/moderationMetadata",
  "turn/plan/updated",
  "turn/started",
  "warning",
  "windows/worldWritableWarning",
  "windowsSandbox/setupCompleted",
]);

const DIAGNOSTIC_ONLY_NOTIFICATION_METHODS = new Set([
  "account/login/completed",
  "account/rateLimits/updated",
  "account/updated",
  "deprecationNotice",
  "externalAgentConfig/import/completed",
  "externalAgentConfig/import/progress",
  "item/fileChange/outputDelta",
  "process/exited",
  "process/outputDelta",
  "rawResponse/completed",
  "rawResponseItem/completed",
  "remoteControl/status/changed",
  "thread/compacted",
]);
const PROJECTED_NOTIFICATION_METHODS = new Set([
  "hook/started",
  "hook/completed",
  "item/autoApprovalReview/started",
  "item/autoApprovalReview/completed",
  "guardianWarning",
  "turn/diff/updated",
  "turn/moderationMetadata",
]);

const MAX_DIAGNOSTIC_FIELDS = 32;

export function listCodexV2NotificationMethods(): string[] {
  return [...CODEX_V2_NOTIFICATION_METHODS].sort((left, right) =>
    left.localeCompare(right),
  );
}

export type AppServerNotificationDriftDisposition =
  | "known_diagnostic_only"
  | "known_projected"
  | "known_unprojected"
  | "unknown";

export interface AppServerNotificationDrift {
  disposition: AppServerNotificationDriftDisposition;
  field_names: readonly string[];
  method: string;
  protocol_revision: string;
  thread_id?: string;
  turn_id?: string;
}

export interface AppServerNotificationDriftRoute {
  threadId: string;
  turnId?: string;
}

export function readAppServerNotificationDrift(
  notification: AppServerJsonRpcNotification,
): AppServerNotificationDrift {
  const params = asRecord(notification.params);
  const route = readAppServerNotificationDriftRoute(notification);
  const known = CODEX_V2_NOTIFICATION_METHODS.has(notification.method);
  return {
    disposition: PROJECTED_NOTIFICATION_METHODS.has(notification.method)
      ? "known_projected"
      : DIAGNOSTIC_ONLY_NOTIFICATION_METHODS.has(notification.method)
        ? "known_diagnostic_only"
        : known
          ? "known_unprojected"
          : "unknown",
    field_names: Object.keys(params ?? {})
      .sort((left, right) => left.localeCompare(right))
      .slice(0, MAX_DIAGNOSTIC_FIELDS),
    method: notification.method,
    protocol_revision: RENDER_PROJECTION_REFERENCE_REVISION,
    ...(route?.threadId ? { thread_id: route.threadId } : {}),
    ...(route?.turnId ? { turn_id: route.turnId } : {}),
  };
}

export function readAppServerNotificationDriftRoute(
  notification: AppServerJsonRpcNotification,
): AppServerNotificationDriftRoute | null {
  const params = asRecord(notification.params);
  if (!params) {
    return null;
  }
  const thread = asRecord(params.thread);
  const turn = asRecord(params.turn);
  const threadId =
    readString(params, "threadId", "thread_id") ??
    readString(thread, "id") ??
    readString(turn, "threadId", "thread_id");
  if (!threadId) {
    return null;
  }
  const turnId =
    readString(params, "turnId", "turn_id") ?? readString(turn, "id");
  return {
    threadId,
    ...(turnId ? { turnId } : {}),
  };
}

export function recordAppServerNotificationDrift(
  notification: AppServerJsonRpcNotification,
): AppServerNotificationDrift {
  const diagnostic = readAppServerNotificationDrift(notification);
  logAgentDebug(
    "AgentProtocol",
    "notificationDrift",
    { ...diagnostic },
    {
      dedupeKey: `app-server-notification-drift:${diagnostic.method}`,
      level: diagnostic.disposition === "unknown" ? "warn" : "info",
      throttleMs: 60_000,
    },
  );
  return diagnostic;
}

export function projectAppServerNotificationDriftPayload(
  notification: AppServerJsonRpcNotification,
): Record<string, unknown> | null {
  const diagnostic = recordAppServerNotificationDrift(notification);
  if (
    diagnostic.disposition === "known_diagnostic_only" ||
    diagnostic.disposition === "known_projected"
  ) {
    return null;
  }
  const unknown = diagnostic.disposition === "unknown";
  return {
    code: `${unknown ? "unknown" : "unprojected"}_app_server_notification:${diagnostic.method}`,
    field_names: diagnostic.field_names,
    message: diagnostic.method,
    protocol_method: diagnostic.method,
    protocol_revision: diagnostic.protocol_revision,
    renderer_event_received_at: Date.now(),
    ...(diagnostic.thread_id
      ? {
          session_id: diagnostic.thread_id,
          thread_id: diagnostic.thread_id,
        }
      : {}),
    ...(diagnostic.turn_id ? { turn_id: diagnostic.turn_id } : {}),
    type: "warning",
  };
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function readString(
  record: Record<string, unknown> | null,
  ...keys: string[]
): string | undefined {
  if (!record) {
    return undefined;
  }
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return undefined;
}
