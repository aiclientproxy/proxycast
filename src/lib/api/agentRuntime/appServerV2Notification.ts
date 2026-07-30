import type {
  AppServerAgentEvent,
  AppServerJsonRpcNotification,
} from "@/lib/api/appServer";
import { readCanonicalThreadItem } from "./appServerCanonicalItemReader";
import { RENDER_PROJECTION_REFERENCE_REVISION } from "./conversationProjection";

const DIRECT_V2_NOTIFICATION_METHODS = new Set([
  "warning",
  "thread/started",
  "turn/started",
  "turn/completed",
  "item/started",
  "item/completed",
  "item/agentMessage/delta",
  "item/commandExecution/outputDelta",
  "item/fileChange/patchUpdated",
  "item/mcpToolCall/progress",
  "item/plan/delta",
  "item/reasoning/summaryTextDelta",
  "item/reasoning/summaryPartAdded",
  "item/reasoning/textDelta",
  "thread/tokenUsage/updated",
]);
const MAX_DIRECT_ITEM_SEQUENCE_TURNS = 512;

type DirectItemSequenceState = {
  itemSequences: Map<string, number>;
  nextSequence: number;
};

const directItemSequenceByTurn = new Map<string, DirectItemSequenceState>();

export function isAppServerV2NotificationMethod(method: string): boolean {
  return DIRECT_V2_NOTIFICATION_METHODS.has(method);
}

export function resetAppServerV2NotificationProjectionState(): void {
  directItemSequenceByTurn.clear();
}

export type AppServerV2NotificationRoute = {
  itemId?: string;
  terminal: boolean;
  threadId: string;
  turnId?: string;
};

export function isAppServerV2Notification(
  notification: AppServerJsonRpcNotification,
): boolean {
  return readAppServerV2NotificationRoute(notification) !== null;
}

export function readAppServerV2NotificationRoute(
  notification: AppServerJsonRpcNotification,
): AppServerV2NotificationRoute | null {
  if (!DIRECT_V2_NOTIFICATION_METHODS.has(notification.method)) {
    return null;
  }

  const params = asRecord(notification.params);
  if (!params) {
    return null;
  }

  switch (notification.method) {
    case "warning": {
      const threadId = readString(params, "threadId");
      const message = readString(params, "message");
      const code = readString(params, "code");
      const validCode =
        params.code === undefined || params.code === null || code !== undefined;
      return threadId && message && validCode
        ? { terminal: false, threadId }
        : null;
    }
    case "thread/started": {
      const thread = asRecord(params.thread);
      const threadId = readString(thread, "id");
      return threadId ? { terminal: false, threadId } : null;
    }
    case "turn/started":
    case "turn/completed": {
      const threadId = readString(params, "threadId");
      const turn = asRecord(params.turn);
      const turnId = readString(turn, "id");
      const status = readString(turn, "status");
      const validStatus =
        notification.method === "turn/started"
          ? status === "inProgress"
          : status === "completed" ||
            status === "failed" ||
            status === "interrupted";
      return threadId && turnId && validStatus
        ? {
            terminal: notification.method === "turn/completed",
            threadId,
            turnId,
          }
        : null;
    }
    case "item/started":
    case "item/completed": {
      const threadId = readString(params, "threadId");
      const turnId = readString(params, "turnId");
      const item = asRecord(params.item);
      const itemId = readString(item, "id");
      const timestampKey =
        notification.method === "item/started"
          ? "startedAtMs"
          : "completedAtMs";
      const timestampMs = readFiniteNumber(params, timestampKey);
      return threadId && turnId && itemId && timestampMs !== undefined
        ? { itemId, terminal: false, threadId, turnId }
        : null;
    }
    case "item/agentMessage/delta":
    case "item/plan/delta": {
      const threadId = readString(params, "threadId");
      const turnId = readString(params, "turnId");
      const itemId = readString(params, "itemId");
      return threadId && turnId && itemId && typeof params.delta === "string"
        ? { itemId, terminal: false, threadId, turnId }
        : null;
    }
    case "item/commandExecution/outputDelta": {
      const threadId = readString(params, "threadId");
      const turnId = readString(params, "turnId");
      const itemId = readString(params, "itemId");
      return threadId && turnId && itemId && typeof params.delta === "string"
        ? { itemId, terminal: false, threadId, turnId }
        : null;
    }
    case "item/fileChange/patchUpdated": {
      const threadId = readString(params, "threadId");
      const turnId = readString(params, "turnId");
      const itemId = readString(params, "itemId");
      return threadId &&
        turnId &&
        itemId &&
        readFileChangePatchUpdatedChanges(params) !== null
        ? { itemId, terminal: false, threadId, turnId }
        : null;
    }
    case "item/mcpToolCall/progress": {
      const threadId = readString(params, "threadId");
      const turnId = readString(params, "turnId");
      const itemId = readString(params, "itemId");
      const message = readString(params, "message");
      return threadId && turnId && itemId && message
        ? { itemId, terminal: false, threadId, turnId }
        : null;
    }
    case "item/reasoning/summaryTextDelta":
      return readReasoningNotificationRoute(params, "summaryIndex", true);
    case "item/reasoning/summaryPartAdded":
      return readReasoningNotificationRoute(params, "summaryIndex", false);
    case "item/reasoning/textDelta":
      return readReasoningNotificationRoute(params, "contentIndex", true);
    case "thread/tokenUsage/updated": {
      const threadId = readString(params, "threadId");
      const turnId = readString(params, "turnId");
      const tokenUsage = asRecord(params.tokenUsage);
      const last = asRecord(tokenUsage?.last);
      return threadId && turnId && last
        ? { terminal: false, threadId, turnId }
        : null;
    }
    default:
      return null;
  }
}

export function projectAppServerV2NotificationPayload(
  notification: AppServerJsonRpcNotification,
): Record<string, unknown> | null {
  const route = readAppServerV2NotificationRoute(notification);
  const params = asRecord(notification.params);
  if (!route || !params) {
    return null;
  }

  const receivedAtMs = Date.now();
  const emittedAtMs = notificationTimestampMs(notification.method, params);
  const timestamp = timestampFromMs(emittedAtMs ?? receivedAtMs);
  const basePayload = {
    protocol_method: notification.method,
    protocol_revision: RENDER_PROJECTION_REFERENCE_REVISION,
    renderer_event_received_at: receivedAtMs,
    server_event_emitted_at: emittedAtMs ?? null,
    session_id: route.threadId,
    thread_id: route.threadId,
    timestamp,
    ...(route.turnId ? { turn_id: route.turnId } : {}),
  };

  switch (notification.method) {
    case "warning": {
      const message = readString(params, "message");
      const code = readString(params, "code");
      return message
        ? {
            ...basePayload,
            type: "warning",
            message,
            ...(code ? { code } : {}),
          }
        : null;
    }
    case "thread/started":
      return {
        ...basePayload,
        type: "thread_started",
      };
    case "turn/started":
    case "turn/completed": {
      const sourceTurn = asRecord(params.turn);
      const turn = projectTurn(sourceTurn, route, timestamp);
      if (!turn) {
        return null;
      }
      const text =
        notification.method === "turn/completed"
          ? completedTurnFinalAnswerText(sourceTurn)
          : undefined;
      const payload = {
        ...basePayload,
        type: turnEventType(turn.status),
        turn,
        ...(text ? { text } : {}),
      };
      if (notification.method === "turn/completed" && route.turnId) {
        directItemSequenceByTurn.delete(
          directItemSequenceTurnKey(route.threadId, route.turnId),
        );
      }
      return payload;
    }
    case "item/started":
    case "item/completed": {
      const itemRecord = asRecord(params.item);
      if (!itemRecord || !route.turnId) {
        return null;
      }
      const sequence = directItemSequence(route, route.itemId);
      const event: AppServerAgentEvent = {
        eventId: "direct-v2",
        payload: params,
        sequence,
        sessionId: route.threadId,
        threadId: route.threadId,
        timestamp,
        turnId: route.turnId,
        type:
          notification.method === "item/started"
            ? "item.started"
            : "item.completed",
      };
      const item = readCanonicalThreadItem(itemRecord, event);
      if (!item) {
        return null;
      }
      return {
        ...basePayload,
        sequence,
        sequence_provenance: "notification_order",
        type:
          notification.method === "item/started"
            ? "item_started"
            : "item_completed",
        item,
      };
    }
    case "item/agentMessage/delta":
      return {
        ...basePayload,
        type: "text_delta",
        text: params.delta,
        itemId: route.itemId,
        item_id: route.itemId,
      };
    case "item/commandExecution/outputDelta":
      return {
        ...basePayload,
        type: "tool_output_delta",
        tool_id: route.itemId,
        toolId: route.itemId,
        sourceItemId: route.itemId,
        source_item_id: route.itemId,
        delta: params.delta,
        metadata: {
          source: "app_server_v2",
          sourceItemId: route.itemId,
          source_item_id: route.itemId,
        },
      };
    case "item/fileChange/patchUpdated": {
      const changes = readFileChangePatchUpdatedChanges(params);
      if (!changes || !route.itemId || !route.turnId) {
        return null;
      }
      const sequence = directItemSequence(route, route.itemId);
      const event: AppServerAgentEvent = {
        eventId: "direct-v2",
        payload: params,
        sequence,
        sessionId: route.threadId,
        threadId: route.threadId,
        timestamp,
        turnId: route.turnId,
        type: "item.updated",
      };
      const item = readCanonicalThreadItem(
        {
          changes,
          id: route.itemId,
          status: "inProgress",
          type: "fileChange",
        },
        event,
      );
      return item
        ? {
            ...basePayload,
            sequence,
            sequence_provenance: "notification_order",
            type: "item_updated",
            item,
          }
        : null;
    }
    case "item/mcpToolCall/progress": {
      const message = readString(params, "message");
      if (!route.itemId || !message) {
        return null;
      }
      return {
        ...basePayload,
        type: "tool_progress",
        tool_id: route.itemId,
        progress: {
          message,
          metadata: {
            notification_kind: "mcp_progress",
            source: "app_server_v2",
            source_item_id: route.itemId,
          },
        },
      };
    }
    case "item/plan/delta":
      return {
        ...basePayload,
        type: "plan_delta",
        text: params.delta,
        delta: params.delta,
        sourceItemId: route.itemId,
        source_item_id: route.itemId,
        revisionId: route.itemId,
        revision_id: route.itemId,
        source: "app_server_v2",
      };
    case "item/reasoning/summaryTextDelta": {
      const summaryIndex = readFiniteNumber(params, "summaryIndex");
      return {
        ...basePayload,
        type: "reasoning_summary_delta",
        reasoningId: route.itemId,
        reasoning_id: route.itemId,
        itemId: route.itemId,
        item_id: route.itemId,
        text: params.delta,
        delta: params.delta,
        summaryIndex,
        summary_index: summaryIndex,
      };
    }
    case "item/reasoning/summaryPartAdded": {
      const summaryIndex = readFiniteNumber(params, "summaryIndex");
      return {
        ...basePayload,
        type: "reasoning_summary_part_added",
        reasoningId: route.itemId,
        reasoning_id: route.itemId,
        itemId: route.itemId,
        item_id: route.itemId,
        summaryIndex,
        summary_index: summaryIndex,
      };
    }
    case "item/reasoning/textDelta": {
      const contentIndex = readFiniteNumber(params, "contentIndex");
      return {
        ...basePayload,
        type: "reasoning_content_delta",
        reasoningId: route.itemId,
        reasoning_id: route.itemId,
        itemId: route.itemId,
        item_id: route.itemId,
        text: params.delta,
        delta: params.delta,
        contentIndex,
        content_index: contentIndex,
      };
    }
    case "thread/tokenUsage/updated": {
      const tokenUsage = asRecord(params.tokenUsage);
      const last = asRecord(tokenUsage?.last);
      if (!last) {
        return null;
      }
      const inputTokens = readFiniteNumber(last, "inputTokens");
      const outputTokens = readFiniteNumber(last, "outputTokens");
      if (inputTokens === undefined || outputTokens === undefined) {
        return null;
      }
      return {
        ...basePayload,
        type: "token_usage_updated",
        usage: {
          input_tokens: inputTokens,
          output_tokens: outputTokens,
          cached_input_tokens: readFiniteNumber(last, "cachedInputTokens"),
          cache_creation_input_tokens: readFiniteNumber(
            last,
            "cacheWriteInputTokens",
          ),
        },
      };
    }
    default:
      return null;
  }
}

function directItemSequence(
  route: AppServerV2NotificationRoute,
  itemId: string | undefined,
): number {
  if (!route.turnId || !itemId) {
    return 0;
  }
  const turnKey = directItemSequenceTurnKey(route.threadId, route.turnId);
  let state = directItemSequenceByTurn.get(turnKey);
  if (!state) {
    if (directItemSequenceByTurn.size >= MAX_DIRECT_ITEM_SEQUENCE_TURNS) {
      const oldestTurnKey = directItemSequenceByTurn.keys().next().value;
      if (oldestTurnKey !== undefined) {
        directItemSequenceByTurn.delete(oldestTurnKey);
      }
    }
    state = { itemSequences: new Map(), nextSequence: 0 };
    directItemSequenceByTurn.set(turnKey, state);
  }
  const existing = state.itemSequences.get(itemId);
  if (existing !== undefined) {
    return existing;
  }
  const sequence = state.nextSequence;
  state.nextSequence += 1;
  state.itemSequences.set(itemId, sequence);
  return sequence;
}

function directItemSequenceTurnKey(threadId: string, turnId: string): string {
  return `${threadId}\u001f${turnId}`;
}

function readReasoningNotificationRoute(
  params: Record<string, unknown>,
  indexKey: "summaryIndex" | "contentIndex",
  requiresDelta: boolean,
): AppServerV2NotificationRoute | null {
  const threadId = readString(params, "threadId");
  const turnId = readString(params, "turnId");
  const itemId = readString(params, "itemId");
  const index = readFiniteNumber(params, indexKey);
  if (
    !threadId ||
    !turnId ||
    !itemId ||
    index === undefined ||
    (requiresDelta && typeof params.delta !== "string")
  ) {
    return null;
  }
  return { itemId, terminal: false, threadId, turnId };
}

function readFileChangePatchUpdatedChanges(
  params: Record<string, unknown>,
): Record<string, unknown>[] | null {
  if (!Array.isArray(params.changes)) {
    return null;
  }
  const changes: Record<string, unknown>[] = [];
  for (const value of params.changes) {
    const change = asRecord(value);
    const kind = asRecord(change?.kind);
    const path = readString(change, "path");
    const kindType = readString(kind, "type");
    if (
      !change ||
      !kind ||
      !path ||
      typeof change.diff !== "string" ||
      (kindType !== "add" && kindType !== "delete" && kindType !== "update")
    ) {
      return null;
    }
    const movePath = kind.move_path;
    if (
      (movePath !== undefined &&
        (kindType !== "update" ||
          typeof movePath !== "string" ||
          movePath.length === 0)) ||
      (kindType !== "update" && "move_path" in kind)
    ) {
      return null;
    }
    changes.push(change);
  }
  return changes;
}

function projectTurn(
  turn: Record<string, unknown> | undefined,
  route: AppServerV2NotificationRoute,
  fallbackTimestamp: string,
): Record<string, unknown> | null {
  const id = readString(turn, "id");
  const status = readString(turn, "status");
  if (!turn || !id || !status || id !== route.turnId) {
    return null;
  }

  const projectedStatus = turnStatus(status);
  if (!projectedStatus) {
    return null;
  }
  const startedAt = timestampFromUnixSeconds(
    readFiniteNumber(turn, "startedAt"),
  );
  const completedAt = timestampFromUnixSeconds(
    readFiniteNumber(turn, "completedAt"),
  );
  const error = asRecord(turn.error);
  const startedTimestamp = startedAt ?? completedAt ?? fallbackTimestamp;
  const completedTimestamp =
    projectedStatus === "running"
      ? undefined
      : (completedAt ?? fallbackTimestamp);

  return {
    id,
    thread_id: route.threadId,
    prompt_text: "",
    status: projectedStatus,
    started_at: startedTimestamp,
    ...(completedTimestamp ? { completed_at: completedTimestamp } : {}),
    error_message:
      readString(error, "message") ??
      (projectedStatus === "failed" ? "App Server turn failed" : undefined),
    created_at: startedTimestamp,
    updated_at: completedTimestamp ?? startedTimestamp,
  };
}

function notificationTimestampMs(
  method: string,
  params: Record<string, unknown>,
): number | undefined {
  if (method === "thread/started") {
    return unixSecondsToMs(
      readFiniteNumber(asRecord(params.thread), "createdAt"),
    );
  }
  if (method === "turn/started" || method === "turn/completed") {
    const turn = asRecord(params.turn);
    return unixSecondsToMs(
      readFiniteNumber(
        turn,
        method === "turn/started" ? "startedAt" : "completedAt",
      ),
    );
  }
  if (method === "item/started") {
    return readFiniteNumber(params, "startedAtMs");
  }
  if (method === "item/completed") {
    return readFiniteNumber(params, "completedAtMs");
  }
  return undefined;
}

function turnStatus(status: string): string | undefined {
  switch (status) {
    case "inProgress":
      return "running";
    case "completed":
      return "completed";
    case "failed":
      return "failed";
    case "interrupted":
      return "interrupted";
    default:
      return undefined;
  }
}

function turnEventType(status: unknown): string {
  switch (status) {
    case "completed":
      return "turn_completed";
    case "failed":
      return "turn_failed";
    case "canceled":
    case "interrupted":
      return "turn_canceled";
    default:
      return "turn_started";
  }
}

function completedTurnFinalAnswerText(
  turn: Record<string, unknown> | undefined,
): string | undefined {
  const items = Array.isArray(turn?.items) ? turn.items : [];
  for (let index = items.length - 1; index >= 0; index -= 1) {
    const item = asRecord(items[index]);
    if (readString(item, "type") !== "agentMessage") {
      continue;
    }
    const phase = readString(item, "phase")?.trim().toLowerCase();
    if (phase !== "final_answer" && phase !== "final") {
      continue;
    }
    const text = readString(item, "text");
    if (text) {
      return text;
    }
  }
  return undefined;
}

function timestampFromUnixSeconds(
  value: number | undefined,
): string | undefined {
  return value === undefined ? undefined : timestampFromMs(value * 1_000);
}

function unixSecondsToMs(value: number | undefined): number | undefined {
  return value === undefined ? undefined : value * 1_000;
}

function timestampFromMs(value: number): string {
  const timestamp = new Date(value);
  return Number.isNaN(timestamp.getTime())
    ? new Date(0).toISOString()
    : timestamp.toISOString();
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function readString(
  record: Record<string, unknown> | undefined,
  key: string,
): string | undefined {
  const value = record?.[key];
  return typeof value === "string" && value.trim().length > 0
    ? value.trim()
    : undefined;
}

function readFiniteNumber(
  record: Record<string, unknown> | undefined,
  key: string,
): number | undefined {
  const value = record?.[key];
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : undefined;
}
