import type { AgentThreadItem, AgentThreadTurn } from "../../agentProtocol";
import { createConversationProjectionReducer } from "./reducer";
import type {
  ConversationProjectionEvent,
  ConversationProjectionSource,
} from "./contracts";
import { RENDER_PROJECTION_REFERENCE_REVISION } from "./protocolDrift";

export function conversationProjectionEventFromPayload(
  payload: Record<string, unknown>,
  source: ConversationProjectionSource,
  eventId?: string,
): ConversationProjectionEvent | null {
  const type = readString(payload, "type");
  if (!type) return null;
  const base = {
    source,
    ...(eventId ? { event_id: eventId } : {}),
    protocol_revision:
      readString(payload, "protocol_revision", "protocolRevision") ??
      RENDER_PROJECTION_REFERENCE_REVISION,
    protocol_method:
      readString(payload, "protocol_method", "protocolMethod") ??
      sourceMethod(source),
  };
  switch (type) {
    case "thread_started": {
      const threadId = readString(
        payload,
        "thread_id",
        "threadId",
        "session_id",
      );
      return threadId
        ? { ...base, type, thread_id: threadId, status: "running" }
        : null;
    }
    case "turn_started":
    case "turn_completed": {
      const turn = readRecord(payload.turn);
      return isAgentThreadTurn(turn)
        ? { ...base, type, turn: turn as unknown as AgentThreadTurn }
        : null;
    }
    case "item_started":
    case "item_updated":
    case "item_completed": {
      const item = readRecord(payload.item);
      return isAgentThreadItem(item)
        ? { ...base, type, item: item as unknown as AgentThreadItem }
        : null;
    }
    case "text_delta": {
      const threadId = readString(
        payload,
        "thread_id",
        "threadId",
        "session_id",
      );
      const turnId = readString(payload, "turn_id", "turnId");
      const itemId = readString(payload, "item_id", "itemId");
      const text = readString(payload, "text", "delta");
      return threadId && turnId && itemId && text !== undefined
        ? {
            ...base,
            type: "item_delta",
            thread_id: threadId,
            turn_id: turnId,
            item_id: itemId,
            sequence: readFiniteNumber(payload, "sequence") ?? 0,
            delta: { kind: "text", value: text },
          }
        : null;
    }
    case "plan_delta": {
      const threadId = readString(
        payload,
        "thread_id",
        "threadId",
        "session_id",
      );
      const turnId = readString(payload, "turn_id", "turnId");
      const itemId = readString(payload, "source_item_id", "sourceItemId");
      const text = readString(payload, "delta", "text");
      return threadId && turnId && itemId && text !== undefined
        ? {
            ...base,
            type: "item_delta",
            thread_id: threadId,
            turn_id: turnId,
            item_id: itemId,
            sequence: readFiniteNumber(payload, "sequence") ?? 0,
            delta: { kind: "text", value: text },
          }
        : null;
    }
    case "tool_output_delta": {
      const threadId = readString(
        payload,
        "thread_id",
        "threadId",
        "session_id",
      );
      const turnId = readString(payload, "turn_id", "turnId");
      const itemId = readString(
        payload,
        "source_item_id",
        "sourceItemId",
        "tool_id",
        "toolId",
      );
      const delta = readString(payload, "delta", "text");
      return threadId && turnId && itemId && delta !== undefined
        ? {
            ...base,
            type: "item_delta",
            thread_id: threadId,
            turn_id: turnId,
            item_id: itemId,
            sequence: readFiniteNumber(payload, "sequence") ?? 0,
            delta: { kind: "output", value: delta },
          }
        : null;
    }
    case "reasoning_summary_delta":
    case "reasoning_content_delta": {
      const threadId = readString(
        payload,
        "thread_id",
        "threadId",
        "session_id",
      );
      const turnId = readString(payload, "turn_id", "turnId");
      const itemId = readString(
        payload,
        "item_id",
        "itemId",
        "reasoning_id",
        "reasoningId",
      );
      const value = readString(payload, "delta", "text");
      const index = readFiniteNumber(
        payload,
        type === "reasoning_summary_delta" ? "summary_index" : "content_index",
        type === "reasoning_summary_delta" ? "summaryIndex" : "contentIndex",
      );
      return threadId &&
        turnId &&
        itemId &&
        value !== undefined &&
        index !== undefined
        ? {
            ...base,
            type: "item_delta",
            thread_id: threadId,
            turn_id: turnId,
            item_id: itemId,
            sequence: readFiniteNumber(payload, "sequence") ?? 0,
            delta: {
              kind:
                type === "reasoning_summary_delta"
                  ? "reasoning_summary"
                  : "reasoning_content",
              index,
              value,
            },
          }
        : null;
    }
    case "tool_progress": {
      const threadId = readString(
        payload,
        "thread_id",
        "threadId",
        "session_id",
      );
      const turnId = readString(payload, "turn_id", "turnId");
      const itemId = readString(
        payload,
        "tool_id",
        "toolId",
        "item_id",
        "itemId",
      );
      const progress = readRecord(payload.progress);
      return threadId && turnId && itemId && progress
        ? {
            ...base,
            type: "item_delta",
            thread_id: threadId,
            turn_id: turnId,
            item_id: itemId,
            sequence: readFiniteNumber(payload, "sequence") ?? 0,
            delta: {
              kind: "tool_progress",
              message: readString(progress, "message"),
              progress: readFiniteNumber(progress, "progress"),
              total: readFiniteNumber(progress, "total"),
              metadata: readRecord(progress.metadata) ?? undefined,
            },
          }
        : null;
    }
    case "transport_disconnected": {
      const threadId = readString(
        payload,
        "thread_id",
        "threadId",
        "session_id",
      );
      return threadId
        ? {
            ...base,
            type,
            thread_id: threadId,
            reason: readString(payload, "reason"),
          }
        : null;
    }
    default:
      return null;
  }
}

export function reduceConversationProjectionPayloads(
  payloads: readonly Record<string, unknown>[],
  source: ConversationProjectionSource,
  threadId?: string,
) {
  const reducer = createConversationProjectionReducer({ threadId });
  payloads.forEach((payload, index) => {
    const event = conversationProjectionEventFromPayload(
      payload,
      source,
      readString(payload, "event_id", "eventId") ?? `${source}:${index}`,
    );
    if (event) reducer.dispatch(event);
  });
  return reducer;
}

function sourceMethod(source: ConversationProjectionSource): string {
  switch (source) {
    case "read":
      return "thread/read";
    case "replay":
      return "replay";
    case "live":
      return "live";
  }
}

function isAgentThreadTurn(value: Record<string, unknown> | null): boolean {
  return Boolean(
    value &&
    typeof value.id === "string" &&
    typeof value.thread_id === "string" &&
    typeof value.status === "string" &&
    typeof value.started_at === "string" &&
    typeof value.created_at === "string" &&
    typeof value.updated_at === "string",
  );
}

function isAgentThreadItem(value: Record<string, unknown> | null): boolean {
  return Boolean(
    value &&
    typeof value.id === "string" &&
    typeof value.thread_id === "string" &&
    typeof value.turn_id === "string" &&
    typeof value.type === "string" &&
    typeof value.status === "string" &&
    typeof value.sequence === "number" &&
    typeof value.started_at === "string" &&
    typeof value.updated_at === "string",
  );
}

function readRecord(value: unknown): Record<string, unknown> | null {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function readString(
  record: Record<string, unknown>,
  ...keys: string[]
): string | undefined {
  for (const key of keys) {
    if (typeof record[key] === "string") return record[key] as string;
  }
  return undefined;
}

function readFiniteNumber(
  record: Record<string, unknown>,
  ...keys: string[]
): number | undefined {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return undefined;
}
