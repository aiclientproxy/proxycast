import type { AgentEvent, AgentThreadItem } from "@/lib/api/agentProtocol";
import {
  conversationProjectionEventFromPayload,
  createConversationProjectionReducer,
  type ConversationProjection,
  type ConversationProjectionEvent,
  type ConversationProjectionReducer,
} from "@/lib/api/agentRuntime/conversationProjection";
import { recordConversationStreamDiagnostic } from "../projection/conversationProjectionStore";
import {
  removeThreadItemState,
  upsertThreadItemState,
} from "./agentThreadState";

export interface AgentStreamConversationProjectionOwner {
  reducer: ConversationProjectionReducer;
  nextEventOrdinal: number;
}

export interface AgentStreamConversationProjectionHost {
  conversationProjectionOwner?: AgentStreamConversationProjectionOwner;
}

export interface AgentStreamConversationProjectionUpdate {
  event: ConversationProjectionEvent;
  projection: ConversationProjection;
}

export function applyAgentStreamConversationProjection(params: {
  event: AgentEvent;
  existingItems: readonly AgentThreadItem[];
  host: AgentStreamConversationProjectionHost;
  threadId: string;
}): AgentStreamConversationProjectionUpdate | null {
  const owner =
    params.host.conversationProjectionOwner ??
    createOwner(params.existingItems);

  const payload = params.event as unknown as Record<string, unknown>;
  const eventId =
    readString(payload, "event_id", "eventId") ??
    `live:${owner.nextEventOrdinal}`;
  const event = conversationProjectionEventFromPayload(
    payload,
    "live",
    eventId,
  );
  if (!event || event.protocol_method === "live") {
    return null;
  }
  params.host.conversationProjectionOwner = owner;
  owner.nextEventOrdinal += 1;

  const previousDiagnostics = new Set(
    owner.reducer.getProjection().diagnostics,
  );
  owner.reducer.dispatch(event);
  const projection = owner.reducer.getProjection();
  for (const diagnostic of projection.diagnostics) {
    if (previousDiagnostics.has(diagnostic)) {
      continue;
    }
    if (diagnostic.code !== "protocol_drift") {
      continue;
    }
    recordConversationStreamDiagnostic({
      phase: "protocol_drift",
      at: Date.now(),
      wallTime: Date.now(),
      sessionId: diagnostic.thread_id ?? params.threadId,
      source: "conversation_projection",
      requestId: diagnostic.event_id,
      metrics: {
        fieldNames: diagnostic.field_names?.join(",") ?? "",
        itemId: diagnostic.item_id ?? "",
        method: diagnostic.protocol_method ?? "unknown",
        protocolRevision: diagnostic.protocol_revision ?? "unknown",
        source: diagnostic.source,
        upstreamType: diagnostic.upstream_type ?? "unknown",
      },
    });
  }

  return { event, projection };
}

export function reconcileAgentStreamProjectionItems(params: {
  current: AgentThreadItem[];
  pendingItemKey: string;
  projected: readonly AgentThreadItem[];
}): AgentThreadItem[] {
  let next = removeThreadItemState(params.current, params.pendingItemKey);
  for (const item of params.projected) {
    next = upsertThreadItemState(next, item);
  }
  return next;
}

function createOwner(
  existingItems: readonly AgentThreadItem[],
): AgentStreamConversationProjectionOwner {
  const reducer = createConversationProjectionReducer();
  for (const item of existingItems) {
    if (item.id.startsWith("pending-item:")) {
      continue;
    }
    reducer.dispatch({
      type: item.status === "in_progress" ? "item_started" : "item_completed",
      source: "live",
      event_id: `live-seed:${item.thread_id}:${item.turn_id}:${item.id}`,
      protocol_method: "state/seed",
      item,
    });
  }
  return { reducer, nextEventOrdinal: 1 };
}

function readString(
  record: Record<string, unknown>,
  ...keys: string[]
): string | undefined {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "string" && value.trim()) {
      return value;
    }
  }
  return undefined;
}
