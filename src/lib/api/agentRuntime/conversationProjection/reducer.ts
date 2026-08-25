import type {
  AgentThreadItem,
  AgentThreadTurn,
  AgentThreadTurnStatus,
} from "../../agentProtocol";
import type {
  ConversationProjection,
  ConversationProjectionDiagnostic,
  ConversationProjectionEvent,
  ConversationProjectionReducer,
  ConversationProjectionSource,
  ConversationProjectionState,
  ConversationProjectionStatus,
  GuardianReviewProjection,
  ItemProjectionDelta,
  NoticeProjection,
  PendingInteractionProjection,
  TurnProjection,
} from "./contracts";
import { buildUnknownItemProtocolDriftDiagnostic } from "./protocolDrift";
import { mergeProjectionOutput, sanitizeProjectionItem } from "./sanitizer";

const ITEM_KEY_SEPARATOR = "\u001f";
const MAX_ORPHAN_DELTAS = 32;
const MAX_DIAGNOSTICS = 200;
const MAX_APPLIED_EVENT_IDS = 2_000;

export function conversationItemKey(
  threadId: string,
  turnId: string,
  itemId: string,
): string {
  return [threadId, turnId, itemId].join(ITEM_KEY_SEPARATOR);
}

export function createInitialConversationProjectionState(
  threadId: string | null = null,
): ConversationProjectionState {
  return {
    thread_id: threadId,
    status: "idle",
    turns: {},
    turn_order: [],
    items: {},
    item_order_by_turn: {},
    pending_interactions: {},
    strict_reviews: {},
    notices: [],
    orphan_deltas: {},
    diagnostics: [],
    applied_event_ids: {},
  };
}

export function createConversationProjectionReducer(
  options: {
    threadId?: string | null;
  } = {},
): ConversationProjectionReducer {
  let state = createInitialConversationProjectionState(
    options.threadId ?? null,
  );
  return {
    dispatch(event) {
      state = reduceConversationProjection(state, event);
      return state;
    },
    getState: () => state,
    getProjection: () => selectConversationProjection(state),
    reset() {
      state = createInitialConversationProjectionState(
        options.threadId ?? null,
      );
      return state;
    },
  };
}

export function reduceConversationProjection(
  state: ConversationProjectionState,
  event: ConversationProjectionEvent,
): ConversationProjectionState {
  if (event.event_id && state.applied_event_ids[event.event_id]) {
    return state;
  }

  let next = markEventApplied(state, event.event_id);
  const eventThreadId = eventThreadIdOf(event);
  if (eventThreadId && next.thread_id && eventThreadId !== next.thread_id) {
    return addDiagnostic(next, {
      code: "thread_mismatch",
      source: event.source,
      event_id: event.event_id,
      thread_id: eventThreadId,
      message: `Ignored event for ${eventThreadId}; reducer owns ${next.thread_id}.`,
    });
  }
  if (!next.thread_id && eventThreadId) {
    next = { ...next, thread_id: eventThreadId };
  }

  switch (event.type) {
    case "thread_started":
      return {
        ...next,
        thread_id: event.thread_id,
        status: event.status ?? "running",
      };
    case "turn_started":
      return upsertTurn(next, event.turn, event.source, event.event_id);
    case "turn_completed":
      return clearStrictReview(
        upsertTurn(
          { ...next, status: statusFromTurn(event.turn.status) },
          event.turn,
          event.source,
          event.event_id,
        ),
        event.turn.id,
      );
    case "turn_diff_updated":
      return applyTurnDiffUpdate(next, event);
    case "turn_moderation_metadata":
      return applyTurnModerationMetadata(next, event);
    case "guardian_warning":
      return addNotice(next, {
        id:
          event.event_id ??
          `guardian-warning:${event.thread_id}:${event.message}`,
        thread_id: event.thread_id,
        level: "warning",
        code: "guardian_warning",
        message: event.message,
      });
    case "strict_review_required":
      return applyStrictReviewRequired(next, event);
    case "guardian_review_started":
      return applyGuardianReviewStarted(next, event);
    case "guardian_review_completed":
      return clearStrictReview(
        applyGuardianReviewCompleted(next, event),
        event.turn_id,
      );
    case "item_started":
    case "item_updated":
    case "item_completed":
      return applyItemSnapshot(next, event);
    case "item_delta":
      return applyItemDeltaEvent(next, event);
    case "server_request_resolved":
      return resolvePendingInteraction(next, event);
    case "transport_disconnected":
      return { ...next, status: "disconnected" };
  }
}

export function selectConversationProjection(
  state: ConversationProjectionState,
): ConversationProjection {
  const turns = state.turn_order
    .map((turnId) => state.turns[turnId])
    .filter((turn): turn is TurnProjection => Boolean(turn));
  const items = turns.flatMap((turn) =>
    (state.item_order_by_turn[turn.id] ?? [])
      .map((key) => state.items[key])
      .filter((item): item is AgentThreadItem => Boolean(item)),
  );
  return {
    thread_id: state.thread_id,
    status: state.status,
    turns,
    items,
    pending_interactions: Object.values(state.pending_interactions),
    strict_reviews: Object.values(state.strict_reviews),
    notices: state.notices,
    diagnostics: state.diagnostics,
  };
}

function applyItemSnapshot(
  state: ConversationProjectionState,
  event: Extract<
    ConversationProjectionEvent,
    { type: "item_started" | "item_updated" | "item_completed" }
  >,
): ConversationProjectionState {
  const snapshot = sanitizeProjectionItem(event.item);
  const key = conversationItemKey(
    snapshot.thread_id,
    snapshot.turn_id,
    snapshot.id,
  );
  const existing = state.items[key];
  const item = existing
    ? {
        ...snapshot,
        sequence: existing.sequence,
        started_at: existing.started_at,
        ...(snapshot.type === "command_execution" &&
        existing.type === "command_execution" &&
        !snapshot.terminal_interactions?.length &&
        existing.terminal_interactions?.length
          ? { terminal_interactions: existing.terminal_interactions }
          : {}),
      }
    : snapshot;
  if (
    event.type === "item_started" &&
    existing &&
    isTerminal(existing.status)
  ) {
    return addDiagnostic(state, {
      code: "item_started_after_terminal",
      source: event.source,
      event_id: event.event_id,
      thread_id: item.thread_id,
      turn_id: item.turn_id,
      item_id: item.id,
      message: `Ignored item/started after terminal item ${item.id}.`,
    });
  }

  let next = ensureTurnForItem(state, item, event.source, event.event_id);
  const itemOrder = [...(next.item_order_by_turn[item.turn_id] ?? [])];
  if (!itemOrder.includes(key)) {
    itemOrder.push(key);
  }
  const items = { ...next.items, [key]: item };
  itemOrder.sort((left, right) => compareItems(items[left], items[right]));

  let diagnostics = next.diagnostics;
  if (event.type === "item_updated" && !existing) {
    diagnostics = appendDiagnostic(diagnostics, {
      code: "item_update_before_start",
      source: event.source,
      event_id: event.event_id,
      thread_id: item.thread_id,
      turn_id: item.turn_id,
      item_id: item.id,
      message: `Accepted item/update as the first snapshot for ${item.id}.`,
    });
  }
  if (
    item.type === "unknown_item" &&
    !diagnostics.some(
      (diagnostic) =>
        diagnostic.code === "protocol_drift" &&
        diagnostic.thread_id === item.thread_id &&
        diagnostic.turn_id === item.turn_id &&
        diagnostic.item_id === item.id &&
        diagnostic.upstream_type === item.upstream_type,
    )
  ) {
    diagnostics = appendDiagnostic(
      diagnostics,
      buildUnknownItemProtocolDriftDiagnostic({
        eventId: event.event_id,
        item,
        method: event.protocol_method,
        protocolRevision: event.protocol_revision,
        source: event.source,
      }),
    );
  }

  const orphanDeltas = next.orphan_deltas[key] ?? [];
  if (event.type === "item_started" && orphanDeltas.length > 0) {
    let hydrated = item;
    for (const orphan of orphanDeltas) {
      if (orphan.type !== "item_delta") continue;
      const applied = applyItemDelta(hydrated, orphan.delta);
      hydrated = applied.item;
      if (applied.unsupported) {
        diagnostics = appendDiagnostic(diagnostics, {
          code: "unsupported_delta",
          source: orphan.source,
          event_id: orphan.event_id,
          thread_id: orphan.thread_id,
          turn_id: orphan.turn_id,
          item_id: orphan.item_id,
          message: applied.unsupported,
        });
      }
    }
    items[key] = hydrated;
  } else if (event.type === "item_completed" && orphanDeltas.length > 0) {
    diagnostics = appendDiagnostic(diagnostics, {
      code: "orphan_delta_discarded",
      source: event.source,
      event_id: event.event_id,
      thread_id: item.thread_id,
      turn_id: item.turn_id,
      item_id: item.id,
      message: `Discarded ${orphanDeltas.length} pre-start delta(s); completed snapshot is authoritative.`,
    });
  }

  const nextOrphans = { ...next.orphan_deltas };
  delete nextOrphans[key];
  const updatedTurn = next.turns[item.turn_id];
  return {
    ...next,
    status: next.status === "disconnected" ? "running" : next.status,
    turns: updatedTurn
      ? {
          ...next.turns,
          [item.turn_id]: {
            ...updatedTurn,
            item_ids: itemOrder.map((entry) =>
              entry === key ? item.id : (items[entry]?.id ?? entry),
            ),
          },
        }
      : next.turns,
    items,
    item_order_by_turn: {
      ...next.item_order_by_turn,
      [item.turn_id]: itemOrder,
    },
    orphan_deltas: nextOrphans,
    diagnostics,
  };
}

function applyItemDeltaEvent(
  state: ConversationProjectionState,
  event: Extract<ConversationProjectionEvent, { type: "item_delta" }>,
): ConversationProjectionState {
  const key = conversationItemKey(
    event.thread_id,
    event.turn_id,
    event.item_id,
  );
  const item = state.items[key];
  if (!item) {
    const orphanDeltas = [...(state.orphan_deltas[key] ?? []), event].slice(
      -MAX_ORPHAN_DELTAS,
    );
    return addDiagnostic(
      {
        ...state,
        orphan_deltas: { ...state.orphan_deltas, [key]: orphanDeltas },
      },
      {
        code: "item_delta_before_start",
        source: event.source,
        event_id: event.event_id,
        thread_id: event.thread_id,
        turn_id: event.turn_id,
        item_id: event.item_id,
        message: `Buffered delta for item ${event.item_id} until its first snapshot.`,
      },
    );
  }
  if (isTerminal(item.status)) {
    return addDiagnostic(state, {
      code: "late_delta_rejected",
      source: event.source,
      event_id: event.event_id,
      thread_id: event.thread_id,
      turn_id: event.turn_id,
      item_id: event.item_id,
      message: `Rejected late delta after terminal item ${event.item_id}.`,
    });
  }

  const applied = applyItemDelta(item, event.delta);
  if (applied.unsupported) {
    return addDiagnostic(state, {
      code: "unsupported_delta",
      source: event.source,
      event_id: event.event_id,
      thread_id: event.thread_id,
      turn_id: event.turn_id,
      item_id: event.item_id,
      message: applied.unsupported,
    });
  }
  return {
    ...state,
    items: { ...state.items, [key]: applied.item },
  };
}

function applyItemDelta(
  item: AgentThreadItem,
  delta: ItemProjectionDelta,
): { item: AgentThreadItem; unsupported?: string } {
  switch (delta.kind) {
    case "text":
      if (item.type !== "agent_message" && item.type !== "plan") {
        return {
          item,
          unsupported: `Text delta is not supported for ${item.type}.`,
        };
      }
      return {
        item: {
          ...item,
          text:
            delta.mode === "replace"
              ? delta.value
              : `${item.text}${delta.value}`,
          updated_at: item.updated_at,
        },
      } as { item: AgentThreadItem };
    case "output":
      if (item.type === "command_execution") {
        return {
          item: {
            ...item,
            aggregated_output: mergeProjectionOutput(
              item.aggregated_output,
              delta.value,
              delta.mode,
            ),
          },
        };
      }
      if (item.type !== "tool_call" && item.type !== "web_search") {
        return {
          item,
          unsupported: `Output delta is not supported for ${item.type}.`,
        };
      }
      return {
        item: {
          ...item,
          output: mergeProjectionOutput(item.output, delta.value, delta.mode),
        },
      } as { item: AgentThreadItem };
    case "aggregated_output":
      if (item.type !== "command_execution") {
        return {
          item,
          unsupported: `Aggregated output delta is not supported for ${item.type}.`,
        };
      }
      return {
        item: {
          ...item,
          aggregated_output: mergeProjectionOutput(
            item.aggregated_output,
            delta.value,
            delta.mode,
          ),
        },
      };
    case "terminal_interaction":
      if (item.type !== "command_execution") {
        return {
          item,
          unsupported: `Terminal interaction is not supported for ${item.type}.`,
        };
      }
      return {
        item: {
          ...item,
          terminal_interactions: [
            ...(item.terminal_interactions ?? []),
            { process_id: delta.process_id, stdin: delta.stdin },
          ].slice(-20),
        },
      };
    case "reasoning_summary":
      if (item.type !== "reasoning") {
        return {
          item,
          unsupported: `Reasoning summary delta is not supported for ${item.type}.`,
        };
      }
      return {
        item: replaceReasoningPart(item, "summary", delta.index, delta.value),
      };
    case "reasoning_content":
      if (item.type !== "reasoning") {
        return {
          item,
          unsupported: `Reasoning content delta is not supported for ${item.type}.`,
        };
      }
      return {
        item: replaceReasoningPart(item, "content", delta.index, delta.value),
      };
    case "patch":
      if (item.type !== "patch") {
        return {
          item,
          unsupported: `Patch delta is not supported for ${item.type}.`,
        };
      }
      return {
        item: {
          ...item,
          changes: [...delta.changes] as typeof item.changes,
          text: delta.text ?? JSON.stringify(delta.changes),
        },
      };
    case "tool_progress":
      if (item.type !== "tool_call") {
        return {
          item,
          unsupported: `Tool progress is not supported for ${item.type}.`,
        };
      }
      return {
        item: {
          ...item,
          metadata: {
            ...(isRecord(item.metadata) ? item.metadata : {}),
            progress: {
              ...(delta.message ? { message: delta.message } : {}),
              ...(delta.progress !== undefined
                ? { progress: delta.progress }
                : {}),
              ...(delta.total !== undefined ? { total: delta.total } : {}),
              ...(delta.metadata ?? {}),
            },
          },
        },
      };
  }
}

function replaceReasoningPart(
  item: Extract<AgentThreadItem, { type: "reasoning" }>,
  field: "summary" | "content",
  index: number,
  value: string,
): AgentThreadItem {
  const values = [...(item[field] ?? [])];
  values[index] = value;
  return {
    ...item,
    [field]: values,
    text: field === "summary" ? values.filter(Boolean).join("\n\n") : item.text,
  };
}

function upsertTurn(
  state: ConversationProjectionState,
  turn: AgentThreadTurn,
  source: ConversationProjectionSource,
  eventId?: string,
): ConversationProjectionState {
  if (state.thread_id && state.thread_id !== turn.thread_id) {
    return addDiagnostic(state, {
      code: "turn_mismatch",
      source,
      event_id: eventId,
      thread_id: turn.thread_id,
      turn_id: turn.id,
      message: `Ignored turn ${turn.id} outside thread ${state.thread_id}.`,
    });
  }
  const existing = state.turns[turn.id];
  const itemIds = existing?.item_ids ?? [];
  const projected: TurnProjection = {
    ...turn,
    ...(turn.unified_diff === undefined && existing?.unified_diff !== undefined
      ? { unified_diff: existing.unified_diff }
      : {}),
    ...(turn.moderation_metadata === undefined &&
    existing?.moderation_metadata !== undefined
      ? { moderation_metadata: existing.moderation_metadata }
      : {}),
    item_ids: itemIds,
  };
  return {
    ...state,
    thread_id: state.thread_id ?? turn.thread_id,
    status: statusFromTurn(turn.status),
    turns: { ...state.turns, [turn.id]: projected },
    turn_order: state.turn_order.includes(turn.id)
      ? state.turn_order
      : [...state.turn_order, turn.id],
  };
}

function applyTurnDiffUpdate(
  state: ConversationProjectionState,
  event: Extract<ConversationProjectionEvent, { type: "turn_diff_updated" }>,
): ConversationProjectionState {
  const turn = state.turns[event.turn_id];
  if (!turn || turn.thread_id !== event.thread_id) {
    return addDiagnostic(state, {
      code: "turn_mismatch",
      source: event.source,
      event_id: event.event_id,
      thread_id: event.thread_id,
      turn_id: event.turn_id,
      message: `Ignored diff update for unknown turn ${event.turn_id}.`,
    });
  }
  if (turn.unified_diff === event.unified_diff) {
    return state;
  }
  return {
    ...state,
    turns: {
      ...state.turns,
      [event.turn_id]: { ...turn, unified_diff: event.unified_diff },
    },
  };
}

function applyTurnModerationMetadata(
  state: ConversationProjectionState,
  event: Extract<
    ConversationProjectionEvent,
    { type: "turn_moderation_metadata" }
  >,
): ConversationProjectionState {
  const turn = state.turns[event.turn_id];
  if (!turn || turn.thread_id !== event.thread_id) {
    return addDiagnostic(state, {
      code: "turn_mismatch",
      source: event.source,
      event_id: event.event_id,
      thread_id: event.thread_id,
      turn_id: event.turn_id,
      message: `Ignored moderation metadata for unknown turn ${event.turn_id}.`,
    });
  }
  if (turn.moderation_metadata === event.moderation_metadata) {
    return state;
  }
  return {
    ...state,
    turns: {
      ...state.turns,
      [event.turn_id]: {
        ...turn,
        moderation_metadata: event.moderation_metadata,
      },
    },
  };
}

function applyGuardianReviewStarted(
  state: ConversationProjectionState,
  event: Extract<
    ConversationProjectionEvent,
    { type: "guardian_review_started" }
  >,
): ConversationProjectionState {
  const existing = state.pending_interactions[event.review_id];
  if (existing && existing.status !== "pending") {
    return addDiagnostic(state, {
      code: "protocol_drift",
      source: event.source,
      event_id: event.event_id,
      thread_id: event.thread_id,
      turn_id: event.turn_id,
      message: `Ignored Guardian review ${event.review_id} after terminal state.`,
    });
  }
  return {
    ...state,
    pending_interactions: {
      ...state.pending_interactions,
      [event.review_id]: {
        id: event.review_id,
        thread_id: event.thread_id,
        turn_id: event.turn_id,
        ...(event.target_item_id ? { item_id: event.target_item_id } : {}),
        kind: "guardian_review",
        status: "pending",
        payload: {
          action: event.action,
          review: event.review,
        },
      },
    },
  };
}

function applyStrictReviewRequired(
  state: ConversationProjectionState,
  event: Extract<
    ConversationProjectionEvent,
    { type: "strict_review_required" }
  >,
): ConversationProjectionState {
  return {
    ...state,
    strict_reviews: {
      ...state.strict_reviews,
      [event.turn_id]: {
        thread_id: event.thread_id,
        turn_id: event.turn_id,
        started_at_ms: event.started_at_ms,
      },
    },
  };
}

function clearStrictReview(
  state: ConversationProjectionState,
  turnId: string,
): ConversationProjectionState {
  if (!state.strict_reviews[turnId]) {
    return state;
  }
  const { [turnId]: _removed, ...strictReviews } = state.strict_reviews;
  return { ...state, strict_reviews: strictReviews };
}

function applyGuardianReviewCompleted(
  state: ConversationProjectionState,
  event: Extract<
    ConversationProjectionEvent,
    { type: "guardian_review_completed" }
  >,
): ConversationProjectionState {
  const existing = state.pending_interactions[event.review_id];
  if (!existing) {
    return addDiagnostic(state, {
      code: "guardian_review_completed_before_started",
      source: event.source,
      event_id: event.event_id,
      thread_id: event.thread_id,
      turn_id: event.turn_id,
      message: `Ignored Guardian completion ${event.review_id} before its start.`,
    });
  }
  const status = guardianInteractionStatus(event.review.status);
  if (!status) {
    return addDiagnostic(state, {
      code: "protocol_drift",
      source: event.source,
      event_id: event.event_id,
      thread_id: event.thread_id,
      turn_id: event.turn_id,
      message: `Ignored unsupported Guardian status for ${event.review_id}.`,
    });
  }
  return {
    ...state,
    pending_interactions: {
      ...state.pending_interactions,
      [event.review_id]: {
        ...existing,
        thread_id: event.thread_id,
        turn_id: event.turn_id,
        ...(event.target_item_id ? { item_id: event.target_item_id } : {}),
        status,
        payload: {
          ...(isRecord(existing.payload) ? existing.payload : {}),
          action: event.action,
          review: event.review,
          decision_source: event.decision_source,
        },
      },
    },
  };
}

function guardianInteractionStatus(
  status: GuardianReviewProjection["status"],
): PendingInteractionProjection["status"] | null {
  switch (status) {
    case "approved":
      return "resolved";
    case "denied":
      return "declined";
    case "timedOut":
    case "aborted":
      return "cancelled";
    default:
      return null;
  }
}

function ensureTurnForItem(
  state: ConversationProjectionState,
  item: AgentThreadItem,
  source: ConversationProjectionSource,
  eventId?: string,
): ConversationProjectionState {
  if (state.turns[item.turn_id]) {
    return state;
  }
  return upsertTurn(
    state,
    {
      id: item.turn_id,
      thread_id: item.thread_id,
      prompt_text: "",
      status: item.status === "in_progress" ? "running" : "completed",
      started_at: item.started_at,
      completed_at: item.completed_at,
      created_at: item.started_at,
      updated_at: item.updated_at,
    },
    source,
    eventId,
  );
}

function resolvePendingInteraction(
  state: ConversationProjectionState,
  event: Extract<
    ConversationProjectionEvent,
    { type: "server_request_resolved" }
  >,
): ConversationProjectionState {
  const existing = state.pending_interactions[event.interaction_id];
  if (!existing) {
    return state;
  }
  return {
    ...state,
    pending_interactions: {
      ...state.pending_interactions,
      [event.interaction_id]: { ...existing, status: event.status },
    },
  };
}

function markEventApplied(
  state: ConversationProjectionState,
  eventId: string | undefined,
): ConversationProjectionState {
  if (!eventId) return state;
  const appliedEventIds: Record<string, true> = {
    ...state.applied_event_ids,
    [eventId]: true,
  };
  const entries = Object.entries(appliedEventIds);
  const bounded = entries.slice(-MAX_APPLIED_EVENT_IDS);
  return {
    ...state,
    applied_event_ids: Object.fromEntries(bounded) as Record<string, true>,
  };
}

function addDiagnostic(
  state: ConversationProjectionState,
  diagnostic: ConversationProjectionDiagnostic,
): ConversationProjectionState {
  return {
    ...state,
    diagnostics: appendDiagnostic(state.diagnostics, diagnostic),
  };
}

function appendDiagnostic(
  diagnostics: readonly ConversationProjectionDiagnostic[],
  diagnostic: ConversationProjectionDiagnostic,
): ConversationProjectionDiagnostic[] {
  return [...diagnostics, diagnostic].slice(-MAX_DIAGNOSTICS);
}

function eventThreadIdOf(
  event: ConversationProjectionEvent,
): string | undefined {
  switch (event.type) {
    case "turn_started":
    case "turn_completed":
      return event.turn.thread_id;
    case "turn_diff_updated":
    case "turn_moderation_metadata":
    case "guardian_warning":
    case "strict_review_required":
      return event.thread_id;
    case "guardian_review_started":
    case "guardian_review_completed":
      return event.thread_id;
    case "item_started":
    case "item_updated":
    case "item_completed":
      return event.item.thread_id;
    case "thread_started":
    case "item_delta":
    case "server_request_resolved":
    case "transport_disconnected":
      return event.thread_id;
  }
}

function statusFromTurn(
  status: AgentThreadTurnStatus,
): ConversationProjectionStatus {
  switch (status) {
    case "running":
      return "running";
    case "failed":
      return "failed";
    case "interrupted":
    case "cancelled":
    case "canceled":
    case "aborted":
      return "interrupted";
    case "completed":
      return "completed";
    default:
      return "idle";
  }
}

function isTerminal(status: AgentThreadItem["status"]): boolean {
  return status === "completed" || status === "failed";
}

function compareItems(
  left: AgentThreadItem | undefined,
  right: AgentThreadItem | undefined,
): number {
  if (!left || !right) return 0;
  return left.sequence - right.sequence || left.id.localeCompare(right.id);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

export function addPendingInteraction(
  state: ConversationProjectionState,
  interaction: PendingInteractionProjection,
): ConversationProjectionState {
  return {
    ...state,
    pending_interactions: {
      ...state.pending_interactions,
      [interaction.id]: interaction,
    },
  };
}

export function addNotice(
  state: ConversationProjectionState,
  notice: NoticeProjection,
): ConversationProjectionState {
  return { ...state, notices: [...state.notices, notice] };
}
