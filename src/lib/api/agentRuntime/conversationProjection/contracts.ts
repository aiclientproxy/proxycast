import type {
  AgentThreadItem,
  AgentThreadTurn,
  AgentThreadTurnStatus,
} from "../../agentProtocol";

export type ConversationProjectionSource = "live" | "read" | "replay";

export type ConversationProjectionStatus =
  | "idle"
  | "running"
  | "completed"
  | "failed"
  | "interrupted"
  | "disconnected";

export interface TurnProjection extends AgentThreadTurn {
  item_ids: readonly string[];
}

export interface PendingInteractionProjection {
  id: string;
  thread_id: string;
  turn_id?: string;
  item_id?: string;
  kind: string;
  status: "pending" | "resolved" | "declined" | "cancelled";
  payload?: unknown;
}

export interface GuardianReviewProjection {
  status: "inProgress" | "approved" | "denied" | "timedOut" | "aborted";
  risk_level?: "low" | "medium" | "high" | "critical" | null;
  user_authorization?: "unknown" | "low" | "medium" | "high" | null;
  rationale?: string | null;
}

export type GuardianReviewActionProjection = Readonly<Record<string, unknown>>;

export interface NoticeProjection {
  id: string;
  thread_id: string;
  turn_id?: string;
  level: "info" | "warning" | "error";
  code: string;
  message: string;
}

export interface StrictReviewProjection {
  thread_id: string;
  turn_id: string;
  started_at_ms: number;
}

export interface ConversationProjectionDiagnostic {
  code:
    | "thread_mismatch"
    | "turn_mismatch"
    | "duplicate_event"
    | "item_update_before_start"
    | "item_started_after_terminal"
    | "item_delta_before_start"
    | "late_delta_rejected"
    | "orphan_delta_discarded"
    | "guardian_review_completed_before_started"
    | "unsupported_delta"
    | "turn_terminal_late_event"
    | "protocol_drift";
  source: ConversationProjectionSource;
  message: string;
  event_id?: string;
  thread_id?: string;
  turn_id?: string;
  item_id?: string;
  protocol_revision?: string;
  protocol_method?: string;
  upstream_type?: string;
  field_names?: readonly string[];
}

export type ItemProjectionDelta =
  | { kind: "text"; value: string; mode?: "append" | "replace" }
  | { kind: "output"; value: string; mode?: "append" | "replace" }
  | {
      kind: "aggregated_output";
      value: string;
      mode?: "append" | "replace";
    }
  | {
      kind: "terminal_interaction";
      process_id: string;
      stdin: string;
    }
  | { kind: "reasoning_summary"; index: number; value: string }
  | { kind: "reasoning_content"; index: number; value: string }
  | { kind: "patch"; changes: readonly unknown[]; text?: string }
  | {
      kind: "tool_progress";
      message?: string;
      progress?: number;
      total?: number;
      metadata?: Record<string, unknown>;
    };

interface ProjectionEventBase {
  source: ConversationProjectionSource;
  event_id?: string;
  protocol_revision?: string;
  protocol_method?: string;
}

export type ConversationProjectionEvent =
  | (ProjectionEventBase & {
      type: "thread_started";
      thread_id: string;
      status?: ConversationProjectionStatus;
    })
  | (ProjectionEventBase & {
      type: "turn_started" | "turn_completed";
      turn: AgentThreadTurn;
    })
  | (ProjectionEventBase & {
      type: "turn_diff_updated";
      thread_id: string;
      turn_id: string;
      unified_diff: string;
    })
  | (ProjectionEventBase & {
      type: "turn_moderation_metadata";
      thread_id: string;
      turn_id: string;
      moderation_metadata: unknown;
    })
  | (ProjectionEventBase & {
      type: "guardian_warning";
      thread_id: string;
      message: string;
    })
  | (ProjectionEventBase & {
      type: "strict_review_required";
      thread_id: string;
      turn_id: string;
      started_at_ms: number;
    })
  | (ProjectionEventBase & {
      type: "guardian_review_started";
      thread_id: string;
      turn_id: string;
      review_id: string;
      target_item_id?: string;
      review: GuardianReviewProjection;
      action: GuardianReviewActionProjection;
    })
  | (ProjectionEventBase & {
      type: "guardian_review_completed";
      thread_id: string;
      turn_id: string;
      review_id: string;
      target_item_id?: string;
      decision_source: "agent";
      review: GuardianReviewProjection;
      action: GuardianReviewActionProjection;
    })
  | (ProjectionEventBase & {
      type: "item_started" | "item_updated" | "item_completed";
      item: AgentThreadItem;
    })
  | (ProjectionEventBase & {
      type: "item_delta";
      thread_id: string;
      turn_id: string;
      item_id: string;
      sequence: number;
      delta: ItemProjectionDelta;
    })
  | (ProjectionEventBase & {
      type: "server_request_resolved";
      thread_id: string;
      interaction_id: string;
      status: "resolved" | "declined" | "cancelled";
    })
  | (ProjectionEventBase & {
      type: "transport_disconnected";
      thread_id: string;
      reason?: string;
    });

export interface ConversationProjectionState {
  thread_id: string | null;
  status: ConversationProjectionStatus;
  turns: Readonly<Record<string, TurnProjection>>;
  turn_order: readonly string[];
  items: Readonly<Record<string, AgentThreadItem>>;
  item_order_by_turn: Readonly<Record<string, readonly string[]>>;
  pending_interactions: Readonly<Record<string, PendingInteractionProjection>>;
  strict_reviews: Readonly<Record<string, StrictReviewProjection>>;
  notices: readonly NoticeProjection[];
  orphan_deltas: Readonly<
    Record<string, readonly ConversationProjectionEvent[]>
  >;
  diagnostics: readonly ConversationProjectionDiagnostic[];
  applied_event_ids: Readonly<Record<string, true>>;
}

export interface ConversationProjection {
  thread_id: string | null;
  status: ConversationProjectionStatus;
  turns: readonly TurnProjection[];
  items: readonly AgentThreadItem[];
  pending_interactions: readonly PendingInteractionProjection[];
  strict_reviews: readonly StrictReviewProjection[];
  notices: readonly NoticeProjection[];
  diagnostics: readonly ConversationProjectionDiagnostic[];
}

export interface ConversationProjectionReducer {
  dispatch(event: ConversationProjectionEvent): ConversationProjectionState;
  getState(): ConversationProjectionState;
  getProjection(): ConversationProjection;
  reset(): ConversationProjectionState;
}

export type ConversationTurnTerminalStatus = Extract<
  AgentThreadTurnStatus,
  "completed" | "failed" | "interrupted" | "cancelled" | "canceled" | "aborted"
>;
