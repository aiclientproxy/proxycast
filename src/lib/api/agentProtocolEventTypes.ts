import type {
  AgentExecutionStrategy,
  AgentSessionExecutionRuntimeCostState,
  AgentSessionExecutionRuntimeLimitEvent,
  AgentSessionExecutionRuntimeLimitState,
  AgentSessionExecutionRuntimeRoutingDecision,
  AgentSessionExecutionRuntimeTaskProfile,
  AgentTurnOutputSchemaRuntime,
} from "./agentExecutionRuntime";
import type {
  AgentActionRequiredQuestion,
  AgentActionRequiredScope,
  AgentActionRequiredType,
  AgentArtifactSignal,
  AgentContextTraceStep,
  AgentMessage,
  AgentThreadItem,
  AgentThreadTurn,
  AgentTokenUsage,
  AgentToolExecutionResult,
  AgentToolProgressPayload,
} from "./agentProtocolCoreTypes";

export interface AgentEventTextDelta {
  type: "text_delta";
  text: string;
  itemId?: string;
  phase?: string;
}

export type AgentEventTextDeltaBatchBoundary =
  | "newline"
  | "backlog"
  | "final"
  | "provider"
  | (string & {});

export interface AgentEventTextDeltaBatch {
  type: "text_delta_batch";
  text: string;
  chunks: string[];
  boundary: AgentEventTextDeltaBatchBoundary;
  itemId?: string;
  phase?: string;
}

export interface AgentEventThreadStarted {
  type: "thread_started";
  thread_id: string;
}

export interface AgentEventTurnStarted {
  type: "turn_started";
  turn: AgentThreadTurn;
}

export interface AgentEventItemStarted {
  type: "item_started";
  item: AgentThreadItem;
}

export interface AgentEventItemUpdated {
  type: "item_updated";
  item: AgentThreadItem;
}

export interface AgentEventItemCompleted {
  type: "item_completed";
  item: AgentThreadItem;
}

export interface AgentEventTurnCompleted {
  type: "turn_completed";
  turn: AgentThreadTurn;
  text?: string;
  usage?: AgentTokenUsage;
}

export interface AgentEventTokenUsageUpdated {
  type: "token_usage_updated";
  turn_id?: string;
  usage: AgentTokenUsage;
}

export interface AgentEventTurnFailed {
  type: "turn_failed";
  turn: AgentThreadTurn;
}

export interface AgentEventTurnCanceled {
  type: "turn_canceled";
  turn: AgentThreadTurn;
}

export interface AgentEventThinkingDelta {
  type: "thinking_delta";
  text: string;
}

export type AgentProviderTraceStage =
  | "request_started"
  | "first_event_received"
  | "first_text_delta_received"
  | "failed"
  | "canceled"
  | (string & {});

export interface AgentEventProviderTrace {
  type: "provider_trace";
  stage: AgentProviderTraceStage;
  provider?: string;
  model?: string;
  attempt?: number;
  elapsed_ms?: number;
  text_chars?: number;
  status?: string;
  failure_category?: string;
  retryable?: boolean;
  non_retryable_provider_rejection?: boolean;
  cancel_reason?: string;
  provider_request_id?: string;
  provider_request_id_header?: string;
  runtime_provider_backend?: string;
  runtime_provider_selector?: string;
  runtime_provider_protocol?: string;
  runtime_provider_active_model?: string;
  runtime_event_type?: string;
}

export interface AgentEventReasoningStarted {
  type: "reasoning_started";
  reasoningId?: string;
  model?: unknown;
  providerMetadata?: Record<string, unknown>;
}

export interface AgentEventReasoningDelta {
  type: "reasoning_delta";
  reasoningId?: string;
  text: string;
  delta?: string;
  model?: unknown;
  providerMetadata?: Record<string, unknown>;
}

export interface AgentEventReasoningSummaryDelta {
  type: "reasoning_summary_delta";
  itemId: string;
  reasoningId: string;
  summaryIndex: number;
  text: string;
  delta: string;
}

export interface AgentEventReasoningSummaryPartAdded {
  type: "reasoning_summary_part_added";
  itemId: string;
  reasoningId: string;
  summaryIndex: number;
}

export interface AgentEventReasoningContentDelta {
  type: "reasoning_content_delta";
  itemId: string;
  reasoningId: string;
  contentIndex: number;
  text: string;
  delta: string;
}

export interface AgentEventReasoningFinal {
  type: "reasoning_final";
  reasoningId?: string;
  text: string;
  model?: unknown;
  providerMetadata?: Record<string, unknown>;
}

export interface AgentEventReasoningEnded {
  type: "reasoning_ended";
  reasoningId?: string;
  status?: string;
  model?: unknown;
  providerMetadata?: Record<string, unknown>;
}

export interface AgentEventPlanDelta {
  type: "plan_delta";
  text: string;
  delta?: string;
  plan?: unknown;
  explanation?: string;
  sourceItemId?: string;
  toolCallId?: string;
  revisionId?: string;
  source?: string;
}

export interface AgentEventPlanFinal {
  type: "plan_final";
  text: string;
  delta?: string;
  plan?: unknown;
  explanation?: string;
  sourceItemId?: string;
  toolCallId?: string;
  revisionId?: string;
  source?: string;
}

export interface AgentEventTurnPlanUpdated {
  type: "turn_plan_updated";
  explanation?: string;
  plan: Array<{
    step: string;
    status: "pending" | "in_progress" | "completed";
  }>;
}

export interface AgentEventTurnDiffUpdated {
  type: "turn_diff_updated";
  unified_diff: string;
}

export interface AgentEventToolStart {
  type: "tool_start";
  tool_name: string;
  tool_id: string;
  arguments?: string;
  metadata?: Record<string, unknown>;
}

export interface AgentEventToolEnd {
  type: "tool_end";
  tool_id: string;
  result: AgentToolExecutionResult;
}

export interface AgentEventImageTaskCreated {
  type: "image_task_created";
  task_id: string;
  task_type?: string;
  task_family?: string;
  turn_id?: string;
  status?: string;
  normalized_status?: string;
  artifact_path?: string;
  absolute_path?: string;
  response?: Record<string, unknown>;
  payload?: Record<string, unknown>;
}

export interface AgentEventImageTaskPresentationGenerated {
  type: "image_task_presentation_generated";
  status?: string;
  workflow_run_id?: string;
  session_id?: string;
  thread_id?: string;
  turn_id?: string;
  presentation?: Record<string, unknown>;
}
export interface AgentEventToolProgress {
  type: "tool_progress";
  tool_id: string;
  progress: AgentToolProgressPayload;
}

export interface AgentEventToolOutputDelta {
  type: "tool_output_delta";
  tool_id: string;
  delta: string;
  output_kind?: string;
  metadata?: Record<string, unknown>;
}

export interface AgentEventToolInputDelta {
  type: "tool_input_delta";
  tool_id: string;
  tool_name?: string;
  delta: string;
  accumulated_arguments?: string;
  provider?: string;
  metadata?: Record<string, unknown>;
}

export interface AgentEventArtifactSnapshot {
  type: "artifact_snapshot";
  artifact: AgentArtifactSignal;
}

export interface AgentEventActionRequired {
  type: "action_required";
  request_id: string;
  action_type: AgentActionRequiredType;
  scope?: AgentActionRequiredScope;
  tool_name?: string;
  arguments?: Record<string, unknown>;
  prompt?: string;
  questions?: AgentActionRequiredQuestion[];
  requested_schema?: Record<string, unknown>;
  available_decisions?: string[];
}

export interface AgentEventActionResolved {
  type: "action_resolved";
  request_id: string;
  action_type: AgentActionRequiredType | "plan_approval" | string;
  scope?: AgentActionRequiredScope;
  approved?: boolean;
  feedback?: string;
  permission_mode?: string;
  data?: Record<string, unknown>;
}

export interface AgentEventContextTrace {
  type: "context_trace";
  steps: AgentContextTraceStep[];
}

export interface AgentContextBudget {
  used_tokens?: number;
  max_tokens?: number;
  remaining_tokens?: number;
  status?: string;
  source?: string;
}

export interface AgentMissingContextFact {
  id?: string;
  kind: string;
  label: string;
  status: string;
  reason?: string;
  source?: string;
}

export interface AgentRetrievalRef {
  source_id: string;
  kind: string;
  title?: string;
  path?: string;
  url?: string;
  score?: number;
  scope?: string;
  status?: string;
  source?: string;
}

export interface AgentTeamMemoryRef {
  key: string;
  repo_scope?: string;
  updated_at?: number;
  priority?: number;
  source?: string;
}

export interface AgentTurnContextSummary {
  memory_budget?: AgentContextBudget | null;
  missing_context?: AgentMissingContextFact[];
  retrieval_refs?: AgentRetrievalRef[];
  team_memory_refs?: AgentTeamMemoryRef[];
}

export interface AgentEventTurnContext {
  type: "turn_context";
  session_id: string;
  thread_id: string;
  turn_id: string;
  execution_strategy?: AgentExecutionStrategy | null;
  output_schema_runtime?: AgentTurnOutputSchemaRuntime | null;
  context_summary?: AgentTurnContextSummary | null;
  approval_policy?: string | null;
  sandbox_policy?: string | null;
}

export interface AgentEventModelChange {
  type: "model_change";
  model: string;
  mode: string;
}

export interface AgentEventModelEffective {
  type: "model_effective";
  model?: unknown;
  modelRef?: unknown;
  provider?: string;
  modelName?: string;
  source?: string;
  serviceModelSlot?: string;
  reasoning?: unknown;
  capability?: unknown;
  toolCalling?: unknown;
  requestedReasoningEffort?: string;
}

export interface AgentRuntimeStatusMetadata {
  [key: string]: unknown;
  sourceType?: string;
  source?: string;
  kind?: string;
  eventClass?: string;
  event_class?: string;
  surface?: string;
  visibility?: string;
  persistence?: string;
  presentation?: string;
  agentui?: Record<string, unknown>;
  agentUi?: Record<string, unknown>;
  team_phase?: string;
  team_parallel_budget?: number;
  team_active_count?: number;
  team_queued_count?: number;
  concurrency_phase?: string;
  concurrency_scope?: string;
  concurrency_active_count?: number;
  concurrency_queued_count?: number;
  concurrency_budget?: number;
  provider_concurrency_group?: string;
  provider_parallel_budget?: number;
  queue_reason?: string;
  retryable_overload?: boolean;
  permission_status?: string;
  required_profile_keys?: string[];
  ask_profile_keys?: string[];
  blocking_profile_keys?: string[];
  decision_source?: string;
  decision_scope?: string;
  confirmation_status?: string;
  confirmation_request_id?: string;
  confirmation_source?: string;
  declared_only?: boolean;
  turn_gating?: boolean;
  limit_status?: string;
  capability_gap?: string;
  keepalive_kind?: string;
  keepalive_sequence?: number;
  keepalive_elapsed_ms?: number;
  skillRuntime?: Record<string, unknown>;
  skill_runtime?: Record<string, unknown>;
}

export interface AgentRuntimeStatusPayload {
  phase:
    | "preparing"
    | "routing"
    | "context"
    | "permission_review"
    | "retrying"
    | "continuing"
    | "synthesizing"
    | "cancelled"
    | "failed";
  title: string;
  detail: string;
  checkpoints?: string[];
  metadata?: AgentRuntimeStatusMetadata;
}

export interface AgentEventRuntimeStatus {
  type: "runtime_status";
  status: AgentRuntimeStatusPayload;
}

export interface AgentEventTaskProfileResolved {
  type: "task_profile_resolved";
  task_profile: AgentSessionExecutionRuntimeTaskProfile;
}

export interface AgentEventCandidateSetResolved {
  type: "candidate_set_resolved";
  routing_decision: AgentSessionExecutionRuntimeRoutingDecision;
}

export interface AgentEventRoutingDecisionMade {
  type: "routing_decision_made";
  routing_decision: AgentSessionExecutionRuntimeRoutingDecision;
}

export interface AgentEventRoutingFallbackApplied {
  type: "routing_fallback_applied";
  routing_decision: AgentSessionExecutionRuntimeRoutingDecision;
}

export interface AgentEventRoutingNotPossible {
  type: "routing_not_possible";
  routing_decision: AgentSessionExecutionRuntimeRoutingDecision;
}
export interface AgentEventLimitStateUpdated {
  type: "limit_state_updated";
  limit_state: AgentSessionExecutionRuntimeLimitState;
}

export interface AgentEventSingleCandidateOnly {
  type: "single_candidate_only";
  limit_state: AgentSessionExecutionRuntimeLimitState;
}

export interface AgentEventSingleCandidateCapabilityGap {
  type: "single_candidate_capability_gap";
  limit_state: AgentSessionExecutionRuntimeLimitState;
}

export interface AgentEventCostEstimated {
  type: "cost_estimated";
  cost_state: AgentSessionExecutionRuntimeCostState;
}

export interface AgentEventCostRecorded {
  type: "cost_recorded";
  cost_state: AgentSessionExecutionRuntimeCostState;
}

export interface AgentEventRateLimitHit {
  type: "rate_limit_hit";
  limit_event: AgentSessionExecutionRuntimeLimitEvent;
}

export interface AgentEventQuotaLow {
  type: "quota_low";
  limit_event: AgentSessionExecutionRuntimeLimitEvent;
}

export interface AgentEventQuotaBlocked {
  type: "quota_blocked";
  limit_event: AgentSessionExecutionRuntimeLimitEvent;
}

export interface AgentEventMessage {
  type: "message";
  message: AgentMessage;
}

export interface AgentEventWarning {
  type: "warning";
  code?: string;
  message: string;
}

export interface AgentEventStrictReviewRequired {
  type: "strict_review_required";
  started_at_ms: number;
}

export interface AgentEventGuardianReviewStarted {
  type: "guardian_review_started";
  review_id: string;
  target_item_id?: string;
  review: Record<string, unknown>;
  action: Record<string, unknown>;
}

export interface AgentEventGuardianReviewCompleted {
  type: "guardian_review_completed";
  review_id: string;
  target_item_id?: string;
  decision_source: "agent";
  review: Record<string, unknown>;
  action: Record<string, unknown>;
}

export interface AgentEventError {
  type: "error";
  message: string;
  will_retry?: boolean;
  codex_error_info?: unknown;
  additional_details?: string | null;
}

export interface AgentEventEnvelope {
  event_id?: string;
  protocol_method?: string;
  protocol_revision?: string;
  renderer_event_received_at?: number;
  request_id?: string;
  run_id?: string;
  sequence?: number;
  session_id?: string;
  server_event_emitted_at?: number;
  thread_id?: string;
  trace_id?: string;
  turn_id?: string;
  timestamp?: string;
}

export type AgentEvent = (
  | AgentEventThreadStarted
  | AgentEventTurnStarted
  | AgentEventItemStarted
  | AgentEventItemUpdated
  | AgentEventItemCompleted
  | AgentEventTurnCompleted
  | AgentEventTokenUsageUpdated
  | AgentEventTurnFailed
  | AgentEventTurnCanceled
  | AgentEventTextDelta
  | AgentEventTextDeltaBatch
  | AgentEventThinkingDelta
  | AgentEventProviderTrace
  | AgentEventReasoningStarted
  | AgentEventReasoningDelta
  | AgentEventReasoningSummaryDelta
  | AgentEventReasoningSummaryPartAdded
  | AgentEventReasoningContentDelta
  | AgentEventReasoningFinal
  | AgentEventReasoningEnded
  | AgentEventPlanDelta
  | AgentEventPlanFinal
  | AgentEventTurnDiffUpdated
  | AgentEventTurnPlanUpdated
  | AgentEventToolStart
  | AgentEventToolEnd
  | AgentEventImageTaskCreated
  | AgentEventImageTaskPresentationGenerated
  | AgentEventToolProgress
  | AgentEventToolOutputDelta
  | AgentEventToolInputDelta
  | AgentEventArtifactSnapshot
  | AgentEventActionRequired
  | AgentEventActionResolved
  | AgentEventTurnContext
  | AgentEventModelChange
  | AgentEventModelEffective
  | AgentEventContextTrace
  | AgentEventRuntimeStatus
  | AgentEventTaskProfileResolved
  | AgentEventCandidateSetResolved
  | AgentEventRoutingDecisionMade
  | AgentEventRoutingFallbackApplied
  | AgentEventRoutingNotPossible
  | AgentEventLimitStateUpdated
  | AgentEventSingleCandidateOnly
  | AgentEventSingleCandidateCapabilityGap
  | AgentEventCostEstimated
  | AgentEventCostRecorded
  | AgentEventRateLimitHit
  | AgentEventQuotaLow
  | AgentEventQuotaBlocked
  | AgentEventMessage
  | AgentEventWarning
  | AgentEventStrictReviewRequired
  | AgentEventGuardianReviewStarted
  | AgentEventGuardianReviewCompleted
  | AgentEventError
) &
  AgentEventEnvelope;
