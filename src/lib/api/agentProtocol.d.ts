import type {
  AgentApprovalPolicy,
  AgentExecutionStrategy,
  AgentSandboxPolicy,
  AgentSessionExecutionRuntimeCostState,
  AgentSessionExecutionRuntimeLimitEvent,
  AgentSessionExecutionRuntimeLimitState,
  AgentSessionExecutionRuntimeRoutingDecision,
  AgentSessionExecutionRuntimeTaskProfile,
  AgentTurnOutputSchemaRuntime,
} from "./agentExecutionRuntime";
import type { TurnStartParams } from "@limecloud/app-server-client";
export interface AgentContextTraceStep {
  stage: string;
  detail: string;
}
export interface AgentToolResultImage {
  src: string;
  mimeType?: string;
  origin?: "data_url" | "tool_payload" | "file_path";
}
export type AgentToolResultMetadata = Record<string, unknown>;
export interface AgentToolExecutionResult {
  success: boolean;
  output: string;
  error?: string;
  images?: AgentToolResultImage[];
  metadata?: AgentToolResultMetadata;
  structuredContent?: unknown;
  structured_content?: unknown;
}
export interface AgentMessageTextElement {
  byte_range: {
    start: number;
    end: number;
  };
  placeholder?: string;
}
export interface AgentMessageContentText {
  type: "text";
  text: string;
  text_elements?: AgentMessageTextElement[];
}
export interface AgentMessageContentThinking {
  type: "thinking";
  text: string;
}
export interface AgentMessageContentToolRequest {
  type: "tool_request";
  id: string;
  tool_name: string;
  arguments: unknown;
}
export interface AgentMessageContentToolResponse {
  type: "tool_response";
  id: string;
  success: boolean;
  output: string;
  error?: string;
  images?: AgentToolResultImage[];
  metadata?: AgentToolResultMetadata;
}
export interface AgentMessageContentActionRequired {
  type: "action_required";
  id: string;
  action_type: AgentActionRequiredType | string;
  data: unknown;
  scope?: AgentActionRequiredScope;
}
export interface AgentMessageContentImage {
  type: "image";
  mime_type: string;
  /** 内联图片使用 data；历史文件/URL 图片使用 uri/source_path，data 保持为空。 */
  data: string;
  uri?: string;
  source_path?: string;
  detail?: "auto" | "low" | "high" | "original";
}
export interface AgentMessageContentSkill {
  type: "skill";
  name: string;
  path: string;
}
export interface AgentMessageContentMention {
  type: "mention";
  name: string;
  path: string;
}
export type AgentMessageContent =
  | AgentMessageContentText
  | AgentMessageContentThinking
  | AgentMessageContentToolRequest
  | AgentMessageContentToolResponse
  | AgentMessageContentActionRequired
  | AgentMessageContentImage
  | AgentMessageContentSkill
  | AgentMessageContentMention;
export interface AgentMessage {
  id?: string;
  role: string;
  content: AgentMessageContent[];
  timestamp: number;
  runtimeTurnId?: string;
  runtime_turn_id?: string;
  usage?: AgentTokenUsage;
}
export interface AgentArtifactSignal {
  artifactId: string;
  filePath?: string;
  content?: string;
  metadata?: Record<string, unknown>;
}
export interface AgentTokenUsage {
  input_tokens: number;
  output_tokens: number;
  cached_input_tokens?: number;
  cache_creation_input_tokens?: number;
}
export type AgentThreadTurnStatus =
  | "running"
  | "completed"
  | "failed"
  | "canceled"
  | "aborted"
  | "cancelled"
  | "interrupted";
export type AgentThreadItemStatus = "in_progress" | "completed" | "failed";
export interface AgentThreadTurn {
  id: string;
  thread_id: string;
  prompt_text: string;
  status: AgentThreadTurnStatus;
  started_at: string;
  completed_at?: string;
  error_message?: string;
  created_at: string;
  updated_at: string;
}
export interface AgentRequestOption {
  label: string;
  description?: string;
}
export interface AgentRequestQuestion {
  question: string;
  header?: string;
  options?: AgentRequestOption[];
  multi_select?: boolean;
}
interface AgentThreadItemBase {
  id: string;
  thread_id: string;
  turn_id: string;
  sequence: number;
  ordinal?: number;
  status: AgentThreadItemStatus;
  started_at: string;
  completed_at?: string;
  updated_at: string;
  metadata?: unknown;
}
export interface AgentThreadUserMessageItem extends AgentThreadItemBase {
  type: "user_message";
  content: string;
  content_parts?: AgentMessageContent[];
}
export interface AgentThreadContentReference {
  uri: string;
  mime_type: string;
  title?: string;
  source_uri?: string;
  source_path?: string;
  preview_url?: string;
  sha256?: string;
  byte_size?: number;
}
export type AgentThreadMessageContentPart =
  | {
      type: "text";
      text: string;
    }
  | {
      type: "media";
      kind: string;
      reference: AgentThreadContentReference;
      caption?: string;
    };
export interface AgentThreadAgentMessageItem extends AgentThreadItemBase {
  type: "agent_message";
  text: string;
  contentParts?: AgentThreadMessageContentPart[];
  phase?: string;
}
export interface AgentThreadPlanItem extends AgentThreadItemBase {
  type: "plan";
  text: string;
}
export interface AgentThreadReasoningItem extends AgentThreadItemBase {
  type: "reasoning";
  text: string;
  summary?: string[];
  content?: string[];
}
export interface AgentThreadToolCallItem extends AgentThreadItemBase {
  type: "tool_call";
  tool_name: string;
  arguments?: unknown;
  output?: string;
  success?: boolean;
  error?: string;
  metadata?: unknown;
}
export interface AgentThreadCommandExecutionItem extends AgentThreadItemBase {
  type: "command_execution";
  command: string;
  cwd: string;
  aggregated_output?: string;
  exit_code?: number;
  error?: string;
}
export type AgentThreadPatchChangeKind =
  | {
      type: "add";
    }
  | {
      type: "delete";
    }
  | {
      type: "update";
      move_path?: string | null;
    };
export interface AgentThreadPatchChange {
  path: string;
  kind: AgentThreadPatchChangeKind;
  diff: string;
}
export type AgentThreadPatchApplyStatus =
  | "inProgress"
  | "completed"
  | "declined"
  | "failed";
export interface AgentThreadPatchItem extends AgentThreadItemBase {
  type: "patch";
  text: string;
  changes?: AgentThreadPatchChange[];
  file_status?: AgentThreadPatchApplyStatus;
  summary?: string[];
  paths?: string[];
  success?: boolean;
  stdout?: string;
  stderr?: string;
  metadata?: unknown;
}
export interface AgentThreadWebSearchItem extends AgentThreadItemBase {
  type: "web_search";
  query?: string;
  action?: string;
  action_data?: unknown;
  results?: unknown[];
  output?: string;
}
export interface AgentThreadHookOutputEntry {
  kind: string;
  text: string;
}
export interface AgentThreadHookItem extends AgentThreadItemBase {
  type: "hook";
  run_id: string;
  event_name?: string;
  handler_type?: string;
  execution_mode?: string;
  scope?: string;
  source_path?: string;
  source?: string;
  display_order?: number;
  status_message?: string;
  duration_ms?: number;
  entries?: AgentThreadHookOutputEntry[];
  output?: string;
  target_item_id?: string;
  hook_status?: string;
  metadata?: unknown;
}
export interface AgentThreadApprovalRequestItem extends AgentThreadItemBase {
  type: "approval_request";
  request_id: string;
  action_type: string;
  prompt?: string;
  tool_name?: string;
  arguments?: unknown;
  available_decisions?: string[];
  response?: unknown;
}
export interface AgentThreadRequestUserInputItem extends AgentThreadItemBase {
  type: "request_user_input";
  request_id: string;
  action_type: string;
  prompt?: string;
  questions?: AgentRequestQuestion[];
  response?: unknown;
}
export interface AgentThreadFileArtifactItem extends AgentThreadItemBase {
  type: "file_artifact";
  path: string;
  source: string;
  content?: string;
  metadata?: unknown;
}

export interface AgentThreadMediaItem extends AgentThreadItemBase {
  type: "media";
  uri: string;
  mime_type: string;
  preview?: string;
}

export interface AgentThreadImageGenerationItem extends AgentThreadItemBase {
  type: "image_generation";
  generation_status: string;
  revised_prompt?: string;
  result: string;
  saved_path?: string;
}

export interface AgentThreadExtensionItem extends AgentThreadItemBase {
  type: "extension";
  name: string;
  data: Record<string, unknown>;
}
export interface AgentThreadSubagentActivityItem extends AgentThreadItemBase {
  type: "subagent_activity";
  status_label: string;
  title?: string;
  summary?: string;
  role?: string;
  model?: string;
  session_id?: string;
}
export interface AgentThreadExpertProfileSwitchItem extends AgentThreadItemBase {
  type: "expert_profile_switch";
  title?: string;
  summary?: string;
  previous_expert_id?: string;
  previous_release_id?: string;
  next_expert_id?: string;
  next_release_id?: string;
  switched_at?: string;
  expert_role_switch?: unknown;
  expert?: unknown;
  harness_expert?: unknown;
  metadata?: unknown;
}
export interface AgentThreadWarningItem extends AgentThreadItemBase {
  type: "warning";
  message: string;
  code?: string;
}
export interface AgentThreadContextCompactionItem extends AgentThreadItemBase {
  type: "context_compaction";
  stage: "started" | "completed" | string;
  trigger?: string;
  detail?: string;
}
export interface AgentThreadUnknownItem extends AgentThreadItemBase {
  type: "unknown_item";
  upstream_type: string;
  field_names: string[];
}
export interface AgentThreadErrorItem extends AgentThreadItemBase {
  type: "error";
  message: string;
}
export interface AgentThreadTurnSummaryItem extends AgentThreadItemBase {
  type: "turn_summary";
  text: string;
  metadata?: Record<string, unknown>;
}
export type AgentThreadItem =
  | AgentThreadUserMessageItem
  | AgentThreadAgentMessageItem
  | AgentThreadPlanItem
  | AgentThreadReasoningItem
  | AgentThreadToolCallItem
  | AgentThreadCommandExecutionItem
  | AgentThreadPatchItem
  | AgentThreadWebSearchItem
  | AgentThreadHookItem
  | AgentThreadApprovalRequestItem
  | AgentThreadRequestUserInputItem
  | AgentThreadFileArtifactItem
  | AgentThreadMediaItem
  | AgentThreadImageGenerationItem
  | AgentThreadExtensionItem
  | AgentThreadSubagentActivityItem
  | AgentThreadExpertProfileSwitchItem
  | AgentThreadWarningItem
  | AgentThreadContextCompactionItem
  | AgentThreadUnknownItem
  | AgentThreadErrorItem
  | AgentThreadTurnSummaryItem;
export interface AgentToolCallState {
  id: string;
  name: string;
  arguments?: string;
  status: "running" | "completed" | "failed";
  result?: AgentToolExecutionResult;
  metadata?: Record<string, unknown>;
  progress?: AgentToolProgressPayload & {
    updatedAt?: Date;
  };
  startTime: Date;
  endTime?: Date;
  logs?: string[];
}
export interface AgentActionRequiredScope {
  session_id?: string;
  thread_id?: string;
  turn_id?: string;
}
export type AgentActionRequiredType =
  | "tool_confirmation"
  | "ask_user"
  | "elicitation";
export interface AgentActionRequiredOption {
  label: string;
  description?: string;
}
export interface AgentActionRequiredQuestion {
  question: string;
  header?: string;
  options?: AgentActionRequiredOption[];
  multiSelect?: boolean;
}
export interface AgentEventTextDelta {
  type: "text_delta";
  text: string;
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
export interface AgentEventToolStart {
  type: "tool_start";
  tool_name: string;
  tool_id: string;
  arguments?: string;
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
export interface AgentToolProgressPayload {
  message?: string;
  progress?: number;
  total?: number;
  metadata?: Record<string, unknown>;
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
export interface AgentEventError {
  type: "error";
  message: string;
}
export type AgentEvent =
  | AgentEventThreadStarted
  | AgentEventTurnStarted
  | AgentEventItemStarted
  | AgentEventItemUpdated
  | AgentEventItemCompleted
  | AgentEventTurnCompleted
  | AgentEventTurnFailed
  | AgentEventTurnCanceled
  | AgentEventTextDelta
  | AgentEventTextDeltaBatch
  | AgentEventThinkingDelta
  | AgentEventReasoningStarted
  | AgentEventReasoningDelta
  | AgentEventReasoningSummaryDelta
  | AgentEventReasoningSummaryPartAdded
  | AgentEventReasoningContentDelta
  | AgentEventReasoningFinal
  | AgentEventReasoningEnded
  | AgentEventPlanDelta
  | AgentEventPlanFinal
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
  | AgentEventError;
export interface AgentUserInputOp {
  type: "user_input";
  eventName: string;
  turn: TurnStartParams;
}
export interface AgentInterruptOp {
  type: "interrupt";
  sessionId: string;
  turnId?: string;
}
export interface AgentRetryOp {
  type: "retry";
  sessionId: string;
  turnId: string;
}
export interface AgentConfigUpdateOp {
  type: "config_update";
  sessionId: string;
  key: string;
  value: unknown;
}
export interface AgentShutdownOp {
  type: "shutdown";
  sessionId?: string;
}
export type AgentOp =
  | AgentUserInputOp
  | AgentInterruptOp
  | AgentRetryOp
  | AgentConfigUpdateOp
  | AgentShutdownOp;
export declare function parseAgentEvent(data: unknown): AgentEvent | null;
export declare function createAgentSessionTurnStartParamsFromUserInputOp(
  op: AgentUserInputOp,
): TurnStartParams;
export {};
