import type {
  AgentContextBudget,
  AgentMessage,
  AgentMissingContextFact,
  AgentRetrievalRef,
  AgentTeamMemoryRef,
  AgentThreadItem,
  AgentThreadTurn,
} from "../agentProtocol";
import type {
  AgentExecutionStrategy,
  AgentSessionExecutionRuntime,
  AgentSessionExecutionRuntimeCostState,
  AgentSessionExecutionRuntimeLimitEvent,
  AgentSessionExecutionRuntimeLimitState,
  AgentSessionExecutionRuntimePermissionState,
} from "../agentExecutionRuntime";
import type { ModelCapabilities } from "@/lib/types/modelRegistry";

export interface AgentRuntimeGeneratedTitleResult {
  title: string;
  sessionId?: string | null;
  executionRuntime?: AgentSessionExecutionRuntime | null;
  usedFallback?: boolean;
  fallbackReason?: string | null;
}

/**
 * 图片输入
 */
export interface ImageInput {
  data: string;
  media_type: string;
}

/**
 * 当前 Provider / Model 选择。
 *
 * 这只是 Desktop Host 的配置读取结果；运行时仅由
 * `turn/start` 在 App Server 中初始化。
 */
export interface RuntimeProviderSelection {
  provider_configured: boolean;
  provider_name?: string;
  provider_selector?: string;
  model_name?: string;
}

/**
 * Runtime Provider 配置
 */
export interface RuntimeProviderConfig {
  provider_id?: string;
  provider_name: string;
  model_name: string;
  api_key?: string;
  base_url?: string;
  model_capabilities?: ModelCapabilities;
  tool_call_strategy?: "native" | "tool_shim";
  toolshim_model?: string;
}

export interface AutoContinueRequestPayload {
  enabled: boolean;
  fast_mode_enabled: boolean;
  continuation_length: number;
  sensitivity: number;
  source?: string;
}

/**
 * Agent 会话信息（匹配后端 SessionInfo 结构）
 */
export interface AgentSessionInfo {
  id: string;
  thread_id?: string;
  name?: string;
  created_at: number;
  updated_at: number;
  archived_at?: number | null;
  model?: string;
  messages_count?: number;
  execution_strategy?: AgentExecutionStrategy;
  workspace_id?: string;
  working_dir?: string;
  session_business_object_ref_metadata?: Record<string, unknown>;
  thread_status?: string;
  latest_turn_status?: string;
  active_turn_id?: string;
  queued_turn_count?: number;
}

export interface AgentRuntimeListSessionsOptions {
  includeArchived?: boolean;
  archivedOnly?: boolean;
  cwd?: string | string[];
  workspaceId?: string;
  limit?: number;
}

export interface AgentTodoItem {
  content: string;
  status: "pending" | "in_progress" | "completed";
  active_form?: string;
}

export interface AgentRuntimeRequestView {
  id: string;
  thread_id: string;
  turn_id?: string;
  item_id?: string;
  request_type: string;
  status: string;
  title?: string;
  payload?: unknown;
  decision?: unknown;
  scope?: Record<string, unknown>;
  created_at?: string | number;
  resolved_at?: string | number;
}

export interface AgentRuntimeOutcomeView {
  thread_id: string;
  turn_id?: string;
  outcome_type: string;
  summary?: string;
  primary_cause?: string;
  retryable?: boolean;
  ended_at?: string | number;
}

export interface AgentRuntimeIncidentView {
  id: string;
  thread_id: string;
  turn_id?: string;
  item_id?: string;
  incident_type: string;
  severity?: string;
  status?: string;
  title?: string;
  details?: unknown;
  detected_at?: string | number;
  cleared_at?: string | number;
}

export interface AgentRuntimeDiagnosticWarningSample {
  item_id: string;
  turn_id?: string;
  code?: string;
  message: string;
  updated_at: string | number;
}

export interface AgentRuntimeDiagnosticContextCompactionSample {
  item_id: string;
  turn_id?: string;
  stage: string;
  trigger?: string;
  detail?: string;
  updated_at: string | number;
}

export interface AgentRuntimeDiagnosticFailedToolSample {
  item_id: string;
  turn_id?: string;
  tool_name: string;
  error?: string;
  updated_at: string | number;
}

export interface AgentRuntimeDiagnosticFailedCommandSample {
  item_id: string;
  turn_id?: string;
  command: string;
  exit_code?: number;
  error?: string;
  updated_at: string | number;
}

export interface AgentRuntimeDiagnosticPendingRequestSample {
  request_id: string;
  turn_id?: string;
  request_type: string;
  title?: string;
  waited_seconds?: number;
  created_at?: string | number;
}

export interface AgentRuntimeCompactionBoundarySnapshot {
  session_id: string;
  summary_preview: string;
  turn_count?: number;
  created_at: string | number;
  trigger?: string;
  detail?: string;
}

export interface AgentRuntimeFileCheckpointSummary {
  checkpoint_id: string;
  turn_id: string;
  path: string;
  source: string;
  updated_at: string | number;
  version_no?: number;
  version_id?: string;
  request_id?: string;
  title?: string;
  kind?: string;
  status?: string;
  preview_text?: string;
  snapshot_path?: string;
  validation_issue_count: number;
}

export interface AgentRuntimeFileCheckpointThreadSummary {
  count: number;
  latest_checkpoint?: AgentRuntimeFileCheckpointSummary | null;
}

export interface AgentRuntimeFileCheckpointListResult {
  session_id: string;
  thread_id: string;
  checkpoint_count: number;
  checkpoints: AgentRuntimeFileCheckpointSummary[];
}

export interface AgentRuntimeFileCheckpointDetail {
  session_id: string;
  thread_id: string;
  checkpoint: AgentRuntimeFileCheckpointSummary;
  live_path: string;
  snapshot_path: string;
  checkpoint_document?: unknown;
  live_document?: unknown;
  version_history: unknown[];
  validation_issues: string[];
  metadata?: unknown;
  content?: string;
}

export interface AgentRuntimeFileCheckpointDiffResult {
  session_id: string;
  thread_id: string;
  checkpoint: AgentRuntimeFileCheckpointSummary;
  current_version_id?: string;
  previous_version_id?: string;
  diff?: unknown;
}

export interface AgentRuntimeFileCheckpointRestoreResult {
  session_id: string;
  thread_id: string;
  checkpoint: AgentRuntimeFileCheckpointSummary;
  live_path: string;
  snapshot_path: string;
  backup_path?: string | null;
  restored_at: string | number;
}

export interface AgentRuntimeDiagnosticProviderSafetyBufferingSample {
  source_event_id?: string;
  source_event_type?: string;
  thread_id?: string | null;
  turn_id?: string | null;
  timestamp?: string | number;
  provider?: string | null;
  model?: string | null;
  use_cases?: string[];
  reasons?: string[];
  show_buffering_ui?: boolean;
  retry_model?: string | null;
  fallback_header_model?: string | null;
  source?: string | null;
  backend?: string | null;
}

export interface AgentRuntimeThreadDiagnostics {
  latest_turn_status?: string;
  latest_turn_started_at?: string | number;
  latest_turn_completed_at?: string | number;
  latest_turn_updated_at?: string | number;
  latest_turn_elapsed_seconds?: number;
  latest_turn_stalled_seconds?: number;
  latest_turn_error_message?: string;
  interrupt_reason?: string;
  runtime_interrupt_source?: string;
  runtime_interrupt_requested_at?: string | number;
  runtime_interrupt_wait_seconds?: number;
  provider_safety_buffering_count?: number;
  warning_count: number;
  context_compaction_count: number;
  failed_tool_call_count: number;
  failed_command_count: number;
  pending_request_count: number;
  oldest_pending_request_wait_seconds?: number;
  primary_blocking_kind?: string;
  primary_blocking_summary?: string;
  latest_provider_safety_buffering?: AgentRuntimeDiagnosticProviderSafetyBufferingSample | null;
  latest_warning?: AgentRuntimeDiagnosticWarningSample | null;
  latest_context_compaction?: AgentRuntimeDiagnosticContextCompactionSample | null;
  latest_failed_tool?: AgentRuntimeDiagnosticFailedToolSample | null;
  latest_failed_command?: AgentRuntimeDiagnosticFailedCommandSample | null;
  latest_pending_request?: AgentRuntimeDiagnosticPendingRequestSample | null;
}

export interface AgentRuntimeThreadReadModel {
  thread_id: string;
  can_accept_direct_input?: boolean | null;
  session_business_object_ref_metadata?: Record<string, unknown> | null;
  status?: string;
  profile_status?: AgentRuntimeProfileStatus;
  active_turn_id?: string;
  turns?: AgentRuntimeThreadTurnProfileView[];
  thread_items?: AgentThreadItem[];
  pending_requests?: AgentRuntimeRequestView[];
  last_outcome?: AgentRuntimeOutcomeView | null;
  incidents?: AgentRuntimeIncidentView[];
  tool_calls?: AgentRuntimeThreadToolCallView[];
  commands?: AgentRuntimeThreadCommandView[];
  tests?: AgentRuntimeThreadTestRunView[];
  active_command_id?: string | null;
  active_test_run_id?: string | null;
  active_action_id?: string | null;
  artifacts?: Record<string, unknown>[];
  workflow_runs?: Record<string, unknown>[];
  workflowRuns?: Record<string, unknown>[];
  workflow_steps?: Record<string, unknown>[];
  workflowSteps?: Record<string, unknown>[];
  article_workspace?: Record<string, unknown> | null;
  articleWorkspace?: Record<string, unknown> | null;
  model_routing?: Record<string, unknown> | null;
  evidence_summary?: AgentRuntimeThreadEvidenceSummary | null;
  telemetry_summary?: AgentRuntimeThreadTelemetrySummary | null;
  context_summary?: AgentRuntimeThreadContextSummary | null;
  interrupt_state?: string;
  updated_at?: string | number;
  latest_compaction_boundary?: AgentRuntimeCompactionBoundarySnapshot | null;
  file_checkpoint_summary?: AgentRuntimeFileCheckpointThreadSummary | null;
  diagnostics?: AgentRuntimeThreadDiagnostics | null;
  task_kind?: string | null;
  service_model_slot?: string | null;
  routing_mode?: string | null;
  decision_source?: string | null;
  decision_reason?: string | null;
  candidate_count?: number | null;
  fallback_chain?: string[] | null;
  capability_gap?: string | null;
  estimated_cost_class?: string | null;
  single_candidate_only?: boolean | null;
  oem_policy?: AgentRuntimeOemPolicySummary | null;
  runtime_summary?: AgentRuntimeSummary | null;
  auxiliary_task_runtime?: Record<string, unknown>[] | null;
  limit_state?: AgentSessionExecutionRuntimeLimitState | null;
  cost_state?: AgentSessionExecutionRuntimeCostState | null;
  permission_state?: AgentSessionExecutionRuntimePermissionState | null;
  limit_event?: AgentSessionExecutionRuntimeLimitEvent | null;
}

export type AgentRuntimeProfileStatus =
  | "idle"
  | "queued"
  | "running"
  | "blocked"
  | "completed"
  | "failed"
  | "cancelled"
  | "stale"
  | "unknown";

export interface AgentRuntimeThreadTurnProfileView {
  turn_id: string;
  status: AgentRuntimeProfileStatus;
  native_status?: string;
}

export interface AgentRuntimeThreadToolCallView {
  tool_call_id: string;
  turn_id: string;
  tool_name: string;
  status: "running" | "completed" | "failed" | string;
  started_at?: string | null;
  finished_at?: string | null;
  updated_at?: string | null;
  arguments?:
    | Record<string, unknown>
    | unknown[]
    | string
    | number
    | boolean
    | null;
  output?: string | null;
  output_preview?: string | null;
  structured_content?: unknown;
  metadata?: Record<string, unknown> | null;
  success?: boolean | null;
  error?: string | null;
  evidence_refs?: string[];
}

export interface AgentRuntimeThreadCommandView {
  command_id: string;
  turn_id?: string;
  status: "running" | "completed" | "failed" | "canceled" | string;
  command?: string;
  canonical_command?: string;
  command_summary?: string;
  command_argv?: string[];
  command_argv_source?: string;
  cwd?: string;
  exit_code?: number;
  process_id?: string;
  execution_process_status?: string;
  execution_process_control_status?: string;
  execution_surface?: string;
  stdin_writable?: boolean;
  output_bytes?: number;
  output_omitted_bytes?: number;
  output_truncated?: boolean;
  stdout_bytes?: number;
  stderr_bytes?: number;
  output_refs?: string[];
  output_preview?: string | null;
  started_at?: string | number | null;
  completed_at?: string | number | null;
  updated_at?: string | number | null;
  source_event_ids?: string[];
}

export interface AgentRuntimeThreadTestRunView {
  test_run_id: string;
  turn_id?: string;
  status: "running" | "completed" | "failed" | "canceled" | string;
  command_id?: string;
  suite?: string;
  result?: string;
  passed?: number;
  failed?: number;
  output_refs?: string[];
  failure_category?: string;
  started_at?: string | number | null;
  completed_at?: string | number | null;
  updated_at?: string | number | null;
  source_event_ids?: string[];
}

export interface AgentRuntimeThreadEvidenceSummary {
  evidence_refs?: string[];
  verification_outcomes?: Record<string, unknown>[];
}

export interface AgentRuntimeThreadTelemetrySummary {
  trace_ids?: string[];
  join_status?: string;
}

export interface AgentRuntimeThreadContextSummary {
  owner?: "AgentContext" | string;
  source?: string;
  sources?: string[];
  memory_budget?: AgentContextBudget | null;
  missing_context?: AgentMissingContextFact[];
  retrieval_refs?: AgentRetrievalRef[];
  team_memory_refs?: AgentTeamMemoryRef[];
  latestCompaction?: AgentRuntimeCompactionBoundarySnapshot | null;
}

export interface AgentRuntimeOemPolicySummary {
  tenantId?: string | null;
  providerSource?: string | null;
  providerKey?: string | null;
  defaultModel?: string | null;
  configMode?: string | null;
  offerState?: string | null;
  quotaStatus?: string | null;
  fallbackToLocalAllowed?: boolean | null;
  canInvoke?: boolean | null;
  locked?: boolean | null;
  quotaLow?: boolean | null;
  limitEventKind?: string | null;
  limitEventMessage?: string | null;
  decisionSource?: string | null;
  selectedProvider?: string | null;
  selectedModel?: string | null;
}

export interface AgentRuntimeSummary {
  candidateCount?: number | null;
  routingMode?: string | null;
  decisionSource?: string | null;
  decisionReason?: string | null;
  fallbackChain?: string[] | null;
  estimatedCostClass?: string | null;
  estimatedTotalCost?: number | null;
  limitStatus?: string | null;
  limitEventKind?: string | null;
  limitEventMessage?: string | null;
  capabilityGap?: string | null;
  singleCandidateOnly?: boolean | null;
  oemLocked?: boolean | null;
  quotaLow?: boolean | null;
  limecorePolicy?: AgentRuntimeThreadLimeCorePolicySummary | null;
}

export interface AgentRuntimeThreadLimeCorePolicySummary {
  contractKey?: string | null;
  snapshotStatus?: string | null;
  decision?: string | null;
  decisionSource?: string | null;
  decisionScope?: string | null;
  decisionReason?: string | null;
  refs?: string[];
  evaluatedRefs?: string[];
  missingInputs?: string[];
  pendingHitRefs?: string[];
  policyValueHitCount?: number | null;
  source?: string | null;
  evaluation?: AgentRuntimeThreadLimeCorePolicyEvaluation | null;
}

export interface AgentRuntimeThreadLimeCorePolicyEvaluation {
  status?: string | null;
  decision?: string | null;
  decisionSource?: string | null;
  decisionScope?: string | null;
  decisionReason?: string | null;
  blockingRefs?: string[];
  askRefs?: string[];
  pendingRefs?: string[];
}

export interface AgentSessionHistoryCursor {
  oldest_message_id?: number | null;
  start_index?: number | null;
  loaded_count?: number | null;
}

/**
 * Agent 会话详情（匹配后端 SessionDetail 结构）
 */
export interface AgentSessionDetail {
  id: string;
  thread_id?: string;
  name?: string;
  created_at: number;
  updated_at: number;
  model?: string;
  workspace_id?: string;
  working_dir?: string;
  execution_strategy?: AgentExecutionStrategy;
  execution_runtime?: AgentSessionExecutionRuntime | null;
  messages_count?: number;
  history_limit?: number | null;
  history_offset?: number | null;
  history_cursor?: AgentSessionHistoryCursor | null;
  history_truncated?: boolean;
  messages: AgentMessage[];
  turns?: AgentThreadTurn[];
  items?: AgentThreadItem[];
  thread_read?: AgentRuntimeThreadReadModel | null;
  todo_items?: AgentTodoItem[];
}
