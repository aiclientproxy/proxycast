export type AgentRuntimeHandoffArtifactKind =
  | "plan"
  | "progress"
  | "handoff"
  | "review_summary";

export interface AgentRuntimeHandoffArtifact {
  kind: AgentRuntimeHandoffArtifactKind;
  title: string;
  relative_path: string;
  absolute_path: string;
  bytes: number;
}

export interface AgentRuntimeHandoffBundle {
  session_id: string;
  thread_id: string;
  workspace_id?: string;
  workspace_root: string;
  bundle_relative_root: string;
  bundle_absolute_root: string;
  exported_at: string;
  thread_status: string;
  latest_turn_status?: string;
  pending_request_count: number;
  queued_turn_count: number;
  active_subagent_count: number;
  todo_total: number;
  todo_pending: number;
  todo_in_progress: number;
  todo_completed: number;
  artifacts: AgentRuntimeHandoffArtifact[];
}

export type AgentRuntimeEvidenceArtifactKind =
  | "summary"
  | "runtime"
  | "timeline"
  | "artifacts";

export interface AgentRuntimeEvidenceArtifact {
  kind: AgentRuntimeEvidenceArtifactKind;
  title: string;
  relative_path: string;
  absolute_path: string;
  bytes: number;
}

export type AgentRuntimeEvidenceVerificationOutcome =
  | "success"
  | "blocking_failure"
  | "advisory_failure"
  | "recovered";

export interface AgentRuntimeEvidenceSignalCoverageEntry {
  signal: string;
  status: string;
  source: string;
  detail: string;
}

export interface AgentRuntimeEvidenceCountEntry {
  count: number;
}

export interface AgentRuntimeEvidenceStatusCount extends AgentRuntimeEvidenceCountEntry {
  status: string;
}

export interface AgentRuntimeEvidenceArtifactKindCount extends AgentRuntimeEvidenceCountEntry {
  artifact_kind: string;
}

export interface AgentRuntimeEvidenceActionCount extends AgentRuntimeEvidenceCountEntry {
  action: string;
}

export interface AgentRuntimeEvidenceBackendCount extends AgentRuntimeEvidenceCountEntry {
  backend: string;
}

export interface AgentRuntimeEvidenceExecutorCount extends AgentRuntimeEvidenceCountEntry {
  executor: string;
}

export interface AgentRuntimeEvidenceBrowserActionItem {
  artifact_path?: string;
  contract_key?: string;
  source?: string;
  entry_source?: string;
  artifact_kind?: string;
  tool_name?: string;
  action?: string;
  action_id?: string;
  status?: string;
  success?: boolean;
  session_id?: string;
  target_id?: string;
  tab_id?: string;
  profile_key?: string;
  backend?: string;
  request_id?: string;
  confirmation_request_id?: string;
  control_mode?: string;
  lifecycle_state?: string;
  human_reason?: string;
  thread_id?: string;
  turn_id?: string;
  content_id?: string;
  executor?: string;
  evidence_refs?: string[];
  last_url?: string;
  title?: string;
  attempt_count?: number;
  observation_available?: boolean;
  screenshot_available?: boolean;
}

export interface AgentRuntimeEvidenceBrowserActionIndex {
  action_count: number;
  session_count: number;
  observation_count: number;
  screenshot_count: number;
  last_url?: string;
  thread_ids: string[];
  turn_ids: string[];
  content_ids: string[];
  session_ids: string[];
  target_ids: string[];
  profile_keys: string[];
  status_counts: AgentRuntimeEvidenceStatusCount[];
  artifact_kind_counts: AgentRuntimeEvidenceArtifactKindCount[];
  action_counts: AgentRuntimeEvidenceActionCount[];
  backend_counts: AgentRuntimeEvidenceBackendCount[];
  executor_counts: AgentRuntimeEvidenceExecutorCount[];
  items: AgentRuntimeEvidenceBrowserActionItem[];
}

export interface AgentRuntimeEvidenceTaskIndexItem {
  artifact_path?: string;
  task_id?: string;
  task_type?: string;
  contract_key?: string;
  source?: string;
  thread_id?: string;
  turn_id?: string;
  content_id?: string;
  entry_key?: string;
  entry_source?: string;
  modality?: string;
  skill_id?: string;
  model_id?: string;
  executor_kind?: string;
  executor_binding_key?: string;
  cost_state?: string;
  limit_state?: string;
  estimated_cost_class?: string;
  limit_event_kind?: string;
  quota_low?: boolean;
  routing_outcome?: string;
}

export interface AgentRuntimeEvidenceTaskIndex {
  snapshot_count: number;
  thread_ids: string[];
  turn_ids: string[];
  content_ids: string[];
  entry_keys: string[];
  modalities: string[];
  skill_ids: string[];
  model_ids: string[];
  executor_kinds: string[];
  executor_binding_keys: string[];
  cost_states: string[];
  limit_states: string[];
  estimated_cost_classes: string[];
  limit_event_kinds: string[];
  quota_low_count: number;
  items: AgentRuntimeEvidenceTaskIndexItem[];
}

export interface AgentRuntimeEvidenceDecisionCount extends AgentRuntimeEvidenceCountEntry {
  decision: string;
}

export interface AgentRuntimeEvidenceLimeCorePolicyItem {
  artifact_path?: string;
  contract_key?: string;
  execution_profile_key?: string;
  executor_adapter_key?: string;
  refs: string[];
  status?: string;
  decision?: string;
  decision_source?: string;
  decision_scope?: string;
  decision_reason?: string;
  evaluated_refs?: string[];
  unresolved_refs?: string[];
  missing_inputs?: string[];
  policy_inputs?: AgentRuntimeEvidenceLimeCorePolicyInput[];
  pending_hit_refs?: string[];
  policy_value_hits?: AgentRuntimeEvidenceLimeCorePolicyValueHit[];
  policy_value_hit_count?: number;
  policy_evaluation?: AgentRuntimeEvidenceLimeCorePolicyEvaluation;
  source?: string;
}

export interface AgentRuntimeEvidenceLimeCorePolicyEvaluation {
  status?: string;
  decision?: string;
  decision_source?: string;
  decision_scope?: string;
  decision_reason?: string;
  blocking_refs?: string[];
  ask_refs?: string[];
  pending_refs?: string[];
}

export interface AgentRuntimeEvidenceLimeCorePolicyInput {
  ref_key: string;
  status?: string;
  source?: string;
  value_source?: string;
}

export interface AgentRuntimeEvidenceLimeCorePolicyValueHit {
  ref_key: string;
  status?: string;
  source?: string;
  value_source?: string;
  value?: unknown;
  summary?: string;
  evidence_ref?: string;
  observed_at?: string;
}

export interface AgentRuntimeEvidenceLimeCorePolicyIndex {
  snapshot_count: number;
  ref_keys: string[];
  missing_inputs?: string[];
  pending_hit_refs?: string[];
  policy_value_hit_count?: number;
  status_counts: AgentRuntimeEvidenceStatusCount[];
  decision_counts: AgentRuntimeEvidenceDecisionCount[];
  items: AgentRuntimeEvidenceLimeCorePolicyItem[];
}

export interface AgentRuntimeEvidenceSnapshotIndex {
  task_index?: AgentRuntimeEvidenceTaskIndex;
  browser_action_index?: AgentRuntimeEvidenceBrowserActionIndex;
  limecore_policy_index?: AgentRuntimeEvidenceLimeCorePolicyIndex;
}

export interface AgentRuntimeEvidenceModalityRuntimeContracts {
  snapshot_count: number;
  snapshot_index?: AgentRuntimeEvidenceSnapshotIndex;
}

export interface AgentRuntimeArtifactValidatorVerificationSummary {
  applicable: boolean;
  record_count: number;
  issue_count: number;
  repaired_count: number;
  fallback_used_count: number;
  outcome?: AgentRuntimeEvidenceVerificationOutcome;
}

export interface AgentRuntimeBrowserVerificationSummary {
  record_count: number;
  success_count: number;
  failure_count: number;
  unknown_count: number;
  latest_updated_at?: string;
  outcome?: AgentRuntimeEvidenceVerificationOutcome;
}

export interface AgentRuntimeGuiSmokeVerificationSummary {
  status?: string;
  exit_code?: number;
  passed: boolean;
  updated_at?: string;
  has_output_preview: boolean;
  outcome?: AgentRuntimeEvidenceVerificationOutcome;
}

export interface AgentRuntimeEvidenceObservabilityVerificationOutcomes {
  blocking_failure: string[];
  advisory_failure: string[];
  recovered: string[];
}

export type AgentRuntimeRequestedFixExecutionStatus =
  | "pending"
  | "assigned"
  | "running"
  | "completed"
  | "failed"
  | "blocked"
  | "cancelled";

export interface AgentRuntimeRequestedFixExecutionResult {
  requested_fix?: string;
  requested_fix_index?: number;
  execution_status: AgentRuntimeRequestedFixExecutionStatus;
  regression_outcome?: AgentRuntimeEvidenceVerificationOutcome;
  summary_preview?: string;
  result_ref?: string;
  artifact_ids: string[];
  artifact_paths: string[];
}

export interface AgentRuntimeEvidenceVerificationSummary {
  artifact_validator?: AgentRuntimeArtifactValidatorVerificationSummary;
  browser_verification?: AgentRuntimeBrowserVerificationSummary;
  gui_smoke?: AgentRuntimeGuiSmokeVerificationSummary;
  observability_verification_outcomes?: AgentRuntimeEvidenceObservabilityVerificationOutcomes;
  focus_verification_failure_outcomes: string[];
  focus_verification_recovered_outcomes: string[];
  requested_fix_execution_results?: AgentRuntimeRequestedFixExecutionResult[];
}

export interface AgentRuntimeEvidenceObservabilitySummary {
  schema_version?: string;
  known_gaps: string[];
  signal_coverage: AgentRuntimeEvidenceSignalCoverageEntry[];
  verification_summary?: AgentRuntimeEvidenceVerificationSummary;
  modality_runtime_contracts?: AgentRuntimeEvidenceModalityRuntimeContracts;
  skill_invocations: AgentRuntimeEvidenceSkillInvocation[];
  skill_searches: AgentRuntimeEvidenceSkillSearch[];
  mcp_tool_results: AgentRuntimeEvidenceMcpToolResult[];
  mcp_resource_reads: AgentRuntimeEvidenceMcpResourceRead[];
}

export interface AgentRuntimeEvidenceSkillInvocation {
  event: "skill_invocation" | string;
  skill_name: string;
  status: string;
  source_event_id: string;
  source_event_type: string;
  turn_id?: string;
  tool_call_id?: string;
  workspace_skill_source?: Record<string, unknown>;
  workspace_skill_runtime_enable?: Record<string, unknown>;
  modality_runtime_contract?: Record<string, unknown>;
}

export interface AgentRuntimeEvidenceSkillSearch {
  event: "skill_search" | string;
  query?: string;
  result_count?: number;
  snapshot_skill_count?: number;
  status: string;
  source_event_id: string;
  source_event_type: string;
  turn_id?: string;
  tool_call_id?: string;
}

export interface AgentRuntimeEvidenceMcpToolResult {
  event: "mcp_tool_result" | string;
  tool_name: string;
  status: string;
  source_event_id: string;
  source_event_type: string;
  has_structured_content: boolean;
  structured_content_keys?: string[];
  turn_id?: string;
  tool_call_id?: string;
}

export interface AgentRuntimeEvidenceMcpResourceContentRef {
  index: number;
  type?: string;
  uri?: string;
  mime_type?: string;
  text_char_count?: number;
  blob_base64_bytes?: number;
}

export interface AgentRuntimeEvidenceMcpResourceRead {
  event: "mcp_resource_read" | string;
  tool_name: string;
  uri: string;
  server?: string;
  status: string;
  source_event_id: string;
  source_event_type: string;
  mime_types: string[];
  content_count?: number;
  content_refs: AgentRuntimeEvidenceMcpResourceContentRef[];
  turn_id?: string;
  tool_call_id?: string;
}

export interface AgentRuntimeCompletionAuditRequiredEvidence {
  automation_owner: boolean;
  workspace_skill_tool_call: boolean;
  artifact_or_timeline: boolean;
  controlled_get_evidence?: boolean;
}

export interface AgentRuntimeCompletionAuditSummary {
  source: string;
  decision: string;
  owner_run_count: number;
  successful_owner_run_count: number;
  workspace_skill_tool_call_count: number;
  artifact_count: number;
  controlled_get_evidence_artifact_count?: number;
  controlled_get_evidence_executed_count?: number;
  controlled_get_evidence_scanned_artifact_count?: number;
  controlled_get_evidence_skipped_unsafe_artifact_count?: number;
  controlled_get_evidence_status_counts?: Record<string, number>;
  controlled_get_evidence_required?: boolean;
  owner_audit_statuses: string[];
  required_evidence: AgentRuntimeCompletionAuditRequiredEvidence;
  blocking_reasons: string[];
  notes: string[];
}

export type AgentRuntimeReplayArtifactKind =
  | "input"
  | "expected"
  | "grader"
  | "evidence_links";

export interface AgentRuntimeReplayArtifact {
  kind: AgentRuntimeReplayArtifactKind;
  title: string;
  relative_path: string;
  absolute_path: string;
  bytes: number;
}

export interface AgentRuntimeReplayCase {
  session_id: string;
  thread_id: string;
  workspace_id?: string;
  workspace_root: string;
  replay_relative_root: string;
  replay_absolute_root: string;
  handoff_bundle_relative_root: string;
  evidence_pack_relative_root: string;
  exported_at: string;
  thread_status: string;
  latest_turn_status?: string;
  pending_request_count: number;
  queued_turn_count: number;
  linked_handoff_artifact_count: number;
  linked_evidence_artifact_count: number;
  recent_artifact_count: number;
  artifacts: AgentRuntimeReplayArtifact[];
}

export type AgentRuntimeAnalysisArtifactKind =
  | "analysis_brief"
  | "analysis_context";

export interface AgentRuntimeAnalysisArtifact {
  kind: AgentRuntimeAnalysisArtifactKind;
  title: string;
  relative_path: string;
  absolute_path: string;
  bytes: number;
}

export interface AgentRuntimeAnalysisHandoff {
  session_id: string;
  thread_id: string;
  workspace_id?: string;
  workspace_root: string;
  analysis_relative_root: string;
  analysis_absolute_root: string;
  handoff_bundle_relative_root: string;
  evidence_pack_relative_root: string;
  replay_case_relative_root: string;
  exported_at: string;
  title: string;
  thread_status: string;
  latest_turn_status?: string;
  pending_request_count: number;
  queued_turn_count: number;
  sanitized_workspace_root: string;
  copy_prompt: string;
  artifacts: AgentRuntimeAnalysisArtifact[];
}

export type AgentRuntimeReviewDecisionArtifactKind =
  | "review_decision_markdown"
  | "review_decision_json";

export type AgentRuntimeReviewDecisionStatus =
  | "accepted"
  | "deferred"
  | "rejected"
  | "needs_more_evidence"
  | "pending_review";

export type AgentRuntimeReviewDecisionRiskLevel =
  | "low"
  | "medium"
  | "high"
  | "unknown";

export interface AgentRuntimeReviewDecisionArtifact {
  kind: AgentRuntimeReviewDecisionArtifactKind;
  title: string;
  relative_path: string;
  absolute_path: string;
  bytes: number;
}

export interface AgentRuntimeReviewDecision {
  decision_status: AgentRuntimeReviewDecisionStatus;
  decision_summary: string;
  chosen_fix_strategy: string;
  risk_level: AgentRuntimeReviewDecisionRiskLevel;
  risk_tags: string[];
  human_reviewer: string;
  reviewed_at?: string;
  followup_actions: string[];
  regression_requirements: string[];
  notes: string;
}

export interface AgentRuntimeSaveReviewDecisionRequest extends AgentRuntimeReviewDecision {
  session_id: string;
  locale?: string | null;
}

export interface AgentRuntimeReviewDecisionTemplate {
  session_id: string;
  thread_id: string;
  workspace_id?: string;
  workspace_root: string;
  review_relative_root: string;
  review_absolute_root: string;
  analysis_relative_root: string;
  analysis_absolute_root: string;
  handoff_bundle_relative_root: string;
  evidence_pack_relative_root: string;
  replay_case_relative_root: string;
  exported_at: string;
  title: string;
  thread_status: string;
  latest_turn_status?: string;
  pending_request_count: number;
  queued_turn_count: number;
  default_decision_status: string;
  verification_summary?: AgentRuntimeEvidenceVerificationSummary;
  limit_status?: string;
  capability_gap?: string;
  user_locked_capability_summary?: string;
  permission_status?: string;
  permission_confirmation_status?: string;
  permission_confirmation_request_id?: string;
  permission_confirmation_source?: string;
  permission_confirmation_summary?: string;
  decision: AgentRuntimeReviewDecision;
  decision_status_options: AgentRuntimeReviewDecisionStatus[];
  risk_level_options: AgentRuntimeReviewDecisionRiskLevel[];
  review_checklist: string[];
  analysis_artifacts: AgentRuntimeAnalysisArtifact[];
  artifacts: AgentRuntimeReviewDecisionArtifact[];
}
