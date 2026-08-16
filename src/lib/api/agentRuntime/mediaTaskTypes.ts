import type { AgentRuntimeGeneratedTitleResult } from "./sessionTypes";

export interface CreateImageGenerationTaskArtifactRequest {
  projectRootPath: string;
  prompt: string;
  title?: string;
  titleGenerationResult?: AgentRuntimeGeneratedTitleResult | null;
  personaContext?: Record<string, unknown>;
  persona_context?: Record<string, unknown>;
  presentation?: Record<string, unknown>;
  tasteContext?: Record<string, unknown>;
  taste_context?: Record<string, unknown>;
  mode?: "generate" | "edit" | "variation";
  rawText?: string;
  layoutHint?: string;
  size?: string;
  aspectRatio?: string;
  count?: number;
  usage?: string;
  style?: string;
  providerId?: string;
  model?: string;
  executorMode?: "images_api" | "responses_image_generation";
  outerModel?: string;
  sessionId?: string;
  threadId?: string;
  turnId?: string;
  projectId?: string;
  contentId?: string;
  entrySource?: string;
  modalityContractKey?: "image_generation";
  modality?: "image";
  requiredCapabilities?: string[];
  routingSlot?: "image_generation_model";
  runtimeContract?: Record<string, unknown>;
  requestedTarget?: "generate" | "cover";
  slotId?: string;
  anchorHint?: string;
  anchorSectionTitle?: string;
  anchorText?: string;
  targetOutputId?: string;
  targetOutputRefId?: string;
  referenceImages?: string[];
  storyboardSlots?: Array<{
    slotId?: string;
    label?: string;
    prompt: string;
    shotType?: string;
  }>;
}

export interface CreateAudioGenerationTaskArtifactRequest {
  projectRootPath: string;
  sourceText: string;
  title?: string;
  rawText?: string;
  voice?: string;
  voiceStyle?: string;
  targetLanguage?: string;
  mimeType?: string;
  audioPath?: string;
  durationMs?: number;
  providerId?: string;
  model?: string;
  sessionId?: string;
  threadId?: string;
  turnId?: string;
  projectId?: string;
  contentId?: string;
  entrySource?: string;
  modalityContractKey?: "voice_generation";
  modality?: "audio";
  requiredCapabilities?: string[];
  routingSlot?: "voice_generation_model";
  runtimeContract?: Record<string, unknown>;
  requestedTarget?: "voice" | "dubbing";
  outputPath?: string;
}

export interface CreateVideoGenerationTaskArtifactRequest {
  projectRootPath: string;
  prompt: string;
  title?: string;
  rawText?: string;
  aspectRatio?: string;
  resolution?: string;
  duration?: number;
  imageUrl?: string;
  endImageUrl?: string;
  seed?: number;
  generateAudio?: boolean;
  cameraFixed?: boolean;
  providerId?: string;
  model?: string;
  sessionId?: string;
  threadId?: string;
  turnId?: string;
  projectId?: string;
  contentId?: string;
  entrySource?: string;
  modalityContractKey?: "video_generation";
  modality?: "video";
  requiredCapabilities?: string[];
  routingSlot?: "video_generation_model";
  runtimeContract?: Record<string, unknown>;
  requestedTarget?: "video";
  outputPath?: string;
}

export interface CreateTranscriptionTaskArtifactRequest {
  projectRootPath: string;
  prompt?: string;
  title?: string;
  rawText?: string;
  sourcePath?: string;
  sourceUrl?: string;
  language?: string;
  outputFormat?: string;
  speakerLabels?: boolean;
  timestamps?: boolean;
  providerId?: string;
  model?: string;
  sessionId?: string;
  threadId?: string;
  turnId?: string;
  projectId?: string;
  contentId?: string;
  entrySource?: string;
  modalityContractKey?: "audio_transcription";
  modality?: "audio";
  requiredCapabilities?: string[];
  routingSlot?: "audio_transcription_model";
  runtimeContract?: Record<string, unknown>;
  requestedTarget?: "transcript";
  outputPath?: string;
}

export interface CompleteAudioGenerationTaskArtifactRequest {
  projectRootPath: string;
  taskRef: string;
  audioPath: string;
  mimeType?: string;
  durationMs?: number;
  providerId?: string;
  model?: string;
}

export interface CompletedImageGenerationTaskInput {
  url: string;
  prompt?: string;
  revisedPrompt?: string;
  size?: string;
  providerId?: string;
  model?: string;
  slotId?: string;
  slotIndex?: number;
  slotPrompt?: string;
}

export interface CompleteImageGenerationTaskArtifactRequest {
  projectRootPath: string;
  taskRef: string;
  images: CompletedImageGenerationTaskInput[];
  responses?: Array<Record<string, unknown>>;
  failures?: Array<Record<string, unknown>>;
  providerId?: string;
  model?: string;
  executorMode?: string;
  responseId?: string;
  status?: string;
}

export interface MediaTaskArtifactRecord {
  task_id: string;
  task_type: string;
  task_family: string;
  title?: string | null;
  summary?: string | null;
  payload: Record<string, unknown>;
  status: string;
  normalized_status: string;
  created_at: string;
  updated_at?: string | null;
  current_attempt_id?: string | null;
  idempotency_key?: string | null;
  retry_count?: number;
  result?: unknown;
  last_error?: Record<string, unknown> | null;
  attempts?: Array<Record<string, unknown>>;
  relationships?: Record<string, unknown>;
  progress?: Record<string, unknown>;
  ui_hints?: Record<string, unknown>;
}

export interface MediaTaskArtifactOutput {
  success: boolean;
  task_id: string;
  task_type: string;
  task_family: string;
  status: string;
  normalized_status: string;
  current_attempt_id?: string | null;
  path: string;
  absolute_path: string;
  artifact_path: string;
  absolute_artifact_path: string;
  reused_existing: boolean;
  idempotency_key?: string | null;
  record: MediaTaskArtifactRecord;
}

export interface MediaTaskLookupRequest {
  projectRootPath: string;
  taskRef: string;
}

export interface ListMediaTaskArtifactsRequest {
  projectRootPath: string;
  status?: string;
  taskFamily?: string;
  taskType?: string;
  modalityContractKey?: string;
  routingOutcome?: "accepted" | "failed" | "blocked";
  limit?: number;
}

export interface MediaTaskListFilters {
  status?: string | null;
  task_family?: string | null;
  task_type?: string | null;
  modality_contract_key?: string | null;
  routing_outcome?: string | null;
  limit?: number | null;
}

export interface MediaTaskModalityRuntimeContractIndexEntry {
  task_id: string;
  task_type: string;
  normalized_status: string;
  contract_key?: string | null;
  entry_key?: string | null;
  thread_id?: string | null;
  turn_id?: string | null;
  content_id?: string | null;
  modality?: string | null;
  skill_id?: string | null;
  model_id?: string | null;
  cost_state?: string | null;
  limit_state?: string | null;
  estimated_cost_class?: string | null;
  limit_event_kind?: string | null;
  quota_low?: boolean | null;
  routing_slot?: string | null;
  provider_id?: string | null;
  model?: string | null;
  execution_profile_key?: string | null;
  executor_adapter_key?: string | null;
  executor_kind?: string | null;
  executor_binding_key?: string | null;
  limecore_policy_refs: string[];
  limecore_policy_snapshot_status?: string | null;
  limecore_policy_decision?: string | null;
  limecore_policy_decision_source?: string | null;
  limecore_policy_decision_scope?: string | null;
  limecore_policy_decision_reason?: string | null;
  limecore_policy_evaluation_status?: string | null;
  limecore_policy_evaluation_decision?: string | null;
  limecore_policy_evaluation_decision_source?: string | null;
  limecore_policy_evaluation_decision_scope?: string | null;
  limecore_policy_evaluation_decision_reason?: string | null;
  limecore_policy_evaluation_blocking_refs?: string[];
  limecore_policy_evaluation_ask_refs?: string[];
  limecore_policy_evaluation_pending_refs?: string[];
  limecore_policy_unresolved_refs?: string[];
  limecore_policy_missing_inputs?: string[];
  limecore_policy_pending_hit_refs?: string[];
  limecore_policy_value_hits?: unknown[];
  limecore_policy_value_hit_count?: number;
  routing_event: string;
  routing_outcome: string;
  failure_code?: string | null;
  model_capability_assessment_source?: string | null;
  model_supports_image_generation?: boolean | null;
  audio_output_status?: string | null;
  audio_output_path?: string | null;
  audio_output_mime_type?: string | null;
  audio_output_duration_ms?: number | null;
  audio_output_error_code?: string | null;
  audio_output_retryable?: boolean | null;
  transcript_status?: string | null;
  transcript_path?: string | null;
  transcript_source_url?: string | null;
  transcript_source_path?: string | null;
  transcript_language?: string | null;
  transcript_output_format?: string | null;
  transcript_error_code?: string | null;
  transcript_retryable?: boolean | null;
}

export interface MediaTaskRoutingOutcomeCount {
  outcome: string;
  count: number;
}

export interface MediaTaskAudioOutputStatusCount {
  status: string;
  count: number;
}

export interface MediaTaskTranscriptStatusCount {
  status: string;
  count: number;
}

export interface MediaTaskLimeCorePolicySnapshotStatusCount {
  status: string;
  count: number;
}

export interface MediaTaskLimeCorePolicyEvaluationStatusCount {
  status: string;
  count: number;
}

export interface MediaTaskModalityRuntimeContractIndex {
  snapshot_count: number;
  contract_keys: string[];
  entry_keys?: string[];
  thread_ids?: string[];
  turn_ids?: string[];
  content_ids?: string[];
  modalities?: string[];
  skill_ids?: string[];
  model_ids?: string[];
  cost_states?: string[];
  limit_states?: string[];
  estimated_cost_classes?: string[];
  limit_event_kinds?: string[];
  quota_low_count?: number;
  execution_profile_keys: string[];
  executor_adapter_keys: string[];
  executor_kinds?: string[];
  executor_binding_keys?: string[];
  limecore_policy_refs: string[];
  limecore_policy_snapshot_count: number;
  limecore_policy_snapshot_statuses: MediaTaskLimeCorePolicySnapshotStatusCount[];
  limecore_policy_decisions: string[];
  limecore_policy_decision_sources?: string[];
  limecore_policy_evaluation_statuses?: MediaTaskLimeCorePolicyEvaluationStatusCount[];
  limecore_policy_evaluation_decisions?: string[];
  limecore_policy_evaluation_decision_sources?: string[];
  limecore_policy_evaluation_blocking_refs?: string[];
  limecore_policy_evaluation_ask_refs?: string[];
  limecore_policy_evaluation_pending_refs?: string[];
  limecore_policy_unresolved_refs?: string[];
  limecore_policy_missing_inputs?: string[];
  limecore_policy_pending_hit_refs?: string[];
  limecore_policy_value_hit_count?: number;
  blocked_count: number;
  routing_outcomes: MediaTaskRoutingOutcomeCount[];
  model_registry_assessment_count: number;
  audio_output_count: number;
  audio_output_statuses: MediaTaskAudioOutputStatusCount[];
  audio_output_error_codes: string[];
  transcript_count: number;
  transcript_statuses: MediaTaskTranscriptStatusCount[];
  transcript_error_codes: string[];
  snapshots: MediaTaskModalityRuntimeContractIndexEntry[];
}

export interface ListMediaTaskArtifactsOutput {
  success: boolean;
  workspace_root: string;
  artifact_root: string;
  filters: MediaTaskListFilters;
  total: number;
  modality_runtime_contracts: MediaTaskModalityRuntimeContractIndex;
  tasks: MediaTaskArtifactOutput[];
}
