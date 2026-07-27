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
  structuredContent?: unknown;
  structured_content?: unknown;
  metadata?: unknown;
}

export interface AgentThreadCommandExecutionItem extends AgentThreadItemBase {
  type: "command_execution";
  command: string;
  cwd: string;
  source?: string;
  process_id?: string;
  duration_ms?: number;
  aggregated_output?: string;
  exit_code?: number;
  error?: string;
}

export type AgentThreadPatchChangeKind =
  | { type: "add" }
  | { type: "delete" }
  | { type: "update"; move_path?: string | null };

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
  | AgentThreadErrorItem
  | AgentThreadTurnSummaryItem;

export interface AgentToolProgressPayload {
  message?: string;
  progress?: number;
  total?: number;
  metadata?: Record<string, unknown>;
}
export interface AgentToolCallState {
  id: string;
  name: string;
  arguments?: string;
  status: "running" | "completed" | "failed";
  result?: AgentToolExecutionResult;
  metadata?: Record<string, unknown>;
  progress?: AgentToolProgressPayload & { updatedAt?: Date };
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
