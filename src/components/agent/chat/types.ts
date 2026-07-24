import type {
  AgentContextTraceStep as ContextTraceStep,
  AgentRuntimeStatusMetadata,
  AgentToolCallState as ToolCallState,
  AgentTokenUsage as TokenUsage,
} from "@/lib/api/agentProtocol";
import type { ActionRequired } from "@/lib/api/agentActionTypes";
import type { Artifact, ArtifactStatus } from "@/lib/artifact/types";
import type { AgentUiProjectionEvent } from "./projection/agentUiEventProjection";
import type { InputCapabilitySendRoute } from "./skill-selection/inputCapabilitySelection";

export type {
  AgentThreadItem,
  AgentThreadItemStatus,
  AgentThreadTurn,
} from "@/lib/api/agentProtocol";
export type {
  ActionRequestGovernanceMeta,
  ActionRequired,
  ActionRequiredScope,
  ApprovalDecision,
  ConfirmResponse,
  Question,
  QuestionOption,
} from "@/lib/api/agentActionTypes";

export interface MessageImage {
  data: string;
  mediaType: string;
  sourceUri?: string;
  sourcePath?: string;
  previewUrl?: string;
  metadata?: Record<string, unknown>;
  index?: number;
}

export interface MessagePathReference {
  id: string;
  path: string;
  name: string;
  isDir: boolean;
  size?: number | null;
  mimeType?: string | null;
  source?: "file_manager" | "system_drop";
}

export interface ImageStoryboardSlot {
  slotId: string;
  slotIndex: number;
  label?: string | null;
  prompt?: string | null;
  shotType?: string | null;
  status?: string | null;
}

export type ImageRuntimeContractRoutingOutcome =
  | "accepted"
  | "failed"
  | "blocked";

export interface ImageRuntimeContractSnapshot {
  contractKey?: string | null;
  routingSlot?: string | null;
  providerId?: string | null;
  model?: string | null;
  routingEvent?: string | null;
  routingOutcome?: ImageRuntimeContractRoutingOutcome | string | null;
  failureCode?: string | null;
  modelCapabilityAssessmentSource?: string | null;
  modelSupportsImageGeneration?: boolean | null;
  limecorePolicySnapshotStatus?: string | null;
  limecorePolicyDecision?: string | null;
  limecorePolicyDecisionSource?: string | null;
  limecorePolicyDecisionScope?: string | null;
  limecorePolicyDecisionReason?: string | null;
  limecorePolicyMissingInputs?: string[];
  limecorePolicyPendingHitRefs?: string[];
  limecorePolicyEvaluationStatus?: string | null;
  limecorePolicyEvaluationDecision?: string | null;
  limecorePolicyEvaluationDecisionSource?: string | null;
  limecorePolicyEvaluationDecisionScope?: string | null;
  limecorePolicyEvaluationDecisionReason?: string | null;
  limecorePolicyEvaluationBlockingRefs?: string[];
  limecorePolicyEvaluationAskRefs?: string[];
  limecorePolicyEvaluationPendingRefs?: string[];
}

export interface ImageGenerationSoulMetadata {
  surface?: string | null;
  phase?: string | null;
  styleLevel?: string | null;
  riskLevel?: string | null;
  toneVariant?: string | null;
  profileId?: string | null;
  packId?: string | null;
  titleStyleLevel?: string | null;
  parameterSummaryStyleLevel?: string | null;
  runningStatusStyleLevel?: string | null;
  assistantIntroStyleLevel?: string | null;
  completionCaptionStyleLevel?: string | null;
  mediaArtifactStyleLevel?: string | null;
  formalArtifactVoiceSource?: string | null;
  productSoulDefault?: string | null;
}

export interface MessageImageWorkbenchPreview {
  taskId: string;
  prompt: string;
  mode?: "generate" | "edit" | "variation";
  status: "running" | "complete" | "partial" | "failed" | "cancelled";
  projectId?: string | null;
  contentId?: string | null;
  taskFilePath?: string | null;
  artifactPath?: string | null;
  imageUrl?: string | null;
  previewImages?: string[];
  imageCount?: number;
  expectedImageCount?: number;
  providerName?: string | null;
  modelName?: string | null;
  caption?: string | null;
  layoutHint?: string | null;
  storyboardSlots?: ImageStoryboardSlot[];
  sourceImageUrl?: string | null;
  sourceImagePrompt?: string | null;
  sourceImageRef?: string | null;
  sourceImageCount?: number;
  requestMetadata?: Record<string, unknown>;
  size?: string;
  phase?: string | null;
  statusMessage?: string | null;
  retryable?: boolean;
  attemptCount?: number;
  placeholderText?: string | null;
  runtimeContract?: ImageRuntimeContractSnapshot | null;
  workflowRun?: ImageCommandRunSnapshot | null;
  soulMetadata?: ImageGenerationSoulMetadata | null;
}

export interface ImageCommandRunSnapshot {
  runId: string;
  sessionId?: string | null;
  threadId?: string | null;
  turnId?: string | null;
  workflowKey?: string | null;
  title: string;
  summary: string;
  requestedCount: number;
  status:
    | "requires_parameters"
    | "queued"
    | "running"
    | "succeeded"
    | "partial"
    | "failed";
  steps: ImageCommandRunStep[];
  branches: ImageGenerationBranch[];
  nextActions: ImageCommandNextAction[];
}

export interface ImageCommandRunStep {
  id: string;
  title: string;
  status: "pending" | "running" | "succeeded" | "failed";
  detail?: string | null;
}

export interface ImageGenerationBranch {
  branchId: string;
  title: string;
  prompt: string;
  taskId?: string | null;
  artifactPath?: string | null;
  status: "queued" | "running" | "succeeded" | "failed" | "retryable";
  previewUrl?: string | null;
  failureReason?: string | null;
  slotId?: string | null;
  shotType?: string | null;
}

export type ImageCommandNextAction =
  | { type: "retry_branch"; branchId: string }
  | { type: "generate_more"; branchId?: string | null }
  | { type: "open_workbench"; taskId?: string | null }
  | { type: "apply_to_document"; slotId: string };

export interface MessageImageWorkbenchPreviewSelection {
  imageUrl?: string | null;
  imageIndex?: number | null;
}

export type MessageTaskPreviewStatus = MessageImageWorkbenchPreview["status"];

export interface MessageVideoTaskPreview {
  kind: "video_generate";
  taskId: string;
  taskType: "video_generate";
  prompt: string;
  status: MessageTaskPreviewStatus;
  projectId?: string | null;
  contentId?: string | null;
  videoUrl?: string | null;
  thumbnailUrl?: string | null;
  durationSeconds?: number;
  aspectRatio?: string;
  resolution?: string;
  providerId?: string | null;
  model?: string | null;
  progress?: number | null;
  phase?: string | null;
  statusMessage?: string | null;
  retryable?: boolean;
}

export interface MessageTaskPreviewImageCandidate {
  id: string;
  thumbnailUrl: string;
  contentUrl?: string | null;
  hostPageUrl?: string | null;
  width?: number;
  height?: number;
  name?: string;
}

export interface MessageTranscriptSegment {
  id: string;
  index: number;
  startMs?: number | null;
  endMs?: number | null;
  speaker?: string | null;
  text: string;
}

export interface MessageGenericTaskPreview {
  kind:
    | "audio_generate"
    | "broadcast_generate"
    | "modal_resource_search"
    | "transcription_generate"
    | "url_parse"
    | "typesetting";
  taskId: string;
  taskType: MessageGenericTaskPreview["kind"];
  prompt: string;
  title?: string;
  status: MessageTaskPreviewStatus;
  projectId?: string | null;
  contentId?: string | null;
  artifactPath?: string | null;
  providerId?: string | null;
  model?: string | null;
  phase?: string | null;
  statusMessage?: string | null;
  retryable?: boolean;
  metaItems?: string[];
  imageCandidates?: MessageTaskPreviewImageCandidate[];
  taskFilePath?: string | null;
  audioUrl?: string | null;
  mimeType?: string | null;
  durationMs?: number | null;
  sourceText?: string | null;
  voice?: string | null;
  transcriptPath?: string | null;
  sourcePath?: string | null;
  sourceUrl?: string | null;
  language?: string | null;
  outputFormat?: string | null;
  transcriptText?: string | null;
  transcriptSegments?: MessageTranscriptSegment[];
  errorCode?: string | null;
  errorMessage?: string | null;
}

export type MessageTaskPreview =
  | MessageVideoTaskPreview
  | MessageGenericTaskPreview;

export type MessagePreviewTarget =
  | {
      kind: "image_workbench";
      preview: MessageImageWorkbenchPreview;
      selection?: MessageImageWorkbenchPreviewSelection;
    }
  | {
      kind: "message_attachment";
      attachment: MessageImage;
      index: number;
    }
  | {
      kind: "media_reference";
      reference: MessageMediaReference;
      index: number;
    }
  | {
      kind: "task";
      preview: MessageTaskPreview;
    };

/**
 * 内容片段类型（用于交错显示）
 *
 * 参考 agent 框架的 MessageContent 设计：
 * - text: 文本内容片段
 * - thinking: 推理内容片段（DeepSeek R1 等模型）
 * - tool_use: 工具调用（包含状态和结果）
 * - action_required: 权限确认请求
 */
interface AgentUiProjectionContentPartMeta {
  agentUiEvent?: AgentUiProjectionEvent;
  metadata?: Record<string, unknown>;
}

export interface MessageMediaReference {
  kind?: string;
  uri: string;
  refId?: string;
  mimeType?: string;
  title?: string;
  caption?: string;
  sidecarRef?: unknown;
  sourceUri?: string;
  sourcePath?: string;
  previewUrl?: string;
  sha256?: string;
  byteSize?: number;
}

export type ContentPart =
  | ({ type: "text"; text: string } & AgentUiProjectionContentPartMeta)
  | ({ type: "thinking"; text: string } & AgentUiProjectionContentPartMeta)
  | ({
      type: "tool_use";
      toolCall: ToolCallState;
    } & AgentUiProjectionContentPartMeta)
  | ({
      type: "action_required";
      actionRequired: ActionRequired;
    } & AgentUiProjectionContentPartMeta)
  | ({
      type: "file_changes_batch";
      aggregate: import("./utils/fileChangeSummary").FileChangesAggregate;
    } & AgentUiProjectionContentPartMeta)
  | ({
      type: "media_reference";
      reference: MessageMediaReference;
    } & AgentUiProjectionContentPartMeta);

export type BrowserTaskRequirement =
  | "optional"
  | "required"
  | "required_with_user_step";

export interface SiteSavedContentTarget {
  projectId: string;
  contentId: string;
  title?: string;
  preferredTarget?: "saved_content" | "project_file";
  projectFile?: {
    relativePath: string;
  };
}

export type PendingA2UISource =
  | {
      kind: "assistant_message";
      messageId: string;
    }
  | {
      kind: "action_request";
      requestId: string;
    }
  | {
      kind: "scene_gate";
      gateKey: string;
      sceneKey: string;
      messageId?: undefined;
    }
  | {
      kind: "service_skill";
      skillId: string;
      requestKey: string;
      messageId?: undefined;
    };

export type ArtifactWriteSource =
  | "tool_start"
  | "artifact_snapshot"
  | "tool_result"
  | "message_content";

export type ArtifactWritePhase =
  | "preparing"
  | "streaming"
  | "persisted"
  | "completed"
  | "failed";

export interface ArtifactWriteMetadata extends Record<string, unknown> {
  writePhase?: ArtifactWritePhase;
  previewText?: string;
  latestChunk?: string;
  isPartial?: boolean;
  lastUpdateSource?: ArtifactWriteSource;
}

export interface WriteArtifactContext {
  artifact?: Artifact;
  artifactId?: string;
  source?: ArtifactWriteSource;
  sourceMessageId?: string;
  status?: ArtifactStatus;
  metadata?: ArtifactWriteMetadata;
}

export interface AgentRuntimeStatus {
  phase:
    | "preparing"
    | "routing"
    | "context"
    | "permission_review"
    | "retrying"
    | "continuing"
    | "synthesizing"
    | "failed"
    | "cancelled";
  title: string;
  detail: string;
  checkpoints?: string[];
  metadata?: AgentRuntimeStatusMetadata;
}

export interface BrowserAssistSessionState {
  sessionId?: string;
  profileKey?: string;
  url?: string;
  title?: string;
  targetId?: string;
  transportKind?: string;
  lifecycleState?: string;
  controlMode?: string;
  source: "tool_call" | "runtime_launch" | "artifact_restore";
  updatedAt: number;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  /** 完整文本内容（向后兼容） */
  content: string;
  images?: MessageImage[];
  timestamp: Date;
  isThinking?: boolean;
  thinkingContent?: string;
  search_results?: any[]; // For potential future use
  /** 工具调用列表（assistant 消息可能包含） - 向后兼容 */
  toolCalls?: ToolCallState[];
  /** Token 使用量（响应完成后） */
  usage?: TokenUsage;
  /** 本轮发送/草稿关联的请求元数据 */
  requestMetadata?: Record<string, unknown>;
  /** 权限确认请求列表 */
  actionRequests?: ActionRequired[];
  /**
   * 交错内容列表（按事件到达顺序排列）
   * 如果存在且非空，StreamingRenderer 会按顺序渲染
   * 否则回退到 content + toolCalls 渲染方式
   */
  contentParts?: ContentPart[];
  /** 上下文准备轨迹（可选） */
  contextTrace?: ContextTraceStep[];
  /** 与当前消息关联的产物列表 */
  artifacts?: Artifact[];
  /** 图片工作台消息卡预览 */
  imageWorkbenchPreview?: MessageImageWorkbenchPreview;
  /** 图片任务运行时合约（旧兼容字段，优先从 imageWorkbenchPreview.runtimeContract 读取） */
  imageRuntimeContract?: ImageRuntimeContractSnapshot;
  /** 通用任务消息卡预览 */
  taskPreview?: MessageTaskPreview;
  /** 首个流式事件到达前的本地运行态 */
  runtimeStatus?: AgentRuntimeStatus;
  /** 当前消息显式绑定的运行时回合 ID，用于避免执行过程跨轮串线 */
  runtimeTurnId?: string;
  /** 完成后仍需保留的内联过程来源。 */
  inlineProcessRetention?: "skill";
  /** 消息用途（用于跳过特定副作用） */
  purpose?: "content_review" | "text_stylize" | "style_rewrite" | "style_audit";
  /** 用户发送时显式选择的输入能力，用于历史气泡还原 @ Skill 等标签。 */
  inputCapabilityRoute?: InputCapabilitySendRoute;
}

export interface ChatSession {
  id: string;
  title: string;
  providerType: string;
  model: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}
