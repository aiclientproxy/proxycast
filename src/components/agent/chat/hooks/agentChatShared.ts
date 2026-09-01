import type { AgentExecutionStrategy } from "@/lib/api/agentExecutionRuntime";
import type {
  AgentSessionInfo,
  AutoContinueRequestPayload,
} from "@/lib/api/agentRuntime/sessionTypes";
import type {
  ModeKind,
  RuntimeSearchMode,
  UserInput,
} from "@limecloud/app-server-client";
import type {
  AgentRuntimeStatus,
  Message,
  MessageImage,
  MessageImageWorkbenchPreview,
  WriteArtifactContext,
} from "../types";
import type { ChatToolPreferences } from "../utils/chatToolPreferences";
import type { InputCapabilitySendRoute } from "../skill-selection/inputCapabilitySelection";
import { normalizeProjectId } from "../utils/topicProjectResolution";
import { normalizeExecutionStrategy } from "./agentChatCoreUtils";
import { resolveUnfinishedSessionProjection } from "../projection/unfinishedSessionProjection";
import { sanitizeMessageTextForPreview } from "../utils/messageDisplaySanitizer";
import { sanitizeGeneratedAutoTitle } from "./agentChatAutoTitleViewModel";
import type { SoulInteractionCopy } from "@/lib/soul/interactionCopy";
import type { ModelCapabilitySummary } from "@/lib/model/inferModelCapabilities";
import type {
  InterruptedInputDraftSnapshot,
  InterruptedInputRestoreRequest,
} from "./agentStreamInputRestoreTypes";
import type {
  ComposerDraftSnapshot,
  ComposerIntent,
  ComposerSubmitTarget,
} from "@/components/input-kit";

export type TaskStatus =
  | "draft"
  | "running"
  | "queued"
  | "waiting"
  | "done"
  | "failed";
export type TaskStatusReason =
  | "default"
  | "workspace_error"
  | "user_action"
  | "tool_failure";

interface RecentSessionLabelSource {
  status: TaskStatus;
  statusReason?: TaskStatusReason;
  messagesCount?: number;
}

export interface Topic {
  id: string;
  title: string;
  createdAt: Date;
  updatedAt: Date;
  workspaceId?: string | null;
  workingDir?: string | null;
  messagesCount: number;
  executionStrategy: AgentExecutionStrategy;
  status: TaskStatus;
  statusReason?: TaskStatusReason;
  queuedTurnCount?: number;
  lastPreview: string;
  section?: { id: string; name: string };
  sectionEnteredAt?: number;
  hasUnread: boolean;
  tag?: string | null;
  sourceSessionId: string;
}

export interface UseAgentChatOptions {
  systemPrompt?: string;
  clawTraceEnabled?: boolean;
  onWriteFile?: (
    content: string,
    fileName: string,
    context?: WriteArtifactContext,
  ) => void;
  workspaceId: string;
  workingDir?: string | null;
  disableSessionRestore?: boolean;
  sessionRestorePresentation?: "foreground" | "background";
  initialTopicsLoadMode?: "immediate" | "deferred";
  initialTopicsDeferredDelayMs?: number;
  initialRuntimeWarmupLoadMode?: "immediate" | "deferred";
  initialRuntimeWarmupDeferredDelayMs?: number;
  getSyncedSessionRecentPreferences?: (
    sessionId: string,
  ) => ChatToolPreferences | null;
  onOpenSubagents?: () => void;
  onRestoreInterruptedInput?: (request: InterruptedInputRestoreRequest) => void;
  soulCopy?: SoulInteractionCopy;
}

export interface SendMessageObserver {
  onTextDelta?: (delta: string, accumulated: string) => void;
  onComplete?: (content: string) => void;
  onError?: (message: string) => void;
}

export interface AssistantDraftState {
  content?: string;
  fallbackContent?: string;
  initialRuntimeStatus?: AgentRuntimeStatus;
  waitingRuntimeStatus?: AgentRuntimeStatus;
  imageWorkbenchPreview?: MessageImageWorkbenchPreview;
  preserveContent?: boolean;
}

export interface SlashSkillRequest {
  images?: MessageImage[];
  requestContext?: Record<string, unknown>;
}

export interface ThreadGoalInput {
  objective: string;
  tokenBudget?: number | null;
}

export type AgentInputMention = Extract<UserInput, { type: "mention" }>;

export interface SendMessageOptions {
  purpose?: Message["purpose"];
  observer?: SendMessageObserver;
  requestMetadata?: Record<string, unknown>;
  collaborationMode?: ModeKind;
  threadGoal?: ThreadGoalInput;
  assistantDraft?: AssistantDraftState;
  inputRestoreDraft?: InterruptedInputDraftSnapshot;
  inputMentions?: readonly AgentInputMention[];
  displayContent?: string;
  capabilityRoute?: InputCapabilitySendRoute;
  skillRequest?: SlashSkillRequest;
  providerOverride?: string;
  modelOverride?: string;
  modelCapabilitySummary?: ModelCapabilitySummary | null;
  reasoningEffort?: string;
  systemPromptOverride?: string;
  searchMode?: RuntimeSearchMode;
  explicitToolPreferences?: boolean;
  targetSessionId?: string;
  skipSessionRestore?: boolean;
  skipSessionStartHooks?: boolean;
  composerIntent?: ComposerIntent;
  composerTarget?: ComposerSubmitTarget;
  composerDraft?: ComposerDraftSnapshot;
}

export interface WorkspacePathMissingState {
  content: string;
  images: MessageImage[];
}

export interface AgentPreferences {
  providerType: string;
  model: string;
}

export interface AgentPreferenceKeys {
  providerKey: string;
  modelKey: string;
  migratedKey: string;
}

export interface SessionModelPreference {
  providerType: string;
  model: string;
}

export interface ClearMessagesOptions {
  showToast?: boolean;
  toastMessage?: string;
}

export interface LiveTaskSnapshot {
  updatedAt?: Date;
  messagesCount: number;
  status: TaskStatus;
  statusReason?: TaskStatusReason;
  lastPreview: string;
  hasUnread: boolean;
}

export interface LiveTaskStatusDescriptor {
  status: TaskStatus;
  statusReason: TaskStatusReason;
}

export type SendMessageFn = (
  content: string,
  images: MessageImage[],
  webSearch?: boolean,
  thinking?: boolean,
  skipUserMessage?: boolean,
  executionStrategyOverride?: AgentExecutionStrategy,
  modelOverride?: string,
  autoContinue?: AutoContinueRequestPayload,
  options?: SendMessageOptions,
) => Promise<void>;

export const getScopedStorageKey = (
  workspaceId: string | null | undefined,
  prefix: string,
): string => {
  const resolvedWorkspaceId = workspaceId?.trim();
  return `${prefix}_${resolvedWorkspaceId || "global"}`;
};

function normalizeTaskPreviewText(value: string) {
  return value.trim().replace(/\s+/g, " ").slice(0, 96);
}

function extractMessageTextContent(message: Message): string {
  const sanitizedContent = sanitizeMessageTextForPreview(
    message.content || "",
    {
      role: message.role,
      hasImages: Array.isArray(message.images) && message.images.length > 0,
    },
  );
  if (sanitizedContent) {
    return normalizeTaskPreviewText(sanitizedContent);
  }

  const partText = message.contentParts
    ?.filter(
      (
        part,
      ): part is Extract<
        (typeof message.contentParts)[number],
        { type: "text" | "thinking" }
      > => part.type === "text" || part.type === "thinking",
    )
    .map((part) =>
      part.type === "text"
        ? sanitizeMessageTextForPreview(part.text, {
            role: message.role,
            hasImages:
              Array.isArray(message.images) && message.images.length > 0,
          })
        : part.text,
    )
    .filter(Boolean)
    .join(" ");

  return normalizeTaskPreviewText(partText || "");
}

function extractToolCallPreview(message: Message): string {
  const latestToolCall = [...(message.toolCalls || [])].reverse().find(Boolean);
  if (!latestToolCall) {
    return "";
  }

  const toolName = latestToolCall.name?.trim() || "工具";
  if (latestToolCall.status === "failed") {
    return `执行失败：${toolName}`;
  }
  if (latestToolCall.status === "running") {
    return `正在执行：${toolName}`;
  }
  return `最近执行：${toolName}`;
}

function hasAssistantFinalText(message: Message | undefined): boolean {
  if (!message || message.role !== "assistant") {
    return false;
  }

  if (message.runtimeStatus?.phase === "failed") {
    return false;
  }

  if (message.content.trim().length > 0) {
    return true;
  }

  return (message.contentParts || []).some(
    (part) => part.type === "text" && part.text.trim().length > 0,
  );
}

function hasActiveAssistantDraft(message: Message | undefined): boolean {
  if (!message || message.role !== "assistant") {
    return false;
  }

  const hasRunningTool = message.toolCalls?.some(
    (item) => item.status === "running",
  );
  if (hasRunningTool) {
    return true;
  }

  if (!message.isThinking) {
    return false;
  }

  const runtimePhase = message.runtimeStatus?.phase;
  return runtimePhase !== "failed" && runtimePhase !== "cancelled";
}

function extractActionRequestPreview(message: Message): string {
  const pendingAction = [...(message.actionRequests || [])]
    .reverse()
    .find((item) => item.status !== "submitted");
  if (!pendingAction) {
    return "";
  }

  const questionText = pendingAction.questions
    ?.map((question) => question.question || question.header || "")
    .join(" ");

  return normalizeTaskPreviewText(
    pendingAction.prompt || questionText || "等待你确认或补充信息后继续执行。",
  );
}

function resolveLatestPendingAction(messages: Message[]) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const pendingAction = [...(messages[index].actionRequests || [])]
      .reverse()
      .find((item) => item.status !== "submitted");
    if (pendingAction) {
      return pendingAction;
    }
  }

  return null;
}

function resolvePendingActionStatusReason(
  _pendingAction: NonNullable<ReturnType<typeof resolveLatestPendingAction>>,
): TaskStatusReason {
  return "user_action";
}

export function extractTaskPreviewFromMessages(messages: Message[]): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    const actionPreview = extractActionRequestPreview(message);
    if (actionPreview) {
      return actionPreview;
    }
  }

  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    const textContent = extractMessageTextContent(message);
    if (textContent) {
      return textContent;
    }
  }

  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    const toolPreview = extractToolCallPreview(message);
    if (toolPreview) {
      return toolPreview;
    }
  }

  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    const runtimeDetail = normalizeTaskPreviewText(
      `${message.runtimeStatus?.title || ""} ${message.runtimeStatus?.detail || ""}`,
    );
    if (runtimeDetail) {
      return runtimeDetail;
    }
  }

  return "";
}

export function deriveTaskStatusFromLiveState(params: {
  messages: Message[];
  isSending: boolean;
  pendingActionCount: number;
  queuedTurnCount?: number;
  threadStatus?: string | null;
  workspaceError: boolean;
}): TaskStatus {
  return deriveTaskLiveState(params).status;
}

export function deriveTaskLiveState(params: {
  messages: Message[];
  isSending: boolean;
  pendingActionCount: number;
  queuedTurnCount?: number;
  threadStatus?: string | null;
  workspaceError: boolean;
}): LiveTaskStatusDescriptor {
  const {
    messages,
    isSending,
    pendingActionCount,
    queuedTurnCount = 0,
    threadStatus = null,
    workspaceError,
  } = params;

  if (workspaceError) {
    return {
      status: "failed",
      statusReason: "workspace_error",
    };
  }

  if (
    threadStatus === "waitingAction" ||
    threadStatus === "waiting_action" ||
    threadStatus === "waiting_request"
  ) {
    return {
      status: "waiting",
      statusReason: "user_action",
    };
  }

  if (isSending || threadStatus === "running") {
    return {
      status: "running",
      statusReason: "default",
    };
  }

  const pendingAction = resolveLatestPendingAction(messages);
  if (pendingActionCount > 0 || pendingAction) {
    return {
      status: "waiting",
      statusReason: pendingAction
        ? resolvePendingActionStatusReason(pendingAction)
        : "user_action",
    };
  }

  if (queuedTurnCount > 0 || threadStatus === "queued") {
    return {
      status: "queued",
      statusReason: "default",
    };
  }

  if (messages.length === 0) {
    return {
      status: "draft",
      statusReason: "default",
    };
  }

  const latestMessage = messages[messages.length - 1];
  const latestToolFailed = latestMessage?.toolCalls?.some(
    (item) => item.status === "failed",
  );
  if (latestToolFailed && !hasAssistantFinalText(latestMessage)) {
    return {
      status: "failed",
      statusReason: "tool_failure",
    };
  }

  if (latestMessage?.role === "user") {
    return {
      status: "running",
      statusReason: "default",
    };
  }

  if (hasActiveAssistantDraft(latestMessage)) {
    return {
      status: "running",
      statusReason: "default",
    };
  }

  return {
    status: "done",
    statusReason: "default",
  };
}

export function buildLiveTaskSnapshot(params: {
  messages: Message[];
  isSending: boolean;
  pendingActionCount: number;
  queuedTurnCount?: number;
  threadStatus?: string | null;
  workspaceError: boolean;
}): LiveTaskSnapshot {
  const {
    messages,
    isSending,
    pendingActionCount,
    queuedTurnCount,
    threadStatus,
    workspaceError,
  } = params;
  const lastMessage = messages[messages.length - 1];
  const preview = extractTaskPreviewFromMessages(messages);
  const taskState = deriveTaskLiveState({
    messages,
    isSending,
    pendingActionCount,
    queuedTurnCount,
    threadStatus,
    workspaceError,
  });

  return {
    updatedAt: lastMessage?.timestamp,
    messagesCount: messages.length,
    status: taskState.status,
    statusReason: taskState.statusReason,
    lastPreview: preview || "等待你补充任务需求后开始执行。",
    hasUnread: false,
  };
}

function resolveRecentTopicPriority(topic: RecentSessionLabelSource): number {
  if (topic.status === "waiting") {
    return 0;
  }

  if (topic.status === "queued") {
    return 1;
  }

  if (topic.status === "failed" && topic.statusReason === "workspace_error") {
    return 2;
  }

  if (topic.status === "running") {
    return 3;
  }

  if (topic.status === "done") {
    return 4;
  }

  if (topic.status === "failed") {
    return 5;
  }

  if ((topic.messagesCount ?? 0) > 0) {
    return 6;
  }

  return 7;
}

export function resolveRecentTopicCandidate(
  topics: Topic[],
  currentTopicId?: string | null,
): Topic | null {
  const candidates = topics.filter(
    (topic) => topic.id.trim() && topic.id !== currentTopicId,
  );

  if (candidates.length === 0) {
    return null;
  }

  const sorted = [...candidates].sort((left, right) => {
    const priorityDiff =
      resolveRecentTopicPriority(left) - resolveRecentTopicPriority(right);
    if (priorityDiff !== 0) {
      return priorityDiff;
    }

    const updatedDiff = right.updatedAt.getTime() - left.updatedAt.getTime();
    if (updatedDiff !== 0) {
      return updatedDiff;
    }

    return right.createdAt.getTime() - left.createdAt.getTime();
  });

  const actionable = sorted.find(
    (topic) => topic.status !== "draft" || topic.messagesCount > 0,
  );

  return actionable ?? sorted[0] ?? null;
}

export function resolveRecentTopicActionLabel(
  topic: RecentSessionLabelSource,
): string {
  switch (topic.status) {
    case "done":
      return "回看最近结果";
    case "draft":
      return "打开最近会话";
    default:
      return "继续最近会话";
  }
}

function resolveSessionTimestampDate(value: unknown): Date {
  if (value instanceof Date) {
    const time = value.getTime();
    return Number.isFinite(time) ? value : new Date(0);
  }

  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      return new Date(0);
    }
    const millis = Math.abs(value) >= 1_000_000_000_000 ? value : value * 1000;
    return new Date(millis);
  }

  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) {
      return new Date(0);
    }
    const parsedNumber = Number(trimmed);
    if (Number.isFinite(parsedNumber)) {
      return resolveSessionTimestampDate(parsedNumber);
    }
    const parsed = new Date(trimmed);
    return Number.isFinite(parsed.getTime()) ? parsed : new Date(0);
  }

  return new Date(0);
}

function buildSessionFallbackTitle(createdAt: Date): string {
  return createdAt.getTime() > 0
    ? `任务 ${createdAt.toLocaleDateString("zh-CN")}`
    : "新任务";
}

export const mapSessionToTopic = (session: AgentSessionInfo): Topic => {
  const updatedAtValue = session.updated_at ?? session.created_at;
  const messagesCount = session.messages_count ?? 0;
  const createdAt = resolveSessionTimestampDate(session.created_at);
  const updatedAt = resolveSessionTimestampDate(updatedAtValue);
  const fallbackTitle = buildSessionFallbackTitle(createdAt);
  const sessionTitle = sanitizeGeneratedAutoTitle(
    session.name || fallbackTitle,
    session.name,
  );
  const unfinishedProjection = resolveUnfinishedSessionProjection(session);
  const status: TaskStatus = unfinishedProjection
    ? unfinishedProjection.status === "waitingAction"
      ? "waiting"
      : unfinishedProjection.status === "queued"
        ? "queued"
        : "running"
    : messagesCount > 0
      ? "done"
      : "draft";
  const statusReason: TaskStatusReason =
    unfinishedProjection?.status === "waitingAction"
      ? "user_action"
      : "default";
  const lastPreview =
    unfinishedProjection?.preview ||
    (messagesCount > 0
      ? `已记录 ${messagesCount} 条消息，可继续补充或接着推进。`
      : "等待你补充任务需求后开始执行。");

  return {
    id: session.id,
    title: sessionTitle || fallbackTitle,
    createdAt,
    updatedAt,
    workspaceId: normalizeProjectId(session.workspace_id),
    workingDir: session.working_dir ?? null,
    messagesCount,
    executionStrategy: normalizeExecutionStrategy(session.execution_strategy),
    status,
    statusReason,
    queuedTurnCount: session.queued_turn_count,
    lastPreview,
    section: session.section,
    sectionEnteredAt: session.section_entered_at,
    hasUnread: false,
    tag: null,
    sourceSessionId: session.id,
  };
};
