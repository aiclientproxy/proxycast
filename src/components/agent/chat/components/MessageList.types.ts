import type React from "react";
import type { Artifact } from "@/lib/artifact/types";
import type { A2UIFormData } from "@/components/workspace/a2ui/types";
import type { AgentRuntimeThreadReadModel } from "@/lib/api/agentRuntime/sessionTypes";
import type { ThreadRevertRequest } from "@/lib/api/threadRevert";
import type { CanonicalChildThreadSummary } from "../projection/canonicalChildThreadSummary";
import type { ArtifactTimelineOpenTarget } from "../utils/artifactTimelineNavigation";
import type { buildMessageRenderGroupsProjection } from "../projection/messageTimelineRenderProjection";
import type { SearchResultPreviewItem } from "../utils/searchResultPreview";
import type {
  ActionRequired,
  AgentThreadItem,
  AgentThreadTurn,
  ConfirmResponse,
  Message,
  MessagePreviewTarget,
  PendingA2UISource,
  SiteSavedContentTarget,
  WriteArtifactContext,
} from "../types";

export type MessageListRenderGroup = ReturnType<
  typeof buildMessageRenderGroupsProjection
>[number];

export interface MessageListProps {
  sessionId?: string | null;
  messages: Message[];
  leadingContent?: React.ReactNode;
  trailingContent?: React.ReactNode;
  emptyStateVariant?: "none" | "task-center";
  turns?: readonly AgentThreadTurn[];
  threadItems?: readonly AgentThreadItem[];
  currentTurnId?: string | null;
  threadRead?: AgentRuntimeThreadReadModel | null;
  pendingActions?: readonly ActionRequired[];
  submittedActionsInFlight?: readonly ActionRequired[];
  canonicalChildren?: CanonicalChildThreadSummary[];
  sessionHistoryWindow?: {
    loadedEntries?: number;
    loadedTurns?: number;
    loadedItems?: number;
    loadedMessages?: number;
    totalMessages?: number | null;
    hasMore?: boolean;
    isFullyLoaded?: boolean;
    isLoadingFull: boolean;
    error?: string | null;
  } | null;
  onLoadFullHistory?: () => void | Promise<void>;
  isSending?: boolean;
  assistantLabel?: string;
  onDeleteMessage?: (id: string) => void;
  onEditMessage?: (id: string, content: string) => void;
  onQuoteMessage?: (content: string, id: string) => void;
  onRevertThread?: (request: ThreadRevertRequest) => Promise<void>;
  /** A2UI 表单提交回调 */
  onA2UISubmit?: (formData: A2UIFormData, messageId: string) => void;
  /** 是否渲染消息内联 A2UI */
  renderA2UIInline?: boolean;
  /** A2UI 表单数据映射（按消息 ID 索引） */
  a2uiFormDataMap?: Record<string, { formId: string; formData: A2UIFormData }>;
  /** A2UI 表单数据变化回调（用于持久化） */
  onA2UIFormChange?: (formId: string, formData: A2UIFormData) => void;
  /** 文件写入回调 */
  onWriteFile?: (
    content: string,
    fileName: string,
    context?: WriteArtifactContext,
  ) => void;
  /** 文件点击回调 */
  onFileClick?: (fileName: string, content: string) => void;
  /** 时间线内 artifact 精确跳转 */
  onOpenArtifactFromTimeline?: (target: ArtifactTimelineOpenTarget) => void;
  /** 打开 URL 来源预览 */
  onOpenUrlPreview?: (item: SearchResultPreviewItem) => void;
  /** 打开站点能力已保存内容 */
  onOpenSavedSiteContent?: (target: SiteSavedContentTarget) => void;
  /** Artifact 点击回调 */
  onArtifactClick?: (artifact: Artifact) => void;
  /** 打开消息结果预览 */
  onOpenMessagePreview?: (
    target: MessagePreviewTarget,
    message: Message,
  ) => void;
  /** 将助手结果沉淀为技能草稿 */
  onSaveMessageAsSkill?: (source: {
    messageId: string;
    content: string;
  }) => void;
  /** 将助手结果保存到项目资料 */
  onSaveMessageAsKnowledge?: (source: {
    messageId: string;
    content: string;
    sourceName?: string;
    description?: string | null;
  }) => void;
  /** 打开子代理会话 */
  onOpenSubagentSession?: (sessionId: string) => void;
  /** 权限确认响应回调 */
  onPermissionResponse?: (response: ConfirmResponse) => void;
  /** 是否折叠代码块（当画布打开时） */
  collapseCodeBlocks?: boolean;
  /** 按代码块决定是否折叠 */
  shouldCollapseCodeBlock?: (language: string, code: string) => boolean;
  /** 代码块点击回调（用于在画布中显示） */
  onCodeBlockClick?: (language: string, code: string) => void;
  /** 是否将待处理问答提升为对话内 A2UI 表单 */
  promoteActionRequestsToA2UI?: boolean;
  /** 会话是否仍在自动恢复 */
  isRestoringSession?: boolean;
  /** 中断当前执行 */
  onInterruptCurrentTurn?: () => void | Promise<void>;
  /** 重新拉起当前最重要的待处理请求 */
  onReplayPendingRequest?: (
    requestId: string,
    assistantMessageId: string,
  ) => boolean | Promise<boolean>;
  /** 是否压缩左侧留白，适用于工作台右栏 */
  compactLeadingSpacing?: boolean;
  /** 需要高亮的 timeline item */
  focusedTimelineItemId?: string | null;
  /** 触发 timeline item 聚焦的请求序号 */
  timelineFocusRequestKey?: number;
  /** 当前仍可在消息正文提交的 A2UI 来源 */
  activePendingA2UISource?: PendingA2UISource | null;
  /** 当前会话的 provider 选择器 */
  providerType?: string;
}
