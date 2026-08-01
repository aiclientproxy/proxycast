import React from "react";
import { vi } from "vitest";

export const parseAIResponseMock = vi.fn();
export const listAgentRuntimeFileCheckpointsMock = vi.fn();
export const restoreAgentRuntimeFileCheckpointMock = vi.fn();
export const openExternalUrlWithSystemBrowserMock = vi
  .fn()
  .mockResolvedValue(undefined);
export const mockMarkdownRenderer = vi.fn(
  ({
    content,
    showBlockActions,
    onQuoteContent,
    renderMode,
  }: {
    content: string;
    showBlockActions?: boolean;
    onQuoteContent?: (content: string) => void;
    renderMode?: "standard" | "light";
  }) => (
    <div
      data-testid="markdown-renderer"
      data-show-block-actions={showBlockActions ? "yes" : "no"}
      data-has-on-quote-content={onQuoteContent ? "yes" : "no"}
      data-render-mode={renderMode || "standard"}
    >
      {content}
    </div>
  ),
);

vi.mock("@/components/workspace/a2ui/parser", () => ({
  parseAIResponse: (...args: unknown[]) => parseAIResponseMock(...args),
}));

vi.mock("@/components/workspace/a2ui/taskCardPresets", () => ({
  CHAT_A2UI_TASK_CARD_PRESET: {},
  TIMELINE_A2UI_TASK_CARD_PRESET: {},
}));

vi.mock("@/lib/artifact/hooks/useDebouncedValue", () => ({
  useDebouncedValue: <T,>(value: T) => value,
}));

vi.mock("@/lib/api/agentRuntime/threadClient", () => ({
  listAgentRuntimeFileCheckpoints: (...args: unknown[]) =>
    listAgentRuntimeFileCheckpointsMock(...args),
  restoreAgentRuntimeFileCheckpoint: (...args: unknown[]) =>
    restoreAgentRuntimeFileCheckpointMock(...args),
}));

vi.mock("@/lib/api/externalUrl", () => ({
  openExternalUrlWithSystemBrowser: (...args: unknown[]) =>
    openExternalUrlWithSystemBrowserMock(...args),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    i18n: { language: "zh-CN" },
    t: (key: string, values?: Record<string, unknown>) => {
      if (key === "agentChat.fileChangesSummary.summary") {
        return `已编辑 ${values?.count ?? 0} 个文件`;
      }
      if (key === "agentChat.fileChangesSummary.review") {
        return "审核";
      }
      if (key === "agentChat.fileChangesSummary.undo") {
        return "撤销";
      }
      if (key === "agentChat.fileChangesSummary.undoUnavailable") {
        return "没有可用的文件快照";
      }
      if (key === "agentChat.fileChangesSummary.undoConfirmTitle") {
        return "撤销这些文件改动？";
      }
      if (key === "agentChat.fileChangesSummary.undoConfirmDescription") {
        return `将从运行时文件快照恢复 ${values?.count ?? 0} 个文件，并保留当前文件备份。`;
      }
      if (key === "agentChat.fileChangesSummary.undoConfirmAction") {
        return "确认撤销";
      }
      if (key === "agentChat.fileChangesSummary.undoCancel") {
        return "取消";
      }
      if (key === "agentChat.fileChangesSummary.undoRestoring") {
        return "正在撤销文件改动…";
      }
      if (key === "agentChat.fileChangesSummary.undoSuccess") {
        return `已撤销 ${values?.count ?? 0} 个文件改动`;
      }
      if (key === "agentChat.fileChangesSummary.undoFailed") {
        return `撤销失败：${values?.error ?? ""}`;
      }
      if (key.startsWith("agentChat.fileChangesSummary.undoError.")) {
        return key;
      }
      if (key === "agentChat.fileChangesSummary.expandFiles") {
        return `展开其余 ${values?.count ?? 0} 个文件`;
      }
      if (key === "agentChat.fileChangesSummary.collapseFiles") {
        return "收起文件";
      }
      if (key === "agentChat.fileChangesSummary.writing") {
        return "正在写入文件…";
      }
      if (key === "agentChat.fileChangesSummary.reviewCanvasTitle") {
        return `${values?.path ?? ""} 的变更审阅`;
      }
      if (key === "agentChat.fileChangesSummary.reviewCanvasStatus") {
        return `状态：${values?.status ?? ""}`;
      }
      if (key === "agentChat.fileChangesSummary.reviewStatus.modified") {
        return "修改";
      }
      if (key === "agentChat.fileChangesSummary.reviewStatus.added") {
        return "新增";
      }
      if (key === "agentChat.fileChangesSummary.reviewStatus.deleted") {
        return "删除";
      }
      if (key === "agentChat.fileChangesSummary.reviewStatus.unknown") {
        return "变更";
      }
      if (key === "agentChat.fileChangesSummary.reviewAdditions") {
        return `+${values?.count ?? 0} 行`;
      }
      if (key === "agentChat.fileChangesSummary.reviewDeletions") {
        return `-${values?.count ?? 0} 行`;
      }
      if (key === "agentChat.fileChangesSummary.reviewHunks") {
        return `${values?.count ?? 0} 处变更`;
      }
      if (key === "agentChat.thinkingBlock.status.completed") {
        return "已完成思考";
      }
      if (key === "agentChat.thinkingBlock.status.running") {
        return "思考中";
      }
      if (key === "agentChat.thinkingBlock.preview.structured") {
        return "在整理结构化内容";
      }
      if (key === "agentChat.processGroup.completedThinking") {
        return "已完成思考";
      }
      if (key === "agentChat.processGroup.failedSteps") {
        return `失败 ${values?.count ?? 0} 个步骤`;
      }
      if (key === "agentChat.processGroup.runningSteps") {
        return `进行中 ${values?.count ?? 0} 个步骤`;
      }
      if (key === "agentChat.processGroup.completedSteps") {
        return `已完成 ${values?.count ?? 0} 个步骤`;
      }
      if (key === "agentChat.processGroup.thinking") {
        return "思考中";
      }
      if (key === "agentChat.processGroup.toolCalls") {
        return `${values?.count ?? 0} 个工具调用`;
      }
      if (key === "agentChat.processGroup.processMessages") {
        return `${values?.count ?? 0} 条过程消息`;
      }
      if (key === "agentChat.processGroup.thinkingNotes") {
        return `${values?.count ?? 0} 条思路`;
      }
      if (key === "agentChat.processGroup.separator") {
        return "，";
      }
      if (key === "agentChat.processGroup.terminalInteractions") {
        return `已发送输入：${values?.value ?? ""}`;
      }
      if (key === "agentChat.processGroup.webSearch.section.webSearchSources") {
        return "搜索来源";
      }
      if (key === "agentChat.processGroup.webSearch.section.webFetchPages") {
        return "读取页面";
      }
      if (key === "agentChat.searchResultPreview.openAria") {
        return `打开搜索结果：${values?.title ?? ""}`;
      }
      if (key === "agentChat.searchResultPreview.emptySnippet") {
        return "暂无摘要";
      }
      if (key === "agentChat.toolCall.inline.expandDetails") {
        return "展开过程详情";
      }
      if (key === "agentChat.toolCall.inline.collapseDetails") {
        return "收起过程详情";
      }
      if (key === "agentChat.toolCall.inline.progress") {
        return `进度：${values?.message ?? ""}`;
      }
      if (key === "agentChat.toolCall.siteResult.openMarkdownPreview") {
        return "在下方预览导出 Markdown";
      }
      if (key === "agentChat.toolCall.siteResult.openSavedContent") {
        return "打开已保存内容";
      }
      if (key === "agentChat.streamingRenderer.mediaReference.title") {
        return "媒体引用";
      }
      if (key === "agentChat.streamingRenderer.mediaReference.kind.image") {
        return "图片";
      }
      if (key === "agentChat.streamingRenderer.mediaReference.kind.audio") {
        return "音频";
      }
      if (key === "agentChat.streamingRenderer.mediaReference.kind.video") {
        return "视频";
      }
      if (key === "agentChat.streamingRenderer.mediaReference.kind.file") {
        return "媒体";
      }
      if (key === "agentChat.streamingRenderer.mediaReference.mime") {
        return `${values?.mime ?? ""}`;
      }
      if (key === "agentChat.streamingRenderer.mediaReference.open") {
        return `打开媒体引用：${values?.title ?? ""}`;
      }
      if (key === "agentChat.streamingRenderer.mediaReference.reference") {
        return `引用：${values?.uri ?? ""}`;
      }
      return key;
    },
  }),
}));

vi.mock("./MarkdownRenderer", () => ({
  MarkdownRenderer: (props: {
    content: string;
    showBlockActions?: boolean;
    onQuoteContent?: (content: string) => void;
    renderMode?: "standard" | "light";
  }) => mockMarkdownRenderer(props),
}));

vi.mock("./A2UITaskCard", () => ({
  A2UITaskCard: ({
    compact,
    className,
    preview,
    onSubmit,
  }: {
    compact?: boolean;
    className?: string;
    preview?: boolean;
    onSubmit?: unknown;
  }) => (
    <div
      data-testid="a2ui-card"
      data-compact={String(compact)}
      data-preview={String(preview)}
      data-has-on-submit={onSubmit ? "yes" : "no"}
      className={className}
    />
  ),
  A2UITaskLoadingCard: ({
    compact,
    className,
  }: {
    compact?: boolean;
    className?: string;
  }) => (
    <div
      data-testid="a2ui-loading-card"
      data-compact={String(compact)}
      className={className}
    />
  ),
}));

vi.mock("./DecisionPanel", () => ({
  DecisionPanel: () => <div data-testid="decision-panel" />,
}));

vi.mock("./AgentPlanBlock", () => ({
  AgentPlanBlock: ({
    content,
    isComplete,
  }: {
    content: string;
    isComplete?: boolean;
  }) => (
    <div data-testid="agent-plan-block">
      {isComplete === false ? "进行中:" : "完成:"}
      {content}
    </div>
  ),
}));
