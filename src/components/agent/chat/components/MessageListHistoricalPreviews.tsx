import React, { useMemo } from "react";
import { ChevronDown } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { AgentThreadItem } from "../types";
import type { ConfirmResponse, SiteSavedContentTarget } from "../types";
import type { ArtifactTimelineOpenTarget } from "../utils/artifactTimelineNavigation";
import { MarkdownRenderer } from "./MarkdownRenderer";
import { TimelineItemDetails } from "./AgentThreadTimelineItemRenderers";
import {
  formatHistoricalContentLength,
  summarizeHistoricalTimelineItems,
} from "./messageListHistoricalPreviewText";

interface HistoricalAssistantMessagePreviewProps {
  content: string;
  contentLength: number;
  variant: "compact" | "long";
  onExpand: () => void;
}

export function HistoricalAssistantMessagePreview({
  content,
  contentLength,
  variant,
  onExpand,
}: HistoricalAssistantMessagePreviewProps) {
  const { t } = useTranslation("agent");
  const isLong = variant === "long";
  const noticeKey = isLong
    ? "agentChat.messageList.historicalAssistantPreview.longNotice"
    : "agentChat.messageList.historicalAssistantPreview.compactNotice";

  return (
    <div
      data-testid={
        isLong
          ? "message-list-long-history-preview"
          : "message-list-historical-assistant-preview"
      }
      data-preview-variant={variant}
      className="space-y-3"
    >
      <div className="break-words text-[15px] leading-7 text-slate-800">
        <MarkdownRenderer
          content={content}
          renderMode="light"
          renderA2UIInline={false}
          readOnlyA2UI={true}
        />
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 rounded-2xl border border-slate-200/80 bg-slate-50/80 px-3 py-2 text-sm text-slate-600">
        <span>
          {t(noticeKey, {
            countLabel: formatHistoricalContentLength(contentLength),
          })}
        </span>
        <button
          type="button"
          className="rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 transition-colors hover:bg-slate-100"
          onClick={onExpand}
        >
          {t("agentChat.messageList.historicalAssistantPreview.expandFull")}
        </button>
      </div>
    </div>
  );
}

export const HistoricalMarkdownHydrationPreview: React.FC<{
  content: string;
}> = ({ content }) => (
  <div
    data-testid="message-list-historical-markdown-preview"
    className="break-words text-[15px] leading-7 text-slate-800"
  >
    <MarkdownRenderer
      content={content}
      renderMode="light"
      renderA2UIInline={false}
      readOnlyA2UI={true}
    />
  </div>
);

export const HistoricalTimelinePreview: React.FC<{
  items: AgentThreadItem[];
  placement: "leading" | "trailing" | "default";
  detailsDeferred?: boolean;
  startedAt?: string | null;
  completedAt?: string | null;
  onFileClick?: (fileName: string, content: string) => void;
  onOpenArtifactFromTimeline?: (target: ArtifactTimelineOpenTarget) => void;
  onOpenSavedSiteContent?: (target: SiteSavedContentTarget) => void;
  onOpenSubagentSession?: (sessionId: string) => void;
  onPermissionResponse?: (response: ConfirmResponse) => void;
  onSaveFileArtifactAsKnowledge?: (source: {
    messageId: string;
    content: string;
    sourceName?: string;
    description?: string | null;
  }) => void;
  sourceMessageId?: string;
  onExpand?: () => void;
}> = ({
  items,
  placement,
  detailsDeferred = false,
  startedAt,
  completedAt,
  onFileClick,
  onOpenArtifactFromTimeline,
  onOpenSavedSiteContent,
  onOpenSubagentSession,
  onPermissionResponse,
  onSaveFileArtifactAsKnowledge,
  sourceMessageId,
  onExpand,
}) => {
  const { t } = useTranslation("agent");
  const summary = useMemo(
    () => summarizeHistoricalTimelineItems(items),
    [items],
  );

  if (summary.stepsCount <= 0 && !detailsDeferred) {
    return null;
  }
  const metaParts = [
    summary.toolStepsCount > 0
      ? t("agentChat.messageList.historicalTimeline.toolSteps", {
          countLabel: formatHistoricalContentLength(summary.toolStepsCount),
        })
      : null,
    summary.thinkingStepsCount > 0
      ? t("agentChat.messageList.historicalTimeline.thinkingSteps", {
          countLabel: formatHistoricalContentLength(summary.thinkingStepsCount),
        })
      : null,
    summary.artifactStepsCount > 0
      ? t("agentChat.messageList.historicalTimeline.artifactSteps", {
          countLabel: formatHistoricalContentLength(summary.artifactStepsCount),
        })
      : null,
  ].filter((part): part is string => Boolean(part));
  const summaryMetaText =
    metaParts.length > 0
      ? metaParts.join(t("agentChat.messageList.historicalTimeline.separator"))
      : t("agentChat.messageList.historicalTimeline.foldedMeta");
  const metaText =
    summary.stepsCount > 0
      ? t("agentChat.messageList.historicalTimeline.meta", {
          stepCountLabel: formatHistoricalContentLength(summary.stepsCount),
          meta: summaryMetaText,
        })
      : t("agentChat.messageList.historicalTimeline.deferredMeta");
  const startedAtMs = startedAt ? new Date(startedAt).getTime() : Number.NaN;
  const completedAtMs = completedAt
    ? new Date(completedAt).getTime()
    : Number.NaN;
  const durationMs =
    Number.isFinite(startedAtMs) &&
    Number.isFinite(completedAtMs) &&
    completedAtMs >= startedAtMs
      ? completedAtMs - startedAtMs
      : null;
  const durationLabel = (() => {
    if (durationMs === null) {
      return null;
    }
    const totalSeconds = Math.max(0, Math.round(durationMs / 1000));
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
  })();
  const title = durationLabel
    ? t("agentChat.messageList.historicalTimeline.titleWithDuration", {
        duration: durationLabel,
      })
    : t("agentChat.messageList.historicalTimeline.title");
  const fileArtifacts = items.filter(
    (item): item is Extract<AgentThreadItem, { type: "file_artifact" }> =>
      item.type === "file_artifact",
  );
  const unknownItems = items.filter(
    (item): item is Extract<AgentThreadItem, { type: "unknown_item" }> =>
      item.type === "unknown_item",
  );
  const summaryRow = (
    <>
      <span className="font-medium">{title}</span>
      <span className="sr-only">{metaText}</span>
      {onExpand ? <ChevronDown className="ml-auto h-4 w-4" /> : null}
    </>
  );

  return (
    <div className="space-y-1.5">
      {onExpand ? (
        <button
          type="button"
          data-testid={`message-list-historical-timeline-preview:${placement}`}
          className="flex w-full items-center gap-1.5 border-b border-slate-200 py-2 text-left text-sm text-slate-500 transition-colors hover:text-slate-700"
          aria-label={`${title}. ${metaText}`}
          onClick={onExpand}
        >
          {summaryRow}
        </button>
      ) : (
        <div
          data-testid={`message-list-historical-timeline-preview:${placement}`}
          className="flex w-full items-center gap-1.5 border-b border-slate-200 py-2 text-left text-sm text-slate-500"
        >
          {summaryRow}
        </div>
      )}
      {fileArtifacts.length > 0 ? (
        <div data-testid="historical-file-artifact-group" className="space-y-1">
          {fileArtifacts.map((item) => (
            <TimelineItemDetails
              key={item.id}
              item={item}
              onFileClick={onFileClick}
              onOpenArtifactFromTimeline={onOpenArtifactFromTimeline}
              onOpenSavedSiteContent={onOpenSavedSiteContent}
              onOpenSubagentSession={onOpenSubagentSession}
              onPermissionResponse={onPermissionResponse}
              sourceMessageId={sourceMessageId}
              onSaveFileArtifactAsKnowledge={onSaveFileArtifactAsKnowledge}
            />
          ))}
        </div>
      ) : null}
      {unknownItems.length > 0 ? (
        <div
          data-testid="historical-unknown-item-group"
          className="space-y-1"
        >
          {unknownItems.map((item) => (
            <TimelineItemDetails key={item.id} item={item} />
          ))}
        </div>
      ) : null}
    </div>
  );
};
