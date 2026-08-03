import React, { useMemo } from "react";
import { ChevronDown } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { AgentThreadItem } from "../types";
import type { ConfirmResponse, SiteSavedContentTarget } from "../types";
import type { ArtifactTimelineOpenTarget } from "../utils/artifactTimelineNavigation";
import { isHiddenConversationArtifactPath } from "../utils/internalArtifactVisibility";
import { MarkdownRenderer } from "./MarkdownRenderer";
import { AgentThreadTimelineAttachmentList } from "./AgentThreadTimelineAttachmentList";
import { TimelineItemDetails } from "./AgentThreadTimelineItemRenderers";
import { hasTimelineFileChangeEvidence } from "./AgentThreadTimelineFileChangesCard";
import {
  formatHistoricalContentLength,
  formatHistoricalTimelineDuration,
  resolveHistoricalTimelineDurationMs,
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
  const durationLabel = formatHistoricalTimelineDuration(
    resolveHistoricalTimelineDurationMs(items, startedAt, completedAt),
  );
  const title = durationLabel
    ? t("agentChat.messageList.historicalTimeline.titleWithDuration", {
        duration: durationLabel,
      })
    : t("agentChat.messageList.historicalTimeline.title");
  const evidencePaths = Array.from(
    new Set(
      items
        .filter(hasTimelineFileChangeEvidence)
        .flatMap((item) =>
          item.type === "patch"
            ? item.paths
            : item.type === "file_artifact"
              ? [item.path]
              : [],
        ),
    ),
  )
    .filter(
      (path): path is string =>
        typeof path === "string" &&
        Boolean(path.trim()) &&
        !isHiddenConversationArtifactPath(path),
    )
    .map((path) => {
      const parts = path.replace(/\\/g, "/").split("/").filter(Boolean);
      return parts.length > 1 ? parts.slice(-2).join("/") : path;
    })
    .filter((path, index, paths) => paths.indexOf(path) === index);
  const hasPatchEvidence = items.some((item) => item.type === "patch");
  const evidenceLimit = 2;
  const visibleEvidencePaths = evidencePaths.slice(0, evidenceLimit);
  const hiddenEvidenceCount = Math.max(
    0,
    evidencePaths.length - visibleEvidencePaths.length,
  );
  const evidenceLabel = hasPatchEvidence
    ? t("agentChat.messageList.historicalTimeline.patchEvidenceCount", {
        count: evidencePaths.length,
      })
    : t("agentChat.messageList.historicalTimeline.fileEvidenceCount", {
        count: evidencePaths.length,
      });
  const unknownItems = items.filter(
    (item): item is Extract<AgentThreadItem, { type: "unknown_item" }> =>
      item.type === "unknown_item",
  );
  const hasFileChangeEvidence = items.some(hasTimelineFileChangeEvidence);
  const fileArtifactItems = hasFileChangeEvidence
    ? []
    : items.filter(
        (item): item is Extract<AgentThreadItem, { type: "file_artifact" }> =>
          item.type === "file_artifact" &&
          !isHiddenConversationArtifactPath(item.path),
      );
  const summaryRow = (
    <>
      <span className="shrink-0 whitespace-nowrap font-medium">{title}</span>
      <span className="min-w-0 flex-1 truncate text-xs text-slate-400">
        {metaText}
      </span>
      {onExpand ? (
        <ChevronDown className="h-4 w-4 shrink-0 text-slate-400" />
      ) : null}
    </>
  );

  return (
    <div className="space-y-1.5">
      {onExpand ? (
        <button
          type="button"
          data-testid={`message-list-historical-timeline-preview:${placement}`}
          className="group flex w-full min-w-0 items-center border-b border-slate-200/80 py-2 text-left text-sm text-slate-500 transition-colors hover:text-slate-700"
          aria-label={`${title}. ${metaText}`}
          aria-expanded={false}
          onClick={onExpand}
        >
          {summaryRow}
        </button>
      ) : (
        <div
          data-testid={`message-list-historical-timeline-preview:${placement}`}
          className="flex w-full min-w-0 items-center border-b border-slate-200/80 py-2 text-left text-sm text-slate-500"
        >
          {summaryRow}
        </div>
      )}
      {evidencePaths.length > 0 ? (
        <div
          data-testid="historical-file-artifact-summary"
          className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1 px-1 text-xs text-slate-500"
          title={evidencePaths.join(", ")}
        >
          <span className="shrink-0 font-medium text-slate-600">
            {evidenceLabel}
          </span>
          {visibleEvidencePaths.map((path) => (
            <span key={path} className="min-w-0 truncate font-mono">
              {path}
            </span>
          ))}
          {hiddenEvidenceCount > 0 ? (
            <span className="shrink-0 text-slate-400">
              {t(
                "agentChat.messageList.historicalTimeline.patchEvidenceOverflow",
                {
                  count: hiddenEvidenceCount,
                },
              )}
            </span>
          ) : null}
        </div>
      ) : null}
      {fileArtifactItems.length > 0 ? (
        <div data-testid="historical-file-artifact-group" className="space-y-1">
          <AgentThreadTimelineAttachmentList
            items={fileArtifactItems}
            onFileClick={onFileClick}
            onOpenArtifactFromTimeline={onOpenArtifactFromTimeline}
            sourceMessageId={sourceMessageId}
            onSaveFileArtifactAsKnowledge={onSaveFileArtifactAsKnowledge}
          />
        </div>
      ) : null}
      {unknownItems.length > 0 ? (
        <div data-testid="historical-unknown-item-group" className="space-y-1">
          {unknownItems.map((item) => (
            <TimelineItemDetails key={item.id} item={item} />
          ))}
        </div>
      ) : null}
    </div>
  );
};
