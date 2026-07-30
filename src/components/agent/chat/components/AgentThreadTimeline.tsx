import React, { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { ChevronDown } from "lucide-react";

import type {
  ActionRequired,
  AgentThreadItem,
  AgentThreadTurn,
  ConfirmResponse,
  SiteSavedContentTarget,
} from "../types";
import {
  buildAgentThreadDisplayModel,
  type AgentThreadOrderedBlock,
} from "../utils/agentThreadGrouping";
import type { AgentRuntimeThreadReadModel } from "@/lib/api/agentRuntime/sessionTypes";
import { cn } from "@/lib/utils";
import {
  AgentThreadTimelineFileChangesCard,
  hasTimelineFileChangeEvidence,
} from "./AgentThreadTimelineFileChangesCard";
import { AgentThreadTimelineAttachmentList } from "./AgentThreadTimelineAttachmentList";
import { isPlainTimelineFileAttachment } from "./AgentThreadTimelineArtifactCard";
import type { ArtifactTimelineOpenTarget } from "../utils/artifactTimelineNavigation";
import {
  buildTimelineBlockRenderPlan,
  hasStructuredThinkingInlinePreview,
  resolveTimelineBlockEmphasis,
  resolveVisibleTimelineItems,
  type TimelineBlockEmphasis,
} from "./AgentThreadTimelineViewModel";
import {
  TimelineItemDetails,
  ThreadInlineStatusHint,
  TimelineBlockStatusIndicator,
} from "./AgentThreadTimelineItemRenderers";
import {
  resolveActiveBlockIndex,
  resolveCollaborationOpenSubagentLabel,
  resolveFocusBlockIndex,
  resolveExpandedBlockIndexes,
  resolveBlockSummaryLines,
  resolveProcessMixLabel,
  resolveThreadInlineStatusHint,
  resolvePendingRuntimeConfirmationPrompt,
  hasSubmittedRuntimeActionConfirmation,
} from "./timeline-utils";

interface AgentThreadTimelineProps {
  turn: AgentThreadTurn;
  items: AgentThreadItem[];
  threadRead?: AgentRuntimeThreadReadModel | null;
  actionRequests?: ActionRequired[];
  isCurrentTurn?: boolean;
  placement?: "leading" | "trailing" | "default";
  onFileClick?: (fileName: string, content: string) => void;
  onOpenArtifactFromTimeline?: (target: ArtifactTimelineOpenTarget) => void;
  sourceMessageId?: string;
  onSaveFileArtifactAsKnowledge?: (source: {
    messageId: string;
    content: string;
    sourceName?: string;
    description?: string | null;
  }) => void;
  onOpenSavedSiteContent?: (target: SiteSavedContentTarget) => void;
  onOpenSubagentSession?: (sessionId: string) => void;
  onPermissionResponse?: (response: ConfirmResponse) => void;
  focusedItemId?: string | null;
  focusRequestKey?: number;
  deferCompletedSingleDetails?: boolean;
  collapseInactiveDetails?: boolean;
  expandCompletedProcessDetails?: boolean;
  showOperationalDetails?: boolean;
  showInlineStatusHint?: boolean;
}

const INITIAL_DETAIL_ITEM_COUNT = 24;
const DETAIL_ITEM_BATCH_SIZE = 24;

function TimelineBlockCard({
  block,
  index,
  emphasis,
  isExpanded,
  onFileClick,
  onOpenArtifactFromTimeline,
  sourceMessageId,
  onSaveFileArtifactAsKnowledge,
  onOpenSavedSiteContent,
  onOpenSubagentSession,
  onPermissionResponse,
  focusedItemId,
  focusRequestKey,
  preferInlineDetails,
  deferCompletedSingleDetails,
  expandCompletedProcessDetails,
  allowOperationalDetails,
}: {
  block: AgentThreadOrderedBlock;
  index: number;
  emphasis: TimelineBlockEmphasis;
  isExpanded: boolean;
  preferInlineDetails: boolean;
  deferCompletedSingleDetails: boolean;
  expandCompletedProcessDetails: boolean;
  allowOperationalDetails: boolean;
  onFileClick?: (fileName: string, content: string) => void;
  onOpenArtifactFromTimeline?: (target: ArtifactTimelineOpenTarget) => void;
  sourceMessageId?: string;
  onSaveFileArtifactAsKnowledge?: (source: {
    messageId: string;
    content: string;
    sourceName?: string;
    description?: string | null;
  }) => void;
  onOpenSavedSiteContent?: (target: SiteSavedContentTarget) => void;
  onOpenSubagentSession?: (sessionId: string) => void;
  onPermissionResponse?: (response: ConfirmResponse) => void;
  focusedItemId?: string | null;
  focusRequestKey?: number;
}) {
  const { t } = useTranslation("agent");
  const openSubagentLabel = resolveCollaborationOpenSubagentLabel();
  const dataTestId = `agent-thread-block:${index + 1}:${block.kind}`;
  const summaryLines = resolveBlockSummaryLines(block);
  const headline = summaryLines[0] || block.title;
  const supportingLines = summaryLines.slice(1, 3);
  const focusedEntryRef = useRef<HTMLDivElement | null>(null);
  const initialPlan = buildTimelineBlockRenderPlan({
    block,
    isExpanded,
    preferInlineDetails,
    deferCompletedSingleDetails:
      deferCompletedSingleDetails || !allowOperationalDetails,
    focusedItemId,
    hasStructuredThinkingInlinePreview,
  });
  const [open, setOpen] = useState(isExpanded || initialPlan.hasFocusedItem);
  const focusedItemIndex = focusedItemId
    ? block.items.findIndex((item) => item.id === focusedItemId)
    : -1;
  const initialMaterializedItemCount = Math.min(
    block.items.length,
    Math.max(INITIAL_DETAIL_ITEM_COUNT, focusedItemIndex + 1),
  );
  const [materializedItemCount, setMaterializedItemCount] = useState(
    initialMaterializedItemCount,
  );
  const renderPlan = buildTimelineBlockRenderPlan({
    block,
    isExpanded: open,
    preferInlineDetails,
    deferCompletedSingleDetails:
      deferCompletedSingleDetails || !allowOperationalDetails,
    focusedItemId,
    hasStructuredThinkingInlinePreview,
  });

  useEffect(() => {
    setOpen(isExpanded || renderPlan.hasFocusedItem);
  }, [block.id, isExpanded, renderPlan.hasFocusedItem]);

  useEffect(() => {
    setMaterializedItemCount(initialMaterializedItemCount);
  }, [block.id, initialMaterializedItemCount]);

  useEffect(() => {
    if (!renderPlan.hasFocusedItem || !focusRequestKey) {
      return;
    }

    setOpen(true);
    focusedEntryRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
  }, [focusRequestKey, renderPlan.hasFocusedItem]);
  const fileArtifactItems = block.items.filter(
    (item): item is Extract<AgentThreadItem, { type: "file_artifact" }> =>
      item.type === "file_artifact",
  );
  const containsOnlyFileArtifacts =
    fileArtifactItems.length === block.items.length;
  const rendersFileChangeCollection =
    fileArtifactItems.length > 1 &&
    containsOnlyFileArtifacts &&
    fileArtifactItems.every(hasTimelineFileChangeEvidence);
  const rendersAttachmentCollection =
    fileArtifactItems.length > 1 &&
    containsOnlyFileArtifacts &&
    fileArtifactItems.every(isPlainTimelineFileAttachment);
  const rendersAggregatedArtifactCollection =
    rendersFileChangeCollection || rendersAttachmentCollection;
  const hasInteractiveDetails =
    renderPlan.hasDetailEntries &&
    (block.kind === "artifact" || allowOperationalDetails);
  const materializedDetailItems = useMemo(() => {
    if (
      !renderPlan.shouldMaterializeDetailEntries ||
      !hasInteractiveDetails ||
      rendersAggregatedArtifactCollection
    ) {
      return [];
    }

    return block.items.slice(0, materializedItemCount);
  }, [
    block.items,
    materializedItemCount,
    hasInteractiveDetails,
    renderPlan.shouldMaterializeDetailEntries,
    rendersAggregatedArtifactCollection,
  ]);
  const detailEntries = useMemo(() => {
    if (materializedDetailItems.length === 0) {
      return [];
    }

    return materializedDetailItems.flatMap((item) => {
      const content = (
        <TimelineItemDetails
          item={item}
          onFileClick={onFileClick}
          onOpenArtifactFromTimeline={onOpenArtifactFromTimeline}
          onOpenSavedSiteContent={onOpenSavedSiteContent}
          onOpenSubagentSession={onOpenSubagentSession}
          onPermissionResponse={onPermissionResponse}
          groupedToolCall={renderPlan.shouldRenderGroupedToolRows}
          groupMarker={block.items[0]?.id === item.id ? "└" : "·"}
          defaultExpandCompletedToolResult={expandCompletedProcessDetails}
          openSubagentLabel={openSubagentLabel}
          sourceMessageId={sourceMessageId}
          onSaveFileArtifactAsKnowledge={onSaveFileArtifactAsKnowledge}
        />
      );

      return content ? [{ id: item.id, content }] : [];
    });
  }, [
    block.items,
    materializedDetailItems,
    onFileClick,
    onOpenArtifactFromTimeline,
    sourceMessageId,
    onSaveFileArtifactAsKnowledge,
    onOpenSavedSiteContent,
    onOpenSubagentSession,
    onPermissionResponse,
    openSubagentLabel,
    renderPlan.shouldRenderGroupedToolRows,
    expandCompletedProcessDetails,
  ]);
  const remainingDetailItemCount = Math.max(
    0,
    block.items.length - materializedDetailItems.length,
  );
  const nextDetailBatchCount = Math.min(
    DETAIL_ITEM_BATCH_SIZE,
    remainingDetailItemCount,
  );

  if (renderPlan.shouldRenderArtifactCardsInline) {
    return (
      <div
        className={cn("py-0.5", block.items.length === 1 && "space-y-2")}
        data-testid={dataTestId}
        data-emphasis={emphasis}
      >
        {rendersFileChangeCollection ? (
          <AgentThreadTimelineFileChangesCard
            items={fileArtifactItems}
            onFileClick={onFileClick}
            onOpenArtifactFromTimeline={onOpenArtifactFromTimeline}
          />
        ) : rendersAttachmentCollection ? (
          <AgentThreadTimelineAttachmentList
            items={fileArtifactItems}
            onFileClick={onFileClick}
            onOpenArtifactFromTimeline={onOpenArtifactFromTimeline}
            sourceMessageId={sourceMessageId}
            onSaveFileArtifactAsKnowledge={onSaveFileArtifactAsKnowledge}
          />
        ) : (
          detailEntries.map((entry) => (
            <div
              key={entry.id}
              data-thread-item-id={entry.id}
              ref={entry.id === focusedItemId ? focusedEntryRef : null}
              className={cn(
                entry.id === focusedItemId &&
                  "rounded-2xl ring-2 ring-sky-200 ring-offset-2 ring-offset-white",
              )}
            >
              {entry.content}
            </div>
          ))
        )}
      </div>
    );
  }

  const singleItemContent = renderPlan.shouldRenderSingleItemInline ? (
    <TimelineItemDetails
      item={block.items[0]!}
      onFileClick={onFileClick}
      onOpenArtifactFromTimeline={onOpenArtifactFromTimeline}
      onOpenSavedSiteContent={onOpenSavedSiteContent}
      onOpenSubagentSession={onOpenSubagentSession}
      onPermissionResponse={onPermissionResponse}
      defaultExpandCompletedToolResult={expandCompletedProcessDetails}
      openSubagentLabel={openSubagentLabel}
      sourceMessageId={sourceMessageId}
      onSaveFileArtifactAsKnowledge={onSaveFileArtifactAsKnowledge}
    />
  ) : null;

  if (singleItemContent) {
    return (
      <div
        className={cn(
          "py-0.5",
          emphasis === "active" &&
            !renderPlan.isThinkingOnlyBlock &&
            "rounded-xl bg-sky-50/45 px-2",
          emphasis === "quiet" && "opacity-80",
        )}
        data-testid={dataTestId}
        data-emphasis={emphasis}
      >
        <div
          data-thread-item-id={block.items[0]?.id}
          ref={block.items[0]?.id === focusedItemId ? focusedEntryRef : null}
          className={cn(
            block.items[0]?.id === focusedItemId &&
              "rounded-xl ring-2 ring-sky-200 ring-offset-2 ring-offset-white",
          )}
        >
          {singleItemContent}
        </div>
      </div>
    );
  }

  const visibleHeadline = allowOperationalDetails ? headline : block.title;
  const visibleSupportingLines =
    !allowOperationalDetails || (renderPlan.isThinkingOnlyBlock && open)
      ? []
      : supportingLines;
  const summaryCountLabel = block.items.length > 1 ? block.countLabel : null;
  const processMixLabel = resolveProcessMixLabel(block);
  const summaryDetailHint =
    allowOperationalDetails &&
    renderPlan.hasDetailEntries &&
    block.items.length > 1 &&
    !open
      ? processMixLabel || block.rawDetailLabel
      : null;
  const summaryToneClassName = cn(
    "text-slate-900",
    block.status === "in_progress" && "text-sky-700",
    block.kind === "approval" &&
      block.status !== "completed" &&
      "text-amber-800",
    (block.status === "failed" || block.kind === "alert") && "text-rose-700",
  );

  return (
    <div
      className="py-0.5"
      data-testid={`${dataTestId}:shell`}
      data-emphasis={emphasis}
    >
      <details
        data-testid={dataTestId}
        data-emphasis={emphasis}
        data-details-available={hasInteractiveDetails ? "true" : "false"}
        open={hasInteractiveDetails ? open : false}
      >
        <summary
          className={cn(
            "list-none rounded-md px-2 py-1.5",
            hasInteractiveDetails ? "cursor-pointer" : "cursor-default",
            emphasis === "active" &&
              !renderPlan.isThinkingOnlyBlock &&
              "bg-sky-50/45",
          )}
          onClick={(event) => {
            if (!hasInteractiveDetails) {
              event.preventDefault();
              return;
            }

            event.preventDefault();
            setOpen((current) => !current);
          }}
        >
          <div className="flex items-start gap-2.5">
            <div className="mt-1 flex h-4 w-4 shrink-0 items-center justify-center">
              <TimelineBlockStatusIndicator block={block} />
            </div>

            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                <span
                  className={cn(
                    "min-w-0 flex-1 text-sm leading-6",
                    summaryToneClassName,
                  )}
                >
                  {visibleHeadline}
                </span>
                {summaryCountLabel ? (
                  <span className="inline-flex items-center rounded-full border border-slate-200 bg-slate-50 px-1.5 py-0.5 text-[11px] leading-none text-slate-500">
                    {summaryCountLabel}
                  </span>
                ) : null}
                {summaryDetailHint ? (
                  <span className="text-xs leading-5 text-slate-400">
                    {summaryDetailHint}
                  </span>
                ) : null}
              </div>

              {visibleSupportingLines.length > 0 ? (
                <div className="mt-0.5 space-y-1">
                  {visibleSupportingLines.map((line) => (
                    <div
                      key={line}
                      className="text-sm leading-6 text-slate-500"
                    >
                      {line}
                    </div>
                  ))}
                </div>
              ) : null}
            </div>

            {hasInteractiveDetails ? (
              <ChevronDown
                className={cn(
                  "mt-1 h-4 w-4 shrink-0 text-slate-400 transition-transform",
                  open && "rotate-180",
                )}
              />
            ) : null}
          </div>
        </summary>

        {hasInteractiveDetails && open ? (
          <div
            className="ml-6 space-y-2 pb-1 pl-3"
            data-testid={`${dataTestId}:details`}
          >
            {detailEntries.map((entry) => (
              <div
                key={entry.id}
                data-thread-item-id={entry.id}
                ref={entry.id === focusedItemId ? focusedEntryRef : null}
                className={cn(
                  entry.id === focusedItemId &&
                    "rounded-2xl ring-2 ring-sky-200 ring-offset-2 ring-offset-white",
                )}
              >
                {entry.content}
              </div>
            ))}
            {remainingDetailItemCount > 0 ? (
              <button
                type="button"
                data-testid={`${dataTestId}:show-more-details`}
                className="flex h-8 items-center gap-1 rounded-md px-2 text-xs font-medium text-slate-500 transition-colors hover:bg-slate-50 hover:text-slate-800"
                onClick={() =>
                  setMaterializedItemCount((current) =>
                    Math.min(
                      block.items.length,
                      current + DETAIL_ITEM_BATCH_SIZE,
                    ),
                  )
                }
              >
                {t("agentChat.messageList.historicalTimeline.showMoreSteps", {
                  count: nextDetailBatchCount,
                })}
                <ChevronDown className="h-3.5 w-3.5" aria-hidden="true" />
              </button>
            ) : null}
          </div>
        ) : null}
      </details>
    </div>
  );
}

export const AgentThreadTimeline: React.FC<AgentThreadTimelineProps> = ({
  turn,
  items,
  threadRead: _threadRead,
  actionRequests = [],
  isCurrentTurn = false,
  placement = "default",
  onFileClick,
  onOpenArtifactFromTimeline,
  sourceMessageId,
  onSaveFileArtifactAsKnowledge,
  onOpenSavedSiteContent,
  onOpenSubagentSession,
  onPermissionResponse,
  focusedItemId = null,
  focusRequestKey = 0,
  deferCompletedSingleDetails = false,
  collapseInactiveDetails = false,
  expandCompletedProcessDetails = false,
  showOperationalDetails = true,
  showInlineStatusHint = true,
}) => {
  const pendingRuntimeConfirmationPrompt = useMemo(
    () => resolvePendingRuntimeConfirmationPrompt({ items, actionRequests }),
    [actionRequests, items],
  );
  const hasSubmittedRuntimeConfirmation = useMemo(
    () => hasSubmittedRuntimeActionConfirmation({ items, actionRequests }),
    [actionRequests, items],
  );
  const visibleItems = useMemo(
    () => resolveVisibleTimelineItems(items),
    [items],
  );

  const displayModel = useMemo(
    () => buildAgentThreadDisplayModel(visibleItems),
    [visibleItems],
  );
  const activeBlockIndex = resolveActiveBlockIndex(displayModel.orderedBlocks);
  const focusBlockIndex = resolveFocusBlockIndex({
    blocks: displayModel.orderedBlocks,
    turn,
    actionRequests,
    activeBlockIndex,
  });
  const expandedBlockIndexes = resolveExpandedBlockIndexes({
    blocks: displayModel.orderedBlocks,
    isCurrentTurn,
    focusBlockIndex,
    turn,
    collapseInactiveDetails,
    expandCompletedProcessDetails,
  });
  const inlineStatusHint = resolveThreadInlineStatusHint({
    turn,
    actionRequests,
    runtimeConfirmationPrompt: pendingRuntimeConfirmationPrompt,
    hasSubmittedRuntimeConfirmation,
  });

  if (visibleItems.length === 0) {
    return null;
  }

  return (
    <div
      className={cn("space-y-2", placement === "leading" ? "mt-0" : "mt-3")}
      data-testid="agent-thread-flow"
      data-placement={placement}
    >
      {showInlineStatusHint && inlineStatusHint ? (
        <ThreadInlineStatusHint hint={inlineStatusHint} />
      ) : null}
      {displayModel.orderedBlocks.map((block, index) => {
        return (
          <TimelineBlockCard
            key={block.id}
            block={block}
            index={index}
            emphasis={resolveTimelineBlockEmphasis({
              block,
              index,
              activeBlockIndex,
              focusedItemId,
            })}
            isExpanded={expandedBlockIndexes.has(index)}
            preferInlineDetails={isCurrentTurn}
            deferCompletedSingleDetails={deferCompletedSingleDetails}
            expandCompletedProcessDetails={expandCompletedProcessDetails}
            allowOperationalDetails={showOperationalDetails}
            onFileClick={onFileClick}
            onOpenArtifactFromTimeline={onOpenArtifactFromTimeline}
            sourceMessageId={sourceMessageId}
            onSaveFileArtifactAsKnowledge={onSaveFileArtifactAsKnowledge}
            onOpenSavedSiteContent={onOpenSavedSiteContent}
            onOpenSubagentSession={onOpenSubagentSession}
            onPermissionResponse={onPermissionResponse}
            focusedItemId={focusedItemId}
            focusRequestKey={focusRequestKey}
          />
        );
      })}
    </div>
  );
};
