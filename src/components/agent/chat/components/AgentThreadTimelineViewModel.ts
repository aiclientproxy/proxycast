import { parseAIResponse } from "@/components/workspace/a2ui/parser";
import type { AgentThreadItem } from "../types";
import type { AgentThreadOrderedBlock } from "../utils/agentThreadGrouping";
import { isHiddenConversationArtifactPath } from "../utils/internalArtifactVisibility";
import { isRuntimePermissionConfirmationWaitMessage } from "../utils/runtimeActionConfirmation";
import {
  isThinkingTimelineItem,
  resolveThinkingDisplayText,
  resolveTurnSummaryDisplayText,
} from "./timeline-utils";

export type TimelineBlockEmphasis = "active" | "default" | "quiet";

export interface TimelineBlockRenderPlan {
  isThinkingOnlyBlock: boolean;
  hasFocusedItem: boolean;
  hasDetailEntries: boolean;
  shouldRenderArtifactCardsInline: boolean;
  shouldRenderActiveSingleThinkingInline: boolean;
  shouldSummarizeSingleThinkingInline: boolean;
  shouldRenderSingleItemInline: boolean;
  shouldRenderGroupedToolRows: boolean;
  shouldMaterializeDetailEntries: boolean;
}

export function resolveVisibleTimelineItems(
  items: AgentThreadItem[],
): AgentThreadItem[] {
  return items.filter(
    (item) =>
      item.type !== "user_message" &&
      item.type !== "agent_message" &&
      !(
        item.type === "error" &&
        isRuntimePermissionConfirmationWaitMessage(item.message)
      ) &&
      !(
        item.type === "file_artifact" &&
        isHiddenConversationArtifactPath(item.path)
      ),
  );
}

export function resolveTimelineBlockEmphasis(params: {
  block: AgentThreadOrderedBlock;
  index: number;
  activeBlockIndex: number;
  focusedItemId?: string | null;
}): TimelineBlockEmphasis {
  const hasFocusedItem = Boolean(
    params.focusedItemId &&
    params.block.items.some((item) => item.id === params.focusedItemId),
  );

  if (hasFocusedItem || params.activeBlockIndex === params.index) {
    return "active";
  }
  return params.block.status === "completed" ? "quiet" : "default";
}

export function hasStructuredThinkingInlinePreview(
  item: AgentThreadItem,
): boolean {
  if (item.type !== "reasoning" && item.type !== "turn_summary") {
    return false;
  }

  const displayText =
    item.type === "reasoning"
      ? resolveThinkingDisplayText(item)
      : resolveTurnSummaryDisplayText(item);
  if (!displayText.trim()) {
    return false;
  }

  const parsed = parseAIResponse(displayText, false);
  return Boolean(parsed.hasA2UI || parsed.hasPending);
}

export function buildTimelineBlockRenderPlan(params: {
  block: AgentThreadOrderedBlock;
  isExpanded: boolean;
  preferInlineDetails: boolean;
  deferCompletedSingleDetails: boolean;
  focusedItemId?: string | null;
  hasStructuredThinkingInlinePreview: (item: AgentThreadItem) => boolean;
}): TimelineBlockRenderPlan {
  const {
    block,
    isExpanded,
    preferInlineDetails,
    deferCompletedSingleDetails,
    focusedItemId,
    hasStructuredThinkingInlinePreview,
  } = params;
  const isThinkingOnlyBlock = block.items.every((item) =>
    isThinkingTimelineItem(item),
  );
  const hasFocusedItem = Boolean(
    focusedItemId && block.items.some((item) => item.id === focusedItemId),
  );
  const hasDetailEntries = block.items.length > 0;
  const hasPatchItem = block.items.some((item) => item.type === "patch");
  const shouldRenderArtifactCardsInline =
    hasDetailEntries &&
    ((block.kind === "artifact" &&
      block.items.every(
        (item) => item.type === "file_artifact" || item.type === "patch",
      )) ||
      (block.kind === "process" && hasPatchItem));
  const singleThinkingItem =
    block.items.length === 1 && isThinkingTimelineItem(block.items[0]!)
      ? block.items[0]!
      : null;
  const shouldRenderActiveSingleThinkingInline =
    Boolean(singleThinkingItem) && block.status === "in_progress";
  const shouldRenderStructuredSingleThinkingInline =
    Boolean(singleThinkingItem) &&
    singleThinkingItem?.type === "reasoning" &&
    hasStructuredThinkingInlinePreview(singleThinkingItem);
  const shouldKeepCompletedSingleReasoningInShell =
    Boolean(singleThinkingItem) &&
    singleThinkingItem?.type === "reasoning" &&
    block.status === "completed" &&
    !shouldRenderStructuredSingleThinkingInline;
  const shouldSummarizeSingleThinkingInline =
    Boolean(singleThinkingItem) &&
    singleThinkingItem?.type === "turn_summary" &&
    !shouldRenderActiveSingleThinkingInline &&
    !hasStructuredThinkingInlinePreview(singleThinkingItem);
  const shouldRenderSingleItemInline =
    block.items.length === 1 &&
    !shouldSummarizeSingleThinkingInline &&
    (shouldRenderActiveSingleThinkingInline ||
      shouldRenderStructuredSingleThinkingInline ||
      (!shouldKeepCompletedSingleReasoningInShell &&
        (!deferCompletedSingleDetails ||
          preferInlineDetails ||
          block.status !== "completed" ||
          hasFocusedItem)));
  const shouldRenderGroupedToolRows =
    block.kind === "process" && block.items.length > 1;
  const shouldMaterializeDetailEntries =
    shouldRenderArtifactCardsInline ||
    (!shouldRenderSingleItemInline && hasDetailEntries && isExpanded);

  return {
    isThinkingOnlyBlock,
    hasFocusedItem,
    hasDetailEntries,
    shouldRenderArtifactCardsInline,
    shouldRenderActiveSingleThinkingInline,
    shouldSummarizeSingleThinkingInline,
    shouldRenderSingleItemInline,
    shouldRenderGroupedToolRows,
    shouldMaterializeDetailEntries,
  };
}
