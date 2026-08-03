import type { AgentThreadItem, AgentThreadItemStatus } from "../types";
import { summarizeThreadProcessBatch } from "./toolBatchGrouping";
import {
  buildPreviewLines,
  classifyItemKind,
  summarizeThinkingItem,
} from "./agentThreadGroupingItemSummary";
import type {
  AgentThreadDisplayModel,
  AgentThreadGroupKind,
  AgentThreadOrderedBlock,
  AgentThreadSemanticGroup,
  AgentThreadSummaryChip,
} from "./agentThreadGroupingTypes";
import {
  isUnifiedWebFetchToolName,
  isUnifiedWebSearchToolName,
} from "./toolNameFamily";

export type {
  AgentThreadDisplayModel,
  AgentThreadGroupKind,
  AgentThreadOrderedBlock,
  AgentThreadSemanticGroup,
  AgentThreadSummaryChip,
} from "./agentThreadGroupingTypes";

function resolveItemTimestamp(item: AgentThreadItem): string {
  return item.completed_at || item.updated_at || item.started_at;
}

function compareItems(left: AgentThreadItem, right: AgentThreadItem): number {
  if (left.turn_id === right.turn_id && left.sequence !== right.sequence) {
    return left.sequence - right.sequence;
  }
  const leftTimestamp = resolveItemTimestamp(left);
  const rightTimestamp = resolveItemTimestamp(right);
  if (leftTimestamp !== rightTimestamp) {
    return leftTimestamp.localeCompare(rightTimestamp);
  }
  if (left.sequence !== right.sequence) {
    return left.sequence - right.sequence;
  }
  return left.id.localeCompare(right.id);
}

function mergeStatuses(
  statuses: AgentThreadItemStatus[],
): AgentThreadItemStatus {
  if (statuses.some((status) => status === "in_progress")) {
    return "in_progress";
  }
  if (statuses.some((status) => status === "failed")) {
    return "failed";
  }
  return "completed";
}

function isWebSearchProcessItem(item: AgentThreadItem): boolean {
  return (
    item.type === "web_search" ||
    (item.type === "tool_call" && isUnifiedWebSearchToolName(item.tool_name))
  );
}

function isWebFetchProcessItem(item: AgentThreadItem): boolean {
  return item.type === "tool_call" && isUnifiedWebFetchToolName(item.tool_name);
}

function isWebRetrievalProcessItem(item: AgentThreadItem): boolean {
  return isWebSearchProcessItem(item) || isWebFetchProcessItem(item);
}

function isFailedWebSearchProcessItem(item: AgentThreadItem): boolean {
  return isWebSearchProcessItem(item) && item.status === "failed";
}

function shouldSplitProcessBlockBeforeItem(
  current: {
    kind: AgentThreadGroupKind;
    items: AgentThreadItem[];
  },
  item: AgentThreadItem,
): boolean {
  if (current.kind !== "process" || !isWebSearchProcessItem(item)) {
    return false;
  }

  if (!current.items.some(isWebRetrievalProcessItem)) {
    return false;
  }

  if (
    current.items.length > 0 &&
    current.items.every(isFailedWebSearchProcessItem) &&
    isFailedWebSearchProcessItem(item)
  ) {
    return false;
  }

  return true;
}

function isImportedHistoryItem(item: AgentThreadItem): boolean {
  const metadata = item.metadata as Record<string, unknown> | undefined;
  return metadata?.imported === true;
}

function shouldMergeImportedPatchIntoProcess(
  current: {
    kind: AgentThreadGroupKind;
    items: AgentThreadItem[];
  } | null,
  kind: AgentThreadGroupKind,
  item: AgentThreadItem,
): boolean {
  return Boolean(
    current &&
      current.kind === "process" &&
      kind === "artifact" &&
      item.type === "patch" &&
      isImportedHistoryItem(item) &&
      current.items.some(isImportedHistoryItem),
  );
}

function resolveGroupTitle(
  kind: Exclude<AgentThreadGroupKind, "other">,
): string {
  switch (kind) {
    case "process":
      return "执行过程";
    case "approval":
      return "等你确认";
    case "alert":
      return "提醒和错误";
    case "artifact":
      return "文件和产物";
    case "subagent":
      return "子任务";
    default:
      return "执行过程";
  }
}

function resolveBlockTitle(kind: AgentThreadGroupKind): string {
  if (kind === "other") {
    return "执行过程";
  }
  return resolveGroupTitle(kind);
}

function resolveCountLabel(kind: AgentThreadGroupKind, count: number): string {
  switch (kind) {
    case "process":
      return `${count} 步`;
    case "artifact":
      return `${count} 份`;
    case "subagent":
      return `${count} 个任务`;
    default:
      return `${count} 项`;
  }
}

function shouldDefaultExpand(
  kind: AgentThreadGroupKind,
  status: AgentThreadItemStatus,
): boolean {
  if (kind === "approval" || kind === "alert") {
    return true;
  }
  return status !== "completed";
}

function buildSummaryText(items: AgentThreadItem[]): string | null {
  const sortedThinking = items
    .filter(
      (item) =>
        item.type === "plan" ||
        item.type === "reasoning" ||
        item.type === "turn_summary" ||
        item.type === "context_compaction",
    )
    .sort(compareItems);

  for (let index = sortedThinking.length - 1; index >= 0; index -= 1) {
    const candidate = summarizeThinkingItem(sortedThinking[index]);
    if (candidate) {
      return candidate;
    }
  }

  return null;
}

export function buildAgentThreadDisplayModel(
  items: AgentThreadItem[],
): AgentThreadDisplayModel {
  const sortedItems = [...items].sort(compareItems);
  const thinkingItems = sortedItems.filter(
    (item) =>
      item.type === "plan" ||
      item.type === "reasoning" ||
      item.type === "turn_summary" ||
      item.type === "context_compaction",
  );
  const orderedBlocks: AgentThreadOrderedBlock[] = [];
  const groups: AgentThreadSemanticGroup[] = [];
  let current: {
    kind: AgentThreadGroupKind;
    items: AgentThreadItem[];
  } | null = null;

  const pushCurrentBlock = () => {
    if (!current) {
      return;
    }

    const status = mergeStatuses(current.items.map((entry) => entry.status));
    const hasReasoningProcessItem = current.items.some(
      (entry) =>
        entry.type === "plan" ||
        entry.type === "reasoning" ||
        entry.type === "turn_summary" ||
        entry.type === "context_compaction",
    );
    const processBatchSummary =
      current.kind === "process" && !hasReasoningProcessItem
        ? summarizeThreadProcessBatch(current.items)
        : null;
    const startedAt =
      current.items[0]?.started_at || current.items[0]?.updated_at || "";
    const completedAt = current.items[current.items.length - 1]?.completed_at;
    const block: AgentThreadOrderedBlock = {
      id: current.items.map((entry) => entry.id).join(":"),
      kind: current.kind,
      title: processBatchSummary?.title || resolveBlockTitle(current.kind),
      status,
      items: current.items,
      previewLines:
        processBatchSummary?.supportingLines ||
        buildPreviewLines(current.kind, current.items),
      countLabel:
        processBatchSummary?.countLabel ||
        resolveCountLabel(current.kind, current.items.length),
      rawDetailLabel:
        processBatchSummary?.rawDetailLabel ||
        (current.kind === "approval"
          ? "查看待处理项"
          : current.kind === "artifact"
            ? "查看产物"
            : current.kind === "subagent"
              ? "查看子任务详情"
              : "查看执行过程"),
      defaultExpanded: shouldDefaultExpand(current.kind, status),
      startedAt,
      completedAt,
    };

    orderedBlocks.push(block);

    if (current.kind !== "other") {
      groups.push({
        id: block.id,
        kind: current.kind,
        title: block.title,
        status: block.status,
        items: block.items,
        previewLines: block.previewLines,
        countLabel: block.countLabel,
        rawDetailLabel: block.rawDetailLabel,
        defaultExpanded: block.defaultExpanded,
        ...(block.forceExpanded ? { forceExpanded: true } : {}),
      });
    }
  };

  for (const item of sortedItems) {
    const kind = classifyItemKind(item);
    if (shouldMergeImportedPatchIntoProcess(current, kind, item)) {
      current!.items.push(item);
      continue;
    }

    if (!current || current.kind !== kind) {
      pushCurrentBlock();
      current = { kind, items: [item] };
      continue;
    }

    if (shouldSplitProcessBlockBeforeItem(current, item)) {
      pushCurrentBlock();
      current = { kind, items: [item] };
      continue;
    }

    current.items.push(item);
  }

  pushCurrentBlock();

  const summaryCounts = new Map<
    AgentThreadSummaryChip["kind"],
    AgentThreadSummaryChip
  >();

  for (const group of groups) {
    if (group.kind === "approval" || group.kind === "alert") {
      continue;
    }

    const existing = summaryCounts.get(group.kind);
    if (existing) {
      existing.count += group.items.length;
      continue;
    }

    summaryCounts.set(group.kind, {
      kind: group.kind,
      label: group.title,
      count: group.items.length,
    });
  }

  return {
    summaryText: buildSummaryText(sortedItems),
    thinkingItems,
    groups,
    orderedBlocks,
    summaryChips: Array.from(summaryCounts.values()),
  };
}
