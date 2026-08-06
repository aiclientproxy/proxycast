import { formatNumber } from "@/i18n/format";
import type { AgentThreadItem } from "../types";
import { summarizeThinkingItem } from "../utils/agentThreadGroupingItemSummary";
import { normalizeComparableThinkingText } from "../utils/internalThinkingText";
import { isHiddenConversationArtifactPath } from "../utils/internalArtifactVisibility";

const MAX_REASONABLE_TURN_TO_ITEM_RATIO = 4;
const MAX_HISTORICAL_THINKING_PREVIEWS = 2;

export function formatHistoricalContentLength(value: number): string {
  return formatNumber(value);
}

export function buildHistoricalMessagePreview(
  content: string,
  previewChars: number,
): string {
  const normalized = content.trim();
  if (normalized.length <= previewChars) {
    return normalized;
  }

  return `${normalized.slice(0, previewChars)}\n\n...`;
}

export function resolveHistoricalTimelineDurationMs(
  items: readonly AgentThreadItem[],
  startedAt?: string | null,
  completedAt?: string | null,
): number | null {
  const turnDurationMs = resolveDurationMs(startedAt, completedAt);
  const itemTimes = items.flatMap((item) => {
    const itemStartedAt = parseTimestamp(item.started_at);
    const itemCompletedAt = parseTimestamp(
      item.completed_at ?? item.updated_at,
    );
    return itemStartedAt !== null &&
      itemCompletedAt !== null &&
      itemCompletedAt >= itemStartedAt
      ? [{ startedAt: itemStartedAt, completedAt: itemCompletedAt }]
      : [];
  });
  if (itemTimes.length > 0) {
    const earliestStartedAt = Math.min(
      ...itemTimes.map((item) => item.startedAt),
    );
    const latestCompletedAt = Math.max(
      ...itemTimes.map((item) => item.completedAt),
    );
    const itemDurationMs = latestCompletedAt - earliestStartedAt;
    if (itemDurationMs > 0) {
      if (
        turnDurationMs !== null &&
        turnDurationMs > 0 &&
        turnDurationMs > itemDurationMs * MAX_REASONABLE_TURN_TO_ITEM_RATIO
      ) {
        return itemDurationMs;
      }

      if (turnDurationMs !== null && turnDurationMs > 0) {
        return turnDurationMs;
      }

      return itemDurationMs;
    }
  }

  return turnDurationMs;
}

export function formatHistoricalTimelineDuration(
  durationMs: number | null,
): string | null {
  if (durationMs === null || !Number.isFinite(durationMs) || durationMs <= 0) {
    return null;
  }

  const totalSeconds = Math.max(1, Math.round(durationMs / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
}

function resolveDurationMs(
  startedAt?: string | null,
  completedAt?: string | null,
): number | null {
  const startedAtMs = parseTimestamp(startedAt);
  const completedAtMs = parseTimestamp(completedAt);
  if (
    startedAtMs === null ||
    completedAtMs === null ||
    completedAtMs < startedAtMs
  ) {
    return null;
  }
  return completedAtMs - startedAtMs;
}

function parseTimestamp(value?: string | null): number | null {
  if (!value) return null;
  const timestamp = new Date(value).getTime();
  return Number.isFinite(timestamp) ? timestamp : null;
}

export function summarizeHistoricalTimelineItems(items: AgentThreadItem[]): {
  stepsCount: number;
  toolStepsCount: number;
  thinkingStepsCount: number;
  artifactStepsCount: number;
  thinkingPreviews: string[];
} {
  const visibleItems = items.filter((item) => {
    if (item.type === "user_message" || item.type === "agent_message") {
      return false;
    }

    return !(
      item.type === "file_artifact" &&
      isHiddenConversationArtifactPath(item.path)
    );
  });
  const toolStepsCount = visibleItems.filter(
    (item) =>
      item.type === "tool_call" ||
      item.type === "command_execution" ||
      item.type === "web_search",
  ).length;
  const thinkingStepsCount = visibleItems.filter(
    (item) =>
      item.type === "reasoning" ||
      item.type === "plan" ||
      item.type === "turn_summary" ||
      item.type === "context_compaction",
  ).length;
  const artifactStepsCount = visibleItems.filter(
    (item) => item.type === "file_artifact",
  ).length;
  const thinkingPreviews: string[] = [];
  const seenThinkingPreviews = new Set<string>();
  for (const item of visibleItems) {
    if (
      item.type !== "reasoning" &&
      item.type !== "plan" &&
      item.type !== "turn_summary" &&
      item.type !== "context_compaction"
    ) {
      continue;
    }

    const hasExplicitReasoningSummary =
      item.type !== "reasoning" ||
      item.summary?.some((line) => line.trim().length > 0) === true;
    const preview = hasExplicitReasoningSummary
      ? summarizeThinkingItem(item)
      : null;
    if (!preview) {
      continue;
    }

    const comparablePreview = normalizeComparableThinkingText(preview);
    if (
      !comparablePreview ||
      seenThinkingPreviews.has(comparablePreview) ||
      thinkingPreviews.length >= MAX_HISTORICAL_THINKING_PREVIEWS
    ) {
      continue;
    }

    seenThinkingPreviews.add(comparablePreview);
    thinkingPreviews.push(preview);
  }
  return {
    stepsCount: visibleItems.length,
    toolStepsCount,
    thinkingStepsCount,
    artifactStepsCount,
    thinkingPreviews,
  };
}
