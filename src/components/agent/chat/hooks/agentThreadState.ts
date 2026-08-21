import type { AgentThreadItem, AgentThreadTurn } from "../types";
import { resolveThreadItemTimelinePosition } from "./agentChatHistoryPrimitives";
import {
  appendInterruptedPlaceholderText,
  INTERRUPTED_PLACEHOLDER_TEXT,
  stripInterruptedPlaceholderText,
} from "./agentInterruptedMessageContent";

export function areJsonLikeValuesEqual(left: unknown, right: unknown): boolean {
  if (Object.is(left, right)) {
    return true;
  }

  if (
    !left ||
    !right ||
    typeof left !== "object" ||
    typeof right !== "object"
  ) {
    return false;
  }

  if (Array.isArray(left) || Array.isArray(right)) {
    if (!Array.isArray(left) || !Array.isArray(right)) {
      return false;
    }
    if (left.length !== right.length) {
      return false;
    }
    return left.every((item, index) =>
      areJsonLikeValuesEqual(item, right[index]),
    );
  }

  const leftRecord = left as Record<string, unknown>;
  const rightRecord = right as Record<string, unknown>;
  const leftKeys = Object.keys(leftRecord);
  const rightKeys = Object.keys(rightRecord);

  if (leftKeys.length !== rightKeys.length) {
    return false;
  }

  return leftKeys.every((key) =>
    Object.prototype.hasOwnProperty.call(rightRecord, key)
      ? areJsonLikeValuesEqual(leftRecord[key], rightRecord[key])
      : false,
  );
}

function compareItemOrder(
  left: AgentThreadItem,
  right: AgentThreadItem,
): number {
  const leftTimelinePosition = resolveThreadItemTimelinePosition(left);
  const rightTimelinePosition = resolveThreadItemTimelinePosition(right);
  if (
    left.turn_id === right.turn_id &&
    leftTimelinePosition !== rightTimelinePosition
  ) {
    return leftTimelinePosition - rightTimelinePosition;
  }
  const leftStartedAt = String(left.started_at || "");
  const rightStartedAt = String(right.started_at || "");
  if (leftStartedAt !== rightStartedAt) {
    return leftStartedAt.localeCompare(rightStartedAt);
  }
  if (leftTimelinePosition !== rightTimelinePosition) {
    return leftTimelinePosition - rightTimelinePosition;
  }
  return String(left.id || "").localeCompare(String(right.id || ""));
}

function threadItemMergeKey(item: AgentThreadItem): string {
  if (item.type === "approval_request" || item.type === "request_user_input") {
    return `${item.turn_id || ""}:action:${item.request_id}`;
  }
  if (
    item.type === "tool_call" ||
    item.type === "command_execution" ||
    item.type === "web_search"
  ) {
    return `${item.turn_id || ""}:${item.id}`;
  }
  return item.id;
}

function isEmptyMergeValue(value: unknown): boolean {
  return (
    value === undefined ||
    value === null ||
    value === "" ||
    (Array.isArray(value) && value.length === 0)
  );
}

function mergeThreadItemUpdate(
  existing: AgentThreadItem,
  nextItem: AgentThreadItem,
): AgentThreadItem {
  if (existing.type !== nextItem.type) {
    return nextItem;
  }

  const merged: Record<string, unknown> = { ...existing };
  for (const [key, value] of Object.entries(nextItem)) {
    if (!isEmptyMergeValue(value) || !Object.hasOwn(existing, key)) {
      merged[key] = value;
    }
  }
  merged.sequence = Math.min(existing.sequence, nextItem.sequence);
  const startedAt = existing.started_at || nextItem.started_at;
  const completedAt = nextItem.completed_at || existing.completed_at;
  const updatedAt = nextItem.updated_at || existing.updated_at;
  if (startedAt) {
    merged.started_at = startedAt;
  }
  if (completedAt) {
    merged.completed_at = completedAt;
  }
  if (updatedAt) {
    merged.updated_at = updatedAt;
  }
  const existingRecord = existing as unknown as Record<string, unknown>;

  for (const [key, value] of Object.entries(nextItem)) {
    if (isEmptyMergeValue(value) && !isEmptyMergeValue(existingRecord[key])) {
      merged[key] = existingRecord[key];
    }
  }

  if (existing.type === "agent_message" && nextItem.type === "agent_message") {
    const existingText = stripInterruptedPlaceholderText(existing.text);
    const nextText = stripInterruptedPlaceholderText(nextItem.text);
    const nextHasInterruptedPlaceholder = nextItem.text
      .trimEnd()
      .endsWith(INTERRUPTED_PLACEHOLDER_TEXT);

    if (
      existingText &&
      (!nextText || nextText.length < existingText.length)
    ) {
      merged.text = nextHasInterruptedPlaceholder
        ? appendInterruptedPlaceholderText(existing.text)
        : existing.text;
    }
  }

  return merged as unknown as AgentThreadItem;
}

export function upsertThreadTurnState(
  turns: AgentThreadTurn[],
  nextTurn: AgentThreadTurn,
): AgentThreadTurn[] {
  const existingIndex = turns.findIndex((turn) => turn.id === nextTurn.id);
  if (existingIndex < 0) {
    return [...turns, nextTurn].sort((left, right) => {
      const startedAtComparison = String(left.started_at || "").localeCompare(
        String(right.started_at || ""),
      );
      if (startedAtComparison !== 0) {
        return startedAtComparison;
      }
      return String(left.id || "").localeCompare(String(right.id || ""));
    });
  }

  const existingTurn = turns[existingIndex];
  if (areJsonLikeValuesEqual(existingTurn, nextTurn)) {
    return turns;
  }

  return turns.map((turn) => (turn.id === nextTurn.id ? nextTurn : turn));
}

export function upsertThreadItemState(
  items: AgentThreadItem[],
  nextItem: AgentThreadItem,
): AgentThreadItem[] {
  const nextKey = threadItemMergeKey(nextItem);
  const existingIndex = items.findIndex(
    (item) => threadItemMergeKey(item) === nextKey,
  );
  if (existingIndex < 0) {
    return [...items, nextItem].sort(compareItemOrder);
  }

  const existingItem = items[existingIndex];
  if (
    (existingItem.status === "completed" || existingItem.status === "failed") &&
    nextItem.status === "in_progress"
  ) {
    return items;
  }
  const mergedItem = mergeThreadItemUpdate(existingItem, nextItem);
  if (areJsonLikeValuesEqual(existingItem, mergedItem)) {
    return items;
  }

  const nextItems = items.map((item) =>
    threadItemMergeKey(item) === nextKey ? mergedItem : item,
  );
  nextItems.sort(compareItemOrder);
  return nextItems;
}

export function removeThreadTurnState(
  turns: AgentThreadTurn[],
  turnId: string,
): AgentThreadTurn[] {
  if (!turns.some((turn) => turn.id === turnId)) {
    return turns;
  }
  return turns.filter((turn) => turn.id !== turnId);
}

export function removeThreadItemState(
  items: AgentThreadItem[],
  itemId: string,
): AgentThreadItem[] {
  if (!items.some((item) => item.id === itemId)) {
    return items;
  }
  return items.filter((item) => item.id !== itemId);
}

export function markThreadActionItemSubmitted(
  items: AgentThreadItem[],
  requestIds: Set<string>,
  response?: string,
  userData?: unknown,
): AgentThreadItem[] {
  const normalizedResponse = response?.trim();

  return items.map((item) => {
    if (
      (item.type !== "approval_request" &&
        item.type !== "request_user_input") ||
      !requestIds.has(item.request_id)
    ) {
      return item;
    }

    const nextResponse = userData ?? normalizedResponse ?? item.response;
    return {
      ...item,
      status: "completed",
      completed_at: item.completed_at || new Date().toISOString(),
      updated_at: new Date().toISOString(),
      response: nextResponse,
    };
  });
}
