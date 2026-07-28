import type { AgentThreadItem, AgentThreadTurn, Message } from "../types";
import { shouldHideTurnSummaryFromConversation } from "./turnSummaryPresentation";

const HIDDEN_CONVERSATION_WARNING_CODES = new Set([
  "artifact_document_repaired",
]);

const HIDDEN_CONVERSATION_AUXILIARY_TURN_ID_PREFIX =
  "auxiliary-runtime-projection-";

export interface MessageTurnTimeline {
  messageId: string;
  turn: AgentThreadTurn;
  items: AgentThreadItem[];
}

function normalizeSortString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function compareStableString(left: unknown, right: unknown): number {
  return normalizeSortString(left).localeCompare(normalizeSortString(right));
}

export function compareThreadTurns(
  left: AgentThreadTurn,
  right: AgentThreadTurn,
): number {
  if (left.started_at !== right.started_at) {
    return compareStableString(left.started_at, right.started_at);
  }
  return compareStableString(left.id, right.id);
}

export function compareThreadItems(
  left: AgentThreadItem,
  right: AgentThreadItem,
): number {
  if (left.turn_id === right.turn_id && left.sequence !== right.sequence) {
    return left.sequence - right.sequence;
  }
  if (left.started_at !== right.started_at) {
    return compareStableString(left.started_at, right.started_at);
  }
  if (left.sequence !== right.sequence) {
    return left.sequence - right.sequence;
  }
  return compareStableString(left.id, right.id);
}

function normalizeThreadWarningCode(value?: string | null): string | null {
  const normalized = value?.trim().toLowerCase();
  return normalized || null;
}

function normalizeThreadArtifactPath(value?: string | null): string {
  return (value || "").trim().replace(/\\/g, "/").toLowerCase();
}

function readMetadataString(
  metadata: unknown,
  keys: readonly string[],
): string | null {
  if (!metadata || typeof metadata !== "object" || Array.isArray(metadata)) {
    return null;
  }

  const record = metadata as Record<string, unknown>;
  for (const key of keys) {
    const value = record[key];
    if (typeof value !== "string") {
      continue;
    }

    const normalized = value.trim().toLowerCase();
    if (normalized) {
      return normalized;
    }
  }

  return null;
}

function shouldHideConversationThreadTurn(turn: AgentThreadTurn): boolean {
  const normalizedId =
    typeof turn.id === "string" ? turn.id.trim().toLowerCase() : "";
  if (normalizedId.startsWith(HIDDEN_CONVERSATION_AUXILIARY_TURN_ID_PREFIX)) {
    return true;
  }

  const normalizedPrompt =
    typeof turn.prompt_text === "string" ? turn.prompt_text.trim() : "";
  return (
    normalizedPrompt.startsWith("辅助标题生成") ||
    normalizedPrompt.startsWith("辅助人设生成")
  );
}

function shouldHideAuxiliaryRuntimeProjectionItem(
  item: AgentThreadItem,
): boolean {
  if (item.type !== "file_artifact") {
    return false;
  }

  const normalizedPath = normalizeThreadArtifactPath(item.path);
  const hasAuxiliaryRuntimePath =
    normalizedPath.endsWith(".json") &&
    normalizedPath.includes("/auxiliary-runtime/") &&
    (normalizedPath.startsWith(".lime/harness/sessions/") ||
      normalizedPath.includes("/.lime/harness/sessions/"));
  if (hasAuxiliaryRuntimePath) {
    return true;
  }

  const artifactType = readMetadataString(item.metadata, [
    "artifactType",
    "artifact_type",
    "task_type",
  ]);
  if (artifactType === "auxiliary_runtime_projection") {
    return true;
  }

  const source = readMetadataString(item.metadata, ["source"]);
  return source?.startsWith("auxiliary.") ?? false;
}

function shouldHideConversationThreadItem(item: AgentThreadItem): boolean {
  if (shouldHideAuxiliaryRuntimeProjectionItem(item)) {
    return true;
  }

  if (item.type === "turn_summary") {
    return shouldHideTurnSummaryFromConversation(item);
  }

  if (item.type !== "warning") {
    return false;
  }

  const normalizedCode = normalizeThreadWarningCode(item.code);
  return (
    normalizedCode !== null &&
    HIDDEN_CONVERSATION_WARNING_CODES.has(normalizedCode)
  );
}

export function filterConversationThreadItems(
  items: readonly AgentThreadItem[],
): AgentThreadItem[] {
  return items.filter((item) => !shouldHideConversationThreadItem(item));
}

function isSortedBy<T>(
  items: readonly T[],
  compare: (left: T, right: T) => number,
) {
  for (let index = 1; index < items.length; index += 1) {
    if (compare(items[index - 1]!, items[index]!) > 0) {
      return false;
    }
  }

  return true;
}

export function sortThreadItems(
  items: readonly AgentThreadItem[],
): AgentThreadItem[] {
  return [...filterConversationThreadItems(items)].sort(compareThreadItems);
}

function resolveSortedThreadTurns(
  turns: readonly AgentThreadTurn[],
): AgentThreadTurn[] {
  const hasHiddenTurns = turns.some(shouldHideConversationThreadTurn);
  const visibleTurns = hasHiddenTurns
    ? turns.filter((turn) => !shouldHideConversationThreadTurn(turn))
    : [...turns];

  return isSortedBy(visibleTurns, compareThreadTurns)
    ? visibleTurns
    : [...visibleTurns].sort(compareThreadTurns);
}

function resolveTimelineThreadItems(
  items: AgentThreadItem[],
): AgentThreadItem[] {
  const hasHiddenItems = items.some(shouldHideConversationThreadItem);
  const visibleItems = hasHiddenItems
    ? filterConversationThreadItems(items)
    : items;

  return isSortedBy(visibleItems, compareThreadItems)
    ? visibleItems
    : [...visibleItems].sort(compareThreadItems);
}

export function mergeThreadTurns(
  ...turnGroups: Array<AgentThreadTurn[] | undefined>
): AgentThreadTurn[] {
  const merged = new Map<string, AgentThreadTurn>();

  for (const turns of turnGroups) {
    if (!Array.isArray(turns)) {
      continue;
    }

    for (const turn of turns) {
      if (shouldHideConversationThreadTurn(turn)) {
        continue;
      }

      merged.set(turn.id, turn);
    }
  }

  return [...merged.values()].sort(compareThreadTurns);
}

export function mergeThreadItems(
  ...itemGroups: Array<AgentThreadItem[] | undefined>
): AgentThreadItem[] {
  const merged = new Map<string, AgentThreadItem>();

  for (const items of itemGroups) {
    if (!Array.isArray(items)) {
      continue;
    }

    for (const item of items) {
      if (shouldHideConversationThreadItem(item)) {
        continue;
      }
      merged.set(item.id, item);
    }
  }

  return sortThreadItems(Array.from(merged.values()));
}

function resolveTimestampMs(value?: string | Date | null): number | null {
  if (value instanceof Date) {
    const time = value.getTime();
    return Number.isFinite(time) ? time : null;
  }

  if (!value) {
    return null;
  }

  const time = new Date(value).getTime();
  return Number.isFinite(time) ? time : null;
}

function isSubstantiveAssistantMessage(message: Message): boolean {
  if (message.content.trim().length > 0) {
    return true;
  }

  if (Array.isArray(message.images) && message.images.length > 0) {
    return true;
  }

  if (
    !Array.isArray(message.contentParts) ||
    message.contentParts.length === 0
  ) {
    return false;
  }

  return message.contentParts.some((part) => {
    if (part.type === "text" || part.type === "thinking") {
      return part.text.trim().length > 0;
    }

    return part.type === "tool_use";
  });
}

interface AssistantTimelineEntry {
  message: Message;
  index: number;
  timestampMs: number | null;
}

interface TurnTimelineEntry {
  turn: AgentThreadTurn;
  startMs: number | null;
  targetMs: number | null;
}

interface AssistantMessageWindow {
  startMs: number | null;
  endMs: number | null;
}

function normalizeRuntimeTurnId(message: Message): string | null {
  const normalized = message.runtimeTurnId?.trim();
  return normalized || null;
}

function buildAssistantMessageWindowById(
  messages: Message[],
): Map<string, AssistantMessageWindow> {
  const windows = new Map<string, AssistantMessageWindow>();
  let currentStartMs: number | null = null;
  let currentAssistantIds: string[] = [];

  const flushCurrentWindow = (endMs: number | null) => {
    for (const assistantId of currentAssistantIds) {
      windows.set(assistantId, {
        startMs: currentStartMs,
        endMs,
      });
    }
    currentAssistantIds = [];
  };

  for (const message of messages) {
    if (message.role === "user") {
      const nextStartMs = resolveTimestampMs(message.timestamp);
      flushCurrentWindow(nextStartMs);
      currentStartMs = nextStartMs;
      continue;
    }

    if (message.role === "assistant") {
      currentAssistantIds.push(message.id);
    }
  }

  flushCurrentWindow(null);
  return windows;
}

function isAssistantInsideTurnMessageWindow(
  assistant: AssistantTimelineEntry,
  turn: TurnTimelineEntry,
  windowByMessageId: Map<string, AssistantMessageWindow>,
): boolean {
  const window = windowByMessageId.get(assistant.message.id);
  if (!window) {
    return true;
  }

  const turnStartMs = turn.startMs ?? turn.targetMs;
  if (turnStartMs === null) {
    return true;
  }
  if (window.startMs !== null && turnStartMs < window.startMs) {
    return false;
  }
  if (window.endMs !== null && turnStartMs >= window.endMs) {
    return false;
  }
  return true;
}

function resolveAssistantDistance(
  assistant: AssistantTimelineEntry,
  targetMs: number | null,
): number {
  if (assistant.timestampMs === null || targetMs === null) {
    return Number.POSITIVE_INFINITY;
  }

  return Math.abs(assistant.timestampMs - targetMs);
}

function pickPreferredAssistantMessage(
  current: AssistantTimelineEntry,
  candidate: AssistantTimelineEntry,
  targetMs: number | null,
): AssistantTimelineEntry {
  const currentDistance = resolveAssistantDistance(current, targetMs);
  const candidateDistance = resolveAssistantDistance(candidate, targetMs);

  if (candidateDistance < currentDistance) {
    return candidate;
  }

  if (candidateDistance !== currentDistance) {
    return current;
  }

  const candidateSubstantive = isSubstantiveAssistantMessage(candidate.message);
  const currentSubstantive = isSubstantiveAssistantMessage(current.message);

  if (candidateSubstantive !== currentSubstantive) {
    return candidateSubstantive ? candidate : current;
  }

  const candidateTimestamp = candidate.timestampMs ?? Number.NEGATIVE_INFINITY;
  const currentTimestamp = current.timestampMs ?? Number.NEGATIVE_INFINITY;

  if (candidateTimestamp > currentTimestamp) {
    return candidate;
  }

  if (
    candidateTimestamp === currentTimestamp &&
    candidate.index > current.index
  ) {
    return candidate;
  }

  return current;
}

function pickClosestUnassignedAssistantMessage(
  assistants: AssistantTimelineEntry[],
  assignedMessageIds: Set<string>,
  targetMs: number | null,
  predicate: (assistant: AssistantTimelineEntry) => boolean,
): AssistantTimelineEntry | null {
  let best: AssistantTimelineEntry | null = null;

  for (const assistant of assistants) {
    if (assignedMessageIds.has(assistant.message.id) || !predicate(assistant)) {
      continue;
    }

    best = best
      ? pickPreferredAssistantMessage(best, assistant, targetMs)
      : assistant;
  }

  return best;
}

function pickLastUnassignedAssistantMessage(
  assistants: AssistantTimelineEntry[],
  assignedMessageIds: Set<string>,
): AssistantTimelineEntry | null {
  for (let index = assistants.length - 1; index >= 0; index -= 1) {
    const assistant = assistants[index];
    if (assistant && !assignedMessageIds.has(assistant.message.id)) {
      return assistant;
    }
  }

  return null;
}

export function buildMessageTurnTimeline(
  messages: Message[],
  turns: AgentThreadTurn[],
  items: AgentThreadItem[],
): Map<string, MessageTurnTimeline> {
  if (turns.length === 0) {
    return new Map();
  }

  const assistantEntries: AssistantTimelineEntry[] = [];
  const explicitAssistantByTurnId = new Map<string, AssistantTimelineEntry>();
  for (const message of messages) {
    if (message.role !== "assistant") {
      continue;
    }

    const assistantEntry = {
      message,
      index: assistantEntries.length,
      timestampMs: resolveTimestampMs(message.timestamp),
    };
    assistantEntries.push(assistantEntry);

    const runtimeTurnId = normalizeRuntimeTurnId(message);
    if (runtimeTurnId && !explicitAssistantByTurnId.has(runtimeTurnId)) {
      explicitAssistantByTurnId.set(runtimeTurnId, assistantEntry);
    }
  }

  if (assistantEntries.length === 0) {
    return new Map();
  }

  const turnEntries: TurnTimelineEntry[] = resolveSortedThreadTurns(turns).map(
    (turn) => {
      const startMs = resolveTimestampMs(turn.started_at);
      return {
        turn,
        startMs,
        targetMs: resolveTimestampMs(turn.completed_at) ?? startMs,
      };
    },
  );

  const itemsByTurnId = new Map<string, AgentThreadItem[]>();
  for (const item of resolveTimelineThreadItems(items)) {
    const existing = itemsByTurnId.get(item.turn_id);
    if (existing) {
      existing.push(item);
    } else {
      itemsByTurnId.set(item.turn_id, [item]);
    }
  }

  const timelineByMessageId = new Map<string, MessageTurnTimeline>();
  const assignedMessageIds = new Set<string>();
  const windowByMessageId = buildAssistantMessageWindowById(messages);

  turnEntries.forEach((turnEntry, index) => {
    if (assignedMessageIds.size >= assistantEntries.length) {
      return;
    }

    const nextTurnStartMs =
      index < turnEntries.length - 1 ? turnEntries[index + 1]?.startMs : null;
    const singleAssistantWindow =
      assistantEntries.length === 1
        ? windowByMessageId.get(assistantEntries[0]!.message.id)
        : null;
    const canUseSingleAssistantFallback =
      assistantEntries.length === 1 && singleAssistantWindow?.endMs === null;

    const assistantMessage =
      (() => {
        const explicitAssistant = explicitAssistantByTurnId.get(
          turnEntry.turn.id,
        );
        if (
          explicitAssistant &&
          !assignedMessageIds.has(explicitAssistant.message.id)
        ) {
          return explicitAssistant;
        }
        return null;
      })() ||
      pickClosestUnassignedAssistantMessage(
        assistantEntries,
        assignedMessageIds,
        turnEntry.targetMs,
        (assistant) => {
          if (
            !isAssistantInsideTurnMessageWindow(
              assistant,
              turnEntry,
              windowByMessageId,
            )
          ) {
            return false;
          }
          if (assistant.timestampMs === null) {
            return true;
          }
          if (
            turnEntry.startMs !== null &&
            assistant.timestampMs < turnEntry.startMs
          ) {
            return false;
          }
          if (
            nextTurnStartMs !== null &&
            assistant.timestampMs >= nextTurnStartMs
          ) {
            return false;
          }
          return true;
        },
      ) ||
      pickClosestUnassignedAssistantMessage(
        assistantEntries,
        assignedMessageIds,
        turnEntry.targetMs,
        (assistant) => {
          if (
            !isAssistantInsideTurnMessageWindow(
              assistant,
              turnEntry,
              windowByMessageId,
            )
          ) {
            return false;
          }
          if (assistant.timestampMs === null || turnEntry.startMs === null) {
            return true;
          }
          return assistant.timestampMs >= turnEntry.startMs;
        },
      ) ||
      (turnEntry.startMs === null || canUseSingleAssistantFallback
        ? pickLastUnassignedAssistantMessage(
            assistantEntries,
            assignedMessageIds,
          )
        : null);

    if (!assistantMessage) {
      return;
    }

    assignedMessageIds.add(assistantMessage.message.id);
    timelineByMessageId.set(assistantMessage.message.id, {
      messageId: assistantMessage.message.id,
      turn: turnEntry.turn,
      items: itemsByTurnId.get(turnEntry.turn.id) || [],
    });
  });

  return timelineByMessageId;
}
