import type { AgentRuntimeThreadReadModel } from "@/lib/api/agentRuntime/sessionTypes";
import type { AgentThreadItem, AgentThreadTurn, Message } from "../types";
import {
  filterConversationThreadItems,
  mergeThreadItems,
} from "../utils/threadTimelineView";
import {
  messageHasInterruptedPlaceholder,
  stripInterruptedPlaceholderText,
} from "./agentInterruptedMessageContent";

export type AgentSessionDetailMergeMode =
  | "history_hydrate"
  | "runtime_sync"
  | "terminal_reconcile";

export interface AgentSessionTimelineMergeDecision {
  mode: AgentSessionDetailMergeMode;
  hasIncomingTerminalTimeline: boolean;
  isLocalTimelineCompatible: boolean;
  shouldPreserveByRuntimeSync: boolean;
  shouldPreserveBySession: boolean;
  shouldIgnoreIncompatibleHydratedMessages: boolean;
}

export function normalizeAgentSessionDetailMergeMode(
  mode?: AgentSessionDetailMergeMode | null,
): AgentSessionDetailMergeMode {
  return mode ?? "history_hydrate";
}

function normalizeConversationText(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function firstUserMessageText(messages: Message[]): string {
  return normalizeConversationText(
    messages.find((message) => message.role === "user")?.content || "",
  );
}

function userMessageTexts(messages: Message[]): string[] {
  return messages
    .filter((message) => message.role === "user")
    .map((message) => normalizeConversationText(message.content || ""));
}

function areConversationTextsCompatible(left: string, right: string): boolean {
  if (!left || !right) {
    return true;
  }

  return left === right || left.includes(right) || right.includes(left);
}

export function isLocalTimelineCompatibleWithHydratedMessages(params: {
  hydratedMessages: Message[];
  localMessages: Message[];
}): boolean {
  const hydratedUserTexts = userMessageTexts(params.hydratedMessages);
  const localUserTexts = userMessageTexts(params.localMessages);
  if (hydratedUserTexts.length === 0 || localUserTexts.length === 0) {
    return true;
  }

  const compareCount = Math.min(
    hydratedUserTexts.length,
    localUserTexts.length,
  );
  for (let index = 0; index < compareCount; index += 1) {
    if (
      !areConversationTextsCompatible(
        localUserTexts[index] || "",
        hydratedUserTexts[index] || "",
      )
    ) {
      return false;
    }
  }

  return true;
}

export function hasAssistantActivitySnapshot(messages: Message[]): boolean {
  return messages.some(
    (message) =>
      message.role === "assistant" &&
      (message.isThinking === true ||
        Boolean(message.runtimeStatus) ||
        Boolean(message.thinkingContent?.trim()) ||
        Boolean(message.contentParts?.some((part) => part.type !== "text")) ||
        messageHasInterruptedPlaceholder(message) ||
        Boolean(message.runtimeTurnId?.trim().startsWith("skill-exec-")) ||
        message.inlineProcessRetention === "skill"),
  );
}

export function shouldSkipStaleEmptyMessagesRefSync(params: {
  nextMessages: Message[];
  currentRefMessages: Message[];
}): boolean {
  return (
    params.nextMessages.length === 0 &&
    params.currentRefMessages.length > 0 &&
    hasAssistantActivitySnapshot(params.currentRefMessages)
  );
}

export function shouldPreserveDetachedLocalSnapshot(params: {
  hydratedMessages: Message[];
  localMessages: Message[];
  sessionId: string | null;
}): boolean {
  if (
    params.sessionId !== null ||
    !hasAssistantActivitySnapshot(params.localMessages)
  ) {
    return false;
  }

  const localUserText = firstUserMessageText(params.localMessages);
  const hydratedUserText = firstUserMessageText(params.hydratedMessages);
  if (!localUserText || !hydratedUserText) {
    return false;
  }

  return areConversationTextsCompatible(localUserText, hydratedUserText);
}

export function mergeRuntimeSyncThreadItems(
  localItems: readonly AgentThreadItem[],
  incomingItems: readonly AgentThreadItem[],
): AgentThreadItem[] {
  const localItemById = new Map(localItems.map((item) => [item.id, item]));
  return filterConversationThreadItems(
    mergeThreadItems([...localItems], [...incomingItems]),
  ).map((item) => {
    const localItem = localItemById.get(item.id);
    if (item.type !== "agent_message" || localItem?.type !== "agent_message") {
      return item;
    }

    const localText = stripInterruptedPlaceholderText(localItem.text);
    const incomingText = stripInterruptedPlaceholderText(item.text);
    if (
      !localText ||
      (incomingText && !localText.includes(incomingText)) ||
      incomingText.length >= localText.length
    ) {
      return item;
    }

    return {
      ...item,
      text: localItem.text,
    };
  });
}

function readRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function readArray(record: Record<string, unknown>, keys: string[]): unknown[] {
  for (const key of keys) {
    const value = record[key];
    if (Array.isArray(value)) {
      return value;
    }
  }
  return [];
}

function normalizeStatus(value: unknown): string {
  return typeof value === "string"
    ? value
        .trim()
        .toLowerCase()
        .replace(/[\s-]+/g, "_")
    : "";
}

function isTerminalTimelineStatus(status: unknown): boolean {
  const normalizedStatus = normalizeStatus(status);
  return (
    Boolean(normalizedStatus) && !ACTIVE_TIMELINE_STATUSES.has(normalizedStatus)
  );
}

function hasTerminalStatusInRecord(
  record: Record<string, unknown>,
  keys: string[],
): boolean {
  return keys.some((key) => isTerminalTimelineStatus(record[key]));
}

function hasTerminalThreadReadTimeline(
  threadRead?: AgentRuntimeThreadReadModel | null,
): boolean {
  if (!threadRead) {
    return false;
  }
  const record = threadRead as unknown as Record<string, unknown>;
  if (
    hasTerminalStatusInRecord(record, [
      "status",
      "profile_status",
      "profileStatus",
      "latest_turn_status",
      "latestTurnStatus",
    ])
  ) {
    return true;
  }

  return readArray(record, ["turns"]).some((turn) => {
    const turnRecord = readRecord(turn);
    return turnRecord
      ? hasTerminalStatusInRecord(turnRecord, [
          "status",
          "profile_status",
          "profileStatus",
          "native_status",
          "nativeStatus",
        ])
      : false;
  });
}

const ACTIVE_TIMELINE_STATUSES = new Set([
  "idle",
  "queued",
  "running",
  "waiting_request",
]);

export function hasTerminalDetailTimeline(detail: {
  thread_read?: AgentRuntimeThreadReadModel | null;
  turns?: AgentThreadTurn[];
}): boolean {
  if (hasTerminalThreadReadTimeline(detail.thread_read)) {
    return true;
  }

  return (detail.turns || []).some((turn) => {
    return isTerminalTimelineStatus(turn.status);
  });
}

export function resolveAgentSessionTimelineMergeDecision(params: {
  mode?: AgentSessionDetailMergeMode | null;
  mayPreserveExistingTimelineBySession: boolean;
  hydratedMessagesForCompatibility: Message[];
  localMessages: Message[];
  threadRead?: AgentRuntimeThreadReadModel | null;
  incomingTurns: AgentThreadTurn[];
}): AgentSessionTimelineMergeDecision {
  const mode = normalizeAgentSessionDetailMergeMode(params.mode);
  const hasIncomingTerminalTimeline = hasTerminalDetailTimeline({
    thread_read: params.threadRead,
    turns: params.incomingTurns,
  });
  const isLocalTimelineCompatible =
    isLocalTimelineCompatibleWithHydratedMessages({
      hydratedMessages: params.hydratedMessagesForCompatibility,
      localMessages: params.localMessages,
    });
  const shouldPreserveByRuntimeSync =
    params.mayPreserveExistingTimelineBySession &&
    mode === "runtime_sync" &&
    !hasIncomingTerminalTimeline;
  const shouldPreserveBySession =
    shouldPreserveByRuntimeSync ||
    (params.mayPreserveExistingTimelineBySession && isLocalTimelineCompatible);

  return {
    mode,
    hasIncomingTerminalTimeline,
    isLocalTimelineCompatible,
    shouldPreserveByRuntimeSync,
    shouldPreserveBySession,
    shouldIgnoreIncompatibleHydratedMessages:
      shouldPreserveByRuntimeSync && !isLocalTimelineCompatible,
  };
}
