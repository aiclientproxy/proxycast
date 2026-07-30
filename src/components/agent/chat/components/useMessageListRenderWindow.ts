import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { scheduleMinimumDelayIdleTask } from "@/lib/utils/scheduleMinimumDelayIdleTask";
import {
  buildCurrentTurnTimelineProjection,
  buildMessageGroupsProjection,
  buildMessageRenderGroupsProjection,
  buildTimelineByMessageIdProjection,
  resolveLastAssistantMessage,
} from "../projection/messageTimelineRenderProjection";
import {
  buildTurnTimelineRenderProjection,
  type TurnTimelineRenderEntry,
} from "../projection/turnTimelineRenderProjection";
import { filterVisibleConversationMessages } from "../projection/messageRenderWindowProjection";
import type { AgentThreadItem, AgentThreadTurn, Message } from "../types";
import { MESSAGE_LIST_RENDER_WINDOW_SETTINGS } from "./messageListConstants";

interface SessionHistoryWindow {
  hasMore?: boolean;
}

interface UseMessageListRenderWindowOptions {
  currentTurnId: string | null;
  isSending: boolean;
  isRestoringSession: boolean;
  isUserScrolling: boolean;
  messages: Message[];
  sessionHistoryWindow: SessionHistoryWindow | null;
  threadItems: readonly AgentThreadItem[];
  turns: readonly AgentThreadTurn[];
}

interface RenderEntryWindowSnapshot {
  firstId: string | null;
  lastId: string | null;
  length: number;
}

function hasPersistedOlderHistory(
  historyWindow: SessionHistoryWindow | null,
): boolean {
  return Boolean(historyWindow && historyWindow.hasMore !== false);
}

function buildOwnershipRenderEntries(params: {
  currentTurnId: string | null;
  messages: Message[];
  threadItems: readonly AgentThreadItem[];
  turns: readonly AgentThreadTurn[];
}): TurnTimelineRenderEntry[] {
  const messageGroups = buildMessageGroupsProjection(params.messages);
  const timelineByMessageId = buildTimelineByMessageIdProjection({
    canBuildHistoricalTimeline: true,
    renderedMessages: params.messages,
    renderedTurns: [...params.turns],
    renderedThreadItems: [...params.threadItems],
  });
  const lastAssistantMessageId =
    resolveLastAssistantMessage(params.messages)?.id ?? null;
  const activeCurrentTurn =
    params.turns.find((turn) => turn.id === params.currentTurnId) ?? null;
  const currentTurnTimeline = buildCurrentTurnTimelineProjection({
    activeCurrentTurnId: params.currentTurnId,
    activeCurrentTurn,
    lastAssistantMessageId,
    timelineByMessageId,
    renderedThreadItems: [...params.threadItems],
    renderedMessages: params.messages,
  });
  const ownershipGroups = buildMessageRenderGroupsProjection({
    messageGroups,
    timelineByMessageId,
    currentTurnTimeline,
    lastAssistantMessageId,
  });

  return buildTurnTimelineRenderProjection({
    messageGroups: ownershipGroups,
    renderedTurns: params.turns,
    renderedThreadItems: params.threadItems,
    currentTurnId: params.currentTurnId,
  });
}

function collectRenderedEntryOwnership(
  entries: readonly TurnTimelineRenderEntry[],
): { messageIds: Set<string>; turnIds: Set<string> } {
  const messageIds = new Set<string>();
  const turnIds = new Set<string>();

  for (const entry of entries) {
    if (entry.kind === "canonical_turn") {
      turnIds.add(entry.turn.id);
      for (const segment of entry.segments) {
        if (segment.kind !== "message") continue;
        messageIds.add(segment.item.id);
        if (segment.item.type === "user_message" && segment.item.client_id) {
          messageIds.add(segment.item.client_id);
        }
      }
      continue;
    }

    const timelineTurnId = entry.group.timeline?.turn.id.trim();
    if (timelineTurnId) turnIds.add(timelineTurnId);
    for (const message of entry.group.messages) {
      messageIds.add(message.id);
      const turnId = message.runtimeTurnId?.trim();
      if (turnId) turnIds.add(turnId);
    }
  }

  return { messageIds, turnIds };
}

function selectRenderedEntries(params: {
  entries: readonly TurnTimelineRenderEntry[];
  renderedEntryCount: number;
  pinnedEntryId: string | null;
}): TurnTimelineRenderEntry[] {
  if (params.renderedEntryCount >= params.entries.length) {
    return [...params.entries];
  }

  const tail = params.entries.slice(-params.renderedEntryCount);
  if (
    !params.pinnedEntryId ||
    tail.some((entry) => entry.id === params.pinnedEntryId)
  ) {
    return tail;
  }

  const pinnedEntry = params.entries.find(
    (entry) => entry.id === params.pinnedEntryId,
  );
  if (!pinnedEntry) return tail;

  const selectedIds = new Set([
    pinnedEntry.id,
    ...tail.map((entry) => entry.id),
  ]);
  return params.entries.filter((entry) => selectedIds.has(entry.id));
}

export function useMessageListRenderWindow({
  currentTurnId,
  isSending,
  isRestoringSession,
  isUserScrolling,
  messages,
  sessionHistoryWindow,
  threadItems,
  turns,
}: UseMessageListRenderWindowOptions) {
  const visibleMessages = useMemo(
    () => filterVisibleConversationMessages(messages),
    [messages],
  );
  const visibleEntries = useMemo(
    () =>
      buildOwnershipRenderEntries({
        currentTurnId,
        messages: visibleMessages,
        threadItems,
        turns,
      }),
    [currentTurnId, threadItems, turns, visibleMessages],
  );
  const visibleEntryFirstId = visibleEntries[0]?.id ?? null;
  const visibleEntryLastId = visibleEntries.at(-1)?.id ?? null;
  const hasPersistedHistory = hasPersistedOlderHistory(sessionHistoryWindow);
  const isRestoredHistoryWindow = isRestoringSession || hasPersistedHistory;
  const [restoredPromptCacheNoticeReady, setRestoredPromptCacheNoticeReady] =
    useState(() => !isRestoredHistoryWindow);

  useEffect(() => {
    if (!isRestoredHistoryWindow) {
      setRestoredPromptCacheNoticeReady(true);
      return;
    }

    setRestoredPromptCacheNoticeReady(false);
    return scheduleMinimumDelayIdleTask(
      () => {
        setRestoredPromptCacheNoticeReady(true);
      },
      {
        minimumDelayMs: 1_500,
        idleTimeoutMs: 3_000,
      },
    );
  }, [isRestoredHistoryWindow, visibleEntryFirstId, visibleEntryLastId]);

  const renderWindowSettings = isRestoredHistoryWindow
    ? MESSAGE_LIST_RENDER_WINDOW_SETTINGS.restored
    : MESSAGE_LIST_RENDER_WINDOW_SETTINGS.regular;
  const progressiveInitialRenderCount = renderWindowSettings.initialRenderCount;
  const progressiveRenderBatchSize = renderWindowSettings.renderBatchSize;
  const progressiveRenderMinimumDelayMs = renderWindowSettings.minimumDelayMs;
  const shouldUseProgressiveRender =
    (!isSending || isRestoredHistoryWindow) &&
    visibleEntries.length > progressiveInitialRenderCount;
  const visibleEntryWindowRef = useRef<RenderEntryWindowSnapshot | null>(null);
  const [renderedEntryCount, setRenderedEntryCount] = useState(() =>
    shouldUseProgressiveRender
      ? Math.min(visibleEntries.length, progressiveInitialRenderCount)
      : visibleEntries.length,
  );

  useEffect(() => {
    const previousWindow = visibleEntryWindowRef.current;
    visibleEntryWindowRef.current = {
      firstId: visibleEntryFirstId,
      lastId: visibleEntryLastId,
      length: visibleEntries.length,
    };

    if (!shouldUseProgressiveRender) {
      setRenderedEntryCount(visibleEntries.length);
      return;
    }

    const isAppendOnlyUpdate =
      previousWindow !== null &&
      previousWindow.firstId === visibleEntryFirstId &&
      previousWindow.length <= visibleEntries.length &&
      previousWindow.lastId !== visibleEntryLastId;

    if (!isAppendOnlyUpdate) {
      setRenderedEntryCount(
        Math.min(visibleEntries.length, progressiveInitialRenderCount),
      );
      return;
    }

    const appendedCount = visibleEntries.length - previousWindow.length;
    if (appendedCount <= 0) return;

    setRenderedEntryCount((current) =>
      Math.min(
        visibleEntries.length,
        Math.max(current + appendedCount, progressiveInitialRenderCount),
      ),
    );
  }, [
    progressiveInitialRenderCount,
    shouldUseProgressiveRender,
    visibleEntries.length,
    visibleEntryFirstId,
    visibleEntryLastId,
  ]);

  const normalizedRenderedEntryCount = shouldUseProgressiveRender
    ? Math.min(visibleEntries.length, Math.max(0, renderedEntryCount))
    : visibleEntries.length;
  const pinnedEntryId = useMemo(() => {
    if (!currentTurnId) return null;
    return (
      visibleEntries.find((entry) => {
        if (entry.kind === "canonical_turn") {
          return entry.turn.id === currentTurnId;
        }
        if (entry.group.timeline?.turn.id === currentTurnId) return true;
        return entry.group.messages.some(
          (message) => message.runtimeTurnId?.trim() === currentTurnId,
        );
      })?.id ?? null
    );
  }, [currentTurnId, visibleEntries]);
  const renderedEntries = useMemo(
    () =>
      selectRenderedEntries({
        entries: visibleEntries,
        renderedEntryCount: normalizedRenderedEntryCount,
        pinnedEntryId,
      }),
    [normalizedRenderedEntryCount, pinnedEntryId, visibleEntries],
  );
  const hiddenHistoryCount = Math.max(
    0,
    visibleEntries.length - renderedEntries.length,
  );
  const shouldAutoHydrateHiddenHistory =
    shouldUseProgressiveRender && !isRestoredHistoryWindow;

  useEffect(() => {
    if (
      !shouldAutoHydrateHiddenHistory ||
      hiddenHistoryCount <= 0 ||
      isUserScrolling
    ) {
      return;
    }

    return scheduleMinimumDelayIdleTask(
      () => {
        setRenderedEntryCount((current) =>
          Math.min(visibleEntries.length, current + progressiveRenderBatchSize),
        );
      },
      {
        minimumDelayMs: progressiveRenderMinimumDelayMs,
        idleTimeoutMs: 1_200,
      },
    );
  }, [
    hiddenHistoryCount,
    isUserScrolling,
    progressiveRenderBatchSize,
    progressiveRenderMinimumDelayMs,
    shouldAutoHydrateHiddenHistory,
    visibleEntries.length,
  ]);

  const renderedOwnership = useMemo(
    () => collectRenderedEntryOwnership(renderedEntries),
    [renderedEntries],
  );
  const renderedMessages = useMemo(
    () =>
      visibleMessages.filter((message) => {
        if (renderedOwnership.messageIds.has(message.id)) return true;
        const turnId = message.runtimeTurnId?.trim();
        return Boolean(turnId && renderedOwnership.turnIds.has(turnId));
      }),
    [renderedOwnership, visibleMessages],
  );
  const renderedTurns = useMemo(
    () => turns.filter((turn) => renderedOwnership.turnIds.has(turn.id)),
    [renderedOwnership, turns],
  );
  const renderedThreadItems = useMemo(
    () =>
      threadItems.filter((item) => renderedOwnership.turnIds.has(item.turn_id)),
    [renderedOwnership, threadItems],
  );
  const handleExpandAllHistory = useCallback(() => {
    setRenderedEntryCount(visibleEntries.length);
  }, [visibleEntries.length]);

  return {
    handleExpandAllHistory,
    hasPersistedOlderHistory: hasPersistedHistory,
    hiddenHistoryCount,
    isRestoredHistoryWindow,
    renderedEntryCount: renderedEntries.length,
    renderedMessages,
    renderedThreadItems,
    renderedTurns,
    restoredPromptCacheNoticeReady,
    visibleEntries,
    visibleMessages,
  };
}
