import { useEffect, useMemo, useState } from "react";
import { scheduleMinimumDelayIdleTask } from "@/lib/utils/scheduleMinimumDelayIdleTask";
import type { AgentRuntimeThreadReadModel } from "@/lib/api/agentRuntime/sessionTypes";
import type { CanonicalChildThreadSummary } from "../projection/canonicalChildThreadSummary";
import { buildInputbarRuntimeStatusLineModel } from "../utils/inputbarRuntimeStatusLine";
import {
  filterConversationThreadItemsForRenderedTurns,
  resolveConversationRenderedTurnIdSet,
} from "../projection/threadTimelineWindowProjection";
import {
  buildCurrentTurnTimelineProjection,
  buildMessageGroupsProjection,
  buildMessageRenderGroupsProjection,
  buildTimelineByMessageIdProjection,
  resolveLastAssistantMessage,
} from "../projection/messageTimelineRenderProjection";
import { buildTurnTimelineRenderProjection } from "../projection/turnTimelineRenderProjection";
import type {
  ActionRequired,
  AgentThreadItem,
  AgentThreadTurn,
  Message,
  PendingA2UISource,
} from "../types";
import {
  MESSAGE_LIST_HISTORICAL_TIMELINE_IDLE_DELAY_MS,
  MESSAGE_LIST_RESTORED_HISTORICAL_TIMELINE_IDLE_DELAY_MS,
  MESSAGE_LIST_TIMELINE_DEFER_ITEM_THRESHOLD,
  MESSAGE_LIST_TIMELINE_DEFER_MESSAGE_THRESHOLD,
} from "./messageListConstants";
import { measureMessageListComputation } from "./messageListPerformance";

interface UseMessageListTimelineStateOptions {
  activePendingA2UISource: PendingA2UISource | null;
  canonicalChildren: CanonicalChildThreadSummary[];
  currentTurnId: string | null;
  focusedTimelineItemId: string | null;
  hiddenHistoryCount: number;
  isRestoredHistoryWindow: boolean;
  isSending: boolean;
  pendingActions: readonly ActionRequired[];
  hasPersistedOlderHistory: boolean;
  renderedMessages: Message[];
  submittedActionsInFlight: readonly ActionRequired[];
  threadItems: readonly AgentThreadItem[];
  threadRead: AgentRuntimeThreadReadModel | null;
  turns: readonly AgentThreadTurn[];
}

export function useMessageListTimelineState({
  activePendingA2UISource,
  canonicalChildren,
  currentTurnId,
  focusedTimelineItemId,
  hiddenHistoryCount,
  isRestoredHistoryWindow,
  isSending,
  pendingActions,
  hasPersistedOlderHistory,
  renderedMessages,
  submittedActionsInFlight,
  threadItems,
  threadRead,
  turns,
}: UseMessageListTimelineStateOptions) {
  // Turn 窗口由 useMessageListRenderWindow 选择；这里仅补充已选 entry 的展示投影。
  const renderedTurns = useMemo(() => [...turns], [turns]);
  const renderedTurnIdSet = useMemo(() => {
    return resolveConversationRenderedTurnIdSet({
      renderedTurns,
      hiddenHistoryCount,
      isRestoredHistoryWindow,
    });
  }, [hiddenHistoryCount, isRestoredHistoryWindow, renderedTurns]);
  const activeCurrentTurn = useMemo(() => {
    if (!currentTurnId) {
      return null;
    }

    return renderedTurns.find((entry) => entry.id === currentTurnId) ?? null;
  }, [currentTurnId, renderedTurns]);
  const activeCurrentTurnId =
    activeCurrentTurn &&
    (activeCurrentTurn.status === "running" ||
      activeCurrentTurn.status === "failed")
      ? activeCurrentTurn.id
      : null;
  const timelineHydrationKey = [
    renderedMessages[renderedMessages.length - 1]?.id ?? "no-message",
    renderedTurns[renderedTurns.length - 1]?.id ?? "no-turn",
    `${threadItems.length}:${
      threadItems[threadItems.length - 1]?.id ?? "no-item"
    }`,
  ].join("|");
  const hasLargeHistoricalThreadItems =
    threadItems.length >= MESSAGE_LIST_TIMELINE_DEFER_ITEM_THRESHOLD;
  const hasHistoricalWindow =
    isRestoredHistoryWindow ||
    hiddenHistoryCount > 0 ||
    hasPersistedOlderHistory;
  const shouldProtectHistoricalWindowDuringSending =
    isSending &&
    hasHistoricalWindow &&
    !activeCurrentTurnId &&
    !focusedTimelineItemId &&
    hasLargeHistoricalThreadItems;
  const shouldDeferHistoricalTimeline =
    !activeCurrentTurnId &&
    !focusedTimelineItemId &&
    hasLargeHistoricalThreadItems &&
    (shouldProtectHistoricalWindowDuringSending ||
      (!isSending &&
        (isRestoredHistoryWindow ||
          renderedMessages.length >=
            MESSAGE_LIST_TIMELINE_DEFER_MESSAGE_THRESHOLD)));
  const shouldDeferHistoricalTimelineDetails =
    !focusedTimelineItemId &&
    (shouldDeferHistoricalTimeline ||
      hiddenHistoryCount > 0 ||
      hasPersistedOlderHistory);
  const [isHistoricalTimelineReady, setIsHistoricalTimelineReady] = useState(
    () => !shouldDeferHistoricalTimeline,
  );

  useEffect(() => {
    if (!shouldDeferHistoricalTimeline) {
      setIsHistoricalTimelineReady(true);
      return;
    }

    setIsHistoricalTimelineReady(false);
    return scheduleMinimumDelayIdleTask(
      () => {
        setIsHistoricalTimelineReady(true);
      },
      {
        minimumDelayMs: isRestoredHistoryWindow
          ? MESSAGE_LIST_RESTORED_HISTORICAL_TIMELINE_IDLE_DELAY_MS
          : MESSAGE_LIST_HISTORICAL_TIMELINE_IDLE_DELAY_MS,
        idleTimeoutMs: isRestoredHistoryWindow ? 1_800 : 900,
      },
    );
  }, [
    isRestoredHistoryWindow,
    shouldDeferHistoricalTimeline,
    timelineHydrationKey,
  ]);

  const canBuildHistoricalTimeline =
    !shouldDeferHistoricalTimeline || isHistoricalTimelineReady;
  const shouldDeferTailRuntimeStatusLine =
    isRestoredHistoryWindow &&
    shouldDeferHistoricalTimeline &&
    !isHistoricalTimelineReady &&
    !isSending &&
    !activeCurrentTurnId &&
    pendingActions.length === 0 &&
    (threadRead?.pending_requests?.length ?? 0) === 0;
  // 首帧必须确定 canonical Item owner，历史细节延迟期间不能退回 Message 窗口。
  const shouldDeferThreadItemsScan = false;
  const renderedThreadItemsMeasurement = useMemo(
    () =>
      measureMessageListComputation(() =>
        filterConversationThreadItemsForRenderedTurns({
          threadItems,
          renderedTurnIdSet,
          shouldDeferThreadItemsScan,
        }),
      ),
    [renderedTurnIdSet, shouldDeferThreadItemsScan, threadItems],
  );
  const renderedThreadItems = renderedThreadItemsMeasurement.value;
  const timelineByMessageIdMeasurement = useMemo(
    () =>
      measureMessageListComputation(() =>
        buildTimelineByMessageIdProjection({
          canBuildHistoricalTimeline: true,
          renderedMessages,
          renderedTurns,
          renderedThreadItems,
        }),
      ),
    [renderedMessages, renderedThreadItems, renderedTurns],
  );
  const timelineByMessageId = timelineByMessageIdMeasurement.value;
  const lastAssistantMessage = useMemo(
    () => resolveLastAssistantMessage(renderedMessages),
    [renderedMessages],
  );
  const lastAssistantMessageId = lastAssistantMessage?.id ?? null;
  const hasActiveInteractiveRuntime =
    isSending ||
    Boolean(activeCurrentTurnId) ||
    pendingActions.length > 0 ||
    (threadRead?.pending_requests?.length ?? 0) > 0 ||
    Boolean(activePendingA2UISource);
  const hasRuntimeStatusLineEvidence =
    hasActiveInteractiveRuntime ||
    turns.length > 0 ||
    threadItems.length > 0 ||
    canonicalChildren.length > 0;
  const activeConversationRuntimeStatusLine = useMemo(() => {
    if (!hasRuntimeStatusLineEvidence) {
      return null;
    }

    return buildInputbarRuntimeStatusLineModel({
      messages: renderedMessages,
      turns: renderedTurns,
      threadItems: renderedThreadItems,
      currentTurnId: activeCurrentTurnId,
      threadRead,
      pendingActions,
      submittedActionsInFlight,
      canonicalChildren,
      isSending,
    });
  }, [
    activeCurrentTurnId,
    canonicalChildren,
    hasRuntimeStatusLineEvidence,
    isSending,
    pendingActions,
    renderedMessages,
    renderedThreadItems,
    renderedTurns,
    submittedActionsInFlight,
    threadRead,
  ]);
  const tailRuntimeStatusLine = useMemo(() => {
    if (!lastAssistantMessageId || shouldDeferTailRuntimeStatusLine) {
      return null;
    }

    return activeConversationRuntimeStatusLine;
  }, [
    activeConversationRuntimeStatusLine,
    lastAssistantMessageId,
    shouldDeferTailRuntimeStatusLine,
  ]);
  const currentTurnTimeline = useMemo(() => {
    return buildCurrentTurnTimelineProjection({
      // completed turn 仍需作为选中锚点投影 canonical ThreadStore 项，但不改变运行态 active id。
      activeCurrentTurnId: currentTurnId,
      activeCurrentTurn,
      lastAssistantMessageId,
      timelineByMessageId,
      renderedThreadItems,
      renderedMessages,
    });
  }, [
    activeCurrentTurn,
    currentTurnId,
    lastAssistantMessageId,
    renderedMessages,
    renderedThreadItems,
    timelineByMessageId,
  ]);
  const messageGroupsMeasurement = useMemo(
    () =>
      measureMessageListComputation(() =>
        buildMessageGroupsProjection(renderedMessages),
      ),
    [renderedMessages],
  );
  const messageGroups = messageGroupsMeasurement.value;
  const renderGroupsMeasurement = useMemo(
    () =>
      measureMessageListComputation(() =>
        buildMessageRenderGroupsProjection({
          messageGroups,
          timelineByMessageId,
          currentTurnTimeline,
          lastAssistantMessageId,
        }),
      ),
    [
      currentTurnTimeline,
      lastAssistantMessageId,
      messageGroups,
      timelineByMessageId,
    ],
  );
  const renderGroups = renderGroupsMeasurement.value;
  const renderEntriesMeasurement = useMemo(
    () =>
      measureMessageListComputation(() =>
        buildTurnTimelineRenderProjection({
          messageGroups: renderGroups,
          renderedTurns,
          renderedThreadItems,
          currentTurnId,
        }),
      ),
    [currentTurnId, renderGroups, renderedThreadItems, renderedTurns],
  );
  const renderEntries = renderEntriesMeasurement.value;

  return {
    activeConversationRuntimeStatusLine,
    activeCurrentTurnId,
    canBuildHistoricalTimeline,
    hasActiveInteractiveRuntime,
    isHistoricalTimelineReady,
    lastAssistantMessage,
    lastAssistantMessageId,
    messageGroups,
    messageGroupsMeasurement,
    renderedThreadItems,
    renderedThreadItemsMeasurement,
    renderedTurns,
    renderEntries,
    renderEntriesMeasurement,
    renderGroups,
    renderGroupsMeasurement,
    shouldDeferHistoricalTimeline,
    shouldDeferHistoricalTimelineDetails,
    shouldDeferTailRuntimeStatusLine,
    shouldDeferThreadItemsScan,
    tailRuntimeStatusLine,
    timelineByMessageIdMeasurement,
  };
}
