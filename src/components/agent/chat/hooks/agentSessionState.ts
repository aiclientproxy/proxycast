import type {
  AgentExecutionStrategy,
  AgentSessionExecutionRuntime,
} from "@/lib/api/agentExecutionRuntime";
import type {
  AgentRuntimeThreadReadModel,
  AgentTodoItem,
} from "@/lib/api/agentRuntime/sessionTypes";
import { resolveRestorableSessionId } from "@/lib/agentSessionRecovery";
import type { AgentThreadItem, AgentThreadTurn, Message } from "../types";
import type { Topic } from "./agentChatShared";
import type { AgentSessionCachedSnapshot } from "./agentSessionScopedStorage";
import { normalizeLegacyThreadItems } from "@/lib/api/agentTextNormalization";
import {
  hydrateSessionDetailMessages,
  mergeHydratedMessagesWithLocalState,
  normalizeHistoricalTopicSnapshotMessages,
  shouldCompactCompletedSessionHistory,
} from "./agentChatHistory";
import {
  filterConversationThreadItems,
  mergeThreadItems,
  mergeThreadTurns,
} from "../utils/threadTimelineView";
import {
  mergeRuntimeSyncThreadItems,
  hasTerminalDetailTimeline,
  normalizeAgentSessionDetailMergeMode,
  resolveAgentSessionTimelineMergeDecision,
  shouldPreserveDetachedLocalSnapshot,
  type AgentSessionDetailMergeMode,
} from "./agentSessionTimelineMergePolicy";
import { collectDetailThreadItems } from "./agentChatHistoryThreadItems";
import { createExecutionRuntimeFromSessionDetail } from "../utils/sessionExecutionRuntime";
import { normalizeExecutionStrategy } from "./agentChatCoreUtils";
import type { AgentRuntimeAdapter } from "./agentRuntimeAdapter";
import { isAuxiliaryAgentSessionId } from "@/lib/api/agentRuntime/sessionIdentity";
import {
  hasRunningThreadReadActivity,
  hasRunningTurnRecordActivity,
} from "../projection/threadReadActivity";
import { resolveFinalAgentMessageItemIds } from "../utils/agentMessagePhase";
import {
  buildInterruptedMessageContentPatch,
  markInterruptedAgentMessageThreadItems,
  messageHasInterruptedPlaceholder,
} from "./agentInterruptedMessageContent";
import { consumeLocallyInterruptedAgentStreamBinding } from "./agentStreamResumeBinding";

export interface AgentSessionSnapshot {
  sessionId: string | null;
  workingDir: string | null;
  messages: Message[];
  threadTurns: AgentThreadTurn[];
  threadItems: AgentThreadItem[];
  currentTurnId: string | null;
  threadRead: AgentRuntimeThreadReadModel | null;
  executionRuntime: AgentSessionExecutionRuntime | null;
  todoItems: AgentTodoItem[];
}

export type { AgentSessionDetailMergeMode } from "./agentSessionTimelineMergePolicy";

function normalizeTimelineStatus(value: unknown): string {
  return typeof value === "string"
    ? value
        .trim()
        .toLowerCase()
        .replace(/[\s-]+/g, "_")
    : "";
}

function isInterruptedTimelineStatus(value: unknown): boolean {
  const status = normalizeTimelineStatus(value);
  return (
    status === "canceled" ||
    status === "cancelled" ||
    status === "aborted" ||
    status === "interrupted"
  );
}

function readTimelineRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function readTimelineString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function readTimelineTurnId(value: unknown): string {
  const record = readTimelineRecord(value);
  if (!record) {
    return "";
  }
  return (
    readTimelineString(record.id) ||
    readTimelineString(record.turn_id) ||
    readTimelineString(record.turnId)
  );
}

function readTimelineItemTurnId(value: unknown): string {
  const record = readTimelineRecord(value);
  if (!record) {
    return "";
  }
  return (
    readTimelineString(record.turn_id) || readTimelineString(record.turnId)
  );
}

function readTimelineTurnStatus(value: unknown): unknown {
  const record = readTimelineRecord(value);
  if (!record) {
    return undefined;
  }
  return record.status ?? record.native_status ?? record.nativeStatus;
}

function collectInterruptedTurnIds(options: {
  items?: readonly unknown[];
  threadRead?: AgentRuntimeThreadReadModel | null;
  turns?: readonly unknown[];
}): Set<string> {
  const interruptedTurnIds = new Set<string>();

  for (const turn of options.turns || []) {
    if (!isInterruptedTimelineStatus(readTimelineTurnStatus(turn))) {
      continue;
    }
    const turnId = readTimelineTurnId(turn);
    if (turnId) {
      interruptedTurnIds.add(turnId);
    }
  }

  for (const turn of options.threadRead?.turns || []) {
    if (!isInterruptedTimelineStatus(readTimelineTurnStatus(turn))) {
      continue;
    }
    const turnId = readTimelineTurnId(turn);
    if (turnId) {
      interruptedTurnIds.add(turnId);
    }
  }

  const activeTurnId = readTimelineString(options.threadRead?.active_turn_id);
  if (
    activeTurnId &&
    (isInterruptedTimelineStatus(options.threadRead?.status) ||
      isInterruptedTimelineStatus(options.threadRead?.profile_status) ||
      isInterruptedTimelineStatus(
        options.threadRead?.diagnostics?.latest_turn_status,
      ))
  ) {
    interruptedTurnIds.add(activeTurnId);
  }

  if (
    interruptedTurnIds.size === 0 &&
    (isInterruptedTimelineStatus(options.threadRead?.status) ||
      isInterruptedTimelineStatus(options.threadRead?.profile_status))
  ) {
    for (const item of options.items || []) {
      const turnId = readTimelineItemTurnId(item);
      if (turnId) {
        interruptedTurnIds.add(turnId);
      }
    }
  }

  return interruptedTurnIds;
}

function collectLocalInterruptedCandidateTurnIds(options: {
  currentTurnId?: string | null;
  items?: readonly unknown[];
  threadRead?: AgentRuntimeThreadReadModel | null;
  turns?: readonly unknown[];
}): string[] {
  const candidateTurnIds: string[] = [];
  const seen = new Set<string>();
  const pushTurnId = (value: unknown) => {
    const turnId = readTimelineString(value);
    if (!turnId || turnId.startsWith("pending-turn:") || seen.has(turnId)) {
      return;
    }
    seen.add(turnId);
    candidateTurnIds.push(turnId);
  };

  pushTurnId(options.currentTurnId);
  pushTurnId(options.threadRead?.active_turn_id);

  for (let index = (options.turns?.length ?? 0) - 1; index >= 0; index -= 1) {
    pushTurnId(readTimelineTurnId(options.turns?.[index]));
  }

  const threadReadTurns = options.threadRead?.turns ?? [];
  for (let index = threadReadTurns.length - 1; index >= 0; index -= 1) {
    pushTurnId(readTimelineTurnId(threadReadTurns[index]));
  }

  for (let index = (options.items?.length ?? 0) - 1; index >= 0; index -= 1) {
    pushTurnId(readTimelineItemTurnId(options.items?.[index]));
  }

  return candidateTurnIds;
}

function collectLocallyInterruptedTurnIds(options: {
  candidateTurnIds: readonly string[];
  sessionId: string;
  threadId?: string | null;
}): Set<string> {
  const interruptedTurnIds = new Set<string>();
  const sessionId = readTimelineString(options.sessionId);
  if (!sessionId) {
    return interruptedTurnIds;
  }

  const threadId = readTimelineString(options.threadId) || sessionId;
  for (const turnId of options.candidateTurnIds) {
    if (
      consumeLocallyInterruptedAgentStreamBinding({
        eventName: `agentSession/event/${sessionId}`,
        sessionId,
        threadId,
        turnId,
      })
    ) {
      interruptedTurnIds.add(turnId);
      return interruptedTurnIds;
    }
  }

  return interruptedTurnIds;
}

function collectLocalInterruptedMarkerTurnIds(options: {
  currentMessages: readonly Message[];
  fallbackTurnId?: string | null;
  incomingItems?: readonly unknown[];
}): Set<string> {
  const interruptedTurnIds = new Set<string>();
  const pushTurnId = (value: unknown) => {
    const turnId = readTimelineString(value);
    if (!turnId || turnId.startsWith("pending-turn:")) {
      return;
    }
    interruptedTurnIds.add(turnId);
  };

  const hasLocalInterruptedMarker = options.currentMessages.some(
    (message) =>
      message.role === "assistant" && messageHasInterruptedPlaceholder(message),
  );
  if (!hasLocalInterruptedMarker) {
    return interruptedTurnIds;
  }

  for (const message of options.currentMessages) {
    if (
      message.role === "assistant" &&
      messageHasInterruptedPlaceholder(message)
    ) {
      pushTurnId(message.runtimeTurnId);
    }
  }

  // 没有具体运行回合归属时，才从当前/入站时间线推断停止回合。
  if (interruptedTurnIds.size === 0) {
    pushTurnId(options.fallbackTurnId);
    for (
      let index = (options.incomingItems?.length ?? 0) - 1;
      index >= 0;
      index -= 1
    ) {
      pushTurnId(readTimelineItemTurnId(options.incomingItems?.[index]));
    }
  }

  return interruptedTurnIds;
}

function markInterruptedAssistantMessages(
  messages: Message[],
  interruptedTurnIds: Set<string>,
  options?: {
    fallbackTurnId?: string | null;
  },
): Message[] {
  if (interruptedTurnIds.size === 0) {
    return messages;
  }
  let hasMarkedMessage = false;
  const nextMessages = messages.map((message) => {
    if (
      message.role !== "assistant" ||
      !message.runtimeTurnId ||
      !interruptedTurnIds.has(message.runtimeTurnId)
    ) {
      return message;
    }
    hasMarkedMessage = true;
    return {
      ...message,
      ...buildInterruptedMessageContentPatch(message),
      isThinking: false,
      runtimeStatus:
        message.runtimeStatus?.phase === "cancelled"
          ? message.runtimeStatus
          : undefined,
    };
  });
  const fallbackTurnId = readTimelineString(options?.fallbackTurnId);
  if (
    hasMarkedMessage ||
    !fallbackTurnId ||
    !interruptedTurnIds.has(fallbackTurnId)
  ) {
    return nextMessages;
  }

  let fallbackAssistantIndex = -1;
  for (let index = nextMessages.length - 1; index >= 0; index -= 1) {
    if (nextMessages[index]?.role === "assistant") {
      fallbackAssistantIndex = index;
      break;
    }
  }
  if (fallbackAssistantIndex < 0) {
    return nextMessages;
  }

  return nextMessages.map((message, index) =>
    index === fallbackAssistantIndex
      ? {
          ...message,
          ...buildInterruptedMessageContentPatch(message),
          isThinking: false,
          runtimeTurnId: message.runtimeTurnId?.startsWith("pending-turn:")
            ? fallbackTurnId
            : (message.runtimeTurnId ?? fallbackTurnId),
          runtimeStatus:
            message.runtimeStatus?.phase === "cancelled"
              ? message.runtimeStatus
              : undefined,
        }
      : message,
  );
}

function markInterruptedThreadItems(
  items: AgentThreadItem[],
  interruptedTurnIds: Set<string>,
): AgentThreadItem[] {
  return markInterruptedAgentMessageThreadItems(items, interruptedTurnIds);
}

export function createEmptyAgentSessionSnapshot(options?: {
  executionRuntime?: AgentSessionExecutionRuntime | null;
  workingDir?: string | null;
}): AgentSessionSnapshot {
  return {
    sessionId: null,
    workingDir: options?.workingDir?.trim() || null,
    messages: [],
    threadTurns: [],
    threadItems: [],
    currentTurnId: null,
    threadRead: null,
    executionRuntime: options?.executionRuntime ?? null,
    todoItems: [],
  };
}

function resolveCurrentTurnIdFromTimeline(params: {
  turns: AgentThreadTurn[];
  items: AgentThreadItem[];
  preferredTurns?: AgentThreadTurn[];
  preferredItems?: AgentThreadItem[];
}): string | null {
  for (
    let index = (params.preferredTurns?.length ?? 0) - 1;
    index >= 0;
    index -= 1
  ) {
    const turnId = params.preferredTurns?.[index]?.id;
    if (typeof turnId === "string" && turnId.trim().length > 0) {
      return turnId;
    }
  }
  for (
    let index = (params.preferredItems?.length ?? 0) - 1;
    index >= 0;
    index -= 1
  ) {
    const turnId = params.preferredItems?.[index]?.turn_id;
    if (typeof turnId === "string" && turnId.trim().length > 0) {
      return turnId;
    }
  }
  for (let index = params.items.length - 1; index >= 0; index -= 1) {
    const turnId = params.items[index]?.turn_id;
    if (typeof turnId === "string" && turnId.trim().length > 0) {
      return turnId;
    }
  }
  for (let index = params.turns.length - 1; index >= 0; index -= 1) {
    const turnId = params.turns[index]?.id;
    if (typeof turnId === "string" && turnId.trim().length > 0) {
      return turnId;
    }
  }
  return null;
}

export function hasSessionHydrationActivity(options: {
  currentTurnId: string | null;
  threadTurnsCount: number;
  threadItemsCount: number;
}) {
  return (
    options.currentTurnId !== null ||
    options.threadTurnsCount > 0 ||
    options.threadItemsCount > 0
  );
}

export function shouldPreserveActiveLocalSessionDuringBackgroundRestoreInitialization(options: {
  activeStreamingTimeline: boolean;
  messagesCount: number;
  sessionId: string | null | undefined;
  shouldRestoreSessionInForeground: boolean;
  threadItemsCount: number;
  threadTurnsCount: number;
}): boolean {
  if (options.shouldRestoreSessionInForeground) {
    return false;
  }

  if (!options.sessionId?.trim()) {
    return false;
  }

  return (
    options.messagesCount > 0 ||
    options.threadTurnsCount > 0 ||
    options.threadItemsCount > 0 ||
    options.activeStreamingTimeline
  );
}

export function hasActiveRuntimeTurn(options: {
  currentTurnId?: string | null;
  threadRead?: AgentRuntimeThreadReadModel | null;
  threadReadStatus?: string | null;
  turns: readonly AgentThreadTurn[];
}): boolean {
  const normalizedThreadReadStatus =
    options.threadReadStatus?.trim().toLowerCase() ?? null;
  const normalizedProfileStatus =
    options.threadRead?.profile_status?.trim().toLowerCase() ?? null;
  if (
    isExplicitTerminalRuntimeStatus(normalizedThreadReadStatus) ||
    isExplicitTerminalRuntimeStatus(normalizedProfileStatus)
  ) {
    return false;
  }
  if (options.threadRead) {
    const readModelReportsRunning =
      normalizedThreadReadStatus === "running" ||
      normalizedProfileStatus === "running";
    const readModelReportsQueued =
      normalizedThreadReadStatus === "queued" ||
      normalizedProfileStatus === "queued";
    const hasAuthoritativeReadModelStatus = Boolean(
      normalizedThreadReadStatus || normalizedProfileStatus,
    );
    if (
      hasAuthoritativeReadModelStatus &&
      !readModelReportsRunning &&
      !readModelReportsQueued
    ) {
      return false;
    }
    if (
      hasRunningThreadReadActivity(options.threadRead, {
        allowThreadStatusWithoutTurn: true,
      })
    ) {
      return true;
    }
    if (readModelReportsQueued) {
      return true;
    }
    if (readModelReportsRunning) {
      return Boolean(options.threadRead.active_turn_id?.trim());
    }
  } else if (
    normalizedThreadReadStatus === "running" ||
    normalizedThreadReadStatus === "queued"
  ) {
    return true;
  }

  if (options.turns.some((turn) => hasRunningTurnRecordActivity(turn))) {
    return true;
  }

  return false;
}

function isExplicitTerminalRuntimeStatus(status: string | null): boolean {
  return (
    status === "completed" ||
    status === "failed" ||
    status === "canceled" ||
    status === "cancelled" ||
    status === "aborted"
  );
}

export function shouldDeferSessionDetailHydration(options: {
  currentSessionId: string | null;
  topicId: string;
  forceRefresh?: boolean;
  resumeSessionStartHooks?: boolean;
  cachedSnapshot?: AgentSessionCachedSnapshot | null;
}) {
  const {
    currentSessionId,
    topicId,
    forceRefresh = false,
    resumeSessionStartHooks = false,
    cachedSnapshot,
  } = options;

  if (
    forceRefresh ||
    resumeSessionStartHooks ||
    !cachedSnapshot ||
    currentSessionId === topicId
  ) {
    return false;
  }

  return (
    cachedSnapshot.messages.length > 0 ||
    hasSessionHydrationActivity({
      currentTurnId: cachedSnapshot.currentTurnId,
      threadTurnsCount: cachedSnapshot.threadTurns.length,
      threadItemsCount: cachedSnapshot.threadItems.length,
    })
  );
}

export function shouldSkipAlreadyHydratedSession(options: {
  hydratedSessionId: string | null;
  sessionId: string;
  currentTurnId: string | null;
  messagesCount: number;
  threadReadStatus?: string | null;
  threadItemsCount: number;
  threadTurnsCount: number;
  selectedTopic?: Pick<Topic, "messagesCount" | "status"> | null;
}): boolean {
  if (options.hydratedSessionId !== options.sessionId) {
    return false;
  }

  const hasLocalSessionContent =
    options.messagesCount > 0 ||
    hasSessionHydrationActivity({
      currentTurnId: options.currentTurnId,
      threadItemsCount: options.threadItemsCount,
      threadTurnsCount: options.threadTurnsCount,
    });
  const normalizedThreadReadStatus = (options.threadReadStatus || "")
    .trim()
    .toLowerCase();
  const hasRuntimeReadModelActivity =
    normalizedThreadReadStatus === "queued" ||
    normalizedThreadReadStatus === "running" ||
    normalizedThreadReadStatus === "waiting_request";
  if (hasLocalSessionContent || hasRuntimeReadModelActivity) {
    return true;
  }

  const topicMessagesCount = options.selectedTopic?.messagesCount ?? 0;
  const topicStatus = options.selectedTopic?.status ?? "draft";
  const topicExpectsRemoteDetail =
    topicMessagesCount > 0 || topicStatus !== "draft";

  return !topicExpectsRemoteDetail;
}

export type MissingSessionFromTopicsAction =
  | {
      kind: "none";
    }
  | {
      kind: "skip_detached";
    }
  | {
      kind: "clear_auxiliary";
    }
  | {
      kind: "clear_inactive";
    }
  | {
      kind: "verify_remote";
    };

export function resolveMissingSessionFromTopicsAction(options: {
  currentTurnId: string | null;
  detachedSessionId: string | null | undefined;
  remoteConfirmed?: boolean;
  restoreCandidateMayLagTopics?: boolean;
  sessionId: string | null | undefined;
  threadItemsCount: number;
  threadTurnsCount: number;
  topicsCount: number;
  topicsReady: boolean;
  topicExists: boolean;
}): MissingSessionFromTopicsAction {
  const sessionId = options.sessionId?.trim();
  if (
    !options.topicsReady ||
    !sessionId ||
    options.topicsCount === 0 ||
    options.topicExists
  ) {
    return { kind: "none" };
  }

  if (options.detachedSessionId === sessionId) {
    return { kind: "skip_detached" };
  }

  if (isAuxiliaryAgentSessionId(sessionId)) {
    return { kind: "clear_auxiliary" };
  }

  const shouldVerifyMissingSession =
    hasSessionHydrationActivity({
      currentTurnId: options.currentTurnId,
      threadTurnsCount: options.threadTurnsCount,
      threadItemsCount: options.threadItemsCount,
    }) ||
    options.remoteConfirmed === true ||
    options.restoreCandidateMayLagTopics === true;

  return shouldVerifyMissingSession
    ? { kind: "verify_remote" }
    : { kind: "clear_inactive" };
}

export function resolveRestorableTopicSessionId(
  candidateSessionId: string | null | undefined,
  topics: Topic[],
  options?: {
    allowDetachedCandidate?: boolean;
  },
): string | null {
  const normalizedCandidate = candidateSessionId?.trim();
  if (topics.length === 0) {
    return normalizedCandidate ?? null;
  }

  if (
    normalizedCandidate &&
    options?.allowDetachedCandidate === true &&
    !topics.some((topic) => topic.id === normalizedCandidate)
  ) {
    return normalizedCandidate;
  }

  return resolveRestorableSessionId({
    candidateSessionId: normalizedCandidate,
    sessions: topics.map((topic) => ({
      id: topic.id,
      createdAt: Math.floor(topic.createdAt.getTime() / 1000),
      updatedAt: Math.floor(topic.updatedAt.getTime() / 1000),
    })),
  });
}

interface BuildHydratedAgentSessionSnapshotOptions {
  topicId: string;
  detail: Awaited<ReturnType<AgentRuntimeAdapter["getSession"]>>;
  currentSessionId: string | null;
  currentMessages: Message[];
  currentThreadTurns: AgentThreadTurn[];
  currentThreadItems: AgentThreadItem[];
  currentExecutionRuntime: AgentSessionExecutionRuntime | null;
  currentExecutionStrategy: AgentExecutionStrategy;
  topics: Topic[];
  localSnapshotOverride?: {
    sessionId: string;
    messages: Message[];
    threadTurns: AgentThreadTurn[];
    threadItems: AgentThreadItem[];
  } | null;
  syncSessionId?: boolean;
  executionStrategyOverride?: AgentExecutionStrategy;
  preserveExecutionStrategyOnMissingDetail?: boolean;
  detailMergeMode?: AgentSessionDetailMergeMode;
}

export function buildHydratedAgentSessionSnapshot(
  options: BuildHydratedAgentSessionSnapshotOptions,
): {
  executionStrategy: AgentExecutionStrategy;
  snapshot: AgentSessionSnapshot;
} {
  const {
    topicId,
    detail,
    currentSessionId,
    currentMessages,
    currentThreadTurns,
    currentThreadItems,
    currentExecutionRuntime,
    currentExecutionStrategy,
    topics,
    localSnapshotOverride,
    syncSessionId = false,
    executionStrategyOverride,
    preserveExecutionStrategyOnMissingDetail = false,
    detailMergeMode,
  } = options;
  const normalizedDetailMergeMode =
    normalizeAgentSessionDetailMergeMode(detailMergeMode);
  const effectiveCurrentSessionId =
    localSnapshotOverride?.sessionId ?? currentSessionId;
  const effectiveCurrentMessages = localSnapshotOverride
    ? normalizeHistoricalTopicSnapshotMessages(localSnapshotOverride.messages)
    : currentMessages;
  const effectiveCurrentThreadTurns =
    localSnapshotOverride?.threadTurns ?? currentThreadTurns;
  const effectiveCurrentThreadItems =
    localSnapshotOverride?.threadItems ?? currentThreadItems;
  const incomingTurns = detail.turns || [];
  const incomingItems = normalizeLegacyThreadItems(
    collectDetailThreadItems(detail),
  );
  const hasIncomingCanonicalUserMessage = incomingItems.some(
    (item) => item.type === "user_message",
  );
  const hasIncomingCanonicalAssistantOutput =
    resolveFinalAgentMessageItemIds(incomingItems).size > 0;
  const hasIncomingCanonicalConversationItems =
    hasIncomingCanonicalUserMessage || hasIncomingCanonicalAssistantOutput;
  const hasIncomingTerminalTimeline = hasTerminalDetailTimeline({
    thread_read: detail.thread_read,
    turns: incomingTurns,
  });
  const shouldReconcileTerminalRuntimeDetail =
    hasIncomingTerminalTimeline &&
    normalizedDetailMergeMode !== "history_hydrate";
  const mayPreserveExistingTimelineBySession =
    effectiveCurrentSessionId === topicId ||
    (effectiveCurrentSessionId === null &&
      syncSessionId &&
      (effectiveCurrentMessages.length > 0 ||
        effectiveCurrentThreadTurns.length > 0 ||
        effectiveCurrentThreadItems.length > 0));
  const hydratedMessagesForCurrentMode = hydrateSessionDetailMessages(
    detail,
    topicId,
    {
      compactCompletedHistory: shouldCompactCompletedSessionHistory(detail),
      includeTimelineFallback:
        hasIncomingCanonicalConversationItems ||
        shouldReconcileTerminalRuntimeDetail ||
        !mayPreserveExistingTimelineBySession ||
        effectiveCurrentMessages.length === 0,
      includeTimelineFallbackUsers:
        hasIncomingCanonicalConversationItems ||
        shouldReconcileTerminalRuntimeDetail,
    },
  );
  const hydratedMessagesForCompatibility =
    mayPreserveExistingTimelineBySession && effectiveCurrentMessages.length > 0
      ? hydrateSessionDetailMessages(detail, topicId, {
          compactCompletedHistory: shouldCompactCompletedSessionHistory(detail),
          includeTimelineFallback: true,
          includeTimelineFallbackUsers: shouldReconcileTerminalRuntimeDetail,
        })
      : hydratedMessagesForCurrentMode;
  const timelineMergeDecision = resolveAgentSessionTimelineMergeDecision({
    mode: normalizedDetailMergeMode,
    mayPreserveExistingTimelineBySession,
    hydratedMessagesForCompatibility,
    localMessages: effectiveCurrentMessages,
    threadRead: detail.thread_read,
    incomingTurns,
  });
  const hydratedMessages =
    timelineMergeDecision.shouldIgnoreIncompatibleHydratedMessages
      ? []
      : timelineMergeDecision.shouldPreserveBySession
        ? hydratedMessagesForCurrentMode
        : hydratedMessagesForCompatibility;
  const shouldPreserveExistingTimeline =
    timelineMergeDecision.shouldPreserveBySession ||
    shouldPreserveDetachedLocalSnapshot({
      hydratedMessages,
      localMessages: effectiveCurrentMessages,
      sessionId: effectiveCurrentSessionId,
    });
  const shouldPreserveExecutionRuntimeOnMissingDetail =
    shouldPreserveExistingTimeline;
  const nextExecutionRuntime = createExecutionRuntimeFromSessionDetail(detail);
  const selectedTopic = topics.find((topic) => topic.id === topicId);
  const nextExecutionStrategy =
    executionStrategyOverride ||
    detail.execution_strategy ||
    selectedTopic?.executionStrategy ||
    (preserveExecutionStrategyOnMissingDetail
      ? currentExecutionStrategy
      : null);
  const nextThreadTurns = shouldPreserveExistingTimeline
    ? mergeThreadTurns(effectiveCurrentThreadTurns, incomingTurns)
    : mergeThreadTurns(incomingTurns);
  const nextThreadItems = shouldPreserveExistingTimeline
    ? timelineMergeDecision.shouldPreserveByRuntimeSync
      ? mergeRuntimeSyncThreadItems(effectiveCurrentThreadItems, incomingItems)
      : filterConversationThreadItems(
          mergeThreadItems(effectiveCurrentThreadItems, incomingItems),
        )
    : filterConversationThreadItems(incomingItems);
  const visibleIncomingTurns = mergeThreadTurns(incomingTurns);
  const visibleIncomingItems = mergeThreadItems(incomingItems);
  const nextMessages =
    shouldPreserveExistingTimeline && hydratedMessages.length === 0
      ? effectiveCurrentMessages
      : shouldPreserveExistingTimeline
        ? mergeHydratedMessagesWithLocalState(
            effectiveCurrentMessages,
            hydratedMessages,
            {
              preferHydratedAssistantOutput:
                hasIncomingCanonicalAssistantOutput ||
                shouldReconcileTerminalRuntimeDetail,
            },
          )
        : hydratedMessages;
  const nextCurrentTurnId = resolveCurrentTurnIdFromTimeline({
    turns: nextThreadTurns,
    items: nextThreadItems,
    preferredTurns: visibleIncomingTurns,
    preferredItems: visibleIncomingItems,
  });
  const interruptedTurnIds = collectInterruptedTurnIds({
    items: incomingItems,
    threadRead: detail.thread_read,
    turns: incomingTurns,
  });
  for (const turnId of collectLocallyInterruptedTurnIds({
    candidateTurnIds: collectLocalInterruptedCandidateTurnIds({
      currentTurnId: nextCurrentTurnId,
      items: incomingItems,
      threadRead: detail.thread_read,
      turns: incomingTurns,
    }),
    sessionId: topicId,
    threadId: detail.thread_id ?? detail.thread_read?.thread_id ?? topicId,
  })) {
    interruptedTurnIds.add(turnId);
  }
  for (const turnId of collectLocalInterruptedMarkerTurnIds({
    currentMessages: effectiveCurrentMessages,
    fallbackTurnId: nextCurrentTurnId,
    incomingItems,
  })) {
    interruptedTurnIds.add(turnId);
  }
  const nextThreadItemsWithTerminalMarkers = markInterruptedThreadItems(
    nextThreadItems,
    interruptedTurnIds,
  );
  const nextMessagesWithTerminalMarkers = markInterruptedAssistantMessages(
    nextMessages,
    interruptedTurnIds,
    { fallbackTurnId: nextCurrentTurnId },
  );

  return {
    executionStrategy: normalizeExecutionStrategy(nextExecutionStrategy),
    snapshot: {
      sessionId: syncSessionId ? topicId : currentSessionId,
      workingDir: detail.working_dir?.trim() || null,
      messages: nextMessagesWithTerminalMarkers,
      threadTurns: nextThreadTurns,
      threadItems: nextThreadItemsWithTerminalMarkers,
      currentTurnId: nextCurrentTurnId,
      threadRead: detail.thread_read ?? null,
      executionRuntime:
        shouldPreserveExecutionRuntimeOnMissingDetail && !nextExecutionRuntime
          ? currentExecutionRuntime
          : nextExecutionRuntime,
      todoItems: detail.todo_items ?? [],
    },
  };
}
