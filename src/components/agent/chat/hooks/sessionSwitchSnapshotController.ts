import type { AgentThreadItem, AgentThreadTurn, Message } from "../types";
import type { AgentSessionCachedSnapshot } from "./agentSessionScopedStorage";

export interface SessionSwitchLocalSnapshotOverride {
  sessionId: string;
  messages: Message[];
  threadTurns: AgentThreadTurn[];
  threadItems: AgentThreadItem[];
}

export interface SessionSwitchStartStatePlan {
  currentSessionIdToPersist: string | null;
  detachedSessionId: string | null;
  shouldClearAutoRestoringSession: boolean;
  shouldResetSessionHydrating: boolean;
}

export interface SessionSwitchCachedSnapshotPlan {
  shouldApplyCachedSnapshot: boolean;
  shouldLoadCachedSnapshot: boolean;
  shouldRefreshCachedSnapshotImmediately: boolean;
}

export interface SessionSwitchDeferHydrationPlan {
  detailLoadMode: "direct" | "deferred";
  restoreCandidateSessionId: string;
  shouldApplyCachedTopicChromeState: boolean;
  shouldClearAutoRestoringSession: boolean;
  shouldResetSessionHydrating: boolean;
}

export interface SessionSwitchPendingShellPlan {
  restoreCandidateSessionId: string;
  sessionId: string;
  shouldApplyCachedTopicChromeState: boolean;
  shouldApplyEmptySessionSnapshot: boolean;
  shouldResetHistoryWindow: boolean;
  shouldSetSessionHydrating: boolean;
}

export function shouldReuseHydratedCurrentSession(params: {
  currentSessionId?: string | null;
  forceRefresh?: boolean;
  hydratedSessionId?: string | null;
  messagesCount: number;
  topicId: string;
}): boolean {
  return Boolean(
    params.forceRefresh !== true &&
      params.currentSessionId === params.topicId &&
      params.hydratedSessionId === params.topicId &&
      params.messagesCount > 0,
  );
}

export function shouldReusePendingCurrentSessionHydration(params: {
  currentSessionId?: string | null;
  forceRefresh?: boolean;
  pendingSessionId?: string | null;
  topicId: string;
}): boolean {
  return Boolean(
    params.forceRefresh !== true &&
      params.currentSessionId === params.topicId &&
      params.pendingSessionId === params.topicId,
  );
}

export function shouldLoadCachedTopicSnapshot(params: {
  currentSessionId?: string | null;
  topicId: string;
}): boolean {
  return params.currentSessionId !== params.topicId;
}

export function shouldApplyCachedTopicSnapshot(params: {
  currentSessionId?: string | null;
  topicId: string;
}): boolean {
  return shouldLoadCachedTopicSnapshot(params);
}

export function shouldRefreshCachedSnapshotImmediately(params: {
  cacheFreshness?: "fresh" | "stale" | null;
  topicStatus?: string | null;
}): boolean {
  return (
    params.cacheFreshness === "stale" ||
    params.topicStatus === "running" ||
    params.topicStatus === "waiting"
  );
}

export function shouldApplyPendingSessionShell(params: {
  currentSessionId?: string | null;
  topicId: string;
  cachedSnapshot?: AgentSessionCachedSnapshot | null;
}): boolean {
  return params.currentSessionId !== params.topicId && !params.cachedSnapshot;
}

export function shouldReuseActiveSessionSwitch(params: {
  activeTopicId?: string | null;
  allowDetachedSession?: boolean;
  forceRefresh?: boolean;
  restoreSource?: "auto" | string;
  resumeSessionStartHooks?: boolean;
  topicId: string;
}): boolean {
  if (
    params.forceRefresh ||
    params.resumeSessionStartHooks ||
    params.allowDetachedSession ||
    params.restoreSource === "auto"
  ) {
    return false;
  }

  return params.activeTopicId === params.topicId;
}

export function buildSessionSwitchStartStatePlan(params: {
  allowDetachedSession?: boolean;
  currentSessionId?: string | null;
  restoreSource?: "auto" | string;
  topicId: string;
}): SessionSwitchStartStatePlan {
  return {
    currentSessionIdToPersist: params.currentSessionId?.trim()
      ? params.currentSessionId
      : null,
    detachedSessionId:
      params.allowDetachedSession === true ? params.topicId : null,
    shouldClearAutoRestoringSession: params.restoreSource !== "auto",
    shouldResetSessionHydrating: true,
  };
}

export function buildSessionSwitchCachedSnapshotPlan(params: {
  cachedSnapshot?: AgentSessionCachedSnapshot | null;
  currentSessionId?: string | null;
  forceRefresh?: boolean;
  topicId: string;
  topicStatus?: string | null;
}): SessionSwitchCachedSnapshotPlan {
  const shouldLoad = Boolean(
    params.forceRefresh !== true &&
      shouldLoadCachedTopicSnapshot({
        currentSessionId: params.currentSessionId,
        topicId: params.topicId,
      }),
  );

  return {
    shouldApplyCachedSnapshot: shouldApplyCachedTopicSnapshot({
      currentSessionId: params.currentSessionId,
      topicId: params.topicId,
    }),
    shouldLoadCachedSnapshot: shouldLoad,
    shouldRefreshCachedSnapshotImmediately:
      shouldRefreshCachedSnapshotImmediately({
        cacheFreshness: params.cachedSnapshot?.cacheMetadata?.freshness,
        topicStatus: params.topicStatus,
      }),
  };
}

export function buildSessionSwitchDeferHydrationPlan(params: {
  refreshCachedSnapshotImmediately: boolean;
  topicId: string;
}): SessionSwitchDeferHydrationPlan {
  return {
    detailLoadMode: params.refreshCachedSnapshotImmediately
      ? "direct"
      : "deferred",
    restoreCandidateSessionId: params.topicId,
    shouldApplyCachedTopicChromeState: true,
    shouldClearAutoRestoringSession: true,
    shouldResetSessionHydrating: true,
  };
}

export function buildSessionSwitchPendingShellPlan(params: {
  topicId: string;
}): SessionSwitchPendingShellPlan {
  return {
    restoreCandidateSessionId: params.topicId,
    sessionId: params.topicId,
    shouldApplyCachedTopicChromeState: true,
    shouldApplyEmptySessionSnapshot: true,
    shouldResetHistoryWindow: true,
    shouldSetSessionHydrating: true,
  };
}

export function buildSessionSwitchStartMetricContext(params: {
  cachedSnapshot?: AgentSessionCachedSnapshot | null;
  currentSessionId?: string | null;
  messagesCount: number;
  refreshCachedSnapshotImmediately: boolean;
  snapshotIndexHadTarget?: boolean | null;
  snapshotIndexHit?: boolean;
  topicId: string;
  workspaceId?: string | null;
}): Record<string, unknown> {
  const cachedSnapshotMetadata = params.cachedSnapshot?.cacheMetadata;
  return {
    cacheFreshness: cachedSnapshotMetadata?.freshness ?? null,
    cacheStorageKind: cachedSnapshotMetadata?.storageKind ?? null,
    cachedLocalMessagesCount: params.cachedSnapshot?.messages.length ?? 0,
    currentSessionId: params.currentSessionId ?? null,
    messagesCount: params.messagesCount,
    refreshCachedSnapshotImmediately: params.refreshCachedSnapshotImmediately,
    sessionId: params.topicId,
    snapshotIndexHadTarget: params.snapshotIndexHadTarget ?? null,
    snapshotIndexHit: params.snapshotIndexHit === true,
    topicId: params.topicId,
    workspaceId: params.workspaceId,
  };
}

export function buildSessionSwitchDeferHydrationMetricContext(params: {
  cachedSnapshot?: AgentSessionCachedSnapshot | null;
  currentSessionId?: string | null;
  refreshImmediately: boolean;
  topicId: string;
  workspaceId?: string | null;
}): Record<string, unknown> {
  const cachedSnapshotMetadata = params.cachedSnapshot?.cacheMetadata;
  return {
    cacheFreshness: cachedSnapshotMetadata?.freshness ?? null,
    cacheStorageKind: cachedSnapshotMetadata?.storageKind ?? null,
    cachedLocalMessagesCount: params.cachedSnapshot?.messages.length ?? 0,
    currentSessionId: params.currentSessionId ?? null,
    refreshImmediately: params.refreshImmediately,
    sessionId: params.topicId,
    topicId: params.topicId,
    workspaceId: params.workspaceId,
  };
}

export function buildPendingSessionShellMetricContext(params: {
  currentSessionId?: string | null;
  topicId: string;
  workspaceId?: string | null;
}): Record<string, unknown> {
  return {
    currentSessionId: params.currentSessionId ?? null,
    sessionId: params.topicId,
    topicId: params.topicId,
    workspaceId: params.workspaceId,
  };
}

export function buildSessionSwitchLocalSnapshotOverride(params: {
  cachedSnapshot?: AgentSessionCachedSnapshot | null;
  currentSessionId?: string | null;
  messages: Message[];
  threadTurns: AgentThreadTurn[];
  threadItems: AgentThreadItem[];
  topicId: string;
}): SessionSwitchLocalSnapshotOverride | null {
  const cachedSnapshot = params.cachedSnapshot;
  if (
    !cachedSnapshot ||
    params.currentSessionId !== params.topicId ||
    params.messages !== cachedSnapshot.messages ||
    params.threadTurns !== cachedSnapshot.threadTurns ||
    params.threadItems !== cachedSnapshot.threadItems
  ) {
    return null;
  }

  return {
    sessionId: params.topicId,
    messages: cachedSnapshot.messages,
    threadTurns: cachedSnapshot.threadTurns,
    threadItems: cachedSnapshot.threadItems,
  };
}
