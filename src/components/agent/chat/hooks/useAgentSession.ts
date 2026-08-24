import {
  startTransition,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type MutableRefObject,
  type SetStateAction,
} from "react";
import { toast } from "sonner";
import type {
  AgentExecutionStrategy,
  AgentSessionExecutionRuntime,
} from "@/lib/api/agentExecutionRuntime";
import type {
  AgentRuntimeThreadReadModel,
  AgentTodoItem,
} from "@/lib/api/agentRuntime/sessionTypes";
import { logAgentDebug } from "@/lib/agentDebug";
import { recordAgentUiPerformanceMetric } from "@/lib/agentUiPerformanceMetrics";
import { normalizeLegacyThreadItems } from "@/lib/api/agentTextNormalization";
import { isAgentSessionNotFoundError } from "@/lib/agentSessionRecovery";
import type { AgentThreadItem, AgentThreadTurn, Message } from "../types";
import {
  mapSessionToTopic,
  type ClearMessagesOptions,
  type SessionModelPreference,
  type Topic,
} from "./agentChatShared";
import {
  loadStoredSessionWorkspaceIdRaw,
  savePersistedSessionWorkspaceId,
} from "./agentProjectStorage";
import { normalizeHistoryMessages } from "./agentChatHistory";
import {
  getAgentSessionCachedSnapshotAvailability,
  getAgentSessionScopedKeys,
  loadAgentSessionCachedSnapshot,
  saveAgentSessionCachedSnapshot,
} from "./agentSessionScopedStorage";
import {
  getExecutionStrategyStorageKey,
  loadPersisted,
  loadPersistedString,
  resolvePersistedAccessMode,
  resolvePersistedExecutionStrategy,
  resolveWorkspaceAgentPreferences,
  loadTransient,
  savePersisted,
  saveTransient,
} from "./agentChatStorage";
import { normalizeExecutionStrategy } from "./agentChatCoreUtils";
import type { AgentRuntimeAdapter } from "./agentRuntimeAdapter";
import { isAuxiliaryAgentSessionId } from "@/lib/api/agentRuntime/sessionIdentity";
import { filterConversationThreadItems } from "../utils/threadTimelineView";
import { shouldResumeTaskSession } from "../utils/taskCenterTabs";
import {
  createSessionAccessModeFromExecutionRuntime,
  createSessionModelPreferenceFromExecutionRuntime,
} from "../utils/sessionExecutionRuntime";
import {
  isLegacyDefaultProjectId,
  normalizeProjectId,
} from "../utils/topicProjectResolution";
import {
  buildHydratedAgentSessionSnapshot,
  createEmptyAgentSessionSnapshot,
  mergeCanonicalAgentMessagesIntoMessages,
  resolveMissingSessionFromTopicsAction,
  resolveRestorableTopicSessionId,
  shouldPreserveActiveLocalSessionDuringBackgroundRestoreInitialization,
  shouldSkipAlreadyHydratedSession,
  shouldDeferSessionDetailHydration,
  hasSessionHydrationActivity,
  type AgentSessionDetailMergeMode,
  type AgentSessionSnapshot,
} from "./agentSessionState";
import {
  buildPendingSessionShellMetricContext,
  buildSessionSwitchCachedSnapshotPlan,
  buildSessionSwitchDeferHydrationPlan,
  buildSessionSwitchDeferHydrationMetricContext,
  buildSessionSwitchLocalSnapshotOverride,
  buildSessionSwitchPendingShellPlan,
  buildSessionSwitchStartStatePlan,
  buildSessionSwitchStartMetricContext,
  shouldApplyPendingSessionShell,
  shouldReuseHydratedCurrentSession,
  shouldReusePendingCurrentSessionHydration,
  shouldReuseActiveSessionSwitch,
} from "./sessionSwitchSnapshotController";
import {
  applyFallbackExecutionStrategyToTopics,
  buildSessionFinalizeLocalStatePlan,
  buildSessionMetadataSyncInputPlan,
  buildSessionMetadataSyncPlan,
  buildSessionMetadataSyncSuccessApplyPlan,
} from "./sessionMetadataSyncController";
import { scheduleSessionMetadataSync } from "./sessionMetadataSyncScheduler";
import {
  buildSessionFinalizeSuccessStatePlan,
  buildSessionWorkspaceRestorePlan,
  normalizeSessionScopeWorkingDir,
  resolveSessionExecutionStrategyOverride,
  resolveShadowSessionExecutionStrategyFallback,
} from "./sessionFinalizeController";
import {
  applyRuntimeTopicWorkspaceIdToTopics,
  buildSessionPostFinalizePersistenceApplyPlan,
  buildSessionPostFinalizePersistencePlan,
} from "./sessionPostFinalizePersistenceController";
import {
  refreshAgentSessionDetailState,
  refreshAgentSessionReadModelState,
  hydrateFreshAgentSessionReadModel,
  mergeAgentSessionReadModelThreadItems,
  type AgentSessionDetailRefreshRequest,
} from "./agentSessionRefresh";
import { reuseStableAgentSessionSnapshotReferences } from "./agentSessionSnapshotStability";
import type { AgentAccessMode } from "./agentChatStorage";
import {
  hasRecoverableSilentTurnActivity,
  hasRecoverableTerminalTurnActivity,
} from "./agentSilentTurnRecovery";
import { shouldSkipStaleEmptyMessagesRefSync } from "./agentSessionTimelineMergePolicy";
import { scheduleMinimumDelayIdleTask } from "@/lib/utils/scheduleMinimumDelayIdleTask";
import { hasDesktopHostInvokeCapability } from "@/lib/desktop-runtime";
import { useTranslation } from "react-i18next";
import {
  buildSessionDetailHydrationOptions,
  isCurrentSessionHydrationRequest,
} from "./sessionHydrationController";
import {
  buildSessionHistoryPageRequestPlan,
  buildSessionHistoryPageResultPlan,
  resolveSessionHistoryWindowFromDetail,
  type SessionHistoryWindowState,
} from "./sessionHistoryPaginationController";
import { buildSessionHistoryMergePlan } from "./sessionHistoryMergeController";
import {
  loadSessionDetailWithPrefetch,
  type SessionDetailFetchEvent,
} from "./sessionDetailFetchController";
import { resolveDeferredSessionHydrationErrorAction } from "./sessionHydrationRetryController";
import { resolveSessionSwitchErrorAction } from "./sessionSwitchErrorController";
import {
  applyTopicExecutionStrategyToTopics,
  applyTopicSnapshotToTopics,
  mapSessionDetailToTopic,
  prependVerifiedSessionTopicFromDetail,
  resolveRestoreCandidateSanitizationPlan,
  selectActiveSessionTransientItems,
  selectActiveSessionTransientMessages,
  selectActiveSessionTransientTurns,
  type TopicSnapshotPatch,
  upsertFreshSessionDraftTopic,
  upsertTopicFromSessionDetail,
} from "./agentSessionTopicViewModel";
import {
  buildAgentSessionRestoreViewModel,
  buildCachedTopicSnapshotViewModel,
} from "./agentSessionRestoreViewModel";
import { hasActiveStreamingTimeline } from "./agentSessionStreamingGuards";
import { preserveAssistantMessageUsage } from "./agentMessageUsageMerge";
import { useAgentSessionTokenUsage } from "./useAgentSessionTokenUsage";

const INITIAL_TOPICS_IDLE_TIMEOUT_MS = 1_500;
const INITIAL_TOPICS_SESSION_REQUEST_LIMIT = 21;
const SESSION_HISTORY_LOAD_PAGE_SIZE = 50;
const ACTIVE_SESSION_TRANSIENT_SAVE_DELAY_MS = 180;
const ACTIVE_SESSION_TRANSIENT_SAVE_IDLE_TIMEOUT_MS = 1_800;
const SESSION_METADATA_SYNC_DELAY_MS = 8_000;
const SESSION_METADATA_SYNC_IDLE_TIMEOUT_MS = 15_000;
const FRESH_SESSION_POST_CREATE_PERSISTENCE_IDLE_TIMEOUT_MS = 1_000;
const SESSION_DETAIL_DEFERRED_HYDRATION_RETRY_DELAY_MS = 15_000;
const SESSION_DETAIL_DEFERRED_HYDRATION_MAX_RETRY = 1;

export type AgentSessionHistoryWindow = SessionHistoryWindowState;

function scheduleActiveSessionTransientSave(task: () => void): () => void {
  if (typeof window === "undefined") {
    task();
    return () => undefined;
  }

  let cancelled = false;
  let idleId: number | null = null;
  const delayId = window.setTimeout(() => {
    if (cancelled) {
      return;
    }

    if (typeof window.requestIdleCallback === "function") {
      idleId = window.requestIdleCallback(
        () => {
          if (!cancelled) {
            task();
          }
        },
        { timeout: ACTIVE_SESSION_TRANSIENT_SAVE_IDLE_TIMEOUT_MS },
      );
      return;
    }

    task();
  }, ACTIVE_SESSION_TRANSIENT_SAVE_DELAY_MS);

  return () => {
    cancelled = true;
    window.clearTimeout(delayId);
    if (idleId !== null && typeof window.cancelIdleCallback === "function") {
      window.cancelIdleCallback(idleId);
    }
  };
}

function scheduleFreshSessionPostCreatePersistence(task: () => void): void {
  scheduleMinimumDelayIdleTask(task, {
    idleTimeoutMs: FRESH_SESSION_POST_CREATE_PERSISTENCE_IDLE_TIMEOUT_MS,
  });
}

function buildFreshSessionProviderModelMetadata(
  providerType: string,
  model: string,
): Record<string, unknown> | undefined {
  const providerSelector = providerType.trim();
  const modelName = model.trim();
  if (!providerSelector || !modelName) {
    return undefined;
  }

  return {
    providerSelector,
    modelName,
    executionRuntime: {
      providerSelector,
      modelName,
    },
    extensionData: {
      "lime_provider_routing.v0": {
        providerSelector,
      },
    },
  };
}

function isSessionWorkspaceMismatchError(error: unknown): boolean {
  return (
    error instanceof Error &&
    error.message.startsWith("session workspace mismatch:")
  );
}

interface UseAgentSessionOptions {
  runtime: AgentRuntimeAdapter;
  workspaceId: string;
  workingDir?: string | null;
  disableSessionRestore: boolean;
  sessionRestorePresentation?: "foreground" | "background";
  initialTopicsLoadMode: "immediate" | "deferred";
  initialTopicsDeferredDelayMs?: number;
  preserveRestoredMessages: boolean;
  executionStrategy: AgentExecutionStrategy;
  accessMode: AgentAccessMode;
  providerTypeRef: MutableRefObject<string>;
  modelRef: MutableRefObject<string>;
  pendingSessionMetadataSyncCancelRef?: MutableRefObject<(() => void) | null>;
  sessionIdRef: MutableRefObject<string | null>;
  currentAssistantMsgIdRef: MutableRefObject<string | null>;
  currentStreamingSessionIdRef: MutableRefObject<string | null>;
  currentStreamingEventNameRef: MutableRefObject<string | null>;
  detachStreamBindingsRef: MutableRefObject<(() => void) | null>;
  resetPendingActions: () => void;
  persistSessionModelPreference: (
    sessionId: string,
    providerType: string,
    model: string,
  ) => void;
  loadSessionModelPreference: (
    sessionId: string,
  ) => SessionModelPreference | null;
  applySessionModelPreference: (
    sessionId: string,
    preference: SessionModelPreference,
    options?: { markSynced?: boolean },
  ) => void;
  markSessionModelPreferenceSynced: (
    sessionId: string,
    providerType: string,
    model: string,
  ) => void;
  markSessionExecutionStrategySynced: (
    sessionId: string,
    executionStrategy: AgentExecutionStrategy,
  ) => void;
  persistSessionAccessMode: (
    sessionId: string,
    accessMode: AgentAccessMode,
  ) => void;
  loadSessionAccessMode: (sessionId: string) => AgentAccessMode | null;
  filterSessionsByWorkspace: <T extends { id: string }>(sessions: T[]) => T[];
  setExecutionStrategyState: (
    executionStrategy: AgentExecutionStrategy,
  ) => void;
  setAccessModeState: (accessMode: AgentAccessMode) => void;
}

export function useAgentSession(options: UseAgentSessionOptions) {
  const {
    runtime,
    workspaceId,
    workingDir,
    disableSessionRestore,
    sessionRestorePresentation = "foreground",
    initialTopicsLoadMode,
    initialTopicsDeferredDelayMs,
    preserveRestoredMessages,
    executionStrategy,
    accessMode,
    providerTypeRef,
    modelRef,
    pendingSessionMetadataSyncCancelRef:
      sharedPendingSessionMetadataSyncCancelRef,
    sessionIdRef,
    currentAssistantMsgIdRef,
    currentStreamingSessionIdRef,
    currentStreamingEventNameRef,
    detachStreamBindingsRef,
    resetPendingActions,
    persistSessionModelPreference,
    loadSessionModelPreference,
    applySessionModelPreference,
    markSessionModelPreferenceSynced,
    markSessionExecutionStrategySynced,
    persistSessionAccessMode,
    loadSessionAccessMode,
    filterSessionsByWorkspace,
    setExecutionStrategyState,
    setAccessModeState,
  } = options;
  const { t: tNavigation } = useTranslation("navigation");
  const shouldRestoreSessionInForeground =
    !disableSessionRestore && sessionRestorePresentation !== "background";
  const shouldRecoverSessionInBackground =
    !disableSessionRestore && sessionRestorePresentation === "background";
  const scopedKeys = useMemo(
    () => getAgentSessionScopedKeys(workspaceId),
    [workspaceId],
  );
  const normalizedWorkingDir = useMemo(() => {
    const value = workingDir?.trim().replace(/[\\/]+$/u, "");
    return value || null;
  }, [workingDir]);
  const sanitizeRestoreCandidateSessionId = useCallback(
    (candidateSessionId: string | null | undefined): string | null => {
      const plan = resolveRestoreCandidateSanitizationPlan({
        candidateSessionId,
        mappedWorkspaceId: candidateSessionId?.trim()
          ? loadStoredSessionWorkspaceIdRaw(candidateSessionId.trim())
          : null,
        workspaceId,
      });
      if (plan.kind === "accept") {
        return plan.sessionId;
      }
      if (plan.kind === "skip_auxiliary") {
        logAgentDebug("useAgentSession", "restoreCandidate.skipAuxiliary", {
          candidateSessionId: plan.candidateSessionId,
          workspaceId: plan.workspaceId,
        });
        return null;
      }
      if (plan.kind === "reject_workspace") {
        logAgentDebug("useAgentSession", "restoreCandidate.rejected", {
          candidateSessionId: plan.candidateSessionId,
          mappedWorkspaceId: plan.mappedWorkspaceId,
          workspaceId: plan.workspaceId,
        });
      }
      return null;
    },
    [workspaceId],
  );

  const loadScopedSessionRestoreCandidate = useCallback(() => {
    if (disableSessionRestore) {
      return null;
    }

    const transientCandidate = loadTransient<string | null>(
      scopedKeys.currentSessionKey,
      null,
    );
    if (transientCandidate?.trim()) {
      return sanitizeRestoreCandidateSessionId(transientCandidate);
    }

    const resolvedWorkspaceId = workspaceId?.trim() || "";
    if (resolvedWorkspaceId) {
      const globalKeys = getAgentSessionScopedKeys("");
      const globalTransientCandidate = loadTransient<string | null>(
        globalKeys.currentSessionKey,
        null,
      );
      if (globalTransientCandidate?.trim()) {
        return sanitizeRestoreCandidateSessionId(globalTransientCandidate);
      }
    }

    const persistedCandidate = loadPersisted<string | null>(
      scopedKeys.persistedSessionKey,
      null,
    );
    if (persistedCandidate?.trim()) {
      return sanitizeRestoreCandidateSessionId(persistedCandidate);
    }

    if (resolvedWorkspaceId) {
      const globalKeys = getAgentSessionScopedKeys("");
      const globalPersistedCandidate = loadPersisted<string | null>(
        globalKeys.persistedSessionKey,
        null,
      );
      if (globalPersistedCandidate?.trim()) {
        return sanitizeRestoreCandidateSessionId(globalPersistedCandidate);
      }
    }

    return sanitizeRestoreCandidateSessionId(null);
  }, [
    disableSessionRestore,
    sanitizeRestoreCandidateSessionId,
    scopedKeys,
    workspaceId,
  ]);
  const loadPersistedSessionRestoreCandidate = useCallback(() => {
    if (disableSessionRestore) {
      return null;
    }

    return sanitizeRestoreCandidateSessionId(
      loadPersisted<string | null>(scopedKeys.persistedSessionKey, null),
    );
  }, [disableSessionRestore, sanitizeRestoreCandidateSessionId, scopedKeys]);

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>(() =>
    !shouldRestoreSessionInForeground
      ? []
      : loadTransient<Message[]>(scopedKeys.messagesKey, []),
  );
  const [threadTurns, setThreadTurns] = useState<AgentThreadTurn[]>(() =>
    !shouldRestoreSessionInForeground
      ? []
      : loadTransient<AgentThreadTurn[]>(scopedKeys.turnsKey, []),
  );
  const [threadItems, setThreadItems] = useState<AgentThreadItem[]>(() =>
    !shouldRestoreSessionInForeground
      ? []
      : filterConversationThreadItems(
          normalizeLegacyThreadItems(
            loadTransient<AgentThreadItem[]>(scopedKeys.itemsKey, []),
          ),
        ),
  );
  const [currentTurnId, setCurrentTurnId] = useState<string | null>(() =>
    !shouldRestoreSessionInForeground
      ? null
      : loadTransient<string | null>(scopedKeys.currentTurnKey, null),
  );
  const [threadRead, setThreadRead] =
    useState<AgentRuntimeThreadReadModel | null>(null);
  const [executionRuntime, setExecutionRuntime] =
    useState<AgentSessionExecutionRuntime | null>(null);
  const [sessionWorkingDir, setSessionWorkingDir] = useState<string | null>(
    null,
  );
  const [todoItems, setTodoItems] = useState<AgentTodoItem[]>([]);
  const [topics, setTopics] = useState<Topic[]>([]);
  const [topicsReady, setTopicsReady] = useState(false);
  const [sessionHistoryWindow, setSessionHistoryWindow] =
    useState<AgentSessionHistoryWindow | null>(null);
  const [isAutoRestoringSession, setIsAutoRestoringSession] = useState(
    () => !disableSessionRestore,
  );
  const [isSessionHydrating, setIsSessionHydrating] = useState(false);
  const [recoveredStreamBindingSessionId, setRecoveredStreamBindingSessionId] =
    useState<string | null>(null);

  const restoredWorkspaceRef = useRef<string | null>(null);
  const restoreInitializationScopeRef = useRef<string | null>(null);
  const hydratedSessionRef = useRef<string | null>(null);
  const skipAutoRestoreRef = useRef(false);
  const sessionSwitchRequestVersionRef = useRef(0);
  const activeSessionSwitchRef = useRef<{
    topicId: string;
    promise: Promise<"success" | "error">;
  } | null>(null);
  const deferredSessionHydrationCancelRef = useRef<(() => void) | null>(null);
  const pendingSessionDetailHydrationRef = useRef<{
    requestVersion: number;
    topicId: string;
  } | null>(null);
  const localPendingSessionMetadataSyncCancelRef = useRef<(() => void) | null>(
    null,
  );
  const pendingSessionMetadataSyncCancelRef =
    sharedPendingSessionMetadataSyncCancelRef ??
    localPendingSessionMetadataSyncCancelRef;
  const createFreshSessionPromiseRef = useRef<Promise<string | null> | null>(
    null,
  );
  const missingSessionVerificationRef = useRef<string | null>(null);
  const detachedSessionIdRef = useRef<string | null>(null);
  const topicsListMayBeTruncatedRef = useRef(false);
  const pendingTopicsRefreshRef = useRef(false);
  const activeStreamingTimelineRef = useRef(false);
  const sessionStateWorkspaceRef = useRef<string | null>(
    workspaceId?.trim() || "",
  );
  const messagesRef = useRef<Message[]>(messages);
  const assistantUsageByRuntimeTurnIdRef = useRef<
    Map<string, NonNullable<Message["usage"]>>
  >(new Map());
  const threadTurnsRef = useRef<AgentThreadTurn[]>(threadTurns);
  const threadItemsRef = useRef<AgentThreadItem[]>(threadItems);
  const threadReadRef = useRef<AgentRuntimeThreadReadModel | null>(threadRead);
  const canonicalThreadIdBySessionIdRef = useRef<Map<string, string>>(
    new Map(),
  );
  const sessionHistoryWindowRef = useRef<AgentSessionHistoryWindow | null>(
    sessionHistoryWindow,
  );
  const executionRuntimeRef = useRef<AgentSessionExecutionRuntime | null>(
    executionRuntime,
  );
  const sessionWorkingDirRef = useRef<string | null>(sessionWorkingDir);
  const appServerConfirmedSessionIdsRef = useRef<Set<string>>(new Set());
  const restoreCandidateSessionIdRef = useRef<string | null>(
    loadScopedSessionRestoreCandidate(),
  );
  const bootPersistedRestoreCandidateSessionIdRef = useRef<string | null>(
    loadPersistedSessionRestoreCandidate(),
  );
  const bootPersistedRestoreWorkspaceIdRef = useRef<string | null>(
    workspaceId?.trim() || "",
  );
  const currentRestoreWorkspaceId = workspaceId?.trim() || "";
  if (
    bootPersistedRestoreWorkspaceIdRef.current !== currentRestoreWorkspaceId
  ) {
    bootPersistedRestoreWorkspaceIdRef.current = currentRestoreWorkspaceId;
    bootPersistedRestoreCandidateSessionIdRef.current =
      loadPersistedSessionRestoreCandidate();
  }

  sessionIdRef.current = sessionId;

  useEffect(() => {
    return () => {
      pendingSessionMetadataSyncCancelRef.current?.();
      pendingSessionMetadataSyncCancelRef.current = null;
    };
  }, [pendingSessionMetadataSyncCancelRef]);

  const resetStreamingRefs = useCallback(() => {
    detachStreamBindingsRef.current?.();
    currentAssistantMsgIdRef.current = null;
    currentStreamingSessionIdRef.current = null;
    currentStreamingEventNameRef.current = null;
  }, [
    currentAssistantMsgIdRef,
    currentStreamingEventNameRef,
    currentStreamingSessionIdRef,
    detachStreamBindingsRef,
  ]);
  const setMessagesState = useCallback<Dispatch<SetStateAction<Message[]>>>(
    (value) => {
      const resolvedMessages =
        typeof value === "function"
          ? (value as (previous: Message[]) => Message[])(messagesRef.current)
          : value;
      const nextMessages = preserveAssistantMessageUsage(
        messagesRef.current,
        resolvedMessages,
        assistantUsageByRuntimeTurnIdRef.current,
      );
      messagesRef.current = nextMessages;
      setMessages(nextMessages);
    },
    [],
  );

  useAgentSessionTokenUsage({
    sessionId,
    sessionIdRef,
    messagesRef,
    threadReadRef,
    threadTurnsRef,
    usageByRuntimeTurnIdRef: assistantUsageByRuntimeTurnIdRef,
    setMessages: setMessagesState,
  });

  const setThreadTurnsState = useCallback<
    Dispatch<SetStateAction<AgentThreadTurn[]>>
  >((value) => {
    const nextThreadTurns =
      typeof value === "function"
        ? (value as (previous: AgentThreadTurn[]) => AgentThreadTurn[])(
            threadTurnsRef.current,
          )
        : value;
    threadTurnsRef.current = nextThreadTurns;
    setThreadTurns(nextThreadTurns);
  }, []);
  const setThreadItemsState = useCallback<
    Dispatch<SetStateAction<AgentThreadItem[]>>
  >((value) => {
    const nextThreadItems =
      typeof value === "function"
        ? (value as (previous: AgentThreadItem[]) => AgentThreadItem[])(
            threadItemsRef.current,
          )
        : value;
    threadItemsRef.current = nextThreadItems;
    setThreadItems(nextThreadItems);
  }, []);
  const getThreadItems = useCallback(() => threadItemsRef.current, []);

  const persistSessionRestoreCandidate = useCallback(
    (nextSessionId: string | null) => {
      const sanitizedSessionId =
        nextSessionId && isAuxiliaryAgentSessionId(nextSessionId)
          ? null
          : nextSessionId;
      restoreCandidateSessionIdRef.current = sanitizedSessionId;
      saveTransient(scopedKeys.currentSessionKey, sanitizedSessionId);
      savePersisted(scopedKeys.persistedSessionKey, sanitizedSessionId);
    },
    [scopedKeys],
  );

  const invalidatePendingSessionSwitches = useCallback(() => {
    deferredSessionHydrationCancelRef.current?.();
    deferredSessionHydrationCancelRef.current = null;
    pendingSessionDetailHydrationRef.current = null;
    pendingSessionMetadataSyncCancelRef.current?.();
    pendingSessionMetadataSyncCancelRef.current = null;
    sessionSwitchRequestVersionRef.current += 1;
    return sessionSwitchRequestVersionRef.current;
  }, [pendingSessionMetadataSyncCancelRef]);

  const listWorkspaceTopics = useCallback(async () => {
    const startedAt = Date.now();
    const resolvedWorkspaceId = workspaceId.trim();
    const sessionListScope = normalizedWorkingDir
      ? { cwd: normalizedWorkingDir }
      : resolvedWorkspaceId
        ? { workspaceId: resolvedWorkspaceId }
        : {};
    const sessions = await runtime.listSessions({
      ...sessionListScope,
      limit: INITIAL_TOPICS_SESSION_REQUEST_LIMIT,
    });
    const listDurationMs = Date.now() - startedAt;
    const workspaceFilterStartedAt = Date.now();
    const workspaceSessions =
      normalizedWorkingDir || !resolvedWorkspaceId
        ? sessions
        : filterSessionsByWorkspace(sessions);
    const workspaceFilterDurationMs = Date.now() - workspaceFilterStartedAt;
    const auxiliaryFilterStartedAt = Date.now();
    const visibleSessions = workspaceSessions.filter(
      (session) => !isAuxiliaryAgentSessionId(session.id),
    );
    const auxiliaryFilterDurationMs = Date.now() - auxiliaryFilterStartedAt;
    const topicMapStartedAt = Date.now();
    const topicList = visibleSessions.map(mapSessionToTopic);
    const topicMapDurationMs = Date.now() - topicMapStartedAt;
    const metricContext = {
      auxiliaryFilterDurationMs,
      hiddenAuxiliarySessionsCount:
        workspaceSessions.length - visibleSessions.length,
      limit: INITIAL_TOPICS_SESSION_REQUEST_LIMIT,
      listDurationMs,
      sessionsCount: sessions.length,
      topicMapDurationMs,
      topicsCount: topicList.length,
      totalDurationMs: Date.now() - startedAt,
      workspaceFilterDurationMs,
      workspaceId: resolvedWorkspaceId || null,
      workspaceSessionsCount: workspaceSessions.length,
    };
    recordAgentUiPerformanceMetric(
      "sidebar.recentConversations.loadBreakdown",
      metricContext,
    );
    logAgentDebug(
      "useAgentSession",
      "recentConversations.loadBreakdown",
      metricContext,
      {
        dedupeKey: `recentConversations.loadBreakdown:${workspaceId ?? "none"}`,
        throttleMs: 1000,
      },
    );

    return {
      sessions,
      workspaceSessions,
      visibleSessions,
      topicList,
    };
  }, [filterSessionsByWorkspace, normalizedWorkingDir, runtime, workspaceId]);

  const applySessionSnapshot = useCallback(
    (snapshot: AgentSessionSnapshot) => {
      const stableSnapshot = reuseStableAgentSessionSnapshotReferences(
        snapshot,
        {
          messages: messagesRef.current,
          threadItems: threadItemsRef.current,
          threadTurns: threadTurnsRef.current,
        },
      );
      const previousSessionId = sessionIdRef.current;
      const snapshotMessages = preserveAssistantMessageUsage(
        messagesRef.current,
        stableSnapshot.messages,
        assistantUsageByRuntimeTurnIdRef.current,
      );
      const preservesCanonicalThreadRead =
        !stableSnapshot.threadRead &&
        stableSnapshot.sessionId &&
        stableSnapshot.sessionId === previousSessionId &&
        threadReadRef.current;
      const resolvedThreadRead =
        stableSnapshot.threadRead ??
        (preservesCanonicalThreadRead ? threadReadRef.current : null);
      sessionIdRef.current = stableSnapshot.sessionId;
      messagesRef.current = snapshotMessages;
      threadTurnsRef.current = stableSnapshot.threadTurns;
      threadItemsRef.current = stableSnapshot.threadItems;
      threadReadRef.current = resolvedThreadRead;
      const resolvedSessionId = stableSnapshot.sessionId?.trim();
      const canonicalThreadId = resolvedThreadRead?.thread_id?.trim();
      if (resolvedSessionId && canonicalThreadId) {
        canonicalThreadIdBySessionIdRef.current.set(
          resolvedSessionId,
          canonicalThreadId,
        );
      }
      executionRuntimeRef.current = stableSnapshot.executionRuntime;
      sessionWorkingDirRef.current = stableSnapshot.workingDir;
      setSessionId(stableSnapshot.sessionId);
      setSessionWorkingDir(stableSnapshot.workingDir);
      setMessages(snapshotMessages);
      setThreadTurns(stableSnapshot.threadTurns);
      setThreadItems(stableSnapshot.threadItems);
      setCurrentTurnId(stableSnapshot.currentTurnId);
      setThreadRead(resolvedThreadRead);
      setExecutionRuntime(stableSnapshot.executionRuntime);
      setTodoItems(stableSnapshot.todoItems);
    },
    [sessionIdRef],
  );

  const applyReadModelSnapshot = useCallback(
    (snapshot: { threadRead: AgentRuntimeThreadReadModel | null }) => {
      if (!snapshot.threadRead) {
        return;
      }
      const nextThreadItems = mergeAgentSessionReadModelThreadItems(
        threadItemsRef.current,
        snapshot,
      );
      setThreadItemsState(nextThreadItems);
      setMessagesState((currentMessages) =>
        mergeCanonicalAgentMessagesIntoMessages(
          currentMessages,
          nextThreadItems,
        ),
      );
      threadReadRef.current = snapshot.threadRead;
      const currentSessionId = sessionIdRef.current?.trim();
      const canonicalThreadId = snapshot.threadRead.thread_id?.trim();
      if (currentSessionId && canonicalThreadId) {
        canonicalThreadIdBySessionIdRef.current.set(
          currentSessionId,
          canonicalThreadId,
        );
      }
      setThreadRead(snapshot.threadRead);
    },
    [setMessagesState, setThreadItemsState, sessionIdRef],
  );
  const getThreadIdForSubmit = useCallback(
    (targetSessionId?: string) => {
      const resolvedTargetSessionId = targetSessionId?.trim();
      if (resolvedTargetSessionId) {
        return canonicalThreadIdBySessionIdRef.current.get(
          resolvedTargetSessionId,
        );
      }
      return (
        canonicalThreadIdBySessionIdRef.current.get(
          sessionIdRef.current?.trim() ?? "",
        ) || threadReadRef.current?.thread_id
      );
    },
    [sessionIdRef],
  );

  const hasActiveStreamingTimelineNow = useCallback(
    () =>
      hasActiveStreamingTimeline({
        currentAssistantMsgId: currentAssistantMsgIdRef.current,
        currentStreamingEventName: currentStreamingEventNameRef.current,
        currentStreamingSessionId: currentStreamingSessionIdRef.current,
      }),
    [
      currentAssistantMsgIdRef,
      currentStreamingEventNameRef,
      currentStreamingSessionIdRef,
    ],
  );
  const activeStreamingTimeline = hasActiveStreamingTimelineNow();
  activeStreamingTimelineRef.current = activeStreamingTimeline;
  const hasActiveLocalSessionSnapshot = useCallback(
    () =>
      shouldPreserveActiveLocalSessionDuringBackgroundRestoreInitialization({
        activeStreamingTimeline: hasActiveStreamingTimelineNow(),
        messagesCount: messagesRef.current.length,
        sessionId: sessionIdRef.current,
        shouldRestoreSessionInForeground,
        threadItemsCount: threadItemsRef.current.length,
        threadTurnsCount: threadTurnsRef.current.length,
      }),
    [
      hasActiveStreamingTimelineNow,
      sessionIdRef,
      shouldRestoreSessionInForeground,
    ],
  );

  const deferTopicsLoadForActiveStream = useCallback(
    (source: "initial" | "manual") => {
      if (
        !activeStreamingTimelineRef.current &&
        !hasActiveStreamingTimelineNow()
      ) {
        return false;
      }

      activeStreamingTimelineRef.current = true;
      pendingTopicsRefreshRef.current = true;
      setTopicsReady(true);
      logAgentDebug(
        "useAgentSession",
        "loadTopics.deferredForActiveStream",
        {
          source,
          workspaceId,
        },
        {
          dedupeKey: `useAgentSession.loadTopics.deferredForActiveStream:${workspaceId ?? "none"}:${source}`,
          throttleMs: 1000,
        },
      );
      return true;
    },
    [hasActiveStreamingTimelineNow, workspaceId],
  );

  const resolveSessionHistoryWindow = useCallback(
    (
      detail: Awaited<ReturnType<AgentRuntimeAdapter["getSession"]>>,
    ): AgentSessionHistoryWindow | null => {
      return resolveSessionHistoryWindowFromDetail(detail);
    },
    [],
  );

  useEffect(() => {
    if (
      shouldSkipStaleEmptyMessagesRefSync({
        currentRefMessages: messagesRef.current,
        nextMessages: messages,
      })
    ) {
      logAgentDebug(
        "useAgentSession",
        "messagesRefSync.skipStaleEmpty",
        {
          currentMessagesCount: messagesRef.current.length,
          sessionId,
          workspaceId,
        },
        { throttleMs: 1000 },
      );
      return;
    }
    messagesRef.current = messages;
  }, [messages, sessionId, workspaceId]);

  useEffect(() => {
    sessionHistoryWindowRef.current = sessionHistoryWindow;
  }, [sessionHistoryWindow]);

  useEffect(() => {
    setMessages((prev) => {
      const normalized = normalizeHistoryMessages(prev);
      return normalized.length === prev.length ? prev : normalized;
    });
  }, [sessionId, workspaceId]);

  useEffect(() => {
    const resolvedWorkspaceId = workspaceId?.trim() || "";
    if (sessionStateWorkspaceRef.current !== resolvedWorkspaceId) {
      return;
    }

    const sessionRestoreCandidate =
      sessionId ?? restoreCandidateSessionIdRef.current;
    if (sessionRestoreCandidate) {
      persistSessionRestoreCandidate(sessionRestoreCandidate);
    }

    if (sessionId && resolvedWorkspaceId) {
      const sessionWorkspaceKey = `agent_session_workspace_${sessionId}`;
      const existingWorkspaceId = loadPersistedString(sessionWorkspaceKey);
      if (
        existingWorkspaceId &&
        existingWorkspaceId !== "__invalid__" &&
        !isLegacyDefaultProjectId(existingWorkspaceId) &&
        existingWorkspaceId !== resolvedWorkspaceId
      ) {
        console.warn("[AgentChat] 检测到会话与工作区映射冲突，跳过覆盖", {
          sessionId,
          existingWorkspaceId,
          currentWorkspaceId: resolvedWorkspaceId,
        });
      } else {
        savePersistedSessionWorkspaceId(sessionId, resolvedWorkspaceId);
      }
    }
  }, [persistSessionRestoreCandidate, sessionId, workspaceId]);

  useEffect(() => {
    const resolvedWorkspaceId = workspaceId?.trim() || "";
    if (sessionStateWorkspaceRef.current !== resolvedWorkspaceId) {
      return;
    }
    const transientMessages = selectActiveSessionTransientMessages(messages);
    return scheduleActiveSessionTransientSave(() => {
      saveTransient(scopedKeys.messagesKey, transientMessages);
    });
  }, [messages, scopedKeys, workspaceId]);

  useEffect(() => {
    const resolvedWorkspaceId = workspaceId?.trim() || "";
    if (sessionStateWorkspaceRef.current !== resolvedWorkspaceId) {
      return;
    }
    const transientTurns = selectActiveSessionTransientTurns(threadTurns);
    return scheduleActiveSessionTransientSave(() => {
      saveTransient(scopedKeys.turnsKey, transientTurns);
    });
  }, [scopedKeys, threadTurns, workspaceId]);

  useEffect(() => {
    threadTurnsRef.current = threadTurns;
  }, [threadTurns]);

  useEffect(() => {
    const resolvedWorkspaceId = workspaceId?.trim() || "";
    if (sessionStateWorkspaceRef.current !== resolvedWorkspaceId) {
      return;
    }
    const transientItems = selectActiveSessionTransientItems(
      threadItems,
      threadTurnsRef.current,
    );
    return scheduleActiveSessionTransientSave(() => {
      saveTransient(scopedKeys.itemsKey, transientItems);
    });
  }, [scopedKeys, threadItems, threadTurns, workspaceId]);

  useEffect(() => {
    threadItemsRef.current = threadItems;
  }, [threadItems]);

  useEffect(() => {
    executionRuntimeRef.current = executionRuntime;
  }, [executionRuntime]);

  useEffect(() => {
    sessionWorkingDirRef.current = sessionWorkingDir;
  }, [sessionWorkingDir]);

  useEffect(
    () => () => {
      deferredSessionHydrationCancelRef.current?.();
      deferredSessionHydrationCancelRef.current = null;
    },
    [],
  );

  useEffect(() => {
    const resolvedWorkspaceId = workspaceId?.trim() || "";
    if (sessionStateWorkspaceRef.current !== resolvedWorkspaceId) {
      return;
    }
    saveTransient(scopedKeys.currentTurnKey, currentTurnId);
  }, [currentTurnId, scopedKeys, workspaceId]);

  useEffect(() => {
    const resolvedWorkspaceId = workspaceId?.trim() || "";
    const resolvedSessionId = sessionId?.trim();
    if (
      !resolvedSessionId ||
      sessionStateWorkspaceRef.current !== resolvedWorkspaceId
    ) {
      return;
    }

    const activeTopic = topics.find((topic) => topic.id === resolvedSessionId);
    const topicMessagesCount = activeTopic?.messagesCount ?? messages.length;
    const transientMessages = selectActiveSessionTransientMessages(messages);
    const transientThreadTurns = selectActiveSessionTransientTurns(threadTurns);
    const transientTurnIds = new Set(
      transientThreadTurns
        .map((turn) => (typeof turn.id === "string" ? turn.id.trim() : ""))
        .filter(Boolean),
    );
    const transientCurrentTurnId =
      currentTurnId && transientTurnIds.has(currentTurnId)
        ? currentTurnId
        : null;
    const transientThreadItems = selectActiveSessionTransientItems(
      threadItems,
      transientThreadTurns,
    );
    return scheduleActiveSessionTransientSave(() => {
      saveAgentSessionCachedSnapshot(
        resolvedWorkspaceId,
        resolvedSessionId,
        {
          messages: transientMessages,
          threadTurns: transientThreadTurns,
          threadItems: transientThreadItems,
          currentTurnId: transientCurrentTurnId,
        },
        {
          sessionUpdatedAt: activeTopic?.updatedAt ?? Date.now(),
          messagesCount: topicMessagesCount,
          historyTruncated: sessionHistoryWindow?.hasMore === true,
        },
      );
    });
  }, [
    currentTurnId,
    messages,
    sessionId,
    sessionHistoryWindow,
    threadItems,
    threadTurns,
    topics,
    workspaceId,
  ]);

  useEffect(() => {
    const resolvedWorkspaceId = workspaceId.trim();
    const restoreInitializationScope = [
      disableSessionRestore ? "disabled" : "enabled",
      resolvedWorkspaceId,
      normalizedWorkingDir ?? "",
      shouldRestoreSessionInForeground ? "foreground" : "background",
    ].join(":");
    if (restoreInitializationScopeRef.current === restoreInitializationScope) {
      return;
    }
    restoreInitializationScopeRef.current = restoreInitializationScope;

    if (disableSessionRestore) {
      sessionStateWorkspaceRef.current = null;
      appServerConfirmedSessionIdsRef.current.clear();
      applySessionSnapshot(createEmptyAgentSessionSnapshot());
      setSessionHistoryWindow(null);
      setIsAutoRestoringSession(false);
      setIsSessionHydrating(false);
      setRecoveredStreamBindingSessionId(null);
      resetPendingActions();
      resetStreamingRefs();
      restoredWorkspaceRef.current = null;
      hydratedSessionRef.current = null;
      restoreCandidateSessionIdRef.current = null;
      skipAutoRestoreRef.current = disableSessionRestore;
      return;
    }

    sessionStateWorkspaceRef.current = resolvedWorkspaceId;
    if (!shouldRestoreSessionInForeground && hasActiveLocalSessionSnapshot()) {
      setIsAutoRestoringSession(false);
      setIsSessionHydrating(false);
      logAgentDebug(
        "useAgentSession",
        "restoreInitialization.skipActiveLocal",
        {
          messagesCount: messagesRef.current.length,
          sessionId: sessionIdRef.current,
          threadItemsCount: threadItemsRef.current.length,
          threadTurnsCount: threadTurnsRef.current.length,
          workspaceId: resolvedWorkspaceId,
        },
      );
      return;
    }
    appServerConfirmedSessionIdsRef.current.clear();
    const scopedSessionCandidate = loadScopedSessionRestoreCandidate();
    setIsAutoRestoringSession(Boolean(scopedSessionCandidate));
    setIsSessionHydrating(false);

    restoreCandidateSessionIdRef.current = scopedSessionCandidate;
    if (shouldRestoreSessionInForeground) {
      const scopedMessages = loadTransient<Message[]>(
        scopedKeys.messagesKey,
        [],
      );
      const scopedTurns = loadTransient<AgentThreadTurn[]>(
        scopedKeys.turnsKey,
        [],
      );
      const scopedItems = loadTransient<AgentThreadItem[]>(
        scopedKeys.itemsKey,
        [],
      );
      const scopedCurrentTurnId = loadTransient<string | null>(
        scopedKeys.currentTurnKey,
        null,
      );
      const cachedScopedSnapshot = scopedSessionCandidate
        ? loadAgentSessionCachedSnapshot(
            resolvedWorkspaceId,
            scopedSessionCandidate,
          )
        : null;
      const restoreViewModel = buildAgentSessionRestoreViewModel({
        cachedSnapshot: cachedScopedSnapshot,
        scopedCurrentTurnId,
        scopedItems,
        scopedMessages,
        scopedSessionCandidate,
        scopedTurns,
      });
      setRecoveredStreamBindingSessionId(restoreViewModel.sessionId);
      applySessionSnapshot({
        ...createEmptyAgentSessionSnapshot(),
        sessionId: restoreViewModel.sessionId,
        messages: restoreViewModel.messages,
        threadTurns: restoreViewModel.threadTurns,
        threadItems: restoreViewModel.threadItems,
        currentTurnId: restoreViewModel.currentTurnId,
      });
      setSessionHistoryWindow(restoreViewModel.historyWindow);
    } else {
      setRecoveredStreamBindingSessionId(null);
      applySessionSnapshot(createEmptyAgentSessionSnapshot());
      setSessionHistoryWindow(null);
    }
    resetPendingActions();
    resetStreamingRefs();
    restoredWorkspaceRef.current = null;
    hydratedSessionRef.current = null;
    skipAutoRestoreRef.current = false;
  }, [
    disableSessionRestore,
    loadScopedSessionRestoreCandidate,
    resetPendingActions,
    resetStreamingRefs,
    scopedKeys,
    normalizedWorkingDir,
    shouldRestoreSessionInForeground,
    workspaceId,
    applySessionSnapshot,
    hasActiveLocalSessionSnapshot,
    sessionIdRef,
  ]);

  useEffect(() => {
    let cancelled = false;

    const runListSessions = () => {
      if (deferTopicsLoadForActiveStream("initial")) {
        return;
      }

      setTopicsReady(false);
      const startedAt = Date.now();
      logAgentDebug("useAgentSession", "listSessions.start", {
        limit: INITIAL_TOPICS_SESSION_REQUEST_LIMIT,
        workspaceId,
      });
      listWorkspaceTopics()
        .then(({ sessions, workspaceSessions, visibleSessions, topicList }) => {
          if (cancelled) {
            return;
          }
          logAgentDebug("useAgentSession", "listSessions.success", {
            durationMs: Date.now() - startedAt,
            hiddenAuxiliarySessionsCount:
              workspaceSessions.length - visibleSessions.length,
            limit: INITIAL_TOPICS_SESSION_REQUEST_LIMIT,
            sessionsCount: sessions.length,
            topicsCount: topicList.length,
            workspaceId,
          });
          topicsListMayBeTruncatedRef.current =
            sessions.length >= INITIAL_TOPICS_SESSION_REQUEST_LIMIT;
          setTopics(topicList);
        })
        .catch((error) => {
          if (cancelled) {
            return;
          }
          topicsListMayBeTruncatedRef.current = false;
          console.error("[AgentChat] 加载话题失败:", error);
          logAgentDebug(
            "useAgentSession",
            "listSessions.error",
            {
              durationMs: Date.now() - startedAt,
              error,
              limit: INITIAL_TOPICS_SESSION_REQUEST_LIMIT,
              workspaceId,
            },
            { level: "error" },
          );
        })
        .finally(() => {
          if (!cancelled) {
            setTopicsReady(true);
          }
        });
    };

    if (initialTopicsLoadMode === "deferred") {
      setTopicsReady(true);
      const cancelDeferredLoad = scheduleMinimumDelayIdleTask(runListSessions, {
        minimumDelayMs: initialTopicsDeferredDelayMs,
        idleTimeoutMs: INITIAL_TOPICS_IDLE_TIMEOUT_MS,
      });
      return () => {
        cancelled = true;
        cancelDeferredLoad();
      };
    }

    runListSessions();

    return () => {
      cancelled = true;
    };
  }, [
    initialTopicsDeferredDelayMs,
    initialTopicsLoadMode,
    deferTopicsLoadForActiveStream,
    listWorkspaceTopics,
    workspaceId,
  ]);

  const loadTopics = useCallback(async () => {
    if (deferTopicsLoadForActiveStream("manual")) {
      return;
    }

    setTopicsReady(false);
    const startedAt = Date.now();
    logAgentDebug("useAgentSession", "loadTopics.start", {
      limit: INITIAL_TOPICS_SESSION_REQUEST_LIMIT,
      workspaceId,
    });
    try {
      const { sessions, workspaceSessions, visibleSessions, topicList } =
        await listWorkspaceTopics();
      logAgentDebug("useAgentSession", "loadTopics.success", {
        durationMs: Date.now() - startedAt,
        hiddenAuxiliarySessionsCount:
          workspaceSessions.length - visibleSessions.length,
        limit: INITIAL_TOPICS_SESSION_REQUEST_LIMIT,
        sessionsCount: sessions.length,
        topicsCount: topicList.length,
        workspaceId,
      });
      topicsListMayBeTruncatedRef.current =
        sessions.length >= INITIAL_TOPICS_SESSION_REQUEST_LIMIT;
      setTopics(topicList);
    } catch (error) {
      topicsListMayBeTruncatedRef.current = false;
      console.error("[AgentChat] 加载话题失败:", error);
      logAgentDebug(
        "useAgentSession",
        "loadTopics.error",
        {
          durationMs: Date.now() - startedAt,
          error,
          limit: INITIAL_TOPICS_SESSION_REQUEST_LIMIT,
          workspaceId,
        },
        { level: "error" },
      );
    } finally {
      setTopicsReady(true);
    }
  }, [deferTopicsLoadForActiveStream, listWorkspaceTopics, workspaceId]);

  useEffect(() => {
    if (activeStreamingTimeline || hasActiveStreamingTimelineNow()) {
      return;
    }
    if (!pendingTopicsRefreshRef.current) {
      return;
    }

    pendingTopicsRefreshRef.current = false;
    void loadTopics();
  }, [activeStreamingTimeline, hasActiveStreamingTimelineNow, loadTopics]);

  const createFreshSession = useCallback(
    async (
      sessionName?: string,
      createOptions?: {
        preserveCurrentSnapshot?: boolean;
        skipSessionStartHooks?: boolean;
      },
    ): Promise<string | null> => {
      if (createFreshSessionPromiseRef.current) {
        return createFreshSessionPromiseRef.current;
      }

      const resolvedWorkspaceId = workspaceId?.trim() || "";
      const sessionScopeId =
        normalizedWorkingDir ?? (resolvedWorkspaceId || "detached");

      const creationPromise = (async () => {
        const startedAt = Date.now();
        const creationExecutionStrategy =
          normalizeExecutionStrategy(executionStrategy);
        try {
          invalidatePendingSessionSwitches();
          skipAutoRestoreRef.current = true;
          logAgentDebug("useAgentSession", "createFreshSession.start", {
            executionStrategy: creationExecutionStrategy,
            sessionName: sessionName?.trim() || null,
            sessionScopeId,
            workspaceId: resolvedWorkspaceId || null,
            workingDir: normalizedWorkingDir,
          });
          const workspaceModelPreference = resolveWorkspaceAgentPreferences(
            resolvedWorkspaceId || workspaceId,
          );
          const nextProviderType =
            workspaceModelPreference.providerType || providerTypeRef.current;
          const nextModel = workspaceModelPreference.model || modelRef.current;
          const newSessionId = await runtime.createSession(
            resolvedWorkspaceId || undefined,
            sessionName,
            creationExecutionStrategy,
            {
              runStartHooks: createOptions?.skipSessionStartHooks !== true,
              workingDir: normalizedWorkingDir,
              metadata: buildFreshSessionProviderModelMetadata(
                nextProviderType,
                nextModel,
              ),
            },
          );
          const freshThreadRead = await hydrateFreshAgentSessionReadModel(
            runtime,
            newSessionId,
          );
          appServerConfirmedSessionIdsRef.current.add(newSessionId);

          const now = new Date();
          applySessionSnapshot({
            ...createEmptyAgentSessionSnapshot({
              workingDir: normalizedWorkingDir,
            }),
            sessionId: newSessionId,
            messages:
              createOptions?.preserveCurrentSnapshot === true
                ? messagesRef.current
                : [],
            threadTurns:
              createOptions?.preserveCurrentSnapshot === true
                ? threadTurnsRef.current
                : [],
            threadItems:
              createOptions?.preserveCurrentSnapshot === true
                ? threadItemsRef.current
                : [],
            threadRead: freshThreadRead,
          });
          setSessionHistoryWindow(null);
          setIsAutoRestoringSession(false);
          setIsSessionHydrating(false);
          setRecoveredStreamBindingSessionId(null);
          setTopics((prev) =>
            upsertFreshSessionDraftTopic(prev, {
              createdAt: now,
              executionStrategy: creationExecutionStrategy,
              sessionId: newSessionId,
              sessionName,
              workspaceId: resolvedWorkspaceId || null,
              workingDir: normalizedWorkingDir,
            }),
          );
          resetPendingActions();
          resetStreamingRefs();
          hydratedSessionRef.current = newSessionId;
          restoredWorkspaceRef.current = resolvedWorkspaceId || null;
          persistSessionRestoreCandidate(newSessionId);

          markSessionExecutionStrategySynced(
            newSessionId,
            creationExecutionStrategy,
          );
          if (nextProviderType.trim() && nextModel.trim()) {
            applySessionModelPreference(
              newSessionId,
              {
                providerType: nextProviderType,
                model: nextModel,
              },
              { markSynced: true },
            );
          }
          const nextScopedKeys = scopedKeys;
          scheduleFreshSessionPostCreatePersistence(() => {
            persistSessionModelPreference(
              newSessionId,
              nextProviderType,
              nextModel,
            );
            persistSessionAccessMode(newSessionId, accessMode);
            saveTransient(nextScopedKeys.messagesKey, []);
            saveTransient(nextScopedKeys.turnsKey, []);
            saveTransient(nextScopedKeys.itemsKey, []);
            saveTransient(nextScopedKeys.currentTurnKey, null);
          });

          logAgentDebug("useAgentSession", "createFreshSession.success", {
            durationMs: Date.now() - startedAt,
            newSessionId,
            sessionName: sessionName?.trim() || null,
            sessionScopeId,
            workspaceId: resolvedWorkspaceId || null,
            workingDir: normalizedWorkingDir,
          });
          return newSessionId;
        } catch (error) {
          skipAutoRestoreRef.current = false;
          console.error("[AgentChat] 创建新任务失败:", error);
          logAgentDebug(
            "useAgentSession",
            "createFreshSession.error",
            {
              durationMs: Date.now() - startedAt,
              error,
              sessionName: sessionName?.trim() || null,
              sessionScopeId,
              workspaceId: resolvedWorkspaceId || null,
              workingDir: normalizedWorkingDir,
            },
            { level: "error" },
          );
          throw error;
        }
      })();

      const trackedCreationPromise = creationPromise.finally(() => {
        if (createFreshSessionPromiseRef.current === trackedCreationPromise) {
          createFreshSessionPromiseRef.current = null;
        }
      });

      createFreshSessionPromiseRef.current = trackedCreationPromise;
      return trackedCreationPromise;
    },
    [
      accessMode,
      applySessionModelPreference,
      applySessionSnapshot,
      executionStrategy,
      invalidatePendingSessionSwitches,
      modelRef,
      markSessionExecutionStrategySynced,
      persistSessionModelPreference,
      persistSessionAccessMode,
      persistSessionRestoreCandidate,
      providerTypeRef,
      normalizedWorkingDir,
      resetPendingActions,
      resetStreamingRefs,
      runtime,
      scopedKeys,
      workspaceId,
    ],
  );

  const bindTargetSessionSnapshot = useCallback(
    (
      targetSessionId: string,
      options?: {
        detail?: Awaited<ReturnType<AgentRuntimeAdapter["getSession"]>>;
      },
    ) => {
      const resolvedSessionId = targetSessionId.trim();
      if (!resolvedSessionId) {
        return;
      }
      if (sessionIdRef.current?.trim() === resolvedSessionId) {
        return;
      }

      const detail = options?.detail;
      const runtimeWorkspaceId = normalizeProjectId(detail?.workspace_id);
      const resolvedWorkspaceId =
        runtimeWorkspaceId || normalizeProjectId(workspaceId);
      const runtimeWorkingDir = normalizeSessionScopeWorkingDir(
        detail?.working_dir,
      );
      const resolvedWorkingDir = runtimeWorkingDir || normalizedWorkingDir;

      invalidatePendingSessionSwitches();
      skipAutoRestoreRef.current = true;
      applySessionSnapshot({
        ...createEmptyAgentSessionSnapshot({
          workingDir: resolvedWorkingDir,
        }),
        sessionId: resolvedSessionId,
      });
      setSessionHistoryWindow(null);
      setIsAutoRestoringSession(false);
      setIsSessionHydrating(false);
      setRecoveredStreamBindingSessionId(null);
      resetPendingActions();
      resetStreamingRefs();
      restoredWorkspaceRef.current = resolvedWorkspaceId || null;
      persistSessionRestoreCandidate(resolvedSessionId);

      if (detail) {
        const canonicalThreadId =
          detail.thread_read?.thread_id?.trim() || detail.thread_id?.trim();
        if (canonicalThreadId) {
          canonicalThreadIdBySessionIdRef.current.set(
            resolvedSessionId,
            canonicalThreadId,
          );
        }
        setTopics((prev) =>
          upsertTopicFromSessionDetail(
            prev,
            mapSessionDetailToTopic(
              resolvedSessionId,
              detail,
              resolvedWorkspaceId,
            ),
          ),
        );
      }

      logAgentDebug("useAgentSession", "ensureSession.targetSessionBound", {
        sessionId: resolvedSessionId,
        workspaceId: resolvedWorkspaceId || null,
        workingDir: resolvedWorkingDir,
      });
    },
    [
      applySessionSnapshot,
      invalidatePendingSessionSwitches,
      normalizedWorkingDir,
      persistSessionRestoreCandidate,
      resetPendingActions,
      resetStreamingRefs,
      sessionIdRef,
      workspaceId,
    ],
  );

  const clearMessages = useCallback(
    (options: ClearMessagesOptions = {}) => {
      const { showToast = true, toastMessage = "新任务已创建" } = options;

      const scopedMessagesKey = scopedKeys.messagesKey;

      applySessionSnapshot(
        createEmptyAgentSessionSnapshot({
          executionRuntime: executionRuntimeRef.current,
        }),
      );
      setSessionHistoryWindow(null);
      invalidatePendingSessionSwitches();
      setIsAutoRestoringSession(false);
      setIsSessionHydrating(false);
      setRecoveredStreamBindingSessionId(null);
      resetPendingActions();
      restoredWorkspaceRef.current = null;
      hydratedSessionRef.current = null;
      skipAutoRestoreRef.current = true;
      resetStreamingRefs();

      persistSessionRestoreCandidate(null);
      saveTransient(scopedMessagesKey, []);
      saveTransient(scopedKeys.turnsKey, []);
      saveTransient(scopedKeys.itemsKey, []);
      saveTransient(scopedKeys.currentTurnKey, null);

      if (showToast) {
        toast.success(toastMessage);
      }
    },
    [
      applySessionSnapshot,
      invalidatePendingSessionSwitches,
      persistSessionRestoreCandidate,
      resetPendingActions,
      resetStreamingRefs,
      scopedKeys,
    ],
  );

  const deleteMessage = useCallback(
    (id: string) => {
      setMessagesState((prev) => prev.filter((msg) => msg.id !== id));
    },
    [setMessagesState],
  );

  const editMessage = useCallback(
    (id: string, newContent: string) => {
      setMessagesState((prev) =>
        prev.map((msg) =>
          msg.id === id ? { ...msg, content: newContent } : msg,
        ),
      );
    },
    [setMessagesState],
  );

  const applySessionDetail = useCallback(
    (
      topicId: string,
      detail: Awaited<ReturnType<AgentRuntimeAdapter["getSession"]>>,
      options?: {
        syncSessionId?: boolean;
        executionStrategyOverride?: AgentExecutionStrategy;
        preserveExecutionStrategyOnMissingDetail?: boolean;
        detailMergeMode?: AgentSessionDetailMergeMode;
        localSnapshotOverride?: {
          sessionId: string;
          messages: Message[];
          threadTurns: AgentThreadTurn[];
          threadItems: AgentThreadItem[];
        } | null;
      },
    ) => {
      appServerConfirmedSessionIdsRef.current.add(topicId);
      const { executionStrategy: nextExecutionStrategy, snapshot } =
        buildHydratedAgentSessionSnapshot({
          topicId,
          detail,
          currentSessionId: sessionIdRef.current,
          currentMessages: messagesRef.current,
          currentThreadTurns: threadTurnsRef.current,
          currentThreadItems: threadItemsRef.current,
          currentExecutionRuntime: executionRuntimeRef.current,
          currentExecutionStrategy: executionStrategy,
          topics,
          localSnapshotOverride: options?.localSnapshotOverride,
          syncSessionId: options?.syncSessionId,
          executionStrategyOverride: options?.executionStrategyOverride,
          detailMergeMode: options?.detailMergeMode,
          preserveExecutionStrategyOnMissingDetail:
            options?.preserveExecutionStrategyOnMissingDetail,
        });
      applySessionSnapshot(snapshot);
      setExecutionStrategyState(nextExecutionStrategy);
    },
    [
      applySessionSnapshot,
      executionStrategy,
      sessionIdRef,
      setExecutionStrategyState,
      topics,
    ],
  );

  const applyCachedTopicSnapshot = useCallback(
    (
      topicId: string,
      cachedSnapshot: ReturnType<typeof loadAgentSessionCachedSnapshot>,
    ) => {
      if (!cachedSnapshot) {
        return false;
      }

      const selectedTopic = topics.find((topic) => topic.id === topicId);
      const cachedTopicViewModel = buildCachedTopicSnapshotViewModel({
        cachedSnapshot,
        selectedTopic,
        topicId,
      });
      applySessionSnapshot({
        ...createEmptyAgentSessionSnapshot(),
        sessionId: cachedTopicViewModel.sessionId,
        messages: cachedTopicViewModel.messages,
        threadTurns: cachedTopicViewModel.threadTurns,
        threadItems: cachedTopicViewModel.threadItems,
        currentTurnId: cachedTopicViewModel.currentTurnId,
      });
      setExecutionStrategyState(
        normalizeExecutionStrategy(
          selectedTopic?.executionStrategy || executionStrategy,
        ),
      );
      setSessionHistoryWindow(cachedTopicViewModel.historyWindow);
      const cachedSnapshotMetricContext = {
        ...cachedTopicViewModel.metricContext,
        topicId,
        workspaceId,
      };
      recordAgentUiPerformanceMetric(
        "session.switch.cachedSnapshotApplied",
        cachedSnapshotMetricContext,
      );
      logAgentDebug(
        "useAgentSession",
        "switchTopic.cachedSnapshotApplied",
        cachedSnapshotMetricContext,
      );
      return true;
    },
    [
      applySessionSnapshot,
      executionStrategy,
      setExecutionStrategyState,
      topics,
      workspaceId,
    ],
  );

  const applyCachedTopicChromeState = useCallback(
    (topicId: string) => {
      const topicPreference = loadSessionModelPreference(topicId);
      if (topicPreference) {
        applySessionModelPreference(topicId, topicPreference, {
          markSynced: false,
        });
      }

      const shadowAccessMode = loadSessionAccessMode(topicId);
      if (shadowAccessMode) {
        setAccessModeState(shadowAccessMode);
      } else {
        setAccessModeState(resolvePersistedAccessMode(workspaceId));
      }
    },
    [
      applySessionModelPreference,
      loadSessionAccessMode,
      loadSessionModelPreference,
      setAccessModeState,
      workspaceId,
    ],
  );

  const emitSessionDetailFetchEvent = useCallback(
    (event: SessionDetailFetchEvent) => {
      if (event.metricName) {
        recordAgentUiPerformanceMetric(
          event.metricName,
          event.metricContext ?? event.logContext,
        );
      }

      const logOptions =
        event.logLevel || event.throttleMs
          ? {
              ...(event.logLevel ? { level: event.logLevel } : {}),
              ...(event.throttleMs ? { throttleMs: event.throttleMs } : {}),
            }
          : undefined;
      logAgentDebug(
        "useAgentSession",
        event.logEvent,
        event.logContext,
        logOptions,
      );
    },
    [],
  );

  const loadRuntimeSessionDetail = useCallback(
    (params: {
      topicId: string;
      startedAt: number;
      mode: "direct" | "deferred";
      resumeSessionStartHooks?: boolean;
    }) =>
      loadSessionDetailWithPrefetch({
        getSession: (topicId, options) => runtime.getSession(topicId, options),
        mode: params.mode,
        onEvent: emitSessionDetailFetchEvent,
        resumeSessionStartHooks: params.resumeSessionStartHooks,
        source: `switchTopic.${params.mode}`,
        startedAt: params.startedAt,
        topicId: params.topicId,
        workspaceId,
      }),
    [emitSessionDetailFetchEvent, runtime, workspaceId],
  );

  const finalizeResolvedTopicDetail = useCallback(
    (params: {
      topicId: string;
      detail: Awaited<ReturnType<AgentRuntimeAdapter["getSession"]>>;
      startedAt: number;
      localSnapshotOverride?: {
        sessionId: string;
        messages: Message[];
        threadTurns: AgentThreadTurn[];
        threadItems: AgentThreadItem[];
      } | null;
      switchRequestVersion: number;
      useTransition?: boolean;
    }) => {
      const {
        topicId,
        detail,
        startedAt,
        localSnapshotOverride = null,
        switchRequestVersion,
        useTransition = false,
      } = params;

      if (
        !isCurrentSessionHydrationRequest({
          currentRequestVersion: sessionSwitchRequestVersionRef.current,
          requestVersion: switchRequestVersion,
        })
      ) {
        logAgentDebug(
          "useAgentSession",
          "switchTopic.staleResultIgnored",
          {
            currentSessionId: sessionIdRef.current,
            switchRequestVersion,
            topicId,
            workspaceId,
          },
          { throttleMs: 1000 },
        );
        return false;
      }

      const runtimePreference =
        createSessionModelPreferenceFromExecutionRuntime(
          detail.execution_runtime,
        );
      const runtimeAccessMode = createSessionAccessModeFromExecutionRuntime(
        detail.execution_runtime,
      );
      const runtimeWorkspaceId = normalizeProjectId(detail.workspace_id);
      const selectedTopic = topics.find((topic) => topic.id === topicId);
      const topicWorkspaceId = normalizeProjectId(selectedTopic?.workspaceId);
      const shadowWorkspaceId = normalizeProjectId(
        loadStoredSessionWorkspaceIdRaw(topicId),
      );
      const resolvedWorkspaceId = normalizeProjectId(workspaceId);
      const workspaceRestorePlan = buildSessionWorkspaceRestorePlan({
        resolvedWorkingDir: normalizedWorkingDir,
        resolvedWorkspaceId,
        runtimeWorkingDir: detail.working_dir,
        runtimeWorkspaceId,
        shadowWorkspaceId,
        topicId,
        topicWorkingDir: selectedTopic?.workingDir,
        topicWorkspaceId,
      });

      if (workspaceRestorePlan.shouldReject) {
        console.warn(
          "[AgentChat] 检测到跨工作区会话恢复，已忽略",
          workspaceRestorePlan.crossWorkspaceContext ?? {
            currentWorkspaceId: resolvedWorkspaceId,
            knownWorkspaceId: workspaceRestorePlan.knownWorkspaceId,
            topicId,
          },
        );
        applySessionSnapshot(createEmptyAgentSessionSnapshot());
        setSessionHistoryWindow(null);
        persistSessionRestoreCandidate(null);
        hydratedSessionRef.current = null;
        restoredWorkspaceRef.current = null;
        skipAutoRestoreRef.current = false;
        setIsSessionHydrating(false);
        return false;
      }

      const runtimeExecutionStrategy = detail.execution_strategy
        ? normalizeExecutionStrategy(detail.execution_strategy)
        : null;
      const topicExecutionStrategy = selectedTopic?.executionStrategy
        ? normalizeExecutionStrategy(selectedTopic.executionStrategy)
        : null;
      const executionStrategyStorageKey =
        getExecutionStrategyStorageKey(workspaceId);
      const persistedExecutionStrategy =
        executionStrategyStorageKey &&
        loadPersistedString(executionStrategyStorageKey);
      const shadowExecutionStrategyFallback =
        resolveShadowSessionExecutionStrategyFallback({
          runtimeExecutionStrategy,
          topicExecutionStrategy,
          persistedExecutionStrategy: persistedExecutionStrategy
            ? resolvePersistedExecutionStrategy(workspaceId)
            : null,
        });
      const storedPreference = loadSessionModelPreference(topicId);
      const shadowAccessMode = loadSessionAccessMode(topicId);

      persistSessionRestoreCandidate(topicId);
      const applyFinalizeSuccessState = () => {
        const finalizeSuccessStatePlan = buildSessionFinalizeSuccessStatePlan();
        if (finalizeSuccessStatePlan.shouldClearAutoRestoringSession) {
          setIsAutoRestoringSession(false);
        }
        if (finalizeSuccessStatePlan.shouldResetSessionHydrating) {
          setIsSessionHydrating(false);
        }
      };
      const applyResolvedDetail = () => {
        applySessionDetail(topicId, detail, {
          localSnapshotOverride,
          syncSessionId: true,
          executionStrategyOverride: resolveSessionExecutionStrategyOverride({
            runtimeExecutionStrategy,
            topicExecutionStrategy,
            shadowExecutionStrategyFallback,
          }),
        });
        hydratedSessionRef.current = topicId;
        applyFinalizeSuccessState();
      };

      if (useTransition) {
        startTransition(applyResolvedDetail);
      } else {
        applyResolvedDetail();
      }
      setSessionHistoryWindow(resolveSessionHistoryWindow(detail));
      const workspaceDefaultAccessMode =
        resolvePersistedAccessMode(workspaceId);
      const metadataSyncInputPlan = buildSessionMetadataSyncInputPlan({
        runtimeAccessMode,
        runtimePreference,
        shadowAccessMode,
        shadowExecutionStrategyFallback,
        storedPreference,
        workspaceDefaultAccessMode,
      });
      const metadataSyncPlan = buildSessionMetadataSyncPlan(
        metadataSyncInputPlan,
      );
      const postFinalizePersistencePlan =
        buildSessionPostFinalizePersistencePlan({
          knownWorkspaceId: workspaceRestorePlan.knownWorkspaceId,
          providerPreferenceToApply: metadataSyncPlan.providerPreferenceToApply,
          resolvedWorkspaceId,
          runtimeWorkspaceId,
        });
      const postFinalizePersistenceApplyPlan =
        buildSessionPostFinalizePersistenceApplyPlan(
          postFinalizePersistencePlan,
        );
      setTopics((prev) =>
        upsertTopicFromSessionDetail(
          prev,
          mapSessionDetailToTopic(
            topicId,
            detail,
            postFinalizePersistencePlan.topicWorkspaceId,
          ),
        ),
      );

      const finalizeLocalStatePlan = buildSessionFinalizeLocalStatePlan({
        durationMs: Date.now() - startedAt,
        itemsCount: detail.items?.length ?? 0,
        messagesCount: detail.messages.length,
        metadataSyncPlan,
        runtimeExecutionStrategy,
        shadowExecutionStrategyFallback,
        topicExecutionStrategy,
        topicId,
        turnsCount: detail.turns?.length ?? 0,
        workspaceId,
      });

      if (finalizeLocalStatePlan.runtimeExecutionStrategyToMarkSynced) {
        markSessionExecutionStrategySynced(
          topicId,
          finalizeLocalStatePlan.runtimeExecutionStrategyToMarkSynced,
        );
      }

      setAccessModeState(finalizeLocalStatePlan.accessModeToApply);
      if (finalizeLocalStatePlan.accessModeToPersist) {
        persistSessionAccessMode(
          topicId,
          finalizeLocalStatePlan.accessModeToPersist,
        );
      }

      recordAgentUiPerformanceMetric(
        "session.switch.success",
        finalizeLocalStatePlan.switchSuccessMetricContext,
      );
      logAgentDebug(
        "useAgentSession",
        "switchTopic.success",
        finalizeLocalStatePlan.switchSuccessMetricContext,
      );

      if (postFinalizePersistenceApplyPlan.sessionWorkspaceIdToPersist) {
        savePersistedSessionWorkspaceId(
          topicId,
          postFinalizePersistenceApplyPlan.sessionWorkspaceIdToPersist,
        );
      }

      if (postFinalizePersistenceApplyPlan.runtimeTopicWorkspaceIdToApply) {
        setTopics((prev) =>
          applyRuntimeTopicWorkspaceIdToTopics(prev, {
            runtimeTopicWorkspaceIdToApply:
              postFinalizePersistenceApplyPlan.runtimeTopicWorkspaceIdToApply,
            topicId,
          }),
        );
      }

      if (postFinalizePersistenceApplyPlan.providerPreferenceToApply) {
        applySessionModelPreference(
          topicId,
          postFinalizePersistenceApplyPlan.providerPreferenceToApply,
          {
            markSynced:
              metadataSyncPlan.modelPreferenceSource === "execution_runtime",
          },
        );
      }

      if (metadataSyncPlan.hasPatch) {
        scheduleSessionMetadataSync({
          getCurrentRequestVersion: () =>
            sessionSwitchRequestVersionRef.current,
          getCurrentSessionId: () => sessionIdRef.current,
          hasRuntimeInvokeCapability: hasDesktopHostInvokeCapability(),
          idleTimeoutMs: SESSION_METADATA_SYNC_IDLE_TIMEOUT_MS,
          minimumDelayMs: SESSION_METADATA_SYNC_DELAY_MS,
          onError: (error) => {
            console.warn("[AgentChat] 迁移会话 metadata fallback 失败:", error);
          },
          onSkipped: (event) => {
            logAgentDebug(
              "useAgentSession",
              event.logEvent,
              event.logContext,
              event.logOptions,
            );
          },
          onSynced: (syncedPlan) => {
            const syncSuccessApplyPlan =
              buildSessionMetadataSyncSuccessApplyPlan(syncedPlan);
            if (syncSuccessApplyPlan.providerPreferenceToMarkSynced) {
              markSessionModelPreferenceSynced(
                topicId,
                syncSuccessApplyPlan.providerPreferenceToMarkSynced
                  .providerType,
                syncSuccessApplyPlan.providerPreferenceToMarkSynced.model,
              );
            }
            if (syncSuccessApplyPlan.executionStrategyToMarkSynced) {
              const fallbackExecutionStrategy =
                syncSuccessApplyPlan.executionStrategyToMarkSynced;
              markSessionExecutionStrategySynced(
                topicId,
                fallbackExecutionStrategy,
              );
            }
            if (syncSuccessApplyPlan.executionStrategyToApplyToTopic) {
              setTopics((prev) =>
                applyFallbackExecutionStrategyToTopics(prev, {
                  executionStrategyToApplyToTopic:
                    syncSuccessApplyPlan.executionStrategyToApplyToTopic,
                  topicId,
                }),
              );
            }
          },
          pendingCancel: pendingSessionMetadataSyncCancelRef.current,
          plan: metadataSyncPlan,
          runtime,
          scheduler: { schedule: scheduleMinimumDelayIdleTask },
          sessionId: topicId,
          setPendingCancel: (cancel) => {
            pendingSessionMetadataSyncCancelRef.current = cancel;
          },
          switchRequestVersion,
          workspaceId,
        });
      }

      return true;
    },
    [
      applySessionDetail,
      applySessionSnapshot,
      applySessionModelPreference,
      loadSessionAccessMode,
      loadSessionModelPreference,
      markSessionExecutionStrategySynced,
      markSessionModelPreferenceSynced,
      normalizedWorkingDir,
      pendingSessionMetadataSyncCancelRef,
      persistSessionAccessMode,
      persistSessionRestoreCandidate,
      resolveSessionHistoryWindow,
      runtime,
      sessionIdRef,
      setAccessModeState,
      topics,
      workspaceId,
    ],
  );

  const handleSwitchTopicError = useCallback(
    (
      error: unknown,
      topicId: string,
      options?: { preserveCurrentSnapshot?: boolean },
    ) => {
      const errorAction = resolveSessionSwitchErrorAction({
        error,
        preserveCurrentSnapshot: options?.preserveCurrentSnapshot,
        topicId,
        workspaceId,
      });

      console.error("[AgentChat] 切换话题失败:", error);
      console.error("[AgentChat] 错误详情:", JSON.stringify(error, null, 2));
      logAgentDebug(
        "useAgentSession",
        "switchTopic.error",
        errorAction.logContext,
        { level: "error" },
      );

      if (errorAction.clearCurrentSnapshot) {
        applySessionSnapshot(createEmptyAgentSessionSnapshot());
        setSessionHistoryWindow(null);
        persistSessionRestoreCandidate(null);
        hydratedSessionRef.current = null;
      }

      if (errorAction.reloadTopics) {
        void loadTopics();
      }

      setIsAutoRestoringSession(false);
      setIsSessionHydrating(false);
      if (errorAction.showToast && errorAction.toastMessage) {
        toast.error(errorAction.toastMessage);
      }
    },
    [
      applySessionSnapshot,
      loadTopics,
      persistSessionRestoreCandidate,
      workspaceId,
    ],
  );

  const switchTopic = useCallback(
    async (
      topicId: string,
      options?: {
        forceRefresh?: boolean;
        resumeSessionStartHooks?: boolean;
        allowDetachedSession?: boolean;
        restoreSource?: "auto";
      },
    ) => {
      if (
        shouldReuseHydratedCurrentSession({
          currentSessionId: sessionIdRef.current,
          forceRefresh: options?.forceRefresh,
          hydratedSessionId: hydratedSessionRef.current,
          messagesCount: messages.length,
          topicId,
        })
      ) {
        return;
      }
      if (
        shouldReusePendingCurrentSessionHydration({
          currentSessionId: sessionIdRef.current,
          forceRefresh: options?.forceRefresh,
          pendingSessionId:
            pendingSessionDetailHydrationRef.current?.topicId ?? null,
          topicId,
        })
      ) {
        return;
      }

      const activeSwitch = activeSessionSwitchRef.current;
      const canReuseActiveSwitch = shouldReuseActiveSessionSwitch({
        activeTopicId: activeSwitch?.topicId,
        allowDetachedSession: options?.allowDetachedSession,
        forceRefresh: options?.forceRefresh,
        restoreSource: options?.restoreSource,
        resumeSessionStartHooks: options?.resumeSessionStartHooks,
        topicId,
      });
      if (canReuseActiveSwitch && activeSwitch) {
        logAgentDebug("useAgentSession", "switchTopic.reuseInFlight", {
          topicId,
          workspaceId,
        });
        return activeSwitch.promise;
      }

      let resolveActiveSwitch: (result: "success" | "error") => void = () => {};
      const activeSwitchPromise = new Promise<"success" | "error">(
        (resolve) => {
          resolveActiveSwitch = resolve;
        },
      );
      activeSessionSwitchRef.current = {
        topicId,
        promise: activeSwitchPromise,
      };

      const currentSessionId = sessionIdRef.current;
      const switchStartStatePlan = buildSessionSwitchStartStatePlan({
        allowDetachedSession: options?.allowDetachedSession,
        currentSessionId,
        restoreSource: options?.restoreSource,
        topicId,
      });
      if (switchStartStatePlan.currentSessionIdToPersist) {
        persistSessionModelPreference(
          switchStartStatePlan.currentSessionIdToPersist,
          providerTypeRef.current,
          modelRef.current,
        );
      }

      skipAutoRestoreRef.current = false;
      detachedSessionIdRef.current = switchStartStatePlan.detachedSessionId;
      if (options?.restoreSource !== "auto") {
        setRecoveredStreamBindingSessionId(null);
      } else {
        setRecoveredStreamBindingSessionId(topicId);
      }
      if (switchStartStatePlan.shouldResetSessionHydrating) {
        setIsSessionHydrating(false);
      }
      const switchRequestVersion = invalidatePendingSessionSwitches();
      if (switchStartStatePlan.shouldClearAutoRestoringSession) {
        setIsAutoRestoringSession(false);
      }
      try {
        const startedAt = Date.now();
        const selectedTopic = topics.find((topic) => topic.id === topicId);
        const cachedSnapshotLoadPlan = buildSessionSwitchCachedSnapshotPlan({
          currentSessionId,
          forceRefresh: options?.forceRefresh,
          topicId,
          topicStatus: selectedTopic?.status,
        });
        const cachedSnapshotAvailability =
          cachedSnapshotLoadPlan.shouldLoadCachedSnapshot
            ? getAgentSessionCachedSnapshotAvailability(workspaceId, topicId)
            : null;
        const cachedTargetSnapshot =
          !cachedSnapshotLoadPlan.shouldLoadCachedSnapshot ||
          cachedSnapshotAvailability?.hasSnapshot === false
            ? null
            : loadAgentSessionCachedSnapshot(workspaceId, topicId, {
                topicUpdatedAt: selectedTopic?.updatedAt ?? null,
                messagesCount: selectedTopic?.messagesCount ?? null,
              });
        const cachedSnapshotPlan = buildSessionSwitchCachedSnapshotPlan({
          cachedSnapshot: cachedTargetSnapshot,
          currentSessionId,
          forceRefresh: options?.forceRefresh,
          topicId,
          topicStatus: selectedTopic?.status,
        });
        const switchStartMetricContext = buildSessionSwitchStartMetricContext({
          cachedSnapshot: cachedTargetSnapshot,
          currentSessionId,
          messagesCount: messages.length,
          snapshotIndexHit: cachedSnapshotAvailability?.hasIndex ?? false,
          snapshotIndexHadTarget:
            cachedSnapshotAvailability?.hasSnapshot ?? null,
          refreshCachedSnapshotImmediately:
            cachedSnapshotPlan.shouldRefreshCachedSnapshotImmediately,
          topicId,
          workspaceId,
        });
        recordAgentUiPerformanceMetric(
          "session.switch.start",
          switchStartMetricContext,
        );
        logAgentDebug(
          "useAgentSession",
          "switchTopic.start",
          switchStartMetricContext,
        );
        if (cachedSnapshotPlan.shouldApplyCachedSnapshot) {
          applyCachedTopicSnapshot(topicId, cachedTargetSnapshot);
        }
        const shouldDeferDetailHydration = shouldDeferSessionDetailHydration({
          currentSessionId,
          topicId,
          forceRefresh: options?.forceRefresh,
          resumeSessionStartHooks: options?.resumeSessionStartHooks,
          cachedSnapshot: cachedTargetSnapshot,
        });

        if (shouldDeferDetailHydration) {
          const deferHydrationPlan = buildSessionSwitchDeferHydrationPlan({
            refreshCachedSnapshotImmediately:
              cachedSnapshotPlan.shouldRefreshCachedSnapshotImmediately,
            topicId,
          });
          if (deferHydrationPlan.shouldApplyCachedTopicChromeState) {
            applyCachedTopicChromeState(topicId);
          }
          persistSessionRestoreCandidate(
            deferHydrationPlan.restoreCandidateSessionId,
          );
          if (deferHydrationPlan.shouldClearAutoRestoringSession) {
            setIsAutoRestoringSession(false);
          }
          if (deferHydrationPlan.shouldResetSessionHydrating) {
            setIsSessionHydrating(false);
          }
          const deferHydrationMetricContext =
            buildSessionSwitchDeferHydrationMetricContext({
              cachedSnapshot: cachedTargetSnapshot,
              currentSessionId,
              refreshImmediately:
                cachedSnapshotPlan.shouldRefreshCachedSnapshotImmediately,
              topicId,
              workspaceId,
            });
          recordAgentUiPerformanceMetric(
            "session.switch.deferHydration",
            deferHydrationMetricContext,
          );
          logAgentDebug(
            "useAgentSession",
            "switchTopic.deferHydration",
            deferHydrationMetricContext,
          );
          let deferredHydrationRetryCount = 0;
          const pendingHydration = {
            requestVersion: switchRequestVersion,
            topicId,
          };
          pendingSessionDetailHydrationRef.current = pendingHydration;
          const clearPendingHydration = () => {
            if (pendingSessionDetailHydrationRef.current === pendingHydration) {
              pendingSessionDetailHydrationRef.current = null;
            }
          };
          const hydrateCachedTopic = () => {
            deferredSessionHydrationCancelRef.current = null;
            void (async () => {
              let keepPendingHydration = false;
              try {
                const detail = await loadRuntimeSessionDetail({
                  topicId,
                  startedAt,
                  mode: deferHydrationPlan.detailLoadMode,
                });
                finalizeResolvedTopicDetail({
                  topicId,
                  detail,
                  startedAt,
                  localSnapshotOverride:
                    buildSessionSwitchLocalSnapshotOverride({
                      cachedSnapshot: cachedTargetSnapshot,
                      currentSessionId: sessionIdRef.current,
                      messages: messagesRef.current,
                      threadTurns: threadTurnsRef.current,
                      threadItems: threadItemsRef.current,
                      topicId,
                    }),
                  switchRequestVersion,
                  useTransition: true,
                });
              } catch (error) {
                if (
                  !isCurrentSessionHydrationRequest({
                    currentRequestVersion:
                      sessionSwitchRequestVersionRef.current,
                    requestVersion: switchRequestVersion,
                  })
                ) {
                  return;
                }
                const retryAction = resolveDeferredSessionHydrationErrorAction({
                  error,
                  retryCount: deferredHydrationRetryCount,
                  maxRetry: SESSION_DETAIL_DEFERRED_HYDRATION_MAX_RETRY,
                  retryDelayMs:
                    SESSION_DETAIL_DEFERRED_HYDRATION_RETRY_DELAY_MS,
                  topicId,
                  workspaceId,
                });
                if (retryAction.kind === "retry") {
                  keepPendingHydration = true;
                  deferredHydrationRetryCount = retryAction.nextRetryCount;
                  recordAgentUiPerformanceMetric(
                    retryAction.metricName,
                    retryAction.logContext,
                  );
                  logAgentDebug(
                    "useAgentSession",
                    retryAction.logEvent,
                    retryAction.logContext,
                    { level: "warn", throttleMs: 1000 },
                  );
                  deferredSessionHydrationCancelRef.current =
                    scheduleMinimumDelayIdleTask(hydrateCachedTopic, {
                      minimumDelayMs: retryAction.retryDelayMs,
                      idleTimeoutMs: retryAction.retryDelayMs,
                    });
                  return;
                }
                if (retryAction.kind === "skip") {
                  recordAgentUiPerformanceMetric(
                    retryAction.metricName,
                    retryAction.logContext,
                  );
                  logAgentDebug(
                    "useAgentSession",
                    retryAction.logEvent,
                    retryAction.logContext,
                    { level: "warn", throttleMs: 1000 },
                  );
                  return;
                }
                handleSwitchTopicError(retryAction.error, topicId, {
                  preserveCurrentSnapshot: true,
                });
              } finally {
                if (!keepPendingHydration) {
                  clearPendingHydration();
                }
              }
            })();
          };

          hydrateCachedTopic();
          return;
        }

        if (
          shouldApplyPendingSessionShell({
            currentSessionId,
            topicId,
            cachedSnapshot: cachedTargetSnapshot,
          })
        ) {
          const pendingShellPlan = buildSessionSwitchPendingShellPlan({
            topicId,
          });
          if (pendingShellPlan.shouldApplyEmptySessionSnapshot) {
            applySessionSnapshot({
              ...createEmptyAgentSessionSnapshot(),
              sessionId: pendingShellPlan.sessionId,
            });
          }
          setExecutionStrategyState(
            normalizeExecutionStrategy(
              selectedTopic?.executionStrategy || executionStrategy,
            ),
          );
          if (pendingShellPlan.shouldResetHistoryWindow) {
            setSessionHistoryWindow(null);
          }
          if (pendingShellPlan.shouldApplyCachedTopicChromeState) {
            applyCachedTopicChromeState(topicId);
          }
          persistSessionRestoreCandidate(
            pendingShellPlan.restoreCandidateSessionId,
          );
          if (pendingShellPlan.shouldSetSessionHydrating) {
            setIsSessionHydrating(true);
          }
          const pendingShellMetricContext =
            buildPendingSessionShellMetricContext({
              currentSessionId,
              topicId,
              workspaceId,
            });
          recordAgentUiPerformanceMetric(
            "session.switch.pendingShellApplied",
            pendingShellMetricContext,
          );
          logAgentDebug(
            "useAgentSession",
            "switchTopic.pendingShellApplied",
            pendingShellMetricContext,
          );
        }

        const detail = await loadRuntimeSessionDetail({
          topicId,
          startedAt,
          mode: "direct",
          resumeSessionStartHooks: options?.resumeSessionStartHooks === true,
        });
        finalizeResolvedTopicDetail({
          topicId,
          detail,
          startedAt,
          localSnapshotOverride: buildSessionSwitchLocalSnapshotOverride({
            cachedSnapshot: cachedTargetSnapshot,
            currentSessionId: sessionIdRef.current,
            messages: messagesRef.current,
            threadTurns: threadTurnsRef.current,
            threadItems: threadItemsRef.current,
            topicId,
          }),
          switchRequestVersion,
          useTransition: currentSessionId !== topicId,
        });
      } catch (error) {
        if (
          !isCurrentSessionHydrationRequest({
            currentRequestVersion: sessionSwitchRequestVersionRef.current,
            requestVersion: switchRequestVersion,
          })
        ) {
          return;
        }
        handleSwitchTopicError(error, topicId);
        resolveActiveSwitch("error");
        return "error" as const;
      } finally {
        if (activeSessionSwitchRef.current?.promise === activeSwitchPromise) {
          activeSessionSwitchRef.current = null;
        }
        resolveActiveSwitch("success");
      }
    },
    [
      applyCachedTopicChromeState,
      finalizeResolvedTopicDetail,
      handleSwitchTopicError,
      loadRuntimeSessionDetail,
      messages.length,
      modelRef,
      applyCachedTopicSnapshot,
      applySessionSnapshot,
      executionStrategy,
      invalidatePendingSessionSwitches,
      persistSessionModelPreference,
      persistSessionRestoreCandidate,
      providerTypeRef,
      sessionIdRef,
      setExecutionStrategyState,
      topics,
      workspaceId,
    ],
  );

  const loadFullSessionHistory = useCallback(async () => {
    const targetSessionId = sessionIdRef.current?.trim();
    if (!targetSessionId) {
      return false;
    }

    const currentHistoryWindow = sessionHistoryWindowRef.current;
    const requestPlan = buildSessionHistoryPageRequestPlan({
      currentHistoryWindow,
      pageSize: SESSION_HISTORY_LOAD_PAGE_SIZE,
    });
    if (!requestPlan) {
      return false;
    }

    const switchRequestVersion = sessionSwitchRequestVersionRef.current;
    const startedAt = Date.now();
    setSessionHistoryWindow(requestPlan.loadingWindow);
    logAgentDebug("useAgentSession", "loadFullHistory.start", {
      historyItemCursor: requestPlan.itemCursor,
      historyTurnCursor: requestPlan.turnCursor,
      loadedEntriesCount: currentHistoryWindow?.loadedEntries ?? 0,
      loadedItemsCount: currentHistoryWindow?.loadedItems ?? 0,
      loadedTurnsCount: currentHistoryWindow?.loadedTurns ?? 0,
      nextHistoryLimit: requestPlan.nextHistoryLimit,
      sessionId: targetSessionId,
      workspaceId,
    });

    try {
      const detail = await runtime.getSession(targetSessionId, {
        ...requestPlan.requestOptions,
        source: "loadFullHistory",
      });
      if (
        !isCurrentSessionHydrationRequest({
          currentRequestVersion: sessionSwitchRequestVersionRef.current,
          requestVersion: switchRequestVersion,
          currentSessionId: sessionIdRef.current,
          targetSessionId,
        })
      ) {
        return false;
      }

      const mergePlan = buildSessionHistoryMergePlan({
        currentMessages: messagesRef.current,
        currentThreadItems: threadItemsRef.current,
        currentThreadTurns: threadTurnsRef.current,
        currentTurnId,
        detail,
        sessionId: targetSessionId,
      });
      const resultPlan = buildSessionHistoryPageResultPlan({
        currentHistoryWindow: requestPlan.loadingWindow,
        detail,
      });

      startTransition(() => {
        applySessionSnapshot({
          sessionId: targetSessionId,
          workingDir: sessionWorkingDirRef.current,
          messages: mergePlan.mergedMessages,
          threadTurns: mergePlan.mergedThreadTurns,
          threadItems: mergePlan.mergedThreadItems,
          currentTurnId: mergePlan.currentTurnId,
          threadRead,
          executionRuntime: executionRuntimeRef.current,
          todoItems,
        });
      });
      setSessionHistoryWindow(resultPlan.nextHistoryWindow);
      setTopics((prev) =>
        upsertTopicFromSessionDetail(
          prev,
          mapSessionDetailToTopic(
            targetSessionId,
            detail,
            normalizeProjectId(detail.workspace_id) ||
              normalizeProjectId(workspaceId),
          ),
        ),
      );
      logAgentDebug("useAgentSession", "loadFullHistory.success", {
        durationMs: Date.now() - startedAt,
        historyItemCursor: requestPlan.itemCursor,
        historyTurnCursor: requestPlan.turnCursor,
        historyTruncated: detail.history_truncated === true,
        incomingMessagesCount: mergePlan.incomingMessages.length,
        loadedEntriesCount: resultPlan.nextLoadedEntries,
        loadedItemsCount: resultPlan.nextLoadedItems,
        loadedTurnsCount: resultPlan.nextLoadedTurns,
        messagesCount: mergePlan.mergedMessages.length,
        nextHistoryLimit: requestPlan.nextHistoryLimit,
        sessionId: targetSessionId,
        workspaceId,
      });
      return true;
    } catch (error) {
      if (
        !isCurrentSessionHydrationRequest({
          currentSessionId: sessionIdRef.current,
          targetSessionId,
        })
      ) {
        return false;
      }

      const message = error instanceof Error ? error.message : String(error);
      setSessionHistoryWindow((current) =>
        current ? { ...current, isLoadingFull: false, error: message } : null,
      );
      logAgentDebug(
        "useAgentSession",
        "loadFullHistory.error",
        {
          durationMs: Date.now() - startedAt,
          error,
          historyItemCursor: requestPlan.itemCursor,
          historyTurnCursor: requestPlan.turnCursor,
          nextHistoryLimit: requestPlan.nextHistoryLimit,
          sessionId: targetSessionId,
          workspaceId,
        },
        { level: "error" },
      );
      toast.error(`加载历史失败: ${message}`);
      return false;
    }
  }, [
    applySessionSnapshot,
    currentTurnId,
    runtime,
    sessionIdRef,
    threadRead,
    todoItems,
    workspaceId,
  ]);

  const ensureSession = useCallback(
    async (options?: {
      targetSessionId?: string;
      skipSessionRestore?: boolean;
      skipSessionStartHooks?: boolean;
    }): Promise<string | null> => {
      const targetSessionId = options?.targetSessionId?.trim();
      if (targetSessionId) {
        if (appServerConfirmedSessionIdsRef.current.has(targetSessionId)) {
          bindTargetSessionSnapshot(targetSessionId);
          return targetSessionId;
        }

        try {
          const detail = await runtime.getSession(
            targetSessionId,
            buildSessionDetailHydrationOptions({
              source: "ensureSessionTarget",
            }),
          );
          const runtimeWorkspaceId = normalizeProjectId(detail.workspace_id);
          const currentWorkspaceId = normalizeProjectId(workspaceId);
          const runtimeWorkingDir = normalizeSessionScopeWorkingDir(
            detail.working_dir,
          );
          const currentWorkingDir =
            normalizeSessionScopeWorkingDir(normalizedWorkingDir);
          if (
            runtimeWorkingDir &&
            currentWorkingDir &&
            runtimeWorkingDir !== currentWorkingDir
          ) {
            throw new Error(
              `session workspace mismatch: expected cwd ${currentWorkingDir}, got ${runtimeWorkingDir}`,
            );
          }
          if (
            !runtimeWorkingDir &&
            runtimeWorkspaceId &&
            currentWorkspaceId &&
            runtimeWorkspaceId !== currentWorkspaceId
          ) {
            throw new Error(
              `session workspace mismatch: expected ${currentWorkspaceId}, got ${runtimeWorkspaceId}`,
            );
          }
          appServerConfirmedSessionIdsRef.current.add(targetSessionId);
          bindTargetSessionSnapshot(targetSessionId, { detail });
          return targetSessionId;
        } catch (error) {
          logAgentDebug(
            "useAgentSession",
            "ensureSession.targetSessionError",
            {
              error,
              sessionId: targetSessionId,
              workspaceId,
            },
            { level: "error" },
          );
          throw error;
        }
      }

      const existingSessionId = sessionIdRef.current?.trim();
      if (existingSessionId) {
        if (appServerConfirmedSessionIdsRef.current.has(existingSessionId)) {
          return existingSessionId;
        }

        try {
          const detail = await runtime.getSession(
            existingSessionId,
            buildSessionDetailHydrationOptions({ source: "ensureSession" }),
          );
          const runtimeWorkspaceId = normalizeProjectId(detail.workspace_id);
          const currentWorkspaceId = normalizeProjectId(workspaceId);
          const runtimeWorkingDir = normalizeSessionScopeWorkingDir(
            detail.working_dir,
          );
          const currentWorkingDir =
            normalizeSessionScopeWorkingDir(normalizedWorkingDir);
          if (
            runtimeWorkingDir &&
            currentWorkingDir &&
            runtimeWorkingDir !== currentWorkingDir
          ) {
            throw new Error(
              `session workspace mismatch: expected cwd ${currentWorkingDir}, got ${runtimeWorkingDir}`,
            );
          }
          if (
            !runtimeWorkingDir &&
            runtimeWorkspaceId &&
            currentWorkspaceId &&
            runtimeWorkspaceId !== currentWorkspaceId
          ) {
            throw new Error(
              `session workspace mismatch: expected ${currentWorkspaceId}, got ${runtimeWorkspaceId}`,
            );
          }
          appServerConfirmedSessionIdsRef.current.add(existingSessionId);
          if (sessionIdRef.current?.trim() === existingSessionId) {
            return existingSessionId;
          }
          return sessionIdRef.current?.trim() || null;
        } catch (error) {
          if (
            !isAgentSessionNotFoundError(error) &&
            !isSessionWorkspaceMismatchError(error)
          ) {
            throw error;
          }

          if (sessionIdRef.current?.trim() !== existingSessionId) {
            return sessionIdRef.current?.trim() || null;
          }

          logAgentDebug(
            "useAgentSession",
            "ensureSession.dropStaleRestoredSession",
            {
              error,
              sessionId: existingSessionId,
              workspaceId,
            },
            { level: "warn" },
          );
          appServerConfirmedSessionIdsRef.current.delete(existingSessionId);
          applySessionSnapshot(createEmptyAgentSessionSnapshot());
          setSessionHistoryWindow(null);
          persistSessionRestoreCandidate(null);
          hydratedSessionRef.current = null;
          restoredWorkspaceRef.current = null;
          missingSessionVerificationRef.current = null;
          setIsAutoRestoringSession(false);
          setIsSessionHydrating(false);
          setRecoveredStreamBindingSessionId(null);

          return createFreshSession(undefined, {
            preserveCurrentSnapshot: false,
            skipSessionStartHooks: options?.skipSessionStartHooks === true,
          });
        }
      }

      return createFreshSession(undefined, {
        preserveCurrentSnapshot: true,
        skipSessionStartHooks: options?.skipSessionStartHooks === true,
      });
    },
    [
      createFreshSession,
      applySessionSnapshot,
      bindTargetSessionSnapshot,
      normalizedWorkingDir,
      persistSessionRestoreCandidate,
      runtime,
      sessionIdRef,
      workspaceId,
    ],
  );

  const recoverSessionInBackground = useCallback(
    async (targetSessionId: string, resumeSessionStartHooks: boolean) => {
      try {
        const detail = await runtime.getSession(
          targetSessionId,
          buildSessionDetailHydrationOptions({
            resumeSessionStartHooks,
            source: "homeBackgroundRecovery",
          }),
        );
        appServerConfirmedSessionIdsRef.current.add(targetSessionId);
        setTopics((prev) =>
          upsertTopicFromSessionDetail(
            prev,
            mapSessionDetailToTopic(
              targetSessionId,
              detail,
              normalizeProjectId(detail.workspace_id) ||
                normalizeProjectId(workspaceId),
            ),
          ),
        );
      } catch (error) {
        if (isAgentSessionNotFoundError(error)) {
          persistSessionRestoreCandidate(null);
          return;
        }
        console.warn("[AgentChat] 后台恢复会话失败:", error);
        logAgentDebug(
          "useAgentSession",
          "backgroundRestore.error",
          {
            error,
            targetSessionId,
            workspaceId,
          },
          { level: "warn" },
        );
      } finally {
        setIsAutoRestoringSession(false);
        setIsSessionHydrating(false);
      }
    },
    [persistSessionRestoreCandidate, runtime, workspaceId],
  );

  const refreshSessionDetail = useCallback(
    async (
      targetSessionId?: string,
      request?: AgentSessionDetailRefreshRequest,
    ) => {
      return refreshAgentSessionDetailState({
        runtime,
        sessionIdRef,
        targetSessionId,
        source: request?.source ?? "runtimeSync.refreshDetail",
        detailMergeMode: request?.detailMergeMode,
        applySessionDetail,
        markSessionExecutionStrategySynced,
        persistSessionAccessMode,
        setAccessModeState,
        onWarn: (error) => {
          console.warn("[AgentChat] 刷新会话详情失败:", error);
        },
      });
    },
    [
      applySessionDetail,
      markSessionExecutionStrategySynced,
      persistSessionAccessMode,
      runtime,
      sessionIdRef,
      setAccessModeState,
    ],
  );

  const refreshSessionReadModel = useCallback(
    async (targetSessionId?: string) => {
      return refreshAgentSessionReadModelState({
        runtime,
        sessionIdRef,
        targetSessionId,
        applyReadModelSnapshot,
        onWarn: (error) => {
          console.warn("[AgentChat] 刷新运行态摘要失败:", error);
        },
      });
    },
    [applyReadModelSnapshot, runtime, sessionIdRef],
  );

  const attemptSilentTurnRecovery = useCallback(
    async (
      targetSessionId: string,
      requestStartedAt: number,
      promptText: string,
      options?: {
        requireTerminal?: boolean;
        turnId?: string | null;
      },
    ) => {
      const resolvedSessionId = targetSessionId.trim();
      if (!resolvedSessionId) {
        return false;
      }

      try {
        const detail = await runtime.getSession(
          resolvedSessionId,
          buildSessionDetailHydrationOptions({ source: "silentTurnRecovery" }),
        );
        if (
          !isCurrentSessionHydrationRequest({
            currentSessionId: sessionIdRef.current,
            targetSessionId: resolvedSessionId,
          })
        ) {
          return false;
        }
        if (
          options?.requireTerminal
            ? !hasRecoverableTerminalTurnActivity(
                detail,
                requestStartedAt,
                promptText,
                options.turnId,
              )
            : !hasRecoverableSilentTurnActivity(
                detail,
                requestStartedAt,
                promptText,
              )
        ) {
          return false;
        }

        applySessionDetail(resolvedSessionId, detail, {
          preserveExecutionStrategyOnMissingDetail: true,
          detailMergeMode: "terminal_reconcile",
        });
        if (detail.execution_strategy) {
          markSessionExecutionStrategySynced(
            resolvedSessionId,
            normalizeExecutionStrategy(detail.execution_strategy),
          );
        }
        return true;
      } catch (error) {
        console.warn("[AgentChat] 静默 turn 恢复失败:", error);
        return false;
      }
    },
    [
      applySessionDetail,
      markSessionExecutionStrategySynced,
      runtime,
      sessionIdRef,
    ],
  );

  useEffect(() => {
    const resolvedWorkspaceId = workspaceId?.trim() || "";
    if (disableSessionRestore) return;
    if (!topicsReady) return;
    if (skipAutoRestoreRef.current) return;
    if (sessionId) return;
    if (sessionIdRef.current || hasActiveLocalSessionSnapshot()) return;
    if (restoredWorkspaceRef.current === resolvedWorkspaceId) return;

    const scopedCandidate = restoreCandidateSessionIdRef.current;
    const targetSessionId = resolveRestorableTopicSessionId(
      scopedCandidate,
      topics,
      {
        allowDetachedCandidate: topicsListMayBeTruncatedRef.current,
      },
    );
    if (!targetSessionId) {
      setIsAutoRestoringSession(false);
      setIsSessionHydrating(false);
      logAgentDebug(
        "useAgentSession",
        "autoRestore.skipWithoutTarget",
        {
          candidateSessionId: scopedCandidate,
          topicsCount: topics.length,
          workspaceId: resolvedWorkspaceId,
        },
        { throttleMs: 1000 },
      );
      return;
    }

    restoredWorkspaceRef.current = resolvedWorkspaceId;
    setIsSessionHydrating(false);
    const targetTopic = topics.find((topic) => topic.id === targetSessionId);
    const shouldResumeRestoredSession =
      shouldResumeTaskSession(targetTopic) ||
      (shouldRecoverSessionInBackground && scopedCandidate === targetSessionId);

    if (shouldRecoverSessionInBackground) {
      persistSessionRestoreCandidate(targetSessionId);
      setRecoveredStreamBindingSessionId(null);
      logAgentDebug("useAgentSession", "backgroundRestore.start", {
        candidateSessionId: scopedCandidate,
        resumeSessionStartHooks: shouldResumeRestoredSession,
        targetSessionId,
        restoreSource: topics.length > 0 ? "topics_snapshot" : "shadow_cache",
        topicsCount: topics.length,
        workspaceId: resolvedWorkspaceId,
      });
      void recoverSessionInBackground(
        targetSessionId,
        shouldResumeRestoredSession,
      );
      return;
    }

    let cancelled = false;
    setIsAutoRestoringSession(true);
    setRecoveredStreamBindingSessionId(targetSessionId);
    logAgentDebug("useAgentSession", "autoRestore.start", {
      candidateSessionId: scopedCandidate,
      resumeSessionStartHooks: shouldResumeRestoredSession,
      targetSessionId,
      restoreSource: topics.length > 0 ? "topics_snapshot" : "shadow_cache",
      topicsCount: topics.length,
      workspaceId: resolvedWorkspaceId,
    });
    switchTopic(targetSessionId, {
      resumeSessionStartHooks: shouldResumeRestoredSession,
      restoreSource: "auto",
    })
      .catch((error) => {
        console.warn("[AgentChat] 自动恢复会话失败:", error);
        logAgentDebug(
          "useAgentSession",
          "autoRestore.error",
          {
            error,
            targetSessionId,
            workspaceId: resolvedWorkspaceId,
          },
          { level: "warn" },
        );
        persistSessionRestoreCandidate(null);
      })
      .finally(() => {
        if (!cancelled) {
          setIsAutoRestoringSession(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [
    disableSessionRestore,
    persistSessionRestoreCandidate,
    recoverSessionInBackground,
    sessionId,
    sessionIdRef,
    hasActiveLocalSessionSnapshot,
    shouldRecoverSessionInBackground,
    switchTopic,
    topics,
    topicsReady,
    workspaceId,
  ]);

  useEffect(() => {
    if (sessionId) {
      skipAutoRestoreRef.current = false;
    }
  }, [sessionId]);

  useEffect(() => {
    if (!sessionId) return;
    if (!topicsReady) return;
    if (activeStreamingTimeline || hasActiveStreamingTimelineNow()) return;

    const sessionMissingFromTopics =
      topics.length > 0 && !topics.some((topic) => topic.id === sessionId);
    if (activeSessionSwitchRef.current?.topicId === sessionId) {
      return;
    }
    if (pendingSessionDetailHydrationRef.current?.topicId === sessionId) {
      return;
    }

    const restoreCandidateSessionId =
      restoreCandidateSessionIdRef.current?.trim() || null;
    const currentSessionHasLocalTimeline =
      messages.length > 0 ||
      hasSessionHydrationActivity({
        currentTurnId,
        threadItemsCount: threadItems.length,
        threadTurnsCount: threadTurns.length,
      });
    const restoreCandidateMayLagTopics =
      restoreCandidateSessionId === sessionId &&
      (bootPersistedRestoreCandidateSessionIdRef.current === sessionId ||
        topicsListMayBeTruncatedRef.current ||
        !currentSessionHasLocalTimeline);
    const missingSessionAction = resolveMissingSessionFromTopicsAction({
      currentTurnId,
      detachedSessionId: detachedSessionIdRef.current,
      remoteConfirmed: appServerConfirmedSessionIdsRef.current.has(sessionId),
      restoreCandidateMayLagTopics,
      sessionId,
      threadItemsCount: threadItems.length,
      threadTurnsCount: threadTurns.length,
      topicsCount: topics.length,
      topicsReady,
      topicExists: !sessionMissingFromTopics,
    });

    if (missingSessionAction.kind === "skip_detached") {
      missingSessionVerificationRef.current = null;
      return;
    }

    if (
      missingSessionAction.kind === "clear_auxiliary" ||
      missingSessionAction.kind === "clear_inactive"
    ) {
      if (missingSessionAction.kind === "clear_inactive") {
        applySessionSnapshot(
          createEmptyAgentSessionSnapshot({
            executionRuntime: executionRuntimeRef.current,
          }),
        );
        hydratedSessionRef.current = null;
      }
      persistSessionRestoreCandidate(null);
      missingSessionVerificationRef.current = null;
      return;
    }

    if (missingSessionAction.kind === "verify_remote") {
      if (missingSessionVerificationRef.current === sessionId) {
        return;
      }

      missingSessionVerificationRef.current = sessionId;
      logAgentDebug(
        "useAgentSession",
        "hydrateSession.sessionMissingFromTopics",
        {
          sessionId,
          topicsCount: topics.length,
          workspaceId,
        },
        { level: "warn" },
      );

      runtime
        .getSession(
          sessionId,
          buildSessionDetailHydrationOptions({
            source: "missingSessionVerify",
          }),
        )
        .then((detail) => {
          if (
            !isCurrentSessionHydrationRequest({
              currentSessionId: sessionIdRef.current,
              targetSessionId: sessionId,
            })
          ) {
            return;
          }

          appServerConfirmedSessionIdsRef.current.add(sessionId);
          setTopics((prev) =>
            prependVerifiedSessionTopicFromDetail(prev, sessionId, detail),
          );
        })
        .catch((error) => {
          if (!isAgentSessionNotFoundError(error)) {
            console.warn("[AgentChat] 校验当前会话存在性失败:", error);
            return;
          }

          if (
            !isCurrentSessionHydrationRequest({
              currentSessionId: sessionIdRef.current,
              targetSessionId: sessionId,
            })
          ) {
            return;
          }

          applySessionSnapshot(createEmptyAgentSessionSnapshot());
          persistSessionRestoreCandidate(null);
          hydratedSessionRef.current = null;
          restoredWorkspaceRef.current = null;
          skipAutoRestoreRef.current = false;
        })
        .finally(() => {
          if (missingSessionVerificationRef.current === sessionId) {
            missingSessionVerificationRef.current = null;
          }
        });
      return;
    }

    missingSessionVerificationRef.current = null;

    const hasLocalTimelineCache =
      messages.length > 0 && (threadTurns.length > 0 || threadItems.length > 0);
    const hasPreservedMessageCache =
      preserveRestoredMessages && messages.length > 0;
    const selectedTopic = topics.find((topic) => topic.id === sessionId);
    if (isSessionHydrating && hydratedSessionRef.current === sessionId) {
      return;
    }
    if (activeSessionSwitchRef.current?.topicId === sessionId) {
      return;
    }
    if (
      shouldSkipAlreadyHydratedSession({
        currentTurnId,
        hydratedSessionId: hydratedSessionRef.current,
        messagesCount: messages.length,
        selectedTopic,
        sessionId,
        threadReadStatus: threadRead?.status,
        threadItemsCount: threadItems.length,
        threadTurnsCount: threadTurns.length,
      })
    ) {
      return;
    }

    const shouldResumeHydrationSession = shouldResumeTaskSession(selectedTopic);
    logAgentDebug("useAgentSession", "hydrateSession.start", {
      cacheMode: hasLocalTimelineCache
        ? "timeline_cache"
        : hasPreservedMessageCache
          ? "message_cache"
          : "empty",
      messagesCount: messages.length,
      resumeSessionStartHooks: shouldResumeHydrationSession,
      sessionId,
      threadItemsCount: threadItems.length,
      threadTurnsCount: threadTurns.length,
      workspaceId,
    });

    switchTopic(sessionId, {
      forceRefresh: true,
      ...(restoreCandidateSessionId === sessionId
        ? { restoreSource: "auto" as const }
        : {}),
      ...(shouldResumeHydrationSession
        ? { resumeSessionStartHooks: true }
        : {}),
      ...(detachedSessionIdRef.current === sessionId
        ? { allowDetachedSession: true }
        : {}),
    }).catch((error) => {
      console.warn("[AgentChat] 会话水合失败:", error);
      logAgentDebug(
        "useAgentSession",
        "hydrateSession.error",
        {
          error,
          sessionId,
          workspaceId,
        },
        { level: "warn" },
      );
      hydratedSessionRef.current = null;
    });
  }, [
    messages.length,
    activeStreamingTimeline,
    currentTurnId,
    hasActiveStreamingTimelineNow,
    preserveRestoredMessages,
    persistSessionRestoreCandidate,
    isSessionHydrating,
    runtime,
    sessionId,
    sessionIdRef,
    switchTopic,
    threadRead?.status,
    threadItems.length,
    threadTurns.length,
    topics,
    topicsReady,
    workspaceId,
    applySessionSnapshot,
  ]);

  useEffect(() => {
    logAgentDebug(
      "useAgentSession",
      "stateSnapshot",
      {
        currentTurnId: currentTurnId ?? null,
        messagesCount: messages.length,
        sessionId: sessionId ?? null,
        threadItemsCount: threadItems.length,
        threadTurnsCount: threadTurns.length,
        topicsCount: topics.length,
        topicsReady,
        workspaceId,
      },
      {
        dedupeKey: JSON.stringify({
          currentTurnId: currentTurnId ?? null,
          messagesCount: messages.length,
          sessionId: sessionId ?? null,
          threadItemsCount: threadItems.length,
          threadTurnsCount: threadTurns.length,
          topicsCount: topics.length,
          topicsReady,
          workspaceId,
        }),
        throttleMs: 800,
      },
    );
  }, [
    currentTurnId,
    messages.length,
    sessionId,
    threadItems.length,
    threadTurns.length,
    topics.length,
    topicsReady,
    activeStreamingTimeline,
    workspaceId,
  ]);

  const deleteTopic = useCallback(
    async (topicId: string) => {
      try {
        await runtime.deleteSession(topicId);
        await loadTopics();

        if (topicId === sessionIdRef.current) {
          applySessionSnapshot(createEmptyAgentSessionSnapshot());
          resetPendingActions();
          resetStreamingRefs();
          hydratedSessionRef.current = null;
          restoredWorkspaceRef.current = null;
          persistSessionRestoreCandidate(null);
          saveTransient(scopedKeys.turnsKey, []);
          saveTransient(scopedKeys.itemsKey, []);
          saveTransient(scopedKeys.currentTurnKey, null);
        }

        toast.success(
          tNavigation("navigation.sidebar.conversations.delete.success"),
        );
      } catch (error) {
        console.error("[AgentChat] 删除任务失败:", error);
        toast.error(
          tNavigation("navigation.sidebar.conversations.delete.error"),
        );
      }
    },
    [
      applySessionSnapshot,
      loadTopics,
      persistSessionRestoreCandidate,
      resetPendingActions,
      resetStreamingRefs,
      runtime,
      scopedKeys,
      sessionIdRef,
      tNavigation,
    ],
  );

  const renameTopic = useCallback(
    async (topicId: string, newTitle: string) => {
      const normalizedTitle = newTitle.trim();
      if (!normalizedTitle) {
        return;
      }

      try {
        await runtime.renameSession(topicId, normalizedTitle);
        await loadTopics();
        toast.success(
          tNavigation("navigation.sidebar.conversations.rename.success"),
        );
      } catch (error) {
        console.error("[AgentChat] 重命名任务失败:", error);
        toast.error(
          tNavigation("navigation.sidebar.conversations.rename.error"),
        );
      }
    },
    [loadTopics, runtime, tNavigation],
  );

  const archiveTopic = useCallback(
    async (topicId: string): Promise<boolean> => {
      try {
        if (!runtime.archiveSession) {
          throw new Error("archive session capability is unavailable");
        }
        await runtime.archiveSession(topicId);
        await loadTopics();

        if (topicId === sessionIdRef.current) {
          applySessionSnapshot(createEmptyAgentSessionSnapshot());
          resetPendingActions();
          resetStreamingRefs();
          hydratedSessionRef.current = null;
          restoredWorkspaceRef.current = null;
          persistSessionRestoreCandidate(null);
          saveTransient(scopedKeys.turnsKey, []);
          saveTransient(scopedKeys.itemsKey, []);
          saveTransient(scopedKeys.currentTurnKey, null);
        }

        toast.success(
          tNavigation("navigation.sidebar.conversations.archive.success"),
        );
        return true;
      } catch (error) {
        console.error("[AgentChat] 归档任务失败:", error);
        toast.error(
          tNavigation("navigation.sidebar.conversations.archive.error"),
        );
        return false;
      }
    },
    [
      applySessionSnapshot,
      loadTopics,
      persistSessionRestoreCandidate,
      resetPendingActions,
      resetStreamingRefs,
      runtime,
      scopedKeys,
      sessionIdRef,
      tNavigation,
    ],
  );

  const forkTopic = useCallback(
    async (topicId: string): Promise<string | null> => {
      try {
        if (!runtime.forkSession) {
          throw new Error("fork session capability is unavailable");
        }
        const forkedSessionId = await runtime.forkSession(topicId);
        await loadTopics();
        toast.success(
          tNavigation("navigation.sidebar.conversations.fork.success"),
        );
        return forkedSessionId;
      } catch (error) {
        console.error("[AgentChat] 分叉任务失败:", error);
        toast.error(tNavigation("navigation.sidebar.conversations.fork.error"));
        return null;
      }
    },
    [loadTopics, runtime, tNavigation],
  );

  const updateTopicExecutionStrategy = useCallback(
    (
      targetSessionId: string,
      nextExecutionStrategy: AgentExecutionStrategy,
    ) => {
      setTopics((prev) =>
        applyTopicExecutionStrategyToTopics(
          prev,
          targetSessionId,
          nextExecutionStrategy,
        ),
      );
    },
    [],
  );

  const updateTopicSnapshot = useCallback(
    (targetSessionId: string, snapshot: TopicSnapshotPatch) => {
      setTopics((prev) =>
        applyTopicSnapshotToTopics(prev, targetSessionId, snapshot),
      );
    },
    [],
  );

  const queuedTurnCount = useMemo(() => {
    const topicCount =
      topics.find((topic) => topic.id === sessionId)?.queuedTurnCount ?? 0;
    const status = threadRead?.status?.trim().toLowerCase();
    if (status === "queued") {
      return Math.max(topicCount, 1);
    }
    if (
      status === "completed" ||
      status === "failed" ||
      status === "canceled" ||
      status === "cancelled" ||
      status === "aborted"
    ) {
      return 0;
    }
    return topicCount;
  }, [sessionId, threadRead?.status, topics]);

  return {
    sessionId,
    setSessionId,
    messages,
    setMessages: setMessagesState,
    threadTurns,
    setThreadTurns: setThreadTurnsState,
    threadItems,
    getThreadItems,
    setThreadItems: setThreadItemsState,
    currentTurnId,
    setCurrentTurnId,
    todoItems,
    setTodoItems,
    queuedTurnCount,
    threadRead,
    getThreadIdForSubmit,
    executionRuntime,
    sessionWorkingDir,
    setExecutionRuntime,
    topics,
    setTopics,
    topicsReady,
    sessionHistoryWindow,
    isAutoRestoringSession,
    isSessionHydrating,
    recoveredStreamBindingSessionId,
    isDetachedActiveSession: detachedSessionIdRef.current === sessionId,
    loadTopics,
    createFreshSession,
    ensureSession,
    switchTopic,
    loadFullSessionHistory,
    deleteTopic,
    renameTopic,
    archiveTopic,
    forkTopic,
    refreshSessionDetail,
    refreshSessionReadModel,
    attemptSilentTurnRecovery,
    clearMessages,
    deleteMessage,
    editMessage,
    updateTopicExecutionStrategy,
    updateTopicSnapshot,
  };
}
