import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AGENT_RUNTIME_SESSIONS_CHANGED_EVENT,
  listAgentRuntimeSessions,
} from "@/lib/api/agentRuntime/sessionClient";
import type { AgentRuntimeSessionsChangedDetail } from "@/lib/api/agentRuntime/sessionClient";
import type { AgentSessionInfo } from "@/lib/api/agentRuntime/sessionTypes";
import {
  recordAgentUiPerformanceMetric,
  subscribeAgentUiPerformanceMetricRecorded,
  type AgentUiPerformanceMetricRecordedDetail,
} from "@/lib/agentUiPerformanceMetrics";
import { logAgentDebug } from "@/lib/agentDebug";
import { scheduleMinimumDelayIdleTask } from "@/lib/utils/scheduleMinimumDelayIdleTask";
import {
  SIDEBAR_CONVERSATION_NAVIGATION_DEFER_MS,
  SIDEBAR_NEW_TASK_HOME_SESSION_LOAD_DEFER_MS,
  SIDEBAR_RECENT_SESSION_PAGE_SIZE,
  SIDEBAR_SEARCH_RESULT_LIMIT,
  SIDEBAR_SESSION_ENTRY_REFRESH_DEFER_MS,
  SIDEBAR_SESSION_LOAD_RESTART_DEFER_MS,
} from "./AppSidebar.constants";
import {
  buildImportedSidebarSession,
  buildSidebarSessionRequestLimit,
  buildVisibleSidebarSessions,
  hasCachedSidebarSessionEntry,
  matchesSidebarSessionTitle,
  normalizeSidebarSearchText,
  filterSidebarSessions,
  splitSidebarSessionResult,
} from "./sidebarSessions";
import type { ConversationImportThreadCommitResponse } from "@/lib/api/conversationImport";

const SIDEBAR_SEND_HOT_PATH_DEFER_PHASES = new Set([
  "homeInput.submit",
  "homeInput.sendDispatch.start",
  "workspaceSend.plan.ready",
  "agentStream.request.start",
  "agentStream.submitDispatched",
]);

interface UseAppSidebarSessionsParams {
  currentSessionId: string | null;
  activeProjectIds?: string[];
  openedProjectCwds?: string[];
  requireOpenedProjectCwd?: boolean;
  shouldShowConversationList: boolean;
  sidebarSearchOpen: boolean;
  sidebarSearchQuery: string;
  isNewTaskHome: boolean;
  isClawTaskCenter: boolean;
  activeAgentStreaming?: boolean;
  conversationUntitledLabel: string;
}

function mergeSidebarSessions(
  sessionGroups: AgentSessionInfo[][],
): AgentSessionInfo[] {
  const sessionsById = new Map<string, AgentSessionInfo>();
  sessionGroups.flat().forEach((session) => {
    sessionsById.set(session.id, session);
  });
  return filterSidebarSessions([...sessionsById.values()]);
}

function normalizeSessionProjectValue(value?: string | null): string | null {
  const normalized = value?.trim();
  return normalized ? normalized : null;
}

function buildSidebarSessionLoadRequests(params: {
  limit: number;
  projectCwds: string[];
}): Array<Promise<AgentSessionInfo[]>> {
  const { limit, projectCwds } = params;
  const requests: Array<Promise<AgentSessionInfo[]>> = [
    listAgentRuntimeSessions({
      limit,
    }),
  ];

  for (const cwd of projectCwds) {
    requests.push(
      listAgentRuntimeSessions({
        limit,
        cwd,
      }),
    );
  }

  return requests;
}

function shouldDeferCurrentSessionMetadataRefresh(params: {
  currentSessionId: string | null;
  detail: unknown;
}): boolean {
  const detail = params.detail as
    | Partial<AgentRuntimeSessionsChangedDetail>
    | null
    | undefined;
  if (detail?.reason !== "updated") {
    return false;
  }

  const changedSessionId = detail.sessionId?.trim();
  return Boolean(
    changedSessionId &&
    params.currentSessionId &&
    changedSessionId === params.currentSessionId,
  );
}

export function useAppSidebarSessions({
  currentSessionId,
  activeProjectIds = [],
  openedProjectCwds = [],
  requireOpenedProjectCwd = false,
  shouldShowConversationList,
  sidebarSearchOpen,
  sidebarSearchQuery,
  isNewTaskHome,
  isClawTaskCenter,
  activeAgentStreaming = false,
  conversationUntitledLabel,
}: UseAppSidebarSessionsParams) {
  const sidebarSessionsRef = useRef<AgentSessionInfo[]>([]);
  const optimisticSidebarSessionsRef = useRef<Map<string, AgentSessionInfo>>(
    new Map(),
  );
  const [sidebarSessions, setSidebarSessions] = useState<AgentSessionInfo[]>(
    [],
  );
  const [sidebarSessionsHasMore, setSidebarSessionsHasMore] = useState(false);
  const [sidebarSessionsLoading, setSidebarSessionsLoading] = useState(false);
  const [sidebarSessionActionId, setSidebarSessionActionId] = useState<
    string | null
  >(null);
  const [recentSessionsVisibleCount, setRecentSessionsVisibleCount] = useState(
    SIDEBAR_RECENT_SESSION_PAGE_SIZE,
  );
  const conversationNavigationDeferUntilRef = useRef(0);
  const recentSidebarLoadInFlightRef = useRef(false);
  const recentSidebarReloadPendingRef = useRef(false);
  const recentSidebarReloadCancelRef = useRef<(() => void) | null>(null);
  const loadRecentSidebarSessionsRef = useRef<() => Promise<void>>(
    async () => undefined,
  );
  const activeAgentStreamingRef = useRef(activeAgentStreaming);
  const agentSendHotPathDeferUntilRef = useRef(0);
  const sidebarFocusRefreshCancelRef = useRef<(() => void) | null>(null);
  const newTaskHomeSessionLoadCancelRef = useRef<(() => void) | null>(null);
  const shouldLoadSidebarConversations =
    shouldShowConversationList || sidebarSearchOpen;
  const openedProjectCwdsKey = useMemo(() => {
    const seen = new Set<string>();
    return openedProjectCwds
      .map((cwd) => cwd.trim().replace(/[\\/]+$/u, ""))
      .filter((cwd) => {
        if (!cwd || seen.has(cwd)) {
          return false;
        }
        seen.add(cwd);
        return true;
      })
      .join("\n");
  }, [openedProjectCwds]);
  const activeProjectIdsKey = useMemo(() => {
    const seen = new Set<string>();
    return activeProjectIds
      .map((projectId) => projectId.trim())
      .filter((projectId) => {
        if (!projectId || seen.has(projectId)) {
          return false;
        }
        seen.add(projectId);
        return true;
      })
      .join("\n");
  }, [activeProjectIds]);
  const normalizedActiveProjectIds = useMemo(
    () => (activeProjectIdsKey ? activeProjectIdsKey.split("\n") : []),
    [activeProjectIdsKey],
  );
  const normalizedOpenedProjectCwds = useMemo(
    () => (openedProjectCwdsKey ? openedProjectCwdsKey.split("\n") : []),
    [openedProjectCwdsKey],
  );
  const shouldShowSessionLoadingState =
    sidebarSessionsLoading && sidebarSessions.length === 0;
  const hasCachedCurrentSessionSidebarEntry = hasCachedSidebarSessionEntry(
    sidebarSessionsRef.current,
    currentSessionId,
  );

  useEffect(() => {
    return () => {
      recentSidebarReloadCancelRef.current?.();
      recentSidebarReloadCancelRef.current = null;
      sidebarFocusRefreshCancelRef.current?.();
      sidebarFocusRefreshCancelRef.current = null;
      newTaskHomeSessionLoadCancelRef.current?.();
      newTaskHomeSessionLoadCancelRef.current = null;
    };
  }, []);

  useEffect(() => {
    sidebarSessionsRef.current = sidebarSessions;
  }, [sidebarSessions]);

  useEffect(() => {
    setRecentSessionsVisibleCount(SIDEBAR_RECENT_SESSION_PAGE_SIZE);
  }, [activeProjectIdsKey, openedProjectCwdsKey]);

  const recentSessionRequestLimit = useMemo(() => {
    return buildSidebarSessionRequestLimit(
      recentSessionsVisibleCount,
      SIDEBAR_RECENT_SESSION_PAGE_SIZE,
    );
  }, [recentSessionsVisibleCount]);

  const scheduleRecentSidebarReload = useCallback(
    (minimumDelayMs: number) => {
      const hotPathRemainingMs = Math.max(
        0,
        agentSendHotPathDeferUntilRef.current - Date.now(),
      );
      const effectiveMinimumDelayMs = Math.max(
        minimumDelayMs,
        hotPathRemainingMs,
      );

      if (
        activeAgentStreamingRef.current &&
        sidebarSessionsRef.current.length > 0
      ) {
        recentSidebarReloadPendingRef.current = true;
        logAgentDebug(
          "AppSidebar",
          "recentConversations.load.deferredForActiveStream",
          {
            currentSessionId,
            minimumDelayMs: effectiveMinimumDelayMs,
          },
          {
            dedupeKey: `appSidebar.recentConversations.load.deferredForActiveStream:${currentSessionId ?? "none"}`,
            throttleMs: 1000,
          },
        );
        return;
      }

      recentSidebarReloadCancelRef.current?.();
      recentSidebarReloadCancelRef.current = scheduleMinimumDelayIdleTask(
        () => {
          recentSidebarReloadCancelRef.current = null;
          recentSidebarReloadPendingRef.current = false;
          void loadRecentSidebarSessionsRef.current();
        },
        {
          minimumDelayMs: effectiveMinimumDelayMs,
          idleTimeoutMs: Math.max(
            effectiveMinimumDelayMs,
            SIDEBAR_SESSION_LOAD_RESTART_DEFER_MS,
          ),
        },
      );
    },
    [currentSessionId],
  );

  useEffect(() => {
    activeAgentStreamingRef.current = activeAgentStreaming;
    if (activeAgentStreaming) {
      recentSidebarReloadCancelRef.current?.();
      recentSidebarReloadCancelRef.current = null;
      return;
    }

    if (!recentSidebarReloadPendingRef.current) {
      return;
    }

    recentSidebarReloadPendingRef.current = false;
    scheduleRecentSidebarReload(SIDEBAR_SESSION_ENTRY_REFRESH_DEFER_MS);
  }, [activeAgentStreaming, scheduleRecentSidebarReload]);

  useEffect(() => {
    const handleMetric = (detail: AgentUiPerformanceMetricRecordedDetail) => {
      if (!SIDEBAR_SEND_HOT_PATH_DEFER_PHASES.has(detail.phase)) {
        return;
      }

      agentSendHotPathDeferUntilRef.current = Math.max(
        agentSendHotPathDeferUntilRef.current,
        Date.now() + SIDEBAR_SESSION_ENTRY_REFRESH_DEFER_MS,
      );
      recentSidebarReloadPendingRef.current = true;
      recentSidebarReloadCancelRef.current?.();
      recentSidebarReloadCancelRef.current = null;
      setSidebarSessionsLoading(false);
      logAgentDebug(
        "AppSidebar",
        "recentConversations.load.deferredForSendHotPath",
        {
          currentSessionId,
          phase: detail.phase,
          minimumDelayMs: SIDEBAR_SESSION_ENTRY_REFRESH_DEFER_MS,
        },
        {
          dedupeKey: `appSidebar.recentConversations.load.deferredForSendHotPath:${currentSessionId ?? "none"}`,
          throttleMs: 1000,
        },
      );
    };

    return subscribeAgentUiPerformanceMetricRecorded(handleMetric);
  }, [currentSessionId]);

  const loadRecentSidebarSessions = useCallback(async () => {
    if (!shouldLoadSidebarConversations) {
      setSidebarSessions([]);
      setSidebarSessionsHasMore(false);
      setSidebarSessionsLoading(false);
      return;
    }

    if (requireOpenedProjectCwd && normalizedOpenedProjectCwds.length === 0) {
      setSidebarSessions([]);
      setSidebarSessionsHasMore(false);
      setSidebarSessionsLoading(false);
      return;
    }

    const hotPathRemainingMs = Math.max(
      0,
      agentSendHotPathDeferUntilRef.current - Date.now(),
    );
    if (hotPathRemainingMs > 0) {
      recentSidebarReloadPendingRef.current = true;
      setSidebarSessionsLoading(false);
      scheduleRecentSidebarReload(hotPathRemainingMs);
      logAgentDebug(
        "AppSidebar",
        "recentConversations.load.deferredForSendHotPath",
        {
          currentSessionId,
          limit: recentSessionRequestLimit,
          minimumDelayMs: hotPathRemainingMs,
          projectIds: normalizedActiveProjectIds,
          projectCwds: normalizedOpenedProjectCwds,
        },
        {
          dedupeKey: `appSidebar.recentConversations.load.deferredForSendHotPath:${currentSessionId ?? "none"}`,
          throttleMs: 1000,
        },
      );
      return;
    }

    if (
      activeAgentStreamingRef.current &&
      sidebarSessionsRef.current.length > 0
    ) {
      recentSidebarReloadPendingRef.current = true;
      setSidebarSessionsLoading(false);
      logAgentDebug(
        "AppSidebar",
        "recentConversations.load.deferredForActiveStream",
        {
          currentSessionId,
          limit: recentSessionRequestLimit,
          projectIds: normalizedActiveProjectIds,
          projectCwds: normalizedOpenedProjectCwds,
        },
        {
          dedupeKey: `appSidebar.recentConversations.load.deferredForActiveStream:${currentSessionId ?? "none"}`,
          throttleMs: 1000,
        },
      );
      return;
    }

    if (recentSidebarLoadInFlightRef.current) {
      recentSidebarReloadPendingRef.current = true;
      return;
    }

    const deferRemainingMs =
      conversationNavigationDeferUntilRef.current - Date.now();
    if (deferRemainingMs > 0 && sidebarSessionsRef.current.length > 0) {
      scheduleRecentSidebarReload(deferRemainingMs);
      return;
    }

    recentSidebarLoadInFlightRef.current = true;
    setSidebarSessionsLoading(
      (current) => current || sidebarSessionsRef.current.length === 0,
    );
    const startedAt = Date.now();
    logAgentDebug("AppSidebar", "recentConversations.load.start", {
      limit: recentSessionRequestLimit,
      projectIds: normalizedActiveProjectIds,
      projectCwds: normalizedOpenedProjectCwds,
    });
    try {
      const sessionGroups = await Promise.all(
        buildSidebarSessionLoadRequests({
          limit: recentSessionRequestLimit,
          projectCwds: normalizedOpenedProjectCwds,
        }),
      );
      const listDurationMs = Date.now() - startedAt;
      const sortStartedAt = Date.now();
      const orderedSessions = mergeSidebarSessions(sessionGroups);
      for (const session of orderedSessions) {
        optimisticSidebarSessionsRef.current.delete(session.id);
      }
      const nextSessions = filterSidebarSessions([
        ...orderedSessions,
        ...optimisticSidebarSessionsRef.current.values(),
      ]);
      const sortDurationMs = Date.now() - sortStartedAt;
      const { hasMore } = splitSidebarSessionResult({
        sessions: nextSessions,
        visibleCount: recentSessionsVisibleCount,
        pageSize: SIDEBAR_RECENT_SESSION_PAGE_SIZE,
      });
      const metricContext = {
        hasMore,
        limit: recentSessionRequestLimit,
        listDurationMs,
        sessionsCount: nextSessions.length,
        sortDurationMs,
        totalDurationMs: Date.now() - startedAt,
        visibleCount: recentSessionsVisibleCount,
        projectIds: normalizedActiveProjectIds,
        projectCwds: normalizedOpenedProjectCwds,
      };
      recordAgentUiPerformanceMetric(
        "appSidebar.recentConversations.loadBreakdown",
        metricContext,
      );
      logAgentDebug(
        "AppSidebar",
        "recentConversations.load.success",
        metricContext,
        {
          dedupeKey: `appSidebar.recentConversations.load.success:${activeProjectIdsKey}:${openedProjectCwdsKey}:${recentSessionRequestLimit}`,
          throttleMs: 1000,
        },
      );
      setSidebarSessions(nextSessions);
      setSidebarSessionsHasMore(hasMore);
    } catch (error) {
      console.warn("加载导航任务列表失败:", error);
      logAgentDebug(
        "AppSidebar",
        "recentConversations.load.error",
        {
          durationMs: Date.now() - startedAt,
          error,
          limit: recentSessionRequestLimit,
          projectIds: normalizedActiveProjectIds,
          projectCwds: normalizedOpenedProjectCwds,
        },
        { level: "warn" },
      );
      setSidebarSessions([]);
      setSidebarSessionsHasMore(false);
    } finally {
      recentSidebarLoadInFlightRef.current = false;
      setSidebarSessionsLoading(false);
      if (recentSidebarReloadPendingRef.current) {
        recentSidebarReloadPendingRef.current = false;
        scheduleRecentSidebarReload(SIDEBAR_SESSION_LOAD_RESTART_DEFER_MS);
      }
    }
  }, [
    normalizedActiveProjectIds,
    normalizedOpenedProjectCwds,
    activeProjectIdsKey,
    openedProjectCwdsKey,
    currentSessionId,
    recentSessionRequestLimit,
    recentSessionsVisibleCount,
    requireOpenedProjectCwd,
    scheduleRecentSidebarReload,
    shouldLoadSidebarConversations,
  ]);

  useEffect(() => {
    loadRecentSidebarSessionsRef.current = loadRecentSidebarSessions;
  }, [loadRecentSidebarSessions]);

  const refreshSidebarSessions = useCallback(async () => {
    await loadRecentSidebarSessions();
  }, [loadRecentSidebarSessions]);

  useEffect(() => {
    if (!shouldLoadSidebarConversations) {
      newTaskHomeSessionLoadCancelRef.current?.();
      newTaskHomeSessionLoadCancelRef.current = null;
      return;
    }

    if (isNewTaskHome && sidebarSessionsRef.current.length === 0) {
      newTaskHomeSessionLoadCancelRef.current?.();
      newTaskHomeSessionLoadCancelRef.current = scheduleMinimumDelayIdleTask(
        () => {
          newTaskHomeSessionLoadCancelRef.current = null;
          void loadRecentSidebarSessionsRef.current();
        },
        {
          minimumDelayMs: SIDEBAR_NEW_TASK_HOME_SESSION_LOAD_DEFER_MS,
          idleTimeoutMs: SIDEBAR_NEW_TASK_HOME_SESSION_LOAD_DEFER_MS,
        },
      );
      return () => {
        newTaskHomeSessionLoadCancelRef.current?.();
        newTaskHomeSessionLoadCancelRef.current = null;
      };
    }

    if (isClawTaskCenter && hasCachedCurrentSessionSidebarEntry) {
      return scheduleMinimumDelayIdleTask(
        () => {
          void loadRecentSidebarSessionsRef.current();
        },
        {
          minimumDelayMs: SIDEBAR_SESSION_ENTRY_REFRESH_DEFER_MS,
          idleTimeoutMs: SIDEBAR_SESSION_ENTRY_REFRESH_DEFER_MS,
        },
      );
    }

    void loadRecentSidebarSessionsRef.current();
  }, [
    hasCachedCurrentSessionSidebarEntry,
    isClawTaskCenter,
    isNewTaskHome,
    activeProjectIdsKey,
    openedProjectCwdsKey,
    shouldLoadSidebarConversations,
  ]);

  useEffect(() => {
    if (!shouldLoadSidebarConversations || typeof window === "undefined") {
      return;
    }

    const handleFocus = () => {
      sidebarFocusRefreshCancelRef.current?.();
      sidebarFocusRefreshCancelRef.current = scheduleMinimumDelayIdleTask(
        () => {
          sidebarFocusRefreshCancelRef.current = null;
          void refreshSidebarSessions();
        },
        {
          minimumDelayMs: SIDEBAR_SESSION_ENTRY_REFRESH_DEFER_MS,
          idleTimeoutMs: SIDEBAR_SESSION_ENTRY_REFRESH_DEFER_MS,
        },
      );
    };

    window.addEventListener("focus", handleFocus);

    return () => {
      sidebarFocusRefreshCancelRef.current?.();
      sidebarFocusRefreshCancelRef.current = null;
      window.removeEventListener("focus", handleFocus);
    };
  }, [refreshSidebarSessions, shouldLoadSidebarConversations]);

  useEffect(() => {
    if (!shouldLoadSidebarConversations || typeof window === "undefined") {
      return;
    }

    const handleSessionsChanged = (event: Event) => {
      const detail = event instanceof CustomEvent ? event.detail : undefined;
      const createdSessionId =
        detail?.reason === "created" && typeof detail.sessionId === "string"
          ? detail.sessionId.trim()
          : "";
      if (createdSessionId) {
        const now = Math.floor(Date.now() / 1000);
        const createdSessionName =
          typeof detail.name === "string" ? detail.name.trim() : "";
        const createdThreadId =
          typeof detail.threadId === "string" ? detail.threadId.trim() : "";
        const createdWorkingDir =
          typeof detail.workingDir === "string"
            ? detail.workingDir.trim()
            : "";
        const optimisticSession: AgentSessionInfo = {
          id: createdSessionId,
          name: createdSessionName || conversationUntitledLabel,
          thread_id: createdThreadId || undefined,
          created_at: now,
          updated_at: now,
          messages_count: 0,
          workspace_id:
            typeof detail.workspaceId === "string"
              ? detail.workspaceId.trim() || undefined
              : undefined,
          working_dir: createdWorkingDir || undefined,
        };
        optimisticSidebarSessionsRef.current.set(
          createdSessionId,
          optimisticSession,
        );
        setSidebarSessions((current) =>
          filterSidebarSessions([
            optimisticSession,
            ...current.filter((session) => session.id !== createdSessionId),
          ]),
        );
      }
      if (
        shouldDeferCurrentSessionMetadataRefresh({
          currentSessionId,
          detail,
        })
      ) {
        scheduleRecentSidebarReload(SIDEBAR_SESSION_ENTRY_REFRESH_DEFER_MS);
        return;
      }

      void refreshSidebarSessions();
    };

    window.addEventListener(
      AGENT_RUNTIME_SESSIONS_CHANGED_EVENT,
      handleSessionsChanged,
    );

    return () => {
      window.removeEventListener(
        AGENT_RUNTIME_SESSIONS_CHANGED_EVENT,
        handleSessionsChanged,
      );
    };
  }, [
    currentSessionId,
    conversationUntitledLabel,
    refreshSidebarSessions,
    scheduleRecentSidebarReload,
    shouldLoadSidebarConversations,
  ]);

  useEffect(() => {
    if (!shouldLoadSidebarConversations || !sidebarSessionsHasMore) {
      return;
    }

    if (recentSessionsVisibleCount < sidebarSessions.length) {
      return;
    }

    void loadRecentSidebarSessions();
  }, [
    loadRecentSidebarSessions,
    recentSessionsVisibleCount,
    shouldLoadSidebarConversations,
    sidebarSessions.length,
    sidebarSessionsHasMore,
  ]);

  const recentSidebarSessions = useMemo(() => {
    const activeProjectIdSet = new Set(normalizedActiveProjectIds);
    const projectCwdSet = new Set(normalizedOpenedProjectCwds);
    return sidebarSessions.filter((session) => {
      if (session.archived_at) {
        return false;
      }
      if (!requireOpenedProjectCwd) {
        return true;
      }

      const workspaceId = normalizeSessionProjectValue(session.workspace_id);
      const workingDir = normalizeSessionProjectValue(
        session.working_dir,
      )?.replace(/[\\/]+$/u, "");
      if (!workspaceId && !workingDir) {
        return true;
      }
      if (workspaceId && activeProjectIdSet.has(workspaceId)) {
        return true;
      }

      return Boolean(workingDir && projectCwdSet.has(workingDir));
    });
  }, [
    normalizedActiveProjectIds,
    normalizedOpenedProjectCwds,
    requireOpenedProjectCwd,
    sidebarSessions,
  ]);
  const visibleRecentSidebarSessions = useMemo(
    () =>
      buildVisibleSidebarSessions({
        sessions: recentSidebarSessions,
        currentSessionId,
        limit: recentSessionsVisibleCount,
      }),
    [currentSessionId, recentSessionsVisibleCount, recentSidebarSessions],
  );
  const hasMoreRecentSidebarSessions =
    sidebarSessionsHasMore ||
    recentSessionsVisibleCount < recentSidebarSessions.length;
  const normalizedSidebarSearchQuery = useMemo(
    () => normalizeSidebarSearchText(sidebarSearchQuery),
    [sidebarSearchQuery],
  );
  const sidebarSearchResultLimit = Math.max(
    recentSessionsVisibleCount,
    SIDEBAR_SEARCH_RESULT_LIMIT,
  );
  const sidebarSearchMatchedSessions = useMemo(() => {
    if (!normalizedSidebarSearchQuery) {
      return recentSidebarSessions;
    }

    return recentSidebarSessions.filter((session) =>
      matchesSidebarSessionTitle(
        session,
        normalizedSidebarSearchQuery,
        conversationUntitledLabel,
      ),
    );
  }, [
    conversationUntitledLabel,
    normalizedSidebarSearchQuery,
    recentSidebarSessions,
  ]);
  const sidebarSearchResultSessions = useMemo(() => {
    if (!normalizedSidebarSearchQuery) {
      return buildVisibleSidebarSessions({
        sessions: recentSidebarSessions,
        currentSessionId,
        limit: sidebarSearchResultLimit,
      });
    }

    return sidebarSearchMatchedSessions.slice(0, sidebarSearchResultLimit);
  }, [
    currentSessionId,
    normalizedSidebarSearchQuery,
    recentSidebarSessions,
    sidebarSearchMatchedSessions,
    sidebarSearchResultLimit,
  ]);
  const sidebarSearchHasQuery = normalizedSidebarSearchQuery.length > 0;
  const sidebarSearchHasMoreResults = sidebarSearchHasQuery
    ? sidebarSessionsHasMore ||
      sidebarSearchResultLimit < sidebarSearchMatchedSessions.length
    : hasMoreRecentSidebarSessions;
  const fallbackSessionId =
    recentSidebarSessions[0]?.id ?? sidebarSessions[0]?.id ?? null;

  const deferConversationNavigation = useCallback(() => {
    conversationNavigationDeferUntilRef.current =
      Date.now() + SIDEBAR_CONVERSATION_NAVIGATION_DEFER_MS;
  }, []);

  const showMoreRecentSessions = useCallback(() => {
    setRecentSessionsVisibleCount(
      (current) => current + SIDEBAR_RECENT_SESSION_PAGE_SIZE,
    );
  }, []);

  const beginSidebarSessionAction = useCallback((sessionId: string) => {
    setSidebarSessionActionId(sessionId);
  }, []);

  const clearSidebarSessionAction = useCallback((sessionId: string) => {
    setSidebarSessionActionId((current) =>
      current === sessionId ? null : current,
    );
  }, []);

  const renameSidebarSessionOptimistically = useCallback(
    (nextSession: AgentSessionInfo) => {
      if (optimisticSidebarSessionsRef.current.has(nextSession.id)) {
        optimisticSidebarSessionsRef.current.set(nextSession.id, nextSession);
      }
      setSidebarSessions((current) =>
        filterSidebarSessions(
          current.map((item) =>
            item.id === nextSession.id ? nextSession : item,
          ),
        ),
      );
    },
    [],
  );

  const moveSidebarSessionArchiveStateOptimistically = useCallback(
    (nextSession: AgentSessionInfo) => {
      if (optimisticSidebarSessionsRef.current.has(nextSession.id)) {
        optimisticSidebarSessionsRef.current.set(nextSession.id, nextSession);
      }
      setSidebarSessions((current) =>
        filterSidebarSessions(
          current
            .map((item) => (item.id === nextSession.id ? nextSession : item))
            .filter((item) => !item.archived_at),
        ),
      );
    },
    [],
  );

  const moveSidebarSessionSectionOptimistically = useCallback(
    (nextSession: AgentSessionInfo) => {
      if (optimisticSidebarSessionsRef.current.has(nextSession.id)) {
        optimisticSidebarSessionsRef.current.set(nextSession.id, nextSession);
      }
      setSidebarSessions((current) =>
        filterSidebarSessions(
          current.map((item) =>
            item.id === nextSession.id ? nextSession : item,
          ),
        ),
      );
    },
    [],
  );

  const removeSidebarSessionOptimistically = useCallback(
    (sessionId: string) => {
      optimisticSidebarSessionsRef.current.delete(sessionId);
      setSidebarSessions((current) =>
        current.filter((item) => item.id !== sessionId),
      );
    },
    [],
  );

  const addImportedSidebarSessionOptimistically = useCallback(
    (response: ConversationImportThreadCommitResponse) => {
      const importedSession = buildImportedSidebarSession(response);
      optimisticSidebarSessionsRef.current.set(
        importedSession.id,
        importedSession,
      );
      setSidebarSessions((current) =>
        filterSidebarSessions([
          importedSession,
          ...current.filter((item) => item.id !== importedSession.id),
        ]),
      );
    },
    [],
  );

  return {
    addImportedSidebarSessionOptimistically,
    beginSidebarSessionAction,
    clearSidebarSessionAction,
    deferConversationNavigation,
    fallbackSessionId,
    hasMoreRecentSidebarSessions,
    moveSidebarSessionArchiveStateOptimistically,
    moveSidebarSessionSectionOptimistically,
    recentSessionsLoading: sidebarSessionsLoading,
    refreshSidebarSessions,
    removeSidebarSessionOptimistically,
    renameSidebarSessionOptimistically,
    shouldShowSessionLoadingState,
    showMoreRecentSessions,
    sidebarSearchHasMoreResults,
    sidebarSearchHasQuery,
    sidebarSearchResultSessions,
    sidebarSessionActionId,
    visibleRecentSidebarSessions,
  };
}
