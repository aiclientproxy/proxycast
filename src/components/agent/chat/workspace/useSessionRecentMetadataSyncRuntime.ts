import { useCallback, useEffect, useMemo, useRef } from "react";
import { updateAgentRuntimeThreadToolPreferences } from "@/lib/api/agentRuntime/sessionClient";
import { logAgentDebug } from "@/lib/agentDebug";
import { scheduleMinimumDelayIdleTask } from "@/lib/utils/scheduleMinimumDelayIdleTask";
import {
  createSessionRecentPreferencesFromChatToolPreferences,
} from "../utils/sessionExecutionRuntime";
import {
  SESSION_RECENT_METADATA_BACKGROUND_SYNC_DELAY_MS,
  SESSION_RECENT_METADATA_BACKGROUND_SYNC_IDLE_TIMEOUT_MS,
  SESSION_RECENT_METADATA_NAVIGATION_DEFER_MS,
  mergeSessionRecentMetadataSyncPriority,
  type PendingSessionRecentMetadataSync,
  type SessionRecentMetadataPatch,
  type SessionRecentMetadataSyncOptions,
  type SessionRecentMetadataSyncPriority,
} from "./agentChatWorkspaceHelpers";

export function useSessionRecentMetadataSyncRuntime() {
  const activeSessionIdRef = useRef<string | null>(null);
  const navigationDeferUntilRef = useRef(0);
  const pendingSyncRef = useRef<Map<string, PendingSessionRecentMetadataSync>>(
    new Map(),
  );

  const deferSessionRecentMetadataSyncForNavigation = useCallback(
    (topicId: string) => {
      const deferUntil =
        Date.now() + SESSION_RECENT_METADATA_NAVIGATION_DEFER_MS;
      navigationDeferUntilRef.current = Math.max(
        navigationDeferUntilRef.current,
        deferUntil,
      );
      logAgentDebug("AgentChatPage", "sessionRecentMetadataSync.defer", {
        deferMs: SESSION_RECENT_METADATA_NAVIGATION_DEFER_MS,
        topicId,
      });
    },
    [],
  );

  const flushSessionRecentMetadataSync = useCallback(
    function runSessionRecentMetadataSyncFlush(sessionId: string) {
      const pending = pendingSyncRef.current.get(sessionId);
      if (!pending) {
        return;
      }

      pending.cancel?.();
      pending.cancel = null;

      if (pending.priority === "background") {
        const remainingNavigationDeferMs =
          navigationDeferUntilRef.current - Date.now();
        if (remainingNavigationDeferMs > 0) {
          pending.cancel = scheduleMinimumDelayIdleTask(
            () => {
              pending.cancel = null;
              runSessionRecentMetadataSyncFlush(sessionId);
            },
            {
              minimumDelayMs: remainingNavigationDeferMs,
              idleTimeoutMs:
                SESSION_RECENT_METADATA_BACKGROUND_SYNC_IDLE_TIMEOUT_MS,
            },
          );
          logAgentDebug(
            "AgentChatPage",
            "sessionRecentMetadataSync.deferredForNavigation",
            {
              deferMs: remainingNavigationDeferMs,
              sessionId,
            },
            {
              dedupeKey: `sessionRecentMetadataSync.deferredForNavigation:${sessionId}`,
              throttleMs: 1000,
            },
          );
          return;
        }
      }

      pendingSyncRef.current.delete(sessionId);

      if (
        pending.priority === "background" &&
        activeSessionIdRef.current !== sessionId
      ) {
        pending.resolvers.forEach((resolve) => resolve());
        return;
      }

      void updateAgentRuntimeThreadToolPreferences(sessionId, pending.patch)
        .then(() => {
          pending.resolvers.forEach((resolve) => resolve());
        })
        .catch((error) => {
          pending.rejecters.forEach((reject) => reject(error));
        });
    },
    [],
  );

  const scheduleSessionRecentMetadataSync = useCallback(
    (sessionId: string, priority: SessionRecentMetadataSyncPriority) => {
      const pending = pendingSyncRef.current.get(sessionId);
      if (!pending) {
        return;
      }

      pending.cancel?.();
      pending.cancel =
        priority === "background"
          ? scheduleMinimumDelayIdleTask(
              () => {
                pending.cancel = null;
                flushSessionRecentMetadataSync(sessionId);
              },
              {
                minimumDelayMs:
                  SESSION_RECENT_METADATA_BACKGROUND_SYNC_DELAY_MS,
                idleTimeoutMs:
                  SESSION_RECENT_METADATA_BACKGROUND_SYNC_IDLE_TIMEOUT_MS,
              },
            )
          : scheduleMinimumDelayIdleTask(
              () => {
                pending.cancel = null;
                flushSessionRecentMetadataSync(sessionId);
              },
              {
                minimumDelayMs: 0,
                idleTimeoutMs: 500,
              },
            );
    },
    [flushSessionRecentMetadataSync],
  );

  const enqueueSessionRecentMetadataSync = useCallback(
    (
      sessionId: string,
      patch: SessionRecentMetadataPatch,
      options?: SessionRecentMetadataSyncOptions,
    ): Promise<void> => {
      const trimmedSessionId = sessionId.trim();
      if (!trimmedSessionId) {
        return Promise.resolve();
      }

      const requestedPriority = options?.priority ?? "immediate";

      return new Promise<void>((resolve, reject) => {
        const pending = pendingSyncRef.current.get(trimmedSessionId);
        if (pending) {
          const previousPriority = pending.priority;
          pending.patch = {
            ...pending.patch,
            ...patch,
          };
          pending.priority = mergeSessionRecentMetadataSyncPriority(
            pending.priority,
            requestedPriority,
          );
          pending.resolvers.push(resolve);
          pending.rejecters.push(reject);
          if (
            previousPriority === "background" &&
            pending.priority === "immediate"
          ) {
            scheduleSessionRecentMetadataSync(trimmedSessionId, "immediate");
          }
          return;
        }

        pendingSyncRef.current.set(trimmedSessionId, {
          patch,
          priority: requestedPriority,
          cancel: null,
          resolvers: [resolve],
          rejecters: [reject],
        });
        scheduleSessionRecentMetadataSync(trimmedSessionId, requestedPriority);
      });
    },
    [scheduleSessionRecentMetadataSync],
  );

  useEffect(() => {
    const pendingSyncMap = pendingSyncRef.current;
    return () => {
      const pendingSyncs = pendingSyncMap.values();
      for (const pending of pendingSyncs) {
        pending.cancel?.();
        pending.resolvers.forEach((resolve) => resolve());
      }
      pendingSyncMap.clear();
    };
  }, []);

  const syncSessionRecentPreferences = useCallback(
    async (
      sessionId: string,
      preferences: Parameters<
        typeof createSessionRecentPreferencesFromChatToolPreferences
      >[0],
      options?: SessionRecentMetadataSyncOptions,
    ) => {
      await enqueueSessionRecentMetadataSync(
        sessionId,
        createSessionRecentPreferencesFromChatToolPreferences(preferences),
        options,
      );
    },
    [enqueueSessionRecentMetadataSync],
  );

  const chatToolPreferenceSessionSync = useMemo(
    () => ({
      getSessionId: () => activeSessionIdRef.current,
      setSessionRecentPreferences: syncSessionRecentPreferences,
    }),
    [syncSessionRecentPreferences],
  );

  return {
    activeSessionIdRef,
    chatToolPreferenceSessionSync,
    deferSessionRecentMetadataSyncForNavigation,
    syncSessionRecentPreferences,
  };
}
