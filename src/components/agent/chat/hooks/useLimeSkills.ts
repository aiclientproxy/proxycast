import { useCallback, useEffect, useRef, useState } from "react";
import { skillsChangedServerNotification } from "@limecloud/app-server-client";
import { logAgentDebug } from "@/lib/agentDebug";
import { subscribeAppServerNotifications } from "@/lib/api/appServerEventBus";
import { skillsApi, type Skill } from "@/lib/api/skills";
import { scheduleMinimumDelayIdleTask } from "@/lib/utils/scheduleMinimumDelayIdleTask";

const SKILLS_IDLE_TIMEOUT_MS = 1_500;

interface UseLimeSkillsOptions {
  autoLoad?: "immediate" | "deferred" | false;
  deferredDelayMs?: number;
  logScope?: string;
  onError?: (error: unknown) => void;
}

export function useLimeSkills(options: UseLimeSkillsOptions = {}) {
  const {
    autoLoad = "immediate",
    deferredDelayMs = SKILLS_IDLE_TIMEOUT_MS,
    logScope = "useLimeSkills",
    onError,
  } = options;
  const [skills, setSkills] = useState<Skill[]>([]);
  const [skillsLoading, setSkillsLoading] = useState(false);
  const latestRequestIdRef = useRef(0);
  const logScopeRef = useRef(logScope);
  const onErrorRef = useRef(onError);

  useEffect(() => {
    logScopeRef.current = logScope;
  }, [logScope]);

  useEffect(() => {
    onErrorRef.current = onError;
  }, [onError]);

  const refreshSkills = useCallback(
    async (includeRemote = false): Promise<Skill[]> => {
      const requestId = latestRequestIdRef.current + 1;
      latestRequestIdRef.current = requestId;

      const startedAt = Date.now();
      logAgentDebug(logScopeRef.current, "loadSkills.start", {
        includeRemote,
      });
      setSkillsLoading(true);

      try {
        const loadedSkills = includeRemote
          ? await skillsApi.getAll("lime")
          : await skillsApi.getRuntimeCatalog();

        if (latestRequestIdRef.current !== requestId) {
          return loadedSkills;
        }

        setSkills(loadedSkills);
        logAgentDebug(logScopeRef.current, "loadSkills.success", {
          durationMs: Date.now() - startedAt,
          includeRemote,
          skillsCount: loadedSkills.length,
        });
        return loadedSkills;
      } catch (error) {
        if (latestRequestIdRef.current !== requestId) {
          return [];
        }

        setSkills([]);
        onErrorRef.current?.(error);
        logAgentDebug(
          logScopeRef.current,
          "loadSkills.error",
          {
            durationMs: Date.now() - startedAt,
            error,
            includeRemote,
          },
          { level: "warn" },
        );
        return [];
      } finally {
        if (latestRequestIdRef.current === requestId) {
          setSkillsLoading(false);
        }
      }
    },
    [],
  );

  useEffect(() => {
    if (autoLoad === false) {
      return;
    }

    if (autoLoad === "deferred") {
      return scheduleMinimumDelayIdleTask(
        () => {
          void refreshSkills(false);
        },
        {
          minimumDelayMs: deferredDelayMs,
          idleTimeoutMs: SKILLS_IDLE_TIMEOUT_MS,
        },
      );
    }

    void refreshSkills(false);
    return;
  }, [autoLoad, deferredDelayMs, refreshSkills]);

  useEffect(() => {
    return subscribeAppServerNotifications({
      onNotifications: (notifications) => {
        const changedCount = notifications.filter(
          (notification) =>
            skillsChangedServerNotification(notification) !== undefined,
        ).length;
        if (changedCount > 0) {
          logAgentDebug(
            logScopeRef.current,
            "skillsChanged.received",
            { method: "skills/changed", count: changedCount },
            { consoleOnly: true },
          );
          void refreshSkills(false);
        }
      },
      onError: (error) => {
        logAgentDebug(
          logScopeRef.current,
          "skillsChanged.subscription.error",
          { error },
          { level: "warn" },
        );
      },
    });
  }, [refreshSkills]);

  useEffect(() => {
    return () => {
      latestRequestIdRef.current += 1;
    };
  }, []);

  return {
    skills,
    skillsLoading,
    refreshSkills,
  };
}
