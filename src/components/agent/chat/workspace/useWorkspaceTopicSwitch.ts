import { useCallback, useEffect } from "react";
import { toast } from "sonner";
import { logAgentDebug } from "@/lib/agentDebug";
import { getProject } from "@/lib/api/project";
import { normalizeProjectId } from "../utils/topicProjectResolution";
import { resolveTopicSwitchProject } from "../utils/topicProjectSwitch";

interface PendingTopicSwitchState {
  topicId: string;
  targetProjectId: string;
  forceRefresh?: boolean;
  resumeSessionStartHooks?: boolean;
  allowDetachedSession?: boolean;
}

interface TopicSwitchOptions {
  forceRefresh?: boolean;
  resumeSessionStartHooks?: boolean;
  allowDetachedSession?: boolean;
}

export type WorkspaceTopicSwitchResult =
  | "success"
  | "deferred"
  | "blocked"
  | "missing"
  | "busy"
  | "error";

interface UseWorkspaceTopicSwitchParams {
  projectId?: string;
  externalProjectId?: string | null;
  originalSwitchTopic: (
    topicId: string,
    options?: TopicSwitchOptions,
  ) => Promise<unknown>;
  onBeforeTopicSwitch?: (topicId: string) => void;
  startTopicProjectResolution: () => boolean;
  finishTopicProjectResolution: () => void;
  deferTopicSwitch: (
    topicId: string,
    targetProjectId: string,
    options?: TopicSwitchOptions,
  ) => void;
  consumePendingTopicSwitch: (
    currentProjectId?: string | null,
  ) => PendingTopicSwitchState | null;
  rememberProjectId: (nextProjectId?: string | null) => void;
  getRememberedProjectId: () => string | null;
  loadTopicBoundProjectId: (topicId: string) => string | null;
  resetTopicLocalState: () => void;
}

export function useWorkspaceTopicSwitch({
  projectId,
  externalProjectId,
  originalSwitchTopic,
  onBeforeTopicSwitch,
  startTopicProjectResolution,
  finishTopicProjectResolution,
  deferTopicSwitch,
  consumePendingTopicSwitch,
  rememberProjectId,
  getRememberedProjectId,
  loadTopicBoundProjectId,
  resetTopicLocalState,
}: UseWorkspaceTopicSwitchParams) {
  const runTopicSwitch = useCallback(
    async (
      topicId: string,
      options?: TopicSwitchOptions,
      runOptions?: { skipBeforeSwitch?: boolean },
    ) => {
      const forwardedOptions =
        options?.forceRefresh === true ||
        options?.resumeSessionStartHooks === true ||
        options?.allowDetachedSession === true
          ? {
              ...(options?.forceRefresh === true ? { forceRefresh: true } : {}),
              ...(options?.resumeSessionStartHooks === true
                ? { resumeSessionStartHooks: true }
                : {}),
              ...(options?.allowDetachedSession === true
                ? { allowDetachedSession: true }
                : {}),
            }
          : undefined;
      const startedAt = Date.now();
      logAgentDebug("AgentChatPage", "runTopicSwitch.start", {
        allowDetachedSession: options?.allowDetachedSession === true,
        currentProjectId: projectId ?? null,
        forceRefresh: options?.forceRefresh === true,
        topicId,
      });
      if (runOptions?.skipBeforeSwitch !== true) {
        onBeforeTopicSwitch?.(topicId);
      }
      resetTopicLocalState();
      try {
        const hydrationResult = forwardedOptions
          ? await originalSwitchTopic(topicId, forwardedOptions)
          : await originalSwitchTopic(topicId);
        if (hydrationResult === "error") {
          logAgentDebug(
            "AgentChatPage",
            "runTopicSwitch.error",
            {
              allowDetachedSession: options?.allowDetachedSession === true,
              durationMs: Date.now() - startedAt,
              forceRefresh: options?.forceRefresh === true,
              hydrationResult,
              topicId,
            },
            { level: "error" },
          );
          return "error" as const;
        }
        logAgentDebug("AgentChatPage", "runTopicSwitch.success", {
          allowDetachedSession: options?.allowDetachedSession === true,
          durationMs: Date.now() - startedAt,
          forceRefresh: options?.forceRefresh === true,
          topicId,
        });
        return "success" as const;
      } catch (error) {
        logAgentDebug(
          "AgentChatPage",
          "runTopicSwitch.error",
          {
            allowDetachedSession: options?.allowDetachedSession === true,
            durationMs: Date.now() - startedAt,
            error,
            forceRefresh: options?.forceRefresh === true,
            topicId,
          },
          { level: "error" },
        );
        throw error;
      }
    },
    [onBeforeTopicSwitch, originalSwitchTopic, projectId, resetTopicLocalState],
  );

  const switchTopic = useCallback(
    async (topicId: string, options?: TopicSwitchOptions) => {
      if (!startTopicProjectResolution()) {
        logAgentDebug(
          "AgentChatPage",
          "switchTopic.skipWhileResolving",
          {
            allowDetachedSession: options?.allowDetachedSession === true,
            forceRefresh: options?.forceRefresh === true,
            topicId,
          },
          { level: "warn", throttleMs: 1000 },
        );
        return "busy" as const;
      }

      let resolutionActive = true;
      const finishResolutionIfNeeded = () => {
        if (!resolutionActive) {
          return;
        }
        resolutionActive = false;
        finishTopicProjectResolution();
      };

      try {
        onBeforeTopicSwitch?.(topicId);
        const currentProjectId = normalizeProjectId(projectId);
        const topicBoundProjectId = normalizeProjectId(
          loadTopicBoundProjectId(topicId),
        );
        logAgentDebug("AgentChatPage", "switchTopic.start", {
          allowDetachedSession: options?.allowDetachedSession === true,
          currentProjectId: currentProjectId ?? null,
          externalProjectId: externalProjectId ?? null,
          forceRefresh: options?.forceRefresh === true,
          topicId,
        });

        if (
          options?.allowDetachedSession === true &&
          !currentProjectId &&
          !topicBoundProjectId
        ) {
          logAgentDebug(
            "AgentChatPage",
            "switchTopic.detachedSessionFastPath",
            {
              topicId,
            },
          );
          finishResolutionIfNeeded();
          return await runTopicSwitch(topicId, options, {
            skipBeforeSwitch: true,
          });
        }

        if (
          currentProjectId &&
          (!topicBoundProjectId || topicBoundProjectId === currentProjectId)
        ) {
          rememberProjectId(currentProjectId);
          logAgentDebug("AgentChatPage", "switchTopic.fastPathCurrentProject", {
            allowDetachedSession: options?.allowDetachedSession === true,
            currentProjectId,
            externalProjectId: externalProjectId ?? null,
            forceRefresh: options?.forceRefresh === true,
            topicBoundProjectId: topicBoundProjectId ?? null,
            topicId,
          });
          finishResolutionIfNeeded();
          return await runTopicSwitch(topicId, options, {
            skipBeforeSwitch: true,
          });
        }

        const decision = await resolveTopicSwitchProject({
          lockedProjectId: externalProjectId ?? null,
          topicBoundProjectId,
          lastProjectId: getRememberedProjectId(),
          loadProjectById: async (candidateProjectId) => {
            const project = await getProject(candidateProjectId);
            return project
              ? { id: project.id, isArchived: project.isArchived }
              : null;
          },
        });
        logAgentDebug("AgentChatPage", "switchTopic.decision", {
          decisionStatus: decision.status,
          projectId: decision.status === "ok" ? decision.projectId : null,
          topicId,
        });

        if (decision.status === "blocked") {
          return "blocked" as const;
        }

        if (decision.status === "missing") {
          toast.error("未找到可用项目，请先创建项目");
          return "missing" as const;
        }

        const targetProjectId = decision.projectId;
        if (currentProjectId !== targetProjectId) {
          deferTopicSwitch(topicId, targetProjectId, options);
          logAgentDebug("AgentChatPage", "switchTopic.deferUntilProjectReady", {
            allowDetachedSession: options?.allowDetachedSession === true,
            currentProjectId,
            forceRefresh: options?.forceRefresh === true,
            targetProjectId,
            topicId,
          });
          return "deferred" as const;
        }

        rememberProjectId(targetProjectId);
        finishResolutionIfNeeded();
        return await runTopicSwitch(topicId, options, {
          skipBeforeSwitch: true,
        });
      } catch (error) {
        console.error("[AgentChatPage] 解析任务项目失败:", error);
        logAgentDebug(
          "AgentChatPage",
          "switchTopic.error",
          {
            allowDetachedSession: options?.allowDetachedSession === true,
            error,
            forceRefresh: options?.forceRefresh === true,
            projectId: projectId ?? null,
            topicId,
          },
          { level: "error" },
        );
        toast.error("切换会话失败，请稍后重试");
        return "error" as const;
      } finally {
        finishResolutionIfNeeded();
      }
    },
    [
      deferTopicSwitch,
      externalProjectId,
      finishTopicProjectResolution,
      getRememberedProjectId,
      loadTopicBoundProjectId,
      onBeforeTopicSwitch,
      projectId,
      rememberProjectId,
      runTopicSwitch,
      startTopicProjectResolution,
    ],
  );

  useEffect(() => {
    const pending = consumePendingTopicSwitch(projectId);
    if (!pending) {
      return;
    }

    const currentProjectId = normalizeProjectId(projectId);
    logAgentDebug("AgentChatPage", "switchTopic.resumePending", {
      allowDetachedSession: pending.allowDetachedSession === true,
      forceRefresh: pending.forceRefresh === true,
      resumeSessionStartHooks: pending.resumeSessionStartHooks === true,
      projectId: currentProjectId,
      topicId: pending.topicId,
    });
    runTopicSwitch(pending.topicId, {
      allowDetachedSession: pending.allowDetachedSession === true,
      forceRefresh: pending.forceRefresh === true,
      resumeSessionStartHooks: pending.resumeSessionStartHooks === true,
    }).catch((error) => {
      console.error("[AgentChatPage] 执行待切换任务失败:", error);
      logAgentDebug(
        "AgentChatPage",
        "switchTopic.resumePendingError",
        {
          allowDetachedSession: pending.allowDetachedSession === true,
          error,
          forceRefresh: pending.forceRefresh === true,
          resumeSessionStartHooks: pending.resumeSessionStartHooks === true,
          projectId: currentProjectId,
          topicId: pending.topicId,
        },
        { level: "error" },
      );
      toast.error("加载会话失败，请重试");
    });
  }, [consumePendingTopicSwitch, projectId, runTopicSwitch]);

  return {
    runTopicSwitch,
    switchTopic,
  };
}
