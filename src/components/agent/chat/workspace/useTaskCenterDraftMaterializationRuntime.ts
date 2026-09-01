import {
  startTransition,
  useCallback,
  useEffect,
  useRef,
  type Dispatch,
  type MutableRefObject,
  type SetStateAction,
} from "react";
import type { Character } from "@/lib/api/projectMemory";
import { logAgentDebug } from "@/lib/agentDebug";
import { recordAgentUiPerformanceMetric } from "@/lib/agentUiPerformanceMetrics";
import type { TaskCenterDraftSendRequest } from "../homePendingPreview";
import { rememberInitialSessionNavigationStart } from "./useWorkspaceInitialSessionNavigation";
import {
  createTaskCenterDraftTabId,
  TASK_CENTER_DRAFT_SESSION_WARMUP_DELAY_MS,
  type TaskCenterDraftTab,
} from "./agentChatWorkspaceHelpers";
import {
  buildTaskCenterDraftTab,
  clearActiveTaskCenterDraftTab,
  markTaskCenterDraftTabFailed,
  removeTaskCenterDraftTab,
  resolveActiveTaskCenterDraftTabId,
  shouldWarmupTaskCenterDraftSession,
  upsertTaskCenterDraftTab,
} from "./taskCenterDraftTabs";

type AgentEntry = "new-task" | "claw";

type CreateFreshSession = (
  sessionName?: string,
  options?: {
    activateSession?: boolean;
    preserveCurrentSnapshot?: boolean;
    skipSessionStartHooks?: boolean;
  },
) => Promise<string | null>;

type SwitchMaterializedSession = (
  sessionId: string,
  options?: { forceRefresh?: boolean; allowDetachedSession?: boolean },
) => Promise<unknown>;

type PersistMaterializedSessionNavigation = (sessionId: string) => void;

interface CommitMaterializedDraftOptions {
  embedHomeSession?: boolean;
  hydrateSession?: boolean;
  preserveInput?: boolean;
  syncRoute?: boolean;
}

interface UseTaskCenterDraftMaterializationRuntimeParams {
  activeTaskCenterDraftTabId: string | null;
  agentEntry: AgentEntry;
  clearMessages: (options?: { showToast?: boolean }) => void;
  createFreshSession: CreateFreshSession;
  input: string;
  isPreparingSend: boolean;
  isSending: boolean;
  markTaskCenterEmbeddedHomeSession: (topicId: string) => void;
  markTaskCenterLocalSessionOverride: (topicId: string) => void;
  resetLocalImageWorkbenchSessionScope: () => void;
  resetTopicLocalState: () => void;
  setActiveTaskCenterDraftTabId: Dispatch<SetStateAction<string | null>>;
  setHomePendingPreviewRequest: Dispatch<
    SetStateAction<TaskCenterDraftSendRequest | null>
  >;
  setInput: Dispatch<SetStateAction<string>>;
  setMentionedCharacters: Dispatch<SetStateAction<Character[]>>;
  setSelectedText: Dispatch<SetStateAction<string>>;
  setTaskCenterDetachedTopicId: Dispatch<SetStateAction<string | null>>;
  setTaskCenterDraftSendRequest: Dispatch<
    SetStateAction<TaskCenterDraftSendRequest | null>
  >;
  setTaskCenterDraftTabs: Dispatch<SetStateAction<TaskCenterDraftTab[]>>;
  setTaskCenterTransitionTopicId: Dispatch<SetStateAction<string | null>>;
  persistMaterializedSessionNavigation?: PersistMaterializedSessionNavigation;
  switchMaterializedSession?: SwitchMaterializedSession;
  taskCenterDraftSurfaceActiveRef: MutableRefObject<boolean>;
  taskCenterDraftTabs: TaskCenterDraftTab[];
  taskCenterWorkspaceId?: string | null;
  upsertTaskCenterOpenTab: (
    topicId: string,
    workspaceIdOverride?: string | null,
  ) => void;
}

export function useTaskCenterDraftMaterializationRuntime({
  activeTaskCenterDraftTabId,
  agentEntry,
  clearMessages,
  createFreshSession,
  input,
  isPreparingSend,
  isSending,
  markTaskCenterEmbeddedHomeSession,
  markTaskCenterLocalSessionOverride,
  resetLocalImageWorkbenchSessionScope,
  resetTopicLocalState,
  setActiveTaskCenterDraftTabId,
  setHomePendingPreviewRequest,
  setInput,
  setMentionedCharacters,
  setSelectedText,
  setTaskCenterDetachedTopicId,
  setTaskCenterDraftSendRequest,
  setTaskCenterDraftTabs,
  setTaskCenterTransitionTopicId,
  persistMaterializedSessionNavigation,
  switchMaterializedSession,
  taskCenterDraftSurfaceActiveRef,
  taskCenterDraftTabs,
  taskCenterWorkspaceId,
  upsertTaskCenterOpenTab,
}: UseTaskCenterDraftMaterializationRuntimeParams) {
  const taskCenterDraftTabsRef = useRef<TaskCenterDraftTab[]>([]);
  const activeTaskCenterDraftTabIdRef = useRef<string | null>(null);
  const taskCenterDraftMaterializePromisesRef = useRef<
    Map<string, Promise<string | null>>
  >(new Map());
  const taskCenterDraftMaterializedSessionIdsRef = useRef<Map<string, string>>(
    new Map(),
  );
  const taskCenterDraftWarmupSessionIdsRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    taskCenterDraftTabsRef.current = taskCenterDraftTabs;
  }, [taskCenterDraftTabs]);

  useEffect(() => {
    activeTaskCenterDraftTabIdRef.current = activeTaskCenterDraftTabId;
  }, [activeTaskCenterDraftTabId]);

  const finalizeFreshTaskCenterConversation = useCallback(
    (
      newSessionId: string,
      workspaceIdOverride?: string | null,
      options?: { embedHomeSession?: boolean; preserveInput?: boolean },
    ) => {
      resetTopicLocalState();
      if (options?.preserveInput !== true) {
        setInput("");
      }
      setSelectedText("");
      setMentionedCharacters([]);
      upsertTaskCenterOpenTab(newSessionId, workspaceIdOverride);
      if (options?.embedHomeSession !== false) {
        markTaskCenterEmbeddedHomeSession(newSessionId);
      }
      markTaskCenterLocalSessionOverride(newSessionId);
    },
    [
      markTaskCenterEmbeddedHomeSession,
      markTaskCenterLocalSessionOverride,
      resetTopicLocalState,
      setInput,
      setMentionedCharacters,
      setSelectedText,
      upsertTaskCenterOpenTab,
    ],
  );

  const openTaskCenterDraftTab = useCallback(
    (options?: { preservePendingSendRequest?: boolean }) => {
      const draftTab = buildTaskCenterDraftTab({
        id: createTaskCenterDraftTabId(),
      });

      taskCenterDraftSurfaceActiveRef.current = true;
      activeTaskCenterDraftTabIdRef.current = draftTab.id;
      taskCenterDraftTabsRef.current = upsertTaskCenterDraftTab(
        taskCenterDraftTabsRef.current,
        draftTab,
      );
      resetLocalImageWorkbenchSessionScope();
      clearMessages({ showToast: false });
      startTransition(() => {
        setTaskCenterTransitionTopicId(null);
        setTaskCenterDetachedTopicId(null);
        setActiveTaskCenterDraftTabId(draftTab.id);
        if (options?.preservePendingSendRequest !== true) {
          setTaskCenterDraftSendRequest(null);
          setHomePendingPreviewRequest(null);
        }
        setTaskCenterDraftTabs((current) =>
          upsertTaskCenterDraftTab(current, draftTab),
        );
        resetTopicLocalState();
        setInput("");
        setSelectedText("");
        setMentionedCharacters([]);
      });

      logAgentDebug("AgentChatPage", "taskCenter.draftTab.open", {
        draftTabId: draftTab.id,
        workspaceId: taskCenterWorkspaceId,
      });

      return draftTab.id;
    },
    [
      clearMessages,
      resetLocalImageWorkbenchSessionScope,
      resetTopicLocalState,
      setActiveTaskCenterDraftTabId,
      setHomePendingPreviewRequest,
      setInput,
      setMentionedCharacters,
      setSelectedText,
      setTaskCenterDetachedTopicId,
      setTaskCenterDraftSendRequest,
      setTaskCenterDraftTabs,
      setTaskCenterTransitionTopicId,
      taskCenterDraftSurfaceActiveRef,
      taskCenterWorkspaceId,
    ],
  );

  const commitMaterializedTaskCenterDraftTab = useCallback(
    (
      draftTabId: string,
      newSessionId: string,
      options?: CommitMaterializedDraftOptions,
    ) => {
      taskCenterDraftSurfaceActiveRef.current = false;
      taskCenterDraftWarmupSessionIdsRef.current.delete(draftTabId);
      startTransition(() => {
        setTaskCenterTransitionTopicId(null);
        setTaskCenterDraftTabs((current) =>
          removeTaskCenterDraftTab(current, draftTabId),
        );
        setActiveTaskCenterDraftTabId((current) =>
          clearActiveTaskCenterDraftTab(current, draftTabId),
        );
        finalizeFreshTaskCenterConversation(
          newSessionId,
          taskCenterWorkspaceId,
          options,
        );
      });
      if (options?.syncRoute !== false) {
        rememberInitialSessionNavigationStart(newSessionId);
        persistMaterializedSessionNavigation?.(newSessionId);
        if (switchMaterializedSession && options?.hydrateSession !== false) {
          void switchMaterializedSession(newSessionId, {
            allowDetachedSession: true,
            forceRefresh: true,
          }).catch((error) => {
            logAgentDebug(
              "AgentChatPage",
              "taskCenter.draftTab.commit.switchSessionError",
              {
                draftTabId,
                error,
                newSessionId,
                workspaceId: taskCenterWorkspaceId,
              },
              { level: "error" },
            );
          });
        }
      }
    },
    [
      finalizeFreshTaskCenterConversation,
      persistMaterializedSessionNavigation,
      setActiveTaskCenterDraftTabId,
      setTaskCenterDraftTabs,
      setTaskCenterTransitionTopicId,
      switchMaterializedSession,
      taskCenterDraftSurfaceActiveRef,
      taskCenterWorkspaceId,
    ],
  );

  const materializeTaskCenterDraftTab = useCallback(
    async (
      draftTabId: string,
      options?: {
        reason?: "send" | "input_warmup";
        commit?: boolean;
      },
    ): Promise<string | null> => {
      const reason = options?.reason ?? "send";
      const materializedSessionId =
        taskCenterDraftMaterializedSessionIdsRef.current.get(draftTabId);
      if (materializedSessionId) {
        return materializedSessionId;
      }

      const existingPromise =
        taskCenterDraftMaterializePromisesRef.current.get(draftTabId);
      if (existingPromise) {
        logAgentDebug(
          "AgentChatPage",
          "taskCenter.draftTab.materialize.reuse",
          {
            draftTabId,
            reason,
            workspaceId: taskCenterWorkspaceId,
          },
        );
        return existingPromise;
      }

      const draftExists = taskCenterDraftTabsRef.current.some(
        (tab) => tab.id === draftTabId,
      );
      if (!draftExists) {
        return null;
      }
      if (reason === "input_warmup") {
        taskCenterDraftWarmupSessionIdsRef.current.add(draftTabId);
      }

      const startedAt = Date.now();
      logAgentDebug("AgentChatPage", "taskCenter.draftTab.materialize.start", {
        draftTabId,
        reason,
        workspaceId: taskCenterWorkspaceId,
      });
      recordAgentUiPerformanceMetric("taskCenter.draftMaterialize.start", {
        sessionId: draftTabId,
        reason,
        workspaceId: taskCenterWorkspaceId,
      });

      const materializePromise = (async () => {
        const newSessionId = await createFreshSession("新对话", {
          activateSession: options?.commit !== false,
          preserveCurrentSnapshot: false,
          skipSessionStartHooks: true,
        });
        if (!newSessionId) {
          setTaskCenterDraftTabs((current) =>
            markTaskCenterDraftTabFailed(current, draftTabId),
          );
          recordAgentUiPerformanceMetric("taskCenter.draftMaterialize.error", {
            durationMs: Date.now() - startedAt,
            reason,
            sessionId: draftTabId,
            workspaceId: taskCenterWorkspaceId,
          });
          taskCenterDraftWarmupSessionIdsRef.current.delete(draftTabId);
          return null;
        }
        taskCenterDraftMaterializedSessionIdsRef.current.set(
          draftTabId,
          newSessionId,
        );

        if (options?.commit !== false) {
          commitMaterializedTaskCenterDraftTab(draftTabId, newSessionId, {
            preserveInput: true,
          });
        }

        logAgentDebug("AgentChatPage", "taskCenter.draftTab.materialize.done", {
          draftTabId,
          durationMs: Date.now() - startedAt,
          newSessionId,
          reason,
          workspaceId: taskCenterWorkspaceId,
        });
        recordAgentUiPerformanceMetric("taskCenter.draftMaterialize.success", {
          durationMs: Date.now() - startedAt,
          materializedSessionId: newSessionId,
          reason,
          sessionId: draftTabId,
          workspaceId: taskCenterWorkspaceId,
        });
        return newSessionId;
      })();

      const trackedPromise = materializePromise.finally(() => {
        if (
          taskCenterDraftMaterializePromisesRef.current.get(draftTabId) ===
          trackedPromise
        ) {
          taskCenterDraftMaterializePromisesRef.current.delete(draftTabId);
        }
      });
      taskCenterDraftMaterializePromisesRef.current.set(
        draftTabId,
        trackedPromise,
      );
      return trackedPromise;
    },
    [
      commitMaterializedTaskCenterDraftTab,
      createFreshSession,
      setTaskCenterDraftTabs,
      taskCenterWorkspaceId,
    ],
  );

  useEffect(() => {
    const draftTabId = resolveActiveTaskCenterDraftTabId({
      draftTabs: taskCenterDraftTabs,
      activeDraftTabId: activeTaskCenterDraftTabId,
    });
    if (
      !draftTabId ||
      !shouldWarmupTaskCenterDraftSession({
        agentEntry,
        activeDraftTabId: draftTabId,
        draftTabs: taskCenterDraftTabs,
        input,
        isPreparingSend,
        isSending,
      })
    ) {
      return;
    }

    const timer = window.setTimeout(() => {
      logAgentDebug("AgentChatPage", "taskCenter.draftTab.warmup.request", {
        draftTabId,
        inputLength: input.trim().length,
        workspaceId: taskCenterWorkspaceId,
      });
      void materializeTaskCenterDraftTab(draftTabId, {
        reason: "input_warmup",
        commit: false,
      }).catch((error) => {
        logAgentDebug(
          "AgentChatPage",
          "taskCenter.draftTab.warmup.error",
          {
            draftTabId,
            error,
            workspaceId: taskCenterWorkspaceId,
          },
          { level: "error" },
        );
      });
    }, TASK_CENTER_DRAFT_SESSION_WARMUP_DELAY_MS);

    return () => {
      window.clearTimeout(timer);
    };
  }, [
    activeTaskCenterDraftTabId,
    agentEntry,
    input,
    isPreparingSend,
    isSending,
    materializeTaskCenterDraftTab,
    taskCenterDraftTabs,
    taskCenterWorkspaceId,
  ]);

  return {
    activeTaskCenterDraftTabIdRef,
    commitMaterializedTaskCenterDraftTab,
    materializeTaskCenterDraftTab,
    openTaskCenterDraftTab,
    taskCenterDraftMaterializedSessionIdsRef,
    taskCenterDraftWarmupSessionIdsRef,
    taskCenterDraftTabsRef,
  };
}
