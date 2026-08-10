import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type MutableRefObject,
} from "react";
import { toast } from "sonner";
import { getLimeI18n } from "@/i18n/createI18n";
import { updateProject } from "@/lib/api/project";
import type { AgentExecutionStrategy } from "@/lib/api/agentExecutionRuntime";
import type { ModelReasoningEffortLevel } from "@/lib/types/modelRegistry";
import { notifyProjectRuntimeAgentsGuide } from "@/components/workspace/services/runtimeAgentsGuideService";
import type {
  SendMessageFn,
  SessionModelPreference,
  WorkspacePathMissingState,
} from "./agentChatShared";
import {
  loadStoredSessionWorkspaceIdRaw,
  savePersistedSessionWorkspaceId,
} from "./agentProjectStorage";
import {
  DEFAULT_AGENT_ACCESS_MODE,
  getAgentPreferenceKeys,
  getAccessModeStorageKey,
  getExecutionStrategyStorageKey,
  getSessionAccessModeKey,
  getSessionModelPreferenceKey,
  loadPersisted,
  normalizeAccessMode,
  resolvePersistedAccessMode,
  resolvePersistedExecutionStrategy,
  resolveWorkspaceAgentPreferences,
  savePersisted,
  type AgentAccessMode,
} from "./agentChatStorage";
import {
  isLegacyDefaultProjectId,
  normalizeProjectId,
} from "../utils/topicProjectResolution";
import { normalizeExecutionStrategy } from "./agentChatCoreUtils";
import { normalizeChatSessionModelPreference } from "../utils/sessionExecutionRuntime";

interface UseAgentContextOptions {
  workspaceId: string;
  sessionIdRef: MutableRefObject<string | null>;
  pendingSessionMetadataSyncCancelRef?: MutableRefObject<(() => void) | null>;
  topicsUpdaterRef: MutableRefObject<
    | ((sessionId: string, executionStrategy: AgentExecutionStrategy) => void)
    | null
  >;
  sendMessageRef: MutableRefObject<SendMessageFn | null>;
  runtime: {
    setSessionExecutionStrategy: (
      sessionId: string,
      executionStrategy: AgentExecutionStrategy,
    ) => Promise<void>;
    setSessionAccessMode?: (
      sessionId: string,
      accessMode: AgentAccessMode,
    ) => Promise<void>;
    setSessionProviderSelection: (
      sessionId: string,
      providerType: string,
      model: string,
      reasoningEffort?: string,
    ) => Promise<void>;
  };
}

interface PendingModelSelection extends SessionModelPreference {
  reasoningEffort: ModelReasoningEffortLevel | "";
}

function modelSettingsUpdateErrorMessage(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  return String(
    getLimeI18n().t("agentChat.modelSettings.updateFailed", {
      ns: "agent",
      message,
    }),
  );
}

export function useAgentContext(options: UseAgentContextOptions) {
  const {
    workspaceId,
    sessionIdRef,
    pendingSessionMetadataSyncCancelRef,
    topicsUpdaterRef,
    sendMessageRef,
    runtime,
  } = options;

  const getWorkspaceIdForSubmit = useCallback((): string | undefined => {
    const resolvedWorkspaceId = workspaceId?.trim();
    return resolvedWorkspaceId || undefined;
  }, [workspaceId]);

  const initialPreferencesRef = useRef(
    resolveWorkspaceAgentPreferences(workspaceId),
  );
  const executionStrategyWorkspaceRef = useRef(workspaceId?.trim() ?? "");
  const [providerType, setProviderTypeState] = useState(
    () => initialPreferencesRef.current.providerType,
  );
  const [model, setModelState] = useState(
    () => initialPreferencesRef.current.model,
  );
  const [reasoningEffort, setReasoningEffortState] = useState<
    ModelReasoningEffortLevel | ""
  >("");
  const [executionStrategy, setExecutionStrategyState] =
    useState<AgentExecutionStrategy>(() =>
      resolvePersistedExecutionStrategy(workspaceId),
    );
  const [accessMode, setAccessModeState] = useState<AgentAccessMode>(() =>
    resolvePersistedAccessMode(workspaceId),
  );
  const [workspacePathMissing, setWorkspacePathMissing] =
    useState<WorkspacePathMissingState | null>(null);
  const resolvedExecutionStrategyWorkspace = workspaceId?.trim() ?? "";
  const effectiveExecutionStrategy =
    executionStrategyWorkspaceRef.current === resolvedExecutionStrategyWorkspace
      ? executionStrategy
      : resolvePersistedExecutionStrategy(workspaceId);

  const providerTypeRef = useRef(providerType);
  const modelRef = useRef(model);
  const reasoningEffortRef = useRef<string>(reasoningEffort);
  const scopedProviderPrefKeyRef = useRef<string>(
    getAgentPreferenceKeys(workspaceId).providerKey,
  );
  const scopedModelPrefKeyRef = useRef<string>(
    getAgentPreferenceKeys(workspaceId).modelKey,
  );
  const pendingSessionProviderSelectionSyncRef = useRef(
    new Map<string, PendingModelSelection>(),
  );
  const activeSessionProviderSelectionSyncRef = useRef(new Set<string>());
  const sessionProviderSelectionSyncPromiseRef = useRef(
    new Map<string, Promise<void>>(),
  );
  const syncedSessionProviderSelectionRef = useRef(
    new Map<string, SessionModelPreference>(),
  );
  const pendingSessionExecutionStrategySyncRef = useRef(
    new Map<string, AgentExecutionStrategy>(),
  );
  const syncedSessionExecutionStrategyRef = useRef(
    new Map<string, AgentExecutionStrategy>(),
  );
  const pendingSessionAccessModeSyncRef = useRef(
    new Map<string, AgentAccessMode>(),
  );

  providerTypeRef.current = providerType;
  modelRef.current = model;
  reasoningEffortRef.current = reasoningEffort;

  const persistSessionModelPreference = useCallback(
    (
      targetSessionId: string,
      targetProviderType: string,
      targetModel: string,
    ) => {
      savePersisted(
        getSessionModelPreferenceKey(workspaceId, targetSessionId),
        {
          providerType: targetProviderType,
          model: targetModel,
        },
      );
    },
    [workspaceId],
  );

  const loadSessionModelPreference = useCallback(
    (sessionId: string): SessionModelPreference | null => {
      const key = getSessionModelPreferenceKey(workspaceId, sessionId);
      const parsed = loadPersisted<SessionModelPreference | null>(key, null);
      if (!parsed) {
        return null;
      }
      if (
        typeof parsed.providerType !== "string" ||
        typeof parsed.model !== "string"
      ) {
        return null;
      }
      return parsed;
    },
    [workspaceId],
  );

  const markSessionModelPreferenceSynced = useCallback(
    (
      targetSessionId: string,
      targetProviderType: string,
      targetModel: string,
    ) => {
      const trimmedSessionId = targetSessionId.trim();
      if (!trimmedSessionId) {
        return;
      }
      syncedSessionProviderSelectionRef.current.set(trimmedSessionId, {
        providerType: targetProviderType,
        model: targetModel,
      });
    },
    [],
  );

  const clearSessionModelPreferenceSynced = useCallback(
    (targetSessionId: string) => {
      const trimmedSessionId = targetSessionId.trim();
      if (!trimmedSessionId) {
        return;
      }
      syncedSessionProviderSelectionRef.current.delete(trimmedSessionId);
    },
    [],
  );

  const getSyncedSessionModelPreference = useCallback(
    (targetSessionId: string): SessionModelPreference | null => {
      const trimmedSessionId = targetSessionId.trim();
      if (!trimmedSessionId) {
        return null;
      }
      return (
        syncedSessionProviderSelectionRef.current.get(trimmedSessionId) || null
      );
    },
    [],
  );

  const markSessionExecutionStrategySynced = useCallback(
    (
      targetSessionId: string,
      targetExecutionStrategy: AgentExecutionStrategy,
    ) => {
      const trimmedSessionId = targetSessionId.trim();
      if (!trimmedSessionId) {
        return;
      }
      syncedSessionExecutionStrategyRef.current.set(
        trimmedSessionId,
        normalizeExecutionStrategy(targetExecutionStrategy),
      );
    },
    [],
  );

  const persistSessionAccessMode = useCallback(
    (targetSessionId: string, targetAccessMode: AgentAccessMode) => {
      savePersisted(
        getSessionAccessModeKey(workspaceId, targetSessionId),
        normalizeAccessMode(targetAccessMode),
      );
    },
    [workspaceId],
  );

  const loadSessionAccessMode = useCallback(
    (targetSessionId: string): AgentAccessMode | null => {
      const parsed = loadPersisted<string | null>(
        getSessionAccessModeKey(workspaceId, targetSessionId),
        null,
      );
      if (parsed === null) {
        return null;
      }
      return normalizeAccessMode(parsed);
    },
    [workspaceId],
  );

  const getSyncedSessionExecutionStrategy = useCallback(
    (targetSessionId: string): AgentExecutionStrategy | null => {
      const trimmedSessionId = targetSessionId.trim();
      if (!trimmedSessionId) {
        return null;
      }
      return (
        syncedSessionExecutionStrategyRef.current.get(trimmedSessionId) || null
      );
    },
    [],
  );

  const scheduleSessionProviderSelectionSync = useCallback(
    (
      targetSessionId: string,
      targetProviderType: string,
      targetModel: string,
      targetReasoningEffort: ModelReasoningEffortLevel | "",
    ) => {
      const cancelPendingSessionMetadataSync =
        pendingSessionMetadataSyncCancelRef?.current;
      if (cancelPendingSessionMetadataSync) {
        cancelPendingSessionMetadataSync();
        pendingSessionMetadataSyncCancelRef.current = null;
      }
      const syncSelection = runtime.setSessionProviderSelection;
      const trimmedSessionId = targetSessionId.trim();
      if (!trimmedSessionId) {
        return;
      }

      const pending = pendingSessionProviderSelectionSyncRef.current;
      pending.set(trimmedSessionId, {
        providerType: targetProviderType,
        model: targetModel,
        reasoningEffort: targetReasoningEffort,
      });
      const active = activeSessionProviderSelectionSyncRef.current;
      if (
        active.has(trimmedSessionId) ||
        sessionProviderSelectionSyncPromiseRef.current.has(trimmedSessionId)
      ) {
        return;
      }

      let resolveSync: (() => void) | undefined;
      const syncPromise = new Promise<void>((resolve) => {
        resolveSync = resolve;
      });
      sessionProviderSelectionSyncPromiseRef.current.set(
        trimmedSessionId,
        syncPromise,
      );

      queueMicrotask(() => {
        if (active.has(trimmedSessionId)) {
          return;
        }
        active.add(trimmedSessionId);

        void (async () => {
          try {
            while (true) {
              const latestPreference = pending.get(trimmedSessionId);
              if (!latestPreference) {
                return;
              }
              pending.delete(trimmedSessionId);

              try {
                await syncSelection(
                  trimmedSessionId,
                  latestPreference.providerType,
                  latestPreference.model,
                  latestPreference.reasoningEffort,
                );
              } catch (error) {
                if (!pending.has(trimmedSessionId)) {
                  console.warn(
                    "[AgentChat] 更新 thread model settings 失败:",
                    error,
                  );
                  toast.error(modelSettingsUpdateErrorMessage(error));
                }
                continue;
              }

              persistSessionModelPreference(
                trimmedSessionId,
                latestPreference.providerType,
                latestPreference.model,
              );
              if (sessionIdRef.current?.trim() === trimmedSessionId) {
                providerTypeRef.current = latestPreference.providerType;
                modelRef.current = latestPreference.model;
                reasoningEffortRef.current = latestPreference.reasoningEffort;
                setProviderTypeState(latestPreference.providerType);
                setModelState(latestPreference.model);
                setReasoningEffortState(latestPreference.reasoningEffort);
                savePersisted(
                  scopedProviderPrefKeyRef.current,
                  latestPreference.providerType,
                );
                savePersisted(
                  scopedModelPrefKeyRef.current,
                  latestPreference.model,
                );
              }
              markSessionModelPreferenceSynced(
                trimmedSessionId,
                latestPreference.providerType,
                latestPreference.model,
              );
              if (pending.has(trimmedSessionId)) {
                continue;
              }
            }
          } finally {
            active.delete(trimmedSessionId);
            if (
              sessionProviderSelectionSyncPromiseRef.current.get(
                trimmedSessionId,
              ) === syncPromise
            ) {
              sessionProviderSelectionSyncPromiseRef.current.delete(
                trimmedSessionId,
              );
            }
            resolveSync?.();
          }
        })();
      });
    },
    [
      markSessionModelPreferenceSynced,
      pendingSessionMetadataSyncCancelRef,
      persistSessionModelPreference,
      runtime.setSessionProviderSelection,
      sessionIdRef,
    ],
  );

  const waitForSessionProviderSelectionSync = useCallback(
    (targetSessionId: string): Promise<void> => {
      const trimmedSessionId = targetSessionId.trim();
      if (!trimmedSessionId) {
        return Promise.resolve();
      }
      return (
        sessionProviderSelectionSyncPromiseRef.current.get(trimmedSessionId) ||
        Promise.resolve()
      );
    },
    [],
  );

  const filterSessionsByWorkspace = useCallback(
    <T extends { id: string; workspace_id?: string | null }>(
      sessions: T[],
    ): T[] => {
      const resolvedWorkspaceId = normalizeProjectId(workspaceId);
      if (!resolvedWorkspaceId) {
        return [];
      }

      return sessions.filter((session) => {
        const runtimeWorkspaceId = normalizeProjectId(session.workspace_id);
        if (runtimeWorkspaceId) {
          savePersistedSessionWorkspaceId(session.id, runtimeWorkspaceId);
          return runtimeWorkspaceId === resolvedWorkspaceId;
        }

        const mappedWorkspaceId = loadStoredSessionWorkspaceIdRaw(session.id);

        if (!mappedWorkspaceId) {
          return false;
        }

        if (isLegacyDefaultProjectId(mappedWorkspaceId)) {
          return false;
        }

        return mappedWorkspaceId === resolvedWorkspaceId;
      });
    },
    [workspaceId],
  );

  const applySessionModelPreference = useCallback(
    (
      sessionId: string,
      preference: SessionModelPreference,
      options?: { markSynced?: boolean },
    ) => {
      const normalizedPreference =
        normalizeChatSessionModelPreference(preference);
      const resolvedPreference = normalizedPreference ?? {
        providerType: "",
        model: "",
      };
      providerTypeRef.current = resolvedPreference.providerType;
      modelRef.current = resolvedPreference.model;
      reasoningEffortRef.current = "";
      setProviderTypeState(resolvedPreference.providerType);
      setModelState(resolvedPreference.model);
      setReasoningEffortState("");
      savePersisted(
        scopedProviderPrefKeyRef.current,
        resolvedPreference.providerType,
      );
      savePersisted(scopedModelPrefKeyRef.current, resolvedPreference.model);
      if (normalizedPreference) {
        persistSessionModelPreference(
          sessionId,
          normalizedPreference.providerType,
          normalizedPreference.model,
        );
      } else {
        savePersisted(
          getSessionModelPreferenceKey(workspaceId, sessionId),
          null,
        );
      }
      if (options?.markSynced === true && normalizedPreference) {
        markSessionModelPreferenceSynced(
          sessionId,
          normalizedPreference.providerType,
          normalizedPreference.model,
        );
      } else {
        clearSessionModelPreferenceSynced(sessionId);
      }
    },
    [
      clearSessionModelPreferenceSynced,
      markSessionModelPreferenceSynced,
      persistSessionModelPreference,
      workspaceId,
    ],
  );

  const applyWorkspaceModelPreference = useCallback(
    (preference: SessionModelPreference) => {
      const normalizedPreference =
        normalizeChatSessionModelPreference(preference);
      const resolvedPreference = normalizedPreference ?? {
        providerType: "",
        model: "",
      };
      providerTypeRef.current = resolvedPreference.providerType;
      modelRef.current = resolvedPreference.model;
      reasoningEffortRef.current = "";
      setProviderTypeState(resolvedPreference.providerType);
      setModelState(resolvedPreference.model);
      setReasoningEffortState("");
      savePersisted(
        scopedProviderPrefKeyRef.current,
        resolvedPreference.providerType,
      );
      savePersisted(scopedModelPrefKeyRef.current, resolvedPreference.model);
    },
    [],
  );

  const setProviderType = useCallback(
    (nextProviderType: string) => {
      const currentSessionId = sessionIdRef.current;
      if (currentSessionId) {
        const pendingSelection =
          pendingSessionProviderSelectionSyncRef.current.get(
            currentSessionId.trim(),
          );
        scheduleSessionProviderSelectionSync(
          currentSessionId,
          nextProviderType,
          pendingSelection?.model ?? modelRef.current,
          "",
        );
        return;
      }

      providerTypeRef.current = nextProviderType;
      reasoningEffortRef.current = "";
      setProviderTypeState(nextProviderType);
      setReasoningEffortState("");
      savePersisted(scopedProviderPrefKeyRef.current, nextProviderType);
    },
    [scheduleSessionProviderSelectionSync, sessionIdRef],
  );

  const setModel = useCallback(
    (nextModel: string) => {
      const currentSessionId = sessionIdRef.current;
      if (currentSessionId) {
        const pendingSelection =
          pendingSessionProviderSelectionSyncRef.current.get(
            currentSessionId.trim(),
          );
        scheduleSessionProviderSelectionSync(
          currentSessionId,
          pendingSelection?.providerType ?? providerTypeRef.current,
          nextModel,
          "",
        );
        return;
      }

      modelRef.current = nextModel;
      reasoningEffortRef.current = "";
      setModelState(nextModel);
      setReasoningEffortState("");
      savePersisted(scopedModelPrefKeyRef.current, nextModel);
    },
    [scheduleSessionProviderSelectionSync, sessionIdRef],
  );

  const setReasoningEffort = useCallback(
    (nextReasoningEffort: ModelReasoningEffortLevel | "") => {
      const currentSessionId = sessionIdRef.current;
      if (currentSessionId) {
        const pendingSelection =
          pendingSessionProviderSelectionSyncRef.current.get(
            currentSessionId.trim(),
          );
        scheduleSessionProviderSelectionSync(
          currentSessionId,
          pendingSelection?.providerType ?? providerTypeRef.current,
          pendingSelection?.model ?? modelRef.current,
          nextReasoningEffort,
        );
        return;
      }
      reasoningEffortRef.current = nextReasoningEffort;
      setReasoningEffortState(nextReasoningEffort);
    },
    [scheduleSessionProviderSelectionSync, sessionIdRef],
  );

  const scheduleSessionExecutionStrategySync = useCallback(
    (
      targetSessionId: string,
      targetExecutionStrategy: AgentExecutionStrategy,
    ) => {
      const trimmedSessionId = targetSessionId.trim();
      if (!trimmedSessionId) {
        return;
      }

      const pending = pendingSessionExecutionStrategySyncRef.current;
      const alreadyQueued = pending.has(trimmedSessionId);
      pending.set(trimmedSessionId, targetExecutionStrategy);
      if (alreadyQueued) {
        return;
      }

      queueMicrotask(() => {
        const latestExecutionStrategy = pending.get(trimmedSessionId);
        pending.delete(trimmedSessionId);
        if (!latestExecutionStrategy) {
          return;
        }

        void runtime
          .setSessionExecutionStrategy(
            trimmedSessionId,
            latestExecutionStrategy,
          )
          .then(() => {
            markSessionExecutionStrategySynced(
              trimmedSessionId,
              latestExecutionStrategy,
            );
            topicsUpdaterRef.current?.(
              trimmedSessionId,
              latestExecutionStrategy,
            );
          })
          .catch((error) => {
            console.warn("[AgentChat] 更新会话执行策略失败:", error);
          });
      });
    },
    [markSessionExecutionStrategySynced, runtime, topicsUpdaterRef],
  );

  const setExecutionStrategy = useCallback(
    (nextStrategy: AgentExecutionStrategy) => {
      const normalized = normalizeExecutionStrategy(nextStrategy);
      setExecutionStrategyState(normalized);

      const currentSessionId = sessionIdRef.current;
      if (currentSessionId) {
        scheduleSessionExecutionStrategySync(currentSessionId, normalized);
      }
    },
    [scheduleSessionExecutionStrategySync, sessionIdRef],
  );

  const scheduleSessionAccessModeSync = useCallback(
    (targetSessionId: string, targetAccessMode: AgentAccessMode) => {
      const syncAccessMode = runtime.setSessionAccessMode;
      const trimmedSessionId = targetSessionId.trim();
      if (!syncAccessMode || !trimmedSessionId) {
        return;
      }

      const normalizedAccessMode = normalizeAccessMode(targetAccessMode);
      const pending = pendingSessionAccessModeSyncRef.current;
      const alreadyQueued = pending.has(trimmedSessionId);
      pending.set(trimmedSessionId, normalizedAccessMode);
      if (alreadyQueued) {
        return;
      }

      queueMicrotask(() => {
        const latestAccessMode = pending.get(trimmedSessionId);
        pending.delete(trimmedSessionId);
        if (!latestAccessMode) {
          return;
        }

        void syncAccessMode(trimmedSessionId, latestAccessMode).catch(
          (error) => {
            console.warn("[AgentChat] 更新会话 accessMode 失败:", error);
          },
        );
      });
    },
    [runtime],
  );

  const setAccessMode = useCallback(
    (nextAccessMode: AgentAccessMode) => {
      const normalized = normalizeAccessMode(nextAccessMode);
      setAccessModeState(normalized);

      const currentSessionId = sessionIdRef.current;
      if (currentSessionId) {
        persistSessionAccessMode(currentSessionId, normalized);
        scheduleSessionAccessModeSync(currentSessionId, normalized);
      }
    },
    [persistSessionAccessMode, scheduleSessionAccessModeSync, sessionIdRef],
  );

  useEffect(() => {
    const { providerKey, modelKey } = getAgentPreferenceKeys(workspaceId);
    scopedProviderPrefKeyRef.current = providerKey;
    scopedModelPrefKeyRef.current = modelKey;
    pendingSessionProviderSelectionSyncRef.current.clear();
    syncedSessionProviderSelectionRef.current.clear();
    pendingSessionExecutionStrategySyncRef.current.clear();
    syncedSessionExecutionStrategyRef.current.clear();
    pendingSessionAccessModeSyncRef.current.clear();

    const scopedPreferences = resolveWorkspaceAgentPreferences(workspaceId);
    providerTypeRef.current = scopedPreferences.providerType;
    modelRef.current = scopedPreferences.model;
    reasoningEffortRef.current = "";
    setProviderTypeState(scopedPreferences.providerType);
    setModelState(scopedPreferences.model);
    setReasoningEffortState("");

    savePersisted(providerKey, scopedPreferences.providerType);
    savePersisted(modelKey, scopedPreferences.model);

    const resolvedWorkspaceId = workspaceId?.trim();
    if (!resolvedWorkspaceId) {
      executionStrategyWorkspaceRef.current = "";
      setExecutionStrategyState("react");
      setAccessModeState(DEFAULT_AGENT_ACCESS_MODE);
      return;
    }
    executionStrategyWorkspaceRef.current = resolvedWorkspaceId;
    setExecutionStrategyState(
      resolvePersistedExecutionStrategy(resolvedWorkspaceId),
    );
    setAccessModeState(resolvePersistedAccessMode(resolvedWorkspaceId));
  }, [workspaceId]);

  useEffect(() => {
    savePersisted(scopedProviderPrefKeyRef.current, providerType);
  }, [providerType]);

  useEffect(() => {
    savePersisted(scopedModelPrefKeyRef.current, model);
  }, [model]);

  useEffect(() => {
    const storageKey = getExecutionStrategyStorageKey(workspaceId);
    if (!storageKey) {
      return;
    }
    savePersisted(storageKey, executionStrategy);
  }, [executionStrategy, workspaceId]);

  useEffect(() => {
    const storageKey = getAccessModeStorageKey(workspaceId);
    if (!storageKey) {
      return;
    }
    savePersisted(storageKey, accessMode);
  }, [accessMode, workspaceId]);

  const triggerAIGuide = useCallback(
    async (initialPrompt?: string) => {
      const sendMessage = sendMessageRef.current;
      if (!sendMessage) {
        throw new Error("发送器尚未就绪");
      }
      await sendMessage(initialPrompt?.trim() || "", [], false, false, true);
    },
    [sendMessageRef],
  );

  const fixWorkspacePathAndRetry = useCallback(
    async (newPath: string) => {
      if (!workspacePathMissing) return;
      const sendMessage = sendMessageRef.current;
      if (!sendMessage) {
        throw new Error("发送器尚未就绪");
      }
      const { content: retryContent, images: retryImages } =
        workspacePathMissing;
      setWorkspacePathMissing(null);
      try {
        await updateProject(workspaceId, { rootPath: newPath });
        notifyProjectRuntimeAgentsGuide(
          {
            id: workspaceId,
            rootPath: newPath,
          },
          {
            successMessage: "工作区目录已重新关联",
            showSuccessWhenGuideAlreadySeen: false,
          },
        );
        await sendMessage(retryContent, retryImages, false, false, true);
      } catch (err) {
        toast.error(
          `修复路径失败: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    },
    [sendMessageRef, workspaceId, workspacePathMissing],
  );

  const dismissWorkspacePathError = useCallback(() => {
    setWorkspacePathMissing(null);
  }, []);

  return {
    providerType,
    setProviderType,
    providerTypeRef,
    model,
    setModel,
    modelRef,
    reasoningEffort,
    setReasoningEffort,
    reasoningEffortRef,
    executionStrategy: effectiveExecutionStrategy,
    setExecutionStrategy,
    setExecutionStrategyState,
    accessMode,
    setAccessMode,
    setAccessModeState,
    workspacePathMissing,
    setWorkspacePathMissing,
    getWorkspaceIdForSubmit,
    persistSessionAccessMode,
    loadSessionAccessMode,
    waitForSessionProviderSelectionSync,
    persistSessionModelPreference,
    loadSessionModelPreference,
    applyWorkspaceModelPreference,
    applySessionModelPreference,
    markSessionModelPreferenceSynced,
    getSyncedSessionModelPreference,
    markSessionExecutionStrategySynced,
    getSyncedSessionExecutionStrategy,
    filterSessionsByWorkspace,
    triggerAIGuide,
    fixWorkspacePathAndRetry,
    dismissWorkspacePathError,
  };
}
