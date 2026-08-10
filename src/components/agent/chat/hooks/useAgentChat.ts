/**
 * Agent Chat Hook
 *
 * 当前事实源：
 * useAgentChat -> useAgentContext / useAgentSession / useAgentTools / useAgentStream
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import type {
  AgentExecutionStrategy,
  AgentSessionExecutionRuntime,
} from "@/lib/api/agentExecutionRuntime";
import { isLikelyImageGenerationModelId } from "@/lib/imageGen/providerMatchers";
import { scheduleMinimumDelayIdleTask } from "@/lib/utils/scheduleMinimumDelayIdleTask";
import {
  defaultAgentRuntimeAdapter,
  type AgentRuntimeAdapter,
} from "./agentRuntimeAdapter";
import { createAgentChatSendMessage } from "./agentChatSendMessage";
import { runAgentChatShellCommand } from "./agentChatShellCommand";
import { useAgentChatStateSnapshotDebug } from "./useAgentChatStateSnapshotDebug";
import { useAgentContext } from "./useAgentContext";
import { useAgentRuntimeSyncEffects } from "./useAgentRuntimeSyncEffects";
import { useAgentSession } from "./useAgentSession";
import { useAgentTools } from "./useAgentTools";
import { useAgentStream } from "./useAgentStream";
import {
  type SendMessageFn,
  type UseAgentChatOptions,
} from "./agentChatShared";
import { useAgentTopicSnapshot } from "./useAgentTopicSnapshot";
import { useAgentSessionThreadGoal } from "./useAgentSessionThreadGoal";
import { hasActiveThreadReadActivity } from "../projection/threadReadActivity";
import { resolveClawWorkspaceProviderSelection } from "../utils/clawWorkspaceProviderSelection";
import {
  applyGeneratedAutoTitleToTopics,
  buildAutoTitleConversationText,
  hasUserTextMessage,
  isAutoTitlePlaceholder,
  sanitizeGeneratedAutoTitle,
  shouldGenerateAutoTitle,
} from "./agentChatAutoTitleViewModel";
import { hasActiveRuntimeTurn } from "./agentSessionState";
import { resolveAgentChatCopy } from "@/i18n/agentChatCopy";

export type { Topic } from "./agentChatShared";

type UseAgentChatRuntimeOptions = UseAgentChatOptions & {
  runtimeAdapter?: AgentRuntimeAdapter;
  preserveRestoredMessages?: boolean;
};

const AUTO_TITLE_DEFERRED_LOAD_MS = 400;
const AUTO_TITLE_IDLE_TIMEOUT_MS = 2_000;
const DETACHED_RUNTIME_WARMUP_SCOPE = "detached";

function normalizeProviderSelection(value?: string | null): string {
  return (value ?? "").trim().toLowerCase();
}

export function useAgentChat(options: UseAgentChatRuntimeOptions) {
  const {
    systemPrompt,
    clawTraceEnabled = false,
    onWriteFile,
    workspaceId,
    workingDir,
    disableSessionRestore = false,
    sessionRestorePresentation = "foreground",
    initialTopicsLoadMode = "immediate",
    initialTopicsDeferredDelayMs,
    initialRuntimeWarmupLoadMode = initialTopicsLoadMode,
    initialRuntimeWarmupDeferredDelayMs = initialTopicsDeferredDelayMs,
    getSyncedSessionRecentPreferences,
    onOpenSubagents,
    onRestoreInterruptedInput,
    soulCopy,
    runtimeAdapter,
    preserveRestoredMessages = false,
  } = options;
  const runtime = runtimeAdapter ?? defaultAgentRuntimeAdapter;

  const [isInitialized, setIsInitialized] = useState(false);
  const activeWorkspaceIdRef = useRef(workspaceId.trim());
  activeWorkspaceIdRef.current = workspaceId.trim();
  const runtimeWarmupPromiseRef = useRef<{
    workspaceId: string;
    promise: Promise<void>;
  } | null>(null);
  const runtimeReadyWorkspaceRef = useRef<string | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const pendingSessionMetadataSyncCancelRef = useRef<(() => void) | null>(
    null,
  );
  const currentAssistantMsgIdRef = useRef<string | null>(null);
  const currentStreamingSessionIdRef = useRef<string | null>(null);
  const currentStreamingEventNameRef = useRef<string | null>(null);
  const detachStreamBindingsRef = useRef<(() => void) | null>(null);
  const autoTitleInFlightSessionIdRef = useRef<string | null>(null);
  const autoTitleCompletedSessionIdsRef = useRef<Set<string>>(new Set());
  const sendMessageRef = useRef<SendMessageFn | null>(null);
  const resetPendingActionsRef = useRef<(() => void) | null>(null);
  const topicsUpdaterRef = useRef<
    | ((sessionId: string, executionStrategy: AgentExecutionStrategy) => void)
    | null
  >(null);

  const resetPendingActions = useCallback(() => {
    resetPendingActionsRef.current?.();
  }, []);

  const context = useAgentContext({
    workspaceId,
    sessionIdRef,
    pendingSessionMetadataSyncCancelRef,
    topicsUpdaterRef,
    sendMessageRef,
    runtime,
  });

  const session = useAgentSession({
    runtime,
    workspaceId,
    workingDir,
    disableSessionRestore,
    sessionRestorePresentation,
    initialTopicsLoadMode,
    initialTopicsDeferredDelayMs,
    preserveRestoredMessages,
    executionStrategy: context.executionStrategy,
    accessMode: context.accessMode,
    providerTypeRef: context.providerTypeRef,
    modelRef: context.modelRef,
    pendingSessionMetadataSyncCancelRef,
    sessionIdRef,
    currentAssistantMsgIdRef,
    currentStreamingSessionIdRef,
    currentStreamingEventNameRef,
    detachStreamBindingsRef,
    resetPendingActions,
    persistSessionModelPreference: context.persistSessionModelPreference,
    loadSessionModelPreference: context.loadSessionModelPreference,
    applySessionModelPreference: context.applySessionModelPreference,
    markSessionModelPreferenceSynced: context.markSessionModelPreferenceSynced,
    markSessionExecutionStrategySynced:
      context.markSessionExecutionStrategySynced,
    persistSessionAccessMode: context.persistSessionAccessMode,
    loadSessionAccessMode: context.loadSessionAccessMode,
    filterSessionsByWorkspace: context.filterSessionsByWorkspace,
    setExecutionStrategyState: context.setExecutionStrategyState,
    setAccessModeState: context.setAccessModeState,
  });
  const applyWorkspaceModelPreference = context.applyWorkspaceModelPreference;
  const threadGoal = useAgentSessionThreadGoal({
    threadId: session.threadRead?.thread_id,
  });

  const resolveWarmupWorkspaceModelPreference = useCallback(
    async (
      status?: {
        provider_configured?: boolean;
        provider_name?: string;
        provider_selector?: string;
        model_name?: string;
      },
      targetWorkspaceId?: string,
    ) => {
      const isCurrentWorkspace = () =>
        !targetWorkspaceId ||
        activeWorkspaceIdRef.current === targetWorkspaceId;
      if (!isCurrentWorkspace()) {
        return;
      }

      const currentProviderType = context.providerTypeRef.current.trim();
      const currentModel = context.modelRef.current.trim();
      const hasPersistedWorkspacePreference = Boolean(
        currentProviderType && currentModel,
      );
      if (sessionIdRef.current && hasPersistedWorkspacePreference) {
        return;
      }

      const runtimeProviderType =
        status?.provider_configured === true
          ? status.provider_selector?.trim() ||
            status.provider_name?.trim() ||
            ""
          : "";
      const runtimeModel = status?.model_name?.trim() || "";
      const runtimeTextModelCandidate =
        !runtimeModel || !isLikelyImageGenerationModelId(runtimeModel);
      const usableRuntimeProviderType = runtimeTextModelCandidate
        ? runtimeProviderType
        : "";
      const usableRuntimeModel = runtimeTextModelCandidate ? runtimeModel : "";
      const hasIncompleteRuntimeSelection = Boolean(
        usableRuntimeProviderType && !usableRuntimeModel,
      );
      const currentProviderMatchesRuntime =
        Boolean(usableRuntimeProviderType) &&
        normalizeProviderSelection(currentProviderType) ===
          normalizeProviderSelection(usableRuntimeProviderType);

      if (
        usableRuntimeProviderType &&
        hasPersistedWorkspacePreference &&
        !currentProviderMatchesRuntime
      ) {
        try {
          const retainedSelection = await resolveClawWorkspaceProviderSelection(
            {
              currentProviderType,
              currentModel,
              theme: "general",
              allowProviderFallback: false,
            },
          );
          if (!isCurrentWorkspace()) {
            return;
          }
          if (retainedSelection) {
            applyWorkspaceModelPreference({
              providerType: retainedSelection.providerType,
              model: retainedSelection.model,
            });
            return;
          }
        } catch (error) {
          console.warn("[AgentChat] 校验当前工作区模型失败:", error);
        }
      }

      let runtimeSelectionRejected =
        hasIncompleteRuntimeSelection && !hasPersistedWorkspacePreference;
      if (
        usableRuntimeProviderType &&
        usableRuntimeModel &&
        (!hasPersistedWorkspacePreference || !currentProviderMatchesRuntime)
      ) {
        try {
          const runtimeSelection = await resolveClawWorkspaceProviderSelection({
            currentProviderType: usableRuntimeProviderType,
            currentModel: usableRuntimeModel,
            theme: "general",
            allowProviderFallback: false,
          });
          if (!isCurrentWorkspace()) {
            return;
          }
          if (runtimeSelection) {
            applyWorkspaceModelPreference({
              providerType: runtimeSelection.providerType,
              model: runtimeSelection.model,
            });
            return;
          }
          runtimeSelectionRejected = true;
        } catch (error) {
          runtimeSelectionRejected = true;
          console.warn("[AgentChat] 校验运行时默认模型失败:", error);
        }
      }

      if (
        status?.provider_configured &&
        hasPersistedWorkspacePreference &&
        (!usableRuntimeProviderType || currentProviderMatchesRuntime)
      ) {
        return;
      }

      try {
        if (!isCurrentWorkspace()) {
          return;
        }
        const fallbackProviderType =
          usableRuntimeProviderType || currentProviderType;
        const fallbackModel =
          !usableRuntimeProviderType || currentProviderMatchesRuntime
            ? currentModel
            : "";
        const resolvedSelectionInput = {
          currentProviderType: fallbackProviderType || undefined,
          currentModel: fallbackModel || null,
          theme: "general",
          ...(usableRuntimeProviderType && !runtimeSelectionRejected
            ? { allowProviderFallback: false }
            : {}),
        };
        const resolvedSelection = await resolveClawWorkspaceProviderSelection(
          resolvedSelectionInput,
        );

        if (!resolvedSelection || !isCurrentWorkspace()) {
          return;
        }

        applyWorkspaceModelPreference({
          providerType: resolvedSelection.providerType,
          model: resolvedSelection.model,
        });
      } catch (error) {
        console.warn("[AgentChat] 预热阶段解析工作区模型失败:", error);
      }
    },
    [applyWorkspaceModelPreference, context.modelRef, context.providerTypeRef],
  );

  const warmupRuntime = useCallback(
    async (options?: { allowDetached?: boolean }) => {
      const resolvedWorkspaceId = workspaceId.trim();
      const allowDetached = options?.allowDetached === true;
      if (!resolvedWorkspaceId && !allowDetached) {
        runtimeReadyWorkspaceRef.current = null;
        runtimeWarmupPromiseRef.current = null;
        setIsInitialized(false);
        return;
      }

      const warmupScopeId =
        resolvedWorkspaceId || DETACHED_RUNTIME_WARMUP_SCOPE;

      if (runtimeReadyWorkspaceRef.current === warmupScopeId) {
        return;
      }

      const activeWarmup = runtimeWarmupPromiseRef.current;
      if (activeWarmup?.workspaceId === warmupScopeId) {
        await activeWarmup.promise;
        return;
      }

      const warmupPromise = runtime
        .getRuntimeProviderSelection()
        .then(async (status) => {
          if (activeWorkspaceIdRef.current !== resolvedWorkspaceId) {
            return;
          }
          await resolveWarmupWorkspaceModelPreference(
            status,
            resolvedWorkspaceId,
          );
          if (activeWorkspaceIdRef.current !== resolvedWorkspaceId) {
            return;
          }
          runtimeReadyWorkspaceRef.current = warmupScopeId;
          setIsInitialized(true);
          console.log("[AgentChat] Provider 配置读取完成");
        })
        .catch((err) => {
          if (runtimeReadyWorkspaceRef.current === warmupScopeId) {
            runtimeReadyWorkspaceRef.current = null;
          }
          setIsInitialized(false);
          console.error("[AgentChat] Provider 配置读取失败:", err);
          throw err;
        })
        .finally(() => {
          const active = runtimeWarmupPromiseRef.current;
          if (
            active?.workspaceId === warmupScopeId &&
            active.promise === warmupPromise
          ) {
            runtimeWarmupPromiseRef.current = null;
          }
        });

      runtimeWarmupPromiseRef.current = {
        workspaceId: warmupScopeId,
        promise: warmupPromise,
      };
      await warmupPromise;
    },
    [resolveWarmupWorkspaceModelPreference, runtime, workspaceId],
  );

  const tools = useAgentTools({
    runtime,
    sessionIdRef,
    currentStreamingSessionIdRef,
    currentStreamingEventNameRef,
    messages: session.messages,
    setMessages: session.setMessages,
    setThreadItems: session.setThreadItems,
    refreshSessionReadModel: session.refreshSessionReadModel,
    refreshSessionDetail: session.refreshSessionDetail,
  });

  resetPendingActionsRef.current = () => tools.setPendingActions([]);

  const threadBusy = hasActiveRuntimeTurn({
    threadRead: session.threadRead,
    threadReadStatus: session.threadRead?.status,
    turns: session.threadTurns,
  });
  const hasRecoverableRunningActivity = hasActiveThreadReadActivity(
    session.threadRead,
  );

  const stream = useAgentStream({
    runtime,
    systemPrompt,
    onWriteFile,
    ensureSession: session.ensureSession,
    attemptSilentTurnRecovery: session.attemptSilentTurnRecovery,
    sessionIdRef,
    sessionId: session.sessionId,
    executionStrategy: context.executionStrategy,
    accessMode: context.accessMode,
    providerTypeRef: context.providerTypeRef,
    modelRef: context.modelRef,
    reasoningEffortRef: context.reasoningEffortRef,
    getSyncedSessionModelPreference: context.getSyncedSessionModelPreference,
    getSyncedSessionExecutionStrategy:
      context.getSyncedSessionExecutionStrategy,
    getSyncedSessionRecentPreferences,
    currentAssistantMsgIdRef,
    currentStreamingSessionIdRef,
    currentStreamingEventNameRef,
    warnedKeysRef: tools.warnedKeysRef,
    getWorkspaceIdForSubmit: context.getWorkspaceIdForSubmit,
    waitForSessionProviderSelectionSync:
      context.waitForSessionProviderSelectionSync,
    getThreadIdForSubmit: session.getThreadIdForSubmit,
    setWorkspacePathMissing: context.setWorkspacePathMissing,
    getMessages: () => session.messages,
    setMessages: session.setMessages,
    getThreadItems: () => session.threadItems,
    setThreadItems: session.setThreadItems,
    setTodoItems: session.setTodoItems,
    setThreadTurns: session.setThreadTurns,
    setCurrentTurnId: session.setCurrentTurnId,
    setExecutionRuntime: session.setExecutionRuntime,
    threadBusy,
    currentTurnId: session.currentTurnId,
    threadRead: session.threadRead,
    threadTurns: session.threadTurns,
    setPendingActions: tools.setPendingActions,
    refreshSessionReadModel: session.refreshSessionReadModel,
    onRestoreInterruptedInput,
    executionRuntime: session.executionRuntime,
    clawTraceEnabled,
    allowRecoveredStreamBinding:
      Boolean(session.sessionId) &&
      (session.recoveredStreamBindingSessionId === session.sessionId ||
        hasRecoverableRunningActivity),
    soulCopy,
  });
  detachStreamBindingsRef.current = stream.detachStreamBindings;
  const setChatMessages = session.setMessages;
  const clearChatMessages = session.clearMessages;
  const createFreshSession = session.createFreshSession;
  const currentTurnId = session.currentTurnId;
  const activeSessionId = session.sessionId;
  const queuedTurnsCount = session.queuedTurnCount;
  const rawSendMessage = stream.sendMessage;
  const compactCurrentSession = stream.compactSession;
  const isStreamSending = stream.isSending;
  const detachStreamBindings = stream.detachStreamBindings;

  const settleActiveRuntimeStream = useCallback(
    (targetSessionId: string) => {
      const activeStreamingSessionId =
        currentStreamingSessionIdRef.current?.trim() || null;
      if (
        activeStreamingSessionId &&
        activeStreamingSessionId !== targetSessionId
      ) {
        return;
      }
      detachStreamBindings();
    },
    [detachStreamBindings],
  );

  const appendLocalAssistantMessage = useCallback(
    (content: string) => {
      setChatMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content,
          timestamp: new Date(),
        },
      ]);
    },
    [setChatMessages],
  );

  const runUserShellCommand = useCallback(
    (command: string) =>
      runAgentChatShellCommand({
        command,
        ensureSession: session.ensureSession,
        getThreadId: session.getThreadIdForSubmit,
        refreshSessionDetail: session.refreshSessionDetail,
        refreshSessionReadModel: session.refreshSessionReadModel,
        runtime,
      }),
    [
      runtime,
      session.ensureSession,
      session.getThreadIdForSubmit,
      session.refreshSessionDetail,
      session.refreshSessionReadModel,
    ],
  );

  const sendMessage = useCallback<SendMessageFn>(
    async (...args) => {
      const hasCompleteModelSelection = Boolean(
        context.providerTypeRef.current.trim() &&
        context.modelRef.current.trim(),
      );
      if (hasCompleteModelSelection) {
        void warmupRuntime({ allowDetached: true }).catch(() => undefined);
      } else {
        await warmupRuntime({ allowDetached: true });
      }
      const send = createAgentChatSendMessage({
        baseStatusSnapshot: {
          sessionId: activeSessionId,
          currentTurnId,
          providerType: context.providerTypeRef.current,
          model: context.modelRef.current,
          executionStrategy: context.executionStrategy,
          queuedTurnsCount,
          isSending: isStreamSending,
        },
        rawSendMessage,
        compactSession: compactCurrentSession,
        clearMessages: clearChatMessages,
        createFreshSession,
        appendAssistantMessage: appendLocalAssistantMessage,
        notifyInfo: (message) => toast.info(message),
        notifySuccess: (message) => toast.success(message),
        notifyError: (message) => toast.error(message),
        shellCommandHelp: resolveAgentChatCopy(
          "inputbar.shellCommand.empty",
          "请输入要执行的命令",
        ),
        shellCommandError: (message) =>
          resolveAgentChatCopy(
            "inputbar.shellCommand.error",
            "命令提交失败：{{message}}",
            {
              message,
            },
          ),
        runUserShellCommand,
        onOpenSubagents,
      });
      await send(...args);
    },
    [
      appendLocalAssistantMessage,
      activeSessionId,
      clearChatMessages,
      compactCurrentSession,
      context.executionStrategy,
      context.modelRef,
      context.providerTypeRef,
      createFreshSession,
      currentTurnId,
      isStreamSending,
      onOpenSubagents,
      queuedTurnsCount,
      rawSendMessage,
      runUserShellCommand,
      warmupRuntime,
    ],
  );

  sendMessageRef.current = sendMessage;
  topicsUpdaterRef.current = session.updateTopicExecutionStrategy;

  const hasActiveTopic = Boolean(
    session.sessionId &&
    session.topics.some((topic) => topic.id === session.sessionId),
  );
  const activeExecutionRuntime =
    useMemo<AgentSessionExecutionRuntime | null>(() => {
      const threadStatus = session.threadRead?.status;
      const shouldPreferRuntime =
        stream.isSending ||
        threadStatus === "running" ||
        threadStatus === "queued";
      return shouldPreferRuntime ? session.executionRuntime : null;
    }, [
      session.executionRuntime,
      session.threadRead?.status,
      stream.isSending,
    ]);

  useAgentRuntimeSyncEffects({
    runtime,
    sessionIdRef,
    sessionId: session.sessionId,
    currentTurnEventName: stream.activeStreamEventName,
    currentStreamTurnId: stream.activeStreamTurnId,
    isSending: stream.isSending,
    threadRead: session.threadRead,
    threadReadStatus: session.threadRead?.status,
    queuedTurnCount: session.queuedTurnCount,
    threadTurns: session.threadTurns,
    refreshSessionDetail: session.refreshSessionDetail,
    refreshSessionReadModel: session.refreshSessionReadModel,
    settleActiveRuntimeStream,
  });

  useAgentTopicSnapshot({
    sessionId: session.sessionId,
    hasActiveTopic,
    suppressInactiveTopicWarning:
      session.isDetachedActiveSession === true ||
      session.isSessionHydrating === true,
    messages: session.messages,
    isSending: stream.isSending,
    pendingActionCount: tools.pendingActions.length,
    queuedTurnCount: session.queuedTurnCount,
    threadStatus: session.threadRead?.status ?? null,
    workspaceId,
    workspacePathMissing: Boolean(context.workspacePathMissing),
    topicsCount: session.topics.length,
    updateTopicSnapshot: session.updateTopicSnapshot,
  });

  const sessionMessages = session.messages;
  const sessionTopics = session.topics;
  const sessionSetTopics = session.setTopics;
  const currentSessionId = session.sessionId;
  const activeSessionTitle = useMemo(() => {
    const activeSessionId = currentSessionId?.trim();
    if (!activeSessionId) {
      return null;
    }

    const activeTopic = sessionTopics.find(
      (topic) => topic.id === activeSessionId,
    );
    return activeTopic?.title?.trim() ?? null;
  }, [currentSessionId, sessionTopics]);

  useEffect(() => {
    const activeSessionId = currentSessionId?.trim();
    if (!activeSessionId || stream.isSending || !runtime.generateSessionTitle) {
      return;
    }

    if (activeSessionTitle === null) {
      return;
    }
    const shouldAutoGenerateTitle = shouldGenerateAutoTitle({
      activeSessionTitle,
      messages: sessionMessages,
    });
    if (!shouldAutoGenerateTitle) {
      autoTitleCompletedSessionIdsRef.current.add(activeSessionId);
      return;
    }

    if (!hasUserTextMessage(sessionMessages)) {
      return;
    }

    if (
      autoTitleCompletedSessionIdsRef.current.has(activeSessionId) ||
      autoTitleInFlightSessionIdRef.current === activeSessionId
    ) {
      return;
    }

    autoTitleInFlightSessionIdRef.current = activeSessionId;
    let cancelled = false;
    let titleApplied = false;

    const cancelDeferredTitle = scheduleMinimumDelayIdleTask(
      () => {
        void (async () => {
          try {
            const conversationText =
              buildAutoTitleConversationText(sessionMessages);

            const rawGeneratedTitle = (
              await runtime.generateSessionTitle?.(
                activeSessionId,
                conversationText,
              )
            )?.trim();
            const generatedTitle = sanitizeGeneratedAutoTitle(
              rawGeneratedTitle,
              conversationText,
            );
            if (
              cancelled ||
              !generatedTitle ||
              isAutoTitlePlaceholder(generatedTitle)
            ) {
              return;
            }

            await runtime.renameSession(activeSessionId, generatedTitle);
            sessionSetTopics((previous) =>
              applyGeneratedAutoTitleToTopics(
                previous,
                activeSessionId,
                generatedTitle,
                conversationText,
              ),
            );
            titleApplied = true;
          } catch (error) {
            console.debug("[AgentChat] 自动生成会话标题失败:", error);
          } finally {
            if (!cancelled && titleApplied) {
              autoTitleCompletedSessionIdsRef.current.add(activeSessionId);
            }
            if (autoTitleInFlightSessionIdRef.current === activeSessionId) {
              autoTitleInFlightSessionIdRef.current = null;
            }
          }
        })();
      },
      {
        minimumDelayMs: AUTO_TITLE_DEFERRED_LOAD_MS,
        idleTimeoutMs: AUTO_TITLE_IDLE_TIMEOUT_MS,
      },
    );

    return () => {
      cancelled = true;
      cancelDeferredTitle();
      if (autoTitleInFlightSessionIdRef.current === activeSessionId) {
        autoTitleInFlightSessionIdRef.current = null;
      }
    };
  }, [
    activeSessionTitle,
    currentSessionId,
    runtime,
    sessionMessages,
    sessionSetTopics,
    stream.isSending,
  ]);

  useAgentChatStateSnapshotDebug({
    hasActiveTopic,
    isSending: stream.isSending,
    messagesCount: session.messages.length,
    pendingActionsCount: tools.pendingActions.length,
    sessionId: session.sessionId ?? null,
    threadTurnsCount: session.threadTurns.length,
    topicsCount: session.topics.length,
    workspaceId,
    workspacePathMissing: context.workspacePathMissing,
  });

  useEffect(() => {
    tools.warnedKeysRef.current.clear();
  }, [tools.warnedKeysRef, workspaceId]);

  useEffect(() => {
    const resolvedWorkspaceId = workspaceId.trim();
    if (runtimeReadyWorkspaceRef.current !== resolvedWorkspaceId) {
      setIsInitialized(false);
    }
    if (!resolvedWorkspaceId) {
      runtimeReadyWorkspaceRef.current = null;
      runtimeWarmupPromiseRef.current = null;
      setIsInitialized(false);
      return;
    }

    if (initialRuntimeWarmupLoadMode === "deferred") {
      return scheduleMinimumDelayIdleTask(
        () => {
          void warmupRuntime().catch(() => undefined);
        },
        {
          minimumDelayMs: initialRuntimeWarmupDeferredDelayMs ?? 0,
        },
      );
    }

    void warmupRuntime().catch(() => undefined);
  }, [
    initialRuntimeWarmupDeferredDelayMs,
    initialRuntimeWarmupLoadMode,
    warmupRuntime,
    workspaceId,
  ]);

  const handleStartProcess = async () => {
    try {
      await warmupRuntime();
    } catch {
      return;
    }
  };

  const handleStopProcess = async () => {
    session.clearMessages({ showToast: false });
  };

  return {
    processStatus: { running: isInitialized },
    handleStartProcess,
    handleStopProcess,

    providerType: context.providerType,
    setProviderType: context.setProviderType,
    model: context.model,
    setModel: context.setModel,
    reasoningEffort: context.reasoningEffort,
    setReasoningEffort: context.setReasoningEffort,
    executionStrategy: context.executionStrategy,
    setExecutionStrategy: context.setExecutionStrategy,
    accessMode: context.accessMode,
    setAccessMode: context.setAccessMode,
    providerConfig: {},
    isConfigLoading: false,

    messages: session.messages,
    setMessages: session.setMessages,
    currentThreadId: session.sessionId,
    currentTurnId: session.currentTurnId,
    turns: session.threadTurns,
    threadItems: session.threadItems,
    todoItems: session.todoItems,
    queuedTurnCount: session.queuedTurnCount,
    threadRead: session.threadRead,
    threadGoal: threadGoal.goal,
    threadGoalError: threadGoal.error,
    isThreadGoalLoading: threadGoal.loading,
    executionRuntime: session.executionRuntime,
    sessionWorkingDir: session.sessionWorkingDir,
    activeExecutionRuntime,
    isSending: stream.isSending,
    sendMessage,
    compactSession: stream.compactSession,
    stopSending: stream.stopSending,
    replayPendingAction: tools.replayPendingAction,
    clearMessages: session.clearMessages,
    deleteMessage: session.deleteMessage,
    editMessage: session.editMessage,
    handlePermissionResponse: tools.handlePermissionResponse,
    triggerAIGuide: context.triggerAIGuide,

    topics: session.topics,
    topicsReady: session.topicsReady,
    sessionHistoryWindow: session.sessionHistoryWindow,
    isAutoRestoringSession: session.isAutoRestoringSession,
    isSessionHydrating: session.isSessionHydrating,
    sessionId: session.sessionId,
    createFreshSession: session.createFreshSession,
    ensureSession: session.ensureSession,
    switchTopic: session.switchTopic,
    loadFullSessionHistory: session.loadFullSessionHistory,
    refreshSessionReadModel: session.refreshSessionReadModel,
    deleteTopic: session.deleteTopic,
    renameTopic: session.renameTopic,
    loadTopics: session.loadTopics,
    updateTopicSnapshot: session.updateTopicSnapshot,

    pendingActions: tools.pendingActions,
    submittedActionsInFlight: tools.submittedActionsInFlight,
    confirmAction: tools.confirmAction,

    workspacePathMissing: context.workspacePathMissing,
    fixWorkspacePathAndRetry: context.fixWorkspacePathAndRetry,
    dismissWorkspacePathError: context.dismissWorkspacePathError,
  };
}
