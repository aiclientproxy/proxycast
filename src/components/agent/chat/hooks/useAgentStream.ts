import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
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
  AutoContinueRequestPayload,
} from "@/lib/api/agentRuntime/sessionTypes";
import {
  type AgentThreadItem,
  type AgentThreadTurn,
} from "@/lib/api/agentProtocol";
import type { ActionRequired, Message, MessageImage } from "../types";
import type { ChatToolPreferences } from "../utils/chatToolPreferences";
import type { AgentAccessMode } from "./agentChatStorage";
import type { SoulInteractionCopy } from "@/lib/soul/interactionCopy";
import type {
  SendMessageOptions,
  SessionModelPreference,
  WorkspacePathMissingState,
} from "./agentChatShared";
import type {
  InterruptedInputDraftSnapshot,
  InterruptedInputRestoreRequest,
} from "./agentStreamInputRestoreTypes";
import type { AgentRuntimeAdapter } from "./agentRuntimeAdapter";
import {
  createAgentStreamPreparedSendEnv,
  type AgentStreamPreparedSendEnv,
} from "./agentStreamPreparedSendEnv";
import {
  bindRecoveredAgentStreamThread,
  hasBlockingActiveAgentStreamBinding,
  resolveAgentStreamResumeBindingTarget,
} from "./agentStreamResumeBinding";
import { AgentStreamSubmitGate } from "./agentStreamSubmitGate";
import {
  normalizeAgentStreamCompactionError,
  runAgentStreamCompaction,
} from "./agentStreamCompaction";
import { stopActiveAgentStream } from "./agentStreamFlowControl";
import { sendAgentStreamMessage } from "./agentStreamSend";
import { useAgentStreamController } from "./useAgentStreamController";

function appendThinkingToParts(
  parts: NonNullable<Message["contentParts"]>,
  textDelta: string,
): NonNullable<Message["contentParts"]> {
  const nextParts = [...parts];
  const lastPart = nextParts[nextParts.length - 1];

  if (lastPart?.type === "thinking") {
    const base = lastPart.text;
    const chunk = textDelta;
    let merged: string;
    if (!base) {
      merged = chunk;
    } else if (!chunk) {
      merged = base;
    } else if (chunk.startsWith(base)) {
      merged = chunk;
    } else if (base.endsWith(chunk)) {
      merged = base;
    } else {
      const maxOverlap = Math.min(base.length, chunk.length);
      let found = false;
      merged = base + chunk;
      for (let overlap = maxOverlap; overlap > 0; overlap -= 1) {
        if (base.slice(-overlap) === chunk.slice(0, overlap)) {
          merged = base + chunk.slice(overlap);
          found = true;
          break;
        }
      }
      void found;
    }
    nextParts[nextParts.length - 1] = { type: "thinking", text: merged };
    return nextParts;
  }

  nextParts.push({ type: "thinking", text: textDelta });
  return nextParts;
}

function cloneInterruptedInputDraft(
  draft: InterruptedInputDraftSnapshot,
): InterruptedInputDraftSnapshot {
  return {
    text: draft.text,
    images: draft.images ? [...draft.images] : [],
    pathReferences: draft.pathReferences ? [...draft.pathReferences] : [],
    textElements: draft.textElements ? [...draft.textElements] : [],
    inputCapabilityRoute: draft.inputCapabilityRoute,
  };
}

interface UseAgentStreamOptions {
  runtime: AgentRuntimeAdapter;
  systemPrompt?: string;
  onWriteFile?: (
    content: string,
    fileName: string,
    context?: import("../types").WriteArtifactContext,
  ) => void;
  ensureSession: (options?: {
    targetSessionId?: string;
    skipSessionRestore?: boolean;
    skipSessionStartHooks?: boolean;
  }) => Promise<string | null>;
  attemptSilentTurnRecovery: (
    sessionId: string,
    requestStartedAt: number,
    promptText: string,
    options?: { requireTerminal?: boolean; turnId?: string | null },
  ) => Promise<boolean>;
  sessionIdRef: MutableRefObject<string | null>;
  sessionId?: string | null;
  executionStrategy: AgentExecutionStrategy;
  accessMode: AgentAccessMode;
  workingDir?: string | null;
  providerTypeRef: MutableRefObject<string>;
  modelRef: MutableRefObject<string>;
  reasoningEffortRef: MutableRefObject<string>;
  getSyncedSessionModelPreference: (
    sessionId: string,
  ) => SessionModelPreference | null;
  getSyncedSessionExecutionStrategy: (
    sessionId: string,
  ) => AgentExecutionStrategy | null;
  getSyncedSessionRecentPreferences?: (
    sessionId: string,
  ) => ChatToolPreferences | null;
  currentAssistantMsgIdRef: MutableRefObject<string | null>;
  currentStreamingSessionIdRef: MutableRefObject<string | null>;
  currentStreamingEventNameRef: MutableRefObject<string | null>;
  warnedKeysRef: MutableRefObject<Set<string>>;
  getWorkspaceIdForSubmit: () => string | undefined;
  waitForSessionProviderSelectionSync: (sessionId: string) => Promise<void>;
  getThreadIdForSubmit: (targetSessionId?: string) => string | undefined;
  setWorkspacePathMissing: Dispatch<
    SetStateAction<WorkspacePathMissingState | null>
  >;
  getMessages?: () => readonly Message[];
  setMessages: Dispatch<SetStateAction<Message[]>>;
  getThreadItems?: () => readonly AgentThreadItem[];
  setThreadItems: Dispatch<SetStateAction<AgentThreadItem[]>>;
  setTodoItems: Dispatch<SetStateAction<AgentTodoItem[]>>;
  setThreadTurns: Dispatch<SetStateAction<AgentThreadTurn[]>>;
  setCurrentTurnId: Dispatch<SetStateAction<string | null>>;
  setExecutionRuntime: Dispatch<
    SetStateAction<AgentSessionExecutionRuntime | null>
  >;
  threadBusy: boolean;
  currentTurnId?: string | null;
  threadRead?: AgentRuntimeThreadReadModel | null;
  threadTurns: readonly AgentThreadTurn[];
  setPendingActions: Dispatch<SetStateAction<ActionRequired[]>>;
  refreshSessionReadModel: (targetSessionId?: string) => Promise<boolean>;
  onRestoreInterruptedInput?: (request: InterruptedInputRestoreRequest) => void;
  executionRuntime: AgentSessionExecutionRuntime | null;
  clawTraceEnabled?: boolean;
  allowRecoveredStreamBinding?: boolean;
  soulCopy?: SoulInteractionCopy;
}

export function useAgentStream(options: UseAgentStreamOptions) {
  const {
    runtime,
    systemPrompt,
    onWriteFile,
    ensureSession,
    attemptSilentTurnRecovery,
    sessionIdRef,
    sessionId,
    executionStrategy,
    accessMode,
    workingDir,
    providerTypeRef,
    modelRef,
    reasoningEffortRef,
    getSyncedSessionModelPreference,
    getSyncedSessionExecutionStrategy,
    getSyncedSessionRecentPreferences,
    currentAssistantMsgIdRef,
    currentStreamingSessionIdRef,
    currentStreamingEventNameRef,
    warnedKeysRef,
    getWorkspaceIdForSubmit,
    waitForSessionProviderSelectionSync,
    getThreadIdForSubmit,
    setWorkspacePathMissing,
    getMessages,
    setMessages,
    getThreadItems,
    setThreadItems,
    setTodoItems,
    setThreadTurns,
    setCurrentTurnId,
    setExecutionRuntime,
    threadBusy,
    currentTurnId,
    threadRead,
    threadTurns,
    setPendingActions,
    refreshSessionReadModel,
    onRestoreInterruptedInput,
    executionRuntime,
    clawTraceEnabled = false,
    allowRecoveredStreamBinding = false,
    soulCopy,
  } = options;

  const {
    isSending,
    activeStreamEventName,
    activeStreamTurnId,
    setIsSending,
    listenerMapRef,
    activeStreamRef,
    setActiveStream,
    clearActiveStreamIfMatch,
    replaceStreamListener,
    removeStreamListener,
    clearStreamBindings,
  } = useAgentStreamController({
    currentAssistantMsgIdRef,
    currentStreamingSessionIdRef,
    currentStreamingEventNameRef,
  });
  const preparedSubmitGateRef = useRef(new AgentStreamSubmitGate());
  const submittedDraftFallbackRef =
    useRef<InterruptedInputDraftSnapshot | null>(null);
  const recoveredBindingAttemptKeyRef = useRef<string | null>(null);
  const getMessagesRef = useRef(getMessages);
  const getThreadItemsRef = useRef(getThreadItems);
  getMessagesRef.current = getMessages;
  getThreadItemsRef.current = getThreadItems;

  const preparedSendEnv = useMemo<AgentStreamPreparedSendEnv>(
    () =>
      createAgentStreamPreparedSendEnv({
        runtime,
        ensureSession,
        attemptSilentTurnRecovery,
        refreshSessionReadModel,
        executionStrategy,
        accessMode,
        workingDir,
        providerTypeRef,
        modelRef,
        reasoningEffortRef,
        sessionIdRef,
        runPreparedSubmit: (task) => preparedSubmitGateRef.current.run(task),
        getWorkspaceIdForSubmit,
        waitForSessionProviderSelectionSync,
        getThreadIdForSubmit,
        getSyncedSessionModelPreference,
        getSyncedSessionExecutionStrategy,
        getSyncedSessionRecentPreferences,
        listenerMapRef,
        activeStreamRef,
        warnedKeysRef,
        onWriteFile,
        executionRuntime,
        clawTraceEnabled,
        soulCopy,
        setActiveStream,
        clearActiveStreamIfMatch,
        setMessages,
        getThreadItems,
        setThreadItems,
        setTodoItems,
        setThreadTurns,
        setCurrentTurnId,
        setExecutionRuntime,
        setPendingActions,
        setWorkspacePathMissing,
        setIsSending,
        appendThinkingToParts,
      }),
    [
      activeStreamRef,
      accessMode,
      attemptSilentTurnRecovery,
      clearActiveStreamIfMatch,
      executionStrategy,
      ensureSession,
      executionRuntime,
      clawTraceEnabled,
      soulCopy,
      getWorkspaceIdForSubmit,
      waitForSessionProviderSelectionSync,
      getThreadIdForSubmit,
      getSyncedSessionModelPreference,
      getSyncedSessionExecutionStrategy,
      getSyncedSessionRecentPreferences,
      getThreadItems,
      listenerMapRef,
      modelRef,
      onWriteFile,
      providerTypeRef,
      reasoningEffortRef,
      refreshSessionReadModel,
      runtime,
      sessionIdRef,
      setActiveStream,
      setCurrentTurnId,
      setExecutionRuntime,
      setIsSending,
      setMessages,
      setPendingActions,
      setThreadItems,
      setTodoItems,
      setThreadTurns,
      setWorkspacePathMissing,
      warnedKeysRef,
      workingDir,
    ],
  );

  const recoveredBindingTarget = useMemo(
    () =>
      resolveAgentStreamResumeBindingTarget({
        currentTurnId,
        sessionId: sessionId ?? sessionIdRef.current,
        threadBusy,
        threadRead,
        threadTurns,
      }),
    [
      currentTurnId,
      sessionId,
      sessionIdRef,
      threadBusy,
      threadRead,
      threadTurns,
    ],
  );
  const recoveredBindingKey = recoveredBindingTarget
    ? `${recoveredBindingTarget.sessionId}:${recoveredBindingTarget.turnId}`
    : null;
  const recoveredBindingTargetRef = useRef(recoveredBindingTarget);
  recoveredBindingTargetRef.current = recoveredBindingTarget;

  useEffect(() => {
    const target = recoveredBindingTargetRef.current;
    if (!allowRecoveredStreamBinding || !target || !recoveredBindingKey) {
      recoveredBindingAttemptKeyRef.current = null;
      return;
    }
    if (
      hasBlockingActiveAgentStreamBinding({
        activeStreamRef,
        listenerMapRef,
        target,
      })
    ) {
      return;
    }
    if (recoveredBindingAttemptKeyRef.current === recoveredBindingKey) {
      return;
    }

    recoveredBindingAttemptKeyRef.current = recoveredBindingKey;
    let disposed = false;
    let cleanup: (() => void) | null = null;
    void bindRecoveredAgentStreamThread({
      activeStreamRef,
      appendThinkingToParts,
      clearActiveStreamIfMatch,
      executionStrategy,
      getMessages: () => getMessagesRef.current?.() ?? [],
      getThreadItems: () => getThreadItemsRef.current?.() ?? [],
      listenerMapRef,
      onWriteFile,
      refreshSessionReadModel,
      runtime,
      setActiveStream,
      setCurrentTurnId,
      setExecutionRuntime,
      setIsSending,
      setMessages,
      setPendingActions,
      setThreadItems,
      setThreadTurns,
      soulCopy,
      target,
      warnedKeysRef,
    })
      .then((nextCleanup) => {
        if (disposed) {
          nextCleanup?.();
          return;
        }
        cleanup = nextCleanup;
        if (!nextCleanup) {
          recoveredBindingAttemptKeyRef.current = null;
        }
      })
      .catch((error) => {
        recoveredBindingAttemptKeyRef.current = null;
        console.error("[AgentChat] 绑定运行中会话失败:", error);
      });

    return () => {
      disposed = true;
      cleanup?.();
    };
  }, [
    activeStreamRef,
    allowRecoveredStreamBinding,
    clearActiveStreamIfMatch,
    executionStrategy,
    getMessagesRef,
    getThreadItemsRef,
    listenerMapRef,
    onWriteFile,
    recoveredBindingKey,
    recoveredBindingTargetRef,
    refreshSessionReadModel,
    runtime,
    setActiveStream,
    setCurrentTurnId,
    setExecutionRuntime,
    setIsSending,
    setMessages,
    setPendingActions,
    setThreadItems,
    setThreadTurns,
    soulCopy,
    warnedKeysRef,
  ]);

  const sendMessage = useCallback(
    async (
      content: string,
      images: MessageImage[],
      webSearch?: boolean,
      _thinking?: boolean,
      skipUserMessage = false,
      executionStrategyOverride?: AgentExecutionStrategy,
      modelOverride?: string,
      autoContinue?: AutoContinueRequestPayload,
      options?: SendMessageOptions,
    ) => {
      submittedDraftFallbackRef.current = cloneInterruptedInputDraft(
        options?.inputRestoreDraft ?? {
          text: options?.displayContent ?? content,
          images,
          textElements: (options?.displayContent ?? content).trim()
            ? [{ type: "text", text: options?.displayContent ?? content }]
            : [],
          inputCapabilityRoute: options?.capabilityRoute,
        },
      );
      await sendAgentStreamMessage({
        content,
        images,
        webSearch,
        thinking: _thinking,
        skipUserMessage,
        executionStrategyOverride,
        modelOverride,
        autoContinue,
        systemPrompt,
        options,
        env: preparedSendEnv,
      });
    },
    [preparedSendEnv, systemPrompt],
  );

  const stopSending = useCallback(async () => {
    await stopActiveAgentStream({
      activeStream: activeStreamRef.current,
      sessionIdRef,
      threadId: getThreadIdForSubmit(),
      currentTurnId,
      runtime,
      removeStreamListener,
      refreshSessionReadModel,
      setThreadItems,
      setThreadTurns,
      setCurrentTurnId,
      setMessages,
      getMessages: () => getMessagesRef.current?.() ?? [],
      getThreadItems: () => getThreadItemsRef.current?.() ?? [],
      setActiveStream,
      submittedDraftFallback: submittedDraftFallbackRef.current,
      onRestoreInterruptedInput,
      notify: {
        info: (message) => toast.info(message),
        error: () => undefined,
      },
      onInterruptError: (error) => {
        console.error("[AgentChat] 停止失败:", error);
      },
    });
    submittedDraftFallbackRef.current = null;
  }, [
    refreshSessionReadModel,
    runtime,
    sessionIdRef,
    onRestoreInterruptedInput,
    setActiveStream,
    setCurrentTurnId,
    setMessages,
    setThreadItems,
    setThreadTurns,
    removeStreamListener,
    activeStreamRef,
    currentTurnId,
    getThreadIdForSubmit,
    getThreadItemsRef,
  ]);

  const compactSession = useCallback(async () => {
    const activeSessionId = sessionIdRef.current;
    if (!activeSessionId) {
      toast.error("当前没有可压缩的会话");
      return;
    }

    if (activeStreamRef.current) {
      toast.info("当前仍有任务执行中，稍后再压缩上下文");
      return;
    }

    try {
      await runAgentStreamCompaction({
        runtime,
        sessionId: activeSessionId,
        warnedKeysRef,
        setActiveStream,
        clearActiveStreamIfMatch,
        replaceStreamListener,
        removeStreamListener,
        setIsSending,
        setCurrentTurnId,
        setThreadItems,
        setThreadTurns,
        notify: {
          info: (message) => toast.info(message),
          warning: (message) => toast.warning(message),
          error: (message) => toast.error(message),
        },
      });
    } catch (error) {
      const compactionError = normalizeAgentStreamCompactionError(error);
      console.error("[AgentChat] 压缩上下文失败:", compactionError);
      if (!compactionError.alreadyNotified) {
        toast.error(compactionError.message);
      }
    }
  }, [
    clearActiveStreamIfMatch,
    removeStreamListener,
    replaceStreamListener,
    runtime,
    sessionIdRef,
    setActiveStream,
    setIsSending,
    setCurrentTurnId,
    setThreadItems,
    setThreadTurns,
    warnedKeysRef,
    activeStreamRef,
  ]);

  return {
    isSending,
    activeStreamEventName,
    activeStreamTurnId,
    sendMessage,
    compactSession,
    stopSending,
    detachStreamBindings: clearStreamBindings,
  };
}
