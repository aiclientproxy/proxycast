import type { Dispatch, MutableRefObject, SetStateAction } from "react";
import type { AgentThreadItem, AgentThreadTurn } from "@/lib/api/agentProtocol";
import type {
  AgentExecutionStrategy,
  AgentSessionExecutionRuntime,
} from "@/lib/api/agentExecutionRuntime";
import type { AutoContinueRequestPayload } from "@/lib/api/agentRuntime/sessionTypes";
import type { AgentTodoItem } from "@/lib/api/agentRuntime/sessionTypes";
import type { TurnSteerParams } from "@limecloud/app-server-client";
import type { ModeKind } from "@limecloud/app-server-client";
import { setThreadGoal } from "@/lib/api/agentRuntime/threadGoalClient";
import { modelRegistryApi } from "@/lib/api/modelRegistry";
import type { ModelCapabilitySummary } from "@/lib/model/inferModelCapabilities";
import {
  buildModelCapabilitySendGateInput,
  evaluateModelInputCapability,
  mergeModelInputCapabilityGateMetadata,
  resolveModelCapabilitySummaryForSelection,
} from "@/lib/model/modelCapabilitySendGate";
import {
  mergeModelRequestPolicyMetadata,
  resolveModelRequestPolicyMetadataForSelection,
} from "@/lib/model/modelRequestPolicyMetadata";
import { MODEL_SELECTION_REQUIRED_ERROR_MESSAGE } from "../utils/agentRuntimeErrorPresentation";
import type {
  AssistantDraftState,
  AgentInputMention,
  SendMessageObserver,
  SessionModelPreference,
  ThreadGoalInput,
} from "./agentChatShared";
import type { AgentRuntimeAdapter } from "./agentRuntimeAdapter";
import type { AgentAccessMode } from "./agentChatStorage";
import type { StreamRequestState } from "./agentStreamSubmissionLifecycle";
import type { ActionRequired, Message, MessageImage } from "../types";
import type { ChatToolPreferences } from "../utils/chatToolPreferences";
import { runAgentStreamSubmitLifecycle } from "./agentStreamSubmitLifecycleController";
import { buildAgentStreamSubmitOp } from "./agentStreamSubmitOpController";
import { buildTurnInput } from "../utils/buildUserInputSubmitOp";
import { resolveAgentStreamSubmitContext } from "./agentStreamSubmitContext";
import { registerAgentStreamTurnEventBinding } from "./agentStreamTurnEventBinding";
import type { SoulInteractionCopy } from "@/lib/soul/interactionCopy";
import {
  extractAgentUiPerformanceTraceMetadata,
  mergeAgentUiPerformanceTraceMetadata,
} from "./agentStreamPerformanceMetrics";
import { resolveCollaborationModeMask } from "@/lib/api/collaborationModes";
import { resolveAllowedPermissionProfile } from "@/lib/api/permissionProfiles";
import { permissionProfileIdFromAccessMode } from "../utils/accessModeRuntime";

type MessageParts = NonNullable<Message["contentParts"]>;

interface ExecuteAgentStreamSubmitOptions {
  runtime: AgentRuntimeAdapter;
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
  refreshSessionReadModel: (targetSessionId?: string) => Promise<boolean>;
  sessionIdRef: MutableRefObject<string | null>;
  getWorkspaceIdForSubmit: () => string | undefined;
  getThreadIdForSubmit: (targetSessionId?: string) => string | undefined;
  getSyncedSessionExecutionStrategy: (
    sessionId: string,
  ) => AgentExecutionStrategy | null;
  getSyncedSessionRecentPreferences?: (
    sessionId: string,
  ) => ChatToolPreferences | null;
  effectiveAccessMode: AgentAccessMode;
  content: string;
  images: MessageImage[];
  inputMentions?: readonly AgentInputMention[];
  skipUserMessage: boolean;
  effectiveProviderType: string;
  effectiveModel: string;
  effectiveExecutionStrategy: AgentExecutionStrategy;
  modelOverride?: string;
  reasoningEffort?: string;
  webSearch?: boolean;
  thinking?: boolean;
  autoContinue?: AutoContinueRequestPayload;
  systemPrompt?: string;
  requestMetadata?: Record<string, unknown>;
  collaborationMode?: ModeKind;
  threadGoal?: ThreadGoalInput;
  assistantDraft?: AssistantDraftState;
  targetSessionId?: string;
  skipSessionRestore?: boolean;
  skipSessionStartHooks?: boolean;
  executionRuntime?: AgentSessionExecutionRuntime | null;
  syncedSessionModelPreference?: SessionModelPreference | null;
  eventName: string;
  clientUserMessageId?: string;
  requestState: StreamRequestState;
  assistantMsgId: string;
  pendingTurnKey: string;
  pendingItemKey: string;
  warnedKeysRef: MutableRefObject<Set<string>>;
  actionLoggedKeys: Set<string>;
  toolLogIdByToolId: Map<string, string>;
  toolStartedAtByToolId: Map<string, number>;
  toolNameByToolId: Map<string, string>;
  observer?: SendMessageObserver;
  onWriteFile?: (
    content: string,
    fileName: string,
    context?: import("../types").WriteArtifactContext,
  ) => void;
  callbacks: {
    activateStream: (
      activeSessionId: string,
      effectiveWaitingRuntimeStatus: NonNullable<Message["runtimeStatus"]>,
      canonicalThreadId?: string,
    ) => void;
    isStreamActivated: () => boolean;
    clearOptimisticItem: () => void;
    clearOptimisticTurn: () => void;
    disposeListener: () => void;
    clearActiveStreamIfMatch: (eventName: string) => boolean;
    registerListener: (unlisten: () => void) => void;
  };
  appendThinkingToParts: (
    parts: MessageParts,
    textDelta: string,
  ) => MessageParts;
  setMessages: Dispatch<SetStateAction<Message[]>>;
  setIsSending: Dispatch<SetStateAction<boolean>>;
  setPendingActions: Dispatch<SetStateAction<ActionRequired[]>>;
  getThreadItems?: () => readonly AgentThreadItem[];
  setThreadItems: Dispatch<SetStateAction<AgentThreadItem[]>>;
  setTodoItems?: Dispatch<SetStateAction<AgentTodoItem[]>>;
  setThreadTurns: Dispatch<SetStateAction<AgentThreadTurn[]>>;
  setCurrentTurnId: Dispatch<SetStateAction<string | null>>;
  setExecutionRuntime: Dispatch<
    SetStateAction<AgentSessionExecutionRuntime | null>
  >;
  soulCopy?: SoulInteractionCopy;
}

async function resolveSubmitModelPolicy(options: {
  images: readonly MessageImage[];
  providerType: string;
  model: string;
}): Promise<{
  modelCapabilitySummary: ModelCapabilitySummary | null | undefined;
  modelRequestPolicyMetadata: ReturnType<
    typeof resolveModelRequestPolicyMetadataForSelection
  >;
}> {
  try {
    const shouldResolveMediaCapability = options.images.length > 0;
    const models = await modelRegistryApi.getModelRegistry(
      shouldResolveMediaCapability ? { forceRefresh: true } : undefined,
    );
    const selection = {
      models,
      providerType: options.providerType,
      model: options.model,
    };

    return {
      modelCapabilitySummary: !shouldResolveMediaCapability
        ? undefined
        : resolveModelCapabilitySummaryForSelection(selection),
      modelRequestPolicyMetadata:
        resolveModelRequestPolicyMetadataForSelection(selection),
    };
  } catch {
    return {
      modelCapabilitySummary: options.images.length === 0 ? undefined : null,
      modelRequestPolicyMetadata: undefined,
    };
  }
}

async function setAcceptedThreadGoal(
  threadId: string,
  threadGoal: ThreadGoalInput | undefined,
): Promise<void> {
  if (!threadGoal) {
    return;
  }
  try {
    await setThreadGoal({
      threadId,
      objective: threadGoal.objective,
      status: "active",
      tokenBudget: threadGoal.tokenBudget,
    });
  } catch (error) {
    console.error(
      "[AgentStream] 用户 Turn 已接受，但设置 ThreadGoal 失败:",
      error,
    );
  }
}

export async function executeAgentStreamSubmit(
  options: ExecuteAgentStreamSubmitOptions,
) {
  const {
    runtime,
    ensureSession,
    attemptSilentTurnRecovery,
    refreshSessionReadModel,
    sessionIdRef,
    getWorkspaceIdForSubmit,
    getThreadIdForSubmit,
    getSyncedSessionExecutionStrategy,
    getSyncedSessionRecentPreferences,
    effectiveAccessMode,
    content,
    images,
    inputMentions,
    skipUserMessage,
    effectiveProviderType,
    effectiveModel,
    effectiveExecutionStrategy,
    modelOverride,
    reasoningEffort,
    webSearch,
    autoContinue,
    systemPrompt,
    requestMetadata,
    collaborationMode,
    threadGoal,
    assistantDraft,
    targetSessionId,
    skipSessionRestore,
    skipSessionStartHooks,
    executionRuntime,
    syncedSessionModelPreference,
    eventName,
    clientUserMessageId,
    requestState,
    assistantMsgId,
    pendingTurnKey,
    pendingItemKey,
    warnedKeysRef,
    actionLoggedKeys,
    toolLogIdByToolId,
    toolStartedAtByToolId,
    toolNameByToolId,
    observer,
    onWriteFile,
    callbacks,
    appendThinkingToParts,
    setMessages,
    setIsSending,
    setPendingActions,
    getThreadItems,
    setThreadItems,
    setTodoItems,
    setThreadTurns,
    setCurrentTurnId,
    setExecutionRuntime,
    soulCopy,
  } = options;

  if (!effectiveProviderType.trim() || !effectiveModel.trim()) {
    throw new Error(MODEL_SELECTION_REQUIRED_ERROR_MESSAGE);
  }
  let resolvedRequestMetadata = requestMetadata;
  let performanceTrace =
    extractAgentUiPerformanceTraceMetadata(requestMetadata);
  requestState.performanceTrace = performanceTrace;

  const {
    activeSessionId,
    resolvedWorkspaceId,
    syncedRecentPreferences,
    effectiveWaitingRuntimeStatus,
  } = await resolveAgentStreamSubmitContext({
    ensureSession,
    sessionIdRef,
    getWorkspaceIdForSubmit,
    getSyncedSessionRecentPreferences,
    getSyncedSessionExecutionStrategy,
    effectiveExecutionStrategy,
    assistantDraft,
    targetSessionId,
    skipSessionRestore,
    skipSessionStartHooks,
    performanceTrace,
    soulCopy,
  });
  const resolvedActiveSessionId = activeSessionId?.trim();
  if (!resolvedActiveSessionId) {
    throw new Error("缺少会话 ID，无法启动流式任务");
  }
  let resolvedThreadId = getThreadIdForSubmit(targetSessionId)?.trim();
  if (!resolvedThreadId) {
    await refreshSessionReadModel(resolvedActiveSessionId);
    resolvedThreadId = getThreadIdForSubmit(resolvedActiveSessionId)?.trim();
  }
  if (!resolvedThreadId) {
    throw new Error("缺少 canonical threadId，无法启动流式任务");
  }
  const resolvedExecutionRuntime =
    executionRuntime?.session_id === resolvedActiveSessionId
      ? executionRuntime
      : null;
  const turnControl = await runtime.getThreadTurnControl(resolvedThreadId);
  const expectedTurnId = turnControl.activeTurnId?.trim();
  if (expectedTurnId) {
    const { modelCapabilitySummary } = await resolveSubmitModelPolicy({
      images,
      providerType: effectiveProviderType,
      model: effectiveModel,
    });
    const params: TurnSteerParams = {
      threadId: resolvedThreadId,
      expectedTurnId,
      input: buildTurnInput({
        content,
        images,
        inputMentions,
        modelCapabilitySummary,
      }),
      ...(clientUserMessageId?.trim()
        ? { clientUserMessageId: clientUserMessageId.trim() }
        : {}),
    };
    if (threadGoal) {
      await setThreadGoal({
        threadId: resolvedThreadId,
        objective: threadGoal.objective,
        status: "active",
        tokenBudget: threadGoal.tokenBudget,
      });
    }
    callbacks.clearOptimisticItem();
    callbacks.clearOptimisticTurn();
    const response = await runtime.steerTurn(params);
    if (response.turnId !== expectedTurnId) {
      throw new Error(
        `turn/steer returned mismatched turnId: expected ${expectedTurnId}, received ${response.turnId || "<empty>"}`,
      );
    }
    await refreshSessionReadModel(resolvedActiveSessionId);
    setMessages((previous) =>
      previous.flatMap((message) => {
        if (message.id === assistantMsgId) {
          return [];
        }
        return [
          message.id === clientUserMessageId
            ? { ...message, runtimeTurnId: expectedTurnId }
            : message,
        ];
      }),
    );
    return;
  }
  callbacks.activateStream(
    resolvedActiveSessionId,
    effectiveWaitingRuntimeStatus,
    resolvedThreadId,
  );
  setIsSending(true);
  if (performanceTrace) {
    resolvedRequestMetadata = mergeAgentUiPerformanceTraceMetadata(
      requestMetadata,
      {
        ...performanceTrace,
        sessionId: resolvedActiveSessionId,
        workspaceId: resolvedWorkspaceId ?? performanceTrace.workspaceId,
      },
    );
    performanceTrace = extractAgentUiPerformanceTraceMetadata(
      resolvedRequestMetadata,
    );
    requestState.performanceTrace = performanceTrace;
  }
  const preserveAssistantContent =
    assistantDraft?.preserveContent === true
      ? assistantDraft.content?.trim() || null
      : null;
  const assistantFallbackContent =
    assistantDraft?.fallbackContent === undefined
      ? null
      : assistantDraft.fallbackContent.trim();
  const eventBindingWorkspaceId = resolvedWorkspaceId ?? "";

  const unlisten = await registerAgentStreamTurnEventBinding({
    runtime,
    eventName,
    requestState,
    attemptSilentTurnRecovery,
    skipUserMessage,
    effectiveProviderType,
    effectiveModel,
    effectiveExecutionStrategy,
    systemPrompt,
    content,
    webSearch,
    autoContinue,
    activeSessionId: resolvedActiveSessionId,
    resolvedWorkspaceId: eventBindingWorkspaceId,
    assistantMsgId,
    pendingTurnKey,
    pendingItemKey,
    effectiveWaitingRuntimeStatus,
    preserveAssistantContent,
    assistantFallbackContent,
    warnedKeysRef,
    actionLoggedKeys,
    toolLogIdByToolId,
    toolStartedAtByToolId,
    toolNameByToolId,
    observer,
    onWriteFile,
    callbacks: {
      activateStream: callbacks.activateStream,
      isStreamActivated: callbacks.isStreamActivated,
      clearOptimisticItem: callbacks.clearOptimisticItem,
      clearOptimisticTurn: callbacks.clearOptimisticTurn,
      disposeListener: callbacks.disposeListener,
      clearActiveStreamIfMatch: callbacks.clearActiveStreamIfMatch,
    },
    appendThinkingToParts,
    setMessages,
    setPendingActions,
    getThreadItems,
    setThreadItems,
    setTodoItems,
    setThreadTurns,
    setCurrentTurnId,
    setExecutionRuntime,
    setIsSending,
    soulCopy,
  });

  callbacks.registerListener(unlisten);

  await runAgentStreamSubmitLifecycle({
    activeSessionId: resolvedActiveSessionId,
    effectiveModel,
    effectiveProviderType,
    eventName,
    onSubmitAccepted: () => {
      requestState.startTerminalRecoveryPoll?.();
    },
    requestState,
    submit: async () => {
      const { modelCapabilitySummary, modelRequestPolicyMetadata } =
        await resolveSubmitModelPolicy({
          images,
          providerType: effectiveProviderType,
          model: effectiveModel,
        });
      resolvedRequestMetadata = mergeModelRequestPolicyMetadata(
        resolvedRequestMetadata,
        modelRequestPolicyMetadata,
      );
      if (modelCapabilitySummary !== undefined) {
        resolvedRequestMetadata = mergeModelInputCapabilityGateMetadata(
          resolvedRequestMetadata,
          evaluateModelInputCapability(
            modelCapabilitySummary,
            buildModelCapabilitySendGateInput({
              text: content,
              imageCount: images.length,
            }),
          ),
        );
      }
      const collaborationModePreset = collaborationMode
        ? await resolveCollaborationModeMask(collaborationMode)
        : undefined;
      await resolveAllowedPermissionProfile(
        permissionProfileIdFromAccessMode(effectiveAccessMode),
      );
      const submitOp = buildAgentStreamSubmitOp({
        content,
        images,
        inputMentions,
        activeThreadId: resolvedThreadId,
        clientUserMessageId,
        eventName,
        requestMetadata: resolvedRequestMetadata,
        collaborationMode: collaborationModePreset,
        executionRuntime: resolvedExecutionRuntime,
        syncedRecentPreferences,
        syncedSessionModelPreference,
        effectiveAccessMode,
        effectiveProviderType,
        effectiveModel,
        modelOverride,
        reasoningEffort,
        modelCapabilitySummary,
      });

      await runtime.submitOp(submitOp);
      await setAcceptedThreadGoal(resolvedThreadId, threadGoal);
    },
  });
}
