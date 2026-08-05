import type { AgentExecutionStrategy } from "@/lib/api/agentExecutionRuntime";
import type { AutoContinueRequestPayload } from "@/lib/api/agentRuntime/sessionTypes";
import type { ModeKind } from "@limecloud/app-server-client";
import type {
  AssistantDraftState,
  AgentInputMention,
  SendMessageObserver,
  SendMessageOptions,
  SessionModelPreference,
  ThreadGoalInput,
} from "./agentChatShared";
import type { Message, MessageImage } from "../types";
import { prepareAgentStreamSubmitDraft } from "./agentStreamSubmitDraft";
import type { AgentStreamPreparedSendEnv } from "./agentStreamPreparedSendEnv";
import type { InterruptedInputDraftSnapshot } from "./agentStreamInputRestoreTypes";
import { normalizeExecutionStrategy } from "./agentChatCoreUtils";
import { ensureAgentUiPerformanceTraceMetadata } from "./agentStreamPerformanceMetrics";
import {
  buildModelCapabilitySendGateInput,
  evaluateModelInputCapability,
  mergeModelInputCapabilityGateMetadata,
  type ModelCapabilitySendGateResult,
} from "@/lib/model/modelCapabilitySendGate";

export type AgentStreamUserInputSendPreparationEnv = Pick<
  AgentStreamPreparedSendEnv,
  | "executionStrategy"
  | "providerTypeRef"
  | "modelRef"
  | "reasoningEffortRef"
  | "sessionIdRef"
  | "executionRuntime"
  | "clawTraceEnabled"
  | "soulCopy"
  | "getWorkspaceIdForSubmit"
  | "getSyncedSessionModelPreference"
  | "setMessages"
  | "setIsSending"
>;

interface PrepareAgentStreamUserInputSendOptions {
  content: string;
  images: MessageImage[];
  webSearch?: boolean;
  thinking?: boolean;
  skipUserMessage: boolean;
  executionStrategyOverride?: AgentExecutionStrategy;
  modelOverride?: string;
  autoContinue?: AutoContinueRequestPayload;
  systemPrompt?: string;
  options?: SendMessageOptions;
  env: AgentStreamUserInputSendPreparationEnv;
}

export interface PreparedAgentStreamUserInputSend {
  content: string;
  images: MessageImage[];
  webSearch?: boolean;
  thinking?: boolean;
  skipUserMessage: boolean;
  effectiveExecutionStrategy: AgentExecutionStrategy;
  effectiveProviderType: string;
  effectiveModel: string;
  modelOverride?: string;
  reasoningEffort?: string;
  autoContinue?: AutoContinueRequestPayload;
  systemPrompt?: string;
  syncedSessionModelPreference: SessionModelPreference | null;
  observer?: SendMessageObserver;
  requestMetadata?: Record<string, unknown>;
  collaborationMode?: ModeKind;
  threadGoal?: ThreadGoalInput;
  modelInputCapabilityGate?: ModelCapabilitySendGateResult;
  assistantDraft?: AssistantDraftState;
  skillRequest?: SendMessageOptions["skillRequest"];
  targetSessionId?: string;
  skipSessionRestore?: boolean;
  skipSessionStartHooks?: boolean;
  assistantMsgId: string;
  userMsgId: string | null;
  userMsg: Message | null;
  assistantMsg: Message;
  submittedDraft?: InterruptedInputDraftSnapshot | null;
  inputMentions?: AgentInputMention[];
}

function shouldProjectModelInputCapabilityGate(
  gate: ModelCapabilitySendGateResult,
  summary: SendMessageOptions["modelCapabilitySummary"],
): boolean {
  if (gate.requiredInputModalities.length === 0) {
    return false;
  }

  return Boolean(summary) || gate.requiresMediaInput;
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function cloneArrayField(value: readonly unknown[] | undefined): unknown[] {
  return value ? [...value] : [];
}

function withInterruptedInputRestoreMetadata(
  metadata: Record<string, unknown> | undefined,
  draft: InterruptedInputDraftSnapshot,
): Record<string, unknown> | undefined {
  const textElements = cloneArrayField(draft.textElements);
  const pathReferences = cloneArrayField(draft.pathReferences);
  const inputCapabilityRoute = draft.inputCapabilityRoute;
  if (
    textElements.length === 0 &&
    pathReferences.length === 0 &&
    !inputCapabilityRoute
  ) {
    return metadata;
  }

  const base = metadata ? { ...metadata } : {};
  const harness = isPlainRecord(base.harness) ? { ...base.harness } : {};

  if (textElements.length > 0) {
    base.text_elements = textElements;
    base.textElements = textElements;
    harness.text_elements = textElements;
    harness.textElements = textElements;
  }
  if (pathReferences.length > 0) {
    base.path_references = pathReferences;
    base.pathReferences = pathReferences;
    harness.file_references = pathReferences;
    harness.fileReferences = pathReferences;
  }
  if (inputCapabilityRoute) {
    base.input_capability_route = inputCapabilityRoute;
    base.inputCapabilityRoute = inputCapabilityRoute;
    harness.input_capability_route = inputCapabilityRoute;
    harness.inputCapabilityRoute = inputCapabilityRoute;
  }

  return {
    ...base,
    harness,
  };
}

interface SendModelPreferenceCandidate {
  providerType?: string | null;
  model?: string | null;
}

function normalizeSendModelPreferenceValue(value?: string | null): string {
  return (value || "").trim();
}

function normalizeProviderIdentity(value?: string | null): string {
  return normalizeSendModelPreferenceValue(value).toLowerCase();
}

function toCompleteSendModelPreference(
  candidate: SendModelPreferenceCandidate,
): SessionModelPreference | null {
  const providerType = normalizeSendModelPreferenceValue(
    candidate.providerType,
  );
  const model = normalizeSendModelPreferenceValue(candidate.model);
  return providerType && model ? { providerType, model } : null;
}

function resolveModelForProvider(
  providerType: string,
  candidates: SendModelPreferenceCandidate[],
): string {
  const normalizedProvider = normalizeProviderIdentity(providerType);
  if (!normalizedProvider) {
    return "";
  }

  for (const candidate of candidates) {
    if (
      normalizeProviderIdentity(candidate.providerType) !== normalizedProvider
    ) {
      continue;
    }
    const model = normalizeSendModelPreferenceValue(candidate.model);
    if (model) {
      return model;
    }
  }

  return "";
}

function resolvePreparedSendModelPreference(options: {
  providerOverride?: string | null;
  modelOverride?: string | null;
  currentProviderType?: string | null;
  currentModel?: string | null;
  runtimeProviderType?: string | null;
  runtimeModel?: string | null;
  syncedSessionModelPreference?: SessionModelPreference | null;
}): SessionModelPreference | null {
  const providerOverride = normalizeSendModelPreferenceValue(
    options.providerOverride,
  );
  const modelOverride = normalizeSendModelPreferenceValue(
    options.modelOverride,
  );
  const currentCandidate: SendModelPreferenceCandidate = {
    providerType: options.currentProviderType,
    model: options.currentModel,
  };
  const runtimeCandidate: SendModelPreferenceCandidate = {
    providerType: options.runtimeProviderType,
    model: options.runtimeModel,
  };
  const syncedCandidate: SendModelPreferenceCandidate | null =
    options.syncedSessionModelPreference
      ? {
          providerType: options.syncedSessionModelPreference.providerType,
          model: options.syncedSessionModelPreference.model,
        }
      : null;
  const fallbackCandidates = [
    currentCandidate,
    runtimeCandidate,
    syncedCandidate,
  ].filter(
    (candidate): candidate is SendModelPreferenceCandidate =>
      candidate !== null,
  );

  if (providerOverride && modelOverride) {
    return { providerType: providerOverride, model: modelOverride };
  }

  if (providerOverride) {
    const matchedModel = resolveModelForProvider(
      providerOverride,
      fallbackCandidates,
    );
    return matchedModel
      ? { providerType: providerOverride, model: matchedModel }
      : null;
  }

  if (modelOverride) {
    const providerType = normalizeSendModelPreferenceValue(
      options.currentProviderType ||
        options.runtimeProviderType ||
        syncedCandidate?.providerType,
    );
    return providerType ? { providerType, model: modelOverride } : null;
  }

  for (const candidate of fallbackCandidates) {
    const completePreference = toCompleteSendModelPreference(candidate);
    if (completePreference) {
      return completePreference;
    }
  }

  return null;
}

export function prepareAgentStreamUserInputSend(
  options: PrepareAgentStreamUserInputSendOptions,
): PreparedAgentStreamUserInputSend {
  const {
    content,
    images,
    webSearch,
    thinking,
    skipUserMessage,
    executionStrategyOverride,
    modelOverride,
    autoContinue,
    systemPrompt,
    options: sendOptions,
    env,
  } = options;

  const effectiveExecutionStrategy = normalizeExecutionStrategy(
    executionStrategyOverride || env.executionStrategy,
  );
  const resolvedProviderOverride = sendOptions?.providerOverride?.trim();
  const resolvedModelOverride =
    sendOptions?.modelOverride?.trim() || modelOverride?.trim();
  const resolvedReasoningEffort =
    sendOptions?.reasoningEffort?.trim() ||
    env.reasoningEffortRef.current.trim();
  const currentSessionId = env.sessionIdRef.current;
  const targetSessionId = sendOptions?.targetSessionId?.trim() || undefined;
  const sessionIdForSend = targetSessionId ?? currentSessionId;
  const syncedSessionModelPreference = sessionIdForSend
    ? env.getSyncedSessionModelPreference(sessionIdForSend)
    : null;
  const currentProviderType = env.providerTypeRef.current.trim();
  const currentModel = env.modelRef.current.trim();
  const runtimeProviderType =
    env.executionRuntime?.provider_selector?.trim() ||
    env.executionRuntime?.provider_name?.trim() ||
    "";
  const runtimeModel = env.executionRuntime?.model_name?.trim() || "";
  const resolvedModelPreference = resolvePreparedSendModelPreference({
    providerOverride: resolvedProviderOverride,
    modelOverride: resolvedModelOverride,
    currentProviderType,
    currentModel,
    runtimeProviderType,
    runtimeModel,
    syncedSessionModelPreference,
  });
  const effectiveProviderType = resolvedModelPreference?.providerType ?? "";
  const effectiveModel = resolvedModelPreference?.model ?? "";
  const observer = sendOptions?.observer;
  const assistantMsgId = crypto.randomUUID();
  const userMsgId = skipUserMessage ? null : crypto.randomUUID();
  const baseRequestMetadata = ensureAgentUiPerformanceTraceMetadata(
    sendOptions?.requestMetadata,
    {
      enabled: true,
      sessionId: sessionIdForSend,
      source: "agent-chat",
      submittedAt: Date.now(),
      workspaceId: env.getWorkspaceIdForSubmit(),
    },
  );
  const modelCapabilityGateInput = buildModelCapabilitySendGateInput({
    text: content,
    imageCount: images.length,
  });
  const modelInputCapabilityGate = evaluateModelInputCapability(
    sendOptions?.modelCapabilitySummary,
    modelCapabilityGateInput,
  );
  const projectedModelInputCapabilityGate =
    shouldProjectModelInputCapabilityGate(
      modelInputCapabilityGate,
      sendOptions?.modelCapabilitySummary,
    )
      ? modelInputCapabilityGate
      : undefined;
  const baseRequestMetadataWithModelGate =
    mergeModelInputCapabilityGateMetadata(
      baseRequestMetadata,
      projectedModelInputCapabilityGate,
    );
  const messagePurpose = sendOptions?.purpose;
  const assistantDraft = sendOptions?.assistantDraft;
  const capabilityRoute = sendOptions?.capabilityRoute;
  const skipSessionRestore = sendOptions?.skipSessionRestore === true;
  const skipSessionStartHooks = sendOptions?.skipSessionStartHooks === true;
  const resolvedSystemPrompt =
    sendOptions?.systemPromptOverride?.trim() || systemPrompt;
  const displayContent = sendOptions?.displayContent;
  const submittedDraft: InterruptedInputDraftSnapshot =
    sendOptions?.inputRestoreDraft ?? {
      text: displayContent ?? content,
      images,
      inputCapabilityRoute: capabilityRoute,
    };
  const requestMetadata = withInterruptedInputRestoreMetadata(
    baseRequestMetadataWithModelGate,
    submittedDraft,
  );
  const skillRequest = sendOptions?.skillRequest;
  const inputMentions = (sendOptions?.inputMentions ?? []).map((mention) => ({
    ...mention,
  }));
  const { assistantMsg, userMsg } = prepareAgentStreamSubmitDraft({
    content,
    displayContent,
    images,
    skipUserMessage,
    assistantMsgId,
    userMsgId,
    assistantDraft,
    requestMetadata,
    messagePurpose,
    capabilityRoute,
    effectiveExecutionStrategy,
    soulCopy: env.soulCopy,
    setMessages: env.setMessages,
    setIsSending: env.setIsSending,
  });

  return {
    content,
    images,
    webSearch,
    thinking,
    skipUserMessage,
    effectiveExecutionStrategy,
    effectiveProviderType,
    effectiveModel,
    modelOverride: resolvedModelOverride,
    reasoningEffort: resolvedReasoningEffort || undefined,
    autoContinue,
    systemPrompt: resolvedSystemPrompt,
    syncedSessionModelPreference,
    observer,
    requestMetadata,
    ...(sendOptions?.collaborationMode
      ? { collaborationMode: sendOptions.collaborationMode }
      : {}),
    ...(sendOptions?.threadGoal ? { threadGoal: sendOptions.threadGoal } : {}),
    modelInputCapabilityGate: projectedModelInputCapabilityGate,
    assistantDraft,
    skillRequest,
    targetSessionId,
    skipSessionRestore,
    skipSessionStartHooks,
    assistantMsgId,
    userMsgId,
    userMsg,
    assistantMsg,
    submittedDraft,
    inputMentions,
  };
}
