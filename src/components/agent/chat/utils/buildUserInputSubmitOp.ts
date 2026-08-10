import {
  createApplicationAdditionalContext,
  type AgentUserInputOp,
} from "@/lib/api/agentProtocolOps";
import type { ModelCapabilitySummary } from "@/lib/model/inferModelCapabilities";
import {
  assertModelInputCapabilityAllowed,
  buildModelCapabilitySendGateInput,
} from "@/lib/model/modelCapabilitySendGate";
import type { AgentSessionExecutionRuntime } from "@/lib/api/agentExecutionRuntime";
import type {
  CollaborationMode,
  CollaborationModeMask,
} from "@limecloud/app-server-client";
import type { AgentAccessMode } from "../hooks/agentChatStorage";
import type { SessionModelPreference } from "../hooks/agentChatShared";
import type { AgentInputMention } from "../hooks/agentChatShared";
import type { MessageImage } from "../types";
import type { ChatToolPreferences } from "./chatToolPreferences";
import {
  createRuntimePoliciesFromAccessMode,
  permissionProfileIdFromAccessMode,
} from "./accessModeRuntime";
import { buildMessageImageDataUrl } from "./imageAttachments";
import { buildSubmitOpRuntimeCompaction } from "./submitOpRuntimeCompaction";

export interface BuildUserInputSubmitOpOptions {
  content: string;
  images: MessageImage[];
  threadId?: string;
  clientUserMessageId?: string;
  eventName: string;
  requestMetadata?: Record<string, unknown>;
  collaborationMode?: CollaborationModeMask;
  executionRuntime?: AgentSessionExecutionRuntime | null;
  syncedRecentPreferences?: ChatToolPreferences | null;
  syncedSessionModelPreference?: SessionModelPreference | null;
  effectiveAccessMode: AgentAccessMode;
  effectiveProviderType: string;
  effectiveModel: string;
  modelOverride?: string;
  reasoningEffort?: string;
  modelCapabilitySummary?: ModelCapabilitySummary | null;
  inputMentions?: readonly AgentInputMention[];
}

export interface BuildTurnInputOptions {
  content: string;
  images: MessageImage[];
  modelCapabilitySummary?: ModelCapabilitySummary | null;
  inputMentions?: readonly AgentInputMention[];
}

function buildCollaborationMode(
  preset: CollaborationModeMask | undefined,
  model: string,
  reasoningEffort: string | undefined,
): CollaborationMode | undefined {
  if (!preset) {
    return undefined;
  }
  if (!preset.mode) {
    throw new Error("collaboration mode preset must include mode");
  }

  return {
    mode: preset.mode,
    settings: {
      model: preset.model?.trim() || model,
      reasoning_effort:
        preset.reasoning_effort?.trim() || reasoningEffort?.trim() || null,
      developer_instructions: null,
    },
  };
}

export function buildTurnInput(
  options: BuildTurnInputOptions,
): AgentUserInputOp["turn"]["input"] {
  const {
    content,
    images,
    inputMentions = [],
    modelCapabilitySummary,
  } = options;
  if (modelCapabilitySummary !== undefined) {
    assertModelInputCapabilityAllowed(
      modelCapabilitySummary,
      buildModelCapabilitySendGateInput({
        text: content,
        imageCount: images.length,
      }),
      { failClosedOnUnknown: false },
    );
  }

  return [
    { type: "text", text: content },
    ...inputMentions.map((mention) => ({ ...mention })),
    ...images.map((image) => ({
      type: "image" as const,
      url: buildMessageImageDataUrl(image),
    })),
  ];
}

export function buildUserInputSubmitOp(
  options: BuildUserInputSubmitOpOptions,
): AgentUserInputOp {
  const {
    content,
    images,
    threadId,
    clientUserMessageId,
    eventName,
    requestMetadata,
    collaborationMode: collaborationModePreset,
    executionRuntime,
    syncedRecentPreferences,
    syncedSessionModelPreference,
    effectiveAccessMode,
    effectiveProviderType,
    effectiveModel,
    modelOverride,
    reasoningEffort,
    modelCapabilitySummary,
    inputMentions,
  } = options;

  const turnModel = modelOverride?.trim() || effectiveModel.trim();
  const compaction = buildSubmitOpRuntimeCompaction({
    requestMetadata,
    executionRuntime,
    syncedRecentPreferences,
    syncedSessionModelPreference,
    effectiveProviderType,
    effectiveModel: turnModel,
  });
  const runtimePolicies =
    createRuntimePoliciesFromAccessMode(effectiveAccessMode);
  const currentThreadId = threadId?.trim();
  if (!currentThreadId) {
    throw new Error("threadId is required to build App Server turn/start");
  }
  const collaborationMode = buildCollaborationMode(
    collaborationModePreset,
    turnModel,
    reasoningEffort,
  );
  const turnReasoningEffort =
    collaborationMode?.settings.reasoning_effort ?? reasoningEffort?.trim();
  const additionalContext = createApplicationAdditionalContext({
    metadata: compaction.metadata,
  });

  return {
    type: "user_input",
    eventName,
    turn: {
      threadId: currentThreadId,
      ...(clientUserMessageId?.trim()
        ? { clientUserMessageId: clientUserMessageId.trim() }
        : {}),
      input: buildTurnInput({
        content,
        images,
        inputMentions,
        modelCapabilitySummary,
      }),
      ...(collaborationMode ? { collaborationMode } : {}),
      ...(compaction.shouldSubmitModel ? { model: turnModel } : {}),
      ...(turnReasoningEffort ? { effort: turnReasoningEffort } : {}),
      approvalPolicy: runtimePolicies.approvalPolicy,
      permissions: permissionProfileIdFromAccessMode(effectiveAccessMode),
      ...(Object.keys(additionalContext).length > 0
        ? { additionalContext }
        : {}),
    },
  };
}
