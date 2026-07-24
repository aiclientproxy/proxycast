import type { AgentEvent, AgentThreadItem } from "@/lib/api/agentProtocol";
import type { Message } from "../types";
import {
  applyAcknowledgedActionRequests,
  removeActionsByRequestIds,
} from "./agentChatActionState";
import { appendTextToParts } from "./agentChatHistory";
import {
  removeThreadItemState,
  removeThreadTurnState,
  upsertThreadItemState,
  upsertThreadTurnState,
} from "./agentThreadState";
import {
  isStreamedReasoningTimelineItem,
  removeStreamedReasoningTimelineItems,
  resetStreamedReasoningSegment,
} from "./agentStreamReasoningTimeline";
import {
  buildAgentStreamTerminalCompletionPlan,
  buildAgentStreamMissingFinalReplyFailurePlan,
  type AgentStreamTerminalCompletionPlan,
  type AgentStreamMissingFinalReplyPlan,
} from "./agentStreamCompletionController";
import {
  isPersistedReasoningContentPart,
  syncAssistantReasoningContentPartFromThreadItem,
} from "./agentStreamReasoningContentSync";
import { syncAssistantAgentMessageContentPartFromThreadItem } from "./agentStreamAgentMessageContentSync";
import { syncMessageToolCallFromThreadItem } from "./agentStreamToolItemMessageSync";
import { buildAgentStreamTurnStartedPendingItemUpdate } from "./agentStreamThreadItemController";
import {
  bindAssistantMessageToRuntimeTurn,
  extractVisibleTextFromAgentMessage,
} from "./agentStreamRuntimeHandlerUtils";
import { resolveAccumulatedFinalContentForCompletion } from "./agentStreamTextDeltaLifecycle";
import {
  isAgentMessageFinalAnswerPhase,
  shouldUseAgentMessageAsFinalText,
} from "../utils/agentMessagePhase";
import { resolveAgentRuntimeErrorPresentation } from "../utils/agentRuntimeErrorPresentation";
import type {
  HandleTurnStreamEventOptions,
  StreamRequestState,
} from "./agentStreamRuntimeHandlerTypes";

type RuntimeHandlerStateSetters = Pick<
  HandleTurnStreamEventOptions,
  | "getThreadItems"
  | "setPendingActions"
  | "setCurrentTurnId"
  | "setMessages"
  | "setThreadItems"
  | "setThreadTurns"
>;

type AgentStreamCompletionCompletePlan = Extract<
  AgentStreamTerminalCompletionPlan,
  { type: "complete" }
>;

type AgentStreamCompletionMessagePlan = Pick<
  AgentStreamCompletionCompletePlan,
  "finalContent" | "requestLogPayload"
> & {
  usage?: Message["usage"];
};

function noteCompletedTextAsAssistantReplyIfNeeded(
  requestState: StreamRequestState,
) {
  if (!requestState.hasFinalAnswerRequiredProcessBoundary) {
    return;
  }
  requestState.hasAssistantTextAfterLatestFinalAnswerRequiredProcessBoundary = true;
}

function syncFinalAgentMessageSnapshotToRequestState(params: {
  event: Extract<
    AgentEvent,
    { type: "item_started" | "item_completed" | "item_updated" }
  >;
  requestState: StreamRequestState;
  shouldPreserveAssistantContent?: boolean;
}): void {
  const item = params.event.item;
  if (
    params.shouldPreserveAssistantContent ||
    item.type !== "agent_message" ||
    !isAgentMessageFinalAnswerPhase(item.phase) ||
    !item.text.trim()
  ) {
    return;
  }

  const existingContent = params.requestState.accumulatedContent;
  const isSameActiveItem =
    params.requestState.activeTextSegmentItemId === item.id;
  const nextContent =
    isSameActiveItem &&
    existingContent.length > item.text.length &&
    existingContent.startsWith(item.text)
      ? existingContent
      : item.text;
  params.requestState.accumulatedContent = nextContent;
  params.requestState.renderedContent = nextContent;
  params.requestState.activeTextSegmentItemId = item.id;
  params.requestState.activeTextSegmentPhase = item.phase ?? null;
  params.requestState.activeTextSegmentSequence = item.sequence;
  params.requestState.activeTextSegmentTurnId = item.turn_id;
  params.requestState.activeTextSegmentStartOffset = 0;
  params.requestState.activeTextSegmentFinalEligibility = "explicit_final";
  params.requestState.latestAssistantTextEventSequence = Math.max(
    params.requestState.latestAssistantTextEventSequence ??
      Number.NEGATIVE_INFINITY,
    item.sequence,
  );

  const latestProcessSequence =
    params.requestState.maxFinalAnswerRequiredProcessEventSequence ??
    params.requestState.maxProcessEventSequence;
  if (
    params.requestState.hasFinalAnswerRequiredProcessBoundary &&
    (typeof latestProcessSequence !== "number" ||
      item.sequence > latestProcessSequence)
  ) {
    params.requestState.hasAssistantTextAfterLatestFinalAnswerRequiredProcessBoundary = true;
  }
}

export function handleAgentStreamMessageSnapshotEvent(params: {
  assistantMsgId: string;
  event: Extract<AgentEvent, { type: "message" }>;
  requestState: StreamRequestState;
  setMessages: RuntimeHandlerStateSetters["setMessages"];
  shouldPreserveAssistantContent: boolean;
  surfaceThinkingDeltas: boolean;
}): void {
  const snapshotText = extractVisibleTextFromAgentMessage(params.event.message);
  const shouldPrefillVisibleText =
    !params.shouldPreserveAssistantContent &&
    params.event.message.role === "assistant" &&
    !params.requestState.renderedContent &&
    !params.requestState.accumulatedContent &&
    snapshotText.trim().length > 0;
  if (!shouldPrefillVisibleText) {
    return;
  }

  params.requestState.accumulatedContent = snapshotText;
  params.requestState.renderedContent = snapshotText;
  params.requestState.prefilledMessageSnapshotReplayOffset = 0;
  params.requestState.prefilledMessageSnapshotText = snapshotText;
  if (!params.requestState.firstTextRenderFlushAt) {
    params.requestState.firstTextRenderFlushAt = Date.now();
  }
  params.setMessages((prev) =>
    prev.map((msg) =>
      msg.id === params.assistantMsgId
        ? {
            ...msg,
            content: snapshotText,
            contentParts: appendTextToParts(
              params.surfaceThinkingDeltas
                ? msg.contentParts || []
                : (msg.contentParts || []).filter(
                    (part) =>
                      part.type !== "thinking" ||
                      isPersistedReasoningContentPart(part),
                  ),
              snapshotText,
            ),
          }
        : msg,
    ),
  );
}

export function handleAgentStreamTurnStartedEvent(params: {
  assistantMsgId: string;
  event: Extract<AgentEvent, { type: "turn_started" }>;
  pendingItemKey: string;
  pendingTurnKey: string;
  requestState: StreamRequestState;
  setters: RuntimeHandlerStateSetters;
}): void {
  params.requestState.currentTurnId = params.event.turn.id;
  bindAssistantMessageToRuntimeTurn(
    params.setters.setMessages,
    params.assistantMsgId,
    params.event.turn.id,
  );
  params.setters.setCurrentTurnId(params.event.turn.id);
  params.setters.setThreadTurns((prev) =>
    upsertThreadTurnState(
      removeThreadTurnState(prev, params.pendingTurnKey),
      params.event.turn,
    ),
  );
  params.setters.setThreadItems((prev) => {
    const pendingItem = prev.find((item) => item.id === params.pendingItemKey);
    const updatedPendingItem = buildAgentStreamTurnStartedPendingItemUpdate({
      pendingItem,
      turn: params.event.turn,
    });
    if (!updatedPendingItem) {
      return prev;
    }

    return upsertThreadItemState(
      removeThreadItemState(prev, params.pendingItemKey),
      updatedPendingItem,
    );
  });
}

export function handleAgentStreamThreadItemLifecycleEvent(params: {
  assistantMsgId: string;
  event: Extract<
    AgentEvent,
    { type: "item_started" | "item_completed" | "item_updated" }
  >;
  pendingItemKey: string;
  requestState: StreamRequestState;
  shouldPreserveAssistantContent?: boolean;
  setters: RuntimeHandlerStateSetters;
}): void {
  params.requestState.currentTurnId = params.event.item.turn_id;
  bindAssistantMessageToRuntimeTurn(
    params.setters.setMessages,
    params.assistantMsgId,
    params.event.item.turn_id,
  );
  let nextThreadItemsForSync: readonly AgentThreadItem[] =
    params.setters.getThreadItems?.() ?? [];
  params.setters.setThreadItems((prev) => {
    const nextItems = upsertThreadItemState(
      params.event.item.type === "reasoning"
        ? removeStreamedReasoningTimelineItems(
            removeThreadItemState(prev, params.pendingItemKey),
            params.event.item.turn_id,
          )
        : removeThreadItemState(prev, params.pendingItemKey),
      params.event.item,
    );
    nextThreadItemsForSync = nextItems;
    return nextItems;
  });
  syncMessageToolCallFromThreadItem({
    assistantMsgId: params.assistantMsgId,
    item: params.event.item,
    setMessages: params.setters.setMessages,
  });
  syncAssistantReasoningContentPartFromThreadItem({
    assistantMsgId: params.assistantMsgId,
    item: params.event.item,
    threadItems: nextThreadItemsForSync,
    setMessages: params.setters.setMessages,
  });
  syncAssistantAgentMessageContentPartFromThreadItem({
    assistantMsgId: params.assistantMsgId,
    item: params.event.item,
    threadItems: nextThreadItemsForSync,
    setMessages: params.setters.setMessages,
  });
  syncFinalAgentMessageSnapshotToRequestState(params);
  if (
    params.event.item.type === "approval_request" &&
    (params.event.item.status === "completed" ||
      params.event.item.status === "failed")
  ) {
    const requestIds = new Set([params.event.item.request_id]);
    params.setters.setPendingActions((prev) =>
      removeActionsByRequestIds(prev, requestIds),
    );
    params.setters.setMessages((prev) =>
      applyAcknowledgedActionRequests({
        messages: prev,
        requestIds,
        shouldPersistSubmittedAction: false,
      }),
    );
  }
}

export function handleAgentStreamTurnCompletedEvent(params: {
  assistantFallbackContent?: string | null;
  assistantMsgId: string;
  completeAssistantStreamMessageFromCompletionPlan: (
    plan: AgentStreamCompletionMessagePlan,
  ) => void;
  event: Extract<AgentEvent, { type: "turn_completed" }>;
  finalizeMissingFinalReplyFailure: (
    plan: AgentStreamMissingFinalReplyPlan,
  ) => void;
  pendingTurnKey: string;
  requestState: StreamRequestState;
  shouldPreserveAssistantContent: boolean;
  setters: RuntimeHandlerStateSetters;
  toolCallCount: number;
}): void {
  params.requestState.currentTurnId = params.event.turn.id;
  bindAssistantMessageToRuntimeTurn(
    params.setters.setMessages,
    params.assistantMsgId,
    params.event.turn.id,
  );
  params.setters.setThreadTurns((prev) =>
    upsertThreadTurnState(
      removeThreadTurnState(prev, params.pendingTurnKey),
      params.event.turn,
    ),
  );
  params.setters.setCurrentTurnId(params.event.turn.id);
  const completedAt =
    params.event.turn.completed_at || new Date().toISOString();
  params.setters.setThreadItems((prev) =>
    prev.map((item) =>
      isStreamedReasoningTimelineItem(item, params.event.turn.id) ||
      (item.type === "agent_message" &&
        item.turn_id === params.event.turn.id &&
        !shouldUseAgentMessageAsFinalText(item.phase))
        ? {
            ...item,
            status: "completed",
            completed_at: item.completed_at || completedAt,
            updated_at: completedAt,
          }
        : item,
    ),
  );
  resetStreamedReasoningSegment(params.requestState);
  if (params.event.text?.trim() && !params.shouldPreserveAssistantContent) {
    const completedText = params.event.text;
    const existingContent = params.requestState.accumulatedContent;
    const shouldPreferCompletedText =
      Boolean(params.requestState.hasFinalAnswerRequiredProcessBoundary) &&
      completedText.trim().length > 0;
    const completedTextAdopted =
      shouldPreferCompletedText || !existingContent.trim()
        ? true
        : completedText.startsWith(existingContent) &&
          completedText.length > existingContent.length;
    const nextContent = shouldPreferCompletedText
      ? completedText
      : !existingContent.trim()
        ? completedText
        : completedText.startsWith(existingContent)
          ? completedText
          : existingContent;
    params.requestState.accumulatedContent = nextContent;
    params.requestState.renderedContent = nextContent;
    if (
      completedTextAdopted ||
      params.requestState.hasFinalAnswerRequiredProcessBoundary
    ) {
      noteCompletedTextAsAssistantReplyIfNeeded(params.requestState);
    }
  }
  const finalAccumulatedContent = resolveAccumulatedFinalContentForCompletion(
    params.requestState,
  );
  const turnCompletedPlan = buildAgentStreamTerminalCompletionPlan({
    accumulatedContent: finalAccumulatedContent,
    fallbackContent: params.assistantFallbackContent,
    hasAssistantTextAfterLatestFinalAnswerRequiredProcessBoundary:
      params.requestState
        .hasAssistantTextAfterLatestFinalAnswerRequiredProcessBoundary,
    hasFinalAnswerRequiredProcessBoundary:
      params.requestState.hasFinalAnswerRequiredProcessBoundary,
    hasMeaningfulCompletionSignal:
      params.requestState.hasMeaningfulCompletionSignal,
    toolCallCount: params.toolCallCount,
    usage: params.event.usage,
  });
  if (turnCompletedPlan.type === "missing_final_reply_failure") {
    params.finalizeMissingFinalReplyFailure(turnCompletedPlan);
    return;
  }
  params.completeAssistantStreamMessageFromCompletionPlan({
    ...turnCompletedPlan,
    usage: params.event.usage,
  });
}

export function handleAgentStreamTurnCanceledEvent(params: {
  assistantMsgId: string;
  completeInterruptedTurn: (
    turn: Extract<AgentEvent, { type: "turn_canceled" }>["turn"],
  ) => void;
  event: Extract<AgentEvent, { type: "turn_canceled" }>;
  pendingTurnKey: string;
  setters: RuntimeHandlerStateSetters;
}): void {
  bindAssistantMessageToRuntimeTurn(
    params.setters.setMessages,
    params.assistantMsgId,
    params.event.turn.id,
  );
  params.setters.setThreadTurns((prev) =>
    upsertThreadTurnState(
      removeThreadTurnState(prev, params.pendingTurnKey),
      params.event.turn,
    ),
  );
  params.completeInterruptedTurn(params.event.turn);
}

export function handleAgentStreamTurnFailedEvent(params: {
  assistantMsgId: string;
  completeAssistantStreamMessageFromCompletionPlan: (
    plan: AgentStreamCompletionMessagePlan,
  ) => void;
  event: Extract<AgentEvent, { type: "turn_failed" }>;
  finalizeMissingFinalReplyFailure: (
    plan: AgentStreamMissingFinalReplyPlan,
  ) => void;
  pendingTurnKey: string;
  requestState: StreamRequestState;
  setters: RuntimeHandlerStateSetters;
  toolCallCount: number;
}): void {
  bindAssistantMessageToRuntimeTurn(
    params.setters.setMessages,
    params.assistantMsgId,
    params.event.turn.id,
  );
  params.setters.setThreadTurns((prev) =>
    upsertThreadTurnState(
      removeThreadTurnState(prev, params.pendingTurnKey),
      params.event.turn,
    ),
  );
  params.setters.setCurrentTurnId(params.event.turn.id);
  const softCompletionPlan = buildAgentStreamTerminalCompletionPlan({
    accumulatedContent: resolveAccumulatedFinalContentForCompletion(
      params.requestState,
    ),
    hasAssistantTextAfterLatestFinalAnswerRequiredProcessBoundary:
      params.requestState
        .hasAssistantTextAfterLatestFinalAnswerRequiredProcessBoundary,
    hasFinalAnswerRequiredProcessBoundary:
      params.requestState.hasFinalAnswerRequiredProcessBoundary,
    hasMeaningfulCompletionSignal:
      params.requestState.hasMeaningfulCompletionSignal,
    toolCallCount: params.toolCallCount,
  });
  if (softCompletionPlan.type === "complete") {
    params.completeAssistantStreamMessageFromCompletionPlan(softCompletionPlan);
    return;
  }

  const errorMessage = params.event.turn.error_message || "当前处理失败";
  params.finalizeMissingFinalReplyFailure(
    buildAgentStreamMissingFinalReplyFailurePlan({
      errorMessage,
      toastMessage:
        resolveAgentRuntimeErrorPresentation(errorMessage).toastMessage,
    }),
  );
}
