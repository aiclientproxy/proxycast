import type { Dispatch, MutableRefObject, SetStateAction } from "react";
import type { AgentThreadItem, AgentThreadTurn } from "@/lib/api/agentProtocol";
import { logAgentDebug } from "@/lib/agentDebug";
import type { Message } from "../types";
import type { ActiveStreamState } from "./agentStreamSubmissionLifecycle";
import type { AgentRuntimeAdapter } from "./agentRuntimeAdapter";
import type {
  InterruptedInputDraftSnapshot,
  InterruptedInputRestoreRequest,
} from "./agentStreamInputRestoreTypes";
import { updateMessageArtifactsStatus } from "../utils/messageArtifacts";
import {
  removeThreadItemState,
  removeThreadTurnState,
  upsertThreadItemState,
} from "./agentThreadState";
import { rememberLocallyInterruptedAgentStreamBinding } from "./agentStreamResumeBinding";
import {
  clearAgentStreamTextOverlay,
  getAgentStreamTextOverlay,
} from "./agentStreamTextOverlayStore";
import { buildAgentStreamProcessBoundaryTextCommitPatch } from "./agentStreamProcessBoundaryCommit";
import { resolveInterruptedInputRestorePlan } from "./agentStreamInputRestorePlan";
import {
  buildInterruptedMessageContentPatch,
  markInterruptedAgentMessageThreadItems,
  stripInterruptedPlaceholderText,
} from "./agentInterruptedMessageContent";

export { buildInterruptedMessageContentPatch } from "./agentInterruptedMessageContent";

interface AgentStreamFlowNotify {
  info: (message: string) => void;
  error: (message: string) => void;
}

interface StopAgentStreamOptions {
  activeStream: ActiveStreamState | null;
  sessionIdRef: MutableRefObject<string | null>;
  threadId?: string | null;
  currentTurnId?: string | null;
  runtime: AgentRuntimeAdapter;
  removeStreamListener: (eventName: string) => boolean;
  refreshSessionReadModel: (targetSessionId?: string) => Promise<boolean>;
  setThreadItems: Dispatch<SetStateAction<AgentThreadItem[]>>;
  setThreadTurns: Dispatch<SetStateAction<AgentThreadTurn[]>>;
  setCurrentTurnId: Dispatch<SetStateAction<string | null>>;
  setMessages: Dispatch<SetStateAction<Message[]>>;
  getMessages?: () => readonly Message[];
  getThreadItems?: () => readonly AgentThreadItem[];
  setActiveStream: (nextActive: ActiveStreamState | null) => void;
  submittedDraftFallback?: InterruptedInputDraftSnapshot | null;
  onRestoreInterruptedInput?: (request: InterruptedInputRestoreRequest) => void;
  notify: AgentStreamFlowNotify;
  onInterruptError?: (error: unknown) => void;
}

function resolveInterruptTurnId(activeStream: ActiveStreamState | null) {
  const explicitTurnId = activeStream?.turnId?.trim();
  if (explicitTurnId) {
    return explicitTurnId;
  }

  const pendingTurnKey = activeStream?.pendingTurnKey?.trim();
  if (pendingTurnKey && !pendingTurnKey.startsWith("pending-turn:")) {
    return pendingTurnKey;
  }

  return undefined;
}

function normalizeConcreteTurnId(value?: string | null): string | undefined {
  const turnId = value?.trim();
  if (!turnId || turnId.startsWith("pending-turn:")) {
    return undefined;
  }
  return turnId;
}

function isSameSessionItem(
  item: AgentThreadItem,
  sessionId?: string | null,
): boolean {
  const normalizedSessionId = sessionId?.trim();
  return !normalizedSessionId || item.thread_id === normalizedSessionId;
}

function resolveThreadItemInterruptedTurnId(options: {
  activeStream: ActiveStreamState | null;
  activeSessionId?: string | null;
  threadItems: readonly AgentThreadItem[];
}): string | undefined {
  const { activeStream, activeSessionId, threadItems } = options;
  const activeThreadId = activeStream?.sessionId || activeSessionId;
  const readTurnId = (value?: string | null): string | undefined => {
    const turnId = normalizeConcreteTurnId(value);
    return turnId;
  };

  const pendingItemKey = activeStream?.pendingItemKey?.trim();
  if (pendingItemKey) {
    const pendingItemTurnId = readTurnId(
      threadItems.find((item) => item.id === pendingItemKey)?.turn_id,
    );
    if (pendingItemTurnId) {
      return pendingItemTurnId;
    }
  }

  const sameSessionItems = threadItems.filter((item) =>
    isSameSessionItem(item, activeThreadId),
  );
  const findLatestTurnId = (predicate: (item: AgentThreadItem) => boolean) => {
    for (let index = sameSessionItems.length - 1; index >= 0; index -= 1) {
      const item = sameSessionItems[index];
      if (!item || !predicate(item)) {
        continue;
      }
      const turnId = readTurnId(item.turn_id);
      if (turnId) {
        return turnId;
      }
    }
    return undefined;
  };

  return (
    findLatestTurnId(
      (item) => item.status === "in_progress" && item.type === "agent_message",
    ) ?? findLatestTurnId((item) => item.status === "in_progress")
  );
}

function resolveInterruptedRuntimeTurnId(options: {
  activeStream: ActiveStreamState | null;
  activeSessionId?: string | null;
  assistantMessage?: Message | null;
  currentTurnId?: string | null;
  interruptTurnId?: string;
  threadItems: readonly AgentThreadItem[];
}): string | undefined {
  const {
    activeStream,
    activeSessionId,
    assistantMessage,
    currentTurnId,
    interruptTurnId,
    threadItems,
  } = options;
  const readTurnId = (value?: string | null): string | undefined => {
    const turnId = normalizeConcreteTurnId(value);
    return turnId;
  };

  return (
    readTurnId(interruptTurnId) ??
    readTurnId(assistantMessage?.runtimeTurnId) ??
    resolveThreadItemInterruptedTurnId({
      activeStream,
      activeSessionId,
      threadItems,
    }) ??
    readTurnId(currentTurnId)
  );
}

function messageVisibleTextLength(message?: Message | null): number {
  if (!message) {
    return 0;
  }

  const contentLength = message.content?.trim().length ?? 0;
  const contentPartsLength =
    message.contentParts?.reduce(
      (length, part) =>
        part.type === "text" ? length + part.text.trim().length : length,
      0,
    ) ?? 0;
  return Math.max(contentLength, contentPartsLength);
}

function resolveMessageVisibleText(message?: Message | null): string {
  if (!message) {
    return "";
  }

  const contentPartsText = (message.contentParts ?? [])
    .filter(
      (
        part,
      ): part is Extract<
        NonNullable<Message["contentParts"]>[number],
        { type: "text" }
      > => part.type === "text",
    )
    .map((part) => part.text)
    .join("");
  return [message.content ?? "", contentPartsText]
    .map(stripInterruptedPlaceholderText)
    .reduce(
      (longest, candidate) =>
        candidate.trim().length > longest.trim().length ? candidate : longest,
      "",
    );
}

function resolveInterruptedCanonicalText(options: {
  activeSessionId?: string | null;
  canonicalThreadId?: string | null;
  runtimeTurnId?: string;
  threadItems: readonly AgentThreadItem[];
}): string {
  const {
    activeSessionId,
    canonicalThreadId,
    runtimeTurnId,
    threadItems,
  } = options;
  if (!runtimeTurnId) {
    return "";
  }

  const acceptedThreadIds = new Set(
    [activeSessionId?.trim(), canonicalThreadId?.trim()].filter(
      (value): value is string => Boolean(value),
    ),
  );
  let visibleText = "";
  let visibleTextLength = 0;
  for (const item of threadItems) {
    const itemThreadId = item.thread_id?.trim();
    if (
      item.type !== "agent_message" ||
      item.turn_id !== runtimeTurnId ||
      (itemThreadId &&
        acceptedThreadIds.size > 0 &&
        !acceptedThreadIds.has(itemThreadId))
    ) {
      continue;
    }
    const candidateText = stripInterruptedPlaceholderText(item.text);
    if (candidateText.trim().length <= visibleTextLength) {
      continue;
    }
    visibleText = candidateText;
    visibleTextLength = candidateText.trim().length;
  }
  return visibleText;
}

function resolveInterruptedVisibleMessage(options: {
  activeMessage?: Message | null;
  messages: readonly Message[];
  runtimeTurnId?: string;
}): Message | null {
  const { activeMessage, messages, runtimeTurnId } = options;
  if (!runtimeTurnId) {
    return activeMessage ?? null;
  }

  let visibleMessage = activeMessage ?? null;
  for (const message of messages) {
    if (
      message.role !== "assistant" ||
      message.runtimeTurnId !== runtimeTurnId ||
      messageVisibleTextLength(message) <= messageVisibleTextLength(visibleMessage)
    ) {
      continue;
    }
    visibleMessage = message;
  }
  return visibleMessage;
}

function patchInterruptedAssistantMessage(options: {
  message: Message;
  runtimeTurnId?: string;
  visibleText: string;
}): Message {
  const { message, runtimeTurnId, visibleText } = options;
  const visiblePatch = visibleText.trim()
    ? buildAgentStreamProcessBoundaryTextCommitPatch({
        accumulatedContent: visibleText,
        parts: message.contentParts,
        renderedContent: visibleText,
        shouldRetainThinkingPart: () => true,
        surfaceThinkingDeltas: true,
      })
    : {};
  const merged = {
    ...settleInterruptedMessageProcess({
      ...message,
      ...(runtimeTurnId ? { runtimeTurnId } : {}),
    }),
    ...visiblePatch,
  };
  return {
    ...updateMessageArtifactsStatus(merged, "complete"),
    ...buildInterruptedMessageContentPatch(merged),
    isThinking: false,
    runtimeTurnId: runtimeTurnId ?? merged.runtimeTurnId,
    runtimeStatus: undefined,
  };
}

function upsertInterruptedAssistantMessage(options: {
  messages: Message[];
  messageId: string;
  runtimeTurnId?: string;
  visibleText: string;
}): Message[] {
  const { messages, messageId, runtimeTurnId, visibleText } = options;
  const exactIndex = messages.findIndex((message) => message.id === messageId);
  const sameTurnIndex =
    exactIndex >= 0 || !runtimeTurnId
      ? -1
      : messages.findIndex(
          (message) =>
            message.role === "assistant" &&
            message.runtimeTurnId === runtimeTurnId,
        );
  const targetIndex = exactIndex >= 0 ? exactIndex : sameTurnIndex;
  if (targetIndex >= 0) {
    return messages.map((message, index) =>
      index === targetIndex
        ? patchInterruptedAssistantMessage({
            message,
            runtimeTurnId,
            visibleText,
          })
        : message,
    );
  }

  const message: Message = {
    id: messageId,
    role: "assistant",
    content: "",
    contentParts: [],
    timestamp: new Date(),
    runtimeTurnId,
  };
  return [
    ...messages,
    patchInterruptedAssistantMessage({
      message,
      runtimeTurnId,
      visibleText,
    }),
  ];
}

function mergeInterruptedAgentMessageText(
  item: AgentThreadItem,
  runtimeTurnId: string | undefined,
  visibleText: string,
): AgentThreadItem {
  if (
    item.type !== "agent_message" ||
    !runtimeTurnId ||
    item.turn_id !== runtimeTurnId ||
    !visibleText.trim() ||
    item.text.trim().length >= visibleText.trim().length
  ) {
    return item;
  }

  return {
    ...item,
    text: visibleText,
  };
}

function ensureInterruptedAgentMessageItem(options: {
  items: AgentThreadItem[];
  activeMessageId: string;
  threadId?: string | null;
  turnId?: string;
  text: string;
}): AgentThreadItem[] {
  const { items, activeMessageId, threadId, turnId, text } = options;
  if (!turnId || !text.trim()) {
    return items;
  }
  const existing = items.some(
    (item) => item.type === "agent_message" && item.turn_id === turnId,
  );
  if (existing) {
    return items;
  }
  const now = new Date().toISOString();
  const sequence = items.reduce(
    (max, item) => Math.max(max, item.sequence),
    0,
  ) + 1;
  return upsertThreadItemState(items, {
    id: activeMessageId,
    thread_id: threadId?.trim() || "",
    turn_id: turnId,
    sequence,
    status: "in_progress",
    started_at: now,
    updated_at: now,
    type: "agent_message",
    text,
    phase: "final_answer",
  });
}

const INTERRUPTED_TOOL_RESULT_TEXT = "本轮已中止";

function settleInterruptedToolCall<
  T extends { status: string; result?: unknown; endTime?: Date },
>(toolCall: T): T {
  if (toolCall.status !== "running") {
    return toolCall;
  }

  return {
    ...toolCall,
    status: "failed",
    endTime: new Date(),
    result: {
      success: false,
      output: "",
      error: INTERRUPTED_TOOL_RESULT_TEXT,
    },
  };
}

export function settleInterruptedMessageProcess(message: Message): Message {
  const nextToolCalls = message.toolCalls?.map(settleInterruptedToolCall);
  const nextContentParts = message.contentParts?.map((part) => {
    if (part.type !== "tool_use") {
      return part;
    }

    return {
      ...part,
      toolCall: settleInterruptedToolCall(part.toolCall),
    };
  });

  return {
    ...message,
    toolCalls: nextToolCalls,
    contentParts: nextContentParts,
  };
}

export async function stopActiveAgentStream(options: StopAgentStreamOptions) {
  const {
    activeStream,
    sessionIdRef,
    threadId,
    currentTurnId,
    runtime,
    removeStreamListener,
    refreshSessionReadModel,
    setThreadItems,
    setThreadTurns,
    setCurrentTurnId,
    setMessages,
    getMessages,
    getThreadItems,
    setActiveStream,
    submittedDraftFallback,
    onRestoreInterruptedInput,
    notify,
    onInterruptError,
  } = options;

  if (activeStream) {
    removeStreamListener(activeStream.eventName);
  }
  rememberLocallyInterruptedAgentStreamBinding(activeStream);

  const activeSessionId = activeStream?.sessionId || sessionIdRef.current;
  const assistantMessage =
    activeStream?.assistantMsgId && getMessages
      ? (getMessages().find(
          (message) => message.id === activeStream.assistantMsgId,
        ) ?? null)
      : null;
  const currentThreadItems = getThreadItems?.() ?? [];
  const restorePlan = resolveInterruptedInputRestorePlan({
    submittedDraft: activeStream?.submittedDraft ?? submittedDraftFallback,
    assistantMessage,
  });
  const interruptTurnId = resolveInterruptTurnId(activeStream);
  const interruptedRuntimeTurnId = resolveInterruptedRuntimeTurnId({
    activeStream,
    activeSessionId,
    assistantMessage,
    currentTurnId,
    interruptTurnId,
    threadItems: currentThreadItems,
  });
  const interruptedVisibleMessage = resolveInterruptedVisibleMessage({
    activeMessage: assistantMessage,
    messages: getMessages?.() ?? [],
    runtimeTurnId: interruptedRuntimeTurnId,
  });
  const interruptedAssistantMessageId =
    interruptedVisibleMessage?.id ??
    activeStream?.assistantMsgId ??
    (interruptedRuntimeTurnId
      ? `interrupted-assistant:${interruptedRuntimeTurnId}`
      : null);
  logAgentDebug("AgentStream", "inputRestorePlan", {
    assistantMessageContentLength:
      assistantMessage?.content?.trim().length ?? 0,
    assistantMessagePartCount: assistantMessage?.contentParts?.length ?? 0,
    draftImageCount: restorePlan.draft?.images?.length ?? 0,
    draftPathReferenceCount: restorePlan.draft?.pathReferences?.length ?? 0,
    draftTextLength: restorePlan.draft?.text.trim().length ?? 0,
    eventName: activeStream?.eventName ?? null,
    hasActiveStream: Boolean(activeStream),
    hasActiveStreamDraft: Boolean(activeStream?.submittedDraft),
    hasSubmittedDraftFallback: Boolean(submittedDraftFallback),
    reason: restorePlan.reason,
    shouldRestoreComposer: restorePlan.shouldRestoreComposer,
  });
  const restoreInterruptedInput = () => {
    if (!restorePlan.shouldRestoreComposer || !restorePlan.draft) {
      return;
    }
    logAgentDebug("AgentStream", "inputRestoreDispatch", {
      draftImageCount: restorePlan.draft.images?.length ?? 0,
      draftPathReferenceCount: restorePlan.draft.pathReferences?.length ?? 0,
      draftTextLength: restorePlan.draft.text.trim().length,
      eventName: activeStream?.eventName ?? null,
      reason: restorePlan.reason,
    });
    onRestoreInterruptedInput?.({
      requestId: crypto.randomUUID(),
      reason: restorePlan.reason,
      draft: restorePlan.draft,
    });
  };
  const runInterruptAndRefresh = async () => {
    if (!activeSessionId) {
      return;
    }
    try {
      const turnIdForCancel =
        interruptTurnId ??
        interruptedRuntimeTurnId ??
        normalizeConcreteTurnId(currentTurnId);
      const canonicalThreadId = threadId?.trim();
      if (turnIdForCancel && canonicalThreadId) {
        await runtime.interruptTurn(
          canonicalThreadId,
          turnIdForCancel,
          activeStream?.eventName,
        );
      } else if (turnIdForCancel) {
        onInterruptError?.(
          new Error("缺少 canonical threadId，无法中止当前回合"),
        );
      }
    } catch (error) {
      onInterruptError?.(error);
    }
    try {
      await refreshSessionReadModel(activeSessionId);
    } catch (error) {
      onInterruptError?.(error);
    }
    let refreshedReadModelItems: AgentThreadItem[] = [];
    try {
      const refreshedReadModel = await runtime.getSessionReadModel(
        activeSessionId,
      );
      refreshedReadModelItems = Array.isArray(
        refreshedReadModel?.thread_items,
      )
        ? refreshedReadModel.thread_items
        : [];
    } catch (error) {
      onInterruptError?.(error);
    }
    let replayedThreadItems: AgentThreadItem[] = [];
    const canonicalThreadId = threadId?.trim();
    if (
      canonicalThreadId &&
      typeof runtime.resumeThread === "function"
    ) {
      try {
        await runtime.resumeThread(canonicalThreadId, (reducer) => {
          replayedThreadItems = [...reducer.getProjection().items];
        });
      } catch (error) {
        onInterruptError?.(error);
      }
    }
    const refreshedThreadItems = [
      ...(getThreadItems?.() ?? []),
      ...refreshedReadModelItems,
      ...replayedThreadItems,
    ];
    const refreshedCanonicalText = resolveInterruptedCanonicalText({
      activeSessionId,
      canonicalThreadId: threadId,
      runtimeTurnId: interruptedRuntimeTurnId,
      threadItems: refreshedThreadItems,
    });
    if (interruptedRuntimeTurnId && refreshedCanonicalText.trim()) {
      setThreadItems((prev) =>
        markInterruptedAgentMessageThreadItems(
          [...refreshedReadModelItems, ...replayedThreadItems].reduce(
            (items, item) => upsertThreadItemState(items, item),
            prev,
          ).map((item) =>
            mergeInterruptedAgentMessageText(
              item,
              interruptedRuntimeTurnId,
              refreshedCanonicalText,
            ),
          ),
          new Set([interruptedRuntimeTurnId]),
        ),
      );
    }
    if (interruptedAssistantMessageId && refreshedCanonicalText.trim()) {
      setMessages((prev) =>
        upsertInterruptedAssistantMessage({
          messages: prev,
          messageId: interruptedAssistantMessageId,
          runtimeTurnId: interruptedRuntimeTurnId,
          visibleText: refreshedCanonicalText,
        }),
      );
    }
  };

  if (interruptedAssistantMessageId) {
    const visibleTextOverlay = getAgentStreamTextOverlay(
      interruptedAssistantMessageId,
    );
    const canonicalVisibleText = resolveInterruptedCanonicalText({
      activeSessionId,
      canonicalThreadId: threadId,
      runtimeTurnId: interruptedRuntimeTurnId,
      threadItems: currentThreadItems,
    });
    const visibleMessage = resolveInterruptedVisibleMessage({
      activeMessage: interruptedVisibleMessage,
      messages: getMessages?.() ?? [],
      runtimeTurnId: interruptedRuntimeTurnId,
    });
    const visibleTextCandidate = [
      visibleTextOverlay?.content ?? "",
      canonicalVisibleText,
      resolveMessageVisibleText(visibleMessage),
    ].reduce((longest, candidate) =>
      candidate.trim().length > longest.trim().length ? candidate : longest,
    );
    clearAgentStreamTextOverlay(interruptedAssistantMessageId);
    if (activeStream?.pendingItemKey || interruptedRuntimeTurnId) {
      setThreadItems((prev) => {
        let nextItems = activeStream?.pendingItemKey
          ? removeThreadItemState(prev, activeStream.pendingItemKey)
          : prev;
        nextItems = ensureInterruptedAgentMessageItem({
          items: nextItems,
          activeMessageId: interruptedAssistantMessageId,
          threadId: threadId || activeSessionId,
          turnId: interruptedRuntimeTurnId,
          text: visibleTextCandidate,
        });
        return markInterruptedAgentMessageThreadItems(
          nextItems.map((item) =>
            mergeInterruptedAgentMessageText(
              item,
              interruptedRuntimeTurnId,
              visibleTextCandidate,
            ),
          ),
          new Set(interruptedRuntimeTurnId ? [interruptedRuntimeTurnId] : []),
        );
      });
    }
    if (activeStream?.pendingTurnKey) {
      setThreadTurns((prev) =>
        removeThreadTurnState(prev, activeStream.pendingTurnKey!),
      );
      setCurrentTurnId((prev) =>
        prev === activeStream.pendingTurnKey ? null : prev,
      );
    }
    setMessages((prev) =>
      upsertInterruptedAssistantMessage({
        messages: prev,
        messageId: interruptedAssistantMessageId,
        runtimeTurnId: interruptedRuntimeTurnId,
        visibleText: visibleTextCandidate,
      }),
    );
    clearAgentStreamTextOverlay(interruptedAssistantMessageId);
  }

  setActiveStream(null);
  restoreInterruptedInput();
  notify.info("已停止生成");
  void runInterruptAndRefresh();
}
