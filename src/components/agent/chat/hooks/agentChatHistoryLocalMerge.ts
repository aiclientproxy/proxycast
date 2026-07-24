import type {
  Message,
  MessageImageWorkbenchPreview,
  MessageTaskPreview,
} from "../types";
import { projectConversationMessagesByRuntimeTurn } from "../utils/conversationTimelineOrdering";
import { mergeArtifacts } from "../utils/messageArtifacts";
import {
  extractThinkingContentFromParts,
  mergeByKey,
} from "./agentChatHistoryPrimitives";
import {
  hasRenderableAssistantTextContent,
  mergeHydratedContentParts,
  mergeHydratedToolStateContentParts,
  mergeImageWorkbenchPreview,
  mergeTaskPreview,
  mergeToolCallStates,
  settleRunningProcessPartsOnRemoteFailure,
  settleRunningToolCallOnRemoteFailure,
} from "./agentChatHistoryProcess";
import type { HistoryToolCall } from "./agentChatHistoryTypes";
import {
  buildInterruptedMessageContentPatch,
  contentHasInterruptedPlaceholder,
  contentPartsHaveInterruptedPlaceholder,
  messageHasInterruptedPlaceholder,
} from "./agentInterruptedMessageContent";
import { hasMessageImages } from "./agentChatHistorySignatures";
import { hasMeaningfulAssistantVisibleText } from "../utils/assistantVisibleText";
import {
  findMatchingLocalAssistantMessageIndex,
  findMatchingLocalUserMessageIndex,
  findNextLocalAssistantAfterUser,
  findNextLocalInterruptedAssistantIndex,
  findNextLocalProcessAssistantIndex,
  hasRecoverableLocalUserBeforeAssistant,
  insertRecoverableLocalUsersForMatchedAssistantTurns,
  isLocalAssistantInMatchedUserTurn,
} from "./agentChatHistoryLocalMergeMatching";
import {
  hasRetainableLocalAssistantProcessState,
  mergeAssistantVisibleOutput,
  shouldMergeLocalAssistantProcessState,
  shouldPreserveLocalAssistantVisibleOutput,
} from "./agentChatHistoryLocalMergeState";
import {
  collectRetainedLocalCanonicalHistory,
  collectRetainedLocalTail,
} from "./agentChatHistoryLocalMergeTail";

export const mergeHydratedMessagesWithLocalState = (
  localMessages: Message[],
  hydratedMessages: Message[],
  options: {
    preferHydratedAssistantOutput?: boolean;
  } = {},
): Message[] => {
  if (hydratedMessages.length === 0) {
    return localMessages;
  }

  const localUserMessages = localMessages.filter(
    (message) => message.role === "user",
  );
  const localAssistantMessages = localMessages.filter(
    (message) => message.role === "assistant",
  );
  const localUserMessageIndexById = new Map<string, number>();
  const localAssistantMessageIndexById = new Map<string, number>();
  const localMessageIndexById = new Map<string, number>();
  const localImagePreviewByTaskId = new Map<
    string,
    MessageImageWorkbenchPreview
  >();
  const localImageAssistantMessageByTaskId = new Map<string, Message>();
  const localTaskPreviewByTaskId = new Map<string, MessageTaskPreview>();
  const hydratedMessageIds = new Set<string>();
  const hydratedRuntimeTurnIds = new Set<string>();

  localUserMessages.forEach((message, index) => {
    if (message.id) {
      localUserMessageIndexById.set(message.id, index);
    }
  });
  localAssistantMessages.forEach((message, index) => {
    if (message.id) {
      localAssistantMessageIndexById.set(message.id, index);
    }
  });
  localMessages.forEach((message, index) => {
    if (message.id) {
      localMessageIndexById.set(message.id, index);
    }
  });
  hydratedMessages.forEach((message) => {
    if (message.id) {
      hydratedMessageIds.add(message.id);
    }
    const runtimeTurnId = message.runtimeTurnId?.trim();
    if (runtimeTurnId && !runtimeTurnId.startsWith("pending-turn:")) {
      hydratedRuntimeTurnIds.add(runtimeTurnId);
    }
  });

  localMessages.forEach((message) => {
    if (message.role !== "assistant") {
      return;
    }

    const imageTaskId = message.imageWorkbenchPreview?.taskId;
    if (imageTaskId) {
      localImagePreviewByTaskId.set(
        imageTaskId,
        message.imageWorkbenchPreview as MessageImageWorkbenchPreview,
      );
      localImageAssistantMessageByTaskId.set(imageTaskId, message);
    }

    const taskPreviewId = message.taskPreview?.taskId;
    if (taskPreviewId) {
      localTaskPreviewByTaskId.set(
        taskPreviewId,
        message.taskPreview as MessageTaskPreview,
      );
    }
  });

  if (
    localUserMessages.length === 0 &&
    localAssistantMessages.length === 0 &&
    localImagePreviewByTaskId.size === 0 &&
    localTaskPreviewByTaskId.size === 0
  ) {
    return hydratedMessages;
  }

  let localUserCursor = 0;
  let localAssistantCursor = 0;
  let lastMatchedLocalUserMessageIndex: number | null = null;
  let hasHydratedUserInCurrentTurn = false;
  const matchedLocalMessageIds = new Set<string>();

  const mergedMessages = hydratedMessages.map((message) => {
    if (message.role === "assistant") {
      const matchedAssistantIndexById = message.id
        ? (localAssistantMessageIndexById.get(message.id) ?? -1)
        : -1;
      const matchedAssistantIndex =
        matchedAssistantIndexById >= 0
          ? matchedAssistantIndexById
          : findMatchingLocalAssistantMessageIndex(
              localAssistantMessages,
              message,
              localAssistantCursor,
            );
      const sequentialAssistantIndex =
        matchedAssistantIndex >= 0
          ? -1
          : findNextLocalAssistantAfterUser(
              localMessages,
              localAssistantMessageIndexById,
              matchedLocalMessageIds,
              lastMatchedLocalUserMessageIndex,
              message,
            );
      const shouldSearchProcessAssistantByRuntimeTurn = Boolean(
        message.runtimeTurnId?.trim(),
      );
      const processAssistantIndex =
        matchedAssistantIndex >= 0 ||
        sequentialAssistantIndex >= 0 ||
        (hasRetainableLocalAssistantProcessState(message) &&
          !shouldSearchProcessAssistantByRuntimeTurn)
          ? -1
          : findNextLocalProcessAssistantIndex(
              localAssistantMessages,
              localAssistantCursor,
              matchedLocalMessageIds,
              message,
            );
      const interruptedAssistantIndex =
        matchedAssistantIndex >= 0 ||
        sequentialAssistantIndex >= 0 ||
        processAssistantIndex >= 0
          ? -1
          : findNextLocalInterruptedAssistantIndex(
              localAssistantMessages,
              localAssistantCursor,
              matchedLocalMessageIds,
              message,
            );
      const resolvedMatchedAssistantIndex =
        matchedAssistantIndex >= 0
          ? matchedAssistantIndex
          : sequentialAssistantIndex >= 0
            ? sequentialAssistantIndex
            : processAssistantIndex >= 0
              ? processAssistantIndex
              : interruptedAssistantIndex;
      const didMatchLocalProcessAssistantOnly =
        processAssistantIndex >= 0 &&
        matchedAssistantIndex < 0 &&
        sequentialAssistantIndex < 0;
      const taskMatchedAssistantMessage =
        resolvedMatchedAssistantIndex < 0 &&
        message.imageWorkbenchPreview?.taskId
          ? localImageAssistantMessageByTaskId.get(
              message.imageWorkbenchPreview.taskId,
            )
          : undefined;
      const localAssistantMessage =
        resolvedMatchedAssistantIndex >= 0
          ? localAssistantMessages[resolvedMatchedAssistantIndex]
          : taskMatchedAssistantMessage;
      if (resolvedMatchedAssistantIndex >= 0) {
        localAssistantCursor = Math.max(
          localAssistantCursor,
          resolvedMatchedAssistantIndex + 1,
        );
        if (localAssistantMessage?.id) {
          matchedLocalMessageIds.add(localAssistantMessage.id);
        }
      } else if (taskMatchedAssistantMessage?.id) {
        matchedLocalMessageIds.add(taskMatchedAssistantMessage.id);
      }

      const localImagePreview =
        (message.imageWorkbenchPreview?.taskId
          ? localImagePreviewByTaskId.get(message.imageWorkbenchPreview.taskId)
          : undefined) ?? localAssistantMessage?.imageWorkbenchPreview;
      const localTaskPreview =
        (message.taskPreview?.taskId
          ? localTaskPreviewByTaskId.get(message.taskPreview.taskId)
          : undefined) ?? localAssistantMessage?.taskPreview;

      if (!localImagePreview && !localTaskPreview && !localAssistantMessage) {
        return message;
      }

      const hasMatchedUserTurn = isLocalAssistantInMatchedUserTurn({
        localAssistantMessage,
        localMessageIndexById,
        localMessages,
        lastMatchedLocalUserMessageIndex,
      });
      const hasRecoverableLocalUserTurn =
        !hasHydratedUserInCurrentTurn &&
        hasRecoverableLocalUserBeforeAssistant({
          localAssistantMessage,
          localMessageIndexById,
          localMessages,
          matchedLocalMessageIds,
        });
      const hasExactAssistantIdentity = matchedAssistantIndexById >= 0;
      const hasMatchingRuntimeTurn = Boolean(
        localAssistantMessage?.runtimeTurnId?.trim() &&
        message.runtimeTurnId?.trim() &&
        localAssistantMessage.runtimeTurnId === message.runtimeTurnId,
      );
      const shouldPreserveLocalRuntimeSnapshot =
        (hasMatchedUserTurn || hasRecoverableLocalUserTurn) &&
        hasRetainableLocalAssistantProcessState(localAssistantMessage) &&
        !hasRetainableLocalAssistantProcessState(message);
      const shouldRestoreHistoricalProcessState =
        (hasMatchedUserTurn ||
          hasRecoverableLocalUserTurn ||
          didMatchLocalProcessAssistantOnly) &&
        hasRetainableLocalAssistantProcessState(localAssistantMessage) &&
        !hasRetainableLocalAssistantProcessState(message);
      const hasMatchedAssistantBySignature =
        matchedAssistantIndex >= 0 &&
        matchedAssistantIndexById < 0 &&
        localUserMessages.length === 0;
      const shouldMergeMatchingProcessState =
        (hasExactAssistantIdentity ||
          hasMatchedAssistantBySignature ||
          hasMatchedUserTurn ||
          hasRecoverableLocalUserTurn ||
          didMatchLocalProcessAssistantOnly ||
          hasMatchingRuntimeTurn) &&
        shouldMergeLocalAssistantProcessState(localAssistantMessage, message);
      const shouldRetainLocalProcessState =
        !hasRenderableAssistantTextContent(message) ||
        shouldPreserveLocalRuntimeSnapshot ||
        shouldRestoreHistoricalProcessState ||
        shouldMergeMatchingProcessState;
      const shouldRetainLocalThinkingState =
        shouldRetainLocalProcessState || Boolean(message.thinkingContent);
      const contentParts = shouldRetainLocalProcessState
        ? mergeHydratedContentParts(
            localAssistantMessage?.contentParts,
            message.contentParts,
          )
        : message.contentParts;
      const toolCalls = (() => {
        const localToolCalls = localAssistantMessage?.toolCalls || [];
        const remoteToolCalls = message.toolCalls || [];
        if (localToolCalls.length === 0) {
          return remoteToolCalls.length > 0 ? remoteToolCalls : undefined;
        }
        if (remoteToolCalls.length === 0) {
          return localToolCalls;
        }

        const mergedToolCalls: HistoryToolCall[] = [...localToolCalls];
        const toolCallIndexById = new Map<string, number>();
        mergedToolCalls.forEach((toolCall, index) => {
          toolCallIndexById.set(toolCall.id, index);
        });

        for (const toolCall of remoteToolCalls) {
          const existingIndex = toolCallIndexById.get(toolCall.id);
          if (existingIndex === undefined) {
            toolCallIndexById.set(toolCall.id, mergedToolCalls.length);
            mergedToolCalls.push(toolCall);
            continue;
          }

          const existingToolCall = mergedToolCalls[existingIndex];
          if (existingToolCall) {
            mergedToolCalls[existingIndex] = mergeToolCallStates(
              existingToolCall,
              toolCall,
            );
          }
        }

        return mergedToolCalls;
      })();
      const actionRequests = mergeByKey(
        localAssistantMessage?.actionRequests,
        message.actionRequests,
        (request) => request.requestId,
      );
      const contextTrace = mergeByKey(
        localAssistantMessage?.contextTrace,
        message.contextTrace,
        (step) => `${step.stage}::${step.detail}`,
      );
      const artifacts = (() => {
        const localArtifacts = localAssistantMessage?.artifacts || [];
        const remoteArtifacts = message.artifacts || [];
        if (localArtifacts.length === 0) {
          return remoteArtifacts.length > 0 ? remoteArtifacts : undefined;
        }
        if (remoteArtifacts.length === 0) {
          return localArtifacts;
        }
        const merged = mergeArtifacts([...localArtifacts, ...remoteArtifacts]);
        return merged.length > 0 ? merged : undefined;
      })();
      const thinkingContent =
        message.thinkingContent ??
        (shouldRetainLocalThinkingState
          ? localAssistantMessage?.thinkingContent
          : undefined) ??
        extractThinkingContentFromParts(contentParts);
      const remoteIsFailedRuntimeStatus =
        message.runtimeStatus?.phase === "failed";
      const shouldPreserveLocalVisibleOutputOnRemoteFailure =
        remoteIsFailedRuntimeStatus &&
        hasMeaningfulAssistantVisibleText(localAssistantMessage);
      const localHasInterruptedStopMarker = messageHasInterruptedPlaceholder(
        localAssistantMessage,
      );
      const shouldPreserveLocalVisibleOutput =
        !options.preferHydratedAssistantOutput &&
        (remoteIsFailedRuntimeStatus
          ? shouldPreserveLocalVisibleOutputOnRemoteFailure
          : localAssistantMessage?.isThinking === true ||
            shouldPreserveLocalRuntimeSnapshot ||
            shouldPreserveLocalAssistantVisibleOutput(
              localAssistantMessage,
              message,
            ));
      const resolvedContentParts = shouldPreserveLocalVisibleOutput
        ? (mergeHydratedToolStateContentParts(
            localAssistantMessage?.contentParts,
            contentParts,
          ) ??
          localAssistantMessage?.contentParts ??
          contentParts)
        : contentParts;
      const resolvedThinkingContent = shouldPreserveLocalVisibleOutput
        ? (localAssistantMessage?.thinkingContent ??
          extractThinkingContentFromParts(resolvedContentParts))
        : thinkingContent;
      const resolvedFailedContentParts = remoteIsFailedRuntimeStatus
        ? settleRunningProcessPartsOnRemoteFailure(
            resolvedContentParts,
            message,
          )
        : resolvedContentParts;
      const resolvedFailedToolCalls =
        remoteIsFailedRuntimeStatus && toolCalls
          ? toolCalls.map((toolCall) =>
              settleRunningToolCallOnRemoteFailure(toolCall, message),
            )
          : toolCalls;
      const resolvedContent = shouldPreserveLocalVisibleOutput
        ? mergeAssistantVisibleOutput({
            localContent: localAssistantMessage?.content || "",
            remoteContent: message.content,
          })
        : message.content;
      const shouldRetainInterruptedStopMarker =
        localHasInterruptedStopMarker &&
        (!contentHasInterruptedPlaceholder(resolvedContent) ||
          !contentPartsHaveInterruptedPlaceholder(resolvedFailedContentParts));
      const interruptedContentPatch = shouldRetainInterruptedStopMarker
        ? buildInterruptedMessageContentPatch({
            ...message,
            content: resolvedContent,
            contentParts: resolvedFailedContentParts,
          })
        : null;

      return {
        ...message,
        id: localAssistantMessage?.id ?? message.id,
        content: interruptedContentPatch?.content ?? resolvedContent,
        usage: message.usage ?? localAssistantMessage?.usage,
        contentParts:
          interruptedContentPatch?.contentParts ?? resolvedFailedContentParts,
        toolCalls: resolvedFailedToolCalls,
        actionRequests,
        contextTrace,
        artifacts,
        thinkingContent: remoteIsFailedRuntimeStatus
          ? extractThinkingContentFromParts(resolvedFailedContentParts)
          : resolvedThinkingContent,
        runtimeTurnId:
          message.runtimeTurnId ?? localAssistantMessage?.runtimeTurnId,
        inlineProcessRetention:
          message.inlineProcessRetention ??
          localAssistantMessage?.inlineProcessRetention,
        imageWorkbenchPreview: mergeImageWorkbenchPreview(
          localImagePreview,
          message.imageWorkbenchPreview,
        ),
        taskPreview: mergeTaskPreview(localTaskPreview, message.taskPreview),
      };
    }

    if (message.role !== "user") {
      return message;
    }

    hasHydratedUserInCurrentTurn = true;
    const matchedIndexById = message.id
      ? (localUserMessageIndexById.get(message.id) ?? -1)
      : -1;
    const matchedIndex =
      matchedIndexById >= 0
        ? matchedIndexById
        : findMatchingLocalUserMessageIndex(
            localUserMessages,
            message,
            localUserCursor,
          );
    if (matchedIndex < 0) {
      lastMatchedLocalUserMessageIndex = null;
      return message;
    }

    localUserCursor = Math.max(localUserCursor, matchedIndex + 1);
    const localMessage = localUserMessages[matchedIndex];
    if (localMessage?.id) {
      matchedLocalMessageIds.add(localMessage.id);
      lastMatchedLocalUserMessageIndex =
        localMessageIndexById.get(localMessage.id) ?? null;
    } else {
      lastMatchedLocalUserMessageIndex = null;
    }
    if (!localMessage) {
      return message;
    }

    if (hasMessageImages(message) || !hasMessageImages(localMessage)) {
      return {
        ...message,
        id: localMessage.id || message.id,
        inputCapabilityRoute:
          message.inputCapabilityRoute ?? localMessage.inputCapabilityRoute,
      };
    }

    return {
      ...message,
      id: localMessage.id || message.id,
      images: localMessage.images,
      inputCapabilityRoute:
        message.inputCapabilityRoute ?? localMessage.inputCapabilityRoute,
    };
  });

  const lastHydratedMessage =
    hydratedMessages.length > 0
      ? (hydratedMessages[hydratedMessages.length - 1] as Message)
      : null;
  const lastMatchedLocalIndex = localMessages.reduce(
    (latest, message, index) => {
      if (!matchedLocalMessageIds.has(message.id)) {
        return latest;
      }
      return index;
    },
    -1,
  );
  const lastMatchedLocalMessage =
    lastMatchedLocalIndex >= 0 ? localMessages[lastMatchedLocalIndex] : null;
  const mergedMessagesWithRecoveredLocalUsers =
    insertRecoverableLocalUsersForMatchedAssistantTurns({
      mergedMessages,
      localMessageIndexById,
      localMessages,
      matchedLocalMessageIds,
    });

  const retainedLocalCanonicalHistory = collectRetainedLocalCanonicalHistory({
    hydratedMessageIds,
    hydratedRuntimeTurnIds,
    lastMatchedLocalIndex,
    localMessages,
    matchedLocalMessageIds,
  });

  const retainedLocalTail = collectRetainedLocalTail({
    hydratedMessageIds,
    lastHydratedMessage,
    lastMatchedLocalIndex,
    lastMatchedLocalMessage,
    localMessages,
    matchedLocalMessageIds,
  });

  return projectConversationMessagesByRuntimeTurn([
    ...retainedLocalCanonicalHistory,
    ...mergedMessagesWithRecoveredLocalUsers,
    ...retainedLocalTail,
  ]);
};
