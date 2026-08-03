import { isHiddenConversationArtifactPath } from "../utils/internalArtifactVisibility";
import { isPureRuntimePeerMessageText } from "../utils/runtimePeerMessageDisplay";
import {
  sanitizeContentPartsForDisplay,
  sanitizeMessageTextForDisplay,
} from "../utils/messageDisplaySanitizer";
import { hasStructuredHistoricalContentHint } from "../projection/historicalMessageHydrationProjection";
import { shouldUseAgentMessageAsFinalText } from "../utils/agentMessagePhase";
import {
  appendInterruptedPlaceholderText,
  contentHasInterruptedPlaceholder,
  contentPartsHaveInterruptedPlaceholder,
} from "../hooks/agentInterruptedMessageContent";
import { buildHistoricalMessagePreview } from "./messageListHistoricalPreviewText";
import {
  parseLeadingUserCommandTag,
  resolveInstalledSkillMessageLabel,
} from "./messageListUserContentState";
import {
  buildTimelineFileChangesContentPart,
  buildTimelineInlineContentParts,
  buildTimelineVisibleTextContentParts,
} from "./messageListTimelineContentParts";
import {
  resolveImageWorkbenchMessageDisplayState,
  resolveImageWorkbenchProcessDisplayState,
  resolveImageWorkbenchRendererProcessState,
} from "./imageWorkbenchMessageDisplay";
import {
  createInlineCoverageMatcher,
  dedupeDeferredTimelineItems,
  filterConversationDisplayContentParts,
  hasTimelineProcessItems,
  isDeferredTimelineItem,
  mergeStreamingOverlayContentParts,
  resolveInlineProcessCoverage,
  hasPersistedReasoningTimelineItem,
  shouldKeepInlineProcessForActiveAssistant,
  shouldRenderConversationTimelineItem,
  shouldSuppressPreAnswerThinkingTimeline,
} from "./messageListInlineProcess";
import {
  MESSAGE_LIST_COMPACT_HISTORICAL_ASSISTANT_PREVIEW_CHARS,
  MESSAGE_LIST_COMPACT_HISTORICAL_ASSISTANT_THRESHOLD,
  MESSAGE_LIST_LONG_HISTORICAL_MESSAGE_PREVIEW_CHARS,
  MESSAGE_LIST_LONG_HISTORICAL_MESSAGE_THRESHOLD,
} from "./messageListConstants";
import {
  collectFileChangeBatchPaths,
  ensureInlineThinkingContentPart,
  hasFinalTextAfterProcessBoundary,
  hasInlineProcessContentParts,
  hasInlineToolUseContentPart,
  normalizeInlineThinkingContentParts,
  resolveAssistantActionContent,
  resolveDeferredTextContentParts,
  resolveProcessSeparatedContentParts,
} from "./messageListProjectionContentParts";
import {
  hasCompletedOrRunningWebRetrievalTimelineItem,
  hasFinalAnswerTextAfterRunningWebRetrieval,
  hasFinalAnswerTextTimelineItem,
  hasRunningWebRetrievalContentPart,
  hasRunningWebRetrievalTimelineItem,
  hideFinalAnswerContentPartsWhileRunning,
  isActiveThreadTurnStatus,
  normalizeInactiveRunningWebRetrievalContentParts,
  normalizeInactiveRunningWebRetrievalTimelineItems,
} from "./messageListProjectionWebRetrieval";
import {
  canMergeTimelineAsSparseProcessPatch,
  canTimelineOwnInlineProcessFlow,
  isTerminalThreadTurnStatus,
  isRuntimeFailureDiagnosticAliasText,
  isRuntimeFailureOnlyAssistantText,
  isTrivialAssistantFinalText,
  resolveRuntimeFailureFallbackAssistantText,
  resolveMessageInteractiveProjectionState,
  resolveTimelineOwnedVisibleText,
  sanitizeRuntimeFailureAssistantText,
  sanitizeProjectedMessageText,
  shouldUseFirstTokenRuntimeStatus,
} from "./messageListItemProjectionHelpers";
import { resolveMessageListItemArtifactProjection } from "./messageListItemProjectionArtifacts";
import { isImportedHistoryMetadata } from "../utils/importedUserMessageDedupe";
import type { ResolveMessageListItemProjectionOptions } from "./messageListItemProjectionTypes";
import type { Message } from "../types";

export function resolveMessageListItemProjection({
  activeCurrentTurnId,
  activePendingA2UISource,
  canOpenSavedSiteContent,
  expandedHistoricalAssistantMessageIds,
  expandedLongHistoricalMessageIds,
  focusedTimelineItemId,
  group,
  hasActiveInteractiveRuntime,
  isRestoredHistoryWindow,
  isSending,
  lastAssistantMessageId,
  message,
  shouldDeferHistoricalAssistantMessageDetails,
  shouldDeferThreadItemsScan,
  streamingTextOverlay = null,
}: ResolveMessageListItemProjectionOptions) {
  const rawStreamingFinalTextOverlayContent =
    message.role === "assistant" &&
    streamingTextOverlay?.content &&
    shouldUseAgentMessageAsFinalText(streamingTextOverlay.phase)
      ? streamingTextOverlay.content
      : null;
  const rawDisplayContent = message.content || "";
  const hasImages = Array.isArray(message.images) && message.images.length > 0;
  const imageWorkbenchDisplayState = resolveImageWorkbenchMessageDisplayState({
    message,
    rawDisplayContent,
  });
  const shouldSuppressStandaloneImageWorkbenchProcess =
    imageWorkbenchDisplayState.shouldSuppressStandaloneProcess;
  const shouldSuppressImageProcessFlow =
    imageWorkbenchDisplayState.shouldSuppressProcessFlow;
  const rawVisibleDisplayContent =
    imageWorkbenchDisplayState.visibleRawDisplayContent;
  const visibleRawDisplayContent =
    message.role === "assistant" &&
    contentPartsHaveInterruptedPlaceholder(message.contentParts) &&
    !contentHasInterruptedPlaceholder(rawVisibleDisplayContent)
      ? appendInterruptedPlaceholderText(rawVisibleDisplayContent)
      : rawVisibleDisplayContent;
  const displayContent = sanitizeMessageTextForDisplay(
    visibleRawDisplayContent,
    {
      role: message.role,
      hasImages,
    },
  );
  const shouldDeferMessageDetails =
    shouldDeferHistoricalAssistantMessageDetails(message);
  const rawRuntimePeerContent = visibleRawDisplayContent.trim();
  const shouldRenderRuntimePeerCards =
    rawRuntimePeerContent.length > 0 &&
    isPureRuntimePeerMessageText(rawRuntimePeerContent);
  const sanitizedDisplayContentParts = shouldDeferMessageDetails
    ? resolveDeferredTextContentParts(message.contentParts, {
        role: message.role,
        hasImages,
      })
    : sanitizeContentPartsForDisplay(message.contentParts, {
        role: message.role,
        hasImages,
      });
  const imageWorkbenchProcessDisplayState =
    resolveImageWorkbenchProcessDisplayState({
      message,
      sanitizedContentParts: sanitizedDisplayContentParts,
      shouldDeferMessageDetails,
      shouldFoldSuppressedProcessFlow:
        imageWorkbenchDisplayState.shouldFoldSuppressedProcessFlow,
      shouldSuppressImageProcessFlow,
    });
  const {
    displayContentParts: rawDisplayContentParts,
    shouldFoldSuppressedProcessFlow,
    shouldSuppressRendererProcessFlow,
  } = imageWorkbenchProcessDisplayState;
  const isConversationTailAssistant =
    message.role === "assistant" && message.id === group.lastAssistantId;
  const isTimelineOwnerAssistant =
    message.role === "assistant" &&
    group.timeline !== null &&
    (group.timelineMessageId === message.id ||
      (!group.timelineMessageId && isConversationTailAssistant));
  const timeline =
    message.role !== "assistant"
      ? null
      : isTimelineOwnerAssistant
        ? group.timeline
        : null;
  const isImportedHistoryTurn = Boolean(
    group.isImportedHistory ||
      [...(group.timeline?.items ?? []), ...(timeline?.items ?? [])].some(
        (item) => isImportedHistoryMetadata(item.metadata),
      ) ||
      message.artifacts?.some((artifact) =>
        isImportedHistoryMetadata(artifact.meta),
      ) ||
      message.contentParts?.some((part) =>
        isImportedHistoryMetadata(part.metadata),
      ),
  );
  const groupTimelineTurnId = group.timeline?.turn.id?.trim() || null;
  const messageTurnId = message.runtimeTurnId?.trim() || groupTimelineTurnId;
  // 一个 canonical turn 可能在 hydration 后对应多个 assistant message。
  // 非 timeline owner 没有自己的 timeline，但仍必须继承所在 group 的终态，
  // 否则历史消息会把完整工具 contentParts 当作当前运行态重新渲染。
  const messageTurnStatus =
    messageTurnId && groupTimelineTurnId === messageTurnId
      ? group.timeline?.turn.status
      : timeline?.turn.status;
  const hasKnownActiveTurn =
    typeof activeCurrentTurnId === "string" &&
    activeCurrentTurnId.trim().length > 0;
  const shouldHideHistoricalRuntimeDetails = Boolean(
    messageTurnId &&
    ((hasKnownActiveTurn && messageTurnId !== activeCurrentTurnId) ||
      (!isSending && isTerminalThreadTurnStatus(messageTurnStatus))),
  );
  const rawTimelineItems = timeline?.items;
  const hasProcessTimelineItems = hasTimelineProcessItems(rawTimelineItems);
  const hasPersistedReasoningTimeline =
    hasPersistedReasoningTimelineItem(rawTimelineItems);
  const hasRunningTimelineProcess =
    timeline !== null && hasRunningWebRetrievalTimelineItem(rawTimelineItems);
  const hasCompletedOrRunningWebRetrievalTimelineProcess =
    timeline !== null &&
    hasCompletedOrRunningWebRetrievalTimelineItem(rawTimelineItems);
  const hasRunningWebRetrievalPart = hasRunningWebRetrievalContentPart(
    rawDisplayContentParts,
  );
  const hasInlineNonThinkingProcessPart = Boolean(
    rawDisplayContentParts?.some(
      (part) =>
        part.type === "tool_use" ||
        part.type === "action_required" ||
        part.type === "file_changes_batch",
    ),
  );
  const hasInlineToolUseProcessPart = hasInlineToolUseContentPart(
    rawDisplayContentParts,
  );
  const hasActiveTimelineTurn =
    timeline !== null && isActiveThreadTurnStatus(timeline.turn.status);
  const isLegacyUnphasedStreamingOverlay =
    message.role === "assistant" &&
    Boolean(rawStreamingFinalTextOverlayContent?.trim()) &&
    !streamingTextOverlay?.phase?.trim();
  const hasStructuredProcessBoundaryForActiveTurn =
    hasProcessTimelineItems ||
    hasCompletedOrRunningWebRetrievalTimelineProcess ||
    hasRunningWebRetrievalPart ||
    hasInlineToolUseProcessPart;
  const shouldHideLegacyUnphasedOverlayDuringProcess =
    isLegacyUnphasedStreamingOverlay &&
    hasStructuredProcessBoundaryForActiveTurn &&
    (message.isThinking || isSending || hasActiveTimelineTurn);
  const streamingFinalTextOverlay = shouldHideLegacyUnphasedOverlayDuringProcess
    ? null
    : rawStreamingFinalTextOverlayContent
      ? streamingTextOverlay
      : null;
  const streamingFinalTextOverlayContent =
    streamingFinalTextOverlay?.content ?? null;
  const hasFinalAnswerAfterRunningWebRetrieval =
    hasFinalAnswerTextAfterRunningWebRetrieval(rawTimelineItems);
  const hasFinalAnswerTimelineItem =
    hasFinalAnswerTextTimelineItem(rawTimelineItems);
  const hasRuntimeFailureTimelineErrorItem = Boolean(
    rawTimelineItems?.some((item) => item.type === "error"),
  );
  const isActiveAssistantOutput =
    message.isThinking ||
    isSending ||
    Boolean(streamingFinalTextOverlayContent?.trim()) ||
    hasActiveTimelineTurn;
  const shouldHoldRunningWebRetrieval =
    (hasRunningTimelineProcess || hasRunningWebRetrievalPart) &&
    isActiveAssistantOutput;
  const shouldNormalizeInactiveRunningWebRetrieval =
    message.role === "assistant" && !shouldHoldRunningWebRetrieval;
  const displayContentParts = normalizeInactiveRunningWebRetrievalContentParts(
    rawDisplayContentParts,
    shouldNormalizeInactiveRunningWebRetrieval,
  );
  const timelineItemsForDisplay =
    normalizeInactiveRunningWebRetrievalTimelineItems(
      rawTimelineItems,
      shouldNormalizeInactiveRunningWebRetrieval,
    );
  const terminalApprovalRequestIds = new Set(
    (timelineItemsForDisplay || []).flatMap((item) =>
      item.type === "approval_request" &&
      (item.status === "completed" || item.status === "failed")
        ? [item.request_id]
        : [],
    ),
  );
  const filterTerminalApprovalActionRequests = (
    actionRequests: Message["actionRequests"],
  ): Message["actionRequests"] => {
    if (!actionRequests?.length || terminalApprovalRequestIds.size === 0) {
      return actionRequests;
    }
    return actionRequests.filter(
      (request) =>
        request.actionType !== "tool_confirmation" ||
        !terminalApprovalRequestIds.has(request.requestId),
    );
  };
  const filteredDisplayContentParts = displayContentParts?.filter(
    (part) =>
      part.type !== "action_required" ||
      part.actionRequired.actionType !== "tool_confirmation" ||
      !terminalApprovalRequestIds.has(part.actionRequired.requestId),
  );
  const filteredActionRequests = filterTerminalApprovalActionRequests(
    message.actionRequests,
  );
  const messageWithoutTerminalApprovalDuplicates =
    terminalApprovalRequestIds.size > 0
      ? {
          ...message,
          actionRequests: filteredActionRequests,
          contentParts: filteredDisplayContentParts,
        }
      : message;
  const isTerminalRuntimeFailure =
    message.role === "assistant" && message.runtimeStatus?.phase === "failed";
  const hasMeaningfulAssistantTextContentPart = Boolean(
    displayContentParts?.some(
      (part) => part.type === "text" && !isTrivialAssistantFinalText(part.text),
    ),
  );
  const hasUserVisibleRuntimeFailureAnswer =
    isTerminalRuntimeFailure &&
    (!isTrivialAssistantFinalText(displayContent) ||
      hasMeaningfulAssistantTextContentPart ||
      hasFinalAnswerTimelineItem);
  const shouldSuppressRuntimeFailureProcessFlow =
    isTerminalRuntimeFailure && !hasUserVisibleRuntimeFailureAnswer;
  const hasTimelinePlanItem = Boolean(
    timelineItemsForDisplay?.some((item) => item.type === "plan"),
  );
  const primaryTimelineKey = timeline ? `leading:${timeline.turn.id}` : null;
  const shouldPreferCompactHistoricalTimeline =
    Boolean(primaryTimelineKey) &&
    isTerminalThreadTurnStatus(timeline?.turn.status) &&
    timeline?.turn.id !== activeCurrentTurnId;
  const shouldProjectHistoricalProcessThroughTimeline =
    shouldPreferCompactHistoricalTimeline;
  const hasStreamingOverlayText = Boolean(
    streamingFinalTextOverlayContent?.trim(),
  );
  const hasStructuredFinalTextAfterProcessBoundary =
    hasFinalTextAfterProcessBoundary(displayContentParts);
  const shouldHideAssistantTextWhileRunning =
    shouldHoldRunningWebRetrieval &&
    !hasFinalAnswerTimelineItem &&
    !hasStreamingOverlayText &&
    !hasStructuredFinalTextAfterProcessBoundary &&
    (hasFinalAnswerAfterRunningWebRetrieval ||
      Boolean(displayContent.trim()) ||
      !message.thinkingContent?.trim());
  const isActiveProcessOnlyOutput =
    message.role === "assistant" &&
    isActiveAssistantOutput &&
    !hasFinalAnswerTimelineItem &&
    !displayContent.trim() &&
    (hasCompletedOrRunningWebRetrievalTimelineProcess ||
      hasRunningWebRetrievalPart);
  const messageContentPartsOwnInlineProcessFlow =
    message.role === "assistant" &&
    hasInlineProcessContentParts(messageWithoutTerminalApprovalDuplicates, {
      displayContent,
      timelineItems: timelineItemsForDisplay,
    });
  const rendererContentPartsOwnInlineProcessFlow =
    messageContentPartsOwnInlineProcessFlow &&
    !shouldProjectHistoricalProcessThroughTimeline;
  const timelineOwnsInlineProcessFlow =
    !rendererContentPartsOwnInlineProcessFlow &&
    canTimelineOwnInlineProcessFlow(timelineItemsForDisplay);
  const shouldMergeTimelineProcessWithInlineThinking =
    rendererContentPartsOwnInlineProcessFlow &&
    !hasInlineNonThinkingProcessPart &&
    canTimelineOwnInlineProcessFlow(timelineItemsForDisplay);
  const timelineInlineContentParts =
    message.role === "assistant" &&
    !shouldSuppressRuntimeFailureProcessFlow &&
    (timelineOwnsInlineProcessFlow ||
      (messageContentPartsOwnInlineProcessFlow &&
        canMergeTimelineAsSparseProcessPatch(timelineItemsForDisplay)) ||
      shouldMergeTimelineProcessWithInlineThinking) &&
    !shouldPreferCompactHistoricalTimeline &&
    !shouldHideHistoricalRuntimeDetails
      ? buildTimelineInlineContentParts({
          displayContent: shouldHideAssistantTextWhileRunning
            ? ""
            : displayContent,
          processPrefaceContent:
            shouldHideAssistantTextWhileRunning &&
            !message.thinkingContent?.trim()
              ? displayContent
              : undefined,
          existingContentParts: filteredDisplayContentParts,
          items: timelineItemsForDisplay,
        })
      : undefined;
  const includeInlineProcessFlow =
    !shouldPreferCompactHistoricalTimeline &&
    !shouldHideHistoricalRuntimeDetails &&
    !shouldDeferMessageDetails &&
    !shouldSuppressRendererProcessFlow &&
    message.role === "assistant" &&
    !shouldSuppressRuntimeFailureProcessFlow &&
    (Boolean(timelineInlineContentParts?.length) ||
      shouldFoldSuppressedProcessFlow ||
      shouldKeepInlineProcessForActiveAssistant(
        message,
        isConversationTailAssistant,
        hasProcessTimelineItems,
        hasPersistedReasoningTimeline,
        Boolean(timeline),
        displayContent,
        isSending,
      ) ||
      // 消息 contentParts 已经持有过程顺序时，始终保留 inline process flow，
      // 让思考、工具、确认与文件改动按时间序穿插显示。
      rendererContentPartsOwnInlineProcessFlow);
  const filteredConversationDisplayContentParts =
    filterConversationDisplayContentParts(filteredDisplayContentParts, {
      includeProcessFlow: includeInlineProcessFlow,
      preserveToolUseParts: !hasProcessTimelineItems,
    });
  const timelineVisibleTextContentParts =
    message.role === "assistant" &&
    shouldProjectHistoricalProcessThroughTimeline
      ? buildTimelineVisibleTextContentParts({
          displayContent: shouldHideAssistantTextWhileRunning
            ? ""
            : displayContent,
          existingContentParts: filteredConversationDisplayContentParts,
          includeCommentary: false,
          items: timelineItemsForDisplay,
        })
      : undefined;
  const timelineFileChangesContentPart =
    message.role === "assistant" &&
    shouldProjectHistoricalProcessThroughTimeline
      ? buildTimelineFileChangesContentPart(timelineItemsForDisplay)
      : undefined;
  const timelineOwnedConversationContentParts =
    timelineVisibleTextContentParts || timelineFileChangesContentPart
      ? [
          ...(timelineFileChangesContentPart
            ? [timelineFileChangesContentPart]
            : []),
          ...(timelineVisibleTextContentParts || []),
          ...(filteredConversationDisplayContentParts || []).filter(
            (part) =>
              part.type !== "text" &&
              (!timelineFileChangesContentPart ||
                part.type !== "file_changes_batch"),
          ),
        ]
      : undefined;
  const ensuredConversationContentParts =
    message.role === "assistant"
      ? ensureInlineThinkingContentPart({
          parts: hideFinalAnswerContentPartsWhileRunning(
            mergeStreamingOverlayContentParts(
              timelineInlineContentParts ||
                timelineOwnedConversationContentParts ||
                filteredConversationDisplayContentParts,
              shouldHideAssistantTextWhileRunning
                ? null
                : streamingFinalTextOverlay,
            ),
            shouldHideAssistantTextWhileRunning,
          ),
          thinkingContent: message.thinkingContent,
          shouldEnsure:
            includeInlineProcessFlow &&
            Boolean(message.thinkingContent?.trim()) &&
            (Boolean(streamingFinalTextOverlayContent?.trim()) ||
              hasActiveTimelineTurn ||
              !hasPersistedReasoningTimeline),
        })
      : filteredDisplayContentParts;
  const conversationContentParts =
    message.role === "assistant"
      ? normalizeInlineThinkingContentParts(ensuredConversationContentParts)
      : ensuredConversationContentParts;
  const conversationThinkingContent =
    message.role === "assistant" && includeInlineProcessFlow
      ? message.thinkingContent
      : undefined;
  const shouldAllowLegacyToolCallsProcess =
    message.role === "assistant" &&
    includeInlineProcessFlow &&
    !hasProcessTimelineItems;
  const conversationToolCalls = shouldAllowLegacyToolCallsProcess
    ? message.toolCalls
    : undefined;
  const inlineProcessCoverage = resolveInlineProcessCoverage({
    contentParts: conversationContentParts,
    thinkingContent: conversationThinkingContent,
    toolCalls: conversationToolCalls,
    actionRequests: filteredActionRequests,
  });
  const shouldLetInlineProcessOwnActiveTurn =
    timeline !== null &&
    hasActiveTimelineTurn &&
    includeInlineProcessFlow &&
    inlineProcessCoverage.hasInlineProcessEntries;
  const timelineConversationItems = timeline
    ? (timelineItemsForDisplay || []).filter((item) =>
        shouldRenderConversationTimelineItem(
          item,
          timelineItemsForDisplay || [],
          {
            hasInlineRuntimeStatus: Boolean(message.runtimeStatus),
          },
        ),
      )
    : [];
  const timelineConversationItemIds =
    timelineConversationItems.length > 0
      ? new Set(timelineConversationItems.map((item) => item.id))
      : null;
  const isInlineCoveredTimelineItem = createInlineCoverageMatcher(
    inlineProcessCoverage,
  );
  const primaryTimelineItems = timeline
    ? (shouldSuppressRuntimeFailureProcessFlow
        ? []
        : timelineItemsForDisplay || []
      ).filter((item) => {
        if (shouldLetInlineProcessOwnActiveTurn) {
          return false;
        }

        if (!timelineConversationItemIds?.has(item.id)) {
          return false;
        }

        if (isDeferredTimelineItem(item)) {
          return false;
        }

        if (!inlineProcessCoverage.hasInlineProcessEntries) {
          return true;
        }

        if (isInlineCoveredTimelineItem(item)) {
          return false;
        }

        return true;
      })
    : [];
  const shouldHoldPreAnswerThinkingTimeline =
    timeline !== null &&
    shouldSuppressPreAnswerThinkingTimeline({
      message,
      turn: timeline.turn,
      items: primaryTimelineItems,
      displayContent,
    });
  const visiblePrimaryTimelineItems = shouldHoldPreAnswerThinkingTimeline
    ? []
    : primaryTimelineItems;
  const fileChangeBatchPaths = [
    ...collectFileChangeBatchPaths(message.contentParts),
    ...collectFileChangeBatchPaths(displayContentParts),
    ...collectFileChangeBatchPaths(conversationContentParts),
  ];
  const trailingTimelineItems = timeline
    ? dedupeDeferredTimelineItems(
        timelineConversationItems.filter((item) =>
          isDeferredTimelineItem(item),
        ),
      ).filter(
        (item) =>
          item.type !== "file_artifact" ||
          !isHiddenConversationArtifactPath(item.path),
      )
    : [];
  const hasDeferredHistoricalTimelineDetails =
    Boolean(timeline) &&
    isRestoredHistoryWindow &&
    shouldDeferThreadItemsScan &&
    isTerminalThreadTurnStatus(timeline?.turn.status) &&
    timeline?.turn.id !== activeCurrentTurnId;
  const compactPrimaryTimelineItems = shouldPreferCompactHistoricalTimeline
    ? [
        ...visiblePrimaryTimelineItems,
        ...trailingTimelineItems.filter(
          (item) => item.type !== "file_artifact",
        ),
      ]
    : visiblePrimaryTimelineItems;
  const visibleTrailingTimelineItems = trailingTimelineItems;
  const primaryTimeline =
    !shouldSuppressImageProcessFlow &&
    timeline &&
    (compactPrimaryTimelineItems.length > 0 ||
      hasDeferredHistoricalTimelineDetails)
      ? { ...timeline, items: compactPrimaryTimelineItems }
      : null;
  const trailingTimeline =
    !shouldSuppressImageProcessFlow &&
    timeline &&
    visibleTrailingTimelineItems.length > 0
      ? { ...timeline, items: visibleTrailingTimelineItems }
      : null;
  const hasTrailingArtifactTimelineItems = trailingTimelineItems.some(
    (item) => item.type === "file_artifact",
  );
  const trailingArtifactPaths = trailingTimelineItems.flatMap((item) =>
    item.type === "file_artifact" ? [item.path] : [],
  );
  const alreadyRenderedArtifactPaths = [
    ...trailingArtifactPaths,
    ...fileChangeBatchPaths,
  ];
  const timelineActionRequests = inlineProcessCoverage.actionRequestCounts.size
    ? undefined
    : filteredActionRequests;
  const primaryActionRequests =
    visiblePrimaryTimelineItems.length > 0 ? timelineActionRequests : undefined;
  const trailingActionRequests =
    visiblePrimaryTimelineItems.length === 0
      ? timelineActionRequests
      : undefined;
  const shouldSuppressInlineA2UI = false;
  const suppressedActionRequestId = null;
  const hasActiveStreamingOverlay = Boolean(
    streamingFinalTextOverlayContent?.trim(),
  );
  const {
    isCurrentInteractiveAssistantMessage,
    shouldReadOnlyInteractiveContent,
  } = resolveMessageInteractiveProjectionState({
    activePendingA2UISource,
    hasActiveInteractiveRuntime,
    hasActiveStreamingOverlay,
    isActiveProcessOnlyOutput,
    isSending,
    lastAssistantMessageId,
    message,
  });
  const usesProcessSeparatedFinalText =
    rendererContentPartsOwnInlineProcessFlow &&
    includeInlineProcessFlow &&
    hasFinalTextAfterProcessBoundary(conversationContentParts);
  const rendererConversationContentParts = usesProcessSeparatedFinalText
    ? resolveProcessSeparatedContentParts(conversationContentParts)
    : conversationContentParts;
  const timelineOwnsVisibleText =
    shouldProjectHistoricalProcessThroughTimeline ||
    timelineOwnsInlineProcessFlow;
  const timelineOwnedActionContent =
    timelineOwnsVisibleText &&
    (shouldProjectHistoricalProcessThroughTimeline ||
      messageContentPartsOwnInlineProcessFlow ||
      !displayContent.trim())
      ? resolveTimelineOwnedVisibleText(
          shouldProjectHistoricalProcessThroughTimeline
            ? rendererConversationContentParts
            : conversationContentParts,
          {
            includeCommentary: false,
          },
        )
      : null;
  const rawActionContent =
    timelineOwnedActionContent ??
    resolveAssistantActionContent({
      displayContent: shouldHideAssistantTextWhileRunning ? "" : displayContent,
      conversationContentParts: usesProcessSeparatedFinalText
        ? rendererConversationContentParts
        : conversationContentParts,
      useProcessSeparatedFinalText: usesProcessSeparatedFinalText,
    });
  const sanitizedRawActionContent = sanitizeProjectedMessageText(
    message,
    rawActionContent,
  );
  const runtimeFailureSanitizedActionContent =
    sanitizeRuntimeFailureAssistantText(message, sanitizedRawActionContent);
  const shouldSuppressDuplicatedFailureText =
    hasRuntimeFailureTimelineErrorItem &&
    isRuntimeFailureOnlyAssistantText(message, sanitizedRawActionContent) &&
    isRuntimeFailureDiagnosticAliasText(message, sanitizedRawActionContent);
  const shouldUseRuntimeFailureSanitizedText =
    runtimeFailureSanitizedActionContent !== sanitizedRawActionContent;
  const shouldBackfillEmptyRuntimeFailureText =
    message.role === "assistant" &&
    message.runtimeStatus?.phase === "failed" &&
    isTrivialAssistantFinalText(runtimeFailureSanitizedActionContent);
  const shouldUseRuntimeFailureFallbackText =
    !shouldSuppressDuplicatedFailureText &&
    shouldBackfillEmptyRuntimeFailureText;
  const runtimeFailureFallbackActionContent =
    shouldUseRuntimeFailureFallbackText
      ? resolveRuntimeFailureFallbackAssistantText(
          message,
          sanitizedRawActionContent,
        )
      : "";
  const actionContent = shouldSuppressDuplicatedFailureText
    ? ""
    : shouldUseRuntimeFailureFallbackText
      ? runtimeFailureFallbackActionContent
      : runtimeFailureSanitizedActionContent;
  const installedSkillMessageLabel =
    message.role === "user" ? resolveInstalledSkillMessageLabel(message) : null;
  const isUserCommandMessage =
    message.role === "user" &&
    !installedSkillMessageLabel &&
    Boolean(
      parseLeadingUserCommandTag(displayContent, message.inputCapabilityRoute),
    );
  const hasVisibleAssistantText = Boolean(actionContent);
  const hasImageWorkbenchLeadContent =
    imageWorkbenchDisplayState.hasLeadContent;
  const shouldCollapseLongHistoricalMessage =
    isRestoredHistoryWindow &&
    message.role === "assistant" &&
    !message.isThinking &&
    !shouldProjectHistoricalProcessThroughTimeline &&
    actionContent.length > MESSAGE_LIST_LONG_HISTORICAL_MESSAGE_THRESHOLD &&
    !expandedLongHistoricalMessageIds.has(message.id);
  const hasNonTextConversationContentParts = Boolean(
    conversationContentParts?.some((part) => part.type !== "text"),
  );
  const shouldFlattenHistoricalAssistantContent =
    isRestoredHistoryWindow &&
    message.role === "assistant" &&
    !message.isThinking &&
    !includeInlineProcessFlow &&
    !hasNonTextConversationContentParts &&
    actionContent.length > 0 &&
    !shouldCollapseLongHistoricalMessage;
  const shouldCompactHistoricalAssistantMessage =
    isRestoredHistoryWindow &&
    message.role === "assistant" &&
    !message.isThinking &&
    !shouldProjectHistoricalProcessThroughTimeline &&
    !focusedTimelineItemId &&
    !includeInlineProcessFlow &&
    !hasNonTextConversationContentParts &&
    !((filteredActionRequests || []).length > 0) &&
    !actionContent.includes("```a2ui") &&
    actionContent.length >
      MESSAGE_LIST_COMPACT_HISTORICAL_ASSISTANT_THRESHOLD &&
    !expandedLongHistoricalMessageIds.has(message.id) &&
    !expandedHistoricalAssistantMessageIds.has(message.id);
  const shouldPreviewHistoricalAssistantMessage =
    shouldCollapseLongHistoricalMessage ||
    shouldCompactHistoricalAssistantMessage;
  const historicalAssistantPreviewContent = shouldCollapseLongHistoricalMessage
    ? buildHistoricalMessagePreview(
        displayContent,
        MESSAGE_LIST_LONG_HISTORICAL_MESSAGE_PREVIEW_CHARS,
      )
    : shouldCompactHistoricalAssistantMessage
      ? buildHistoricalMessagePreview(
          displayContent,
          MESSAGE_LIST_COMPACT_HISTORICAL_ASSISTANT_PREVIEW_CHARS,
        )
      : "";
  const rendererContent = shouldCollapseLongHistoricalMessage
    ? buildHistoricalMessagePreview(
        actionContent,
        MESSAGE_LIST_LONG_HISTORICAL_MESSAGE_PREVIEW_CHARS,
      )
    : actionContent;
  const rawRendererRawContent = shouldSuppressDuplicatedFailureText
    ? actionContent
    : shouldUseRuntimeFailureSanitizedText
      ? actionContent
      : shouldCollapseLongHistoricalMessage ||
          shouldFlattenHistoricalAssistantContent
        ? rendererContent
        : usesProcessSeparatedFinalText
          ? actionContent
          : actionContent ||
            (shouldHideAssistantTextWhileRunning
              ? ""
              : visibleRawDisplayContent);
  const rendererRawContent = sanitizeProjectedMessageText(
    message,
    rawRendererRawContent,
  );
  const runtimeFailureTextContentParts =
    (shouldUseRuntimeFailureSanitizedText ||
      shouldUseRuntimeFailureFallbackText) &&
    actionContent
      ? ([{ type: "text" as const, text: actionContent }] satisfies NonNullable<
          typeof rendererConversationContentParts
        >)
      : undefined;
  const rendererContentParts =
    shouldSuppressDuplicatedFailureText ||
    shouldCollapseLongHistoricalMessage ||
    shouldFlattenHistoricalAssistantContent
      ? undefined
      : runtimeFailureTextContentParts ||
        (shouldPreferCompactHistoricalTimeline ||
        shouldHideHistoricalRuntimeDetails
          ? rendererConversationContentParts?.filter(
              (part) =>
                part.type === "text" ||
                part.type === "file_changes_batch" ||
                part.type === "media_reference",
            )
          : rendererConversationContentParts);
  const rendererHasProvenanceThinkingContentPart = Boolean(
    rendererContentParts?.some(
      (part) =>
        part.type === "thinking" &&
        part.text.trim().length > 0 &&
        (part.metadata?.source ||
          part.metadata?.threadItemId ||
          part.metadata?.turnId ||
          typeof part.metadata?.sequence === "number"),
    ),
  );
  const rendererThinkingContent =
    shouldPreferCompactHistoricalTimeline ||
    shouldHideHistoricalRuntimeDetails ||
    shouldCollapseLongHistoricalMessage
      ? undefined
      : rendererHasProvenanceThinkingContentPart
        ? undefined
        : conversationThinkingContent;
  const rendererToolCalls =
    shouldPreferCompactHistoricalTimeline ||
    shouldHideHistoricalRuntimeDetails ||
    shouldCollapseLongHistoricalMessage ||
    hasInlineToolUseContentPart(rendererConversationContentParts)
      ? undefined
      : conversationToolCalls;
  const rendererActionRequests =
    shouldPreferCompactHistoricalTimeline ||
    shouldHideHistoricalRuntimeDetails ||
    shouldCollapseLongHistoricalMessage ||
    shouldSuppressRendererProcessFlow
      ? undefined
      : filteredActionRequests;
  const rendererMarkdownRenderMode =
    shouldCollapseLongHistoricalMessage ||
    shouldFlattenHistoricalAssistantContent
      ? ("light" as const)
      : ("standard" as const);
  const canQuoteMessage = Boolean(actionContent);
  const canCopyMessage = Boolean(actionContent);
  const canSaveMessageAsSkill = Boolean(
    message.role === "assistant" &&
    !message.imageWorkbenchPreview &&
    !message.isThinking &&
    actionContent &&
    actionContent.length >= 24,
  );
  const {
    canSaveMessageAsKnowledge,
    hasArticleArtifactFrame,
    knowledgeArtifactSource,
    messageCanvasShortcutPath,
    messageCanvasShortcutTitle,
    messageSavedSiteContentTarget,
    shouldRenderMessageCanvasShortcut,
    visibleAssistantArtifacts,
  } = resolveMessageListItemArtifactProjection({
    actionContent,
    alreadyRenderedArtifactPaths,
    canOpenSavedSiteContent,
    hasTrailingArtifactTimelineItems,
    isImportedHistoryTurn,
    message,
    shouldSuppressImageProcessFlow,
  });
  const shouldDeferHistoricalMarkdownRender =
    shouldDeferMessageDetails &&
    message.role === "assistant" &&
    hasVisibleAssistantText &&
    !shouldPreviewHistoricalAssistantMessage &&
    !hasImages &&
    !hasNonTextConversationContentParts &&
    visibleAssistantArtifacts.length === 0 &&
    !shouldRenderMessageCanvasShortcut &&
    !message.imageWorkbenchPreview &&
    !message.taskPreview &&
    !hasStructuredHistoricalContentHint(actionContent);
  const hasAssistantRenderableContentBeforeFirstToken =
    hasVisibleAssistantText ||
    Boolean(conversationContentParts?.length) ||
    Boolean(message.thinkingContent?.trim()) ||
    Boolean(conversationThinkingContent?.trim()) ||
    Boolean(conversationToolCalls?.length) ||
    Boolean((rendererActionRequests || []).length > 0) ||
    Boolean(primaryTimeline) ||
    Boolean(trailingTimeline) ||
    Boolean((message.images || []).length > 0) ||
    visibleAssistantArtifacts.length > 0 ||
    shouldRenderMessageCanvasShortcut ||
    Boolean(message.imageWorkbenchPreview) ||
    Boolean(message.taskPreview);
  const hasLocalFirstTokenRuntime =
    message.isThinking || (isSending && message.id === lastAssistantMessageId);
  const hasActiveFirstTokenRuntime =
    hasActiveTimelineTurn || hasLocalFirstTokenRuntime;
  const shouldRenderFirstTokenRuntimeStatus =
    message.role === "assistant" &&
    isConversationTailAssistant &&
    !isRestoredHistoryWindow &&
    hasActiveFirstTokenRuntime &&
    shouldUseFirstTokenRuntimeStatus(message.runtimeStatus) &&
    !hasAssistantRenderableContentBeforeFirstToken;
  const shouldCollapseAssistantShell =
    message.role === "assistant" &&
    !shouldRenderFirstTokenRuntimeStatus &&
    !hasVisibleAssistantText &&
    !conversationContentParts?.length &&
    !conversationThinkingContent?.trim() &&
    !conversationToolCalls?.length &&
    !((rendererActionRequests || []).length > 0) &&
    !primaryTimeline &&
    !trailingTimeline &&
    !((message.images || []).length > 0) &&
    visibleAssistantArtifacts.length === 0 &&
    !shouldRenderMessageCanvasShortcut &&
    !message.imageWorkbenchPreview &&
    !message.taskPreview;
  const hasAssistantBodyContent =
    message.role !== "assistant" || !shouldCollapseAssistantShell;
  const imageWorkbenchRendererState = resolveImageWorkbenchRendererProcessState(
    {
      actionContent,
      message,
      rendererActionRequests,
      rendererContentParts,
      rendererToolCalls,
      shouldSuppressRendererProcessFlow,
    },
  );
  const arePrimaryTimelineDetailsDeferred =
    Boolean(primaryTimeline) &&
    shouldDeferThreadItemsScan &&
    isTerminalThreadTurnStatus(primaryTimeline?.turn.status) &&
    primaryTimeline?.turn.id !== activeCurrentTurnId;
  const shouldRenderCompactPrimaryTimeline =
    Boolean(primaryTimelineKey) &&
    isTerminalThreadTurnStatus(primaryTimeline?.turn.status) &&
    primaryTimeline?.turn.id !== activeCurrentTurnId &&
    shouldPreferCompactHistoricalTimeline;
  const shouldRenderPrimaryTimelineOutsideBubble =
    message.role === "assistant" &&
    Boolean(primaryTimeline) &&
    (hasVisibleAssistantText || !hasAssistantBodyContent);
  const shouldRenderProposedPlanBlocks = !hasTimelinePlanItem;
  const shouldRenderImageWorkbenchBareBubble =
    message.role === "assistant" &&
    Boolean(message.imageWorkbenchPreview) &&
    !primaryTimeline &&
    !trailingTimeline &&
    !hasImages &&
    visibleAssistantArtifacts.length === 0 &&
    !shouldRenderMessageCanvasShortcut &&
    !message.taskPreview;

  return {
    actionContent,
    arePrimaryTimelineDetailsDeferred,
    canCopyMessage,
    canQuoteMessage,
    canSaveMessageAsKnowledge,
    canSaveMessageAsSkill,
    displayContent,
    hasAssistantBodyContent,
    hasImageWorkbenchLeadContent,
    historicalAssistantPreviewContent,
    imageWorkbenchRendererState,
    inlineProcessCoverage,
    installedSkillMessageLabel,
    isActiveProcessOnlyOutput,
    isConversationTailAssistant,
    isCurrentInteractiveAssistantMessage,
    isUserCommandMessage,
    knowledgeArtifactSource,
    messageCanvasShortcutPath,
    messageCanvasShortcutTitle,
    messageSavedSiteContentTarget,
    primaryActionRequests,
    primaryTimeline,
    primaryTimelineKey,
    rawRuntimePeerContent,
    rendererActionRequests,
    rendererContent,
    rendererContentParts,
    rendererMarkdownRenderMode,
    rendererRawContent,
    rendererThinkingContent,
    rendererToolCalls,
    shouldCollapseLongHistoricalMessage,
    shouldDeferHistoricalMarkdownRender,
    shouldPreviewHistoricalAssistantMessage,
    shouldReadOnlyInteractiveContent,
    shouldRenderCompactPrimaryTimeline,
    shouldRenderFirstTokenRuntimeStatus,
    shouldRenderImageWorkbenchBareBubble,
    shouldRenderMessageCanvasShortcut,
    shouldRenderPrimaryTimelineOutsideBubble,
    shouldRenderProposedPlanBlocks,
    shouldRenderRuntimePeerCards,
    shouldSuppressInlineA2UI,
    shouldSuppressRendererProcessFlow,
    shouldSuppressStandaloneImageWorkbenchProcess,
    suppressedActionRequestId,
    trailingActionRequests,
    trailingTimeline,
    visibleAssistantArtifacts,
    hasArticleArtifactFrame,
  };
}
