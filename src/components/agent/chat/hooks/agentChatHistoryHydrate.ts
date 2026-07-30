import type { AgentSessionDetail } from "@/lib/api/agentRuntime/sessionTypes";
import type {
  ContentPart,
  Message,
  MessageImage,
  MessageImageWorkbenchPreview,
  MessageTaskPreview,
} from "../types";
import { projectConversationMessagesByRuntimeTurn } from "../utils/conversationTimelineOrdering";
import {
  sanitizeContentPartsForDisplay,
  sanitizeMessageTextForDisplay,
} from "../utils/messageDisplaySanitizer";
import {
  buildImageTaskPreviewFromToolResult,
  buildTaskPreviewFromToolResult,
} from "../utils/taskPreviewFromToolResult";
import {
  extractLimeToolMetadataBlock,
  isToolResultSuccessful,
  normalizeHistoryImagePart,
  normalizeToolResultImages,
  normalizeToolResultMetadata,
  resolveHistoryUserDataText,
  stringifyToolArguments,
} from "./agentChatToolResult";
import { mergeAdjacentAssistantMessages } from "./agentChatHistoryAdjacentMerge";
import { hydrateSessionDetailMessagesFromArtifacts } from "./agentChatHistoryArtifacts";
import {
  appendHistoryMessageAttachments,
  appendUniqueMessageImage,
} from "./agentChatHistoryImages";
import { mergeHydratedMessagesWithLocalState } from "./agentChatHistoryLocalMerge";
import {
  compactHistoricalRestoreMessage,
  resolveHistoryToolName,
  shouldCompactCompletedSessionHistory,
} from "./agentChatHistoryNormalize";
import {
  appendTextWithOverlapDetection,
  appendThinkingToHistoryParts,
  extractThinkingContentFromParts,
  normalizeHistoryPartType,
  readHistoryString,
} from "./agentChatHistoryPrimitives";
import {
  mergeImageWorkbenchPreview,
  mergeTaskPreview,
  contentPartContainsProcess,
  resolveImageWorkbenchHistoryAssistantIntro,
} from "./agentChatHistoryProcess";
import {
  hydrateFailedRuntimeReadModelMessage,
  hydrateSessionDetailMessagesFromThreadReadToolCalls,
} from "./agentChatHistoryReadModel";
import { dedupeAdjacentHistoryMessages } from "./agentChatHistorySignatures";
import {
  collectDetailThreadItems,
  hydrateSessionDetailMessagesFromThreadItems,
} from "./agentChatHistoryThreadItems";
import {
  hydrateSessionDetailMessagesFromTurns,
  isAuxiliaryHistoryTurn,
} from "./agentChatHistoryTimelineBasics";
import {
  mergeMissingUserMessagesFromTimeline,
  shouldMergeTimelineProcessMessages,
} from "./agentChatHistoryTimelineMerge";
import type { HydrateSessionDetailMessagesOptions } from "./agentChatHistoryTypes";
import {
  normalizeHistoryUsage,
  resolveSessionDetailTurnUsage,
} from "./agentChatHistoryUsage";

export const hydrateSessionDetailMessages = (
  detail: AgentSessionDetail,
  topicId: string,
  options: HydrateSessionDetailMessagesOptions = {},
): Message[] => {
  const historyToolNameById = new Map<string, string>();
  const historyToolArgumentsById = new Map<string, string | undefined>();
  const compactCompletedHistory =
    options.compactCompletedHistory === true &&
    shouldCompactCompletedSessionHistory(detail);

  const loadedMessages: Message[] = detail.messages
    .filter(
      (msg) =>
        msg.role === "user" || msg.role === "assistant" || msg.role === "tool",
    )
    .flatMap((msg, index) => {
      const contentParts: ContentPart[] = [];
      const textParts: string[] = [];
      const toolCalls: Message["toolCalls"] = [];
      const images: MessageImage[] = [];
      const messageTimestamp = new Date(msg.timestamp * 1000);
      const rawParts = Array.isArray(msg.content) ? msg.content : [];
      let imageWorkbenchPreview: MessageImageWorkbenchPreview | undefined;
      let taskPreview: MessageTaskPreview | undefined;

      const appendText = (value: unknown) => {
        if (typeof value !== "string") return;
        const normalized = value.trim();
        if (!normalized) return;
        const lastPart = contentParts[contentParts.length - 1];
        if (lastPart?.type === "text") {
          const mergedText = appendTextWithOverlapDetection(
            lastPart.text,
            normalized,
          );
          lastPart.text =
            mergedText === `${lastPart.text}${normalized}`
              ? `${lastPart.text}\n${normalized}`
              : mergedText;
          textParts[textParts.length - 1] = lastPart.text;
          return;
        }

        textParts.push(normalized);
        contentParts.push({ type: "text", text: normalized });
      };

      for (const rawPart of rawParts) {
        if (!rawPart || typeof rawPart !== "object") continue;
        const part = rawPart as unknown as Record<string, unknown>;
        const partType = normalizeHistoryPartType(part.type);

        if (
          partType === "text" ||
          partType === "input_text" ||
          partType === "output_text"
        ) {
          appendText(part.text ?? part.content);
          continue;
        }

        if (partType === "skill" || partType === "mention") {
          const name = readHistoryString(part.name);
          const path = readHistoryString(part.path);
          if (name && path) {
            appendText(`${partType === "skill" ? "$" : "@"}${name} (${path})`);
          }
          continue;
        }

        if (
          compactCompletedHistory &&
          (partType === "thinking" ||
            partType === "reasoning" ||
            partType === "tool_request")
        ) {
          continue;
        }

        if (partType === "thinking" || partType === "reasoning") {
          const rawThinking =
            typeof part.thinking === "string"
              ? part.thinking
              : typeof part.reasoning === "string"
                ? part.reasoning
                : typeof part.text === "string"
                  ? part.text
                  : typeof part.content === "string"
                    ? part.content
                    : "";

          if (rawThinking) {
            const mergedThinkingParts = appendThinkingToHistoryParts(
              contentParts,
              rawThinking,
            );
            contentParts.splice(0, contentParts.length, ...mergedThinkingParts);
          }
          continue;
        }

        if (
          partType === "image" ||
          partType === "input_image" ||
          partType === "image_url"
        ) {
          const normalizedImage = normalizeHistoryImagePart(part);
          if (normalizedImage) {
            appendUniqueMessageImage(images, normalizedImage);
          }
          continue;
        }

        if (partType === "tool_request") {
          if (!part.id || typeof part.id !== "string") continue;
          const nestedToolCall =
            part.toolCall && typeof part.toolCall === "object"
              ? (part.toolCall as Record<string, unknown>)
              : part.tool_call && typeof part.tool_call === "object"
                ? (part.tool_call as Record<string, unknown>)
                : undefined;
          const nestedToolCallValue =
            nestedToolCall?.value && typeof nestedToolCall.value === "object"
              ? (nestedToolCall.value as Record<string, unknown>)
              : undefined;
          const toolName =
            (typeof part.tool_name === "string" && part.tool_name.trim()) ||
            (typeof part.toolName === "string" && part.toolName.trim()) ||
            (typeof part.name === "string" && part.name.trim()) ||
            (typeof nestedToolCallValue?.name === "string" &&
              nestedToolCallValue.name.trim()) ||
            resolveHistoryToolName(part.id, historyToolNameById);
          const rawArguments =
            part.arguments ??
            nestedToolCallValue?.arguments ??
            nestedToolCall?.arguments;
          const toolCall = {
            id: part.id,
            name: toolName,
            arguments: stringifyToolArguments(rawArguments),
            status: "running" as const,
            startTime: messageTimestamp,
          };
          historyToolNameById.set(part.id, toolName);
          historyToolArgumentsById.set(part.id, toolCall.arguments);
          toolCalls.push(toolCall);
          contentParts.push({ type: "tool_use", toolCall });
          continue;
        }

        if (partType === "tool_response") {
          if (!part.id || typeof part.id !== "string") continue;
          const toolName = resolveHistoryToolName(part.id, historyToolNameById);
          const rawOutputText =
            typeof part.output === "string" ? part.output : "";
          const rawErrorText = typeof part.error === "string" ? part.error : "";
          const normalizedOutput = extractLimeToolMetadataBlock(rawOutputText);
          const normalizedError = extractLimeToolMetadataBlock(rawErrorText);
          const metadata = normalizeToolResultMetadata(
            part.metadata,
            rawOutputText,
            rawErrorText,
          );
          const normalizedResult = {
            success: part.success !== false,
            output: normalizedOutput.text,
            error: normalizedError.text || undefined,
            images: normalizeToolResultImages(
              part.images,
              normalizedOutput.text,
              metadata,
            ),
            metadata,
            structuredContent:
              part.structuredContent ?? part.structured_content,
          };
          const success = isToolResultSuccessful(normalizedResult);
          const normalizedResultRecord =
            normalizedResult &&
            typeof normalizedResult === "object" &&
            !Array.isArray(normalizedResult)
              ? (normalizedResult as Record<string, unknown>)
              : undefined;
          const toolArguments = historyToolArgumentsById.get(part.id);
          const imageWorkbenchPreviewFromTool =
            buildImageTaskPreviewFromToolResult({
              toolId: part.id,
              toolName,
              toolArguments,
              toolResult: normalizedResultRecord,
              fallbackPrompt: textParts.join("\n").trim(),
            });
          if (compactCompletedHistory) {
            imageWorkbenchPreview = mergeImageWorkbenchPreview(
              imageWorkbenchPreview,
              imageWorkbenchPreviewFromTool || undefined,
            );
            continue;
          }
          const toolCall = {
            id: part.id,
            name: toolName,
            arguments: toolArguments,
            status: success ? ("completed" as const) : ("failed" as const),
            startTime: messageTimestamp,
            endTime: messageTimestamp,
            result: {
              ...normalizedResult,
              success,
            },
          };
          toolCalls.push(toolCall);
          contentParts.push({ type: "tool_use", toolCall });
          imageWorkbenchPreview = mergeImageWorkbenchPreview(
            imageWorkbenchPreview,
            imageWorkbenchPreviewFromTool || undefined,
          );
          taskPreview = imageWorkbenchPreviewFromTool
            ? undefined
            : mergeTaskPreview(
                taskPreview,
                buildTaskPreviewFromToolResult({
                  toolId: part.id,
                  toolName,
                  toolArguments,
                  toolResult: normalizedResultRecord,
                  fallbackPrompt: textParts.join("\n").trim(),
                }) || undefined,
              );
          continue;
        }

        if (partType !== "action_required") continue;

        const actionType =
          typeof part.action_type === "string" ? part.action_type : "";
        if (actionType !== "elicitation_response") continue;

        const data =
          part.data && typeof part.data === "object"
            ? (part.data as Record<string, unknown>)
            : undefined;
        const userData =
          data && "user_data" in data ? data.user_data : part.data;
        const resolved = resolveHistoryUserDataText(userData);
        if (!resolved) continue;

        textParts.push(resolved);
        contentParts.push({ type: "text", text: resolved });
      }

      appendHistoryMessageAttachments(images, msg);

      const rawContent = textParts.join("\n").trim();
      let normalizedRole =
        msg.role === "tool" ? "assistant" : (msg.role as "user" | "assistant");
      const sanitizedRawContent = sanitizeMessageTextForDisplay(rawContent, {
        role: normalizedRole,
        hasImages: images.length > 0,
      });
      const imageWorkbenchAssistantIntro =
        normalizedRole === "assistant" && imageWorkbenchPreview
          ? resolveImageWorkbenchHistoryAssistantIntro(imageWorkbenchPreview)
          : "";
      const content = sanitizedRawContent || imageWorkbenchAssistantIntro;
      const displayContentParts =
        !sanitizedRawContent && imageWorkbenchAssistantIntro
          ? [
              { type: "text" as const, text: imageWorkbenchAssistantIntro },
              ...contentParts,
            ]
          : contentParts;
      const sanitizedContentParts =
        sanitizeContentPartsForDisplay(displayContentParts, {
          role: normalizedRole,
          hasImages: images.length > 0,
        }) || [];

      const hasToolMetadata =
        toolCalls.length > 0 ||
        sanitizedContentParts.some((part) => part.type === "tool_use");
      const hasProcessPreview =
        Boolean(imageWorkbenchPreview) || Boolean(taskPreview);

      if (normalizedRole === "user" && !content && images.length === 0) {
        if (hasToolMetadata || hasProcessPreview) {
          normalizedRole = "assistant";
        } else {
          return [];
        }
      }

      const runtimeTurnId =
        readHistoryString(msg.runtimeTurnId) ||
        readHistoryString(msg.runtime_turn_id);
      const usage =
        normalizedRole === "assistant"
          ? (normalizeHistoryUsage(msg.usage) ??
            resolveSessionDetailTurnUsage(detail, runtimeTurnId))
          : undefined;

      if (
        !content &&
        images.length === 0 &&
        sanitizedContentParts.length === 0 &&
        toolCalls.length === 0 &&
        !hasProcessPreview &&
        !(normalizedRole === "assistant" && usage)
      ) {
        return [];
      }
      const hydratedMessage: Message = {
        id: readHistoryString(msg.id) || `${topicId}-${index}`,
        role: normalizedRole,
        content,
        images: images.length > 0 ? images : undefined,
        contentParts:
          sanitizedContentParts.length > 0 ? sanitizedContentParts : undefined,
        toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
        timestamp: messageTimestamp,
        isThinking: false,
        usage,
        thinkingContent: extractThinkingContentFromParts(sanitizedContentParts),
        imageWorkbenchPreview,
        taskPreview,
        runtimeTurnId: runtimeTurnId || undefined,
      };

      return [
        compactCompletedHistory
          ? compactHistoricalRestoreMessage(hydratedMessage)
          : hydratedMessage,
      ];
    });

  const hydratedMessages = mergeAdjacentAssistantMessages(
    dedupeAdjacentHistoryMessages(loadedMessages),
  );
  const threadItemTimelineMessages =
    options.includeTimelineFallback === false
      ? []
      : hydrateSessionDetailMessagesFromThreadItems(detail, topicId);
  const detailThreadItems = collectDetailThreadItems(detail);
  const auxiliaryTurnIds = new Set(
    (detail.turns || [])
      .filter((turn) => isAuxiliaryHistoryTurn(turn))
      .map((turn) => turn.id),
  );
  const hasCanonicalRenderableItems = detailThreadItems.some(
    (item) => !auxiliaryTurnIds.has(item.turn_id),
  );
  const hasCanonicalProcessItems = detailThreadItems.some(
    (item) =>
      !auxiliaryTurnIds.has(item.turn_id) &&
      item.type !== "user_message" &&
      item.type !== "agent_message",
  );
  const hasCanonicalConversationItems = detailThreadItems.some(
    (item) => item.type === "user_message" || item.type === "agent_message",
  );
  const artifactTimelineMessages =
    options.includeTimelineFallback === false
      ? []
      : hydrateSessionDetailMessagesFromArtifacts(detail, topicId);
  const timelineFallbackMessages = [
    ...threadItemTimelineMessages,
    ...artifactTimelineMessages,
  ];
  const hasThreadItemProcessMessages = threadItemTimelineMessages.some(
    (message) =>
      message.role === "assistant" &&
      ((message.contentParts || []).some(contentPartContainsProcess) ||
        (message.toolCalls?.length || 0) > 0 ||
        (message.actionRequests?.length || 0) > 0),
  );
  const failedRuntimeMessage = hydrateFailedRuntimeReadModelMessage(
    detail,
    topicId,
  );
  const threadReadToolCallMessages =
    hasThreadItemProcessMessages || hasCanonicalProcessItems
    ? []
    : hydrateSessionDetailMessagesFromThreadReadToolCalls(detail, topicId);
  const readModelProcessMessages = [
    ...threadReadToolCallMessages,
    ...(failedRuntimeMessage ? [failedRuntimeMessage] : []),
  ];
  const timelineMessages =
    readModelProcessMessages.length === 0 &&
    timelineFallbackMessages.length === 0
      ? []
      : mergeAdjacentAssistantMessages(
          dedupeAdjacentHistoryMessages([
            ...(hasThreadItemProcessMessages ? threadItemTimelineMessages : []),
            ...readModelProcessMessages,
            ...(hasThreadItemProcessMessages ? [] : threadItemTimelineMessages),
            ...artifactTimelineMessages,
          ]),
        );

  if (hasCanonicalConversationItems && threadItemTimelineMessages.length > 0) {
    const canonicalMessages =
      timelineMessages.length > 0
        ? timelineMessages
        : threadItemTimelineMessages;
    const enrichedCanonicalMessages =
      hydratedMessages.length > 0
        ? mergeHydratedMessagesWithLocalState(
            hydratedMessages,
            canonicalMessages,
            { preferHydratedAssistantOutput: true },
          )
        : canonicalMessages;
    return projectConversationMessagesByRuntimeTurn(
      options.includeTimelineFallbackUsers === true
        ? mergeMissingUserMessagesFromTimeline(
            enrichedCanonicalMessages,
            detail,
            topicId,
          )
        : enrichedCanonicalMessages,
    );
  }

  if (hydratedMessages.length > 0) {
    const hydratedWithTimelineProcess = shouldMergeTimelineProcessMessages(
      timelineMessages,
    )
      ? mergeHydratedMessagesWithLocalState(timelineMessages, hydratedMessages)
      : hydratedMessages;
    const hydratedWithFailedRuntime = failedRuntimeMessage
      ? mergeAdjacentAssistantMessages(
          dedupeAdjacentHistoryMessages([
            ...hydratedWithTimelineProcess,
            ...(!hydratedWithTimelineProcess.some(
              (message) => message.id === failedRuntimeMessage.id,
            )
              ? [failedRuntimeMessage]
              : []),
          ]),
        )
      : hydratedWithTimelineProcess;
    return projectConversationMessagesByRuntimeTurn(
      mergeMissingUserMessagesFromTimeline(
        hydratedWithFailedRuntime,
        detail,
        topicId,
      ),
    );
  }

  if (detail.messages.length > 0) {
    return projectConversationMessagesByRuntimeTurn(
      timelineMessages.length > 0 ? timelineMessages : hydratedMessages,
    );
  }

  if (options.includeTimelineFallback !== false) {
    if (timelineMessages.length > 0) {
      return projectConversationMessagesByRuntimeTurn(
        options.includeTimelineFallbackUsers === true
          ? mergeMissingUserMessagesFromTimeline(
              timelineMessages,
              detail,
              topicId,
            )
          : timelineMessages,
      );
    }

    if (hasCanonicalRenderableItems) {
      return [];
    }

    return projectConversationMessagesByRuntimeTurn(
      hydrateSessionDetailMessagesFromTurns(detail, topicId),
    );
  }

  return [];
};
