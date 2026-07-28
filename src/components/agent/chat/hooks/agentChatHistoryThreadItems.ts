import type { AgentThreadItem } from "@/lib/api/agentProtocol";
import type { AgentSessionDetail } from "@/lib/api/agentRuntime/sessionTypes";
import type { Message } from "../types";
import {
  sanitizeContentPartsForDisplay,
  sanitizeMessageTextForDisplay,
} from "../utils/messageDisplaySanitizer";
import {
  isAgentMessageCommentaryPhase,
  resolveFinalAgentMessageItemIds,
  shouldUseAgentMessageAsFinalText,
} from "../utils/agentMessagePhase";
import {
  aggregateCanonicalPatchChanges,
  aggregateFileChanges,
  isFileMutationToolName,
} from "../utils/fileChangeSummary";
import { SKILL_INLINE_PROCESS_RETENTION } from "../utils/skillInlineProcessRetention";
import {
  buildImageTaskPreviewFromToolResult,
  buildTaskPreviewFromToolResult,
} from "../utils/taskPreviewFromToolResult";
import { normalizeToolResultMetadata } from "./agentChatToolResult";
import {
  toActionRequired,
  toToolCallState,
} from "../components/timeline-utils/itemConverters";
import { mergeAdjacentAssistantMessages } from "./agentChatHistoryAdjacentMerge";
import {
  appendTextToParts,
  appendThinkingToHistoryParts,
  parseHistoryTimestamp,
  mergeByKey,
  resolveThreadItemTimelinePosition,
} from "./agentChatHistoryPrimitives";
import {
  contentPartMetadataFromThreadReasoningItem,
  contentPartMetadataFromThreadToolItem,
} from "./agentChatHistoryReasoning";
import { extractThinkingContentFromParts } from "./agentChatHistoryPrimitives";
import {
  mergeHydratedToolStateContentParts,
  mergeImageWorkbenchPreview,
  mergeTaskPreview,
  resolveImageWorkbenchHistoryAssistantIntro,
} from "./agentChatHistoryProcess";
import { dedupeAdjacentHistoryMessages } from "./agentChatHistorySignatures";
import {
  isAuxiliaryHistoryTurn,
  readThreadItemText,
} from "./agentChatHistoryTimelineBasics";
import {
  imageGenerationContentPartFromThreadItem,
  mediaReferenceContentPartFromThreadItem,
  messageContentPartsFromAgentThreadItem,
} from "./agentThreadMessageContentParts";
import { isUpdatePlanToolName } from "../utils/toolNameFamily";
import { resolveSessionDetailTurnUsage } from "./agentChatHistoryUsage";

function readPlanRevisionId(metadata: unknown): string | null {
  if (!metadata || typeof metadata !== "object" || Array.isArray(metadata)) {
    return null;
  }
  const record = metadata as Record<string, unknown>;
  const revisionId = record.revisionId ?? record.revision_id;
  return typeof revisionId === "string" && revisionId.trim()
    ? revisionId.trim()
    : null;
}

function shouldHydratePlanItem(item: AgentThreadItem): boolean {
  return item.type === "plan" && Boolean(readPlanRevisionId(item.metadata));
}

function formatPlanItemAsProposedPlan(text: string): string {
  const normalized = text.trim();
  return normalized ? `<proposed_plan>\n${normalized}\n</proposed_plan>\n` : "";
}

function agentMessageContentPartMetadata(
  item: Extract<AgentThreadItem, { type: "agent_message" }>,
): Record<string, unknown> {
  return {
    source: "agent_thread_item",
    threadItemId: item.id,
    turnId: item.turn_id,
    sequence: resolveThreadItemTimelinePosition(item),
    ...(item.phase ? { phase: item.phase } : {}),
  };
}

function contentPartsText(parts: NonNullable<Message["contentParts"]>): string {
  return parts
    .map((part) => {
      if (part.type === "text") {
        return part.text.trim();
      }
      if (part.type === "media_reference") {
        return (
          part.reference.caption ||
          part.reference.title ||
          part.reference.uri
        ).trim();
      }
      return "";
    })
    .filter(Boolean)
    .join("\n\n");
}

function buildMessageFromThreadItem(
  item: AgentThreadItem,
  topicId: string,
  finalAgentMessageItemIds: Set<string>,
): Message | null {
  if (item.type !== "user_message" && item.type !== "agent_message") {
    return null;
  }
  if (
    item.type === "agent_message" &&
    (!shouldUseAgentMessageAsFinalText(item.phase) ||
      !finalAgentMessageItemIds.has(item.id))
  ) {
    return null;
  }

  const content =
    item.type === "user_message"
      ? readThreadItemText(item, ["content", "text", "message"])
      : readThreadItemText(item, ["text", "content", "message"]);
  const role = item.type === "user_message" ? "user" : "assistant";
  const itemContentParts =
    item.type === "agent_message"
      ? messageContentPartsFromAgentThreadItem(item)
      : [];
  const sanitizedContent = sanitizeMessageTextForDisplay(content, {
    role,
    hasImages: false,
  });
  if (!sanitizedContent && itemContentParts.length === 0) {
    return null;
  }
  const contentPartText = contentPartsText(itemContentParts);
  const messageContent = sanitizedContent || contentPartText;

  const timestamp = new Date(item.completed_at || item.updated_at);
  return {
    id: `${topicId}-timeline-${item.id}`,
    role,
    content: messageContent,
    contentParts:
      role === "assistant" && itemContentParts.length > 0
        ? itemContentParts
        : role === "assistant" && messageContent
          ? [
              {
                type: "text",
                text: messageContent,
              },
            ]
          : undefined,
    timestamp: Number.isNaN(timestamp.getTime()) ? new Date(0) : timestamp,
    runtimeTurnId: item.turn_id,
  };
}

export function hydrateSessionDetailMessagesFromThreadItems(
  detail: AgentSessionDetail,
  topicId: string,
): Message[] {
  const turnOrder = new Map<string, number>();
  (detail.turns || [])
    .filter((turn) => !isAuxiliaryHistoryTurn(turn))
    .forEach((turn, index) => {
      turnOrder.set(turn.id, index);
    });

  const sortedItems = collectDetailThreadItems(detail).sort((left, right) => {
    const leftTurnOrder =
      turnOrder.get(left.turn_id) ?? Number.MAX_SAFE_INTEGER;
    const rightTurnOrder =
      turnOrder.get(right.turn_id) ?? Number.MAX_SAFE_INTEGER;
    if (leftTurnOrder !== rightTurnOrder) {
      return leftTurnOrder - rightTurnOrder;
    }
    const leftTimelinePosition = resolveThreadItemTimelinePosition(left);
    const rightTimelinePosition = resolveThreadItemTimelinePosition(right);
    if (leftTimelinePosition !== rightTimelinePosition) {
      return leftTimelinePosition - rightTimelinePosition;
    }
    if (left.sequence !== right.sequence) {
      return left.sequence - right.sequence;
    }
    const leftTimestamp = parseHistoryTimestamp(
      left.started_at || left.updated_at,
    ).getTime();
    const rightTimestamp = parseHistoryTimestamp(
      right.started_at || right.updated_at,
    ).getTime();
    if (leftTimestamp !== rightTimestamp) {
      return leftTimestamp - rightTimestamp;
    }
    return left.id.localeCompare(right.id);
  });
  const finalAgentMessageItemIds = resolveFinalAgentMessageItemIds(sortedItems);

  const messages: Message[] = [];
  let assistantDraft: Message | null = null;

  const flushAssistantDraft = () => {
    if (!assistantDraft) {
      return;
    }
    if (
      !assistantDraft.content.trim() &&
      assistantDraft.imageWorkbenchPreview
    ) {
      assistantDraft.content = resolveImageWorkbenchHistoryAssistantIntro(
        assistantDraft.imageWorkbenchPreview,
      );
    }
    const sanitizedParts =
      sanitizeContentPartsForDisplay(assistantDraft.contentParts || [], {
        role: "assistant",
        hasImages: Boolean(assistantDraft.images?.length),
      }) || [];

    // 历史重建：把文件工具的 tool_use parts 聚合成 file_changes_batch，
    // 与流式路径（agentStreamEventProcessor.ts）保持一致。
    const finalParts = (() => {
      const hasFileMutationParts = sanitizedParts.some(
        (part) =>
          part.type === "tool_use" &&
          isFileMutationToolName(part.toolCall.name),
      );
      if (!hasFileMutationParts) {
        return sanitizedParts;
      }
      const nonFileParts = sanitizedParts.filter(
        (part) =>
          !(
            part.type === "tool_use" &&
            isFileMutationToolName(part.toolCall.name)
          ),
      );
      const fileCalls = (assistantDraft?.toolCalls || []).filter(
        (tc) => isFileMutationToolName(tc.name) && tc.status !== "running",
      );
      const aggregate = aggregateFileChanges(fileCalls);
      if (aggregate.fileCount === 0) {
        return sanitizedParts;
      }
      return [
        ...nonFileParts,
        { type: "file_changes_batch" as const, aggregate },
      ];
    })();

    messages.push({
      ...assistantDraft,
      contentParts: finalParts.length > 0 ? finalParts : undefined,
      thinkingContent: extractThinkingContentFromParts(sanitizedParts),
    });
    assistantDraft = null;
  };

  const ensureAssistantDraft = (item: AgentThreadItem): Message => {
    if (assistantDraft?.runtimeTurnId === item.turn_id) {
      return assistantDraft;
    }
    flushAssistantDraft();
    const timestamp = parseHistoryTimestamp(
      item.started_at || item.updated_at || item.completed_at,
    );
    assistantDraft = {
      id:
        item.type === "agent_message"
          ? `${topicId}-timeline-${item.id}`
          : `${topicId}-timeline-assistant-${item.turn_id}`,
      role: "assistant",
      content: "",
      contentParts: [],
      timestamp,
      isThinking: false,
      runtimeTurnId: item.turn_id,
      usage: resolveSessionDetailTurnUsage(detail, item.turn_id),
    };
    return assistantDraft;
  };

  const appendAssistantText = (
    draft: Message,
    text: string,
    item: Extract<AgentThreadItem, { type: "agent_message" }>,
  ) => {
    const itemContentParts = messageContentPartsFromAgentThreadItem(item);
    if (itemContentParts.length > 0) {
      const textContent = contentPartsText(itemContentParts);
      if (textContent) {
        draft.content = [draft.content.trim(), textContent]
          .filter(Boolean)
          .join("\n\n");
      }
      draft.contentParts = [...(draft.contentParts || []), ...itemContentParts];
      return;
    }

    const sanitizedText = sanitizeMessageTextForDisplay(text, {
      role: "assistant",
      hasImages: false,
    });
    if (!sanitizedText) {
      return;
    }
    draft.content = [draft.content.trim(), sanitizedText]
      .filter(Boolean)
      .join("\n\n");
    draft.contentParts = appendTextToParts(
      draft.contentParts || [],
      sanitizedText,
      {
        metadata: agentMessageContentPartMetadata(item),
        preserveEventBoundary: true,
      },
    );
  };

  const appendAssistantProcessText = (
    draft: Message,
    item: Extract<AgentThreadItem, { type: "agent_message" }>,
  ): boolean => {
    const itemContentParts = messageContentPartsFromAgentThreadItem(item);
    if (itemContentParts.length > 0) {
      draft.contentParts = [...(draft.contentParts || []), ...itemContentParts];
      return true;
    }

    const text = readThreadItemText(item, ["text", "content", "message"]);
    const sanitizedText = sanitizeMessageTextForDisplay(text, {
      role: "assistant",
      hasImages: false,
    });
    if (!sanitizedText) {
      return false;
    }
    if (isAgentMessageCommentaryPhase(item.phase) || !item.phase?.trim()) {
      draft.contentParts = appendTextToParts(
        draft.contentParts || [],
        sanitizedText,
        {
          metadata: agentMessageContentPartMetadata(item),
          preserveEventBoundary: true,
        },
      );
      return true;
    }
    draft.contentParts = appendThinkingToHistoryParts(
      draft.contentParts || [],
      sanitizedText,
      agentMessageContentPartMetadata(item),
    );
    return true;
  };

  for (const item of sortedItems) {
    if (item.type === "user_message" || item.type === "agent_message") {
      const message = buildMessageFromThreadItem(
        item,
        topicId,
        finalAgentMessageItemIds,
      );
      if (!message) {
        if (item.type === "agent_message") {
          const hasContentParts =
            messageContentPartsFromAgentThreadItem(item).length > 0;
          const processText = sanitizeMessageTextForDisplay(
            readThreadItemText(item, ["text", "content", "message"]),
            { role: "assistant", hasImages: false },
          );
          if (!hasContentParts && !processText) {
            continue;
          }
          const draft = ensureAssistantDraft(item);
          const appended = appendAssistantProcessText(draft, item);
          if (appended) {
            draft.timestamp = parseHistoryTimestamp(
              item.completed_at || item.updated_at || item.started_at,
            );
          }
        }
        continue;
      }
      if (item.type === "user_message") {
        flushAssistantDraft();
        messages.push(message);
        continue;
      }

      const draft = ensureAssistantDraft(item);
      appendAssistantText(draft, message.content, item);
      draft.timestamp = message.timestamp;
      continue;
    }

    if (item.type === "plan") {
      if (!shouldHydratePlanItem(item)) {
        continue;
      }
      const planText = formatPlanItemAsProposedPlan(item.text);
      if (!planText) {
        continue;
      }
      const draft = ensureAssistantDraft(item);
      draft.contentParts = appendTextToParts(
        draft.contentParts || [],
        planText,
      );
      draft.timestamp = parseHistoryTimestamp(
        item.completed_at || item.updated_at || item.started_at,
      );
      continue;
    }

    if (item.type === "reasoning") {
      const draft = ensureAssistantDraft(item);
      draft.contentParts = appendThinkingToHistoryParts(
        draft.contentParts || [],
        item.text,
        contentPartMetadataFromThreadReasoningItem(item),
      );
      draft.timestamp = parseHistoryTimestamp(
        item.completed_at || item.updated_at || item.started_at,
      );
      continue;
    }

    if (item.type === "media") {
      const mediaPart = mediaReferenceContentPartFromThreadItem(item);
      if (!mediaPart) {
        continue;
      }
      const draft = ensureAssistantDraft(item);
      draft.contentParts = [...(draft.contentParts || []), mediaPart];
      draft.timestamp = parseHistoryTimestamp(
        item.completed_at || item.updated_at || item.started_at,
      );
      continue;
    }

    if (item.type === "image_generation") {
      const imagePart = imageGenerationContentPartFromThreadItem(item);
      if (!imagePart) {
        continue;
      }
      const draft = ensureAssistantDraft(item);
      draft.contentParts = [...(draft.contentParts || []), imagePart];
      draft.timestamp = parseHistoryTimestamp(
        item.completed_at || item.updated_at || item.started_at,
      );
      continue;
    }

    if (
      item.type === "tool_call" ||
      item.type === "command_execution" ||
      item.type === "patch" ||
      item.type === "web_search"
    ) {
      if (item.type === "tool_call" && isUpdatePlanToolName(item.tool_name)) {
        continue;
      }
      const draft = ensureAssistantDraft(item);
      if (item.type === "patch" && item.changes?.length) {
        const aggregate = aggregateCanonicalPatchChanges(
          item.changes,
          item.status === "in_progress" ? "running" : "completed",
          item.file_status,
        );
        if (aggregate.fileCount > 0) {
          draft.contentParts = [
            ...(draft.contentParts || []),
            {
              type: "file_changes_batch",
              aggregate,
              metadata: {
                source: "thread_item_patch",
                threadItemId: item.id,
                sequence: resolveThreadItemTimelinePosition(item),
                turnId: item.turn_id,
              },
            },
          ];
          draft.timestamp = parseHistoryTimestamp(
            item.completed_at || item.updated_at || item.started_at,
          );
          continue;
        }
      }
      const toolCall = toToolCallState(item);
      if (!toolCall) {
        continue;
      }
      const toolArguments = toolCall.arguments;
      const itemMetadata = contentPartMetadataFromThreadToolItem(
        item,
        toolCall,
      );
      const normalizedResult =
        toolCall.result && typeof toolCall.result === "object"
          ? {
              ...toolCall.result,
              metadata:
                normalizeToolResultMetadata(
                  toolCall.result.metadata,
                  toolCall.result.output || "",
                  toolCall.result.error || "",
                ) || itemMetadata,
            }
          : toolCall.result;
      const normalizedToolCall =
        normalizedResult && typeof normalizedResult === "object"
          ? {
              ...toolCall,
              metadata: itemMetadata,
              result: normalizedResult,
            }
          : {
              ...toolCall,
              metadata: itemMetadata,
            };
      draft.toolCalls = mergeByKey(
        draft.toolCalls,
        [normalizedToolCall],
        (tool) => tool.id,
      );
      draft.contentParts = [
        ...(draft.contentParts || []),
        {
          type: "tool_use",
          toolCall: normalizedToolCall,
          ...(itemMetadata ? { metadata: itemMetadata } : {}),
        },
      ];
      const normalizedResultRecord =
        normalizedResult &&
        typeof normalizedResult === "object" &&
        !Array.isArray(normalizedResult)
          ? (normalizedResult as Record<string, unknown>)
          : undefined;
      const imageWorkbenchPreviewFromTool = buildImageTaskPreviewFromToolResult(
        {
          toolId: item.id,
          toolName: toolCall.name,
          toolArguments,
          toolResult: normalizedResultRecord,
          fallbackPrompt: draft.content,
        },
      );
      draft.imageWorkbenchPreview = mergeImageWorkbenchPreview(
        draft.imageWorkbenchPreview,
        imageWorkbenchPreviewFromTool || undefined,
      );
      draft.taskPreview = imageWorkbenchPreviewFromTool
        ? undefined
        : mergeTaskPreview(
            draft.taskPreview,
            buildTaskPreviewFromToolResult({
              toolId: item.id,
              toolName: toolCall.name,
              toolArguments,
              toolResult: normalizedResultRecord,
              fallbackPrompt: draft.content,
            }) || undefined,
          );
      draft.timestamp = parseHistoryTimestamp(
        item.completed_at || item.updated_at || item.started_at,
      );
      const metadata = normalizedResult?.metadata as
        | Record<string, unknown>
        | undefined;
      if (
        toolCall.name === "Skill" ||
        metadata?.tool_family === "skill" ||
        metadata?.skill_source === "SKILL.md"
      ) {
        draft.inlineProcessRetention = SKILL_INLINE_PROCESS_RETENTION;
      }
      continue;
    }

    if (
      item.type === "approval_request" ||
      item.type === "request_user_input"
    ) {
      const actionRequired = toActionRequired(item);
      if (!actionRequired) {
        continue;
      }
      const draft = ensureAssistantDraft(item);
      draft.actionRequests = mergeByKey(
        draft.actionRequests,
        [actionRequired],
        (action) => action.requestId,
      );
      draft.contentParts = mergeHydratedToolStateContentParts(
        draft.contentParts || [],
        [{ type: "action_required", actionRequired }],
      );
      draft.timestamp = parseHistoryTimestamp(
        item.completed_at || item.updated_at || item.started_at,
      );
    }
  }
  flushAssistantDraft();

  return mergeAdjacentAssistantMessages(
    dedupeAdjacentHistoryMessages(messages),
  );
}

export function collectDetailThreadItems(
  detail: AgentSessionDetail,
): AgentThreadItem[] {
  const seen = new Set<string>();
  const items: AgentThreadItem[] = [];
  for (const item of [
    ...(detail.items || []),
    ...(detail.thread_read?.thread_items || []),
  ]) {
    if (seen.has(item.id)) {
      continue;
    }
    seen.add(item.id);
    items.push(item);
  }
  return items;
}
