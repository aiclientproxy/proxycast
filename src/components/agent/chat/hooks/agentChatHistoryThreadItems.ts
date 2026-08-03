import type { AgentThreadItem } from "@/lib/api/agentProtocol";
import type { AgentSessionDetail } from "@/lib/api/agentRuntime/sessionTypes";
import type { Message, MessageImage } from "../types";
import { sanitizeMessageTextForDisplay } from "../utils/messageDisplaySanitizer";
import {
  resolveFinalAgentMessageItemIds,
  shouldUseAgentMessageAsFinalText,
} from "../utils/agentMessagePhase";
import { mergeAdjacentAssistantMessages } from "./agentChatHistoryAdjacentMerge";
import {
  parseHistoryTimestamp,
  resolveThreadItemTimelinePosition,
} from "./agentChatHistoryPrimitives";
import { dedupeAdjacentHistoryMessages } from "./agentChatHistorySignatures";
import { appendUniqueMessageImage } from "./agentChatHistoryImages";
import { normalizeHistoryImagePart } from "./agentChatToolResult";
import {
  isAuxiliaryHistoryTurn,
  readThreadItemText,
} from "./agentChatHistoryTimelineBasics";
import { dedupeImportedUserMessageItems } from "../utils/importedUserMessageDedupe";

function readUserMessageImages(item: AgentThreadItem): MessageImage[] {
  if (item.type !== "user_message" || !Array.isArray(item.content_parts)) {
    return [];
  }

  const images: MessageImage[] = [];
  for (const part of item.content_parts) {
    if (!part || typeof part !== "object" || part.type !== "image") {
      continue;
    }
    const image = normalizeHistoryImagePart(
      part as unknown as Record<string, unknown>,
    );
    if (image) {
      appendUniqueMessageImage(images, image);
    }
  }
  return images;
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
  const images = readUserMessageImages(item);
  const sanitizedContent = sanitizeMessageTextForDisplay(content, {
    role,
    hasImages: images.length > 0,
  });
  if (!sanitizedContent && images.length === 0) {
    return null;
  }

  const timestamp = new Date(item.completed_at || item.updated_at);
  const agentItemMetadata =
    item.type === "agent_message"
      ? {
          source: "agent_thread_item",
          threadItemId: item.id,
          turnId: item.turn_id,
          sequence: resolveThreadItemTimelinePosition(item),
          ...(item.phase ? { phase: item.phase } : {}),
        }
      : undefined;
  return {
    id: `${topicId}-timeline-${item.id}`,
    role,
    content: sanitizedContent,
    images: images.length > 0 ? images : undefined,
    contentParts:
      role === "assistant"
        ? [
            {
              type: "text",
              text: sanitizedContent,
              metadata: agentItemMetadata,
            },
          ]
        : undefined,
    isThinking: role === "assistant" ? false : undefined,
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

  const messages = sortedItems.flatMap((item) => {
    if (item.type !== "user_message" && item.type !== "agent_message") {
      return [];
    }
    const message = buildMessageFromThreadItem(
      item,
      topicId,
      finalAgentMessageItemIds,
    );
    return message ? [message] : [];
  });

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
  return dedupeImportedUserMessageItems(items);
}
