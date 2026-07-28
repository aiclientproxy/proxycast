import type { Dispatch, SetStateAction } from "react";
import type { AgentThreadItem } from "@/lib/api/agentProtocol";
import type { Message } from "../types";
import {
  isAgentMessageCommentaryPhase,
  isAgentMessageFinalAnswerPhase,
} from "../utils/agentMessagePhase";
import {
  buildAgentTextDeltaContentPartMetadata,
  readContentPartSequence,
} from "../utils/contentPartTimeline";
import {
  imageGenerationContentPartFromThreadItem,
  mediaReferenceContentPartFromThreadItem,
  messageContentPartsFromAgentThreadItem,
} from "./agentThreadMessageContentParts";
import { resolveThreadItemTimelinePosition } from "./agentChatHistoryPrimitives";

type MessageContentPart = NonNullable<Message["contentParts"]>[number];
type MessageContentParts = NonNullable<Message["contentParts"]>;

function isComparableSequence(value: unknown): value is number {
  return (
    typeof value === "number" &&
    Number.isFinite(value) &&
    value < Number.MAX_SAFE_INTEGER
  );
}

function threadItemToolSequenceById(
  items: readonly AgentThreadItem[],
  turnId?: string | null,
): Map<string, number> {
  const sequenceById = new Map<string, number>();
  const normalizedTurnId = turnId?.trim();
  for (const item of items) {
    if (item.type !== "tool_call") {
      continue;
    }
    if (normalizedTurnId && item.turn_id !== normalizedTurnId) {
      continue;
    }
    sequenceById.set(item.id, resolveThreadItemTimelinePosition(item));
  }
  return sequenceById;
}

function contentPartSequence(
  part: MessageContentPart,
  sequenceByToolId: Map<string, number>,
): number | null {
  const metadataSequence = readContentPartSequence(part);
  if (metadataSequence !== null) {
    return metadataSequence;
  }
  if (part.type === "tool_use") {
    const itemSequence = sequenceByToolId.get(part.toolCall.id);
    return isComparableSequence(itemSequence) ? itemSequence : null;
  }
  return null;
}

function isAgentMessageTextPartForItem(
  part: MessageContentPart,
  itemId: string,
): boolean {
  return (
    part.type === "text" &&
    (part.metadata?.itemId === itemId || part.metadata?.threadItemId === itemId)
  );
}

function isAgentMessageStructuredPartForItem(
  part: MessageContentPart,
  itemId: string,
): boolean {
  const source = part.metadata?.source;
  return (
    (source === "agent_text_delta" || source === "agent_media_reference") &&
    (part.metadata?.itemId === itemId || part.metadata?.threadItemId === itemId)
  );
}

function insertContentPartBySequence(params: {
  nextPart: MessageContentPart;
  parts: MessageContentParts;
  sequence: number;
  sequenceByToolId: Map<string, number>;
}): MessageContentParts {
  if (!isComparableSequence(params.sequence)) {
    return [...params.parts, params.nextPart];
  }

  const insertBeforeIndex = params.parts.findIndex((part) => {
    const existingSequence = contentPartSequence(part, params.sequenceByToolId);
    return existingSequence !== null && existingSequence > params.sequence;
  });

  if (insertBeforeIndex < 0) {
    return [...params.parts, params.nextPart];
  }

  return [
    ...params.parts.slice(0, insertBeforeIndex),
    params.nextPart,
    ...params.parts.slice(insertBeforeIndex),
  ];
}

function shouldSyncAgentMessageContentPartPhase(
  phase?: string | null,
): boolean {
  return (
    isAgentMessageCommentaryPhase(phase) ||
    isAgentMessageFinalAnswerPhase(phase)
  );
}

function upsertAgentMessageStructuredContentParts(params: {
  item: Extract<AgentThreadItem, { type: "agent_message" }>;
  parts: MessageContentParts;
  threadItems?: readonly AgentThreadItem[];
}): MessageContentParts {
  const itemParts = messageContentPartsFromAgentThreadItem(params.item);
  if (itemParts.length === 0) {
    return params.parts;
  }

  const sequenceByToolId = threadItemToolSequenceById(
    params.threadItems ?? [],
    params.item.turn_id,
  );
  const timelinePosition = resolveThreadItemTimelinePosition(params.item);
  const retainedParts = params.parts.filter(
    (part) => !isAgentMessageStructuredPartForItem(part, params.item.id),
  );
  return itemParts.reduce<MessageContentParts>(
    (parts, nextPart) =>
      insertContentPartBySequence({
        nextPart: {
          ...nextPart,
          metadata: {
            ...(nextPart.metadata ?? {}),
            sequence: timelinePosition,
          },
        },
        parts,
        sequence: timelinePosition,
        sequenceByToolId,
      }),
    retainedParts,
  );
}

function upsertAgentMessageContentPart(params: {
  item: AgentThreadItem;
  parts: MessageContentParts;
  threadItems?: readonly AgentThreadItem[];
}): MessageContentParts {
  if (params.item.type === "media" || params.item.type === "image_generation") {
    const nextPart =
      params.item.type === "media"
        ? mediaReferenceContentPartFromThreadItem(params.item)
        : imageGenerationContentPartFromThreadItem(params.item);
    if (!nextPart) {
      return params.parts;
    }
    const retainedParts = params.parts.filter(
      (part) =>
        !(
          part.type === "media_reference" &&
          part.metadata?.threadItemId === params.item.id
        ),
    );
    return insertContentPartBySequence({
      nextPart,
      parts: retainedParts,
      sequence: resolveThreadItemTimelinePosition(params.item),
      sequenceByToolId: threadItemToolSequenceById(
        params.threadItems ?? [],
        params.item.turn_id,
      ),
    });
  }

  if (
    params.item.type !== "agent_message" ||
    !shouldSyncAgentMessageContentPartPhase(params.item.phase)
  ) {
    return params.parts;
  }

  const item = params.item;
  const timelinePosition = resolveThreadItemTimelinePosition(item);
  const structuredParts = upsertAgentMessageStructuredContentParts({
    item,
    parts: params.parts,
    threadItems: params.threadItems,
  });
  if (structuredParts !== params.parts) {
    return structuredParts;
  }

  const text = item.text.trim();
  if (!text) {
    return params.parts;
  }

  const metadata = buildAgentTextDeltaContentPartMetadata({
    itemId: item.id,
    phase: item.phase,
    sequence: timelinePosition,
    turnId: item.turn_id,
  });
  const nextPart: MessageContentPart = {
    type: "text",
    text,
    ...(metadata ? { metadata } : {}),
  };
  const existingIndex = params.parts.findIndex((part) =>
    isAgentMessageTextPartForItem(part, item.id),
  );

  if (existingIndex >= 0) {
    const existingPart = params.parts[existingIndex];
    if (
      existingPart?.type === "text" &&
      existingPart.text === text &&
      existingPart.metadata?.turnId === item.turn_id &&
      existingPart.metadata?.phase === item.phase
    ) {
      return params.parts;
    }
    const nextParts = [...params.parts];
    nextParts[existingIndex] = nextPart;
    return nextParts;
  }

  const sequenceByToolId = threadItemToolSequenceById(
    params.threadItems ?? [],
    item.turn_id,
  );
  return insertContentPartBySequence({
    nextPart,
    parts: params.parts,
    sequence: timelinePosition,
    sequenceByToolId,
  });
}

export function mergeAssistantAgentMessageContentPartsFromThreadItems(params: {
  items: readonly AgentThreadItem[];
  parts?: MessageContentParts;
  turnId?: string | null;
}): MessageContentParts | undefined {
  const normalizedTurnId = params.turnId?.trim();
  const contentItems = params.items.filter(
    (item) =>
      (!normalizedTurnId || item.turn_id === normalizedTurnId) &&
      (item.type === "media" ||
        item.type === "image_generation" ||
        (item.type === "agent_message" &&
          shouldSyncAgentMessageContentPartPhase(item.phase))),
  );
  if (contentItems.length === 0) {
    return params.parts;
  }

  return contentItems.reduce<MessageContentParts>(
    (parts, item) =>
      upsertAgentMessageContentPart({
        item,
        parts,
        threadItems: params.items,
      }),
    params.parts || [],
  );
}

export function syncAssistantAgentMessageContentPartFromThreadItem(params: {
  assistantMsgId: string;
  item: AgentThreadItem;
  threadItems?: readonly AgentThreadItem[];
  setMessages: Dispatch<SetStateAction<Message[]>>;
}): void {
  params.setMessages((prev) => {
    let changed = false;
    const next = prev.map((message) => {
      if (message.id !== params.assistantMsgId) {
        return message;
      }

      const currentParts = message.contentParts || [];
      const nextParts = upsertAgentMessageContentPart({
        item: params.item,
        parts: currentParts,
        threadItems: params.threadItems,
      });
      if (nextParts === currentParts) {
        return message;
      }
      changed = true;
      return {
        ...message,
        contentParts: nextParts,
      };
    });
    return changed ? next : prev;
  });
}
