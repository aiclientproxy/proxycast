import type { AgentThreadItem } from "@/lib/api/agentProtocol";
import type { Message } from "../types";

export const INTERRUPTED_PLACEHOLDER_TEXT = "(已停止)";

export function stripInterruptedPlaceholderText(
  value: string | null | undefined,
): string {
  const trimmed = typeof value === "string" ? value.trimEnd() : "";
  return trimmed.endsWith(INTERRUPTED_PLACEHOLDER_TEXT)
    ? trimmed.slice(0, -INTERRUPTED_PLACEHOLDER_TEXT.length).trimEnd()
    : trimmed;
}

export function contentHasInterruptedPlaceholder(
  content?: string | null,
): boolean {
  return Boolean(content?.includes(INTERRUPTED_PLACEHOLDER_TEXT));
}

export function contentPartsHaveInterruptedPlaceholder(
  parts?: Message["contentParts"],
): boolean {
  return Boolean(
    parts?.some(
      (part) =>
        part.type === "text" && contentHasInterruptedPlaceholder(part.text),
    ),
  );
}

export function messageHasInterruptedPlaceholder(
  message?: Pick<Message, "content" | "contentParts"> | null,
): boolean {
  return (
    contentHasInterruptedPlaceholder(message?.content) ||
    contentPartsHaveInterruptedPlaceholder(message?.contentParts)
  );
}

export function appendInterruptedPlaceholderText(content: string): string {
  if (contentHasInterruptedPlaceholder(content)) {
    return content;
  }

  const visibleContent = content.trimEnd();
  if (!visibleContent) {
    return INTERRUPTED_PLACEHOLDER_TEXT;
  }

  return `${visibleContent}\n\n${INTERRUPTED_PLACEHOLDER_TEXT}`;
}

function appendInterruptedPlaceholderContentPart(
  parts: Message["contentParts"],
): Message["contentParts"] {
  if (contentPartsHaveInterruptedPlaceholder(parts)) {
    return parts;
  }

  return [
    ...(parts || []),
    { type: "text", text: INTERRUPTED_PLACEHOLDER_TEXT },
  ];
}

export function buildInterruptedMessageContentPatch(
  message: Message,
): Pick<Message, "content" | "contentParts"> {
  return {
    content: appendInterruptedPlaceholderText(message.content),
    contentParts: appendInterruptedPlaceholderContentPart(message.contentParts),
  };
}

function appendInterruptedPlaceholderThreadContentParts(
  parts: Extract<AgentThreadItem, { type: "agent_message" }>["contentParts"],
  visibleText: string,
): Extract<AgentThreadItem, { type: "agent_message" }>["contentParts"] {
  const existingParts = parts || [];
  const visibleContent = stripInterruptedPlaceholderText(visibleText);
  const hasVisibleTextPart = existingParts.some(
    (part) =>
      part.type === "text" &&
      stripInterruptedPlaceholderText(part.text).trim().length > 0,
  );
  const normalizedParts =
    visibleContent.trim() && !hasVisibleTextPart
      ? [{ type: "text" as const, text: visibleContent }, ...existingParts]
      : existingParts;

  if (
    normalizedParts.some(
      (part) =>
        part.type === "text" && contentHasInterruptedPlaceholder(part.text),
    )
  ) {
    return normalizedParts;
  }

  return [
    ...normalizedParts,
    { type: "text", text: INTERRUPTED_PLACEHOLDER_TEXT },
  ];
}

export function appendInterruptedPlaceholderToThreadItem(
  item: AgentThreadItem,
): AgentThreadItem {
  if (item.type !== "agent_message") {
    return item;
  }

  return {
    ...item,
    text: appendInterruptedPlaceholderText(item.text),
    contentParts: appendInterruptedPlaceholderThreadContentParts(
      item.contentParts,
      item.text,
    ),
  };
}

export function markInterruptedAgentMessageThreadItems(
  items: AgentThreadItem[],
  interruptedTurnIds: ReadonlySet<string>,
): AgentThreadItem[] {
  if (interruptedTurnIds.size === 0) {
    return items;
  }

  return items.map((item) =>
    item.type === "agent_message" &&
    item.turn_id &&
    interruptedTurnIds.has(item.turn_id)
      ? appendInterruptedPlaceholderToThreadItem(item)
      : item,
  );
}
