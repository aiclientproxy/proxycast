import type { AgentThreadItem, AgentThreadTurn, Message } from "../types";
import {
  buildMessageTurnTimeline,
  filterConversationThreadItems,
  sortThreadItems,
  type MessageTurnTimeline,
} from "../utils/threadTimelineView";
import {
  buildMessageTurnGroups,
  type MessageTurnGroup,
} from "../utils/messageTurnGrouping";

export interface CurrentTurnTimelineProjection {
  messageId: string;
  turn: AgentThreadTurn;
  items: AgentThreadItem[];
}

export interface MessageRenderGroupProjection extends MessageTurnGroup {
  lastAssistantId: string | null;
  timelineMessageId: string | null;
  timeline: MessageTurnTimeline | CurrentTurnTimelineProjection | null;
  isActiveGroup: boolean;
  isImportedHistory?: boolean;
}

export function buildTimelineByMessageIdProjection(params: {
  canBuildHistoricalTimeline: boolean;
  renderedMessages: Message[];
  renderedTurns: AgentThreadTurn[];
  renderedThreadItems: AgentThreadItem[];
}): Map<string, MessageTurnTimeline> {
  if (!params.canBuildHistoricalTimeline) {
    return new Map<string, MessageTurnTimeline>();
  }

  const timelineByMessageId = buildMessageTurnTimeline(
    params.renderedMessages,
    params.renderedTurns,
    params.renderedThreadItems,
  );

  return attachRuntimeTurnItemFallbackTimelines({
    renderedMessages: params.renderedMessages,
    renderedTurns: params.renderedTurns,
    renderedThreadItems: params.renderedThreadItems,
    timelineByMessageId,
  });
}

export function buildDeferredTimelineByMessageIdProjection(params: {
  renderedMessages: Message[];
  renderedTurns: AgentThreadTurn[];
}): Map<string, MessageTurnTimeline> {
  return buildMessageTurnTimeline(
    params.renderedMessages,
    params.renderedTurns,
    [],
  );
}

function normalizeRuntimeTurnId(message: Message): string | null {
  const normalized = message.runtimeTurnId?.trim();
  return normalized || null;
}

function hasConversationProcessItem(items: AgentThreadItem[]): boolean {
  return items.some(
    (item) => item.type !== "user_message" && item.type !== "agent_message",
  );
}

function collectMessageToolIds(message: Message): Set<string> {
  const ids = new Set<string>();
  for (const toolCall of message.toolCalls || []) {
    if (toolCall.id?.trim()) {
      ids.add(toolCall.id.trim());
    }
  }
  for (const part of message.contentParts || []) {
    if (part.type === "tool_use" && part.toolCall.id?.trim()) {
      ids.add(part.toolCall.id.trim());
    }
  }
  return ids;
}

function collectThreadItemToolIds(items: AgentThreadItem[]): Set<string> {
  const ids = new Set<string>();
  for (const item of items) {
    if (
      item.type === "tool_call" ||
      item.type === "command_execution" ||
      item.type === "patch" ||
      item.type === "web_search"
    ) {
      ids.add(item.id);
    }
  }
  return ids;
}

function hasSharedToolId(
  left: ReadonlySet<string>,
  right: ReadonlySet<string>,
): boolean {
  for (const value of left) {
    if (right.has(value)) {
      return true;
    }
  }
  return false;
}

function resolveFallbackTurnStatus(
  items: AgentThreadItem[],
): AgentThreadTurn["status"] {
  if (items.some((item) => item.status === "failed")) {
    return "failed";
  }
  if (items.some((item) => item.status === "in_progress")) {
    return "running";
  }
  return "completed";
}

function buildFallbackTurnFromItems(
  turnId: string,
  items: AgentThreadItem[],
  message: Message,
): AgentThreadTurn {
  const firstItem = items[0];
  const lastItem = items[items.length - 1] || firstItem;
  const messageTimestamp =
    message.timestamp instanceof Date
      ? message.timestamp.toISOString()
      : new Date(message.timestamp).toISOString();
  const startedAt = firstItem?.started_at || messageTimestamp;
  const updatedAt = lastItem?.updated_at || lastItem?.completed_at || startedAt;
  const status = resolveFallbackTurnStatus(items);

  return {
    id: turnId,
    thread_id: firstItem?.thread_id || "",
    prompt_text: "",
    status,
    started_at: startedAt,
    ...(status !== "running"
      ? { completed_at: lastItem?.completed_at || updatedAt }
      : {}),
    created_at: startedAt,
    updated_at: updatedAt,
  };
}

function attachRuntimeTurnItemFallbackTimelines(params: {
  renderedMessages: Message[];
  renderedTurns: AgentThreadTurn[];
  renderedThreadItems: AgentThreadItem[];
  timelineByMessageId: Map<string, MessageTurnTimeline>;
}): Map<string, MessageTurnTimeline> {
  if (params.renderedThreadItems.length === 0) {
    return params.timelineByMessageId;
  }

  const mappedTurnIds = new Set(
    [...params.timelineByMessageId.values()].map((entry) => entry.turn.id),
  );
  const turnById = new Map(params.renderedTurns.map((turn) => [turn.id, turn]));
  const itemsByTurnId = new Map<string, AgentThreadItem[]>();
  for (const item of sortThreadItems(
    filterConversationThreadItems(params.renderedThreadItems),
  )) {
    const existing = itemsByTurnId.get(item.turn_id);
    if (existing) {
      existing.push(item);
    } else {
      itemsByTurnId.set(item.turn_id, [item]);
    }
  }

  let nextTimelineByMessageId = params.timelineByMessageId;
  const candidateProcessTurns = [...itemsByTurnId.entries()]
    .filter(([turnId, items]) => {
      if (mappedTurnIds.has(turnId) || !hasConversationProcessItem(items)) {
        return false;
      }
      return collectThreadItemToolIds(items).size > 0;
    })
    .map(([turnId, items]) => ({
      turnId,
      items,
      toolIds: collectThreadItemToolIds(items),
    }));

  for (const message of params.renderedMessages) {
    if (
      message.role !== "assistant" ||
      nextTimelineByMessageId.has(message.id)
    ) {
      continue;
    }

    const messageToolIds = collectMessageToolIds(message);
    if (messageToolIds.size === 0) {
      continue;
    }

    const matchingTurns = candidateProcessTurns.filter(
      (entry) =>
        !mappedTurnIds.has(entry.turnId) &&
        hasSharedToolId(messageToolIds, entry.toolIds),
    );
    if (matchingTurns.length !== 1) {
      continue;
    }

    const [{ turnId, items }] = matchingTurns;
    if (nextTimelineByMessageId === params.timelineByMessageId) {
      nextTimelineByMessageId = new Map(params.timelineByMessageId);
    }
    nextTimelineByMessageId.set(message.id, {
      messageId: message.id,
      turn:
        turnById.get(turnId) ||
        buildFallbackTurnFromItems(turnId, items, message),
      items,
    });
    mappedTurnIds.add(turnId);
  }

  for (const message of params.renderedMessages) {
    if (
      message.role !== "assistant" ||
      nextTimelineByMessageId.has(message.id)
    ) {
      continue;
    }

    const runtimeTurnId = normalizeRuntimeTurnId(message);
    if (!runtimeTurnId || mappedTurnIds.has(runtimeTurnId)) {
      continue;
    }

    const items = itemsByTurnId.get(runtimeTurnId);
    if (!items?.length || !hasConversationProcessItem(items)) {
      continue;
    }

    if (nextTimelineByMessageId === params.timelineByMessageId) {
      nextTimelineByMessageId = new Map(params.timelineByMessageId);
    }
    nextTimelineByMessageId.set(message.id, {
      messageId: message.id,
      turn:
        turnById.get(runtimeTurnId) ||
        buildFallbackTurnFromItems(runtimeTurnId, items, message),
      items,
    });
    mappedTurnIds.add(runtimeTurnId);
  }

  return nextTimelineByMessageId;
}

export function resolveLastAssistantMessage(
  renderedMessages: readonly Message[],
): Message | null {
  for (let index = renderedMessages.length - 1; index >= 0; index -= 1) {
    const message = renderedMessages[index];
    if (message?.role === "assistant") {
      return message;
    }
  }
  return null;
}

export function buildCurrentTurnTimelineProjection(params: {
  activeCurrentTurnId: string | null;
  activeCurrentTurn: AgentThreadTurn | null;
  lastAssistantMessageId: string | null;
  timelineByMessageId: Map<string, MessageTurnTimeline>;
  renderedThreadItems: AgentThreadItem[];
  renderedMessages?: Message[];
}): CurrentTurnTimelineProjection | null {
  if (
    !params.activeCurrentTurnId ||
    !params.activeCurrentTurn ||
    !params.lastAssistantMessageId
  ) {
    return null;
  }

  let mappedMessageId: string | null = null;
  for (const entry of params.timelineByMessageId.values()) {
    if (entry.turn.id === params.activeCurrentTurnId) {
      mappedMessageId = entry.messageId;
      break;
    }
  }

  if (!mappedMessageId && params.renderedMessages) {
    const explicitAssistant = params.renderedMessages.find(
      (message) =>
        message.role === "assistant" &&
        message.runtimeTurnId?.trim() === params.activeCurrentTurnId,
    );
    mappedMessageId = explicitAssistant?.id ?? null;
  }

  if (!mappedMessageId && params.renderedMessages) {
    const lastAssistant = params.renderedMessages.find(
      (message) => message.id === params.lastAssistantMessageId,
    );
    const lastAssistantTurnId = lastAssistant?.runtimeTurnId?.trim();
    if (
      lastAssistantTurnId &&
      lastAssistantTurnId !== params.activeCurrentTurnId
    ) {
      return null;
    }
  }

  return {
    messageId: mappedMessageId || params.lastAssistantMessageId,
    turn: params.activeCurrentTurn,
    items: filterConversationThreadItems(
      params.renderedThreadItems.filter(
        (item) => item.turn_id === params.activeCurrentTurnId,
      ),
    ),
  };
}

export function buildMessageRenderGroupsProjection(params: {
  messageGroups: MessageTurnGroup[];
  timelineByMessageId: Map<string, MessageTurnTimeline>;
  currentTurnTimeline: CurrentTurnTimelineProjection | null;
  lastAssistantMessageId: string | null;
}): MessageRenderGroupProjection[] {
  const renderedTimelineTurnIds = new Set<string>();

  return params.messageGroups.map((group) => {
    const lastAssistantId =
      group.assistantMessages[group.assistantMessages.length - 1]?.id ?? null;
    let mappedTimeline: MessageTurnTimeline | null = null;
    for (const message of group.assistantMessages) {
      mappedTimeline = params.timelineByMessageId.get(message.id) ?? null;
      if (mappedTimeline) {
        break;
      }
    }
    const currentTurnMessageId = params.currentTurnTimeline?.messageId ?? null;
    const isCurrentTurnGroup = Boolean(
      currentTurnMessageId &&
        group.assistantMessages.some(
          (message) => message.id === currentTurnMessageId,
        ),
    );
    const isActiveGroup =
      Boolean(lastAssistantId) &&
      lastAssistantId === params.lastAssistantMessageId;
    const mappedGroupTimeline =
      mappedTimeline ||
      (isCurrentTurnGroup ? params.currentTurnTimeline : null);
    const timelineTurnId = mappedGroupTimeline?.turn.id ?? null;
    const timeline =
      mappedGroupTimeline &&
      (!timelineTurnId || !renderedTimelineTurnIds.has(timelineTurnId))
        ? mappedGroupTimeline
        : null;
    if (timelineTurnId && timeline) {
      renderedTimelineTurnIds.add(timelineTurnId);
    }

    return {
      ...group,
      lastAssistantId,
      timelineMessageId: timeline?.messageId ?? null,
      timeline,
      isActiveGroup,
    };
  });
}

export function buildMessageGroupsProjection(
  renderedMessages: Message[],
): MessageTurnGroup[] {
  return buildMessageTurnGroups(renderedMessages);
}
