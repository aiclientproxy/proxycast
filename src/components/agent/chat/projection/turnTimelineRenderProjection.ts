import type { AgentThreadItem, AgentThreadTurn, Message } from "../types";
import type { MessageRenderGroupProjection } from "./messageTimelineRenderProjection";
import { filterConversationThreadItems } from "../utils/threadTimelineView";

type CanonicalMessageItem = Extract<
  AgentThreadItem,
  { type: "user_message" | "agent_message" }
>;

type CanonicalMediaItem = Extract<
  AgentThreadItem,
  { type: "media" | "image_generation" }
>;

export interface CanonicalTurnMessageSegment {
  kind: "message";
  id: string;
  item: CanonicalMessageItem;
}

export interface CanonicalTurnProcessSegment {
  kind: "process";
  id: string;
  items: AgentThreadItem[];
}

export interface CanonicalTurnMediaSegment {
  kind: "media";
  id: string;
  item: CanonicalMediaItem;
}

export type CanonicalTurnRenderSegment =
  | CanonicalTurnMessageSegment
  | CanonicalTurnMediaSegment
  | CanonicalTurnProcessSegment;

export interface CanonicalTurnRenderEntry {
  kind: "canonical_turn";
  id: string;
  turn: AgentThreadTurn;
  segments: CanonicalTurnRenderSegment[];
  isActive: boolean;
}

export interface ResidualMessageRenderEntry {
  kind: "message_group";
  id: string;
  group: MessageRenderGroupProjection;
}

export type TurnTimelineRenderEntry =
  | CanonicalTurnRenderEntry
  | ResidualMessageRenderEntry;

interface CanonicalTurnOwner {
  turn: AgentThreadTurn;
  items: AgentThreadItem[];
  messageItems: CanonicalMessageItem[];
}

export function buildTurnTimelineRenderProjection(params: {
  messageGroups: readonly MessageRenderGroupProjection[];
  renderedTurns: readonly AgentThreadTurn[];
  renderedThreadItems: readonly AgentThreadItem[];
  currentTurnId?: string | null;
}): TurnTimelineRenderEntry[] {
  const owners = buildCanonicalTurnOwners(params);
  const entries: TurnTimelineRenderEntry[] = [];
  const emittedTurnIds = new Set<string>();

  for (const group of params.messageGroups) {
    const hasAnchoredOwner = group.messages.some(
      (message) => resolveMessageOwners(message, owners).length > 0,
    );
    let residualMessages: Message[] = [];
    const flushResidualMessages = () => {
      const residualGroup = buildResidualMessageGroup(
        group,
        residualMessages,
        hasAnchoredOwner,
      );
      residualMessages = [];
      if (!residualGroup) return;
      entries.push({
        kind: "message_group",
        id: `message-group:${residualGroup.id}`,
        group: residualGroup,
      });
    };

    for (const message of group.messages) {
      const messageOwners = resolveMessageOwners(message, owners);
      if (messageOwners.length > 0) {
        flushResidualMessages();
        for (const owner of messageOwners) {
          if (emittedTurnIds.has(owner.turn.id)) continue;
          entries.push(buildCanonicalTurnEntry(owner, params.currentTurnId));
          emittedTurnIds.add(owner.turn.id);
        }
      }
      if (!isMessageOwnedByCanonicalTurn(message, owners)) {
        const residualMessage = buildResidualMessage(message, messageOwners);
        if (residualMessage) residualMessages.push(residualMessage);
      }
    }
    flushResidualMessages();
  }

  const turnOrder = new Map(
    params.renderedTurns.map((turn, index) => [turn.id, index]),
  );
  for (const turn of params.renderedTurns) {
    const owner = owners.get(turn.id);
    if (!owner || emittedTurnIds.has(turn.id)) continue;
    const entry = buildCanonicalTurnEntry(owner, params.currentTurnId);
    const ownerOrder = turnOrder.get(turn.id) ?? Number.MAX_SAFE_INTEGER;
    const insertBeforeIndex = entries.findIndex(
      (candidate) =>
        candidate.kind === "canonical_turn" &&
        (turnOrder.get(candidate.turn.id) ?? Number.MAX_SAFE_INTEGER) >
          ownerOrder,
    );
    if (insertBeforeIndex >= 0) {
      entries.splice(insertBeforeIndex, 0, entry);
    } else {
      entries.push(entry);
    }
    emittedTurnIds.add(turn.id);
  }

  return entries;
}

function buildCanonicalTurnOwners(params: {
  messageGroups: readonly MessageRenderGroupProjection[];
  renderedTurns: readonly AgentThreadTurn[];
  renderedThreadItems: readonly AgentThreadItem[];
}): Map<string, CanonicalTurnOwner> {
  const visibleItems = filterConversationThreadItems(
    params.renderedThreadItems,
  );
  const itemsByTurnId = new Map<string, AgentThreadItem[]>();
  for (const item of visibleItems) {
    const items = itemsByTurnId.get(item.turn_id) ?? [];
    items.push(item);
    itemsByTurnId.set(item.turn_id, items);
  }
  for (const items of itemsByTurnId.values()) {
    items.sort(compareCanonicalItems);
  }

  const messageTurnIds = collectMessageTurnIds(params.messageGroups);
  const owners = new Map<string, CanonicalTurnOwner>();
  for (const turn of params.renderedTurns) {
    const items = itemsByTurnId.get(turn.id) ?? [];
    if (items.length === 0) continue;
    const messageItems = items.filter(isCanonicalMessageItem);
    if (messageItems.length === 0) {
      if (messageTurnIds.has(turn.id)) {
        continue;
      }
    } else if (
      messageTurnIds.has(turn.id) &&
      !hasStableMessageOwnership(turn.id, messageItems, params.messageGroups)
    ) {
      continue;
    }
    owners.set(turn.id, { turn, items, messageItems });
  }
  return owners;
}

function hasStableMessageOwnership(
  turnId: string,
  messageItems: readonly CanonicalMessageItem[],
  groups: readonly MessageRenderGroupProjection[],
): boolean {
  return groups.some((group) =>
    group.messages.some((message) => {
      if (message.runtimeTurnId?.trim() === turnId) return true;
      return messageItems.some((item) => messageMatchesItem(message, item));
    }),
  );
}

function collectMessageTurnIds(
  groups: readonly MessageRenderGroupProjection[],
): Set<string> {
  const ids = new Set<string>();
  for (const group of groups) {
    const timelineTurnId = group.timeline?.turn.id.trim();
    if (timelineTurnId) ids.add(timelineTurnId);
    for (const message of group.messages) {
      const turnId = message.runtimeTurnId?.trim();
      if (turnId) ids.add(turnId);
    }
  }
  return ids;
}

function resolveMessageOwners(
  message: Message,
  owners: ReadonlyMap<string, CanonicalTurnOwner>,
): CanonicalTurnOwner[] {
  const turnIds = new Set<string>();
  const turnId = message.runtimeTurnId?.trim();
  if (turnId) turnIds.add(turnId);
  for (const [ownerTurnId, owner] of owners) {
    if (owner.messageItems.some((item) => messageMatchesItem(message, item))) {
      turnIds.add(ownerTurnId);
    }
  }
  return [...turnIds].flatMap((turnId) => {
    const owner = owners.get(turnId);
    return owner ? [owner] : [];
  });
}

function isMessageOwnedByCanonicalTurn(
  message: Message,
  owners: ReadonlyMap<string, CanonicalTurnOwner>,
): boolean {
  for (const owner of owners.values()) {
    const matchingItem = owner.messageItems.find((item) =>
      messageMatchesItem(message, item),
    );
    if (matchingItem) return !hasMessageOnlySurface(message);
  }

  const turnId = message.runtimeTurnId?.trim();
  const owner = turnId ? owners.get(turnId) : undefined;
  if (!owner || hasMessageOnlySurface(message)) return false;
  return owner.messageItems.some((item) => itemRole(item) === message.role);
}

function messageMatchesItem(
  message: Message,
  item: CanonicalMessageItem,
): boolean {
  if (message.role !== itemRole(item)) return false;
  if (message.id === item.id) return true;
  return item.type === "user_message" && item.client_id === message.id;
}

function hasMessageOnlySurface(message: Message): boolean {
  return Boolean(
    message.imageWorkbenchPreview ||
    message.taskPreview ||
    message.images?.length ||
    message.artifacts?.length ||
    message.inputCapabilityRoute ||
    message.actionRequests?.length ||
    message.contentParts?.some(
      (part) =>
        part.type === "file_changes_batch" || part.type === "media_reference",
    ),
  );
}

function buildResidualMessage(
  message: Message,
  canonicalOwners: readonly CanonicalTurnOwner[],
): Message | null {
  if (canonicalOwners.length === 0 || !hasMessageOnlySurface(message)) {
    return message;
  }

  const canonicalMediaKeys = collectCanonicalMediaKeys(canonicalOwners);
  const richContentParts = message.contentParts?.filter(
    (part) =>
      part.type === "file_changes_batch" ||
      (part.type === "media_reference" &&
        !mediaReferenceKeys(part.reference).some((key) =>
          canonicalMediaKeys.has(key),
        )),
  );
  const residualMessage: Message = {
    ...message,
    content: "",
    images: isCanonicalUserMessageOwner(message, canonicalOwners)
      ? undefined
      : message.images,
    isThinking: undefined,
    thinkingContent: undefined,
    toolCalls: undefined,
    contentParts: richContentParts?.length ? richContentParts : undefined,
    runtimeStatus: undefined,
    inlineProcessRetention: undefined,
  };
  return hasMessageOnlySurface(residualMessage) ? residualMessage : null;
}

function isCanonicalUserMessageOwner(
  message: Message,
  owners: readonly CanonicalTurnOwner[],
): boolean {
  return (
    message.role === "user" &&
    owners.some((owner) =>
      owner.messageItems.some(
        (item) => item.type === "user_message" && messageMatchesItem(message, item),
      ),
    )
  );
}

function collectCanonicalMediaKeys(
  owners: readonly CanonicalTurnOwner[],
): Set<string> {
  const keys = new Set<string>();
  for (const owner of owners) {
    for (const item of owner.items) {
      if (item.type === "media") {
        const uri = item.uri.trim();
        if (uri) keys.add(uri);
      } else if (item.type === "image_generation") {
        const savedPath = item.saved_path?.trim();
        if (savedPath) keys.add(savedPath);
      }
    }
  }
  return keys;
}

function mediaReferenceKeys(reference: {
  uri: string;
  refId?: string;
  sourceUri?: string;
  sourcePath?: string;
}): string[] {
  return [
    reference.refId,
    reference.uri,
    reference.sourcePath,
    reference.sourceUri,
  ].flatMap((value) => {
    const normalized = value?.trim();
    return normalized ? [normalized] : [];
  });
}

function itemRole(item: CanonicalMessageItem): Message["role"] {
  return item.type === "user_message" ? "user" : "assistant";
}

function isCanonicalMessageItem(
  item: AgentThreadItem,
): item is CanonicalMessageItem {
  return item.type === "user_message" || item.type === "agent_message";
}

function isCanonicalMediaItem(
  item: AgentThreadItem,
): item is CanonicalMediaItem {
  return item.type === "media" || item.type === "image_generation";
}

function buildCanonicalTurnEntry(
  owner: CanonicalTurnOwner,
  currentTurnId?: string | null,
): CanonicalTurnRenderEntry {
  return {
    kind: "canonical_turn",
    id: `canonical-turn:${owner.turn.id}`,
    turn: owner.turn,
    segments: buildCanonicalTurnSegments(owner.items),
    isActive: owner.turn.id === currentTurnId,
  };
}

function buildCanonicalTurnSegments(
  items: readonly AgentThreadItem[],
): CanonicalTurnRenderSegment[] {
  const segments: CanonicalTurnRenderSegment[] = [];
  let processItems: AgentThreadItem[] = [];
  const flushProcessItems = () => {
    if (processItems.length === 0) return;
    segments.push({
      kind: "process",
      id: `process:${processItems[0]!.id}:${processItems.at(-1)!.id}`,
      items: processItems,
    });
    processItems = [];
  };

  for (const item of items) {
    if (isCanonicalMessageItem(item)) {
      flushProcessItems();
      segments.push({ kind: "message", id: item.id, item });
    } else if (isCanonicalMediaItem(item)) {
      flushProcessItems();
      segments.push({ kind: "media", id: item.id, item });
    } else {
      processItems.push(item);
    }
  }
  flushProcessItems();
  return segments;
}

function buildResidualMessageGroup(
  group: MessageRenderGroupProjection,
  messages: Message[],
  clearCanonicalTimeline: boolean,
): MessageRenderGroupProjection | null {
  if (messages.length === 0) return null;
  if (!clearCanonicalTimeline && messages.length === group.messages.length) {
    return group;
  }
  const assistantMessages = messages.filter(
    (message) => message.role === "assistant",
  );
  return {
    ...group,
    id: `residual:${group.id}:${messages[0]!.id}`,
    messages,
    userMessage: messages.find((message) => message.role === "user") ?? null,
    assistantMessages,
    startedAt: messages.reduce(
      (earliest, message) =>
        message.timestamp < earliest ? message.timestamp : earliest,
      messages[0]!.timestamp,
    ),
    endedAt: messages.reduce(
      (latest, message) =>
        message.timestamp > latest ? message.timestamp : latest,
      messages[0]!.timestamp,
    ),
    lastAssistantId: assistantMessages.at(-1)?.id ?? null,
    timelineMessageId: null,
    timeline: null,
  };
}

function compareCanonicalItems(
  left: AgentThreadItem,
  right: AgentThreadItem,
): number {
  return left.sequence - right.sequence || left.id.localeCompare(right.id);
}
