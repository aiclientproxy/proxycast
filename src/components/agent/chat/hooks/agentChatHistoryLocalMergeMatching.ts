import type { Message } from "../types";
import { normalizeSignatureText } from "./agentChatHistoryPrimitives";
import {
  buildAssistantHydrationSignature,
  buildHistoryMessageSignature,
  messageImageSignature,
} from "./agentChatHistorySignatures";
import { messageHasInterruptedPlaceholder } from "./agentInterruptedMessageContent";
import {
  hasRetainableLocalAssistantProcessState,
  hasRetainableLocalMessageState,
} from "./agentChatHistoryLocalMergeState";

function hasLocalInterruptedStopMarker(message: Message): boolean {
  return messageHasInterruptedPlaceholder(message);
}

function isPendingRuntimeTurnId(value?: string | null): boolean {
  return Boolean(value?.trim().startsWith("pending-turn:"));
}

function readCanonicalRuntimeTurnId(message: Message): string | null {
  const runtimeTurnId = message.runtimeTurnId?.trim();
  if (!runtimeTurnId || isPendingRuntimeTurnId(runtimeTurnId)) {
    return null;
  }

  return runtimeTurnId;
}

function canFallbackMatchCanonicalTurn(
  candidate: Message,
  targetCanonicalTurnId: string | null,
): boolean {
  if (!targetCanonicalTurnId) {
    return true;
  }

  const candidateCanonicalTurnId = readCanonicalRuntimeTurnId(candidate);
  return (
    !candidateCanonicalTurnId ||
    candidateCanonicalTurnId === targetCanonicalTurnId
  );
}

function findCanonicalTurnMatchIndex(
  messages: Message[],
  targetCanonicalTurnId: string | null,
  startIndex: number,
): number {
  if (!targetCanonicalTurnId) {
    return -1;
  }

  return messages.findIndex(
    (candidate, index) =>
      index >= startIndex &&
      readCanonicalRuntimeTurnId(candidate) === targetCanonicalTurnId,
  );
}

function canMatchLocalProcessAssistantToTarget(
  candidate: Message,
  targetRuntimeTurnId?: string,
): boolean {
  if (!targetRuntimeTurnId) {
    return true;
  }

  const candidateRuntimeTurnId = candidate.runtimeTurnId?.trim();
  if (candidateRuntimeTurnId === targetRuntimeTurnId) {
    return true;
  }

  return (
    hasLocalInterruptedStopMarker(candidate) &&
    (!candidateRuntimeTurnId || isPendingRuntimeTurnId(candidateRuntimeTurnId))
  );
}

export const findMatchingLocalUserMessageIndex = (
  localUserMessages: Message[],
  targetMessage: Message,
  startIndex: number,
): number => {
  const targetContent = normalizeSignatureText(targetMessage.content || "");
  const targetCanonicalTurnId = readCanonicalRuntimeTurnId(targetMessage);
  const hasLooseContentMatch = (candidateContent: string): boolean => {
    const minLength = Math.min(candidateContent.length, targetContent.length);
    return (
      minLength >= 24 &&
      (candidateContent.includes(targetContent) ||
        targetContent.includes(candidateContent))
    );
  };

  const canonicalTurnMatchIndex = findCanonicalTurnMatchIndex(
    localUserMessages,
    targetCanonicalTurnId,
    startIndex,
  );
  if (canonicalTurnMatchIndex >= 0) {
    return canonicalTurnMatchIndex;
  }

  for (let index = startIndex; index < localUserMessages.length; index += 1) {
    const candidate = localUserMessages[index];
    if (
      !candidate ||
      !canFallbackMatchCanonicalTurn(candidate, targetCanonicalTurnId)
    ) {
      continue;
    }

    const candidateContent = normalizeSignatureText(candidate.content || "");
    if (
      candidateContent === targetContent ||
      hasLooseContentMatch(candidateContent)
    ) {
      return index;
    }
  }

  return -1;
};

export const findMatchingLocalAssistantMessageIndex = (
  localAssistantMessages: Message[],
  targetMessage: Message,
  startIndex: number,
): number => {
  const targetCanonicalTurnId = readCanonicalRuntimeTurnId(targetMessage);
  const canonicalTurnMatchIndex = findCanonicalTurnMatchIndex(
    localAssistantMessages,
    targetCanonicalTurnId,
    startIndex,
  );
  if (canonicalTurnMatchIndex >= 0) {
    return canonicalTurnMatchIndex;
  }

  const targetSignature = buildHistoryMessageSignature({
    ...targetMessage,
    usage: undefined,
  });

  for (
    let index = startIndex;
    index < localAssistantMessages.length;
    index += 1
  ) {
    const candidate = localAssistantMessages[index];
    if (
      !candidate ||
      !canFallbackMatchCanonicalTurn(candidate, targetCanonicalTurnId)
    ) {
      continue;
    }

    const candidateSignature = buildHistoryMessageSignature({
      ...candidate,
      usage: undefined,
    });
    if (candidateSignature === targetSignature) {
      return index;
    }
  }

  const fallbackSignature = buildAssistantHydrationSignature(targetMessage);
  if (!fallbackSignature) {
    return -1;
  }

  for (
    let index = startIndex;
    index < localAssistantMessages.length;
    index += 1
  ) {
    const candidate = localAssistantMessages[index];
    if (
      !candidate ||
      !canFallbackMatchCanonicalTurn(candidate, targetCanonicalTurnId)
    ) {
      continue;
    }

    const candidateSignature = buildAssistantHydrationSignature(candidate);
    if (candidateSignature === fallbackSignature) {
      return index;
    }
  }

  return -1;
};

export function findNextLocalAssistantAfterUser(
  localMessages: Message[],
  localAssistantIndexById: Map<string, number>,
  matchedLocalMessageIds: Set<string>,
  userMessageIndex: number | null,
  targetMessage: Message,
): number {
  if (userMessageIndex === null || userMessageIndex < 0) {
    return -1;
  }
  const targetCanonicalTurnId = readCanonicalRuntimeTurnId(targetMessage);

  for (
    let index = userMessageIndex + 1;
    index < localMessages.length;
    index += 1
  ) {
    const candidate = localMessages[index];
    if (!candidate) {
      continue;
    }
    if (candidate.role === "user") {
      return -1;
    }
    if (candidate.role !== "assistant") {
      continue;
    }
    if (candidate.id && matchedLocalMessageIds.has(candidate.id)) {
      continue;
    }
    if (!canFallbackMatchCanonicalTurn(candidate, targetCanonicalTurnId)) {
      return -1;
    }

    return candidate.id
      ? (localAssistantIndexById.get(candidate.id) ?? -1)
      : -1;
  }

  return -1;
}

export function findNextLocalProcessAssistantIndex(
  localAssistantMessages: Message[],
  startIndex: number,
  matchedLocalMessageIds: Set<string>,
  targetMessage?: Message,
): number {
  const targetRuntimeTurnId = targetMessage?.runtimeTurnId?.trim();
  for (
    let index = Math.max(0, startIndex);
    index < localAssistantMessages.length;
    index += 1
  ) {
    const candidate = localAssistantMessages[index];
    if (!candidate) {
      continue;
    }
    if (candidate.id && matchedLocalMessageIds.has(candidate.id)) {
      continue;
    }
    if (
      !canMatchLocalProcessAssistantToTarget(candidate, targetRuntimeTurnId)
    ) {
      continue;
    }
    if (
      hasRetainableLocalAssistantProcessState(candidate) ||
      hasLocalInterruptedStopMarker(candidate)
    ) {
      return index;
    }
  }

  return -1;
}

export function findNextLocalInterruptedAssistantIndex(
  localAssistantMessages: Message[],
  startIndex: number,
  matchedLocalMessageIds: Set<string>,
  targetMessage: Message,
): number {
  const targetCanonicalTurnId = readCanonicalRuntimeTurnId(targetMessage);
  for (
    let index = Math.max(0, startIndex);
    index < localAssistantMessages.length;
    index += 1
  ) {
    const candidate = localAssistantMessages[index];
    if (!candidate) {
      continue;
    }
    if (candidate.id && matchedLocalMessageIds.has(candidate.id)) {
      continue;
    }
    if (!canFallbackMatchCanonicalTurn(candidate, targetCanonicalTurnId)) {
      continue;
    }
    if (messageHasInterruptedPlaceholder(candidate)) {
      return index;
    }
  }

  return -1;
}

export function isLocalAssistantInMatchedUserTurn(params: {
  localAssistantMessage?: Message;
  localMessageIndexById: Map<string, number>;
  localMessages: Message[];
  lastMatchedLocalUserMessageIndex: number | null;
}): boolean {
  const { localAssistantMessage, lastMatchedLocalUserMessageIndex } = params;
  if (!localAssistantMessage?.id || lastMatchedLocalUserMessageIndex === null) {
    return false;
  }

  const assistantIndex = params.localMessageIndexById.get(
    localAssistantMessage.id,
  );
  if (
    assistantIndex === undefined ||
    assistantIndex <= lastMatchedLocalUserMessageIndex
  ) {
    return false;
  }

  return !params.localMessages
    .slice(lastMatchedLocalUserMessageIndex + 1, assistantIndex)
    .some((message) => message.role === "user");
}

function findPreviousLocalUserInSameTurn(params: {
  localAssistantMessage?: Message;
  localMessageIndexById: Map<string, number>;
  localMessages: Message[];
}): Message | null {
  const { localAssistantMessage } = params;
  if (!localAssistantMessage?.id) {
    return null;
  }

  const assistantIndex = params.localMessageIndexById.get(
    localAssistantMessage.id,
  );
  if (assistantIndex === undefined || assistantIndex <= 0) {
    return null;
  }

  for (let index = assistantIndex - 1; index >= 0; index -= 1) {
    const candidate = params.localMessages[index];
    if (!candidate) {
      continue;
    }

    if (candidate.role === "user") {
      return candidate;
    }
  }

  return null;
}

export function hasRecoverableLocalUserBeforeAssistant(params: {
  localAssistantMessage?: Message;
  localMessageIndexById: Map<string, number>;
  localMessages: Message[];
  matchedLocalMessageIds: Set<string>;
}): boolean {
  const userMessage = findPreviousLocalUserInSameTurn(params);
  return Boolean(
    userMessage &&
    (!userMessage.id || !params.matchedLocalMessageIds.has(userMessage.id)) &&
    hasRetainableLocalMessageState(userMessage),
  );
}

export function insertRecoverableLocalUsersForMatchedAssistantTurns(params: {
  mergedMessages: Message[];
  localMessageIndexById: Map<string, number>;
  localMessages: Message[];
  matchedLocalMessageIds: Set<string>;
}): Message[] {
  const existingUserSignatures = new Set(
    params.mergedMessages
      .filter((message) => message.role === "user")
      .map((message) =>
        [
          normalizeSignatureText(message.content || ""),
          messageImageSignature(message.images),
        ].join("::"),
      ),
  );
  let inserted = false;
  const nextMessages: Message[] = [];

  for (const message of params.mergedMessages) {
    if (message.role !== "assistant") {
      nextMessages.push(message);
      continue;
    }

    const hasMergedUserInCurrentTurn = (() => {
      for (let index = nextMessages.length - 1; index >= 0; index -= 1) {
        const candidate = nextMessages[index];
        if (candidate?.role === "user") {
          return true;
        }
        if (candidate?.role === "assistant") {
          return false;
        }
      }
      return false;
    })();
    if (hasMergedUserInCurrentTurn) {
      nextMessages.push(message);
      continue;
    }

    const localAssistantMessage =
      message.id && params.localMessageIndexById.has(message.id)
        ? params.localMessages[params.localMessageIndexById.get(message.id)!]
        : undefined;
    const userMessage = findPreviousLocalUserInSameTurn({
      localAssistantMessage,
      localMessageIndexById: params.localMessageIndexById,
      localMessages: params.localMessages,
    });
    if (userMessage && hasRetainableLocalMessageState(userMessage)) {
      const signature = [
        normalizeSignatureText(userMessage.content || ""),
        messageImageSignature(userMessage.images),
      ].join("::");
      const isAlreadyMatched = Boolean(
        userMessage.id && params.matchedLocalMessageIds.has(userMessage.id),
      );
      if (!isAlreadyMatched && !existingUserSignatures.has(signature)) {
        nextMessages.push(userMessage);
        existingUserSignatures.add(signature);
        if (userMessage.id) {
          params.matchedLocalMessageIds.add(userMessage.id);
        }
        inserted = true;
      }
    }

    nextMessages.push(message);
  }

  return inserted ? nextMessages : params.mergedMessages;
}
