export type { HydrateSessionDetailMessagesOptions } from "./agentChatHistoryTypes";
export {
  appendTextToParts,
  appendTextWithOverlapDetection,
  appendThinkingToHistoryParts,
  extractThinkingContentFromParts,
  normalizeHistoryPartType,
} from "./agentChatHistoryPrimitives";
export {
  compactHistoricalRestoreMessages,
  hasLegacyFallbackToolNames,
  normalizeHistoricalTopicSnapshotMessage,
  normalizeHistoricalTopicSnapshotMessages,
  normalizeHistoryMessage,
  normalizeHistoryMessages,
  resolveHistoryToolName,
  shouldCompactCompletedSessionHistory,
} from "./agentChatHistoryNormalize";
export { mergeAdjacentAssistantMessages } from "./agentChatHistoryAdjacentMerge";
export { mergeHydratedMessagesWithLocalState } from "./agentChatHistoryLocalMerge";
export { dedupeAdjacentHistoryMessages } from "./agentChatHistorySignatures";
export { hydrateSessionDetailMessages } from "./agentChatHistoryHydrate";
