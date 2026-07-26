import {
  LIVE_TAIL_COMMIT_PROMPT,
  MCP_STRUCTURED_CONTENT_PROMPT,
  REASONING_FIRST_VISIBLE_PROMPT,
} from "./claw-chat-current-fixture-constants.mjs";
import {
  MEDIA_REFERENCE_MIME_TYPE,
  MEDIA_REFERENCE_PROMPT,
  MEDIA_REFERENCE_SUMMARY_TEXT,
  MEDIA_REFERENCE_TITLE,
  MEDIA_REFERENCE_URI,
} from "./claw-chat-current-fixture-media-reference.mjs";

export function buildReasoningFirstVisibleScenarioAssertions({
  reasoningFirstVisibleTurnStart,
  summary,
}) {
  return {
    reasoningFirstVisiblePromptReachedBackend:
      reasoningFirstVisibleTurnStart?.inputText ===
      REASONING_FIRST_VISIBLE_PROMPT,
    guiReasoningFirstVisibleInputSubmitted:
      summary.reasoningFirstVisibleInputSend?.afterFill
        ?.promptVisibleInTextarea === true &&
      summary.reasoningFirstVisibleInputSend?.clicked?.clicked === true,
    guiReasoningFirstVisibleBeforeAnswer:
      summary.guiReasoningFirstVisibleBeforeAnswer
        ?.reasoningFirstVisibleBeforeAnswerCaptured === true &&
      summary.guiReasoningFirstVisibleBeforeAnswer?.hasPrompt === true &&
      summary.guiReasoningFirstVisibleBeforeAnswer?.hasReasoningText === true &&
      summary.guiReasoningFirstVisibleBeforeAnswer?.hasReasoningProcess ===
        true &&
      summary.guiReasoningFirstVisibleBeforeAnswer
        ?.hasReasoningBeforeFinalAnswer === true &&
      summary.guiReasoningFirstVisibleBeforeAnswer?.hasFinalText === false &&
      summary.guiReasoningFirstVisibleBeforeAnswer?.startupNoteVisible ===
        false,
    guiReasoningFirstVisibleCompleted:
      summary.guiReasoningFirstVisibleCompleted?.hasPrompt === true &&
      summary.guiReasoningFirstVisibleCompleted?.hasReasoningText === true &&
      summary.guiReasoningFirstVisibleCompleted?.hasFinalText === true &&
      summary.guiReasoningFirstVisibleCompleted
        ?.hasReasoningBeforeFinalAnswer === true &&
      summary.guiReasoningFirstVisibleCompleted?.startupNoteVisible === false &&
      summary.guiReasoningFirstVisibleCompleted?.textareaDisabled === false &&
      summary.guiReasoningFirstVisibleCompleted?.stopButtonVisible === false,
    readModelReasoningFirstVisibleCompleted:
      summary.readModelReasoningFirstVisibleCompleted?.includesPrompt ===
        true &&
      summary.readModelReasoningFirstVisibleCompleted?.latestTurnStatus ===
        "completed" &&
      summary.readModelReasoningFirstVisibleCompleted?.includesFinalText ===
        true &&
      summary.readModelReasoningFirstVisibleCompleted?.includesReasoningText ===
        true,
    readModelReasoningFirstVisibleItemObserved:
      summary.readModelReasoningFirstVisibleCompleted?.includesReasoningItem ===
        true &&
      summary.readModelReasoningFirstVisibleCompleted?.reasoningItemCount >=
        1 &&
      summary.readModelReasoningFirstVisibleCompleted
        ?.reasoningSequenceBeforeFinal === true,
  };
}

export function buildLiveTailCommitScenarioAssertions({
  liveTailCommitTurnStart,
  summary,
}) {
  return {
    liveTailCommitPromptReachedBackend:
      liveTailCommitTurnStart?.inputText === LIVE_TAIL_COMMIT_PROMPT,
    guiLiveTailCommitInputSubmitted:
      summary.liveTailCommitInputSend?.afterFill?.promptVisibleInTextarea ===
        true && summary.liveTailCommitInputSend?.clicked?.clicked === true,
    guiLiveTailFirstVisibleBeforeCommit:
      summary.guiLiveTailFirstVisibleBeforeCommit?.hasPrompt === true &&
      summary.guiLiveTailFirstVisibleBeforeCommit?.hasFirstText === true &&
      summary.guiLiveTailFirstVisibleBeforeCommit?.firstTextOccurrenceCount ===
        1 &&
      summary.guiLiveTailFirstVisibleBeforeCommit?.hasDoneText === false &&
      summary.guiLiveTailFirstVisibleBeforeCommit?.hasOverflowMarker ===
        false &&
      summary.guiLiveTailFirstVisibleBeforeCommit?.hasTableTail === false,
    guiLiveTailRunningStatusPreserved:
      summary.guiLiveTailFirstVisibleBeforeCommit?.runningStatusVisible ===
        true &&
      summary.guiLiveTailFirstVisibleBeforeCommit?.stopButtonVisible === true,
    guiLiveTailNoStartupNote:
      summary.guiLiveTailFirstVisibleBeforeCommit?.startupNoteVisible ===
        false && summary.guiLiveTailVisualOracle?.startupNoteVisible === false,
    guiLiveTailOverflowCommitted:
      summary.guiLiveTailVisualOracle?.hasOverflowMarker === true &&
      summary.guiLiveTailVisualOracle?.overflowCommitted === true &&
      summary.guiLiveTailVisualOracle?.firstTextBeforeOverflow === true,
    guiLiveTailTableTailVisible:
      summary.guiLiveTailVisualOracle?.hasTableHeader === true &&
      summary.guiLiveTailVisualOracle?.hasTableTail === true &&
      summary.guiLiveTailVisualOracle?.markdownTableRendered === true &&
      summary.guiLiveTailVisualOracle?.firstTextBeforeTableTail === true,
    guiLiveTailCanonicalRepairExactOnce:
      summary.guiLiveTailVisualOracle?.expectedItemIdentityVisible === true &&
      summary.guiLiveTailVisualOracle?.canonicalMarkersExactOnce === true &&
      summary.guiLiveTailVisualOracle?.firstTextOccurrenceCount === 1 &&
      summary.guiLiveTailVisualOracle?.overflowMarkerOccurrenceCount === 1 &&
      summary.guiLiveTailVisualOracle?.tableHeaderOccurrenceCount === 1 &&
      summary.guiLiveTailVisualOracle?.tableTailOccurrenceCount === 1 &&
      summary.guiLiveTailVisualOracle?.doneTextOccurrenceCount === 1,
    guiLiveTailScrollAnchorStable:
      summary.guiLiveTailVisualOracle?.scrollAnchorStable === true,
    guiLiveTailCompleted:
      summary.guiLiveTailCompleted?.hasPrompt === true &&
      (summary.guiLiveTailCompleted?.hasAssistantSummary === true ||
        summary.guiLiveTailCompleted?.hasDoneText === true) &&
      summary.guiLiveTailCompleted?.textareaVisible === true &&
      summary.guiLiveTailCompleted?.textareaDisabled === false &&
      summary.guiLiveTailCompleted?.stopButtonVisible === false,
    readModelLiveTailCommitCompleted:
      summary.readModelLiveTailCommitCompleted?.includesPrompt === true &&
      summary.readModelLiveTailCommitCompleted?.latestTurnStatus ===
        "completed" &&
      summary.readModelLiveTailCommitCompleted?.includesFirstText === true &&
      summary.readModelLiveTailCommitCompleted?.includesOverflowMarker ===
        true &&
      summary.readModelLiveTailCommitCompleted?.includesTableHeader === true &&
      summary.readModelLiveTailCommitCompleted?.includesTableTail === true &&
      summary.readModelLiveTailCommitCompleted?.includesAssistantDone ===
        true &&
      summary.readModelLiveTailCommitCompleted
        ?.canonicalTextMatchesCompletedItem === true &&
      summary.readModelLiveTailCommitCompleted?.canonicalMarkerExactOnce ===
        true &&
      summary.readModelLiveTailCommitCompleted?.identityMatches === true,
    backendLiveTailCommitRecorded:
      summary.liveTailCommitBackendCompleted?.droppedEventType ===
        "message.delta" &&
      summary.liveTailCommitBackendCompleted?.repairEventType ===
        "item.completed" &&
      summary.liveTailCommitBackendCompleted?.terminalEventType ===
        "turn.completed" &&
      JSON.stringify(
        summary.liveTailCommitBackendCompleted?.emittedEventTypes,
      ) === JSON.stringify(["item.completed", "turn.completed"]) &&
      summary.liveTailCommitBackendCompleted?.itemId ===
        summary.guiLiveTailVisualOracle?.expectedItemId &&
      summary.liveTailCommitBackendCompleted?.threadId ===
        summary.readModelLiveTailCommitCompleted?.readModelThreadId &&
      summary.liveTailCommitBackendCompleted?.turnId ===
        liveTailCommitTurnStart?.turnId,
  };
}

export function buildMcpStructuredContentScenarioAssertions({
  mcpStructuredContentTurnStart,
  summary,
}) {
  return {
    mcpStructuredContentPromptReachedBackend:
      mcpStructuredContentTurnStart?.inputText ===
      MCP_STRUCTURED_CONTENT_PROMPT,
    guiMcpStructuredContentInputSubmitted:
      summary.mcpStructuredContentInputSend?.afterFill
        ?.promptVisibleInTextarea === true &&
      summary.mcpStructuredContentInputSend?.clicked?.clicked === true,
    guiMcpStructuredContentVisible:
      summary.guiMcpStructuredContentCompleted?.hasPrompt === true &&
      ((summary.guiMcpStructuredContentCompleted?.hasStructuredAnswer ===
        true &&
        summary.guiMcpStructuredContentCompleted?.hasReferenceId === true &&
        (summary.guiMcpStructuredContentCompleted?.hasToolName === true ||
          summary.guiMcpStructuredContentCompleted?.expandedDetails
            ?.hasToolName === true)) ||
        summary.guiMcpStructuredContentCompleted?.terminalDetailsCompacted ===
          true) &&
      summary.guiMcpStructuredContentCompleted?.textareaVisible === true &&
      summary.guiMcpStructuredContentCompleted?.textareaDisabled === false &&
      summary.guiMcpStructuredContentCompleted?.stopButtonVisible === false,
    guiMcpStructuredContentEnvelopeHidden:
      summary.guiMcpStructuredContentCompleted?.envelopeVisible === false,
    readModelMcpStructuredContentCompleted:
      summary.readModelMcpStructuredContentCompleted?.includesPrompt === true &&
      (summary.readModelMcpStructuredContentCompleted?.includesAssistantDone ===
        true ||
        summary.readModelMcpStructuredContentCompleted
          ?.includesAssistantSummary === true) &&
      summary.readModelMcpStructuredContentCompleted?.includesMcpTool === true,
    readModelMcpStructuredContentObserved:
      summary.readModelMcpStructuredContentCompleted
        ?.includesStructuredContent === true &&
      summary.readModelMcpStructuredContentCompleted
        ?.structuredContentAnswerVisible === true &&
      summary.readModelMcpStructuredContentCompleted
        ?.structuredContentReferenceVisible === true &&
      summary.readModelMcpStructuredContentCompleted?.outputContainsEnvelope ===
        true,
  };
}

export function buildMediaReferenceScenarioAssertions({
  mediaReferenceTurnStart,
  pageText,
  summary,
}) {
  return {
    mediaReferencePromptReachedBackend:
      mediaReferenceTurnStart?.inputText === MEDIA_REFERENCE_PROMPT,
    guiMediaReferenceInputSubmitted:
      summary.mediaReferenceInputSend?.afterFill?.promptVisibleInTextarea ===
        true && summary.mediaReferenceInputSend?.clicked?.clicked === true,
    guiMediaReferenceCardVisible:
      summary.guiMediaReferenceCompleted?.hasPrompt === true &&
      summary.guiMediaReferenceCompleted?.hasAssistantSummary === true &&
      summary.guiMediaReferenceSnapshot?.hasCard === true &&
      summary.guiMediaReferenceSnapshot?.hasUri === true &&
      summary.guiMediaReferenceSnapshot?.hasMimeType === true,
    guiMediaReferenceDoesNotExposeInlinePayload:
      summary.guiMediaReferenceSnapshot?.bodyTextIncludesInlinePayload ===
        false &&
      summary.readModelMediaReferenceCompleted?.noInlinePayload === true,
    guiMediaReferencePreviewOpened:
      summary.guiMediaReferencePreview?.click?.clicked === true &&
      summary.guiMediaReferencePreview?.preview?.workbenchPreviewVisible ===
        true &&
      summary.guiMediaReferencePreview?.preview?.previewImageVisible === true &&
      summary.guiMediaReferencePreview?.preview
        ?.previewTextIncludesSidecarSource === false &&
      summary.guiMediaReferencePreview?.preview
        ?.bodyTextIncludesInlinePayload === false,
    readModelMediaReferenceCompleted:
      summary.readModelMediaReferenceCompleted?.includesPrompt === true &&
      (summary.readModelMediaReferenceCompleted?.includesAssistantDone ===
        true ||
        summary.readModelMediaReferenceCompleted?.includesAssistantSummary ===
          true) &&
      summary.readModelMediaReferenceCompleted?.latestTurnStatus ===
        "completed",
    readModelMediaReferenceObserved:
      summary.readModelMediaReferenceCompleted?.hasMediaReference === true &&
      summary.readModelMediaReferenceCompleted?.hasReferenceUri === true &&
      summary.readModelMediaReferenceCompleted?.hasSourceOwner === true &&
      summary.readModelMediaReferenceCompleted?.contentPartsKeyObserved ===
        false &&
      pageText.includes(MEDIA_REFERENCE_PROMPT) &&
      pageText.includes(MEDIA_REFERENCE_SUMMARY_TEXT) &&
      pageText.includes(MEDIA_REFERENCE_TITLE) &&
      pageText.includes(MEDIA_REFERENCE_MIME_TYPE),
  };
}
