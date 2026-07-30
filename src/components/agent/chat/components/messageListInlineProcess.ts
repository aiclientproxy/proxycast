import type { AgentThreadItem, AgentThreadTurn, Message } from "../types";
import { isRetainedSkillProcessMessage } from "../utils/skillInlineProcessRetention";
import { isUnifiedWebSearchToolName } from "../utils/searchResultPreview";
import {
  isUpdatePlanToolName,
  isUnifiedWebFetchToolName,
} from "../utils/toolNameFamily";
import { shouldHideTurnSummaryFromConversation } from "../utils/turnSummaryPresentation";
import {
  buildAgentTextDeltaContentPartMetadata,
  canMergeCoalescibleContentParts,
  mergeIncrementalTextWithOverlap,
} from "../utils/contentPartTimeline";

export function isDeferredTimelineItem(item: AgentThreadItem): boolean {
  return item.type === "file_artifact" || item.type === "turn_summary";
}

export function dedupeDeferredTimelineItems(
  items: AgentThreadItem[],
): AgentThreadItem[] {
  const seenItemIds = new Set<string>();
  return items.filter((item) => {
    if (seenItemIds.has(item.id)) {
      return false;
    }
    seenItemIds.add(item.id);
    return true;
  });
}

export function shouldRenderConversationTimelineItem(
  item: AgentThreadItem,
  timelineItems: AgentThreadItem[],
  options?: {
    hasInlineRuntimeStatus?: boolean;
  },
): boolean {
  if (item.type === "user_message" || item.type === "agent_message") {
    return false;
  }

  if (item.type !== "turn_summary") {
    return true;
  }

  if (shouldHideTurnSummaryFromConversation(item)) {
    return false;
  }

  if (item.status === "in_progress" && options?.hasInlineRuntimeStatus) {
    return false;
  }

  if (item.status !== "completed") {
    return true;
  }

  return !timelineItems.some(
    (entry) => entry.id !== item.id && entry.type !== "turn_summary",
  );
}

export function hasTimelineProcessItems(items?: AgentThreadItem[]): boolean {
  return Boolean(
    items?.some(
      (item) =>
        item.type === "plan" ||
        item.type === "reasoning" ||
        (item.type === "tool_call" && !isUpdatePlanToolName(item.tool_name)) ||
        item.type === "command_execution" ||
        item.type === "patch" ||
        item.type === "web_search" ||
        item.type === "hook_prompt" ||
        item.type === "sleep" ||
        item.type === "review_boundary" ||
        item.type === "approval_request" ||
        item.type === "request_user_input" ||
        item.type === "context_compaction",
    ),
  );
}

export function hasPersistedReasoningTimelineItem(
  items?: AgentThreadItem[],
): boolean {
  return Boolean(items?.some((item) => item.type === "reasoning"));
}

function hasInlineThinkingContent(message: Message): boolean {
  return (
    Boolean(message.thinkingContent?.trim()) ||
    Boolean(
      message.contentParts?.some(
        (part) => part.type === "thinking" && part.text.trim().length > 0,
      ),
    )
  );
}

export function resolveInlineThinkingContent(
  message: Message,
): string | undefined {
  const explicitThinking = message.thinkingContent?.trim()
    ? message.thinkingContent
    : undefined;
  if (explicitThinking) {
    return explicitThinking;
  }

  const thinkingText = (message.contentParts || [])
    .filter(
      (
        part,
      ): part is Extract<
        NonNullable<Message["contentParts"]>[number],
        { type: "thinking"; text: string }
      > => part.type === "thinking" && part.text.trim().length > 0,
    )
    .map((part) => part.text)
    .join("");

  return thinkingText.trim() ? thinkingText : undefined;
}

function hasCompletedOrRunningWebRetrievalProcess(
  message: Message,
  displayContent: string,
): boolean {
  if (displayContent.trim()) {
    return false;
  }

  const isCompletedOrRunningWebRetrievalTool = (toolCall: {
    name: string;
    status: string;
  }) =>
    (toolCall.status === "running" || toolCall.status === "completed") &&
    (isUnifiedWebSearchToolName(toolCall.name) ||
      isUnifiedWebFetchToolName(toolCall.name));

  return (
    (message.toolCalls || []).some(isCompletedOrRunningWebRetrievalTool) ||
    (message.contentParts || []).some(
      (part) =>
        part.type === "tool_use" &&
        isCompletedOrRunningWebRetrievalTool(part.toolCall),
    )
  );
}

export function shouldKeepInlineProcessForActiveAssistant(
  message: Message,
  isConversationTailAssistant: boolean,
  hasProcessTimelineItems: boolean,
  hasPersistedReasoningTimeline: boolean,
  hasTurnContext: boolean,
  displayContent: string,
  isSending: boolean,
): boolean {
  if (message.role !== "assistant") {
    return false;
  }

  if (message.isThinking) {
    return true;
  }

  if (
    isRetainedSkillProcessMessage(message) &&
    hasInlineThinkingContent(message)
  ) {
    return true;
  }

  if (!isConversationTailAssistant) {
    return false;
  }

  if (
    hasTurnContext &&
    !hasProcessTimelineItems &&
    !hasPersistedReasoningTimeline &&
    hasInlineThinkingContent(message)
  ) {
    return true;
  }

  if (
    !hasTurnContext &&
    message.runtimeTurnId?.trim() &&
    hasInlineThinkingContent(message)
  ) {
    return true;
  }

  const hasRunningToolCall =
    !displayContent.trim() &&
    ((message.toolCalls || []).some(
      (toolCall) => toolCall.status === "running",
    ) ||
      (message.contentParts || []).some(
        (part) =>
          part.type === "tool_use" && part.toolCall.status === "running",
      ));
  const hasPendingActionRequest =
    (message.actionRequests || []).some(
      (request) => request.status !== "submitted",
    ) ||
    (message.contentParts || []).some(
      (part) =>
        part.type === "action_required" &&
        part.actionRequired.status !== "submitted",
    );
  const hasActiveWebRetrievalProcess = hasCompletedOrRunningWebRetrievalProcess(
    message,
    displayContent,
  );
  const hasActiveRuntimeStatus =
    Boolean(message.runtimeStatus) &&
    (message.isThinking || isSending || hasActiveWebRetrievalProcess) &&
    message.runtimeStatus?.phase !== "failed" &&
    message.runtimeStatus?.phase !== "cancelled";

  return (
    hasRunningToolCall || hasPendingActionRequest || hasActiveRuntimeStatus
  );
}

function isPreAnswerThinkingTimelineItem(item: AgentThreadItem): boolean {
  if (item.status === "failed") {
    return false;
  }

  return (
    item.type === "plan" ||
    item.type === "turn_summary" ||
    item.type === "context_compaction"
  );
}

export function shouldSuppressPreAnswerThinkingTimeline(params: {
  message: Message;
  turn: AgentThreadTurn;
  items: AgentThreadItem[];
  displayContent: string;
}): boolean {
  if (
    !params.message.isThinking ||
    params.turn.status === "completed" ||
    params.displayContent.trim() ||
    params.items.length === 0
  ) {
    return false;
  }

  return params.items.every(isPreAnswerThinkingTimelineItem);
}

export interface InlineProcessCoverage {
  hasInlineProcessEntries: boolean;
  plan: boolean;
  thinking: boolean;
  toolNameCounts: Map<string, number>;
  actionRequestCounts: Map<string, number>;
}

function normalizeInlineCoverageKey(
  value: string | null | undefined,
): string | null {
  const normalized = value?.trim().toLowerCase();
  if (!normalized) {
    return null;
  }

  if (
    normalized === "bash" ||
    normalized === "exec_command" ||
    normalized === "command_execution"
  ) {
    return "command_execution";
  }

  if (
    normalized === "websearch" ||
    normalized === "web_search" ||
    normalized === "search_query"
  ) {
    return "web_search";
  }

  return normalized;
}

function incrementInlineCoverageCount(
  counts: Map<string, number>,
  key: string | null,
) {
  if (!key) {
    return;
  }
  counts.set(key, (counts.get(key) || 0) + 1);
}

function consumeInlineCoverageCount(
  counts: Map<string, number>,
  key: string | null,
): boolean {
  if (!key) {
    return false;
  }
  const current = counts.get(key) || 0;
  if (current <= 0) {
    return false;
  }
  if (current === 1) {
    counts.delete(key);
  } else {
    counts.set(key, current - 1);
  }
  return true;
}

export function createInlineCoverageMatcher(coverage: InlineProcessCoverage) {
  const remainingToolNameCounts = new Map(coverage.toolNameCounts);
  const remainingActionRequestCounts = new Map(coverage.actionRequestCounts);

  return (item: AgentThreadItem): boolean => {
    switch (item.type) {
      case "plan":
        return coverage.plan;
      case "reasoning":
        return coverage.thinking;
      case "tool_call":
        if (isUpdatePlanToolName(item.tool_name)) {
          return true;
        }
        return consumeInlineCoverageCount(
          remainingToolNameCounts,
          normalizeInlineCoverageKey(item.tool_name),
        );
      case "command_execution":
        return consumeInlineCoverageCount(
          remainingToolNameCounts,
          normalizeInlineCoverageKey("command_execution"),
        );
      case "patch":
        return consumeInlineCoverageCount(
          remainingToolNameCounts,
          normalizeInlineCoverageKey("patch"),
        );
      case "web_search":
        return consumeInlineCoverageCount(
          remainingToolNameCounts,
          normalizeInlineCoverageKey("web_search"),
        );
      case "approval_request":
      case "request_user_input":
        return consumeInlineCoverageCount(
          remainingActionRequestCounts,
          normalizeInlineCoverageKey(item.request_id),
        );
      default:
        return false;
    }
  };
}

export function resolveInlineProcessCoverage(params: {
  contentParts?: Message["contentParts"];
  thinkingContent?: string;
  toolCalls?: Message["toolCalls"];
  actionRequests?: Message["actionRequests"];
}): InlineProcessCoverage {
  const contentParts = params.contentParts || [];
  const toolNameCounts = new Map<string, number>();
  const actionRequestCounts = new Map<string, number>();
  let plan = false;
  let thinking = false;

  if (contentParts.length > 0) {
    plan = contentParts.some(
      (part) => part.type === "text" && part.text.includes("<proposed_plan>"),
    );
    thinking = contentParts.some(
      (part) => part.type === "thinking" && part.text.trim().length > 0,
    );
    contentParts.forEach((part) => {
      if (part.type === "tool_use") {
        incrementInlineCoverageCount(
          toolNameCounts,
          normalizeInlineCoverageKey(part.toolCall.name),
        );
        return;
      }
      if (part.type === "action_required") {
        incrementInlineCoverageCount(
          actionRequestCounts,
          normalizeInlineCoverageKey(part.actionRequired.requestId),
        );
      }
    });
  } else {
    thinking = Boolean(params.thinkingContent?.trim());
    (params.toolCalls || []).forEach((toolCall) => {
      incrementInlineCoverageCount(
        toolNameCounts,
        normalizeInlineCoverageKey(toolCall.name),
      );
    });
  }

  (params.actionRequests || []).forEach((actionRequest) => {
    const actionKey = normalizeInlineCoverageKey(actionRequest.requestId);
    if (actionKey && !actionRequestCounts.has(actionKey)) {
      incrementInlineCoverageCount(actionRequestCounts, actionKey);
    }
  });

  return {
    hasInlineProcessEntries:
      plan ||
      thinking ||
      toolNameCounts.size > 0 ||
      actionRequestCounts.size > 0,
    plan,
    thinking,
    toolNameCounts,
    actionRequestCounts,
  };
}

export function filterConversationDisplayContentParts(
  parts: Message["contentParts"] | undefined,
  options: {
    includeProcessFlow: boolean;
    preserveToolUseParts: boolean;
  },
): Message["contentParts"] | undefined {
  if (!parts || parts.length === 0 || options.includeProcessFlow) {
    return parts;
  }

  const filtered = parts.filter((part) => {
    if (part.type === "thinking") {
      return false;
    }

    if (part.type === "tool_use") {
      return options.preserveToolUseParts;
    }

    // file_changes_batch 是文件改动汇总卡，不属于 process flow，始终保留
    if (part.type === "file_changes_batch") {
      return true;
    }

    return true;
  });
  return filtered.length > 0 ? filtered : undefined;
}

function isTimelineProcessBoundaryPart(
  part: NonNullable<Message["contentParts"]>[number],
): boolean {
  return (
    part.type === "tool_use" ||
    part.type === "action_required" ||
    part.type === "file_changes_batch"
  );
}

type StreamingOverlayInput =
  | string
  | {
      content?: string | null;
      itemId?: string | null;
      phase?: string | null;
      sequence?: number | null;
      turnId?: string | null;
    }
  | null
  | undefined;

type TextContentPart = Extract<
  NonNullable<Message["contentParts"]>[number],
  { type: "text" }
>;

function normalizeStreamingOverlayInput(
  overlay: StreamingOverlayInput,
): { content: string; metadata?: Record<string, unknown> } | null {
  if (!overlay) {
    return null;
  }
  if (typeof overlay === "string") {
    return overlay ? { content: overlay } : null;
  }

  const content = overlay.content || "";
  if (!content) {
    return null;
  }
  const metadata = buildAgentTextDeltaContentPartMetadata({
    itemId: overlay.itemId,
    phase: overlay.phase,
    sequence: overlay.sequence,
    turnId: overlay.turnId,
  });
  return metadata ? { content, metadata } : { content };
}

function createOverlayTextPart(
  text: string,
  metadata?: Record<string, unknown>,
): TextContentPart {
  return {
    type: "text",
    text,
    ...(metadata ? { metadata } : {}),
  };
}

function hasStructuredTextProvenance(part: TextContentPart): boolean {
  const metadata = part.metadata;
  return Boolean(
    metadata?.itemId ||
    metadata?.threadItemId ||
    metadata?.turnId ||
    metadata?.phase ||
    typeof metadata?.sequence === "number",
  );
}

function mergeTextPart(
  base: TextContentPart,
  chunk: TextContentPart,
): TextContentPart {
  return {
    ...base,
    type: "text",
    text: mergeIncrementalTextWithOverlap(base.text, chunk.text),
    metadata: base.metadata ?? chunk.metadata,
  };
}

function mergeOverlayIntoLastPreProcessText(
  parts: NonNullable<Message["contentParts"]>,
  pendingPart: TextContentPart,
): Message["contentParts"] | null {
  const processBoundaryIndex = parts.findIndex(isTimelineProcessBoundaryPart);
  if (processBoundaryIndex < 0) {
    return null;
  }

  for (let index = processBoundaryIndex - 1; index >= 0; index -= 1) {
    const part = parts[index];
    if (part?.type !== "text") {
      continue;
    }
    if (
      !hasStructuredTextProvenance(part) ||
      !hasStructuredTextProvenance(pendingPart) ||
      !canMergeCoalescibleContentParts(part, pendingPart)
    ) {
      return null;
    }
    const nextParts = [...parts];
    nextParts[index] = mergeTextPart(part, pendingPart);
    return nextParts;
  }

  return null;
}

export function mergeStreamingOverlayContentParts(
  parts: Message["contentParts"] | undefined,
  overlay: StreamingOverlayInput,
): Message["contentParts"] | undefined {
  const normalizedOverlay = normalizeStreamingOverlayInput(overlay);
  if (!normalizedOverlay) {
    return parts;
  }

  const overlayContent = normalizedOverlay.content;
  const textPart = createOverlayTextPart(
    overlayContent,
    normalizedOverlay.metadata,
  );
  if (!parts?.length) {
    return [textPart];
  }

  const committedText = parts
    .filter((part): part is Extract<typeof part, { type: "text" }> => {
      return part.type === "text";
    })
    .map((part) => part.text)
    .join("");
  if (committedText && overlayContent.startsWith(committedText)) {
    const pendingText = overlayContent.slice(committedText.length);
    if (!pendingText) {
      return parts;
    }
    const nextParts = [...parts];
    const lastPart = nextParts[nextParts.length - 1];
    if (lastPart?.type === "text") {
      const pendingPart = createOverlayTextPart(
        pendingText,
        normalizedOverlay.metadata,
      );
      if (canMergeCoalescibleContentParts(lastPart, pendingPart)) {
        nextParts[nextParts.length - 1] = mergeTextPart(lastPart, pendingPart);
      } else {
        nextParts.push(pendingPart);
      }
      return nextParts;
    }
    const preProcessMergedParts = mergeOverlayIntoLastPreProcessText(
      nextParts,
      createOverlayTextPart(pendingText, normalizedOverlay.metadata),
    );
    if (preProcessMergedParts) {
      return preProcessMergedParts;
    }
    return [
      ...nextParts,
      createOverlayTextPart(pendingText, normalizedOverlay.metadata),
    ];
  }

  const processBoundaryIndex = parts.findIndex(isTimelineProcessBoundaryPart);
  if (processBoundaryIndex >= 0) {
    const textIndexes = parts.flatMap((part, index) =>
      part.type === "text" ? [index] : [],
    );
    const lastTextAfterProcessIndex = textIndexes
      .slice()
      .reverse()
      .find((index) => index > processBoundaryIndex);

    if (lastTextAfterProcessIndex !== undefined) {
      const prefixBeforeLast = textIndexes
        .filter((index) => index < lastTextAfterProcessIndex)
        .map((index) => {
          const part = parts[index];
          return part?.type === "text" ? part.text : "";
        })
        .join("");
      const nextText =
        prefixBeforeLast && overlayContent.startsWith(prefixBeforeLast)
          ? overlayContent.slice(prefixBeforeLast.length)
          : overlayContent;
      const nextParts = [...parts];
      nextParts[lastTextAfterProcessIndex] = {
        ...nextParts[lastTextAfterProcessIndex],
        type: "text",
        text: nextText,
        ...(normalizedOverlay.metadata
          ? { metadata: normalizedOverlay.metadata }
          : {}),
      };
      return nextParts;
    }

    return [...parts, textPart];
  }

  const firstTextIndex = parts.findIndex((part) => part.type === "text");
  if (firstTextIndex < 0) {
    return [...parts, textPart];
  }

  return parts.flatMap<NonNullable<Message["contentParts"]>[number]>(
    (part, index) => {
      if (part.type !== "text") {
        return [part];
      }
      return index === firstTextIndex ? [textPart] : [];
    },
  );
}
