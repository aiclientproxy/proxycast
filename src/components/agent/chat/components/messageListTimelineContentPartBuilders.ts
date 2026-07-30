import type { AgentToolCallState } from "@/lib/api/agentProtocol";
import type { AgentThreadItem } from "../types";
import {
  aggregateCanonicalPatchChanges,
  aggregateFileChanges,
  type FileChangesAggregate,
} from "../utils/fileChangeSummary";
import { isUpdatePlanToolName } from "../utils/toolNameFamily";
import {
  toActionRequired,
  toToolCallState,
} from "./timeline-utils/itemConverters";
import type { MessageContentPart } from "./messageListTimelineContentPartTypes";
import {
  appendTextContentPart,
  appendThinkingContentPart,
  shouldRenderTimelineAgentMessageAsCommentaryText,
} from "./messageListTimelineContentPartText";

function stringifyTimelineArguments(value: unknown): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value === "string") {
    return value;
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function resolveTimelineToolStatus(
  status: AgentThreadItem["status"],
): AgentToolCallState["status"] {
  if (status === "in_progress") {
    return "running";
  }
  return status;
}

export function metadataRecord(
  value: unknown,
): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function mergeTimelineMetadata(
  base: Record<string, unknown> | undefined,
  extra: Record<string, unknown>,
): Record<string, unknown> {
  return {
    ...(base || {}),
    ...extra,
  };
}

export function timelineItemMetadata(
  item: AgentThreadItem,
  source: string,
): Record<string, unknown> {
  return mergeTimelineMetadata(metadataRecord(item.metadata), {
    source,
    threadItemId: item.id,
    turnId: item.turn_id,
    sequence: item.sequence,
    ...(item.type === "agent_message" && item.phase
      ? { phase: item.phase }
      : {}),
  });
}

export function timelineTextMetadata(
  item: Extract<AgentThreadItem, { type: "agent_message" }>,
): Record<string, unknown> {
  return timelineItemMetadata(item, "agent_thread_item");
}

export function buildTimelineToolContentPart(
  item: AgentThreadItem,
): MessageContentPart | null {
  if (item.type === "tool_call" && isUpdatePlanToolName(item.tool_name)) {
    return null;
  }

  if (
    item.type === "tool_call" ||
    item.type === "command_execution" ||
    item.type === "web_search" ||
    item.type === "hook"
  ) {
    const toolCall = toToolCallState(item);
    if (!toolCall) {
      return null;
    }
    const metadata = timelineItemMetadata(item, "agent_thread_item");
    return {
      type: "tool_use",
      toolCall,
      metadata,
    };
  }

  return null;
}

export function buildTimelineActionContentPart(
  item: AgentThreadItem,
): MessageContentPart | null {
  if (item.type !== "approval_request" && item.type !== "request_user_input") {
    return null;
  }

  const actionRequired = toActionRequired(item);
  if (!actionRequired) {
    return null;
  }

  return {
    type: "action_required",
    actionRequired,
  };
}

export function isTimelineProcessItem(item: AgentThreadItem): boolean {
  return (
    item.type === "reasoning" ||
    item.type === "plan" ||
    (item.type === "tool_call" && !isUpdatePlanToolName(item.tool_name)) ||
    item.type === "command_execution" ||
    item.type === "patch" ||
    item.type === "web_search" ||
    item.type === "hook" ||
    item.type === "hook_prompt" ||
    item.type === "sleep" ||
    item.type === "review_boundary" ||
    item.type === "subagent_activity" ||
    item.type === "context_compaction" ||
    item.type === "approval_request" ||
    item.type === "request_user_input"
  );
}

export function buildSparseTimelineInlinePart(
  item: AgentThreadItem,
): MessageContentPart | null {
  if (item.type === "reasoning") {
    const part: MessageContentPart[] = [];
    appendThinkingContentPart(
      part,
      item.text,
      timelineItemMetadata(item, "thread_item_reasoning"),
    );
    return part[0] ?? null;
  }

  if (shouldRenderTimelineAgentMessageAsCommentaryText(item)) {
    const part: MessageContentPart[] = [];
    appendTextContentPart(
      part,
      item.text,
      timelineItemMetadata(item, "agent_thread_item"),
    );
    return part[0] ?? null;
  }

  return null;
}

export function buildTimelinePatchContentPart(
  item: AgentThreadItem,
): MessageContentPart | null {
  if (item.type !== "patch") {
    return null;
  }

  const metadata = timelineItemMetadata(item, "thread_item_patch");
  const canonicalAggregate = aggregateCanonicalPatchChanges(
    item.changes ?? [],
    resolveTimelineToolStatus(item.status),
    item.file_status,
  );
  if (canonicalAggregate.fileCount > 0) {
    return {
      type: "file_changes_batch",
      aggregate: canonicalAggregate,
      metadata,
    };
  }
  const aggregate = aggregateFileChanges([buildTimelinePatchToolCall(item)]);
  if (aggregate.fileCount === 0) {
    const fallbackAggregate = buildTimelinePatchFallbackAggregate(item);
    return fallbackAggregate
      ? {
          type: "file_changes_batch",
          aggregate: fallbackAggregate,
          metadata,
        }
      : null;
  }

  return { type: "file_changes_batch", aggregate, metadata };
}

function buildTimelinePatchToolCall(
  item: Extract<AgentThreadItem, { type: "patch" }>,
): AgentToolCallState {
  const status = resolveTimelineToolStatus(item.status);
  const metadata = metadataRecord(item.metadata);
  const primaryPath = item.paths?.find((path) => path.trim().length > 0);
  return {
    id: item.id,
    name: "Patch",
    arguments: stringifyTimelineArguments({
      path: primaryPath,
      paths: item.paths,
    }),
    status,
    startTime: new Date(item.started_at),
    endTime: item.completed_at ? new Date(item.completed_at) : undefined,
    metadata,
    result:
      status === "running"
        ? undefined
        : {
            success: item.success !== false,
            output: [item.stdout, item.stderr, item.text]
              .filter((value): value is string => Boolean(value?.trim()))
              .join("\n"),
            metadata,
          },
  };
}

function buildTimelinePatchFallbackAggregate(
  item: Extract<AgentThreadItem, { type: "patch" }>,
): FileChangesAggregate | null {
  const paths = (item.paths?.length ? item.paths : item.summary || [])
    .map((path) => path.trim())
    .filter(Boolean);
  const uniquePaths = Array.from(new Set(paths));
  if (uniquePaths.length === 0) {
    return null;
  }

  const files = uniquePaths.map((path) => ({
    path,
    kind: "update" as const,
    linesAdded: 0,
    linesRemoved: 0,
    diff: [],
    truncated: false,
    source: "backend" as const,
    status: resolveTimelineToolStatus(item.status),
    fileStatus: item.file_status,
  }));

  return {
    files,
    totalAdded: 0,
    totalRemoved: 0,
    fileCount: files.length,
  };
}
