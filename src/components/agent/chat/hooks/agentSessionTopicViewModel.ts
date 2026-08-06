import type { AgentSessionDetail } from "@/lib/api/agentRuntime/sessionTypes";
import type { AgentExecutionStrategy } from "@/lib/api/agentExecutionRuntime";
import { normalizeLegacyThreadItems } from "@/lib/api/agentTextNormalization";
import { isAuxiliaryAgentSessionId } from "@/lib/api/agentRuntime/sessionIdentity";
import type { AgentThreadItem, AgentThreadTurn, Message } from "../types";
import { filterConversationThreadItems } from "../utils/threadTimelineView";
import {
  isLegacyDefaultProjectId,
  normalizeProjectId,
} from "../utils/topicProjectResolution";
import { mapSessionToTopic, type Topic } from "./agentChatShared";

export const ACTIVE_SESSION_TRANSIENT_MESSAGES_LIMIT = 48;
export const ACTIVE_SESSION_TRANSIENT_TURNS_LIMIT = 48;
export const ACTIVE_SESSION_TRANSIENT_ITEMS_LIMIT = 160;

export type RestoreCandidateSanitizationPlan =
  | {
      kind: "accept";
      sessionId: string;
    }
  | {
      kind: "skip_auxiliary";
      candidateSessionId: string;
      workspaceId: string | null;
    }
  | {
      kind: "reject_workspace";
      candidateSessionId: string;
      mappedWorkspaceId: string;
      workspaceId: string | null;
    }
  | {
      kind: "empty";
    };

export function resolveRestoreCandidateSanitizationPlan(options: {
  candidateSessionId: string | null | undefined;
  mappedWorkspaceId: string | null | undefined;
  workspaceId: string | null | undefined;
}): RestoreCandidateSanitizationPlan {
  const normalizedCandidate = options.candidateSessionId?.trim();
  if (!normalizedCandidate) {
    return { kind: "empty" };
  }

  const resolvedWorkspaceId = normalizeProjectId(options.workspaceId);
  if (isAuxiliaryAgentSessionId(normalizedCandidate)) {
    return {
      kind: "skip_auxiliary",
      candidateSessionId: normalizedCandidate,
      workspaceId: resolvedWorkspaceId,
    };
  }

  const mappedWorkspaceId = options.mappedWorkspaceId;
  if (!mappedWorkspaceId) {
    return {
      kind: "accept",
      sessionId: normalizedCandidate,
    };
  }

  if (isLegacyDefaultProjectId(mappedWorkspaceId)) {
    return {
      kind: "reject_workspace",
      candidateSessionId: normalizedCandidate,
      mappedWorkspaceId,
      workspaceId: resolvedWorkspaceId,
    };
  }

  const normalizedMappedWorkspaceId = normalizeProjectId(mappedWorkspaceId);
  if (!normalizedMappedWorkspaceId) {
    return {
      kind: "accept",
      sessionId: normalizedCandidate,
    };
  }

  if (
    resolvedWorkspaceId &&
    normalizedMappedWorkspaceId !== resolvedWorkspaceId
  ) {
    return {
      kind: "reject_workspace",
      candidateSessionId: normalizedCandidate,
      mappedWorkspaceId,
      workspaceId: resolvedWorkspaceId,
    };
  }

  return {
    kind: "accept",
    sessionId: normalizedCandidate,
  };
}

export function resolveRuntimeThreadStatusFromSessionDetail(
  detail: AgentSessionDetail,
): Topic["status"] | null {
  const status = detail.thread_read?.status?.trim().toLowerCase();
  if (
    status === "waitingaction" ||
    status === "waiting_action" ||
    status === "waiting_request" ||
    status === "needs_input" ||
    (detail.thread_read?.pending_requests?.length ?? 0) > 0
  ) {
    return "waiting";
  }

  if (status === "queued") {
    return "queued";
  }

  if (status === "running") {
    return "running";
  }

  if (status === "failed") {
    return "failed";
  }

  return null;
}

export function mapSessionDetailToTopic(
  sessionId: string,
  detail: AgentSessionDetail,
  fallbackWorkspaceId: string | null,
): Topic {
  const topic = mapSessionToTopic({
    id: sessionId,
    name: detail.name,
    created_at: detail.created_at,
    updated_at: detail.updated_at,
    model: detail.model,
    messages_count: detail.messages_count ?? detail.messages.length,
    execution_strategy: detail.execution_strategy,
    workspace_id: detail.workspace_id ?? fallbackWorkspaceId ?? undefined,
    working_dir: detail.working_dir,
    section: detail.section,
    section_entered_at: detail.section_entered_at,
  });
  const runtimeStatus = resolveRuntimeThreadStatusFromSessionDetail(detail);
  if (!runtimeStatus) {
    return topic;
  }

  return {
    ...topic,
    status: runtimeStatus,
    statusReason: runtimeStatus === "waiting" ? "user_action" : "default",
  };
}

export function upsertTopicFromSessionDetail(
  topics: Topic[],
  detailTopic: Topic,
): Topic[] {
  const existingTopic = topics.find((topic) => topic.id === detailTopic.id);
  const mergedTopic = existingTopic
    ? {
        ...detailTopic,
        title: shouldPreserveExistingTopicTitle(existingTopic, detailTopic)
          ? existingTopic.title
          : detailTopic.title,
        workspaceId: detailTopic.workspaceId ?? existingTopic.workspaceId,
        workingDir: detailTopic.workingDir ?? existingTopic.workingDir,
        queuedTurnCount:
          detailTopic.queuedTurnCount ?? existingTopic.queuedTurnCount,
        section: detailTopic.section ?? existingTopic.section,
        sectionEnteredAt:
          detailTopic.sectionEnteredAt ?? existingTopic.sectionEnteredAt,
        hasUnread: existingTopic.hasUnread,
        tag: existingTopic.tag,
      }
    : detailTopic;
  const nextTopics = existingTopic
    ? topics.map((topic) => (topic.id === detailTopic.id ? mergedTopic : topic))
    : [mergedTopic, ...topics];

  return nextTopics;
}

function shouldPreserveExistingTopicTitle(
  existingTopic: Topic,
  detailTopic: Topic,
): boolean {
  const existingTitle = existingTopic.title.trim();
  const detailTitle = detailTopic.title.trim();
  if (!existingTitle || existingTitle === detailTitle) {
    return false;
  }
  if (existingTitle === "新任务") {
    return false;
  }
  return (
    detailTitle === "新任务" ||
    /^任务 \d{4}\/\d{1,2}\/\d{1,2}$/.test(detailTitle)
  );
}

export function upsertFreshSessionDraftTopic(
  topics: Topic[],
  params: {
    createdAt: Date;
    executionStrategy: AgentExecutionStrategy;
    sessionId: string;
    sessionName?: string | null;
    workspaceId: string | null | undefined;
    workingDir?: string | null;
  },
): Topic[] {
  const title = params.sessionName?.trim() || "新任务";
  const topic: Topic = {
    id: params.sessionId,
    title,
    createdAt: params.createdAt,
    updatedAt: params.createdAt,
    workspaceId: params.workspaceId,
    workingDir: params.workingDir ?? null,
    messagesCount: 0,
    executionStrategy: params.executionStrategy,
    status: "draft",
    lastPreview: "等待你补充任务需求后开始执行。",
    hasUnread: false,
    tag: null,
    sourceSessionId: params.sessionId,
  };

  return [topic, ...topics.filter((item) => item.id !== params.sessionId)];
}

export function prependVerifiedSessionTopicFromDetail(
  topics: Topic[],
  sessionId: string,
  detail: AgentSessionDetail,
): Topic[] {
  if (topics.some((topic) => topic.id === sessionId)) {
    return topics;
  }

  return [
    mapSessionToTopic({
      id: sessionId,
      name: detail.name,
      created_at: detail.created_at,
      updated_at: detail.updated_at,
      model: detail.model,
      messages_count: detail.messages.length,
      execution_strategy: detail.execution_strategy,
      workspace_id: detail.workspace_id,
      working_dir: detail.working_dir,
      section: detail.section,
      section_entered_at: detail.section_entered_at,
    }),
    ...topics,
  ];
}

export function applyTopicExecutionStrategyToTopics(
  topics: Topic[],
  targetSessionId: string,
  nextExecutionStrategy: AgentExecutionStrategy,
): Topic[] {
  return topics.map((topic) =>
    topic.id === targetSessionId
      ? { ...topic, executionStrategy: nextExecutionStrategy }
      : topic,
  );
}

export type TopicSnapshotPatch = Partial<
  Pick<
    Topic,
    | "updatedAt"
    | "messagesCount"
    | "status"
    | "statusReason"
    | "queuedTurnCount"
    | "lastPreview"
    | "hasUnread"
  >
>;

export function applyTopicSnapshotToTopics(
  topics: Topic[],
  targetSessionId: string,
  snapshot: TopicSnapshotPatch,
): Topic[] {
  let changed = false;
  const nextTopics = topics.map((topic) => {
    if (topic.id !== targetSessionId) {
      return topic;
    }

    const { updatedAt, ...restSnapshot } = snapshot;
    const nextTopic = {
      ...topic,
      ...restSnapshot,
      ...(updatedAt ? { updatedAt } : {}),
    };

    const unchanged =
      nextTopic.messagesCount === topic.messagesCount &&
      nextTopic.status === topic.status &&
      nextTopic.statusReason === topic.statusReason &&
      nextTopic.queuedTurnCount === topic.queuedTurnCount &&
      nextTopic.lastPreview === topic.lastPreview &&
      nextTopic.hasUnread === topic.hasUnread &&
      nextTopic.updatedAt?.getTime() === topic.updatedAt?.getTime();

    if (unchanged) {
      return topic;
    }

    changed = true;
    return nextTopic;
  });

  return changed ? nextTopics : topics;
}

function takeTail<T>(items: T[], limit: number): T[] {
  if (items.length <= limit) {
    return items;
  }

  return items.slice(-limit);
}

export function selectActiveSessionTransientTurns(
  turns: AgentThreadTurn[],
): AgentThreadTurn[] {
  return takeTail(turns, ACTIVE_SESSION_TRANSIENT_TURNS_LIMIT);
}

export function selectActiveSessionTransientMessages(
  messages: Message[],
): Message[] {
  return takeTail(messages, ACTIVE_SESSION_TRANSIENT_MESSAGES_LIMIT);
}

export function selectActiveSessionTransientItems(
  items: AgentThreadItem[],
  turns: AgentThreadTurn[],
): AgentThreadItem[] {
  const retainedTurnIds = new Set(
    selectActiveSessionTransientTurns(turns)
      .map((turn) => (typeof turn.id === "string" ? turn.id.trim() : ""))
      .filter(Boolean),
  );
  if (retainedTurnIds.size === 0) {
    return filterConversationThreadItems(
      normalizeLegacyThreadItems(
        takeTail(items, ACTIVE_SESSION_TRANSIENT_ITEMS_LIMIT),
      ),
    );
  }

  const scopedItems: AgentThreadItem[] = [];
  for (
    let index = items.length - 1;
    index >= 0 && scopedItems.length < ACTIVE_SESSION_TRANSIENT_ITEMS_LIMIT;
    index -= 1
  ) {
    const item = items[index];
    if (!item) {
      continue;
    }

    const turnId = typeof item.turn_id === "string" ? item.turn_id.trim() : "";
    if (!turnId || retainedTurnIds.has(turnId)) {
      scopedItems.push(item);
    }
  }
  scopedItems.reverse();

  return filterConversationThreadItems(normalizeLegacyThreadItems(scopedItems));
}
