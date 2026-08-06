import type { AgentSessionInfo } from "@/lib/api/agentRuntime/sessionTypes";
import type { ConversationImportThreadCommitResponse } from "@/lib/api/conversationImport";
import { isAuxiliaryAgentSessionId } from "@/lib/api/agentRuntime/sessionIdentity";
import { resolveSidebarSessionTitle } from "./sidebarSessionFormatting";

export function buildSidebarSessionRequestLimit(
  visibleCount: number,
  pageSize: number,
): number {
  const normalizedVisibleCount = Math.max(visibleCount, pageSize);
  return normalizedVisibleCount + 1;
}

export function splitSidebarSessionResult(params: {
  sessions: AgentSessionInfo[];
  visibleCount: number;
  pageSize: number;
}): {
  sessions: AgentSessionInfo[];
  hasMore: boolean;
} {
  const { sessions, visibleCount, pageSize } = params;
  const targetCount = Math.max(visibleCount, pageSize);
  return {
    sessions: sessions.slice(0, targetCount),
    hasMore: sessions.length > targetCount,
  };
}

export function hasCachedSidebarSessionEntry(
  sessions: AgentSessionInfo[],
  sessionId?: string | null,
): boolean {
  const normalizedSessionId = sessionId?.trim();
  if (!normalizedSessionId) {
    return false;
  }

  return sessions.some((session) => session.id === normalizedSessionId);
}

export function filterSidebarSessions(
  sessions: AgentSessionInfo[],
): AgentSessionInfo[] {
  return sessions.filter((session) => !isAuxiliaryAgentSessionId(session.id));
}

export function buildVisibleSidebarSessions(params: {
  sessions: AgentSessionInfo[];
  currentSessionId?: string | null;
  limit: number;
}): AgentSessionInfo[] {
  const { sessions, currentSessionId, limit } = params;
  if (limit <= 0) {
    return [];
  }

  if (sessions.length <= limit) {
    return sessions;
  }

  const visibleSessions = sessions.slice(0, limit);
  const normalizedCurrentSessionId = currentSessionId?.trim();
  if (
    !normalizedCurrentSessionId ||
    visibleSessions.some((session) => session.id === normalizedCurrentSessionId)
  ) {
    return visibleSessions;
  }

  const currentSession = sessions.find(
    (session) => session.id === normalizedCurrentSessionId,
  );
  if (!currentSession) {
    return visibleSessions;
  }

  return [...visibleSessions.slice(0, Math.max(limit - 1, 0)), currentSession];
}

export function normalizeSidebarSearchText(value: string): string {
  return value.trim().toLocaleLowerCase();
}

export function matchesSidebarSessionTitle(
  session: AgentSessionInfo,
  normalizedQuery: string,
  fallbackTitle: string,
): boolean {
  if (!normalizedQuery) {
    return true;
  }

  return normalizeSidebarSearchText(
    resolveSidebarSessionTitle(session, fallbackTitle),
  ).includes(normalizedQuery);
}

function parseTimestampSeconds(value?: string | null): number | null {
  if (!value) {
    return null;
  }
  const timestampMs = Date.parse(value);
  if (!Number.isFinite(timestampMs)) {
    return null;
  }
  return Math.floor(timestampMs / 1000);
}

export function buildImportedSidebarSession(
  response: ConversationImportThreadCommitResponse,
): AgentSessionInfo {
  const createdAt =
    parseTimestampSeconds(response.session.createdAt) ??
    parseTimestampSeconds(response.thread.createdAt) ??
    Math.floor(Date.now() / 1000);
  const updatedAt =
    parseTimestampSeconds(response.session.updatedAt) ??
    parseTimestampSeconds(response.thread.updatedAt) ??
    createdAt;

  return {
    id: response.session.sessionId,
    name: response.thread.title?.trim() || undefined,
    created_at: createdAt,
    updated_at: updatedAt,
    archived_at: response.thread.archived ? updatedAt : null,
    model: response.thread.modelProvider,
    messages_count: response.importedMessages,
    workspace_id: response.session.workspaceId,
    working_dir: response.thread.cwd,
  };
}
