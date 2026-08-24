import {
  AppServerClient,
  type AppServerThreadListParams,
  type AppServerThreadListResponse,
  type AppServerThreadSectionListResponse,
  type AppServerThreadReadParams,
} from "@/lib/api/appServer";
import { METHOD_THREAD_LIST } from "../../../../packages/app-server-client/src/protocol";
import type {
  AgentExecutionStrategy,
  AgentSessionExecutionRuntimePreferences,
} from "../agentExecutionRuntime";
import {
  readCanonicalThreadDetail,
  readCanonicalThreadListResponse,
} from "./appServerCanonicalThreadProjection";
import { readCanonicalThreadHistoryWindow } from "./canonicalThreadHistoryWindow";
import type {
  AgentRuntimeCreateSessionOptions,
  AgentRuntimeGetSessionOptions,
} from "./requestTypes";
import type {
  AgentSessionDetail,
  AgentSessionInfo,
  AgentRuntimeListSessionsOptions,
} from "./sessionTypes";

const THREAD_LIST_PAGE_LIMIT = 100;

export type AppServerSessionRpcClient = Pick<
  AppServerClient,
  | "startSession"
  | "listThreadSections"
  | "readThread"
  | "updateThreadSettings"
  | "archiveThread"
  | "unarchiveThread"
  | "forkThread"
  | "deleteThread"
  | "request"
>;

export type AppServerAgentSessionOverview = {
  sessionId: string;
  threadId?: string;
  parentThreadId?: string;
  canAcceptDirectInput?: boolean;
  title?: string;
  businessObjectRefMetadata?: unknown;
  model: string;
  createdAt: string;
  updatedAt: string;
  archivedAt?: string | null;
  workspaceId?: string;
  workingDir?: string;
  executionStrategy?: string;
  messagesCount: number;
  threadStatus?: string;
  latestTurnStatus?: string;
  activeTurnId?: string;
  queuedTurnCount?: number;
  section?: { id: string; name: string };
  sectionEnteredAt?: number;
};

export interface AppServerSessionClientDeps {
  appServerClient?: AppServerSessionRpcClient;
}

export function createAppServerSessionClient({
  appServerClient = new AppServerClient(),
}: AppServerSessionClientDeps = {}) {
  const sessionThreadIds = new Map<string, string>();

  function rememberSessionThreadId(
    sessionId: string,
    threadId: string | undefined,
  ): void {
    const normalizedSessionId = sessionId.trim();
    const normalizedThreadId = threadId?.trim();
    if (normalizedSessionId && normalizedThreadId) {
      sessionThreadIds.set(normalizedSessionId, normalizedThreadId);
    }
  }

  async function createAgentRuntimeSession(
    workspaceId?: string,
    name?: string,
    executionStrategy?: AgentExecutionStrategy,
    options?: AgentRuntimeCreateSessionOptions,
  ): Promise<string> {
    const sessionScope = normalizeCreateSessionScope(workspaceId, options);
    const normalizedName = name?.trim() || "新对话";
    const route = readThreadStartRoute(options?.metadata);
    const response = await appServerClient.startSession({
      cwd: sessionScope.workingDir,
      ...(route
        ? { model: route.model, modelProvider: route.modelProvider }
        : {}),
      serviceName: normalizedName,
      threadSource: "appServer",
      historyMode: "paginated",
    });
    const thread = readCanonicalThreadFromResult(response.result);
    if (
      !thread ||
      !readStringField(thread, "id") ||
      !readStringField(thread, "sessionId")
    ) {
      throw new Error("thread/start did not return canonical Thread");
    }
    const sessionId = readStringField(thread, "sessionId");
    if (!sessionId) {
      throw new Error("thread/start returned an empty canonical sessionId");
    }
    return sessionId;
  }

  async function listAgentRuntimeSessions(
    options?: AgentRuntimeListSessionsOptions,
  ): Promise<AgentSessionInfo[]> {
    const sessions = await listCanonicalSessionOverviews(
      appServerClient,
      options,
    );
    if (!sessions) {
      throw new Error("thread/list did not return session list");
    }
    for (const session of sessions) {
      rememberSessionThreadId(session.sessionId, session.threadId);
    }
    return sessions.map(appServerSessionOverviewToRuntimeInfo);
  }

  async function getAgentRuntimeSession(
    sessionId: string,
    options?: AgentRuntimeGetSessionOptions,
  ): Promise<AgentSessionDetail> {
    const normalizedSessionId = sessionId.trim();
    if (!normalizedSessionId) {
      throw new Error("sessionId is required to read App Server session");
    }

    let threadId =
      sessionThreadIds.get(normalizedSessionId) ?? normalizedSessionId;
    let readResult: unknown;
    try {
      readResult = (
        await appServerClient.readThread(
          appServerThreadReadParams(threadId, false),
        )
      ).result;
    } catch (directReadError) {
      const matchingThreadId = await findCanonicalThreadIdBySessionId(
        appServerClient,
        normalizedSessionId,
      );
      if (!matchingThreadId) {
        throw directReadError;
      }
      threadId = matchingThreadId;
      readResult = (
        await appServerClient.readThread(
          appServerThreadReadParams(threadId, false),
        )
      ).result;
    }

    const canonicalThread = readCanonicalThreadFromResult(readResult);
    if (canonicalThread) {
      assertCanonicalThreadIdentity(normalizedSessionId, canonicalThread);
      rememberSessionThreadId(
        readStringField(canonicalThread, "sessionId") || normalizedSessionId,
        readStringField(canonicalThread, "id") || threadId,
      );
    }
    const historyWindow = canonicalThread
      ? await readCanonicalThreadHistoryWindow(
          appServerClient,
          canonicalThread,
          options,
        )
      : null;
    if (historyWindow) {
      readResult = { thread: historyWindow.thread };
    }

    const canonicalDetail = readCanonicalThreadDetail(readResult);
    if (!canonicalDetail) {
      throw new Error("thread/read did not return canonical session detail");
    }
    if (!historyWindow) {
      return canonicalDetail;
    }
    const detailWithHistory: AgentSessionDetail = {
      ...canonicalDetail,
      history_limit: historyWindow.historyLimit,
      history_cursor: historyWindow.historyCursor,
      history_truncated: historyWindow.historyTruncated,
    };
    return detailWithHistory;
  }

  async function updateAgentRuntimeThreadToolPreferences(
    sessionId: string,
    preferences: AgentSessionExecutionRuntimePreferences,
  ): Promise<void> {
    const threadId = await resolveCanonicalThreadId(
      appServerClient,
      sessionId,
      "thread/settings/update",
    );
    await appServerClient.updateThreadSettings({
      threadId,
      toolPreferences: preferences,
    });
  }

  async function archiveAgentRuntimeSession(sessionId: string): Promise<void> {
    const threadId = await resolveCanonicalThreadId(
      appServerClient,
      sessionId,
      "thread/archive",
    );
    await appServerClient.archiveThread({ threadId });
  }

  async function forkAgentRuntimeSession(sessionId: string): Promise<string> {
    const threadId = await resolveCanonicalThreadId(
      appServerClient,
      sessionId,
      "thread/fork",
    );
    const response = await appServerClient.forkThread({ threadId });
    const thread = readCanonicalThreadFromResult(response.result);
    if (!thread) {
      throw new Error("thread/fork did not return canonical Thread");
    }
    const forkedThreadId = readStringField(thread, "id");
    const forkedSessionId = readStringField(thread, "sessionId");
    if (!forkedThreadId || !forkedSessionId) {
      throw new Error("thread/fork returned an incomplete canonical Thread");
    }
    rememberSessionThreadId(forkedSessionId, forkedThreadId);
    return forkedSessionId;
  }

  async function unarchiveAgentRuntimeSession(
    sessionId: string,
  ): Promise<void> {
    const threadId = await resolveCanonicalThreadId(
      appServerClient,
      sessionId,
      "thread/unarchive",
    );
    const response = await appServerClient.unarchiveThread({ threadId });
    const thread = readCanonicalThreadFromResult(response.result);
    const restoredThreadId = thread
      ? readStringField(thread, "id") || readStringField(thread, "threadId")
      : "";
    if (restoredThreadId !== threadId) {
      throw new Error("thread/unarchive did not return the restored thread");
    }
  }

  async function deleteAgentRuntimeSession(sessionId: string): Promise<void> {
    const threadId = await resolveCanonicalThreadId(
      appServerClient,
      sessionId,
      "thread/delete",
    );
    await appServerClient.deleteThread({ threadId });
  }

  return {
    archiveAgentRuntimeSession,
    createAgentRuntimeSession,
    deleteAgentRuntimeSession,
    forkAgentRuntimeSession,
    getAgentRuntimeSession,
    listAgentRuntimeSessions,
    unarchiveAgentRuntimeSession,
    updateAgentRuntimeThreadToolPreferences,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function readField(
  record: Record<string, unknown>,
  camelKey: string,
  snakeKey?: string,
): unknown {
  if (Object.prototype.hasOwnProperty.call(record, camelKey)) {
    return record[camelKey];
  }
  return snakeKey ? record[snakeKey] : undefined;
}

function readStringField(
  record: Record<string, unknown>,
  camelKey: string,
  snakeKey?: string,
): string {
  const value = readField(record, camelKey, snakeKey);
  return typeof value === "string" ? value : "";
}

function assertCanonicalThreadIdentity(
  requestedId: string,
  thread: Record<string, unknown>,
): void {
  const threadId = readStringField(thread, "id").trim();
  const sessionId = readStringField(thread, "sessionId").trim();
  if (requestedId !== threadId && requestedId !== sessionId) {
    throw new Error("thread/read canonical identity mismatch");
  }
}

function readOptionalStringField(
  record: Record<string, unknown>,
  camelKey: string,
  snakeKey?: string,
): string | undefined {
  const value = readField(record, camelKey, snakeKey);
  return typeof value === "string" ? value : undefined;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function normalizeCreateSessionScope(
  workspaceId: string | undefined,
  options?: AgentRuntimeCreateSessionOptions,
): { workspaceId?: string; workingDir?: string; scopeId: string } {
  const normalizedWorkingDir = normalizeCwd(options?.workingDir ?? undefined);
  const normalizedWorkspaceId = workspaceId?.trim() || undefined;
  if (normalizedWorkingDir) {
    return {
      workspaceId: normalizedWorkspaceId,
      workingDir: normalizedWorkingDir,
      scopeId: normalizedWorkingDir,
    };
  }
  if (normalizedWorkspaceId) {
    return {
      workspaceId: normalizedWorkspaceId,
      scopeId: normalizedWorkspaceId,
    };
  }
  return {
    scopeId: "detached",
  };
}

function readThreadStartRoute(
  metadata: Record<string, unknown> | undefined,
): { model: string; modelProvider: string } | null {
  const source = metadata ?? {};
  const modelProvider = readOptionalStringField(source, "providerSelector");
  const model = readOptionalStringField(source, "modelName");
  return modelProvider && model ? { model, modelProvider } : null;
}

async function listCanonicalSessionOverviews(
  client: AppServerSessionRpcClient,
  options?: AgentRuntimeListSessionsOptions,
): Promise<AppServerAgentSessionOverview[] | null> {
  const requestedLimit = normalizeListLimit(options?.limit);
  if (requestedLimit === 0) {
    return [];
  }

  const workspaceId = options?.workspaceId?.trim() || undefined;
  const cwd = normalizeCwdFilter(options?.cwd);
  const sessions: AppServerAgentSessionOverview[] = [];
  const pageLimit = requestedLimit ?? THREAD_LIST_PAGE_LIMIT;

  const sectionIds = await listCanonicalSectionIds(client);
  const sectionScopes: Array<string | null> = [...sectionIds, null];

  for (const archived of archivedFilters(options)) {
    for (const sectionId of sectionScopes) {
      const seenCursors = new Set<string>();
      let cursor: string | undefined;
      do {
        const response = await client.request<AppServerThreadListResponse>(
          METHOD_THREAD_LIST,
          appServerThreadListParams({
            archived,
            cursor,
            cwd,
            limit: pageLimit,
            sectionId,
          }),
        );
        const page = readCanonicalThreadListResponse(response.result, {
          archived,
        });
        if (!page) {
          return null;
        }
        sessions.push(
          ...page.filter(
            (session) =>
              matchesWorkspace(session, workspaceId) &&
              matchesCwd(session, cwd),
          ),
        );

        const collapsed = collapseCanonicalSessionOverviews(sessions);
        if (
          requestedLimit !== undefined &&
          collapsed.length >= requestedLimit
        ) {
          break;
        }
        const nextCursor = response.result.nextCursor?.trim() || undefined;
        if (!nextCursor || seenCursors.has(nextCursor)) {
          break;
        }
        seenCursors.add(nextCursor);
        cursor = nextCursor;
      } while (cursor);

      const collapsed = collapseCanonicalSessionOverviews(sessions);
      if (requestedLimit !== undefined && collapsed.length >= requestedLimit) {
        break;
      }
    }
    const collapsed = collapseCanonicalSessionOverviews(sessions);
    if (requestedLimit !== undefined && collapsed.length >= requestedLimit) {
      break;
    }
  }

  const collapsed = collapseCanonicalSessionOverviews(sessions);
  return requestedLimit === undefined
    ? collapsed
    : collapsed.slice(0, requestedLimit);
}

async function findCanonicalThreadIdBySessionId(
  client: AppServerSessionRpcClient,
  sessionId: string,
): Promise<string | undefined> {
  let childFallback: AppServerAgentSessionOverview | undefined;
  for (const archived of [false, true]) {
    const seenCursors = new Set<string>();
    let cursor: string | undefined;
    do {
      const response = await client.request<AppServerThreadListResponse>(
        METHOD_THREAD_LIST,
        appServerThreadListParams({
          archived,
          cursor,
          limit: THREAD_LIST_PAGE_LIMIT,
        }),
      );
      const page = readCanonicalThreadListResponse(response.result, {
        archived,
      });
      if (!page) {
        throw new Error("thread/list did not return session list");
      }
      const exactThread = page.find((thread) => thread.threadId === sessionId);
      if (exactThread?.threadId) {
        return exactThread.threadId;
      }
      const root = page.find(
        (thread) => thread.sessionId === sessionId && !thread.parentThreadId,
      );
      if (root?.threadId) {
        return root.threadId;
      }
      childFallback ??= page.find((thread) => thread.sessionId === sessionId);

      const nextCursor = response.result.nextCursor?.trim() || undefined;
      if (!nextCursor || seenCursors.has(nextCursor)) {
        break;
      }
      seenCursors.add(nextCursor);
      cursor = nextCursor;
    } while (cursor);
  }
  return childFallback?.threadId;
}

async function resolveCanonicalThreadId(
  client: AppServerSessionRpcClient,
  sessionId: string,
  method:
    | "thread/archive"
    | "thread/delete"
    | "thread/settings/update"
    | "thread/fork"
    | "thread/unarchive",
): Promise<string> {
  const normalizedSessionId = sessionId.trim();
  if (!normalizedSessionId) {
    throw new Error(`sessionId is required for ${method}`);
  }
  const threadId = await findCanonicalThreadIdBySessionId(
    client,
    normalizedSessionId,
  );
  if (!threadId) {
    throw new Error(`${method} could not resolve canonical thread`);
  }
  return threadId;
}

function appServerThreadListParams({
  archived,
  cursor,
  cwd,
  limit,
  sectionId,
}: {
  archived: boolean;
  cursor?: string;
  cwd?: string | string[];
  limit: number;
  sectionId?: string | null;
}): AppServerThreadListParams {
  return omitUndefined({
    archived,
    cursor,
    cwd,
    limit,
    sectionId,
    sortKey:
      typeof sectionId === "string" && sectionId.trim()
        ? ("section_position" as const)
        : undefined,
  });
}

async function listCanonicalSectionIds(
  client: AppServerSessionRpcClient,
): Promise<string[]> {
  const sectionIds: string[] = [];
  const seenCursors = new Set<string>();
  let cursor: string | undefined;
  do {
    const response = await client.listThreadSections({
      ...(cursor ? { cursor } : {}),
      limit: THREAD_LIST_PAGE_LIMIT,
    });
    const page = response.result as AppServerThreadSectionListResponse;
    if (!page || !Array.isArray(page.data)) {
      throw new Error("threadSection/list did not return section list");
    }
    for (const section of page.data) {
      if (section.id.trim()) {
        sectionIds.push(section.id);
      }
    }
    const nextCursor = page.nextCursor?.trim() || undefined;
    if (!nextCursor || seenCursors.has(nextCursor)) {
      break;
    }
    seenCursors.add(nextCursor);
    cursor = nextCursor;
  } while (cursor);
  return sectionIds;
}

function archivedFilters(
  options?: AgentRuntimeListSessionsOptions,
): readonly boolean[] {
  if (options?.archivedOnly === true) {
    return [true];
  }
  return options?.includeArchived === true ? [false, true] : [false];
}

function normalizeListLimit(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? Math.trunc(value)
    : undefined;
}

function matchesWorkspace(
  session: AppServerAgentSessionOverview,
  workspaceId: string | undefined,
): boolean {
  return workspaceId === undefined || session.workspaceId === workspaceId;
}

function matchesCwd(
  session: AppServerAgentSessionOverview,
  cwd: string | string[] | undefined,
): boolean {
  if (cwd === undefined) {
    return true;
  }
  const accepted = Array.isArray(cwd) ? cwd : [cwd];
  return Boolean(
    session.workingDir &&
    accepted.includes(normalizeCwd(session.workingDir) ?? ""),
  );
}

function collapseCanonicalSessionOverviews(
  sessions: AppServerAgentSessionOverview[],
): AppServerAgentSessionOverview[] {
  const bySessionId = new Map<string, AppServerAgentSessionOverview>();
  for (const session of sessions) {
    const current = bySessionId.get(session.sessionId);
    if (!current || preferCanonicalSessionRoot(session, current)) {
      bySessionId.set(session.sessionId, session);
    }
  }
  return [...bySessionId.values()];
}

function preferCanonicalSessionRoot(
  candidate: AppServerAgentSessionOverview,
  current: AppServerAgentSessionOverview,
): boolean {
  const candidateIsRoot = !candidate.parentThreadId;
  const currentIsRoot = !current.parentThreadId;
  if (candidateIsRoot !== currentIsRoot) {
    return candidateIsRoot;
  }
  const candidateUpdatedAt = timestampMillis(candidate.updatedAt);
  const currentUpdatedAt = timestampMillis(current.updatedAt);
  if (candidateUpdatedAt !== currentUpdatedAt) {
    return candidateUpdatedAt > currentUpdatedAt;
  }
  return (candidate.threadId ?? "") < (current.threadId ?? "");
}

function normalizeCwdFilter(cwd: string | string[] | undefined) {
  if (Array.isArray(cwd)) {
    const normalized = cwd
      .map((value) => normalizeCwd(value))
      .filter((value): value is string => Boolean(value));
    return normalized.length > 0 ? normalized : undefined;
  }
  return normalizeCwd(cwd);
}

function normalizeCwd(cwd: string | undefined) {
  const value = cwd?.trim();
  if (!value) {
    return undefined;
  }
  const trimmed = value.replace(/[\\/]+$/u, "");
  return trimmed || value;
}

function appServerThreadReadParams(
  threadId: string,
  includeTurns: boolean,
): AppServerThreadReadParams {
  const normalizedThreadId = threadId.trim();
  if (!normalizedThreadId) {
    throw new Error("sessionId is required to read App Server session");
  }

  return {
    threadId: normalizedThreadId,
    includeTurns,
  };
}

function readCanonicalThreadFromResult(
  value: unknown,
): Record<string, unknown> | null {
  return isRecord(value) && isRecord(value.thread) ? value.thread : null;
}

function appServerSessionOverviewToRuntimeInfo(
  session: AppServerAgentSessionOverview,
): AgentSessionInfo {
  return omitUndefined({
    id: session.sessionId,
    thread_id: session.threadId ?? session.sessionId,
    name: session.title,
    created_at: timestampMillis(session.createdAt),
    updated_at: timestampMillis(session.updatedAt),
    archived_at: session.archivedAt
      ? timestampMillis(session.archivedAt)
      : session.archivedAt === null
        ? null
        : undefined,
    model: session.model,
    messages_count: session.messagesCount,
    execution_strategy: executionStrategyFromProtocol(
      session.executionStrategy,
    ),
    session_business_object_ref_metadata: isPlainObject(
      session.businessObjectRefMetadata,
    )
      ? session.businessObjectRefMetadata
      : undefined,
    workspace_id: session.workspaceId,
    working_dir: session.workingDir,
    thread_status: session.threadStatus,
    latest_turn_status: session.latestTurnStatus,
    active_turn_id: session.activeTurnId,
    queued_turn_count: session.queuedTurnCount,
    section: session.section,
    section_entered_at:
      session.sectionEnteredAt === undefined
        ? undefined
        : session.sectionEnteredAt,
  });
}

function timestampMillis(value: string | undefined): number {
  if (!value) {
    return Date.now();
  }
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : Date.now();
}

function executionStrategyFromProtocol(
  value: unknown,
): AgentExecutionStrategy | undefined {
  return value === "react" ? "react" : undefined;
}

function omitUndefined<T extends Record<string, unknown>>(value: T): T {
  return Object.fromEntries(
    Object.entries(value).filter(([, entry]) => entry !== undefined),
  ) as T;
}

export type AppServerSessionClient = ReturnType<
  typeof createAppServerSessionClient
>;
