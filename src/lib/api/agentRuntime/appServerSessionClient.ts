import {
  AppServerClient,
  type AppServerThreadListParams,
  type AppServerThreadListResponse,
  type AppServerThreadReadParams,
} from "@/lib/api/appServer";
import {
  METHOD_THREAD_LIST,
  METHOD_THREAD_ITEMS_LIST,
  METHOD_THREAD_TURNS_LIST,
  type ThreadItemsListResponse,
  type ThreadTurnsListResponse,
} from "../../../../packages/app-server-client/src/protocol";
import type {
  AgentExecutionStrategy,
  AgentSessionExecutionRuntimePreferences,
} from "../agentExecutionRuntime";
import {
  readCanonicalThreadDetail,
  readCanonicalThreadListResponse,
} from "./appServerCanonicalThreadProjection";
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
  | "readThread"
  | "updateThreadSettings"
  | "archiveThread"
  | "unarchiveThread"
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
};

export interface AppServerSessionClientDeps {
  appServerClient?: AppServerSessionRpcClient;
}

export function createAppServerSessionClient({
  appServerClient = new AppServerClient(),
}: AppServerSessionClientDeps = {}) {
  async function createAgentRuntimeSession(
    workspaceId?: string,
    name?: string,
    executionStrategy?: AgentExecutionStrategy,
    options?: AgentRuntimeCreateSessionOptions,
  ): Promise<string> {
    const sessionScope = normalizeCreateSessionScope(workspaceId, options);
    const normalizedName = name?.trim() || "新对话";
    const route = readThreadStartRoute(options?.metadata);
    if (!route) {
      throw new Error(
        "thread/start requires current providerSelector and modelName",
      );
    }
    const response = await appServerClient.startSession({
      cwd: sessionScope.workingDir,
      model: route.model,
      modelProvider: route.modelProvider,
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
    return sessions.map(appServerSessionOverviewToRuntimeInfo);
  }

  async function getAgentRuntimeSession(
    sessionId: string,
    _options?: AgentRuntimeGetSessionOptions,
  ): Promise<AgentSessionDetail> {
    const normalizedSessionId = sessionId.trim();
    if (!normalizedSessionId) {
      throw new Error("sessionId is required to read App Server session");
    }

    let threadId = normalizedSessionId;
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
      if (readStringField(canonicalThread, "historyMode") === "paginated") {
        readResult = await readPaginatedCanonicalThread(
          appServerClient,
          canonicalThread,
        );
      } else if (
        !Array.isArray(canonicalThread.turns) ||
        canonicalThread.turns.length === 0
      ) {
        readResult = (
          await appServerClient.readThread(
            appServerThreadReadParams(threadId, true),
          )
        ).result;
      }
    }

    const canonicalDetail = readCanonicalThreadDetail(readResult);
    if (!canonicalDetail) {
      throw new Error("thread/read did not return canonical session detail");
    }
    return canonicalDetail;
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

  for (const archived of archivedFilters(options)) {
    const seenCursors = new Set<string>();
    let cursor: string | undefined;
    do {
      const response = await client.request<AppServerThreadListResponse>(
        METHOD_THREAD_LIST,
        appServerThreadListParams({ archived, cursor, cwd, limit: pageLimit }),
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
            matchesWorkspace(session, workspaceId) && matchesCwd(session, cwd),
        ),
      );

      const collapsed = collapseCanonicalSessionOverviews(sessions);
      if (requestedLimit !== undefined && collapsed.length >= requestedLimit) {
        break;
      }
      const nextCursor = response.result.nextCursor?.trim() || undefined;
      if (!nextCursor || seenCursors.has(nextCursor)) {
        break;
      }
      seenCursors.add(nextCursor);
      cursor = nextCursor;
    } while (cursor);
  }

  const collapsed = collapseCanonicalSessionOverviews(sessions).sort(
    (left, right) =>
      timestampMillis(right.updatedAt) - timestampMillis(left.updatedAt),
  );
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
}: {
  archived: boolean;
  cursor?: string;
  cwd?: string | string[];
  limit: number;
}): AppServerThreadListParams {
  return omitUndefined({ archived, cursor, cwd, limit });
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

async function readPaginatedCanonicalThread(
  client: AppServerSessionRpcClient,
  thread: Record<string, unknown>,
): Promise<{ thread: Record<string, unknown> }> {
  const threadId =
    readStringField(thread, "threadId") || readStringField(thread, "id");
  if (!threadId) {
    throw new Error("thread/read returned an empty canonical thread id");
  }

  const turns: Record<string, unknown>[] = [];
  const turnCursors = new Set<string>();
  let turnCursor: string | undefined;
  do {
    const response = await client.request<ThreadTurnsListResponse>(
      METHOD_THREAD_TURNS_LIST,
      omitUndefined({
        threadId,
        cursor: turnCursor,
        limit: THREAD_LIST_PAGE_LIMIT,
        sortDirection: "asc",
        itemsView: "summary",
      }),
    );
    if (!isRecord(response.result) || !Array.isArray(response.result.data)) {
      throw new Error("thread/turns/list did not return turn page");
    }
    for (const turn of response.result.data) {
      if (isRecord(turn)) {
        turns.push(turn);
      }
    }
    const nextCursor = readNextCursor(response.result);
    if (!nextCursor || turnCursors.has(nextCursor)) {
      break;
    }
    turnCursors.add(nextCursor);
    turnCursor = nextCursor;
  } while (turnCursor);

  const itemsByTurnId = new Map<string, unknown[]>();
  const itemCursors = new Set<string>();
  let itemCursor: string | undefined;
  do {
    const response = await client.request<ThreadItemsListResponse>(
      METHOD_THREAD_ITEMS_LIST,
      omitUndefined({
        threadId,
        cursor: itemCursor,
        limit: THREAD_LIST_PAGE_LIMIT,
        sortDirection: "asc",
      }),
    );
    if (!isRecord(response.result) || !Array.isArray(response.result.data)) {
      throw new Error("thread/items/list did not return item page");
    }
    for (const entry of response.result.data) {
      if (!isRecord(entry)) {
        continue;
      }
      const turnId = readStringField(entry, "turnId");
      const item = entry.item;
      if (!turnId || !isRecord(item)) {
        continue;
      }
      const items = itemsByTurnId.get(turnId) ?? [];
      items.push(item);
      itemsByTurnId.set(turnId, items);
    }
    const nextCursor = readNextCursor(response.result);
    if (!nextCursor || itemCursors.has(nextCursor)) {
      break;
    }
    itemCursors.add(nextCursor);
    itemCursor = nextCursor;
  } while (itemCursor);

  return {
    thread: {
      ...thread,
      turns: turns.map((turn) => ({
        ...turn,
        items: itemsByTurnId.get(canonicalTurnId(turn)) ?? [],
      })),
    },
  };
}

function readNextCursor(value: Record<string, unknown>): string | undefined {
  const cursor = value.nextCursor;
  return typeof cursor === "string" && cursor.trim()
    ? cursor.trim()
    : undefined;
}

function canonicalTurnId(value: Record<string, unknown>): string {
  return readStringField(value, "turnId") || readStringField(value, "id");
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
