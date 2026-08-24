import { logAgentDebug } from "@/lib/agentDebug";
import { recordAgentUiPerformanceMetric } from "@/lib/agentUiPerformanceMetrics";
import { normalizeLegacyThreadItem } from "../agentTextNormalization";
import type {
  AgentExecutionStrategy,
  AgentSessionExecutionRuntimePreferences,
} from "../agentExecutionRuntime";
import type { AgentThreadItem } from "../agentProtocol";
import {
  createAppServerSessionClient,
  type AppServerSessionClient,
  type AppServerSessionRpcClient,
} from "./appServerSessionClient";
import { normalizeThreadReadModel } from "./normalizers";
import type {
  AgentRuntimeCreateSessionOptions,
  AgentRuntimeGetSessionOptions,
} from "./requestTypes";
import type {
  AgentSessionDetail,
  AgentSessionInfo,
  AgentRuntimeListSessionsOptions,
} from "./sessionTypes";

function isTransientSessionReadError(error: unknown): boolean {
  const message =
    error instanceof Error ? error.message : String(error || "Unknown error");
  const normalizedMessage = message.toLowerCase();

  return (
    message.includes("Failed to fetch") ||
    message.includes("NetworkError") ||
    message.includes("ERR_CONNECTION_REFUSED") ||
    message.includes("Load failed") ||
    message.includes("ECONNREFUSED") ||
    normalizedMessage.includes("timeout") ||
    normalizedMessage.includes("aborterror")
  );
}

function normalizeOpaqueHistoryCursor(
  options: AgentRuntimeGetSessionOptions | undefined,
  key: "historyItemCursor" | "historyTurnCursor",
): string | null | undefined {
  if (!options || !(key in options)) {
    return undefined;
  }
  const value = options[key];
  return typeof value === "string" && value.length > 0 ? value : null;
}

function omitUndefinedSessionOptionalFields(
  detail: AgentSessionDetail,
): AgentSessionDetail {
  if (detail.execution_runtime === undefined) {
    delete detail.execution_runtime;
  }
  return detail;
}

export interface AgentRuntimeSessionClientDeps {
  appServerClient?: AppServerSessionRpcClient;
  appServerSessionClient?: AppServerSessionClient;
}

const defaultAppServerSessionClient = createAppServerSessionClient();

export const AGENT_RUNTIME_SESSIONS_CHANGED_EVENT =
  "lime:agent-runtime-sessions-changed";

export interface AgentRuntimeSessionsChangedDetail {
  reason:
    | "created"
    | "updated"
    | "archived"
    | "unarchived"
    | "deleted"
    | "external";
  sessionId?: string;
  workspaceId?: string;
}

export function notifyAgentRuntimeSessionsChanged(
  detail: AgentRuntimeSessionsChangedDetail,
): void {
  if (typeof window === "undefined") {
    return;
  }

  window.dispatchEvent(
    new CustomEvent(AGENT_RUNTIME_SESSIONS_CHANGED_EVENT, { detail }),
  );
}

export function createSessionClient(deps: AgentRuntimeSessionClientDeps = {}) {
  const appServerSessionClient =
    deps.appServerSessionClient ??
    (deps.appServerClient
      ? createAppServerSessionClient({ appServerClient: deps.appServerClient })
      : defaultAppServerSessionClient);
  async function createAgentRuntimeSession(
    workspaceId?: string,
    name?: string,
    executionStrategy?: AgentExecutionStrategy,
    options?: AgentRuntimeCreateSessionOptions,
  ): Promise<string> {
    const normalizedWorkspaceId = workspaceId?.trim() || undefined;
    const sessionId = await appServerSessionClient.createAgentRuntimeSession(
      normalizedWorkspaceId,
      name,
      executionStrategy,
      options,
    );
    notifyAgentRuntimeSessionsChanged({
      reason: "created",
      sessionId,
      workspaceId: normalizedWorkspaceId,
    });
    return sessionId;
  }

  async function listAgentRuntimeSessions(
    options?: AgentRuntimeListSessionsOptions,
  ): Promise<AgentSessionInfo[]> {
    const startedAt = Date.now();
    let settled = false;
    const includeArchived = options?.includeArchived === true;
    const archivedOnly = options?.archivedOnly === true;
    const cwd = options?.cwd;
    const workspaceId = options?.workspaceId?.trim();
    const limit =
      typeof options?.limit === "number" &&
      Number.isFinite(options.limit) &&
      options.limit >= 0
        ? Math.trunc(options.limit)
        : undefined;
    const slowTimer: number | null =
      typeof window !== "undefined"
        ? window.setTimeout(() => {
            if (settled) {
              return;
            }

            logAgentDebug(
              "AgentApi",
              "runtimeListSessions.slow",
              {
                elapsedMs: Date.now() - startedAt,
              },
              {
                dedupeKey: "runtimeListSessions.slow",
                level: "info",
                throttleMs: 1000,
              },
            );
          }, 1000)
        : null;

    const listMetricContext = {
      archivedOnly,
      cwd: cwd ?? null,
      includeArchived,
      limit: limit ?? null,
      workspaceId: workspaceId ?? null,
    };
    recordAgentUiPerformanceMetric(
      "agentRuntime.listSessions.start",
      listMetricContext,
    );
    logAgentDebug("AgentApi", "runtimeListSessions.start", listMetricContext);

    try {
      const sessions = await appServerSessionClient.listAgentRuntimeSessions({
        includeArchived,
        archivedOnly,
        cwd,
        workspaceId,
        limit,
      });
      settled = true;
      recordAgentUiPerformanceMetric("agentRuntime.listSessions.success", {
        ...listMetricContext,
        durationMs: Date.now() - startedAt,
        sessionsCount: sessions.length,
      });
      logAgentDebug("AgentApi", "runtimeListSessions.success", {
        archivedOnly,
        durationMs: Date.now() - startedAt,
        limit,
        sessionsCount: sessions.length,
        includeArchived,
        cwd: cwd ?? null,
        workspaceId: workspaceId ?? null,
      });
      return sessions;
    } catch (error) {
      settled = true;
      recordAgentUiPerformanceMetric("agentRuntime.listSessions.error", {
        ...listMetricContext,
        durationMs: Date.now() - startedAt,
      });
      logAgentDebug(
        "AgentApi",
        "runtimeListSessions.error",
        {
          archivedOnly,
          durationMs: Date.now() - startedAt,
          error,
          limit,
          cwd: cwd ?? null,
          workspaceId: workspaceId ?? null,
        },
        { level: "warn" },
      );
      throw error;
    } finally {
      if (slowTimer !== null) {
        clearTimeout(slowTimer);
      }
    }
  }

  async function getAgentRuntimeSession(
    sessionId: string,
    options?: AgentRuntimeGetSessionOptions,
  ): Promise<AgentSessionDetail> {
    const startedAt = Date.now();
    let settled = false;
    const resumeSessionStartHooks = options?.resumeSessionStartHooks === true;
    const source = options?.source?.trim() || null;
    const historyLimit =
      typeof options?.historyLimit === "number" &&
      Number.isFinite(options.historyLimit) &&
      options.historyLimit >= 0
        ? Math.trunc(options.historyLimit)
        : undefined;
    const historyItemCursor = normalizeOpaqueHistoryCursor(
      options,
      "historyItemCursor",
    );
    const historyTurnCursor = normalizeOpaqueHistoryCursor(
      options,
      "historyTurnCursor",
    );
    const slowTimer: number | null =
      typeof window !== "undefined"
        ? window.setTimeout(() => {
            if (settled) {
              return;
            }

            logAgentDebug(
              "AgentApi",
              "runtimeGetSession.slow",
              {
                elapsedMs: Date.now() - startedAt,
                historyItemCursor: historyItemCursor ?? null,
                historyLimit: historyLimit ?? null,
                historyTurnCursor: historyTurnCursor ?? null,
                resumeSessionStartHooks,
                sessionId,
                source,
              },
              {
                dedupeKey: `runtimeGetSession.slow:${sessionId}`,
                level: "info",
                throttleMs: 1000,
              },
            );
          }, 1000)
        : null;

    const getSessionMetricContext = {
      historyItemCursor: historyItemCursor ?? null,
      historyLimit: historyLimit ?? null,
      historyTurnCursor: historyTurnCursor ?? null,
      resumeSessionStartHooks,
      sessionId,
      source,
    };
    recordAgentUiPerformanceMetric(
      "agentRuntime.getSession.start",
      getSessionMetricContext,
    );
    logAgentDebug(
      "AgentApi",
      "runtimeGetSession.start",
      getSessionMetricContext,
    );

    try {
      const detail = await appServerSessionClient.getAgentRuntimeSession(
        sessionId,
        {
          ...(resumeSessionStartHooks ? { resumeSessionStartHooks: true } : {}),
          ...(typeof historyLimit === "number" ? { historyLimit } : {}),
          ...(historyItemCursor !== undefined ? { historyItemCursor } : {}),
          ...(historyTurnCursor !== undefined ? { historyTurnCursor } : {}),
        },
      );
      const normalizedDetail = detail as AgentSessionDetail | null | undefined;
      const normalizedSessionDetail: AgentSessionDetail = {
        ...detail,
        messages: Array.isArray(normalizedDetail?.messages)
          ? normalizedDetail.messages
          : [],
        turns: Array.isArray(normalizedDetail?.turns)
          ? normalizedDetail.turns
          : [],
        items: Array.isArray(normalizedDetail?.items)
          ? normalizedDetail.items.map((item) =>
              normalizeLegacyThreadItem(item as AgentThreadItem),
            )
          : [],
        thread_read: normalizeThreadReadModel(normalizedDetail?.thread_read),
        todo_items: Array.isArray(normalizedDetail?.todo_items)
          ? normalizedDetail.todo_items
          : [],
      };
      omitUndefinedSessionOptionalFields(normalizedSessionDetail);
      settled = true;
      recordAgentUiPerformanceMetric("agentRuntime.getSession.success", {
        ...getSessionMetricContext,
        durationMs: Date.now() - startedAt,
        itemsCount: normalizedSessionDetail.items?.length ?? 0,
        messagesCount: normalizedSessionDetail.messages?.length ?? 0,
        turnsCount: normalizedSessionDetail.turns?.length ?? 0,
      });
      logAgentDebug("AgentApi", "runtimeGetSession.success", {
        durationMs: Date.now() - startedAt,
        historyItemCursor: historyItemCursor ?? null,
        historyLimit: historyLimit ?? null,
        historyTurnCursor: historyTurnCursor ?? null,
        itemsCount: normalizedSessionDetail.items?.length ?? 0,
        messagesCount: normalizedSessionDetail.messages?.length ?? 0,
        resumeSessionStartHooks,
        sessionId,
        source,
        turnsCount: normalizedSessionDetail.turns?.length ?? 0,
      });
      return normalizedSessionDetail;
    } catch (error) {
      settled = true;
      recordAgentUiPerformanceMetric("agentRuntime.getSession.error", {
        ...getSessionMetricContext,
        durationMs: Date.now() - startedAt,
      });
      logAgentDebug(
        "AgentApi",
        "runtimeGetSession.error",
        {
          durationMs: Date.now() - startedAt,
          error,
          historyItemCursor: historyItemCursor ?? null,
          historyLimit: historyLimit ?? null,
          historyTurnCursor: historyTurnCursor ?? null,
          resumeSessionStartHooks,
          sessionId,
          source,
        },
        { level: isTransientSessionReadError(error) ? "warn" : "error" },
      );
      throw error;
    } finally {
      if (slowTimer !== null) {
        clearTimeout(slowTimer);
      }
    }
  }

  async function updateAgentRuntimeThreadToolPreferences(
    sessionId: string,
    preferences: AgentSessionExecutionRuntimePreferences,
  ): Promise<void> {
    await appServerSessionClient.updateAgentRuntimeThreadToolPreferences(
      sessionId,
      preferences,
    );
  }

  async function archiveAgentRuntimeSession(sessionId: string): Promise<void> {
    await appServerSessionClient.archiveAgentRuntimeSession(sessionId);
    notifyAgentRuntimeSessionsChanged({
      reason: "archived",
      sessionId: sessionId.trim(),
    });
  }

  async function forkAgentRuntimeSession(sessionId: string): Promise<string> {
    const forkedSessionId =
      await appServerSessionClient.forkAgentRuntimeSession(sessionId);
    notifyAgentRuntimeSessionsChanged({
      reason: "created",
      sessionId: forkedSessionId,
    });
    return forkedSessionId;
  }

  async function unarchiveAgentRuntimeSession(
    sessionId: string,
  ): Promise<void> {
    await appServerSessionClient.unarchiveAgentRuntimeSession(sessionId);
    notifyAgentRuntimeSessionsChanged({
      reason: "unarchived",
      sessionId: sessionId.trim(),
    });
  }

  async function deleteAgentRuntimeSession(sessionId: string): Promise<void> {
    await appServerSessionClient.deleteAgentRuntimeSession(sessionId);
    notifyAgentRuntimeSessionsChanged({
      reason: "deleted",
      sessionId: sessionId.trim(),
    });
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

export const {
  archiveAgentRuntimeSession,
  createAgentRuntimeSession,
  deleteAgentRuntimeSession,
  forkAgentRuntimeSession,
  getAgentRuntimeSession,
  listAgentRuntimeSessions,
  unarchiveAgentRuntimeSession,
  updateAgentRuntimeThreadToolPreferences,
} = createSessionClient();
