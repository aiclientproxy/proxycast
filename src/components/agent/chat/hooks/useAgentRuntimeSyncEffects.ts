import { useCallback, useEffect, useRef, type MutableRefObject } from "react";
import { isRuntimeSettledStatusValue } from "@limecloud/agent-ui-contracts";
import { isAppServerBridgeAvailable } from "@/lib/api/appServerBridgeAvailability";
import { hasDevBridgeEventListenerCapability } from "@/lib/api/bridgeEvents";
import { parseAgentEvent } from "@/lib/api/agentProtocol";
import { hasDesktopHostEventListenerCapability } from "@/lib/desktop-runtime";
import type { AgentThreadTurn } from "../types";
import type { AgentRuntimeAdapter } from "./agentRuntimeAdapter";
import type { AgentSessionDetailRefreshRequest } from "./agentSessionRefresh";
import { hasRunningThreadReadActivity } from "../projection/threadReadActivity";

const APP_SERVER_BRIDGE_RUNTIME_POLL_MS = 1000;
const APP_SERVER_RUNTIME_EVENT_BIND_GRACE_MS = 250;
const RECOVERED_RUNTIME_POLL_ACTIVE_WINDOW_MS = 30 * 60 * 1000;
const RUNTIME_DETAIL_REFRESH_COALESCE_MS = 120;

const RUNTIME_SYNC_REFRESH_REQUESTS = {
  sendSettled: {
    source: "runtimeSync.sendSettled",
    detailMergeMode: "runtime_sync",
  },
  recoveredPoll: {
    source: "runtimeSync.recoveredPoll",
    detailMergeMode: "runtime_sync",
  },
  poll: {
    source: "runtimeSync.poll",
    detailMergeMode: "runtime_sync",
  },
  event: {
    source: "runtimeSync.event",
    detailMergeMode: "runtime_sync",
  },
  terminalEvent: {
    source: "runtimeSync.event",
    detailMergeMode: "terminal_reconcile",
  },
} satisfies Record<string, AgentSessionDetailRefreshRequest>;

function parseTimestampMs(value: string | null | undefined): number | null {
  if (!value) {
    return null;
  }
  const timestampMs = Date.parse(value);
  return Number.isFinite(timestampMs) ? timestampMs : null;
}

function readRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function readArray(record: Record<string, unknown>, keys: string[]): unknown[] {
  for (const key of keys) {
    const value = record[key];
    if (Array.isArray(value)) {
      return value;
    }
  }
  return [];
}

function readString(
  record: Record<string, unknown>,
  keys: string[],
): string | null {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return null;
}

function readNestedRecord(
  record: Record<string, unknown>,
  keys: string[],
): Record<string, unknown> | null {
  for (const key of keys) {
    const value = readRecord(record[key]);
    if (value) {
      return value;
    }
  }
  return null;
}

function normalizeRuntimeStatus(value: unknown): string | null {
  return typeof value === "string"
    ? value
        .trim()
        .toLowerCase()
        .replace(/[\s-]+/g, "_") || null
    : null;
}

function isTerminalStatusLike(status: unknown): boolean {
  const normalizedStatus = normalizeRuntimeStatus(status);
  return (
    normalizedStatus === "done" ||
    isExplicitTerminalRuntimeStatus(normalizedStatus)
  );
}

function hasTerminalStatusInRecord(
  record: Record<string, unknown>,
  keys: string[],
): boolean {
  return keys.some((key) => isTerminalStatusLike(record[key]));
}

function readTurnIdFromRecord(record: Record<string, unknown>): string | null {
  return readString(record, ["turn_id", "turnId", "id"]);
}

function isTerminalTurnRecord(turn: unknown): boolean {
  const record = readRecord(turn);
  if (!record) {
    return false;
  }
  return hasTerminalStatusInRecord(record, [
    "status",
    "profile_status",
    "profileStatus",
    "native_status",
    "nativeStatus",
  ]);
}

interface ThreadReadTerminalInfo {
  activeTurnId: string | null;
  activeTurnMatchesCurrent: boolean;
  hasAuthoritativeTerminal: boolean;
  hasCurrentTurnTerminal: boolean;
  hasTerminal: boolean;
}

function readThreadReadTerminalInfo(
  threadRead: unknown,
  currentTurnId?: string | null,
): ThreadReadTerminalInfo {
  const record = readRecord(threadRead);
  if (!record) {
    return {
      activeTurnId: null,
      activeTurnMatchesCurrent: false,
      hasAuthoritativeTerminal: false,
      hasCurrentTurnTerminal: false,
      hasTerminal: false,
    };
  }

  const normalizedCurrentTurnId = currentTurnId?.trim() || null;
  const threadStatuses = [
    normalizeRuntimeStatus(record.status),
    normalizeRuntimeStatus(record.profile_status),
    normalizeRuntimeStatus(record.profileStatus),
  ].filter((status): status is string => Boolean(status));
  const hasTerminalThreadStatus = threadStatuses.some(isTerminalStatusLike);
  const turns = readArray(record, ["turns"]);
  const activeTurnId = readString(record, ["active_turn_id", "activeTurnId"]);
  const activeTurn = activeTurnId
    ? turns
        .map(readRecord)
        .find((turn) => turn && readTurnIdFromRecord(turn) === activeTurnId)
    : null;
  const activeTurnMatchesCurrent = normalizedCurrentTurnId
    ? activeTurnId === normalizedCurrentTurnId
    : Boolean(activeTurnId);
  const hasTerminalActiveTurn = Boolean(
    activeTurn && activeTurnMatchesCurrent && isTerminalTurnRecord(activeTurn),
  );
  const hasCurrentTurnTerminal = Boolean(
    normalizedCurrentTurnId &&
    turns.some((turn) => {
      const turnRecord = readRecord(turn);
      return (
        turnRecord &&
        readTurnIdFromRecord(turnRecord) === normalizedCurrentTurnId &&
        isTerminalTurnRecord(turnRecord)
      );
    }),
  );
  const diagnostics = readNestedRecord(record, ["diagnostics"]);
  const hasTerminalLatestTurnStatus =
    hasTerminalStatusInRecord(record, [
      "latest_turn_status",
      "latestTurnStatus",
    ]) ||
    (diagnostics
      ? hasTerminalStatusInRecord(diagnostics, [
          "latest_turn_status",
          "latestTurnStatus",
        ])
      : false);
  const hasTerminalTurn = turns.some(isTerminalTurnRecord);
  const hasAuthoritativeThreadStatus = threadStatuses.some((status) => {
    if (!isTerminalStatusLike(status)) {
      return false;
    }
    return !normalizedCurrentTurnId || activeTurnId === normalizedCurrentTurnId;
  });
  const hasAuthoritativeTerminal =
    hasAuthoritativeThreadStatus ||
    hasTerminalActiveTurn ||
    hasCurrentTurnTerminal;

  return {
    activeTurnId,
    activeTurnMatchesCurrent,
    hasAuthoritativeTerminal,
    hasCurrentTurnTerminal,
    hasTerminal:
      hasAuthoritativeTerminal ||
      hasTerminalThreadStatus ||
      hasTerminalLatestTurnStatus ||
      hasTerminalTurn,
  };
}

function hasRunningThreadReadForRuntimeSync(threadRead: unknown): boolean {
  const record = readRecord(threadRead);
  if (!record) {
    return false;
  }

  const terminalInfo = readThreadReadTerminalInfo(record);
  if (terminalInfo.hasTerminal) {
    return false;
  }

  if (
    hasRunningThreadReadActivity(threadRead, {
      allowThreadStatusWithoutTurn: true,
    })
  ) {
    return true;
  }

  const status =
    normalizeRuntimeStatus(record.status) ??
    normalizeRuntimeStatus(record.profile_status) ??
    normalizeRuntimeStatus(record.profileStatus);
  const activeTurnId = readString(record, ["active_turn_id", "activeTurnId"]);
  return status === "running" && Boolean(activeTurnId);
}

function hasRecentRunningTurn(threadTurns: AgentThreadTurn[]): boolean {
  const nowMs = Date.now();
  return threadTurns.some((turn) => {
    if (turn.status !== "running") {
      return false;
    }

    const timestampMs =
      parseTimestampMs(turn.updated_at) ?? parseTimestampMs(turn.started_at);
    if (timestampMs === null) {
      return false;
    }

    return nowMs - timestampMs <= RECOVERED_RUNTIME_POLL_ACTIVE_WINDOW_MS;
  });
}

function shouldPollRecoveredRuntimeWork(params: {
  threadRead?: unknown;
  threadReadStatus?: string | null;
  queuedTurnCount: number;
  threadTurns: AgentThreadTurn[];
}): boolean {
  if (params.queuedTurnCount > 0) {
    return true;
  }

  const normalizedThreadReadStatus = (
    params.threadReadStatus || ""
  ).toLowerCase();
  if (isExplicitTerminalRuntimeStatus(normalizedThreadReadStatus)) {
    return false;
  }

  if (params.threadRead !== undefined && params.threadRead !== null) {
    if (hasRunningThreadReadForRuntimeSync(params.threadRead)) {
      return true;
    }
    return normalizedThreadReadStatus === "queued";
  }

  const hasRunningTurn = params.threadTurns.some(
    (turn) => turn.status === "running",
  );

  if (hasRunningTurn) {
    return hasRecentRunningTurn(params.threadTurns);
  }

  return (
    normalizedThreadReadStatus === "running" ||
    normalizedThreadReadStatus === "queued"
  );
}

function resolveRefreshRequestForTurnEventPayload(
  payload: unknown,
): AgentSessionDetailRefreshRequest | null {
  const data = parseAgentEvent(payload);
  if (!data) {
    return null;
  }

  switch (data.type) {
    case "error":
    case "turn_completed":
    case "turn_canceled":
    case "turn_failed":
      return RUNTIME_SYNC_REFRESH_REQUESTS.terminalEvent;
    case "action_required":
    case "action_resolved":
    case "artifact_snapshot":
    case "runtime_status":
    case "warning":
      return RUNTIME_SYNC_REFRESH_REQUESTS.event;
    default:
      return null;
  }
}

function preferRuntimeRefreshRequest(
  current: AgentSessionDetailRefreshRequest | null,
  next: AgentSessionDetailRefreshRequest,
): AgentSessionDetailRefreshRequest {
  if (next.detailMergeMode === "terminal_reconcile") {
    return next;
  }
  return current ?? next;
}

function isTerminalRuntimeStatus(status?: string | null): boolean {
  return isRuntimeSettledStatusValue(status);
}

function isExplicitTerminalRuntimeStatus(status?: string | null): boolean {
  const normalizedStatus = (status || "").trim().toLowerCase();
  return (
    normalizedStatus === "completed" ||
    normalizedStatus === "failed" ||
    normalizedStatus === "canceled" ||
    normalizedStatus === "cancelled" ||
    normalizedStatus === "aborted" ||
    (normalizedStatus !== "idle" && isTerminalRuntimeStatus(status))
  );
}

function shouldForceSettleStaleRunningTurn(status?: string | null): boolean {
  return isExplicitTerminalRuntimeStatus(status);
}

function isThreadStatusAuthoritativeForCurrentStream(params: {
  status?: string | null;
  currentTurnId?: string | null;
  threadReadTerminalInfo: ThreadReadTerminalInfo;
}): boolean {
  if (!isExplicitTerminalRuntimeStatus(params.status)) {
    return false;
  }
  if (!params.currentTurnId?.trim()) {
    return true;
  }
  return (
    params.threadReadTerminalInfo.hasCurrentTurnTerminal ||
    params.threadReadTerminalInfo.activeTurnMatchesCurrent
  );
}

function hasRunningTurn(threadTurns: AgentThreadTurn[]): boolean {
  return threadTurns.some((turn) => turn.status === "running");
}

function hasTerminalTurn(
  threadTurns: AgentThreadTurn[],
  currentTurnId?: string | null,
): boolean {
  const normalizedCurrentTurnId = currentTurnId?.trim() || null;
  return threadTurns.some((turn) => {
    if (!isTerminalRuntimeStatus(turn.status)) {
      return false;
    }
    return !normalizedCurrentTurnId || turn.id === normalizedCurrentTurnId;
  });
}

interface UseAgentRuntimeSyncEffectsOptions {
  runtime: Pick<AgentRuntimeAdapter, "listenToTurnEvents">;
  sessionIdRef: MutableRefObject<string | null>;
  sessionId: string | null;
  currentTurnEventName?: string | null;
  currentTurnEventNameRef?: MutableRefObject<string | null>;
  currentStreamTurnId?: string | null;
  isSending: boolean;
  threadRead?: unknown;
  threadReadStatus?: string | null;
  queuedTurnCount: number;
  threadTurns: AgentThreadTurn[];
  refreshSessionDetail: (
    targetSessionId?: string,
    request?: AgentSessionDetailRefreshRequest,
  ) => Promise<unknown>;
  refreshSessionReadModel: (targetSessionId?: string) => Promise<unknown>;
  settleActiveRuntimeStream?: (targetSessionId: string) => void;
}

export function useAgentRuntimeSyncEffects(
  options: UseAgentRuntimeSyncEffectsOptions,
) {
  const {
    runtime,
    sessionIdRef,
    sessionId,
    currentTurnEventName,
    currentTurnEventNameRef,
    currentStreamTurnId,
    isSending,
    threadRead,
    threadReadStatus,
    queuedTurnCount,
    threadTurns,
    refreshSessionDetail,
    settleActiveRuntimeStream,
  } = options;
  const normalizedCurrentTurnEventName = currentTurnEventName?.trim() || null;
  const normalizedCurrentStreamTurnId = currentStreamTurnId?.trim() || null;
  const lastIsSendingRef = useRef(isSending);
  const lastCurrentTurnEventNameRef = useRef(normalizedCurrentTurnEventName);
  const observedActiveRuntimeWorkRef = useRef(false);
  const refreshInFlightSessionRef = useRef<string | null>(null);
  const refreshTimerRef = useRef<number | null>(null);
  const deferredRuntimeRefreshRequestRef =
    useRef<AgentSessionDetailRefreshRequest | null>(null);
  const hasDesktopRuntimeEventListenerCapability =
    hasDesktopHostEventListenerCapability();
  const hasRuntimeEventListenerCapability =
    hasDesktopRuntimeEventListenerCapability ||
    hasDevBridgeEventListenerCapability();
  const cannotSubscribeCurrentTurnRuntimeEvent =
    !hasRuntimeEventListenerCapability || !normalizedCurrentTurnEventName;
  const hasActiveRuntimeWork = shouldPollRecoveredRuntimeWork({
    threadRead,
    threadReadStatus,
    queuedTurnCount,
    threadTurns,
  });
  const shouldUseAppServerBridgeRuntimePolling =
    Boolean(sessionId) &&
    isSending &&
    !normalizedCurrentTurnEventName &&
    cannotSubscribeCurrentTurnRuntimeEvent &&
    isAppServerBridgeAvailable();
  const refreshSessionDetailOnce = useCallback(
    (targetSessionId: string, request: AgentSessionDetailRefreshRequest) => {
      if (refreshInFlightSessionRef.current === targetSessionId) {
        return;
      }

      refreshInFlightSessionRef.current = targetSessionId;
      void refreshSessionDetail(targetSessionId, request).finally(() => {
        if (refreshInFlightSessionRef.current === targetSessionId) {
          refreshInFlightSessionRef.current = null;
        }
      });
    },
    [refreshSessionDetail],
  );

  const scheduleRefreshSessionDetail = useCallback(
    (targetSessionId: string, request: AgentSessionDetailRefreshRequest) => {
      if (refreshInFlightSessionRef.current === targetSessionId) {
        return;
      }

      if (refreshTimerRef.current !== null) {
        window.clearTimeout(refreshTimerRef.current);
      }

      refreshTimerRef.current = window.setTimeout(() => {
        refreshTimerRef.current = null;
        refreshSessionDetailOnce(targetSessionId, request);
      }, RUNTIME_DETAIL_REFRESH_COALESCE_MS);
    },
    [refreshSessionDetailOnce],
  );

  const scheduleRuntimeSyncRefresh = useCallback(
    (targetSessionId: string, request: AgentSessionDetailRefreshRequest) => {
      if (request.detailMergeMode === "terminal_reconcile") {
        scheduleRefreshSessionDetail(targetSessionId, request);
        return;
      }

      scheduleRefreshSessionDetail(targetSessionId, request);
    },
    [scheduleRefreshSessionDetail],
  );

  useEffect(
    () => () => {
      if (refreshTimerRef.current !== null) {
        window.clearTimeout(refreshTimerRef.current);
        refreshTimerRef.current = null;
      }
    },
    [],
  );

  useEffect(() => {
    const wasSending = lastIsSendingRef.current;
    const wasCurrentTurnEventName = lastCurrentTurnEventNameRef.current;
    lastIsSendingRef.current = isSending;
    lastCurrentTurnEventNameRef.current = normalizedCurrentTurnEventName;

    if (!sessionId || isSending || normalizedCurrentTurnEventName) {
      return;
    }

    if (!wasSending && !wasCurrentTurnEventName) {
      return;
    }

    const deferredRefreshRequest = deferredRuntimeRefreshRequestRef.current;
    deferredRuntimeRefreshRequestRef.current = null;
    scheduleRuntimeSyncRefresh(
      sessionId,
      deferredRefreshRequest ?? RUNTIME_SYNC_REFRESH_REQUESTS.sendSettled,
    );
  }, [
    isSending,
    normalizedCurrentTurnEventName,
    scheduleRuntimeSyncRefresh,
    sessionId,
  ]);

  useEffect(() => {
    if (!isSending) {
      observedActiveRuntimeWorkRef.current = false;
      return;
    }

    if (hasActiveRuntimeWork) {
      observedActiveRuntimeWorkRef.current = true;
    }
  }, [hasActiveRuntimeWork, isSending]);

  useEffect(() => {
    if (!sessionId || !isSending || !settleActiveRuntimeStream) {
      return;
    }
    const latestCurrentTurnEventName =
      currentTurnEventNameRef?.current?.trim() || null;
    // A bound listener without a runtime turn id is still in the start/accept
    // window. The read model may only contain the previous completed turn.
    if (
      (normalizedCurrentTurnEventName || latestCurrentTurnEventName) &&
      !normalizedCurrentStreamTurnId
    ) {
      return;
    }
    const hasRunningLocalTurn = hasRunningTurn(threadTurns);
    const hasTerminalTurnInReadModel =
      hasRunningLocalTurn && !normalizedCurrentStreamTurnId
        ? false
        : hasTerminalTurn(threadTurns, normalizedCurrentStreamTurnId);
    const threadReadTerminalInfo = readThreadReadTerminalInfo(
      threadRead,
      normalizedCurrentStreamTurnId,
    );
    const hasAuthoritativeThreadReadStatus =
      isThreadStatusAuthoritativeForCurrentStream({
        status: threadReadStatus,
        currentTurnId: normalizedCurrentStreamTurnId,
        threadReadTerminalInfo,
      });
    const hasTerminalThreadReadModel = normalizedCurrentStreamTurnId
      ? threadReadTerminalInfo.hasAuthoritativeTerminal ||
        hasAuthoritativeThreadReadStatus
      : threadReadTerminalInfo.hasTerminal;
    const hasTerminalReadModel =
      hasAuthoritativeThreadReadStatus ||
      hasTerminalThreadReadModel ||
      hasTerminalTurnInReadModel;
    if (!observedActiveRuntimeWorkRef.current && !hasTerminalReadModel) {
      return;
    }
    const normalizedThreadReadStatus = normalizeRuntimeStatus(threadReadStatus);
    if (
      hasRunningLocalTurn &&
      !normalizedCurrentStreamTurnId &&
      isExplicitTerminalRuntimeStatus(threadReadStatus)
    ) {
      return;
    }
    const shouldForceSettleByThreadStatus =
      shouldForceSettleStaleRunningTurn(threadReadStatus) &&
      isThreadStatusAuthoritativeForCurrentStream({
        status: normalizedThreadReadStatus,
        currentTurnId: normalizedCurrentStreamTurnId,
        threadReadTerminalInfo,
      });
    if (
      queuedTurnCount > 0 ||
      hasRunningThreadReadForRuntimeSync(threadRead) ||
      hasRunningLocalTurn
    ) {
      if (
        shouldForceSettleByThreadStatus ||
        threadReadTerminalInfo.hasAuthoritativeTerminal ||
        hasTerminalTurnInReadModel
      ) {
        observedActiveRuntimeWorkRef.current = false;
        settleActiveRuntimeStream(sessionId);
      }
      return;
    }
    if (
      !hasTerminalTurnInReadModel &&
      !hasTerminalThreadReadModel &&
      !isTerminalRuntimeStatus(threadReadStatus)
    ) {
      return;
    }

    observedActiveRuntimeWorkRef.current = false;
    settleActiveRuntimeStream(sessionId);
  }, [
    isSending,
    queuedTurnCount,
    sessionId,
    settleActiveRuntimeStream,
    threadRead,
    threadReadStatus,
    threadTurns,
    normalizedCurrentStreamTurnId,
    currentTurnEventNameRef,
  ]);

  useEffect(() => {
    if (!sessionId || isSending || normalizedCurrentTurnEventName) {
      return;
    }

    if (
      !shouldPollRecoveredRuntimeWork({
        threadRead,
        threadReadStatus,
        queuedTurnCount,
        threadTurns,
      })
    ) {
      return;
    }

    const timer = window.setInterval(() => {
      refreshSessionDetailOnce(
        sessionId,
        RUNTIME_SYNC_REFRESH_REQUESTS.recoveredPoll,
      );
    }, 1500);

    return () => {
      window.clearInterval(timer);
    };
  }, [
    isSending,
    threadRead,
    threadReadStatus,
    queuedTurnCount,
    normalizedCurrentTurnEventName,
    refreshSessionDetailOnce,
    sessionId,
    threadTurns,
  ]);

  useEffect(() => {
    if (!sessionId || !shouldUseAppServerBridgeRuntimePolling) {
      return;
    }

    let graceTimer: number | null = null;
    let pollTimer: number | null = null;

    const startPolling = () => {
      refreshSessionDetailOnce(sessionId, RUNTIME_SYNC_REFRESH_REQUESTS.poll);

      pollTimer = window.setInterval(() => {
        refreshSessionDetailOnce(sessionId, RUNTIME_SYNC_REFRESH_REQUESTS.poll);
      }, APP_SERVER_BRIDGE_RUNTIME_POLL_MS);
    };

    if (hasRuntimeEventListenerCapability) {
      graceTimer = window.setTimeout(
        startPolling,
        APP_SERVER_RUNTIME_EVENT_BIND_GRACE_MS,
      );
    } else {
      startPolling();
    }

    return () => {
      if (graceTimer !== null) {
        window.clearTimeout(graceTimer);
      }
      if (pollTimer !== null) {
        window.clearInterval(pollTimer);
      }
    };
  }, [
    hasRuntimeEventListenerCapability,
    refreshSessionDetailOnce,
    sessionId,
    shouldUseAppServerBridgeRuntimePolling,
  ]);

  useEffect(() => {
    if (!sessionId || !normalizedCurrentTurnEventName) {
      return;
    }

    let disposed = false;
    let unlisten: (() => void) | null = null;

    const subscribe = async () => {
      const nextUnlisten = await runtime.listenToTurnEvents(
        normalizedCurrentTurnEventName,
        (event) => {
          const refreshRequest = resolveRefreshRequestForTurnEventPayload(
            event.payload,
          );
          if (
            disposed ||
            sessionIdRef.current !== sessionId ||
            !refreshRequest
          ) {
            return;
          }
          deferredRuntimeRefreshRequestRef.current =
            preferRuntimeRefreshRequest(
              deferredRuntimeRefreshRequestRef.current,
              refreshRequest,
            );
        },
      );

      if (disposed) {
        nextUnlisten();
        return;
      }

      unlisten = nextUnlisten;
    };

    void subscribe();

    return () => {
      disposed = true;
      unlisten?.();
    };
  }, [normalizedCurrentTurnEventName, runtime, sessionId, sessionIdRef]);
}
