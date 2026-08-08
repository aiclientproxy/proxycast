import { useCallback, useEffect, useMemo, useState } from "react";
import { isAppServerBridgeAvailable } from "@/lib/api/appServerBridgeAvailability";
import {
  consumeWorkspaceRightSurfacePending,
  dismissWorkspaceRightSurfacePending,
  listWorkspaceRightSurfacePending,
  subscribeWorkspaceRightSurfacePendingChangedNotifications,
  type WorkspaceRightSurfacePendingChangedParams,
  type WorkspaceRightSurfacePendingChangedSubscriber,
  type WorkspaceRightSurfacePendingConsumeParams,
  type WorkspaceRightSurfacePendingConsumeResponse,
  type WorkspaceRightSurfacePendingDismissParams,
  type WorkspaceRightSurfacePendingDismissResponse,
  type WorkspaceRightSurfacePendingListParams,
  type WorkspaceRightSurfacePendingRequest,
} from "@/lib/api/workspaceRightSurface";
import {
  buildWorkspaceRightSurfaceAppServerPendingIntents,
  normalizeWorkspaceRightSurfaceKind,
  type WorkspaceRightSurfaceKind,
  type WorkspaceRightSurfaceIntent,
} from "./right-surface";
import type { WorkspaceFilesSurfaceTarget } from "./WorkspaceFilesSurface";
import {
  buildWorkspacePluginSurfacesFromPendingRequests,
  type WorkspacePluginSurfaceDescriptor,
} from "./workspacePluginSurfaceModel";
import type { WorkspaceObjectCanvasCandidate } from "./workspaceObjectCanvasModel";
import {
  buildWorkspaceArticleWorkspaceFromPendingRequests,
  type WorkspaceArticleWorkspace,
} from "./workspaceArticleWorkspaceModel";
import {
  buildWorkspaceRightSurfacePendingBrowserIntent,
  type WorkspaceRightSurfaceBrowserIntent,
} from "./workspaceRightSurfaceBrowserIntent";

const DEFAULT_RIGHT_SURFACE_PENDING_POLL_MS = 5_000;
const DEFAULT_RIGHT_SURFACE_PENDING_EVENT_DRAIN_MS = 250;
const DEFAULT_RIGHT_SURFACE_PENDING_EVENT_LIMIT = 20;
const DEFAULT_RIGHT_SURFACE_PENDING_LIMIT = 50;

export interface UseWorkspaceRightSurfacePendingRuntimeOptions {
  enabled: boolean;
  autoRefreshEnabled?: boolean;
  workspaceId?: string | null;
  workspaceRoot?: string | null;
  sessionId?: string | null;
  pollIntervalMs?: number;
  eventDrainIntervalMs?: number;
  eventDrainLimit?: number;
  limit?: number;
  isBridgeAvailable?: () => boolean;
  listPending?: (
    params: WorkspaceRightSurfacePendingListParams,
  ) => Promise<{ pending?: WorkspaceRightSurfacePendingRequest[] }>;
  consumePending?: (
    params: WorkspaceRightSurfacePendingConsumeParams,
  ) => Promise<WorkspaceRightSurfacePendingConsumeResponse>;
  dismissPending?: (
    params: WorkspaceRightSurfacePendingDismissParams,
  ) => Promise<WorkspaceRightSurfacePendingDismissResponse>;
  drainPendingChanges?: (
    limit?: number,
  ) => Promise<WorkspaceRightSurfacePendingChangedParams[]>;
  subscribePendingChanges?: WorkspaceRightSurfacePendingChangedSubscriber;
  now?: () => number;
}

export interface WorkspaceRightSurfacePendingRuntime {
  pendingRequests: WorkspaceRightSurfacePendingRequest[];
  pendingIntents: WorkspaceRightSurfaceIntent[];
  pendingFileTarget: WorkspaceFilesSurfaceTarget | null;
  pendingPluginSurface: WorkspacePluginSurfaceDescriptor | null;
  pendingPluginSurfaces: WorkspacePluginSurfaceDescriptor[];
  pendingObjectCanvasCandidate: WorkspaceObjectCanvasCandidate | null;
  pendingArticleWorkspace: WorkspaceArticleWorkspace | null;
  pendingBrowserIntent: WorkspaceRightSurfaceBrowserIntent | null;
  lastError: Error | null;
  refreshPendingRequests: () => Promise<void>;
  consumePendingRequestsForSurface: (
    surfaceKind: WorkspaceRightSurfaceKind,
  ) => Promise<void>;
  dismissPendingRequestsForSurface: (
    surfaceKind: WorkspaceRightSurfaceKind,
    reason?: string,
  ) => Promise<void>;
}

export function buildWorkspaceRightSurfacePendingListParams({
  limit = DEFAULT_RIGHT_SURFACE_PENDING_LIMIT,
  sessionId,
  workspaceId,
  workspaceRoot,
}: Pick<
  UseWorkspaceRightSurfacePendingRuntimeOptions,
  "limit" | "sessionId" | "workspaceId" | "workspaceRoot"
>): WorkspaceRightSurfacePendingListParams | null {
  const normalizedWorkspaceId = normalizeOptionalString(workspaceId);
  const normalizedWorkspaceRoot = normalizeOptionalString(workspaceRoot);
  const normalizedSessionId = normalizeOptionalString(sessionId);

  if (
    !normalizedWorkspaceId &&
    !normalizedWorkspaceRoot &&
    !normalizedSessionId
  ) {
    return null;
  }

  return {
    limit,
    ...(normalizedWorkspaceId ? { workspaceId: normalizedWorkspaceId } : {}),
    ...(normalizedWorkspaceRoot
      ? { workspaceRoot: normalizedWorkspaceRoot }
      : {}),
    ...(!normalizedWorkspaceId &&
    !normalizedWorkspaceRoot &&
    normalizedSessionId
      ? { sessionId: normalizedSessionId }
      : {}),
  };
}

export function useWorkspaceRightSurfacePendingRuntime({
  enabled,
  autoRefreshEnabled = enabled,
  workspaceId,
  workspaceRoot,
  sessionId,
  pollIntervalMs = DEFAULT_RIGHT_SURFACE_PENDING_POLL_MS,
  eventDrainIntervalMs = DEFAULT_RIGHT_SURFACE_PENDING_EVENT_DRAIN_MS,
  eventDrainLimit = DEFAULT_RIGHT_SURFACE_PENDING_EVENT_LIMIT,
  limit = DEFAULT_RIGHT_SURFACE_PENDING_LIMIT,
  isBridgeAvailable = isAppServerBridgeAvailable,
  listPending = listWorkspaceRightSurfacePending,
  consumePending = consumeWorkspaceRightSurfacePending,
  dismissPending = dismissWorkspaceRightSurfacePending,
  subscribePendingChanges = subscribeWorkspaceRightSurfacePendingChangedNotifications,
  drainPendingChanges,
  now = Date.now,
}: UseWorkspaceRightSurfacePendingRuntimeOptions): WorkspaceRightSurfacePendingRuntime {
  const [pendingRequests, setPendingRequests] = useState<
    WorkspaceRightSurfacePendingRequest[]
  >([]);
  const [lastError, setLastError] = useState<Error | null>(null);
  const listParams = useMemo(
    () =>
      buildWorkspaceRightSurfacePendingListParams({
        limit,
        sessionId,
        workspaceId,
        workspaceRoot,
      }),
    [limit, sessionId, workspaceId, workspaceRoot],
  );

  const refreshPendingRequests = useCallback(async () => {
    if (!enabled || !listParams || !isBridgeAvailable()) {
      setPendingRequests((current) => (current.length === 0 ? current : []));
      setLastError((current) => (current === null ? current : null));
      return;
    }

    try {
      const response = await listPending(listParams);
      const nextPending = response.pending ?? [];
      setPendingRequests((current) =>
        current.length === 0 && nextPending.length === 0
          ? current
          : nextPending,
      );
      setLastError((current) => (current === null ? current : null));
    } catch (error) {
      setPendingRequests((current) => (current.length === 0 ? current : []));
      setLastError(error instanceof Error ? error : new Error(String(error)));
    }
  }, [enabled, isBridgeAvailable, listParams, listPending]);

  const consumePendingRequestsForSurface = useCallback(
    async (surfaceKind: WorkspaceRightSurfaceKind) => {
      if (!enabled || !isBridgeAvailable()) {
        return;
      }

      const requestIds = pendingRequests
        .filter(
          (request) =>
            request.status === "pending" &&
            normalizeWorkspaceRightSurfaceKind(request.surfaceKind) ===
              surfaceKind,
        )
        .map((request) => request.requestId)
        .filter((requestId) => requestId.trim().length > 0);

      if (requestIds.length === 0) {
        return;
      }

      try {
        const response = await consumePending({ requestIds });
        const consumedIds = new Set(response.consumedRequestIds ?? []);
        if (consumedIds.size > 0) {
          setPendingRequests((current) =>
            current.filter((request) => !consumedIds.has(request.requestId)),
          );
        }
        setLastError((current) => (current === null ? current : null));
      } catch (error) {
        setLastError(error instanceof Error ? error : new Error(String(error)));
      }
    },
    [consumePending, enabled, isBridgeAvailable, pendingRequests],
  );

  const dismissPendingRequestsForSurface = useCallback(
    async (surfaceKind: WorkspaceRightSurfaceKind, reason?: string) => {
      if (!enabled || !isBridgeAvailable()) {
        return;
      }

      const requestIds = pendingRequests
        .filter(
          (request) =>
            request.status === "pending" &&
            normalizeWorkspaceRightSurfaceKind(request.surfaceKind) ===
              surfaceKind,
        )
        .map((request) => request.requestId)
        .filter((requestId) => requestId.trim().length > 0);

      if (requestIds.length === 0) {
        return;
      }

      try {
        const normalizedReason = normalizeOptionalString(reason);
        const response = await dismissPending({
          requestIds,
          ...(normalizedReason ? { reason: normalizedReason } : {}),
        });
        const dismissedIds = new Set(response.dismissedRequestIds ?? []);
        if (dismissedIds.size > 0) {
          setPendingRequests((current) =>
            current.filter((request) => !dismissedIds.has(request.requestId)),
          );
        }
        setLastError((current) => (current === null ? current : null));
      } catch (error) {
        setLastError(error instanceof Error ? error : new Error(String(error)));
      }
    },
    [dismissPending, enabled, isBridgeAvailable, pendingRequests],
  );

  useEffect(() => {
    if (!autoRefreshEnabled) {
      return undefined;
    }

    void refreshPendingRequests();
    if (!enabled || !listParams || pollIntervalMs <= 0) {
      return undefined;
    }

    const timer = window.setInterval(() => {
      void refreshPendingRequests();
    }, pollIntervalMs);

    return () => window.clearInterval(timer);
  }, [
    autoRefreshEnabled,
    enabled,
    listParams,
    pollIntervalMs,
    refreshPendingRequests,
  ]);

  useEffect(() => {
    if (
      !autoRefreshEnabled ||
      !enabled ||
      !listParams ||
      eventDrainIntervalMs <= 0
    ) {
      return undefined;
    }

    const applyChanges = (
      changes: readonly WorkspaceRightSurfacePendingChangedParams[],
    ) => {
      if (changes.length === 0) {
        return;
      }
      setPendingRequests((current) =>
        applyWorkspaceRightSurfacePendingChanges(current, changes, listParams),
      );
      setLastError((current) => (current === null ? current : null));
    };

    if (!drainPendingChanges) {
      if (!isBridgeAvailable()) {
        return undefined;
      }

      return subscribePendingChanges(
        {
          onChanges: applyChanges,
          onError: (error) => {
            setLastError(
              error instanceof Error ? error : new Error(String(error)),
            );
          },
        },
        {
          intervalMs: eventDrainIntervalMs,
          isBridgeAvailable,
          limit: eventDrainLimit,
        },
      );
    }

    let disposed = false;
    let draining = false;

    const drainOnce = async () => {
      if (disposed || draining || !isBridgeAvailable()) {
        return;
      }

      draining = true;
      try {
        const changes = await drainPendingChanges(eventDrainLimit);
        if (disposed || changes.length === 0) {
          return;
        }
        applyChanges(changes);
      } catch (error) {
        if (!disposed) {
          setLastError(
            error instanceof Error ? error : new Error(String(error)),
          );
        }
      } finally {
        draining = false;
      }
    };

    void drainOnce();
    const timer = window.setInterval(() => {
      void drainOnce();
    }, eventDrainIntervalMs);

    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, [
    drainPendingChanges,
    autoRefreshEnabled,
    enabled,
    eventDrainIntervalMs,
    eventDrainLimit,
    isBridgeAvailable,
    listParams,
    subscribePendingChanges,
  ]);

  const pendingIntents = useMemo(() => {
    const createdAt = now();
    return buildWorkspaceRightSurfaceAppServerPendingIntents(
      pendingRequests,
      createdAt,
    );
  }, [now, pendingRequests]);
  const pendingFileTarget = useMemo(
    () => buildWorkspaceRightSurfacePendingFileTarget(pendingRequests),
    [pendingRequests],
  );
  const pendingPluginSurfaces = useMemo(
    () => buildWorkspacePluginSurfacesFromPendingRequests(pendingRequests),
    [pendingRequests],
  );
  const pendingPluginSurface = pendingPluginSurfaces[0] ?? null;
  const pendingObjectCanvasCandidate = useMemo(
    () =>
      buildWorkspaceRightSurfacePendingObjectCanvasCandidate(pendingRequests),
    [pendingRequests],
  );
  const pendingArticleWorkspace = useMemo(
    () => buildWorkspaceArticleWorkspaceFromPendingRequests(pendingRequests),
    [pendingRequests],
  );
  const pendingBrowserIntent = useMemo(
    () => buildWorkspaceRightSurfacePendingBrowserIntent(pendingRequests),
    [pendingRequests],
  );

  return {
    pendingRequests,
    pendingIntents,
    pendingFileTarget,
    pendingPluginSurface,
    pendingPluginSurfaces,
    pendingObjectCanvasCandidate,
    pendingArticleWorkspace,
    pendingBrowserIntent,
    lastError,
    refreshPendingRequests,
    consumePendingRequestsForSurface,
    dismissPendingRequestsForSurface,
  };
}

export function applyWorkspaceRightSurfacePendingChanges(
  currentRequests: readonly WorkspaceRightSurfacePendingRequest[],
  changes: readonly WorkspaceRightSurfacePendingChangedParams[],
  listParams?: WorkspaceRightSurfacePendingListParams | null,
): WorkspaceRightSurfacePendingRequest[] {
  let nextRequests = [...currentRequests];

  for (const change of changes) {
    const changeType = change.changeType.trim();
    if (changeType === "requested") {
      for (const request of change.pending ?? []) {
        if (
          request.status !== "pending" ||
          !matchesPendingListParams(request, listParams)
        ) {
          continue;
        }
        nextRequests = upsertPendingRequest(nextRequests, request);
      }
      continue;
    }

    if (changeType === "consumed") {
      nextRequests = removePendingRequestIds(nextRequests, [
        ...(change.requestIds ?? []),
        ...(change.consumedRequestIds ?? []),
      ]);
      continue;
    }

    if (changeType === "dismissed") {
      nextRequests = removePendingRequestIds(nextRequests, [
        ...(change.requestIds ?? []),
        ...(change.dismissedRequestIds ?? []),
      ]);
    }
  }

  return nextRequests;
}

export function buildWorkspaceRightSurfacePendingFileTarget(
  pendingRequests: readonly WorkspaceRightSurfacePendingRequest[],
): WorkspaceFilesSurfaceTarget | null {
  for (const request of pendingRequests) {
    if (request.status !== "pending" || request.surfaceKind !== "files") {
      continue;
    }

    const metadata = asRecord(request.metadata);
    const relativePath = normalizePath(
      firstString(
        metadata?.relativePath,
        metadata?.path,
        metadata?.filePath,
        request.candidateId,
      ),
    );
    if (!relativePath) {
      continue;
    }

    return {
      relativePath,
      title:
        firstString(metadata?.title, metadata?.name, metadata?.filename) ??
        extractFileName(relativePath),
    };
  }

  return null;
}

export function buildWorkspaceRightSurfacePendingObjectCanvasCandidate(
  pendingRequests: readonly WorkspaceRightSurfacePendingRequest[],
): WorkspaceObjectCanvasCandidate | null {
  for (const request of pendingRequests) {
    if (
      request.status !== "pending" ||
      request.surfaceKind !== "objectCanvas"
    ) {
      continue;
    }

    const metadata = asRecord(request.metadata);
    const candidateId = firstString(
      request.candidateId,
      metadata?.candidateId,
      metadata?.id,
      request.requestId,
    );
    if (!candidateId) {
      continue;
    }

    return {
      candidateId,
      title: firstString(metadata?.title, metadata?.name),
      url: firstString(metadata?.url, metadata?.href),
      sessionId: firstString(metadata?.sessionId, request.sessionId),
      profileKey: firstString(metadata?.profileKey),
      targetId: firstString(metadata?.targetId),
      lifecycleState: firstString(metadata?.lifecycleState, metadata?.status),
      controlMode: firstString(metadata?.controlMode),
      transportKind: firstString(metadata?.transportKind),
      launching: metadata?.launching === true,
      sourceKind: "rightSurfacePending",
      sourceRequestId: request.requestId,
    };
  }

  return null;
}

function normalizeOptionalString(value?: string | null): string | null {
  const normalized = value?.trim();
  return normalized ? normalized : null;
}

function matchesPendingListParams(
  request: WorkspaceRightSurfacePendingRequest,
  listParams?: WorkspaceRightSurfacePendingListParams | null,
): boolean {
  if (!listParams) {
    return true;
  }

  const workspaceId = normalizeOptionalString(listParams.workspaceId);
  if (
    workspaceId &&
    normalizeOptionalString(request.workspaceId) !== workspaceId
  ) {
    return false;
  }

  const workspaceRoot = normalizeOptionalString(listParams.workspaceRoot);
  if (
    workspaceRoot &&
    normalizeOptionalString(request.workspaceRoot) !== workspaceRoot
  ) {
    return false;
  }

  const sessionId = normalizeOptionalString(listParams.sessionId);
  if (sessionId && normalizeOptionalString(request.sessionId) !== sessionId) {
    return false;
  }

  return true;
}

function upsertPendingRequest(
  requests: WorkspaceRightSurfacePendingRequest[],
  request: WorkspaceRightSurfacePendingRequest,
): WorkspaceRightSurfacePendingRequest[] {
  const index = requests.findIndex(
    (item) => item.requestId === request.requestId,
  );
  if (index < 0) {
    return [...requests, request];
  }

  const nextRequests = [...requests];
  nextRequests[index] = request;
  return nextRequests;
}

function removePendingRequestIds(
  requests: WorkspaceRightSurfacePendingRequest[],
  requestIds: readonly string[],
): WorkspaceRightSurfacePendingRequest[] {
  const ids = new Set(
    requestIds.map((requestId) => requestId.trim()).filter(Boolean),
  );
  if (ids.size === 0) {
    return requests;
  }
  return requests.filter((request) => !ids.has(request.requestId));
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function firstString(...values: unknown[]): string | null {
  for (const value of values) {
    if (typeof value !== "string") {
      continue;
    }
    const normalized = value.trim();
    if (normalized) {
      return normalized;
    }
  }
  return null;
}

function normalizePath(value: string | null): string | null {
  const normalized = value?.replace(/\\/g, "/").trim();
  return normalized ? normalized : null;
}

function extractFileName(path: string): string {
  return path.split("/").filter(Boolean).at(-1) ?? path;
}
