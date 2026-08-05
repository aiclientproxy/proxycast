import type { WorkspaceRightSurfacePendingRequest } from "@/lib/api/workspaceRightSurface";
import type { AgentThreadItem } from "@/lib/api/agentProtocol";

export type WorkspacePluginSurfaceStrategy =
  | "controlledBrowserWindow"
  | "webContentsView";

export interface WorkspacePluginSurfaceDescriptor {
  appId: string;
  title: string;
  entryUrl?: string;
  mcpApp?: {
    resourceUri: string;
    serverName: string;
    toolItemId: string;
  };
  containerId: string;
  activeStrategy: WorkspacePluginSurfaceStrategy;
  supportedStrategies: WorkspacePluginSurfaceStrategy[];
  sourceRequestId?: string;
}

export interface WorkspacePluginSurfaceSessionEpoch {
  sessionId: string;
  threadId: string;
}

export function resolveWorkspacePluginSurfaceThreadId(
  currentThreadId: string | null | undefined,
  threadItems: readonly AgentThreadItem[],
): string | null {
  const explicitThreadId = normalizeKey(currentThreadId);
  if (explicitThreadId) {
    return explicitThreadId;
  }

  const itemThreadIds = new Set(
    threadItems.map((item) => normalizeKey(item.thread_id)).filter(Boolean),
  );
  return itemThreadIds.size === 1 ? [...itemThreadIds][0] ?? null : null;
}

export function resolveWorkspacePluginSurfaceSessionEpoch({
  currentSessionId,
  currentThreadId,
  previousEpoch,
}: {
  currentSessionId?: string | null;
  currentThreadId?: string | null;
  previousEpoch?: WorkspacePluginSurfaceSessionEpoch | null;
}): {
  epoch: WorkspacePluginSurfaceSessionEpoch | null;
  ready: boolean;
} {
  const sessionId = normalizeKey(currentSessionId);
  const threadId = normalizeKey(currentThreadId);
  if (!sessionId || !threadId) {
    return { epoch: previousEpoch ?? null, ready: false };
  }
  if (
    previousEpoch &&
    previousEpoch.sessionId !== sessionId &&
    previousEpoch.threadId === threadId
  ) {
    return { epoch: previousEpoch, ready: false };
  }
  return {
    epoch: { sessionId, threadId },
    ready: true,
  };
}

export function buildWorkspacePluginSurfacesFromThreadItems(
  threadItems: readonly AgentThreadItem[],
  dismissedContainerIds: readonly string[] = [],
  threadId?: string | null,
): WorkspacePluginSurfaceDescriptor[] {
  const dismissed = new Set(
    dismissedContainerIds.map(normalizeKey).filter(Boolean),
  );
  const normalizedThreadId = normalizeKey(threadId);
  const next: WorkspacePluginSurfaceDescriptor[] = [];
  for (const item of threadItems) {
    if (
      normalizedThreadId &&
      normalizeKey(item.thread_id) !== normalizedThreadId
    ) {
      continue;
    }
    if (item.type !== "tool_call" || item.status !== "completed") {
      continue;
    }
    const metadata = asRecord(item.metadata);
    if (firstString(metadata?.canonical_type) !== "mcpToolCall") {
      continue;
    }
    const pluginId = firstString(metadata?.plugin_id);
    const resourceUri = firstString(metadata?.mcp_app_resource_uri);
    const serverName = firstString(metadata?.server);
    if (!pluginId || !resourceUri || !serverName || !isMcpAppUri(resourceUri)) {
      continue;
    }
    const containerId = `mcp-app-${item.id}`;
    if (dismissed.has(normalizeKey(containerId))) {
      continue;
    }
    upsertWorkspacePluginSurfaceDescriptor(next, {
      appId: pluginId,
      title: pluginId,
      containerId,
      activeStrategy: "webContentsView",
      supportedStrategies: ["webContentsView"],
      mcpApp: {
        resourceUri,
        serverName,
        toolItemId: item.id,
      },
    });
  }
  return next;
}

export function buildWorkspacePluginSurfaceFromPendingRequests(
  pendingRequests: readonly WorkspaceRightSurfacePendingRequest[],
): WorkspacePluginSurfaceDescriptor | null {
  return (
    buildWorkspacePluginSurfacesFromPendingRequests(pendingRequests)[0] ?? null
  );
}

export function buildWorkspacePluginSurfacesFromPendingRequests(
  pendingRequests: readonly WorkspaceRightSurfacePendingRequest[],
): WorkspacePluginSurfaceDescriptor[] {
  const next: WorkspacePluginSurfaceDescriptor[] = [];
  for (const request of pendingRequests) {
    if (request.status !== "pending" || request.surfaceKind !== "appSurface") {
      continue;
    }

    const descriptor = buildWorkspacePluginSurfaceFromPendingRequest(request);
    if (descriptor) {
      upsertWorkspacePluginSurfaceDescriptor(next, descriptor);
    }
  }

  return next;
}

export function buildWorkspacePluginSurfaceFromPendingRequest(
  request: WorkspaceRightSurfacePendingRequest,
): WorkspacePluginSurfaceDescriptor | null {
  const metadata = asRecord(request.metadata);
  const surface = asRecord(metadata?.surface) ?? metadata;
  const embedding = asRecord(surface?.embedding);
  const entryUrl = firstString(
    surface?.entryUrl,
    metadata?.entryUrl,
    metadata?.url,
  );
  const appId = firstString(
    metadata?.appId,
    surface?.appId,
    request.candidateId,
  );
  const containerId = firstString(
    surface?.containerId,
    metadata?.containerId,
    request.candidateId,
    request.requestId,
  );
  if (!entryUrl || !appId || !containerId) {
    return null;
  }

  const supportedStrategies = normalizeStrategies(
    surface?.supportedStrategies,
    metadata?.supportedStrategies,
  );
  if (!supportedStrategies.includes("webContentsView")) {
    return null;
  }
  if (embedding?.rightSurfaceDock === false) {
    return null;
  }
  if (embedding?.iframe === true || embedding?.browserView === true) {
    return null;
  }

  return {
    appId,
    title:
      firstString(
        metadata?.title,
        metadata?.name,
        metadata?.appName,
        surface?.title,
      ) ?? appId,
    entryUrl,
    containerId,
    activeStrategy:
      normalizeStrategy(surface?.activeStrategy) ?? "webContentsView",
    supportedStrategies,
    sourceRequestId: request.requestId,
  };
}

export function mergeWorkspacePluginSurfaceDescriptors(
  currentSurfaces: readonly WorkspacePluginSurfaceDescriptor[],
  incomingSurfaces: readonly WorkspacePluginSurfaceDescriptor[],
): WorkspacePluginSurfaceDescriptor[] {
  const next = [...currentSurfaces];
  for (const surface of incomingSurfaces) {
    upsertWorkspacePluginSurfaceDescriptor(next, surface);
  }
  return next;
}

export function selectWorkspacePluginSurfaceDescriptor(
  surfaces: readonly WorkspacePluginSurfaceDescriptor[],
  activeContainerId?: string | null,
): WorkspacePluginSurfaceDescriptor | null {
  const normalizedActiveContainerId = normalizeKey(activeContainerId);
  if (normalizedActiveContainerId) {
    const selected = surfaces.find(
      (surface) =>
        normalizeKey(surface.containerId) === normalizedActiveContainerId,
    );
    if (selected) {
      return selected;
    }
  }
  return surfaces[0] ?? null;
}

export function resolveWorkspacePluginSurfaceActiveContainerId({
  activeContainerId,
  preferredContainerId,
  surfaces,
}: {
  activeContainerId?: string | null;
  preferredContainerId?: string | null;
  surfaces: readonly WorkspacePluginSurfaceDescriptor[];
}): string | null {
  const preferred = normalizeKey(preferredContainerId);
  if (
    preferred &&
    surfaces.some((surface) => normalizeKey(surface.containerId) === preferred)
  ) {
    return preferred;
  }

  const active = normalizeKey(activeContainerId);
  if (
    active &&
    surfaces.some((surface) => normalizeKey(surface.containerId) === active)
  ) {
    return active;
  }

  return surfaces[0]?.containerId ?? null;
}

export function closeWorkspacePluginSurfaceDescriptor({
  activeContainerId,
  containerId,
  surfaces,
}: {
  activeContainerId?: string | null;
  containerId: string;
  surfaces: readonly WorkspacePluginSurfaceDescriptor[];
}): {
  activeContainerId: string | null;
  surfaces: WorkspacePluginSurfaceDescriptor[];
} {
  const normalizedContainerId = normalizeKey(containerId);
  const closedIndex = surfaces.findIndex(
    (surface) => normalizeKey(surface.containerId) === normalizedContainerId,
  );
  if (closedIndex < 0) {
    return {
      activeContainerId:
        resolveWorkspacePluginSurfaceActiveContainerId({
          activeContainerId,
          surfaces,
        }) ?? null,
      surfaces: [...surfaces],
    };
  }

  const nextSurfaces = surfaces.filter(
    (surface) => normalizeKey(surface.containerId) !== normalizedContainerId,
  );
  const active = normalizeKey(activeContainerId);
  if (active && active !== normalizedContainerId) {
    return {
      activeContainerId: resolveWorkspacePluginSurfaceActiveContainerId({
        activeContainerId: active,
        surfaces: nextSurfaces,
      }),
      surfaces: nextSurfaces,
    };
  }

  return {
    activeContainerId:
      nextSurfaces[Math.min(closedIndex, nextSurfaces.length - 1)]
        ?.containerId ?? null,
    surfaces: nextSurfaces,
  };
}

function upsertWorkspacePluginSurfaceDescriptor(
  surfaces: WorkspacePluginSurfaceDescriptor[],
  incoming: WorkspacePluginSurfaceDescriptor,
): void {
  const incomingKey = normalizeKey(incoming.containerId);
  const existingIndex = surfaces.findIndex(
    (surface) => normalizeKey(surface.containerId) === incomingKey,
  );
  if (existingIndex >= 0) {
    surfaces[existingIndex] = incoming;
    return;
  }
  surfaces.push(incoming);
}

function normalizeStrategies(
  ...values: unknown[]
): WorkspacePluginSurfaceStrategy[] {
  const strategies = values.flatMap((value) =>
    Array.isArray(value) ? value : [],
  );
  const next = strategies
    .map(normalizeStrategy)
    .filter(
      (strategy): strategy is WorkspacePluginSurfaceStrategy =>
        strategy !== null,
    );
  return next.length > 0 ? Array.from(new Set(next)) : ["webContentsView"];
}

function normalizeStrategy(
  value: unknown,
): WorkspacePluginSurfaceStrategy | null {
  return value === "controlledBrowserWindow" || value === "webContentsView"
    ? value
    : null;
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

function normalizeKey(value: string | null | undefined): string | null {
  const normalized = value?.trim();
  return normalized ? normalized : null;
}

function isMcpAppUri(value: string): boolean {
  try {
    return new URL(value).protocol === "ui:";
  } catch {
    return false;
  }
}
