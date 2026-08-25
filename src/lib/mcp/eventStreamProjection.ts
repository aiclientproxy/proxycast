import type { McpServerEventStreamNotification } from "@limecloud/app-server-client";

export type McpEventStreamPhase = "active" | "terminated";

export interface McpEventStreamState {
  subscriptionId: string;
  phase: McpEventStreamPhase;
  lastEventMethod: string | null;
  lastEventName: string | null;
  eventCount: number;
  activeCount: number;
  reconnectCount: number;
  updatedAt: number;
}

export type McpEventStreamStateMap = Record<string, McpEventStreamState>;

function readEventName(params: unknown): string | null {
  if (!params || typeof params !== "object" || Array.isArray(params)) {
    return null;
  }
  const name = (params as Record<string, unknown>).name;
  return typeof name === "string" && name.trim() ? name : null;
}

export function reduceMcpEventStreamNotification(
  state: McpEventStreamStateMap,
  notification: McpServerEventStreamNotification,
  updatedAt = Date.now(),
): McpEventStreamStateMap {
  const { subscriptionId, notification: nested } = notification;
  const previous = state[subscriptionId];
  const isActive = nested.method === "notifications/events/active";
  const isTerminated = nested.method === "notifications/events/terminated";
  const activeCount = (previous?.activeCount ?? 0) + (isActive ? 1 : 0);
  const reconnectCount = Math.max(
    previous?.reconnectCount ?? 0,
    activeCount > 1 ? activeCount - 1 : 0,
  );

  return {
    ...state,
    [subscriptionId]: {
      subscriptionId,
      phase: isTerminated ? "terminated" : "active",
      lastEventMethod: nested.method,
      lastEventName:
        isActive || isTerminated ? null : readEventName(nested.params),
      eventCount:
        (previous?.eventCount ?? 0) + (isActive || isTerminated ? 0 : 1),
      activeCount,
      reconnectCount,
      updatedAt,
    },
  };
}

export function selectMcpEventStreams(
  state: McpEventStreamStateMap,
): McpEventStreamState[] {
  return Object.values(state).sort((left, right) => {
    if (right.updatedAt !== left.updatedAt) {
      return right.updatedAt - left.updatedAt;
    }
    return left.subscriptionId.localeCompare(right.subscriptionId);
  });
}
