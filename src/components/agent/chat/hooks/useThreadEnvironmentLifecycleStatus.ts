import { useEffect, useMemo, useState } from "react";
import type { AppServerJsonRpcNotification } from "@/lib/api/appServer";
import {
  subscribeAppServerNotifications,
  type AppServerEventBusSubscription,
} from "@/lib/api/appServerEventBus";
import type { AgentRuntimeThreadReadModel } from "@/lib/api/agentRuntime/sessionTypes";
import {
  readEnvironmentRuntimeStatuses,
  type EnvironmentRuntimeConnectionState,
} from "@/lib/api/environmentLifecycle";

export type ThreadEnvironmentLifecycleState = EnvironmentRuntimeConnectionState;

type EnvironmentNotificationSubscriber = (
  subscription: AppServerEventBusSubscription,
) => () => void;
type EnvironmentStatusReader = (
  environmentIds: readonly string[],
) => Promise<EnvironmentRuntimeConnectionState[]>;

function readRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function readEnvironmentId(value: unknown): string | null {
  const record = readRecord(value);
  const candidate =
    typeof value === "string"
      ? value
      : (record?.environmentId ?? record?.environment_id ?? record?.id);
  return typeof candidate === "string" && candidate.trim()
    ? candidate.trim()
    : null;
}

export function readThreadEnvironmentIds(
  threadRead?: AgentRuntimeThreadReadModel | null,
): string[] {
  const metadata = readRecord(threadRead?.session_business_object_ref_metadata);
  const environments = metadata?.environments;
  if (!Array.isArray(environments)) {
    return [];
  }
  return Array.from(
    new Set(
      environments
        .map(readEnvironmentId)
        .filter((value): value is string => Boolean(value)),
    ),
  );
}

export function isThreadEnvironmentLifecycleNotification(
  notification: AppServerJsonRpcNotification,
  threadId: string,
): boolean {
  if (
    notification.method !== "thread/environment/connected" &&
    notification.method !== "thread/environment/disconnected"
  ) {
    return false;
  }
  const params = readRecord(notification.params);
  return (
    params?.threadId === threadId &&
    Boolean(readEnvironmentId(params?.environmentId))
  );
}

export function projectThreadEnvironmentLifecycleStatus(
  current: readonly ThreadEnvironmentLifecycleState[],
  notification: AppServerJsonRpcNotification,
  threadId: string,
): ThreadEnvironmentLifecycleState[] {
  if (!isThreadEnvironmentLifecycleNotification(notification, threadId)) {
    return [...current];
  }
  const params = readRecord(notification.params);
  const environmentId = readEnvironmentId(params?.environmentId);
  if (!environmentId) {
    return [...current];
  }
  const status: EnvironmentRuntimeConnectionState["status"] =
    notification.method === "thread/environment/connected"
      ? "connected"
      : "disconnected";
  return current.some((item) => item.environmentId === environmentId)
    ? current.map((item) =>
        item.environmentId === environmentId ? { ...item, status } : item,
      )
    : [...current, { environmentId, status }];
}

export function useThreadEnvironmentLifecycleStatus(options: {
  threadId?: string | null;
  threadRead?: AgentRuntimeThreadReadModel | null;
  subscribeNotifications?: EnvironmentNotificationSubscriber;
  readStatuses?: EnvironmentStatusReader;
}): ThreadEnvironmentLifecycleState[] {
  const threadId = options.threadId?.trim() || null;
  const environmentIds = useMemo(
    () => readThreadEnvironmentIds(options.threadRead),
    [options.threadRead],
  );
  const environmentKey = environmentIds.join("\u0000");
  const subscribeNotifications =
    options.subscribeNotifications ?? subscribeAppServerNotifications;
  const readStatuses = options.readStatuses ?? readEnvironmentRuntimeStatuses;
  const [state, setState] = useState<ThreadEnvironmentLifecycleState[]>([]);

  useEffect(() => {
    const initial = environmentIds.map(
      (environmentId) =>
        ({
          environmentId,
          status: environmentId === "local" ? "connected" : "pending",
        }) satisfies ThreadEnvironmentLifecycleState,
    );
    setState(initial);
    if (!threadId || environmentIds.length === 0) {
      return;
    }

    let active = true;
    const notifiedEnvironmentIds = new Set<string>();
    const unsubscribe = subscribeNotifications({
      getDrainOptions: () => ({ includeRecent: true }),
      onNotifications: (notifications) => {
        if (!active) {
          return;
        }
        setState((current) =>
          notifications.reduce((next, notification) => {
            if (
              isThreadEnvironmentLifecycleNotification(notification, threadId)
            ) {
              const params = readRecord(notification.params);
              const environmentId = readEnvironmentId(params?.environmentId);
              if (environmentId) {
                notifiedEnvironmentIds.add(environmentId);
              }
            }
            return projectThreadEnvironmentLifecycleStatus(
              next,
              notification,
              threadId,
            );
          }, current),
        );
      },
    });
    void readStatuses(environmentIds).then(
      (statuses) => {
        if (!active) {
          return;
        }
        setState((current) => {
          const currentById = new Map(
            current.map((environment) => [
              environment.environmentId,
              environment,
            ]),
          );
          return statuses.map((environment) =>
            notifiedEnvironmentIds.has(environment.environmentId)
              ? (currentById.get(environment.environmentId) ?? environment)
              : environment,
          );
        });
      },
      () => undefined,
    );
    return () => {
      active = false;
      unsubscribe();
    };
  }, [
    environmentKey,
    environmentIds,
    readStatuses,
    subscribeNotifications,
    threadId,
  ]);

  return state;
}
