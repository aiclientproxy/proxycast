import { useCallback, useEffect, useState } from "react";
import type { QueuedSubmission } from "@limecloud/app-server-client";
import type { AppServerJsonRpcNotification } from "@/lib/api/appServer";
import {
  subscribeAppServerNotifications,
  type AppServerEventBusSubscription,
} from "@/lib/api/appServerEventBus";
import { listThreadQueue } from "@/lib/api/agentRuntime/threadQueueClient";

type ThreadQueueReader = (threadId: string) => Promise<QueuedSubmission[]>;
type ThreadQueueSubscriber = (
  subscription: AppServerEventBusSubscription,
) => () => void;

interface ThreadQueueState {
  error: unknown;
  loading: boolean;
  submissions: QueuedSubmission[];
  threadId: string | null;
}

function readRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

export function isScopedThreadQueueChangedNotification(
  notification: AppServerJsonRpcNotification,
  currentThreadId: string,
): boolean {
  if (notification.method !== "thread/queue/changed") {
    return false;
  }
  const params = readRecord(notification.params);
  return (
    typeof params?.threadId === "string" &&
    params.threadId.trim() === currentThreadId
  );
}

export function useAgentSessionThreadQueue(params: {
  readQueue?: ThreadQueueReader;
  subscribeNotifications?: ThreadQueueSubscriber;
  threadId?: string | null;
}): Omit<ThreadQueueState, "threadId"> & { refresh: () => void } {
  const readQueue = params.readQueue ?? listThreadQueue;
  const subscribeNotifications =
    params.subscribeNotifications ?? subscribeAppServerNotifications;
  const threadId = params.threadId?.trim() || null;
  const [state, setState] = useState<ThreadQueueState>({
    error: null,
    loading: false,
    submissions: [],
    threadId: null,
  });
  const [refreshToken, setRefreshToken] = useState(0);
  const refresh = useCallback(() => {
    setRefreshToken((value) => value + 1);
  }, []);

  useEffect(() => {
    if (!threadId) {
      setState({
        error: null,
        loading: false,
        submissions: [],
        threadId: null,
      });
      return;
    }

    let active = true;
    let readRevision = 0;
    const readNow = () => {
      const revision = ++readRevision;
      void readQueue(threadId).then(
        (submissions) => {
          if (!active || revision !== readRevision) {
            return;
          }
          setState({
            error: null,
            loading: false,
            submissions,
            threadId,
          });
        },
        (error) => {
          if (!active || revision !== readRevision) {
            return;
          }
          setState((current) => ({
            error,
            loading: false,
            submissions:
              current.threadId === threadId ? current.submissions : [],
            threadId,
          }));
        },
      );
    };

    setState({
      error: null,
      loading: true,
      submissions: [],
      threadId,
    });
    const unsubscribe = subscribeNotifications({
      onError: (error) => {
        if (!active) {
          return;
        }
        setState((current) =>
          current.threadId === threadId ? { ...current, error } : current,
        );
      },
      onNotifications: (notifications) => {
        if (
          active &&
          notifications.some((notification) =>
            isScopedThreadQueueChangedNotification(notification, threadId),
          )
        ) {
          readNow();
        }
      },
    });
    readNow();

    return () => {
      active = false;
      readRevision += 1;
      unsubscribe();
    };
  }, [readQueue, refreshToken, subscribeNotifications, threadId]);

  if (state.threadId !== threadId) {
    return {
      error: null,
      loading: Boolean(threadId),
      submissions: [],
      refresh,
    };
  }
  return {
    error: state.error,
    loading: state.loading,
    submissions: state.submissions,
    refresh,
  };
}
