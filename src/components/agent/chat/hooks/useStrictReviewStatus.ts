import { useEffect, useState } from "react";
import {
  isGuardianReviewCompletedNotification,
  isStrictReviewRequiredNotification,
  isTurnCompletedNotification,
} from "@limecloud/app-server-client";
import type { AppServerJsonRpcNotification } from "@/lib/api/appServer";
import {
  subscribeAppServerNotifications,
  type AppServerEventBusSubscription,
} from "@/lib/api/appServerEventBus";

type StrictReviewSubscriber = (
  subscription: AppServerEventBusSubscription,
) => () => void;

export interface StrictReviewStatus {
  startedAtMs: number;
  threadId: string;
  turnId: string;
}

export function projectStrictReviewStatus(
  current: StrictReviewStatus | null,
  notification: AppServerJsonRpcNotification,
  threadId: string,
): StrictReviewStatus | null {
  if (isStrictReviewRequiredNotification(notification)) {
    const params = notification.params;
    return params.threadId === threadId
      ? {
          startedAtMs: params.startedAtMs,
          threadId: params.threadId,
          turnId: params.turnId,
        }
      : current;
  }
  if (!current || current.threadId !== threadId) {
    return current;
  }
  if (isGuardianReviewCompletedNotification(notification)) {
    const params = notification.params;
    return params.threadId === current.threadId &&
      params.turnId === current.turnId
      ? null
      : current;
  }
  if (isTurnCompletedNotification(notification)) {
    const params = notification.params;
    return params.threadId === current.threadId &&
      params.turn.id === current.turnId
      ? null
      : current;
  }
  return current;
}

export function useStrictReviewStatus(options: {
  subscribeNotifications?: StrictReviewSubscriber;
  threadId?: string | null;
}): StrictReviewStatus | null {
  const threadId = options.threadId?.trim() || null;
  const subscribeNotifications =
    options.subscribeNotifications ?? subscribeAppServerNotifications;
  const [status, setStatus] = useState<StrictReviewStatus | null>(null);

  useEffect(() => {
    setStatus(null);
    if (!threadId) {
      return;
    }
    let active = true;
    const unsubscribe = subscribeNotifications({
      getDrainOptions: () => ({ includeRecent: true }),
      onNotifications: (notifications) => {
        if (!active) {
          return;
        }
        setStatus((current) =>
          notifications.reduce(
            (next, notification) =>
              projectStrictReviewStatus(next, notification, threadId),
            current,
          ),
        );
      },
    });
    return () => {
      active = false;
      unsubscribe();
    };
  }, [subscribeNotifications, threadId]);

  return status?.threadId === threadId ? status : null;
}
