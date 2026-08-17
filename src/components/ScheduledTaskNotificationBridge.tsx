import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { isAppServerBridgeAvailable } from "@/lib/api/appServerBridgeAvailability";
import { showDesktopNotification } from "@/lib/api/desktopNotification";
import {
  subscribeScheduledTaskNotifications,
  type ScheduledTaskRunUpdatedNotification,
} from "@/lib/api/scheduledTasks";

const MAX_SEEN_RUNS = 2_048;

export function ScheduledTaskNotificationBridge() {
  const { t } = useTranslation("workspace");
  const seenRunsRef = useRef(new Set<string>());

  useEffect(() => {
    async function deliverDesktopNotification(
      notification: ScheduledTaskRunUpdatedNotification,
    ): Promise<void> {
      const taskTitle = notification.title?.trim() || t("scheduledTasks.title");
      const status = t(statusTranslationKey(notification.status));
      try {
        const result = await showDesktopNotification({
          body: notification.error?.trim()
            ? t("scheduledTasks.systemNotification.bodyWithError", {
                error: notification.error.trim(),
                status,
                title: taskTitle,
              })
            : t("scheduledTasks.systemNotification.body", {
                status,
                title: taskTitle,
              }),
          tag: `scheduled-task:${notification.taskId}:${notification.runId}`,
          title: notification.attention
            ? t("scheduledTasks.systemNotification.attentionTitle")
            : t("scheduledTasks.systemNotification.title"),
        });
        if (result.status !== "sent") {
          showDeliveryError(result.status, result.reason);
        }
      } catch (error) {
        showDeliveryError("failed", errorMessage(error));
      }
    }

    function showDeliveryError(
      status: "failed" | "unsupported",
      reason?: string,
    ): void {
      toast.error(t("scheduledTasks.systemNotification.deliveryTitle"), {
        description: t(
          status === "unsupported"
            ? "scheduledTasks.systemNotification.unsupported"
            : "scheduledTasks.systemNotification.failed",
          { reason: reason?.trim() || t("scheduledTasks.value.notSet") },
        ),
        duration: 12_000,
      });
    }

    return subscribeScheduledTaskNotifications(
      {
        onRunUpdated: (notification) => {
          if (
            !shouldShowDesktopNotification(notification) ||
            seenRunsRef.current.has(notification.runId)
          ) {
            return;
          }
          rememberSeenRun(seenRunsRef.current, notification.runId);
          void deliverDesktopNotification(notification);
        },
      },
      { isBridgeAvailable: isAppServerBridgeAvailable },
    );
  }, [t]);

  return null;
}

function shouldShowDesktopNotification(
  notification: ScheduledTaskRunUpdatedNotification,
): boolean {
  return (
    notification.notificationPolicy === "all_runs" ||
    (notification.notificationPolicy === "failures" && notification.attention)
  );
}

function statusTranslationKey(status: string): string {
  switch (status) {
    case "success":
      return "scheduledTasks.status.completed";
    case "error":
      return "scheduledTasks.status.failed";
    case "canceled":
      return "scheduledTasks.status.canceled";
    case "timeout":
      return "scheduledTasks.status.timeout";
    case "missed":
      return "scheduledTasks.status.missed";
    default:
      return "scheduledTasks.value.notSet";
  }
}

function rememberSeenRun(seenRuns: Set<string>, runId: string): void {
  seenRuns.add(runId);
  while (seenRuns.size > MAX_SEEN_RUNS) {
    const oldest = seenRuns.values().next().value;
    if (oldest === undefined) {
      return;
    }
    seenRuns.delete(oldest);
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
