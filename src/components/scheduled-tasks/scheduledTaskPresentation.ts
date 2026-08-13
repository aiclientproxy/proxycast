import { formatDate, formatList } from "@/i18n/format";
import type { TFunction } from "i18next";
import type {
  ScheduledTaskRunSummary,
  ScheduledTaskSchedule,
  ScheduledTaskWeekday,
} from "@/lib/api/scheduledTasks";

export interface ScheduledTaskPresentationCopy {
  hourly: (hours: number, minute: number) => string;
  daily: (time: string) => string;
  weekdays: (time: string) => string;
  weekly: (days: string, time: string) => string;
  weekday: Record<ScheduledTaskWeekday, string>;
  status: Record<string, string>;
  never: string;
}

export function scheduledTaskPresentationCopy(
  t: TFunction<"workspace">,
): ScheduledTaskPresentationCopy {
  return {
    hourly: (hours, minute) =>
      t("scheduledTasks.schedule.hourly", {
        hours,
        minute: String(minute).padStart(2, "0"),
      }),
    daily: (time) => t("scheduledTasks.schedule.daily", { time }),
    weekdays: (time) => t("scheduledTasks.schedule.weekdays", { time }),
    weekly: (days, time) =>
      t("scheduledTasks.schedule.weekly", { days, time }),
    weekday: {
      MO: t("scheduledTasks.weekday.MO"),
      TU: t("scheduledTasks.weekday.TU"),
      WE: t("scheduledTasks.weekday.WE"),
      TH: t("scheduledTasks.weekday.TH"),
      FR: t("scheduledTasks.weekday.FR"),
      SA: t("scheduledTasks.weekday.SA"),
      SU: t("scheduledTasks.weekday.SU"),
    },
    status: {
      enabled: t("scheduledTasks.status.enabled"),
      paused: t("scheduledTasks.status.paused"),
      attention: t("scheduledTasks.status.attention"),
      queued: t("scheduledTasks.status.queued"),
      running: t("scheduledTasks.status.running"),
      waiting_action: t("scheduledTasks.status.waiting"),
      completed: t("scheduledTasks.status.completed"),
      success: t("scheduledTasks.status.completed"),
      failed: t("scheduledTasks.status.failed"),
      error: t("scheduledTasks.status.failed"),
      canceled: t("scheduledTasks.status.canceled"),
      timeout: t("scheduledTasks.status.timeout"),
      missed: t("scheduledTasks.status.missed"),
    },
    never: t("scheduledTasks.value.never"),
  };
}

export function describeScheduledTaskSchedule(
  schedule: ScheduledTaskSchedule,
  copy: ScheduledTaskPresentationCopy,
  locale?: string,
): string {
  switch (schedule.type) {
    case "hourly": {
      const base = copy.hourly(schedule.intervalHours, schedule.minute);
      const days = schedule.days?.map((day) => copy.weekday[day]) ?? [];
      return days.length
        ? `${base} · ${formatList(days, { locale, style: "short" })}`
        : base;
    }
    case "daily":
      return copy.daily(schedule.time);
    case "weekdays":
      return copy.weekdays(schedule.time);
    case "weekly":
      return copy.weekly(
        formatList(schedule.days.map((day) => copy.weekday[day]), {
          locale,
          style: "short",
        }),
        schedule.time,
      );
  }
}

export function scheduledTaskStatusLabel(
  run: ScheduledTaskRunSummary | null | undefined,
  enabled: boolean,
  attention: boolean,
  copy: ScheduledTaskPresentationCopy,
): string {
  if (!enabled) return copy.status.paused;
  if (attention) return copy.status.attention;
  if (run?.status) return copy.status[run.status] ?? run.status;
  return copy.status.enabled;
}

export function formatScheduledTaskTime(
  value: string | null | undefined,
  locale?: string,
): string {
  if (!value) return "-";
  return (
    formatDate(value, {
      locale,
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }) || value
  );
}

export function formatScheduledTaskRunDuration(
  run: ScheduledTaskRunSummary,
): number | null {
  if (!run.startedAt || !run.finishedAt) return null;
  const started = Date.parse(run.startedAt);
  const finished = Date.parse(run.finishedAt);
  if (!Number.isFinite(started) || !Number.isFinite(finished)) return null;
  return Math.max(0, Math.round((finished - started) / 1000));
}
