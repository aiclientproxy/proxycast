import type {
  ScheduledTask,
  ScheduledTaskCreateRequest,
  ScheduledTaskSchedule,
  ScheduledTaskSummary,
  ScheduledTaskUpdateRequest,
  ScheduledTaskWeekday,
} from "@/lib/api/scheduledTasks";

export type ScheduledTaskFilter = "all" | "enabled" | "paused";

export interface ScheduledTaskFormState {
  title: string;
  prompt: string;
  enabled: boolean;
  scheduleType: ScheduledTaskSchedule["type"];
  intervalHours: number;
  days: ScheduledTaskWeekday[];
  time: string;
  timezone: string;
  threadMode: "new_thread" | "continue_thread";
  sourceThreadId: string;
  projectId: string;
  cwd: string;
  modelId: string;
  reasoningEffort: string;
  notificationPolicy: ScheduledTask["notificationPolicy"];
}

export interface ScheduledTaskFormErrors {
  title?: string;
  prompt?: string;
  time?: string;
  days?: string;
  intervalHours?: string;
  timezone?: string;
  sourceThreadId?: string;
}

export function defaultScheduledTaskForm(
  timezone = resolveSystemTimezone(),
): ScheduledTaskFormState {
  return {
    title: "",
    prompt: "",
    enabled: true,
    scheduleType: "weekdays",
    intervalHours: 1,
    days: ["MO", "TU", "WE", "TH", "FR"],
    time: "09:00",
    timezone,
    threadMode: "new_thread",
    sourceThreadId: "",
    projectId: "",
    cwd: "",
    modelId: "",
    reasoningEffort: "",
    notificationPolicy: "failures",
  };
}

export function scheduledTaskToForm(task: ScheduledTask): ScheduledTaskFormState {
  const schedule = task.schedule;
  return {
    title: task.title,
    prompt: task.prompt,
    enabled: task.enabled,
    scheduleType: schedule.type,
    intervalHours: schedule.type === "hourly" ? schedule.intervalHours : 1,
    days:
      schedule.type === "weekly" || schedule.type === "hourly"
        ? [...(schedule.days ?? [])]
        : ["MO", "TU", "WE", "TH", "FR"],
    time:
      schedule.type === "daily" ||
      schedule.type === "weekdays" ||
      schedule.type === "weekly"
        ? schedule.time
        : `00:${String(schedule.minute).padStart(2, "0")}`,
    timezone: schedule.timezone,
    threadMode: task.execution.threadMode,
    sourceThreadId: task.execution.sourceThreadId ?? "",
    projectId: task.execution.projectId ?? "",
    cwd: task.execution.cwd ?? "",
    modelId: task.execution.modelId ?? "",
    reasoningEffort: task.execution.reasoningEffort ?? "",
    notificationPolicy: task.notificationPolicy,
  };
}

export function validateScheduledTaskForm(
  form: ScheduledTaskFormState,
): ScheduledTaskFormErrors {
  const errors: ScheduledTaskFormErrors = {};
  const title = form.title.trim();
  const prompt = form.prompt.trim();
  if (!title || title.length > 80) errors.title = "title";
  if (!prompt || prompt.length > 20_000) errors.prompt = "prompt";
  if (!/^([01]\d|2[0-3]):[0-5]\d$/.test(form.time)) errors.time = "time";
  if (!form.timezone.trim()) errors.timezone = "timezone";
  if (form.scheduleType === "weekly" && form.days.length === 0) {
    errors.days = "days";
  }
  if (
    form.scheduleType === "hourly" &&
    (!Number.isInteger(form.intervalHours) ||
      form.intervalHours < 1 ||
      form.intervalHours > 24)
  ) {
    errors.intervalHours = "intervalHours";
  }
  if (form.threadMode === "continue_thread" && !form.sourceThreadId.trim()) {
    errors.sourceThreadId = "sourceThreadId";
  }
  return errors;
}

export function buildScheduledTaskSchedule(
  form: ScheduledTaskFormState,
): ScheduledTaskSchedule {
  const timezone = form.timezone.trim();
  switch (form.scheduleType) {
    case "hourly":
      return {
        type: "hourly",
        intervalHours: form.intervalHours,
        days: form.days.length ? uniqueWeekdays(form.days) : null,
        minute: readMinute(form.time),
        timezone,
      };
    case "daily":
      return { type: "daily", time: form.time, timezone };
    case "weekdays":
      return { type: "weekdays", time: form.time, timezone };
    case "weekly":
      return {
        type: "weekly",
        days: uniqueWeekdays(form.days),
        time: form.time,
        timezone,
      };
  }
}

export function buildScheduledTaskCreateRequest(
  form: ScheduledTaskFormState,
): ScheduledTaskCreateRequest {
  return {
    title: form.title.trim(),
    prompt: form.prompt.trim(),
    enabled: form.enabled,
    schedule: buildScheduledTaskSchedule(form),
    execution: {
      threadMode: form.threadMode,
      sourceThreadId: optionalText(form.sourceThreadId),
      projectId: optionalText(form.projectId),
      cwd: optionalText(form.cwd),
      modelId: optionalText(form.modelId),
      reasoningEffort: optionalText(form.reasoningEffort),
    },
    notificationPolicy: form.notificationPolicy,
    overlapPolicy: "skip_if_running",
  };
}

export function buildScheduledTaskUpdateRequest(
  form: ScheduledTaskFormState,
  revision: string,
): ScheduledTaskUpdateRequest {
  return { ...buildScheduledTaskCreateRequest(form), revision };
}

export function filterScheduledTasks(
  tasks: ScheduledTaskSummary[],
  query: string,
  filter: ScheduledTaskFilter,
): ScheduledTaskSummary[] {
  const normalized = query.trim().toLocaleLowerCase();
  return tasks.filter((task) => {
    const matchesFilter =
      filter === "all" ||
      (filter === "enabled" ? task.enabled : !task.enabled);
    return (
      matchesFilter &&
      (!normalized || task.title.toLocaleLowerCase().includes(normalized))
    );
  });
}

export function toggleScheduledTaskWeekday(
  days: ScheduledTaskWeekday[],
  day: ScheduledTaskWeekday,
): ScheduledTaskWeekday[] {
  return days.includes(day)
    ? days.filter((candidate) => candidate !== day)
    : uniqueWeekdays([...days, day]);
}

function uniqueWeekdays(days: ScheduledTaskWeekday[]): ScheduledTaskWeekday[] {
  const order: ScheduledTaskWeekday[] = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"];
  const selected = new Set(days);
  return order.filter((day) => selected.has(day));
}

function readMinute(value: string): number {
  return Number(value.split(":")[1] ?? 0);
}

function optionalText(value: string): string | null {
  return value.trim() || null;
}

function resolveSystemTimezone(): string {
  return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
}
