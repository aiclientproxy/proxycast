import type { ScheduledTaskFormState } from "@/components/scheduled-tasks/scheduledTaskViewModel";
import { resolveBaseSetupAutomationProjectionForSkill } from "@/lib/base-setup/automationProjection";
import { buildHarnessRequestMetadata } from "../utils/harnessRequestMetadata";
import { composeServiceSkillAutomationPrompt } from "./promptComposer";
import type {
  ServiceSkillItem,
  ServiceSkillSlotDefinition,
  ServiceSkillSlotValues,
} from "./types";
import { buildServiceSkillWorkspaceSeed } from "./workspaceLaunch";

const DEFAULT_SCHEDULE_INTERVAL_HOURS = 24;
const WEEKDAY_TO_CRON_DAY: Record<string, string> = {
  一: "1",
  二: "2",
  三: "3",
  四: "4",
  五: "5",
  六: "6",
  日: "0",
  天: "0",
};

interface BuildServiceSkillScheduledTaskInitialFormInput {
  skill: ServiceSkillItem;
  slotValues: ServiceSkillSlotValues;
  userInput?: string;
  workspaceId: string;
}

interface BuildServiceSkillAutomationAgentTurnPayloadContextInput {
  skill: ServiceSkillItem;
  slotValues?: ServiceSkillSlotValues;
  userInput?: string;
  contentId?: string | null;
}

function resolveSlotValue(
  slot: ServiceSkillSlotDefinition,
  slotValues: ServiceSkillSlotValues,
): string {
  const currentValue = slotValues[slot.key]?.trim();
  if (currentValue) {
    return currentValue;
  }
  return slot.defaultValue?.trim() || "";
}

function normalizeOptionalText(value?: string | null): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }

  const normalized = value.trim();
  return normalized ? normalized : undefined;
}

function resolveSlotDisplayValue(
  slot: ServiceSkillSlotDefinition,
  slotValues: ServiceSkillSlotValues,
): string {
  const resolved = resolveSlotValue(slot, slotValues);
  if (!resolved) {
    return "";
  }

  const matchedOption = slot.options?.find(
    (option) => option.value === resolved,
  );
  return matchedOption?.label?.trim() || resolved;
}

function summarizeMetadataValue(value: string, maxLength = 120): string {
  const normalized = value.replace(/\s+/g, " ").trim();
  if (normalized.length <= maxLength) {
    return normalized;
  }
  return `${normalized.slice(0, maxLength).trim()}...`;
}

function buildServiceSkillAutomationSlotSummary(
  skill: ServiceSkillItem,
  slotValues: ServiceSkillSlotValues,
): Array<{ key: string; label: string; value: string }> {
  return skill.slotSchema
    .map((slot) => {
      const displayValue = resolveSlotDisplayValue(slot, slotValues);
      if (!displayValue) {
        return null;
      }
      return {
        key: slot.key,
        label: slot.label,
        value: summarizeMetadataValue(displayValue),
      };
    })
    .filter((item): item is { key: string; label: string; value: string } =>
      Boolean(item),
    );
}

function resolveLocalTimeZone(): string {
  if (
    typeof Intl !== "undefined" &&
    typeof Intl.DateTimeFormat === "function"
  ) {
    const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    if (typeof timeZone === "string" && timeZone.trim()) {
      return timeZone;
    }
  }
  return "Asia/Shanghai";
}

function resolveScheduleSlotValue(
  skill: ServiceSkillItem,
  slotValues: ServiceSkillSlotValues,
  preferredSlotKey?: string,
): string {
  if (preferredSlotKey) {
    const currentValue = slotValues[preferredSlotKey]?.trim();
    if (currentValue) {
      return currentValue;
    }
    return "";
  }

  const scheduleSlot = skill.slotSchema.find(
    (slot) => slot.type === "schedule_time",
  );
  if (!scheduleSlot) {
    return "";
  }
  return resolveSlotValue(scheduleSlot, slotValues);
}

type ScheduledTaskSchedulePrefill = Pick<
  ScheduledTaskFormState,
  "scheduleType" | "intervalHours" | "days" | "time" | "timezone"
>;

function buildDefaultSchedulePrefill(): ScheduledTaskSchedulePrefill {
  return {
    scheduleType: "hourly",
    intervalHours: DEFAULT_SCHEDULE_INTERVAL_HOURS,
    days: [],
    time: "00:00",
    timezone: resolveLocalTimeZone(),
  };
}

function buildIntervalPrefill(everySecs: number): ScheduledTaskSchedulePrefill {
  if (
    !Number.isInteger(everySecs) ||
    everySecs <= 0 ||
    everySecs % 3_600 !== 0 ||
    everySecs > 86_400
  ) {
    throw new Error("当前定时任务仅支持 1 到 24 小时的整数间隔");
  }
  return {
    ...buildDefaultSchedulePrefill(),
    intervalHours: everySecs / 3_600,
  };
}

function buildCronPrefill(
  expr: string,
  timezone = resolveLocalTimeZone(),
): ScheduledTaskSchedulePrefill {
  const parts = expr.trim().split(/\s+/);
  const minute = Number(parts[0]);
  const hour = Number(parts[1]);
  if (
    parts.length !== 5 ||
    !Number.isInteger(minute) ||
    minute < 0 ||
    minute > 59 ||
    !Number.isInteger(hour) ||
    hour < 0 ||
    hour > 23 ||
    parts[2] !== "*" ||
    parts[3] !== "*"
  ) {
    throw new Error("当前定时任务仅支持按时间和星期设置 Cron");
  }
  const time = `${String(hour).padStart(2, "0")}:${String(minute).padStart(2, "0")}`;
  const base = {
    intervalHours: 1,
    time,
    timezone,
  };
  if (parts[4] === "*") {
    return { ...base, scheduleType: "daily", days: [] };
  }
  if (parts[4] === "1-5") {
    return { ...base, scheduleType: "weekdays", days: [] };
  }
  const weekdayMap = {
    "0": "SU",
    "1": "MO",
    "2": "TU",
    "3": "WE",
    "4": "TH",
    "5": "FR",
    "6": "SA",
  } as const;
  const days = parts[4]
    ?.split(",")
    .map((day) => weekdayMap[day as keyof typeof weekdayMap])
    .filter((day): day is NonNullable<typeof day> => Boolean(day));
  if (!days?.length || new Set(days).size !== days.length) {
    throw new Error("当前定时任务不支持该 Cron 星期范围");
  }
  return { ...base, scheduleType: "weekly", days };
}

function parseScheduleTextToPrefill(
  rawValue: string,
): ScheduledTaskSchedulePrefill {
  const value = rawValue.trim();
  if (!value) {
    return buildDefaultSchedulePrefill();
  }

  const everyMatch = value.match(/^每\s*(\d+)\s*(秒|分钟|分|小时|时)$/);
  if (everyMatch) {
    const amount = Number(everyMatch[1]);
    const unit = everyMatch[2];
    if (Number.isFinite(amount) && amount > 0) {
      const everySecs =
        unit === "秒"
          ? amount
          : unit === "分钟" || unit === "分"
            ? amount * 60
            : amount * 3_600;
      return buildIntervalPrefill(everySecs);
    }
  }

  const dailyMatch = value.match(/^(?:每天|每日)\s*(\d{1,2}):(\d{2})$/);
  if (dailyMatch) {
    return buildCronPrefill(`${dailyMatch[2]} ${dailyMatch[1]} * * *`);
  }

  const weekdayMatch = value.match(
    /^每周([一二三四五六日天])\s*(\d{1,2}):(\d{2})$/,
  );
  if (weekdayMatch) {
    const cronDay = WEEKDAY_TO_CRON_DAY[weekdayMatch[1]];
    if (cronDay) {
      return buildCronPrefill(
        `${weekdayMatch[3]} ${weekdayMatch[2]} * * ${cronDay}`,
      );
    }
  }

  const weekdayDailyMatch = value.match(
    /^(?:工作日|每个工作日)\s*(\d{1,2}):(\d{2})$/,
  );
  if (weekdayDailyMatch) {
    return buildCronPrefill(
      `${weekdayDailyMatch[2]} ${weekdayDailyMatch[1]} * * 1-5`,
    );
  }

  return buildDefaultSchedulePrefill();
}

function buildServiceSkillAutomationName(skill: ServiceSkillItem): string {
  if (skill.runnerType === "managed") {
    return `${skill.title}｜持续跟踪`;
  }
  if (skill.runnerType === "scheduled") {
    return `${skill.title}｜定时执行`;
  }
  return `${skill.title}｜本地任务`;
}

function buildServiceSkillAutomationMetadata(input: {
  skill: ServiceSkillItem;
  slotValues?: ServiceSkillSlotValues;
  userInput?: string;
}): Record<string, unknown> {
  const { skill, slotValues, userInput } = input;
  const automationProjection =
    resolveBaseSetupAutomationProjectionForSkill(skill);
  const slotSummary = slotValues
    ? buildServiceSkillAutomationSlotSummary(skill, slotValues)
    : [];
  const normalizedUserInput = normalizeOptionalText(userInput);

  return {
    id: skill.id,
    title: skill.title,
    runner_type: skill.runnerType,
    execution_location: skill.executionLocation,
    source: skill.source,
    base_setup:
      automationProjection.refs.packageId ||
      automationProjection.refs.packageVersion ||
      automationProjection.refs.projectionId ||
      automationProjection.refs.automationProfileRef
        ? {
            package_id: automationProjection.refs.packageId ?? null,
            package_version: automationProjection.refs.packageVersion ?? null,
            projection_id: automationProjection.refs.projectionId ?? null,
            automation_profile_ref:
              automationProjection.refs.automationProfileRef ?? null,
          }
        : undefined,
    slot_values: slotSummary,
    slot_summary: slotSummary.map((item) => `${item.label}: ${item.value}`),
    user_input: normalizedUserInput ?? null,
  };
}

function buildServiceSkillAutomationRequestMetadata(input: {
  skill: ServiceSkillItem;
  slotValues?: ServiceSkillSlotValues;
  userInput?: string;
  contentId?: string | null;
}): Record<string, unknown> | undefined {
  const { skill, slotValues, userInput, contentId } = input;
  const targetTheme = skill.themeTarget?.trim();
  const workspaceSeed = buildServiceSkillWorkspaceSeed(skill, targetTheme);

  return {
    ...(workspaceSeed?.requestMetadata ?? {}),
    service_skill: buildServiceSkillAutomationMetadata({
      skill,
      slotValues,
      userInput,
    }),
    harness: buildHarnessRequestMetadata({
      theme: targetTheme || "general",
      preferences: {
        task: false,
        subagent: false,
      },
      sessionMode: workspaceSeed ? "general_workbench" : "default",
      runTitle: skill.title,
      contentId: contentId || undefined,
    }),
  };
}

export function supportsServiceSkillLocalAutomation(
  skill: ServiceSkillItem,
): boolean {
  return (
    skill.executionLocation === "client_default" &&
    skill.runnerType !== "instant"
  );
}

export function buildServiceSkillAutomationAgentTurnPayloadContext({
  skill,
  slotValues,
  userInput,
  contentId,
}: BuildServiceSkillAutomationAgentTurnPayloadContextInput): {
  content_id?: string | null;
  request_metadata?: Record<string, unknown> | null;
} {
  const normalizedContentId = contentId?.trim() || null;
  const requestMetadata = buildServiceSkillAutomationRequestMetadata({
    skill,
    slotValues,
    userInput,
    contentId: normalizedContentId,
  });

  return {
    content_id: normalizedContentId,
    request_metadata: requestMetadata ?? null,
  };
}

export function buildServiceSkillScheduledTaskInitialForm({
  skill,
  slotValues,
  userInput,
  workspaceId,
}: BuildServiceSkillScheduledTaskInitialFormInput): ScheduledTaskFormState {
  const automationProjection =
    resolveBaseSetupAutomationProjectionForSkill(skill);
  const scheduleValue = resolveScheduleSlotValue(
    skill,
    slotValues,
    automationProjection.profile?.schedule?.slotKey,
  );
  const schedulePrefill = scheduleValue.trim()
    ? parseScheduleTextToPrefill(scheduleValue)
    : automationProjection.profile?.schedule?.kind === "every"
      ? buildIntervalPrefill(automationProjection.profile.schedule.everySecs)
      : automationProjection.profile?.schedule?.kind === "cron"
        ? buildCronPrefill(
            automationProjection.profile.schedule.cronExpr,
            automationProjection.profile.schedule.cronTz ||
              resolveLocalTimeZone(),
          )
        : automationProjection.profile?.schedule?.kind === "at"
          ? (() => {
              throw new Error("当前定时任务不支持一次性执行时间");
            })()
          : buildDefaultSchedulePrefill();

  return {
    title: buildServiceSkillAutomationName(skill),
    prompt: composeServiceSkillAutomationPrompt({
      skill,
      slotValues,
      userInput,
    }),
    enabled: automationProjection.profile?.enabledByDefault ?? true,
    threadMode: "continue_thread",
    sourceThreadId: "",
    projectId: workspaceId,
    cwd: "",
    modelId: "",
    reasoningEffort: "",
    notificationPolicy: "failures",
    ...schedulePrefill,
  };
}
