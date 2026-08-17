import type { ScheduledTask } from "@/lib/api/scheduledTasks";
import type {
  ServiceSkillAutomationLinkRecord,
  ServiceSkillAutomationStatus,
} from "./types";

const SERVICE_SKILL_AUTOMATION_LINKS_STORAGE_KEY =
  "lime:service-skill-automation-links:v1";
export const SERVICE_SKILL_AUTOMATION_LINKS_CHANGED_EVENT =
  "lime:service-skill-automation-links-changed";

function hasWindow(): boolean {
  return typeof window !== "undefined";
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function normalizeNonEmptyText(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const normalized = value.trim();
  return normalized || null;
}

function parseLinkedAt(value?: string | null): number {
  if (!value) {
    return 0;
  }
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function isValidAutomationLinkRecord(
  value: unknown,
): value is ServiceSkillAutomationLinkRecord {
  if (!value || typeof value !== "object") {
    return false;
  }

  const record = value as Partial<ServiceSkillAutomationLinkRecord>;
  return (
    typeof record.skillId === "string" &&
    record.skillId.length > 0 &&
    typeof record.jobId === "string" &&
    record.jobId.length > 0 &&
    typeof record.jobName === "string" &&
    record.jobName.length > 0 &&
    typeof record.linkedAt === "number" &&
    Number.isFinite(record.linkedAt)
  );
}

function emitAutomationLinksChanged(): void {
  if (!hasWindow()) {
    return;
  }

  window.dispatchEvent(
    new CustomEvent(SERVICE_SKILL_AUTOMATION_LINKS_CHANGED_EVENT, {
      detail: {
        timestamp: Date.now(),
      },
    }),
  );
}

function persistAutomationLinks(
  records: ServiceSkillAutomationLinkRecord[],
): ServiceSkillAutomationLinkRecord[] {
  if (!hasWindow()) {
    return records;
  }

  try {
    window.localStorage.setItem(
      SERVICE_SKILL_AUTOMATION_LINKS_STORAGE_KEY,
      JSON.stringify(records),
    );
  } catch {
    // ignore write errors
  }

  emitAutomationLinksChanged();
  return records;
}

function formatAutomationTime(value?: string | null): string | null {
  if (!value) {
    return null;
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
}

function resolveStatusLabel(status?: string | null): string {
  switch (status) {
    case "queued":
      return "排队中";
    case "success":
      return "成功";
    case "running":
      return "运行中";
    case "waiting_for_human":
      return "等待人工处理";
    case "human_controlling":
      return "人工接管中";
    case "agent_resuming":
      return "恢复给 Agent";
    case "error":
      return "失败";
    case "timeout":
      return "超时";
    default:
      return "待执行";
  }
}

function resolveStatusTone(
  status?: string | null,
): ServiceSkillAutomationStatus["tone"] {
  if (status === "success") {
    return "emerald";
  }
  if (
    status === "queued" ||
    status === "running" ||
    status === "agent_resuming"
  ) {
    return "sky";
  }
  if (
    status === "waiting_for_human" ||
    status === "human_controlling" ||
    status === "timeout"
  ) {
    return "amber";
  }
  if (status === "error") {
    return "amber";
  }
  return "slate";
}

function resolveStatusDetail(task: ScheduledTask): string | null {
  const lastRun = task.lastRunSummary;
  if (lastRun?.status === "running" && lastRun.startedAt) {
    const startedAt = formatAutomationTime(lastRun.startedAt);
    return startedAt ? `开始于 ${startedAt}` : null;
  }

  if (lastRun?.status === "success") {
    const nextRunAt = formatAutomationTime(task.nextRunAt);
    const finishedAt = formatAutomationTime(lastRun.finishedAt);
    if (nextRunAt) {
      return `下次 ${nextRunAt}`;
    }
    if (finishedAt) {
      return `完成于 ${finishedAt}`;
    }
    return null;
  }

  if (
    lastRun?.status === "error" ||
    lastRun?.status === "timeout" ||
    lastRun?.status === "missed"
  ) {
    const finishedAt =
      formatAutomationTime(lastRun.finishedAt) ??
      formatAutomationTime(task.updatedAt);
    if (finishedAt) {
      return `最近一次 ${finishedAt}`;
    }
    return null;
  }

  if (task.nextRunAt) {
    const nextRunAt = formatAutomationTime(task.nextRunAt);
    return nextRunAt ? `下次 ${nextRunAt}` : null;
  }

  if (!task.enabled) {
    return "任务已停用";
  }

  return null;
}

function extractPersistedServiceSkillLink(
  task: ScheduledTask,
): ServiceSkillAutomationLinkRecord | null {
  const requestMetadata = task.execution.requestMetadata;
  if (!isPlainRecord(requestMetadata)) {
    return null;
  }

  const serviceSkillValue =
    requestMetadata.service_skill ?? requestMetadata.serviceSkill;
  if (!isPlainRecord(serviceSkillValue)) {
    return null;
  }

  const skillId =
    normalizeNonEmptyText(serviceSkillValue.id) ??
    normalizeNonEmptyText(serviceSkillValue.skill_id) ??
    normalizeNonEmptyText(serviceSkillValue.skillId);
  if (!skillId) {
    return null;
  }

  return {
    skillId,
    jobId: task.id,
    jobName: task.title,
    linkedAt: parseLinkedAt(task.updatedAt) || parseLinkedAt(task.createdAt),
  };
}

export function resolveServiceSkillAutomationLinks(
  tasks: readonly ScheduledTask[],
): ServiceSkillAutomationLinkRecord[] {
  const merged = new Map<string, ServiceSkillAutomationLinkRecord>();

  tasks.forEach((task) => {
    const persistedLink = extractPersistedServiceSkillLink(task);
    if (!persistedLink) {
      return;
    }

    const current = merged.get(persistedLink.skillId);
    if (!current || persistedLink.linkedAt >= current.linkedAt) {
      merged.set(persistedLink.skillId, persistedLink);
    }
  });

  listServiceSkillAutomationLinks().forEach((link) => {
    const current = merged.get(link.skillId);
    if (!current || link.linkedAt > current.linkedAt) {
      merged.set(link.skillId, link);
    }
  });

  return [...merged.values()].sort(
    (left, right) => right.linkedAt - left.linkedAt,
  );
}

export function listServiceSkillAutomationLinks(): ServiceSkillAutomationLinkRecord[] {
  if (!hasWindow()) {
    return [];
  }

  try {
    const raw = window.localStorage.getItem(
      SERVICE_SKILL_AUTOMATION_LINKS_STORAGE_KEY,
    );
    if (!raw) {
      return [];
    }

    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }

    return parsed
      .filter(isValidAutomationLinkRecord)
      .sort((left, right) => right.linkedAt - left.linkedAt);
  } catch {
    return [];
  }
}

export function recordServiceSkillAutomationLink(
  input: Omit<ServiceSkillAutomationLinkRecord, "linkedAt"> & {
    linkedAt?: number;
  },
): ServiceSkillAutomationLinkRecord[] {
  const nextRecord: ServiceSkillAutomationLinkRecord = {
    skillId: input.skillId,
    jobId: input.jobId,
    jobName: input.jobName,
    linkedAt: input.linkedAt ?? Date.now(),
  };

  const nextRecords = [
    nextRecord,
    ...listServiceSkillAutomationLinks().filter(
      (record) => record.skillId !== nextRecord.skillId,
    ),
  ];

  return persistAutomationLinks(nextRecords);
}

export function subscribeServiceSkillAutomationLinksChanged(
  callback: () => void,
): () => void {
  if (!hasWindow()) {
    return () => undefined;
  }

  const customEventHandler = () => {
    callback();
  };

  const storageHandler = (event: StorageEvent) => {
    if (event.key !== SERVICE_SKILL_AUTOMATION_LINKS_STORAGE_KEY) {
      return;
    }
    callback();
  };

  window.addEventListener(
    SERVICE_SKILL_AUTOMATION_LINKS_CHANGED_EVENT,
    customEventHandler,
  );
  window.addEventListener("storage", storageHandler);

  return () => {
    window.removeEventListener(
      SERVICE_SKILL_AUTOMATION_LINKS_CHANGED_EVENT,
      customEventHandler,
    );
    window.removeEventListener("storage", storageHandler);
  };
}

export function buildServiceSkillAutomationStatusMap(
  tasks: readonly ScheduledTask[],
): Record<string, ServiceSkillAutomationStatus> {
  const tasksById = new Map(tasks.map((task) => [task.id, task]));

  return resolveServiceSkillAutomationLinks(tasks).reduce<
    Record<string, ServiceSkillAutomationStatus>
  >((result, link) => {
    const task = tasksById.get(link.jobId);
    if (!task) {
      return result;
    }

    const status = task.lastRunSummary?.status ?? null;

    result[link.skillId] = {
      jobId: task.id,
      jobName: task.title || link.jobName,
      statusLabel: resolveStatusLabel(status),
      tone: resolveStatusTone(status),
      detail: resolveStatusDetail(task),
    };
    return result;
  }, {});
}
