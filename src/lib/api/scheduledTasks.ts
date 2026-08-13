import { AppServerClient } from "@/lib/api/appServer";
import {
  METHOD_SCHEDULED_TASK_CREATE,
  METHOD_SCHEDULED_TASK_DELETE,
  METHOD_SCHEDULED_TASK_ENABLED_SET,
  METHOD_SCHEDULED_TASK_LIST,
  METHOD_SCHEDULED_TASK_READ,
  METHOD_SCHEDULED_TASK_RUN_LIST,
  METHOD_SCHEDULED_TASK_RUN_START,
  METHOD_SCHEDULED_TASK_SCHEDULE_PREVIEW,
  METHOD_SCHEDULED_TASK_UPDATE,
  type ScheduledTask,
  type ScheduledTaskCreateRequest,
  type ScheduledTaskExecution,
  type ScheduledTaskListParams,
  type ScheduledTaskListResponse,
  type ScheduledTaskRunSummary,
  type ScheduledTaskSchedule,
  type ScheduledTaskSchedulePreviewResponse as AppServerScheduledTaskSchedulePreviewResponse,
  type ScheduledTaskSummary,
  type ScheduledTaskUpdateRequest,
  type ScheduledTaskWeekday,
} from "../../../packages/app-server-client/src/protocol";

export {
  METHOD_SCHEDULED_TASK_CREATE,
  METHOD_SCHEDULED_TASK_DELETE,
  METHOD_SCHEDULED_TASK_ENABLED_SET,
  METHOD_SCHEDULED_TASK_LIST,
  METHOD_SCHEDULED_TASK_READ,
  METHOD_SCHEDULED_TASK_RUN_LIST,
  METHOD_SCHEDULED_TASK_RUN_START,
  METHOD_SCHEDULED_TASK_SCHEDULE_PREVIEW,
  METHOD_SCHEDULED_TASK_UPDATE,
};
export type {
  ScheduledTask,
  ScheduledTaskCreateRequest,
  ScheduledTaskExecution,
  ScheduledTaskListParams,
  ScheduledTaskListResponse,
  ScheduledTaskRunSummary,
  ScheduledTaskSchedule,
  ScheduledTaskSummary,
  ScheduledTaskUpdateRequest,
  ScheduledTaskWeekday,
};

export type ScheduledTaskSchedulePreviewResponse =
  AppServerScheduledTaskSchedulePreviewResponse & { warnings: string[] };

type ScheduledTaskAppServerClient = Pick<AppServerClient, "request">;

async function requestScheduledTask<T>(
  method: string,
  params: unknown,
  client: ScheduledTaskAppServerClient,
): Promise<T> {
  const response = await client.request<T>(method, params);
  return response.result;
}

export function createScheduledTasksApi(
  client: ScheduledTaskAppServerClient = new AppServerClient(),
) {
  return {
    async list(
      params: ScheduledTaskListParams = {},
    ): Promise<ScheduledTaskListResponse> {
      const response = await requestScheduledTask<unknown>(
        METHOD_SCHEDULED_TASK_LIST,
        params,
        client,
      );
      const record = requireRecord(response, METHOD_SCHEDULED_TASK_LIST);
      if (!Array.isArray(record.items)) {
        throw new Error(`${METHOD_SCHEDULED_TASK_LIST} did not return items`);
      }
      return {
        items: record.items.map((item) =>
          requireTaskSummary(item, METHOD_SCHEDULED_TASK_LIST),
        ),
        nextCursor:
          typeof record.nextCursor === "string" ? record.nextCursor : null,
      };
    },

    async read(id: string): Promise<ScheduledTask | null> {
      const response = await requestScheduledTask<unknown>(
        METHOD_SCHEDULED_TASK_READ,
        { id: requiredText(id, "id") },
        client,
      );
      const task = requireRecord(response, METHOD_SCHEDULED_TASK_READ).task;
      return task === null || task === undefined
        ? null
        : requireTask(task, METHOD_SCHEDULED_TASK_READ);
    },

    async create(task: ScheduledTaskCreateRequest): Promise<ScheduledTask> {
      return requireWriteTask(
        await requestScheduledTask<unknown>(
          METHOD_SCHEDULED_TASK_CREATE,
          { task },
          client,
        ),
        METHOD_SCHEDULED_TASK_CREATE,
      );
    },

    async update(
      id: string,
      task: ScheduledTaskUpdateRequest,
    ): Promise<ScheduledTask> {
      return requireWriteTask(
        await requestScheduledTask<unknown>(
          METHOD_SCHEDULED_TASK_UPDATE,
          { id: requiredText(id, "id"), task },
          client,
        ),
        METHOD_SCHEDULED_TASK_UPDATE,
      );
    },

    async remove(id: string): Promise<boolean> {
      const response = requireRecord(
        await requestScheduledTask<unknown>(
          METHOD_SCHEDULED_TASK_DELETE,
          { id: requiredText(id, "id") },
          client,
        ),
        METHOD_SCHEDULED_TASK_DELETE,
      );
      if (typeof response.deleted !== "boolean") {
        throw new Error(`${METHOD_SCHEDULED_TASK_DELETE} did not return deleted`);
      }
      return response.deleted;
    },

    async setEnabled(id: string, enabled: boolean): Promise<ScheduledTask> {
      return requireWriteTask(
        await requestScheduledTask<unknown>(
          METHOD_SCHEDULED_TASK_ENABLED_SET,
          { id: requiredText(id, "id"), enabled },
          client,
        ),
        METHOD_SCHEDULED_TASK_ENABLED_SET,
      );
    },

    async startRun(id: string): Promise<ScheduledTaskRunSummary> {
      const response = requireRecord(
        await requestScheduledTask<unknown>(
          METHOD_SCHEDULED_TASK_RUN_START,
          { id: requiredText(id, "id") },
          client,
        ),
        METHOD_SCHEDULED_TASK_RUN_START,
      );
      return requireRun(response.run, METHOD_SCHEDULED_TASK_RUN_START);
    },

    async listRuns(
      taskId: string,
      limit = 20,
    ): Promise<ScheduledTaskRunSummary[]> {
      const response = requireRecord(
        await requestScheduledTask<unknown>(
          METHOD_SCHEDULED_TASK_RUN_LIST,
          { taskId: requiredText(taskId, "taskId"), limit },
          client,
        ),
        METHOD_SCHEDULED_TASK_RUN_LIST,
      );
      if (!Array.isArray(response.runs)) {
        throw new Error(`${METHOD_SCHEDULED_TASK_RUN_LIST} did not return runs`);
      }
      return response.runs.map((run) =>
        requireRun(run, METHOD_SCHEDULED_TASK_RUN_LIST),
      );
    },

    async previewSchedule(
      schedule: ScheduledTaskSchedule,
    ): Promise<ScheduledTaskSchedulePreviewResponse> {
      const response = requireRecord(
        await requestScheduledTask<unknown>(
          METHOD_SCHEDULED_TASK_SCHEDULE_PREVIEW,
          { schedule },
          client,
        ),
        METHOD_SCHEDULED_TASK_SCHEDULE_PREVIEW,
      );
      if (
        !Array.isArray(response.nextRunAt) ||
        !response.nextRunAt.every((value) => typeof value === "string") ||
        !Array.isArray(response.warnings) ||
        !response.warnings.every((value) => typeof value === "string")
      ) {
        throw new Error(
          `${METHOD_SCHEDULED_TASK_SCHEDULE_PREVIEW} returned an invalid preview`,
        );
      }
      return {
        nextRunAt: response.nextRunAt,
        warnings: response.warnings,
      };
    },
  };
}

export const scheduledTasksApi = createScheduledTasksApi();

function requireWriteTask(value: unknown, method: string): ScheduledTask {
  return requireTask(requireRecord(value, method).task, method);
}

function requireTask(value: unknown, method: string): ScheduledTask {
  const task = requireRecord(value, method);
  const execution = requireRecord(task.execution, method);
  if (
    typeof task.id !== "string" ||
    typeof task.title !== "string" ||
    typeof task.prompt !== "string" ||
    typeof task.enabled !== "boolean" ||
    !isSchedule(task.schedule) ||
    typeof task.createdAt !== "string" ||
    typeof task.updatedAt !== "string" ||
    !isThreadMode(execution.threadMode) ||
    !isNotificationPolicy(task.notificationPolicy) ||
    task.overlapPolicy !== "skip_if_running"
  ) {
    throw new Error(`${method} returned an invalid task`);
  }
  return task as unknown as ScheduledTask;
}

function requireTaskSummary(value: unknown, method: string): ScheduledTaskSummary {
  const task = requireRecord(value, method);
  if (
    typeof task.id !== "string" ||
    typeof task.title !== "string" ||
    typeof task.enabled !== "boolean" ||
    typeof task.attention !== "boolean" ||
    !isSchedule(task.schedule)
  ) {
    throw new Error(`${method} returned an invalid task summary`);
  }
  return task as unknown as ScheduledTaskSummary;
}

function requireRun(value: unknown, method: string): ScheduledTaskRunSummary {
  const run = requireRecord(value, method);
  if (
    typeof run.id !== "string" ||
    typeof run.taskId !== "string" ||
    typeof run.status !== "string"
  ) {
    throw new Error(`${method} returned an invalid run`);
  }
  return run as unknown as ScheduledTaskRunSummary;
}

function requireRecord(value: unknown, method: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${method} did not return an object`);
  }
  return value as Record<string, unknown>;
}

function requiredText(value: string, field: string): string {
  const normalized = value.trim();
  if (!normalized) {
    throw new Error(`${field} must not be empty`);
  }
  return normalized;
}

function isSchedule(value: unknown): value is ScheduledTaskSchedule {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return false;
  }
  const type = (value as { type?: unknown }).type;
  return ["hourly", "daily", "weekdays", "weekly"].includes(String(type));
}

function isThreadMode(value: unknown): value is ScheduledTaskExecution["threadMode"] {
  return value === "new_thread" || value === "continue_thread";
}

function isNotificationPolicy(
  value: unknown,
): value is ScheduledTask["notificationPolicy"] {
  return value === "all_runs" || value === "failures" || value === "none";
}
