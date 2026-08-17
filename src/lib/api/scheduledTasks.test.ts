import { describe, expect, it, vi } from "vitest";
import type { AppServerEventBusSubscription } from "./appServerEventBus";
import {
  METHOD_SCHEDULED_TASK_CREATE,
  METHOD_SCHEDULED_TASK_LIST,
  METHOD_SCHEDULED_TASK_RUN_LIST,
  createScheduledTasksApi,
  readScheduledTaskChangedNotification,
  readScheduledTaskRunUpdatedNotification,
  subscribeScheduledTaskNotifications,
  type ScheduledTask,
} from "./scheduledTasks";

function sampleTask(): ScheduledTask {
  return {
    id: "task-1",
    title: "每日简报",
    prompt: "整理今天的重要进展",
    enabled: true,
    schedule: {
      type: "weekdays",
      time: "08:30",
      timezone: "Asia/Shanghai",
    },
    execution: { threadMode: "new_thread", projectId: "project-1" },
    notificationPolicy: "failures",
    overlapPolicy: "skip_if_running",
    nextRunAt: "2026-08-14T00:30:00Z",
    createdAt: "2026-08-13T00:00:00Z",
    updatedAt: "2026-08-13T00:00:00Z",
  };
}

function client(result: unknown) {
  return { request: vi.fn().mockResolvedValue({ result }) };
}

describe("scheduledTasks gateway", () => {
  it("通过 exact App Server method 读取任务目录", async () => {
    const summary = { ...sampleTask(), attention: false };
    const appServer = client({ items: [summary], nextCursor: null });
    const api = createScheduledTasksApi(appServer);

    await expect(api.list({ query: "简报", enabled: true })).resolves.toEqual({
      items: [summary],
      nextCursor: null,
    });
    expect(appServer.request).toHaveBeenCalledWith(METHOD_SCHEDULED_TASK_LIST, {
      query: "简报",
      enabled: true,
    });
  });

  it("创建时保留 typed task envelope", async () => {
    const appServer = client({ task: sampleTask() });
    const api = createScheduledTasksApi(appServer);
    const request = {
      title: "每日简报",
      prompt: "整理今天的重要进展",
      enabled: true,
      schedule: sampleTask().schedule,
      execution: sampleTask().execution,
    } as const;

    await expect(api.create(request)).resolves.toEqual(sampleTask());
    expect(appServer.request).toHaveBeenCalledWith(
      METHOD_SCHEDULED_TASK_CREATE,
      {
        task: request,
      },
    );
  });

  it("listDetailed 应从 current read method 补齐 execution metadata", async () => {
    const task = {
      ...sampleTask(),
      execution: {
        ...sampleTask().execution,
        requestMetadata: { harness: { service_skill: { id: "daily" } } },
      },
    };
    const appServer = {
      request: vi
        .fn()
        .mockResolvedValueOnce({
          result: { items: [{ ...task, attention: false }], nextCursor: null },
        })
        .mockResolvedValueOnce({ result: { task } }),
    };

    await expect(
      createScheduledTasksApi(appServer).listDetailed({ limit: 20 }),
    ).resolves.toEqual([task]);
  });

  it("read 应拒绝非对象 requestMetadata", async () => {
    const task = {
      ...sampleTask(),
      execution: {
        ...sampleTask().execution,
        requestMetadata: "legacy",
      },
    };
    await expect(
      createScheduledTasksApi(client({ task })).read(task.id),
    ).rejects.toThrow("returned an invalid task");
  });

  it("运行历史形状无效时 fail closed", async () => {
    const api = createScheduledTasksApi(
      client({ runs: [{ status: "success" }] }),
    );
    await expect(api.listRuns("task-1")).rejects.toThrow(
      `${METHOD_SCHEDULED_TASK_RUN_LIST} returned an invalid run`,
    );
  });

  it("拒绝空任务 identity", async () => {
    const appServer = client({ task: sampleTask() });
    const api = createScheduledTasksApi(appServer);
    await expect(api.read("  ")).rejects.toThrow("id must not be empty");
    expect(appServer.request).not.toHaveBeenCalled();
  });

  it("只投影严格的 typed Scheduled Task notifications", () => {
    const onChanged = vi.fn();
    const onRunUpdated = vi.fn();
    const unsubscribe = vi.fn();
    let eventSubscription: AppServerEventBusSubscription | undefined;

    const dispose = subscribeScheduledTaskNotifications(
      { onChanged, onRunUpdated },
      {
        subscribeNotifications: (subscription) => {
          eventSubscription = subscription;
          return unsubscribe;
        },
      },
    );
    eventSubscription?.onNotifications?.([
      {
        method: "scheduledTask/changed",
        params: { change: "updated", taskId: "task-1" },
      },
      {
        method: "scheduledTask/changed",
        params: { change: "updated", taskId: "task-1" },
      },
      {
        method: "scheduledTask/run/updated",
        params: {
          attention: true,
          error: "provider unavailable",
          notificationPolicy: "failures",
          runId: "run-1",
          status: "error",
          taskId: "task-1",
          title: "每日简报",
        },
      },
      {
        method: "scheduledTask/run/updated",
        params: {
          attention: false,
          notificationPolicy: "all_runs",
          runId: "run-running",
          status: "running",
          taskId: "task-1",
        },
      },
    ]);

    expect(onChanged).toHaveBeenCalledOnce();
    expect(onChanged).toHaveBeenCalledWith({
      change: "updated",
      taskId: "task-1",
    });
    expect(onRunUpdated).toHaveBeenCalledOnce();
    expect(onRunUpdated).toHaveBeenCalledWith(
      expect.objectContaining({ runId: "run-1", status: "error" }),
    );
    dispose();
    expect(unsubscribe).toHaveBeenCalledOnce();
  });

  it("notification reader 对额外字段 fail closed", () => {
    expect(
      readScheduledTaskChangedNotification({
        method: "scheduledTask/changed",
        params: { change: "created", taskId: "task-1", source: "legacy" },
      }),
    ).toBeNull();
    expect(
      readScheduledTaskRunUpdatedNotification({
        method: "scheduledTask/run/updated",
        params: {
          attention: false,
          notificationPolicy: "all_runs",
          runId: "run-1",
          status: "running",
          taskId: "task-1",
        },
      }),
    ).toBeNull();
  });
});
