import { describe, expect, it, vi } from "vitest";
import {
  METHOD_SCHEDULED_TASK_CREATE,
  METHOD_SCHEDULED_TASK_LIST,
  METHOD_SCHEDULED_TASK_RUN_LIST,
  createScheduledTasksApi,
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
    expect(appServer.request).toHaveBeenCalledWith(METHOD_SCHEDULED_TASK_CREATE, {
      task: request,
    });
  });

  it("运行历史形状无效时 fail closed", async () => {
    const api = createScheduledTasksApi(client({ runs: [{ status: "success" }] }));
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
});
