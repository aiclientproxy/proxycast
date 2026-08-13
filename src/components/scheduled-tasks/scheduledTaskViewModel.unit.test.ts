import { describe, expect, it } from "vitest";
import {
  buildScheduledTaskCreateRequest,
  defaultScheduledTaskForm,
  filterScheduledTasks,
  toggleScheduledTaskWeekday,
  validateScheduledTaskForm,
} from "./scheduledTaskViewModel";

describe("scheduledTaskViewModel", () => {
  it("构建 new thread 创建请求时不伪造来源 lineage", () => {
    const form = {
      ...defaultScheduledTaskForm("Asia/Shanghai"),
      title: " 每日简报 ",
      prompt: " 整理今天的重要进展 ",
      scheduleType: "hourly" as const,
      intervalHours: 2,
      time: "09:15",
      days: ["FR", "MO", "FR"] as const,
      projectId: " project-1 ",
    };

    expect(buildScheduledTaskCreateRequest(form)).toEqual(
      expect.objectContaining({
        title: "每日简报",
        prompt: "整理今天的重要进展",
        schedule: {
          type: "hourly",
          intervalHours: 2,
          days: ["MO", "FR"],
          minute: 15,
          timezone: "Asia/Shanghai",
        },
        execution: expect.objectContaining({
          threadMode: "new_thread",
          sourceThreadId: null,
          projectId: "project-1",
        }),
      }),
    );
  });

  it("continue thread 必须提供来源 Thread", () => {
    const errors = validateScheduledTaskForm({
      ...defaultScheduledTaskForm(),
      title: "每日简报",
      prompt: "整理今天的重要进展",
      threadMode: "continue_thread",
    });
    expect(errors.sourceThreadId).toBe("sourceThreadId");
  });

  it("按查询和启用状态筛选任务", () => {
    const tasks = [
      {
        id: "1",
        title: "每日简报",
        enabled: true,
        attention: false,
        schedule: { type: "daily" as const, time: "09:00", timezone: "UTC" },
      },
      {
        id: "2",
        title: "每周回顾",
        enabled: false,
        attention: false,
        schedule: { type: "daily" as const, time: "18:00", timezone: "UTC" },
      },
    ];
    expect(filterScheduledTasks(tasks, "简报", "enabled").map((item) => item.id)).toEqual([
      "1",
    ]);
    expect(filterScheduledTasks(tasks, "", "paused").map((item) => item.id)).toEqual([
      "2",
    ]);
  });

  it("星期切换保持稳定 wire 顺序", () => {
    expect(toggleScheduledTaskWeekday(["FR", "MO"], "WE")).toEqual([
      "MO",
      "WE",
      "FR",
    ]);
    expect(toggleScheduledTaskWeekday(["MO", "WE", "FR"], "WE")).toEqual([
      "MO",
      "FR",
    ]);
  });
});
