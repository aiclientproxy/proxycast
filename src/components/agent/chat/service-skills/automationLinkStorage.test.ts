import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ScheduledTask } from "@/lib/api/scheduledTasks";
import {
  buildServiceSkillAutomationStatusMap,
  listServiceSkillAutomationLinks,
  recordServiceSkillAutomationLink,
  resolveServiceSkillAutomationLinks,
  subscribeServiceSkillAutomationLinksChanged,
} from "./automationLinkStorage";

function buildTask(overrides: Partial<ScheduledTask> = {}): ScheduledTask {
  const finishedAt = "2026-03-23T09:00:10.000Z";
  return {
    id: "automation-job-1",
    title: "每日趋势摘要｜定时执行",
    prompt: "prompt",
    enabled: true,
    schedule: {
      type: "daily",
      time: "09:00",
      timezone: "Asia/Shanghai",
    },
    execution: {
      threadMode: "new_thread",
      projectId: "project-1",
      requestMetadata: {
        service_skill: {
          id: "daily-trend-briefing",
          title: "每日趋势摘要",
          runner_type: "scheduled",
        },
      },
    },
    notificationPolicy: "failures",
    overlapPolicy: "skip_if_running",
    nextRunAt: "2026-03-24T09:00:00.000Z",
    lastRunSummary: {
      id: "run-1",
      taskId: "automation-job-1",
      status: "success",
      startedAt: "2026-03-23T09:00:00.000Z",
      finishedAt,
    },
    createdAt: "2026-03-22T09:00:00.000Z",
    updatedAt: finishedAt,
    ...overrides,
  };
}

describe("automationLinkStorage", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  afterEach(() => {
    window.localStorage.clear();
    vi.restoreAllMocks();
  });

  it("记录关联后应能读回最后一次 skill -> job 绑定", () => {
    recordServiceSkillAutomationLink({
      skillId: "daily-trend-briefing",
      jobId: "automation-job-1",
      jobName: "每日趋势摘要｜定时执行",
      linkedAt: 1,
    });
    recordServiceSkillAutomationLink({
      skillId: "daily-trend-briefing",
      jobId: "automation-job-2",
      jobName: "每日趋势摘要｜持续执行",
      linkedAt: 2,
    });

    expect(listServiceSkillAutomationLinks()).toEqual([
      {
        skillId: "daily-trend-briefing",
        jobId: "automation-job-2",
        jobName: "每日趋势摘要｜持续执行",
        linkedAt: 2,
      },
    ]);
  });

  it("变更关联时应广播事件", () => {
    const callback = vi.fn();
    const unsubscribe = subscribeServiceSkillAutomationLinksChanged(callback);

    try {
      recordServiceSkillAutomationLink({
        skillId: "daily-trend-briefing",
        jobId: "automation-job-1",
        jobName: "每日趋势摘要｜定时执行",
      });

      expect(callback).toHaveBeenCalledTimes(1);
    } finally {
      unsubscribe();
    }
  });

  it("应把关联 job 汇总成首页可显示的状态摘要", () => {
    recordServiceSkillAutomationLink({
      skillId: "daily-trend-briefing",
      jobId: "automation-job-1",
      jobName: "每日趋势摘要｜定时执行",
    });

    const statusMap = buildServiceSkillAutomationStatusMap([buildTask()]);

    expect(statusMap["daily-trend-briefing"]).toEqual(
      expect.objectContaining({
        jobId: "automation-job-1",
        statusLabel: "成功",
        tone: "emerald",
      }),
    );
    expect(statusMap["daily-trend-briefing"]?.detail).toContain("下次");
  });

  it("应从任务 requestMetadata 恢复持久化的服务型技能关联", () => {
    const links = resolveServiceSkillAutomationLinks([buildTask()]);

    expect(links).toEqual([
      expect.objectContaining({
        skillId: "daily-trend-briefing",
        jobId: "automation-job-1",
        jobName: "每日趋势摘要｜定时执行",
      }),
    ]);
  });

  it("没有本地 link 时也应根据持久化关联构建首页状态", () => {
    const statusMap = buildServiceSkillAutomationStatusMap([buildTask()]);

    expect(statusMap["daily-trend-briefing"]).toEqual(
      expect.objectContaining({
        jobId: "automation-job-1",
        statusLabel: "成功",
        tone: "emerald",
      }),
    );
  });
});
