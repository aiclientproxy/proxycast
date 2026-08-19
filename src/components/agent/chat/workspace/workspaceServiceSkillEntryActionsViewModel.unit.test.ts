import { describe, expect, it } from "vitest";
import type { ScheduledTaskFormState } from "@/components/scheduled-tasks/scheduledTaskViewModel";
import type { ServiceSkillHomeItem } from "../service-skills/types";
import {
  buildServiceSkillAutomationSetupState,
  buildServiceSkillScheduledTaskCreateRequest,
  buildServiceSkillSelectionPlan,
  getWorkspaceServiceSkillErrorMessage,
  normalizeWorkspaceServiceSkillOptionalText,
  resolveServiceSkillLaunchUserInput,
  shouldCreateServiceSkillAutomationContent,
} from "./workspaceServiceSkillEntryActionsViewModel";

function createServiceSkill(
  overrides: Partial<ServiceSkillHomeItem> = {},
): ServiceSkillHomeItem {
  return {
    id: "github-repo-radar",
    title: "GitHub 仓库线索检索",
    summary: "检索主题仓库并沉淀成结构化线索。",
    category: "情报研究",
    outputHint: "仓库列表 + 关键线索",
    source: "cloud_catalog",
    runnerType: "instant",
    defaultExecutorBinding: "agent_turn",
    executionLocation: "client_default",
    defaultArtifactKind: "analysis",
    themeTarget: "general",
    version: "seed-v1",
    slotSchema: [],
    readinessRequirements: {
      requiresProject: true,
    },
    badge: "云目录",
    recentUsedAt: null,
    isRecent: false,
    runnerLabel: "Agent 调度",
    runnerTone: "slate",
    runnerDescription: "通过统一 Agent 发送链执行。",
    actionLabel: "立即启动",
    automationStatus: null,
    ...overrides,
  } as ServiceSkillHomeItem;
}

function createScheduledTaskForm(): ScheduledTaskFormState {
  return {
    title: "每日趋势摘要｜定时执行",
    prompt: "定时任务 prompt",
    enabled: true,
    scheduleType: "daily",
    intervalHours: 1,
    days: [],
    time: "09:00",
    timezone: "Asia/Shanghai",
    threadMode: "continue_thread",
    sourceThreadId: "thread-current",
    projectId: "project-1",
    cwd: "",
    modelId: "",
    reasoningEffort: "",
    notificationPolicy: "failures",
  };
}

describe("workspaceServiceSkillEntryActionsViewModel", () => {
  it("应把未知异常归一为可展示错误文案", () => {
    expect(getWorkspaceServiceSkillErrorMessage(new Error("连接失败"))).toBe(
      "连接失败",
    );
    expect(getWorkspaceServiceSkillErrorMessage("权限不足")).toBe("权限不足");
    expect(getWorkspaceServiceSkillErrorMessage({ code: "UNKNOWN" })).toBe(
      "请稍后重试",
    );
  });

  it("应把服务技能表单与 metadata 构建为 current Scheduled Task", () => {
    const scheduledTask = buildServiceSkillScheduledTaskCreateRequest({
      form: createScheduledTaskForm(),
      contentId: null,
      pendingAutomation: {
        skill: createServiceSkill(),
        prompt: "自动化 prompt",
        slotValues: {},
        threadLineage: {
          sessionId: "session-current",
          threadId: "thread-current",
        },
        usage: {
          skillId: "github-repo-radar",
          runnerType: "instant",
          slotValues: {},
        },
      },
    });

    expect(scheduledTask).toMatchObject({
      title: "每日趋势摘要｜定时执行",
      schedule: {
        type: "daily",
        time: "09:00",
        timezone: "Asia/Shanghai",
      },
      execution: {
        threadMode: "continue_thread",
        sourceThreadId: "thread-current",
        projectId: "project-1",
      },
    });
  });

  it("应归一化入口输入，并让显式 launchUserInput 覆盖当前输入", () => {
    expect(normalizeWorkspaceServiceSkillOptionalText("  继续分析  ")).toBe(
      "继续分析",
    );
    expect(normalizeWorkspaceServiceSkillOptionalText("   ")).toBeUndefined();
    expect(resolveServiceSkillLaunchUserInput(" 当前输入 ")).toBe("当前输入");
    expect(
      resolveServiceSkillLaunchUserInput(" 当前输入 ", {
        launchUserInput: " 显式输入 ",
      }),
    ).toBe("显式输入");
    expect(
      resolveServiceSkillLaunchUserInput(" 当前输入 ", {
        launchUserInput: null,
      }),
    ).toBeUndefined();
  });

  it("服务技能是否需要项目由 readinessRequirements 声明", () => {
    expect(createServiceSkill().readinessRequirements?.requiresProject).toBe(
      true,
    );
    expect(
      createServiceSkill({
        readinessRequirements: undefined,
      }).readinessRequirements?.requiresProject,
    ).toBeUndefined();
    expect(
      createServiceSkill({
        readinessRequirements: {
          requiresProject: false,
        },
      }).readinessRequirements?.requiresProject,
    ).toBe(false);
  });

  it("技能参数齐全时应生成直接启动计划", () => {
    const skill = createServiceSkill({
      slotSchema: [
        {
          key: "repository_query",
          label: "检索主题",
          type: "text",
          required: true,
          placeholder: "例如 browser assist mcp",
        },
      ],
    });

    const plan = buildServiceSkillSelectionPlan({
      skill,
      options: {
        initialSlotValues: {
          repository_query: "browser assist mcp",
        },
        launchUserInput: " 优先看最近 30 天 ",
      },
      nextRequestCount: 3,
    });

    expect(plan).toEqual({
      kind: "launch",
      slotValues: {
        repository_query: "browser assist mcp",
      },
      launchUserInput: "优先看最近 30 天",
    });
  });

  it("技能缺少必填参数时应生成挂起 A2UI 补参计划", () => {
    const skill = createServiceSkill({
      slotSchema: [
        {
          key: "repository_query",
          label: "检索主题",
          type: "text",
          required: true,
          placeholder: "例如 browser assist mcp",
        },
      ],
    });

    const plan = buildServiceSkillSelectionPlan({
      skill,
      options: {
        requestKey: 20260409,
        initialSlotValues: {
          repository_query: "",
        },
        prefillHint: "已根据 Skills 页入口推荐自动预填。",
      },
      nextRequestCount: 3,
    });

    expect(plan).toMatchObject({
      kind: "pending",
      pendingInput: {
        requestKey: "github-repo-radar:20260409",
        skill,
        initialSlotValues: {
          repository_query: "",
        },
        prefillHint: "已根据 Skills 页入口推荐自动预填。",
      },
    });
  });

  it("挂起补参计划缺少显式 requestKey 时应使用下一次请求计数", () => {
    const skill = createServiceSkill({
      slotSchema: [
        {
          key: "repository_query",
          label: "检索主题",
          type: "text",
          required: true,
          placeholder: "例如 browser assist mcp",
        },
      ],
    });

    const plan = buildServiceSkillSelectionPlan({
      skill,
      nextRequestCount: 8,
    });

    expect(plan).toMatchObject({
      kind: "pending",
      pendingInput: {
        requestKey: "github-repo-radar:8",
      },
    });
  });

  it("应构造本地自动化 setup 的初始值和 pending usage", () => {
    const skill = createServiceSkill({
      id: "daily-trend-briefing",
      title: "每日趋势摘要",
      summary: "围绕指定平台与关键词输出趋势摘要。",
      runnerType: "scheduled",
      defaultExecutorBinding: "automation_job",
      slotSchema: [
        {
          key: "platform",
          label: "监测平台",
          type: "platform",
          required: true,
          placeholder: "选择平台",
          defaultValue: "x",
          options: [{ value: "x", label: "X / Twitter" }],
        },
        {
          key: "industry_keywords",
          label: "行业关键词",
          type: "textarea",
          required: true,
          placeholder: "输入关键词",
        },
        {
          key: "schedule_time",
          label: "推送时间",
          type: "schedule_time",
          required: false,
          placeholder: "例如 每天 09:00",
          defaultValue: "每天 09:00",
        },
      ],
      siteCapabilityBinding: undefined,
    });
    const slotValues = {
      platform: "x",
      industry_keywords: "AI Agent，创作者工具",
      schedule_time: "每天 09:00",
    };

    const state = buildServiceSkillAutomationSetupState({
      skill,
      slotValues,
      input: "  请重点看最近 30 天  ",
      workspaceId: "project-1",
      threadLineage: {
        sessionId: "session-current",
        threadId: "thread-current",
      },
    });

    expect(state.pendingAutomation).toMatchObject({
      skill,
      slotValues,
      userInput: "请重点看最近 30 天",
      usage: {
        skillId: "daily-trend-briefing",
        runnerType: "scheduled",
        slotValues,
      },
    });
    expect(state.pendingAutomation.prompt).toContain("请重点看最近 30 天");
    expect(state.dialogInitialValues).toMatchObject({
      title: "每日趋势摘要｜定时执行",
      projectId: "project-1",
      threadMode: "continue_thread",
      sourceThreadId: "thread-current",
      scheduleType: "daily",
      time: "09:00",
    });
  });

  it("应规划自动化提交时的主稿创建和 agent_turn payload metadata", () => {
    const skill = createServiceSkill({
      id: "daily-trend-briefing",
      title: "每日趋势摘要",
      summary: "围绕指定平台与关键词输出趋势摘要。",
      runnerType: "scheduled",
      defaultExecutorBinding: "automation_job",
      slotSchema: [
        {
          key: "industry_keywords",
          label: "行业关键词",
          type: "textarea",
          required: true,
          placeholder: "输入关键词",
        },
      ],
      siteCapabilityBinding: undefined,
    });
    const slotValues = {
      industry_keywords: "AI Agent，创作者工具",
    };
    const setupState = buildServiceSkillAutomationSetupState({
      skill,
      slotValues,
      input: "请重点看最近 30 天",
      workspaceId: "project-1",
      threadLineage: {
        sessionId: "session-current",
        threadId: "thread-current",
      },
    });
    expect(
      shouldCreateServiceSkillAutomationContent({
        pendingAutomation: setupState.pendingAutomation,
        contentId: null,
      }),
    ).toBe(true);
    expect(
      shouldCreateServiceSkillAutomationContent({
        pendingAutomation: setupState.pendingAutomation,
        contentId: "content-current",
      }),
    ).toBe(false);

    const request = buildServiceSkillScheduledTaskCreateRequest({
      pendingAutomation: setupState.pendingAutomation,
      form: setupState.dialogInitialValues,
      contentId: "content-current",
    });

    expect(request.execution).toMatchObject({
      threadMode: "continue_thread",
      sourceThreadId: "thread-current",
      requestMetadata: {
        service_skill: expect.objectContaining({
          id: "daily-trend-briefing",
          title: "每日趋势摘要",
          runner_type: "scheduled",
        }),
      },
    });
  });
});
