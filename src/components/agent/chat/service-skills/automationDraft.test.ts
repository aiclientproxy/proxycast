import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createStoredBaseSetupPackageSnapshot } from "@/lib/base-setup/storage";
import type {
  BaseSetupPackage,
  BaseSetupProjectionIndex,
} from "@/lib/base-setup/types";
import {
  buildServiceSkillAutomationAgentTurnPayloadContext,
  buildServiceSkillScheduledTaskInitialForm,
  supportsServiceSkillLocalAutomation,
} from "./automationDraft";
import type { ServiceSkillItem } from "./types";

const SCHEDULED_SKILL: ServiceSkillItem = {
  id: "daily-trend-briefing",
  title: "每日趋势摘要",
  summary: "围绕指定平台与关键词输出趋势摘要。",
  category: "内容运营",
  outputHint: "趋势摘要 + 调度建议",
  source: "cloud_catalog",
  runnerType: "scheduled",
  defaultExecutorBinding: "automation_job",
  executionLocation: "client_default",
  defaultArtifactKind: "analysis",
  themeTarget: "general",
  version: "seed-v1",
  skillBundle: {
    name: "daily-trend-briefing",
    description: "seeded skill",
    resourceSummary: {
      hasScripts: false,
      hasReferences: false,
      hasAssets: false,
    },
    standardCompliance: {
      isStandard: true,
    },
    metadata: {
      Lime_base_setup_package_id: "automation-pack",
      Lime_base_setup_package_version: "0.4.0",
      Lime_projection_id: "daily-trend-briefing",
      Lime_automation_profile_ref: "trend-automation",
    },
  },
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
      defaultValue: "每天 09:00",
      placeholder: "例如 每天 09:00",
    },
  ],
};

function createBaseSetupPackage(): BaseSetupPackage {
  return {
    id: "automation-pack",
    version: "0.4.0",
    title: "Automation Pack",
    summary: "用于测试 durable automation projection",
    bundleRefs: [
      {
        id: "bundle-1",
        source: "remote",
        pathOrUri: "lime://bundle-1",
        kind: "skill_bundle",
      },
    ],
    catalogProjections: [
      {
        id: "daily-trend-briefing",
        targetCatalog: "service_skill_catalog",
        entryKey: "daily-trend-briefing",
        title: "每日趋势摘要",
        summary: "趋势摘要",
        category: "内容运营",
        outputHint: "趋势摘要",
        bundleRefId: "bundle-1",
        slotProfileRef: "slot-1",
        bindingProfileRef: "binding-1",
        artifactProfileRef: "artifact-1",
        scorecardProfileRef: "scorecard-1",
        policyProfileRef: "policy-1",
        automationProfileRef: "trend-automation",
      },
    ],
    slotProfiles: [
      {
        id: "slot-1",
        slots: [],
      },
    ],
    bindingProfiles: [
      {
        id: "binding-1",
        bindingFamily: "automation_job",
        runnerType: "scheduled",
        executionLocation: "client_default",
      },
    ],
    artifactProfiles: [
      {
        id: "artifact-1",
        deliveryContract: "artifact_bundle",
        requiredParts: ["index.md"],
        viewerKind: "artifact_bundle",
      },
    ],
    scorecardProfiles: [
      {
        id: "scorecard-1",
        metrics: ["success_rate"],
      },
    ],
    automationProfiles: [
      {
        id: "trend-automation",
        enabledByDefault: false,
        schedule: {
          kind: "cron",
          cronExpr: "30 8 * * *",
          cronTz: "Asia/Shanghai",
          slotKey: "schedule_time",
        },
        maxRetries: 4,
        delivery: {
          mode: "announce",
          channel: "local_file",
          target: "reports/daily-trend.md",
          outputSchema: "text",
          outputFormat: "text",
          bestEffort: false,
        },
      },
    ],
    policyProfiles: [
      {
        id: "policy-1",
      },
    ],
    compatibility: {
      minAppVersion: "1.11.0",
      requiredKernelCapabilities: ["automation_job"],
    },
  };
}

function createProjectionIndex(): BaseSetupProjectionIndex {
  return {
    artifactProfileRefsByProjectionId: {
      "daily-trend-briefing": "artifact-1",
    },
    scorecardProfileRefsByProjectionId: {
      "daily-trend-briefing": "scorecard-1",
    },
    policyProfileRefsByProjectionId: {
      "daily-trend-briefing": "policy-1",
    },
    automationProfileRefsByProjectionId: {
      "daily-trend-briefing": "trend-automation",
    },
    compositionBlueprintRefsByProjectionId: {},
  };
}

describe("service skill automation draft", () => {
  beforeEach(() => {
    window.localStorage.clear();
    const snapshot = createStoredBaseSetupPackageSnapshot({
      package: createBaseSetupPackage(),
      projectionIndex: createProjectionIndex(),
      tenantId: "tenant-demo",
      syncedAt: "2026-04-15T12:00:00.000Z",
    });
    window.localStorage.setItem(
      "lime:base-setup-package:v1",
      JSON.stringify(snapshot),
    );
  });

  afterEach(() => {
    window.localStorage.clear();
  });

  it("应识别可转本地自动化的服务型技能", () => {
    expect(supportsServiceSkillLocalAutomation(SCHEDULED_SKILL)).toBe(true);
    expect(
      supportsServiceSkillLocalAutomation({
        ...SCHEDULED_SKILL,
        runnerType: "instant",
      }),
    ).toBe(false);
  });

  it("应把 schedule_time 预填为 Scheduled Task 创建表单", () => {
    const initialValues = buildServiceSkillScheduledTaskInitialForm({
      skill: SCHEDULED_SKILL,
      slotValues: {
        platform: "x",
        industry_keywords: "AI Agent，创作者工具",
        schedule_time: "每天 09:00",
      },
      userInput: "重点关注新增热点与异常波动。",
      workspaceId: "project-1",
    });

    expect(initialValues.title).toContain("每日趋势摘要");
    expect(initialValues.projectId).toBe("project-1");
    expect(initialValues.threadMode).toBe("continue_thread");
    expect(initialValues.scheduleType).toBe("daily");
    expect(initialValues.time).toBe("09:00");
    expect(initialValues.timezone).toBe("Asia/Shanghai");
    expect(initialValues.enabled).toBe(false);
    expect(initialValues.prompt).toContain("[技能任务] 每日趋势摘要");
    expect(initialValues.prompt).toContain("[自动化执行要求]");
  });

  it("应为自动化 agent_turn 生成可复用的 artifact/content payload 上下文", () => {
    const payloadContext = buildServiceSkillAutomationAgentTurnPayloadContext({
      skill: SCHEDULED_SKILL,
      contentId: "content-1",
    });

    expect(payloadContext).toEqual(
      expect.objectContaining({
        content_id: "content-1",
        request_metadata: expect.objectContaining({
          artifact: expect.objectContaining({
            artifact_mode: "draft",
            artifact_kind: "analysis",
          }),
          service_skill: expect.objectContaining({
            id: "daily-trend-briefing",
            title: "每日趋势摘要",
            runner_type: "scheduled",
            base_setup: expect.objectContaining({
              package_id: "automation-pack",
              projection_id: "daily-trend-briefing",
              automation_profile_ref: "trend-automation",
            }),
            slot_values: [],
            slot_summary: [],
            user_input: null,
          }),
          harness: expect.objectContaining({
            theme: "general",
            session_mode: "general_workbench",
            content_id: "content-1",
          }),
        }),
      }),
    );
  });

  it("没有 schedule slot 输入时应回退到 automation profile 预设", () => {
    const initialValues = buildServiceSkillScheduledTaskInitialForm({
      skill: SCHEDULED_SKILL,
      slotValues: {
        platform: "x",
        industry_keywords: "AI Agent",
      },
      workspaceId: "project-1",
    });

    expect(initialValues.scheduleType).toBe("daily");
    expect(initialValues.time).toBe("08:30");
    expect(initialValues.timezone).toBe("Asia/Shanghai");
  });
});
