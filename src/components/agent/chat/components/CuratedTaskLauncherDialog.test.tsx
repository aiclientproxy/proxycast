import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CuratedTaskLauncherDialog } from "./CuratedTaskLauncherDialog";
import { findCuratedTaskTemplateById } from "@/components/agent/chat/utils/curatedTaskTemplates";
import { buildCuratedTaskReferenceEntries } from "@/components/agent/chat/utils/curatedTaskReferenceSelection";
import {
  CURATED_TASK_RECOMMENDATION_SIGNAL_EVENT,
  recordCuratedTaskRecommendationSignalFromReviewDecision,
} from "@/components/agent/chat/utils/curatedTaskRecommendationSignals";
import { changeLimeLocale } from "@/i18n/createI18n";

function buildExperienceMemoryReferenceEntry() {
  return buildCuratedTaskReferenceEntries([
    {
      id: "memory-review-1",
      session_id: "session-1",
      memory_type: "conversation",
      category: "experience",
      title: "短视频编排 · 复核阻塞",
      content: [
        "场景：短视频编排",
        "平台：X + TikTok",
        "地区：北美",
        "目标受众：正在复盘短视频增长的品牌运营",
        "结果摘要：这轮内容已经产出一版完整结果包。",
        "当前交付：已交付 3/4 个部件",
        "建议下一步：先完成复核，再决定下一轮放量",
        "当前信号：复核阻塞",
        "当前判断：先补复核与修复",
      ].join("\n"),
      summary: "当前结果包已完整回流，可继续进入下一轮。",
      tags: ["短视频", "复核阻塞"],
      metadata: {
        confidence: 0.9,
        importance: 8,
        access_count: 1,
        last_accessed_at: null,
        source: "manual",
        embedding: null,
      },
      created_at: 1_712_345_670_000,
      updated_at: 1_712_345_678_000,
      archived: false,
    },
  ])[0];
}

function findConfirmButton() {
  return document.body.querySelector(
    '[data-testid="curated-task-launcher-confirm"]',
  ) as HTMLButtonElement | null;
}

describe("CuratedTaskLauncherDialog", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(async () => {
    await changeLimeLocale("zh-CN");

    // React 18 createRoot + act 在当前 Vitest 环境里需要显式打开该标记。
    (
      globalThis as typeof globalThis & {
        IS_REACT_ACT_ENVIRONMENT?: boolean;
      }
    ).IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => {
      root.unmount();
    });
    container.remove();
    window.localStorage.clear();
    document.body.innerHTML = "";
  });

  it("应让启动表单主体成为可滚动区域，避免内容过长时整窗无法下拉", async () => {
    const task = findCuratedTaskTemplateById("daily-trend-briefing");
    expect(task).not.toBeNull();

    await act(async () => {
      root.render(
        <CuratedTaskLauncherDialog
          open
          task={task}
          initialReferenceMemoryIds={["memory:content-pack:run:1"]}
          initialReferenceEntries={[
            {
              id: "memory-internal-error",
              sourceKind: "memory",
              title: "运行异常记录",
              summary: "这条参考来自一次执行异常，默认不展开技术细节。",
              category: "context",
              categoryLabel: "参考",
              tags: ["用户访谈"],
            },
          ]}
          onOpenChange={() => undefined}
          onConfirm={() => undefined}
        />,
      );
    });

    const dialog = document.body.querySelector('[role="dialog"]');
    expect(dialog).not.toBeNull();

    const scrollBody = document.body.querySelector(
      '[data-testid="curated-task-launcher-scroll-body"]',
    );
    expect(scrollBody).not.toBeNull();
    expect(scrollBody?.className).toContain("flex-1");
    expect(scrollBody?.className).toContain("overflow-y-auto");
    expect(dialog?.textContent).toContain("每日趋势摘要");
    expect(dialog?.textContent).toContain("还差 2 项关键信息");
    expect(dialog?.textContent).toContain("开始这一步前，我先确认几件事。");
    expect(dialog?.textContent).toContain("想的话，再带几条参考");
    expect(dialog?.textContent).toContain(
      "风格、偏好、项目结果和当前上下文都可以。",
    );
  });

  it("共享启动弹层应接入工作台主题变量，而不是固定白绿底色", async () => {
    const task = findCuratedTaskTemplateById("daily-trend-briefing");
    expect(task).not.toBeNull();

    await act(async () => {
      root.render(
        <CuratedTaskLauncherDialog
          open
          task={task}
          initialReferenceEntries={[
            {
              id: "memory-internal-error",
              sourceKind: "memory",
              title: "运行异常记录",
              summary: "这条参考来自一次执行异常，默认不展开技术细节。",
              category: "context",
              categoryLabel: "参考",
              tags: ["用户访谈"],
            },
          ]}
          onOpenChange={() => undefined}
          onConfirm={() => undefined}
        />,
      );
    });

    const dialog = document.body.querySelector('[role="dialog"]');
    expect(dialog?.className).toContain("lime-workbench-theme-scope");
    expect(dialog?.className).toContain("lime-workbench-surface-scope");
    expect(dialog?.className).toContain("bg-[color:var(--lime-surface)]");
    expect(dialog?.className).toContain(
      "border-[color:var(--lime-surface-border)]",
    );

    const header = Array.from(dialog?.querySelectorAll("div") ?? []).find(
      (element) =>
        (element as HTMLElement).className.includes(
          "bg-[image:var(--lime-card-subtle)]",
        ),
    ) as HTMLElement | undefined;
    expect(header?.className).toContain(
      "border-[color:var(--lime-surface-border)]",
    );

    const confirmButton = document.body.querySelector(
      '[data-testid="curated-task-launcher-confirm"]',
    ) as HTMLButtonElement | null;
    expect(confirmButton?.className).toContain(
      "bg-[color:var(--lime-brand-strong)]",
    );
    expect(confirmButton?.className).toContain(
      "hover:bg-[color:var(--lime-brand)]",
    );
  });

  it("应把参考对象来源收成轻量元信息，而不是标签墙", async () => {
    const task = findCuratedTaskTemplateById("account-project-review");
    expect(task).not.toBeNull();
    await act(async () => {
      root.render(
        <CuratedTaskLauncherDialog
          open
          task={task}
          initialReferenceEntries={[
            {
              id: "memory:content-pack:run:1",
              sourceKind: "memory",
              title: "AI 内容周报",
              summary: "当前已有一轮项目结果，可直接作为复盘基线。",
              category: "experience",
              categoryLabel: "成果",
              tags: ["复盘"],
            },
            {
              id: "memory-style-1",
              sourceKind: "memory",
              title: "品牌风格样本",
              summary: "偏好克制的科技蓝与留白型构图。",
              category: "identity",
              categoryLabel: "风格",
              tags: ["科技蓝", "留白"],
            },
          ]}
          onOpenChange={() => undefined}
          onConfirm={() => undefined}
        />,
      );
    });

    const dialog = document.body.querySelector('[role="dialog"]');
    expect(dialog?.textContent).toContain("记忆参考 · 成果");
    expect(dialog?.textContent).toContain("记忆参考 · 风格");
    expect(dialog?.textContent).toContain("AI 内容周报");
    expect(dialog?.textContent).toContain("品牌风格样本");
    expect(dialog?.textContent).toContain("相关线索：科技蓝、留白");
  });

  it("启动参考对象不应向普通用户暴露自动记忆内部标签和原始错误", async () => {
    const task = findCuratedTaskTemplateById("daily-trend-briefing");
    expect(task).not.toBeNull();
    await act(async () => {
      root.render(
        <CuratedTaskLauncherDialog
          open
          task={task}
          initialReferenceEntries={[
            {
              id: "memory-internal-error",
              sourceKind: "memory",
              title: "自动分析提取（AI 响应）：Pexels API Key -32603",
              summary: "execution failed: Pexels API Key -32603",
              category: "context",
              categoryLabel: "参考",
              tags: ["auto_analysis", "用户访谈"],
            },
          ]}
          onOpenChange={() => undefined}
          onConfirm={() => undefined}
        />,
      );
      await Promise.resolve();
    });

    const dialog = document.body.querySelector('[role="dialog"]');
    expect(dialog?.textContent).toContain("运行异常记录");
    expect(dialog?.textContent).toContain(
      "这条参考来自一次执行异常，默认不展开技术细节。",
    );
    expect(dialog?.textContent).toContain("相关线索：用户访谈");
    expect(dialog?.textContent).not.toContain("auto_analysis");
    expect(dialog?.textContent).not.toContain("Pexels API Key");
    expect(dialog?.textContent).not.toContain("-32603");
  });

  it("命中最近判断偏好的结果模板时，应在 launcher 内显影判断提示", async () => {
    const task = findCuratedTaskTemplateById("account-project-review");
    expect(task).not.toBeNull();

    recordCuratedTaskRecommendationSignalFromReviewDecision(
      {
        session_id: "session-review-needs-evidence",
        decision_status: "needs_more_evidence",
        decision_summary:
          "这轮结果还缺证据，需要回到账号表现和爆款样本继续补证据。",
        chosen_fix_strategy: "先补账号数据复盘，再拆一轮高表现内容做对照。",
        risk_level: "medium",
        risk_tags: ["证据不足", "需要复盘"],
        followup_actions: ["补账号数据复盘", "拆解一条高表现内容"],
      },
      {
        projectId: "project-review",
        sceneTitle: "短视频编排",
      },
    );

    await act(async () => {
      root.render(
        <CuratedTaskLauncherDialog
          open
          task={task}
          projectId="project-review"
          onOpenChange={() => undefined}
          onConfirm={() => undefined}
        />,
      );
    });

    const banner = document.body.querySelector(
      '[data-testid="curated-task-launcher-review-feedback-banner"]',
    );
    expect(banner?.textContent).toContain("围绕最近判断");
    expect(banner?.textContent).toContain(
      "最近判断已更新：短视频编排 · 补证据",
    );
    expect(banner?.textContent).toContain("这轮结果还缺证据");
  });

  it("当前 launcher 不是最近判断首选时，应允许直接切到推荐模板", async () => {
    const task = findCuratedTaskTemplateById("daily-trend-briefing");
    expect(task).not.toBeNull();
    const onApplyReviewSuggestion = vi.fn();

    recordCuratedTaskRecommendationSignalFromReviewDecision(
      {
        session_id: "session-review-switch",
        decision_status: "needs_more_evidence",
        decision_summary:
          "这轮结果还缺证据，需要回到账号表现和爆款样本继续补证据。",
        chosen_fix_strategy: "先补账号数据复盘，再拆一轮高表现内容做对照。",
        risk_level: "medium",
        risk_tags: ["证据不足", "需要复盘"],
        followup_actions: ["补账号数据复盘", "拆解一条高表现内容"],
      },
      {
        projectId: "project-review-switch",
        sceneTitle: "短视频编排",
      },
    );

    await act(async () => {
      root.render(
        <CuratedTaskLauncherDialog
          open
          task={task}
          projectId="project-review-switch"
          initialReferenceEntries={[buildExperienceMemoryReferenceEntry()]}
          onOpenChange={() => undefined}
          onApplyReviewSuggestion={onApplyReviewSuggestion}
          onConfirm={() => undefined}
        />,
      );
      await Promise.resolve();
    });

    const banner = document.body.querySelector(
      '[data-testid="curated-task-launcher-review-feedback-banner"]',
    );
    const actionButton = document.body.querySelector(
      '[data-testid="curated-task-launcher-review-feedback-banner-action"]',
    ) as HTMLButtonElement | null;

    expect(banner?.textContent).toContain(
      "最近判断已更新：短视频编排 · 补证据",
    );
    expect(banner?.textContent).toContain(
      "这轮判断更建议优先回到「复盘这个账号/项目」",
    );
    expect(actionButton?.textContent).toContain("改用「复盘这个账号/项目」");

    await act(async () => {
      actionButton?.click();
    });

    expect(onApplyReviewSuggestion).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "account-project-review",
        title: "复盘这个账号/项目",
      }),
      expect.objectContaining({
        referenceSelection: expect.objectContaining({
          referenceEntries: expect.arrayContaining([
            expect.objectContaining({
              id: "memory-review-1",
            }),
          ]),
        }),
      }),
    );
  });

  it("成果类灵感 reference 进入下游结果模板时，也应带出基线和默认预填", async () => {
    const task = findCuratedTaskTemplateById("daily-trend-briefing");
    const memoryReferenceEntry = buildExperienceMemoryReferenceEntry();
    expect(task).not.toBeNull();
    expect(memoryReferenceEntry).toBeTruthy();

    await act(async () => {
      root.render(
        <CuratedTaskLauncherDialog
          open
          task={task}
          initialReferenceEntries={
            memoryReferenceEntry ? [memoryReferenceEntry] : []
          }
          onOpenChange={() => undefined}
          onConfirm={() => undefined}
        />,
      );
      await Promise.resolve();
    });

    const baselineCard = document.body.querySelector(
      '[data-testid="curated-task-launcher-result-baseline-card"]',
    );
    const themeField = document.body.querySelector(
      "#curated-task-daily-trend-briefing-theme_target",
    ) as HTMLInputElement | null;
    const platformField = document.body.querySelector(
      "#curated-task-daily-trend-briefing-platform_region",
    ) as HTMLInputElement | null;

    expect(baselineCard?.textContent).toContain("当前结果基线");
    expect(baselineCard?.textContent).toContain("短视频编排 · 复核阻塞");
    expect(baselineCard?.textContent).toContain("当前判断：先补复核与修复");
    expect(baselineCard?.textContent).toContain("当前卡点：复核阻塞");
    expect(themeField?.value).toBe("短视频编排");
    expect(platformField?.value).toBe("X + TikTok（北美）");
    expect(document.body.textContent).toContain(
      "下面的 主题或赛道 / 希望关注的平台/地域 已按这轮结果自动带入，你可以直接改成这次真正想推进的版本。",
    );
  });

  it("成果类灵感 reference 进入内容主稿模板时，也应默认带好主题基线", async () => {
    const task = findCuratedTaskTemplateById("social-post-starter");
    const memoryReferenceEntry = buildExperienceMemoryReferenceEntry();
    expect(task).not.toBeNull();
    expect(memoryReferenceEntry).toBeTruthy();

    await act(async () => {
      root.render(
        <CuratedTaskLauncherDialog
          open
          task={task}
          initialReferenceEntries={
            memoryReferenceEntry ? [memoryReferenceEntry] : []
          }
          onOpenChange={() => undefined}
          onConfirm={() => undefined}
        />,
      );
      await Promise.resolve();
    });

    const subjectField = document.body.querySelector(
      "#curated-task-social-post-starter-subject_or_product",
    ) as HTMLTextAreaElement | null;
    const audienceField = document.body.querySelector(
      "#curated-task-social-post-starter-target_audience",
    ) as HTMLInputElement | null;

    expect(subjectField?.value).toContain("当前主题：短视频编排");
    expect(subjectField?.value).toContain("当前结果基线：");
    expect(audienceField?.value).toBe("正在复盘短视频增长的品牌运营");
    expect(document.body.textContent).toContain(
      "下面的 主题或产品信息 / 目标受众 已按这轮结果自动带入，你可以直接改成这次真正想推进的版本。",
    );
  });

  it("launcher 收到推荐信号时不应再从旧库刷新参考对象", async () => {
    const task = findCuratedTaskTemplateById("account-project-review");
    expect(task).not.toBeNull();
    const onConfirm = vi.fn();

    await act(async () => {
      root.render(
        <CuratedTaskLauncherDialog
          open
          task={task}
          initialReferenceEntries={[
            {
              id: "memory:content-pack:run:1",
              sourceKind: "memory",
              title: "AI 内容周报",
              summary: "当前已有一轮项目结果，可直接作为复盘基线。",
              category: "experience",
              categoryLabel: "成果",
              tags: ["复盘"],
              taskPrefillByTaskId: {
                "account-project-review": {
                  project_goal: "AI 内容周报",
                  existing_results:
                    "当前已有一轮项目结果，可直接作为复盘基线。",
                },
              },
            },
          ]}
          onOpenChange={() => undefined}
          onConfirm={onConfirm}
        />,
      );
      await Promise.resolve();
    });

    expect(document.body.textContent).toContain("AI 内容周报");
    expect(document.body.textContent).not.toContain("品牌语气偏好");

    await act(async () => {
      window.dispatchEvent(
        new CustomEvent(CURATED_TASK_RECOMMENDATION_SIGNAL_EVENT),
      );
      await Promise.resolve();
    });

    expect(document.body.textContent).toContain("AI 内容周报");
    expect(document.body.textContent).not.toContain("品牌语气偏好");

    const confirmButton = findConfirmButton();
    expect(confirmButton).toBeTruthy();
    expect(confirmButton?.getAttribute("disabled")).toBeNull();
    await act(async () => {
      confirmButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });
    expect(onConfirm).toHaveBeenCalledWith(
      expect.objectContaining({ id: "account-project-review" }),
      expect.objectContaining({
        project_goal: "AI 内容周报",
        existing_results: "当前已有一轮项目结果，可直接作为复盘基线。",
      }),
      expect.objectContaining({
        referenceEntries: [
          expect.objectContaining({
            id: "memory:content-pack:run:1",
            sourceKind: "memory",
          }),
        ],
      }),
    );
  });

  it("launcher 应保留显式传入的参考对象并忽略推荐事件刷新", async () => {
    const task = findCuratedTaskTemplateById("account-project-review");
    expect(task).not.toBeNull();
    const onConfirm = vi.fn();

    await act(async () => {
      root.render(
        <CuratedTaskLauncherDialog
          open
          task={task}
          initialReferenceEntries={[
            {
              id: "memory:content-pack:run:1",
              sourceKind: "memory",
              title: "AI 内容周报",
              summary: "当前已有一轮项目结果，可直接作为复盘基线。",
              category: "experience",
              categoryLabel: "成果",
              tags: ["复盘"],
              taskPrefillByTaskId: {
                "account-project-review": {
                  project_goal: "AI 内容周报",
                  existing_results:
                    "当前已有一轮项目结果，可直接作为复盘基线。",
                },
              },
            },
            {
              id: "memory-preference-2",
              sourceKind: "memory",
              title: "品牌语气偏好",
              summary: "整体语气偏轻盈但专业。",
              category: "preference",
              categoryLabel: "偏好",
              tags: ["语气", "品牌"],
            },
            {
              id: "memory-context-3",
              sourceKind: "memory",
              title: "活动背景补充",
              summary: "这轮内容主要围绕新品发布前 48 小时预热。",
              category: "context",
              categoryLabel: "参考",
              tags: ["预热", "活动"],
            },
          ]}
          onOpenChange={() => undefined}
          onConfirm={onConfirm}
        />,
      );
      await Promise.resolve();
    });

    expect(document.body.textContent).toContain("品牌语气偏好");
    expect(document.body.textContent).toContain("已选择 3 条参考对象");

    await act(async () => {
      window.dispatchEvent(
        new CustomEvent(CURATED_TASK_RECOMMENDATION_SIGNAL_EVENT),
      );
      await Promise.resolve();
    });

    expect(document.body.textContent).toContain("活动背景补充");
    expect(document.body.textContent).toContain("品牌语气偏好");

    const confirmButton = findConfirmButton();
    expect(confirmButton).toBeTruthy();

    await act(async () => {
      confirmButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
    });

    expect(onConfirm).toHaveBeenCalledWith(
      expect.objectContaining({ id: "account-project-review" }),
      expect.objectContaining({
        project_goal: "AI 内容周报",
        existing_results: "当前已有一轮项目结果，可直接作为复盘基线。",
      }),
      expect.objectContaining({
        referenceEntries: expect.arrayContaining([
          expect.objectContaining({
            id: "memory:content-pack:run:1",
            sourceKind: "memory",
          }),
          expect.objectContaining({
            id: "memory-preference-2",
            title: "品牌语气偏好",
          }),
          expect.objectContaining({
            id: "memory-context-3",
            title: "活动背景补充",
          }),
        ]),
      }),
    );
  });
});
