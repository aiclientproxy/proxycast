import { act } from "react";
import { describe, expect, it, vi } from "vitest";
import type { InputCapabilitySelection } from "./inputCapabilitySelection";
import {
  buildCatalogWithSceneEntry,
  createSkill,
  findLauncherConfirmButton,
  getTextarea,
  renderHarness,
  typeSlashAndWait,
  updateFieldValue,
} from "./CharacterMention.testFixtures";
import { saveSkillCatalog } from "@/lib/api/skillCatalog";
import {
  buildCuratedTaskLaunchPrompt,
  findCuratedTaskTemplateById,
  recordCuratedTaskTemplateUsage,
} from "../utils/curatedTaskTemplates";
import {
  recordCuratedTaskRecommendationSignalFromMemoryReference,
  recordCuratedTaskRecommendationSignalFromReviewDecision,
} from "../utils/curatedTaskRecommendationSignals";

describe("CharacterMention slash templates", () => {
  it("输入 / 时应优先显示先拿结果、已经沉淀的 Skills与工作台操作，而不是把全部命令摊平", async () => {
    const container = renderHarness({
      skills: [createSkill("本地做法A", "local-skill-a", true)],
    });
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea);

    expect(document.body.textContent).toContain("先拿结果");
    expect(document.body.textContent).toContain("工作台操作");
    expect(document.body.textContent).toContain("新建任务");
    expect(document.body.textContent).toContain("清空任务");
    expect(document.body.textContent).toContain("压缩上下文");
    expect(document.body.textContent).toContain(
      "整理当前任务时再用，不会替代上面的结果入口。",
    );
    expect(document.body.textContent).toContain("/compact");
    expect(document.body.textContent).not.toContain(
      "压缩当前会话上下文并写入摘要",
    );
    expect(document.body.textContent).toContain("已经沉淀的 Skills");
    expect(document.body.textContent).not.toContain("/review");
    expect(document.body.textContent).not.toContain("/help");
    expect(document.body.textContent).not.toContain("/quit");

    const bodyText = document.body.textContent ?? "";
    expect(bodyText.indexOf("先拿结果")).toBeLessThan(
      bodyText.indexOf("已经沉淀的 Skills"),
    );
    expect(bodyText.indexOf("已经沉淀的 Skills")).toBeLessThan(
      bodyText.indexOf("工作台操作"),
    );
  });

  it("统一目录中的结果模板应出现在 slash 面板里", async () => {
    act(() => {
      saveSkillCatalog(buildCatalogWithSceneEntry(), "bootstrap_sync");
    });

    const container = renderHarness();
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea, "/camp");

    expect(document.body.textContent).toContain("结果模板");
    expect(document.body.textContent).toContain("新品发布场景");
    expect(document.body.textContent).not.toContain("/campaign-launch");
    expect(document.body.textContent).toContain(
      "把链接解析、配图和封面串成一条产品链路。",
    );
  });

  it("共享 curated task 结果模板也应出现在 slash 面板里，并通过 launcher 回填启动提示", async () => {
    const onChangeSpy = vi.fn<(value: string) => void>();
    const template = findCuratedTaskTemplateById("daily-trend-briefing");
    expect(template).toBeTruthy();
    const container = renderHarness({
      onChangeSpy,
      syncValue: true,
    });
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea, "/趋势");

    expect(document.body.textContent).toContain("结果模板");
    expect(document.body.textContent).toContain(
      "需要：主题或赛道、希望关注的平台/地域",
    );
    expect(document.body.textContent).toContain("交付：趋势摘要、3 个优先选题");
    expect(document.body.textContent).toContain(
      "去向：趋势摘要会先写回当前内容，方便继续展开选题和主稿。",
    );
    const templateButton = Array.from(
      document.body.querySelectorAll("button"),
    ).find((button) => button.textContent?.includes("每日趋势摘要"));
    expect(templateButton).toBeTruthy();

    act(() => {
      templateButton?.click();
    });

    expect(document.body.textContent).toContain(
      "开始这一步前，我先确认几件事。",
    );
    expect(onChangeSpy).not.toHaveBeenCalled();

    const themeInput = document.body.querySelector(
      "#curated-task-daily-trend-briefing-theme_target",
    ) as HTMLInputElement | null;
    const platformInput = document.body.querySelector(
      "#curated-task-daily-trend-briefing-platform_region",
    ) as HTMLInputElement | null;

    await act(async () => {
      updateFieldValue(themeInput, "AI 内容创作");
      updateFieldValue(platformInput, "X 与 TikTok 北美区");
      await Promise.resolve();
    });

    const confirmButton = findLauncherConfirmButton();
    expect(confirmButton).toBeTruthy();

    await act(async () => {
      confirmButton?.click();
      await Promise.resolve();
    });

    expect(onChangeSpy).toHaveBeenCalledWith(
      buildCuratedTaskLaunchPrompt({
        task: template!,
        inputValues: {
          theme_target: "AI 内容创作",
          platform_region: "X 与 TikTok 北美区",
        },
      }),
    );
  });

  it("slash 面板里的复盘结果模板应显影当前结果基线摘要", async () => {
    const container = renderHarness({
      defaultCuratedTaskReferenceEntries: [
        {
          id: "memory:ai-weekly:run:1",
          sourceKind: "memory",
          title: "AI 内容周报",
          summary: "已有一轮可继续放量的结果。",
          category: "experience",
          categoryLabel: "成果",
          tags: ["复盘", "增长"],
          taskPrefillByTaskId: {
            "account-project-review": {
              project_goal: "AI 内容周报",
              existing_results:
                "当前判断：适合继续放量 经营动作：保留品牌联名方向 更适合去向：内容主稿生成 / 渠道改写",
            },
          },
        },
      ],
    });
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea, "/复盘");

    expect(document.body.textContent).toContain("结果模板");
    expect(document.body.textContent).toContain("复盘这个账号/项目");
    expect(document.body.textContent).toContain("当前结果基线：AI 内容周报");
    expect(document.body.textContent).toContain("当前判断：适合继续放量");
    expect(document.body.textContent).toContain(
      "更适合去向：内容主稿生成 / 渠道改写",
    );
  });

  it("slash 面板里的下游结果模板也应继续显影 结果基线摘要", async () => {
    const container = renderHarness({
      defaultCuratedTaskReferenceEntries: [
        {
          id: "memory:ai-weekly:run:1",
          sourceKind: "memory",
          title: "AI 内容周报",
          summary: "已有一轮可继续放量的结果。",
          category: "experience",
          categoryLabel: "成果",
          tags: ["复盘", "增长"],
          taskPrefillByTaskId: {
            "account-project-review": {
              project_goal: "AI 内容周报",
              existing_results:
                "当前判断：适合继续放量 经营动作：保留品牌联名方向 更适合去向：内容主稿生成 / 渠道改写",
            },
          },
        },
      ],
    });
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea, "/趋势");

    expect(document.body.textContent).toContain("每日趋势摘要");
    expect(document.body.textContent).toContain("当前结果基线：AI 内容周报");
    expect(document.body.textContent).toContain("当前判断：适合继续放量");
    expect(document.body.textContent).toContain(
      "更适合去向：内容主稿生成 / 渠道改写",
    );
  });

  it("搜索命中的普通结果模板也应继续沿用最近一次启动参数", async () => {
    act(() => {
      recordCuratedTaskTemplateUsage({
        templateId: "daily-trend-briefing",
        launchInputValues: {
          theme_target: "AI 内容创作",
          platform_region: "X 与 TikTok 北美区",
        },
      });
    });

    const container = renderHarness({
      syncValue: true,
    });
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea, "/趋势");

    const templateButton = Array.from(
      document.body.querySelectorAll("button"),
    ).find((button) => button.textContent?.includes("每日趋势摘要"));
    expect(templateButton).toBeTruthy();

    await act(async () => {
      templateButton?.click();
      await Promise.resolve();
    });

    expect(document.body.textContent).toContain(
      "已根据你上次启动 每日趋势摘要 时的参数自动预填，可继续修改后进入生成。",
    );

    const themeInput = document.body.querySelector(
      "#curated-task-daily-trend-briefing-theme_target",
    ) as HTMLInputElement | null;
    const platformInput = document.body.querySelector(
      "#curated-task-daily-trend-briefing-platform_region",
    ) as HTMLInputElement | null;

    expect(themeInput?.value).toBe("AI 内容创作");
    expect(platformInput?.value).toBe("X 与 TikTok 北美区");
  });

  it("提供 onSelectInputCapability 时，slash 面板选择结果模板应在 launcher 确认后走统一 capability 回调", async () => {
    const onChangeSpy = vi.fn<(value: string) => void>();
    const onSelectInputCapability =
      vi.fn<
        (
          capability: InputCapabilitySelection,
          options?: { replayText?: string },
        ) => void
      >();
    const template = findCuratedTaskTemplateById("daily-trend-briefing");
    expect(template).toBeTruthy();
    const container = renderHarness({
      onChangeSpy,
      onSelectInputCapability,
      syncValue: true,
    });
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea, "/趋势");

    const templateButton = Array.from(
      document.body.querySelectorAll("button"),
    ).find((button) => button.textContent?.includes("每日趋势摘要"));
    expect(templateButton).toBeTruthy();

    act(() => {
      templateButton?.click();
    });

    expect(onChangeSpy).not.toHaveBeenCalled();
    expect(onSelectInputCapability).not.toHaveBeenCalled();

    const themeInput = document.body.querySelector(
      "#curated-task-daily-trend-briefing-theme_target",
    ) as HTMLInputElement | null;
    const platformInput = document.body.querySelector(
      "#curated-task-daily-trend-briefing-platform_region",
    ) as HTMLInputElement | null;

    await act(async () => {
      updateFieldValue(themeInput, "AI 内容创作");
      updateFieldValue(platformInput, "X 与 TikTok 北美区");
      await Promise.resolve();
    });

    const confirmButton = findLauncherConfirmButton();
    expect(confirmButton).toBeTruthy();

    await act(async () => {
      confirmButton?.click();
      await Promise.resolve();
    });

    const prompt = buildCuratedTaskLaunchPrompt({
      task: template!,
      inputValues: {
        theme_target: "AI 内容创作",
        platform_region: "X 与 TikTok 北美区",
      },
    });

    expect(onChangeSpy).toHaveBeenCalledWith(prompt);
    expect(onSelectInputCapability).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "curated_task",
        task: expect.objectContaining({
          id: "daily-trend-briefing",
          title: "每日趋势摘要",
          prompt,
        }),
      }),
      {
        replayText: prompt,
      },
    );
  });

  it("slash 面板启动结果模板时，应默认沿用当前带入的记忆参考", async () => {
    const container = renderHarness({
      defaultCuratedTaskReferenceMemoryIds: ["memory-1"],
      defaultCuratedTaskReferenceEntries: [
        {
          id: "memory-1",
          title: "品牌风格样本",
          summary: "保留轻盈但专业的表达。",
          category: "context",
          categoryLabel: "参考",
          tags: ["品牌", "语气"],
        },
      ],
    });
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea, "/趋势");

    const templateButton = Array.from(
      document.body.querySelectorAll("button"),
    ).find((button) => button.textContent?.includes("每日趋势摘要"));
    expect(templateButton).toBeTruthy();

    act(() => {
      templateButton?.click();
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(document.body.textContent).toContain(
      "已选择 1 条参考对象，本轮会一起带入生成。",
    );
  });

  it("slash 面板的结果模板应复用记忆参考推荐信号排序", async () => {
    recordCuratedTaskRecommendationSignalFromMemoryReference(
      {
        id: "memory-review-1",
        session_id: "session-review-1",
        memory_type: "project",
        title: "本周账号复盘线索",
        category: "experience",
        summary: "内容表现、增长拐点和掉量问题都在这里。",
        content: "内容表现、增长拐点和掉量问题都在这里。",
        tags: ["复盘", "增长", "账号"],
        metadata: {
          confidence: 0.96,
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
      {
        projectId: "project-review-1",
      },
    );

    const container = renderHarness({
      projectId: "project-review-1",
    });
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea);

    expect(document.body.textContent).toContain("成果：本周账号复盘线索");

    const resultTemplateSection = Array.from(
      document.body.querySelectorAll("section"),
    ).find((section) => section.textContent?.includes("先拿结果"));
    expect(resultTemplateSection).toBeTruthy();

    const buttonTexts = Array.from(
      resultTemplateSection?.querySelectorAll("button") ?? [],
    ).map((button) => button.textContent ?? "");
    const reviewIndex = buttonTexts.findIndex((text) =>
      text.includes("复盘这个账号/项目"),
    );
    const trendIndex = buttonTexts.findIndex((text) =>
      text.includes("每日趋势摘要"),
    );

    expect(reviewIndex).toBeGreaterThanOrEqual(0);
    expect(trendIndex).toBeGreaterThanOrEqual(0);
    expect(reviewIndex).toBeLessThan(trendIndex);
  });

  it("slash 面板的结果模板分组应显影最近判断横幅", async () => {
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
        projectId: "project-review-2",
        sceneTitle: "短视频编排",
      },
    );

    const container = renderHarness({
      projectId: "project-review-2",
    });
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea);

    const banner = document.body.querySelector(
      '[data-testid="input-capability-section-banner-result-templates"]',
    );
    expect(banner?.textContent).toContain(
      "最近判断已更新：短视频编排 · 补证据",
    );
    expect(banner?.textContent).toContain("这轮结果还缺证据");
    expect(banner?.textContent).toContain("这轮判断更建议优先回到");
    expect(banner?.textContent).toContain("复盘这个账号/项目");
    expect(banner?.textContent).toContain("拆解一条爆款内容");
    expect(banner?.textContent).toContain(
      "更适合继续：复盘这个账号/项目 / 拆解一条爆款内容",
    );

    const bannerAction = document.body.querySelector(
      '[data-testid="input-capability-section-banner-action-result-templates"]',
    ) as HTMLButtonElement | null;
    expect(bannerAction?.textContent).toContain("继续去「复盘这个账号/项目」");

    await act(async () => {
      bannerAction?.click();
      await Promise.resolve();
    });

    expect(document.body.textContent).toContain(
      "开始这一步前，我先确认几件事。",
    );
    expect(document.body.textContent).toContain("复盘这个账号/项目");
  });

  it("搜索未接入的 slash 命令时，应单独显示暂未接入分组", async () => {
    const container = renderHarness();
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea, "/qui");

    expect(document.body.textContent).not.toContain("工作台操作");
    expect(document.body.textContent).toContain("暂未接入");
    expect(document.body.textContent).toContain("/quit");
  });

  it("slash 搜索提示类命令时，应按提示命令分组展开", async () => {
    const container = renderHarness();
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea, "/rev");

    expect(document.body.textContent).toContain("提示命令");
    expect(document.body.textContent).toContain("/review");
    expect(document.body.textContent).not.toContain("状态 / 帮助");
  });

  it("slash 搜索状态类命令时，应按状态 / 帮助分组展开", async () => {
    const container = renderHarness();
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea, "/help");

    expect(document.body.textContent).toContain("状态 / 帮助");
    expect(document.body.textContent).toContain("/help");
  });
});
