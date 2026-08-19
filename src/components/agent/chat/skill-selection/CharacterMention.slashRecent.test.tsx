import { act } from "react";
import { describe, expect, it, vi } from "vitest";
import {
  buildCatalogWithSceneEntry,
  buildCatalogWithStoryVideoSceneEntry,
  createSkill,
  createStoryVideoSceneServiceSkill,
  getTextarea,
  mockListServiceSkills,
  renderHarness,
  typeSlashAndWait,
} from "./CharacterMention.testFixtures";
import { saveSkillCatalog } from "@/lib/api/skillCatalog";
import { recordSlashEntryUsage } from "./slashEntryUsage";
import { recordCuratedTaskTemplateUsage } from "../utils/curatedTaskTemplates";

describe("CharacterMention slash recent entries", () => {
  it("slash 空查询时应优先显示继续上次 Skill，且不在原分组重复", async () => {
    act(() => {
      saveSkillCatalog(buildCatalogWithSceneEntry(), "bootstrap_sync");
      recordSlashEntryUsage({
        kind: "command",
        entryId: "compact",
        usedAt: 1_712_345_678_900,
      });
      recordSlashEntryUsage({
        kind: "scene",
        entryId: "campaign-launch",
        usedAt: 1_712_345_678_800,
      });
      recordSlashEntryUsage({
        kind: "skill",
        entryId: "skill-a",
        usedAt: 1_712_345_678_700,
      });
      recordCuratedTaskTemplateUsage("social-post-starter");
    });

    const container = renderHarness({
      skills: [createSkill("技能A", "skill-a", true)],
    });
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea);

    expect(document.body.textContent).toContain("继续上次 Skill");
    expect(document.body.textContent).toContain("最近操作");
    expect(document.body.textContent).toContain("压缩上下文");
    expect(document.body.textContent).toContain(
      "最近用过的工作台动作；如果是继续产出，优先看上面的 Skill。",
    );

    const compactButtons = Array.from(
      document.body.querySelectorAll("button"),
    ).filter((button) => button.textContent?.includes("/compact"));
    expect(compactButtons).toHaveLength(1);

    const sceneButtons = Array.from(
      document.body.querySelectorAll("button"),
    ).filter((button) => button.textContent?.includes("新品发布场景"));
    expect(sceneButtons).toHaveLength(1);
    expect(document.body.textContent).not.toContain("/campaign-launch");

    const skillButtons = Array.from(
      document.body.querySelectorAll("button"),
    ).filter((button) => button.textContent?.includes("技能A"));
    expect(skillButtons).toHaveLength(1);
    expect(document.body.textContent).not.toContain("/skill-a");

    const curatedTaskButtons = Array.from(
      document.body.querySelectorAll("button"),
    ).filter((button) => button.textContent?.includes("内容主稿生成"));
    expect(curatedTaskButtons).toHaveLength(1);
    expect(document.body.textContent).toContain(
      "去向：首版主稿会先进入当前内容，方便继续改写、拆成多平台版本。",
    );
    expect(document.body.textContent).toContain("下一步：改成多平台版本");
  });

  it("slash 面板中的已经沉淀的 Skills与继续上次 Skill应展示统一轻合同", async () => {
    act(() => {
      recordSlashEntryUsage({
        kind: "skill",
        entryId: "skill-writer",
        usedAt: 1_712_345_678_900,
      });
    });

    const container = renderHarness({
      skills: [
        createSkill("写作助手", "skill-writer", true, {
          description: "本地补充技能",
          metadata: {
            lime_when_to_use: "当你需要复用本地写作 Skill 时使用。",
            lime_argument_hint: "主题、受众与语气要求",
          },
        }),
        createSkill("脚本助手", "skill-script", true, {
          description: "脚本改写做法",
          metadata: {
            lime_when_to_use:
              "当你已经有一版脚本，想继续整理成更适合生成的做法时使用。",
            lime_argument_hint: "已有脚本、目标平台或表达方式",
          },
        }),
      ],
    });
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea);

    expect(document.body.textContent).toContain("已经沉淀的 Skills");
    expect(document.body.textContent).toContain("继续上次 Skill");
    expect(document.body.textContent).toContain(
      "优先接着已经跑过的方法，通常比重新挑一条更省重来成本。",
    );
    expect(document.body.textContent).not.toContain("/skill-writer");
    expect(document.body.textContent).toContain(
      "写作助手 · 当你需要复用本地写作 Skill 时使用。",
    );
    expect(document.body.textContent).toContain(
      "没命中上面的继续项时，再从这里换一条已经沉淀下来的方法。",
    );
    expect(document.body.textContent).toContain("脚本助手");
    expect(document.body.textContent).toContain("需要：主题、受众与语气要求");
    expect(document.body.textContent).toContain(
      "交付：带着这个技能回到首页输入框",
    );
  });

  it("slash 面板打开后新增本地 skill 使用记录时，应即时刷新继续上次 Skill分组", async () => {
    const container = renderHarness({
      skills: [
        createSkill("写作助手", "skill-writer", true, {
          description: "本地补充技能",
          metadata: {
            lime_when_to_use: "当你需要复用本地写作 Skill 时使用。",
            lime_argument_hint: "主题、受众与语气要求",
          },
        }),
      ],
    });
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea);

    expect(document.body.textContent).not.toContain("继续上次 Skill");

    await act(async () => {
      recordSlashEntryUsage({
        kind: "skill",
        entryId: "skill-writer",
        usedAt: 1_712_345_678_901,
      });
      await Promise.resolve();
    });

    expect(document.body.textContent).toContain("继续上次 Skill");
    expect(document.body.textContent).not.toContain("/skill-writer");
    expect(document.body.textContent).toContain("写作助手");
  });

  it("选择最近使用的 slash 命令时应回填上次成功参数", async () => {
    const replayText = "lime-rs packages";
    act(() => {
      recordSlashEntryUsage({
        kind: "command",
        entryId: "review",
        usedAt: 1_712_345_678_900,
        replayText,
      });
    });

    const onChangeSpy = vi.fn<(value: string) => void>();
    const container = renderHarness({
      onChangeSpy,
    });
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea);

    expect(document.body.textContent).toContain(`上次输入：${replayText}`);

    const recentCommandButton = Array.from(
      document.body.querySelectorAll("button"),
    ).find((button) => button.textContent?.includes("/review"));
    expect(recentCommandButton).toBeTruthy();

    act(() => {
      recentCommandButton?.click();
    });

    expect(onChangeSpy).toHaveBeenLastCalledWith("/review lime-rs packages");
  });

  it("选择最近使用的 scene 时应优先回填上次成功参数，而不是再次挂起补参卡", async () => {
    act(() => {
      saveSkillCatalog(buildCatalogWithStoryVideoSceneEntry(), "bootstrap_sync");
      recordSlashEntryUsage({
        kind: "scene",
        entryId: "story-video-suite",
        usedAt: 1_712_345_678_900,
        replayText: "把城市露营咖啡节编排成 60 秒短视频",
      });
    });
    mockListServiceSkills.mockResolvedValueOnce([
      createStoryVideoSceneServiceSkill(),
    ]);

    const onChangeSpy = vi.fn<(value: string) => void>();
    const container = renderHarness({
      onChangeSpy,
    });
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea);

    const recentSceneButton = Array.from(
      document.body.querySelectorAll("button"),
    ).find((button) => button.textContent?.includes("短视频编排"));
    expect(recentSceneButton).toBeTruthy();

    act(() => {
      recentSceneButton?.click();
    });

    expect(onChangeSpy).toHaveBeenLastCalledWith(
      "/story-video-suite 把城市露营咖啡节编排成 60 秒短视频",
    );
  });

  it("选择最近使用的结果模板时应预填上次启动参数与引用", async () => {
    act(() => {
      recordCuratedTaskTemplateUsage({
        templateId: "social-post-starter",
        launchInputValues: {
          subject_or_product: "上次的品牌 campaign 主线",
          target_audience: "海外增长负责人",
        },
        referenceMemoryIds: ["memory-idea-1"],
        referenceEntries: [
          {
            id: "memory-idea-1",
            title: "上次 campaign 参考",
            summary: "延续上次的品牌表达和平台拆分方式",
            category: "context",
            categoryLabel: "参考",
            tags: ["campaign", "品牌"],
          },
        ],
      });
    });

    const container = renderHarness();
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea);

    const recentCuratedTaskButton = Array.from(
      document.body.querySelectorAll("button"),
    ).find((button) => button.textContent?.includes("内容主稿生成"));
    expect(recentCuratedTaskButton).toBeTruthy();
    expect(recentCuratedTaskButton?.textContent).toContain(
      "上次填写：主题或产品信息=上次的品牌 campaign 主线；目标受众=海外增长负责人",
    );
    expect(recentCuratedTaskButton?.textContent).toContain(
      "参考：上次 campaign 参考",
    );

    await act(async () => {
      recentCuratedTaskButton?.click();
      await Promise.resolve();
    });

    expect(document.body.textContent).toContain(
      "已根据你上次启动 内容主稿生成 时的参数自动预填，可继续修改后进入生成。",
    );

    const subjectInput = document.body.querySelector(
      "#curated-task-social-post-starter-subject_or_product",
    ) as HTMLTextAreaElement | null;
    const audienceInput = document.body.querySelector(
      "#curated-task-social-post-starter-target_audience",
    ) as HTMLInputElement | null;

    expect(subjectInput?.value).toBe("上次的品牌 campaign 主线");
    expect(audienceInput?.value).toBe("海外增长负责人");
    expect(document.body.textContent).toContain(
      "已选择 1 条参考对象，本轮会一起带入生成。",
    );
  });

  it("slash 搜索时不应显示最近使用，而应回到搜索结果分组", async () => {
    act(() => {
      recordSlashEntryUsage({
        kind: "command",
        entryId: "compact",
        usedAt: 1_712_345_678_900,
      });
    });

    const container = renderHarness();
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea, "/com");

    expect(document.body.textContent).not.toContain("最近使用");
    expect(document.body.textContent).toContain("工作台操作");
    expect(document.body.textContent).toContain("/compact");
  });
});
