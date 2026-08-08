import { act } from "react";
import { describe, expect, it, vi } from "vitest";
import type { InputCapabilitySelection } from "./inputCapabilitySelection";
import {
  createServiceSkill,
  createSkill,
  findButtonContaining,
  getButtonsContaining,
  getMentionPopoverContent,
  getTextarea,
  renderHarness,
  typeAtAndWait,
  typeMentionAndWait,
} from "./CharacterMention.testFixtures";
import { recordServiceSkillUsage } from "@/components/agent/chat/service-skills/storage";
import type {
  InputbarPluginCapability,
  InputbarPluginSelectionOptions,
  InputbarPluginSkillCapability,
} from "../components/Inputbar/pluginInputCapability";

describe("CharacterMention mention catalog", () => {
  it("@ 面板中的已安装技能应展示统一的轻量 skill 合同", async () => {
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
          description: "另一条备用本地 Skill",
          metadata: {
            lime_when_to_use: "当你需要改写脚本结构时使用。",
            lime_argument_hint: "脚本目标与表达风格",
          },
        }),
      ],
    });
    const textarea = getTextarea(container);

    await typeAtAndWait(textarea);

    expect(document.body.textContent).toContain("Skills");
    expect(document.body.textContent).toContain("写作助手");
    expect(document.body.textContent).toContain(
      "当你需要复用本地写作 Skill 时使用。",
    );
    expect(document.body.textContent).toContain("需要：主题、受众与语气要求");
    expect(document.body.textContent).toContain(
      "交付：带着这个技能回到首页输入框",
    );
  });

  it("只有最近使用的服务技能时，不应同时出现空态文案", async () => {
    const container = renderHarness({
      serviceSkills: [
        createServiceSkill({
          id: "recent-trend-briefing",
          title: "最近趋势摘要",
          recentUsedAt: 1_712_345_678_000,
          isRecent: true,
        }),
      ],
    });
    const textarea = getTextarea(container);

    await typeAtAndWait(textarea);

    expect(document.body.textContent).toContain("最近调用");
    expect(document.body.textContent).toContain("最近趋势摘要");
    expect(document.body.textContent).not.toContain("暂无可用角色或技能");
  });

  it("输入 @ 查询服务技能时，应回到按技能组展示搜索结果", async () => {
    const container = renderHarness({
      serviceSkills: [
        createServiceSkill({
          id: "github-repo-radar",
          title: "GitHub 仓库雷达",
          aliases: ["仓库雷达", "GitHub 搜索"],
          groupKey: "github",
          recentUsedAt: 1_712_345_678_000,
          isRecent: true,
          runnerType: "instant",
          defaultExecutorBinding: "browser_assist",
          runnerLabel: "浏览器协助",
          runnerTone: "sky",
          runnerDescription: "进入真实浏览器执行只读采集。",
          actionLabel: "对话内补参",
        }),
      ],
    });
    const textarea = getTextarea(container);

    await act(async () => {
      await import("./CharacterMentionPanel");
    });

    act(() => {
      textarea.focus();
      textarea.value = "@git";
      textarea.setSelectionRange(4, 4);
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(document.body.textContent).not.toContain("最近调用");
    expect(document.body.textContent).not.toContain("场景 Skills");
    expect(document.body.textContent).toContain("GitHub");
    expect(document.body.textContent).toContain("GitHub 仓库雷达");
  });

  it("@ 面板技能组标题和排序应优先复用后端目录分组", async () => {
    const container = renderHarness({
      serviceSkills: [
        createServiceSkill({
          id: "creative-workbench-brief",
          title: "创作工作台摘要",
          aliases: ["创作摘要"],
          groupKey: "creative-workbench",
        }),
        createServiceSkill({
          id: "general-brief",
          title: "通用创作摘要",
          aliases: ["通用摘要"],
          groupKey: "general",
        }),
      ],
      serviceSkillGroups: [
        {
          key: "general",
          title: "通用技能",
          summary: "常规创作技能。",
          sort: 90,
          itemCount: 1,
        },
        {
          key: "creative-workbench",
          title: "创作中台",
          summary: "围绕创作链路的协作技能。",
          sort: 5,
          itemCount: 1,
        },
      ],
    });
    const textarea = getTextarea(container);

    await typeMentionAndWait(textarea, "@摘要");

    const bodyText = document.body.textContent ?? "";
    expect(bodyText).toContain("创作中台");
    expect(bodyText).toContain("通用技能");
    expect(bodyText).not.toContain("技能组 · creative-workbench");
    expect(bodyText.indexOf("创作中台")).toBeLessThan(
      bodyText.indexOf("通用技能"),
    );
  });

  it("服务技能应出现在 @ 面板里", async () => {
    const container = renderHarness({
      serviceSkills: [
        createServiceSkill(),
        createServiceSkill({
          id: "github-repo-radar",
          title: "GitHub 仓库雷达",
          summary: "围绕仓库与 Issue 快速扫描线索。",
          entryHint: "补一个关键词，我先帮你扫 GitHub 仓库与讨论。",
          aliases: ["仓库雷达", "GitHub 搜索"],
          category: "GitHub",
          runnerType: "instant",
          defaultExecutorBinding: "browser_assist",
          slotSchema: [
            {
              key: "repository_query",
              label: "仓库关键词",
              type: "text",
              required: true,
              placeholder: "例如 AI Agent",
            },
          ],
          runnerLabel: "浏览器协助",
          runnerTone: "sky",
          runnerDescription: "进入真实浏览器执行只读采集。",
          actionLabel: "对话内补参",
          groupKey: "github",
        }),
      ],
    });
    const textarea = getTextarea(container);

    await typeAtAndWait(textarea);

    expect(document.body.textContent).toContain("场景 Skills");
    expect(document.body.textContent).toContain("每日趋势摘要");
    expect(document.body.textContent).toContain("GitHub 仓库雷达");
    expect(document.body.textContent).toContain("需要：当前无必填信息");
    expect(document.body.textContent).toContain("交付：趋势摘要 + 调度建议");
    expect(document.body.textContent).toContain("需要：仓库关键词");
    const bodyText = document.body.textContent ?? "";
    expect(bodyText.indexOf("搜索 / 读取")).toBeLessThan(
      bodyText.indexOf("场景 Skills"),
    );
  });

  it("输入 @ 查询 Plugin 时，应激活应用并移除当前 mention", async () => {
    const plugin: InputbarPluginCapability = {
      pluginId: "content-factory-app",
      displayName: "内容工厂",
      description: "从创作需求生成文章、配图、视频分镜和交付检查清单。",
    };
    const onSelectPlugin =
      vi.fn<
        (
          selectedPlugin: InputbarPluginCapability,
          skill?: InputbarPluginSkillCapability,
          options?: InputbarPluginSelectionOptions,
        ) => void
      >();
    const onChangeSpy = vi.fn<(value: string) => void>();
    const container = renderHarness({
      pluginSuggestions: [plugin],
      onSelectPlugin,
      onChangeSpy,
    });
    const textarea = getTextarea(container);

    await typeMentionAndWait(textarea, "@内容");

    expect(document.body.textContent).toContain("Plugins");
    expect(document.body.textContent).toContain("内容工厂");
    expect(document.body.textContent).toContain("从创作需求生成文章");
    expect(document.body.textContent).not.toContain("暂无可用 @命令");

    const pluginButton = findButtonContaining("内容工厂");
    expect(pluginButton).toBeTruthy();

    act(() => {
      pluginButton?.click();
    });

    expect(onChangeSpy).toHaveBeenCalledWith("");
    expect(onSelectPlugin).toHaveBeenCalledWith(plugin, undefined, {
      inputOverride: "",
      preserveInputOverride: true,
    });
    expect(getMentionPopoverContent()).toBeNull();
  });

  it("输入单独 @ 时应展示 Plugin 候选", async () => {
    const plugin: InputbarPluginCapability = {
      pluginId: "content-factory-app",
      displayName: "写文章",
      trigger: "@写文章",
      description: "生成文章草稿。",
    };
    const container = renderHarness({
      pluginSuggestions: [plugin],
    });
    const textarea = getTextarea(container);

    await typeAtAndWait(textarea);

    expect(getMentionPopoverContent()).not.toBeNull();
    expect(document.body.textContent).toContain("Plugins");
    expect(document.body.textContent).toContain("写文章");
  });

  it("输入部分 Plugin 触发词时应继续展示候选", async () => {
    const plugin: InputbarPluginCapability = {
      pluginId: "content-factory-app",
      displayName: "写文章",
      trigger: "@写文章",
      description: "生成文章草稿。",
    };
    const container = renderHarness({
      pluginSuggestions: [plugin],
    });
    const textarea = getTextarea(container);

    await typeMentionAndWait(textarea, "@写");

    expect(getMentionPopoverContent()).not.toBeNull();
    expect(document.body.textContent).toContain("Plugins");
    expect(document.body.textContent).toContain("写文章");
  });

  it("已停用 Plugin 在 @ 查询中仍应展示为不可用候选", async () => {
    const pluginSuggestions: InputbarPluginCapability[] = [
      {
        pluginId: "content-factory-app",
        displayName: "写文章",
        trigger: "@写文章",
        description: "生成文章草稿。",
        disabled: true,
        blockerCodes: ["PLUGIN_DISABLED"],
      },
    ];
    const container = renderHarness({
      pluginSuggestions,
    });
    const textarea = getTextarea(container);

    await typeMentionAndWait(textarea, "@写");

    expect(getMentionPopoverContent()).not.toBeNull();
    expect(document.body.textContent).toContain("Plugins");
    expect(document.body.textContent).toContain("写文章");
    expect(document.body.textContent).not.toContain("暂无可用 @命令");
  });

  it("输入完整 Plugin 触发词时不应展示 @ 面板", async () => {
    const plugin: InputbarPluginCapability = {
      pluginId: "content-factory-app",
      displayName: "写文章",
      trigger: "@写文章",
      description: "生成文章草稿。",
    };
    const container = renderHarness({
      pluginSuggestions: [plugin],
    });
    const textarea = getTextarea(container);

    await typeMentionAndWait(textarea, "@写文章");

    expect(getMentionPopoverContent()).toBeNull();
    expect(document.body.textContent).not.toContain("暂无可用 @命令");
  });

  it("输入完整 Plugin 技能触发词时不应展示 @ 面板", async () => {
    const plugin: InputbarPluginCapability = {
      pluginId: "content-workbench",
      displayName: "内容工厂",
      description: "整理内容生产资料。",
      skills: [
        {
          skillId: "article-writer",
          title: "文章写作",
          description: "生成文章草稿。",
        },
      ],
    };
    const container = renderHarness({
      pluginSuggestions: [plugin],
    });
    const textarea = getTextarea(container);

    await typeMentionAndWait(textarea, "@内容工厂:文章写作");

    expect(getMentionPopoverContent()).toBeNull();
    expect(document.body.textContent).not.toContain("暂无可用 @命令");
  });

  it("选择 Plugin 时即使输入框 selection 丢失，也应移除当前 mention", async () => {
    const plugin: InputbarPluginCapability = {
      pluginId: "content-factory-app",
      displayName: "内容工厂",
      description: "从创作需求生成文章、配图、视频分镜和交付检查清单。",
    };
    const onSelectPlugin =
      vi.fn<
        (
          selectedPlugin: InputbarPluginCapability,
          skill?: InputbarPluginSkillCapability,
          options?: InputbarPluginSelectionOptions,
        ) => void
      >();
    const onChangeSpy = vi.fn<(value: string) => void>();
    const container = renderHarness({
      pluginSuggestions: [plugin],
      onSelectPlugin,
      onChangeSpy,
    });
    const textarea = getTextarea(container);

    await typeMentionAndWait(textarea, "@内容");

    act(() => {
      textarea.setSelectionRange(0, 0);
    });

    const pluginButton = findButtonContaining("内容工厂");
    expect(pluginButton).toBeTruthy();

    act(() => {
      pluginButton?.click();
    });

    expect(onChangeSpy).toHaveBeenCalledWith("");
    expect(onSelectPlugin).toHaveBeenCalledWith(plugin, undefined, {
      inputOverride: "",
      preserveInputOverride: true,
    });
    expect(getMentionPopoverContent()).toBeNull();
  });

  it("最近使用的服务技能应优先显示在独立分组，且不在技能组里重复", async () => {
    act(() => {
      recordServiceSkillUsage({
        skillId: "recent-trend-briefing",
        runnerType: "scheduled",
        slotValues: {
          platform_focus: "X + TikTok",
          keyword_focus: "AI 内容创作",
        },
      });
    });
    const recentSkill = createServiceSkill({
      id: "recent-trend-briefing",
      title: "最近趋势摘要",
      slotSchema: [
        {
          key: "platform_focus",
          label: "关注平台",
          type: "text",
          required: true,
          placeholder: "例如 X + TikTok",
        },
        {
          key: "keyword_focus",
          label: "关键词",
          type: "text",
          required: true,
          placeholder: "例如 AI 内容创作",
        },
      ],
      recentUsedAt: 1_712_345_678_000,
      isRecent: true,
    });
    const regularSkill = createServiceSkill({
      id: "regular-trend-briefing",
      title: "常规趋势摘要",
    });
    const container = renderHarness({
      serviceSkills: [recentSkill, regularSkill],
    });
    const textarea = getTextarea(container);

    await typeAtAndWait(textarea);

    expect(document.body.textContent).toContain("最近调用");
    expect(document.body.textContent).toContain("场景 Skills");
    expect(document.body.textContent).toContain(
      "上次填写：关注平台=X + TikTok；关键词=AI 内容创作",
    );
    expect(getButtonsContaining("最近趋势摘要")).toHaveLength(1);
    expect(getButtonsContaining("常规趋势摘要")).toHaveLength(1);
  });

  it("提供统一 capability 回调时，选择服务技能应走 current 主链", async () => {
    const onSelectInputCapability =
      vi.fn<
        (
          capability: InputCapabilitySelection,
          options?: { replayText?: string },
        ) => void
      >();
    const onChangeSpy = vi.fn<(value: string) => void>();
    const serviceSkill = createServiceSkill();
    const container = renderHarness({
      serviceSkills: [serviceSkill],
      onSelectInputCapability,
      onChangeSpy,
    });
    const textarea = getTextarea(container);

    await typeAtAndWait(textarea);

    const serviceSkillButton = findButtonContaining("每日趋势摘要");
    expect(serviceSkillButton).toBeTruthy();

    act(() => {
      serviceSkillButton?.click();
    });

    expect(onChangeSpy).toHaveBeenCalledWith("");
    expect(onSelectInputCapability).toHaveBeenCalledWith({
      kind: "service_skill",
      skill: serviceSkill,
    });
  });

  it("未提供统一 capability 回调时，选择已安装技能应回填到输入框", async () => {
    const onChangeSpy = vi.fn<(value: string) => void>();
    const container = renderHarness({
      skills: [createSkill("技能A", "skill-a", true)],
      onChangeSpy,
    });
    const textarea = getTextarea(container);

    await typeAtAndWait(textarea);

    const skillButton = findButtonContaining("技能A");
    expect(skillButton).toBeTruthy();

    act(() => {
      skillButton?.click();
    });

    expect(onChangeSpy).toHaveBeenCalledWith("/skill-a ");
  });
});
