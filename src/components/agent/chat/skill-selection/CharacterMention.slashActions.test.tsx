import { act } from "react";
import { describe, expect, it, vi } from "vitest";
import type { InputCapabilitySelection } from "./inputCapabilitySelection";
import {
  buildCatalogWithSceneEntry,
  buildCatalogWithStoryVideoSceneEntry,
  createSkill,
  createStoryVideoSceneServiceSkill,
  getTextarea,
  mockListServiceSkills,
  renderHarness,
  typeAtAndWait,
  typeSlashAndWait,
} from "./CharacterMention.testFixtures";
import { saveSkillCatalog } from "@/lib/api/skillCatalog";
import { recordSlashEntryUsage } from "./slashEntryUsage";

describe("CharacterMention slash actions", () => {
  it("slash 面板选择 Lime 命令时应回填到输入框", async () => {
    const onChangeSpy = vi.fn<(value: string) => void>();
    const container = renderHarness({
      onChangeSpy,
    });
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea, "/com");

    const commandButton = Array.from(
      document.body.querySelectorAll("button"),
    ).find((button) => button.textContent?.includes("/compact"));
    expect(commandButton).toBeTruthy();

    act(() => {
      commandButton?.click();
    });

    expect(onChangeSpy).toHaveBeenCalledWith("/compact ");
  });

  it("slash 面板选择服务端 scene 时应回填场景命令", async () => {
    act(() => {
      saveSkillCatalog(buildCatalogWithSceneEntry(), "bootstrap_sync");
    });

    const onChangeSpy = vi.fn<(value: string) => void>();
    const container = renderHarness({
      onChangeSpy,
    });
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea, "/camp");

    const sceneButton = Array.from(
      document.body.querySelectorAll("button"),
    ).find((button) => button.textContent?.includes("新品发布场景"));
    expect(sceneButton).toBeTruthy();

    await act(async () => {
      sceneButton?.click();
      await Promise.resolve();
    });

    expect(onChangeSpy).toHaveBeenCalledWith("/campaign-launch ");
  });

  it("提供 onSelectInputCapability 时，slash 面板选择 scene 应走统一 capability 回调", async () => {
    act(() => {
      saveSkillCatalog(buildCatalogWithSceneEntry(), "bootstrap_sync");
    });

    const onChangeSpy = vi.fn<(value: string) => void>();
    const onSelectInputCapability =
      vi.fn<
        (
          capability: InputCapabilitySelection,
          options?: { replayText?: string },
        ) => void
      >();
    const container = renderHarness({
      onChangeSpy,
      onSelectInputCapability,
    });
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea, "/camp");

    const sceneButton = Array.from(
      document.body.querySelectorAll("button"),
    ).find((button) => button.textContent?.includes("新品发布场景"));
    expect(sceneButton).toBeTruthy();

    act(() => {
      sceneButton?.click();
    });

    expect(onSelectInputCapability).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "runtime_scene",
        command: expect.objectContaining({
          key: "campaign-launch",
          commandPrefix: "/campaign-launch",
        }),
      }),
      undefined,
    );
    expect(onChangeSpy).toHaveBeenCalledWith("");
  });

  it("提供 onSelectInputCapability 时，最近使用的 scene 应带 replayText 走统一 capability 回调", async () => {
    act(() => {
      saveSkillCatalog(buildCatalogWithSceneEntry(), "bootstrap_sync");
      recordSlashEntryUsage({
        kind: "scene",
        entryId: "campaign-launch",
        usedAt: 1_712_345_678_900,
        replayText: "帮我做一版新品活动启动方案",
      });
    });

    const onChangeSpy = vi.fn<(value: string) => void>();
    const onSelectInputCapability =
      vi.fn<
        (
          capability: InputCapabilitySelection,
          options?: { replayText?: string },
        ) => void
      >();
    const container = renderHarness({
      onChangeSpy,
      onSelectInputCapability,
    });
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea);

    const recentSceneButton = Array.from(
      document.body.querySelectorAll("button"),
    ).find((button) => button.textContent?.includes("新品发布场景"));
    expect(recentSceneButton).toBeTruthy();

    act(() => {
      recentSceneButton?.click();
    });

    expect(onSelectInputCapability).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "runtime_scene",
        command: expect.objectContaining({
          key: "campaign-launch",
          commandPrefix: "/campaign-launch",
        }),
      }),
      { replayText: "帮我做一版新品活动启动方案" },
    );
    expect(onChangeSpy).toHaveBeenLastCalledWith("帮我做一版新品活动启动方案");
  });

  it("slash 面板选择带必填参数的 scene 时应交给父层 A2UI 补参接管", async () => {
    act(() => {
      saveSkillCatalog(buildCatalogWithStoryVideoSceneEntry(), "bootstrap_sync");
    });
    mockListServiceSkills.mockResolvedValueOnce([
      createStoryVideoSceneServiceSkill(),
    ]);

    const onChangeSpy = vi.fn<(value: string) => void>();
    const onSelectInputCapability =
      vi.fn<
        (
          capability: InputCapabilitySelection,
          options?: { replayText?: string },
        ) => void
      >();
    const container = renderHarness({
      onChangeSpy,
      onSelectInputCapability,
    });
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea, "/story");

    const sceneButton = Array.from(
      document.body.querySelectorAll("button"),
    ).find((button) => button.textContent?.includes("短视频编排"));
    expect(sceneButton).toBeTruthy();

    await act(async () => {
      sceneButton?.click();
      await Promise.resolve();
    });

    expect(onSelectInputCapability).toHaveBeenCalledWith({
      kind: "service_skill",
      skill: expect.objectContaining({
        id: "story-video-suite",
        title: "短视频编排",
      }),
    });
    expect(onChangeSpy).not.toHaveBeenCalled();
  });

  it("slash 面板选择已安装技能时应直接回填 slash skill", async () => {
    const onChangeSpy = vi.fn<(value: string) => void>();
    const container = renderHarness({
      skills: [createSkill("技能A", "skill-a", true)],
      onChangeSpy,
    });
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea, "/ski");

    const skillButton = Array.from(
      document.body.querySelectorAll("button"),
    ).find((button) => button.textContent?.includes("技能A"));
    expect(skillButton).toBeTruthy();

    act(() => {
      skillButton?.click();
    });

    expect(onChangeSpy).toHaveBeenCalledWith("/skill-a ");
  });

  it("提供 onSelectInputCapability 时，slash 面板选择已安装技能应走统一 capability 回调", async () => {
    const onChangeSpy = vi.fn<(value: string) => void>();
    const onSelectInputCapability =
      vi.fn<
        (
          capability: InputCapabilitySelection,
          options?: { replayText?: string },
        ) => void
      >();
    const container = renderHarness({
      skills: [createSkill("技能A", "skill-a", true)],
      onChangeSpy,
      onSelectInputCapability,
    });
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea, "/ski");

    const skillButton = Array.from(
      document.body.querySelectorAll("button"),
    ).find((button) => button.textContent?.includes("技能A"));
    expect(skillButton).toBeTruthy();

    act(() => {
      skillButton?.click();
    });

    expect(onSelectInputCapability).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "installed_skill",
        skill: expect.objectContaining({
          key: "skill-a",
          name: "技能A",
        }),
      }),
      undefined,
    );
    expect(onChangeSpy).toHaveBeenCalledWith("");
  });

  it("提供 onSelectInputCapability 时，最近使用的已安装技能应带 replayText 走统一 capability 回调", async () => {
    act(() => {
      recordSlashEntryUsage({
        kind: "skill",
        entryId: "skill-a",
        usedAt: 1_712_345_678_900,
        replayText: "整理最近发布计划",
      });
    });

    const onChangeSpy = vi.fn<(value: string) => void>();
    const onSelectInputCapability =
      vi.fn<
        (
          capability: InputCapabilitySelection,
          options?: { replayText?: string },
        ) => void
      >();
    const container = renderHarness({
      skills: [createSkill("技能A", "skill-a", true)],
      onChangeSpy,
      onSelectInputCapability,
    });
    const textarea = getTextarea(container);

    await typeSlashAndWait(textarea);

    const recentSkillButton = Array.from(
      document.body.querySelectorAll("button"),
    ).find((button) => button.textContent?.includes("技能A"));
    expect(recentSkillButton).toBeTruthy();

    act(() => {
      recentSkillButton?.click();
    });

    expect(onSelectInputCapability).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "installed_skill",
        skill: expect.objectContaining({
          key: "skill-a",
          name: "技能A",
        }),
      }),
      { replayText: "整理最近发布计划" },
    );
    expect(onChangeSpy).toHaveBeenLastCalledWith("整理最近发布计划");
  });

  it("提及面板应锚定在输入框正上方，并禁止自动翻转到下方", async () => {
    const container = renderHarness();
    const textarea = getTextarea(container);
    vi.spyOn(textarea, "getBoundingClientRect").mockReturnValue({
      x: 120,
      y: 240,
      left: 120,
      top: 240,
      right: 720,
      bottom: 360,
      width: 600,
      height: 120,
      toJSON: () => ({}),
    });

    await typeAtAndWait(textarea);

    const anchor = document.body.querySelector(
      '[data-testid="mention-anchor"]',
    ) as HTMLDivElement | null;
    const popover = document.body.querySelector(
      '[data-testid="mention-popover-content"]',
    ) as HTMLDivElement | null;

    expect(anchor?.style.top).toBe("240px");
    expect(anchor?.style.left).toBe("120px");
    expect(anchor?.style.width).toBe("600px");
    expect(popover?.getAttribute("data-side")).toBe("top");
    expect(popover?.getAttribute("data-align")).toBe("start");
    expect(popover?.getAttribute("data-avoid-collisions")).toBe("false");
    expect(popover?.style.width).toBe("600px");
    expect(popover?.style.bottom).toBe("536px");
  });
});
