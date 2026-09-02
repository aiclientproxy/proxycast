import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  clearSkillCatalogCache,
  findModelBoundImageCommandEntryForModel,
  getSkillCatalog,
  listSkillCatalogCommandEntries,
  listLocalModelBoundImageCommandEntries,
  listSkillCatalogSceneEntries,
  saveSkillCatalog,
  upsertLocalModelBoundImageCommandBinding,
} from "./skillCatalog";
import {
  buildBaseSetupPackage,
  buildLegacyCatalogWithSiteEntries,
} from "./skillCatalogTestFixtures";

describe("skillCatalog", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  afterEach(() => {
    window.localStorage.clear();
    clearSkillCatalogCache();
  });

  it("含已退役站点 adapter 的旧目录应被拒绝", async () => {
    expect(() =>
      saveSkillCatalog(buildLegacyCatalogWithSiteEntries(), "bootstrap_sync"),
    ).toThrow("invalid skill catalog");

    const catalog = await getSkillCatalog();

    expect(catalog.items.some((item) => item.id === "legacy-site-skill")).toBe(
      false,
    );

    const stored = window.localStorage.getItem("lime:skill-catalog:v1");
    expect(stored).not.toContain("legacy-site-skill");
  });

  it("应支持从 Base Setup Package 编译 skill catalog 与显式 scene projection", () => {
    const catalog = saveSkillCatalog(buildBaseSetupPackage(), "bootstrap_sync");
    const sceneEntry = listSkillCatalogSceneEntries(catalog).find(
      (entry) => entry.sceneKey === "story-video-suite",
    );
    const commandEntry = listSkillCatalogCommandEntries(catalog).find(
      (entry) => entry.commandKey === "voice_runtime",
    );
    const skillEntry = catalog.items.find(
      (item) => item.id === "content-workflow-service",
    );

    expect(skillEntry).toEqual(
      expect.objectContaining({
        id: "content-workflow-service",
        groupKey: "workflows",
        execution: expect.objectContaining({
          kind: "agent_turn",
        }),
      }),
    );
    expect(sceneEntry).toEqual(
      expect.objectContaining({
        title: "短视频编排",
        commandPrefix: "/story-video-suite",
        summary: "把文本生成线框图、配乐、剧本和短视频串成一条场景链。",
        aliases: ["story-video", "mv-pipeline"],
        linkedSkillId: "content-workflow-service",
        skillLocator: {
          source: "catalog",
          name: "story-video-suite",
        },
        executionKind: "agent_turn",
        surfaceScopes: ["mention", "workspace"],
      }),
    );
    expect(sceneEntry?.title).not.toBe("旧版自动场景标题");
    expect(sceneEntry?.commandPrefix).not.toBe("/legacy-story-video");
    expect(commandEntry).toEqual(
      expect.objectContaining({
        id: "command:voice_runtime",
        title: "短视频配音入口",
        summary: "用显式 command projection 覆盖 seeded voice_runtime。",
        aliases: ["短视频配音", "story-voice"],
        surfaceScopes: ["mention", "workspace"],
        triggers: [
          { mode: "mention", prefix: "@配音" },
          { mode: "slash", prefix: "/voice-runtime" },
        ],
        binding: {
          skillId: "content-workflow-service",
          skillLocator: {
            source: "catalog",
            name: "voice_runtime",
          },
          executionKind: "agent_turn",
          requestDefaults: {
            launch_hint: "voice_scene",
          },
          intentConfirmation: {
            id: "plain_voice_request",
            ruleKey: "agentChat.voice.intentRules",
            confirmationKey: "agentChat.voice.confirmPlainRequest",
            systemPromptKey: "agentChat.voice.confirmPlainRequestPrompt",
          },
        },
      }),
    );
    expect(commandEntry?.summary).not.toBe(
      "把视频或旁白需求切到云端配音技能主链，优先提交服务型技能运行。",
    );
  });

  it("应把本地图片模型 @命令绑定合并进当前目录", async () => {
    const entry = upsertLocalModelBoundImageCommandBinding({
      trigger: "@GPT Images 2",
      providerId: "yunwu.ai",
      modelId: "gpt-image-2",
      executorMode: "responses_image_generation",
    });

    expect(entry).toMatchObject({
      commandKey: "image_model_gpt_images_2",
      binding: {
        requestDefaults: expect.objectContaining({
          imageWorkbench: "true",
          modelBoundImageTask: "true",
          entrySource: "at_gpt_images_2_model_command",
          providerId: "yunwu.ai",
          model: "gpt-image-2",
          executorMode: "responses_image_generation",
          bindingSource: "local_provider_settings",
        }),
      },
    });
    expect(listLocalModelBoundImageCommandEntries()).toHaveLength(1);

    const catalog = await getSkillCatalog();
    const mergedEntry = findModelBoundImageCommandEntryForModel(
      catalog,
      "yunwu.ai",
      "gpt-image-2",
    );

    expect(mergedEntry).toMatchObject({
      commandKey: "image_model_gpt_images_2",
      triggers: [expect.objectContaining({ prefix: "@GPT Images 2" })],
      binding: {
        requestDefaults: expect.objectContaining({
          providerId: "yunwu.ai",
          model: "gpt-image-2",
          executorMode: "responses_image_generation",
        }),
      },
    });
    expect(
      listSkillCatalogCommandEntries(catalog).filter((catalogEntry) =>
        catalogEntry.triggers.some(
          (trigger) => trigger.prefix === "@GPT Images 2",
        ),
      ),
    ).toEqual([
      expect.objectContaining({
        commandKey: "image_model_gpt_images_2",
      }),
    ]);
  });
});
