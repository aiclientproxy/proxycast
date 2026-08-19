import { describe, expect, it } from "vitest";
import { getSeededServiceSkillCatalog } from "@/lib/api/serviceSkills";
import { compileCommandCatalogProjection } from "./compat/commandCatalogProjection";
import { createSeededCommandCatalogBaseSetupPackage } from "./seededCommandPackage";
import { SEEDED_SERVICE_SKILL_CATALOG_VERSION } from "./seededServiceSkillPackage";

describe("seededCommandPackage", () => {
  it("应提供 seeded command 的基础设置包事实源", () => {
    const pkg = createSeededCommandCatalogBaseSetupPackage();

    expect(pkg.id).toBe("lime-seeded-command-catalog");
    expect(pkg.version).toBe(SEEDED_SERVICE_SKILL_CATALOG_VERSION);
    expect(pkg.catalogProjections).toHaveLength(40);
    expect(pkg.bindingProfiles.map((profile) => profile.id)).toEqual(
      expect.arrayContaining(["agent-turn-instant", "native-skill-instant"]),
    );
    expect(pkg.compatibility.requiredKernelCapabilities).not.toContain(
      "cloud_scene",
    );
  });

  it("应把 seeded command 包编译成与当前命令目录一致的 command entries", () => {
    const entries = compileCommandCatalogProjection(
      createSeededCommandCatalogBaseSetupPackage(),
      getSeededServiceSkillCatalog().items,
    );

    expect(entries).toHaveLength(40);
    expect(
      entries.every(
        (entry) => entry.binding?.skillLocator?.source === "catalog",
      ),
    ).toBe(true);
    expect(entries).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "command:image_generate",
          commandKey: "image_generate",
          binding: expect.objectContaining({
            skillId: "image_generate",
            executionKind: "task_queue",
            intentConfirmation: {
              id: "plain_image_generation",
              ruleKey: "agentChat.inputIntent.imageGeneration.rules",
              confirmationKey: "agentChat.inputIntent.imageGeneration.confirm",
              systemPromptKey:
                "agentChat.inputIntent.imageGeneration.systemPrompt",
            },
          }),
          renderContract: expect.objectContaining({
            resultKind: "image_gallery",
            detailKind: "media_detail",
          }),
          triggers: expect.arrayContaining([
            { mode: "mention", prefix: "@配图" },
            { mode: "mention", prefix: "@Vision 1" },
            { mode: "mention", prefix: "@image" },
            { mode: "slash", prefix: "/image" },
          ]),
        }),
        expect.objectContaining({
          id: "command:image_storyboard",
          commandKey: "image_storyboard",
          binding: expect.objectContaining({
            skillId: "image_generate",
            executionKind: "task_queue",
          }),
          renderContract: expect.objectContaining({
            resultKind: "image_gallery",
            detailKind: "media_detail",
          }),
        }),
        expect.objectContaining({
          id: "command:image_generate_nanobanana_pro",
          commandKey: "image_generate_nanobanana_pro",
          binding: expect.objectContaining({
            skillId: "image_generate",
            executionKind: "task_queue",
            requestDefaults: expect.objectContaining({
              providerId: "fal",
              model: "fal-ai/nano-banana-pro",
            }),
          }),
          renderContract: expect.objectContaining({
            resultKind: "image_gallery",
            detailKind: "media_detail",
          }),
          triggers: expect.arrayContaining([
            { mode: "mention", prefix: "@Nanobanana Pro" },
          ]),
        }),
        expect.objectContaining({
          id: "command:image_generate_gpt_images_2",
          commandKey: "image_generate_gpt_images_2",
          binding: expect.objectContaining({
            skillId: "image_generate",
            executionKind: "task_queue",
            requestDefaults: expect.objectContaining({
              providerId: "new-api",
              model: "gpt-images-2",
              executorMode: "responses_image_generation",
            }),
          }),
          renderContract: expect.objectContaining({
            resultKind: "image_gallery",
            detailKind: "media_detail",
          }),
          triggers: expect.arrayContaining([
            { mode: "mention", prefix: "@GPT Images 2" },
          ]),
        }),
        expect.objectContaining({
          id: "command:voice_runtime",
          commandKey: "voice_runtime",
          binding: expect.objectContaining({
            skillId: "cloud-video-dubbing",
            executionKind: "agent_turn",
            skillLocator: {
              source: "catalog",
              name: "voice_runtime",
            },
          }),
          renderContract: expect.objectContaining({
            resultKind: "tool_timeline",
            detailKind: "scene_detail",
          }),
          triggers: expect.arrayContaining([
            { mode: "mention", prefix: "@配音" },
            { mode: "mention", prefix: "@Website Voiceover" },
          ]),
        }),
        expect.objectContaining({
          id: "command:growth_runtime",
          commandKey: "growth_runtime",
          binding: expect.objectContaining({
            skillId: "account-performance-tracking",
            executionKind: "agent_turn",
            skillLocator: {
              source: "catalog",
              name: "growth_runtime",
            },
          }),
          renderContract: expect.objectContaining({
            resultKind: "tool_timeline",
            detailKind: "scene_detail",
          }),
          triggers: expect.arrayContaining([
            { mode: "mention", prefix: "@增长" },
            { mode: "mention", prefix: "@Growth Expert" },
          ]),
        }),
        expect.objectContaining({
          id: "command:writing_runtime",
          commandKey: "writing_runtime",
          binding: expect.objectContaining({
            skillId: "content_post_with_cover",
            executionKind: "native_skill",
            skillLocator: {
              source: "catalog",
              name: "writing_runtime",
            },
          }),
          renderContract: expect.objectContaining({
            resultKind: "artifact",
            detailKind: "artifact_detail",
          }),
          triggers: expect.arrayContaining([
            { mode: "mention", prefix: "@文案" },
            { mode: "mention", prefix: "@Writing Partner" },
            { mode: "mention", prefix: "@Writers 1" },
            { mode: "mention", prefix: "@Blog 1" },
            { mode: "mention", prefix: "@Newsletters Pro" },
            { mode: "mention", prefix: "@Web Copy" },
          ]),
        }),
        expect.objectContaining({
          id: "command:code_runtime",
          commandKey: "code_runtime",
          binding: expect.objectContaining({
            skillId: undefined,
            skillLocator: {
              source: "catalog",
              name: "code_runtime",
            },
            executionKind: "agent_turn",
            requestDefaults: {
              executionStrategy: "react",
            },
          }),
          renderContract: expect.objectContaining({
            resultKind: "tool_timeline",
            detailKind: "json",
          }),
          triggers: expect.arrayContaining([
            { mode: "mention", prefix: "@代码" },
            { mode: "mention", prefix: "@code" },
            { mode: "mention", prefix: "@coding" },
            { mode: "mention", prefix: "@开发" },
            { mode: "mention", prefix: "@Code Agent" },
          ]),
        }),
        expect.objectContaining({
          id: "command:browser_runtime",
          commandKey: "browser_runtime",
          triggers: expect.arrayContaining([
            { mode: "mention", prefix: "@浏览器" },
            { mode: "mention", prefix: "@Browser Agent" },
            { mode: "mention", prefix: "@Mini Tester" },
            { mode: "mention", prefix: "@Web Scheduler" },
            { mode: "mention", prefix: "@Web Manage" },
          ]),
        }),
        expect.objectContaining({
          id: "command:web_scrape",
          commandKey: "web_scrape",
          triggers: expect.arrayContaining([
            { mode: "mention", prefix: "@抓取" },
            { mode: "mention", prefix: "@Fetch" },
          ]),
        }),
        expect.objectContaining({
          id: "command:webpage_generate",
          commandKey: "webpage_generate",
          triggers: expect.arrayContaining([
            { mode: "mention", prefix: "@网页" },
            { mode: "mention", prefix: "@Web Composer" },
            { mode: "mention", prefix: "@HTML Preview" },
            { mode: "mention", prefix: "@Web Style" },
          ]),
        }),
        expect.objectContaining({
          id: "command:webpage_read",
          commandKey: "webpage_read",
          triggers: expect.arrayContaining([
            { mode: "mention", prefix: "@网页读取" },
            { mode: "mention", prefix: "@Read Webpage" },
            { mode: "mention", prefix: "@Get Homepage" },
            { mode: "mention", prefix: "@URL Summarize" },
          ]),
        }),
        expect.objectContaining({
          id: "command:modal_resource_search",
          commandKey: "modal_resource_search",
          triggers: expect.arrayContaining([
            { mode: "mention", prefix: "@素材" },
            { mode: "mention", prefix: "@Image Search" },
            { mode: "mention", prefix: "@Fetch Image" },
            { mode: "mention", prefix: "@Pinterest Image Search" },
            { mode: "mention", prefix: "@Video Search" },
          ]),
        }),
        expect.objectContaining({
          id: "command:deep_search",
          commandKey: "deep_search",
          triggers: expect.arrayContaining([
            { mode: "mention", prefix: "@深搜" },
            { mode: "mention", prefix: "@Researchers Pro" },
          ]),
        }),
        expect.objectContaining({
          id: "command:research",
          commandKey: "research",
          triggers: expect.arrayContaining([
            { mode: "mention", prefix: "@搜索" },
            { mode: "mention", prefix: "@Search" },
            { mode: "mention", prefix: "@Google Search" },
            { mode: "mention", prefix: "@Daily Search" },
            { mode: "mention", prefix: "@Search Agent" },
            { mode: "mention", prefix: "@Instagram Research" },
          ]),
        }),
        expect.objectContaining({
          id: "command:research_report",
          commandKey: "research_report",
          triggers: expect.arrayContaining([
            { mode: "mention", prefix: "@研报" },
            { mode: "mention", prefix: "@Report Search" },
          ]),
        }),
        expect.objectContaining({
          id: "command:channel_preview_runtime",
          commandKey: "channel_preview_runtime",
          triggers: expect.arrayContaining([
            { mode: "mention", prefix: "@渠道预览" },
            { mode: "mention", prefix: "@Instagram Preview" },
            { mode: "mention", prefix: "@TikTok Preview" },
            { mode: "mention", prefix: "@Twitter Preview" },
            { mode: "mention", prefix: "@YouTube Preview" },
          ]),
        }),
        expect.objectContaining({
          id: "command:publish_runtime",
          commandKey: "publish_runtime",
          triggers: expect.arrayContaining([
            { mode: "mention", prefix: "@发布" },
            { mode: "mention", prefix: "@TikTok Publish" },
            { mode: "mention", prefix: "@Twitter Publish" },
            { mode: "mention", prefix: "@YouTube Publish" },
          ]),
        }),
        expect.objectContaining({
          id: "command:competitor_research",
          commandKey: "competitor_research",
          triggers: expect.arrayContaining([
            { mode: "mention", prefix: "@竞品" },
            { mode: "mention", prefix: "@Product Search" },
          ]),
        }),
        expect.objectContaining({
          id: "command:typesetting",
          commandKey: "typesetting",
          binding: expect.objectContaining({
            skillId: "typesetting",
            executionKind: "cli",
            skillLocator: {
              source: "catalog",
              name: "typesetting",
            },
          }),
          renderContract: expect.objectContaining({
            resultKind: "tool_timeline",
            detailKind: "task_detail",
          }),
        }),
        expect.objectContaining({
          id: "command:broadcast_generate",
          commandKey: "broadcast_generate",
          triggers: expect.arrayContaining([
            { mode: "mention", prefix: "@播报" },
            { mode: "mention", prefix: "@Speaker 1" },
          ]),
        }),
        expect.objectContaining({
          id: "command:presentation_generate",
          commandKey: "presentation_generate",
          triggers: expect.arrayContaining([
            { mode: "mention", prefix: "@PPT" },
            { mode: "mention", prefix: "@Sales 1" },
          ]),
        }),
        expect.objectContaining({
          id: "command:poster_generate",
          commandKey: "poster_generate",
          triggers: expect.arrayContaining([
            { mode: "mention", prefix: "@海报" },
            { mode: "mention", prefix: "@Flyer 3" },
          ]),
        }),
        expect.objectContaining({
          id: "command:file_read_runtime",
          commandKey: "file_read_runtime",
          binding: expect.objectContaining({
            skillId: "summary",
            executionKind: "agent_turn",
            skillLocator: {
              source: "catalog",
              name: "file_read_runtime",
            },
          }),
          renderContract: expect.objectContaining({
            resultKind: "tool_timeline",
            detailKind: "json",
          }),
          triggers: expect.arrayContaining([
            { mode: "mention", prefix: "@读文件" },
            { mode: "mention", prefix: "@Read File Content" },
          ]),
        }),
        expect.objectContaining({
          id: "command:knowledge_pack",
          commandKey: "knowledge_pack",
          renderContract: expect.objectContaining({
            resultKind: "tool_timeline",
            detailKind: "json",
          }),
          triggers: expect.arrayContaining([
            { mode: "mention", prefix: "@资料" },
          ]),
        }),
        expect.objectContaining({
          id: "command:knowledge_settle",
          commandKey: "knowledge_settle",
          renderContract: expect.objectContaining({
            resultKind: "tool_timeline",
            detailKind: "json",
          }),
          triggers: expect.arrayContaining([
            { mode: "mention", prefix: "@沉淀资料" },
          ]),
        }),
        expect.objectContaining({
          id: "command:logo_decomposition",
          commandKey: "logo_decomposition",
          binding: expect.objectContaining({
            skillId: "analysis",
            executionKind: "agent_turn",
            skillLocator: {
              source: "catalog",
              name: "logo_decomposition",
            },
          }),
          renderContract: expect.objectContaining({
            resultKind: "tool_timeline",
            detailKind: "json",
          }),
          triggers: expect.arrayContaining([
            { mode: "mention", prefix: "@Logo拆解" },
            { mode: "mention", prefix: "@Image Logo Decomposition" },
          ]),
        }),
      ]),
    );
  });
});
