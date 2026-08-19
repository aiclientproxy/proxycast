import { describe, expect, it } from "vitest";
import {
  findModelBoundImageCommandEntryForModel,
  getSeededSkillCatalog,
  listSkillCatalogCommandEntries,
  listSkillCatalogSceneEntries,
  listSkillCatalogSkillEntries,
  parseSkillCatalog,
  type SkillCatalog,
} from "./skillCatalog";
import {
  buildLegacyCatalogWithSiteEntries,
  buildLegacyCloudSceneCatalog,
} from "./skillCatalogTestFixtures";

describe("skillCatalog pure catalog projection", () => {
  it("seeded 目录不应再暴露站点 adapter 或 browser assist 首页入口", () => {
    const catalog = getSeededSkillCatalog();

    expect(catalog.groups.map((group) => group.key)).toEqual(["general"]);
    expect(
      catalog.items.some((item) => item.execution.kind === "site_adapter"),
    ).toBe(false);
    expect(
      catalog.items.some(
        (item) =>
          item.defaultExecutorBinding === "browser_assist" ||
          Boolean(item.siteCapabilityBinding),
      ),
    ).toBe(false);
    expect(catalog.groups.find((group) => group.key === "general")).toEqual(
      expect.objectContaining({
        title: "通用技能",
        summary:
          "保留现有写作、调研、趋势选题与增长跟踪能力，作为站点组之外的创作技能入口。",
      }),
    );
    expect(
      catalog.items.find((item) => item.id === "account-performance-tracking"),
    ).toEqual(
      expect.objectContaining({
        title: "账号增长跟踪",
      }),
    );
    expect(
      catalog.items.find((item) => item.id === "personal-ip-knowledge-builder"),
    ).toEqual(
      expect.objectContaining({
        defaultExecutorBinding: "native_skill",
        skillBundle: expect.objectContaining({
          metadata: expect.objectContaining({
            Lime_knowledge_pack_type: "personal-profile",
            Lime_knowledge_family: "persona",
          }),
        }),
      }),
    );
    expect(
      catalog.items.find(
        (item) => item.id === "brand-persona-knowledge-builder",
      ),
    ).toEqual(
      expect.objectContaining({
        defaultExecutorBinding: "native_skill",
        skillBundle: expect.objectContaining({
          metadata: expect.objectContaining({
            Lime_knowledge_pack_type: "brand-persona",
            Lime_knowledge_family: "persona",
          }),
        }),
      }),
    );
    expect(
      catalog.items.find(
        (item) => item.id === "content-operations-knowledge-builder",
      ),
    ).toEqual(
      expect.objectContaining({
        defaultExecutorBinding: "native_skill",
        skillBundle: expect.objectContaining({
          metadata: expect.objectContaining({
            Lime_knowledge_pack_type: "content-operations",
            Lime_knowledge_family: "data",
          }),
        }),
      }),
    );
    expect(
      catalog.items.find(
        (item) => item.id === "brand-product-knowledge-builder",
      ),
    ).toEqual(
      expect.objectContaining({
        defaultExecutorBinding: "native_skill",
        skillBundle: expect.objectContaining({
          metadata: expect.objectContaining({
            Lime_knowledge_pack_type: "brand-product",
            Lime_knowledge_family: "data",
          }),
        }),
      }),
    );
    expect(
      listSkillCatalogSkillEntries(catalog).find(
        (entry) => entry.skillId === "personal-ip-knowledge-builder",
      ),
    ).toEqual(
      expect.objectContaining({
        kind: "skill",
        execution: expect.objectContaining({
          kind: "native_skill",
        }),
        skillBundle: expect.objectContaining({
          resourceSummary: expect.objectContaining({
            hasReferences: true,
            hasScripts: true,
          }),
        }),
      }),
    );
    expect(
      listSkillCatalogSkillEntries(catalog).find(
        (entry) => entry.skillId === "brand-persona-knowledge-builder",
      ),
    ).toEqual(
      expect.objectContaining({
        kind: "skill",
        execution: expect.objectContaining({
          kind: "native_skill",
        }),
        skillBundle: expect.objectContaining({
          resourceSummary: expect.objectContaining({
            hasReferences: true,
            hasScripts: false,
          }),
        }),
      }),
    );
    const reparsedCatalog = parseSkillCatalog(catalog);
    expect(
      listSkillCatalogSkillEntries(reparsedCatalog!).find(
        (entry) => entry.skillId === "personal-ip-knowledge-builder",
      ),
    ).toEqual(
      expect.objectContaining({
        skillBundle: expect.objectContaining({
          name: "personal-ip-knowledge-builder",
          resourceSummary: expect.objectContaining({
            hasReferences: true,
            hasScripts: true,
          }),
        }),
      }),
    );
  });

  it("含已退役站点 adapter 的旧目录应被拒绝", () => {
    expect(parseSkillCatalog(buildLegacyCatalogWithSiteEntries())).toBeNull();
  });

  it("应解析服务端下发的首页展示协议并允许 home-only command 无触发词", () => {
    const seeded = getSeededSkillCatalog();
    const catalog = parseSkillCatalog({
      ...seeded,
      version: "tenant-home-presentation",
      entries: [
        {
          id: "home:input-suggestion:email",
          kind: "command",
          title: "帮我写一封工作邮件",
          summary: "输入框 Tab 起手建议。",
          commandKey: "home_input_email",
          surfaceScopes: ["home"],
          homePresentation: {
            slot: "input_suggestion",
            label: "帮我写一封工作邮件",
            order: 10,
            prompt: "请帮我写一封工作邮件。",
          },
        },
      ],
    });

    const entry = listSkillCatalogCommandEntries(catalog!).find(
      (candidate) => candidate.commandKey === "home_input_email",
    );

    expect(entry).toEqual(
      expect.objectContaining({
        id: "home:input-suggestion:email",
        triggers: [],
        surfaceScopes: ["home"],
        homePresentation: expect.objectContaining({
          slot: "input_suggestion",
          label: "帮我写一封工作邮件",
          prompt: "请帮我写一封工作邮件。",
        }),
      }),
    );
  });

  it("解析旧版 raw skill catalog 时应把 cloud_scene 正规化为本地 agent_turn", () => {
    const catalog = parseSkillCatalog(buildLegacyCloudSceneCatalog());
    const skillItem = catalog?.items.find(
      (item) => item.id === "legacy-cloud-scene-skill",
    );
    const autoSceneEntry = listSkillCatalogSceneEntries(catalog!).find(
      (entry) => entry.id === "scene:legacy-cloud-scene-skill",
    );
    const commandEntry = listSkillCatalogCommandEntries(catalog!).find(
      (entry) => entry.commandKey === "legacy_voice_runtime",
    );

    expect(skillItem).toEqual(
      expect.objectContaining({
        defaultExecutorBinding: "agent_turn",
        executionLocation: "client_default",
        execution: expect.objectContaining({
          kind: "agent_turn",
        }),
      }),
    );
    expect(autoSceneEntry).toEqual(
      expect.objectContaining({
        linkedSkillId: "legacy-cloud-scene-skill",
        executionKind: "agent_turn",
      }),
    );
    expect(commandEntry).toEqual(
      expect.objectContaining({
        binding: expect.objectContaining({
          skillId: "legacy-cloud-scene-skill",
          executionKind: "agent_turn",
        }),
      }),
    );
    expect(JSON.stringify(catalog)).toContain(
      '"defaultExecutorBinding":"agent_turn"',
    );
    expect(JSON.stringify(catalog)).not.toContain(
      '"defaultExecutorBinding":"cloud_scene"',
    );
    expect(JSON.stringify(catalog)).not.toContain('"kind":"cloud_scene"');
  });

  it("应从统一目录中暴露 command 与 scene 扩展入口", () => {
    const seeded = getSeededSkillCatalog();
    const formEntry = listSkillCatalogCommandEntries(seeded).find(
      (entry) => entry.commandKey === "form_generate",
    );
    const imageEntry = listSkillCatalogCommandEntries(seeded).find(
      (entry) => entry.commandKey === "image_generate",
    );
    const posterEntry = listSkillCatalogCommandEntries(seeded).find(
      (entry) => entry.commandKey === "poster_generate",
    );
    const browserEntry = listSkillCatalogCommandEntries(seeded).find(
      (entry) => entry.commandKey === "browser_runtime",
    );
    const webScrapeEntry = listSkillCatalogCommandEntries(seeded).find(
      (entry) => entry.commandKey === "web_scrape",
    );
    const webpageEntry = listSkillCatalogCommandEntries(seeded).find(
      (entry) => entry.commandKey === "webpage_generate",
    );
    const webpageReadEntry = listSkillCatalogCommandEntries(seeded).find(
      (entry) => entry.commandKey === "webpage_read",
    );
    const competitorEntry = listSkillCatalogCommandEntries(seeded).find(
      (entry) => entry.commandKey === "competitor_research",
    );
    const codeEntry = listSkillCatalogCommandEntries(seeded).find(
      (entry) => entry.commandKey === "code_runtime",
    );
    const voiceEntry = listSkillCatalogCommandEntries(seeded).find(
      (entry) => entry.commandKey === "voice_runtime",
    );
    const channelPreviewEntry = listSkillCatalogCommandEntries(seeded).find(
      (entry) => entry.commandKey === "channel_preview_runtime",
    );
    const writingEntry = listSkillCatalogCommandEntries(seeded).find(
      (entry) => entry.commandKey === "writing_runtime",
    );
    const uploadEntry = listSkillCatalogCommandEntries(seeded).find(
      (entry) => entry.commandKey === "upload_runtime",
    );
    const complianceEntry = listSkillCatalogCommandEntries(seeded).find(
      (entry) => entry.commandKey === "publish_compliance",
    );
    const publishEntry = listSkillCatalogCommandEntries(seeded).find(
      (entry) => entry.commandKey === "publish_runtime",
    );
    const logoDecompositionEntry = listSkillCatalogCommandEntries(seeded).find(
      (entry) => entry.commandKey === "logo_decomposition",
    );
    const fileReadEntry = listSkillCatalogCommandEntries(seeded).find(
      (entry) => entry.commandKey === "file_read_runtime",
    );

    expect(
      listSkillCatalogCommandEntries(seeded).map((entry) => entry.commandKey),
    ).toEqual(
      expect.arrayContaining([
        "image_generate",
        "cover_generate",
        "poster_generate",
        "video_generate",
        "broadcast_generate",
        "modal_resource_search",
        "research",
        "deep_search",
        "research_report",
        "competitor_research",
        "read_pdf",
        "file_read_runtime",
        "summary",
        "translation",
        "analysis",
        "logo_decomposition",
        "transcription_generate",
        "web_scrape",
        "webpage_read",
        "url_parse",
        "typesetting",
        "form_generate",
        "browser_runtime",
        "voice_runtime",
        "growth_runtime",
        "writing_runtime",
        "channel_preview_runtime",
        "upload_runtime",
        "code_runtime",
        "publish_runtime",
        "publish_compliance",
      ]),
    );
    expect(formEntry?.renderContract).toMatchObject({
      resultKind: "form",
      detailKind: "json",
      supportsStreaming: true,
      supportsTimeline: true,
    });
    expect(imageEntry?.binding).toMatchObject({
      skillId: "image_generate",
      executionKind: "task_queue",
      requestDefaults: {
        imageWorkbench: "true",
      },
      intentConfirmation: {
        id: "plain_image_generation",
        ruleKey: "agentChat.inputIntent.imageGeneration.rules",
        confirmationKey: "agentChat.inputIntent.imageGeneration.confirm",
        systemPromptKey: "agentChat.inputIntent.imageGeneration.systemPrompt",
      },
    });
    expect(posterEntry?.binding).toMatchObject({
      skillId: "image_generate",
      executionKind: "task_queue",
    });
    expect(browserEntry?.renderContract).toMatchObject({
      resultKind: "tool_timeline",
      detailKind: "json",
      supportsStreaming: true,
      supportsTimeline: true,
    });
    expect(browserEntry?.triggers).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ prefix: "@Browser Agent" }),
        expect.objectContaining({ prefix: "@Mini Tester" }),
        expect.objectContaining({ prefix: "@Web Scheduler" }),
        expect.objectContaining({ prefix: "@Web Manage" }),
      ]),
    );
    expect(webScrapeEntry?.binding).toMatchObject({
      skillId: "url_parse",
      executionKind: "task_queue",
    });
    expect(webScrapeEntry?.triggers).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ prefix: "@抓取" }),
        expect.objectContaining({ prefix: "@Fetch" }),
      ]),
    );
    expect(webpageReadEntry?.binding).toMatchObject({
      skillId: "url_parse",
      executionKind: "task_queue",
    });
    expect(competitorEntry?.binding).toMatchObject({
      skillId: "report_generate",
      executionKind: "agent_turn",
    });
    expect(codeEntry?.renderContract).toMatchObject({
      resultKind: "tool_timeline",
      detailKind: "json",
      supportsStreaming: true,
      supportsTimeline: true,
    });
    expect(voiceEntry?.renderContract).toMatchObject({
      resultKind: "tool_timeline",
      detailKind: "scene_detail",
      supportsStreaming: true,
      supportsTimeline: true,
    });
    expect(channelPreviewEntry?.binding).toMatchObject({
      skillId: "content_post_with_cover",
      executionKind: "native_skill",
    });
    expect(writingEntry?.binding).toMatchObject({
      skillId: "content_post_with_cover",
      executionKind: "native_skill",
    });
    expect(writingEntry?.triggers).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ prefix: "@Web Copy" }),
      ]),
    );
    expect(webpageEntry?.triggers).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ prefix: "@Web Composer" }),
        expect.objectContaining({ prefix: "@HTML Preview" }),
        expect.objectContaining({ prefix: "@Web Style" }),
      ]),
    );
    expect(fileReadEntry?.binding).toMatchObject({
      skillId: "summary",
      executionKind: "agent_turn",
    });
    expect(
      listSkillCatalogCommandEntries(seeded).find(
        (entry) => entry.commandKey === "research",
      )?.triggers,
    ).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ prefix: "@Search Agent" }),
        expect.objectContaining({ prefix: "@Instagram Research" }),
      ]),
    );
    expect(fileReadEntry?.triggers).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ prefix: "@读文件" }),
        expect.objectContaining({ prefix: "@Read File Content" }),
      ]),
    );
    expect(uploadEntry?.binding).toMatchObject({
      skillId: "content_post_with_cover",
      executionKind: "native_skill",
    });
    expect(complianceEntry?.binding).toMatchObject({
      skillId: "analysis",
      executionKind: "agent_turn",
    });
    expect(logoDecompositionEntry?.binding).toMatchObject({
      skillId: "analysis",
      executionKind: "agent_turn",
    });
    expect(logoDecompositionEntry?.triggers).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ prefix: "@Logo拆解" }),
        expect.objectContaining({ prefix: "@Image Logo Decomposition" }),
      ]),
    );
    expect(publishEntry?.renderContract).toMatchObject({
      resultKind: "artifact",
      detailKind: "artifact_detail",
      supportsStreaming: true,
      supportsTimeline: true,
    });
  });

  it("服务端 Lime Cloud 下发的图片模型命令应保持为标准图片命令入口", () => {
    const seeded = getSeededSkillCatalog();
    const catalog = parseSkillCatalog({
      ...seeded,
      version: "cloud-image-command-2026-05-14",
      tenantId: "tenant-cloud",
      syncedAt: "2026-05-14T00:00:00.000Z",
      entries: [
        ...seeded.entries,
        {
          id: "command:image-model:nano_banana_2_cloud",
          kind: "command",
          title: "Nano Banana 2 Cloud",
          summary: "云端目录声明的图片模型入口。",
          command_key: "image_model_nano_banana_2_cloud",
          triggers: [{ mode: "mention", prefix: "@Nano Banana Cloud" }],
          binding: {
            skill_id: "image_generate",
            execution_kind: "task_queue",
            request_defaults: {
              image_workbench: "true",
              model_bound_image_task: "true",
              provider_id: "fal",
              model_id: "fal-ai/nano-banana-2",
              bindingSource: "lime_cloud",
            },
          },
          render_contract: {
            result_kind: "image_gallery",
            detail_kind: "media_detail",
            supports_streaming: true,
            supports_timeline: true,
          },
        },
      ],
    });

    expect(
      listSkillCatalogCommandEntries(catalog!).find(
        (entry) => entry.commandKey === "image_model_nano_banana_2_cloud",
      ),
    ).toMatchObject({
      commandKey: "image_model_nano_banana_2_cloud",
      binding: {
        requestDefaults: expect.objectContaining({
          bindingSource: "lime_cloud",
        }),
      },
    });
    expect(
      findModelBoundImageCommandEntryForModel(
        catalog!,
        "fal",
        "fal-ai/nano-banana-2",
      ),
    ).toMatchObject({
      commandKey: "image_model_nano_banana_2_cloud",
    });
  });

  it("解析远端 scene entry 时保留模板、占位和 request defaults", () => {
    const remoteCatalog: SkillCatalog = {
      ...getSeededSkillCatalog(),
      tenantId: "tenant-demo",
      version: "tenant-2026-03-30",
      syncedAt: "2026-03-30T12:00:00.000Z",
      entries: [
        {
          id: "scene:campaign-launch",
          kind: "scene",
          title: "新品发布场景",
          summary: "把链接解析、配图与封面串成一个可复用场景。",
          sceneKey: "campaign-launch",
          commandPrefix: "/campaign-launch",
          linkedEntryId: "skill:tenant-daily-briefing",
          placeholder: "输入新品链接或发布主题",
          templates: [
            {
              id: "default",
              title: "发布启动",
              description: "从一个主题启动发布链路",
              prompt: "请帮我规划新品发布内容。",
            },
          ],
          aliases: ["launch", "campaign"],
          executionKind: "scene",
          requestDefaults: {
            executionStrategy: "react",
          },
          renderContract: {
            resultKind: "tool_timeline",
            detailKind: "scene_detail",
            supportsStreaming: true,
            supportsTimeline: true,
          },
        },
      ],
    };

    const catalog = parseSkillCatalog(remoteCatalog);

    expect(
      listSkillCatalogSceneEntries(catalog!).map((entry) => entry.sceneKey),
    ).toEqual(expect.arrayContaining(["campaign-launch", "story-video-suite"]));
    expect(
      listSkillCatalogSceneEntries(catalog!).map((entry) => entry.sceneKey),
    ).not.toContain("legacy-site-export");
    expect(
      listSkillCatalogSceneEntries(catalog!).find(
        (entry) => entry.sceneKey === "campaign-launch",
      ),
    ).toMatchObject({
      linkedEntryId: "skill:tenant-daily-briefing",
      placeholder: "输入新品链接或发布主题",
      templates: [
        {
          id: "default",
          title: "发布启动",
          description: "从一个主题启动发布链路",
          prompt: "请帮我规划新品发布内容。",
        },
      ],
      requestDefaults: {
        executionStrategy: "react",
      },
    });
  });
});
