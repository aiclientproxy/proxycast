import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ServiceSkillHomeItem } from "../service-skills/types";
import {
  buildServiceSceneLaunchRequestMetadata,
  matchesRuntimeSceneEntry,
  parseRuntimeSceneCommand,
  RuntimeSceneLaunchValidationError,
  resolveRuntimeSceneLaunchRequest,
} from "./serviceSkillSceneLaunch";

const mockGetSkillCatalog = vi.hoisted(() => vi.fn());
const mockListSkillCatalogSceneEntries = vi.hoisted(() => vi.fn());
const mockListServiceSkills = vi.hoisted(() => vi.fn());
const mockGetOrCreateDefaultProject = vi.hoisted(() => vi.fn());
const mockResolveOemCloudRuntimeContext = vi.hoisted(() => vi.fn());
const mockSiteListAdapters = vi.hoisted(() => vi.fn());

vi.mock("@/lib/api/skillCatalog", () => ({
  getSkillCatalog: () => mockGetSkillCatalog(),
  listSkillCatalogSceneEntries: (catalog: unknown) =>
    mockListSkillCatalogSceneEntries(catalog),
}));

vi.mock("@/lib/api/serviceSkills", async () => {
  const actual = await vi.importActual<
    typeof import("@/lib/api/serviceSkills")
  >("@/lib/api/serviceSkills");
  return {
    ...actual,
    listServiceSkills: () => mockListServiceSkills(),
  };
});

vi.mock("@/lib/api/project", () => ({
  getOrCreateDefaultProject: () => mockGetOrCreateDefaultProject(),
}));

vi.mock("@/lib/api/oemCloudRuntime", () => ({
  resolveOemCloudRuntimeContext: () => mockResolveOemCloudRuntimeContext(),
}));

vi.mock("@/lib/webview-api", () => ({
  siteListAdapters: (...args: unknown[]) => mockSiteListAdapters(...args),
}));

function createCloudSceneSkill(): ServiceSkillHomeItem {
  return {
    id: "cloud-video-dubbing",
    skillKey: "campaign-launch",
    title: "视频配音",
    summary: "围绕视频文案与素材整理一版可继续加工的配音稿。",
    category: "视频创作",
    outputHint: "配音文案 + 结果摘要",
    source: "cloud_catalog",
    runnerType: "instant",
    defaultExecutorBinding: "agent_turn",
    executionLocation: "client_default",
    version: "seed-v1",
    badge: "云目录",
    recentUsedAt: null,
    isRecent: false,
    runnerLabel: "立即开始",
    runnerTone: "slate",
    runnerDescription: "直接在当前工作区整理首版配音稿。",
    actionLabel: "对话内补参",
    automationStatus: null,
    slotSchema: [],
  };
}

function createXArticleExportSkill(): ServiceSkillHomeItem {
  return {
    id: "x-article-export",
    skillKey: "x-article-export",
    title: "X 文章转存",
    summary: "复用 X 登录态把长文导出成 Markdown 和图片目录。",
    category: "站点采集",
    outputHint: "Markdown 正文 + 图片目录",
    source: "local_custom",
    runnerType: "instant",
    defaultExecutorBinding: "browser_assist",
    executionLocation: "client_default",
    version: "seed-v1",
    badge: "本地技能",
    recentUsedAt: null,
    isRecent: false,
    runnerLabel: "浏览器站点执行",
    runnerTone: "emerald",
    runnerDescription: "直接复用浏览器登录态执行。",
    actionLabel: "启动采集",
    automationStatus: null,
    slotSchema: [
      {
        key: "article_url",
        label: "X 文章链接",
        type: "url",
        required: true,
        placeholder: "https://x.com/<账号>/article/<文章ID>",
        helpText: "支持 x.com 和 twitter.com 的 Article 链接。",
      },
      {
        key: "target_language",
        label: "目标语言",
        type: "text",
        required: false,
        defaultValue: "中文",
        placeholder: "例如 中文、英文、日文",
        helpText: "仅翻译正文，代码块、链接和图片路径保持原样。",
      },
    ],
    readinessRequirements: {
      requiresBrowser: true,
      requiresProject: true,
    },
    siteCapabilityBinding: {
      autoRun: true,
      requireAttachedSession: true,
      saveMode: "project_resource",
      siteLabel: "X",
      adapterMatch: {
        urlArgName: "url",
        requiredCapabilities: ["article_export", "markdown_bundle"],
        hostAliases: ["twitter.com", "www.twitter.com", "www.x.com"],
      },
      slotArgMap: {
        article_url: "url",
        target_language: "target_language",
      },
    },
    sceneBinding: {
      sceneKey: "x-article-export",
      commandPrefix: "/x文章转存",
      title: "X文章转存",
      summary: "把 X 长文导出成 Markdown。",
      aliases: ["x文章转存", "x转存"],
    },
  };
}

describe("serviceSkillSceneLaunch", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetSkillCatalog.mockResolvedValue({ entries: [] });
    mockListSkillCatalogSceneEntries.mockReturnValue([]);
    mockListServiceSkills.mockResolvedValue([]);
    mockGetOrCreateDefaultProject.mockResolvedValue({
      id: "project-default",
    });
    mockResolveOemCloudRuntimeContext.mockReturnValue(null);
    mockSiteListAdapters.mockResolvedValue([
      {
        name: "x/article-export",
        domain: "x.com",
        description: "导出 X 长文",
        read_only: true,
        capabilities: ["article_export", "markdown_bundle"],
        input_schema: {},
        example_args: {},
        example: "",
      },
    ]);
  });

  it("应解析 slash scene 命令", () => {
    expect(
      parseRuntimeSceneCommand("/campaign-launch 帮我做一版新品活动方案"),
    ).toEqual({
      sceneKey: "campaign-launch",
      userInput: "帮我做一版新品活动方案",
    });
    expect(
      parseRuntimeSceneCommand("/x文章转存 https://x.com/a/article/1"),
    ).toEqual({
      sceneKey: "x文章转存",
      userInput: "https://x.com/a/article/1",
    });
    expect(parseRuntimeSceneCommand("campaign-launch")).toBeNull();
  });

  it("scene 匹配时应同时支持 sceneKey、commandPrefix 与 alias", () => {
    const entry = {
      sceneKey: "campaign-launch",
      commandPrefix: "/campaign-launch",
      aliases: ["campaign", "launch"],
    };

    expect(matchesRuntimeSceneEntry(entry as never, "campaign-launch")).toBe(
      true,
    );
    expect(matchesRuntimeSceneEntry(entry as never, "/campaign-launch")).toBe(
      true,
    );
    expect(matchesRuntimeSceneEntry(entry as never, "campaign")).toBe(true);
    expect(matchesRuntimeSceneEntry(entry as never, "other")).toBe(false);
  });

  it("scene entry 缺失 linkedSkillId 时，仍应通过 service skill 的 sceneBinding 回落命中", async () => {
    mockGetSkillCatalog.mockResolvedValueOnce({ entries: [] });
    mockListSkillCatalogSceneEntries.mockReturnValueOnce([
      {
        id: "scene:x-article-export",
        kind: "scene",
        title: "X文章转存",
        summary: "把 X 长文导出成 Markdown。",
        sceneKey: "x-article-export",
        commandPrefix: "/x文章转存",
        aliases: ["x转存"],
        executionKind: "browser_assist",
      },
    ]);

    const request = await resolveRuntimeSceneLaunchRequest({
      rawText:
        "/x转存 https://x.com/GoogleCloudTech/article/2033953579824758855",
      serviceSkills: [createXArticleExportSkill()],
      projectId: "project-1",
      contentId: "content-1",
    });

    expect(request).not.toBeNull();
    expect(request?.skill.id).toBe("x-article-export");
  });

  it("应构建统一的本地 service scene launch request metadata", async () => {
    mockGetSkillCatalog.mockResolvedValueOnce({ entries: [] });
    mockListSkillCatalogSceneEntries.mockReturnValueOnce([
      {
        id: "scene:campaign-launch",
        kind: "scene",
        title: "活动启动场景",
        summary: "围绕活动目标生成启动方案。",
        sceneKey: "campaign-launch",
        commandPrefix: "/campaign-launch",
        linkedSkillId: "cloud-video-dubbing",
        executionKind: "agent_turn",
      },
    ]);
    const request = await resolveRuntimeSceneLaunchRequest({
      rawText: "/campaign-launch 帮我做一版新品活动启动方案",
      serviceSkills: [createCloudSceneSkill()],
      projectId: "project-1",
      contentId: "content-1",
    });

    expect(request).not.toBeNull();
    const requestMetadata = buildServiceSceneLaunchRequestMetadata(
      undefined,
      request!.requestContext,
    );

    expect(requestMetadata).toMatchObject({
      harness: {
        service_scene_launch: {
          kind: "local_service_skill",
          service_scene_run: {
            scene_key: "campaign-launch",
            skill_id: "cloud-video-dubbing",
            skill_locator: {
              source: "catalog",
              name: "campaign-launch",
            },
            project_id: "project-1",
            content_id: "content-1",
            user_input: "帮我做一版新品活动启动方案",
          },
        },
      },
    });
    expect(request?.dispatchText).toContain("[技能任务] 视频配音");
  });

  it("service scene 应根据 skill 声明注入 service_scene_launch metadata", async () => {
    mockGetSkillCatalog.mockResolvedValueOnce({ entries: [] });
    mockListSkillCatalogSceneEntries.mockReturnValueOnce([
      {
        id: "scene:x-article-export",
        kind: "scene",
        title: "X文章转存",
        summary: "把 X 长文导出成 Markdown。",
        sceneKey: "x-article-export",
        commandPrefix: "/x文章转存",
        aliases: ["x文章转存", "x转存"],
        linkedSkillId: "x-article-export",
        executionKind: "site_adapter",
      },
    ]);
    const request = await resolveRuntimeSceneLaunchRequest({
      rawText:
        "/x文章转存 https://x.com/GoogleCloudTech/article/2033953579824758855",
      serviceSkills: [createXArticleExportSkill()],
      projectId: "project-1",
      contentId: "content-1",
    });

    expect(request).not.toBeNull();
    const requestMetadata = buildServiceSceneLaunchRequestMetadata(
      {
        harness: {
          chat_mode: "general",
        },
      },
      request!.requestContext,
    );

    expect(requestMetadata).toMatchObject({
      harness: {
        chat_mode: "general",
        service_scene_launch: {
          kind: "local_service_skill",
          service_scene_run: {
            scene_key: "x-article-export",
            skill_id: "x-article-export",
            project_id: "project-1",
            content_id: "content-1",
            slot_values: {
              article_url:
                "https://x.com/GoogleCloudTech/article/2033953579824758855",
            },
          },
        },
      },
    });
  });

  it("site skill scene 缺少当前项目时应显式报错", async () => {
    mockGetSkillCatalog.mockResolvedValueOnce({ entries: [] });
    mockListSkillCatalogSceneEntries.mockReturnValueOnce([
      {
        id: "scene:x-article-export",
        kind: "scene",
        title: "X文章转存",
        summary: "把 X 长文导出成 Markdown。",
        sceneKey: "x-article-export",
        commandPrefix: "/x文章转存",
        aliases: ["x文章转存", "x转存"],
        linkedSkillId: "x-article-export",
        executionKind: "site_adapter",
      },
    ]);

    let capturedError: RuntimeSceneLaunchValidationError | null = null;
    try {
      await resolveRuntimeSceneLaunchRequest({
        rawText:
          "/x文章转存 https://x.com/GoogleCloudTech/article/2033953579824758855",
        serviceSkills: [createXArticleExportSkill()],
        projectId: null,
        contentId: null,
      });
    } catch (error) {
      capturedError = error as RuntimeSceneLaunchValidationError;
    }

    expect(capturedError).toBeInstanceOf(RuntimeSceneLaunchValidationError);
    expect(capturedError?.message).toBe("还需要选择项目工作区，补齐后再继续。");
    expect(capturedError?.gateRequest).toMatchObject({
      sceneKey: "x-article-export",
      commandPrefix: "/x文章转存",
      skillId: "x-article-export",
      fields: [
        {
          kind: "project",
          key: "project_id",
          label: "项目工作区",
        },
      ],
    });

    expect(mockGetOrCreateDefaultProject).not.toHaveBeenCalled();
  });

  it("site skill scene 缺少必填 slot 时应返回可恢复的 gate request", async () => {
    mockGetSkillCatalog.mockResolvedValueOnce({ entries: [] });
    mockListSkillCatalogSceneEntries.mockReturnValueOnce([
      {
        id: "scene:x-article-export",
        kind: "scene",
        title: "X文章转存",
        summary: "把 X 长文导出成 Markdown。",
        sceneKey: "x-article-export",
        commandPrefix: "/x文章转存",
        aliases: ["x文章转存", "x转存"],
        linkedSkillId: "x-article-export",
        executionKind: "site_adapter",
      },
    ]);

    let capturedError: RuntimeSceneLaunchValidationError | null = null;
    try {
      await resolveRuntimeSceneLaunchRequest({
        rawText: "/x文章转存",
        serviceSkills: [createXArticleExportSkill()],
        projectId: "project-1",
        contentId: null,
      });
    } catch (error) {
      capturedError = error as RuntimeSceneLaunchValidationError;
    }

    expect(capturedError).toBeInstanceOf(RuntimeSceneLaunchValidationError);
    expect(capturedError?.message).toBe("还差X 文章链接，补齐后再继续。");
    expect(capturedError?.gateRequest).toMatchObject({
      sceneKey: "x-article-export",
      fields: [
        {
          kind: "slot",
          key: "article_url",
          label: "X 文章链接",
          slotType: "url",
        },
      ],
    });
  });

  it("当前首页列表未暴露 scene 绑定 skill 时，应回退完整技能目录解析", async () => {
    mockGetSkillCatalog.mockResolvedValueOnce({ entries: [] });
    mockListSkillCatalogSceneEntries.mockReturnValueOnce([
      {
        id: "scene:x-article-export",
        kind: "scene",
        title: "X文章转存",
        summary: "把 X 长文导出成 Markdown。",
        sceneKey: "x-article-export",
        commandPrefix: "/x文章转存",
        linkedSkillId: "x-article-export",
        executionKind: "site_adapter",
      },
    ]);
    mockListServiceSkills.mockResolvedValueOnce([createXArticleExportSkill()]);

    const request = await resolveRuntimeSceneLaunchRequest({
      rawText:
        "/x文章转存 https://x.com/GoogleCloudTech/article/2033953579824758855",
      serviceSkills: [],
      projectId: "project-1",
    });

    expect(request?.skill.id).toBe("x-article-export");
    expect(request?.requestContext).toMatchObject({
      kind: "local_service_skill",
      service_scene_run: expect.objectContaining({
        scene_key: "x-article-export",
        skill_id: "x-article-export",
        project_id: "project-1",
      }),
    });
  });

  it("site skill scene 应根据 URL 和能力标签解析真正的 adapter", async () => {
    mockGetSkillCatalog.mockResolvedValueOnce({ entries: [] });
    mockListSkillCatalogSceneEntries.mockReturnValueOnce([
      {
        id: "scene:x-article-export",
        kind: "scene",
        title: "X文章转存",
        summary: "把 X 长文导出成 Markdown。",
        sceneKey: "x-article-export",
        commandPrefix: "/x文章转存",
        linkedSkillId: "x-article-export",
        executionKind: "site_adapter",
      },
    ]);
    mockSiteListAdapters.mockResolvedValueOnce([
      {
        name: "x/timeline",
        domain: "x.com",
        description: "读取 X 时间线",
        read_only: true,
        capabilities: ["timeline"],
        input_schema: {},
        example_args: {},
        example: "",
      },
      {
        name: "x/article-export",
        domain: "x.com",
        description: "导出 X 长文",
        read_only: true,
        capabilities: ["article_export", "markdown_bundle"],
        input_schema: {},
        example_args: {},
        example: "",
      },
    ]);

    const request = await resolveRuntimeSceneLaunchRequest({
      rawText:
        "/x文章转存 https://twitter.com/GoogleCloudTech/article/2033953579824758855",
      serviceSkills: [createXArticleExportSkill()],
      projectId: "project-1",
    });

    expect(request?.requestContext).toMatchObject({
      kind: "local_service_skill",
      service_scene_run: expect.objectContaining({
        scene_key: "x-article-export",
        slot_values: {
          article_url:
            "https://twitter.com/GoogleCloudTech/article/2033953579824758855",
        },
      }),
    });
  });
});
