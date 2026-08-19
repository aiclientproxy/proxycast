import { act } from "react";
import { createRoot } from "react-dom/client";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { toast } from "sonner";
import { changeLimeLocale, getLimeI18n } from "@/i18n/createI18n";
import type { ServiceSkillHomeItem } from "../service-skills/types";
import type { Message } from "../types";
import type { InitialDispatchPreviewSnapshot } from "./workspaceSendHelpers";
import {
  listMentionEntryUsage,
  recordMentionEntryUsage,
} from "../skill-selection/mentionEntryUsage";
import { listServiceSkillUsage } from "../service-skills/storage";
import { useWorkspaceSendActions } from "./useWorkspaceSendActions";
import { listSlashEntryUsage } from "../skill-selection/slashEntryUsage";
import {
  saveSkillCatalog,
  upsertLocalModelBoundImageCommandBinding,
} from "@/lib/api/skillCatalog";

const mockGetSkillCatalog = vi.hoisted(() => vi.fn());
const mockListSkillCatalogSceneEntries = vi.hoisted(() => vi.fn());
const mockGetOrCreateDefaultProject = vi.hoisted(() => vi.fn());
const mockResolveOemCloudRuntimeContext = vi.hoisted(() => vi.fn());
const mockGetOemCloudBootstrapSnapshot = vi.hoisted(() => vi.fn());
const mockUseGlobalMediaGenerationDefaults = vi.hoisted(() => vi.fn());
const mockInstallSkillFromPromptInstruction = vi.hoisted(() => vi.fn());

vi.mock("@/lib/api/skillCatalog", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api/skillCatalog")>(
    "@/lib/api/skillCatalog",
  );

  return {
    ...actual,
    getSkillCatalog: () => mockGetSkillCatalog(),
    listSkillCatalogSceneEntries: (catalog: unknown) =>
      mockListSkillCatalogSceneEntries(catalog),
  };
});

vi.mock("@/lib/api/oemCloudRuntime", () => ({
  resolveOemCloudRuntimeContext: () => mockResolveOemCloudRuntimeContext(),
}));

vi.mock("@/lib/oemCloudSession", async () => {
  const actual = await vi.importActual<typeof import("@/lib/oemCloudSession")>(
    "@/lib/oemCloudSession",
  );

  return {
    ...actual,
    getOemCloudBootstrapSnapshot: () => mockGetOemCloudBootstrapSnapshot(),
  };
});

vi.mock("@/hooks/useGlobalMediaGenerationDefaults", () => ({
  readGlobalMediaGenerationDefaults: () =>
    mockUseGlobalMediaGenerationDefaults(),
}));

vi.mock("@/lib/skills/skillInstallPrompt", async () => {
  const actual = await vi.importActual<
    typeof import("@/lib/skills/skillInstallPrompt")
  >("@/lib/skills/skillInstallPrompt");

  return {
    ...actual,
    installSkillFromPromptInstruction: (
      ...args: Parameters<typeof actual.installSkillFromPromptInstruction>
    ) => mockInstallSkillFromPromptInstruction(...args),
  };
});

vi.mock("@/lib/api/project", async () => {
  const actual =
    await vi.importActual<typeof import("@/lib/api/project")>(
      "@/lib/api/project",
    );

  return {
    ...actual,
    getOrCreateDefaultProject: () => mockGetOrCreateDefaultProject(),
  };
});

type HookProps = Parameters<typeof useWorkspaceSendActions>[0];

interface HookHarness {
  getValue: () => ReturnType<typeof useWorkspaceSendActions>;
  unmount: () => void;
}

const mockSendMessage = vi.fn<HookProps["sendMessage"]>(async () => undefined);
const mockFinalizeAfterSendSuccess = vi.fn();
const mockRollbackAfterSendFailure = vi.fn();
const mockSetInput = vi.fn();
const mockSetMentionedCharacters = vi.fn();
const mockSetChatMessages = vi.fn();
const mockSetChatToolPreferences = vi.fn();
const mockOpenRuntimeSceneGate = vi.fn(async () => undefined);
const mockCreateImageGenerationTask = vi.fn();
const mockEnsureSessionForCommandMetadata = vi.fn<
  NonNullable<HookProps["ensureSessionForCommandMetadata"]>
>(async () => null);
const mockPrepareImageWorkbenchSkillSend = vi.fn<
  NonNullable<HookProps["prepareImageWorkbenchSkillSend"]>
>(() => true);
const mockResolveImageWorkbenchCommandRequest = vi.fn<
  HookProps["resolveImageWorkbenchCommandRequest"]
>(() => null);

function createDeferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((nextResolve, nextReject) => {
    resolve = nextResolve;
    reject = nextReject;
  });
  return { promise, resolve, reject };
}

function createGithubSiteSkill(): ServiceSkillHomeItem {
  return {
    id: "github-repo-radar",
    title: "GitHub 仓库线索检索",
    summary: "复用 GitHub 登录态检索项目。",
    category: "情报研究",
    outputHint: "仓库列表 + 关键线索",
    source: "cloud_catalog",
    runnerType: "instant",
    defaultExecutorBinding: "agent_turn",
    executionLocation: "client_default",
    version: "seed-v1",
    badge: "云目录",
    recentUsedAt: null,
    isRecent: false,
    runnerLabel: "Agent 调度",
    runnerTone: "slate",
    runnerDescription: "通过统一 Agent 发送链执行。",
    actionLabel: "立即启动",
    automationStatus: null,
    slotSchema: [
      {
        key: "repository_query",
        label: "检索主题",
        type: "text",
        required: true,
        placeholder: "例如 AI Agent",
      },
    ],
  };
}

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
    slotSchema: [
      {
        key: "reference_video",
        label: "参考视频链接/素材",
        type: "url",
        required: false,
        placeholder: "输入视频链接",
      },
    ],
  };
}

function createGrowthTrackingSkill(): ServiceSkillHomeItem {
  return {
    id: "account-performance-tracking",
    skillKey: "account-performance-tracking",
    title: "账号增长跟踪",
    summary: "围绕目标账号整理首版增长打法，并持续跟踪关键指标与提醒条件。",
    category: "内容运营",
    outputHint: "增长策略 + 跟踪指标",
    source: "cloud_catalog",
    runnerType: "managed",
    defaultExecutorBinding: "automation_job",
    executionLocation: "client_default",
    version: "seed-v1",
    badge: "云目录",
    recentUsedAt: null,
    isRecent: false,
    runnerLabel: "持续跟踪",
    runnerTone: "slate",
    runnerDescription: "先产出首版增长方案，再持续跟踪。",
    actionLabel: "立即启动",
    automationStatus: null,
    slotSchema: [
      {
        key: "platform",
        label: "目标平台",
        type: "platform",
        required: true,
        defaultValue: "x",
        placeholder: "选择账号平台",
      },
      {
        key: "account_list",
        label: "参考账号 / 目标账号",
        type: "account_list",
        required: true,
        placeholder: "每行一个账号，或用逗号分隔多个账号",
      },
      {
        key: "report_cadence",
        label: "回报频率",
        type: "schedule_time",
        required: false,
        placeholder: "例如 每天 10:00",
      },
      {
        key: "alert_threshold",
        label: "告警阈值",
        type: "text",
        required: false,
        placeholder: "例如 日增粉低于 1%",
      },
    ],
    readinessRequirements: {
      requiresModel: true,
      requiresProject: true,
    },
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
    defaultExecutorBinding: "agent_turn",
    executionLocation: "client_default",
    version: "seed-v1",
    badge: "本地技能",
    recentUsedAt: null,
    isRecent: false,
    runnerLabel: "Agent 调度",
    runnerTone: "slate",
    runnerDescription: "通过统一 Agent 发送链执行。",
    actionLabel: "立即启动",
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
      requiresProject: true,
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

function createGeneralServiceSkill(
  overrides: Partial<ServiceSkillHomeItem> = {},
): ServiceSkillHomeItem {
  return {
    id: "research",
    title: "搜索",
    summary: "针对当前主题执行联网检索与轻量调研。",
    category: "通用技能",
    outputHint: "结论 + 来源",
    source: "cloud_catalog",
    runnerType: "instant",
    defaultExecutorBinding: "agent_turn",
    executionLocation: "client_default",
    version: "seed-v1",
    badge: "云目录",
    recentUsedAt: null,
    isRecent: false,
    runnerLabel: "Agent 调度",
    runnerTone: "slate",
    runnerDescription: "通过统一 Agent 发送链执行。",
    actionLabel: "立即启动",
    automationStatus: null,
    slotSchema: [],
    ...overrides,
  };
}

function createBootstrapDispatchSnapshot(
  prompt = "请开始处理这个任务",
): InitialDispatchPreviewSnapshot {
  return {
    key: "bootstrap-dispatch-1",
    prompt,
    images: [],
  };
}

function mountHook(initialProps?: Partial<HookProps>): HookHarness {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  let hookValue: ReturnType<typeof useWorkspaceSendActions> | null = null;
  const props: HookProps = {
    input: "继续处理当前话题",
    setInput: mockSetInput,
    mentionedCharacters: [],
    setMentionedCharacters: mockSetMentionedCharacters,
    chatToolPreferences: {
      task: false,
      subagent: false,
    },
    setChatToolPreferences: mockSetChatToolPreferences,
    serviceSkills: [],
    activeTheme: "general",
    mappedTheme: "general",
    isThemeWorkbench: false,
    contextWorkspace: {
      enabled: false,
      activeContextPrompt: "",
      prepareActiveContextPrompt: async () => "",
    },
    projectId: "project-1",
    projectRootPath: "/workspace",
    sessionId: "session-1",
    executionStrategy: "react",
    accessMode: "current",
    currentGateKey: "default_gate",
    themeWorkbenchActiveQueueTitle: undefined,
    contentId: null,
    workspaceRequestMetadataBase: undefined,
    messages: [],
    setChatMessages: mockSetChatMessages,
    bootstrapDispatchPreview: null,
    sendMessage: mockSendMessage,
    resolveSendBoundary: (({ sourceText }) => ({
      sourceText,
      browserRequirementMatch: null,
      shouldConsumePendingGeneralWorkbenchInitialPrompt: false,
      shouldDismissGeneralWorkbenchEntryPrompt: false,
    })) as HookProps["resolveSendBoundary"],
    finalizeAfterSendSuccess:
      mockFinalizeAfterSendSuccess as HookProps["finalizeAfterSendSuccess"],
    rollbackAfterSendFailure:
      mockRollbackAfterSendFailure as HookProps["rollbackAfterSendFailure"],
    openRuntimeSceneGate:
      mockOpenRuntimeSceneGate as HookProps["openRuntimeSceneGate"],
    ensureSessionForCommandMetadata:
      mockEnsureSessionForCommandMetadata as HookProps["ensureSessionForCommandMetadata"],
    prepareImageWorkbenchSkillSend:
      mockPrepareImageWorkbenchSkillSend as HookProps["prepareImageWorkbenchSkillSend"],
    resolveImageWorkbenchCommandRequest:
      mockResolveImageWorkbenchCommandRequest as HookProps["resolveImageWorkbenchCommandRequest"],
    ...initialProps,
  };

  function TestComponent() {
    hookValue = useWorkspaceSendActions(props);
    return null;
  }

  act(() => {
    root.render(<TestComponent />);
  });

  return {
    getValue: () => {
      if (!hookValue) {
        throw new Error("hook 尚未初始化");
      }
      return hookValue;
    },
    unmount: () => {
      act(() => {
        root.unmount();
      });
      container.remove();
    },
  };
}

describe("useWorkspaceSendActions", () => {
  beforeEach(async () => {
    await changeLimeLocale("zh-CN");
    (
      globalThis as typeof globalThis & {
        IS_REACT_ACT_ENVIRONMENT?: boolean;
      }
    ).IS_REACT_ACT_ENVIRONMENT = true;

    vi.clearAllMocks();
    window.localStorage.clear();
    mockResolveImageWorkbenchCommandRequest.mockReturnValue(null);
    mockPrepareImageWorkbenchSkillSend.mockReturnValue(true);
    mockEnsureSessionForCommandMetadata.mockResolvedValue(null);
    mockGetSkillCatalog.mockResolvedValue({ entries: [] });
    mockListSkillCatalogSceneEntries.mockReturnValue([]);
    mockGetOrCreateDefaultProject.mockResolvedValue({
      id: "project-default",
      name: "默认项目",
      workspaceType: "general",
      rootPath: "/workspace/default",
      isDefault: true,
      createdAt: 0,
      updatedAt: 0,
      isFavorite: false,
      isArchived: false,
      tags: [],
    });
    mockResolveOemCloudRuntimeContext.mockReturnValue(null);
    mockGetOemCloudBootstrapSnapshot.mockReturnValue(null);
    mockUseGlobalMediaGenerationDefaults.mockReturnValue({});
    mockInstallSkillFromPromptInstruction.mockResolvedValue({
      directory: "viral-content-breakdown",
      inspection: {
        content: "",
        metadata: {},
        allowedTools: [],
        resourceSummary: {
          hasScripts: false,
          hasReferences: false,
          hasAssets: false,
        },
        standardCompliance: {
          isStandard: true,
          validationErrors: [],
          deprecatedFields: [],
        },
      },
    });
    mockOpenRuntimeSceneGate.mockResolvedValue(undefined);
  });

  it("普通发送不应把当前工作区模型当成 modelOverride", async () => {
    const harness = mountHook();
    const autoContinue = {
      enabled: true,
      fast_mode_enabled: false,
      continuation_length: 3,
      sensitivity: 0.4,
    };

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend(
            [],
            false,
            false,
            "继续处理当前话题",
            "react",
            autoContinue,
          );
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      const args = mockSendMessage.mock.calls[0] as Parameters<
        HookProps["sendMessage"]
      >;
      expect(args?.[6]).toBeUndefined();
      expect(args?.[7]).toEqual(autoContinue);
      expect(args?.[8]).toMatchObject({
        requestMetadata: {
          harness: expect.objectContaining({
            theme: "general",
            session_mode: "default",
          }),
        },
      });
    } finally {
      harness.unmount();
    }
  });

  it("普通发送不应读取媒体默认配置", async () => {
    const harness = mountHook({
      input: "整理一下今天的国际新闻",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockUseGlobalMediaGenerationDefaults).not.toHaveBeenCalled();
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
    } finally {
      harness.unmount();
    }
  });

  it("没有项目时普通对话仍应允许发送", async () => {
    const harness = mountHook({
      input: "你好",
      projectId: null,
      projectRootPath: null,
      workspaceRequestMetadataBase: undefined,
    });

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend([], false, false, "你好", "react");
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
    } finally {
      harness.unmount();
    }
  });

  it("@ mention 发送不应注入 Renderer 私有 Plugin activation metadata", async () => {
    const text = "@内容工厂 写一篇公众号文章";
    const harness = mountHook({ input: text });

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend([], false, false, text, "react");
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(text);
      const sendOptions = mockSendMessage.mock.calls[0]?.[8];
      expect(sendOptions).not.toHaveProperty(
        "requestMetadata.harness.plugin_activation",
      );
      expect(sendOptions).not.toHaveProperty(
        "requestMetadata.harness.plugin_activation_intent",
      );
    } finally {
      harness.unmount();
    }
  });

  it("首轮初始预览绘制不应阻塞消息提交", async () => {
    const originalRequestAnimationFrameDescriptor =
      Object.getOwnPropertyDescriptor(window, "requestAnimationFrame");
    const requestAnimationFrameSpy = vi.fn(
      (_callback: Parameters<typeof window.requestAnimationFrame>[0]) => 1,
    );
    Object.defineProperty(window, "requestAnimationFrame", {
      configurable: true,
      writable: true,
      value: requestAnimationFrameSpy,
    });
    const harness = mountHook();

    try {
      let result: boolean | "timeout" = "timeout";
      await act(async () => {
        result = await Promise.race([
          harness
            .getValue()
            .handleSend([], false, false, "只回答一个字：好", "react"),
          new Promise<"timeout">((resolve) =>
            setTimeout(() => resolve("timeout"), 50),
          ),
        ]);
      });

      expect(result).toBe(true);
      expect(requestAnimationFrameSpy).toHaveBeenCalledTimes(1);
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
    } finally {
      if (originalRequestAnimationFrameDescriptor) {
        Object.defineProperty(
          window,
          "requestAnimationFrame",
          originalRequestAnimationFrameDescriptor,
        );
      } else {
        Reflect.deleteProperty(window, "requestAnimationFrame");
      }
      harness.unmount();
    }
  });

  it("首轮普通对话不应在 renderer 注入模型路由指令", async () => {
    const harness = mountHook();

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend([], false, false, "请只回复一个字：好", "react");
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      const sendOptions = mockSendMessage.mock.calls[0]?.[8];
      expect(sendOptions?.assistantDraft).toBeUndefined();
      expect(sendOptions?.systemPromptOverride).toBeUndefined();
      expect(sendOptions?.providerOverride).toBeUndefined();
      expect(sendOptions?.modelOverride).toBeUndefined();
      const requestHarness = ((sendOptions?.requestMetadata?.harness as Record<
        string,
        unknown
      >) || {}) as Record<string, unknown>;
      expect(requestHarness.fast_response_routing).toBeUndefined();
      expect(requestHarness.browser_assist).toBeUndefined();
    } finally {
      harness.unmount();
    }
  });

  it("首轮普通对话不应等待 Provider 列表或生成 renderer 路由元数据", async () => {
    const harness = mountHook();

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend([], false, false, "只回答一个字：好", "react");
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      const requestMetadata = mockSendMessage.mock.calls[0]?.[8]
        ?.requestMetadata as { harness?: Record<string, unknown> } | undefined;
      expect(requestMetadata?.harness?.fast_response_routing).toBeUndefined();
      expect(
        mockSendMessage.mock.calls[0]?.[8]?.systemPromptOverride,
      ).toBeUndefined();
      expect(
        mockSendMessage.mock.calls[0]?.[8]?.providerOverride,
      ).toBeUndefined();
      expect(mockSendMessage.mock.calls[0]?.[8]?.modelOverride).toBeUndefined();
    } finally {
      harness.unmount();
    }
  });

  it("首轮轻量对话应把 responsive_chat 配置投影到后端 fast slot", async () => {
    const harness = mountHook({
      serviceModels: {
        responsive_chat: {
          preferredProviderId: "responsive-provider",
          preferredModelId: "fast-chat",
        },
      },
    });

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend([], false, false, "只回答一个字：好", "react");
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      const sendOptions = mockSendMessage.mock.calls[0]?.[8];
      expect(sendOptions?.providerOverride).toBeUndefined();
      expect(sendOptions?.modelOverride).toBeUndefined();
      expect(sendOptions?.requestMetadata).toMatchObject({
        harness: {
          model_slots: {
            fast: {
              provider: "responsive-provider",
              model: "fast-chat",
              source: "service_models.responsive_chat",
              reason: "service_model_preference",
            },
          },
        },
      });
      expect(
        (sendOptions?.requestMetadata?.harness as Record<string, unknown>)
          ?.fast_response_routing,
      ).toBeUndefined();
    } finally {
      harness.unmount();
    }
  });

  it("普通首轮发送不应等待非必要的延迟 service model 配置", async () => {
    const deferredServiceModels = createDeferred<{
      serviceModels?: HookProps["serviceModels"];
      agentResponseLanguage?: string | null;
    }>();
    const resolveServiceModelsBeforeSend = vi.fn(
      async () => deferredServiceModels.promise,
    );
    const harness = mountHook({
      resolveServiceModelsBeforeSend,
    });

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend([], false, false, "只回答一个字：好", "react");
        expect(started).toBe(true);
      });

      expect(resolveServiceModelsBeforeSend).not.toHaveBeenCalled();
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      const sendOptions = mockSendMessage.mock.calls[0]?.[8];
      expect(
        (sendOptions?.requestMetadata?.harness as Record<string, unknown>)
          ?.fast_response_routing,
      ).toBeUndefined();
      expect(JSON.stringify(sendOptions?.requestMetadata)).not.toContain(
        "responsive-provider",
      );
    } finally {
      deferredServiceModels.resolve({});
      harness.unmount();
    }
  });

  it("Task Center 首页 hot path 可跳过工作台命令解析并直发普通对话", async () => {
    const harness = mountHook();

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend(
            [],
            false,
            false,
            "@配图 画一张封面",
            "react",
            undefined,
            {
              skipSessionRestore: true,
              skipSessionStartHooks: true,
              skipWorkspaceCommandRouting: true,
            },
          );
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).not.toHaveBeenCalled();
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe("@配图 画一张封面");
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        skipSessionRestore: true,
        skipSessionStartHooks: true,
        skipWorkspaceCommandRouting: true,
      });
    } finally {
      harness.unmount();
    }
  });

  it("专用服务模型场景应按需补齐延迟加载的 service model 配置", async () => {
    const resolveServiceModelsBeforeSend = vi.fn(async () => ({
      agentResponseLanguage: "en-US",
      serviceModels: {
        prompt_rewrite: {
          preferredProviderId: "rewrite-provider",
          preferredModelId: "rewrite-model",
        },
      },
    }));
    const harness = mountHook({
      resolveServiceModelsBeforeSend,
    });

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend([], false, false, "请改写这段文案", "react", undefined, {
            purpose: "style_rewrite",
          });
        expect(started).toBe(true);
      });

      expect(resolveServiceModelsBeforeSend).toHaveBeenCalledTimes(1);
      const sendOptions = mockSendMessage.mock.calls[0]?.[8];
      expect(sendOptions).toMatchObject({
        purpose: "style_rewrite",
        providerOverride: "rewrite-provider",
        modelOverride: "rewrite-model",
      });
      expect(sendOptions?.requestMetadata).toMatchObject({
        harness: {
          agent_response_language: "en-US",
        },
      });
    } finally {
      harness.unmount();
    }
  });

  it("首轮普通对话不应在前端改写模型或系统提示", async () => {
    const harness = mountHook();

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend([], false, false, "只回答一个字：好", "react");
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      const sendOptions = mockSendMessage.mock.calls[0]?.[8];
      expect(
        (sendOptions?.requestMetadata?.harness as Record<string, unknown>)
          ?.fast_response_routing,
      ).toBeUndefined();
      expect(sendOptions?.systemPromptOverride).toBeUndefined();
      expect(sendOptions?.providerOverride).toBeUndefined();
      expect(sendOptions?.modelOverride).toBeUndefined();
    } finally {
      harness.unmount();
    }
  });

  it("普通视觉 brief 没有 @命令时应直接进入 image_command_intent", async () => {
    const brief =
      "青柠新品发布海报，竖版 4:5，水彩插画风格，清爽绿色调，留出标题区：鲜榨灵感";
    mockResolveImageWorkbenchCommandRequest.mockReturnValueOnce({
      images: [],
      requestContext: {
        kind: "image_task",
        image_task: {
          mode: "generate",
          prompt: brief,
          raw_text: `@配图 ${brief}`,
          count: 1,
        },
      },
    });
    const harness = mountHook({ providerType: "deepseek" });

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend([], false, false, brief, "react");
        expect(started).toBe(true);
      });

      expect(mockSetChatMessages).not.toHaveBeenCalled();
      expect(mockResolveImageWorkbenchCommandRequest).toHaveBeenCalledTimes(1);
      expect(mockResolveImageWorkbenchCommandRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          rawText: `@配图 ${brief}`,
          parsedCommand: expect.objectContaining({
            trigger: "@配图",
            mode: "generate",
            prompt: expect.stringContaining("青柠新品发布海报"),
            aspectRatio: "4:5",
          }),
          images: [],
        }),
      );
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(`@配图 ${brief}`);
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        displayContent: brief,
        requestMetadata: {
          harness: {
            image_command_intent: {
              kind: "image_task",
              image_task: {
                prompt: brief,
              },
            },
          },
        },
      });
      expect(mockSetInput).toHaveBeenCalledWith("");
      expect(mockFinalizeAfterSendSuccess).toHaveBeenCalledTimes(1);
    } finally {
      harness.unmount();
    }
  });

  it("普通视觉 brief 没有项目时也应自动补齐默认项目并进入 image_command_intent 对话回合", async () => {
    const brief =
      "参考图生成一张小红书封面，保留醒目的标题层级、清晰的主体构图和适合收藏转发的视觉冲击力";
    mockResolveImageWorkbenchCommandRequest.mockReturnValueOnce({
      images: [],
      requestContext: {
        kind: "image_task",
        image_task: {
          mode: "generate",
          prompt: brief,
          raw_text: `@配图 ${brief}`,
          count: 1,
        },
      },
    });
    const harness = mountHook({
      messages: [],
      projectId: null,
      projectRootPath: null,
    });

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend([], false, false, brief, "react");
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).toHaveBeenCalledTimes(1);
      expect(mockGetOrCreateDefaultProject).toHaveBeenCalledTimes(1);
      expect(mockResolveImageWorkbenchCommandRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          projectId: "project-default",
          projectRootPath: "/workspace/default",
        }),
      );
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(`@配图 ${brief}`);
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        displayContent: brief,
        requestMetadata: {
          harness: {
            image_command_intent: {
              image_task: {
                prompt: brief,
              },
            },
          },
        },
      });
      expect(harness.getValue().displayMessages).toEqual([]);
      expect(harness.getValue().isPreparingSend).toBe(false);
    } finally {
      harness.unmount();
    }
  });

  it("普通视觉 brief 建会话悬挂时也不应卡在准备回复", async () => {
    const brief =
      "参考图生成一张小红书封面，保留醒目的标题层级、清晰的主体构图和适合收藏转发的视觉冲击力";
    const deferredSession = createDeferred<string | null>();
    const deferredSend = createDeferred<void>();
    mockEnsureSessionForCommandMetadata.mockImplementationOnce(
      async () => deferredSession.promise,
    );
    mockResolveImageWorkbenchCommandRequest.mockReturnValueOnce({
      images: [],
      requestContext: {
        kind: "image_task",
        image_task: {
          mode: "generate",
          prompt: brief,
          raw_text: `@配图 ${brief}`,
          count: 1,
          session_id: "__local_image_workbench__:draft",
        },
      },
    });
    mockSendMessage.mockImplementationOnce(async () => deferredSend.promise);
    const harness = mountHook();

    let sendPromise: Promise<boolean> | null = null;

    try {
      await act(async () => {
        sendPromise = harness
          .getValue()
          .handleSend([], false, false, brief, "react");
        await Promise.resolve();
      });

      expect(mockEnsureSessionForCommandMetadata).toHaveBeenCalledTimes(1);
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(`@配图 ${brief}`);
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        displayContent: brief,
        requestMetadata: {
          harness: {
            image_command_intent: {
              image_task: {
                prompt: brief,
              },
            },
          },
        },
      });
      const imageCommandIntentHarness = ((mockSendMessage.mock.calls[0]?.[8]
        ?.requestMetadata?.harness as Record<string, unknown> | undefined) ||
        {}) as Record<string, unknown>;
      expect(
        (
          imageCommandIntentHarness.image_command_intent as
            | { image_task?: { session_id?: string } }
            | undefined
        )?.image_task?.session_id,
      ).toBeUndefined();

      await act(async () => {
        deferredSend.resolve();
        await sendPromise;
      });

      expect(mockFinalizeAfterSendSuccess).toHaveBeenCalledTimes(1);
      expect(harness.getValue().displayMessages).toEqual([]);
    } finally {
      deferredSession.resolve("session-image-late");
      harness.unmount();
    }
  });

  it("普通视觉 brief 无法创建图片任务时应清理准备态预览", async () => {
    const brief =
      "参考图生成一张小红书封面，保留醒目的标题层级、清晰的主体构图和适合收藏转发的视觉冲击力";
    mockResolveImageWorkbenchCommandRequest.mockReturnValueOnce(null);
    const harness = mountHook({
      messages: [],
      projectId: null,
      projectRootPath: null,
    });

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend([], false, false, brief, "react");
        expect(started).toBe(false);
        await Promise.resolve();
      });

      expect(mockResolveImageWorkbenchCommandRequest).toHaveBeenCalledTimes(1);
      expect(mockSendMessage).not.toHaveBeenCalled();
      expect(harness.getValue().displayMessages).toEqual([]);
      expect(harness.getValue().isPreparingSend).toBe(false);
    } finally {
      harness.unmount();
    }
  });

  it("普通视觉描述要求提示词时不应触发图片生成主链", async () => {
    const promptText =
      "帮我写一段小红书封面提示词，竖版 4:5，水彩插画风格，先不要生成图片";
    const harness = mountHook();

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend([], false, false, promptText, "react");
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).not.toHaveBeenCalled();
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(promptText);
      const imageCommandIntentHarness =
        (mockSendMessage.mock.calls[0]?.[8]?.requestMetadata?.harness as
          | Record<string, unknown>
          | undefined) ?? {};
      expect(imageCommandIntentHarness.image_command_intent).toBeUndefined();
    } finally {
      harness.unmount();
    }
  });

  it("粘贴官网 Agent Skill 安装 Prompt 时应本地安装并停止发送给模型", async () => {
    const harness = mountHook();
    const prompt = `Download and install a skill. Follow these steps EXACTLY. If any step fails, STOP and report the error.
SKILL_NAME="viral-content-breakdown"
DOWNLOAD_URL="https://limeai.run/skill-packages/viral-content-breakdown/latest/viral-content-breakdown.zip"
1. Download the Skill package.
2. Extract it into the Agent Skills directory.
3. Restart or reload the Agent so the Skill becomes available.`;

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend([], false, false, prompt, "react");
        expect(started).toBe(true);
      });

      expect(mockInstallSkillFromPromptInstruction).toHaveBeenCalledTimes(1);
      expect(mockInstallSkillFromPromptInstruction.mock.calls[0]?.[0]).toEqual({
        skillName: "viral-content-breakdown",
        downloadUrl:
          "https://limeai.run/skill-packages/viral-content-breakdown/latest/viral-content-breakdown.zip",
        source: "assignment_prompt",
      });
      expect(mockInstallSkillFromPromptInstruction.mock.calls[0]?.[1]).toBe(
        "lime",
      );
      expect(mockSendMessage).not.toHaveBeenCalled();
      const applyLocalMessages = mockSetChatMessages.mock.calls[0]?.[0] as
        | ((previous: Message[]) => Message[])
        | undefined;
      const nextMessages = applyLocalMessages?.([]);
      expect(nextMessages?.[0]).toMatchObject({
        role: "user",
        content: prompt,
      });
      expect(nextMessages?.[1]).toMatchObject({
        role: "assistant",
        content: getLimeI18n().t(
          "agentChat.skillInstallPrompt.installedConfirmation",
          {
            ns: "agent",
            skill: "viral-content-breakdown",
          },
        ),
      });
      expect(mockFinalizeAfterSendSuccess).toHaveBeenCalledTimes(1);
    } finally {
      harness.unmount();
    }
  });

  it("Skill 安装 Prompt 的本地确认应跟随全局语言", async () => {
    await changeLimeLocale("en-US");
    const harness = mountHook();

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend(
          [],
          false,
          false,
          `Download and install a skill.
SKILL_NAME="viral-content-breakdown"
DOWNLOAD_URL="https://limeai.run/skill-packages/viral-content-breakdown/latest/viral-content-breakdown.zip"
Extract it into the Agent Skills directory.`,
          "react",
        );
        expect(started).toBe(true);
      });

      const applyLocalMessages = mockSetChatMessages.mock.calls[0]?.[0] as
        | ((previous: Message[]) => Message[])
        | undefined;
      const nextMessages = applyLocalMessages?.([]);
      expect(nextMessages?.[1]?.content).toContain(
        "Installed Skill: viral-content-breakdown",
      );
    } finally {
      harness.unmount();
    }
  });

  it("普通发送不应等待 Provider 列表预加载", async () => {
    const harness = mountHook();

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend([], false, false, "只回答一个字：好", "react");
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      const requestMetadata = mockSendMessage.mock.calls[0]?.[8]
        ?.requestMetadata as { harness?: Record<string, unknown> } | undefined;
      expect(requestMetadata?.harness?.fast_response_routing).toBeUndefined();
      expect(
        mockSendMessage.mock.calls[0]?.[8]?.systemPromptOverride,
      ).toBeUndefined();
      expect(
        mockSendMessage.mock.calls[0]?.[8]?.providerOverride,
      ).toBeUndefined();
      expect(mockSendMessage.mock.calls[0]?.[8]?.modelOverride).toBeUndefined();
    } finally {
      harness.unmount();
    }
  });

  it("首页快路径首轮普通发送不应预热旧会话恢复", async () => {
    const harness = mountHook();

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend(
            [],
            false,
            false,
            "只回答一个字：好",
            "react",
            undefined,
            {
              skipSessionRestore: true,
              skipSessionStartHooks: true,
            },
          );
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockEnsureSessionForCommandMetadata).not.toHaveBeenCalled();
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        skipSessionRestore: true,
        skipSessionStartHooks: true,
      });
      const requestMetadata = mockSendMessage.mock.calls[0]?.[8]
        ?.requestMetadata as { harness?: Record<string, unknown> } | undefined;
      expect(requestMetadata?.harness?.fast_response_routing).toBeUndefined();
      expect(
        mockSendMessage.mock.calls[0]?.[8]?.providerOverride,
      ).toBeUndefined();
      expect(mockSendMessage.mock.calls[0]?.[8]?.modelOverride).toBeUndefined();
    } finally {
      harness.unmount();
    }
  });

  it("OEM 云端低额度应透传到 harness oem_routing 供后端生成 quota_low 事件", async () => {
    mockResolveOemCloudRuntimeContext.mockReturnValue({
      baseUrl: "https://lime.example.com",
      controlPlaneBaseUrl: "https://lime.example.com",
      sceneBaseUrl: "https://lime.example.com",
      gatewayBaseUrl: "https://gateway.example.com",
      tenantId: "tenant-1",
      sessionToken: "session-token",
      hubProviderName: "lime-hub",
      loginPath: "/login",
      desktopClientId: "desktop-client",
      desktopOauthRedirectUrl: "lime://oauth/callback",
      desktopOauthNextPath: "/welcome",
    });
    mockGetOemCloudBootstrapSnapshot.mockReturnValue({
      providerPreference: {
        tenantId: "tenant-1",
        userId: "user-1",
        providerSource: "oem_cloud",
        providerKey: "lime-hub",
        defaultModel: "claude-sonnet-4",
        needsValidation: false,
        updatedAt: "2026-04-24T00:00:00.000Z",
      },
      providerOffersSummary: [
        {
          providerKey: "lime-hub",
          source: "oem_cloud",
          state: "available_quota_low",
          defaultModel: "claude-sonnet-4",
          configMode: "managed",
          quotaStatus: "low",
          fallbackToLocalAllowed: false,
          canInvoke: true,
        },
      ],
    });
    const harness = mountHook();

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend([], false, false, "继续处理当前话题", "react");
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      const args = mockSendMessage.mock.calls[0] as Parameters<
        HookProps["sendMessage"]
      >;
      expect(args?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            oem_routing: {
              tenant_id: "tenant-1",
              provider_source: "oem_cloud",
              provider_key: "lime-hub",
              default_model: "claude-sonnet-4",
              config_mode: "managed",
              offer_state: "available_quota_low",
              quota_status: "low",
              fallback_to_local_allowed: false,
              can_invoke: true,
            },
          },
        },
      });
    } finally {
      harness.unmount();
    }
  });

  it("切到非 OEM Provider 时不应继续透传 OEM 低额度路由", async () => {
    mockResolveOemCloudRuntimeContext.mockReturnValue({
      baseUrl: "https://lime.example.com",
      controlPlaneBaseUrl: "https://lime.example.com",
      sceneBaseUrl: "https://lime.example.com",
      gatewayBaseUrl: "https://gateway.example.com",
      tenantId: "tenant-1",
      sessionToken: "session-token",
      hubProviderName: "lime-hub",
      loginPath: "/login",
      desktopClientId: "desktop-client",
      desktopOauthRedirectUrl: "lime://oauth/callback",
      desktopOauthNextPath: "/welcome",
    });
    mockGetOemCloudBootstrapSnapshot.mockReturnValue({
      providerPreference: {
        tenantId: "tenant-1",
        userId: "user-1",
        providerSource: "oem_cloud",
        providerKey: "lime-hub",
        defaultModel: "gpt-5.5",
        needsValidation: false,
        updatedAt: "2026-04-24T00:00:00.000Z",
      },
      providerOffersSummary: [
        {
          providerKey: "lime-hub",
          source: "oem_cloud",
          state: "available_quota_low",
          defaultModel: "gpt-5.5",
          configMode: "managed",
          quotaStatus: "low",
          fallbackToLocalAllowed: false,
          canInvoke: true,
        },
      ],
    });
    const harness = mountHook({ providerType: "deepseek" });

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend([], false, false, "继续处理当前话题", "react");
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      const args = mockSendMessage.mock.calls[0] as Parameters<
        HookProps["sendMessage"]
      >;
      const harnessMetadata = ((args?.[8]?.requestMetadata?.harness as Record<
        string,
        unknown
      >) || {}) as Record<string, unknown>;
      expect(harnessMetadata).not.toHaveProperty("oem_routing");
    } finally {
      harness.unmount();
    }
  });

  it("OEM bootstrap features 应透传为 tenant_feature_flags 策略输入", async () => {
    mockResolveOemCloudRuntimeContext.mockReturnValue({
      baseUrl: "https://lime.example.com",
      controlPlaneBaseUrl: "https://lime.example.com",
      sceneBaseUrl: "https://lime.example.com",
      gatewayBaseUrl: "https://gateway.example.com",
      tenantId: "tenant-1",
      sessionToken: "session-token",
      hubProviderName: "lime-hub",
      loginPath: "/login",
      desktopClientId: "desktop-client",
      desktopOauthRedirectUrl: "lime://oauth/callback",
      desktopOauthNextPath: "/welcome",
    });
    mockGetOemCloudBootstrapSnapshot.mockReturnValue({
      features: {
        oauthLoginEnabled: true,
        emailCodeLoginEnabled: true,
        passwordLoginEnabled: false,
        profileEditable: true,
        hubTokensEnabled: false,
        billingEnabled: true,
        referralEnabled: false,
        gatewayEnabled: true,
      },
    });
    const harness = mountHook();

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend([], false, false, "继续处理当前话题", "react");
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      const args = mockSendMessage.mock.calls[0] as Parameters<
        HookProps["sendMessage"]
      >;
      expect(args?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            tenant_feature_flags: {
              tenant_id: "tenant-1",
              source: "oem_cloud_bootstrap",
              flags: {
                gatewayEnabled: true,
                billingEnabled: true,
                referralEnabled: false,
                passwordLoginEnabled: false,
              },
            },
          },
        },
      });
      expect(JSON.stringify(args?.[8]?.requestMetadata)).not.toContain(
        "session-token",
      );
    } finally {
      harness.unmount();
    }
  });

  it("显式改写类 purpose 应复用 prompt_rewrite 服务模型", async () => {
    const harness = mountHook({
      input: "请改写这段文案",
      serviceModels: {
        prompt_rewrite: {
          preferredProviderId: "rewrite-provider",
          preferredModelId: "rewrite-model",
        },
      },
    });

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend([], false, false, "请改写这段文案", "react", undefined, {
            purpose: "style_rewrite",
          });
        expect(started).toBe(true);
      });

      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        purpose: "style_rewrite",
        providerOverride: "rewrite-provider",
        modelOverride: "rewrite-model",
      });
    } finally {
      harness.unmount();
    }
  });

  it("无真实消息时不再展示 bootstrap 启动预览消息", () => {
    const harness = mountHook({
      bootstrapDispatchPreview: createBootstrapDispatchSnapshot(),
    });

    try {
      expect(harness.getValue().displayMessages).toEqual([]);
    } finally {
      harness.unmount();
    }
  });

  it("发送准备阶段应立即暴露 submission preview，并在真实提交接管后清理", async () => {
    const deferredSend = createDeferred<void>();
    mockSendMessage.mockImplementationOnce(async () => deferredSend.promise);
    const harness = mountHook({
      input: "帮我找一下今天的新闻",
    });

    let sendPromise: Promise<boolean> | null = null;

    try {
      await act(async () => {
        sendPromise = harness.getValue().handleSend();
        await Promise.resolve();
      });

      expect(harness.getValue().displayMessages).toEqual([
        expect.objectContaining({
          role: "user",
          content: "帮我找一下今天的新闻",
        }),
        expect.objectContaining({
          role: "assistant",
          isThinking: true,
          runtimeStatus: expect.objectContaining({
            phase: "preparing",
          }),
        }),
      ]);

      await act(async () => {
        deferredSend.resolve();
        await sendPromise;
      });

      expect(harness.getValue().displayMessages).toEqual([]);
    } finally {
      harness.unmount();
    }
  });

  it("普通 Claw 空上下文准备较慢时不应阻塞发送", async () => {
    const deferredContext = createDeferred<string>();
    const deferredSend = createDeferred<void>();
    const prepareActiveContextPrompt = vi.fn(
      async () => deferredContext.promise,
    );
    mockSendMessage.mockImplementationOnce(async () => deferredSend.promise);
    const harness = mountHook({
      input: "帮我整理一下今天的重要新闻",
      contextWorkspace: {
        enabled: true,
        activeContextPrompt: "",
        prepareActiveContextPrompt,
      },
    });

    let sendPromise: Promise<boolean> | null = null;

    try {
      await act(async () => {
        sendPromise = harness.getValue().handleSend();
        await Promise.resolve();
      });

      expect(harness.getValue().displayMessages).toEqual([
        expect.objectContaining({
          role: "user",
          content: "帮我整理一下今天的重要新闻",
        }),
        expect.objectContaining({
          role: "assistant",
          isThinking: true,
          runtimeStatus: expect.objectContaining({
            phase: "preparing",
          }),
        }),
      ]);
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "帮我整理一下今天的重要新闻",
      );
      expect(prepareActiveContextPrompt).not.toHaveBeenCalled();

      await act(async () => {
        deferredContext.resolve("[上下文]\n今天关注科技与国际新闻。");
        await Promise.resolve();
      });

      await act(async () => {
        deferredSend.resolve();
        await sendPromise;
      });

      expect(harness.getValue().displayMessages).toEqual([]);
    } finally {
      harness.unmount();
    }
  });

  it("主题工作台首轮发送等待上下文时应并行预热会话", async () => {
    const deferredContext = createDeferred<string>();
    const deferredSession = createDeferred<string | null>();
    const deferredSend = createDeferred<void>();
    mockEnsureSessionForCommandMetadata.mockImplementationOnce(
      async () => deferredSession.promise,
    );
    mockSendMessage.mockImplementationOnce(async () => deferredSend.promise);
    const harness = mountHook({
      input: "帮我整理一下今天的重要新闻",
      isThemeWorkbench: true,
      contextWorkspace: {
        enabled: true,
        activeContextPrompt: "",
        prepareActiveContextPrompt: async () => deferredContext.promise,
      },
    });

    let sendPromise: Promise<boolean> | null = null;

    try {
      await act(async () => {
        sendPromise = harness.getValue().handleSend();
        await Promise.resolve();
      });

      expect(mockEnsureSessionForCommandMetadata).toHaveBeenCalledTimes(1);
      expect(mockSendMessage).not.toHaveBeenCalled();

      await act(async () => {
        deferredContext.resolve("[上下文]\n今天关注科技与国际新闻。");
        await Promise.resolve();
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(
        mockEnsureSessionForCommandMetadata.mock.invocationCallOrder[0],
      ).toBeLessThan(mockSendMessage.mock.invocationCallOrder[0] ?? Infinity);

      await act(async () => {
        deferredSession.resolve("session-prewarmed");
        deferredSend.resolve();
        await sendPromise;
      });
    } finally {
      harness.unmount();
    }
  });

  it("发送准备阶段不应重复递交同一条首页消息", async () => {
    const deferredSend = createDeferred<void>();
    mockSendMessage.mockImplementationOnce(async () => deferredSend.promise);
    const harness = mountHook({
      input: "帮我生成一版首页插图方案",
    });

    let firstSendPromise: Promise<boolean> | null = null;
    let secondSendPromise: Promise<boolean> | null = null;

    try {
      await act(async () => {
        firstSendPromise = harness.getValue().handleSend();
        secondSendPromise = harness.getValue().handleSend();
        await Promise.resolve();
      });

      expect(await secondSendPromise).toBe(false);
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(harness.getValue().displayMessages).toEqual([
        expect.objectContaining({
          role: "user",
          content: "帮我生成一版首页插图方案",
        }),
        expect.objectContaining({
          role: "assistant",
          isThinking: true,
          runtimeStatus: expect.objectContaining({
            phase: "preparing",
          }),
        }),
      ]);

      await act(async () => {
        deferredSend.resolve();
        expect(await firstSendPromise).toBe(true);
      });

      expect(harness.getValue().isPreparingSend).toBe(false);
    } finally {
      harness.unmount();
    }
  });

  it("普通自然句不再隐式命中已删除的站点技能自动启动链", async () => {
    const harness = mountHook({
      input: "请帮我使用 GitHub 查一下 AI Agent 项目",
      serviceSkills: [createGithubSiteSkill()],
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "请帮我使用 GitHub 查一下 AI Agent 项目",
      );
    } finally {
      harness.unmount();
    }
  });

  it("纯文本 @配图 应保留原始消息，并通过 image_command_intent metadata 交给 Agent", async () => {
    mockResolveImageWorkbenchCommandRequest.mockReturnValueOnce({
      images: [],
      requestContext: {
        kind: "image_task",
        image_task: {
          mode: "generate",
          prompt: "一张春日咖啡馆插画",
          count: 2,
          aspect_ratio: "16:9",
        },
      },
    });
    const harness = mountHook({
      input: "@配图 生成 一张春日咖啡馆插画，16:9，出 2 张",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).toHaveBeenCalledTimes(1);
      expect(mockResolveImageWorkbenchCommandRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          rawText: "@配图 生成 一张春日咖啡馆插画，16:9，出 2 张",
          parsedCommand: expect.objectContaining({
            trigger: "@配图",
            mode: "generate",
            prompt: "一张春日咖啡馆插画",
            aspectRatio: "16:9",
            count: 2,
          }),
          images: [],
          sessionIdOverride: undefined,
        }),
      );
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@配图 生成 一张春日咖啡馆插画，16:9，出 2 张",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            image_command_intent: {
              kind: "image_task",
            },
          },
        },
        assistantDraft: {
          content: "",
          fallbackContent: "",
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "image_generate",
          replayText: "16:9 一张春日咖啡馆插画 出 2 张",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("普通画图句式应复用 @配图 主链并保留用户原始展示文案", async () => {
    mockResolveImageWorkbenchCommandRequest.mockReturnValueOnce({
      images: [],
      requestContext: {
        kind: "image_task",
        image_task: {
          mode: "generate",
          prompt: "画一张广州夏天的图",
          count: 1,
        },
      },
    });
    const harness = mountHook({
      input: "画一张广州夏天的图",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).toHaveBeenCalledTimes(1);
      expect(mockResolveImageWorkbenchCommandRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          rawText: "@配图 画一张广州夏天的图",
          parsedCommand: expect.objectContaining({
            trigger: "@配图",
            mode: "generate",
            prompt: "画一张广州夏天的图",
            count: 1,
          }),
          images: [],
          sessionIdOverride: undefined,
        }),
      );
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@配图 画一张广州夏天的图",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        displayContent: "画一张广州夏天的图",
        requestMetadata: {
          harness: {
            image_command_intent: {
              kind: "image_task",
            },
          },
        },
      });
    } finally {
      harness.unmount();
    }
  });

  it("纯文本 @配图 在图片服务未就绪时不应创建空 provider 任务", async () => {
    mockPrepareImageWorkbenchSkillSend.mockReturnValueOnce(false);
    const harness = mountHook({
      input: "@配图 生成 一张春日咖啡馆插画",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(false);
      });

      expect(mockPrepareImageWorkbenchSkillSend).toHaveBeenCalledTimes(1);
      expect(mockResolveImageWorkbenchCommandRequest).not.toHaveBeenCalled();
      expect(mockSendMessage).not.toHaveBeenCalled();
    } finally {
      harness.unmount();
    }
  });

  it("纯文本 @配图 应等待图片服务刷新完成后再构造 skillRequest", async () => {
    const deferredPrepare = createDeferred<boolean>();
    mockPrepareImageWorkbenchSkillSend.mockImplementationOnce(
      async () => deferredPrepare.promise,
    );
    mockResolveImageWorkbenchCommandRequest.mockReturnValueOnce({
      images: [],
      requestContext: {
        kind: "image_task",
        image_task: {
          mode: "generate",
          prompt: "一张春日咖啡馆插画",
          provider_id: "fal",
          model: "fal-ai/nano-banana-pro",
        },
      },
    });
    const harness = mountHook({
      input: "@配图 生成 一张春日咖啡馆插画",
    });
    let sendPromise: Promise<boolean> | null = null;

    try {
      await act(async () => {
        sendPromise = harness.getValue().handleSend();
        await Promise.resolve();
      });

      expect(mockPrepareImageWorkbenchSkillSend).toHaveBeenCalledTimes(1);
      expect(mockResolveImageWorkbenchCommandRequest).not.toHaveBeenCalled();
      expect(mockSendMessage).not.toHaveBeenCalled();

      await act(async () => {
        deferredPrepare.resolve(true);
        await sendPromise;
      });

      expect(await sendPromise).toBe(true);
      expect(mockResolveImageWorkbenchCommandRequest).toHaveBeenCalledTimes(1);
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            image_command_intent: {
              image_task: {
                provider_id: "fal",
                model: "fal-ai/nano-banana-pro",
              },
            },
          },
        },
      });
    } finally {
      harness.unmount();
    }
  });

  it("带 builtin route 的 @配图 发送不应重复包装或重复递交", async () => {
    mockResolveImageWorkbenchCommandRequest.mockReturnValueOnce({
      images: [],
      requestContext: {
        kind: "image_task",
        image_task: {
          mode: "generate",
          prompt: "一张春日咖啡馆插画",
          count: 1,
        },
      },
    });
    const harness = mountHook({
      input: "@配图 生成 一张春日咖啡馆插画",
    });

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend(
            [],
            false,
            false,
            "@配图 生成 一张春日咖啡馆插画",
            "react",
            undefined,
            {
              capabilityRoute: {
                kind: "builtin_command",
                commandKey: "image_generate",
                commandPrefix: "@配图",
              },
              displayContent: "@配图 生成 一张春日咖啡馆插画",
            },
          );
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).toHaveBeenCalledTimes(1);
      expect(mockResolveImageWorkbenchCommandRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          rawText: "@配图 生成 一张春日咖啡馆插画",
          parsedCommand: expect.objectContaining({
            trigger: "@配图",
            prompt: "一张春日咖啡馆插画",
          }),
        }),
      );
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@配图 生成 一张春日咖啡馆插画",
      );
      const sendOptions = mockSendMessage.mock.calls[0]?.[8];
      expect(sendOptions?.displayContent).toBe("@配图 生成 一张春日咖啡馆插画");
      expect(sendOptions?.capabilityRoute).toMatchObject({
        kind: "builtin_command",
        commandKey: "image_generate",
        commandPrefix: "@配图",
      });
      expect(
        JSON.stringify(sendOptions?.requestMetadata).match(
          /image_command_intent/g,
        ) ?? [],
      ).toHaveLength(1);
      expect(sendOptions?.requestMetadata).toMatchObject({
        harness: {
          image_command_intent: {
            image_task: {
              prompt: "一张春日咖啡馆插画",
            },
          },
        },
      });
    } finally {
      harness.unmount();
    }
  });

  it("通过 @ 面板选择配图后输入普通描述时应补齐 @配图 并进入图片任务", async () => {
    mockResolveImageWorkbenchCommandRequest.mockReturnValueOnce({
      images: [],
      requestContext: {
        kind: "image_task",
        image_task: {
          mode: "generate",
          prompt: "画一张广州夏天的图",
          count: 1,
        },
      },
    });
    const harness = mountHook({
      input: "画一张广州夏天的图",
    });

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend(
            [],
            false,
            false,
            "画一张广州夏天的图",
            "react",
            undefined,
            {
              capabilityRoute: {
                kind: "builtin_command",
                commandKey: "image_generate",
                commandPrefix: "@配图",
              },
              displayContent: "画一张广州夏天的图",
            },
          );
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).toHaveBeenCalledTimes(1);
      expect(mockResolveImageWorkbenchCommandRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          rawText: "@配图 画一张广州夏天的图",
          parsedCommand: expect.objectContaining({
            trigger: "@配图",
            prompt: "画一张广州夏天的图",
          }),
        }),
      );
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@配图 画一张广州夏天的图",
      );
      const sendOptions = mockSendMessage.mock.calls[0]?.[8];
      expect(sendOptions?.displayContent).toBe("画一张广州夏天的图");
      expect(sendOptions?.requestMetadata).toMatchObject({
        harness: {
          image_command_intent: {
            image_task: {
              prompt: "画一张广州夏天的图",
            },
          },
        },
      });
    } finally {
      harness.unmount();
    }
  });

  it("@Nanobanana Pro 首次发送不应伪造本地图片任务轻卡", async () => {
    mockResolveImageWorkbenchCommandRequest.mockReturnValueOnce({
      images: [],
      requestContext: {
        kind: "image_task",
        image_task: {
          mode: "generate",
          prompt: "一张广州塔，从花城汇看过去的春天的照片",
          raw_text:
            "@Nanobanana Pro 生成一张广州塔，从花城汇看过去的春天的照片",
          count: 1,
          provider_id: "fal",
          model: "fal-ai/nano-banana-pro",
        },
      },
    });
    const harness = mountHook({
      input: "@Nanobanana Pro 生成一张广州塔，从花城汇看过去的春天的照片",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      const sendOptions = mockSendMessage.mock.calls[0]?.[8];
      expect(sendOptions).toMatchObject({
        assistantDraft: {
          content: "",
          fallbackContent: "",
        },
        requestMetadata: {
          harness: {
            image_command_intent: {
              image_task: {
                model: "fal-ai/nano-banana-pro",
                provider_id: "fal",
              },
            },
          },
        },
      });
      expect(
        sendOptions?.assistantDraft?.imageWorkbenchPreview,
      ).toBeUndefined();
      expect(sendOptions?.assistantDraft?.content).toBe("");
      expect(sendOptions?.assistantDraft?.fallbackContent).toBe("");
      expect(sendOptions?.assistantDraft?.preserveContent).toBeUndefined();
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "image_generate_nanobanana_pro",
          replayText: "一张广州塔，从花城汇看过去的春天的照片",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("写文章 dispatch 不应注入空 assistantDraft 抢占普通 Agent 输出", async () => {
    const harness = mountHook({
      input: "@写文章 写一篇公众号文章",
    });
    mockEnsureSessionForCommandMetadata.mockResolvedValueOnce(
      "session-write-article-fallback",
    );

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      const sendOptions = mockSendMessage.mock.calls[0]?.[8];
      expect(sendOptions?.assistantDraft).toBeUndefined();
    } finally {
      harness.unmount();
    }
  });

  it("@Nano Banana 2 应进入 image_command_intent 同一回合并保留过程", async () => {
    upsertLocalModelBoundImageCommandBinding({
      trigger: "@Nano Banana 2",
      providerId: "fal",
      modelId: "fal-ai/nano-banana-2",
      executorMode: "images_api",
    });
    mockEnsureSessionForCommandMetadata.mockResolvedValueOnce(
      "session-direct-1",
    );
    mockResolveImageWorkbenchCommandRequest.mockReturnValueOnce({
      images: [],
      requestContext: {
        kind: "image_task",
        image_task: {
          mode: "generate",
          prompt: "一张广州塔，从花城汇看过去的春天的照片",
          raw_text: "@Nano Banana 2 生成一张广州塔，从花城汇看过去的春天的照片",
          count: 1,
          size: "1024x1024",
          usage: "claw-image-workbench",
          provider_id: "fal",
          model: "fal-ai/nano-banana-2",
          executor_mode: "images_api",
          project_id: "project-1",
          entry_source: "at_nano_banana_2_model_command",
          modality_contract_key: "image_generation",
          modality: "image",
          required_capabilities: ["image_generation"],
          routing_slot: "image_generation_model",
          runtime_contract: {
            contract_key: "image_generation",
            routing_slot: "image_generation_model",
          },
        },
      },
    });
    const harness = mountHook({
      input: "@Nano Banana 2 生成一张广州塔，从花城汇看过去的春天的照片",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockCreateImageGenerationTask).not.toHaveBeenCalled();
      expect(mockResolveImageWorkbenchCommandRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          rawText: "@Nano Banana 2 生成一张广州塔，从花城汇看过去的春天的照片",
          parsedCommand: expect.objectContaining({
            commandKey: "image_model_nano_banana_2",
            providerId: "fal",
            modelId: "fal-ai/nano-banana-2",
            executorMode: "images_api",
          }),
          entrySource: "at_nano_banana_2_model_command",
          sessionIdOverride: undefined,
        }),
      );
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@Nano Banana 2 生成一张广州塔，从花城汇看过去的春天的照片",
      );
      const sendOptions = mockSendMessage.mock.calls[0]?.[8];
      expect(sendOptions).toMatchObject({
        assistantDraft: {
          content: "",
          fallbackContent: "",
        },
        requestMetadata: {
          harness: {
            image_command_intent: {
              kind: "image_task",
              image_task: {
                provider_id: "fal",
                model: "fal-ai/nano-banana-2",
                executor_mode: "images_api",
                entry_source: "at_nano_banana_2_model_command",
                runtime_contract: {
                  contract_key: "image_generation",
                  routing_slot: "image_generation_model",
                },
              },
            },
          },
        },
      });
      expect(
        sendOptions?.assistantDraft?.imageWorkbenchPreview,
      ).toBeUndefined();
      expect(sendOptions?.assistantDraft?.content).toBe("");
      expect(sendOptions?.assistantDraft?.fallbackContent).toBe("");
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "image_model_nano_banana_2",
          replayText: "一张广州塔，从花城汇看过去的春天的照片",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@Vision 1 应继续走图片生成主链，不新增视觉专家协议", async () => {
    mockResolveImageWorkbenchCommandRequest.mockReturnValueOnce({
      images: [],
      requestContext: {
        kind: "image_task",
        image_task: {
          mode: "generate",
          prompt: "春日咖啡品牌主视觉海报",
          count: 1,
          size: "864x1152",
          aspect_ratio: "4:5",
          entry_source: "at_image_command",
        },
      },
    });
    const harness = mountHook({
      input: "@Vision 1 春日咖啡品牌主视觉海报，4:5",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@Vision 1 春日咖啡品牌主视觉海报，4:5",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            image_command_intent: {
              image_task: {
                entry_source: "at_image_command",
                aspect_ratio: "4:5",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "image_generate",
          replayText: "4:5 春日咖啡品牌主视觉海报",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@海报 应保留原始消息，并复用 image_command_intent 主链生成海报任务", async () => {
    mockResolveImageWorkbenchCommandRequest.mockReturnValueOnce({
      images: [],
      requestContext: {
        kind: "image_task",
        image_task: {
          mode: "generate",
          prompt: "适用于小红书，清新拼贴风格，海报设计，春日咖啡市集活动海报",
          count: 1,
          size: "864x1152",
          aspect_ratio: "4:5",
          entry_source: "at_poster_command",
        },
      },
    });
    const harness = mountHook({
      input: "@海报 小红书 风格: 清新拼贴 春日咖啡市集活动海报",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).toHaveBeenCalledTimes(1);
      expect(mockResolveImageWorkbenchCommandRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          rawText: "@海报 小红书 风格: 清新拼贴 春日咖啡市集活动海报",
          parsedCommand: expect.objectContaining({
            trigger: "@配图",
            mode: "generate",
            prompt:
              "适用于小红书，清新拼贴风格，海报设计，春日咖啡市集活动海报",
            aspectRatio: "4:5",
            size: "864x1152",
            count: 1,
          }),
          entrySource: "at_poster_command",
          images: [],
          sessionIdOverride: undefined,
        }),
      );
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@海报 小红书 风格: 清新拼贴 春日咖啡市集活动海报",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            image_command_intent: {
              kind: "image_task",
              image_task: {
                entry_source: "at_poster_command",
                size: "864x1152",
                aspect_ratio: "4:5",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "poster_generate",
          replayText: "平台:小红书 风格:清新拼贴 春日咖啡市集活动海报",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@Flyer 3 应继续走海报主链，不新增第二套命令协议", async () => {
    mockResolveImageWorkbenchCommandRequest.mockReturnValueOnce({
      images: [],
      requestContext: {
        kind: "image_task",
        image_task: {
          mode: "generate",
          prompt: "适用于小红书，清新拼贴风格，海报设计，春日咖啡市集活动海报",
          count: 1,
          size: "864x1152",
          aspect_ratio: "4:5",
          entry_source: "at_poster_command",
        },
      },
    });
    const harness = mountHook({
      input: "@Flyer 3 小红书 风格: 清新拼贴 春日咖啡市集活动海报",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@Flyer 3 小红书 风格: 清新拼贴 春日咖啡市集活动海报",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            image_command_intent: {
              image_task: {
                entry_source: "at_poster_command",
                aspect_ratio: "4:5",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "poster_generate",
          replayText: "平台:小红书 风格:清新拼贴 春日咖啡市集活动海报",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@配图 首次发送时应先绑定真实 session_id 到 image_task", async () => {
    mockEnsureSessionForCommandMetadata.mockResolvedValue("session-image-1");
    mockResolveImageWorkbenchCommandRequest.mockReturnValueOnce({
      images: [],
      requestContext: {
        kind: "image_task",
        image_task: {
          mode: "generate",
          prompt: "一张春日咖啡馆插画",
          session_id: "session-image-1",
        },
      },
    });
    const harness = mountHook({
      input: "@配图 生成 一张春日咖啡馆插画",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockEnsureSessionForCommandMetadata).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            image_command_intent: {
              image_task: {
                session_id: "session-image-1",
              },
            },
          },
        },
      });
    } finally {
      harness.unmount();
    }
  });

  it("@配图 首次发送建会话较慢时，也应先展示 submission preview", async () => {
    const deferredSession = createDeferred<string | null>();
    const deferredSend = createDeferred<void>();
    mockEnsureSessionForCommandMetadata.mockImplementationOnce(
      async () => deferredSession.promise,
    );
    mockResolveImageWorkbenchCommandRequest.mockReturnValueOnce({
      images: [],
      requestContext: {
        kind: "image_task",
        image_task: {
          mode: "generate",
          prompt: "一张春日咖啡馆插画",
          session_id: "__local_image_workbench__:draft",
        },
      },
    });
    mockSendMessage.mockImplementationOnce(async () => deferredSend.promise);
    const harness = mountHook({
      input: "@配图 生成 一张春日咖啡馆插画",
    });

    let sendPromise: Promise<boolean> | null = null;

    try {
      await act(async () => {
        sendPromise = harness.getValue().handleSend();
        await Promise.resolve();
      });

      expect(harness.getValue().displayMessages).toEqual([
        expect.objectContaining({
          role: "user",
          content: "@配图 生成 一张春日咖啡馆插画",
        }),
        expect.objectContaining({
          role: "assistant",
          isThinking: true,
          runtimeStatus: expect.objectContaining({
            phase: "preparing",
          }),
        }),
      ]);
      expect(harness.getValue().displayMessages[0]).toMatchObject({
        role: "user",
        content: "@配图 生成 一张春日咖啡馆插画",
      });
      expect(mockSendMessage).not.toHaveBeenCalled();

      await act(async () => {
        deferredSession.resolve("session-image-1");
        await Promise.resolve();
      });

      expect(mockEnsureSessionForCommandMetadata).toHaveBeenCalledTimes(1);
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            image_command_intent: {
              image_task: {
                session_id: "session-image-1",
              },
            },
          },
        },
      });

      await act(async () => {
        deferredSend.resolve();
        await sendPromise;
      });

      expect(harness.getValue().displayMessages).toEqual([]);
    } finally {
      harness.unmount();
    }
  });

  it("@封面 应保留原始消息，并通过 cover_skill_launch metadata 交给 Agent 调度技能", async () => {
    const harness = mountHook({
      input:
        "@封面 小红书 标题: 春日咖啡快闪 风格: 清新插画, 1:1 春日咖啡市集封面",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).not.toHaveBeenCalled();
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@封面 小红书 标题: 春日咖啡快闪 风格: 清新插画, 1:1 春日咖啡市集封面",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            cover_skill_launch: {
              skill_name: "cover_generate",
              kind: "cover_task",
              cover_task: {
                prompt: "春日咖啡市集封面",
                title: "春日咖啡快闪",
                platform: "小红书",
                size: "1:1",
                style: "清新插画",
                entry_source: "at_cover_command",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "cover_generate",
          replayText:
            "平台:小红书 标题:春日咖啡快闪 风格:清新插画 1:1 春日咖啡市集封面",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@封面 首次发送时应先绑定真实 session_id 到 cover_task", async () => {
    mockEnsureSessionForCommandMetadata.mockResolvedValue("session-cover-1");
    const harness = mountHook({
      input:
        "@封面 小红书 标题: 春日咖啡快闪 风格: 清新插画, 1:1 春日咖啡市集封面",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockEnsureSessionForCommandMetadata).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            cover_skill_launch: {
              cover_task: {
                session_id: "session-cover-1",
              },
            },
          },
        },
      });
    } finally {
      harness.unmount();
    }
  });

  it("带引用图的图片命令也应先交给 Agent，而不是直建任务", async () => {
    mockResolveImageWorkbenchCommandRequest.mockReturnValueOnce({
      images: [],
      requestContext: {
        kind: "image_task",
        image_task: {
          mode: "edit",
          prompt: "去掉角标，保留主体",
          target_output_ref_id: "img-2",
        },
      },
    });
    const harness = mountHook({
      input: "@配图 编辑 #img-2 去掉角标，保留主体",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).toHaveBeenCalledTimes(1);
      expect(mockResolveImageWorkbenchCommandRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          rawText: "@配图 编辑 #img-2 去掉角标，保留主体",
          parsedCommand: expect.objectContaining({
            trigger: "@配图",
            mode: "edit",
            targetRef: "img-2",
          }),
          images: [],
        }),
      );
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
    } finally {
      harness.unmount();
    }
  });

  it("@修图 应保留原始消息，并通过 image_command_intent metadata 交给 Agent", async () => {
    mockResolveImageWorkbenchCommandRequest.mockReturnValueOnce({
      images: [],
      requestContext: {
        kind: "image_task",
        image_task: {
          mode: "edit",
          prompt: "去掉角标，保留主体",
          target_output_ref_id: "img-2",
        },
      },
    });
    const harness = mountHook({
      input: "@修图 #img-2 去掉角标，保留主体",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).toHaveBeenCalledTimes(1);
      expect(mockResolveImageWorkbenchCommandRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          rawText: "@修图 #img-2 去掉角标，保留主体",
          parsedCommand: expect.objectContaining({
            trigger: "@修图",
            mode: "edit",
            targetRef: "img-2",
          }),
          images: [],
        }),
      );
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
    } finally {
      harness.unmount();
    }
  });

  it("@重绘 应保留原始消息，并通过 image_command_intent metadata 交给 Agent", async () => {
    mockResolveImageWorkbenchCommandRequest.mockReturnValueOnce({
      images: [],
      requestContext: {
        kind: "image_task",
        image_task: {
          mode: "variation",
          prompt: "更偏插画风，保留主视觉",
          target_output_ref_id: "img-2",
        },
      },
    });
    const harness = mountHook({
      input: "@重绘 #img-2 更偏插画风，保留主视觉",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).toHaveBeenCalledTimes(1);
      expect(mockResolveImageWorkbenchCommandRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          rawText: "@重绘 #img-2 更偏插画风，保留主视觉",
          parsedCommand: expect.objectContaining({
            trigger: "@重绘",
            mode: "variation",
            targetRef: "img-2",
          }),
          images: [],
        }),
      );
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
    } finally {
      harness.unmount();
    }
  });

  it("/image 仍应保留原始消息，并通过 image_command_intent metadata 交给 Agent", async () => {
    mockResolveImageWorkbenchCommandRequest.mockReturnValueOnce({
      images: [],
      requestContext: {
        kind: "image_task",
        image_task: {
          mode: "generate",
          prompt: "春日咖啡馆插画",
        },
      },
    });
    const harness = mountHook({
      input: "/image 春日咖啡馆插画，16:9，出 2 张",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).toHaveBeenCalledTimes(1);
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "/image 春日咖啡馆插画，16:9，出 2 张",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            image_command_intent: {
              kind: "image_task",
            },
          },
        },
      });
    } finally {
      harness.unmount();
    }
  });

  it("预绑定 image_command_intent metadata 时不应重新解析图片命令，并且仍会绑定真实 session_id", async () => {
    mockEnsureSessionForCommandMetadata.mockResolvedValue(
      "session-image-bound",
    );
    const harness = mountHook({
      input: "@配图 生成 一张春日咖啡馆插画",
    });

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend(
            [],
            false,
            false,
            "@配图 生成 一张春日咖啡馆插画",
            undefined,
            undefined,
            {
              requestMetadata: {
                harness: {
                  image_command_intent: {
                    kind: "image_task",
                    image_task: {
                      prompt: "一张春日咖啡馆插画",
                      session_id: "__local_image_workbench__:draft",
                    },
                  },
                },
              },
            },
          );
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).not.toHaveBeenCalled();
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockEnsureSessionForCommandMetadata).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            image_command_intent: {
              image_task: {
                session_id: "session-image-bound",
              },
            },
          },
        },
      });
    } finally {
      harness.unmount();
    }
  });

  it("@视频 应保留原始消息，并通过 video_skill_launch metadata 交给 Agent 调度技能", async () => {
    const harness = mountHook({
      input: "@视频 15秒 新品发布短视频，16:9，720p",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).not.toHaveBeenCalled();
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@视频 15秒 新品发布短视频，16:9，720p",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            video_skill_launch: {
              skill_name: "video_generate",
              kind: "video_task",
              video_task: {
                prompt: "新品发布短视频",
                duration: 15,
                aspect_ratio: "16:9",
                resolution: "720p",
                entry_source: "at_video_command",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "video_generate",
          replayText: "15秒 16:9 720p 新品发布短视频",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@视频 首次发送时应先绑定真实 session_id 到 video_task", async () => {
    mockEnsureSessionForCommandMetadata.mockResolvedValue("session-video-1");
    const harness = mountHook({
      input: "@视频 15秒 新品发布短视频，16:9，720p",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockEnsureSessionForCommandMetadata).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            video_skill_launch: {
              video_task: {
                session_id: "session-video-1",
              },
            },
          },
        },
      });
    } finally {
      harness.unmount();
    }
  });

  it("@播报 应保留原始消息，并通过 broadcast_skill_launch metadata 交给 Agent 调度技能", async () => {
    const harness = mountHook({
      input:
        "@播报 标题: 创始人周报 听众: AI 创业者 语气: 口语化 时长: 5分钟 把下面文章整理成播报文本",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).not.toHaveBeenCalled();
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@播报 标题: 创始人周报 听众: AI 创业者 语气: 口语化 时长: 5分钟 把下面文章整理成播报文本",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            broadcast_skill_launch: {
              skill_name: "broadcast_generate",
              kind: "broadcast_task",
              broadcast_task: {
                prompt: "把下面文章整理成播报文本",
                content: "把下面文章整理成播报文本",
                title: "创始人周报",
                audience: "AI 创业者",
                tone: "口语化",
                duration_hint_minutes: 5,
                entry_source: "at_broadcast_command",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "broadcast_generate",
          replayText:
            "标题:创始人周报 听众:AI 创业者 语气:口语化 时长:5分钟 把下面文章整理成播报文本",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@Speaker 1 应继续走播报主链，不新增第二套口播命令", async () => {
    const harness = mountHook({
      input:
        "@Speaker 1 audience: founders tone: friendly 8 minutes turn this article into an audio-ready brief",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@Speaker 1 audience: founders tone: friendly 8 minutes turn this article into an audio-ready brief",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            broadcast_skill_launch: {
              skill_name: "broadcast_generate",
              broadcast_task: {
                audience: "founders",
                tone: "friendly",
                duration_hint_minutes: 8,
                entry_source: "at_broadcast_command",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "broadcast_generate",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@播报 首次发送时应先绑定真实 session_id 到 broadcast_task", async () => {
    mockEnsureSessionForCommandMetadata.mockResolvedValue(
      "session-broadcast-1",
    );
    const harness = mountHook({
      input:
        "@播报 标题: 创始人周报 听众: AI 创业者 语气: 口语化 时长: 5分钟 把下面文章整理成播报文本",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockEnsureSessionForCommandMetadata).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            broadcast_skill_launch: {
              broadcast_task: {
                session_id: "session-broadcast-1",
              },
            },
          },
        },
      });
    } finally {
      harness.unmount();
    }
  });

  it("@素材 应保留原始消息，并通过 resource_search_skill_launch metadata 交给 Agent 调度技能", async () => {
    const harness = mountHook({
      input: "@素材 类型:图片 关键词:咖啡馆木桌背景 用途:公众号头图 数量:8",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).not.toHaveBeenCalled();
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@素材 类型:图片 关键词:咖啡馆木桌背景 用途:公众号头图 数量:8",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            resource_search_skill_launch: {
              skill_name: "modal_resource_search",
              kind: "resource_search_task",
              resource_search_task: {
                prompt: "咖啡馆木桌背景 公众号头图",
                resource_type: "image",
                query: "咖啡馆木桌背景",
                usage: "公众号头图",
                count: 8,
                entry_source: "at_resource_search_command",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "modal_resource_search",
          replayText: "类型:图片 关键词:咖啡馆木桌背景 用途:公众号头图 数量:8",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@Pinterest Image Search 应保留原始消息，并继续走图片素材检索主链", async () => {
    const harness = mountHook({
      input:
        "@Pinterest Image Search coffee packaging reference for hero banner",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@Pinterest Image Search coffee packaging reference for hero banner",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            resource_search_skill_launch: {
              skill_name: "modal_resource_search",
              kind: "resource_search_task",
              resource_search_task: {
                prompt: "coffee packaging reference",
                resource_type: "image",
                query: "coffee packaging reference",
                usage: "hero banner",
                entry_source: "at_resource_search_command",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "modal_resource_search",
          replayText:
            "类型:图片 关键词:coffee packaging reference 用途:hero banner",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@Video Search 应继续走素材检索主链，并默认锁到 video 类型", async () => {
    const harness = mountHook({
      input: "@Video Search AI startup keynote clips for 9:16 ads",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@Video Search AI startup keynote clips for 9:16 ads",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            resource_search_skill_launch: {
              skill_name: "modal_resource_search",
              resource_search_task: {
                resource_type: "video",
                query: "AI startup keynote clips",
                usage: "9:16 ads",
                entry_source: "at_resource_search_command",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "modal_resource_search",
          replayText: "类型:视频 关键词:AI startup keynote clips 用途:9:16 ads",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@素材 首次发送时应先绑定真实 session_id 到 resource_search_task", async () => {
    mockEnsureSessionForCommandMetadata.mockResolvedValue("session-resource-1");
    const harness = mountHook({
      input: "@素材 类型:图片 关键词:咖啡馆木桌背景 用途:公众号头图 数量:8",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockEnsureSessionForCommandMetadata).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            resource_search_skill_launch: {
              resource_search_task: {
                session_id: "session-resource-1",
              },
            },
          },
        },
      });
    } finally {
      harness.unmount();
    }
  });

  it("@素材 应把项目资料提词重写配置注入到 metadata 与发送覆盖", async () => {
    const harness = mountHook({
      input: "@素材 类型:图片 关键词:咖啡馆木桌背景 用途:公众号头图 数量:8",
      serviceModels: {
        resource_prompt_rewrite: {
          preferredProviderId: "resource-provider",
          preferredModelId: "resource-model",
          customPrompt: "请先结合项目资料语义补齐检索意图",
        },
      },
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        providerOverride: "resource-provider",
        modelOverride: "resource-model",
        requestMetadata: {
          harness: {
            resource_search_skill_launch: {
              resource_search_task: {
                prompt:
                  "请先结合项目资料语义补齐检索意图\n\n咖啡馆木桌背景 公众号头图",
              },
            },
          },
        },
      });
    } finally {
      harness.unmount();
    }
  });

  it("@搜索 应保留原始消息，并通过 research_skill_launch metadata 交给 Agent 调度技能", async () => {
    const harness = mountHook({
      input:
        "@搜索 关键词:AI Agent 融资 站点:36Kr 时间:近30天 深度:深度 重点:融资额与产品发布 输出:要点",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).not.toHaveBeenCalled();
      expect(mockEnsureSessionForCommandMetadata).not.toHaveBeenCalled();
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@搜索 关键词:AI Agent 融资 站点:36Kr 时间:近30天 深度:深度 重点:融资额与产品发布 输出:要点",
      );
      expect(mockSendMessage.mock.calls[0]?.[2]).toBe(true);
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        searchMode: "required",
        explicitToolPreferences: true,
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            research_skill_launch: {
              skill_name: "research",
              kind: "research_request",
              research_request: {
                prompt: "AI Agent 融资 36Kr 近30天 融资额与产品发布",
                query: "AI Agent 融资",
                site: "36Kr",
                time_range: "近30天",
                depth: "deep",
                focus: "融资额与产品发布",
                output_format: "要点",
                entry_source: "at_search_command",
                modality_contract_key: "web_research",
                modality: "mixed",
                required_capabilities: expect.arrayContaining([
                  "text_generation",
                  "web_search",
                  "structured_document_generation",
                  "long_context",
                ]),
                routing_slot: "report_generation_model",
                runtime_contract: expect.objectContaining({
                  contract_key: "web_research",
                  routing_slot: "report_generation_model",
                }),
              },
            },
          },
        },
      });
    } finally {
      harness.unmount();
    }
  });

  it("裸 @搜索 发送时也应自动带入最近成功参数", async () => {
    recordMentionEntryUsage({
      kind: "builtin_command",
      entryId: "research",
      usedAt: 1_712_345_678_900,
      slotValues: {
        query: "AI Agent 融资",
        site: "36Kr",
        time_range: "近30天",
        depth: "deep",
        focus: "融资额与产品发布",
        output_format: "要点",
      },
    });

    const harness = mountHook({
      input: "@搜索",
      serviceSkills: [
        createGeneralServiceSkill({
          id: "research",
          title: "搜索",
        }),
      ],
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@搜索 关键词:AI Agent 融资 站点:36Kr 时间:近30天 深度:深度 重点:融资额与产品发布 输出:要点",
      );
      expect(mockSendMessage.mock.calls[0]?.[2]).toBe(true);
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        searchMode: "required",
        explicitToolPreferences: true,
        requestMetadata: {
          harness: {
            research_skill_launch: {
              kind: "research_request",
              research_request: {
                query: "AI Agent 融资",
                site: "36Kr",
                time_range: "近30天",
                depth: "deep",
                focus: "融资额与产品发布",
                output_format: "要点",
                entry_source: "at_search_command",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "research",
          replayText:
            "关键词:AI Agent 融资 站点:36Kr 时间:近30天 深度:深度 重点:融资额与产品发布 输出:要点",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("部分 @搜索 输入也应补齐最近成功默认字段", async () => {
    recordMentionEntryUsage({
      kind: "builtin_command",
      entryId: "research",
      usedAt: 1_712_345_678_900,
      slotValues: {
        query: "AI Agent 融资",
        site: "36Kr",
        time_range: "近30天",
        depth: "deep",
        focus: "融资额与产品发布",
        output_format: "要点",
      },
    });

    const harness = mountHook({
      input: "@搜索 OpenAI 最新融资",
      serviceSkills: [
        createGeneralServiceSkill({
          id: "research",
          title: "搜索",
        }),
      ],
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@搜索 关键词:OpenAI 最新融资 站点:36Kr 时间:近30天 深度:深度 重点:融资额与产品发布 输出:要点",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            research_skill_launch: {
              kind: "research_request",
              research_request: {
                prompt: "OpenAI 最新融资 36Kr 近30天 融资额与产品发布",
                query: "OpenAI 最新融资",
                site: "36Kr",
                time_range: "近30天",
                depth: "deep",
                focus: "融资额与产品发布",
                output_format: "要点",
                entry_source: "at_search_command",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "research",
          replayText:
            "关键词:OpenAI 最新融资 站点:36Kr 时间:近30天 深度:深度 重点:融资额与产品发布 输出:要点",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@Daily Search 应保留原始消息，并把默认最近一天带进同一条搜索主链", async () => {
    const harness = mountHook({
      input: "@Daily Search OpenAI agent release notes",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@Daily Search OpenAI agent release notes",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            research_skill_launch: {
              skill_name: "research",
              kind: "research_request",
              research_request: {
                prompt: "OpenAI agent release notes",
                query: "OpenAI agent release notes",
                time_range: "最近一天",
                entry_source: "at_search_command",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "research",
          replayText:
            "关键词:OpenAI agent release notes 时间:最近一天 深度:标准",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@Search Agent 应继续复用 research 主链，不新增 agent 槽位协议", async () => {
    const harness = mountHook({
      input: "@Search Agent GitHub 最近一周 openai agents sdk issue 讨论",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@Search Agent GitHub 最近一周 openai agents sdk issue 讨论",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            research_skill_launch: {
              skill_name: "research",
              kind: "research_request",
              research_request: {
                query: "openai agents sdk issue 讨论",
                site: "GitHub",
                time_range: "最近一周",
                entry_source: "at_search_command",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "research",
          replayText:
            "关键词:openai agents sdk issue 讨论 站点:GitHub 时间:最近一周 深度:标准",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@Instagram Research 应默认带入 Instagram 站点并继续走 research 主链", async () => {
    const harness = mountHook({
      input: "@Instagram Research Nike reels 爆款框架",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@Instagram Research Nike reels 爆款框架",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            research_skill_launch: {
              skill_name: "research",
              kind: "research_request",
              research_request: {
                prompt: "Nike reels 爆款框架",
                query: "Nike reels 爆款框架",
                site: "Instagram",
                entry_source: "at_search_command",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "research",
          replayText: "关键词:Nike reels 爆款框架 站点:Instagram 深度:标准",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@搜索 直接输入成功后应回写对应 service skill 最近使用", async () => {
    const harness = mountHook({
      input:
        "@搜索 关键词:AI Agent 融资 站点:36Kr 时间:近30天 深度:深度 重点:融资额与产品发布 输出:要点",
      serviceSkills: [
        createGeneralServiceSkill({
          id: "research",
          title: "搜索",
        }),
      ],
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(listServiceSkillUsage()).toEqual([
        expect.objectContaining({
          skillId: "research",
          runnerType: "instant",
          launchUserInput: "AI Agent 融资 36Kr 近30天 融资额与产品发布",
          slotValues: {
            prompt: "AI Agent 融资 36Kr 近30天 融资额与产品发布",
            query: "AI Agent 融资",
            site: "36Kr",
            time_range: "近30天",
            depth: "deep",
            focus: "融资额与产品发布",
            output_format: "要点",
          },
        }),
      ]);
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "research",
          replayText:
            "关键词:AI Agent 融资 站点:36Kr 时间:近30天 深度:深度 重点:融资额与产品发布 输出:要点",
          slotValues: {
            prompt: "AI Agent 融资 36Kr 近30天 融资额与产品发布",
            query: "AI Agent 融资",
            site: "36Kr",
            time_range: "近30天",
            depth: "deep",
            focus: "融资额与产品发布",
            output_format: "要点",
          },
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("通过 active builtin command 发送时，也应回写字段化输入骨架", async () => {
    const harness = mountHook({
      input: "GitHub 最近一周 openai agents sdk issue 讨论",
      serviceSkills: [
        createGeneralServiceSkill({
          id: "research",
          title: "搜索",
        }),
      ],
    });

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend(
            undefined,
            undefined,
            undefined,
            "@搜索 GitHub 最近一周 openai agents sdk issue 讨论",
          );
        expect(started).toBe(true);
      });

      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "research",
          replayText:
            "关键词:openai agents sdk issue 讨论 站点:GitHub 时间:最近一周 深度:标准",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("通过 active installed skill route 发送时，应归一 local:* 并回写 slash skill 最近使用", async () => {
    const harness = mountHook({
      input: "整理最近发布计划",
    });

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend(
            undefined,
            undefined,
            undefined,
            undefined,
            "react",
            undefined,
            {
              capabilityRoute: {
                kind: "installed_skill",
                skillKey: "local:writer",
                skillName: "写作助手",
              },
              displayContent: "整理最近发布计划",
            },
          );
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "/writer 整理最近发布计划",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        displayContent: "整理最近发布计划",
        capabilityRoute: {
          kind: "installed_skill",
          skillKey: "writer",
          skillName: "写作助手",
        },
      });
      expect(listSlashEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "skill",
          entryId: "writer",
          replayText: "整理最近发布计划",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("通过 active builtin command route 发送时，应由发送链还原内部命令文本并保留原始显示文案", async () => {
    const harness = mountHook({
      input: "GitHub 最近一周 openai agents sdk issue 讨论",
      serviceSkills: [
        createGeneralServiceSkill({
          id: "research",
          title: "搜索",
        }),
      ],
    });

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend(
            undefined,
            undefined,
            undefined,
            undefined,
            "react",
            undefined,
            {
              capabilityRoute: {
                kind: "builtin_command",
                commandKey: "research",
                commandPrefix: "@搜索",
              },
              displayContent: "GitHub 最近一周 openai agents sdk issue 讨论",
            },
          );
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@搜索 GitHub 最近一周 openai agents sdk issue 讨论",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        displayContent: "GitHub 最近一周 openai agents sdk issue 讨论",
        capabilityRoute: {
          kind: "builtin_command",
          commandKey: "research",
          commandPrefix: "@搜索",
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "research",
          replayText:
            "关键词:openai agents sdk issue 讨论 站点:GitHub 时间:最近一周 深度:标准",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("通过 active runtime scene route 发送时，应由发送链还原 scene 命令文本并回写 scene 最近使用", async () => {
    mockGetSkillCatalog.mockResolvedValueOnce({
      entries: [
        {
          id: "scene:campaign-launch",
        },
      ],
    });
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
    const harness = mountHook({
      input: "帮我做一版新品活动启动方案",
      serviceSkills: [createCloudSceneSkill()],
    });

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend(
            undefined,
            undefined,
            undefined,
            undefined,
            "react",
            undefined,
            {
              capabilityRoute: {
                kind: "runtime_scene",
                sceneKey: "campaign-launch",
                commandPrefix: "/campaign-launch",
              },
              displayContent: "帮我做一版新品活动启动方案",
            },
          );
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toContain(
        "[技能任务] 视频配音",
      );
      expect(mockSendMessage.mock.calls[0]?.[0]).toContain(
        "[补充要求] 帮我做一版新品活动启动方案",
      );
      const sendOptions = mockSendMessage.mock.calls[0]?.[8];
      expect(sendOptions).toMatchObject({
        displayContent: "帮我做一版新品活动启动方案",
        capabilityRoute: {
          kind: "runtime_scene",
          sceneKey: "campaign-launch",
          commandPrefix: "/campaign-launch",
        },
      });
      expect(sendOptions?.requestMetadata).toMatchObject({
        harness: {
          service_scene_launch: {
            kind: "local_service_skill",
            service_scene_run: expect.objectContaining({
              scene_key: "campaign-launch",
              user_input: "帮我做一版新品活动启动方案",
            }),
          },
        },
      });
      expect(listSlashEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "scene",
          entryId: "campaign-launch",
          replayText: "帮我做一版新品活动启动方案",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@深搜 应保留原始消息，并通过 deep_search_skill_launch metadata 交给 Agent 调度 research 技能", async () => {
    const harness = mountHook({
      input:
        "@深搜 关键词:AI Agent 融资 站点:36Kr 时间:近30天 重点:融资额与产品发布 输出:对比表",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).not.toHaveBeenCalled();
      expect(mockEnsureSessionForCommandMetadata).not.toHaveBeenCalled();
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@深搜 关键词:AI Agent 融资 站点:36Kr 时间:近30天 重点:融资额与产品发布 输出:对比表",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            deep_search_skill_launch: {
              skill_name: "research",
              kind: "deep_search_request",
              deep_search_request: {
                prompt: "AI Agent 融资 36Kr 近30天 融资额与产品发布",
                query: "AI Agent 融资",
                site: "36Kr",
                time_range: "近30天",
                depth: "deep",
                focus: "融资额与产品发布",
                output_format: "对比表",
                entry_source: "at_deep_search_command",
                modality_contract_key: "web_research",
                modality: "mixed",
                required_capabilities: expect.arrayContaining([
                  "text_generation",
                  "web_search",
                  "structured_document_generation",
                  "long_context",
                ]),
                routing_slot: "report_generation_model",
                runtime_contract: expect.objectContaining({
                  contract_key: "web_research",
                  routing_slot: "report_generation_model",
                }),
              },
            },
          },
        },
      });
    } finally {
      harness.unmount();
    }
  });

  it("@Researchers Pro 应继续走深搜主链，不新增专家命令协议", async () => {
    const harness = mountHook({
      input: "@Researchers Pro GitHub 最近一周 openai agents sdk issue 讨论",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@Researchers Pro GitHub 最近一周 openai agents sdk issue 讨论",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            deep_search_skill_launch: {
              skill_name: "research",
              deep_search_request: {
                site: "GitHub",
                time_range: "最近一周",
                query: "openai agents sdk issue 讨论",
                depth: "deep",
                entry_source: "at_deep_search_command",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "deep_search",
          replayText:
            "关键词:openai agents sdk issue 讨论 站点:GitHub 时间:最近一周 深度:深度",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@深搜 成功后应回写字段化最近使用", async () => {
    const harness = mountHook({
      input: "@深搜 GitHub 最近一周 openai agents sdk issue 讨论",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "deep_search",
          replayText:
            "关键词:openai agents sdk issue 讨论 站点:GitHub 时间:最近一周 深度:深度",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@研报 应保留原始消息，并通过 report_skill_launch metadata 交给 Agent 调度 report_generate 技能", async () => {
    const harness = mountHook({
      input:
        "@研报 关键词:AI Agent 融资 站点:36Kr 时间:近30天 重点:融资额与代表产品 输出:投资人研报",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).not.toHaveBeenCalled();
      expect(mockEnsureSessionForCommandMetadata).not.toHaveBeenCalled();
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@研报 关键词:AI Agent 融资 站点:36Kr 时间:近30天 重点:融资额与代表产品 输出:投资人研报",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            report_skill_launch: {
              skill_name: "report_generate",
              kind: "report_request",
              report_request: {
                prompt: "AI Agent 融资 36Kr 近30天 融资额与代表产品 投资人研报",
                query: "AI Agent 融资",
                site: "36Kr",
                time_range: "近30天",
                depth: "deep",
                focus: "融资额与代表产品",
                output_format: "投资人研报",
                entry_source: "at_report_command",
                modality_contract_key: "web_research",
                modality: "mixed",
                required_capabilities: expect.arrayContaining(["web_search"]),
                routing_slot: "report_generation_model",
                runtime_contract: expect.objectContaining({
                  contract_key: "web_research",
                  routing_slot: "report_generation_model",
                }),
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "research_report",
          replayText:
            "关键词:AI Agent 融资 站点:36Kr 时间:近30天 重点:融资额与代表产品 输出:投资人研报",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@Report Search 应保留原始消息，并继续走同一条研报主链", async () => {
    const harness = mountHook({
      input:
        "@Report Search query: AI Agent funding site: 36Kr range: 2026 output: investor memo",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@Report Search query: AI Agent funding site: 36Kr range: 2026 output: investor memo",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            report_skill_launch: {
              skill_name: "report_generate",
              kind: "report_request",
              report_request: {
                prompt: "AI Agent funding 36Kr 2026 investor memo",
                query: "AI Agent funding",
                site: "36Kr",
                time_range: "2026",
                depth: "deep",
                output_format: "investor memo",
                entry_source: "at_report_command",
                modality_contract_key: "web_research",
                modality: "mixed",
                required_capabilities: expect.arrayContaining(["web_search"]),
                routing_slot: "report_generation_model",
                runtime_contract: expect.objectContaining({
                  contract_key: "web_research",
                  routing_slot: "report_generation_model",
                }),
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "research_report",
          replayText:
            "关键词:AI Agent funding 站点:36Kr 时间:2026 输出:investor memo",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@竞品 应保留原始消息，并通过 report_skill_launch metadata 交给 Agent 调度竞品分析主链", async () => {
    const harness = mountHook({
      input: "@竞品 Claude 与 Gemini 在中国开发者市场的差异",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@竞品 Claude 与 Gemini 在中国开发者市场的差异",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            report_skill_launch: {
              skill_name: "report_generate",
              kind: "report_request",
              report_request: {
                prompt: "Claude 与 Gemini 在中国开发者市场的差异",
                query: "Claude 与 Gemini 在中国开发者市场的差异",
                depth: "deep",
                focus:
                  "产品定位、目标用户、核心功能、定价模式、渠道策略、差异化优劣势",
                output_format: "竞品分析",
                entry_source: "at_competitor_command",
                modality_contract_key: "web_research",
                modality: "mixed",
                required_capabilities: expect.arrayContaining(["web_search"]),
                routing_slot: "report_generation_model",
                runtime_contract: expect.objectContaining({
                  contract_key: "web_research",
                  routing_slot: "report_generation_model",
                }),
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "competitor_research",
          replayText:
            "关键词:Claude 与 Gemini 在中国开发者市场的差异 重点:产品定位、目标用户、核心功能、定价模式、渠道策略、差异化优劣势 输出:竞品分析",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@Product Search 应继续走竞品研究主链，不新增产品搜索协议", async () => {
    const harness = mountHook({
      input: "@Product Search OpenAI Operator 与 Manus 的产品定位差异",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@Product Search OpenAI Operator 与 Manus 的产品定位差异",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            report_skill_launch: {
              skill_name: "report_generate",
              kind: "report_request",
              report_request: {
                prompt: "OpenAI Operator 与 Manus 的产品定位差异",
                query: "OpenAI Operator 与 Manus 的产品定位差异",
                focus:
                  "产品定位、目标用户、核心功能、定价模式、渠道策略、差异化优劣势",
                output_format: "竞品分析",
                entry_source: "at_competitor_command",
                modality_contract_key: "web_research",
                modality: "mixed",
                required_capabilities: expect.arrayContaining(["web_search"]),
                routing_slot: "report_generation_model",
                runtime_contract: expect.objectContaining({
                  contract_key: "web_research",
                  routing_slot: "report_generation_model",
                }),
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "competitor_research",
          replayText:
            "关键词:OpenAI Operator 与 Manus 的产品定位差异 重点:产品定位、目标用户、核心功能、定价模式、渠道策略、差异化优劣势 输出:竞品分析",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@站点搜索 应保留原始消息，并通过 site_search_skill_launch metadata 交给 Agent 调度技能", async () => {
    const harness = mountHook({
      input: "@站点搜索 站点:GitHub 关键词:openai agents sdk issue 数量:8",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).not.toHaveBeenCalled();
      expect(mockEnsureSessionForCommandMetadata).not.toHaveBeenCalled();
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@站点搜索 站点:GitHub 关键词:openai agents sdk issue 数量:8",
      );
      expect(mockSendMessage.mock.calls[0]?.[2]).toBe(false);
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            site_search_skill_launch: {
              skill_name: "site_search",
              kind: "site_search_request",
              site_search_request: {
                prompt: "openai agents sdk issue",
                site: "GitHub",
                query: "openai agents sdk issue",
                limit: 8,
                entry_source: "at_site_search_command",
                modality_contract_key: "web_research",
                modality: "mixed",
                required_capabilities: expect.arrayContaining([
                  "text_generation",
                  "web_search",
                  "structured_document_generation",
                  "long_context",
                ]),
                routing_slot: "report_generation_model",
                runtime_contract: expect.objectContaining({
                  contract_key: "web_research",
                  routing_slot: "report_generation_model",
                }),
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "site_search",
          replayText: "站点:GitHub 关键词:openai agents sdk issue 数量:8",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@读PDF 应保留原始消息，并通过 pdf_read_skill_launch metadata 交给 Agent 调度技能", async () => {
    const harness = mountHook({
      input: '@读PDF "/tmp/agent-report.pdf" 提炼三点结论 输出:投资人摘要',
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        '@读PDF "/tmp/agent-report.pdf" 提炼三点结论 输出:投资人摘要',
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            pdf_read_skill_launch: {
              skill_name: "pdf_read",
              kind: "pdf_read_request",
              pdf_read_request: {
                prompt: "提炼三点结论",
                source_path: "/tmp/agent-report.pdf",
                output_format: "投资人摘要",
                entry_source: "at_pdf_read_command",
                modality_contract_key: "pdf_extract",
                modality: "document",
                required_capabilities: expect.arrayContaining([
                  "local_file_read",
                ]),
                routing_slot: "base_model",
                runtime_contract: expect.objectContaining({
                  contract_key: "pdf_extract",
                  executor_binding: expect.objectContaining({
                    binding_key: "pdf_read",
                  }),
                }),
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "read_pdf",
          replayText:
            "文件:/tmp/agent-report.pdf 输出:投资人摘要 要求:提炼三点结论",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@读文件 应保留原始消息，并通过 summary_skill_launch metadata 交给 Agent 调度技能", async () => {
    const harness = mountHook({
      input: '@读文件 "/tmp/agent-notes.md" 提炼三点结论 输出:投资人摘要',
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        '@读文件 "/tmp/agent-notes.md" 提炼三点结论 输出:投资人摘要',
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            summary_skill_launch: {
              skill_name: "summary",
              kind: "summary_request",
              summary_request: {
                prompt: "提炼三点结论",
                source_path: "/tmp/agent-notes.md",
                output_format: "投资人摘要",
                entry_source: "at_file_read_command",
                modality_contract_key: "text_transform",
                modality: "document",
                required_capabilities: expect.arrayContaining([
                  "text_generation",
                  "local_file_read",
                ]),
                routing_slot: "base_model",
                runtime_contract: expect.objectContaining({
                  contract_key: "text_transform",
                  executor_binding: expect.objectContaining({
                    binding_key: "text_transform",
                  }),
                }),
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "file_read_runtime",
          replayText:
            "文件:/tmp/agent-notes.md 输出:投资人摘要 要求:提炼三点结论",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@Read File Content 应兼容英文 trigger，并继续走 summary 当前主链", async () => {
    const harness = mountHook({
      input: '@Read File Content "/tmp/agent-notes.md" output:三点要点',
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        '@Read File Content "/tmp/agent-notes.md" output:三点要点',
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            summary_skill_launch: {
              summary_request: {
                source_path: "/tmp/agent-notes.md",
                output_format: "三点要点",
                entry_source: "at_file_read_command",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          entryId: "file_read_runtime",
          replayText: "文件:/tmp/agent-notes.md 输出:三点要点",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@总结 应保留原始消息，并通过 summary_skill_launch metadata 交给 Agent 调度技能", async () => {
    const harness = mountHook({
      input:
        "@总结 内容:这是一篇关于 AI Agent 融资的长文 重点:融资额与发布时间 长度:简短 风格:投资人简报 输出:三点要点",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).not.toHaveBeenCalled();
      expect(mockEnsureSessionForCommandMetadata).not.toHaveBeenCalled();
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@总结 内容:这是一篇关于 AI Agent 融资的长文 重点:融资额与发布时间 长度:简短 风格:投资人简报 输出:三点要点",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            summary_skill_launch: {
              skill_name: "summary",
              kind: "summary_request",
              summary_request: {
                prompt:
                  "这是一篇关于 AI Agent 融资的长文 融资额与发布时间 short 投资人简报",
                content: "这是一篇关于 AI Agent 融资的长文",
                focus: "融资额与发布时间",
                length: "short",
                style: "投资人简报",
                output_format: "三点要点",
                entry_source: "at_summary_command",
                modality_contract_key: "text_transform",
                modality: "document",
                required_capabilities: expect.arrayContaining([
                  "text_generation",
                  "local_file_read",
                ]),
                routing_slot: "base_model",
                runtime_contract: expect.objectContaining({
                  contract_key: "text_transform",
                  executor_binding: expect.objectContaining({
                    binding_key: "text_transform",
                  }),
                }),
              },
            },
          },
        },
      });
    } finally {
      harness.unmount();
    }
  });

  it("@总结 成功后应按固定字段顺序回写最近使用", async () => {
    const harness = mountHook({
      input:
        "@总结 风格:投资人简报 输出:三点要点 长度:简短 内容:这是一篇关于 AI Agent 融资的长文 重点:融资额与发布时间",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "summary",
          replayText:
            "内容:这是一篇关于 AI Agent 融资的长文 重点:融资额与发布时间 长度:简短 风格:投资人简报 输出:三点要点",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("部分 @总结 输入应补齐最近成功偏好，但不复用旧内容", async () => {
    recordMentionEntryUsage({
      kind: "builtin_command",
      entryId: "summary",
      usedAt: 1_712_345_678_900,
      slotValues: {
        content: "旧内容不应复用",
        focus: "融资额与发布时间",
        length: "short",
        style: "投资人简报",
        output_format: "三点要点",
      },
    });

    const harness = mountHook({
      input: "@总结 内容:OpenAI 发布会纪要",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@总结 内容:OpenAI 发布会纪要",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            summary_skill_launch: {
              summary_request: {
                prompt: "OpenAI 发布会纪要",
                content: "OpenAI 发布会纪要",
                focus: "融资额与发布时间",
                length: "short",
                style: "投资人简报",
                output_format: "三点要点",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          entryId: "summary",
          replayText:
            "内容:OpenAI 发布会纪要 重点:融资额与发布时间 长度:简短 风格:投资人简报 输出:三点要点",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@翻译 应保留原始消息，并通过 translation_skill_launch metadata 交给 Agent 调度技能", async () => {
    const harness = mountHook({
      input:
        "@翻译 内容:hello world 原语言:英语 目标语言:中文 风格:产品文案 输出:只输出译文",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).not.toHaveBeenCalled();
      expect(mockEnsureSessionForCommandMetadata).not.toHaveBeenCalled();
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@翻译 内容:hello world 原语言:英语 目标语言:中文 风格:产品文案 输出:只输出译文",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            translation_skill_launch: {
              skill_name: "translation",
              kind: "translation_request",
              translation_request: {
                prompt: "hello world 从英语 译为中文 产品文案 只输出译文",
                content: "hello world",
                source_language: "英语",
                target_language: "中文",
                style: "产品文案",
                output_format: "只输出译文",
                entry_source: "at_translation_command",
                modality_contract_key: "text_transform",
                modality: "document",
                required_capabilities: expect.arrayContaining([
                  "text_generation",
                  "local_file_read",
                ]),
                routing_slot: "base_model",
                runtime_contract: expect.objectContaining({
                  contract_key: "text_transform",
                  executor_binding: expect.objectContaining({
                    binding_key: "text_transform",
                  }),
                }),
              },
            },
          },
        },
      });
    } finally {
      harness.unmount();
    }
  });

  it("@翻译 成功后应按固定字段顺序回写最近使用", async () => {
    const harness = mountHook({
      input:
        "@翻译 风格:产品文案 输出:只输出译文 目标语言:中文 内容:hello world 原语言:英语",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "translation",
          replayText:
            "内容:hello world 原语言:英语 目标语言:中文 风格:产品文案 输出:只输出译文",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@翻译 应把 translation 服务模型写入发送覆盖", async () => {
    const harness = mountHook({
      input:
        "@翻译 内容:hello world 原语言:英语 目标语言:中文 风格:产品文案 输出:只输出译文",
      serviceModels: {
        translation: {
          preferredProviderId: "translation-provider",
          preferredModelId: "translation-model",
        },
      },
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        providerOverride: "translation-provider",
        modelOverride: "translation-model",
      });
    } finally {
      harness.unmount();
    }
  });

  it("部分 @翻译 输入应补齐最近成功语言与风格偏好", async () => {
    recordMentionEntryUsage({
      kind: "builtin_command",
      entryId: "translation",
      usedAt: 1_712_345_678_900,
      slotValues: {
        content: "旧内容不应复用",
        source_language: "英语",
        target_language: "中文",
        style: "产品文案",
        output_format: "只输出译文",
      },
    });

    const harness = mountHook({
      input: "@翻译 内容:hello world",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage.mock.calls[0]?.[0]).toBe("@翻译 内容:hello world");
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            translation_skill_launch: {
              translation_request: {
                prompt: "hello world",
                content: "hello world",
                source_language: "英语",
                target_language: "中文",
                style: "产品文案",
                output_format: "只输出译文",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          entryId: "translation",
          replayText:
            "内容:hello world 原语言:英语 目标语言:中文 风格:产品文案 输出:只输出译文",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@分析 应保留原始消息，并通过 analysis_skill_launch metadata 交给 Agent 调度技能", async () => {
    const harness = mountHook({
      input:
        "@分析 内容:OpenAI 发布新模型 重点:商业影响 风格:投资备忘 输出:三点判断",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).not.toHaveBeenCalled();
      expect(mockEnsureSessionForCommandMetadata).not.toHaveBeenCalled();
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@分析 内容:OpenAI 发布新模型 重点:商业影响 风格:投资备忘 输出:三点判断",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            analysis_skill_launch: {
              skill_name: "analysis",
              kind: "analysis_request",
              analysis_request: {
                prompt: "OpenAI 发布新模型 围绕商业影响 投资备忘 三点判断",
                content: "OpenAI 发布新模型",
                focus: "商业影响",
                style: "投资备忘",
                output_format: "三点判断",
                entry_source: "at_analysis_command",
                modality_contract_key: "text_transform",
                modality: "document",
                required_capabilities: expect.arrayContaining([
                  "text_generation",
                  "local_file_read",
                ]),
                routing_slot: "base_model",
                runtime_contract: expect.objectContaining({
                  contract_key: "text_transform",
                  executor_binding: expect.objectContaining({
                    binding_key: "text_transform",
                  }),
                }),
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "analysis",
          replayText:
            "内容:OpenAI 发布新模型 重点:商业影响 风格:投资备忘 输出:三点判断",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("部分 @分析 输入应补齐最近成功的分析偏好", async () => {
    recordMentionEntryUsage({
      kind: "builtin_command",
      entryId: "analysis",
      usedAt: 1_712_345_678_900,
      slotValues: {
        content: "旧内容不应复用",
        focus: "商业影响",
        style: "投资备忘",
        output_format: "三点判断",
      },
    });

    const harness = mountHook({
      input: "@分析 内容:OpenAI 发布新模型",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@分析 内容:OpenAI 发布新模型",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            analysis_skill_launch: {
              analysis_request: {
                prompt: "OpenAI 发布新模型",
                content: "OpenAI 发布新模型",
                focus: "商业影响",
                style: "投资备忘",
                output_format: "三点判断",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          entryId: "analysis",
          replayText:
            "内容:OpenAI 发布新模型 重点:商业影响 风格:投资备忘 输出:三点判断",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@Logo拆解 应复用 analysis_skill_launch，但保留独立前台入口", async () => {
    const harness = mountHook({
      input: "@Logo拆解 内容:品牌新 Logo 重点:配色与字形 输出:拆解清单",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@Logo拆解 内容:品牌新 Logo 重点:配色与字形 输出:拆解清单",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            analysis_skill_launch: {
              skill_name: "analysis",
              kind: "analysis_request",
              analysis_request: {
                prompt: "品牌新 Logo 围绕配色与字形 拆解清单",
                content: "品牌新 Logo",
                focus: "配色与字形",
                output_format: "拆解清单",
                analysis_mode: "image_logo_decomposition",
                entry_source: "at_logo_decomposition_command",
                modality_contract_key: "text_transform",
                modality: "document",
                required_capabilities: expect.arrayContaining([
                  "text_generation",
                  "local_file_read",
                ]),
                routing_slot: "base_model",
                runtime_contract: expect.objectContaining({
                  contract_key: "text_transform",
                  executor_binding: expect.objectContaining({
                    binding_key: "text_transform",
                  }),
                }),
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "logo_decomposition",
          replayText: "内容:品牌新 Logo 重点:配色与字形 输出:拆解清单",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@发布合规 应保留原始消息，并通过 analysis_skill_launch metadata 交给 Agent 调度技能", async () => {
    const harness = mountHook({
      input:
        "@发布合规 内容:这是一篇小红书种草文案 重点:夸大宣传 输出:风险清单",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).not.toHaveBeenCalled();
      expect(mockEnsureSessionForCommandMetadata).not.toHaveBeenCalled();
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@发布合规 内容:这是一篇小红书种草文案 重点:夸大宣传 输出:风险清单",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            analysis_skill_launch: {
              skill_name: "analysis",
              kind: "analysis_request",
              analysis_request: {
                prompt: "这是一篇小红书种草文案 围绕夸大宣传 风险清单",
                content: "这是一篇小红书种草文案",
                focus: "夸大宣传",
                style: "合规审校",
                output_format: "风险清单",
                entry_source: "at_publish_compliance_command",
                modality_contract_key: "text_transform",
                modality: "document",
                required_capabilities: expect.arrayContaining([
                  "text_generation",
                  "local_file_read",
                ]),
                routing_slot: "base_model",
                runtime_contract: expect.objectContaining({
                  contract_key: "text_transform",
                  executor_binding: expect.objectContaining({
                    binding_key: "text_transform",
                  }),
                }),
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "publish_compliance",
          replayText:
            "内容:这是一篇小红书种草文案 重点:夸大宣传 风格:合规审校 输出:风险清单",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("部分 @发布合规 输入应补齐最近成功的合规偏好", async () => {
    recordMentionEntryUsage({
      kind: "builtin_command",
      entryId: "publish_compliance",
      usedAt: 1_712_345_678_900,
      slotValues: {
        content: "旧内容不应复用",
        focus: "夸大宣传",
        style: "法务审校",
        output_format: "风险清单",
      },
    });

    const harness = mountHook({
      input: "@发布合规 内容:这是一篇小红书种草文案",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@发布合规 内容:这是一篇小红书种草文案",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            analysis_skill_launch: {
              analysis_request: {
                prompt: "这是一篇小红书种草文案",
                content: "这是一篇小红书种草文案",
                focus: "夸大宣传",
                style: "法务审校",
                output_format: "风险清单",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          entryId: "publish_compliance",
          replayText:
            "内容:这是一篇小红书种草文案 重点:夸大宣传 风格:法务审校 输出:风险清单",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@转写 应保留原始消息，并通过 transcription_skill_launch metadata 交给 Agent 调度技能", async () => {
    const harness = mountHook({
      input:
        "@转写 https://example.com/interview.mp4 生成逐字稿 导出 srt 带时间戳 区分说话人",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).not.toHaveBeenCalled();
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@转写 https://example.com/interview.mp4 生成逐字稿 导出 srt 带时间戳 区分说话人",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            transcription_skill_launch: {
              skill_name: "transcription_generate",
              kind: "transcription_task",
              transcription_task: {
                prompt: "逐字稿",
                source_url: "https://example.com/interview.mp4",
                output_format: "srt",
                timestamps: true,
                speaker_labels: true,
                entry_source: "at_transcription_command",
                modality_contract_key: "audio_transcription",
                modality: "audio",
                required_capabilities: [
                  "text_generation",
                  "audio_transcription",
                ],
                routing_slot: "audio_transcription_model",
                runtime_contract: expect.objectContaining({
                  contract_key: "audio_transcription",
                  routing_slot: "audio_transcription_model",
                  executor_binding: expect.objectContaining({
                    binding_key: "mediaTaskArtifact/transcription/create",
                  }),
                }),
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "transcription_generate",
          replayText:
            "https://example.com/interview.mp4 格式:srt 区分说话人 带时间戳 逐字稿",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@转写 首次发送时应先绑定真实 session_id 到 transcription_task", async () => {
    mockEnsureSessionForCommandMetadata.mockResolvedValue(
      "session-transcription-1",
    );
    const harness = mountHook({
      input:
        "@转写 https://example.com/interview.mp4 生成逐字稿 导出 srt 带时间戳 区分说话人",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockEnsureSessionForCommandMetadata).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            transcription_skill_launch: {
              transcription_task: {
                session_id: "session-transcription-1",
              },
            },
          },
        },
      });
    } finally {
      harness.unmount();
    }
  });

  it("@修图 直接输入成功后应归并回写 image_generate 的最近使用", async () => {
    mockResolveImageWorkbenchCommandRequest.mockReturnValueOnce({
      images: [],
      requestContext: {
        kind: "image_task",
        image_task: {
          mode: "edit",
          prompt: "去掉角标，保留主体",
          target_output_ref_id: "img-2",
        },
      },
    });
    const harness = mountHook({
      input: "@修图 #img-2 去掉角标，保留主体",
      serviceSkills: [
        createGeneralServiceSkill({
          id: "image_generate",
          title: "配图",
          defaultExecutorBinding: "native_skill",
        }),
      ],
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(listServiceSkillUsage()).toEqual([
        expect.objectContaining({
          skillId: "image_generate",
          runnerType: "instant",
          slotValues: {
            mode: "edit",
            prompt: "去掉角标，保留主体",
            target_output_ref_id: "img-2",
          },
        }),
      ]);
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "image_edit",
          replayText: "#img-2 去掉角标，保留主体",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@搜索 发送失败时不应误记最近使用", async () => {
    mockSendMessage.mockRejectedValueOnce(new Error("network down"));
    const toastErrorSpy = vi
      .spyOn(toast, "error")
      .mockImplementation(() => "test-toast");
    const harness = mountHook({
      input:
        "@搜索 关键词:AI Agent 融资 站点:36Kr 时间:近30天 深度:深度 重点:融资额与产品发布 输出:要点",
      serviceSkills: [
        createGeneralServiceSkill({
          id: "research",
          title: "搜索",
        }),
      ],
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(false);
      });

      expect(toastErrorSpy).toHaveBeenCalledWith("发送失败: network down");
      expect(listServiceSkillUsage()).toEqual([]);
      expect(listMentionEntryUsage()).toEqual([]);
    } finally {
      harness.unmount();
      toastErrorSpy.mockRestore();
    }
  });

  it("@链接解析 应保留原始消息，并通过 url_parse_skill_launch metadata 交给 Agent 调度技能", async () => {
    const harness = mountHook({
      input:
        "@链接解析 https://example.com/agent 提取要点 并整理成投资人可读摘要",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).not.toHaveBeenCalled();
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@链接解析 https://example.com/agent 提取要点 并整理成投资人可读摘要",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            url_parse_skill_launch: {
              skill_name: "url_parse",
              kind: "url_parse_task",
              url_parse_task: {
                url: "https://example.com/agent",
                extract_goal: "key_points",
                prompt: "并整理成投资人可读摘要",
                entry_source: "at_url_parse_command",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "url_parse",
          replayText:
            "链接:https://example.com/agent 提取:要点 要求:并整理成投资人可读摘要",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@链接解析 首次发送时应先绑定真实 session_id 到 url_parse_task", async () => {
    mockEnsureSessionForCommandMetadata.mockResolvedValue(
      "session-url-parse-1",
    );
    const harness = mountHook({
      input:
        "@链接解析 https://example.com/agent 提取要点 并整理成投资人可读摘要",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockEnsureSessionForCommandMetadata).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            url_parse_skill_launch: {
              url_parse_task: {
                session_id: "session-url-parse-1",
              },
            },
          },
        },
      });
    } finally {
      harness.unmount();
    }
  });

  it("@抓取 应保留原始消息，并复用 url_parse task 主链提交网页正文抓取任务", async () => {
    const harness = mountHook({
      input: "@抓取 https://example.com/post 帮我抓正文并整理成素材库摘要",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@抓取 https://example.com/post 帮我抓正文并整理成素材库摘要",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            url_parse_skill_launch: {
              skill_name: "url_parse",
              kind: "url_parse_task",
              url_parse_task: {
                url: "https://example.com/post",
                extract_goal: "full_text",
                prompt: "帮我抓正文并整理成素材库摘要",
                entry_source: "at_web_scrape_command",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "web_scrape",
          replayText:
            "链接:https://example.com/post 提取:正文 要求:帮我抓正文并整理成素材库摘要",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@Fetch 应保留原始消息，并复用同一条网页抓取主链", async () => {
    const harness = mountHook({
      input: "@Fetch https://example.com/post fetch full text for archive",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@Fetch https://example.com/post fetch full text for archive",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            url_parse_skill_launch: {
              skill_name: "url_parse",
              kind: "url_parse_task",
              url_parse_task: {
                url: "https://example.com/post",
                extract_goal: "full_text",
                prompt: "for archive",
                entry_source: "at_web_scrape_command",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "web_scrape",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@网页读取 应保留原始消息，并复用 url_parse task 主链提交网页阅读任务", async () => {
    const harness = mountHook({
      input:
        "@网页读取 https://example.com/post 帮我读这篇文章并告诉我核心结论",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@网页读取 https://example.com/post 帮我读这篇文章并告诉我核心结论",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            url_parse_skill_launch: {
              skill_name: "url_parse",
              kind: "url_parse_task",
              url_parse_task: {
                url: "https://example.com/post",
                extract_goal: "summary",
                prompt: "帮我读这篇文章并告诉我核心结论",
                entry_source: "at_webpage_read_command",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "webpage_read",
          replayText:
            "链接:https://example.com/post 提取:摘要 要求:帮我读这篇文章并告诉我核心结论",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@Read Webpage 应保留原始消息，并复用同一条网页阅读主链", async () => {
    const harness = mountHook({
      input:
        "@Read Webpage https://example.com/post summarize the launch notes",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@Read Webpage https://example.com/post summarize the launch notes",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            url_parse_skill_launch: {
              skill_name: "url_parse",
              kind: "url_parse_task",
              url_parse_task: {
                url: "https://example.com/post",
                extract_goal: "summary",
                prompt: "the launch notes",
                entry_source: "at_webpage_read_command",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "webpage_read",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@URL Summarize 应保留原始消息，并复用同一条网页摘要主链", async () => {
    const harness = mountHook({
      input:
        "@URL Summarize https://example.com/post summarize the launch notes",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@URL Summarize https://example.com/post summarize the launch notes",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            url_parse_skill_launch: {
              skill_name: "url_parse",
              kind: "url_parse_task",
              url_parse_task: {
                url: "https://example.com/post",
                extract_goal: "summary",
                prompt: "the launch notes",
                entry_source: "at_webpage_read_command",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "webpage_read",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@排版 应保留原始消息，并通过 typesetting_skill_launch metadata 交给 Agent 调度技能", async () => {
    const harness = mountHook({
      input: "@排版 平台:小红书 帮我把下面文案整理成短句节奏",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).not.toHaveBeenCalled();
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@排版 平台:小红书 帮我把下面文案整理成短句节奏",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            typesetting_skill_launch: {
              skill_name: "typesetting",
              kind: "typesetting_task",
              typesetting_task: {
                prompt: "帮我把下面文案整理成短句节奏",
                content: "平台:小红书 帮我把下面文案整理成短句节奏",
                target_platform: "小红书",
                entry_source: "at_typesetting_command",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "typesetting",
          replayText: "平台:小红书 要求:帮我把下面文案整理成短句节奏",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@排版 首次发送时应先绑定真实 session_id 到 typesetting_task", async () => {
    mockEnsureSessionForCommandMetadata.mockResolvedValue(
      "session-typesetting-1",
    );
    const harness = mountHook({
      input: "@排版 平台:公众号 帮我整理成更适合阅读的段落节奏",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockEnsureSessionForCommandMetadata).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            typesetting_skill_launch: {
              typesetting_task: {
                session_id: "session-typesetting-1",
              },
            },
          },
        },
      });
    } finally {
      harness.unmount();
    }
  });

  it("部分 @排版 输入应补齐最近成功的平台偏好", async () => {
    recordMentionEntryUsage({
      kind: "builtin_command",
      entryId: "typesetting",
      usedAt: 1_712_345_678_900,
      slotValues: {
        target_platform: "公众号",
        prompt: "旧要求不应复用",
        content: "旧内容不应复用",
      },
    });

    const harness = mountHook({
      input: "@排版 帮我把下面文案整理成短句节奏",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@排版 帮我把下面文案整理成短句节奏",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            typesetting_skill_launch: {
              typesetting_task: {
                prompt: "帮我把下面文案整理成短句节奏",
                content: "帮我把下面文案整理成短句节奏",
                target_platform: "公众号",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          entryId: "typesetting",
          replayText: "平台:公众号 要求:帮我把下面文案整理成短句节奏",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@网页 应保留原始消息，并通过 webpage_skill_launch metadata 交给 Agent 调度技能", async () => {
    const harness = mountHook({
      input:
        "@网页 类型:落地页 风格:未来感 技术:原生 HTML 帮我做一个 AI 代码助手官网",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockResolveImageWorkbenchCommandRequest).not.toHaveBeenCalled();
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@网页 类型:落地页 风格:未来感 技术:原生 HTML 帮我做一个 AI 代码助手官网",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            webpage_skill_launch: {
              skill_name: "webpage_generate",
              kind: "webpage_request",
              webpage_request: {
                prompt: "帮我做一个 AI 代码助手官网",
                content:
                  "类型:落地页 风格:未来感 技术:原生 HTML 帮我做一个 AI 代码助手官网",
                page_type: "landing_page",
                style: "未来感",
                tech_stack: "原生 HTML",
                entry_source: "at_webpage_command",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "webpage_generate",
          replayText:
            "类型:落地页 风格:未来感 技术:原生 HTML 要求:帮我做一个 AI 代码助手官网",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@Web Style 应保留原始消息，并继续走同一条网页生成主链", async () => {
    const harness = mountHook({
      input:
        "@Web Style 类型:官网 风格:glassmorphism 帮我做一个 AI workspace 官网",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@Web Style 类型:官网 风格:glassmorphism 帮我做一个 AI workspace 官网",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            webpage_skill_launch: {
              skill_name: "webpage_generate",
              kind: "webpage_request",
              webpage_request: {
                page_type: "homepage",
                style: "glassmorphism",
                prompt: "帮我做一个 AI workspace 官网",
                entry_source: "at_webpage_command",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "webpage_generate",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("部分 @网页 输入应补齐最近成功的页面偏好", async () => {
    recordMentionEntryUsage({
      kind: "builtin_command",
      entryId: "webpage_generate",
      usedAt: 1_712_345_678_900,
      slotValues: {
        page_type: "landing_page",
        style: "未来感",
        tech_stack: "原生 HTML",
        prompt: "旧要求不应复用",
        content: "旧内容不应复用",
      },
    });

    const harness = mountHook({
      input: "@网页 帮我做一个 AI 代码助手官网",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@网页 帮我做一个 AI 代码助手官网",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            webpage_skill_launch: {
              webpage_request: {
                prompt: "帮我做一个 AI 代码助手官网",
                content: "帮我做一个 AI 代码助手官网",
                page_type: "landing_page",
                style: "未来感",
                tech_stack: "原生 HTML",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          entryId: "webpage_generate",
          replayText:
            "类型:落地页 风格:未来感 技术:原生 HTML 要求:帮我做一个 AI 代码助手官网",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@PPT 应保留原始消息，并通过 presentation_skill_launch metadata 交给 Agent 调度技能", async () => {
    const harness = mountHook({
      input:
        "@PPT 类型:路演PPT 风格:极简科技 受众:投资人 页数:10 帮我做一个 AI 助手创业项目融资演示稿",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@PPT 类型:路演PPT 风格:极简科技 受众:投资人 页数:10 帮我做一个 AI 助手创业项目融资演示稿",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            presentation_skill_launch: {
              skill_name: "presentation_generate",
              kind: "presentation_request",
              presentation_request: {
                prompt: "帮我做一个 AI 助手创业项目融资演示稿",
                content:
                  "类型:路演PPT 风格:极简科技 受众:投资人 页数:10 帮我做一个 AI 助手创业项目融资演示稿",
                deck_type: "pitch_deck",
                style: "极简科技",
                audience: "投资人",
                slide_count: 10,
                entry_source: "at_presentation_command",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "presentation_generate",
          replayText:
            "类型:路演PPT 风格:极简科技 受众:投资人 页数:10 要求:帮我做一个 AI 助手创业项目融资演示稿",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@Sales 1 应继续走演示稿主链，并默认锁成 sales_deck", async () => {
    const harness = mountHook({
      input:
        "@Sales 1 风格: 极简商业 受众: 销售团队 帮我做一个企业服务产品销售演示稿",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@Sales 1 风格: 极简商业 受众: 销售团队 帮我做一个企业服务产品销售演示稿",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            presentation_skill_launch: {
              skill_name: "presentation_generate",
              kind: "presentation_request",
              presentation_request: {
                deck_type: "sales_deck",
                style: "极简商业",
                audience: "销售团队",
                prompt: "帮我做一个企业服务产品销售演示稿",
                entry_source: "at_presentation_command",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "presentation_generate",
          replayText:
            "类型:销售PPT 风格:极简商业 受众:销售团队 要求:帮我做一个企业服务产品销售演示稿",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("部分 @PPT 输入应补齐最近成功的演示偏好", async () => {
    recordMentionEntryUsage({
      kind: "builtin_command",
      entryId: "presentation_generate",
      usedAt: 1_712_345_678_900,
      slotValues: {
        deck_type: "pitch_deck",
        style: "极简科技",
        audience: "投资人",
        slide_count: "10",
        prompt: "旧要求不应复用",
        content: "旧内容不应复用",
      },
    });

    const harness = mountHook({
      input: "@PPT 帮我做一个 AI 助手创业项目融资演示稿",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@PPT 帮我做一个 AI 助手创业项目融资演示稿",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            presentation_skill_launch: {
              presentation_request: {
                prompt: "帮我做一个 AI 助手创业项目融资演示稿",
                content: "帮我做一个 AI 助手创业项目融资演示稿",
                deck_type: "pitch_deck",
                style: "极简科技",
                audience: "投资人",
                slide_count: 10,
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          entryId: "presentation_generate",
          replayText:
            "类型:路演PPT 风格:极简科技 受众:投资人 页数:10 要求:帮我做一个 AI 助手创业项目融资演示稿",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@表单 应保留原始消息，并通过 form_skill_launch metadata 交给 Agent 调度技能", async () => {
    const harness = mountHook({
      input:
        "@表单 类型:报名表单 风格:简洁专业 受众:活动嘉宾 字段数:8 帮我做一个 AI Workshop 报名表",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@表单 类型:报名表单 风格:简洁专业 受众:活动嘉宾 字段数:8 帮我做一个 AI Workshop 报名表",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            allow_model_skills: true,
            form_skill_launch: {
              skill_name: "form_generate",
              kind: "form_request",
              form_request: {
                prompt: "帮我做一个 AI Workshop 报名表",
                content:
                  "类型:报名表单 风格:简洁专业 受众:活动嘉宾 字段数:8 帮我做一个 AI Workshop 报名表",
                form_type: "registration_form",
                style: "简洁专业",
                audience: "活动嘉宾",
                field_count: 8,
                entry_source: "at_form_command",
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "form_generate",
          replayText:
            "类型:报名表单 风格:简洁专业 受众:活动嘉宾 字段数:8 要求:帮我做一个 AI Workshop 报名表",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("部分 @表单 输入应补齐最近成功的表单偏好", async () => {
    recordMentionEntryUsage({
      kind: "builtin_command",
      entryId: "form_generate",
      usedAt: 1_712_345_678_900,
      slotValues: {
        form_type: "registration_form",
        style: "简洁专业",
        audience: "活动嘉宾",
        field_count: "8",
        prompt: "旧要求不应复用",
        content: "旧内容不应复用",
      },
    });

    const harness = mountHook({
      input: "@表单 帮我做一个 AI Workshop 报名表",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@表单 帮我做一个 AI Workshop 报名表",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            form_skill_launch: {
              form_request: {
                prompt: "帮我做一个 AI Workshop 报名表",
                content: "帮我做一个 AI Workshop 报名表",
                form_type: "registration_form",
                style: "简洁专业",
                audience: "活动嘉宾",
                field_count: 8,
              },
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          entryId: "form_generate",
          replayText:
            "类型:报名表单 风格:简洁专业 受众:活动嘉宾 字段数:8 要求:帮我做一个 AI Workshop 报名表",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@代码 应保留原始消息，并继续走普通 Agent 主链", async () => {
    const harness = mountHook({
      input: "@代码 修复消息历史切换后图片卡片丢失的问题，并补一个回归测试",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@代码 修复消息历史切换后图片卡片丢失的问题，并补一个回归测试",
      );
      expect(mockSendMessage.mock.calls[0]?.[5]).toBe("react");
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            preferences: {
              task: false,
              subagent: false,
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "code_runtime",
          replayText: "修复消息历史切换后图片卡片丢失的问题，并补一个回归测试",
        }),
      ]);
      const harnessMetadata = (
        mockSendMessage.mock.calls[0]?.[8]?.requestMetadata as
          | { harness?: Record<string, unknown> }
          | undefined
      )?.harness;
      expect(harnessMetadata?.code_command).toBeUndefined();
      expect(listServiceSkillUsage()).toEqual([]);
    } finally {
      harness.unmount();
    }
  });

  it("catalog 下发的 legacy requestDefaults 应归一到普通 Agent 主链", async () => {
    saveSkillCatalog(
      {
        version: "tenant-2026-05-26",
        tenantId: "tenant-demo",
        syncedAt: "2026-05-26T12:00:00.000Z",
        groups: [
          {
            key: "engineering",
            title: "工程能力",
            summary: "租户下发的工程类快捷入口。",
            sort: 10,
            itemCount: 0,
          },
        ],
        items: [],
        entries: [
          {
            id: "command:engineering_runtime",
            kind: "command",
            title: "工程模式",
            summary: "通过统一 catalog requestDefaults 进入编程底座。",
            commandKey: "engineering_runtime",
            triggers: [{ mode: "mention", prefix: "@工程模式" }],
            binding: {
              executionKind: "agent_turn",
              requestDefaults: {
                executionStrategy: "react",
              },
            },
          },
        ],
      },
      "bootstrap_sync",
    );
    const harness = mountHook({
      input: "@工程模式 收口 workspace 发送链的硬编码分支，并补治理 guard",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@工程模式 收口 workspace 发送链的硬编码分支，并补治理 guard",
      );
      expect(mockSendMessage.mock.calls[0]?.[5]).toBe("react");
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            preferences: {
              task: false,
              subagent: false,
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "engineering_runtime",
          replayText: "收口 workspace 发送链的硬编码分支，并补治理 guard",
        }),
      ]);
      const harnessMetadata = (
        mockSendMessage.mock.calls[0]?.[8]?.requestMetadata as
          | { harness?: Record<string, unknown> }
          | undefined
      )?.harness;
      expect(harnessMetadata?.code_command).toBeUndefined();
      expect(listServiceSkillUsage()).toEqual([]);
    } finally {
      harness.unmount();
    }
  });

  it("普通输入走 current react 时仍不应被 @代码 parser 改写", async () => {
    const harness = mountHook({
      input: "继续修复消息历史切换后图片卡片丢失的问题，并补一个回归测试",
      executionStrategy: "react",
      chatToolPreferences: {
        task: false,
        subagent: false,
      },
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "继续修复消息历史切换后图片卡片丢失的问题，并补一个回归测试",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            preferences: {
              task: false,
              subagent: false,
            },
          },
        },
      });
      const harnessMetadata = (
        mockSendMessage.mock.calls[0]?.[8]?.requestMetadata as
          | { harness?: Record<string, unknown> }
          | undefined
      )?.harness;
      expect(harnessMetadata?.code_command).toBeUndefined();
      expect(harnessMetadata?.fast_response_routing).toBeUndefined();
      expect(listMentionEntryUsage()).toEqual([]);
      expect(listServiceSkillUsage()).toEqual([]);
    } finally {
      harness.unmount();
    }
  });

  it("@发布 应保留原始消息，并将 dispatch 接到现有发布工作流", async () => {
    const harness = mountHook({
      input: "@发布 平台:微信公众号后台 帮我把这篇文章整理成可直接发布的版本",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "/content_post_with_cover 平台:微信公众号后台 帮我把这篇文章整理成可直接发布的版本",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        displayContent:
          "@发布 平台:微信公众号后台 帮我把这篇文章整理成可直接发布的版本",
        requestMetadata: {
          harness: {
            browser_requirement: "required_with_user_step",
            browser_launch_url: "https://mp.weixin.qq.com/",
            publish_command: {
              prompt: "帮我把这篇文章整理成可直接发布的版本",
              content:
                "平台:微信公众号后台 帮我把这篇文章整理成可直接发布的版本",
              platform_type: "wechat_official_account",
              platform_label: "微信公众号后台",
              entry_source: "at_publish_command",
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "publish_runtime",
          replayText:
            "平台:微信公众号后台 要求:帮我把这篇文章整理成可直接发布的版本",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@Writing Partner 应保留原始消息，并复用现有内容成稿主链", async () => {
    const harness = mountHook({
      input:
        "@Writing Partner help me turn this launch brief into a stronger narrative",
      serviceSkills: [
        createGeneralServiceSkill({
          id: "content_post_with_cover",
          title: "发布工作流",
        }),
      ],
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toContain(
        "/content_post_with_cover",
      );
      expect(mockSendMessage.mock.calls[0]?.[0]).toContain("写作主稿");
      expect(mockSendMessage.mock.calls[0]?.[0]).toContain(
        "help me turn this launch brief into a stronger narrative",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        displayContent:
          "@Writing Partner help me turn this launch brief into a stronger narrative",
        requestMetadata: {
          harness: {
            publish_command: {
              prompt:
                "help me turn this launch brief into a stronger narrative",
              content:
                "help me turn this launch brief into a stronger narrative",
              entry_source: "at_writing_command",
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "writing_runtime",
          replayText:
            "要求:help me turn this launch brief into a stronger narrative",
        }),
      ]);
      expect(listServiceSkillUsage()).toEqual([
        expect.objectContaining({
          skillId: "content_post_with_cover",
          runnerType: "instant",
          launchUserInput:
            "help me turn this launch brief into a stronger narrative",
          slotValues: {
            prompt: "help me turn this launch brief into a stronger narrative",
            content: "help me turn this launch brief into a stronger narrative",
          },
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@Web Copy 应保留原始消息，并继续走同一条写作主链", async () => {
    const harness = mountHook({
      input: "@Web Copy rewrite this SaaS homepage headline and CTA",
      serviceSkills: [
        createGeneralServiceSkill({
          id: "content_post_with_cover",
          title: "发布工作流",
        }),
      ],
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toContain(
        "/content_post_with_cover",
      );
      expect(mockSendMessage.mock.calls[0]?.[0]).toContain("写作主稿");
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        displayContent: "@Web Copy rewrite this SaaS homepage headline and CTA",
        requestMetadata: {
          harness: {
            publish_command: {
              prompt: "rewrite this SaaS homepage headline and CTA",
              content: "rewrite this SaaS homepage headline and CTA",
              entry_source: "at_writing_command",
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "writing_runtime",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("部分 @Blog 1 输入应补齐最近成功的平台偏好，并保留 blog 主稿语义", async () => {
    recordMentionEntryUsage({
      kind: "builtin_command",
      entryId: "writing_runtime",
      usedAt: 1_712_345_678_900,
      slotValues: {
        prompt: "旧要求不应复用",
        content: "旧内容不应复用",
        platform_type: "wechat_official_account",
        platform_label: "微信公众号后台",
      },
    });

    const harness = mountHook({
      input: "@Blog 1 帮我写一篇 AI 浏览器产品观察文章",
      serviceSkills: [
        createGeneralServiceSkill({
          id: "content_post_with_cover",
          title: "发布工作流",
        }),
      ],
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toContain(
        "/content_post_with_cover 平台:微信公众号后台",
      );
      expect(mockSendMessage.mock.calls[0]?.[0]).toContain("Blog 文章主稿");
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        displayContent: "@Blog 1 帮我写一篇 AI 浏览器产品观察文章",
        requestMetadata: {
          harness: {
            publish_command: {
              prompt: "帮我写一篇 AI 浏览器产品观察文章",
              content: "帮我写一篇 AI 浏览器产品观察文章",
              platform_type: "wechat_official_account",
              platform_label: "微信公众号后台",
              entry_source: "at_writing_command",
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          entryId: "writing_runtime",
          replayText:
            "平台:微信公众号后台 要求:帮我写一篇 AI 浏览器产品观察文章",
        }),
      ]);
      expect(listServiceSkillUsage()).toEqual([
        expect.objectContaining({
          skillId: "content_post_with_cover",
          runnerType: "instant",
          launchUserInput: "帮我写一篇 AI 浏览器产品观察文章",
          slotValues: {
            prompt: "帮我写一篇 AI 浏览器产品观察文章",
            content: "帮我写一篇 AI 浏览器产品观察文章",
            platform_type: "wechat_official_account",
            platform_label: "微信公众号后台",
          },
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("部分 @发布 输入应补齐最近成功的平台偏好，并同步触发对应发布前置要求", async () => {
    recordMentionEntryUsage({
      kind: "builtin_command",
      entryId: "publish_runtime",
      usedAt: 1_712_345_678_900,
      slotValues: {
        prompt: "旧要求不应复用",
        content: "旧内容不应复用",
        platform_type: "wechat_official_account",
        platform_label: "微信公众号后台",
      },
    });

    const harness = mountHook({
      input: "@发布 帮我把这篇文章整理成可直接发布的版本",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "/content_post_with_cover 平台:微信公众号后台 帮我把这篇文章整理成可直接发布的版本",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        displayContent: "@发布 帮我把这篇文章整理成可直接发布的版本",
        requestMetadata: {
          harness: {
            browser_requirement: "required_with_user_step",
            browser_launch_url: "https://mp.weixin.qq.com/",
            publish_command: {
              prompt: "帮我把这篇文章整理成可直接发布的版本",
              content: "帮我把这篇文章整理成可直接发布的版本",
              platform_type: "wechat_official_account",
              platform_label: "微信公众号后台",
              entry_source: "at_publish_command",
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          entryId: "publish_runtime",
          replayText:
            "平台:微信公众号后台 要求:帮我把这篇文章整理成可直接发布的版本",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@渠道预览 应保留原始消息，并复用现有发布工作流生成预览稿", async () => {
    const harness = mountHook({
      input: "@渠道预览 平台:小红书 帮我预览这篇春日咖啡活动文案的首屏效果",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toContain(
        "/content_post_with_cover 平台:小红书",
      );
      expect(mockSendMessage.mock.calls[0]?.[0]).toContain("渠道预览稿");
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        displayContent:
          "@渠道预览 平台:小红书 帮我预览这篇春日咖啡活动文案的首屏效果",
        requestMetadata: {
          harness: {
            publish_command: {
              prompt: "帮我预览这篇春日咖啡活动文案的首屏效果",
              content: "平台:小红书 帮我预览这篇春日咖啡活动文案的首屏效果",
              platform_type: "xiaohongshu",
              platform_label: "小红书",
              intent: "preview",
              entry_source: "at_channel_preview_command",
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "channel_preview_runtime",
          replayText: "平台:小红书 要求:帮我预览这篇春日咖啡活动文案的首屏效果",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("部分 @渠道预览 输入应补齐最近成功的平台偏好", async () => {
    recordMentionEntryUsage({
      kind: "builtin_command",
      entryId: "channel_preview_runtime",
      usedAt: 1_712_345_678_900,
      slotValues: {
        prompt: "旧要求不应复用",
        content: "旧内容不应复用",
        platform_type: "xiaohongshu",
        platform_label: "小红书",
        intent: "preview",
      },
    });

    const harness = mountHook({
      input: "@渠道预览 帮我预览这篇春日咖啡活动文案的首屏效果",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toContain(
        "/content_post_with_cover 平台:小红书",
      );
      expect(mockSendMessage.mock.calls[0]?.[0]).toContain("渠道预览稿");
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        displayContent: "@渠道预览 帮我预览这篇春日咖啡活动文案的首屏效果",
        requestMetadata: {
          harness: {
            publish_command: {
              prompt: "帮我预览这篇春日咖啡活动文案的首屏效果",
              content: "帮我预览这篇春日咖啡活动文案的首屏效果",
              platform_type: "xiaohongshu",
              platform_label: "小红书",
              intent: "preview",
              entry_source: "at_channel_preview_command",
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          entryId: "channel_preview_runtime",
          replayText: "平台:小红书 要求:帮我预览这篇春日咖啡活动文案的首屏效果",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@Instagram Preview 应保留原始消息，并复用同一条渠道预览主链", async () => {
    const harness = mountHook({
      input:
        "@Instagram Preview turn this launch brief into a first-screen draft",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toContain(
        "/content_post_with_cover 平台:Instagram",
      );
      expect(mockSendMessage.mock.calls[0]?.[0]).toContain("渠道预览稿");
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        displayContent:
          "@Instagram Preview turn this launch brief into a first-screen draft",
        requestMetadata: {
          harness: {
            publish_command: {
              prompt: "turn this launch brief into a first-screen draft",
              content: "turn this launch brief into a first-screen draft",
              platform_type: "instagram",
              platform_label: "Instagram",
              intent: "preview",
              entry_source: "at_channel_preview_command",
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "channel_preview_runtime",
          replayText:
            "平台:Instagram 要求:turn this launch brief into a first-screen draft",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@上传 应保留原始消息，并复用现有发布工作流生成上传稿", async () => {
    const harness = mountHook({
      input:
        "@上传 平台:微信公众号后台 帮我把这篇春日咖啡活动文案整理成可直接上传的版本",
      serviceSkills: [
        createGeneralServiceSkill({
          id: "content_post_with_cover",
          title: "发布工作流",
        }),
      ],
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toContain(
        "/content_post_with_cover 平台:微信公众号后台",
      );
      expect(mockSendMessage.mock.calls[0]?.[0]).toContain("上传稿");
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        displayContent:
          "@上传 平台:微信公众号后台 帮我把这篇春日咖啡活动文案整理成可直接上传的版本",
        requestMetadata: {
          harness: {
            browser_requirement: "required_with_user_step",
            browser_launch_url: "https://mp.weixin.qq.com/",
            publish_command: {
              prompt: "帮我把这篇春日咖啡活动文案整理成可直接上传的版本",
              content:
                "平台:微信公众号后台 帮我把这篇春日咖啡活动文案整理成可直接上传的版本",
              platform_type: "wechat_official_account",
              platform_label: "微信公众号后台",
              intent: "upload",
              entry_source: "at_upload_command",
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "upload_runtime",
          replayText:
            "平台:微信公众号后台 要求:帮我把这篇春日咖啡活动文案整理成可直接上传的版本",
        }),
      ]);
      expect(listServiceSkillUsage()).toEqual([
        expect.objectContaining({
          skillId: "content_post_with_cover",
          runnerType: "instant",
          launchUserInput: "帮我把这篇春日咖啡活动文案整理成可直接上传的版本",
          slotValues: {
            prompt: "帮我把这篇春日咖啡活动文案整理成可直接上传的版本",
            content:
              "平台:微信公众号后台 帮我把这篇春日咖啡活动文案整理成可直接上传的版本",
            platform_type: "wechat_official_account",
            platform_label: "微信公众号后台",
            intent: "upload",
          },
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("部分 @上传 输入应补齐最近成功的平台偏好，并保留上传主链语义", async () => {
    recordMentionEntryUsage({
      kind: "builtin_command",
      entryId: "upload_runtime",
      usedAt: 1_712_345_678_900,
      slotValues: {
        prompt: "旧要求不应复用",
        content: "旧内容不应复用",
        platform_type: "wechat_official_account",
        platform_label: "微信公众号后台",
        intent: "upload",
      },
    });

    const harness = mountHook({
      input: "@上传 帮我把这篇春日咖啡活动文案整理成可直接上传的版本",
      serviceSkills: [
        createGeneralServiceSkill({
          id: "content_post_with_cover",
          title: "发布工作流",
        }),
      ],
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toContain(
        "/content_post_with_cover 平台:微信公众号后台",
      );
      expect(mockSendMessage.mock.calls[0]?.[0]).toContain("上传稿");
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        displayContent:
          "@上传 帮我把这篇春日咖啡活动文案整理成可直接上传的版本",
        requestMetadata: {
          harness: {
            browser_requirement: "required_with_user_step",
            browser_launch_url: "https://mp.weixin.qq.com/",
            publish_command: {
              prompt: "帮我把这篇春日咖啡活动文案整理成可直接上传的版本",
              content: "帮我把这篇春日咖啡活动文案整理成可直接上传的版本",
              platform_type: "wechat_official_account",
              platform_label: "微信公众号后台",
              intent: "upload",
              entry_source: "at_upload_command",
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          entryId: "upload_runtime",
          replayText:
            "平台:微信公众号后台 要求:帮我把这篇春日咖啡活动文案整理成可直接上传的版本",
        }),
      ]);
      expect(listServiceSkillUsage()).toEqual([
        expect.objectContaining({
          skillId: "content_post_with_cover",
          runnerType: "instant",
          slotValues: {
            prompt: "帮我把这篇春日咖啡活动文案整理成可直接上传的版本",
            content: "帮我把这篇春日咖啡活动文案整理成可直接上传的版本",
            platform_type: "wechat_official_account",
            platform_label: "微信公众号后台",
            intent: "upload",
          },
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@YouTube Publish 应保留原始消息，并复用同一条发布主链与浏览器前置要求", async () => {
    const harness = mountHook({
      input:
        "@YouTube Publish refine this launch draft into a ready-to-post script",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "/content_post_with_cover 平台:YouTube refine this launch draft into a ready-to-post script",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        displayContent:
          "@YouTube Publish refine this launch draft into a ready-to-post script",
        requestMetadata: {
          harness: {
            browser_requirement: "required_with_user_step",
            browser_launch_url: "https://studio.youtube.com/",
            publish_command: {
              prompt: "refine this launch draft into a ready-to-post script",
              content: "refine this launch draft into a ready-to-post script",
              platform_type: "youtube",
              platform_label: "YouTube",
              entry_source: "at_publish_command",
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "publish_runtime",
          replayText:
            "平台:YouTube 要求:refine this launch draft into a ready-to-post script",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@配音 应保留原始消息，并通过本地 service scene launch 注入配音 prompt", async () => {
    mockUseGlobalMediaGenerationDefaults.mockReturnValue({
      voice: {
        preferredProviderId: "openai-tts",
        preferredModelId: "gpt-4o-mini-tts",
        allowFallback: false,
      },
    });
    const harness = mountHook({
      input: "@配音 目标语言: 英文 风格: 科技感 给这个新品视频做一版发布配音稿",
      serviceSkills: [createCloudSceneSkill()],
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toContain(
        "[技能任务] 视频配音",
      );
      expect(mockSendMessage.mock.calls[0]?.[0]).toContain(
        "[补充要求] 给这个新品视频做一版发布配音稿",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        displayContent:
          "@配音 目标语言: 英文 风格: 科技感 给这个新品视频做一版发布配音稿",
        requestMetadata: {
          harness: {
            service_scene_launch: {
              kind: "local_service_skill",
              service_scene_run: expect.objectContaining({
                scene_key: "voice_runtime",
                command_prefix: "@配音",
                skill_id: "cloud-video-dubbing",
                skill_title: "视频配音",
                user_input: "给这个新品视频做一版发布配音稿",
                entry_source: "at_voice_command",
                modality_contract_key: "voice_generation",
                modality: "audio",
                required_capabilities: ["text_generation", "voice_generation"],
                routing_slot: "voice_generation_model",
                runtime_contract: expect.objectContaining({
                  contract_key: "voice_generation",
                  routing_slot: "voice_generation_model",
                  route_execution_status: "executable",
                }),
                project_id: "project-1",
                target_language: "英文",
                voice_style: "科技感",
                preferred_provider_id: "openai-tts",
                preferred_model_id: "gpt-4o-mini-tts",
                allow_fallback: false,
              }),
            },
          },
        },
      });
      expect(
        mockSendMessage.mock.calls[0]?.[8]?.providerOverride,
      ).toBeUndefined();
      expect(mockSendMessage.mock.calls[0]?.[8]?.modelOverride).toBeUndefined();
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "voice_runtime",
          replayText:
            "目标语言:英文 风格:科技感 给这个新品视频做一版发布配音稿",
        }),
      ]);
      expect(listServiceSkillUsage()).toEqual([
        expect.objectContaining({
          skillId: "cloud-video-dubbing",
          runnerType: "instant",
          launchUserInput: "给这个新品视频做一版发布配音稿",
          slotValues: {
            user_input: "给这个新品视频做一版发布配音稿",
            target_language: "英文",
            voice_style: "科技感",
          },
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@配音 当只声明首选 provider 且未显式选模型时，不应误注入发送覆盖", async () => {
    mockUseGlobalMediaGenerationDefaults.mockReturnValue({
      voice: {
        preferredProviderId: "openai-tts",
        allowFallback: true,
      },
    });
    const harness = mountHook({
      input: "@配音 给这个新品视频做一版发布配音稿",
      serviceSkills: [createCloudSceneSkill()],
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            service_scene_launch: {
              service_scene_run: expect.objectContaining({
                preferred_provider_id: "openai-tts",
              }),
            },
          },
        },
      });
      expect(
        mockSendMessage.mock.calls[0]?.[8]?.providerOverride,
      ).toBeUndefined();
      expect(mockSendMessage.mock.calls[0]?.[8]?.modelOverride).toBeUndefined();
    } finally {
      harness.unmount();
    }
  });

  it("@Growth Expert 应保留原始消息，并通过本地 service scene launch 注入增长跟踪 prompt", async () => {
    const harness = mountHook({
      input:
        "@Growth Expert 平台:X 账号:@openai,@anthropic 回报频率:每天 09:00 告警阈值:互动率低于 2% 帮我先做一版增长跟踪策略",
      serviceSkills: [createGrowthTrackingSkill()],
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toContain(
        "[技能任务] 账号增长跟踪",
      );
      expect(mockSendMessage.mock.calls[0]?.[0]).toContain(
        "[任务形态] 持续跟踪",
      );
      expect(mockSendMessage.mock.calls[0]?.[0]).toContain(
        "[补充要求] 帮我先做一版增长跟踪策略",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        displayContent:
          "@Growth Expert 平台:X 账号:@openai,@anthropic 回报频率:每天 09:00 告警阈值:互动率低于 2% 帮我先做一版增长跟踪策略",
        requestMetadata: {
          harness: {
            service_scene_launch: {
              kind: "local_service_skill",
              service_scene_run: expect.objectContaining({
                scene_key: "growth_runtime",
                command_prefix: "@Growth Expert",
                skill_id: "account-performance-tracking",
                skill_title: "账号增长跟踪",
                user_input: "帮我先做一版增长跟踪策略",
                entry_source: "at_growth_command",
                project_id: "project-1",
                platform: "x",
                account_list: "@openai, @anthropic",
                report_cadence: "每天 09:00",
                alert_threshold: "互动率低于 2%",
              }),
            },
          },
        },
      });
      expect(listMentionEntryUsage()).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            kind: "builtin_command",
            entryId: "growth_runtime",
            slotValues: {
              user_input: "帮我先做一版增长跟踪策略",
              platform: "x",
              account_list: "@openai, @anthropic",
              report_cadence: "每天 09:00",
              alert_threshold: "互动率低于 2%",
            },
          }),
        ]),
      );
      expect(listServiceSkillUsage()).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            skillId: "account-performance-tracking",
            runnerType: "managed",
            launchUserInput: "帮我先做一版增长跟踪策略",
            slotValues: {
              user_input: "帮我先做一版增长跟踪策略",
              platform: "x",
              account_list: "@openai, @anthropic",
              report_cadence: "每天 09:00",
              alert_threshold: "互动率低于 2%",
            },
          }),
        ]),
      );
    } finally {
      harness.unmount();
    }
  });

  it("@搜索 的 service skill usage 应优先跟随当前 command catalog 绑定，而不是 seeded 默认映射", async () => {
    const tenantResearchSkill = createGeneralServiceSkill({
      id: "tenant-research",
      skillKey: "tenant-research",
      title: "租户搜索",
    });
    saveSkillCatalog(
      {
        version: "tenant-2026-04-15",
        tenantId: "tenant-demo",
        syncedAt: "2026-04-15T12:00:00.000Z",
        groups: [
          {
            key: "general",
            title: "通用技能",
            summary: "租户下发目录",
            sort: 90,
            itemCount: 1,
          },
        ],
        items: [
          {
            ...tenantResearchSkill,
            groupKey: "general",
            execution: {
              kind: "agent_turn",
            },
          },
        ],
        entries: [
          {
            id: "command:research",
            kind: "command",
            title: "租户搜索",
            summary: "把 @搜索 绑定到租户下发的搜索技能。",
            commandKey: "research",
            triggers: [{ mode: "mention", prefix: "@搜索" }],
            binding: {
              skillId: "tenant-research",
              executionKind: "agent_turn",
            },
          },
        ],
      },
      "bootstrap_sync",
    );
    const harness = mountHook({
      input: "@搜索 query: AI Agent 聚焦 2026 年新发布",
      serviceSkills: [tenantResearchSkill],
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(listServiceSkillUsage()).toEqual([
        expect.objectContaining({
          skillId: "tenant-research",
          runnerType: "instant",
          slotValues: expect.objectContaining({
            query: "AI Agent 聚焦 2026 年新发布",
          }),
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@浏览器 应保留原始消息，并显式切到真实浏览器执行主链", async () => {
    const harness = mountHook({
      input: "@浏览器 打开 https://news.baidu.com 并提炼页面主要内容",
      chatToolPreferences: {
        task: false,
        subagent: false,
      },
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@浏览器 打开 https://news.baidu.com 并提炼页面主要内容",
      );
      expect(mockSendMessage.mock.calls[0]?.[2]).toBe(false);
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            browser_requirement: "required",
            browser_launch_url: "https://news.baidu.com",
            browser_user_step_required: false,
          },
        },
      });
      const sentHarness = mockSendMessage.mock.calls[0]?.[8]?.requestMetadata
        ?.harness as Record<string, unknown>;
      expect(sentHarness.browser_assist).toBeUndefined();
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "browser_runtime",
          replayText: "https://news.baidu.com 并提炼页面主要内容",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@浏览器 不应把 Renderer 可见 session 写入 harness metadata", async () => {
    const harness = mountHook({
      input: "@浏览器 打开 https://example.com/current 并总结当前页面",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[2]).toBe(false);
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            browser_requirement: "required",
            browser_launch_url: "https://example.com/current",
            browser_user_step_required: false,
          },
        },
      });
      const sentHarness = mockSendMessage.mock.calls[0]?.[8]?.requestMetadata
        ?.harness as Record<string, unknown>;
      expect(sentHarness.browser_assist).toBeUndefined();
    } finally {
      harness.unmount();
    }
  });

  it("@Browser Agent 应保留原始消息，并显式切到同一条浏览器执行主链", async () => {
    const harness = mountHook({
      input: "@Browser Agent open https://openai.com/pricing and compare plans",
      chatToolPreferences: {
        task: false,
        subagent: false,
      },
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@Browser Agent open https://openai.com/pricing and compare plans",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            browser_requirement: "required",
            browser_launch_url: "https://openai.com/pricing",
            browser_user_step_required: false,
          },
        },
      });
      const sentHarness = mockSendMessage.mock.calls[0]?.[8]?.requestMetadata
        ?.harness as Record<string, unknown>;
      expect(sentHarness.browser_assist).toBeUndefined();
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "browser_runtime",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("@Mini Tester 应保留原始消息，并显式切到同一条浏览器执行主链", async () => {
    const harness = mountHook({
      input: "@Mini Tester open https://example.com and verify the CTA flow",
      chatToolPreferences: {
        task: false,
        subagent: false,
      },
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toBe(
        "@Mini Tester open https://example.com and verify the CTA flow",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            browser_requirement: "required",
            browser_launch_url: "https://example.com",
            browser_user_step_required: false,
          },
        },
      });
      const sentHarness = mockSendMessage.mock.calls[0]?.[8]?.requestMetadata
        ?.harness as Record<string, unknown>;
      expect(sentHarness.browser_assist).toBeUndefined();
      expect(listMentionEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "builtin_command",
          entryId: "browser_runtime",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("/scene-key 命中运行时场景时应注入统一 scene launch metadata 后继续走 Agent 发送主线", async () => {
    mockGetSkillCatalog.mockResolvedValueOnce({
      entries: [
        {
          id: "scene:campaign-launch",
        },
      ],
    });
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
    const harness = mountHook({
      input: "/campaign-launch 帮我做一版新品活动启动方案",
      serviceSkills: [createCloudSceneSkill()],
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[0]).toContain(
        "[技能任务] 视频配音",
      );
      expect(mockSendMessage.mock.calls[0]?.[0]).toContain(
        "[补充要求] 帮我做一版新品活动启动方案",
      );
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        displayContent: "/campaign-launch 帮我做一版新品活动启动方案",
        capabilityRoute: {
          kind: "runtime_scene",
          sceneKey: "campaign-launch",
          commandPrefix: "/campaign-launch",
        },
        requestMetadata: {
          harness: {
            service_scene_launch: {
              kind: "local_service_skill",
              service_scene_run: expect.objectContaining({
                scene_key: "campaign-launch",
                skill_id: "cloud-video-dubbing",
                skill_title: "视频配音",
                user_input: "帮我做一版新品活动启动方案",
                project_id: "project-1",
              }),
            },
          },
        },
      });
      expect(listSlashEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "scene",
          entryId: "campaign-launch",
          replayText: "帮我做一版新品活动启动方案",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("/runtime scene 应把 legacy requestDefaults 归一到普通 Agent 主链", async () => {
    mockGetSkillCatalog.mockResolvedValueOnce({
      entries: [
        {
          id: "scene:engineering-mode",
        },
      ],
    });
    mockListSkillCatalogSceneEntries.mockReturnValueOnce([
      {
        id: "scene:engineering-mode",
        kind: "scene",
        title: "工程模式",
        summary: "通过统一 runtime scene 进入编程底座。",
        sceneKey: "engineering-mode",
        commandPrefix: "/engineering-mode",
        linkedSkillId: "cloud-video-dubbing",
        executionKind: "agent_turn",
        requestDefaults: {
          executionStrategy: "react",
        },
      },
    ]);
    const harness = mountHook({
      input:
        "/engineering-mode 收口 workspace 发送链的硬编码分支，并补治理 guard",
      serviceSkills: [createCloudSceneSkill()],
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[5]).toBe("react");
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        displayContent:
          "/engineering-mode 收口 workspace 发送链的硬编码分支，并补治理 guard",
        capabilityRoute: {
          kind: "runtime_scene",
          sceneKey: "engineering-mode",
          commandPrefix: "/engineering-mode",
        },
        requestMetadata: {
          harness: {
            preferences: {
              task: false,
              subagent: false,
            },
            service_scene_launch: {
              kind: "local_service_skill",
              service_scene_run: expect.objectContaining({
                scene_key: "engineering-mode",
                skill_id: "cloud-video-dubbing",
                user_input: "收口 workspace 发送链的硬编码分支，并补治理 guard",
              }),
            },
          },
        },
      });
      const harnessMetadata = (
        mockSendMessage.mock.calls[0]?.[8]?.requestMetadata as
          | { harness?: Record<string, unknown> }
          | undefined
      )?.harness;
      expect(harnessMetadata?.code_command).toBeUndefined();
      expect(listSlashEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "scene",
          entryId: "engineering-mode",
          replayText: "收口 workspace 发送链的硬编码分支，并补治理 guard",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("service scene 应从 skill 声明推导 local_service_skill metadata 并继续发送", async () => {
    mockGetSkillCatalog.mockResolvedValueOnce({
      entries: [
        {
          id: "scene:x-article-export",
        },
      ],
    });
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
        executionKind: "agent_turn",
      },
    ]);
    const harness = mountHook({
      input:
        "/x文章转存 https://x.com/GoogleCloudTech/article/2033953579824758855",
      serviceSkills: [createXArticleExportSkill()],
      contentId: "content-1",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      expect(mockSendMessage.mock.calls[0]?.[8]).toMatchObject({
        requestMetadata: {
          harness: {
            service_scene_launch: {
              kind: "local_service_skill",
              service_scene_run: expect.objectContaining({
                scene_key: "x-article-export",
                skill_id: "x-article-export",
                skill_title: "X 文章转存",
                user_input:
                  "https://x.com/GoogleCloudTech/article/2033953579824758855",
                project_id: "project-1",
                slot_values: {
                  article_url:
                    "https://x.com/GoogleCloudTech/article/2033953579824758855",
                },
              }),
            },
          },
        },
      });
      expect(listSlashEntryUsage()).toEqual([
        expect.objectContaining({
          kind: "scene",
          entryId: "x-article-export",
          replayText:
            "https://x.com/GoogleCloudTech/article/2033953579824758855",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("service scene 缺少当前项目时应拦截发送并提示先选择项目", async () => {
    mockGetSkillCatalog.mockResolvedValueOnce({
      entries: [
        {
          id: "scene:x-article-export",
        },
      ],
    });
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
        executionKind: "agent_turn",
      },
    ]);
    const harness = mountHook({
      input:
        "/x文章转存 https://x.com/GoogleCloudTech/article/2033953579824758855",
      serviceSkills: [createXArticleExportSkill()],
      projectId: null,
      contentId: null,
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(false);
      });

      expect(mockGetOrCreateDefaultProject).not.toHaveBeenCalled();
      expect(mockSendMessage).not.toHaveBeenCalled();
      expect(mockOpenRuntimeSceneGate).toHaveBeenCalledTimes(1);
      expect(mockOpenRuntimeSceneGate).toHaveBeenCalledWith(
        expect.objectContaining({
          sceneKey: "x-article-export",
          fields: [
            expect.objectContaining({
              kind: "project",
              key: "project_id",
            }),
          ],
        }),
      );
    } finally {
      harness.unmount();
    }
  });

  it("service scene 缺少必填 slot 时应打开 scene gate，而不是直接发送", async () => {
    mockGetSkillCatalog.mockResolvedValueOnce({
      entries: [
        {
          id: "scene:x-article-export",
        },
      ],
    });
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
        executionKind: "agent_turn",
      },
    ]);
    const harness = mountHook({
      input: "/x文章转存",
      serviceSkills: [createXArticleExportSkill()],
      projectId: "project-1",
      contentId: null,
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(false);
      });

      expect(mockSendMessage).not.toHaveBeenCalled();
      expect(mockOpenRuntimeSceneGate).toHaveBeenCalledTimes(1);
      expect(mockOpenRuntimeSceneGate).toHaveBeenCalledWith(
        expect.objectContaining({
          sceneKey: "x-article-export",
          fields: [
            expect.objectContaining({
              kind: "slot",
              key: "article_url",
              label: "X 文章链接",
            }),
          ],
        }),
      );
    } finally {
      harness.unmount();
    }
  });

  it("已携带 service_scene_launch metadata 时不应再被前端二次路由或浏览器预热", async () => {
    const harness = mountHook({
      input: "请帮我使用 GitHub 查一下 AI Agent 项目",
      serviceSkills: [createGithubSiteSkill()],
      resolveSendBoundary: (({ sourceText }) => ({
        sourceText,
        browserRequirementMatch: {
          requirement: "required",
          reason: "当前任务需要真实浏览器页面",
          launchUrl: "https://github.com",
          platformLabel: "GitHub",
        },
        shouldConsumePendingGeneralWorkbenchInitialPrompt: false,
        shouldDismissGeneralWorkbenchEntryPrompt: false,
      })) as HookProps["resolveSendBoundary"],
    });

    try {
      await act(async () => {
        const started = await harness
          .getValue()
          .handleSend([], false, false, undefined, undefined, undefined, {
            requestMetadata: {
              harness: {
                service_scene_launch: {
                  kind: "local_service_skill",
                  service_scene_run: {
                    skill_id: "github-repo-radar",
                    scene_key: "github-research",
                    user_input: "请帮我使用 GitHub 查一下 AI Agent 项目",
                  },
                },
              },
            },
          });
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      const args = mockSendMessage.mock.calls[0] as Parameters<
        HookProps["sendMessage"]
      >;
      expect(args?.[0]).toBe("请帮我使用 GitHub 查一下 AI Agent 项目");
      expect(args?.[8]).toMatchObject({
        requestMetadata: {
          harness: expect.objectContaining({
            service_scene_launch: expect.objectContaining({
              kind: "local_service_skill",
            }),
          }),
        },
      });
      const sentHarness = args?.[8]?.requestMetadata?.harness as Record<
        string,
        unknown
      >;
      expect(sentHarness.browser_assist).toBeUndefined();
    } finally {
      harness.unmount();
    }
  });

  it("已有活动上下文快照时不应等待正文加载后才发送", async () => {
    const slowPrepareActiveContextPrompt = vi.fn().mockImplementation(
      () =>
        new Promise<string>((resolve) => {
          setTimeout(() => resolve("[生效上下文]\n1. [素材] 品牌手册"), 50);
        }),
    );
    const harness = mountHook({
      contextWorkspace: {
        enabled: true,
        activeContextPrompt: "[生效上下文]\n1. [素材] 品牌手册",
        prepareActiveContextPrompt: slowPrepareActiveContextPrompt,
      },
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      const args = mockSendMessage.mock.calls[0] as Parameters<
        HookProps["sendMessage"]
      >;
      expect(args?.[0]).toContain("[生效上下文]\n1. [素材] 品牌手册");
      expect(slowPrepareActiveContextPrompt).toHaveBeenCalledTimes(1);
    } finally {
      harness.unmount();
    }
  });

  it("应保留 workspace 级 request metadata，并与 harness 元数据合并", async () => {
    const harness = mountHook({
      isThemeWorkbench: true,
      mappedTheme: "general",
      contentId: "content-service-skill-1",
      workspaceRequestMetadataBase: {
        artifact: {
          artifact_mode: "draft",
          artifact_kind: "analysis",
        },
      },
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      const args = mockSendMessage.mock.calls[0] as Parameters<
        HookProps["sendMessage"]
      >;
      expect(args?.[8]).toMatchObject({
        requestMetadata: {
          artifact: {
            artifact_mode: "draft",
            artifact_kind: "analysis",
          },
          harness: expect.objectContaining({
            theme: "general",
            session_mode: "general_workbench",
            content_id: "content-service-skill-1",
          }),
        },
      });
    } finally {
      harness.unmount();
    }
  });

  it("应把 workspace response language 偏好写入 request metadata", async () => {
    const harness = mountHook({
      agentResponseLanguage: "en-US",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      const args = mockSendMessage.mock.calls[0] as Parameters<
        HookProps["sendMessage"]
      >;
      expect(args?.[8]).toMatchObject({
        requestMetadata: {
          harness: expect.objectContaining({
            agent_response_language: "en-US",
          }),
        },
      });
    } finally {
      harness.unmount();
    }
  });

  it("聊天主路径不应再把 accessMode 写入 harness request metadata", async () => {
    const harness = mountHook({
      accessMode: "full-access",
    });

    try {
      await act(async () => {
        const started = await harness.getValue().handleSend();
        expect(started).toBe(true);
      });

      expect(mockSendMessage).toHaveBeenCalledTimes(1);
      const args = mockSendMessage.mock.calls[0] as Parameters<
        HookProps["sendMessage"]
      >;
      const harnessMetadata = ((args?.[8]?.requestMetadata?.harness as Record<
        string,
        unknown
      >) || {}) as Record<string, unknown>;
      expect(harnessMetadata).not.toHaveProperty("access_mode");
    } finally {
      harness.unmount();
    }
  });
});
