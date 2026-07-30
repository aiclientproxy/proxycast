import { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";
import { buildModelReasoningPolicy } from "@/lib/model/modelReasoningPolicy";
import {
  cleanupMountedModelSelectors,
  clickBodyButtonByText,
  clickModelSelectorTrigger,
  clickNoProviderGuideDismissButton,
  createModelMetadata,
  createReasoningModelMetadata,
  createTextOnlyModelMetadata,
  getBodyText,
  renderModelSelector,
} from "./ModelSelector.testFixtures";

const {
  mockUseConfiguredProviders,
  mockUseProviderModels,
  mockFilterModelsByTheme,
} = vi.hoisted(() => ({
  mockUseConfiguredProviders: vi.fn(),
  mockUseProviderModels: vi.fn(),
  mockFilterModelsByTheme: vi.fn(),
}));

vi.mock("@/hooks/useConfiguredProviders", () => ({
  useConfiguredProviders: (...args: unknown[]) =>
    mockUseConfiguredProviders(...args),
  findConfiguredProviderBySelection: (
    providers: Array<{ key: string; providerId?: string }>,
    selection?: string | null,
  ) => {
    const normalizedSelection = (selection || "").trim().toLowerCase();
    const keyMatch =
      providers.find(
        (provider) => provider.key.trim().toLowerCase() === normalizedSelection,
      ) ?? null;
    const providerIdMatch =
      providers.find(
        (provider) =>
          (provider.providerId || "").trim().toLowerCase() ===
          normalizedSelection,
      ) ?? null;

    if (keyMatch && providerIdMatch && keyMatch !== providerIdMatch) {
      if (!keyMatch.providerId && providerIdMatch.providerId) {
        return providerIdMatch;
      }
    }

    return keyMatch ?? providerIdMatch ?? null;
  },
}));

vi.mock("@/hooks/useProviderModels", () => ({
  useProviderModels: (...args: unknown[]) => mockUseProviderModels(...args),
}));

vi.mock("@/components/agent/chat/utils/modelThemePolicy", () => ({
  filterModelsByTheme: (...args: unknown[]) => mockFilterModelsByTheme(...args),
}));

beforeEach(async () => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;

  await changeLimeLocale("zh-CN");

  vi.clearAllMocks();
  window.localStorage.clear();

  mockUseConfiguredProviders.mockReturnValue({
    providers: [
      {
        key: "custom-local-cli",
        label: "Local CLI Custom",
        registryId: "custom-local-cli",
        fallbackRegistryId: "codex",
        type: "codex",
        providerId: "custom-local-cli",
        apiHost: "https://api.openai.com/v1",
      },
    ],
    loading: false,
  });

  mockUseProviderModels.mockReturnValue({
    modelIds: ["gpt-5.3-codex", "gpt-5.2-codex"],
    models: [
      createReasoningModelMetadata("gpt-5.3-codex"),
      createTextOnlyModelMetadata("gpt-5.2-codex"),
    ],
    loading: false,
    error: null,
  });

  mockFilterModelsByTheme.mockImplementation((_theme, models) => ({
    models,
    usedFallback: false,
    filteredOutCount: 0,
    policyName: "none",
  }));
});

afterEach(async () => {
  cleanupMountedModelSelectors();
  window.localStorage.clear();
  await changeLimeLocale("zh-CN");
});

describe("ModelSelector", () => {
  it("应暴露稳定的禁用触发器供工作区验证", () => {
    const { container } = renderModelSelector({
      compactTrigger: true,
      disabled: true,
    });

    const trigger = container.querySelector(
      '[data-testid="model-selector"]',
    ) as HTMLButtonElement | null;
    expect(trigger).toBeTruthy();
    expect(trigger?.disabled).toBe(true);
  });

  it("Agnes 目录只应允许切换 RuntimeCore 可执行模型", () => {
    const setModel = vi.fn();
    mockUseProviderModels.mockReturnValue({
      modelIds: ["agnes-2.0-flash", "agnes-2.5-flash", "agnes-video-v2.0"],
      models: [
        createModelMetadata("agnes-2.0-flash", {
          capability_provenance: "canonical",
        }),
        createModelMetadata("agnes-2.5-flash", {
          capability_provenance: "canonical",
        }),
        createModelMetadata("agnes-video-v2.0", {
          capability_provenance: "inferred_hint",
        }),
      ],
      loading: false,
      error: null,
    });

    const { container } = renderModelSelector({
      model: "agnes-2.0-flash",
      setModel,
    });

    clickModelSelectorTrigger(container);

    expect(getBodyText()).toContain("agnes-2.5-flash");
    expect(getBodyText()).not.toContain("agnes-video-v2.0");

    clickBodyButtonByText("agnes-2.5-flash");

    expect(setModel).toHaveBeenCalledWith("agnes-2.5-flash");
  });

  it("只有模型没有 provider 时不应展示成 Lime Hub 的已选模型", () => {
    const { container } = renderModelSelector({
      providerType: "",
      model: "gpt-5.5",
    });

    expect(container.textContent).toContain("选择模型");
    expect(container.textContent).not.toContain("gpt-5.5");
  });

  it("后端回填原始 providerId 时，应解析到真实受管 Provider 读取模型", () => {
    mockUseConfiguredProviders.mockReturnValue({
      providers: [
        {
          key: "openai",
          label: "OpenAI",
          registryId: "openai",
          type: "openai",
        },
        {
          key: "openai_api_key",
          label: "OpenAI API Key",
          registryId: "openai",
          fallbackRegistryId: "openai",
          type: "openai",
          providerId: "openai",
          apiHost: "https://api.openai.com/v1",
        },
      ],
      loading: false,
    });

    renderModelSelector({
      providerType: "openai",
    });

    expect(mockUseProviderModels).toHaveBeenCalledWith(
      expect.objectContaining({
        key: "openai_api_key",
        providerId: "openai",
      }),
      expect.objectContaining({
        returnFullMetadata: true,
        autoLoad: true,
        liveFetchOnly: true,
        hasApiKey: true,
      }),
    );
  });

  it("禁用后台预加载时，关闭状态下应延后加载模型选择数据", () => {
    renderModelSelector({
      backgroundPreload: "disabled",
    });

    expect(mockUseConfiguredProviders).toHaveBeenCalledWith({
      autoLoad: false,
    });
    expect(mockUseProviderModels).toHaveBeenCalledWith(
      expect.objectContaining({ key: "custom-local-cli" }),
      expect.objectContaining({
        returnFullMetadata: true,
        autoLoad: false,
        liveFetchOnly: true,
        hasApiKey: true,
      }),
    );
  });

  it("禁用后台预加载且模型元数据未加载时，不应清空已选 reasoning effort", () => {
    const setReasoningEffort = vi.fn();
    mockUseProviderModels.mockReturnValue({
      modelIds: [],
      models: [],
      loading: false,
      error: null,
    });

    renderModelSelector({
      backgroundPreload: "disabled",
      providerType: "custom-local-cli",
      model: "gpt-5.3-codex",
      reasoningEffort: "medium",
      setReasoningEffort,
    });

    expect(mockUseProviderModels).toHaveBeenCalledWith(
      expect.objectContaining({ key: "custom-local-cli" }),
      expect.objectContaining({
        returnFullMetadata: true,
        autoLoad: false,
      }),
    );
    expect(setReasoningEffort).not.toHaveBeenCalled();
  });

  it("Lime Hub 已有声明模型但无凭证时不应强制实时目录", () => {
    mockUseConfiguredProviders.mockReturnValue({
      providers: [
        {
          key: "lime-hub",
          label: "Lime Hub",
          registryId: "lime-hub",
          type: "openai",
          providerId: "lime-hub",
          apiHost: "https://llm.limeai.run",
          models: ["gpt-5.2-pro"],
          hasApiKey: false,
        },
      ],
      loading: false,
    });

    renderModelSelector({
      providerType: "lime-hub",
      model: "gpt-5.2-pro",
    });

    expect(mockUseProviderModels).toHaveBeenCalledWith(
      expect.objectContaining({ key: "lime-hub" }),
      expect.objectContaining({
        returnFullMetadata: true,
        autoLoad: true,
        liveFetchOnly: false,
        hasApiKey: false,
      }),
    );
  });

  it("默认后台预加载开启时，应在未展开选择器前纠正失效持久化模型", async () => {
    const setModel = vi.fn();

    mockUseProviderModels.mockReturnValue({
      modelIds: ["gpt-5.2-codex", "gpt-5.1-codex-mini"],
      models: [
        createModelMetadata("gpt-5.2-codex", {
          capabilities: {
            vision: true,
            tools: true,
            streaming: true,
            json_mode: true,
            function_calling: true,
            reasoning: true,
          },
        }),
        createModelMetadata("gpt-5.1-codex-mini", {
          capabilities: {
            vision: false,
            tools: true,
            streaming: true,
            json_mode: true,
            function_calling: true,
            reasoning: false,
          },
        }),
      ],
      loading: false,
      error: null,
    });

    renderModelSelector({
      model: "gpt-5.9-codex-preview",
      setModel,
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(mockUseProviderModels).toHaveBeenCalledWith(
      expect.objectContaining({ key: "custom-local-cli" }),
      expect.objectContaining({
        returnFullMetadata: true,
        autoLoad: true,
      }),
    );
    expect(setModel).toHaveBeenCalledWith("gpt-5.2-codex");
  });

  it("首次自动选择 provider 时应跳过需要登录的 Lime Hub 提示", async () => {
    const setProviderType = vi.fn();
    mockUseConfiguredProviders.mockReturnValue({
      providers: [
        {
          key: "lime-hub",
          label: "Lime 云端",
          registryId: "lime-hub",
          type: "openai",
          providerId: "lime-hub",
          apiHost: "https://llm.limeai.run",
          models: [],
          authStatus: "login_required",
        },
        {
          key: "deepseek",
          label: "DeepSeek",
          registryId: "deepseek",
          type: "openai",
          providerId: "deepseek",
          apiHost: "https://api.deepseek.com",
          models: ["deepseek-chat"],
        },
      ],
      loading: false,
    });

    renderModelSelector({
      providerType: "",
      model: "",
      setProviderType,
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(setProviderType).toHaveBeenCalledWith("deepseek");
    expect(setProviderType).not.toHaveBeenCalledWith("lime-hub");
  });

  it("Lime Hub 有声明模型时应自动选中 provider 和默认模型", async () => {
    const setProviderType = vi.fn();
    const setModel = vi.fn();
    mockUseConfiguredProviders.mockReturnValue({
      providers: [
        {
          key: "lime-hub",
          label: "Lime Hub",
          registryId: "lime-hub",
          type: "openai",
          providerId: "lime-hub",
          apiHost: "https://llm.limeai.run#lime_tenant_id=tenant-0001",
          models: ["gpt-5.2-pro"],
          authStatus: "login_required",
        },
      ],
      loading: false,
    });

    renderModelSelector({
      providerType: "",
      model: "",
      setProviderType,
      setModel,
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(setProviderType).toHaveBeenCalledWith("lime-hub");
    expect(setModel).toHaveBeenCalledWith("gpt-5.2-pro");
  });

  it("Lime Hub 有声明模型时应展示模型列表而不是登录阻断空态", () => {
    mockUseConfiguredProviders.mockReturnValue({
      providers: [
        {
          key: "lime-hub",
          label: "Lime Hub",
          registryId: "lime-hub",
          type: "openai",
          providerId: "lime-hub",
          apiHost: "https://llm.limeai.run#lime_tenant_id=tenant-0001",
          models: ["gpt-5.2-pro"],
          authStatus: "login_required",
        },
      ],
      loading: false,
    });
    mockUseProviderModels.mockReturnValue({
      modelIds: ["gpt-5.2-pro"],
      models: [createModelMetadata("gpt-5.2-pro")],
      loading: false,
      error: null,
    });

    const { container } = renderModelSelector({
      providerType: "lime-hub",
      model: "gpt-5.2-pro",
    });

    clickModelSelectorTrigger(container);

    const pageText = getBodyText();
    expect(pageText).toContain("gpt-5.2-pro");
    expect(pageText).not.toContain("登录后会自动同步 Lime Hub 的可用模型");
  });

  it("打开选择器后应加载数据并回退到兼容模型", () => {
    const setModel = vi.fn();
    const { container } = renderModelSelector({
      model: "gpt-5.3-codex",
      setModel,
    });

    clickModelSelectorTrigger(container);

    expect(setModel).toHaveBeenCalledWith("gpt-5.2-codex");
  });

  it("当前模型暂未出现在异步模型列表时不应被旧缓存覆盖", async () => {
    const setModel = vi.fn();
    mockUseConfiguredProviders.mockReturnValue({
      providers: [
        {
          key: "lime-hub",
          label: "Lime Hub",
          registryId: "lime-hub",
          type: "openai",
          providerId: "lime-hub",
          apiHost: "https://llm.limeai.run",
          models: ["gpt-5.2-pro"],
        },
      ],
      loading: false,
    });
    mockUseProviderModels.mockReturnValue({
      modelIds: ["gpt-5.2-pro"],
      models: [createReasoningModelMetadata("gpt-5.2-pro")],
      loading: false,
      error: null,
    });

    renderModelSelector({
      providerType: "lime-hub",
      model: "gpt-5.4-mini",
      setModel,
      preserveUnknownModelSelection: true,
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(setModel).not.toHaveBeenCalledWith("gpt-5.2-pro");
  });

  it("切换供应商时应同步切换到该供应商的首个已配置模型，避免沿用旧模型", () => {
    const setProviderType = vi.fn();
    const setModel = vi.fn();
    mockUseConfiguredProviders.mockReturnValue({
      providers: [
        {
          key: "lime-hub",
          label: "Lime 云端",
          registryId: "lime-hub",
          type: "openai",
          providerId: "lime-hub",
          apiHost: "https://llm.limeai.run",
          models: ["gpt-5.5"],
        },
        {
          key: "deepseek",
          label: "DeepSeek",
          registryId: "deepseek",
          type: "openai",
          providerId: "deepseek",
          apiHost: "https://api.deepseek.com",
          models: ["deepseek-chat", "deepseek-reasoner"],
        },
      ],
      loading: false,
    });
    mockUseProviderModels.mockReturnValue({
      modelIds: ["gpt-5.5"],
      models: [createModelMetadata("gpt-5.5")],
      loading: false,
      error: null,
    });

    const { container } = renderModelSelector({
      providerType: "lime-hub",
      setProviderType,
      model: "gpt-5.5",
      setModel,
    });

    clickModelSelectorTrigger(container);
    clickBodyButtonByText("DeepSeek");

    expect(setProviderType).toHaveBeenCalledWith("deepseek");
    expect(setModel).toHaveBeenCalledWith("deepseek-chat");
    expect(setModel).not.toHaveBeenCalledWith("gpt-5.5");
  });

  it("提供原子选择回调时，切换供应商应只提交一次 Provider 与模型组合", () => {
    const setProviderType = vi.fn();
    const setModel = vi.fn();
    const setProviderAndModel = vi.fn();
    mockUseConfiguredProviders.mockReturnValue({
      providers: [
        {
          key: "lime-hub",
          label: "Lime 云端",
          registryId: "lime-hub",
          type: "openai",
          providerId: "lime-hub",
          models: ["gpt-5.5"],
        },
        {
          key: "deepseek",
          label: "DeepSeek",
          registryId: "deepseek",
          type: "openai",
          providerId: "deepseek",
          models: ["deepseek-chat"],
        },
      ],
      loading: false,
    });

    const { container } = renderModelSelector({
      providerType: "lime-hub",
      setProviderType,
      model: "gpt-5.5",
      setModel,
      setProviderAndModel,
    });

    setProviderType.mockClear();
    setModel.mockClear();
    setProviderAndModel.mockClear();
    clickModelSelectorTrigger(container);
    clickBodyButtonByText("DeepSeek");

    expect(setProviderAndModel).toHaveBeenCalledTimes(1);
    expect(setProviderAndModel).toHaveBeenCalledWith(
      "deepseek",
      "deepseek-chat",
    );
    expect(setProviderType).not.toHaveBeenCalled();
    expect(setModel).not.toHaveBeenCalled();
  });

  it("带模型过滤器的设置页切换供应商时不应回填该供应商不适用的首个自定义模型", () => {
    const setProviderType = vi.fn();
    const setModel = vi.fn();
    mockUseConfiguredProviders.mockReturnValue({
      providers: [
        {
          key: "openai",
          label: "OpenAI",
          registryId: "openai",
          type: "openai",
          providerId: "openai",
          apiHost: "https://api.openai.com/v1",
          models: ["gpt-4o-mini"],
        },
        {
          key: "deepseek",
          label: "DeepSeek",
          registryId: "deepseek",
          type: "openai",
          providerId: "deepseek",
          apiHost: "https://api.deepseek.com",
          models: ["gpt-image-2", "deepseek-v4-pro"],
        },
      ],
      loading: false,
    });
    mockUseProviderModels.mockReturnValue({
      modelIds: ["gpt-4o-mini"],
      models: [createTextOnlyModelMetadata("gpt-4o-mini")],
      loading: false,
      error: null,
    });

    const { container } = renderModelSelector({
      providerType: "openai",
      setProviderType,
      model: "gpt-4o-mini",
      setModel,
      modelFilter: (item) => item.id !== "gpt-image-2",
    });

    clickModelSelectorTrigger(container);
    clickBodyButtonByText("DeepSeek");

    expect(setProviderType).toHaveBeenCalledWith("deepseek");
    expect(setModel).toHaveBeenCalledWith("deepseek-v4-pro");
    expect(setModel).not.toHaveBeenCalledWith("gpt-image-2");
  });

  it("模型过滤清空本地模型时应展示调用方提供的图片模型回退", () => {
    mockUseConfiguredProviders.mockReturnValue({
      providers: [
        {
          key: "fal",
          label: "Fal",
          registryId: "fal",
          fallbackRegistryId: "openai",
          type: "openai",
          providerId: "fal",
          apiHost: "https://fal.run/fal-ai",
          models: ["gpt-5.2-pro"],
        },
      ],
      loading: false,
    });
    mockUseProviderModels.mockReturnValue({
      modelIds: ["gpt-5.2-pro"],
      models: [createModelMetadata("gpt-5.2-pro")],
      loading: false,
      error: null,
    });

    const { container } = renderModelSelector({
      providerType: "fal",
      model: "",
      modelFilter: (item) => item.id.startsWith("fal-ai/"),
      getFallbackModels: () => [createModelMetadata("fal-ai/nano-banana-pro")],
    });

    clickModelSelectorTrigger(container);

    const pageText = getBodyText();
    expect(pageText).toContain("fal-ai/nano-banana-pro");
    expect(pageText).not.toContain("gpt-5.2-pro");
    expect(pageText).not.toContain("暂无可用模型");
  });

  it("展开后应显示模型的思考与多模态能力标签", () => {
    mockUseProviderModels.mockReturnValue({
      modelIds: ["gpt-5.3-codex", "text-only-chat"],
      models: [
        createReasoningModelMetadata("gpt-5.3-codex"),
        createTextOnlyModelMetadata("text-only-chat"),
      ],
      loading: false,
      error: null,
    });

    const { container } = renderModelSelector();

    clickModelSelectorTrigger(container);

    const pageText = getBodyText();
    expect(pageText).toContain("支持思考");
    expect(pageText).toContain("支持多模态");
    expect(pageText).toContain("无多模态");
  });

  it("接口声明支持 reasoning_effort 时应展示强度选择并保持浮层打开", () => {
    const setReasoningEffort = vi.fn();
    mockUseProviderModels.mockReturnValue({
      modelIds: ["gpt-5.2-codex"],
      models: [
        createModelMetadata("gpt-5.2-codex", {
          source: "api",
          capabilities: {
            reasoning: true,
          },
          reasoning_policy: buildModelReasoningPolicy({
            defaultReasoningLevel: "medium",
            supportedReasoningLevels: ["low", "medium", "high"],
          }),
        }),
      ],
      loading: false,
      error: null,
    });

    const { container } = renderModelSelector({
      model: "gpt-5.2-codex",
      reasoningEffort: "medium",
      setReasoningEffort,
    });

    clickModelSelectorTrigger(container);

    const panel = document.body.querySelector(
      '[data-testid="model-selector-reasoning-effort"]',
    );
    expect(panel?.textContent).toContain("推理强度");
    expect(panel?.textContent).toContain("接口声明");

    clickBodyButtonByText("高");

    expect(setReasoningEffort).toHaveBeenCalledWith("high");
    expect(
      document.body.querySelector(
        '[data-testid="model-selector-reasoning-effort"]',
      ),
    ).not.toBeNull();
  });

  it("provider 未声明默认值时应按 Codex 模型切换语义选择中位档", () => {
    const setReasoningEffort = vi.fn();
    mockUseProviderModels.mockReturnValue({
      modelIds: ["provider-reasoning"],
      models: [
        createModelMetadata("provider-reasoning", {
          source: "api",
          capabilities: { reasoning: true },
          reasoning_policy: buildModelReasoningPolicy({
            supportedReasoningLevels: ["low", "medium", "high"],
          }),
        }),
      ],
      loading: false,
      error: null,
    });

    renderModelSelector({
      model: "provider-reasoning",
      reasoningEffort: "",
      setReasoningEffort,
    });

    expect(setReasoningEffort).toHaveBeenCalledWith("medium");
  });

  it("切换模型时应保留新模型支持的当前 reasoning effort", () => {
    const setModel = vi.fn();
    const setReasoningEffort = vi.fn();
    const policy = buildModelReasoningPolicy({
      supportedReasoningLevels: ["low", "medium", "high"],
    });
    mockUseProviderModels.mockReturnValue({
      modelIds: ["provider-a", "provider-b"],
      models: [
        createModelMetadata("provider-a", {
          capabilities: { reasoning: true },
          reasoning_policy: policy,
        }),
        createModelMetadata("provider-b", {
          capabilities: { reasoning: true },
          reasoning_policy: policy,
        }),
      ],
      loading: false,
      error: null,
    });

    const { container } = renderModelSelector({
      model: "provider-a",
      reasoningEffort: "high",
      setModel,
      setReasoningEffort,
    });

    clickModelSelectorTrigger(container);
    clickBodyButtonByText("provider-b");

    expect(setModel).toHaveBeenCalledWith("provider-b");
    expect(setReasoningEffort).toHaveBeenCalledWith("high");
  });

  it("切换模型时当前档位不受支持应选择新模型 catalog 中位档", () => {
    const setReasoningEffort = vi.fn();
    mockUseProviderModels.mockReturnValue({
      modelIds: ["provider-a", "provider-b"],
      models: [
        createModelMetadata("provider-a", {
          capabilities: { reasoning: true },
          reasoning_policy: buildModelReasoningPolicy({
            supportedReasoningLevels: ["low", "xhigh"],
          }),
        }),
        createModelMetadata("provider-b", {
          capabilities: { reasoning: true },
          reasoning_policy: buildModelReasoningPolicy({
            defaultReasoningLevel: "medium",
            supportedReasoningLevels: ["minimal", "low", "medium", "high"],
          }),
        }),
      ],
      loading: false,
      error: null,
    });

    const { container } = renderModelSelector({
      model: "provider-a",
      reasoningEffort: "xhigh",
      setReasoningEffort,
    });

    clickModelSelectorTrigger(container);
    clickBodyButtonByText("provider-b");

    expect(setReasoningEffort).toHaveBeenCalledWith("low");
  });

  it("展示 grok-build 的 reasoning label，但只提交 canonical wire value", () => {
    const setReasoningEffort = vi.fn();
    mockUseProviderModels.mockReturnValue({
      modelIds: ["grok-4.5"],
      models: [
        createModelMetadata("grok-4.5", {
          source: "api",
          capabilities: { reasoning: true },
          reasoning_policy: buildModelReasoningPolicy({
            defaultReasoningLevel: "balanced",
            supportedReasoningLevels: [
              {
                id: "deep",
                label: "Deep",
                value: "xhigh",
                description: "Maximum reasoning",
              },
              {
                id: "balanced",
                label: "Balanced",
                value: "medium",
              },
            ],
          }),
        }),
      ],
      loading: false,
      error: null,
    });

    const { container } = renderModelSelector({
      model: "grok-4.5",
      reasoningEffort: "medium",
      setReasoningEffort,
    });

    clickModelSelectorTrigger(container);

    const panel = document.body.querySelector(
      '[data-testid="model-selector-reasoning-effort"]',
    );
    expect(panel?.textContent).toContain("Deep");
    expect(panel?.textContent).toContain("Balanced");

    clickBodyButtonByText("Deep");

    expect(setReasoningEffort).toHaveBeenCalledWith("xhigh");
    expect(setReasoningEffort).not.toHaveBeenCalledWith("deep");
  });

  it("未由接口声明 reasoning_effort 时不展示强度选择", () => {
    mockUseProviderModels.mockReturnValue({
      modelIds: ["gpt-5.2-codex", "reasoning-only"],
      models: [
        createModelMetadata("gpt-5.2-codex", {
          source: "api",
          capabilities: {
            reasoning: true,
            reasoning_effort: {
              supported: true,
              levels: ["low", "medium", "high"],
              default: "medium",
              source: "registry",
            },
          },
        }),
        createModelMetadata("reasoning-only", {
          source: "api",
          capabilities: {
            reasoning: true,
          },
        }),
      ],
      loading: false,
      error: null,
    });

    const { container } = renderModelSelector({
      model: "gpt-5.2-codex",
    });

    clickModelSelectorTrigger(container);

    expect(
      document.body.querySelector(
        '[data-testid="model-selector-reasoning-effort"]',
      ),
    ).toBeNull();
  });

  it("en-US locale 应展示 common namespace 里的模型选择 chrome", async () => {
    await changeLimeLocale("en-US");
    mockUseProviderModels.mockReturnValue({
      modelIds: ["gpt-5.3-codex", "text-only-chat"],
      models: [
        createReasoningModelMetadata("gpt-5.3-codex"),
        createTextOnlyModelMetadata("text-only-chat"),
      ],
      loading: false,
      error: null,
    });

    const { container } = renderModelSelector({
      onManageProviders: vi.fn(),
    });

    clickModelSelectorTrigger(container);

    const pageText = getBodyText();
    expect(pageText).toContain("Model selection");
    expect(pageText).toContain("Currently organized by General chat");
    expect(pageText).toContain("Providers");
    expect(pageText).toContain("Model list");
    expect(pageText).toContain("Supports reasoning");
    expect(pageText).toContain("No multimodal");
    expect(pageText).toContain("Manage providers");
    expect(pageText).not.toContain("模型选择");
    expect(pageText).not.toContain("支持思考");
  });

  it("展开后应把 Lime 云端模型与本地供应商分组显示", () => {
    mockUseConfiguredProviders.mockReturnValue({
      providers: [
        {
          key: "lime-hub",
          label: "Lime 云端",
          registryId: "lime-hub",
          type: "openai",
          providerId: "lime-hub",
          apiHost: "https://llm.limeai.run",
        },
        {
          key: "custom-local-cli",
          label: "Local CLI Custom",
          registryId: "custom-local-cli",
          fallbackRegistryId: "codex",
          type: "codex",
          providerId: "custom-local-cli",
          apiHost: "https://api.openai.com/v1",
        },
      ],
      loading: false,
    });
    mockUseProviderModels.mockReturnValue({
      modelIds: ["gpt-5.5", "deepseek-v4-flash"],
      models: [
        createReasoningModelMetadata("gpt-5.5"),
        createTextOnlyModelMetadata("deepseek-v4-flash"),
      ],
      loading: false,
      error: null,
    });

    const { container } = renderModelSelector({
      providerType: "lime-hub",
      model: "gpt-5.5",
    });

    clickModelSelectorTrigger(container);

    const pageText = getBodyText();
    expect(pageText).toContain("云端模型");
    expect(pageText).toContain("本地与自定义");
    expect(pageText).toContain("Lime 云端");
    expect(pageText).toContain("Local CLI Custom");
    expect(pageText).toContain("gpt-5.5");
    expect(pageText).toContain("deepseek-v4-flash");
    expect(pageText).not.toContain("暂无可用模型");
  });

  it("Lime Hub 未登录时应在下拉框提示登录且不展示本地兜底模型", () => {
    const onManageProviders = vi.fn();
    mockUseConfiguredProviders.mockReturnValue({
      providers: [
        {
          key: "lime-hub",
          label: "Lime 云端",
          registryId: "lime-hub",
          type: "openai",
          providerId: "lime-hub",
          apiHost: "https://llm.limeai.run",
          models: [],
          authStatus: "login_required",
        },
      ],
      loading: false,
    });
    mockUseProviderModels.mockReturnValue({
      modelIds: ["gpt-5.5", "gpt-5.4"],
      models: [createModelMetadata("gpt-5.5"), createModelMetadata("gpt-5.4")],
      loading: false,
      error: null,
    });

    const { container } = renderModelSelector({
      providerType: "lime-hub",
      model: "gpt-5.5",
      onManageProviders,
    });

    expect(container.textContent).toContain("Lime 云端");
    expect(container.textContent).toContain("需要登录");

    clickModelSelectorTrigger(container);

    expect(mockUseProviderModels).toHaveBeenCalledWith(
      expect.objectContaining({
        key: "lime-hub",
        providerId: "lime-hub",
        authStatus: "login_required",
      }),
      expect.objectContaining({
        returnFullMetadata: true,
        autoLoad: false,
      }),
    );

    const pageText = getBodyText();
    expect(pageText).toContain("云端模型");
    expect(pageText).toContain("Lime 云端");
    expect(pageText).toContain("需要登录");
    expect(pageText).toContain("登录后会自动同步 Lime Hub 的可用模型");
    expect(pageText).toContain("去登录");
    expect(pageText).not.toContain("gpt-5.5");
    expect(pageText).not.toContain("gpt-5.4");

    clickBodyButtonByText("去登录");

    expect(onManageProviders).toHaveBeenCalledTimes(1);
  });

  it("未知 anthropic-compatible Provider 应在选择器中展示显式缓存提示", () => {
    mockUseConfiguredProviders.mockReturnValue({
      providers: [
        {
          key: "custom-anthropic-compatible",
          label: "GLM Anthropic",
          registryId: "custom-anthropic-compatible",
          fallbackRegistryId: "anthropic",
          type: "anthropic-compatible",
          providerId: "custom-anthropic-compatible",
          apiHost: "https://api.example.com/anthropic",
        },
      ],
      loading: false,
    });
    mockUseProviderModels.mockReturnValue({
      modelIds: ["glm-5.1"],
      models: [createReasoningModelMetadata("glm-5.1")],
      loading: false,
      error: null,
    });

    const { container } = renderModelSelector({
      providerType: "custom-anthropic-compatible",
      model: "glm-5.1",
    });

    clickModelSelectorTrigger(container);

    const pageText = getBodyText();
    expect(pageText).toContain("显式缓存");
    expect(pageText).toContain("未声明自动 Prompt Cache");
    expect(pageText).toContain("cache_control");
  });

  it.each([
    {
      label: "GLM Anthropic",
      apiHost: "https://open.bigmodel.cn/api/anthropic",
      model: "glm-5.1",
    },
    {
      label: "Z.AI Anthropic",
      apiHost: "https://api.z.ai/api/anthropic",
      model: "glm-4.7",
    },
    {
      label: "Kimi Anthropic",
      apiHost: "https://api.moonshot.cn/anthropic",
      model: "kimi-k2.5",
    },
    {
      label: "Kimi Global Anthropic",
      apiHost: "https://api.moonshot.ai/anthropic",
      model: "kimi-k2.5",
    },
    {
      label: "Kimi Code Subscription",
      apiHost: "https://api.kimi.com/coding/",
      model: "k2p5",
    },
    {
      label: "MiniMax Anthropic",
      apiHost: "https://api.minimaxi.com/anthropic",
      model: "minimax-m1",
    },
    {
      label: "MiniMax Global Anthropic",
      apiHost: "https://api.minimax.io/anthropic",
      model: "minimax-m1",
    },
    {
      label: "Alibaba Coding Anthropic",
      apiHost: "https://coding.dashscope.aliyuncs.com/apps/anthropic",
      model: "qwen3-coder-plus",
    },
    {
      label: "Alibaba Global Coding Anthropic",
      apiHost: "https://coding-intl.dashscope.aliyuncs.com/apps/anthropic",
      model: "qwen3-coder-plus",
    },
    {
      label: "MiMo Anthropic",
      apiHost: "https://token-plan-cn.xiaomimimo.com/anthropic",
      model: "mimo-v2-flash",
    },
  ])("$label 不应在选择器中误报显式缓存提示", ({ label, apiHost, model }) => {
    mockUseConfiguredProviders.mockReturnValue({
      providers: [
        {
          key: "custom-anthropic-compatible",
          label,
          registryId: "custom-anthropic-compatible",
          fallbackRegistryId: "anthropic",
          type: "anthropic-compatible",
          providerId: "custom-anthropic-compatible",
          apiHost,
        },
      ],
      loading: false,
    });
    mockUseProviderModels.mockReturnValue({
      modelIds: [model],
      models: [createReasoningModelMetadata(model)],
      loading: false,
      error: null,
    });

    const { container } = renderModelSelector({
      providerType: "custom-anthropic-compatible",
      model,
    });

    clickModelSelectorTrigger(container);

    const pageText = getBodyText();
    expect(pageText).not.toContain("显式缓存");
    expect(pageText).not.toContain("未声明自动 Prompt Cache");
  });

  it("显式声明 automatic 的 anthropic-compatible Provider 不应在选择器中误报显式缓存提示", () => {
    mockUseConfiguredProviders.mockReturnValue({
      providers: [
        {
          key: "custom-anthropic-compatible",
          label: "GLM Anthropic Automatic",
          registryId: "custom-anthropic-compatible",
          fallbackRegistryId: "anthropic",
          type: "anthropic-compatible",
          providerId: "custom-anthropic-compatible",
          apiHost: "https://open.bigmodel.cn/api/anthropic",
          promptCacheMode: "automatic",
        },
      ],
      loading: false,
    });
    mockUseProviderModels.mockReturnValue({
      modelIds: ["glm-5.1"],
      models: [createReasoningModelMetadata("glm-5.1")],
      loading: false,
      error: null,
    });

    const { container } = renderModelSelector({
      providerType: "custom-anthropic-compatible",
      model: "glm-5.1",
    });

    clickModelSelectorTrigger(container);

    const pageText = getBodyText();
    expect(pageText).not.toContain("显式缓存");
    expect(pageText).not.toContain("未声明自动 Prompt Cache");
  });

  it("无 Provider 引导关闭后应隐藏，并在重新挂载时保持关闭状态", () => {
    mockUseConfiguredProviders.mockReturnValue({
      providers: [],
      loading: false,
    });
    mockUseProviderModels.mockReturnValue({
      modelIds: [],
      models: [],
      loading: false,
      error: null,
    });

    const firstRender = renderModelSelector({
      providerType: "",
      model: "",
    });

    expect(firstRender.container.textContent).toContain("工具模型未配置");

    clickNoProviderGuideDismissButton(firstRender.container);

    expect(firstRender.container.textContent ?? "").not.toContain(
      "工具模型未配置",
    );
    expect(
      window.localStorage.getItem(
        "lime_model_selector_no_provider_guide_dismissed_v1",
      ),
    ).toBe("1");

    const secondRender = renderModelSelector({
      providerType: "",
      model: "",
    });

    expect(secondRender.container.textContent ?? "").not.toContain(
      "工具模型未配置",
    );
  });
});
