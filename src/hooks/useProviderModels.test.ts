import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ConfiguredProvider } from "./useConfiguredProviders";
import { loadProviderModels } from "./useProviderModels";

const {
  mockGetModelRegistry,
  mockGetProviderAliasConfig,
  mockFetchProviderModelsAuto,
  mockNormalizeFetchProviderModelsSource,
} = vi.hoisted(() => ({
  mockGetModelRegistry: vi.fn(),
  mockGetProviderAliasConfig: vi.fn(),
  mockFetchProviderModelsAuto: vi.fn(),
  mockNormalizeFetchProviderModelsSource: vi.fn(
    (result) => result?.source ?? "Error",
  ),
}));

vi.mock("@/lib/api/modelRegistry", () => ({
  normalizeFetchProviderModelsSource: mockNormalizeFetchProviderModelsSource,
  modelRegistryApi: {
    getModelRegistry: mockGetModelRegistry,
    getProviderAliasConfig: mockGetProviderAliasConfig,
    fetchProviderModelsAuto: mockFetchProviderModelsAuto,
  },
}));

function createProvider(
  overrides: Partial<ConfiguredProvider> = {},
): ConfiguredProvider {
  return {
    key: "openai_api_key",
    label: "OpenAI API Key",
    registryId: "openai",
    fallbackRegistryId: "openai",
    type: "openai",
    providerId: "openai",
    apiHost: "https://api.openai.com/v1",
    ...overrides,
  };
}

function createLimeHubProvider(
  overrides: Partial<ConfiguredProvider> = {},
): ConfiguredProvider {
  return createProvider({
    key: "lime-hub",
    label: "Lime Hub",
    registryId: "lime-hub",
    fallbackRegistryId: undefined,
    type: "openai",
    providerId: "lime-hub",
    apiHost: "https://llm.limeai.run#lime_tenant_id=tenant-0001",
    ...overrides,
  });
}

function createModelMetadata(id: string) {
  return {
    id,
    display_name: id,
    provider_id: "openai",
    provider_name: "OpenAI",
    family: null,
    tier: "pro" as const,
    capabilities: {
      vision: false,
      tools: true,
      streaming: true,
      json_mode: true,
      function_calling: true,
      reasoning: false,
    },
    pricing: null,
    limits: {
      context_length: null,
      max_output_tokens: null,
      requests_per_minute: null,
      tokens_per_minute: null,
    },
    status: "active" as const,
    release_date: null,
    is_latest: false,
    description: null,
    source: "custom" as const,
    created_at: 0,
    updated_at: 0,
  };
}

describe("loadProviderModels", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetModelRegistry.mockResolvedValue([]);
    mockGetProviderAliasConfig.mockResolvedValue(null);
    mockFetchProviderModelsAuto.mockResolvedValue({
      models: [],
      source: "Api",
      error: null,
    });
  });

  it("实时拉取时应优先使用真实 providerId，而不是前端去重后的 key", async () => {
    mockFetchProviderModelsAuto.mockResolvedValueOnce({
      models: [
        {
          id: "gpt-5.1",
          display_name: "GPT-5.1",
          provider_id: "openai",
          provider_name: "OpenAI",
          family: null,
          tier: "pro",
          capabilities: {
            vision: true,
            tools: true,
            streaming: true,
            json_mode: true,
            function_calling: true,
            reasoning: true,
          },
          pricing: null,
          limits: {
            context_length: null,
            max_output_tokens: null,
            requests_per_minute: null,
            tokens_per_minute: null,
          },
          status: "active",
          release_date: null,
          is_latest: true,
          description: null,
          source: "custom",
          created_at: 0,
          updated_at: 0,
        },
      ],
      source: "Api",
      error: null,
    });

    const models = await loadProviderModels(createProvider(), {
      liveFetchOnly: true,
      hasApiKey: true,
    });

    expect(models.map((model) => model.id)).toEqual(["gpt-5.1"]);
    expect(mockFetchProviderModelsAuto).toHaveBeenCalledWith("openai");
  });

  it("实时模型目录应合并当前 Provider 已启用模型，避免设置页模型在选择器里消失", async () => {
    mockFetchProviderModelsAuto.mockResolvedValueOnce({
      models: [createModelMetadata("Pro/moonshotai/Kimi-K2.6")],
      source: "Api",
      error: null,
    });

    const models = await loadProviderModels(
      createProvider({
        key: "siliconflow-cn",
        label: "Silicon Flow (国内)",
        registryId: "siliconflow-cn",
        providerId: "siliconflow-cn",
        apiHost: "https://api.siliconflow.cn/v1",
        models: ["deepseek-ai/DeepSeek-V4-Flash"],
      }),
      {
        liveFetchOnly: true,
        hasApiKey: true,
      },
    );

    expect(models.map((model) => model.id)).toEqual([
      "deepseek-ai/DeepSeek-V4-Flash",
      "Pro/moonshotai/Kimi-K2.6",
    ]);
  });

  it("实时目录读取失败且无本地兜底时，不应继续把旧模型当成最新模型展示", async () => {
    mockFetchProviderModelsAuto.mockResolvedValueOnce({
      models: [],
      source: "Error",
      error: "API 获取失败: 401 Unauthorized。本地模型兜底已下线。",
    });

    const models = await loadProviderModels(createProvider(), {
      liveFetchOnly: true,
      hasApiKey: true,
    });

    expect(models).toEqual([]);
  });

  it("错误来源即使夹带模型，也不应视为有效实时模型真相", async () => {
    mockFetchProviderModelsAuto.mockResolvedValueOnce({
      models: [
        {
          id: "MiniMax-M2.7",
          display_name: "MiniMax-M2.7",
          provider_id: "minimax",
          provider_name: "MiniMax",
          family: null,
          tier: "pro",
          capabilities: {
            vision: false,
            tools: true,
            streaming: true,
            json_mode: true,
            function_calling: true,
            reasoning: true,
          },
          pricing: null,
          limits: {
            context_length: null,
            max_output_tokens: null,
            requests_per_minute: null,
            tokens_per_minute: null,
          },
          status: "active",
          release_date: null,
          is_latest: true,
          description: null,
          source: "local",
          created_at: 0,
          updated_at: 0,
        },
      ],
      source: "Error",
      error: "API 获取失败，本地模型兜底已下线。",
    });

    const models = await loadProviderModels(
      createProvider({
        key: "minimax-test",
        providerId: "custom-minimax",
        type: "anthropic-compatible" as ConfiguredProvider["type"],
        apiHost: "https://api.minimaxi.com/anthropic",
      }),
      {
        liveFetchOnly: true,
        hasApiKey: true,
      },
    );

    expect(models).toEqual([]);
  });

  it("接口读取失败时仍保留当前 Provider 已显式配置的模型", async () => {
    mockFetchProviderModelsAuto.mockResolvedValueOnce({
      models: [],
      source: "Error",
      error: "当前 Anthropic 兼容入口未提供标准 /models 接口。",
    });

    const models = await loadProviderModels(
      createProvider({
        key: "minimax-test",
        providerId: "custom-minimax",
        type: "anthropic-compatible" as ConfiguredProvider["type"],
        apiHost: "https://api.minimaxi.com/anthropic",
        models: ["MiniMax-M2.7"],
      }),
      {
        liveFetchOnly: true,
        hasApiKey: true,
      },
    );

    expect(models.map((model) => model.id)).toEqual(["MiniMax-M2.7"]);
  });

  it("实时目录无 API Key 时不触发后台模型拉取", async () => {
    const models = await loadProviderModels(
      createProvider({
        models: ["manual-model"],
      }),
      {
        liveFetchOnly: true,
        hasApiKey: false,
      },
    );

    expect(models.map((model) => model.id)).toEqual(["manual-model"]);
    expect(mockFetchProviderModelsAuto).not.toHaveBeenCalled();
  });

  it("Lime Hub 实时目录为空时不应注入本地 mock 模型", async () => {
    mockFetchProviderModelsAuto.mockResolvedValueOnce({
      models: [],
      source: "Api",
      error: null,
    });

    const models = await loadProviderModels(createLimeHubProvider(), {
      liveFetchOnly: true,
      hasApiKey: true,
    });

    expect(models).toEqual([]);
    expect(mockFetchProviderModelsAuto).toHaveBeenCalledWith("lime-hub");
  });

  it("普通 Provider 实时目录为空时也不应注入本地模型兜底", async () => {
    mockFetchProviderModelsAuto.mockResolvedValueOnce({
      models: [],
      source: "Api",
      error: null,
    });

    const models = await loadProviderModels(createProvider(), {
      liveFetchOnly: true,
      hasApiKey: true,
    });

    expect(models).toEqual([]);
  });

  it("Lime Hub 未登录时不读取模型注册表和实时接口", async () => {
    const models = await loadProviderModels(
      createLimeHubProvider({
        authStatus: "login_required",
      }),
      {
        liveFetchOnly: true,
        hasApiKey: true,
      },
    );

    expect(models).toEqual([]);
    expect(mockGetModelRegistry).not.toHaveBeenCalled();
    expect(mockGetProviderAliasConfig).not.toHaveBeenCalled();
    expect(mockFetchProviderModelsAuto).not.toHaveBeenCalled();
  });

  it("Lime Hub 未登录但已有声明模型时应返回声明模型且不读取实时接口", async () => {
    const models = await loadProviderModels(
      createLimeHubProvider({
        authStatus: "login_required",
        models: ["gpt-5.2-pro"],
      }),
      {
        liveFetchOnly: true,
        hasApiKey: true,
      },
    );

    expect(models.map((model) => model.id)).toEqual(["gpt-5.2-pro"]);
    expect(mockGetModelRegistry).not.toHaveBeenCalled();
    expect(mockGetProviderAliasConfig).not.toHaveBeenCalled();
    expect(mockFetchProviderModelsAuto).not.toHaveBeenCalled();
  });
});
