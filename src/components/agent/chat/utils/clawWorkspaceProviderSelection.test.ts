import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ConfiguredProvider } from "@/hooks/useConfiguredProviders";
import type { EnhancedModelMetadata } from "@/lib/types/modelRegistry";
import { resolveClawWorkspaceProviderSelection } from "./clawWorkspaceProviderSelection";

const {
  mockLoadConfiguredProviders,
  mockGetModelRegistry,
  mockGetProviderAliasConfig,
  mockFetchProviderModelsAuto,
  mockNormalizeFetchProviderModelsSource,
  mockFilterModelsByTheme,
} = vi.hoisted(() => ({
  mockLoadConfiguredProviders: vi.fn(),
  mockGetModelRegistry: vi.fn(),
  mockGetProviderAliasConfig: vi.fn(),
  mockFetchProviderModelsAuto: vi.fn(),
  mockNormalizeFetchProviderModelsSource: vi.fn(
    (result) => result?.source ?? "Error",
  ),
  mockFilterModelsByTheme: vi.fn(),
}));

vi.mock("@/hooks/useConfiguredProviders", () => ({
  loadConfiguredProviders: mockLoadConfiguredProviders,
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

vi.mock("@/lib/api/modelRegistry", () => ({
  normalizeFetchProviderModelsSource: mockNormalizeFetchProviderModelsSource,
  modelRegistryApi: {
    getModelRegistry: mockGetModelRegistry,
    getProviderAliasConfig: mockGetProviderAliasConfig,
    fetchProviderModelsAuto: mockFetchProviderModelsAuto,
  },
}));

vi.mock("./modelThemePolicy", () => ({
  filterModelsByTheme: mockFilterModelsByTheme,
}));

function createProvider(
  overrides: Partial<ConfiguredProvider> = {},
): ConfiguredProvider {
  return {
    key: "custom-social-provider",
    label: "Custom Social Provider",
    registryId: "custom-social-provider",
    fallbackRegistryId: "openai",
    type: "openai",
    ...overrides,
  };
}

function createModel(
  id: string,
  overrides: Partial<EnhancedModelMetadata> = {},
): EnhancedModelMetadata {
  return {
    id,
    display_name: id,
    provider_id: "custom-social-provider",
    provider_name: "Custom Social Provider",
    family: null,
    tier: "pro",
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
    status: "active",
    release_date: null,
    is_latest: false,
    description: null,
    source: "local",
    created_at: 0,
    updated_at: 0,
    ...overrides,
  };
}

describe("resolveClawWorkspaceProviderSelection", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockLoadConfiguredProviders.mockResolvedValue([]);
    mockGetModelRegistry.mockResolvedValue([]);
    mockGetProviderAliasConfig.mockResolvedValue(null);
    mockFetchProviderModelsAuto.mockResolvedValue({
      models: [],
      source: "Error",
      error: null,
    });
    mockFilterModelsByTheme.mockImplementation(
      (_theme: string | undefined, models: EnhancedModelMetadata[]) => ({
        models,
        usedFallback: false,
        filteredOutCount: 0,
        policyName: "mock",
      }),
    );
  });

  it("本地注册表存在模型时应直接选中，不再触发 provider API fallback", async () => {
    mockLoadConfiguredProviders.mockResolvedValueOnce([
      createProvider(),
      createProvider({
        key: "backup-provider",
        label: "Backup Provider",
        registryId: "backup-provider",
      }),
    ]);
    mockGetModelRegistry.mockResolvedValueOnce([
      createModel("social-model-1", {
        is_latest: true,
      }),
    ]);

    const result = await resolveClawWorkspaceProviderSelection({
      currentProviderType: "custom-social-provider",
      currentModel: "legacy-model",
      theme: "general",
    });

    expect(result).toEqual({
      providerType: "custom-social-provider",
      model: "social-model-1",
    });
    expect(mockFetchProviderModelsAuto).not.toHaveBeenCalled();
  });

  it("解析普通聊天模型时应跳过纯图片模型", async () => {
    mockLoadConfiguredProviders.mockResolvedValueOnce([createProvider()]);
    mockGetModelRegistry.mockResolvedValueOnce([
      createModel("gpt-image-1", {
        task_families: ["image_generation"],
        output_modalities: ["image"],
        is_latest: true,
      }),
      createModel("gpt-5.4-mini", {
        task_families: ["chat"],
        output_modalities: ["text"],
      }),
    ]);

    const result = await resolveClawWorkspaceProviderSelection({
      currentProviderType: "custom-social-provider",
      currentModel: null,
      theme: "general",
    });

    expect(result).toEqual({
      providerType: "custom-social-provider",
      model: "gpt-5.4-mini",
    });
    expect(mockFetchProviderModelsAuto).not.toHaveBeenCalled();
  });

  it("本地注册表无模型时应回退到后端 provider API 结果", async () => {
    mockLoadConfiguredProviders.mockResolvedValueOnce([
      createProvider({
        providerId: "managed-social-provider",
        apiHost: "https://api.example.com/v1",
      }),
    ]);
    mockFetchProviderModelsAuto.mockResolvedValueOnce({
      models: [
        createModel("social-model-api", {
          source: "custom",
          is_latest: true,
        }),
      ],
      source: "Api",
      error: null,
    });

    const result = await resolveClawWorkspaceProviderSelection({
      currentProviderType: "custom-social-provider",
      currentModel: null,
      theme: "general",
    });

    expect(result).toEqual({
      providerType: "managed-social-provider",
      model: "social-model-api",
    });
    expect(mockFetchProviderModelsAuto).toHaveBeenCalledWith(
      "managed-social-provider",
    );
  });

  it("当前 provider 只回填原始 providerId 时，应优先解析到真实受管 Provider", async () => {
    mockLoadConfiguredProviders.mockResolvedValueOnce([
      createProvider({
        key: "openai",
        label: "OpenAI",
        registryId: "openai",
        fallbackRegistryId: undefined,
      }),
      createProvider({
        key: "openai_api_key",
        label: "OpenAI API Key",
        registryId: "openai",
        providerId: "openai",
        apiHost: "https://api.openai.com/v1",
      }),
    ]);
    mockFetchProviderModelsAuto.mockResolvedValueOnce({
      models: [
        createModel("gpt-5.4-mini", {
          provider_id: "openai",
          provider_name: "OpenAI",
          source: "custom",
          is_latest: true,
        }),
      ],
      source: "Api",
      error: null,
    });

    const result = await resolveClawWorkspaceProviderSelection({
      currentProviderType: "openai",
      currentModel: null,
      theme: "general",
    });

    expect(result).toEqual({
      providerType: "openai",
      model: "gpt-5.4-mini",
    });
    expect(mockFetchProviderModelsAuto).toHaveBeenCalledWith("openai");
  });

  it("关闭 provider fallback 时不应跨到其他已配置 Provider", async () => {
    mockLoadConfiguredProviders.mockResolvedValueOnce([
      createProvider({
        key: "lime-hub",
        label: "Lime Hub",
        registryId: "lime-hub",
        providerId: "lime-hub",
        apiHost: "https://hub.example.com/v1",
      }),
      createProvider({
        key: "deepseek",
        label: "DeepSeek",
        registryId: "deepseek",
        providerId: "deepseek",
        apiHost: "https://api.deepseek.com/v1",
      }),
    ]);
    mockGetModelRegistry.mockResolvedValueOnce([
      createModel("deepseek-v4-pro", {
        provider_id: "deepseek",
        provider_name: "DeepSeek",
      }),
    ]);
    mockFetchProviderModelsAuto.mockResolvedValueOnce({
      models: [],
      source: "Error",
      error: null,
    });

    const result = await resolveClawWorkspaceProviderSelection({
      currentProviderType: "lime-hub",
      currentModel: null,
      theme: "general",
      allowProviderFallback: false,
    });

    expect(result).toBeNull();
    expect(mockFetchProviderModelsAuto).toHaveBeenCalledTimes(1);
    expect(mockFetchProviderModelsAuto).toHaveBeenCalledWith("lime-hub");
  });

  it("自动解析 Claw provider 时应跳过需要登录的 Lime Hub 提示", async () => {
    mockLoadConfiguredProviders.mockResolvedValueOnce([
      createProvider({
        key: "lime-hub",
        label: "Lime Hub",
        registryId: "lime-hub",
        providerId: "lime-hub",
        apiHost: "https://hub.example.com/v1",
        authStatus: "login_required",
      }),
      createProvider({
        key: "deepseek",
        label: "DeepSeek",
        registryId: "deepseek",
        providerId: "deepseek",
        apiHost: "https://api.deepseek.com/v1",
      }),
    ]);
    mockFetchProviderModelsAuto.mockResolvedValueOnce({
      models: [
        createModel("deepseek-chat", {
          provider_id: "deepseek",
          provider_name: "DeepSeek",
        }),
      ],
      source: "Api",
      error: null,
    });

    const result = await resolveClawWorkspaceProviderSelection({
      currentProviderType: "lime-hub",
      currentModel: "gpt-5.5",
      theme: "general",
    });

    expect(result).toEqual({
      providerType: "deepseek",
      model: "deepseek-chat",
    });
    expect(mockFetchProviderModelsAuto).toHaveBeenCalledTimes(1);
    expect(mockFetchProviderModelsAuto).toHaveBeenCalledWith("deepseek");
  });

  it("需要登录但已有声明模型的 Lime Hub 应保留为可选模型", async () => {
    mockLoadConfiguredProviders.mockResolvedValueOnce([
      createProvider({
        key: "lime-hub",
        label: "Lime Hub",
        registryId: "lime-hub",
        providerId: "lime-hub",
        apiHost: "https://llm.limeai.run#lime_tenant_id=tenant-0001",
        authStatus: "login_required",
        models: ["agnes-2.0-flash"],
      }),
      createProvider({
        key: "deepseek",
        label: "DeepSeek",
        registryId: "deepseek",
        providerId: "deepseek",
        apiHost: "https://api.deepseek.com/v1",
      }),
    ]);

    const result = await resolveClawWorkspaceProviderSelection({
      currentProviderType: "lime-hub",
      currentModel: "",
      theme: "general",
    });

    expect(result).toEqual({
      providerType: "lime-hub",
      model: "agnes-2.0-flash",
    });
    expect(mockFetchProviderModelsAuto).not.toHaveBeenCalled();
  });

  it("只有图片生成模型时不应为普通 Claw 聊天选择该 Provider", async () => {
    mockLoadConfiguredProviders.mockResolvedValueOnce([
      createProvider({
        key: "fixture-image-provider",
        label: "Fixture Image Provider",
        registryId: "openai",
        providerId: "custom-image-provider",
        apiHost: "http://127.0.0.1:56755/v1",
      }),
    ]);
    mockFetchProviderModelsAuto.mockResolvedValueOnce({
      models: [
        createModel("gpt-image-1", {
          provider_id: "openai",
          provider_name: "OpenAI",
          task_families: ["image_generation"],
          input_modalities: ["text"],
          output_modalities: ["image"],
          source: "custom",
        }),
      ],
      source: "Api",
      error: null,
    });

    const result = await resolveClawWorkspaceProviderSelection({
      currentProviderType: "fixture-provider",
      currentModel: "fixture-model",
      theme: "general",
    });

    expect(result).toBeNull();
  });
});
