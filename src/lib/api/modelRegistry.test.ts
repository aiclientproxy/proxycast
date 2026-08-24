import { beforeEach, describe, expect, it, vi } from "vitest";
import { safeInvoke } from "@/lib/dev-bridge";
import type { AppServerEventBusSubscription } from "@/lib/api/appServerEventBus";
import {
  getAllAliasConfigs,
  getModelRegistry,
  getModelRegistryProviderIds,
  readModelProviderCapabilities,
  getModelPreferences,
  getProviderAliasConfig,
  getModelSyncState,
  fetchProviderModelsAuto,
  hideModel,
  invalidateModelRegistryCache,
  recordModelUsage,
  refreshModelRegistry,
  searchModels,
  subscribeModelRegistryUpdates,
  toggleModelFavorite,
} from "./modelRegistry";

const appServerRequestMock = vi.hoisted(() => vi.fn());
const subscribeAppServerNotificationsMock = vi.hoisted(() => vi.fn());

vi.mock("@/lib/api/appServer", () => ({
  AppServerClient: vi.fn(() => ({
    request: appServerRequestMock,
  })),
}));

vi.mock("@/lib/api/appServerEventBus", () => ({
  subscribeAppServerNotifications: subscribeAppServerNotificationsMock,
}));

vi.mock("@/lib/dev-bridge", () => ({
  safeInvoke: vi.fn(),
}));

function resolveAppServerRequest<T>(result: T): void {
  appServerRequestMock.mockResolvedValueOnce({ result });
}

function expectAppServerRequest(
  index: number,
  method: string,
  params: unknown,
): void {
  expect(appServerRequestMock).toHaveBeenNthCalledWith(index, method, params);
}

function createModelInfo(overrides: Record<string, unknown> = {}) {
  return {
    id: "gpt-4.1",
    displayName: "GPT-4.1",
    providerId: "openai",
    providerName: "OpenAI",
    family: null,
    tier: "pro",
    capabilities: {
      vision: false,
      tools: true,
      streaming: true,
      jsonMode: true,
      functionCalling: true,
      reasoning: false,
      reasoningEffort: null,
    },
    capabilityProvenance: "provider_explicit",
    taskFamilies: ["chat"],
    inputModalities: ["text"],
    outputModalities: ["text"],
    runtimeFeatures: ["streaming"],
    deploymentSource: "user_cloud",
    managementPlane: "local_settings",
    canonicalModelId: null,
    providerModelId: null,
    aliasSource: null,
    pricing: null,
    limits: {},
    status: "active",
    releaseDate: null,
    isLatest: false,
    description: null,
    source: "api",
    createdAt: 1,
    updatedAt: 2,
    ...overrides,
  };
}

function routePart(value: string): string {
  return btoa(value)
    .replaceAll("+", "-")
    .replaceAll("/", "_")
    .replaceAll("=", "");
}

function createCatalogModel(
  providerId = "openai",
  modelId = "gpt-4.1",
  overrides: Record<string, unknown> = {},
) {
  return {
    id: `route:${routePart(providerId)}.${routePart(modelId)}`,
    providerId,
    model: modelId,
    upgrade: null,
    upgradeInfo: null,
    availabilityNux: null,
    displayName: modelId,
    description: "",
    hidden: false,
    supportedReasoningEfforts: [],
    defaultReasoningEffort: "none",
    inputModalities: ["text"],
    capabilitySnapshot: {
      taskFamilies: ["chat"],
      inputModalities: ["text"],
      outputModalities: ["text"],
      runtimeFeatures: ["streaming"],
      capabilities: {
        vision: false,
        tools: false,
        streaming: true,
        jsonMode: false,
        functionCalling: false,
        reasoning: false,
        reasoningEffort: null,
      },
      source: "provider_explicit",
      reasonCode: null,
    },
    contextWindow: null,
    maxOutputTokens: null,
    supportsPersonality: false,
    multiAgentVersion: null,
    additionalSpeedTiers: [],
    serviceTiers: [],
    defaultServiceTier: null,
    isDefault: false,
    ...overrides,
  };
}

function createProviderInfo(overrides: Record<string, unknown> = {}) {
  return {
    id: "openai",
    name: "OpenAI",
    providerType: "openai",
    apiHost: "https://api.openai.com",
    group: "global",
    enabled: true,
    isSystem: true,
    sortOrder: 1,
    apiVersion: null,
    project: null,
    location: null,
    region: null,
    models: [],
    promptCacheMode: null,
    apiKeyCount: 0,
    apiKeys: [],
    legacyIds: [],
    createdAt: null,
    updatedAt: null,
    ...overrides,
  };
}

describe("modelRegistry API", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    appServerRequestMock.mockReset();
    invalidateModelRegistryCache();
  });

  it("readModelProviderCapabilities 应通过 exact App Server method 读取布尔能力", async () => {
    resolveAppServerRequest({
      namespaceTools: true,
      imageGeneration: false,
      webSearch: true,
    });

    await expect(readModelProviderCapabilities()).resolves.toEqual({
      namespaceTools: true,
      imageGeneration: false,
      webSearch: true,
    });
    expectAppServerRequest(1, "modelProvider/capabilities/read", {});
  });

  it("readModelProviderCapabilities 遇到未知响应时 fail closed", async () => {
    resolveAppServerRequest({
      namespaceTools: true,
      imageGeneration: "unknown",
      webSearch: true,
    });

    await expect(readModelProviderCapabilities()).resolves.toEqual({
      namespaceTools: false,
      imageGeneration: false,
      webSearch: false,
    });
  });

  it("getModelRegistry 应缓存并复用同一轮读取结果", async () => {
    resolveAppServerRequest({
      data: [createCatalogModel()],
      nextCursor: null,
    });

    const [first, second] = await Promise.all([
      getModelRegistry(),
      getModelRegistry(),
    ]);

    expect(appServerRequestMock).toHaveBeenCalledTimes(1);
    expectAppServerRequest(1, "model/list", {});
    expect(safeInvoke).not.toHaveBeenCalled();
    expect(first).toEqual(second);
    expect(first).not.toBe(second);
  });

  it("getModelRegistry 应聚合 model/list 的全部分页", async () => {
    resolveAppServerRequest({
      data: [createCatalogModel("openai", "gpt-5.6-sol")],
      nextCursor: "1",
    });
    resolveAppServerRequest({
      data: [createCatalogModel("anthropic", "claude-sonnet-4")],
      nextCursor: null,
    });

    await expect(getModelRegistry()).resolves.toEqual([
      expect.objectContaining({ id: "gpt-5.6-sol", provider_id: "openai" }),
      expect.objectContaining({
        id: "claude-sonnet-4",
        provider_id: "anthropic",
      }),
    ]);
    expectAppServerRequest(1, "model/list", {});
    expectAppServerRequest(2, "model/list", { cursor: "1" });
  });

  it("model/list 的 provider 与 opaque route 不一致时应 fail closed", async () => {
    resolveAppServerRequest({
      data: [
        createCatalogModel("openai", "gpt-5.6-sol", {
          providerId: "anthropic",
        }),
      ],
      nextCursor: null,
    });

    await expect(getModelRegistry()).rejects.toThrow(
      "App Server model/list returned mismatched provider",
    );
  });

  it("getModelRegistry 遇到重复 cursor 时应 fail closed", async () => {
    resolveAppServerRequest({ data: [], nextCursor: "1" });
    resolveAppServerRequest({ data: [], nextCursor: "1" });

    await expect(getModelRegistry()).rejects.toThrow(
      "App Server model/list repeated cursor: 1",
    );
  });

  it("getModelRegistry 应隔离默认目录与 includeHidden 缓存", async () => {
    resolveAppServerRequest({
      data: [createCatalogModel("openai", "visible")],
      nextCursor: null,
    });
    resolveAppServerRequest({
      data: [
        createCatalogModel("openai", "visible"),
        createCatalogModel("openai", "hidden", { hidden: true }),
      ],
      nextCursor: null,
    });

    await expect(getModelRegistry()).resolves.toHaveLength(1);
    await expect(
      getModelRegistry({ includeHidden: true }),
    ).resolves.toHaveLength(2);
    await expect(getModelRegistry()).resolves.toHaveLength(1);

    expect(appServerRequestMock).toHaveBeenCalledTimes(2);
    expectAppServerRequest(1, "model/list", {});
    expectAppServerRequest(2, "model/list", { includeHidden: true });
  });

  it("model/list 应把 App Server Codex policy 字段归一到 registry metadata", async () => {
    resolveAppServerRequest({
      data: [
        createCatalogModel("openai", "gpt-5.6-sol", {
          model: "gpt-5.6-sol-wire",
          displayName: "GPT-5.6 Sol",
          description: "Frontier coding model",
          hidden: false,
          isDefault: true,
          serviceTiers: [
            {
              id: "default",
              name: "Default",
              description: "Standard routing",
            },
            {
              id: "flex",
              name: "Flex",
              description: "Lower priority routing",
            },
          ],
          defaultServiceTier: "flex",
          defaultReasoningEffort: "high",
          supportedReasoningEfforts: [
            {
              reasoningEffort: "low",
              description: "Fast",
            },
            {
              reasoningEffort: "high",
              description: "Deep",
            },
          ],
          inputModalities: ["text", "image", "video", "file"],
          capabilitySnapshot: {
            taskFamilies: ["chat", "reasoning", "vision_understanding"],
            inputModalities: ["text", "image", "video", "file"],
            outputModalities: ["text", "json"],
            runtimeFeatures: [
              "streaming",
              "tool_calling",
              "json_schema",
              "reasoning",
            ],
            capabilities: {
              vision: true,
              tools: true,
              streaming: true,
              jsonMode: true,
              functionCalling: true,
              reasoning: true,
              reasoningEffort: {
                supported: true,
                levels: ["low", "high"],
                options: [
                  {
                    id: "low",
                    value: "low",
                    label: "Fast",
                    description: "Fast",
                  },
                  {
                    id: "high",
                    value: "high",
                    label: "Deep",
                    description: "Deep",
                  },
                ],
                default: "high",
                source: "api",
              },
            },
            source: "provider_explicit",
            reasonCode: null,
          },
          contextWindow: 400_000,
          maxOutputTokens: 128_000,
          multiAgentVersion: "v2",
        }),
      ],
      nextCursor: null,
    });

    const models = await getModelRegistry();

    expect(models).toEqual([
      expect.objectContaining({
        id: "gpt-5.6-sol",
        provider_id: "openai",
        is_default: true,
        canonical_model_id: "gpt-5.6-sol",
        provider_model_id: "gpt-5.6-sol-wire",
        picker_policy: {
          visibility: "list",
          show_in_picker: true,
          service_tiers: [
            {
              id: "default",
              name: "Default",
              description: "Standard routing",
            },
            {
              id: "flex",
              name: "Flex",
              description: "Lower priority routing",
            },
          ],
          supported_service_tier_ids: ["default", "flex"],
          default_service_tier: "flex",
        },
        reasoning_policy: {
          supports_reasoning_summaries: false,
          default_reasoning_level: "high",
          supported_reasoning_levels: [
            {
              id: "low",
              label: "Fast",
              value: "low",
              description: "Fast",
            },
            {
              id: "high",
              label: "Deep",
              value: "high",
              description: "Deep",
            },
          ],
          supported_reasoning_efforts: ["low", "high"],
          can_set_reasoning_effort: true,
        },
        capabilities: expect.objectContaining({
          vision: true,
          tools: true,
          streaming: true,
          json_mode: true,
          function_calling: true,
          reasoning: true,
          reasoning_effort: {
            supported: true,
            levels: ["low", "high"],
            options: [
              {
                id: "low",
                value: "low",
                label: "Fast",
                description: "Fast",
              },
              {
                id: "high",
                value: "high",
                label: "Deep",
                description: "Deep",
              },
            ],
            default: "high",
            source: "api",
          },
        }),
        input_modality_policy: {
          input_modalities: ["text", "image", "video", "file"],
          send_gate_modalities: ["text", "image", "video", "file"],
          unknown_input_modalities: [],
          supports_text_input: true,
          supports_media_input: true,
          supports_image_input: true,
          source: "explicit",
        },
        task_families: ["chat", "reasoning", "vision_understanding"],
        input_modalities: ["text", "image", "video", "file"],
        output_modalities: ["text", "json"],
        runtime_features: [
          "streaming",
          "tool_calling",
          "json_schema",
          "reasoning",
        ],
        multi_agent_version: "v2",
        limits: {
          context_length: 400_000,
          max_output_tokens: 128_000,
          requests_per_minute: null,
          tokens_per_minute: null,
        },
      }),
    ]);
    const [model] = models;
    expect(model.capability_provenance).toBe("provider_explicit");
    expect(model).not.toHaveProperty("execution_policy");
    expect(model).not.toHaveProperty("context_policy");
    expect(model).not.toHaveProperty("tool_call_policy");
    expect(model).not.toHaveProperty("reasoning_output_policy");
    expect(model).not.toHaveProperty("responses_policy");
    expect(model).not.toHaveProperty("truncation_policy");
    expect(model).not.toHaveProperty("native_tool_policy");

    expectAppServerRequest(1, "model/list", {});
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("getProviderAliasConfig 应复用已加载的全量别名配置", async () => {
    resolveAppServerRequest({
      configs: {
        "custom-provider": {
          models: ["kimi-k2"],
          aliases: {
            "kimi-k2": {
              actual: "kimi-k2",
            },
          },
        },
      },
    });

    await expect(getAllAliasConfigs()).resolves.toEqual(
      expect.objectContaining({
        "custom-provider": expect.objectContaining({
          models: ["kimi-k2"],
        }),
      }),
    );
    await expect(getProviderAliasConfig("custom-provider")).resolves.toEqual(
      expect.objectContaining({ models: ["kimi-k2"] }),
    );

    expect(appServerRequestMock).toHaveBeenCalledTimes(1);
    expectAppServerRequest(1, "modelProviderAlias/list", {});
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("refreshModelRegistry 后应失效缓存并触发下一次重新读取", async () => {
    resolveAppServerRequest({
      data: [createCatalogModel()],
      nextCursor: null,
    });
    resolveAppServerRequest({
      data: [createCatalogModel("openai", "gpt-5", { displayName: "GPT-5" })],
      nextCursor: null,
    });

    await getModelRegistry();
    await expect(refreshModelRegistry()).resolves.toBe(1);
    await expect(getModelRegistry()).resolves.toEqual([
      expect.objectContaining({ id: "gpt-5" }),
    ]);

    expectAppServerRequest(1, "model/list", {});
    expectAppServerRequest(2, "model/list", {});
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("model/list/updated 应失效缓存并投影 typed 更新", async () => {
    let onNotifications: AppServerEventBusSubscription["onNotifications"];
    const unsubscribe = vi.fn();
    subscribeAppServerNotificationsMock.mockImplementationOnce(
      (subscription) => {
        onNotifications = subscription.onNotifications;
        return unsubscribe;
      },
    );
    resolveAppServerRequest({
      data: [createCatalogModel("openai", "gpt-4.1")],
      nextCursor: null,
    });
    resolveAppServerRequest({
      data: [createCatalogModel("openai", "gpt-5")],
      nextCursor: null,
    });
    await getModelRegistry();
    const onUpdate = vi.fn();
    const stop = subscribeModelRegistryUpdates(onUpdate);

    onNotifications?.([
      { method: "thread/started", params: { thread: { id: "thread-1" } } },
      {
        method: "model/list/updated",
        params: { generation: 17, providerId: "openai" },
      },
    ]);

    expect(onUpdate).toHaveBeenCalledWith({
      generation: 17,
      providerId: "openai",
    });
    await expect(getModelRegistry()).resolves.toEqual([
      expect.objectContaining({ id: "gpt-5" }),
    ]);
    stop();
    expect(unsubscribe).toHaveBeenCalledOnce();
    expect(appServerRequestMock).toHaveBeenCalledTimes(2);
  });

  it("searchModels 应基于 App Server current 模型列表做前端过滤", async () => {
    resolveAppServerRequest({
      data: [
        createCatalogModel("openai", "gpt-4.1", { displayName: "GPT-4.1" }),
        createCatalogModel("anthropic", "claude-sonnet-4", {
          displayName: "Claude Sonnet 4",
        }),
      ],
      nextCursor: null,
    });

    await expect(searchModels("gpt", 1)).resolves.toEqual([
      expect.objectContaining({ id: "gpt-4.1" }),
    ]);

    expectAppServerRequest(1, "model/list", {});
    expect(safeInvoke).not.toHaveBeenCalledWith("search_models", {
      query: "gpt",
      limit: 1,
    });
  });

  it("模型偏好与同步状态读取应走 App Server current", async () => {
    resolveAppServerRequest({
      preferences: [
        {
          model_id: "gpt-4.1",
          is_favorite: true,
          is_hidden: false,
          custom_alias: null,
          usage_count: 3,
          last_used_at: null,
          created_at: 1,
          updated_at: 2,
        },
      ],
    });
    resolveAppServerRequest({
      syncState: {
        last_sync_at: 1,
        model_count: 2,
        is_syncing: false,
        last_error: null,
      },
    });
    await expect(getModelPreferences()).resolves.toEqual([
      expect.objectContaining({ model_id: "gpt-4.1" }),
    ]);
    await expect(getModelSyncState()).resolves.toEqual(
      expect.objectContaining({ model_count: 2 }),
    );
    expectAppServerRequest(1, "modelPreferences/list", {});
    expectAppServerRequest(2, "modelSyncState/read", {});
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("单个 provider alias 应通过 App Server 读取并缓存", async () => {
    resolveAppServerRequest({
      config: {
        provider: "custom-provider",
        models: ["kimi-k2"],
        aliases: {},
      },
    });

    await expect(getProviderAliasConfig("custom-provider")).resolves.toEqual(
      expect.objectContaining({ provider: "custom-provider" }),
    );
    await expect(getProviderAliasConfig("custom-provider")).resolves.toEqual(
      expect.objectContaining({ provider: "custom-provider" }),
    );

    expect(appServerRequestMock).toHaveBeenCalledTimes(1);
    expectAppServerRequest(1, "modelProviderAlias/read", {
      provider: "custom-provider",
    });
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("Provider 实时模型抓取应通过 App Server current", async () => {
    resolveAppServerRequest({
      models: [createModelInfo()],
      source: "Api",
      error: null,
      requestUrl: "https://api.openai.com/v1/models",
      diagnosticHint: null,
      errorKind: null,
      shouldPromptError: false,
      fromCache: true,
    });

    await expect(fetchProviderModelsAuto("openai")).resolves.toEqual(
      expect.objectContaining({
        source: "Api",
        request_url: "https://api.openai.com/v1/models",
        from_cache: true,
        models: [
          expect.objectContaining({
            id: "gpt-4.1",
            capability_provenance: "provider_explicit",
          }),
        ],
      }),
    );

    expectAppServerRequest(1, "modelProvider/fetchModels", {
      providerId: "openai",
    });
    expect(safeInvoke).not.toHaveBeenCalledWith("fetch_provider_models_auto");
  });

  it("Provider 实时目录应保留 inferred_hint 能力来源", async () => {
    resolveAppServerRequest({
      models: [
        createModelInfo({
          id: "unknown-model",
          capabilityProvenance: "inferred_hint",
        }),
      ],
      source: "Api",
      error: null,
      shouldPromptError: false,
      fromCache: true,
    });

    await expect(fetchProviderModelsAuto("custom-provider")).resolves.toEqual(
      expect.objectContaining({
        models: [
          expect.objectContaining({
            id: "unknown-model",
            capability_provenance: "inferred_hint",
          }),
        ],
      }),
    );
  });

  it("App Server 模型读链缺少必需 result 时不应回退 legacy", async () => {
    resolveAppServerRequest({});
    await expect(getModelRegistry()).rejects.toThrow(
      "App Server model/list did not return data",
    );

    appServerRequestMock.mockReset();
    resolveAppServerRequest({});
    await expect(getModelPreferences()).rejects.toThrow(
      "App Server modelPreferences/list did not return preferences",
    );

    appServerRequestMock.mockReset();
    resolveAppServerRequest({});
    await expect(getModelSyncState()).rejects.toThrow(
      "App Server modelSyncState/read did not return syncState",
    );

    appServerRequestMock.mockReset();
    resolveAppServerRequest({});
    await expect(getAllAliasConfigs()).rejects.toThrow(
      "App Server modelProviderAlias/list did not return configs",
    );

    expect(safeInvoke).not.toHaveBeenCalledWith("get_model_registry");
    expect(safeInvoke).not.toHaveBeenCalledWith("get_model_preferences");
    expect(safeInvoke).not.toHaveBeenCalledWith("get_model_sync_state");
    expect(safeInvoke).not.toHaveBeenCalledWith("get_all_alias_configs");
  });

  it("getModelRegistryProviderIds 应通过 App Server provider list 派生去重 id", async () => {
    resolveAppServerRequest({
      providers: [
        createProviderInfo(),
        createProviderInfo({
          id: "anthropic",
          name: "Anthropic",
          providerType: "anthropic",
        }),
        createProviderInfo({ id: "openai", name: "OpenAI duplicate" }),
        createProviderInfo({ id: "", name: "invalid empty" }),
      ],
    });

    await expect(getModelRegistryProviderIds()).resolves.toEqual([
      "openai",
      "anthropic",
    ]);

    expectAppServerRequest(1, "modelProvider/list", {});
    expect(safeInvoke).not.toHaveBeenCalledWith(
      "get_model_registry_provider_ids",
    );
  });

  it("getModelRegistryProviderIds 缺少 App Server providers 时应 fail closed", async () => {
    resolveAppServerRequest({});

    await expect(getModelRegistryProviderIds()).rejects.toThrow(
      "App Server modelProvider/list did not return providers",
    );
    expect(safeInvoke).not.toHaveBeenCalledWith(
      "get_model_registry_provider_ids",
    );
  });

  it("模型偏好写链缺少 App Server current owner 时应 fail closed", async () => {
    await expect(toggleModelFavorite("gpt-4.1")).rejects.toThrow(
      "toggleModelFavorite 尚未接入 App Server model preference current 写链；旧 Tauri 模型注册表业务命令已退役。",
    );
    await expect(hideModel("gpt-4.1")).rejects.toThrow(
      "hideModel 尚未接入 App Server model preference current 写链；旧 Tauri 模型注册表业务命令已退役。",
    );
    await expect(recordModelUsage("gpt-4.1")).rejects.toThrow(
      "recordModelUsage 尚未接入 App Server model preference current 写链；旧 Tauri 模型注册表业务命令已退役。",
    );
    expect(appServerRequestMock).not.toHaveBeenCalled();
    expect(safeInvoke).not.toHaveBeenCalled();
  });
});
