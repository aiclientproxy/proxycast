import { beforeEach, describe, expect, it, vi } from "vitest";
import { safeInvoke } from "@/lib/dev-bridge";
import {
  apiKeyProviderApi,
  invalidateApiKeyProviderCache,
  providerModelIds,
  reconcileProviderModels,
} from "./apiKeyProvider";

const appServerRequestMock = vi.hoisted(() => vi.fn());

vi.mock("@/lib/api/appServer", () => ({
  AppServerClient: vi.fn(() => ({
    request: appServerRequestMock,
  })),
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
    apiKeyCount: 1,
    apiKeys: [createProviderKeyInfo()],
    legacyIds: [],
    createdAt: "2026-06-17T00:00:00Z",
    updatedAt: "2026-06-17T00:00:00Z",
    ...overrides,
  };
}

function createProviderKeyInfo(overrides: Record<string, unknown> = {}) {
  return {
    id: "key-1",
    providerId: "openai",
    apiKeyMasked: "sk-test****1234",
    alias: null,
    enabled: true,
    usageCount: 0,
    errorCount: 0,
    lastUsedAt: null,
    createdAt: "2026-06-17T00:00:00Z",
    ...overrides,
  };
}

describe("apiKeyProvider API", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    appServerRequestMock.mockReset();
    invalidateApiKeyProviderCache();
  });

  it("模型 ID 投影与编辑应保留已有 typed capability", () => {
    const existingModels = [
      {
        id: "grok-4.5",
        displayName: "Grok 4.5",
        capability: {
          taskFamilies: ["chat", "reasoning"],
          inputModalities: ["text", "image"],
          outputModalities: ["text"],
          runtimeFeatures: ["streaming", "tool_calling"],
          capabilities: {
            vision: true,
            tools: true,
            streaming: true,
            jsonMode: true,
            functionCalling: true,
            reasoning: true,
            reasoningEffort: {
              supported: true,
              levels: ["medium", "xhigh"],
              options: [
                {
                  id: "deep",
                  value: "xhigh",
                  label: "Deep",
                  default: true,
                },
              ],
              default: "xhigh",
              source: "api",
            },
          },
        },
      },
    ];

    expect(providerModelIds(existingModels)).toEqual(["grok-4.5"]);
    const reconciled = reconcileProviderModels(existingModels, [
      " GROK-4.5 ",
      "new-model",
      "new-model",
    ]);

    expect(reconciled).toEqual([
      { ...existingModels[0], id: "grok-4.5" },
      { id: "new-model" },
    ]);
    expect(reconciled[0]).not.toBe(existingModels[0]);
    expect(reconciled[0].capability).not.toBe(existingModels[0].capability);
    expect(reconciled[0].capability?.capabilities.reasoningEffort).not.toBe(
      existingModels[0].capability.capabilities.reasoningEffort,
    );
  });

  it("Provider 列表应通过 App Server modelProvider/list 读取", async () => {
    resolveAppServerRequest({
      providers: [createProviderInfo()],
    });

    await expect(apiKeyProviderApi.getProviders()).resolves.toEqual([
      expect.objectContaining({ id: "openai" }),
    ]);

    expectAppServerRequest(1, "modelProvider/list", {});
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("系统 Provider Catalog 应通过 App Server modelProvider/catalog/list 读取", async () => {
    resolveAppServerRequest({
      providers: [createProviderInfo()],
    });

    await expect(apiKeyProviderApi.getSystemProviderCatalog()).resolves.toEqual(
      [expect.objectContaining({ id: "openai" })],
    );

    expectAppServerRequest(1, "modelProvider/catalog/list", {});
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("Provider 读链缺少必需 result 时不应回退 legacy", async () => {
    resolveAppServerRequest({});
    await expect(apiKeyProviderApi.getProviders()).rejects.toThrow(
      "App Server modelProvider/list did not return providers",
    );

    appServerRequestMock.mockReset();
    resolveAppServerRequest({});
    await expect(apiKeyProviderApi.getSystemProviderCatalog()).rejects.toThrow(
      "App Server modelProvider/catalog/list did not return providers",
    );

    expect(safeInvoke).not.toHaveBeenCalledWith("get_api_key_providers");
    expect(safeInvoke).not.toHaveBeenCalledWith("get_system_provider_catalog");
  });

  it("应通过 App Server current 执行 provider side-effect 命令", async () => {
    resolveAppServerRequest({
      key: createProviderKeyInfo(),
    });
    resolveAppServerRequest({
      success: true,
      latencyMs: 12,
    });

    await expect(
      apiKeyProviderApi.addApiKey({
        provider_id: "openai",
        api_key: "sk-test",
        replace_existing: true,
      }),
    ).resolves.toEqual(expect.objectContaining({ id: "key-1" }));
    await expect(
      apiKeyProviderApi.testConnection("openai", "gpt-4.1"),
    ).resolves.toEqual(
      expect.objectContaining({ success: true, latency_ms: 12 }),
    );
    expectAppServerRequest(1, "modelProviderKey/create", {
      providerId: "openai",
      apiKey: "sk-test",
      alias: undefined,
      replaceExisting: true,
    });
    expectAppServerRequest(2, "modelProvider/testConnection", {
      providerId: "openai",
      modelName: "gpt-4.1",
    });
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("Provider create/update 应发送 typed App Server 参数", async () => {
    resolveAppServerRequest({
      provider: createProviderInfo({
        id: "custom",
        name: "Custom",
        providerType: "openai-compatible",
        apiHost: "https://api.example.com/v1",
        isSystem: false,
        apiKeyCount: 0,
        apiKeys: [],
      }),
    });
    resolveAppServerRequest({
      provider: createProviderInfo({
        id: "custom",
        name: "Custom Renamed",
        providerType: "openai-compatible",
        apiHost: "https://api.example.com/v2",
        isSystem: false,
        apiKeyCount: 0,
        models: [{ id: "custom-model" }],
        apiKeys: [],
      }),
    });

    await expect(
      apiKeyProviderApi.addCustomProvider({
        name: "Custom",
        type: "openai-compatible",
        api_host: "https://api.example.com/v1",
        prompt_cache_mode: "automatic",
      }),
    ).resolves.toEqual(expect.objectContaining({ id: "custom" }));
    await expect(
      apiKeyProviderApi.updateProvider("custom", {
        name: "Custom Renamed",
        api_host: "https://api.example.com/v2",
        models: [{ id: "custom-model" }],
      }),
    ).resolves.toEqual(
      expect.objectContaining({ models: [{ id: "custom-model" }] }),
    );

    expectAppServerRequest(1, "modelProvider/create", {
      name: "Custom",
      providerType: "openai-compatible",
      apiHost: "https://api.example.com/v1",
      apiVersion: undefined,
      project: undefined,
      location: undefined,
      region: undefined,
      promptCacheMode: "automatic",
    });
    expectAppServerRequest(2, "modelProvider/update", {
      providerId: "custom",
      name: "Custom Renamed",
      providerType: undefined,
      apiHost: "https://api.example.com/v2",
      enabled: undefined,
      sortOrder: undefined,
      apiVersion: undefined,
      project: undefined,
      location: undefined,
      region: undefined,
      promptCacheMode: undefined,
      models: [{ id: "custom-model" }],
    });
  });

  it("不应继续暴露旧 API Key 迁移 API", () => {
    expect("getLegacyApiKeyCredentials" in apiKeyProviderApi).toBe(false);
    expect("migrateLegacyCredentials" in apiKeyProviderApi).toBe(false);
    expect("deleteLegacyCredential" in apiKeyProviderApi).toBe(false);
  });

  it("getProviders 应缓存并复用同一轮读取结果", async () => {
    resolveAppServerRequest({
      providers: [createProviderInfo()],
    });

    const [first, second] = await Promise.all([
      apiKeyProviderApi.getProviders(),
      apiKeyProviderApi.getProviders(),
    ]);

    expect(appServerRequestMock).toHaveBeenCalledTimes(1);
    expectAppServerRequest(1, "modelProvider/list", {});
    expect(safeInvoke).not.toHaveBeenCalled();
    expect(first).toEqual(second);
    expect(first).not.toBe(second);
  });

  it("forceRefresh 应绕过 Provider 缓存", async () => {
    resolveAppServerRequest({
      providers: [createProviderInfo({ apiKeys: [] })],
    });
    resolveAppServerRequest({
      providers: [
        createProviderInfo({
          id: "deepseek",
          name: "DeepSeek",
          providerType: "deepseek",
          apiHost: "https://api.deepseek.com",
          apiKeyCount: 2,
          apiKeys: [],
        }),
      ],
    });

    await expect(apiKeyProviderApi.getProviders()).resolves.toEqual([
      expect.objectContaining({ id: "openai" }),
    ]);
    await expect(
      apiKeyProviderApi.getProviders({ forceRefresh: true }),
    ).resolves.toEqual([expect.objectContaining({ id: "deepseek" })]);

    expect(appServerRequestMock).toHaveBeenCalledTimes(2);
    expectAppServerRequest(1, "modelProvider/list", {});
    expectAppServerRequest(2, "modelProvider/list", {});
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("Provider 读取失败时不应注入本地 mock 或写入缓存", async () => {
    appServerRequestMock.mockRejectedValueOnce(
      new Error("App Server unavailable"),
    );

    await expect(
      apiKeyProviderApi.getProviders({ forceRefresh: true }),
    ).rejects.toThrow("App Server unavailable");

    resolveAppServerRequest({
      providers: [
        createProviderInfo({
          id: "custom-openai-images",
          name: "OpenAI-gpt-images-2",
          models: [{ id: "gpt-images-2" }],
          apiKeys: [],
        }),
      ],
    });

    await expect(apiKeyProviderApi.getProviders()).resolves.toEqual([
      expect.objectContaining({
        id: "custom-openai-images",
        models: [{ id: "gpt-images-2" }],
      }),
    ]);
    expect(appServerRequestMock).toHaveBeenCalledTimes(2);
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("写操作成功后应失效缓存", async () => {
    resolveAppServerRequest({
      providers: [createProviderInfo({ apiKeys: [] })],
    });
    resolveAppServerRequest({
      key: createProviderKeyInfo({ id: "key-2" }),
    });
    resolveAppServerRequest({
      providers: [
        createProviderInfo({
          apiKeyCount: 2,
          apiKeys: [createProviderKeyInfo({ id: "key-2" })],
        }),
      ],
    });

    await apiKeyProviderApi.getProviders();
    await apiKeyProviderApi.addApiKey({
      provider_id: "openai",
      api_key: "sk-test",
    });
    await expect(apiKeyProviderApi.getProviders()).resolves.toEqual([
      expect.objectContaining({ api_key_count: 2 }),
    ]);

    expect(appServerRequestMock).toHaveBeenCalledTimes(3);
    expectAppServerRequest(1, "modelProvider/list", {});
    expectAppServerRequest(2, "modelProviderKey/create", {
      providerId: "openai",
      apiKey: "sk-test",
      alias: undefined,
      replaceExisting: undefined,
    });
    expectAppServerRequest(3, "modelProvider/list", {});
    expect(safeInvoke).not.toHaveBeenCalled();
  });
});
