import { describe, expect, it } from "vitest";
import type { ProviderWithKeysDisplay } from "@/lib/api/apiKeyProvider";
import {
  buildConfiguredProviders,
  findConfiguredProviderBySelection,
  resolveConfiguredProviderPromptCacheSupportNotice,
} from "./useConfiguredProviders";
import type { OemCloudRuntimeContext } from "@/lib/api/oemCloudRuntime";

function createApiKeyProvider(
  overrides: Partial<ProviderWithKeysDisplay> = {},
): ProviderWithKeysDisplay {
  const apiKeyCount = overrides.api_key_count ?? 0;
  return {
    id: "openai",
    name: "OpenAI",
    type: "openai",
    api_host: "https://api.openai.com/v1",
    is_system: true,
    group: "cloud",
    enabled: true,
    sort_order: 0,
    custom_models: [],
    prompt_cache_mode: null,
    api_key_count: apiKeyCount,
    api_keys:
      apiKeyCount > 0
        ? [
            {
              id: "key-1",
              provider_id: overrides.id ?? "openai",
              api_key_masked: "sk-***",
              enabled: true,
              usage_count: 0,
              error_count: 0,
              created_at: "2026-04-01T00:00:00Z",
            },
          ]
        : [],
    created_at: "2026-04-01T00:00:00Z",
    updated_at: "2026-04-01T00:00:00Z",
    ...overrides,
  };
}

function createOemRuntime(
  overrides: Partial<OemCloudRuntimeContext> = {},
): OemCloudRuntimeContext {
  return {
    baseUrl: "https://user.limeai.run",
    controlPlaneBaseUrl: "https://user.limeai.run",
    sceneBaseUrl: "https://scene.limeai.run",
    gatewayBaseUrl: "https://llm.limeai.run",
    tenantId: "tenant-0001",
    sessionToken: null,
    hubProviderName: "Lime 云端",
    loginPath: "/login",
    desktopClientId: "desktop-client",
    desktopOauthRedirectUrl: "lime://oauth/callback",
    desktopOauthNextPath: "/welcome",
    pluginSignatureTrustRoots: [],
    ...overrides,
  };
}

describe("buildConfiguredProviders", () => {
  it("应将无 Key 但已启用且地址有效的 Ollama 视为已配置渠道", () => {
    const providers = buildConfiguredProviders([
      createApiKeyProvider({
        id: "ollama",
        name: "Ollama (本地)",
        type: "ollama",
        api_host: "http://127.0.0.1:11434",
        api_key_count: 0,
      }),
    ]);

    expect(providers).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          key: "ollama",
          providerId: "ollama",
          type: "ollama",
          apiHost: "http://127.0.0.1:11434",
          promptCacheMode: null,
        }),
      ]),
    );
  });

  it("不应误把其他无 Key 云渠道当成已配置", () => {
    const providers = buildConfiguredProviders([
      createApiKeyProvider({
        id: "openai",
        name: "OpenAI",
        type: "openai",
        api_key_count: 0,
      }),
    ]);

    expect(providers).toEqual([]);
  });

  it("API Key 计数非零但所有 Key 已禁用时不应展示", () => {
    const providers = buildConfiguredProviders([
      createApiKeyProvider({
        id: "openai-disabled",
        api_key_count: 1,
        api_keys: [
          {
            id: "key-disabled",
            provider_id: "openai-disabled",
            api_key_masked: "sk-***",
            enabled: false,
            usage_count: 0,
            error_count: 0,
            created_at: "2026-04-01T00:00:00Z",
          },
        ],
      }),
    ]);

    expect(providers).toEqual([]);
  });

  it("OEM Runtime 未登录时应展示 Lime Hub 登录提示 Provider", () => {
    const providers = buildConfiguredProviders(
      [
        createApiKeyProvider({
          id: "lime-hub",
          name: "Lime 云端",
          type: "openai",
          api_host: "https://llm.limeai.run",
          api_key_count: 0,
          custom_models: [],
        }),
      ],
      {
        oemRuntime: createOemRuntime({ sessionToken: null }),
      },
    );

    expect(providers).toEqual([
      expect.objectContaining({
        key: "lime-hub",
        providerId: "lime-hub",
        label: "Lime 云端",
        authStatus: "login_required",
        customModels: [],
      }),
    ]);
  });

  it("OEM Runtime 未登录但托管网关已有声明模型时应作为可选 Provider", () => {
    const providers = buildConfiguredProviders(
      [
        createApiKeyProvider({
          id: "lime-hub",
          name: "Lime 云端",
          type: "openai",
          api_host: "https://llm.limeai.run#lime_tenant_id=tenant-0001",
          api_key_count: 0,
          custom_models: ["agnes-2.0-flash"],
        }),
      ],
      {
        oemRuntime: createOemRuntime({ sessionToken: null }),
      },
    );

    expect(providers).toEqual([
      expect.objectContaining({
        key: "lime-hub",
        providerId: "lime-hub",
        label: "Lime 云端",
        authStatus: "ready",
        customModels: ["agnes-2.0-flash"],
      }),
    ]);
  });

  it("后端未返回 Lime Hub 但 OEM Runtime 未登录时应合成登录提示 Provider", () => {
    const providers = buildConfiguredProviders([], {
      oemRuntime: createOemRuntime({ sessionToken: null }),
    });

    expect(providers).toEqual([
      expect.objectContaining({
        key: "lime-hub",
        providerId: "lime-hub",
        label: "Lime 云端",
        authStatus: "login_required",
        apiHost: "https://llm.limeai.run#lime_tenant_id=tenant-0001",
      }),
    ]);
  });

  it("OEM Runtime 已登录但无 Key 时不应合成 Lime Hub 登录提示 Provider", () => {
    const providers = buildConfiguredProviders([], {
      oemRuntime: createOemRuntime({ sessionToken: "session-token" }),
    });

    expect(providers).toEqual([]);
  });

  it("OEM Runtime 已登录且后端返回 tenant Lime Hub 时应视为已配置 Provider", () => {
    const providers = buildConfiguredProviders(
      [
        createApiKeyProvider({
          id: "lime-hub",
          name: "Lime Hub",
          type: "openai",
          api_host: "https://llm.limeai.run#lime_tenant_id=tenant-0001",
          api_key_count: 0,
          custom_models: [],
        }),
      ],
      {
        oemRuntime: createOemRuntime({ sessionToken: "session-token" }),
      },
    );

    expect(providers).toEqual([
      expect.objectContaining({
        key: "lime-hub",
        providerId: "lime-hub",
        label: "Lime Hub",
        authStatus: "ready",
        customModels: ["gpt-5.2-pro"],
      }),
    ]);
  });

  it("tenant Lime Hub 后端暂未返回模型目录时应使用默认聊天模型", () => {
    const providers = buildConfiguredProviders(
      [
        createApiKeyProvider({
          id: "lime-hub",
          name: "Lime Hub",
          type: "openai",
          api_host: "https://llm.limeai.run#lime_tenant_id=tenant-0001",
          api_key_count: 0,
          custom_models: [],
        }),
      ],
      {
        oemRuntime: createOemRuntime({ sessionToken: null }),
      },
    );

    expect(providers).toEqual([
      expect.objectContaining({
        key: "lime-hub",
        providerId: "lime-hub",
        authStatus: "ready",
        customModels: ["gpt-5.2-pro"],
      }),
    ]);
  });

  it("无 Key 的 Ollama 缺少地址或被禁用时不应展示", () => {
    const missingHostProviders = buildConfiguredProviders([
      createApiKeyProvider({
        id: "ollama",
        name: "Ollama (本地)",
        type: "ollama",
        api_host: "   ",
        api_key_count: 0,
      }),
    ]);
    const disabledProviders = buildConfiguredProviders([
      createApiKeyProvider({
        id: "ollama",
        name: "Ollama (本地)",
        type: "ollama",
        api_host: "http://127.0.0.1:11434",
        enabled: false,
        api_key_count: 0,
      }),
    ]);

    expect(missingHostProviders).toEqual([]);
    expect(disabledProviders).toEqual([]);
  });

  it("后端返回原始 providerId 时，应命中真实受管 Provider", () => {
    const providers = buildConfiguredProviders([
      createApiKeyProvider({
        id: "openai",
        name: "OpenAI API Key",
        api_key_count: 1,
      }),
    ]);

    const resolvedProvider = findConfiguredProviderBySelection(
      providers,
      "openai",
    );

    expect(resolvedProvider).toEqual(
      expect.objectContaining({
        key: "openai",
        providerId: "openai",
      }),
    );
  });

  it("旧版 MiMo 选择值也应命中后端受管的 Xiaomi Provider", () => {
    const providers = buildConfiguredProviders([
      createApiKeyProvider({
        id: "xiaomi",
        name: "小米 MiMo",
        type: "openai",
        api_key_count: 1,
      }),
    ]);

    const resolvedProvider = findConfiguredProviderBySelection(
      providers,
      "mimo",
    );

    expect(resolvedProvider).toEqual(
      expect.objectContaining({
        key: "xiaomi",
        providerId: "xiaomi",
      }),
    );
  });

  it("Provider 列表短暂缺失时不应让设置页选择器崩溃", () => {
    expect(findConfiguredProviderBySelection(undefined, "openai")).toBeNull();
    expect(() =>
      resolveConfiguredProviderPromptCacheSupportNotice(undefined, "openai"),
    ).not.toThrow();
  });

  it("应基于真实受管 Provider 解析 prompt cache 提示", () => {
    const providers = buildConfiguredProviders([
      createApiKeyProvider({
        id: "custom-provider-id",
        name: "GLM Anthropic",
        type: "anthropic-compatible",
        api_key_count: 1,
      }),
    ]);

    const notice = resolveConfiguredProviderPromptCacheSupportNotice(
      providers,
      "custom-provider-id",
    );

    expect(notice).toEqual(
      expect.objectContaining({
        label: "未声明自动缓存",
        source: "configured_provider",
      }),
    );
  });

  it("显式声明 automatic 的 anthropic-compatible Provider 不应误报 prompt cache 提示", () => {
    const providers = buildConfiguredProviders([
      createApiKeyProvider({
        id: "glm-anthropic",
        name: "GLM Anthropic",
        type: "anthropic-compatible",
        prompt_cache_mode: "automatic",
        api_key_count: 1,
      }),
    ]);

    const notice = resolveConfiguredProviderPromptCacheSupportNotice(
      providers,
      "glm-anthropic",
    );

    expect(notice).toBeNull();
  });

  it.each([
    {
      id: "glm-anthropic",
      name: "GLM Anthropic",
      apiHost: "https://open.bigmodel.cn/api/anthropic",
    },
    {
      id: "zai-anthropic",
      name: "Z.AI Anthropic",
      apiHost: "https://api.z.ai/api/anthropic",
    },
    {
      id: "kimi-anthropic",
      name: "Kimi Anthropic",
      apiHost: "https://api.moonshot.cn/anthropic",
    },
    {
      id: "kimi-global-anthropic",
      name: "Kimi Global Anthropic",
      apiHost: "https://api.moonshot.ai/anthropic",
    },
    {
      id: "kimi-code-subscription",
      name: "Kimi Code Subscription",
      apiHost: "https://api.kimi.com/coding/",
    },
    {
      id: "minimax-anthropic",
      name: "MiniMax Anthropic",
      apiHost: "https://api.minimaxi.com/anthropic",
    },
    {
      id: "minimax-global-anthropic",
      name: "MiniMax Global Anthropic",
      apiHost: "https://api.minimax.io/anthropic",
    },
    {
      id: "alibaba-coding-anthropic",
      name: "Alibaba Coding Anthropic",
      apiHost: "https://coding.dashscope.aliyuncs.com/apps/anthropic",
    },
    {
      id: "alibaba-global-coding-anthropic",
      name: "Alibaba Global Coding Anthropic",
      apiHost: "https://coding-intl.dashscope.aliyuncs.com/apps/anthropic",
    },
    {
      id: "mimo-anthropic",
      name: "MiMo Anthropic",
      apiHost: "https://token-plan-cn.xiaomimimo.com/anthropic",
    },
  ])(
    "$name 官方 Anthropic 兼容 Host 不应误报 prompt cache 提示",
    ({ id, name, apiHost }) => {
      const providers = buildConfiguredProviders([
        createApiKeyProvider({
          id,
          name,
          type: "anthropic-compatible",
          api_host: apiHost,
          api_key_count: 1,
        }),
      ]);

      const notice = resolveConfiguredProviderPromptCacheSupportNotice(
        providers,
        id,
      );

      expect(notice).toBeNull();
    },
  );
});
