import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";
import {
  clearOemCloudBootstrapSnapshot,
  clearStoredOemCloudSessionState,
  setOemCloudBootstrapSnapshot,
  setStoredOemCloudSessionState,
} from "@/lib/oemCloudSession";

const apiKeyProviderMocks = vi.hoisted(() => ({
  addApiKey: vi.fn(),
  getProviders: vi.fn(),
  getUiState: vi.fn(),
  setUiState: vi.fn(),
  toggleApiKey: vi.fn(),
  updateProvider: vi.fn(),
}));

const controlPlaneMocks = vi.hoisted(() => ({
  createClientAccessToken: vi.fn(),
  listClientProviderOfferModels: vi.fn(),
}));

const desktopRuntimeMocks = vi.hoisted(() => ({
  hasDesktopHostInvokeCapability: vi.fn(() => true),
  hasDesktopHostRuntimeMarkers: vi.fn(() => false),
}));

function providerModels(...ids: string[]) {
  return ids.map((id) => ({ id }));
}

vi.mock("@/lib/api/apiKeyProvider", () => ({
  providerModelIds: (models: Array<{ id: string }> = []) =>
    models.map((model) => model.id),
  reconcileProviderModels: (
    existingModels: Array<{ id: string }> = [],
    modelIds: string[],
  ) => {
    const existingById = new Map(
      existingModels.map((model) => [model.id.toLowerCase(), model]),
    );
    return modelIds.map(
      (id) => existingById.get(id.toLowerCase()) ?? { id },
    );
  },
  apiKeyProviderApi: {
    addApiKey: apiKeyProviderMocks.addApiKey,
    getProviders: apiKeyProviderMocks.getProviders,
    getUiState: apiKeyProviderMocks.getUiState,
    setUiState: apiKeyProviderMocks.setUiState,
    toggleApiKey: apiKeyProviderMocks.toggleApiKey,
    updateProvider: apiKeyProviderMocks.updateProvider,
  },
}));

vi.mock("@/lib/api/oemCloudControlPlane", () => ({
  createClientAccessToken: controlPlaneMocks.createClientAccessToken,
  listClientProviderOfferModels:
    controlPlaneMocks.listClientProviderOfferModels,
}));

vi.mock("@/lib/desktop-runtime", () => ({
  hasDesktopHostInvokeCapability:
    desktopRuntimeMocks.hasDesktopHostInvokeCapability,
  hasDesktopHostRuntimeMarkers:
    desktopRuntimeMocks.hasDesktopHostRuntimeMarkers,
}));

import { useOemLimeHubProviderSync } from "./useOemLimeHubProviderSync";

interface MountedHarness {
  container: HTMLDivElement;
  root: Root;
}

function HookHarness() {
  useOemLimeHubProviderSync();
  return <div data-testid="oem-lime-hub-provider-sync" />;
}

async function flushEffects() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
  });
}

describe("useOemLimeHubProviderSync", () => {
  let mountedHarness: MountedHarness | null = null;

  beforeEach(async () => {
    await changeLimeLocale("en-US");
    (
      globalThis as typeof globalThis & {
        IS_REACT_ACT_ENVIRONMENT?: boolean;
      }
    ).IS_REACT_ACT_ENVIRONMENT = true;

    window.localStorage.clear();
    delete window.__LIME_SESSION_TOKEN__;
    delete window.__LIME_BOOTSTRAP__;
    window.__LIME_OEM_CLOUD__ = {
      enabled: true,
      baseUrl: "https://user.limeai.run",
      gatewayBaseUrl: "https://gateway-api.limeai.run/root",
      tenantId: "tenant-0001",
      hubProviderName: "Acme Hub",
    };

    apiKeyProviderMocks.getProviders.mockResolvedValue([
      {
        id: "lime-hub",
        name: "旧 Provider",
        api_host: "https://old-gateway.example.com",
        type: "anthropic",
        enabled: false,
        sort_order: 9,
        models: [],
        api_keys: [],
      },
    ]);
    apiKeyProviderMocks.addApiKey.mockResolvedValue({ id: "local-key-001" });
    apiKeyProviderMocks.getUiState.mockResolvedValue(null);
    apiKeyProviderMocks.setUiState.mockResolvedValue(undefined);
    apiKeyProviderMocks.toggleApiKey.mockResolvedValue({ id: "old-key-001" });
    apiKeyProviderMocks.updateProvider.mockResolvedValue(undefined);
    desktopRuntimeMocks.hasDesktopHostInvokeCapability.mockReturnValue(true);
    desktopRuntimeMocks.hasDesktopHostRuntimeMarkers.mockReturnValue(false);
    controlPlaneMocks.createClientAccessToken.mockResolvedValue({
      token: { id: "cloud-token-001" },
      apiKey: "sk-lime-desktop",
    });
    controlPlaneMocks.listClientProviderOfferModels.mockResolvedValue([
      {
        id: "model-001",
        modelId: "gpt-5.2-fast",
      },
      {
        id: "model-002",
        modelId: "gpt-5.2-pro",
      },
    ]);
  });

  afterEach(async () => {
    vi.clearAllMocks();
    clearStoredOemCloudSessionState();
    clearOemCloudBootstrapSnapshot();
    delete window.__LIME_OEM_CLOUD__;

    if (mountedHarness) {
      act(() => {
        mountedHarness?.root.unmount();
      });
      mountedHarness.container.remove();
      mountedHarness = null;
    }
    await changeLimeLocale("zh-CN");
  });

  it("Desktop Host 无 invoke 能力时不应触发 Provider 同步", async () => {
    desktopRuntimeMocks.hasDesktopHostInvokeCapability.mockReturnValue(false);
    desktopRuntimeMocks.hasDesktopHostRuntimeMarkers.mockReturnValue(true);
    setStoredOemCloudSessionState({
      token: "session-token-001",
      tenant: {
        id: "tenant-0001",
      },
      user: {
        id: "user-001",
      },
      session: {
        id: "session-001",
      },
    });

    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    mountedHarness = { container, root };

    act(() => {
      root.render(<HookHarness />);
    });

    await flushEffects();

    expect(apiKeyProviderMocks.getProviders).not.toHaveBeenCalled();
    expect(apiKeyProviderMocks.updateProvider).not.toHaveBeenCalled();
    expect(controlPlaneMocks.createClientAccessToken).not.toHaveBeenCalled();
  });

  it("未登录且云端目录不可读时应给 Lime Hub 写入默认聊天模型", async () => {
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    mountedHarness = { container, root };

    act(() => {
      root.render(<HookHarness />);
    });

    await flushEffects();

    expect(apiKeyProviderMocks.updateProvider).toHaveBeenCalledWith(
      "lime-hub",
      expect.objectContaining({
        models: providerModels("gpt-5.2-pro"),
      }),
    );
    expect(
      controlPlaneMocks.listClientProviderOfferModels,
    ).not.toHaveBeenCalled();
    expect(controlPlaneMocks.createClientAccessToken).not.toHaveBeenCalled();
    expect(apiKeyProviderMocks.addApiKey).not.toHaveBeenCalled();
  });

  it("应把默认云端来源的模型目录同步到内部 lime-hub provider", async () => {
    controlPlaneMocks.listClientProviderOfferModels.mockImplementation(
      async (_tenantId: string, providerKey: string) =>
        providerKey === "coding-main"
          ? [
              {
                id: "model-101",
                modelId: "gpt-5.5",
              },
              {
                id: "model-102",
                modelId: "gpt-5.4",
              },
              {
                id: "model-103",
                modelId: "gpt-5.4-mini",
              },
              {
                id: "model-104",
                modelId: "gpt-5.3-codex",
              },
              {
                id: "model-105",
                modelId: "gpt-5.2",
              },
              {
                id: "model-106",
                modelId: "claude-opus-4-5",
              },
              {
                id: "model-107",
                modelId: "claude-sonnet-4-5",
              },
              {
                id: "model-108",
                modelId: "claude-haiku-4-5",
              },
              {
                id: "model-109",
                modelId: "gemini-2.5-pro",
              },
              {
                id: "model-110",
                modelId: "gemini-2.5-flash",
              },
              {
                id: "model-111",
                modelId: "kimi-coding-plan",
              },
              {
                id: "model-112",
                modelId: "glm-coding-plan",
              },
              {
                id: "model-113",
                modelId: "minimax-coding-plan",
              },
              {
                id: "model-114",
                modelId: "mimo-coding-plan",
              },
              {
                id: "model-115",
                modelId: "deepseek-coding-plan",
              },
            ]
          : [
              {
                id: "model-001",
                modelId: "gpt-5.2-fast",
              },
              {
                id: "model-002",
                modelId: "gpt-5.2-pro",
              },
            ],
    );
    setStoredOemCloudSessionState({
      token: "session-token-001",
      tenant: {
        id: "tenant-0001",
      },
      user: {
        id: "user-001",
      },
      session: {
        id: "session-001",
      },
    });
    setOemCloudBootstrapSnapshot({
      providerPreference: {
        providerSource: "oem_cloud",
        providerKey: "offer-main",
        defaultModel: "gpt-5.2-pro",
      },
      providerOffersSummary: [
        {
          source: "oem_cloud",
          providerKey: "offer-main",
          defaultModel: "gpt-5.2-pro",
        },
        {
          source: "oem_cloud",
          providerKey: "coding-main",
          defaultModel: "gpt-5.5",
        },
      ],
    });

    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    mountedHarness = { container, root };

    act(() => {
      root.render(<HookHarness />);
    });

    await flushEffects();

    expect(
      controlPlaneMocks.listClientProviderOfferModels,
    ).toHaveBeenCalledWith("tenant-0001", "offer-main");
    expect(
      controlPlaneMocks.listClientProviderOfferModels,
    ).toHaveBeenCalledWith("tenant-0001", "coding-main");
    expect(apiKeyProviderMocks.updateProvider).toHaveBeenCalledWith(
      "lime-hub",
      {
        name: "Acme Hub",
        api_host:
          "https://gateway-api.limeai.run/root#lime_tenant_id=tenant-0001",
        type: "openai",
        enabled: true,
        sort_order: 0,
        models: providerModels(
          "gpt-5.2-pro",
          "gpt-5.2-fast",
          "gpt-5.5",
          "gpt-5.4",
          "gpt-5.4-mini",
          "gpt-5.3-codex",
          "gpt-5.2",
          "claude-opus-4-5",
          "claude-sonnet-4-5",
          "claude-haiku-4-5",
          "gemini-2.5-pro",
          "gemini-2.5-flash",
          "kimi-coding-plan",
          "glm-coding-plan",
          "minimax-coding-plan",
          "mimo-coding-plan",
          "deepseek-coding-plan",
        ),
      },
    );
    expect(controlPlaneMocks.createClientAccessToken).toHaveBeenCalledWith(
      "tenant-0001",
      {
        name: "Lime Desktop Cloud Model Key",
        scopes: ["llm:invoke"],
        allowedModels: [
          "gpt-5.2-pro",
          "gpt-5.2-fast",
          "gpt-5.5",
          "gpt-5.4",
          "gpt-5.4-mini",
          "gpt-5.3-codex",
          "gpt-5.2",
          "claude-opus-4-5",
          "claude-sonnet-4-5",
          "claude-haiku-4-5",
          "gemini-2.5-pro",
          "gemini-2.5-flash",
          "kimi-coding-plan",
          "glm-coding-plan",
          "minimax-coding-plan",
          "mimo-coding-plan",
          "deepseek-coding-plan",
        ],
      },
    );
    expect(apiKeyProviderMocks.addApiKey).toHaveBeenCalledWith({
      provider_id: "lime-hub",
      api_key: "sk-lime-desktop",
      alias: "Lime cloud models",
    });
  });

  it("已有本地云端 Key 时不应重复创建桌面 Key", async () => {
    apiKeyProviderMocks.getUiState.mockResolvedValue(
      JSON.stringify({
        tenantId: "tenant-0001",
        models: ["gpt-5.2-pro", "gpt-5.2-fast"],
      }),
    );
    setStoredOemCloudSessionState({
      token: "session-token-001",
      tenant: {
        id: "tenant-0001",
      },
      user: {
        id: "user-001",
      },
      session: {
        id: "session-001",
      },
    });
    setOemCloudBootstrapSnapshot({
      providerPreference: {
        providerSource: "oem_cloud",
        providerKey: "offer-main",
        defaultModel: "gpt-5.2-pro",
      },
      providerOffersSummary: [
        {
          source: "oem_cloud",
          providerKey: "offer-main",
          defaultModel: "gpt-5.2-pro",
        },
      ],
    });

    apiKeyProviderMocks.getProviders.mockResolvedValue([
      {
        id: "lime-hub",
        name: "Acme Hub",
        api_host: "https://gateway-api.limeai.run/root",
        type: "openai",
        enabled: true,
        sort_order: 0,
        models: providerModels("gpt-5.2-pro", "gpt-5.2-fast"),
        api_key_count: 1,
        api_keys: [{ enabled: true }],
      },
    ]);

    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    mountedHarness = { container, root };

    act(() => {
      root.render(<HookHarness />);
    });

    await flushEffects();

    expect(controlPlaneMocks.createClientAccessToken).not.toHaveBeenCalled();
    expect(apiKeyProviderMocks.addApiKey).not.toHaveBeenCalled();
  });

  it("云端模型目录扩展时应重签本地托管 Key 并停用旧托管 Key", async () => {
    apiKeyProviderMocks.getUiState.mockResolvedValue(
      JSON.stringify({
        tenantId: "tenant-0001",
        models: ["gpt-5.2-pro", "gpt-5.2-fast"],
      }),
    );
    controlPlaneMocks.listClientProviderOfferModels.mockResolvedValue([
      {
        id: "model-001",
        modelId: "gpt-5.2-fast",
      },
      {
        id: "model-002",
        modelId: "claude-opus-4-7",
      },
    ]);
    setStoredOemCloudSessionState({
      token: "session-token-001",
      tenant: {
        id: "tenant-0001",
      },
      user: {
        id: "user-001",
      },
      session: {
        id: "session-001",
      },
    });
    setOemCloudBootstrapSnapshot({
      providerPreference: {
        providerSource: "oem_cloud",
        providerKey: "offer-main",
        defaultModel: "gpt-5.2-pro",
      },
      providerOffersSummary: [
        {
          source: "oem_cloud",
          providerKey: "offer-main",
          defaultModel: "gpt-5.2-pro",
        },
      ],
    });

    apiKeyProviderMocks.getProviders.mockResolvedValue([
      {
        id: "lime-hub",
        name: "Acme Hub",
        api_host: "https://gateway-api.limeai.run/root",
        type: "openai",
        enabled: true,
        sort_order: 0,
        models: providerModels("gpt-5.2-pro", "gpt-5.2-fast"),
        api_key_count: 1,
        api_keys: [
          {
            id: "old-managed-key",
            alias: "Lime 云端模型",
            enabled: true,
          },
        ],
      },
    ]);

    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    mountedHarness = { container, root };

    act(() => {
      root.render(<HookHarness />);
    });

    await flushEffects();

    expect(controlPlaneMocks.createClientAccessToken).toHaveBeenCalledWith(
      "tenant-0001",
      {
        name: "Lime Desktop Cloud Model Key",
        scopes: ["llm:invoke"],
        allowedModels: ["gpt-5.2-pro", "gpt-5.2-fast", "claude-opus-4-7"],
      },
    );
    expect(apiKeyProviderMocks.addApiKey).toHaveBeenCalledWith({
      provider_id: "lime-hub",
      api_key: "sk-lime-desktop",
      alias: "Lime cloud models",
    });
    expect(apiKeyProviderMocks.toggleApiKey).toHaveBeenCalledWith(
      "old-managed-key",
      false,
    );
    expect(apiKeyProviderMocks.setUiState).toHaveBeenCalledWith(
      "oem_lime_hub_provider_sync:managed_key_models",
      JSON.stringify({
        tenantId: "tenant-0001",
        models: ["gpt-5.2-pro", "gpt-5.2-fast", "claude-opus-4-7"],
      }),
    );
  });

  it("本地云端 Key 已禁用时应重新创建桌面 Key", async () => {
    setStoredOemCloudSessionState({
      token: "session-token-001",
      tenant: {
        id: "tenant-0001",
      },
      user: {
        id: "user-001",
      },
      session: {
        id: "session-001",
      },
    });
    setOemCloudBootstrapSnapshot({
      providerPreference: {
        providerSource: "oem_cloud",
        providerKey: "offer-main",
        defaultModel: "gpt-5.2-pro",
      },
      providerOffersSummary: [
        {
          source: "oem_cloud",
          providerKey: "offer-main",
          defaultModel: "gpt-5.2-pro",
        },
      ],
    });

    apiKeyProviderMocks.getProviders.mockResolvedValue([
      {
        id: "lime-hub",
        name: "Acme Hub",
        api_host: "https://gateway-api.limeai.run/root",
        type: "openai",
        enabled: true,
        sort_order: 0,
        models: providerModels("gpt-5.2-pro", "gpt-5.2-fast"),
        api_key_count: 1,
        api_keys: [{ enabled: false }],
      },
    ]);

    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    mountedHarness = { container, root };

    act(() => {
      root.render(<HookHarness />);
    });

    await flushEffects();

    expect(controlPlaneMocks.createClientAccessToken).toHaveBeenCalledWith(
      "tenant-0001",
      expect.objectContaining({
        allowedModels: ["gpt-5.2-pro", "gpt-5.2-fast"],
      }),
    );
    expect(apiKeyProviderMocks.addApiKey).toHaveBeenCalledWith(
      expect.objectContaining({
        provider_id: "lime-hub",
        api_key: "sk-lime-desktop",
      }),
    );
  });

  it("登录后即使当前偏好为本地来源，也应同步可见云端模型目录", async () => {
    setStoredOemCloudSessionState({
      token: "session-token-001",
      tenant: {
        id: "tenant-0001",
      },
      user: {
        id: "user-001",
      },
      session: {
        id: "session-001",
      },
    });
    setOemCloudBootstrapSnapshot({
      providerPreference: {
        providerSource: "local",
        providerKey: "openai",
      },
      providerOffersSummary: [
        {
          source: "oem_cloud",
          providerKey: "coding-main",
          defaultModel: "gpt-5.5",
        },
      ],
    });
    controlPlaneMocks.listClientProviderOfferModels.mockResolvedValue([
      {
        id: "model-101",
        modelId: "gpt-5.5",
      },
      {
        id: "model-102",
        modelId: "minimax-coding-plan",
      },
    ]);

    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    mountedHarness = { container, root };

    act(() => {
      root.render(<HookHarness />);
    });

    await flushEffects();

    expect(apiKeyProviderMocks.updateProvider).toHaveBeenCalledWith(
      "lime-hub",
      expect.objectContaining({
        models: providerModels("gpt-5.5", "minimax-coding-plan"),
      }),
    );
    expect(apiKeyProviderMocks.addApiKey).toHaveBeenCalledWith(
      expect.objectContaining({
        provider_id: "lime-hub",
        api_key: "sk-lime-desktop",
      }),
    );
  });

  it("bootstrap 切换到本地来源后应清空内部云端模型目录", async () => {
    setStoredOemCloudSessionState({
      token: "session-token-001",
      tenant: {
        id: "tenant-0001",
      },
      user: {
        id: "user-001",
      },
      session: {
        id: "session-001",
      },
    });
    setOemCloudBootstrapSnapshot({
      providerPreference: {
        providerSource: "local",
        providerKey: "openai",
      },
    });

    apiKeyProviderMocks.getProviders.mockResolvedValue([
      {
        id: "lime-hub",
        name: "Acme Hub",
        api_host: "https://gateway-api.limeai.run/root",
        type: "openai",
        enabled: true,
        sort_order: 0,
        models: providerModels("stale-model"),
        api_keys: [],
      },
    ]);

    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    mountedHarness = { container, root };

    act(() => {
      root.render(<HookHarness />);
    });

    await flushEffects();

    expect(
      controlPlaneMocks.listClientProviderOfferModels,
    ).not.toHaveBeenCalled();
    expect(apiKeyProviderMocks.updateProvider).toHaveBeenCalledWith(
      "lime-hub",
      {
        api_host:
          "https://gateway-api.limeai.run/root#lime_tenant_id=tenant-0001",
        models: [],
      },
    );
    expect(controlPlaneMocks.createClientAccessToken).not.toHaveBeenCalled();
    expect(apiKeyProviderMocks.addApiKey).not.toHaveBeenCalled();
  });
});
