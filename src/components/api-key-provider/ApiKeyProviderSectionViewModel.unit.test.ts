import { describe, expect, it } from "vitest";
import type { ProviderWithKeysDisplay } from "@/lib/api/apiKeyProvider";
import {
  buildApiKeyProviderSectionViewModel,
  isSelectedProviderLoginRequired,
  planDeleteProviderConfig,
  planEnabledModelSelection,
} from "./ApiKeyProviderSectionViewModel";

function createProvider(
  overrides: Partial<ProviderWithKeysDisplay> = {},
): ProviderWithKeysDisplay {
  return {
    id: "deepseek",
    name: "DeepSeek",
    type: "openai",
    api_host: "https://api.deepseek.com",
    is_system: true,
    group: "mainstream",
    enabled: true,
    sort_order: 1,
    api_key_count: 1,
    models: [{ id: "deepseek-chat" }],
    prompt_cache_mode: null,
    created_at: "2026-03-15T00:00:00.000Z",
    updated_at: "2026-03-15T00:00:00.000Z",
    api_keys: [
      {
        id: "key-1",
        provider_id: "deepseek",
        api_key_masked: "sk-****1234",
        enabled: true,
        usage_count: 0,
        error_count: 0,
        created_at: "2026-03-15T00:00:00.000Z",
      },
    ],
    ...overrides,
  };
}

describe("ApiKeyProviderSectionViewModel", () => {
  it("未登录 Lime Hub 只有显式暴露登录提示时才进入 login_required", () => {
    const limeHub = createProvider({
      id: "lime-hub",
      name: "Lime Hub",
      api_key_count: 0,
      api_keys: [],
      models: [],
    });

    expect(
      isSelectedProviderLoginRequired({
        provider: limeHub,
        exposeOemLoginPrompt: true,
      }),
    ).toBe(true);
    expect(
      isSelectedProviderLoginRequired({
        provider: createProvider({
          id: "lime-hub",
          api_key_count: undefined as unknown as number,
          api_keys: [],
          models: [],
        }),
        exposeOemLoginPrompt: true,
      }),
    ).toBe(true);
    expect(
      isSelectedProviderLoginRequired({
        provider: limeHub,
        exposeOemLoginPrompt: false,
      }),
    ).toBe(false);
    expect(
      isSelectedProviderLoginRequired({
        provider: createProvider({
          id: "lime-hub",
          api_key_count: 1,
        }),
        exposeOemLoginPrompt: true,
      }),
    ).toBe(false);
    expect(
      isSelectedProviderLoginRequired({
        provider: createProvider({
          id: "lime-hub",
          api_key_count: 0,
          api_keys: [],
          models: [{ id: "gpt-5.2-pro" }],
        }),
        exposeOemLoginPrompt: true,
      }),
    ).toBe(false);
  });

  it("构建设置区 VM 时保留未登录 Lime Hub 选择，不重定向到本地 Provider", () => {
    const limeHub = createProvider({
      id: "lime-hub",
      name: "Lime Hub",
      sort_order: 0,
      api_key_count: 0,
      api_keys: [],
      models: [],
    });
    const deepseek = createProvider({ sort_order: 1 });

    const viewModel = buildApiKeyProviderSectionViewModel({
      providers: [limeHub, deepseek],
      selectedProvider: limeHub,
      exposeOemLoginPrompt: true,
    });

    expect(viewModel.enabledModelItems.map((item) => item.id)).toEqual([
      "lime-hub",
      "deepseek",
    ]);
    expect(viewModel.selectedProviderLoginRequired).toBe(true);
    expect(
      planEnabledModelSelection({
        enabledModelItems: viewModel.enabledModelItems,
        selectedProviderId: "lime-hub",
        showAddModelFlow: false,
      }),
    ).toEqual({ type: "none" });
  });

  it("自动选择计划只修正空选中或不可见选中项", () => {
    const items = [
      { id: "deepseek", status: "ready" as const },
      { id: "openai", status: "ready" as const },
    ];

    expect(
      planEnabledModelSelection({
        enabledModelItems: items,
        selectedProviderId: "openai",
        showAddModelFlow: false,
      }),
    ).toEqual({ type: "none" });
    expect(
      planEnabledModelSelection({
        enabledModelItems: items,
        selectedProviderId: "missing",
        showAddModelFlow: false,
      }),
    ).toEqual({ type: "select", providerId: "deepseek" });
    expect(
      planEnabledModelSelection({
        enabledModelItems: items,
        selectedProviderId: null,
        showAddModelFlow: false,
      }),
    ).toEqual({ type: "select", providerId: "deepseek" });
    expect(
      planEnabledModelSelection({
        enabledModelItems: items,
        selectedProviderId: "missing",
        showAddModelFlow: true,
      }),
    ).toEqual({ type: "none" });
  });

  it("存在未登录 Lime Hub 提示时应优先选中登录提示", () => {
    const items = [
      { id: "lime-hub", status: "login_required" as const },
      { id: "openai", status: "ready" as const },
    ];

    expect(
      planEnabledModelSelection({
        enabledModelItems: items,
        selectedProviderId: "openai",
        showAddModelFlow: false,
      }),
    ).toEqual({ type: "select", providerId: "lime-hub" });
  });

  it("没有可见模型时只在存在旧选中项时清空选择", () => {
    expect(
      planEnabledModelSelection({
        enabledModelItems: [],
        selectedProviderId: "deepseek",
        showAddModelFlow: false,
      }),
    ).toEqual({ type: "select", providerId: null });
    expect(
      planEnabledModelSelection({
        enabledModelItems: [],
        selectedProviderId: null,
        showAddModelFlow: false,
      }),
    ).toEqual({ type: "none" });
  });

  it("删除配置计划区分缺失、自定义 Provider 和系统 Provider", () => {
    const custom = createProvider({
      id: "custom-1",
      name: "Custom",
      is_system: false,
    });
    const system = createProvider();

    expect(
      planDeleteProviderConfig({
        providers: [custom, system],
        providerId: "missing",
        selectedProviderId: "missing",
      }),
    ).toEqual({ type: "missing" });
    expect(
      planDeleteProviderConfig({
        providers: [custom, system],
        providerId: "custom-1",
        selectedProviderId: "custom-1",
      }),
    ).toEqual({ type: "delete-custom", providerId: "custom-1" });
    expect(
      planDeleteProviderConfig({
        providers: [custom, system],
        providerId: "deepseek",
        selectedProviderId: "deepseek",
      }),
    ).toEqual({
      type: "reset-system",
      providerId: "deepseek",
      apiKeyIds: ["key-1"],
      update: {
        enabled: false,
        models: [],
      },
      clearSelection: true,
    });
  });
});
