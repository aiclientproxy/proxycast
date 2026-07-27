import { describe, expect, it } from "vitest";
import type { ProviderWithKeysDisplay } from "@/lib/api/apiKeyProvider";
import {
  buildEnabledModelItems,
  isProviderVisibleInEnabledModelList,
} from "./ModelProviderList.utils";

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
    sort_order: 2,
    api_key_count: 1,
    models: [{ id: "deepseek-chat" }],
    prompt_cache_mode: null,
    created_at: new Date("2026-03-15T00:00:00.000Z").toISOString(),
    updated_at: new Date("2026-03-15T00:00:00.000Z").toISOString(),
    api_keys: [
      {
        id: "key-1",
        provider_id: "deepseek",
        api_key_masked: "sk-****1234",
        enabled: true,
        usage_count: 0,
        error_count: 0,
        created_at: new Date("2026-03-15T00:00:00.000Z").toISOString(),
      },
    ],
    ...overrides,
  };
}

describe("ModelProviderList helpers", () => {
  it("只把启用且已配置的 Provider 放入左侧模型列表", () => {
    const enabled = createProvider();
    const disabled = createProvider({
      id: "openai",
      name: "OpenAI",
      enabled: false,
      models: [],
      api_keys: [],
    });
    const unconfigured = createProvider({
      id: "anthropic",
      name: "Anthropic",
      models: [],
      api_keys: [],
    });

    expect(isProviderVisibleInEnabledModelList(enabled)).toBe(true);
    expect(isProviderVisibleInEnabledModelList(disabled)).toBe(false);
    expect(isProviderVisibleInEnabledModelList(unconfigured)).toBe(false);
  });

  it("显式打开 OEM 登录提示时，应展示未登录 Lime Hub 但不标为默认模型", () => {
    const limeHub = createProvider({
      id: "lime-hub",
      name: "Lime Hub",
      sort_order: 0,
      models: [],
      api_keys: [],
      api_key_count: 0,
    });
    const deepseek = createProvider({
      id: "deepseek",
      name: "DeepSeek",
      sort_order: 1,
    });

    expect(isProviderVisibleInEnabledModelList(limeHub)).toBe(false);
    expect(
      isProviderVisibleInEnabledModelList(limeHub, {
        exposeOemLoginPrompt: true,
      }),
    ).toBe(true);

    const items = buildEnabledModelItems([limeHub, deepseek], {
      exposeOemLoginPrompt: true,
    });

    expect(items.map((item) => [item.id, item.status, item.isDefault])).toEqual(
      [
        ["lime-hub", "login_required", false],
        ["deepseek", "ready", true],
      ],
    );
  });

  it("按 sort_order 排序并将首项标记为默认", () => {
    const first = createProvider({ id: "a", name: "A", sort_order: 1 });
    const second = createProvider({ id: "b", name: "B", sort_order: 2 });

    const items = buildEnabledModelItems([second, first]);

    expect(items.map((item) => item.id)).toEqual(["a", "b"]);
    expect(items[0].isDefault).toBe(true);
    expect(items[1].isDefault).toBe(false);
  });
});
