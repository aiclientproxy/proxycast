import type { ProviderWithKeysDisplay } from "@/lib/api/apiKeyProvider";

export const defaultSystemProviderCatalog = [
  {
    id: "deepseek",
    name: "DeepSeek",
    type: "openai",
    api_host: "https://api.deepseek.com",
    group: "mainstream",
    sort_order: 1,
    legacy_ids: [],
  },
];

export function createProvider(
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

export function createApiKeyProviderHookState(
  mockUseApiKeyProvider: { mockReturnValue: (value: unknown) => unknown },
  createMock: () => {
    mockResolvedValue: (value: unknown) => unknown;
  },
  overrides: Record<string, unknown> = {},
) {
  const deepseek = createProvider();
  const openai = createProvider({
    id: "openai",
    name: "OpenAI",
    enabled: false,
    sort_order: 2,
    models: [],
    api_keys: [],
    api_key_count: 0,
  });
  const state = {
    providers: [deepseek, openai],
    selectedProviderId: "deepseek",
    selectedProvider: deepseek,
    loading: false,
    error: null,
    searchQuery: "",
    collapsedGroups: new Set(),
    refresh: createMock().mockResolvedValue(undefined),
    selectProvider: createMock(),
    setSearchQuery: createMock(),
    toggleGroup: createMock(),
    reorderProviders: createMock().mockResolvedValue(undefined),
    addCustomProvider: createMock().mockResolvedValue({ id: "custom-1" }),
    updateProvider: createMock().mockResolvedValue({ id: "custom-1" }),
    deleteCustomProvider: createMock().mockResolvedValue(true),
    toggleProviderEnabled: createMock().mockResolvedValue({ id: "deepseek" }),
    addApiKey: createMock().mockResolvedValue({ id: "key-new" }),
    deleteApiKey: createMock().mockResolvedValue(true),
    toggleApiKey: createMock().mockResolvedValue({ id: "key-1" }),
    updateApiKeyAlias: createMock().mockResolvedValue({ id: "key-1" }),
    exportConfig: createMock().mockResolvedValue("{}"),
    importConfig: createMock().mockResolvedValue({ success: true }),
    filteredProviders: [deepseek, openai],
    providersByGroup: new Map(),
    ...overrides,
  };
  mockUseApiKeyProvider.mockReturnValue(state);
  return state;
}

export function findByTestId<T extends HTMLElement>(testId: string): T {
  const element = document.querySelector(`[data-testid="${testId}"]`);
  if (!(element instanceof HTMLElement)) {
    throw new Error(`未找到 data-testid=${testId} 的节点`);
  }
  return element as T;
}

export function maybeByTestId<T extends HTMLElement>(
  root: Document | HTMLElement,
  testId: string,
): T | null {
  return root.querySelector<T>(`[data-testid="${testId}"]`);
}

export function maybeProviderItem(
  root: Document | HTMLElement,
  providerId: string,
): HTMLButtonElement | null {
  return root.querySelector<HTMLButtonElement>(
    `[data-testid="enabled-model-item"][data-provider-id="${providerId}"]`,
  );
}

export function maybeTemplateCard(
  root: Document | HTMLElement,
  templateId: string,
): HTMLButtonElement | null {
  return root.querySelector<HTMLButtonElement>(
    `[data-template-id="${templateId}"]`,
  );
}

export function setInputValue(input: HTMLInputElement, value: string) {
  const valueSetter = Object.getOwnPropertyDescriptor(input, "value")?.set;
  const prototypeValueSetter = Object.getOwnPropertyDescriptor(
    Object.getPrototypeOf(input),
    "value",
  )?.set;

  if (prototypeValueSetter && valueSetter !== prototypeValueSetter) {
    prototypeValueSetter.call(input, value);
  } else {
    input.value = value;
  }

  input.dispatchEvent(new Event("input", { bubbles: true }));
  input.dispatchEvent(new Event("change", { bubbles: true }));
}
