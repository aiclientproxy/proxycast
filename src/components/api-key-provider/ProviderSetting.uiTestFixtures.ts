import type { ProviderWithKeysDisplay } from "@/lib/api/apiKeyProvider";
import settingsZhCN from "@/i18n/resources/zh-CN/settings.json";

const settingsDictionary = settingsZhCN as Record<string, string>;

function interpolateTemplate(
  template: string,
  values?: Record<string, unknown>,
): string {
  return template.replace(/\{\{(\w+)\}\}/g, (_, name: string) =>
    String(values?.[name] ?? ""),
  );
}

export function translate(
  key: string,
  fallbackOrOptions?: string | { defaultValue?: string },
) {
  if (typeof fallbackOrOptions === "string") {
    return fallbackOrOptions;
  }

  if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
    const template =
      settingsDictionary[key] ||
      (typeof fallbackOrOptions.defaultValue === "string"
        ? fallbackOrOptions.defaultValue
        : key);
    return interpolateTemplate(
      template,
      fallbackOrOptions as Record<string, unknown>,
    );
  }

  return settingsDictionary[key] || key;
}

export function createProvider(
  overrides: Partial<ProviderWithKeysDisplay> = {},
): ProviderWithKeysDisplay {
  return {
    id: "deepseek",
    name: "DeepSeek",
    type: "openai",
    api_host: "https://api.deepseek.com",
    is_system: false,
    group: "mainstream",
    enabled: true,
    sort_order: 1,
    api_key_count: 1,
    models: [{ id: "deepseek-chat" }],
    created_at: new Date("2026-03-15T00:00:00.000Z").toISOString(),
    updated_at: new Date("2026-03-15T00:00:00.000Z").toISOString(),
    api_keys: [
      {
        id: "key-001",
        provider_id: "deepseek",
        api_key_masked: "sk-****1234",
        alias: "生产账号",
        enabled: true,
        usage_count: 12,
        error_count: 0,
        last_used_at: new Date("2026-03-15T08:00:00.000Z").toISOString(),
        created_at: new Date("2026-03-14T00:00:00.000Z").toISOString(),
      },
    ],
    ...overrides,
  };
}

export function changeInput(input: HTMLInputElement, value: string) {
  const valueSetter = Object.getOwnPropertyDescriptor(
    HTMLInputElement.prototype,
    "value",
  )?.set;
  valueSetter?.call(input, value);
  input.dispatchEvent(new Event("input", { bubbles: true }));
}

export function getApiModelSuggestionLabels(container: HTMLElement): string[] {
  return Array.from(
    container.querySelectorAll<HTMLButtonElement>(
      '[data-testid="api-model-suggestion"]',
    ),
  ).map((button) => button.textContent?.trim() ?? "");
}
