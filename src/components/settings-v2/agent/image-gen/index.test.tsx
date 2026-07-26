import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";
import type { Config } from "@/lib/api/appConfig";

const { mockGetConfig, mockSaveConfig, mockUpdateConfig } = vi.hoisted(() => ({
  mockGetConfig: vi.fn(),
  mockSaveConfig: vi.fn(),
  mockUpdateConfig: vi.fn(),
}));
const { mockModelSelectorRender } = vi.hoisted(() => ({
  mockModelSelectorRender: vi.fn(),
}));

vi.mock("@/lib/api/appConfig", () => ({
  getConfig: mockGetConfig,
  saveConfig: mockSaveConfig,
  updateConfig: mockUpdateConfig,
}));

vi.mock("@/components/input-kit", () => ({
  ModelSelector: (props: {
    providerType: string;
    model: string;
    placeholderLabel?: string;
    setProviderType: (value: string) => void;
    setModel: (value: string) => void;
    setProviderAndModel?: (providerType: string, model: string) => void;
    providerFilter?: (provider: {
      key: string;
      label: string;
      registryId: string;
      fallbackRegistryId?: string;
      type: string;
      providerId?: string;
      apiHost?: string;
      customModels?: string[];
      authStatus?: "ready" | "login_required";
    }) => boolean;
    modelFilter?: (
      model: { id: string; task_families?: string[] },
      provider: {
        key: string;
        label: string;
        registryId: string;
        fallbackRegistryId?: string;
        type: string;
        providerId?: string;
        apiHost?: string;
        customModels?: string[];
      },
    ) => boolean;
    getFallbackModels?: (provider: {
      key: string;
      label: string;
      registryId: string;
      fallbackRegistryId?: string;
      type: string;
      providerId?: string;
      apiHost?: string;
      customModels?: string[];
    }) => Array<{ id: string }>;
  }) => {
    mockModelSelectorRender(props);
    const {
      providerType,
      model,
      placeholderLabel,
      providerFilter,
      modelFilter,
      getFallbackModels,
    } = props;
    const providerFixtures = [
      {
        key: "relay-openai",
        label: "Relay OpenAI",
        registryId: "relay-openai",
        type: "openai",
        providerId: "relay-openai",
        customModels: ["gpt-images-2"],
      },
      {
        key: "fal",
        label: "Fal",
        registryId: "fal",
        fallbackRegistryId: "openai",
        type: "openai",
        providerId: "fal",
        apiHost: "https://fal.run/fal-ai",
        customModels: ["gpt-5.2-pro"],
      },
      {
        key: "agnes",
        label: "agnes",
        registryId: "agnes",
        type: "openai",
        providerId: "agnes",
        apiHost: "https://agnes.example.test/v1",
        customModels: [
          "agnes-2.0-flash",
          "agnes-image-2.1-flash",
          "agnes-image-2.0-flash",
        ],
      },
    ];
    const selectedProvider = providerFixtures.find(
      (provider) => provider.key === providerType,
    );
    const liveModelFixtures: Record<
      string,
      Array<{ id: string; task_families?: string[] }>
    > = {
      agnes: [
        {
          id: "agnes-live-creator-v9",
          task_families: ["image_generation"],
        },
        {
          id: "agnes-live-chat-v9",
          task_families: ["chat"],
        },
      ],
    };
    const providerLabel =
      providerType === "relay-openai"
        ? "Relay OpenAI"
        : providerType === "fal"
          ? "Fal"
          : providerType === "agnes"
            ? "agnes"
            : providerType;
    const fallbackModelIds =
      selectedProvider && providerFilter?.(selectedProvider)
        ? (getFallbackModels?.(selectedProvider)
            ?.filter((item) => modelFilter?.(item, selectedProvider) ?? true)
            .map((item) => item.id) ?? [])
        : [];
    const liveModelIds =
      selectedProvider && providerFilter?.(selectedProvider)
        ? (liveModelFixtures[selectedProvider.key] ?? [])
            .filter((item) => modelFilter?.(item, selectedProvider) ?? true)
            .map((item) => item.id)
        : [];
    return (
      <div data-testid="image-model-selector">
        {providerLabel || placeholderLabel || "Auto select"} /{" "}
        {model || placeholderLabel || "Auto select"}
        {fallbackModelIds.length > 0 ? (
          <span> / {fallbackModelIds.join(",")}</span>
        ) : null}
        {liveModelIds.length > 0 ? (
          <span> / live:{liveModelIds.join(",")}</span>
        ) : null}
      </div>
    );
  },
}));

vi.mock("@/hooks/useApiKeyProvider", () => ({
  useApiKeyProvider: () => ({
    providers: [
      {
        id: "relay-openai",
        type: "openai",
        name: "Relay OpenAI",
        enabled: true,
        api_key_count: 1,
        api_keys: [{ enabled: true }],
        custom_models: ["gpt-images-2"],
      },
      {
        id: "fal",
        type: "openai",
        name: "Fal",
        enabled: true,
        api_key_count: 1,
        api_keys: [{ enabled: true }],
        api_host: "https://fal.run/fal-ai",
        custom_models: ["gpt-5.2-pro"],
      },
      {
        id: "agnes",
        type: "openai",
        name: "agnes",
        enabled: true,
        api_key_count: 1,
        api_keys: [{ enabled: true }],
        api_host: "https://agnes.example.test/v1",
        custom_models: [
          "agnes-2.0-flash",
          "agnes-image-2.1-flash",
          "agnes-image-2.0-flash",
        ],
      },
      {
        id: "tts-only",
        type: "audio",
        name: "TTS Only",
        enabled: true,
        api_key_count: 1,
        api_keys: [{ enabled: true }],
        custom_models: ["gpt-4o-mini-tts"],
      },
    ],
    loading: false,
  }),
}));

vi.mock("@/hooks/useProviderModels", () => ({
  useProviderModels: () => ({
    models: [
      {
        id: "agnes-live-creator-v9",
        display_name: "Agnes Live Creator V9",
        provider_id: "agnes",
        provider_name: "agnes",
        family: null,
        tier: "pro",
        capabilities: {
          vision: false,
          tools: false,
          streaming: false,
          json_mode: false,
          function_calling: false,
          reasoning: false,
        },
        task_families: ["image_generation"],
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
        source: "api",
        created_at: 0,
        updated_at: 0,
      },
    ],
    loading: false,
    error: null,
  }),
}));

import { ImageGenSettings } from ".";

interface Mounted {
  container: HTMLDivElement;
  root: Root;
}

const mounted: Mounted[] = [];
let persistedConfig: Config;

function renderComponent(): HTMLDivElement {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => {
    root.render(<ImageGenSettings />);
  });
  mounted.push({ container, root });
  return container;
}

async function flushEffects(times = 2) {
  await act(async () => {
    for (let index = 0; index < times; index += 1) {
      await Promise.resolve();
    }
  });
}

function getBodyText() {
  return document.body.textContent ?? "";
}

async function hoverTip(ariaLabel: string) {
  const trigger = document.body.querySelector(
    `button[aria-label='${ariaLabel}']`,
  );
  expect(trigger).toBeInstanceOf(HTMLButtonElement);

  await act(async () => {
    trigger?.dispatchEvent(new MouseEvent("mouseover", { bubbles: true }));
    await Promise.resolve();
  });

  return trigger as HTMLButtonElement;
}

async function leaveTip(trigger: HTMLButtonElement | null) {
  await act(async () => {
    trigger?.dispatchEvent(new MouseEvent("mouseout", { bubbles: true }));
    await Promise.resolve();
  });
}

function findButton(container: HTMLElement, text: string): HTMLButtonElement {
  const target = Array.from(container.querySelectorAll("button")).find(
    (button) => button.textContent?.includes(text),
  );
  if (!target) {
    throw new Error(`未找到按钮: ${text}`);
  }
  return target as HTMLButtonElement;
}

function findSection(container: HTMLElement, title: string): HTMLElement {
  const heading = Array.from(container.querySelectorAll("h3")).find((node) =>
    node.textContent?.includes(title),
  );
  if (!heading) {
    throw new Error(`未找到区块标题: ${title}`);
  }
  const section = heading.closest("section");
  if (!section) {
    throw new Error(`未找到区块容器: ${title}`);
  }
  return section as HTMLElement;
}

beforeEach(async () => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;

  vi.clearAllMocks();
  mockModelSelectorRender.mockReset();
  await changeLimeLocale("en-US");

  persistedConfig = {
    default_provider: "openai",
    workspace_preferences: {
      media_defaults: {
        image: {
          preferredProviderId: "relay-openai",
          preferredModelId: "gpt-images-2",
          allowFallback: true,
        },
      },
    },
    image_gen: {
      default_count: 3,
      default_quality: "hd",
    },
  } as Config;
  mockGetConfig.mockImplementation(async () => persistedConfig);
  mockSaveConfig.mockResolvedValue(undefined);
  mockUpdateConfig.mockImplementation(
    async (updater: (current: Config) => Config) => {
      persistedConfig = updater(persistedConfig);
      await mockSaveConfig(persistedConfig);
      return persistedConfig;
    },
  );
});

afterEach(async () => {
  while (mounted.length > 0) {
    const target = mounted.pop();
    if (!target) break;
    act(() => {
      target.root.unmount();
    });
    target.container.remove();
  }
  vi.clearAllMocks();
  await changeLimeLocale("zh-CN");
});

describe("ImageGenSettings", () => {
  it("应加载简化后的图片服务模型设置，并保留 gpt-images-2 选择", async () => {
    const container = renderComponent();
    await flushEffects(3);

    expect(container.textContent).toContain("Image Service Model");
    expect(container.textContent).toContain("Relay OpenAI");
    expect(container.textContent).toContain("gpt-images-2");
    expect(container.textContent).not.toContain("默认图像生成服务");
    expect(container.textContent).not.toContain("默认图像数量");
    expect(container.textContent).not.toContain("图像质量");

    const section = findSection(container, "Image Service Model");
    expect(section.className).toContain("overflow-visible");
    expect(section.className).not.toContain("overflow-hidden");
  });

  it("应把图片设置补充说明收进 tips", async () => {
    renderComponent();
    await flushEffects(3);

    expect(getBodyText()).not.toContain(
      "这里只配置图片生成任务的默认 Provider、模型与回退策略；默认图片数量等全局参数统一收口到同页下方的 AI 图片设置。",
    );
    expect(getBodyText()).not.toContain(
      "关闭后，若当前默认图片服务缺失、被禁用或无可用 Key，将直接提示错误。",
    );
    expect(getBodyText()).not.toContain("settings.mediaGeneration");

    const sectionTip = await hoverTip("Image Service Model info");
    expect(getBodyText()).toContain(
      "Configure the default Provider, model, and fallback policy for image generation here; global parameters such as default image count stay in the AI Image settings below.",
    );
    await leaveTip(sectionTip);

    const fallbackTip = await hoverTip(
      "Auto fallback when Provider is unavailable info",
    );
    expect(getBodyText()).toContain(
      "When disabled, Lime shows an error if the default image service is missing, disabled, or has no usable key.",
    );
    await leaveTip(fallbackTip);
  });

  it("恢复默认后应清空图片服务覆盖", async () => {
    const container = renderComponent();
    await flushEffects(3);

    await act(async () => {
      findButton(container, "Restore defaults").click();
      await flushEffects(2);
    });

    expect(mockSaveConfig).toHaveBeenCalledTimes(1);
    const savedConfig = mockSaveConfig.mock.calls[0][0];
    expect(
      savedConfig.workspace_preferences.media_defaults.image,
    ).toBeUndefined();
    expect(container.textContent).toContain("Settings saved");
  });

  it("跨 Provider 选择应原子保存 Provider 与图片模型", async () => {
    renderComponent();
    await flushEffects(3);

    const selectorProps = mockModelSelectorRender.mock.calls.at(-1)?.[0];
    expect(selectorProps?.setProviderAndModel).toBeTypeOf("function");

    await act(async () => {
      selectorProps.setProviderAndModel("agnes", "agnes-image-2.1-flash");
      await flushEffects(3);
    });

    expect(mockSaveConfig).toHaveBeenCalledTimes(1);
    expect(
      mockSaveConfig.mock.calls[0]?.[0].workspace_preferences.media_defaults
        .image,
    ).toEqual({
      preferredProviderId: "agnes",
      preferredModelId: "agnes-image-2.1-flash",
      allowFallback: true,
    });
  });

  it("Fal 只配置文本自定义模型时，图片模型选择器应回退到内置 Fal 图片模型", async () => {
    mockGetConfig.mockResolvedValueOnce({
      workspace_preferences: {
        media_defaults: {
          image: {
            preferredProviderId: "fal",
            allowFallback: true,
          },
        },
      },
      image_gen: {
        default_count: 3,
        default_quality: "hd",
      },
    });

    const container = renderComponent();
    await flushEffects(3);

    expect(container.textContent).toContain("Fal / Auto select");
    expect(container.textContent).toContain("fal-ai/nano-banana-pro");
    expect(container.textContent).not.toContain("gpt-5.2-pro");
  });

  it("OpenAI 兼容中转拉取的图片模型应进入默认图片模型选择器", async () => {
    mockGetConfig.mockResolvedValueOnce({
      workspace_preferences: {
        media_defaults: {
          image: {
            preferredProviderId: "agnes",
            allowFallback: true,
          },
        },
      },
      image_gen: {
        default_count: 3,
        default_quality: "hd",
      },
    });

    const container = renderComponent();
    await flushEffects(3);

    expect(container.textContent).toContain("agnes / Auto select");
    expect(container.textContent).toContain("agnes-image-2.1-flash");
    expect(container.textContent).toContain("agnes-image-2.0-flash");
    expect(container.textContent).not.toContain("agnes-2.0-flash");
  });

  it("后端实时返回的图片任务族模型应进入默认图片模型选择器", async () => {
    mockGetConfig.mockResolvedValueOnce({
      workspace_preferences: {
        media_defaults: {
          image: {
            preferredProviderId: "agnes",
            allowFallback: true,
          },
        },
      },
      image_gen: {
        default_count: 3,
        default_quality: "hd",
      },
    });

    const container = renderComponent();
    await flushEffects(3);

    expect(container.textContent).toContain("live:agnes-live-creator-v9");
    expect(container.textContent).not.toContain("agnes-live-chat-v9");
  });

  it("保存的后端实时图片模型不应被误报不可用", async () => {
    mockGetConfig.mockResolvedValueOnce({
      workspace_preferences: {
        media_defaults: {
          image: {
            preferredProviderId: "agnes",
            preferredModelId: "agnes-live-creator-v9",
            allowFallback: true,
          },
        },
      },
      image_gen: {
        default_count: 3,
        default_quality: "hd",
      },
    });

    const container = renderComponent();
    await flushEffects(3);

    expect(container.textContent).toContain("agnes / agnes-live-creator-v9");
    expect(container.textContent).not.toContain(
      "agnes-live-creator-v9 is unavailable",
    );
  });
});
