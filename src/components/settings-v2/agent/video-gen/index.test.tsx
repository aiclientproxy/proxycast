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

vi.mock("@/lib/api/appConfig", () => ({
  getConfig: mockGetConfig,
  saveConfig: mockSaveConfig,
  updateConfig: mockUpdateConfig,
}));

vi.mock("@/components/input-kit", () => ({
  ModelSelector: ({
    providerType,
    model,
    placeholderLabel,
  }: {
    providerType: string;
    model: string;
    placeholderLabel?: string;
  }) => {
    const providerLabel =
      providerType === "doubao-video" ? "Doubao Video" : providerType;
    return (
      <div data-testid="video-model-selector">
        {providerLabel || placeholderLabel || "Auto select"} /{" "}
        {model || placeholderLabel || "Auto select"}
      </div>
    );
  },
}));

vi.mock("@/hooks/useApiKeyProvider", () => ({
  useApiKeyProvider: () => ({
    providers: [
      {
        id: "doubao-video",
        type: "openai",
        name: "Doubao Video",
        enabled: true,
        api_key_count: 1,
        api_keys: [{ enabled: true }],
        models: [{ id: "seedance-1-5-pro-251215" }],
      },
    ],
    loading: false,
  }),
}));

import { VideoGenSettings } from ".";

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
    root.render(<VideoGenSettings />);
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

beforeEach(async () => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;

  vi.clearAllMocks();
  await changeLimeLocale("en-US");

  persistedConfig = {
    default_provider: "openai",
    workspace_preferences: {
      media_defaults: {
        video: {
          preferredProviderId: "doubao-video",
          preferredModelId: "seedance-1-5-pro-251215",
          allowFallback: false,
        },
      },
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

describe("VideoGenSettings", () => {
  it("应加载简化后的视频服务模型设置", async () => {
    const container = renderComponent();
    await flushEffects(3);

    expect(container.textContent).toContain("Video Service Model");
    expect(container.textContent).toContain("Doubao Video");
    expect(container.textContent).toContain("seedance-1-5-pro-251215");
    expect(container.textContent).not.toContain("全局默认视频服务");
  });

  it("应把视频设置补充说明收进 tips", async () => {
    renderComponent();
    await flushEffects(3);

    expect(getBodyText()).not.toContain(
      "这里配置视频任务的默认 Provider、模型与回退策略，保持和图片、语音一致的简洁设置结构。",
    );
    expect(getBodyText()).not.toContain(
      "关闭后，若当前默认视频服务缺失、被禁用或无可用 Key，将直接提示错误。",
    );
    expect(getBodyText()).not.toContain("settings.mediaGeneration");

    const sectionTip = await hoverTip("Video Service Model info");
    expect(getBodyText()).toContain(
      "Configure the default Provider, model, and fallback policy for video tasks while keeping the same simple structure as image and voice.",
    );
    await leaveTip(sectionTip);

    const fallbackTip = await hoverTip(
      "Auto fallback when Provider is unavailable info",
    );
    expect(getBodyText()).toContain(
      "When disabled, Lime shows an error if the default video service is missing, disabled, or has no usable key.",
    );
    await leaveTip(fallbackTip);
  });

  it("恢复默认后应清空视频服务覆盖", async () => {
    const container = renderComponent();
    await flushEffects(3);

    await act(async () => {
      findButton(container, "Restore defaults").click();
      await flushEffects(2);
    });

    expect(mockSaveConfig).toHaveBeenCalledTimes(1);
    const savedConfig = mockSaveConfig.mock.calls[0][0];
    expect(
      savedConfig.workspace_preferences.media_defaults.video,
    ).toBeUndefined();
    expect(container.textContent).toContain("Settings saved");
  });
});
