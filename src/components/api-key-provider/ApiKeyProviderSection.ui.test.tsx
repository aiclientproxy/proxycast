import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ProviderWithKeysDisplay } from "@/lib/api/apiKeyProvider";
import type { ProviderSettingsFocusContext } from "@/types/page";

const {
  mockUseApiKeyProvider,
  mockUseModelRegistry,
  mockGetSystemProviderCatalog,
  mockTestConnection,
  mockTestChat,
  mockFetchProviderModelsAuto,
  mockOpenExternalUrlWithSystemBrowser,
} = vi.hoisted(() => ({
  mockUseApiKeyProvider: vi.fn(),
  mockUseModelRegistry: vi.fn(),
  mockGetSystemProviderCatalog: vi.fn(),
  mockTestConnection: vi.fn(),
  mockTestChat: vi.fn(),
  mockFetchProviderModelsAuto: vi.fn(),
  mockOpenExternalUrlWithSystemBrowser: vi.fn(),
}));

vi.mock("@/hooks/useApiKeyProvider", () => ({
  useApiKeyProvider: mockUseApiKeyProvider,
}));

vi.mock("@/hooks/useModelRegistry", () => ({
  useModelRegistry: mockUseModelRegistry,
}));

vi.mock("@/lib/api/apiKeyProvider", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/lib/api/apiKeyProvider")>();
  return {
    ...actual,
    apiKeyProviderApi: {
      ...actual.apiKeyProviderApi,
      getSystemProviderCatalog: mockGetSystemProviderCatalog,
      testConnection: mockTestConnection,
      testChat: mockTestChat,
    },
  };
});

vi.mock("@/lib/api/modelRegistry", () => ({
  fetchProviderModelsAuto: (...args: unknown[]) =>
    mockFetchProviderModelsAuto(...args),
  normalizeFetchProviderModelsSource: (result: {
    source: "Api" | "Error";
    models: unknown[];
    error: string | null;
  }) => result.source,
}));

vi.mock("@/lib/api/externalUrl", () => ({
  openExternalUrlWithSystemBrowser: (...args: unknown[]) =>
    mockOpenExternalUrlWithSystemBrowser(...args),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      fallbackOrOptions?: string | { defaultValue?: string },
    ) => {
      const template =
        typeof fallbackOrOptions === "string"
          ? fallbackOrOptions
          : (fallbackOrOptions?.defaultValue ?? _key);

      if (!fallbackOrOptions || typeof fallbackOrOptions === "string") {
        return template;
      }

      return template.replace(/{{\s*(\w+)\s*}}/g, (match, name) => {
        const value = (fallbackOrOptions as Record<string, unknown>)[name];
        return value == null ? match : String(value);
      });
    },
    i18n: {
      language: "zh-CN",
      resolvedLanguage: "zh-CN",
    },
  }),
}));

vi.mock("./ProviderSetting", () => ({
  ProviderSetting: (props: {
    provider: ProviderWithKeysDisplay | null;
    focus?: ProviderSettingsFocusContext | null;
    authStatus?: "ready" | "login_required";
    onLogin?: () => void | Promise<void>;
    onTestConnection?: (
      providerId: string,
      options?: {
        modelName?: string;
        requireChatReady?: boolean;
        prompt?: string;
      },
    ) => Promise<unknown>;
    onDeleteProvider?: (providerId: string) => Promise<boolean | void>;
  }) => {
    if (props.provider && props.authStatus === "login_required") {
      return (
        <div data-testid="provider-login-required">
          {props.provider.name} 需要登录
          <p>登录后会自动同步 Lime Hub 的可用模型</p>
          <button
            type="button"
            data-testid="provider-login-button"
            onClick={() => {
              void props.onLogin?.();
            }}
          >
            去登录
          </button>
        </div>
      );
    }

    return (
      <div
        data-testid="provider-setting-stub"
        data-focus-provider-id={props.focus?.providerId ?? ""}
        data-focus-model-id={props.focus?.modelId ?? ""}
        data-focus-reason-code={props.focus?.reasonCode ?? ""}
      >
        {props.provider?.name ?? "未选择模型"}
        {props.provider ? (
          <button
            type="button"
            data-testid="provider-setting-test-chat-stub"
            onClick={() => {
              void props.onTestConnection?.(props.provider!.id, {
                modelName: props.provider!.models?.[0]?.id,
                requireChatReady: true,
                prompt: "试跑",
              });
            }}
          >
            试跑
          </button>
        ) : null}
        {props.provider && props.onDeleteProvider ? (
          <button
            type="button"
            data-testid="provider-setting-delete-stub"
            onClick={() => {
              void props.onDeleteProvider?.(props.provider!.id);
            }}
          >
            删除配置
          </button>
        ) : null}
      </div>
    );
  },
}));

vi.mock("./ImportExportDialog", () => ({
  ImportExportDialog: () => null,
}));

import { ApiKeyProviderSection } from "./ApiKeyProviderSection";
import {
  createApiKeyProviderHookState,
  createProvider,
  defaultSystemProviderCatalog,
  findByTestId,
  maybeByTestId,
  maybeProviderItem,
  maybeTemplateCard,
  setInputValue,
} from "./ApiKeyProviderSection.uiTestFixtures";

interface MountedRoot {
  container: HTMLDivElement;
  root: Root;
}

const mountedRoots: MountedRoot[] = [];

function createHookState(overrides: Record<string, unknown> = {}) {
  return createApiKeyProviderHookState(mockUseApiKeyProvider, vi.fn, overrides);
}

function renderSection(
  props: Partial<React.ComponentProps<typeof ApiKeyProviderSection>> = {},
) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(<ApiKeyProviderSection {...props} />);
  });

  mountedRoots.push({ container, root });
  return container;
}

async function flushEffects(times = 2) {
  await act(async () => {
    for (let index = 0; index < times; index += 1) {
      await Promise.resolve();
    }
  });
}

async function clickByTestId(testId: string, flushes = 1) {
  await act(async () => {
    findByTestId<HTMLButtonElement>(testId).click();
    for (let index = 0; index < flushes; index += 1) {
      await Promise.resolve();
    }
  });
}

async function openCustomProviderForm() {
  await clickByTestId("add-model-button");
  await clickByTestId("custom-provider-template-card");
}

async function submitCustomProviderDraft({
  name,
  apiHost,
  apiKey = "sk-test",
  model,
}: {
  name: string;
  apiHost: string;
  apiKey?: string;
  model: string;
}) {
  await fillCustomProviderDraft({ name, apiHost, apiKey, model });
  await clickByTestId("model-activate-button", 3);
}

async function fillCustomProviderDraft({
  name,
  apiHost,
  apiKey = "sk-test",
  model,
}: {
  name: string;
  apiHost: string;
  apiKey?: string;
  model: string;
}) {
  await act(async () => {
    setInputValue(
      findByTestId<HTMLInputElement>("model-provider-name-input"),
      name,
    );
    setInputValue(
      findByTestId<HTMLInputElement>("model-api-host-input"),
      apiHost,
    );
    setInputValue(
      findByTestId<HTMLInputElement>("model-api-key-input"),
      apiKey,
    );
    setInputValue(findByTestId<HTMLInputElement>("model-draft-input"), model);
  });
  await clickByTestId("model-draft-add-button");
}

function collectReadableText(node: Node): string {
  if (node instanceof Element && node.getAttribute("aria-hidden") === "true") {
    return "";
  }

  if (node.nodeType === Node.TEXT_NODE) {
    return node.textContent ?? "";
  }

  return Array.from(node.childNodes)
    .map((childNode) => collectReadableText(childNode))
    .join("");
}

beforeEach(() => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  vi.clearAllMocks();
  mockGetSystemProviderCatalog.mockResolvedValue(defaultSystemProviderCatalog);
  mockUseModelRegistry.mockReturnValue({
    groupedByProvider: new Map([
      [
        "kimi-for-coding",
        [
          {
            id: "kimi-for-coding",
            provider_name: "Kimi Coding Plan",
          },
        ],
      ],
    ]),
  });
  mockTestConnection.mockResolvedValue({ success: true, latency_ms: 12 });
  mockTestChat.mockResolvedValue({ success: true, latency_ms: 21 });
  mockFetchProviderModelsAuto.mockResolvedValue({
    source: "Api",
    models: [{ id: "deepseek-chat" }],
    error: null,
  });
  mockOpenExternalUrlWithSystemBrowser.mockResolvedValue(undefined);
});

afterEach(() => {
  while (mountedRoots.length > 0) {
    const mounted = mountedRoots.pop();
    if (!mounted) {
      break;
    }
    act(() => {
      mounted.root.unmount();
    });
    mounted.container.remove();
  }
  vi.clearAllMocks();
});

describe("ApiKeyProviderSection 模型管理布局", () => {
  it("Provider 加载失败时应显示安全错误态并提供重试", async () => {
    const hookState = createHookState({
      providers: [],
      selectedProviderId: null,
      selectedProvider: null,
      loading: false,
      error:
        "数据库迁移失败: /Users/private/app-server/lime.db permission denied",
    });
    const container = renderSection();
    await flushEffects();

    expect(maybeByTestId(container, "provider-load-error")).not.toBeNull();
    expect(container.textContent ?? "").toContain("本地模型设置暂时不可用");
    expect(container.textContent ?? "").toContain("检查应用数据目录权限");
    expect(container.textContent ?? "").not.toContain("/Users/private");
    expect(maybeByTestId(container, "enabled-model-list")).toBeNull();

    await clickByTestId("provider-load-retry");
    expect(hookState.refresh).toHaveBeenCalledTimes(1);
  });

  it("常态左侧只展示启用模型，不再展示旧 Provider 分组列表", async () => {
    createHookState();
    const container = renderSection();
    await flushEffects();

    expect(maybeByTestId(container, "provider-list")).toBeNull();
    expect(maybeByTestId(container, "enabled-model-list")).not.toBeNull();
    expect(
      maybeByTestId(container, "api-key-provider-section")?.className,
    ).toContain("min-h-0");
    expect(
      maybeByTestId(container, "api-key-provider-detail")?.className,
    ).toContain("min-h-0");
    expect(maybeByTestId(container, "enabled-model-list")?.className).toContain(
      "overflow-hidden",
    );
    expect(
      maybeByTestId(container, "enabled-model-scroll-region")?.className,
    ).toContain("overflow-y-auto");
    expect(
      maybeByTestId(container, "enabled-model-scroll-region")?.className,
    ).toContain("overscroll-contain");
    expect(container.textContent ?? "").toContain("启用的模型");
    expect(container.textContent ?? "").toContain("添加模型");
    expect(container.textContent ?? "").toContain("导入 / 导出配置");
    expect(container.textContent ?? "").toContain("DeepSeek");
    expect(container.textContent ?? "").not.toContain("OpenAI");
  });

  it("initialFocus 命中 providerId 时应切换到目标服务商并透传焦点", async () => {
    const customCoder = createProvider({
      id: "custom-coder",
      name: "Custom Coder",
      sort_order: 3,
      models: [{ id: "coder-small" }],
      api_keys: [],
      api_key_count: 0,
    });
    const hookState = createHookState({
      providers: [createProvider(), customCoder],
      selectedProviderId: "deepseek",
      selectedProvider: customCoder,
      filteredProviders: [createProvider(), customCoder],
    });

    const container = renderSection({
      initialFocus: {
        providerId: "custom-coder",
        modelId: "coder-large",
        reasonCode: "missing_enabled_api_key",
        recoveryAction: "add_enabled_api_key",
      },
    });
    await flushEffects();

    expect(hookState.selectProvider).toHaveBeenCalledWith("custom-coder");
    const stub = maybeByTestId(container, "provider-setting-stub");
    expect(stub?.getAttribute("data-focus-provider-id")).toBe("custom-coder");
    expect(stub?.getAttribute("data-focus-model-id")).toBe("coder-large");
    expect(stub?.getAttribute("data-focus-reason-code")).toBe(
      "missing_enabled_api_key",
    );
  });

  it("initialFocus 只有 modelId 时应按已启用模型反查服务商", async () => {
    const codingProvider = createProvider({
      id: "coding-provider",
      name: "Coding Provider",
      sort_order: 3,
      models: [{ id: "coder-large" }],
    });
    const hookState = createHookState({
      providers: [createProvider(), codingProvider],
      selectedProviderId: "deepseek",
      selectedProvider: codingProvider,
      filteredProviders: [createProvider(), codingProvider],
    });

    renderSection({
      initialFocus: {
        modelId: "coder-large",
        reasonCode: "model_not_enabled",
      },
    });
    await flushEffects();

    expect(hookState.selectProvider).toHaveBeenCalledWith("coding-provider");
  });

  it("未登录时可在 AI 服务商列表中展示 Lime Hub 登录提示", async () => {
    const onOemLogin = vi.fn();
    const limeHub = createProvider({
      id: "lime-hub",
      name: "Lime 云端",
      group: "cloud",
      sort_order: 0,
      models: [],
      api_keys: [],
      api_key_count: 0,
    });
    const deepseek = createProvider();
    const hookState = createHookState({
      providers: [limeHub, deepseek],
      selectedProviderId: "lime-hub",
      selectedProvider: limeHub,
      filteredProviders: [deepseek],
    });

    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    act(() => {
      root.render(
        <ApiKeyProviderSection exposeOemLoginPrompt onOemLogin={onOemLogin} />,
      );
    });
    mountedRoots.push({ container, root });
    await flushEffects();

    expect(maybeProviderItem(container, "lime-hub")).not.toBeNull();
    expect(container.textContent ?? "").toContain("Lime 云端");
    expect(container.textContent ?? "").toContain("需要登录");
    expect(container.textContent ?? "").toContain(
      "登录后会自动同步 Lime Hub 的可用模型",
    );
    expect(hookState.selectProvider).not.toHaveBeenCalledWith("deepseek");

    await act(async () => {
      maybeByTestId<HTMLButtonElement>(
        container,
        "provider-login-button",
      )?.click();
      await Promise.resolve();
    });

    expect(onOemLogin).toHaveBeenCalledTimes(1);
  });

  it("Provider 设置页请求聊天试跑时应走真实聊天测试命令", async () => {
    createHookState();
    renderSection();
    await flushEffects();

    await act(async () => {
      findByTestId<HTMLButtonElement>(
        "provider-setting-test-chat-stub",
      ).click();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockTestChat).toHaveBeenCalledWith(
      "deepseek",
      "deepseek-chat",
      "试跑",
    );
    expect(mockTestConnection).not.toHaveBeenCalled();
  });

  it("点击添加模型后，右侧进入可筛选的服务商目录", async () => {
    createHookState();
    const container = renderSection();

    await act(async () => {
      findByTestId<HTMLButtonElement>("add-model-button").click();
      await Promise.resolve();
    });

    expect(maybeByTestId(container, "model-add-catalog")).not.toBeNull();
    expect(container.textContent ?? "").toContain("推荐服务");
    expect(
      maybeByTestId(container, "custom-provider-template-card"),
    ).not.toBeNull();
    expect(maybeByTestId(container, "provider-setting-stub")).toBeNull();
  });

  it("添加流程应提供与详情页一致的接口获取模型入口，并自动填入小模型列表", async () => {
    mockFetchProviderModelsAuto.mockResolvedValueOnce({
      source: "Api",
      models: [{ id: "deepseek-v4-pro" }, { id: "deepseek-v4-flash" }],
      error: null,
    });
    const deepseek = createProvider({
      enabled: false,
      models: [],
      api_keys: [],
      api_key_count: 0,
    });
    const hookState = createHookState({
      providers: [deepseek],
      selectedProviderId: null,
      selectedProvider: null,
      filteredProviders: [deepseek],
    });
    const container = renderSection();

    await act(async () => {
      findByTestId<HTMLButtonElement>("add-model-button").click();
      await flushEffects(4);
    });

    await act(async () => {
      findByTestId<HTMLButtonElement>("model-catalog-category-cn").click();
      await Promise.resolve();
    });

    await act(async () => {
      maybeTemplateCard(container, "catalog-deepseek")?.click();
      await Promise.resolve();
    });

    await act(async () => {
      setInputValue(
        findByTestId<HTMLInputElement>("model-api-key-input"),
        "sk-test",
      );
      await Promise.resolve();
    });

    await act(async () => {
      findByTestId<HTMLButtonElement>("fetch-models-button").click();
      await flushEffects(6);
    });

    expect(hookState.updateProvider).toHaveBeenCalledWith(
      "deepseek",
      expect.objectContaining({
        type: "openai",
        api_host: "https://api.deepseek.com",
        enabled: true,
        models: [],
      }),
    );
    expect(hookState.addApiKey).toHaveBeenCalledWith(
      "deepseek",
      "sk-test",
      undefined,
      { replaceExisting: true },
    );
    expect(mockFetchProviderModelsAuto).toHaveBeenCalledWith("deepseek");
    expect(container.textContent ?? "").toContain("从接口获取");
    expect(container.textContent ?? "").toContain("已自动加入模型优先级");
    expect(container.textContent ?? "").toContain("deepseek-v4-pro");
    expect(container.textContent ?? "").toContain("deepseek-v4-flash");
  });

  it("添加流程的 API Key 获取链接应走 Desktop Host 外链网关", async () => {
    createHookState();
    const container = renderSection();

    await act(async () => {
      findByTestId<HTMLButtonElement>("add-model-button").click();
      await flushEffects(4);
    });

    await act(async () => {
      findByTestId<HTMLButtonElement>(
        "model-catalog-category-overseas",
      ).click();
      await Promise.resolve();
    });

    await act(async () => {
      maybeTemplateCard(container, "kimi-code-subscription")?.click();
      await Promise.resolve();
    });

    const link = findByTestId<HTMLAnchorElement>("provider-api-key-link");
    expect(link.getAttribute("href")).toBe("https://www.kimi.com/code");
    expect(link.getAttribute("target")).toBeNull();
    expect(link.getAttribute("rel")).toBe("noreferrer noopener");

    const clickEvent = new MouseEvent("click", {
      bubbles: true,
      cancelable: true,
    });
    await act(async () => {
      link.dispatchEvent(clickEvent);
      await Promise.resolve();
    });

    expect(clickEvent.defaultPrevented).toBe(true);
    expect(mockOpenExternalUrlWithSystemBrowser).toHaveBeenCalledWith(
      "https://www.kimi.com/code",
    );
  });

  it("系统 Provider 删除配置应清空模型、删除本地 Key 并停用入口", async () => {
    const hookState = createHookState();
    renderSection();

    await act(async () => {
      findByTestId<HTMLButtonElement>("provider-setting-delete-stub").click();
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(hookState.deleteApiKey).toHaveBeenCalledWith("key-1");
    expect(hookState.updateProvider).toHaveBeenCalledWith("deepseek", {
      enabled: false,
      models: [],
    });
    expect(hookState.selectProvider).toHaveBeenCalledWith(null);
    expect(hookState.deleteCustomProvider).not.toHaveBeenCalled();
  });

  it("添加流程中点击左侧已有模型，应退出目录并展开该模型配置", async () => {
    const hookState = createHookState();
    const container = renderSection();

    await act(async () => {
      findByTestId<HTMLButtonElement>("add-model-button").click();
      await Promise.resolve();
    });
    expect(maybeByTestId(container, "model-add-catalog")).not.toBeNull();

    await act(async () => {
      maybeProviderItem(container, "deepseek")?.click();
      await Promise.resolve();
    });

    expect(hookState.selectProvider).toHaveBeenCalledWith("deepseek");
    expect(maybeByTestId(container, "model-add-catalog")).toBeNull();
    expect(maybeByTestId(container, "provider-setting-stub")).not.toBeNull();
  });

  it("国内分类应展示 DeepSeek，资源模型目录里的渠道也应进入添加列表", async () => {
    createHookState();
    const container = renderSection();

    await act(async () => {
      findByTestId<HTMLButtonElement>("add-model-button").click();
      await Promise.resolve();
      await Promise.resolve();
    });

    await act(async () => {
      findByTestId<HTMLButtonElement>("model-catalog-category-cn").click();
      await Promise.resolve();
    });

    expect(container.textContent ?? "").toContain("DeepSeek");
    expect(container.textContent ?? "").toContain("Kimi API（国内按量）");
    expect(container.textContent ?? "").toContain("GLM Coding Plan（国内）");
    expect(container.textContent ?? "").not.toContain("Kimi Code 会员（订阅）");
    expect(container.textContent ?? "").not.toContain(
      "Z.AI Coding Plan（海外）",
    );
    expect(maybeByTestId(container, "model-add-catalog")?.className).toContain(
      "overflow-y-auto",
    );

    await act(async () => {
      maybeTemplateCard(container, "glm-cn-coding-plan")?.click();
      await Promise.resolve();
    });

    expect(container.textContent ?? "").toContain(
      "https://open.bigmodel.cn/api/anthropic",
    );
  });

  it("国内分类里的 SenseNova 应使用 v2 OpenAI 兼容接口", async () => {
    mockGetSystemProviderCatalog.mockResolvedValueOnce([
      {
        id: "sensenova",
        name: "SenseNova",
        type: "openai",
        api_host: "https://api.sensenova.cn/compatible-mode/v2",
        group: "chinese",
        sort_order: 29,
        legacy_ids: [],
      },
    ]);
    createHookState();
    const container = renderSection();

    await act(async () => {
      findByTestId<HTMLButtonElement>("add-model-button").click();
      await Promise.resolve();
      await Promise.resolve();
    });

    await act(async () => {
      findByTestId<HTMLButtonElement>("model-catalog-category-cn").click();
      await Promise.resolve();
    });

    await act(async () => {
      maybeTemplateCard(container, "catalog-sensenova")?.click();
      await Promise.resolve();
    });

    expect(container.textContent ?? "").toContain(
      "https://api.sensenova.cn/compatible-mode/v2",
    );
    expect(container.textContent ?? "").toContain("SenseChat-5");
  });

  it("推荐 Agnes 模板应使用 OpenAI 格式和官方图片模型", async () => {
    const hookState = createHookState();
    const container = renderSection();

    await act(async () => {
      findByTestId<HTMLButtonElement>("add-model-button").click();
      await Promise.resolve();
      await Promise.resolve();
    });

    await act(async () => {
      maybeTemplateCard(container, "agnes-image-flash")?.click();
      await Promise.resolve();
    });

    expect(container.textContent ?? "").toContain("Agnes");
    expect(container.textContent ?? "").toContain(
      "https://apihub.agnes-ai.com/v1",
    );
    expect(container.textContent ?? "").toContain("OpenAI 兼容");
    expect(container.textContent ?? "").toContain("agnes-image-2.1-flash");
    expect(container.textContent ?? "").not.toContain("Anthropic 格式");

    await act(async () => {
      setInputValue(
        findByTestId<HTMLInputElement>("model-api-key-input"),
        "sk-agnes-test",
      );
    });
    await clickByTestId("model-activate-button", 3);

    expect(hookState.addCustomProvider).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "Agnes",
        type: "openai",
        api_host: "https://apihub.agnes-ai.com/v1",
      }),
    );
    expect(hookState.updateProvider).toHaveBeenCalledWith(
      "custom-1",
      expect.objectContaining({
        enabled: true,
        models: [{ id: "agnes-image-2.1-flash" }],
      }),
    );
    expect(mockTestConnection).toHaveBeenCalledWith(
      "custom-1",
      "agnes-image-2.1-flash",
    );
    expect(hookState.selectProvider).toHaveBeenCalledWith("custom-1");
  });

  it("海外分类应展示国内厂商的国际订阅入口", async () => {
    createHookState();
    const container = renderSection();

    await act(async () => {
      findByTestId<HTMLButtonElement>("add-model-button").click();
      await Promise.resolve();
      await Promise.resolve();
    });

    await act(async () => {
      findByTestId<HTMLButtonElement>(
        "model-catalog-category-overseas",
      ).click();
      await Promise.resolve();
    });

    const text = container.textContent ?? "";
    expect(text).toContain("Kimi Code 会员（订阅）");
    expect(text).toContain("Kimi API（海外按量）");
    expect(text).toContain("Z.AI Coding Plan（海外）");
    expect(text).toContain("MiniMax Coding Plan（海外）");
    expect(text).toContain("Alibaba Coding Plan（海外）");
    expect(text).not.toContain("GLM Coding Plan（国内）");
    expect(
      maybeTemplateCard(container, "kimi-code-subscription"),
    ).not.toBeNull();
    const zaiTemplateCard = maybeTemplateCard(container, "zai-coding-plan");
    expect(zaiTemplateCard).not.toBeNull();
    const zaiReadableText = collectReadableText(zaiTemplateCard!);
    expect(zaiReadableText).toContain("Z.AI Coding Plan（海外）");
    expect(zaiReadableText).not.toContain(".bg { fill");
    expect(zaiReadableText).not.toContain(".fg { fill");
    expect(
      maybeTemplateCard(container, "minimax-coding-plan-global"),
    ).not.toBeNull();
    expect(
      maybeTemplateCard(container, "alibaba-coding-plan-global"),
    ).not.toBeNull();

    await act(async () => {
      maybeTemplateCard(container, "kimi-code-subscription")?.click();
      await Promise.resolve();
    });

    expect(container.textContent ?? "").toContain(
      "https://api.kimi.com/coding/",
    );
  });

  it("自定义供应商可在添加流程内完成创建、加 Key、写入模型并激活", async () => {
    const hookState = createHookState();
    renderSection();

    await openCustomProviderForm();
    await submitCustomProviderDraft({
      name: "My API",
      apiHost: "https://api.example.com/v1",
      model: "my-model",
    });

    expect(hookState.addCustomProvider).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "My API",
        type: "openai",
        api_host: "https://api.example.com/v1",
      }),
    );
    expect(hookState.updateProvider).toHaveBeenCalledWith(
      "custom-1",
      expect.objectContaining({
        enabled: true,
        models: [{ id: "my-model" }],
      }),
    );
    expect(hookState.addApiKey).toHaveBeenCalledWith(
      "custom-1",
      "sk-test",
      undefined,
      { replaceExisting: true },
    );
    expect(mockTestConnection).toHaveBeenCalledWith("custom-1", "my-model");
    expect(hookState.selectProvider).toHaveBeenCalledWith("custom-1");
  });

  it("添加流程应把 SenseNova 文档页修正为真实 API Base URL", async () => {
    const hookState = createHookState();
    renderSection();

    await openCustomProviderForm();
    await submitCustomProviderDraft({
      name: "SenseNova",
      apiHost: "https://platform.sensenova.cn/docs",
      model: "sensenova-test-model",
    });

    expect(hookState.addCustomProvider).toHaveBeenCalledWith(
      expect.objectContaining({
        api_host: "https://api.sensenova.cn/compatible-mode/v2",
      }),
    );
    expect(hookState.selectProvider).toHaveBeenCalledWith("custom-1");
  });

  it("添加流程在保存成功但连接测试失败时应保留表单并给出恢复路径", async () => {
    mockTestConnection.mockResolvedValueOnce({
      success: false,
      error: "模型无权限",
    });
    const hookState = createHookState();
    const container = renderSection();

    await openCustomProviderForm();
    const activateButton = findByTestId<HTMLButtonElement>(
      "model-activate-button",
    );
    expect(activateButton.className).toContain("bg-slate-950");
    expect(activateButton.className).toContain("text-white");
    await submitCustomProviderDraft({
      name: "My API",
      apiHost: "https://api.example.com/v1",
      model: "my-model",
    });

    expect(hookState.addCustomProvider).toHaveBeenCalled();
    expect(hookState.updateProvider).toHaveBeenCalledWith(
      "custom-1",
      expect.objectContaining({
        enabled: true,
        models: [{ id: "my-model" }],
      }),
    );
    expect(mockTestConnection).toHaveBeenCalledWith("custom-1", "my-model");
    expect(hookState.selectProvider).not.toHaveBeenCalled();
    const error = maybeByTestId(container, "model-add-error");
    expect(error?.getAttribute("role")).toBe("alert");
    expect(error?.textContent ?? "").toContain(
      "配置已保存，但连接测试未通过：模型无权限",
    );
    expect(maybeByTestId(container, "model-add-configure")).not.toBeNull();
    expect(
      findByTestId<HTMLButtonElement>("model-add-cancel-button").textContent,
    ).toContain("完成添加");

    await clickByTestId("model-add-cancel-button");

    expect(hookState.selectProvider).toHaveBeenCalledWith("custom-1");
    expect(maybeByTestId(container, "model-add-configure")).toBeNull();
  });

  it("未保存的自定义供应商退出动作应明确显示取消且不伪装成完成", async () => {
    const hookState = createHookState();
    const container = renderSection();

    await openCustomProviderForm();
    await fillCustomProviderDraft({
      name: "Unsaved API",
      apiHost: "https://api.example.com/v1",
      model: "unsaved-model",
    });

    const cancelButton = findByTestId<HTMLButtonElement>(
      "model-add-cancel-button",
    );
    expect(cancelButton.textContent).toContain("取消");
    expect(cancelButton.textContent).not.toContain("完成添加");

    await clickByTestId("model-add-cancel-button");

    expect(maybeByTestId(container, "model-add-configure")).toBeNull();
    expect(hookState.addCustomProvider).not.toHaveBeenCalled();
    expect(hookState.updateProvider).not.toHaveBeenCalled();
  });

  it("添加流程连接超时时不应向用户暴露 Desktop Host 内部命令", async () => {
    mockTestConnection.mockRejectedValueOnce(
      new Error(
        '[Electron] Desktop Host IPC 命令 "app_server_handle_json_lines" 在 5000ms 内未返回，已按 fail-closed 结束。',
      ),
    );
    const hookState = createHookState();
    const container = renderSection();

    await openCustomProviderForm();
    await submitCustomProviderDraft({
      name: "My API",
      apiHost: "https://api.example.com/v1",
      model: "my-model",
    });

    expect(hookState.selectProvider).not.toHaveBeenCalled();
    const errorText =
      maybeByTestId(container, "model-add-error")?.textContent ?? "";
    expect(errorText).toContain("配置已保存，但连接测试未通过");
    expect(errorText).toContain("连接测试超时");
    expect(errorText).not.toContain("app_server_handle_json_lines");
    expect(errorText).not.toContain("fail-closed");
  });
});
