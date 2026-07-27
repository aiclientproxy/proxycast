import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ProviderWithKeysDisplay } from "@/lib/api/apiKeyProvider";
import {
  findModelBoundImageCommandEntryForModel,
  getCurrentSkillCatalogSnapshot,
} from "@/lib/api/skillCatalog";
import {
  changeInput,
  createProvider,
  getApiModelSuggestionLabels,
  translate,
} from "./ProviderSetting.uiTestFixtures";

const { mockFetchProviderModelsAuto } = vi.hoisted(() => ({
  mockFetchProviderModelsAuto: vi.fn(),
}));

const { mockOpenExternalUrlWithSystemBrowser } = vi.hoisted(() => ({
  mockOpenExternalUrlWithSystemBrowser: vi.fn(),
}));

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
    t: translate,
  }),
}));

import { ProviderSetting } from "./ProviderSetting";

interface MountedRoot {
  container: HTMLDivElement;
  root: Root;
}

const mountedRoots: MountedRoot[] = [];

function renderSetting(
  provider: ProviderWithKeysDisplay | null,
  props: Partial<React.ComponentProps<typeof ProviderSetting>> = {},
) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(<ProviderSetting provider={provider} {...props} />);
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

function byTestId<T extends HTMLElement>(
  container: HTMLElement,
  testId: string,
): T | null {
  return container.querySelector<T>(`[data-testid="${testId}"]`);
}

beforeEach(() => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  window.localStorage.clear();
  vi.clearAllMocks();
  mockFetchProviderModelsAuto.mockResolvedValue({
    source: "Api",
    models: [{ id: "deepseek-chat" }],
    error: null,
  });
  mockOpenExternalUrlWithSystemBrowser.mockResolvedValue(undefined);
});

afterEach(() => {
  vi.clearAllMocks();
  window.localStorage.clear();

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
});

describe("ProviderSetting", () => {
  it("空状态应提示选择或添加模型", async () => {
    const container = renderSetting(null);
    await flushEffects();

    expect(container.textContent ?? "").toContain("选择或添加模型");
    expect(container.textContent ?? "").toContain("密钥、模型优先级和测试连接");
  });

  it("详情页应保留密钥保存、模型优先级和模型试跑", async () => {
    const container = renderSetting(createProvider());
    await flushEffects();
    const text = container.textContent ?? "";

    expect(text).toContain("DeepSeek");
    expect(text).toContain("API Host");
    expect(text).toContain("API 密钥");
    expect(text).toContain("保存密钥");
    expect(text).toContain("模型优先级");
    expect(text).toContain("从接口获取");
    expect(text).toContain("试跑当前模型");
    expect(text).toContain("主模型");
    expect(text).toContain("deepseek-chat");
    expect(text).not.toContain("协议配置表单");
    expect(text).not.toContain("连接验证");
    expect(text).not.toContain("支持的模型");
    expect(byTestId(container, "provider-simple-card")).not.toBeNull();
    expect(byTestId(container, "provider-api-host-input")).not.toBeNull();
    expect(
      byTestId(container, "provider-test-connection-button"),
    ).not.toBeNull();
    expect(byTestId(container, "provider-api-key-save-button")).not.toBeNull();
  });

  it("运行诊断缺少 API Key 时应在服务商页展示恢复提示", async () => {
    const container = renderSetting(
      createProvider({ api_key_count: 0, api_keys: [] }),
      {
        focus: {
          providerId: "deepseek",
          modelId: "deepseek-chat",
          reasonCode: "missing_enabled_api_key",
          recoveryAction: "add_enabled_api_key",
        },
      },
    );
    await flushEffects();

    const banner = byTestId<HTMLElement>(container, "provider-focus-banner");
    expect(banner).not.toBeNull();
    expect(banner?.textContent ?? "").toContain("补齐 API 密钥后可继续运行");
    expect(banner?.textContent ?? "").toContain("missing_enabled_api_key");
    expect(byTestId(container, "provider-api-key-input")).not.toBeNull();
  });

  it("运行诊断缺少模型时应允许一键加入模型优先级", async () => {
    const onUpdate = vi.fn().mockResolvedValue(undefined);
    const container = renderSetting(createProvider(), {
      focus: {
        providerId: "deepseek",
        modelId: "deepseek-coder-large",
        reasonCode: "model_not_enabled",
      },
      onUpdate,
    });
    await flushEffects();

    const button = byTestId<HTMLButtonElement>(
      container,
      "provider-focus-add-model-button",
    );
    expect(button).not.toBeNull();
    expect(container.textContent ?? "").toContain("补齐模型后可继续运行");
    expect(container.textContent ?? "").toContain("deepseek-coder-large");

    await act(async () => {
      button?.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(onUpdate).toHaveBeenCalledWith("deepseek", {
      models: [
        { id: "deepseek-chat" },
        { id: "deepseek-coder-large" },
      ],
    });
  });

  it("获取 API Key 链接应走 Desktop Host 外链网关", async () => {
    const container = renderSetting(createProvider());
    await flushEffects();

    const link = byTestId<HTMLAnchorElement>(
      container,
      "provider-api-key-link",
    );
    expect(link).not.toBeNull();
    expect(link?.getAttribute("href")).toBe(
      "https://platform.deepseek.com/api_keys",
    );
    expect(link?.getAttribute("target")).toBeNull();
    expect(link?.getAttribute("rel")).toBe("noreferrer noopener");

    const clickEvent = new MouseEvent("click", {
      bubbles: true,
      cancelable: true,
    });
    await act(async () => {
      link?.dispatchEvent(clickEvent);
      await Promise.resolve();
    });

    expect(clickEvent.defaultPrevented).toBe(true);
    expect(mockOpenExternalUrlWithSystemBrowser).toHaveBeenCalledWith(
      "https://platform.deepseek.com/api_keys",
    );
  });

  it("应支持修改并保存 Provider API Host", async () => {
    const onUpdate = vi.fn().mockResolvedValue(undefined);
    const container = renderSetting(
      createProvider({
        id: "xiaomi",
        name: "Mimo",
        type: "anthropic-compatible",
        api_host: "https://token-plan-cn.xiaomimimo.com/anthropic",
        models: [{ id: "mimo-v2.5-pro" }],
      }),
      {
        onUpdate,
      },
    );
    await flushEffects();

    const input = byTestId<HTMLInputElement>(
      container,
      "provider-api-host-input",
    );
    const saveButton = byTestId<HTMLButtonElement>(
      container,
      "provider-api-host-save-button",
    );

    expect(input?.value).toBe("https://token-plan-cn.xiaomimimo.com/anthropic");
    expect(saveButton).not.toBeNull();
    expect(saveButton?.disabled).toBe(true);

    await act(async () => {
      changeInput(input!, "https://token-plan-sgp.xiaomimimo.com/anthropic");
      await Promise.resolve();
    });

    expect(container.textContent ?? "").toContain("未保存");
    expect(saveButton?.disabled).toBe(false);

    await act(async () => {
      saveButton?.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(onUpdate).toHaveBeenCalledWith("xiaomi", {
      api_host: "https://token-plan-sgp.xiaomimimo.com/anthropic",
    });
    expect(container.textContent ?? "").toContain("API Host 已保存");
  });

  it("测试连接前应先保存未保存的 API Host", async () => {
    const callOrder: string[] = [];
    const onUpdate = vi.fn().mockImplementation(async () => {
      callOrder.push("update");
    });
    const onTestConnection = vi.fn().mockImplementation(async () => {
      callOrder.push("test");
      return {
        success: true,
        latencyMs: 96,
      };
    });
    const container = renderSetting(
      createProvider({
        id: "xiaomi",
        name: "Mimo",
        type: "anthropic-compatible",
        api_host: "https://token-plan-cn.xiaomimimo.com/anthropic",
        models: [{ id: "mimo-v2.5-pro" }],
      }),
      {
        onUpdate,
        onTestConnection,
      },
    );
    await flushEffects();

    const hostInput = byTestId<HTMLInputElement>(
      container,
      "provider-api-host-input",
    );
    const testButton = byTestId<HTMLButtonElement>(
      container,
      "provider-test-connection-button",
    );

    await act(async () => {
      changeInput(
        hostInput!,
        "https://token-plan-sgp.xiaomimimo.com/anthropic",
      );
      await Promise.resolve();
    });

    await act(async () => {
      testButton?.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(onUpdate).toHaveBeenCalledWith("xiaomi", {
      api_host: "https://token-plan-sgp.xiaomimimo.com/anthropic",
    });
    expect(onTestConnection).toHaveBeenCalledWith(
      "xiaomi",
      expect.objectContaining({
        modelName: "mimo-v2.5-pro",
        requireChatReady: true,
      }),
    );
    expect(callOrder).toEqual(["update", "test"]);
    expect(container.textContent ?? "").toContain("聊天试跑通过 · 96ms");
  });

  it("Lime Hub 未登录时应展示登录提示，不展示模型配置表单", async () => {
    const onLogin = vi.fn();
    const container = renderSetting(
      createProvider({
        id: "lime-hub",
        name: "Lime 云端",
        models: [],
        api_keys: [],
        api_key_count: 0,
      }),
      {
        authStatus: "login_required",
        onLogin,
      },
    );
    await flushEffects();

    expect(byTestId(container, "provider-login-required")).not.toBeNull();
    expect(container.textContent ?? "").toContain("Lime 云端");
    expect(container.textContent ?? "").toContain("需要登录");
    expect(container.textContent ?? "").toContain(
      "登录后会自动同步 Lime Hub 的可用模型",
    );
    expect(container.textContent ?? "").not.toContain("模型优先级");
    expect(container.textContent ?? "").not.toContain("从接口获取");

    await act(async () => {
      byTestId<HTMLButtonElement>(container, "provider-login-button")?.click();
      await Promise.resolve();
    });

    expect(onLogin).toHaveBeenCalledTimes(1);
  });

  it("手动添加模型应直接更新 models", async () => {
    const onUpdate = vi.fn().mockResolvedValue(undefined);
    const container = renderSetting(createProvider({ models: [] }), {
      onUpdate,
    });
    await flushEffects();

    const input = byTestId<HTMLInputElement>(container, "model-draft-input");
    const button = byTestId<HTMLButtonElement>(
      container,
      "model-draft-add-button",
    );

    expect(input).not.toBeNull();
    expect(button).not.toBeNull();

    await act(async () => {
      changeInput(input!, "deepseek-reasoner");
      await Promise.resolve();
    });

    await act(async () => {
      button?.click();
      await Promise.resolve();
    });

    expect(onUpdate).toHaveBeenCalledWith("deepseek", {
      models: [{ id: "deepseek-reasoner" }],
    });
    expect(container.textContent ?? "").toContain("deepseek-reasoner");
  });

  it("接口获取失败时不展示错误来源夹带的非实时模型", async () => {
    mockFetchProviderModelsAuto.mockResolvedValueOnce({
      source: "Error",
      models: [{ id: "wrong-fallback-model" }],
      error: "API 获取失败，本地模型兜底已下线。",
    });
    const container = renderSetting(createProvider({ models: [] }));
    await flushEffects();

    const button = byTestId<HTMLButtonElement>(
      container,
      "fetch-models-button",
    );

    await act(async () => {
      button?.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockFetchProviderModelsAuto).toHaveBeenCalledWith("deepseek");
    expect(container.textContent ?? "").toContain("API 获取失败");
    expect(container.textContent ?? "").not.toContain("wrong-fallback-model");
    expect(byTestId(container, "api-model-suggestions")).toBeNull();
  });

  it("图片模型行应能创建本地 @命令绑定", async () => {
    const container = renderSetting(
      createProvider({
        id: "airgate-openai-images",
        name: "OpenAI Images",
        type: "openai",
        api_host: "https://api.openai.com/v1",
        models: [{ id: "gpt-images-2" }],
      }),
    );
    await flushEffects();

    const createButton = byTestId<HTMLButtonElement>(
      container,
      "create-image-command-button",
    );
    expect(createButton).not.toBeNull();
    expect(createButton?.textContent ?? "").toContain("创建 @命令");

    await act(async () => {
      createButton?.click();
      await Promise.resolve();
    });

    const input = byTestId<HTMLInputElement>(
      container,
      "image-command-trigger-input",
    );
    expect(input?.value).toBe("@GPT Images 2");

    await act(async () => {
      byTestId<HTMLButtonElement>(
        container,
        "image-command-save-button",
      )?.click();
      await Promise.resolve();
    });

    expect(container.textContent ?? "").toContain("已创建 @GPT Images 2");
    expect(container.textContent ?? "").toContain("已绑定 @GPT Images 2");
    expect(
      findModelBoundImageCommandEntryForModel(
        getCurrentSkillCatalogSnapshot(),
        "airgate-openai-images",
        "gpt-images-2",
      ),
    ).toMatchObject({
      commandKey: "image_model_gpt_images_2",
      binding: {
        requestDefaults: expect.objectContaining({
          modelBoundImageTask: "true",
          providerId: "airgate-openai-images",
          model: "gpt-images-2",
          executorMode: "responses_image_generation",
        }),
      },
    });
  });

  it("OpenAI 兼容中转图片模型行应能创建本地 @命令绑定", async () => {
    const container = renderSetting(
      createProvider({
        id: "agnes",
        name: "agnes",
        type: "openai",
        api_host: "https://agnes.example.test/v1",
        models: [{ id: "agnes-image-2.1-flash" }],
      }),
    );
    await flushEffects();

    const createButton = byTestId<HTMLButtonElement>(
      container,
      "create-image-command-button",
    );
    expect(createButton).not.toBeNull();

    await act(async () => {
      createButton?.click();
      await Promise.resolve();
    });

    const input = byTestId<HTMLInputElement>(
      container,
      "image-command-trigger-input",
    );
    expect(input?.value).toBe("@Agnes Image 2.1 Flash");

    await act(async () => {
      byTestId<HTMLButtonElement>(
        container,
        "image-command-save-button",
      )?.click();
      await Promise.resolve();
    });

    expect(container.textContent ?? "").toContain(
      "已创建 @Agnes Image 2.1 Flash",
    );
    expect(
      findModelBoundImageCommandEntryForModel(
        getCurrentSkillCatalogSnapshot(),
        "agnes",
        "agnes-image-2.1-flash",
      ),
    ).toMatchObject({
      commandKey: "image_model_agnes_image_2_1_flash",
      binding: {
        requestDefaults: expect.objectContaining({
          modelBoundImageTask: "true",
          providerId: "agnes",
          model: "agnes-image-2.1-flash",
          executorMode: "images_api",
        }),
      },
    });
  });

  it("Fal Provider 没有 API Key 时仍可确认手动声明模型", async () => {
    mockFetchProviderModelsAuto.mockResolvedValueOnce({
      source: "Api",
      models: [{ id: "fal-ai/nano-banana-pro" }],
      error: null,
    });
    const container = renderSetting(
      createProvider({
        id: "fal",
        name: "Fal",
        type: "openai",
        api_host: "https://fal.run/fal-ai",
        api_key_count: 0,
        api_keys: [],
        models: [{ id: "fal-ai/nano-banana-pro" }],
      }),
    );
    await flushEffects();

    const fetchButton = byTestId<HTMLButtonElement>(
      container,
      "fetch-models-button",
    );
    const connectionButton = byTestId<HTMLButtonElement>(
      container,
      "provider-test-connection-button",
    );
    expect(fetchButton?.disabled).toBe(false);
    expect(connectionButton?.disabled).toBe(true);
    expect(container.textContent ?? "").toContain(
      "先填写 API 密钥，再测试连接",
    );
    expect(container.textContent ?? "").not.toContain("API 密钥（可选）");

    await act(async () => {
      fetchButton?.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockFetchProviderModelsAuto).toHaveBeenCalledWith("fal");
    const status = byTestId<HTMLElement>(container, "model-fetch-status");
    expect(status?.textContent ?? "").toContain("已确认 1 个模型");
    expect(status?.textContent ?? "").not.toContain("请先填写并保存 API 密钥");
  });

  it("接口获取成功后点击模型建议才加入优先级", async () => {
    const onUpdate = vi.fn().mockResolvedValue(undefined);
    mockFetchProviderModelsAuto.mockResolvedValueOnce({
      source: "Api",
      models: [{ id: "deepseek-chat" }, { id: "deepseek-reasoner" }],
      error: null,
    });
    const container = renderSetting(createProvider({ models: [] }), {
      onUpdate,
    });
    await flushEffects();

    await act(async () => {
      container
        .querySelector<HTMLButtonElement>('[data-testid="fetch-models-button"]')
        ?.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(container.textContent ?? "").toContain("接口返回 2 个模型");
    expect(onUpdate).not.toHaveBeenCalled();

    const suggestions = Array.from(
      container.querySelectorAll<HTMLButtonElement>(
        '[data-testid="api-model-suggestion"]',
      ),
    );
    expect(suggestions.map((button) => button.textContent?.trim())).toContain(
      "deepseek-chat",
    );

    await act(async () => {
      suggestions[0]?.click();
      await Promise.resolve();
    });

    expect(onUpdate).toHaveBeenCalledWith("deepseek", {
      models: [{ id: "deepseek-chat" }],
    });
  });

  it("接口获取成功后应展示完整模型列表，并支持筛选长列表", async () => {
    const apiModels = Array.from({ length: 12 }, (_, index) => ({
      id: `provider-model-${String(index + 1).padStart(2, "0")}`,
    }));
    mockFetchProviderModelsAuto.mockResolvedValueOnce({
      source: "Api",
      models: apiModels,
      error: null,
    });
    const container = renderSetting(createProvider({ models: [] }));
    await flushEffects();

    await act(async () => {
      container
        .querySelector<HTMLButtonElement>('[data-testid="fetch-models-button"]')
        ?.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(container.textContent ?? "").toContain("接口返回 12 个模型");
    expect(container.textContent ?? "").toContain("显示 12 / 12 个");
    expect(getApiModelSuggestionLabels(container)).toHaveLength(12);
    expect(getApiModelSuggestionLabels(container)).toContain(
      "provider-model-12",
    );

    const filterInput = byTestId<HTMLInputElement>(
      container,
      "api-model-filter-input",
    );

    await act(async () => {
      changeInput(filterInput!, "model-12");
      await Promise.resolve();
    });

    expect(container.textContent ?? "").toContain("显示 1 / 12 个");
    expect(getApiModelSuggestionLabels(container)).toEqual([
      "provider-model-12",
    ]);
  });

  it("非系统 Provider 应展示删除配置按钮，并在确认后调用删除回调", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    const onDeleteProvider = vi.fn().mockResolvedValue(true);
    const container = renderSetting(createProvider(), {
      onDeleteProvider,
    });
    await flushEffects();

    const button = byTestId<HTMLButtonElement>(
      container,
      "provider-delete-button",
    );

    expect(button).not.toBeNull();
    expect(button?.textContent ?? "").toContain("删除配置");

    await act(async () => {
      button?.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(confirmSpy).toHaveBeenCalledWith(
      "确认删除「DeepSeek」配置？此操作会移除该服务商和关联密钥。",
    );
    expect(onDeleteProvider).toHaveBeenCalledWith("deepseek");

    confirmSpy.mockRestore();
  });

  it("系统 Provider 也应展示删除配置按钮，并说明只移除本地配置", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    const onDeleteProvider = vi.fn().mockResolvedValue(true);
    const container = renderSetting(createProvider({ is_system: true }), {
      onDeleteProvider,
    });
    await flushEffects();

    const button = byTestId<HTMLButtonElement>(
      container,
      "provider-delete-button",
    );

    expect(button).not.toBeNull();

    await act(async () => {
      button?.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(confirmSpy).toHaveBeenCalledWith(
      "确认删除「DeepSeek」配置？此操作会移除该服务商的已启用模型和本地 API 密钥，系统服务商入口仍可重新添加。",
    );
    expect(onDeleteProvider).toHaveBeenCalledWith("deepseek");

    confirmSpy.mockRestore();
  });

  it("编辑 API 密钥后可显式保存，不必先测试连接", async () => {
    const onAddApiKey = vi.fn().mockResolvedValue(undefined);
    const onTestConnection = vi.fn().mockResolvedValue({
      success: true,
    });
    const container = renderSetting(createProvider(), {
      onAddApiKey,
      onTestConnection,
    });
    await flushEffects();

    const input = byTestId<HTMLInputElement>(
      container,
      "provider-api-key-input",
    );
    const saveButton = byTestId<HTMLButtonElement>(
      container,
      "provider-api-key-save-button",
    );

    expect(saveButton).not.toBeNull();
    expect(saveButton?.disabled).toBe(true);

    await act(async () => {
      changeInput(input!, "sk-updated-key");
      await Promise.resolve();
    });

    expect(container.textContent ?? "").toContain("未保存");
    expect(saveButton?.disabled).toBe(false);

    await act(async () => {
      saveButton?.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(onAddApiKey).toHaveBeenCalledWith(
      "deepseek",
      "sk-updated-key",
      undefined,
      { replaceExisting: true },
    );
    expect(onTestConnection).not.toHaveBeenCalled();
    expect(container.textContent ?? "").toContain("API 密钥已保存");
    expect(byTestId(container, "api-key-status")).not.toBeNull();
  });

  it("保存 API 密钥遇到重复 Key 时不应假装替换成功", async () => {
    const onAddApiKey = vi.fn().mockRejectedValue("该 API Key 已存在");
    const container = renderSetting(createProvider(), {
      onAddApiKey,
    });
    await flushEffects();

    const input = byTestId<HTMLInputElement>(
      container,
      "provider-api-key-input",
    );
    const saveButton = byTestId<HTMLButtonElement>(
      container,
      "provider-api-key-save-button",
    );

    await act(async () => {
      changeInput(input!, "sk-duplicated-key");
      await Promise.resolve();
    });

    await act(async () => {
      saveButton?.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    const status = byTestId<HTMLElement>(container, "api-key-status");
    expect(onAddApiKey).toHaveBeenCalledWith(
      "deepseek",
      "sk-duplicated-key",
      undefined,
      { replaceExisting: true },
    );
    expect(status?.textContent ?? "").toContain("该 API Key 已存在");
    expect(status?.className ?? "").toContain("border-rose-200");
    expect(status?.textContent ?? "").not.toContain(
      "这个 API 密钥已在当前服务商中",
    );
  });

  it("保存 API 密钥失败时应展示后端返回的字符串原因", async () => {
    const onAddApiKey = vi.fn().mockRejectedValue("Provider 不存在: missing");
    const container = renderSetting(createProvider(), {
      onAddApiKey,
    });
    await flushEffects();

    const input = byTestId<HTMLInputElement>(
      container,
      "provider-api-key-input",
    );
    const saveButton = byTestId<HTMLButtonElement>(
      container,
      "provider-api-key-save-button",
    );

    await act(async () => {
      changeInput(input!, "sk-missing-provider");
      await Promise.resolve();
    });

    await act(async () => {
      saveButton?.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    const status = byTestId<HTMLElement>(container, "api-key-status");
    expect(onAddApiKey).toHaveBeenCalledWith(
      "deepseek",
      "sk-missing-provider",
      undefined,
      { replaceExisting: true },
    );
    expect(status?.textContent ?? "").toContain("Provider 不存在: missing");
    expect(status?.textContent ?? "").not.toContain("保存 API 密钥失败");
  });

  it("测试连接应先保存新密钥，并只显示简洁状态", async () => {
    const onAddApiKey = vi.fn().mockResolvedValue(undefined);
    const onTestConnection = vi.fn().mockResolvedValue({
      success: true,
      latencyMs: 128,
    });
    const container = renderSetting(
      createProvider({ api_key_count: 0, api_keys: [] }),
      {
        onAddApiKey,
        onTestConnection,
      },
    );
    await flushEffects();

    const input = byTestId<HTMLInputElement>(
      container,
      "provider-api-key-input",
    );
    const button = byTestId<HTMLButtonElement>(
      container,
      "provider-test-connection-button",
    );

    await act(async () => {
      changeInput(input!, "sk-new-key");
      await Promise.resolve();
    });

    expect(button?.disabled).toBe(false);

    await act(async () => {
      button?.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(onAddApiKey).toHaveBeenCalledWith(
      "deepseek",
      "sk-new-key",
      undefined,
      { replaceExisting: true },
    );
    expect(onTestConnection).toHaveBeenCalledWith(
      "deepseek",
      expect.objectContaining({
        modelName: "deepseek-chat",
        requireChatReady: true,
      }),
    );
    expect(container.textContent ?? "").toContain("聊天试跑通过 · 128ms");
    expect(container.textContent ?? "").not.toContain("错误详情");
    expect(container.textContent ?? "").not.toContain("对话测试");
  });

  it("模型试跑遇到 402 余额不足时应展示可操作提示", async () => {
    const onTestConnection = vi.fn().mockResolvedValue({
      success: false,
      error:
        "Agent provider execution failed: Request failed with status 402 Payment Required: Insufficient Balance",
    });
    const container = renderSetting(createProvider(), {
      onTestConnection,
    });
    await flushEffects();

    const button = byTestId<HTMLButtonElement>(
      container,
      "provider-test-connection-button",
    );

    await act(async () => {
      button?.click();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(onTestConnection).toHaveBeenCalledWith(
      "deepseek",
      expect.objectContaining({
        modelName: "deepseek-chat",
        requireChatReady: true,
      }),
    );
    expect(container.textContent ?? "").toMatch(
      /当前模型通道返回了计费或额度类错误|The current model channel returned a billing or quota error/,
    );
    expect(container.textContent ?? "").not.toContain("Insufficient Balance");
  });
});
