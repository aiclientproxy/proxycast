import { act, useEffect, useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  cleanupMountedRoots,
  flushEffects,
  renderIntoDom,
  setReactActEnvironment,
  type MountedRoot,
} from "./test-utils";

const { mockRefreshProviders, mockUseApiKeyProvider } = vi.hoisted(() => ({
  mockRefreshProviders: vi.fn(async () => undefined),
  mockUseApiKeyProvider: vi.fn(),
}));

vi.mock("@/hooks/useApiKeyProvider", () => ({
  useApiKeyProvider: (options?: { autoLoad?: boolean }) =>
    mockUseApiKeyProvider(options),
}));

import { useImageGen } from "./useImageGen";

interface HookHarness {
  getValue: () => ReturnType<typeof useImageGen>;
}

const mountedRoots: MountedRoot[] = [];

function createImageProviders() {
  return [
    {
      id: "zhipuai",
      type: "zhipuai",
      name: "智谱AI",
      enabled: true,
      api_key_count: 1,
      api_host: "https://api.zhipu.test",
    },
    {
      id: "fal",
      type: "fal",
      name: "Fal",
      enabled: true,
      api_key_count: 1,
      api_host: "https://fal.run",
    },
  ];
}

function mountHook(preferredProviderId?: string): HookHarness {
  let hookValue: ReturnType<typeof useImageGen> | null = null;

  function TestComponent() {
    hookValue = useImageGen({ preferredProviderId });
    return null;
  }

  renderIntoDom(<TestComponent />, mountedRoots);

  return {
    getValue: () => {
      if (!hookValue) {
        throw new Error("hook 尚未初始化");
      }
      return hookValue;
    },
  };
}

function mountHookWithOptions(options: {
  preferredProviderId?: string;
  preferredModelId?: string;
  allowFallback?: boolean;
  providerLoadEnabled?: boolean;
  providerLoadMode?: "immediate" | "deferred";
  selectionScopeKey?: string;
}): HookHarness {
  let hookValue: ReturnType<typeof useImageGen> | null = null;

  function TestComponent() {
    hookValue = useImageGen(options);
    return null;
  }

  renderIntoDom(<TestComponent />, mountedRoots);

  return {
    getValue: () => {
      if (!hookValue) {
        throw new Error("hook 尚未初始化");
      }
      return hookValue;
    },
  };
}

function mountHookWithReactiveOptions(initialOptions: {
  preferredProviderId?: string;
  preferredModelId?: string;
  allowFallback?: boolean;
  selectionScopeKey?: string;
}): HookHarness & {
  updateOptions: (next: {
    preferredProviderId?: string;
    preferredModelId?: string;
    allowFallback?: boolean;
    selectionScopeKey?: string;
  }) => Promise<void>;
} {
  let hookValue: ReturnType<typeof useImageGen> | null = null;
  let setOptions:
    | ((next: {
        preferredProviderId?: string;
        preferredModelId?: string;
        allowFallback?: boolean;
        selectionScopeKey?: string;
      }) => void)
    | null = null;

  function TestComponent() {
    const [options, setInnerOptions] = useState(initialOptions);
    useEffect(() => {
      setOptions = setInnerOptions;
    }, []);
    hookValue = useImageGen(options);
    return null;
  }

  renderIntoDom(<TestComponent />, mountedRoots);

  return {
    getValue: () => {
      if (!hookValue) {
        throw new Error("hook 尚未初始化");
      }
      return hookValue;
    },
    updateOptions: async (next) => {
      await act(async () => {
        setOptions?.(next);
        await Promise.resolve();
      });
    },
  };
}

async function waitForReady(harness: HookHarness, timeout = 40): Promise<void> {
  for (let i = 0; i < timeout; i += 1) {
    const value = harness.getValue();
    if (value.selectedProvider && value.selectedModelId) {
      return;
    }
    await flushEffects();
  }
  throw new Error("useImageGen 未在预期时间内就绪");
}

beforeEach(() => {
  setReactActEnvironment();
  mockRefreshProviders.mockClear();
  const providers = createImageProviders();
  mockUseApiKeyProvider.mockImplementation(
    (options?: { autoLoad?: boolean }) => ({
      providers: options?.autoLoad === false ? [] : providers,
      loading: false,
      refresh: mockRefreshProviders,
    }),
  );
});

afterEach(() => {
  cleanupMountedRoots(mountedRoots);
  vi.restoreAllMocks();
});

describe("useImageGen 项目偏好", () => {
  it("应优先选择项目指定的图片 Provider 和模型", async () => {
    const harness = mountHook("fal");
    await act(async () => {
      await waitForReady(harness);
    });

    expect(harness.getValue().selectedProvider?.id).toBe("fal");
    expect(harness.getValue().selectedModelId).toBe("fal-ai/nano-banana-pro");
  });

  it("应优先选择项目指定的图片模型", async () => {
    const harness = mountHookWithOptions({
      preferredProviderId: "fal",
      preferredModelId: "fal-ai/flux-kontext/dev",
    });
    await act(async () => {
      await waitForReady(harness);
    });

    expect(harness.getValue().selectedProvider?.id).toBe("fal");
    expect(harness.getValue().selectedModelId).toBe("fal-ai/flux-kontext/dev");
  });

  it("偏好在首次挂载后到达时，也应切换到设置指定的图片 Provider 和模型", async () => {
    const harness = mountHookWithReactiveOptions({});
    await act(async () => {
      await waitForReady(harness);
    });

    expect(harness.getValue().selectedProvider?.id).toBe("fal");

    await harness.updateOptions({
      preferredProviderId: "fal",
      preferredModelId: "fal-ai/flux-kontext/dev",
    });

    await act(async () => {
      await waitForReady(harness);
    });

    expect(harness.getValue().selectedProvider?.id).toBe("fal");
    expect(harness.getValue().selectedModelId).toBe("fal-ai/flux-kontext/dev");
  });

  it("未显式指定图片 Provider 时，应优先命中 openai-like 图片渠道", async () => {
    const providers = [
      {
        id: "zhipuai",
        type: "zhipuai",
        name: "智谱AI",
        enabled: true,
        api_key_count: 1,
        api_host: "https://api.zhipu.test",
      },
      {
        id: "airgate-openai-images",
        type: "openai",
        name: "OpenAI-gpt-images-2",
        enabled: true,
        api_key_count: 1,
        api_host: "https://airgate.k8ray.com/v1",
        models: [{ id: "gpt-images-2" }],
      },
    ];
    mockUseApiKeyProvider.mockImplementation(
      (options?: { autoLoad?: boolean }) => ({
        providers: options?.autoLoad === false ? [] : providers,
        loading: false,
        refresh: mockRefreshProviders,
      }),
    );
    const harness = mountHookWithOptions({});
    await act(async () => {
      await waitForReady(harness);
    });

    expect(harness.getValue().selectedProvider?.id).toBe(
      "airgate-openai-images",
    );
    expect(harness.getValue().selectedModelId).toBe("gpt-images-2");
  });

  it("默认图片服务不可用且禁止回退时，不应偷偷切到其他 Provider", async () => {
    const harness = mountHookWithOptions({
      preferredProviderId: "missing-provider",
      allowFallback: false,
    });
    await act(async () => {
      await flushEffects();
    });

    expect(harness.getValue().selectedProvider).toBeUndefined();
    expect(harness.getValue().selectedProviderId).toBe("");
    expect(harness.getValue().preferredProviderUnavailable).toBe(true);
  });

  it("禁用自动 Provider 加载时不应读取列表，显式请求后再加载", async () => {
    const harness = mountHookWithOptions({
      providerLoadEnabled: false,
    });

    await act(async () => {
      await flushEffects();
    });

    expect(mockUseApiKeyProvider).toHaveBeenLastCalledWith({
      autoLoad: false,
    });
    expect(harness.getValue().availableProviders).toEqual([]);

    await act(async () => {
      await harness.getValue().ensureProvidersLoaded();
      await flushEffects();
    });

    expect(mockRefreshProviders).toHaveBeenCalledTimes(1);
    expect(mockUseApiKeyProvider).toHaveBeenLastCalledWith({
      autoLoad: true,
    });
  });

  it("显式请求 Provider 加载时应强制刷新 Provider 列表", async () => {
    const harness = mountHookWithOptions({
      providerLoadMode: "deferred",
    });

    await act(async () => {
      await flushEffects();
    });

    expect(mockRefreshProviders).not.toHaveBeenCalled();

    await act(async () => {
      await harness.getValue().ensureProvidersLoaded();
      await flushEffects();
    });

    expect(mockRefreshProviders).toHaveBeenCalledTimes(1);
    expect(mockUseApiKeyProvider).toHaveBeenLastCalledWith({
      autoLoad: true,
    });
  });

  it("显式请求 Provider 加载时应立即打开加载闸门", async () => {
    const harness = mountHookWithOptions({
      providerLoadMode: "deferred",
    });

    await act(async () => {
      await harness.getValue().ensureProvidersLoaded();
      await flushEffects();
    });

    expect(mockUseApiKeyProvider).toHaveBeenLastCalledWith({
      autoLoad: true,
    });
  });

  it("切换会话范围后，应回到当前偏好的图片 Provider 和模型", async () => {
    const harness = mountHookWithReactiveOptions({
      preferredProviderId: "fal",
      preferredModelId: "fal-ai/flux-kontext/dev",
      selectionScopeKey: "project-a:session-a",
    });
    await act(async () => {
      await waitForReady(harness);
    });

    expect(harness.getValue().selectedProvider?.id).toBe("fal");
    expect(harness.getValue().selectedModelId).toBe("fal-ai/flux-kontext/dev");

    await act(async () => {
      harness.getValue().setSelectedProviderId("zhipuai");
      await Promise.resolve();
    });
    await act(async () => {
      await waitForReady(harness);
    });

    expect(harness.getValue().selectedProvider?.id).toBe("zhipuai");

    await harness.updateOptions({
      preferredProviderId: "fal",
      preferredModelId: "fal-ai/flux-kontext/dev",
      selectionScopeKey: "project-b:session-b",
    });

    await act(async () => {
      await waitForReady(harness);
    });

    expect(harness.getValue().selectedProvider?.id).toBe("fal");
    expect(harness.getValue().selectedModelId).toBe("fal-ai/flux-kontext/dev");
  });
});
