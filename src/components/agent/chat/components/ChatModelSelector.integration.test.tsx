import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";

const {
  mockGetRuntimeProviderSelection,
  mockCreateAgentRuntimeSession,
  mockListAgentRuntimeSessions,
  mockGetAgentRuntimeSession,
  mockUpdateAgentRuntimeThreadSettings,
  mockParseAgentEvent,
  mockSafeListen,
  mockToast,
  mockLoadConfiguredProviders,
  mockLoadProviderModels,
  mockUseConfiguredProviders,
  mockUseProviderModels,
  mockApiKeyProvidersGetProviders,
  mockGetDefaultProvider,
  mockEmitProviderDataChanged,
  mockWechatChannelSetRuntimeModel,
} = vi.hoisted(() => ({
  mockGetRuntimeProviderSelection: vi.fn(),
  mockCreateAgentRuntimeSession: vi.fn(),
  mockListAgentRuntimeSessions: vi.fn(),
  mockGetAgentRuntimeSession: vi.fn(),
  mockUpdateAgentRuntimeThreadSettings: vi.fn(),
  mockParseAgentEvent: vi.fn((payload: unknown) => payload),
  mockSafeListen: vi.fn(),
  mockToast: {
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
  },
  mockLoadConfiguredProviders: vi.fn(),
  mockLoadProviderModels: vi.fn(),
  mockUseConfiguredProviders: vi.fn(),
  mockUseProviderModels: vi.fn(),
  mockApiKeyProvidersGetProviders: vi.fn(),
  mockGetDefaultProvider: vi.fn(),
  mockEmitProviderDataChanged: vi.fn(),
  mockWechatChannelSetRuntimeModel: vi.fn(async () => undefined),
}));

vi.mock("@/lib/api/agentRuntime/agentClient", async () => {
  const actual = await vi.importActual<
    typeof import("@/lib/api/agentRuntime/agentClient")
  >("@/lib/api/agentRuntime/agentClient");

  return {
    ...actual,
    getRuntimeProviderSelection: mockGetRuntimeProviderSelection,
  };
});

vi.mock("@/lib/api/agentRuntime/sessionClient", async () => {
  const actual = await vi.importActual<
    typeof import("@/lib/api/agentRuntime/sessionClient")
  >("@/lib/api/agentRuntime/sessionClient");

  return {
    ...actual,
    createAgentRuntimeSession: mockCreateAgentRuntimeSession,
    listAgentRuntimeSessions: mockListAgentRuntimeSessions,
    getAgentRuntimeSession: mockGetAgentRuntimeSession,
  };
});

vi.mock("@/lib/api/agentProtocol", async () => {
  const actual = await vi.importActual<
    typeof import("@/lib/api/agentProtocol")
  >("@/lib/api/agentProtocol");
  return {
    ...actual,
    parseAgentEvent: mockParseAgentEvent,
  };
});

vi.mock("@/lib/dev-bridge", async () => {
  const actual =
    await vi.importActual<typeof import("@/lib/dev-bridge")>(
      "@/lib/dev-bridge",
    );

  return {
    ...actual,
    safeListen: mockSafeListen,
  };
});

vi.mock("sonner", () => ({
  toast: mockToast,
}));

vi.mock("@/hooks/useConfiguredProviders", () => ({
  loadConfiguredProviders: mockLoadConfiguredProviders,
  useConfiguredProviders: mockUseConfiguredProviders,
  findConfiguredProviderBySelection: (
    providers: Array<{ key: string; providerId?: string }>,
    selection?: string | null,
  ) => {
    const normalizedSelection = (selection || "").trim().toLowerCase();
    const keyMatch =
      providers.find(
        (provider) => provider.key.trim().toLowerCase() === normalizedSelection,
      ) ?? null;
    const providerIdMatch =
      providers.find(
        (provider) =>
          (provider.providerId || "").trim().toLowerCase() ===
          normalizedSelection,
      ) ?? null;

    if (keyMatch && providerIdMatch && keyMatch !== providerIdMatch) {
      if (!keyMatch.providerId && providerIdMatch.providerId) {
        return providerIdMatch;
      }
    }

    return keyMatch ?? providerIdMatch ?? null;
  },
}));

vi.mock("@/hooks/useProviderModels", () => ({
  loadProviderModels: mockLoadProviderModels,
  useProviderModels: mockUseProviderModels,
}));

vi.mock("@/lib/api/apiKeyProvider", () => ({
  apiKeyProviderApi: {
    getProviders: mockApiKeyProvidersGetProviders,
  },
}));

vi.mock("@/lib/api/appConfig", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api/appConfig")>(
    "@/lib/api/appConfig",
  );
  return {
    ...actual,
    getDefaultProvider: mockGetDefaultProvider,
  };
});

vi.mock("@/lib/providerDataEvents", () => ({
  emitProviderDataChanged: mockEmitProviderDataChanged,
}));

vi.mock("@/lib/api/channelsRuntime", () => ({
  wechatChannelSetRuntimeModel: mockWechatChannelSetRuntimeModel,
}));

import { useAgentChat } from "../hooks";
import type { AgentRuntimeAdapter } from "../hooks/agentRuntimeAdapter";
import { ChatModelSelector } from "./ChatModelSelector";

interface MountedRoot {
  container: HTMLDivElement;
  root: Root;
}

interface MountedHarness extends MountedRoot {
  getChat: () => ReturnType<typeof useAgentChat>;
}

const mountedRoots: MountedRoot[] = [];

interface MountOptions {
  onManageProviders?: () => void;
  chatModelSelectorProps?: Partial<
    React.ComponentProps<typeof ChatModelSelector>
  >;
}

function createModel(id: string, providerId: string) {
  return {
    id,
    display_name: id,
    provider_id: providerId,
    provider_name: providerId,
    family: null,
    tier: "pro",
    capabilities: {
      vision: false,
      tools: true,
      streaming: true,
      json_mode: true,
      function_calling: true,
      reasoning: false,
    },
    pricing: null,
    limits: {
      context_length: null,
      max_output_tokens: null,
      requests_per_minute: null,
      tokens_per_minute: null,
    },
    status: "active",
    release_date: null,
    is_latest: false,
    description: id,
    source: "custom",
    created_at: Date.now() / 1000,
    updated_at: Date.now() / 1000,
  };
}

function createConfiguredProviderFixtures() {
  return [
    { key: "gemini", label: "Gemini", registryId: "gemini", type: "gemini" },
    {
      key: "deepseek",
      label: "DeepSeek",
      registryId: "deepseek",
      type: "deepseek",
    },
  ];
}

function createProviderModelsFixture(providerKey?: string | null) {
  if (providerKey === "gemini") {
    return [
      createModel("gemini-2.5-pro", "gemini"),
      createModel("gemini-2.5-flash", "gemini"),
    ];
  }

  if (providerKey === "deepseek") {
    return [
      createModel("deepseek-chat", "deepseek"),
      createModel("deepseek-reasoner", "deepseek"),
    ];
  }

  return [];
}

function createRuntimeAdapterFixture(): AgentRuntimeAdapter {
  const unsupported = async () => {
    throw new Error("当前测试未覆盖该 runtime adapter 方法");
  };

  return {
    getRuntimeProviderSelection: () => mockGetRuntimeProviderSelection(),
    createSession: (workspaceId, name, executionStrategy, options) =>
      mockCreateAgentRuntimeSession(
        workspaceId,
        name,
        executionStrategy,
        options,
      ),
    listSessions: (options) => mockListAgentRuntimeSessions(options),
    getSession: (sessionId, options) =>
      mockGetAgentRuntimeSession(sessionId, options),
    getSessionReadModel: async () => null,
    getThreadTurnControl: unsupported,
    replayRequest: async () => null,
    renameSession: async (sessionId, title) => {
      void sessionId;
      void title;
    },
    deleteSession: unsupported,
    setSessionExecutionStrategy: async (sessionId, executionStrategy) => {
      void sessionId;
      void executionStrategy;
    },
    setSessionAccessMode: async (sessionId, accessMode) => {
      void sessionId;
      void accessMode;
    },
    setSessionProviderSelection: async (sessionId, providerType, model) => {
      await mockUpdateAgentRuntimeThreadSettings({
        threadId: sessionId,
        modelProvider: providerType,
        model,
      });
    },
    updateSessionMetadata: async (sessionId, patch) => {
      await mockUpdateAgentRuntimeThreadSettings({
        threadId: sessionId,
        ...(patch.providerType ? { modelProvider: patch.providerType } : {}),
        ...(patch.model ? { model: patch.model } : {}),
      });
    },
    generateSessionTitle: async () => "",
    submitOp: unsupported,
    steerTurn: unsupported,
    compactSession: unsupported,
    interruptTurn: async () => false,
    resumeThread: async () => false,
    respondToAction: unsupported,
    listenToTurnEvents: (eventName, handler) =>
      mockSafeListen(eventName, handler),
  };
}

function mount(
  workspaceId: string,
  options: MountOptions = {},
): MountedHarness {
  const { onManageProviders, chatModelSelectorProps } = options;
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  const runtimeAdapter = createRuntimeAdapterFixture();
  let chatValue: ReturnType<typeof useAgentChat> | null = null;

  function TestComponent() {
    const chat = useAgentChat({ workspaceId, runtimeAdapter });
    chatValue = chat;
    return (
      <div>
        <ChatModelSelector
          providerType={chat.providerType}
          setProviderType={chat.setProviderType}
          model={chat.model}
          setModel={chat.setModel}
          activeTheme="general"
          onManageProviders={onManageProviders}
          {...chatModelSelectorProps}
        />
        <div data-testid="current-model">
          {chat.providerType}/{chat.model}
        </div>
      </div>
    );
  }

  act(() => {
    root.render(<TestComponent />);
  });

  mountedRoots.push({ container, root });
  return {
    container,
    root,
    getChat: () => {
      if (!chatValue) {
        throw new Error("chat 尚未初始化");
      }
      return chatValue;
    },
  };
}

function getComboboxTrigger(container: HTMLElement): HTMLButtonElement {
  const trigger = container.querySelector(
    'button[role="combobox"]',
  ) as HTMLButtonElement | null;
  if (!trigger) {
    throw new Error("未找到模型选择触发器");
  }
  return trigger;
}

function findButtonByText(
  text: string,
  options: { excludeCombobox?: boolean } = {},
): HTMLButtonElement {
  const { excludeCombobox = false } = options;
  const target = Array.from(document.querySelectorAll("button")).find(
    (node) => {
      if (excludeCombobox && node.getAttribute("role") === "combobox") {
        return false;
      }
      return node.textContent?.includes(text);
    },
  );
  if (!target) {
    throw new Error(`未找到按钮文本: ${text}`);
  }
  return target as HTMLButtonElement;
}

async function flushEffects() {
  await act(async () => {
    await Promise.resolve();
  });
}

beforeEach(async () => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;

  await changeLimeLocale("zh-CN");

  vi.clearAllMocks();
  localStorage.clear();
  sessionStorage.clear();

  mockGetRuntimeProviderSelection.mockResolvedValue({
    provider_configured: false,
  });
  mockCreateAgentRuntimeSession.mockResolvedValue("created-session");
  mockUpdateAgentRuntimeThreadSettings.mockResolvedValue(undefined);
  mockSafeListen.mockResolvedValue(() => {});
  mockApiKeyProvidersGetProviders.mockResolvedValue([]);
  mockGetDefaultProvider.mockResolvedValue("");
  mockEmitProviderDataChanged.mockImplementation(() => {});
  mockWechatChannelSetRuntimeModel.mockResolvedValue(undefined);
  const configuredProviders = createConfiguredProviderFixtures();

  const createdAt = Math.floor(Date.now() / 1000);
  mockListAgentRuntimeSessions.mockResolvedValue([
    {
      id: "topic-a",
      name: "话题 A",
      created_at: createdAt,
      updated_at: createdAt,
      messages_count: 0,
    },
    {
      id: "topic-b",
      name: "话题 B",
      created_at: createdAt,
      updated_at: createdAt,
      messages_count: 0,
    },
  ]);
  mockGetAgentRuntimeSession.mockImplementation(async (topicId: string) => ({
    id: topicId,
    created_at: createdAt,
    updated_at: createdAt,
    messages: [],
    execution_strategy: "react",
    turns: [],
    items: [],
  }));

  mockUseConfiguredProviders.mockReturnValue({
    providers: configuredProviders,
    loading: false,
  });
  mockLoadConfiguredProviders.mockResolvedValue(configuredProviders);

  mockUseProviderModels.mockImplementation(
    (selectedProvider: { key: string } | null) => {
      const models = createProviderModelsFixture(selectedProvider?.key);

      return {
        modelIds: models.map((item) => item.id),
        models,
        loading: false,
        error: null,
      };
    },
  );
  mockLoadProviderModels.mockImplementation(
    async (selectedProvider?: { key?: string | null } | null) =>
      createProviderModelsFixture(selectedProvider?.key),
  );
});

afterEach(async () => {
  while (mountedRoots.length > 0) {
    const mounted = mountedRoots.pop();
    if (!mounted) break;
    act(() => {
      mounted.root.unmount();
    });
    mounted.container.remove();
  }
  localStorage.clear();
  sessionStorage.clear();
  await changeLimeLocale("zh-CN");
});

describe("ChatModelSelector + useAgentChat 集成", () => {
  it("通过 UI 选择模型后切换话题再切回，应恢复会话模型", async () => {
    const workspaceId = "ws-model-selector-integration";
    const harness = mount(workspaceId);
    const { container } = harness;

    await flushEffects();
    await flushEffects();

    await act(async () => {
      await harness.getChat().switchTopic("topic-a");
    });
    await flushEffects();
    await flushEffects();

    await act(async () => {
      getComboboxTrigger(container).click();
    });
    await flushEffects();

    await act(async () => {
      findButtonByText("Gemini", { excludeCombobox: true }).click();
    });
    await flushEffects();

    await act(async () => {
      findButtonByText("gemini-2.5-pro", { excludeCombobox: true }).click();
    });
    await flushEffects();

    await act(async () => {
      await harness.getChat().switchTopic("topic-b");
    });
    await flushEffects();
    await flushEffects();

    await act(async () => {
      getComboboxTrigger(container).click();
    });
    await flushEffects();

    await act(async () => {
      findButtonByText("DeepSeek", { excludeCombobox: true }).click();
    });
    await flushEffects();

    await act(async () => {
      findButtonByText("deepseek-chat", { excludeCombobox: true }).click();
    });
    await flushEffects();

    await act(async () => {
      await harness.getChat().switchTopic("topic-a");
    });
    await flushEffects();
    await flushEffects();

    const currentModel = container.querySelector(
      '[data-testid="current-model"]',
    ) as HTMLDivElement | null;
    expect(currentModel?.textContent).toContain("gemini/gemini-2.5-pro");

    expect(
      JSON.parse(
        localStorage.getItem(`agent_topic_model_pref_${workspaceId}_topic-a`) ||
          "null",
      ),
    ).toEqual({
      providerType: "gemini",
      model: "gemini-2.5-pro",
    });
    expect(
      JSON.parse(
        localStorage.getItem(`agent_topic_model_pref_${workspaceId}_topic-b`) ||
          "null",
      ),
    ).toEqual({
      providerType: "deepseek",
      model: "deepseek-chat",
    });
  });

  it("无 Provider 时应显示配置引导并支持点击配置", async () => {
    mockUseConfiguredProviders.mockReturnValue({
      providers: [],
      loading: false,
    });
    mockLoadConfiguredProviders.mockResolvedValue([]);
    mockUseProviderModels.mockReturnValue({
      modelIds: [],
      models: [],
      loading: false,
      error: null,
    });

    const onManageProviders = vi.fn();
    const { container } = mount("ws-no-provider-guide", { onManageProviders });

    await flushEffects();

    expect(container.textContent).toContain("工具模型未配置");

    const configButton = findButtonByText("配置");
    await act(async () => {
      configButton.click();
    });

    expect(onManageProviders).toHaveBeenCalledTimes(1);
  });

  it("无 Provider 时应支持关闭配置引导", async () => {
    mockUseConfiguredProviders.mockReturnValue({
      providers: [],
      loading: false,
    });
    mockLoadConfiguredProviders.mockResolvedValue([]);
    mockUseProviderModels.mockReturnValue({
      modelIds: [],
      models: [],
      loading: false,
      error: null,
    });

    const { container } = mount("ws-no-provider-guide-dismiss");

    await flushEffects();

    expect(container.textContent).toContain("工具模型未配置");

    const dismissButton = container.querySelector(
      'button[aria-label="关闭工具模型未配置提示"]',
    ) as HTMLButtonElement | null;
    if (!dismissButton) {
      throw new Error("未找到关闭引导按钮");
    }

    await act(async () => {
      dismissButton.click();
    });

    expect(container.textContent ?? "").not.toContain("工具模型未配置");
  });

  it("关闭后台预加载时，应在打开选择器后再加载 Provider 和模型", async () => {
    const workspaceId = "ws-model-selector-lazy-provider-load";
    localStorage.setItem(`agent_pref_provider_${workspaceId}`, "gemini");
    localStorage.setItem(`agent_pref_model_${workspaceId}`, "gemini-2.5-pro");
    localStorage.setItem(`agent_pref_migrated_${workspaceId}`, "true");

    const { container } = mount(workspaceId, {
      chatModelSelectorProps: {
        backgroundPreload: "disabled",
      },
    });

    await flushEffects();

    expect(
      mockUseConfiguredProviders.mock.calls.some(
        ([options]) => options?.autoLoad === false,
      ),
    ).toBe(true);
    expect(
      mockUseConfiguredProviders.mock.calls.some(
        ([options]) => options?.autoLoad === true,
      ),
    ).toBe(false);

    expect(
      mockUseProviderModels.mock.calls.some(
        ([, options]) => options?.autoLoad === false,
      ),
    ).toBe(true);
    expect(
      mockUseProviderModels.mock.calls.some(
        ([, options]) => options?.autoLoad === true,
      ),
    ).toBe(false);

    await act(async () => {
      getComboboxTrigger(container).click();
    });
    await flushEffects();

    expect(
      mockUseConfiguredProviders.mock.calls.some(
        ([options]) => options?.autoLoad === true,
      ),
    ).toBe(true);
    expect(
      mockUseProviderModels.mock.calls.some(
        ([, options]) => options?.autoLoad === true,
      ),
    ).toBe(true);
  });
});
