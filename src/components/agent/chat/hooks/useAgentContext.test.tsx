import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const {
  mockNotifyProjectRuntimeAgentsGuide,
  mockSetSessionExecutionStrategy,
  mockSetSessionProviderSelection,
  mockTopicsUpdater,
  mockToastError,
  mockUpdateProject,
} = vi.hoisted(() => ({
  mockNotifyProjectRuntimeAgentsGuide: vi.fn(),
  mockSetSessionExecutionStrategy: vi.fn(async () => undefined),
  mockSetSessionProviderSelection: vi.fn(async () => undefined),
  mockTopicsUpdater: vi.fn(),
  mockToastError: vi.fn(),
  mockUpdateProject: vi.fn(async () => undefined),
}));
const LEGACY_DESKTOP_HOST_INTERNALS_KEY = ["__TA", "URI_INTERNALS__"].join("");

vi.mock("sonner", () => ({
  toast: {
    error: mockToastError,
  },
}));

vi.mock("@/lib/api/project", () => ({
  updateProject: mockUpdateProject,
}));

vi.mock("@/components/workspace/services/runtimeAgentsGuideService", () => ({
  notifyProjectRuntimeAgentsGuide: mockNotifyProjectRuntimeAgentsGuide,
}));

import { useAgentContext } from "./useAgentContext";
import { loadPersistedSessionWorkspaceId } from "./agentProjectStorage";

interface HookHarness {
  getValue: () => ReturnType<typeof useAgentContext>;
  unmount: () => void;
  sendMessage: ReturnType<typeof vi.fn>;
  pendingSessionMetadataSyncCancelRef: {
    current: (() => void) | null;
  };
}

function mountHook(
  workspaceId = "workspace-1",
  sessionId: string | null = null,
  pendingSessionMetadataSyncCancelRef = {
    current: null as (() => void) | null,
  },
): HookHarness {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  const sendMessage = vi.fn(async () => undefined);
  let hookValue: ReturnType<typeof useAgentContext> | null = null;

  function TestComponent() {
    hookValue = useAgentContext({
      workspaceId,
      sessionIdRef: { current: sessionId },
      pendingSessionMetadataSyncCancelRef,
      topicsUpdaterRef: { current: mockTopicsUpdater },
      sendMessageRef: { current: sendMessage },
      runtime: {
        setSessionExecutionStrategy: mockSetSessionExecutionStrategy,
        setSessionProviderSelection: mockSetSessionProviderSelection,
      },
    });
    return null;
  }

  act(() => {
    root.render(<TestComponent />);
  });

  return {
    getValue: () => {
      if (!hookValue) {
        throw new Error("hook 尚未初始化");
      }
      return hookValue;
    },
    unmount: () => {
      act(() => {
        root.unmount();
      });
      container.remove();
    },
    sendMessage,
    pendingSessionMetadataSyncCancelRef,
  };
}

function mountRerenderableHook(
  initialWorkspaceId = "",
  sessionId: string | null = null,
): HookHarness & { rerender: (workspaceId: string) => void } {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  const sendMessage = vi.fn(async () => undefined);
  const pendingSessionMetadataSyncCancelRef = {
    current: null as (() => void) | null,
  };
  let hookValue: ReturnType<typeof useAgentContext> | null = null;

  function TestComponent({ workspaceId }: { workspaceId: string }) {
    hookValue = useAgentContext({
      workspaceId,
      sessionIdRef: { current: sessionId },
      pendingSessionMetadataSyncCancelRef,
      topicsUpdaterRef: { current: mockTopicsUpdater },
      sendMessageRef: { current: sendMessage },
      runtime: {
        setSessionExecutionStrategy: mockSetSessionExecutionStrategy,
        setSessionProviderSelection: mockSetSessionProviderSelection,
      },
    });
    return null;
  }

  act(() => {
    root.render(<TestComponent workspaceId={initialWorkspaceId} />);
  });

  return {
    getValue: () => {
      if (!hookValue) {
        throw new Error("hook 尚未初始化");
      }
      return hookValue;
    },
    rerender: (workspaceId: string) => {
      act(() => {
        root.render(<TestComponent workspaceId={workspaceId} />);
      });
    },
    unmount: () => {
      act(() => {
        root.unmount();
      });
      container.remove();
    },
    sendMessage,
    pendingSessionMetadataSyncCancelRef,
  };
}

describe("useAgentContext", () => {
  beforeEach(() => {
    (
      globalThis as typeof globalThis & {
        IS_REACT_ACT_ENVIRONMENT?: boolean;
      }
    ).IS_REACT_ACT_ENVIRONMENT = true;
    (
      window as unknown as Window & {
        [LEGACY_DESKTOP_HOST_INTERNALS_KEY]?: {
          invoke?: () => Promise<void>;
        };
      }
    )[LEGACY_DESKTOP_HOST_INTERNALS_KEY] = {
      invoke: async () => undefined,
    };
    mockNotifyProjectRuntimeAgentsGuide.mockReset();
    mockSetSessionExecutionStrategy.mockClear();
    mockSetSessionProviderSelection.mockClear();
    mockTopicsUpdater.mockReset();
    mockToastError.mockReset();
    mockUpdateProject.mockReset();
    localStorage.clear();
    sessionStorage.clear();
  });

  afterEach(() => {
    delete (
      window as unknown as Window & {
        [LEGACY_DESKTOP_HOST_INTERNALS_KEY]?: unknown;
      }
    )[LEGACY_DESKTOP_HOST_INTERNALS_KEY];
    document.body.innerHTML = "";
  });

  it("未命中持久化配置时应默认使用完全访问", async () => {
    const harness = mountHook("workspace-default-access");

    await act(async () => {
      await Promise.resolve();
    });

    expect(harness.getValue().accessMode).toBe("full-access");
    expect(
      JSON.parse(
        localStorage.getItem("agent_access_mode_workspace-default-access") ||
          "null",
      ),
    ).toBe("full-access");

    harness.unmount();
  });

  it("workspace 从空值解析完成的同一帧应投影默认普通 Agent 主链策略", async () => {
    const harness = mountRerenderableHook("");

    expect(harness.getValue().executionStrategy).toBe("react");

    harness.rerender("workspace-code-runtime");

    expect(harness.getValue().executionStrategy).toBe("react");

    await act(async () => {
      await Promise.resolve();
    });

    expect(harness.getValue().executionStrategy).toBe("react");
    expect(
      JSON.parse(
        localStorage.getItem(
          "agent_execution_strategy_workspace-code-runtime",
        ) || "null",
      ),
    ).toBe("react");

    harness.unmount();
  });

  it("切换 provider 和 model 时不应写入微信运行时配置", async () => {
    const harness = mountHook();

    await act(async () => {
      harness.getValue().setProviderType("deepseek");
      harness.getValue().setModel("deepseek-reasoner");
      await Promise.resolve();
    });

    expect(localStorage.getItem("agent_pref_provider_workspace-1")).toBe(
      JSON.stringify("deepseek"),
    );
    expect(localStorage.getItem("agent_pref_model_workspace-1")).toBe(
      JSON.stringify("deepseek-reasoner"),
    );

    harness.unmount();
  });

  it("当前会话切换 provider/model/effort 时应原子回写并在成功后提交 UI", async () => {
    const harness = mountHook("workspace-1", "session-1");

    await act(async () => {
      harness.getValue().setProviderType("deepseek");
      harness.getValue().setModel("deepseek-reasoner");
      harness.getValue().setReasoningEffort("xhigh");
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockSetSessionProviderSelection).toHaveBeenCalledTimes(1);
    expect(mockSetSessionProviderSelection).toHaveBeenCalledWith(
      "session-1",
      "deepseek",
      "deepseek-reasoner",
      "xhigh",
    );
    expect(harness.getValue().providerType).toBe("deepseek");
    expect(harness.getValue().model).toBe("deepseek-reasoner");
    expect(harness.getValue().reasoningEffort).toBe("xhigh");
    expect(
      harness.getValue().getSyncedSessionModelPreference("session-1"),
    ).toEqual({
      providerType: "deepseek",
      model: "deepseek-reasoner",
    });

    harness.unmount();
  });

  it("发送前等待应覆盖目标会话尚未完成的 provider selection 写入", async () => {
    let releaseSelectionSync: (() => void) | undefined;
    mockSetSessionProviderSelection.mockImplementationOnce(
      () =>
        new Promise<undefined>((resolve) => {
          releaseSelectionSync = () => resolve(undefined);
        }),
    );
    const harness = mountHook("workspace-1", "session-target");

    act(() => {
      harness.getValue().setProviderType("provider-enabled");
      harness.getValue().setModel("shared-model");
    });
    await act(async () => {
      await Promise.resolve();
    });

    let waitCompleted = false;
    const wait = harness
      .getValue()
      .waitForSessionProviderSelectionSync("session-target")
      .then(() => {
        waitCompleted = true;
      });
    await Promise.resolve();

    expect(mockSetSessionProviderSelection).toHaveBeenCalledWith(
      "session-target",
      "provider-enabled",
      "shared-model",
      "",
    );
    expect(waitCompleted).toBe(false);

    await act(async () => {
      releaseSelectionSync?.();
      await wait;
    });

    expect(waitCompleted).toBe(true);
    expect(
      harness.getValue().getSyncedSessionModelPreference("session-target"),
    ).toEqual({
      providerType: "provider-enabled",
      model: "shared-model",
    });

    harness.unmount();
  });

  it("用户切换 provider/model 时应取消 hydration metadata 回填", async () => {
    const cancelPendingMetadataSync = vi.fn();
    const harness = mountHook("workspace-1", "session-target", {
      current: cancelPendingMetadataSync,
    });

    act(() => {
      harness.getValue().setProviderType("provider-enabled");
      harness.getValue().setModel("shared-model");
    });
    await act(async () => {
      await Promise.resolve();
    });

    expect(cancelPendingMetadataSync).toHaveBeenCalledTimes(1);
    expect(
      harness.pendingSessionMetadataSyncCancelRef.current,
    ).toBeNull();
    harness.unmount();
  });

  it("thread model settings 更新失败时应保留原 UI 和持久化状态", async () => {
    const error = new Error("provider route unavailable");
    mockSetSessionProviderSelection.mockRejectedValueOnce(error);
    const harness = mountHook("workspace-1", "session-1");

    act(() => {
      harness
        .getValue()
        .applySessionModelPreference(
          "session-1",
          { providerType: "openai", model: "gpt-5.4-mini" },
          { markSynced: true },
        );
    });

    await act(async () => {
      harness.getValue().setProviderType("xai");
      harness.getValue().setModel("grok-4.5");
      harness.getValue().setReasoningEffort("xhigh");
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(harness.getValue().providerType).toBe("openai");
    expect(harness.getValue().model).toBe("gpt-5.4-mini");
    expect(harness.getValue().reasoningEffort).toBe("");
    expect(
      JSON.parse(
        localStorage.getItem("agent_topic_model_pref_workspace-1_session-1") ||
          "null",
      ),
    ).toEqual({ providerType: "openai", model: "gpt-5.4-mini" });
    expect(mockToastError).toHaveBeenCalledWith(
      expect.stringContaining("provider route unavailable"),
    );

    harness.unmount();
  });

  it("当前会话收到 current 执行策略时应批量回写 session 并同步话题快照", async () => {
    const harness = mountHook("workspace-1", "session-1");

    await act(async () => {
      harness.getValue().setExecutionStrategy("react");
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockSetSessionExecutionStrategy).toHaveBeenCalledTimes(1);
    expect(mockSetSessionExecutionStrategy).toHaveBeenCalledWith(
      "session-1",
      "react",
    );
    expect(mockTopicsUpdater).toHaveBeenCalledWith("session-1", "react");
    expect(
      harness.getValue().getSyncedSessionExecutionStrategy("session-1"),
    ).toBe("react");
    expect(
      JSON.parse(
        localStorage.getItem("agent_execution_strategy_workspace-1") || "null",
      ),
    ).toBe("react");

    harness.unmount();
  });

  it("无当前会话时收到 current 执行策略应只把 current 策略写入影子缓存", async () => {
    const harness = mountHook("workspace-shadow");

    await act(async () => {
      harness.getValue().setExecutionStrategy("react");
      await Promise.resolve();
    });

    expect(mockSetSessionExecutionStrategy).not.toHaveBeenCalled();
    expect(
      JSON.parse(
        localStorage.getItem("agent_execution_strategy_workspace-shadow") ||
          "null",
      ),
    ).toBe("react");

    harness.unmount();
  });

  it("过滤会话时应优先使用 runtime workspace_id 并回填影子缓存", () => {
    localStorage.setItem(
      "agent_session_workspace_session-runtime-current",
      JSON.stringify("workspace-stale"),
    );
    localStorage.setItem(
      "agent_session_workspace_session-runtime-other",
      JSON.stringify("workspace-1"),
    );

    const harness = mountHook("workspace-1");

    expect(
      harness
        .getValue()
        .filterSessionsByWorkspace([
          {
            id: "session-runtime-current",
            workspace_id: "workspace-1",
          },
          {
            id: "session-runtime-other",
            workspace_id: "workspace-2",
          },
          {
            id: "session-legacy-without-workspace",
          },
        ])
        .map((session) => session.id),
    ).toEqual(["session-runtime-current"]);
    expect(loadPersistedSessionWorkspaceId("session-runtime-current")).toBe(
      "workspace-1",
    );
    expect(loadPersistedSessionWorkspaceId("session-runtime-other")).toBe(
      "workspace-2",
    );

    harness.unmount();
  });

  it("修复目录并重试时应触发运行时 AGENTS 引导", async () => {
    const harness = mountHook();

    act(() => {
      harness.getValue().setWorkspacePathMissing({
        content: "继续上次对话",
        images: [],
      });
    });

    await act(async () => {
      await harness
        .getValue()
        .fixWorkspacePathAndRetry("/tmp/workspace-linked");
    });

    expect(mockUpdateProject).toHaveBeenCalledWith("workspace-1", {
      rootPath: "/tmp/workspace-linked",
    });
    expect(mockNotifyProjectRuntimeAgentsGuide).toHaveBeenCalledWith(
      {
        id: "workspace-1",
        rootPath: "/tmp/workspace-linked",
      },
      {
        successMessage: "工作区目录已重新关联",
        showSuccessWhenGuideAlreadySeen: false,
      },
    );
    expect(harness.sendMessage).toHaveBeenCalledWith(
      "继续上次对话",
      [],
      false,
      false,
      true,
    );

    harness.unmount();
  });
});
