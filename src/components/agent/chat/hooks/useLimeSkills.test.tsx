import React from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useLimeSkills } from "./useLimeSkills";

const mockGetLocal = vi.hoisted(() => vi.fn());
const mockGetAll = vi.hoisted(() => vi.fn());
const mockGetRuntimeCatalog = vi.hoisted(() => vi.fn());
const mockLogAgentDebug = vi.hoisted(() => vi.fn());
const mockSubscribeAppServerNotifications = vi.hoisted(() => vi.fn());

vi.mock("@/lib/api/appServerEventBus", () => ({
  subscribeAppServerNotifications: mockSubscribeAppServerNotifications,
}));

vi.mock("@/lib/api/skills", () => ({
  skillsApi: {
    getLocal: mockGetLocal,
    getAll: mockGetAll,
    getRuntimeCatalog: mockGetRuntimeCatalog,
  },
}));

vi.mock("@/lib/agentDebug", () => ({
  logAgentDebug: mockLogAgentDebug,
}));

interface HookHarness {
  getValue: () => ReturnType<typeof useLimeSkills>;
  unmount: () => void;
}

function mountHook(
  options: Parameters<typeof useLimeSkills>[0] = {
    autoLoad: "deferred",
  },
): HookHarness {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  let hookValue: ReturnType<typeof useLimeSkills> | null = null;

  function TestComponent() {
    hookValue = useLimeSkills(options);
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
  };
}

async function flushEffects() {
  await act(async () => {
    await Promise.resolve();
  });
}

interface RerenderHarness extends HookHarness {
  rerender: () => void;
}

function mountHookWithInlineErrorRerender(): RerenderHarness {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  let hookValue: ReturnType<typeof useLimeSkills> | null = null;
  let rerender: (() => void) | null = null;

  function TestComponent() {
    const [tick, setTick] = React.useState(0);
    rerender = () => setTick((value) => value + 1);
    hookValue = useLimeSkills({
      autoLoad: "immediate",
      logScope: "InlineErrorScope",
      onError: (error) => {
        if (tick < 0) {
          throw error;
        }
      },
    });
    return <span data-tick={tick} />;
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
    rerender: () => {
      if (!rerender) {
        throw new Error("rerender 尚未就绪");
      }
      act(() => {
        rerender?.();
      });
    },
    unmount: () => {
      act(() => {
        root.unmount();
      });
      container.remove();
    },
  };
}

describe("useLimeSkills", () => {
  beforeEach(() => {
    (
      globalThis as typeof globalThis & {
        IS_REACT_ACT_ENVIRONMENT?: boolean;
      }
    ).IS_REACT_ACT_ENVIRONMENT = true;
    vi.clearAllMocks();
    vi.useFakeTimers();
    mockSubscribeAppServerNotifications.mockReturnValue(vi.fn());
    mockGetRuntimeCatalog.mockResolvedValue([
      {
        key: "local-skill",
        name: "本地技能",
        description: "默认加载",
        directory: "local-skill",
        installed: true,
        sourceKind: "builtin",
      },
    ]);
    mockGetAll.mockResolvedValue([
      {
        key: "remote-skill",
        name: "远端技能",
        description: "刷新加载",
        directory: "remote-skill",
        installed: true,
        sourceKind: "builtin",
      },
    ]);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it("deferred 模式应在最小延迟后才加载本地 skills", async () => {
    const harness = mountHook({ autoLoad: "deferred", logScope: "TestScope" });

    try {
      act(() => {
        vi.advanceTimersByTime(1499);
      });
      await flushEffects();

      expect(mockGetRuntimeCatalog).not.toHaveBeenCalled();

      act(() => {
        vi.advanceTimersByTime(1);
      });
      await flushEffects();

      expect(mockGetRuntimeCatalog).toHaveBeenCalledOnce();
      expect(harness.getValue().skills[0]?.key).toBe("local-skill");
      expect(mockLogAgentDebug).toHaveBeenCalledWith(
        "TestScope",
        "loadSkills.start",
        { includeRemote: false },
      );
    } finally {
      harness.unmount();
    }
  });

  it("immediate 模式应立即加载，并支持手动刷新远端", async () => {
    const harness = mountHook({ autoLoad: "immediate" });

    try {
      await flushEffects();

      expect(mockGetRuntimeCatalog).toHaveBeenCalledOnce();

      await act(async () => {
        await harness.getValue().refreshSkills(true);
      });

      expect(mockGetAll).toHaveBeenCalledWith("lime");
      expect(harness.getValue().skills[0]?.key).toBe("remote-skill");
    } finally {
      harness.unmount();
    }
  });

  it("调用方每次渲染传入新的 onError 时，不应重复触发自动加载", async () => {
    const harness = mountHookWithInlineErrorRerender();

    try {
      await flushEffects();
      expect(mockGetRuntimeCatalog).toHaveBeenCalledTimes(1);

      harness.rerender();
      await flushEffects();

      expect(mockGetRuntimeCatalog).toHaveBeenCalledTimes(1);
      expect(harness.getValue().skills[0]?.key).toBe("local-skill");
    } finally {
      harness.unmount();
    }
  });

  it("typed skills/changed 应重新读取本地 catalog，畸形 payload 不触发", async () => {
    let onNotifications:
      | ((notifications: Array<{ method: string; params: unknown }>) => void)
      | undefined;
    const unsubscribe = vi.fn();
    mockSubscribeAppServerNotifications.mockImplementationOnce(
      (subscription) => {
        onNotifications = subscription.onNotifications;
        return unsubscribe;
      },
    );
    const harness = mountHook({ autoLoad: false });

    try {
      expect(mockGetRuntimeCatalog).not.toHaveBeenCalled();
      act(() => {
        onNotifications?.([
          { method: "skills/changed", params: { path: "/private/skill" } },
        ]);
      });
      await flushEffects();
      expect(mockGetRuntimeCatalog).not.toHaveBeenCalled();

      act(() => {
        onNotifications?.([{ method: "skills/changed", params: {} }]);
      });
      await flushEffects();
      expect(mockGetRuntimeCatalog).toHaveBeenCalledOnce();
      expect(mockGetLocal).not.toHaveBeenCalled();
      expect(mockGetAll).not.toHaveBeenCalled();
      expect(mockLogAgentDebug).toHaveBeenCalledWith(
        "useLimeSkills",
        "skillsChanged.received",
        { method: "skills/changed", count: 1 },
        { consoleOnly: true },
      );
    } finally {
      harness.unmount();
    }
    expect(unsubscribe).toHaveBeenCalledOnce();
  });
});
