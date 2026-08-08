import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type {
  GeneralWorkbenchRunState,
  GeneralWorkbenchRunTerminalItem,
  GeneralWorkbenchRunTodoItem,
} from "@/lib/api/executionRun";
import { useWorkspaceHarnessInventoryRuntime } from "./useWorkspaceHarnessInventoryRuntime";

const mockGetAgentRuntimeToolInventory = vi.hoisted(() => vi.fn());

vi.mock("@/lib/api/agentRuntime/inventoryClient", () => ({
  getAgentRuntimeToolInventory: mockGetAgentRuntimeToolInventory,
}));

type HookProps = Parameters<typeof useWorkspaceHarnessInventoryRuntime>[0];

interface HookHarness {
  getValue: () => ReturnType<typeof useWorkspaceHarnessInventoryRuntime>;
  rerender: (props?: Partial<HookProps>) => void;
  unmount: () => void;
}

function mountHook(initialProps?: Partial<HookProps>): HookHarness {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  let hookValue: ReturnType<typeof useWorkspaceHarnessInventoryRuntime> | null =
    null;
  let currentProps: HookProps = {
    enabled: true,
    chatMode: "agent",
    mappedTheme: "general",
    harnessPanelVisible: false,
    harnessRequestMetadata: {},
    isThemeWorkbench: true,
    themeWorkbenchRunState: "auto_running",
    currentGate: {
      title: "写作闸门",
      description: "生成首版草稿",
    },
    themeWorkbenchBackendRunState: null,
    themeWorkbenchActiveQueueItem: null,
    harnessPendingCount: 0,
    ...initialProps,
  };

  function TestComponent() {
    hookValue = useWorkspaceHarnessInventoryRuntime(currentProps);
    return null;
  }

  const render = (nextProps?: Partial<HookProps>) => {
    currentProps = { ...currentProps, ...nextProps };
    act(() => {
      root.render(<TestComponent />);
    });
  };

  render();

  return {
    getValue: () => {
      if (!hookValue) {
        throw new Error("hook 尚未初始化");
      }
      return hookValue;
    },
    rerender: render,
    unmount: () => {
      act(() => {
        root.unmount();
      });
      container.remove();
    },
  };
}

describe("useWorkspaceHarnessInventoryRuntime", () => {
  beforeEach(() => {
    (
      globalThis as typeof globalThis & {
        IS_REACT_ACT_ENVIRONMENT?: boolean;
      }
    ).IS_REACT_ACT_ENVIRONMENT = true;
    mockGetAgentRuntimeToolInventory.mockReset();
    mockGetAgentRuntimeToolInventory.mockResolvedValue({
      agent_initialized: true,
      runtime_tools: [],
      native_tools: [],
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("应通过 artifact protocol 统计活动队列项中的产物数量", () => {
    const activeQueueItem = {
      run_id: "run-1",
      title: "生成首版草稿",
      gate_key: "write_mode",
      status: "running",
      source: "skill",
      source_ref: null,
      started_at: "2026-03-24T14:00:00.000Z",
      filePath: "content-posts/demo.md",
      artifactPath: "content-posts\\demo-cover.png",
    } as unknown as GeneralWorkbenchRunTodoItem;

    const harness = mountHook({
      themeWorkbenchActiveQueueItem: activeQueueItem,
      harnessPendingCount: 2,
    });

    try {
      expect(harness.getValue().generalWorkbenchHarnessSummary).toMatchObject({
        runState: "auto_running",
        runTitle: "生成首版草稿",
        artifactCount: 2,
        pendingCount: 2,
      });
    } finally {
      harness.unmount();
    }
  });

  it("活动队列缺少产物路径时应回退到最新终态记录", () => {
    const latestTerminal = {
      run_id: "run-terminal",
      title: "排版完成",
      gate_key: "write_mode",
      status: "success",
      source: "skill",
      source_ref: null,
      started_at: "2026-03-24T14:00:00.000Z",
      finished_at: "2026-03-24T14:00:08.000Z",
      outputPath: "content-posts/final.md",
    } as unknown as GeneralWorkbenchRunTerminalItem;
    const backendRunState = {
      run_state: "auto_running",
      current_gate_key: "write_mode",
      queue_items: [],
      latest_terminal: latestTerminal,
      updated_at: "2026-03-24T14:00:08.000Z",
    } as GeneralWorkbenchRunState;

    const harness = mountHook({
      themeWorkbenchActiveQueueItem: {
        run_id: "run-queue",
        title: "排版中",
        gate_key: "write_mode",
        status: "running",
        source: "skill",
        source_ref: null,
        started_at: "2026-03-24T14:00:00.000Z",
      },
      themeWorkbenchBackendRunState: backendRunState,
    });

    try {
      expect(
        harness.getValue().generalWorkbenchHarnessSummary?.artifactCount,
      ).toBe(1);
    } finally {
      harness.unmount();
    }
  });

  it("详情面板未展开时不应预取工具库存", async () => {
    const harness = mountHook({
      enabled: true,
      harnessPanelVisible: false,
    });

    try {
      await act(async () => {
        await Promise.resolve();
      });

      expect(mockGetAgentRuntimeToolInventory).not.toHaveBeenCalled();
      expect(harness.getValue().toolInventory).toBeNull();
    } finally {
      harness.unmount();
    }
  });

  it("详情面板展开时应读取工具库存", async () => {
    const harness = mountHook({
      enabled: true,
      harnessPanelVisible: true,
    });

    try {
      await act(async () => {
        await Promise.resolve();
      });

      expect(mockGetAgentRuntimeToolInventory).toHaveBeenCalledTimes(1);
      expect(harness.getValue().toolInventory).toMatchObject({
        agent_initialized: true,
      });
    } finally {
      harness.unmount();
    }
  });
});
