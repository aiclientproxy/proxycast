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
const mockExecutePrepareRequests = vi.hoisted(() => vi.fn());
const mockExecuteCallProofRequests = vi.hoisted(() => vi.fn());

vi.mock("@/lib/api/agentRuntime/inventoryClient", () => ({
  getAgentRuntimeToolInventory: mockGetAgentRuntimeToolInventory,
}));

vi.mock("@/lib/api/mcp", () => ({
  mcpApi: {
    executeCallProofRequests: mockExecuteCallProofRequests,
    executePrepareRequests: mockExecutePrepareRequests,
  },
}));

interface HookProps {
  enabled: boolean;
  chatMode: "agent" | "general" | "workbench";
  mappedTheme: string;
  harnessPanelVisible: boolean;
  harnessRequestMetadata: Record<string, unknown>;
  isThemeWorkbench: boolean;
  themeWorkbenchRunState: "idle" | "auto_running" | "await_user_decision";
  currentGate: {
    title: string;
    description: string;
  };
  themeWorkbenchBackendRunState: Parameters<
    typeof useWorkspaceHarnessInventoryRuntime
  >[0]["themeWorkbenchBackendRunState"];
  themeWorkbenchActiveQueueItem: Parameters<
    typeof useWorkspaceHarnessInventoryRuntime
  >[0]["themeWorkbenchActiveQueueItem"];
  harnessPendingCount: number;
  threadId?: string | null;
}

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
    threadId: "thread-1",
    ...initialProps,
  };

  function TestComponent() {
    hookValue = useWorkspaceHarnessInventoryRuntime(currentProps);
    return null;
  }

  const render = (nextProps?: Partial<HookProps>) => {
    currentProps = {
      ...currentProps,
      ...nextProps,
    };
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

async function flushHookEffects() {
  await act(async () => {
    await Promise.resolve();
    await new Promise((resolve) => window.setTimeout(resolve, 0));
  });
}

describe("useWorkspaceHarnessInventoryRuntime", () => {
  beforeEach(() => {
    (
      globalThis as typeof globalThis & {
        IS_REACT_ACT_ENVIRONMENT?: boolean;
      }
    ).IS_REACT_ACT_ENVIRONMENT = true;
    mockGetAgentRuntimeToolInventory.mockReset();
    mockExecutePrepareRequests.mockReset();
    mockGetAgentRuntimeToolInventory.mockResolvedValue({
      sections: [],
      toolCount: 0,
    });
    mockExecutePrepareRequests.mockResolvedValue([]);
    mockExecuteCallProofRequests.mockResolvedValue([]);
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
    mockGetAgentRuntimeToolInventory.mockResolvedValueOnce({
      agent_initialized: true,
      runtime_tools: [
        { name: "WebSearch" },
        { name: "Agent" },
        { name: "SendMessage" },
        { name: "TeamCreate" },
        { name: "TeamDelete" },
        { name: "ListPeers" },
        { name: "update_plan" },
      ],
      native_tools: [],
    });

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
    mockGetAgentRuntimeToolInventory.mockResolvedValueOnce({
      agent_initialized: true,
      runtime_tools: [{ name: "WebSearch" }],
      native_tools: [],
    });

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

  it("应执行插件 MCP prepare requests 并刷新工具库存", async () => {
    const prepareRequests = [
      {
        method: "mcpServer/start",
        params: { name: "context7" },
        reason: "server_stopped",
        status: "candidate",
      },
      {
        method: "mcpTool/listForContext",
        params: { caller: "plugin:docs-plugin", includeDeferred: true },
        reason: "tool_listing",
        status: "candidate",
      },
    ];
    mockGetAgentRuntimeToolInventory
      .mockResolvedValueOnce({
        agent_initialized: true,
        plugin_mcp_targets: [
          {
            pluginId: "docs-plugin",
            expectedToolName: "mcp__context7__resolve-library-id",
            prepareRequests,
          },
        ],
      })
      .mockResolvedValueOnce({
        agent_initialized: true,
        plugin_mcp_targets: [
          {
            pluginId: "docs-plugin",
            prepareRequests: [],
          },
        ],
      });

    const harness = mountHook({
      enabled: true,
      harnessPanelVisible: true,
    });
    mockExecutePrepareRequests.mockResolvedValueOnce([
      {
        method: "mcpServer/start",
        status: "completed",
      },
      {
        method: "mcpTool/listForContext",
        status: "completed",
        toolCount: 1,
        tools: [
          {
            name: "mcp__context7__resolve-library-id",
            server_name: "context7",
            description: "resolve",
            input_schema: {},
          },
        ],
      },
    ]);

    try {
      await flushHookEffects();

      expect(harness.getValue().mcpPrepareCandidateCount).toBe(2);
      expect(mockExecutePrepareRequests).not.toHaveBeenCalled();

      await act(async () => {
        await harness.getValue().prepareMcpTargets();
      });

      expect(mockExecutePrepareRequests).toHaveBeenCalledTimes(1);
      expect(mockExecutePrepareRequests).toHaveBeenCalledWith(prepareRequests);
      expect(mockExecuteCallProofRequests).not.toHaveBeenCalled();
      expect(mockGetAgentRuntimeToolInventory).toHaveBeenCalledTimes(2);
      expect(harness.getValue().mcpPrepareError).toBeNull();
    } finally {
      harness.unmount();
    }
  });

  it("自动 list proof 未暴露目标工具时不刷新库存", async () => {
    mockGetAgentRuntimeToolInventory.mockResolvedValueOnce({
      agent_initialized: true,
      plugin_mcp_targets: [
        {
          pluginId: "docs-plugin",
          expectedToolName: "mcp__context7__resolve-library-id",
          prepareRequests: [
            {
              method: "mcpTool/listForContext",
              params: { caller: "plugin:docs-plugin", includeDeferred: true },
              reason: "tool_listing",
              status: "candidate",
            },
          ],
        },
      ],
    });
    mockExecutePrepareRequests.mockResolvedValueOnce([
      {
        method: "mcpTool/listForContext",
        status: "completed",
        toolCount: 1,
        tools: [
          {
            name: "mcp__context7__search",
            server_name: "context7",
            description: "search",
            input_schema: {},
          },
        ],
      },
    ]);

    const harness = mountHook({
      enabled: true,
      harnessPanelVisible: true,
    });

    try {
      await flushHookEffects();
      await flushHookEffects();

      expect(mockExecutePrepareRequests).toHaveBeenCalledTimes(1);
      expect(mockExecuteCallProofRequests).not.toHaveBeenCalled();
      expect(mockGetAgentRuntimeToolInventory).toHaveBeenCalledTimes(1);
      expect(harness.getValue().mcpPrepareError).toBe("准备 MCP 工具失败");
    } finally {
      harness.unmount();
    }
  });

  it("MCP prepare 后应执行显式 call proof request 再刷新库存", async () => {
    const prepareRequests = [
      {
        method: "mcpTool/listForContext",
        params: { caller: "plugin:docs-plugin", includeDeferred: true },
        reason: "tool_listing",
        status: "candidate",
      },
    ];
    const callProofRequest = {
      method: "mcpServer/tool/call",
      params: {
        server: "context7",
        tool: "resolve-library-id",
        arguments: { libraryName: "react" },
      },
      reason: "tool_call_proof",
      status: "candidate",
    };
    mockGetAgentRuntimeToolInventory
      .mockResolvedValueOnce({
        agent_initialized: true,
        plugin_mcp_targets: [
          {
            pluginId: "docs-plugin",
            expectedToolName: "mcp__context7__resolve-library-id",
            callProofRequest,
            prepareRequests,
          },
        ],
      })
      .mockResolvedValueOnce({
        agent_initialized: true,
        plugin_mcp_targets: [
          {
            pluginId: "docs-plugin",
            prepareRequests: [],
          },
        ],
      });
    mockExecutePrepareRequests.mockResolvedValueOnce([
      {
        method: "mcpTool/listForContext",
        status: "completed",
        toolCount: 1,
        tools: [
          {
            name: "mcp__context7__resolve-library-id",
            server_name: "context7",
            description: "resolve",
            input_schema: {},
          },
        ],
      },
    ]);
    mockExecuteCallProofRequests.mockResolvedValueOnce([
      {
        method: "mcpServer/tool/call",
        status: "completed",
        result: {
          content: [{ type: "text", text: "ok" }],
          is_error: false,
        },
      },
    ]);

    const harness = mountHook({
      enabled: true,
      harnessPanelVisible: true,
    });

    try {
      await flushHookEffects();

      expect(harness.getValue().mcpPrepareCandidateCount).toBe(2);
      expect(mockExecutePrepareRequests).not.toHaveBeenCalled();
      expect(mockExecuteCallProofRequests).not.toHaveBeenCalled();

      await act(async () => {
        await harness.getValue().prepareMcpTargets();
      });

      expect(mockExecutePrepareRequests).toHaveBeenCalledWith(prepareRequests);
      expect(mockExecuteCallProofRequests).toHaveBeenCalledWith(
        [callProofRequest],
        "thread-1",
      );
      expect(mockGetAgentRuntimeToolInventory).toHaveBeenCalledTimes(2);
      expect(harness.getValue().mcpPrepareError).toBeNull();
    } finally {
      harness.unmount();
    }
  });

  it("MCP call proof 失败时不刷新库存", async () => {
    const callProofRequest = {
      method: "mcpServer/tool/call",
      params: {
        server: "context7",
        tool: "resolve-library-id",
        arguments: { libraryName: "react" },
      },
      reason: "tool_call_proof",
      status: "candidate",
    };
    mockGetAgentRuntimeToolInventory.mockResolvedValueOnce({
      agent_initialized: true,
      plugin_mcp_targets: [
        {
          pluginId: "docs-plugin",
          expectedToolName: "mcp__context7__resolve-library-id",
          callProofRequest,
          prepareRequests: [],
        },
      ],
    });
    mockExecuteCallProofRequests.mockRejectedValueOnce(
      new Error("MCP 工具调用证明失败"),
    );

    const harness = mountHook({
      enabled: true,
      harnessPanelVisible: true,
    });

    try {
      await flushHookEffects();

      expect(harness.getValue().mcpPrepareCandidateCount).toBe(1);

      await act(async () => {
        await harness.getValue().prepareMcpTargets();
      });

      expect(mockExecutePrepareRequests).not.toHaveBeenCalled();
      expect(mockExecuteCallProofRequests).toHaveBeenCalledWith(
        [callProofRequest],
        "thread-1",
      );
      expect(mockGetAgentRuntimeToolInventory).toHaveBeenCalledTimes(1);
      expect(harness.getValue().mcpPrepareError).toBe("MCP 工具调用证明失败");
    } finally {
      harness.unmount();
    }
  });

  it("缺少显式 call proof 时应自动使用 toolListRequest 做默认可见性证明", async () => {
    const defaultProofRequest = {
      method: "mcpTool/listForContext",
      params: { caller: "plugin:docs-plugin", includeDeferred: true },
      reason: "tool_listing_default_proof",
      status: "candidate",
    };
    mockGetAgentRuntimeToolInventory
      .mockResolvedValueOnce({
        agent_initialized: true,
        plugin_mcp_targets: [
          {
            pluginId: "docs-plugin",
            expectedToolName: "mcp__context7__resolve-library-id",
            prepareRequests: [],
            toolListRequest: {
              caller: "plugin:docs-plugin",
              includeDeferred: true,
            },
          },
        ],
      })
      .mockResolvedValueOnce({
        agent_initialized: true,
        plugin_mcp_targets: [
          {
            pluginId: "docs-plugin",
            prepareRequests: [],
          },
        ],
      });
    mockExecutePrepareRequests.mockResolvedValueOnce([
      {
        method: "mcpTool/listForContext",
        status: "completed",
        toolCount: 1,
        tools: [
          {
            name: "mcp__context7__resolve-library-id",
            server_name: "context7",
            description: "resolve",
            input_schema: {},
          },
        ],
      },
    ]);

    const harness = mountHook({
      enabled: true,
      harnessPanelVisible: true,
    });

    try {
      await flushHookEffects();
      await flushHookEffects();

      expect(mockExecutePrepareRequests).toHaveBeenCalledWith([
        defaultProofRequest,
      ]);
      expect(mockExecuteCallProofRequests).not.toHaveBeenCalled();
      expect(mockGetAgentRuntimeToolInventory).toHaveBeenCalledTimes(2);
      expect(harness.getValue().mcpPrepareError).toBeNull();
    } finally {
      harness.unmount();
    }
  });

  it("自动默认 list proof 缺少目标工具时不刷新库存", async () => {
    mockGetAgentRuntimeToolInventory.mockResolvedValueOnce({
      agent_initialized: true,
      plugin_mcp_targets: [
        {
          pluginId: "docs-plugin",
          expectedToolName: "mcp__context7__resolve-library-id",
          prepareRequests: [],
          toolListRequest: {
            caller: "plugin:docs-plugin",
            includeDeferred: true,
          },
        },
      ],
    });
    mockExecutePrepareRequests.mockResolvedValueOnce([
      {
        method: "mcpTool/listForContext",
        status: "completed",
        toolCount: 1,
        tools: [
          {
            name: "mcp__context7__search",
            server_name: "context7",
            description: "search",
            input_schema: {},
          },
        ],
      },
    ]);

    const harness = mountHook({
      enabled: true,
      harnessPanelVisible: true,
    });

    try {
      await flushHookEffects();
      await flushHookEffects();

      expect(mockExecutePrepareRequests).toHaveBeenCalledTimes(1);
      expect(mockExecuteCallProofRequests).not.toHaveBeenCalled();
      expect(mockGetAgentRuntimeToolInventory).toHaveBeenCalledTimes(1);
      expect(harness.getValue().mcpPrepareError).toBe("准备 MCP 工具失败");
    } finally {
      harness.unmount();
    }
  });

  it("没有 candidate prepare request 时不应调用 MCP API", async () => {
    mockGetAgentRuntimeToolInventory.mockResolvedValueOnce({
      agent_initialized: true,
      plugin_mcp_targets: [
        {
          pluginId: "docs-plugin",
          prepareRequests: [
            {
              method: "mcpTool/listForContext",
              params: { caller: "plugin:docs-plugin" },
              status: "ready",
            },
          ],
        },
      ],
    });

    const harness = mountHook({
      enabled: true,
      harnessPanelVisible: true,
    });

    try {
      await flushHookEffects();

      expect(harness.getValue().mcpPrepareCandidateCount).toBe(0);

      await act(async () => {
        await harness.getValue().prepareMcpTargets();
      });

      expect(mockExecutePrepareRequests).not.toHaveBeenCalled();
      expect(mockExecuteCallProofRequests).not.toHaveBeenCalled();
      expect(mockGetAgentRuntimeToolInventory).toHaveBeenCalledTimes(1);
    } finally {
      harness.unmount();
    }
  });

  it("开关关闭时不应继续读取工具库存，也不应生成摘要", async () => {
    const harness = mountHook({
      enabled: false,
      harnessPanelVisible: true,
    });

    try {
      await act(async () => {
        await Promise.resolve();
      });

      expect(mockGetAgentRuntimeToolInventory).not.toHaveBeenCalled();
      expect(harness.getValue().toolInventory).toBeNull();
      expect(harness.getValue().generalWorkbenchHarnessSummary).toBeNull();
    } finally {
      harness.unmount();
    }
  });
});
