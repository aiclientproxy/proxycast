import { useEffect } from "react";
import { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  cleanupMountedRoots,
  flushEffects,
  mountHarness,
  setupReactActEnvironment,
  type MountedRoot,
} from "@/components/workspace/hooks/testUtils";
import type { McpServerInfo } from "@/lib/api/mcp";
import { useMcp, type UseMcpReturn } from "./useMcp";

const mcpApiMocks = vi.hoisted(() => ({
  listServersWithStatus: vi.fn(),
  listTools: vi.fn(),
  listPrompts: vi.fn(),
  listResources: vi.fn(),
  startServer: vi.fn(),
  stopServer: vi.fn(),
  loginOAuthServer: vi.fn(),
  callTool: vi.fn(),
  getPrompt: vi.fn(),
  readResource: vi.fn(),
  subscribeResource: vi.fn(),
  unsubscribeResource: vi.fn(),
}));

const bridgeMocks = vi.hoisted(() => ({
  safeListen: vi.fn(),
}));

const appServerEventBusMocks = vi.hoisted(() => ({
  subscribeAppServerNotifications: vi.fn(),
}));

vi.mock("@/lib/api/mcp", async () => {
  const actual =
    await vi.importActual<typeof import("@/lib/api/mcp")>("@/lib/api/mcp");

  return {
    ...actual,
    mcpApi: {
      ...actual.mcpApi,
      listServersWithStatus: (...args: unknown[]) =>
        mcpApiMocks.listServersWithStatus(...args),
      listTools: (...args: unknown[]) => mcpApiMocks.listTools(...args),
      listPrompts: (...args: unknown[]) => mcpApiMocks.listPrompts(...args),
      listResources: (...args: unknown[]) => mcpApiMocks.listResources(...args),
      startServer: (...args: unknown[]) => mcpApiMocks.startServer(...args),
      stopServer: (...args: unknown[]) => mcpApiMocks.stopServer(...args),
      loginOAuthServer: (...args: unknown[]) =>
        mcpApiMocks.loginOAuthServer(...args),
      callTool: (...args: unknown[]) => mcpApiMocks.callTool(...args),
      getPrompt: (...args: unknown[]) => mcpApiMocks.getPrompt(...args),
      readResource: (...args: unknown[]) => mcpApiMocks.readResource(...args),
      subscribeResource: (...args: unknown[]) =>
        mcpApiMocks.subscribeResource(...args),
      unsubscribeResource: (...args: unknown[]) =>
        mcpApiMocks.unsubscribeResource(...args),
    },
  };
});

vi.mock("@/lib/dev-bridge", () => ({
  safeListen: (...args: unknown[]) => bridgeMocks.safeListen(...args),
}));

vi.mock("@/lib/api/appServerEventBus", () => ({
  subscribeAppServerNotifications: (...args: unknown[]) =>
    appServerEventBusMocks.subscribeAppServerNotifications(...args),
}));

interface HarnessProps {
  onReady: (value: UseMcpReturn) => void;
}

function HookHarness({ onReady }: HarnessProps) {
  const value = useMcp();

  useEffect(() => {
    onReady(value);
  }, [onReady, value]);

  return null;
}

const mountedRoots: MountedRoot[] = [];

function createServer(overrides: Partial<McpServerInfo> = {}): McpServerInfo {
  return {
    id: "server-demo",
    name: "docs",
    config: {
      command: "node",
      args: ["mcp-server.js"],
    },
    is_running: false,
    server_info: undefined,
    enabled_lime: true,
    enabled_claude: false,
    enabled_codex: true,
    enabled_gemini: false,
    ...overrides,
  };
}

function createResource(name: string, serverName = "docs") {
  return {
    uri: `file:///${name}.md`,
    name,
    server_name: serverName,
    mime_type: "text/markdown",
  };
}

async function renderHook(onReady: (value: UseMcpReturn) => void) {
  mountHarness(HookHarness, { onReady }, mountedRoots);
  await flushEffects(8);
}

describe("useMcp", () => {
  let latestValue: UseMcpReturn | null = null;
  let consoleErrorSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    setupReactActEnvironment();
    latestValue = null;
    vi.clearAllMocks();
    consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    mcpApiMocks.listServersWithStatus.mockResolvedValue([]);
    mcpApiMocks.listTools.mockResolvedValue([]);
    mcpApiMocks.listPrompts.mockResolvedValue([]);
    mcpApiMocks.listResources.mockResolvedValue([]);
    mcpApiMocks.startServer.mockResolvedValue(undefined);
    mcpApiMocks.stopServer.mockResolvedValue(undefined);
    mcpApiMocks.loginOAuthServer.mockResolvedValue({
      authorizationUrl: "https://auth.example/authorize",
      state: "state-1",
    });
    mcpApiMocks.callTool.mockResolvedValue({ content: [], is_error: false });
    mcpApiMocks.getPrompt.mockResolvedValue({ messages: [] });
    mcpApiMocks.readResource.mockResolvedValue({ uri: "docs://readme" });
    mcpApiMocks.subscribeResource.mockResolvedValue(undefined);
    mcpApiMocks.unsubscribeResource.mockResolvedValue(undefined);
    bridgeMocks.safeListen.mockResolvedValue(() => undefined);
    appServerEventBusMocks.subscribeAppServerNotifications.mockReturnValue(
      () => undefined,
    );
  });

  afterEach(() => {
    cleanupMountedRoots(mountedRoots);
    consoleErrorSpy.mockRestore();
  });

  function getLatestValue(): UseMcpReturn {
    expect(latestValue).not.toBeNull();
    return latestValue as UseMcpReturn;
  }

  it("服务器状态刷新成功后应清除上一次 MCP 获取错误", async () => {
    mcpApiMocks.listServersWithStatus
      .mockRejectedValueOnce(new Error("bridge warming up"))
      .mockResolvedValueOnce([createServer()]);

    await renderHook((value) => {
      latestValue = value;
    });

    expect(getLatestValue().error).toBe("bridge warming up");

    await act(async () => {
      await getLatestValue().refreshServers();
    });
    await flushEffects(4);

    expect(getLatestValue().servers).toHaveLength(1);
    expect(getLatestValue().error).toBeNull();
  });

  it("运行态工具列表尚未 current 化时不应阻断 MCP 服务器配置读取", async () => {
    mcpApiMocks.listServersWithStatus.mockResolvedValueOnce([
      createServer({ is_running: true }),
    ]);
    mcpApiMocks.listTools.mockRejectedValueOnce(
      new Error("mcpTool/list runtime not available"),
    );

    await renderHook((value) => {
      latestValue = value;
    });

    expect(getLatestValue().servers).toHaveLength(1);
    expect(getLatestValue().tools).toEqual([]);
    expect(getLatestValue().loading).toBe(false);
    expect(getLatestValue().error).toBeNull();
  });

  it("Desktop event listener 挂起时也应立即订阅 OAuth typed notification", async () => {
    bridgeMocks.safeListen.mockImplementation(() => new Promise(() => {}));

    await renderHook((value) => {
      latestValue = value;
    });
    await flushEffects(1);

    expect(
      appServerEventBusMocks.subscribeAppServerNotifications,
    ).toHaveBeenCalledTimes(1);
  });

  it("OAuth 完成事件应刷新服务器状态和工具列表", async () => {
    let onNotifications:
      | ((notifications: Record<string, unknown>[]) => void)
      | undefined;
    appServerEventBusMocks.subscribeAppServerNotifications.mockImplementation(
      (subscription: {
        onNotifications: (notifications: Record<string, unknown>[]) => void;
      }) => {
        onNotifications = subscription.onNotifications;
        return () => undefined;
      },
    );
    mcpApiMocks.listServersWithStatus
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([
        createServer({
          name: "remote-docs",
          is_running: false,
          runtime_status: {
            name: "remote-docs",
            transport: "streamable_http",
            enabled: true,
            is_running: false,
            required: false,
            supports_parallel_tool_calls: false,
            startup_timeout: 30,
            tool_timeout: 30,
            disabled_tools: [],
            auth_status: {
              mode: "oauth",
              available: true,
            },
          },
        }),
      ]);

    await renderHook((value) => {
      latestValue = value;
    });
    await flushEffects(4);

    await act(async () => {
      onNotifications?.([
        {
          method: "mcpServer/oauthLogin/completed",
          params: {
            name: "remote-docs",
            threadId: null,
            success: true,
          },
        },
      ]);
      await Promise.resolve();
      await Promise.resolve();
    });
    await flushEffects(4);

    expect(mcpApiMocks.listServersWithStatus).toHaveBeenCalledTimes(2);
    expect(mcpApiMocks.listTools).toHaveBeenCalledTimes(2);
    expect(getLatestValue().oauthCompletion?.serverName).toBe("remote-docs");
    expect(getLatestValue().servers[0]?.runtime_status?.auth_status).toEqual({
      mode: "oauth",
      available: true,
    });
  });

  it("OAuth 失败通知不应触发成功完成态", async () => {
    let onNotifications:
      | ((notifications: Record<string, unknown>[]) => void)
      | undefined;
    appServerEventBusMocks.subscribeAppServerNotifications.mockImplementation(
      (subscription: {
        onNotifications: (notifications: Record<string, unknown>[]) => void;
      }) => {
        onNotifications = subscription.onNotifications;
        return () => undefined;
      },
    );

    await renderHook((value) => {
      latestValue = value;
    });
    await flushEffects(4);

    await act(async () => {
      onNotifications?.([
        {
          method: "mcpServer/oauthLogin/completed",
          params: {
            name: "remote-docs",
            threadId: null,
            success: false,
            error: "scope rejected",
          },
        },
      ]);
      await Promise.resolve();
      await Promise.resolve();
    });
    await flushEffects(4);

    expect(getLatestValue().oauthCompletion).toBeNull();
    expect(getLatestValue().error).toBe("remote-docs: scope rejected");
    expect(
      getLatestValue().serverConnectionStates["remote-docs"],
    ).toMatchObject({
      phase: "idle",
      error: "scope rejected",
    });
    expect(mcpApiMocks.listServersWithStatus).toHaveBeenCalledTimes(2);
    expect(mcpApiMocks.listTools).toHaveBeenCalledTimes(2);
  });

  it("资源列表更新事件应刷新资源列表", async () => {
    const listeners = new Map<string, (event: { payload: unknown }) => void>();
    bridgeMocks.safeListen.mockImplementation(
      async (
        eventName: string,
        handler: (event: { payload: unknown }) => void,
      ) => {
        listeners.set(eventName, handler);
        return () => listeners.delete(eventName);
      },
    );
    mcpApiMocks.listResources
      .mockResolvedValueOnce([createResource("before")])
      .mockResolvedValueOnce([createResource("after")]);

    await renderHook((value) => {
      latestValue = value;
    });
    await flushEffects(4);

    expect(getLatestValue().resources).toEqual([createResource("before")]);

    await act(async () => {
      listeners.get("mcp:resources_updated")?.({
        payload: { server_name: "docs" },
      });
      await Promise.resolve();
      await Promise.resolve();
    });
    await flushEffects(4);

    expect(mcpApiMocks.listResources).toHaveBeenCalledTimes(2);
    expect(getLatestValue().resources).toEqual([createResource("after")]);
  });

  it("单个资源更新事件也应刷新资源列表", async () => {
    const listeners = new Map<string, (event: { payload: unknown }) => void>();
    bridgeMocks.safeListen.mockImplementation(
      async (
        eventName: string,
        handler: (event: { payload: unknown }) => void,
      ) => {
        listeners.set(eventName, handler);
        return () => listeners.delete(eventName);
      },
    );
    mcpApiMocks.listResources
      .mockResolvedValueOnce([createResource("readme")])
      .mockResolvedValueOnce([createResource("readme-updated")]);

    await renderHook((value) => {
      latestValue = value;
    });
    await flushEffects(4);

    await act(async () => {
      listeners.get("mcp:resource_updated")?.({
        payload: { server_name: "docs", uri: "file:///readme.md" },
      });
      await Promise.resolve();
      await Promise.resolve();
    });
    await flushEffects(4);

    expect(mcpApiMocks.listResources).toHaveBeenCalledTimes(2);
    expect(getLatestValue().resources).toEqual([
      createResource("readme-updated"),
    ]);
  });

  it("资源订阅操作应通过 MCP current API fail closed 透传", async () => {
    await renderHook((value) => {
      latestValue = value;
    });

    await act(async () => {
      await getLatestValue().subscribeResource("docs", "file:///readme.md");
      await getLatestValue().unsubscribeResource("docs", "file:///readme.md");
    });

    expect(mcpApiMocks.subscribeResource).toHaveBeenCalledWith(
      "docs",
      "file:///readme.md",
    );
    expect(mcpApiMocks.unsubscribeResource).toHaveBeenCalledWith(
      "docs",
      "file:///readme.md",
    );
  });

  it("提示词操作应透传精确 server 和 name", async () => {
    await renderHook((value) => {
      latestValue = value;
    });

    await act(async () => {
      await getLatestValue().getPrompt("docs", "summarize", { topic: "lime" });
    });

    expect(mcpApiMocks.getPrompt).toHaveBeenCalledWith("docs", "summarize", {
      topic: "lime",
    });
  });
});
