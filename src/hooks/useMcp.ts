/**
 * MCP 运行时状态管理 Hook
 *
 * 提供 MCP 服务器的运行时状态管理，包括：
 * - 服务器启动/停止
 * - 工具列表
 * - 提示词列表和获取
 * - 资源列表和读取
 * - Desktop Host 事件监听
 *
 * @module hooks/useMcp
 */

import { useState, useEffect, useCallback } from "react";
import type { UnlistenFn } from "@/lib/desktop-host/event";
import {
  mcpApi,
  McpServerInfo,
  McpToolDefinition,
  McpPromptDefinition,
  McpResourceDefinition,
  McpPromptResult,
  McpResourceContent,
  McpServerOAuthLoginOptions,
  McpServerOAuthLoginResponse,
} from "@/lib/api/mcp";
import {
  setupMcpEventListeners,
  type McpOAuthCompletionState,
  type McpEventStreamState,
  type McpEventStreamStateMap,
  type McpServerConnectionState,
} from "./useMcpEvents";

// ============================================================================
// Hook 返回类型
// ============================================================================

export type {
  McpEventStreamState,
  McpEventStreamStateMap,
  McpOAuthCompletionState,
  McpServerConnectionState,
};

export interface UseMcpReturn {
  // 状态
  servers: McpServerInfo[];
  tools: McpToolDefinition[];
  prompts: McpPromptDefinition[];
  resources: McpResourceDefinition[];
  loading: boolean;
  error: string | null;
  serverConnectionStates: Record<string, McpServerConnectionState>;
  oauthCompletion: McpOAuthCompletionState | null;
  eventStreams: McpEventStreamStateMap;

  // 服务器操作
  startServer: (name: string) => Promise<void>;
  stopServer: (name: string) => Promise<void>;
  reconnectServer: (name: string) => Promise<void>;
  loginOAuthServer: (
    name: string,
    options?: McpServerOAuthLoginOptions,
  ) => Promise<McpServerOAuthLoginResponse>;
  refreshServers: () => Promise<void>;

  // 工具操作。Settings 只浏览工具；执行必须由真实 Agent Thread 发起。
  refreshTools: () => Promise<void>;

  // 提示词操作
  refreshPrompts: () => Promise<void>;
  getPrompt: (
    server: string,
    name: string,
    args: Record<string, unknown>,
  ) => Promise<McpPromptResult>;

  // 资源操作
  refreshResources: () => Promise<void>;
  readResource: (server: string, uri: string) => Promise<McpResourceContent>;
  subscribeResource: (server: string, uri: string) => Promise<void>;
  unsubscribeResource: (server: string, uri: string) => Promise<void>;
}

// ============================================================================
// Hook 实现
// ============================================================================

export function useMcp(): UseMcpReturn {
  // 状态
  const [servers, setServers] = useState<McpServerInfo[]>([]);
  const [tools, setTools] = useState<McpToolDefinition[]>([]);
  const [prompts, setPrompts] = useState<McpPromptDefinition[]>([]);
  const [resources, setResources] = useState<McpResourceDefinition[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [oauthCompletion, setOAuthCompletion] =
    useState<McpOAuthCompletionState | null>(null);
  const [serverConnectionStates, setServerConnectionStates] = useState<
    Record<string, McpServerConnectionState>
  >({});
  const [eventStreams, setEventStreams] = useState<McpEventStreamStateMap>({});

  const updateServerConnectionState = useCallback(
    (
      serverName: string,
      nextState: Partial<McpServerConnectionState> & {
        phase: McpServerConnectionState["phase"];
      },
    ) => {
      setServerConnectionStates((prev) => ({
        ...prev,
        [serverName]: {
          phase: nextState.phase,
          error: nextState.error ?? null,
          updatedAt: Date.now(),
        },
      }));
    },
    [],
  );

  // --------------------------------------------------------------------------
  // 数据获取方法
  // --------------------------------------------------------------------------

  const refreshServers = useCallback(async () => {
    try {
      const list = await mcpApi.listServersWithStatus();
      setServers(list);
      setError(null);
    } catch (e) {
      console.error("[useMcp] 获取服务器列表失败:", e);
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const refreshTools = useCallback(async () => {
    try {
      const list = await mcpApi.listTools();
      setTools(list);
    } catch (e) {
      console.error("[useMcp] 获取工具列表失败:", e);
      // 工具列表获取失败不设置全局错误
    }
  }, []);

  const refreshPrompts = useCallback(async () => {
    try {
      const list = await mcpApi.listPrompts();
      setPrompts(list);
    } catch (e) {
      console.error("[useMcp] 获取提示词列表失败:", e);
    }
  }, []);

  const refreshResources = useCallback(async () => {
    try {
      const list = await mcpApi.listResources();
      setResources(list);
    } catch (e) {
      console.error("[useMcp] 获取资源列表失败:", e);
    }
  }, []);

  // --------------------------------------------------------------------------
  // 服务器操作
  // --------------------------------------------------------------------------

  const startServer = useCallback(
    async (name: string) => {
      try {
        setError(null);
        updateServerConnectionState(name, {
          phase: "starting",
        });
        await mcpApi.startServer(name);
        // 启动后刷新服务器列表和工具列表
        await refreshServers();
        await refreshTools();
        updateServerConnectionState(name, {
          phase: "idle",
        });
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
        updateServerConnectionState(name, {
          phase: "idle",
          error: msg,
        });
        throw e;
      }
    },
    [refreshServers, refreshTools, updateServerConnectionState],
  );

  const stopServer = useCallback(
    async (name: string) => {
      try {
        setError(null);
        updateServerConnectionState(name, {
          phase: "stopping",
        });
        await mcpApi.stopServer(name);
        // 停止后刷新服务器列表和工具列表
        await refreshServers();
        await refreshTools();
        updateServerConnectionState(name, {
          phase: "idle",
        });
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
        updateServerConnectionState(name, {
          phase: "idle",
          error: msg,
        });
        throw e;
      }
    },
    [refreshServers, refreshTools, updateServerConnectionState],
  );

  const reconnectServer = useCallback(
    async (name: string) => {
      updateServerConnectionState(name, {
        phase: "reconnecting",
      });
      try {
        const target = servers.find((server) => server.name === name);
        if (target?.is_running) {
          await mcpApi.stopServer(name);
        }
        await mcpApi.startServer(name);
        await refreshServers();
        await refreshTools();
        updateServerConnectionState(name, {
          phase: "idle",
        });
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
        updateServerConnectionState(name, {
          phase: "idle",
          error: msg,
        });
        throw e;
      }
    },
    [refreshServers, refreshTools, servers, updateServerConnectionState],
  );

  const loginOAuthServer = useCallback(
    async (
      name: string,
      options: McpServerOAuthLoginOptions = {},
    ): Promise<McpServerOAuthLoginResponse> => {
      try {
        setError(null);
        return await mcpApi.loginOAuthServer(name, options);
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
        updateServerConnectionState(name, {
          phase: "idle",
          error: msg,
        });
        throw e;
      }
    },
    [updateServerConnectionState],
  );

  // --------------------------------------------------------------------------
  // 提示词操作
  // --------------------------------------------------------------------------

  const getPrompt = useCallback(
    async (
      server: string,
      name: string,
      args: Record<string, unknown>,
    ): Promise<McpPromptResult> => {
      try {
        return await mcpApi.getPrompt(server, name, args);
      } catch (e) {
        console.error("[useMcp] 获取提示词失败:", e);
        throw e;
      }
    },
    [],
  );

  // --------------------------------------------------------------------------
  // 资源操作
  // --------------------------------------------------------------------------

  const readResource = useCallback(
    async (server: string, uri: string): Promise<McpResourceContent> => {
      try {
        return await mcpApi.readResource(server, uri);
      } catch (e) {
        console.error("[useMcp] 读取资源失败:", e);
        throw e;
      }
    },
    [],
  );

  const subscribeResource = useCallback(
    async (server: string, uri: string): Promise<void> => {
      try {
        await mcpApi.subscribeResource(server, uri);
      } catch (e) {
        console.error("[useMcp] 订阅资源失败:", e);
        throw e;
      }
    },
    [],
  );

  const unsubscribeResource = useCallback(
    async (server: string, uri: string): Promise<void> => {
      try {
        await mcpApi.unsubscribeResource(server, uri);
      } catch (e) {
        console.error("[useMcp] 取消订阅资源失败:", e);
        throw e;
      }
    },
    [],
  );

  // --------------------------------------------------------------------------
  // 初始化和事件监听
  // --------------------------------------------------------------------------

  useEffect(() => {
    let mounted = true;
    const unlisteners: UnlistenFn[] = [];

    const init = async () => {
      setLoading(true);
      try {
        await refreshServers();
        await refreshTools();
        await refreshPrompts();
        await refreshResources();
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    const setupListeners = async () => {
      try {
        const registered = await setupMcpEventListeners({
          isMounted: () => mounted,
          updateServerConnectionState,
          refreshServers,
          refreshTools,
          refreshResources,
          setError,
          setTools,
          setOAuthCompletion,
          setEventStreams,
        });
        unlisteners.push(...registered);
      } catch (error) {
        console.error("[useMcp] 注册 MCP 事件监听失败:", error);
      }
    };

    init();
    setupListeners();

    return () => {
      mounted = false;
      unlisteners.forEach((unlisten) => unlisten());
    };
  }, [
    refreshServers,
    refreshTools,
    refreshPrompts,
    refreshResources,
    updateServerConnectionState,
  ]);

  return {
    servers,
    tools,
    prompts,
    resources,
    loading,
    error,
    serverConnectionStates,
    oauthCompletion,
    eventStreams,
    startServer,
    stopServer,
    reconnectServer,
    loginOAuthServer,
    refreshServers,
    refreshTools,
    refreshPrompts,
    getPrompt,
    refreshResources,
    readResource,
    subscribeResource,
    unsubscribeResource,
  };
}
