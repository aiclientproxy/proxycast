import {
  mcpServerOauthLoginCompletedServerNotification,
  mcpServerStatusUpdatedServerNotification,
} from "@limecloud/app-server-client";
import { safeListen } from "@/lib/api/bridgeEvents";
import { subscribeAppServerNotifications } from "@/lib/api/appServerEventBus";
import type { UnlistenFn } from "@/lib/desktop-host/event";
import type { McpToolDefinition } from "@/lib/api/mcp";

interface McpToolsUpdatedPayload {
  tools: McpToolDefinition[];
}

interface McpResourcesUpdatedPayload {
  server_name: string;
}

interface McpResourceUpdatedPayload {
  server_name: string;
  uri: string;
}

export interface McpServerConnectionState {
  phase: "idle" | "starting" | "stopping" | "reconnecting";
  error: string | null;
  updatedAt: number | null;
}

export interface McpOAuthCompletionState {
  serverName: string;
  completedAt: number;
}

export interface SetupMcpEventListenersOptions {
  isMounted: () => boolean;
  updateServerConnectionState: (
    serverName: string,
    nextState: Partial<McpServerConnectionState> & {
      phase: McpServerConnectionState["phase"];
    },
  ) => void;
  refreshServers: () => void | Promise<void>;
  refreshTools: () => void | Promise<void>;
  refreshResources: () => void | Promise<void>;
  setError: (error: string) => void;
  setTools: (tools: McpToolDefinition[]) => void;
  setOAuthCompletion: (completion: McpOAuthCompletionState) => void;
}

export async function setupMcpEventListeners({
  isMounted,
  updateServerConnectionState,
  refreshServers,
  refreshTools,
  refreshResources,
  setError,
  setTools,
  setOAuthCompletion,
}: SetupMcpEventListenersOptions): Promise<UnlistenFn[]> {
  const unlisteners: UnlistenFn[] = [];

  try {
    // 必须先同步订阅；否则快速 MCP 通知可能在 Desktop listener 就绪前被排空。
    const unlistenAppServerMcpNotifications = subscribeAppServerNotifications({
      onNotifications: (notifications) => {
        for (const message of notifications) {
          const startupNotification =
            mcpServerStatusUpdatedServerNotification(message);
          if (startupNotification) {
            const { error, name, status } = startupNotification.params;
            if (status === "starting") {
              console.log("[useMcp] 服务器正在启动:", name);
              updateServerConnectionState(name, {
                phase: "starting",
              });
              continue;
            }

            if (status === "failed") {
              console.warn("[useMcp] 服务器启动失败:", name, error);
            } else {
              console.log("[useMcp] 服务器启动状态已更新:", name, status);
            }
            updateServerConnectionState(name, {
              phase: "idle",
              error: status === "failed" ? error : null,
            });
            const refresh = Promise.all([
              Promise.resolve(refreshServers()),
              Promise.resolve(refreshTools()),
            ]);
            if (status === "failed" && error) {
              void refresh.then(() => {
                if (isMounted()) {
                  setError(`${name}: ${error}`);
                }
              });
            }
            continue;
          }

          const notification =
            mcpServerOauthLoginCompletedServerNotification(message);
          if (!notification) {
            continue;
          }
          const { error, name, success } = notification.params;
          if (success) {
            console.log("[useMcp] OAuth 授权已完成:", name);
            updateServerConnectionState(name, {
              phase: "idle",
            });
            if (isMounted()) {
              setOAuthCompletion({
                serverName: name,
                completedAt: Date.now(),
              });
            }
          } else {
            console.error("[useMcp] OAuth 授权失败:", name, error);
            updateServerConnectionState(name, {
              phase: "idle",
              error: error ?? null,
            });
          }
          const refresh = Promise.all([
            Promise.resolve(refreshServers()),
            Promise.resolve(refreshTools()),
          ]);
          if (!success && error) {
            void refresh.then(() => {
              if (isMounted()) {
                setError(`${name}: ${error}`);
              }
            });
          }
        }
      },
    });
    unlisteners.push(unlistenAppServerMcpNotifications);

    const unlistenTools = await safeListen<McpToolsUpdatedPayload>(
      "mcp:tools_updated",
      (event) => {
        console.log("[useMcp] 工具列表已更新:", event.payload.tools.length);
        if (isMounted()) {
          setTools(event.payload.tools);
        }
      },
    );
    unlisteners.push(unlistenTools);

    const unlistenResources = await safeListen<McpResourcesUpdatedPayload>(
      "mcp:resources_updated",
      (event) => {
        console.log("[useMcp] 资源列表已更新:", event.payload.server_name);
        refreshResources();
      },
    );
    unlisteners.push(unlistenResources);

    const unlistenResource = await safeListen<McpResourceUpdatedPayload>(
      "mcp:resource_updated",
      (event) => {
        console.log(
          "[useMcp] 资源已更新:",
          event.payload.server_name,
          event.payload.uri,
        );
        refreshResources();
      },
    );
    unlisteners.push(unlistenResource);

    return unlisteners;
  } catch (error) {
    unlisteners.forEach((unlisten) => unlisten());
    throw error;
  }
}
