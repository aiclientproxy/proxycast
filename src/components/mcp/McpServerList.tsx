/**
 * MCP 服务器列表组件
 *
 * 显示所有 MCP 服务器及其运行状态，支持启动/停止操作。
 *
 * @module components/mcp/McpServerList
 */

import { useState } from "react";
import { AlertCircle, RefreshCw, Server, Settings2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { cn } from "@/lib/utils";
import type { McpServerInfo, McpServerOAuthLoginOptions } from "@/lib/api/mcp";
import type { McpServerConnectionState } from "@/hooks/useMcp";
import type { McpEventStreamStateMap } from "@/lib/mcp/eventStreamProjection";
import { McpServerRow } from "./McpServerRow";
import { McpEventStreamStatus } from "./McpEventStreamStatus";
import { getMcpServerListSummary } from "./mcpServerListModel";

interface McpServerListProps {
  servers: McpServerInfo[];
  loading: boolean;
  error: string | null;
  onStartServer: (name: string) => Promise<void>;
  onStopServer: (name: string) => Promise<void>;
  onReconnectServer: (name: string) => Promise<void>;
  onLoginOAuthServer?: (
    name: string,
    options?: McpServerOAuthLoginOptions,
  ) => Promise<void>;
  onRefresh: () => Promise<void>;
  onSelectServer?: (server: McpServerInfo) => void;
  selectedServerName?: string;
  serverConnectionStates: Record<string, McpServerConnectionState>;
  eventStreams: McpEventStreamStateMap;
}

export function McpServerList({
  servers,
  loading,
  error,
  onStartServer,
  onStopServer,
  onReconnectServer,
  onLoginOAuthServer,
  onRefresh,
  onSelectServer,
  selectedServerName,
  serverConnectionStates,
  eventStreams,
}: McpServerListProps) {
  const { t } = useTranslation("settings");
  const [operatingServer, setOperatingServer] = useState<string | null>(null);
  const [oauthLoginServer, setOAuthLoginServer] = useState<string | null>(null);

  const runServerOperation = async (
    name: string,
    operation: (name: string) => Promise<void>,
  ) => {
    setOperatingServer(name);
    try {
      await operation(name);
    } finally {
      setOperatingServer(null);
    }
  };

  const handleOAuthLogin = async (server: McpServerInfo) => {
    if (!onLoginOAuthServer) {
      return;
    }
    setOAuthLoginServer(server.name);
    try {
      await onLoginOAuthServer(server.name, {
        scopes: server.runtime_status?.auth_status.action_plan?.scopes,
      });
    } finally {
      setOAuthLoginServer(null);
    }
  };

  const hasInteractiveSelection = Boolean(onSelectServer);
  const summary = getMcpServerListSummary(servers);

  return (
    <div className="flex min-h-[464px] flex-col bg-white">
      {/* 标题栏 */}
      <div className="flex items-center justify-between border-b border-slate-200/80 px-5 py-4">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-2xl border border-sky-100 bg-sky-50 text-sky-700">
            <Server className="h-4 w-4" />
          </div>
          <div>
            <p className="text-sm font-semibold text-slate-900">
              {t("settings.mcpPage.runtime.serverList.title")}
            </p>
            <p className="text-xs text-slate-500">
              {t("settings.mcpPage.runtime.serverList.summary", {
                total: summary.total,
                running: summary.running,
              })}
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={() => onRefresh()}
          disabled={loading}
          className="inline-flex h-9 w-9 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-500 transition hover:border-slate-300 hover:text-slate-900 disabled:cursor-not-allowed disabled:opacity-50"
          title={t("settings.mcpPage.runtime.serverList.refreshTitle")}
          aria-label={t("settings.mcpPage.runtime.serverList.refreshAria")}
        >
          <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
        </button>
      </div>

      {/* 错误提示 */}
      {error && (
        <div className="border-b border-rose-100 bg-rose-50 px-5 py-3">
          <div className="flex items-start gap-2 text-sm text-rose-700">
            <AlertCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
            <span>{error}</span>
          </div>
        </div>
      )}

      <McpEventStreamStatus eventStreams={eventStreams} />

      {/* 服务器列表 */}
      <div className="flex-1 overflow-auto p-4">
        {loading && servers.length === 0 ? (
          <div className="flex min-h-[320px] items-center justify-center">
            <div className="flex flex-col items-center gap-3 text-sm text-slate-500">
              <RefreshCw className="h-5 w-5 animate-spin" />
              {t("settings.mcpPage.runtime.serverList.loading")}
            </div>
          </div>
        ) : servers.length === 0 ? (
          <div className="flex min-h-[320px] items-center justify-center text-center">
            <div className="max-w-sm space-y-3">
              <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-3xl border border-slate-200 bg-slate-50 text-slate-500">
                <Settings2 className="h-5 w-5" />
              </div>
              <div className="space-y-1">
                <p className="text-sm font-semibold text-slate-900">
                  {t("settings.mcpPage.runtime.serverList.emptyTitle")}
                </p>
                <p className="text-sm leading-6 text-slate-500">
                  {t("settings.mcpPage.runtime.serverList.emptyDescription")}
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="grid gap-3 lg:grid-cols-2">
            {servers.map((server) => (
              <McpServerRow
                key={server.id}
                server={server}
                connectionState={serverConnectionStates[server.name]}
                isSelected={selectedServerName === server.name}
                isInteractive={hasInteractiveSelection}
                isOperating={operatingServer === server.name}
                isOAuthOperating={oauthLoginServer === server.name}
                onSelectServer={onSelectServer}
                onStartServer={(name) =>
                  runServerOperation(name, onStartServer)
                }
                onStopServer={(name) => runServerOperation(name, onStopServer)}
                onReconnectServer={(name) =>
                  runServerOperation(name, onReconnectServer)
                }
                onLoginOAuthServer={
                  onLoginOAuthServer ? handleOAuthLogin : undefined
                }
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
