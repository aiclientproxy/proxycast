/**
 * MCP 管理面板
 *
 * 整合配置管理、运行时状态、工具/提示词/资源浏览为一体的完整 MCP 管理界面。
 * 采用左右分栏布局：左侧为服务器列表和运行控制，右侧为 Tab 切换的功能面板。
 *
 * @module components/mcp/McpPanel
 */

import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { useMcp } from "@/hooks/useMcp";
import { openExternalUrlWithSystemBrowser } from "@/lib/api/externalUrl";
import { McpPage } from "./McpPage";
import { McpPanelHeader } from "./McpPanelHeader";
import { McpPanelTabs } from "./McpPanelTabs";
import { McpServerList } from "./McpServerList";
import { McpToolsBrowser } from "./McpToolsBrowser";
import { McpPromptsBrowser } from "./McpPromptsBrowser";
import { McpResourcesBrowser } from "./McpResourcesBrowser";
import {
  getMcpCapabilityCount,
  getMcpPanelStatusMeta,
  getMcpTabCount,
  getRunningMcpServerCount,
  mcpPanelTabs,
  type McpTab,
  type McpPanelTabCounts,
} from "./mcpPanelModel";
import type { McpServerOAuthLoginOptions } from "@/lib/api/mcp";

interface McpPanelProps {
  hideHeader?: boolean;
}

export function McpPanel({ hideHeader = false }: McpPanelProps) {
  const { t } = useTranslation("settings");
  const [activeTab, setActiveTab] = useState<McpTab>("runtime");
  const {
    servers,
    tools,
    prompts,
    resources,
    loading,
    error,
    serverConnectionStates,
    oauthCompletion,
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
  } = useMcp();

  const runningServerCount = getRunningMcpServerCount(servers);
  const capabilityCount = getMcpCapabilityCount({
    tools,
    prompts,
    resources,
  });
  const tabCounts: McpPanelTabCounts = {
    servers: servers.length,
    tools: tools.length,
    prompts: prompts.length,
    resources: resources.length,
  };
  const activeTabDefinition =
    mcpPanelTabs.find((tab) => tab.id === activeTab) ?? mcpPanelTabs[0];
  const ActiveTabIcon = activeTabDefinition.icon;
  const statusMeta = getMcpPanelStatusMeta({ loading, error });
  const getTabCount = (tab: McpTab) => getMcpTabCount(tab, tabCounts);

  useEffect(() => {
    if (!oauthCompletion) {
      return;
    }
    toast.success(
      t("settings.mcpPage.runtime.auth.oauthCompleted", {
        name: oauthCompletion.serverName,
      }),
    );
  }, [oauthCompletion, t]);

  const handleLoginOAuthServer = async (
    serverName: string,
    options?: McpServerOAuthLoginOptions,
  ): Promise<void> => {
    try {
      const response = await loginOAuthServer(serverName, options);
      await openExternalUrlWithSystemBrowser(response.authorizationUrl);
      toast.success(
        t("settings.mcpPage.runtime.auth.oauthOpened", {
          name: serverName,
        }),
      );
    } catch (error) {
      toast.error(
        t("settings.mcpPage.runtime.auth.oauthOpenFailed", {
          message: error instanceof Error ? error.message : String(error),
        }),
      );
    }
  };

  return (
    <div
      data-settings-embedded={hideHeader ? "true" : "false"}
      className="space-y-6 pb-20 text-slate-900"
    >
      <McpPanelHeader
        serverCount={servers.length}
        runningServerCount={runningServerCount}
        capabilityCount={capabilityCount}
        statusMeta={statusMeta}
      />

      <McpPanelTabs
        tabs={mcpPanelTabs}
        activeTab={activeTab}
        onTabChange={setActiveTab}
        getTabCount={getTabCount}
      />

      {/* Tab 内容 */}
      <section className="min-h-[520px] overflow-hidden rounded-[26px] border border-slate-200/80 bg-white shadow-sm shadow-slate-950/5">
        <div className="border-b border-slate-200/80 bg-slate-50/80 px-5 py-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
            <ActiveTabIcon className="h-4 w-4 text-sky-600" />
            {t(activeTabDefinition.labelKey)}
          </div>
        </div>
        <div className="min-h-[464px]">
          {/* 运行状态 Tab */}
          {activeTab === "runtime" && (
            <div className="min-h-[464px]">
              <McpServerList
                servers={servers}
                loading={loading}
                error={error}
                serverConnectionStates={serverConnectionStates}
                onStartServer={startServer}
                onStopServer={stopServer}
                onReconnectServer={reconnectServer}
                onLoginOAuthServer={handleLoginOAuthServer}
                onRefresh={refreshServers}
              />
            </div>
          )}

          {/* 工具 Tab */}
          {activeTab === "tools" && (
            <div className="min-h-[464px] p-4">
              <div className="min-h-[420px] overflow-hidden rounded-[22px] border border-slate-200/80 bg-white">
                <McpToolsBrowser
                  tools={tools}
                  loading={loading}
                  onRefresh={refreshTools}
                  serverCount={servers.length}
                  runningServerCount={runningServerCount}
                  onOpenRuntimeTab={() => setActiveTab("runtime")}
                  onOpenConfigTab={() => setActiveTab("config")}
                />
              </div>
            </div>
          )}

          {/* 提示词 Tab */}
          {activeTab === "prompts" && (
            <div className="min-h-[464px] p-4">
              <div className="min-h-[420px] overflow-hidden rounded-[22px] border border-slate-200/80 bg-white">
                <McpPromptsBrowser
                  prompts={prompts}
                  loading={loading}
                  onRefresh={refreshPrompts}
                  onGetPrompt={getPrompt}
                />
              </div>
            </div>
          )}

          {/* 资源 Tab */}
          {activeTab === "resources" && (
            <div className="min-h-[464px] p-4">
              <div className="min-h-[420px] overflow-hidden rounded-[22px] border border-slate-200/80 bg-white">
                <McpResourcesBrowser
                  resources={resources}
                  loading={loading}
                  onRefresh={refreshResources}
                  onReadResource={readResource}
                  onSubscribeResource={subscribeResource}
                  onUnsubscribeResource={unsubscribeResource}
                />
              </div>
            </div>
          )}

          {/* 配置管理 Tab */}
          {activeTab === "config" && (
            <div className="min-h-[464px] overflow-auto p-4">
              <McpPage hideHeader />
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
