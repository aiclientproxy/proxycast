import { useState, useEffect, forwardRef, useImperativeHandle } from "react";
import {
  RefreshCw,
  Plus,
  Heart,
  HeartOff,
  RotateCcw,
  Activity,
  Download,
} from "lucide-react";
import { useProviderPool } from "@/hooks/useProviderPool";
import { CredentialCard } from "./CredentialCard";
import { AddCredentialModal } from "./AddCredentialModal";
import { EditCredentialModal } from "./EditCredentialModal";
import { ErrorDisplay, useErrorDisplay } from "./ErrorDisplay";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { HelpTip } from "@/components/HelpTip";
import { getConfig, saveConfig, Config } from "@/hooks/useTauri";
import { GeminiApiKeySection } from "./GeminiApiKeySection";
import { VertexAISection } from "./VertexAISection";
import { AmpConfigSection } from "./AmpConfigSection";
import { ProviderIcon } from "@/icons/providers";
import type {
  PoolProviderType,
  CredentialDisplay,
  UpdateCredentialRequest,
} from "@/lib/api/providerPool";

export interface ProviderPoolPageRef {
  refresh: () => void;
}

// All provider types (OAuth/API Key 凭证)
const allProviderTypes: PoolProviderType[] = [
  "kiro",
  "gemini",
  "qwen",
  "antigravity",
  "openai",
  "claude",
  "codex",
  "claude_oauth",
  "iflow",
];

// 配置类型 tab（非凭证池）
type ConfigTabType = "gemini_api" | "vertex" | "amp";

// 所有 tab 类型
type TabType = PoolProviderType | ConfigTabType;

const providerLabels: Record<PoolProviderType, string> = {
  kiro: "Kiro (AWS)",
  gemini: "Gemini (Google)",
  qwen: "Qwen (阿里)",
  antigravity: "Antigravity (Gemini 3 Pro)",
  openai: "OpenAI",
  claude: "Claude (Anthropic)",
  codex: "Codex (OAuth / API Key)",
  claude_oauth: "Claude OAuth",
  iflow: "iFlow",
};

const configTabLabels: Record<ConfigTabType, string> = {
  gemini_api: "Gemini API Key",
  vertex: "Vertex AI",
  amp: "Amp CLI",
};

// 判断是否为配置类型 tab
const isConfigTab = (tab: TabType): tab is ConfigTabType => {
  return ["gemini_api", "vertex", "amp"].includes(tab);
};

export const ProviderPoolPage = forwardRef<ProviderPoolPageRef>(
  (_props, ref) => {
    const [addModalOpen, setAddModalOpen] = useState(false);
    const [editModalOpen, setEditModalOpen] = useState(false);
    const [editingCredential, setEditingCredential] =
      useState<CredentialDisplay | null>(null);
    const [activeTab, setActiveTab] = useState<TabType>("kiro");
    const [deletingCredentials, setDeletingCredentials] = useState<Set<string>>(
      new Set(),
    );
    const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
    const { errors, showError, showSuccess, dismissError } = useErrorDisplay();

    const {
      overview,
      loading,
      error,
      checkingHealth,
      refreshingToken,
      refresh,
      deleteCredential,
      toggleCredential,
      resetCredential,
      resetHealth,
      checkCredentialHealth,
      checkTypeHealth,
      refreshCredentialToken,
      updateCredential,
      migratePrivateConfig,
    } = useProviderPool();

    const [migrating, setMigrating] = useState(false);

    // 配置 tab 相关状态
    const [config, setConfig] = useState<Config | null>(null);
    const [configLoading, setConfigLoading] = useState(false);
    const [configSaving, setConfigSaving] = useState(false);

    // 加载配置
    const loadConfig = async () => {
      setConfigLoading(true);
      try {
        const c = await getConfig();
        setConfig(c);
      } catch (e) {
        console.error("Failed to load config:", e);
        showError("加载配置失败", "config");
      }
      setConfigLoading(false);
    };

    // 保存配置
    const handleSaveConfig = async () => {
      if (!config) return;
      setConfigSaving(true);
      try {
        await saveConfig(config);
        showSuccess("配置已保存");
      } catch (e) {
        showError(e instanceof Error ? e.message : String(e), "config");
      }
      setConfigSaving(false);
    };

    // 切换到配置 tab 时加载配置
    useEffect(() => {
      if (isConfigTab(activeTab)) {
        loadConfig();
      }
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [activeTab]);

    useImperativeHandle(ref, () => ({
      refresh,
    }));

    const handleDeleteClick = (uuid: string) => {
      setDeleteConfirm(uuid);
    };

    const handleDeleteConfirm = async () => {
      if (!deleteConfirm) return;
      const uuid = deleteConfirm;
      setDeleteConfirm(null);
      setDeletingCredentials((prev) => new Set(prev).add(uuid));
      try {
        // Pass activeTab (provider_type) to enable YAML config sync
        // 只有凭证池 tab 才传递 provider_type
        const providerType = !isConfigTab(activeTab)
          ? (activeTab as PoolProviderType)
          : undefined;
        await deleteCredential(uuid, providerType);
      } catch (e) {
        showError(e instanceof Error ? e.message : String(e), "delete", uuid);
      } finally {
        setDeletingCredentials((prev) => {
          const next = new Set(prev);
          next.delete(uuid);
          return next;
        });
      }
    };

    const handleToggle = async (credential: CredentialDisplay) => {
      try {
        await toggleCredential(credential.uuid, !credential.is_disabled);
      } catch (e) {
        showError(
          e instanceof Error ? e.message : String(e),
          "toggle",
          credential.uuid,
        );
      }
    };

    const handleReset = async (uuid: string) => {
      try {
        await resetCredential(uuid);
      } catch (e) {
        showError(e instanceof Error ? e.message : String(e), "reset", uuid);
      }
    };

    const handleCheckHealth = async (uuid: string) => {
      try {
        const result = await checkCredentialHealth(uuid);
        if (result.success) {
          showSuccess("健康检查通过！", uuid);
        } else {
          showError(result.message || "健康检查未通过", "health_check", uuid);
        }
      } catch (e) {
        showError(
          e instanceof Error ? e.message : String(e),
          "health_check",
          uuid,
        );
      }
    };

    const handleCheckTypeHealth = async (providerType: PoolProviderType) => {
      try {
        await checkTypeHealth(providerType);
      } catch (e) {
        showError(e instanceof Error ? e.message : String(e), "health_check");
      }
    };

    const handleResetTypeHealth = async (providerType: PoolProviderType) => {
      try {
        await resetHealth(providerType);
      } catch (e) {
        showError(e instanceof Error ? e.message : String(e), "reset");
      }
    };

    // 迁移 Private 配置到凭证池
    const handleMigratePrivateConfig = async () => {
      setMigrating(true);
      try {
        const config = await getConfig();
        const result = await migratePrivateConfig(config);
        if (result.migrated_count > 0) {
          showSuccess(
            `成功迁移 ${result.migrated_count} 个凭证${result.skipped_count > 0 ? `，跳过 ${result.skipped_count} 个已存在的凭证` : ""}`,
          );
        } else if (result.skipped_count > 0) {
          showSuccess(`所有凭证已存在，跳过 ${result.skipped_count} 个`);
        } else {
          showSuccess("没有需要迁移的凭证");
        }
        if (result.errors.length > 0) {
          showError(`部分迁移失败: ${result.errors.join(", ")}`, "migrate");
        }
      } catch (e) {
        showError(e instanceof Error ? e.message : String(e), "migrate");
      } finally {
        setMigrating(false);
      }
    };

    const handleRefreshToken = async (uuid: string) => {
      try {
        await refreshCredentialToken(uuid);
        showSuccess("Token 刷新成功！", uuid);
      } catch (e) {
        showError(
          e instanceof Error ? e.message : String(e),
          "refresh_token",
          uuid,
        );
      }
    };

    const handleEdit = (credential: CredentialDisplay) => {
      setEditingCredential(credential);
      setEditModalOpen(true);
    };

    const handleEditSubmit = async (
      uuid: string,
      request: UpdateCredentialRequest,
    ) => {
      try {
        await updateCredential(uuid, request);
      } catch (e) {
        throw new Error(
          `编辑失败: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    };

    const closeEditModal = () => {
      setEditModalOpen(false);
      setEditingCredential(null);
    };

    const openAddModal = () => {
      setAddModalOpen(true);
    };

    const getProviderOverview = (providerType: PoolProviderType) => {
      return overview.find((p) => p.provider_type === providerType);
    };

    const getCredentialCount = (providerType: PoolProviderType) => {
      const pool = getProviderOverview(providerType);
      return pool?.credentials?.length || 0;
    };

    // Current tab data (仅用于凭证池 tab)
    const currentPool = !isConfigTab(activeTab)
      ? getProviderOverview(activeTab as PoolProviderType)
      : null;
    const currentStats = currentPool?.stats;
    const currentCredentials = currentPool?.credentials || [];

    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold">凭证池</h2>
            <p className="text-muted-foreground">
              管理多个 AI 服务凭证，支持负载均衡和健康检测
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleMigratePrivateConfig}
              disabled={migrating || loading}
              className="flex items-center gap-2 rounded-lg border px-3 py-2 text-sm hover:bg-muted disabled:opacity-50"
              title="从高级设置导入 Private 凭证"
            >
              <Download
                className={`h-4 w-4 ${migrating ? "animate-pulse" : ""}`}
              />
              导入配置
            </button>
            <button
              onClick={refresh}
              disabled={loading}
              className="flex items-center gap-2 rounded-lg border px-3 py-2 text-sm hover:bg-muted disabled:opacity-50"
            >
              <RefreshCw
                className={`h-4 w-4 ${loading ? "animate-spin" : ""}`}
              />
              刷新
            </button>
          </div>
        </div>

        <HelpTip title="什么是凭证池？" variant="amber">
          <ul className="list-disc list-inside space-y-1 text-sm text-amber-700 dark:text-amber-400">
            <li>
              <span className="font-medium">Kiro/Gemini/Qwen</span>
              ：上传对应工具的凭证文件，ProxyCast 会自动管理 Token 刷新
            </li>
            <li>
              <span className="font-medium">OpenAI/Claude</span>：直接填入 API
              Key，用于转发请求
            </li>
            <li>多个凭证会自动轮询负载均衡，单个凭证失效不影响服务</li>
            <li>
              在 <span className="font-medium">API Server</span> 页面选择默认
              Provider 后，请求会自动使用对应凭证池
            </li>
          </ul>
        </HelpTip>

        {error && (
          <div className="rounded-lg border border-red-500 bg-red-50 p-4 text-red-700 dark:bg-red-950/30">
            {error}
          </div>
        )}

        {/* Tabs */}
        <div className="flex gap-2 border-b overflow-x-auto">
          {/* 凭证池 tabs */}
          {allProviderTypes.map((providerType) => {
            const count = getCredentialCount(providerType);
            return (
              <button
                key={providerType}
                onClick={() => setActiveTab(providerType)}
                className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px whitespace-nowrap flex items-center gap-2 ${
                  activeTab === providerType
                    ? "border-primary text-primary"
                    : "border-transparent text-muted-foreground hover:text-foreground"
                }`}
              >
                <ProviderIcon providerType={providerType} size={16} />
                {providerLabels[providerType]}
                {count > 0 && (
                  <span className="rounded-full bg-muted px-1.5 py-0.5 text-xs">
                    {count}
                  </span>
                )}
              </button>
            );
          })}
          {/* 分隔符 */}
          <div className="border-l mx-2" />
          {/* 配置 tabs */}
          {(Object.keys(configTabLabels) as ConfigTabType[]).map((tabId) => (
            <button
              key={tabId}
              onClick={() => setActiveTab(tabId)}
              className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px whitespace-nowrap ${
                activeTab === tabId
                  ? "border-primary text-primary"
                  : "border-transparent text-muted-foreground hover:text-foreground"
              }`}
            >
              {configTabLabels[tabId]}
            </button>
          ))}
        </div>

        {/* 配置 Tab 内容 */}
        {isConfigTab(activeTab) ? (
          configLoading ? (
            <div className="flex items-center justify-center py-12">
              <RefreshCw className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : config ? (
            <div className="space-y-4">
              {activeTab === "gemini_api" && (
                <>
                  <GeminiApiKeySection
                    entries={config.credential_pool?.gemini_api_keys ?? []}
                    onChange={(entries) =>
                      setConfig({
                        ...config,
                        credential_pool: {
                          ...config.credential_pool,
                          gemini_api_keys: entries,
                        },
                      })
                    }
                  />
                  <button
                    onClick={handleSaveConfig}
                    disabled={configSaving}
                    className="w-full px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 disabled:opacity-50"
                  >
                    {configSaving ? "保存中..." : "保存配置"}
                  </button>
                </>
              )}

              {activeTab === "vertex" && (
                <>
                  <VertexAISection
                    entries={config.credential_pool?.vertex_api_keys ?? []}
                    onChange={(entries) =>
                      setConfig({
                        ...config,
                        credential_pool: {
                          ...config.credential_pool,
                          vertex_api_keys: entries,
                        },
                      })
                    }
                  />
                  <button
                    onClick={handleSaveConfig}
                    disabled={configSaving}
                    className="w-full px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 disabled:opacity-50"
                  >
                    {configSaving ? "保存中..." : "保存配置"}
                  </button>
                </>
              )}

              {activeTab === "amp" && (
                <AmpConfigSection
                  config={
                    config.ampcode ?? {
                      upstream_url: null,
                      model_mappings: [],
                      restrict_management_to_localhost: false,
                    }
                  }
                  onChange={(ampConfig) =>
                    setConfig({
                      ...config,
                      ampcode: ampConfig,
                    })
                  }
                  onSave={handleSaveConfig}
                />
              )}
            </div>
          ) : (
            <div className="flex items-center justify-center py-12 text-muted-foreground">
              加载配置失败
            </div>
          )
        ) : loading ? (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <div className="space-y-4">
            {/* Stats and Actions Bar */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                {currentStats && currentStats.total > 0 && (
                  <div className="flex items-center gap-3 text-sm text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <Heart className="h-4 w-4 text-green-500" />
                      健康: {currentStats.healthy}
                    </span>
                    <span className="flex items-center gap-1">
                      <HeartOff className="h-4 w-4 text-red-500" />
                      不健康: {currentStats.unhealthy}
                    </span>
                    <span>总计: {currentStats.total}</span>
                  </div>
                )}
              </div>
              <div className="flex items-center gap-2">
                {currentCredentials.length > 0 && (
                  <>
                    <button
                      onClick={() =>
                        handleCheckTypeHealth(activeTab as PoolProviderType)
                      }
                      disabled={checkingHealth === activeTab}
                      className="flex items-center gap-1 rounded-lg border px-3 py-1.5 text-sm hover:bg-muted disabled:opacity-50"
                    >
                      <Activity
                        className={`h-4 w-4 ${checkingHealth === activeTab ? "animate-pulse" : ""}`}
                      />
                      检测全部
                    </button>
                    <button
                      onClick={() =>
                        handleResetTypeHealth(activeTab as PoolProviderType)
                      }
                      className="flex items-center gap-1 rounded-lg border px-3 py-1.5 text-sm hover:bg-muted"
                    >
                      <RotateCcw className="h-4 w-4" />
                      重置状态
                    </button>
                  </>
                )}
                <button
                  onClick={openAddModal}
                  className="flex items-center gap-1 rounded-lg bg-primary px-3 py-1.5 text-sm text-primary-foreground hover:bg-primary/90"
                >
                  <Plus className="h-4 w-4" />
                  添加凭证
                </button>
              </div>
            </div>

            {/* Credentials List */}
            {currentCredentials.length === 0 ? (
              <div className="flex flex-col items-center justify-center rounded-lg border border-dashed py-12 text-muted-foreground">
                <p className="text-lg">
                  暂无 {providerLabels[activeTab as PoolProviderType]} 凭证
                </p>
                <p className="mt-1 text-sm">点击上方"添加凭证"按钮添加</p>
                <button
                  onClick={openAddModal}
                  className="mt-4 flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm text-primary-foreground hover:bg-primary/90"
                >
                  <Plus className="h-4 w-4" />
                  添加第一个凭证
                </button>
              </div>
            ) : (
              <div className="flex flex-col gap-4">
                {currentCredentials.map((credential) => {
                  // 判断是否为 OAuth 类型（需要刷新 Token 功能）
                  const isOAuthType =
                    credential.credential_type.includes("oauth");
                  // 判断是否为 Kiro 凭证（支持用量查询）
                  const isKiroCredential = activeTab === "kiro";
                  return (
                    <CredentialCard
                      key={credential.uuid}
                      credential={credential}
                      onToggle={() => handleToggle(credential)}
                      onDelete={() => handleDeleteClick(credential.uuid)}
                      onReset={() => handleReset(credential.uuid)}
                      onCheckHealth={() => handleCheckHealth(credential.uuid)}
                      onRefreshToken={
                        isOAuthType
                          ? () => handleRefreshToken(credential.uuid)
                          : undefined
                      }
                      onEdit={() => handleEdit(credential)}
                      deleting={deletingCredentials.has(credential.uuid)}
                      checkingHealth={checkingHealth === credential.uuid}
                      refreshingToken={refreshingToken === credential.uuid}
                      isKiroCredential={isKiroCredential}
                    />
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* Add Credential Modal (仅凭证池 tab) */}
        {addModalOpen && !isConfigTab(activeTab) && (
          <AddCredentialModal
            providerType={activeTab as PoolProviderType}
            onClose={() => {
              setAddModalOpen(false);
            }}
            onSuccess={() => {
              setAddModalOpen(false);
              refresh();
            }}
          />
        )}

        {/* Edit Credential Modal */}
        <EditCredentialModal
          credential={editingCredential}
          isOpen={editModalOpen}
          onClose={closeEditModal}
          onEdit={handleEditSubmit}
        />

        {/* Error Display */}
        <ErrorDisplay
          errors={errors}
          onDismiss={dismissError}
          onRetry={(error) => {
            // 根据错误类型提供重试功能
            switch (error.type) {
              case "health_check":
                if (error.uuid) {
                  handleCheckHealth(error.uuid);
                }
                break;
              case "refresh_token":
                if (error.uuid) {
                  handleRefreshToken(error.uuid);
                }
                break;
              case "reset":
                if (error.uuid) {
                  handleReset(error.uuid);
                }
                break;
            }
            dismissError(error.id);
          }}
        />

        <ConfirmDialog
          isOpen={!!deleteConfirm}
          title="删除确认"
          message="确定要删除这个凭证吗？"
          onConfirm={handleDeleteConfirm}
          onCancel={() => setDeleteConfirm(null)}
        />
      </div>
    );
  },
);

ProviderPoolPage.displayName = "ProviderPoolPage";
