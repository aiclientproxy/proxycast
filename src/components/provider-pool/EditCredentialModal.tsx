import { useState, useEffect } from "react";
import {
  X,
  Eye,
  EyeOff,
  Settings,
  Upload,
  CheckCircle,
  Ban,
  Globe,
} from "lucide-react";
import { open } from "@tauri-apps/plugin-dialog";
import {
  CredentialDisplay,
  UpdateCredentialRequest,
  PoolProviderType,
} from "@/lib/api/providerPool";
import { validateProxyUrl } from "@/lib/utils";

interface EditCredentialModalProps {
  credential: CredentialDisplay | null;
  isOpen: boolean;
  onClose: () => void;
  onEdit: (uuid: string, request: UpdateCredentialRequest) => Promise<void>;
}

// 各 Provider 支持的模型列表 (参考 AIClient-2-API/src/provider-models.js)
const providerModels: Record<PoolProviderType, string[]> = {
  kiro: [
    "claude-opus-4-5",
    "claude-opus-4-5-20251101",
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
  ],
  gemini: [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.5-pro-preview-06-05",
    "gemini-2.5-flash-preview-09-2025",
    "gemini-3-pro-preview",
  ],
  qwen: ["qwen3-coder-plus", "qwen3-coder-flash"],
  antigravity: [
    "gemini-3-pro-preview",
    "gemini-3-pro-image-preview",
    "gemini-2.5-computer-use-preview-10-2025",
    "gemini-claude-sonnet-4-5",
    "gemini-claude-sonnet-4-5-thinking",
  ],
  openai: [], // 自定义 API，无预设模型
  claude: [], // 自定义 API，无预设模型
  codex: ["gpt-4o", "gpt-4o-mini", "o1", "o1-mini", "o3-mini"], // Codex（OAuth / API Key）
  claude_oauth: [
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
    "claude-sonnet-4-20250514",
  ], // Claude OAuth
  iflow: ["deepseek-chat", "deepseek-reasoner"], // iFlow
};

export function EditCredentialModal({
  credential,
  isOpen,
  onClose,
  onEdit,
}: EditCredentialModalProps) {
  const [name, setName] = useState("");
  const [checkHealth, setCheckHealth] = useState(true);
  const [checkModelName, setCheckModelName] = useState("");
  const [notSupportedModels, setNotSupportedModels] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showCredentialDetails, setShowCredentialDetails] = useState(false);

  // 重新上传文件相关状态
  const [newCredFilePath, setNewCredFilePath] = useState("");
  const [newProjectId, setNewProjectId] = useState("");
  // API Key 相关状态
  const [newBaseUrl, setNewBaseUrl] = useState("");
  const [newApiKey, setNewApiKey] = useState("");
  const [showApiKey, setShowApiKey] = useState(false);

  // 代理 URL 相关状态
  const [proxyUrl, setProxyUrl] = useState("");
  const [proxyError, setProxyError] = useState<string | null>(null);

  // 初始化表单数据
  useEffect(() => {
    if (credential) {
      console.log("[EditCredentialModal] 初始化表单数据:", {
        uuid: credential.uuid,
        name: credential.name,
        check_model_name: credential.check_model_name,
        not_supported_models: credential.not_supported_models,
        base_url: credential.base_url,
        api_key: credential.api_key ? "***" : undefined,
      });
      setName(credential.name || "");
      setCheckHealth(credential.check_health);
      setCheckModelName(credential.check_model_name || "");
      setNotSupportedModels(credential.not_supported_models || []);
      setNewCredFilePath("");
      setNewProjectId("");
      // 初始化 base_url 为已保存的值
      setNewBaseUrl(credential.base_url || "");
      // 初始化 api_key 为已保存的值
      setNewApiKey(credential.api_key || "");
      setShowApiKey(false);
      // 初始化代理 URL 为已保存的值
      setProxyUrl(credential.proxy_url || "");
      setProxyError(null);
      setError(null);
    }
  }, [credential]);

  if (!isOpen || !credential) {
    return null;
  }

  const isOAuth = credential.credential_type.includes("oauth");
  const isApiKey = credential.credential_type.includes("key");

  // 获取当前 provider 类型
  const getProviderType = (): PoolProviderType => {
    if (credential.credential_type.includes("kiro")) return "kiro";
    if (credential.credential_type.includes("gemini")) return "gemini";
    if (credential.credential_type.includes("qwen")) return "qwen";
    if (credential.credential_type.includes("codex")) return "codex";
    if (credential.credential_type === "claude_oauth") return "claude_oauth";
    if (credential.credential_type.includes("iflow")) return "iflow";
    if (credential.credential_type.includes("openai")) return "openai";
    if (credential.credential_type.includes("claude")) return "claude";
    return "kiro";
  };

  const currentProviderModels = providerModels[getProviderType()] || [];

  const handleSelectNewFile = async () => {
    try {
      const selected = await open({
        multiple: false,
        filters: [{ name: "JSON", extensions: ["json"] }],
      });
      if (selected) {
        setNewCredFilePath(selected as string);
      }
    } catch (e) {
      console.error("Failed to open file dialog:", e);
    }
  };

  const getMaskedCredentialInfo = () => {
    if (isOAuth) {
      const path = credential.display_credential;
      const parts = path.split("/");
      if (parts.length > 1) {
        const fileName = parts[parts.length - 1];
        const dirPath = parts.slice(0, -1).join("/");
        return `${dirPath}/***${fileName.slice(-8)}`;
      }
      return `***${path.slice(-12)}`;
    } else {
      return credential.display_credential;
    }
  };

  const toggleModelSupport = (model: string) => {
    setNotSupportedModels((prev) =>
      prev.includes(model) ? prev.filter((m) => m !== model) : [...prev, model],
    );
  };

  const handleProxyUrlChange = (value: string) => {
    setProxyUrl(value);
    if (value && !validateProxyUrl(value)) {
      setProxyError(
        "代理 URL 格式无效，请使用 http://、https:// 或 socks5:// 开头的地址",
      );
    } else {
      setProxyError(null);
    }
  };

  const handleSubmit = async () => {
    // 验证代理 URL 格式
    if (proxyUrl && !validateProxyUrl(proxyUrl)) {
      setProxyError(
        "代理 URL 格式无效，请使用 http://、https:// 或 socks5:// 开头的地址",
      );
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const updateRequest: UpdateCredentialRequest = {
        // 始终传递 name，空字符串表示清除名称
        name: name.trim(),
        check_health: checkHealth,
        // 始终传递 check_model_name，空字符串表示清除
        check_model_name: checkModelName.trim(),
        // 始终传递 not_supported_models，即使为空数组（用于清除选择）
        not_supported_models: notSupportedModels,
        new_creds_file_path: newCredFilePath.trim() || undefined,
        new_project_id: newProjectId.trim() || undefined,
        // API Key 的 base_url（始终传递当前值，空字符串表示使用默认 URL）
        new_base_url: isApiKey ? newBaseUrl.trim() : undefined,
        // API Key 的 api_key（始终传递当前值）
        new_api_key: isApiKey ? newApiKey.trim() : undefined,
        // 代理 URL（空字符串表示清除，使用全局代理）
        new_proxy_url: proxyUrl.trim() || undefined,
      };

      console.log("[EditCredentialModal] 提交更新请求:", updateRequest);
      await onEdit(credential.uuid, updateRequest);
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="w-full max-w-2xl max-h-[85vh] rounded-lg bg-background shadow-xl flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between border-b pb-4 px-6 pt-6 shrink-0">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Settings className="h-5 w-5" />
            编辑凭证
          </h3>
          <button onClick={onClose} className="rounded-lg p-1 hover:bg-muted">
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content - Scrollable */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          <div className="space-y-5">
            {/* 名称 + 健康检查 */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="mb-1 block text-sm font-medium">
                  名称 (选填)
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="给这个凭证起个名字..."
                  className="w-full rounded-lg border bg-background px-3 py-2 text-sm"
                />
              </div>
              <div>
                <label className="mb-1 block text-sm font-medium">
                  健康检查
                </label>
                <select
                  value={checkHealth ? "enabled" : "disabled"}
                  onChange={(e) => setCheckHealth(e.target.value === "enabled")}
                  className="w-full rounded-lg border bg-background px-3 py-2 text-sm"
                >
                  <option value="enabled">启用</option>
                  <option value="disabled">禁用</option>
                </select>
              </div>
            </div>

            {/* 检查模型名称 */}
            <div>
              <label className="mb-1 block text-sm font-medium">
                检查模型名称 (选填)
              </label>
              <input
                type="text"
                value={checkModelName}
                onChange={(e) => setCheckModelName(e.target.value)}
                placeholder="用于健康检查的模型名称..."
                className="w-full rounded-lg border bg-background px-3 py-2 text-sm"
              />
            </div>

            {/* OAuth凭据文件路径 */}
            {isOAuth && (
              <div>
                <label className="mb-1 block text-sm font-medium">
                  OAuth凭据文件路径
                </label>
                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    value={
                      showCredentialDetails
                        ? credential.display_credential
                        : getMaskedCredentialInfo()
                    }
                    readOnly
                    className="flex-1 rounded-lg border bg-muted/50 px-3 py-2 text-sm text-muted-foreground"
                  />
                  <button
                    type="button"
                    onClick={() =>
                      setShowCredentialDetails(!showCredentialDetails)
                    }
                    className="rounded-lg border p-2 hover:bg-muted"
                    title={showCredentialDetails ? "隐藏" : "显示"}
                  >
                    {showCredentialDetails ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={handleSelectNewFile}
                    className="rounded-lg border p-2 hover:bg-muted"
                    title="上传新文件"
                  >
                    <Upload className="h-4 w-4" />
                  </button>
                </div>
                {newCredFilePath && (
                  <div className="mt-2 text-xs text-green-600 dark:text-green-400 flex items-center gap-1">
                    <CheckCircle className="h-3 w-3" />
                    新文件已选择: {newCredFilePath.split("/").pop()}
                  </div>
                )}
              </div>
            )}

            {/* Gemini Project ID */}
            {credential.credential_type === "gemini_oauth" &&
              newCredFilePath && (
                <div>
                  <label className="mb-1 block text-sm font-medium">
                    项目ID（可选）
                  </label>
                  <input
                    type="text"
                    value={newProjectId}
                    onChange={(e) => setNewProjectId(e.target.value)}
                    placeholder="留空保持当前项目ID..."
                    className="w-full rounded-lg border bg-background px-3 py-2 text-sm"
                  />
                </div>
              )}

            {/* API Key 编辑 */}
            {isApiKey && (
              <>
                <div>
                  <label className="mb-1 block text-sm font-medium">
                    API Key
                  </label>
                  <div className="flex items-center gap-2">
                    <input
                      type={showApiKey ? "text" : "password"}
                      value={newApiKey}
                      onChange={(e) => setNewApiKey(e.target.value)}
                      placeholder="留空保持当前 Key，或输入新的 API Key..."
                      className="flex-1 rounded-lg border bg-background px-3 py-2 text-sm"
                    />
                    <button
                      type="button"
                      onClick={() => setShowApiKey(!showApiKey)}
                      className="rounded-lg border p-2 hover:bg-muted"
                      title={showApiKey ? "隐藏" : "显示"}
                    >
                      {showApiKey ? (
                        <EyeOff className="h-4 w-4" />
                      ) : (
                        <Eye className="h-4 w-4" />
                      )}
                    </button>
                  </div>
                  <p className="mt-1 text-xs text-muted-foreground">
                    当前: {credential.display_credential}
                  </p>
                </div>
                <div>
                  <label className="mb-1 block text-sm font-medium">
                    Base URL（可选）
                  </label>
                  <input
                    type="text"
                    value={newBaseUrl}
                    onChange={(e) => setNewBaseUrl(e.target.value)}
                    placeholder={
                      credential.credential_type === "openai_key"
                        ? "https://api.openai.com"
                        : "https://api.anthropic.com"
                    }
                    className="w-full rounded-lg border bg-background px-3 py-2 text-sm"
                  />
                  <p className="mt-1 text-xs text-muted-foreground">
                    留空使用默认 URL，或输入自定义代理地址（不要包含 /v1 后缀）
                  </p>
                </div>
              </>
            )}

            {/* 不支持的模型 - Checkbox Grid */}
            <div>
              <div className="flex items-center gap-2 mb-3">
                <Ban className="h-4 w-4 text-muted-foreground" />
                <label className="text-sm font-medium">不支持的模型</label>
                <span className="text-xs text-muted-foreground">
                  选择此提供商不支持的模型，系统会自动排除这些模型
                </span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                {currentProviderModels.map((model) => (
                  <label
                    key={model}
                    className={`flex items-center gap-2 rounded-lg border px-3 py-2 cursor-pointer transition-colors ${
                      notSupportedModels.includes(model)
                        ? "border-red-300 bg-red-50 dark:border-red-800 dark:bg-red-950/30"
                        : "border-border hover:bg-muted/50"
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={notSupportedModels.includes(model)}
                      onChange={() => toggleModelSupport(model)}
                      className="rounded border-gray-300"
                    />
                    <span className="text-sm truncate">{model}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* 高级选项：代理设置 */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Globe className="h-4 w-4 text-blue-500" />
                <label className="text-sm font-medium">凭证代理设置</label>
                <span className="text-xs text-muted-foreground">
                  （高级选项）
                </span>
              </div>
              <div className="rounded-lg border p-4 space-y-3">
                <div>
                  <label className="block text-sm font-medium mb-1.5">
                    代理 URL（可选）
                  </label>
                  <input
                    type="text"
                    value={proxyUrl}
                    onChange={(e) => handleProxyUrlChange(e.target.value)}
                    placeholder="例如: http://127.0.0.1:7890 或 socks5://127.0.0.1:1080"
                    className={`w-full rounded-lg border bg-background px-3 py-2 text-sm ${
                      proxyError ? "border-red-500" : ""
                    }`}
                  />
                  {proxyError ? (
                    <p className="text-xs text-red-500 mt-1">{proxyError}</p>
                  ) : (
                    <p className="text-xs text-muted-foreground mt-1">
                      留空则使用全局代理设置。支持 http://、https://、socks5://
                      协议
                    </p>
                  )}
                </div>
                <div className="rounded-lg bg-blue-50 dark:bg-blue-900/20 p-3 text-xs">
                  <p className="font-medium text-blue-700 dark:text-blue-300">
                    代理优先级说明：
                  </p>
                  <ul className="mt-1 list-inside list-disc text-blue-600 dark:text-blue-400">
                    <li>此凭证代理优先于全局代理</li>
                    <li>留空时使用全局代理设置</li>
                    <li>全局代理可在「设置 → 通用」中配置</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* 使用统计（只读） */}
            <div className="rounded-lg bg-muted/50 p-4">
              <label className="mb-3 block text-sm font-medium">使用统计</label>
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <span className="text-muted-foreground block text-xs">
                    使用次数
                  </span>
                  <span className="font-semibold">
                    {credential.usage_count}
                  </span>
                </div>
                <div>
                  <span className="text-muted-foreground block text-xs">
                    错误次数
                  </span>
                  <span className="font-semibold">
                    {credential.error_count}
                  </span>
                </div>
                <div>
                  <span className="text-muted-foreground block text-xs">
                    最后使用
                  </span>
                  <span className="text-xs">
                    {credential.last_used || "从未"}
                  </span>
                </div>
              </div>
            </div>

            {/* Error */}
            {error && (
              <div className="rounded-lg border border-red-500 bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950/30">
                {error}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="border-t px-6 py-4 flex justify-end gap-2 shrink-0">
          <button
            onClick={onClose}
            className="rounded-lg border px-4 py-2 text-sm hover:bg-muted"
          >
            取消
          </button>
          <button
            onClick={handleSubmit}
            disabled={loading || !!proxyError}
            className="rounded-lg bg-primary px-4 py-2 text-sm text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            {loading ? "保存中..." : "保存更改"}
          </button>
        </div>
      </div>
    </div>
  );
}
