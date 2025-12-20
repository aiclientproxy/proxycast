import { useState, useEffect } from "react";
import {
  Globe,
  Eye,
  EyeOff,
  CheckCircle2,
  AlertTriangle,
  Copy,
  Check,
} from "lucide-react";
import {
  getConfig,
  saveConfig,
  Config,
  RemoteManagementConfig,
} from "@/hooks/useTauri";

export function RemoteManagementSettings() {
  const [config, setConfig] = useState<Config | null>(null);
  const [saving, setSaving] = useState(false);
  const [showSecretKey, setShowSecretKey] = useState(false);
  const [copied, setCopied] = useState(false);
  const [message, setMessage] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);

  useEffect(() => {
    loadConfig();
  }, []);

  const loadConfig = async () => {
    try {
      const c = await getConfig();
      setConfig(c);
    } catch (e) {
      console.error(e);
    }
  };

  const handleSave = async () => {
    if (!config) return;
    setSaving(true);
    setMessage(null);
    try {
      await saveConfig(config);
      setMessage({ type: "success", text: "远程管理设置已保存" });
      setTimeout(() => setMessage(null), 3000);
    } catch (e: unknown) {
      const errorMessage = e instanceof Error ? e.message : String(e);
      setMessage({ type: "error", text: `保存失败: ${errorMessage}` });
    }
    setSaving(false);
  };

  const updateRemoteManagement = (updates: Partial<RemoteManagementConfig>) => {
    if (!config) return;
    setConfig({
      ...config,
      remote_management: { ...config.remote_management, ...updates },
    });
  };

  const copySecretKey = () => {
    if (config?.remote_management.secret_key) {
      navigator.clipboard.writeText(config.remote_management.secret_key);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const generateSecretKey = () => {
    const chars =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let key = "";
    for (let i = 0; i < 32; i++) {
      key += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    updateRemoteManagement({ secret_key: key });
  };

  if (!config) {
    return (
      <div className="flex items-center justify-center h-32">
        <div className="animate-spin h-6 w-6 border-2 border-primary border-t-transparent rounded-full" />
      </div>
    );
  }

  const rm = config.remote_management;
  const isEnabled = Boolean(rm.secret_key && rm.secret_key.length > 0);
  const remoteAccessSupported = false;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <Globe className="h-5 w-5 text-blue-500" />
        <div>
          <h3 className="text-sm font-medium">远程管理</h3>
          <p className="text-xs text-muted-foreground">
            配置远程管理 API 的访问控制
          </p>
        </div>
      </div>

      {/* 消息提示 */}
      {message && (
        <div
          className={`rounded-lg border p-3 text-sm flex items-center gap-2 ${
            message.type === "error"
              ? "border-destructive bg-destructive/10 text-destructive"
              : "border-green-500 bg-green-50 text-green-700 dark:bg-green-900/20 dark:text-green-400"
          }`}
        >
          {message.type === "success" ? (
            <CheckCircle2 className="h-4 w-4" />
          ) : (
            <AlertTriangle className="h-4 w-4" />
          )}
          {message.text}
        </div>
      )}

      <div className="p-4 rounded-lg border space-y-4">
        {!remoteAccessSupported && (
          <div className="flex items-start gap-2 rounded-lg bg-yellow-50 dark:bg-yellow-900/20 p-3 text-sm text-yellow-700 dark:text-yellow-400">
            <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
            <span>当前版本未启用 TLS，暂不支持远程管理访问，请保持关闭。</span>
          </div>
        )}

        {/* 管理密钥 */}
        <div>
          <label className="block text-sm font-medium mb-1.5">管理密钥</label>
          <div className="flex gap-2">
            <div className="relative flex-1">
              <input
                type={showSecretKey ? "text" : "password"}
                value={rm.secret_key || ""}
                onChange={(e) =>
                  updateRemoteManagement({ secret_key: e.target.value || null })
                }
                placeholder="留空则禁用管理 API"
                className="w-full px-3 py-2 pr-20 rounded-lg border bg-background text-sm font-mono focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none"
              />
              <div className="absolute right-2 top-1/2 flex -translate-y-1/2 gap-1">
                <button
                  type="button"
                  onClick={() => setShowSecretKey(!showSecretKey)}
                  className="p-1.5 rounded hover:bg-muted"
                  title={showSecretKey ? "隐藏" : "显示"}
                >
                  {showSecretKey ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </button>
                {rm.secret_key && (
                  <button
                    type="button"
                    onClick={copySecretKey}
                    className="p-1.5 rounded hover:bg-muted"
                    title="复制"
                  >
                    {copied ? (
                      <Check className="h-4 w-4 text-green-500" />
                    ) : (
                      <Copy className="h-4 w-4" />
                    )}
                  </button>
                )}
              </div>
            </div>
            <button
              type="button"
              onClick={generateSecretKey}
              className="px-3 py-2 rounded-lg border text-sm hover:bg-muted"
            >
              生成
            </button>
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            用于验证管理 API 请求，留空则禁用所有管理端点
          </p>
        </div>

        {/* 允许远程访问 */}
        <label
          className={`flex items-center justify-between p-3 rounded-lg border cursor-pointer hover:bg-muted/50 ${!isEnabled || !remoteAccessSupported ? "opacity-50 pointer-events-none" : ""}`}
        >
          <div>
            <span className="text-sm font-medium">允许远程访问</span>
            <p className="text-xs text-muted-foreground">
              允许非 localhost 地址访问管理 API
            </p>
          </div>
          <input
            type="checkbox"
            checked={rm.allow_remote}
            onChange={(e) =>
              updateRemoteManagement({ allow_remote: e.target.checked })
            }
            className="w-4 h-4 rounded border-gray-300"
            disabled={!isEnabled || !remoteAccessSupported}
          />
        </label>

        {/* 禁用控制面板 */}
        <label
          className={`flex items-center justify-between p-3 rounded-lg border cursor-pointer hover:bg-muted/50 ${!isEnabled ? "opacity-50 pointer-events-none" : ""}`}
        >
          <div>
            <span className="text-sm font-medium">禁用控制面板</span>
            <p className="text-xs text-muted-foreground">
              禁用 Web 控制面板，仅保留 API 访问
            </p>
          </div>
          <input
            type="checkbox"
            checked={rm.disable_control_panel}
            onChange={(e) =>
              updateRemoteManagement({
                disable_control_panel: e.target.checked,
              })
            }
            className="w-4 h-4 rounded border-gray-300"
            disabled={!isEnabled}
          />
        </label>

        {/* 警告提示 */}
        {remoteAccessSupported && isEnabled && rm.allow_remote && (
          <div className="flex items-start gap-2 rounded-lg bg-yellow-50 dark:bg-yellow-900/20 p-3 text-sm text-yellow-700 dark:text-yellow-400">
            <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
            <span>
              允许远程访问可能带来安全风险，请确保使用强密钥并在安全网络环境中使用
            </span>
          </div>
        )}

        <button
          onClick={handleSave}
          disabled={saving}
          className="w-full px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 disabled:opacity-50"
        >
          {saving ? "保存中..." : "保存远程管理设置"}
        </button>
      </div>
    </div>
  );
}
