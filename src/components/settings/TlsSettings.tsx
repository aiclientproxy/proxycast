import { useState, useEffect } from "react";
import { Shield, FolderOpen, AlertTriangle, CheckCircle2 } from "lucide-react";
import { open } from "@tauri-apps/plugin-dialog";
import { getConfig, saveConfig, Config, TlsConfig } from "@/hooks/useTauri";

export function TlsSettings() {
  const [config, setConfig] = useState<Config | null>(null);
  const [saving, setSaving] = useState(false);
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
      setMessage({
        type: "success",
        text: "TLS 设置已保存，需要重启服务器生效",
      });
      setTimeout(() => setMessage(null), 5000);
    } catch (e: unknown) {
      const errorMessage = e instanceof Error ? e.message : String(e);
      setMessage({ type: "error", text: `保存失败: ${errorMessage}` });
    }
    setSaving(false);
  };

  const updateTls = (updates: Partial<TlsConfig>) => {
    if (!config) return;
    setConfig({
      ...config,
      server: {
        ...config.server,
        tls: { ...config.server.tls, ...updates },
      },
    });
  };

  const handleSelectCert = async () => {
    try {
      const selected = await open({
        multiple: false,
        filters: [{ name: "Certificate", extensions: ["pem", "crt", "cer"] }],
      });
      if (selected) {
        updateTls({ cert_path: selected as string });
      }
    } catch (e) {
      console.error("Failed to open file dialog:", e);
    }
  };

  const handleSelectKey = async () => {
    try {
      const selected = await open({
        multiple: false,
        filters: [{ name: "Private Key", extensions: ["pem", "key"] }],
      });
      if (selected) {
        updateTls({ key_path: selected as string });
      }
    } catch (e) {
      console.error("Failed to open file dialog:", e);
    }
  };

  if (!config) {
    return (
      <div className="flex items-center justify-center h-32">
        <div className="animate-spin h-6 w-6 border-2 border-primary border-t-transparent rounded-full" />
      </div>
    );
  }

  const tlsSupported = false;
  const tls = config.server.tls;
  const isConfigValid = !tls.enable || (tls.cert_path && tls.key_path);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <Shield className="h-5 w-5 text-green-500" />
        <div>
          <h3 className="text-sm font-medium">TLS/HTTPS 配置</h3>
          <p className="text-xs text-muted-foreground">启用 HTTPS 加密通信</p>
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
        {!tlsSupported && (
          <div className="flex items-start gap-2 rounded-lg bg-yellow-50 dark:bg-yellow-900/20 p-3 text-sm text-yellow-700 dark:text-yellow-400">
            <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
            <span>
              当前版本暂不支持 TLS。启用后服务将无法启动，请使用反向代理或 TLS
              终止。
            </span>
          </div>
        )}

        {/* 启用开关 */}
        <label className="flex items-center justify-between p-3 rounded-lg border cursor-pointer hover:bg-muted/50">
          <div>
            <span className="text-sm font-medium">启用 TLS</span>
            <p className="text-xs text-muted-foreground">
              使用 HTTPS 协议提供服务
            </p>
          </div>
          <input
            type="checkbox"
            checked={tls.enable}
            onChange={(e) => updateTls({ enable: e.target.checked })}
            disabled={!tlsSupported}
            className="w-4 h-4 rounded border-gray-300"
          />
        </label>

        {/* 证书路径 */}
        <div
          className={
            tls.enable && tlsSupported ? "" : "opacity-50 pointer-events-none"
          }
        >
          <label className="block text-sm font-medium mb-1.5">
            证书文件路径 {tls.enable && <span className="text-red-500">*</span>}
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={tls.cert_path || ""}
              onChange={(e) => updateTls({ cert_path: e.target.value || null })}
              placeholder="/path/to/cert.pem"
              className="flex-1 px-3 py-2 rounded-lg border bg-background text-sm focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none"
            />
            <button
              type="button"
              onClick={handleSelectCert}
              className="flex items-center gap-1 rounded-lg border px-3 py-2 text-sm hover:bg-muted"
            >
              <FolderOpen className="h-4 w-4" />
              浏览
            </button>
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            PEM 格式的 SSL/TLS 证书文件
          </p>
        </div>

        {/* 私钥路径 */}
        <div
          className={
            tls.enable && tlsSupported ? "" : "opacity-50 pointer-events-none"
          }
        >
          <label className="block text-sm font-medium mb-1.5">
            私钥文件路径 {tls.enable && <span className="text-red-500">*</span>}
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={tls.key_path || ""}
              onChange={(e) => updateTls({ key_path: e.target.value || null })}
              placeholder="/path/to/key.pem"
              className="flex-1 px-3 py-2 rounded-lg border bg-background text-sm focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none"
            />
            <button
              type="button"
              onClick={handleSelectKey}
              className="flex items-center gap-1 rounded-lg border px-3 py-2 text-sm hover:bg-muted"
            >
              <FolderOpen className="h-4 w-4" />
              浏览
            </button>
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            PEM 格式的私钥文件
          </p>
        </div>

        {/* 警告提示 */}
        {tls.enable && !isConfigValid && (
          <div className="flex items-start gap-2 rounded-lg bg-yellow-50 dark:bg-yellow-900/20 p-3 text-sm text-yellow-700 dark:text-yellow-400">
            <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
            <span>启用 TLS 需要同时配置证书和私钥文件路径</span>
          </div>
        )}

        <button
          onClick={handleSave}
          disabled={saving || !tlsSupported || (tls.enable && !isConfigValid)}
          className="w-full px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 disabled:opacity-50"
        >
          {saving ? "保存中..." : "保存 TLS 设置"}
        </button>
      </div>
    </div>
  );
}
