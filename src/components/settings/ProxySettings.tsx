import { useState, useEffect } from "react";
import {
  Eye,
  EyeOff,
  Copy,
  Check,
  Shield,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Loader2,
  RefreshCw,
} from "lucide-react";
import {
  getConfig,
  saveConfig,
  Config,
  checkApiCompatibility,
  ApiCompatibilityResult,
} from "@/hooks/useTauri";

export function ProxySettings() {
  const [config, setConfig] = useState<Config | null>(null);
  const [showApiKey, setShowApiKey] = useState(false);
  const [copied, setCopied] = useState(false);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);

  // API Compatibility Check
  const [checking, setChecking] = useState(false);
  const [checkResult, setCheckResult] = useState<ApiCompatibilityResult | null>(
    null,
  );
  const [lastCheckTime, setLastCheckTime] = useState<Date | null>(null);

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
      setMessage({ type: "success", text: "设置已保存" });
      setTimeout(() => setMessage(null), 3000);
    } catch (e: unknown) {
      const errorMessage = e instanceof Error ? e.message : String(e);
      setMessage({ type: "error", text: `保存失败: ${errorMessage}` });
    }
    setSaving(false);
  };

  const copyApiKey = () => {
    if (config) {
      navigator.clipboard.writeText(config.server.api_key);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleCheckApiCompatibility = async (provider: string) => {
    setChecking(true);
    setCheckResult(null);
    try {
      const result = await checkApiCompatibility(provider);
      setCheckResult(result);
      setLastCheckTime(new Date());
    } catch (e) {
      setMessage({ type: "error", text: `API 检测失败: ${e}` });
      setTimeout(() => setMessage(null), 5000);
    }
    setChecking(false);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "healthy":
        return <CheckCircle2 className="h-5 w-5 text-green-500" />;
      case "partial":
        return <AlertTriangle className="h-5 w-5 text-yellow-500" />;
      case "error":
        return <XCircle className="h-5 w-5 text-red-500" />;
      default:
        return null;
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case "healthy":
        return "所有模型可用";
      case "partial":
        return "部分模型可用";
      case "error":
        return "API 不可用";
      default:
        return "未知";
    }
  };

  if (!config) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-2xl">
      {/* 消息提示 */}
      {message && (
        <div
          className={`rounded-lg border p-3 text-sm ${
            message.type === "error"
              ? "border-destructive bg-destructive/10 text-destructive"
              : "border-green-500 bg-green-50 text-green-700 dark:bg-green-900/20 dark:text-green-400"
          }`}
        >
          {message.text}
        </div>
      )}

      {/* 服务器配置 */}
      <div className="space-y-4">
        <div>
          <h3 className="text-sm font-medium">代理服务配置</h3>
          <p className="text-xs text-muted-foreground">
            配置本地代理服务器参数
          </p>
        </div>

        <div className="space-y-4 p-4 rounded-lg border">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1.5">
                监听地址
              </label>
              <input
                type="text"
                value={config.server.host}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    server: { ...config.server, host: e.target.value },
                  })
                }
                className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1.5">端口</label>
              <input
                type="number"
                value={config.server.port}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    server: {
                      ...config.server,
                      port: parseInt(e.target.value) || 8999,
                    },
                  })
                }
                className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1.5">API Key</label>
            <div className="relative">
              <input
                type={showApiKey ? "text" : "password"}
                value={config.server.api_key}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    server: { ...config.server, api_key: e.target.value },
                  })
                }
                className="w-full px-3 py-2 pr-20 rounded-lg border bg-background text-sm font-mono focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none"
              />
              <div className="absolute right-2 top-1/2 flex -translate-y-1/2 gap-1">
                <button
                  type="button"
                  onClick={() => setShowApiKey(!showApiKey)}
                  className="p-1.5 rounded hover:bg-muted"
                  title={showApiKey ? "隐藏" : "显示"}
                >
                  {showApiKey ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </button>
                <button
                  type="button"
                  onClick={copyApiKey}
                  className="p-1.5 rounded hover:bg-muted"
                  title="复制"
                >
                  {copied ? (
                    <Check className="h-4 w-4 text-green-500" />
                  ) : (
                    <Copy className="h-4 w-4" />
                  )}
                </button>
              </div>
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              用于验证 API 请求的密钥
            </p>
          </div>

          <button
            onClick={handleSave}
            disabled={saving}
            className="w-full px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 disabled:opacity-50"
          >
            {saving ? "保存中..." : "保存设置"}
          </button>
        </div>
      </div>

      {/* Claude Code 兼容性检测 */}
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Shield className="h-5 w-5 text-purple-500" />
          <div>
            <h3 className="text-sm font-medium">Claude Code 兼容性检测</h3>
            <p className="text-xs text-muted-foreground">
              检测 API 是否支持 Claude Code 所需的功能
            </p>
          </div>
        </div>

        <div className="p-4 rounded-lg border space-y-4">
          <div className="rounded-lg bg-purple-50 dark:bg-purple-900/20 p-3 text-sm">
            <p className="font-medium text-purple-700 dark:text-purple-300">
              检测项目：
            </p>
            <ul className="mt-1 list-inside list-disc text-purple-600 dark:text-purple-400 text-xs">
              <li>基础对话能力 (basic)</li>
              <li>Tool Calls 支持 (tool_call) - Claude Code 核心功能</li>
            </ul>
          </div>

          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => handleCheckApiCompatibility("kiro")}
              disabled={checking}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-purple-600 text-white text-sm font-medium hover:bg-purple-700 disabled:opacity-50"
            >
              {checking ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Shield className="h-4 w-4" />
              )}
              检测 Kiro
            </button>
            <button
              onClick={() => handleCheckApiCompatibility("gemini")}
              disabled={checking}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
            >
              {checking ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Shield className="h-4 w-4" />
              )}
              检测 Gemini
            </button>
            <button
              onClick={() => handleCheckApiCompatibility("qwen")}
              disabled={checking}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-green-600 text-white text-sm font-medium hover:bg-green-700 disabled:opacity-50"
            >
              {checking ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Shield className="h-4 w-4" />
              )}
              检测 Qwen
            </button>
          </div>

          {lastCheckTime && (
            <p className="text-xs text-muted-foreground">
              最后检测时间: {lastCheckTime.toLocaleString()}
            </p>
          )}

          {checkResult && (
            <div className="space-y-3 rounded-lg border p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {getStatusIcon(checkResult.overall_status)}
                  <span className="font-medium">
                    {checkResult.provider.toUpperCase()} -{" "}
                    {getStatusText(checkResult.overall_status)}
                  </span>
                </div>
                <span className="text-xs text-muted-foreground">
                  {new Date(checkResult.checked_at).toLocaleString()}
                </span>
              </div>

              <div className="space-y-2">
                <p className="text-sm font-medium">检测结果:</p>
                {checkResult.results.map((r) => (
                  <div
                    key={r.model}
                    className={`flex items-center justify-between rounded p-2 text-sm ${
                      r.available
                        ? "bg-green-50 dark:bg-green-900/20"
                        : "bg-red-50 dark:bg-red-900/20"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      {r.available ? (
                        <CheckCircle2 className="h-4 w-4 text-green-500" />
                      ) : (
                        <XCircle className="h-4 w-4 text-red-500" />
                      )}
                      <span
                        className={
                          r.model.includes("tool_call")
                            ? "font-medium text-purple-600"
                            : ""
                        }
                      >
                        {r.model}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      {r.status > 0 && <span>HTTP {r.status}</span>}
                      <span>{r.time_ms}ms</span>
                      {r.error_type && (
                        <span className="rounded bg-red-100 dark:bg-red-900/50 px-1 text-red-600 dark:text-red-400">
                          {r.error_type}
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              {checkResult.warnings.length > 0 && (
                <div className="space-y-1">
                  <p className="text-sm font-medium text-yellow-600">警告:</p>
                  {checkResult.warnings.map((w, i) => (
                    <div
                      key={i}
                      className="flex items-start gap-2 rounded bg-yellow-50 dark:bg-yellow-900/20 p-2 text-sm text-yellow-700 dark:text-yellow-400"
                    >
                      <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
                      <span>{w}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
