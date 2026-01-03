import { useState, useEffect } from "react";
import {
  Play,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  RefreshCw,
} from "lucide-react";
import { LogsTab } from "./LogsTab";
import { RoutesTab } from "./RoutesTab";
import { ProviderIcon } from "@/icons/providers";
import {
  startServer,
  stopServer,
  getServerStatus,
  getConfig,
  saveConfig,
  reloadCredentials,
  testApi,
  ServerStatus,
  Config,
  TestResult,
  getDefaultProvider,
  setDefaultProvider,
  getNetworkInfo,
  NetworkInfo,
} from "@/hooks/useTauri";
import { providerPoolApi, ProviderPoolOverview } from "@/lib/api/providerPool";

interface TestState {
  endpoint: string;
  status: "idle" | "loading" | "success" | "error";
  response?: string;
  time?: number;
  httpStatus?: number;
}

type TabId = "server" | "routes" | "logs";

export function ApiServerPage() {
  const [status, setStatus] = useState<ServerStatus | null>(null);
  const [config, setConfig] = useState<Config | null>(null);
  const [loading, setLoading] = useState(false);
  const [_error, setError] = useState<string | null>(null);
  const [testResults, setTestResults] = useState<Record<string, TestState>>({});
  const [copiedCmd, setCopiedCmd] = useState<string | null>(null);
  const [expandedTest, setExpandedTest] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>("server");

  // Config editing
  const [editPort, setEditPort] = useState<string>("");
  const [editApiKey, setEditApiKey] = useState<string>("");
  const [defaultProvider, setDefaultProviderState] = useState<string>("kiro");

  const [message, setMessage] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);

  // 网络信息
  const [networkInfo, setNetworkInfo] = useState<NetworkInfo | null>(null);

  // 自动清除消息
  useEffect(() => {
    if (message) {
      const timer = setTimeout(() => {
        setMessage(null);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [message]);

  const fetchStatus = async () => {
    try {
      const s = await getServerStatus();
      setStatus(s);
    } catch (e) {
      console.error(e);
    }
  };

  const fetchConfig = async () => {
    try {
      const c = await getConfig();
      setConfig(c);
      setEditPort(c.server.port.toString());
      setEditApiKey(c.server.api_key);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchStatus();
    fetchConfig();
    loadDefaultProvider();
    loadNetworkInfo();

    const statusInterval = setInterval(fetchStatus, 3000);
    return () => clearInterval(statusInterval);
  }, []);

  const loadNetworkInfo = async () => {
    try {
      const info = await getNetworkInfo();
      setNetworkInfo(info);
    } catch (e) {
      console.error("Failed to get network info:", e);
    }
  };

  const loadDefaultProvider = async () => {
    try {
      const dp = await getDefaultProvider();
      setDefaultProviderState(dp);
    } catch (e) {
      console.error("Failed to get default provider:", e);
    }
  };

  const handleStart = async () => {
    setLoading(true);
    setError(null);
    try {
      await reloadCredentials();
      await startServer();
      await fetchStatus();
      setMessage({ type: "success", text: "服务已启动" });
    } catch (e: unknown) {
      const errMsg = e instanceof Error ? e.message : String(e);
      setError(errMsg);
      setMessage({ type: "error", text: `启动失败: ${errMsg}` });
    }
    setLoading(false);
  };

  const handleStop = async () => {
    setLoading(true);
    try {
      await stopServer();
      await fetchStatus();
      setMessage({ type: "success", text: "服务已停止" });
    } catch (e: unknown) {
      const errMsg = e instanceof Error ? e.message : String(e);
      setError(errMsg);
      setMessage({ type: "error", text: `停止失败: ${errMsg}` });
    }
    setLoading(false);
  };

  const handleSaveServerConfig = async () => {
    if (!config) return;
    setLoading(true);
    try {
      const newConfig = {
        ...config,
        server: {
          ...config.server,
          port: parseInt(editPort) || 8999,
          api_key: editApiKey,
        },
      };
      await saveConfig(newConfig);
      await fetchConfig();
      setMessage({ type: "success", text: "服务器配置已保存" });
    } catch (e: unknown) {
      const errMsg = e instanceof Error ? e.message : String(e);
      setMessage({ type: "error", text: `保存失败: ${errMsg}` });
    }
    setLoading(false);
  };

  const providerLabels: Record<string, string> = {
    kiro: "Kiro (AWS)",
    gemini: "Gemini (Google)",
    qwen: "Qwen (阿里)",
    antigravity: "Antigravity (Gemini 3 Pro)",
    openai: "OpenAI",
    claude: "Claude (Anthropic)",
  };

  const [poolOverview, setPoolOverview] = useState<ProviderPoolOverview[]>([]);
  const [providerSwitchMsg, setProviderSwitchMsg] = useState<string | null>(
    null,
  );

  // 加载凭证池概览
  const loadPoolOverview = async () => {
    try {
      const overview = await providerPoolApi.getOverview();
      setPoolOverview(overview);
    } catch (e) {
      console.error("Failed to load pool overview:", e);
    }
  };

  useEffect(() => {
    loadPoolOverview();
    // 定时刷新凭证池数据，以便使用次数能够更新
    const poolInterval = setInterval(loadPoolOverview, 5000);
    return () => clearInterval(poolInterval);
  }, []);

  // 自动清除 provider 切换提示
  useEffect(() => {
    if (providerSwitchMsg) {
      const timer = setTimeout(() => setProviderSwitchMsg(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [providerSwitchMsg]);

  const handleSetDefaultProvider = async (providerId: string) => {
    try {
      await setDefaultProvider(providerId);
      setDefaultProviderState(providerId);

      // 获取最新的凭证池数据
      const freshOverview = await providerPoolApi.getOverview();
      setPoolOverview(freshOverview);

      // 获取该类型下的凭证数量
      const typeOverview = freshOverview.find(
        (o) => o.provider_type === providerId,
      );
      const credCount = typeOverview?.stats.total || 0;
      const healthyCount = typeOverview?.stats.healthy || 0;
      const label = providerLabels[providerId] || providerId;

      setProviderSwitchMsg(
        `已切换到 ${label}` +
          (credCount > 0 ? `（${healthyCount}/${credCount} 可用）` : ""),
      );
    } catch (e: unknown) {
      const errMsg = e instanceof Error ? e.message : String(e);
      setProviderSwitchMsg(`切换失败: ${errMsg}`);
    }
  };

  const serverUrl = status
    ? `http://${status.host}:${status.port}`
    : `http://localhost:${config?.server.port ?? 8999}`;
  const apiKey = config?.server.api_key ?? "";

  // 根据 Provider 类型获取测试模型
  const getTestModel = (provider: string): string => {
    switch (provider) {
      case "antigravity":
        return "gemini-3-pro-preview";
      case "gemini":
        return "gemini-2.0-flash";
      case "qwen":
        return "qwen-max";
      case "openai":
        return "gpt-4o";
      case "claude":
        return "claude-sonnet-4-20250514";
      case "kiro":
      default:
        return "claude-opus-4-5-20251101";
    }
  };

  const testModel = getTestModel(defaultProvider);

  // 根据 Provider 类型获取 Gemini 测试模型列表
  const getGeminiTestModels = (provider: string): string[] => {
    switch (provider) {
      case "antigravity":
        return [
          "gemini-3-pro-preview",
          "gemini-3-pro-image-preview",
          "gemini-3-flash-preview",
          "gemini-claude-sonnet-4-5",
        ];
      case "gemini":
        return ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"];
      default:
        return ["gemini-2.0-flash"];
    }
  };

  const geminiTestModels = getGeminiTestModels(defaultProvider);

  // 是否显示 Gemini 测试端点
  const showGeminiTest =
    defaultProvider === "antigravity" || defaultProvider === "gemini";

  // Test endpoints
  const testEndpoints = [
    {
      id: "health",
      name: "健康检查",
      method: "GET",
      path: "/health",
      needsAuth: false,
      body: null,
    },
    {
      id: "models",
      name: "模型列表",
      method: "GET",
      path: "/v1/models",
      needsAuth: true,
      body: null,
    },
    {
      id: "chat",
      name: "OpenAI Chat",
      method: "POST",
      path: "/v1/chat/completions",
      needsAuth: true,
      body: JSON.stringify({
        model: testModel,
        messages: [{ role: "user", content: "Say hi in one word" }],
      }),
    },
    {
      id: "anthropic",
      name: "Anthropic Messages",
      method: "POST",
      path: "/v1/messages",
      needsAuth: true,
      body: JSON.stringify({
        model: testModel,
        max_tokens: 100,
        messages: [
          {
            role: "user",
            content: "What is 1+1? Answer with just the number.",
          },
        ],
      }),
    },
    // Gemini 原生协议测试（仅在 Antigravity 或 Gemini Provider 时显示）
    ...(showGeminiTest
      ? geminiTestModels.map((model, index) => ({
          id: `gemini-${index}`,
          name: `Gemini ${model}`,
          method: "POST",
          path: `/v1/gemini/${model}:generateContent`,
          needsAuth: true,
          body: JSON.stringify({
            contents: [
              {
                role: "user",
                parts: [{ text: "What is 2+2? Answer with just the number." }],
              },
            ],
            generationConfig: {
              maxOutputTokens: 100,
            },
          }),
        }))
      : []),
  ];

  const runTest = async (endpoint: (typeof testEndpoints)[0]) => {
    setTestResults((prev) => ({
      ...prev,
      [endpoint.id]: { endpoint: endpoint.path, status: "loading" },
    }));

    try {
      const result: TestResult = await testApi(
        endpoint.method,
        endpoint.path,
        endpoint.body,
        endpoint.needsAuth,
      );

      setTestResults((prev) => ({
        ...prev,
        [endpoint.id]: {
          endpoint: endpoint.path,
          status: result.success ? "success" : "error",
          response: result.body || `HTTP ${result.status}: 无响应内容`,
          time: result.time_ms,
          httpStatus: result.status,
        },
      }));

      // 测试成功后立即刷新凭证池数据，更新使用次数
      if (result.success) {
        await loadPoolOverview();
      }
    } catch (e: unknown) {
      const errMsg = e instanceof Error ? e.message : String(e);
      setTestResults((prev) => ({
        ...prev,
        [endpoint.id]: {
          endpoint: endpoint.path,
          status: "error",
          response: `请求失败: ${errMsg}`,
        },
      }));
    }
  };

  const runAllTests = async () => {
    for (const endpoint of testEndpoints) {
      await runTest(endpoint);
    }
  };

  const getCurlCommand = (endpoint: (typeof testEndpoints)[0]) => {
    let cmd = `curl -s ${serverUrl}${endpoint.path}`;
    if (endpoint.needsAuth) {
      cmd += ` \\\n  -H "Authorization: Bearer ${apiKey}"`;
    }
    if (endpoint.body) {
      cmd += ` \\\n  -H "Content-Type: application/json"`;
      cmd += ` \\\n  -d '${endpoint.body}'`;
    }
    return cmd;
  };

  const copyCommand = (id: string, cmd: string) => {
    navigator.clipboard.writeText(cmd);
    setCopiedCmd(id);
    setTimeout(() => setCopiedCmd(null), 2000);
  };

  const getStatusBadge = (result?: TestState) => {
    if (!result || result.status === "idle") {
      return <span className="text-xs text-gray-400">未测试</span>;
    }
    if (result.status === "loading") {
      return <span className="text-xs text-blue-500">测试中...</span>;
    }
    if (result.status === "success") {
      return <span className="text-xs text-green-600">{result.time}ms</span>;
    }
    return (
      <span className="text-xs text-red-500">
        失败 {result.httpStatus ? `(${result.httpStatus})` : ""}
      </span>
    );
  };

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <h2 className="text-2xl font-bold">API Server</h2>
          <p className="text-muted-foreground text-sm">
            本地代理服务器，支持 OpenAI/Anthropic 格式
            {networkInfo && (
              <>
                {" "}
                <code className="px-1 py-0.5 rounded bg-muted text-xs">
                  {networkInfo.localhost}:{config?.server.port ?? 8999}
                </code>
                {networkInfo.lan_ip && (
                  <>
                    {" | "}
                    <code className="px-1 py-0.5 rounded bg-muted text-xs">
                      {networkInfo.lan_ip}:{config?.server.port ?? 8999}
                    </code>
                    <span className="text-xs"> (局域网)</span>
                  </>
                )}
              </>
            )}
            <span className="ml-2">
              <span
                className={`inline-block h-2 w-2 rounded-full ${status?.running ? "bg-green-500" : "bg-red-500"}`}
              />{" "}
              {status?.running ? "运行中" : "已停止"}
              {" · "}
              {status?.requests || 0} 请求
              {" · "}
              <span className="capitalize">{defaultProvider}</span>
            </span>
          </p>
        </div>
      </div>

      {message && (
        <div
          className={`flex items-center gap-2 rounded-lg border p-3 text-sm ${
            message.type === "success"
              ? "border-green-500 bg-green-50 text-green-700 dark:bg-green-950/30"
              : "border-red-500 bg-red-50 text-red-700 dark:bg-red-950/30"
          }`}
        >
          {message.type === "success" ? (
            <Check className="h-4 w-4" />
          ) : (
            <RefreshCw className="h-4 w-4" />
          )}
          {message.text}
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-2 border-b overflow-x-auto">
        {[
          { id: "server" as TabId, name: "服务器控制" },
          { id: "routes" as TabId, name: "路由端点" },
          { id: "logs" as TabId, name: "系统日志" },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px whitespace-nowrap ${
              activeTab === tab.id
                ? "border-primary text-primary"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            {tab.name}
          </button>
        ))}
      </div>

      {/* Server Control Tab */}
      {activeTab === "server" && (
        <div className="space-y-4">
          {/* Server Control - 紧凑版 */}
          <div className="rounded-lg border bg-card p-4">
            <div className="flex items-center gap-4">
              <button
                className={`rounded-lg px-4 py-1.5 text-sm font-medium text-white disabled:opacity-50 ${
                  status?.running
                    ? "bg-red-600 hover:bg-red-700"
                    : "bg-green-600 hover:bg-green-700"
                }`}
                onClick={status?.running ? handleStop : handleStart}
                disabled={loading}
              >
                {loading
                  ? "处理中..."
                  : status?.running
                    ? "停止服务"
                    : "启动服务"}
              </button>
              <div className="flex items-center gap-3 text-sm">
                <div className="flex items-center gap-2">
                  <span className="text-muted-foreground">端口:</span>
                  <input
                    type="number"
                    value={editPort}
                    onChange={(e) => setEditPort(e.target.value)}
                    className="w-20 rounded border bg-background px-2 py-1 text-sm"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-muted-foreground">API Key:</span>
                  <input
                    type="text"
                    value={editApiKey}
                    onChange={(e) => setEditApiKey(e.target.value)}
                    className="w-40 rounded border bg-background px-2 py-1 text-sm"
                  />
                </div>
                <button
                  onClick={handleSaveServerConfig}
                  disabled={loading}
                  className="rounded border px-3 py-1 text-sm hover:bg-muted disabled:opacity-50"
                >
                  保存
                </button>
              </div>
            </div>
          </div>

          {/* Default Provider - 紧凑版 */}
          <div className="rounded-lg border bg-card p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="font-medium text-sm">默认 Provider</span>
              {providerSwitchMsg && (
                <span className="text-xs text-green-600 flex items-center gap-1">
                  <Check className="h-3 w-3" />
                  {providerSwitchMsg}
                </span>
              )}
            </div>
            <div className="flex flex-wrap gap-2">
              {(
                [
                  { id: "kiro", label: "Kiro", iconType: "kiro" },
                  { id: "gemini", label: "Gemini", iconType: "gemini" },
                  { id: "qwen", label: "Qwen", iconType: "qwen" },
                  {
                    id: "antigravity",
                    label: "Antigravity",
                    iconType: "gemini",
                  },
                  { id: "openai", label: "OpenAI", iconType: "openai" },
                  { id: "claude", label: "Claude", iconType: "claude" },
                ] as const
              ).map((p) => {
                const overview = poolOverview.find(
                  (o) => o.provider_type === p.id,
                );
                const count = overview?.stats.total || 0;
                return (
                  <button
                    key={p.id}
                    onClick={() => handleSetDefaultProvider(p.id)}
                    disabled={loading}
                    className={`flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-sm transition-colors ${
                      defaultProvider === p.id
                        ? "border-primary bg-primary/10 text-primary"
                        : "border-border bg-card hover:bg-muted text-muted-foreground hover:text-foreground"
                    } disabled:opacity-50`}
                  >
                    <ProviderIcon providerType={p.iconType} size={14} />
                    {p.label}
                    {count > 0 && (
                      <span className="text-xs opacity-70">({count})</span>
                    )}
                  </button>
                );
              })}
            </div>

            {/* 当前选中类型的凭证列表 */}
            {(() => {
              const currentOverview = poolOverview.find(
                (o) => o.provider_type === defaultProvider,
              );
              const allCredentials = currentOverview?.credentials || [];
              // 过滤掉禁用的凭证，只显示启用的凭证
              const credentials = allCredentials.filter(
                (cred) => !cred.is_disabled,
              );
              if (credentials.length === 0) {
                return (
                  <div className="mt-4 rounded-lg border border-dashed p-4 text-center text-sm text-muted-foreground">
                    当前类型无可用凭证，请先在凭证池中添加
                  </div>
                );
              }
              return (
                <div className="mt-4 space-y-2">
                  <p className="text-xs text-muted-foreground">
                    当前可用凭证 ({credentials.length}):
                  </p>
                  <div className="space-y-1">
                    {credentials.map((cred) => (
                      <div
                        key={cred.uuid}
                        className="flex items-center justify-between rounded-lg border border-border bg-muted/30 px-3 py-2 text-sm"
                      >
                        <div className="flex items-center gap-2">
                          <span
                            className={`h-2 w-2 rounded-full ${
                              cred.is_healthy ? "bg-green-500" : "bg-yellow-500"
                            }`}
                          />
                          <span>{cred.name || cred.uuid.slice(0, 8)}</span>
                        </div>
                        <span className="text-xs text-muted-foreground">
                          使用 {cred.usage_count} 次
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })()}
          </div>

          {/* API Testing */}
          <div className="rounded-lg border bg-card p-6">
            <div className="mb-4 flex items-center justify-between">
              <h3 className="font-semibold">API 测试</h3>
              <button
                onClick={runAllTests}
                disabled={!status?.running}
                className="flex items-center gap-2 rounded-lg bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
              >
                <Play className="h-4 w-4" />
                测试全部
              </button>
            </div>

            <div className="space-y-3">
              {testEndpoints.map((endpoint) => {
                const result = testResults[endpoint.id];
                const isExpanded = expandedTest === endpoint.id;
                const curlCmd = getCurlCommand(endpoint);

                return (
                  <div
                    key={endpoint.id}
                    className="rounded-lg border bg-background"
                  >
                    <div className="flex items-center justify-between p-3">
                      <div className="flex items-center gap-3">
                        <span
                          className={`rounded px-2 py-0.5 text-xs font-medium ${
                            endpoint.method === "GET"
                              ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"
                              : "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400"
                          }`}
                        >
                          {endpoint.method}
                        </span>
                        <span className="font-medium">{endpoint.name}</span>
                        <code className="text-xs text-muted-foreground">
                          {endpoint.path}
                        </code>
                        {getStatusBadge(result)}
                      </div>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => copyCommand(endpoint.id, curlCmd)}
                          className="rounded p-1.5 hover:bg-muted"
                          title="复制 curl 命令"
                        >
                          {copiedCmd === endpoint.id ? (
                            <Check className="h-4 w-4 text-green-500" />
                          ) : (
                            <Copy className="h-4 w-4" />
                          )}
                        </button>
                        <button
                          onClick={() => runTest(endpoint)}
                          disabled={
                            !status?.running || result?.status === "loading"
                          }
                          className="rounded bg-primary/10 px-2 py-1 text-xs font-medium text-primary hover:bg-primary/20 disabled:opacity-50"
                        >
                          测试
                        </button>
                        <button
                          onClick={() =>
                            setExpandedTest(isExpanded ? null : endpoint.id)
                          }
                          className="rounded p-1.5 hover:bg-muted"
                        >
                          {isExpanded ? (
                            <ChevronUp className="h-4 w-4" />
                          ) : (
                            <ChevronDown className="h-4 w-4" />
                          )}
                        </button>
                      </div>
                    </div>

                    {isExpanded && (
                      <div className="border-t p-3 space-y-3">
                        <div>
                          <p className="mb-1 text-xs font-medium text-muted-foreground">
                            curl 命令
                          </p>
                          <pre className="rounded bg-muted p-2 text-xs overflow-x-auto">
                            {curlCmd}
                          </pre>
                        </div>
                        {result?.response && (
                          <div>
                            <p className="mb-1 text-xs font-medium text-muted-foreground">
                              响应{" "}
                              {result.httpStatus &&
                                `(HTTP ${result.httpStatus})`}
                            </p>
                            <pre
                              className={`rounded p-2 text-xs overflow-x-auto max-h-40 ${
                                result.status === "success"
                                  ? "bg-green-50 dark:bg-green-950/30"
                                  : "bg-red-50 dark:bg-red-950/30"
                              }`}
                            >
                              {(() => {
                                try {
                                  return JSON.stringify(
                                    JSON.parse(result.response),
                                    null,
                                    2,
                                  );
                                } catch {
                                  return result.response || "(空响应)";
                                }
                              })()}
                            </pre>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Routes Tab */}
      {activeTab === "routes" && <RoutesTab />}

      {/* Logs Tab */}
      {activeTab === "logs" && <LogsTab />}
    </div>
  );
}
