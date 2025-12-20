import React, { useState, useMemo, useCallback } from "react";
import { X, ExternalLink, Wand2, Eye, EyeOff } from "lucide-react";
import { Provider, AppType } from "@/lib/api/switch";
import { getConfig } from "@/hooks/useTauri";
import { cn } from "@/lib/utils";
import { ProviderIcon } from "@/icons/providers";

interface ProviderFormProps {
  appType: AppType;
  provider: Provider | null;
  onSave: (
    data: Omit<Provider, "id" | "is_current" | "created_at">,
  ) => Promise<void>;
  onCancel: () => void;
}

// 供应商分类
type ProviderCategory =
  | "official"
  | "cn_official"
  | "aggregator"
  | "third_party"
  | "proxy"
  | "custom";

// 预设供应商接口
interface ProviderPreset {
  id: string;
  name: string;
  category: ProviderCategory;
  iconColor?: string;
  websiteUrl?: string;
  apiKeyUrl?: string;
  // Claude 专属默认配置
  defaultBaseUrl?: string;
  defaultModel?: string;
  // Codex 专属默认配置
  defaultCodexAuth?: Record<string, unknown>;
  defaultCodexConfig?: string;
  // Gemini 专属默认配置
  defaultGeminiEnv?: Record<string, string>;
}

// 预设供应商配置
const presets: Record<AppType, ProviderPreset[]> = {
  claude: [
    // 官方
    {
      id: "anthropic",
      name: "Anthropic 官方",
      category: "official",
      iconColor: "#D97757",
      websiteUrl: "https://www.anthropic.com/claude-code",
      apiKeyUrl: "https://console.anthropic.com/settings/keys",
    },
    // 国内官方
    {
      id: "deepseek",
      name: "DeepSeek",
      category: "cn_official",
      iconColor: "#0066FF",
      websiteUrl: "https://platform.deepseek.com",
      apiKeyUrl: "https://platform.deepseek.com/api_keys",
      defaultBaseUrl: "https://api.deepseek.com/anthropic",
    },
    {
      id: "zhipu",
      name: "智谱 GLM",
      category: "cn_official",
      iconColor: "#5B6BE8",
      websiteUrl: "https://open.bigmodel.cn",
      apiKeyUrl: "https://open.bigmodel.cn/usercenter/apikeys",
      defaultBaseUrl: "https://open.bigmodel.cn/api/anthropic",
    },
    {
      id: "qwen",
      name: "通义千问 Coder",
      category: "cn_official",
      iconColor: "#6B4FBB",
      websiteUrl: "https://bailian.console.aliyun.com",
      apiKeyUrl: "https://bailian.console.aliyun.com/?apiKey=1#/api-key",
      defaultBaseUrl:
        "https://dashscope.aliyuncs.com/api/v2/apps/claude-code-proxy",
    },
    {
      id: "kimi",
      name: "Kimi k2",
      category: "cn_official",
      iconColor: "#000000",
      websiteUrl: "https://platform.moonshot.cn/console",
      apiKeyUrl: "https://platform.moonshot.cn/console/api-keys",
      defaultBaseUrl: "https://api.moonshot.cn/anthropic",
    },
    {
      id: "kimi-coding",
      name: "Kimi For Coding",
      category: "cn_official",
      iconColor: "#000000",
      websiteUrl: "https://www.kimi.com/coding/docs/",
      apiKeyUrl: "https://www.kimi.com/coding/profile",
      defaultBaseUrl: "https://api.kimi.com/coding/",
    },
    {
      id: "minimax",
      name: "MiniMax",
      category: "cn_official",
      iconColor: "#F97316",
      websiteUrl: "https://platform.minimaxi.com",
      apiKeyUrl:
        "https://platform.minimaxi.com/user-center/basic-information/interface-key",
      defaultBaseUrl: "https://api.minimaxi.com/anthropic",
    },
    {
      id: "doubao",
      name: "豆包 Seed",
      category: "cn_official",
      iconColor: "#5DADEC",
      websiteUrl: "https://www.volcengine.com/product/doubao",
      apiKeyUrl:
        "https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey",
      defaultBaseUrl: "https://ark.cn-beijing.volces.com/api/coding",
    },
    // 聚合服务
    {
      id: "openrouter",
      name: "OpenRouter",
      category: "aggregator",
      iconColor: "#6366f1",
      websiteUrl: "https://openrouter.ai",
      apiKeyUrl: "https://openrouter.ai/keys",
      defaultBaseUrl: "https://openrouter.ai/api",
    },
    // 本地代理
    {
      id: "proxycast",
      name: "ProxyCast",
      category: "proxy",
      iconColor: "#3b82f6",
      defaultBaseUrl: "http://127.0.0.1:8999",
    },
    // 自定义
    {
      id: "custom",
      name: "自定义",
      category: "custom",
      iconColor: "#8b5cf6",
    },
  ],
  codex: [
    // 官方
    {
      id: "openai",
      name: "OpenAI 官方",
      category: "official",
      iconColor: "#10a37f",
      websiteUrl: "https://chatgpt.com/codex",
      apiKeyUrl: "https://platform.openai.com/api-keys",
    },
    // 第三方
    {
      id: "azure",
      name: "Azure OpenAI",
      category: "third_party",
      iconColor: "#0078D4",
      websiteUrl: "https://learn.microsoft.com/azure/ai-services/openai",
      defaultCodexAuth: {
        api_key: "",
        api_base_url: "https://YOUR_RESOURCE_NAME.openai.azure.com/openai",
      },
      defaultCodexConfig: `# Azure OpenAI 配置
model = "gpt-4o"
`,
    },
    // 本地代理
    {
      id: "proxycast",
      name: "ProxyCast",
      category: "proxy",
      iconColor: "#3b82f6",
      defaultCodexAuth: {
        api_key: "proxycast",
        api_base_url: "http://127.0.0.1:8999/v1",
      },
    },
    // 自定义
    {
      id: "custom",
      name: "自定义",
      category: "custom",
      iconColor: "#8b5cf6",
    },
  ],
  gemini: [
    // 官方
    {
      id: "google",
      name: "Google 官方",
      category: "official",
      iconColor: "#4285f4",
      websiteUrl: "https://ai.google.dev/",
      apiKeyUrl: "https://aistudio.google.com/app/apikey",
    },
    // 本地代理
    {
      id: "proxycast",
      name: "ProxyCast",
      category: "proxy",
      iconColor: "#3b82f6",
      defaultGeminiEnv: {
        GEMINI_API_KEY: "proxycast",
        GOOGLE_GEMINI_BASE_URL: "http://127.0.0.1:8999",
        GEMINI_MODEL: "gemini-2.0-flash",
      },
    },
    // 自定义
    {
      id: "custom",
      name: "自定义",
      category: "custom",
      iconColor: "#8b5cf6",
    },
  ],
  proxycast: [],
};

const categoryLabels: Record<ProviderCategory, string> = {
  official: "官方",
  cn_official: "国内官方",
  aggregator: "聚合服务",
  third_party: "第三方",
  proxy: "本地代理",
  custom: "自定义",
};

// 默认 Codex auth.json 模板
const defaultCodexAuth = JSON.stringify(
  {
    api_key: "",
    api_base_url: "",
  },
  null,
  2,
);

// 默认 Codex config.toml 模板
const defaultCodexConfig = `# Codex 配置文件
model = "gpt-4"
`;

// 默认 Gemini .env 模板
const defaultGeminiEnv = `GEMINI_API_KEY=
GOOGLE_GEMINI_BASE_URL=
GEMINI_MODEL=gemini-2.0-flash`;

// 默认 Gemini settings.json 模板
const defaultGeminiSettings = JSON.stringify(
  {
    mcpServers: {},
  },
  null,
  2,
);

export function ProviderForm({
  appType,
  provider,
  onSave,
  onCancel,
}: ProviderFormProps) {
  const isEditMode = Boolean(provider);
  const appPresets = useMemo(() => presets[appType] || [], [appType]);

  // 基础字段
  const [selectedPresetId, setSelectedPresetId] = useState<string | null>(
    isEditMode ? null : "custom",
  );
  const [name, setName] = useState(provider?.name || "");
  const [notes, setNotes] = useState(provider?.notes || "");
  const [iconColor, setIconColor] = useState(provider?.icon_color || "#6366f1");

  // 从 provider.settings_config 中提取 Claude 配置
  const extractClaudeConfig = () => {
    if (
      !provider?.settings_config ||
      typeof provider.settings_config !== "object"
    ) {
      return {
        apiKey: "",
        baseUrl: "",
        primaryModel: "",
        haikuModel: "",
        sonnetModel: "",
        opusModel: "",
      };
    }
    const env = (provider.settings_config as Record<string, unknown>).env as
      | Record<string, string>
      | undefined;
    if (!env) {
      return {
        apiKey: "",
        baseUrl: "",
        primaryModel: "",
        haikuModel: "",
        sonnetModel: "",
        opusModel: "",
      };
    }
    return {
      apiKey: env.ANTHROPIC_API_KEY || env.ANTHROPIC_AUTH_TOKEN || "",
      baseUrl: env.ANTHROPIC_BASE_URL || "",
      primaryModel: env.ANTHROPIC_MODEL || "",
      haikuModel: env.ANTHROPIC_DEFAULT_HAIKU_MODEL || "",
      sonnetModel: env.ANTHROPIC_DEFAULT_SONNET_MODEL || "",
      opusModel: env.ANTHROPIC_DEFAULT_OPUS_MODEL || "",
    };
  };

  const claudeConfig = extractClaudeConfig();

  // Claude 专属字段 - 编辑模式从现有配置加载
  const [apiKey, setApiKey] = useState(claudeConfig.apiKey);
  const [baseUrl, setBaseUrl] = useState(claudeConfig.baseUrl);
  const [primaryModel, setPrimaryModel] = useState(claudeConfig.primaryModel);
  const [haikuModel, setHaikuModel] = useState(claudeConfig.haikuModel);
  const [sonnetModel, setSonnetModel] = useState(claudeConfig.sonnetModel);
  const [opusModel, setOpusModel] = useState(claudeConfig.opusModel);

  // Claude 配置 JSON（可编辑）
  const [claudeConfigJson, setClaudeConfigJson] = useState(() => {
    if (
      provider?.settings_config &&
      typeof provider.settings_config === "object"
    ) {
      return JSON.stringify(provider.settings_config, null, 2);
    }
    return JSON.stringify({ env: {} }, null, 2);
  });
  const [jsonError, setJsonError] = useState<string | null>(null);

  // 当表单字段变化时，同步更新 JSON（仅当 JSON 未被手动修改时）
  const [jsonManuallyEdited, setJsonManuallyEdited] = useState(false);

  // 从表单字段生成 JSON
  const generateJsonFromFields = useCallback(() => {
    const env: Record<string, string> = {};
    if (apiKey) {
      // 同时设置两个 API Key，支持 OpenAI 和 Anthropic 两种协议
      env.ANTHROPIC_API_KEY = apiKey;
      env.OPENAI_API_KEY = apiKey;
    }
    if (baseUrl) {
      env.ANTHROPIC_BASE_URL = baseUrl;
      env.OPENAI_BASE_URL = baseUrl;
    }
    if (primaryModel) env.ANTHROPIC_MODEL = primaryModel;
    if (haikuModel) env.ANTHROPIC_DEFAULT_HAIKU_MODEL = haikuModel;
    if (sonnetModel) env.ANTHROPIC_DEFAULT_SONNET_MODEL = sonnetModel;
    if (opusModel) env.ANTHROPIC_DEFAULT_OPUS_MODEL = opusModel;
    return JSON.stringify({ env }, null, 2);
  }, [apiKey, baseUrl, primaryModel, haikuModel, sonnetModel, opusModel]);

  // 同步表单到 JSON（当表单字段变化且 JSON 未被手动编辑时）
  React.useEffect(() => {
    if (!jsonManuallyEdited && appType === "claude") {
      setClaudeConfigJson(generateJsonFromFields());
    }
  }, [generateJsonFromFields, jsonManuallyEdited, appType]);

  // 处理 JSON 编辑
  const handleJsonChange = (value: string) => {
    setClaudeConfigJson(value);
    setJsonManuallyEdited(true);
    setJsonError(null);

    // 尝试解析并同步回表单字段
    try {
      const parsed = JSON.parse(value);
      const env = parsed.env || {};
      setApiKey(env.ANTHROPIC_AUTH_TOKEN || env.ANTHROPIC_API_KEY || "");
      setBaseUrl(env.ANTHROPIC_BASE_URL || "");
      setPrimaryModel(env.ANTHROPIC_MODEL || "");
      setHaikuModel(env.ANTHROPIC_DEFAULT_HAIKU_MODEL || "");
      setSonnetModel(env.ANTHROPIC_DEFAULT_SONNET_MODEL || "");
      setOpusModel(env.ANTHROPIC_DEFAULT_OPUS_MODEL || "");
    } catch {
      setJsonError("JSON 格式错误");
    }
  };

  // 格式化 JSON
  const handleFormatJson = () => {
    try {
      const parsed = JSON.parse(claudeConfigJson);
      setClaudeConfigJson(JSON.stringify(parsed, null, 2));
      setJsonError(null);
    } catch {
      setJsonError("JSON 格式错误，无法格式化");
    }
  };

  // Codex 专属字段 - 使用代码编辑器
  const [codexAuth, setCodexAuth] = useState(() => {
    if (
      provider?.settings_config &&
      typeof provider.settings_config === "object"
    ) {
      const auth = (provider.settings_config as Record<string, unknown>).auth;
      if (auth) return JSON.stringify(auth, null, 2);
    }
    return defaultCodexAuth;
  });
  const [codexConfig, setCodexConfig] = useState(() => {
    if (
      provider?.settings_config &&
      typeof provider.settings_config === "object"
    ) {
      const config = (provider.settings_config as Record<string, unknown>)
        .config;
      if (typeof config === "string") return config;
    }
    return defaultCodexConfig;
  });

  // Gemini 专属字段 - 使用代码编辑器
  const [geminiEnv, setGeminiEnv] = useState(() => {
    if (
      provider?.settings_config &&
      typeof provider.settings_config === "object"
    ) {
      const env = (provider.settings_config as Record<string, unknown>).env;
      if (env && typeof env === "object") {
        return Object.entries(env as Record<string, string>)
          .map(([k, v]) => `${k}=${v}`)
          .join("\n");
      }
    }
    return defaultGeminiEnv;
  });
  const [geminiSettings, setGeminiSettings] = useState(() => {
    if (
      provider?.settings_config &&
      typeof provider.settings_config === "object"
    ) {
      const config = (provider.settings_config as Record<string, unknown>)
        .config;
      if (config && typeof config === "object") {
        return JSON.stringify(config, null, 2);
      }
    }
    return defaultGeminiSettings;
  });

  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showApiKey, setShowApiKey] = useState(false);

  // 当前选中的预设
  const selectedPreset = useMemo(() => {
    return appPresets.find((p) => p.id === selectedPresetId);
  }, [appPresets, selectedPresetId]);

  // 是否显示 API Key 和端点字段（编辑模式始终显示，新增模式非官方时显示）
  const showApiFields = isEditMode || selectedPreset?.category !== "official";

  // 按分类分组预设
  const groupedPresets = useMemo(() => {
    const groups: Partial<Record<ProviderCategory, typeof appPresets>> = {};
    for (const preset of appPresets) {
      const cat = preset.category;
      if (!groups[cat]) groups[cat] = [];
      groups[cat]!.push(preset);
    }
    return groups;
  }, [appPresets]);

  const handlePresetChange = async (presetId: string) => {
    setSelectedPresetId(presetId);
    const preset = appPresets.find((p) => p.id === presetId);
    if (preset) {
      setName(preset.name);
      setIconColor(preset.iconColor || "#6366f1");

      // ProxyCast 预设：自动从设置中读取 API Key 和端口
      if (preset.id === "proxycast") {
        try {
          const config = await getConfig();
          const proxyApiKey = config.server.api_key || "";
          const proxyHost = config.server.host || "127.0.0.1";
          const proxyPort = config.server.port || 8999;
          const proxyBaseUrl = `http://${proxyHost}:${proxyPort}`;

          if (appType === "claude") {
            setApiKey(proxyApiKey);
            setBaseUrl(proxyBaseUrl);
            // 重置 JSON 手动编辑标记，让表单同步到 JSON
            setJsonManuallyEdited(false);
          } else if (appType === "codex") {
            setCodexAuth(
              JSON.stringify(
                {
                  api_key: proxyApiKey,
                  api_base_url: `${proxyBaseUrl}/v1`,
                },
                null,
                2,
              ),
            );
          } else if (appType === "gemini") {
            setGeminiEnv(
              `GEMINI_API_KEY=${proxyApiKey}\nGOOGLE_GEMINI_BASE_URL=${proxyBaseUrl}\nGEMINI_MODEL=gemini-2.0-flash`,
            );
          }
          return;
        } catch (e) {
          console.error("Failed to load ProxyCast config:", e);
        }
      }

      // 应用预设的默认配置
      if (appType === "claude") {
        // Claude: 设置默认 base URL
        if (preset.defaultBaseUrl) {
          setBaseUrl(preset.defaultBaseUrl);
        } else {
          setBaseUrl("");
        }
        if (preset.defaultModel) {
          setPrimaryModel(preset.defaultModel);
        }
        // 重置 JSON 手动编辑标记
        setJsonManuallyEdited(false);
      } else if (appType === "codex") {
        // Codex: 设置默认 auth.json
        if (preset.defaultCodexAuth) {
          setCodexAuth(JSON.stringify(preset.defaultCodexAuth, null, 2));
        } else {
          setCodexAuth(defaultCodexAuth);
        }
        if (preset.defaultCodexConfig) {
          setCodexConfig(preset.defaultCodexConfig);
        } else {
          setCodexConfig(defaultCodexConfig);
        }
      } else if (appType === "gemini") {
        // Gemini: 设置默认 env
        if (preset.defaultGeminiEnv) {
          const envLines = Object.entries(preset.defaultGeminiEnv)
            .map(([k, v]) => `${k}=${v}`)
            .join("\n");
          setGeminiEnv(envLines);
        } else {
          setGeminiEnv(defaultGeminiEnv);
        }
      }
    }
  };

  const buildSettingsConfig = (): Record<string, unknown> => {
    if (appType === "claude") {
      // 直接使用 JSON 编辑器的值
      try {
        return JSON.parse(claudeConfigJson);
      } catch {
        throw new Error("配置 JSON 格式错误");
      }
    }

    if (appType === "codex") {
      try {
        const auth = JSON.parse(codexAuth);
        return { auth, config: codexConfig };
      } catch {
        throw new Error("auth.json 格式错误");
      }
    }

    if (appType === "gemini") {
      const env: Record<string, string> = {};
      for (const line of geminiEnv.split("\n")) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith("#")) continue;
        const idx = trimmed.indexOf("=");
        if (idx > 0) {
          const key = trimmed.slice(0, idx).trim();
          const value = trimmed.slice(idx + 1).trim();
          env[key] = value;
        }
      }
      try {
        const config = JSON.parse(geminiSettings);
        return { env, config };
      } catch {
        throw new Error("settings.json 格式错误");
      }
    }

    return {};
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!name.trim()) {
      setError("请填写供应商名称");
      return;
    }

    // 非官方供应商必填校验（新增模式必填，编辑模式可留空保持原值）
    if (
      !isEditMode &&
      showApiFields &&
      appType === "claude" &&
      !apiKey.trim()
    ) {
      setError("请填写 API Key");
      return;
    }

    try {
      setSaving(true);
      const settings_config = buildSettingsConfig();
      await onSave({
        app_type: appType,
        name: name.trim(),
        settings_config,
        category: selectedPreset?.category || "custom",
        icon_color: iconColor,
        notes: notes.trim() || undefined,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-background rounded-xl shadow-lg w-full max-w-3xl max-h-[90vh] overflow-auto border border-border">
        <div className="flex items-center justify-between p-4 border-b">
          <h3 className="text-lg font-semibold">
            {isEditMode ? "编辑供应商" : "添加供应商"}
          </h3>
          <button onClick={onCancel} className="p-1 hover:bg-muted rounded">
            <X className="h-5 w-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          {/* 预设选择器（仅新增模式） */}
          {!isEditMode && (
            <div className="space-y-3">
              <label className="block text-sm font-medium">选择预设</label>
              <div className="space-y-4">
                {Object.entries(groupedPresets).map(([category, items]) => (
                  <div key={category}>
                    <p className="text-xs text-muted-foreground mb-2">
                      {categoryLabels[category as ProviderCategory] || category}
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {items.map((preset) => (
                        <button
                          key={preset.id}
                          type="button"
                          onClick={() => handlePresetChange(preset.id)}
                          className={cn(
                            "flex items-center gap-2 px-3 py-2 rounded-lg border text-sm transition-all",
                            selectedPresetId === preset.id
                              ? "border-primary bg-primary/10 text-primary"
                              : "border-border hover:border-muted-foreground/50",
                          )}
                        >
                          <ProviderIcon providerType={preset.id} size={16} />
                          {preset.name}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 基础字段 */}
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1.5">
                供应商名称 <span className="text-destructive">*</span>
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full px-3 py-2 rounded-lg border bg-background focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none"
                placeholder="输入供应商名称"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1.5">备注</label>
              <input
                type="text"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                className="w-full px-3 py-2 rounded-lg border bg-background focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none"
                placeholder="可选备注信息"
              />
            </div>
          </div>

          {/* Claude 表单字段 */}
          {appType === "claude" && showApiFields && (
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <label className="text-sm font-medium">
                    API Key <span className="text-destructive">*</span>
                  </label>
                  {(selectedPreset?.apiKeyUrl ||
                    selectedPreset?.websiteUrl) && (
                    <a
                      href={
                        selectedPreset?.apiKeyUrl || selectedPreset?.websiteUrl
                      }
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-primary hover:underline flex items-center gap-1"
                    >
                      获取 API Key
                      <ExternalLink className="h-3 w-3" />
                    </a>
                  )}
                </div>
                <div className="relative">
                  <input
                    type={showApiKey ? "text" : "password"}
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    className="w-full px-3 py-2 pr-10 rounded-lg border bg-background focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none font-mono"
                    placeholder="sk-..."
                  />
                  <button
                    type="button"
                    onClick={() => setShowApiKey(!showApiKey)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 rounded hover:bg-muted"
                    title={showApiKey ? "隐藏" : "显示"}
                  >
                    {showApiKey ? (
                      <EyeOff className="h-4 w-4 text-muted-foreground" />
                    ) : (
                      <Eye className="h-4 w-4 text-muted-foreground" />
                    )}
                  </button>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1.5">
                  API 端点
                </label>
                <input
                  type="text"
                  value={baseUrl}
                  onChange={(e) => setBaseUrl(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg border bg-background focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none font-mono"
                  placeholder="https://api.example.com/v1"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  留空使用默认端点
                </p>
              </div>

              <div className="space-y-4">
                <label className="block text-sm font-medium">模型配置</label>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs text-muted-foreground mb-1">
                      主模型
                    </label>
                    <input
                      type="text"
                      value={primaryModel}
                      onChange={(e) => setPrimaryModel(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg border bg-background focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none text-sm"
                      placeholder="claude-sonnet-4-20250514"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-muted-foreground mb-1">
                      Haiku 默认模型
                    </label>
                    <input
                      type="text"
                      value={haikuModel}
                      onChange={(e) => setHaikuModel(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg border bg-background focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-muted-foreground mb-1">
                      Sonnet 默认模型
                    </label>
                    <input
                      type="text"
                      value={sonnetModel}
                      onChange={(e) => setSonnetModel(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg border bg-background focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-muted-foreground mb-1">
                      Opus 默认模型
                    </label>
                    <input
                      type="text"
                      value={opusModel}
                      onChange={(e) => setOpusModel(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg border bg-background focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none text-sm"
                    />
                  </div>
                </div>
                <p className="text-xs text-muted-foreground">
                  可选：指定默认使用的 Claude 模型，留空则使用系统默认。
                </p>
              </div>

              {/* 配置 JSON 编辑器 */}
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <label className="text-sm font-medium">配置 JSON</label>
                  <button
                    type="button"
                    onClick={handleFormatJson}
                    className="flex items-center gap-1 text-xs text-primary hover:underline"
                  >
                    <Wand2 className="h-3 w-3" />
                    格式化
                  </button>
                </div>
                <textarea
                  value={claudeConfigJson}
                  onChange={(e) => handleJsonChange(e.target.value)}
                  className={cn(
                    "w-full px-3 py-2 rounded-lg border bg-muted/50 font-mono text-sm focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none resize-none",
                    jsonError &&
                      "border-destructive focus:ring-destructive/20 focus:border-destructive",
                  )}
                  rows={10}
                  placeholder='{"env": {}}'
                />
                {jsonError ? (
                  <p className="text-xs text-destructive mt-1">{jsonError}</p>
                ) : (
                  <p className="text-xs text-muted-foreground mt-1">
                    可手动编辑 JSON，修改会同步到上方表单
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Codex 配置编辑器 */}
          {appType === "codex" && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                {/* auth.json 编辑器 */}
                <div>
                  <label className="block text-sm font-medium mb-1.5">
                    auth.json
                  </label>
                  <textarea
                    value={codexAuth}
                    onChange={(e) => setCodexAuth(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg border bg-muted/50 focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none font-mono text-sm"
                    rows={8}
                    placeholder='{"api_key": "", "api_base_url": ""}'
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    JSON 格式的认证配置
                  </p>
                </div>

                {/* config.toml 编辑器 */}
                <div>
                  <label className="block text-sm font-medium mb-1.5">
                    config.toml
                  </label>
                  <textarea
                    value={codexConfig}
                    onChange={(e) => setCodexConfig(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg border bg-muted/50 focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none font-mono text-sm"
                    rows={8}
                    placeholder='model = "gpt-4"'
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    TOML 格式的配置文件
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Gemini 配置编辑器 */}
          {appType === "gemini" && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                {/* .env 编辑器 */}
                <div>
                  <label className="block text-sm font-medium mb-1.5">
                    env 环境变量
                  </label>
                  <textarea
                    value={geminiEnv}
                    onChange={(e) => setGeminiEnv(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg border bg-muted/50 focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none font-mono text-sm"
                    rows={8}
                    placeholder="GEMINI_API_KEY=&#10;GOOGLE_GEMINI_BASE_URL=&#10;GEMINI_MODEL=gemini-2.0-flash"
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    每行一个环境变量，格式：KEY=VALUE
                  </p>
                </div>

                {/* settings.json 编辑器 */}
                <div>
                  <label className="block text-sm font-medium mb-1.5">
                    settings.json
                  </label>
                  <textarea
                    value={geminiSettings}
                    onChange={(e) => setGeminiSettings(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg border bg-muted/50 focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none font-mono text-sm"
                    rows={8}
                    placeholder='{"mcpServers": {}}'
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    JSON 格式的配置文件（MCP 服务器等）
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* 错误提示 */}
          {error && (
            <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20">
              <p className="text-sm text-destructive">{error}</p>
            </div>
          )}

          {/* 按钮 */}
          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onCancel}
              className="px-4 py-2 rounded-lg border hover:bg-muted transition-colors"
            >
              取消
            </button>
            <button
              type="submit"
              disabled={saving || !name.trim()}
              className="px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 transition-colors"
            >
              {saving ? "保存中..." : isEditMode ? "保存" : "添加"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
