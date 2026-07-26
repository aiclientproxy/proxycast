/**
 * @file ModelAddPanel 组件
 * @description 模型添加流程，负责服务商分类筛选与最小配置表单。
 * @module components/api-key-provider/ModelAddPanel
 */

import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ProviderIcon } from "@/icons/providers";
import {
  interceptHttpExternalLinkClick,
  resolveHttpExternalHref,
} from "@/lib/markdown/externalLinks";
import { cn } from "@/lib/utils";
import {
  apiKeyProviderApi,
  type AddCustomProviderRequest,
  type ProviderDisplay,
  type ProviderWithKeysDisplay,
  type SystemProviderCatalogItem,
  type UpdateProviderRequest,
} from "@/lib/api/apiKeyProvider";
import {
  fetchProviderModelsAuto,
  normalizeFetchProviderModelsSource,
} from "@/lib/api/modelRegistry";
import type {
  ProviderDeclaredPromptCacheMode,
  ProviderType,
} from "@/lib/types/provider";
import { useModelRegistry } from "@/hooks/useModelRegistry";
import { getProviderModelAutoFetchCapability } from "@/lib/model/providerModelFetchSupport";
import {
  dedupeModelIds,
  AGNES_OPENAI_COMPATIBLE_API_HOST,
  getProviderTypeLabel,
  isSupportedProviderType,
  normalizeKnownProviderApiHost,
  resolvePromptCacheModeRequestValue,
  SENSENOVA_OPENAI_COMPATIBLE_API_HOST,
} from "./providerConfigUtils";
import {
  AlertCircle,
  ArrowLeft,
  Check,
  ExternalLink,
  Eye,
  Globe2,
  Loader2,
  Plus,
  RefreshCw,
  ServerCog,
  SlidersHorizontal,
  Sparkles,
  Zap,
} from "lucide-react";
import {
  buildFalModelFetchStatus,
  buildResponsesModelFetchStatus,
  extractApiModelIds,
  isFalProviderLike,
  isLikelyFalImageModel,
  isProviderApiKeyRequired,
  type ProviderModelFetchProfile,
  type ProviderModelFetchStatus,
  type ProviderModelFetchStatusCopy,
} from "./providerModelFetchHelpers";
import { formatProviderConnectionError } from "./providerConnectionError";

export type ModelAddView = "catalog" | "configure";
export type ModelCatalogCategory =
  | "recommended"
  | "cn"
  | "aggregator"
  | "overseas"
  | "local";
type ProviderRegion = "cn" | "global";
type ProviderBillingMode = "payg" | "coding_plan" | "subscription";

interface ProviderTemplate {
  id: string;
  name: string;
  description: string;
  category: ModelCatalogCategory;
  type: ProviderType;
  apiHost: string;
  recommended?: boolean;
  apiKeyUrl?: string;
  defaultModels: string[];
  iconProviderId?: string;
  systemProviderId?: string;
  providerResourceId?: string;
  isCustom?: boolean;
  region?: ProviderRegion;
  billingMode?: ProviderBillingMode;
}

interface FormState {
  name: string;
  type: ProviderType;
  promptCacheMode: ProviderDeclaredPromptCacheMode;
  apiHost: string;
  apiKey: string;
  models: string[];
}

interface ModelAddValidationCopy {
  providerNameRequired: string;
  apiHostRequired: string;
  apiHostNormalizedActivate: string;
  apiHostNormalizedFetch: string;
  apiHostInvalid: string;
  apiKeyRequired: string;
  modelRequired: string;
  fetchProviderNameRequired: string;
  fetchApiHostRequired: string;
}

interface ModelAddCatalogCopy {
  apiHostPlaceholder: string;
  registryDescriptionWithApiHost: (apiHost: string) => string;
  registryDescriptionWithoutApiHost: string;
  loadErrorDefault: string;
  templateName: (templateId: string, fallback: string) => string;
  templateDescription: (templateId: string, fallback: string) => string;
  providerTypeLabel: (providerType: string) => string;
  badgeRecommended: string;
  badgeRegionCn: string;
  badgeRegionGlobal: string;
  badgeBillingPayg: string;
  badgeBillingCodingPlan: string;
  badgeBillingSubscription: string;
}

interface ModelAddPanelProps {
  providers: ProviderWithKeysDisplay[];
  onAddProvider: (
    request: AddCustomProviderRequest,
  ) => Promise<ProviderDisplay>;
  onUpdateProvider: (
    id: string,
    request: UpdateProviderRequest,
  ) => Promise<ProviderDisplay>;
  onAddApiKey: (
    providerId: string,
    apiKey: string,
    alias?: string,
    options?: { replaceExisting?: boolean },
  ) => Promise<unknown>;
  onActivated: (providerId: string) => void;
  onCancel: () => void;
  className?: string;
}

const CATEGORY_OPTIONS: Array<{
  value: ModelCatalogCategory;
  labelKey: string;
  labelFallback: string;
}> = [
  {
    value: "recommended",
    labelKey: "settings.providers.modelAdd.catalog.category.recommended",
    labelFallback: "推荐服务",
  },
  {
    value: "cn",
    labelKey: "settings.providers.modelAdd.catalog.category.cn",
    labelFallback: "国内服务",
  },
  {
    value: "aggregator",
    labelKey: "settings.providers.modelAdd.catalog.category.aggregator",
    labelFallback: "聚合平台",
  },
  {
    value: "overseas",
    labelKey: "settings.providers.modelAdd.catalog.category.overseas",
    labelFallback: "海外平台",
  },
  {
    value: "local",
    labelKey: "settings.providers.modelAdd.catalog.category.local",
    labelFallback: "本地模型",
  },
];

const FEATURED_TEMPLATES: ProviderTemplate[] = [
  {
    id: "kimi-code-subscription",
    name: "Kimi Code 会员（订阅）",
    description:
      "Kimi Code 官方订阅入口，Anthropic 协议，适合 Claude Code 等编码客户端",
    category: "overseas",
    type: "anthropic-compatible",
    apiHost: "https://api.kimi.com/coding/",
    recommended: true,
    apiKeyUrl: "https://www.kimi.com/code",
    defaultModels: ["k2p5"],
    iconProviderId: "moonshotai",
    providerResourceId: "kimi-for-coding",
    region: "global",
    billingMode: "subscription",
  },
  {
    id: "kimi-api-cn",
    name: "Kimi API（国内按量）",
    description: "Moonshot 中国区 Anthropic 协议 API，适合按量接入 Kimi 模型",
    category: "cn",
    type: "anthropic-compatible",
    apiHost: "https://api.moonshot.cn/anthropic",
    recommended: true,
    apiKeyUrl: "https://platform.moonshot.cn/console/api-keys",
    defaultModels: ["kimi-k2.5"],
    iconProviderId: "moonshotai",
    providerResourceId: "moonshotai-cn",
    region: "cn",
    billingMode: "payg",
  },
  {
    id: "kimi-api-global",
    name: "Kimi API（海外按量）",
    description: "Moonshot 国际区 Anthropic 协议 API，适合海外账号按量接入",
    category: "overseas",
    type: "anthropic-compatible",
    apiHost: "https://api.moonshot.ai/anthropic",
    recommended: true,
    apiKeyUrl: "https://platform.moonshot.ai/console/api-keys",
    defaultModels: ["kimi-k2.5"],
    iconProviderId: "moonshotai",
    providerResourceId: "moonshotai",
    region: "global",
    billingMode: "payg",
  },
  {
    id: "minimax-coding-plan",
    name: "MiniMax Coding Plan（国内）",
    description: "MiniMax 中国区 Anthropic 协议编码套餐，默认使用 MiniMax-M2.7",
    category: "cn",
    type: "anthropic-compatible",
    apiHost: "https://api.minimaxi.com/anthropic",
    recommended: true,
    apiKeyUrl:
      "https://platform.minimaxi.com/user-center/basic-information/interface-key",
    defaultModels: ["MiniMax-M2.7"],
    iconProviderId: "minimax-cn",
    providerResourceId: "minimax-cn",
    region: "cn",
    billingMode: "coding_plan",
  },
  {
    id: "minimax-coding-plan-global",
    name: "MiniMax Coding Plan（海外）",
    description: "MiniMax 国际区 Anthropic 协议编码套餐，使用海外订阅入口",
    category: "overseas",
    type: "anthropic-compatible",
    apiHost: "https://api.minimax.io/anthropic",
    recommended: true,
    apiKeyUrl:
      "https://platform.minimax.io/user-center/basic-information/interface-key",
    defaultModels: ["MiniMax-M2.7"],
    iconProviderId: "minimax",
    providerResourceId: "minimax",
    region: "global",
    billingMode: "coding_plan",
  },
  {
    id: "glm-cn-coding-plan",
    name: "GLM Coding Plan（国内）",
    description: "智谱 BigModel 中国区 Anthropic/Claude API 兼容编码入口",
    category: "cn",
    type: "anthropic-compatible",
    apiHost: "https://open.bigmodel.cn/api/anthropic",
    recommended: true,
    apiKeyUrl: "https://open.bigmodel.cn/usercenter/apikeys",
    defaultModels: ["glm-4.7"],
    iconProviderId: "zhipuai",
    providerResourceId: "zhipuai-coding-plan",
    region: "cn",
    billingMode: "coding_plan",
  },
  {
    id: "zai-coding-plan",
    name: "Z.AI Coding Plan（海外）",
    description: "Z.AI 国际区 Anthropic/Claude API 兼容编码入口",
    category: "overseas",
    type: "anthropic-compatible",
    apiHost: "https://api.z.ai/api/anthropic",
    recommended: true,
    apiKeyUrl: "https://z.ai/manage-apikey/apikey-list",
    defaultModels: ["glm-4.7"],
    iconProviderId: "zai",
    providerResourceId: "zai-coding-plan",
    region: "global",
    billingMode: "coding_plan",
  },
  {
    id: "mimo-coding-plan",
    name: "MiMo Coding Plan",
    description: "小米 MiMo Token Plan，兼容 Claude Code 的 Anthropic 协议",
    category: "cn",
    type: "anthropic-compatible",
    apiHost: "https://token-plan-cn.xiaomimimo.com/anthropic",
    recommended: true,
    apiKeyUrl: "https://mimo.mi.com/",
    defaultModels: [],
    iconProviderId: "xiaomi",
    providerResourceId: "xiaomi",
    region: "cn",
    billingMode: "subscription",
  },
  {
    id: "alibaba-coding-plan-cn",
    name: "Alibaba Coding Plan（国内）",
    description: "阿里云百炼中国区 Claude Code Coding Plan 专用 Anthropic 入口",
    category: "cn",
    type: "anthropic-compatible",
    apiHost: "https://coding.dashscope.aliyuncs.com/apps/anthropic",
    recommended: true,
    apiKeyUrl: "https://bailian.console.aliyun.com/",
    defaultModels: ["qwen3-coder-plus"],
    iconProviderId: "alibaba-cn",
    providerResourceId: "alibaba-cn",
    region: "cn",
    billingMode: "coding_plan",
  },
  {
    id: "alibaba-coding-plan-global",
    name: "Alibaba Coding Plan（海外）",
    description:
      "阿里云 Model Studio 国际区 Claude Code Coding Plan 专用 Anthropic 入口",
    category: "overseas",
    type: "anthropic-compatible",
    apiHost: "https://coding-intl.dashscope.aliyuncs.com/apps/anthropic",
    recommended: true,
    apiKeyUrl: "https://modelstudio.console.alibabacloud.com/",
    defaultModels: ["qwen3-coder-plus"],
    iconProviderId: "alibaba",
    providerResourceId: "alibaba",
    region: "global",
    billingMode: "coding_plan",
  },
  {
    id: "agnes-image-flash",
    name: "Agnes",
    description:
      "Agnes 官方图片生成入口，OpenAI 兼容接口，默认使用 agnes-image-2.1-flash。",
    category: "recommended",
    type: "openai",
    apiHost: AGNES_OPENAI_COMPATIBLE_API_HOST,
    recommended: true,
    defaultModels: ["agnes-image-2.1-flash"],
    iconProviderId: "agnes",
    providerResourceId: "agnes",
    region: "global",
    billingMode: "payg",
  },
  {
    id: "aihubmix-recommended",
    name: "AiHubMix",
    description: "聚合 Claude、OpenAI、Gemini 的常用中转服务",
    category: "aggregator",
    type: "openai",
    apiHost: "https://aihubmix.com",
    recommended: true,
    defaultModels: ["claude-sonnet-4-5"],
    iconProviderId: "aihubmix",
    systemProviderId: "aihubmix",
  },
  {
    id: "openrouter-recommended",
    name: "OpenRouter",
    description: "海外模型聚合平台，可按模型 ID 灵活接入",
    category: "overseas",
    type: "openai",
    apiHost: "https://openrouter.ai/api/v1/",
    recommended: true,
    apiKeyUrl: "https://openrouter.ai/keys",
    defaultModels: ["anthropic/claude-sonnet-4.5"],
    iconProviderId: "openrouter",
    systemProviderId: "openrouter",
  },
];

const FEATURED_TEMPLATE_IDS = new Set(
  FEATURED_TEMPLATES.map((template) => template.id),
);

const CUSTOM_TEMPLATE: ProviderTemplate = {
  id: "custom-provider",
  name: "自定义供应商",
  description: "配置自定义 API 兼容的供应商",
  category: "recommended",
  type: "openai",
  apiHost: "",
  defaultModels: [],
  isCustom: true,
};

const CN_PROVIDER_IDS = new Set([
  "alibaba-cn",
  "bailing",
  "baidu-cloud",
  "deepseek",
  "doubao",
  "giteeai",
  "hunyuan",
  "iflowcn",
  "infini",
  "internlm",
  "kimi-for-coding",
  "minimax-cn",
  "modelscope",
  "moonshotai-cn",
  "sensenova",
  "spark",
  "stepfun",
  "taichu",
  "tencent-cloud-ti",
  "tencentcloud",
  "xiaomi",
  "xirang",
  "yi",
  "zhipuai",
  "zhipuai-coding-plan",
  "zhinao",
  "ai360",
]);

const AGGREGATOR_PROVIDER_IDS = new Set([
  "302ai",
  "abacus",
  "aihubmix",
  "baseten",
  "chutes",
  "cloudflare-ai-gateway",
  "cloudflare-workers-ai",
  "deepinfra",
  "fastrouter",
  "fireworks-ai",
  "helicone",
  "huggingface",
  "inference",
  "io-net",
  "lucidquery",
  "nano-gpt",
  "nebius",
  "novita",
  "openrouter",
  "poe",
  "requesty",
  "siliconflow",
  "siliconflow-cn",
  "submodel",
  "synthetic",
  "togetherai",
  "upstage",
  "v0",
  "venice",
  "vercel",
  "vultr",
  "zenmux",
]);

const LOCAL_PROVIDER_IDS = new Set([
  "llama",
  "lmstudio",
  "ollama",
  "ollama-cloud",
]);

const RESOURCE_PROVIDER_API_HOSTS: Record<string, string> = {
  aihubmix: "https://aihubmix.com",
  alibaba: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/",
  "alibaba-cn": "https://dashscope.aliyuncs.com/compatible-mode/v1/",
  agnes: AGNES_OPENAI_COMPATIBLE_API_HOST,
  anthropic: "https://api.anthropic.com",
  deepseek: "https://api.deepseek.com",
  "kimi-for-coding": "https://api.kimi.com/coding/",
  lmstudio: "http://localhost:1234",
  minimax: "https://api.minimax.io/anthropic",
  "minimax-cn": "https://api.minimaxi.com/anthropic",
  moonshotai: "https://api.moonshot.ai",
  "moonshotai-cn": "https://api.moonshot.cn",
  openai: "https://api.openai.com",
  openrouter: "https://openrouter.ai/api/v1/",
  sensenova: SENSENOVA_OPENAI_COMPATIBLE_API_HOST,
  siliconflow: "https://api.siliconflow.cn",
  "siliconflow-cn": "https://api.siliconflow.cn",
  xiaomi: "https://token-plan-cn.xiaomimimo.com/anthropic",
  "zai-coding-plan": "https://api.z.ai/api/anthropic",
  "zhipuai-coding-plan": "https://open.bigmodel.cn/api/anthropic",
};

const RESOURCE_PROVIDER_DEFAULT_MODELS: Record<string, string[]> = {
  agnes: ["agnes-image-2.1-flash"],
  sensenova: ["SenseChat-5"],
};

const ANTHROPIC_COMPATIBLE_REGISTRY_PROVIDER_IDS = new Set([
  "kimi-for-coding",
  "minimax",
  "minimax-cn",
  "xiaomi",
  "zai-coding-plan",
  "zhipuai-coding-plan",
]);

function normalizeCatalogProviderType(providerType: string): ProviderType {
  return isSupportedProviderType(providerType) ? providerType : "openai";
}

function resolveProviderCategory(
  providerId: string,
  group?: string,
): ModelCatalogCategory {
  if (CN_PROVIDER_IDS.has(providerId)) {
    return "cn";
  }
  if (AGGREGATOR_PROVIDER_IDS.has(providerId)) {
    return "aggregator";
  }
  if (LOCAL_PROVIDER_IDS.has(providerId)) {
    return "local";
  }

  switch (group) {
    case "chinese":
      return "cn";
    case "aggregator":
      return "aggregator";
    case "local":
      return "local";
    default:
      return "overseas";
  }
}

function buildCatalogTemplates(
  catalog: SystemProviderCatalogItem[],
  copy: ModelAddCatalogCopy,
): ProviderTemplate[] {
  return catalog.map((item) => {
    const providerType = normalizeCatalogProviderType(item.type);
    return {
      id: `catalog-${item.id}`,
      name: item.name,
      description: `${copy.providerTypeLabel(providerType)} · ${
        item.api_host || copy.apiHostPlaceholder
      }`,
      category: resolveProviderCategory(item.id, item.group),
      type: providerType,
      apiHost: item.api_host,
      defaultModels: RESOURCE_PROVIDER_DEFAULT_MODELS[item.id] ?? [],
      iconProviderId: item.id,
      systemProviderId: item.id,
    };
  });
}

function buildRegistryTemplates(
  groupedByProvider: Map<string, Array<{ id: string; provider_name: string }>>,
  catalogTemplates: ProviderTemplate[],
  copy: ModelAddCatalogCopy,
): ProviderTemplate[] {
  const catalogIds = new Set(
    catalogTemplates
      .map((template) => template.systemProviderId)
      .filter((id): id is string => Boolean(id)),
  );

  const templates: ProviderTemplate[] = [];

  groupedByProvider.forEach((models, providerId) => {
    if (catalogIds.has(providerId) || models.length === 0) {
      return;
    }

    const firstModel = models[0];
    const apiHost = RESOURCE_PROVIDER_API_HOSTS[providerId] ?? "";
    const defaultModels =
      RESOURCE_PROVIDER_DEFAULT_MODELS[providerId] ??
      models.map((model) => model.id).slice(0, 3);
    templates.push({
      id: `registry-${providerId}`,
      name: firstModel.provider_name || providerId,
      description: apiHost
        ? copy.registryDescriptionWithApiHost(apiHost)
        : copy.registryDescriptionWithoutApiHost,
      category: resolveProviderCategory(providerId),
      type: ANTHROPIC_COMPATIBLE_REGISTRY_PROVIDER_IDS.has(providerId)
        ? "anthropic-compatible"
        : "openai",
      apiHost,
      defaultModels,
      iconProviderId: providerId,
      providerResourceId: providerId,
    });
  });

  return templates.sort((a, b) => a.name.localeCompare(b.name));
}

function dedupeTemplates(templates: ProviderTemplate[]): ProviderTemplate[] {
  const seen = new Set<string>();
  const result: ProviderTemplate[] = [];

  for (const template of templates) {
    const key = template.providerResourceId
      ? `provider:${template.providerResourceId}:${template.apiHost}`
      : (template.systemProviderId ?? template.id);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    result.push(template);
  }

  return result;
}

function createInitialFormState(
  template: ProviderTemplate,
  templateName = template.name,
): FormState {
  return {
    name: template.isCustom ? "" : templateName,
    type: template.type,
    promptCacheMode: "explicit_only",
    apiHost: template.apiHost,
    apiKey: "",
    models: template.defaultModels,
  };
}

function isApiKeyRequired(type: ProviderType): boolean {
  return type !== "ollama";
}

function validateForm(
  state: FormState,
  copy: ModelAddValidationCopy,
): string | null {
  if (!state.name.trim()) {
    return copy.providerNameRequired;
  }
  if (!state.apiHost.trim()) {
    return copy.apiHostRequired;
  }
  if (state.apiHost.trim() !== normalizeKnownProviderApiHost(state.apiHost)) {
    return copy.apiHostNormalizedActivate;
  }
  try {
    new URL(state.apiHost.trim());
  } catch {
    return copy.apiHostInvalid;
  }
  if (isApiKeyRequired(state.type) && !state.apiKey.trim()) {
    return copy.apiKeyRequired;
  }
  if (state.models.length === 0) {
    return copy.modelRequired;
  }
  return null;
}

function validateModelFetchForm(
  state: FormState,
  copy: ModelAddValidationCopy,
): string | null {
  if (!state.name.trim()) {
    return copy.fetchProviderNameRequired;
  }
  if (!state.apiHost.trim()) {
    return copy.fetchApiHostRequired;
  }
  if (state.apiHost.trim() !== normalizeKnownProviderApiHost(state.apiHost)) {
    return copy.apiHostNormalizedFetch;
  }
  try {
    new URL(state.apiHost.trim());
  } catch {
    return copy.apiHostInvalid;
  }
  return null;
}

function getTemplateName(
  template: ProviderTemplate,
  copy: ModelAddCatalogCopy,
): string {
  if (!FEATURED_TEMPLATE_IDS.has(template.id)) {
    return template.name;
  }
  return copy.templateName(template.id, template.name);
}

function getTemplateDescription(
  template: ProviderTemplate,
  copy: ModelAddCatalogCopy,
): string {
  if (!FEATURED_TEMPLATE_IDS.has(template.id)) {
    return template.description;
  }
  return copy.templateDescription(template.id, template.description);
}

function renderTemplateIcon(
  template: ProviderTemplate,
  copy: ModelAddCatalogCopy,
) {
  if (template.isCustom) {
    return <SlidersHorizontal className="h-5 w-5 text-slate-500" />;
  }

  return (
    <ProviderIcon
      providerType={
        template.iconProviderId ?? template.systemProviderId ?? template.id
      }
      fallbackText={getTemplateName(template, copy)}
      size={24}
    />
  );
}

function getRegionLabel(
  region: ProviderRegion | undefined,
  copy: ModelAddCatalogCopy,
): string | null {
  switch (region) {
    case "cn":
      return copy.badgeRegionCn;
    case "global":
      return copy.badgeRegionGlobal;
    default:
      return null;
  }
}

function getBillingModeLabel(
  mode: ProviderBillingMode | undefined,
  copy: ModelAddCatalogCopy,
): string | null {
  switch (mode) {
    case "payg":
      return copy.badgeBillingPayg;
    case "coding_plan":
      return copy.badgeBillingCodingPlan;
    case "subscription":
      return copy.badgeBillingSubscription;
    default:
      return null;
  }
}

function renderTemplateBadges(
  template: ProviderTemplate,
  copy: ModelAddCatalogCopy,
) {
  const badges = [
    template.recommended ? copy.badgeRecommended : null,
    getRegionLabel(template.region, copy),
    getBillingModeLabel(template.billingMode, copy),
  ].filter((item): item is string => Boolean(item));

  if (badges.length === 0) {
    return null;
  }

  return (
    <span className="flex flex-wrap items-center gap-1.5">
      {badges.map((badge) => (
        <Badge
          key={badge}
          className="border border-slate-200 bg-slate-50 px-2 py-0 text-[11px] text-slate-600 hover:bg-slate-50"
        >
          {badge}
        </Badge>
      ))}
    </span>
  );
}

function buildStatusClass(tone: ProviderModelFetchStatus["tone"]): string {
  switch (tone) {
    case "success":
      return "border-emerald-200 bg-emerald-50 text-emerald-700";
    case "error":
      return "border-rose-200 bg-rose-50 text-rose-700";
    case "info":
    default:
      return "border-sky-200 bg-sky-50 text-sky-700";
  }
}

function getStatusIcon(tone: ProviderModelFetchStatus["tone"]) {
  if (tone === "success") {
    return <Check className="h-4 w-4" />;
  }
  if (tone === "error") {
    return <AlertCircle className="h-4 w-4" />;
  }
  return <Sparkles className="h-4 w-4" />;
}

function hasConfiguredApiKey(provider?: ProviderWithKeysDisplay | null) {
  if (!provider) {
    return false;
  }
  if (provider.api_keys?.length) {
    return provider.api_keys.some((apiKey) => apiKey.enabled);
  }
  return provider.api_key_count > 0;
}

export const ModelAddPanel: React.FC<ModelAddPanelProps> = ({
  providers,
  onAddProvider,
  onUpdateProvider,
  onAddApiKey,
  onActivated,
  onCancel,
  className,
}) => {
  const { t } = useTranslation("settings");
  const [catalog, setCatalog] = useState<SystemProviderCatalogItem[]>([]);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [category, setCategory] = useState<ModelCatalogCategory>("recommended");
  const [view, setView] = useState<ModelAddView>("catalog");
  const [selectedTemplate, setSelectedTemplate] =
    useState<ProviderTemplate | null>(null);
  const [formState, setFormState] = useState<FormState>(
    createInitialFormState(CUSTOM_TEMPLATE),
  );
  const [modelDraft, setModelDraft] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [showApiKey, setShowApiKey] = useState(false);
  const [fetchingModels, setFetchingModels] = useState(false);
  const [modelFetchStatus, setModelFetchStatus] =
    useState<ProviderModelFetchStatus | null>(null);
  const [apiModelIds, setApiModelIds] = useState<string[]>([]);
  const [apiModelQuery, setApiModelQuery] = useState("");
  const [draftProviderId, setDraftProviderId] = useState<string | null>(null);
  const [persistedApiKey, setPersistedApiKey] = useState<{
    providerId: string;
    value: string;
  } | null>(null);
  const { groupedByProvider } = useModelRegistry({
    autoLoad: true,
    includeHidden: true,
  });
  const validationCopy = useMemo<ModelAddValidationCopy>(
    () => ({
      providerNameRequired: t(
        "settings.providers.modelAdd.feedback.validation.providerNameRequired",
        "请填写供应商名称。",
      ),
      apiHostRequired: t(
        "settings.providers.modelAdd.feedback.validation.apiHostRequired",
        "请填写 API Base URL。",
      ),
      apiHostNormalizedActivate: t(
        "settings.providers.modelAdd.feedback.validation.apiHostNormalizedActivate",
        "检测到你填写的是文档页或旧接口地址，已自动修正 API Base URL，请确认后再激活。",
      ),
      apiHostNormalizedFetch: t(
        "settings.providers.modelAdd.feedback.validation.apiHostNormalizedFetch",
        "检测到你填写的是文档页或旧接口地址，已自动修正 API Base URL，请确认后再获取模型。",
      ),
      apiHostInvalid: t(
        "settings.providers.modelAdd.feedback.validation.apiHostInvalid",
        "请输入有效的 API Base URL。",
      ),
      apiKeyRequired: t(
        "settings.providers.modelAdd.feedback.validation.apiKeyRequired",
        "请填写 API 密钥。",
      ),
      modelRequired: t(
        "settings.providers.modelAdd.feedback.validation.modelRequired",
        "请手动添加至少一个模型；保存后也可以在配置页从接口获取模型。",
      ),
      fetchProviderNameRequired: t(
        "settings.providers.modelAdd.feedback.validation.fetchProviderNameRequired",
        "请先填写供应商名称。",
      ),
      fetchApiHostRequired: t(
        "settings.providers.modelAdd.feedback.validation.fetchApiHostRequired",
        "请先填写 API Base URL。",
      ),
    }),
    [t],
  );
  const catalogCopy = useMemo<ModelAddCatalogCopy>(
    () => ({
      apiHostPlaceholder: t(
        "settings.providers.modelAdd.catalog.apiHostPlaceholder",
        "按本地配置填写地址",
      ),
      registryDescriptionWithApiHost: (apiHost) =>
        t(
          "settings.providers.modelAdd.catalog.registryDescriptionWithApiHost",
          {
            apiHost,
            defaultValue: "模型目录 · {{apiHost}}",
          },
        ),
      registryDescriptionWithoutApiHost: t(
        "settings.providers.modelAdd.catalog.registryDescriptionWithoutApiHost",
        "模型目录供应商，按服务文档补充 API Base URL",
      ),
      loadErrorDefault: t(
        "settings.providers.modelAdd.catalog.loadErrorDefault",
        "读取服务商目录失败",
      ),
      templateName: (templateId, fallback) =>
        t(`settings.providers.modelAdd.catalog.template.${templateId}.name`, {
          defaultValue: fallback,
        }),
      templateDescription: (templateId, fallback) =>
        t(
          `settings.providers.modelAdd.catalog.template.${templateId}.description`,
          { defaultValue: fallback },
        ),
      providerTypeLabel: (providerType) =>
        t(`settings.providers.modelAdd.catalog.providerType.${providerType}`, {
          defaultValue: getProviderTypeLabel(providerType),
        }),
      badgeRecommended: t(
        "settings.providers.modelAdd.catalog.badge.recommended",
        "推荐",
      ),
      badgeRegionCn: t(
        "settings.providers.modelAdd.catalog.badge.region.cn",
        "国内",
      ),
      badgeRegionGlobal: t(
        "settings.providers.modelAdd.catalog.badge.region.global",
        "海外",
      ),
      badgeBillingPayg: t(
        "settings.providers.modelAdd.catalog.badge.billing.payg",
        "按量 API",
      ),
      badgeBillingCodingPlan: t(
        "settings.providers.modelAdd.catalog.badge.billing.codingPlan",
        "Coding Plan",
      ),
      badgeBillingSubscription: t(
        "settings.providers.modelAdd.catalog.badge.billing.subscription",
        "订阅套餐",
      ),
    }),
    [t],
  );

  useEffect(() => {
    let cancelled = false;

    void (async () => {
      try {
        const items = await apiKeyProviderApi.getSystemProviderCatalog();
        if (!cancelled) {
          setCatalog(items);
          setCatalogError(null);
        }
      } catch (error) {
        if (!cancelled) {
          setCatalog([]);
          setCatalogError(
            error instanceof Error
              ? error.message
              : catalogCopy.loadErrorDefault,
          );
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [catalogCopy.loadErrorDefault]);

  const templates = useMemo(() => {
    const catalogTemplates = buildCatalogTemplates(catalog, catalogCopy);
    const registryTemplates = buildRegistryTemplates(
      groupedByProvider,
      catalogTemplates,
      catalogCopy,
    );

    return dedupeTemplates([
      ...FEATURED_TEMPLATES,
      ...catalogTemplates,
      ...registryTemplates,
    ]);
  }, [catalog, catalogCopy, groupedByProvider]);

  const visibleTemplates = useMemo(() => {
    if (category === "recommended") {
      return templates.filter((template) => template.recommended);
    }
    return templates.filter((template) => template.category === category);
  }, [category, templates]);

  const existingProviderById = useMemo(() => {
    const map = new Map<string, ProviderWithKeysDisplay>();
    providers.forEach((provider) => map.set(provider.id, provider));
    return map;
  }, [providers]);

  const selectTemplate = useCallback(
    (template: ProviderTemplate) => {
      setSelectedTemplate(template);
      setFormState(
        createInitialFormState(
          template,
          getTemplateName(template, catalogCopy),
        ),
      );
      setModelDraft("");
      setSubmitError(null);
      setModelFetchStatus(null);
      setApiModelIds([]);
      setApiModelQuery("");
      setDraftProviderId(null);
      setPersistedApiKey(null);
      setView("configure");
    },
    [catalogCopy],
  );

  const addModelDraft = useCallback(() => {
    const nextModels = dedupeModelIds(
      modelDraft
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean),
    );

    if (nextModels.length === 0) {
      return;
    }

    setFormState((previous) => ({
      ...previous,
      models: dedupeModelIds([...previous.models, ...nextModels]),
    }));
    setModelDraft("");
  }, [modelDraft]);

  const removeModel = useCallback((modelId: string) => {
    setFormState((previous) => ({
      ...previous,
      models: previous.models.filter(
        (item) => item.toLowerCase() !== modelId.toLowerCase(),
      ),
    }));
  }, []);

  const setMainModel = useCallback((modelId: string) => {
    setFormState((previous) => ({
      ...previous,
      models: [
        modelId,
        ...previous.models.filter(
          (item) => item.toLowerCase() !== modelId.toLowerCase(),
        ),
      ],
    }));
  }, []);

  const template = selectedTemplate ?? CUSTOM_TEMPLATE;
  const configuredProvider = useMemo(() => {
    if (draftProviderId) {
      return existingProviderById.get(draftProviderId) ?? null;
    }

    return template.systemProviderId
      ? (existingProviderById.get(template.systemProviderId) ?? null)
      : null;
  }, [draftProviderId, existingProviderById, template.systemProviderId]);

  const modelFetchProfile = useMemo<ProviderModelFetchProfile>(
    () => ({
      id:
        draftProviderId ??
        template.systemProviderId ??
        template.providerResourceId ??
        template.iconProviderId ??
        template.id,
      type: formState.type,
      api_host: normalizeKnownProviderApiHost(formState.apiHost),
    }),
    [
      draftProviderId,
      formState.apiHost,
      formState.type,
      template.iconProviderId,
      template.id,
      template.providerResourceId,
      template.systemProviderId,
    ],
  );

  const modelAutoFetchCapability = useMemo(
    () =>
      getProviderModelAutoFetchCapability({
        providerId: modelFetchProfile.id,
        providerType: formState.type,
        apiHost: modelFetchProfile.api_host,
      }),
    [formState.type, modelFetchProfile.api_host, modelFetchProfile.id],
  );

  const modelFetchApiKeyRequired = modelAutoFetchCapability.requiresApiKey;
  const providerApiKeyRequired = isProviderApiKeyRequired(
    modelFetchProfile,
    modelFetchApiKeyRequired,
  );
  const canReadModelsFromApi =
    Boolean(formState.apiHost.trim()) &&
    modelAutoFetchCapability.supported &&
    (!modelFetchApiKeyRequired ||
      hasConfiguredApiKey(configuredProvider) ||
      formState.apiKey.trim().length > 0);
  const modelFetchUnsupportedMessage = useMemo(() => {
    switch (formState.type) {
      case "azure-openai":
        return t(
          "settings.providers.modelAdd.feedback.modelFetch.unsupported.azureOpenai",
          "Azure OpenAI 的模型枚举仍需单独适配资源端点与 API Version，当前不展示自动获取入口。",
        );
      case "vertexai":
        return t(
          "settings.providers.modelAdd.feedback.modelFetch.unsupported.vertexAi",
          "Vertex AI 需要单独的云端认证与项目上下文，当前不展示自动获取入口。",
        );
      case "aws-bedrock":
        return t(
          "settings.providers.modelAdd.feedback.modelFetch.unsupported.awsBedrock",
          "AWS Bedrock 需要专门的云凭证签名流程，当前不展示自动获取入口。",
        );
      default:
        return t(
          "settings.providers.modelAdd.feedback.modelFetch.unsupported.default",
          "当前协议暂不支持自动获取最新模型，请手动添加模型 ID。",
        );
    }
  }, [formState.type, t]);
  const modelFetchStatusCopy = useMemo<ProviderModelFetchStatusCopy>(
    () => ({
      responsesConfirmedImage: (imageModel) =>
        t(
          "settings.providers.modelAdd.feedback.modelFetch.responses.confirmedImage",
          {
            imageModel,
            defaultValue:
              "已确认 Responses 图片模型 {{imageModel}}，该入口无需标准 /models 枚举，图片生成会走 Responses image_generation。",
          },
        ),
      responsesManualImage: t(
        "settings.providers.modelAdd.feedback.modelFetch.responses.manualImage",
        "该 Responses 图片入口不提供标准 /models 枚举；请手动添加 gpt-images-2 或 gpt-image-2，图片生成会走 Responses image_generation。",
      ),
      falConfirmedModel: (modelId) =>
        t(
          "settings.providers.modelAdd.feedback.modelFetch.fal.confirmedModel",
          {
            modelId,
            defaultValue:
              "已确认 Fal 模型 {{modelId}}，Fal 不提供标准 /models 枚举，后续会使用手动声明的模型 ID。",
          },
        ),
      falManualModel: t(
        "settings.providers.modelAdd.feedback.modelFetch.fal.manualModel",
        "Fal 不提供标准 /models 枚举；当前模型优先级没有可用 Fal 图片模型，请手动添加 fal-ai/nano-banana-pro、fal-ai/flux-pro 或其他 fal-ai/... 模型 ID。",
      ),
    }),
    [t],
  );
  const normalizedModelSet = useMemo(
    () => new Set(formState.models.map((model) => model.toLowerCase())),
    [formState.models],
  );
  const availableApiModels = useMemo(
    () =>
      apiModelIds.filter(
        (modelId) => !normalizedModelSet.has(modelId.toLowerCase()),
      ),
    [apiModelIds, normalizedModelSet],
  );
  const normalizedApiModelQuery = apiModelQuery.trim().toLowerCase();
  const suggestedApiModels = useMemo(
    () =>
      normalizedApiModelQuery
        ? availableApiModels.filter((modelId) =>
            modelId.toLowerCase().includes(normalizedApiModelQuery),
          )
        : availableApiModels,
    [availableApiModels, normalizedApiModelQuery],
  );

  const addFetchedModels = useCallback((models: string[]) => {
    if (models.length === 0) {
      return;
    }

    setFormState((previous) => ({
      ...previous,
      models: dedupeModelIds([...previous.models, ...models]),
    }));
  }, []);

  const persistProviderForModelFetch = useCallback(
    async (state: FormState): Promise<string> => {
      const request: AddCustomProviderRequest = {
        name: state.name.trim(),
        type: state.type,
        api_host: state.apiHost,
        prompt_cache_mode: resolvePromptCacheModeRequestValue(
          state.type,
          state.promptCacheMode,
          state.apiHost,
        ),
      };

      const existingProvider = draftProviderId
        ? (existingProviderById.get(draftProviderId) ?? null)
        : template.systemProviderId
          ? (existingProviderById.get(template.systemProviderId) ?? null)
          : null;
      let providerId = existingProvider?.id ?? draftProviderId;

      if (providerId) {
        await onUpdateProvider(providerId, {
          type: request.type,
          api_host: request.api_host,
          enabled: true,
          prompt_cache_mode: request.prompt_cache_mode,
          custom_models: state.models,
        });
      } else {
        const created = await onAddProvider(request);
        providerId = created.id;
        await onUpdateProvider(providerId, {
          enabled: true,
          custom_models: state.models,
        });
      }

      setDraftProviderId(providerId);

      const nextApiKey = state.apiKey.trim();
      if (
        nextApiKey &&
        !(
          persistedApiKey?.providerId === providerId &&
          persistedApiKey.value === nextApiKey
        )
      ) {
        await onAddApiKey(providerId, nextApiKey, undefined, {
          replaceExisting: true,
        });
        setPersistedApiKey({ providerId, value: nextApiKey });
      }

      return providerId;
    },
    [
      draftProviderId,
      existingProviderById,
      onAddApiKey,
      onAddProvider,
      onUpdateProvider,
      persistedApiKey,
      template.systemProviderId,
    ],
  );

  const handleFetchModelsFromApi = useCallback(async () => {
    const normalizedFormState = {
      ...formState,
      apiHost: normalizeKnownProviderApiHost(formState.apiHost),
    };
    if (normalizedFormState.apiHost !== formState.apiHost) {
      setFormState((previous) => ({
        ...previous,
        apiHost: normalizedFormState.apiHost,
      }));
    }

    const validationError = validateModelFetchForm(
      normalizedFormState,
      validationCopy,
    );
    if (validationError) {
      setModelFetchStatus({ tone: "error", message: validationError });
      return;
    }

    if (!modelAutoFetchCapability.supported) {
      setModelFetchStatus({
        tone: "info",
        message: modelFetchUnsupportedMessage,
      });
      return;
    }

    if (!canReadModelsFromApi) {
      setModelFetchStatus({
        tone: "error",
        message: modelFetchApiKeyRequired
          ? t(
              "settings.providers.modelAdd.feedback.modelFetch.needApiKey",
              "请先填写 API 密钥，再从接口获取模型。",
            )
          : t(
              "settings.providers.modelAdd.feedback.modelFetch.needApiHost",
              "请先补全 API Base URL，再从接口获取模型。",
            ),
      });
      return;
    }

    setFetchingModels(true);
    setModelFetchStatus(null);

    try {
      const providerId =
        await persistProviderForModelFetch(normalizedFormState);
      const result = await fetchProviderModelsAuto(providerId);
      const source = normalizeFetchProviderModelsSource(result);
      const fetchedModelIds = extractApiModelIds(result.models ?? []);

      if (source !== "Api") {
        setApiModelIds([]);
        const responsesStatus = buildResponsesModelFetchStatus(
          result,
          normalizedFormState.models,
          modelFetchStatusCopy,
        );
        const falStatus = buildFalModelFetchStatus(
          modelFetchProfile,
          result,
          normalizedFormState.models,
          modelFetchStatusCopy,
        );
        setModelFetchStatus({
          tone:
            responsesStatus?.tone ??
            falStatus?.tone ??
            (source === "Error" ? "error" : "info"),
          message:
            responsesStatus?.message ??
            falStatus?.message ??
            (source === "Error"
              ? (result.error ??
                t(
                  "settings.providers.modelAdd.feedback.modelFetch.errorWithManualFallback",
                  "接口获取模型失败。请手动添加模型 ID。",
                ))
              : t(
                  "settings.providers.modelAdd.feedback.modelFetch.emptyRealtime",
                  "接口没有返回实时模型列表。请手动添加模型 ID。",
                )),
        });
        return;
      }

      const effectiveFetchedModelIds = isFalProviderLike(modelFetchProfile)
        ? fetchedModelIds.filter(isLikelyFalImageModel)
        : fetchedModelIds;

      if (effectiveFetchedModelIds.length === 0) {
        setApiModelIds([]);
        setModelFetchStatus({
          tone: "info",
          message: isFalProviderLike(modelFetchProfile)
            ? t(
                "settings.providers.modelAdd.feedback.modelFetch.emptyFal",
                "Fal 不提供标准 /models 枚举；当前没有可用 Fal 图片模型，请手动添加 fal-ai/nano-banana-pro、fal-ai/flux-pro 或其他 fal-ai/... 模型 ID。",
              )
            : t(
                "settings.providers.modelAdd.feedback.modelFetch.empty",
                "接口已响应，但没有返回可添加的模型 ID。请手动添加模型。",
              ),
        });
        return;
      }

      setApiModelIds(effectiveFetchedModelIds);
      setApiModelQuery("");

      const existingFetchedModelCount = effectiveFetchedModelIds.filter(
        (modelId) => normalizedModelSet.has(modelId.toLowerCase()),
      ).length;
      const shouldAutoFillModels =
        formState.models.length === 0 && effectiveFetchedModelIds.length <= 6;
      if (shouldAutoFillModels) {
        addFetchedModels(effectiveFetchedModelIds);
        setModelFetchStatus({
          tone: "success",
          message: t(
            "settings.providers.modelAdd.feedback.modelFetch.autoFilled",
            {
              count: effectiveFetchedModelIds.length,
              defaultValue: "接口返回 {{count}} 个模型，已自动加入模型优先级。",
            },
          ),
        });
        return;
      }

      if (existingFetchedModelCount === effectiveFetchedModelIds.length) {
        setModelFetchStatus({
          tone: "success",
          message: t(
            "settings.providers.modelAdd.feedback.modelFetch.confirmedAll",
            {
              count: effectiveFetchedModelIds.length,
              defaultValue:
                "已确认 {{count}} 个模型，当前模型优先级已包含全部结果。",
            },
          ),
        });
        return;
      }

      setModelFetchStatus({
        tone: "success",
        message: t("settings.providers.modelAdd.feedback.modelFetch.fetched", {
          count: effectiveFetchedModelIds.length,
          defaultValue:
            "接口返回 {{count}} 个模型，点击下方模型即可加入优先级。",
        }),
      });
    } catch (error) {
      setModelFetchStatus({
        tone: "error",
        message:
          error instanceof Error
            ? error.message
            : t(
                "settings.providers.modelAdd.feedback.modelFetch.errorDefault",
                "接口获取模型失败",
              ),
      });
    } finally {
      setFetchingModels(false);
    }
  }, [
    addFetchedModels,
    canReadModelsFromApi,
    formState,
    modelFetchStatusCopy,
    modelFetchUnsupportedMessage,
    modelAutoFetchCapability.supported,
    modelFetchApiKeyRequired,
    modelFetchProfile,
    normalizedModelSet,
    persistProviderForModelFetch,
    t,
    validationCopy,
  ]);

  const activateProvider = useCallback(async () => {
    const normalizedFormState = {
      ...formState,
      apiHost: normalizeKnownProviderApiHost(formState.apiHost),
    };
    if (normalizedFormState.apiHost !== formState.apiHost) {
      setFormState((previous) => ({
        ...previous,
        apiHost: normalizedFormState.apiHost,
      }));
    }

    const validationError = validateForm(normalizedFormState, validationCopy);
    if (validationError) {
      setSubmitError(validationError);
      return;
    }

    setSubmitting(true);
    setSubmitError(null);
    let persistedProviderId: string | null = null;

    try {
      const request: AddCustomProviderRequest = {
        name: normalizedFormState.name.trim(),
        type: normalizedFormState.type,
        api_host: normalizedFormState.apiHost,
        prompt_cache_mode: resolvePromptCacheModeRequestValue(
          normalizedFormState.type,
          normalizedFormState.promptCacheMode,
          normalizedFormState.apiHost,
        ),
      };

      const existingProvider = draftProviderId
        ? (existingProviderById.get(draftProviderId) ?? null)
        : template.systemProviderId
          ? (existingProviderById.get(template.systemProviderId) ?? null)
          : null;
      let providerId = existingProvider?.id ?? draftProviderId;

      if (providerId) {
        await onUpdateProvider(providerId, {
          type: request.type,
          api_host: request.api_host,
          enabled: true,
          prompt_cache_mode: request.prompt_cache_mode,
          custom_models: normalizedFormState.models,
        });
      } else {
        const created = await onAddProvider(request);
        providerId = created.id;
        setDraftProviderId(providerId);
        await onUpdateProvider(providerId, {
          enabled: true,
          custom_models: normalizedFormState.models,
        });
      }

      const nextApiKey = normalizedFormState.apiKey.trim();
      if (
        nextApiKey &&
        !(
          persistedApiKey?.providerId === providerId &&
          persistedApiKey.value === nextApiKey
        )
      ) {
        await onAddApiKey(providerId, nextApiKey, undefined, {
          replaceExisting: true,
        });
        setPersistedApiKey({ providerId, value: nextApiKey });
      }

      persistedProviderId = providerId;

      const testResult = await apiKeyProviderApi.testConnection(
        providerId,
        normalizedFormState.models[0],
      );

      if (!testResult.success) {
        const message = formatProviderConnectionError(testResult.error, {
          fallback: t(
            "settings.providers.setting.feedback.connection.failureDefault",
          ),
          timeout: t(
            "settings.providers.feedback.connection.timeout",
            "连接测试超时。请确认 API Base URL 可访问后重试。",
          ),
        });
        setSubmitError(
          t("settings.providers.modelAdd.feedback.submit.savedButTestFailed", {
            message,
            defaultValue: "配置已保存，但连接测试未通过：{{message}}",
          }),
        );
        return;
      }

      onActivated(providerId);
    } catch (error) {
      const message = formatProviderConnectionError(error, {
        fallback: t(
          "settings.providers.modelAdd.feedback.submit.errorDefault",
          "保存并测试失败",
        ),
        timeout: t(
          "settings.providers.feedback.connection.timeout",
          "连接测试超时。请确认 API Base URL 可访问后重试。",
        ),
      });
      setSubmitError(
        persistedProviderId
          ? t(
              "settings.providers.modelAdd.feedback.submit.savedButTestFailed",
              {
                message,
                defaultValue: "配置已保存，但连接测试未通过：{{message}}",
              },
            )
          : t("settings.providers.modelAdd.feedback.submit.saveFailed", {
              message,
              defaultValue: "保存配置失败：{{message}}",
            }),
      );
    } finally {
      setSubmitting(false);
    }
  }, [
    draftProviderId,
    existingProviderById,
    formState,
    onActivated,
    onAddApiKey,
    onAddProvider,
    onUpdateProvider,
    persistedApiKey,
    t,
    template.systemProviderId,
    validationCopy,
  ]);

  if (view === "catalog") {
    return (
      <div
        className={cn(
          "flex h-full flex-col overflow-y-auto bg-white px-4 py-4 lg:px-5",
          className,
        )}
        data-testid="model-add-catalog"
      >
        <div className="mb-4 grid grid-cols-2 gap-1 rounded-[18px] bg-slate-100 p-1 sm:flex sm:flex-wrap sm:items-center sm:gap-2 sm:rounded-full">
          {CATEGORY_OPTIONS.map((option) => (
            <button
              key={option.value}
              type="button"
              onClick={() => setCategory(option.value)}
              className={cn(
                "min-w-0 rounded-[14px] px-3 py-2 text-sm font-semibold transition sm:min-w-[118px] sm:rounded-full sm:px-4",
                category === option.value
                  ? "bg-white text-slate-900 shadow-sm shadow-slate-950/5"
                  : "text-slate-500 hover:text-slate-800",
              )}
              data-testid={`model-catalog-category-${option.value}`}
            >
              {t(option.labelKey, option.labelFallback)}
            </button>
          ))}
        </div>

        {catalogError ? (
          <div className="mb-4 rounded-[18px] border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
            {t("settings.providers.modelAdd.catalog.errorFallback", {
              error: catalogError,
              defaultValue: "{{error}}，已先展示内置推荐服务。",
            })}
          </div>
        ) : null}

        <div
          className="grid gap-3 lg:grid-cols-2"
          data-testid="model-template-grid"
        >
          {visibleTemplates.map((template) => (
            <button
              key={template.id}
              type="button"
              onClick={() => selectTemplate(template)}
              className="group rounded-[22px] border border-slate-200/80 bg-white px-4 py-4 text-left shadow-sm shadow-slate-950/5 transition hover:border-slate-300 hover:bg-slate-50"
              data-testid="model-template-card"
              data-template-id={template.id}
            >
              <div className="flex items-start gap-3">
                <span
                  className="flex h-8 w-8 items-center justify-center rounded-[10px] bg-slate-50"
                  aria-hidden="true"
                >
                  {renderTemplateIcon(template, catalogCopy)}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="flex min-w-0 flex-col gap-1 sm:flex-row sm:items-center sm:gap-2">
                    <span className="min-w-0 truncate text-sm font-semibold text-slate-900">
                      {getTemplateName(template, catalogCopy)}
                    </span>
                    {renderTemplateBadges(template, catalogCopy)}
                  </span>
                  <span className="mt-1 block text-sm leading-5 text-slate-500">
                    {getTemplateDescription(template, catalogCopy)}
                  </span>
                </span>
              </div>
            </button>
          ))}

          <button
            type="button"
            onClick={() => selectTemplate(CUSTOM_TEMPLATE)}
            className="rounded-[22px] border border-slate-200/80 bg-slate-100/80 px-4 py-4 text-left transition hover:bg-slate-100"
            data-testid="custom-provider-template-card"
          >
            <div className="flex items-start gap-3">
              <span className="flex h-8 w-8 items-center justify-center rounded-[10px] bg-white text-slate-500">
                <SlidersHorizontal className="h-5 w-5" />
              </span>
              <span>
                <span className="block text-sm font-semibold text-slate-900">
                  {t(
                    "settings.providers.modelAdd.catalog.custom.title",
                    "自定义供应商",
                  )}
                </span>
                <span className="mt-1 block text-sm leading-5 text-slate-500">
                  {t(
                    "settings.providers.modelAdd.catalog.custom.description",
                    "配置自定义 API 兼容的供应商",
                  )}
                </span>
              </span>
            </div>
          </button>
        </div>
      </div>
    );
  }

  const apiKeyRequired =
    providerApiKeyRequired || isApiKeyRequired(formState.type);
  const apiKeyHref = template.apiKeyUrl;
  const apiKeyRel = resolveHttpExternalHref(apiKeyHref)
    ? "noreferrer noopener"
    : undefined;

  return (
    <div
      className={cn(
        "flex h-full flex-col overflow-y-auto bg-white px-4 py-4 lg:px-5",
        className,
      )}
      data-testid="model-add-configure"
    >
      <div className="mb-3">
        <button
          type="button"
          onClick={() => {
            setView("catalog");
            setSubmitError(null);
          }}
          className="inline-flex items-center gap-2 rounded-full bg-slate-100 px-3 py-2 text-sm font-medium text-slate-600 transition hover:bg-slate-200 hover:text-slate-900"
          data-testid="model-add-back-button"
        >
          <ArrowLeft className="h-4 w-4" />
          {t("settings.providers.modelAdd.action.backToCatalog", "返回列表")}
        </button>
      </div>

      <section className="rounded-[24px] border border-slate-200/80 bg-white p-5 shadow-sm shadow-slate-950/5">
        <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-3">
            <span
              className="flex h-9 w-9 items-center justify-center rounded-[12px] bg-slate-50"
              aria-hidden="true"
            >
              {renderTemplateIcon(template, catalogCopy)}
            </span>
            <div className="min-w-0">
              <div className="flex min-w-0 flex-col gap-1 sm:flex-row sm:flex-wrap sm:items-center sm:gap-2">
                <h3 className="min-w-0 truncate text-lg font-semibold text-slate-900">
                  {template.isCustom
                    ? t(
                        "settings.providers.modelAdd.catalog.custom.title",
                        "自定义供应商",
                      )
                    : t("settings.providers.modelAdd.configure.title", {
                        providerName: getTemplateName(template, catalogCopy),
                        defaultValue: "配置 {{providerName}}",
                      })}
                </h3>
                {renderTemplateBadges(template, catalogCopy)}
              </div>
              <p className="mt-1 text-sm text-slate-500">
                {getTemplateDescription(template, catalogCopy)}
              </p>
            </div>
          </div>
          {apiKeyHref ? (
            <a
              href={apiKeyHref}
              rel={apiKeyRel}
              onAuxClick={(event) => {
                interceptHttpExternalLinkClick(event, apiKeyHref);
              }}
              onClick={(event) => {
                interceptHttpExternalLinkClick(event, apiKeyHref);
              }}
              className="inline-flex items-center gap-1.5 text-sm font-medium text-emerald-700 hover:text-emerald-800"
              data-testid="provider-api-key-link"
            >
              {t(
                "settings.providers.modelAdd.action.getApiKey",
                "去获取 API 密钥",
              )}
              <ExternalLink className="h-3.5 w-3.5" />
            </a>
          ) : null}
        </div>

        <div className="space-y-4">
          {template.isCustom ? (
            <>
              <div className="space-y-1.5">
                <Label
                  htmlFor="model-provider-name"
                  className="text-sm text-slate-600"
                >
                  {t(
                    "settings.providers.modelAdd.form.providerName.label",
                    "供应商名称",
                  )}
                </Label>
                <Input
                  id="model-provider-name"
                  value={formState.name}
                  onChange={(event) =>
                    setFormState((previous) => ({
                      ...previous,
                      name: event.target.value,
                    }))
                  }
                  placeholder={t(
                    "settings.providers.modelAdd.form.providerName.placeholder",
                    "例如：My API Provider",
                  )}
                  className="h-12 rounded-[18px] border-slate-200 bg-white px-4"
                  disabled={submitting}
                  data-testid="model-provider-name-input"
                />
              </div>

              <div className="space-y-1.5">
                <Label
                  htmlFor="model-api-host"
                  className="text-sm text-slate-600"
                >
                  {t(
                    "settings.providers.modelAdd.form.apiHost.label",
                    "API Base URL",
                  )}
                </Label>
                <Input
                  id="model-api-host"
                  value={formState.apiHost}
                  onChange={(event) =>
                    setFormState((previous) => ({
                      ...previous,
                      apiHost: event.target.value,
                    }))
                  }
                  onBlur={() =>
                    setFormState((previous) => ({
                      ...previous,
                      apiHost: normalizeKnownProviderApiHost(previous.apiHost),
                    }))
                  }
                  placeholder={t(
                    "settings.providers.modelAdd.form.apiHost.placeholder",
                    "https://api.example.com/v1",
                  )}
                  className="h-12 rounded-[18px] border-slate-200 bg-white px-4"
                  disabled={submitting}
                  data-testid="model-api-host-input"
                />
              </div>

              <div className="space-y-1.5">
                <Label className="text-sm text-slate-600">
                  {t(
                    "settings.providers.modelAdd.form.apiFormat.label",
                    "API 格式",
                  )}
                </Label>
                <div className="grid rounded-full bg-slate-100 p-1 sm:grid-cols-2">
                  {[
                    {
                      type: "openai" as ProviderType,
                      label: t(
                        "settings.providers.modelAdd.form.apiFormat.openai",
                        "OpenAI 格式",
                      ),
                    },
                    {
                      type: "anthropic-compatible" as ProviderType,
                      label: t(
                        "settings.providers.modelAdd.form.apiFormat.anthropic",
                        "Anthropic 格式",
                      ),
                    },
                  ].map((option) => (
                    <button
                      key={option.type}
                      type="button"
                      onClick={() =>
                        setFormState((previous) => ({
                          ...previous,
                          type: option.type,
                        }))
                      }
                      className={cn(
                        "rounded-full px-4 py-2.5 text-sm font-semibold transition",
                        formState.type === option.type
                          ? "bg-white text-slate-900 shadow-sm shadow-slate-950/5"
                          : "text-slate-500 hover:text-slate-800",
                      )}
                      disabled={submitting}
                      data-testid={`model-api-format-${option.type}`}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div className="grid gap-3 sm:grid-cols-2">
              {template.apiHost ? (
                <div className="rounded-[18px] border border-slate-200/80 bg-slate-50 px-4 py-3">
                  <div className="flex items-center gap-2 text-xs text-slate-500">
                    <Globe2 className="h-3.5 w-3.5" />
                    {t(
                      "settings.providers.modelAdd.form.apiHost.label",
                      "API Base URL",
                    )}
                  </div>
                  <p className="mt-2 break-all text-sm font-medium text-slate-900">
                    {formState.apiHost}
                  </p>
                </div>
              ) : (
                <div className="space-y-1.5">
                  <Label
                    htmlFor="template-api-host"
                    className="text-sm text-slate-600"
                  >
                    {t(
                      "settings.providers.modelAdd.form.apiHost.label",
                      "API Base URL",
                    )}
                  </Label>
                  <Input
                    id="template-api-host"
                    value={formState.apiHost}
                    onChange={(event) =>
                      setFormState((previous) => ({
                        ...previous,
                        apiHost: event.target.value,
                      }))
                    }
                    onBlur={() =>
                      setFormState((previous) => ({
                        ...previous,
                        apiHost: normalizeKnownProviderApiHost(
                          previous.apiHost,
                        ),
                      }))
                    }
                    placeholder={t(
                      "settings.providers.modelAdd.form.apiHost.placeholder",
                      "https://api.example.com/v1",
                    )}
                    className="h-12 rounded-[18px] border-slate-200 bg-white px-4"
                    disabled={submitting}
                    data-testid="template-api-host-input"
                  />
                </div>
              )}
              <div className="rounded-[18px] border border-slate-200/80 bg-slate-50 px-4 py-3">
                <div className="flex items-center gap-2 text-xs text-slate-500">
                  <ServerCog className="h-3.5 w-3.5" />
                  {t(
                    "settings.providers.modelAdd.form.apiFormat.label",
                    "API 格式",
                  )}
                </div>
                <p className="mt-2 text-sm font-medium text-slate-900">
                  {catalogCopy.providerTypeLabel(formState.type)}
                </p>
              </div>
            </div>
          )}

          <div className="space-y-1.5">
            <Label htmlFor="model-api-key" className="text-sm text-slate-600">
              {apiKeyRequired
                ? t("settings.providers.modelAdd.form.apiKey.label", "API 密钥")
                : t(
                    "settings.providers.modelAdd.form.apiKey.optionalLabel",
                    "API 密钥（可选）",
                  )}
            </Label>
            <div className="relative">
              <Input
                id="model-api-key"
                type={showApiKey ? "text" : "password"}
                value={formState.apiKey}
                onChange={(event) =>
                  setFormState((previous) => ({
                    ...previous,
                    apiKey: event.target.value,
                  }))
                }
                placeholder={
                  apiKeyRequired
                    ? t(
                        "settings.providers.modelAdd.form.apiKey.placeholder",
                        "输入 API 密钥",
                      )
                    : t(
                        "settings.providers.modelAdd.form.apiKey.optionalPlaceholder",
                        "本地模型可留空",
                      )
                }
                className="h-12 rounded-[18px] border-slate-200 bg-white px-4 pr-11"
                disabled={submitting}
                data-testid="model-api-key-input"
              />
              <button
                type="button"
                onClick={() => setShowApiKey((previous) => !previous)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 transition hover:text-slate-700"
                aria-label={t(
                  "settings.providers.modelAdd.form.apiKey.toggleVisibility",
                  "显示或隐藏 API 密钥",
                )}
              >
                <Eye className="h-4 w-4" />
              </button>
            </div>
          </div>

          <div className="space-y-3">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <Label className="text-sm text-slate-600">
                  {t(
                    "settings.providers.modelAdd.models.title",
                    "模型优先级（至少添加一个）",
                  )}
                </Label>
                <p className="mt-1 text-xs leading-5 text-slate-500">
                  {t(
                    "settings.providers.modelAdd.models.description",
                    "只使用接口返回或你手动添加的模型，不再显示本地兜底模型。",
                  )}
                </p>
              </div>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-9 rounded-full border-slate-200 bg-white"
                onClick={() => {
                  void handleFetchModelsFromApi();
                }}
                disabled={submitting || fetchingModels || !canReadModelsFromApi}
                data-testid="fetch-models-button"
              >
                {fetchingModels ? (
                  <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                ) : (
                  <RefreshCw className="mr-1.5 h-3.5 w-3.5" />
                )}
                {t(
                  "settings.providers.modelAdd.models.action.fetch",
                  "从接口获取",
                )}
              </Button>
            </div>

            {!modelAutoFetchCapability.supported &&
            modelAutoFetchCapability.unsupportedReason ? (
              <p className="rounded-[16px] border border-slate-200 bg-slate-50 px-3 py-2 text-xs leading-5 text-slate-600">
                {modelAutoFetchCapability.unsupportedReason}
              </p>
            ) : null}

            {modelFetchStatus ? (
              <div
                className={cn(
                  "flex items-start gap-2 rounded-[16px] border px-3 py-2 text-sm",
                  buildStatusClass(modelFetchStatus.tone),
                )}
                data-testid="model-fetch-status"
              >
                {getStatusIcon(modelFetchStatus.tone)}
                <span className="leading-5">{modelFetchStatus.message}</span>
              </div>
            ) : null}

            {availableApiModels.length > 0 ? (
              <div
                className="rounded-[18px] border border-slate-200/80 bg-slate-50 p-3"
                data-testid="api-model-suggestions"
              >
                <div className="mb-2 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                  <div className="text-xs font-medium text-slate-500">
                    {t("settings.providers.modelAdd.models.apiSuggestions", {
                      visible: suggestedApiModels.length,
                      total: availableApiModels.length,
                      defaultValue:
                        "接口模型（显示 {{visible}} / {{total}} 个，点击添加）",
                    })}
                  </div>
                  <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                    <Input
                      value={apiModelQuery}
                      onChange={(event) => setApiModelQuery(event.target.value)}
                      placeholder={t(
                        "settings.providers.modelAdd.models.filterPlaceholder",
                        "筛选接口模型",
                      )}
                      className="h-8 rounded-full border-slate-200 bg-white px-3 text-xs normal-case sm:w-[220px]"
                      autoCapitalize="none"
                      autoCorrect="off"
                      spellCheck={false}
                      data-testid="api-model-filter-input"
                    />
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="h-8 rounded-full border-slate-200 bg-white px-3 text-xs"
                      onClick={() => addFetchedModels(suggestedApiModels)}
                      disabled={suggestedApiModels.length === 0}
                      data-testid="api-model-add-all-button"
                    >
                      {t(
                        "settings.providers.modelAdd.models.action.addAll",
                        "添加全部",
                      )}
                    </Button>
                  </div>
                </div>
                <div
                  className="max-h-56 overflow-y-auto pr-1"
                  data-testid="api-model-suggestion-list"
                >
                  {suggestedApiModels.length > 0 ? (
                    <div className="flex flex-wrap gap-2">
                      {suggestedApiModels.map((modelId) => (
                        <button
                          key={modelId}
                          type="button"
                          onClick={() => addFetchedModels([modelId])}
                          className="max-w-full rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs text-slate-700 transition hover:border-slate-300 hover:text-slate-900"
                          data-testid="api-model-suggestion"
                        >
                          <span className="block max-w-[220px] truncate normal-case">
                            {modelId}
                          </span>
                        </button>
                      ))}
                    </div>
                  ) : (
                    <div className="rounded-[14px] border border-dashed border-slate-200 bg-white px-3 py-4 text-center text-xs text-slate-500">
                      {t(
                        "settings.providers.modelAdd.models.noApiMatches",
                        "没有匹配的接口模型。",
                      )}
                    </div>
                  )}
                </div>
              </div>
            ) : null}

            <div
              className="rounded-[22px] bg-slate-100 p-3"
              data-testid="model-priority-list"
            >
              {formState.models.length > 0 ? (
                <div className="space-y-2">
                  {formState.models.map((model, index) => (
                    <div
                      key={model}
                      className="flex items-center gap-3 rounded-[16px] bg-white px-3 py-2 text-sm text-slate-800"
                    >
                      <span className="text-slate-400">::</span>
                      {index === 0 ? (
                        <Badge className="border border-amber-200 bg-amber-50 px-2 py-0 text-[11px] text-amber-700 hover:bg-amber-50">
                          {t(
                            "settings.providers.modelAdd.models.primaryBadge",
                            "主模型",
                          )}
                        </Badge>
                      ) : null}
                      <span className="min-w-0 flex-1 truncate normal-case">
                        {model}
                      </span>
                      {index > 0 ? (
                        <button
                          type="button"
                          onClick={() => setMainModel(model)}
                          className="text-xs font-medium text-slate-500 hover:text-slate-900"
                        >
                          {t(
                            "settings.providers.modelAdd.models.action.setPrimary",
                            "设为主模型",
                          )}
                        </button>
                      ) : null}
                      <button
                        type="button"
                        onClick={() => removeModel(model)}
                        className="text-xs font-medium text-slate-400 hover:text-rose-600"
                      >
                        {t(
                          "settings.providers.modelAdd.models.action.remove",
                          "移除",
                        )}
                      </button>
                    </div>
                  ))}
                </div>
              ) : null}

              <div className="mt-2 flex flex-col gap-2 sm:flex-row">
                <Input
                  value={modelDraft}
                  onChange={(event) => setModelDraft(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === ",") {
                      event.preventDefault();
                      addModelDraft();
                    }
                  }}
                  placeholder={t(
                    "settings.providers.modelAdd.models.draftPlaceholder",
                    "输入模型 ID，按 Enter 添加",
                  )}
                  className="h-11 rounded-[16px] border-slate-200 bg-white px-4 normal-case"
                  disabled={submitting}
                  data-testid="model-draft-input"
                />
                <Button
                  type="button"
                  variant="secondary"
                  className="h-11 rounded-[16px] px-4"
                  onClick={addModelDraft}
                  disabled={submitting || !modelDraft.trim()}
                  data-testid="model-draft-add-button"
                >
                  <Plus className="mr-1 h-4 w-4" />
                  {t(
                    "settings.providers.modelAdd.models.action.add",
                    "添加模型",
                  )}
                </Button>
              </div>
            </div>
          </div>

          {submitError ? (
            <div
              className="flex items-start gap-2 rounded-[18px] border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700"
              data-testid="model-add-error"
              role="alert"
              aria-live="polite"
            >
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              <span className="min-w-0 break-words leading-5">
                {submitError}
              </span>
            </div>
          ) : null}

          <Button
            type="button"
            onClick={() => {
              void activateProvider();
            }}
            disabled={submitting}
            className="h-12 w-full rounded-full bg-slate-950 text-sm font-semibold text-white shadow-sm shadow-slate-950/10 hover:bg-slate-800"
            data-testid="model-activate-button"
          >
            {submitting ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                {t(
                  "settings.providers.modelAdd.action.testingConnection",
                  "正在保存并测试...",
                )}
              </>
            ) : (
              <>
                <Zap className="mr-2 h-4 w-4" />
                {t(
                  "settings.providers.modelAdd.action.saveAndTest",
                  "保存并测试",
                )}
              </>
            )}
          </Button>
        </div>
      </section>

      <button
        type="button"
        onClick={onCancel}
        className="mt-4 inline-flex items-center gap-2 self-start rounded-full px-3 py-2 text-sm font-medium text-slate-500 transition hover:bg-slate-100 hover:text-slate-800"
        data-testid="model-add-cancel-button"
      >
        <Check className="h-4 w-4" />
        {t("settings.providers.modelAdd.action.finish", "完成添加")}
      </button>
    </div>
  );
};
