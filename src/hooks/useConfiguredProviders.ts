/**
 * @file 已配置 Provider 列表 Hook
 * @description 从 API Key Provider 中提取已配置的 Provider 列表
 * @module hooks/useConfiguredProviders
 */

import { useEffect, useMemo, useState } from "react";
import {
  apiKeyProviderApi,
  providerModelIds,
} from "@/lib/api/apiKeyProvider";
import type { ProviderWithKeysDisplay } from "@/lib/api/apiKeyProvider";
import { useApiKeyProvider } from "./useApiKeyProvider";
import { getRegistryIdFromType } from "@/lib/constants/providerMappings";
import { resolvePromptCacheSupportNotice } from "@/lib/model/providerPromptCacheSupport";
import { isManagedLimeHubTenantModelEndpoint } from "@/lib/model/providerModelFetchSupport";
import type { ProviderDeclaredPromptCacheMode } from "@/lib/types/provider";
import {
  buildOemLimeHubApiHost,
  DEFAULT_OEM_LIME_HUB_CHAT_MODEL,
  OEM_LIME_HUB_PROVIDER_ID,
  resolveOemLimeHubProviderName,
} from "@/lib/oemLimeHubProvider";
import {
  resolveOemCloudRuntimeContext,
  type OemCloudRuntimeContext,
} from "@/lib/api/oemCloudRuntime";
import {
  subscribeOemCloudBootstrapChanged,
  subscribeOemCloudSessionChanged,
} from "@/lib/oemCloudSession";

// ============================================================================
// 类型定义
// ============================================================================

/**
 * 已配置的 Provider 信息
 */
export interface ConfiguredProvider {
  /** Provider 唯一标识 */
  key: string;
  /** 显示标签 */
  label: string;
  /** 模型注册表中的 provider_id */
  registryId: string;
  /** 回退的 registry_id（当 registryId 没有模型时使用） */
  fallbackRegistryId?: string;
  /** 原始 provider type，用于确定 API 协议 */
  type: string;
  /** 凭证类型（用于特殊处理） */
  credentialType?: string;
  /** Provider ID（用于 API Key Provider） */
  providerId?: string;
  /** Provider API Host */
  apiHost?: string;
  /** Provider 声明的 Prompt Cache 模式 */
  promptCacheMode?: ProviderDeclaredPromptCacheMode | null;
  /** 自定义模型列表（用于 API Key Provider） */
  models?: string[];
  /** 当前 Provider 是否具备可直接调用需要凭证的模型接口的条件 */
  hasApiKey?: boolean;
  /** 需要登录或授权时，供模型选择器展示明确状态 */
  authStatus?: "ready" | "login_required";
}

export interface UseConfiguredProvidersResult {
  /** 已配置的 Provider 列表 */
  providers: ConfiguredProvider[];
  /** 是否正在加载 */
  loading: boolean;
}

export interface UseConfiguredProvidersOptions {
  autoLoad?: boolean;
}

interface LoadConfiguredProvidersOptions {
  forceRefresh?: boolean;
}

interface BuildConfiguredProvidersOptions {
  oemRuntime?: OemCloudRuntimeContext | null;
}

function normalizeProviderType(value?: string | null): string {
  return (value || "").trim().toLowerCase();
}

const CONFIGURED_PROVIDER_SELECTION_ALIASES: Record<string, string> = {
  mimo: "xiaomi",
  xiaomimimo: "xiaomi",
};

function normalizeConfiguredProviderSelector(value?: string | null): string {
  const normalized = (value || "").trim().toLowerCase();
  return CONFIGURED_PROVIDER_SELECTION_ALIASES[normalized] || normalized;
}

function hasConfiguredKeylessAccess(
  provider: ProviderWithKeysDisplay,
): boolean {
  const apiHost = (provider.api_host || "").trim();
  if (!provider.enabled || apiHost.length === 0) {
    return false;
  }

  return (
    normalizeProviderType(provider.type) === "ollama" ||
    isManagedLimeHubTenantModelEndpoint({ apiHost })
  );
}

function isConfiguredApiKeyProvider(
  provider: ProviderWithKeysDisplay,
): boolean {
  return provider.enabled && hasProviderModelApiAccess(provider);
}

function hasProviderModelApiAccess(provider: ProviderWithKeysDisplay): boolean {
  return (
    provider.api_keys.some((apiKey) => apiKey.enabled) ||
    hasConfiguredKeylessAccess(provider)
  );
}

function isLimeHubProvider(provider: ProviderWithKeysDisplay): boolean {
  return provider.id.trim().toLowerCase() === OEM_LIME_HUB_PROVIDER_ID;
}

function hasOemCloudLogin(runtime?: OemCloudRuntimeContext | null): boolean {
  return Boolean(runtime?.sessionToken?.trim());
}

function shouldExposeLimeHubLoginPrompt(
  provider: ProviderWithKeysDisplay,
  runtime?: OemCloudRuntimeContext | null,
): boolean {
  if (
    hasConfiguredKeylessAccess(provider) &&
    resolveConfiguredProviderCustomModels(provider).length > 0
  ) {
    return false;
  }

  return Boolean(
    runtime &&
    provider.enabled &&
    isLimeHubProvider(provider) &&
    !hasOemCloudLogin(runtime),
  );
}

function resolveConfiguredProviderCustomModels(
  provider: ProviderWithKeysDisplay,
): string[] {
  const models = providerModelIds(provider.models);
  if (models.length > 0) {
    return Array.from(new Set(models));
  }
  if (isLimeHubProvider(provider) && hasConfiguredKeylessAccess(provider)) {
    return [DEFAULT_OEM_LIME_HUB_CHAT_MODEL];
  }
  return [];
}

function buildConfiguredProviderFromApiKeyProvider(
  provider: ProviderWithKeysDisplay,
  authStatus: ConfiguredProvider["authStatus"] = "ready",
): ConfiguredProvider {
  return {
    key: provider.id,
    label: provider.name,
    registryId: provider.id,
    fallbackRegistryId: getRegistryIdFromType(provider.type, provider.api_host),
    type: provider.type,
    credentialType: `${provider.type}_key`,
    providerId: provider.id,
    apiHost: provider.api_host,
    promptCacheMode: provider.prompt_cache_mode,
    models: resolveConfiguredProviderCustomModels(provider),
    hasApiKey: hasProviderModelApiAccess(provider),
    authStatus,
  };
}

function buildSyntheticLimeHubLoginProvider(
  runtime: OemCloudRuntimeContext,
): ConfiguredProvider {
  return {
    key: OEM_LIME_HUB_PROVIDER_ID,
    label: resolveOemLimeHubProviderName(runtime),
    registryId: OEM_LIME_HUB_PROVIDER_ID,
    fallbackRegistryId: getRegistryIdFromType("openai", runtime.gatewayBaseUrl),
    type: "openai",
    credentialType: "openai_key",
    providerId: OEM_LIME_HUB_PROVIDER_ID,
    apiHost: buildOemLimeHubApiHost(runtime) ?? runtime.gatewayBaseUrl,
    promptCacheMode: null,
    models: [DEFAULT_OEM_LIME_HUB_CHAT_MODEL],
    hasApiKey: false,
    authStatus: "login_required",
  };
}

export function buildConfiguredProviders(
  apiKeyProviders: ProviderWithKeysDisplay[],
  options: BuildConfiguredProvidersOptions = {},
): ConfiguredProvider[] {
  const safeApiKeyProviders = Array.isArray(apiKeyProviders)
    ? apiKeyProviders
    : [];
  const oemRuntime = options.oemRuntime;
  const providerMap = new Map<string, ConfiguredProvider>();

  safeApiKeyProviders.forEach((provider) => {
    const loginRequired = shouldExposeLimeHubLoginPrompt(provider, oemRuntime);
    if (
      !providerMap.has(provider.id) &&
      (isConfiguredApiKeyProvider(provider) || loginRequired)
    ) {
      providerMap.set(
        provider.id,
        buildConfiguredProviderFromApiKeyProvider(
          provider,
          loginRequired ? "login_required" : "ready",
        ),
      );
    }
  });

  if (
    oemRuntime &&
    !hasOemCloudLogin(oemRuntime) &&
    !providerMap.has(OEM_LIME_HUB_PROVIDER_ID)
  ) {
    providerMap.set(
      OEM_LIME_HUB_PROVIDER_ID,
      buildSyntheticLimeHubLoginProvider(oemRuntime),
    );
  }

  return Array.from(providerMap.values());
}

export function findConfiguredProviderBySelection(
  providers: ConfiguredProvider[] | null | undefined,
  selection?: string | null,
): ConfiguredProvider | null {
  const normalizedSelection = normalizeConfiguredProviderSelector(selection);
  if (!normalizedSelection) {
    return null;
  }
  const safeProviders = Array.isArray(providers) ? providers : [];

  const keyMatch =
    safeProviders.find(
      (provider) =>
        normalizeConfiguredProviderSelector(provider.key) ===
        normalizedSelection,
    ) ?? null;
  const providerIdMatch =
    safeProviders.find(
      (provider) =>
        normalizeConfiguredProviderSelector(provider.providerId) ===
        normalizedSelection,
    ) ?? null;

  if (keyMatch && providerIdMatch && keyMatch !== providerIdMatch) {
    if (!keyMatch.providerId && providerIdMatch.providerId) {
      return providerIdMatch;
    }
  }

  return keyMatch ?? providerIdMatch ?? null;
}

export function resolveConfiguredProviderPromptCacheSupportNotice(
  providers: ConfiguredProvider[] | null | undefined,
  selection?: string | null,
) {
  const selectedProvider = findConfiguredProviderBySelection(
    providers,
    selection,
  );
  return resolvePromptCacheSupportNotice({
    providerType: selection,
    configuredProviderType: selectedProvider?.type,
    configuredApiHost: selectedProvider?.apiHost,
    configuredPromptCacheMode: selectedProvider?.promptCacheMode,
  });
}

export async function loadConfiguredProviders(
  options: LoadConfiguredProvidersOptions = {},
): Promise<ConfiguredProvider[]> {
  const sourceOptions = options.forceRefresh
    ? { forceRefresh: true }
    : undefined;
  const apiKeyProviders = await apiKeyProviderApi.getProviders(sourceOptions);

  return buildConfiguredProviders(apiKeyProviders, {
    oemRuntime: resolveOemCloudRuntimeContext(),
  });
}

// ============================================================================
// Hook 实现
// ============================================================================

/**
 * 获取已配置的 Provider 列表
 *
 * 从 API Key Provider 中提取已配置凭证的 Provider 列表。
 *
 * @returns 已配置的 Provider 列表和加载状态
 *
 * @example
 * ```tsx
 * const { providers, loading } = useConfiguredProviders();
 *
 * if (loading) return <Spinner />;
 *
 * return (
 *   <select>
 *     {providers.map(p => (
 *       <option key={p.key} value={p.key}>{p.label}</option>
 *     ))}
 *   </select>
 * );
 * ```
 */
export function useConfiguredProviders(
  options: UseConfiguredProvidersOptions = {},
): UseConfiguredProvidersResult {
  const { autoLoad = true } = options;
  const [oemCloudRevision, setOemCloudRevision] = useState(0);
  const { providers: apiKeyProviders, loading: apiKeyLoading } =
    useApiKeyProvider({ autoLoad, hydrateUiState: false });

  useEffect(() => {
    if (!autoLoad) {
      return;
    }

    const bumpRevision = () => setOemCloudRevision((revision) => revision + 1);
    const unsubscribeSession = subscribeOemCloudSessionChanged(bumpRevision);
    const unsubscribeBootstrap =
      subscribeOemCloudBootstrapChanged(bumpRevision);

    return () => {
      unsubscribeSession();
      unsubscribeBootstrap();
    };
  }, [autoLoad]);

  // 计算已配置的 Provider 列表
  const providers = useMemo(() => {
    if (!autoLoad) {
      return buildConfiguredProviders(apiKeyProviders);
    }
    void oemCloudRevision;
    return buildConfiguredProviders(apiKeyProviders, {
      oemRuntime: resolveOemCloudRuntimeContext(),
    });
  }, [apiKeyProviders, autoLoad, oemCloudRevision]);

  return {
    providers,
    loading: apiKeyLoading,
  };
}
