/**
 * @file Provider 模型列表 Hook
 * @description 根据 Provider 获取对应的模型列表
 * @module hooks/useProviderModels
 */

import { useMemo, useState, useEffect } from "react";
import {
  modelRegistryApi,
  normalizeFetchProviderModelsSource,
  type FetchProviderModelsResult,
} from "@/lib/api/modelRegistry";
import { useModelRegistry } from "./useModelRegistry";
import { useAliasConfig } from "./useAliasConfig";
import {
  getAliasConfigKey,
  isAliasProvider,
} from "@/lib/constants/providerMappings";
import {
  buildProviderModelsFromBackendModelIds,
  buildProviderModelsFromRegistry,
} from "@/lib/model/providerModelsCatalog";
import { getProviderModelAutoFetchCapability } from "@/lib/model/providerModelFetchSupport";
import type { ConfiguredProvider } from "./useConfiguredProviders";
import type { EnhancedModelMetadata } from "@/lib/types/modelRegistry";

// ============================================================================
// 类型定义
// ============================================================================

interface UseProviderModelsOptions {
  /** 是否返回完整的模型元数据（默认只返回模型 ID） */
  returnFullMetadata?: boolean;
  /** 是否自动加载模型注册表 */
  autoLoad?: boolean;
  /** 对支持实时拉取模型的 Provider，仅接受实时模型目录 */
  liveFetchOnly?: boolean;
  /** 当前 Provider 是否有可用 API Key */
  hasApiKey?: boolean;
}

interface UseProviderModelsResult {
  /** 模型 ID 列表 */
  modelIds: string[];
  /** 完整的模型元数据列表（仅当 returnFullMetadata 为 true 时有值） */
  models: EnhancedModelMetadata[];
  /** 是否正在加载 */
  loading: boolean;
  /** 加载错误 */
  error: string | null;
}

interface LoadProviderModelsOptions {
  forceRefresh?: boolean;
  liveFetchOnly?: boolean;
  hasApiKey?: boolean;
}

function getProviderAutoFetchCapability(selectedProvider: ConfiguredProvider) {
  return getProviderModelAutoFetchCapability({
    providerId: selectedProvider.providerId ?? selectedProvider.key,
    providerType: selectedProvider.type,
    apiHost: selectedProvider.apiHost,
  });
}

function canFetchProviderModelsFromApi(
  selectedProvider: ConfiguredProvider,
  options: { hasApiKey?: boolean } = {},
): boolean {
  const capability = getProviderAutoFetchCapability(selectedProvider);
  if (!capability.supported) {
    return false;
  }
  return !capability.requiresApiKey || options.hasApiKey !== false;
}

function mergeConfiguredAndFetchedModels(
  selectedProvider: ConfiguredProvider,
  registryModels: EnhancedModelMetadata[],
  fetchedModels: EnhancedModelMetadata[],
): EnhancedModelMetadata[] {
  const configuredModels = buildProviderModelsFromBackendModelIds(
    selectedProvider,
    registryModels,
    selectedProvider.models ?? [],
  );
  const seenModelIds = new Set<string>();

  return [...configuredModels, ...fetchedModels].filter((model) => {
    const normalizedModelId = model.id.trim().toLowerCase();
    if (!normalizedModelId || seenModelIds.has(normalizedModelId)) {
      return false;
    }
    seenModelIds.add(normalizedModelId);
    return true;
  });
}

function hasDeclaredProviderModels(
  selectedProvider: ConfiguredProvider | undefined | null,
): boolean {
  return Boolean(
    selectedProvider?.models?.some((modelId) => modelId.trim()),
  );
}

function blocksProviderModelLoad(
  selectedProvider: ConfiguredProvider | undefined | null,
): boolean {
  return (
    selectedProvider?.authStatus === "login_required" &&
    !hasDeclaredProviderModels(selectedProvider)
  );
}

async function fetchProviderModelsFromApi(
  selectedProvider: ConfiguredProvider,
  options: { hasApiKey?: boolean } = {},
): Promise<EnhancedModelMetadata[]> {
  if (!canFetchProviderModelsFromApi(selectedProvider, options)) {
    return [];
  }

  try {
    const result: FetchProviderModelsResult =
      await modelRegistryApi.fetchProviderModelsAuto(
        selectedProvider.providerId ?? selectedProvider.key,
      );
    const normalizedSource = normalizeFetchProviderModelsSource(result);

    if (
      result &&
      normalizedSource === "Api" &&
      result.models &&
      result.models.length > 0
    ) {
      return result.models;
    }
  } catch {
    // ignore and fall back below
  }

  return [];
}

export async function loadProviderModels(
  selectedProvider: ConfiguredProvider | undefined | null,
  options: LoadProviderModelsOptions = {},
): Promise<EnhancedModelMetadata[]> {
  if (!selectedProvider) {
    return [];
  }

  if (selectedProvider.authStatus === "login_required") {
    return buildProviderModelsFromRegistry(selectedProvider, [], null).models;
  }

  const sourceOptions = options.forceRefresh
    ? { forceRefresh: true }
    : undefined;
  const aliasConfigPromise = isAliasProvider(selectedProvider.key)
    ? modelRegistryApi.getProviderAliasConfig(
        getAliasConfigKey(selectedProvider.key),
        sourceOptions,
      )
    : Promise.resolve(null);

  const [registryModels, aliasConfig] = await Promise.all([
    modelRegistryApi.getModelRegistry(sourceOptions),
    aliasConfigPromise,
  ]);
  const autoFetchCapability = getProviderAutoFetchCapability(selectedProvider);
  const canUseApiModelFetch = canFetchProviderModelsFromApi(selectedProvider, {
    hasApiKey: options.hasApiKey,
  });
  const useLiveFetchTruthOnly =
    options.liveFetchOnly && canUseApiModelFetch;

  const localResult = buildProviderModelsFromRegistry(
    selectedProvider,
    registryModels,
    aliasConfig,
  );
  if (useLiveFetchTruthOnly) {
    return mergeConfiguredAndFetchedModels(
      selectedProvider,
      registryModels,
      await fetchProviderModelsFromApi(selectedProvider, {
        hasApiKey: options.hasApiKey,
      }),
    );
  }

  if (localResult.hasLocalModels || localResult.models.length > 0) {
    return localResult.models;
  }

  if (isAliasProvider(selectedProvider.key)) {
    return localResult.models;
  }

  if (!autoFetchCapability.supported || !canUseApiModelFetch) {
    return localResult.models;
  }

  const apiModels = await fetchProviderModelsFromApi(selectedProvider, {
    hasApiKey: options.hasApiKey,
  });
  if (apiModels.length === 0) {
    return localResult.models;
  }
  const existingModelIds = new Set(
    localResult.models.map((model) => model.id.toLowerCase()),
  );

  return [
    ...localResult.models,
    ...apiModels.filter((model) => {
      const normalizedModelId = model.id.toLowerCase();
      if (existingModelIds.has(normalizedModelId)) {
        return false;
      }
      existingModelIds.add(normalizedModelId);
      return true;
    }),
  ];
}

// ============================================================================
// Hook 实现
// ============================================================================

/**
 * 获取 Provider 的模型列表
 *
 * 根据 Provider 类型，从别名配置或模型注册表获取模型列表。
 * 如果本地没有模型，会尝试从 Provider API 获取。
 * 支持返回模型 ID 列表或完整的模型元数据。
 *
 * @param selectedProvider 当前选中的 Provider
 * @param options 配置选项
 * @returns 模型列表、加载状态和错误信息
 *
 * @example
 * ```tsx
 * // 只获取模型 ID
 * const { modelIds, loading } = useProviderModels(selectedProvider);
 *
 * // 获取完整元数据
 * const { models, loading } = useProviderModels(selectedProvider, {
 *   returnFullMetadata: true
 * });
 * ```
 */
export function useProviderModels(
  selectedProvider: ConfiguredProvider | undefined | null,
  options: UseProviderModelsOptions = {},
): UseProviderModelsResult {
  const {
    returnFullMetadata = false,
    autoLoad = true,
    liveFetchOnly = false,
    hasApiKey,
  } = options;
  const selectedProviderBlocksModelLoad =
    blocksProviderModelLoad(selectedProvider);
  const effectiveAutoLoad = autoLoad && !selectedProviderBlocksModelLoad;

  // 获取模型注册表数据
  const {
    models: registryModels,
    loading: registryLoading,
    error: registryError,
  } = useModelRegistry({ autoLoad: effectiveAutoLoad });

  // 获取别名配置
  const { aliasConfig, loading: aliasLoading } = useAliasConfig(
    selectedProvider,
    { autoLoad: effectiveAutoLoad },
  );

  // API 获取的模型缓存
  const [apiModels, setApiModels] = useState<EnhancedModelMetadata[]>([]);
  const [apiLoading, setApiLoading] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);

  // 计算本地模型列表
  const localResult = useMemo(
    () =>
      buildProviderModelsFromRegistry(
        selectedProvider,
        registryModels,
        aliasConfig,
      ),
    [selectedProvider, registryModels, aliasConfig],
  );
  const autoFetchCapability = useMemo(
    () =>
      selectedProvider
        ? getProviderAutoFetchCapability(selectedProvider)
        : null,
    [selectedProvider],
  );
  const useLiveFetchTruthOnly = Boolean(
    liveFetchOnly &&
      autoFetchCapability?.supported &&
      (!autoFetchCapability.requiresApiKey || hasApiKey !== false),
  );
  // 当本地没有模型时，从 API 获取
  useEffect(() => {
    if (!selectedProvider) {
      setApiModels([]);
      setApiLoading(false);
      setApiError(null);
      return;
    }

    if (selectedProvider.authStatus === "login_required") {
      setApiModels([]);
      setApiLoading(false);
      setApiError(null);
      return;
    }

    if (!effectiveAutoLoad) {
      setApiModels([]);
      setApiLoading(false);
      setApiError(null);
      return;
    }

    // 如果是别名 Provider，不从 API 获取
    if (isAliasProvider(selectedProvider.key)) {
      setApiModels([]);
      setApiLoading(false);
      setApiError(null);
      return;
    }

    if (
      !autoFetchCapability?.supported ||
      (autoFetchCapability.requiresApiKey && hasApiKey === false)
    ) {
      setApiModels([]);
      setApiLoading(false);
      setApiError(null);
      return;
    }

    // 如果本地有模型，不需要从 API 获取
    if (!useLiveFetchTruthOnly && localResult.hasLocalModels) {
      setApiModels([]);
      return;
    }

    // 如果还在加载本地数据，等待
    if (registryLoading || aliasLoading) {
      return;
    }

    // 从 API 获取模型
    const fetchFromApi = async () => {
      setApiLoading(true);
      setApiError(null);

      try {
        setApiModels(
          await fetchProviderModelsFromApi(selectedProvider, { hasApiKey }),
        );
      } catch (err) {
        setApiError(err instanceof Error ? err.message : String(err));
        setApiModels([]);
      } finally {
        setApiLoading(false);
      }
    };

    fetchFromApi();
  }, [
    selectedProvider,
    effectiveAutoLoad,
    autoFetchCapability,
    hasApiKey,
    useLiveFetchTruthOnly,
    localResult.hasLocalModels,
    registryLoading,
    aliasLoading,
  ]);

  // 合并本地模型和 API 模型
  const finalResult = useMemo(() => {
    if (selectedProviderBlocksModelLoad) {
      return {
        modelIds: [],
        models: [],
      };
    }

    if (useLiveFetchTruthOnly) {
      const models = selectedProvider
        ? mergeConfiguredAndFetchedModels(
            selectedProvider,
            registryModels,
            apiModels,
          )
        : apiModels;
      return {
        modelIds: models.map((model) => model.id),
        models: returnFullMetadata ? models : [],
      };
    }

    // 如果有本地模型，使用本地模型
    if (localResult.hasLocalModels || localResult.models.length > 0) {
      return {
        modelIds: localResult.modelIds,
        models: returnFullMetadata ? localResult.models : [],
      };
    }

    // 否则使用 API 模型
    if (apiModels.length > 0) {
      const existingModelIds = new Set(
        localResult.models.map((model) => model.id.toLowerCase()),
      );
      const allModels = [
        ...localResult.models,
        ...apiModels.filter((model) => {
          const normalizedModelId = model.id.toLowerCase();
          if (existingModelIds.has(normalizedModelId)) {
            return false;
          }
          existingModelIds.add(normalizedModelId);
          return true;
        }),
      ];
      const allModelIds = allModels.map((m) => m.id);

      return {
        modelIds: allModelIds,
        models: returnFullMetadata ? allModels : [],
      };
    }

    return {
      modelIds: localResult.modelIds,
      models: returnFullMetadata ? localResult.models : [],
    };
  }, [
    apiModels,
    localResult,
    registryModels,
    returnFullMetadata,
    selectedProvider,
    selectedProviderBlocksModelLoad,
    useLiveFetchTruthOnly,
  ]);

  // 计算加载状态
  const loading = registryLoading || aliasLoading || apiLoading;

  // 计算错误状态
  const error = registryError || apiError || null;

  return {
    ...finalResult,
    loading,
    error,
  };
}
