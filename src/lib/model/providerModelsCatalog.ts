import { isAliasProvider } from "@/lib/constants/providerMappings";
import {
  inferInputModalities,
  inferModelAliasSource,
  inferModelCapabilities,
  inferModelDeploymentSource,
  inferModelManagementPlane,
  inferModelTaskFamilies,
  inferOutputModalities,
  inferRuntimeFeatures,
} from "@/lib/model/inferModelCapabilities";
import type {
  EnhancedModelMetadata,
  ProviderAliasConfig,
} from "@/lib/types/modelRegistry";

interface ProviderModelsCatalogProvider {
  key: string;
  label: string;
  registryId: string;
  fallbackRegistryId?: string;
  models?: string[];
}

interface BuiltProviderModelsResult {
  modelIds: string[];
  models: EnhancedModelMetadata[];
  hasLocalModels: boolean;
}

function sortModels(models: EnhancedModelMetadata[]): EnhancedModelMetadata[] {
  return [...models].sort((a, b) => {
    if (a.is_latest && !b.is_latest) return -1;
    if (!a.is_latest && b.is_latest) return 1;

    if (a.release_date && b.release_date) {
      return b.release_date.localeCompare(a.release_date);
    }
    if (a.release_date && !b.release_date) return -1;
    if (!a.release_date && b.release_date) return 1;

    return a.display_name.localeCompare(b.display_name);
  });
}

function normalizeModelId(value: string): string {
  return value.trim().toLowerCase();
}

function collectRegistryProviderIds(
  selectedProvider: ProviderModelsCatalogProvider,
): string[] {
  return Array.from(
    new Set(
      [
        selectedProvider.registryId,
        selectedProvider.fallbackRegistryId,
        selectedProvider.key,
      ]
        .map((value) => value?.trim())
        .filter((value): value is string => Boolean(value)),
    ),
  );
}

function convertCustomModelsToMetadata(
  models: string[],
  providerId: string,
  providerName: string,
): EnhancedModelMetadata[] {
  return models.map((modelName): EnhancedModelMetadata => {
    const taxonomyParams = {
      modelId: modelName,
      providerId,
    };
    return {
      id: modelName,
      display_name: modelName,
      provider_id: providerId,
      provider_name: providerName,
      family: null,
      tier: "pro" as const,
      capabilities: inferModelCapabilities(taxonomyParams),
      task_families: inferModelTaskFamilies(taxonomyParams),
      input_modalities: inferInputModalities(taxonomyParams),
      output_modalities: inferOutputModalities(taxonomyParams),
      runtime_features: inferRuntimeFeatures(taxonomyParams),
      deployment_source: inferModelDeploymentSource(taxonomyParams),
      management_plane: inferModelManagementPlane(taxonomyParams),
      canonical_model_id: null,
      provider_model_id: modelName,
      alias_source: inferModelAliasSource(taxonomyParams),
      pricing: null,
      limits: {
        context_length: null,
        max_output_tokens: null,
        requests_per_minute: null,
        tokens_per_minute: null,
      },
      status: "active" as const,
      release_date: null,
      is_latest: false,
      description: `自定义模型: ${modelName}`,
      source: "custom" as const,
      created_at: Date.now() / 1000,
      updated_at: Date.now() / 1000,
    };
  });
}

function convertBackendModelIdsToMetadata(
  modelIds: string[],
  selectedProvider: ProviderModelsCatalogProvider,
  registryModels: EnhancedModelMetadata[],
): EnhancedModelMetadata[] {
  const providerIds = new Set(collectRegistryProviderIds(selectedProvider));
  const registryModelMap = new Map(
    registryModels
      .filter((model) => providerIds.has(model.provider_id))
      .map((model) => [normalizeModelId(model.id), model] as const),
  );
  const dedupedModelIds = Array.from(
    new Map(
      modelIds
        .map((modelId) => modelId.trim())
        .filter((modelId) => modelId.length > 0)
        .map((modelId) => [normalizeModelId(modelId), modelId] as const),
    ).values(),
  );
  const now = Date.now() / 1000;

  return dedupedModelIds.map((modelId): EnhancedModelMetadata => {
    const registryModel = registryModelMap.get(normalizeModelId(modelId));
    if (registryModel) {
      return registryModel;
    }

    const taxonomyParams = {
      modelId,
      providerId: selectedProvider.registryId || selectedProvider.key,
    };
    return {
      id: modelId,
      display_name: modelId,
      provider_id: selectedProvider.registryId || selectedProvider.key,
      provider_name: selectedProvider.label,
      family: null,
      tier: "pro" as const,
      capabilities: inferModelCapabilities(taxonomyParams),
      task_families: inferModelTaskFamilies(taxonomyParams),
      input_modalities: inferInputModalities(taxonomyParams),
      output_modalities: inferOutputModalities(taxonomyParams),
      runtime_features: inferRuntimeFeatures(taxonomyParams),
      deployment_source: inferModelDeploymentSource(taxonomyParams),
      management_plane: inferModelManagementPlane(taxonomyParams),
      canonical_model_id: null,
      provider_model_id: modelId,
      alias_source: inferModelAliasSource(taxonomyParams),
      pricing: null,
      limits: {
        context_length: null,
        max_output_tokens: null,
        requests_per_minute: null,
        tokens_per_minute: null,
      },
      status: "active" as const,
      release_date: null,
      is_latest: false,
      description: null,
      source: "custom" as const,
      created_at: now,
      updated_at: now,
    };
  });
}

function convertAliasModelsToMetadata(
  models: string[],
  aliasConfig: ProviderAliasConfig,
  providerId: string,
  providerName: string,
): EnhancedModelMetadata[] {
  return models.map((modelName): EnhancedModelMetadata => {
    const aliasInfo = aliasConfig.aliases[modelName];
    const taxonomyParams = {
      modelId: modelName,
      providerId: aliasInfo?.provider || providerId,
      family: aliasInfo?.provider || null,
      description: aliasInfo?.description || null,
      providerModelId: modelName,
      canonicalModelId: aliasInfo?.actual || null,
      aliasSource: "relay" as const,
    };
    return {
      id: modelName,
      display_name: modelName,
      provider_id: providerId,
      provider_name: providerName,
      family: aliasInfo?.provider || null,
      tier: "pro" as const,
      capabilities: inferModelCapabilities(taxonomyParams),
      task_families: inferModelTaskFamilies(taxonomyParams),
      input_modalities: inferInputModalities(taxonomyParams),
      output_modalities: inferOutputModalities(taxonomyParams),
      runtime_features: inferRuntimeFeatures(taxonomyParams),
      deployment_source: inferModelDeploymentSource(taxonomyParams),
      management_plane: inferModelManagementPlane(taxonomyParams),
      canonical_model_id: aliasInfo?.actual || null,
      provider_model_id: modelName,
      alias_source: inferModelAliasSource(taxonomyParams),
      pricing: null,
      limits: {
        context_length: null,
        max_output_tokens: null,
        requests_per_minute: null,
        tokens_per_minute: null,
      },
      status: "active" as const,
      release_date: null,
      is_latest: false,
      description:
        aliasInfo?.description || `${aliasInfo?.actual || modelName}`,
      source: "custom" as const,
      created_at: Date.now() / 1000,
      updated_at: Date.now() / 1000,
    };
  });
}

function buildLocalProviderModels(
  selectedProvider: ProviderModelsCatalogProvider | undefined | null,
  registryModels: EnhancedModelMetadata[],
  aliasConfig: ProviderAliasConfig | null,
): BuiltProviderModelsResult {
  if (!selectedProvider) {
    return { modelIds: [], models: [], hasLocalModels: false };
  }

  let allModels: EnhancedModelMetadata[] = [];
  let allModelIds: string[] = [];

  if (
    selectedProvider.models &&
    selectedProvider.models.length > 0
  ) {
    const models = convertCustomModelsToMetadata(
      selectedProvider.models,
      selectedProvider.key,
      selectedProvider.label,
    );
    allModels = [...models];
    allModelIds = [...selectedProvider.models];
  }

  const findModelIndexById = (modelId: string): number => {
    const targetId = modelId.toLowerCase();
    return allModels.findIndex((model) => model.id.toLowerCase() === targetId);
  };

  if (isAliasProvider(selectedProvider.key) && aliasConfig) {
    const aliasModels = convertAliasModelsToMetadata(
      aliasConfig.models,
      aliasConfig,
      selectedProvider.key,
      selectedProvider.label,
    );
    const newAliasModels = aliasModels.filter(
      (model) =>
        !allModelIds.some(
          (existingModelId) =>
            existingModelId.toLowerCase() === model.id.toLowerCase(),
        ),
    );
    allModels = [...allModels, ...newAliasModels];
    allModelIds = [...allModelIds, ...newAliasModels.map((model) => model.id)];
  }

  const registryFilteredModels = registryModels.filter(
    (model) => model.provider_id === selectedProvider.registryId,
  );
  const sortedRegistryModels = sortModels(registryFilteredModels);

  for (const registryModel of sortedRegistryModels) {
    const existingIndex = findModelIndexById(registryModel.id);
    if (existingIndex >= 0) {
      allModels[existingIndex] = registryModel;
      continue;
    }

    allModels.push(registryModel);
    allModelIds.push(registryModel.id);
  }

  const hasLocalModels = Boolean(
    sortedRegistryModels.length > 0 ||
    (isAliasProvider(selectedProvider.key) &&
      aliasConfig &&
      aliasConfig.models.length > 0),
  );

  return {
    modelIds: allModelIds,
    models: allModels,
    hasLocalModels,
  };
}

export function buildProviderModelsFromRegistry(
  selectedProvider: ProviderModelsCatalogProvider | undefined | null,
  registryModels: EnhancedModelMetadata[],
  aliasConfig: ProviderAliasConfig | null,
): BuiltProviderModelsResult {
  const localResult = buildLocalProviderModels(
    selectedProvider,
    registryModels,
    aliasConfig,
  );

  if (!selectedProvider) {
    return localResult;
  }

  if (localResult.hasLocalModels || localResult.models.length > 0) {
    return localResult;
  }

  if (!selectedProvider.fallbackRegistryId) {
    return localResult;
  }

  const fallbackModels = sortModels(
    registryModels.filter(
      (model) => model.provider_id === selectedProvider.fallbackRegistryId,
    ),
  );

  if (fallbackModels.length === 0) {
    return localResult;
  }

  return {
    modelIds: fallbackModels.map((model) => model.id),
    models: fallbackModels,
    hasLocalModels: false,
  };
}

export function buildProviderModelsFromBackendModelIds(
  selectedProvider: ProviderModelsCatalogProvider | undefined | null,
  registryModels: EnhancedModelMetadata[],
  modelIds: string[],
): EnhancedModelMetadata[] {
  if (!selectedProvider) {
    return [];
  }

  return convertBackendModelIdsToMetadata(
    modelIds,
    selectedProvider,
    registryModels,
  );
}
