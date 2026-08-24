/**
 * 模型注册表 API
 *
 * 提供与后端 ModelRegistryService 交互的 API
 */

import { AppServerClient } from "@/lib/api/appServer";
import { subscribeAppServerNotifications } from "@/lib/api/appServerEventBus";
import {
  buildModelContextPolicy,
  type ModelContextPolicyInput,
} from "@/lib/model/modelContextPolicy";
import {
  buildModelExecutionPolicy,
  type ModelExecutionPolicyInput,
} from "@/lib/model/modelExecutionPolicy";
import {
  buildModelInputModalityPolicy,
  type ModelInputModalityPolicyInput,
} from "@/lib/model/modelInputModalityPolicy";
import {
  buildModelNativeToolPolicy,
  type ModelNativeToolPolicyInput,
} from "@/lib/model/modelNativeToolPolicy";
import {
  buildCatalogModelPickerPolicy,
  buildModelPickerPolicy,
  type ModelPickerPolicyInput,
} from "@/lib/model/modelPickerPolicy";
import {
  buildModelReasoningOutputPolicy,
  type ModelReasoningOutputPolicyInput,
} from "@/lib/model/modelReasoningOutputPolicy";
import {
  buildModelReasoningPolicy,
  type ModelReasoningEffortPreset,
  type ModelReasoningPolicyInput,
} from "@/lib/model/modelReasoningPolicy";
import {
  buildModelResponsesPolicy,
  type ModelResponsesPolicyInput,
} from "@/lib/model/modelResponsesPolicy";
import {
  buildModelToolCallPolicy,
  type ModelToolCallPolicyInput,
} from "@/lib/model/modelToolCallPolicy";
import {
  buildModelTruncationPolicy,
  type ModelTruncationPolicyInput,
} from "@/lib/model/modelTruncationPolicy";
import type {
  EnhancedModelMetadata,
  ModelCapabilityProvenance,
  ModelReasoningEffortLevel,
  ModelReasoningEffortSource,
  ModelReasoningEffortSupport,
  ModelSyncState,
  ProviderAliasConfig,
  UserModelPreference,
} from "@/lib/types/modelRegistry";
import { decodeModelRouteSelector } from "../../../packages/app-server-client/src/model-route";
import { modelListUpdatedServerNotification } from "../../../packages/app-server-client/src/server-notifications";
import {
  METHOD_MODEL_LIST,
  METHOD_MODEL_PROVIDER_CAPABILITIES_READ,
  METHOD_MODEL_PREFERENCES_LIST,
  METHOD_MODEL_PROVIDER_LIST,
  METHOD_MODEL_PROVIDER_ALIAS_LIST,
  METHOD_MODEL_PROVIDER_ALIAS_READ,
  METHOD_MODEL_PROVIDER_FETCH_MODELS,
  METHOD_MODEL_SYNC_STATE_READ,
  type Model,
  type ModelInfo,
  type ModelListParams,
  type ModelListResponse,
  type ModelProviderFetchModelsResponse,
  type ModelProviderListResponse,
  type ProviderInfo,
} from "../../../packages/app-server-client/src/protocol";

type ModelRegistryAppServerClient = Pick<AppServerClient, "request">;

type ModelPreferencesListAppServerResponse = {
  preferences?: UserModelPreference[] | null;
};

type ModelSyncStateReadAppServerResponse = {
  syncState?: ModelSyncState | null;
};

type ModelProviderAliasReadAppServerResponse = {
  config?: ProviderAliasConfig | null;
};

type ModelProviderAliasListAppServerResponse = {
  configs?: Record<string, ProviderAliasConfig> | null;
};

type ModelProviderIdRecord = {
  id?: unknown;
};

export type ModelProviderCapabilities = {
  namespaceTools: boolean;
  imageGeneration: boolean;
  webSearch: boolean;
};

function emptyModelProviderCapabilities(): ModelProviderCapabilities {
  return {
    namespaceTools: false,
    imageGeneration: false,
    webSearch: false,
  };
}

function parseModelProviderCapabilities(
  value: unknown,
): ModelProviderCapabilities {
  const source = recordValue(value);
  if (
    !source ||
    typeof source.namespaceTools !== "boolean" ||
    typeof source.imageGeneration !== "boolean" ||
    typeof source.webSearch !== "boolean"
  ) {
    return emptyModelProviderCapabilities();
  }
  return {
    namespaceTools: source.namespaceTools,
    imageGeneration: source.imageGeneration,
    webSearch: source.webSearch,
  };
}

function recordValue(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function nonEmptyString(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const normalized = value.trim();
  return normalized || null;
}

function reasoningEffortPresetFromUnknown(
  value: unknown,
): ModelReasoningEffortPreset | null {
  const source = recordValue(value);
  if (!source) {
    return null;
  }
  const canonicalValue = nonEmptyString(source.value);
  if (!canonicalValue) {
    return null;
  }
  return {
    id: nonEmptyString(source.id) ?? canonicalValue,
    value: canonicalValue,
    label: nonEmptyString(source.label) ?? "",
    description: nonEmptyString(source.description) ?? "",
  };
}

function reasoningEffortSourceFromUnknown(
  value: unknown,
): ModelReasoningEffortSource | undefined {
  return value === "api" || value === "registry" || value === "custom"
    ? value
    : undefined;
}

function reasoningEffortSupportFromUnknown(
  value: unknown,
): ModelReasoningEffortSupport | null {
  const source = recordValue(value);
  if (!source || typeof source.supported !== "boolean") {
    return null;
  }
  const levels = Array.isArray(source.levels)
    ? source.levels
        .map(nonEmptyString)
        .filter((level): level is ModelReasoningEffortLevel => level !== null)
    : [];
  const options = Array.isArray(source.options)
    ? source.options
        .map(reasoningEffortPresetFromUnknown)
        .filter(
          (option): option is ModelReasoningEffortPreset => option !== null,
        )
    : [];
  return {
    supported: source.supported,
    levels,
    options,
    default: nonEmptyString(source.default),
    source: reasoningEffortSourceFromUnknown(source.source),
  };
}

function capabilityProvenanceFromSnapshot(
  value: unknown,
  modelId: string,
): ModelCapabilityProvenance {
  if (
    value === "canonical" ||
    value === "provider_explicit" ||
    value === "inferred_hint"
  ) {
    return value;
  }
  throw new Error(
    `App Server model/list returned invalid capability source for ${modelId}`,
  );
}

function toSnakeModelInfo(model: ModelInfo): EnhancedModelMetadata {
  return {
    id: model.id,
    display_name: model.displayName,
    provider_id: model.providerId,
    provider_name: model.providerName,
    family: model.family ?? null,
    tier: model.tier as EnhancedModelMetadata["tier"],
    capabilities: {
      vision: Boolean(model.capabilities?.vision),
      tools: Boolean(model.capabilities?.tools),
      streaming: Boolean(model.capabilities?.streaming),
      json_mode: Boolean(model.capabilities?.jsonMode),
      function_calling: Boolean(model.capabilities?.functionCalling),
      reasoning: Boolean(model.capabilities?.reasoning),
      reasoning_effort: reasoningEffortSupportFromUnknown(
        model.capabilities?.reasoningEffort,
      ),
    },
    capability_provenance: capabilityProvenanceFromSnapshot(
      model.capabilityProvenance,
      model.id,
    ),
    execution_policy: buildModelExecutionPolicy(
      model as ModelExecutionPolicyInput,
    ),
    context_policy: buildModelContextPolicy(model as ModelContextPolicyInput),
    picker_policy: buildModelPickerPolicy(model as ModelPickerPolicyInput),
    tool_call_policy: buildModelToolCallPolicy(
      model as ModelToolCallPolicyInput,
    ),
    reasoning_policy: buildModelReasoningPolicy(
      model as ModelReasoningPolicyInput,
    ),
    reasoning_output_policy: buildModelReasoningOutputPolicy(
      model as ModelReasoningOutputPolicyInput,
    ),
    input_modality_policy: buildModelInputModalityPolicy(
      model as ModelInputModalityPolicyInput,
    ),
    responses_policy: buildModelResponsesPolicy(
      model as ModelResponsesPolicyInput,
    ),
    truncation_policy: buildModelTruncationPolicy(
      model as ModelTruncationPolicyInput,
    ),
    native_tool_policy: buildModelNativeToolPolicy(
      model as ModelNativeToolPolicyInput,
    ),
    task_families: (model.taskFamilies ??
      []) as EnhancedModelMetadata["task_families"],
    input_modalities: (model.inputModalities ??
      []) as EnhancedModelMetadata["input_modalities"],
    output_modalities: (model.outputModalities ??
      []) as EnhancedModelMetadata["output_modalities"],
    runtime_features: (model.runtimeFeatures ??
      []) as EnhancedModelMetadata["runtime_features"],
    deployment_source:
      model.deploymentSource as EnhancedModelMetadata["deployment_source"],
    management_plane:
      model.managementPlane as EnhancedModelMetadata["management_plane"],
    canonical_model_id: model.canonicalModelId ?? null,
    provider_model_id: model.providerModelId ?? null,
    alias_source:
      (model.aliasSource as EnhancedModelMetadata["alias_source"]) ?? null,
    pricing: (model.pricing as EnhancedModelMetadata["pricing"]) ?? null,
    limits: model.limits as EnhancedModelMetadata["limits"],
    status: model.status as EnhancedModelMetadata["status"],
    release_date: model.releaseDate ?? null,
    is_latest: Boolean(model.isLatest),
    description: model.description ?? null,
    source: model.source as EnhancedModelMetadata["source"],
    created_at: model.createdAt,
    updated_at: model.updatedAt,
  };
}

function toRegistryModel(model: Model): EnhancedModelMetadata {
  const route = decodeModelRouteSelector(model.id);
  if (!route) {
    throw new Error(
      `App Server model/list returned invalid model id: ${model.id}`,
    );
  }
  if (route.providerId !== model.providerId) {
    throw new Error(
      `App Server model/list returned mismatched provider for ${model.id}: ${model.providerId}`,
    );
  }
  const snapshot = model.capabilitySnapshot;
  if (!snapshot.capabilities) {
    throw new Error(
      `App Server model/list returned missing capabilities for ${model.id}`,
    );
  }
  const reasoningOptions = model.supportedReasoningEfforts.map((option) => ({
    id: option.reasoningEffort,
    value: option.reasoningEffort,
    label: option.description,
    description: option.description,
  }));
  const reasoningEffort = reasoningEffortSupportFromUnknown(
    snapshot.capabilities.reasoningEffort,
  );
  const taskFamilies = (snapshot.taskFamilies ??
    []) as EnhancedModelMetadata["task_families"];
  const inputModalities = (snapshot.inputModalities ??
    []) as EnhancedModelMetadata["input_modalities"];
  const outputModalities = (snapshot.outputModalities ??
    []) as EnhancedModelMetadata["output_modalities"];
  const runtimeFeatures = (snapshot.runtimeFeatures ??
    []) as EnhancedModelMetadata["runtime_features"];
  const policyInput = {
    serviceTiers: model.serviceTiers,
    defaultServiceTier: model.defaultServiceTier,
    defaultReasoningLevel: model.defaultReasoningEffort,
    supportedReasoningLevels: reasoningOptions,
    inputModalities,
  };

  return {
    id: route.modelId,
    display_name: model.displayName,
    provider_id: model.providerId,
    provider_name: model.providerId,
    is_default: model.isDefault,
    family: null,
    tier: "pro",
    capabilities: {
      vision: snapshot.capabilities.vision,
      tools: snapshot.capabilities.tools,
      streaming: snapshot.capabilities.streaming,
      json_mode: snapshot.capabilities.jsonMode,
      function_calling: snapshot.capabilities.functionCalling,
      reasoning: snapshot.capabilities.reasoning,
      reasoning_effort: reasoningEffort,
    },
    capability_provenance: capabilityProvenanceFromSnapshot(
      snapshot.source,
      model.id,
    ),
    picker_policy: buildCatalogModelPickerPolicy(model.hidden, policyInput),
    reasoning_policy: buildModelReasoningPolicy(policyInput),
    input_modality_policy: buildModelInputModalityPolicy(policyInput),
    task_families: taskFamilies,
    input_modalities: inputModalities,
    output_modalities: outputModalities,
    runtime_features: runtimeFeatures,
    multi_agent_version: model.multiAgentVersion ?? null,
    deployment_source: "user_cloud",
    management_plane: "local_settings",
    canonical_model_id: route.modelId,
    provider_model_id: model.model,
    alias_source: null,
    pricing: null,
    limits: {
      context_length: model.contextWindow ?? null,
      max_output_tokens: model.maxOutputTokens ?? null,
      requests_per_minute: null,
      tokens_per_minute: null,
    },
    status: "active",
    release_date: null,
    is_latest: false,
    description: model.description || null,
    source: "api",
    created_at: 0,
    updated_at: 0,
  };
}

function assertModelInfos(
  models: ModelInfo[] | null | undefined,
  method: string,
): EnhancedModelMetadata[] {
  if (!Array.isArray(models)) {
    throw new Error(`App Server ${method} did not return models`);
  }
  return models.map(toSnakeModelInfo);
}

function assertCatalogModels(
  models: Model[] | null | undefined,
): EnhancedModelMetadata[] {
  if (!Array.isArray(models)) {
    throw new Error("App Server model/list did not return data");
  }
  return models.map(toRegistryModel);
}

async function requestModelRegistryAppServer<T>(
  method: string,
  params: unknown,
  appServerClient: ModelRegistryAppServerClient = new AppServerClient(),
): Promise<T> {
  const response = await appServerClient.request<T>(method, params);
  return response.result;
}

async function readModelsFromAppServer(
  includeHidden = false,
): Promise<EnhancedModelMetadata[]> {
  const models: EnhancedModelMetadata[] = [];
  const seenCursors = new Set<string>();
  let cursor: string | null = null;

  do {
    const params: ModelListParams = {
      ...(cursor ? { cursor } : {}),
      ...(includeHidden ? { includeHidden: true } : {}),
    };
    const response = await requestModelRegistryAppServer<ModelListResponse>(
      METHOD_MODEL_LIST,
      params,
    );
    models.push(...assertCatalogModels(response.data));
    cursor = response.nextCursor ?? null;
    if (cursor && seenCursors.has(cursor)) {
      throw new Error(`App Server model/list repeated cursor: ${cursor}`);
    }
    if (cursor) {
      seenCursors.add(cursor);
    }
  } while (cursor);

  return models;
}

export async function readModelProviderCapabilities(
  appServerClient: ModelRegistryAppServerClient = new AppServerClient(),
): Promise<ModelProviderCapabilities> {
  const response = await requestModelRegistryAppServer<unknown>(
    METHOD_MODEL_PROVIDER_CAPABILITIES_READ,
    {},
    appServerClient,
  );
  return parseModelProviderCapabilities(response);
}

interface ModelRegistryQueryOptions {
  forceRefresh?: boolean;
  includeHidden?: boolean;
}

export interface FetchProviderModelsResult {
  models: EnhancedModelMetadata[];
  source: "Api" | "Error";
  error: string | null;
  request_url?: string | null;
  diagnostic_hint?: string | null;
  error_kind?:
    | "not_found"
    | "unauthorized"
    | "forbidden"
    | "network"
    | "invalid_response"
    | "other"
    | null;
  should_prompt_error?: boolean;
  from_cache?: boolean;
}

export function normalizeFetchProviderModelsSource(
  result: Pick<FetchProviderModelsResult, "source" | "models" | "error">,
): FetchProviderModelsResult["source"] {
  return result.source;
}

const modelRegistryCache = new Map<boolean, EnhancedModelMetadata[]>();
const modelRegistryLoadingPromises = new Map<
  boolean,
  Promise<EnhancedModelMetadata[]>
>();
let allAliasConfigsCache: Record<string, ProviderAliasConfig> | null = null;
let allAliasConfigsLoadingPromise: Promise<
  Record<string, ProviderAliasConfig>
> | null = null;
const providerAliasConfigCache = new Map<string, ProviderAliasConfig | null>();
const providerAliasConfigLoadingPromises = new Map<
  string,
  Promise<ProviderAliasConfig | null>
>();

function cloneValue<T>(value: T): T {
  if (typeof structuredClone === "function") {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value)) as T;
}

function normalizeProviderKey(provider: string): string {
  return provider.trim();
}

function invalidateAliasConfigCache(): void {
  allAliasConfigsCache = null;
  allAliasConfigsLoadingPromise = null;
  providerAliasConfigCache.clear();
  providerAliasConfigLoadingPromises.clear();
}

export function invalidateModelRegistryCache(): void {
  modelRegistryCache.clear();
  modelRegistryLoadingPromises.clear();
  invalidateAliasConfigCache();
}

export function subscribeModelRegistryUpdates(
  onUpdate: (update: { generation: number; providerId: string | null }) => void,
): () => void {
  return subscribeAppServerNotifications({
    onNotifications: (notifications) => {
      let update: ReturnType<typeof modelListUpdatedServerNotification>;
      for (const notification of notifications) {
        update = modelListUpdatedServerNotification(notification) ?? update;
      }
      if (!update) {
        return;
      }
      invalidateModelRegistryCache();
      onUpdate({
        generation: update.params.generation,
        providerId: update.params.providerId ?? null,
      });
    },
  });
}

function assertModelProviderIds(
  response: ModelProviderListResponse | null | undefined,
): string[] {
  if (!response || typeof response !== "object") {
    throw new Error("App Server modelProvider/list did not return providers");
  }
  if (!Array.isArray(response.providers)) {
    throw new Error("App Server modelProvider/list did not return providers");
  }

  return Array.from(
    new Set(
      response.providers
        .map((provider: ProviderInfo | ModelProviderIdRecord) =>
          typeof provider.id === "string" ? provider.id.trim() : "",
        )
        .filter((providerId) => providerId.length > 0),
    ),
  );
}

function modelMatchesSearchQuery(
  model: EnhancedModelMetadata,
  normalizedQuery: string,
): boolean {
  if (!normalizedQuery) {
    return true;
  }

  return [
    model.id,
    model.display_name,
    model.provider_id,
    model.provider_name,
  ].some((value) => value?.toLowerCase().includes(normalizedQuery));
}

function modelPreferenceMutationUnavailable(operation: string): never {
  throw new Error(
    `${operation} 尚未接入 App Server model preference current 写链；旧 Tauri 模型注册表业务命令已退役。`,
  );
}

/**
 * 获取所有模型
 */
export async function getModelRegistry(
  options: ModelRegistryQueryOptions = {},
): Promise<EnhancedModelMetadata[]> {
  const includeHidden = options.includeHidden === true;
  if (options.forceRefresh) {
    modelRegistryCache.delete(includeHidden);
    modelRegistryLoadingPromises.delete(includeHidden);
  }

  const cached = modelRegistryCache.get(includeHidden);
  if (cached) {
    return cloneValue(cached);
  }

  let loadingPromise = modelRegistryLoadingPromises.get(includeHidden);
  if (!loadingPromise) {
    loadingPromise = readModelsFromAppServer(includeHidden)
      .then((models) => {
        const snapshot = cloneValue(models);
        modelRegistryCache.set(includeHidden, snapshot);
        return snapshot;
      })
      .finally(() => {
        modelRegistryLoadingPromises.delete(includeHidden);
      });
    modelRegistryLoadingPromises.set(includeHidden, loadingPromise);
  }

  return cloneValue(await loadingPromise);
}

export async function getModelRegistryProviderIds(): Promise<string[]> {
  const response =
    await requestModelRegistryAppServer<ModelProviderListResponse>(
      METHOD_MODEL_PROVIDER_LIST,
      {},
    );
  return assertModelProviderIds(response);
}

/**
 * 刷新模型注册表（清空已下线的本地模型注册缓存）
 * @returns 当前模型数量
 */
export async function refreshModelRegistry(): Promise<number> {
  invalidateModelRegistryCache();
  const models = await readModelsFromAppServer(false);
  modelRegistryCache.set(false, cloneValue(models));
  return models.length;
}

/**
 * 搜索模型
 * @param query 搜索关键词
 * @param limit 返回数量限制
 */
export async function searchModels(
  query: string,
  limit?: number,
): Promise<EnhancedModelMetadata[]> {
  const normalizedQuery = query.trim().toLowerCase();
  const models = await getModelRegistry();
  const filteredModels = models.filter((model) =>
    modelMatchesSearchQuery(model, normalizedQuery),
  );
  const safeLimit =
    typeof limit === "number" && Number.isFinite(limit)
      ? Math.max(0, Math.floor(limit))
      : undefined;
  return typeof safeLimit === "number"
    ? filteredModels.slice(0, safeLimit)
    : filteredModels;
}

/**
 * 获取用户模型偏好
 */
export async function getModelPreferences(): Promise<UserModelPreference[]> {
  const response =
    await requestModelRegistryAppServer<ModelPreferencesListAppServerResponse>(
      METHOD_MODEL_PREFERENCES_LIST,
      {},
    );
  if (!Array.isArray(response.preferences)) {
    throw new Error(
      "App Server modelPreferences/list did not return preferences",
    );
  }
  return response.preferences;
}

/**
 * 切换模型收藏状态
 * @param modelId 模型 ID
 * @returns 新的收藏状态
 */
export async function toggleModelFavorite(modelId: string): Promise<boolean> {
  void modelId;
  return modelPreferenceMutationUnavailable("toggleModelFavorite");
}

/**
 * 隐藏模型
 * @param modelId 模型 ID
 */
export async function hideModel(modelId: string): Promise<void> {
  void modelId;
  modelPreferenceMutationUnavailable("hideModel");
}

/**
 * 记录模型使用
 * @param modelId 模型 ID
 */
export async function recordModelUsage(modelId: string): Promise<void> {
  void modelId;
  modelPreferenceMutationUnavailable("recordModelUsage");
}

/**
 * 获取模型同步状态
 */
export async function getModelSyncState(): Promise<ModelSyncState> {
  const response =
    await requestModelRegistryAppServer<ModelSyncStateReadAppServerResponse>(
      METHOD_MODEL_SYNC_STATE_READ,
      {},
    );
  if (!response.syncState) {
    throw new Error("App Server modelSyncState/read did not return syncState");
  }
  return response.syncState;
}

export async function fetchProviderModelsAuto(
  providerId: string,
): Promise<FetchProviderModelsResult> {
  const response =
    await requestModelRegistryAppServer<ModelProviderFetchModelsResponse>(
      METHOD_MODEL_PROVIDER_FETCH_MODELS,
      { providerId },
    );
  return {
    models: assertModelInfos(response.models, "modelProvider/fetchModels"),
    source: response.source === "Api" ? "Api" : "Error",
    error: response.error ?? null,
    request_url: response.requestUrl ?? null,
    diagnostic_hint: response.diagnosticHint ?? null,
    error_kind:
      (response.errorKind as FetchProviderModelsResult["error_kind"]) ?? null,
    should_prompt_error: Boolean(response.shouldPromptError),
    from_cache: Boolean(response.fromCache),
  };
}

/**
 * 获取指定 Provider 的别名配置
 * 用于获取中转服务或协议转换相关的模型别名映射
 * @param provider Provider ID
 */
export async function getProviderAliasConfig(
  provider: string,
  options: ModelRegistryQueryOptions = {},
): Promise<ProviderAliasConfig | null> {
  const normalizedProvider = normalizeProviderKey(provider);
  if (!normalizedProvider) {
    return null;
  }

  if (options.forceRefresh) {
    invalidateAliasConfigCache();
  }

  if (allAliasConfigsCache) {
    return cloneValue(allAliasConfigsCache[normalizedProvider] ?? null);
  }

  if (providerAliasConfigCache.has(normalizedProvider)) {
    return cloneValue(providerAliasConfigCache.get(normalizedProvider) ?? null);
  }

  const existingPromise =
    providerAliasConfigLoadingPromises.get(normalizedProvider);
  if (existingPromise) {
    return cloneValue(await existingPromise);
  }

  const loadingPromise =
    requestModelRegistryAppServer<ModelProviderAliasReadAppServerResponse>(
      METHOD_MODEL_PROVIDER_ALIAS_READ,
      { provider: normalizedProvider },
    )
      .then((config) => {
        const snapshot = config.config ? cloneValue(config.config) : null;
        providerAliasConfigCache.set(normalizedProvider, snapshot);
        return snapshot;
      })
      .finally(() => {
        providerAliasConfigLoadingPromises.delete(normalizedProvider);
      });

  providerAliasConfigLoadingPromises.set(normalizedProvider, loadingPromise);
  return cloneValue(await loadingPromise);
}

/**
 * 获取所有 Provider 的别名配置
 */
export async function getAllAliasConfigs(): Promise<
  Record<string, ProviderAliasConfig>
> {
  return getAllAliasConfigsCached();
}

async function getAllAliasConfigsCached(
  options: ModelRegistryQueryOptions = {},
): Promise<Record<string, ProviderAliasConfig>> {
  if (options.forceRefresh) {
    invalidateAliasConfigCache();
  }

  if (allAliasConfigsCache) {
    return cloneValue(allAliasConfigsCache);
  }

  if (!allAliasConfigsLoadingPromise) {
    allAliasConfigsLoadingPromise =
      requestModelRegistryAppServer<ModelProviderAliasListAppServerResponse>(
        METHOD_MODEL_PROVIDER_ALIAS_LIST,
        {},
      )
        .then((configs) => {
          if (!configs.configs) {
            throw new Error(
              "App Server modelProviderAlias/list did not return configs",
            );
          }
          allAliasConfigsCache = cloneValue(configs.configs);
          providerAliasConfigCache.clear();
          Object.entries(allAliasConfigsCache).forEach(([key, value]) => {
            providerAliasConfigCache.set(key, cloneValue(value));
          });
          return allAliasConfigsCache;
        })
        .finally(() => {
          allAliasConfigsLoadingPromise = null;
        });
  }

  return cloneValue(await allAliasConfigsLoadingPromise);
}

/**
 * 模型注册表 API 对象
 */
export const modelRegistryApi = {
  getModelRegistry,
  readModelProviderCapabilities,
  getModelRegistryProviderIds,
  refreshModelRegistry,
  searchModels,
  getModelPreferences,
  toggleModelFavorite,
  hideModel,
  recordModelUsage,
  getModelSyncState,
  subscribeModelRegistryUpdates,
  fetchProviderModelsAuto,
  normalizeFetchProviderModelsSource,
  getProviderAliasConfig,
  getAllAliasConfigs: getAllAliasConfigsCached,
};
