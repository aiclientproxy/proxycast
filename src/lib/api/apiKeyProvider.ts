/**
 * @file API Key Provider API 模块
 * @description 封装 API Key Provider 相关的 Desktop Host / App Server 命令调用
 * @module lib/api/apiKeyProvider
 *
 * **Feature: provider-ui-refactor**
 * **Validates: Requirements 9.1**
 */

import { AppServerClient } from "@/lib/api/appServer";
import type { ProviderDeclaredPromptCacheMode } from "@/lib/types/provider";
import {
  METHOD_MODEL_PROVIDER_CATALOG_LIST,
  METHOD_MODEL_PROVIDER_CONFIG_EXPORT,
  METHOD_MODEL_PROVIDER_CONFIG_IMPORT,
  METHOD_MODEL_PROVIDER_CREATE,
  METHOD_MODEL_PROVIDER_DELETE,
  METHOD_MODEL_PROVIDER_KEY_CREATE,
  METHOD_MODEL_PROVIDER_KEY_DELETE,
  METHOD_MODEL_PROVIDER_KEY_UPDATE,
  METHOD_MODEL_PROVIDER_LIST,
  METHOD_MODEL_PROVIDER_READ,
  METHOD_MODEL_PROVIDER_SORT_ORDERS_UPDATE,
  METHOD_MODEL_PROVIDER_TEST_CHAT,
  METHOD_MODEL_PROVIDER_TEST_CONNECTION,
  METHOD_MODEL_PROVIDER_UI_STATE_READ,
  METHOD_MODEL_PROVIDER_UI_STATE_WRITE,
  METHOD_MODEL_PROVIDER_UPDATE,
  type ModelProviderCatalogListResponse as AppServerModelProviderCatalogListResponse,
  type ModelProviderConfigExportResponse as AppServerModelProviderConfigExportResponse,
  type ModelProviderConfigImportResponse as AppServerModelProviderConfigImportResponse,
  type ModelProviderDeleteResponse as AppServerModelProviderDeleteResponse,
  type ModelProviderKeyDeleteResponse as AppServerModelProviderKeyDeleteResponse,
  type ModelProviderKeyWriteResponse as AppServerModelProviderKeyWriteResponse,
  type ModelProviderListResponse as AppServerModelProviderListResponse,
  type ModelProviderReadResponse as AppServerModelProviderReadResponse,
  type ModelProviderTestChatResponse as AppServerModelProviderTestChatResponse,
  type ModelProviderTestConnectionResponse as AppServerModelProviderTestConnectionResponse,
  type ModelProviderUiStateReadResponse as AppServerModelProviderUiStateReadResponse,
  type ModelProviderWriteResponse as AppServerModelProviderWriteResponse,
  type ProviderInfo,
  type ProviderKeyInfo,
  type ProviderModelConfig as AppServerProviderModelConfig,
} from "../../../packages/app-server-client/src/protocol";

type ApiKeyProviderAppServerClient = Pick<AppServerClient, "request">;

interface ProviderQueryOptions {
  forceRefresh?: boolean;
  appServerClient?: ApiKeyProviderAppServerClient;
}

let providersCache: ProviderWithKeysDisplay[] | null = null;
let providersLoadingPromise: Promise<ProviderWithKeysDisplay[]> | null = null;

export type ProviderModelConfig = AppServerProviderModelConfig;

function cloneProviderModel(model: ProviderModelConfig): ProviderModelConfig {
  const capability = model.capability;
  return {
    ...model,
    capability: capability
      ? {
          ...capability,
          taskFamilies: [...(capability.taskFamilies ?? [])],
          inputModalities: [...(capability.inputModalities ?? [])],
          outputModalities: [...(capability.outputModalities ?? [])],
          runtimeFeatures: [...(capability.runtimeFeatures ?? [])],
          capabilities: capability.capabilities
            ? {
                ...capability.capabilities,
                reasoningEffort: capability.capabilities.reasoningEffort
                  ? {
                      ...capability.capabilities.reasoningEffort,
                      levels: [
                        ...(capability.capabilities.reasoningEffort.levels ?? []),
                      ],
                      options: (
                        capability.capabilities.reasoningEffort.options ?? []
                      ).map((option) => ({ ...option })),
                    }
                  : capability.capabilities.reasoningEffort,
              }
            : capability.capabilities,
        }
      : capability,
  };
}

export function providerModelIds(
  models: readonly ProviderModelConfig[] | null | undefined,
): string[] {
  return Array.from(
    new Set(
      (models ?? [])
        .map((model) => model.id.trim())
        .filter((modelId) => modelId.length > 0),
    ),
  );
}

export function reconcileProviderModels(
  existingModels: readonly ProviderModelConfig[] | null | undefined,
  modelIds: readonly string[],
): ProviderModelConfig[] {
  const existingById = new Map(
    (existingModels ?? []).map((model) => [
      model.id.trim().toLowerCase(),
      model,
    ]),
  );
  const seen = new Set<string>();
  const models: ProviderModelConfig[] = [];
  for (const value of modelIds) {
    const id = value.trim();
    const normalized = id.toLowerCase();
    if (!id || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    const existing = existingById.get(normalized);
    models.push(existing ? cloneProviderModel(existing) : { id });
  }
  return models;
}

async function requestApiKeyProviderAppServer<T>(
  method: string,
  params: unknown,
  appServerClient: ApiKeyProviderAppServerClient = new AppServerClient(),
): Promise<T> {
  const response = await appServerClient.request<T>(method, params);
  return response.result;
}

function cloneProviderList(
  providers: ProviderWithKeysDisplay[],
): ProviderWithKeysDisplay[] {
  return providers.map((provider) => ({
    ...provider,
    api_keys: Array.isArray(provider.api_keys)
      ? provider.api_keys.map((apiKey) => ({ ...apiKey }))
      : [],
    models: Array.isArray(provider.models)
      ? provider.models.map(cloneProviderModel)
      : [],
    prompt_cache_mode: provider.prompt_cache_mode ?? null,
  }));
}

function toProviderDisplay(provider: ProviderInfo): ProviderDisplay {
  return {
    id: provider.id,
    name: provider.name,
    type: provider.providerType,
    api_host: provider.apiHost,
    is_system: provider.isSystem,
    group: provider.group,
    enabled: provider.enabled,
    sort_order: provider.sortOrder,
    api_version: provider.apiVersion ?? undefined,
    project: provider.project ?? undefined,
    location: provider.location ?? undefined,
    region: provider.region ?? undefined,
    models: (provider.models ?? []).map(cloneProviderModel),
    prompt_cache_mode:
      (provider.promptCacheMode as ProviderDeclaredPromptCacheMode | null) ??
      null,
    api_key_count: provider.apiKeyCount,
    created_at: provider.createdAt ?? "",
    updated_at: provider.updatedAt ?? "",
  };
}

function toApiKeyDisplay(apiKey: ProviderKeyInfo): ApiKeyDisplay {
  return {
    id: apiKey.id,
    provider_id: apiKey.providerId,
    api_key_masked: apiKey.apiKeyMasked,
    alias: apiKey.alias ?? undefined,
    enabled: apiKey.enabled,
    usage_count: apiKey.usageCount,
    error_count: apiKey.errorCount,
    last_used_at: apiKey.lastUsedAt ?? undefined,
    created_at: apiKey.createdAt,
  };
}

function toProviderWithKeysDisplay(
  provider: ProviderInfo,
): ProviderWithKeysDisplay {
  return {
    ...toProviderDisplay(provider),
    api_keys: (provider.apiKeys ?? []).map(toApiKeyDisplay),
  };
}

function toSystemProviderCatalogItem(
  provider: ProviderInfo,
): SystemProviderCatalogItem {
  return {
    id: provider.id,
    name: provider.name,
    type: provider.providerType,
    api_host: provider.apiHost,
    group: provider.group,
    sort_order: provider.sortOrder,
    api_version: provider.apiVersion ?? undefined,
    legacy_ids: [...(provider.legacyIds ?? [])],
  };
}

function normalizeModelProviderListResponse(
  response: AppServerModelProviderListResponse | null | undefined,
): ProviderWithKeysDisplay[] {
  if (!response || typeof response !== "object") {
    throw new Error("App Server modelProvider/list did not return providers");
  }

  if (!Array.isArray(response.providers)) {
    throw new Error("App Server modelProvider/list did not return providers");
  }

  return response.providers.map(toProviderWithKeysDisplay);
}

function normalizeModelProviderCatalogListResponse(
  response: AppServerModelProviderCatalogListResponse | null | undefined,
): SystemProviderCatalogItem[] {
  if (!response || typeof response !== "object") {
    throw new Error(
      "App Server modelProvider/catalog/list did not return providers",
    );
  }

  if (!Array.isArray(response.providers)) {
    throw new Error(
      "App Server modelProvider/catalog/list did not return providers",
    );
  }

  return response.providers.map(toSystemProviderCatalogItem);
}

function normalizeProviderReadResponse(
  response: AppServerModelProviderReadResponse | null | undefined,
): ProviderWithKeysDisplay | null {
  if (!response || typeof response !== "object") {
    throw new Error("App Server modelProvider/read did not return provider");
  }
  return response.provider
    ? toProviderWithKeysDisplay(response.provider)
    : null;
}

function normalizeProviderWriteResponse(
  response: AppServerModelProviderWriteResponse | null | undefined,
): ProviderDisplay {
  if (!response || typeof response !== "object" || !response.provider) {
    throw new Error("App Server modelProvider write did not return provider");
  }
  return toProviderDisplay(response.provider);
}

function toCreateProviderParams(request: AddCustomProviderRequest) {
  return {
    name: request.name,
    providerType: request.type,
    apiHost: request.api_host,
    apiVersion: request.api_version,
    project: request.project,
    location: request.location,
    region: request.region,
    promptCacheMode: request.prompt_cache_mode,
  };
}

function toUpdateProviderParams(id: string, request: UpdateProviderRequest) {
  return {
    providerId: id,
    name: request.name,
    providerType: request.type,
    apiHost: request.api_host,
    enabled: request.enabled,
    sortOrder: request.sort_order,
    apiVersion: request.api_version,
    project: request.project,
    location: request.location,
    region: request.region,
    promptCacheMode: request.prompt_cache_mode,
    models: request.models,
  };
}

function normalizeProviderDeleteResponse(
  response: AppServerModelProviderDeleteResponse | null | undefined,
): boolean {
  if (!response || typeof response.deleted !== "boolean") {
    throw new Error("App Server modelProvider/delete did not return deleted");
  }
  return response.deleted;
}

function normalizeProviderKeyWriteResponse(
  response: AppServerModelProviderKeyWriteResponse | null | undefined,
): ApiKeyDisplay {
  if (!response || typeof response !== "object" || !response.key) {
    throw new Error("App Server modelProviderKey write did not return key");
  }
  return toApiKeyDisplay(response.key);
}

function normalizeProviderKeyDeleteResponse(
  response: AppServerModelProviderKeyDeleteResponse | null | undefined,
): boolean {
  if (!response || typeof response.deleted !== "boolean") {
    throw new Error(
      "App Server modelProviderKey/delete did not return deleted",
    );
  }
  return response.deleted;
}

function normalizeUiStateReadResponse(
  response: AppServerModelProviderUiStateReadResponse | null | undefined,
): string | null {
  if (!response || typeof response !== "object") {
    throw new Error(
      "App Server modelProviderUiState/read did not return value",
    );
  }
  return response.value ?? null;
}

function normalizeConfigExportResponse(
  response: AppServerModelProviderConfigExportResponse | null | undefined,
): string {
  if (!response || typeof response.configJson !== "string") {
    throw new Error(
      "App Server modelProviderConfig/export did not return configJson",
    );
  }
  return response.configJson;
}

function normalizeConfigImportResponse(
  response: AppServerModelProviderConfigImportResponse | null | undefined,
): ImportResult {
  if (!response || typeof response !== "object") {
    throw new Error(
      "App Server modelProviderConfig/import did not return result",
    );
  }
  return {
    success: Boolean(response.success),
    imported_providers: response.importedProviders ?? 0,
    imported_api_keys: response.importedApiKeys ?? 0,
    skipped_providers: response.skippedProviders ?? 0,
    errors: Array.isArray(response.errors) ? response.errors : [],
  };
}

function normalizeConnectionTestResponse(
  response: AppServerModelProviderTestConnectionResponse | null | undefined,
): ConnectionTestResult {
  if (!response || typeof response !== "object") {
    throw new Error(
      "App Server modelProvider/testConnection did not return result",
    );
  }
  return {
    success: Boolean(response.success),
    latency_ms: response.latencyMs ?? undefined,
    error: response.error ?? undefined,
    models: response.models ?? undefined,
  };
}

function normalizeChatTestResponse(
  response: AppServerModelProviderTestChatResponse | null | undefined,
): ChatTestResult {
  if (!response || typeof response !== "object") {
    throw new Error("App Server modelProvider/testChat did not return result");
  }
  return {
    success: Boolean(response.success),
    latency_ms: response.latencyMs ?? undefined,
    error: response.error ?? undefined,
    content: response.content ?? undefined,
    raw: response.raw ?? undefined,
  };
}

export function invalidateApiKeyProviderCache(): void {
  providersCache = null;
  providersLoadingPromise = null;
}

async function loadProviders(
  options: ProviderQueryOptions = {},
): Promise<ProviderWithKeysDisplay[]> {
  if (options.forceRefresh) {
    invalidateApiKeyProviderCache();
  }

  if (providersCache) {
    return cloneProviderList(providersCache);
  }

  if (!providersLoadingPromise) {
    providersLoadingPromise =
      requestApiKeyProviderAppServer<AppServerModelProviderListResponse>(
        METHOD_MODEL_PROVIDER_LIST,
        {},
        options.appServerClient,
      )
        .then(normalizeModelProviderListResponse)
        .then((providers) => {
          providersCache = cloneProviderList(providers);
          return providersCache;
        })
        .finally(() => {
          providersLoadingPromise = null;
        });
  }

  return cloneProviderList(await providersLoadingPromise);
}

async function invalidateAfterMutation<T>(promise: Promise<T>): Promise<T> {
  const result = await promise;
  invalidateApiKeyProviderCache();
  return result;
}

// ============================================================================
// 请求类型
// ============================================================================

/**
 * 添加自定义 Provider 请求
 */
export interface AddCustomProviderRequest {
  name: string;
  type: string;
  api_host: string;
  api_version?: string;
  project?: string;
  location?: string;
  region?: string;
  prompt_cache_mode?: ProviderDeclaredPromptCacheMode | null;
}

/**
 * 更新 Provider 请求
 */
export interface UpdateProviderRequest {
  name?: string;
  /** Provider 类型（系统/自定义 Provider 均可修改） */
  type?: string;
  api_host?: string;
  enabled?: boolean;
  sort_order?: number;
  api_version?: string;
  project?: string;
  location?: string;
  region?: string;
  prompt_cache_mode?: ProviderDeclaredPromptCacheMode | null;
  /** Provider 模型配置 */
  models?: ProviderModelConfig[];
}

/**
 * 添加 API Key 请求
 */
export interface AddApiKeyRequest {
  provider_id: string;
  api_key: string;
  alias?: string;
  /** 单 Key 设置入口使用：替换当前 Provider 的旧 Key，而不是追加轮询 Key。 */
  replace_existing?: boolean;
}

// ============================================================================
// 响应类型
// ============================================================================

/**
 * Provider 显示数据（用于前端）
 */
export interface ProviderDisplay {
  id: string;
  name: string;
  type: string;
  api_host: string;
  is_system: boolean;
  group: string;
  enabled: boolean;
  sort_order: number;
  api_version?: string;
  project?: string;
  location?: string;
  region?: string;
  /** Provider 模型配置 */
  models?: ProviderModelConfig[];
  /** Provider 当前生效的 Prompt Cache 模式（已包含官方兼容端点推断与类型回退） */
  prompt_cache_mode?: ProviderDeclaredPromptCacheMode | null;
  api_key_count: number;
  created_at: string;
  updated_at: string;
}

/**
 * API Key 显示数据（用于前端，掩码显示）
 */
export interface ApiKeyDisplay {
  id: string;
  provider_id: string;
  /** 掩码后的 API Key */
  api_key_masked: string;
  alias?: string;
  enabled: boolean;
  usage_count: number;
  error_count: number;
  last_used_at?: string;
  created_at: string;
}

/**
 * Provider 完整显示数据（包含 API Keys）
 */
export interface ProviderWithKeysDisplay extends ProviderDisplay {
  api_keys: ApiKeyDisplay[];
}

/**
 * 系统 Provider Catalog 条目
 * 用于前端动态构建系统 Provider 元信息
 */
export interface SystemProviderCatalogItem {
  id: string;
  name: string;
  type: string;
  api_host: string;
  group: string;
  sort_order: number;
  api_version?: string;
  /** 兼容旧版本前端/历史配置的别名 ID */
  legacy_ids: string[];
}

/**
 * 导入结果
 */
export interface ImportResult {
  success: boolean;
  imported_providers: number;
  imported_api_keys: number;
  skipped_providers: number;
  errors: string[];
}

export interface ChatTestResult {
  success: boolean;
  latency_ms?: number;
  error?: string;
  content?: string;
  raw?: string;
}

// ============================================================================
// API 函数
// ============================================================================

/**
 * API Key Provider API 封装
 */
export const apiKeyProviderApi = {
  /**
   * 获取系统 Provider Catalog
   */
  async getSystemProviderCatalog(): Promise<SystemProviderCatalogItem[]> {
    const response =
      await requestApiKeyProviderAppServer<AppServerModelProviderCatalogListResponse>(
        METHOD_MODEL_PROVIDER_CATALOG_LIST,
        {},
      );
    return normalizeModelProviderCatalogListResponse(response);
  },

  /**
   * 获取所有 API Key Provider（包含 API Keys）
   */
  async getProviders(
    options: ProviderQueryOptions = {},
  ): Promise<ProviderWithKeysDisplay[]> {
    return loadProviders(options);
  },

  /**
   * 获取单个 API Key Provider（包含 API Keys）
   */
  async getProvider(id: string): Promise<ProviderWithKeysDisplay | null> {
    const response =
      await requestApiKeyProviderAppServer<AppServerModelProviderReadResponse>(
        METHOD_MODEL_PROVIDER_READ,
        { providerId: id },
      );
    return normalizeProviderReadResponse(response);
  },

  /**
   * 添加自定义 Provider
   */
  async addCustomProvider(
    request: AddCustomProviderRequest,
  ): Promise<ProviderDisplay> {
    return invalidateAfterMutation(
      requestApiKeyProviderAppServer<AppServerModelProviderWriteResponse>(
        METHOD_MODEL_PROVIDER_CREATE,
        toCreateProviderParams(request),
      ).then(normalizeProviderWriteResponse),
    );
  },

  /**
   * 更新 Provider 配置
   */
  async updateProvider(
    id: string,
    request: UpdateProviderRequest,
  ): Promise<ProviderDisplay> {
    return invalidateAfterMutation(
      requestApiKeyProviderAppServer<AppServerModelProviderWriteResponse>(
        METHOD_MODEL_PROVIDER_UPDATE,
        toUpdateProviderParams(id, request),
      ).then(normalizeProviderWriteResponse),
    );
  },

  /**
   * 删除自定义 Provider
   */
  async deleteCustomProvider(id: string): Promise<boolean> {
    return invalidateAfterMutation(
      requestApiKeyProviderAppServer<AppServerModelProviderDeleteResponse>(
        METHOD_MODEL_PROVIDER_DELETE,
        { providerId: id },
      ).then(normalizeProviderDeleteResponse),
    );
  },

  /**
   * 添加 API Key
   */
  async addApiKey(request: AddApiKeyRequest): Promise<ApiKeyDisplay> {
    return invalidateAfterMutation(
      requestApiKeyProviderAppServer<AppServerModelProviderKeyWriteResponse>(
        METHOD_MODEL_PROVIDER_KEY_CREATE,
        {
          providerId: request.provider_id,
          apiKey: request.api_key,
          alias: request.alias,
          replaceExisting: request.replace_existing,
        },
      ).then(normalizeProviderKeyWriteResponse),
    );
  },

  /**
   * 删除 API Key
   */
  async deleteApiKey(keyId: string): Promise<boolean> {
    return invalidateAfterMutation(
      requestApiKeyProviderAppServer<AppServerModelProviderKeyDeleteResponse>(
        METHOD_MODEL_PROVIDER_KEY_DELETE,
        { keyId },
      ).then(normalizeProviderKeyDeleteResponse),
    );
  },

  /**
   * 切换 API Key 启用状态
   */
  async toggleApiKey(keyId: string, enabled: boolean): Promise<ApiKeyDisplay> {
    return invalidateAfterMutation(
      requestApiKeyProviderAppServer<AppServerModelProviderKeyWriteResponse>(
        METHOD_MODEL_PROVIDER_KEY_UPDATE,
        { keyId, enabled },
      ).then(normalizeProviderKeyWriteResponse),
    );
  },

  /**
   * 更新 API Key 别名
   */
  async updateApiKeyAlias(
    keyId: string,
    alias?: string,
  ): Promise<ApiKeyDisplay> {
    return invalidateAfterMutation(
      requestApiKeyProviderAppServer<AppServerModelProviderKeyWriteResponse>(
        METHOD_MODEL_PROVIDER_KEY_UPDATE,
        { keyId, alias },
      ).then(normalizeProviderKeyWriteResponse),
    );
  },

  /**
   * 获取 UI 状态
   */
  async getUiState(key: string): Promise<string | null> {
    const response =
      await requestApiKeyProviderAppServer<AppServerModelProviderUiStateReadResponse>(
        METHOD_MODEL_PROVIDER_UI_STATE_READ,
        { key },
      );
    return normalizeUiStateReadResponse(response);
  },

  /**
   * 设置 UI 状态
   */
  async setUiState(key: string, value: string): Promise<void> {
    await requestApiKeyProviderAppServer(METHOD_MODEL_PROVIDER_UI_STATE_WRITE, {
      key,
      value,
    });
  },

  /**
   * 批量更新 Provider 排序顺序
   * **Validates: Requirements 8.4**
   */
  async updateSortOrders(sortOrders: [string, number][]): Promise<void> {
    return invalidateAfterMutation(
      requestApiKeyProviderAppServer(METHOD_MODEL_PROVIDER_SORT_ORDERS_UPDATE, {
        sortOrders: sortOrders.map(([providerId, sortOrder]) => ({
          providerId,
          sortOrder,
        })),
      }).then(() => undefined),
    );
  },

  /**
   * 导出 Provider 配置
   */
  async exportConfig(includeKeys: boolean): Promise<string> {
    const response =
      await requestApiKeyProviderAppServer<AppServerModelProviderConfigExportResponse>(
        METHOD_MODEL_PROVIDER_CONFIG_EXPORT,
        { includeKeys },
      );
    return normalizeConfigExportResponse(response);
  },

  /**
   * 导入 Provider 配置
   */
  async importConfig(configJson: string): Promise<ImportResult> {
    return invalidateAfterMutation(
      requestApiKeyProviderAppServer<AppServerModelProviderConfigImportResponse>(
        METHOD_MODEL_PROVIDER_CONFIG_IMPORT,
        { configJson },
      ).then(normalizeConfigImportResponse),
    );
  },

  // ============================================================================
  // 连接测试 API
  // ============================================================================

  /**
   * 测试 API Key Provider 连接
   *
   * 方案 C 实现：
   * 1. 默认使用 /v1/models 端点测试
   * 2. 如果提供了 modelName，用该模型发送简单请求
   *
   * @param providerId Provider ID
   * @param modelName 可选的模型名称，用于发送测试请求
   */
  async testConnection(
    providerId: string,
    modelName?: string,
  ): Promise<ConnectionTestResult> {
    const response =
      await requestApiKeyProviderAppServer<AppServerModelProviderTestConnectionResponse>(
        METHOD_MODEL_PROVIDER_TEST_CONNECTION,
        { providerId, modelName },
      );
    return normalizeConnectionTestResponse(response);
  },

  async testChat(
    providerId: string,
    modelName: string | undefined,
    prompt: string,
  ): Promise<ChatTestResult> {
    const response =
      await requestApiKeyProviderAppServer<AppServerModelProviderTestChatResponse>(
        METHOD_MODEL_PROVIDER_TEST_CHAT,
        { providerId, modelName, prompt },
      );
    return normalizeChatTestResponse(response);
  },
};

// ============================================================================
// 连接测试类型
// ============================================================================

/**
 * 连接测试结果
 */
export interface ConnectionTestResult {
  /** 是否成功 */
  success: boolean;
  /** 延迟（毫秒） */
  latency_ms?: number;
  /** 错误信息 */
  error?: string;
  /** 测试方法 */
  method?: string;
  /** 模型列表（如果使用 models 端点测试） */
  models?: string[];
}
