/**
 * 模型注册表 Hook
 *
 * 提供模型数据管理、搜索、收藏等功能
 */

import { useState, useEffect, useCallback, useMemo } from "react";
import { modelRegistryApi } from "@/lib/api/modelRegistry";
import type {
  EnhancedModelMetadata,
  UserModelPreference,
  ModelTier,
} from "@/lib/types/modelRegistry";

interface UseModelRegistryOptions {
  /** 自动加载 */
  autoLoad?: boolean;
  /** 包含不在默认模型选择器中展示的目录项 */
  includeHidden?: boolean;
  /** 过滤的 Provider ID 列表 */
  providerFilter?: string[];
  /** 过滤的服务等级 */
  tierFilter?: ModelTier[];
  /** 只显示收藏 */
  favoritesOnly?: boolean;
}

interface UseModelRegistryReturn {
  /** 模型列表 */
  models: EnhancedModelMetadata[];
  /** 用户偏好 */
  preferences: Map<string, UserModelPreference>;
  /** 是否加载中 */
  loading: boolean;
  /** 错误信息 */
  error: string | null;
  /** 最后同步时间 */
  lastSyncAt: number | null;
  /** 刷新模型列表 */
  refresh: () => Promise<void>;
  /** 搜索模型 */
  search: (query: string) => EnhancedModelMetadata[];
  /** 切换收藏 */
  toggleFavorite: (modelId: string) => Promise<void>;
  /** 隐藏模型 */
  hideModel: (modelId: string) => Promise<void>;
  /** 获取模型详情 */
  getModel: (modelId: string) => EnhancedModelMetadata | undefined;
  /** 按 Provider 分组 */
  groupedByProvider: Map<string, EnhancedModelMetadata[]>;
  /** 按等级分组 */
  groupedByTier: Map<ModelTier, EnhancedModelMetadata[]>;
}

/**
 * 智能排序函数
 *
 * 修复：不再仅依赖 is_latest 标记，而是按版本号数字大小排序
 */
function sortModels(
  models: EnhancedModelMetadata[],
  preferences: Map<string, UserModelPreference>,
): EnhancedModelMetadata[] {
  return [...models].sort((a, b) => {
    const prefA = preferences.get(a.id);
    const prefB = preferences.get(b.id);

    // 1. 收藏优先
    if (prefA?.is_favorite && !prefB?.is_favorite) return -1;
    if (!prefA?.is_favorite && prefB?.is_favorite) return 1;

    // 2. 活跃状态优先
    if (a.status === "active" && b.status !== "active") return -1;
    if (a.status !== "active" && b.status === "active") return 1;

    // 3. 使用频率
    const usageA = prefA?.usage_count || 0;
    const usageB = prefB?.usage_count || 0;
    if (usageA !== usageB) return usageB - usageA;

    // 4. 版本号排序（数字大的优先）- 修复核心问题
    const versionA = extractVersionNumber(a.id);
    const versionB = extractVersionNumber(b.id);
    if (versionA !== null && versionB !== null && versionA !== versionB) {
      return versionB - versionA; // 数字大的排前面
    }

    // 5. 如果版本号相同或无法提取，则使用 is_latest 作为辅助
    if (a.is_latest && !b.is_latest) return -1;
    if (!a.is_latest && b.is_latest) return 1;

    // 6. 按名称字母序
    return a.display_name.localeCompare(b.display_name);
  });
}

/**
 * 从模型 ID 中提取版本号
 *
 * 支持的格式：
 * - claude-3-5-haiku-20241022 -> 20241022
 * - claude-haiku-4-5-20251001 -> 20251001
 * - gpt-4o-2024-11-20 -> 20241120
 * - claude-3.5-sonnet -> 3.5
 *
 * @param modelId 模型 ID
 * @returns 提取的版本号，如果无法提取则返回 null
 */
function extractVersionNumber(modelId: string): number | null {
  // 1. 优先匹配日期格式 (YYYYMMDD 或 YYYY-MM-DD)
  const dateMatch = modelId.match(/(\d{4})[-]?(\d{2})[-]?(\d{2})/);
  if (dateMatch) {
    const [, year, month, day] = dateMatch;
    return parseInt(year + month + day, 10);
  }

  // 2. 匹配版本号格式 (如 3.5, 4.5, 4-5)
  const versionMatch = modelId.match(/(\d+)[.-](\d+)/);
  if (versionMatch) {
    const [, major, minor] = versionMatch;
    return parseFloat(major + "." + minor);
  }

  // 3. 匹配单独的数字 (如 claude-3, gpt-4)
  const singleNumberMatch = modelId.match(/(\d+)(?![\d.-])/);
  if (singleNumberMatch) {
    return parseInt(singleNumberMatch[1], 10);
  }

  return null;
}

/**
 * 简单的模糊搜索
 */
function fuzzySearch(
  models: EnhancedModelMetadata[],
  query: string,
): EnhancedModelMetadata[] {
  if (!query.trim()) {
    return models;
  }

  const queryLower = query.toLowerCase();

  return models
    .map((model) => {
      let score = 0;

      // 精确匹配 ID（最高优先级）
      if (model.id.toLowerCase() === queryLower) {
        score += 1000;
      } else if (model.id.toLowerCase().startsWith(queryLower)) {
        // ID 以搜索词开头
        score += 500;
      } else if (model.id.toLowerCase().includes(queryLower)) {
        score += 100;
      }

      // 显示名称匹配
      if (model.display_name.toLowerCase().startsWith(queryLower)) {
        score += 80;
      } else if (model.display_name.toLowerCase().includes(queryLower)) {
        score += 40;
      }

      // Provider 匹配
      if (model.provider_id.toLowerCase() === queryLower) {
        score += 200;
      } else if (model.provider_name.toLowerCase().includes(queryLower)) {
        score += 30;
      }

      // 家族匹配
      if (model.family?.toLowerCase().includes(queryLower)) {
        score += 20;
      }

      // 只有在有匹配的情况下，才给最新版本和活跃状态加分
      if (score > 0) {
        if (model.is_latest) {
          score += 5;
        }
        if (model.status === "active") {
          score += 3;
        }
      }

      return { model, score };
    })
    .filter(({ score }) => score > 0)
    .sort((a, b) => b.score - a.score)
    .map(({ model }) => model);
}

export function useModelRegistry(
  options: UseModelRegistryOptions = {},
): UseModelRegistryReturn {
  const {
    autoLoad = true,
    includeHidden = false,
    providerFilter,
    tierFilter,
    favoritesOnly = false,
  } = options;

  const [allModels, setAllModels] = useState<EnhancedModelMetadata[]>([]);
  const [preferences, setPreferences] = useState<
    Map<string, UserModelPreference>
  >(new Map());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastSyncAt, setLastSyncAt] = useState<number | null>(null);

  // 加载模型数据（带重试机制）
  const loadModels = useCallback(
    async (forceRefresh = false) => {
      setLoading(true);
      setError(null);

      const maxRetries = 5;
      const retryDelay = 500; // 500ms

      for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
          const [models, prefs, syncState] = await Promise.all([
            modelRegistryApi.getModelRegistry(
              forceRefresh
                ? { forceRefresh: true, includeHidden }
                : { includeHidden },
            ),
            modelRegistryApi.getModelPreferences(),
            modelRegistryApi.getModelSyncState(),
          ]);

          setAllModels(models);
          setPreferences(new Map(prefs.map((p) => [p.model_id, p])));
          setLastSyncAt(syncState.last_sync_at);
          setLoading(false);
          return; // 成功，退出
        } catch (e) {
          const errorMsg = e instanceof Error ? e.message : String(e);

          // 如果是"服务未初始化"错误，且还有重试次数，则等待后重试
          if (errorMsg.includes("未初始化") && attempt < maxRetries - 1) {
            console.log(
              `[ModelRegistry] 服务未初始化，${retryDelay}ms 后重试 (${attempt + 1}/${maxRetries})`,
            );
            await new Promise((resolve) => setTimeout(resolve, retryDelay));
            continue;
          }

          // 其他错误或已达到最大重试次数
          setError(errorMsg);
          setLoading(false);
          return;
        }
      }
    },
    [includeHidden],
  );

  // 刷新（强制从内嵌资源重新加载）
  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const count = await modelRegistryApi.refreshModelRegistry();
      console.log(`[ModelRegistry] 刷新完成，加载了 ${count} 个模型`);
      await loadModels(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [loadModels]);

  // 过滤后的模型列表
  const models = useMemo(() => {
    let filtered = allModels;

    // Provider 过滤
    if (providerFilter && providerFilter.length > 0) {
      filtered = filtered.filter((m) => providerFilter.includes(m.provider_id));
    }

    // 等级过滤
    if (tierFilter && tierFilter.length > 0) {
      filtered = filtered.filter((m) =>
        tierFilter.includes(m.tier as ModelTier),
      );
    }

    // 收藏过滤
    if (favoritesOnly) {
      filtered = filtered.filter((m) => preferences.get(m.id)?.is_favorite);
    }

    // 隐藏过滤
    filtered = filtered.filter((m) => !preferences.get(m.id)?.is_hidden);

    // 智能排序
    return sortModels(filtered, preferences);
  }, [allModels, providerFilter, tierFilter, favoritesOnly, preferences]);

  // 模糊搜索
  const search = useCallback(
    (query: string): EnhancedModelMetadata[] => {
      return fuzzySearch(models, query);
    },
    [models],
  );

  // 切换收藏
  const toggleFavorite = useCallback(async (modelId: string) => {
    try {
      const newState = await modelRegistryApi.toggleModelFavorite(modelId);
      setPreferences((prev) => {
        const newPrefs = new Map(prev);
        const current = newPrefs.get(modelId);
        if (current) {
          newPrefs.set(modelId, {
            ...current,
            is_favorite: newState,
          });
        } else {
          newPrefs.set(modelId, {
            model_id: modelId,
            is_favorite: newState,
            is_hidden: false,
            custom_alias: null,
            usage_count: 0,
            last_used_at: null,
            created_at: Date.now() / 1000,
            updated_at: Date.now() / 1000,
          });
        }
        return newPrefs;
      });
    } catch (e) {
      console.error("Failed to toggle favorite:", e);
    }
  }, []);

  // 隐藏模型
  const hideModel = useCallback(async (modelId: string) => {
    try {
      await modelRegistryApi.hideModel(modelId);
      setPreferences((prev) => {
        const newPrefs = new Map(prev);
        const current = newPrefs.get(modelId);
        if (current) {
          newPrefs.set(modelId, { ...current, is_hidden: true });
        }
        return newPrefs;
      });
    } catch (e) {
      console.error("Failed to hide model:", e);
    }
  }, []);

  // 获取单个模型
  const getModel = useCallback(
    (modelId: string) => {
      return allModels.find((m) => m.id === modelId);
    },
    [allModels],
  );

  // 按 Provider 分组
  const groupedByProvider = useMemo(() => {
    const groups = new Map<string, EnhancedModelMetadata[]>();
    for (const model of models) {
      const existing = groups.get(model.provider_id) || [];
      existing.push(model);
      groups.set(model.provider_id, existing);
    }
    return groups;
  }, [models]);

  // 按等级分组
  const groupedByTier = useMemo(() => {
    const groups = new Map<ModelTier, EnhancedModelMetadata[]>();
    for (const model of models) {
      const tier = model.tier as ModelTier;
      const existing = groups.get(tier) || [];
      existing.push(model);
      groups.set(tier, existing);
    }
    return groups;
  }, [models]);

  // 自动加载并在模型目录变更后刷新已挂载视图
  useEffect(() => {
    if (!autoLoad) {
      return;
    }

    const unsubscribe = modelRegistryApi.subscribeModelRegistryUpdates(() => {
      void loadModels(true);
    });
    void loadModels();
    return unsubscribe;
  }, [autoLoad, loadModels]);

  return {
    models,
    preferences,
    loading,
    error,
    lastSyncAt,
    refresh,
    search,
    toggleFavorite,
    hideModel,
    getModel,
    groupedByProvider,
    groupedByTier,
  };
}
