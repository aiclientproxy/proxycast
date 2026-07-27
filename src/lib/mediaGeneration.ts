/**
 * @file 视频与语音生成选择辅助
 * @description 统一视频 / 语音 Provider 识别、模型解析与默认选择策略
 * @module lib/mediaGeneration
 */

export interface MediaProviderCandidate {
  id: string;
  type?: string;
  models?: string[];
}

export interface MediaGenerationPreference {
  preferredProviderId?: string;
  preferredModelId?: string;
  allowFallback?: boolean;
}

export interface MediaGenerationDefaults {
  image?: MediaGenerationPreference;
  video?: MediaGenerationPreference;
  voice?: MediaGenerationPreference;
}

export interface ResolvedMediaGenerationPreference {
  preferredProviderId?: string;
  preferredModelId?: string;
  allowFallback: boolean;
  source: "project" | "global" | "auto";
}

export type VideoModelPreset = "keling" | "jimeng" | "wan-2-5";
export type VideoModelVersion = "v2-1-master" | "v2" | "v1-6";

const VIDEO_MODEL_PRESETS: Record<string, string[]> = {
  doubao: ["seedance-1-5-pro-251215", "seedance-1-5-lite-250428"],
  volcengine: ["seedance-1-5-pro-251215", "seedance-1-5-lite-250428"],
  dashscope: ["wanx2.1-t2v-turbo", "wanx2.1-kf2v-plus"],
  alibaba: ["wanx2.1-t2v-turbo", "wanx2.1-kf2v-plus"],
  qwen: ["wanx2.1-t2v-turbo", "wanx2.1-kf2v-plus"],
  sora: ["sora-2", "sora-2-pro"],
  openai: ["sora-2", "sora-2-pro"],
  veo: ["veo-3.1"],
  google: ["veo-3.1"],
  vertex: ["veo-3.1"],
  kling: ["kling-2.6"],
  minimax: ["minimax-hailuo-2.3", "minimax-hailuo-02"],
  hailuo: ["minimax-hailuo-2.3", "minimax-hailuo-02"],
  runway: ["runway-gen-4-turbo"],
};

function normalizeMaybeString(value?: string | null): string | undefined {
  const normalized = value?.trim();
  return normalized ? normalized : undefined;
}

export function normalizeMediaGenerationPreference(
  preference?: MediaGenerationPreference | null,
): MediaGenerationPreference {
  return {
    preferredProviderId: normalizeMaybeString(preference?.preferredProviderId),
    preferredModelId: normalizeMaybeString(preference?.preferredModelId),
    allowFallback: preference?.allowFallback,
  };
}

export function hasMediaGenerationPreferenceOverride(
  preference?: MediaGenerationPreference | null,
): boolean {
  const normalized = normalizeMediaGenerationPreference(preference);
  return Boolean(
    normalized.preferredProviderId ||
    normalized.preferredModelId ||
    normalized.allowFallback === false,
  );
}

export function buildPersistedMediaGenerationPreference(
  preference?: MediaGenerationPreference | null,
): MediaGenerationPreference | undefined {
  const normalized = normalizeMediaGenerationPreference(preference);

  if (!normalized.preferredProviderId) {
    normalized.preferredModelId = undefined;
  }

  if (!hasMediaGenerationPreferenceOverride(normalized)) {
    return undefined;
  }

  return {
    preferredProviderId: normalized.preferredProviderId,
    preferredModelId: normalized.preferredModelId,
    allowFallback: normalized.allowFallback ?? true,
  };
}

export function resolveMediaGenerationPreference(
  projectPreference?: MediaGenerationPreference | null,
  globalPreference?: MediaGenerationPreference | null,
): ResolvedMediaGenerationPreference {
  const normalizedProject =
    normalizeMediaGenerationPreference(projectPreference);
  const normalizedGlobal = normalizeMediaGenerationPreference(globalPreference);

  const projectSelectsModel = Boolean(
    normalizedProject.preferredProviderId || normalizedProject.preferredModelId,
  );
  const globalSelectsModel = Boolean(
    normalizedGlobal.preferredProviderId || normalizedGlobal.preferredModelId,
  );

  const preferredProviderId =
    normalizedProject.preferredProviderId ??
    normalizedGlobal.preferredProviderId;

  const preferredModelId = normalizedProject.preferredProviderId
    ? normalizedProject.preferredModelId
    : (normalizedProject.preferredModelId ?? normalizedGlobal.preferredModelId);

  return {
    preferredProviderId,
    preferredModelId,
    allowFallback:
      normalizedProject.allowFallback ?? normalizedGlobal.allowFallback ?? true,
    source: projectSelectsModel
      ? "project"
      : globalSelectsModel
        ? "global"
        : "auto",
  };
}

export function findMediaProviderById<T extends MediaProviderCandidate>(
  providers: T[] | null | undefined,
  providerId?: string | null,
): T | null {
  const normalizedProviderId = providerId?.trim().toLowerCase();
  if (!normalizedProviderId) {
    return null;
  }
  const safeProviders = Array.isArray(providers) ? providers : [];

  return (
    safeProviders.find(
      (provider) => provider.id.trim().toLowerCase() === normalizedProviderId,
    ) ?? null
  );
}

export function isVideoProvider(providerId: string): boolean {
  const normalized = providerId.toLowerCase();
  const isAudioOnlyOpenai =
    normalized.includes("openai") &&
    (normalized.includes("tts") ||
      normalized.includes("voice") ||
      normalized.includes("audio"));
  return (
    !isAudioOnlyOpenai &&
    (normalized.includes("doubao") ||
      normalized.includes("volc") ||
      normalized.includes("dashscope") ||
      normalized.includes("alibaba") ||
      normalized.includes("qwen") ||
      normalized.includes("openai") ||
      normalized.includes("video") ||
      normalized.includes("runway") ||
      normalized.includes("minimax") ||
      normalized.includes("kling") ||
      normalized.includes("sora") ||
      normalized.includes("veo"))
  );
}

export function getVideoModelsForProvider(
  providerId: string,
  models?: string[],
): string[] {
  if (models && models.length > 0) {
    return models;
  }

  const normalizedId = providerId.toLowerCase();
  for (const [key, models] of Object.entries(VIDEO_MODEL_PRESETS)) {
    if (normalizedId.includes(key)) {
      return models;
    }
  }

  return [];
}

export function findVideoProviderForSelection<T extends MediaProviderCandidate>(
  providers: T[] | null | undefined,
  modelType: VideoModelPreset,
): T | null {
  const safeProviders = Array.isArray(providers) ? providers : [];
  const preferredKeywords: string[] =
    modelType === "keling"
      ? ["kling", "hailuo", "minimax"]
      : modelType === "jimeng"
        ? ["doubao", "volc"]
        : ["dashscope", "alibaba", "qwen"];

  for (const keyword of preferredKeywords) {
    const matched = safeProviders.find((provider) =>
      provider.id.toLowerCase().includes(keyword),
    );
    if (matched) {
      return matched;
    }
  }

  return safeProviders[0] ?? null;
}

export function pickVideoModelByVersion(
  models: string[],
  version: VideoModelVersion,
): string {
  if (models.length === 0) {
    return "";
  }

  const normalizedModels = models.map((model) => model.toLowerCase());
  const priorities =
    version === "v2-1-master"
      ? ["2.1", "master", "pro", "turbo"]
      : version === "v2"
        ? ["v2", "2.0", "pro", "turbo"]
        : ["1.6", "v1-6", "lite", "1.5"];

  for (const keyword of priorities) {
    const index = normalizedModels.findIndex((model) =>
      model.includes(keyword),
    );
    if (index >= 0) {
      return models[index] ?? models[0] ?? "";
    }
  }

  return models[0] ?? "";
}

export function isTtsProvider(
  providerId: string,
  providerType: string,
): boolean {
  const normalized = `${providerId}:${providerType}`.toLowerCase();
  return (
    normalized.includes("openai") ||
    normalized.includes("new-api") ||
    normalized.includes("azure") ||
    normalized.includes("google") ||
    normalized.includes("voice") ||
    normalized.includes("tts")
  );
}

export function findTtsProviderForSelection<T extends MediaProviderCandidate>(
  providers: T[] | null | undefined,
): T | null {
  const safeProviders = Array.isArray(providers) ? providers : [];
  const preferredKeywords = ["openai", "new-api", "azure", "google", "tts"];
  for (const keyword of preferredKeywords) {
    const matched = safeProviders.find((provider) =>
      `${provider.id}:${provider.type ?? ""}`.toLowerCase().includes(keyword),
    );
    if (matched) {
      return matched;
    }
  }
  return safeProviders[0] ?? null;
}

export function getTtsModelsForProvider(models?: string[]): string[] {
  if (models && models.length > 0) {
    return models;
  }
  return ["gpt-4o-mini-tts"];
}

export function pickTtsModel(models: string[]): string {
  if (models.length === 0) {
    return "gpt-4o-mini-tts";
  }
  const normalized = models.map((model) => model.toLowerCase());
  const preferredKeywords = ["tts", "speech", "audio", "gpt-4o-mini-tts"];
  for (const keyword of preferredKeywords) {
    const index = normalized.findIndex((model) => model.includes(keyword));
    if (index >= 0) {
      return models[index] ?? "gpt-4o-mini-tts";
    }
  }
  const nonImageModel = models.find(
    (model) => !model.toLowerCase().includes("image"),
  );
  if (nonImageModel) {
    return nonImageModel;
  }
  return "gpt-4o-mini-tts";
}
