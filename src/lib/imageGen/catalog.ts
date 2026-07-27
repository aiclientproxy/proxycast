/**
 * 图片能力目录与解析器
 *
 * 这里是图片 Provider / 模型 / 默认选择的单一目录入口。
 */

import { inferModelTaskFamilies } from "@/lib/model/inferModelCapabilities";
import { IMAGE_GEN_MODELS, type ImageGenModel } from "./models";
import {
  isFalImageProviderLike,
  isLikelyImageGenerationModelId,
  isLikelyFalImageModelId,
} from "./providerMatchers";

export type { ImageGenModel } from "./models";

export type ImageCapabilityTransport =
  | "openai_images"
  | "openai_responses_image"
  | "gemini_image"
  | "provider_native"
  | "fal_queue";

export interface ImageCapabilitySelectionCandidate {
  id: string;
  type: string;
  models?: string[];
  api_host?: string;
}

export interface ImageCapabilityProviderEntry {
  providerKey: string;
  displayName: string;
  match: {
    providerIds?: string[];
    providerTypes?: string[];
    providerIdIncludes?: string[];
    providerTypeIncludes?: string[];
    apiHostIncludes?: string[];
  };
  transport: ImageCapabilityTransport;
  endpointPath?: string;
  models: ImageGenModel[];
  fallbackPriority: number;
}

export interface ImageCapabilityModelMetadataCandidate {
  id?: string | null;
  task_families?: string[] | null;
  taskFamilies?: string[] | null;
}

function normalizeText(value?: string | null): string {
  return (value ?? "").trim().toLowerCase();
}

function matchesAnyExactValue(
  normalizedValue: string,
  values?: string[],
): boolean {
  return Boolean(
    normalizedValue &&
    Array.isArray(values) &&
    values.some((value) => normalizeText(value) === normalizedValue),
  );
}

function matchesAnyIncludesValue(
  normalizedValue: string,
  values?: string[],
): boolean {
  return Boolean(
    normalizedValue &&
    Array.isArray(values) &&
    values.some((value) => {
      const normalizedNeedle = normalizeText(value);
      return (
        normalizedNeedle.length > 0 &&
        normalizedValue.includes(normalizedNeedle)
      );
    }),
  );
}

function isImageGenerationModelId(
  modelId: string,
  providerId: string,
  providerType: string,
): boolean {
  if (isLikelyImageGenerationModelId(modelId)) {
    return true;
  }

  return inferModelTaskFamilies({
    modelId,
    providerId,
    providerType,
    providerModelId: modelId,
  }).includes("image_generation");
}

function normalizeGeneratedModel(modelId: string): ImageGenModel {
  return {
    id: modelId,
    name: modelId,
    supportedSizes: [
      "1024x1024",
      "768x1344",
      "1344x768",
      "1792x1024",
      "1024x1792",
    ],
  };
}

function hasImageGenerationTaskFamily(
  model: ImageCapabilityModelMetadataCandidate,
): boolean {
  const taskFamilies = model.task_families ?? model.taskFamilies ?? [];
  return taskFamilies.some(
    (taskFamily) => normalizeText(taskFamily) === "image_generation",
  );
}

export const IMAGE_CAPABILITY_CATALOG: ImageCapabilityProviderEntry[] = [
  {
    providerKey: "openai-compatible-responses",
    displayName: "OpenAI-compatible Responses Images",
    match: {
      providerIds: ["new-api", "newapi"],
      providerTypes: ["new-api", "NewApi", "newapi"],
    },
    transport: "openai_responses_image",
    endpointPath: "/v1/responses",
    models: IMAGE_GEN_MODELS["new-api"],
    fallbackPriority: 10,
  },
  {
    providerKey: "openai-images",
    displayName: "OpenAI Images",
    match: {
      providerIds: ["openai"],
      providerTypes: ["openai", "openai-response"],
      providerIdIncludes: ["openai"],
      providerTypeIncludes: ["openai"],
    },
    transport: "openai_images",
    endpointPath: "/v1/images/generations",
    models: IMAGE_GEN_MODELS.openai,
    fallbackPriority: 10,
  },
  {
    providerKey: "agnes",
    displayName: "Agnes",
    match: {
      providerIds: ["agnes"],
      providerTypes: ["agnes"],
      providerIdIncludes: ["agnes"],
      providerTypeIncludes: ["agnes"],
      apiHostIncludes: ["agnes-ai.com"],
    },
    transport: "openai_images",
    endpointPath: "/v1/images/generations",
    models: IMAGE_GEN_MODELS.agnes,
    fallbackPriority: 18,
  },
  {
    providerKey: "fal",
    displayName: "Fal",
    match: {
      providerIds: ["fal"],
      providerTypes: ["fal"],
      providerIdIncludes: ["fal"],
      providerTypeIncludes: ["fal"],
      apiHostIncludes: ["fal.run", "queue.fal.run"],
    },
    transport: "fal_queue",
    models: IMAGE_GEN_MODELS.fal,
    fallbackPriority: 20,
  },
  {
    providerKey: "gemini",
    displayName: "Gemini",
    match: {
      providerIds: ["gemini", "google", "vertexai", "google-vertex"],
      providerTypes: ["gemini", "vertexai", "google"],
      providerIdIncludes: ["gemini", "google", "vertexai"],
      providerTypeIncludes: ["gemini", "vertexai", "google"],
      apiHostIncludes: ["generativelanguage.googleapis.com"],
    },
    transport: "gemini_image",
    endpointPath: "/v1beta/interactions",
    models: IMAGE_GEN_MODELS.gemini,
    fallbackPriority: 15,
  },
  {
    providerKey: "dmxapi",
    displayName: "DMXAPI",
    match: {
      providerIds: ["dmxapi"],
      providerIdIncludes: ["dmxapi"],
      providerTypeIncludes: ["dmxapi"],
    },
    transport: "openai_images",
    endpointPath: "/v1/images/generations",
    models: IMAGE_GEN_MODELS.dmxapi,
    fallbackPriority: 30,
  },
  {
    providerKey: "tokenflux",
    displayName: "TokenFlux",
    match: {
      providerIds: ["tokenflux"],
      providerIdIncludes: ["tokenflux"],
      providerTypeIncludes: ["tokenflux"],
    },
    transport: "openai_images",
    endpointPath: "/v1/images/generations",
    models: IMAGE_GEN_MODELS.tokenflux,
    fallbackPriority: 30,
  },
  {
    providerKey: "aihubmix",
    displayName: "AiHubMix",
    match: {
      providerIds: ["aihubmix"],
      providerIdIncludes: ["aihubmix"],
      providerTypeIncludes: ["aihubmix"],
    },
    transport: "openai_images",
    endpointPath: "/v1/images/generations",
    models: IMAGE_GEN_MODELS.aihubmix,
    fallbackPriority: 35,
  },
  {
    providerKey: "siliconflow",
    displayName: "SiliconFlow",
    match: {
      providerIds: ["siliconflow", "siliconflow-cn"],
      providerTypes: ["siliconflow", "siliconflow-cn"],
      providerIdIncludes: ["siliconflow"],
      providerTypeIncludes: ["siliconflow"],
    },
    transport: "openai_images",
    endpointPath: "/v1/images/generations",
    models: IMAGE_GEN_MODELS.siliconflow,
    fallbackPriority: 40,
  },
  {
    providerKey: "zhipu",
    displayName: "Zhipu AI",
    match: {
      providerIds: ["zhipuai", "zhipu"],
      providerTypes: ["zhipuai", "zhipu"],
      providerIdIncludes: ["zhipu"],
      providerTypeIncludes: ["zhipu"],
    },
    transport: "provider_native",
    endpointPath: "/api/paas/v4/images/generations",
    models: IMAGE_GEN_MODELS.zhipuai,
    fallbackPriority: 50,
  },
  {
    providerKey: "dashscope",
    displayName: "DashScope",
    match: {
      providerIds: ["alibaba", "alibaba-cn", "dashscope", "qwen", "tongyi"],
      providerTypes: ["alibaba", "alibaba-cn", "dashscope", "qwen", "tongyi"],
      providerIdIncludes: ["dashscope", "alibaba", "qwen", "tongyi"],
      providerTypeIncludes: ["dashscope", "alibaba", "qwen", "tongyi"],
      apiHostIncludes: [
        "dashscope.aliyuncs.com",
        "dashscope-intl.aliyuncs.com",
        "maas.aliyuncs.com",
      ],
    },
    transport: "provider_native",
    endpointPath: "/api/v1/services/aigc/multimodal-generation/generation",
    models: IMAGE_GEN_MODELS.dashscope,
    fallbackPriority: 55,
  },
];

export function resolveImageCapabilityProviderEntry(
  candidate: ImageCapabilitySelectionCandidate,
): ImageCapabilityProviderEntry | null {
  const normalizedProviderId = normalizeText(candidate.id);
  const normalizedProviderType = normalizeText(candidate.type);
  const normalizedApiHost = normalizeText(candidate.api_host);

  return (
    IMAGE_CAPABILITY_CATALOG.map((entry, index) => {
      const exactProviderIdMatch = matchesAnyExactValue(
        normalizedProviderId,
        entry.match.providerIds,
      );
      const exactProviderTypeMatch = matchesAnyExactValue(
        normalizedProviderType,
        entry.match.providerTypes,
      );
      const includesProviderIdMatch = matchesAnyIncludesValue(
        normalizedProviderId,
        entry.match.providerIdIncludes,
      );
      const includesProviderTypeMatch = matchesAnyIncludesValue(
        normalizedProviderType,
        entry.match.providerTypeIncludes,
      );
      const apiHostMatch = matchesAnyIncludesValue(
        normalizedApiHost,
        entry.match.apiHostIncludes,
      );

      const score =
        (exactProviderIdMatch ? 100 : 0) +
        (apiHostMatch ? 140 : 0) +
        (includesProviderIdMatch ? 60 : 0) +
        (exactProviderTypeMatch ? 40 : 0) +
        (includesProviderTypeMatch ? 20 : 0);

      return score > 0 ? { entry, score, index } : null;
    })
      .filter(
        (
          match,
        ): match is {
          entry: ImageCapabilityProviderEntry;
          score: number;
          index: number;
        } => match !== null,
      )
      .sort(
        (left, right) => right.score - left.score || left.index - right.index,
      )
      .at(0)?.entry ?? null
  );
}

export function resolveImageCapabilityModels(
  candidate: ImageCapabilitySelectionCandidate,
): ImageGenModel[] {
  const entry = resolveImageCapabilityProviderEntry(candidate);
  const builtinModels = entry?.models ?? [];
  const models =
    candidate.models && candidate.models.length > 0
      ? candidate.models
          .filter((modelId) =>
            isFalImageProviderLike(candidate)
              ? isLikelyFalImageModelId(modelId)
              : isImageGenerationModelId(modelId, candidate.id, candidate.type),
          )
          .map(normalizeGeneratedModel)
      : [];

  if (models.length > 0) {
    const mergedModels = new Map(
      models.map((model) => [model.id, model]),
    );
    for (const model of builtinModels) {
      mergedModels.set(model.id, model);
    }
    return Array.from(mergedModels.values());
  }

  return builtinModels;
}

export function isImageCapabilityModelId(
  candidate: ImageCapabilitySelectionCandidate,
  modelId?: string | null,
): boolean {
  const normalizedModelId = modelId?.trim();
  if (!normalizedModelId) {
    return false;
  }

  const catalogModels = resolveImageCapabilityModels(candidate);
  if (
    catalogModels.some(
      (model) => normalizeText(model.id) === normalizeText(normalizedModelId),
    )
  ) {
    return true;
  }

  return isFalImageProviderLike(candidate)
    ? isLikelyFalImageModelId(normalizedModelId)
    : isImageGenerationModelId(normalizedModelId, candidate.id, candidate.type);
}

export function isImageCapabilityModelMetadata(
  candidate: ImageCapabilitySelectionCandidate,
  model: ImageCapabilityModelMetadataCandidate,
): boolean {
  if (hasImageGenerationTaskFamily(model)) {
    return true;
  }

  return isImageCapabilityModelId(candidate, model.id);
}

export function resolveImageCapabilityModelIds(
  candidate: ImageCapabilitySelectionCandidate,
): string[] {
  return resolveImageCapabilityModels(candidate).map((model) => model.id);
}

export function isImageCapabilityProvider(
  candidate: ImageCapabilitySelectionCandidate,
): boolean {
  if (resolveImageCapabilityProviderEntry(candidate)) {
    return true;
  }

  return Boolean(
    Array.isArray(candidate.models) &&
    candidate.models.some((modelId) =>
      isImageGenerationModelId(modelId, candidate.id, candidate.type),
    ),
  );
}

export function findDefaultImageCapabilityProvider<
  T extends ImageCapabilitySelectionCandidate,
>(providers: T[] | null | undefined): T | null {
  const safeProviders = Array.isArray(providers) ? providers : [];
  const rankedProviders = safeProviders
    .map((provider, index) => {
      const entry = resolveImageCapabilityProviderEntry(provider);
      if (!isImageCapabilityProvider(provider)) {
        return null;
      }

      return {
        provider,
        index,
        priority: entry?.fallbackPriority ?? 1000,
      };
    })
    .filter(
      (
        item,
      ): item is {
        provider: T;
        index: number;
        priority: number;
      } => Boolean(item),
    )
    .sort((left, right) => {
      if (left.priority !== right.priority) {
        return left.priority - right.priority;
      }
      return left.index - right.index;
    });

  return rankedProviders[0]?.provider ?? null;
}
