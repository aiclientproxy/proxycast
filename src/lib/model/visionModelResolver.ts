import type { EnhancedModelMetadata } from "@/lib/types/modelRegistry";
import { inferVisionCapability } from "./inferModelCapabilities";
import { isLikelyImageGenerationSearchText } from "@/lib/imageGen/providerMatchers";

export type VisionResolveReason =
  | "already_vision"
  | "matched"
  | "fallback_latest"
  | "no_vision_model";

export interface VisionResolveResult {
  targetModelId: string;
  switched: boolean;
  reason: VisionResolveReason;
}

interface ResolveVisionModelParams {
  currentModelId: string;
  models: EnhancedModelMetadata[];
}

const TIER_WEIGHT: Record<EnhancedModelMetadata["tier"], number> = {
  mini: 1,
  pro: 2,
  max: 3,
};

const normalize = (value?: string | null): string =>
  (value || "").trim().toLowerCase();

const findModelMeta = (
  modelId: string,
  models: EnhancedModelMetadata[],
): EnhancedModelMetadata | undefined => {
  const normalizedId = normalize(modelId);
  return models.find((model) => normalize(model.id) === normalizedId);
};

const buildSearchText = (model: EnhancedModelMetadata): string =>
  [model.id, model.display_name, model.family || "", model.description || ""]
    .join(" ")
    .toLowerCase();

const isLikelyImageGenerationModel = (
  model: EnhancedModelMetadata,
): boolean => {
  const canReturnText =
    !model.output_modalities?.length ||
    model.output_modalities.includes("text");
  if (model.task_families?.includes("vision_understanding") && canReturnText) {
    return false;
  }

  if (
    model.task_families?.includes("image_generation") ||
    model.task_families?.includes("image_edit") ||
    model.output_modalities?.includes("image")
  ) {
    return true;
  }

  const text = buildSearchText(model);
  if (!isLikelyImageGenerationSearchText(text)) {
    return false;
  }

  return (
    model.capabilities.tools === false &&
    model.capabilities.function_calling === false &&
    model.capabilities.json_mode === false
  );
};

const supportsVision = (
  model: EnhancedModelMetadata | undefined,
  fallbackModelId?: string,
): boolean => {
  if (model) {
    const hasExplicitVisionSignal =
      model.capabilities.vision ||
      model.task_families?.includes("vision_understanding") ||
      model.input_modalities?.includes("image");

    if (hasExplicitVisionSignal) {
      return !isLikelyImageGenerationModel(model);
    }
  }

  if (!fallbackModelId) {
    return false;
  }

  return inferVisionCapability({
    modelId: fallbackModelId,
    providerId: model?.provider_id,
    family: model?.family,
    description: model?.description,
  });
};

const capabilityScore = (model: EnhancedModelMetadata): number => {
  let score = 0;
  if (model.capabilities.tools) score += 5;
  if (model.capabilities.function_calling) score += 4;
  if (model.capabilities.json_mode) score += 3;
  if (model.capabilities.reasoning) score += 2;
  if (model.capabilities.streaming) score += 1;
  return score;
};

function compareReleaseDateDesc(
  left: EnhancedModelMetadata,
  right: EnhancedModelMetadata,
): number {
  if (left.release_date && right.release_date) {
    return right.release_date.localeCompare(left.release_date);
  }
  if (left.release_date && !right.release_date) return -1;
  if (!left.release_date && right.release_date) return 1;
  return 0;
}

export function resolveVisionModel(
  params: ResolveVisionModelParams,
): VisionResolveResult {
  const { currentModelId, models } = params;
  const currentModel = findModelMeta(currentModelId, models);

  if (supportsVision(currentModel, currentModelId)) {
    return {
      targetModelId: currentModel?.id || currentModelId,
      switched: false,
      reason: "already_vision",
    };
  }

  const currentFamily = normalize(currentModel?.family);
  const candidates = models.filter((model) => supportsVision(model, model.id));

  if (candidates.length === 0) {
    return {
      targetModelId: currentModelId,
      switched: false,
      reason: "no_vision_model",
    };
  }

  const sortedCandidates = [...candidates].sort((left, right) => {
    const leftSameFamily =
      currentFamily.length > 0 && normalize(left.family) === currentFamily;
    const rightSameFamily =
      currentFamily.length > 0 && normalize(right.family) === currentFamily;
    if (leftSameFamily !== rightSameFamily) {
      return leftSameFamily ? -1 : 1;
    }

    const capabilityDelta = capabilityScore(right) - capabilityScore(left);
    if (capabilityDelta !== 0) {
      return capabilityDelta;
    }

    if (left.is_latest !== right.is_latest) {
      return left.is_latest ? -1 : 1;
    }

    const tierDelta = TIER_WEIGHT[right.tier] - TIER_WEIGHT[left.tier];
    if (tierDelta !== 0) {
      return tierDelta;
    }

    const releaseDelta = compareReleaseDateDesc(left, right);
    if (releaseDelta !== 0) {
      return releaseDelta;
    }

    return left.id.localeCompare(right.id);
  });

  const target = sortedCandidates[0];
  const reason =
    currentFamily.length > 0 && normalize(target.family) === currentFamily
      ? "matched"
      : "fallback_latest";

  return {
    targetModelId: target.id,
    switched: normalize(target.id) !== normalize(currentModelId),
    reason,
  };
}
