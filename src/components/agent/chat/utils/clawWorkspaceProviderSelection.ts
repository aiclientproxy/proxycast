import { modelRegistryApi } from "@/lib/api/modelRegistry";
import { isLikelyImageGenerationModelId } from "@/lib/imageGen/providerMatchers";
import { type EnhancedModelMetadata } from "@/lib/types/modelRegistry";
import { filterModelsByTheme } from "./modelThemePolicy";

export interface ResolveClawWorkspaceProviderSelectionInput {
  currentProviderType?: string | null;
  currentModel?: string | null;
  theme?: string;
  allowProviderFallback?: boolean;
}

export interface ClawWorkspaceProviderSelection {
  providerType: string;
  model: string;
}

function normalizeValue(value?: string | null): string {
  return (value || "").trim().toLowerCase();
}

function isTextChatCandidateModel(model: EnhancedModelMetadata): boolean {
  const outputModalities = model.output_modalities ?? [];
  const canReturnText =
    outputModalities.length === 0 || outputModalities.includes("text");
  if (!canReturnText) {
    return false;
  }

  const taskFamilies = model.task_families ?? [];
  const isImageTaskModel =
    taskFamilies.includes("image_generation") ||
    taskFamilies.includes("image_edit") ||
    isLikelyImageGenerationModelId(model.id);
  return !isImageTaskModel || outputModalities.includes("text");
}

export async function resolveClawWorkspaceProviderSelection(
  input: ResolveClawWorkspaceProviderSelectionInput,
): Promise<ClawWorkspaceProviderSelection | null> {
  const {
    currentProviderType,
    currentModel,
    theme,
    allowProviderFallback = true,
  } = input;
  const catalog = await modelRegistryApi.getModelRegistry();
  const themedModels = filterModelsByTheme(theme, catalog).models;
  const candidates = (themedModels.length > 0 ? themedModels : catalog).filter(
    isTextChatCandidateModel,
  );
  if (candidates.length === 0) {
    return null;
  }

  const normalizedProvider = normalizeValue(currentProviderType);
  const normalizedModel = normalizeValue(currentModel);
  const providerCandidates = normalizedProvider
    ? candidates.filter(
        (model) => normalizeValue(model.provider_id) === normalizedProvider,
      )
    : [];
  const retained = providerCandidates.find(
    (model) => normalizeValue(model.id) === normalizedModel,
  );
  const selected =
    retained ??
    providerCandidates.find((model) => model.is_default) ??
    providerCandidates[0] ??
    (allowProviderFallback || !normalizedProvider
      ? (candidates.find((model) => model.is_default) ?? candidates[0])
      : null);

  return selected
    ? { providerType: selected.provider_id, model: selected.id }
    : null;
}
