import type { EnhancedModelMetadata } from "@/lib/types/modelRegistry";
import { normalizeThemeType } from "@/lib/workspace/workbenchContract";
import { modelSupportsTaskFamily } from "@/lib/model/inferModelCapabilities";
import { isLikelyImageGenerationSearchText } from "@/lib/imageGen/providerMatchers";

export interface ThemeModelFilterResult {
  models: EnhancedModelMetadata[];
  usedFallback: boolean;
  filteredOutCount: number;
  policyName: string;
}

const IMAGE_EXCLUDE_KEYWORDS = [
  "embedding",
  "rerank",
  "tts",
  "stt",
  "asr",
  "transcribe",
  "audio",
  "speech",
  "reasoner",
  "thinking",
  "computer-use",
];

const NON_CHAT_KEYWORDS = [
  "embedding",
  "rerank",
  "tts",
  "stt",
  "asr",
  "transcribe",
  "audio",
  "speech",
  "whisper",
];

function containsAnyKeyword(text: string, keywords: string[]): boolean {
  return keywords.some((keyword) => text.includes(keyword));
}

function buildModelSearchText(model: EnhancedModelMetadata): string {
  return [
    model.id,
    model.display_name,
    model.family || "",
    model.description || "",
  ]
    .join(" ")
    .toLowerCase();
}

function looksLikeImageGenerationModel(model: EnhancedModelMetadata): boolean {
  const canReturnText =
    !model.output_modalities?.length ||
    model.output_modalities.includes("text");
  if (modelSupportsTaskFamily(model, "vision_understanding") && canReturnText) {
    return false;
  }

  if (
    modelSupportsTaskFamily(model, "image_generation") ||
    modelSupportsTaskFamily(model, "image_edit")
  ) {
    return true;
  }

  const text = buildModelSearchText(model);

  if (
    containsAnyKeyword(text, IMAGE_EXCLUDE_KEYWORDS) ||
    containsAnyKeyword(text, NON_CHAT_KEYWORDS)
  ) {
    return false;
  }

  if (isLikelyImageGenerationSearchText(text)) {
    return true;
  }

  return (
    model.capabilities.vision &&
    model.capabilities.tools === false &&
    model.capabilities.function_calling === false &&
    model.capabilities.json_mode === false
  );
}

function looksLikeChatModel(model: EnhancedModelMetadata): boolean {
  if (looksLikeImageGenerationModel(model)) {
    return false;
  }

  if (
    modelSupportsTaskFamily(model, "embedding") ||
    modelSupportsTaskFamily(model, "rerank") ||
    modelSupportsTaskFamily(model, "moderation") ||
    modelSupportsTaskFamily(model, "speech_to_text") ||
    modelSupportsTaskFamily(model, "text_to_speech")
  ) {
    return false;
  }

  if (
    modelSupportsTaskFamily(model, "chat") ||
    modelSupportsTaskFamily(model, "reasoning") ||
    modelSupportsTaskFamily(model, "vision_understanding")
  ) {
    return true;
  }

  const text = buildModelSearchText(model);

  if (containsAnyKeyword(text, NON_CHAT_KEYWORDS)) {
    return false;
  }

  return !looksLikeImageGenerationModel(model);
}

export function filterModelsByTheme(
  theme: string | undefined,
  models: EnhancedModelMetadata[] | null | undefined,
): ThemeModelFilterResult {
  const safeModels = Array.isArray(models) ? models : [];
  if (safeModels.length === 0) {
    return {
      models: [],
      usedFallback: false,
      filteredOutCount: 0,
      policyName: "none",
    };
  }

  if (!theme?.trim()) {
    return {
      models: safeModels,
      usedFallback: false,
      filteredOutCount: 0,
      policyName: "none",
    };
  }

  if (normalizeThemeType(theme) === "general") {
    const chatModels = safeModels.filter(looksLikeChatModel);
    return {
      models: chatModels,
      usedFallback: false,
      filteredOutCount: safeModels.length - chatModels.length,
      policyName: "chat-only",
    };
  }

  return {
    models: safeModels,
    usedFallback: false,
    filteredOutCount: 0,
    policyName: "none",
  };
}
