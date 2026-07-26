import type {
  EnhancedModelMetadata,
  ModelAliasSource,
  ModelCapabilities,
  ModelDeploymentSource,
  ModelManagementPlane,
  ModelModality,
  ModelRuntimeFeature,
  ModelTaskFamily,
} from "@/lib/types/modelRegistry";
import { isLikelyImageGenerationSearchText } from "@/lib/imageGen/providerMatchers";

const VISION_HINT_PATTERN =
  /\b(vision|multimodal|multi-modal|omni|image-input|image understanding)\b/i;
const NON_VISION_PATTERN =
  /\b(embedding|embed|rerank|tts|stt|transcribe|transcription|speech|audio|moderation)\b/i;
const IMAGE_EDIT_PATTERN =
  /\b(edit|inpaint|outpaint|img2img|image-edit|image_edit|image edits)\b/i;
const OPENAI_VISION_PATTERN =
  /\b(chatgpt-4o|gpt-5(?:[._/-]|\b)|gpt-4o(?:[._/-]|\b)|gpt-4\.1(?:[._/-]|\b)|gpt-4\.5(?:[._/-]|\b)|gpt-4-turbo(?:[._/-]|\b)|gpt-5.*codex|o1(?!-(?:mini|preview))(?:[._/-]|\b)|o3(?!-mini)(?:[._/-]|\b)|o4-mini(?:[._/-]|\b))\b/i;
const GEMINI_VISION_PATTERN = /\bgemini(?:[._/-]|\b)/i;
const CLAUDE_VISION_PATTERN = /\bclaude(?:[._/-]|\b)/i;
const QWEN_VISION_PATTERN =
  /\bqwen(?:[._/-]|\b).*(vl|vision)|\bqvq\b|\bqwen3[._-][56](?:[._/-]|\b)/i;
const GLM_VISION_PATTERN = /\bglm-[\w.-]*v[\w.-]*\b/i;
const XAI_VISION_PATTERN =
  /\bgrok-(?:vision|2-vision|4(?:-1|-fast|\.20|\.3))(?:[._/-]|\b)/i;
const MISTRAL_VISION_PATTERN =
  /\b(pixtral|mistral-(?:small-latest|large-2512|medium-3\.1))(?:[._/-]|\b)/i;
const GEMMA_VISION_PATTERN = /\bgemma-3(?!n)(?:[._/-]|\b)/i;
const LLAMA_VISION_PATTERN = /\bllama-4-(?:maverick|scout)(?:[._/-]|\b)/i;
const EMBEDDING_PATTERN =
  /\b(embedding|embed|text-embedding|voyage|jina-embeddings)\b/i;
const RERANK_PATTERN = /\b(rerank|re-rank)\b/i;
const MODERATION_PATTERN = /\bmoderation\b/i;
const SPEECH_TO_TEXT_PATTERN =
  /\b(stt|asr|speech[- ]?to[- ]?text|transcribe|transcription|whisper)\b/i;
const TEXT_TO_SPEECH_PATTERN =
  /\b(tts|text[- ]?to[- ]?speech|speech[- ]?synthesis|voice[- ]?synth)\b/i;

const LOCAL_PROVIDER_PATTERN = /\b(ollama|lmstudio|gpustack|ovms|comfyui)\b/i;
const OEM_PROVIDER_PATTERN = /\b(lime[\s-_]?hub|oem|partner[\s-_]?hub)\b/i;

const TASK_FAMILY_SET = new Set<ModelTaskFamily>([
  "chat",
  "reasoning",
  "vision_understanding",
  "image_generation",
  "image_edit",
  "speech_to_text",
  "text_to_speech",
  "embedding",
  "rerank",
  "moderation",
]);
const MODALITY_SET = new Set<ModelModality>([
  "text",
  "image",
  "audio",
  "video",
  "file",
  "embedding",
  "json",
]);
const RUNTIME_FEATURE_SET = new Set<ModelRuntimeFeature>([
  "streaming",
  "tool_calling",
  "json_schema",
  "reasoning",
  "prompt_cache",
  "responses_api",
  "chat_completions_api",
  "images_api",
]);

interface InferModelTaxonomyParams {
  modelId: string;
  providerId?: string | null;
  providerType?: string | null;
  family?: string | null;
  description?: string | null;
  capabilities?: Partial<ModelCapabilities> | null;
  explicitTaskFamilies?: string[] | null;
  explicitInputModalities?: string[] | null;
  explicitOutputModalities?: string[] | null;
  explicitRuntimeFeatures?: string[] | null;
  deploymentSource?: ModelDeploymentSource | null;
  managementPlane?: ModelManagementPlane | null;
  source?: string | null;
  providerModelId?: string | null;
  canonicalModelId?: string | null;
  aliasSource?: ModelAliasSource | null;
}

type ModelTaxonomyInput = Pick<
  EnhancedModelMetadata,
  | "id"
  | "provider_id"
  | "family"
  | "description"
  | "capabilities"
  | "task_families"
  | "input_modalities"
  | "output_modalities"
  | "runtime_features"
  | "deployment_source"
  | "management_plane"
  | "source"
  | "provider_model_id"
  | "canonical_model_id"
  | "alias_source"
>;

export interface ModelCapabilitySummary {
  capabilities: ModelCapabilities;
  task_families: ModelTaskFamily[];
  input_modalities: ModelModality[];
  output_modalities: ModelModality[];
  runtime_features: ModelRuntimeFeature[];
  supports_tools: boolean;
  supports_reasoning: boolean;
  supports_prompt_cache: boolean;
  supports_media_input: boolean;
  supports_media_output: boolean;
  context_length: number | null;
  max_output_tokens: number | null;
}

function normalize(value?: string | null): string {
  return (value || "").trim().toLowerCase();
}

function buildSearchText(parts: Array<string | null | undefined>): string {
  return parts
    .map((part) => normalize(part))
    .filter(Boolean)
    .join(" ");
}

function uniqueValues<T extends string>(values: T[]): T[] {
  return Array.from(new Set(values));
}

function normalizeTaskFamilies(values?: string[] | null): ModelTaskFamily[] {
  return uniqueValues(
    (values ?? []).filter((value): value is ModelTaskFamily =>
      TASK_FAMILY_SET.has(value as ModelTaskFamily),
    ),
  );
}

function normalizeModalities(values?: string[] | null): ModelModality[] {
  return uniqueValues(
    (values ?? []).filter((value): value is ModelModality =>
      MODALITY_SET.has(value as ModelModality),
    ),
  );
}

function normalizeRuntimeFeatures(
  values?: string[] | null,
): ModelRuntimeFeature[] {
  return uniqueValues(
    (values ?? []).filter((value): value is ModelRuntimeFeature =>
      RUNTIME_FEATURE_SET.has(value as ModelRuntimeFeature),
    ),
  );
}

export function inferVisionCapability(params: {
  modelId: string;
  providerId?: string | null;
  family?: string | null;
  description?: string | null;
}): boolean {
  const { modelId, providerId, family, description } = params;
  const text = buildSearchText([modelId, family, description]);
  const provider = normalize(providerId);

  if (!text) {
    return false;
  }

  if (
    NON_VISION_PATTERN.test(text) ||
    isLikelyImageGenerationSearchText(text)
  ) {
    return false;
  }

  if (VISION_HINT_PATTERN.test(text)) {
    return true;
  }

  if (OPENAI_VISION_PATTERN.test(text)) {
    return true;
  }

  if (provider === "codex" || provider === "openai") {
    return OPENAI_VISION_PATTERN.test(text);
  }

  if (provider === "gemini" || provider === "google") {
    return GEMINI_VISION_PATTERN.test(text) || GEMMA_VISION_PATTERN.test(text);
  }

  if (provider === "anthropic" || provider === "claude") {
    return CLAUDE_VISION_PATTERN.test(text);
  }

  if (provider === "qwen" || provider === "alibaba") {
    return QWEN_VISION_PATTERN.test(text);
  }

  if (provider === "zhipuai") {
    return GLM_VISION_PATTERN.test(text);
  }
  if (provider === "xai" || provider === "x-ai") {
    return XAI_VISION_PATTERN.test(text);
  }
  if (provider === "mistral" || provider === "mistralai") {
    return MISTRAL_VISION_PATTERN.test(text);
  }
  if (
    provider === "llama" ||
    provider === "meta" ||
    provider === "meta-llama"
  ) {
    return LLAMA_VISION_PATTERN.test(text);
  }

  return (
    GEMINI_VISION_PATTERN.test(text) ||
    CLAUDE_VISION_PATTERN.test(text) ||
    QWEN_VISION_PATTERN.test(text) ||
    GLM_VISION_PATTERN.test(text) ||
    XAI_VISION_PATTERN.test(text) ||
    MISTRAL_VISION_PATTERN.test(text) ||
    GEMMA_VISION_PATTERN.test(text) ||
    LLAMA_VISION_PATTERN.test(text)
  );
}

function inferBaseSignals(params: InferModelTaxonomyParams) {
  const text = buildSearchText([
    params.modelId,
    params.family,
    params.description,
    params.providerModelId,
    params.canonicalModelId,
  ]);
  const capabilities = params.capabilities ?? {};
  const explicitInputModalities = normalizeModalities(
    params.explicitInputModalities,
  );
  const explicitOutputModalities = normalizeModalities(
    params.explicitOutputModalities,
  );
  const explicitRuntimeFeatures = normalizeRuntimeFeatures(
    params.explicitRuntimeFeatures,
  );
  const hasExplicitImageOutput = explicitOutputModalities.includes("image");
  const hasExplicitImageInput = explicitInputModalities.includes("image");
  const hasExplicitTextOutput =
    explicitOutputModalities.length === 0 ||
    explicitOutputModalities.includes("text");
  const hasExplicitVisionInput = hasExplicitImageInput && hasExplicitTextOutput;
  const declaredReasoning =
    capabilities.reasoning === true ||
    explicitRuntimeFeatures.includes("reasoning");
  const inferredVisionByName =
    inferVisionCapability({
      modelId: params.modelId,
      providerId: params.providerId,
      family: params.family,
      description: params.description,
    }) ||
    (params.providerModelId
      ? inferVisionCapability({
          modelId: params.providerModelId,
          providerId: params.providerId,
          family: params.family,
          description: params.description,
        })
      : false) ||
    (params.canonicalModelId
      ? inferVisionCapability({
          modelId: params.canonicalModelId,
          providerId: params.providerId,
          family: params.family,
          description: params.description,
        })
      : false);
  const inferredVision =
    capabilities.vision === true ||
    hasExplicitVisionInput ||
    inferredVisionByName;

  const isEmbedding = EMBEDDING_PATTERN.test(text);
  const isRerank = RERANK_PATTERN.test(text);
  const isModeration = MODERATION_PATTERN.test(text);
  const isSpeechToText = SPEECH_TO_TEXT_PATTERN.test(text);
  const isTextToSpeech = TEXT_TO_SPEECH_PATTERN.test(text);
  const isImageGeneration =
    hasExplicitImageOutput || isLikelyImageGenerationSearchText(text);
  const isImageEdit =
    IMAGE_EDIT_PATTERN.test(text) ||
    (hasExplicitImageInput && hasExplicitImageOutput);

  return {
    text,
    capabilities,
    explicitInputModalities,
    explicitOutputModalities,
    declaredReasoning,
    inferredVision,
    hasExplicitVisionInput,
    isEmbedding,
    isRerank,
    isModeration,
    isSpeechToText,
    isTextToSpeech,
    isImageGeneration,
    isImageEdit,
  };
}

export function inferModelTaskFamilies(
  params: InferModelTaxonomyParams,
): ModelTaskFamily[] {
  const explicitTaskFamilies = normalizeTaskFamilies(
    params.explicitTaskFamilies,
  );

  const {
    capabilities,
    declaredReasoning,
    inferredVision,
    isEmbedding,
    isRerank,
    isModeration,
    isSpeechToText,
    isTextToSpeech,
    isImageGeneration,
    isImageEdit,
    hasExplicitVisionInput,
  } = inferBaseSignals(params);

  const families: ModelTaskFamily[] = [...explicitTaskFamilies];

  if (isEmbedding) {
    families.push("embedding");
  }
  if (isRerank) {
    families.push("rerank");
  }
  if (isModeration) {
    families.push("moderation");
  }
  if (isSpeechToText) {
    families.push("speech_to_text");
  }
  if (isTextToSpeech) {
    families.push("text_to_speech");
  }
  if (isImageGeneration) {
    families.push("image_generation");
  }
  if (isImageEdit) {
    families.push("image_edit");
  }
  if (inferredVision && (!isImageGeneration || hasExplicitVisionInput)) {
    families.push("vision_understanding");
  }
  if (declaredReasoning) {
    families.push("reasoning");
  }

  const isSpecializedOnly =
    families.includes("embedding") ||
    families.includes("rerank") ||
    families.includes("moderation") ||
    families.includes("speech_to_text") ||
    families.includes("text_to_speech") ||
    families.includes("image_generation") ||
    families.includes("image_edit");

  if (
    !isSpecializedOnly ||
    inferredVision ||
    declaredReasoning ||
    capabilities.tools ||
    capabilities.function_calling ||
    capabilities.json_mode
  ) {
    families.push("chat");
  }

  return uniqueValues(families);
}

export function inferInputModalities(
  params: InferModelTaxonomyParams,
): ModelModality[] {
  const explicitInputModalities = normalizeModalities(
    params.explicitInputModalities,
  );
  if (explicitInputModalities.length > 0) {
    return explicitInputModalities;
  }

  const {
    inferredVision,
    isEmbedding,
    isRerank,
    isModeration,
    isSpeechToText,
    isTextToSpeech,
    isImageEdit,
  } = inferBaseSignals(params);
  const modalities: ModelModality[] = [];

  if (!isSpeechToText) {
    modalities.push("text");
  }
  if (isSpeechToText) {
    modalities.push("audio");
  }
  if (isImageEdit || inferredVision) {
    modalities.push("image");
  }
  if (isEmbedding) {
    modalities.push("text");
  }
  if (isRerank || isModeration || isTextToSpeech) {
    modalities.push("text");
  }

  return uniqueValues(modalities);
}

export function inferOutputModalities(
  params: InferModelTaxonomyParams,
): ModelModality[] {
  const explicitOutputModalities = normalizeModalities(
    params.explicitOutputModalities,
  );
  if (explicitOutputModalities.length > 0) {
    return explicitOutputModalities;
  }

  const { capabilities, isEmbedding, isSpeechToText, isTextToSpeech } =
    inferBaseSignals(params);
  const taskFamilies = inferModelTaskFamilies(params);
  const modalities: ModelModality[] = [];

  if (
    taskFamilies.some((family) =>
      [
        "chat",
        "reasoning",
        "vision_understanding",
        "speech_to_text",
        "rerank",
        "moderation",
      ].includes(family),
    )
  ) {
    modalities.push("text");
  }
  if (
    taskFamilies.includes("image_generation") ||
    taskFamilies.includes("image_edit")
  ) {
    modalities.push("image");
  }
  if (isTextToSpeech) {
    modalities.push("audio");
  }
  if (isEmbedding) {
    modalities.push("embedding");
  }
  if (capabilities.json_mode && !isSpeechToText) {
    modalities.push("json");
  }

  return uniqueValues(modalities);
}

export function inferRuntimeFeatures(
  params: InferModelTaxonomyParams,
): ModelRuntimeFeature[] {
  const explicitRuntimeFeatures = normalizeRuntimeFeatures(
    params.explicitRuntimeFeatures,
  );
  if (explicitRuntimeFeatures.length > 0) {
    return explicitRuntimeFeatures;
  }

  const capabilities = params.capabilities ?? {};
  const taskFamilies = inferModelTaskFamilies(params);
  const providerType = normalize(params.providerType || params.providerId);
  const features: ModelRuntimeFeature[] = [];

  if (capabilities.streaming !== false) {
    features.push("streaming");
  }
  if (capabilities.tools || capabilities.function_calling) {
    features.push("tool_calling");
  }
  if (capabilities.json_mode) {
    features.push("json_schema");
  }
  if (capabilities.reasoning || taskFamilies.includes("reasoning")) {
    features.push("reasoning");
  }
  if (providerType === "openai-response" || providerType === "codex") {
    features.push("responses_api");
  }
  if (["openai", "new-api", "azure-openai", "gateway"].includes(providerType)) {
    features.push("chat_completions_api");
  }
  if (
    taskFamilies.includes("image_generation") ||
    taskFamilies.includes("image_edit")
  ) {
    features.push("images_api");
  }

  return uniqueValues(features);
}

export function inferModelDeploymentSource(
  params: InferModelTaxonomyParams,
): ModelDeploymentSource {
  if (params.deploymentSource) {
    return params.deploymentSource;
  }

  const searchText = buildSearchText([
    params.providerId,
    params.providerType,
    params.description,
  ]);
  if (LOCAL_PROVIDER_PATTERN.test(searchText)) {
    return "local";
  }
  if (OEM_PROVIDER_PATTERN.test(searchText)) {
    return "oem_cloud";
  }

  return "user_cloud";
}

export function inferModelManagementPlane(
  params: InferModelTaxonomyParams,
): ModelManagementPlane {
  if (params.managementPlane) {
    return params.managementPlane;
  }

  const deploymentSource = inferModelDeploymentSource(params);
  if (deploymentSource === "local") {
    return "local_settings";
  }
  if (deploymentSource === "oem_cloud") {
    return "oem_control_plane";
  }

  return "local_settings";
}

export function inferModelAliasSource(
  params: InferModelTaxonomyParams,
): ModelAliasSource | null {
  if (params.aliasSource) {
    return params.aliasSource;
  }

  if (
    params.providerModelId &&
    params.canonicalModelId &&
    normalize(params.providerModelId) !== normalize(params.canonicalModelId)
  ) {
    return "relay";
  }

  return null;
}

export function inferModelCapabilities(
  params: InferModelTaxonomyParams,
): ModelCapabilities {
  const taskFamilies = inferModelTaskFamilies(params);
  return {
    vision: taskFamilies.includes("vision_understanding"),
    tools:
      params.capabilities?.tools ?? !taskFamilies.includes("image_generation"),
    streaming: params.capabilities?.streaming ?? true,
    json_mode:
      params.capabilities?.json_mode ??
      ![
        "image_generation",
        "image_edit",
        "speech_to_text",
        "text_to_speech",
        "embedding",
        "rerank",
      ].some((family) => taskFamilies.includes(family as ModelTaskFamily)),
    function_calling:
      params.capabilities?.function_calling ??
      ![
        "image_generation",
        "image_edit",
        "speech_to_text",
        "text_to_speech",
        "embedding",
        "rerank",
      ].some((family) => taskFamilies.includes(family as ModelTaskFamily)),
    reasoning:
      params.capabilities?.reasoning === true ||
      taskFamilies.includes("reasoning"),
  };
}

export function getModelTaskFamilies(
  model: ModelTaxonomyInput,
): ModelTaskFamily[] {
  return inferModelTaskFamilies(modelToTaxonomyParams(model));
}

export function modelSupportsTaskFamily(
  model: ModelTaxonomyInput,
  family: ModelTaskFamily,
): boolean {
  return getModelTaskFamilies(model).includes(family);
}

export function getModelInputModalities(
  model: ModelTaxonomyInput,
): ModelModality[] {
  return inferInputModalities(modelToTaxonomyParams(model));
}

export function getModelOutputModalities(
  model: ModelTaxonomyInput,
): ModelModality[] {
  return inferOutputModalities(modelToTaxonomyParams(model));
}

export function getModelRuntimeFeatures(
  model: ModelTaxonomyInput,
): ModelRuntimeFeature[] {
  return inferRuntimeFeatures(modelToTaxonomyParams(model));
}

export function getModelDeploymentSource(
  model: ModelTaxonomyInput,
): ModelDeploymentSource {
  return inferModelDeploymentSource(modelToTaxonomyParams(model));
}

export function getModelManagementPlane(
  model: ModelTaxonomyInput,
): ModelManagementPlane {
  return inferModelManagementPlane(modelToTaxonomyParams(model));
}

export function getModelAliasSource(
  model: ModelTaxonomyInput,
): ModelAliasSource | null {
  return inferModelAliasSource(modelToTaxonomyParams(model));
}

export function getModelCapabilitySummary(
  model: ModelTaxonomyInput & Pick<EnhancedModelMetadata, "limits">,
): ModelCapabilitySummary {
  const task_families = getModelTaskFamilies(model);
  const input_modalities = getModelInputModalities(model);
  const output_modalities = getModelOutputModalities(model);
  const runtime_features = getModelRuntimeFeatures(model);
  const capabilities = inferModelCapabilities(modelToTaxonomyParams(model));

  return {
    capabilities,
    task_families,
    input_modalities,
    output_modalities,
    runtime_features,
    supports_tools:
      capabilities.tools || runtime_features.includes("tool_calling"),
    supports_reasoning:
      capabilities.reasoning ||
      task_families.includes("reasoning") ||
      runtime_features.includes("reasoning"),
    supports_prompt_cache: runtime_features.includes("prompt_cache"),
    supports_media_input: input_modalities.some(isMediaModality),
    supports_media_output: output_modalities.some(isMediaModality),
    context_length: model.limits.context_length,
    max_output_tokens: model.limits.max_output_tokens,
  };
}

function modelToTaxonomyParams(
  model: ModelTaxonomyInput,
): InferModelTaxonomyParams {
  return {
    modelId: model.id,
    providerId: model.provider_id,
    family: model.family,
    description: model.description,
    capabilities: model.capabilities,
    explicitTaskFamilies: model.task_families,
    explicitInputModalities: model.input_modalities,
    explicitOutputModalities: model.output_modalities,
    explicitRuntimeFeatures: model.runtime_features,
    deploymentSource: model.deployment_source,
    managementPlane: model.management_plane,
    source: model.source,
    providerModelId: model.provider_model_id,
    canonicalModelId: model.canonical_model_id,
    aliasSource: model.alias_source,
  };
}

function isMediaModality(modality: ModelModality): boolean {
  return modality === "image" || modality === "audio" || modality === "video";
}
