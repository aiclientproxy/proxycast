//! 模型注册服务
//!
//! 管理运行期模型数据和 Provider 实时模型读取。
//!
//! 旧的本地模型资源目录已下线；模型列表以 Provider 实时接口和用户
//! 显式配置的 `models` 为准。

use lime_core::api_host_utils::{
    is_openai_responses_compatible_host, normalize_openai_model_discovery_host,
};
use lime_core::database::dao::api_key_provider::{infer_managed_runtime_spec, ApiProviderType};
use lime_core::database::dao::route_state::RouteStateDao;
use lime_core::database::DbConnection;
use lime_core::image_generation_matcher::is_likely_image_generation_search_text;
use lime_core::models::model_registry::{
    EnhancedModelMetadata, ModelAliasSource, ModelCapabilities, ModelCapabilityProvenance,
    ModelDeploymentSource, ModelLimits, ModelManagementPlane, ModelModality,
    ModelReasoningEffortOption, ModelReasoningEffortSource, ModelReasoningEffortSupport,
    ModelRuntimeFeature, ModelServiceTier, ModelSource, ModelStatus, ModelSyncState,
    ModelTaskFamily, ModelTier, ModelVisibility, ProviderAliasConfig, ProviderModelConfig,
    UserModelPreference,
};
use model_provider::canonical::{maybe_get_canonical_model, CanonicalModel};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use url::form_urlencoded;

mod managed_model_fetch_access;
mod runtime_metadata;
pub use runtime_metadata::{
    ProviderModelCacheAccess, ProviderModelRegistryMetadata, ProviderModelRegistryMetadataSource,
};

const LIME_TENANT_HEADER: &str = "X-Lime-Tenant-ID";
const LIME_TENANT_PARAM: &str = "lime_tenant_id";
const PROVIDER_MODELS_CACHE_KEY_PREFIX: &str = "provider_models_fetch_cache:";
const PROVIDER_MODELS_CACHE_TTL_SECONDS: i64 = 10 * 24 * 60 * 60;
const PROVIDER_MODELS_CACHE_TAXONOMY_VERSION: u32 = 2;
const XIAOMI_MODEL_FETCH_HOST_KEYWORDS: &[&str] = &["xiaomimimo.com"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelFetchProtocol {
    OpenAiCompatible,
    ResponsesCompatible,
    Anthropic,
    Gemini,
    Ollama,
    Unsupported,
}

#[derive(Debug, Clone)]
struct PreparedModelFetchRequest {
    protocol: ModelFetchProtocol,
    url: String,
    headers: Vec<(String, String)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProviderModelsCachePayload {
    #[serde(default)]
    taxonomy_version: u32,
    provider_id: String,
    api_host: String,
    provider_type: Option<String>,
    request_url: Option<String>,
    fetched_at: i64,
    expires_at: i64,
    models: Vec<EnhancedModelMetadata>,
}

fn normalize_identifier(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}

fn build_search_text(parts: &[Option<String>]) -> String {
    parts
        .iter()
        .filter_map(|part| {
            part.as_ref()
                .map(|value| normalize_identifier(value))
                .filter(|value| !value.is_empty())
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn text_contains_any(text: &str, keywords: &[&str]) -> bool {
    keywords.iter().any(|keyword| text.contains(keyword))
}

fn push_unique<T: PartialEq>(target: &mut Vec<T>, value: T) {
    if !target.contains(&value) {
        target.push(value);
    }
}

fn model_service_tier_from_value(value: &serde_json::Value) -> Option<ModelServiceTier> {
    match value {
        serde_json::Value::String(id) => {
            let id = id.trim();
            (!id.is_empty()).then(|| ModelServiceTier {
                id: id.to_string(),
                name: id.to_string(),
                description: String::new(),
            })
        }
        serde_json::Value::Object(fields) => {
            let id = fields
                .get("id")
                .or_else(|| fields.get("value"))
                .and_then(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|id| !id.is_empty())?;
            let name = fields
                .get("name")
                .and_then(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .unwrap_or(id);
            let description = fields
                .get("description")
                .and_then(serde_json::Value::as_str)
                .map(str::trim)
                .unwrap_or_default();
            Some(ModelServiceTier {
                id: id.to_string(),
                name: name.to_string(),
                description: description.to_string(),
            })
        }
        _ => None,
    }
}

fn parse_model_service_tiers(value: Option<&serde_json::Value>) -> Vec<ModelServiceTier> {
    let Some(serde_json::Value::Array(items)) = value else {
        return Vec::new();
    };
    let mut tiers = Vec::new();
    for item in items {
        let Some(tier) = model_service_tier_from_value(item) else {
            continue;
        };
        if tiers
            .iter()
            .any(|existing: &ModelServiceTier| existing.id == tier.id)
        {
            continue;
        }
        tiers.push(tier);
    }
    tiers
}

fn parse_modality(value: &str) -> Option<ModelModality> {
    match normalize_identifier(value).as_str() {
        "text" => Some(ModelModality::Text),
        "image" => Some(ModelModality::Image),
        "audio" => Some(ModelModality::Audio),
        "video" => Some(ModelModality::Video),
        "file" => Some(ModelModality::File),
        "embedding" => Some(ModelModality::Embedding),
        "json" => Some(ModelModality::Json),
        _ => None,
    }
}

fn parse_modalities(values: &[String]) -> Vec<ModelModality> {
    let mut modalities = Vec::new();
    for value in values {
        if let Some(modality) = parse_modality(value) {
            push_unique(&mut modalities, modality);
        }
    }
    modalities
}

fn extract_string_list(value: Option<&serde_json::Value>) -> Vec<String> {
    let Some(value) = value else {
        return Vec::new();
    };

    match value {
        serde_json::Value::String(item) => vec![item.clone()],
        serde_json::Value::Array(items) => items
            .iter()
            .flat_map(|item| extract_string_list(Some(item)))
            .collect(),
        serde_json::Value::Object(map) => map
            .iter()
            .filter_map(|(key, value)| match value {
                serde_json::Value::Bool(true) => Some(key.clone()),
                serde_json::Value::String(item) => Some(item.clone()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn parse_modalities_value(value: Option<&serde_json::Value>) -> Vec<ModelModality> {
    parse_modalities(&extract_string_list(value))
}

fn merge_modalities(target: &mut Vec<ModelModality>, modalities: Vec<ModelModality>) {
    for modality in modalities {
        push_unique(target, modality);
    }
}

fn parse_task_family(value: &str) -> Option<ModelTaskFamily> {
    match normalize_identifier(value)
        .replace(['-', ' '], "_")
        .as_str()
    {
        "llm" | "chat" | "dialog" | "conversation" | "text_generation" => {
            Some(ModelTaskFamily::Chat)
        }
        "reasoning" | "thinking" => Some(ModelTaskFamily::Reasoning),
        "vision"
        | "vlm"
        | "multimodal"
        | "omni"
        | "vision_understanding"
        | "vision_language"
        | "image_input"
        | "input_image"
        | "image_understanding" => Some(ModelTaskFamily::VisionUnderstanding),
        "image" | "image_generation" | "text_to_image" | "drawing" | "image_output" => {
            Some(ModelTaskFamily::ImageGeneration)
        }
        "image_edit" | "edit" | "img2img" | "inpaint" | "outpaint" => {
            Some(ModelTaskFamily::ImageEdit)
        }
        "video" | "video_generation" | "text_to_video" | "image_to_video" => {
            Some(ModelTaskFamily::VideoGeneration)
        }
        "speech_to_text" | "stt" | "asr" | "transcribe" | "transcription" => {
            Some(ModelTaskFamily::SpeechToText)
        }
        "text_to_speech" | "tts" | "speech_synthesis" => Some(ModelTaskFamily::TextToSpeech),
        "embedding" | "embed" => Some(ModelTaskFamily::Embedding),
        "rerank" | "re_rank" | "retrieval" => Some(ModelTaskFamily::Rerank),
        "moderation" | "safety" => Some(ModelTaskFamily::Moderation),
        _ => None,
    }
}

fn parse_task_families_value(value: Option<&serde_json::Value>) -> Vec<ModelTaskFamily> {
    let mut families = Vec::new();
    for item in extract_string_list(value) {
        if let Some(family) = parse_task_family(&item) {
            push_unique(&mut families, family);
        }
    }
    families
}

fn parse_runtime_feature(value: &str) -> Option<ModelRuntimeFeature> {
    match normalize_identifier(value)
        .replace(['-', ' '], "_")
        .as_str()
    {
        "streaming" | "stream" => Some(ModelRuntimeFeature::Streaming),
        "tool_calling" | "tools" | "function_calling" | "functions" => {
            Some(ModelRuntimeFeature::ToolCalling)
        }
        "json_schema" | "json_mode" | "structured_output" => Some(ModelRuntimeFeature::JsonSchema),
        "reasoning" | "thinking" => Some(ModelRuntimeFeature::Reasoning),
        "prompt_cache" | "prompt_caching" => Some(ModelRuntimeFeature::PromptCache),
        "responses_api" | "responses" => Some(ModelRuntimeFeature::ResponsesApi),
        "chat_completions_api" | "chat_completions" => {
            Some(ModelRuntimeFeature::ChatCompletionsApi)
        }
        "images_api" | "image_generation" | "image_edit" => Some(ModelRuntimeFeature::ImagesApi),
        _ => None,
    }
}

fn parse_runtime_features_value(value: Option<&serde_json::Value>) -> Vec<ModelRuntimeFeature> {
    let mut features = Vec::new();
    for item in extract_string_list(value) {
        if let Some(feature) = parse_runtime_feature(&item) {
            push_unique(&mut features, feature);
        }
    }
    features
}

fn collect_capability_signals(value: Option<&serde_json::Value>, target: &mut Vec<String>) {
    let Some(value) = value else {
        return;
    };

    match value {
        serde_json::Value::String(item) => target.push(item.clone()),
        serde_json::Value::Array(items) => {
            for item in items {
                collect_capability_signals(Some(item), target);
            }
        }
        serde_json::Value::Object(map) => {
            for (key, value) in map {
                match value {
                    serde_json::Value::Bool(true) => target.push(key.clone()),
                    serde_json::Value::String(item) => target.push(item.clone()),
                    serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
                        collect_capability_signals(Some(value), target);
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

fn merge_api_capability_signals(
    capabilities: Option<&serde_json::Value>,
    task_families: &mut Vec<ModelTaskFamily>,
    input_modalities: &mut Vec<ModelModality>,
    output_modalities: &mut Vec<ModelModality>,
    runtime_features: &mut Vec<ModelRuntimeFeature>,
) {
    let mut signals = Vec::new();
    collect_capability_signals(capabilities, &mut signals);

    for signal in signals {
        let normalized = normalize_identifier(&signal).replace(['-', ' '], "_");
        match normalized.as_str() {
            "vision"
            | "vision_understanding"
            | "vision_language"
            | "vlm"
            | "multimodal"
            | "omni"
            | "image_input"
            | "input_image"
            | "images_input"
            | "image_understanding"
            | "supports_vision"
            | "supports_images" => {
                push_unique(task_families, ModelTaskFamily::VisionUnderstanding);
                push_unique(input_modalities, ModelModality::Image);
            }
            "image_generation" | "text_to_image" | "image_output" | "images_api" => {
                push_unique(task_families, ModelTaskFamily::ImageGeneration);
                push_unique(output_modalities, ModelModality::Image);
                push_unique(runtime_features, ModelRuntimeFeature::ImagesApi);
            }
            "image_edit" | "img2img" | "inpaint" | "outpaint" => {
                push_unique(task_families, ModelTaskFamily::ImageEdit);
                push_unique(input_modalities, ModelModality::Image);
                push_unique(output_modalities, ModelModality::Image);
                push_unique(runtime_features, ModelRuntimeFeature::ImagesApi);
            }
            "reasoning" | "thinking" => {
                push_unique(task_families, ModelTaskFamily::Reasoning);
                push_unique(runtime_features, ModelRuntimeFeature::Reasoning);
            }
            "tools" | "tool_calling" | "function_calling" | "functions" => {
                push_unique(runtime_features, ModelRuntimeFeature::ToolCalling);
            }
            "json_mode" | "json_schema" | "structured_output" => {
                push_unique(runtime_features, ModelRuntimeFeature::JsonSchema);
            }
            "streaming" | "stream" => {
                push_unique(runtime_features, ModelRuntimeFeature::Streaming);
            }
            "embedding" | "embed" => {
                push_unique(task_families, ModelTaskFamily::Embedding);
                push_unique(output_modalities, ModelModality::Embedding);
            }
            "rerank" | "re_rank" => {
                push_unique(task_families, ModelTaskFamily::Rerank);
            }
            "moderation" | "safety" => {
                push_unique(task_families, ModelTaskFamily::Moderation);
            }
            _ => {}
        }
    }
}

fn parse_reasoning_effort_level(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn reasoning_effort_option_label(id: &str) -> String {
    let mut characters = id.chars();
    match characters.next() {
        Some(first) => first.to_uppercase().chain(characters).collect(),
        None => String::new(),
    }
}

fn reasoning_effort_option_from_value(
    value: &serde_json::Value,
) -> Option<ModelReasoningEffortOption> {
    match value {
        serde_json::Value::String(value) => {
            let canonical = parse_reasoning_effort_level(value)?;
            let id = canonical.clone();
            Some(ModelReasoningEffortOption {
                label: reasoning_effort_option_label(&id),
                id,
                value: canonical,
                description: None,
                default: false,
            })
        }
        serde_json::Value::Object(source) => {
            let canonical = source
                .get("value")
                .and_then(serde_json::Value::as_str)
                .and_then(parse_reasoning_effort_level)?;
            let id = source
                .get("id")
                .and_then(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .unwrap_or(canonical.as_str())
                .to_string();
            let label = source
                .get("label")
                .and_then(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .unwrap_or_else(|| reasoning_effort_option_label(&id));
            let description = source
                .get("description")
                .and_then(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string);
            Some(ModelReasoningEffortOption {
                id,
                value: canonical,
                label,
                description,
                default: source
                    .get("default")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false),
            })
        }
        _ => None,
    }
}

fn parse_reasoning_effort_options_value(
    value: Option<&serde_json::Value>,
) -> Vec<ModelReasoningEffortOption> {
    let Some(serde_json::Value::Array(items)) = value else {
        return Vec::new();
    };
    let mut options = Vec::new();
    for item in items {
        let Some(option) = reasoning_effort_option_from_value(item) else {
            continue;
        };
        if options.iter().any(|existing: &ModelReasoningEffortOption| {
            existing.id.eq_ignore_ascii_case(&option.id) || existing.value == option.value
        }) {
            continue;
        }
        options.push(option);
    }
    options
}

fn push_unique_reasoning_effort_level(levels: &mut Vec<String>, level: String) {
    if !levels.contains(&level) {
        levels.push(level);
    }
}

fn parse_reasoning_effort_levels_value(value: Option<&serde_json::Value>) -> Vec<String> {
    let mut levels = Vec::new();
    let Some(value) = value else {
        return levels;
    };

    match value {
        serde_json::Value::String(item) => {
            if let Some(level) = parse_reasoning_effort_level(item) {
                push_unique_reasoning_effort_level(&mut levels, level);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                for level in parse_reasoning_effort_levels_value(Some(item)) {
                    push_unique_reasoning_effort_level(&mut levels, level);
                }
            }
        }
        serde_json::Value::Object(map) => {
            for (key, value) in map {
                if api_field_matches(
                    key,
                    &[
                        "value",
                        "level",
                        "effort",
                        "levels",
                        "efforts",
                        "supported_levels",
                        "reasoning_effort_levels",
                        "reasoning_efforts",
                        "supported_reasoning_efforts",
                    ],
                ) {
                    for level in parse_reasoning_effort_levels_value(Some(value)) {
                        push_unique_reasoning_effort_level(&mut levels, level);
                    }
                } else if matches!(value, serde_json::Value::Bool(true))
                    && !api_field_matches(key, &["supported", "enabled", "default"])
                {
                    if let Some(level) = parse_reasoning_effort_level(key) {
                        push_unique_reasoning_effort_level(&mut levels, level);
                    }
                }
            }
        }
        _ => {}
    }

    levels
}

fn normalize_api_field_name(value: &str) -> String {
    let mut normalized = String::new();
    let mut previous_was_separator = true;
    for character in value.trim().chars() {
        if character.is_ascii_uppercase() {
            if !previous_was_separator && !normalized.ends_with('_') {
                normalized.push('_');
            }
            normalized.push(character.to_ascii_lowercase());
            previous_was_separator = false;
        } else if character == '-' || character == ' ' || character == '.' {
            if !normalized.ends_with('_') {
                normalized.push('_');
            }
            previous_was_separator = true;
        } else {
            normalized.push(character.to_ascii_lowercase());
            previous_was_separator = character == '_';
        }
    }
    normalized.trim_matches('_').to_string()
}

fn api_field_matches(key: &str, expected: &[&str]) -> bool {
    let normalized = normalize_api_field_name(key);
    expected
        .iter()
        .any(|candidate| normalized == normalize_api_field_name(candidate))
}

fn collect_named_reasoning_effort_levels(
    value: Option<&serde_json::Value>,
    field_names: &[&str],
    levels: &mut Vec<String>,
) {
    let Some(value) = value else {
        return;
    };
    let serde_json::Value::Object(map) = value else {
        return;
    };

    for (key, item) in map {
        if api_field_matches(key, field_names) {
            for level in parse_reasoning_effort_levels_value(Some(item)) {
                push_unique_reasoning_effort_level(levels, level);
            }
        }
        if item.is_object() {
            collect_named_reasoning_effort_levels(Some(item), field_names, levels);
        }
    }
}

fn contains_reasoning_effort_parameter(value: Option<&serde_json::Value>) -> bool {
    let Some(value) = value else {
        return false;
    };

    match value {
        serde_json::Value::String(item) => matches!(
            normalize_api_field_name(item).as_str(),
            "reasoning_effort" | "reasoning_effort_level" | "reasoning_effort_levels"
        ),
        serde_json::Value::Array(items) => items
            .iter()
            .any(|item| contains_reasoning_effort_parameter(Some(item))),
        serde_json::Value::Object(map) => map.iter().any(|(key, item)| {
            let is_reasoning_effort_key = matches!(
                normalize_api_field_name(key).as_str(),
                "reasoning_effort"
                    | "reasoning_effort_level"
                    | "reasoning_effort_levels"
                    | "supported_reasoning_efforts"
                    | "reasoning_efforts"
            );
            match item {
                serde_json::Value::Bool(true) => is_reasoning_effort_key,
                serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
                    contains_reasoning_effort_parameter(Some(item))
                }
                serde_json::Value::String(_) => is_reasoning_effort_key,
                _ => false,
            }
        }),
        _ => false,
    }
}

fn explicit_reasoning_effort_supported(value: Option<&serde_json::Value>) -> bool {
    let Some(value) = value else {
        return false;
    };

    match value {
        serde_json::Value::Bool(supported) => *supported,
        serde_json::Value::String(_) | serde_json::Value::Array(_) => {
            !parse_reasoning_effort_levels_value(Some(value)).is_empty()
        }
        serde_json::Value::Object(map) => {
            let explicitly_disabled = map.iter().any(|(key, item)| {
                api_field_matches(key, &["supported", "enabled"])
                    && matches!(item, serde_json::Value::Bool(false))
            });
            if explicitly_disabled {
                return false;
            }
            map.iter().any(|(key, item)| {
                (api_field_matches(key, &["supported", "enabled"])
                    && matches!(item, serde_json::Value::Bool(true)))
                    || (api_field_matches(
                        key,
                        &[
                            "value",
                            "level",
                            "effort",
                            "levels",
                            "efforts",
                            "supported_levels",
                            "reasoning_effort_levels",
                            "reasoning_efforts",
                            "supported_reasoning_efforts",
                        ],
                    ) && !parse_reasoning_effort_levels_value(Some(item)).is_empty())
            })
        }
        _ => false,
    }
}

fn explicit_reasoning_effort_disabled(value: Option<&serde_json::Value>) -> bool {
    let Some(value) = value else {
        return false;
    };

    match value {
        serde_json::Value::Bool(supported) => !supported,
        serde_json::Value::Object(map) => map.iter().any(|(key, item)| {
            api_field_matches(key, &["supported", "enabled"])
                && matches!(item, serde_json::Value::Bool(false))
        }),
        _ => false,
    }
}

fn contains_disabled_reasoning_effort_support(value: Option<&serde_json::Value>) -> bool {
    let Some(value) = value else {
        return false;
    };

    match value {
        serde_json::Value::Array(items) => items
            .iter()
            .any(|item| contains_disabled_reasoning_effort_support(Some(item))),
        serde_json::Value::Object(map) => map.iter().any(|(key, item)| {
            let is_reasoning_effort_key = api_field_matches(
                key,
                &[
                    "reasoning_effort",
                    "reasoningEffort",
                    "supported_reasoning_efforts",
                    "supportedReasoningEfforts",
                ],
            );
            (is_reasoning_effort_key && explicit_reasoning_effort_disabled(Some(item)))
                || contains_disabled_reasoning_effort_support(Some(item))
        }),
        _ => false,
    }
}

fn collect_default_reasoning_effort_level(value: Option<&serde_json::Value>) -> Option<String> {
    let serde_json::Value::Object(map) = value? else {
        return None;
    };

    for (key, item) in map {
        if api_field_matches(
            key,
            &[
                "default_reasoning_effort",
                "defaultReasoningEffort",
                "default_effort",
                "defaultEffort",
                "default",
            ],
        ) {
            if let Some(default) = parse_reasoning_effort_levels_value(Some(item))
                .into_iter()
                .next()
            {
                return Some(default);
            }
        }
        if item.is_object() {
            if let Some(default) = collect_default_reasoning_effort_level(Some(item)) {
                return Some(default);
            }
        }
    }

    None
}

fn resolve_api_reasoning_effort_support(
    model: &ApiModelResponse,
) -> Option<ModelReasoningEffortSupport> {
    if explicit_reasoning_effort_disabled(model.reasoning_effort.as_ref())
        || contains_disabled_reasoning_effort_support(model.reasoning.as_ref())
        || contains_disabled_reasoning_effort_support(model.capabilities.as_ref())
    {
        return None;
    }

    let mut options = parse_reasoning_effort_options_value(model.reasoning_efforts.as_ref());
    let mut levels = options
        .iter()
        .map(|option| option.value.clone())
        .collect::<Vec<_>>();
    for value in [
        model.reasoning_effort.as_ref(),
        model.reasoning_effort_levels.as_ref(),
        model.reasoning_efforts.as_ref(),
        model.supported_reasoning_efforts.as_ref(),
    ] {
        for level in parse_reasoning_effort_levels_value(value) {
            push_unique_reasoning_effort_level(&mut levels, level);
        }
    }

    for value in [
        model.reasoning.as_ref(),
        model.capabilities.as_ref(),
        model.reasoning_effort.as_ref(),
    ] {
        collect_named_reasoning_effort_levels(
            value,
            &[
                "levels",
                "efforts",
                "supported",
                "supported_levels",
                "supportedLevels",
                "effort",
                "reasoning_effort",
                "reasoningEffort",
                "reasoning_effort_levels",
                "reasoningEffortLevels",
                "reasoning_efforts",
                "supported_reasoning_efforts",
                "supportedReasoningEfforts",
            ],
            &mut levels,
        );
    }

    let supports_parameter = [
        model.supported_parameters.as_ref(),
        model.reasoning_effort.as_ref(),
        model.reasoning.as_ref(),
        model.capabilities.as_ref(),
    ]
    .into_iter()
    .any(contains_reasoning_effort_parameter)
        || explicit_reasoning_effort_supported(model.reasoning_effort.as_ref());

    if levels.is_empty() && !supports_parameter {
        return None;
    }

    if options.is_empty() {
        options = levels
            .iter()
            .cloned()
            .map(|value| {
                let id = value.clone();
                ModelReasoningEffortOption {
                    label: reasoning_effort_option_label(&id),
                    id,
                    value,
                    description: None,
                    default: false,
                }
            })
            .collect();
    }

    let default = options
        .iter()
        .find(|option| option.default)
        .map(|option| option.value.clone())
        .or_else(|| {
            [
                model.reasoning_effort.as_ref(),
                model.reasoning.as_ref(),
                model.capabilities.as_ref(),
            ]
            .into_iter()
            .find_map(collect_default_reasoning_effort_level)
        })
        .filter(|level| levels.contains(level));

    Some(ModelReasoningEffortSupport {
        supported: true,
        levels,
        options,
        default,
        source: Some(ModelReasoningEffortSource::Api),
    })
}

fn model_id_has_openai_vision_capability(model_id: &str, text: &str) -> bool {
    let model = normalize_identifier(model_id);
    let model_tail = model.rsplit('/').next().unwrap_or(model.as_str());
    text_contains_any(
        text,
        &[
            "chatgpt-4o",
            "gpt-5",
            "gpt-4o",
            "gpt-4.1",
            "gpt-4.5",
            "gpt-4-turbo",
            "codex",
        ],
    ) || model_tail == "o1"
        || model_tail == "o1-pro"
        || model_tail.starts_with("o1-pro-")
        || model_tail.starts_with("o1-202")
        || model_tail == "o3"
        || (model_tail.starts_with("o3-") && !model_tail.starts_with("o3-mini"))
        || model_tail.starts_with("o4-mini")
}

fn model_id_has_agnes_vision_capability(model_id: &str) -> bool {
    let model = normalize_identifier(model_id);
    let model_tail = model.rsplit('/').next().unwrap_or(model.as_str());
    model_tail == "agnes-2.0-flash"
}

fn model_id_has_qwen_vision_capability(model_id: &str, text: &str) -> bool {
    let model = normalize_identifier(model_id);
    let model_tail = model.rsplit('/').next().unwrap_or(model.as_str());
    (text.contains("qwen") && (text.contains("vl") || text.contains("vision")))
        || text.contains("qvq")
        || model_tail.starts_with("qwen3.5")
        || model_tail.starts_with("qwen3-5")
        || model_tail.starts_with("qwen3.6")
        || model_tail.starts_with("qwen3-6")
}

fn model_id_has_xai_vision_capability(model_id: &str) -> bool {
    let model = normalize_identifier(model_id);
    model.contains("grok-vision")
        || model.contains("grok-2-vision")
        || model.starts_with("grok-4-1")
        || model.starts_with("grok-4-fast")
        || model.starts_with("grok-4.20")
        || model.starts_with("grok-4.3")
}

fn model_id_has_mistral_vision_capability(model_id: &str, text: &str) -> bool {
    let model = normalize_identifier(model_id);
    let model_tail = model.rsplit('/').next().unwrap_or(model.as_str());
    text.contains("pixtral")
        || model_tail.starts_with("mistral-small-latest")
        || model_tail.starts_with("mistral-large-2512")
        || model_tail.starts_with("mistral-medium-3.1")
}

fn model_id_has_gemma_vision_capability(model_id: &str) -> bool {
    let model = normalize_identifier(model_id);
    let model_tail = model.rsplit('/').next().unwrap_or(model.as_str());
    model_tail.starts_with("gemma-3") && !model_tail.starts_with("gemma-3n")
}

fn model_id_has_llama_vision_capability(model_id: &str) -> bool {
    let model = normalize_identifier(model_id);
    model.contains("llama-4-maverick") || model.contains("llama-4-scout")
}

fn infer_vision_capability(
    model_id: &str,
    provider_id: Option<&str>,
    family: Option<&str>,
    description: Option<&str>,
) -> bool {
    let text = build_search_text(&[
        Some(model_id.to_string()),
        family.map(ToString::to_string),
        description.map(ToString::to_string),
    ]);
    if text.is_empty() {
        return false;
    }

    if text_contains_any(
        &text,
        &[
            "embedding",
            "embed",
            "rerank",
            "tts",
            "stt",
            "transcribe",
            "transcription",
            "speech",
            "audio",
            "moderation",
        ],
    ) || is_likely_image_generation_search_text(&text)
    {
        return false;
    }

    if text_contains_any(
        &text,
        &[
            "vision",
            "multimodal",
            "multi-modal",
            "omni",
            "image-input",
            "image understanding",
        ],
    ) {
        return true;
    }

    let provider = provider_id.map(normalize_identifier).unwrap_or_default();
    let openai_like = model_id_has_openai_vision_capability(model_id, &text);
    if provider == "openai" || provider == "codex" {
        return openai_like;
    }
    if provider == "gemini" || provider == "google" {
        return text.contains("gemini") || model_id_has_gemma_vision_capability(model_id);
    }
    if provider == "anthropic" || provider == "claude" {
        return text.contains("claude");
    }
    if provider == "qwen" || provider == "alibaba" {
        return model_id_has_qwen_vision_capability(model_id, &text);
    }
    if provider == "zhipuai" {
        return text.contains("glm-") && text.contains('v');
    }
    if provider == "xai" || provider == "x-ai" {
        return model_id_has_xai_vision_capability(model_id);
    }
    if provider == "mistral" || provider == "mistralai" {
        return model_id_has_mistral_vision_capability(model_id, &text);
    }
    if provider == "llama" || provider == "meta" || provider == "meta-llama" {
        return model_id_has_llama_vision_capability(model_id);
    }

    openai_like
        || model_id_has_agnes_vision_capability(model_id)
        || text.contains("gemini")
        || text.contains("claude")
        || model_id_has_qwen_vision_capability(model_id, &text)
        || (text.contains("glm-") && text.contains('v'))
        || model_id_has_xai_vision_capability(model_id)
        || model_id_has_mistral_vision_capability(model_id, &text)
        || model_id_has_gemma_vision_capability(model_id)
        || model_id_has_llama_vision_capability(model_id)
}

fn infer_image_generation_capability(
    model_id: &str,
    family: Option<&str>,
    description: Option<&str>,
    provider_model_id: Option<&str>,
    canonical_model_id: Option<&str>,
    input_modalities: &[ModelModality],
    output_modalities: &[ModelModality],
) -> bool {
    output_modalities.contains(&ModelModality::Image)
        || is_likely_image_generation_search_text(&build_search_text(&[
            Some(model_id.to_string()),
            family.map(ToString::to_string),
            description.map(ToString::to_string),
            provider_model_id.map(ToString::to_string),
            canonical_model_id.map(ToString::to_string),
        ]))
        || (input_modalities.contains(&ModelModality::Image)
            && output_modalities.contains(&ModelModality::Image))
}

fn infer_image_edit_capability(
    model_id: &str,
    family: Option<&str>,
    description: Option<&str>,
    input_modalities: &[ModelModality],
    output_modalities: &[ModelModality],
) -> bool {
    text_contains_any(
        &build_search_text(&[
            Some(model_id.to_string()),
            family.map(ToString::to_string),
            description.map(ToString::to_string),
        ]),
        &[
            "edit",
            "inpaint",
            "outpaint",
            "img2img",
            "image-edit",
            "image_edit",
            "image edits",
        ],
    ) || (input_modalities.contains(&ModelModality::Image)
        && output_modalities.contains(&ModelModality::Image))
}

fn infer_model_task_families(
    model_id: &str,
    provider_id: Option<&str>,
    family: Option<&str>,
    description: Option<&str>,
    capabilities: Option<&ModelCapabilities>,
    explicit_task_families: &[ModelTaskFamily],
    input_modalities: &[ModelModality],
    output_modalities: &[ModelModality],
    provider_model_id: Option<&str>,
    canonical_model_id: Option<&str>,
) -> Vec<ModelTaskFamily> {
    let text = build_search_text(&[
        Some(model_id.to_string()),
        family.map(ToString::to_string),
        description.map(ToString::to_string),
        provider_model_id.map(ToString::to_string),
        canonical_model_id.map(ToString::to_string),
    ]);
    let inferred_reasoning = capabilities.map(|caps| caps.reasoning).unwrap_or(false);
    let has_explicit_vision_input = input_modalities.contains(&ModelModality::Image)
        && (output_modalities.is_empty() || output_modalities.contains(&ModelModality::Text));
    let inferred_vision_by_name =
        infer_vision_capability(model_id, provider_id, family, description)
            || provider_model_id
                .map(|id| infer_vision_capability(id, provider_id, family, description))
                .unwrap_or(false)
            || canonical_model_id
                .map(|id| infer_vision_capability(id, provider_id, family, description))
                .unwrap_or(false);
    let inferred_vision = capabilities.map(|caps| caps.vision).unwrap_or(false)
        || has_explicit_vision_input
        || inferred_vision_by_name;
    let is_embedding = text_contains_any(&text, &["embedding", "embed", "text-embedding"]);
    let is_rerank = text_contains_any(&text, &["rerank", "re-rank"]);
    let is_moderation = text_contains_any(&text, &["moderation"]);
    let is_speech_to_text = text_contains_any(
        &text,
        &[
            "stt",
            "asr",
            "speech-to-text",
            "speech to text",
            "transcribe",
            "transcription",
            "whisper",
        ],
    );
    let is_text_to_speech = text_contains_any(
        &text,
        &[
            "tts",
            "text-to-speech",
            "text to speech",
            "speech synthesis",
            "voice-synth",
        ],
    );
    let is_image_generation = infer_image_generation_capability(
        model_id,
        family,
        description,
        provider_model_id,
        canonical_model_id,
        input_modalities,
        output_modalities,
    );
    let is_image_edit = infer_image_edit_capability(
        model_id,
        family,
        description,
        input_modalities,
        output_modalities,
    );

    let mut families = explicit_task_families.to_vec();
    if is_embedding {
        push_unique(&mut families, ModelTaskFamily::Embedding);
    }
    if is_rerank {
        push_unique(&mut families, ModelTaskFamily::Rerank);
    }
    if is_moderation {
        push_unique(&mut families, ModelTaskFamily::Moderation);
    }
    if is_speech_to_text {
        push_unique(&mut families, ModelTaskFamily::SpeechToText);
    }
    if is_text_to_speech {
        push_unique(&mut families, ModelTaskFamily::TextToSpeech);
    }
    if is_image_generation {
        push_unique(&mut families, ModelTaskFamily::ImageGeneration);
    }
    if is_image_edit {
        push_unique(&mut families, ModelTaskFamily::ImageEdit);
    }
    if inferred_vision && (!is_image_generation || has_explicit_vision_input) {
        push_unique(&mut families, ModelTaskFamily::VisionUnderstanding);
    }
    if inferred_reasoning {
        push_unique(&mut families, ModelTaskFamily::Reasoning);
    }

    let specialized_only = families.iter().any(|family| {
        matches!(
            family,
            ModelTaskFamily::Embedding
                | ModelTaskFamily::Rerank
                | ModelTaskFamily::Moderation
                | ModelTaskFamily::SpeechToText
                | ModelTaskFamily::TextToSpeech
                | ModelTaskFamily::ImageGeneration
                | ModelTaskFamily::ImageEdit
                | ModelTaskFamily::VideoGeneration
        )
    });

    if !specialized_only
        || inferred_vision
        || inferred_reasoning
        || capabilities.map(|caps| caps.tools).unwrap_or(false)
        || capabilities
            .map(|caps| caps.function_calling)
            .unwrap_or(false)
        || capabilities.map(|caps| caps.json_mode).unwrap_or(false)
    {
        push_unique(&mut families, ModelTaskFamily::Chat);
    }

    families
}

fn infer_model_capabilities(task_families: &[ModelTaskFamily]) -> ModelCapabilities {
    let specialized_only = task_families.iter().any(|family| {
        matches!(
            family,
            ModelTaskFamily::ImageGeneration
                | ModelTaskFamily::ImageEdit
                | ModelTaskFamily::VideoGeneration
                | ModelTaskFamily::SpeechToText
                | ModelTaskFamily::TextToSpeech
                | ModelTaskFamily::Embedding
                | ModelTaskFamily::Rerank
        )
    });

    ModelCapabilities {
        vision: task_families.contains(&ModelTaskFamily::VisionUnderstanding),
        tools: !task_families.contains(&ModelTaskFamily::ImageGeneration)
            && !task_families.contains(&ModelTaskFamily::ImageEdit)
            && !task_families.contains(&ModelTaskFamily::VideoGeneration),
        streaming: true,
        json_mode: !specialized_only,
        function_calling: !task_families.contains(&ModelTaskFamily::ImageGeneration)
            && !task_families.contains(&ModelTaskFamily::ImageEdit)
            && !task_families.contains(&ModelTaskFamily::VideoGeneration)
            && !task_families.contains(&ModelTaskFamily::SpeechToText)
            && !task_families.contains(&ModelTaskFamily::TextToSpeech)
            && !task_families.contains(&ModelTaskFamily::Embedding)
            && !task_families.contains(&ModelTaskFamily::Rerank),
        reasoning: task_families.contains(&ModelTaskFamily::Reasoning),
        reasoning_effort: None,
    }
}

fn infer_input_modalities(
    task_families: &[ModelTaskFamily],
    explicit_input_modalities: &[ModelModality],
) -> Vec<ModelModality> {
    if !explicit_input_modalities.is_empty() {
        return explicit_input_modalities.to_vec();
    }

    let mut modalities = Vec::new();
    if !task_families.contains(&ModelTaskFamily::SpeechToText) {
        push_unique(&mut modalities, ModelModality::Text);
    }
    if task_families.contains(&ModelTaskFamily::SpeechToText) {
        push_unique(&mut modalities, ModelModality::Audio);
    }
    if task_families.contains(&ModelTaskFamily::ImageEdit)
        || task_families.contains(&ModelTaskFamily::VideoGeneration)
        || task_families.contains(&ModelTaskFamily::VisionUnderstanding)
    {
        push_unique(&mut modalities, ModelModality::Image);
    }
    if task_families.iter().any(|family| {
        matches!(
            family,
            ModelTaskFamily::Embedding
                | ModelTaskFamily::Rerank
                | ModelTaskFamily::Moderation
                | ModelTaskFamily::TextToSpeech
        )
    }) {
        push_unique(&mut modalities, ModelModality::Text);
    }

    modalities
}

fn infer_output_modalities(
    task_families: &[ModelTaskFamily],
    explicit_output_modalities: &[ModelModality],
    capabilities: Option<&ModelCapabilities>,
) -> Vec<ModelModality> {
    if !explicit_output_modalities.is_empty() {
        return explicit_output_modalities.to_vec();
    }

    let mut modalities = Vec::new();
    if task_families.iter().any(|family| {
        matches!(
            family,
            ModelTaskFamily::Chat
                | ModelTaskFamily::Reasoning
                | ModelTaskFamily::VisionUnderstanding
                | ModelTaskFamily::SpeechToText
                | ModelTaskFamily::Rerank
                | ModelTaskFamily::Moderation
        )
    }) {
        push_unique(&mut modalities, ModelModality::Text);
    }
    if task_families.contains(&ModelTaskFamily::ImageGeneration)
        || task_families.contains(&ModelTaskFamily::ImageEdit)
    {
        push_unique(&mut modalities, ModelModality::Image);
    }
    if task_families.contains(&ModelTaskFamily::VideoGeneration) {
        push_unique(&mut modalities, ModelModality::Video);
    }
    if task_families.contains(&ModelTaskFamily::TextToSpeech) {
        push_unique(&mut modalities, ModelModality::Audio);
    }
    if task_families.contains(&ModelTaskFamily::Embedding) {
        push_unique(&mut modalities, ModelModality::Embedding);
    }
    if capabilities.map(|caps| caps.json_mode).unwrap_or(false)
        && !task_families.contains(&ModelTaskFamily::SpeechToText)
    {
        push_unique(&mut modalities, ModelModality::Json);
    }

    modalities
}

fn infer_runtime_features(
    provider_id: Option<&str>,
    capabilities: Option<&ModelCapabilities>,
    task_families: &[ModelTaskFamily],
    explicit_runtime_features: &[ModelRuntimeFeature],
) -> Vec<ModelRuntimeFeature> {
    if !explicit_runtime_features.is_empty() {
        return explicit_runtime_features.to_vec();
    }

    let provider_key = provider_id.map(normalize_identifier).unwrap_or_default();
    let mut features = Vec::new();
    if capabilities.map(|caps| caps.streaming).unwrap_or(true) {
        push_unique(&mut features, ModelRuntimeFeature::Streaming);
    }
    if capabilities
        .map(|caps| caps.tools || caps.function_calling)
        .unwrap_or(false)
    {
        push_unique(&mut features, ModelRuntimeFeature::ToolCalling);
    }
    if capabilities.map(|caps| caps.json_mode).unwrap_or(false) {
        push_unique(&mut features, ModelRuntimeFeature::JsonSchema);
    }
    if capabilities.map(|caps| caps.reasoning).unwrap_or(false)
        || task_families.contains(&ModelTaskFamily::Reasoning)
    {
        push_unique(&mut features, ModelRuntimeFeature::Reasoning);
    }
    if matches!(provider_key.as_str(), "codex" | "azure-openai") {
        push_unique(&mut features, ModelRuntimeFeature::ResponsesApi);
    }
    if matches!(provider_key.as_str(), "openai" | "new-api" | "gateway") {
        push_unique(&mut features, ModelRuntimeFeature::ChatCompletionsApi);
    }
    if task_families.contains(&ModelTaskFamily::ImageGeneration)
        || task_families.contains(&ModelTaskFamily::ImageEdit)
    {
        push_unique(&mut features, ModelRuntimeFeature::ImagesApi);
    }

    features
}

fn infer_deployment_source(
    provider_id: Option<&str>,
    description: Option<&str>,
    explicit_deployment_source: Option<ModelDeploymentSource>,
) -> ModelDeploymentSource {
    if let Some(source) = explicit_deployment_source {
        return source;
    }

    let text = build_search_text(&[
        provider_id.map(ToString::to_string),
        description.map(ToString::to_string),
    ]);
    if text_contains_any(
        &text,
        &["ollama", "lmstudio", "gpustack", "ovms", "comfyui"],
    ) {
        return ModelDeploymentSource::Local;
    }
    if text_contains_any(
        &text,
        &["lime-hub", "lime hub", "oem", "partner-hub", "partner hub"],
    ) {
        return ModelDeploymentSource::OemCloud;
    }

    ModelDeploymentSource::UserCloud
}

fn infer_management_plane(
    deployment_source: &ModelDeploymentSource,
    explicit_management_plane: Option<ModelManagementPlane>,
) -> ModelManagementPlane {
    if let Some(plane) = explicit_management_plane {
        return plane;
    }

    match deployment_source {
        ModelDeploymentSource::Local => ModelManagementPlane::LocalSettings,
        ModelDeploymentSource::OemCloud => ModelManagementPlane::OemControlPlane,
        ModelDeploymentSource::UserCloud => ModelManagementPlane::LocalSettings,
    }
}

fn infer_alias_source(
    explicit_alias_source: Option<ModelAliasSource>,
    provider_model_id: Option<&str>,
    canonical_model_id: Option<&str>,
    canonical_model: Option<&CanonicalModel>,
) -> Option<ModelAliasSource> {
    if let Some(alias_source) = explicit_alias_source {
        return Some(alias_source);
    }

    if provider_model_id.is_some() && canonical_model_id.is_some() {
        if canonical_model.is_some() {
            return Some(ModelAliasSource::Official);
        }
        if provider_model_id.map(normalize_identifier)
            != canonical_model_id.map(normalize_identifier)
        {
            return Some(ModelAliasSource::Relay);
        }
    }

    None
}

#[derive(Debug, Clone)]
struct InferredModelTaxonomy {
    task_families: Vec<ModelTaskFamily>,
    input_modalities: Vec<ModelModality>,
    output_modalities: Vec<ModelModality>,
    runtime_features: Vec<ModelRuntimeFeature>,
    deployment_source: ModelDeploymentSource,
    management_plane: ModelManagementPlane,
    canonical_model_id: Option<String>,
    provider_model_id: Option<String>,
    alias_source: Option<ModelAliasSource>,
}

struct ModelTaxonomyInput<'a> {
    model_id: &'a str,
    provider_id: Option<&'a str>,
    family: Option<&'a str>,
    description: Option<&'a str>,
    capabilities: Option<&'a ModelCapabilities>,
    explicit_task_families: &'a [ModelTaskFamily],
    explicit_input_modalities: &'a [ModelModality],
    explicit_output_modalities: &'a [ModelModality],
    explicit_runtime_features: &'a [ModelRuntimeFeature],
    explicit_deployment_source: Option<ModelDeploymentSource>,
    explicit_management_plane: Option<ModelManagementPlane>,
    provider_model_id: Option<&'a str>,
    canonical_model_id: Option<&'a str>,
    explicit_alias_source: Option<ModelAliasSource>,
    canonical_model: Option<&'a CanonicalModel>,
}

fn infer_model_taxonomy(input: ModelTaxonomyInput<'_>) -> InferredModelTaxonomy {
    let canonical_task_families = input
        .canonical_model
        .map(|model| {
            model
                .task_families
                .iter()
                .filter_map(|family| parse_task_family(family))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let canonical_input_modalities = input
        .canonical_model
        .map(|model| parse_modalities(&model.input_modalities))
        .unwrap_or_default();
    let canonical_output_modalities = input
        .canonical_model
        .map(|model| parse_modalities(&model.output_modalities))
        .unwrap_or_default();
    let canonical_runtime_features = input
        .canonical_model
        .map(|model| {
            model
                .runtime_features
                .iter()
                .filter_map(|feature| parse_runtime_feature(feature))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let task_family_seed = if input.explicit_task_families.is_empty() {
        canonical_task_families.as_slice()
    } else {
        input.explicit_task_families
    };
    let input_seed = if input.explicit_input_modalities.is_empty() {
        canonical_input_modalities.as_slice()
    } else {
        input.explicit_input_modalities
    };
    let output_seed = if input.explicit_output_modalities.is_empty() {
        canonical_output_modalities.as_slice()
    } else {
        input.explicit_output_modalities
    };
    let task_families = infer_model_task_families(
        input.model_id,
        input.provider_id,
        input.family,
        input.description,
        input.capabilities,
        task_family_seed,
        input_seed,
        output_seed,
        input.provider_model_id,
        input.canonical_model_id,
    );
    let input_modalities = if !input.explicit_input_modalities.is_empty() {
        input.explicit_input_modalities.to_vec()
    } else if !canonical_input_modalities.is_empty() {
        canonical_input_modalities
    } else {
        infer_input_modalities(&task_families, input.explicit_input_modalities)
    };
    let output_modalities = if !input.explicit_output_modalities.is_empty() {
        input.explicit_output_modalities.to_vec()
    } else if !canonical_output_modalities.is_empty() {
        canonical_output_modalities
    } else {
        infer_output_modalities(
            &task_families,
            input.explicit_output_modalities,
            input.capabilities,
        )
    };
    let runtime_feature_seed = if input.explicit_runtime_features.is_empty() {
        canonical_runtime_features.as_slice()
    } else {
        input.explicit_runtime_features
    };
    let runtime_features = infer_runtime_features(
        input.provider_id,
        input.capabilities,
        &task_families,
        runtime_feature_seed,
    );
    let deployment_source = infer_deployment_source(
        input.provider_id,
        input.description,
        input.explicit_deployment_source,
    );
    let management_plane =
        infer_management_plane(&deployment_source, input.explicit_management_plane);
    let provider_model_id = input
        .provider_model_id
        .map(ToString::to_string)
        .or_else(|| Some(input.model_id.to_string()));
    let canonical_model_id = input
        .canonical_model_id
        .map(ToString::to_string)
        .or_else(|| input.canonical_model.map(|model| model.id.clone()));
    let alias_source = infer_alias_source(
        input.explicit_alias_source,
        provider_model_id.as_deref(),
        canonical_model_id.as_deref(),
        input.canonical_model,
    );

    InferredModelTaxonomy {
        task_families,
        input_modalities,
        output_modalities,
        runtime_features,
        deployment_source,
        management_plane,
        canonical_model_id,
        provider_model_id,
        alias_source,
    }
}

/// 模型注册服务
pub struct ModelRegistryService {
    /// 数据库连接
    db: DbConnection,
    /// Provider 别名配置缓存（provider_id -> ProviderAliasConfig）
    aliases_cache: Arc<RwLock<HashMap<String, ProviderAliasConfig>>>,
    /// 同步状态
    sync_state: Arc<RwLock<ModelSyncState>>,
}

impl ModelRegistryService {
    /// 创建新的模型注册服务
    pub fn new(db: DbConnection) -> Self {
        Self {
            db,
            aliases_cache: Arc::new(RwLock::new(HashMap::new())),
            sync_state: Arc::new(RwLock::new(ModelSyncState::default())),
        }
    }

    /// 初始化服务。
    ///
    /// 本地模型目录已下线；后续模型列表只由启用 Provider 的实时接口或
    /// 用户显式配置提供。
    pub async fn initialize(&self) -> Result<(), String> {
        tracing::info!("[ModelRegistry] 初始化模型注册服务");

        self.aliases_cache.write().await.clear();

        // 更新同步状态
        {
            let mut state = self.sync_state.write().await;
            state.model_count = 0;
            state.last_sync_at = Some(chrono::Utc::now().timestamp());
            state.is_syncing = false;
            state.last_error = None;
        }

        Ok(())
    }

    /// 获取同步状态
    pub async fn get_sync_state(&self) -> ModelSyncState {
        self.sync_state.read().await.clone()
    }

    /// 刷新模型注册表。
    ///
    /// 本地模型目录已下线，刷新只清空 Provider 实时模型缓存。
    pub async fn force_reload(&self) -> Result<u32, String> {
        tracing::info!("[ModelRegistry] 清空模型注册表缓存");

        self.aliases_cache.write().await.clear();
        let cleared_provider_cache = self.clear_provider_models_cache()?;
        if cleared_provider_cache > 0 {
            tracing::info!(
                "[ModelRegistry] 已清空 {} 条 Provider 实时模型缓存",
                cleared_provider_cache
            );
        }

        // 更新同步状态
        {
            let mut state = self.sync_state.write().await;
            state.model_count = 0;
            state.last_sync_at = Some(chrono::Utc::now().timestamp());
            state.is_syncing = false;
            state.last_error = None;
        }

        Ok(0)
    }

    // ========== 用户偏好相关方法 ==========

    /// 获取所有用户偏好
    pub async fn get_all_preferences(&self) -> Result<Vec<UserModelPreference>, String> {
        let conn = self.db.lock().map_err(|e| e.to_string())?;

        let mut stmt = conn
            .prepare(
                "SELECT model_id, is_favorite, is_hidden, custom_alias,
                        usage_count, last_used_at, created_at, updated_at
                 FROM user_model_preferences",
            )
            .map_err(|e| e.to_string())?;

        let prefs = stmt
            .query_map([], |row| {
                Ok(UserModelPreference {
                    model_id: row.get(0)?,
                    is_favorite: row.get::<_, i32>(1)? != 0,
                    is_hidden: row.get::<_, i32>(2)? != 0,
                    custom_alias: row.get(3)?,
                    usage_count: row.get::<_, i32>(4)? as u32,
                    last_used_at: row.get(5)?,
                    created_at: row.get(6)?,
                    updated_at: row.get(7)?,
                })
            })
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;

        Ok(prefs)
    }

    /// 切换收藏状态
    pub async fn toggle_favorite(&self, model_id: &str) -> Result<bool, String> {
        let conn = self.db.lock().map_err(|e| e.to_string())?;
        let now = chrono::Utc::now().timestamp();

        // 检查是否存在
        let exists: bool = conn
            .query_row(
                "SELECT 1 FROM user_model_preferences WHERE model_id = ?",
                params![model_id],
                |_| Ok(true),
            )
            .unwrap_or(false);

        if exists {
            // 切换状态
            conn.execute(
                "UPDATE user_model_preferences
                 SET is_favorite = NOT is_favorite, updated_at = ?
                 WHERE model_id = ?",
                params![now, model_id],
            )
            .map_err(|e| e.to_string())?;
        } else {
            // 创建新记录
            conn.execute(
                "INSERT INTO user_model_preferences
                 (model_id, is_favorite, is_hidden, usage_count, created_at, updated_at)
                 VALUES (?, 1, 0, 0, ?, ?)",
                params![model_id, now, now],
            )
            .map_err(|e| e.to_string())?;
        }

        // 返回新状态
        let new_state: bool = conn
            .query_row(
                "SELECT is_favorite FROM user_model_preferences WHERE model_id = ?",
                params![model_id],
                |row| Ok(row.get::<_, i32>(0)? != 0),
            )
            .unwrap_or(false);

        Ok(new_state)
    }

    /// 隐藏模型
    pub async fn hide_model(&self, model_id: &str) -> Result<(), String> {
        let conn = self.db.lock().map_err(|e| e.to_string())?;
        let now = chrono::Utc::now().timestamp();

        conn.execute(
            "INSERT INTO user_model_preferences
             (model_id, is_favorite, is_hidden, usage_count, created_at, updated_at)
             VALUES (?, 0, 1, 0, ?, ?)
             ON CONFLICT(model_id) DO UPDATE SET is_hidden = 1, updated_at = ?",
            params![model_id, now, now, now],
        )
        .map_err(|e| e.to_string())?;

        Ok(())
    }

    /// 记录模型使用
    pub async fn record_usage(&self, model_id: &str) -> Result<(), String> {
        let conn = self.db.lock().map_err(|e| e.to_string())?;
        let now = chrono::Utc::now().timestamp();

        conn.execute(
            "INSERT INTO user_model_preferences
             (model_id, is_favorite, is_hidden, usage_count, last_used_at, created_at, updated_at)
             VALUES (?, 0, 0, 1, ?, ?, ?)
             ON CONFLICT(model_id) DO UPDATE SET
                usage_count = usage_count + 1,
                last_used_at = ?,
                updated_at = ?",
            params![model_id, now, now, now, now, now],
        )
        .map_err(|e| e.to_string())?;

        Ok(())
    }

    // ========== Provider 别名相关方法 ==========

    /// 获取指定 Provider 的别名配置
    pub async fn get_provider_alias_config(&self, provider: &str) -> Option<ProviderAliasConfig> {
        self.aliases_cache.read().await.get(provider).cloned()
    }

    /// 检查指定 Provider 是否支持某个模型
    pub async fn provider_supports_model(&self, provider: &str, model: &str) -> bool {
        if let Some(config) = self.aliases_cache.read().await.get(provider) {
            config.supports_model(model)
        } else {
            // 如果没有别名配置，默认支持所有模型
            true
        }
    }

    /// 获取模型在指定 Provider 中的内部名称
    pub async fn get_model_internal_name(&self, provider: &str, model: &str) -> Option<String> {
        self.aliases_cache
            .read()
            .await
            .get(provider)
            .and_then(|config| config.get_internal_name(model).map(|s| s.to_string()))
    }

    /// 获取所有 Provider 别名配置
    pub async fn get_all_alias_configs(&self) -> HashMap<String, ProviderAliasConfig> {
        self.aliases_cache.read().await.clone()
    }

    // ========== 从 Provider API 获取模型 ==========

    pub fn requires_api_key_for_model_fetch(
        provider_id: &str,
        api_host: &str,
        provider_type: ApiProviderType,
    ) -> bool {
        if Self::uses_declared_models_for_model_fetch(provider_id, api_host, Some(provider_type)) {
            return false;
        }

        match provider_type {
            ApiProviderType::Ollama => false,
            ApiProviderType::Openai
            | ApiProviderType::OpenaiResponse
            | ApiProviderType::Codex
            | ApiProviderType::NewApi
            | ApiProviderType::Gateway
            | ApiProviderType::Fal => !Self::is_keyless_openai_like_provider(provider_id, api_host),
            ApiProviderType::Anthropic
            | ApiProviderType::AnthropicCompatible
            | ApiProviderType::Gemini
            | ApiProviderType::AzureOpenai
            | ApiProviderType::Vertexai
            | ApiProviderType::AwsBedrock => true,
        }
    }

    /// Runtime chat execution requires a repository credential unless the provider is
    /// explicitly known to be keyless. Model discovery may be keyless for additional
    /// local/custom providers, but that must not make execution silently unauthenticated.
    pub fn requires_api_key_for_runtime(
        provider_id: &str,
        api_host: &str,
        provider_type: ApiProviderType,
    ) -> bool {
        if matches!(provider_type, ApiProviderType::Ollama) {
            return false;
        }

        let normalized_provider_id = provider_id.trim().to_ascii_lowercase();
        if matches!(
            normalized_provider_id.as_str(),
            "ollama" | "lmstudio" | "gpustack" | "ovms"
        ) {
            return false;
        }

        if managed_model_fetch_access::is_lime_managed_api_host(api_host)
            && Self::lime_tenant_id_from_api_host(api_host).is_some()
        {
            return false;
        }

        true
    }

    fn is_fal_like_model_fetch(
        provider_id: &str,
        api_host: &str,
        provider_type: Option<ApiProviderType>,
    ) -> bool {
        if matches!(provider_type, Some(ApiProviderType::Fal)) {
            return true;
        }

        let provider = provider_id.trim().to_ascii_lowercase();
        let host = api_host.trim().to_ascii_lowercase();

        provider == "fal"
            || provider.starts_with("fal-")
            || provider.contains("fal.ai")
            || host.contains("fal.run")
            || host.contains("queue.fal.run")
    }

    fn is_xiaomi_like_model_fetch(
        provider_id: &str,
        api_host: &str,
        provider_type: Option<ApiProviderType>,
    ) -> bool {
        let provider = provider_id.trim().to_ascii_lowercase();
        let host = api_host.trim().to_ascii_lowercase();
        let provider_type = provider_type.map(|value| value.to_string());
        let provider_type = provider_type.as_deref().unwrap_or_default();

        matches!(provider.as_str(), "xiaomi" | "mimo" | "xiaomimimo")
            || matches!(provider_type, "xiaomi" | "mimo" | "xiaomimimo")
            || XIAOMI_MODEL_FETCH_HOST_KEYWORDS
                .iter()
                .any(|keyword| host.contains(keyword))
    }

    fn uses_declared_models_for_model_fetch(
        provider_id: &str,
        api_host: &str,
        provider_type: Option<ApiProviderType>,
    ) -> bool {
        Self::is_fal_like_model_fetch(provider_id, api_host, provider_type)
    }

    #[cfg(test)]
    fn provider_models_cache_key(
        provider_id: &str,
        api_host: &str,
        provider_type: Option<ApiProviderType>,
    ) -> String {
        Self::provider_models_cache_key_scoped(provider_id, api_host, provider_type, None)
    }

    fn provider_models_cache_key_scoped(
        provider_id: &str,
        api_host: &str,
        provider_type: Option<ApiProviderType>,
        credential_fingerprint: Option<&str>,
    ) -> String {
        let protocol = Self::resolve_model_fetch_protocol(provider_id, api_host, provider_type);
        let scope = format!(
            "{}\n{}\n{protocol:?}\n{}",
            provider_id.trim().to_ascii_lowercase(),
            api_host.trim().trim_end_matches('/'),
            credential_fingerprint.unwrap_or("unscoped")
        );
        let digest = Sha256::digest(scope.as_bytes());
        let mut hash = String::with_capacity(digest.len() * 2);
        for byte in digest {
            use std::fmt::Write as _;
            let _ = write!(&mut hash, "{byte:02x}");
        }
        format!("{PROVIDER_MODELS_CACHE_KEY_PREFIX}{hash}")
    }

    fn credential_cache_fingerprint(api_key: &str) -> Option<String> {
        let api_key = api_key.trim();
        if api_key.is_empty() {
            return None;
        }

        let digest = Sha256::digest(api_key.as_bytes());
        let mut fingerprint = String::with_capacity(digest.len() * 2);
        for byte in digest {
            use std::fmt::Write as _;
            let _ = write!(&mut fingerprint, "{byte:02x}");
        }
        Some(fingerprint)
    }

    /// 读取 10 天内有效的 Provider 实时模型缓存。
    ///
    /// 缓存不包含 API Key，自动获取模型前可以先读缓存，避免无 Key 或临时
    /// 网络失败时把已确认的模型列表清空。
    pub fn get_cached_provider_models(
        &self,
        provider_id: &str,
        api_host: &str,
        provider_type: Option<ApiProviderType>,
    ) -> Result<Option<FetchModelsResult>, String> {
        self.get_cached_provider_models_scoped(provider_id, api_host, provider_type, None)
    }

    fn get_cached_provider_models_scoped(
        &self,
        provider_id: &str,
        api_host: &str,
        provider_type: Option<ApiProviderType>,
        credential_fingerprint: Option<&str>,
    ) -> Result<Option<FetchModelsResult>, String> {
        let cache_key = Self::provider_models_cache_key_scoped(
            provider_id,
            api_host,
            provider_type,
            credential_fingerprint,
        );
        let now = chrono::Utc::now().timestamp();
        let mut conn = self.db.lock().map_err(|e| e.to_string())?;

        let cached_value: Option<String> = conn
            .query_row(
                "SELECT value FROM settings WHERE key = ?1",
                params![cache_key],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| format!("读取 Provider 模型缓存失败: {e}"))?;

        let Some(cached_value) = cached_value else {
            return Ok(None);
        };

        let payload: ProviderModelsCachePayload = match serde_json::from_str(&cached_value) {
            Ok(payload) => payload,
            Err(error) => {
                tracing::warn!("[ModelRegistry] Provider 模型缓存解析失败，已忽略: {error}");
                if let Err(delete_error) =
                    Self::delete_provider_models_cache_entry(&mut conn, &cache_key)
                {
                    tracing::warn!(
                        "[ModelRegistry] 删除损坏的 Provider 模型缓存失败: {delete_error}"
                    );
                }
                return Ok(None);
            }
        };

        if payload.taxonomy_version != PROVIDER_MODELS_CACHE_TAXONOMY_VERSION {
            tracing::info!(
                "[ModelRegistry] Provider 模型缓存能力分类版本已过期: provider={}, host={}, cached={}, current={}",
                provider_id,
                api_host,
                payload.taxonomy_version,
                PROVIDER_MODELS_CACHE_TAXONOMY_VERSION
            );
            if let Err(error) = Self::delete_provider_models_cache_entry(&mut conn, &cache_key) {
                tracing::warn!("[ModelRegistry] 删除过期分类的 Provider 模型缓存失败: {error}");
            }
            return Ok(None);
        }

        if payload.expires_at <= now {
            tracing::info!(
                "[ModelRegistry] Provider 模型缓存已过期: provider={}, host={}",
                provider_id,
                api_host
            );
            if let Err(error) = Self::delete_provider_models_cache_entry(&mut conn, &cache_key) {
                tracing::warn!("[ModelRegistry] 删除过期 Provider 模型缓存失败: {error}");
            }
            return Ok(None);
        }

        let mut models = payload.models;

        if Self::uses_declared_models_for_model_fetch(provider_id, api_host, provider_type) {
            let original_count = models.len();
            models.retain(|model| Self::is_likely_fal_declared_model(&model.id));
            if models.len() != original_count {
                tracing::info!(
                    "[ModelRegistry] Fal Provider 模型缓存包含非 Fal 模型，已清理: provider={}, host={}",
                    provider_id,
                    api_host
                );
                if let Err(error) = Self::delete_provider_models_cache_entry(&mut conn, &cache_key)
                {
                    tracing::warn!("[ModelRegistry] 删除无效 Fal 模型缓存失败: {error}");
                }
            }
        }

        if models.is_empty() {
            if let Err(error) = Self::delete_provider_models_cache_entry(&mut conn, &cache_key) {
                tracing::warn!("[ModelRegistry] 删除空 Provider 模型缓存失败: {error}");
            }
            return Ok(None);
        }

        tracing::info!(
            "[ModelRegistry] 命中 Provider 模型缓存: provider={}, host={}, models={}",
            provider_id,
            api_host,
            models.len()
        );

        Ok(Some(FetchModelsResult {
            models,
            source: ModelFetchSource::Api,
            error: None,
            request_url: payload.request_url,
            diagnostic_hint: Some("已使用 10 天内缓存的模型列表。".to_string()),
            error_kind: None,
            should_prompt_error: false,
            from_cache: true,
        }))
    }

    #[cfg(test)]
    fn save_provider_models_cache(
        &self,
        provider_id: &str,
        api_host: &str,
        provider_type: Option<ApiProviderType>,
        models: &[EnhancedModelMetadata],
        request_url: Option<String>,
        fetched_at: i64,
    ) -> Result<(), String> {
        self.save_provider_models_cache_scoped(
            provider_id,
            api_host,
            provider_type,
            models,
            request_url,
            fetched_at,
            None,
        )
    }

    fn save_provider_models_cache_scoped(
        &self,
        provider_id: &str,
        api_host: &str,
        provider_type: Option<ApiProviderType>,
        models: &[EnhancedModelMetadata],
        request_url: Option<String>,
        fetched_at: i64,
        credential_fingerprint: Option<&str>,
    ) -> Result<(), String> {
        let scoped_cache_key = Self::provider_models_cache_key_scoped(
            provider_id,
            api_host,
            provider_type,
            credential_fingerprint,
        );
        let mut cache_keys = vec![scoped_cache_key];
        if credential_fingerprint.is_some() {
            let last_success_cache_key =
                Self::provider_models_cache_key_scoped(provider_id, api_host, provider_type, None);
            if cache_keys[0] != last_success_cache_key {
                cache_keys.push(last_success_cache_key);
            }
        }
        let payload = if models.is_empty() {
            None
        } else {
            Some(
                serde_json::to_string(&ProviderModelsCachePayload {
                    taxonomy_version: PROVIDER_MODELS_CACHE_TAXONOMY_VERSION,
                    provider_id: provider_id.trim().to_string(),
                    api_host: api_host.trim().to_string(),
                    provider_type: provider_type.map(|provider_type| provider_type.to_string()),
                    request_url,
                    fetched_at,
                    expires_at: fetched_at + PROVIDER_MODELS_CACHE_TTL_SECONDS,
                    models: models.to_vec(),
                })
                .map_err(|e| format!("序列化 Provider 模型缓存失败: {e}"))?,
            )
        };

        let mut conn = self.db.lock().map_err(|e| e.to_string())?;
        let tx = conn
            .transaction()
            .map_err(|e| format!("开始 Provider 模型缓存事务失败: {e}"))?;
        let mut changed = false;
        for cache_key in cache_keys {
            let existing_payload: Option<String> = tx
                .query_row(
                    "SELECT value FROM settings WHERE key = ?1",
                    params![&cache_key],
                    |row| row.get(0),
                )
                .optional()
                .map_err(|e| format!("读取 Provider 模型缓存失败: {e}"))?;
            match payload.as_deref() {
                Some(payload) if existing_payload.as_deref() != Some(payload) => {
                    tx.execute(
                        "INSERT OR REPLACE INTO settings (key, value) VALUES (?1, ?2)",
                        params![&cache_key, payload],
                    )
                    .map_err(|e| format!("写入 Provider 模型缓存失败: {e}"))?;
                    changed = true;
                }
                None if existing_payload.is_some() => {
                    tx.execute("DELETE FROM settings WHERE key = ?1", params![&cache_key])
                        .map_err(|e| format!("删除 Provider 模型缓存失败: {e}"))?;
                    changed = true;
                }
                _ => {}
            }
        }
        if changed {
            RouteStateDao::advance_generation(&tx)
                .map_err(|e| format!("推进模型路由代际失败: {e}"))?;
        }
        tx.commit()
            .map_err(|e| format!("提交 Provider 模型缓存事务失败: {e}"))?;

        Ok(())
    }

    fn delete_provider_models_cache_entry(
        conn: &mut Connection,
        cache_key: &str,
    ) -> Result<bool, String> {
        let tx = conn
            .transaction()
            .map_err(|e| format!("开始 Provider 模型缓存删除事务失败: {e}"))?;
        let deleted = tx
            .execute("DELETE FROM settings WHERE key = ?1", params![cache_key])
            .map_err(|e| format!("删除 Provider 模型缓存失败: {e}"))?;
        if deleted > 0 {
            RouteStateDao::advance_generation(&tx)
                .map_err(|e| format!("推进模型路由代际失败: {e}"))?;
        }
        tx.commit()
            .map_err(|e| format!("提交 Provider 模型缓存删除事务失败: {e}"))?;
        Ok(deleted > 0)
    }

    fn clear_provider_models_cache(&self) -> Result<usize, String> {
        let mut conn = self.db.lock().map_err(|e| e.to_string())?;
        let tx = conn
            .transaction()
            .map_err(|e| format!("开始 Provider 模型缓存清理事务失败: {e}"))?;
        let deleted = tx
            .execute(
                "DELETE FROM settings WHERE key GLOB ?1",
                params![format!("{PROVIDER_MODELS_CACHE_KEY_PREFIX}*")],
            )
            .map_err(|e| format!("清空 Provider 模型缓存失败: {e}"))?;
        if deleted > 0 {
            RouteStateDao::advance_generation(&tx)
                .map_err(|e| format!("推进模型路由代际失败: {e}"))?;
        }
        tx.commit()
            .map_err(|e| format!("提交 Provider 模型缓存清理事务失败: {e}"))?;
        Ok(deleted)
    }

    /// 从 Provider API 获取模型列表
    ///
    /// 优先读取 10 天内 Provider 实时模型缓存；未命中时调用 Provider
    /// 的 `/models` 端点获取模型列表。
    ///
    /// # 参数
    /// - `provider_id`: Provider ID（如 "siliconflow", "openai"）
    /// - `api_host`: API 主机地址
    /// - `api_key`: API Key
    ///
    /// # 返回
    /// - `Ok(FetchModelsResult)`: 获取结果，包含模型列表和来源
    pub async fn fetch_models_from_api(
        &self,
        provider_id: &str,
        api_host: &str,
        api_key: &str,
    ) -> Result<FetchModelsResult, String> {
        self.fetch_models_from_api_with_hints(provider_id, api_host, api_key, None, &[])
            .await
    }

    /// 从 Provider API 获取模型列表。
    ///
    /// 本地模型目录已下线：有效缓存未命中且 API 不可用时直接返回错误来源，
    /// 不再展示 `models` 或内置厂商目录作为“接口获取”的结果。
    pub async fn fetch_models_from_api_with_hints(
        &self,
        provider_id: &str,
        api_host: &str,
        api_key: &str,
        provider_type: Option<ApiProviderType>,
        models: &[ProviderModelConfig],
    ) -> Result<FetchModelsResult, String> {
        tracing::info!(
            "[ModelRegistry] 从 API 获取模型: provider={}, host={}",
            provider_id,
            api_host
        );
        let credential_fingerprint = Self::credential_cache_fingerprint(api_key);

        match self.get_cached_provider_models_scoped(
            provider_id,
            api_host,
            provider_type,
            credential_fingerprint.as_deref(),
        ) {
            Ok(Some(cached)) => return Ok(cached),
            Ok(None) => {}
            Err(error) => {
                tracing::warn!("[ModelRegistry] 读取 Provider 模型缓存失败，继续访问 API: {error}");
            }
        }

        if Self::is_xiaomi_like_model_fetch(provider_id, api_host, provider_type) {
            let now = chrono::Utc::now().timestamp();
            let models = self.build_declared_models(provider_id, models, now);

            if models.is_empty() {
                return Ok(FetchModelsResult {
                    models,
                    source: ModelFetchSource::Error,
                    error: Some("Mimo / xiaomimimo Anthropic 兼容入口不提供标准 /models 枚举。".to_string()),
                    request_url: None,
                    diagnostic_hint: Some(
                        "请在 Provider 模型列表中填写当前服务商实际可用的模型 ID；Lime 不再内置固定模型名作为兜底。"
                            .to_string(),
                    ),
                    error_kind: Some(ModelFetchErrorKind::Other),
                    should_prompt_error: false,
                    from_cache: false,
                });
            }

            if let Err(error) = self.save_provider_models_cache_scoped(
                provider_id,
                api_host,
                provider_type,
                &models,
                None,
                now,
                credential_fingerprint.as_deref(),
            ) {
                tracing::warn!("[ModelRegistry] 写入 Mimo Provider 模型缓存失败: {error}");
            }

            return Ok(FetchModelsResult {
                models,
                source: ModelFetchSource::Api,
                error: None,
                request_url: None,
                diagnostic_hint: Some(
                    "Mimo / xiaomimimo Anthropic 兼容入口不提供标准 /models 枚举；已使用 Provider 中声明的模型并写入 10 天缓存。"
                        .to_string(),
                ),
                error_kind: None,
                should_prompt_error: false,
                from_cache: false,
            });
        }

        if Self::uses_declared_models_for_model_fetch(provider_id, api_host, provider_type) {
            let now = chrono::Utc::now().timestamp();
            let models = self.build_fal_declared_models(provider_id, models, now);

            if !models.is_empty() {
                if let Err(error) = self.save_provider_models_cache_scoped(
                    provider_id,
                    api_host,
                    provider_type,
                    &models,
                    None,
                    now,
                    credential_fingerprint.as_deref(),
                ) {
                    tracing::warn!("[ModelRegistry] 写入声明模型缓存失败: {error}");
                }

                return Ok(FetchModelsResult {
                    models,
                    source: ModelFetchSource::Api,
                    error: None,
                    request_url: None,
                    diagnostic_hint: Some(
                        "Fal 不提供标准 /models 枚举；已确认 Provider 中声明的 Fal 图片模型，并写入 10 天缓存。"
                            .to_string(),
                    ),
                    error_kind: None,
                    should_prompt_error: false,
                    from_cache: false,
                });
            }

            return Ok(FetchModelsResult {
                models: Vec::new(),
                source: ModelFetchSource::Error,
                error: Some("Fal 不提供标准 /models 枚举。".to_string()),
                request_url: None,
                diagnostic_hint: Some(
                    "当前模型优先级没有可用 Fal 图片模型；请手动添加 fal-ai/nano-banana-pro、fal-ai/flux-pro 或其他 fal-ai/... 模型 ID。"
                        .to_string(),
                ),
                error_kind: Some(ModelFetchErrorKind::Other),
                should_prompt_error: false,
                from_cache: false,
            });
        }

        let fetch_protocol =
            Self::resolve_model_fetch_protocol(provider_id, api_host, provider_type);
        if fetch_protocol == ModelFetchProtocol::ResponsesCompatible {
            let now = chrono::Utc::now().timestamp();
            let models = self.build_responses_compatible_declared_models(provider_id, models, now);

            if !models.is_empty() {
                if let Err(error) = self.save_provider_models_cache_scoped(
                    provider_id,
                    api_host,
                    provider_type,
                    &models,
                    None,
                    now,
                    credential_fingerprint.as_deref(),
                ) {
                    tracing::warn!(
                        "[ModelRegistry] 写入 Responses Provider 声明模型缓存失败: {error}"
                    );
                }

                return Ok(FetchModelsResult {
                    models,
                    source: ModelFetchSource::Api,
                    error: None,
                    request_url: None,
                    diagnostic_hint: Some(
                        "当前 Responses 图片入口不提供标准 /models 枚举；已使用 Provider 中声明的图片模型作为可用模型，并写入 10 天缓存。"
                            .to_string(),
                    ),
                    error_kind: None,
                    should_prompt_error: false,
                    from_cache: false,
                });
            }

            return Ok(FetchModelsResult {
                models: Vec::new(),
                source: ModelFetchSource::Error,
                error: Some("当前 Responses 兼容入口未提供标准 /models 接口。".to_string()),
                request_url: None,
                diagnostic_hint: Some(
                    "当前 Base URL 走 `/responses` 主链，Lime 不再探测 `/v1/models`；请先在 Provider 中填写 gpt-images-2 或其他图片模型。"
                        .to_string(),
                ),
                error_kind: Some(ModelFetchErrorKind::NotFound),
                should_prompt_error: false,
                from_cache: false,
            });
        }

        let api_url = Self::build_diagnostic_models_api_url(provider_id, api_host, provider_type);
        if let Some(url) = api_url.as_ref() {
            tracing::info!("[ModelRegistry] API URL: {}", url);
        }
        let diagnostic_hint = api_url
            .as_ref()
            .and_then(|url| Self::build_models_api_hint(provider_id, api_host, url));

        // 尝试从 API 获取
        match self
            .call_models_api(provider_id, api_host, api_key, provider_type)
            .await
        {
            Ok((api_models, request_url)) => {
                tracing::info!("[ModelRegistry] 从 API 获取到 {} 个模型", api_models.len());

                // 转换为内部格式
                let now = chrono::Utc::now().timestamp();
                let models: Vec<EnhancedModelMetadata> = api_models
                    .into_iter()
                    .map(|m| self.convert_api_model_for_endpoint(m, provider_id, api_host, now))
                    .collect();

                if let Err(error) = self.save_provider_models_cache_scoped(
                    provider_id,
                    api_host,
                    provider_type,
                    &models,
                    Some(request_url.clone()),
                    now,
                    credential_fingerprint.as_deref(),
                ) {
                    tracing::warn!("[ModelRegistry] 写入 Provider 模型缓存失败: {error}");
                }

                Ok(FetchModelsResult {
                    models,
                    source: ModelFetchSource::Api,
                    error: None,
                    request_url: Some(request_url),
                    diagnostic_hint: None,
                    error_kind: None,
                    should_prompt_error: false,
                    from_cache: false,
                })
            }
            Err(api_error) => {
                tracing::warn!(
                    "[ModelRegistry] API 获取失败，本地模型兜底已下线: {}",
                    api_error.message
                );

                Ok(FetchModelsResult {
                    models: Vec::new(),
                    source: ModelFetchSource::Error,
                    error: Some(format!(
                        "API 获取失败: {}。本地模型兜底已下线，请检查 API Host / 密钥，或手动添加模型 ID。",
                        api_error.message
                    )),
                    request_url: api_url,
                    diagnostic_hint,
                    error_kind: Some(api_error.kind.clone()),
                    should_prompt_error: Self::should_prompt_model_fetch_error(&api_error.kind),
                    from_cache: false,
                })
            }
        }
    }

    fn is_keyless_openai_like_provider(provider_id: &str, api_host: &str) -> bool {
        let normalized_provider_id = provider_id.trim().to_lowercase();
        if matches!(
            normalized_provider_id.as_str(),
            "ollama" | "lmstudio" | "gpustack" | "ovms"
        ) {
            return true;
        }

        if managed_model_fetch_access::is_lime_managed_api_host(api_host)
            && Self::lime_tenant_id_from_api_host(api_host).is_some()
        {
            return true;
        }

        let normalized_host = api_host.trim().to_lowercase();
        matches!(
            normalized_host.as_str(),
            host if host.contains("://localhost")
                || host.contains("://127.0.0.1")
                || host.contains("://0.0.0.0")
                || host.contains("://host.docker.internal")
        )
    }

    fn resolve_model_fetch_protocol(
        provider_id: &str,
        api_host: &str,
        provider_type: Option<ApiProviderType>,
    ) -> ModelFetchProtocol {
        if let Some(provider_type) = provider_type {
            return match provider_type {
                ApiProviderType::OpenaiResponse | ApiProviderType::Codex => {
                    ModelFetchProtocol::ResponsesCompatible
                }
                ApiProviderType::Openai
                | ApiProviderType::NewApi
                | ApiProviderType::Gateway
                | ApiProviderType::Fal => {
                    if is_openai_responses_compatible_host(api_host) {
                        ModelFetchProtocol::ResponsesCompatible
                    } else {
                        ModelFetchProtocol::OpenAiCompatible
                    }
                }
                ApiProviderType::Anthropic | ApiProviderType::AnthropicCompatible => {
                    ModelFetchProtocol::Anthropic
                }
                ApiProviderType::Gemini => ModelFetchProtocol::Gemini,
                ApiProviderType::Ollama => ModelFetchProtocol::Ollama,
                ApiProviderType::AzureOpenai
                | ApiProviderType::Vertexai
                | ApiProviderType::AwsBedrock => ModelFetchProtocol::Unsupported,
            };
        }

        let normalized_provider = provider_id.trim().to_lowercase();
        let normalized_host = api_host.trim().to_lowercase();

        if normalized_provider == "ollama"
            || normalized_host.contains("ollama")
            || normalized_host.contains("://localhost:11434")
            || normalized_host.contains("://127.0.0.1:11434")
        {
            return ModelFetchProtocol::Ollama;
        }

        if normalized_provider.contains("gemini")
            || normalized_provider == "google"
            || normalized_host.contains("generativelanguage.googleapis.com")
        {
            return ModelFetchProtocol::Gemini;
        }

        if normalized_provider.contains("anthropic") || normalized_host.contains("anthropic.com") {
            return ModelFetchProtocol::Anthropic;
        }

        if is_openai_responses_compatible_host(api_host) {
            return ModelFetchProtocol::ResponsesCompatible;
        }

        ModelFetchProtocol::OpenAiCompatible
    }

    /// 构建模型枚举 API URL
    fn build_models_api_url(api_host: &str) -> String {
        let normalized_host = normalize_openai_model_discovery_host(api_host);
        let host = normalized_host.trim_end_matches('/');

        if host.ends_with("/models") {
            return host.to_string();
        }

        // 检查是否已经包含 /v1 路径
        if host.ends_with("/v1") || host.ends_with("/v1/") {
            format!("{}/models", host.trim_end_matches('/'))
        } else if host.contains("/v1/") {
            // 如果路径中间有 /v1/，直接追加 models
            format!("{}models", host.trim_end_matches('/').to_string() + "/")
        } else if Self::has_versioned_api_suffix(host) {
            format!("{host}/models")
        } else {
            format!("{host}/v1/models")
        }
    }

    fn parse_api_host_url(api_host: &str) -> Option<reqwest::Url> {
        let trimmed = api_host.trim();
        if trimmed.is_empty() {
            return None;
        }

        reqwest::Url::parse(trimmed)
            .or_else(|_| reqwest::Url::parse(&format!("https://{trimmed}")))
            .ok()
    }

    fn canonical_provider_id_for_api_host(api_host: &str) -> Option<&'static str> {
        let url = Self::parse_api_host_url(api_host)?;
        let is_official_agnes = url.scheme() == "https"
            && url
                .host_str()
                .is_some_and(|host| host.eq_ignore_ascii_case("apihub.agnes-ai.com"))
            && url.port_or_known_default() == Some(443);
        is_official_agnes.then_some("agnes")
    }

    fn api_host_without_query_fragment(api_host: &str) -> String {
        let trimmed = api_host.trim().trim_end_matches('/');
        if trimmed.is_empty() {
            return String::new();
        }

        let had_scheme = trimmed.starts_with("http://") || trimmed.starts_with("https://");
        if let Some(mut url) = Self::parse_api_host_url(trimmed) {
            url.set_query(None);
            url.set_fragment(None);
            let normalized = url.to_string().trim_end_matches('/').to_string();
            return if had_scheme {
                normalized
            } else {
                normalized
                    .trim_start_matches("https://")
                    .trim_end_matches('/')
                    .to_string()
            };
        }

        trimmed
            .split(['?', '#'])
            .next()
            .unwrap_or(trimmed)
            .trim_end_matches('/')
            .to_string()
    }

    fn normalize_lime_tenant_id(value: &str) -> Option<String> {
        let tenant_id = value.trim();
        if tenant_id.is_empty() {
            return None;
        }

        tenant_id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_'))
            .then(|| tenant_id.to_string())
    }

    fn parse_lime_tenant_id_from_pairs(value: &str) -> Option<String> {
        form_urlencoded::parse(value.as_bytes()).find_map(|(key, value)| {
            (key == LIME_TENANT_PARAM)
                .then(|| Self::normalize_lime_tenant_id(&value))
                .flatten()
        })
    }

    fn lime_tenant_id_from_api_host(api_host: &str) -> Option<String> {
        let url = Self::parse_api_host_url(api_host)?;

        url.query()
            .and_then(Self::parse_lime_tenant_id_from_pairs)
            .or_else(|| {
                url.fragment()
                    .and_then(Self::parse_lime_tenant_id_from_pairs)
            })
    }

    fn build_gemini_models_api_url(api_host: &str) -> String {
        let host = api_host.trim_end_matches('/');

        if host.ends_with("/models") {
            return host.to_string();
        }

        if host.ends_with("/v1beta") || host.ends_with("/v1") {
            return format!("{host}/models");
        }

        if host.contains("/v1beta/") || host.contains("/v1/") {
            return format!("{}/models", host.trim_end_matches('/'));
        }

        format!("{host}/v1beta/models")
    }

    fn build_ollama_models_api_url(api_host: &str) -> String {
        let normalized_host = Self::normalize_ollama_loopback_host(api_host);
        let host = normalized_host.trim_end_matches('/');

        if host.ends_with("/api/tags") {
            return host.to_string();
        }

        if host.ends_with("/api") {
            return format!("{host}/tags");
        }

        format!("{host}/api/tags")
    }

    fn normalize_ollama_loopback_host(api_host: &str) -> String {
        let trimmed = api_host.trim();
        if trimmed.is_empty() {
            return String::new();
        }

        let had_scheme = trimmed.starts_with("http://") || trimmed.starts_with("https://");
        let parse_target = if had_scheme {
            trimmed.to_string()
        } else {
            format!("http://{trimmed}")
        };

        let Ok(mut url) = reqwest::Url::parse(&parse_target) else {
            return trimmed.to_string();
        };

        if matches!(url.host_str(), Some("localhost")) && url.set_host(Some("127.0.0.1")).is_err() {
            return trimmed.to_string();
        }

        let normalized = url.to_string();
        if had_scheme {
            normalized.trim_end_matches('/').to_string()
        } else {
            normalized
                .trim_start_matches("http://")
                .trim_end_matches('/')
                .to_string()
        }
    }

    fn build_diagnostic_models_api_url(
        provider_id: &str,
        api_host: &str,
        provider_type: Option<ApiProviderType>,
    ) -> Option<String> {
        let host = api_host.trim();
        if host.is_empty() {
            return None;
        }

        let url = match Self::resolve_model_fetch_protocol(provider_id, api_host, provider_type) {
            ModelFetchProtocol::Gemini => Self::build_gemini_models_api_url(host),
            ModelFetchProtocol::Ollama => Self::build_ollama_models_api_url(host),
            ModelFetchProtocol::ResponsesCompatible => return None,
            ModelFetchProtocol::Anthropic | ModelFetchProtocol::OpenAiCompatible => {
                Self::build_models_api_url(host)
            }
            ModelFetchProtocol::Unsupported => host.trim_end_matches('/').to_string(),
        };

        Some(url)
    }

    fn has_versioned_api_suffix(api_host: &str) -> bool {
        let path = api_host
            .split_once("://")
            .map(|(_, rest)| rest)
            .unwrap_or(api_host)
            .split_once('/')
            .map(|(_, path)| path)
            .unwrap_or("");

        let segments: Vec<&str> = path
            .split('/')
            .filter(|segment| !segment.is_empty())
            .collect();
        if segments.is_empty() {
            return false;
        }

        let version = segments[segments.len() - 1];
        version.starts_with('v')
            && version
                .strip_prefix('v')
                .map(|suffix| !suffix.is_empty() && suffix.chars().all(|ch| ch.is_ascii_digit()))
                .unwrap_or(false)
    }

    fn build_models_api_hint(provider_id: &str, api_host: &str, api_url: &str) -> Option<String> {
        let normalized_host = normalize_openai_model_discovery_host(api_host);
        let original_host = Self::api_host_without_query_fragment(api_host);
        if !original_host.is_empty()
            && normalized_host.trim_end_matches('/') != original_host.trim_end_matches('/')
        {
            return Some(format!(
                "当前 API Host 看起来是具体接口地址而不是基础地址。Lime 已自动回退到 `{api_url}` 尝试模型枚举；如果上游本身不提供 `/models`，请改填基础 API Host，或直接在 Provider 中填写自定义模型。"
            ));
        }

        let host = api_host.to_lowercase();
        let provider = provider_id.to_lowercase();

        if provider.contains("doubao")
            || provider.contains("volc")
            || host.contains("volces.com")
            || host.contains("volcengine")
        {
            return Some(format!(
                "豆包 / 火山方舟通常应使用 Base URL `https://ark.cn-beijing.volces.com/api/v3`。当前模型列表请求为 `{api_url}`，如果出现 404，请优先检查 Base URL 是否配置为该地址。"
            ));
        }

        if provider.contains("zhipu") || host.contains("bigmodel.cn/api/paas") {
            return Some(format!(
                "智谱 GLM 官方 OpenAI 兼容 Base URL 通常应使用 `https://open.bigmodel.cn/api/paas/v4`。当前模型列表请求为 `{api_url}`，如果出现 404，请优先检查 Base URL 是否保留 `/api/paas/v4`，不要再额外改成 `/v1` 风格地址。"
            ));
        }

        None
    }

    fn should_prompt_model_fetch_error(kind: &ModelFetchErrorKind) -> bool {
        matches!(
            kind,
            ModelFetchErrorKind::NotFound
                | ModelFetchErrorKind::Unauthorized
                | ModelFetchErrorKind::Forbidden
        )
    }

    fn prepare_model_fetch_request(
        provider_id: &str,
        api_host: &str,
        api_key: &str,
        provider_type: Option<ApiProviderType>,
    ) -> Result<PreparedModelFetchRequest, ModelsApiError> {
        let protocol = Self::resolve_model_fetch_protocol(provider_id, api_host, provider_type);
        let normalized_host = api_host.trim();

        if normalized_host.is_empty() {
            return Err(ModelsApiError::new(
                ModelFetchErrorKind::Other,
                "Provider 没有配置 API Host".to_string(),
            ));
        }

        if matches!(
            protocol,
            ModelFetchProtocol::Unsupported | ModelFetchProtocol::ResponsesCompatible
        ) {
            let message = if protocol == ModelFetchProtocol::ResponsesCompatible {
                "当前 Responses 兼容入口未提供标准 /models 接口".to_string()
            } else {
                "当前协议暂不支持自动获取最新模型".to_string()
            };
            return Err(ModelsApiError::new(ModelFetchErrorKind::Other, message));
        }

        let request_type = provider_type.unwrap_or(match protocol {
            ModelFetchProtocol::Anthropic => ApiProviderType::Anthropic,
            ModelFetchProtocol::Gemini => ApiProviderType::Gemini,
            ModelFetchProtocol::Ollama => ApiProviderType::Ollama,
            ModelFetchProtocol::OpenAiCompatible
            | ModelFetchProtocol::ResponsesCompatible
            | ModelFetchProtocol::Unsupported => ApiProviderType::Openai,
        });

        let url = match protocol {
            ModelFetchProtocol::OpenAiCompatible | ModelFetchProtocol::Anthropic => {
                Self::build_models_api_url(normalized_host)
            }
            ModelFetchProtocol::Gemini => Self::build_gemini_models_api_url(normalized_host),
            ModelFetchProtocol::Ollama => Self::build_ollama_models_api_url(normalized_host),
            ModelFetchProtocol::ResponsesCompatible => unreachable!(),
            ModelFetchProtocol::Unsupported => unreachable!(),
        };

        let mut headers = vec![("Content-Type".to_string(), "application/json".to_string())];
        let runtime_spec = infer_managed_runtime_spec(request_type, normalized_host);
        if !api_key.trim().is_empty() {
            let trimmed_api_key = api_key.trim();
            let auth_value = runtime_spec
                .auth_prefix
                .map(|prefix| format!("{prefix} {trimmed_api_key}"))
                .unwrap_or_else(|| trimmed_api_key.to_string());
            headers.push((runtime_spec.auth_header.to_string(), auth_value));

            if matches!(
                request_type,
                ApiProviderType::Anthropic | ApiProviderType::AnthropicCompatible
            ) && runtime_spec
                .auth_header
                .eq_ignore_ascii_case("Authorization")
            {
                headers.push(("x-api-key".to_string(), trimmed_api_key.to_string()));
            }
        }

        for (name, value) in runtime_spec.extra_headers {
            headers.push(((*name).to_string(), (*value).to_string()));
        }
        if let Some(tenant_id) = Self::lime_tenant_id_from_api_host(normalized_host) {
            headers.push((LIME_TENANT_HEADER.to_string(), tenant_id));
        }

        Ok(PreparedModelFetchRequest {
            protocol,
            url,
            headers,
        })
    }

    async fn call_models_api(
        &self,
        provider_id: &str,
        api_host: &str,
        api_key: &str,
        provider_type: Option<ApiProviderType>,
    ) -> Result<(Vec<ApiModelResponse>, String), ModelsApiError> {
        let request =
            Self::prepare_model_fetch_request(provider_id, api_host, api_key, provider_type)?;

        let mut client_builder =
            reqwest::Client::builder().timeout(std::time::Duration::from_secs(30));
        if Self::should_bypass_proxy_for_models_api_url(&request.url) {
            tracing::info!("[ModelRegistry] 本地模型地址绕过系统代理: {}", request.url);
            client_builder = client_builder.no_proxy();
        }

        let client = client_builder.build().map_err(|e| {
            ModelsApiError::new(
                ModelFetchErrorKind::Other,
                format!("创建 HTTP 客户端失败: {e}"),
            )
        })?;

        let models = match request.protocol {
            ModelFetchProtocol::OpenAiCompatible => {
                let body = Self::send_models_api_request(
                    &client,
                    &request.url,
                    &request.headers,
                    request.protocol,
                )
                .await?;
                Self::parse_openai_models_response(&body)?
            }
            ModelFetchProtocol::Anthropic => {
                Self::call_anthropic_models_api(&client, &request.url, &request.headers).await?
            }
            ModelFetchProtocol::Gemini => {
                Self::call_gemini_models_api(&client, &request.url, &request.headers).await?
            }
            ModelFetchProtocol::Ollama => {
                let body = Self::send_models_api_request(
                    &client,
                    &request.url,
                    &request.headers,
                    request.protocol,
                )
                .await?;
                Self::parse_ollama_models_response(&body)?
            }
            ModelFetchProtocol::ResponsesCompatible => unreachable!(),
            ModelFetchProtocol::Unsupported => unreachable!(),
        };

        Ok((models, request.url))
    }

    fn summarize_http_error_body(body: &str) -> Option<String> {
        let normalized = body.split_whitespace().collect::<Vec<_>>().join(" ");
        if normalized.is_empty() {
            return None;
        }

        if Self::looks_like_html_error_body(&normalized) {
            return Some("上游返回了 HTML 错误页".to_string());
        }

        const MAX_BODY_CHARS: usize = 240;
        let mut summarized = String::new();
        for (index, ch) in normalized.chars().enumerate() {
            if index >= MAX_BODY_CHARS {
                summarized.push_str("...");
                break;
            }
            summarized.push(ch);
        }

        Some(summarized)
    }

    fn looks_like_html_error_body(body: &str) -> bool {
        let preview = body.trim_start().chars().take(256).collect::<String>();
        if preview.is_empty() {
            return false;
        }

        let preview = preview.to_ascii_lowercase();
        preview.starts_with("<!doctype html")
            || preview.starts_with("<html")
            || preview.contains("<head>")
            || preview.contains("<body>")
    }

    fn format_models_api_not_found_message(protocol: ModelFetchProtocol, body: &str) -> String {
        match protocol {
            ModelFetchProtocol::Anthropic => "当前 Anthropic 兼容入口未提供标准 /models 接口；若消息测试可用，请直接使用已配置的自定义模型，或改用厂商文档提供的模型枚举入口。"
                .to_string(),
            ModelFetchProtocol::ResponsesCompatible => {
                "当前 Responses 兼容入口未提供标准 /models 接口；请直接使用已配置的自定义模型。"
                    .to_string()
            }
            _ => {
                let base_message = match Self::summarize_http_error_body(body) {
                    Some(summary) => format!("API 返回错误 404 Not Found: {summary}。"),
                    None => "API 返回错误 404 Not Found。".to_string(),
                };

                format!(
                    "{base_message}这通常表示 Base URL 路径不兼容，请检查 Provider Base URL 是否已经包含版本路径，或是否应直接使用 /models 端点。"
                )
            }
        }
    }

    async fn send_models_api_request(
        client: &reqwest::Client,
        url: &str,
        headers: &[(String, String)],
        protocol: ModelFetchProtocol,
    ) -> Result<String, ModelsApiError> {
        let mut request_builder = client.get(url);
        for (name, value) in headers {
            request_builder = request_builder.header(name, value);
        }

        let response = request_builder.send().await.map_err(|e| {
            ModelsApiError::new(ModelFetchErrorKind::Network, format!("请求失败: {e}"))
        })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "无法读取响应体".to_string());
            if status == reqwest::StatusCode::NOT_FOUND {
                return Err(ModelsApiError::new(
                    ModelFetchErrorKind::NotFound,
                    Self::format_models_api_not_found_message(protocol, &body),
                ));
            }
            let kind = if status == reqwest::StatusCode::UNAUTHORIZED {
                ModelFetchErrorKind::Unauthorized
            } else if status == reqwest::StatusCode::FORBIDDEN {
                ModelFetchErrorKind::Forbidden
            } else {
                ModelFetchErrorKind::Other
            };
            let message = match Self::summarize_http_error_body(&body) {
                Some(summary) => format!("API 返回错误 {status}: {summary}"),
                None => format!("API 返回错误 {status}"),
            };
            return Err(ModelsApiError::new(kind, message));
        }

        response.text().await.map_err(|e| {
            ModelsApiError::new(
                ModelFetchErrorKind::InvalidResponse,
                format!("读取响应失败: {e}"),
            )
        })
    }

    fn parse_openai_models_response(body: &str) -> Result<Vec<ApiModelResponse>, ModelsApiError> {
        let api_response: ApiModelsResponse = serde_json::from_str(body).map_err(|e| {
            ModelsApiError::new(
                ModelFetchErrorKind::InvalidResponse,
                format!("解析响应失败: {e}"),
            )
        })?;

        Ok(api_response.data)
    }

    async fn call_anthropic_models_api(
        client: &reqwest::Client,
        base_url: &str,
        headers: &[(String, String)],
    ) -> Result<Vec<ApiModelResponse>, ModelsApiError> {
        let mut models = Vec::new();
        let mut after_id: Option<String> = None;

        loop {
            let mut request_url = reqwest::Url::parse(base_url).map_err(|e| {
                ModelsApiError::new(
                    ModelFetchErrorKind::Other,
                    format!("无效的 Anthropic 模型地址: {e}"),
                )
            })?;
            request_url.query_pairs_mut().append_pair("limit", "1000");
            if let Some(after) = after_id.as_deref() {
                request_url.query_pairs_mut().append_pair("after_id", after);
            }

            let body = Self::send_models_api_request(
                client,
                request_url.as_ref(),
                headers,
                ModelFetchProtocol::Anthropic,
            )
            .await?;
            let response = Self::parse_anthropic_models_response(&body)?;
            models.extend(response.models);

            if !response.has_more {
                break;
            }

            let Some(next_after_id) = response.last_id else {
                break;
            };
            if next_after_id.trim().is_empty() {
                break;
            }
            after_id = Some(next_after_id);
        }

        Ok(models)
    }

    fn parse_anthropic_models_response(
        body: &str,
    ) -> Result<AnthropicModelsResponse, ModelsApiError> {
        let response: RawAnthropicModelsResponse = serde_json::from_str(body).map_err(|e| {
            ModelsApiError::new(
                ModelFetchErrorKind::InvalidResponse,
                format!("解析 Anthropic 响应失败: {e}"),
            )
        })?;

        let models = response
            .data
            .into_iter()
            .map(|model| ApiModelResponse {
                id: model.id.clone(),
                display_name: model.display_name,
                provider_name: None,
                family: None,
                context_length: None,
                task_families: None,
                input_modalities: None,
                output_modalities: None,
                modalities: None,
                runtime_features: None,
                vision_supported: None,
                capabilities: None,
                supported_parameters: None,
                reasoning: None,
                reasoning_effort: None,
                reasoning_effort_levels: None,
                reasoning_efforts: None,
                supported_reasoning_efforts: None,
                visibility: None,
                service_tiers: None,
                default_service_tier: None,
            })
            .collect();

        Ok(AnthropicModelsResponse {
            models,
            has_more: response.has_more,
            last_id: response.last_id,
        })
    }

    async fn call_gemini_models_api(
        client: &reqwest::Client,
        base_url: &str,
        headers: &[(String, String)],
    ) -> Result<Vec<ApiModelResponse>, ModelsApiError> {
        let mut models = Vec::new();
        let mut next_page_token: Option<String> = None;

        loop {
            let mut request_url = reqwest::Url::parse(base_url).map_err(|e| {
                ModelsApiError::new(
                    ModelFetchErrorKind::Other,
                    format!("无效的 Gemini 模型地址: {e}"),
                )
            })?;
            request_url
                .query_pairs_mut()
                .append_pair("pageSize", "1000");
            if let Some(page_token) = next_page_token.as_deref() {
                request_url
                    .query_pairs_mut()
                    .append_pair("pageToken", page_token);
            }

            let body = Self::send_models_api_request(
                client,
                request_url.as_ref(),
                headers,
                ModelFetchProtocol::Gemini,
            )
            .await?;
            let response = Self::parse_gemini_models_response(&body)?;
            models.extend(response.models);

            let Some(page_token) = response.next_page_token else {
                break;
            };
            if page_token.trim().is_empty() {
                break;
            }
            next_page_token = Some(page_token);
        }

        Ok(models)
    }

    fn parse_gemini_models_response(body: &str) -> Result<GeminiModelsResponse, ModelsApiError> {
        let response: RawGeminiModelsResponse = serde_json::from_str(body).map_err(|e| {
            ModelsApiError::new(
                ModelFetchErrorKind::InvalidResponse,
                format!("解析 Gemini 响应失败: {e}"),
            )
        })?;

        let models = response
            .models
            .into_iter()
            .filter(|model| {
                model
                    .supported_generation_methods
                    .as_ref()
                    .is_none_or(|methods| {
                        methods
                            .iter()
                            .any(|method| method.eq_ignore_ascii_case("generateContent"))
                    })
            })
            .map(|model| ApiModelResponse {
                id: model.name.trim_start_matches("models/").to_string(),
                display_name: model.display_name,
                provider_name: None,
                family: None,
                context_length: model.input_token_limit,
                task_families: None,
                input_modalities: None,
                output_modalities: None,
                modalities: None,
                runtime_features: None,
                vision_supported: None,
                capabilities: None,
                supported_parameters: None,
                reasoning: None,
                reasoning_effort: None,
                reasoning_effort_levels: None,
                reasoning_efforts: None,
                supported_reasoning_efforts: None,
                visibility: None,
                service_tiers: None,
                default_service_tier: None,
            })
            .collect();

        Ok(GeminiModelsResponse {
            models,
            next_page_token: response.next_page_token,
        })
    }

    fn parse_ollama_models_response(body: &str) -> Result<Vec<ApiModelResponse>, ModelsApiError> {
        let response: OllamaModelsResponse = serde_json::from_str(body).map_err(|e| {
            ModelsApiError::new(
                ModelFetchErrorKind::InvalidResponse,
                format!("解析 Ollama 响应失败: {e}"),
            )
        })?;

        Ok(response
            .models
            .into_iter()
            .map(|model| ApiModelResponse {
                id: model.name.clone(),
                display_name: Some(model.name),
                provider_name: None,
                family: model.details.and_then(|details| details.family),
                context_length: None,
                task_families: None,
                input_modalities: None,
                output_modalities: None,
                modalities: None,
                runtime_features: None,
                vision_supported: None,
                capabilities: None,
                supported_parameters: None,
                reasoning: None,
                reasoning_effort: None,
                reasoning_effort_levels: None,
                reasoning_efforts: None,
                supported_reasoning_efforts: None,
                visibility: None,
                service_tiers: None,
                default_service_tier: None,
            })
            .collect())
    }

    /// 转换 API 模型格式为内部格式
    fn convert_api_model(
        &self,
        model: ApiModelResponse,
        provider_id: &str,
        now: i64,
    ) -> EnhancedModelMetadata {
        self.convert_api_model_with_canonical_provider(model, provider_id, provider_id, now)
    }

    fn convert_api_model_for_endpoint(
        &self,
        model: ApiModelResponse,
        provider_id: &str,
        api_host: &str,
        now: i64,
    ) -> EnhancedModelMetadata {
        let canonical_provider_id =
            Self::canonical_provider_id_for_api_host(api_host).unwrap_or(provider_id);
        self.convert_api_model_with_canonical_provider(
            model,
            provider_id,
            canonical_provider_id,
            now,
        )
    }

    fn convert_api_model_with_canonical_provider(
        &self,
        model: ApiModelResponse,
        provider_id: &str,
        canonical_provider_id: &str,
        now: i64,
    ) -> EnhancedModelMetadata {
        let display_name = model.display_name.clone().unwrap_or_else(|| {
            model
                .id
                .split('/')
                .next_back()
                .unwrap_or(&model.id)
                .to_string()
        });
        let canonical_model = maybe_get_canonical_model(canonical_provider_id, &model.id);
        let capability_provenance = if canonical_model.is_some() {
            ModelCapabilityProvenance::Canonical
        } else if api_model_has_explicit_capability_snapshot(&model) {
            ModelCapabilityProvenance::ProviderExplicit
        } else {
            ModelCapabilityProvenance::InferredHint
        };
        let mut api_task_families = parse_task_families_value(model.task_families.as_ref());
        let mut api_input_modalities = parse_modalities_value(model.input_modalities.as_ref());
        let mut api_output_modalities = parse_modalities_value(model.output_modalities.as_ref());
        if let Some(modalities) = model.modalities.as_ref() {
            merge_modalities(
                &mut api_input_modalities,
                parse_modalities_value(modalities.input.as_ref()),
            );
            merge_modalities(
                &mut api_output_modalities,
                parse_modalities_value(modalities.output.as_ref()),
            );
        }
        let mut api_runtime_features =
            parse_runtime_features_value(model.runtime_features.as_ref());
        if model.vision_supported.unwrap_or(false) {
            push_unique(&mut api_task_families, ModelTaskFamily::VisionUnderstanding);
            push_unique(&mut api_input_modalities, ModelModality::Image);
        }
        merge_api_capability_signals(
            model.capabilities.as_ref(),
            &mut api_task_families,
            &mut api_input_modalities,
            &mut api_output_modalities,
            &mut api_runtime_features,
        );
        let api_reasoning_effort_support = resolve_api_reasoning_effort_support(&model);
        let visibility = model
            .visibility
            .as_deref()
            .and_then(|value| value.parse::<ModelVisibility>().ok())
            .unwrap_or_default();
        let service_tiers = parse_model_service_tiers(model.service_tiers.as_ref());
        let default_service_tier = model
            .default_service_tier
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .filter(|value| service_tiers.iter().any(|tier| tier.id == *value))
            .map(str::to_string);
        if api_reasoning_effort_support.is_some() {
            push_unique(&mut api_task_families, ModelTaskFamily::Reasoning);
            push_unique(&mut api_runtime_features, ModelRuntimeFeature::Reasoning);
        }
        let initial_taxonomy = infer_model_taxonomy(ModelTaxonomyInput {
            model_id: &model.id,
            provider_id: Some(provider_id),
            family: model.family.as_deref(),
            description: None,
            capabilities: None,
            explicit_task_families: &api_task_families,
            explicit_input_modalities: &api_input_modalities,
            explicit_output_modalities: &api_output_modalities,
            explicit_runtime_features: &api_runtime_features,
            explicit_deployment_source: None,
            explicit_management_plane: None,
            provider_model_id: Some(model.id.as_str()),
            canonical_model_id: None,
            explicit_alias_source: None,
            canonical_model: canonical_model.as_ref(),
        });
        let mut capabilities = infer_model_capabilities(&initial_taxonomy.task_families);
        if let Some(reasoning_effort) = api_reasoning_effort_support {
            capabilities.reasoning = true;
            capabilities.reasoning_effort = Some(reasoning_effort);
        }
        let taxonomy = infer_model_taxonomy(ModelTaxonomyInput {
            model_id: &model.id,
            provider_id: Some(provider_id),
            family: model.family.as_deref(),
            description: None,
            capabilities: Some(&capabilities),
            explicit_task_families: &initial_taxonomy.task_families,
            explicit_input_modalities: &initial_taxonomy.input_modalities,
            explicit_output_modalities: &initial_taxonomy.output_modalities,
            explicit_runtime_features: &initial_taxonomy.runtime_features,
            explicit_deployment_source: None,
            explicit_management_plane: None,
            provider_model_id: Some(model.id.as_str()),
            canonical_model_id: initial_taxonomy.canonical_model_id.as_deref(),
            explicit_alias_source: initial_taxonomy.alias_source.clone(),
            canonical_model: canonical_model.as_ref(),
        });

        EnhancedModelMetadata {
            id: model.id.clone(),
            display_name,
            provider_id: provider_id.to_string(),
            provider_name: model
                .provider_name
                .unwrap_or_else(|| provider_id.to_string()),
            family: model.family,
            tier: ModelTier::Pro,
            capabilities,
            capability_provenance,
            visibility,
            service_tiers,
            default_service_tier,
            task_families: taxonomy.task_families,
            input_modalities: taxonomy.input_modalities,
            output_modalities: taxonomy.output_modalities,
            runtime_features: taxonomy.runtime_features,
            deployment_source: taxonomy.deployment_source,
            management_plane: taxonomy.management_plane,
            canonical_model_id: taxonomy.canonical_model_id,
            provider_model_id: taxonomy.provider_model_id,
            alias_source: taxonomy.alias_source,
            pricing: None,
            limits: ModelLimits {
                context_length: model.context_length,
                max_output_tokens: None,
                requests_per_minute: None,
                tokens_per_minute: None,
            },
            status: ModelStatus::Active,
            release_date: None,
            is_latest: false,
            description: None,
            source: ModelSource::Api,
            created_at: now,
            updated_at: now,
        }
    }

    fn build_provider_declared_model(
        &self,
        model: &ProviderModelConfig,
        provider_id: &str,
        now: i64,
    ) -> EnhancedModelMetadata {
        let model_id = model.id.trim();
        let mut metadata = self
            .convert_api_model(
                ApiModelResponse {
                    id: model_id.to_string(),
                    display_name: model.display_name.clone(),
                    provider_name: Some(provider_id.to_string()),
                    family: None,
                    context_length: None,
                    task_families: None,
                    input_modalities: None,
                    output_modalities: None,
                    modalities: None,
                    runtime_features: None,
                    vision_supported: None,
                    capabilities: None,
                    supported_parameters: None,
                    reasoning: None,
                    reasoning_effort: None,
                    reasoning_effort_levels: None,
                    reasoning_efforts: None,
                    supported_reasoning_efforts: None,
                    visibility: None,
                    service_tiers: None,
                    default_service_tier: None,
                },
                provider_id,
                now,
            )
            .with_source(ModelSource::Custom);
        if let Some(capability) = model.capability.as_ref() {
            metadata.task_families = capability.task_families.clone();
            metadata.input_modalities = capability.input_modalities.clone();
            metadata.output_modalities = capability.output_modalities.clone();
            metadata.runtime_features = capability.runtime_features.clone();
            metadata.capabilities = capability.capabilities.clone();
            metadata.capability_provenance = ModelCapabilityProvenance::ProviderExplicit;
        }
        metadata
    }

    pub fn build_declared_model_metadata(
        &self,
        provider_id: &str,
        model: &ProviderModelConfig,
    ) -> EnhancedModelMetadata {
        self.build_provider_declared_model(model, provider_id, chrono::Utc::now().timestamp())
    }

    fn build_declared_models(
        &self,
        provider_id: &str,
        models: &[ProviderModelConfig],
        now: i64,
    ) -> Vec<EnhancedModelMetadata> {
        let mut seen = std::collections::HashSet::new();
        models
            .iter()
            .filter(|model| !model.id.trim().is_empty())
            .filter(|model| seen.insert(model.id.trim().to_ascii_lowercase()))
            .map(|model| self.build_provider_declared_model(model, provider_id, now))
            .collect()
    }

    fn is_likely_fal_declared_model(model_id: &str) -> bool {
        let normalized = model_id.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            return false;
        }

        normalized.starts_with("fal-ai/")
            || text_contains_any(
                &normalized,
                &[
                    "nano-banana",
                    "banana",
                    "flux",
                    "seedream",
                    "kontext",
                    "recraft",
                    "ideogram",
                    "sdxl",
                    "stable-diffusion",
                    "image",
                ],
            )
    }

    fn build_fal_declared_models(
        &self,
        provider_id: &str,
        models: &[ProviderModelConfig],
        now: i64,
    ) -> Vec<EnhancedModelMetadata> {
        self.build_declared_models(provider_id, models, now)
            .into_iter()
            .filter(|model| Self::is_likely_fal_declared_model(&model.id))
            .collect()
    }

    fn build_responses_compatible_declared_models(
        &self,
        provider_id: &str,
        models: &[ProviderModelConfig],
        now: i64,
    ) -> Vec<EnhancedModelMetadata> {
        self.build_declared_models(provider_id, models, now)
            .into_iter()
            .filter(|model| {
                model
                    .task_families
                    .contains(&ModelTaskFamily::ImageGeneration)
                    || model.output_modalities.contains(&ModelModality::Image)
            })
            .collect()
    }

    fn should_bypass_proxy_for_models_api_url(url: &str) -> bool {
        let Ok(parsed) = reqwest::Url::parse(url) else {
            return false;
        };

        let Some(host) = parsed.host_str() else {
            return false;
        };
        let normalized_host = host.trim_matches(['[', ']']);

        if matches!(
            normalized_host,
            "localhost" | "127.0.0.1" | "::1" | "0.0.0.0" | "host.docker.internal"
        ) {
            return true;
        }

        normalized_host
            .parse::<std::net::IpAddr>()
            .map(|ip| ip.is_loopback() || ip.is_unspecified())
            .unwrap_or(false)
    }
}

// ============================================================================
// API 响应类型
// ============================================================================

#[derive(Debug, Clone)]
struct ModelsApiError {
    kind: ModelFetchErrorKind,
    message: String,
}

impl ModelsApiError {
    fn new(kind: ModelFetchErrorKind, message: String) -> Self {
        Self { kind, message }
    }
}

/// OpenAI /v1/models API 响应格式
#[derive(Debug, Deserialize)]
struct ApiModelsResponse {
    data: Vec<ApiModelResponse>,
}

#[derive(Debug, Deserialize)]
struct ApiModelModalitiesResponse {
    #[serde(default, alias = "inputModalities", alias = "input_modalities")]
    input: Option<serde_json::Value>,
    #[serde(default, alias = "outputModalities", alias = "output_modalities")]
    output: Option<serde_json::Value>,
}

/// 单个模型的 API 响应
#[derive(Debug, Deserialize)]
struct ApiModelResponse {
    id: String,
    #[serde(default)]
    display_name: Option<String>,
    #[serde(default, alias = "owned_by")]
    provider_name: Option<String>,
    #[serde(default)]
    family: Option<String>,
    #[serde(default)]
    context_length: Option<u32>,
    #[serde(default, alias = "taskFamilies")]
    task_families: Option<serde_json::Value>,
    #[serde(
        default,
        alias = "inputModalities",
        alias = "supportedInputModalities",
        alias = "supported_input_modalities"
    )]
    input_modalities: Option<serde_json::Value>,
    #[serde(
        default,
        alias = "outputModalities",
        alias = "supportedOutputModalities",
        alias = "supported_output_modalities"
    )]
    output_modalities: Option<serde_json::Value>,
    #[serde(default)]
    modalities: Option<ApiModelModalitiesResponse>,
    #[serde(default, alias = "runtimeFeatures")]
    runtime_features: Option<serde_json::Value>,
    #[serde(
        default,
        alias = "visionSupported",
        alias = "supportsVision",
        alias = "supports_vision"
    )]
    vision_supported: Option<bool>,
    #[serde(default)]
    capabilities: Option<serde_json::Value>,
    #[serde(default, alias = "supportedParameters", alias = "supported_params")]
    supported_parameters: Option<serde_json::Value>,
    #[serde(default)]
    reasoning: Option<serde_json::Value>,
    #[serde(default, alias = "reasoningEffort")]
    reasoning_effort: Option<serde_json::Value>,
    #[serde(default, alias = "reasoningEffortLevels")]
    reasoning_effort_levels: Option<serde_json::Value>,
    #[serde(default, alias = "reasoningEfforts")]
    reasoning_efforts: Option<serde_json::Value>,
    #[serde(default, alias = "supportedReasoningEfforts")]
    supported_reasoning_efforts: Option<serde_json::Value>,
    #[serde(default)]
    visibility: Option<String>,
    #[serde(default, alias = "serviceTiers")]
    service_tiers: Option<serde_json::Value>,
    #[serde(default, alias = "defaultServiceTier")]
    default_service_tier: Option<String>,
}

fn api_model_has_explicit_capability_snapshot(model: &ApiModelResponse) -> bool {
    model.vision_supported.is_some()
        || [
            model.task_families.as_ref(),
            model.input_modalities.as_ref(),
            model.output_modalities.as_ref(),
            model.runtime_features.as_ref(),
            model.capabilities.as_ref(),
            model.reasoning.as_ref(),
        ]
        .into_iter()
        .flatten()
        .any(json_value_declares_capability)
        || resolve_api_reasoning_effort_support(model).is_some()
        || explicit_reasoning_effort_disabled(model.reasoning_effort.as_ref())
        || model.modalities.as_ref().is_some_and(|modalities| {
            modalities
                .input
                .as_ref()
                .into_iter()
                .chain(modalities.output.as_ref())
                .any(json_value_declares_capability)
        })
}

fn json_value_declares_capability(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => false,
        serde_json::Value::String(value) => !value.trim().is_empty(),
        serde_json::Value::Array(values) => !values.is_empty(),
        serde_json::Value::Object(values) => !values.is_empty(),
        serde_json::Value::Bool(_) | serde_json::Value::Number(_) => true,
    }
}

#[derive(Debug)]
struct AnthropicModelsResponse {
    models: Vec<ApiModelResponse>,
    has_more: bool,
    last_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawAnthropicModelsResponse {
    #[serde(default)]
    data: Vec<RawAnthropicModelResponse>,
    #[serde(default)]
    has_more: bool,
    #[serde(default)]
    last_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawAnthropicModelResponse {
    id: String,
    #[serde(default)]
    display_name: Option<String>,
}

#[derive(Debug)]
struct GeminiModelsResponse {
    models: Vec<ApiModelResponse>,
    next_page_token: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawGeminiModelsResponse {
    #[serde(default)]
    models: Vec<RawGeminiModelResponse>,
    #[serde(default)]
    next_page_token: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawGeminiModelResponse {
    name: String,
    #[serde(default)]
    display_name: Option<String>,
    #[serde(default)]
    input_token_limit: Option<u32>,
    #[serde(default)]
    supported_generation_methods: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct OllamaModelsResponse {
    #[serde(default)]
    models: Vec<OllamaModelResponse>,
}

#[derive(Debug, Deserialize)]
struct OllamaModelResponse {
    name: String,
    #[serde(default)]
    details: Option<OllamaModelDetails>,
}

#[derive(Debug, Deserialize)]
struct OllamaModelDetails {
    #[serde(default)]
    family: Option<String>,
}

/// 模型获取来源
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelFetchSource {
    /// 从 API 获取
    Api,
    /// API 获取失败且没有本地兜底
    Error,
}

/// 模型获取错误类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ModelFetchErrorKind {
    NotFound,
    Unauthorized,
    Forbidden,
    Network,
    InvalidResponse,
    Other,
}

/// 从 API 获取模型的结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FetchModelsResult {
    /// 模型列表
    pub models: Vec<EnhancedModelMetadata>,
    /// 数据来源
    pub source: ModelFetchSource,
    /// 错误信息（如果有）
    pub error: Option<String>,
    /// 实际请求 URL（如果有）
    pub request_url: Option<String>,
    /// 面向用户的诊断建议（如果有）
    pub diagnostic_hint: Option<String>,
    /// 错误类型（如果有）
    pub error_kind: Option<ModelFetchErrorKind>,
    /// 是否应将该错误作为配置问题强提示
    pub should_prompt_error: bool,
    /// 是否来自 10 天内的 Provider 实时模型缓存
    #[serde(default)]
    pub from_cache: bool,
}

#[cfg(test)]
mod tests {
    use super::{
        infer_model_capabilities, infer_model_taxonomy, infer_runtime_features,
        infer_vision_capability, ModelFetchErrorKind, ModelFetchProtocol, ModelFetchSource,
        ModelRegistryService, ModelTaxonomyInput, LIME_TENANT_HEADER,
        PROVIDER_MODELS_CACHE_TTL_SECONDS,
    };
    use lime_core::database::dao::api_key_provider::ApiProviderType;
    use lime_core::database::dao::route_state::RouteStateDao;
    use lime_core::database::DbConnection;
    use lime_core::models::model_registry::{
        EnhancedModelMetadata, ModelCapabilities, ModelCapabilityProvenance, ModelModality,
        ModelReasoningEffortSource, ModelRuntimeFeature, ModelSource, ModelTaskFamily,
        ModelVisibility, ProviderModelConfig,
    };
    use rusqlite::{params, Connection};
    use std::sync::{Arc, Mutex};

    const RESPONSES_IMAGE_TEST_HOST: &str = "https://images.example/codex";

    fn setup_cache_service() -> (ModelRegistryService, DbConnection) {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
            [],
        )
        .unwrap();
        let db = Arc::new(Mutex::new(conn));
        (ModelRegistryService::new(Arc::clone(&db)), db)
    }

    fn route_generation(db: &DbConnection) -> u64 {
        let conn = db.lock().unwrap();
        RouteStateDao::read_generation(&conn).unwrap()
    }

    fn create_cached_model(id: &str) -> EnhancedModelMetadata {
        EnhancedModelMetadata::new(
            id.to_string(),
            id.to_string(),
            "openai".to_string(),
            "OpenAI".to_string(),
        )
        .with_source(ModelSource::Api)
    }

    #[test]
    fn test_infer_model_taxonomy_uses_image_input_as_vision_signal() {
        let input_modalities = vec![ModelModality::Text, ModelModality::Image];
        let output_modalities = vec![ModelModality::Text];
        let taxonomy = infer_model_taxonomy(ModelTaxonomyInput {
            model_id: "provider-vlm-chat",
            provider_id: Some("custom-provider"),
            family: None,
            description: None,
            capabilities: None,
            explicit_task_families: &[],
            explicit_input_modalities: &input_modalities,
            explicit_output_modalities: &output_modalities,
            explicit_runtime_features: &[],
            explicit_deployment_source: None,
            explicit_management_plane: None,
            provider_model_id: Some("provider-vlm-chat"),
            canonical_model_id: None,
            explicit_alias_source: None,
            canonical_model: None,
        });
        let capabilities = infer_model_capabilities(&taxonomy.task_families);

        assert!(taxonomy
            .task_families
            .contains(&ModelTaskFamily::VisionUnderstanding));
        assert!(capabilities.vision);
    }

    #[test]
    fn test_azure_openai_infers_responses_runtime_feature() {
        let features =
            infer_runtime_features(Some("azure-openai"), None, &[ModelTaskFamily::Chat], &[]);

        assert!(features.contains(&ModelRuntimeFeature::ResponsesApi));
        assert!(!features.contains(&ModelRuntimeFeature::ChatCompletionsApi));
    }

    #[test]
    fn test_infer_model_taxonomy_merges_incomplete_task_family_vision_signals() {
        let explicit_task_families = vec![ModelTaskFamily::Chat];
        let input_modalities = vec![ModelModality::Text, ModelModality::Image];
        let output_modalities = vec![ModelModality::Text];
        let stale_capabilities = ModelCapabilities {
            vision: false,
            tools: true,
            streaming: true,
            json_mode: true,
            function_calling: true,
            reasoning: false,
            reasoning_effort: None,
        };
        let modality_taxonomy = infer_model_taxonomy(ModelTaxonomyInput {
            model_id: "provider-vlm-chat",
            provider_id: Some("custom-provider"),
            family: None,
            description: None,
            capabilities: Some(&stale_capabilities),
            explicit_task_families: &explicit_task_families,
            explicit_input_modalities: &input_modalities,
            explicit_output_modalities: &output_modalities,
            explicit_runtime_features: &[],
            explicit_deployment_source: None,
            explicit_management_plane: None,
            provider_model_id: Some("provider-vlm-chat"),
            canonical_model_id: None,
            explicit_alias_source: None,
            canonical_model: None,
        });
        let name_taxonomy = infer_model_taxonomy(ModelTaxonomyInput {
            model_id: "o3",
            provider_id: Some("openai"),
            family: None,
            description: None,
            capabilities: Some(&stale_capabilities),
            explicit_task_families: &explicit_task_families,
            explicit_input_modalities: &[],
            explicit_output_modalities: &[],
            explicit_runtime_features: &[],
            explicit_deployment_source: None,
            explicit_management_plane: None,
            provider_model_id: Some("o3"),
            canonical_model_id: None,
            explicit_alias_source: None,
            canonical_model: None,
        });

        assert!(modality_taxonomy
            .task_families
            .contains(&ModelTaskFamily::VisionUnderstanding));
        assert!(name_taxonomy
            .task_families
            .contains(&ModelTaskFamily::VisionUnderstanding));
    }

    #[test]
    fn test_infer_model_taxonomy_uses_shared_image_generation_matcher_shape() {
        let agnes_taxonomy = infer_model_taxonomy(ModelTaxonomyInput {
            model_id: "agnes-image-2.1-flash",
            provider_id: Some("openai-compatible"),
            family: None,
            description: None,
            capabilities: None,
            explicit_task_families: &[],
            explicit_input_modalities: &[],
            explicit_output_modalities: &[],
            explicit_runtime_features: &[],
            explicit_deployment_source: None,
            explicit_management_plane: None,
            provider_model_id: Some("agnes-image-2.1-flash"),
            canonical_model_id: None,
            explicit_alias_source: None,
            canonical_model: None,
        });
        let image_input_modalities = vec![ModelModality::Text, ModelModality::Image];
        let text_output_modalities = vec![ModelModality::Text];
        let vision_taxonomy = infer_model_taxonomy(ModelTaxonomyInput {
            model_id: "provider-image-input-chat",
            provider_id: Some("custom-provider"),
            family: None,
            description: None,
            capabilities: None,
            explicit_task_families: &[],
            explicit_input_modalities: &image_input_modalities,
            explicit_output_modalities: &text_output_modalities,
            explicit_runtime_features: &[],
            explicit_deployment_source: None,
            explicit_management_plane: None,
            provider_model_id: Some("provider-image-input-chat"),
            canonical_model_id: None,
            explicit_alias_source: None,
            canonical_model: None,
        });

        assert!(agnes_taxonomy
            .task_families
            .contains(&ModelTaskFamily::ImageGeneration));
        assert!(!agnes_taxonomy
            .task_families
            .contains(&ModelTaskFamily::VisionUnderstanding));
        assert!(vision_taxonomy
            .task_families
            .contains(&ModelTaskFamily::VisionUnderstanding));
        assert!(!vision_taxonomy
            .task_families
            .contains(&ModelTaskFamily::ImageGeneration));
    }

    #[test]
    fn test_model_registry_service_does_not_restore_local_image_generation_matchers() {
        let source = include_str!("model_registry_service.rs");
        let looks_like_marker = ["fn looks_like_", "image_generation_text"].concat();
        let bounded_token_marker = ["fn contains_", "bounded_token"].concat();
        let keywords_marker = ["IMAGE_MODEL", "_KEYWORDS"].concat();

        assert!(!source.contains(&looks_like_marker));
        assert!(!source.contains(&bounded_token_marker));
        assert!(!source.contains(&keywords_marker));
    }

    #[test]
    fn test_official_agnes_endpoint_uses_canonical_model_capabilities() {
        let (service, _db) = setup_cache_service();
        let response = ModelRegistryService::parse_openai_models_response(
            r#"{"data":[{"id":"agnes-2.0-flash"}]}"#,
        )
        .expect("parse Agnes response");
        let model = service.convert_api_model_for_endpoint(
            response.into_iter().next().expect("Agnes model"),
            "custom-provider",
            "https://apihub.agnes-ai.com/v1",
            0,
        );

        assert_eq!(model.provider_id, "custom-provider");
        assert_eq!(
            model.canonical_model_id.as_deref(),
            Some("agnes/agnes-2.0-flash")
        );
        assert_eq!(
            model.capability_provenance,
            ModelCapabilityProvenance::Canonical
        );
        assert!(model.capabilities.vision);
        assert!(model.capabilities.tools);
        assert!(model.capabilities.streaming);
        assert!(model.capabilities.reasoning);
        assert!(model.input_modalities.contains(&ModelModality::Image));
        assert!(model
            .runtime_features
            .contains(&ModelRuntimeFeature::ToolCalling));
    }

    #[test]
    fn test_non_official_agnes_endpoint_does_not_gain_canonical_capabilities() {
        let (service, _db) = setup_cache_service();
        for api_host in [
            "http://apihub.agnes-ai.com/v1",
            "https://apihub.agnes-ai.com:8443/v1",
            "https://apihub.agnes-ai.com.evil.test/v1",
            "https://gateway.example.com/v1",
        ] {
            let response = ModelRegistryService::parse_openai_models_response(
                r#"{"data":[{"id":"agnes-2.0-flash"}]}"#,
            )
            .expect("parse Agnes-shaped response");
            let model = service.convert_api_model_for_endpoint(
                response.into_iter().next().expect("Agnes-shaped model"),
                "custom-provider",
                api_host,
                0,
            );

            assert_eq!(
                model.capability_provenance,
                ModelCapabilityProvenance::InferredHint,
                "{api_host} must not authorize canonical Agnes capabilities"
            );
            assert!(model.canonical_model_id.is_none());
        }
    }

    #[test]
    fn test_convert_api_model_preserves_live_modality_fields() {
        let (service, _db) = setup_cache_service();
        let response = ModelRegistryService::parse_openai_models_response(
            r#"{
              "data": [
                {
                  "id": "provider-vlm-chat",
                  "display_name": "Provider VLM Chat",
                  "task_families": ["chat"],
                  "input_modalities": ["text", "image"],
                  "output_modalities": ["text"],
                  "capabilities": { "vision": false }
                }
              ]
            }"#,
        )
        .expect("parse response");
        let model =
            service.convert_api_model(response.into_iter().next().expect("model"), "gateway", 0);

        assert!(model.input_modalities.contains(&ModelModality::Image));
        assert!(model
            .task_families
            .contains(&ModelTaskFamily::VisionUnderstanding));
        assert!(model.capabilities.vision);
        assert_eq!(
            model.capability_provenance,
            ModelCapabilityProvenance::ProviderExplicit
        );
    }

    #[test]
    fn test_convert_api_model_uses_capability_object_as_modality_signal() {
        let (service, _db) = setup_cache_service();
        let response = ModelRegistryService::parse_openai_models_response(
            r#"{
              "data": [
                {
                  "id": "provider-live-vlm",
                  "capabilities": {
                    "image_input": true,
                    "tool_calling": true,
                    "json_schema": true
                  }
                }
              ]
            }"#,
        )
        .expect("parse response");
        let model =
            service.convert_api_model(response.into_iter().next().expect("model"), "gateway", 0);

        assert!(model.input_modalities.contains(&ModelModality::Image));
        assert!(model
            .task_families
            .contains(&ModelTaskFamily::VisionUnderstanding));
        assert!(model.capabilities.vision);
        assert!(model
            .runtime_features
            .contains(&ModelRuntimeFeature::ToolCalling));
        assert!(model
            .runtime_features
            .contains(&ModelRuntimeFeature::JsonSchema));
    }

    #[test]
    fn test_convert_api_model_uses_nested_modalities_as_vision_signal() {
        let (service, _db) = setup_cache_service();
        let response = ModelRegistryService::parse_openai_models_response(
            r#"{
              "data": [
                {
                  "id": "grok-4.3",
                  "modalities": {
                    "input": ["text", "image"],
                    "output": ["text"]
                  }
                }
              ]
            }"#,
        )
        .expect("parse response");
        let model =
            service.convert_api_model(response.into_iter().next().expect("model"), "xai", 0);

        assert!(model.input_modalities.contains(&ModelModality::Image));
        assert!(model
            .task_families
            .contains(&ModelTaskFamily::VisionUnderstanding));
        assert!(model.capabilities.vision);
    }

    #[test]
    fn test_convert_api_model_uses_warp_style_vision_supported_flag() {
        let (service, _db) = setup_cache_service();
        let response = ModelRegistryService::parse_openai_models_response(
            r#"{
              "data": [
                {
                  "id": "provider-auto",
                  "vision_supported": true
                }
              ]
            }"#,
        )
        .expect("parse response");
        let model =
            service.convert_api_model(response.into_iter().next().expect("model"), "gateway", 0);

        assert!(model.input_modalities.contains(&ModelModality::Image));
        assert!(model
            .task_families
            .contains(&ModelTaskFamily::VisionUnderstanding));
        assert!(model.capabilities.vision);
    }

    #[test]
    fn test_convert_api_model_uses_supported_parameters_for_reasoning_effort() {
        let (service, _db) = setup_cache_service();
        let response = ModelRegistryService::parse_openai_models_response(
            r#"{
              "data": [
                {
                  "id": "o3-mini",
                  "supported_parameters": ["temperature", "reasoning_effort"]
                }
              ]
            }"#,
        )
        .expect("parse response");
        let model =
            service.convert_api_model(response.into_iter().next().expect("model"), "gateway", 0);
        let support = model
            .capabilities
            .reasoning_effort
            .expect("reasoning effort support");

        assert!(support.supported);
        assert_eq!(support.source, Some(ModelReasoningEffortSource::Api));
        assert!(support.levels.is_empty());
        assert!(support.options.is_empty());
        assert_eq!(support.default, None);
    }

    #[test]
    fn test_convert_api_model_preserves_catalog_visibility_and_service_tiers() {
        let (service, _db) = setup_cache_service();
        let response = ModelRegistryService::parse_openai_models_response(
            r#"{
              "data": [
                {
                  "id": "gpt-5.6-sol",
                  "visibility": "list",
                  "service_tiers": [
                    {"id": "priority", "name": "Priority", "description": "Fast queue"},
                    "flex",
                    {"id": "priority", "name": "Duplicate", "description": "ignored"}
                  ],
                  "default_service_tier": "priority"
                }
              ]
            }"#,
        )
        .expect("parse response");
        let model =
            service.convert_api_model(response.into_iter().next().expect("model"), "openai", 0);

        assert_eq!(model.visibility, ModelVisibility::List);
        assert_eq!(
            model
                .service_tiers
                .iter()
                .map(|tier| tier.id.as_str())
                .collect::<Vec<_>>(),
            vec!["priority", "flex"]
        );
        assert_eq!(model.default_service_tier.as_deref(), Some("priority"));
        assert!(model.supports_service_tier("priority"));
        assert!(!model.supports_service_tier("unknown"));
    }

    #[test]
    fn test_convert_api_model_drops_unlisted_default_service_tier() {
        let (service, _db) = setup_cache_service();
        let response = ModelRegistryService::parse_openai_models_response(
            r#"{
              "data": [
                {
                  "id": "gpt-5.6-sol",
                  "service_tiers": [{"id": "priority"}],
                  "default_service_tier": "unknown"
                }
              ]
            }"#,
        )
        .expect("parse response");
        let model =
            service.convert_api_model(response.into_iter().next().expect("model"), "openai", 0);

        assert_eq!(model.default_service_tier, None);
    }

    #[test]
    fn test_convert_api_model_preserves_explicit_reasoning_effort_levels() {
        let (service, _db) = setup_cache_service();
        let response = ModelRegistryService::parse_openai_models_response(
            r#"{
              "data": [
                {
                  "id": "gpt-5.4",
                  "reasoning_effort_levels": ["low", "medium", "high", "xhigh"]
                }
              ]
            }"#,
        )
        .expect("parse response");
        let model =
            service.convert_api_model(response.into_iter().next().expect("model"), "gateway", 0);
        let support = model
            .capabilities
            .reasoning_effort
            .expect("reasoning effort support");

        assert_eq!(
            support.levels,
            vec![
                "low".to_string(),
                "medium".to_string(),
                "high".to_string(),
                "xhigh".to_string(),
            ]
        );
        assert_eq!(support.default, None);
    }

    #[test]
    fn test_convert_api_model_does_not_use_first_reasoning_effort_as_default() {
        let (service, _db) = setup_cache_service();
        let response = ModelRegistryService::parse_openai_models_response(
            r#"{
              "data": [
                {
                  "id": "provider-reasoning",
                  "reasoning_effort": ["low", "high"]
                }
              ]
            }"#,
        )
        .expect("parse response");
        let model =
            service.convert_api_model(response.into_iter().next().expect("model"), "gateway", 0);
        let support = model
            .capabilities
            .reasoning_effort
            .expect("reasoning effort support");

        assert_eq!(support.levels, vec!["low".to_string(), "high".to_string()]);
        assert_eq!(support.default, None);
    }

    #[test]
    fn test_convert_api_model_preserves_reasoning_effort_menu_and_wire_values() {
        let (service, _db) = setup_cache_service();
        let response = ModelRegistryService::parse_openai_models_response(
            r#"{
              "data": [
                {
                  "id": "grok-4.5",
                  "reasoning_efforts": [
                    {
                      "id": "deep",
                      "value": "xhigh",
                      "label": "Deep",
                      "description": "Maximum reasoning",
                      "default": true
                    },
                    {
                      "id": "native",
                      "value": "provider-native",
                      "label": "Native"
                    },
                    "ultra"
                  ]
                }
              ]
            }"#,
        )
        .expect("parse response");
        let model =
            service.convert_api_model(response.into_iter().next().expect("model"), "xai", 0);
        let support = model
            .capabilities
            .reasoning_effort
            .expect("reasoning effort support");

        assert_eq!(
            support.levels,
            vec![
                "xhigh".to_string(),
                "provider-native".to_string(),
                "ultra".to_string(),
            ]
        );
        assert_eq!(support.default, Some("xhigh".to_string()));
        assert_eq!(support.options.len(), 3);
        assert_eq!(support.options[0].id, "deep");
        assert_eq!(support.options[0].label, "Deep");
        assert_eq!(support.options[0].value, "xhigh");
        assert_eq!(
            support.options[0].description.as_deref(),
            Some("Maximum reasoning")
        );
        assert_eq!(support.options[1].id, "native");
        assert_eq!(support.options[1].value, "provider-native");
        assert_eq!(support.options[2].id, "ultra");
        assert_eq!(support.options[2].value, "ultra");
    }

    #[test]
    fn test_convert_api_model_does_not_infer_reasoning_from_model_or_provider_name() {
        let (service, _db) = setup_cache_service();
        let response = ModelRegistryService::parse_openai_models_response(
            r#"{
              "data": [
                {
                  "id": "thinking-reasoning-codex"
                }
              ]
            }"#,
        )
        .expect("parse response");
        let model =
            service.convert_api_model(response.into_iter().next().expect("model"), "codex", 0);

        assert!(!model.capabilities.reasoning);
        assert!(model.capabilities.reasoning_effort.is_none());
        assert_eq!(
            model.capability_provenance,
            ModelCapabilityProvenance::InferredHint
        );
        assert!(!model.task_families.contains(&ModelTaskFamily::Reasoning));
        assert!(!model
            .runtime_features
            .contains(&ModelRuntimeFeature::Reasoning));
    }

    #[test]
    fn test_convert_api_model_does_not_infer_reasoning_effort_from_reasoning_flag() {
        let (service, _db) = setup_cache_service();
        let response = ModelRegistryService::parse_openai_models_response(
            r#"{
              "data": [
                {
                  "id": "reasoning-chat",
                  "capabilities": { "reasoning": true }
                }
              ]
            }"#,
        )
        .expect("parse response");
        let model =
            service.convert_api_model(response.into_iter().next().expect("model"), "gateway", 0);

        assert!(model.capabilities.reasoning);
        assert!(model.capabilities.reasoning_effort.is_none());
        assert_eq!(
            model.capability_provenance,
            ModelCapabilityProvenance::ProviderExplicit
        );
    }

    #[test]
    fn test_convert_api_model_does_not_enable_reasoning_effort_from_generic_effort_parameter() {
        let (service, _db) = setup_cache_service();
        let response = ModelRegistryService::parse_openai_models_response(
            r#"{
              "data": [
                {
                  "id": "proxy-chat",
                  "supported_parameters": ["temperature", "effort"]
                }
              ]
            }"#,
        )
        .expect("parse response");
        let model =
            service.convert_api_model(response.into_iter().next().expect("model"), "gateway", 0);

        assert!(model.capabilities.reasoning_effort.is_none());
        assert_eq!(
            model.capability_provenance,
            ModelCapabilityProvenance::InferredHint
        );
    }

    #[test]
    fn test_convert_api_model_does_not_treat_reasoning_effort_metadata_as_capability() {
        let (service, _db) = setup_cache_service();
        let response = ModelRegistryService::parse_openai_models_response(
            r#"{
              "data": [
                {
                  "id": "proxy-chat",
                  "reasoning_effort": {
                    "label": "Reasoning effort",
                    "description": "Provider metadata only"
                  }
                }
              ]
            }"#,
        )
        .expect("parse response");
        let model =
            service.convert_api_model(response.into_iter().next().expect("model"), "gateway", 0);

        assert!(model.capabilities.reasoning_effort.is_none());
        assert_eq!(
            model.capability_provenance,
            ModelCapabilityProvenance::InferredHint
        );
    }

    #[test]
    fn test_convert_api_model_respects_disabled_reasoning_effort_signal() {
        let (service, _db) = setup_cache_service();
        let response = ModelRegistryService::parse_openai_models_response(
            r#"{
              "data": [
                {
                  "id": "proxy-chat",
                  "reasoning_effort": { "supported": false, "levels": ["low", "medium", "high"] }
                }
              ]
            }"#,
        )
        .expect("parse response");
        let model =
            service.convert_api_model(response.into_iter().next().expect("model"), "gateway", 0);

        assert!(model.capabilities.reasoning_effort.is_none());
        assert_eq!(
            model.capability_provenance,
            ModelCapabilityProvenance::ProviderExplicit
        );
    }

    #[test]
    fn test_infer_vision_capability_recognizes_modern_vision_models() {
        for (provider, model) in [
            ("openai", "o3"),
            ("openai", "o4-mini"),
            ("custom-provider", "agnes-2.0-flash"),
            ("xai", "grok-4.3"),
            ("mistral", "mistral-small-latest"),
            ("alibaba", "qwen3.5-27b"),
            ("google", "gemma-3-27b-it"),
        ] {
            assert!(
                infer_vision_capability(model, Some(provider), None, None),
                "{provider}:{model} should support image input"
            );
        }
    }

    #[test]
    fn test_infer_vision_capability_keeps_non_vision_siblings_false() {
        for (provider, model) in [
            ("openai", "o1-mini"),
            ("openai", "o1-preview"),
            ("openai", "o3-mini"),
            ("custom-provider", "agnes-image-2.0-flash"),
            ("custom-provider", "agnes-image-2.1-flash"),
            ("xai", "grok-3-mini"),
            ("google", "gemma-3n-e4b-it"),
        ] {
            assert!(
                !infer_vision_capability(model, Some(provider), None, None),
                "{provider}:{model} should not be inferred as image input"
            );
        }
    }

    #[test]
    fn test_declared_agnes_models_keep_input_and_output_capabilities_distinct() {
        let (service, _db) = setup_cache_service();
        let chat = service.build_declared_model_metadata(
            "custom-provider",
            &ProviderModelConfig::hint("agnes-2.0-flash"),
        );
        let image = service.build_declared_model_metadata(
            "custom-provider",
            &ProviderModelConfig::hint("agnes-image-2.1-flash"),
        );

        assert_eq!(
            chat.capability_provenance,
            ModelCapabilityProvenance::InferredHint
        );
        assert_eq!(
            image.capability_provenance,
            ModelCapabilityProvenance::InferredHint
        );

        assert!(chat.capabilities.vision);
        assert!(chat
            .task_families
            .contains(&ModelTaskFamily::VisionUnderstanding));
        assert!(chat.input_modalities.contains(&ModelModality::Image));
        assert!(chat.output_modalities.contains(&ModelModality::Text));

        assert!(!image.capabilities.vision);
        assert!(image
            .task_families
            .contains(&ModelTaskFamily::ImageGeneration));
        assert_eq!(image.input_modalities, vec![ModelModality::Text]);
        assert_eq!(image.output_modalities, vec![ModelModality::Image]);
    }

    #[test]
    fn test_infer_model_taxonomy_uses_provider_and_canonical_ids_for_vision() {
        let provider_model_taxonomy = infer_model_taxonomy(ModelTaxonomyInput {
            model_id: "relay-fast-default",
            provider_id: Some("openai"),
            family: None,
            description: None,
            capabilities: None,
            explicit_task_families: &[],
            explicit_input_modalities: &[],
            explicit_output_modalities: &[],
            explicit_runtime_features: &[],
            explicit_deployment_source: None,
            explicit_management_plane: None,
            provider_model_id: Some("o3"),
            canonical_model_id: None,
            explicit_alias_source: None,
            canonical_model: None,
        });
        let canonical_taxonomy = infer_model_taxonomy(ModelTaxonomyInput {
            model_id: "relay-grok-latest",
            provider_id: Some("xai"),
            family: None,
            description: None,
            capabilities: None,
            explicit_task_families: &[],
            explicit_input_modalities: &[],
            explicit_output_modalities: &[],
            explicit_runtime_features: &[],
            explicit_deployment_source: None,
            explicit_management_plane: None,
            provider_model_id: Some("relay-grok-latest"),
            canonical_model_id: Some("grok-4.3"),
            explicit_alias_source: None,
            canonical_model: None,
        });

        assert!(provider_model_taxonomy
            .task_families
            .contains(&ModelTaskFamily::VisionUnderstanding));
        assert!(canonical_taxonomy
            .task_families
            .contains(&ModelTaskFamily::VisionUnderstanding));

        let openai_canonical_taxonomy = infer_model_taxonomy(ModelTaxonomyInput {
            model_id: "relay-o3-latest",
            provider_id: Some("openai"),
            family: None,
            description: None,
            capabilities: None,
            explicit_task_families: &[],
            explicit_input_modalities: &[],
            explicit_output_modalities: &[],
            explicit_runtime_features: &[],
            explicit_deployment_source: None,
            explicit_management_plane: None,
            provider_model_id: Some("relay-o3-latest"),
            canonical_model_id: Some("openai/o3"),
            explicit_alias_source: None,
            canonical_model: None,
        });
        assert!(openai_canonical_taxonomy
            .task_families
            .contains(&ModelTaskFamily::VisionUnderstanding));
    }

    #[test]
    fn test_build_models_api_url() {
        assert_eq!(
            ModelRegistryService::build_models_api_url("https://api.openai.com"),
            "https://api.openai.com/v1/models"
        );
        assert_eq!(
            ModelRegistryService::build_models_api_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1/models"
        );
        assert_eq!(
            ModelRegistryService::build_models_api_url("https://open.bigmodel.cn/api/anthropic"),
            "https://open.bigmodel.cn/api/anthropic/v1/models"
        );
        assert_eq!(
            ModelRegistryService::build_models_api_url("https://open.bigmodel.cn/api/paas/v4/"),
            "https://open.bigmodel.cn/api/paas/v4/models"
        );
        assert_eq!(
            ModelRegistryService::build_models_api_url("https://ark.cn-beijing.volces.com/api/v3/"),
            "https://ark.cn-beijing.volces.com/api/v3/models"
        );
        assert_eq!(
            ModelRegistryService::build_models_api_url("https://example.com/proxy/api/v9"),
            "https://example.com/proxy/api/v9/models"
        );
        assert_eq!(
            ModelRegistryService::build_models_api_url(
                "https://gateway.example.com/proxy/responses"
            ),
            "https://gateway.example.com/proxy/v1/models"
        );
    }

    #[test]
    fn test_prepare_model_fetch_request_adds_lime_tenant_header_from_fragment() {
        let request = ModelRegistryService::prepare_model_fetch_request(
            "lime-hub",
            "https://llm.limeai.run#lime_tenant_id=tenant-0001",
            "sk-lime-test",
            Some(ApiProviderType::Openai),
        )
        .expect("prepare Lime model fetch request");

        assert_eq!(request.url, "https://llm.limeai.run/v1/models");
        assert!(request
            .headers
            .iter()
            .any(|(name, value)| { name == "Authorization" && value == "Bearer sk-lime-test" }));
        assert!(request
            .headers
            .iter()
            .any(|(name, value)| { name == LIME_TENANT_HEADER && value == "tenant-0001" }));
    }

    #[test]
    fn test_build_models_api_hint_ignores_lime_tenant_fragment() {
        let hint = ModelRegistryService::build_models_api_hint(
            "lime-hub",
            "https://llm.limeai.run#lime_tenant_id=tenant-0001",
            "https://llm.limeai.run/v1/models",
        );

        assert_eq!(hint, None);
    }

    #[test]
    fn test_provider_models_cache_hits_within_ten_days() {
        let (service, db) = setup_cache_service();
        let now = chrono::Utc::now().timestamp();
        let request_url = "https://api.openai.com/v1/models".to_string();

        service
            .save_provider_models_cache(
                "openai",
                "https://api.openai.com/v1",
                Some(ApiProviderType::Openai),
                &[create_cached_model("gpt-5.1")],
                Some(request_url.clone()),
                now,
            )
            .expect("cache should be saved");
        assert_eq!(route_generation(&db), 1);

        let cached = service
            .get_cached_provider_models(
                "openai",
                "https://api.openai.com/v1",
                Some(ApiProviderType::Openai),
            )
            .expect("cache read should not fail")
            .expect("cache should hit");

        assert_eq!(cached.source, ModelFetchSource::Api);
        assert!(cached.from_cache);
        assert_eq!(cached.request_url.as_deref(), Some(request_url.as_str()));
        assert_eq!(cached.models.len(), 1);
        assert_eq!(cached.models[0].id, "gpt-5.1");
        assert_eq!(route_generation(&db), 1);
    }

    #[test]
    fn test_provider_models_cache_generation_advances_only_for_changed_payload() {
        let (service, db) = setup_cache_service();
        let now = chrono::Utc::now().timestamp();
        let cached_model = create_cached_model("gpt-5.1");

        service
            .save_provider_models_cache(
                "openai",
                "https://api.openai.com/v1",
                Some(ApiProviderType::Openai),
                std::slice::from_ref(&cached_model),
                None,
                now,
            )
            .expect("new cache should be saved");
        assert_eq!(route_generation(&db), 1);

        service
            .save_provider_models_cache(
                "openai",
                "https://api.openai.com/v1",
                Some(ApiProviderType::Openai),
                std::slice::from_ref(&cached_model),
                None,
                now,
            )
            .expect("identical cache should be ignored");
        assert_eq!(route_generation(&db), 1);

        service
            .save_provider_models_cache(
                "openai",
                "https://api.openai.com/v1",
                Some(ApiProviderType::Openai),
                &[create_cached_model("gpt-5.2")],
                None,
                now,
            )
            .expect("changed cache should be saved");
        assert_eq!(route_generation(&db), 2);
    }

    #[test]
    fn credential_scoped_catalog_keeps_one_last_success_snapshot() {
        let (service, db) = setup_cache_service();
        let now = chrono::Utc::now().timestamp();
        let provider_id = "openai";
        let api_host = "https://api.openai.com/v1";
        let provider_type = Some(ApiProviderType::Openai);

        service
            .save_provider_models_cache_scoped(
                provider_id,
                api_host,
                provider_type,
                &[create_cached_model("gpt-5.6")],
                None,
                now,
                Some("credential-a"),
            )
            .expect("credential catalog should be saved");

        let scoped = service
            .get_cached_provider_models_scoped(
                provider_id,
                api_host,
                provider_type,
                Some("credential-a"),
            )
            .expect("scoped cache read")
            .expect("scoped cache");
        let last_success = service
            .get_cached_provider_models(provider_id, api_host, provider_type)
            .expect("last success cache read")
            .expect("last success cache");
        assert_eq!(scoped.models[0].id, "gpt-5.6");
        assert_eq!(last_success.models[0].id, "gpt-5.6");
        assert_eq!(route_generation(&db), 1);
    }

    #[test]
    fn successful_empty_catalog_clears_last_success_without_touching_old_scope() {
        let (service, db) = setup_cache_service();
        let now = chrono::Utc::now().timestamp();
        let provider_id = "openai";
        let api_host = "https://api.openai.com/v1";
        let provider_type = Some(ApiProviderType::Openai);

        service
            .save_provider_models_cache_scoped(
                provider_id,
                api_host,
                provider_type,
                &[create_cached_model("gpt-5.6")],
                None,
                now,
                Some("credential-a"),
            )
            .expect("first credential catalog should be saved");
        service
            .save_provider_models_cache_scoped(
                provider_id,
                api_host,
                provider_type,
                &[],
                None,
                now + 1,
                Some("credential-b"),
            )
            .expect("empty catalog should replace last success");

        assert!(service
            .get_cached_provider_models(provider_id, api_host, provider_type)
            .expect("last success cache read")
            .is_none());
        assert!(service
            .get_cached_provider_models_scoped(
                provider_id,
                api_host,
                provider_type,
                Some("credential-b"),
            )
            .expect("new scoped cache read")
            .is_none());
        assert_eq!(
            service
                .get_cached_provider_models_scoped(
                    provider_id,
                    api_host,
                    provider_type,
                    Some("credential-a"),
                )
                .expect("old scoped cache read")
                .expect("old scoped cache")
                .models[0]
                .id,
            "gpt-5.6"
        );
        assert_eq!(route_generation(&db), 2);
    }

    #[test]
    fn test_clear_provider_models_cache_advances_only_when_rows_are_deleted() {
        let (service, db) = setup_cache_service();
        let now = chrono::Utc::now().timestamp();

        assert_eq!(service.clear_provider_models_cache().unwrap(), 0);
        assert_eq!(route_generation(&db), 0);

        service
            .save_provider_models_cache(
                "openai",
                "https://api.openai.com/v1",
                Some(ApiProviderType::Openai),
                &[create_cached_model("gpt-5.1")],
                None,
                now,
            )
            .expect("cache should be saved");
        assert_eq!(route_generation(&db), 1);

        assert_eq!(service.clear_provider_models_cache().unwrap(), 1);
        assert_eq!(route_generation(&db), 2);
        assert_eq!(service.clear_provider_models_cache().unwrap(), 0);
        assert_eq!(route_generation(&db), 2);
    }

    #[test]
    fn test_provider_models_cache_rejects_previous_taxonomy_version() {
        let (service, db) = setup_cache_service();
        let now = chrono::Utc::now().timestamp();
        let provider_id = "custom-provider";
        let api_host = "https://gateway.example.com/v1";
        let cache_key = ModelRegistryService::provider_models_cache_key(
            provider_id,
            api_host,
            Some(ApiProviderType::Openai),
        );
        let legacy_payload = serde_json::json!({
            "provider_id": provider_id,
            "api_host": api_host,
            "provider_type": "openai",
            "request_url": "https://gateway.example.com/v1/models",
            "fetched_at": now,
            "expires_at": now + PROVIDER_MODELS_CACHE_TTL_SECONDS,
            "models": [create_cached_model("agnes-2.0-flash")]
        });
        db.lock()
            .unwrap()
            .execute(
                "INSERT INTO settings (key, value) VALUES (?1, ?2)",
                params![cache_key, legacy_payload.to_string()],
            )
            .expect("insert legacy cache");

        let cached = service
            .get_cached_provider_models(provider_id, api_host, Some(ApiProviderType::Openai))
            .expect("cache read should not fail");

        assert!(cached.is_none());
        let remaining: i64 = db
            .lock()
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM settings WHERE key = ?1",
                params![cache_key],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(remaining, 0);
    }

    #[test]
    fn test_provider_models_cache_expires_after_ten_days() {
        let (service, db) = setup_cache_service();
        let expired_at = chrono::Utc::now().timestamp() - PROVIDER_MODELS_CACHE_TTL_SECONDS - 1;

        service
            .save_provider_models_cache(
                "openai",
                "https://api.openai.com/v1",
                Some(ApiProviderType::Openai),
                &[create_cached_model("gpt-5.1")],
                Some("https://api.openai.com/v1/models".to_string()),
                expired_at,
            )
            .expect("cache should be saved");

        let cached = service
            .get_cached_provider_models(
                "openai",
                "https://api.openai.com/v1",
                Some(ApiProviderType::Openai),
            )
            .expect("cache read should not fail");

        assert!(cached.is_none());
        assert_eq!(route_generation(&db), 2);

        let remaining: i64 = db
            .lock()
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM settings WHERE key GLOB ?1",
                params![format!("{}*", super::PROVIDER_MODELS_CACHE_KEY_PREFIX)],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(remaining, 0);
    }

    #[tokio::test]
    async fn test_responses_compatible_fetch_uses_declared_image_models() {
        let (service, _db) = setup_cache_service();

        let result = service
            .fetch_models_from_api_with_hints(
                "airgate-openai-images",
                RESPONSES_IMAGE_TEST_HOST,
                "sk-test",
                Some(ApiProviderType::Openai),
                &[ProviderModelConfig::hint("gpt-images-2")],
            )
            .await
            .expect("responses declared image models should resolve");

        assert_eq!(result.source, ModelFetchSource::Api);
        assert_eq!(result.models.len(), 1);
        assert_eq!(result.models[0].id, "gpt-images-2");
        assert_eq!(result.models[0].source, ModelSource::Custom);
        assert!(result.models[0]
            .task_families
            .contains(&lime_core::models::model_registry::ModelTaskFamily::ImageGeneration));
        assert!(result.models[0]
            .runtime_features
            .contains(&lime_core::models::model_registry::ModelRuntimeFeature::ImagesApi));
        assert!(result
            .diagnostic_hint
            .as_deref()
            .unwrap_or_default()
            .contains("已使用 Provider 中声明的图片模型"));

        let credential_fingerprint = ModelRegistryService::credential_cache_fingerprint("sk-test");
        let cached = service
            .get_cached_provider_models_scoped(
                "airgate-openai-images",
                RESPONSES_IMAGE_TEST_HOST,
                Some(ApiProviderType::Openai),
                credential_fingerprint.as_deref(),
            )
            .expect("cache read should not fail")
            .expect("declared image models should be cached");

        assert!(cached.from_cache);
        assert_eq!(cached.models[0].id, "gpt-images-2");
    }

    #[test]
    fn provider_models_cache_isolated_by_credential_fingerprint() {
        let first = ModelRegistryService::provider_models_cache_key_scoped(
            "openai",
            "https://api.openai.com/v1",
            Some(ApiProviderType::Openai),
            Some("credential-a"),
        );
        let second = ModelRegistryService::provider_models_cache_key_scoped(
            "openai",
            "https://api.openai.com/v1",
            Some(ApiProviderType::Openai),
            Some("credential-b"),
        );
        let unscoped = ModelRegistryService::provider_models_cache_key(
            "openai",
            "https://api.openai.com/v1",
            Some(ApiProviderType::Openai),
        );

        assert_ne!(first, second);
        assert_ne!(first, unscoped);
        assert_ne!(second, unscoped);
    }

    #[tokio::test]
    async fn test_openai_response_fetch_uses_declared_image_models_on_plain_base_url() {
        let (service, _db) = setup_cache_service();

        let result = service
            .fetch_models_from_api_with_hints(
                "openai-gpt-images-2",
                "https://api.openai.com/v1",
                "sk-test",
                Some(ApiProviderType::OpenaiResponse),
                &[ProviderModelConfig::hint("gpt-images-2")],
            )
            .await
            .expect("openai-response declared image models should resolve");

        assert_eq!(result.source, ModelFetchSource::Api);
        assert_eq!(result.models.len(), 1);
        assert_eq!(result.models[0].id, "gpt-images-2");
        assert!(result
            .diagnostic_hint
            .as_deref()
            .unwrap_or_default()
            .contains("已使用 Provider 中声明的图片模型"));
    }

    #[tokio::test]
    async fn test_fal_like_fetch_uses_declared_models_without_models_api() {
        let (service, _db) = setup_cache_service();

        let result = service
            .fetch_models_from_api_with_hints(
                "fal",
                "https://fal.run/fal-ai",
                "",
                Some(ApiProviderType::Openai),
                &[ProviderModelConfig::hint("fal-ai/nano-banana-pro")],
            )
            .await
            .expect("fal declared models should resolve");

        assert_eq!(result.source, ModelFetchSource::Api);
        assert_eq!(result.request_url, None);
        assert_eq!(result.models.len(), 1);
        assert_eq!(result.models[0].id, "fal-ai/nano-banana-pro");
        assert_eq!(result.models[0].source, ModelSource::Custom);
        assert!(result
            .diagnostic_hint
            .as_deref()
            .unwrap_or_default()
            .contains("Fal 不提供标准 /models 枚举"));

        let cached = service
            .get_cached_provider_models(
                "fal",
                "https://fal.run/fal-ai",
                Some(ApiProviderType::Openai),
            )
            .expect("cache read should not fail")
            .expect("declared fal models should be cached");

        assert!(cached.from_cache);
        assert_eq!(cached.models[0].id, "fal-ai/nano-banana-pro");
    }

    #[tokio::test]
    async fn test_xiaomi_fetch_uses_declared_models_without_models_api() {
        let (service, _db) = setup_cache_service();

        let result = service
            .fetch_models_from_api_with_hints(
                "xiaomi",
                "https://token-plan-sgp.xiaomimimo.com/anthropic",
                "sk-test",
                Some(ApiProviderType::Openai),
                &[
                    ProviderModelConfig::hint("mimo-v2.5-flash"),
                    ProviderModelConfig::hint("mimo-v2.5-pro"),
                ],
            )
            .await
            .expect("mimo declared models should resolve without /models");

        assert_eq!(result.source, ModelFetchSource::Api);
        assert_eq!(result.request_url, None);
        assert_eq!(result.models.len(), 2);
        assert_eq!(result.models[0].id, "mimo-v2.5-flash");
        assert_eq!(result.models[1].id, "mimo-v2.5-pro");
        assert_eq!(result.models[0].source, ModelSource::Custom);
        assert!(result
            .diagnostic_hint
            .as_deref()
            .unwrap_or_default()
            .contains("不提供标准 /models 枚举"));

        let credential_fingerprint = ModelRegistryService::credential_cache_fingerprint("sk-test");
        let cached = service
            .get_cached_provider_models_scoped(
                "xiaomi",
                "https://token-plan-sgp.xiaomimimo.com/anthropic",
                Some(ApiProviderType::Openai),
                credential_fingerprint.as_deref(),
            )
            .expect("cache read should not fail")
            .expect("mimo known model should be cached");

        assert!(cached.from_cache);
        assert_eq!(cached.models[0].id, "mimo-v2.5-flash");
        assert_eq!(cached.models[1].id, "mimo-v2.5-pro");
    }

    #[tokio::test]
    async fn test_xiaomi_fetch_errors_without_declared_models() {
        let (service, _db) = setup_cache_service();

        let result = service
            .fetch_models_from_api_with_hints(
                "xiaomi",
                "https://token-plan-sgp.xiaomimimo.com/anthropic",
                "sk-test",
                Some(ApiProviderType::Openai),
                &[],
            )
            .await
            .expect("mimo missing declared models should return structured error");

        assert_eq!(result.source, ModelFetchSource::Error);
        assert!(result.models.is_empty());
        assert!(result
            .diagnostic_hint
            .as_deref()
            .unwrap_or_default()
            .contains("不再内置固定模型名"));
    }

    #[tokio::test]
    async fn test_fal_like_fetch_errors_without_declared_models() {
        let (service, _db) = setup_cache_service();

        let result = service
            .fetch_models_from_api_with_hints(
                "fal",
                "https://fal.run/fal-ai",
                "",
                Some(ApiProviderType::Openai),
                &[],
            )
            .await
            .expect("fal without declared models should return structured result");

        assert_eq!(result.source, ModelFetchSource::Error);
        assert_eq!(result.request_url, None);
        assert!(result.models.is_empty());
        assert_eq!(result.error_kind, Some(ModelFetchErrorKind::Other));
        assert!(result
            .diagnostic_hint
            .as_deref()
            .unwrap_or_default()
            .contains("fal-ai/..."));
    }

    #[tokio::test]
    async fn test_fal_like_fetch_ignores_non_fal_declared_models() {
        let (service, _db) = setup_cache_service();

        let result = service
            .fetch_models_from_api_with_hints(
                "fal",
                "https://fal.run/fal-ai",
                "",
                Some(ApiProviderType::Openai),
                &[ProviderModelConfig::hint("gpt-5.2-pro")],
            )
            .await
            .expect("fal non-image declared models should return structured result");

        assert_eq!(result.source, ModelFetchSource::Error);
        assert!(result.models.is_empty());
        assert!(result
            .diagnostic_hint
            .as_deref()
            .unwrap_or_default()
            .contains("当前模型优先级没有可用 Fal 图片模型"));
    }

    #[tokio::test]
    async fn test_responses_compatible_fetch_still_errors_without_declared_image_models() {
        let (service, _db) = setup_cache_service();

        let result = service
            .fetch_models_from_api_with_hints(
                "airgate-openai-images",
                RESPONSES_IMAGE_TEST_HOST,
                "sk-test",
                Some(ApiProviderType::Openai),
                &[],
            )
            .await
            .expect("responses without declared models should return structured result");

        assert_eq!(result.source, ModelFetchSource::Error);
        assert!(result.models.is_empty());
        assert_eq!(result.error_kind, Some(ModelFetchErrorKind::NotFound));
        assert!(result
            .diagnostic_hint
            .as_deref()
            .unwrap_or_default()
            .contains("请先在 Provider 中填写 gpt-images-2"));
    }

    #[test]
    fn test_provider_models_cache_key_keeps_tenant_scope() {
        let tenant_a_key = ModelRegistryService::provider_models_cache_key(
            "lime-hub",
            "https://llm.limeai.run#lime_tenant_id=tenant-a",
            Some(ApiProviderType::Openai),
        );
        let tenant_b_key = ModelRegistryService::provider_models_cache_key(
            "lime-hub",
            "https://llm.limeai.run#lime_tenant_id=tenant-b",
            Some(ApiProviderType::Openai),
        );

        assert_ne!(tenant_a_key, tenant_b_key);
    }

    #[test]
    fn test_provider_models_cache_key_shares_inferred_openai_reads() {
        let inferred_key = ModelRegistryService::provider_models_cache_key(
            "openai",
            "https://api.openai.com/v1",
            None,
        );
        let typed_key = ModelRegistryService::provider_models_cache_key(
            "openai",
            "https://api.openai.com/v1",
            Some(ApiProviderType::Openai),
        );

        assert_eq!(inferred_key, typed_key);
    }

    #[test]
    fn test_fal_cache_ignores_non_fal_models() {
        let (service, _db) = setup_cache_service();
        let now = chrono::Utc::now().timestamp();

        service
            .save_provider_models_cache(
                "fal",
                "https://fal.run/fal-ai",
                Some(ApiProviderType::Openai),
                &[create_cached_model("gpt-5.2-pro")],
                None,
                now,
            )
            .expect("cache should be saved");

        let cached = service
            .get_cached_provider_models(
                "fal",
                "https://fal.run/fal-ai",
                Some(ApiProviderType::Openai),
            )
            .expect("cache read should not fail");

        assert!(cached.is_none());
    }

    #[test]
    fn test_build_gemini_models_api_url() {
        assert_eq!(
            ModelRegistryService::build_gemini_models_api_url(
                "https://generativelanguage.googleapis.com"
            ),
            "https://generativelanguage.googleapis.com/v1beta/models"
        );
        assert_eq!(
            ModelRegistryService::build_gemini_models_api_url(
                "https://generativelanguage.googleapis.com/v1beta"
            ),
            "https://generativelanguage.googleapis.com/v1beta/models"
        );
    }

    #[test]
    fn test_build_ollama_models_api_url() {
        assert_eq!(
            ModelRegistryService::build_ollama_models_api_url("http://localhost:11434"),
            "http://127.0.0.1:11434/api/tags"
        );
        assert_eq!(
            ModelRegistryService::build_ollama_models_api_url("http://localhost:11434/api"),
            "http://127.0.0.1:11434/api/tags"
        );
        assert_eq!(
            ModelRegistryService::build_ollama_models_api_url("http://127.0.0.1:11434"),
            "http://127.0.0.1:11434/api/tags"
        );
        assert_eq!(
            ModelRegistryService::build_ollama_models_api_url("http://127.0.0.1:11434/api"),
            "http://127.0.0.1:11434/api/tags"
        );
    }

    #[test]
    fn test_should_bypass_proxy_for_models_api_url() {
        assert!(
            ModelRegistryService::should_bypass_proxy_for_models_api_url(
                "http://127.0.0.1:11434/api/tags"
            )
        );
        assert!(
            ModelRegistryService::should_bypass_proxy_for_models_api_url(
                "http://localhost:11434/api/tags"
            )
        );
        assert!(
            ModelRegistryService::should_bypass_proxy_for_models_api_url(
                "http://[::1]:11434/api/tags"
            )
        );
        assert!(
            !ModelRegistryService::should_bypass_proxy_for_models_api_url(
                "https://api.openai.com/v1/models"
            )
        );
    }

    #[test]
    fn test_build_models_api_hint_for_doubao() {
        let hint = ModelRegistryService::build_models_api_hint(
            "doubao",
            "https://ark.cn-beijing.volces.com/api/v3",
            "https://ark.cn-beijing.volces.com/api/v3/models",
        );

        assert!(hint.is_some());
        assert!(hint
            .unwrap()
            .contains("https://ark.cn-beijing.volces.com/api/v3"));
    }

    #[test]
    fn test_build_models_api_hint_for_zhipu() {
        let hint = ModelRegistryService::build_models_api_hint(
            "zhipu",
            "https://open.bigmodel.cn/api/paas/v4",
            "https://open.bigmodel.cn/api/paas/v4/models",
        );

        assert!(hint.is_some());
        assert!(hint
            .unwrap()
            .contains("https://open.bigmodel.cn/api/paas/v4"));
    }

    #[test]
    fn test_build_models_api_hint_explains_endpoint_host_normalization() {
        let hint = ModelRegistryService::build_models_api_hint(
            "custom-codex",
            "https://gateway.example.com/proxy/responses",
            "https://gateway.example.com/proxy/v1/models",
        )
        .expect("responses endpoint should produce hint");

        assert!(hint.contains("具体接口地址而不是基础地址"));
        assert!(hint.contains("https://gateway.example.com/proxy/v1/models"));
    }

    #[test]
    fn test_format_models_api_not_found_message_for_anthropic() {
        let message = ModelRegistryService::format_models_api_not_found_message(
            ModelFetchProtocol::Anthropic,
            "<html>404</html>",
        );

        assert!(message.contains("当前 Anthropic 兼容入口未提供标准 /models 接口"));
        assert!(!message.contains("<html>"));
        assert!(!message.contains("404 Not Found"));
        assert!(!message.contains("Base URL 路径不兼容"));
    }

    #[test]
    fn test_format_models_api_not_found_message_summarizes_html_body_for_non_anthropic() {
        let message = ModelRegistryService::format_models_api_not_found_message(
            ModelFetchProtocol::OpenAiCompatible,
            "<html><head><title>404 Not Found</title></head><body>oops</body></html>",
        );

        assert!(message.contains("API 返回错误 404 Not Found: 上游返回了 HTML 错误页。"));
        assert!(!message.contains("<html>"));
        assert!(message.contains("Base URL 路径不兼容"));
    }

    #[test]
    fn test_prepare_model_fetch_request_adds_dual_auth_for_anthropic_compatible_host() {
        let request = ModelRegistryService::prepare_model_fetch_request(
            "compatible-test",
            "https://api.minimaxi.com/anthropic",
            "test-key",
            Some(ApiProviderType::AnthropicCompatible),
        )
        .expect("anthropic-compatible request should be prepared");

        assert_eq!(request.protocol, ModelFetchProtocol::Anthropic);
        assert_eq!(request.url, "https://api.minimaxi.com/anthropic/v1/models");
        assert!(request
            .headers
            .iter()
            .any(|(name, value)| name == "Authorization" && value == "Bearer test-key"));
        assert!(request
            .headers
            .iter()
            .any(|(name, value)| name == "x-api-key" && value == "test-key"));
    }

    #[test]
    fn test_prepare_model_fetch_request_keeps_x_api_key_for_official_anthropic() {
        let request = ModelRegistryService::prepare_model_fetch_request(
            "anthropic",
            "https://api.anthropic.com",
            "test-key",
            Some(ApiProviderType::Anthropic),
        )
        .expect("request should be prepared");

        assert!(request
            .headers
            .iter()
            .any(|(name, value)| name == "x-api-key" && value == "test-key"));
        assert!(!request
            .headers
            .iter()
            .any(|(name, _)| name == "Authorization"));
    }

    #[test]
    fn test_requires_api_key_for_model_fetch() {
        assert!(ModelRegistryService::requires_api_key_for_model_fetch(
            "openai",
            "https://api.openai.com",
            ApiProviderType::Openai
        ));
        assert!(!ModelRegistryService::requires_api_key_for_model_fetch(
            "ollama",
            "http://127.0.0.1:11434",
            ApiProviderType::Ollama
        ));
        assert!(!ModelRegistryService::requires_api_key_for_model_fetch(
            "lmstudio",
            "http://127.0.0.1:1234/v1",
            ApiProviderType::Openai
        ));
        assert!(!ModelRegistryService::requires_api_key_for_model_fetch(
            "lime-hub",
            "https://llm.limeai.run#lime_tenant_id=tenant-0001",
            ApiProviderType::Openai
        ));
        assert!(ModelRegistryService::requires_api_key_for_model_fetch(
            "lime-hub",
            "https://llm.limeai.run",
            ApiProviderType::Openai
        ));
    }

    #[test]
    fn test_resolve_model_fetch_protocol() {
        assert_eq!(
            ModelRegistryService::resolve_model_fetch_protocol(
                "google",
                "https://generativelanguage.googleapis.com",
                Some(ApiProviderType::Gemini)
            ),
            ModelFetchProtocol::Gemini
        );
        assert_eq!(
            ModelRegistryService::resolve_model_fetch_protocol(
                "anthropic",
                "https://api.anthropic.com",
                Some(ApiProviderType::Anthropic)
            ),
            ModelFetchProtocol::Anthropic
        );
        assert_eq!(
            ModelRegistryService::resolve_model_fetch_protocol(
                "azure-openai",
                "https://example.openai.azure.com",
                Some(ApiProviderType::AzureOpenai)
            ),
            ModelFetchProtocol::Unsupported
        );
        assert_eq!(
            ModelRegistryService::resolve_model_fetch_protocol(
                "custom-yls-images",
                "https://gateway.example.com/codex",
                Some(ApiProviderType::Openai)
            ),
            ModelFetchProtocol::ResponsesCompatible
        );
        assert_eq!(
            ModelRegistryService::resolve_model_fetch_protocol(
                "codex",
                "https://chatgpt.com/backend-api/codex",
                Some(ApiProviderType::Codex)
            ),
            ModelFetchProtocol::ResponsesCompatible
        );
    }

    #[test]
    fn test_fal_like_provider_does_not_require_key_before_cache_or_declared_models() {
        assert!(!ModelRegistryService::requires_api_key_for_model_fetch(
            "fal",
            "https://fal.run/fal-ai",
            ApiProviderType::Openai
        ));
    }

    #[test]
    fn test_parse_anthropic_models_response() {
        let response = ModelRegistryService::parse_anthropic_models_response(
            r#"{
              "data": [
                { "id": "claude-sonnet-4-5", "display_name": "Claude Sonnet 4.5" }
              ],
              "has_more": false,
              "last_id": "claude-sonnet-4-5"
            }"#,
        )
        .expect("parse anthropic response");

        assert_eq!(response.models.len(), 1);
        assert_eq!(response.models[0].id, "claude-sonnet-4-5");
        assert_eq!(
            response.models[0].display_name.as_deref(),
            Some("Claude Sonnet 4.5")
        );
    }

    #[test]
    fn test_parse_gemini_models_response() {
        let response = ModelRegistryService::parse_gemini_models_response(
            r#"{
              "models": [
                {
                  "name": "models/gemini-2.5-pro",
                  "displayName": "Gemini 2.5 Pro",
                  "inputTokenLimit": 1048576,
                  "supportedGenerationMethods": ["generateContent"]
                },
                {
                  "name": "models/text-embedding-004",
                  "displayName": "Embedding",
                  "supportedGenerationMethods": ["embedContent"]
                }
              ],
              "nextPageToken": "next-page"
            }"#,
        )
        .expect("parse gemini response");

        assert_eq!(response.models.len(), 1);
        assert_eq!(response.models[0].id, "gemini-2.5-pro");
        assert_eq!(response.next_page_token.as_deref(), Some("next-page"));
    }

    #[test]
    fn test_parse_ollama_models_response() {
        let models = ModelRegistryService::parse_ollama_models_response(
            r#"{
              "models": [
                {
                  "name": "qwen3:14b",
                  "details": { "family": "qwen3" }
                }
              ]
            }"#,
        )
        .expect("parse ollama response");

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "qwen3:14b");
        assert_eq!(models[0].family.as_deref(), Some("qwen3"));
    }
}
