use serde::{Deserialize, Serialize};

const TASK_CHAT: &str = "chat";
const TASK_REASONING: &str = "reasoning";
const TASK_VISION_UNDERSTANDING: &str = "vision_understanding";
const TASK_IMAGE_GENERATION: &str = "image_generation";
const TASK_IMAGE_EDIT: &str = "image_edit";
const TASK_SPEECH_TO_TEXT: &str = "speech_to_text";
const TASK_TEXT_TO_SPEECH: &str = "text_to_speech";
const TASK_EMBEDDING: &str = "embedding";

const MODALITY_TEXT: &str = "text";
const MODALITY_IMAGE: &str = "image";
const MODALITY_AUDIO: &str = "audio";
const MODALITY_VIDEO: &str = "video";
const MODALITY_FILE: &str = "file";
const MODALITY_EMBEDDING: &str = "embedding";

const FEATURE_STREAMING: &str = "streaming";
const FEATURE_TOOL_CALLING: &str = "tool_calling";
const FEATURE_JSON_SCHEMA: &str = "json_schema";
const FEATURE_REASONING: &str = "reasoning";
const FEATURE_PROMPT_CACHE: &str = "prompt_cache";
const FEATURE_IMAGES_API: &str = "images_api";

/// Pricing information for a model (all costs in USD per token)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pricing {
    /// Cost per prompt token
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<f64>,

    /// Cost per completion token
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion: Option<f64>,

    /// Cost per request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<f64>,

    /// Cost per image
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<f64>,
}

/// Canonical representation of a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalModel {
    /// Model identifier (e.g., "anthropic/claude-3-5-sonnet" or "openai/gpt-4o:extended")
    pub id: String,

    /// Human-readable name (e.g., "Claude 3.5 Sonnet")
    pub name: String,

    /// Maximum context window size in tokens
    pub context_length: usize,

    /// Maximum completion tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<usize>,

    /// Task families supported by the model (e.g., ["chat", "reasoning"])
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub task_families: Vec<String>,

    /// Input modalities supported (e.g., ["text", "image"])
    #[serde(default)]
    pub input_modalities: Vec<String>,

    /// Output modalities supported (e.g., ["text"])
    #[serde(default)]
    pub output_modalities: Vec<String>,

    /// Runtime features supported by the model (e.g., ["streaming", "tool_calling"])
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub runtime_features: Vec<String>,

    /// Model-selected runtime tool surface. Unknown values are ignored by consumers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_mode: Option<String>,

    /// Whether the model supports tool calling
    #[serde(default)]
    pub supports_tools: bool,

    /// Whether the model supports reasoning/thinking controls
    #[serde(default, skip_serializing_if = "is_false")]
    pub supports_reasoning: bool,

    /// Whether the model supports prompt cache controls
    #[serde(default, skip_serializing_if = "is_false")]
    pub supports_prompt_cache: bool,

    /// Pricing for this model
    pub pricing: Pricing,
}

/// Legacy-compatible capability flags used by renderer model metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalModelCapabilities {
    pub vision: bool,
    pub tools: bool,
    pub streaming: bool,
    pub json_mode: bool,
    pub function_calling: bool,
    pub reasoning: bool,
}

/// Provider-neutral capability summary for canonical model routing and UI projection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalModelCapabilitySummary {
    pub capabilities: CanonicalModelCapabilities,
    pub task_families: Vec<String>,
    pub input_modalities: Vec<String>,
    pub output_modalities: Vec<String>,
    pub runtime_features: Vec<String>,
    pub supports_tools: bool,
    pub supports_reasoning: bool,
    pub supports_prompt_cache: bool,
    pub supports_media_input: bool,
    pub supports_media_output: bool,
    pub context_length: usize,
    pub max_output_tokens: Option<usize>,
}

impl CanonicalModel {
    pub fn capability_summary(&self) -> CanonicalModelCapabilitySummary {
        let task_families = self.resolved_task_families();
        let runtime_features = self.resolved_runtime_features(&task_families);
        let supports_tools =
            self.supports_tools || contains_value(&runtime_features, FEATURE_TOOL_CALLING);
        let supports_reasoning =
            self.has_reasoning_signal() || contains_value(&task_families, TASK_REASONING);
        let supports_prompt_cache =
            self.supports_prompt_cache || contains_value(&runtime_features, FEATURE_PROMPT_CACHE);
        let supports_media_input = self
            .input_modalities
            .iter()
            .any(|value| is_media_modality(value));
        let supports_media_output = self
            .output_modalities
            .iter()
            .any(|value| is_media_modality(value));

        CanonicalModelCapabilitySummary {
            capabilities: CanonicalModelCapabilities {
                vision: self.has_input_modality(MODALITY_IMAGE)
                    && self.has_output_modality(MODALITY_TEXT),
                tools: supports_tools,
                streaming: contains_value(&runtime_features, FEATURE_STREAMING),
                json_mode: contains_value(&runtime_features, FEATURE_JSON_SCHEMA),
                function_calling: supports_tools,
                reasoning: supports_reasoning,
            },
            task_families,
            input_modalities: self.input_modalities.clone(),
            output_modalities: self.output_modalities.clone(),
            runtime_features,
            supports_tools,
            supports_reasoning,
            supports_prompt_cache,
            supports_media_input,
            supports_media_output,
            context_length: self.context_length,
            max_output_tokens: self.max_completion_tokens,
        }
    }

    fn resolved_task_families(&self) -> Vec<String> {
        if !self.task_families.is_empty() {
            return unique_values(self.task_families.clone());
        }

        let mut families = Vec::new();
        let has_text_output = self.has_output_modality(MODALITY_TEXT);
        let has_image_input = self.has_input_modality(MODALITY_IMAGE);
        let has_image_output = self.has_output_modality(MODALITY_IMAGE);
        let has_audio_input = self.has_input_modality(MODALITY_AUDIO);
        let has_audio_output = self.has_output_modality(MODALITY_AUDIO);
        let has_embedding_output = self.has_output_modality(MODALITY_EMBEDDING);

        if has_text_output {
            push_unique(&mut families, TASK_CHAT);
        }
        if has_image_input && has_text_output {
            push_unique(&mut families, TASK_VISION_UNDERSTANDING);
        }
        if has_image_output {
            push_unique(&mut families, TASK_IMAGE_GENERATION);
        }
        if has_image_input && has_image_output {
            push_unique(&mut families, TASK_IMAGE_EDIT);
        }
        if has_audio_input && has_text_output {
            push_unique(&mut families, TASK_SPEECH_TO_TEXT);
        }
        if has_audio_output {
            push_unique(&mut families, TASK_TEXT_TO_SPEECH);
        }
        if has_embedding_output {
            push_unique(&mut families, TASK_EMBEDDING);
        }
        if self.has_reasoning_signal() {
            push_unique(&mut families, TASK_REASONING);
        }

        families
    }

    fn resolved_runtime_features(&self, task_families: &[String]) -> Vec<String> {
        let mut features = unique_values(self.runtime_features.clone());

        if task_families.iter().any(|family| {
            [
                TASK_CHAT,
                TASK_REASONING,
                TASK_VISION_UNDERSTANDING,
                TASK_SPEECH_TO_TEXT,
            ]
            .iter()
            .any(|expected| family.eq_ignore_ascii_case(expected))
        }) {
            push_unique(&mut features, FEATURE_STREAMING);
        }
        if self.supports_tools {
            push_unique(&mut features, FEATURE_TOOL_CALLING);
        }
        if task_families
            .iter()
            .any(|family| family.eq_ignore_ascii_case(TASK_CHAT))
        {
            push_unique(&mut features, FEATURE_JSON_SCHEMA);
        }
        if self.has_reasoning_signal()
            || task_families
                .iter()
                .any(|family| family.eq_ignore_ascii_case(TASK_REASONING))
        {
            push_unique(&mut features, FEATURE_REASONING);
        }
        if self.supports_prompt_cache {
            push_unique(&mut features, FEATURE_PROMPT_CACHE);
        }
        if task_families.iter().any(|family| {
            [TASK_IMAGE_GENERATION, TASK_IMAGE_EDIT]
                .iter()
                .any(|expected| family.eq_ignore_ascii_case(expected))
        }) {
            push_unique(&mut features, FEATURE_IMAGES_API);
        }

        features
    }

    fn has_reasoning_signal(&self) -> bool {
        self.supports_reasoning
            || contains_value(&self.runtime_features, FEATURE_REASONING)
            || contains_reasoning_marker(&self.id)
            || contains_reasoning_marker(&self.name)
    }

    fn has_input_modality(&self, modality: &str) -> bool {
        contains_value(&self.input_modalities, modality)
    }

    fn has_output_modality(&self, modality: &str) -> bool {
        contains_value(&self.output_modalities, modality)
    }
}

fn contains_value(values: &[String], expected: &str) -> bool {
    values
        .iter()
        .any(|value| value.eq_ignore_ascii_case(expected))
}

fn contains_reasoning_marker(value: &str) -> bool {
    let normalized = value.to_ascii_lowercase();
    normalized.contains("reasoning") || normalized.contains("thinking")
}

fn is_media_modality(value: &str) -> bool {
    [
        MODALITY_IMAGE,
        MODALITY_AUDIO,
        MODALITY_VIDEO,
        MODALITY_FILE,
    ]
    .iter()
    .any(|expected| value.eq_ignore_ascii_case(expected))
}

fn unique_values(values: Vec<String>) -> Vec<String> {
    let mut unique = Vec::new();
    for value in values {
        if !unique.iter().any(|existing: &String| existing == &value) {
            unique.push(value);
        }
    }
    unique
}

fn push_unique(values: &mut Vec<String>, value: &str) {
    if !contains_value(values, value) {
        values.push(value.to_string());
    }
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pricing() -> Pricing {
        Pricing {
            prompt: Some(0.000003),
            completion: Some(0.000015),
            request: Some(0.0),
            image: Some(0.0),
        }
    }

    #[test]
    fn capability_summary_infers_frontend_aligned_fields_from_legacy_shape() {
        let model = CanonicalModel {
            id: "anthropic/claude-3.7-sonnet:thinking".to_string(),
            name: "Anthropic: Claude 3.7 Sonnet (thinking)".to_string(),
            context_length: 200_000,
            max_completion_tokens: Some(64_000),
            task_families: Vec::new(),
            input_modalities: vec!["text".to_string(), "image".to_string(), "file".to_string()],
            output_modalities: vec!["text".to_string()],
            runtime_features: Vec::new(),
            tool_mode: None,
            supports_tools: true,
            supports_reasoning: false,
            supports_prompt_cache: false,
            pricing: pricing(),
        };

        let summary = model.capability_summary();

        assert!(summary.supports_tools);
        assert!(summary.supports_reasoning);
        assert!(!summary.supports_prompt_cache);
        assert!(summary.supports_media_input);
        assert!(!summary.supports_media_output);
        assert_eq!(summary.context_length, 200_000);
        assert_eq!(summary.max_output_tokens, Some(64_000));
        assert!(contains_value(&summary.task_families, TASK_CHAT));
        assert!(contains_value(
            &summary.task_families,
            TASK_VISION_UNDERSTANDING
        ));
        assert!(contains_value(&summary.task_families, TASK_REASONING));
        assert!(contains_value(&summary.runtime_features, FEATURE_STREAMING));
        assert!(contains_value(
            &summary.runtime_features,
            FEATURE_TOOL_CALLING
        ));
        assert!(contains_value(&summary.runtime_features, FEATURE_REASONING));
        assert_eq!(
            summary.capabilities,
            CanonicalModelCapabilities {
                vision: true,
                tools: true,
                streaming: true,
                json_mode: true,
                function_calling: true,
                reasoning: true,
            }
        );
    }

    #[test]
    fn capability_summary_preserves_explicit_media_output_and_cache_features() {
        let model = CanonicalModel {
            id: "openai/gpt-image-1".to_string(),
            name: "OpenAI: GPT Image 1".to_string(),
            context_length: 128_000,
            max_completion_tokens: None,
            task_families: vec!["image_generation".to_string()],
            input_modalities: vec!["text".to_string()],
            output_modalities: vec!["image".to_string()],
            runtime_features: vec!["images_api".to_string()],
            tool_mode: None,
            supports_tools: false,
            supports_reasoning: false,
            supports_prompt_cache: true,
            pricing: pricing(),
        };

        let summary = model.capability_summary();

        assert_eq!(summary.task_families, vec!["image_generation"]);
        assert_eq!(summary.runtime_features, vec!["images_api", "prompt_cache"]);
        assert!(!summary.supports_tools);
        assert!(!summary.supports_reasoning);
        assert!(summary.supports_prompt_cache);
        assert!(!summary.supports_media_input);
        assert!(summary.supports_media_output);
        assert_eq!(summary.capabilities.vision, false);
        assert_eq!(summary.capabilities.streaming, false);
        assert_eq!(summary.capabilities.json_mode, false);
    }

    #[test]
    fn legacy_json_deserializes_without_explicit_summary_fields() {
        let model: CanonicalModel = serde_json::from_str(
            r#"{
                "id": "provider/chat",
                "name": "Provider Chat",
                "context_length": 8192,
                "max_completion_tokens": 2048,
                "input_modalities": ["text"],
                "output_modalities": ["text"],
                "supports_tools": true,
                "pricing": {
                    "prompt": 0.000001,
                    "completion": 0.000002,
                    "request": 0.0,
                    "image": 0.0
                }
            }"#,
        )
        .expect("legacy canonical model JSON should remain valid");

        assert!(model.task_families.is_empty());
        assert!(model.runtime_features.is_empty());
        assert!(!model.supports_reasoning);
        assert!(!model.supports_prompt_cache);
        assert!(model.capability_summary().supports_tools);
    }
}
