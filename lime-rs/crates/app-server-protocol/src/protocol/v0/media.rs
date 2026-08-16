use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ImageStoryboardSlotInput {
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "slot_id")]
    pub slot_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "shot_type")]
    pub shot_type: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MediaTaskArtifactImageCreateParams {
    pub project_root_path: String,
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "title_generation_result"
    )]
    pub title_generation_result: Option<serde_json::Value>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "persona_context"
    )]
    pub persona_context: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presentation: Option<serde_json::Value>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "taste_context"
    )]
    pub taste_context: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "raw_text")]
    pub raw_text: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "layout_hint"
    )]
    pub layout_hint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "aspect_ratio"
    )]
    pub aspect_ratio: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "provider_id"
    )]
    pub provider_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "executor_mode"
    )]
    pub executor_mode: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "outer_model"
    )]
    pub outer_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "session_id")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "thread_id")]
    pub thread_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "turn_id")]
    pub turn_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "project_id")]
    pub project_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "content_id")]
    pub content_id: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "entry_source"
    )]
    pub entry_source: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "modality_contract_key"
    )]
    pub modality_contract_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modality: Option<String>,
    #[serde(default, alias = "required_capabilities")]
    pub required_capabilities: Vec<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "routing_slot"
    )]
    pub routing_slot: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "runtime_contract"
    )]
    pub runtime_contract: Option<serde_json::Value>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "requested_target"
    )]
    pub requested_target: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "slot_id")]
    pub slot_id: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "anchor_hint"
    )]
    pub anchor_hint: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "anchor_section_title"
    )]
    pub anchor_section_title: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "anchor_text"
    )]
    pub anchor_text: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "target_output_id"
    )]
    pub target_output_id: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "target_output_ref_id"
    )]
    pub target_output_ref_id: Option<String>,
    #[serde(default, alias = "reference_images")]
    pub reference_images: Vec<String>,
    #[serde(default, alias = "storyboard_slots")]
    pub storyboard_slots: Vec<ImageStoryboardSlotInput>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MediaTaskArtifactAudioCreateParams {
    pub project_root_path: String,
    #[serde(alias = "source_text", alias = "prompt", alias = "text")]
    pub source_text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "raw_text")]
    pub raw_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "voice_style"
    )]
    pub voice_style: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "target_language"
    )]
    pub target_language: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "mime_type")]
    pub mime_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "audio_path")]
    pub audio_path: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "duration_ms"
    )]
    pub duration_ms: Option<u64>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "provider_id"
    )]
    pub provider_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "session_id")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "thread_id")]
    pub thread_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "turn_id")]
    pub turn_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "project_id")]
    pub project_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "content_id")]
    pub content_id: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "entry_source"
    )]
    pub entry_source: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "modality_contract_key"
    )]
    pub modality_contract_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modality: Option<String>,
    #[serde(default, alias = "required_capabilities")]
    pub required_capabilities: Vec<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "routing_slot"
    )]
    pub routing_slot: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "runtime_contract"
    )]
    pub runtime_contract: Option<serde_json::Value>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "requested_target"
    )]
    pub requested_target: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "output_path"
    )]
    pub output_path: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MediaTaskArtifactTranscriptionCreateParams {
    pub project_root_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "raw_text")]
    pub raw_text: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "source_path"
    )]
    pub source_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "source_url")]
    pub source_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "output_format"
    )]
    pub output_format: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "speaker_labels"
    )]
    pub speaker_labels: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamps: Option<bool>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "provider_id"
    )]
    pub provider_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "session_id")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "thread_id")]
    pub thread_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "turn_id")]
    pub turn_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "project_id")]
    pub project_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "content_id")]
    pub content_id: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "entry_source"
    )]
    pub entry_source: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "modality_contract_key"
    )]
    pub modality_contract_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modality: Option<String>,
    #[serde(default, alias = "required_capabilities")]
    pub required_capabilities: Vec<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "routing_slot"
    )]
    pub routing_slot: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "runtime_contract"
    )]
    pub runtime_contract: Option<serde_json::Value>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "requested_target"
    )]
    pub requested_target: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "output_path"
    )]
    pub output_path: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MediaTaskArtifactVideoCreateParams {
    pub project_root_path: String,
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "raw_text")]
    pub raw_text: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "aspect_ratio"
    )]
    pub aspect_ratio: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "image_url")]
    pub image_url: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "end_image_url"
    )]
    pub end_image_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "generate_audio"
    )]
    pub generate_audio: Option<bool>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "camera_fixed"
    )]
    pub camera_fixed: Option<bool>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "provider_id"
    )]
    pub provider_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "session_id")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "thread_id")]
    pub thread_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "turn_id")]
    pub turn_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "project_id")]
    pub project_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "content_id")]
    pub content_id: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "entry_source"
    )]
    pub entry_source: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "modality_contract_key"
    )]
    pub modality_contract_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modality: Option<String>,
    #[serde(default, alias = "required_capabilities")]
    pub required_capabilities: Vec<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "routing_slot"
    )]
    pub routing_slot: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "runtime_contract"
    )]
    pub runtime_contract: Option<serde_json::Value>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "requested_target"
    )]
    pub requested_target: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "output_path"
    )]
    pub output_path: Option<String>,
}
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MediaTaskArtifactCompletedImageInput {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "revised_prompt"
    )]
    pub revised_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "provider_id"
    )]
    pub provider_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "slot_id")]
    pub slot_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "slot_index")]
    pub slot_index: Option<u32>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "slot_prompt"
    )]
    pub slot_prompt: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MediaTaskArtifactImageCompleteParams {
    pub project_root_path: String,
    pub task_ref: String,
    #[serde(default, alias = "images")]
    pub images: Vec<MediaTaskArtifactCompletedImageInput>,
    #[serde(default, alias = "responses")]
    pub responses: Vec<serde_json::Value>,
    #[serde(default, alias = "failures")]
    pub failures: Vec<serde_json::Value>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "provider_id"
    )]
    pub provider_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "executor_mode"
    )]
    pub executor_mode: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "response_id"
    )]
    pub response_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MediaTaskArtifactAudioCompleteParams {
    pub project_root_path: String,
    pub task_ref: String,
    #[serde(alias = "audio_path", alias = "audio_url")]
    pub audio_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "mime_type")]
    pub mime_type: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "duration_ms"
    )]
    pub duration_ms: Option<u64>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "provider_id"
    )]
    pub provider_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MediaTaskArtifactLookupParams {
    pub project_root_path: String,
    pub task_ref: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MediaTaskArtifactListParams {
    pub project_root_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "task_family"
    )]
    pub task_family: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "task_type")]
    pub task_type: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "modality_contract_key"
    )]
    pub modality_contract_key: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "routing_outcome"
    )]
    pub routing_outcome: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct MediaTaskArtifactListFilters {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_family: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modality_contract_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing_outcome: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct MediaTaskArtifactResponse {
    pub success: bool,
    pub task_id: String,
    pub task_type: String,
    #[serde(default)]
    pub task_family: String,
    pub status: String,
    pub normalized_status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_attempt_id: Option<String>,
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub absolute_path: String,
    #[serde(default)]
    pub artifact_path: String,
    #[serde(default)]
    pub absolute_artifact_path: String,
    #[serde(default)]
    pub reused_existing: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
    #[serde(default)]
    pub record: serde_json::Value,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct MediaTaskArtifactListResponse {
    pub success: bool,
    pub workspace_root: String,
    pub artifact_root: String,
    pub filters: MediaTaskArtifactListFilters,
    pub total: usize,
    #[serde(default)]
    pub modality_runtime_contracts: serde_json::Value,
    #[serde(default)]
    pub tasks: Vec<MediaTaskArtifactResponse>,
}
