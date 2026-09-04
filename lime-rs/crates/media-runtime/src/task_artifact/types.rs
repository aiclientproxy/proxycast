use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    ImageGenerate,
    AudioGenerate,
    CoverGenerate,
    VideoGenerate,
    TranscriptionGenerate,
    BroadcastGenerate,
    UrlParse,
    Typesetting,
    ModalResourceSearch,
}

impl TaskType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ImageGenerate => "image_generate",
            Self::AudioGenerate => "audio_generate",
            Self::CoverGenerate => "cover_generate",
            Self::VideoGenerate => "video_generate",
            Self::TranscriptionGenerate => "transcription_generate",
            Self::BroadcastGenerate => "broadcast_generate",
            Self::UrlParse => "url_parse",
            Self::Typesetting => "typesetting",
            Self::ModalResourceSearch => "modal_resource_search",
        }
    }

    pub fn command_name(self) -> &'static str {
        match self {
            Self::ImageGenerate => "image",
            Self::AudioGenerate => "audio",
            Self::CoverGenerate => "cover",
            Self::VideoGenerate => "video",
            Self::TranscriptionGenerate => "transcription",
            Self::BroadcastGenerate => "broadcast",
            Self::UrlParse => "url-parse",
            Self::Typesetting => "typesetting",
            Self::ModalResourceSearch => "resource-search",
        }
    }

    pub fn default_status(self) -> &'static str {
        match self {
            Self::VideoGenerate => "queued",
            Self::ImageGenerate
            | Self::AudioGenerate
            | Self::CoverGenerate
            | Self::TranscriptionGenerate
            | Self::BroadcastGenerate
            | Self::UrlParse
            | Self::Typesetting
            | Self::ModalResourceSearch => "pending_submit",
        }
    }

    pub fn family(self) -> &'static str {
        match self {
            Self::ImageGenerate | Self::CoverGenerate => "image",
            Self::AudioGenerate => "audio",
            Self::VideoGenerate => "video",
            Self::TranscriptionGenerate
            | Self::BroadcastGenerate
            | Self::UrlParse
            | Self::Typesetting => "document",
            Self::ModalResourceSearch => "resource",
        }
    }

    pub fn all() -> &'static [Self] {
        &[
            Self::ImageGenerate,
            Self::AudioGenerate,
            Self::CoverGenerate,
            Self::VideoGenerate,
            Self::TranscriptionGenerate,
            Self::BroadcastGenerate,
            Self::UrlParse,
            Self::Typesetting,
            Self::ModalResourceSearch,
        ]
    }
}

impl FromStr for TaskType {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "image" | "image_generate" => Ok(Self::ImageGenerate),
            "audio" | "audio_generate" | "voice" | "voice_generate" => Ok(Self::AudioGenerate),
            "cover" | "cover_generate" => Ok(Self::CoverGenerate),
            "video" | "video_generate" => Ok(Self::VideoGenerate),
            "transcription" | "transcribe" | "transcription_generate" => {
                Ok(Self::TranscriptionGenerate)
            }
            "broadcast" | "broadcast_generate" => Ok(Self::BroadcastGenerate),
            "url-parse" | "url_parse" | "urlparse" => Ok(Self::UrlParse),
            "typesetting" => Ok(Self::Typesetting),
            "resource-search" | "resource_search" | "modal_resource_search" | "resource" => {
                Ok(Self::ModalResourceSearch)
            }
            _ => Err(()),
        }
    }
}

pub type MediaTaskType = TaskType;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TaskErrorRecord {
    pub code: String,
    pub message: String,
    #[serde(default)]
    pub retryable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub occurred_at: Option<String>,
}

impl TaskErrorRecord {
    fn from_legacy_message(message: String) -> Option<Self> {
        let trimmed = message.trim();
        if trimmed.is_empty() {
            return None;
        }

        Some(Self {
            code: "legacy_error".to_string(),
            message: trimmed.to_string(),
            retryable: false,
            stage: None,
            provider_code: None,
            occurred_at: None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TaskRelationships {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_task_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root_task_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub depends_on_task_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub child_task_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_asset_ids: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub derived_from_attempt_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub triggered_by_skill: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub triggered_by_message_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slot_id: Option<String>,
}

impl TaskRelationships {
    fn is_empty(&self) -> bool {
        self.parent_task_id.is_none()
            && self.root_task_id.is_none()
            && self.depends_on_task_ids.is_empty()
            && self.child_task_ids.is_empty()
            && self.source_asset_ids.is_empty()
            && self.derived_from_attempt_id.is_none()
            && self.triggered_by_skill.is_none()
            && self.triggered_by_message_id.is_none()
            && self.slot_id.is_none()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TaskPreviewSlot {
    pub slot_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slot_index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shot_type: Option<String>,
    pub status: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TaskProgress {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub percent: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub preview_slots: Vec<TaskPreviewSlot>,
}

impl TaskProgress {
    pub(crate) fn is_empty(&self) -> bool {
        self.phase.is_none()
            && self.percent.is_none()
            && self.message.is_none()
            && self.preview_slots.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TaskUiHints {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub render_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub placeholder_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preferred_surface: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub open_action: Option<String>,
}

impl TaskUiHints {
    pub(crate) fn is_empty(&self) -> bool {
        self.render_mode.is_none()
            && self.placeholder_text.is_none()
            && self.preferred_surface.is_none()
            && self.open_action.is_none()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TaskAttemptMetrics {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queue_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_ms: Option<u64>,
}

impl TaskAttemptMetrics {
    pub(crate) fn is_empty(&self) -> bool {
        self.queue_ms.is_none() && self.run_ms.is_none()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct TaskAttemptRecord {
    pub attempt_id: String,
    pub attempt_index: u32,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queued_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_id: Option<String>,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub input_snapshot: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_snapshot: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<TaskErrorRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<TaskAttemptMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logs_ref: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskArtifactRecord {
    pub task_id: String,
    pub task_type: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub task_family: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    pub payload: Value,
    pub status: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub normalized_status: String,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub submitted_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cancelled_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub retry_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_task_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(
        default,
        deserialize_with = "deserialize_task_error_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub last_error: Option<TaskErrorRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_attempt_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attempts: Vec<TaskAttemptRecord>,
    #[serde(default, skip_serializing_if = "TaskRelationships::is_empty")]
    pub relationships: TaskRelationships,
    #[serde(default, skip_serializing_if = "TaskProgress::is_empty")]
    pub progress: TaskProgress,
    #[serde(default, skip_serializing_if = "TaskUiHints::is_empty")]
    pub ui_hints: TaskUiHints,
}

pub type MediaTaskArtifactRecord = TaskArtifactRecord;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskOutput {
    pub success: bool,
    pub task_id: String,
    pub task_type: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub task_family: String,
    pub status: String,
    pub normalized_status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_attempt_id: Option<String>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub attempt_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<TaskErrorRecord>,
    #[serde(default, skip_serializing_if = "TaskProgress::is_empty")]
    pub progress: TaskProgress,
    #[serde(default, skip_serializing_if = "TaskUiHints::is_empty")]
    pub ui_hints: TaskUiHints,
    pub path: String,
    pub absolute_path: String,
    pub artifact_path: String,
    pub absolute_artifact_path: String,
    #[serde(default)]
    pub reused_existing: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
    pub record: TaskArtifactRecord,
}

impl TaskOutput {
    pub fn artifact_paths(&self) -> Vec<String> {
        vec![self.path.clone()]
    }
}

pub type MediaTaskOutput = TaskOutput;

fn is_zero(value: &u32) -> bool {
    *value == 0
}

fn deserialize_task_error_opt<'de, D>(deserializer: D) -> Result<Option<TaskErrorRecord>, D::Error>
where
    D: Deserializer<'de>,
{
    let Some(value) = Option::<Value>::deserialize(deserializer)? else {
        return Ok(None);
    };

    match value {
        Value::Null => Ok(None),
        Value::String(message) => Ok(TaskErrorRecord::from_legacy_message(message)),
        other => serde_json::from_value(other)
            .map(Some)
            .map_err(serde::de::Error::custom),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskErrorOutput {
    pub success: bool,
    pub error_code: String,
    pub error_message: String,
    pub retryable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
}

impl TaskErrorOutput {
    pub fn from_error(error: &MediaRuntimeError) -> Self {
        Self {
            success: false,
            error_code: error.code().to_string(),
            error_message: error.to_string(),
            retryable: error.retryable(),
            hint: error.hint().map(ToOwned::to_owned),
            task_id: error.task_id(),
            idempotency_key: error.idempotency_key(),
        }
    }
}

pub type MediaTaskErrorOutput = TaskErrorOutput;

#[derive(Debug, Error)]
pub enum MediaRuntimeError {
    #[error("{0}")]
    InvalidParams(String),
    #[error("{0}")]
    Io(String),
    #[error("未找到任务: {task_ref}")]
    TaskNotFound { task_ref: String },
    #[error("{0}")]
    Conflict(String),
    #[error("{0}")]
    InvalidState(String),
    #[error("{0}")]
    NotRetryable(String),
}

impl MediaRuntimeError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidParams(_) => "invalid_params",
            Self::Io(_) => "io_error",
            Self::TaskNotFound { .. } => "task_not_found",
            Self::Conflict(_) => "task_conflict",
            Self::InvalidState(_) => "invalid_state",
            Self::NotRetryable(_) => "not_retryable",
        }
    }

    pub fn exit_code(&self) -> i32 {
        match self {
            Self::InvalidParams(_) => 2,
            Self::TaskNotFound { .. } => 3,
            Self::Io(_) => 4,
            Self::Conflict(_) => 5,
            Self::InvalidState(_) => 6,
            Self::NotRetryable(_) => 7,
        }
    }

    pub fn retryable(&self) -> bool {
        matches!(self, Self::Io(_))
    }

    pub fn hint(&self) -> Option<&'static str> {
        match self {
            Self::InvalidParams(_) => Some("请检查命令参数、路径和 JSON 字段是否完整。"),
            Self::Io(_) => Some("请检查工作目录、文件权限，或稍后重试。"),
            Self::TaskNotFound { .. } => Some("请检查任务标识和当前项目目录。"),
            Self::Conflict(_) => Some("请更换 `--output`，或使用稳定的 `--idempotency-key` 重试。"),
            Self::InvalidState(_) => Some("请先通过任务中心查看当前状态。"),
            Self::NotRetryable(_) => Some("只有 failed 或 cancelled 的任务可以重试。"),
        }
    }

    pub fn task_id(&self) -> Option<String> {
        match self {
            Self::TaskNotFound { task_ref } => Some(task_ref.clone()),
            _ => None,
        }
    }

    pub fn idempotency_key(&self) -> Option<String> {
        None
    }
}

#[derive(Debug, Clone, Default)]
pub struct TaskWriteOptions<'a> {
    pub status: Option<String>,
    pub output_path: Option<&'a str>,
    pub artifact_dir: Option<&'a str>,
    pub idempotency_key: Option<&'a str>,
    pub relationships: TaskRelationships,
}

#[derive(Debug, Clone, Default)]
pub struct TaskArtifactPatch {
    pub status: Option<String>,
    pub payload_patch: Option<Value>,
    pub result: Option<Option<Value>>,
    pub last_error: Option<Option<TaskErrorRecord>>,
    pub progress: Option<TaskProgress>,
    pub ui_hints: Option<TaskUiHints>,
    pub current_attempt_worker_id: Option<Option<String>>,
    pub current_attempt_metrics: Option<Option<TaskAttemptMetrics>>,
}
