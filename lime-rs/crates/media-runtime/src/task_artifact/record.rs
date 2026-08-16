use std::path::{Component, Path, PathBuf};

use chrono::Utc;
use serde_json::{Map, Value};
use uuid::Uuid;

use super::types::{
    MediaRuntimeError, TaskArtifactRecord, TaskAttemptMetrics, TaskAttemptRecord, TaskErrorRecord,
    TaskProgress, TaskType, TaskUiHints,
};
use super::DEFAULT_ARTIFACT_ROOT;

fn is_safe_relative_path(path: &Path) -> bool {
    if path.is_absolute() {
        return false;
    }

    !path.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    })
}

fn normalize_relative_path(raw: &str, field_name: &str) -> Result<PathBuf, MediaRuntimeError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(MediaRuntimeError::InvalidParams(format!(
            "{field_name} 不能为空字符串"
        )));
    }

    let candidate = PathBuf::from(trimmed);
    if !is_safe_relative_path(&candidate) {
        return Err(MediaRuntimeError::InvalidParams(format!(
            "{field_name} 必须是安全的相对路径，且不能包含 '..'"
        )));
    }

    Ok(candidate)
}

pub(super) fn normalize_optional_text(value: Option<String>) -> Option<String> {
    value.and_then(|raw| {
        let trimmed = raw.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    })
}

pub(super) fn apply_payload_patch(
    payload: &mut Value,
    patch: Value,
) -> Result<(), MediaRuntimeError> {
    if !payload.is_object() {
        return Err(MediaRuntimeError::InvalidState(
            "任务 payload 必须是 JSON object 才能应用 patch".to_string(),
        ));
    }
    let Value::Object(patch) = patch else {
        return Err(MediaRuntimeError::InvalidParams(
            "payloadPatch 必须是 JSON object".to_string(),
        ));
    };

    let target = payload
        .as_object_mut()
        .expect("payload object was checked above");
    merge_payload_object(target, patch);
    Ok(())
}

fn merge_payload_object(target: &mut Map<String, Value>, patch: Map<String, Value>) {
    for (key, patch_value) in patch {
        let merged = match (target.remove(&key), patch_value) {
            (Some(Value::Object(mut target_object)), Value::Object(patch_object)) => {
                merge_payload_object(&mut target_object, patch_object);
                Value::Object(target_object)
            }
            (_, patch_value) => patch_value,
        };
        target.insert(key, merged);
    }
}

pub(super) fn normalize_idempotency_key(
    raw: Option<&str>,
) -> Result<Option<String>, MediaRuntimeError> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(MediaRuntimeError::InvalidParams(
            "idempotencyKey 不能为空字符串".to_string(),
        ));
    }
    Ok(Some(trimmed.to_string()))
}

pub(super) fn resolve_artifact_root_relative_path(
    artifact_dir: Option<&str>,
) -> Result<PathBuf, MediaRuntimeError> {
    match artifact_dir {
        Some(raw) => normalize_relative_path(raw, "artifactDir"),
        None => Ok(PathBuf::from(DEFAULT_ARTIFACT_ROOT)),
    }
}

pub(super) fn resolve_output_relative_path(
    task_type: TaskType,
    output_path: Option<&str>,
    artifact_dir: Option<&str>,
) -> Result<PathBuf, MediaRuntimeError> {
    if let Some(raw) = output_path {
        return normalize_relative_path(raw, "output");
    }

    let artifact_root = resolve_artifact_root_relative_path(artifact_dir)?;
    let timestamp = Utc::now().format("%Y%m%d-%H%M%S").to_string();
    let suffix = Uuid::new_v4().simple().to_string();
    Ok(artifact_root
        .join(task_type.as_str())
        .join(format!("{timestamp}-{suffix}.json")))
}

fn task_family_for_type(task_type: &str) -> String {
    task_type
        .parse::<TaskType>()
        .ok()
        .map(|value| value.family().to_string())
        .unwrap_or_else(|| match task_type.trim().to_ascii_lowercase().as_str() {
            value if value.contains("image") || value.contains("cover") => "image".to_string(),
            value if value.contains("audio") || value.contains("voice") => "audio".to_string(),
            value if value.contains("video") => "video".to_string(),
            value if value.contains("resource") => "resource".to_string(),
            "transcription_generate" | "broadcast_generate" | "url_parse" | "typesetting" => {
                "document".to_string()
            }
            _ => "automation".to_string(),
        })
}

fn payload_string(payload: &Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        payload
            .get(*key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
    })
}

fn summarize_text(raw: &str, limit: usize) -> String {
    let total = raw.chars().count();
    if total <= limit {
        return raw.to_string();
    }

    let summary: String = raw.chars().take(limit).collect();
    format!("{summary}...")
}

pub(super) fn derive_task_summary(
    task_type: &str,
    title: Option<&str>,
    payload: &Value,
) -> Option<String> {
    title
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| {
            let candidate = match task_type {
                "image_generate" | "cover_generate" | "video_generate" => {
                    payload_string(payload, &["prompt", "usage"])
                }
                "audio_generate" => {
                    payload_string(payload, &["source_text", "prompt", "voice", "voice_style"])
                }
                "transcription_generate" => {
                    payload_string(payload, &["prompt", "source_path", "source_url"])
                }
                "broadcast_generate" | "typesetting" => {
                    payload_string(payload, &["content", "targetPlatform"])
                }
                "url_parse" => payload_string(payload, &["summary", "url"]),
                "modal_resource_search" => payload_string(payload, &["query", "usage"]),
                _ => payload_string(payload, &["prompt", "query", "content", "summary"]),
            }?;
            Some(summarize_text(&candidate, 48))
        })
}

fn derive_task_ui_hints(task_family: &str, summary: Option<&str>) -> TaskUiHints {
    match task_family {
        "image" => TaskUiHints {
            render_mode: Some("media_placeholder_card".to_string()),
            placeholder_text: Some(format!("[img:{}]", summary.unwrap_or("图片任务"))),
            preferred_surface: Some("claw_chat".to_string()),
            open_action: Some("open_image_workbench".to_string()),
        },
        "video" => TaskUiHints {
            render_mode: Some("media_placeholder_card".to_string()),
            placeholder_text: Some(format!("[video:{}]", summary.unwrap_or("视频任务"))),
            preferred_surface: Some("claw_chat".to_string()),
            open_action: Some("open_video_workbench".to_string()),
        },
        "audio" => TaskUiHints {
            render_mode: Some("media_placeholder_card".to_string()),
            placeholder_text: Some(format!("[audio:{}]", summary.unwrap_or("音频任务"))),
            preferred_surface: Some("claw_chat".to_string()),
            open_action: Some("open_audio_player".to_string()),
        },
        _ => TaskUiHints {
            render_mode: Some("task_status_card".to_string()),
            placeholder_text: None,
            preferred_surface: Some("task_panel".to_string()),
            open_action: Some("open_task_panel".to_string()),
        },
    }
}

pub(super) fn derive_task_progress(
    status: &str,
    last_error: Option<&TaskErrorRecord>,
) -> TaskProgress {
    let normalized_status = normalize_status(status);
    match normalized_status.as_str() {
        "pending" => TaskProgress {
            phase: Some("pending_submit".to_string()),
            percent: Some(0),
            message: Some("任务已创建，等待进入队列".to_string()),
            preview_slots: Vec::new(),
        },
        "queued" => TaskProgress {
            phase: Some("queued".to_string()),
            percent: Some(0),
            message: Some("任务已进入队列".to_string()),
            preview_slots: Vec::new(),
        },
        "running" => TaskProgress {
            phase: Some("running".to_string()),
            percent: None,
            message: Some("任务执行中".to_string()),
            preview_slots: Vec::new(),
        },
        "partial" => TaskProgress {
            phase: Some("partial".to_string()),
            percent: None,
            message: Some("任务已返回部分结果".to_string()),
            preview_slots: Vec::new(),
        },
        "succeeded" => TaskProgress {
            phase: Some("succeeded".to_string()),
            percent: Some(100),
            message: Some("任务已完成".to_string()),
            preview_slots: Vec::new(),
        },
        "failed" => TaskProgress {
            phase: Some("failed".to_string()),
            percent: None,
            message: Some(
                last_error
                    .map(|value| value.message.clone())
                    .unwrap_or_else(|| "任务执行失败".to_string()),
            ),
            preview_slots: Vec::new(),
        },
        "cancelled" => TaskProgress {
            phase: Some("cancelled".to_string()),
            percent: None,
            message: Some("任务已取消".to_string()),
            preview_slots: Vec::new(),
        },
        _ => TaskProgress::default(),
    }
}

fn infer_attempt_provider(payload: &Value) -> Option<String> {
    payload_string(payload, &["provider", "providerId"])
}

fn infer_attempt_model(payload: &Value) -> Option<String> {
    payload_string(payload, &["model"])
}

pub(super) fn new_attempt_id() -> String {
    format!("attempt_{}", Uuid::new_v4().simple())
}

fn legacy_attempt_id(task_id: &str, attempt_index: u32) -> String {
    format!("{task_id}:attempt:{attempt_index}")
}

fn fallback_attempt_index(record: &TaskArtifactRecord) -> u32 {
    record.retry_count.saturating_add(1).max(1)
}

pub(super) struct AttemptRecordInput {
    pub(super) attempt_id: String,
    pub(super) attempt_index: u32,
    pub(super) status: String,
    pub(super) queued_at: Option<String>,
    pub(super) started_at: Option<String>,
    pub(super) completed_at: Option<String>,
    pub(super) result_snapshot: Option<Value>,
    pub(super) error: Option<TaskErrorRecord>,
}

pub(super) fn build_attempt_record(
    task_id: &str,
    payload: &Value,
    input: AttemptRecordInput,
) -> TaskAttemptRecord {
    let AttemptRecordInput {
        attempt_id,
        attempt_index,
        status,
        queued_at,
        started_at,
        completed_at,
        result_snapshot,
        error,
    } = input;

    TaskAttemptRecord {
        attempt_id,
        attempt_index,
        status,
        queued_at,
        started_at,
        completed_at,
        provider: infer_attempt_provider(payload),
        model: infer_attempt_model(payload),
        worker_id: None,
        input_snapshot: payload.clone(),
        result_snapshot,
        error,
        metrics: None,
        logs_ref: Some(format!(
            ".lime/task-logs/{task_id}/attempt_{attempt_index}.jsonl"
        )),
    }
}

pub(super) fn current_attempt_index(record: &TaskArtifactRecord) -> Option<usize> {
    if record.attempts.is_empty() {
        return None;
    }

    record
        .current_attempt_id
        .as_deref()
        .and_then(|attempt_id| {
            record
                .attempts
                .iter()
                .position(|attempt| attempt.attempt_id == attempt_id)
        })
        .or_else(|| record.attempts.len().checked_sub(1))
}

pub(super) fn canonicalize_task_record(mut record: TaskArtifactRecord) -> TaskArtifactRecord {
    record.title = normalize_optional_text(record.title);
    record.summary = normalize_optional_text(record.summary);
    record.task_family = if record.task_family.trim().is_empty() {
        task_family_for_type(&record.task_type)
    } else {
        record.task_family.trim().to_string()
    };
    record.normalized_status = normalize_status(&record.status);
    if record.summary.is_none() {
        record.summary =
            derive_task_summary(&record.task_type, record.title.as_deref(), &record.payload);
    }

    if record.attempts.is_empty() {
        let attempt_index = fallback_attempt_index(&record);
        let anchor_time = record
            .updated_at
            .clone()
            .unwrap_or_else(|| record.created_at.clone());
        let normalized_status = record.normalized_status.clone();
        let queued_at = matches!(
            normalized_status.as_str(),
            "queued" | "running" | "partial" | "succeeded" | "failed" | "cancelled"
        )
        .then(|| {
            record
                .submitted_at
                .clone()
                .unwrap_or_else(|| anchor_time.clone())
        });
        let started_at = matches!(
            normalized_status.as_str(),
            "running" | "partial" | "succeeded" | "failed" | "cancelled"
        )
        .then(|| {
            record
                .started_at
                .clone()
                .unwrap_or_else(|| anchor_time.clone())
        });
        let completed_at = matches!(
            normalized_status.as_str(),
            "partial" | "succeeded" | "failed" | "cancelled"
        )
        .then(|| {
            record
                .completed_at
                .clone()
                .or_else(|| record.cancelled_at.clone())
                .unwrap_or_else(|| anchor_time.clone())
        });

        record.attempts.push(build_attempt_record(
            &record.task_id,
            &record.payload,
            AttemptRecordInput {
                attempt_id: legacy_attempt_id(&record.task_id, attempt_index),
                attempt_index,
                status: record.status.clone(),
                queued_at,
                started_at,
                completed_at,
                result_snapshot: record.result.clone(),
                error: record.last_error.clone(),
            },
        ));
    }

    for (index, attempt) in record.attempts.iter_mut().enumerate() {
        if attempt.attempt_index == 0 {
            attempt.attempt_index = index as u32 + 1;
        }
        if attempt.attempt_id.trim().is_empty() {
            attempt.attempt_id = legacy_attempt_id(&record.task_id, attempt.attempt_index);
        }
        if attempt.status.trim().is_empty() {
            attempt.status = record.status.clone();
        } else if let Ok(status) = normalize_mutation_status(&attempt.status) {
            attempt.status = status;
        }
        if attempt.input_snapshot.is_null() {
            attempt.input_snapshot = record.payload.clone();
        }
        if attempt.provider.is_none() {
            attempt.provider = infer_attempt_provider(&attempt.input_snapshot);
        }
        if attempt.model.is_none() {
            attempt.model = infer_attempt_model(&attempt.input_snapshot);
        }
        if attempt.logs_ref.is_none() {
            attempt.logs_ref = Some(format!(
                ".lime/task-logs/{}/attempt_{}.jsonl",
                record.task_id, attempt.attempt_index
            ));
        }
        if attempt
            .metrics
            .as_ref()
            .is_some_and(TaskAttemptMetrics::is_empty)
        {
            attempt.metrics = None;
        }
    }

    if let Some(index) = current_attempt_index(&record) {
        let current_attempt = &mut record.attempts[index];
        if record.normalized_status == "failed" && current_attempt.error.is_none() {
            current_attempt.error = record.last_error.clone();
        }
        if matches!(record.normalized_status.as_str(), "partial" | "succeeded")
            && current_attempt.result_snapshot.is_none()
        {
            current_attempt.result_snapshot = record.result.clone();
        }
        record.current_attempt_id = Some(current_attempt.attempt_id.clone());
    } else {
        record.current_attempt_id = None;
    }

    record.retry_count = record.attempts.len().saturating_sub(1) as u32;
    if record.progress.is_empty() {
        record.progress = derive_task_progress(&record.status, record.last_error.as_ref());
    }
    if record.ui_hints.is_empty() {
        record.ui_hints = derive_task_ui_hints(&record.task_family, record.summary.as_deref());
    }

    record
}

pub(super) fn normalize_status(status: &str) -> String {
    match status.trim().to_ascii_lowercase().as_str() {
        "pending" | "pending_submit" => "pending".to_string(),
        "queued" => "queued".to_string(),
        "running" | "processing" | "in_progress" => "running".to_string(),
        "partial" => "partial".to_string(),
        "completed" | "success" | "succeeded" => "succeeded".to_string(),
        "failed" | "error" => "failed".to_string(),
        "cancelled" | "canceled" => "cancelled".to_string(),
        other => other.to_string(),
    }
}

pub(super) fn normalize_mutation_status(status: &str) -> Result<String, MediaRuntimeError> {
    let normalized = status.trim().to_ascii_lowercase();
    let resolved = match normalized.as_str() {
        "pending" | "pending_submit" => "pending_submit",
        "queued" => "queued",
        "running" => "running",
        "partial" => "partial",
        "succeeded" | "completed" | "success" => "succeeded",
        "failed" | "error" => "failed",
        "cancelled" | "canceled" => "cancelled",
        _ => {
            return Err(MediaRuntimeError::InvalidParams(format!(
                "不支持的任务状态: {status}"
            )));
        }
    };
    Ok(resolved.to_string())
}

pub(crate) fn read_payload_string(payload: &Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        payload
            .get(*key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
    })
}
