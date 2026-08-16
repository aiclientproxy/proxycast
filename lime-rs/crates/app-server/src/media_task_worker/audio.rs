use super::{workspace_root_from_task, MediaTaskWorkerContext};
use app_server_protocol::{MediaTaskArtifactAudioCompleteParams, MediaTaskArtifactResponse};
use chrono::Utc;
use lime_media_runtime::{
    load_task_output, patch_task_artifact, MediaTaskOutput, MediaTaskType, TaskArtifactPatch,
    TaskErrorRecord, TaskProgress,
};
use model_provider::audio::{execute_speech_generation, AudioProviderError, SpeechProviderRequest};
use serde_json::{json, Value};
use std::path::{Component, Path, PathBuf};
use tokio::task::JoinHandle;

pub(super) const AUDIO_TASK_RUNNER_WORKER_ID: &str = "lime-audio-provider-worker";

pub(super) fn should_execute_created_audio_task(task: &MediaTaskArtifactResponse) -> bool {
    task.task_type == MediaTaskType::AudioGenerate.as_str()
        && !task.reused_existing
        && matches!(
            task.normalized_status.as_str(),
            "pending" | "pending_submit" | "queued" | "running"
        )
}

pub(super) fn should_execute_pending_audio_task(task: &MediaTaskArtifactResponse) -> bool {
    task.task_type == MediaTaskType::AudioGenerate.as_str()
        && matches!(
            task.normalized_status.as_str(),
            "pending" | "pending_submit" | "queued"
        )
}

pub(super) fn mark_stale_running_audio_task_failed_for_retry(
    workspace_root: &Path,
    task_id: &str,
) -> Result<MediaTaskOutput, String> {
    mark_audio_failed(
        workspace_root,
        task_id,
        AudioProviderError {
            code: "audio_worker_stale_running_recovered",
            message: "音频 worker 运行租约已过期，正在恢复任务。".to_string(),
            retryable: true,
            stage: "worker_recovery",
            status: None,
            provider_code: None,
        },
    )
}

pub(crate) fn spawn_audio_task_worker_for_created_task(
    task: &MediaTaskArtifactResponse,
    context: MediaTaskWorkerContext,
) -> Option<JoinHandle<Result<MediaTaskOutput, String>>> {
    if !should_execute_created_audio_task(task) {
        return None;
    }
    let workspace_root = workspace_root_from_task(task)?;
    let task_id = task.task_id.clone();
    Some(tokio::spawn(async move {
        execute_audio_task(workspace_root, task_id, &context).await
    }))
}

pub(super) fn spawn_audio_task_worker_for_existing_task(
    workspace_root: &Path,
    task: &MediaTaskArtifactResponse,
    context: MediaTaskWorkerContext,
) -> Option<JoinHandle<Result<MediaTaskOutput, String>>> {
    if !should_execute_pending_audio_task(task) {
        return None;
    }
    let workspace_root = workspace_root.to_path_buf();
    let task_id = task.task_id.clone();
    Some(tokio::spawn(async move {
        execute_audio_task(workspace_root, task_id, &context).await
    }))
}

pub(super) fn list_audio_tasks_for_workspace(
    workspace_root: impl AsRef<Path>,
    limit: Option<usize>,
) -> Result<Vec<MediaTaskArtifactResponse>, String> {
    let listed = crate::media_task::list_media_task_artifacts(
        app_server_protocol::MediaTaskArtifactListParams {
            project_root_path: workspace_root.as_ref().to_string_lossy().to_string(),
            task_type: Some(MediaTaskType::AudioGenerate.as_str().to_string()),
            limit,
            ..Default::default()
        },
    )?;
    Ok(listed.tasks)
}

pub(super) async fn execute_audio_task(
    workspace_root: PathBuf,
    task_id: String,
    context: &MediaTaskWorkerContext,
) -> Result<MediaTaskOutput, String> {
    let task =
        load_task_output(&workspace_root, &task_id, None).map_err(|error| error.to_string())?;
    if task.normalized_status == "cancelled" {
        return Ok(task);
    }
    let config = match super::route::audio_speech_runner_config_from_resolved_route(
        &workspace_root,
        &task_id,
        context,
    ) {
        Ok(Some(config)) => config,
        Ok(None) => {
            return mark_audio_start_failed(
                &workspace_root,
                &task_id,
                "音频任务缺少 openai_audio_speech resolved route，请重新创建任务。".to_string(),
            );
        }
        Err(error) => return mark_audio_start_failed(&workspace_root, &task_id, error),
    };

    let payload = load_task_output(&workspace_root, &task_id, None)
        .map_err(|error| error.to_string())?
        .record
        .payload;
    patch_task_artifact(
        &workspace_root,
        &task_id,
        None,
        TaskArtifactPatch {
            status: Some("running".to_string()),
            payload_patch: Some(audio_running_payload_patch(
                &payload,
                &config.provider_id,
                &config.model_id,
            )),
            progress: Some(TaskProgress {
                phase: Some("running".to_string()),
                percent: Some(5),
                message: Some("音频 Provider 请求已开始。".to_string()),
                preview_slots: Vec::new(),
            }),
            current_attempt_worker_id: Some(Some(AUDIO_TASK_RUNNER_WORKER_ID.to_string())),
            ..TaskArtifactPatch::default()
        },
    )
    .map_err(|error| error.to_string())?;

    let input = read_string(&payload, &["source_text", "sourceText", "prompt"])
        .ok_or_else(|| "音频任务缺少 source_text".to_string())?;
    let request = SpeechProviderRequest {
        model_id: config.model_id.clone(),
        input,
        voice: read_string(&payload, &["voice"]),
        instructions: read_string(&payload, &["voice_style", "voiceStyle"]),
        response_format: read_string(&payload, &["response_format", "responseFormat"]).or_else(
            || {
                read_string(&payload, &["mime_type", "mimeType"])
                    .map(|mime| audio_format_from_mime(&mime))
            },
        ),
        speed: read_string(&payload, &["speed"]),
    };
    let provider_output = match execute_speech_generation(&config.provider, &request).await {
        Ok(output) => output,
        Err(error) => return mark_audio_failed(&workspace_root, &task_id, error),
    };
    let current =
        load_task_output(&workspace_root, &task_id, None).map_err(|error| error.to_string())?;
    if current.normalized_status == "cancelled" {
        return Ok(current);
    }
    let (absolute_path, relative_path) = match output_path(
        &workspace_root,
        &task_id,
        &payload,
        &provider_output.mime_type,
    ) {
        Ok(value) => value,
        Err(message) => {
            return mark_audio_failed(
                &workspace_root,
                &task_id,
                audio_worker_error("audio_output_path_invalid", message, false, "output"),
            )
        }
    };
    if let Some(parent) = absolute_path.parent() {
        if let Err(error) = tokio::fs::create_dir_all(parent).await {
            return mark_audio_failed(
                &workspace_root,
                &task_id,
                audio_worker_error(
                    "audio_output_directory_failed",
                    format!("创建音频输出目录失败: {error}"),
                    true,
                    "output",
                ),
            );
        }
    }
    if let Err(error) = tokio::fs::write(&absolute_path, provider_output.audio).await {
        remove_unreferenced_audio_output(&workspace_root, &task_id, &relative_path);
        return mark_audio_failed(
            &workspace_root,
            &task_id,
            audio_worker_error(
                "audio_output_write_failed",
                format!("写入音频输出失败: {error}"),
                true,
                "output",
            ),
        );
    }

    let current =
        load_task_output(&workspace_root, &task_id, None).map_err(|error| error.to_string())?;
    if current.normalized_status == "cancelled" {
        remove_unreferenced_audio_output(&workspace_root, &task_id, &relative_path);
        return Ok(current);
    }

    if let Err(error) = crate::media_task::complete_audio_generation_task_artifact(
        MediaTaskArtifactAudioCompleteParams {
            project_root_path: workspace_root.to_string_lossy().to_string(),
            task_ref: task_id.clone(),
            audio_path: relative_path.clone(),
            mime_type: Some(provider_output.mime_type.clone()),
            duration_ms: None,
            provider_id: Some(config.provider_id.clone()),
            model: Some(config.model_id.clone()),
        },
        context.sidecar_store.as_deref(),
    ) {
        let current = load_task_output(&workspace_root, &task_id, None)
            .map_err(|load_error| load_error.to_string())?;
        if current.normalized_status == "cancelled" {
            remove_unreferenced_audio_output(&workspace_root, &task_id, &relative_path);
            return Ok(current);
        }
        remove_unreferenced_audio_output(&workspace_root, &task_id, &relative_path);
        return mark_audio_failed(
            &workspace_root,
            &task_id,
            audio_worker_error(
                "audio_artifact_complete_failed",
                error.to_string(),
                true,
                "artifact",
            ),
        );
    }

    patch_task_artifact(
        &workspace_root,
        &task_id,
        None,
        TaskArtifactPatch {
            payload_patch: Some(audio_completed_payload_patch(
                &payload,
                &config.provider_id,
                &config.model_id,
                &provider_output.response,
            )),
            progress: Some(TaskProgress {
                phase: Some("succeeded".to_string()),
                percent: Some(100),
                message: Some("音频任务已完成。".to_string()),
                preview_slots: Vec::new(),
            }),
            current_attempt_worker_id: Some(Some(AUDIO_TASK_RUNNER_WORKER_ID.to_string())),
            ..TaskArtifactPatch::default()
        },
    )
    .map_err(|error| error.to_string())?;
    load_task_output(&workspace_root, &task_id, None).map_err(|error| error.to_string())
}

fn mark_audio_start_failed(
    workspace_root: &Path,
    task_id: &str,
    message: String,
) -> Result<MediaTaskOutput, String> {
    mark_audio_failed(
        workspace_root,
        task_id,
        AudioProviderError {
            code: "audio_worker_start_failed",
            message,
            retryable: false,
            stage: "worker_start",
            status: None,
            provider_code: None,
        },
    )
}

fn mark_audio_failed(
    workspace_root: &Path,
    task_id: &str,
    error: AudioProviderError,
) -> Result<MediaTaskOutput, String> {
    let task =
        load_task_output(workspace_root, task_id, None).map_err(|value| value.to_string())?;
    if task.normalized_status == "cancelled" {
        return Ok(task);
    }
    let task_error = TaskErrorRecord {
        code: error.code.to_string(),
        message: error.message,
        retryable: error.retryable,
        stage: Some(error.stage.to_string()),
        provider_code: error.provider_code,
        occurred_at: Some(Utc::now().to_rfc3339()),
    };
    patch_task_artifact(
        workspace_root,
        task_id,
        None,
        TaskArtifactPatch {
            status: Some("failed".to_string()),
            payload_patch: Some(audio_failed_payload_patch(
                &task.record.payload,
                &task_error,
            )),
            last_error: Some(Some(task_error.clone())),
            progress: Some(TaskProgress {
                phase: Some("failed".to_string()),
                percent: Some(100),
                message: Some(task_error.message.clone()),
                preview_slots: Vec::new(),
            }),
            current_attempt_worker_id: Some(Some(AUDIO_TASK_RUNNER_WORKER_ID.to_string())),
            ..TaskArtifactPatch::default()
        },
    )
    .map_err(|value| value.to_string())
}

fn audio_worker_error(
    code: &'static str,
    message: String,
    retryable: bool,
    stage: &'static str,
) -> AudioProviderError {
    AudioProviderError {
        code,
        message,
        retryable,
        stage,
        status: None,
        provider_code: None,
    }
}

fn remove_unreferenced_audio_output(workspace_root: &Path, task_id: &str, relative_path: &str) {
    if !is_worker_owned_audio_path(task_id, relative_path) {
        return;
    }
    let path = workspace_root.join(relative_path);
    if let Err(error) = std::fs::remove_file(&path) {
        if error.kind() != std::io::ErrorKind::NotFound {
            tracing::warn!(path = %path.display(), error = %error, "清理未引用音频输出失败");
        }
    }
}

fn is_worker_owned_audio_path(task_id: &str, relative_path: &str) -> bool {
    let path = Path::new(relative_path);
    path.parent() == Some(Path::new(".lime/media/audio"))
        && path
            .file_stem()
            .and_then(|value| value.to_str())
            .is_some_and(|stem| stem == task_id)
}

fn output_path(
    workspace_root: &Path,
    task_id: &str,
    payload: &Value,
    mime_type: &str,
) -> Result<(PathBuf, String), String> {
    let requested = read_string(
        payload,
        &["output_path", "outputPath", "audio_path", "audioPath"],
    );
    let relative = requested.unwrap_or_else(|| {
        format!(
            ".lime/media/audio/{}.{}",
            task_id,
            audio_extension_from_mime(mime_type)
        )
    });
    let path = Path::new(&relative);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err("音频 output_path 必须是 workspace 内的相对路径".to_string());
    }
    Ok((
        workspace_root.join(path),
        path.to_string_lossy().to_string(),
    ))
}

fn audio_extension_from_mime(mime_type: &str) -> &'static str {
    match mime_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "audio/wav" | "audio/x-wav" => "wav",
        "audio/ogg" | "audio/opus" => "ogg",
        "audio/aac" => "aac",
        "audio/flac" => "flac",
        _ => "mp3",
    }
}

fn audio_format_from_mime(mime_type: &str) -> String {
    audio_extension_from_mime(mime_type).to_string()
}

fn read_string(payload: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|key| payload.get(*key))
        .find_map(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn audio_running_payload_patch(payload: &Value, provider_id: &str, model_id: &str) -> Value {
    let mut patch = audio_event_patch(
        "message.created",
        json!({"role": "assistant"}),
        json!({
            "status": "running",
            "providerId": provider_id,
            "modelId": model_id,
        }),
    );
    if let Some(object) = patch.as_object_mut() {
        object.insert("audio_output".to_string(), json!({"status": "running"}));
    }
    let _ = payload;
    patch
}

fn audio_completed_payload_patch(
    payload: &Value,
    provider_id: &str,
    model_id: &str,
    provider_response: &Value,
) -> Value {
    let mut patch = audio_event_patch(
        "turn.completed",
        json!({"backend": "media_runtime"}),
        json!({
            "status": "succeeded",
            "providerId": provider_id,
            "modelId": model_id,
            "response": provider_response,
        }),
    );
    if let Some(object) = patch.as_object_mut() {
        object.insert("audio_output".to_string(), json!({"status": "completed"}));
    }
    let _ = payload;
    patch
}

fn audio_failed_payload_patch(payload: &Value, error: &TaskErrorRecord) -> Value {
    let mut patch = audio_event_patch(
        "turn.failed",
        json!({"backend": "media_runtime", "code": error.code, "message": error.message}),
        json!({
            "status": "failed",
            "errorCode": error.code,
            "errorStage": error.stage,
            "retryable": error.retryable,
        }),
    );
    if let Some(object) = patch.as_object_mut() {
        object.insert("audio_output".to_string(), json!({"status": "failed"}));
    }
    let _ = payload;
    patch
}

fn audio_event_patch(event_type: &str, event_payload: Value, diagnostics: Value) -> Value {
    let event = json!({"type": event_type, "payload": event_payload});
    json!({
        "llm_events": [event.clone()],
        "llmEvents": [event],
        "provider_diagnostics": {
            "taskFamily": "text_to_speech",
            "transport": "provider_http",
            "credential": "not_embedded",
            "eventOwner": "media_runtime",
            "details": diagnostics,
        },
        "providerDiagnostics": {
            "taskFamily": "text_to_speech",
            "transport": "provider_http",
            "credential": "not_embedded",
            "eventOwner": "media_runtime",
            "details": diagnostics,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_path_rejects_escape_and_defaults_inside_workspace() {
        let workspace = Path::new("/tmp/lime-workspace");
        let (path, relative) = output_path(workspace, "task-1", &json!({}), "audio/mpeg")
            .expect("default output path");
        assert_eq!(relative, ".lime/media/audio/task-1.mp3");
        assert_eq!(path, workspace.join(relative));
        assert!(output_path(
            workspace,
            "task-1",
            &json!({"output_path": "../x.mp3"}),
            "audio/mpeg"
        )
        .is_err());
        assert!(output_path(
            workspace,
            "task-1",
            &json!({"output_path": "/tmp/x.mp3"}),
            "audio/mpeg"
        )
        .is_err());
        assert!(is_worker_owned_audio_path(
            "task-1",
            ".lime/media/audio/task-1.mp3"
        ));
        assert!(!is_worker_owned_audio_path("task-1", "exports/task-1.mp3"));
    }
}
