use super::{workspace_root_from_task, MediaTaskWorkerContext};
use app_server_protocol::MediaTaskArtifactResponse;
use chrono::Utc;
use lime_media_runtime::{
    load_task_output, patch_task_artifact, MediaTaskOutput, MediaTaskType, TaskArtifactPatch,
    TaskErrorRecord, TaskProgress,
};
use model_provider::audio::{
    execute_transcription, AudioProviderError, TranscriptionProviderRequest,
};
use serde_json::{json, Value};
use std::path::{Component, Path, PathBuf};
use tokio::task::JoinHandle;

pub(super) const TRANSCRIPTION_TASK_RUNNER_WORKER_ID: &str = "lime-transcription-provider-worker";

pub(super) fn should_execute_created_transcription_task(task: &MediaTaskArtifactResponse) -> bool {
    task.task_type == MediaTaskType::TranscriptionGenerate.as_str()
        && !task.reused_existing
        && matches!(
            task.normalized_status.as_str(),
            "pending" | "pending_submit" | "queued" | "running"
        )
}

pub(super) fn should_execute_pending_transcription_task(task: &MediaTaskArtifactResponse) -> bool {
    task.task_type == MediaTaskType::TranscriptionGenerate.as_str()
        && matches!(
            task.normalized_status.as_str(),
            "pending" | "pending_submit" | "queued"
        )
}

pub(crate) fn spawn_transcription_task_worker_for_created_task(
    task: &MediaTaskArtifactResponse,
    context: MediaTaskWorkerContext,
) -> Option<JoinHandle<Result<MediaTaskOutput, String>>> {
    if !should_execute_created_transcription_task(task) {
        return None;
    }
    let workspace_root = workspace_root_from_task(task)?;
    let task_id = task.task_id.clone();
    Some(tokio::spawn(async move {
        execute_transcription_task(workspace_root, task_id, &context).await
    }))
}

pub(super) fn spawn_transcription_task_worker_for_existing_task(
    workspace_root: &Path,
    task: &MediaTaskArtifactResponse,
    context: MediaTaskWorkerContext,
) -> Option<JoinHandle<Result<MediaTaskOutput, String>>> {
    if !should_execute_pending_transcription_task(task) {
        return None;
    }
    let workspace_root = workspace_root.to_path_buf();
    let task_id = task.task_id.clone();
    Some(tokio::spawn(async move {
        execute_transcription_task(workspace_root, task_id, &context).await
    }))
}

pub(super) fn list_transcription_tasks_for_workspace(
    workspace_root: impl AsRef<Path>,
    limit: Option<usize>,
) -> Result<Vec<MediaTaskArtifactResponse>, String> {
    let listed = crate::media_task::list_media_task_artifacts(
        app_server_protocol::MediaTaskArtifactListParams {
            project_root_path: workspace_root.as_ref().to_string_lossy().to_string(),
            task_type: Some(MediaTaskType::TranscriptionGenerate.as_str().to_string()),
            limit,
            ..Default::default()
        },
    )?;
    Ok(listed.tasks)
}

pub(super) async fn execute_transcription_task(
    workspace_root: PathBuf,
    task_id: String,
    context: &MediaTaskWorkerContext,
) -> Result<MediaTaskOutput, String> {
    let task =
        load_task_output(&workspace_root, &task_id, None).map_err(|error| error.to_string())?;
    if task.normalized_status == "cancelled" {
        return Ok(task);
    }
    let config = match super::route::audio_transcription_runner_config_from_resolved_route(
        &workspace_root,
        &task_id,
        context,
    ) {
        Ok(Some(config)) => config,
        Ok(None) => {
            return mark_transcription_start_failed_with_code(
                &workspace_root,
                &task_id,
                "transcription_provider_unconfigured",
                "转写任务缺少 openai_audio_transcription resolved route，请重新创建任务。"
                    .to_string(),
            )
        }
        Err(error) => {
            return mark_transcription_start_failed_with_code(
                &workspace_root,
                &task_id,
                "transcription_provider_resolution_failed",
                error,
            )
        }
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
            payload_patch: Some(transcription_running_payload_patch(
                &config.provider_id,
                &config.model_id,
            )),
            progress: Some(TaskProgress {
                phase: Some("running".to_string()),
                percent: Some(5),
                message: Some("音频转写 Provider 请求已开始。".to_string()),
                preview_slots: Vec::new(),
            }),
            current_attempt_worker_id: Some(Some(TRANSCRIPTION_TASK_RUNNER_WORKER_ID.to_string())),
            ..TaskArtifactPatch::default()
        },
    )
    .map_err(|error| error.to_string())?;

    let (audio, filename, mime_type) = match read_audio_source(&workspace_root, &payload).await {
        Ok(value) => value,
        Err(error) => return mark_transcription_failed(&workspace_root, &task_id, error),
    };
    let output_format = read_string(&payload, &["output_format", "outputFormat"])
        .map(|value| normalize_output_format(&value))
        .unwrap_or_else(|| "txt".to_string());
    let response_format = Some(provider_response_format(&output_format));
    let request = TranscriptionProviderRequest {
        model_id: config.model_id.clone(),
        audio,
        filename,
        mime_type,
        language: read_string(&payload, &["language"]),
        prompt: read_string(&payload, &["prompt"]),
        response_format,
    };
    let provider_output = match execute_transcription(&config.provider, &request).await {
        Ok(output) => output,
        Err(error) => return mark_transcription_failed(&workspace_root, &task_id, error),
    };
    let current =
        load_task_output(&workspace_root, &task_id, None).map_err(|error| error.to_string())?;
    if current.normalized_status == "cancelled" {
        return Ok(current);
    }

    let (absolute_path, relative_path) =
        match transcript_output_path(&workspace_root, &task_id, &payload, &output_format) {
            Ok(value) => value,
            Err(message) => {
                return mark_transcription_failed(
                    &workspace_root,
                    &task_id,
                    audio_worker_error(
                        "invalid_transcription_task_payload",
                        message,
                        false,
                        "output",
                    ),
                )
            }
        };
    if let Some(parent) = absolute_path.parent() {
        if let Err(error) = tokio::fs::create_dir_all(parent).await {
            return mark_transcription_failed(
                &workspace_root,
                &task_id,
                audio_worker_error(
                    "transcript_output_write_failed",
                    format!("创建 transcript 输出目录失败: {error}"),
                    true,
                    "output",
                ),
            );
        }
    }
    let document = transcription_document(&provider_output, &output_format);
    if let Err(error) = tokio::fs::write(&absolute_path, document.as_bytes()).await {
        remove_unreferenced_transcript_output(&workspace_root, &task_id, &relative_path);
        return mark_transcription_failed(
            &workspace_root,
            &task_id,
            audio_worker_error(
                "transcript_output_write_failed",
                format!("写入 transcript 输出失败: {error}"),
                true,
                "output",
            ),
        );
    }
    let current =
        load_task_output(&workspace_root, &task_id, None).map_err(|error| error.to_string())?;
    if current.normalized_status == "cancelled" {
        remove_unreferenced_transcript_output(&workspace_root, &task_id, &relative_path);
        return Ok(current);
    }

    patch_task_artifact(
        &workspace_root,
        &task_id,
        None,
        TaskArtifactPatch {
            status: Some("succeeded".to_string()),
            payload_patch: Some(transcription_completed_payload_patch(
                &payload,
                &relative_path,
                &output_format,
                &config.provider_id,
                &config.model_id,
                &provider_output,
            )),
            result: Some(Some(json!({
                "kind": "transcription_result",
                "status": "completed",
                "transcript_path": relative_path,
                "output_format": output_format,
                "text": provider_output.text,
                "language": provider_output.language,
                "provider_id": config.provider_id,
                "model": config.model_id,
            }))),
            progress: Some(TaskProgress {
                phase: Some("succeeded".to_string()),
                percent: Some(100),
                message: Some("音频转写任务已完成。".to_string()),
                preview_slots: Vec::new(),
            }),
            current_attempt_worker_id: Some(Some(TRANSCRIPTION_TASK_RUNNER_WORKER_ID.to_string())),
            ..TaskArtifactPatch::default()
        },
    )
    .map_err(|error| error.to_string())?;
    load_task_output(&workspace_root, &task_id, None).map_err(|error| error.to_string())
}

async fn read_audio_source(
    workspace_root: &Path,
    payload: &Value,
) -> Result<(Vec<u8>, String, String), AudioProviderError> {
    if let Some(source_path) = read_string(payload, &["source_path", "sourcePath"]) {
        let workspace_root = tokio::fs::canonicalize(workspace_root).await.map_err(|error| {
            audio_worker_error(
                "transcription_source_unavailable",
                format!("解析 workspace 根目录失败: {error}"),
                false,
                "input",
            )
        })?;
        let candidate = PathBuf::from(&source_path);
        let candidate = if candidate.is_absolute() {
            candidate
        } else {
            workspace_root.join(candidate)
        };
        let path = tokio::fs::canonicalize(&candidate).await.map_err(|error| {
            audio_worker_error(
                "transcription_source_unavailable",
                format!("读取音频源文件失败: {error}"),
                false,
                "input",
            )
        })?;
        if path.parent().is_none() || !path.starts_with(&workspace_root) {
            return Err(audio_worker_error(
                "transcription_source_unavailable",
                "音频 source_path 必须位于 workspace 内".to_string(),
                false,
                "input",
            ));
        }
        let bytes = tokio::fs::read(&path).await.map_err(|error| {
            audio_worker_error(
                "transcription_source_unavailable",
                format!("读取音频源文件失败: {error}"),
                false,
                "input",
            )
        })?;
        if bytes.is_empty() {
            return Err(audio_worker_error(
                "transcription_source_empty",
                "音频源文件为空".to_string(),
                false,
                "input",
            ));
        }
        let filename = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("audio.bin")
            .to_string();
        let mime_type = mime_type_from_filename(&filename);
        return Ok((bytes, filename, mime_type));
    }
    let source_url = read_string(payload, &["source_url", "sourceUrl"]).ok_or_else(|| {
        audio_worker_error(
            "invalid_transcription_task_payload",
            "转写任务缺少 source_path 或 source_url".to_string(),
            false,
            "input",
        )
    })?;
    let url = reqwest::Url::parse(&source_url).map_err(|error| {
        audio_worker_error(
            "transcription_source_download_failed",
            format!("音频 source_url 无效: {error}"),
            false,
            "input",
        )
    })?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(audio_worker_error(
            "transcription_source_download_failed",
            "音频 source_url 只允许 http 或 https".to_string(),
            false,
            "input",
        ));
    }
    let response = reqwest::Client::builder()
        .no_proxy()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|error| {
            audio_worker_error(
                "transcription_source_download_failed",
                format!("初始化音频 source_url client 失败: {error}"),
                false,
                "input",
            )
        })?
        .get(url.clone())
        .send()
        .await
        .map_err(|error| {
            audio_worker_error(
                "transcription_source_download_failed",
                format!("读取音频 source_url 失败: {error}"),
                true,
                "input",
            )
        })?;
    let status = response.status();
    if !status.is_success() {
        return Err(audio_worker_error(
            "transcription_source_download_failed",
            format!("读取音频 source_url 返回 HTTP {}", status.as_u16()),
            status.is_server_error() || status.as_u16() == 429,
            "input",
        ));
    }
    let mime_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .unwrap_or_else(|| mime_type_from_filename(url.path()));
    let filename = Path::new(url.path())
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("audio.bin")
        .to_string();
    let bytes = response.bytes().await.map_err(|error| {
        audio_worker_error(
            "transcription_source_download_failed",
            format!("读取音频 source_url 内容失败: {error}"),
            true,
            "input",
        )
    })?;
    if bytes.is_empty() {
        return Err(audio_worker_error(
            "transcription_source_empty",
            "音频 source_url 返回空内容".to_string(),
            false,
            "input",
        ));
    }
    Ok((bytes.to_vec(), filename, mime_type))
}

fn transcript_output_path(
    workspace_root: &Path,
    task_id: &str,
    payload: &Value,
    output_format: &str,
) -> Result<(PathBuf, String), String> {
    let extension = transcript_extension(output_format);
    let relative = read_string(payload, &["output_path", "outputPath"])
        .unwrap_or_else(|| format!(".lime/media/transcription/{task_id}.{extension}"));
    let path = Path::new(&relative);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err("transcript output_path 必须是 workspace 内的相对路径".to_string());
    }
    Ok((
        workspace_root.join(path),
        path.to_string_lossy().to_string(),
    ))
}

fn remove_unreferenced_transcript_output(
    workspace_root: &Path,
    task_id: &str,
    relative_path: &str,
) {
    if !is_worker_owned_transcript_path(task_id, relative_path) {
        return;
    }
    let path = workspace_root.join(relative_path);
    if let Err(error) = std::fs::remove_file(&path) {
        if error.kind() != std::io::ErrorKind::NotFound {
            tracing::warn!(path = %path.display(), error = %error, "清理未引用 transcript 输出失败");
        }
    }
}

fn is_worker_owned_transcript_path(task_id: &str, relative_path: &str) -> bool {
    let path = Path::new(relative_path);
    path.parent() == Some(Path::new(".lime/media/transcription"))
        && path
            .file_stem()
            .and_then(|value| value.to_str())
            .is_some_and(|stem| stem == task_id)
}

fn transcription_document(
    output: &model_provider::audio::TranscriptionProviderOutput,
    format: &str,
) -> String {
    match normalize_response_format(format).as_str() {
        "json" | "verbose_json" => serde_json::to_string_pretty(&output.response)
            .unwrap_or_else(|_| json!({"text": output.text}).to_string()),
        _ => output.text.clone(),
    }
}

fn transcription_running_payload_patch(provider_id: &str, model_id: &str) -> Value {
    transcription_event_patch(
        "message.created",
        json!({"role": "assistant"}),
        json!({"status": "running", "providerId": provider_id, "modelId": model_id}),
        json!({"status": "running"}),
    )
}

fn transcription_completed_payload_patch(
    payload: &Value,
    transcript_path: &str,
    output_format: &str,
    provider_id: &str,
    model_id: &str,
    output: &model_provider::audio::TranscriptionProviderOutput,
) -> Value {
    let mut patch = transcription_event_patch(
        "turn.completed",
        json!({"backend": "media_runtime"}),
        json!({
            "status": "succeeded",
            "providerId": provider_id,
            "modelId": model_id,
            "response": output.response.clone(),
        }),
        json!({
            "status": "completed",
            "transcript_path": transcript_path,
            "output_format": output_format,
            "language": output.language.clone(),
            "provider_id": provider_id,
            "model": model_id,
        }),
    );
    if let Some(object) = patch.as_object_mut() {
        object.insert(
            "transcript".to_string(),
            json!({
                "kind": "transcript",
                "status": "completed",
                "transcript_path": transcript_path,
                "source_path": read_string(payload, &["source_path", "sourcePath"]),
                "source_url": read_string(payload, &["source_url", "sourceUrl"]),
                "language": output.language.clone(),
                "output_format": output_format,
                "provider_id": provider_id,
                "model": model_id,
            }),
        );
    }
    patch
}

fn mark_transcription_start_failed_with_code(
    workspace_root: &Path,
    task_id: &str,
    code: &'static str,
    message: String,
) -> Result<MediaTaskOutput, String> {
    mark_transcription_failed(
        workspace_root,
        task_id,
        audio_worker_error(
            code,
            message,
            false,
            "worker_start",
        ),
    )
}

fn mark_transcription_failed(
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
            payload_patch: Some(transcription_failed_payload_patch(
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
            current_attempt_worker_id: Some(Some(TRANSCRIPTION_TASK_RUNNER_WORKER_ID.to_string())),
            ..TaskArtifactPatch::default()
        },
    )
    .map_err(|value| value.to_string())
}

fn transcription_failed_payload_patch(payload: &Value, error: &TaskErrorRecord) -> Value {
    let mut patch = transcription_event_patch(
        "turn.failed",
        json!({"backend": "media_runtime", "code": error.code, "message": error.message}),
        json!({"status": "failed", "errorCode": error.code, "retryable": error.retryable}),
        json!({
            "status": "failed",
            "error_code": error.code,
            "retryable": error.retryable,
        }),
    );
    if let Some(object) = patch.as_object_mut() {
        let mut transcript = payload
            .get("transcript")
            .cloned()
            .unwrap_or_else(|| json!({"kind": "transcript"}));
        if let Some(transcript_object) = transcript.as_object_mut() {
            transcript_object.insert("status".to_string(), json!("failed"));
            transcript_object.insert("error_code".to_string(), json!(error.code));
            transcript_object.insert("retryable".to_string(), json!(error.retryable));
        }
        object.insert("transcript".to_string(), transcript);
    }
    patch
}

fn transcription_event_patch(
    event_type: &str,
    event_payload: Value,
    diagnostics: Value,
    transcript: Value,
) -> Value {
    let event = json!({"type": event_type, "payload": event_payload});
    json!({
        "llm_events": [event.clone()],
        "llmEvents": [event],
        "provider_diagnostics": {
            "taskFamily": "speech_to_text",
            "transport": "provider_http",
            "credential": "not_embedded",
            "eventOwner": "media_runtime",
            "details": diagnostics,
        },
        "providerDiagnostics": {
            "taskFamily": "speech_to_text",
            "transport": "provider_http",
            "credential": "not_embedded",
            "eventOwner": "media_runtime",
            "details": diagnostics,
        },
        "transcript": transcript,
    })
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

fn read_string(payload: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|key| payload.get(*key))
        .find_map(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn normalize_response_format(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "json" | "verbose_json" | "srt" | "vtt" | "text" => value.trim().to_ascii_lowercase(),
        _ => "text".to_string(),
    }
}

fn normalize_output_format(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "json" | "verbose_json" | "srt" | "vtt" | "txt" | "text" => {
            if value.trim().eq_ignore_ascii_case("text") {
                "txt".to_string()
            } else {
                value.trim().to_ascii_lowercase()
            }
        }
        _ => "txt".to_string(),
    }
}

fn provider_response_format(output_format: &str) -> String {
    match normalize_output_format(output_format).as_str() {
        "txt" => "text".to_string(),
        value => value.to_string(),
    }
}

fn transcript_extension(format: &str) -> &'static str {
    match normalize_response_format(format).as_str() {
        "json" | "verbose_json" => "json",
        "srt" => "srt",
        "vtt" => "vtt",
        _ => "txt",
    }
}

fn mime_type_from_filename(filename: &str) -> String {
    match Path::new(filename)
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        "m4a" => "audio/mp4",
        "ogg" | "oga" => "audio/ogg",
        "opus" => "audio/opus",
        "flac" => "audio/flac",
        "webm" => "audio/webm",
        _ => "application/octet-stream",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcript_output_defaults_to_worker_owned_path() {
        let (path, relative) = transcript_output_path(
            Path::new("/tmp/lime-workspace"),
            "task-1",
            &json!({}),
            "srt",
        )
        .expect("output path");
        assert_eq!(relative, ".lime/media/transcription/task-1.srt");
        assert_eq!(path, Path::new("/tmp/lime-workspace").join(relative));
        assert!(is_worker_owned_transcript_path(
            "task-1",
            ".lime/media/transcription/task-1.srt"
        ));
        assert!(!is_worker_owned_transcript_path(
            "task-1",
            "exports/task-1.srt"
        ));
    }

    #[test]
    fn transcript_output_rejects_workspace_escape() {
        assert!(transcript_output_path(
            Path::new("/tmp/lime-workspace"),
            "task-1",
            &json!({"output_path": "../outside.txt"}),
            "txt"
        )
        .is_err());
    }

    #[test]
    fn transcript_output_rejects_absolute_path() {
        assert!(transcript_output_path(
            Path::new("/tmp/lime-workspace"),
            "task-1",
            &json!({"output_path": "/tmp/out.txt"}),
            "txt"
        )
        .is_err());
    }

    #[test]
    fn output_format_normalizes_text_alias_and_unknown_values() {
        assert_eq!(normalize_output_format("TEXT"), "txt");
        assert_eq!(normalize_output_format("verbose_json"), "verbose_json");
        assert_eq!(normalize_output_format("docx"), "txt");
        assert_eq!(provider_response_format("txt"), "text");
        assert_eq!(provider_response_format("vtt"), "vtt");
    }

    #[tokio::test]
    async fn source_path_must_resolve_inside_workspace() {
        let workspace = tempfile::tempdir().expect("workspace");
        let outside = tempfile::tempdir().expect("outside");
        let source = outside.path().join("interview.wav");
        tokio::fs::write(&source, b"audio")
            .await
            .expect("write source");

        let error = read_audio_source(
            workspace.path(),
            &json!({"source_path": source.to_string_lossy()}),
        )
        .await
        .expect_err("outside source must be rejected");
        assert_eq!(error.code, "transcription_source_unavailable");
    }

    #[tokio::test]
    async fn source_path_reads_workspace_file_and_infers_mime() {
        let workspace = tempfile::tempdir().expect("workspace");
        let source = workspace.path().join("interview.wav");
        tokio::fs::write(&source, b"audio")
            .await
            .expect("write source");

        let (bytes, filename, mime_type) = read_audio_source(
            workspace.path(),
            &json!({"source_path": "interview.wav"}),
        )
        .await
        .expect("workspace source");
        assert_eq!(bytes, b"audio");
        assert_eq!(filename, "interview.wav");
        assert_eq!(mime_type, "audio/wav");
    }

    #[tokio::test]
    async fn source_url_rejects_non_http_scheme_without_network_call() {
        let workspace = tempfile::tempdir().expect("workspace");
        let error = read_audio_source(
            workspace.path(),
            &json!({"source_url": "file:///tmp/interview.wav"}),
        )
        .await
        .expect_err("non-http source must be rejected");
        assert_eq!(error.code, "transcription_source_download_failed");
    }
}
