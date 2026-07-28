use super::{workspace_root_from_task, MediaTaskWorkerContext};
use app_server_protocol::MediaTaskArtifactResponse;
use chrono::{DateTime, Duration, Utc};
use lime_media_runtime::{
    execute_video_generation_task, patch_task_artifact, MediaTaskOutput, MediaTaskType,
    TaskArtifactPatch, TaskErrorRecord, TaskProgress, VIDEO_TASK_RUNNER_WORKER_ID,
};
use std::path::{Path, PathBuf};
use tokio::task::JoinHandle;

use super::route::video_generation_runner_config_from_resolved_route;

pub(super) const VIDEO_TASK_WORKER_STALE_RUNNING_SECS: i64 = 60;

fn should_execute_created_video_task(task: &MediaTaskArtifactResponse) -> bool {
    task.task_type == MediaTaskType::VideoGenerate.as_str()
        && !task.reused_existing
        && matches!(
            task.normalized_status.as_str(),
            "pending" | "pending_submit" | "queued" | "running"
        )
}

pub(super) fn should_execute_pending_video_task(task: &MediaTaskArtifactResponse) -> bool {
    task.task_type == MediaTaskType::VideoGenerate.as_str()
        && matches!(
            task.normalized_status.as_str(),
            "pending" | "pending_submit" | "queued"
        )
}

pub(super) fn should_recover_stale_running_video_task(
    task: &MediaTaskArtifactResponse,
    now: DateTime<Utc>,
) -> bool {
    if task.task_type != MediaTaskType::VideoGenerate.as_str()
        || task.normalized_status != "running"
        || current_attempt_worker_id(task).as_deref() != Some(VIDEO_TASK_RUNNER_WORKER_ID)
        || provider_task_protocol(task).as_deref() != Some("xai_video")
        || provider_task_request_id(task).is_none()
    {
        return false;
    }
    running_updated_at(task).is_some_and(|updated_at| {
        now.signed_duration_since(updated_at)
            >= Duration::seconds(VIDEO_TASK_WORKER_STALE_RUNNING_SECS)
    })
}

pub(crate) fn spawn_video_task_worker_for_created_task(
    task: &MediaTaskArtifactResponse,
    context: MediaTaskWorkerContext,
) -> Option<JoinHandle<Result<MediaTaskOutput, String>>> {
    if !should_execute_created_video_task(task) {
        tracing::info!(
            task_id = %task.task_id,
            task_type = %task.task_type,
            status = %task.normalized_status,
            reused_existing = task.reused_existing,
            "video task worker skipped created task"
        );
        return None;
    }

    let Some(workspace_root) = workspace_root_from_task(task) else {
        tracing::warn!(
            task_id = %task.task_id,
            artifact_path = %task.artifact_path,
            absolute_artifact_path = %task.absolute_artifact_path,
            "video task worker could not resolve workspace root"
        );
        return None;
    };
    let task_id = task.task_id.clone();
    tracing::info!(
        task_id = %task_id,
        workspace_root = %workspace_root.display(),
        status = %task.normalized_status,
        "video task worker spawned for created task"
    );
    Some(tokio::spawn(async move {
        let result = execute_video_task(workspace_root, task_id.clone(), &context).await;
        if let Err(error) = &result {
            tracing::warn!(task_id = %task_id, error = %error, "video task worker failed");
        }
        result
    }))
}

pub(super) fn spawn_video_task_worker_for_existing_task(
    workspace_root: &Path,
    task: &MediaTaskArtifactResponse,
    context: MediaTaskWorkerContext,
) -> Option<JoinHandle<Result<MediaTaskOutput, String>>> {
    if !should_execute_pending_video_task(task)
        && !(task.normalized_status == "running"
            && provider_task_protocol(task).as_deref() == Some("xai_video")
            && provider_task_request_id(task).is_some())
    {
        return None;
    }
    let workspace_root = workspace_root.to_path_buf();
    let task_id = task.task_id.clone();
    Some(tokio::spawn(async move {
        let result = execute_video_task(workspace_root, task_id.clone(), &context).await;
        if let Err(error) = &result {
            tracing::warn!(task_id = %task_id, error = %error, "video task recovery worker failed");
        }
        result
    }))
}

async fn execute_video_task(
    workspace_root: PathBuf,
    task_id: String,
    context: &MediaTaskWorkerContext,
) -> Result<MediaTaskOutput, String> {
    tracing::info!(
        task_id = %task_id,
        workspace_root = %workspace_root.display(),
        "video task worker resolving runner config"
    );
    let runner_config = match video_generation_runner_config_from_resolved_route(
        &workspace_root,
        &task_id,
        context,
    ) {
        Ok(Some(config)) => config,
        Ok(None) => {
            return mark_video_task_worker_start_failed(
                &workspace_root,
                &task_id,
                "视频任务缺少完整 resolved route，请重新创建任务。".to_string(),
            );
        }
        Err(error) => {
            tracing::warn!(
                task_id = %task_id,
                error = %error,
                "failed to resolve video task route runner config"
            );
            return mark_video_task_worker_start_failed(&workspace_root, &task_id, error);
        }
    };

    tracing::info!(
        task_id = %task_id,
        endpoint = %runner_config.endpoint,
        "video task worker using resolved route"
    );
    let output = execute_video_generation_task(&workspace_root, &task_id, &runner_config)
        .await
        .map_err(|error| error.to_string())?;
    tracing::info!(
        task_id = %task_id,
        status = %output.normalized_status,
        attempt_count = output.attempt_count,
        "video task worker finished"
    );
    Ok(output)
}

fn mark_video_task_worker_start_failed(
    workspace_root: &Path,
    task_id: &str,
    message: String,
) -> Result<MediaTaskOutput, String> {
    let error = TaskErrorRecord {
        code: "video_worker_start_failed".to_string(),
        message: message.clone(),
        retryable: false,
        stage: Some("worker_start".to_string()),
        provider_code: None,
        occurred_at: None,
    };
    patch_task_artifact(
        workspace_root,
        task_id,
        None,
        TaskArtifactPatch {
            status: Some("failed".to_string()),
            last_error: Some(Some(error)),
            progress: Some(TaskProgress {
                phase: Some("failed".to_string()),
                percent: Some(100),
                message: Some(message),
                preview_slots: Vec::new(),
            }),
            current_attempt_worker_id: Some(Some(VIDEO_TASK_RUNNER_WORKER_ID.to_string())),
            ..TaskArtifactPatch::default()
        },
    )
    .map_err(|error| error.to_string())
}

fn provider_task_protocol(task: &MediaTaskArtifactResponse) -> Option<String> {
    task.record
        .get("payload")
        .and_then(|payload| {
            payload
                .get("provider_task")
                .or_else(|| payload.get("providerTask"))
        })
        .and_then(|provider_task| provider_task.get("protocol"))
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|protocol| !protocol.is_empty())
        .map(ToString::to_string)
}

fn provider_task_request_id(task: &MediaTaskArtifactResponse) -> Option<String> {
    task.record
        .get("payload")
        .and_then(|payload| {
            payload
                .get("provider_task")
                .or_else(|| payload.get("providerTask"))
        })
        .and_then(|provider_task| {
            provider_task
                .get("request_id")
                .or_else(|| provider_task.get("requestId"))
        })
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|request_id| !request_id.is_empty())
        .map(ToString::to_string)
}

fn current_attempt_worker_id(task: &MediaTaskArtifactResponse) -> Option<String> {
    task.record
        .get("attempts")
        .and_then(serde_json::Value::as_array)
        .and_then(|attempts| attempts.last())
        .and_then(|attempt| attempt.get("worker_id").or_else(|| attempt.get("workerId")))
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|worker_id| !worker_id.is_empty())
        .map(ToString::to_string)
}

fn running_updated_at(task: &MediaTaskArtifactResponse) -> Option<DateTime<Utc>> {
    task.record
        .get("updated_at")
        .or_else(|| task.record.get("updatedAt"))
        .and_then(serde_json::Value::as_str)
        .and_then(|raw| DateTime::parse_from_rfc3339(raw.trim()).ok())
        .map(|value| value.with_timezone(&Utc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::MediaTaskArtifactVideoCreateParams;

    #[test]
    fn created_video_task_admission_is_narrow_and_skips_reused_artifacts() {
        let workspace = tempfile::tempdir().expect("workspace");
        let task = crate::media_task::create_video_generation_task_artifact(
            MediaTaskArtifactVideoCreateParams {
                project_root_path: workspace.path().to_string_lossy().to_string(),
                prompt: "生成视频".to_string(),
                ..MediaTaskArtifactVideoCreateParams::default()
            },
            None,
        )
        .expect("video task");

        assert!(should_execute_created_video_task(&task));

        let mut reused = task.clone();
        reused.reused_existing = true;
        assert!(!should_execute_created_video_task(&reused));
    }

    #[test]
    fn only_stale_xai_running_tasks_with_request_id_are_recoverable() {
        let now = Utc::now();
        let mut task = MediaTaskArtifactResponse {
            task_type: MediaTaskType::VideoGenerate.as_str().to_string(),
            normalized_status: "running".to_string(),
            record: serde_json::json!({
                "updated_at": (now - Duration::seconds(61)).to_rfc3339(),
                "payload": {
                    "provider_task": {
                        "protocol": "xai_video",
                        "request_id": "request-1"
                    }
                },
                "attempts": [{
                    "worker_id": VIDEO_TASK_RUNNER_WORKER_ID
                }]
            }),
            ..MediaTaskArtifactResponse::default()
        };

        assert!(should_recover_stale_running_video_task(&task, now));

        task.record["updated_at"] = serde_json::json!((now - Duration::seconds(30)).to_rfc3339());
        assert!(!should_recover_stale_running_video_task(&task, now));

        task.record["updated_at"] = serde_json::json!((now - Duration::seconds(61)).to_rfc3339());
        task.record["payload"]["provider_task"]["request_id"] = serde_json::Value::Null;
        assert!(!should_recover_stale_running_video_task(&task, now));
    }
}
