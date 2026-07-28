use app_server_protocol::MediaTaskArtifactResponse;
use lime_core::database::DbConnection;
use lime_media_runtime::{
    execute_image_generation_task, patch_task_artifact, ImageGenerationRunnerConfig,
    MediaTaskOutput, MediaTaskType, TaskArtifactPatch, TaskErrorRecord, TaskProgress,
    IMAGE_TASK_RUNNER_WORKER_ID,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::task::JoinHandle;

mod route;
mod scheduler;
mod video;
use route::image_generation_runner_config_from_resolved_route;
use scheduler::should_execute_pending_image_task;
#[cfg(test)]
use scheduler::{
    list_image_tasks_for_workspace, recover_stale_running_image_task_for_worker,
    retry_failed_image_task_for_worker, should_recover_stale_running_image_task,
    IMAGE_TASK_WORKER_STALE_RUNNING_SECS,
};
pub(crate) use scheduler::{
    spawn_media_task_worker_scheduler, spawn_pending_image_task_workers_for_workspace,
};
pub(crate) use video::spawn_video_task_worker_for_created_task;
use video::{
    should_execute_pending_video_task, should_recover_stale_running_video_task,
    spawn_video_task_worker_for_existing_task,
};

#[derive(Clone)]
pub(crate) struct MediaTaskWorkerContext {
    db: DbConnection,
    sidecar_store: Option<Arc<crate::runtime::SidecarStore>>,
}

impl MediaTaskWorkerContext {
    pub(crate) fn new(db: DbConnection) -> Self {
        Self {
            db,
            sidecar_store: None,
        }
    }

    pub(crate) fn with_sidecar_store(
        mut self,
        sidecar_store: Option<Arc<crate::runtime::SidecarStore>>,
    ) -> Self {
        self.sidecar_store = sidecar_store;
        self
    }
}

pub(crate) fn should_execute_created_image_task(task: &MediaTaskArtifactResponse) -> bool {
    task.task_type == MediaTaskType::ImageGenerate.as_str()
        && !task.reused_existing
        && matches!(
            task.normalized_status.as_str(),
            "pending" | "pending_submit" | "queued" | "running"
        )
}

pub(crate) fn spawn_image_task_worker_for_created_task(
    task: &MediaTaskArtifactResponse,
    context: MediaTaskWorkerContext,
) -> Option<JoinHandle<Result<MediaTaskOutput, String>>> {
    if !should_execute_created_image_task(task) {
        tracing::info!(
            task_id = %task.task_id,
            task_type = %task.task_type,
            status = %task.normalized_status,
            reused_existing = task.reused_existing,
            "image task worker skipped created task"
        );
        return None;
    }

    let Some(workspace_root) = workspace_root_from_task(task) else {
        tracing::warn!(
            task_id = %task.task_id,
            artifact_path = %task.artifact_path,
            absolute_artifact_path = %task.absolute_artifact_path,
            "image task worker could not resolve workspace root"
        );
        return None;
    };
    let task_id = task.task_id.clone();
    tracing::info!(
        task_id = %task_id,
        workspace_root = %workspace_root.display(),
        status = %task.normalized_status,
        "image task worker spawned for created task"
    );
    Some(tokio::spawn(async move {
        let result = execute_image_task(workspace_root, task_id.clone(), &context).await;
        if let Err(error) = &result {
            tracing::warn!(task_id = %task_id, error = %error, "image task worker failed");
        }
        result
    }))
}

pub(super) fn mark_stale_running_image_task_failed_for_retry(
    workspace_root: &Path,
    task_id: &str,
) -> Result<MediaTaskOutput, String> {
    let message = "图片 worker 运行租约已过期，正在恢复任务。".to_string();
    let error = TaskErrorRecord {
        code: "image_worker_stale_running_recovered".to_string(),
        message: message.clone(),
        retryable: true,
        stage: Some("worker_recovery".to_string()),
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
            current_attempt_worker_id: Some(Some(IMAGE_TASK_RUNNER_WORKER_ID.to_string())),
            ..TaskArtifactPatch::default()
        },
    )
    .map_err(|error| error.to_string())
}

pub(super) fn spawn_image_task_worker_for_existing_task(
    workspace_root: &Path,
    task: &MediaTaskArtifactResponse,
    context: MediaTaskWorkerContext,
) -> Option<JoinHandle<Result<MediaTaskOutput, String>>> {
    if !should_execute_pending_image_task(task) {
        tracing::info!(
            task_id = %task.task_id,
            task_type = %task.task_type,
            status = %task.normalized_status,
            "image task worker skipped existing task"
        );
        return None;
    }

    let workspace_root = workspace_root.to_path_buf();
    let task_id = task.task_id.clone();
    tracing::info!(
        task_id = %task_id,
        workspace_root = %workspace_root.display(),
        status = %task.normalized_status,
        "image task worker spawned for existing task"
    );
    Some(tokio::spawn(async move {
        let result = execute_image_task(workspace_root, task_id.clone(), &context).await;
        if let Err(error) = &result {
            tracing::warn!(task_id = %task_id, error = %error, "image task worker failed");
        }
        result
    }))
}

pub(super) async fn execute_image_task(
    workspace_root: PathBuf,
    task_id: String,
    context: &MediaTaskWorkerContext,
) -> Result<MediaTaskOutput, String> {
    tracing::info!(
        task_id = %task_id,
        workspace_root = %workspace_root.display(),
        "image task worker resolving runner config"
    );
    match image_generation_runner_config_from_resolved_route(&workspace_root, &task_id, context) {
        Ok(Some(runner_config)) => {
            tracing::info!(
                task_id = %task_id,
                endpoint = %runner_config.endpoint,
                request_body_format = %runner_config.request_body_format.as_str(),
                "image task worker using resolved route"
            );
            return execute_image_task_with_runner_config_and_sidecar(
                workspace_root,
                task_id,
                runner_config,
                context.sidecar_store.as_deref(),
            )
            .await;
        }
        Ok(None) => {
            return mark_image_task_worker_start_failed(
                &workspace_root,
                &task_id,
                "图片任务缺少完整 resolved route，请重新创建任务。".to_string(),
            );
        }
        Err(error) => {
            tracing::warn!(
                task_id = %task_id,
                error = %error,
                "failed to resolve image task route runner config"
            );
            return mark_image_task_worker_start_failed(&workspace_root, &task_id, error);
        }
    }
}

#[cfg(test)]
async fn execute_image_task_with_runner_config(
    workspace_root: PathBuf,
    task_id: String,
    runner_config: ImageGenerationRunnerConfig,
) -> Result<MediaTaskOutput, String> {
    execute_image_task_with_runner_config_and_sidecar(workspace_root, task_id, runner_config, None)
        .await
}

async fn execute_image_task_with_runner_config_and_sidecar(
    workspace_root: PathBuf,
    task_id: String,
    runner_config: ImageGenerationRunnerConfig,
    sidecar_store: Option<&crate::runtime::SidecarStore>,
) -> Result<MediaTaskOutput, String> {
    let output =
        execute_image_generation_task(Path::new(&workspace_root), &task_id, &runner_config)
            .await
            .map_err(|error| error.to_string())?;
    let output = crate::media_task::attach_image_result_sidecar_refs(
        Path::new(&workspace_root),
        &task_id,
        output,
        sidecar_store,
    )
    .await?;
    tracing::info!(
        task_id = %task_id,
        status = %output.normalized_status,
        attempt_count = output.attempt_count,
        "image task worker finished"
    );
    Ok(output)
}

fn workspace_root_from_task(task: &MediaTaskArtifactResponse) -> Option<PathBuf> {
    let absolute_path = PathBuf::from(task.absolute_artifact_path.trim());
    let task_relative_path = task.artifact_path.trim();
    if absolute_path.is_absolute() && !task_relative_path.is_empty() {
        let relative_path = Path::new(task_relative_path);
        if !relative_path.is_absolute() {
            let mut workspace_root = absolute_path.clone();
            for _ in relative_path.components() {
                workspace_root.pop();
            }
            if !workspace_root.as_os_str().is_empty() {
                return Some(workspace_root);
            }
        }
    }

    task.record
        .get("payload")
        .and_then(|value| value.get("project_root_path"))
        .or_else(|| {
            task.record
                .get("payload")
                .and_then(|value| value.get("projectRootPath"))
        })
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn mark_image_task_worker_start_failed(
    workspace_root: &Path,
    task_id: &str,
    message: String,
) -> Result<MediaTaskOutput, String> {
    let error = TaskErrorRecord {
        code: "image_worker_start_failed".to_string(),
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
            current_attempt_worker_id: Some(Some(IMAGE_TASK_RUNNER_WORKER_ID.to_string())),
            ..TaskArtifactPatch::default()
        },
    )
    .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media_task::{
        cancel_media_task_artifact, complete_image_generation_task_artifact,
        create_image_generation_task_artifact,
    };
    use crate::runtime::sidecar_store::SidecarStore;
    use app_server_protocol::{
        MediaTaskArtifactImageCompleteParams, MediaTaskArtifactImageCreateParams,
        MediaTaskArtifactLookupParams,
    };
    use chrono::{Duration, Utc};
    use lime_core::database::dao::api_key_provider::ApiProviderType;
    use lime_core::database::schema::create_tables;
    use lime_media_runtime::{load_task_output, update_task_status, IMAGE_TASK_RUNNER_WORKER_ID};
    use lime_services::api_key_provider_service::ApiKeyProviderService;
    use rusqlite::Connection;
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::thread;

    fn test_db() -> DbConnection {
        let conn = Connection::open_in_memory().expect("open in-memory db");
        create_tables(&conn).expect("create schema");
        Arc::new(Mutex::new(conn))
    }

    #[test]
    fn should_execute_created_image_task_ignores_reused_existing_tasks() {
        let task = MediaTaskArtifactResponse {
            task_type: "image_generate".to_string(),
            normalized_status: "pending".to_string(),
            reused_existing: true,
            ..MediaTaskArtifactResponse::default()
        };

        assert!(!should_execute_created_image_task(&task));
    }

    #[tokio::test]
    async fn workspace_recovery_scans_only_pending_image_tasks() {
        let workspace = tempfile::tempdir().expect("workspace");
        let pending = create_image_generation_task_artifact(
            MediaTaskArtifactImageCreateParams {
                project_root_path: workspace.path().to_string_lossy().to_string(),
                prompt: "生成待恢复的青柠主视觉".to_string(),
                count: Some(1),
                provider_id: Some("fal".to_string()),
                model: Some("fal-ai/nano-banana-pro".to_string()),
                executor_mode: Some("images_api".to_string()),
                ..MediaTaskArtifactImageCreateParams::default()
            },
            None,
        )
        .expect("create pending image task");
        let cancelled = create_image_generation_task_artifact(
            MediaTaskArtifactImageCreateParams {
                project_root_path: workspace.path().to_string_lossy().to_string(),
                prompt: "生成随后取消的青柠主视觉".to_string(),
                count: Some(1),
                provider_id: Some("fal".to_string()),
                model: Some("fal-ai/nano-banana-pro".to_string()),
                executor_mode: Some("images_api".to_string()),
                ..MediaTaskArtifactImageCreateParams::default()
            },
            None,
        )
        .expect("create cancelled image task");
        cancel_media_task_artifact(MediaTaskArtifactLookupParams {
            project_root_path: workspace.path().to_string_lossy().to_string(),
            task_ref: cancelled.task_id,
        })
        .expect("cancel image task");
        let failed = create_image_generation_task_artifact(
            MediaTaskArtifactImageCreateParams {
                project_root_path: workspace.path().to_string_lossy().to_string(),
                prompt: "生成随后失败的青柠主视觉".to_string(),
                count: Some(1),
                provider_id: Some("fal".to_string()),
                model: Some("fal-ai/nano-banana-pro".to_string()),
                executor_mode: Some("images_api".to_string()),
                ..MediaTaskArtifactImageCreateParams::default()
            },
            None,
        )
        .expect("create failed image task");
        complete_image_generation_task_artifact(
            MediaTaskArtifactImageCompleteParams {
                project_root_path: workspace.path().to_string_lossy().to_string(),
                task_ref: failed.task_id,
                status: Some("failed".to_string()),
                failures: vec![serde_json::json!({
                    "code": "local_image_server_unavailable",
                    "message": "本地图片服务不可用",
                    "retryable": true,
                    "stage": "execute"
                })],
                provider_id: Some("fal".to_string()),
                model: Some("fal-ai/nano-banana-pro".to_string()),
                executor_mode: Some("images_api".to_string()),
                ..MediaTaskArtifactImageCompleteParams::default()
            },
            None,
        )
        .await
        .expect("mark failed image task");
        let running = create_image_generation_task_artifact(
            MediaTaskArtifactImageCreateParams {
                project_root_path: workspace.path().to_string_lossy().to_string(),
                prompt: "生成已经由 worker 接手的青柠主视觉".to_string(),
                count: Some(1),
                provider_id: Some("fal".to_string()),
                model: Some("fal-ai/nano-banana-pro".to_string()),
                executor_mode: Some("images_api".to_string()),
                ..MediaTaskArtifactImageCreateParams::default()
            },
            None,
        )
        .expect("create running image task");
        update_task_status(workspace.path(), &running.task_id, None, "running")
            .expect("mark running image task");

        let recoverable = list_image_tasks_for_workspace(workspace.path(), None)
            .expect("scan tasks")
            .into_iter()
            .filter(should_execute_pending_image_task)
            .collect::<Vec<_>>();

        assert_eq!(recoverable.len(), 1);
        assert_eq!(recoverable[0].task_id, pending.task_id);
        assert_eq!(recoverable[0].normalized_status, "pending");
    }

    #[tokio::test]
    async fn retryable_failed_image_task_reuses_same_task_with_new_attempt() {
        let workspace = tempfile::tempdir().expect("workspace");
        let failed = create_image_generation_task_artifact(
            MediaTaskArtifactImageCreateParams {
                project_root_path: workspace.path().to_string_lossy().to_string(),
                prompt: "生成可重试的青柠主视觉".to_string(),
                count: Some(1),
                provider_id: Some("fal".to_string()),
                model: Some("fal-ai/nano-banana-pro".to_string()),
                executor_mode: Some("images_api".to_string()),
                ..MediaTaskArtifactImageCreateParams::default()
            },
            None,
        )
        .expect("create failed image task");
        let failed = complete_image_generation_task_artifact(
            MediaTaskArtifactImageCompleteParams {
                project_root_path: workspace.path().to_string_lossy().to_string(),
                task_ref: failed.task_id.clone(),
                status: Some("failed".to_string()),
                failures: vec![serde_json::json!({
                    "code": "local_image_server_unavailable",
                    "message": "本地图片服务不可用",
                    "retryable": true,
                    "stage": "execute"
                })],
                provider_id: Some("fal".to_string()),
                model: Some("fal-ai/nano-banana-pro".to_string()),
                executor_mode: Some("images_api".to_string()),
                ..MediaTaskArtifactImageCompleteParams::default()
            },
            None,
        )
        .await
        .expect("mark failed image task");

        let retried = retry_failed_image_task_for_worker(workspace.path(), &failed.task_id)
            .expect("retry failed image task")
            .expect("retryable image task");

        assert_eq!(retried.task_id, failed.task_id);
        assert_eq!(retried.normalized_status, "pending");
        assert_eq!(retried.record["retry_count"].as_u64(), Some(1));
        assert_eq!(
            retried.record["relationships"]["derived_from_attempt_id"],
            failed.record["current_attempt_id"]
        );
        assert_eq!(
            retried
                .record
                .get("attempts")
                .and_then(serde_json::Value::as_array)
                .map(Vec::len),
            Some(2)
        );
    }

    #[test]
    fn stale_running_image_task_reuses_same_task_with_new_attempt() {
        let workspace = tempfile::tempdir().expect("workspace");
        let running = create_image_generation_task_artifact(
            MediaTaskArtifactImageCreateParams {
                project_root_path: workspace.path().to_string_lossy().to_string(),
                prompt: "生成悬挂中的青柠主视觉".to_string(),
                count: Some(1),
                provider_id: Some("fal".to_string()),
                model: Some("fal-ai/nano-banana-pro".to_string()),
                executor_mode: Some("images_api".to_string()),
                ..MediaTaskArtifactImageCreateParams::default()
            },
            None,
        )
        .expect("create running image task");
        let running = patch_task_artifact(
            workspace.path(),
            &running.task_id,
            None,
            TaskArtifactPatch {
                status: Some("running".to_string()),
                current_attempt_worker_id: Some(Some(IMAGE_TASK_RUNNER_WORKER_ID.to_string())),
                ..TaskArtifactPatch::default()
            },
        )
        .expect("mark task running");
        let old_started_at = (Utc::now()
            - Duration::seconds(IMAGE_TASK_WORKER_STALE_RUNNING_SECS + 60))
        .to_rfc3339();
        force_task_running_started_at(&running.absolute_path, &old_started_at);

        let retried =
            recover_stale_running_image_task_for_worker(workspace.path(), &running.task_id)
                .expect("recover stale running image task")
                .expect("stale running task should retry");

        assert_eq!(retried.task_id, running.task_id);
        assert_eq!(retried.normalized_status, "pending");
        assert_eq!(retried.record["retry_count"].as_u64(), Some(1));
        assert_eq!(
            retried.record["relationships"]["derived_from_attempt_id"].as_str(),
            running.current_attempt_id.as_deref()
        );
        assert_eq!(
            retried
                .record
                .get("attempts")
                .and_then(serde_json::Value::as_array)
                .map(Vec::len),
            Some(2)
        );
    }

    #[test]
    fn fresh_running_image_task_is_not_stale_recovered() {
        let now = Utc::now();
        let task = MediaTaskArtifactResponse {
            task_type: "image_generate".to_string(),
            normalized_status: "running".to_string(),
            record: serde_json::json!({
                "current_attempt_id": "attempt-current",
                "attempts": [{
                    "attempt_id": "attempt-current",
                    "status": "running",
                    "started_at": now.to_rfc3339(),
                    "worker_id": IMAGE_TASK_RUNNER_WORKER_ID
                }]
            }),
            ..MediaTaskArtifactResponse::default()
        };

        assert!(!should_recover_stale_running_image_task(&task, now));
    }

    fn force_task_running_started_at(task_path: &str, started_at: &str) {
        let raw = fs::read_to_string(task_path).expect("read task artifact");
        let mut value: serde_json::Value = serde_json::from_str(&raw).expect("parse task json");
        value["started_at"] = serde_json::json!(started_at);
        value["updated_at"] = serde_json::json!(started_at);
        if let Some(attempt) = value
            .get_mut("attempts")
            .and_then(serde_json::Value::as_array_mut)
            .and_then(|attempts| attempts.last_mut())
        {
            attempt["started_at"] = serde_json::json!(started_at);
            attempt["worker_id"] = serde_json::json!(IMAGE_TASK_RUNNER_WORKER_ID);
        }
        fs::write(
            task_path,
            serde_json::to_vec_pretty(&value).expect("serialize task json"),
        )
        .expect("write task artifact");
    }

    #[tokio::test]
    async fn executes_created_image_task_with_existing_media_runtime_worker() {
        let workspace = tempfile::tempdir().expect("workspace");
        let image_server = SingleImageGenerationServer::start();
        let created = create_image_generation_task_artifact(
            MediaTaskArtifactImageCreateParams {
                project_root_path: workspace.path().to_string_lossy().to_string(),
                prompt: "生成青柠科技主视觉".to_string(),
                count: Some(1),
                provider_id: Some("fal".to_string()),
                model: Some("fal-ai/nano-banana-pro".to_string()),
                executor_mode: Some("images_api".to_string()),
                ..MediaTaskArtifactImageCreateParams::default()
            },
            None,
        )
        .expect("create image task");
        let runner_config = ImageGenerationRunnerConfig {
            endpoint: format!("http://{}/v1/images/generations", image_server.address),
            api_key: "test-key".to_string(),
            request_body_format: Default::default(),
        };

        let result = execute_image_task_with_runner_config(
            workspace.path().to_path_buf(),
            created.task_id.clone(),
            runner_config,
        )
        .await
        .expect("execute image task");

        assert_eq!(result.normalized_status, "succeeded");
        assert_eq!(
            result
                .record
                .attempts
                .last()
                .and_then(|attempt| attempt.worker_id.as_deref()),
            Some(IMAGE_TASK_RUNNER_WORKER_ID)
        );
        assert_eq!(image_server.join(), 1);

        let persisted = load_task_output(workspace.path(), &created.task_id, None)
            .expect("load persisted task");
        assert_eq!(persisted.normalized_status, "succeeded");
    }

    #[tokio::test]
    async fn execute_image_task_writes_worker_output_to_media_sidecar() {
        let workspace = tempfile::tempdir().expect("workspace");
        let sidecar_root = tempfile::tempdir().expect("sidecar root");
        let sidecar_store =
            Arc::new(SidecarStore::new(sidecar_root.path()).expect("sidecar store"));
        let image_server = SingleImageGenerationServer::start();
        let created = create_image_generation_task_artifact(
            MediaTaskArtifactImageCreateParams {
                project_root_path: workspace.path().to_string_lossy().to_string(),
                prompt: "生成由 worker 写入 sidecar 的青柠主视觉".to_string(),
                count: Some(1),
                provider_id: Some("fal".to_string()),
                model: Some("fal-ai/nano-banana-pro".to_string()),
                executor_mode: Some("images_api".to_string()),
                session_id: Some("session-worker-sidecar".to_string()),
                ..MediaTaskArtifactImageCreateParams::default()
            },
            None,
        )
        .expect("create image task");
        let runner_config = ImageGenerationRunnerConfig {
            endpoint: format!("http://{}/v1/images/generations", image_server.address),
            api_key: "test-key".to_string(),
            request_body_format: Default::default(),
        };

        let result = execute_image_task_with_runner_config_and_sidecar(
            workspace.path().to_path_buf(),
            created.task_id.clone(),
            runner_config,
            Some(sidecar_store.as_ref()),
        )
        .await
        .expect("execute image task with sidecar");

        assert_eq!(result.normalized_status, "succeeded");
        let sidecar_ref =
            &result.record.result.as_ref().expect("result")["images"][0]["sidecarRef"];
        assert_eq!(sidecar_ref["kind"].as_str(), Some("media"));
        assert_eq!(sidecar_ref["mimeType"].as_str(), Some("image/png"));
        assert!(sidecar_ref["ref"]
            .as_str()
            .is_some_and(|value| value.starts_with("sidecar://media/")));
        let relative_path = sidecar_ref["relativePath"].as_str().expect("relative path");
        let sha256 = sidecar_ref["sha256"].as_str();
        let bytes = sidecar_store
            .read_bytes_verified(relative_path, sha256, 64)
            .expect("read sidecar bytes")
            .expect("sidecar bytes");
        assert_eq!(bytes.bytes, b"fake-lime-image".to_vec());
        assert_eq!(
            result
                .record
                .attempts
                .last()
                .and_then(|attempt| attempt.result_snapshot.as_ref())
                .and_then(|snapshot| snapshot.pointer("/images/0/sidecarRef/ref"))
                .and_then(serde_json::Value::as_str),
            sidecar_ref["ref"].as_str()
        );
        assert_eq!(image_server.join(), 1);
    }

    #[tokio::test]
    async fn execute_image_task_rejects_route_less_task_without_reading_provider_key() {
        let workspace = tempfile::tempdir().expect("workspace");
        let db = test_db();
        let service = ApiKeyProviderService::new();
        let provider = service
            .add_custom_provider(
                &db,
                "Provider Store Images".to_string(),
                ApiProviderType::NewApi,
                "http://127.0.0.1:9".to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .expect("create provider");
        service
            .add_api_key(&db, &provider.id, "provider-db-key", None, true)
            .expect("add provider key");
        let created = create_image_generation_task_artifact(
            MediaTaskArtifactImageCreateParams {
                project_root_path: workspace.path().to_string_lossy().to_string(),
                prompt: "从 Provider DB 恢复执行图片任务".to_string(),
                count: Some(1),
                provider_id: Some(provider.id.clone()),
                model: Some("gpt-image-1".to_string()),
                executor_mode: Some("images_api".to_string()),
                ..MediaTaskArtifactImageCreateParams::default()
            },
            None,
        )
        .expect("create image task");

        let result = execute_image_task(
            workspace.path().to_path_buf(),
            created.task_id.clone(),
            &MediaTaskWorkerContext::new(db.clone()),
        )
        .await
        .expect("mark route-less image task failed");

        assert_eq!(result.normalized_status, "failed");
        assert_eq!(
            result.last_error.as_ref().map(|error| error.code.as_str()),
            Some("image_worker_start_failed")
        );
        assert!(result
            .last_error
            .as_ref()
            .is_some_and(|error| error.message.contains("resolved route")));
        let persisted_provider = service
            .get_provider(&db, &provider.id)
            .expect("read provider")
            .expect("provider");
        assert!(
            persisted_provider
                .api_keys
                .iter()
                .all(|key| key.usage_count == 0),
            "route-less task must not select or consume a provider key"
        );
    }

    #[tokio::test]
    async fn execute_image_task_fails_closed_without_provider_route() {
        let workspace = tempfile::tempdir().expect("workspace");
        let db = test_db();
        let created = create_image_generation_task_artifact(
            MediaTaskArtifactImageCreateParams {
                project_root_path: workspace.path().to_string_lossy().to_string(),
                prompt: "缺少 Provider 的图片任务".to_string(),
                count: Some(1),
                ..MediaTaskArtifactImageCreateParams::default()
            },
            None,
        )
        .expect("create image task");

        let result = execute_image_task(
            workspace.path().to_path_buf(),
            created.task_id.clone(),
            &MediaTaskWorkerContext::new(db),
        )
        .await
        .expect("mark task failed");

        assert_eq!(result.normalized_status, "failed");
        assert_eq!(
            result.last_error.as_ref().map(|error| error.code.as_str()),
            Some("image_worker_start_failed")
        );
        assert_eq!(
            result.last_error.as_ref().map(|error| error.retryable),
            Some(false)
        );
    }

    struct SingleImageGenerationServer {
        address: std::net::SocketAddr,
        handle: thread::JoinHandle<usize>,
    }

    impl SingleImageGenerationServer {
        fn start() -> Self {
            let (listener, address) = Self::bind();
            Self::start_on(listener, address, "test-key", "fal")
        }

        fn bind() -> (TcpListener, std::net::SocketAddr) {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind image api");
            let address = listener.local_addr().expect("local address");
            (listener, address)
        }

        fn start_on(
            listener: TcpListener,
            address: std::net::SocketAddr,
            expected_api_key: impl Into<String>,
            expected_provider_id: impl Into<String>,
        ) -> Self {
            let expected_api_key = expected_api_key.into().to_ascii_lowercase();
            let expected_provider_id = expected_provider_id.into().to_ascii_lowercase();
            let handle = thread::spawn(move || {
                let (mut stream, _) = listener.accept().expect("accept image request");
                let mut buffer = [0_u8; 4096];
                let bytes_read = stream.read(&mut buffer).expect("read image request");
                let request = String::from_utf8_lossy(&buffer[..bytes_read]);
                let normalized_request = request.to_ascii_lowercase();
                assert!(request.starts_with("POST /v1/images/generations "));
                assert!(normalized_request
                    .contains(&format!("authorization: bearer {expected_api_key}")));
                assert!(
                    normalized_request.contains(&format!("x-provider-id: {expected_provider_id}"))
                );
                let body = r#"{"created":1,"data":[{"b64_json":"ZmFrZS1saW1lLWltYWdl","revised_prompt":"fixture revised prompt"}]}"#;
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream
                    .write_all(response.as_bytes())
                    .expect("write image response");
                1
            });

            Self { address, handle }
        }

        fn join(self) -> usize {
            self.handle.join().expect("image server join")
        }
    }
}
