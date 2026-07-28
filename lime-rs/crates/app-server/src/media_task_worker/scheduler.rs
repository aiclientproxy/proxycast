use super::{
    mark_stale_running_image_task_failed_for_retry, should_execute_pending_video_task,
    should_recover_stale_running_video_task, spawn_image_task_worker_for_existing_task,
    spawn_video_task_worker_for_existing_task, MediaTaskWorkerContext,
};
use app_server_protocol::{
    MediaTaskArtifactListParams, MediaTaskArtifactLookupParams, MediaTaskArtifactResponse,
};
use chrono::{DateTime, Duration, Utc};
use lime_core::database;
use lime_media_runtime::{MediaTaskOutput, MediaTaskType, IMAGE_TASK_RUNNER_WORKER_ID};
use std::path::{Path, PathBuf};
use std::time::Duration as StdDuration;
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;

const IMAGE_TASK_WORKER_MAX_RECOVERY_RETRIES: u64 = 1;
const IMAGE_TASK_WORKER_DEFAULT_SCAN_INTERVAL_SECS: u64 = 30;
const IMAGE_TASK_WORKER_DEFAULT_SCAN_LIMIT_PER_WORKSPACE: usize = 8;
pub(super) const IMAGE_TASK_WORKER_STALE_RUNNING_SECS: i64 = 10 * 60;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MediaTaskWorkerSchedulerConfig {
    pub(crate) scan_interval: StdDuration,
    pub(crate) per_workspace_limit: usize,
}

impl Default for MediaTaskWorkerSchedulerConfig {
    fn default() -> Self {
        Self {
            scan_interval: StdDuration::from_secs(IMAGE_TASK_WORKER_DEFAULT_SCAN_INTERVAL_SECS),
            per_workspace_limit: IMAGE_TASK_WORKER_DEFAULT_SCAN_LIMIT_PER_WORKSPACE,
        }
    }
}

pub(crate) fn spawn_media_task_worker_scheduler(context: MediaTaskWorkerContext) -> JoinHandle<()> {
    spawn_media_task_worker_scheduler_with_config(
        context,
        MediaTaskWorkerSchedulerConfig::default(),
    )
}

pub(crate) fn spawn_media_task_worker_scheduler_with_config(
    context: MediaTaskWorkerContext,
    config: MediaTaskWorkerSchedulerConfig,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        run_media_task_worker_scheduler(context, config).await;
    })
}

async fn run_media_task_worker_scheduler(
    context: MediaTaskWorkerContext,
    config: MediaTaskWorkerSchedulerConfig,
) {
    let mut interval = tokio::time::interval(config.scan_interval);
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);

    loop {
        interval.tick().await;
        match spawn_pending_media_task_workers_for_registered_workspaces(
            &context,
            config.per_workspace_limit,
        ) {
            Ok(handles) if !handles.is_empty() => {
                tracing::info!(
                    spawned = handles.len(),
                    "scheduled pending media task workers"
                );
            }
            Ok(_) => {}
            Err(error) => {
                tracing::warn!(error = %error, "failed to schedule pending media task workers");
            }
        }
    }
}

pub(super) fn spawn_pending_media_task_workers_for_registered_workspaces(
    context: &MediaTaskWorkerContext,
    per_workspace_limit: usize,
) -> Result<Vec<JoinHandle<Result<MediaTaskOutput, String>>>, String> {
    let workspace_roots = list_registered_workspace_roots(context)?;
    let mut handles = Vec::new();
    for workspace_root in workspace_roots {
        match spawn_pending_image_task_workers_for_workspace(
            &workspace_root,
            Some(per_workspace_limit),
            context.clone(),
        ) {
            Ok(mut workspace_handles) => handles.append(&mut workspace_handles),
            Err(error) => {
                tracing::warn!(
                    workspace_root = %workspace_root.display(),
                    error = %error,
                    "failed to scan image tasks for workspace"
                );
            }
        }
        match spawn_pending_video_task_workers_for_workspace(
            &workspace_root,
            Some(per_workspace_limit),
            context.clone(),
        ) {
            Ok(mut workspace_handles) => handles.append(&mut workspace_handles),
            Err(error) => {
                tracing::warn!(
                    workspace_root = %workspace_root.display(),
                    error = %error,
                    "failed to scan video tasks for workspace"
                );
            }
        }
    }
    Ok(handles)
}

fn list_registered_workspace_roots(
    context: &MediaTaskWorkerContext,
) -> Result<Vec<PathBuf>, String> {
    let conn = database::lock_db(&context.db)
        .map_err(|error| format!("读取 workspace 数据库失败: {error}"))?;
    let mut stmt = conn
        .prepare(
            "SELECT root_path
             FROM workspaces
             WHERE COALESCE(is_archived, 0) = 0
             ORDER BY updated_at DESC",
        )
        .map_err(|error| format!("读取 workspace 列表失败: {error}"))?;
    let roots = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|error| format!("读取 workspace 根目录失败: {error}"))?
        .filter_map(|row| match row {
            Ok(root) => {
                let root = root.trim();
                if root.is_empty() {
                    None
                } else {
                    Some(Ok(PathBuf::from(root)))
                }
            }
            Err(error) => Some(Err(format!("读取 workspace 根目录失败: {error}"))),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(roots)
}

pub(crate) fn spawn_pending_image_task_workers_for_workspace(
    workspace_root: impl AsRef<Path>,
    limit: Option<usize>,
    context: MediaTaskWorkerContext,
) -> Result<Vec<JoinHandle<Result<MediaTaskOutput, String>>>, String> {
    let workspace_root = workspace_root.as_ref();
    let tasks = list_image_tasks_for_workspace(workspace_root, limit)?;
    let mut handles = Vec::new();
    for task in tasks {
        if let Some(handle) =
            spawn_image_task_worker_for_existing_task(workspace_root, &task, context.clone())
        {
            handles.push(handle);
            continue;
        }
        if should_recover_stale_running_image_task(&task, Utc::now()) {
            if let Some(handle) = spawn_stale_running_image_task_worker(
                workspace_root,
                &task.task_id,
                context.clone(),
            )? {
                handles.push(handle);
            }
            continue;
        }
        if !should_retry_failed_image_task(&task) {
            continue;
        }
        if let Some(handle) = spawn_retryable_failed_image_task_worker(
            workspace_root,
            &task.task_id,
            context.clone(),
        )? {
            handles.push(handle);
        }
    }
    Ok(handles)
}

pub(super) fn spawn_pending_video_task_workers_for_workspace(
    workspace_root: impl AsRef<Path>,
    limit: Option<usize>,
    context: MediaTaskWorkerContext,
) -> Result<Vec<JoinHandle<Result<MediaTaskOutput, String>>>, String> {
    let workspace_root = workspace_root.as_ref();
    let tasks = list_video_tasks_for_workspace(workspace_root, limit)?;
    let mut handles = Vec::new();
    for task in tasks {
        if should_execute_pending_video_task(&task)
            || should_recover_stale_running_video_task(&task, Utc::now())
        {
            if let Some(handle) =
                spawn_video_task_worker_for_existing_task(workspace_root, &task, context.clone())
            {
                handles.push(handle);
            }
        }
    }
    Ok(handles)
}

pub(super) fn list_image_tasks_for_workspace(
    workspace_root: impl AsRef<Path>,
    limit: Option<usize>,
) -> Result<Vec<MediaTaskArtifactResponse>, String> {
    let workspace_root = workspace_root.as_ref();
    let listed = crate::media_task::list_media_task_artifacts(MediaTaskArtifactListParams {
        project_root_path: workspace_root.to_string_lossy().to_string(),
        task_type: Some(MediaTaskType::ImageGenerate.as_str().to_string()),
        limit,
        ..MediaTaskArtifactListParams::default()
    })?;
    Ok(listed.tasks)
}

pub(super) fn list_video_tasks_for_workspace(
    workspace_root: impl AsRef<Path>,
    limit: Option<usize>,
) -> Result<Vec<MediaTaskArtifactResponse>, String> {
    let workspace_root = workspace_root.as_ref();
    let listed = crate::media_task::list_media_task_artifacts(MediaTaskArtifactListParams {
        project_root_path: workspace_root.to_string_lossy().to_string(),
        task_type: Some(MediaTaskType::VideoGenerate.as_str().to_string()),
        limit,
        ..MediaTaskArtifactListParams::default()
    })?;
    Ok(listed.tasks)
}

pub(crate) fn spawn_retryable_failed_image_task_worker(
    workspace_root: impl AsRef<Path>,
    task_ref: &str,
    context: MediaTaskWorkerContext,
) -> Result<Option<JoinHandle<Result<MediaTaskOutput, String>>>, String> {
    let workspace_root = workspace_root.as_ref();
    let Some(retried) = retry_failed_image_task_for_worker(workspace_root, task_ref)? else {
        return Ok(None);
    };
    Ok(spawn_image_task_worker_for_existing_task(
        workspace_root,
        &retried,
        context,
    ))
}

pub(crate) fn spawn_stale_running_image_task_worker(
    workspace_root: impl AsRef<Path>,
    task_ref: &str,
    context: MediaTaskWorkerContext,
) -> Result<Option<JoinHandle<Result<MediaTaskOutput, String>>>, String> {
    let workspace_root = workspace_root.as_ref();
    let Some(retried) = recover_stale_running_image_task_for_worker(workspace_root, task_ref)?
    else {
        return Ok(None);
    };
    Ok(spawn_image_task_worker_for_existing_task(
        workspace_root,
        &retried,
        context,
    ))
}

pub(super) fn recover_stale_running_image_task_for_worker(
    workspace_root: impl AsRef<Path>,
    task_ref: &str,
) -> Result<Option<MediaTaskArtifactResponse>, String> {
    let workspace_root = workspace_root.as_ref();
    let current = crate::media_task::get_media_task_artifact(MediaTaskArtifactLookupParams {
        project_root_path: workspace_root.to_string_lossy().to_string(),
        task_ref: task_ref.to_string(),
    })?;
    if !should_recover_stale_running_image_task(&current, Utc::now()) {
        return Ok(None);
    }
    mark_stale_running_image_task_failed_for_retry(workspace_root, task_ref)?;
    retry_failed_image_task_for_worker(workspace_root, task_ref)
}

pub(super) fn retry_failed_image_task_for_worker(
    workspace_root: impl AsRef<Path>,
    task_ref: &str,
) -> Result<Option<MediaTaskArtifactResponse>, String> {
    let workspace_root = workspace_root.as_ref();
    let current = crate::media_task::get_media_task_artifact(MediaTaskArtifactLookupParams {
        project_root_path: workspace_root.to_string_lossy().to_string(),
        task_ref: task_ref.to_string(),
    })?;
    if !should_retry_failed_image_task(&current) {
        return Ok(None);
    }
    crate::media_task::retry_media_task_artifact(MediaTaskArtifactLookupParams {
        project_root_path: workspace_root.to_string_lossy().to_string(),
        task_ref: task_ref.to_string(),
    })
    .map(Some)
}

pub(super) fn should_execute_pending_image_task(task: &MediaTaskArtifactResponse) -> bool {
    task.task_type == MediaTaskType::ImageGenerate.as_str()
        && matches!(
            task.normalized_status.as_str(),
            "pending" | "pending_submit" | "queued"
        )
}

fn should_retry_failed_image_task(task: &MediaTaskArtifactResponse) -> bool {
    if task.task_type != MediaTaskType::ImageGenerate.as_str()
        || task.normalized_status.as_str() != "failed"
    {
        return false;
    }
    task.record
        .get("last_error")
        .and_then(|value| value.get("retryable"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
        && task
            .record
            .get("retry_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            < IMAGE_TASK_WORKER_MAX_RECOVERY_RETRIES
}

pub(super) fn should_recover_stale_running_image_task(
    task: &MediaTaskArtifactResponse,
    now: DateTime<Utc>,
) -> bool {
    if task.task_type != MediaTaskType::ImageGenerate.as_str()
        || task.normalized_status.as_str() != "running"
    {
        return false;
    }
    if current_attempt_worker_id(task).as_deref() != Some(IMAGE_TASK_RUNNER_WORKER_ID) {
        return false;
    }
    let Some(started_at) = running_started_at(task) else {
        return false;
    };
    now.signed_duration_since(started_at) >= Duration::seconds(IMAGE_TASK_WORKER_STALE_RUNNING_SECS)
}

fn current_attempt_worker_id(task: &MediaTaskArtifactResponse) -> Option<String> {
    current_attempt_value(task).and_then(|attempt| {
        attempt
            .get("worker_id")
            .or_else(|| attempt.get("workerId"))
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToString::to_string)
    })
}

fn running_started_at(task: &MediaTaskArtifactResponse) -> Option<DateTime<Utc>> {
    current_attempt_value(task)
        .and_then(|attempt| {
            timestamp_field(attempt, &["started_at", "startedAt"])
                .or_else(|| timestamp_field(attempt, &["queued_at", "queuedAt"]))
        })
        .or_else(|| timestamp_field(&task.record, &["started_at", "startedAt"]))
        .or_else(|| timestamp_field(&task.record, &["updated_at", "updatedAt"]))
        .or_else(|| timestamp_field(&task.record, &["created_at", "createdAt"]))
}

fn current_attempt_value(task: &MediaTaskArtifactResponse) -> Option<&serde_json::Value> {
    let attempts = task.record.get("attempts")?.as_array()?;
    if let Some(current_attempt_id) = task
        .record
        .get("current_attempt_id")
        .or_else(|| task.record.get("currentAttemptId"))
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        if let Some(attempt) = attempts.iter().find(|attempt| {
            attempt
                .get("attempt_id")
                .or_else(|| attempt.get("attemptId"))
                .and_then(serde_json::Value::as_str)
                == Some(current_attempt_id)
        }) {
            return Some(attempt);
        }
    }
    attempts.last()
}

fn timestamp_field(value: &serde_json::Value, keys: &[&str]) -> Option<DateTime<Utc>> {
    keys.iter()
        .find_map(|key| value.get(*key))
        .and_then(serde_json::Value::as_str)
        .and_then(parse_timestamp)
}

fn parse_timestamp(raw: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(raw.trim())
        .ok()
        .map(|value| value.with_timezone(&Utc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lime_core::database::dao::api_key_provider::ApiProviderType;
    use lime_core::database::schema::create_tables;
    use lime_services::api_key_provider_service::ApiKeyProviderService;
    use rusqlite::Connection;
    use std::sync::{Arc, Mutex};

    fn test_context() -> MediaTaskWorkerContext {
        let conn = Connection::open_in_memory().expect("open in-memory db");
        create_tables(&conn).expect("create schema");
        MediaTaskWorkerContext::new(Arc::new(Mutex::new(conn)))
    }

    #[test]
    fn scheduler_lists_registered_workspace_roots_from_db() {
        let context = test_context();
        {
            let db = database::lock_db(&context.db).expect("db");
            db.execute(
                "INSERT INTO workspaces
                 (id, name, workspace_type, root_path, is_default, settings_json,
                  created_at, updated_at, icon, color, is_favorite, is_archived,
                  tags_json, default_persona_id)
                 VALUES (?1, ?2, 'persistent', ?3, 0, '{}', 1, ?4, NULL, NULL, 0, ?5, '[]', NULL)",
                (
                    "workspace-active",
                    "Active",
                    "/tmp/active-image-workspace",
                    2_i64,
                    false,
                ),
            )
            .expect("insert active workspace");
            db.execute(
                "INSERT INTO workspaces
                 (id, name, workspace_type, root_path, is_default, settings_json,
                  created_at, updated_at, icon, color, is_favorite, is_archived,
                  tags_json, default_persona_id)
                 VALUES (?1, ?2, 'persistent', ?3, 0, '{}', 1, ?4, NULL, NULL, 0, ?5, '[]', NULL)",
                (
                    "workspace-archived",
                    "Archived",
                    "/tmp/archived-image-workspace",
                    3_i64,
                    true,
                ),
            )
            .expect("insert archived workspace");
        }

        let roots = list_registered_workspace_roots(&context).expect("workspace roots");

        assert_eq!(roots, vec![PathBuf::from("/tmp/active-image-workspace")]);
    }

    #[tokio::test]
    async fn scheduler_scan_uses_registered_workspace_and_returns_spawned_handles() {
        let context = test_context();
        let workspace = tempfile::tempdir().expect("workspace");
        {
            let db = database::lock_db(&context.db).expect("db");
            db.execute(
                "INSERT INTO workspaces
                 (id, name, workspace_type, root_path, is_default, settings_json,
                  created_at, updated_at, icon, color, is_favorite, is_archived,
                  tags_json, default_persona_id)
                 VALUES (?1, ?2, 'persistent', ?3, 0, '{}', 1, 1, NULL, NULL, 0, 0, '[]', NULL)",
                (
                    "workspace-image-worker",
                    "Image Worker",
                    workspace.path().to_string_lossy().to_string(),
                ),
            )
            .expect("insert workspace");
        }
        let service = ApiKeyProviderService::new();
        let provider = service
            .add_custom_provider(
                &context.db,
                "Provider Without Test Server".to_string(),
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
            .add_api_key(&context.db, &provider.id, "provider-db-key", None, true)
            .expect("add provider key");
        crate::media_task::create_image_generation_task_artifact(
            app_server_protocol::MediaTaskArtifactImageCreateParams {
                project_root_path: workspace.path().to_string_lossy().to_string(),
                prompt: "调度器接管待执行图片任务".to_string(),
                count: Some(1),
                provider_id: Some(provider.id),
                model: Some("gpt-image-1".to_string()),
                executor_mode: Some("images_api".to_string()),
                ..app_server_protocol::MediaTaskArtifactImageCreateParams::default()
            },
            None,
        )
        .expect("create image task");

        let handles = spawn_pending_media_task_workers_for_registered_workspaces(&context, 8)
            .expect("scan registered workspaces");

        assert_eq!(handles.len(), 1);
        handles[0].abort();
    }
}
