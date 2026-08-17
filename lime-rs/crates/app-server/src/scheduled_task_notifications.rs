use crate::local_data_source::is_scheduled_task_job;
use app_server_protocol::protocol::v2::ServerNotification;
use app_server_protocol::{
    ScheduledTask, ScheduledTaskChange, ScheduledTaskChangedNotification,
    ScheduledTaskNotificationPolicy, ScheduledTaskRunSummary, ScheduledTaskRunUpdatedNotification,
};
use lime_core::database;
use lime_core::database::dao::agent_run::{AgentRun, AgentRunDao};
use lime_core::database::dao::automation_job::{AutomationJob, AutomationJobDao};
use lime_core::database::DbConnection;
use serde_json::Value;

pub(crate) fn changed_notification(
    task_id: impl Into<String>,
    change: ScheduledTaskChange,
) -> ServerNotification {
    ServerNotification::ScheduledTaskChanged(ScheduledTaskChangedNotification {
        task_id: task_id.into(),
        change,
    })
}

pub(crate) fn run_notification_from_projection(
    task: &ScheduledTask,
    run: &ScheduledTaskRunSummary,
) -> Option<ServerNotification> {
    build_run_notification(
        task.id.as_str(),
        task.title.as_str(),
        task.notification_policy,
        run.id.as_str(),
        run.status.as_str(),
        run.thread_id.clone(),
        run.turn_id.clone(),
        run.error.clone(),
    )
}

pub(crate) fn load_run_notification(
    db: &DbConnection,
    task_id: &str,
    run_id: &str,
) -> Result<Option<ServerNotification>, String> {
    let conn = database::lock_db(db)?;
    let Some(job) = AutomationJobDao::get_including_deleted(&conn, task_id)
        .map_err(|error| error.to_string())?
    else {
        return Ok(None);
    };
    if !is_scheduled_task_job(&job) {
        return Ok(None);
    }
    let Some(run) = AgentRunDao::get_run(&conn, run_id).map_err(|error| error.to_string())? else {
        return Ok(None);
    };
    if run.source != "automation" || run.source_ref.as_deref() != Some(task_id) {
        return Ok(None);
    }
    run_notification_from_records(&job, &run)
}

fn run_notification_from_records(
    job: &AutomationJob,
    run: &AgentRun,
) -> Result<Option<ServerNotification>, String> {
    let policy = job
        .payload
        .get("notification_policy")
        .cloned()
        .ok_or_else(|| "已安排任务缺少 notification_policy".to_string())
        .and_then(|value| serde_json::from_value(value).map_err(|error| error.to_string()))?;
    let metadata = run
        .metadata
        .as_deref()
        .and_then(|value| serde_json::from_str::<Value>(value).ok());
    Ok(build_run_notification(
        job.id.as_str(),
        job.name.as_str(),
        policy,
        run.id.as_str(),
        run.status.as_str(),
        metadata_string(&metadata, &["threadId", "thread_id"]),
        metadata_string(&metadata, &["turnId", "turn_id"]),
        run.error_message.clone(),
    ))
}

#[allow(clippy::too_many_arguments)]
fn build_run_notification(
    task_id: &str,
    title: &str,
    policy: ScheduledTaskNotificationPolicy,
    run_id: &str,
    status: &str,
    thread_id: Option<String>,
    turn_id: Option<String>,
    error: Option<String>,
) -> Option<ServerNotification> {
    if !matches!(
        status,
        "success" | "error" | "canceled" | "timeout" | "missed"
    ) {
        return None;
    }
    let attention = matches!(status, "error" | "timeout" | "missed");
    Some(ServerNotification::ScheduledTaskRunUpdated(
        ScheduledTaskRunUpdatedNotification {
            task_id: task_id.to_string(),
            run_id: run_id.to_string(),
            status: status.to_string(),
            attention,
            notification_policy: notification_policy_value(policy).to_string(),
            title: Some(title.to_string()),
            thread_id,
            turn_id,
            error,
        },
    ))
}

fn notification_policy_value(policy: ScheduledTaskNotificationPolicy) -> &'static str {
    match policy {
        ScheduledTaskNotificationPolicy::AllRuns => "all_runs",
        ScheduledTaskNotificationPolicy::Failures => "failures",
        ScheduledTaskNotificationPolicy::None => "none",
    }
}

fn metadata_string(metadata: &Option<Value>, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        metadata
            .as_ref()
            .and_then(|value| value.get(key))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::{
        ScheduledTaskExecution, ScheduledTaskOverlapPolicy, ScheduledTaskSchedule,
        ScheduledTaskThreadMode,
    };

    fn task(policy: ScheduledTaskNotificationPolicy) -> ScheduledTask {
        ScheduledTask {
            id: "task-1".to_string(),
            title: "每日简报".to_string(),
            prompt: "整理进展".to_string(),
            enabled: true,
            schedule: ScheduledTaskSchedule::Hourly {
                interval_hours: 1,
                days: None,
                minute: 0,
                timezone: "Asia/Shanghai".to_string(),
            },
            execution: ScheduledTaskExecution {
                thread_mode: ScheduledTaskThreadMode::NewThread,
                source_thread_id: None,
                project_id: None,
                cwd: None,
                model_id: None,
                reasoning_effort: None,
                approval_policy: None,
                sandbox_policy: None,
                request_metadata: None,
            },
            notification_policy: policy,
            overlap_policy: ScheduledTaskOverlapPolicy::SkipIfRunning,
            next_run_at: None,
            last_run_summary: None,
            created_at: "2026-08-17T00:00:00Z".to_string(),
            updated_at: "2026-08-17T00:00:00Z".to_string(),
        }
    }

    fn run(status: &str) -> ScheduledTaskRunSummary {
        ScheduledTaskRunSummary {
            id: "run-1".to_string(),
            task_id: "task-1".to_string(),
            status: status.to_string(),
            scheduled_for: None,
            started_at: Some("2026-08-17T00:00:00Z".to_string()),
            finished_at: Some("2026-08-17T00:00:01Z".to_string()),
            session_id: Some("session-1".to_string()),
            thread_id: Some("thread-1".to_string()),
            turn_id: Some("turn-1".to_string()),
            summary: None,
            error: None,
        }
    }

    #[test]
    fn terminal_projection_preserves_policy_and_attention() {
        let notification = run_notification_from_projection(
            &task(ScheduledTaskNotificationPolicy::Failures),
            &run("error"),
        )
        .expect("terminal notification");
        let ServerNotification::ScheduledTaskRunUpdated(notification) = notification else {
            panic!("expected scheduled task run notification");
        };
        assert!(notification.attention);
        assert_eq!(notification.notification_policy, "failures");
        assert_eq!(notification.thread_id.as_deref(), Some("thread-1"));
    }

    #[test]
    fn non_terminal_projection_is_not_published() {
        assert!(run_notification_from_projection(
            &task(ScheduledTaskNotificationPolicy::AllRuns),
            &run("running"),
        )
        .is_none());
    }
}
