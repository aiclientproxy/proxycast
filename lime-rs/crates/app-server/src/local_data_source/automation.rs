use super::data_error;
use super::values_from_serializable_vec;
use crate::automation_execution::{
    apply_automation_run_finished, apply_automation_run_started, build_automation_run_start,
    build_scheduled_task_manual_run_start, next_run_for_automation_schedule,
    validate_automation_schedule_value, AutomationRunFailure, AutomationRunFinish,
    AutomationRunIdentity, AutomationRunStart,
};
use crate::RuntimeCoreError;
mod health;
use app_server_protocol::AutomationJobCreateParams;
use app_server_protocol::AutomationJobDeleteResponse;
use app_server_protocol::AutomationJobIdParams;
use app_server_protocol::AutomationJobListResponse;
use app_server_protocol::AutomationJobReadResponse;
use app_server_protocol::AutomationJobRunHistoryParams;
use app_server_protocol::AutomationJobRunHistoryResponse;
use app_server_protocol::AutomationJobUpdateParams;
use app_server_protocol::AutomationJobWriteResponse;
use app_server_protocol::AutomationScheduleParams;
use app_server_protocol::AutomationSchedulePreviewResponse;
use app_server_protocol::AutomationScheduleValidateResponse;
use app_server_protocol::AutomationSchedulerConfigReadResponse;
use app_server_protocol::AutomationSchedulerConfigUpdateParams;
use app_server_protocol::AutomationSchedulerConfigUpdateResponse;
use app_server_protocol::AutomationSchedulerStatusResponse;
mod scheduled_tasks;
use chrono::Utc;
pub(crate) use health::read_automation_health;
use lime_core::config::load_config;
use lime_core::config::save_config;
use lime_core::config::AutomationExecutionMode;
use lime_core::config::AutomationSettings;
use lime_core::config::DeliveryConfig;
use lime_core::config::TaskSchedule;
use lime_core::database;
use lime_core::database::dao::agent_run::AgentRunDao;
use lime_core::database::dao::automation_job::AutomationJob;
use lime_core::database::dao::automation_job::AutomationJobDao;
use lime_core::database::DbConnection;
pub(crate) use scheduled_tasks::{
    create_scheduled_task, delete_scheduled_task, is_scheduled_task_job, list_scheduled_task_runs,
    list_scheduled_tasks, preview_scheduled_task_schedule, read_scheduled_task,
    require_scheduled_task_job, set_scheduled_task_enabled, update_scheduled_task,
};
use serde::Deserialize;
use serde_json::json;
use serde_json::Value;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
struct AutomationSchedulerConfigRequest {
    #[serde(default)]
    enabled: bool,
    #[serde(default = "default_automation_poll_interval_secs")]
    poll_interval_secs: u64,
    #[serde(default = "default_automation_enable_history")]
    enable_history: bool,
}

#[derive(Debug, Deserialize)]
struct AutomationJobCreateRequest {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    enabled: Option<bool>,
    workspace_id: String,
    #[serde(default)]
    execution_mode: Option<AutomationExecutionMode>,
    schedule: TaskSchedule,
    payload: Value,
    #[serde(default)]
    delivery: Option<DeliveryConfig>,
    #[serde(default)]
    timeout_secs: Option<u64>,
    #[serde(default)]
    max_retries: Option<u32>,
}

#[derive(Debug, Deserialize, Default)]
struct AutomationJobUpdateRequest {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    enabled: Option<bool>,
    #[serde(default)]
    workspace_id: Option<String>,
    #[serde(default)]
    execution_mode: Option<AutomationExecutionMode>,
    #[serde(default)]
    schedule: Option<TaskSchedule>,
    #[serde(default)]
    payload: Option<Value>,
    #[serde(default)]
    delivery: Option<DeliveryConfig>,
    #[serde(default)]
    timeout_secs: Option<u64>,
    #[serde(default)]
    clear_timeout_secs: Option<bool>,
    #[serde(default)]
    max_retries: Option<u32>,
}

pub(crate) fn list_automation_jobs(
    db: &DbConnection,
) -> Result<AutomationJobListResponse, RuntimeCoreError> {
    let conn = database::lock_db(db).map_err(data_error)?;
    let jobs = AutomationJobDao::list(&conn).map_err(data_error)?;
    Ok(AutomationJobListResponse {
        jobs: values_from_serializable_vec(jobs)?,
    })
}

pub(crate) fn read_automation_scheduler_config(
) -> Result<AutomationSchedulerConfigReadResponse, RuntimeCoreError> {
    Ok(AutomationSchedulerConfigReadResponse {
        config: automation_scheduler_config_value(load_config().map_err(data_error)?.automation),
    })
}

pub(crate) fn update_automation_scheduler_config(
    params: AutomationSchedulerConfigUpdateParams,
) -> Result<AutomationSchedulerConfigUpdateResponse, RuntimeCoreError> {
    let input: AutomationSchedulerConfigRequest =
        serde_json::from_value(params.config).map_err(data_error)?;
    let mut config = load_config().map_err(data_error)?;
    config.automation = AutomationSettings {
        enabled: input.enabled,
        poll_interval_secs: input.poll_interval_secs.max(5),
        enable_history: input.enable_history,
    };
    save_config(&config).map_err(data_error)?;
    Ok(AutomationSchedulerConfigUpdateResponse {
        config: automation_scheduler_config_value(config.automation),
    })
}

pub(crate) fn read_automation_scheduler_status(
) -> Result<AutomationSchedulerStatusResponse, RuntimeCoreError> {
    Ok(AutomationSchedulerStatusResponse {
        status: json!({
            "running": false,
            "last_polled_at": null,
            "next_poll_at": null,
            "last_job_count": 0,
            "total_executions": 0,
            "active_job_id": null,
            "active_job_name": null,
        }),
    })
}

pub(crate) fn read_automation_job(
    db: &DbConnection,
    params: AutomationJobIdParams,
) -> Result<AutomationJobReadResponse, RuntimeCoreError> {
    let id = normalize_automation_job_id(&params.id)?;
    let conn = database::lock_db(db).map_err(data_error)?;
    let job = AutomationJobDao::get(&conn, &id).map_err(data_error)?;
    Ok(AutomationJobReadResponse {
        job: job
            .map(serde_json::to_value)
            .transpose()
            .map_err(data_error)?,
    })
}

pub(crate) fn create_automation_job(
    db: &DbConnection,
    params: AutomationJobCreateParams,
) -> Result<AutomationJobWriteResponse, RuntimeCoreError> {
    let request: AutomationJobCreateRequest =
        serde_json::from_value(params.request).map_err(data_error)?;
    validate_automation_job_create_request(&request)?;
    let now = Utc::now().to_rfc3339();
    let next_run_at = if request.enabled.unwrap_or(true) {
        preview_next_automation_run(&request.schedule).map_err(data_error)?
    } else {
        None
    };
    let job = AutomationJob {
        id: Uuid::new_v4().to_string(),
        name: request.name.trim().to_string(),
        description: normalize_optional_string(request.description),
        enabled: request.enabled.unwrap_or(true),
        workspace_id: request.workspace_id.trim().to_string(),
        execution_mode: request
            .execution_mode
            .unwrap_or(AutomationExecutionMode::Intelligent),
        schedule: request.schedule,
        payload: request.payload,
        delivery: request.delivery.unwrap_or_default(),
        timeout_secs: request.timeout_secs,
        max_retries: request.max_retries.unwrap_or(3).max(1),
        next_run_at,
        last_status: None,
        last_error: None,
        last_run_at: None,
        last_finished_at: None,
        running_started_at: None,
        consecutive_failures: 0,
        last_retry_count: 0,
        auto_disabled_until: None,
        last_delivery: None,
        created_at: now.clone(),
        updated_at: now,
    };
    let conn = database::lock_db(db).map_err(data_error)?;
    AutomationJobDao::create(&conn, &job).map_err(data_error)?;
    Ok(AutomationJobWriteResponse {
        job: serde_json::to_value(job).map_err(data_error)?,
    })
}

pub(crate) fn update_automation_job(
    db: &DbConnection,
    params: AutomationJobUpdateParams,
) -> Result<AutomationJobWriteResponse, RuntimeCoreError> {
    let id = normalize_automation_job_id(&params.id)?;
    let request: AutomationJobUpdateRequest =
        serde_json::from_value(params.request).map_err(data_error)?;
    let conn = database::lock_db(db).map_err(data_error)?;
    let mut job = AutomationJobDao::get(&conn, &id)
        .map_err(data_error)?
        .ok_or_else(|| RuntimeCoreError::Backend(format!("自动化任务不存在: {id}")))?;

    if let Some(name) = request.name {
        if name.trim().is_empty() {
            return Err(RuntimeCoreError::Backend("任务名称不能为空".to_string()));
        }
        job.name = name.trim().to_string();
    }
    if request.description.is_some() {
        job.description = normalize_optional_string(request.description);
    }
    if let Some(enabled) = request.enabled {
        job.enabled = enabled;
    }
    if let Some(workspace_id) = request.workspace_id {
        if workspace_id.trim().is_empty() {
            return Err(RuntimeCoreError::Backend("workspace_id 必填".to_string()));
        }
        job.workspace_id = workspace_id.trim().to_string();
    }
    if let Some(execution_mode) = request.execution_mode {
        job.execution_mode = execution_mode;
    }
    if let Some(schedule) = request.schedule {
        validate_automation_schedule_value(&schedule, Utc::now()).map_err(data_error)?;
        job.schedule = schedule;
    }
    if let Some(payload) = request.payload {
        validate_automation_payload(&payload)?;
        job.payload = payload;
    }
    if let Some(delivery) = request.delivery {
        job.delivery = delivery;
    }
    if request.clear_timeout_secs.unwrap_or(false) {
        job.timeout_secs = None;
    } else if request.timeout_secs.is_some() {
        job.timeout_secs = request.timeout_secs;
    }
    if let Some(max_retries) = request.max_retries {
        job.max_retries = max_retries.max(1);
    }
    job.next_run_at = if job.enabled && job.running_started_at.is_none() {
        preview_next_automation_run(&job.schedule).map_err(data_error)?
    } else {
        None
    };
    job.updated_at = Utc::now().to_rfc3339();

    validate_automation_job_record(&job)?;
    AutomationJobDao::update(&conn, &job).map_err(data_error)?;
    Ok(AutomationJobWriteResponse {
        job: serde_json::to_value(job).map_err(data_error)?,
    })
}

pub(crate) fn delete_automation_job(
    db: &DbConnection,
    params: AutomationJobIdParams,
) -> Result<AutomationJobDeleteResponse, RuntimeCoreError> {
    let id = normalize_automation_job_id(&params.id)?;
    let conn = database::lock_db(db).map_err(data_error)?;
    let deleted = AutomationJobDao::delete(&conn, &id).map_err(data_error)?;
    Ok(AutomationJobDeleteResponse { deleted })
}

pub(crate) fn start_automation_job_run(
    db: &DbConnection,
    id: String,
) -> Result<AutomationRunStart, RuntimeCoreError> {
    let id = normalize_automation_job_id(&id)?;
    let conn = database::lock_db(db).map_err(data_error)?;
    let job = AutomationJobDao::get(&conn, &id)
        .map_err(data_error)?
        .ok_or_else(|| RuntimeCoreError::Backend(format!("自动化任务不存在: {id}")))?;
    let mut start = build_automation_run_start(job, None)?;
    AgentRunDao::create_run(&conn, &start.run).map_err(data_error)?;
    apply_automation_run_started(&mut start.job, &start.run);
    AutomationJobDao::update(&conn, &start.job).map_err(data_error)?;
    Ok(start)
}

pub(crate) fn start_scheduled_task_run_record(
    db: &DbConnection,
    id: String,
    identity: Option<AutomationRunIdentity>,
) -> Result<AutomationRunStart, RuntimeCoreError> {
    let id = normalize_automation_job_id(&id)?;
    let conn = database::lock_db(db).map_err(data_error)?;
    let job = AutomationJobDao::get(&conn, &id)
        .map_err(data_error)?
        .ok_or_else(|| RuntimeCoreError::Backend(format!("已安排任务不存在: {id}")))?;
    require_scheduled_task_job(&job)?;
    let mut start = build_scheduled_task_manual_run_start(job, identity)?;
    AgentRunDao::create_run(&conn, &start.run).map_err(data_error)?;
    apply_automation_run_started(&mut start.job, &start.run);
    start.ownership_started_at = start.job.running_started_at.clone();
    start.task_revision = Some(start.job.updated_at.clone());
    AutomationJobDao::update(&conn, &start.job).map_err(data_error)?;
    Ok(start)
}

pub(crate) fn finish_automation_job_run(
    db: &DbConnection,
    finish: AutomationRunFinish,
) -> Result<(), RuntimeCoreError> {
    let conn = database::lock_db(db).map_err(data_error)?;
    let metadata = serde_json::to_string(&finish.metadata).map_err(data_error)?;
    let finished = AgentRunDao::finish_run(
        &conn,
        &finish.run_id,
        finish.status.clone(),
        &finish.finished_at,
        finish.duration_ms,
        finish.error_code.as_deref(),
        finish.error_message.as_deref(),
        Some(metadata.as_str()),
    )
    .map_err(data_error)?;
    if !finished {
        return Ok(());
    }
    let Some(mut job) = AutomationJobDao::get(&conn, &finish.job.id).map_err(data_error)? else {
        return Ok(());
    };
    let recompute_next_run = match finish.task_revision.as_deref() {
        Some(task_revision) => {
            if job.running_started_at.as_deref() != finish.ownership_started_at.as_deref() {
                return Ok(());
            }
            task_revision != job.updated_at
        }
        None => true,
    };
    apply_automation_run_finished(
        &mut job,
        &finish.status,
        finish.finished_at,
        finish.error_message,
        recompute_next_run,
    );
    AutomationJobDao::update(&conn, &job).map_err(data_error)?;
    Ok(())
}

pub(crate) fn fail_automation_job_run(
    db: &DbConnection,
    failure: AutomationRunFailure,
) -> Result<(), RuntimeCoreError> {
    let conn = database::lock_db(db).map_err(data_error)?;
    let metadata = serde_json::to_string(&failure.metadata).map_err(data_error)?;
    let finished = if let Some(run) = failure.run.as_ref() {
        AgentRunDao::finish_run(
            &conn,
            &run.id,
            failure.status.clone(),
            &failure.finished_at,
            failure.duration_ms,
            Some(failure.error_code.as_str()),
            Some(failure.error_message.as_str()),
            Some(metadata.as_str()),
        )
        .map_err(data_error)?
    } else {
        false
    };
    if failure.run.is_some() && !finished {
        return Ok(());
    }
    let Some(mut job) = AutomationJobDao::get(&conn, &failure.job.id).map_err(data_error)? else {
        return Ok(());
    };
    let recompute_next_run = match failure.task_revision.as_deref() {
        Some(task_revision) => {
            if job.running_started_at.as_deref() != failure.ownership_started_at.as_deref() {
                return Ok(());
            }
            task_revision != job.updated_at
        }
        None => true,
    };
    apply_automation_run_finished(
        &mut job,
        &failure.status,
        failure.finished_at,
        Some(failure.error_message),
        recompute_next_run,
    );
    AutomationJobDao::update(&conn, &job).map_err(data_error)?;
    Ok(())
}

pub(crate) fn read_automation_run_history(
    db: &DbConnection,
    params: AutomationJobRunHistoryParams,
) -> Result<AutomationJobRunHistoryResponse, RuntimeCoreError> {
    let id = normalize_automation_job_id(&params.id)?;
    let limit = params.limit.unwrap_or(20).clamp(1, 200);
    let conn = database::lock_db(db).map_err(data_error)?;
    let runs = AgentRunDao::list_runs_by_source_ref(&conn, "automation", &id, limit)
        .map_err(data_error)?;
    Ok(AutomationJobRunHistoryResponse {
        runs: values_from_serializable_vec(runs)?,
    })
}

pub(crate) fn preview_automation_schedule(
    params: AutomationScheduleParams,
) -> Result<AutomationSchedulePreviewResponse, RuntimeCoreError> {
    let schedule: TaskSchedule = serde_json::from_value(params.schedule).map_err(data_error)?;
    Ok(AutomationSchedulePreviewResponse {
        next_run_at: preview_next_automation_run(&schedule).map_err(data_error)?,
    })
}

pub(crate) fn validate_automation_schedule(
    params: AutomationScheduleParams,
) -> Result<AutomationScheduleValidateResponse, RuntimeCoreError> {
    let schedule: TaskSchedule = serde_json::from_value(params.schedule).map_err(data_error)?;
    Ok(
        match validate_automation_schedule_value(&schedule, Utc::now()) {
            Ok(()) => AutomationScheduleValidateResponse {
                valid: true,
                error: None,
            },
            Err(error) => AutomationScheduleValidateResponse {
                valid: false,
                error: Some(error),
            },
        },
    )
}

fn default_automation_poll_interval_secs() -> u64 {
    30
}

fn default_automation_enable_history() -> bool {
    true
}

fn automation_scheduler_config_value(config: AutomationSettings) -> Value {
    json!({
        "enabled": config.enabled,
        "poll_interval_secs": config.poll_interval_secs,
        "enable_history": config.enable_history,
    })
}

fn normalize_automation_job_id(id: &str) -> Result<String, RuntimeCoreError> {
    let id = id.trim();
    if id.is_empty() {
        return Err(RuntimeCoreError::Backend(
            "automation job id is required".to_string(),
        ));
    }
    Ok(id.to_string())
}

fn normalize_optional_string(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn validate_automation_job_create_request(
    request: &AutomationJobCreateRequest,
) -> Result<(), RuntimeCoreError> {
    if request.name.trim().is_empty() {
        return Err(RuntimeCoreError::Backend("任务名称不能为空".to_string()));
    }
    if request.workspace_id.trim().is_empty() {
        return Err(RuntimeCoreError::Backend("workspace_id 必填".to_string()));
    }
    validate_automation_schedule_value(&request.schedule, Utc::now()).map_err(data_error)?;
    validate_automation_payload(&request.payload)?;
    Ok(())
}

fn validate_automation_job_record(job: &AutomationJob) -> Result<(), RuntimeCoreError> {
    if job.name.trim().is_empty() {
        return Err(RuntimeCoreError::Backend("任务名称不能为空".to_string()));
    }
    if job.workspace_id.trim().is_empty() {
        return Err(RuntimeCoreError::Backend("workspace_id 必填".to_string()));
    }
    validate_automation_schedule_value(&job.schedule, Utc::now()).map_err(data_error)?;
    validate_automation_payload(&job.payload)?;
    Ok(())
}

fn validate_automation_payload(payload: &Value) -> Result<(), RuntimeCoreError> {
    let Some(payload) = payload.as_object() else {
        return Err(RuntimeCoreError::Backend(
            "自动化任务 payload 必须为对象".to_string(),
        ));
    };
    let kind = payload
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or_default();
    match kind {
        "agent_turn" => {
            let prompt = payload
                .get("prompt")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if prompt.trim().is_empty() {
                return Err(RuntimeCoreError::Backend(
                    "自动化任务内容不能为空".to_string(),
                ));
            }
            let thread_mode = payload
                .get("thread_mode")
                .or_else(|| payload.get("threadMode"))
                .and_then(Value::as_str);
            let scheduled_continue_thread = thread_mode == Some("continue_thread")
                && payload
                    .get("scheduled_task_schedule")
                    .is_some_and(|value| !value.is_null())
                && payload
                    .get("source_thread_id")
                    .or_else(|| payload.get("sourceThreadId"))
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .is_some_and(|value| !value.is_empty());
            if thread_mode != Some("new_thread") && !scheduled_continue_thread {
                for field in ["session_id", "thread_id"] {
                    let value = payload
                        .get(field)
                        .or_else(|| {
                            if field == "session_id" {
                                payload.get("sessionId")
                            } else {
                                payload.get("threadId")
                            }
                        })
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .filter(|value| !value.is_empty());
                    if value.is_none() {
                        return Err(RuntimeCoreError::Backend(format!(
                            "自动化任务 agent_turn payload 必须显式绑定 {field}"
                        )));
                    }
                }
            }
            if let Some(content_id) = payload
                .get("content_id")
                .or_else(|| payload.get("contentId"))
            {
                if content_id
                    .as_str()
                    .map(str::trim)
                    .unwrap_or_default()
                    .is_empty()
                {
                    return Err(RuntimeCoreError::Backend(
                        "自动化任务 content_id 不能为空字符串".to_string(),
                    ));
                }
            }
            if let Some(metadata) = payload
                .get("request_metadata")
                .or_else(|| payload.get("requestMetadata"))
            {
                if !metadata.is_object() {
                    return Err(RuntimeCoreError::Backend(
                        "自动化任务 request_metadata 必须为对象".to_string(),
                    ));
                }
            }
            Ok(())
        }
        "browser_session" => Err(RuntimeCoreError::Backend(
            "浏览器自动化任务已下线，不再允许创建或执行".to_string(),
        )),
        _ => Err(RuntimeCoreError::Backend(format!(
            "不支持的自动化任务 payload.kind: {kind}"
        ))),
    }
}

fn preview_next_automation_run(schedule: &TaskSchedule) -> Result<Option<String>, String> {
    Ok(next_run_for_automation_schedule(schedule, Utc::now())?.map(|value| value.to_rfc3339()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_agent_turn_payload_requires_thread_lineage() {
        let payload = json!({
            "kind": "agent_turn",
            "prompt": "生成摘要",
            "session_id": "session-job-1"
        });

        let error = validate_automation_payload(&payload).expect_err("should reject");
        assert!(error.to_string().contains("thread_id"));
    }

    #[test]
    fn validate_agent_turn_payload_accepts_explicit_thread_lineage() {
        let payload = json!({
            "kind": "agent_turn",
            "prompt": "生成摘要",
            "session_id": "session-job-1",
            "thread_id": "thread-job-1"
        });

        validate_automation_payload(&payload).expect("valid automation payload");
    }

    #[test]
    fn validate_agent_turn_payload_accepts_deferred_new_thread_lineage() {
        let payload = json!({
            "kind": "agent_turn",
            "prompt": "生成摘要",
            "thread_mode": "new_thread"
        });

        validate_automation_payload(&payload).expect("valid deferred lineage payload");
    }

    #[test]
    fn validate_agent_turn_payload_accepts_scheduled_canonical_thread_lookup() {
        let payload = json!({
            "kind": "agent_turn",
            "prompt": "生成摘要",
            "thread_mode": "continue_thread",
            "source_thread_id": "canonical-thread-1",
            "scheduled_task_schedule": {
                "type": "daily",
                "time": "08:30",
                "timezone": "Asia/Shanghai"
            }
        });

        validate_automation_payload(&payload).expect("valid scheduled canonical thread lookup");
    }
}
