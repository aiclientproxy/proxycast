use super::{
    create_automation_job, data_error, normalize_automation_job_id, update_automation_job,
    AutomationJobCreateParams, AutomationJobUpdateParams,
};
use crate::automation_execution::next_run_for_automation_schedule;
use crate::RuntimeCoreError;
use app_server_protocol::{
    ScheduledTask, ScheduledTaskCreateParams, ScheduledTaskCreateRequest,
    ScheduledTaskDeleteResponse, ScheduledTaskEnabledSetParams, ScheduledTaskExecution,
    ScheduledTaskIdParams, ScheduledTaskListParams, ScheduledTaskListResponse,
    ScheduledTaskNotificationPolicy, ScheduledTaskOverlapPolicy, ScheduledTaskReadResponse,
    ScheduledTaskRunListParams, ScheduledTaskRunListResponse, ScheduledTaskRunSummary,
    ScheduledTaskSchedule, ScheduledTaskSchedulePreviewParams,
    ScheduledTaskSchedulePreviewResponse, ScheduledTaskSummary, ScheduledTaskThreadMode,
    ScheduledTaskUpdateParams, ScheduledTaskUpdateRequest, ScheduledTaskWeekday,
    ScheduledTaskWriteResponse,
};
use chrono::Utc;
use lime_core::config::TaskSchedule;
use lime_core::database;
use lime_core::database::dao::agent_run::{AgentRun, AgentRunDao};
use lime_core::database::dao::automation_job::{AutomationJob, AutomationJobDao};
use lime_core::database::DbConnection;
use serde_json::{json, Value};

pub(crate) fn list_scheduled_tasks(
    db: &DbConnection,
    params: ScheduledTaskListParams,
) -> Result<ScheduledTaskListResponse, RuntimeCoreError> {
    let conn = database::lock_db(db).map_err(data_error)?;
    let query = params
        .query
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let limit = params.limit.unwrap_or(100).clamp(1, 200);
    let items = AutomationJobDao::list(&conn)
        .map_err(data_error)?
        .into_iter()
        .filter(is_scheduled_task_job)
        .filter(|job| params.enabled.is_none_or(|enabled| job.enabled == enabled))
        .filter(|job| {
            query.is_none_or(|query| job.name.to_lowercase().contains(&query.to_lowercase()))
        })
        .take(limit)
        .map(|job| {
            let last_run = last_scheduled_task_run(&conn, &job.id)?;
            scheduled_task_summary_from_job(job, last_run)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ScheduledTaskListResponse {
        items,
        next_cursor: None,
    })
}

pub(crate) fn read_scheduled_task(
    db: &DbConnection,
    params: ScheduledTaskIdParams,
) -> Result<ScheduledTaskReadResponse, RuntimeCoreError> {
    let conn = database::lock_db(db).map_err(data_error)?;
    let id = normalize_automation_job_id(&params.id)?;
    let task = match AutomationJobDao::get(&conn, &id).map_err(data_error)? {
        Some(job) => {
            require_scheduled_task_job(&job)?;
            let last_run = last_scheduled_task_run(&conn, &job.id)?;
            Some(scheduled_task_from_job(job, last_run)?)
        }
        None => None,
    };
    Ok(ScheduledTaskReadResponse { task })
}

pub(crate) fn create_scheduled_task(
    db: &DbConnection,
    params: ScheduledTaskCreateParams,
) -> Result<ScheduledTaskWriteResponse, RuntimeCoreError> {
    let job_params = lower_scheduled_task_create(params.task)?;
    let response = create_automation_job(db, job_params)?;
    let job: AutomationJob = serde_json::from_value(response.job).map_err(data_error)?;
    Ok(ScheduledTaskWriteResponse {
        task: scheduled_task_from_job(job, None)?,
    })
}

pub(crate) fn update_scheduled_task(
    db: &DbConnection,
    params: ScheduledTaskUpdateParams,
) -> Result<ScheduledTaskWriteResponse, RuntimeCoreError> {
    let id = normalize_automation_job_id(&params.id)?;
    let current = {
        let conn = database::lock_db(db).map_err(data_error)?;
        AutomationJobDao::get(&conn, &id)
            .map_err(data_error)?
            .filter(is_scheduled_task_job)
            .ok_or_else(|| scheduled_task_not_found(&id))?
    };
    let job_params = lower_scheduled_task_update(&current, params.task)?;
    let response = update_automation_job(db, job_params)?;
    let job: AutomationJob = serde_json::from_value(response.job).map_err(data_error)?;
    let last_run = {
        let conn = database::lock_db(db).map_err(data_error)?;
        last_scheduled_task_run(&conn, &job.id)?
    };
    Ok(ScheduledTaskWriteResponse {
        task: scheduled_task_from_job(job, last_run)?,
    })
}

pub(crate) fn set_scheduled_task_enabled(
    db: &DbConnection,
    params: ScheduledTaskEnabledSetParams,
) -> Result<ScheduledTaskWriteResponse, RuntimeCoreError> {
    update_scheduled_task(
        db,
        ScheduledTaskUpdateParams {
            id: params.id,
            task: ScheduledTaskUpdateRequest {
                title: None,
                prompt: None,
                schedule: None,
                execution: None,
                enabled: Some(params.enabled),
                notification_policy: None,
                overlap_policy: None,
                revision: None,
            },
        },
    )
}

pub(crate) fn delete_scheduled_task(
    db: &DbConnection,
    params: ScheduledTaskIdParams,
) -> Result<ScheduledTaskDeleteResponse, RuntimeCoreError> {
    let id = normalize_automation_job_id(&params.id)?;
    let conn = database::lock_db(db).map_err(data_error)?;
    let Some(job) = AutomationJobDao::get(&conn, &id).map_err(data_error)? else {
        return Ok(ScheduledTaskDeleteResponse { deleted: false });
    };
    require_scheduled_task_job(&job)?;
    let deleted =
        AutomationJobDao::soft_delete(&conn, &id, &Utc::now().to_rfc3339()).map_err(data_error)?;
    Ok(ScheduledTaskDeleteResponse { deleted })
}

pub(crate) fn list_scheduled_task_runs(
    db: &DbConnection,
    params: ScheduledTaskRunListParams,
) -> Result<ScheduledTaskRunListResponse, RuntimeCoreError> {
    let id = normalize_automation_job_id(&params.task_id)?;
    let limit = params.limit.unwrap_or(20).clamp(1, 200);
    let conn = database::lock_db(db).map_err(data_error)?;
    let job = AutomationJobDao::get_including_deleted(&conn, &id)
        .map_err(data_error)?
        .ok_or_else(|| scheduled_task_not_found(&id))?;
    require_scheduled_task_job(&job)?;
    let runs = AgentRunDao::list_runs_by_source_ref(&conn, "automation", &id, limit)
        .map_err(data_error)?
        .into_iter()
        .map(scheduled_task_run_from_agent_run)
        .collect();
    Ok(ScheduledTaskRunListResponse { runs })
}

pub(crate) fn preview_scheduled_task_schedule(
    params: ScheduledTaskSchedulePreviewParams,
) -> Result<ScheduledTaskSchedulePreviewResponse, RuntimeCoreError> {
    let schedule = lower_scheduled_schedule(&params.schedule)?;
    let mut cursor = Utc::now();
    let mut next_run_at = Vec::new();
    for _ in 0..5 {
        let Some(next) = next_run_for_automation_schedule(&schedule, cursor).map_err(data_error)?
        else {
            break;
        };
        next_run_at.push(next.to_rfc3339());
        cursor = next;
    }
    Ok(ScheduledTaskSchedulePreviewResponse {
        next_run_at,
        warnings: Vec::new(),
    })
}

fn lower_scheduled_task_create(
    task: ScheduledTaskCreateRequest,
) -> Result<AutomationJobCreateParams, RuntimeCoreError> {
    let (workspace_id, mut payload) = execution_to_payload(&task.execution, &task.prompt)?;
    let schedule = lower_scheduled_schedule(&task.schedule)?;
    let payload = payload
        .as_object_mut()
        .expect("scheduled task payload object");
    payload.insert(
        "scheduled_task_schedule".to_string(),
        serde_json::to_value(&task.schedule).map_err(data_error)?,
    );
    payload.insert(
        "notification_policy".to_string(),
        json!(task
            .notification_policy
            .unwrap_or(ScheduledTaskNotificationPolicy::AllRuns)),
    );
    payload.insert(
        "overlap_policy".to_string(),
        json!(task
            .overlap_policy
            .unwrap_or(ScheduledTaskOverlapPolicy::SkipIfRunning)),
    );
    Ok(AutomationJobCreateParams {
        request: json!({
            "name": task.title,
            "enabled": task.enabled,
            "workspace_id": workspace_id,
            "schedule": schedule,
            "payload": payload,
        }),
    })
}

fn lower_scheduled_task_update(
    current: &AutomationJob,
    task: ScheduledTaskUpdateRequest,
) -> Result<AutomationJobUpdateParams, RuntimeCoreError> {
    if let Some(revision) = task.revision.as_deref() {
        if revision != current.updated_at {
            return Err(RuntimeCoreError::Backend(
                "已安排任务已在其他窗口更新，请刷新后重试".to_string(),
            ));
        }
    }
    let mut request = serde_json::Map::new();
    let mut payload = current
        .payload
        .as_object()
        .cloned()
        .ok_or_else(|| RuntimeCoreError::Backend("任务 payload 无效".to_string()))?;
    let mut payload_changed = false;
    if let Some(title) = task.title {
        request.insert("name".to_string(), json!(title));
    }
    if let Some(prompt) = task.prompt {
        payload.insert("prompt".to_string(), json!(prompt));
        payload_changed = true;
    }
    if let Some(schedule) = task.schedule {
        request.insert(
            "schedule".to_string(),
            serde_json::to_value(lower_scheduled_schedule(&schedule)?).map_err(data_error)?,
        );
        payload.insert(
            "scheduled_task_schedule".to_string(),
            serde_json::to_value(schedule).map_err(data_error)?,
        );
        payload_changed = true;
    }
    if let Some(execution) = task.execution {
        let prompt = payload
            .get("prompt")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let (workspace_id, execution_payload) = execution_to_payload(&execution, prompt)?;
        request.insert("workspace_id".to_string(), json!(workspace_id));
        for key in [
            "session_id",
            "sessionId",
            "thread_id",
            "threadId",
            "source_thread_id",
            "sourceThreadId",
            "thread_mode",
            "threadMode",
            "cwd",
            "model",
            "model_id",
            "modelId",
            "reasoning_effort",
            "reasoningEffort",
            "approval_policy",
            "approvalPolicy",
            "sandbox_policy",
            "sandboxPolicy",
            "request_metadata",
            "requestMetadata",
        ] {
            payload.remove(key);
        }
        for (key, value) in execution_payload
            .as_object()
            .expect("scheduled task payload object")
        {
            payload.insert(key.clone(), value.clone());
        }
        payload_changed = true;
    }
    if let Some(enabled) = task.enabled {
        request.insert("enabled".to_string(), json!(enabled));
    }
    if let Some(notification_policy) = task.notification_policy {
        payload.insert(
            "notification_policy".to_string(),
            json!(notification_policy),
        );
        payload_changed = true;
    }
    if let Some(overlap_policy) = task.overlap_policy {
        payload.insert("overlap_policy".to_string(), json!(overlap_policy));
        payload_changed = true;
    }
    if payload_changed {
        request.insert("payload".to_string(), Value::Object(payload));
    }
    Ok(AutomationJobUpdateParams {
        id: current.id.clone(),
        request: serde_json::Value::Object(request),
    })
}

fn execution_to_payload(
    execution: &ScheduledTaskExecution,
    prompt: &str,
) -> Result<(String, Value), RuntimeCoreError> {
    let workspace_id = execution
        .project_id
        .clone()
        .or_else(|| execution.cwd.clone())
        .unwrap_or_else(|| "default".to_string());
    let mut payload = json!({
        "kind": "agent_turn",
        "prompt": prompt,
        "thread_mode": execution.thread_mode,
    });
    let object = payload.as_object_mut().expect("payload object");
    match execution.thread_mode {
        ScheduledTaskThreadMode::ContinueThread => {
            let thread_id = execution
                .source_thread_id
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    RuntimeCoreError::Backend("continue_thread 必须提供 sourceThreadId".to_string())
                })?;
            object.insert("source_thread_id".to_string(), json!(thread_id));
        }
        ScheduledTaskThreadMode::NewThread => {}
    }
    if let Some(cwd) = execution.cwd.as_deref() {
        object.insert("cwd".to_string(), json!(cwd));
    }
    if let Some(model_id) = execution.model_id.as_deref() {
        object.insert("model".to_string(), json!(model_id));
    }
    if let Some(reasoning_effort) = execution.reasoning_effort.as_deref() {
        object.insert("reasoning_effort".to_string(), json!(reasoning_effort));
    }
    if let Some(value) = execution.approval_policy.clone() {
        object.insert("approval_policy".to_string(), value);
    }
    if let Some(value) = execution.sandbox_policy.clone() {
        object.insert("sandbox_policy".to_string(), value);
    }
    if let Some(value) = execution.request_metadata.clone() {
        if !value.is_object() {
            return Err(RuntimeCoreError::Backend(
                "requestMetadata 必须为对象".to_string(),
            ));
        }
        object.insert("request_metadata".to_string(), value);
    }
    Ok((workspace_id, payload))
}

fn lower_scheduled_schedule(
    schedule: &ScheduledTaskSchedule,
) -> Result<TaskSchedule, RuntimeCoreError> {
    match schedule {
        ScheduledTaskSchedule::Hourly {
            interval_hours,
            days,
            minute,
            timezone,
        } => {
            if *interval_hours == 0 || *interval_hours > 24 || *minute > 59 {
                return Err(RuntimeCoreError::Backend(
                    "hourly schedule 参数无效".to_string(),
                ));
            }
            if timezone.trim().is_empty() {
                return Err(RuntimeCoreError::Backend(
                    "hourly schedule 时区不能为空".to_string(),
                ));
            }
            if days.as_ref().is_some_and(Vec::is_empty) {
                return Err(RuntimeCoreError::Backend(
                    "hourly schedule days 不能为空数组".to_string(),
                ));
            }
            if *interval_hours == 1 && *minute == 0 && days.is_none() {
                Ok(TaskSchedule::Every { every_secs: 3600 })
            } else {
                let weekdays = days
                    .as_ref()
                    .map(|days| normalized_weekdays(days))
                    .transpose()?
                    .unwrap_or_else(|| "*".to_string());
                Ok(TaskSchedule::Cron {
                    expr: format!("{} */{} * * {}", minute, interval_hours, weekdays),
                    tz: Some(timezone.clone()),
                })
            }
        }
        ScheduledTaskSchedule::Daily { time, timezone } => {
            cron_schedule_for_days(time, "*", timezone)
        }
        ScheduledTaskSchedule::Weekdays { time, timezone } => {
            cron_schedule_for_days(time, "1-5", timezone)
        }
        ScheduledTaskSchedule::Weekly {
            days,
            time,
            timezone,
        } => {
            let weekdays = normalized_weekdays(days)?;
            cron_schedule_for_days(time, &weekdays, timezone)
        }
    }
}

fn normalized_weekdays(days: &[ScheduledTaskWeekday]) -> Result<String, RuntimeCoreError> {
    if days.is_empty() {
        return Err(RuntimeCoreError::Backend(
            "weekly schedule days 不能为空".to_string(),
        ));
    }
    let mut values = days.iter().map(weekday_to_cron).collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();
    Ok(values.join(","))
}

fn cron_schedule_for_days(
    time: &str,
    weekdays: &str,
    timezone: &str,
) -> Result<TaskSchedule, RuntimeCoreError> {
    let parts = time.split(':').collect::<Vec<_>>();
    if parts.len() != 2 {
        return Err(RuntimeCoreError::Backend("time 必须为 HH:mm".to_string()));
    }
    let hour: u8 = parts[0]
        .parse()
        .map_err(|_| RuntimeCoreError::Backend("time 小时无效".to_string()))?;
    let minute: u8 = parts[1]
        .parse()
        .map_err(|_| RuntimeCoreError::Backend("time 分钟无效".to_string()))?;
    if hour > 23 || minute > 59 || timezone.trim().is_empty() {
        return Err(RuntimeCoreError::Backend(
            "schedule 时间或时区无效".to_string(),
        ));
    }
    Ok(TaskSchedule::Cron {
        expr: format!("{} {} * * {}", minute, hour, weekdays),
        tz: Some(timezone.to_string()),
    })
}

fn weekday_to_cron(day: &ScheduledTaskWeekday) -> &'static str {
    match day {
        ScheduledTaskWeekday::MO => "1",
        ScheduledTaskWeekday::TU => "2",
        ScheduledTaskWeekday::WE => "3",
        ScheduledTaskWeekday::TH => "4",
        ScheduledTaskWeekday::FR => "5",
        ScheduledTaskWeekday::SA => "6",
        ScheduledTaskWeekday::SU => "0",
    }
}

pub(crate) fn is_scheduled_task_job(job: &AutomationJob) -> bool {
    job.payload.as_object().is_some_and(|payload| {
        payload.get("kind").and_then(Value::as_str) == Some("agent_turn")
            && payload
                .get("scheduled_task_schedule")
                .and_then(|value| {
                    serde_json::from_value::<ScheduledTaskSchedule>(value.clone()).ok()
                })
                .is_some()
            && payload_enum::<ScheduledTaskThreadMode>(payload, "thread_mode").is_some()
            && payload_enum::<ScheduledTaskNotificationPolicy>(payload, "notification_policy")
                .is_some()
            && payload_enum::<ScheduledTaskOverlapPolicy>(payload, "overlap_policy").is_some()
    })
}

pub(crate) fn require_scheduled_task_job(job: &AutomationJob) -> Result<(), RuntimeCoreError> {
    if is_scheduled_task_job(job) {
        return Ok(());
    }
    Err(scheduled_task_not_found(&job.id))
}

fn scheduled_task_not_found(id: &str) -> RuntimeCoreError {
    RuntimeCoreError::Backend(format!("已安排任务不存在: {id}"))
}

fn last_scheduled_task_run(
    conn: &rusqlite::Connection,
    task_id: &str,
) -> Result<Option<ScheduledTaskRunSummary>, RuntimeCoreError> {
    Ok(
        AgentRunDao::list_runs_by_source_ref(conn, "automation", task_id, 1)
            .map_err(data_error)?
            .into_iter()
            .next()
            .map(scheduled_task_run_from_agent_run),
    )
}

fn scheduled_task_from_job(
    job: AutomationJob,
    last_run_summary: Option<ScheduledTaskRunSummary>,
) -> Result<ScheduledTask, RuntimeCoreError> {
    require_scheduled_task_job(&job)?;
    let payload = job
        .payload
        .as_object()
        .ok_or_else(|| RuntimeCoreError::Backend("任务 payload 无效".to_string()))?;
    let schedule = payload
        .get("scheduled_task_schedule")
        .cloned()
        .ok_or_else(|| scheduled_task_not_found(&job.id))
        .and_then(|value| serde_json::from_value(value).map_err(data_error))?;
    let thread_mode = payload_enum(payload, "thread_mode")
        .ok_or_else(|| RuntimeCoreError::Backend("任务 thread_mode 无效".to_string()))?;
    let execution = ScheduledTaskExecution {
        source_thread_id: payload
            .get("source_thread_id")
            .or_else(|| payload.get("sourceThreadId"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        thread_mode,
        project_id: Some(job.workspace_id.clone()),
        cwd: payload
            .get("cwd")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        model_id: payload
            .get("model")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        reasoning_effort: payload
            .get("reasoning_effort")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        approval_policy: payload.get("approval_policy").cloned(),
        sandbox_policy: payload.get("sandbox_policy").cloned(),
        request_metadata: payload
            .get("request_metadata")
            .or_else(|| payload.get("requestMetadata"))
            .cloned(),
    };
    Ok(ScheduledTask {
        id: job.id,
        title: job.name,
        prompt: payload
            .get("prompt")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        enabled: job.enabled,
        schedule,
        execution,
        notification_policy: payload_enum(payload, "notification_policy").ok_or_else(|| {
            RuntimeCoreError::Backend("任务 notification_policy 无效".to_string())
        })?,
        overlap_policy: payload_enum(payload, "overlap_policy")
            .ok_or_else(|| RuntimeCoreError::Backend("任务 overlap_policy 无效".to_string()))?,
        next_run_at: job.next_run_at,
        last_run_summary,
        created_at: job.created_at,
        updated_at: job.updated_at,
    })
}

fn scheduled_task_summary_from_job(
    job: AutomationJob,
    last_run: Option<ScheduledTaskRunSummary>,
) -> Result<ScheduledTaskSummary, RuntimeCoreError> {
    let task = scheduled_task_from_job(job, last_run)?;
    let attention = task
        .last_run_summary
        .as_ref()
        .is_some_and(|run| matches!(run.status.as_str(), "error" | "timeout" | "missed"));
    Ok(ScheduledTaskSummary {
        id: task.id,
        title: task.title,
        enabled: task.enabled,
        attention,
        schedule: task.schedule,
        next_run_at: task.next_run_at,
        last_run: task.last_run_summary,
    })
}

fn scheduled_task_run_from_agent_run(run: AgentRun) -> ScheduledTaskRunSummary {
    let metadata = run
        .metadata
        .as_deref()
        .and_then(|value| serde_json::from_str::<Value>(value).ok());
    ScheduledTaskRunSummary {
        id: run.id,
        task_id: run.source_ref.unwrap_or_default(),
        status: run.status.as_str().to_string(),
        scheduled_for: metadata
            .as_ref()
            .and_then(|value| value.get("scheduledFor"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        started_at: Some(run.started_at),
        finished_at: run.finished_at,
        session_id: run.session_id,
        thread_id: metadata
            .as_ref()
            .and_then(|value| value.get("threadId"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        turn_id: metadata
            .as_ref()
            .and_then(|value| value.get("turnId"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        summary: None,
        error: run.error_message,
    }
}

fn payload_enum<T>(payload: &serde_json::Map<String, Value>, key: &str) -> Option<T>
where
    T: serde::de::DeserializeOwned,
{
    serde_json::from_value(payload.get(key)?.clone()).ok()
}

#[cfg(test)]
#[path = "scheduled_tasks/tests.rs"]
mod tests;
