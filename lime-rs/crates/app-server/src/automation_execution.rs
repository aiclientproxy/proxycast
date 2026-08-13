use app_server_protocol::AgentSessionStartParams;
use app_server_protocol::AgentTurnStatus;
use app_server_protocol::AutomationJobIdParams;
use app_server_protocol::AutomationJobRunNowResponse;
use app_server_protocol::BusinessObjectRef;
use app_server_protocol::RuntimeOptions;
use chrono::DateTime;
use chrono::Duration;
use chrono::LocalResult;
use chrono::TimeZone;
use chrono::Utc;
use lime_core::config::TaskSchedule;
use lime_core::database::dao::agent_run::AgentRun;
use lime_core::database::dao::agent_run::AgentRunStatus;
use lime_core::database::dao::automation_job::AutomationJob;
use serde_json::json;
use serde_json::Value;
use std::str::FromStr;
use uuid::Uuid;

use crate::runtime::TurnStartRequest;
use crate::RuntimeCore;
use crate::RuntimeCoreError;
use crate::RuntimeHostContext;

const AUTOMATION_SOURCE: &str = "automation";

#[derive(Debug, Clone)]
pub struct AutomationRunStart {
    pub job: AutomationJob,
    pub run: AgentRun,
    pub session_id: String,
    pub thread_id: String,
    pub turn_id: String,
    pub prompt: String,
    pub runtime_options: RuntimeOptions,
    pub trigger: String,
    pub scheduled_for: Option<String>,
    pub claimed_at: Option<String>,
    pub ownership_started_at: Option<String>,
    pub task_revision: Option<String>,
    pub catch_up: bool,
    pub skipped_window_count: u32,
}

#[derive(Debug, Clone)]
pub struct AutomationRunFinish {
    pub job: AutomationJob,
    pub run_id: String,
    pub status: AgentRunStatus,
    pub finished_at: String,
    pub duration_ms: Option<i64>,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
    pub metadata: Value,
    pub ownership_started_at: Option<String>,
    pub task_revision: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AutomationRunFailure {
    pub job: AutomationJob,
    pub run: Option<AgentRun>,
    pub status: AgentRunStatus,
    pub finished_at: String,
    pub duration_ms: Option<i64>,
    pub error_code: String,
    pub error_message: String,
    pub metadata: Value,
    pub ownership_started_at: Option<String>,
    pub task_revision: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AutomationRunIdentity {
    pub session_id: String,
    pub thread_id: String,
}

#[derive(Debug, Clone)]
pub struct ClaimedAutomationRun {
    pub run_id: String,
    pub scheduled_for: String,
    pub claimed_at: String,
    pub task_revision: String,
    pub catch_up: bool,
    pub skipped_window_count: u32,
}

impl RuntimeCore {
    pub async fn execute_automation_job_now(
        &self,
        params: AutomationJobIdParams,
        host: RuntimeHostContext,
    ) -> Result<AutomationJobRunNowResponse, RuntimeCoreError> {
        let start = self.start_automation_job_run(params.id.clone()).await?;
        self.execute_started_automation_job(params.id, start, host)
            .await
    }

    pub(crate) async fn execute_scheduled_task_now(
        &self,
        id: String,
        identity: Option<AutomationRunIdentity>,
        host: RuntimeHostContext,
    ) -> Result<AutomationJobRunNowResponse, RuntimeCoreError> {
        let start = self
            .start_scheduled_task_run_record(id.clone(), identity)
            .await?;
        self.execute_started_automation_job(id, start, host).await
    }

    pub(crate) async fn execute_started_automation_job(
        &self,
        id: String,
        start: AutomationRunStart,
        host: RuntimeHostContext,
    ) -> Result<AutomationJobRunNowResponse, RuntimeCoreError> {
        let started_ms = Utc::now().timestamp_millis();

        let start_result = self.start_automation_turn(&start, host).await;
        match start_result {
            Ok(turn) => {
                let mut finished_at = None;
                let status = match turn.status {
                    AgentTurnStatus::Completed => Some(AgentRunStatus::Success),
                    AgentTurnStatus::Failed => Some(AgentRunStatus::Error),
                    AgentTurnStatus::Canceled => Some(AgentRunStatus::Canceled),
                    _ => None,
                };
                if let Some(status) = status {
                    let terminal_at = Utc::now().to_rfc3339();
                    let finish = AutomationRunFinish {
                        job: start.job,
                        run_id: start.run.id.clone(),
                        status,
                        finished_at: terminal_at.clone(),
                        duration_ms: Some(Utc::now().timestamp_millis().saturating_sub(started_ms)),
                        error_code: None,
                        error_message: None,
                        metadata: json!({
                            "jobId": id,
                            "sessionId": start.session_id,
                            "threadId": start.thread_id,
                            "turnId": turn.turn_id,
                            "turnStatus": agent_turn_status_value(turn.status),
                            "trigger": start.trigger,
                            "scheduledFor": start.scheduled_for,
                            "claimedAt": start.claimed_at,
                            "taskRevision": start.task_revision,
                            "catchUp": start.catch_up,
                            "skippedWindowCount": start.skipped_window_count,
                        }),
                        ownership_started_at: start.ownership_started_at,
                        task_revision: start.task_revision,
                    };
                    self.finish_automation_job_run(finish).await?;
                    finished_at = Some(terminal_at);
                }
                Ok(AutomationJobRunNowResponse {
                    result: json!({
                        "job_id": id,
                        "run_id": start.run.id,
                        "session_id": start.session_id,
                        "thread_id": start.thread_id,
                        "turn_id": turn.turn_id,
                        "started_at": start.run.started_at,
                        "finished_at": finished_at,
                        "started": true,
                        "status": agent_turn_status_value(turn.status),
                    }),
                })
            }
            Err(error) => {
                let failure = AutomationRunFailure {
                    job: start.job,
                    run: Some(start.run),
                    status: AgentRunStatus::Error,
                    finished_at: Utc::now().to_rfc3339(),
                    duration_ms: Some(Utc::now().timestamp_millis().saturating_sub(started_ms)),
                    error_code: "automation_turn_start_failed".to_string(),
                    error_message: error.to_string(),
                    metadata: json!({
                        "jobId": id,
                        "sessionId": start.session_id,
                        "threadId": start.thread_id,
                        "turnId": start.turn_id,
                        "trigger": start.trigger,
                        "scheduledFor": start.scheduled_for,
                        "claimedAt": start.claimed_at,
                        "taskRevision": start.task_revision,
                        "catchUp": start.catch_up,
                        "skippedWindowCount": start.skipped_window_count,
                    }),
                    ownership_started_at: start.ownership_started_at,
                    task_revision: start.task_revision,
                };
                self.fail_automation_job_run(failure).await?;
                Err(error)
            }
        }
    }

    async fn start_automation_turn(
        &self,
        start: &AutomationRunStart,
        host: RuntimeHostContext,
    ) -> Result<app_server_protocol::AgentTurn, RuntimeCoreError> {
        match self.start_session(AgentSessionStartParams {
            session_id: Some(start.session_id.clone()),
            thread_id: Some(start.thread_id.clone()),
            app_id: "automation".to_string(),
            workspace_id: Some(start.job.workspace_id.clone()),
            business_object_ref: Some(BusinessObjectRef {
                kind: "automation_job".to_string(),
                id: start.job.id.clone(),
                title: Some(start.job.name.clone()),
                uri: None,
                metadata: Some(json!({
                    "source": AUTOMATION_SOURCE,
                    "runId": start.run.id,
                    "executionMode": start.job.execution_mode,
                    "trigger": start.trigger,
                    "scheduledFor": start.scheduled_for,
                    "claimedAt": start.claimed_at,
                })),
            }),
            locale: None,
        }) {
            Ok(_) => {}
            Err(RuntimeCoreError::SessionAlreadyExists(_)) => {}
            Err(error) => return Err(error),
        }

        let output = self
            .start_turn(
                TurnStartRequest {
                    session_id: start.session_id.clone(),
                    turn_id: Some(start.turn_id.clone()),
                    input: vec![agent_protocol::AgentInput::text(start.prompt.clone())],
                    runtime_options: Some(start.runtime_options.clone()),
                    queue_if_busy: false,
                    skip_pre_submit_resume: false,
                },
                host,
            )
            .await?;
        Ok(output.response.turn)
    }
}

pub fn build_automation_run_start(
    job: AutomationJob,
    resolved_identity: Option<AutomationRunIdentity>,
) -> Result<AutomationRunStart, RuntimeCoreError> {
    build_automation_run_start_inner(job, resolved_identity, None, false)
}

pub fn build_scheduled_task_manual_run_start(
    job: AutomationJob,
    resolved_identity: Option<AutomationRunIdentity>,
) -> Result<AutomationRunStart, RuntimeCoreError> {
    build_automation_run_start_inner(job, resolved_identity, None, true)
}

pub fn build_claimed_automation_run_start(
    job: AutomationJob,
    resolved_identity: Option<AutomationRunIdentity>,
    claim: ClaimedAutomationRun,
) -> Result<AutomationRunStart, RuntimeCoreError> {
    build_automation_run_start_inner(job, resolved_identity, Some(claim), false)
}

fn build_automation_run_start_inner(
    job: AutomationJob,
    resolved_identity: Option<AutomationRunIdentity>,
    claim: Option<ClaimedAutomationRun>,
    allow_disabled: bool,
) -> Result<AutomationRunStart, RuntimeCoreError> {
    validate_automation_job_for_run(&job, claim.as_ref(), allow_disabled)?;
    let payload = job
        .payload
        .as_object()
        .ok_or_else(|| RuntimeCoreError::Backend("自动化任务 payload 必须为对象".to_string()))?;
    let prompt = string_field(payload, &["prompt"])
        .ok_or_else(|| RuntimeCoreError::Backend("自动化任务内容不能为空".to_string()))?;
    let run_id = claim
        .as_ref()
        .map(|claim| claim.run_id.clone())
        .unwrap_or_else(|| format!("automation-run-{}", Uuid::new_v4()));
    let (session_id, thread_id) =
        if string_field(payload, &["thread_mode", "threadMode"]).as_deref() == Some("new_thread") {
            let lineage_id = Uuid::new_v4().to_string();
            (
                format!("scheduled-session-{lineage_id}"),
                format!("scheduled-thread-{lineage_id}"),
            )
        } else if let Some(identity) = resolved_identity {
            (identity.session_id, identity.thread_id)
        } else {
            (
                required_agent_turn_payload_field(payload, &["session_id", "sessionId"])?,
                required_agent_turn_payload_field(payload, &["thread_id", "threadId"])?,
            )
        };
    let turn_id =
        string_field(payload, &["turn_id", "turnId"]).unwrap_or_else(|| format!("turn-{run_id}"));
    let started_at = Utc::now().to_rfc3339();
    let trigger = claim
        .as_ref()
        .map(|claim| {
            if claim.catch_up {
                "catch_up"
            } else {
                "schedule"
            }
        })
        .unwrap_or("manual");
    let scheduled_for = claim.as_ref().map(|claim| claim.scheduled_for.as_str());
    let claimed_at = claim.as_ref().map(|claim| claim.claimed_at.as_str());
    let ownership_started_at = claim.as_ref().map(|claim| claim.claimed_at.as_str());
    let task_revision = claim.as_ref().map(|claim| claim.task_revision.as_str());
    let catch_up = claim.as_ref().is_some_and(|claim| claim.catch_up);
    let skipped_window_count = claim.as_ref().map_or(0, |claim| claim.skipped_window_count);
    let runtime_options = build_runtime_options(
        &job,
        &run_id,
        &session_id,
        &thread_id,
        &turn_id,
        trigger,
        scheduled_for,
        claimed_at,
    )?;
    let run = AgentRun {
        id: run_id,
        source: AUTOMATION_SOURCE.to_string(),
        source_ref: Some(job.id.clone()),
        session_id: Some(session_id.clone()),
        status: AgentRunStatus::Running,
        started_at: started_at.clone(),
        finished_at: None,
        duration_ms: None,
        error_code: None,
        error_message: None,
        metadata: Some(
            serde_json::to_string(&json!({
                "jobId": job.id,
                "jobName": job.name,
                "sessionId": session_id,
                "threadId": thread_id,
                "turnId": turn_id,
                "payloadKind": "agent_turn",
                "trigger": trigger,
                "scheduledFor": scheduled_for,
                "claimedAt": claimed_at,
                "taskRevision": task_revision,
                "catchUp": catch_up,
                "skippedWindowCount": skipped_window_count,
                "runtimeOptions": runtime_options,
            }))
            .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?,
        ),
        created_at: started_at.clone(),
        updated_at: started_at,
    };

    Ok(AutomationRunStart {
        job,
        run,
        session_id,
        thread_id,
        turn_id,
        prompt,
        runtime_options,
        trigger: trigger.to_string(),
        scheduled_for: scheduled_for.map(ToOwned::to_owned),
        claimed_at: claimed_at.map(ToOwned::to_owned),
        ownership_started_at: ownership_started_at.map(ToOwned::to_owned),
        task_revision: task_revision.map(ToOwned::to_owned),
        catch_up,
        skipped_window_count,
    })
}

pub fn apply_automation_run_started(job: &mut AutomationJob, run: &AgentRun) {
    job.running_started_at = Some(run.started_at.clone());
    job.last_run_at = Some(run.started_at.clone());
    job.last_finished_at = None;
    job.last_status = Some("running".to_string());
    job.last_error = None;
    job.updated_at = run.started_at.clone();
}

pub fn apply_automation_run_finished(
    job: &mut AutomationJob,
    status: &AgentRunStatus,
    finished_at: String,
    error_message: Option<String>,
    recompute_next_run: bool,
) {
    job.running_started_at = None;
    job.last_finished_at = Some(finished_at.clone());
    job.last_status = Some(status.as_str().to_string());
    job.last_error = error_message.clone();
    match status {
        AgentRunStatus::Success => {
            job.consecutive_failures = 0;
            job.last_retry_count = 0;
            job.auto_disabled_until = None;
        }
        AgentRunStatus::Error | AgentRunStatus::Timeout => {
            job.consecutive_failures = job.consecutive_failures.saturating_add(1);
            job.last_retry_count = job.last_retry_count.saturating_add(1);
        }
        AgentRunStatus::Canceled => {}
        AgentRunStatus::Missed => {}
        AgentRunStatus::Queued | AgentRunStatus::Running => {}
    }
    if recompute_next_run {
        job.next_run_at = if job.enabled {
            next_run_for_automation_schedule(&job.schedule, Utc::now())
                .ok()
                .flatten()
                .map(|value| value.to_rfc3339())
        } else {
            None
        };
    }
    job.updated_at = finished_at;
}

pub fn next_run_for_automation_schedule(
    schedule: &TaskSchedule,
    from: DateTime<Utc>,
) -> Result<Option<DateTime<Utc>>, String> {
    match schedule {
        TaskSchedule::Every { every_secs } => {
            let secs = (*every_secs).max(60);
            Ok(Some(from + Duration::seconds(secs as i64)))
        }
        TaskSchedule::Cron { expr, tz } => {
            let normalized = normalize_cron_expression(expr);
            let cron_schedule = cron::Schedule::from_str(&normalized)
                .map_err(|error| format!("无效的 Cron 表达式: {error}"))?;
            let next = if let Some(tz_str) = tz
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                let timezone: chrono_tz::Tz = tz_str
                    .parse()
                    .map_err(|_| format!("无效的时区: {tz_str}"))?;
                next_distinct_cron_occurrence(&cron_schedule, from, timezone)
            } else {
                cron_schedule.after(&from).next()
            };
            Ok(next)
        }
        TaskSchedule::At { at } => {
            let target = DateTime::parse_from_rfc3339(at)
                .map_err(|error| format!("无效的时间格式（需要 RFC3339）: {error}"))?
                .with_timezone(&Utc);
            if target > from {
                Ok(Some(target))
            } else {
                Ok(None)
            }
        }
    }
}

fn next_distinct_cron_occurrence(
    schedule: &cron::Schedule,
    from: DateTime<Utc>,
    timezone: chrono_tz::Tz,
) -> Option<DateTime<Utc>> {
    let local_from = from.with_timezone(&timezone).naive_local();
    let wall_clock_cursor = Utc.from_utc_datetime(&local_from);
    for wall_clock_occurrence in schedule.after(&wall_clock_cursor) {
        let local_occurrence = wall_clock_occurrence.naive_utc();
        let occurrence = match timezone.from_local_datetime(&local_occurrence) {
            LocalResult::Single(value) => value,
            LocalResult::Ambiguous(earlier, _) => earlier,
            LocalResult::None => first_valid_local_time(timezone, local_occurrence)?,
        }
        .with_timezone(&Utc);
        if occurrence > from {
            return Some(occurrence);
        }
    }
    None
}

fn first_valid_local_time(
    timezone: chrono_tz::Tz,
    missing: chrono::NaiveDateTime,
) -> Option<DateTime<chrono_tz::Tz>> {
    const MAX_DST_GAP_MINUTES: i64 = 26 * 60;
    (1..=MAX_DST_GAP_MINUTES).find_map(|minutes| {
        let candidate = missing + Duration::minutes(minutes);
        match timezone.from_local_datetime(&candidate) {
            LocalResult::Single(value) => Some(value),
            LocalResult::Ambiguous(earlier, _) => Some(earlier),
            LocalResult::None => None,
        }
    })
}

pub fn validate_automation_schedule_value(
    schedule: &TaskSchedule,
    now: DateTime<Utc>,
) -> Result<(), String> {
    match schedule {
        TaskSchedule::Every { every_secs } => {
            if *every_secs < 60 {
                return Err("间隔时间不能小于 60 秒".to_string());
            }
            Ok(())
        }
        TaskSchedule::Cron { expr, tz } => {
            let normalized = normalize_cron_expression(expr);
            cron::Schedule::from_str(&normalized)
                .map_err(|error| format!("无效的 Cron 表达式: {error}"))?;
            if let Some(tz_str) = tz
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                let _: chrono_tz::Tz = tz_str
                    .parse()
                    .map_err(|_| format!("无效的时区: {tz_str}"))?;
            }
            Ok(())
        }
        TaskSchedule::At { at } => {
            let target = DateTime::parse_from_rfc3339(at)
                .map_err(|error| format!("无效的时间格式: {error}"))?
                .with_timezone(&Utc);
            if target <= now {
                return Err("指定时间已过期".to_string());
            }
            Ok(())
        }
    }
}

fn validate_automation_job_for_run(
    job: &AutomationJob,
    claim: Option<&ClaimedAutomationRun>,
    allow_disabled: bool,
) -> Result<(), RuntimeCoreError> {
    if !allow_disabled && !job.enabled {
        return Err(RuntimeCoreError::Backend(format!(
            "自动化任务已禁用: {}",
            job.id
        )));
    }
    if let Some(claim) = claim {
        if job.running_started_at.as_deref() != Some(claim.claimed_at.as_str()) {
            return Err(RuntimeCoreError::Backend(format!(
                "已安排任务 claim ownership 无效: {}",
                job.id
            )));
        }
    } else if job.running_started_at.is_some() {
        return Err(RuntimeCoreError::Backend(format!(
            "自动化任务正在运行: {}",
            job.id
        )));
    }
    if let Some(until) = job.auto_disabled_until.as_deref() {
        let until = DateTime::parse_from_rfc3339(until)
            .map(|value| value.with_timezone(&Utc))
            .map_err(|error| {
                RuntimeCoreError::Backend(format!("自动化任务冷却时间无效: {error}"))
            })?;
        if until > Utc::now() {
            return Err(RuntimeCoreError::Backend(format!(
                "自动化任务处于冷却中，直到 {}",
                until.to_rfc3339()
            )));
        }
    }
    let kind = job
        .payload
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if kind != "agent_turn" {
        return Err(RuntimeCoreError::Backend(format!(
            "不支持执行自动化任务 payload.kind: {kind}"
        )));
    }
    Ok(())
}

fn build_runtime_options(
    job: &AutomationJob,
    run_id: &str,
    session_id: &str,
    thread_id: &str,
    turn_id: &str,
    trigger: &str,
    scheduled_for: Option<&str>,
    claimed_at: Option<&str>,
) -> Result<RuntimeOptions, RuntimeCoreError> {
    let payload = job
        .payload
        .as_object()
        .ok_or_else(|| RuntimeCoreError::Backend("自动化任务 payload 必须为对象".to_string()))?;
    let request_metadata = payload
        .get("request_metadata")
        .or_else(|| payload.get("requestMetadata"))
        .cloned()
        .filter(Value::is_object);
    let working_dir = string_field(
        payload,
        &[
            "working_dir",
            "workingDir",
            "working_directory",
            "workingDirectory",
            "cwd",
        ],
    );
    let project_root = string_field(
        payload,
        &[
            "project_root",
            "projectRoot",
            "workspace_root",
            "workspaceRoot",
        ],
    );
    let reasoning_effort = string_field(
        payload,
        &[
            "reasoning_effort",
            "reasoningEffort",
            "model_reasoning_effort",
            "modelReasoningEffort",
        ],
    )
    .or_else(|| json_pointer_string_value(&job.payload, &["/reasoning/effort"]));
    let provider_preference = string_field(
        payload,
        &["provider_preference", "providerPreference", "provider"],
    );
    let model_preference = string_field(payload, &["model_preference", "modelPreference", "model"]);
    let system_prompt = string_field(payload, &["system_prompt", "systemPrompt"]);
    let approval_policy = string_field(payload, &["approval_policy", "approvalPolicy"]);
    let sandbox_policy = string_field(payload, &["sandbox_policy", "sandboxPolicy"]);
    let search_mode = string_field(payload, &["search_mode", "searchMode"]);
    let web_search = bool_field(payload, &["web_search", "webSearch"]);
    let provider_config = payload
        .get("provider_config")
        .or_else(|| payload.get("providerConfig"))
        .cloned();
    let output_schema = payload
        .get("output_schema")
        .or_else(|| payload.get("outputSchema"))
        .cloned()
        .or_else(|| {
            job.delivery
                .output_schema
                .as_ref()
                .map(|value| json!(value))
        });

    let runtime_metadata = json_strip_nulls(json!({
        "source": AUTOMATION_SOURCE,
        "automation_job": {
            "id": job.id,
            "name": job.name,
            "workspace_id": job.workspace_id,
            "execution_mode": job.execution_mode,
            "run_id": run_id,
        },
        "request_metadata": request_metadata,
        "harness": {
            "automation_job": {
                "id": job.id,
                "run_id": run_id,
                "session_id": session_id,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "trigger": trigger,
                "scheduled_for": scheduled_for,
                "claimed_at": claimed_at,
            },
            "reasoning_effort": reasoning_effort,
            "search_mode": search_mode,
            "web_search": web_search,
            "working_dir": working_dir,
            "project_root": project_root,
        },
        "turn_config": {
            "reasoning_effort": reasoning_effort,
            "working_dir": working_dir,
            "project_root": project_root,
            "output_schema": output_schema,
        },
    }));

    let runtime_request = app_server_protocol::RuntimeRequest {
        provider_config: provider_config
            .clone()
            .and_then(|value| serde_json::from_value(value).ok()),
        provider_preference: provider_preference.clone(),
        model_preference: model_preference.clone(),
        reasoning_effort: reasoning_effort.clone(),
        approval_policy: approval_policy.clone(),
        sandbox_policy: sandbox_policy.clone(),
        workspace_id: Some(job.workspace_id.clone()),
        working_dir: working_dir.clone(),
        project_root: project_root.clone(),
        web_search,
        search_mode: search_mode.as_deref().and_then(|mode| match mode {
            "disabled" => Some(app_server_protocol::RuntimeSearchMode::Disabled),
            "auto" => Some(app_server_protocol::RuntimeSearchMode::Auto),
            "required" => Some(app_server_protocol::RuntimeSearchMode::Required),
            _ => None,
        }),
        system_prompt: system_prompt.clone(),
        metadata: Some(runtime_metadata),
        ..app_server_protocol::RuntimeRequest::default()
    };

    Ok(RuntimeOptions {
        capability_id: None,
        stream: true,
        event_name: Some("automation.turn".to_string()),
        queued_turn_id: None,
        runtime_request: Some(runtime_request),
        ..RuntimeOptions::default()
    })
}

fn normalize_cron_expression(expr: &str) -> String {
    let parts = expr.split_whitespace().collect::<Vec<_>>();
    if parts.len() == 5 {
        format!("0 {}", expr.trim())
    } else {
        expr.trim().to_string()
    }
}

fn agent_turn_status_value(status: AgentTurnStatus) -> &'static str {
    match status {
        AgentTurnStatus::Accepted => "accepted",
        AgentTurnStatus::Queued => "queued",
        AgentTurnStatus::Running => "running",
        AgentTurnStatus::WaitingAction => "waiting_action",
        AgentTurnStatus::Completed => "completed",
        AgentTurnStatus::Failed => "failed",
        AgentTurnStatus::Canceled => "canceled",
    }
}

fn string_field(map: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|key| map.get(*key))
        .find_map(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn required_agent_turn_payload_field(
    map: &serde_json::Map<String, Value>,
    keys: &[&str],
) -> Result<String, RuntimeCoreError> {
    string_field(map, keys).ok_or_else(|| {
        RuntimeCoreError::Backend(format!(
            "自动化任务 agent_turn payload 必须显式绑定 {}",
            keys[0]
        ))
    })
}

fn json_pointer_string_value(value: &Value, pointers: &[&str]) -> Option<String> {
    pointers.iter().find_map(|pointer| {
        value
            .pointer(pointer)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
    })
}

fn bool_field(map: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<bool> {
    keys.iter()
        .filter_map(|key| map.get(*key))
        .find_map(|value| match value {
            Value::Bool(flag) => Some(*flag),
            Value::String(raw) => match raw.trim().to_ascii_lowercase().as_str() {
                "true" | "1" | "yes" | "on" | "enabled" => Some(true),
                "false" | "0" | "no" | "off" | "disabled" => Some(false),
                _ => None,
            },
            Value::Number(number) => number.as_i64().map(|value| value != 0),
            _ => None,
        })
}

fn json_strip_nulls(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut next = serde_json::Map::new();
            for (key, value) in map {
                let value = json_strip_nulls(value);
                if !value.is_null() {
                    next.insert(key, value);
                }
            }
            Value::Object(next)
        }
        Value::Array(items) => Value::Array(items.into_iter().map(json_strip_nulls).collect()),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ActionRespondRequest;
    use crate::CancelExecutionRequest;
    use crate::ExecutionBackend;
    use crate::ExecutionRequest;
    use crate::LocalAppDataSource;
    use crate::RuntimeEvent;
    use crate::RuntimeEventSink;
    use app_server_protocol::EvidenceExportParams;
    use lime_core::config::AutomationExecutionMode;
    use lime_core::config::DeliveryConfig;
    use lime_core::database;
    use lime_core::database::dao::agent_run::AgentRunDao;
    use lime_core::database::dao::automation_job::AutomationJobDao;
    use std::sync::Arc;
    use std::sync::Mutex;

    struct RecordingCompletedBackend {
        requests: Mutex<Vec<ExecutionRequest>>,
    }

    #[async_trait::async_trait]
    impl ExecutionBackend for RecordingCompletedBackend {
        async fn start_turn(
            &self,
            request: ExecutionRequest,
            sink: &mut dyn RuntimeEventSink,
        ) -> Result<(), RuntimeCoreError> {
            self.requests
                .lock()
                .expect("test backend requests mutex poisoned")
                .push(request);
            sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
            sink.emit(RuntimeEvent::new(
                "message.delta",
                json!({ "text": "自动化任务已完成" }),
            ))?;
            sink.emit(RuntimeEvent::new("turn.completed", json!({})))
        }

        async fn cancel_turn(
            &self,
            _request: CancelExecutionRequest,
            _sink: &mut dyn RuntimeEventSink,
        ) -> Result<(), RuntimeCoreError> {
            Ok(())
        }

        async fn respond_action(
            &self,
            _request: ActionRespondRequest,
            _sink: &mut dyn RuntimeEventSink,
        ) -> Result<(), RuntimeCoreError> {
            Ok(())
        }
    }

    fn sample_job(payload: Value) -> AutomationJob {
        let now = Utc::now().to_rfc3339();
        AutomationJob {
            id: "job-1".to_string(),
            name: "每日摘要".to_string(),
            description: None,
            enabled: true,
            workspace_id: "workspace-1".to_string(),
            execution_mode: AutomationExecutionMode::Skill,
            schedule: TaskSchedule::Every { every_secs: 300 },
            payload,
            delivery: DeliveryConfig::default(),
            timeout_secs: None,
            max_retries: 3,
            next_run_at: Some(now.clone()),
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
        }
    }

    #[test]
    fn builds_runtime_options_for_agent_turn_payload() {
        let job = sample_job(json!({
            "kind": "agent_turn",
            "prompt": "生成摘要",
            "session_id": "session-job-1",
            "thread_id": "thread-job-1",
            "provider": "openai",
            "model": "gpt-4.1",
            "system_prompt": "请求级提示",
            "reasoningEffort": "high",
            "webSearch": true,
            "searchMode": "required",
            "workingDir": "/tmp/workspace/packages/app",
            "projectRoot": "/tmp/workspace",
            "approvalPolicy": "on-request",
            "sandboxPolicy": "workspace-write",
            "request_metadata": {
                "harness": {
                    "service_skill": {
                        "id": "daily"
                    }
                }
            }
        }));

        let start = build_automation_run_start(job, None).expect("run start");
        let runtime_request = start
            .runtime_options
            .runtime_request
            .expect("runtime request");
        let metadata = runtime_request.metadata.as_ref().expect("metadata");

        assert_eq!(start.prompt, "生成摘要");
        assert_eq!(start.session_id, "session-job-1");
        assert_eq!(start.thread_id, "thread-job-1");
        assert_eq!(
            runtime_request.provider_preference.as_deref(),
            Some("openai")
        );
        assert_eq!(runtime_request.model_preference.as_deref(), Some("gpt-4.1"));
        assert_eq!(
            metadata
                .pointer("/turn_config/reasoning_effort")
                .and_then(Value::as_str),
            Some("high")
        );
        assert_eq!(
            metadata
                .pointer("/turn_config/working_dir")
                .and_then(Value::as_str),
            Some("/tmp/workspace/packages/app")
        );
        assert_eq!(
            metadata
                .pointer("/turn_config/project_root")
                .and_then(Value::as_str),
            Some("/tmp/workspace")
        );
        assert_eq!(runtime_request.system_prompt.as_deref(), Some("请求级提示"));
        assert_eq!(
            runtime_request.search_mode,
            Some(app_server_protocol::RuntimeSearchMode::Required)
        );
        assert_eq!(
            runtime_request
                .metadata
                .as_ref()
                .and_then(|metadata| {
                    metadata.pointer("/request_metadata/harness/service_skill/id")
                })
                .and_then(Value::as_str),
            Some("daily")
        );
    }

    #[test]
    fn rejects_non_agent_turn_payload() {
        let job = sample_job(json!({
            "kind": "browser_session",
            "prompt": "旧浏览器自动化"
        }));

        let error = build_automation_run_start(job, None).expect_err("should reject");
        assert!(error.to_string().contains("browser_session"));
    }

    #[test]
    fn rejects_agent_turn_payload_without_thread_lineage() {
        let job = sample_job(json!({
            "kind": "agent_turn",
            "prompt": "生成摘要",
            "session_id": "session-job-1"
        }));

        let error = build_automation_run_start(job, None).expect_err("should reject");
        assert!(error.to_string().contains("thread_id"));
    }

    #[test]
    fn new_thread_mode_creates_fresh_lineage_for_every_run() {
        let job = sample_job(json!({
            "kind": "agent_turn",
            "prompt": "生成摘要",
            "thread_mode": "new_thread"
        }));

        let first = build_automation_run_start(job.clone(), None).expect("first run start");
        let second = build_automation_run_start(job, None).expect("second run start");

        assert_ne!(first.session_id, second.session_id);
        assert_ne!(first.thread_id, second.thread_id);
        assert!(first.thread_id.starts_with("scheduled-thread-"));
        assert!(second.thread_id.starts_with("scheduled-thread-"));
    }

    #[test]
    fn claimed_run_reuses_run_id_and_projects_schedule_metadata() {
        let scheduled_for = Utc::now().to_rfc3339();
        let claimed_at = Utc::now().to_rfc3339();
        let mut job = sample_job(json!({
            "kind": "agent_turn",
            "prompt": "生成摘要",
            "thread_mode": "new_thread"
        }));
        job.running_started_at = Some(claimed_at.clone());

        let task_revision = job.updated_at.clone();
        let start = build_claimed_automation_run_start(
            job,
            None,
            ClaimedAutomationRun {
                run_id: "scheduled-run-task-1-window-1".to_string(),
                scheduled_for: scheduled_for.clone(),
                claimed_at: claimed_at.clone(),
                task_revision,
                catch_up: true,
                skipped_window_count: 2,
            },
        )
        .expect("claimed run start");
        let metadata: Value =
            serde_json::from_str(start.run.metadata.as_deref().expect("claimed run metadata"))
                .expect("parse claimed run metadata");

        assert_eq!(start.run.id, "scheduled-run-task-1-window-1");
        assert_eq!(metadata["trigger"], "catch_up");
        assert_eq!(metadata["scheduledFor"], scheduled_for);
        assert_eq!(metadata["claimedAt"], claimed_at);
        assert_eq!(metadata["catchUp"], true);
        assert_eq!(metadata["skippedWindowCount"], 2);
        assert_eq!(
            start
                .runtime_options
                .runtime_request
                .as_ref()
                .and_then(|request| request.metadata.as_ref())
                .and_then(|metadata| metadata.pointer("/harness/automation_job/run_id"))
                .and_then(Value::as_str),
            Some("scheduled-run-task-1-window-1")
        );
    }

    #[test]
    fn cron_spring_forward_shifts_missing_local_time_to_first_valid_instant() {
        let schedule = TaskSchedule::Cron {
            expr: "30 2 * * *".to_string(),
            tz: Some("America/New_York".to_string()),
        };
        let before_gap = DateTime::parse_from_rfc3339("2026-03-07T07:30:00Z")
            .expect("parse spring-forward cursor")
            .with_timezone(&Utc);

        let shifted = next_run_for_automation_schedule(&schedule, before_gap)
            .expect("calculate spring-forward run")
            .expect("spring-forward run exists");
        assert_eq!(shifted.to_rfc3339(), "2026-03-08T07:00:00+00:00");
        let following = next_run_for_automation_schedule(&schedule, shifted)
            .expect("calculate post-gap run")
            .expect("post-gap run exists");
        assert_eq!(following.to_rfc3339(), "2026-03-09T06:30:00+00:00");
    }

    #[test]
    fn cron_fall_back_uses_ambiguous_local_time_only_once() {
        let schedule = TaskSchedule::Cron {
            expr: "30 1 * * *".to_string(),
            tz: Some("America/New_York".to_string()),
        };
        let before_repeat = DateTime::parse_from_rfc3339("2026-10-31T05:30:00Z")
            .expect("parse fall-back cursor")
            .with_timezone(&Utc);

        let first = next_run_for_automation_schedule(&schedule, before_repeat)
            .expect("calculate first ambiguous run")
            .expect("ambiguous run exists");
        assert_eq!(first.to_rfc3339(), "2026-11-01T05:30:00+00:00");
        let following = next_run_for_automation_schedule(&schedule, first)
            .expect("calculate run after repeated hour")
            .expect("run after repeated hour exists");
        assert_eq!(following.to_rfc3339(), "2026-11-02T06:30:00+00:00");
    }

    #[tokio::test]
    async fn run_now_executes_agent_turn_and_persists_run_state() {
        let temp = tempfile::tempdir().expect("temp dir");
        let db = database::init_database_at_path(temp.path().join("automation.db"))
            .expect("init test db");
        let data_source = LocalAppDataSource::initialize_with_roots(
            db.clone(),
            temp.path(),
            temp.path().join("app-server"),
        )
        .await
        .expect("local data source");
        let job = sample_job(json!({
            "kind": "agent_turn",
            "prompt": "生成今日摘要",
            "session_id": "session-job-1",
            "thread_id": "thread-job-1",
            "provider": "openai",
            "model": "gpt-4.1",
            "system_prompt": "请求级提示",
            "reasoningEffort": "high",
            "webSearch": true,
            "searchMode": "required",
            "workingDir": "/tmp/workspace/packages/app",
            "projectRoot": "/tmp/workspace",
            "approvalPolicy": "on-request",
            "sandboxPolicy": "workspace-write",
            "request_metadata": {
                "harness": {
                    "service_skill": {
                        "id": "daily"
                    }
                }
            }
        }));
        {
            let conn = database::lock_db(&db).expect("lock db");
            AutomationJobDao::create(&conn, &job).expect("insert job");
        }

        let backend = Arc::new(RecordingCompletedBackend {
            requests: Mutex::new(Vec::new()),
        });
        let core =
            RuntimeCore::with_backend(backend.clone()).with_app_data_source(Arc::new(data_source));

        let response = core
            .run_automation_job_now(
                AutomationJobIdParams {
                    id: "job-1".to_string(),
                },
                RuntimeHostContext::default(),
            )
            .await
            .expect("run automation job");

        assert_eq!(
            response.result.get("status").and_then(Value::as_str),
            Some("completed")
        );
        let requests = backend
            .requests
            .lock()
            .expect("test backend requests mutex poisoned");
        assert_eq!(requests.len(), 1);
        let request = &requests[0];
        assert_eq!(request.session.session_id, "session-job-1");
        assert_eq!(request.session.thread_id, "thread-job-1");
        assert_eq!(request.session.workspace_id.as_deref(), Some("workspace-1"));
        assert_eq!(
            request
                .session
                .business_object_ref
                .as_ref()
                .map(|value| value.kind.as_str()),
            Some("automation_job")
        );
        assert_eq!(
            request
                .session
                .business_object_ref
                .as_ref()
                .map(|value| value.id.as_str()),
            Some("job-1")
        );
        assert_eq!(request.input.concat_text(), "生成今日摘要");
        assert_eq!(request.provider_preference(), Some("openai"));
        assert_eq!(request.model_preference(), Some("gpt-4.1"));
        assert_eq!(
            request
                .runtime_metadata()
                .and_then(|value| value.pointer("/harness/automation_job/session_id"))
                .and_then(Value::as_str),
            Some("session-job-1")
        );
        assert_eq!(
            request
                .runtime_metadata()
                .and_then(|value| value.pointer("/harness/automation_job/thread_id"))
                .and_then(Value::as_str),
            Some("thread-job-1")
        );
        assert_eq!(
            request
                .runtime_options
                .as_ref()
                .and_then(|options| options.runtime_request.as_ref())
                .and_then(|request| request.reasoning_effort.as_deref()),
            Some("high")
        );
        assert_eq!(
            request
                .runtime_metadata()
                .and_then(|value| value.pointer("/turn_config/working_dir"))
                .and_then(Value::as_str),
            Some("/tmp/workspace/packages/app")
        );
        assert_eq!(
            request
                .runtime_metadata()
                .and_then(|value| value.pointer("/turn_config/project_root"))
                .and_then(Value::as_str),
            Some("/tmp/workspace")
        );
        assert_eq!(
            request
                .runtime_options
                .as_ref()
                .and_then(|options| options.runtime_request.as_ref())
                .and_then(|request| request.project_root.as_deref()),
            Some("/tmp/workspace")
        );
        assert_eq!(
            request
                .runtime_metadata()
                .and_then(|value| value.pointer("/harness/automation_job/id"))
                .and_then(Value::as_str),
            Some("job-1")
        );
        assert_eq!(
            response.result.get("session_id").and_then(Value::as_str),
            Some("session-job-1")
        );
        assert_eq!(
            response.result.get("thread_id").and_then(Value::as_str),
            Some("thread-job-1")
        );
        let response_turn_id = response
            .result
            .get("turn_id")
            .and_then(Value::as_str)
            .expect("response turn_id")
            .to_string();
        let evidence_response = core
            .export_evidence(EvidenceExportParams {
                session_id: "session-job-1".to_string(),
                turn_id: Some(response_turn_id.clone()),
                include_events: Some(true),
                include_artifacts: Some(true),
                include_evidence_pack: Some(true),
            })
            .await
            .expect("export automation evidence");
        assert_eq!(evidence_response.session.session_id, "session-job-1");
        assert_eq!(evidence_response.session.thread_id, "thread-job-1");
        assert!(evidence_response.events.iter().any(|event| {
            event.event_type == "message.delta"
                && event.thread_id.as_deref() == Some("thread-job-1")
                && event.turn_id.as_deref() == Some(response_turn_id.as_str())
        }));
        let evidence_pack = evidence_response.evidence_pack.expect("evidence pack");
        assert_eq!(evidence_pack.turn_count, 1);

        let conn = database::lock_db(&db).expect("lock db");
        let updated_job = AutomationJobDao::get(&conn, "job-1")
            .expect("read job")
            .expect("job exists");
        assert_eq!(updated_job.last_status.as_deref(), Some("success"));
        assert!(updated_job.last_finished_at.is_some());
        assert!(updated_job.running_started_at.is_none());
        let runs = AgentRunDao::list_runs_by_source_ref(&conn, AUTOMATION_SOURCE, "job-1", 10)
            .expect("list runs");
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].status, AgentRunStatus::Success);
        assert_eq!(runs[0].session_id.as_deref(), Some("session-job-1"));
        let run_metadata: Value = serde_json::from_str(
            runs[0]
                .metadata
                .as_deref()
                .expect("automation run metadata"),
        )
        .expect("parse automation run metadata");
        assert_eq!(
            run_metadata.get("threadId").and_then(Value::as_str),
            Some("thread-job-1")
        );
    }
}
