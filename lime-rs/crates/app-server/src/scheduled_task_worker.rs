use crate::automation_execution::{
    build_claimed_automation_run_start, next_run_for_automation_schedule, AutomationRunFailure,
    AutomationRunIdentity, ClaimedAutomationRun,
};
use crate::local_data_source::is_scheduled_task_job;
use crate::{RuntimeCore, RuntimeCoreError, RuntimeHostContext};
use app_server_protocol::ScheduledTaskThreadMode;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use lime_core::database;
use lime_core::database::dao::agent_run::{AgentRunDao, AgentRunStatus};
use lime_core::database::dao::automation_job::{AutomationJob, AutomationJobDao};
use lime_core::database::DbConnection;
use lime_scheduler::{
    AutomationRunScheduleRecovery, AutomationRunTerminal, AutomationWindowClaim,
    AutomationWindowClaimDao,
};
use serde_json::{json, Value};
use std::time::Duration;
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;

const DEFAULT_SCAN_INTERVAL: Duration = Duration::from_secs(30);
const DEFAULT_SCAN_LIMIT: usize = 32;
const CATCH_UP_WINDOW_HOURS: i64 = 24;
const MAX_RECONCILED_WINDOWS: u32 = 100_000;

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReconciledDueWindow {
    expected_next_run_at: String,
    scheduled_for: String,
    next_run_at: Option<String>,
    skipped_window_count: u32,
    catch_up: bool,
    within_catch_up_window: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ScheduledTaskWorkerConfig {
    scan_interval: Duration,
    scan_limit: usize,
}

impl Default for ScheduledTaskWorkerConfig {
    fn default() -> Self {
        Self {
            scan_interval: DEFAULT_SCAN_INTERVAL,
            scan_limit: DEFAULT_SCAN_LIMIT,
        }
    }
}

pub fn spawn_scheduled_task_worker(db: DbConnection, runtime: RuntimeCore) -> JoinHandle<()> {
    spawn_scheduled_task_worker_with_config(db, runtime, ScheduledTaskWorkerConfig::default())
}

fn spawn_scheduled_task_worker_with_config(
    db: DbConnection,
    runtime: RuntimeCore,
    config: ScheduledTaskWorkerConfig,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let recovered_at = Utc::now().to_rfc3339();
        match recover_stale_scheduled_task_runs(&db, &runtime, &recovered_at).await {
            Ok(recoveries) => {
                for recovery in recoveries {
                    tracing::info!(
                        task_id = %recovery.task_id,
                        run_id = recovery.run_id.as_deref().unwrap_or("missing"),
                        status = recovery.status.as_str(),
                        kind = ?recovery.kind,
                        "recovered stale scheduled task run"
                    );
                }
            }
            Err(error) => {
                tracing::warn!(error = %error, "failed to recover stale scheduled task runs");
            }
        }
        let mut interval = tokio::time::interval(config.scan_interval);
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
        loop {
            interval.tick().await;
            let now = Utc::now().to_rfc3339();
            match claim_due_scheduled_tasks(&db, &now, config.scan_limit) {
                Ok(claims) => {
                    for claim in claims {
                        let db = db.clone();
                        let runtime = runtime.clone();
                        tokio::spawn(async move {
                            let task_id = claim.job.id.clone();
                            let run_id = claim.run_id.clone();
                            if let Err(error) =
                                execute_claimed_scheduled_task(&db, &runtime, claim).await
                            {
                                tracing::warn!(
                                    task_id = %task_id,
                                    run_id = %run_id,
                                    error = %error,
                                    "scheduled task execution failed"
                                );
                            }
                        });
                    }
                }
                Err(error) => {
                    tracing::warn!(error = %error, "failed to scan due scheduled tasks");
                }
            }
        }
    })
}

async fn recover_stale_scheduled_task_runs(
    db: &DbConnection,
    runtime: &RuntimeCore,
    recovered_at: &str,
) -> Result<Vec<lime_scheduler::AutomationRunRecovery>, String> {
    let candidates = {
        let conn = database::lock_db(db)?;
        AutomationJobDao::list(&conn).map_err(|error| error.to_string())?
    };
    let mut recoveries = Vec::new();
    for job in candidates.into_iter().filter(is_scheduled_task_job) {
        let Some(ownership_started_at) = job.running_started_at.as_deref() else {
            continue;
        };
        let matching_run = {
            let conn = database::lock_db(db)?;
            AutomationWindowClaimDao::find_owned_run(&conn, &job.id, ownership_started_at)
                .map_err(|error| error.to_string())?
        };
        let canonical_terminal = match matching_run.as_ref() {
            Some(run) if matches!(run.status, AgentRunStatus::Queued | AgentRunStatus::Running) => {
                canonical_terminal_for_run(runtime, run, recovered_at).await
            }
            _ => None,
        };
        let schedule_recovery = matching_run
            .as_ref()
            .and_then(|run| scheduled_run_task_revision(run, ownership_started_at))
            .filter(|revision| revision != &job.updated_at)
            .map(|_| recompute_recovered_next_run(&job, recovered_at))
            .transpose()?;
        let mut conn = database::lock_db(db)?;
        if let Some(recovery) = AutomationWindowClaimDao::recover_stale_run_with_terminal(
            &mut conn,
            &job.id,
            &job.updated_at,
            ownership_started_at,
            recovered_at,
            canonical_terminal.as_ref(),
            schedule_recovery.as_ref(),
        )
        .map_err(|error| error.to_string())?
        {
            recoveries.push(recovery);
        }
    }
    Ok(recoveries)
}

fn scheduled_run_task_revision(
    run: &lime_scheduler::AutomationOwnedRun,
    ownership_started_at: &str,
) -> Option<String> {
    let revision = run
        .metadata
        .as_deref()
        .and_then(|metadata| serde_json::from_str::<Value>(metadata).ok())
        .and_then(|metadata| {
            metadata
                .get("taskRevision")
                .or_else(|| metadata.get("task_revision"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        });
    revision.or_else(|| (run.started_at == ownership_started_at).then(|| run.started_at.clone()))
}

fn recompute_recovered_next_run(
    job: &AutomationJob,
    recovered_at: &str,
) -> Result<AutomationRunScheduleRecovery, String> {
    let recovered_at = DateTime::parse_from_rfc3339(recovered_at)
        .map_err(|error| format!("scheduled task recovery time is invalid: {error}"))?
        .with_timezone(&Utc);
    let next_run_at = if job.enabled {
        next_run_for_automation_schedule(&job.schedule, recovered_at)?
            .map(|value| value.to_rfc3339())
    } else {
        None
    };
    Ok(AutomationRunScheduleRecovery { next_run_at })
}

async fn canonical_terminal_for_run(
    runtime: &RuntimeCore,
    run: &lime_scheduler::AutomationOwnedRun,
    recovered_at: &str,
) -> Option<AutomationRunTerminal> {
    let metadata = run
        .metadata
        .as_deref()
        .and_then(|metadata| serde_json::from_str::<Value>(metadata).ok())?;
    let thread_id = metadata
        .get("threadId")
        .or_else(|| metadata.get("thread_id"))
        .and_then(Value::as_str)?;
    let turn_id = metadata
        .get("turnId")
        .or_else(|| metadata.get("turn_id"))
        .and_then(Value::as_str)?;
    let response = runtime
        .read_thread(app_server_protocol::ThreadReadParams {
            thread_id: agent_protocol::ThreadId::new(thread_id),
            turns_view: agent_protocol::ThreadTurnsView::Full,
        })
        .await
        .ok()?;
    let turn = response
        .thread
        .turns
        .iter()
        .find(|turn| turn.turn_id.as_str() == turn_id && turn.status.is_terminal())?;
    let (status, default_error_code) = match turn.status {
        agent_protocol::TurnStatus::Completed => (AgentRunStatus::Success, None),
        agent_protocol::TurnStatus::Interrupted => {
            (AgentRunStatus::Canceled, Some("scheduled_turn_interrupted"))
        }
        agent_protocol::TurnStatus::Failed => {
            (AgentRunStatus::Error, Some("scheduled_turn_failed"))
        }
        agent_protocol::TurnStatus::InProgress => return None,
    };
    let finished_at = turn
        .completed_at_ms
        .and_then(|value| chrono::DateTime::<Utc>::from_timestamp_millis(value))
        .map(|value| value.to_rfc3339())
        .unwrap_or_else(|| recovered_at.to_string());
    Some(AutomationRunTerminal {
        status,
        finished_at,
        duration_ms: turn
            .duration_ms
            .map(|value| i64::try_from(value).unwrap_or(i64::MAX)),
        error_code: turn
            .error
            .as_ref()
            .and_then(|error| error.code.clone())
            .or_else(|| default_error_code.map(ToOwned::to_owned)),
        error_message: turn.error.as_ref().map(|error| error.message.clone()),
    })
}

fn claim_due_scheduled_tasks(
    db: &DbConnection,
    now: &str,
    limit: usize,
) -> Result<Vec<AutomationWindowClaim>, String> {
    let mut conn = database::lock_db(db)?;
    let candidates = AutomationWindowClaimDao::list_due_candidates(&conn, now)
        .map_err(|error| error.to_string())?;
    let overlap_candidates = AutomationWindowClaimDao::list_overlap_candidates(&conn, now)
        .map_err(|error| error.to_string())?;
    let mut claims = Vec::new();
    for job in candidates
        .into_iter()
        .filter(is_scheduled_task_job)
        .take(limit)
    {
        let window = reconcile_due_window(&job, now)?;
        if window.within_catch_up_window {
            if let Some(claim) = AutomationWindowClaimDao::try_claim_reconciled(
                &mut conn,
                &job.id,
                &window.expected_next_run_at,
                &window.scheduled_for,
                window.next_run_at.as_deref(),
                now,
                now,
                window.catch_up,
                window.skipped_window_count,
            )
            .map_err(|error| error.to_string())?
            {
                claims.push(claim);
            }
            continue;
        }
        AutomationWindowClaimDao::record_missed_and_advance(
            &mut conn,
            &job.id,
            &job.updated_at,
            &window.expected_next_run_at,
            &window.scheduled_for,
            window.next_run_at.as_deref(),
            now,
            window.skipped_window_count,
            &window.scheduled_for,
        )
        .map_err(|error| error.to_string())?;
    }
    for job in overlap_candidates
        .into_iter()
        .filter(is_scheduled_task_job)
        .take(limit.saturating_sub(claims.len()))
    {
        let window = reconcile_due_window(&job, now)?;
        let Some(running_started_at) = job.running_started_at.as_deref() else {
            continue;
        };
        AutomationWindowClaimDao::record_overlap_missed_and_advance(
            &mut conn,
            &job.id,
            &job.updated_at,
            &window.expected_next_run_at,
            &window.scheduled_for,
            window.next_run_at.as_deref(),
            now,
            window.skipped_window_count,
            running_started_at,
        )
        .map_err(|error| error.to_string())?;
    }
    Ok(claims)
}

fn reconcile_due_window(job: &AutomationJob, now: &str) -> Result<ReconciledDueWindow, String> {
    let expected_next_run_at = job
        .next_run_at
        .as_deref()
        .ok_or_else(|| format!("已安排任务缺少 next_run_at: {}", job.id))?;
    let first_due = DateTime::parse_from_rfc3339(expected_next_run_at)
        .map_err(|error| format!("已安排任务 next_run_at 无效: {error}"))?
        .with_timezone(&Utc);
    let now = DateTime::parse_from_rfc3339(now)
        .map_err(|error| format!("scheduler now 无效: {error}"))?
        .with_timezone(&Utc);
    if first_due > now {
        return Err(format!("已安排任务尚未到期: {}", job.id));
    }

    let mut latest_due = first_due;
    let mut due_window_count = 1_u32;
    let next_run_at = loop {
        let Some(next) = next_run_for_automation_schedule(&job.schedule, latest_due)? else {
            break None;
        };
        if next > now {
            break Some(next.to_rfc3339());
        }
        if next <= latest_due {
            return Err(format!("已安排任务日程未向前推进: {}", job.id));
        }
        latest_due = next;
        due_window_count = due_window_count.saturating_add(1);
        if due_window_count > MAX_RECONCILED_WINDOWS {
            return Err(format!("已安排任务漏跑窗口过多: {}", job.id));
        }
    };
    let skipped_window_count = due_window_count.saturating_sub(1);
    let delay = now.signed_duration_since(latest_due);
    let within_catch_up_window = delay <= ChronoDuration::hours(CATCH_UP_WINDOW_HOURS);
    let catch_up = skipped_window_count > 0 && within_catch_up_window;

    Ok(ReconciledDueWindow {
        expected_next_run_at: expected_next_run_at.to_string(),
        scheduled_for: latest_due.to_rfc3339(),
        next_run_at,
        skipped_window_count,
        catch_up,
        within_catch_up_window,
    })
}

async fn execute_claimed_scheduled_task(
    db: &DbConnection,
    runtime: &RuntimeCore,
    claim: AutomationWindowClaim,
) -> Result<(), RuntimeCoreError> {
    if !claim_is_current(db, &claim)? {
        invalidate_claim(db, &claim, "task changed after claim")?;
        return Ok(());
    }

    let identity = match resolve_claim_identity(runtime, &claim).await {
        Ok(identity) => identity,
        Err(error) => {
            fail_claimed_run(
                db,
                runtime,
                &claim,
                "scheduled_task_identity_failed",
                &error.to_string(),
            )
            .await?;
            return Err(error);
        }
    };

    if !claim_is_current(db, &claim)? {
        invalidate_claim(db, &claim, "task changed while resolving thread identity")?;
        return Ok(());
    }

    let start = match build_claimed_automation_run_start(
        claim.job.clone(),
        identity,
        ClaimedAutomationRun {
            run_id: claim.run_id.clone(),
            scheduled_for: claim.scheduled_for.clone(),
            claimed_at: claim.claimed_at.clone(),
            task_revision: claim.task_revision.clone(),
            catch_up: claim.catch_up,
            skipped_window_count: claim.skipped_window_count,
        },
    ) {
        Ok(start) => start,
        Err(error) => {
            fail_claimed_run(
                db,
                runtime,
                &claim,
                "scheduled_task_start_invalid",
                &error.to_string(),
            )
            .await?;
            return Err(error);
        }
    };
    let metadata = start
        .run
        .metadata
        .clone()
        .unwrap_or_else(|| "{}".to_string());
    let started = {
        let mut conn = database::lock_db(db).map_err(RuntimeCoreError::Backend)?;
        AutomationWindowClaimDao::mark_started(
            &mut conn,
            &claim,
            &start.run.started_at,
            &start.session_id,
            &metadata,
        )
        .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?
    };
    if !started {
        invalidate_claim(db, &claim, "task changed before runtime start")?;
        return Ok(());
    }

    runtime
        .execute_started_automation_job(claim.job.id.clone(), start, RuntimeHostContext::default())
        .await?;
    Ok(())
}

fn claim_is_current(
    db: &DbConnection,
    claim: &AutomationWindowClaim,
) -> Result<bool, RuntimeCoreError> {
    let conn = database::lock_db(db).map_err(RuntimeCoreError::Backend)?;
    AutomationWindowClaimDao::is_current(&conn, claim)
        .map_err(|error| RuntimeCoreError::Backend(error.to_string()))
}

fn invalidate_claim(
    db: &DbConnection,
    claim: &AutomationWindowClaim,
    reason: &str,
) -> Result<(), RuntimeCoreError> {
    let mut conn = database::lock_db(db).map_err(RuntimeCoreError::Backend)?;
    AutomationWindowClaimDao::invalidate(&mut conn, claim, &Utc::now().to_rfc3339(), reason)
        .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;
    Ok(())
}

async fn resolve_claim_identity(
    runtime: &RuntimeCore,
    claim: &AutomationWindowClaim,
) -> Result<Option<AutomationRunIdentity>, RuntimeCoreError> {
    let payload =
        claim.job.payload.as_object().ok_or_else(|| {
            RuntimeCoreError::Backend("已安排任务 payload 必须为对象".to_string())
        })?;
    let thread_mode = serde_json::from_value::<ScheduledTaskThreadMode>(
        payload
            .get("thread_mode")
            .cloned()
            .ok_or_else(|| RuntimeCoreError::Backend("已安排任务缺少 thread_mode".to_string()))?,
    )
    .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;
    match thread_mode {
        ScheduledTaskThreadMode::NewThread => Ok(None),
        ScheduledTaskThreadMode::ContinueThread => {
            let thread_id = payload
                .get("source_thread_id")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    RuntimeCoreError::InvalidRequest(
                        "continue_thread 必须提供 sourceThreadId".to_string(),
                    )
                })?;
            let resumed = runtime
                .resume_thread(agent_protocol::ThreadId::new(thread_id))
                .await?;
            Ok(Some(AutomationRunIdentity {
                session_id: resumed.thread.session_id.to_string(),
                thread_id: resumed.thread.thread_id.to_string(),
            }))
        }
    }
}

async fn fail_claimed_run(
    db: &DbConnection,
    runtime: &RuntimeCore,
    claim: &AutomationWindowClaim,
    error_code: &str,
    error_message: &str,
) -> Result<(), RuntimeCoreError> {
    let run = {
        let conn = database::lock_db(db).map_err(RuntimeCoreError::Backend)?;
        AgentRunDao::get_run(&conn, &claim.run_id)
            .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?
    };
    runtime
        .fail_automation_job_run(AutomationRunFailure {
            job: claim.job.clone(),
            run,
            status: AgentRunStatus::Error,
            finished_at: Utc::now().to_rfc3339(),
            duration_ms: None,
            error_code: error_code.to_string(),
            error_message: error_message.to_string(),
            metadata: json!({
                "jobId": claim.job.id,
                "runId": claim.run_id,
                "trigger": if claim.catch_up { "catch_up" } else { "schedule" },
                "scheduledFor": claim.scheduled_for,
                "claimedAt": claim.claimed_at,
                "catchUp": claim.catch_up,
                "skippedWindowCount": claim.skipped_window_count,
            }),
            ownership_started_at: Some(claim.claimed_at.clone()),
            task_revision: Some(claim.task_revision.clone()),
        })
        .await
}

#[cfg(test)]
mod tests;
