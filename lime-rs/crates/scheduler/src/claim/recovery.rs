use super::AutomationWindowClaimDao;
use lime_core::database::dao::agent_run::AgentRunStatus;
use rusqlite::{params, Connection, TransactionBehavior};
use serde_json::{json, Value};

const INTERRUPTED_ERROR_CODE: &str = "scheduled_run_interrupted";
const INTERRUPTED_ERROR_MESSAGE: &str = "应用退出时任务仍在运行，已在重启后标记为中断";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutomationRunRecoveryKind {
    Interrupted,
    Terminal,
    MissingRun,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutomationRunRecovery {
    pub task_id: String,
    pub run_id: Option<String>,
    pub status: AgentRunStatus,
    pub kind: AutomationRunRecoveryKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutomationRunTerminal {
    pub status: AgentRunStatus,
    pub finished_at: String,
    pub duration_ms: Option<i64>,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutomationRunScheduleRecovery {
    pub next_run_at: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AutomationOwnedRun {
    pub id: String,
    pub status: AgentRunStatus,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub duration_ms: Option<i64>,
    pub error_message: Option<String>,
    pub metadata: Option<String>,
}

impl AutomationWindowClaimDao {
    pub fn find_owned_run(
        conn: &Connection,
        task_id: &str,
        ownership_started_at: &str,
    ) -> Result<Option<AutomationOwnedRun>, rusqlite::Error> {
        let mut statement = conn.prepare(
            "SELECT id, status, started_at, finished_at, duration_ms,
                    error_message, metadata
             FROM agent_runs
             WHERE source = 'automation'
               AND source_ref = ?1
             ORDER BY datetime(started_at) DESC, id DESC",
        )?;
        let rows = statement.query_map(params![task_id], |row| {
            let status = row.get::<_, String>(1)?;
            let status = AgentRunStatus::try_from(status.as_str()).map_err(|_| {
                rusqlite::Error::InvalidColumnType(
                    1,
                    "status".to_string(),
                    rusqlite::types::Type::Text,
                )
            })?;
            Ok(AutomationOwnedRun {
                id: row.get(0)?,
                status,
                started_at: row.get(2)?,
                finished_at: row.get(3)?,
                duration_ms: row.get(4)?,
                error_message: row.get(5)?,
                metadata: row.get(6)?,
            })
        })?;
        for run in rows {
            let run = run?;
            if run_owns_task(&run, ownership_started_at) {
                return Ok(Some(run));
            }
        }
        Ok(None)
    }

    /// 收口一次进程退出遗留的 Scheduled Task ownership。
    ///
    /// 调用方负责先按 Scheduled Task marker 过滤任务。本方法在一个 immediate
    /// transaction 内复核 task revision 与 ownership，保证重复启动幂等。
    pub fn recover_stale_run(
        conn: &mut Connection,
        task_id: &str,
        expected_task_revision: &str,
        ownership_started_at: &str,
        recovered_at: &str,
    ) -> Result<Option<AutomationRunRecovery>, rusqlite::Error> {
        Self::recover_stale_run_with_terminal(
            conn,
            task_id,
            expected_task_revision,
            ownership_started_at,
            recovered_at,
            None,
            None,
        )
    }

    pub fn recover_stale_run_with_terminal(
        conn: &mut Connection,
        task_id: &str,
        expected_task_revision: &str,
        ownership_started_at: &str,
        recovered_at: &str,
        canonical_terminal: Option<&AutomationRunTerminal>,
        schedule_recovery: Option<&AutomationRunScheduleRecovery>,
    ) -> Result<Option<AutomationRunRecovery>, rusqlite::Error> {
        let transaction = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let owns_task = transaction.query_row(
            "SELECT EXISTS(
                SELECT 1
                FROM automation_jobs
                WHERE id = ?1
                  AND updated_at = ?2
                  AND running_started_at = ?3
            )",
            params![task_id, expected_task_revision, ownership_started_at],
            |row| row.get::<_, bool>(0),
        )?;
        if !owns_task {
            transaction.commit()?;
            return Ok(None);
        }

        let matching_run = Self::find_owned_run(&transaction, task_id, ownership_started_at)?;

        let recovery = match matching_run {
            Some(run) if run.status.is_terminal() => {
                let finished_at = run
                    .finished_at
                    .as_deref()
                    .unwrap_or(recovered_at)
                    .to_string();
                let changed = finish_task_ownership(
                    &transaction,
                    task_id,
                    expected_task_revision,
                    ownership_started_at,
                    &run.status,
                    &run.started_at,
                    &finished_at,
                    run.error_message.as_deref(),
                    recovered_at,
                    schedule_recovery,
                )?;
                if changed != 1 {
                    return Ok(None);
                }
                AutomationRunRecovery {
                    task_id: task_id.to_string(),
                    run_id: Some(run.id),
                    status: run.status,
                    kind: AutomationRunRecoveryKind::Terminal,
                }
            }
            Some(run)
                if canonical_terminal.is_some_and(|terminal| terminal.status.is_terminal()) =>
            {
                let terminal = canonical_terminal.expect("terminal checked above");
                let run_changed = transaction.execute(
                    "UPDATE agent_runs
                     SET status = ?1,
                         finished_at = ?2,
                         duration_ms = ?3,
                         error_code = ?4,
                         error_message = ?5,
                         updated_at = ?2
                     WHERE id = ?6
                       AND status IN ('queued', 'running')
                       AND finished_at IS NULL",
                    params![
                        terminal.status.as_str(),
                        terminal.finished_at,
                        terminal.duration_ms,
                        terminal.error_code,
                        terminal.error_message,
                        run.id,
                    ],
                )?;
                if run_changed != 1 {
                    return Ok(None);
                }
                let task_changed = finish_task_ownership(
                    &transaction,
                    task_id,
                    expected_task_revision,
                    ownership_started_at,
                    &terminal.status,
                    &run.started_at,
                    &terminal.finished_at,
                    terminal.error_message.as_deref(),
                    recovered_at,
                    schedule_recovery,
                )?;
                if task_changed != 1 {
                    return Ok(None);
                }
                AutomationRunRecovery {
                    task_id: task_id.to_string(),
                    run_id: Some(run.id),
                    status: terminal.status.clone(),
                    kind: AutomationRunRecoveryKind::Terminal,
                }
            }
            Some(run) => {
                let metadata = interrupted_metadata(run.metadata.as_deref(), recovered_at);
                let duration_ms = run
                    .duration_ms
                    .or_else(|| elapsed_millis(&run.started_at, recovered_at));
                let run_changed = transaction.execute(
                    "UPDATE agent_runs
                     SET status = 'error',
                         finished_at = ?1,
                         duration_ms = ?2,
                         error_code = ?3,
                         error_message = ?4,
                         metadata = ?5,
                         updated_at = ?1
                     WHERE id = ?6
                       AND status IN ('queued', 'running')
                       AND finished_at IS NULL",
                    params![
                        recovered_at,
                        duration_ms,
                        INTERRUPTED_ERROR_CODE,
                        INTERRUPTED_ERROR_MESSAGE,
                        metadata,
                        run.id,
                    ],
                )?;
                if run_changed != 1 {
                    return Ok(None);
                }
                let status = AgentRunStatus::Error;
                let task_changed = finish_task_ownership(
                    &transaction,
                    task_id,
                    expected_task_revision,
                    ownership_started_at,
                    &status,
                    &run.started_at,
                    recovered_at,
                    Some(INTERRUPTED_ERROR_MESSAGE),
                    recovered_at,
                    schedule_recovery,
                )?;
                if task_changed != 1 {
                    return Ok(None);
                }
                AutomationRunRecovery {
                    task_id: task_id.to_string(),
                    run_id: Some(run.id),
                    status,
                    kind: AutomationRunRecoveryKind::Interrupted,
                }
            }
            None => {
                let status = AgentRunStatus::Error;
                let task_changed = finish_task_ownership(
                    &transaction,
                    task_id,
                    expected_task_revision,
                    ownership_started_at,
                    &status,
                    ownership_started_at,
                    recovered_at,
                    Some(INTERRUPTED_ERROR_MESSAGE),
                    recovered_at,
                    schedule_recovery,
                )?;
                if task_changed != 1 {
                    return Ok(None);
                }
                AutomationRunRecovery {
                    task_id: task_id.to_string(),
                    run_id: None,
                    status,
                    kind: AutomationRunRecoveryKind::MissingRun,
                }
            }
        };

        transaction.commit()?;
        Ok(Some(recovery))
    }
}

#[allow(clippy::too_many_arguments)]
fn finish_task_ownership(
    conn: &Connection,
    task_id: &str,
    expected_task_revision: &str,
    ownership_started_at: &str,
    status: &AgentRunStatus,
    started_at: &str,
    finished_at: &str,
    error_message: Option<&str>,
    recovered_at: &str,
    schedule_recovery: Option<&AutomationRunScheduleRecovery>,
) -> Result<usize, rusqlite::Error> {
    let failure = matches!(status, AgentRunStatus::Error | AgentRunStatus::Timeout);
    let success = matches!(status, AgentRunStatus::Success);
    conn.execute(
        "UPDATE automation_jobs
         SET running_started_at = NULL,
             last_status = ?1,
             last_error = ?2,
             last_run_at = COALESCE(last_run_at, ?3),
             last_finished_at = ?4,
             consecutive_failures = CASE
                 WHEN ?5 = 1 THEN consecutive_failures + 1
                 WHEN ?6 = 1 THEN 0
                 ELSE consecutive_failures
             END,
             last_retry_count = CASE
                 WHEN ?5 = 1 THEN last_retry_count + 1
                 WHEN ?6 = 1 THEN 0
                 ELSE last_retry_count
             END,
             auto_disabled_until = CASE
                 WHEN ?6 = 1 THEN NULL
                 ELSE auto_disabled_until
             END,
             next_run_at = CASE
                 WHEN ?7 = 1 THEN ?8
                 ELSE next_run_at
             END,
             updated_at = ?9
         WHERE id = ?10
           AND updated_at = ?11
           AND running_started_at = ?12",
        params![
            status.as_str(),
            error_message,
            started_at,
            finished_at,
            failure,
            success,
            schedule_recovery.is_some(),
            schedule_recovery.and_then(|recovery| recovery.next_run_at.as_deref()),
            recovered_at,
            task_id,
            expected_task_revision,
            ownership_started_at,
        ],
    )
}

fn run_owns_task(run: &AutomationOwnedRun, ownership_started_at: &str) -> bool {
    if run.started_at == ownership_started_at {
        return true;
    }
    run.metadata
        .as_deref()
        .and_then(|metadata| serde_json::from_str::<Value>(metadata).ok())
        .and_then(|metadata| {
            metadata
                .get("claimedAt")
                .or_else(|| metadata.get("claimed_at"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
        .as_deref()
        == Some(ownership_started_at)
}

fn interrupted_metadata(metadata: Option<&str>, recovered_at: &str) -> String {
    let mut metadata = metadata
        .and_then(|value| serde_json::from_str::<Value>(value).ok())
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    metadata.insert(
        "recovery".to_string(),
        json!({
            "reason": "app_restart",
            "recoveredAt": recovered_at,
        }),
    );
    Value::Object(metadata).to_string()
}

fn elapsed_millis(started_at: &str, finished_at: &str) -> Option<i64> {
    let started_at = chrono::DateTime::parse_from_rfc3339(started_at).ok()?;
    let finished_at = chrono::DateTime::parse_from_rfc3339(finished_at).ok()?;
    Some(
        finished_at
            .signed_duration_since(started_at)
            .num_milliseconds()
            .max(0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AutomationWindowClaimDao;
    use chrono::{Duration, Utc};
    use lime_core::config::{AutomationExecutionMode, DeliveryConfig, TaskSchedule};
    use lime_core::database::dao::agent_run::AgentRunDao;
    use lime_core::database::dao::automation_job::{AutomationJob, AutomationJobDao};
    use lime_core::database::schema::create_tables;
    use rusqlite::Connection;
    use serde_json::json;

    fn insert_job(conn: &Connection, id: &str, next_run_at: &str) {
        let created_at = (Utc::now() - Duration::hours(1)).to_rfc3339();
        AutomationJobDao::create(
            conn,
            &AutomationJob {
                id: id.to_string(),
                name: id.to_string(),
                description: None,
                enabled: true,
                workspace_id: "workspace-1".to_string(),
                execution_mode: AutomationExecutionMode::Skill,
                schedule: TaskSchedule::Every { every_secs: 3_600 },
                payload: json!({"kind": "agent_turn"}),
                delivery: DeliveryConfig::default(),
                timeout_secs: None,
                max_retries: 3,
                next_run_at: Some(next_run_at.to_string()),
                last_status: None,
                last_error: None,
                last_run_at: None,
                last_finished_at: None,
                running_started_at: None,
                consecutive_failures: 0,
                last_retry_count: 0,
                auto_disabled_until: None,
                last_delivery: None,
                created_at: created_at.clone(),
                updated_at: created_at,
            },
        )
        .expect("insert recovery job");
    }

    fn claim_job(
        conn: &mut Connection,
        id: &str,
        scheduled_for: &str,
        next_run_at: &str,
        now: &str,
    ) -> super::super::AutomationWindowClaim {
        AutomationWindowClaimDao::try_claim_reconciled(
            conn,
            id,
            scheduled_for,
            scheduled_for,
            Some(next_run_at),
            now,
            now,
            false,
            0,
        )
        .expect("claim recovery job")
        .expect("recovery claim exists")
    }

    #[test]
    fn active_run_becomes_interrupted_and_preserves_advanced_window() {
        let mut conn = Connection::open_in_memory().expect("open recovery db");
        create_tables(&conn).expect("create recovery schema");
        let now = Utc::now();
        let due_at = (now - Duration::minutes(1)).to_rfc3339();
        let next_run_at = (now + Duration::minutes(59)).to_rfc3339();
        insert_job(&conn, "interrupted-task", &due_at);
        let claim = claim_job(
            &mut conn,
            "interrupted-task",
            &due_at,
            &next_run_at,
            &now.to_rfc3339(),
        );
        assert!(AutomationWindowClaimDao::mark_started(
            &mut conn,
            &claim,
            &(now + Duration::seconds(1)).to_rfc3339(),
            "session-1",
            &json!({"claimedAt": claim.claimed_at}).to_string(),
        )
        .expect("mark recovery run started"));
        let revision = AutomationJobDao::get(&conn, "interrupted-task")
            .expect("read claimed job")
            .expect("claimed job exists")
            .updated_at;
        let recovered_at = (now + Duration::minutes(5)).to_rfc3339();

        let recovery = AutomationWindowClaimDao::recover_stale_run(
            &mut conn,
            "interrupted-task",
            &revision,
            &claim.claimed_at,
            &recovered_at,
        )
        .expect("recover interrupted run")
        .expect("interrupted recovery result");
        assert_eq!(recovery.kind, AutomationRunRecoveryKind::Interrupted);
        assert_eq!(recovery.status, AgentRunStatus::Error);

        let run = AgentRunDao::get_run(&conn, &claim.run_id)
            .expect("read interrupted run")
            .expect("interrupted run exists");
        assert_eq!(run.status, AgentRunStatus::Error);
        assert_eq!(run.error_code.as_deref(), Some(INTERRUPTED_ERROR_CODE));
        let job = AutomationJobDao::get(&conn, "interrupted-task")
            .expect("read recovered job")
            .expect("recovered job exists");
        assert!(job.running_started_at.is_none());
        assert_eq!(job.next_run_at.as_deref(), Some(next_run_at.as_str()));
        assert_eq!(job.last_status.as_deref(), Some("error"));
        assert_eq!(job.consecutive_failures, 1);

        assert!(AutomationWindowClaimDao::recover_stale_run(
            &mut conn,
            "interrupted-task",
            &job.updated_at,
            &claim.claimed_at,
            &(now + Duration::minutes(6)).to_rfc3339(),
        )
        .expect("repeat recovery")
        .is_none());
    }

    #[test]
    fn terminal_run_closes_task_without_rewriting_run() {
        let mut conn = Connection::open_in_memory().expect("open terminal recovery db");
        create_tables(&conn).expect("create terminal recovery schema");
        let now = Utc::now();
        let due_at = (now - Duration::minutes(1)).to_rfc3339();
        let next_run_at = (now + Duration::minutes(59)).to_rfc3339();
        insert_job(&conn, "terminal-task", &due_at);
        let claim = claim_job(
            &mut conn,
            "terminal-task",
            &due_at,
            &next_run_at,
            &now.to_rfc3339(),
        );
        let finished_at = (now + Duration::seconds(10)).to_rfc3339();
        assert!(AgentRunDao::finish_run(
            &conn,
            &claim.run_id,
            AgentRunStatus::Success,
            &finished_at,
            Some(10_000),
            None,
            None,
            None,
        )
        .expect("finish run before task writeback"));

        let recovery = AutomationWindowClaimDao::recover_stale_run(
            &mut conn,
            "terminal-task",
            &claim.task_revision,
            &claim.claimed_at,
            &(now + Duration::minutes(1)).to_rfc3339(),
        )
        .expect("recover terminal run")
        .expect("terminal recovery result");
        assert_eq!(recovery.kind, AutomationRunRecoveryKind::Terminal);
        assert_eq!(recovery.status, AgentRunStatus::Success);
        let job = AutomationJobDao::get(&conn, "terminal-task")
            .expect("read terminal job")
            .expect("terminal job exists");
        assert!(job.running_started_at.is_none());
        assert_eq!(job.last_status.as_deref(), Some("success"));
        assert_eq!(job.last_finished_at.as_deref(), Some(finished_at.as_str()));
        assert_eq!(job.next_run_at.as_deref(), Some(next_run_at.as_str()));
    }

    #[test]
    fn canonical_terminal_can_close_an_active_agent_run() {
        let mut conn = Connection::open_in_memory().expect("open canonical recovery db");
        create_tables(&conn).expect("create canonical recovery schema");
        let now = Utc::now();
        let due_at = (now - Duration::minutes(1)).to_rfc3339();
        let next_run_at = (now + Duration::minutes(59)).to_rfc3339();
        insert_job(&conn, "canonical-task", &due_at);
        let claim = claim_job(
            &mut conn,
            "canonical-task",
            &due_at,
            &next_run_at,
            &now.to_rfc3339(),
        );
        let finished_at = (now + Duration::seconds(10)).to_rfc3339();
        let terminal = AutomationRunTerminal {
            status: AgentRunStatus::Success,
            finished_at: finished_at.clone(),
            duration_ms: Some(10_000),
            error_code: None,
            error_message: None,
        };

        let recovery = AutomationWindowClaimDao::recover_stale_run_with_terminal(
            &mut conn,
            "canonical-task",
            &claim.task_revision,
            &claim.claimed_at,
            &(now + Duration::minutes(1)).to_rfc3339(),
            Some(&terminal),
            None,
        )
        .expect("recover canonical terminal")
        .expect("canonical terminal recovery result");
        assert_eq!(recovery.kind, AutomationRunRecoveryKind::Terminal);
        let run = AgentRunDao::get_run(&conn, &claim.run_id)
            .expect("read canonical recovered run")
            .expect("canonical recovered run exists");
        assert_eq!(run.status, AgentRunStatus::Success);
        assert_eq!(run.finished_at.as_deref(), Some(finished_at.as_str()));
        let job = AutomationJobDao::get(&conn, "canonical-task")
            .expect("read canonical recovered task")
            .expect("canonical recovered task exists");
        assert!(job.running_started_at.is_none());
        assert_eq!(job.last_status.as_deref(), Some("success"));
    }

    #[test]
    fn revision_change_prevents_stale_recovery_writeback() {
        let mut conn = Connection::open_in_memory().expect("open revision recovery db");
        create_tables(&conn).expect("create revision recovery schema");
        let now = Utc::now();
        let due_at = (now - Duration::minutes(1)).to_rfc3339();
        let next_run_at = (now + Duration::minutes(59)).to_rfc3339();
        insert_job(&conn, "revised-task", &due_at);
        let claim = claim_job(
            &mut conn,
            "revised-task",
            &due_at,
            &next_run_at,
            &now.to_rfc3339(),
        );
        conn.execute(
            "UPDATE automation_jobs SET updated_at = ?1 WHERE id = ?2",
            params![(now + Duration::seconds(1)).to_rfc3339(), "revised-task"],
        )
        .expect("revise task before recovery");

        assert!(AutomationWindowClaimDao::recover_stale_run(
            &mut conn,
            "revised-task",
            &claim.task_revision,
            &claim.claimed_at,
            &(now + Duration::minutes(1)).to_rfc3339(),
        )
        .expect("reject stale recovery")
        .is_none());
        let run = AgentRunDao::get_run(&conn, &claim.run_id)
            .expect("read unrecovered run")
            .expect("unrecovered run exists");
        assert_eq!(run.status, AgentRunStatus::Queued);
    }
}
