//! Scheduled Task 窗口抢占。
//!
//! 当前任务事实源是 `automation_jobs`。本模块只提供原子 claim/release，
//! 不读写旧 `scheduled_tasks` 表，也不承担 RuntimeCore 执行。

use lime_core::database::dao::agent_run::{AgentRunDao, AgentRunStatus};
use lime_core::database::dao::automation_job::{AutomationJob, AutomationJobDao};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};

mod recovery;

pub use recovery::{
    AutomationOwnedRun, AutomationRunRecovery, AutomationRunRecoveryKind,
    AutomationRunScheduleRecovery, AutomationRunTerminal,
};

#[derive(Debug, Clone)]
pub struct AutomationWindowClaim {
    pub job: AutomationJob,
    pub scheduled_for: String,
    pub next_run_at: Option<String>,
    pub claimed_at: String,
    pub task_revision: String,
    pub claim_key: String,
    pub run_id: String,
    previous_status: Option<String>,
    pub catch_up: bool,
    pub skipped_window_count: u32,
}

pub struct AutomationWindowClaimDao;

impl AutomationWindowClaimDao {
    pub fn list_due_candidates(
        conn: &Connection,
        now: &str,
    ) -> Result<Vec<AutomationJob>, rusqlite::Error> {
        let now =
            chrono::DateTime::parse_from_rfc3339(now).map_err(|_| rusqlite::Error::InvalidQuery)?;
        let mut candidates = AutomationJobDao::list(conn)?
            .into_iter()
            .filter(|job| {
                if !job.enabled || job.running_started_at.is_some() {
                    return false;
                }
                let due = job
                    .next_run_at
                    .as_deref()
                    .and_then(|value| chrono::DateTime::parse_from_rfc3339(value).ok())
                    .is_some_and(|value| value <= now);
                let cooldown_elapsed = job
                    .auto_disabled_until
                    .as_deref()
                    .and_then(|value| chrono::DateTime::parse_from_rfc3339(value).ok())
                    .is_none_or(|value| value <= now);
                due && cooldown_elapsed
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| left.next_run_at.cmp(&right.next_run_at));
        Ok(candidates)
    }

    pub fn list_overlap_candidates(
        conn: &Connection,
        now: &str,
    ) -> Result<Vec<AutomationJob>, rusqlite::Error> {
        let now =
            chrono::DateTime::parse_from_rfc3339(now).map_err(|_| rusqlite::Error::InvalidQuery)?;
        let mut candidates = AutomationJobDao::list(conn)?
            .into_iter()
            .filter(|job| {
                job.enabled
                    && job.running_started_at.is_some()
                    && job.next_run_at.as_deref().is_some_and(|value| {
                        chrono::DateTime::parse_from_rfc3339(value).is_ok_and(|value| value <= now)
                    })
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| left.next_run_at.cmp(&right.next_run_at));
        Ok(candidates)
    }

    pub fn try_claim(
        conn: &mut Connection,
        task_id: &str,
        scheduled_for: &str,
        now: &str,
        claimed_at: &str,
    ) -> Result<Option<AutomationWindowClaim>, rusqlite::Error> {
        Self::try_claim_reconciled(
            conn,
            task_id,
            scheduled_for,
            scheduled_for,
            Some(scheduled_for),
            now,
            claimed_at,
            false,
            0,
        )
    }

    pub fn try_claim_reconciled(
        conn: &mut Connection,
        task_id: &str,
        expected_next_run_at: &str,
        scheduled_for: &str,
        next_run_at: Option<&str>,
        now: &str,
        claimed_at: &str,
        catch_up: bool,
        skipped_window_count: u32,
    ) -> Result<Option<AutomationWindowClaim>, rusqlite::Error> {
        let transaction = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let candidate = transaction
            .query_row(
                "SELECT updated_at, last_status
                 FROM automation_jobs
                 WHERE id = ?1
                   AND enabled = 1
                   AND next_run_at = ?2
                   AND datetime(next_run_at) <= datetime(?3)
                   AND running_started_at IS NULL
                   AND (
                        auto_disabled_until IS NULL
                        OR datetime(auto_disabled_until) <= datetime(?3)
                   )",
                params![task_id, expected_next_run_at, now],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)),
            )
            .optional()?;
        let Some((task_revision, previous_status)) = candidate else {
            transaction.commit()?;
            return Ok(None);
        };

        let claim_key = format!("{task_id}:{scheduled_for}");
        let run_id = format!("scheduled-run-{task_id}-{scheduled_for}");
        let trigger = if catch_up { "catch_up" } else { "schedule" };
        let metadata = serde_json::to_string(&serde_json::json!({
            "taskId": task_id,
            "trigger": trigger,
            "scheduledFor": scheduled_for,
            "claimKey": claim_key,
            "taskRevision": task_revision,
            "catchUp": catch_up,
            "skippedWindowCount": skipped_window_count,
        }))
        .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
        let inserted = transaction.execute(
            "INSERT OR IGNORE INTO agent_runs (
                id, source, source_ref, session_id, status, started_at, finished_at,
                duration_ms, error_code, error_message, metadata, created_at, updated_at
             ) VALUES (
                ?1, 'automation', ?2, NULL, 'queued', ?3, NULL,
                NULL, NULL, NULL, ?4, ?3, ?3
             )",
            params![run_id, task_id, claimed_at, metadata],
        )?;
        if inserted != 1 {
            return Ok(None);
        }

        let changed = transaction.execute(
            "UPDATE automation_jobs
             SET running_started_at = ?1,
                 next_run_at = ?2,
                 last_status = 'queued',
                 last_error = NULL
             WHERE id = ?3
               AND enabled = 1
               AND next_run_at = ?4
               AND updated_at = ?5
               AND running_started_at IS NULL
               AND datetime(next_run_at) <= datetime(?6)
               AND (
                    auto_disabled_until IS NULL
                    OR datetime(auto_disabled_until) <= datetime(?6)
               )",
            params![
                claimed_at,
                next_run_at,
                task_id,
                expected_next_run_at,
                task_revision,
                now
            ],
        )?;
        if changed != 1 {
            return Ok(None);
        }

        let job = AutomationJobDao::get(&transaction, task_id)?
            .ok_or_else(|| rusqlite::Error::QueryReturnedNoRows)?;
        transaction.commit()?;
        Ok(Some(AutomationWindowClaim {
            job,
            scheduled_for: scheduled_for.to_string(),
            next_run_at: next_run_at.map(ToOwned::to_owned),
            claimed_at: claimed_at.to_string(),
            task_revision,
            claim_key,
            run_id,
            previous_status,
            catch_up,
            skipped_window_count,
        }))
    }

    pub fn is_current(
        conn: &Connection,
        claim: &AutomationWindowClaim,
    ) -> Result<bool, rusqlite::Error> {
        conn.query_row(
            "SELECT EXISTS(
                SELECT 1
                FROM automation_jobs
                WHERE id = ?1
                  AND enabled = 1
                  AND updated_at = ?2
                  AND running_started_at = ?3
            )",
            params![claim.job.id, claim.task_revision, claim.claimed_at,],
            |row| row.get::<_, bool>(0),
        )
    }

    pub fn mark_started(
        conn: &mut Connection,
        claim: &AutomationWindowClaim,
        started_at: &str,
        session_id: &str,
        metadata: &str,
    ) -> Result<bool, rusqlite::Error> {
        let transaction = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let task_changed = transaction.execute(
            "UPDATE automation_jobs
             SET last_status = 'running',
                 last_run_at = ?1,
                 last_error = NULL
             WHERE id = ?2
               AND enabled = 1
               AND updated_at = ?3
               AND running_started_at = ?4",
            params![
                started_at,
                claim.job.id,
                claim.task_revision,
                claim.claimed_at,
            ],
        )?;
        if task_changed != 1 {
            transaction.commit()?;
            return Ok(false);
        }

        let run_changed = transaction.execute(
            "UPDATE agent_runs
             SET status = 'running',
                 session_id = ?1,
                 started_at = ?2,
                 metadata = ?3,
                 updated_at = ?2
             WHERE id = ?4
               AND status = 'queued'
               AND finished_at IS NULL",
            params![session_id, started_at, metadata, claim.run_id],
        )?;
        if run_changed != 1 {
            return Ok(false);
        }
        transaction.commit()?;
        Ok(true)
    }

    /// 将一组无法补跑的到期窗口折叠成一条 missed 历史，并推进下一次运行。
    ///
    /// `task_revision` 和 `scheduled_for` 共同构成 CAS，任务编辑、暂停或删除后
    /// 不会覆盖用户的新配置。
    pub fn record_missed_and_advance(
        conn: &mut Connection,
        task_id: &str,
        task_revision: &str,
        expected_next_run_at: &str,
        scheduled_for: &str,
        next_run_at: Option<&str>,
        finished_at: &str,
        skipped_window_count: u32,
        latest_due_at: &str,
    ) -> Result<bool, rusqlite::Error> {
        let transaction = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let run_id = format!("scheduled-run-{task_id}-{scheduled_for}");
        let metadata = serde_json::to_string(&serde_json::json!({
            "jobId": task_id,
            "trigger": "schedule",
            "scheduledFor": scheduled_for,
            "latestDueAt": latest_due_at,
            "catchUp": false,
            "skippedWindowCount": skipped_window_count,
            "missedReason": "outside_catch_up_window",
        }))
        .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
        let changed = transaction.execute(
            "UPDATE automation_jobs
             SET next_run_at = ?1,
                 last_status = 'missed',
                 last_error = ?2,
                 last_run_at = ?3,
                 last_finished_at = ?4,
                 running_started_at = NULL,
                 updated_at = ?4
             WHERE id = ?5
               AND updated_at = ?6
               AND next_run_at = ?7
               AND running_started_at IS NULL",
            params![
                next_run_at,
                "错过运行窗口，超过 24 小时补跑范围",
                scheduled_for,
                finished_at,
                task_id,
                task_revision,
                expected_next_run_at,
            ],
        )?;
        if changed != 1 {
            transaction.commit()?;
            return Ok(false);
        }
        transaction.execute(
            "INSERT OR IGNORE INTO agent_runs (
                id, source, source_ref, session_id, status, started_at, finished_at,
                duration_ms, error_code, error_message, metadata, created_at, updated_at
             ) VALUES (
                ?1, 'automation', ?2, NULL, 'missed', ?3, ?4,
                NULL, 'scheduled_run_missed', ?5, ?6, ?4, ?4
             )",
            params![
                run_id,
                task_id,
                scheduled_for,
                finished_at,
                "错过运行窗口，超过 24 小时补跑范围",
                metadata,
            ],
        )?;
        transaction.commit()?;
        Ok(true)
    }

    pub fn record_overlap_missed_and_advance(
        conn: &mut Connection,
        task_id: &str,
        task_revision: &str,
        expected_next_run_at: &str,
        scheduled_for: &str,
        next_run_at: Option<&str>,
        finished_at: &str,
        skipped_window_count: u32,
        running_started_at: &str,
    ) -> Result<bool, rusqlite::Error> {
        let transaction = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let run_id = format!("scheduled-run-{task_id}-{scheduled_for}");
        let metadata = serde_json::to_string(&serde_json::json!({
            "jobId": task_id,
            "trigger": "schedule",
            "scheduledFor": scheduled_for,
            "catchUp": false,
            "skippedWindowCount": skipped_window_count,
            "missedReason": "overlap_skip_if_running",
        }))
        .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
        let changed = transaction.execute(
            "UPDATE automation_jobs
             SET next_run_at = ?1
             WHERE id = ?2
               AND updated_at = ?3
               AND next_run_at = ?4
               AND running_started_at = ?5",
            params![
                next_run_at,
                task_id,
                task_revision,
                expected_next_run_at,
                running_started_at,
            ],
        )?;
        if changed != 1 {
            transaction.commit()?;
            return Ok(false);
        }
        transaction.execute(
            "INSERT OR IGNORE INTO agent_runs (
                id, source, source_ref, session_id, status, started_at, finished_at,
                duration_ms, error_code, error_message, metadata, created_at, updated_at
             ) VALUES (
                ?1, 'automation', ?2, NULL, 'missed', ?3, ?4,
                NULL, 'scheduled_run_overlap', ?5, ?6, ?4, ?4
             )",
            params![
                run_id,
                task_id,
                scheduled_for,
                finished_at,
                "任务已在运行，跳过重叠窗口",
                metadata,
            ],
        )?;
        transaction.commit()?;
        Ok(true)
    }

    pub fn invalidate(
        conn: &mut Connection,
        claim: &AutomationWindowClaim,
        invalidated_at: &str,
        reason: &str,
    ) -> Result<bool, rusqlite::Error> {
        let transaction = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let task_changed = transaction.execute(
            "UPDATE automation_jobs
             SET running_started_at = NULL,
                 last_status = CASE
                     WHEN last_status = 'queued' THEN ?1
                     ELSE last_status
                 END
             WHERE id = ?2
               AND running_started_at = ?3",
            params![claim.previous_status, claim.job.id, claim.claimed_at],
        )?;
        let run_finished = AgentRunDao::finish_run(
            &transaction,
            &claim.run_id,
            AgentRunStatus::Canceled,
            invalidated_at,
            None,
            Some("scheduled_claim_invalidated"),
            Some(reason),
            None,
        )?;
        transaction.commit()?;
        Ok(task_changed == 1 || run_finished)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};
    use lime_core::config::{AutomationExecutionMode, DeliveryConfig, TaskSchedule};
    use lime_core::database::schema::create_tables;
    use serde_json::json;
    use tempfile::TempDir;

    fn insert_job(conn: &Connection, id: &str, next_run_at: &str) {
        let now = Utc::now().to_rfc3339();
        AutomationJobDao::create(
            conn,
            &AutomationJob {
                id: id.to_string(),
                name: "每日简报".to_string(),
                description: None,
                enabled: true,
                workspace_id: "workspace-1".to_string(),
                execution_mode: AutomationExecutionMode::Skill,
                schedule: TaskSchedule::Every { every_secs: 3_600 },
                payload: json!({
                    "kind": "agent_turn",
                    "thread_mode": "new_thread",
                    "scheduled_task_schedule": {
                        "type": "daily",
                        "time": "08:30",
                        "timezone": "Asia/Shanghai"
                    },
                    "notification_policy": "failures",
                    "overlap_policy": "skip_if_running"
                }),
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
                created_at: now.clone(),
                updated_at: now,
            },
        )
        .expect("insert automation job");
    }

    #[test]
    fn only_one_claim_owns_a_scheduled_window() {
        let mut conn = Connection::open_in_memory().expect("open scheduler test db");
        create_tables(&conn).expect("create scheduler test schema");
        let now = Utc::now();
        let scheduled_for = (now - Duration::minutes(1)).to_rfc3339();
        insert_job(&conn, "task-1", &scheduled_for);

        let claim = AutomationWindowClaimDao::try_claim(
            &mut conn,
            "task-1",
            &scheduled_for,
            &now.to_rfc3339(),
            &now.to_rfc3339(),
        )
        .expect("claim scheduled window")
        .expect("first claim owns window");
        let duplicate = AutomationWindowClaimDao::try_claim(
            &mut conn,
            "task-1",
            &scheduled_for,
            &now.to_rfc3339(),
            &(now + Duration::seconds(1)).to_rfc3339(),
        )
        .expect("retry scheduled window claim");

        assert!(duplicate.is_none());
        assert_eq!(claim.claim_key, format!("task-1:{scheduled_for}"));
        assert!(claim.run_id.starts_with("scheduled-run-"));
        assert!(AutomationWindowClaimDao::is_current(&conn, &claim).expect("read claim ownership"));
        let run = AgentRunDao::get_run(&conn, &claim.run_id)
            .expect("read claimed run")
            .expect("claimed run exists");
        assert_eq!(run.status, AgentRunStatus::Queued);

        assert!(AutomationWindowClaimDao::mark_started(
            &mut conn,
            &claim,
            &(now + Duration::seconds(1)).to_rfc3339(),
            "session-1",
            "{}",
        )
        .expect("start claimed run"));
        assert!(!AutomationWindowClaimDao::mark_started(
            &mut conn,
            &claim,
            &(now + Duration::seconds(2)).to_rfc3339(),
            "session-2",
            "{}",
        )
        .expect("reject duplicate claimed run start"));
        let run = AgentRunDao::get_run(&conn, &claim.run_id)
            .expect("read started run")
            .expect("started run exists");
        assert_eq!(run.status, AgentRunStatus::Running);
        assert_eq!(run.session_id.as_deref(), Some("session-1"));
    }

    #[test]
    fn one_shot_claim_with_null_next_run_keeps_ownership_until_start() {
        let mut conn = Connection::open_in_memory().expect("open scheduler test db");
        create_tables(&conn).expect("create scheduler test schema");
        let now = Utc::now();
        let scheduled_for = (now - Duration::minutes(1)).to_rfc3339();
        insert_job(&conn, "one-shot-task", &scheduled_for);

        let claim = AutomationWindowClaimDao::try_claim_reconciled(
            &mut conn,
            "one-shot-task",
            &scheduled_for,
            &scheduled_for,
            None,
            &now.to_rfc3339(),
            &now.to_rfc3339(),
            false,
            0,
        )
        .expect("claim one-shot window")
        .expect("one-shot claim exists");

        assert!(claim.next_run_at.is_none());
        assert!(claim.job.next_run_at.is_none());
        assert!(AutomationWindowClaimDao::is_current(&conn, &claim)
            .expect("read one-shot claim ownership"));
        assert!(AutomationWindowClaimDao::mark_started(
            &mut conn,
            &claim,
            &(now + Duration::seconds(1)).to_rfc3339(),
            "session-one-shot",
            "{}",
        )
        .expect("start one-shot claim"));
    }

    #[test]
    fn overlap_missed_advances_window_without_releasing_running_owner() {
        let mut conn = Connection::open_in_memory().expect("open scheduler test db");
        create_tables(&conn).expect("create scheduler test schema");
        let now = Utc::now();
        let first_window = (now - Duration::minutes(1)).to_rfc3339();
        let overlap_window = (now + Duration::minutes(59)).to_rfc3339();
        let following_window = (now + Duration::minutes(119)).to_rfc3339();
        insert_job(&conn, "overlap-task", &first_window);
        let claim = AutomationWindowClaimDao::try_claim_reconciled(
            &mut conn,
            "overlap-task",
            &first_window,
            &first_window,
            Some(&overlap_window),
            &now.to_rfc3339(),
            &now.to_rfc3339(),
            false,
            0,
        )
        .expect("claim first window")
        .expect("first claim exists");
        assert!(AutomationWindowClaimDao::mark_started(
            &mut conn,
            &claim,
            &(now + Duration::seconds(1)).to_rfc3339(),
            "session-overlap",
            "{}",
        )
        .expect("start first window"));

        assert!(AutomationWindowClaimDao::record_overlap_missed_and_advance(
            &mut conn,
            "overlap-task",
            &claim.task_revision,
            &overlap_window,
            &overlap_window,
            Some(&following_window),
            &(now + Duration::hours(1)).to_rfc3339(),
            0,
            &claim.claimed_at,
        )
        .expect("record overlap window"));

        let job = AutomationJobDao::get(&conn, "overlap-task")
            .expect("read overlap task")
            .expect("overlap task exists");
        assert_eq!(job.next_run_at.as_deref(), Some(following_window.as_str()));
        assert_eq!(job.last_status.as_deref(), Some("running"));
        assert_eq!(
            job.running_started_at.as_deref(),
            Some(claim.claimed_at.as_str())
        );
        assert!(AutomationWindowClaimDao::is_current(&conn, &claim)
            .expect("running claim remains current"));
        let runs = AgentRunDao::list_runs_by_source_ref(&conn, "automation", "overlap-task", 10)
            .expect("read overlap run history");
        assert_eq!(runs.len(), 2);
        assert!(runs.iter().any(|run| {
            run.status == AgentRunStatus::Missed
                && run.error_code.as_deref() == Some("scheduled_run_overlap")
        }));
    }

    #[test]
    fn pause_or_revision_change_invalidates_claim_start() {
        let mut conn = Connection::open_in_memory().expect("open scheduler test db");
        create_tables(&conn).expect("create scheduler test schema");
        let now = Utc::now();
        let scheduled_for = (now - Duration::minutes(1)).to_rfc3339();
        insert_job(&conn, "task-1", &scheduled_for);
        let claim = AutomationWindowClaimDao::try_claim(
            &mut conn,
            "task-1",
            &scheduled_for,
            &now.to_rfc3339(),
            &now.to_rfc3339(),
        )
        .expect("claim scheduled window")
        .expect("claim exists");

        conn.execute(
            "UPDATE automation_jobs
             SET enabled = 0, updated_at = ?1
             WHERE id = ?2",
            params![(now + Duration::seconds(1)).to_rfc3339(), "task-1"],
        )
        .expect("pause task after claim");

        assert!(
            !AutomationWindowClaimDao::is_current(&conn, &claim).expect("read invalidated claim")
        );
        assert!(AutomationWindowClaimDao::invalidate(
            &mut conn,
            &claim,
            &(now + Duration::seconds(2)).to_rfc3339(),
            "task changed after claim",
        )
        .expect("invalidate claim"));
        let job = AutomationJobDao::get(&conn, "task-1")
            .expect("read task")
            .expect("task exists");
        assert!(!job.enabled);
        assert!(job.running_started_at.is_none());
        let run = AgentRunDao::get_run(&conn, &claim.run_id)
            .expect("read invalidated run")
            .expect("invalidated run exists");
        assert_eq!(run.status, AgentRunStatus::Canceled);
        assert_eq!(
            run.error_code.as_deref(),
            Some("scheduled_claim_invalidated")
        );
    }

    #[test]
    fn future_or_running_window_cannot_be_claimed() {
        let mut conn = Connection::open_in_memory().expect("open scheduler test db");
        create_tables(&conn).expect("create scheduler test schema");
        let now = Utc::now();
        let future = (now + Duration::hours(1)).to_rfc3339();
        insert_job(&conn, "task-1", &future);

        let claim = AutomationWindowClaimDao::try_claim(
            &mut conn,
            "task-1",
            &future,
            &now.to_rfc3339(),
            &now.to_rfc3339(),
        )
        .expect("attempt future claim");
        assert!(claim.is_none());
        assert!(
            AutomationWindowClaimDao::list_due_candidates(&conn, &now.to_rfc3339())
                .expect("list due candidates")
                .is_empty()
        );
    }

    #[test]
    fn deleting_task_after_claim_still_cancels_queued_run() {
        let mut conn = Connection::open_in_memory().expect("open scheduler test db");
        create_tables(&conn).expect("create scheduler test schema");
        let now = Utc::now();
        let scheduled_for = (now - Duration::minutes(1)).to_rfc3339();
        insert_job(&conn, "task-1", &scheduled_for);
        let claim = AutomationWindowClaimDao::try_claim(
            &mut conn,
            "task-1",
            &scheduled_for,
            &now.to_rfc3339(),
            &now.to_rfc3339(),
        )
        .expect("claim scheduled window")
        .expect("claim exists");

        AutomationJobDao::delete(&conn, "task-1").expect("delete task after claim");

        assert!(AutomationWindowClaimDao::invalidate(
            &mut conn,
            &claim,
            &(now + Duration::seconds(1)).to_rfc3339(),
            "task deleted after claim",
        )
        .expect("invalidate deleted task claim"));
        let run = AgentRunDao::get_run(&conn, &claim.run_id)
            .expect("read invalidated run")
            .expect("invalidated run exists");
        assert_eq!(run.status, AgentRunStatus::Canceled);
    }

    #[test]
    fn concurrent_connections_cannot_claim_the_same_window() {
        let temp = TempDir::new().expect("create scheduler test directory");
        let db_path = temp.path().join("scheduler.sqlite");
        let first = Connection::open(&db_path).expect("open first scheduler connection");
        create_tables(&first).expect("create scheduler test schema");
        let now = Utc::now();
        let scheduled_for = (now - Duration::minutes(1)).to_rfc3339();
        insert_job(&first, "task-1", &scheduled_for);
        drop(first);

        let mut first = Connection::open(&db_path).expect("reopen first scheduler connection");
        let mut second = Connection::open(&db_path).expect("open second scheduler connection");
        let first_claim = AutomationWindowClaimDao::try_claim(
            &mut first,
            "task-1",
            &scheduled_for,
            &now.to_rfc3339(),
            &now.to_rfc3339(),
        )
        .expect("first connection claim");
        let second_claim = AutomationWindowClaimDao::try_claim(
            &mut second,
            "task-1",
            &scheduled_for,
            &now.to_rfc3339(),
            &(now + Duration::seconds(1)).to_rfc3339(),
        )
        .expect("second connection claim");

        assert!(first_claim.is_some());
        assert!(second_claim.is_none());
    }
}
