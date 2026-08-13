//! 统一执行追踪（agent_runs）数据访问对象
//!
//! 提供跨 chat / skill / automation 的执行摘要记录能力。

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AgentRunStatus {
    Queued,
    Running,
    Success,
    Error,
    Canceled,
    Timeout,
    Missed,
}

impl AgentRunStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Success => "success",
            Self::Error => "error",
            Self::Canceled => "canceled",
            Self::Timeout => "timeout",
            Self::Missed => "missed",
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Success | Self::Error | Self::Canceled | Self::Timeout | Self::Missed
        )
    }
}

impl TryFrom<&str> for AgentRunStatus {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, String> {
        match value {
            "queued" => Ok(Self::Queued),
            "running" => Ok(Self::Running),
            "success" => Ok(Self::Success),
            "error" => Ok(Self::Error),
            "canceled" => Ok(Self::Canceled),
            "timeout" => Ok(Self::Timeout),
            "missed" => Ok(Self::Missed),
            other => Err(format!("未知执行状态: {other}")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRun {
    pub id: String,
    pub source: String,
    pub source_ref: Option<String>,
    pub session_id: Option<String>,
    pub status: AgentRunStatus,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub duration_ms: Option<i64>,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
    pub metadata: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

pub struct AgentRunDao;

impl AgentRunDao {
    pub fn create_run(conn: &Connection, run: &AgentRun) -> Result<(), rusqlite::Error> {
        conn.execute(
            "INSERT INTO agent_runs (
                id, source, source_ref, session_id, status, started_at, finished_at, duration_ms,
                error_code, error_message, metadata, created_at, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![
                run.id,
                run.source,
                run.source_ref,
                run.session_id,
                run.status.as_str(),
                run.started_at,
                run.finished_at,
                run.duration_ms,
                run.error_code,
                run.error_message,
                run.metadata,
                run.created_at,
                run.updated_at,
            ],
        )?;
        Ok(())
    }

    /// 仅允许从非终态更新到终态，幂等保护：`finished_at IS NULL`
    #[allow(clippy::too_many_arguments)]
    pub fn finish_run(
        conn: &Connection,
        id: &str,
        status: AgentRunStatus,
        finished_at: &str,
        duration_ms: Option<i64>,
        error_code: Option<&str>,
        error_message: Option<&str>,
        metadata: Option<&str>,
    ) -> Result<bool, rusqlite::Error> {
        if !status.is_terminal() {
            return Ok(false);
        }

        let changed = conn.execute(
            "UPDATE agent_runs
             SET status = ?1,
                 finished_at = ?2,
                 duration_ms = ?3,
                 error_code = ?4,
                 error_message = ?5,
                 metadata = COALESCE(?6, metadata),
                 updated_at = ?2
             WHERE id = ?7
               AND finished_at IS NULL",
            params![
                status.as_str(),
                finished_at,
                duration_ms,
                error_code,
                error_message,
                metadata,
                id,
            ],
        )?;
        Ok(changed > 0)
    }

    pub fn refresh_running_run(
        conn: &Connection,
        id: &str,
        updated_at: &str,
        session_id: Option<&str>,
        metadata: Option<&str>,
    ) -> Result<bool, rusqlite::Error> {
        let changed = conn.execute(
            "UPDATE agent_runs
             SET session_id = COALESCE(?1, session_id),
                 metadata = COALESCE(?2, metadata),
                 updated_at = ?3
             WHERE id = ?4
               AND finished_at IS NULL
               AND status IN ('queued', 'running')",
            params![session_id, metadata, updated_at, id],
        )?;
        Ok(changed > 0)
    }

    pub fn get_run(conn: &Connection, id: &str) -> Result<Option<AgentRun>, rusqlite::Error> {
        let mut stmt = conn.prepare(
            "SELECT id, source, source_ref, session_id, status, started_at, finished_at, duration_ms,
                    error_code, error_message, metadata, created_at, updated_at
             FROM agent_runs
             WHERE id = ?1",
        )?;

        let mut rows = stmt.query(params![id])?;
        if let Some(row) = rows.next()? {
            let status_raw: String = row.get(4)?;
            let status = AgentRunStatus::try_from(status_raw.as_str()).map_err(|_| {
                rusqlite::Error::InvalidColumnType(4, "status".into(), rusqlite::types::Type::Text)
            })?;
            Ok(Some(AgentRun {
                id: row.get(0)?,
                source: row.get(1)?,
                source_ref: row.get(2)?,
                session_id: row.get(3)?,
                status,
                started_at: row.get(5)?,
                finished_at: row.get(6)?,
                duration_ms: row.get(7)?,
                error_code: row.get(8)?,
                error_message: row.get(9)?,
                metadata: row.get(10)?,
                created_at: row.get(11)?,
                updated_at: row.get(12)?,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn list_runs(
        conn: &Connection,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<AgentRun>, rusqlite::Error> {
        let mut stmt = conn.prepare(
            "SELECT id, source, source_ref, session_id, status, started_at, finished_at, duration_ms,
                    error_code, error_message, metadata, created_at, updated_at
             FROM agent_runs
             ORDER BY started_at DESC
             LIMIT ?1 OFFSET ?2",
        )?;

        let iter = stmt.query_map(params![limit as i64, offset as i64], |row| {
            let status_raw: String = row.get(4)?;
            let status =
                AgentRunStatus::try_from(status_raw.as_str()).unwrap_or(AgentRunStatus::Error);
            Ok(AgentRun {
                id: row.get(0)?,
                source: row.get(1)?,
                source_ref: row.get(2)?,
                session_id: row.get(3)?,
                status,
                started_at: row.get(5)?,
                finished_at: row.get(6)?,
                duration_ms: row.get(7)?,
                error_code: row.get(8)?,
                error_message: row.get(9)?,
                metadata: row.get(10)?,
                created_at: row.get(11)?,
                updated_at: row.get(12)?,
            })
        })?;

        iter.collect()
    }

    pub fn list_runs_by_session(
        conn: &Connection,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<AgentRun>, rusqlite::Error> {
        let mut stmt = conn.prepare(
            "SELECT id, source, source_ref, session_id, status, started_at, finished_at, duration_ms,
                    error_code, error_message, metadata, created_at, updated_at
             FROM agent_runs
             WHERE session_id = ?1
             ORDER BY started_at DESC
             LIMIT ?2",
        )?;

        let iter = stmt.query_map(params![session_id, limit as i64], |row| {
            let status_raw: String = row.get(4)?;
            let status =
                AgentRunStatus::try_from(status_raw.as_str()).unwrap_or(AgentRunStatus::Error);
            Ok(AgentRun {
                id: row.get(0)?,
                source: row.get(1)?,
                source_ref: row.get(2)?,
                session_id: row.get(3)?,
                status,
                started_at: row.get(5)?,
                finished_at: row.get(6)?,
                duration_ms: row.get(7)?,
                error_code: row.get(8)?,
                error_message: row.get(9)?,
                metadata: row.get(10)?,
                created_at: row.get(11)?,
                updated_at: row.get(12)?,
            })
        })?;

        iter.collect()
    }

    pub fn list_runs_by_source_ref(
        conn: &Connection,
        source: &str,
        source_ref: &str,
        limit: usize,
    ) -> Result<Vec<AgentRun>, rusqlite::Error> {
        let mut stmt = conn.prepare(
            "SELECT id, source, source_ref, session_id, status, started_at, finished_at, duration_ms,
                    error_code, error_message, metadata, created_at, updated_at
             FROM agent_runs
             WHERE source = ?1
               AND source_ref = ?2
             ORDER BY started_at DESC
             LIMIT ?3",
        )?;

        let iter = stmt.query_map(params![source, source_ref, limit as i64], |row| {
            let status_raw: String = row.get(4)?;
            let status =
                AgentRunStatus::try_from(status_raw.as_str()).unwrap_or(AgentRunStatus::Error);
            Ok(AgentRun {
                id: row.get(0)?,
                source: row.get(1)?,
                source_ref: row.get(2)?,
                session_id: row.get(3)?,
                status,
                started_at: row.get(5)?,
                finished_at: row.get(6)?,
                duration_ms: row.get(7)?,
                error_code: row.get(8)?,
                error_message: row.get(9)?,
                metadata: row.get(10)?,
                created_at: row.get(11)?,
                updated_at: row.get(12)?,
            })
        })?;

        iter.collect()
    }

    pub fn list_active_runs_by_session(
        conn: &Connection,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<AgentRun>, rusqlite::Error> {
        let mut stmt = conn.prepare(
            "SELECT id, source, source_ref, session_id, status, started_at, finished_at, duration_ms,
                    error_code, error_message, metadata, created_at, updated_at
             FROM agent_runs
             WHERE session_id = ?1
               AND status IN ('queued', 'running')
             ORDER BY started_at DESC
             LIMIT ?2",
        )?;

        let iter = stmt.query_map(params![session_id, limit as i64], |row| {
            let status_raw: String = row.get(4)?;
            let status =
                AgentRunStatus::try_from(status_raw.as_str()).unwrap_or(AgentRunStatus::Error);
            Ok(AgentRun {
                id: row.get(0)?,
                source: row.get(1)?,
                source_ref: row.get(2)?,
                session_id: row.get(3)?,
                status,
                started_at: row.get(5)?,
                finished_at: row.get(6)?,
                duration_ms: row.get(7)?,
                error_code: row.get(8)?,
                error_message: row.get(9)?,
                metadata: row.get(10)?,
                created_at: row.get(11)?,
                updated_at: row.get(12)?,
            })
        })?;

        iter.collect()
    }

    pub fn list_terminal_runs_by_session(
        conn: &Connection,
        session_id: &str,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<AgentRun>, rusqlite::Error> {
        let mut stmt = conn.prepare(
            "SELECT id, source, source_ref, session_id, status, started_at, finished_at, duration_ms,
                    error_code, error_message, metadata, created_at, updated_at
             FROM agent_runs
             WHERE session_id = ?1
               AND status IN ('success', 'error', 'canceled', 'timeout', 'missed')
             ORDER BY started_at DESC
             LIMIT ?2 OFFSET ?3",
        )?;

        let iter = stmt.query_map(params![session_id, limit as i64, offset as i64], |row| {
            let status_raw: String = row.get(4)?;
            let status =
                AgentRunStatus::try_from(status_raw.as_str()).unwrap_or(AgentRunStatus::Error);
            Ok(AgentRun {
                id: row.get(0)?,
                source: row.get(1)?,
                source_ref: row.get(2)?,
                session_id: row.get(3)?,
                status,
                started_at: row.get(5)?,
                finished_at: row.get(6)?,
                duration_ms: row.get(7)?,
                error_code: row.get(8)?,
                error_message: row.get(9)?,
                metadata: row.get(10)?,
                created_at: row.get(11)?,
                updated_at: row.get(12)?,
            })
        })?;

        iter.collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::schema::create_tables;
    use chrono::Utc;
    use rusqlite::Connection;

    fn setup_conn() -> Connection {
        let conn = Connection::open_in_memory().expect("创建内存数据库失败");
        create_tables(&conn).expect("创建表结构失败");
        conn
    }

    fn sample_run(id: &str, status: AgentRunStatus) -> AgentRun {
        let now = Utc::now().to_rfc3339();
        AgentRun {
            id: id.to_string(),
            source: "chat".to_string(),
            source_ref: Some("sample".to_string()),
            session_id: Some("s1".to_string()),
            status,
            started_at: now.clone(),
            finished_at: None,
            duration_ms: None,
            error_code: None,
            error_message: None,
            metadata: None,
            created_at: now.clone(),
            updated_at: now,
        }
    }

    #[test]
    fn create_and_get_run_should_work() {
        let conn = setup_conn();
        let run = sample_run("run-1", AgentRunStatus::Running);
        AgentRunDao::create_run(&conn, &run).expect("写入 run 失败");

        let fetched = AgentRunDao::get_run(&conn, "run-1")
            .expect("查询 run 失败")
            .expect("run 不存在");
        assert_eq!(fetched.id, "run-1");
        assert_eq!(fetched.status, AgentRunStatus::Running);
    }

    #[test]
    fn finish_run_should_be_idempotent() {
        let conn = setup_conn();
        let run = sample_run("run-2", AgentRunStatus::Running);
        AgentRunDao::create_run(&conn, &run).expect("写入 run 失败");

        let now = Utc::now().to_rfc3339();
        let first = AgentRunDao::finish_run(
            &conn,
            "run-2",
            AgentRunStatus::Success,
            &now,
            Some(100),
            None,
            None,
            None,
        )
        .expect("第一次结束 run 失败");
        let second = AgentRunDao::finish_run(
            &conn,
            "run-2",
            AgentRunStatus::Error,
            &now,
            Some(120),
            Some("err"),
            Some("error"),
            None,
        )
        .expect("第二次结束 run 失败");

        assert!(first);
        assert!(!second);

        let fetched = AgentRunDao::get_run(&conn, "run-2")
            .expect("查询 run 失败")
            .expect("run 不存在");
        assert_eq!(fetched.status, AgentRunStatus::Success);
        assert_eq!(fetched.duration_ms, Some(100));
    }

    #[test]
    fn list_runs_by_session_should_filter_and_sort() {
        let conn = setup_conn();

        let mut run_1 = sample_run("run-a-1", AgentRunStatus::Success);
        run_1.session_id = Some("session-a".to_string());
        run_1.started_at = "2026-03-06T10:00:00Z".to_string();
        run_1.created_at = run_1.started_at.clone();
        run_1.updated_at = run_1.started_at.clone();
        AgentRunDao::create_run(&conn, &run_1).expect("写入 run-a-1 失败");

        let mut run_2 = sample_run("run-b-1", AgentRunStatus::Running);
        run_2.session_id = Some("session-b".to_string());
        run_2.started_at = "2026-03-06T11:00:00Z".to_string();
        run_2.created_at = run_2.started_at.clone();
        run_2.updated_at = run_2.started_at.clone();
        AgentRunDao::create_run(&conn, &run_2).expect("写入 run-b-1 失败");

        let mut run_3 = sample_run("run-a-2", AgentRunStatus::Error);
        run_3.session_id = Some("session-a".to_string());
        run_3.started_at = "2026-03-06T12:00:00Z".to_string();
        run_3.created_at = run_3.started_at.clone();
        run_3.updated_at = run_3.started_at.clone();
        AgentRunDao::create_run(&conn, &run_3).expect("写入 run-a-2 失败");

        let runs = AgentRunDao::list_runs_by_session(&conn, "session-a", 10)
            .expect("按 session 查询执行记录失败");
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].id, "run-a-2");
        assert_eq!(runs[1].id, "run-a-1");
    }

    #[test]
    fn list_terminal_runs_by_session_should_filter_terminal_status_and_offset() {
        let conn = setup_conn();

        let mut run_success = sample_run("run-success", AgentRunStatus::Success);
        run_success.session_id = Some("session-a".to_string());
        run_success.started_at = "2026-03-06T10:00:00Z".to_string();
        run_success.created_at = run_success.started_at.clone();
        run_success.updated_at = run_success.started_at.clone();
        AgentRunDao::create_run(&conn, &run_success).expect("写入 run-success 失败");

        let mut run_running = sample_run("run-running", AgentRunStatus::Running);
        run_running.session_id = Some("session-a".to_string());
        run_running.started_at = "2026-03-06T11:00:00Z".to_string();
        run_running.created_at = run_running.started_at.clone();
        run_running.updated_at = run_running.started_at.clone();
        AgentRunDao::create_run(&conn, &run_running).expect("写入 run-running 失败");

        let mut run_error = sample_run("run-error", AgentRunStatus::Error);
        run_error.session_id = Some("session-a".to_string());
        run_error.started_at = "2026-03-06T12:00:00Z".to_string();
        run_error.created_at = run_error.started_at.clone();
        run_error.updated_at = run_error.started_at.clone();
        AgentRunDao::create_run(&conn, &run_error).expect("写入 run-error 失败");

        let mut run_timeout = sample_run("run-timeout", AgentRunStatus::Timeout);
        run_timeout.session_id = Some("session-a".to_string());
        run_timeout.started_at = "2026-03-06T13:00:00Z".to_string();
        run_timeout.created_at = run_timeout.started_at.clone();
        run_timeout.updated_at = run_timeout.started_at.clone();
        AgentRunDao::create_run(&conn, &run_timeout).expect("写入 run-timeout 失败");

        let mut run_missed = sample_run("run-missed", AgentRunStatus::Missed);
        run_missed.session_id = Some("session-a".to_string());
        run_missed.started_at = "2026-03-06T14:00:00Z".to_string();
        run_missed.created_at = run_missed.started_at.clone();
        run_missed.updated_at = run_missed.started_at.clone();
        AgentRunDao::create_run(&conn, &run_missed).expect("写入 run-missed 失败");

        let first_page = AgentRunDao::list_terminal_runs_by_session(&conn, "session-a", 2, 0)
            .expect("查询第一页终态记录失败");
        assert_eq!(first_page.len(), 2);
        assert_eq!(first_page[0].id, "run-missed");
        assert_eq!(first_page[1].id, "run-timeout");

        let second_page = AgentRunDao::list_terminal_runs_by_session(&conn, "session-a", 2, 2)
            .expect("查询第二页终态记录失败");
        assert_eq!(second_page.len(), 2);
        assert_eq!(second_page[0].id, "run-error");
        assert_eq!(second_page[1].id, "run-success");
    }

    #[test]
    fn list_active_runs_by_session_should_filter_active_status() {
        let conn = setup_conn();

        let mut run_queued = sample_run("run-queued", AgentRunStatus::Queued);
        run_queued.session_id = Some("session-a".to_string());
        run_queued.started_at = "2026-03-06T10:00:00Z".to_string();
        run_queued.created_at = run_queued.started_at.clone();
        run_queued.updated_at = run_queued.started_at.clone();
        AgentRunDao::create_run(&conn, &run_queued).expect("写入 run-queued 失败");

        let mut run_running = sample_run("run-running", AgentRunStatus::Running);
        run_running.session_id = Some("session-a".to_string());
        run_running.started_at = "2026-03-06T11:00:00Z".to_string();
        run_running.created_at = run_running.started_at.clone();
        run_running.updated_at = run_running.started_at.clone();
        AgentRunDao::create_run(&conn, &run_running).expect("写入 run-running 失败");

        let mut run_success = sample_run("run-success", AgentRunStatus::Success);
        run_success.session_id = Some("session-a".to_string());
        run_success.started_at = "2026-03-06T12:00:00Z".to_string();
        run_success.created_at = run_success.started_at.clone();
        run_success.updated_at = run_success.started_at.clone();
        AgentRunDao::create_run(&conn, &run_success).expect("写入 run-success 失败");

        let mut run_other_session = sample_run("run-other", AgentRunStatus::Running);
        run_other_session.session_id = Some("session-b".to_string());
        run_other_session.started_at = "2026-03-06T13:00:00Z".to_string();
        run_other_session.created_at = run_other_session.started_at.clone();
        run_other_session.updated_at = run_other_session.started_at.clone();
        AgentRunDao::create_run(&conn, &run_other_session).expect("写入 run-other 失败");

        let active = AgentRunDao::list_active_runs_by_session(&conn, "session-a", 10)
            .expect("查询活跃 run 失败");
        assert_eq!(active.len(), 2);
        assert_eq!(active[0].id, "run-running");
        assert_eq!(active[1].id, "run-queued");
    }

    #[test]
    fn refresh_running_run_should_update_session_and_metadata() {
        let conn = setup_conn();
        let mut run = sample_run("run-refresh", AgentRunStatus::Running);
        run.session_id = None;
        AgentRunDao::create_run(&conn, &run).expect("写入 run 失败");

        let updated_at = Utc::now().to_rfc3339();
        let changed = AgentRunDao::refresh_running_run(
            &conn,
            "run-refresh",
            &updated_at,
            Some("session-refresh"),
            Some("{\"browser_lifecycle_state\":\"waiting_for_human\"}"),
        )
        .expect("刷新运行中 run 失败");

        assert!(changed);

        let fetched = AgentRunDao::get_run(&conn, "run-refresh")
            .expect("查询 run 失败")
            .expect("run 不存在");
        assert_eq!(fetched.session_id.as_deref(), Some("session-refresh"));
        assert_eq!(
            fetched.metadata.as_deref(),
            Some("{\"browser_lifecycle_state\":\"waiting_for_human\"}")
        );
        assert_eq!(fetched.updated_at, updated_at);
    }
}
