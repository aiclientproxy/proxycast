//! 自动化任务数据访问对象
//!
//! 负责结构化自动化任务的持久化与查询。

use crate::config::{AutomationExecutionMode, DeliveryConfig, TaskSchedule};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationJobLastDelivery {
    pub success: bool,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    pub output_kind: String,
    pub output_schema: String,
    pub output_format: String,
    pub output_preview: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delivery_attempt_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    #[serde(default)]
    pub execution_retry_count: u32,
    #[serde(default)]
    pub delivery_attempts: u32,
    pub attempted_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationJob {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub enabled: bool,
    pub workspace_id: String,
    pub execution_mode: AutomationExecutionMode,
    pub schedule: TaskSchedule,
    pub payload: Value,
    pub delivery: DeliveryConfig,
    pub timeout_secs: Option<u64>,
    pub max_retries: u32,
    pub next_run_at: Option<String>,
    pub last_status: Option<String>,
    pub last_error: Option<String>,
    pub last_run_at: Option<String>,
    pub last_finished_at: Option<String>,
    pub running_started_at: Option<String>,
    pub consecutive_failures: u32,
    pub last_retry_count: u32,
    pub auto_disabled_until: Option<String>,
    pub deleted_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_delivery: Option<AutomationJobLastDelivery>,
    pub created_at: String,
    pub updated_at: String,
}

pub struct AutomationJobDao;

impl AutomationJobDao {
    pub fn create(conn: &Connection, job: &AutomationJob) -> Result<(), rusqlite::Error> {
        let schedule_json = serde_json::to_string(&job.schedule)
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
        let payload_json = serde_json::to_string(&job.payload)
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
        let delivery_json = serde_json::to_string(&job.delivery)
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
        let last_delivery_json = job
            .last_delivery
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

        conn.execute(
            "INSERT INTO automation_jobs (
                id, name, description, enabled, workspace_id, execution_mode,
                schedule_json, payload_json, delivery_json, timeout_secs, max_retries,
                next_run_at, last_status, last_error, last_run_at, last_finished_at,
                running_started_at, consecutive_failures, last_retry_count,
                auto_disabled_until, created_at, updated_at, last_delivery_json
                , deleted_at
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6,
                ?7, ?8, ?9, ?10, ?11,
                ?12, ?13, ?14, ?15, ?16,
                ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24
            )",
            params![
                job.id,
                job.name,
                job.description,
                if job.enabled { 1 } else { 0 },
                job.workspace_id,
                serde_json::to_string(&job.execution_mode)
                    .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?,
                schedule_json,
                payload_json,
                delivery_json,
                job.timeout_secs.map(|v| v as i64),
                job.max_retries as i64,
                job.next_run_at,
                job.last_status,
                job.last_error,
                job.last_run_at,
                job.last_finished_at,
                job.running_started_at,
                job.consecutive_failures as i64,
                job.last_retry_count as i64,
                job.auto_disabled_until,
                job.created_at,
                job.updated_at,
                last_delivery_json,
                job.deleted_at,
            ],
        )?;

        Ok(())
    }

    pub fn get(conn: &Connection, id: &str) -> Result<Option<AutomationJob>, rusqlite::Error> {
        let mut stmt = conn.prepare(
            "SELECT
                id, name, description, enabled, workspace_id, execution_mode,
                schedule_json, payload_json, delivery_json, timeout_secs, max_retries,
                next_run_at, last_status, last_error, last_run_at, last_finished_at,
                running_started_at, consecutive_failures, last_retry_count,
                auto_disabled_until, created_at, updated_at, last_delivery_json,
                deleted_at
             FROM automation_jobs
             WHERE id = ?1 AND deleted_at IS NULL",
        )?;

        let mut rows = stmt.query(params![id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(Self::row_to_job(row)?))
        } else {
            Ok(None)
        }
    }

    pub fn get_including_deleted(
        conn: &Connection,
        id: &str,
    ) -> Result<Option<AutomationJob>, rusqlite::Error> {
        let mut stmt = conn.prepare(
            "SELECT
                id, name, description, enabled, workspace_id, execution_mode,
                schedule_json, payload_json, delivery_json, timeout_secs, max_retries,
                next_run_at, last_status, last_error, last_run_at, last_finished_at,
                running_started_at, consecutive_failures, last_retry_count,
                auto_disabled_until, created_at, updated_at, last_delivery_json,
                deleted_at
             FROM automation_jobs
             WHERE id = ?1",
        )?;
        let mut rows = stmt.query(params![id])?;
        rows.next()?.map(Self::row_to_job).transpose()
    }

    pub fn list(conn: &Connection) -> Result<Vec<AutomationJob>, rusqlite::Error> {
        let mut stmt = conn.prepare(
            "SELECT
                id, name, description, enabled, workspace_id, execution_mode,
                schedule_json, payload_json, delivery_json, timeout_secs, max_retries,
                next_run_at, last_status, last_error, last_run_at, last_finished_at,
                running_started_at, consecutive_failures, last_retry_count,
                auto_disabled_until, created_at, updated_at, last_delivery_json
                , deleted_at
             FROM automation_jobs
             WHERE deleted_at IS NULL
             ORDER BY updated_at DESC, created_at DESC",
        )?;

        let rows = stmt.query_map([], Self::row_to_job)?;
        rows.collect()
    }

    pub fn list_due(
        conn: &Connection,
        now_rfc3339: &str,
        limit: usize,
    ) -> Result<Vec<AutomationJob>, rusqlite::Error> {
        let mut stmt = conn.prepare(
            "SELECT
                id, name, description, enabled, workspace_id, execution_mode,
                schedule_json, payload_json, delivery_json, timeout_secs, max_retries,
                next_run_at, last_status, last_error, last_run_at, last_finished_at,
                running_started_at, consecutive_failures, last_retry_count,
                auto_disabled_until, created_at, updated_at, last_delivery_json
                , deleted_at
             FROM automation_jobs
             WHERE enabled = 1
               AND deleted_at IS NULL
               AND next_run_at IS NOT NULL
               AND datetime(next_run_at) <= datetime(?1)
               AND running_started_at IS NULL
               AND (
                    auto_disabled_until IS NULL
                    OR datetime(auto_disabled_until) <= datetime(?1)
               )
             ORDER BY datetime(next_run_at) ASC
             LIMIT ?2",
        )?;

        let rows = stmt.query_map(params![now_rfc3339, limit as i64], Self::row_to_job)?;
        rows.collect()
    }

    pub fn update(conn: &Connection, job: &AutomationJob) -> Result<(), rusqlite::Error> {
        let schedule_json = serde_json::to_string(&job.schedule)
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
        let payload_json = serde_json::to_string(&job.payload)
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
        let delivery_json = serde_json::to_string(&job.delivery)
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
        let last_delivery_json = job
            .last_delivery
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

        conn.execute(
            "UPDATE automation_jobs
             SET name = ?1,
                 description = ?2,
                 enabled = ?3,
                 workspace_id = ?4,
                 execution_mode = ?5,
                 schedule_json = ?6,
                 payload_json = ?7,
                 delivery_json = ?8,
                 timeout_secs = ?9,
                 max_retries = ?10,
                 next_run_at = ?11,
                 last_status = ?12,
                 last_error = ?13,
                 last_run_at = ?14,
                 last_finished_at = ?15,
                 running_started_at = ?16,
                 consecutive_failures = ?17,
                 last_retry_count = ?18,
                 auto_disabled_until = ?19,
                 updated_at = ?20,
                 last_delivery_json = ?21
                 , deleted_at = ?22
             WHERE id = ?23 AND deleted_at IS NULL",
            params![
                job.name,
                job.description,
                if job.enabled { 1 } else { 0 },
                job.workspace_id,
                serde_json::to_string(&job.execution_mode)
                    .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?,
                schedule_json,
                payload_json,
                delivery_json,
                job.timeout_secs.map(|v| v as i64),
                job.max_retries as i64,
                job.next_run_at,
                job.last_status,
                job.last_error,
                job.last_run_at,
                job.last_finished_at,
                job.running_started_at,
                job.consecutive_failures as i64,
                job.last_retry_count as i64,
                job.auto_disabled_until,
                job.updated_at,
                last_delivery_json,
                job.deleted_at,
                job.id,
            ],
        )?;

        Ok(())
    }

    pub fn delete(conn: &Connection, id: &str) -> Result<bool, rusqlite::Error> {
        let rows = conn.execute("DELETE FROM automation_jobs WHERE id = ?1", params![id])?;
        Ok(rows > 0)
    }

    pub fn soft_delete(
        conn: &Connection,
        id: &str,
        deleted_at: &str,
    ) -> Result<bool, rusqlite::Error> {
        let rows = conn.execute(
            "UPDATE automation_jobs
             SET enabled = 0, next_run_at = NULL, deleted_at = ?2, updated_at = ?2
             WHERE id = ?1 AND deleted_at IS NULL",
            params![id, deleted_at],
        )?;
        Ok(rows > 0)
    }

    fn row_to_job(row: &rusqlite::Row<'_>) -> Result<AutomationJob, rusqlite::Error> {
        let execution_mode_raw: String = row.get(5)?;
        let schedule_json: String = row.get(6)?;
        let payload_json: String = row.get(7)?;
        let delivery_json: String = row.get(8)?;
        let last_delivery_json: Option<String> = row.get(22)?;
        let deleted_at: Option<String> = row.get(23)?;

        let execution_mode = serde_json::from_str(&execution_mode_raw).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(5, rusqlite::types::Type::Text, Box::new(e))
        })?;
        let schedule = serde_json::from_str(&schedule_json).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(6, rusqlite::types::Type::Text, Box::new(e))
        })?;
        let payload = serde_json::from_str(&payload_json).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(7, rusqlite::types::Type::Text, Box::new(e))
        })?;
        let delivery = serde_json::from_str(&delivery_json).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(8, rusqlite::types::Type::Text, Box::new(e))
        })?;
        let last_delivery = last_delivery_json
            .map(|value| {
                serde_json::from_str::<AutomationJobLastDelivery>(&value).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        22,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    )
                })
            })
            .transpose()?;

        Ok(AutomationJob {
            id: row.get(0)?,
            name: row.get(1)?,
            description: row.get(2)?,
            enabled: row.get::<_, i64>(3)? != 0,
            workspace_id: row.get(4)?,
            execution_mode,
            schedule,
            payload,
            delivery,
            timeout_secs: row.get::<_, Option<i64>>(9)?.map(|v| v as u64),
            max_retries: row.get::<_, i64>(10)? as u32,
            next_run_at: row.get(11)?,
            last_status: row.get(12)?,
            last_error: row.get(13)?,
            last_run_at: row.get(14)?,
            last_finished_at: row.get(15)?,
            running_started_at: row.get(16)?,
            consecutive_failures: row.get::<_, i64>(17)? as u32,
            last_retry_count: row.get::<_, i64>(18)? as u32,
            auto_disabled_until: row.get(19)?,
            deleted_at,
            last_delivery,
            created_at: row.get(20)?,
            updated_at: row.get(21)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::schema::create_tables;
    use rusqlite::Connection;
    use serde_json::json;

    #[test]
    fn create_and_get_should_preserve_last_delivery() {
        let conn = Connection::open_in_memory().expect("创建内存数据库失败");
        create_tables(&conn).expect("创建数据表失败");

        let job = AutomationJob {
            id: "job-1".to_string(),
            name: "巡检任务".to_string(),
            description: Some("测试最近一次投递结果".to_string()),
            enabled: true,
            workspace_id: "workspace-1".to_string(),
            execution_mode: AutomationExecutionMode::Intelligent,
            schedule: TaskSchedule::Every { every_secs: 300 },
            payload: json!({
                "kind": "agent_turn",
                "prompt": "汇总今日异常",
                "web_search": false
            }),
            delivery: DeliveryConfig::default(),
            timeout_secs: Some(120),
            max_retries: 2,
            next_run_at: Some("2026-03-16T00:10:00Z".to_string()),
            last_status: Some("error".to_string()),
            last_error: Some("写入本地文件失败".to_string()),
            last_run_at: Some("2026-03-16T00:00:00Z".to_string()),
            last_finished_at: Some("2026-03-16T00:00:05Z".to_string()),
            running_started_at: None,
            consecutive_failures: 1,
            last_retry_count: 0,
            auto_disabled_until: None,
            deleted_at: None,
            last_delivery: Some(AutomationJobLastDelivery {
                success: false,
                message: "写入本地文件失败: permission denied".to_string(),
                channel: Some("local_file".to_string()),
                target: Some("/tmp/lime/output.json".to_string()),
                output_kind: "json".to_string(),
                output_schema: "json".to_string(),
                output_format: "json".to_string(),
                output_preview: "{\"status\":\"error\"}".to_string(),
                delivery_attempt_id: Some("dlv-run-1".to_string()),
                run_id: Some("run-1".to_string()),
                execution_retry_count: 1,
                delivery_attempts: 2,
                attempted_at: "2026-03-16T00:00:04Z".to_string(),
            }),
            created_at: "2026-03-16T00:00:00Z".to_string(),
            updated_at: "2026-03-16T00:00:05Z".to_string(),
        };

        AutomationJobDao::create(&conn, &job).expect("创建自动化任务失败");
        let loaded = AutomationJobDao::get(&conn, "job-1")
            .expect("读取自动化任务失败")
            .expect("自动化任务不存在");

        assert_eq!(
            loaded
                .last_delivery
                .as_ref()
                .and_then(|value| value.channel.as_deref()),
            Some("local_file")
        );
        assert_eq!(
            loaded.last_delivery.as_ref().map(|value| value.success),
            Some(false)
        );
        assert_eq!(
            loaded
                .last_delivery
                .as_ref()
                .map(|value| value.output_preview.as_str()),
            Some("{\"status\":\"error\"}")
        );
        assert_eq!(
            loaded
                .last_delivery
                .as_ref()
                .and_then(|value| value.delivery_attempt_id.as_deref()),
            Some("dlv-run-1")
        );
        assert_eq!(
            loaded
                .last_delivery
                .as_ref()
                .map(|value| value.delivery_attempts),
            Some(2)
        );
    }

    #[test]
    fn soft_delete_hides_job_but_preserves_tombstone() {
        let conn = Connection::open_in_memory().expect("创建内存数据库失败");
        create_tables(&conn).expect("创建数据表失败");
        let mut job = sample_job("job-soft-delete");
        AutomationJobDao::create(&conn, &job).expect("创建自动化任务失败");

        let deleted_at = "2026-03-16T01:00:00Z";
        assert!(AutomationJobDao::soft_delete(&conn, &job.id, deleted_at)
            .expect("软删除自动化任务失败"));
        assert!(!AutomationJobDao::soft_delete(&conn, &job.id, deleted_at)
            .expect("重复软删除自动化任务失败"));
        assert!(AutomationJobDao::get(&conn, &job.id)
            .expect("读取自动化任务失败")
            .is_none());
        assert!(AutomationJobDao::list(&conn)
            .expect("列出自动化任务失败")
            .is_empty());
        assert!(AutomationJobDao::list_due(&conn, deleted_at, 10)
            .expect("列出到期自动化任务失败")
            .is_empty());

        let deleted = AutomationJobDao::get_including_deleted(&conn, &job.id)
            .expect("读取软删除自动化任务失败")
            .expect("软删除自动化任务不存在");
        assert!(!deleted.enabled);
        assert!(deleted.next_run_at.is_none());
        assert_eq!(deleted.deleted_at.as_deref(), Some(deleted_at));

        job.name = "不应复活".to_string();
        job.updated_at = "2026-03-16T02:00:00Z".to_string();
        AutomationJobDao::update(&conn, &job).expect("更新软删除任务应安全忽略");
        let still_deleted = AutomationJobDao::get_including_deleted(&conn, &job.id)
            .expect("重新读取软删除自动化任务失败")
            .expect("软删除自动化任务不存在");
        assert_eq!(still_deleted.name, "巡检任务");
        assert_eq!(still_deleted.deleted_at.as_deref(), Some(deleted_at));
    }

    fn sample_job(id: &str) -> AutomationJob {
        AutomationJob {
            id: id.to_string(),
            name: "巡检任务".to_string(),
            description: None,
            enabled: true,
            workspace_id: "workspace-1".to_string(),
            execution_mode: AutomationExecutionMode::Intelligent,
            schedule: TaskSchedule::Every { every_secs: 300 },
            payload: json!({"kind": "agent_turn"}),
            delivery: DeliveryConfig::default(),
            timeout_secs: None,
            max_retries: 2,
            next_run_at: Some("2026-03-16T00:10:00Z".to_string()),
            last_status: None,
            last_error: None,
            last_run_at: None,
            last_finished_at: None,
            running_started_at: None,
            consecutive_failures: 0,
            last_retry_count: 0,
            auto_disabled_until: None,
            deleted_at: None,
            last_delivery: None,
            created_at: "2026-03-16T00:00:00Z".to_string(),
            updated_at: "2026-03-16T00:00:00Z".to_string(),
        }
    }
}
