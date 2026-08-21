use agent_protocol::AgentInput;
use app_server_protocol::AgentEvent;
use app_server_protocol::AgentSession;
use app_server_protocol::AgentSessionListParams;
use app_server_protocol::AgentSessionOverview;
use app_server_protocol::AgentTurn;
use rusqlite::params;
use rusqlite::Connection;
use rusqlite::OptionalExtension;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::canonical_rollout::RolloutStore;
use super::projection_item_events::query_projected_session_item_events;
use super::projection_payload_summary::bounded_payload_summary;
use super::projection_protocol::{
    projected_import_reference_from_metadata, projected_import_session_to_protocol,
    projected_session_to_protocol, projected_turn_to_protocol,
};
use super::projection_schema::create_schema;
use super::projection_status::{session_status_from_event, turn_status_from_event};
use super::session_list_scope::SessionListScope;
use super::session_title;
use super::status::resolve_session_runtime_state;
use super::status::RuntimeTurnSnapshot;

mod session_settings;
pub(in crate::runtime) mod thread_delete;
mod thread_product_projection;

const PROJECTION_SUMMARY_MESSAGE_TEXT_MAX_CHARS: usize = 2_000;
const PROJECTION_SUMMARY_MESSAGE_ROW_LIMIT: i64 = 20_000;

#[derive(Debug, Clone)]
pub struct ProjectionStore {
    path: PathBuf,
    state_path: PathBuf,
    thread_history_path: PathBuf,
    rollout_store: Option<RolloutStore>,
    pub(in crate::runtime) goal_accounting:
        Arc<super::canonical_thread_store::goal_idle::GoalAccountingState>,
}

impl PartialEq for ProjectionStore {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
            && self.state_path == other.state_path
            && self.thread_history_path == other.thread_history_path
            && self.rollout_store == other.rollout_store
    }
}

impl Eq for ProjectionStore {}

#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionReadSession {
    pub session: AgentSession,
    pub turns: Vec<AgentTurn>,
    pub item_count: usize,
    pub messages_count: usize,
    pub messages_start_index: usize,
    pub messages: Vec<Value>,
    pub item_events: Vec<AgentEvent>,
    pub last_event_sequence: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProjectionReadWindow {
    pub history_limit: Option<usize>,
    pub history_offset: usize,
    pub history_before_message_id: Option<i64>,
}

impl ProjectionReadWindow {
    pub fn from_read_params(params: &app_server_protocol::AgentSessionReadParams) -> Self {
        Self {
            history_limit: params.history_limit.map(|value| value as usize),
            history_offset: params.history_offset.unwrap_or_default() as usize,
            history_before_message_id: params.history_before_message_id,
        }
    }

    pub fn tail(history_limit: Option<usize>) -> Self {
        Self {
            history_limit,
            ..Self::default()
        }
    }

    fn cursor_before(self) -> Option<i64> {
        self.history_before_message_id.filter(|value| *value > 0)
    }
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectionSession {
    pub session_id: String,
    pub thread_id: String,
    pub status: String,
    pub last_event_sequence: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjectionWatermark {
    pub last_sequence: u64,
}

impl ProjectionStore {
    pub fn initialize(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref().to_path_buf();
        Self::initialize_inner(path.clone(), path.clone(), path, None)
    }

    pub fn initialize_with_agent_root(
        path: impl AsRef<Path>,
        agent_root: impl AsRef<Path>,
    ) -> Result<Self, String> {
        let path = path.as_ref().to_path_buf();
        Self::initialize_inner(
            path.clone(),
            path.clone(),
            path,
            Some(RolloutStore::new(agent_root)),
        )
    }

    pub fn initialize_with_storage_paths(
        projection_path: impl AsRef<Path>,
        state_path: impl AsRef<Path>,
        thread_history_path: impl AsRef<Path>,
        agent_root: impl AsRef<Path>,
    ) -> Result<Self, String> {
        Self::initialize_inner(
            projection_path,
            state_path,
            thread_history_path,
            Some(RolloutStore::new(agent_root)),
        )
    }

    fn initialize_inner(
        projection_path: impl AsRef<Path>,
        state_path: impl AsRef<Path>,
        thread_history_path: impl AsRef<Path>,
        rollout_store: Option<RolloutStore>,
    ) -> Result<Self, String> {
        let path = projection_path.as_ref().to_path_buf();
        let state_path = state_path.as_ref().to_path_buf();
        let thread_history_path = thread_history_path.as_ref().to_path_buf();
        for database_path in [&path, &state_path, &thread_history_path] {
            let Some(parent) = database_path.parent() else {
                continue;
            };
            fs::create_dir_all(parent).map_err(|error| {
                format!(
                    "无法创建 App Server SQLite 目录 {}: {error}",
                    parent.display()
                )
            })?;
        }
        let conn = Connection::open(&path)
            .map_err(|error| format!("无法打开 Projection DB {}: {error}", path.display()))?;
        create_schema(&conn)?;
        let store = Self {
            path,
            state_path,
            thread_history_path,
            rollout_store,
            goal_accounting: Arc::new(Default::default()),
        };
        store.ensure_canonical_thread_store()?;
        store.rebuild_canonical_rollouts_if_empty()?;
        Ok(store)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn state_path(&self) -> &Path {
        &self.state_path
    }

    pub fn thread_history_path(&self) -> &Path {
        &self.thread_history_path
    }

    fn open_projection_store(&self) -> Result<Connection, String> {
        self.open_thread_store().map_err(|error| error.to_string())
    }

    pub(super) fn rollout_store(&self) -> Option<&RolloutStore> {
        self.rollout_store.as_ref()
    }

    pub fn apply_event(&self, event: &AgentEvent) -> Result<(), String> {
        self.apply_events(std::slice::from_ref(event)).map(|_| ())
    }

    pub fn apply_events(&self, events: &[AgentEvent]) -> Result<usize, String> {
        if events.is_empty() {
            return Ok(0);
        }
        let mut conn = self.open_projection_store()?;
        let tx = conn
            .transaction()
            .map_err(|error| format!("无法开始 Projection DB 事务: {error}"))?;
        let mut applied = 0;
        for event in events {
            if apply_event_in_tx(&tx, event)? {
                applied += 1;
            }
        }
        tx.commit()
            .map_err(|error| format!("无法提交 Projection DB 事务: {error}"))?;
        Ok(applied)
    }

    pub fn clear_session(&self, session_id: &str) -> Result<(), String> {
        let mut conn = self.open_projection_store()?;
        let tx = conn
            .transaction()
            .map_err(|error| format!("无法开始 Projection DB 事务: {error}"))?;
        clear_session_in_tx(&tx, session_id)?;
        tx.commit()
            .map_err(|error| format!("无法提交 Projection DB 事务: {error}"))?;
        Ok(())
    }

    pub fn repair_session(&self, session_id: &str, events: &[AgentEvent]) -> Result<usize, String> {
        if let Some(invalid_session_id) = events
            .iter()
            .map(|event| event.session_id.as_str())
            .find(|event_session_id| *event_session_id != session_id)
        {
            return Err(format!(
                "Projection repair session mismatch: expected {session_id}, got {invalid_session_id}"
            ));
        }

        let mut conn = self.open_projection_store()?;
        let tx = conn
            .transaction()
            .map_err(|error| format!("无法开始 Projection DB 事务: {error}"))?;
        clear_session_in_tx(&tx, session_id)?;
        let mut applied = 0;
        for event in events {
            if apply_event_in_tx(&tx, event)? {
                applied += 1;
            }
        }
        tx.commit()
            .map_err(|error| format!("无法提交 Projection DB 事务: {error}"))?;
        Ok(applied)
    }

    pub fn read_session_projection(
        &self,
        session_id: &str,
        window: ProjectionReadWindow,
    ) -> Result<Option<ProjectionReadSession>, String> {
        let conn = self.open_projection_store()?;
        let Some(session_row) = query_projected_session(&conn, session_id)? else {
            return Ok(None);
        };
        let turns = query_projected_turns(&conn, session_id)?;
        let item_count = query_projected_item_count(&conn, session_id)?;
        let messages_count = query_projected_message_count(&conn, session_id)?;
        let (messages, messages_start_index) =
            query_projected_window_messages(&conn, session_id, window, messages_count)?;
        let item_events = query_projected_session_item_events(&conn, session_id, &messages)?;
        let first_user_message = query_projected_first_user_message(&conn, session_id)?;
        Ok(Some(ProjectionReadSession {
            session: projected_session_to_protocol(&session_row, first_user_message),
            turns: turns.into_iter().map(projected_turn_to_protocol).collect(),
            item_count,
            messages_count,
            messages_start_index,
            messages,
            item_events,
            last_event_sequence: session_row.last_event_sequence,
        }))
    }

    pub(in crate::runtime) fn find_session_by_import_source(
        &self,
        source_kind: &str,
        source_client: &str,
        source_thread_id: &str,
    ) -> Result<Option<AgentSession>, String> {
        let normalized_kind = source_kind.trim();
        let normalized_client = source_client.trim();
        let normalized_thread_id = source_thread_id.trim();
        if normalized_kind.is_empty()
            || normalized_client.is_empty()
            || normalized_thread_id.is_empty()
        {
            return Ok(None);
        }

        let conn = self.open_projection_store()?;
        let rows =
            query_projected_import_session_rows(&conn, normalized_client, normalized_thread_id)?;
        let matched = rows
            .into_iter()
            .filter_map(|row| {
                projected_import_reference_from_metadata(
                    &row,
                    normalized_kind,
                    normalized_client,
                    normalized_thread_id,
                )
                .map(|reference| (row, reference))
            })
            .max_by(|(left, _), (right, _)| {
                left.updated_at
                    .cmp(&right.updated_at)
                    .then_with(|| left.session_id.cmp(&right.session_id))
            });
        let Some((row, business_object_ref)) = matched else {
            return Ok(None);
        };
        Ok(Some(projected_import_session_to_protocol(
            row,
            business_object_ref,
        )))
    }

    pub fn read_watermark(&self, session_id: &str) -> Result<Option<ProjectionWatermark>, String> {
        let conn = self.open_projection_store()?;
        conn.query_row(
            "SELECT last_sequence FROM projection_watermarks WHERE session_id = ?1",
            params![session_id],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .map(|value| {
            value.map(|last_sequence| ProjectionWatermark {
                last_sequence: last_sequence.max(0) as u64,
            })
        })
        .map_err(|error| format!("无法读取 Projection DB watermark: {error}"))
    }

    #[cfg(test)]
    pub fn read_session(&self, session_id: &str) -> Result<Option<ProjectionSession>, String> {
        let conn = self.open_projection_store()?;
        let mut stmt = conn
            .prepare(
                "SELECT session_id, thread_id, status, last_event_sequence
                 FROM projected_sessions
                 WHERE session_id = ?1",
            )
            .map_err(|error| format!("无法准备 Projection DB 查询: {error}"))?;
        let mut rows = stmt
            .query(params![session_id])
            .map_err(|error| format!("无法查询 Projection DB: {error}"))?;
        let Some(row) = rows
            .next()
            .map_err(|error| format!("无法读取 Projection DB 行: {error}"))?
        else {
            return Ok(None);
        };
        Ok(Some(ProjectionSession {
            session_id: row.get(0).map_err(|error| error.to_string())?,
            thread_id: row.get(1).map_err(|error| error.to_string())?,
            status: row.get(2).map_err(|error| error.to_string())?,
            last_event_sequence: row
                .get::<_, i64>(3)
                .map(|value| value as u64)
                .map_err(|error| error.to_string())?,
        }))
    }

    #[cfg(test)]
    pub(super) fn read_item_summary_for_test(
        &self,
        event_id: &str,
    ) -> Result<Option<String>, String> {
        let conn = self.open_projection_store()?;
        conn.query_row(
            "SELECT payload_summary_json FROM projected_items WHERE event_id = ?1",
            params![event_id],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(|error| format!("无法读取 projected_items 摘要: {error}"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ProjectedSessionRow {
    pub(super) session_id: String,
    pub(super) thread_id: String,
    pub(super) status: String,
    pub(super) created_at: Option<String>,
    pub(super) updated_at: String,
    pub(super) archived_at: Option<String>,
    pub(super) title: Option<String>,
    pub(super) model: Option<String>,
    pub(super) workspace_id: Option<String>,
    pub(super) working_dir: Option<String>,
    pub(super) execution_strategy: Option<String>,
    pub(super) metadata_json: Option<String>,
    pub(super) last_event_sequence: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ProjectedTurnRow {
    pub(super) turn_id: String,
    pub(super) session_id: String,
    pub(super) thread_id: String,
    pub(super) status: String,
    pub(super) started_at: Option<String>,
    pub(super) completed_at: Option<String>,
}

fn query_projected_session(
    conn: &Connection,
    session_id: &str,
) -> Result<Option<ProjectedSessionRow>, String> {
    conn.query_row(
        "SELECT session_id, thread_id, status, created_at, updated_at,
                archived_at, title, model, workspace_id, working_dir,
                execution_strategy, metadata_json, last_event_sequence
         FROM projected_sessions
         WHERE session_id = ?1
           AND NOT EXISTS (
                SELECT 1 FROM canonical_thread_spawn_edges AS edge
                WHERE edge.child_thread_id = projected_sessions.thread_id
                  AND edge.status = 'pending'
           )",
        params![session_id],
        projected_session_row,
    )
    .optional()
    .map_err(|error| format!("无法读取 projected_sessions: {error}"))
}

fn query_projected_import_session_rows(
    conn: &Connection,
    source_client: &str,
    source_thread_id: &str,
) -> Result<Vec<ProjectedSessionRow>, String> {
    let source_client_json = serde_json::to_string(source_client)
        .map_err(|error| format!("无法序列化 import sourceClient 查询: {error}"))?;
    let source_thread_id_json = serde_json::to_string(source_thread_id)
        .map_err(|error| format!("无法序列化 import sourceThreadId 查询: {error}"))?;
    let source_client_camel = format!("\"sourceClient\":{source_client_json}");
    let source_client_snake = format!("\"source_client\":{source_client_json}");
    let source_thread_camel = format!("\"sourceThreadId\":{source_thread_id_json}");
    let source_thread_snake = format!("\"source_thread_id\":{source_thread_id_json}");
    let mut stmt = conn
        .prepare(
            "SELECT session_id, thread_id, status, created_at, updated_at,
                    archived_at, title, model, workspace_id, working_dir,
                    execution_strategy, metadata_json, last_event_sequence
             FROM projected_sessions
             WHERE metadata_json IS NOT NULL
               AND (instr(metadata_json, ?1) > 0 OR instr(metadata_json, ?2) > 0)
               AND (instr(metadata_json, ?3) > 0 OR instr(metadata_json, ?4) > 0)
             ORDER BY updated_at DESC, session_id DESC",
        )
        .map_err(|error| format!("无法准备 projected_sessions import 查询: {error}"))?;
    let rows = stmt
        .query_map(
            params![
                source_client_camel,
                source_client_snake,
                source_thread_camel,
                source_thread_snake
            ],
            projected_session_row,
        )
        .map_err(|error| format!("无法查询 projected_sessions import 状态: {error}"))?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("无法读取 projected_sessions import 状态: {error}"))
}

fn projected_session_row(row: &rusqlite::Row<'_>) -> Result<ProjectedSessionRow, rusqlite::Error> {
    Ok(ProjectedSessionRow {
        session_id: row.get(0)?,
        thread_id: row.get(1)?,
        status: row.get(2)?,
        created_at: row.get(3)?,
        updated_at: row.get(4)?,
        archived_at: row.get(5)?,
        title: row.get(6)?,
        model: row.get(7)?,
        workspace_id: row.get(8)?,
        working_dir: row.get(9)?,
        execution_strategy: row.get(10)?,
        metadata_json: row.get(11)?,
        last_event_sequence: row.get::<_, i64>(12)?.max(0) as u64,
    })
}

fn query_projected_session_overviews(
    conn: &Connection,
    include_archived: bool,
    archived_only: bool,
    scope: &SessionListScope,
    limit: Option<usize>,
) -> Result<Vec<AgentSessionOverview>, String> {
    let limit = limit.unwrap_or(1_000);
    let cwd_filters = scope.cwd_filters();
    let workspace_id_filters = scope.workspace_id_filters();
    let cwd_filter_clause = if cwd_filters.is_empty() {
        None
    } else {
        let placeholders = (0..cwd_filters.len())
            .map(|index| format!("?{}", index + 4))
            .collect::<Vec<_>>()
            .join(", ");
        Some(format!("working_dir IN ({placeholders})"))
    };
    let workspace_id_filter_clause = if workspace_id_filters.is_empty() {
        None
    } else {
        let offset = 4 + cwd_filters.len();
        let placeholders = (0..workspace_id_filters.len())
            .map(|index| format!("?{}", offset + index))
            .collect::<Vec<_>>()
            .join(", ");
        Some(format!("workspace_id IN ({placeholders})"))
    };
    let scope_filter_sql = match (cwd_filter_clause, workspace_id_filter_clause) {
        (Some(cwd), Some(workspace_id)) => format!(" AND (({cwd}) OR ({workspace_id}))"),
        (Some(cwd), None) => format!(" AND {cwd}"),
        (None, Some(workspace_id)) => format!(" AND {workspace_id}"),
        (None, None) => String::new(),
    };
    let sql = format!(
        "SELECT session_id, thread_id, status, created_at, updated_at,
                archived_at, title, model, workspace_id, working_dir,
                execution_strategy, metadata_json, last_event_sequence
         FROM projected_sessions
         WHERE (
                (?1 = 1 AND archived_at IS NOT NULL)
                OR (?1 = 0 AND (?2 = 1 OR archived_at IS NULL))
            )
           AND NOT EXISTS (
                SELECT 1 FROM canonical_thread_spawn_edges AS edge
                WHERE edge.child_thread_id = projected_sessions.thread_id
                  AND edge.status = 'pending'
           )
           {scope_filter_sql}
         ORDER BY updated_at DESC, session_id DESC
         LIMIT ?3"
    );
    let mut stmt = conn
        .prepare(&sql)
        .map_err(|error| format!("无法准备 projected_sessions list 查询: {error}"))?;
    let limit_i64 = limit as i64;
    let mut query_params: Vec<&dyn rusqlite::ToSql> =
        vec![&archived_only, &include_archived, &limit_i64];
    for cwd in cwd_filters {
        query_params.push(cwd);
    }
    for workspace_id in workspace_id_filters {
        query_params.push(workspace_id);
    }
    let rows = stmt
        .query_map(
            rusqlite::params_from_iter(query_params),
            projected_session_row,
        )
        .map_err(|error| format!("无法查询 projected_sessions list: {error}"))?;
    rows.map(|row| row.map(|row| projected_session_overview(conn, row)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("无法读取 projected_sessions list: {error}"))
}

fn projected_session_overview(conn: &Connection, row: ProjectedSessionRow) -> AgentSessionOverview {
    let first_user_message =
        query_projected_first_user_message(conn, row.session_id.as_str()).unwrap_or_default();
    let turns = query_projected_turns(conn, row.session_id.as_str()).unwrap_or_default();
    let runtime_state = resolve_session_runtime_state(
        row.status.as_str(),
        0,
        turns.iter().map(|turn| RuntimeTurnSnapshot {
            turn_id: turn.turn_id.as_str(),
            status: turn.status.as_str(),
        }),
        None,
    );
    let messages_count = conn
        .query_row(
            "SELECT COUNT(1)
             FROM projected_items
             WHERE session_id = ?1
               AND item_type IN ('message.created', 'message.delta')",
            params![row.session_id.as_str()],
            |row| row.get::<_, i64>(0),
        )
        .map(|value| value.max(0) as usize)
        .unwrap_or_default();
    AgentSessionOverview {
        session_id: row.session_id.clone(),
        thread_id: Some(row.thread_id),
        title: session_title::resolve_session_title(row.title, first_user_message),
        business_object_ref_metadata: row
            .metadata_json
            .as_deref()
            .and_then(|value| serde_json::from_str::<Value>(value).ok()),
        model: row.model.unwrap_or_default(),
        created_at: row.created_at.unwrap_or_else(|| row.updated_at.clone()),
        updated_at: row.updated_at,
        archived_at: row.archived_at,
        workspace_id: row.workspace_id,
        working_dir: row.working_dir,
        execution_strategy: row.execution_strategy,
        messages_count,
        thread_status: runtime_state.thread_status,
        latest_turn_status: runtime_state.latest_turn_status,
        active_turn_id: runtime_state.active_turn_id,
        queued_turn_count: runtime_state.queued_turn_count,
    }
}

impl ProjectionStore {
    pub(in crate::runtime) fn list_queued_session_ids(&self) -> Result<Vec<String>, String> {
        let conn = self.open_projection_store()?;
        let projection_table = if self.state_path() == self.path() {
            "projected_turns"
        } else {
            "projection.projected_turns"
        };
        let rollout_predicate = self
            .rollout_store()
            .is_some()
            .then_some("AND threads.rollout_path IS NOT NULL")
            .unwrap_or_default();
        let sql = format!(
            "SELECT DISTINCT queued.session_id
             FROM {projection_table} AS queued
             JOIN canonical_threads AS threads
               ON threads.thread_id = queued.thread_id
              AND threads.session_id = queued.session_id
             WHERE queued.status = 'queued'
               {rollout_predicate}
             ORDER BY queued.session_id ASC"
        );
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|error| format!("无法准备 current queued session 查询: {error}"))?;
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(|error| format!("无法查询 current queued session: {error}"))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("无法读取 current queued session: {error}"))
    }

    pub fn list_session_overviews(
        &self,
        params: &AgentSessionListParams,
    ) -> Result<Vec<AgentSessionOverview>, String> {
        let conn = self.open_projection_store()?;
        let scope = SessionListScope::from_params(params);
        query_projected_session_overviews(
            &conn,
            params.include_archived.unwrap_or(false),
            params.archived_only.unwrap_or(false),
            &scope,
            params.limit.map(|value| value as usize),
        )
    }
}

fn query_projected_turns(
    conn: &Connection,
    session_id: &str,
) -> Result<Vec<ProjectedTurnRow>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT turn_id, session_id, thread_id, status, started_at, completed_at
             FROM projected_turns
             WHERE session_id = ?1
             ORDER BY last_event_sequence ASC, turn_id ASC",
        )
        .map_err(|error| format!("无法准备 projected_turns 查询: {error}"))?;
    let rows = stmt
        .query_map(params![session_id], |row| {
            Ok(ProjectedTurnRow {
                turn_id: row.get(0)?,
                session_id: row.get(1)?,
                thread_id: row.get(2)?,
                status: row.get(3)?,
                started_at: row.get(4)?,
                completed_at: row.get(5)?,
            })
        })
        .map_err(|error| format!("无法查询 projected_turns: {error}"))?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("无法读取 projected_turns: {error}"))
}

fn query_projected_item_count(conn: &Connection, session_id: &str) -> Result<usize, String> {
    conn.query_row(
        "SELECT COUNT(1) FROM projected_items WHERE session_id = ?1",
        params![session_id],
        |row| row.get::<_, i64>(0),
    )
    .map(|value| value.max(0) as usize)
    .map_err(|error| format!("无法统计 projected_items: {error}"))
}

fn query_projected_message_count(conn: &Connection, session_id: &str) -> Result<usize, String> {
    let user_count = conn
        .query_row(
            "SELECT COUNT(1)
             FROM projected_items
             WHERE session_id = ?1
               AND item_type = 'message.created'",
            params![session_id],
            |row| row.get::<_, i64>(0),
        )
        .map(|value| value.max(0) as usize)
        .map_err(|error| format!("无法统计 projected_items 用户消息: {error}"))?;
    let assistant_count = conn
        .query_row(
            "SELECT COUNT(DISTINCT COALESCE(turn_id, event_id))
             FROM projected_items
             WHERE session_id = ?1
               AND item_type IN ('message.delta', 'message.delta_batch', 'message.batch')",
            params![session_id],
            |row| row.get::<_, i64>(0),
        )
        .map(|value| value.max(0) as usize)
        .map_err(|error| format!("无法统计 projected_items assistant 消息: {error}"))?;
    Ok(user_count + assistant_count)
}

fn query_projected_message_count_before(
    conn: &Connection,
    session_id: &str,
    before_sequence: i64,
) -> Result<usize, String> {
    let user_count = conn
        .query_row(
            "SELECT COUNT(1)
             FROM projected_items
             WHERE session_id = ?1
               AND sequence < ?2
               AND item_type = 'message.created'",
            params![session_id, before_sequence],
            |row| row.get::<_, i64>(0),
        )
        .map(|value| value.max(0) as usize)
        .map_err(|error| format!("无法统计 projected_items cursor 前用户消息: {error}"))?;
    let assistant_count = conn
        .query_row(
            "SELECT COUNT(DISTINCT COALESCE(turn_id, event_id))
             FROM projected_items
             WHERE session_id = ?1
               AND sequence < ?2
               AND item_type IN ('message.delta', 'message.delta_batch', 'message.batch')",
            params![session_id, before_sequence],
            |row| row.get::<_, i64>(0),
        )
        .map(|value| value.max(0) as usize)
        .map_err(|error| format!("无法统计 projected_items cursor 前 assistant 消息: {error}"))?;
    Ok(user_count + assistant_count)
}

fn query_projected_window_messages(
    conn: &Connection,
    session_id: &str,
    window: ProjectionReadWindow,
    messages_count: usize,
) -> Result<(Vec<Value>, usize), String> {
    let limit = window.history_limit.unwrap_or(40).max(1);
    let before_sequence = window.cursor_before();
    let rows = if let Some(before_sequence) = before_sequence {
        let mut stmt = conn
            .prepare(
                "SELECT event_id, turn_id, sequence, item_type, payload_summary_json, created_at
                 FROM projected_items
                 WHERE session_id = ?1
                   AND sequence < ?2
                   AND item_type IN ('message.created', 'message.delta', 'message.delta_batch', 'message.batch')
                 ORDER BY sequence DESC, event_id DESC
                 LIMIT ?3",
            )
            .map_err(|error| format!("无法准备 projected_items cursor 消息查询: {error}"))?;
        let rows = stmt
            .query_map(
                params![
                    session_id,
                    before_sequence,
                    PROJECTION_SUMMARY_MESSAGE_ROW_LIMIT
                ],
                projected_message_row,
            )
            .map_err(|error| format!("无法查询 projected_items cursor 消息: {error}"))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("无法读取 projected_items cursor 消息: {error}"))?
    } else {
        let mut stmt = conn
            .prepare(
                "SELECT event_id, turn_id, sequence, item_type, payload_summary_json, created_at
                 FROM projected_items
                 WHERE session_id = ?1
                   AND item_type IN ('message.created', 'message.delta', 'message.delta_batch', 'message.batch')
                 ORDER BY sequence DESC, event_id DESC
                 LIMIT ?2",
            )
            .map_err(|error| format!("无法准备 projected_items tail 消息查询: {error}"))?;
        let rows = stmt
            .query_map(
                params![session_id, PROJECTION_SUMMARY_MESSAGE_ROW_LIMIT],
                projected_message_row,
            )
            .map_err(|error| format!("无法查询 projected_items tail 消息: {error}"))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("无法读取 projected_items tail 消息: {error}"))?
    };
    let mut rows = rows;
    rows.reverse();
    let messages = projected_messages_from_rows(rows);
    let available = messages.len();
    let end = available.saturating_sub(window.history_offset.min(available));
    let start = end.saturating_sub(limit);
    let messages = messages.into_iter().skip(start).take(end - start).collect();
    let prefix_count = if let Some(before_sequence) = before_sequence {
        query_projected_message_count_before(conn, session_id, before_sequence)?
    } else {
        messages_count
    };
    let absolute_start = prefix_count.saturating_sub(available).saturating_add(start);
    Ok((messages, absolute_start))
}

#[derive(Debug, Clone)]
struct ProjectedMessageRow {
    event_id: String,
    turn_id: Option<String>,
    sequence: i64,
    item_type: String,
    payload: Value,
    created_at: String,
}

fn projected_message_row(row: &rusqlite::Row<'_>) -> Result<ProjectedMessageRow, rusqlite::Error> {
    let payload_summary_json: String = row.get(4)?;
    Ok(ProjectedMessageRow {
        event_id: row.get(0)?,
        turn_id: row.get(1)?,
        sequence: row.get::<_, i64>(2)?.max(0),
        item_type: row.get(3)?,
        payload: serde_json::from_str::<Value>(&payload_summary_json).unwrap_or(Value::Null),
        created_at: row.get(5)?,
    })
}

fn projected_messages_from_rows(rows: Vec<ProjectedMessageRow>) -> Vec<Value> {
    let mut assistant_by_turn: BTreeMap<String, ProjectedAssistantSummary> = BTreeMap::new();
    let mut messages = Vec::new();
    for row in rows {
        if is_projected_assistant_message_event_type(&row.item_type) {
            if !should_project_message_delta_as_final_text(&row.payload) {
                continue;
            }
            let key = row
                .turn_id
                .clone()
                .unwrap_or_else(|| format!("sequence:{}", row.sequence));
            assistant_by_turn
                .entry(key)
                .or_insert_with(|| ProjectedAssistantSummary::new(&row))
                .push(&row);
            continue;
        }
        flush_assistant_summaries_before(&mut messages, &mut assistant_by_turn, row.sequence);
        messages.push(projected_user_message_value(row));
    }
    flush_all_assistant_summaries(&mut messages, &mut assistant_by_turn);
    messages.sort_by_key(|message| {
        message
            .get("id")
            .and_then(Value::as_i64)
            .unwrap_or(i64::MAX)
    });
    messages
}

fn projected_user_message_value(row: ProjectedMessageRow) -> Value {
    let text = truncate_projection_summary_text(projected_message_text(&row.payload));
    let source_event_id = row.event_id.clone();
    let text_elements = projected_message_text_elements(&row.payload);
    let attachments = projected_message_attachments(&row.payload);
    let mut content = vec![serde_json::json!({
        "type": "text",
        "text": text,
    })];
    let mut text_content_values = vec![text.clone()];
    for element in &text_elements {
        let Some(element_text) = projected_text_element_text(element) else {
            continue;
        };
        if text_content_values
            .iter()
            .any(|existing| existing.trim() == element_text.trim())
        {
            continue;
        }
        text_content_values.push(element_text.to_string());
        content.push(element.clone());
    }
    for attachment in &attachments {
        content.push(serde_json::json!({
            "type": value_string(Some(attachment), &["kind", "type"]).unwrap_or_else(|| "attachment".to_string()),
            "uri": attachment.get("uri").cloned().unwrap_or(Value::Null),
            "metadata": attachment.get("metadata").cloned().unwrap_or(Value::Null),
        }));
    }
    let mut message = serde_json::json!({
        "id": row.sequence,
        "role": "user",
        "runtimeTurnId": row.turn_id,
        "runtime_turn_id": row.turn_id,
        "content": content,
        "attachments": attachments,
        "timestamp": super::timestamp_seconds(Some(row.created_at.as_str())),
        "metadata": {
            "source": "projection_summary",
            "source_event_id": source_event_id,
            "source_event_count": 1,
            "truncated": true,
        },
    });
    if !text_elements.is_empty() {
        if let Some(message_object) = message.as_object_mut() {
            message_object.insert(
                "textElements".to_string(),
                Value::Array(text_elements.clone()),
            );
            message_object.insert("text_elements".to_string(), Value::Array(text_elements));
        }
    }
    message
}

fn flush_assistant_summaries_before(
    messages: &mut Vec<Value>,
    assistant_by_turn: &mut BTreeMap<String, ProjectedAssistantSummary>,
    sequence: i64,
) {
    let ready_keys = assistant_by_turn
        .iter()
        .filter_map(|(key, summary)| (summary.last_sequence < sequence).then_some(key.clone()))
        .collect::<Vec<_>>();
    for key in ready_keys {
        if let Some(summary) = assistant_by_turn.remove(&key) {
            messages.push(summary.into_message());
        }
    }
}

fn flush_all_assistant_summaries(
    messages: &mut Vec<Value>,
    assistant_by_turn: &mut BTreeMap<String, ProjectedAssistantSummary>,
) {
    let summaries = std::mem::take(assistant_by_turn);
    messages.extend(
        summaries
            .into_values()
            .map(ProjectedAssistantSummary::into_message),
    );
}

#[derive(Debug, Clone)]
struct ProjectedAssistantSummary {
    first_sequence: i64,
    last_sequence: i64,
    turn_id: Option<String>,
    text: String,
    created_at: String,
    event_ids: Vec<String>,
}

impl ProjectedAssistantSummary {
    fn new(row: &ProjectedMessageRow) -> Self {
        Self {
            first_sequence: row.sequence,
            last_sequence: row.sequence,
            turn_id: row.turn_id.clone(),
            text: String::new(),
            created_at: row.created_at.clone(),
            event_ids: Vec::new(),
        }
    }

    fn push(&mut self, row: &ProjectedMessageRow) {
        self.last_sequence = row.sequence;
        self.created_at = row.created_at.clone();
        self.text
            .push_str(projected_message_text(&row.payload).as_str());
        self.event_ids.push(row.event_id.clone());
    }

    fn into_message(self) -> Value {
        projected_message_value(
            self.first_sequence,
            "assistant",
            self.turn_id,
            self.text,
            self.created_at,
            self.event_ids,
        )
    }
}

fn projected_message_value(
    sequence: i64,
    role: &str,
    turn_id: Option<String>,
    text: String,
    created_at: String,
    event_ids: Vec<String>,
) -> Value {
    let text = truncate_projection_summary_text(text);
    let source_event_id = event_ids.first().cloned();
    let source_event_count = event_ids.len();
    serde_json::json!({
        "id": sequence,
        "role": role,
        "runtimeTurnId": turn_id,
        "runtime_turn_id": turn_id,
        "content": [
            {
                "type": "text",
                "text": text,
            }
        ],
        "timestamp": super::timestamp_seconds(Some(created_at.as_str())),
        "metadata": {
            "source": "projection_summary",
            "source_event_id": source_event_id,
            "source_event_count": source_event_count,
            "truncated": true,
        },
    })
}

fn truncate_projection_summary_text(text: String) -> String {
    if text.chars().count() <= PROJECTION_SUMMARY_MESSAGE_TEXT_MAX_CHARS {
        return text;
    }
    let mut truncated = text
        .chars()
        .take(PROJECTION_SUMMARY_MESSAGE_TEXT_MAX_CHARS)
        .collect::<String>();
    truncated.push('…');
    truncated
}

fn projected_message_text_elements(payload: &Value) -> Vec<Value> {
    payload
        .get("textElements")
        .or_else(|| payload.get("text_elements"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn projected_message_attachments(payload: &Value) -> Vec<Value> {
    if let Some(input) = payload
        .get("input")
        .and_then(|value| serde_json::from_value::<Vec<AgentInput>>(value.clone()).ok())
    {
        return input
            .iter()
            .filter_map(|part| match part {
                AgentInput::Image { uri, detail } => Some(serde_json::json!({
                    "kind": "image",
                    "uri": uri,
                    "detail": detail,
                })),
                AgentInput::LocalImage { path, detail } => Some(serde_json::json!({
                    "kind": "image",
                    "uri": path,
                    "detail": detail,
                    "metadata": {"localPath": path},
                })),
                AgentInput::Text { .. } | AgentInput::Skill { .. } | AgentInput::Mention { .. } => {
                    None
                }
            })
            .collect();
    }
    payload
        .get("attachments")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn projected_text_element_text(element: &Value) -> Option<&str> {
    element
        .get("text")
        .or_else(|| element.get("content"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
}

fn projected_message_text(payload: &Value) -> String {
    projected_text_from_payload(payload)
        .or_else(|| {
            payload
                .get("input")
                .and_then(|value| serde_json::from_value::<Vec<AgentInput>>(value.clone()).ok())
                .map(|input| {
                    input
                        .iter()
                        .filter_map(|part| match part {
                            AgentInput::Text { text, .. } => Some(text.as_str()),
                            AgentInput::Image { .. }
                            | AgentInput::LocalImage { .. }
                            | AgentInput::Skill { .. }
                            | AgentInput::Mention { .. } => None,
                        })
                        .collect::<String>()
                })
        })
        .unwrap_or_default()
}

fn projected_text_from_payload(payload: &Value) -> Option<String> {
    if let Some(text) = payload
        .as_str()
        .map(str::to_string)
        .filter(|text| !text.is_empty())
    {
        return Some(text);
    }
    value_string(
        Some(payload),
        &[
            "text",
            "delta",
            "content",
            "message",
            "outputText",
            "output_text",
        ],
    )
    .or_else(|| {
        payload
            .get("content")
            .and_then(|content| value_string(Some(content), &["text", "message"]))
    })
    .or_else(|| {
        for key in ["deltas", "messages", "items", "parts", "content"] {
            let Some(values) = payload.get(key).and_then(Value::as_array) else {
                continue;
            };
            let text = values
                .iter()
                .filter_map(projected_text_from_payload)
                .collect::<String>();
            if !text.is_empty() {
                return Some(text);
            }
        }
        None
    })
}

fn is_projected_assistant_message_event_type(event_type: &str) -> bool {
    matches!(
        event_type,
        "message.delta" | "message.delta_batch" | "message.batch"
    )
}

fn should_project_message_delta_as_final_text(payload: &Value) -> bool {
    let Some(phase) = value_string(Some(payload), &["phase", "messagePhase", "message_phase"])
    else {
        return true;
    };
    let normalized = phase.trim().to_ascii_lowercase();
    matches!(normalized.as_str(), "final" | "final_answer")
}

fn query_projected_first_user_message(
    conn: &Connection,
    session_id: &str,
) -> Result<Option<String>, String> {
    let payload_summary = conn
        .query_row(
            "SELECT payload_summary_json
             FROM projected_items
             WHERE session_id = ?1 AND item_type = 'message.created'
             ORDER BY sequence ASC, event_id ASC
             LIMIT 1",
            params![session_id],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(|error| format!("无法读取 projected_items 首条用户消息: {error}"))?;

    Ok(payload_summary
        .and_then(|summary| serde_json::from_str::<Value>(&summary).ok())
        .and_then(|payload| session_title::first_user_message_from_runtime_payload(&payload)))
}

fn apply_event_in_tx(conn: &Connection, event: &AgentEvent) -> Result<bool, String> {
    let thread_id = resolve_projected_thread_id(conn, event)?;
    let existing_event = conn
        .query_row(
            "SELECT session_id, sequence, thread_id, turn_id, item_type,
                    payload_summary_json, created_at
             FROM projected_items WHERE event_id = ?1 LIMIT 1",
            params![event.event_id.as_str()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, String>(6)?,
                ))
            },
        )
        .optional()
        .map_err(|error| format!("无法检查 Projection DB 重复事件: {error}"))?;
    if let Some((
        existing_session_id,
        existing_sequence,
        existing_thread_id,
        existing_turn_id,
        existing_event_type,
        existing_payload_summary,
        existing_timestamp,
    )) = existing_event
    {
        let same_identity =
            existing_session_id == event.session_id && existing_sequence == event.sequence as i64;
        let same_content = existing_thread_id == thread_id
            && existing_turn_id == event.turn_id
            && existing_event_type == event.event_type
            && existing_payload_summary == bounded_payload_summary(&event.payload)
            && existing_timestamp == event.timestamp;
        if same_identity && same_content {
            tracing::debug!(
                session_id = %event.session_id,
                event_id = %event.event_id,
                sequence = event.sequence,
                "忽略已物化的重复 projection event"
            );
            return Ok(false);
        }
        return Err(format!(
            "Projection event identity collision: event_id={} existing_session={} existing_sequence={} incoming_session={} incoming_sequence={}",
            event.event_id,
            existing_session_id,
            existing_sequence,
            event.session_id,
            event.sequence,
        ));
    }

    let last_sequence = conn
        .query_row(
            "SELECT last_sequence FROM projection_watermarks WHERE session_id = ?1",
            params![event.session_id.as_str()],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .map_err(|error| format!("无法读取 Projection DB watermark: {error}"))?;
    if let Some(last_sequence) = last_sequence {
        if event.sequence == last_sequence.max(0) as u64 {
            return Err(format!(
                "Projection sequence collision: session_id={} sequence={} event_id={}",
                event.session_id, event.sequence, event.event_id
            ));
        }
        if event.sequence < last_sequence.max(0) as u64 {
            tracing::warn!(
                session_id = %event.session_id,
                event_id = %event.event_id,
                sequence = event.sequence,
                last_sequence,
                "丢弃 sequence 落后的 projection event"
            );
            return Ok(false);
        }
    }

    assert_projected_turn_owner(conn, event, &thread_id)?;
    upsert_projected_session(conn, event, &thread_id)?;
    apply_projected_queue_event(conn, event)?;
    upsert_projected_turn(conn, event, &thread_id)?;
    insert_projected_item(conn, event, &thread_id)?;
    upsert_watermark(conn, event)?;
    Ok(true)
}

fn resolve_projected_thread_id(conn: &Connection, event: &AgentEvent) -> Result<String, String> {
    let event_thread_id = normalized_text(event.thread_id.as_deref());
    let projected_thread_id = conn
        .query_row(
            "SELECT thread_id FROM projected_sessions WHERE session_id = ?1 LIMIT 1",
            params![event.session_id.as_str()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(|error| format!("无法读取 projected_sessions thread owner: {error}"))?;

    match (projected_thread_id, event_thread_id) {
        (Some(projected_thread_id), Some(event_thread_id)) => {
            if projected_thread_id != event_thread_id {
                return Err(format!(
                    "Projection session thread identity conflict: session_id={} projected_thread_id={} event_thread_id={}",
                    event.session_id, projected_thread_id, event_thread_id
                ));
            }
            Ok(event_thread_id)
        }
        (Some(projected_thread_id), None) => Ok(projected_thread_id),
        (None, Some(event_thread_id)) => Ok(event_thread_id),
        (None, None) => Err(format!(
            "Projection event missing thread identity: session_id={} event_id={}",
            event.session_id, event.event_id
        )),
    }
}

fn assert_projected_turn_owner(
    conn: &Connection,
    event: &AgentEvent,
    thread_id: &str,
) -> Result<(), String> {
    let Some(turn_id) = event.turn_id.as_deref() else {
        return Ok(());
    };
    let existing = conn
        .query_row(
            "SELECT session_id, thread_id FROM projected_turns WHERE turn_id = ?1 LIMIT 1",
            params![turn_id],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(|error| format!("无法读取 projected_turns owner: {error}"))?;
    if let Some((existing_session_id, existing_thread_id)) = existing {
        if existing_session_id != event.session_id || existing_thread_id != thread_id {
            return Err(format!(
                "Projection turn identity conflict: turn_id={} existing_session_id={} existing_thread_id={} event_session_id={} event_thread_id={}",
                turn_id,
                existing_session_id,
                existing_thread_id,
                event.session_id,
                thread_id,
            ));
        }
    }
    Ok(())
}

fn apply_projected_queue_event(conn: &Connection, event: &AgentEvent) -> Result<(), String> {
    if event.event_type != "queue.removed" {
        return Ok(());
    }

    let Some(queued_turn_id) =
        value_string(Some(&event.payload), &["queuedTurnId", "queued_turn_id"])
    else {
        return Ok(());
    };
    conn.execute(
        "DELETE FROM projected_turns WHERE session_id = ?1 AND turn_id = ?2 AND status = 'queued'",
        params![event.session_id.as_str(), queued_turn_id.as_str()],
    )
    .map_err(|error| format!("无法清理 projected_turns queued turn: {error}"))?;
    Ok(())
}

fn clear_session_in_tx(conn: &Connection, session_id: &str) -> Result<(), String> {
    conn.execute(
        "DELETE FROM projected_items WHERE session_id = ?1",
        params![session_id],
    )
    .map_err(|error| format!("无法清理 projected_items: {error}"))?;
    conn.execute(
        "DELETE FROM projected_turns WHERE session_id = ?1",
        params![session_id],
    )
    .map_err(|error| format!("无法清理 projected_turns: {error}"))?;
    conn.execute(
        "DELETE FROM projection_watermarks WHERE session_id = ?1",
        params![session_id],
    )
    .map_err(|error| format!("无法清理 projection_watermarks: {error}"))?;
    conn.execute(
        "DELETE FROM projected_sessions WHERE session_id = ?1",
        params![session_id],
    )
    .map_err(|error| format!("无法清理 projected_sessions: {error}"))?;
    Ok(())
}

fn upsert_projected_session(
    conn: &Connection,
    event: &AgentEvent,
    thread_id: &str,
) -> Result<(), String> {
    let fields = projected_session_fields_from_event(event);
    conn.execute(
        r#"
        INSERT INTO projected_sessions (
            session_id, thread_id, status, created_at, updated_at,
            archived_at, title, model, workspace_id, working_dir,
            execution_strategy, metadata_json, last_event_sequence, last_event_id
        )
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
        ON CONFLICT(session_id) DO UPDATE SET
            status = excluded.status,
            created_at = COALESCE(projected_sessions.created_at, excluded.created_at),
            updated_at = excluded.updated_at,
            archived_at = COALESCE(excluded.archived_at, projected_sessions.archived_at),
            title = COALESCE(excluded.title, projected_sessions.title),
            model = COALESCE(excluded.model, projected_sessions.model),
            workspace_id = COALESCE(excluded.workspace_id, projected_sessions.workspace_id),
            working_dir = COALESCE(excluded.working_dir, projected_sessions.working_dir),
            execution_strategy = COALESCE(
                excluded.execution_strategy,
                projected_sessions.execution_strategy
            ),
            metadata_json = COALESCE(excluded.metadata_json, projected_sessions.metadata_json),
            last_event_sequence = excluded.last_event_sequence,
            last_event_id = excluded.last_event_id
        "#,
        params![
            event.session_id,
            thread_id,
            session_status_from_event(event.event_type.as_str()),
            fields
                .created_at
                .as_deref()
                .unwrap_or(event.timestamp.as_str()),
            event.timestamp,
            fields.archived_at,
            fields.title,
            fields.model,
            fields.workspace_id,
            fields.working_dir,
            fields.execution_strategy,
            fields.metadata_json,
            event.sequence as i64,
            event.event_id,
        ],
    )
    .map_err(|error| format!("无法写入 projected_sessions: {error}"))?;
    Ok(())
}

#[derive(Debug, Default)]
struct ProjectedSessionFields {
    created_at: Option<String>,
    archived_at: Option<String>,
    title: Option<String>,
    model: Option<String>,
    workspace_id: Option<String>,
    working_dir: Option<String>,
    execution_strategy: Option<String>,
    metadata_json: Option<String>,
}

fn projected_session_fields_from_event(event: &AgentEvent) -> ProjectedSessionFields {
    let session = event.payload.get("session");
    ProjectedSessionFields {
        created_at: value_string(session, &["createdAt", "created_at"]),
        archived_at: value_string(session, &["archivedAt", "archived_at"]),
        title: value_string(session, &["title"]),
        model: value_string(session, &["model", "modelName", "model_name"]),
        workspace_id: value_string(session, &["workspaceId", "workspace_id"]),
        working_dir: value_string(session, &["workingDir", "working_dir"]),
        execution_strategy: value_string(session, &["executionStrategy", "execution_strategy"]),
        metadata_json: session
            .and_then(|session| {
                session
                    .get("metadata")
                    .or_else(|| session.get("metadataJson"))
            })
            .map(normalize_metadata_value)
            .transpose()
            .unwrap_or_default(),
    }
}

fn normalize_metadata_value(value: &Value) -> Result<String, String> {
    let metadata = match value {
        Value::String(raw) => {
            serde_json::from_str::<Value>(raw).unwrap_or_else(|_| Value::String(raw.clone()))
        }
        value => value.clone(),
    };
    serde_json::to_string(&metadata)
        .map_err(|error| format!("无法序列化 projected session metadata: {error}"))
}

fn normalized_text(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn value_string(value: Option<&Value>, keys: &[&str]) -> Option<String> {
    let value = value?;
    keys.iter()
        .filter_map(|key| value.get(*key))
        .find_map(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn upsert_projected_turn(
    conn: &Connection,
    event: &AgentEvent,
    thread_id: &str,
) -> Result<(), String> {
    let Some(turn_id) = event.turn_id.as_deref() else {
        return Ok(());
    };
    conn.execute(
        r#"
        INSERT INTO projected_turns (
            turn_id, session_id, thread_id, status, started_at, completed_at,
            last_event_sequence
        )
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
        ON CONFLICT(turn_id) DO UPDATE SET
            status = excluded.status,
            completed_at = COALESCE(excluded.completed_at, projected_turns.completed_at),
            last_event_sequence = excluded.last_event_sequence
        "#,
        params![
            turn_id,
            event.session_id,
            thread_id,
            turn_status_from_event(event.event_type.as_str()),
            event.timestamp,
            turn_completed_at(event),
            event.sequence as i64,
        ],
    )
    .map_err(|error| format!("无法写入 projected_turns: {error}"))?;
    Ok(())
}

fn insert_projected_item(
    conn: &Connection,
    event: &AgentEvent,
    thread_id: &str,
) -> Result<(), String> {
    conn.execute(
        r#"
        INSERT OR IGNORE INTO projected_items (
            event_id, session_id, thread_id, turn_id, sequence,
            item_type, payload_summary_json, created_at
        )
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
        "#,
        params![
            event.event_id,
            event.session_id,
            thread_id,
            event.turn_id,
            event.sequence as i64,
            event.event_type,
            bounded_payload_summary(&event.payload),
            event.timestamp,
        ],
    )
    .map_err(|error| format!("无法写入 projected_items: {error}"))?;
    Ok(())
}

fn upsert_watermark(conn: &Connection, event: &AgentEvent) -> Result<(), String> {
    conn.execute(
        r#"
        INSERT INTO projection_watermarks (
            session_id, last_sequence, last_event_id, updated_at
        )
        VALUES (?1, ?2, ?3, ?4)
        ON CONFLICT(session_id) DO UPDATE SET
            last_sequence = excluded.last_sequence,
            last_event_id = excluded.last_event_id,
            updated_at = excluded.updated_at
        "#,
        params![
            event.session_id,
            event.sequence as i64,
            event.event_id,
            event.timestamp,
        ],
    )
    .map_err(|error| format!("无法写入 projection_watermarks: {error}"))?;
    Ok(())
}

fn turn_completed_at(event: &AgentEvent) -> Option<&str> {
    matches!(
        event.event_type.as_str(),
        "turn.completed" | "turn.failed" | "turn.canceled"
    )
    .then_some(event.timestamp.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projected_user_message_preserves_text_elements_and_attachments() {
        let message = projected_user_message_value(ProjectedMessageRow {
            event_id: "event-user-rich-input".to_string(),
            turn_id: Some("turn-rich-input".to_string()),
            sequence: 7,
            item_type: "message.created".to_string(),
            created_at: "2026-06-08T00:00:01.000Z".to_string(),
            payload: serde_json::json!({
                "input": [
                    {"type": "text", "text": "分析图片"},
                    {
                        "type": "image",
                        "uri": "file:///tmp/rich-input.png"
                    }
                ],
                "content": {
                    "kind": "inline_text",
                    "text": "分析图片"
                },
                "textElements": [
                    {
                        "type": "text",
                        "text": "结合图片说明问题"
                    }
                ],
                "text_elements": [
                    {
                        "type": "text",
                        "text": "结合图片说明问题"
                    }
                ]
            }),
        });

        assert_eq!(message["role"], "user");
        assert_eq!(
            message["attachments"][0]["uri"],
            "file:///tmp/rich-input.png"
        );
        assert_eq!(message["textElements"][0]["text"], "结合图片说明问题");
        assert_eq!(message["text_elements"][0]["text"], "结合图片说明问题");
        assert!(message["content"]
            .as_array()
            .expect("content")
            .iter()
            .any(|part| part["text"] == "结合图片说明问题"));
        assert!(message["content"]
            .as_array()
            .expect("content")
            .iter()
            .any(|part| part["uri"] == "file:///tmp/rich-input.png"));
    }
}
