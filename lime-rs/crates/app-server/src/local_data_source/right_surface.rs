use super::data_error;
use crate::RuntimeCoreError;
use app_server_protocol::WorkspaceRightSurfacePendingListParams;
use app_server_protocol::WorkspaceRightSurfacePendingRequest;
use chrono::SecondsFormat;
use chrono::Utc;
use lime_core::database;
use lime_core::database::DbConnection;
use rusqlite::params;
use rusqlite::Row;

const STATUS_PENDING: &str = "pending";

pub(crate) fn save_pending_request(
    db: &DbConnection,
    request: WorkspaceRightSurfacePendingRequest,
) -> Result<(), RuntimeCoreError> {
    let request_json = serde_json::to_string(&request).map_err(data_error)?;
    let conn = database::lock_db(db).map_err(data_error)?;
    ensure_table(&conn)?;
    conn.execute(
        "INSERT OR REPLACE INTO workspace_right_surface_pending_requests (
             request_id, workspace_id, workspace_root, session_id, surface_kind,
             status, requested_at, expires_at, request_json, updated_at
         )
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            request.request_id,
            request.workspace_id,
            request.workspace_root,
            request.session_id,
            request.surface_kind,
            request.status,
            request.requested_at,
            request.expires_at,
            request_json,
            now_timestamp(),
        ],
    )
    .map_err(data_error)?;
    Ok(())
}

pub(crate) fn list_pending_requests(
    db: &DbConnection,
    params: WorkspaceRightSurfacePendingListParams,
) -> Result<Vec<WorkspaceRightSurfacePendingRequest>, RuntimeCoreError> {
    let conn = database::lock_db(db).map_err(data_error)?;
    ensure_table(&conn)?;
    prune_expired(&conn)?;

    let mut stmt = conn
        .prepare(
            "SELECT request_json
             FROM workspace_right_surface_pending_requests
             WHERE status = ?1
             ORDER BY requested_at DESC",
        )
        .map_err(data_error)?;
    let mut requests = stmt
        .query_map(params![STATUS_PENDING], row_to_pending_request)
        .map_err(data_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(data_error)?;

    let workspace_id = optional_trimmed(params.workspace_id);
    let workspace_root = optional_trimmed(params.workspace_root);
    let session_id = optional_trimmed(params.session_id);
    let surface_kind = optional_trimmed(params.surface_kind);
    requests.retain(|request| {
        optional_filter_matches(&workspace_id, request.workspace_id.as_deref())
            && optional_filter_matches(&workspace_root, request.workspace_root.as_deref())
            && optional_filter_matches(&session_id, request.session_id.as_deref())
            && surface_kind
                .as_ref()
                .is_none_or(|value| request.surface_kind == *value)
    });
    if let Some(limit) = params.limit.map(|value| value as usize) {
        requests.truncate(limit);
    }
    Ok(requests)
}

pub(crate) fn delete_pending_requests(
    db: &DbConnection,
    request_ids: Vec<String>,
) -> Result<Vec<String>, RuntimeCoreError> {
    let conn = database::lock_db(db).map_err(data_error)?;
    ensure_table(&conn)?;

    let mut deleted = Vec::new();
    for request_id in request_ids {
        let request_id = request_id.trim();
        if request_id.is_empty() {
            continue;
        }
        let affected = conn
            .execute(
                "DELETE FROM workspace_right_surface_pending_requests WHERE request_id = ?1",
                params![request_id],
            )
            .map_err(data_error)?;
        if affected > 0 {
            deleted.push(request_id.to_string());
        }
    }
    Ok(deleted)
}

fn ensure_table(conn: &rusqlite::Connection) -> Result<(), RuntimeCoreError> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS workspace_right_surface_pending_requests (
             request_id TEXT PRIMARY KEY,
             workspace_id TEXT,
             workspace_root TEXT,
             session_id TEXT,
             surface_kind TEXT NOT NULL,
             status TEXT NOT NULL,
             requested_at TEXT NOT NULL,
             expires_at TEXT,
             request_json TEXT NOT NULL,
             updated_at TEXT NOT NULL
         );
         CREATE INDEX IF NOT EXISTS idx_right_surface_pending_workspace
             ON workspace_right_surface_pending_requests(workspace_id, surface_kind, status);
         CREATE INDEX IF NOT EXISTS idx_right_surface_pending_session
             ON workspace_right_surface_pending_requests(session_id, surface_kind, status);
         CREATE INDEX IF NOT EXISTS idx_right_surface_pending_expires
             ON workspace_right_surface_pending_requests(status, expires_at);",
    )
    .map_err(data_error)?;
    Ok(())
}

fn prune_expired(conn: &rusqlite::Connection) -> Result<(), RuntimeCoreError> {
    conn.execute(
        "DELETE FROM workspace_right_surface_pending_requests
         WHERE status = ?1 AND expires_at IS NOT NULL AND expires_at <= ?2",
        params![STATUS_PENDING, now_timestamp()],
    )
    .map_err(data_error)?;
    Ok(())
}

fn row_to_pending_request(
    row: &Row<'_>,
) -> Result<WorkspaceRightSurfacePendingRequest, rusqlite::Error> {
    let request_json: String = row.get(0)?;
    serde_json::from_str(&request_json).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(error))
    })
}

fn now_timestamp() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
}

fn optional_trimmed(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn optional_filter_matches(filter: &Option<String>, value: Option<&str>) -> bool {
    filter
        .as_ref()
        .is_none_or(|filter| value == Some(filter.as_str()))
}
