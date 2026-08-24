use super::super::ProjectionStore;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use rusqlite::{params, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thread_store::{
    CreateProjectParams, CreatedProject, DeletedProject, ListProjectsParams, MoveProjectParams,
    ProjectMoveOutcome, StoredProject, StoredProjectRoot, StoredProjectsPage, ThreadStoreError,
    ThreadStoreResult, UpdateProjectParams, UpdatedProject,
};

const MAX_PAGE_SIZE: u32 = 500;

#[derive(Debug, Deserialize, Serialize)]
struct ProjectCursor {
    position: i64,
    id: String,
}

pub(super) fn list_projects(
    store: &ProjectionStore,
    params: ListProjectsParams,
) -> ThreadStoreResult<StoredProjectsPage> {
    let conn = store.open_thread_store()?;
    let limit = params.limit.clamp(1, MAX_PAGE_SIZE) as usize;
    let cursor = params.cursor.as_deref().map(decode_cursor).transpose()?;
    let (anchor_position, anchor_id) = cursor
        .map(|cursor| (cursor.position, cursor.id))
        .unwrap_or((-1, String::new()));
    let mut statement = conn
        .prepare(
            "SELECT project_id, name, metadata_json, position, created_at_ms, updated_at_ms
             FROM projects
             WHERE position > ?1 OR (position = ?1 AND project_id > ?2)
             ORDER BY position, project_id LIMIT ?3",
        )
        .map_err(error)?;
    let rows = statement
        .query_map(params![anchor_position, anchor_id, limit + 1], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, i64>(5)?,
            ))
        })
        .map_err(error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(error)?;
    let mut projects = rows
        .into_iter()
        .map(
            |(project_id, name, metadata_json, position, created_at_ms, updated_at_ms)| {
                Ok(StoredProject {
                    roots: roots_for(&conn, &project_id).map_err(error)?,
                    id: project_id,
                    name,
                    metadata: decode_metadata(&metadata_json)?,
                    position,
                    created_at_ms,
                    updated_at_ms,
                })
            },
        )
        .collect::<ThreadStoreResult<Vec<_>>>()?;
    let next_cursor = if projects.len() > limit {
        projects.get(limit - 1).map(encode_cursor).transpose()?
    } else {
        None
    };
    projects.truncate(limit);
    Ok(StoredProjectsPage {
        projects,
        next_cursor,
    })
}

pub(super) fn read_project(
    store: &ProjectionStore,
    project_id: String,
) -> ThreadStoreResult<Option<StoredProject>> {
    let conn = store.open_thread_store()?;
    let row = conn
        .query_row(
            "SELECT project_id, name, metadata_json, position, created_at_ms, updated_at_ms
         FROM projects WHERE project_id = ?1",
            params![project_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, i64>(5)?,
                ))
            },
        )
        .optional()
        .map_err(error)?;
    row.map(
        |(project_id, name, metadata_json, position, created_at_ms, updated_at_ms)| {
            Ok(StoredProject {
                roots: roots_for(&conn, &project_id).map_err(error)?,
                id: project_id,
                name,
                metadata: decode_metadata(&metadata_json)?,
                position,
                created_at_ms,
                updated_at_ms,
            })
        },
    )
    .transpose()
}

pub(super) fn create_project(
    store: &ProjectionStore,
    params: CreateProjectParams,
) -> ThreadStoreResult<CreatedProject> {
    let mut conn = store.open_thread_store()?;
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(error)?;
    if let Some(project_id) = tx
        .query_row(
            "SELECT project_id FROM project_idempotency_keys WHERE idempotency_key = ?1",
            params![params.idempotency_key],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(error)?
    {
        let project = read_project_tx(&tx, &project_id)?.ok_or_else(|| {
            ThreadStoreError::invalid_request(format!(
                "idempotency key refers to deleted project: {}",
                params.idempotency_key
            ))
        })?;
        tx.commit().map_err(error)?;
        return Ok(CreatedProject {
            project,
            created: false,
        });
    }
    validate_threads_exist(&tx, &params.thread_ids)?;
    let project_id = uuid::Uuid::now_v7().to_string();
    let now = chrono::Utc::now().timestamp_millis();
    let position = tx
        .query_row(
            "SELECT COALESCE(MAX(position), -1) + 1 FROM projects",
            [],
            |row| row.get::<_, i64>(0),
        )
        .map_err(error)?;
    tx.execute(
        "INSERT INTO projects (project_id, name, metadata_json, position, created_at_ms, updated_at_ms)
         VALUES (?1, ?2, ?3, ?4, ?5, ?5)",
        params![
            project_id,
            params.name,
            serde_json::to_string(&params.metadata).map_err(error)?,
            position,
            now
        ],
    )
    .map_err(error)?;
    for (ordinal, root) in params.roots.iter().enumerate() {
        tx.execute(
            "INSERT INTO project_roots (project_id, ordinal, path) VALUES (?1, ?2, ?3)",
            params![project_id, ordinal as i64, root.path],
        )
        .map_err(error)?;
    }
    tx.execute(
        "INSERT INTO project_idempotency_keys (idempotency_key, project_id, created_at_ms)
         VALUES (?1, ?2, ?3)",
        params![params.idempotency_key, project_id, now],
    )
    .map_err(error)?;
    assign_threads(&tx, &params.thread_ids, Some(&project_id))?;
    let project = read_project_tx(&tx, &project_id)?.expect("created project exists");
    tx.commit().map_err(error)?;
    Ok(CreatedProject {
        project,
        created: true,
    })
}

pub(super) fn update_project(
    store: &ProjectionStore,
    params: UpdateProjectParams,
) -> ThreadStoreResult<Option<UpdatedProject>> {
    let mut conn = store.open_thread_store()?;
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(error)?;
    let Some(current) = read_project_tx(&tx, &params.project_id)? else {
        return Ok(None);
    };
    let name = params.name.unwrap_or_else(|| current.name.clone());
    let roots = params.roots.unwrap_or_else(|| current.roots.clone());
    let metadata = params.metadata.unwrap_or_else(|| current.metadata.clone());
    let changed = name != current.name || roots != current.roots || metadata != current.metadata;
    if changed {
        let now = chrono::Utc::now().timestamp_millis();
        tx.execute(
            "UPDATE projects SET name = ?2, metadata_json = ?3, updated_at_ms = ?4 WHERE project_id = ?1",
            params![params.project_id, name, serde_json::to_string(&metadata).map_err(error)?, now],
        )
        .map_err(error)?;
        if roots != current.roots {
            tx.execute(
                "DELETE FROM project_roots WHERE project_id = ?1",
                params![params.project_id],
            )
            .map_err(error)?;
            for (ordinal, root) in roots.iter().enumerate() {
                tx.execute(
                    "INSERT INTO project_roots (project_id, ordinal, path) VALUES (?1, ?2, ?3)",
                    params![params.project_id, ordinal as i64, root.path],
                )
                .map_err(error)?;
            }
        }
    }
    let project = read_project_tx(&tx, &params.project_id)?.expect("project exists");
    tx.commit().map_err(error)?;
    Ok(Some(UpdatedProject { project, changed }))
}

pub(super) fn move_project(
    store: &ProjectionStore,
    params: MoveProjectParams,
) -> ThreadStoreResult<Option<ProjectMoveOutcome>> {
    let mut conn = store.open_thread_store()?;
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(error)?;
    if read_project_tx(&tx, &params.project_id)?.is_none() {
        return Ok(None);
    }
    if params.before_project_id.as_deref() == Some(params.project_id.as_str()) {
        return Err(ThreadStoreError::invalid_request(format!(
            "project {} cannot be moved before itself",
            params.project_id
        )));
    }
    let original = project_ids(&tx)?;
    let mut ordered = original
        .iter()
        .cloned()
        .filter(|id| id != &params.project_id)
        .collect::<Vec<_>>();
    let index = match params.before_project_id.as_deref() {
        Some(before_project_id) => ordered
            .iter()
            .position(|candidate| candidate == before_project_id)
            .ok_or_else(|| {
                ThreadStoreError::invalid_request(format!(
                    "before project not found: {before_project_id}"
                ))
            })?,
        None => ordered.len(),
    };
    ordered.insert(index, params.project_id.clone());
    if ordered == original {
        return Ok(Some(ProjectMoveOutcome::Unchanged));
    }
    for (position, project_id) in ordered.iter().enumerate() {
        tx.execute(
            "UPDATE projects SET position = ?2 WHERE project_id = ?1",
            params![project_id, position as i64],
        )
        .map_err(error)?;
    }
    tx.execute(
        "UPDATE projects SET updated_at_ms = ?2 WHERE project_id = ?1",
        params![params.project_id, chrono::Utc::now().timestamp_millis()],
    )
    .map_err(error)?;
    tx.commit().map_err(error)?;
    Ok(Some(ProjectMoveOutcome::Moved))
}

pub(super) fn delete_project(
    store: &ProjectionStore,
    project_id: String,
) -> ThreadStoreResult<Option<DeletedProject>> {
    let mut conn = store.open_thread_store()?;
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(error)?;
    if read_project_tx(&tx, &project_id)?.is_none() {
        return Ok(None);
    }
    let (affected_active_thread_ids, affected_archived_thread_ids) =
        thread_ids_for_project(&tx, &project_id)?;
    let affected = affected_active_thread_ids
        .iter()
        .chain(&affected_archived_thread_ids)
        .cloned()
        .collect::<Vec<_>>();
    assign_threads(&tx, &affected, None)?;
    tx.execute(
        "DELETE FROM projects WHERE project_id = ?1",
        params![project_id],
    )
    .map_err(error)?;
    tx.commit().map_err(error)?;
    Ok(Some(DeletedProject {
        affected_active_thread_ids,
        affected_archived_thread_ids,
    }))
}

fn read_project_tx(
    tx: &rusqlite::Transaction<'_>,
    project_id: &str,
) -> ThreadStoreResult<Option<StoredProject>> {
    let row = tx
        .query_row(
        "SELECT project_id, name, metadata_json, position, created_at_ms, updated_at_ms FROM projects WHERE project_id = ?1",
        params![project_id],
        |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, i64>(5)?,
            ))
        },
    )
    .optional()
    .map_err(error)?;
    row.map(
        |(id, name, metadata_json, position, created_at_ms, updated_at_ms)| {
            let roots = roots_for_tx(tx, &id)?;
            Ok(StoredProject {
                id,
                name,
                roots,
                metadata: decode_metadata(&metadata_json)?,
                position,
                created_at_ms,
                updated_at_ms,
            })
        },
    )
    .transpose()
}

fn roots_for(
    conn: &rusqlite::Connection,
    project_id: &str,
) -> rusqlite::Result<Vec<StoredProjectRoot>> {
    conn.prepare("SELECT path FROM project_roots WHERE project_id = ?1 ORDER BY ordinal")?
        .query_map(params![project_id], |row| {
            Ok(StoredProjectRoot { path: row.get(0)? })
        })?
        .collect()
}

fn roots_for_tx(
    tx: &rusqlite::Transaction<'_>,
    project_id: &str,
) -> ThreadStoreResult<Vec<StoredProjectRoot>> {
    tx.prepare("SELECT path FROM project_roots WHERE project_id = ?1 ORDER BY ordinal")
        .map_err(error)?
        .query_map(params![project_id], |row| {
            Ok(StoredProjectRoot { path: row.get(0)? })
        })
        .map_err(error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(error)
}

fn project_ids(tx: &rusqlite::Transaction<'_>) -> ThreadStoreResult<Vec<String>> {
    tx.prepare("SELECT project_id FROM projects ORDER BY position, project_id")
        .map_err(error)?
        .query_map([], |row| row.get(0))
        .map_err(error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(error)
}

fn thread_ids_for_project(
    tx: &rusqlite::Transaction<'_>,
    project_id: &str,
) -> ThreadStoreResult<(Vec<String>, Vec<String>)> {
    let mut statement = tx
        .prepare(
            "SELECT thread_id, thread_json, archived FROM canonical_threads ORDER BY thread_id",
        )
        .map_err(error)?;
    let mut active = Vec::new();
    let mut archived = Vec::new();
    let rows = statement
        .query_map([], |row| {
            let id: String = row.get(0)?;
            let json: String = row.get(1)?;
            let archived: bool = row.get(2)?;
            Ok((id, json, archived))
        })
        .map_err(error)?;
    for row in rows {
        let (id, json, is_archived) = row.map_err(error)?;
        let value = serde_json::from_str::<Value>(&json).map_err(error)?;
        if thread_project_id(&value) == Some(project_id) {
            if is_archived {
                archived.push(id);
            } else {
                active.push(id);
            }
        }
    }
    Ok((active, archived))
}

fn assign_threads(
    tx: &rusqlite::Transaction<'_>,
    thread_ids: &[String],
    project_id: Option<&str>,
) -> ThreadStoreResult<()> {
    for thread_id in thread_ids {
        let json = tx
            .query_row(
                "SELECT thread_json FROM canonical_threads WHERE thread_id = ?1",
                params![thread_id],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(error)?
            .ok_or_else(|| {
                ThreadStoreError::thread_not_found(format!("thread not found: {thread_id}"))
            })?;
        let mut value: Value = serde_json::from_str(&json).map_err(error)?;
        let object = value
            .as_object_mut()
            .ok_or_else(|| ThreadStoreError::new("canonical thread JSON is not an object"))?;
        let metadata = object
            .entry("metadata")
            .or_insert_with(|| Value::Object(Default::default()))
            .as_object_mut()
            .ok_or_else(|| ThreadStoreError::new("canonical thread metadata is not an object"))?;
        match project_id {
            Some(project_id) => {
                metadata.insert("projectId".into(), Value::String(project_id.into()));
            }
            None => {
                metadata.remove("projectId");
                metadata.remove("project_id");
            }
        }
        let now = chrono::Utc::now().timestamp_millis();
        object.insert("updatedAtMs".into(), Value::Number(now.into()));
        tx.execute(
            "UPDATE canonical_threads SET thread_json = ?, updated_at_ms = ? WHERE thread_id = ?",
            params![
                serde_json::to_string(&value).map_err(error)?,
                now,
                thread_id
            ],
        )
        .map_err(error)?;
    }
    Ok(())
}

fn validate_threads_exist(
    tx: &rusqlite::Transaction<'_>,
    thread_ids: &[String],
) -> ThreadStoreResult<()> {
    for thread_id in thread_ids {
        let exists = tx
            .query_row(
                "SELECT 1 FROM canonical_threads WHERE thread_id = ?1",
                params![thread_id],
                |_| Ok(()),
            )
            .optional()
            .map_err(error)?
            .is_some();
        if !exists {
            return Err(ThreadStoreError::thread_not_found(format!(
                "thread not found: {thread_id}"
            )));
        }
    }
    Ok(())
}

fn thread_project_id(value: &Value) -> Option<&str> {
    let metadata = value.get("metadata")?;
    metadata
        .get("projectId")
        .or_else(|| metadata.get("project_id"))
        .and_then(Value::as_str)
}

fn encode_cursor(project: &StoredProject) -> ThreadStoreResult<String> {
    let encoded = serde_json::to_vec(&ProjectCursor {
        position: project.position,
        id: project.id.clone(),
    })
    .map_err(error)?;
    Ok(URL_SAFE_NO_PAD.encode(encoded))
}

fn decode_cursor(cursor: &str) -> ThreadStoreResult<ProjectCursor> {
    let bytes = URL_SAFE_NO_PAD
        .decode(cursor)
        .map_err(|_| ThreadStoreError::invalid_request("invalid project cursor"))?;
    let cursor = serde_json::from_slice::<ProjectCursor>(&bytes)
        .map_err(|_| ThreadStoreError::invalid_request("invalid project cursor"))?;
    if cursor.id.is_empty() || cursor.position < 0 {
        return Err(ThreadStoreError::invalid_request("invalid project cursor"));
    }
    Ok(cursor)
}

fn decode_metadata(value: &str) -> ThreadStoreResult<std::collections::BTreeMap<String, String>> {
    serde_json::from_str(value).map_err(error)
}

fn error(error: impl std::fmt::Display) -> ThreadStoreError {
    ThreadStoreError::new(error.to_string())
}

#[cfg(test)]
#[path = "projects_tests.rs"]
mod tests;
