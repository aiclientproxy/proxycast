// Adapted from Codex state/src/runtime/threads.rs and agent-graph-store/local.rs
// (5c19155cbd93bfa099016e7487259f61669823ff), Apache-2.0; see repository NOTICE.

use agent_protocol::ThreadId;
use rusqlite::{params, Connection, OptionalExtension, Params, TransactionBehavior};
use thread_store::{
    AgentGraphStore, AgentGraphStoreFuture, ThreadSpawnEdgeStatus, ThreadSpawnParent,
    ThreadStoreResult,
};

use super::{error, store_error, ProjectionStore};

impl ProjectionStore {
    fn create_pending_thread_spawn_edge_sync(
        &self,
        parent_thread_id: ThreadId,
        child_thread_id: ThreadId,
        child_session_id: String,
    ) -> ThreadStoreResult<()> {
        let child_session_id = child_session_id.trim();
        if child_session_id.is_empty() {
            return Err(error("pending child session id is empty"));
        }
        let mut conn = self.open_thread_store()?;
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(store_error)?;
        ensure_acyclic_spawn_edge(&tx, &parent_thread_id, &child_thread_id)?;
        tx.execute(
            "INSERT INTO canonical_thread_spawn_edges (
                parent_thread_id, child_thread_id, status, pending_session_id
             ) VALUES (?1, ?2, 'pending', ?3)",
            params![
                parent_thread_id.as_str(),
                child_thread_id.as_str(),
                child_session_id
            ],
        )
        .map_err(store_error)?;
        tx.commit().map_err(store_error)
    }

    fn commit_pending_thread_spawn_edge_sync(
        &self,
        child_thread_id: ThreadId,
        child_session_id: String,
    ) -> ThreadStoreResult<bool> {
        let child_session_id = child_session_id.trim();
        if child_session_id.is_empty() {
            return Err(error("pending child session id is empty"));
        }
        let conn = self.open_thread_store()?;
        conn.execute(
            "UPDATE canonical_thread_spawn_edges
             SET status = 'open', pending_session_id = NULL
             WHERE child_thread_id = ?1
               AND status = 'pending'
               AND pending_session_id = ?2",
            params![child_thread_id.as_str(), child_session_id],
        )
        .map(|changed| changed == 1)
        .map_err(store_error)
    }

    fn upsert_thread_spawn_edge_sync(
        &self,
        parent_thread_id: ThreadId,
        child_thread_id: ThreadId,
        status: ThreadSpawnEdgeStatus,
    ) -> ThreadStoreResult<()> {
        if status == ThreadSpawnEdgeStatus::Pending {
            return Err(error(
                "pending thread spawn edges must use the durable reservation boundary",
            ));
        }
        let mut conn = self.open_thread_store()?;
        let tx = conn.transaction().map_err(store_error)?;
        ensure_acyclic_spawn_edge(&tx, &parent_thread_id, &child_thread_id)?;
        tx.execute(
            "INSERT INTO canonical_thread_spawn_edges (
                parent_thread_id, child_thread_id, status, pending_session_id
             ) VALUES (?1, ?2, ?3, NULL)
             ON CONFLICT(child_thread_id) DO UPDATE SET
                parent_thread_id = excluded.parent_thread_id,
                status = excluded.status,
                pending_session_id = NULL",
            params![
                parent_thread_id.as_str(),
                child_thread_id.as_str(),
                status_str(status)
            ],
        )
        .map_err(store_error)?;
        tx.commit().map_err(store_error)?;
        Ok(())
    }

    fn set_thread_spawn_edge_status_sync(
        &self,
        child_thread_id: ThreadId,
        status: ThreadSpawnEdgeStatus,
    ) -> ThreadStoreResult<()> {
        if status == ThreadSpawnEdgeStatus::Pending {
            return Err(error(
                "pending thread spawn edges must use the durable reservation boundary",
            ));
        }
        let conn = self.open_thread_store()?;
        conn.execute(
            "UPDATE canonical_thread_spawn_edges
             SET status = ?2, pending_session_id = NULL
             WHERE child_thread_id = ?1",
            params![child_thread_id.as_str(), status_str(status)],
        )
        .map_err(store_error)?;
        Ok(())
    }

    fn delete_thread_spawn_edge_sync(&self, child_thread_id: ThreadId) -> ThreadStoreResult<()> {
        let conn = self.open_thread_store()?;
        conn.execute(
            "DELETE FROM canonical_thread_spawn_edges WHERE child_thread_id = ?1",
            params![child_thread_id.as_str()],
        )
        .map_err(store_error)?;
        Ok(())
    }

    pub(crate) fn read_thread_spawn_parent_sync(
        &self,
        child_thread_id: ThreadId,
    ) -> ThreadStoreResult<Option<ThreadSpawnParent>> {
        let conn = self.open_thread_store()?;
        self.read_thread_spawn_parent_with_conn(&conn, &child_thread_id)
    }

    pub(super) fn read_thread_spawn_parent_with_conn(
        &self,
        conn: &Connection,
        child_thread_id: &ThreadId,
    ) -> ThreadStoreResult<Option<ThreadSpawnParent>> {
        conn.query_row(
            "SELECT parent_thread_id, status
             FROM canonical_thread_spawn_edges
             WHERE child_thread_id = ?1",
            params![child_thread_id.as_str()],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(store_error)?
        .map(|(parent_thread_id, status)| {
            Ok(ThreadSpawnParent {
                parent_thread_id: ThreadId::new(parent_thread_id),
                status: parse_status(&status)?,
            })
        })
        .transpose()
    }

    fn list_thread_spawn_children_sync(
        &self,
        parent_thread_id: ThreadId,
        status_filter: Option<ThreadSpawnEdgeStatus>,
    ) -> ThreadStoreResult<Vec<ThreadId>> {
        let conn = self.open_thread_store()?;
        match status_filter {
            Some(status) => query_thread_ids(
                &conn,
                "SELECT child_thread_id
                 FROM canonical_thread_spawn_edges
                 WHERE parent_thread_id = ?1 AND status = ?2
                 ORDER BY child_thread_id ASC",
                params![parent_thread_id.as_str(), status_str(status)],
            ),
            None => query_thread_ids(
                &conn,
                "SELECT child_thread_id
                 FROM canonical_thread_spawn_edges
                 WHERE parent_thread_id = ?1
                 ORDER BY child_thread_id ASC",
                params![parent_thread_id.as_str()],
            ),
        }
    }

    fn list_thread_spawn_descendants_sync(
        &self,
        root_thread_id: ThreadId,
        status_filter: Option<ThreadSpawnEdgeStatus>,
    ) -> ThreadStoreResult<Vec<ThreadId>> {
        let conn = self.open_thread_store()?;
        match status_filter {
            Some(status) => query_thread_ids(
                &conn,
                "WITH RECURSIVE subtree(child_thread_id, depth) AS (
                    SELECT child_thread_id, 1
                    FROM canonical_thread_spawn_edges
                    WHERE parent_thread_id = ?1 AND status = ?2
                    UNION ALL
                    SELECT edge.child_thread_id, subtree.depth + 1
                    FROM canonical_thread_spawn_edges AS edge
                    JOIN subtree ON edge.parent_thread_id = subtree.child_thread_id
                    WHERE edge.status = ?2
                 )
                 SELECT child_thread_id
                 FROM subtree
                 ORDER BY depth ASC, child_thread_id ASC",
                params![root_thread_id.as_str(), status_str(status)],
            ),
            None => query_thread_ids(
                &conn,
                "WITH RECURSIVE subtree(child_thread_id, depth) AS (
                    SELECT child_thread_id, 1
                    FROM canonical_thread_spawn_edges
                    WHERE parent_thread_id = ?1
                    UNION ALL
                    SELECT edge.child_thread_id, subtree.depth + 1
                    FROM canonical_thread_spawn_edges AS edge
                    JOIN subtree ON edge.parent_thread_id = subtree.child_thread_id
                 )
                 SELECT child_thread_id
                 FROM subtree
                 ORDER BY depth ASC, child_thread_id ASC",
                params![root_thread_id.as_str()],
            ),
        }
    }

    pub(crate) fn list_pending_thread_spawn_intents_sync(
        &self,
    ) -> ThreadStoreResult<Vec<(ThreadId, ThreadId, String)>> {
        let conn = self.open_thread_store()?;
        let mut statement = conn
            .prepare(
                "SELECT parent_thread_id, child_thread_id, pending_session_id
                 FROM canonical_thread_spawn_edges
                 WHERE status = 'pending'
                 ORDER BY child_thread_id ASC",
            )
            .map_err(store_error)?;
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            })
            .map_err(store_error)?;
        rows.map(|row| {
            let (parent_thread_id, child_thread_id, session_id) = row.map_err(store_error)?;
            let session_id = session_id
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| {
                    error(format!(
                        "pending thread spawn {child_thread_id} has no recovery session id"
                    ))
                })?;
            Ok((
                ThreadId::new(parent_thread_id),
                ThreadId::new(child_thread_id),
                session_id,
            ))
        })
        .collect()
    }

    pub(crate) fn list_pending_thread_spawn_ids_sync(&self) -> ThreadStoreResult<Vec<ThreadId>> {
        let conn = self.open_thread_store()?;
        query_thread_ids(
            &conn,
            "SELECT child_thread_id
             FROM canonical_thread_spawn_edges
             WHERE status = 'pending'
             ORDER BY child_thread_id ASC",
            [],
        )
    }
}

impl AgentGraphStore for ProjectionStore {
    fn create_pending_thread_spawn_edge(
        &self,
        parent_thread_id: ThreadId,
        child_thread_id: ThreadId,
        child_session_id: String,
    ) -> AgentGraphStoreFuture<'_, ()> {
        let store = self.clone();
        Box::pin(async move {
            store.create_pending_thread_spawn_edge_sync(
                parent_thread_id,
                child_thread_id,
                child_session_id,
            )
        })
    }

    fn commit_pending_thread_spawn_edge(
        &self,
        child_thread_id: ThreadId,
        child_session_id: String,
    ) -> AgentGraphStoreFuture<'_, bool> {
        let store = self.clone();
        Box::pin(async move {
            store.commit_pending_thread_spawn_edge_sync(child_thread_id, child_session_id)
        })
    }

    fn upsert_thread_spawn_edge(
        &self,
        parent_thread_id: ThreadId,
        child_thread_id: ThreadId,
        status: ThreadSpawnEdgeStatus,
    ) -> AgentGraphStoreFuture<'_, ()> {
        let store = self.clone();
        Box::pin(async move {
            store.upsert_thread_spawn_edge_sync(parent_thread_id, child_thread_id, status)
        })
    }

    fn set_thread_spawn_edge_status(
        &self,
        child_thread_id: ThreadId,
        status: ThreadSpawnEdgeStatus,
    ) -> AgentGraphStoreFuture<'_, ()> {
        let store = self.clone();
        Box::pin(async move { store.set_thread_spawn_edge_status_sync(child_thread_id, status) })
    }

    fn delete_thread_spawn_edge(&self, child_thread_id: ThreadId) -> AgentGraphStoreFuture<'_, ()> {
        let store = self.clone();
        Box::pin(async move { store.delete_thread_spawn_edge_sync(child_thread_id) })
    }

    fn read_thread_spawn_parent(
        &self,
        child_thread_id: ThreadId,
    ) -> AgentGraphStoreFuture<'_, Option<ThreadSpawnParent>> {
        let store = self.clone();
        Box::pin(async move { store.read_thread_spawn_parent_sync(child_thread_id) })
    }

    fn list_thread_spawn_children(
        &self,
        parent_thread_id: ThreadId,
        status_filter: Option<ThreadSpawnEdgeStatus>,
    ) -> AgentGraphStoreFuture<'_, Vec<ThreadId>> {
        let store = self.clone();
        Box::pin(
            async move { store.list_thread_spawn_children_sync(parent_thread_id, status_filter) },
        )
    }

    fn list_thread_spawn_descendants(
        &self,
        root_thread_id: ThreadId,
        status_filter: Option<ThreadSpawnEdgeStatus>,
    ) -> AgentGraphStoreFuture<'_, Vec<ThreadId>> {
        let store = self.clone();
        Box::pin(
            async move { store.list_thread_spawn_descendants_sync(root_thread_id, status_filter) },
        )
    }
}

fn ensure_acyclic_spawn_edge(
    conn: &Connection,
    parent_thread_id: &ThreadId,
    child_thread_id: &ThreadId,
) -> ThreadStoreResult<()> {
    if parent_thread_id == child_thread_id {
        return Err(error("thread spawn edge cannot target itself"));
    }

    let parent_is_descendant = conn
        .query_row(
            "WITH RECURSIVE descendants(thread_id) AS (
                SELECT child_thread_id
                FROM canonical_thread_spawn_edges
                WHERE parent_thread_id = ?1
                UNION
                SELECT edge.child_thread_id
                FROM canonical_thread_spawn_edges AS edge
                JOIN descendants ON edge.parent_thread_id = descendants.thread_id
             )
             SELECT EXISTS(
                SELECT 1 FROM descendants WHERE thread_id = ?2
             )",
            params![child_thread_id.as_str(), parent_thread_id.as_str()],
            |row| row.get::<_, bool>(0),
        )
        .map_err(store_error)?;
    if parent_is_descendant {
        return Err(error("thread spawn edge would create a cycle"));
    }
    Ok(())
}

fn query_thread_ids<P: Params>(
    conn: &Connection,
    sql: &str,
    params: P,
) -> ThreadStoreResult<Vec<ThreadId>> {
    let mut statement = conn.prepare(sql).map_err(store_error)?;
    let rows = statement
        .query_map(params, |row| row.get::<_, String>(0))
        .map_err(store_error)?;
    rows.map(|row| row.map(ThreadId::new).map_err(store_error))
        .collect()
}

fn status_str(status: ThreadSpawnEdgeStatus) -> &'static str {
    match status {
        ThreadSpawnEdgeStatus::Pending => "pending",
        ThreadSpawnEdgeStatus::Open => "open",
        ThreadSpawnEdgeStatus::Closed => "closed",
    }
}

fn parse_status(status: &str) -> ThreadStoreResult<ThreadSpawnEdgeStatus> {
    match status {
        "pending" => Ok(ThreadSpawnEdgeStatus::Pending),
        "open" => Ok(ThreadSpawnEdgeStatus::Open),
        "closed" => Ok(ThreadSpawnEdgeStatus::Closed),
        _ => Err(error(format!("unknown thread spawn edge status: {status}"))),
    }
}

#[cfg(test)]
#[path = "agent_graph_tests.rs"]
mod tests;
