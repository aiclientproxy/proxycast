use agent_protocol::ThreadId;
use rusqlite::{params, Connection, OptionalExtension};
use thread_store::{
    canonical_agent_path_task_name, AgentIdentity, AgentIdentityStore, AgentIdentityStoreFuture,
    ThreadStoreError, ThreadStoreResult,
};

use crate::ProjectionStore;

impl ProjectionStore {
    fn open_agent_identity_store(&self) -> ThreadStoreResult<Connection> {
        let conn = Connection::open(self.path()).map_err(store_error)?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS agent_identities (
                thread_id TEXT PRIMARY KEY,
                root_thread_id TEXT NOT NULL,
                agent_path TEXT NOT NULL,
                nickname TEXT,
                role TEXT,
                last_task_message TEXT,
                UNIQUE (root_thread_id, agent_path)
            );
            CREATE INDEX IF NOT EXISTS idx_agent_identities_root_path
                ON agent_identities (root_thread_id, agent_path, thread_id);
            "#,
        )
        .map_err(store_error)?;
        Ok(conn)
    }

    fn upsert_agent_identity_sync(
        &self,
        identity: AgentIdentity,
    ) -> ThreadStoreResult<AgentIdentity> {
        validate_identity(&identity)?;
        let mut conn = self.open_agent_identity_store()?;
        let tx = conn.transaction().map_err(store_error)?;
        if let Some(existing) = read_identity_by_thread(&tx, &identity.thread_id)? {
            if existing.root_thread_id != identity.root_thread_id
                || existing.agent_path != identity.agent_path
            {
                return Err(error(format!(
                    "agent thread {} cannot be rebound to a different root or path",
                    identity.thread_id
                )));
            }
            tx.execute(
                "UPDATE agent_identities
                 SET nickname = ?2, role = ?3, last_task_message = ?4
                 WHERE thread_id = ?1",
                params![
                    identity.thread_id.as_str(),
                    identity.nickname,
                    identity.role,
                    identity.last_task_message,
                ],
            )
            .map_err(store_error)?;
            tx.commit().map_err(store_error)?;
            return Ok(identity);
        }
        if let Some(existing) =
            read_identity_by_root_path(&tx, &identity.root_thread_id, &identity.agent_path)?
        {
            return Err(error(format!(
                "agent path {} in root {} already belongs to thread {}",
                identity.agent_path, identity.root_thread_id, existing.thread_id
            )));
        }
        tx.execute(
            "INSERT INTO agent_identities (
                thread_id, root_thread_id, agent_path, nickname, role, last_task_message
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                identity.thread_id.as_str(),
                identity.root_thread_id.as_str(),
                identity.agent_path,
                identity.nickname,
                identity.role,
                identity.last_task_message,
            ],
        )
        .map_err(store_error)?;
        tx.commit().map_err(store_error)?;
        Ok(identity)
    }

    pub(crate) fn read_agent_identity_sync(
        &self,
        thread_id: ThreadId,
    ) -> ThreadStoreResult<Option<AgentIdentity>> {
        validate_thread_id(&thread_id, "thread")?;
        let conn = self.open_agent_identity_store()?;
        read_identity_by_thread(&conn, &thread_id)
    }

    fn delete_agent_identity_sync(&self, thread_id: ThreadId) -> ThreadStoreResult<()> {
        validate_thread_id(&thread_id, "thread")?;
        let conn = self.open_agent_identity_store()?;
        conn.execute(
            "DELETE FROM agent_identities WHERE thread_id = ?1",
            params![thread_id.as_str()],
        )
        .map_err(store_error)?;
        Ok(())
    }

    pub(crate) fn list_agent_identities_sync(
        &self,
        root_thread_id: ThreadId,
    ) -> ThreadStoreResult<Vec<AgentIdentity>> {
        validate_thread_id(&root_thread_id, "root thread")?;
        let conn = self.open_agent_identity_store()?;
        let mut statement = conn
            .prepare(
                "SELECT root_thread_id, thread_id, agent_path, nickname, role, last_task_message
                 FROM agent_identities
                 WHERE root_thread_id = ?1
                 ORDER BY agent_path ASC, thread_id ASC",
            )
            .map_err(store_error)?;
        let rows = statement
            .query_map(params![root_thread_id.as_str()], row_to_identity)
            .map_err(store_error)?;
        rows.map(|row| row.map_err(store_error)).collect()
    }
}

impl AgentIdentityStore for ProjectionStore {
    fn upsert_agent_identity(
        &self,
        identity: AgentIdentity,
    ) -> AgentIdentityStoreFuture<'_, AgentIdentity> {
        let store = self.clone();
        Box::pin(async move { store.upsert_agent_identity_sync(identity) })
    }

    fn read_agent_identity(
        &self,
        thread_id: ThreadId,
    ) -> AgentIdentityStoreFuture<'_, Option<AgentIdentity>> {
        let store = self.clone();
        Box::pin(async move { store.read_agent_identity_sync(thread_id) })
    }

    fn delete_agent_identity(&self, thread_id: ThreadId) -> AgentIdentityStoreFuture<'_, ()> {
        let store = self.clone();
        Box::pin(async move { store.delete_agent_identity_sync(thread_id) })
    }

    fn list_agent_identities(
        &self,
        root_thread_id: ThreadId,
    ) -> AgentIdentityStoreFuture<'_, Vec<AgentIdentity>> {
        let store = self.clone();
        Box::pin(async move { store.list_agent_identities_sync(root_thread_id) })
    }
}

fn read_identity_by_thread(
    conn: &Connection,
    thread_id: &ThreadId,
) -> ThreadStoreResult<Option<AgentIdentity>> {
    conn.query_row(
        "SELECT root_thread_id, thread_id, agent_path, nickname, role, last_task_message
         FROM agent_identities WHERE thread_id = ?1",
        params![thread_id.as_str()],
        row_to_identity,
    )
    .optional()
    .map_err(store_error)
}

fn read_identity_by_root_path(
    conn: &Connection,
    root_thread_id: &ThreadId,
    agent_path: &str,
) -> ThreadStoreResult<Option<AgentIdentity>> {
    conn.query_row(
        "SELECT root_thread_id, thread_id, agent_path, nickname, role, last_task_message
         FROM agent_identities WHERE root_thread_id = ?1 AND agent_path = ?2",
        params![root_thread_id.as_str(), agent_path],
        row_to_identity,
    )
    .optional()
    .map_err(store_error)
}

fn row_to_identity(row: &rusqlite::Row<'_>) -> rusqlite::Result<AgentIdentity> {
    Ok(AgentIdentity {
        root_thread_id: ThreadId::new(row.get::<_, String>(0)?),
        thread_id: ThreadId::new(row.get::<_, String>(1)?),
        agent_path: row.get(2)?,
        nickname: row.get(3)?,
        role: row.get(4)?,
        last_task_message: row.get(5)?,
    })
}

fn validate_identity(identity: &AgentIdentity) -> ThreadStoreResult<()> {
    validate_thread_id(&identity.root_thread_id, "root thread")?;
    validate_thread_id(&identity.thread_id, "thread")?;
    canonical_agent_path_task_name(&identity.agent_path)?;
    Ok(())
}

fn validate_thread_id(thread_id: &ThreadId, field: &str) -> ThreadStoreResult<()> {
    if thread_id.as_str().trim().is_empty() {
        return Err(error(format!("agent identity {field} must not be empty")));
    }
    Ok(())
}

fn error(message: impl Into<String>) -> ThreadStoreError {
    ThreadStoreError::new(message)
}

fn store_error(source: impl std::fmt::Display) -> ThreadStoreError {
    error(source.to_string())
}

#[cfg(test)]
mod tests {
    use agent_protocol::ThreadId;
    use futures::executor::block_on;
    use thread_store::{AgentIdentity, AgentIdentityStore};

    use crate::ProjectionStore;

    fn identity(thread_id: &str, root: &str, path: &str) -> AgentIdentity {
        AgentIdentity {
            root_thread_id: ThreadId::new(root),
            thread_id: ThreadId::new(thread_id),
            agent_path: path.to_string(),
            nickname: Some(format!("nickname-{thread_id}")),
            role: Some("worker".to_string()),
            last_task_message: Some("initial task".to_string()),
        }
    }

    #[test]
    fn identity_is_durable_path_sorted_and_derives_task_name() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("projection.sqlite");
        let store = ProjectionStore::initialize(&path).expect("projection store");
        let beta = identity("thread-beta", "root", "/root/beta");
        let alpha = identity("thread-alpha", "root", "/root/alpha");
        block_on(store.upsert_agent_identity(beta)).expect("store beta");
        block_on(store.upsert_agent_identity(alpha.clone())).expect("store alpha");
        drop(store);

        let reopened = ProjectionStore::initialize(path).expect("reopen projection store");
        let identities = block_on(reopened.list_agent_identities(ThreadId::new("root")))
            .expect("list identities");
        assert_eq!(
            identities
                .iter()
                .map(|identity| identity.agent_path.as_str())
                .collect::<Vec<_>>(),
            vec!["/root/alpha", "/root/beta"]
        );
        assert_eq!(identities[0].task_name().expect("task name"), "alpha");
        assert_eq!(
            block_on(reopened.read_agent_identity(ThreadId::new("thread-alpha")))
                .expect("read identity"),
            Some(alpha)
        );
    }

    #[test]
    fn identity_upsert_refreshes_mutable_fields_but_rejects_rebinding() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store");
        let original = identity("thread", "root", "/root/worker");
        block_on(store.upsert_agent_identity(original.clone())).expect("store identity");

        let mut updated = original.clone();
        updated.last_task_message = Some("next task".to_string());
        assert_eq!(
            block_on(store.upsert_agent_identity(updated.clone())).expect("update identity"),
            updated
        );

        let rebound = identity("thread", "other-root", "/root/other_worker");
        assert!(block_on(store.upsert_agent_identity(rebound))
            .expect_err("rebind")
            .to_string()
            .contains("cannot be rebound"));
    }

    #[test]
    fn identity_rejects_duplicate_root_path_for_another_thread() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store");
        block_on(store.upsert_agent_identity(identity("thread-1", "root", "/root/worker")))
            .expect("store identity");
        let error =
            block_on(store.upsert_agent_identity(identity("thread-2", "root", "/root/worker")))
                .expect_err("duplicate path");
        assert!(error.to_string().contains("already belongs"));
    }
}
