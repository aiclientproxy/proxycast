use std::future::Future;
use std::pin::Pin;

use agent_protocol::ThreadId;
use serde::{Deserialize, Serialize};

use crate::{ThreadStoreError, ThreadStoreResult};

/// Durable identity for an agent thread inside one root-thread tree.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentIdentity {
    pub root_thread_id: ThreadId,
    pub thread_id: ThreadId,
    pub agent_path: String,
    pub nickname: Option<String>,
    pub role: Option<String>,
    pub last_task_message: Option<String>,
}

impl AgentIdentity {
    /// Derive the V2 task name from the final canonical path segment instead of storing it twice.
    pub fn task_name(&self) -> ThreadStoreResult<&str> {
        canonical_agent_path_task_name(&self.agent_path)
    }
}

/// Validates a Codex-compatible absolute agent path and returns its final task-name segment.
pub fn canonical_agent_path_task_name(agent_path: &str) -> ThreadStoreResult<&str> {
    if agent_path == "/morpheus" {
        return Ok("morpheus");
    }

    let Some(path) = agent_path.strip_prefix('/') else {
        return Err(ThreadStoreError::new(
            "absolute agent paths must start with `/root` or be `/morpheus`",
        ));
    };
    let mut segments = path.split('/');
    if segments.next() != Some("root") {
        return Err(ThreadStoreError::new(
            "absolute agent paths must start with `/root` or be `/morpheus`",
        ));
    }
    if path.ends_with('/') {
        return Err(ThreadStoreError::new(
            "absolute agent path must not end with `/`",
        ));
    }

    let mut task_name = "root";
    for segment in segments {
        validate_agent_name(segment)?;
        task_name = segment;
    }
    Ok(task_name)
}

fn validate_agent_name(agent_name: &str) -> ThreadStoreResult<()> {
    if agent_name.is_empty() {
        return Err(ThreadStoreError::new("agent name must not be empty"));
    }
    if matches!(agent_name, "root" | "." | "..") {
        return Err(ThreadStoreError::new("agent name is reserved"));
    }
    if !agent_name
        .chars()
        .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
    {
        return Err(ThreadStoreError::new(
            "agent name must use only lowercase letters, digits, and underscores",
        ));
    }
    Ok(())
}

/// Future returned by [`AgentIdentityStore`] operations.
pub type AgentIdentityStoreFuture<'a, T> =
    Pin<Box<dyn Future<Output = ThreadStoreResult<T>> + Send + 'a>>;

/// Storage-neutral owner for durable agent identity and path resolution.
///
/// `thread_id` and `(root_thread_id, agent_path)` are both unique. Upsert may refresh display and
/// task-message fields, but it must not rebind an existing thread to a different root or path.
pub trait AgentIdentityStore: Send + Sync {
    fn upsert_agent_identity(
        &self,
        identity: AgentIdentity,
    ) -> AgentIdentityStoreFuture<'_, AgentIdentity>;

    fn read_agent_identity(
        &self,
        thread_id: ThreadId,
    ) -> AgentIdentityStoreFuture<'_, Option<AgentIdentity>>;

    /// Deletes a child identity during spawn compensation before it becomes usable.
    fn delete_agent_identity(&self, thread_id: ThreadId) -> AgentIdentityStoreFuture<'_, ()>;

    fn list_agent_identities(
        &self,
        root_thread_id: ThreadId,
    ) -> AgentIdentityStoreFuture<'_, Vec<AgentIdentity>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_codex_agent_paths_and_derives_task_name() {
        for (path, expected_task_name) in [
            ("/root", "root"),
            ("/morpheus", "morpheus"),
            ("/root/researcher", "researcher"),
            ("/root/worker_2", "worker_2"),
            ("/root/research/refactor_v2", "refactor_v2"),
        ] {
            assert_eq!(
                canonical_agent_path_task_name(path).expect("canonical agent path"),
                expected_task_name,
                "{path}"
            );
        }
    }

    #[test]
    fn rejects_non_codex_agent_paths() {
        for path in [
            "",
            "/",
            "root/worker",
            "/other/worker",
            "/root/",
            "/root//worker",
            "/root/./worker",
            "/root/../worker",
            "/root/Worker",
            "/root/worker-name",
            "/root/worker name",
            "/root/工作者",
            "/root/root",
            "/morpheus/worker",
        ] {
            assert!(canonical_agent_path_task_name(path).is_err(), "{path}");
        }
    }

    #[allow(dead_code)]
    fn assert_agent_identity_store_object_safe(store: &dyn AgentIdentityStore) {
        let _ = store;
    }
}
