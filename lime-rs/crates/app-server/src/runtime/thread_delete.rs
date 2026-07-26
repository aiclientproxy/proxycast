use std::collections::HashMap;
use std::path::Path;

use agent_protocol::{ThreadId, ThreadTurnsView};
use thread_store::{ReadThreadParams, ThreadStore};

use super::{RuntimeCore, RuntimeCoreError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeletedThread {
    pub(crate) thread_id: String,
    pub(crate) session_id: String,
}

impl RuntimeCore {
    pub(crate) async fn delete_thread(
        &self,
        root_thread_id: ThreadId,
    ) -> Result<Vec<DeletedThread>, RuntimeCoreError> {
        let store = self.projection_store.as_deref().ok_or_else(|| {
            RuntimeCoreError::Backend("canonical thread store is unavailable".to_string())
        })?;
        let root = store
            .read_thread(ReadThreadParams {
                thread_id: root_thread_id.clone(),
                include_archived: true,
                turns_view: ThreadTurnsView::NotLoaded,
            })
            .await
            .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?
            .ok_or_else(|| {
                RuntimeCoreError::Backend(format!("thread not found: {root_thread_id}"))
            })?;
        if root
            .metadata
            .get("ephemeral")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            return Err(RuntimeCoreError::Backend(format!(
                "cannot delete ephemeral thread: {root_thread_id}"
            )));
        }

        let snapshot = store
            .snapshot_thread_delete_subtree(&root_thread_id)
            .map_err(RuntimeCoreError::Backend)?;
        let identities = snapshot
            .persisted
            .iter()
            .map(|thread| {
                (
                    thread.thread_id.clone(),
                    thread.session_id.clone(),
                    thread.rollout_path.as_deref(),
                    thread.archived,
                )
            })
            .chain(snapshot.pending_only.iter().map(|thread| {
                (
                    thread.thread_id.clone(),
                    thread.pending_session_id.clone(),
                    None,
                    false,
                )
            }))
            .collect::<Vec<_>>();

        for (thread_id, session_id, _, _) in &identities {
            self.session_loops
                .shutdown(session_id)
                .await
                .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;
            self.backend
                .close_session(session_id, thread_id.as_str())
                .await?;
        }

        for (thread_id, session_id, rollout_path, archived) in &identities {
            if let (Some(rollout_store), Some(rollout_path)) = (store.rollout_store(), rollout_path)
            {
                rollout_store
                    .delete(
                        Path::new(rollout_path),
                        session_id,
                        thread_id.as_str(),
                        *archived,
                    )
                    .map_err(RuntimeCoreError::Backend)?;
            }
            if let Some(event_log_writer) = self.event_log_writer.as_ref() {
                event_log_writer
                    .clear_session(session_id)
                    .map_err(RuntimeCoreError::Backend)?;
            }
            if let Some(trace_event_writer) = self.trace_event_writer.as_ref() {
                trace_event_writer
                    .clear_session(session_id)
                    .map_err(RuntimeCoreError::Backend)?;
            }
            if let Some(sidecar_store) = self.sidecar_store.as_ref() {
                sidecar_store
                    .clear_session(session_id)
                    .map_err(RuntimeCoreError::Backend)?;
            }
            if let Some(telemetry_store) = self.telemetry_store.as_ref() {
                telemetry_store
                    .clear_session(session_id)
                    .map_err(RuntimeCoreError::Backend)?;
            }
        }

        store
            .delete_thread_subtree_data(&snapshot)
            .map_err(RuntimeCoreError::Backend)?;

        let session_ids_by_thread = identities
            .iter()
            .map(|(thread_id, session_id, _, _)| (thread_id.as_str(), session_id.as_str()))
            .collect::<HashMap<_, _>>();
        let deleted = snapshot
            .thread_ids_deepest_first
            .iter()
            .map(|thread_id| {
                let session_id =
                    session_ids_by_thread
                        .get(thread_id.as_str())
                        .ok_or_else(|| {
                            RuntimeCoreError::Backend(format!(
                                "thread delete snapshot identity missing: {thread_id}"
                            ))
                        })?;
                Ok(DeletedThread {
                    thread_id: thread_id.to_string(),
                    session_id: session_id.to_string(),
                })
            })
            .collect::<Result<Vec<_>, RuntimeCoreError>>()?;

        {
            let mut state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            for (thread_id, session_id, _, _) in &identities {
                super::approval_cache::remove_session(
                    &mut state.session_approval_cache,
                    session_id,
                );
                state.thread_elicitation_counts.remove(thread_id.as_str());
                state.thread_goal_continuations.remove(session_id);
                state.sessions.remove(session_id);
            }
        }
        Ok(deleted)
    }
}
