use agent_protocol::{Thread, ThreadId, ThreadTurnsView, TurnId};
use serde_json::json;
use thread_store::{ReadThreadParams, StoreCursor};

use super::{RuntimeCore, RuntimeCoreError, RuntimeEvent};

pub(crate) struct RevertedThread {
    pub(crate) thread: Thread,
    pub(crate) turns_backwards_cursor: Option<StoreCursor>,
    pub(crate) items_backwards_cursor: Option<StoreCursor>,
}

impl RuntimeCore {
    pub(crate) async fn revert_thread_history(
        &self,
        thread_id: ThreadId,
        before_turn_id: TurnId,
    ) -> Result<RevertedThread, RuntimeCoreError> {
        let store = self.canonical_thread_store()?;
        let projection_store = self.canonical_projection_store()?;
        let thread = store
            .read_thread(ReadThreadParams {
                thread_id: thread_id.clone(),
                include_archived: false,
                turns_view: ThreadTurnsView::Full,
            })
            .await
            .map_err(store_error)?
            .ok_or_else(|| {
                RuntimeCoreError::InvalidRequest(format!("thread not found: {thread_id}"))
            })?;
        if !is_paginated_history(&thread) {
            return Err(RuntimeCoreError::InvalidRequest(
                "thread/revert only supports paginated threads".to_string(),
            ));
        }
        if !thread
            .turns
            .iter()
            .any(|turn| turn.turn_id == before_turn_id)
        {
            return Err(RuntimeCoreError::InvalidRequest(format!(
                "turn not found: {before_turn_id}"
            )));
        }

        let session_id = thread.session_id.as_str().to_string();
        self.ensure_current_session_hydrated(&session_id).await?;
        if self
            .session_loops
            .snapshot(&session_id)
            .await
            .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?
            .and_then(|snapshot| snapshot.active_turn_id)
            .is_some()
        {
            return Err(RuntimeCoreError::InvalidRequest(
                "active turn must be interrupted before thread/revert".to_string(),
            ));
        }
        let boundary = projection_store
            .thread_revert_boundary_sync(&thread_id, &before_turn_id)
            .map_err(store_error)?;

        self.append_runtime_events(
            &session_id,
            thread_id.as_str(),
            None,
            vec![RuntimeEvent::new(
                super::history_replacement::HISTORY_ROLLBACK_EVENT_TYPE,
                json!({
                    "rollbackToSequence": boundary.rollback_to_sequence,
                    "beforeTurnId": before_turn_id.as_str(),
                }),
            )],
        )?;

        self.session_loops
            .shutdown(&session_id)
            .await
            .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;

        let close_result = self
            .backend
            .close_session(&session_id, thread_id.as_str())
            .await;
        {
            let mut state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            state.sessions.remove(&session_id);
            state.thread_elicitation_counts.remove(thread_id.as_str());
            state.thread_goal_continuations.remove(&session_id);
        }
        close_result?;

        let resumed = self.resume_thread(thread_id).await?;
        Ok(RevertedThread {
            thread: resumed.thread,
            turns_backwards_cursor: boundary.turns_backwards_cursor,
            items_backwards_cursor: boundary.items_backwards_cursor,
        })
    }
}

fn is_paginated_history(thread: &Thread) -> bool {
    thread
        .metadata
        .get("historyMode")
        .or_else(|| thread.metadata.get("history_mode"))
        .and_then(serde_json::Value::as_str)
        == Some("paginated")
}

fn store_error(error: thread_store::ThreadStoreError) -> RuntimeCoreError {
    RuntimeCoreError::Backend(error.to_string())
}
