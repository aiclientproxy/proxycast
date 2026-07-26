use super::status::agent_turn_is_active;
use super::{RuntimeCore, RuntimeCoreError, RuntimeEvent};
use agent_protocol::ResponseItem;
use agent_runtime::session_loop::RuntimeSessionInput;
use app_server_protocol::protocol::v2::{ThreadInjectItemsParams, ThreadInjectItemsResponse};
use serde_json::json;

pub(super) const RESPONSE_ITEM_INJECTED_EVENT_TYPE: &str = "response_item.injected";

impl RuntimeCore {
    pub async fn inject_thread_items(
        &self,
        mut params: ThreadInjectItemsParams,
    ) -> Result<ThreadInjectItemsResponse, RuntimeCoreError> {
        params.thread_id = params.thread_id.trim().to_string();
        if params.thread_id.is_empty() {
            return Err(RuntimeCoreError::InvalidRequest(
                "thread/inject_items requires threadId".to_string(),
            ));
        }
        if params.items.is_empty() {
            return Err(RuntimeCoreError::InvalidRequest(
                "items must not be empty".to_string(),
            ));
        }

        for (index, item) in params.items.iter().enumerate() {
            let parsed = serde_json::from_value::<ResponseItem>(item.clone()).map_err(|error| {
                RuntimeCoreError::InvalidRequest(format!(
                    "items[{index}] is not a valid response item: {error}"
                ))
            })?;
            if parsed.contains_remote_image_url() {
                return Err(RuntimeCoreError::InvalidRequest(
                    "remote image URLs are not allowed; use a local or data URL".to_string(),
                ));
            }
        }

        let session_id = match self.loaded_session_id_for_thread(&params.thread_id) {
            Some(session_id) => session_id,
            None => {
                let thread = self
                    .read_thread(agent_protocol::thread::ThreadReadParams {
                        thread_id: agent_protocol::ThreadId::new(params.thread_id.clone()),
                        turns_view: agent_protocol::ThreadTurnsView::NotLoaded,
                    })
                    .await
                    .map_err(map_thread_lookup_error)?;
                if thread.thread.archived {
                    return Err(RuntimeCoreError::InvalidRequest(format!(
                        "thread/inject_items cannot inject into archived thread {}",
                        params.thread_id
                    )));
                }
                let resumed = self
                    .resume_thread(agent_protocol::ThreadId::new(params.thread_id.clone()))
                    .await
                    .map_err(map_thread_lookup_error)?;
                resumed.thread.session_id.to_string()
            }
        };
        let active_turn_id = {
            let state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .get(&session_id)
                .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.clone()))?;
            stored
                .turns
                .iter()
                .find(|turn| agent_turn_is_active(turn.status))
                .map(|turn| turn.turn_id.clone())
        };
        let active_input = params
            .items
            .iter()
            .cloned()
            .map(RuntimeSessionInput::RawResponseItem)
            .collect::<Vec<_>>();
        let events = params
            .items
            .into_iter()
            .map(|item| {
                RuntimeEvent::new(
                    RESPONSE_ITEM_INJECTED_EVENT_TYPE,
                    json!({
                        "visibility": "provider_only",
                        "source": "thread/inject_items",
                        "item": item,
                    }),
                )
            })
            .collect();
        self.append_runtime_events(&session_id, &params.thread_id, None, events)?;

        if let Some(active_turn_id) = active_turn_id {
            if let Some(session) = self.session_loops.get_existing(&session_id).await {
                let _ = session
                    .steer_for_turn_id(Some(&active_turn_id), active_input)
                    .await;
            }
        }

        Ok(ThreadInjectItemsResponse {})
    }
}

fn map_thread_lookup_error(error: RuntimeCoreError) -> RuntimeCoreError {
    match error {
        RuntimeCoreError::Backend(message) if message.starts_with("thread not found:") => {
            RuntimeCoreError::InvalidRequest(message)
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_runtime::session_loop::{RuntimeSessionClosureTask, RuntimeSessionTaskOutcome};
    use app_server_protocol::{AgentSessionStartParams, AgentTurn, AgentTurnStatus};
    use model_provider::current_client::CurrentProviderContent;
    use serde_json::json;
    use std::sync::Arc;
    use tokio::sync::{Mutex, Notify};
    use tokio::time::{timeout, Duration};

    #[tokio::test]
    async fn injected_response_item_is_durable_provider_history_only() {
        let runtime = RuntimeCore::default();
        let session = runtime
            .start_session(AgentSessionStartParams {
                session_id: Some("session-inject-items".to_string()),
                thread_id: Some("thread-inject-items".to_string()),
                app_id: "test".to_string(),
                workspace_id: None,
                business_object_ref: None,
                locale: None,
            })
            .expect("inject-items session")
            .session;
        let item = json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "injected context"}]
        });

        runtime
            .inject_thread_items(ThreadInjectItemsParams {
                thread_id: session.thread_id.clone(),
                items: vec![item.clone()],
            })
            .await
            .expect("inject response item");

        let history = {
            let state = runtime
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .get(&session.session_id)
                .expect("stored injected session");
            super::super::provider_history::provider_history_excluding_current_turn_input(
                stored,
                None,
                "future-turn",
            )
            .expect("provider history")
        };

        assert!(matches!(
            history.as_slice(),
            [message]
                if matches!(message.content.as_slice(), [CurrentProviderContent::RawResponseItem(value)] if value == &item)
        ));
    }

    #[tokio::test]
    async fn injection_rejects_empty_malformed_and_remote_image_items() {
        let runtime = RuntimeCore::default();
        runtime
            .start_session(AgentSessionStartParams {
                session_id: Some("session-inject-invalid".to_string()),
                thread_id: Some("thread-inject-invalid".to_string()),
                app_id: "test".to_string(),
                workspace_id: None,
                business_object_ref: None,
                locale: None,
            })
            .expect("inject-items session");

        for items in [
            vec![],
            vec![json!({"type": "message", "role": "assistant"})],
            vec![json!({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_image", "image_url": "https://example.com/a.png"}]
            })],
        ] {
            let error = runtime
                .inject_thread_items(ThreadInjectItemsParams {
                    thread_id: "thread-inject-invalid".to_string(),
                    items,
                })
                .await
                .expect_err("invalid injection must fail closed");
            assert!(matches!(error, RuntimeCoreError::InvalidRequest(_)));
        }
    }

    #[tokio::test]
    async fn active_turn_receives_raw_response_item_through_session_actor() {
        let runtime = RuntimeCore::default();
        let session = runtime
            .start_session(AgentSessionStartParams {
                session_id: Some("session-inject-active".to_string()),
                thread_id: Some("thread-inject-active".to_string()),
                app_id: "test".to_string(),
                workspace_id: None,
                business_object_ref: None,
                locale: None,
            })
            .expect("active inject session")
            .session;
        let turn_id = "turn-inject-active";
        let started = Arc::new(Notify::new());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let task_started = Arc::clone(&started);
        let task_seen = Arc::clone(&seen);
        let task = RuntimeSessionClosureTask::new(
            turn_id,
            Vec::new(),
            move |context, _initial_input, _cancellation| {
                let started = Arc::clone(&task_started);
                let seen = Arc::clone(&task_seen);
                Box::pin(async move {
                    started.notify_one();
                    context.wait_for_pending_input().await;
                    seen.lock()
                        .await
                        .extend(context.take_pending_input(false).await);
                    Ok(())
                })
            },
        );
        let actor = runtime
            .session_loops
            .get_or_create(&session.session_id)
            .await;
        let submission = actor
            .submit(Arc::new(task), false)
            .await
            .expect("submit active inject task");
        timeout(Duration::from_secs(2), started.notified())
            .await
            .expect("active inject task must start");
        {
            let mut state = runtime
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            state
                .sessions
                .get_mut(&session.session_id)
                .expect("stored active inject session")
                .turns
                .push(AgentTurn {
                    turn_id: turn_id.to_string(),
                    session_id: session.session_id.clone(),
                    thread_id: session.thread_id.clone(),
                    status: AgentTurnStatus::Running,
                    started_at: None,
                    completed_at: None,
                });
        }
        let item = json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "active injected context"}],
            "provider_extension": {"keep": true}
        });

        runtime
            .inject_thread_items(ThreadInjectItemsParams {
                thread_id: session.thread_id.clone(),
                items: vec![item.clone()],
            })
            .await
            .expect("inject into active turn");

        assert_eq!(
            timeout(Duration::from_secs(2), submission.completion)
                .await
                .expect("active inject completion timeout")
                .expect("active inject completion channel")
                .expect("active inject task result"),
            RuntimeSessionTaskOutcome::Completed
        );
        let seen = seen.lock().await;
        assert!(matches!(
            seen.as_slice(),
            [RuntimeSessionInput::RawResponseItem(value)] if value == &item
        ));
    }

    #[tokio::test]
    async fn cold_thread_is_hydrated_and_archived_thread_stays_unloaded() {
        let temp = tempfile::tempdir().expect("inject restart temp dir");
        let roots = crate::StorageRoots::initialize(temp.path(), temp.path().join("app-server"))
            .expect("inject restart storage roots");
        let database_path = roots.projection_db_path.clone();
        let event_log_root = roots.event_log_root.clone();
        let thread_id = "thread-inject-restart";
        let first_item = json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "before restart"}]
        });
        {
            let projection = Arc::new(
                crate::ProjectionStore::initialize(&database_path)
                    .expect("inject restart projection store"),
            );
            let event_log = Arc::new(
                crate::EventLogWriter::new(&event_log_root)
                    .expect("inject restart event log writer"),
            );
            let runtime = RuntimeCore::default()
                .with_event_log_writer(event_log)
                .with_projection_store(projection);
            runtime
                .start_session(AgentSessionStartParams {
                    session_id: Some("session-inject-restart".to_string()),
                    thread_id: Some(thread_id.to_string()),
                    app_id: "test".to_string(),
                    workspace_id: None,
                    business_object_ref: None,
                    locale: None,
                })
                .expect("inject restart session");
            runtime
                .inject_thread_items(ThreadInjectItemsParams {
                    thread_id: thread_id.to_string(),
                    items: vec![first_item.clone()],
                })
                .await
                .expect("inject before restart");
        }

        let projection = Arc::new(
            crate::ProjectionStore::initialize(&database_path)
                .expect("reopen inject restart projection store"),
        );
        let event_log = Arc::new(
            crate::EventLogWriter::new(&event_log_root)
                .expect("reopen inject restart event log writer"),
        );
        let runtime = RuntimeCore::default()
            .with_event_log_writer(event_log)
            .with_projection_store(projection);
        assert_eq!(runtime.loaded_session_id_for_thread(thread_id), None);
        let second_item = json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "after restart"}]
        });
        runtime
            .inject_thread_items(ThreadInjectItemsParams {
                thread_id: thread_id.to_string(),
                items: vec![second_item.clone()],
            })
            .await
            .expect("inject into cold thread");

        let session_id = runtime
            .loaded_session_id_for_thread(thread_id)
            .expect("cold inject must hydrate the exact thread");
        let history = {
            let state = runtime
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .get(&session_id)
                .expect("hydrated injected session");
            super::super::provider_history::provider_history_excluding_current_turn_input(
                stored,
                None,
                "future-turn",
            )
            .expect("restarted provider history")
        };
        assert!(matches!(
            history.as_slice(),
            [first, second]
                if matches!(first.content.as_slice(), [CurrentProviderContent::RawResponseItem(value)] if value == &first_item)
                    && matches!(second.content.as_slice(), [CurrentProviderContent::RawResponseItem(value)] if value == &second_item)
        ));

        runtime
            .archive_thread(agent_protocol::ThreadId::new(thread_id))
            .await
            .expect("archive injected thread");
        assert_eq!(runtime.loaded_session_id_for_thread(thread_id), None);
        let error = runtime
            .inject_thread_items(ThreadInjectItemsParams {
                thread_id: thread_id.to_string(),
                items: vec![second_item],
            })
            .await
            .expect_err("archived inject must fail closed");
        assert!(
            matches!(error, RuntimeCoreError::InvalidRequest(message) if message.contains("cannot inject into archived thread"))
        );
        assert_eq!(runtime.loaded_session_id_for_thread(thread_id), None);
    }
}
