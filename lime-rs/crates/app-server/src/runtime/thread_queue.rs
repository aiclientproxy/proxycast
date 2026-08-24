use super::session_control::QueuedTurnResume;
use super::turn_start::{user_input_text, validate_user_input};
use super::*;
use agent_protocol::AgentInput;
use app_server_protocol::{AgentTurn, AgentTurnStatus, RuntimeOptions, RuntimeRequest};
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

const MAX_QUEUE_SUBMISSIONS: usize = 100;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ThreadQueuedSubmission {
    pub id: String,
    pub input: Vec<AgentInput>,
    pub client_user_message_id: String,
}

impl RuntimeCore {
    pub(crate) async fn add_thread_queue_submission(
        &self,
        thread_id: &str,
        input: Vec<AgentInput>,
        client_user_message_id: String,
    ) -> Result<ThreadQueuedSubmission, RuntimeCoreError> {
        validate_queue_input(&input)?;
        let client_user_message_id = required_value(
            client_user_message_id,
            "thread/queue/add requires clientUserMessageId",
        )?;
        let session_id = self.ensure_queue_thread_hydrated(thread_id).await?;
        self.ensure_queue_capacity(&session_id)?;
        let input = super::input_media::prepare_runtime_input(
            input,
            self.sidecar_store.as_deref(),
            &session_id,
        )
        .map_err(RuntimeCoreError::InvalidRequest)?
        .durable;
        let queued = {
            let mut state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .get_mut(&session_id)
                .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.clone()))?;
            ensure_stored_queue_capacity(stored)?;

            let id = Uuid::new_v4().to_string();
            let turn = AgentTurn {
                turn_id: id.clone(),
                session_id: stored.session.session_id.clone(),
                thread_id: stored.session.thread_id.clone(),
                status: AgentTurnStatus::Queued,
                started_at: Some(timestamp()),
                completed_at: None,
            };
            stored.turn_inputs.insert(id.clone(), input.clone());
            stored.turn_runtime_options.insert(
                id.clone(),
                runtime_options_with_client_id(&client_user_message_id),
            );
            stored.turns.push(turn);
            ThreadQueuedSubmission {
                id,
                input,
                client_user_message_id,
            }
        };

        let added_event = {
            let state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .get(&session_id)
                .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.clone()))?;
            queue_added_event(stored, &queued.id, "thread/queue/add")?
        };
        if let Err(error) =
            self.append_runtime_events(&session_id, thread_id, Some(&queued.id), vec![added_event])
        {
            self.rollback_queue_add(&session_id, &queued.id);
            return Err(error);
        }
        Ok(queued)
    }

    pub(crate) async fn list_thread_queue_submissions(
        &self,
        thread_id: &str,
    ) -> Result<Vec<ThreadQueuedSubmission>, RuntimeCoreError> {
        let session_id = self.ensure_queue_thread_hydrated(thread_id).await?;
        let state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        let stored = state
            .sessions
            .get(&session_id)
            .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id))?;
        Ok(queued_submissions(stored))
    }

    pub(crate) async fn update_thread_queue_submission(
        &self,
        thread_id: &str,
        queued_submission_id: &str,
        input: Vec<AgentInput>,
    ) -> Result<ThreadQueuedSubmission, RuntimeCoreError> {
        validate_queue_input(&input)?;
        let queued_submission_id = required_value(
            queued_submission_id.to_string(),
            "thread/queue/update requires queuedSubmissionId",
        )?;
        let session_id = self.ensure_queue_thread_hydrated(thread_id).await?;
        {
            let state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .get(&session_id)
                .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.clone()))?;
            require_queued_turn(stored, &queued_submission_id)?;
        }
        let input = super::input_media::prepare_runtime_input(
            input,
            self.sidecar_store.as_deref(),
            &session_id,
        )
        .map_err(RuntimeCoreError::InvalidRequest)?
        .durable;
        let (previous_input, client_user_message_id) = {
            let mut state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .get_mut(&session_id)
                .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.clone()))?;
            require_queued_turn(stored, &queued_submission_id)?;
            let previous = stored
                .turn_inputs
                .insert(queued_submission_id.clone(), input.clone())
                .unwrap_or_default();
            let client_id = queued_client_id(stored, &queued_submission_id);
            (previous, client_id)
        };
        let event = RuntimeEvent::new(
            "queue.updated",
            json!({
                "queuedTurnId": queued_submission_id,
                "queuedSubmissionId": queued_submission_id,
                "clientId": client_user_message_id,
                "input": input,
                "content": {"kind": "inline_text", "text": user_input_text(&input)},
            }),
        );
        if let Err(error) = self.append_runtime_events(
            &session_id,
            thread_id,
            Some(&queued_submission_id),
            vec![event],
        ) {
            let mut state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            if let Some(stored) = state.sessions.get_mut(&session_id) {
                stored
                    .turn_inputs
                    .insert(queued_submission_id.clone(), previous_input);
            }
            return Err(error);
        }
        Ok(ThreadQueuedSubmission {
            id: queued_submission_id,
            input,
            client_user_message_id,
        })
    }

    pub(crate) async fn delete_thread_queue_submission(
        &self,
        thread_id: &str,
        queued_submission_id: &str,
    ) -> Result<bool, RuntimeCoreError> {
        let queued_submission_id = required_value(
            queued_submission_id.to_string(),
            "thread/queue/delete requires queuedSubmissionId",
        )?;
        let session_id = self.ensure_queue_thread_hydrated(thread_id).await?;
        let removed = {
            let mut state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .get_mut(&session_id)
                .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.clone()))?;
            let Some(index) = stored.turns.iter().position(|turn| {
                turn.turn_id == queued_submission_id && turn.status == AgentTurnStatus::Queued
            }) else {
                return Ok(false);
            };
            let turn = stored.turns.remove(index);
            let input = stored.turn_inputs.remove(&queued_submission_id);
            let runtime_options = stored.turn_runtime_options.remove(&queued_submission_id);
            (index, turn, input, runtime_options)
        };
        if let Err(error) = self.append_runtime_events(
            &session_id,
            thread_id,
            None,
            vec![RuntimeEvent::new(
                "queue.removed",
                json!({
                    "source": "thread/queue/delete",
                    "queuedTurnId": queued_submission_id,
                    "queuedSubmissionId": queued_submission_id,
                }),
            )],
        ) {
            let mut state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            if let Some(stored) = state.sessions.get_mut(&session_id) {
                let index = removed.0.min(stored.turns.len());
                stored.turns.insert(index, removed.1);
                if let Some(input) = removed.2 {
                    stored
                        .turn_inputs
                        .insert(queued_submission_id.clone(), input);
                }
                if let Some(options) = removed.3 {
                    stored
                        .turn_runtime_options
                        .insert(queued_submission_id.clone(), options);
                }
            }
            return Err(error);
        }
        Ok(true)
    }

    pub(crate) async fn reorder_thread_queue_submissions(
        &self,
        thread_id: &str,
        queued_submission_ids: Vec<String>,
    ) -> Result<(), RuntimeCoreError> {
        let session_id = self.ensure_queue_thread_hydrated(thread_id).await?;
        let (previous_turns, added_events) = {
            let mut state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .get_mut(&session_id)
                .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.clone()))?;
            validate_reorder(stored, &queued_submission_ids)?;
            let added_events = queued_submission_ids
                .iter()
                .map(|id| queue_added_event(stored, id, "thread/queue/reorder"))
                .collect::<Result<Vec<_>, _>>()?;
            let previous = stored.turns.clone();
            reorder_queued_turns(stored, &queued_submission_ids);
            (previous, added_events)
        };

        let mut events = queued_submission_ids
            .iter()
            .map(|id| {
                RuntimeEvent::new(
                    "queue.removed",
                    json!({"queuedTurnId": id, "queuedSubmissionId": id}),
                )
            })
            .collect::<Vec<_>>();
        events.extend(added_events);
        if let Err(error) = self.append_runtime_events(&session_id, thread_id, None, events) {
            let mut state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            if let Some(stored) = state.sessions.get_mut(&session_id) {
                stored.turns = previous_turns;
            }
            return Err(error);
        }
        Ok(())
    }

    pub(crate) async fn start_thread_queue_submission(
        &self,
        thread_id: &str,
        queued_submission_id: Option<&str>,
        host: RuntimeHostContext,
    ) -> Result<AgentTurn, RuntimeCoreError> {
        let session_id = self
            .loaded_session_id_for_thread(thread_id)
            .ok_or_else(|| {
                RuntimeCoreError::InvalidRequest(
                    "resume the thread before starting a queued message".to_string(),
                )
            })?;
        match self
            .resume_queued_turn_if_idle_selected(&session_id, queued_submission_id, host)
            .await?
        {
            QueuedTurnResume::Started { queued_turn_id, .. } => self
                .state
                .lock()
                .expect("runtime core state mutex poisoned")
                .sessions
                .get(&session_id)
                .and_then(|stored| {
                    stored
                        .turns
                        .iter()
                        .find(|turn| turn.turn_id == queued_turn_id)
                })
                .cloned()
                .ok_or_else(|| {
                    RuntimeCoreError::Backend(
                        "started queued turn is missing from RuntimeCore state".to_string(),
                    )
                }),
            QueuedTurnResume::Blocked => Err(RuntimeCoreError::InvalidRequest(
                "thread already has an active or pending turn".to_string(),
            )),
            QueuedTurnResume::Empty => Err(RuntimeCoreError::InvalidRequest(
                "queued submission not found".to_string(),
            )),
        }
    }

    pub(crate) fn wake_thread_queue_if_idle(&self, thread_id: &str, host: RuntimeHostContext) {
        if let Some(session_id) = self.loaded_session_id_for_thread(thread_id) {
            if self.thread_queue_is_paused_after_interruption(&session_id) {
                return;
            }
            self.wake_pending_session_work(session_id, host, None);
        }
    }

    fn thread_queue_is_paused_after_interruption(&self, session_id: &str) -> bool {
        self.state
            .lock()
            .expect("runtime core state mutex poisoned")
            .sessions
            .get(session_id)
            .and_then(|stored| {
                stored
                    .turns
                    .iter()
                    .rev()
                    .find(|turn| turn.status != AgentTurnStatus::Queued)
            })
            .is_some_and(|turn| turn.status == AgentTurnStatus::Canceled)
    }

    async fn ensure_queue_thread_hydrated(
        &self,
        thread_id: &str,
    ) -> Result<String, RuntimeCoreError> {
        let thread_id = required_value(thread_id.to_string(), "thread queue requires threadId")?;
        let thread = self
            .read_thread(app_server_protocol::ThreadReadParams {
                thread_id: agent_protocol::ThreadId::new(thread_id.clone()),
                turns_view: agent_protocol::ThreadTurnsView::NotLoaded,
            })
            .await?
            .thread;
        if thread.archived {
            return Err(RuntimeCoreError::InvalidRequest(format!(
                "archived thread does not support queued submissions: {thread_id}"
            )));
        }
        if thread
            .metadata
            .get("ephemeral")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Err(RuntimeCoreError::InvalidRequest(format!(
                "ephemeral thread does not support queued submissions: {thread_id}"
            )));
        }
        let session_id = thread.session_id.to_string();
        self.ensure_current_session_hydrated(&session_id).await?;
        Ok(session_id)
    }

    fn rollback_queue_add(&self, session_id: &str, queued_submission_id: &str) {
        let mut state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        let Some(stored) = state.sessions.get_mut(session_id) else {
            return;
        };
        stored
            .turns
            .retain(|turn| turn.turn_id != queued_submission_id);
        stored.turn_inputs.remove(queued_submission_id);
        stored.turn_runtime_options.remove(queued_submission_id);
    }

    fn ensure_queue_capacity(&self, session_id: &str) -> Result<(), RuntimeCoreError> {
        let state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        let stored = state
            .sessions
            .get(session_id)
            .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.to_string()))?;
        ensure_stored_queue_capacity(stored)
    }
}

fn ensure_stored_queue_capacity(stored: &StoredSession) -> Result<(), RuntimeCoreError> {
    if queued_submissions(stored).len() >= MAX_QUEUE_SUBMISSIONS {
        return Err(RuntimeCoreError::InvalidRequest(format!(
            "queue cannot contain more than {MAX_QUEUE_SUBMISSIONS} submissions"
        )));
    }
    Ok(())
}

fn queue_added_event(
    stored: &StoredSession,
    queued_submission_id: &str,
    source: &str,
) -> Result<RuntimeEvent, RuntimeCoreError> {
    require_queued_turn(stored, queued_submission_id)?;
    let input = stored
        .turn_inputs
        .get(queued_submission_id)
        .cloned()
        .ok_or_else(|| {
            RuntimeCoreError::Backend(format!(
                "queued submission input is missing: {queued_submission_id}"
            ))
        })?;
    let client_user_message_id = queued_client_id(stored, queued_submission_id);
    let intent = super::queued_turn_intent::snapshot_value(
        stored.turn_runtime_options.get(queued_submission_id),
    )
    .map_err(RuntimeCoreError::Backend)?;
    Ok(RuntimeEvent::new(
        "queue.added",
        json!({
            "source": source,
            "queuedTurnId": queued_submission_id,
            "queuedSubmissionId": queued_submission_id,
            "clientId": client_user_message_id,
            "input": input,
            "content": {
                "kind": "inline_text",
                "text": user_input_text(&input),
            },
            "queuedTurnIntent": intent,
        }),
    ))
}

fn queued_submissions(stored: &StoredSession) -> Vec<ThreadQueuedSubmission> {
    stored
        .turns
        .iter()
        .filter(|turn| turn.status == AgentTurnStatus::Queued)
        .map(|turn| ThreadQueuedSubmission {
            id: turn.turn_id.clone(),
            input: stored
                .turn_inputs
                .get(&turn.turn_id)
                .cloned()
                .unwrap_or_default(),
            client_user_message_id: queued_client_id(stored, &turn.turn_id),
        })
        .collect()
}

fn queued_client_id(stored: &StoredSession, queued_submission_id: &str) -> String {
    stored
        .turn_runtime_options
        .get(queued_submission_id)
        .and_then(RuntimeOptions::runtime_metadata)
        .and_then(|metadata| metadata.get("clientUserMessageId"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| queued_submission_id.to_string())
}

fn runtime_options_with_client_id(client_user_message_id: &str) -> RuntimeOptions {
    let mut metadata = Map::new();
    metadata.insert(
        "clientUserMessageId".to_string(),
        Value::String(client_user_message_id.to_string()),
    );
    RuntimeOptions {
        runtime_request: Some(RuntimeRequest {
            metadata: Some(Value::Object(metadata)),
            ..RuntimeRequest::default()
        }),
        ..RuntimeOptions::default()
    }
}

fn require_queued_turn(
    stored: &StoredSession,
    queued_submission_id: &str,
) -> Result<(), RuntimeCoreError> {
    stored
        .turns
        .iter()
        .any(|turn| turn.turn_id == queued_submission_id && turn.status == AgentTurnStatus::Queued)
        .then_some(())
        .ok_or_else(|| {
            RuntimeCoreError::InvalidRequest(format!(
                "queued submission not found: {queued_submission_id}"
            ))
        })
}

fn validate_reorder(
    stored: &StoredSession,
    queued_submission_ids: &[String],
) -> Result<(), RuntimeCoreError> {
    let current = stored
        .turns
        .iter()
        .filter(|turn| turn.status == AgentTurnStatus::Queued)
        .map(|turn| turn.turn_id.as_str())
        .collect::<HashSet<_>>();
    let requested = queued_submission_ids
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    if requested.len() != queued_submission_ids.len() || requested != current {
        return Err(RuntimeCoreError::InvalidRequest(
            "queuedSubmissionIds must contain every queued submission exactly once".to_string(),
        ));
    }
    Ok(())
}

fn reorder_queued_turns(stored: &mut StoredSession, queued_submission_ids: &[String]) {
    let queued_by_id = stored
        .turns
        .iter()
        .filter(|turn| turn.status == AgentTurnStatus::Queued)
        .cloned()
        .map(|turn| (turn.turn_id.clone(), turn))
        .collect::<HashMap<_, _>>();
    let mut reordered = queued_submission_ids
        .iter()
        .filter_map(|id| queued_by_id.get(id).cloned());
    for turn in &mut stored.turns {
        if turn.status == AgentTurnStatus::Queued {
            *turn = reordered
                .next()
                .expect("validated queue reorder must preserve cardinality");
        }
    }
}

fn validate_queue_input(input: &[AgentInput]) -> Result<(), RuntimeCoreError> {
    validate_user_input(input).map_err(RuntimeCoreError::InvalidRequest)
}

fn required_value(value: String, message: &str) -> Result<String, RuntimeCoreError> {
    let value = value.trim();
    if value.is_empty() {
        Err(RuntimeCoreError::InvalidRequest(message.to_string()))
    } else {
        Ok(value.to_string())
    }
}
