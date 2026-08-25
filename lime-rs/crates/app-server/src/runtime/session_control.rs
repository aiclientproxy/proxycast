use super::status::{agent_turn_blocks_queue_resume, agent_turn_is_active};
use super::*;
use agent_protocol::AgentInput as UserInput;
use agent_runtime::session_loop::{
    RuntimeSessionClosureTask, RuntimeSessionOperation, RuntimeSessionOperationResult,
    RuntimeSessionOperationSubmission, RuntimeSessionSubmitResult, RuntimeSessionTaskFailure,
    RuntimeSessionTaskKind, RuntimeSessionTaskOutcome,
};
use app_server_protocol::protocol::v2::{
    ThreadCompactStartParams, ThreadCompactStartResponse, METHOD_THREAD_COMPACT_START,
};
use app_server_protocol::*;
use serde_json::json;
use serde_json::Map;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::{oneshot, Mutex};

#[derive(Debug)]
pub(in crate::runtime) enum QueuedTurnResume {
    Empty,
    Blocked,
    Started {
        queued_turn_id: String,
        events: Vec<AgentEvent>,
    },
}

#[derive(Clone)]
pub(in crate::runtime) struct QueuedTurnResumeLease {
    index: usize,
    turn: AgentTurn,
    input: Vec<UserInput>,
    runtime_options: Option<RuntimeOptions>,
}

fn normalize_session_control_id(value: &str, message: &str) -> Result<String, RuntimeCoreError> {
    let normalized = value.trim();
    if normalized.is_empty() {
        Err(RuntimeCoreError::Backend(message.to_string()))
    } else {
        Ok(normalized.to_string())
    }
}

impl RuntimeCore {
    pub async fn compact_thread(
        &self,
        params: ThreadCompactStartParams,
    ) -> Result<RuntimeCoreOutput<ThreadCompactStartResponse>, RuntimeCoreError> {
        let thread_id = normalize_session_control_id(
            &params.thread_id,
            "threadId is required for thread/compact/start",
        )?;
        let session_id = self
            .loaded_session_id_for_thread(&thread_id)
            .ok_or_else(|| {
                RuntimeCoreError::InvalidRequest(format!("thread not found: {thread_id}"))
            })?;
        let runtime = self.clone();
        let actor_session_id = session_id.clone();
        let task = RuntimeSessionClosureTask::new(
            new_id("compact"),
            Vec::new(),
            move |context, _input, _cancel| {
                let runtime = runtime.clone();
                let session_id = session_id.clone();
                let turn_id = context.turn_id().to_string();
                Box::pin(async move {
                    let output = runtime
                        .compact_session_now(
                            &session_id,
                            &turn_id,
                            None,
                            METHOD_THREAD_COMPACT_START,
                            "manual",
                            None,
                        )
                        .await
                        .map_err(|error| RuntimeSessionTaskFailure {
                            message: error.to_string(),
                            reason_code: None,
                        })?;
                    for event in output.events {
                        runtime.event_hub.publish(event);
                    }
                    Ok(())
                })
            },
        )
        .with_kind(RuntimeSessionTaskKind::Compact);
        let session = self
            .session_loops
            .get_or_create(&actor_session_id, &thread_id)
            .await
            .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;
        let result = session
            .dispatch(RuntimeSessionOperationSubmission::new(
                RuntimeSessionOperation::Compact {
                    task: Arc::new(task),
                },
            ))
            .await
            .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;
        match result {
            RuntimeSessionOperationResult::Submission(submission)
                if matches!(submission.result, RuntimeSessionSubmitResult::Started) =>
            {
                Ok(RuntimeCoreOutput {
                    response: ThreadCompactStartResponse {},
                    events: Vec::new(),
                })
            }
            RuntimeSessionOperationResult::Submission(submission) => {
                Err(RuntimeCoreError::Backend(format!(
                    "compact operation was not started: {:?}",
                    submission.result
                )))
            }
            _ => Err(RuntimeCoreError::Backend(
                "compact operation returned an invalid receipt".to_string(),
            )),
        }
    }

    pub(in crate::runtime) async fn compact_session_with_trigger(
        &self,
        session_id: &str,
        event_name: Option<&str>,
        source: &str,
        trigger: &str,
        trigger_context: Option<Value>,
    ) -> Result<RuntimeCoreOutput<ThreadCompactStartResponse>, RuntimeCoreError> {
        self.ensure_current_session_hydrated(session_id).await?;
        let thread_id = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned")
            .sessions
            .get(session_id)
            .map(|stored| stored.session.thread_id.clone())
            .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.to_string()))?;
        let runtime = self.clone();
        let session_id = session_id.to_string();
        let event_name = event_name.map(str::to_string);
        let source = source.to_string();
        let trigger = trigger.to_string();
        let actor_session_id = session_id.clone();
        let (output_tx, output_rx) = oneshot::channel();
        let output_tx = Arc::new(Mutex::new(Some(output_tx)));
        let task = RuntimeSessionClosureTask::new(
            new_id("compact"),
            Vec::new(),
            move |context, _input, _cancel| {
                let runtime = runtime.clone();
                let session_id = session_id.clone();
                let turn_id = context.turn_id().to_string();
                let event_name = event_name.clone();
                let source = source.clone();
                let trigger = trigger.clone();
                let trigger_context = trigger_context.clone();
                let output_tx = Arc::clone(&output_tx);
                Box::pin(async move {
                    let output = runtime
                        .compact_session_now(
                            &session_id,
                            &turn_id,
                            event_name.as_deref(),
                            &source,
                            &trigger,
                            trigger_context,
                        )
                        .await;
                    match output {
                        Ok(output) => {
                            if let Some(sender) = output_tx.lock().await.take() {
                                let _ = sender.send(Ok(output));
                            }
                            Ok(())
                        }
                        Err(error) => {
                            let message = error.to_string();
                            if let Some(sender) = output_tx.lock().await.take() {
                                let _ = sender.send(Err(message.clone()));
                            }
                            Err(RuntimeSessionTaskFailure {
                                message,
                                reason_code: None,
                            })
                        }
                    }
                })
            },
        )
        .with_kind(RuntimeSessionTaskKind::Compact);
        let session = self
            .session_loops
            .get_or_create(&actor_session_id, &thread_id)
            .await
            .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;
        let result = session
            .dispatch(RuntimeSessionOperationSubmission::new(
                RuntimeSessionOperation::Compact {
                    task: Arc::new(task),
                },
            ))
            .await
            .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;
        let RuntimeSessionOperationResult::Submission(submission) = result else {
            return Err(RuntimeCoreError::Backend(
                "compact operation returned an invalid receipt".to_string(),
            ));
        };
        match submission.completion.await.map_err(|_| {
            RuntimeCoreError::Backend("compact operation completion channel closed".to_string())
        })? {
            Ok(RuntimeSessionTaskOutcome::Completed) => {}
            Ok(outcome) => {
                return Err(RuntimeCoreError::Backend(format!(
                    "compact operation ended without completion: {outcome:?}"
                )));
            }
            Err(error) => return Err(RuntimeCoreError::Backend(error.message)),
        }
        output_rx
            .await
            .map_err(|_| RuntimeCoreError::Backend("compact output channel closed".to_string()))?
            .map_err(RuntimeCoreError::Backend)
    }

    async fn compact_session_now(
        &self,
        session_id: &str,
        turn_id: &str,
        event_name: Option<&str>,
        source: &str,
        trigger: &str,
        trigger_context: Option<Value>,
    ) -> Result<RuntimeCoreOutput<ThreadCompactStartResponse>, RuntimeCoreError> {
        self.ensure_current_session_hydrated(session_id).await?;
        let (_, turns) = self.session_snapshot(session_id)?;
        let active_turn_ids = turns
            .iter()
            .filter(|turn| agent_turn_is_active(turn.status))
            .map(|turn| turn.turn_id.clone())
            .collect::<Vec<_>>();
        for turn_id in active_turn_ids {
            self.cancel_turn(
                AgentSessionTurnCancelParams {
                    session_id: session_id.to_string(),
                    turn_id,
                },
                RuntimeHostContext::default(),
            )
            .await?;
        }
        let (session, turns) = self.session_snapshot(session_id)?;
        let existing_events = self.events_for_session(session_id)?;
        let compaction = super::context_compaction::build_session_context_compaction(
            &session,
            &turns,
            &existing_events,
            self.sidecar_store.as_deref(),
        )?;
        self.create_compaction_turn(&session.session_id, &session.thread_id, turn_id)?;
        let event_name = event_name
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(METHOD_THREAD_COMPACT_START);
        let mut completed_payload = Map::new();
        completed_payload.insert("source".to_string(), json!(source));
        completed_payload.insert("eventName".to_string(), json!(event_name));
        completed_payload.insert("compactionId".to_string(), json!(compaction.compaction_id));
        completed_payload.insert("contextEpoch".to_string(), json!(compaction.context_epoch));
        completed_payload.insert(
            "tailStartTurnId".to_string(),
            json!(compaction.tail_start_turn_id),
        );
        completed_payload.insert(
            "replacementHistory".to_string(),
            json!(compaction.window.replacement_history),
        );
        completed_payload.insert(
            "windowNumber".to_string(),
            json!(compaction.window.window_number),
        );
        completed_payload.insert(
            "firstWindowId".to_string(),
            json!(compaction.window.first_window_id),
        );
        completed_payload.insert(
            "previousWindowId".to_string(),
            json!(compaction.window.previous_window_id),
        );
        completed_payload.insert("windowId".to_string(), json!(compaction.window.window_id));
        completed_payload.insert("turnCount".to_string(), json!(turns.len()));
        completed_payload.insert("trigger".to_string(), json!(trigger));
        if let Some(trigger_context) = trigger_context.clone() {
            completed_payload.insert("triggerContext".to_string(), trigger_context);
        }
        completed_payload.insert("summary".to_string(), json!(compaction.summary));
        completed_payload.insert("artifact".to_string(), json!(compaction.artifact));
        completed_payload.insert(
            output_refs::SIDECAR_REF_FIELD.to_string(),
            json!(compaction.sidecar_ref),
        );

        let events = self.append_runtime_events(
            &session.session_id,
            &session.thread_id,
            Some(turn_id),
            vec![
                RuntimeEvent::new("turn.accepted", json!({"source": source})),
                RuntimeEvent::new("turn.started", json!({"source": source})),
                RuntimeEvent::new(
                    "context.compaction.started",
                    json!({
                        "source": source,
                        "eventName": event_name,
                        "compactionId": compaction.compaction_id,
                        "contextEpoch": compaction.context_epoch,
                        "tailStartTurnId": compaction.tail_start_turn_id,
                        "replacementHistory": compaction.window.replacement_history,
                        "windowNumber": compaction.window.window_number,
                        "firstWindowId": compaction.window.first_window_id,
                        "previousWindowId": compaction.window.previous_window_id,
                        "windowId": compaction.window.window_id,
                        "turnCount": turns.len(),
                        "trigger": trigger,
                        "triggerContext": trigger_context,
                    }),
                ),
                RuntimeEvent::new(
                    "context.compaction.completed",
                    Value::Object(completed_payload),
                ),
                RuntimeEvent::new("turn.completed", json!({"source": source})),
            ],
        )?;
        Ok(RuntimeCoreOutput {
            response: ThreadCompactStartResponse {},
            events,
        })
    }

    fn create_compaction_turn(
        &self,
        session_id: &str,
        thread_id: &str,
        turn_id: &str,
    ) -> Result<(), RuntimeCoreError> {
        let mut state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        let stored = state
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.to_string()))?;
        if stored.session.thread_id != thread_id {
            return Err(RuntimeCoreError::InvalidRequest(
                "session/thread identity mismatch for context compaction".to_string(),
            ));
        }
        if stored
            .turns
            .iter()
            .any(|turn| agent_turn_is_active(turn.status))
        {
            return Err(RuntimeCoreError::InvalidRequest(
                "session actor and runtime turn state disagree for context compaction".to_string(),
            ));
        }
        let now = super::timestamp();
        stored.session.status = AgentSessionStatus::Running;
        stored.session.updated_at = now.clone();
        stored.turns.push(AgentTurn {
            turn_id: turn_id.to_string(),
            session_id: session_id.to_string(),
            thread_id: thread_id.to_string(),
            status: AgentTurnStatus::Accepted,
            started_at: Some(now),
            completed_at: None,
        });
        Ok(())
    }

    #[cfg(test)]
    pub(in crate::runtime) async fn resume_next_queued_turn_if_idle(
        &self,
        session_id: &str,
        host: RuntimeHostContext,
    ) -> Result<QueuedTurnResume, RuntimeCoreError> {
        self.resume_queued_turn_if_idle_selected(session_id, None, host)
            .await
    }

    pub(in crate::runtime) async fn resume_queued_turn_if_idle_selected(
        &self,
        session_id: &str,
        queued_submission_id: Option<&str>,
        host: RuntimeHostContext,
    ) -> Result<QueuedTurnResume, RuntimeCoreError> {
        self.resume_queued_turn_if_idle_selected_with_mode(
            session_id,
            queued_submission_id,
            host,
            true,
        )
        .await
    }

    pub(in crate::runtime) async fn resume_next_queued_turn_for_recovery(
        &self,
        session_id: &str,
        host: RuntimeHostContext,
    ) -> Result<QueuedTurnResume, RuntimeCoreError> {
        self.resume_queued_turn_if_idle_selected_with_mode(session_id, None, host, false)
            .await
    }

    async fn resume_queued_turn_if_idle_selected_with_mode(
        &self,
        session_id: &str,
        queued_submission_id: Option<&str>,
        host: RuntimeHostContext,
        return_after_admission: bool,
    ) -> Result<QueuedTurnResume, RuntimeCoreError> {
        self.ensure_current_session_hydrated(session_id).await?;
        let queued = {
            let mut state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .get_mut(session_id)
                .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.to_string()))?;
            if stored
                .turns
                .iter()
                .any(|turn| agent_turn_blocks_queue_resume(turn.status))
            {
                return Ok(QueuedTurnResume::Blocked);
            }
            let index = match queued_submission_id {
                Some(queued_submission_id) => stored.turns.iter().position(|turn| {
                    turn.turn_id == queued_submission_id
                        && matches!(turn.status, AgentTurnStatus::Queued)
                }),
                None => stored
                    .turns
                    .iter()
                    .position(|turn| matches!(turn.status, AgentTurnStatus::Queued)),
            };
            let Some(index) = index else {
                if let Some(queued_submission_id) = queued_submission_id {
                    return Err(RuntimeCoreError::InvalidRequest(format!(
                        "queued submission not found: {queued_submission_id}"
                    )));
                }
                return Ok(QueuedTurnResume::Empty);
            };
            let turn = stored.turns.remove(index);
            let input = stored.turn_inputs.remove(&turn.turn_id).unwrap_or_default();
            let runtime_options = stored.turn_runtime_options.remove(&turn.turn_id);
            QueuedTurnResumeLease {
                index,
                turn,
                input,
                runtime_options,
            }
        };
        let queued_turn_id = queued.turn.turn_id.clone();
        let mut resumed_runtime_options = queued.runtime_options.clone().unwrap_or_default();
        resumed_runtime_options.queued_turn_id = Some(queued_turn_id.clone());
        let request = TurnStartRequest {
            session_id: session_id.to_string(),
            turn_id: Some(queued_turn_id.clone()),
            input: queued.input.clone(),
            runtime_options: Some(resumed_runtime_options),
            queue_if_busy: false,
            skip_pre_submit_resume: true,
        };
        let start_result = if return_after_admission {
            self.start_queued_turn_inner(
                request,
                host,
                None,
                false,
                true,
                super::turn_start::TurnStartInputKind::QueuedUser,
                queued.clone(),
            )
            .await
        } else {
            let event_hub = self.event_hub.clone();
            let mut publish_event = move |event| {
                event_hub.publish(event);
                Ok(())
            };
            self.start_queued_turn_inner(
                request,
                host,
                Some(&mut publish_event),
                false,
                false,
                super::turn_start::TurnStartInputKind::QueuedUser,
                queued.clone(),
            )
            .await
        };
        let output = match start_result {
            Ok(output) => output,
            Err(error) => {
                self.restore_queued_turn_if_missing(
                    session_id,
                    queued.index,
                    queued.turn,
                    queued.input,
                    queued.runtime_options,
                );
                return Err(error);
            }
        };
        Ok(QueuedTurnResume::Started {
            queued_turn_id,
            events: output.events,
        })
    }

    pub(in crate::runtime) fn restore_queued_turn_lease(
        &self,
        session_id: &str,
        queued: QueuedTurnResumeLease,
    ) {
        self.restore_queued_turn_if_missing(
            session_id,
            queued.index,
            queued.turn,
            queued.input,
            queued.runtime_options,
        );
    }

    pub async fn list_agent_session_file_checkpoints(
        &self,
        params: AgentSessionFileCheckpointListParams,
    ) -> Result<AgentSessionFileCheckpointListResponse, RuntimeCoreError> {
        let detail = self
            .read_current_detail_for_file_checkpoint(&params.session_id)
            .await?;
        crate::file_checkpoint::list_file_checkpoints(&detail).map_err(RuntimeCoreError::Backend)
    }

    pub async fn get_agent_session_file_checkpoint(
        &self,
        params: AgentSessionFileCheckpointGetParams,
    ) -> Result<AgentSessionFileCheckpointDetail, RuntimeCoreError> {
        let detail = self
            .read_current_detail_for_file_checkpoint(&params.session_id)
            .await?;
        let workspace_root = crate::file_checkpoint::resolve_workspace_root(&detail)
            .map_err(RuntimeCoreError::Backend)?;
        crate::file_checkpoint::get_file_checkpoint(
            &detail,
            workspace_root.as_path(),
            self.file_checkpoint_snapshot_store.as_ref(),
            &params.checkpoint_id,
        )
        .map_err(RuntimeCoreError::Backend)
    }

    pub async fn diff_agent_session_file_checkpoint(
        &self,
        params: AgentSessionFileCheckpointDiffParams,
    ) -> Result<AgentSessionFileCheckpointDiffResponse, RuntimeCoreError> {
        let detail = self
            .read_current_detail_for_file_checkpoint(&params.session_id)
            .await?;
        crate::file_checkpoint::diff_file_checkpoint(&detail, &params.checkpoint_id)
            .map_err(RuntimeCoreError::Backend)
    }

    pub async fn restore_agent_session_file_checkpoint(
        &self,
        params: AgentSessionFileCheckpointRestoreParams,
    ) -> Result<AgentSessionFileCheckpointRestoreResponse, RuntimeCoreError> {
        let detail = self
            .read_current_detail_for_file_checkpoint(&params.session_id)
            .await?;
        let workspace_root = crate::file_checkpoint::resolve_workspace_root(&detail)
            .map_err(RuntimeCoreError::Backend)?;
        crate::file_checkpoint::restore_file_checkpoint(
            &detail,
            workspace_root.as_path(),
            self.file_checkpoint_snapshot_store.as_ref(),
            &params.checkpoint_id,
            params.confirm_restore,
            params.create_backup,
        )
        .map_err(RuntimeCoreError::Backend)
    }

    async fn read_current_detail_for_file_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<serde_json::Value, RuntimeCoreError> {
        let normalized_session_id = session_id.trim();
        if normalized_session_id.is_empty() {
            return Err(RuntimeCoreError::Backend(
                "sessionId is required for agentSession/fileCheckpoint".to_string(),
            ));
        }
        let response = self
            .read_session_current(AgentSessionReadParams {
                session_id: normalized_session_id.to_string(),
                history_limit: Some(1_000),
                history_offset: None,
                history_before_message_id: None,
            })
            .await?;
        response.detail.ok_or_else(|| {
            RuntimeCoreError::Backend(
                "agentSession/fileCheckpoint requires current session detail".to_string(),
            )
        })
    }
}
