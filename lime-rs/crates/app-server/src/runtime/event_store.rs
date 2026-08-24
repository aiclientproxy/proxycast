mod canonical_message_lifecycle;
mod canonical_notifications;
mod validation;

pub(in crate::runtime) use self::canonical_message_lifecycle::CanonicalMessageLifecycleState;
use self::canonical_message_lifecycle::{
    attach_canonical_item_entity, with_canonical_message_reasoning_lifecycle,
};
use self::canonical_notifications::notification_events_with_canonical_entities;
use self::validation::EventValidationContext;
use super::status::{agent_turn_is_terminal, session_status_from_turn_status};
use super::trace;
use super::trace_store::TraceEventWriter;
use super::turn_input_events;
use super::*;
use crate::agent_ui_event_schema;
use crate::runtime_backend::tool_process_metadata::SoulStyleMetadata;
use crate::runtime_backend::{
    current_agent_runtime_config_metadata, tool_process_external_metadata,
};
use app_server_protocol::*;
use serde_json::Value;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TerminalTurnAppendPolicy {
    Reject,
    Allow,
}

impl RuntimeCoreEventAppender {
    pub fn append_external_runtime_events(
        &self,
        session_id: &str,
        turn_id: Option<&str>,
        runtime_events: Vec<RuntimeEvent>,
    ) -> Result<Vec<AgentEvent>, RuntimeCoreError> {
        let mut state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        let stored = state
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.to_string()))?;
        if let Some(turn_id) = turn_id {
            if !stored.turns.iter().any(|turn| turn.turn_id == turn_id) {
                return Err(RuntimeCoreError::TurnNotActive(turn_id.to_string()));
            }
        }
        let thread_id = stored.session.thread_id.clone();

        append_runtime_events_to_stored_session(
            stored,
            self.file_checkpoint_snapshot_store.as_ref(),
            self.output_snapshot_store.as_ref(),
            self.sidecar_store.as_deref(),
            self.event_log_writer.as_deref(),
            self.trace_event_writer.as_deref(),
            self.projection_store.as_deref(),
            Some(&self.session_loops),
            session_id,
            &thread_id,
            turn_id,
            runtime_events,
            None,
            TerminalTurnAppendPolicy::Reject,
        )
    }

    pub(in crate::runtime) fn append_artifact_snapshot(
        &self,
        session_id: &str,
        turn_id: Option<&str>,
        payload: Value,
    ) -> Result<Vec<AgentEvent>, RuntimeCoreError> {
        let mut state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        let stored = state
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.to_string()))?;
        if let Some(turn_id) = turn_id {
            if !stored.turns.iter().any(|turn| turn.turn_id == turn_id) {
                return Err(RuntimeCoreError::TurnNotActive(turn_id.to_string()));
            }
        }
        let thread_id = stored.session.thread_id.clone();

        append_runtime_events_to_stored_session(
            stored,
            self.file_checkpoint_snapshot_store.as_ref(),
            self.output_snapshot_store.as_ref(),
            self.sidecar_store.as_deref(),
            self.event_log_writer.as_deref(),
            self.trace_event_writer.as_deref(),
            self.projection_store.as_deref(),
            Some(&self.session_loops),
            session_id,
            &thread_id,
            turn_id,
            vec![RuntimeEvent::new("artifact.snapshot", payload)],
            None,
            TerminalTurnAppendPolicy::Allow,
        )
    }
}

impl RuntimeCore {
    /// Replays an EventLog-first tail after a downstream canonical projection failure.
    ///
    /// This is intentionally narrower than normal append: callers have already proved the exact
    /// events are durable in the canonical JSONL log, so replay must never write that log again.
    pub(in crate::runtime) fn replay_durable_runtime_events(
        &self,
        session_id: &str,
        durable_events: Vec<AgentEvent>,
    ) -> Result<Vec<AgentEvent>, RuntimeCoreError> {
        if durable_events.is_empty() {
            return Ok(Vec::new());
        }
        let mut state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        let stored = state
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.to_string()))?;

        let mut replay = Vec::new();
        for event in durable_events {
            if event.session_id != session_id {
                return Err(RuntimeCoreError::Backend(format!(
                    "durable event replay scope mismatch for session {session_id}"
                )));
            }
            if let Some(existing) = stored.events.iter().find(|existing| {
                existing.event_id == event.event_id || existing.sequence == event.sequence
            }) {
                if existing != &event {
                    return Err(RuntimeCoreError::Backend(format!(
                        "durable event replay identity conflict for session {session_id}"
                    )));
                }
                continue;
            }
            replay.push(event);
        }
        if replay.is_empty() {
            return Ok(Vec::new());
        }
        replay.sort_by_key(|event| event.sequence);
        let next_sequence = stored
            .events
            .last()
            .map(|event| event.sequence.saturating_add(1))
            .unwrap_or(1);
        if replay
            .iter()
            .enumerate()
            .any(|(index, event)| event.sequence != next_sequence + index as u64)
        {
            return Err(RuntimeCoreError::Backend(format!(
                "durable event replay is not the contiguous EventLog tail for session {session_id}"
            )));
        }

        let notifications = notification_events_with_canonical_entities(stored, &replay)?;
        let projection_store = self.projection_store.as_deref().ok_or_else(|| {
            RuntimeCoreError::Backend(
                "durable event replay requires a ProjectionStore for canonical recovery"
                    .to_string(),
            )
        })?;
        projection_store
            .apply_canonical_events(stored, &replay)
            .map_err(|error| {
                RuntimeCoreError::Backend(format!(
                    "canonical mailbox Item recovery failed after EventLog append: {error}"
                ))
            })?;
        let terminal_activities = projection_store
            .append_terminal_agent_results_sync(
                &agent_protocol::ThreadId::new(stored.session.thread_id.clone()),
                &replay,
            )
            .map_err(|error| {
                RuntimeCoreError::Backend(format!(
                    "durable child terminal activity recovery failed: {error}"
                ))
            })?;
        super::agent_terminal_activity::publish_terminal_agent_activities(
            &self.session_loops,
            terminal_activities,
        );
        if let Err(error) = projection_store.apply_events(&replay) {
            tracing::warn!(
                "[projection-store] failed to replay {} durable events for session {}: {}",
                replay.len(),
                session_id,
                error
            );
        }
        for event in replay {
            apply_runtime_event_state_transition(
                stored,
                event.turn_id.as_deref(),
                event.event_type.as_str(),
                &event.timestamp,
            );
            stored.events.push(event);
        }
        Ok(notifications)
    }

    /// Appends a terminal event for a turn that exists only in a verified durable EventLog tail.
    ///
    /// Normal callers must target a loaded turn. Mailbox recovery is different: the failed turn
    /// was intentionally rolled back from memory while its EventLog-first input remained durable.
    pub(in crate::runtime) fn append_durable_recovery_terminal_event(
        &self,
        session_id: &str,
        turn_id: &str,
        runtime_event: RuntimeEvent,
    ) -> Result<Vec<AgentEvent>, RuntimeCoreError> {
        if !is_turn_terminal_event_class(normalized_runtime_event_class(&runtime_event.event_type))
        {
            return Err(RuntimeCoreError::Backend(
                "durable recovery accepts terminal events only".to_string(),
            ));
        }
        let mut state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        let stored = state
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.to_string()))?;
        let thread_id = stored.session.thread_id.clone();
        let session_status = stored.session.status;
        let session_updated_at = stored.session.updated_at.clone();
        let events = append_runtime_events_to_stored_session(
            stored,
            self.file_checkpoint_snapshot_store.as_ref(),
            self.output_snapshot_store.as_ref(),
            self.sidecar_store.as_deref(),
            self.event_log_writer.as_deref(),
            self.trace_event_writer.as_deref(),
            self.projection_store.as_deref(),
            Some(&self.session_loops),
            session_id,
            &thread_id,
            Some(turn_id),
            vec![runtime_event],
            None,
            TerminalTurnAppendPolicy::Allow,
        )?;
        stored.session.status = session_status;
        stored.session.updated_at = session_updated_at;
        Ok(events)
    }
}

pub(in crate::runtime) fn append_workflow_audit_runtime_events(
    event_log_writer: Option<&EventLogWriter>,
    session_id: &str,
    thread_id: &str,
    turn_id: Option<&str>,
    runtime_events: Vec<RuntimeEvent>,
) -> Result<(), RuntimeCoreError> {
    let Some(event_log_writer) = event_log_writer else {
        return Ok(());
    };
    if runtime_events.is_empty() {
        return Ok(());
    }

    let base_sequence = event_log_writer
        .read_session_workflow_audit_events(session_id)
        .map_err(RuntimeCoreError::Backend)?
        .len() as u64;
    let audit_events = runtime_events
        .into_iter()
        .enumerate()
        .map(|(index, runtime_event)| AgentEvent {
            event_id: new_id("audit"),
            sequence: base_sequence + index as u64 + 1,
            session_id: session_id.to_string(),
            thread_id: Some(thread_id.to_string()),
            turn_id: turn_id.map(str::to_string),
            event_type: runtime_event.event_type,
            timestamp: timestamp(),
            payload: runtime_event.payload,
        })
        .collect::<Vec<_>>();
    event_log_writer
        .append_workflow_audit_events(session_id, &audit_events)
        .map_err(RuntimeCoreError::Backend)?;
    Ok(())
}

pub(in crate::runtime) fn append_runtime_events_to_state(
    state: &Arc<Mutex<RuntimeCoreState>>,
    file_checkpoint_snapshot_store: &dyn crate::file_checkpoint_snapshot::FileCheckpointSnapshotStore,
    output_snapshot_store: &dyn output_refs::OutputSnapshotStore,
    sidecar_store: Option<&SidecarStore>,
    event_log_writer: Option<&EventLogWriter>,
    trace_event_writer: Option<&TraceEventWriter>,
    projection_store: Option<&ProjectionStore>,
    session_loops: Option<&agent_runtime::session_loop::RuntimeSessionRegistry>,
    session_id: &str,
    thread_id: &str,
    turn_id: Option<&str>,
    runtime_events: Vec<RuntimeEvent>,
) -> Result<Vec<AgentEvent>, RuntimeCoreError> {
    let mut state = state.lock().expect("runtime core state mutex poisoned");
    let stored = state
        .sessions
        .get_mut(session_id)
        .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.to_string()))?;
    append_runtime_events_to_stored_session(
        stored,
        file_checkpoint_snapshot_store,
        output_snapshot_store,
        sidecar_store,
        event_log_writer,
        trace_event_writer,
        projection_store,
        session_loops,
        session_id,
        thread_id,
        turn_id,
        runtime_events,
        None,
        TerminalTurnAppendPolicy::Reject,
    )
}

pub(in crate::runtime) fn append_runtime_events_to_state_with_message_lifecycle(
    state: &Arc<Mutex<RuntimeCoreState>>,
    file_checkpoint_snapshot_store: &dyn crate::file_checkpoint_snapshot::FileCheckpointSnapshotStore,
    output_snapshot_store: &dyn output_refs::OutputSnapshotStore,
    sidecar_store: Option<&SidecarStore>,
    event_log_writer: Option<&EventLogWriter>,
    trace_event_writer: Option<&TraceEventWriter>,
    projection_store: Option<&ProjectionStore>,
    session_loops: Option<&agent_runtime::session_loop::RuntimeSessionRegistry>,
    session_id: &str,
    thread_id: &str,
    turn_id: &str,
    runtime_events: Vec<RuntimeEvent>,
    message_lifecycle: &mut CanonicalMessageLifecycleState,
) -> Result<Vec<AgentEvent>, RuntimeCoreError> {
    let mut state = state.lock().expect("runtime core state mutex poisoned");
    let stored = state
        .sessions
        .get_mut(session_id)
        .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.to_string()))?;
    append_runtime_events_to_stored_session(
        stored,
        file_checkpoint_snapshot_store,
        output_snapshot_store,
        sidecar_store,
        event_log_writer,
        trace_event_writer,
        projection_store,
        session_loops,
        session_id,
        thread_id,
        Some(turn_id),
        runtime_events,
        Some(message_lifecycle),
        TerminalTurnAppendPolicy::Reject,
    )
}

fn append_runtime_events_to_stored_session(
    stored: &mut StoredSession,
    file_checkpoint_snapshot_store: &dyn crate::file_checkpoint_snapshot::FileCheckpointSnapshotStore,
    output_snapshot_store: &dyn output_refs::OutputSnapshotStore,
    sidecar_store: Option<&SidecarStore>,
    event_log_writer: Option<&EventLogWriter>,
    trace_event_writer: Option<&TraceEventWriter>,
    projection_store: Option<&ProjectionStore>,
    session_loops: Option<&agent_runtime::session_loop::RuntimeSessionRegistry>,
    session_id: &str,
    thread_id: &str,
    turn_id: Option<&str>,
    runtime_events: Vec<RuntimeEvent>,
    message_lifecycle: Option<&mut CanonicalMessageLifecycleState>,
    terminal_turn_policy: TerminalTurnAppendPolicy,
) -> Result<Vec<AgentEvent>, RuntimeCoreError> {
    if terminal_turn_policy == TerminalTurnAppendPolicy::Reject && is_terminal_turn(stored, turn_id)
    {
        return Ok(Vec::new());
    }
    let runtime_events = deduplicate_mailbox_runtime_events(stored, runtime_events);
    let mut runtime_events = runtime_events_with_turn_input(stored, turn_id, runtime_events);
    for event in &mut runtime_events {
        if turn_input_events::runtime_event_is_provider_input(event) {
            super::input_media::attach_input_media_output_refs(
                &mut event.payload,
                sidecar_store,
                session_id,
            )
            .map_err(RuntimeCoreError::Backend)?;
        }
    }
    let runtime_events = with_canonical_message_reasoning_lifecycle(
        &stored.events,
        turn_id,
        runtime_events,
        message_lifecycle,
    )
    .map_err(RuntimeCoreError::Backend)?;
    if let Some(retired_event) = runtime_events.iter().find(|event| {
        is_retired_tool_wire_event_class(normalized_runtime_event_class(&event.event_type))
    }) {
        return Err(RuntimeCoreError::Backend(format!(
            "retired raw tool wire event is forbidden: {}",
            retired_event.event_type
        )));
    }
    if runtime_events.is_empty() {
        return Ok(Vec::new());
    }
    let should_resolve_soul_style = runtime_events.iter().any(|event| {
        let event_class = normalized_runtime_event_class(&event.event_type);
        should_enrich_tool_process_event_payload_class(event_class)
            || should_enrich_tool_policy_event_payload_class(event_class)
    });
    let soul_config_metadata = should_resolve_soul_style
        .then(current_agent_runtime_config_metadata)
        .flatten();
    let fallback_soul_style =
        SoulStyleMetadata::from_config_metadata(soul_config_metadata.as_ref());
    let (workflow_audit_events, runtime_events): (Vec<_>, Vec<_>) = runtime_events
        .into_iter()
        .partition(|event| event.event_type.starts_with("workflow."));
    append_workflow_audit_runtime_events(
        event_log_writer,
        session_id,
        thread_id,
        turn_id,
        workflow_audit_events,
    )?;
    if runtime_events.is_empty() {
        return Ok(Vec::new());
    }
    let trace_context = trace::trace_context_for_turn(stored, turn_id);
    let mut events = Vec::with_capacity(runtime_events.len());
    let mut validation = EventValidationContext::from_events(&stored.events, session_id, turn_id);
    let mut output_records = Vec::new();
    let mut pending_terminal_for_turn = false;
    for runtime_event in runtime_events {
        let event_type = runtime_event.event_type;
        let event_class = normalized_runtime_event_class(&event_type);
        let needs_sequence_context =
            should_build_validation_context(event_class, pending_terminal_for_turn);
        let needs_policy_normalization = should_normalize_policy_event_payload_class(event_class);
        let needs_tool_lifecycle_validation =
            should_validate_tool_lifecycle_event_class(event_class);
        let payload = if needs_policy_normalization {
            tool_lifecycle::normalize_policy_event_payload(
                validation.events(),
                turn_id,
                &event_type,
                runtime_event.payload,
            )
        } else {
            runtime_event.payload
        };
        let payload = if should_enrich_tool_process_event_payload_class(event_class) {
            let mut payload = payload;
            if let Value::Object(payload_object) = &mut payload {
                tool_process_external_metadata::enrich_external_tool_process_payload(
                    validation.events(),
                    &event_type,
                    payload_object,
                    fallback_soul_style.as_ref(),
                );
            }
            payload
        } else {
            payload
        };
        let payload = if should_enrich_tool_policy_event_payload_class(event_class) {
            let mut payload = payload;
            if let Value::Object(payload_object) = &mut payload {
                tool_process_external_metadata::enrich_external_tool_policy_payload(
                    validation.events(),
                    &event_type,
                    payload_object,
                    fallback_soul_style.as_ref(),
                );
            }
            payload
        } else {
            payload
        };
        let text_delta_event = is_text_delta_event_class(event_class);
        let turn_terminal_event = is_turn_terminal_event_class(event_class);
        let normalized = if text_delta_event {
            output_refs::NormalizedOutputPayload {
                payload,
                output_blob: None,
            }
        } else {
            output_refs::normalize_large_output_payload(&event_type, payload)
        };
        let event_timestamp = runtime_event_timestamp(event_class, &normalized.payload);
        let mut event = AgentEvent {
            event_id: new_id("evt"),
            sequence: stored
                .events
                .last()
                .map(|event| event.sequence)
                .unwrap_or(0)
                .checked_add(u64::try_from(events.len()).map_err(|_| {
                    RuntimeCoreError::Backend("runtime event batch length exceeds u64".to_string())
                })?)
                .and_then(|sequence| sequence.checked_add(1))
                .ok_or_else(|| RuntimeCoreError::Backend("event sequence overflow".to_string()))?,
            session_id: session_id.to_string(),
            thread_id: Some(thread_id.to_string()),
            turn_id: turn_id.map(str::to_string),
            event_type,
            timestamp: event_timestamp,
            payload: normalized.payload,
        };
        if let Some(trace_context) = trace_context.as_ref() {
            trace::attach_agent_event_trace(&mut event, trace_context);
        }
        if !text_delta_event {
            crate::file_checkpoint_snapshot::persist_runtime_file_checkpoint_snapshot(
                &mut event,
                session_id,
                file_checkpoint_snapshot_store,
            )
            .map_err(RuntimeCoreError::Backend)?;
            artifact_sidecar::persist_artifact_snapshot_payload(
                event.event_type.as_str(),
                &mut event.payload,
                session_id,
                event.event_id.as_str(),
                sidecar_store,
            )?;
            output_media::persist_local_image_output_payload(
                event.event_type.as_str(),
                &mut event.payload,
                session_id,
                sidecar_store,
            )?;
        }
        attach_session_projection_metadata(&mut event, stored);
        attach_goal_accounting_mode(&mut event, stored);
        attach_canonical_item_entity(stored, &events, &mut event)
            .map_err(RuntimeCoreError::Backend)?;
        super::thread_usage::canonicalize_runtime_usage_event(&mut event, &stored.events, &events);
        agent_ui_event_schema::validate_agent_event(&event).map_err(RuntimeCoreError::Backend)?;
        validation
            .validate_and_observe(
                &event,
                needs_sequence_context,
                needs_tool_lifecycle_validation,
            )
            .map_err(RuntimeCoreError::Backend)?;
        if let Some(output_blob) = normalized.output_blob {
            let output = output_refs::record_output_blob(&event, output_blob);
            let output =
                output_refs::persist_output_record(output, session_id, output_snapshot_store)?;
            output_refs::attach_output_snapshot_ref(&mut event.payload, &output);
            output_records.push(output);
        }
        if event.turn_id.as_deref() == turn_id && turn_terminal_event {
            pending_terminal_for_turn = true;
        }
        events.push(event);
    }
    let appended_events = events;
    let notification_events =
        notification_events_with_canonical_entities(stored, &appended_events)?;
    let requires_mailbox_canonical_persistence = requires_canonical_persistence(&appended_events);
    let terminal_result_required = projection_store
        .map(|store| {
            store.terminal_agent_result_required_sync(
                &agent_protocol::ThreadId::new(thread_id),
                &appended_events,
            )
        })
        .transpose()
        .map_err(|error| {
            RuntimeCoreError::Backend(format!(
                "failed to resolve durable child terminal activity owner: {error}"
            ))
        })?
        .unwrap_or(false);
    let requires_canonical_persistence =
        requires_mailbox_canonical_persistence || terminal_result_required;
    if let Some(event_log_writer) = event_log_writer {
        event_log_writer
            .append_events(&appended_events)
            .map_err(RuntimeCoreError::Backend)?;
    }
    if let Some(trace_event_writer) = trace_event_writer {
        if let Err(error) = trace_event_writer.append_agent_events(&appended_events) {
            tracing::warn!(
                "[trace-event-store] failed to append {} trace candidate events for session {}: {}",
                appended_events.len(),
                session_id,
                error
            );
        }
    }
    if let Some(projection_store) = projection_store {
        if requires_canonical_persistence {
            projection_store
                .apply_canonical_events(stored, &appended_events)
                .map_err(|error| {
                    let message = if requires_mailbox_canonical_persistence {
                        "canonical mailbox Item must persist before delivery acknowledgement"
                    } else {
                        "canonical child terminal Turn must persist before parent activity"
                    };
                    RuntimeCoreError::Backend(format!("{message}: {error}"))
                })?;
        }
        if let Err(error) = projection_store.apply_events(&appended_events) {
            tracing::warn!(
                "[projection-store] failed to apply {} events for session {}: {}",
                appended_events.len(),
                session_id,
                error
            );
        }
        if !requires_canonical_persistence {
            projection_store
                .apply_canonical_events(stored, &appended_events)
                .map_err(|error| {
                    RuntimeCoreError::Backend(format!(
                        "canonical ThreadStore projection failed after EventLog append: {error}"
                    ))
                })?;
        }
        if terminal_result_required {
            let terminal_activities = projection_store
                .append_terminal_agent_results_sync(
                    &agent_protocol::ThreadId::new(thread_id),
                    &appended_events,
                )
                .map_err(|error| {
                    RuntimeCoreError::Backend(format!(
                        "failed to persist durable child terminal activity: {error}"
                    ))
                })?;
            if let Some(session_loops) = session_loops {
                super::agent_terminal_activity::publish_terminal_agent_activities(
                    session_loops,
                    terminal_activities,
                );
            }
        }
    }
    for event in appended_events.iter().cloned() {
        apply_runtime_event_state_transition(
            stored,
            turn_id,
            event.event_type.as_str(),
            &event.timestamp,
        );
        stored.events.push(event);
    }
    for output in output_records {
        stored
            .output_blobs
            .insert(output.output_ref.clone(), output);
    }
    Ok(notification_events)
}

fn deduplicate_mailbox_runtime_events(
    stored: &StoredSession,
    runtime_events: Vec<RuntimeEvent>,
) -> Vec<RuntimeEvent> {
    let mut seen_message_ids = stored
        .events
        .iter()
        .filter_map(|event| mailbox_message_id(&event.payload).map(str::to_string))
        .collect::<HashSet<_>>();
    runtime_events
        .into_iter()
        .filter(|event| {
            mailbox_message_id(&event.payload)
                .map(|message_id| seen_message_ids.insert(message_id.to_string()))
                .unwrap_or(true)
        })
        .collect()
}

fn mailbox_message_id(payload: &Value) -> Option<&str> {
    payload
        .pointer("/mailbox/messageId")
        .and_then(Value::as_str)
}

fn requires_canonical_persistence(events: &[AgentEvent]) -> bool {
    events.iter().any(|event| {
        event.payload.get("mailbox").is_some()
            || event.payload.get("mailboxRecovery") == Some(&Value::Bool(true))
    })
}

fn runtime_events_with_turn_input(
    stored: &StoredSession,
    turn_id: Option<&str>,
    runtime_events: Vec<RuntimeEvent>,
) -> Vec<RuntimeEvent> {
    let Some(turn_id) = turn_id else {
        return runtime_events;
    };
    if stored.events.iter().any(|event| {
        event.turn_id.as_deref() == Some(turn_id)
            && turn_input_events::is_provider_input_event(event)
    }) || runtime_events
        .iter()
        .any(turn_input_events::runtime_event_is_provider_input)
    {
        return runtime_events;
    }
    let Some(input) = stored.turn_inputs.get(turn_id) else {
        return runtime_events;
    };
    let client_user_message_id = stored
        .turn_runtime_options
        .get(turn_id)
        .and_then(|options| options.runtime_metadata())
        .and_then(|metadata| super::session_submission::metadata(Some(metadata)).0);
    let Some(input_event) = turn_input_events::runtime_event_for_turn_input_with_client_id(
        input,
        client_user_message_id.as_deref(),
    ) else {
        return runtime_events;
    };

    let mut events = Vec::with_capacity(runtime_events.len() + 1);
    events.push(input_event);
    events.extend(runtime_events);
    events
}

fn attach_session_projection_metadata(event: &mut AgentEvent, stored: &StoredSession) {
    let Some(payload) = event.payload.as_object_mut() else {
        return;
    };
    let session_payload = payload
        .entry("session")
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    if !session_payload.is_object() {
        *session_payload = serde_json::Value::Object(serde_json::Map::new());
    }
    let Some(session_payload) = session_payload.as_object_mut() else {
        return;
    };

    insert_session_projection_string(
        session_payload,
        "createdAt",
        Some(stored.session.created_at.as_str()),
    );
    insert_session_projection_string(
        session_payload,
        "updatedAt",
        Some(stored.session.updated_at.as_str()),
    );
    insert_session_projection_string(
        session_payload,
        "appId",
        Some(stored.session.app_id.as_str()),
    );
    insert_session_projection_string(
        session_payload,
        "workspaceId",
        stored.session.workspace_id.as_deref(),
    );
    insert_session_projection_string(
        session_payload,
        "title",
        session_business_object_title(stored).as_deref(),
    );
    insert_session_projection_string(
        session_payload,
        "model",
        session_business_object_metadata(stored, &["model", "modelName"]).as_deref(),
    );
    insert_session_projection_string(
        session_payload,
        "workingDir",
        session_business_object_metadata(stored, &["workingDir", "working_dir"]).as_deref(),
    );
    insert_session_projection_string(
        session_payload,
        "executionStrategy",
        session_business_object_metadata(stored, &["executionStrategy", "execution_strategy"])
            .as_deref(),
    );
    insert_session_projection_metadata(session_payload, stored);
}

fn attach_goal_accounting_mode(event: &mut AgentEvent, stored: &StoredSession) {
    if event.event_type != "turn.accepted" {
        return;
    }
    let Some(turn_id) = event.turn_id.as_deref() else {
        return;
    };
    let Some(payload) = event.payload.as_object_mut() else {
        return;
    };
    let plan = super::thread_goal::turn_uses_plan_mode(stored.turn_runtime_options.get(turn_id));
    payload.insert(
        "goalAccountingMode".to_string(),
        Value::String(if plan { "plan" } else { "default" }.to_string()),
    );
}

fn insert_session_projection_string(
    payload: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    value: Option<&str>,
) {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return;
    };
    payload.insert(
        key.to_string(),
        serde_json::Value::String(value.to_string()),
    );
}

fn insert_session_projection_metadata(
    payload: &mut serde_json::Map<String, serde_json::Value>,
    stored: &StoredSession,
) {
    let Some(metadata) = stored
        .session
        .business_object_ref
        .as_ref()
        .and_then(|reference| reference.metadata.as_ref())
    else {
        return;
    };
    payload.insert("metadata".to_string(), metadata.clone());
}

fn session_business_object_title(stored: &StoredSession) -> Option<String> {
    stored
        .session
        .business_object_ref
        .as_ref()
        .and_then(|reference| reference.title.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| session_business_object_metadata(stored, &["title"]))
}

fn session_business_object_metadata(stored: &StoredSession, keys: &[&str]) -> Option<String> {
    let metadata = stored
        .session
        .business_object_ref
        .as_ref()
        .and_then(|reference| reference.metadata.as_ref())?;
    keys.iter()
        .filter_map(|key| metadata.get(*key))
        .find_map(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn should_build_validation_context(event_class: &str, pending_terminal_for_turn: bool) -> bool {
    requires_sequence_validation_context_class(event_class) || pending_terminal_for_turn
}

fn requires_sequence_validation_context_class(event_class: &str) -> bool {
    matches!(
        event_class,
        "item.started" | "item.updated" | "item.completed"
    ) || event_class.starts_with("tool.")
        || event_class.starts_with("hook.")
        || event_class.starts_with("action.")
        || event_class.starts_with("patch.")
        || event_class.starts_with("command.")
        || event_class.starts_with("test.")
        || event_class.starts_with("permission.")
        || event_class.starts_with("sandbox.")
        || matches!(
            event_class,
            "turn.completed" | "turn.failed" | "turn.canceled"
        )
}

fn should_include_in_validation_context(event_type: &str) -> bool {
    let event_class = normalized_runtime_event_class(event_type);
    requires_sequence_validation_context_class(event_class)
        || should_validate_tool_lifecycle_event_class(event_class)
        || should_normalize_policy_event_payload_class(event_class)
        || is_turn_terminal_event_class(event_class)
}

fn is_approval_session_cache_auto_resolved(event: &AgentEvent) -> bool {
    normalized_runtime_event_class(&event.event_type) == "action.resolved"
        && payload_string(&event.payload, &["source"]).as_deref() == Some("approval_session_cache")
        && payload_string(&event.payload, &["actionType", "action_type"]).as_deref()
            == Some("tool_confirmation")
        && payload_string(&event.payload, &["decision"]).as_deref() == Some("allow_for_session")
        && payload_string(&event.payload, &["decisionScope", "decision_scope"]).as_deref()
            == Some("session")
}

fn should_normalize_policy_event_payload_class(event_class: &str) -> bool {
    matches!(
        event_class,
        "action.required"
            | "action.resolved"
            | "action.cancelled"
            | "action.canceled"
            | "action.expired"
            | "permission.denied"
            | "sandbox.blocked"
    )
}

fn should_validate_tool_lifecycle_event_class(event_class: &str) -> bool {
    matches!(
        event_class,
        "item.started"
            | "item.updated"
            | "item.completed"
            | "tool.progress"
            | "tool.output.delta"
            | "action.required"
            | "action.resolved"
            | "action.cancelled"
            | "action.canceled"
            | "action.expired"
            | "permission.denied"
            | "sandbox.blocked"
    )
}

fn should_enrich_tool_process_event_payload_class(event_class: &str) -> bool {
    matches!(
        event_class,
        "tool.started"
            | "tool.args.delta"
            | "tool.input.delta"
            | "tool.progress"
            | "tool.output.delta"
            | "tool.result"
            | "tool.failed"
    )
}

fn should_enrich_tool_policy_event_payload_class(event_class: &str) -> bool {
    matches!(
        event_class,
        "action.required" | "permission.denied" | "sandbox.blocked"
    )
}

fn payload_string(value: &Value, keys: &[&str]) -> Option<String> {
    let object = value.as_object()?;
    keys.iter()
        .filter_map(|key| object.get(*key))
        .find_map(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn normalized_runtime_event_class(event_type: &str) -> &str {
    match event_type {
        "message.created" => "message.created",
        "tool_args" => "tool.args",
        "tool_args_delta" => "tool.args.delta",
        "tool_output_delta" => "tool.output.delta",
        "tool_input_delta" => "tool.input.delta",
        "turn.canceled" => "turn.canceled",
        value => value,
    }
}

fn is_retired_tool_wire_event_class(event_class: &str) -> bool {
    matches!(
        event_class,
        "tool.started"
            | "tool.args"
            | "tool.result"
            | "tool.failed"
            | "tool_end"
            | "tool.args.delta"
            | "tool.input.delta"
    )
}

fn is_text_delta_event_class(event_class: &str) -> bool {
    matches!(
        event_class,
        "message.delta" | "message.delta_batch" | "message.batch" | "message.created"
    )
}

fn is_turn_terminal_event_class(event_class: &str) -> bool {
    matches!(
        event_class,
        "turn.completed" | "turn.failed" | "turn.canceled"
    )
}

fn runtime_event_timestamp(event_class: &str, payload: &Value) -> String {
    let is_imported_codex_event = payload.get("imported").and_then(Value::as_bool) == Some(true)
        && payload
            .get("sourceClient")
            .or_else(|| payload.get("source_client"))
            .and_then(Value::as_str)
            == Some("codex");
    if is_imported_codex_event && event_class == "turn.started" {
        if let Some(started_at) = payload
            .get("startedAt")
            .or_else(|| payload.get("started_at"))
            .and_then(Value::as_str)
            .filter(|value| chrono::DateTime::parse_from_rfc3339(value).is_ok())
        {
            return started_at.to_string();
        }
    }
    if is_imported_codex_event && is_turn_terminal_event_class(event_class) {
        if let Some(completed_at) = payload
            .get("completedAt")
            .or_else(|| payload.get("completed_at"))
            .and_then(Value::as_str)
            .filter(|value| chrono::DateTime::parse_from_rfc3339(value).is_ok())
        {
            return completed_at.to_string();
        }
    }
    timestamp()
}

fn is_terminal_turn(stored: &StoredSession, turn_id: Option<&str>) -> bool {
    let Some(turn_id) = turn_id else {
        return false;
    };
    stored
        .turns
        .iter()
        .find(|turn| turn.turn_id == turn_id)
        .is_some_and(|turn| agent_turn_is_terminal(turn.status))
}

fn apply_runtime_event_state_transition(
    stored: &mut StoredSession,
    turn_id: Option<&str>,
    event_type: &str,
    event_timestamp: &str,
) {
    let Some(turn_id) = turn_id else {
        return;
    };
    let Some(next_status) = turn_status_from_runtime_event(event_type) else {
        return;
    };
    let completed_at = matches!(
        next_status,
        AgentTurnStatus::Completed | AgentTurnStatus::Failed | AgentTurnStatus::Canceled
    )
    .then(|| event_timestamp.to_string());

    if let Some(turn) = stored.turns.iter_mut().find(|turn| turn.turn_id == turn_id) {
        turn.status = next_status;
        if let Some(completed_at) = completed_at.clone() {
            turn.completed_at = Some(completed_at);
        }
    }

    stored.session.status = session_status_from_turn_status(next_status);
    stored.session.updated_at = completed_at.unwrap_or_else(|| event_timestamp.to_string());
}

fn turn_status_from_runtime_event(event_type: &str) -> Option<AgentTurnStatus> {
    match event_type {
        "turn.started" => Some(AgentTurnStatus::Running),
        "turn.completed" => Some(AgentTurnStatus::Completed),
        "turn.failed" => Some(AgentTurnStatus::Failed),
        "turn.canceled" => Some(AgentTurnStatus::Canceled),
        "action.required" => Some(AgentTurnStatus::WaitingAction),
        "action.resolved" | "action.cancelled" | "action.canceled" | "action.expired" => {
            Some(AgentTurnStatus::Running)
        }
        _ => None,
    }
}
