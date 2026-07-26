use std::collections::{BTreeMap, HashSet};

use agent_protocol::{
    ItemStatus, SessionId, Thread, ThreadHistoryChangeSet, ThreadId, ThreadItem, ThreadItemPayload,
    ThreadStatus, ThreadTurnsView, ToolOutput, Turn, TurnApprovalState, TurnQueueState, TurnStatus,
};
use app_server_protocol::protocol::v2::ThreadForkParams;
use app_server_protocol::{
    AgentEvent, AgentSession, AgentSessionStatus, AgentTurn, AgentTurnStatus, BusinessObjectRef,
};
use chrono::TimeZone;
use serde_json::{Map, Value};
use thread_store::{ApplyThreadHistoryParams, CreateThreadParams, ReadThreadParams, ThreadStore};
use uuid::Uuid;

use super::{RuntimeCore, RuntimeCoreError, StoredSession};

#[cfg(test)]
mod tests;

pub(in crate::runtime) const FORK_CANONICAL_ITEM_EVENT_TYPE: &str = "thread.fork.canonical_item";
pub(in crate::runtime) const FORK_INTERRUPTED_MARKER_EVENT_TYPE: &str =
    "thread.fork.interrupted_marker";
const INTERRUPTED_DEVELOPER_GUIDANCE: &str = "The previous turn was interrupted on purpose. Any running unified exec processes may still be running in the background. If any tools/commands were aborted, they may have partially executed.";

struct ForkHistory {
    turn_ids: HashSet<String>,
    changes: Option<ThreadHistoryChangeSet>,
    interrupted_turn_id: Option<String>,
}

impl RuntimeCore {
    pub(crate) async fn fork_thread(
        &self,
        params: ThreadForkParams,
    ) -> Result<Thread, RuntimeCoreError> {
        validate_fork_params(&params)?;
        let source_thread_id = ThreadId::new(params.thread_id.trim().to_string());
        let store = self.projection_store.as_deref().ok_or_else(|| {
            RuntimeCoreError::Backend("canonical thread store is unavailable".to_string())
        })?;
        let source_metadata = store
            .read_thread(ReadThreadParams {
                thread_id: source_thread_id.clone(),
                include_archived: true,
                turns_view: ThreadTurnsView::NotLoaded,
            })
            .await
            .map_err(store_error)?
            .ok_or_else(|| {
                RuntimeCoreError::Backend(format!("thread not found: {source_thread_id}"))
            })?;
        if is_paginated_history(&source_metadata.metadata) {
            return Err(RuntimeCoreError::MethodNotFound(
                "paginated_threads is not supported yet".to_string(),
            ));
        }
        let source = store
            .read_thread(ReadThreadParams {
                thread_id: source_thread_id.clone(),
                include_archived: true,
                turns_view: ThreadTurnsView::Full,
            })
            .await
            .map_err(store_error)?
            .ok_or_else(|| {
                RuntimeCoreError::Backend(format!("thread not found: {source_thread_id}"))
            })?;
        let source_session_id = source.session_id.as_str().to_string();
        self.ensure_current_session_hydrated(&source_session_id)
            .await?;
        let source_stored = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned")
            .sessions
            .get(&source_session_id)
            .cloned()
            .ok_or_else(|| RuntimeCoreError::SessionNotFound(source_session_id.clone()))?;
        let target_thread_id = Uuid::now_v7().to_string();
        let target_session_id = target_thread_id.clone();
        let history = fork_history(&source, &params, &target_session_id, &target_thread_id)?;
        validate_fork_provider_history(&history)?;
        let history_sequence = history
            .changes
            .as_ref()
            .map(|changes| changes.sequence)
            .unwrap_or_default();
        let interrupted_boundaries = fork_interrupted_boundary_events(
            &source_stored.events,
            &history.turn_ids,
            history.interrupted_turn_id.as_deref(),
            &target_session_id,
            &target_thread_id,
            history_sequence,
        )?;
        let durable_sequence = interrupted_boundaries
            .last()
            .map(|event| event.sequence)
            .unwrap_or(history_sequence)
            .max(history_sequence);
        let token_usage_event = fork_token_usage_event(
            &source_stored.events,
            &history.turn_ids,
            &target_session_id,
            &target_thread_id,
            durable_sequence,
        )?;
        let now_ms = chrono::Utc::now().timestamp_millis();
        let target = fork_thread_snapshot(
            source,
            &target_session_id,
            &target_thread_id,
            now_ms,
            &params,
            history_sequence,
        )?;
        let mut target_stored = fork_stored_session(
            source_stored,
            &target_session_id,
            &target_thread_id,
            &target.metadata,
            &history.turn_ids,
            history
                .changes
                .as_ref()
                .map(|changes| changes.changed_turns.as_slice())
                .unwrap_or_default(),
            history
                .changes
                .as_ref()
                .map(|changes| changes.changed_items.as_slice())
                .unwrap_or_default(),
            history_sequence,
        )?;
        let mut durable_events = interrupted_boundaries;
        durable_events.extend(token_usage_event);
        target_stored.events = merge_fork_history_events(target_stored.events, durable_events)?;

        store
            .create_thread(CreateThreadParams {
                thread: target.clone(),
            })
            .await
            .map_err(store_error)?;
        let persist_result = async {
            if let Some(changes) = history.changes.clone() {
                store
                    .apply_history(ApplyThreadHistoryParams {
                        session_id: SessionId::new(target_session_id.clone()),
                        thread_id: ThreadId::new(target_thread_id.clone()),
                        changes,
                    })
                    .await
                    .map_err(store_error)?;
            }
            if params.defer_goal_continuation {
                store
                    .inherit_thread_goal_for_fork_sync(source_thread_id.as_str(), &target_thread_id)
                    .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;
            }
            if let Some(writer) = self.event_log_writer.as_ref() {
                writer
                    .append_events(&target_stored.events)
                    .map_err(RuntimeCoreError::Backend)?;
            }
            Ok::<(), RuntimeCoreError>(())
        }
        .await;
        if let Err(error) = persist_result {
            let _ = store.delete_session_data(&target_session_id);
            return Err(error);
        }

        self.state
            .lock()
            .expect("runtime core state mutex poisoned")
            .sessions
            .insert(target_session_id.clone(), target_stored);

        store
            .read_thread(ReadThreadParams {
                thread_id: ThreadId::new(target_thread_id.clone()),
                include_archived: false,
                turns_view: if params.exclude_turns {
                    ThreadTurnsView::NotLoaded
                } else {
                    ThreadTurnsView::Full
                },
            })
            .await
            .map_err(store_error)?
            .ok_or_else(|| {
                RuntimeCoreError::Backend(format!(
                    "forked thread disappeared after creation: {target_thread_id}"
                ))
            })
    }

    pub(in crate::runtime) fn hydrate_fork_session_from_canonical(
        &self,
        thread: &Thread,
    ) -> Result<(), RuntimeCoreError> {
        if thread.forked_from_id.is_none() {
            return Err(RuntimeCoreError::SessionNotFound(
                thread.session_id.as_str().to_string(),
            ));
        }
        let metadata = thread.metadata.as_object().ok_or_else(|| {
            RuntimeCoreError::Backend("forked thread metadata must be a JSON object".to_string())
        })?;
        let timestamp = |millis: i64| {
            chrono::Utc
                .timestamp_millis_opt(millis)
                .single()
                .map(|value| value.to_rfc3339())
                .unwrap_or_else(super::value_fields::timestamp)
        };
        let session_id = thread.session_id.as_str().to_string();
        let thread_id = thread.thread_id.as_str().to_string();
        let history_sequence = metadata
            .get("forkSequence")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                RuntimeCoreError::Backend(
                    "forked thread metadata omitted canonical forkSequence".to_string(),
                )
            })?;
        let mut stored = StoredSession {
            session: AgentSession {
                session_id: session_id.clone(),
                thread_id: thread_id.clone(),
                app_id: thread
                    .product
                    .clone()
                    .unwrap_or_else(|| "agent-chat".to_string()),
                workspace_id: metadata
                    .get("workspaceId")
                    .and_then(Value::as_str)
                    .map(ToString::to_string),
                business_object_ref: Some(BusinessObjectRef {
                    kind: "agent.thread".to_string(),
                    id: thread_id.clone(),
                    title: thread.name.clone(),
                    uri: None,
                    metadata: Some(thread.metadata.clone()),
                }),
                status: session_status(&thread.status),
                created_at: timestamp(thread.created_at_ms),
                updated_at: timestamp(thread.updated_at_ms),
            },
            turns: thread
                .turns
                .iter()
                .map(|turn| AgentTurn {
                    turn_id: turn.turn_id.as_str().to_string(),
                    session_id: session_id.clone(),
                    thread_id: thread_id.clone(),
                    status: turn_status(turn.status),
                    started_at: turn.started_at_ms.map(timestamp),
                    completed_at: turn.completed_at_ms.map(timestamp),
                })
                .collect(),
            turn_inputs: Default::default(),
            turn_runtime_options: Default::default(),
            events: Vec::new(),
            output_blobs: Default::default(),
        };
        let items = thread
            .turns
            .iter()
            .flat_map(|turn| turn.items.iter())
            .filter(|item| item.sequence <= history_sequence)
            .cloned()
            .collect::<Vec<_>>();
        let item_turn_ids = items
            .iter()
            .map(|item| item.turn_id.as_str())
            .collect::<HashSet<_>>();
        let turns = thread
            .turns
            .iter()
            .filter(|turn| item_turn_ids.contains(turn.turn_id.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        stored.events =
            fork_history_seed_events(&stored.session, &turns, &items, history_sequence)?;
        let event_log_events = self
            .event_log_writer
            .as_ref()
            .map(|writer| {
                writer
                    .read_session_events(&session_id)
                    .map(|records| records.into_iter().map(|record| record.event).collect())
                    .map_err(RuntimeCoreError::Backend)
            })
            .transpose()?
            .unwrap_or_default();
        stored.events = merge_fork_history_events(stored.events, event_log_events)?;
        stored.turn_inputs = super::turn_input_events::turn_inputs_from_events(&stored.events);
        stored.turn_runtime_options =
            super::queued_turn_intent::runtime_options_from_events(&stored.turns, &stored.events)
                .map_err(RuntimeCoreError::Backend)?;
        stored.output_blobs = stored
            .events
            .iter()
            .filter_map(super::output_refs::output_record_from_event)
            .map(|record| (record.output_ref.clone(), record))
            .collect();
        let mut state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        match state.sessions.entry(session_id) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(stored);
            }
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                merge_fork_history_seed(entry.get_mut(), stored.events)?;
            }
        }
        Ok(())
    }
}

fn is_paginated_history(metadata: &Value) -> bool {
    metadata
        .get("historyMode")
        .or_else(|| metadata.get("history_mode"))
        .and_then(Value::as_str)
        .is_some_and(|mode| mode == "paginated")
}

fn fork_token_usage_event(
    source_events: &[AgentEvent],
    turn_ids: &HashSet<String>,
    target_session_id: &str,
    target_thread_id: &str,
    durable_sequence: u64,
) -> Result<Option<AgentEvent>, RuntimeCoreError> {
    let included_events = source_events
        .iter()
        .filter(|event| {
            event
                .turn_id
                .as_deref()
                .is_some_and(|turn_id| turn_ids.contains(turn_id))
        })
        .cloned()
        .collect::<Vec<_>>();
    let Some(snapshot) =
        super::thread_usage::thread_token_usage_snapshot_from_events(&included_events)
    else {
        return Ok(None);
    };
    let sequence = durable_sequence
        .checked_add(1)
        .ok_or_else(|| invalid("thread/fork token usage sequence overflow"))?;
    let timestamp = included_events
        .iter()
        .find(|event| event.sequence == snapshot.source_sequence)
        .map(|event| event.timestamp.clone())
        .unwrap_or_else(super::value_fields::timestamp);
    let usage = |value: &super::thread_usage::TokenUsageSnapshot| {
        serde_json::json!({
            "total_tokens": value.total_tokens,
            "input_tokens": value.input_tokens,
            "cached_input_tokens": value.cached_input_tokens,
            "cache_write_input_tokens": value.cache_write_input_tokens,
            "output_tokens": value.output_tokens,
            "reasoning_output_tokens": value.reasoning_output_tokens,
        })
    };
    Ok(Some(AgentEvent {
        event_id: format!("evt-thread-fork-token-usage-{target_session_id}"),
        sequence,
        session_id: target_session_id.to_string(),
        thread_id: Some(target_thread_id.to_string()),
        turn_id: Some(snapshot.turn_id),
        event_type: "thread.token_usage".to_string(),
        timestamp,
        payload: serde_json::json!({
            "token_usage": {
                "total_token_usage": usage(&snapshot.total_token_usage),
                "last_token_usage": usage(&snapshot.last_token_usage),
                "model_context_window": snapshot.model_context_window,
            }
        }),
    }))
}

fn fork_interrupted_boundary_events(
    source_events: &[AgentEvent],
    turn_ids: &HashSet<String>,
    interrupted_turn_id: Option<&str>,
    target_session_id: &str,
    target_thread_id: &str,
    history_sequence: u64,
) -> Result<Vec<AgentEvent>, RuntimeCoreError> {
    let mut boundaries = source_events
        .iter()
        .filter(|event| {
            (event.event_type == FORK_INTERRUPTED_MARKER_EVENT_TYPE
                || (event.event_type == "turn.canceled"
                    && event.payload.get("forkSnapshot").and_then(Value::as_bool) == Some(true)))
                && event
                    .turn_id
                    .as_deref()
                    .is_some_and(|turn_id| turn_ids.contains(turn_id))
        })
        .cloned()
        .collect::<Vec<_>>();
    boundaries.sort_by_key(|event| event.sequence);
    let mut seen_sequences = HashSet::new();
    for event in &mut boundaries {
        if event.sequence == 0 || !seen_sequences.insert(event.sequence) {
            return Err(invalid(
                "thread/fork interrupted boundaries must have unique positive sequences",
            ));
        }
        event.event_id = format!(
            "evt-thread-fork-interrupted-{}-{}-{}",
            target_session_id,
            event.sequence,
            event.event_type.replace('.', "-")
        );
        event.session_id = target_session_id.to_string();
        event.thread_id = Some(target_thread_id.to_string());
    }
    if let Some(turn_id) = interrupted_turn_id {
        let marker_sequence = history_sequence
            .checked_add(1)
            .ok_or_else(|| invalid("thread/fork interrupted marker sequence overflow"))?;
        let boundary_sequence = marker_sequence
            .checked_add(1)
            .ok_or_else(|| invalid("thread/fork interrupted boundary sequence overflow"))?;
        if boundaries
            .iter()
            .any(|event| matches!(event.sequence, sequence if sequence == marker_sequence || sequence == boundary_sequence))
        {
            return Err(invalid(
                "thread/fork interrupted boundary collides with inherited history",
            ));
        }
        let timestamp = source_events
            .iter()
            .rev()
            .find(|event| event.turn_id.as_deref() == Some(turn_id))
            .map(|event| event.timestamp.clone())
            .unwrap_or_else(super::value_fields::timestamp);
        boundaries.push(AgentEvent {
            event_id: format!(
                "evt-thread-fork-interrupted-{target_session_id}-{marker_sequence}-marker"
            ),
            sequence: marker_sequence,
            session_id: target_session_id.to_string(),
            thread_id: Some(target_thread_id.to_string()),
            turn_id: Some(turn_id.to_string()),
            event_type: FORK_INTERRUPTED_MARKER_EVENT_TYPE.to_string(),
            timestamp,
            payload: serde_json::json!({
                "reason": "interrupted",
                "role": "developer",
                "text": INTERRUPTED_DEVELOPER_GUIDANCE,
                "forkSnapshot": true,
            }),
        });
        boundaries.push(AgentEvent {
            event_id: format!(
                "evt-thread-fork-interrupted-{target_session_id}-{boundary_sequence}-boundary"
            ),
            sequence: boundary_sequence,
            session_id: target_session_id.to_string(),
            thread_id: Some(target_thread_id.to_string()),
            turn_id: Some(turn_id.to_string()),
            event_type: "turn.canceled".to_string(),
            timestamp: super::value_fields::timestamp(),
            payload: serde_json::json!({
                "reason": "interrupted",
                "forkSnapshot": true,
            }),
        });
    }
    boundaries.sort_by_key(|event| event.sequence);
    Ok(boundaries)
}

fn validate_fork_provider_history(history: &ForkHistory) -> Result<(), RuntimeCoreError> {
    for item in history
        .changes
        .iter()
        .flat_map(|changes| changes.changed_items.iter())
    {
        validate_fork_canonical_item(item)?;
    }
    Ok(())
}

fn validate_fork_canonical_item(item: &ThreadItem) -> Result<(), RuntimeCoreError> {
    match &item.payload {
        ThreadItemPayload::UserMessage { .. }
        | ThreadItemPayload::AgentMessage { .. }
        | ThreadItemPayload::Reasoning { .. }
        | ThreadItemPayload::Tool { .. }
        | ThreadItemPayload::McpToolCall { .. }
            if !item.status.is_terminal() =>
        {
            return Err(invalid(format!(
                "thread/fork cannot preserve non-terminal canonical item {}",
                item.item_id
            )));
        }
        ThreadItemPayload::AgentMessage { content_parts, .. }
            if content_parts
                .iter()
                .any(|part| matches!(part, agent_protocol::MessageContentPart::Media { .. })) =>
        {
            return Err(invalid(
                "thread/fork cannot preserve assistant media content from canonical history",
            ));
        }
        ThreadItemPayload::Tool { output, .. } | ThreadItemPayload::McpToolCall { output, .. }
            if output.is_none() =>
        {
            return Err(invalid(format!(
                "thread/fork cannot preserve tool item {} without a canonical result",
                item.item_id
            )));
        }
        ThreadItemPayload::CollabAgentToolCall { .. } => {
            return Err(invalid(
                "thread/fork cannot preserve collab tool arguments from canonical history",
            ));
        }
        ThreadItemPayload::Media { .. } => {
            return Err(invalid(
                "thread/fork cannot preserve media content from canonical history",
            ));
        }
        ThreadItemPayload::ContextCompaction {
            replacement_history,
            ..
        } => {
            if replacement_history.is_empty() {
                return Err(invalid(
                    "thread/fork cannot preserve compacted provider history without complete canonical lineage",
                ));
            }
            let event = AgentEvent {
                event_id: format!("evt-thread-fork-validate-{}", item.item_id),
                sequence: item.sequence,
                session_id: item.session_id.as_str().to_string(),
                thread_id: Some(item.thread_id.as_str().to_string()),
                turn_id: Some(item.turn_id.as_str().to_string()),
                event_type: "context.compaction.completed".to_string(),
                timestamp: super::value_fields::timestamp(),
                payload: fork_compaction_payload(item)
                    .expect("context compaction payload was matched"),
            };
            super::context_compaction::latest_fork_compaction_seed(&[event])
                .map_err(|error| {
                    invalid(format!(
                        "thread/fork cannot preserve compacted provider history without complete canonical lineage: {error}"
                    ))
                })?
                .ok_or_else(|| {
                    invalid(
                        "thread/fork cannot preserve compacted provider history without complete canonical lineage",
                    )
                })?;
        }
        ThreadItemPayload::Extension { .. } => {
            return Err(invalid(
                "thread/fork cannot preserve extension provider history from canonical history",
            ));
        }
        _ => {}
    }
    Ok(())
}

fn validate_fork_params(params: &ThreadForkParams) -> Result<(), RuntimeCoreError> {
    if params.thread_id.trim().is_empty() {
        return Err(invalid("thread/fork requires a non-empty threadId"));
    }
    if params.last_turn_id.is_some() && params.before_turn_id.is_some() {
        return Err(invalid(
            "thread/fork beforeTurnId cannot be combined with lastTurnId",
        ));
    }
    if params.permissions.is_some() && params.sandbox.is_some() {
        return Err(invalid(
            "thread/fork permissions cannot be combined with sandbox",
        ));
    }
    if params
        .path
        .as_deref()
        .is_some_and(|path| !path.trim().is_empty())
    {
        return Err(invalid(
            "thread/fork path is not implemented by the current runtime boundary",
        ));
    }
    if params.ephemeral && params.defer_goal_continuation {
        return Err(invalid(
            "thread/fork deferGoalContinuation cannot be combined with ephemeral",
        ));
    }
    if params.ephemeral {
        return Err(invalid(
            "thread/fork ephemeral storage is not implemented by the current runtime boundary",
        ));
    }
    for (name, value) in [
        ("lastTurnId", params.last_turn_id.as_deref()),
        ("beforeTurnId", params.before_turn_id.as_deref()),
    ] {
        if value.is_some_and(|value| value.trim().is_empty()) {
            return Err(invalid(format!("thread/fork {name} must not be empty")));
        }
    }
    Ok(())
}

fn fork_history(
    source: &Thread,
    params: &ThreadForkParams,
    target_session_id: &str,
    target_thread_id: &str,
) -> Result<ForkHistory, RuntimeCoreError> {
    let interrupted_snapshot = params.last_turn_id.is_none() && params.before_turn_id.is_none();
    let end = if let Some(last_turn_id) = params.last_turn_id.as_deref() {
        source
            .turns
            .iter()
            .position(|turn| turn.turn_id.as_str() == last_turn_id.trim())
            .map(|index| index + 1)
            .ok_or_else(|| invalid(format!("turn not found: {}", last_turn_id.trim())))?
    } else if let Some(before_turn_id) = params.before_turn_id.as_deref() {
        source
            .turns
            .iter()
            .position(|turn| turn.turn_id.as_str() == before_turn_id.trim())
            .ok_or_else(|| invalid(format!("turn not found: {}", before_turn_id.trim())))?
    } else {
        source.turns.len()
    };
    let selected = &source.turns[..end];
    let non_terminal = selected
        .iter()
        .enumerate()
        .filter(|(_, turn)| !turn.is_terminal())
        .collect::<Vec<_>>();
    if !non_terminal.is_empty()
        && (!interrupted_snapshot
            || non_terminal.len() != 1
            || non_terminal[0].0 + 1 != selected.len())
    {
        return Err(invalid(format!(
            "cannot fork through in-progress turn: {}",
            non_terminal[0].1.turn_id
        )));
    }
    let interrupted_turn_id = non_terminal
        .first()
        .map(|(_, turn)| turn.turn_id.as_str().to_string());
    let turn_ids = selected
        .iter()
        .map(|turn| turn.turn_id.as_str().to_string())
        .collect::<HashSet<_>>();
    if selected.is_empty() {
        return Ok(ForkHistory {
            turn_ids,
            changes: None,
            interrupted_turn_id: None,
        });
    }

    let target_session_id = SessionId::new(target_session_id);
    let target_thread_id = ThreadId::new(target_thread_id);
    let mut changed_turns = Vec::with_capacity(selected.len());
    let mut changed_items = Vec::new();
    let mut sequence = 1;
    for source_turn in selected {
        let mut turn = source_turn.clone();
        if interrupted_turn_id.as_deref() == Some(turn.turn_id.as_str()) {
            interrupt_fork_turn(&mut turn);
        }
        turn.session_id = target_session_id.clone();
        turn.thread_id = target_thread_id.clone();
        for source_item in std::mem::take(&mut turn.items) {
            let mut item = source_item;
            item.session_id = target_session_id.clone();
            item.thread_id = target_thread_id.clone();
            sequence = sequence.max(item.sequence);
            changed_items.push(item);
        }
        changed_turns.push(turn);
    }
    Ok(ForkHistory {
        turn_ids,
        changes: Some(ThreadHistoryChangeSet {
            sequence,
            changed_turns,
            changed_items,
            ..Default::default()
        }),
        interrupted_turn_id,
    })
}

fn interrupt_fork_turn(turn: &mut Turn) {
    turn.status = TurnStatus::Interrupted;
    turn.queue = TurnQueueState::NotQueued;
    if turn.approval == TurnApprovalState::Pending {
        turn.approval = TurnApprovalState::Cancelled;
    }
    turn.error = None;
    turn.completed_at_ms = None;
    turn.duration_ms = None;
    for item in &mut turn.items {
        if !item.status.is_terminal() {
            item.status = ItemStatus::Interrupted;
            item.completed_at_ms = None;
        }
        match &mut item.payload {
            ThreadItemPayload::Tool { output, .. }
            | ThreadItemPayload::McpToolCall { output, .. }
                if output.is_none() =>
            {
                *output = Some(ToolOutput {
                    error: Some("turn interrupted before tool completed".to_string()),
                    ..ToolOutput::default()
                });
            }
            _ => {}
        }
    }
}

fn fork_thread_snapshot(
    mut source: Thread,
    target_session_id: &str,
    target_thread_id: &str,
    now_ms: i64,
    params: &ThreadForkParams,
    history_sequence: u64,
) -> Result<Thread, RuntimeCoreError> {
    source.session_id = SessionId::new(target_session_id);
    source.thread_id = ThreadId::new(target_thread_id);
    source.status = ThreadStatus::Idle;
    source.created_at_ms = now_ms;
    source.updated_at_ms = now_ms;
    source.recency_at_ms = Some(now_ms);
    source.archived = false;
    source.parent_thread_id = None;
    source.agent_path = None;
    source.agent_nickname = None;
    source.agent_role = None;
    source.last_task_message = None;
    source.agent_state = None;
    source.forked_from_id = Some(ThreadId::new(params.thread_id.trim()));
    source.turns.clear();
    source.turns_view = ThreadTurnsView::NotLoaded;

    let metadata = source
        .metadata
        .as_object_mut()
        .ok_or_else(|| invalid("thread/fork source metadata must be a JSON object"))?;
    apply_fork_overrides(metadata, params);
    metadata.insert("forkSequence".to_string(), Value::from(history_sequence));
    if let Some(model_provider) = params.model_provider.as_deref() {
        source.model_provider = model_provider.to_string();
    }
    Ok(source)
}

fn apply_fork_overrides(metadata: &mut Map<String, Value>, params: &ThreadForkParams) {
    for (key, value) in [
        (
            "modelName",
            params
                .model
                .as_ref()
                .map(|value| Value::String(value.clone())),
        ),
        (
            "providerSelector",
            params
                .model_provider
                .as_ref()
                .map(|value| Value::String(value.clone())),
        ),
        (
            "providerName",
            params
                .model_provider
                .as_ref()
                .map(|value| Value::String(value.clone())),
        ),
        (
            "workingDir",
            params
                .cwd
                .as_ref()
                .map(|value| Value::String(value.clone())),
        ),
        (
            "runtimeWorkspaceRoots",
            params
                .runtime_workspace_roots
                .as_ref()
                .and_then(|value| serde_json::to_value(value).ok()),
        ),
        ("approvalPolicy", params.approval_policy.clone()),
        ("approvalsReviewer", params.approvals_reviewer.clone()),
        ("sandbox", params.sandbox.clone()),
        (
            "permissions",
            params
                .permissions
                .as_ref()
                .map(|value| Value::String(value.clone())),
        ),
        (
            "config",
            params
                .config
                .as_ref()
                .and_then(|value| serde_json::to_value(value).ok()),
        ),
        (
            "baseInstructions",
            params
                .base_instructions
                .as_ref()
                .map(|value| Value::String(value.clone())),
        ),
        (
            "developerInstructions",
            params
                .developer_instructions
                .as_ref()
                .map(|value| Value::String(value.clone())),
        ),
        (
            "threadSource",
            params
                .thread_source
                .as_ref()
                .map(|value| Value::String(value.clone())),
        ),
    ] {
        if let Some(value) = value {
            metadata.insert(key.to_string(), value);
        }
    }
    if let Some(service_tier) = params.service_tier.as_ref() {
        metadata.insert(
            "serviceTier".to_string(),
            service_tier
                .as_ref()
                .map(|value| Value::String(value.clone()))
                .unwrap_or(Value::Null),
        );
    }
    metadata.insert("ephemeral".to_string(), Value::Bool(false));
}

fn fork_stored_session(
    mut source: StoredSession,
    target_session_id: &str,
    target_thread_id: &str,
    metadata: &Value,
    turn_ids: &HashSet<String>,
    canonical_turns: &[Turn],
    canonical_items: &[ThreadItem],
    history_sequence: u64,
) -> Result<StoredSession, RuntimeCoreError> {
    source.session.session_id = target_session_id.to_string();
    source.session.thread_id = target_thread_id.to_string();
    source.session.status = AgentSessionStatus::Idle;
    source.session.created_at = super::value_fields::timestamp();
    source.session.updated_at = source.session.created_at.clone();
    if let Some(reference) = source.session.business_object_ref.as_mut() {
        reference.id = target_thread_id.to_string();
        reference.metadata = Some(metadata.clone());
    }
    source.turns.retain(|turn| turn_ids.contains(&turn.turn_id));
    for turn in &mut source.turns {
        turn.session_id = target_session_id.to_string();
        turn.thread_id = target_thread_id.to_string();
        if let Some(canonical) = canonical_turns
            .iter()
            .find(|canonical| canonical.turn_id.as_str() == turn.turn_id)
        {
            turn.status = turn_status(canonical.status);
            turn.started_at = canonical.started_at_ms.map(|millis| {
                chrono::Utc
                    .timestamp_millis_opt(millis)
                    .single()
                    .map(|value| value.to_rfc3339())
                    .unwrap_or_else(super::value_fields::timestamp)
            });
            turn.completed_at = canonical.completed_at_ms.map(|millis| {
                chrono::Utc
                    .timestamp_millis_opt(millis)
                    .single()
                    .map(|value| value.to_rfc3339())
                    .unwrap_or_else(super::value_fields::timestamp)
            });
        }
    }
    source
        .turn_inputs
        .retain(|turn_id, _| turn_ids.contains(turn_id));
    source
        .turn_runtime_options
        .retain(|turn_id, _| turn_ids.contains(turn_id));
    source.events.clear();
    source.events = fork_history_seed_events(
        &source.session,
        canonical_turns,
        canonical_items,
        history_sequence,
    )?;
    source.output_blobs.clear();
    Ok(source)
}

pub(in crate::runtime) fn fork_history_seed_events(
    session: &AgentSession,
    canonical_turns: &[Turn],
    canonical_items: &[ThreadItem],
    through_sequence: u64,
) -> Result<Vec<AgentEvent>, RuntimeCoreError> {
    let mut turns_by_id = BTreeMap::new();
    for turn in canonical_turns {
        if turns_by_id
            .insert(turn.turn_id.as_str().to_string(), turn)
            .is_some()
        {
            return Err(invalid(format!(
                "thread/fork canonical history has duplicate turn {}",
                turn.turn_id
            )));
        }
    }
    let mut items_by_sequence = BTreeMap::new();
    for item in canonical_items {
        validate_fork_canonical_item(item)?;
        if item.sequence == 0 || item.sequence > through_sequence {
            return Err(invalid(format!(
                "thread/fork canonical item {} has invalid sequence {} through {}",
                item.item_id, item.sequence, through_sequence
            )));
        }
        if items_by_sequence.insert(item.sequence, item).is_some() {
            return Err(invalid(format!(
                "thread/fork canonical history has duplicate item sequence {}",
                item.sequence
            )));
        }
        if !turns_by_id.contains_key(item.turn_id.as_str()) {
            return Err(invalid(format!(
                "thread/fork canonical item {} has no owning turn",
                item.item_id
            )));
        }
    }
    for turn in canonical_turns {
        if !canonical_items
            .iter()
            .any(|item| item.turn_id == turn.turn_id)
        {
            return Err(invalid(format!(
                "thread/fork cannot preserve turn {} without canonical items",
                turn.turn_id
            )));
        }
    }

    (1..=through_sequence)
        .map(|sequence| {
            let item = items_by_sequence.get(&sequence).copied();
            let timestamp = item
                .and_then(|item| {
                    chrono::Utc
                        .timestamp_millis_opt(item.updated_at_ms)
                        .single()
                        .map(|value| value.to_rfc3339())
                })
                .unwrap_or_else(|| session.updated_at.clone());
            let (event_type, payload) = item.map_or_else(
                || ("thread.fork.baseline".to_string(), Value::Null),
                |item| {
                    let mut turn = (*turns_by_id
                        .get(item.turn_id.as_str())
                        .expect("canonical item turn was validated"))
                    .clone();
                    turn.items.clear();
                    if let Some(mut payload) = fork_compaction_payload(item) {
                        payload["forkTurn"] = serde_json::to_value(turn)
                            .expect("validated canonical turn must serialize");
                        return ("context.compaction.completed".to_string(), payload);
                    }
                    (
                        FORK_CANONICAL_ITEM_EVENT_TYPE.to_string(),
                        serde_json::json!({ "item": item, "forkTurn": turn }),
                    )
                },
            );
            Ok(AgentEvent {
                event_id: format!("evt-thread-fork-baseline-{}-{sequence}", session.session_id),
                sequence,
                session_id: session.session_id.clone(),
                thread_id: Some(session.thread_id.clone()),
                turn_id: item.map(|item| item.turn_id.as_str().to_string()),
                event_type,
                timestamp,
                payload,
            })
        })
        .collect()
}

fn fork_compaction_payload(item: &ThreadItem) -> Option<Value> {
    let ThreadItemPayload::ContextCompaction {
        summary,
        replacement_history,
        window_number,
        first_window_id,
        previous_window_id,
        window_id,
        tail_start_turn_id,
    } = &item.payload
    else {
        return None;
    };
    Some(serde_json::json!({
        "itemId": item.item_id.as_str(),
        "summary": summary,
        "replacementHistory": replacement_history,
        "windowNumber": window_number,
        "firstWindowId": first_window_id,
        "previousWindowId": previous_window_id,
        "windowId": window_id,
        "tailStartTurnId": tail_start_turn_id,
    }))
}

fn merge_fork_history_seed(
    stored: &mut StoredSession,
    seed: Vec<AgentEvent>,
) -> Result<(), RuntimeCoreError> {
    stored.events = merge_fork_history_events(seed, std::mem::take(&mut stored.events))?;
    Ok(())
}

pub(in crate::runtime) fn merge_fork_history_events(
    seed: Vec<AgentEvent>,
    mut existing: Vec<AgentEvent>,
) -> Result<Vec<AgentEvent>, RuntimeCoreError> {
    let prefix_len = seed.len();
    let mut merged = seed;
    existing.sort_by_key(|event| event.sequence);

    for event in existing {
        let sequence = usize::try_from(event.sequence)
            .map_err(|_| invalid("thread/fork event sequence does not fit in memory"))?;
        if sequence == 0 {
            return Err(invalid("thread/fork event sequence must start at one"));
        }
        if sequence <= prefix_len {
            let canonical = &merged[sequence - 1];
            if canonical.event_type == "thread.fork.baseline"
                && (event.event_type == FORK_INTERRUPTED_MARKER_EVENT_TYPE
                    || (event.event_type == "turn.canceled"
                        && event.payload.get("forkSnapshot").and_then(Value::as_bool)
                            == Some(true)))
            {
                merged[sequence - 1] = event;
                continue;
            }
            if canonical.sequence != event.sequence
                || canonical.event_id != event.event_id
                || canonical.session_id != event.session_id
                || canonical.thread_id != event.thread_id
                || canonical.turn_id != event.turn_id
                || canonical.event_type != event.event_type
                || canonical.payload != event.payload
            {
                return Err(invalid(format!(
                    "thread/fork canonical seed conflicts at sequence {}: canonical={canonical:?}, existing={event:?}",
                    event.sequence,
                )));
            }
            continue;
        }
        let expected = merged.len() + 1;
        if sequence != expected {
            return Err(invalid(format!(
                "thread/fork target EventLog is not contiguous: expected {expected}, got {sequence}"
            )));
        }
        merged.push(event);
    }
    Ok(merged)
}

fn session_status(status: &ThreadStatus) -> AgentSessionStatus {
    match status {
        ThreadStatus::Active { .. } => AgentSessionStatus::Running,
        ThreadStatus::SystemError => AgentSessionStatus::Failed,
        ThreadStatus::NotLoaded | ThreadStatus::Idle => AgentSessionStatus::Idle,
    }
}

fn turn_status(status: agent_protocol::TurnStatus) -> AgentTurnStatus {
    match status {
        agent_protocol::TurnStatus::InProgress => AgentTurnStatus::Running,
        agent_protocol::TurnStatus::Completed => AgentTurnStatus::Completed,
        agent_protocol::TurnStatus::Interrupted => AgentTurnStatus::Canceled,
        agent_protocol::TurnStatus::Failed => AgentTurnStatus::Failed,
    }
}

fn invalid(message: impl Into<String>) -> RuntimeCoreError {
    RuntimeCoreError::InvalidRequest(message.into())
}

fn store_error(error: impl std::fmt::Display) -> RuntimeCoreError {
    RuntimeCoreError::Backend(error.to_string())
}
