use super::output_refs;
use super::projection_repair::ProjectionRepair;
use super::projection_store::ProjectionReadSession;
use super::projection_store::ProjectionReadWindow;
use super::queued_turn_intent;
use super::read_model;
use super::read_model_turn_usage;
use super::turn_input_events;
use super::RuntimeCore;
use super::RuntimeCoreError;
use super::StoredSession;
use app_server_protocol::AgentEvent;
use app_server_protocol::AgentSessionReadParams;
use app_server_protocol::AgentSessionReadResponse;
use serde_json::json;
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub(in crate::runtime) struct SessionLoadContext {
    pub response: AgentSessionReadResponse,
    pub stored: StoredSession,
    pub workflow_audit_events: Vec<AgentEvent>,
}

impl RuntimeCore {
    pub(in crate::runtime) async fn load_session_current(
        &self,
        params: AgentSessionReadParams,
    ) -> Result<SessionLoadContext, RuntimeCoreError> {
        if let Some(mut context) = self.load_runtime_core_session(&params).await? {
            if should_prefer_projection_history_read(&context, &params) {
                if let Some(mut projection_context) = self.load_projection_session(&params).await? {
                    self.enrich_session_load_context_with_media_task_results(
                        &mut projection_context,
                    )
                    .await;
                    return Ok(projection_context);
                }
            }
            self.enrich_session_load_context_with_media_task_results(&mut context)
                .await;
            return Ok(context);
        }
        if let Some(mut context) = self.load_projection_session(&params).await? {
            self.enrich_session_load_context_with_media_task_results(&mut context)
                .await;
            return Ok(context);
        }
        Err(RuntimeCoreError::SessionNotFound(params.session_id))
    }

    async fn load_runtime_core_session(
        &self,
        params: &AgentSessionReadParams,
    ) -> Result<Option<SessionLoadContext>, RuntimeCoreError> {
        let stored = {
            let state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            state.sessions.get(&params.session_id).cloned()
        };
        let Some(stored) = stored else {
            return Ok(None);
        };
        if self.is_pending_agent_control_thread(&stored.session.thread_id)? {
            return Ok(None);
        }
        let active_turn_id = self.active_turn_id_for_session(&params.session_id).await?;
        let mut read_snapshot = stored.clone();
        super::status::normalize_agent_session_runtime_snapshot(
            &mut read_snapshot.session,
            &mut read_snapshot.turns,
            active_turn_id.as_deref(),
        );
        let workflow_audit_events =
            self.read_workflow_audit_events_for_session(&params.session_id)?;
        let detail = match self.projection_store.as_deref() {
            Some(projection_store) => read_model::runtime_session_read_detail_from_thread_store(
                &read_snapshot,
                read_model::ReadDetailOptions::from_params(params),
                &workflow_audit_events,
                projection_store,
                active_turn_id.as_deref(),
            )
            .await
            .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?,
            None => read_model::runtime_session_read_detail_with_options(
                &read_snapshot,
                read_model::ReadDetailOptions::from_params(params),
                &workflow_audit_events,
                active_turn_id.as_deref(),
            ),
        };
        let response = AgentSessionReadResponse {
            session: read_snapshot.session,
            turns: read_snapshot.turns,
            detail: Some(detail),
        };
        Ok(Some(SessionLoadContext {
            response,
            stored,
            workflow_audit_events,
        }))
    }

    pub(in crate::runtime) async fn load_projection_session(
        &self,
        params: &AgentSessionReadParams,
    ) -> Result<Option<SessionLoadContext>, RuntimeCoreError> {
        let (Some(event_log_writer), Some(projection_store)) = (
            self.event_log_writer.as_ref(),
            self.projection_store.as_ref(),
        ) else {
            return Ok(None);
        };
        let repair =
            ProjectionRepair::new((**event_log_writer).clone(), (**projection_store).clone());
        let Some((projection, events)) = repair
            .read_repaired_session(
                &params.session_id,
                params
                    .history_limit
                    .map(|_| ProjectionReadWindow::from_read_params(params)),
            )
            .map_err(RuntimeCoreError::Backend)?
        else {
            return Ok(None);
        };
        let workflow_audit_events =
            self.read_workflow_audit_events_for_session(&params.session_id)?;
        let projection_support_events = if events.is_empty() && params.history_limit.is_some() {
            self.read_projection_support_events_for_session(&params.session_id)?
        } else {
            Vec::new()
        };
        let active_turn_id = self.active_turn_id_for_session(&params.session_id).await?;
        Ok(Some(
            projection_load_context(
                projection,
                events,
                params,
                workflow_audit_events,
                projection_support_events,
                projection_store,
                active_turn_id.as_deref(),
            )
            .await?,
        ))
    }

    pub(in crate::runtime) fn read_workflow_audit_events_for_session(
        &self,
        session_id: &str,
    ) -> Result<Vec<AgentEvent>, RuntimeCoreError> {
        self.event_log_writer
            .as_ref()
            .map(|writer| {
                writer
                    .read_session_workflow_audit_events(session_id)
                    .map(|records| {
                        records
                            .into_iter()
                            .map(|record| record.event)
                            .collect::<Vec<_>>()
                    })
                    .map_err(RuntimeCoreError::Backend)
            })
            .transpose()
            .map(|events| events.unwrap_or_default())
    }

    fn read_projection_support_events_for_session(
        &self,
        session_id: &str,
    ) -> Result<Vec<AgentEvent>, RuntimeCoreError> {
        self.event_log_writer
            .as_ref()
            .map(|writer| {
                writer
                    .read_session_events(session_id)
                    .map(|records| {
                        records
                            .into_iter()
                            .map(|record| record.event)
                            .filter(|event| {
                                is_turn_completed_usage_event(event)
                                    || event.event_type == "runtime.warning"
                            })
                            .collect::<Vec<_>>()
                    })
                    .map_err(RuntimeCoreError::Backend)
            })
            .transpose()
            .map(|events| events.unwrap_or_default())
    }
}

fn should_prefer_projection_history_read(
    context: &SessionLoadContext,
    params: &AgentSessionReadParams,
) -> bool {
    params.history_limit.is_some()
        && context.stored.events.is_empty()
        && context.response.detail.as_ref().is_some_and(|detail| {
            detail_array_is_empty(detail, "messages") && detail_array_is_empty(detail, "items")
        })
}

fn detail_array_is_empty(detail: &Value, key: &str) -> bool {
    detail
        .get(key)
        .and_then(Value::as_array)
        .map_or(true, Vec::is_empty)
}

pub(in crate::runtime) async fn projection_load_context(
    projection: ProjectionReadSession,
    events: Vec<AgentEvent>,
    params: &AgentSessionReadParams,
    workflow_audit_events: Vec<AgentEvent>,
    projection_support_events: Vec<AgentEvent>,
    projection_store: &super::ProjectionStore,
    active_turn_id: Option<&str>,
) -> Result<SessionLoadContext, RuntimeCoreError> {
    let events = super::history_replacement::effective_events(&events);
    let stored = StoredSession {
        session: projection.session.clone(),
        turns: projection.turns.clone(),
        turn_inputs: turn_input_events::turn_inputs_from_events(&events),
        turn_runtime_options: queued_turn_intent::runtime_options_from_events(
            &projection.turns,
            &events,
        )
        .map_err(RuntimeCoreError::Backend)?,
        events: events.clone(),
        output_blobs: events
            .iter()
            .filter_map(output_refs::output_record_from_event)
            .map(|record| (record.output_ref.clone(), record))
            .collect(),
    };
    let mut read_snapshot = stored.clone();
    super::status::normalize_agent_session_runtime_snapshot(
        &mut read_snapshot.session,
        &mut read_snapshot.turns,
        active_turn_id,
    );
    let mut detail = if read_snapshot.events.is_empty() && params.history_limit.is_some() {
        let (mut usage_events, notification_events): (Vec<_>, Vec<_>) = projection_support_events
            .into_iter()
            .partition(is_turn_completed_usage_event);
        usage_events.extend(
            workflow_audit_events
                .iter()
                .filter(|event| is_turn_completed_usage_event(event))
                .cloned(),
        );
        projection_summary_detail(
            &read_snapshot,
            &projection,
            params,
            &usage_events,
            &notification_events,
            projection_store,
            active_turn_id,
        )
        .await?
    } else {
        read_model::runtime_session_read_detail_from_thread_store(
            &read_snapshot,
            read_model::ReadDetailOptions::from_params(params),
            &workflow_audit_events,
            projection_store,
            active_turn_id,
        )
        .await
        .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?
    };
    if let Some(detail_object) = detail.as_object_mut() {
        detail_object.insert(
            "projection_source".to_string(),
            json!("runtime.projection_1"),
        );
        detail_object.insert(
            "projection_item_count".to_string(),
            json!(projection.item_count),
        );
        detail_object.insert(
            "projection_last_event_sequence".to_string(),
            json!(projection.last_event_sequence),
        );
    }
    let response = AgentSessionReadResponse {
        session: read_snapshot.session,
        turns: read_snapshot.turns,
        detail: Some(detail),
    };
    Ok(SessionLoadContext {
        response,
        stored,
        workflow_audit_events,
    })
}

async fn projection_summary_detail(
    stored: &StoredSession,
    projection: &ProjectionReadSession,
    params: &AgentSessionReadParams,
    usage_events: &[AgentEvent],
    notification_events: &[AgentEvent],
    projection_store: &super::ProjectionStore,
    active_turn_id: Option<&str>,
) -> Result<serde_json::Value, RuntimeCoreError> {
    let messages = projection.messages.clone();
    let process_detail = projection_process_detail(stored, projection, active_turn_id);
    let process_thread_read = process_detail
        .as_ref()
        .and_then(|detail| detail.get("thread_read"))
        .and_then(Value::as_object);
    let canonical_items = read_model::canonical_items_from_thread_store(projection_store, stored)
        .await
        .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;
    let mut notification_stored = stored.clone();
    notification_stored.events = notification_events.to_vec();
    let items = Value::Array(read_model::merge_runtime_notification_items(
        &notification_stored,
        canonical_items,
    ));
    let thread_items = items.clone();
    let artifacts = process_detail_array(process_detail.as_ref(), "artifacts");
    let outputs = process_detail_array(process_detail.as_ref(), "outputs");
    let pending_requests = process_thread_read_array(process_thread_read, "pending_requests");
    let tool_calls = process_thread_read_array(process_thread_read, "tool_calls");
    let commands = process_thread_read_array(process_thread_read, "commands");
    let tests = process_thread_read_array(process_thread_read, "tests");
    let active_turn_id = process_thread_read_value(process_thread_read, "active_turn_id")
        .or_else(|| active_turn_id_from_stored_turns(stored));
    let change_summary = process_thread_read_value(process_thread_read, "change_summary");
    let active_command_id = process_thread_read_value(process_thread_read, "active_command_id");
    let active_test_run_id = process_thread_read_value(process_thread_read, "active_test_run_id");
    let active_action_id = process_thread_read_value(process_thread_read, "active_action_id");
    let model_routing = process_thread_read_value(process_thread_read, "model_routing");
    let service_model_slot = process_thread_read_value(process_thread_read, "service_model_slot");
    let latest_turn_id = stored.turns.last().map(|turn| turn.turn_id.as_str());
    let latest_turn_usage =
        read_model_turn_usage::latest_usage_for_turn(usage_events, latest_turn_id);
    let mut diagnostics =
        process_thread_read_value(process_thread_read, "diagnostics").unwrap_or_else(|| {
            json!({
                "latest_turn_status": stored.turns.last().map(|turn| super::status::agent_turn_status_label(turn.status)),
                "latest_turn_error_message": null,
                "pending_request_count": 0,
                "command_count": 0,
                "test_count": 0,
                "changed_file_count": 0,
                "patch_count": 0,
            })
        });
    if let Some(usage) = latest_turn_usage.clone() {
        if let Some(diagnostics_object) = diagnostics.as_object_mut() {
            diagnostics_object.insert("latest_turn_usage".to_string(), usage);
        }
    }
    let mut runtime_summary = process_thread_read_value(process_thread_read, "runtime_summary")
        .unwrap_or_else(|| json!({}));
    if let Some(usage) = latest_turn_usage {
        if let Some(runtime_summary_object) = runtime_summary.as_object_mut() {
            runtime_summary_object.insert("latestTurnUsage".to_string(), usage);
        }
    }
    let turns = read_model_turn_usage::turns_with_usage(&stored.turns, usage_events);
    let loaded_count = messages.len();
    let messages_count = projection.messages_count;
    let history_limit = params
        .history_limit
        .map(|value| value as usize)
        .unwrap_or(messages_count);
    let history_offset = params.history_offset.unwrap_or_default() as usize;
    let start_index = projection.messages_start_index;
    let oldest_message_id = messages.first().and_then(|message| {
        message.get("id").and_then(|value| match value {
            serde_json::Value::Number(number) => number.as_i64(),
            serde_json::Value::String(value) => value.parse::<i64>().ok(),
            _ => None,
        })
    });
    let status = super::status::agent_session_status_label(stored.session.status);
    let thread_read = serde_json::json!({
        "session_id": stored.session.session_id,
        "thread_id": stored.session.thread_id,
        "status": status,
        "turns": turns.clone(),
        "pending_requests": pending_requests,
        "queued_turns": [],
        "active_turn_id": active_turn_id,
        "thread_items": thread_items,
        "tool_calls": tool_calls,
        "commands": commands,
        "tests": tests,
        "active_command_id": active_command_id,
        "active_test_run_id": active_test_run_id,
        "active_action_id": active_action_id,
        "change_summary": change_summary,
        "model_routing": model_routing,
        "service_model_slot": service_model_slot,
        "artifacts": process_thread_read_array(process_thread_read, "artifacts"),
        "outputs": process_thread_read_array(process_thread_read, "outputs"),
        "diagnostics": diagnostics,
        "runtime_summary": runtime_summary,
    });
    let mut detail = serde_json::json!({
        "id": stored.session.session_id,
        "session_id": stored.session.session_id,
        "thread_id": stored.session.thread_id,
        "workspace_id": stored.session.workspace_id,
        "status": status,
        "working_dir": process_detail_value(process_detail.as_ref(), "working_dir"),
        "archived_at": process_detail_value(process_detail.as_ref(), "archived_at"),
        "execution_strategy": process_detail_value(process_detail.as_ref(), "execution_strategy"),
        "execution_runtime": process_detail_value(process_detail.as_ref(), "execution_runtime"),
        "messages_count": messages_count,
        "history_limit": history_limit,
        "history_offset": history_offset,
        "history_cursor": {
            "oldest_message_id": oldest_message_id,
            "start_index": start_index,
            "loaded_count": loaded_count,
        },
        "history_truncated": loaded_count < messages_count,
        "messages": messages,
        "turns": turns,
        "items": items,
        "queued_turns": [],
        "artifacts": artifacts,
        "outputs": outputs,
        "thread_read": thread_read,
    });
    merge_process_detail_value(&mut detail, process_detail.as_ref(), "article_workspace");
    merge_process_detail_value(&mut detail, process_detail.as_ref(), "articleWorkspace");
    merge_process_thread_read_value(&mut detail, process_thread_read, "article_workspace");
    merge_process_thread_read_value(&mut detail, process_thread_read, "articleWorkspace");
    merge_process_thread_read_value(
        &mut detail,
        process_thread_read,
        "article_workspace_actions",
    );
    merge_process_thread_read_value(&mut detail, process_thread_read, "articleWorkspaceActions");
    Ok(detail)
}

fn is_turn_completed_usage_event(event: &AgentEvent) -> bool {
    event.event_type == "turn.completed"
        && event
            .payload
            .get("usage")
            .is_some_and(serde_json::Value::is_object)
}

fn projection_process_detail(
    stored: &StoredSession,
    projection: &ProjectionReadSession,
    active_turn_id: Option<&str>,
) -> Option<Value> {
    if projection.item_events.is_empty() {
        return None;
    }
    let process_stored = StoredSession {
        session: stored.session.clone(),
        turns: stored.turns.clone(),
        turn_inputs: HashMap::new(),
        turn_runtime_options: HashMap::new(),
        events: projection.item_events.clone(),
        output_blobs: HashMap::new(),
    };
    Some(read_model::runtime_session_read_detail_with_options(
        &process_stored,
        read_model::ReadDetailOptions::default(),
        &[],
        active_turn_id,
    ))
}

fn process_detail_array(detail: Option<&Value>, key: &str) -> Value {
    detail
        .and_then(|detail| detail.get(key))
        .filter(|value| value.is_array())
        .cloned()
        .unwrap_or_else(|| json!([]))
}

fn process_detail_value(detail: Option<&Value>, key: &str) -> Value {
    detail
        .and_then(|detail| detail.get(key))
        .cloned()
        .unwrap_or(Value::Null)
}

fn process_thread_read_array(
    thread_read: Option<&serde_json::Map<String, Value>>,
    key: &str,
) -> Value {
    process_thread_read_value(thread_read, key)
        .filter(|value| value.is_array())
        .unwrap_or_else(|| json!([]))
}

fn process_thread_read_value(
    thread_read: Option<&serde_json::Map<String, Value>>,
    key: &str,
) -> Option<Value> {
    thread_read
        .and_then(|thread_read| thread_read.get(key))
        .cloned()
}

fn active_turn_id_from_stored_turns(stored: &StoredSession) -> Option<Value> {
    stored
        .turns
        .iter()
        .rev()
        .find(|turn| super::status::agent_turn_blocks_queue_resume(turn.status))
        .map(|turn| json!(turn.turn_id))
}

fn merge_process_detail_value(detail: &mut Value, process_detail: Option<&Value>, key: &str) {
    let Some(value) = process_detail.and_then(|detail| detail.get(key)).cloned() else {
        return;
    };
    let Some(detail_object) = detail.as_object_mut() else {
        return;
    };
    detail_object.insert(key.to_string(), value);
}

fn merge_process_thread_read_value(
    detail: &mut Value,
    process_thread_read: Option<&serde_json::Map<String, Value>>,
    key: &str,
) {
    let Some(value) = process_thread_read
        .and_then(|thread_read| thread_read.get(key))
        .cloned()
    else {
        return;
    };
    let Some(thread_read) = detail.get_mut("thread_read").and_then(Value::as_object_mut) else {
        return;
    };
    thread_read.insert(key.to_string(), value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::AgentSession;
    use app_server_protocol::AgentSessionStatus;
    use app_server_protocol::AgentTurn;
    use app_server_protocol::AgentTurnStatus;

    #[test]
    fn active_turn_id_from_stored_turns_uses_latest_executing_turn() {
        let mut stored = StoredSession {
            session: AgentSession {
                session_id: "sess-active-turn".to_string(),
                thread_id: "thread-active-turn".to_string(),
                app_id: "agent-chat".to_string(),
                workspace_id: Some("workspace-current".to_string()),
                business_object_ref: None,
                status: AgentSessionStatus::Running,
                created_at: "2026-06-08T00:00:00.000Z".to_string(),
                updated_at: "2026-06-08T00:00:02.000Z".to_string(),
            },
            turns: vec![
                AgentTurn {
                    turn_id: "turn-completed".to_string(),
                    session_id: "sess-active-turn".to_string(),
                    thread_id: "thread-active-turn".to_string(),
                    status: AgentTurnStatus::Completed,
                    started_at: Some("2026-06-08T00:00:00.000Z".to_string()),
                    completed_at: Some("2026-06-08T00:00:01.000Z".to_string()),
                },
                AgentTurn {
                    turn_id: "turn-running".to_string(),
                    session_id: "sess-active-turn".to_string(),
                    thread_id: "thread-active-turn".to_string(),
                    status: AgentTurnStatus::Running,
                    started_at: Some("2026-06-08T00:00:02.000Z".to_string()),
                    completed_at: None,
                },
            ],
            turn_inputs: HashMap::new(),
            turn_runtime_options: HashMap::new(),
            events: Vec::new(),
            output_blobs: HashMap::new(),
        };

        assert_eq!(
            active_turn_id_from_stored_turns(&stored),
            Some(json!("turn-running"))
        );

        stored.turns[1].status = AgentTurnStatus::Queued;
        assert_eq!(active_turn_id_from_stored_turns(&stored), None);
    }
}
