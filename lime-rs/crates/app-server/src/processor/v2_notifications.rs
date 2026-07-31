use super::deprecated_agent_event_notification;
use super::thread::{project_event, ProjectedEvent};
use app_server_protocol::protocol::v2::{self, ServerNotification};
use app_server_protocol::{error_codes, AgentEvent, JsonRpcError, JsonRpcNotification};
use serde_json::Value;
use std::collections::HashSet;

mod command;
pub(crate) mod error;
mod file_change;
mod mcp;
mod plan;
mod thread_status;
mod turn_plan;
mod warning;

enum EventProjection {
    Direct(Vec<JsonRpcNotification>),
    SideChannel,
    Reject(JsonRpcError),
}

#[derive(Default)]
pub(crate) struct V2NotificationProjector {
    started_turn_ids: HashSet<String>,
    model_reroute_turn_ids: HashSet<String>,
    model_verification_turn_ids: HashSet<String>,
    started_plan_item_ids: HashSet<String>,
    completed_plan_item_ids: HashSet<String>,
    started_command_item_ids: HashSet<String>,
    completed_command_item_ids: HashSet<String>,
    started_file_change_item_ids: HashSet<String>,
    completed_file_change_item_ids: HashSet<String>,
    started_mcp_item_ids: HashSet<String>,
    completed_mcp_item_ids: HashSet<String>,
    terminal_error_turn_ids: HashSet<String>,
    thread_status: thread_status::ThreadStatusProjector,
}

impl V2NotificationProjector {
    pub(crate) fn project(
        &mut self,
        event: AgentEvent,
    ) -> Result<Vec<JsonRpcNotification>, JsonRpcError> {
        match self.classify(&event) {
            EventProjection::Direct(notifications) => Ok(notifications),
            EventProjection::SideChannel => Ok(vec![deprecated_agent_event_notification(event)]),
            EventProjection::Reject(error) => Err(error),
        }
    }

    fn classify(&mut self, event: &AgentEvent) -> EventProjection {
        let notification = match event.event_type.as_str() {
            "thread.created" | "thread.started" => return self.project_thread_started(event),
            "turn.accepted" => return EventProjection::Direct(Vec::new()),
            "turn.started" => return self.project_turn_started(event),
            "turn.completed" => {
                return self.project_turn_completed_with_usage(event, v2::TurnStatus::Completed)
            }
            "turn.failed" => return self.project_turn_failed(event),
            "turn.canceled" => {
                return self.project_turn_completed_with_usage(event, v2::TurnStatus::Interrupted)
            }
            "plugin_worker.retry" => return self.project_error(event, Some(true)),
            "runtime.error" => return self.project_error(event, None),
            "action.required" => {
                return EventProjection::Direct(
                    self.thread_status
                        .project_action_required(event)
                        .into_iter()
                        .map(Into::into)
                        .collect(),
                )
            }
            "action.resolved" | "action.canceled" | "action.cancelled" | "action.expired" => {
                return EventProjection::Direct(
                    self.thread_status
                        .project_action_terminal(event)
                        .into_iter()
                        .map(Into::into)
                        .collect(),
                )
            }
            "thread.goal.continuation" => return EventProjection::Direct(Vec::new()),
            "thread.settings.updated" => return self.project_thread_settings_updated(event),
            "provider.usage" => return self.project_token_usage(event),
            "item.started" | "command.started" => self.project_item(event, false),
            "item.completed" | "command.exited" => self.project_item(event, true),
            "context.compaction.started" => self.project_item(event, false),
            "context.compaction.completed" => self.project_item(event, true),
            "command.output" => {
                return command::project_output_delta(
                    &self.started_command_item_ids,
                    &self.completed_command_item_ids,
                    event,
                )
            }
            "patch.started" => {
                return file_change::project_started(
                    &mut self.started_file_change_item_ids,
                    &self.completed_file_change_item_ids,
                    event,
                )
            }
            "patch.applied" | "patch.failed" | "patch.declined" => {
                return file_change::project_final(
                    &self.started_file_change_item_ids,
                    &mut self.completed_file_change_item_ids,
                    event,
                )
            }
            "plan.delta" => {
                return plan::project_delta(
                    &mut self.started_plan_item_ids,
                    &self.completed_plan_item_ids,
                    event,
                )
            }
            "plan.final" => {
                return plan::project_final(
                    &mut self.started_plan_item_ids,
                    &mut self.completed_plan_item_ids,
                    event,
                )
            }
            "turn.plan.updated" => return turn_plan::project(event),
            "tool.progress" => {
                return mcp::project_progress(
                    &self.started_mcp_item_ids,
                    &self.completed_mcp_item_ids,
                    event,
                )
            }
            "message.delta" | "message.delta_batch" | "message.batch" => {
                self.project_agent_message_delta(event)
            }
            "reasoning.summary" => self.project_reasoning_summary_text_delta(event),
            "reasoning.summary_part_added" => self.project_reasoning_summary_part_added(event),
            "reasoning.delta" => self.project_reasoning_text_delta(event),
            "model.server_reported" => return EventProjection::Direct(Vec::new()),
            "model.rerouted" => return self.project_model_rerouted(event),
            "model.verification" => return self.project_model_verification(event),
            "provider_safety_buffering" => self.project_model_safety_buffering(event),
            "runtime.warning" => return warning::project(event),
            _ => return EventProjection::SideChannel,
        };
        match notification {
            Some(notification) => EventProjection::Direct(vec![notification.into()]),
            None => EventProjection::Reject(projection_error(event)),
        }
    }

    fn project_thread_started(&mut self, event: &AgentEvent) -> EventProjection {
        let ProjectedEvent::Thread(thread) = (match project_event(event) {
            Some(projected) => projected,
            None => return EventProjection::Reject(projection_error(event)),
        }) else {
            return EventProjection::Reject(projection_error(event));
        };
        self.thread_status.note_thread_started(&thread.id);
        EventProjection::Direct(vec![ServerNotification::ThreadStarted(
            v2::ThreadStartedNotification { thread },
        )
        .into()])
    }

    fn project_thread_settings_updated(&self, event: &AgentEvent) -> EventProjection {
        let Some(thread_id) = required_event_id(event.thread_id.as_deref()) else {
            return EventProjection::Reject(projection_error(event));
        };
        let Some(settings) = event.payload.get("threadSettings") else {
            return EventProjection::Reject(projection_error(event));
        };
        let Ok(thread_settings) = serde_json::from_value(settings.clone()) else {
            return EventProjection::Reject(projection_error(event));
        };
        EventProjection::Direct(vec![ServerNotification::ThreadSettingsUpdated(
            v2::ThreadSettingsUpdatedNotification {
                thread_id,
                thread_settings,
            },
        )
        .into()])
    }

    fn project_turn_started(&mut self, event: &AgentEvent) -> EventProjection {
        let Some((thread_id, turn_id, turn)) = project_turn(event, v2::TurnStatus::InProgress)
        else {
            return EventProjection::Reject(projection_error(event));
        };
        if !self.started_turn_ids.insert(turn_id) {
            return EventProjection::Direct(Vec::new());
        }
        let mut notifications = Vec::with_capacity(2);
        if let Some(status) = self.thread_status.project_turn_started(&thread_id) {
            notifications.push(status.into());
        }
        notifications.push(
            ServerNotification::TurnStarted(v2::TurnStartedNotification { thread_id, turn }).into(),
        );
        EventProjection::Direct(notifications)
    }

    fn project_turn_completed(
        &self,
        event: &AgentEvent,
        expected_status: v2::TurnStatus,
    ) -> Option<ServerNotification> {
        let (thread_id, _, turn) = project_turn(event, expected_status)?;
        Some(ServerNotification::TurnCompleted(
            v2::TurnCompletedNotification { thread_id, turn },
        ))
    }

    fn project_turn_completed_with_usage(
        &mut self,
        event: &AgentEvent,
        expected_status: v2::TurnStatus,
    ) -> EventProjection {
        let Some(turn_notification) = self.project_turn_completed(event, expected_status) else {
            return EventProjection::Reject(projection_error(event));
        };
        let ServerNotification::TurnCompleted(v2::TurnCompletedNotification {
            ref thread_id, ..
        }) = turn_notification
        else {
            unreachable!("turn completion projector returned a non-turn notification")
        };
        let mut notifications = Vec::with_capacity(3);
        if let Some(status) = self.thread_status.project_turn_terminal(thread_id) {
            notifications.push(status.into());
        }
        if let Some(usage_notification) = self.project_token_usage_notification(event) {
            notifications.push(usage_notification.into());
        }
        notifications.push(turn_notification.into());
        EventProjection::Direct(notifications)
    }

    fn project_error(
        &mut self,
        event: &AgentEvent,
        forced_will_retry: Option<bool>,
    ) -> EventProjection {
        let Some(projected) = error::project(event, forced_will_retry) else {
            return EventProjection::Reject(projection_error(event));
        };
        if !projected.will_retry
            && !self
                .terminal_error_turn_ids
                .insert(projected.turn_id.clone())
        {
            return EventProjection::Direct(Vec::new());
        }
        EventProjection::Direct(vec![projected.notification.into()])
    }

    fn project_turn_failed(&mut self, event: &AgentEvent) -> EventProjection {
        let Some(turn_id) = required_event_id(event.turn_id.as_deref()) else {
            return EventProjection::Reject(projection_error(event));
        };
        let completion = self.project_turn_completed_with_usage(event, v2::TurnStatus::Failed);
        let EventProjection::Direct(mut notifications) = completion else {
            return completion;
        };
        if self.terminal_error_turn_ids.remove(&turn_id) {
            return EventProjection::Direct(notifications);
        }
        let Some(projected) = error::project(event, Some(false)) else {
            return EventProjection::Reject(projection_error(event));
        };
        notifications.insert(0, projected.notification.into());
        EventProjection::Direct(notifications)
    }

    fn project_item(&mut self, event: &AgentEvent, completed: bool) -> Option<ServerNotification> {
        let thread_id = required_event_id(event.thread_id.as_deref())?;
        let turn_id = required_event_id(event.turn_id.as_deref())?;
        let item = match project_event(event)? {
            ProjectedEvent::Item(item) => item,
            _ => return None,
        };
        if let v2::ThreadItem::Plan { id, .. } = &item {
            if completed {
                self.completed_plan_item_ids.insert(id.clone());
            } else {
                self.started_plan_item_ids.insert(id.clone());
            }
        }
        if let v2::ThreadItem::CommandExecution { id, .. } = &item {
            if completed {
                self.completed_command_item_ids.insert(id.clone());
            } else {
                self.started_command_item_ids.insert(id.clone());
            }
        }
        if let v2::ThreadItem::McpToolCall { id, .. } = &item {
            if completed {
                self.completed_mcp_item_ids.insert(id.clone());
            } else {
                self.started_mcp_item_ids.insert(id.clone());
            }
        }
        let timestamp_ms = timestamp_millis(&event.timestamp)?;
        if completed {
            return Some(ServerNotification::ItemCompleted(
                v2::ItemCompletedNotification {
                    item,
                    thread_id,
                    turn_id,
                    completed_at_ms: timestamp_ms,
                },
            ));
        }
        Some(ServerNotification::ItemStarted(
            v2::ItemStartedNotification {
                item,
                thread_id,
                turn_id,
                started_at_ms: timestamp_ms,
            },
        ))
    }

    fn project_token_usage(&self, event: &AgentEvent) -> EventProjection {
        EventProjection::Direct(
            self.project_token_usage_notification(event)
                .into_iter()
                .map(Into::into)
                .collect(),
        )
    }

    fn project_token_usage_notification(&self, event: &AgentEvent) -> Option<ServerNotification> {
        let thread_id = required_event_id(event.thread_id.as_deref())?;
        let turn_id = required_event_id(event.turn_id.as_deref())?;
        let usage = event.payload.get("usage")?;
        let total = usage
            .get("total_token_usage")
            .and_then(project_token_usage_breakdown)?;
        let last = usage
            .get("last_token_usage")
            .and_then(project_token_usage_breakdown)?;
        let model_context_window = usage.get("model_context_window").and_then(Value::as_i64);

        Some(ServerNotification::ThreadTokenUsageUpdated(
            v2::ThreadTokenUsageUpdatedNotification {
                thread_id,
                turn_id,
                token_usage: v2::ThreadTokenUsage {
                    total,
                    last,
                    model_context_window,
                },
            },
        ))
    }

    fn project_model_safety_buffering(&self, event: &AgentEvent) -> Option<ServerNotification> {
        let thread_id = required_event_id(event.thread_id.as_deref())?;
        let turn_id = required_event_id(event.turn_id.as_deref())?;
        let model = payload_string(&event.payload, &["model"])?;
        let use_cases = payload_string_array(&event.payload, "useCases")?;
        let reasons = payload_string_array(&event.payload, "reasons")?;
        let show_buffering_ui = event.payload.get("showBufferingUi")?.as_bool()?;
        let faster_model = match event.payload.get("retryModel") {
            None | Some(Value::Null) => None,
            Some(value) => Some(
                value
                    .as_str()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())?
                    .to_string(),
            ),
        };

        Some(ServerNotification::ModelSafetyBufferingUpdated(
            v2::ModelSafetyBufferingUpdatedNotification {
                thread_id,
                turn_id,
                model,
                use_cases,
                reasons,
                show_buffering_ui,
                faster_model,
            },
        ))
    }

    fn project_model_verification(&mut self, event: &AgentEvent) -> EventProjection {
        let Some(thread_id) = required_event_id(event.thread_id.as_deref()) else {
            return EventProjection::Reject(projection_error(event));
        };
        let Some(turn_id) = required_event_id(event.turn_id.as_deref()) else {
            return EventProjection::Reject(projection_error(event));
        };
        let Some(values) = event.payload.get("verifications").and_then(Value::as_array) else {
            return EventProjection::Reject(projection_error(event));
        };
        let verifications = values
            .iter()
            .map(|value| match value.as_str() {
                Some("trusted_access_for_cyber") => {
                    Some(v2::ModelVerification::TrustedAccessForCyber)
                }
                _ => None,
            })
            .collect::<Option<Vec<_>>>();
        let Some(verifications) = verifications.filter(|values| !values.is_empty()) else {
            return EventProjection::Reject(projection_error(event));
        };
        if !self.model_verification_turn_ids.insert(turn_id.clone()) {
            return EventProjection::Direct(Vec::new());
        }
        EventProjection::Direct(vec![ServerNotification::ModelVerification(
            v2::ModelVerificationNotification {
                thread_id,
                turn_id,
                verifications,
            },
        )
        .into()])
    }

    fn project_model_rerouted(&mut self, event: &AgentEvent) -> EventProjection {
        let Some(thread_id) = required_event_id(event.thread_id.as_deref()) else {
            return EventProjection::Reject(projection_error(event));
        };
        let Some(turn_id) = required_event_id(event.turn_id.as_deref()) else {
            return EventProjection::Reject(projection_error(event));
        };
        let Some(from_model) = payload_string(&event.payload, &["from_model"]) else {
            return EventProjection::Reject(projection_error(event));
        };
        let Some(to_model) = payload_string(&event.payload, &["to_model"]) else {
            return EventProjection::Reject(projection_error(event));
        };
        let Some(reason) = event.payload.get("reason").and_then(Value::as_str) else {
            return EventProjection::Reject(projection_error(event));
        };
        let reason = match reason {
            "high_risk_cyber_activity" => v2::ModelRerouteReason::HighRiskCyberActivity,
            _ => return EventProjection::Reject(projection_error(event)),
        };
        if !self.model_reroute_turn_ids.insert(turn_id.clone()) {
            return EventProjection::Direct(Vec::new());
        }
        EventProjection::Direct(vec![ServerNotification::ModelRerouted(
            v2::ModelReroutedNotification {
                thread_id,
                turn_id,
                from_model,
                to_model,
                reason,
            },
        )
        .into()])
    }

    fn project_agent_message_delta(&self, event: &AgentEvent) -> Option<ServerNotification> {
        let thread_id = required_event_id(event.thread_id.as_deref())?;
        let turn_id = required_event_id(event.turn_id.as_deref())?;
        let projected_item_id = match project_event(event) {
            Some(ProjectedEvent::Item(v2::ThreadItem::AgentMessage { id, .. })) => Some(id),
            Some(ProjectedEvent::Item(_)) => return None,
            _ => None,
        };
        let payload_item_id = payload_string(
            &event.payload,
            &["itemId", "item_id", "messageId", "message_id", "id"],
        );
        if let (Some(projected), Some(payload)) =
            (projected_item_id.as_ref(), payload_item_id.as_ref())
        {
            let canonical_payload = agent_protocol::ItemId::new(payload.clone());
            if projected != canonical_payload.as_str() {
                return None;
            }
        }
        let item_id = projected_item_id.or(payload_item_id)?;
        let delta = text_from_payload(&event.payload)?;
        Some(ServerNotification::AgentMessageDelta(
            v2::AgentMessageDeltaNotification {
                thread_id,
                turn_id,
                item_id,
                delta,
            },
        ))
    }

    fn project_reasoning_summary_text_delta(
        &self,
        event: &AgentEvent,
    ) -> Option<ServerNotification> {
        let (thread_id, turn_id, item_id) = reasoning_identity(event)?;
        let delta = payload_string(&event.payload, &["summary"])?;
        let summary_index = payload_i64(&event.payload, "summaryIndex")?;
        Some(ServerNotification::ReasoningSummaryTextDelta(
            v2::ReasoningSummaryTextDeltaNotification {
                thread_id,
                turn_id,
                item_id,
                delta,
                summary_index,
            },
        ))
    }

    fn project_reasoning_summary_part_added(
        &self,
        event: &AgentEvent,
    ) -> Option<ServerNotification> {
        let (thread_id, turn_id, item_id) = reasoning_identity(event)?;
        let summary_index = payload_i64(&event.payload, "summaryIndex")?;
        Some(ServerNotification::ReasoningSummaryPartAdded(
            v2::ReasoningSummaryPartAddedNotification {
                thread_id,
                turn_id,
                item_id,
                summary_index,
            },
        ))
    }

    fn project_reasoning_text_delta(&self, event: &AgentEvent) -> Option<ServerNotification> {
        let (thread_id, turn_id, item_id) = reasoning_identity(event)?;
        let delta = payload_string(&event.payload, &["delta"])?;
        let content_index = payload_i64(&event.payload, "contentIndex")?;
        Some(ServerNotification::ReasoningTextDelta(
            v2::ReasoningTextDeltaNotification {
                thread_id,
                turn_id,
                item_id,
                delta,
                content_index,
            },
        ))
    }
}

pub(super) fn project_events(
    projector: &mut V2NotificationProjector,
    events: Vec<AgentEvent>,
) -> Result<Vec<JsonRpcNotification>, JsonRpcError> {
    let mut notifications = Vec::new();
    for event in events {
        notifications.extend(projector.project(event)?);
    }
    Ok(notifications)
}

fn project_turn(
    event: &AgentEvent,
    expected_status: v2::TurnStatus,
) -> Option<(String, String, v2::Turn)> {
    let thread_id = required_event_id(event.thread_id.as_deref())?;
    let turn_id = required_event_id(event.turn_id.as_deref())?;
    let turn = match project_event(event)? {
        ProjectedEvent::Turn(turn) => turn,
        _ => return None,
    };
    (turn.id == turn_id && turn.status == expected_status).then_some((thread_id, turn_id, turn))
}

fn required_event_id(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn payload_string(payload: &Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        payload
            .get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    })
}

fn payload_i64(payload: &Value, key: &str) -> Option<i64> {
    payload.get(key).and_then(Value::as_i64)
}

fn payload_string_array(payload: &Value, key: &str) -> Option<Vec<String>> {
    payload
        .get(key)?
        .as_array()?
        .iter()
        .map(|value| value.as_str().map(str::to_string))
        .collect()
}

fn reasoning_identity(event: &AgentEvent) -> Option<(String, String, String)> {
    Some((
        required_event_id(event.thread_id.as_deref())?,
        required_event_id(event.turn_id.as_deref())?,
        payload_string(&event.payload, &["itemId"])?,
    ))
}

fn text_from_payload(payload: &Value) -> Option<String> {
    if let Some(text) = payload.as_str().filter(|text| !text.is_empty()) {
        return Some(text.to_string());
    }
    if let Some(text) = payload_string(
        payload,
        &[
            "text",
            "delta",
            "content",
            "message",
            "outputText",
            "output_text",
        ],
    ) {
        return Some(text);
    }
    for key in ["deltas", "messages", "items", "parts", "content"] {
        let Some(values) = payload.get(key).and_then(Value::as_array) else {
            continue;
        };
        let text = values
            .iter()
            .filter_map(text_from_payload)
            .collect::<String>();
        if !text.is_empty() {
            return Some(text);
        }
    }
    None
}

fn project_token_usage_breakdown(value: &Value) -> Option<v2::TokenUsageBreakdown> {
    Some(v2::TokenUsageBreakdown {
        total_tokens: value.get("total_tokens")?.as_i64()?,
        input_tokens: value.get("input_tokens")?.as_i64()?,
        cached_input_tokens: value.get("cached_input_tokens")?.as_i64()?,
        cache_write_input_tokens: value
            .get("cache_write_input_tokens")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        output_tokens: value.get("output_tokens")?.as_i64()?,
        reasoning_output_tokens: value.get("reasoning_output_tokens")?.as_i64()?,
    })
}

fn timestamp_millis(value: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|value| value.timestamp_millis())
}

fn projection_error(event: &AgentEvent) -> JsonRpcError {
    JsonRpcError::new(
        error_codes::RUNTIME_ERROR,
        format!(
            "recognized lifecycle event {} ({}) has no valid v2 projection",
            event.event_id, event.event_type
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn event(event_type: &str, payload: Value) -> AgentEvent {
        AgentEvent {
            event_id: format!("evt-{event_type}"),
            sequence: 1,
            session_id: "session-1".to_string(),
            thread_id: Some("thread-1".to_string()),
            turn_id: Some("turn-1".to_string()),
            event_type: event_type.to_string(),
            timestamp: "2026-07-19T00:00:01.000Z".to_string(),
            payload,
        }
    }

    fn canonical_turn(status: &str) -> Value {
        json!({
            "sessionId": "session-1",
            "threadId": "thread-1",
            "turnId": "turn-1",
            "status": status,
            "createdAtMs": 1,
            "updatedAtMs": 2,
            "startedAtMs": 1,
            "completedAtMs": (status != "inProgress").then_some(2),
            "items": [],
            "itemsView": "full"
        })
    }

    fn canonical_item(status: &str) -> Value {
        json!({
            "sessionId": "session-1",
            "threadId": "thread-1",
            "turnId": "turn-1",
            "itemId": "item-1",
            "sequence": 1,
            "ordinal": 1,
            "createdAtMs": 1,
            "updatedAtMs": 2,
            "completedAtMs": (status == "completed").then_some(2),
            "kind": "agentMessage",
            "status": status,
            "payload": {"type": "agentMessage", "text": "hello"},
            "metadata": {}
        })
    }

    fn canonical_plan_item(status: &str) -> Value {
        json!({
            "sessionId": "session-1",
            "threadId": "thread-1",
            "turnId": "turn-1",
            "itemId": "plan_turn-1_proposed_plan:1",
            "sequence": 1,
            "ordinal": 1,
            "createdAtMs": 1,
            "updatedAtMs": 2,
            "completedAtMs": (status == "completed").then_some(2),
            "kind": "plan",
            "status": status,
            "payload": {
                "type": "plan",
                "text": "- [ ] 验证计划通知",
                "revision_id": "proposed_plan:1",
                "source": "proposed_plan",
                "plan": [{"step": "验证计划通知", "status": "pending"}]
            },
            "metadata": {}
        })
    }

    fn canonical_reasoning_item(status: &str, summary: Vec<&str>, content: Vec<&str>) -> Value {
        json!({
            "sessionId": "session-1",
            "threadId": "thread-1",
            "turnId": "turn-1",
            "itemId": "reasoning-1",
            "sequence": 1,
            "ordinal": 1,
            "createdAtMs": 1,
            "updatedAtMs": 2,
            "completedAtMs": (status == "completed").then_some(2),
            "kind": "reasoning",
            "status": status,
            "payload": {
                "type": "reasoning",
                "summary": summary,
                "content": content
            },
            "metadata": {}
        })
    }

    fn canonical_command_item(status: &str) -> Value {
        json!({
            "sessionId": "session-1",
            "threadId": "thread-1",
            "turnId": "turn-1",
            "itemId": "shell-1",
            "sequence": 1,
            "ordinal": 1,
            "createdAtMs": 1,
            "updatedAtMs": 2,
            "completedAtMs": (status == "completed").then_some(2),
            "kind": "command",
            "status": status,
            "payload": {
                "type": "command",
                "command": "printf ready",
                "cwd": "/workspace",
                "output": (status == "completed").then_some("ready"),
                "exitCode": (status == "completed").then_some(0)
            },
            "metadata": {
                "commandExecutionSource": "userShell",
                "processId": "process-1",
                "durationMs": 42
            }
        })
    }

    fn canonical_file_change_item(status: &str) -> Value {
        let file_status = match status {
            "inProgress" => "proposed",
            "completed" => "applied",
            "declined" => "rejected",
            "failed" => "failed",
            _ => status,
        };
        json!({
            "sessionId": "session-1",
            "threadId": "thread-1",
            "turnId": "turn-1",
            "itemId": "item_patch-1",
            "sequence": 1,
            "ordinal": 1,
            "createdAtMs": 1,
            "updatedAtMs": 2,
            "completedAtMs": (status != "inProgress").then_some(2),
            "kind": "file",
            "status": if status == "inProgress" { "inProgress" } else { "completed" },
            "payload": {
                "type": "file",
                "changes": [
                    {
                        "path": "src/lib.rs",
                        "kind": { "type": "update", "move_path": "src/main.rs" },
                        "diff": "-old\n+new"
                    }
                ],
                "status": file_status
            },
            "metadata": {}
        })
    }

    fn canonical_thread() -> Value {
        json!({
            "sessionId": "session-1",
            "threadId": "thread-1",
            "status": {"type": "idle"},
            "createdAtMs": 1,
            "updatedAtMs": 2,
            "archived": false,
            "preview": "hello",
            "modelProvider": "openai",
            "metadata": {},
            "turns": [],
            "turnsView": "full"
        })
    }

    #[test]
    fn maps_thread_started_to_the_direct_v2_shape() {
        let mut projector = V2NotificationProjector::default();
        let notifications = projector
            .project(event(
                "thread.started",
                json!({"thread": canonical_thread()}),
            ))
            .expect("thread started");

        assert_eq!(notifications[0].method, "thread/started");
        assert_eq!(
            notifications[0].params.as_ref().expect("params")["thread"]["id"],
            "thread-1"
        );
    }

    #[test]
    fn maps_runtime_model_reconciliation_to_thread_settings_updated() {
        let mut projector = V2NotificationProjector::default();
        let notifications = projector
            .project(event(
                "thread.settings.updated",
                json!({
                    "threadSettings": {
                        "cwd": "",
                        "approvalPolicy": null,
                        "approvalsReviewer": null,
                        "sandboxPolicy": null,
                        "model": "model-b",
                        "modelProvider": "provider-b",
                        "collaborationMode": {
                            "mode": "default",
                            "settings": { "model": "model-b" }
                        }
                    }
                }),
            ))
            .expect("thread settings update");

        assert_eq!(notifications.len(), 1);
        assert_eq!(notifications[0].method, "thread/settings/updated");
        assert_eq!(
            notifications[0].params.as_ref().expect("settings params")["threadSettings"]["model"],
            "model-b"
        );
    }

    #[test]
    fn accepted_is_internal_and_started_emits_status_before_turn_started() {
        let mut projector = V2NotificationProjector::default();
        let accepted = projector
            .project(event(
                "turn.accepted",
                json!({"turn": canonical_turn("inProgress")}),
            ))
            .expect("accepted turn");
        let duplicate = projector
            .project(event(
                "turn.started",
                json!({"turn": canonical_turn("inProgress")}),
            ))
            .expect("started turn");

        assert!(accepted.is_empty());
        assert_eq!(duplicate.len(), 2);
        assert_eq!(duplicate[0].method, "thread/status/changed");
        assert_eq!(
            duplicate[0].params.as_ref().expect("status params")["status"],
            json!({"type": "active", "activeFlags": []})
        );
        assert_eq!(duplicate[1].method, "turn/started");
    }

    #[test]
    fn maps_item_and_terminal_lifecycle_to_direct_v2() {
        let cases = [
            (
                "turn.completed",
                json!({"turn": canonical_turn("completed")}),
                vec!["turn/completed"],
            ),
            (
                "turn.failed",
                json!({
                    "message": "provider failed",
                    "turn": canonical_turn("failed")
                }),
                vec!["error", "turn/completed"],
            ),
            (
                "turn.canceled",
                json!({"turn": canonical_turn("interrupted")}),
                vec!["turn/completed"],
            ),
            (
                "item.started",
                json!({"item": canonical_item("inProgress")}),
                vec!["item/started"],
            ),
            (
                "item.completed",
                json!({"item": canonical_item("completed")}),
                vec!["item/completed"],
            ),
        ];
        for (event_type, payload, methods) in cases {
            let mut projector = V2NotificationProjector::default();
            let notifications = projector
                .project(event(event_type, payload))
                .expect("direct lifecycle");
            assert_eq!(
                notifications
                    .iter()
                    .map(|notification| notification.method.as_str())
                    .collect::<Vec<_>>(),
                methods
            );
        }
    }

    #[test]
    fn maps_plan_to_one_started_typed_deltas_and_one_completed_notification() {
        let mut projector = V2NotificationProjector::default();
        let first = projector
            .project(event(
                "plan.delta",
                json!({
                    "delta": "- [ ] 读协议",
                    "item": canonical_plan_item("inProgress")
                }),
            ))
            .expect("first plan delta");
        let second = projector
            .project(event(
                "plan.delta",
                json!({
                    "delta": "\n- [ ] 接 GUI",
                    "item": canonical_plan_item("inProgress")
                }),
            ))
            .expect("second plan delta");
        let completed = projector
            .project(event(
                "plan.final",
                json!({"item": canonical_plan_item("completed")}),
            ))
            .expect("plan completed");

        assert_eq!(
            first
                .iter()
                .map(|notification| notification.method.as_str())
                .collect::<Vec<_>>(),
            vec!["item/started", "item/plan/delta"]
        );
        assert_eq!(second[0].method, "item/plan/delta");
        assert_eq!(
            second[0].params.as_ref().expect("delta params")["delta"],
            "\n- [ ] 接 GUI"
        );
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].method, "item/completed");
    }

    #[test]
    fn rejects_plan_delta_after_completed_item() {
        let mut projector = V2NotificationProjector::default();
        projector
            .project(event(
                "plan.final",
                json!({"item": canonical_plan_item("completed")}),
            ))
            .expect("plan completed");

        let error = projector
            .project(event(
                "plan.delta",
                json!({
                    "delta": "late",
                    "item": canonical_plan_item("inProgress")
                }),
            ))
            .expect_err("late plan delta must fail closed");
        assert_eq!(error.code, error_codes::RUNTIME_ERROR);
        assert!(error.message.contains("plan.delta"));
    }

    #[test]
    fn maps_terminal_usage_to_direct_v2_notification_without_context_window() {
        let mut projector = V2NotificationProjector::default();
        let notifications = projector
            .project(event(
                "turn.completed",
                json!({
                    "turn": canonical_turn("completed"),
                    "usage": {
                        "total_token_usage": {
                            "total_tokens": 31_000,
                            "input_tokens": 31_000,
                            "cached_input_tokens": 0,
                            "output_tokens": 0,
                            "reasoning_output_tokens": 0
                        },
                        "last_token_usage": {
                            "total_tokens": 31_000,
                            "input_tokens": 31_000,
                            "cached_input_tokens": 0,
                            "output_tokens": 0,
                            "reasoning_output_tokens": 0
                        }
                    }
                }),
            ))
            .expect("terminal usage");

        assert_eq!(notifications.len(), 2);
        assert_eq!(notifications[0].method, "thread/tokenUsage/updated");
        assert_eq!(notifications[1].method, "turn/completed");
        let params = notifications[0].params.as_ref().expect("usage params");
        assert_eq!(params["tokenUsage"]["last"]["inputTokens"], 31_000);
        assert_eq!(params["tokenUsage"]["modelContextWindow"], Value::Null);
    }

    #[test]
    fn maps_command_lifecycle_to_direct_v2_item_notifications() {
        let mut projector = V2NotificationProjector::default();
        let started = projector
            .project(event(
                "command.started",
                json!({"item": canonical_command_item("inProgress")}),
            ))
            .expect("command started");
        let completed = projector
            .project(event(
                "command.exited",
                json!({"item": canonical_command_item("completed")}),
            ))
            .expect("command exited");

        assert_eq!(started[0].method, "item/started");
        assert_eq!(completed[0].method, "item/completed");
        assert_eq!(
            completed[0].params.as_ref().expect("completed params")["item"]["source"],
            "userShell"
        );
    }

    #[test]
    fn maps_command_output_to_typed_delta_between_item_lifecycle_events() {
        let mut projector = V2NotificationProjector::default();
        assert!(
            projector
                .project(event(
                    "command.output",
                    json!({
                        "commandId": "shell-1",
                        "delta": "stdout\n",
                        "item": canonical_command_item("inProgress")
                    }),
                ))
                .is_err(),
            "output before start must fail closed"
        );

        projector
            .project(event(
                "command.started",
                json!({"item": canonical_command_item("inProgress")}),
            ))
            .expect("command started");
        let output = projector
            .project(event(
                "command.output",
                json!({
                    "commandId": "shell-1",
                    "delta": "stdout\n",
                    "item": canonical_command_item("inProgress")
                }),
            ))
            .expect("command output");
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].method, "item/commandExecution/outputDelta");
        assert_eq!(
            output[0].params.as_ref().expect("output params"),
            &json!({
                "threadId": "thread-1",
                "turnId": "turn-1",
                "itemId": "shell-1",
                "delta": "stdout\n"
            })
        );

        projector
            .project(event(
                "command.exited",
                json!({"item": canonical_command_item("completed")}),
            ))
            .expect("command exited");
        assert!(
            projector
                .project(event(
                    "command.output",
                    json!({
                        "commandId": "shell-1",
                        "delta": "late",
                        "item": canonical_command_item("inProgress")
                    }),
                ))
                .is_err(),
            "late command output must fail closed"
        );
    }

    #[test]
    fn maps_file_change_to_started_patch_updated_and_completed() {
        let mut projector = V2NotificationProjector::default();
        let started = projector
            .project(event(
                "patch.started",
                json!({
                    "patchId": "patch-1",
                    "item": canonical_file_change_item("inProgress")
                }),
            ))
            .expect("file change started");
        let completed = projector
            .project(event(
                "patch.applied",
                json!({
                    "patchId": "patch-1",
                    "item": canonical_file_change_item("completed")
                }),
            ))
            .expect("file change completed");

        assert_eq!(
            started
                .iter()
                .map(|notification| notification.method.as_str())
                .collect::<Vec<_>>(),
            vec!["item/started", "item/fileChange/patchUpdated"]
        );
        assert_eq!(
            started[1].params.as_ref().expect("patch params"),
            &json!({
                "threadId": "thread-1",
                "turnId": "turn-1",
                "itemId": "item_patch-1",
                "changes": [{
                    "path": "src/lib.rs",
                    "kind": { "type": "update", "move_path": "src/main.rs" },
                    "diff": "-old\n+new"
                }]
            })
        );
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].method, "item/completed");
        assert_eq!(
            completed[0].params.as_ref().expect("completed params")["item"]["id"],
            "item_patch-1"
        );
    }

    #[test]
    fn file_change_lifecycle_fails_closed_outside_started_terminal_order() {
        let mut projector = V2NotificationProjector::default();
        let terminal_before_start = projector.project(event(
            "patch.failed",
            json!({"item": canonical_file_change_item("failed")}),
        ));
        assert!(terminal_before_start.is_err());

        projector
            .project(event(
                "patch.started",
                json!({"item": canonical_file_change_item("inProgress")}),
            ))
            .expect("file change started");
        assert!(projector
            .project(event(
                "patch.started",
                json!({"item": canonical_file_change_item("inProgress")}),
            ))
            .is_err());
        projector
            .project(event(
                "patch.declined",
                json!({"item": canonical_file_change_item("declined")}),
            ))
            .expect("file change declined");
        assert!(projector
            .project(event(
                "patch.applied",
                json!({"item": canonical_file_change_item("completed")}),
            ))
            .is_err());
    }

    #[test]
    fn file_change_with_empty_snapshot_does_not_invent_patch_update() {
        let mut item = canonical_file_change_item("inProgress");
        item["payload"]["changes"] = json!([]);
        let mut projector = V2NotificationProjector::default();

        let notifications = projector
            .project(event("patch.started", json!({"item": item})))
            .expect("empty file change snapshot");

        assert_eq!(notifications.len(), 1);
        assert_eq!(notifications[0].method, "item/started");
    }

    #[test]
    fn delta_accepts_the_real_outer_item_identity_shape() {
        let mut projector = V2NotificationProjector::default();
        let notifications = projector
            .project(event(
                "message.delta",
                json!({"itemId": "item-1", "text": "hello"}),
            ))
            .expect("direct delta");

        assert_eq!(notifications[0].method, "item/agentMessage/delta");
        let params = notifications[0].params.as_ref().expect("delta params");
        assert_eq!(params["itemId"], "item-1");
        assert_eq!(params["delta"], "hello");
    }

    #[test]
    fn delta_compares_outer_identity_after_canonical_item_normalization() {
        let mut item = canonical_item("inProgress");
        item["itemId"] = json!("item_assistant-1");
        let mut projector = V2NotificationProjector::default();
        let notifications = projector
            .project(event(
                "message.delta",
                json!({
                    "itemId": "assistant-1",
                    "item": item,
                    "text": "hello"
                }),
            ))
            .expect("canonicalized direct delta");

        let params = notifications[0].params.as_ref().expect("delta params");
        assert_eq!(params["itemId"], "item_assistant-1");
        assert_eq!(params["delta"], "hello");
    }

    #[test]
    fn delta_rejects_real_outer_and_canonical_item_identity_drift() {
        let mut item = canonical_item("inProgress");
        item["itemId"] = json!("item_assistant-1");
        let mut projector = V2NotificationProjector::default();
        let error = projector
            .project(event(
                "message.delta",
                json!({
                    "itemId": "assistant-2",
                    "item": item,
                    "text": "hello"
                }),
            ))
            .expect_err("identity drift must fail closed");

        assert_eq!(error.code, error_codes::RUNTIME_ERROR);
        assert!(error.message.contains("message.delta"));
    }

    #[test]
    fn maps_indexed_reasoning_notifications_in_codex_order() {
        let mut projector = V2NotificationProjector::default();
        let events = [
            event(
                "item.started",
                json!({"item": canonical_reasoning_item("inProgress", vec![], vec![])}),
            ),
            event(
                "reasoning.summary",
                json!({
                    "itemId": "reasoning-1",
                    "summary": "first summary",
                    "summaryIndex": 0
                }),
            ),
            event(
                "reasoning.summary_part_added",
                json!({"itemId": "reasoning-1", "summaryIndex": 1}),
            ),
            event(
                "reasoning.delta",
                json!({
                    "itemId": "reasoning-1",
                    "delta": "raw reasoning",
                    "contentIndex": 0
                }),
            ),
            event(
                "item.completed",
                json!({
                    "item": canonical_reasoning_item(
                        "completed",
                        vec!["first summary", "second summary"],
                        vec!["raw reasoning"]
                    )
                }),
            ),
        ];

        let notifications = events
            .into_iter()
            .flat_map(|event| projector.project(event).expect("reasoning projection"))
            .collect::<Vec<_>>();

        assert_eq!(
            notifications
                .iter()
                .map(|notification| notification.method.as_str())
                .collect::<Vec<_>>(),
            [
                "item/started",
                "item/reasoning/summaryTextDelta",
                "item/reasoning/summaryPartAdded",
                "item/reasoning/textDelta",
                "item/completed",
            ]
        );
        assert_eq!(
            notifications[1].params.as_ref().expect("summary params"),
            &json!({
                "threadId": "thread-1",
                "turnId": "turn-1",
                "itemId": "reasoning-1",
                "delta": "first summary",
                "summaryIndex": 0
            })
        );
        assert_eq!(
            notifications[2].params.as_ref().expect("part params")["summaryIndex"],
            1
        );
        assert_eq!(
            notifications[3].params.as_ref().expect("raw params")["contentIndex"],
            0
        );
        assert_eq!(
            notifications[4].params.as_ref().expect("completed params")["item"]["type"],
            "reasoning"
        );
    }

    #[test]
    fn malformed_reasoning_notification_is_rejected_without_wrapper_fallback() {
        let mut projector = V2NotificationProjector::default();
        let error = projector
            .project(event(
                "reasoning.summary",
                json!({"itemId": "reasoning-1", "summary": "missing index"}),
            ))
            .expect_err("missing summary index must reject");

        assert_eq!(error.code, error_codes::RUNTIME_ERROR);
        assert!(error.message.contains("reasoning.summary"));
    }

    #[test]
    fn side_channel_keeps_the_deprecated_envelope() {
        let mut projector = V2NotificationProjector::default();
        let notifications = projector
            .project(event("provider.request.started", json!({})))
            .expect("side channel");
        assert_eq!(notifications[0].method, "agentSession/event");
    }

    #[test]
    fn maps_provider_safety_buffering_to_direct_codex_notification() {
        let mut projector = V2NotificationProjector::default();
        let notifications = projector
            .project(event(
                "provider_safety_buffering",
                json!({
                    "provider": "openai",
                    "model": "gpt-5-codex",
                    "useCases": ["policy"],
                    "reasons": ["buffering"],
                    "showBufferingUi": true,
                    "retryModel": "gpt-5-mini"
                }),
            ))
            .expect("direct safety buffering notification");

        assert_eq!(notifications.len(), 1);
        assert_eq!(notifications[0].method, "model/safetyBuffering/updated");
        assert_eq!(
            notifications[0]
                .params
                .as_ref()
                .expect("notification params"),
            &json!({
                "threadId": "thread-1",
                "turnId": "turn-1",
                "model": "gpt-5-codex",
                "useCases": ["policy"],
                "reasons": ["buffering"],
                "showBufferingUi": true,
                "fasterModel": "gpt-5-mini"
            })
        );
    }

    #[test]
    fn maps_model_reroute_once_per_turn_to_direct_codex_notification() {
        let mut projector = V2NotificationProjector::default();
        let model_event = event(
            "model.rerouted",
            json!({
                "from_model": "gpt-5-codex",
                "to_model": "gpt-5.1-codex",
                "reason": "high_risk_cyber_activity"
            }),
        );
        let notifications = projector
            .project(model_event.clone())
            .expect("direct model reroute notification");

        assert_eq!(notifications.len(), 1);
        assert_eq!(notifications[0].method, "model/rerouted");
        assert_eq!(
            notifications[0].params.as_ref().expect("reroute params"),
            &json!({
                "threadId": "thread-1",
                "turnId": "turn-1",
                "fromModel": "gpt-5-codex",
                "toModel": "gpt-5.1-codex",
                "reason": "highRiskCyberActivity"
            })
        );
        assert!(projector
            .project(model_event)
            .expect("duplicate reroute is ignored")
            .is_empty());
    }

    #[test]
    fn maps_model_verification_once_per_turn_to_direct_codex_notification() {
        let mut projector = V2NotificationProjector::default();
        let model_event = event(
            "model.verification",
            json!({"verifications": ["trusted_access_for_cyber"]}),
        );
        let notifications = projector
            .project(model_event.clone())
            .expect("direct model verification notification");

        assert_eq!(notifications.len(), 1);
        assert_eq!(notifications[0].method, "model/verification");
        assert_eq!(
            notifications[0]
                .params
                .as_ref()
                .expect("verification params"),
            &json!({
                "threadId": "thread-1",
                "turnId": "turn-1",
                "verifications": ["trustedAccessForCyber"]
            })
        );
        assert!(projector
            .project(model_event)
            .expect("duplicate verification is ignored")
            .is_empty());
    }

    #[test]
    fn model_verification_and_server_model_fail_closed_at_v2_boundary() {
        for payload in [
            json!({}),
            json!({
                "from_model": "",
                "to_model": "gpt-5.1-codex",
                "reason": "high_risk_cyber_activity"
            }),
            json!({
                "from_model": "gpt-5-codex",
                "to_model": "gpt-5.1-codex",
                "reason": "unknown"
            }),
        ] {
            let error = V2NotificationProjector::default()
                .project(event("model.rerouted", payload))
                .expect_err("malformed reroute must fail closed");
            assert_eq!(error.code, error_codes::RUNTIME_ERROR);
            assert!(error.message.contains("model.rerouted"));
        }

        for payload in [
            json!({}),
            json!({"verifications": []}),
            json!({"verifications": ["unknown"]}),
            json!({"verifications": "trusted_access_for_cyber"}),
        ] {
            let error = V2NotificationProjector::default()
                .project(event("model.verification", payload))
                .expect_err("malformed verification must fail closed");
            assert_eq!(error.code, error_codes::RUNTIME_ERROR);
            assert!(error.message.contains("model.verification"));
        }

        let mut missing_identity = event(
            "model.verification",
            json!({"verifications": ["trusted_access_for_cyber"]}),
        );
        missing_identity.turn_id = None;
        assert!(V2NotificationProjector::default()
            .project(missing_identity)
            .is_err());

        assert!(V2NotificationProjector::default()
            .project(event(
                "model.server_reported",
                json!({"model": "gpt-5-codex"}),
            ))
            .expect("server model is diagnostic-only")
            .is_empty());
    }

    #[test]
    fn malformed_provider_safety_buffering_is_rejected_without_side_channel_fallback() {
        let invalid_payloads = [
            json!({
                "useCases": ["policy"],
                "reasons": ["buffering"],
                "showBufferingUi": true
            }),
            json!({
                "model": "gpt-5-codex",
                "useCases": ["policy", 1],
                "reasons": ["buffering"],
                "showBufferingUi": true
            }),
            json!({
                "model": "gpt-5-codex",
                "useCases": ["policy"],
                "reasons": "buffering",
                "showBufferingUi": true
            }),
        ];

        for payload in invalid_payloads {
            let mut projector = V2NotificationProjector::default();
            let error = projector
                .project(event("provider_safety_buffering", payload))
                .expect_err("malformed safety buffering must fail closed");
            assert_eq!(error.code, error_codes::RUNTIME_ERROR);
            assert!(error.message.contains("provider_safety_buffering"));
        }

        let mut missing_identity = event(
            "provider_safety_buffering",
            json!({
                "model": "gpt-5-codex",
                "useCases": [],
                "reasons": [],
                "showBufferingUi": false
            }),
        );
        missing_identity.turn_id = None;
        assert!(V2NotificationProjector::default()
            .project(missing_identity)
            .is_err());
    }

    #[test]
    fn action_lifecycle_is_internal_to_typed_server_requests() {
        for event_type in [
            "action.required",
            "action.resolved",
            "action.canceled",
            "action.cancelled",
            "action.expired",
        ] {
            let mut projector = V2NotificationProjector::default();
            let notifications = projector
                .project(event(event_type, json!({})))
                .expect("internal action lifecycle");
            assert!(notifications.is_empty(), "{event_type}");
        }
    }

    #[test]
    fn thread_goal_continuation_context_is_not_sent_to_clients() {
        let mut projector = V2NotificationProjector::default();
        let notifications = projector
            .project(event(
                "thread.goal.continuation",
                json!({"input": [{"type": "text", "text": "internal objective"}]}),
            ))
            .expect("internal thread goal continuation");

        assert!(notifications.is_empty());
    }

    #[test]
    fn maps_canonical_provider_usage_to_direct_v2_notification() {
        let mut projector = V2NotificationProjector::default();
        let notifications = projector
            .project(event(
                "provider.usage",
                json!({
                    "usage": {
                        "total_token_usage": {
                            "total_tokens": 31_000,
                            "input_tokens": 31_000,
                            "cached_input_tokens": 0,
                            "cache_write_input_tokens": 12,
                            "output_tokens": 0,
                            "reasoning_output_tokens": 0
                        },
                        "last_token_usage": {
                            "total_tokens": 31_000,
                            "input_tokens": 31_000,
                            "cached_input_tokens": 0,
                            "cache_write_input_tokens": 12,
                            "output_tokens": 0,
                            "reasoning_output_tokens": 0
                        },
                        "model_context_window": 128_000
                    }
                }),
            ))
            .expect("provider usage");

        assert_eq!(notifications[0].method, "thread/tokenUsage/updated");
        let params = notifications[0].params.as_ref().expect("usage params");
        assert_eq!(params["threadId"], "thread-1");
        assert_eq!(params["turnId"], "turn-1");
        assert_eq!(params["tokenUsage"]["last"]["inputTokens"], 31_000);
        assert_eq!(params["tokenUsage"]["last"]["cacheWriteInputTokens"], 12);
    }

    #[test]
    fn malformed_recognized_lifecycle_is_rejected_without_wrapper_fallback() {
        let mut projector = V2NotificationProjector::default();
        let error = projector
            .project(event("item.completed", json!({})))
            .expect_err("malformed lifecycle must reject");
        assert_eq!(error.code, error_codes::RUNTIME_ERROR);
        assert!(error.message.contains("item.completed"));
    }
}
