use super::RuntimeEvent;
use agent_protocol::AgentInput;
use app_server_protocol::AgentEvent;
use serde_json::json;
use serde_json::Value;
use std::collections::HashMap;

pub(super) const TURN_INPUT_EVENT_TYPE: &str = "message.created";
pub(super) const QUEUE_INPUT_ADDED_EVENT_TYPE: &str = "queue.added";
pub(super) const QUEUE_INPUT_UPDATED_EVENT_TYPE: &str = "queue.updated";
pub(super) const THREAD_GOAL_CONTINUATION_EVENT_TYPE: &str = "thread.goal.continuation";
pub(super) const REVIEW_INPUT_EVENT_TYPE: &str = "review.input";

pub(super) fn runtime_event_for_turn_input(input: &[AgentInput]) -> Option<super::RuntimeEvent> {
    runtime_event_for_turn_input_with_client_id(input, None)
}

pub(super) fn runtime_event_for_turn_input_with_client_id(
    input: &[AgentInput],
    client_id: Option<&str>,
) -> Option<super::RuntimeEvent> {
    runtime_event_for_turn_input_kind(
        TURN_INPUT_EVENT_TYPE,
        input,
        client_id,
        "user_visible",
        None,
    )
}

pub(super) fn runtime_event_for_goal_continuation(
    input: &[AgentInput],
) -> Option<super::RuntimeEvent> {
    runtime_event_for_turn_input_kind(
        THREAD_GOAL_CONTINUATION_EVENT_TYPE,
        input,
        None,
        "agent_only",
        Some("thread_goal"),
    )
}

pub(super) fn runtime_event_for_review(input: &[AgentInput]) -> Option<super::RuntimeEvent> {
    runtime_event_for_turn_input_kind(
        REVIEW_INPUT_EVENT_TYPE,
        input,
        None,
        "agent_only",
        Some("review"),
    )
}

fn runtime_event_for_turn_input_kind(
    event_type: &str,
    input: &[AgentInput],
    client_id: Option<&str>,
    visibility: &str,
    source: Option<&str>,
) -> Option<super::RuntimeEvent> {
    if super::turn_start::validate_user_input(input).is_err() {
        return None;
    }
    let text = super::turn_start::user_input_text(input);
    let mut payload = json!({
        "role": "user",
        "visibility": visibility,
        "input": input,
        "content": {
            "kind": "inline_text",
            "text": text,
        },
    });
    if let Some(source) = source {
        payload["source"] = Value::String(source.to_string());
    }
    if let Some(client_id) = client_id
        .map(str::trim)
        .filter(|client_id| !client_id.is_empty())
    {
        payload["clientId"] = Value::String(client_id.to_string());
    }
    Some(super::RuntimeEvent::new(event_type, payload))
}

pub(super) fn turn_inputs_from_events(events: &[AgentEvent]) -> HashMap<String, Vec<AgentInput>> {
    events
        .iter()
        .filter(|event| is_turn_input_event_type(&event.event_type))
        .filter_map(turn_input_from_event)
        .collect()
}

pub(super) fn is_turn_input_event(event: &AgentEvent) -> bool {
    is_turn_input_event_type(&event.event_type)
        && event
            .payload
            .get("mailbox")
            .is_none_or(|mailbox| mailbox["turnInput"] == true)
}

pub(super) fn is_provider_input_event(event: &AgentEvent) -> bool {
    matches!(
        event.event_type.as_str(),
        TURN_INPUT_EVENT_TYPE
            | QUEUE_INPUT_UPDATED_EVENT_TYPE
            | THREAD_GOAL_CONTINUATION_EVENT_TYPE
            | REVIEW_INPUT_EVENT_TYPE
    ) && event
        .payload
        .get("mailbox")
        .is_none_or(|mailbox| mailbox["turnInput"] == true)
}

pub(super) fn is_turn_input_event_type(event_type: &str) -> bool {
    matches!(
        event_type,
        TURN_INPUT_EVENT_TYPE | QUEUE_INPUT_ADDED_EVENT_TYPE | QUEUE_INPUT_UPDATED_EVENT_TYPE
    )
}

pub(super) fn runtime_event_is_provider_input(event: &RuntimeEvent) -> bool {
    matches!(
        event.event_type.as_str(),
        TURN_INPUT_EVENT_TYPE
            | QUEUE_INPUT_UPDATED_EVENT_TYPE
            | THREAD_GOAL_CONTINUATION_EVENT_TYPE
            | REVIEW_INPUT_EVENT_TYPE
    ) && event
        .payload
        .get("mailbox")
        .is_none_or(|mailbox| mailbox["turnInput"] == true)
}

fn turn_input_from_event(event: &AgentEvent) -> Option<(String, Vec<AgentInput>)> {
    let turn_id = event.turn_id.clone().or_else(|| {
        event
            .payload
            .get("queuedTurnId")
            .or_else(|| event.payload.get("queuedSubmissionId"))
            .and_then(Value::as_str)
            .map(str::to_string)
    })?;
    if let Some(input) = event
        .payload
        .get("input")
        .and_then(|value| serde_json::from_value::<Vec<AgentInput>>(value.clone()).ok())
    {
        return Some((turn_id, input));
    }

    let text = turn_input_text(&event.payload)?;
    Some((turn_id, vec![AgentInput::text(text)]))
}

fn turn_input_text(payload: &Value) -> Option<String> {
    payload
        .get("content")
        .and_then(|content| {
            content
                .get("text")
                .or_else(|| content.get("message"))
                .and_then(Value::as_str)
        })
        .or_else(|| payload.get("text").and_then(Value::as_str))
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(ToString::to_string)
}
