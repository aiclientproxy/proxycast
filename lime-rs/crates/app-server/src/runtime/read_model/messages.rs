use super::super::raw_string_field;
use super::super::timestamp_seconds;
use super::super::turn_input_events;
use super::super::StoredSession;
use agent_protocol::AgentInput;
use app_server_protocol::AgentEvent;
use app_server_protocol::AgentTurn;
use serde_json::{json, Value};

pub(super) fn message_numeric_id(message: &serde_json::Value) -> Option<i64> {
    message.get("id").and_then(|value| match value {
        serde_json::Value::Number(number) => number.as_i64(),
        serde_json::Value::String(value) => value.parse::<i64>().ok(),
        _ => None,
    })
}

pub(in crate::runtime) fn runtime_session_messages(
    stored: &StoredSession,
) -> Vec<serde_json::Value> {
    let mut messages = Vec::new();
    for turn in &stored.turns {
        let input_event_payload = turn_input_event_payload(&stored.events, &turn.turn_id);
        let input = stored
            .turn_inputs
            .get(&turn.turn_id)
            .cloned()
            .or_else(|| turn_input_from_events(&stored.events, &turn.turn_id));
        if let Some(input) = input.as_ref() {
            if let Some(message) = runtime_user_message_from_turn(turn, input, input_event_payload)
            {
                messages.push(message);
            }
        }
        if let Some(message) = runtime_assistant_message_from_events(turn, &stored.events) {
            messages.push(message);
        }
    }
    messages
}

fn turn_input_event_payload<'a>(events: &'a [AgentEvent], turn_id: &str) -> Option<&'a Value> {
    events
        .iter()
        .rev()
        .find(|event| {
            event.turn_id.as_deref() == Some(turn_id)
                && turn_input_events::is_turn_input_event(event)
        })
        .map(|event| &event.payload)
}

pub(super) fn turn_input_from_events(
    events: &[app_server_protocol::AgentEvent],
    turn_id: &str,
) -> Option<Vec<AgentInput>> {
    events
        .iter()
        .rev()
        .find(|event| {
            event.turn_id.as_deref() == Some(turn_id)
                && turn_input_events::is_turn_input_event(event)
        })
        .and_then(|event| event.payload.get("input"))
        .and_then(|value| serde_json::from_value(value.clone()).ok())
        .or_else(|| {
            events
                .iter()
                .rev()
                .find(|event| {
                    event.turn_id.as_deref() == Some(turn_id)
                        && turn_input_events::is_turn_input_event(event)
                })
                .and_then(|event| {
                    event
                        .payload
                        .get("content")
                        .and_then(|content| content.get("text").or_else(|| content.get("message")))
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_string)
                        .filter(|text| !text.trim().is_empty())
                        .map(|text| vec![AgentInput::text(text)])
                })
        })
}

fn runtime_user_message_from_turn(
    turn: &AgentTurn,
    input: &[AgentInput],
    input_event_payload: Option<&Value>,
) -> Option<serde_json::Value> {
    let text = input
        .iter()
        .filter_map(|part| match part {
            AgentInput::Text { text, .. } => Some(text.as_str()),
            AgentInput::Image { .. }
            | AgentInput::LocalImage { .. }
            | AgentInput::Skill { .. }
            | AgentInput::Mention { .. } => None,
        })
        .collect::<String>();
    let text = text.trim();
    let typed_attachments = input
        .iter()
        .filter_map(|part| match part {
            AgentInput::Image { uri, detail } => Some(json!({
                "kind": "image",
                "uri": uri,
                "detail": detail,
            })),
            AgentInput::LocalImage { path, detail } => Some(json!({
                "kind": "image",
                "uri": path,
                "detail": detail,
                "metadata": {"localPath": path},
            })),
            AgentInput::Text { .. } | AgentInput::Skill { .. } | AgentInput::Mention { .. } => None,
        })
        .collect::<Vec<_>>();
    let attachments = if typed_attachments.is_empty() {
        input_event_payload
            .and_then(|payload| payload.get("attachments"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
    } else {
        typed_attachments
    };
    if text.is_empty() && attachments.is_empty() {
        return None;
    }
    let mut content = Vec::new();
    let mut text_content_values = Vec::new();
    if !text.is_empty() {
        content.push(json!({
            "type": "text",
            "text": text,
        }));
        text_content_values.push(text.to_string());
    }
    let text_elements = user_text_elements_from_payload(input_event_payload);
    if let Some(elements) = text_elements.as_ref() {
        for element in elements {
            let Some(element_text) = element_text(element) else {
                continue;
            };
            if text_content_values
                .iter()
                .any(|existing| existing.trim() == element_text.trim())
            {
                continue;
            }
            text_content_values.push(element_text.to_string());
            content.push(element.clone());
        }
    }
    for attachment in &attachments {
        let mut projected = attachment.clone();
        if let Some(object) = projected.as_object_mut() {
            let kind = object
                .get("kind")
                .cloned()
                .unwrap_or_else(|| Value::String("image".to_string()));
            object.insert("type".to_string(), kind);
        }
        content.push(projected);
    }

    let mut message = json!({
        "id": format!("{}:user", turn.turn_id),
        "role": "user",
        "runtimeTurnId": turn.turn_id,
        "runtime_turn_id": turn.turn_id,
        "content": content,
        "attachments": attachments,
        "timestamp": timestamp_seconds(turn.started_at.as_deref()),
    });
    if let Some(elements) = text_elements {
        if let Some(message_object) = message.as_object_mut() {
            message_object.insert("textElements".to_string(), Value::Array(elements.clone()));
            message_object.insert("text_elements".to_string(), Value::Array(elements));
        }
    }
    Some(message)
}

fn user_text_elements_from_payload(payload: Option<&Value>) -> Option<Vec<Value>> {
    let elements = payload
        .and_then(|payload| {
            payload
                .get("textElements")
                .or_else(|| payload.get("text_elements"))
        })
        .and_then(Value::as_array)?;
    if elements.is_empty() {
        return None;
    }
    Some(elements.clone())
}

fn element_text(element: &Value) -> Option<&str> {
    element
        .get("text")
        .or_else(|| element.get("content"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
}

fn runtime_assistant_message_from_events(
    turn: &AgentTurn,
    events: &[AgentEvent],
) -> Option<serde_json::Value> {
    let mut text = String::new();
    let mut timestamp_value: Option<&str> = None;
    for event in events.iter().filter(|event| {
        event.turn_id.as_deref() == Some(turn.turn_id.as_str())
            && is_assistant_message_event_type(&event.event_type)
            && should_use_message_delta_as_final_text(event)
    }) {
        if let Some(delta) = assistant_message_text_from_payload(&event.payload) {
            text.push_str(&delta);
            timestamp_value = Some(event.timestamp.as_str());
        }
    }
    let text = text.trim();
    if text.is_empty() {
        return None;
    }

    Some(json!({
        "id": format!("{}:assistant", turn.turn_id),
        "role": "assistant",
        "runtimeTurnId": turn.turn_id,
        "runtime_turn_id": turn.turn_id,
        "content": [
            {
                "type": "text",
                "text": text,
            }
        ],
        "timestamp": timestamp_seconds(timestamp_value.or(turn.completed_at.as_deref())),
    }))
}

fn should_use_message_delta_as_final_text(event: &AgentEvent) -> bool {
    match raw_string_field(&event.payload, &["phase", "messagePhase", "message_phase"]) {
        None => true,
        Some(phase) => {
            let normalized = phase.trim().to_ascii_lowercase();
            normalized == "final" || normalized == "final_answer"
        }
    }
}

fn is_assistant_message_event_type(event_type: &str) -> bool {
    matches!(
        event_type,
        "message.delta" | "message.delta_batch" | "message.batch"
    )
}

fn assistant_message_text_from_payload(payload: &serde_json::Value) -> Option<String> {
    if let Some(text) = payload
        .as_str()
        .map(str::to_string)
        .filter(|text| !text.is_empty())
    {
        return Some(text);
    }
    raw_string_field(
        payload,
        &[
            "text",
            "delta",
            "content",
            "message",
            "outputText",
            "output_text",
        ],
    )
    .or_else(|| {
        payload
            .get("content")
            .and_then(|content| raw_string_field(content, &["text", "message"]))
    })
    .or_else(|| {
        for key in ["deltas", "messages", "items", "parts", "content"] {
            let Some(values) = payload.get(key).and_then(serde_json::Value::as_array) else {
                continue;
            };
            let text = values
                .iter()
                .filter_map(assistant_message_text_from_payload)
                .collect::<String>();
            if !text.is_empty() {
                return Some(text);
            }
        }
        None
    })
}
