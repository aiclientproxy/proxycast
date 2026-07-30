use super::super::raw_string_field;
use super::super::StoredSession;
use app_server_protocol::AgentEvent;
use serde_json::json;

pub(super) fn runtime_warning_items_from_events(stored: &StoredSession) -> Vec<serde_json::Value> {
    stored
        .events
        .iter()
        .filter(|event| event.event_type == "runtime.warning")
        .filter_map(|event| {
            let thread_id = event.thread_id.as_deref()?.trim();
            if thread_id.is_empty() || thread_id != stored.session.thread_id {
                return None;
            }
            let turn_id = event.turn_id.as_deref()?.trim();
            if turn_id.is_empty() || !stored.turns.iter().any(|turn| turn.turn_id == turn_id) {
                return None;
            }
            let message = runtime_warning_message_from_event(event)?;
            let code = runtime_warning_code_from_event(event)?;
            let mut item = json!({
                "id": format!("{}:warning:{}", turn_id, event.event_id),
                "thread_id": thread_id,
                "turn_id": turn_id,
                "sequence": event.sequence,
                "type": "warning",
                "status": "warning",
                "message": message,
                "started_at": event.timestamp,
                "completed_at": event.timestamp,
                "updated_at": event.timestamp,
            });
            if let Some(code) = code {
                item["code"] = json!(code);
            }
            Some(item)
        })
        .collect()
}

fn runtime_warning_message_from_event(event: &AgentEvent) -> Option<String> {
    if event.event_type != "runtime.warning" {
        return None;
    }
    event
        .payload
        .get("message")?
        .as_str()
        .map(str::trim)
        .filter(|message| !message.is_empty())
        .map(ToString::to_string)
}

fn runtime_warning_code_from_event(event: &AgentEvent) -> Option<Option<String>> {
    if event.event_type != "runtime.warning" {
        return None;
    }
    match event.payload.get("code") {
        None | Some(serde_json::Value::Null) => Some(None),
        Some(serde_json::Value::String(code)) => {
            let code = code.trim();
            Some((!code.is_empty()).then(|| code.to_string()))
        }
        Some(_) => None,
    }
}

pub(super) fn runtime_error_items_from_events(stored: &StoredSession) -> Vec<serde_json::Value> {
    stored
        .events
        .iter()
        .filter(|event| matches!(event.event_type.as_str(), "turn.failed" | "runtime.error"))
        .filter_map(|event| {
            let message = runtime_error_message_from_event(event)?;
            let turn_id = event
                .turn_id
                .clone()
                .or_else(|| stored.turns.last().map(|turn| turn.turn_id.clone()))?;
            Some(json!({
                "id": format!("{}:error:{}", turn_id, event.event_id),
                "thread_id": event.thread_id.clone().unwrap_or_else(|| stored.session.thread_id.clone()),
                "turn_id": turn_id,
                "sequence": event.sequence,
                "type": "error",
                "status": "failed",
                "message": message,
                "started_at": event.timestamp,
                "completed_at": event.timestamp,
                "updated_at": event.timestamp,
            }))
        })
        .collect()
}

fn runtime_error_message_from_event(event: &AgentEvent) -> Option<String> {
    if !matches!(event.event_type.as_str(), "turn.failed" | "runtime.error") {
        return None;
    }
    raw_string_field(
        &event.payload,
        &[
            "message",
            "error",
            "reason",
            "detail",
            "details",
            "error_message",
            "errorMessage",
        ],
    )
    .map(|message| message.trim().to_string())
    .filter(|message| !message.is_empty())
}

pub(super) fn latest_turn_error_message(
    stored: &StoredSession,
    turn_id: Option<&str>,
) -> Option<String> {
    stored
        .events
        .iter()
        .rev()
        .filter(|event| match turn_id {
            Some(turn_id) => event.turn_id.as_deref() == Some(turn_id),
            None => true,
        })
        .find_map(runtime_error_message_from_event)
}
