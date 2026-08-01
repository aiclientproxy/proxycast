use super::{projection_error, required_event_id, EventProjection};
use crate::processor::thread::{project_event, ProjectedEvent};
use app_server_protocol::protocol::v2::{self, ServerNotification};
use app_server_protocol::AgentEvent;
use std::collections::HashSet;

pub(super) fn project_output_delta(
    started_item_ids: &HashSet<String>,
    completed_item_ids: &HashSet<String>,
    event: &AgentEvent,
) -> EventProjection {
    let Some((thread_id, turn_id, item_id)) = command_projection(event) else {
        return EventProjection::Reject(projection_error(event));
    };
    let Some(delta) = event
        .payload
        .get("delta")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
    else {
        return EventProjection::Reject(projection_error(event));
    };
    if !started_item_ids.contains(&item_id) {
        return EventProjection::Reject(projection_error(event));
    }
    if completed_item_ids.contains(&item_id) {
        return EventProjection::Reject(projection_error(event));
    }

    EventProjection::Direct(vec![ServerNotification::CommandExecutionOutputDelta(
        v2::CommandExecutionOutputDeltaNotification {
            thread_id,
            turn_id,
            item_id,
            delta: delta.to_string(),
        },
    )
    .into()])
}

pub(super) fn project_terminal_interaction(
    started_item_ids: &HashSet<String>,
    completed_item_ids: &HashSet<String>,
    event: &AgentEvent,
) -> EventProjection {
    let Some(thread_id) = required_event_id(event.thread_id.as_deref()) else {
        return EventProjection::Reject(projection_error(event));
    };
    let Some(turn_id) = required_event_id(event.turn_id.as_deref()) else {
        return EventProjection::Reject(projection_error(event));
    };
    let Some(item_id) = event
        .payload
        .get("commandId")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
    else {
        return EventProjection::Reject(projection_error(event));
    };
    let Some(process_id) = event
        .payload
        .get("processId")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
    else {
        return EventProjection::Reject(projection_error(event));
    };
    if !started_item_ids.contains(&item_id) || completed_item_ids.contains(&item_id) {
        return EventProjection::Reject(projection_error(event));
    }
    let Some(stdin) = event
        .payload
        .get("stdin")
        .and_then(serde_json::Value::as_str)
        .filter(|value| terminal_interaction_summary_is_safe(value))
    else {
        return EventProjection::Reject(projection_error(event));
    };

    EventProjection::Direct(vec![
        ServerNotification::CommandExecutionTerminalInteraction(
            v2::CommandExecutionTerminalInteractionNotification {
                thread_id,
                turn_id,
                item_id,
                process_id: process_id.to_string(),
                stdin: stdin.to_string(),
            },
        )
        .into(),
    ])
}

fn terminal_interaction_summary_is_safe(value: &str) -> bool {
    if matches!(value, "(poll)" | "(interrupt)") {
        return true;
    }
    value
        .strip_prefix("sent ")
        .and_then(|value| value.strip_suffix(" chars"))
        .is_some_and(|count| !count.is_empty() && count.bytes().all(|byte| byte.is_ascii_digit()))
}

fn command_projection(event: &AgentEvent) -> Option<(String, String, String)> {
    let thread_id = required_event_id(event.thread_id.as_deref())?;
    let turn_id = required_event_id(event.turn_id.as_deref())?;
    let ProjectedEvent::Item(v2::ThreadItem::CommandExecution { id, .. }) = project_event(event)?
    else {
        return None;
    };
    Some((thread_id, turn_id, id))
}
