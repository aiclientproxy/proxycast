use super::{projection_error, required_event_id, timestamp_millis, EventProjection};
use crate::processor::thread::{project_event, ProjectedEvent};
use app_server_protocol::protocol::v2::{self, ServerNotification};
use app_server_protocol::AgentEvent;
use std::collections::HashSet;

pub(super) fn project_delta(
    started_item_ids: &mut HashSet<String>,
    completed_item_ids: &HashSet<String>,
    event: &AgentEvent,
) -> EventProjection {
    let Some((thread_id, turn_id, item, item_id)) = plan_projection(event) else {
        return EventProjection::Reject(projection_error(event));
    };
    let Some(delta) = event
        .payload
        .get("delta")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
    else {
        return EventProjection::Reject(projection_error(event));
    };
    if completed_item_ids.contains(&item_id) {
        return EventProjection::Reject(projection_error(event));
    }

    let mut notifications = Vec::with_capacity(2);
    if started_item_ids.insert(item_id.clone()) {
        let Some(started_at_ms) = timestamp_millis(&event.timestamp) else {
            return EventProjection::Reject(projection_error(event));
        };
        notifications.push(
            ServerNotification::ItemStarted(v2::ItemStartedNotification {
                item,
                thread_id: thread_id.clone(),
                turn_id: turn_id.clone(),
                started_at_ms,
            })
            .into(),
        );
    }
    notifications.push(
        ServerNotification::PlanDelta(v2::PlanDeltaNotification {
            thread_id,
            turn_id,
            item_id,
            delta,
        })
        .into(),
    );
    EventProjection::Direct(notifications)
}

pub(super) fn project_final(
    started_item_ids: &mut HashSet<String>,
    completed_item_ids: &mut HashSet<String>,
    event: &AgentEvent,
) -> EventProjection {
    let Some((thread_id, turn_id, item, item_id)) = plan_projection(event) else {
        return EventProjection::Reject(projection_error(event));
    };
    if completed_item_ids.contains(&item_id) {
        return EventProjection::Reject(projection_error(event));
    }
    let Some(timestamp_ms) = timestamp_millis(&event.timestamp) else {
        return EventProjection::Reject(projection_error(event));
    };

    let mut notifications = Vec::with_capacity(2);
    if started_item_ids.insert(item_id.clone()) {
        notifications.push(
            ServerNotification::ItemStarted(v2::ItemStartedNotification {
                item: item.clone(),
                thread_id: thread_id.clone(),
                turn_id: turn_id.clone(),
                started_at_ms: timestamp_ms,
            })
            .into(),
        );
    }
    completed_item_ids.insert(item_id);
    notifications.push(
        ServerNotification::ItemCompleted(v2::ItemCompletedNotification {
            item,
            thread_id,
            turn_id,
            completed_at_ms: timestamp_ms,
        })
        .into(),
    );
    EventProjection::Direct(notifications)
}

fn plan_projection(event: &AgentEvent) -> Option<(String, String, v2::ThreadItem, String)> {
    let thread_id = required_event_id(event.thread_id.as_deref())?;
    let turn_id = required_event_id(event.turn_id.as_deref())?;
    let ProjectedEvent::Item(item) = project_event(event)? else {
        return None;
    };
    let v2::ThreadItem::Plan { id, .. } = &item else {
        return None;
    };
    Some((thread_id, turn_id, item.clone(), id.clone()))
}
