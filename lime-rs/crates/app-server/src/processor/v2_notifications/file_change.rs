use super::{projection_error, required_event_id, timestamp_millis, EventProjection};
use crate::processor::thread::{project_event, ProjectedEvent};
use app_server_protocol::protocol::v2::{self, ServerNotification};
use app_server_protocol::AgentEvent;
use std::collections::HashSet;

pub(super) fn project_started(
    started_item_ids: &mut HashSet<String>,
    completed_item_ids: &HashSet<String>,
    event: &AgentEvent,
) -> EventProjection {
    let Some((thread_id, turn_id, item, item_id, changes, status)) = file_change_projection(event)
    else {
        return EventProjection::Reject(projection_error(event));
    };
    if status != v2::PatchApplyStatus::InProgress
        || completed_item_ids.contains(&item_id)
        || started_item_ids.contains(&item_id)
    {
        return EventProjection::Reject(projection_error(event));
    }
    let Some(started_at_ms) = timestamp_millis(&event.timestamp) else {
        return EventProjection::Reject(projection_error(event));
    };
    started_item_ids.insert(item_id.clone());

    let mut notifications = Vec::with_capacity(if changes.is_empty() { 1 } else { 2 });
    notifications.push(
        ServerNotification::ItemStarted(v2::ItemStartedNotification {
            item,
            thread_id: thread_id.clone(),
            turn_id: turn_id.clone(),
            started_at_ms,
        })
        .into(),
    );
    if !changes.is_empty() {
        notifications.push(
            ServerNotification::FileChangePatchUpdated(v2::FileChangePatchUpdatedNotification {
                thread_id,
                turn_id,
                item_id,
                changes,
            })
            .into(),
        );
    }
    EventProjection::Direct(notifications)
}

pub(super) fn project_final(
    started_item_ids: &HashSet<String>,
    completed_item_ids: &mut HashSet<String>,
    event: &AgentEvent,
) -> EventProjection {
    let Some((thread_id, turn_id, item, item_id, _, status)) = file_change_projection(event) else {
        return EventProjection::Reject(projection_error(event));
    };
    if status == v2::PatchApplyStatus::InProgress
        || !started_item_ids.contains(&item_id)
        || completed_item_ids.contains(&item_id)
    {
        return EventProjection::Reject(projection_error(event));
    }
    let Some(completed_at_ms) = timestamp_millis(&event.timestamp) else {
        return EventProjection::Reject(projection_error(event));
    };
    completed_item_ids.insert(item_id);

    EventProjection::Direct(vec![ServerNotification::ItemCompleted(
        v2::ItemCompletedNotification {
            item,
            thread_id,
            turn_id,
            completed_at_ms,
        },
    )
    .into()])
}

fn file_change_projection(
    event: &AgentEvent,
) -> Option<(
    String,
    String,
    v2::ThreadItem,
    String,
    Vec<v2::FileUpdateChange>,
    v2::PatchApplyStatus,
)> {
    let thread_id = required_event_id(event.thread_id.as_deref())?;
    let turn_id = required_event_id(event.turn_id.as_deref())?;
    let ProjectedEvent::Item(item) = project_event(event)? else {
        return None;
    };
    let v2::ThreadItem::FileChange {
        id,
        changes,
        status,
        ..
    } = &item
    else {
        return None;
    };
    Some((
        thread_id,
        turn_id,
        item.clone(),
        id.clone(),
        changes.clone(),
        *status,
    ))
}
