use super::super::{thread_item_projection, RuntimeCoreError, StoredSession};
use app_server_protocol::AgentEvent;

#[derive(Clone, Copy)]
enum CanonicalNotificationTarget {
    Turn,
    Item,
}

pub(super) fn notification_events_with_canonical_entities(
    stored: &StoredSession,
    events: &[AgentEvent],
) -> Result<Vec<AgentEvent>, RuntimeCoreError> {
    let mut materializer = thread_item_projection::IncrementalMaterializer::from_events(
        &stored.events,
        &stored.session.session_id,
        &stored.session.thread_id,
    )
    .map_err(|error| {
        RuntimeCoreError::Backend(format!(
            "cannot initialize canonical notification materializer: {error}"
        ))
    })?;
    let mut notifications = Vec::with_capacity(events.len());

    for event in events {
        let entities = materializer.apply(event).map_err(|error| {
            RuntimeCoreError::Backend(format!(
                "cannot materialize canonical notification for event {} ({}): {error}",
                event.event_id, event.event_type
            ))
        })?;
        let Some(target) = canonical_notification_target(event) else {
            notifications.push(event.clone());
            continue;
        };
        let mut notification = event.clone();
        let payload = notification.payload.as_object_mut().ok_or_else(|| {
            RuntimeCoreError::Backend(format!(
                "cannot attach canonical notification entity to non-object payload for event {} ({})",
                event.event_id, event.event_type
            ))
        })?;

        match target {
            CanonicalNotificationTarget::Turn => {
                let turn_id = event.turn_id.as_deref().ok_or_else(|| {
                    RuntimeCoreError::Backend(format!(
                        "cannot materialize canonical turn notification without turn id for event {} ({})",
                        event.event_id, event.event_type
                    ))
                })?;
                let mut turn = entities
                    .turn
                    .filter(|turn| turn.turn_id.as_str() == turn_id)
                    .ok_or_else(|| {
                        RuntimeCoreError::Backend(format!(
                            "canonical materializer produced no turn for event {} ({})",
                            event.event_id, event.event_type
                        ))
                    })?;
                turn.items = materializer.items_for_turn(turn_id);
                turn.items_view = agent_protocol::TurnItemsView::Full;
                payload.insert(
                    "turn".to_string(),
                    serde_json::to_value(turn).map_err(|error| {
                        RuntimeCoreError::Backend(format!(
                            "cannot serialize canonical turn for event {}: {error}",
                            event.event_id
                        ))
                    })?,
                );
            }
            CanonicalNotificationTarget::Item => {
                let item = entities
                    .item
                    .filter(|item| item.sequence == event.sequence)
                    .ok_or_else(|| {
                        RuntimeCoreError::Backend(format!(
                            "canonical materializer produced no item for event {} ({})",
                            event.event_id, event.event_type
                        ))
                    })?;
                payload.insert(
                    "item".to_string(),
                    serde_json::to_value(item).map_err(|error| {
                        RuntimeCoreError::Backend(format!(
                            "cannot serialize canonical item for event {}: {error}",
                            event.event_id
                        ))
                    })?,
                );
            }
        }
        notifications.push(notification);
    }

    Ok(notifications)
}

fn canonical_notification_target(event: &AgentEvent) -> Option<CanonicalNotificationTarget> {
    let event_type = event.event_type.as_str();
    if matches!(
        event_type,
        "turn.accepted" | "turn.started" | "turn.completed" | "turn.failed" | "turn.canceled"
    ) {
        return Some(CanonicalNotificationTarget::Turn);
    }
    if matches!(
        event_type,
        "approval.session_cache.hit"
            | "item.removed"
            | "item.deleted"
            | "message.removed"
            | "tool.removed"
            | "turn.removed"
            | "turn.deleted"
    ) {
        return None;
    }
    if event.turn_id.is_none() {
        return None;
    }
    let item_lifecycle = [
        "item.",
        "message.",
        "plan.",
        "reasoning.",
        "tool.",
        "mcp.",
        "collab.",
        "action.",
        "approval.",
        "command.",
        "patch.",
        "file.",
        "artifact.",
        "media.",
        "subagent.",
        "sub_agent.",
        "context.compaction",
    ]
    .iter()
    .any(|prefix| event_type.starts_with(prefix));
    item_lifecycle.then_some(CanonicalNotificationTarget::Item)
}

#[cfg(test)]
mod tests {
    use super::{
        canonical_notification_target, notification_events_with_canonical_entities,
        CanonicalNotificationTarget,
    };
    use crate::runtime::StoredSession;
    use app_server_protocol::{AgentEvent, AgentSession, AgentSessionStatus};
    use serde_json::json;
    use std::collections::HashMap;

    fn event(event_type: &str) -> AgentEvent {
        AgentEvent {
            event_id: format!("event-{event_type}"),
            sequence: 1,
            session_id: "session-1".to_string(),
            thread_id: Some("thread-1".to_string()),
            turn_id: Some("turn-1".to_string()),
            event_type: event_type.to_string(),
            timestamp: "2026-07-13T00:00:00Z".to_string(),
            payload: json!({}),
        }
    }

    fn stored_session() -> StoredSession {
        StoredSession {
            session: AgentSession {
                session_id: "session-1".to_string(),
                thread_id: "thread-1".to_string(),
                app_id: "agent-runtime".to_string(),
                workspace_id: None,
                business_object_ref: None,
                status: AgentSessionStatus::Running,
                created_at: "2026-07-13T00:00:00Z".to_string(),
                updated_at: "2026-07-13T00:00:00Z".to_string(),
            },
            turns: Vec::new(),
            turn_inputs: HashMap::new(),
            turn_runtime_options: HashMap::new(),
            events: Vec::new(),
            output_blobs: HashMap::new(),
        }
    }

    #[test]
    fn approval_session_cache_hit_is_audit_only_notification() {
        assert!(canonical_notification_target(&event("approval.session_cache.hit")).is_none());
    }

    #[test]
    fn thread_goal_continuation_is_internal_context_without_an_item_notification() {
        assert!(canonical_notification_target(&event("thread.goal.continuation")).is_none());
    }

    #[test]
    fn approval_resolution_remains_a_canonical_item_notification() {
        assert!(matches!(
            canonical_notification_target(&event("action.resolved")),
            Some(CanonicalNotificationTarget::Item)
        ));
    }

    #[test]
    fn plan_lifecycle_is_a_canonical_item_notification() {
        for event_type in ["plan.delta", "plan.final"] {
            assert!(matches!(
                canonical_notification_target(&event(event_type)),
                Some(CanonicalNotificationTarget::Item)
            ));
        }
    }

    #[test]
    fn terminal_turn_notification_includes_current_canonical_items() {
        let events = vec![
            AgentEvent {
                event_id: "event-item-started".to_string(),
                sequence: 1,
                session_id: "session-1".to_string(),
                thread_id: Some("thread-1".to_string()),
                turn_id: Some("turn-1".to_string()),
                event_type: "item.started".to_string(),
                timestamp: "2026-07-13T00:00:01Z".to_string(),
                payload: json!({
                    "itemType": "agent_message",
                    "itemId": "message-1",
                    "status": "in_progress"
                }),
            },
            AgentEvent {
                event_id: "event-message-delta".to_string(),
                sequence: 2,
                session_id: "session-1".to_string(),
                thread_id: Some("thread-1".to_string()),
                turn_id: Some("turn-1".to_string()),
                event_type: "message.delta".to_string(),
                timestamp: "2026-07-13T00:00:02Z".to_string(),
                payload: json!({
                    "itemId": "message-1",
                    "phase": "final_answer",
                    "text": "final answer"
                }),
            },
            AgentEvent {
                event_id: "event-turn-completed".to_string(),
                sequence: 3,
                session_id: "session-1".to_string(),
                thread_id: Some("thread-1".to_string()),
                turn_id: Some("turn-1".to_string()),
                event_type: "turn.completed".to_string(),
                timestamp: "2026-07-13T00:00:03Z".to_string(),
                payload: json!({"status": "completed"}),
            },
        ];

        let notifications = notification_events_with_canonical_entities(&stored_session(), &events)
            .expect("canonical notification entities");
        let turn = notifications
            .last()
            .and_then(|event| event.payload.get("turn"))
            .expect("terminal turn entity");

        assert_eq!(turn["items"][0]["payload"]["type"], "agentMessage");
        assert_eq!(turn["items"][0]["payload"]["phase"], "final_answer");
        assert_eq!(turn["items"][0]["payload"]["text"], "final answer");
    }

    #[test]
    fn unknown_item_notification_attaches_sanitized_canonical_entity() {
        let event = AgentEvent {
            event_id: "event-unknown-started".to_string(),
            sequence: 1,
            session_id: "session-1".to_string(),
            thread_id: Some("thread-1".to_string()),
            turn_id: Some("turn-1".to_string()),
            event_type: "item.started".to_string(),
            timestamp: "2026-07-13T00:00:01Z".to_string(),
            payload: json!({"item": {
                "id": "unknown-1",
                "type": "futureCapability",
                "label": "future capability",
                "secretToken": "UNKNOWN_ITEM_SECRET_MUST_NOT_LEAK",
                "status": "inProgress"
            }}),
        };

        let notifications =
            notification_events_with_canonical_entities(&stored_session(), &[event])
                .expect("unknown item canonical notification");
        let item = &notifications[0].payload["item"];
        assert_eq!(item["kind"], "unknown");
        assert_eq!(item["payload"]["type"], "unknown");
        assert_eq!(item["payload"]["upstream_type"], "futureCapability");
        assert_eq!(
            item["payload"]["field_names"],
            json!(["[redacted]", "label", "status"])
        );
        let wire = item.to_string();
        assert!(!wire.contains("UNKNOWN_ITEM_SECRET_MUST_NOT_LEAK"));
        assert!(!wire.contains("future capability"));
    }

    #[test]
    fn malformed_unknown_item_notification_fails_closed() {
        let mut malformed = event("item.started");
        malformed.payload = json!({"item": {
            "id": "unknown-1",
            "type": "future_capability"
        }});

        let error = notification_events_with_canonical_entities(&stored_session(), &[malformed])
            .expect_err("malformed unknown item must fail closed");
        assert!(error
            .to_string()
            .contains("canonical materializer produced no item"));
    }
}
