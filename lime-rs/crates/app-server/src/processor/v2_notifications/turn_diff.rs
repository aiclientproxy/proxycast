use super::{projection_error, required_event_id, EventProjection};
use app_server_protocol::protocol::v2::{self, ServerNotification};
use app_server_protocol::AgentEvent;

pub(super) fn project(event: &AgentEvent) -> EventProjection {
    let Some(thread_id) = required_event_id(event.thread_id.as_deref()) else {
        return EventProjection::Reject(projection_error(event));
    };
    let Some(turn_id) = required_event_id(event.turn_id.as_deref()) else {
        return EventProjection::Reject(projection_error(event));
    };
    let Some(diff) = event.payload.get("diff").and_then(|value| value.as_str()) else {
        return EventProjection::Reject(projection_error(event));
    };

    EventProjection::Direct(vec![ServerNotification::TurnDiffUpdated(
        v2::TurnDiffUpdatedNotification {
            thread_id,
            turn_id,
            diff: diff.to_string(),
        },
    )
    .into()])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn event(payload: Value) -> AgentEvent {
        AgentEvent {
            event_id: "evt-diff-1".to_string(),
            sequence: 3,
            session_id: "session-diff-1".to_string(),
            thread_id: Some("thread-diff-1".to_string()),
            turn_id: Some("turn-diff-1".to_string()),
            event_type: "turn.diff.updated".to_string(),
            timestamp: "2026-08-09T00:00:00.000Z".to_string(),
            payload,
        }
    }

    #[test]
    fn projects_exact_turn_diff_notification() {
        let EventProjection::Direct(notifications) = project(&event(json!({
            "diff": "diff --git a/a.txt b/a.txt\n"
        }))) else {
            panic!("expected direct notification");
        };
        assert_eq!(notifications.len(), 1);
        assert_eq!(notifications[0].method, "turn/diff/updated");
        assert_eq!(
            notifications[0].params,
            Some(json!({
                "threadId": "thread-diff-1",
                "turnId": "turn-diff-1",
                "diff": "diff --git a/a.txt b/a.txt\n"
            }))
        );
    }

    #[test]
    fn projects_empty_diff_to_clear_previous_snapshot() {
        let EventProjection::Direct(notifications) = project(&event(json!({ "diff": "" }))) else {
            panic!("expected direct notification");
        };
        assert_eq!(
            notifications[0].params,
            Some(json!({
                "threadId": "thread-diff-1",
                "turnId": "turn-diff-1",
                "diff": ""
            }))
        );
    }

    #[test]
    fn rejects_missing_or_non_string_diff() {
        assert!(matches!(
            project(&event(json!({}))),
            EventProjection::Reject(_)
        ));
        assert!(matches!(
            project(&event(json!({ "diff": null }))),
            EventProjection::Reject(_)
        ));
    }
}
