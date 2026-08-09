use super::{payload_string, projection_error, required_event_id, EventProjection};
use app_server_protocol::protocol::v2::{self, ServerNotification};
use app_server_protocol::AgentEvent;

pub(super) fn project(event: &AgentEvent) -> EventProjection {
    let Some(thread_id) = required_event_id(event.thread_id.as_deref()) else {
        return EventProjection::Reject(projection_error(event));
    };
    let Some(message) = payload_string(&event.payload, &["message"]) else {
        return EventProjection::Reject(projection_error(event));
    };
    EventProjection::Direct(vec![ServerNotification::GuardianWarning(
        v2::GuardianWarningNotification { thread_id, message },
    )
    .into()])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn event(thread_id: Option<&str>, message: Option<&str>) -> AgentEvent {
        AgentEvent {
            event_id: "guardian-warning-1".to_string(),
            sequence: 1,
            session_id: "session-guardian".to_string(),
            thread_id: thread_id.map(str::to_string),
            turn_id: Some("turn-guardian".to_string()),
            event_type: "guardian.warning".to_string(),
            timestamp: "2026-08-09T00:00:00.000Z".to_string(),
            payload: message
                .map(|message| json!({ "message": message }))
                .unwrap_or_else(|| json!({})),
        }
    }

    #[test]
    fn projects_guardian_warning_as_independent_high_priority_notification() {
        let EventProjection::Direct(notifications) = project(&event(
            Some("thread-guardian"),
            Some("Automatic approval review interrupted this turn."),
        )) else {
            panic!("guardian warning should project directly");
        };
        assert_eq!(notifications[0].method, "guardianWarning");
        assert_eq!(
            notifications[0].params.as_ref().expect("warning params"),
            &json!({
                "threadId": "thread-guardian",
                "message": "Automatic approval review interrupted this turn."
            })
        );
    }

    #[test]
    fn rejects_guardian_warning_without_thread_or_message() {
        assert!(matches!(
            project(&event(None, Some("message"))),
            EventProjection::Reject(_)
        ));
        assert!(matches!(
            project(&event(Some("thread-guardian"), None)),
            EventProjection::Reject(_)
        ));
        assert!(matches!(
            project(&event(Some("thread-guardian"), Some("  "))),
            EventProjection::Reject(_)
        ));
    }
}
