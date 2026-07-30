use super::{payload_string, projection_error, required_event_id, EventProjection};
use app_server_protocol::protocol::v2::{self, ServerNotification};
use app_server_protocol::AgentEvent;
use serde_json::Value;

pub(super) fn project(event: &AgentEvent) -> EventProjection {
    let Some(thread_id) = required_event_id(event.thread_id.as_deref()) else {
        return EventProjection::Reject(projection_error(event));
    };
    let Some(message) = payload_string(&event.payload, &["message"]) else {
        return EventProjection::Reject(projection_error(event));
    };
    let code = match event.payload.get("code") {
        None | Some(Value::Null) => None,
        Some(Value::String(code)) => {
            let code = code.trim();
            if code.is_empty() {
                None
            } else {
                Some(code.to_string())
            }
        }
        Some(_) => return EventProjection::Reject(projection_error(event)),
    };

    EventProjection::Direct(vec![ServerNotification::Warning(v2::WarningNotification {
        thread_id: Some(thread_id),
        message,
        code,
    })
    .into()])
}
