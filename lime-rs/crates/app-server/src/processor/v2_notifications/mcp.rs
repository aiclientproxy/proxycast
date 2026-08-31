use super::{payload_string, projection_error, required_event_id, EventProjection};
use crate::processor::thread::{project_event, ProjectedEvent};
use app_server_protocol::protocol::v2::{self, ServerNotification};
use app_server_protocol::AgentEvent;
use std::collections::HashSet;

pub(super) fn project_progress(
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
    let Some(raw_tool_id) = payload_string(&event.payload, &["tool_id", "toolId"]) else {
        return EventProjection::Reject(projection_error(event));
    };
    let item_id = agent_protocol::ItemId::new(raw_tool_id)
        .as_str()
        .to_string();
    let Some(message) = event
        .payload
        .pointer("/progress/message")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|message| !message.is_empty())
        .map(str::to_string)
    else {
        return EventProjection::Reject(projection_error(event));
    };
    let Some(notification_kind) = event
        .payload
        .pointer("/progress/metadata/notification_kind")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|kind| {
            matches!(
                *kind,
                "mcp_progress"
                    | "mcp_resources_changed"
                    | "mcp_tools_changed"
                    | "mcp_prompts_changed"
            )
        })
    else {
        return EventProjection::Reject(projection_error(event));
    };
    let Some(ProjectedEvent::Item(v2::ThreadItem::McpToolCall {
        id: projected_item_id,
        ..
    })) = project_event(event)
    else {
        return EventProjection::Reject(projection_error(event));
    };
    if projected_item_id != item_id
        || !started_item_ids.contains(&item_id)
        || completed_item_ids.contains(&item_id)
    {
        return EventProjection::Reject(projection_error(event));
    }

    EventProjection::Direct(vec![ServerNotification::McpToolCallProgress(
        v2::McpToolCallProgressNotification {
            thread_id,
            turn_id,
            item_id,
            message,
            notification_kind: (notification_kind != "mcp_progress")
                .then(|| notification_kind.to_string()),
        },
    )
    .into()])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn progress_event(
        item_type: &str,
        message: &str,
        notification_kind: Option<&str>,
    ) -> AgentEvent {
        let (kind, payload) = if item_type == "mcpToolCall" {
            (
                "mcpToolCall",
                json!({
                    "type": "mcpToolCall",
                    "call_id": "mcp-call-1",
                    "server_name": "docs",
                    "tool_name": "search",
                    "arguments": [],
                    "output": null
                }),
            )
        } else {
            (
                "tool",
                json!({
                    "type": "tool",
                    "call_id": "mcp-call-1",
                    "name": "search",
                    "arguments": [],
                    "output": null
                }),
            )
        };
        AgentEvent {
            event_id: "event-mcp-progress-1".to_string(),
            sequence: 2,
            session_id: "session-1".to_string(),
            thread_id: Some("thread-1".to_string()),
            turn_id: Some("turn-1".to_string()),
            event_type: "tool.progress".to_string(),
            timestamp: "2026-07-24T00:00:00Z".to_string(),
            payload: json!({
                "tool_id": "mcp-call-1",
                "progress": {
                    "message": message,
                    "metadata": notification_kind.map(|kind| json!({
                        "notification_kind": kind
                    })).unwrap_or_else(|| json!({}))
                },
                "item": {
                    "sessionId": "session-1",
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "itemId": "item_mcp-call-1",
                    "sequence": 2,
                    "ordinal": 1,
                    "createdAtMs": 1,
                    "updatedAtMs": 2,
                    "completedAtMs": null,
                    "kind": kind,
                    "status": "inProgress",
                    "payload": payload,
                    "metadata": {}
                }
            }),
        }
    }

    #[test]
    fn projects_progress_for_the_started_canonical_mcp_item() {
        let started = HashSet::from(["item_mcp-call-1".to_string()]);
        let EventProjection::Direct(notifications) = project_progress(
            &started,
            &HashSet::new(),
            &progress_event("mcpToolCall", "  正在检索文档  ", Some("mcp_progress")),
        ) else {
            panic!("expected MCP progress notification");
        };
        assert_eq!(notifications.len(), 1);
        let notification = serde_json::to_value(&notifications[0]).expect("notification JSON");
        assert_eq!(
            notification["method"],
            Value::String("item/mcpToolCall/progress".to_string())
        );
        assert_eq!(notification["params"]["itemId"], json!("item_mcp-call-1"));
        assert_eq!(notification["params"]["message"], json!("正在检索文档"));
        assert!(notification["params"].get("notificationKind").is_none());
    }

    #[test]
    fn projects_supported_mcp_list_changed_notifications() {
        let started = HashSet::from(["item_mcp-call-1".to_string()]);
        for kind in [
            "mcp_resources_changed",
            "mcp_tools_changed",
            "mcp_prompts_changed",
        ] {
            let EventProjection::Direct(notifications) = project_progress(
                &started,
                &HashSet::new(),
                &progress_event("mcpToolCall", "工具服务列表已更新", Some(kind)),
            ) else {
                panic!("expected MCP list_changed notification for {kind}");
            };
            let notification = serde_json::to_value(&notifications[0]).expect("notification JSON");
            assert_eq!(
                notification["params"]["notificationKind"],
                Value::String(kind.to_string())
            );
        }
    }

    #[test]
    fn rejects_progress_outside_the_canonical_mcp_lifecycle() {
        let event = progress_event("mcpToolCall", "正在检索文档", Some("mcp_progress"));
        assert!(matches!(
            project_progress(&HashSet::new(), &HashSet::new(), &event),
            EventProjection::Reject(_)
        ));

        let started = HashSet::from(["item_mcp-call-1".to_string()]);
        let completed = started.clone();
        assert!(matches!(
            project_progress(&started, &completed, &event),
            EventProjection::Reject(_)
        ));
        assert!(matches!(
            project_progress(
                &started,
                &HashSet::new(),
                &progress_event("tool", "正在检索文档", Some("mcp_progress"))
            ),
            EventProjection::Reject(_)
        ));
    }

    #[test]
    fn rejects_empty_or_untrusted_progress_messages() {
        let started = HashSet::from(["item_mcp-call-1".to_string()]);
        for event in [
            progress_event("mcpToolCall", "   ", Some("mcp_progress")),
            progress_event("mcpToolCall", "正在检索文档", None),
            progress_event("mcpToolCall", "正在检索文档", Some("mcp_log")),
        ] {
            assert!(matches!(
                project_progress(&started, &HashSet::new(), &event),
                EventProjection::Reject(_)
            ));
        }
    }
}
