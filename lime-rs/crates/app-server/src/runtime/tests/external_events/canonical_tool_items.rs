use super::*;
use serde_json::Value;

fn canonical_tool_item(call_id: &str, status: &str) -> Value {
    json!({
        "item": {
            "sessionId": "nested-session",
            "threadId": "nested-thread",
            "turnId": "nested-turn",
            "itemId": format!("item_{call_id}"),
            "sequence": 1,
            "ordinal": 1,
            "createdAtMs": 1,
            "updatedAtMs": 1,
            "completedAtMs": (status != "inProgress").then_some(1),
            "kind": "tool",
            "status": status,
            "payload": {
                "type": "tool",
                "call_id": call_id,
                "name": "Bash",
                "arguments": [],
                "output": (status != "inProgress").then(|| json!({ "text": "ok" })),
            },
            "metadata": {},
        }
    })
}

fn canonical_mcp_tool_item(call_id: &str, status: &str) -> Value {
    json!({
        "item": {
            "sessionId": "nested-session",
            "threadId": "nested-thread",
            "turnId": "nested-turn",
            "itemId": format!("item_{call_id}"),
            "sequence": 1,
            "ordinal": 1,
            "createdAtMs": 1,
            "updatedAtMs": 1,
            "completedAtMs": (status != "inProgress").then_some(1),
            "kind": "mcpToolCall",
            "status": status,
            "payload": {
                "type": "mcpToolCall",
                "call_id": call_id,
                "server_name": "docs",
                "tool_name": "search",
                "arguments": [],
                "output": (status != "inProgress").then(|| json!({ "text": "ok" })),
            },
            "metadata": {},
        }
    })
}

#[tokio::test]
async fn append_external_runtime_events_accepts_mcp_progress_inside_canonical_lifecycle() {
    let (core, session_id, turn_id) = runtime_with_active_turn(
        "sess_canonical_mcp_progress",
        "thread_canonical_mcp_progress",
        "turn_canonical_mcp_progress",
    )
    .await;

    let events = core
        .append_external_runtime_events(
            &session_id,
            Some(&turn_id),
            vec![
                RuntimeEvent::new(
                    "item.started",
                    canonical_mcp_tool_item("mcp-call-1", "inProgress"),
                ),
                RuntimeEvent::new(
                    "tool.progress",
                    json!({
                        "tool_id": "mcp-call-1",
                        "serverName": "docs",
                        "toolName": "search",
                        "progress": {
                            "message": "正在检索文档",
                            "metadata": {"notification_kind": "mcp_progress"}
                        }
                    }),
                ),
                RuntimeEvent::new(
                    "item.completed",
                    canonical_mcp_tool_item("mcp-call-1", "completed"),
                ),
            ],
        )
        .expect("canonical MCP progress lifecycle should append");

    assert_eq!(events.len(), 3);
    let progress_item =
        serde_json::from_value::<agent_protocol::ThreadItem>(events[1].payload["item"].clone())
            .expect("progress canonical item");
    assert_eq!(progress_item.item_id.as_str(), "item_mcp-call-1");
    assert!(matches!(
        progress_item.payload,
        agent_protocol::ThreadItemPayload::McpToolCall { .. }
    ));
}

#[tokio::test]
async fn append_external_runtime_events_rejects_canonical_tool_completed_without_start() {
    let (core, session_id, turn_id) = runtime_with_active_turn(
        "sess_canonical_tool_without_start",
        "thread_canonical_tool_without_start",
        "turn_canonical_tool_without_start",
    )
    .await;

    let error = core
        .append_external_runtime_events(
            &session_id,
            Some(&turn_id),
            vec![RuntimeEvent::new(
                "item.completed",
                canonical_tool_item("call_without_start", "completed"),
            )],
        )
        .expect_err("canonical tool completion without start must fail closed");

    assert!(error
        .to_string()
        .contains("tool_item_completed_without_start"));
}

#[tokio::test]
async fn append_external_runtime_events_rejects_duplicate_canonical_tool_start() {
    let (core, session_id, turn_id) = runtime_with_active_turn(
        "sess_canonical_tool_duplicate_start",
        "thread_canonical_tool_duplicate_start",
        "turn_canonical_tool_duplicate_start",
    )
    .await;
    core.append_external_runtime_events(
        &session_id,
        Some(&turn_id),
        vec![RuntimeEvent::new(
            "item.started",
            canonical_tool_item("call_duplicate", "inProgress"),
        )],
    )
    .expect("first canonical tool start should append");

    let error = core
        .append_external_runtime_events(
            &session_id,
            Some(&turn_id),
            vec![RuntimeEvent::new(
                "item.started",
                canonical_tool_item("call_duplicate", "inProgress"),
            )],
        )
        .expect_err("duplicate canonical tool start must fail closed");

    assert!(error
        .to_string()
        .contains("tool_item_started_already_active"));
}

#[tokio::test]
async fn append_external_runtime_events_rejects_retired_raw_tool_wire() {
    let (core, session_id, turn_id) = runtime_with_active_turn(
        "sess_retired_tool_wire",
        "thread_retired_tool_wire",
        "turn_retired_tool_wire",
    )
    .await;

    for event_type in [
        "tool.started",
        "tool.args",
        "tool.result",
        "tool.failed",
        "tool_end",
        "tool.args.delta",
        "tool.input.delta",
    ] {
        let error = core
            .append_external_runtime_events(
                &session_id,
                Some(&turn_id),
                vec![RuntimeEvent::new(
                    event_type,
                    json!({ "toolCallId": "retired-tool" }),
                )],
            )
            .expect_err("retired raw tool wire must fail closed");
        assert!(error
            .to_string()
            .contains("retired raw tool wire event is forbidden"));
    }
}

#[tokio::test]
async fn append_external_runtime_events_rejects_retired_raw_tool_wire_with_import_markers() {
    let (core, session_id, turn_id) = runtime_with_active_turn(
        "sess_imported_tool_wire",
        "thread_imported_tool_wire",
        "turn_imported_tool_wire",
    )
    .await;

    for (event_type, payload) in [
        (
            "tool.started",
            json!({ "toolCallId": "imported-tool", "imported": true }),
        ),
        (
            "tool.result",
            json!({
                "toolCallId": "imported-tool",
                "sourceProvenance": { "sourceClient": "codex" }
            }),
        ),
        (
            "tool.failed",
            json!({
                "toolCallId": "imported-tool",
                "source_provenance": { "source_client": "codex" }
            }),
        ),
    ] {
        let error = core
            .append_external_runtime_events(
                &session_id,
                Some(&turn_id),
                vec![RuntimeEvent::new(event_type, payload)],
            )
            .expect_err("import markers must not bypass retired raw tool wire rejection");
        assert!(error
            .to_string()
            .contains("retired raw tool wire event is forbidden"));
    }
}
