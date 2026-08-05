use super::*;
use agent_protocol::{
    CollabAgentOperation, ItemId, ItemKind, ItemStatus, SessionId, ThreadId, ToolOutput, TurnId,
};
use serde_json::json;

#[derive(Clone, Copy)]
enum ToolFamily {
    Tool,
    Mcp,
    Collab,
}

fn canonical_tool_payload(
    family: ToolFamily,
    text: Option<String>,
    output_ref: Option<&str>,
) -> Value {
    let output = Some(ToolOutput {
        text,
        output_ref: output_ref.map(str::to_string),
        ..ToolOutput::default()
    });
    let (kind, payload, call_id) = match family {
        ToolFamily::Tool => (
            ItemKind::Tool,
            ThreadItemPayload::Tool {
                call_id: "call-tool-large".to_string(),
                name: "Read".to_string(),
                arguments: Vec::new(),
                output,
            },
            "call-tool-large",
        ),
        ToolFamily::Mcp => (
            ItemKind::McpToolCall,
            ThreadItemPayload::McpToolCall {
                call_id: "call-mcp-large".to_string(),
                server_name: "files".to_string(),
                tool_name: "read_resource".to_string(),
                mcp_app_resource_uri: None,
                plugin_id: None,
                arguments: Vec::new(),
                output,
            },
            "call-mcp-large",
        ),
        ToolFamily::Collab => (
            ItemKind::CollabAgentToolCall,
            ThreadItemPayload::CollabAgentToolCall {
                call_id: "call-collab-large".to_string(),
                operation: CollabAgentOperation::SendMessage,
                target_thread_id: Some(ThreadId::new("thread-child")),
                message: Some("continue".to_string()),
                output,
                agent_states: Default::default(),
            },
            "call-collab-large",
        ),
    };
    let item = ThreadItem {
        session_id: SessionId::new("session-1"),
        thread_id: ThreadId::new("thread-1"),
        turn_id: TurnId::new("turn-1"),
        item_id: ItemId::new(format!("item-{call_id}")),
        sequence: 1,
        ordinal: 1,
        created_at_ms: 1,
        updated_at_ms: 2,
        completed_at_ms: Some(2),
        kind,
        status: ItemStatus::Completed,
        payload,
        metadata: Value::Null,
    };
    json!({ "item": item })
}

struct TestSnapshotStore {
    content: String,
}

impl OutputSnapshotStore for TestSnapshotStore {
    fn save_output_snapshot(
        &self,
        _request: &OutputSnapshotSaveRequest,
    ) -> Result<Option<OutputSnapshotRecord>, RuntimeCoreError> {
        Ok(None)
    }

    fn read_output_snapshot(&self, _request: &OutputSnapshotReadRequest) -> Option<String> {
        Some(self.content.clone())
    }
}

#[test]
fn normalizes_large_terminal_output_for_every_canonical_tool_family() {
    for (family, expected_call_id) in [
        (ToolFamily::Tool, "call-tool-large"),
        (ToolFamily::Mcp, "call-mcp-large"),
        (ToolFamily::Collab, "call-collab-large"),
    ] {
        let output = format!(
            "{expected_call_id}:{}",
            "x".repeat(MAX_INLINE_TOOL_OUTPUT_CHARS)
        );
        let normalized = normalize_large_output_payload(
            "item.completed",
            canonical_tool_payload(family, Some(output.clone()), None),
        );
        let payload = normalized.payload;
        let output_ref = payload
            .get("outputRef")
            .and_then(Value::as_str)
            .expect("canonical output ref");
        let nested_output = payload
            .get("item")
            .and_then(|item| item.get("payload"))
            .and_then(|payload| payload.get("output"))
            .expect("nested canonical output");

        assert_eq!(
            nested_output.get("outputRef").and_then(Value::as_str),
            Some(output_ref)
        );
        assert_eq!(
            nested_output.get("truncated").and_then(Value::as_bool),
            Some(true)
        );
        assert!(nested_output
            .get("text")
            .and_then(Value::as_str)
            .is_some_and(|text| text.chars().count() <= TOOL_OUTPUT_PREVIEW_CHARS + 1));

        let output_blob = normalized.output_blob.expect("canonical output blob");
        assert_eq!(output_blob.content, output);
        assert_eq!(output_blob.output_ref, output_ref);
        assert_eq!(output_blob.ref_ids, vec![output_ref]);
    }
}

#[test]
fn canonical_nested_output_ref_is_the_only_reused_ref() {
    let output = "x".repeat(MAX_INLINE_TOOL_OUTPUT_CHARS + 1);
    let mut payload = canonical_tool_payload(
        ToolFamily::Tool,
        Some(output),
        Some("output://canonical-nested"),
    );
    payload["outputRef"] = Value::String("output://outer-decoy".to_string());
    payload["refIds"] = json!(["output://outer-decoy"]);

    let normalized = normalize_large_output_payload("item.completed", payload);
    let output_blob = normalized.output_blob.expect("canonical output blob");
    assert_eq!(output_blob.output_ref, "output://canonical-nested");
    assert_eq!(output_blob.ref_ids, vec!["output://canonical-nested"]);
}

#[test]
fn canonical_output_record_round_trips_snapshot_ref_for_hydration() {
    let full_output = "y".repeat(MAX_INLINE_TOOL_OUTPUT_CHARS + 1);
    let normalized = normalize_large_output_payload(
        "item.completed",
        canonical_tool_payload(ToolFamily::Tool, Some(full_output.clone()), None),
    );
    let mut event = AgentEvent {
        event_id: "event-1".to_string(),
        sequence: 1,
        session_id: "session-1".to_string(),
        thread_id: Some("thread-1".to_string()),
        turn_id: Some("turn-1".to_string()),
        event_type: "item.completed".to_string(),
        timestamp: "2026-07-13T00:00:00.000Z".to_string(),
        payload: normalized.payload,
    };
    let mut record = record_output_blob(
        &event,
        normalized.output_blob.expect("canonical output blob"),
    );
    assert_eq!(record.tool_call_id.as_deref(), Some("call-tool-large"));
    record.content = None;
    record.snapshot_file = Some("runtime-outputs/canonical.txt".to_string());
    attach_output_snapshot_ref(&mut event.payload, &record);

    let hydrated = output_record_from_event(&event).expect("hydrated canonical output record");
    assert_eq!(hydrated.output_ref, record.output_ref);
    assert_eq!(hydrated.tool_call_id.as_deref(), Some("call-tool-large"));
    assert_eq!(
        hydrated.snapshot_file.as_deref(),
        Some("runtime-outputs/canonical.txt")
    );
    let outputs = HashMap::from([(hydrated.output_ref.clone(), hydrated)]);
    let store = TestSnapshotStore {
        content: full_output.clone(),
    };
    assert_eq!(
        output_content(&outputs, &store, "session-1", record.output_ref.as_str()).as_deref(),
        Some(full_output.as_str())
    );
}

#[test]
fn raw_tool_terminal_events_cannot_create_output_blobs() {
    for event_type in ["tool.result", "tool.failed", "tool_end"] {
        let payload = json!({
            "output": "x".repeat(MAX_INLINE_TOOL_OUTPUT_CHARS + 1),
            "result": { "output": "y".repeat(MAX_INLINE_TOOL_OUTPUT_CHARS + 1) },
            "runtimeEvent": {
                "type": "tool_end",
                "result": { "output": "z".repeat(MAX_INLINE_TOOL_OUTPUT_CHARS + 1) }
            }
        });
        let normalized = normalize_large_output_payload(event_type, payload.clone());
        assert_eq!(normalized.payload, payload);
        assert!(normalized.output_blob.is_none());
    }
}

#[test]
fn canonical_item_does_not_fall_back_to_outer_output_fields() {
    let mut payload =
        canonical_tool_payload(ToolFamily::Tool, None, Some("output://canonical-empty"));
    payload["output"] = Value::String("x".repeat(MAX_INLINE_TOOL_OUTPUT_CHARS + 1));
    payload["result"] = json!({ "output": "y".repeat(MAX_INLINE_TOOL_OUTPUT_CHARS + 1) });
    payload["runtimeEvent"] = json!({
        "type": "tool_end",
        "result": { "output": "z".repeat(MAX_INLINE_TOOL_OUTPUT_CHARS + 1) }
    });

    let normalized = normalize_large_output_payload("item.completed", payload.clone());
    assert_eq!(normalized.payload, payload);
    assert!(normalized.output_blob.is_none());
}

#[test]
fn hydrated_output_without_event_type_defaults_to_item_completed() {
    let record = output_record_from_read_model(&json!({
        "outputRef": "output://hydrated",
        "outputBytes": 42,
        "outputSnapshotFile": "runtime-outputs/hydrated.txt"
    }))
    .expect("hydrated output record");

    assert_eq!(record.event_type, "item.completed");
}
