use super::materializer::materialize_events;
use agent_protocol::{
    AgentInput, ApprovalAction, ApprovalDecision, ApprovalScope, ByteRange, CollabAgentOperation,
    FileChange, FileChangeKind, ImageDetail, ItemKind, ItemStatus, PlanStepStatus, TextElement,
    ThreadItemPayload, TurnApprovalState, TurnQueueState, TurnStatus,
};
use app_server_protocol::AgentEvent;
use serde_json::json;

mod canonical_lifecycle;
mod incremental;

fn event(
    id: &str,
    sequence: u64,
    event_type: &str,
    turn_id: &str,
    payload: serde_json::Value,
) -> AgentEvent {
    AgentEvent {
        event_id: id.to_string(),
        sequence,
        session_id: "session-1".to_string(),
        thread_id: Some("thread-1".to_string()),
        turn_id: Some(turn_id.to_string()),
        event_type: event_type.to_string(),
        timestamp: format!("2026-07-12T00:00:{:02}Z", sequence.min(59)),
        payload,
    }
}

fn canonical_tool_item(item_id: &str, status: &str, ordinal: u64) -> serde_json::Value {
    json!({
        "sessionId": "nested-session",
        "threadId": "nested-thread",
        "turnId": "nested-turn",
        "itemId": item_id,
        "sequence": 999,
        "ordinal": ordinal,
        "createdAtMs": 111,
        "updatedAtMs": 222,
        "completedAtMs": 333,
        "kind": "tool",
        "status": status,
        "payload": {
            "type": "tool",
            "call_id": format!("call-{item_id}"),
            "name": "rg",
            "arguments": [],
            "output": null
        },
        "metadata": {"source": "nested-canonical"}
    })
}

#[test]
fn artifact_snapshot_lowers_to_applied_file_change() {
    let changes = materialize_events(
        &[event(
            "artifact-snapshot",
            1,
            "artifact.snapshot",
            "turn-1",
            json!({
                "artifact": {
                    "artifactId": "artifact-1",
                    "filePath": "src/generated.ts",
                    "content": "export const generated = true;"
                }
            }),
        )],
        "session-1",
        "thread-1",
    )
    .expect("materialize artifact snapshot");

    assert_eq!(changes.changed_items.len(), 1);
    let item = &changes.changed_items[0];
    assert_eq!(item.status, ItemStatus::Completed);
    assert_eq!(
        item.payload,
        ThreadItemPayload::File {
            changes: vec![FileChange {
                path: "src/generated.ts".to_string(),
                kind: FileChangeKind::Update { move_path: None },
                diff: "export const generated = true;".to_string(),
            }],
            status: agent_protocol::FileChangeStatus::Applied,
        }
    );
}

#[test]
fn patch_batch_preserves_changes_when_declined_terminal_omits_snapshot() {
    let changes = materialize_events(
        &[
            event(
                "patch-started",
                1,
                "patch.started",
                "turn-1",
                json!({
                    "patchId": "patch-1",
                    "changes": [
                        {"kind": "add", "path": "new.txt", "diff": "+new"},
                        {"kind": "delete", "path": "dead.txt", "diff": "-dead"},
                        {"kind": "update", "path": "same.txt", "diff": "-old\n+new"},
                        {
                            "kind": "update",
                            "path": "source.txt",
                            "movePath": "target.txt",
                            "diff": "-before\n+after"
                        }
                    ]
                }),
            ),
            event(
                "patch-declined",
                2,
                "patch.declined",
                "turn-1",
                json!({"patchId": "patch-1", "status": "declined"}),
            ),
        ],
        "session-1",
        "thread-1",
    )
    .expect("materialize patch batch");

    assert_eq!(changes.changed_items.len(), 1);
    let item = &changes.changed_items[0];
    assert_eq!(item.status, ItemStatus::Completed);
    let ThreadItemPayload::File { changes, status } = &item.payload else {
        panic!("file change payload");
    };
    assert_eq!(*status, agent_protocol::FileChangeStatus::Rejected);
    assert_eq!(changes.len(), 4);
    assert_eq!(
        changes[3].kind,
        FileChangeKind::Update {
            move_path: Some("target.txt".to_string())
        }
    );
    assert_eq!(changes[3].path, "source.txt");
}

#[test]
fn coalesces_message_delta_into_one_stable_typed_item() {
    let changes = materialize_events(
        &[
            event(
                "event-1",
                1,
                "message.delta",
                "turn-1",
                json!({"itemId": "msg-1", "role": "assistant", "text": "hel"}),
            ),
            event(
                "event-2",
                2,
                "message.delta",
                "turn-1",
                json!({"itemId": "msg-1", "role": "assistant", "text": "hello"}),
            ),
        ],
        "session-1",
        "thread-1",
    )
    .expect("materialize");

    assert_eq!(changes.changed_items.len(), 1);
    let item = &changes.changed_items[0];
    assert_eq!(item.item_id.as_str(), "item_msg-1");
    assert_eq!(item.sequence, 2);
    assert_eq!(item.status, ItemStatus::InProgress);
    assert_eq!(
        item.payload,
        ThreadItemPayload::AgentMessage {
            text: "hello".to_string(),
            phase: None,
            content_parts: Vec::new(),
        }
    );
    assert_eq!(changes.changed_turns.len(), 1);
    assert_eq!(changes.changed_turns[0].status, TurnStatus::InProgress);
}

#[test]
fn user_and_agent_messages_have_explicit_terminal_item_lifecycle() {
    let changes = materialize_events(
        &[
            event(
                "user-created",
                1,
                "message.created",
                "turn-1",
                json!({"role": "user", "input": [AgentInput::text("hello")]}),
            ),
            event(
                "agent-delta",
                2,
                "message.delta",
                "turn-1",
                json!({"role": "assistant", "text": "answer"}),
            ),
            event(
                "agent-completed",
                3,
                "message.completed",
                "turn-1",
                json!({
                    "role": "assistant",
                    "phase": "final_answer",
                    "status": "completed"
                }),
            ),
        ],
        "session-1",
        "thread-1",
    )
    .expect("materialize message lifecycle");

    assert_eq!(changes.changed_items.len(), 2);
    let user = &changes.changed_items[0];
    assert_eq!(user.item_id.as_str(), "item_user-turn-1");
    assert_eq!(user.status, ItemStatus::Completed);
    assert_eq!(user.completed_at_ms, Some(user.updated_at_ms));

    let agent = &changes.changed_items[1];
    assert_eq!(agent.item_id.as_str(), "item_agent-turn-1");
    assert_eq!(agent.sequence, 3);
    assert_eq!(agent.ordinal, 2);
    assert_eq!(agent.status, ItemStatus::Completed);
    assert_eq!(agent.completed_at_ms, Some(agent.updated_at_ms));
    assert_eq!(
        agent.payload,
        ThreadItemPayload::AgentMessage {
            text: "answer".to_string(),
            phase: Some(agent_protocol::response_item::MessagePhase::FinalAnswer),
            content_parts: Vec::new(),
        }
    );
}

#[test]
fn canceled_agent_message_preserves_partial_text_and_interrupts_item() {
    let changes = materialize_events(
        &[
            event(
                "agent-delta",
                1,
                "message.delta",
                "turn-1",
                json!({"role": "assistant", "text": "partial"}),
            ),
            event(
                "agent-interrupted",
                2,
                "message.completed",
                "turn-1",
                json!({"role": "assistant", "status": "interrupted"}),
            ),
        ],
        "session-1",
        "thread-1",
    )
    .expect("materialize interrupted message");

    let item = &changes.changed_items[0];
    assert_eq!(item.status, ItemStatus::Interrupted);
    assert_eq!(item.completed_at_ms, Some(item.updated_at_ms));
    assert!(matches!(
        &item.payload,
        ThreadItemPayload::AgentMessage { text, .. } if text == "partial"
    ));
}

#[test]
fn plan_delta_and_final_share_turn_scoped_revision_identity() {
    let changes = materialize_events(
        &[
            event(
                "plan-delta",
                1,
                "plan.delta",
                "turn-1",
                json!({
                    "text": "- [ ] inspect",
                    "revisionId": "proposed_plan:1",
                    "source": "proposed_plan"
                }),
            ),
            event(
                "plan-final",
                2,
                "plan.final",
                "turn-1",
                json!({
                    "text": "- [x] inspect\n- [ ] verify",
                    "revisionId": "proposed_plan:1",
                    "source": "proposed_plan",
                    "plan": [
                        {"step": "inspect", "status": "completed"},
                        {"step": "verify", "status": "in_progress"}
                    ]
                }),
            ),
            event(
                "next-turn-plan-final",
                3,
                "plan.final",
                "turn-2",
                json!({
                    "text": "- [ ] another",
                    "revisionId": "proposed_plan:1",
                    "source": "proposed_plan",
                    "plan": [{"step": "another", "status": "pending"}]
                }),
            ),
        ],
        "session-1",
        "thread-1",
    )
    .expect("materialize plan lifecycle");

    assert_eq!(changes.changed_items.len(), 2);
    let first = &changes.changed_items[0];
    assert_eq!(first.item_id.as_str(), "plan_turn-1_proposed_plan:1");
    assert_eq!(first.sequence, 2);
    assert_eq!(first.ordinal, 1);
    assert_eq!(first.status, ItemStatus::Completed);
    let ThreadItemPayload::Plan {
        text,
        revision_id,
        source,
        plan,
        ..
    } = &first.payload
    else {
        panic!("plan payload");
    };
    assert_eq!(text, "- [x] inspect\n- [ ] verify");
    assert_eq!(revision_id, "proposed_plan:1");
    assert_eq!(source.as_deref(), Some("proposed_plan"));
    assert_eq!(plan.len(), 2);
    assert_eq!(plan[1].status, PlanStepStatus::InProgress);

    assert_eq!(
        changes.changed_items[1].item_id.as_str(),
        "plan_turn-2_proposed_plan:1"
    );
}

#[test]
fn queue_events_materialize_canonical_turn_state_and_removal() {
    let queue_event = |id: &str, sequence: u64, event_type: &str| AgentEvent {
        event_id: id.to_string(),
        sequence,
        session_id: "session-1".to_string(),
        thread_id: Some("thread-1".to_string()),
        turn_id: Some("wrong-outer-turn".to_string()),
        event_type: event_type.to_string(),
        timestamp: format!("2026-07-12T00:00:{sequence:02}Z"),
        payload: json!({"queuedTurnId": "turn-queued", "position": 2}),
    };

    let added = materialize_events(
        &[queue_event("queue-added", 1, "queue.added")],
        "session-1",
        "thread-1",
    )
    .expect("materialize queue added");
    assert_eq!(added.changed_turns.len(), 1);
    assert_eq!(added.changed_turns[0].turn_id.as_str(), "turn-queued");
    assert_eq!(added.changed_turns[0].status, TurnStatus::InProgress);
    assert_eq!(
        added.changed_turns[0].queue,
        TurnQueueState::Queued { position: Some(2) }
    );

    let promoted = materialize_events(
        &[queue_event("queue-promoted", 2, "queue.promoted")],
        "session-1",
        "thread-1",
    )
    .expect("materialize queue promoted");
    assert!(promoted.changed_turns.is_empty());
    assert!(promoted.removed_turn_ids.is_empty());

    let started = materialize_events(
        &[event(
            "turn-started",
            3,
            "turn.started",
            "turn-queued",
            json!({}),
        )],
        "session-1",
        "thread-1",
    )
    .expect("materialize turn started");
    assert_eq!(started.changed_turns[0].status, TurnStatus::InProgress);
    assert_eq!(started.changed_turns[0].queue, TurnQueueState::Running);

    let removed = materialize_events(
        &[queue_event("queue-removed", 4, "queue.removed")],
        "session-1",
        "thread-1",
    )
    .expect("materialize queue removed");
    assert!(removed.changed_turns.is_empty());
    assert_eq!(removed.removed_turn_ids.len(), 1);
    assert_eq!(removed.removed_turn_ids[0].as_str(), "turn-queued");
}

#[test]
fn coalesces_message_delta_batch_into_the_same_typed_item() {
    let changes = materialize_events(
        &[
            event(
                "event-1",
                1,
                "message.delta",
                "turn-1",
                json!({
                    "itemId": "msg-1",
                    "role": "assistant",
                    "text": "hello ",
                    "phase": "commentary"
                }),
            ),
            event(
                "event-2",
                2,
                "message.delta_batch",
                "turn-1",
                json!({
                    "itemId": "msg-1",
                    "role": "assistant",
                    "deltas": [{"text": "world"}, {"text": "!"}],
                    "phase": "commentary"
                }),
            ),
        ],
        "session-1",
        "thread-1",
    )
    .expect("materialize");

    assert_eq!(changes.changed_items.len(), 1);
    assert_eq!(
        changes.changed_items[0].payload,
        ThreadItemPayload::AgentMessage {
            text: "hello world!".to_string(),
            phase: Some(agent_protocol::response_item::MessagePhase::Commentary),
            content_parts: Vec::new(),
        }
    );
}

#[test]
fn maps_core_families_to_typed_payloads() {
    let changes = materialize_events(
        &[
            event(
                "user",
                1,
                "message.created",
                "turn-1",
                json!({"text": "hi"}),
            ),
            event(
                "reasoning",
                2,
                "reasoning.summary",
                "turn-1",
                json!({"summary": ["why"]}),
            ),
            event(
                "tool",
                3,
                "item.started",
                "turn-1",
                json!({"item": canonical_tool_item("core-tool", "inProgress", 3)}),
            ),
            event(
                "approval",
                4,
                "action.required",
                "turn-1",
                json!({"actionKind": "tool", "description": "run"}),
            ),
            event(
                "subagent",
                5,
                "subagent.activity",
                "turn-1",
                json!({"childThreadId": "child", "activity": "started"}),
            ),
            event(
                "compact",
                6,
                "context.compaction.completed",
                "turn-1",
                json!({
                    "summary": "trimmed",
                    "replacementHistory": [{
                        "role": "user",
                        "content": [{"type": "input_text", "text": "compacted prefix"}]
                    }],
                    "windowNumber": 2,
                    "firstWindowId": "01890f3e-7b12-7cc0-8c2f-2d0fdab7c123",
                    "previousWindowId": "01890f3e-7b12-7cc0-8c2f-2d0fdab7c124",
                    "windowId": "01890f3e-7b12-7cc0-8c2f-2d0fdab7c125",
                    "tailStartTurnId": "turn-1"
                }),
            ),
        ],
        "session-1",
        "thread-1",
    )
    .expect("materialize");

    assert_eq!(changes.changed_items.len(), 6);
    assert!(changes
        .changed_items
        .iter()
        .any(|item| matches!(item.payload, ThreadItemPayload::UserMessage { .. })));
    assert!(changes
        .changed_items
        .iter()
        .any(|item| matches!(item.payload, ThreadItemPayload::Reasoning { .. })));
    assert!(changes
        .changed_items
        .iter()
        .any(|item| matches!(item.payload, ThreadItemPayload::Tool { .. })));
    assert!(changes.changed_items.iter().any(|item| matches!(
        item.payload,
        ThreadItemPayload::Approval { decision: None, .. }
    )));
    assert!(changes.changed_items.iter().any(|item| matches!(
        item.payload,
        ThreadItemPayload::SubAgent {
            activity: agent_protocol::SubAgentActivityKind::Started,
            ..
        }
    )));
    let compaction = changes
        .changed_items
        .iter()
        .find(|item| matches!(item.payload, ThreadItemPayload::ContextCompaction { .. }))
        .expect("context compaction item");
    assert_eq!(
        compaction.payload,
        ThreadItemPayload::ContextCompaction {
            summary: Some("trimmed".to_string()),
            replacement_history: vec![json!({
                "role": "user",
                "content": [{"type": "input_text", "text": "compacted prefix"}]
            })],
            window_number: Some(2),
            first_window_id: Some("01890f3e-7b12-7cc0-8c2f-2d0fdab7c123".to_string()),
            previous_window_id: Some("01890f3e-7b12-7cc0-8c2f-2d0fdab7c124".to_string()),
            window_id: Some("01890f3e-7b12-7cc0-8c2f-2d0fdab7c125".to_string()),
            tail_start_turn_id: Some("turn-1".to_string()),
        }
    );
}

#[test]
fn user_message_materialization_preserves_ordered_input_parts() {
    let input = vec![
        AgentInput::Text {
            text: "inspect this".to_string(),
            text_elements: vec![TextElement::new(
                ByteRange { start: 0, end: 7 },
                Some("inspect".to_string()),
            )],
        },
        AgentInput::Image {
            uri: "https://example.com/remote.png".to_string(),
            detail: Some(ImageDetail::High),
        },
        AgentInput::LocalImage {
            path: "/tmp/local.png".to_string(),
            detail: Some(ImageDetail::Original),
        },
        AgentInput::Skill {
            name: "review".to_string(),
            path: "/skills/review/SKILL.md".to_string(),
        },
        AgentInput::Mention {
            name: "docs".to_string(),
            path: "app://docs".to_string(),
        },
    ];
    let changes = materialize_events(
        &[event(
            "multimodal-user",
            1,
            "message.created",
            "turn-1",
            json!({"input": input, "clientId": "client-1"}),
        )],
        "session-1",
        "thread-1",
    )
    .expect("materialize multimodal user message");

    assert_eq!(
        changes.changed_items[0].payload,
        ThreadItemPayload::UserMessage {
            content: input,
            client_id: Some("client-1".to_string()),
        }
    );
}

#[test]
fn invalid_structured_user_input_does_not_create_a_canonical_item() {
    let changes = materialize_events(
        &[event(
            "invalid-user",
            1,
            "message.created",
            "turn-1",
            json!({"input": [{"type": "image", "uri": ""}]}),
        )],
        "session-1",
        "thread-1",
    )
    .expect("invalid input event remains readable");

    assert!(changes.changed_items.is_empty());
}

#[test]
fn user_message_snapshot_does_not_restore_empty_content_from_a_previous_item() {
    let previous = materialize_events(
        &[event(
            "user-created",
            1,
            "message.created",
            "turn-1",
            json!({"input": [AgentInput::text("first")], "clientId": "client-1"}),
        )],
        "session-1",
        "thread-1",
    )
    .expect("materialize user message")
    .changed_items
    .into_iter()
    .next()
    .expect("canonical user item");
    let mut next = previous.clone();
    next.sequence = 2;
    next.payload = ThreadItemPayload::UserMessage {
        content: Vec::new(),
        client_id: None,
    };

    let merged = super::change_set::merge_item_snapshot(previous, next);
    assert_eq!(
        merged.payload,
        ThreadItemPayload::UserMessage {
            content: Vec::new(),
            client_id: Some("client-1".to_string()),
        }
    );
}

#[test]
fn canonical_display_payloads_merge_create_update_and_terminal_fields() {
    let changes = materialize_events(
        &[
            event(
                "tool-start",
                1,
                "item.started",
                "turn-1",
                json!({"item": {
                    "id": "tool-1",
                    "type": "tool_call",
                    "callId": "call-1",
                    "toolName": "rg",
                    "arguments": {"pattern": "needle"}
                }}),
            ),
            event(
                "tool-end",
                2,
                "item.completed",
                "turn-1",
                json!({"item": {
                    "id": "tool-1",
                    "type": "tool_call",
                    "callId": "call-1",
                    "toolName": "rg",
                    "output": "found",
                    "structuredContent": {"matches": 1},
                    "durationMs": 12,
                    "truncated": true,
                    "outputRef": "sidecar://tool-1"
                }}),
            ),
            event(
                "mcp-start",
                3,
                "item.started",
                "turn-1",
                json!({"item": {
                    "id": "mcp-1",
                    "type": "mcp_tool_call",
                    "callId": "call-mcp",
                    "serverName": "docs",
                    "toolName": "search",
                    "arguments": {"query": "typed items"}
                }}),
            ),
            event(
                "mcp-end",
                4,
                "item.completed",
                "turn-1",
                json!({"item": {
                    "id": "mcp-1",
                    "type": "mcp_tool_call",
                    "callId": "call-mcp",
                    "serverName": "docs",
                    "toolName": "search",
                    "output": "done"
                }}),
            ),
            event(
                "collab-start",
                5,
                "item.started",
                "turn-1",
                json!({"item": {
                    "id": "collab-1",
                    "type": "collab_agent_tool_call",
                    "callId": "call-collab",
                    "operation": "wait"
                }}),
            ),
            event(
                "collab-end",
                6,
                "item.completed",
                "turn-1",
                json!({"item": {
                    "id": "collab-1",
                    "type": "collab_agent_tool_call",
                    "callId": "call-collab",
                    "operation": "wait",
                    "output": "ready"
                }}),
            ),
            event(
                "approval-start",
                7,
                "approval.required",
                "turn-1",
                json!({
                    "actionId": "approval-1",
                    "requestId": "request-1",
                    "actionKind": "tool",
                    "description": "run command",
                    "scope": {
                        "sessionId": "session-1",
                        "threadId": "thread-1",
                        "turnId": "turn-1"
                    },
                    "availableDecisions": [
                        "allow_once",
                        "allow_for_session",
                        "decline",
                        "cancel"
                    ],
                    "expiresAtMs": 99
                }),
            ),
            event(
                "approval-end",
                8,
                "approval.resolved",
                "turn-1",
                json!({
                    "actionId": "approval-1",
                    "requestId": "request-1",
                    "actionKind": "tool",
                    "description": "run command",
                    "decision": "allow_for_session",
                    "decisionScope": "session",
                    "reasonCode": "user_approved"
                }),
            ),
        ],
        "session-1",
        "thread-1",
    )
    .expect("materialize");

    assert_eq!(changes.changed_items.len(), 4);
    let tool = changes
        .changed_items
        .iter()
        .find(|item| item.item_id.as_str() == "item_tool-1")
        .expect("tool item");
    assert_eq!(tool.status, ItemStatus::Completed);
    let ThreadItemPayload::Tool { output, .. } = &tool.payload else {
        panic!("tool payload");
    };
    let output = output.as_ref().expect("tool output");
    assert_eq!(output.text.as_deref(), Some("found"));
    assert_eq!(output.structured_content, Some(json!({"matches": 1})));
    assert_eq!(output.duration_ms, Some(12));
    assert!(output.truncated);
    assert_eq!(output.output_ref.as_deref(), Some("sidecar://tool-1"));

    assert!(changes.changed_items.iter().any(|item| matches!(
        &item.payload,
        ThreadItemPayload::McpToolCall {
            server_name,
            tool_name,
            ..
        } if server_name == "docs" && tool_name == "search"
    )));
    assert!(changes.changed_items.iter().any(|item| matches!(
        &item.payload,
        ThreadItemPayload::CollabAgentToolCall {
            operation: CollabAgentOperation::Wait,
            ..
        }
    )));

    let approval = changes
        .changed_items
        .iter()
        .find(|item| item.item_id.as_str() == "item_request-1")
        .expect("approval item");
    assert_eq!(approval.status, ItemStatus::Completed);
    assert!(matches!(
        &approval.payload,
        ThreadItemPayload::Approval {
            request_id,
            scope: ApprovalScope::Session,
            available_decisions,
            decision: Some(ApprovalDecision::ApprovedForSession),
            requested_at_ms: Some(_),
            resolved_at_ms: Some(_),
            reason_code: Some(reason),
            ..
        } if request_id == "request-1"
            && available_decisions == &vec![
                ApprovalDecision::Approved,
                ApprovalDecision::ApprovedForSession,
                ApprovalDecision::Denied,
                ApprovalDecision::Abort,
            ]
            && reason == "user_approved"
    ));
}

#[test]
fn only_explicit_wait_materializes_as_collab_agent_tool_call() {
    let changes = materialize_events(
        &[
            event(
                "send-start",
                1,
                "item.started",
                "turn-1",
                json!({"item": {
                    "id": "send-1",
                    "type": "tool_call",
                    "callId": "call-send",
                    "toolName": "send_message",
                    "operation": "send_message",
                    "targetThreadId": "thread-child",
                    "message": "continue"
                }}),
            ),
            event(
                "send-end",
                2,
                "item.completed",
                "turn-1",
                json!({"item": {
                    "id": "send-1",
                    "type": "tool_call",
                    "callId": "call-send",
                    "toolName": "send_message",
                    "operation": "send_message",
                    "output": "delivered"
                }}),
            ),
            event(
                "legacy-collab-start",
                3,
                "item.started",
                "turn-1",
                json!({"item": {
                    "id": "legacy-collab-1",
                    "type": "collab_agent_tool_call",
                    "callId": "call-legacy-collab",
                    "operation": "send_message"
                }}),
            ),
            event(
                "raw-collab",
                4,
                "collab.completed",
                "turn-1",
                json!({
                    "callId": "call-raw-collab",
                    "operation": "wait"
                }),
            ),
        ],
        "session-1",
        "thread-1",
    )
    .expect("materialize collab producer boundary");

    assert_eq!(changes.changed_items.len(), 1);
    assert!(matches!(
        &changes.changed_items[0].payload,
        ThreadItemPayload::Tool { call_id, name, .. }
            if call_id == "call-send" && name == "send_message"
    ));
    assert!(!changes
        .changed_items
        .iter()
        .any(|item| matches!(item.payload, ThreadItemPayload::CollabAgentToolCall { .. })));
}

#[test]
fn tool_and_approval_with_shared_tool_call_id_remain_distinct_items() {
    let mut started = canonical_tool_item("tool-1", "inProgress", 1);
    started["payload"]["call_id"] = json!("tool-1");
    started["payload"]["name"] = json!("browser_control");
    let mut completed = canonical_tool_item("tool-1", "completed", 1);
    completed["payload"]["call_id"] = json!("tool-1");
    completed["payload"]["name"] = json!("browser_control");
    completed["payload"]["output"] = json!({"text": "opened"});

    let changes = materialize_events(
        &[
            event(
                "tool-start",
                1,
                "item.started",
                "turn-1",
                json!({"item": started}),
            ),
            event(
                "approval-start",
                2,
                "action.required",
                "turn-1",
                json!({
                    "toolCallId": "tool-1",
                    "requestId": "approval-1",
                    "actionId": "approval-1",
                    "actionType": "tool_confirmation",
                    "description": "allow browser control",
                    "scope": {
                        "sessionId": "session-1",
                        "threadId": "thread-1",
                        "turnId": "turn-1"
                    },
                    "availableDecisions": [
                        "allow_once",
                        "allow_for_session",
                        "decline",
                        "cancel"
                    ]
                }),
            ),
            event(
                "approval-end",
                3,
                "action.resolved",
                "turn-1",
                json!({
                    "toolCallId": "tool-1",
                    "requestId": "approval-1",
                    "actionId": "approval-1",
                    "actionType": "tool_confirmation",
                    "decision": "allow_for_session",
                    "decisionScope": "session"
                }),
            ),
            event(
                "tool-end",
                4,
                "item.completed",
                "turn-1",
                json!({"item": completed}),
            ),
        ],
        "session-1",
        "thread-1",
    )
    .expect("materialize distinct tool and approval items");

    assert_eq!(changes.changed_items.len(), 2);
    let tool = changes
        .changed_items
        .iter()
        .find(|item| item.item_id.as_str() == "tool-1")
        .expect("tool item");
    assert_eq!(tool.status, ItemStatus::Completed);
    assert!(matches!(tool.payload, ThreadItemPayload::Tool { .. }));

    let approval = changes
        .changed_items
        .iter()
        .find(|item| item.item_id.as_str() == "item_approval-1")
        .expect("approval item");
    assert_eq!(approval.status, ItemStatus::Completed);
    assert!(matches!(
        &approval.payload,
        ThreadItemPayload::Approval {
            request_id,
            scope: ApprovalScope::Session,
            decision: Some(ApprovalDecision::ApprovedForSession),
            ..
        } if request_id == "approval-1"
    ));
}

#[test]
fn request_id_does_not_collapse_user_agent_and_reasoning_items() {
    let changes = materialize_events(
        &[
            event(
                "user-created",
                1,
                "message.created",
                "turn-1",
                json!({
                    "request_id": "request-1",
                    "role": "user",
                    "input": [AgentInput::text("hello")]
                }),
            ),
            event(
                "reasoning-started",
                2,
                "reasoning.started",
                "turn-1",
                json!({
                    "request_id": "request-1",
                    "reasoningId": "runtime-thinking"
                }),
            ),
            event(
                "reasoning-delta",
                3,
                "reasoning.delta",
                "turn-1",
                json!({
                    "request_id": "request-1",
                    "reasoningId": "runtime-thinking",
                    "text": "inspect"
                }),
            ),
            event(
                "agent-delta",
                4,
                "message.delta",
                "turn-1",
                json!({
                    "request_id": "request-1",
                    "role": "assistant",
                    "text": "answer"
                }),
            ),
            event(
                "reasoning-final",
                5,
                "reasoning.final",
                "turn-1",
                json!({
                    "request_id": "request-1",
                    "reasoningId": "runtime-thinking",
                    "text": "inspect inputs"
                }),
            ),
            event(
                "reasoning-ended",
                6,
                "reasoning.ended",
                "turn-1",
                json!({
                    "request_id": "request-1",
                    "reasoningId": "runtime-thinking",
                    "status": "completed"
                }),
            ),
        ],
        "session-1",
        "thread-1",
    )
    .expect("materialize distinct turn items");

    assert_eq!(changes.changed_items.len(), 3);
    let user = &changes.changed_items[0];
    let reasoning = &changes.changed_items[1];
    let agent = &changes.changed_items[2];
    assert_eq!(user.item_id.as_str(), "item_user-turn-1");
    assert_eq!(reasoning.item_id.as_str(), "item_reasoning-turn-1");
    assert_eq!(agent.item_id.as_str(), "item_agent-turn-1");
    assert!(matches!(
        user.payload,
        ThreadItemPayload::UserMessage { .. }
    ));
    assert!(matches!(
        reasoning.payload,
        ThreadItemPayload::Reasoning { .. }
    ));
    assert!(matches!(
        agent.payload,
        ThreadItemPayload::AgentMessage { .. }
    ));
    assert_eq!(reasoning.ordinal, 2);
    assert_eq!(reasoning.sequence, 6);
    assert_eq!(agent.ordinal, 4);
    assert_eq!(agent.sequence, 4);
}
