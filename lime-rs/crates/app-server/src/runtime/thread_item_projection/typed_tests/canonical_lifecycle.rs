use super::*;
use agent_protocol::{
    CollabAgentOperation, CollabAgentState, CollabAgentStatus, MessageContentPart, SessionId,
    ThreadId, ThreadItem, TurnId,
};
use std::collections::HashMap;

#[test]
fn canonical_nested_item_uses_outer_event_identity_time_and_lifecycle_status() {
    let changes = materialize_events(
        &[
            event(
                "canonical-start",
                1,
                "item.started",
                "outer-turn",
                json!({"item": canonical_tool_item("nested-start", "completed", 101)}),
            ),
            event(
                "canonical-update",
                2,
                "item.updated",
                "outer-turn",
                json!({"item": canonical_tool_item("nested-update", "completed", 102)}),
            ),
            event(
                "canonical-failed",
                3,
                "item.completed",
                "outer-turn",
                json!({"item": canonical_tool_item("nested-failed", "failed", 103)}),
            ),
        ],
        "session-1",
        "thread-1",
    )
    .expect("materialize canonical items");

    assert_eq!(changes.changed_items.len(), 3);
    assert_eq!(changes.changed_items[0].status, ItemStatus::InProgress);
    assert_eq!(changes.changed_items[1].status, ItemStatus::InProgress);

    let failed = &changes.changed_items[2];
    assert_eq!(failed.session_id.as_str(), "session-1");
    assert_eq!(failed.thread_id.as_str(), "thread-1");
    assert_eq!(failed.turn_id.as_str(), "outer-turn");
    assert_eq!(failed.item_id.as_str(), "nested-failed");
    assert_eq!(failed.sequence, 3);
    assert_eq!(failed.ordinal, 3);
    assert_eq!(failed.created_at_ms, 1_783_814_403_000);
    assert_eq!(failed.updated_at_ms, 1_783_814_403_000);
    assert_eq!(failed.completed_at_ms, Some(1_783_814_403_000));
    assert_eq!(failed.kind, ItemKind::Tool);
    assert_eq!(failed.status, ItemStatus::Failed);
    assert_eq!(failed.metadata, json!({"source": "nested-canonical"}));
}

#[test]
fn canonical_nested_tool_output_and_metadata_are_preserved_across_lifecycle() {
    let mut started = canonical_tool_item("nested-tool", "pending", 7);
    started["payload"]["arguments"] = json!([
        {"name": "pattern", "value": "needle"},
        {"name": "path", "value": "src"}
    ]);
    started["payload"]["output"] = serde_json::Value::Null;

    let mut completed = canonical_tool_item("nested-tool", "completed", 99);
    completed["payload"]["arguments"] = json!([]);
    completed["payload"]["output"] = json!({
        "text": "found",
        "structuredContent": {"matches": [{"path": "src/lib.rs", "line": 12}]},
        "durationMs": 42,
        "truncated": true,
        "outputRef": "sidecar://nested-tool"
    });
    completed["metadata"] = json!({
        "source": "runtime-tool",
        "extension": {"provider": "local"}
    });

    let changes = materialize_events(
        &[
            event(
                "nested-tool-start",
                1,
                "item.started",
                "outer-turn",
                json!({"item": started}),
            ),
            event(
                "nested-tool-completed",
                2,
                "item.completed",
                "outer-turn",
                json!({"item": completed}),
            ),
        ],
        "session-1",
        "thread-1",
    )
    .expect("materialize nested canonical tool");

    assert_eq!(changes.changed_items.len(), 1);
    let item = &changes.changed_items[0];
    assert_eq!(item.item_id.as_str(), "nested-tool");
    assert_eq!(item.sequence, 2);
    assert_eq!(item.ordinal, 1);
    assert_eq!(item.created_at_ms, 1_783_814_401_000);
    assert_eq!(item.updated_at_ms, 1_783_814_402_000);
    assert_eq!(item.completed_at_ms, Some(1_783_814_402_000));
    assert_eq!(item.status, ItemStatus::Completed);
    assert_eq!(item.kind, ItemKind::Tool);
    assert_eq!(
        item.metadata,
        json!({
            "source": "runtime-tool",
            "extension": {"provider": "local"}
        })
    );

    let ThreadItemPayload::Tool {
        call_id,
        name,
        arguments,
        output,
    } = &item.payload
    else {
        panic!("canonical tool payload");
    };
    assert_eq!(call_id, "call-nested-tool");
    assert_eq!(name, "rg");
    assert_eq!(arguments.len(), 2);
    let output = output.as_ref().expect("canonical tool output");
    assert_eq!(output.text.as_deref(), Some("found"));
    assert_eq!(
        output.structured_content,
        Some(json!({"matches": [{"path": "src/lib.rs", "line": 12}]}))
    );
    assert_eq!(output.duration_ms, Some(42));
    assert!(output.truncated);
    assert_eq!(output.output_ref.as_deref(), Some("sidecar://nested-tool"));
}

#[test]
fn canonical_wait_agent_states_survive_materialization_and_lifecycle_merge() {
    let mut started = ThreadItem::new(
        SessionId::new("nested-session"),
        ThreadId::new("nested-thread"),
        TurnId::new("nested-turn"),
        7,
        7,
        ThreadItemPayload::CollabAgentToolCall {
            call_id: "wait-call".to_string(),
            operation: CollabAgentOperation::Wait,
            target_thread_id: None,
            message: None,
            output: None,
            agent_states: HashMap::new(),
        },
    );
    started.status = ItemStatus::InProgress;
    let mut completed = started.clone();
    completed.sequence = 8;
    completed.status = ItemStatus::Completed;
    completed.completed_at_ms = Some(8);
    let ThreadItemPayload::CollabAgentToolCall { agent_states, .. } = &mut completed.payload else {
        panic!("wait payload");
    };
    agent_states.insert(
        ThreadId::new("child-thread"),
        CollabAgentState {
            status: CollabAgentStatus::Completed,
            message: None,
        },
    );

    let changes = materialize_events(
        &[
            event(
                "wait-started",
                1,
                "item.started",
                "outer-turn",
                json!({"item": started}),
            ),
            event(
                "wait-completed",
                2,
                "item.completed",
                "outer-turn",
                json!({"item": completed}),
            ),
        ],
        "session-1",
        "thread-1",
    )
    .expect("materialize canonical wait lifecycle");

    assert_eq!(changes.changed_items.len(), 1);
    let ThreadItemPayload::CollabAgentToolCall { agent_states, .. } =
        &changes.changed_items[0].payload
    else {
        panic!("materialized wait payload");
    };
    assert_eq!(
        agent_states[&ThreadId::new("child-thread")].status,
        CollabAgentStatus::Completed
    );
}

#[test]
fn current_codex_import_uses_lime_event_log_ordinal() {
    let changes = materialize_events(
        &[event(
            "imported-tool-start",
            11,
            "item.started",
            "outer-turn",
            json!({
                "imported": true,
                "sourceClient": "codex",
                "importVersion": 2,
                "item": canonical_tool_item("imported-tool", "inProgress", 7)
            }),
        )],
        "session-1",
        "thread-1",
    )
    .expect("materialize current Codex import");

    assert_eq!(changes.changed_items.len(), 1);
    assert_eq!(changes.changed_items[0].ordinal, 11);
}

#[test]
fn canonical_agent_message_merges_typed_content_parts_across_lifecycle() {
    let changes = materialize_events(
        &[
            event(
                "media-message-started",
                1,
                "item.started",
                "turn-media",
                json!({
                    "item": {
                        "id": "media-message",
                        "type": "agent_message",
                        "text": "result",
                        "contentParts": [{"type": "text", "text": "result"}]
                    }
                }),
            ),
            event(
                "media-message-updated",
                2,
                "item.updated",
                "turn-media",
                json!({
                    "item": {
                        "id": "media-message",
                        "type": "agent_message",
                        "text": "result",
                        "contentParts": [
                            {"type": "text", "text": "result"},
                            {
                                "type": "media",
                                "kind": "image",
                                "caption": "draft image",
                                "reference": {
                                    "uri": "sidecar://media/result",
                                    "mime_type": "image/png",
                                    "source_path": "/tmp/media/draft.png"
                                }
                            }
                        ]
                    }
                }),
            ),
            event(
                "media-message-completed",
                3,
                "item.completed",
                "turn-media",
                json!({
                    "item": {
                        "id": "media-message",
                        "type": "agent_message",
                        "text": "result",
                        "phase": "final_answer",
                        "contentPart": {
                            "type": "media",
                            "kind": "image",
                            "caption": "result image",
                            "reference": {
                                "uri": "sidecar://media/result",
                                "mime_type": "image/png",
                                "source_path": "/tmp/media/result.png"
                            }
                        }
                    }
                }),
            ),
        ],
        "session-1",
        "thread-1",
    )
    .expect("materialize canonical media message");

    assert_eq!(changes.changed_items.len(), 1);
    let item = &changes.changed_items[0];
    assert_eq!(item.status, ItemStatus::Completed);
    let ThreadItemPayload::AgentMessage {
        text,
        phase,
        content_parts,
    } = &item.payload
    else {
        panic!("canonical agent message payload");
    };
    assert_eq!(text, "result");
    assert_eq!(
        phase,
        &Some(agent_protocol::response_item::MessagePhase::FinalAnswer)
    );
    assert_eq!(content_parts.len(), 2);
    assert!(matches!(
        &content_parts[0],
        MessageContentPart::Text { text } if text == "result"
    ));
    assert!(matches!(
        &content_parts[1],
        MessageContentPart::Media { kind, reference, .. }
            if kind == "image"
                && reference.uri == "sidecar://media/result"
                && reference.mime_type == "image/png"
                && reference.source_path.as_deref() == Some("/tmp/media/result.png")
    ));
}

#[test]
fn canonical_agent_message_preserves_interleaved_text_image_and_file_order() {
    let content_parts = json!([
        {"type": "text", "text": "before"},
        {
            "type": "media",
            "kind": "image",
            "reference": {
                "uri": "sidecar://media/image-1",
                "mime_type": "image/png"
            }
        },
        {"type": "text", "text": "between"},
        {
            "type": "media",
            "kind": "file",
            "reference": {
                "uri": "sidecar://media/spec-1",
                "mime_type": "application/pdf"
            }
        }
    ]);
    let changes = materialize_events(
        &[
            event(
                "mixed-message-started",
                1,
                "item.started",
                "turn-mixed-media",
                json!({
                    "item": {
                        "id": "mixed-media-message",
                        "type": "agent_message",
                        "status": "in_progress",
                        "contentParts": content_parts
                    }
                }),
            ),
            event(
                "mixed-message-completed",
                2,
                "item.completed",
                "turn-mixed-media",
                json!({
                    "item": {
                        "id": "mixed-media-message",
                        "type": "agent_message",
                        "status": "completed",
                        "phase": "final_answer",
                        "contentParts": content_parts
                    }
                }),
            ),
        ],
        "session-1",
        "thread-1",
    )
    .expect("materialize interleaved content parts");

    let ThreadItemPayload::AgentMessage { content_parts, .. } = &changes.changed_items[0].payload
    else {
        panic!("canonical agent message payload");
    };
    assert_eq!(
        serde_json::to_value(content_parts).expect("serialize content parts"),
        json!([
            {"type": "text", "text": "before"},
            {
                "type": "media",
                "kind": "image",
                "reference": {
                    "uri": "sidecar://media/image-1",
                    "mime_type": "image/png"
                }
            },
            {"type": "text", "text": "between"},
            {
                "type": "media",
                "kind": "file",
                "reference": {
                    "uri": "sidecar://media/spec-1",
                    "mime_type": "application/pdf"
                }
            }
        ])
    );
}

#[test]
fn malformed_or_inline_agent_message_content_parts_are_rejected() {
    let changes = materialize_events(
        &[
            event(
                "malformed-content-parts",
                1,
                "item.started",
                "turn-media",
                json!({
                    "item": {
                        "id": "malformed-message",
                        "type": "agent_message",
                        "text": "must not downgrade",
                        "contentParts": {"type": "media"}
                    }
                }),
            ),
            event(
                "inline-content-part",
                2,
                "item.started",
                "turn-media",
                json!({
                    "item": {
                        "id": "inline-message",
                        "type": "agent_message",
                        "text": "must not persist",
                        "contentParts": [{
                            "type": "media",
                            "kind": "image",
                            "reference": {
                                "uri": "data:image/png;base64,AAAA",
                                "mime_type": "image/png"
                            }
                        }]
                    }
                }),
            ),
        ],
        "session-1",
        "thread-1",
    )
    .expect("reject invalid content parts without failing unrelated turn projection");

    assert!(changes.changed_items.is_empty());
}

#[test]
fn resolved_user_input_without_decision_is_terminal_and_clears_pending() {
    let changes = materialize_events(
        &[
            event(
                "approval-start",
                1,
                "action.required",
                "turn-1",
                json!({
                    "requestId": "approval-1",
                    "actionType": "ask_user",
                    "prompt": "choose"
                }),
            ),
            event(
                "approval-ambiguous",
                2,
                "action.resolved",
                "turn-1",
                json!({
                    "requestId": "approval-1",
                    "actionType": "ask_user"
                }),
            ),
        ],
        "session-1",
        "thread-1",
    )
    .expect("materialize");

    assert_eq!(changes.changed_items[0].status, ItemStatus::Completed);
    assert!(matches!(
        &changes.changed_items[0].payload,
        ThreadItemPayload::Approval {
            request_id,
            action: ApprovalAction { kind, description },
            decision: None,
            requested_at_ms: Some(_),
            resolved_at_ms: Some(_),
            ..
        } if request_id == "approval-1" && kind == "ask_user" && description == "choose"
    ));
    assert_eq!(
        changes.changed_turns[0].approval,
        TurnApprovalState::Resolved
    );
}

#[test]
fn approval_session_cache_hit_remains_audit_only_before_auto_resolution() {
    let changes = materialize_events(
        &[
            event(
                "approval-cache-hit",
                1,
                "approval.session_cache.hit",
                "turn-1",
                json!({
                    "request_id": "provider-request-1",
                    "sourceRequestId": "approval-turn-initial",
                    "decision": "allow_for_session",
                    "decisionScope": "session",
                }),
            ),
            event(
                "approval-auto-resolved",
                2,
                "action.resolved",
                "turn-1",
                json!({
                    "requestId": "permission-turn-1",
                    "actionId": "permission-turn-1",
                    "actionType": "tool_confirmation",
                    "source": "approval_session_cache",
                    "decision": "allow_for_session",
                    "decisionScope": "session",
                }),
            ),
        ],
        "session-1",
        "thread-1",
    )
    .expect("materialize cache-backed approval resolution");

    assert_eq!(changes.changed_items.len(), 1);
    assert!(matches!(
        &changes.changed_items[0].payload,
        ThreadItemPayload::Approval {
            request_id,
            decision: Some(ApprovalDecision::ApprovedForSession),
            requested_at_ms: None,
            resolved_at_ms: Some(_),
            ..
        } if request_id == "permission-turn-1"
    ));
    assert_eq!(changes.changed_items[0].status, ItemStatus::Completed);
    assert_eq!(
        changes.changed_turns[0].approval,
        TurnApprovalState::Approved
    );
}

#[test]
fn approval_uses_top_level_identity_when_data_and_trace_request_id_are_present() {
    let changes = materialize_events(
        &[event(
            "approval-start",
            1,
            "action.required",
            "turn-1",
            json!({
                "request_id": "trace-request-1",
                "actionType": "tool_confirmation",
                "actionKind": "permission_preflight",
                "data": {
                    "prompt": "allow browser control",
                    "availableDecisions": ["allow_once", "decline", "cancel"]
                },
                "runtimeEvent": {
                    "type": "action_required",
                    "request_id": "approval-1",
                    "action_type": "tool_confirmation",
                    "data": {
                        "prompt": "allow browser control",
                        "availableDecisions": ["allow_once", "decline", "cancel"]
                    }
                }
            }),
        )],
        "session-1",
        "thread-1",
    )
    .expect("materialize approval with nested data");

    let approval = changes
        .changed_items
        .iter()
        .find(|item| item.item_id.as_str() == "item_approval-1")
        .expect("top-level approval identity");
    assert!(matches!(
        &approval.payload,
        ThreadItemPayload::Approval {
            request_id,
            action: ApprovalAction { kind, description },
            available_decisions,
            ..
        } if request_id == "approval-1"
            && kind == "tool_confirmation"
            && description == "allow browser control"
            && available_decisions == &vec![
                ApprovalDecision::Approved,
                ApprovalDecision::Denied,
                ApprovalDecision::Abort,
            ]
    ));
}

#[test]
fn duplicate_and_stale_events_do_not_regress_the_snapshot() {
    let changes = materialize_events(
        &[
            event(
                "event-1",
                2,
                "message.delta",
                "turn-1",
                json!({"itemId": "msg", "text": "new"}),
            ),
            event(
                "event-1",
                2,
                "message.delta",
                "turn-1",
                json!({"itemId": "msg", "text": "new"}),
            ),
            event(
                "event-old",
                1,
                "message.delta",
                "turn-1",
                json!({"itemId": "old", "text": "old"}),
            ),
        ],
        "session-1",
        "thread-1",
    )
    .expect("materialize");
    assert_eq!(changes.changed_items.len(), 1);
    assert_eq!(changes.changed_items[0].sequence, 2);
}

#[test]
fn rollback_removes_items_after_target_sequence() {
    let changes = materialize_events(
        &[
            event(
                "one",
                1,
                "message.delta",
                "turn-1",
                json!({"itemId": "one", "text": "one"}),
            ),
            event(
                "two",
                2,
                "message.delta",
                "turn-1",
                json!({"itemId": "two", "text": "two"}),
            ),
            event(
                "rollback",
                3,
                "history.rollback",
                "turn-1",
                json!({"rollbackToSequence": 1}),
            ),
        ],
        "session-1",
        "thread-1",
    )
    .expect("materialize");
    assert_eq!(changes.rollback_to_sequence, Some(1));
    assert_eq!(changes.removed_item_ids.len(), 1);
    assert_eq!(changes.removed_item_ids[0].as_str(), "item_two");
}

#[test]
fn explicit_remove_and_item_lifecycle_events_use_the_same_identity() {
    let changes = materialize_events(
        &[
            event(
                "start",
                1,
                "item.started",
                "turn-1",
                json!({"item": {"id": "agent-1", "type": "agent_message", "text": "hello"}}),
            ),
            event(
                "remove",
                2,
                "item.removed",
                "turn-1",
                json!({"itemId": "agent-1"}),
            ),
        ],
        "session-1",
        "thread-1",
    )
    .expect("materialize");

    assert!(changes.changed_items.is_empty());
    assert_eq!(changes.removed_item_ids[0].as_str(), "item_agent-1");
}

#[test]
fn identity_collisions_are_reported() {
    let error = materialize_events(
        &[
            event("same", 1, "message.delta", "turn-1", json!({"text": "one"})),
            event("same", 2, "message.delta", "turn-1", json!({"text": "two"})),
        ],
        "session-1",
        "thread-1",
    )
    .expect_err("collision");
    assert!(matches!(
        error,
        super::super::change_set::MaterializationError::EventIdentityCollision { .. }
    ));
}
