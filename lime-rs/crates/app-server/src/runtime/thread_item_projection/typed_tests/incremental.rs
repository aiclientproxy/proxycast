use super::super::materializer::IncrementalMaterializer;
use super::{event, materialize_events};
use agent_protocol::hook::{
    HookEventName, HookExecutionMode, HookHandlerType, HookOutputEntry, HookOutputEntryKind,
    HookRunStatus, HookScope, HookSource,
};
use agent_protocol::{
    ItemId, ItemStatus, SessionId, ThreadId, ThreadItem, ThreadItemPayload, TurnId,
};
use serde_json::json;

#[test]
fn incremental_item_snapshot_matches_full_history_materialization() {
    let first = event(
        "event-1",
        1,
        "message.delta",
        "turn-1",
        json!({"itemId": "msg-1", "role": "assistant", "text": "hel"}),
    );
    let second = event(
        "event-2",
        2,
        "message.delta",
        "turn-1",
        json!({"itemId": "msg-1", "role": "assistant", "text": "hello"}),
    );
    let expected = materialize_events(&[first.clone(), second.clone()], "session-1", "thread-1")
        .expect("full history materialization");

    let mut incremental = IncrementalMaterializer::from_events(&[], "session-1", "thread-1")
        .expect("incremental materializer");
    incremental.apply(&first).expect("first event");
    let entities = incremental.apply(&second).expect("second event");

    assert_eq!(entities.item.as_ref(), expected.changed_items.first());
    assert_eq!(entities.turn.as_ref(), expected.changed_turns.first());
}

#[test]
fn reasoning_deltas_preserve_repeated_fragments_and_final_snapshot() {
    let events = [
        event(
            "reasoning-1",
            1,
            "reasoning.delta",
            "turn-1",
            json!({"reasoningId": "reasoning-1", "delta": "你"}),
        ),
        event(
            "reasoning-2",
            2,
            "reasoning.delta",
            "turn-1",
            json!({"reasoningId": "reasoning-1", "delta": "好"}),
        ),
        event(
            "reasoning-3",
            3,
            "reasoning.delta",
            "turn-1",
            json!({"reasoningId": "reasoning-1", "delta": "你"}),
        ),
        event(
            "reasoning-final",
            4,
            "reasoning.final",
            "turn-1",
            json!({"reasoningId": "reasoning-1", "text": "你好你"}),
        ),
    ];

    let deltas = materialize_events(&events[..3], "session-1", "thread-1")
        .expect("materialize repeated reasoning deltas");
    let delta_reasoning = deltas
        .changed_items
        .iter()
        .find(|item| {
            matches!(
                item.payload,
                agent_protocol::ThreadItemPayload::Reasoning { .. }
            )
        })
        .expect("reasoning delta item");

    assert_eq!(
        delta_reasoning.payload,
        agent_protocol::ThreadItemPayload::Reasoning {
            summary: Vec::new(),
            content: vec!["你".to_string(), "好".to_string(), "你".to_string()],
        }
    );

    let changes = materialize_events(&events, "session-1", "thread-1")
        .expect("materialize repeated reasoning deltas");
    let reasoning = changes
        .changed_items
        .iter()
        .find(|item| {
            matches!(
                item.payload,
                agent_protocol::ThreadItemPayload::Reasoning { .. }
            )
        })
        .expect("reasoning item");

    assert_eq!(
        reasoning.payload,
        agent_protocol::ThreadItemPayload::Reasoning {
            summary: Vec::new(),
            content: vec!["你好你".to_string()],
        }
    );
}

#[test]
fn empty_reasoning_completion_preserves_materialized_content() {
    let delta = materialize_events(
        &[
            event(
                "reasoning-started",
                1,
                "reasoning.started",
                "turn-1",
                json!({"reasoningId": "reasoning-1"}),
            ),
            event(
                "reasoning-delta",
                2,
                "reasoning.delta",
                "turn-1",
                json!({"reasoningId": "reasoning-1", "text": "inspect inputs"}),
            ),
        ],
        "session-1",
        "thread-1",
    )
    .expect("materialize reasoning delta");
    let mut completed_item = delta
        .changed_items
        .into_iter()
        .find(|item| {
            matches!(
                item.payload,
                agent_protocol::ThreadItemPayload::Reasoning { .. }
            )
        })
        .expect("reasoning delta item");
    completed_item.status = agent_protocol::ItemStatus::Completed;
    completed_item.completed_at_ms = Some(3);
    completed_item.sequence = 3;
    completed_item.updated_at_ms = 3;
    completed_item.metadata = json!({"source_event_type": "item.completed"});
    completed_item.payload = agent_protocol::ThreadItemPayload::Reasoning {
        summary: Vec::new(),
        content: Vec::new(),
    };

    let changes = materialize_events(
        &[
            event(
                "reasoning-started",
                1,
                "reasoning.started",
                "turn-1",
                json!({"reasoningId": "reasoning-1"}),
            ),
            event(
                "reasoning-delta",
                2,
                "reasoning.delta",
                "turn-1",
                json!({"reasoningId": "reasoning-1", "text": "inspect inputs"}),
            ),
            event(
                "reasoning-completed",
                3,
                "item.completed",
                "turn-1",
                json!({"item": serde_json::to_value(completed_item).expect("serialize item")}),
            ),
        ],
        "session-1",
        "thread-1",
    )
    .expect("materialize empty reasoning completion");
    let reasoning = changes
        .changed_items
        .iter()
        .find(|item| {
            matches!(
                item.payload,
                agent_protocol::ThreadItemPayload::Reasoning { .. }
            )
        })
        .expect("completed reasoning item");
    assert_eq!(reasoning.status, agent_protocol::ItemStatus::Completed);
    assert_eq!(
        reasoning.payload,
        agent_protocol::ThreadItemPayload::Reasoning {
            summary: Vec::new(),
            content: vec!["inspect inputs".to_string()],
        }
    );
}

#[test]
fn incremental_materializer_does_not_revive_removed_item_identity() {
    let started = event(
        "event-started",
        1,
        "item.started",
        "turn-1",
        json!({"item": {"id": "agent-1", "type": "agent_message", "text": "hello"}}),
    );
    let removed = event(
        "event-removed",
        2,
        "item.removed",
        "turn-1",
        json!({"itemId": "agent-1"}),
    );
    let completed = event(
        "event-completed",
        3,
        "item.completed",
        "turn-1",
        json!({"item": {"id": "agent-1", "type": "agent_message", "text": "late"}}),
    );
    let expected = materialize_events(
        &[started.clone(), removed.clone(), completed.clone()],
        "session-1",
        "thread-1",
    )
    .expect("full history materialization");

    let mut incremental =
        IncrementalMaterializer::from_events(&[started, removed], "session-1", "thread-1")
            .expect("incremental materializer");
    let entities = incremental.apply(&completed).expect("late item event");

    assert!(expected.changed_items.is_empty());
    assert!(entities.item.is_none());
}

#[test]
fn hook_lifecycle_reuses_one_canonical_item_through_history_restore() {
    fn hook_item(status: HookRunStatus, completed_at: Option<i64>) -> ThreadItem {
        let mut item = ThreadItem::new(
            SessionId::new("session-1"),
            ThreadId::new("thread-1"),
            TurnId::new("turn-1"),
            0,
            0,
            ThreadItemPayload::Hook {
                run: agent_protocol::hook::HookRunSummary {
                    id: "hook-run-restore".to_string(),
                    event_name: HookEventName::PreToolUse,
                    handler_type: HookHandlerType::Command,
                    execution_mode: HookExecutionMode::Sync,
                    scope: HookScope::Turn,
                    source_path: "/workspace/.codex/hooks/check.sh".into(),
                    source: HookSource::Project,
                    display_order: 0,
                    status,
                    status_message: Some("checking".to_string()),
                    started_at: 1_783_814_400_900,
                    completed_at,
                    duration_ms: completed_at.map(|_| 600),
                    entries: vec![HookOutputEntry {
                        kind: HookOutputEntryKind::Feedback,
                        text: "checking".to_string(),
                    }],
                },
            },
        );
        item.item_id = ItemId::new("item_hook-run-restore");
        item.status = if completed_at.is_some() {
            ItemStatus::Completed
        } else {
            ItemStatus::InProgress
        };
        item
    }

    let started = event(
        "hook-started",
        1,
        "hook.started",
        "turn-1",
        json!({"item": serde_json::to_value(hook_item(HookRunStatus::Running, None)).expect("hook started")}),
    );
    let completed_item = hook_item(HookRunStatus::Completed, Some(1_783_814_401_500));
    let completed = event(
        "hook-completed",
        2,
        "hook.completed",
        "turn-1",
        json!({"item": serde_json::to_value(completed_item).expect("hook completed")}),
    );

    let changes = materialize_events(&[started, completed], "session-1", "thread-1")
        .expect("materialize hook lifecycle");
    let item = changes
        .changed_items
        .iter()
        .find(|item| matches!(item.payload, ThreadItemPayload::Hook { .. }))
        .expect("restored hook item");
    assert_eq!(item.item_id.as_str(), "item_hook-run-restore");
    assert_eq!(item.status, ItemStatus::Completed);
    assert_eq!(item.completed_at_ms, Some(1_783_814_402_000));
}
