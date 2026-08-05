use super::*;
use agent_protocol::{
    AgentInput, CollabAgentOperation, CollabAgentState, CollabAgentStatus, ImageDetail, ItemKind,
    ItemStatus, MessageContentPart, MessageContentReference, ThreadItemPayload, TurnAdmissionState,
    TurnApprovalState, TurnItemsView, TurnQueueState, TurnStatus,
};
use std::collections::HashMap;

fn item(sequence: u64, ordinal: u64, id: &str, text: &str) -> ThreadItem {
    ThreadItem {
        session_id: SessionId::new("session-1"),
        thread_id: ThreadId::new("thread-1"),
        turn_id: TurnId::new("turn-1"),
        item_id: ItemId::new(id),
        sequence,
        ordinal,
        created_at_ms: sequence as i64,
        updated_at_ms: sequence as i64,
        completed_at_ms: None,
        kind: ItemKind::AgentMessage,
        status: ItemStatus::InProgress,
        payload: ThreadItemPayload::AgentMessage {
            text: text.to_string(),
            phase: None,
            content_parts: vec![MessageContentPart::Text {
                text: text.to_string(),
            }],
        },
        metadata: serde_json::Value::Null,
    }
}

fn reasoning_item(sequence: u64, content: &str, source_event_type: &str) -> ThreadItem {
    let mut item = ThreadItem::new(
        SessionId::new("session-1"),
        ThreadId::new("thread-1"),
        TurnId::new("turn-1"),
        sequence,
        1,
        ThreadItemPayload::Reasoning {
            summary: Vec::new(),
            content: vec![content.to_string()],
        },
    );
    item.item_id = ItemId::new("reasoning-1");
    item.metadata = serde_json::json!({"source_event_type": source_event_type});
    item
}

fn turn(id: &str, status: TurnStatus) -> Turn {
    Turn {
        session_id: SessionId::new("session-1"),
        thread_id: ThreadId::new("thread-1"),
        turn_id: TurnId::new(id),
        status,
        admission: TurnAdmissionState::Accepted,
        queue: TurnQueueState::Running,
        approval: TurnApprovalState::NotRequired,
        items: Vec::new(),
        items_view: TurnItemsView::NotLoaded,
        error: None,
        created_at_ms: 1,
        updated_at_ms: 1,
        started_at_ms: Some(1),
        completed_at_ms: None,
        duration_ms: None,
    }
}

fn media_item(sequence: u64, id: &str, uri: &str) -> ThreadItem {
    let mut item = item(sequence, sequence, id, "media");
    item.payload = ThreadItemPayload::AgentMessage {
        text: "media".to_string(),
        phase: None,
        content_parts: vec![MessageContentPart::Media {
            kind: "image".to_string(),
            reference: MessageContentReference {
                uri: uri.to_string(),
                mime_type: "image/png".to_string(),
                title: None,
                source_uri: None,
                source_path: None,
                preview_url: None,
                sidecar_ref: None,
                sha256: Some("abc123".to_string()),
                byte_size: Some(4),
            },
            caption: None,
        }],
    };
    item
}

#[test]
fn coalesces_repeated_item_snapshots_and_preserves_first_order() {
    let mut builder = ThreadHistoryBuilder::new();
    let first = item(1, 10, "item-a", "first");
    let second = item(2, 20, "item-b", "second");
    let mut update = first.clone();
    update.sequence = 3;
    update.ordinal = 99;
    if let ThreadItemPayload::AgentMessage { text, .. } = &mut update.payload {
        *text = "latest".to_string();
    }

    let changes = builder
        .append_items_at(1, vec![first.clone(), second.clone()])
        .expect("initial append");
    assert_eq!(changes.changed_items, vec![first.clone(), second]);
    let changes = builder
        .append_items_at(2, vec![update.clone()])
        .expect("snapshot update");
    assert_eq!(changes.changed_items.len(), 1);
    assert_eq!(builder.raw_items()[0].ordinal, 10);
    assert_eq!(builder.raw_items()[0].sequence, 3);
    assert!(matches!(
        &builder.raw_items()[0].payload,
        ThreadItemPayload::AgentMessage { text, .. } if text == "firstlatest"
    ));
}

#[test]
fn reasoning_history_preserves_repeated_deltas_and_replaces_final_snapshot() {
    let mut builder = ThreadHistoryBuilder::new();
    builder
        .append_turns_at(1, vec![turn("turn-1", TurnStatus::InProgress)])
        .expect("turn append");
    builder
        .append_items_at(2, vec![reasoning_item(2, ".", "reasoning.delta")])
        .expect("first reasoning delta");
    builder
        .append_items_at(3, vec![reasoning_item(3, ".", "reasoning.delta")])
        .expect("repeated reasoning delta");

    assert!(matches!(
        &builder.raw_items()[0].payload,
        ThreadItemPayload::Reasoning { content, .. }
            if content == &[".".to_string(), ".".to_string()]
    ));

    builder
        .append_items_at(4, vec![reasoning_item(4, "done", "reasoning.final")])
        .expect("reasoning final snapshot");
    assert!(matches!(
        &builder.raw_items()[0].payload,
        ThreadItemPayload::Reasoning { content, .. } if content == &["done".to_string()]
    ));

    let mut completed = reasoning_item(5, "", "item.completed");
    completed.status = ItemStatus::Completed;
    completed.completed_at_ms = Some(5);
    completed.payload = ThreadItemPayload::Reasoning {
        summary: Vec::new(),
        content: Vec::new(),
    };
    builder
        .append_items_at(5, vec![completed])
        .expect("empty reasoning completion");
    assert!(matches!(
        &builder.raw_items()[0].payload,
        ThreadItemPayload::Reasoning { content, .. } if content == &["done".to_string()]
    ));
    assert_eq!(builder.raw_items()[0].status, ItemStatus::Completed);
}

#[test]
fn item_first_append_attaches_items_when_turn_arrives() {
    let mut builder = ThreadHistoryBuilder::new();
    let first = item(1, 1, "item-a", "first");
    builder
        .append_items_at(1, vec![first.clone()])
        .expect("item-first append");
    assert!(builder.turns().is_empty());

    let changes = builder
        .append_turns_at(2, vec![turn("turn-1", TurnStatus::InProgress)])
        .expect("turn append");
    assert_eq!(changes.changed_turns[0].items, vec![first.clone()]);
    assert_eq!(builder.turns()[0].items, vec![first.clone()]);
    assert_eq!(builder.snapshot().turns[0].items, vec![first]);
}

#[test]
fn item_updates_replace_nested_turn_projection_without_duplicates() {
    let mut builder = ThreadHistoryBuilder::new();
    builder
        .append_turns_at(1, vec![turn("turn-1", TurnStatus::InProgress)])
        .expect("turn append");
    let first = item(2, 1, "item-a", "first");
    builder
        .append_items_at(2, vec![first])
        .expect("item append");

    let mut update = item(3, 99, "item-a", "second");
    update.completed_at_ms = Some(3);
    builder
        .append_items_at(3, vec![update])
        .expect("item update");

    let nested = &builder.turns()[0].items;
    assert_eq!(nested.len(), 1);
    assert_eq!(nested[0].sequence, 3);
    assert_eq!(nested[0].ordinal, 1);
    assert!(matches!(
        &nested[0].payload,
        ThreadItemPayload::AgentMessage { text, .. } if text == "firstsecond"
    ));
    assert_eq!(builder.snapshot().turns[0].items, nested.to_vec());
}

#[test]
fn collab_wait_completion_preserves_typed_agent_states_in_history() {
    let mut builder = ThreadHistoryBuilder::new();
    builder
        .append_turns_at(1, vec![turn("turn-1", TurnStatus::InProgress)])
        .expect("turn append");
    let mut started = ThreadItem::new(
        SessionId::new("session-1"),
        ThreadId::new("thread-1"),
        TurnId::new("turn-1"),
        2,
        1,
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
    builder
        .append_items_at(2, vec![started.clone()])
        .expect("wait start");

    let mut completed = started;
    completed.sequence = 3;
    completed.status = ItemStatus::Completed;
    completed.completed_at_ms = Some(3);
    let ThreadItemPayload::CollabAgentToolCall { agent_states, .. } = &mut completed.payload else {
        panic!("collab wait payload");
    };
    agent_states.insert(
        ThreadId::new("thread-child"),
        CollabAgentState {
            status: CollabAgentStatus::Completed,
            message: None,
        },
    );
    builder
        .append_items_at(3, vec![completed])
        .expect("wait completion");

    let ThreadItemPayload::CollabAgentToolCall { agent_states, .. } =
        &builder.turns()[0].items[0].payload
    else {
        panic!("merged collab wait payload");
    };
    assert_eq!(
        agent_states[&ThreadId::new("thread-child")].status,
        CollabAgentStatus::Completed
    );
}

#[test]
fn exact_retry_is_idempotent_and_different_payload_collides() {
    let mut builder = ThreadHistoryBuilder::new();
    let first = item(1, 1, "item-a", "same");
    assert!(
        builder
            .append_items_at(1, vec![first.clone()])
            .unwrap()
            .changed_items
            .len()
            == 1
    );
    assert!(builder
        .append_items_at(1, vec![first])
        .expect("exact retry")
        .changed_items
        .is_empty());
    let different = item(1, 2, "item-b", "different");
    assert!(matches!(
        builder.append_items_at(1, vec![different]),
        Err(ThreadHistoryBuilderError::SequenceCollision { sequence: 1 })
    ));
}

#[test]
fn unsafe_media_batch_is_rejected_atomically() {
    let mut builder = ThreadHistoryBuilder::new();
    let safe = item(1, 1, "item-safe", "safe");
    let unsafe_media = media_item(1, "item-unsafe", "data:image/png;base64,AAAA");

    assert!(matches!(
        builder.append_items_at(1, vec![safe, unsafe_media]),
        Err(ThreadHistoryBuilderError::UnsafeItemContent { item_id })
            if item_id == ItemId::new("item-unsafe")
    ));
    assert!(builder.raw_items().is_empty());
    assert_eq!(builder.sequence(), None);
}

#[test]
fn unsafe_direct_media_payload_is_rejected_before_history_mutation() {
    let mut builder = ThreadHistoryBuilder::new();
    let mut unsafe_media = item(1, 1, "item-media", "media");
    unsafe_media.kind = ItemKind::Media;
    unsafe_media.payload = ThreadItemPayload::Media {
        uri: "sidecar://media/result".to_string(),
        mime_type: "image/png".to_string(),
        preview: Some(" DATA:image/png;base64,AAAA".to_string()),
    };

    assert!(matches!(
        builder.append_items_at(1, vec![unsafe_media]),
        Err(ThreadHistoryBuilderError::UnsafeItemContent { .. })
    ));
    assert!(builder.raw_items().is_empty());
}

#[test]
fn safe_media_retry_and_snapshot_rebuild_are_stable() {
    let mut builder = ThreadHistoryBuilder::new();
    builder
        .append_turns_at(1, vec![turn("turn-1", TurnStatus::InProgress)])
        .expect("canonical turn");
    let media = media_item(2, "item-media", "sidecar://media/result");
    builder
        .append_items_at(2, vec![media.clone()])
        .expect("safe media append");

    assert!(builder
        .append_items_at(2, vec![media])
        .expect("exact safe media retry")
        .changed_items
        .is_empty());
    let snapshot = builder.snapshot();
    let rebuilt =
        ThreadHistoryBuilder::from_snapshot(snapshot.clone()).expect("safe media snapshot rebuild");
    let rebuilt_snapshot = rebuilt.snapshot();
    assert_eq!(rebuilt_snapshot.sequence, snapshot.sequence);
    assert_eq!(rebuilt_snapshot.turn_sequences, snapshot.turn_sequences);
    assert_eq!(rebuilt_snapshot.items, snapshot.items);
    assert_eq!(rebuilt_snapshot.turns[0].items, snapshot.items);
    assert_eq!(
        ThreadHistoryBuilder::from_snapshot(rebuilt_snapshot.clone())
            .expect("normalized media snapshot rebuild")
            .snapshot(),
        rebuilt_snapshot
    );
}

#[test]
fn multimodal_user_message_snapshot_rebuild_is_exact() {
    let mut builder = ThreadHistoryBuilder::new();
    builder
        .append_turns_at(1, vec![turn("turn-1", TurnStatus::Completed)])
        .expect("canonical turn");
    let mut user = item(2, 1, "item-user", "unused");
    user.kind = ItemKind::UserMessage;
    user.status = ItemStatus::Completed;
    user.payload = ThreadItemPayload::UserMessage {
        content: vec![
            AgentInput::text("inspect"),
            AgentInput::Image {
                uri: "https://example.com/remote.png".to_string(),
                detail: Some(ImageDetail::High),
            },
            AgentInput::LocalImage {
                path: "/tmp/local.png".to_string(),
                detail: Some(ImageDetail::Original),
            },
        ],
        client_id: Some("client-1".to_string()),
    };
    builder
        .append_items_at(2, vec![user.clone()])
        .expect("multimodal user item");

    let snapshot = builder.snapshot();
    let rebuilt = ThreadHistoryBuilder::from_snapshot(snapshot).expect("rebuild canonical history");
    assert_eq!(rebuilt.raw_items(), &[user]);
}

#[test]
fn unsafe_media_snapshot_rebuild_fails_closed() {
    let unsafe_media = media_item(2, "item-media", "data:image/png;base64,AAAA");
    let snapshot = CanonicalHistory {
        session_id: Some(SessionId::new("session-1")),
        thread_id: Some(ThreadId::new("thread-1")),
        sequence: Some(2),
        turns: vec![turn("turn-1", TurnStatus::Completed)],
        turn_sequences: [(TurnId::new("turn-1"), 2)].into_iter().collect(),
        items: vec![unsafe_media],
    };

    assert!(matches!(
        ThreadHistoryBuilder::from_snapshot(snapshot),
        Err(ThreadHistoryBuilderError::UnsafeItemContent { .. })
    ));
}

#[test]
fn rollback_removes_raw_tail_and_reports_removed_identity() {
    let mut builder = ThreadHistoryBuilder::new();
    let mut first = item(1, 1, "item-a", "first");
    first.turn_id = TurnId::new("turn-1");
    let mut second = item(2, 2, "item-b", "second");
    second.turn_id = TurnId::new("turn-2");
    let mut first_turn = turn("turn-1", TurnStatus::InProgress);
    first_turn.items = vec![first.clone()];
    builder
        .apply_change_set(ThreadHistoryChangeSet {
            sequence: 1,
            changed_turns: vec![first_turn],
            changed_items: vec![first.clone()],
            ..Default::default()
        })
        .unwrap();
    let mut second_turn = turn("turn-2", TurnStatus::Completed);
    second_turn.updated_at_ms = 2;
    builder
        .apply_change_set(ThreadHistoryChangeSet {
            sequence: 2,
            changed_turns: vec![second_turn],
            changed_items: vec![second.clone()],
            ..Default::default()
        })
        .unwrap();
    let changes = builder.rollback_at(3, 1).expect("rollback");
    assert_eq!(changes.rollback_to_sequence, Some(1));
    assert_eq!(changes.removed_item_ids, vec![second.item_id]);
    assert_eq!(changes.removed_turn_ids, vec![TurnId::new("turn-2")]);
    assert_eq!(builder.raw_items(), &[first]);
    assert_eq!(builder.turns()[0].items.len(), 1);
    assert_eq!(builder.sequence(), Some(3));
}

#[test]
fn active_snapshot_and_pages_are_stable() {
    let mut builder = ThreadHistoryBuilder::new();
    let mut open = turn("turn-1", TurnStatus::InProgress);
    open.updated_at_ms = 1;
    builder.append_turns_at(1, vec![open.clone()]).unwrap();
    assert_eq!(builder.active_turn_snapshot(), Some(open));
    builder
        .append_items_at(2, vec![item(2, 1, "item-a", "a")])
        .unwrap();
    let page = builder.page_items(0, 1, SortDirection::Asc);
    assert_eq!(page.data.len(), 1);
    assert_eq!(page.next_offset, None);
    assert_eq!(
        builder.snapshot().thread_id,
        Some(ThreadId::new("thread-1"))
    );
}

#[test]
fn identity_validation_is_atomic() {
    let mut builder =
        ThreadHistoryBuilder::for_thread(SessionId::new("session-1"), ThreadId::new("thread-1"));
    let mut wrong = item(1, 1, "item-a", "wrong");
    wrong.session_id = SessionId::new("session-2");
    assert!(matches!(
        builder.append_items_at(1, vec![wrong]),
        Err(ThreadHistoryBuilderError::SessionIdentityMismatch { .. })
    ));
    assert!(builder.raw_items().is_empty());
    assert_eq!(builder.sequence(), None);
}

#[test]
fn turn_updates_preserve_first_created_time_and_active_state() {
    let mut builder = ThreadHistoryBuilder::new();
    let open = turn("turn-1", TurnStatus::InProgress);
    builder.append_turns_at(1, vec![open.clone()]).unwrap();
    let mut done = open.clone();
    done.status = TurnStatus::Completed;
    done.created_at_ms = 99;
    done.completed_at_ms = Some(100);
    let changes = builder.append_turns_at(2, vec![done.clone()]).unwrap();
    assert_eq!(changes.changed_turns.len(), 1);
    assert_eq!(builder.turns()[0].created_at_ms, open.created_at_ms);
    let mut expected = done;
    expected.created_at_ms = open.created_at_ms;
    assert_eq!(builder.active_turn_snapshot(), Some(expected));
}

#[test]
fn page_descending_uses_reverse_stable_order() {
    let mut builder = ThreadHistoryBuilder::new();
    let items = (1..=3)
        .map(|sequence| item(sequence, sequence, &format!("item-{sequence}"), "x"))
        .collect::<Vec<_>>();
    builder.append_items_at(3, items).unwrap();
    let page = builder.page_items(0, 2, SortDirection::Desc);
    assert_eq!(
        page.data
            .iter()
            .map(|item| item.sequence)
            .collect::<Vec<_>>(),
        vec![3, 2]
    );
    assert_eq!(page.next_offset, Some(2));
}

#[test]
fn apply_change_set_coalesces_typed_payload_and_is_atomic_on_conflict() {
    let mut builder =
        ThreadHistoryBuilder::for_thread(SessionId::new("session-1"), ThreadId::new("thread-1"));
    let open = turn("turn-1", TurnStatus::InProgress);
    let first = item(1, 1, "item-a", "hello");
    builder
        .apply_change_set(ThreadHistoryChangeSet {
            sequence: 1,
            changed_turns: vec![open],
            changed_items: vec![first],
            ..Default::default()
        })
        .expect("initial canonical batch");

    let mut delta = item(2, 99, "item-a", " world");
    delta.completed_at_ms = Some(2);
    let normalized = builder
        .apply_change_set(ThreadHistoryChangeSet {
            sequence: 2,
            changed_items: vec![delta],
            ..Default::default()
        })
        .expect("typed delta");
    assert_eq!(normalized.changed_items.len(), 1);
    assert_eq!(builder.raw_items()[0].ordinal, 1);
    assert_eq!(builder.raw_items()[0].sequence, 2);
    assert!(matches!(
        &builder.raw_items()[0].payload,
        ThreadItemPayload::AgentMessage { text, .. } if text == "hello world"
    ));

    let before = builder.snapshot();
    let conflict = item(2, 2, "item-b", "conflict");
    assert!(matches!(
        builder.apply_change_set(ThreadHistoryChangeSet {
            sequence: 2,
            changed_items: vec![conflict],
            ..Default::default()
        }),
        Err(ThreadHistoryBuilderError::SequenceCollision { sequence: 2 })
    ));
    assert_eq!(builder.snapshot(), before);
}

#[test]
fn snapshot_rebuild_rejects_orphan_items_and_preserves_turn_sequences() {
    let open = turn("turn-1", TurnStatus::InProgress);
    let stored_item = item(4, 1, "item-a", "hello");
    let snapshot = CanonicalHistory {
        session_id: Some(SessionId::new("session-1")),
        thread_id: Some(ThreadId::new("thread-1")),
        sequence: Some(4),
        turns: vec![open],
        turn_sequences: [(TurnId::new("turn-1"), 3)].into_iter().collect(),
        items: vec![stored_item.clone()],
    };
    let builder = ThreadHistoryBuilder::from_snapshot(snapshot).expect("snapshot");
    assert_eq!(builder.snapshot().turn_sequences[&TurnId::new("turn-1")], 3);
    assert_eq!(builder.raw_items(), &[stored_item]);

    let orphan = CanonicalHistory {
        session_id: Some(SessionId::new("session-1")),
        thread_id: Some(ThreadId::new("thread-1")),
        sequence: Some(1),
        turns: Vec::new(),
        turn_sequences: Default::default(),
        items: vec![item(1, 1, "orphan", "orphan")],
    };
    assert!(matches!(
        ThreadHistoryBuilder::from_snapshot(orphan),
        Err(ThreadHistoryBuilderError::MissingTurn { .. })
    ));
}
