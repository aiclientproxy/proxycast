use super::*;
use agent_protocol::{
    AgentInput, CollabAgentOperation, ItemStatus, SessionId, ThreadId, ThreadItemPayload,
    ToolOutput, TurnId,
};

fn completed_item(payload: ThreadItemPayload) -> ThreadItem {
    let mut item = ThreadItem::new(
        SessionId::new("session-1"),
        ThreadId::new("thread-1"),
        TurnId::new("turn-1"),
        1,
        1,
        payload,
    );
    item.status = ItemStatus::Completed;
    item
}

#[test]
fn fork_rejects_canonical_history_that_cannot_be_lowered_without_loss() {
    let cases = [
        (
            completed_item(ThreadItemPayload::ContextCompaction {
                summary: Some("summary without replacement history".to_string()),
                replacement_history: Vec::new(),
                window_number: None,
                first_window_id: None,
                previous_window_id: None,
                window_id: None,
                tail_start_turn_id: None,
            }),
            "compacted provider history",
        ),
        (
            completed_item(ThreadItemPayload::CollabAgentToolCall {
                call_id: "collab-1".to_string(),
                operation: CollabAgentOperation::Spawn,
                target_thread_id: Some(ThreadId::new("thread-child")),
                message: Some("spawn".to_string()),
                output: Some(ToolOutput {
                    text: Some("spawned".to_string()),
                    ..ToolOutput::default()
                }),
            }),
            "collab tool arguments",
        ),
        (
            completed_item(ThreadItemPayload::Media {
                uri: "sidecar://media/image".to_string(),
                mime_type: "image/png".to_string(),
                preview: None,
            }),
            "media content",
        ),
        (
            completed_item(ThreadItemPayload::Tool {
                call_id: "tool-1".to_string(),
                name: "read_file".to_string(),
                arguments: Vec::new(),
                output: None,
            }),
            "without a canonical result",
        ),
    ];

    for (item, expected) in cases {
        let error = validate_fork_canonical_item(&item).expect_err("fork must fail closed");
        let RuntimeCoreError::InvalidRequest(message) = error else {
            panic!("unexpected fork validation error: {error}");
        };
        assert!(message.contains(expected), "unexpected error: {message}");
    }
}

#[test]
fn fork_accepts_complete_compaction_lineage_and_rejects_incomplete_variants() {
    let first_window_id = uuid::Uuid::now_v7().to_string();
    let window_id = uuid::Uuid::now_v7().to_string();
    let replacement_history = vec![serde_json::json!({
        "role": "user",
        "content": [{"type": "input_text", "text": "compacted prefix"}],
    })];
    let complete = || ThreadItemPayload::ContextCompaction {
        summary: Some("summary".to_string()),
        replacement_history: replacement_history.clone(),
        window_number: Some(1),
        first_window_id: Some(first_window_id.clone()),
        previous_window_id: None,
        window_id: Some(window_id.clone()),
        tail_start_turn_id: Some("turn-1".to_string()),
    };

    validate_fork_canonical_item(&completed_item(complete()))
        .expect("complete compaction lineage is forkable");

    let mut incomplete = Vec::new();
    let mut missing_replacement = complete();
    let ThreadItemPayload::ContextCompaction {
        replacement_history,
        ..
    } = &mut missing_replacement
    else {
        unreachable!();
    };
    replacement_history.clear();
    incomplete.push(missing_replacement);

    let mut missing_window = complete();
    let ThreadItemPayload::ContextCompaction { window_id, .. } = &mut missing_window else {
        unreachable!();
    };
    *window_id = None;
    incomplete.push(missing_window);

    let mut missing_tail = complete();
    let ThreadItemPayload::ContextCompaction {
        tail_start_turn_id, ..
    } = &mut missing_tail
    else {
        unreachable!();
    };
    *tail_start_turn_id = None;
    incomplete.push(missing_tail);

    for payload in incomplete {
        let error = validate_fork_canonical_item(&completed_item(payload))
            .expect_err("incomplete compaction lineage must fail closed");
        let RuntimeCoreError::InvalidRequest(message) = error else {
            panic!("unexpected fork validation error: {error}");
        };
        assert!(
            message.contains("complete canonical lineage"),
            "unexpected error: {message}"
        );
    }
}

#[test]
fn fork_accepts_image_input_preserved_by_canonical_user_message() {
    for media in [
        AgentInput::Image {
            uri: "https://example.invalid/image.png".to_string(),
            detail: None,
        },
        AgentInput::LocalImage {
            path: "/tmp/image.png".to_string(),
            detail: None,
        },
    ] {
        let item = completed_item(ThreadItemPayload::UserMessage {
            content: vec![AgentInput::text("inspect"), media],
            client_id: Some("client-1".to_string()),
        });
        let history = ForkHistory {
            turn_ids: HashSet::from(["turn-1".to_string()]),
            changes: Some(ThreadHistoryChangeSet {
                sequence: 1,
                changed_items: vec![item],
                ..Default::default()
            }),
            interrupted_turn_id: None,
        };

        validate_fork_provider_history(&history).expect("canonical image input is forkable");
    }
}

#[test]
fn fork_history_interrupts_only_the_default_trailing_turn() {
    let source = thread_with_turns(&[
        ("turn-complete", TurnStatus::Completed),
        ("turn-active", TurnStatus::InProgress),
    ]);
    let target = fork_history(&source, &fork_params(), "target-session", "target-thread")
        .expect("default fork must snapshot the trailing active turn");
    let changes = target.changes.expect("fork history changes");
    assert_eq!(target.interrupted_turn_id.as_deref(), Some("turn-active"));
    assert_eq!(changes.changed_turns[0].status, TurnStatus::Completed);
    assert_eq!(changes.changed_turns[1].status, TurnStatus::Interrupted);
    assert_eq!(changes.changed_turns[1].completed_at_ms, None);
    assert_eq!(changes.changed_turns[1].duration_ms, None);

    let mut last_turn = fork_params();
    last_turn.last_turn_id = Some("turn-active".to_string());
    let error = match fork_history(&source, &last_turn, "target-session", "target-thread") {
        Ok(_) => panic!("lastTurnId must not include an active turn"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("in-progress turn"));

    let mut before_turn = fork_params();
    before_turn.before_turn_id = Some("turn-active".to_string());
    let target = fork_history(&source, &before_turn, "target-session", "target-thread")
        .expect("beforeTurnId may cut before an active turn");
    let changes = target.changes.expect("fork history changes");
    assert_eq!(target.interrupted_turn_id, None);
    assert_eq!(changes.changed_turns.len(), 1);
    assert_eq!(changes.changed_turns[0].turn_id.as_str(), "turn-complete");
}

fn fork_params() -> ThreadForkParams {
    serde_json::from_value(serde_json::json!({
        "threadId": "source-thread"
    }))
    .expect("fork params")
}

fn thread_with_turns(turns: &[(&str, TurnStatus)]) -> Thread {
    let now = 1_700_000_000_000;
    let mut thread = Thread {
        session_id: SessionId::new("source-session"),
        thread_id: ThreadId::new("source-thread"),
        status: ThreadStatus::Idle,
        created_at_ms: now,
        updated_at_ms: now,
        archived: false,
        recency_at_ms: None,
        parent_thread_id: None,
        agent_path: None,
        agent_nickname: None,
        agent_role: None,
        last_task_message: None,
        agent_state: None,
        forked_from_id: None,
        preview: String::new(),
        model_provider: "fixture-provider".to_string(),
        product: None,
        name: None,
        metadata: serde_json::json!({}),
        turns: Vec::new(),
        turns_view: ThreadTurnsView::Full,
    };
    thread.turns = turns
        .iter()
        .enumerate()
        .map(|(index, (turn_id, status))| Turn {
            session_id: SessionId::new("source-session"),
            thread_id: ThreadId::new("source-thread"),
            turn_id: TurnId::new(*turn_id),
            status: *status,
            admission: Default::default(),
            queue: if *status == TurnStatus::InProgress {
                TurnQueueState::Running
            } else {
                TurnQueueState::NotQueued
            },
            approval: Default::default(),
            items: vec![completed_item(ThreadItemPayload::UserMessage {
                content: vec![AgentInput::text(format!("input {index}"))],
                client_id: None,
            })],
            items_view: Default::default(),
            error: None,
            created_at_ms: now + index as i64,
            updated_at_ms: now + index as i64,
            started_at_ms: Some(now + index as i64),
            completed_at_ms: status.is_terminal().then_some(now + index as i64),
            duration_ms: status.is_terminal().then_some(1),
        })
        .collect();
    for turn in &mut thread.turns {
        for item in &mut turn.items {
            item.turn_id = turn.turn_id.clone();
        }
    }
    thread
}
