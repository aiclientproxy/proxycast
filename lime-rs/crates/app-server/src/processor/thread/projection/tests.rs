use super::*;
use serde_json::json;

#[test]
fn v2_read_params_choose_exact_turn_loading_mode() {
    let without_turns = lower_thread_read_params(&v2::ThreadReadParams {
        thread_id: "thread-1".to_string(),
        include_turns: false,
    })
    .expect("lower read without turns");
    assert_eq!(
        without_turns.turns_view,
        canonical::ThreadTurnsView::NotLoaded
    );

    let with_turns = lower_thread_read_params(&v2::ThreadReadParams {
        thread_id: "thread-1".to_string(),
        include_turns: true,
    })
    .expect("lower read with turns");
    assert_eq!(with_turns.turns_view, canonical::ThreadTurnsView::Full);
}

#[test]
fn canonical_thread_projects_to_v2_shape_and_seconds() {
    let thread = canonical_thread(false);
    let projected = project_thread(thread).expect("project thread");

    assert_eq!(projected.id, "thread-1");
    assert_eq!(projected.session_id, "session-1");
    assert_eq!(projected.created_at, 1_700_000_000);
    assert_eq!(projected.updated_at, 1_700_000_002);
    assert_eq!(projected.cwd, "/workspace");
    assert_eq!(projected.can_accept_direct_input, Some(true));
    assert_eq!(projected.history_mode, v2::ThreadHistoryMode::Paginated);
    assert_eq!(projected.turns.len(), 1);
    assert!(matches!(
        projected.turns[0].items[0],
        v2::ThreadItem::UserMessage { .. }
    ));
}

#[test]
fn spawned_child_thread_projects_as_parent_owned() {
    let mut thread = canonical_thread(false);
    thread.parent_thread_id = Some(canonical::ThreadId::new("thread-parent"));

    let projected = project_thread(thread).expect("project parent-owned thread");

    assert_eq!(projected.parent_thread_id.as_deref(), Some("thread-parent"));
    assert_eq!(projected.can_accept_direct_input, Some(false));
}

#[test]
fn canonical_user_message_projects_ordered_parts_without_flattening() {
    let mut thread = canonical_thread(false);
    let content = vec![
        canonical::AgentInput::Text {
            text: "inspect".to_string(),
            text_elements: vec![canonical::TextElement::new(0..7, None)],
        },
        canonical::AgentInput::Image {
            uri: "https://example.com/remote.png".to_string(),
            detail: Some(canonical::ImageDetail::High),
        },
        canonical::AgentInput::LocalImage {
            path: "/tmp/local.png".to_string(),
            detail: Some(canonical::ImageDetail::Original),
        },
        canonical::AgentInput::Skill {
            name: "review".to_string(),
            path: "/skills/review/SKILL.md".to_string(),
        },
        canonical::AgentInput::Mention {
            name: "docs".to_string(),
            path: "app://docs".to_string(),
        },
    ];
    thread.turns[0].items[0].payload = canonical::ThreadItemPayload::UserMessage {
        content,
        client_id: Some("client-1".to_string()),
    };

    let projected = project_thread(thread).expect("project multimodal user message");
    let v2::ThreadItem::UserMessage {
        client_id, content, ..
    } = &projected.turns[0].items[0]
    else {
        panic!("user message item");
    };
    assert_eq!(client_id.as_deref(), Some("client-1"));
    assert!(matches!(
        &content[..],
        [
            v2::UserInput::Text { text, text_elements },
            v2::UserInput::Image { url, detail: Some(canonical::ImageDetail::High) },
            v2::UserInput::LocalImage { path, detail: Some(canonical::ImageDetail::Original) },
            v2::UserInput::Skill { name, .. },
            v2::UserInput::Mention { path: mention_path, .. },
        ] if text == "inspect"
            && text_elements.len() == 1
            && url == "https://example.com/remote.png"
            && path == "/tmp/local.png"
            && name == "review"
            && mention_path == "app://docs"
    ));
}

#[test]
fn user_shell_command_metadata_projects_to_the_v2_item() {
    let mut thread = canonical_thread(false);
    thread.turns[0].items[0] = canonical::ThreadItem {
        session_id: canonical::SessionId::new("session-1"),
        thread_id: canonical::ThreadId::new("thread-1"),
        turn_id: canonical::TurnId::new("turn-1"),
        item_id: canonical::ItemId::new("shell-1"),
        sequence: 1,
        ordinal: 1,
        created_at_ms: 1_700_000_000_500,
        updated_at_ms: 1_700_000_001_000,
        completed_at_ms: Some(1_700_000_001_000),
        kind: canonical::ItemKind::Command,
        status: canonical::ItemStatus::Completed,
        payload: canonical::ThreadItemPayload::Command {
            command: "printf ready".to_string(),
            cwd: Some("/workspace".to_string()),
            output: Some("ready".to_string()),
            exit_code: Some(0),
        },
        metadata: json!({
            "commandExecutionSource": "userShell",
            "processId": "process-1",
            "durationMs": 42
        }),
    };

    let projected = project_thread(thread).expect("project user shell command");
    let v2::ThreadItem::CommandExecution {
        source,
        process_id,
        duration_ms,
        ..
    } = &projected.turns[0].items[0]
    else {
        panic!("command item projection");
    };
    assert_eq!(*source, v2::CommandExecutionSource::UserShell);
    assert_eq!(process_id.as_deref(), Some("process-1"));
    assert_eq!(*duration_ms, Some(42));
}

#[test]
fn canonical_mcp_output_projects_codex_result_shape_when_only_truncation_remains() {
    let mut thread = canonical_thread(false);
    thread.turns[0].items[0] = canonical::ThreadItem {
        session_id: canonical::SessionId::new("session-1"),
        thread_id: canonical::ThreadId::new("thread-1"),
        turn_id: canonical::TurnId::new("turn-1"),
        item_id: canonical::ItemId::new("mcp-1"),
        sequence: 1,
        ordinal: 1,
        created_at_ms: 1_700_000_000_500,
        updated_at_ms: 1_700_000_001_000,
        completed_at_ms: Some(1_700_000_001_000),
        kind: canonical::ItemKind::McpToolCall,
        status: canonical::ItemStatus::Failed,
        payload: canonical::ThreadItemPayload::McpToolCall {
            call_id: "mcp-1".to_string(),
            server_name: "node_repl".to_string(),
            tool_name: "exec".to_string(),
            arguments: Vec::new(),
            output: Some(canonical::ToolOutput {
                error: Some("tool output unavailable".to_string()),
                truncated: true,
                output_ref: Some("output-1".to_string()),
                ..Default::default()
            }),
        },
        metadata: json!({}),
    };

    let projected = project_thread(thread).expect("project MCP result");
    let v2::ThreadItem::McpToolCall {
        result,
        error,
        status,
        ..
    } = &projected.turns[0].items[0]
    else {
        panic!("MCP item");
    };
    assert_eq!(*status, v2::McpToolCallStatus::Failed);
    assert_eq!(
        result,
        &Some(json!({
            "content": [],
            "_meta": {
                "truncated": true,
                "outputRef": "output-1"
            }
        }))
    );
    assert_eq!(
        error,
        &Some(json!({ "message": "tool output unavailable" }))
    );
}

#[test]
fn file_change_projects_complete_batch_and_move_identity() {
    let mut thread = canonical_thread(false);
    thread.turns[0].items[0] = canonical::ThreadItem {
        session_id: canonical::SessionId::new("session-1"),
        thread_id: canonical::ThreadId::new("thread-1"),
        turn_id: canonical::TurnId::new("turn-1"),
        item_id: canonical::ItemId::new("patch-1"),
        sequence: 1,
        ordinal: 1,
        created_at_ms: 1_700_000_000_500,
        updated_at_ms: 1_700_000_001_000,
        completed_at_ms: Some(1_700_000_001_000),
        kind: canonical::ItemKind::File,
        status: canonical::ItemStatus::Completed,
        payload: canonical::ThreadItemPayload::File {
            changes: vec![
                canonical::FileChange {
                    path: "new.txt".to_string(),
                    kind: canonical::FileChangeKind::Add,
                    diff: "+new".to_string(),
                },
                canonical::FileChange {
                    path: "dead.txt".to_string(),
                    kind: canonical::FileChangeKind::Delete,
                    diff: "-dead".to_string(),
                },
                canonical::FileChange {
                    path: "same.txt".to_string(),
                    kind: canonical::FileChangeKind::Update { move_path: None },
                    diff: "-old\n+new".to_string(),
                },
                canonical::FileChange {
                    path: "source.txt".to_string(),
                    kind: canonical::FileChangeKind::Update {
                        move_path: Some("target.txt".to_string()),
                    },
                    diff: "-before\n+after".to_string(),
                },
            ],
            status: canonical::FileChangeStatus::Rejected,
        },
        metadata: json!({}),
    };

    let projected = project_thread(thread).expect("project file change");
    let v2::ThreadItem::FileChange {
        changes, status, ..
    } = &projected.turns[0].items[0]
    else {
        panic!("file change projection");
    };
    assert_eq!(*status, v2::PatchApplyStatus::Declined);
    assert_eq!(changes.len(), 4);
    assert_eq!(changes[0].kind, v2::PatchChangeKind::Add);
    assert_eq!(changes[1].kind, v2::PatchChangeKind::Delete);
    assert_eq!(
        changes[3].kind,
        v2::PatchChangeKind::Update {
            move_path: Some("target.txt".to_string())
        }
    );
    assert_eq!(changes[3].path, "source.txt");
}

#[test]
fn v2_archived_list_filter_is_exact_even_when_store_page_contains_both() {
    let response = canonical::ThreadListResponse {
        data: vec![canonical_thread(false), canonical_thread(true)],
        next_cursor: Some("next".to_string()),
        backwards_cursor: None,
    };
    let params = v2::ThreadListParams {
        archived: Some(true),
        ..Default::default()
    };

    let projected = project_thread_list_response(response, &params).expect("project list");
    assert_eq!(projected.data.len(), 1);
    assert_eq!(projected.data[0].id, "thread-1");
    assert_eq!(projected.next_cursor.as_deref(), Some("next"));
}

#[test]
fn unsupported_canonical_item_fails_closed() {
    let mut thread = canonical_thread(false);
    thread.turns[0].items[0].payload = canonical::ThreadItemPayload::Extension {
        name: "unknown".to_string(),
        data: json!({"raw": true}),
    };

    let error = project_thread(thread).expect_err("reject extension item");
    assert_eq!(error.code, error_codes::RUNTIME_ERROR);
    assert!(error.message.contains("no v2 ThreadItem representation"));
}

#[test]
fn approval_control_items_stay_out_of_codex_v2_thread_items() {
    let mut thread = canonical_thread(false);
    let approval = canonical::ThreadItem {
        session_id: canonical::SessionId::new("session-1"),
        thread_id: canonical::ThreadId::new("thread-1"),
        turn_id: canonical::TurnId::new("turn-1"),
        item_id: canonical::ItemId::new("approval-1"),
        sequence: 2,
        ordinal: 2,
        created_at_ms: 1_700_000_001_000,
        updated_at_ms: 1_700_000_001_000,
        completed_at_ms: None,
        kind: canonical::ItemKind::Approval,
        status: canonical::ItemStatus::Pending,
        payload: canonical::ThreadItemPayload::Approval {
            request_id: "approval-1".to_string(),
            action: canonical::ApprovalAction {
                kind: "tool_confirmation".to_string(),
                description: "Allow command?".to_string(),
            },
            scope: canonical::ApprovalScope::Once,
            available_decisions: vec![canonical::ApprovalDecision::Abort],
            decision: None,
            requested_at_ms: Some(1_700_000_001_000),
            resolved_at_ms: None,
            reason_code: None,
            expires_at_ms: None,
        },
        metadata: json!({}),
    };
    thread.turns[0].items.push(approval.clone());

    let projected = project_thread(thread).expect("project thread without approval item");
    assert_eq!(projected.turns[0].items.len(), 1);
    assert!(matches!(
        projected.turns[0].items[0],
        v2::ThreadItem::UserMessage { .. }
    ));

    let projected = project_thread_items_list_response(canonical::ThreadItemsListResponse {
        data: vec![approval],
        next_cursor: None,
        backwards_cursor: None,
    })
    .expect("project item page without approval item");
    assert!(projected.data.is_empty());
}

fn canonical_thread(archived: bool) -> canonical::Thread {
    canonical::Thread {
        session_id: canonical::SessionId::new("session-1"),
        thread_id: canonical::ThreadId::new("thread-1"),
        status: canonical::ThreadStatus::Idle,
        created_at_ms: 1_700_000_000_123,
        updated_at_ms: 1_700_000_002_456,
        archived,
        recency_at_ms: Some(1_700_000_002_456),
        parent_thread_id: None,
        agent_path: None,
        agent_nickname: None,
        agent_role: None,
        last_task_message: None,
        agent_state: None,
        forked_from_id: None,
        preview: "hello".to_string(),
        model_provider: "openai".to_string(),
        product: None,
        name: Some("Thread".to_string()),
        metadata: json!({
            "workingDir": "/workspace",
            "historyMode": "paginated",
            "source": "appServer",
            "cliVersion": "test"
        }),
        turns: vec![canonical::Turn {
            session_id: canonical::SessionId::new("session-1"),
            thread_id: canonical::ThreadId::new("thread-1"),
            turn_id: canonical::TurnId::new("turn-1"),
            status: canonical::TurnStatus::Completed,
            admission: canonical::TurnAdmissionState::Accepted,
            queue: canonical::TurnQueueState::Running,
            approval: canonical::TurnApprovalState::NotRequired,
            items: vec![canonical::ThreadItem {
                session_id: canonical::SessionId::new("session-1"),
                thread_id: canonical::ThreadId::new("thread-1"),
                turn_id: canonical::TurnId::new("turn-1"),
                item_id: canonical::ItemId::new("message-1"),
                sequence: 1,
                ordinal: 1,
                created_at_ms: 1_700_000_000_500,
                updated_at_ms: 1_700_000_001_000,
                completed_at_ms: Some(1_700_000_001_000),
                kind: canonical::ItemKind::UserMessage,
                status: canonical::ItemStatus::Completed,
                payload: canonical::ThreadItemPayload::UserMessage {
                    content: vec![canonical::AgentInput::text("hello")],
                    client_id: Some("client-1".to_string()),
                },
                metadata: json!({}),
            }],
            items_view: canonical::TurnItemsView::Full,
            error: None,
            created_at_ms: 1_700_000_000_500,
            updated_at_ms: 1_700_000_001_000,
            started_at_ms: Some(1_700_000_000_500),
            completed_at_ms: Some(1_700_000_001_000),
            duration_ms: Some(500),
        }],
        turns_view: canonical::ThreadTurnsView::Full,
    }
}
