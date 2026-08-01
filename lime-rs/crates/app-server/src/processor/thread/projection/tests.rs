use super::*;
use serde_json::json;
use std::collections::HashMap;

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
    assert_eq!(result, &None);
    assert_eq!(
        error,
        &Some(v2::McpToolCallError {
            message: "tool output unavailable".to_string()
        })
    );
}

#[test]
fn canonical_mcp_output_is_size_bounded_and_redacts_sensitive_fields() {
    let mut thread = canonical_thread(false);
    thread.turns[0].items[0] = canonical::ThreadItem {
        session_id: canonical::SessionId::new("session-1"),
        thread_id: canonical::ThreadId::new("thread-1"),
        turn_id: canonical::TurnId::new("turn-1"),
        item_id: canonical::ItemId::new("mcp-safe"),
        sequence: 1,
        ordinal: 1,
        created_at_ms: 1_700_000_000_500,
        updated_at_ms: 1_700_000_001_000,
        completed_at_ms: Some(1_700_000_001_000),
        kind: canonical::ItemKind::McpToolCall,
        status: canonical::ItemStatus::Completed,
        payload: canonical::ThreadItemPayload::McpToolCall {
            call_id: "mcp-safe".to_string(),
            server_name: "docs".to_string(),
            tool_name: "search".to_string(),
            arguments: vec![canonical::ToolArgument {
                name: "token".to_string(),
                value: "secret-value".to_string(),
            }],
            output: Some(canonical::ToolOutput {
                text: Some("x".repeat(MAX_DISPLAY_JSON_BYTES * 2)),
                structured_content: Some(json!({
                    "password": "secret-value",
                    "safe": "visible"
                })),
                output_ref: Some("sidecar://private-output-id".to_string()),
                ..Default::default()
            }),
        },
        metadata: json!({}),
    };

    let projected = project_thread(thread).expect("project bounded MCP result");
    let v2::ThreadItem::McpToolCall {
        arguments, result, ..
    } = &projected.turns[0].items[0]
    else {
        panic!("MCP item");
    };
    let result = result.as_deref().expect("successful MCP result");
    let wire = serde_json::to_value(result).expect("MCP result wire");
    assert!(serde_json::to_vec(result).unwrap().len() <= MAX_DISPLAY_JSON_BYTES);
    assert_eq!(arguments[0]["value"], "[redacted]");
    assert_eq!(wire["structuredContent"]["password"], "[redacted]");
    assert_eq!(wire["structuredContent"]["safe"], "visible");
    assert_eq!(wire["_meta"]["truncated"], true);
    assert_eq!(wire["_meta"]["outputAvailable"], true);
    assert!(!wire.to_string().contains("private-output-id"));
    assert!(!wire.to_string().contains("secret-value"));
}

#[test]
fn canonical_dynamic_tool_output_is_tagged_bounded_and_redacted() {
    let mut thread = canonical_thread(false);
    thread.turns[0].items[0].kind = canonical::ItemKind::DynamicToolCall;
    thread.turns[0].items[0].status = canonical::ItemStatus::Completed;
    thread.turns[0].items[0].payload = canonical::ThreadItemPayload::DynamicToolCall {
        call_id: "dynamic-safe".to_string(),
        namespace: Some("desktop".to_string()),
        tool: "lookup".to_string(),
        arguments: json!({"password": "argument-secret", "safe": "visible"}),
        content_items: vec![
            canonical::DynamicToolCallContentItem::InputText {
                text: "x".repeat(MAX_DISPLAY_JSON_BYTES * 2),
            },
            canonical::DynamicToolCallContentItem::InputText {
                text: json!({
                    "password": "secret-value",
                    "safe": "visible"
                })
                .to_string(),
            },
        ],
        success: Some(true),
        duration_ms: Some(11),
    };

    let projected = project_thread(thread).expect("project bounded dynamic output");
    let v2::ThreadItem::DynamicToolCall {
        namespace,
        tool,
        arguments,
        content_items: Some(content_items),
        success,
        duration_ms,
        ..
    } = &projected.turns[0].items[0]
    else {
        panic!("dynamic tool item");
    };
    assert_eq!(namespace.as_deref(), Some("desktop"));
    assert_eq!(tool, "lookup");
    assert_eq!(arguments["password"], "[redacted]");
    assert_eq!(arguments["safe"], "visible");
    assert_eq!(*success, Some(true));
    assert_eq!(*duration_ms, Some(11));
    let wire = serde_json::to_value(content_items).expect("dynamic output wire");
    assert_eq!(wire[0]["type"], "inputText");
    assert!(wire[0]["text"].as_str().unwrap().len() <= MAX_DISPLAY_STRING_BYTES);
    assert!(wire[0]["text"]
        .as_str()
        .unwrap()
        .ends_with("... [truncated]"));
    assert_eq!(wire[1]["type"], "inputText");
    assert_eq!(
        serde_json::from_str::<Value>(wire[1]["text"].as_str().unwrap()).unwrap(),
        json!({"password": "[redacted]", "safe": "visible"})
    );
    assert!(!wire.to_string().contains("secret-value"));
}

#[test]
fn canonical_wait_projects_typed_agent_states_without_raw_output() {
    let mut thread = canonical_thread(false);
    thread.turns[0].items[0].kind = canonical::ItemKind::CollabAgentToolCall;
    thread.turns[0].items[0].status = canonical::ItemStatus::Completed;
    thread.turns[0].items[0].payload = canonical::ThreadItemPayload::CollabAgentToolCall {
        call_id: "wait-typed-states".to_string(),
        operation: canonical::CollabAgentOperation::Wait,
        target_thread_id: None,
        message: None,
        output: Some(canonical::ToolOutput {
            text: Some("model-visible wait output".to_string()),
            ..Default::default()
        }),
        agent_states: HashMap::from([
            (
                canonical::ThreadId::new("child-completed"),
                canonical::CollabAgentState {
                    status: canonical::CollabAgentStatus::Completed,
                    message: None,
                },
            ),
            (
                canonical::ThreadId::new("child-failed"),
                canonical::CollabAgentState {
                    status: canonical::CollabAgentStatus::Errored,
                    message: Some("child failed".to_string()),
                },
            ),
        ]),
    };

    let projected = project_thread(thread).expect("project typed wait states");
    let v2::ThreadItem::CollabAgentToolCall { agents_states, .. } = &projected.turns[0].items[0]
    else {
        panic!("collab wait item");
    };
    assert_eq!(agents_states.len(), 2);
    assert_eq!(
        agents_states["child-completed"].status,
        v2::CollabAgentStatus::Completed
    );
    assert_eq!(
        agents_states["child-failed"],
        v2::CollabAgentState {
            status: v2::CollabAgentStatus::Errored,
            message: Some("child failed".to_string()),
        }
    );
    assert!(!serde_json::to_string(&projected)
        .expect("serialize projected thread")
        .contains("model-visible wait output"));
}

#[test]
fn provider_image_generation_projects_exact_codex_item() {
    let mut thread = canonical_thread(false);
    thread.turns[0].items[0] = canonical::ThreadItem {
        session_id: canonical::SessionId::new("session-1"),
        thread_id: canonical::ThreadId::new("thread-1"),
        turn_id: canonical::TurnId::new("turn-1"),
        item_id: canonical::ItemId::new("ig_1"),
        sequence: 1,
        ordinal: 1,
        created_at_ms: 1_700_000_000_500,
        updated_at_ms: 1_700_000_001_000,
        completed_at_ms: Some(1_700_000_001_000),
        kind: canonical::ItemKind::Tool,
        status: canonical::ItemStatus::Completed,
        payload: canonical::ThreadItemPayload::Tool {
            call_id: "ig_1".to_string(),
            name: "image_generation".to_string(),
            arguments: Vec::new(),
            output: None,
        },
        metadata: json!({
            "provider_metadata": {
                "raw_response_item": {
                    "id": "ig_1",
                    "type": "image_generation_call",
                    "status": "completed",
                    "revised_prompt": "a blue square",
                    "result": "Zm9v"
                }
            }
        }),
    };

    let projected = project_thread(thread).expect("project image generation");
    assert_eq!(
        projected.turns[0].items[0],
        v2::ThreadItem::ImageGeneration(v2::ImageGenerationItem {
            id: "ig_1".to_string(),
            status: "completed".to_string(),
            revised_prompt: Some("a blue square".to_string()),
            result: "Zm9v".to_string(),
            saved_path: None,
        })
    );
}

#[test]
fn interrupted_provider_image_generation_projects_failed_status() {
    let mut thread = canonical_thread(false);
    thread.turns[0].items[0].kind = canonical::ItemKind::Tool;
    thread.turns[0].items[0].status = canonical::ItemStatus::Interrupted;
    thread.turns[0].items[0].payload = canonical::ThreadItemPayload::Tool {
        call_id: "ig_interrupted".to_string(),
        name: "image_generation".to_string(),
        arguments: Vec::new(),
        output: None,
    };
    thread.turns[0].items[0].metadata = json!({
        "provider_metadata": {
            "raw_response_item": {
                "id": "ig_interrupted",
                "type": "image_generation_call"
            }
        }
    });

    let projected = project_thread(thread).expect("project interrupted image generation");
    let v2::ThreadItem::ImageGeneration(item) = &projected.turns[0].items[0] else {
        panic!("expected image generation item");
    };
    assert_eq!(item.status, "failed");
    assert!(item.result.is_empty());
}

#[test]
fn completed_provider_image_generation_without_result_fails_closed() {
    let mut thread = canonical_thread(false);
    thread.turns[0].items[0].kind = canonical::ItemKind::Tool;
    thread.turns[0].items[0].status = canonical::ItemStatus::Completed;
    thread.turns[0].items[0].payload = canonical::ThreadItemPayload::Tool {
        call_id: "ig_missing".to_string(),
        name: "image_generation".to_string(),
        arguments: Vec::new(),
        output: None,
    };
    thread.turns[0].items[0].metadata = json!({
        "provider_metadata": {
            "raw_response_item": {
                "id": "ig_missing",
                "type": "image_generation_call",
                "status": "completed"
            }
        }
    });

    let error = project_thread(thread).expect_err("missing image result must fail");
    assert!(error.message.contains("omitted string result"));
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
fn canonical_unknown_item_projects_typed_v2_diagnostic() {
    let mut thread = canonical_thread(false);
    thread.turns[0].items[0].kind = canonical::ItemKind::Unknown;
    thread.turns[0].items[0].status = canonical::ItemStatus::Completed;
    thread.turns[0].items[0].payload = canonical::ThreadItemPayload::Unknown {
        upstream_type: "futureCapability".to_string(),
        field_names: vec![
            "[redacted]".to_string(),
            "label".to_string(),
            "opaquePayload".to_string(),
        ],
    };

    let projected = project_thread(thread).expect("project typed unknown item");
    let v2::ThreadItem::UnknownItem {
        id,
        upstream_type,
        field_names,
    } = &projected.turns[0].items[0]
    else {
        panic!("typed unknown item");
    };
    assert_eq!(id, "item_message-1");
    assert_eq!(upstream_type, "futureCapability");
    assert_eq!(
        field_names,
        &vec![
            "[redacted]".to_string(),
            "label".to_string(),
            "opaquePayload".to_string(),
        ]
    );
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
