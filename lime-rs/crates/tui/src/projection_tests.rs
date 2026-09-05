use super::*;
use app_server_protocol::protocol::v2::{
    AgentMessageDeltaNotification, CollabAgentState, CollabAgentStatus, CollabAgentTool,
    CommandExecutionOutputDeltaNotification, CommandExecutionSource,
    DynamicToolCallOutputContentItem, FileChangePatchUpdatedNotification, FileUpdateChange,
    ImageGenerationItem, ItemCompletedNotification, ItemStartedNotification, McpToolCallError,
    McpToolCallResult, PatchChangeKind, SessionSource, Thread, ThreadActiveFlag, ThreadItem,
    ThreadStatus, Turn, TurnCompletedNotification, TurnDiffUpdatedNotification, TurnItemsView,
    TurnPlanStep, TurnPlanStepStatus, TurnPlanUpdatedNotification,
};
use serde_json::json;
use std::collections::HashMap;
use std::path::PathBuf;

#[test]
fn completed_agent_item_replaces_streaming_delta() {
    let mut projection = ConversationProjection::default();
    projection.apply(ServerNotification::AgentMessageDelta(
        AgentMessageDeltaNotification {
            thread_id: "thread-1".to_string(),
            turn_id: "turn-1".to_string(),
            item_id: "item-1".to_string(),
            delta: "你".to_string(),
        },
    ));
    projection.apply(ServerNotification::AgentMessageDelta(
        AgentMessageDeltaNotification {
            thread_id: "thread-1".to_string(),
            turn_id: "turn-1".to_string(),
            item_id: "item-1".to_string(),
            delta: "好".to_string(),
        },
    ));
    projection.apply(ServerNotification::ItemCompleted(
        ItemCompletedNotification {
            item: ThreadItem::AgentMessage {
                id: "item-1".to_string(),
                metadata: None,
                text: "你好。".to_string(),
                phase: None,
                memory_citation: None,
                delivery: None,
            },
            thread_id: "thread-1".to_string(),
            turn_id: "turn-1".to_string(),
            completed_at_ms: 1,
        },
    ));

    assert_eq!(projection.entries().len(), 1);
    assert_eq!(projection.entries()[0].text, "你好。");
    assert!(!projection.entries()[0].streaming);
    assert_eq!(projection.final_answer(), "你好。");
}

#[test]
fn hydrate_thread_restores_items_and_active_turn_identity() {
    let mut projection = ConversationProjection::default();
    projection.hydrate_thread(Thread {
        id: "thread-restore".to_string(),
        extra: None,
        session_id: "session-restore".to_string(),
        forked_from_id: None,
        parent_thread_id: None,
        preview: "恢复测试".to_string(),
        ephemeral: false,
        section: None,
        section_entered_at: None,
        project_id: None,
        history_mode: Default::default(),
        model_provider: "fixture".to_string(),
        created_at: 1,
        updated_at: 1,
        recency_at: None,
        status: ThreadStatus::Active {
            active_flags: vec![ThreadActiveFlag::WaitingOnUserInput],
        },
        path: None,
        cwd: PathBuf::from("/workspace"),
        cli_version: "test".to_string(),
        source: SessionSource::Cli,
        can_accept_direct_input: Some(true),
        thread_source: None,
        agent_nickname: None,
        agent_role: None,
        git_info: None,
        name: None,
        turns: vec![Turn {
            id: "turn-restore".to_string(),
            items: vec![ThreadItem::AgentMessage {
                id: "item-restore".to_string(),
                metadata: None,
                text: "已恢复".to_string(),
                phase: None,
                memory_citation: None,
                delivery: None,
            }],
            items_view: TurnItemsView::Full,
            status: TurnStatus::InProgress,
            error: None,
            started_at: Some(1),
            completed_at: None,
            duration_ms: None,
        }],
    });

    assert_eq!(projection.active_turn_id(), Some("turn-restore"));
    assert_eq!(projection.status(), "running");
    assert_eq!(projection.final_answer(), "已恢复");
}

#[test]
fn start_turn_records_active_turn_from_request_response() {
    let mut projection = ConversationProjection::default();
    projection.start_turn("turn-response".to_string());

    assert_eq!(projection.active_turn_id(), Some("turn-response"));
    assert_eq!(projection.status(), "running");
}

#[test]
fn plan_and_diff_notifications_replace_stable_projection_entries() {
    let mut projection = ConversationProjection::default();
    projection.apply(ServerNotification::TurnPlanUpdated(
        TurnPlanUpdatedNotification {
            thread_id: "thread-1".to_string(),
            turn_id: "turn-1".to_string(),
            explanation: None,
            plan: vec![TurnPlanStep {
                step: "运行测试".to_string(),
                status: TurnPlanStepStatus::InProgress,
            }],
        },
    ));
    projection.apply(ServerNotification::TurnDiffUpdated(
        TurnDiffUpdatedNotification {
            thread_id: "thread-1".to_string(),
            turn_id: "turn-1".to_string(),
            diff: "@@ -1 +1 @@\n-旧\n+新".to_string(),
        },
    ));
    projection.apply(ServerNotification::FileChangePatchUpdated(
        FileChangePatchUpdatedNotification {
            thread_id: "thread-1".to_string(),
            turn_id: "turn-1".to_string(),
            item_id: "patch-1".to_string(),
            changes: vec![FileUpdateChange {
                path: "src/lib.rs".to_string(),
                kind: PatchChangeKind::Update { move_path: None },
                diff: "+新".to_string(),
            }],
        },
    ));

    assert_eq!(projection.entries().len(), 3);
    assert!(projection.entries()[0].text.contains("运行测试"));
    assert_eq!(projection.entries()[0].kind, EntryKind::Plan);
    assert!(projection.entries()[1].text.contains("+新"));
    assert_eq!(projection.entries()[1].kind, EntryKind::Patch);
    assert!(projection.entries()[2].text.contains("src/lib.rs"));
    assert_eq!(projection.entries()[2].kind, EntryKind::Patch);
}

#[test]
fn patch_format_preserves_rename_destination() {
    let text = format_patch(&[FileUpdateChange {
        path: "/workspace/src/old.rs".to_string(),
        kind: PatchChangeKind::Update {
            move_path: Some("/workspace/src/new.rs".to_string()),
        },
        diff: "@@ -1 +1 @@\n-old\n+new".to_string(),
    }]);

    assert!(text.starts_with("updated /workspace/src/old.rs → /workspace/src/new.rs\n"));
}

#[test]
fn item_result_fields_become_structured_display_summaries() {
    let command = project_item(
        &ThreadItem::CommandExecution {
            id: "command-1".to_string(),
            metadata: None,
            plugin_id: None,
            script_path: None,
            command: "cargo test -p tui".to_string(),
            cwd: "/workspace".to_string(),
            process_id: None,
            source: CommandExecutionSource::Agent,
            status: CommandExecutionStatus::Completed,
            command_actions: Vec::new(),
            aggregated_output: Some("ok".to_string()),
            exit_code: Some(0),
            duration_ms: Some(42),
            terminal_interactions: Vec::new(),
        },
        false,
    )
    .expect("command projection");
    assert_eq!(command.status, Some(EntryStatus::Completed));
    assert!(command.text.contains("cargo test -p tui\nok"));
    assert_eq!(command.summary, vec!["exit 0", "duration 42ms"]);

    let patch = project_item(
        &ThreadItem::FileChange {
            id: "patch-1".to_string(),
            metadata: None,
            changes: vec![
                FileUpdateChange {
                    path: "src/new.rs".to_string(),
                    kind: PatchChangeKind::Add,
                    diff: "+new".to_string(),
                },
                FileUpdateChange {
                    path: "src/lib.rs".to_string(),
                    kind: PatchChangeKind::Update { move_path: None },
                    diff: "+changed".to_string(),
                },
            ],
            status: PatchApplyStatus::Completed,
        },
        false,
    )
    .expect("patch projection");
    assert_eq!(patch.summary, vec!["files: 2", "added: 1", "updated: 1"]);

    let mcp = project_item(
        &ThreadItem::McpToolCall {
            id: "mcp-1".to_string(),
            metadata: None,
            server: "docs".to_string(),
            tool: "search".to_string(),
            status: McpToolCallStatus::Failed,
            arguments: json!({"query": "tui"}),
            app_context: None,
            mcp_app_resource_uri: None,
            plugin_id: None,
            read_only_hint: None,
            result: Some(Box::new(McpToolCallResult {
                content: vec![json!("one")],
                structured_content: None,
                meta: None,
            })),
            error: Some(McpToolCallError {
                message: "upstream unavailable".to_string(),
            }),
            duration_ms: Some(9),
        },
        false,
    )
    .expect("mcp projection");
    assert_eq!(mcp.status, Some(EntryStatus::Failed));
    assert_eq!(
        mcp.summary,
        vec![
            "result items: 1",
            "error: upstream unavailable",
            "duration 9ms"
        ]
    );

    let dynamic = project_item(
        &ThreadItem::DynamicToolCall {
            id: "dynamic-1".to_string(),
            metadata: None,
            namespace: Some("browser".to_string()),
            tool: "open".to_string(),
            arguments: json!({}),
            status: DynamicToolCallStatus::Completed,
            content_items: Some(vec![DynamicToolCallOutputContentItem::InputText {
                text: "done".to_string(),
            }]),
            success: Some(true),
            duration_ms: Some(12),
        },
        false,
    )
    .expect("dynamic projection");
    assert_eq!(
        dynamic.summary,
        vec!["success: true", "content items: 1", "duration 12ms"]
    );

    let mut agents_states = HashMap::new();
    agents_states.insert(
        "agent-1".to_string(),
        CollabAgentState {
            status: CollabAgentStatus::Running,
            message: None,
        },
    );
    agents_states.insert(
        "agent-2".to_string(),
        CollabAgentState {
            status: CollabAgentStatus::Completed,
            message: None,
        },
    );
    let collab = project_item(
        &ThreadItem::CollabAgentToolCall {
            id: "collab-1".to_string(),
            metadata: None,
            tool: CollabAgentTool::SpawnAgent,
            status: CollabAgentToolCallStatus::Completed,
            sender_thread_id: "thread-1".to_string(),
            receiver_thread_ids: vec!["agent-1".to_string(), "agent-2".to_string()],
            prompt: None,
            model: None,
            reasoning_effort: None,
            agents_states,
        },
        false,
    )
    .expect("collab projection");
    assert_eq!(
        collab.summary,
        vec!["agents: 2", "completed: 1", "running: 1"]
    );

    let image = project_item(
        &ThreadItem::ImageGeneration(ImageGenerationItem {
            id: "image-1".to_string(),
            metadata: None,
            status: "completed".to_string(),
            revised_prompt: Some("a concise prompt".to_string()),
            result: "https://example.test/image.png".to_string(),
            saved_path: Some("/tmp/image.png".to_string()),
        }),
        false,
    )
    .expect("image projection");
    assert_eq!(image.status, Some(EntryStatus::Completed));
    assert_eq!(
        image.summary,
        vec![
            "result: https://example.test/image.png",
            "saved: /tmp/image.png",
            "revised prompt: a concise prompt"
        ]
    );
}

#[test]
fn command_output_deltas_follow_the_command_without_splitting_chunks() {
    let mut projection = ConversationProjection::default();
    projection.apply(ServerNotification::ItemStarted(ItemStartedNotification {
        item: command_item(
            "command-1",
            "printf 'stdout\\nstderr\\n'",
            CommandExecutionStatus::InProgress,
            None,
        ),
        thread_id: "thread-1".to_string(),
        turn_id: "turn-1".to_string(),
        started_at_ms: 1,
    }));
    for delta in ["std", "out\nstderr\n"] {
        projection.apply(ServerNotification::CommandExecutionOutputDelta(
            CommandExecutionOutputDeltaNotification {
                thread_id: "thread-1".to_string(),
                turn_id: "turn-1".to_string(),
                item_id: "command-1".to_string(),
                delta: delta.to_string(),
            },
        ));
    }

    assert_eq!(
        projection.entries()[0].text,
        "printf 'stdout\\nstderr\\n'\nstdout\nstderr\n"
    );
    assert!(projection.entries()[0].streaming);
    assert_eq!(projection.entries()[0].status, Some(EntryStatus::Running));
}

#[test]
fn completed_command_item_replaces_live_output_with_canonical_output() {
    let mut projection = ConversationProjection::default();
    projection.apply(ServerNotification::ItemStarted(ItemStartedNotification {
        item: command_item(
            "command-1",
            "printf data",
            CommandExecutionStatus::InProgress,
            None,
        ),
        thread_id: "thread-1".to_string(),
        turn_id: "turn-1".to_string(),
        started_at_ms: 1,
    }));
    projection.apply(ServerNotification::CommandExecutionOutputDelta(
        CommandExecutionOutputDeltaNotification {
            thread_id: "thread-1".to_string(),
            turn_id: "turn-1".to_string(),
            item_id: "command-1".to_string(),
            delta: "partial".to_string(),
        },
    ));
    projection.apply(ServerNotification::ItemCompleted(
        ItemCompletedNotification {
            item: command_item(
                "command-1",
                "printf data",
                CommandExecutionStatus::Completed,
                Some("canonical output"),
            ),
            thread_id: "thread-1".to_string(),
            turn_id: "turn-1".to_string(),
            completed_at_ms: 2,
        },
    ));

    assert_eq!(
        projection.entries()[0].text,
        "printf data\ncanonical output"
    );
    assert!(!projection.entries()[0].streaming);
    assert_eq!(projection.entries()[0].status, Some(EntryStatus::Completed));
}

fn command_item(
    id: &str,
    command: &str,
    status: CommandExecutionStatus,
    aggregated_output: Option<&str>,
) -> ThreadItem {
    ThreadItem::CommandExecution {
        id: id.to_string(),
        metadata: None,
        plugin_id: None,
        script_path: None,
        command: command.to_string(),
        cwd: "/workspace".to_string(),
        process_id: None,
        source: CommandExecutionSource::Agent,
        status,
        command_actions: Vec::new(),
        aggregated_output: aggregated_output.map(str::to_string),
        exit_code: (status == CommandExecutionStatus::Completed).then_some(0),
        duration_ms: None,
        terminal_interactions: Vec::new(),
    }
}

#[test]
fn collab_agent_summary_keeps_requested_model_effort_and_prompt() {
    let collab = project_item(
        &ThreadItem::CollabAgentToolCall {
            id: "collab-1".to_string(),
            metadata: None,
            tool: CollabAgentTool::SpawnAgent,
            status: CollabAgentToolCallStatus::Completed,
            sender_thread_id: "thread-1".to_string(),
            receiver_thread_ids: vec!["agent-1".to_string()],
            prompt: Some("Inspect the repository structure".to_string()),
            model: Some("fixture-model".to_string()),
            reasoning_effort: Some(
                app_server_protocol::protocol::v2::ReasoningEffort::new("high").expect("effort"),
            ),
            agents_states: HashMap::new(),
        },
        false,
    )
    .expect("collab projection");

    assert_eq!(
        collab.summary,
        vec![
            "agents: 0",
            "model: fixture-model",
            "effort: high",
            "prompt: Inspect the repository structure",
        ]
    );
}

#[test]
fn turn_terminal_status_settles_in_progress_items() {
    let mut projection = ConversationProjection::default();
    projection.apply(ServerNotification::ItemStarted(ItemStartedNotification {
        item: ThreadItem::CommandExecution {
            id: "command-1".to_string(),
            metadata: None,
            plugin_id: None,
            script_path: None,
            command: "cargo test -p tui".to_string(),
            cwd: "/workspace".to_string(),
            process_id: None,
            source: CommandExecutionSource::Agent,
            status: CommandExecutionStatus::InProgress,
            command_actions: Vec::new(),
            aggregated_output: None,
            exit_code: None,
            duration_ms: None,
            terminal_interactions: Vec::new(),
        },
        thread_id: "thread-1".to_string(),
        turn_id: "turn-1".to_string(),
        started_at_ms: 1,
    }));
    projection.apply(ServerNotification::TurnCompleted(
        TurnCompletedNotification {
            thread_id: "thread-1".to_string(),
            turn: Turn {
                id: "turn-1".to_string(),
                items: Vec::new(),
                items_view: TurnItemsView::Full,
                status: TurnStatus::Interrupted,
                error: None,
                started_at: Some(1),
                completed_at: Some(2),
                duration_ms: Some(1),
            },
        },
    ));

    assert_eq!(projection.status(), "interrupted");
    assert_eq!(
        projection.entries()[0].status,
        Some(EntryStatus::Interrupted)
    );
}

#[test]
fn turn_completion_repairs_missing_streamed_items_from_canonical_turn() {
    let mut projection = ConversationProjection::default();
    projection.start_turn("turn-1".to_string());

    projection.apply(ServerNotification::TurnCompleted(
        TurnCompletedNotification {
            thread_id: "thread-1".to_string(),
            turn: Turn {
                id: "turn-1".to_string(),
                items: vec![ThreadItem::AgentMessage {
                    id: "answer-1".to_string(),
                    metadata: None,
                    text: "最终回答".to_string(),
                    phase: None,
                    memory_citation: None,
                    delivery: None,
                }],
                items_view: TurnItemsView::Full,
                status: TurnStatus::Completed,
                error: None,
                started_at: Some(1),
                completed_at: Some(2),
                duration_ms: Some(1),
            },
        },
    ));

    assert_eq!(projection.final_answer(), "最终回答");
    assert_eq!(projection.entries().len(), 1);
    assert!(!projection.entries()[0].streaming);
}

#[test]
fn turn_completion_replaces_streaming_item_with_canonical_text() {
    let mut projection = ConversationProjection::default();
    projection.apply(ServerNotification::AgentMessageDelta(
        AgentMessageDeltaNotification {
            thread_id: "thread-1".to_string(),
            turn_id: "turn-1".to_string(),
            item_id: "answer-1".to_string(),
            delta: "部分".to_string(),
        },
    ));

    projection.apply(ServerNotification::TurnCompleted(
        TurnCompletedNotification {
            thread_id: "thread-1".to_string(),
            turn: Turn {
                id: "turn-1".to_string(),
                items: vec![ThreadItem::AgentMessage {
                    id: "answer-1".to_string(),
                    metadata: None,
                    text: "完整最终回答".to_string(),
                    phase: None,
                    memory_citation: None,
                    delivery: None,
                }],
                items_view: TurnItemsView::Full,
                status: TurnStatus::Completed,
                error: None,
                started_at: Some(1),
                completed_at: Some(2),
                duration_ms: Some(1),
            },
        },
    ));

    assert_eq!(projection.entries().len(), 1);
    assert_eq!(projection.entries()[0].text, "完整最终回答");
    assert!(!projection.entries()[0].streaming);
}

#[test]
fn turn_completion_inserts_missing_canonical_items_before_known_following_items() {
    let mut projection = ConversationProjection::default();
    projection.apply(ServerNotification::AgentMessageDelta(
        AgentMessageDeltaNotification {
            thread_id: "thread-1".to_string(),
            turn_id: "turn-1".to_string(),
            item_id: "answer-1".to_string(),
            delta: "回答".to_string(),
        },
    ));

    projection.apply(ServerNotification::TurnCompleted(
        TurnCompletedNotification {
            thread_id: "thread-1".to_string(),
            turn: Turn {
                id: "turn-1".to_string(),
                items: vec![
                    ThreadItem::UserMessage {
                        id: "user-1".to_string(),
                        metadata: None,
                        client_id: Some("client-1".to_string()),
                        content: vec![app_server_protocol::protocol::v2::UserInput::Text {
                            text: "请求".to_string(),
                            text_elements: Vec::new(),
                        }],
                    },
                    ThreadItem::AgentMessage {
                        id: "answer-1".to_string(),
                        metadata: None,
                        text: "完整回答".to_string(),
                        phase: None,
                        memory_citation: None,
                        delivery: None,
                    },
                ],
                items_view: TurnItemsView::Full,
                status: TurnStatus::Completed,
                error: None,
                started_at: Some(1),
                completed_at: Some(2),
                duration_ms: Some(1),
            },
        },
    ));

    assert_eq!(
        projection
            .entries()
            .iter()
            .map(|entry| entry.text.as_str())
            .collect::<Vec<_>>(),
        vec!["请求", "完整回答"]
    );
}
