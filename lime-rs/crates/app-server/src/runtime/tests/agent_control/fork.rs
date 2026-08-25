use super::*;
use agent_protocol::{ItemId, SessionId, ThreadItem, ToolOutput, TurnId};
use model_provider::current_client::{CurrentProviderContent, CurrentProviderRole};
use uuid::Uuid;

fn persistent_core() -> (
    tempfile::TempDir,
    Arc<EventLogWriter>,
    Arc<ProjectionStore>,
    RuntimeCore,
) {
    let temp = tempfile::tempdir().expect("tempdir");
    let event_log_writer =
        Arc::new(EventLogWriter::new(temp.path().join("event-log")).expect("event log writer"));
    let store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store"),
    );
    let core = RuntimeCore::default()
        .with_event_log_writer(event_log_writer.clone())
        .with_projection_store(store.clone());
    (temp, event_log_writer, store, core)
}

fn completed_item(
    session: &AgentSession,
    turn_id: &str,
    item_id: &str,
    payload: ThreadItemPayload,
) -> ThreadItem {
    let mut item = ThreadItem::new(
        SessionId::new(session.session_id.clone()),
        ThreadId::new(session.thread_id.clone()),
        TurnId::new(turn_id),
        0,
        0,
        payload,
    );
    item.item_id = ItemId::new(item_id);
    item.status = ItemStatus::Completed;
    item
}

fn item_lifecycle(item: ThreadItem) -> Vec<RuntimeEvent> {
    let mut started = item.clone();
    started.status = ItemStatus::InProgress;
    started.completed_at_ms = None;
    vec![
        RuntimeEvent::new("item.started", json!({ "item": started })),
        RuntimeEvent::new("item.completed", json!({ "item": item })),
    ]
}

async fn append_parent_turn(
    core: &RuntimeCore,
    session: &AgentSession,
    turn_id: &str,
    input: AgentInput,
    extra_events: Vec<RuntimeEvent>,
) {
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: session.session_id.clone(),
            turn_id: Some(turn_id.to_string()),
            input,
            runtime_options: None,
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("start parent turn");
    let mut events = extra_events;
    events.extend([
        RuntimeEvent::new(
            "message.delta",
            json!({
                "itemId": format!("final-{turn_id}"),
                "phase": "final_answer",
                "text": format!("answer for {turn_id}"),
            }),
        ),
        RuntimeEvent::new(
            "message.completed",
            json!({
                "itemId": format!("final-{turn_id}"),
                "phase": "final_answer",
                "status": "completed",
                "text": format!("answer for {turn_id}"),
            }),
        ),
        RuntimeEvent::new("turn.completed", json!({})),
    ]);
    core.append_external_runtime_events(&session.session_id, Some(turn_id), events)
        .expect("complete parent turn");
}

async fn assert_rejected_child_is_clean(
    core: &RuntimeCore,
    event_log_writer: &EventLogWriter,
    store: &ProjectionStore,
    parent_thread_id: &str,
    child_session_id: &str,
    child_thread_id: &str,
) {
    assert!(event_log_writer
        .read_session_events(child_session_id)
        .expect("read rejected child EventLog")
        .is_empty());
    assert!(store
        .read_thread(ReadThreadParams {
            thread_id: ThreadId::new(child_thread_id),
            include_archived: true,
            turns_view: ThreadTurnsView::Full,
        })
        .await
        .expect("read rejected child thread")
        .is_none());
    assert!(!core
        .is_pending_agent_control_thread(child_thread_id)
        .expect("read pending spawn intent"));
    assert!(store
        .read_agent_identity(ThreadId::new(child_thread_id))
        .await
        .expect("read rejected child identity")
        .is_none());
    assert!(store
        .list_pending_agent_mailbox_messages(
            ThreadId::new(parent_thread_id),
            ThreadId::new(child_thread_id),
        )
        .await
        .expect("read rejected child mailbox")
        .is_empty());
    assert!(matches!(
        core.read_session(AgentSessionReadParams {
            session_id: child_session_id.to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        }),
        Err(RuntimeCoreError::SessionNotFound(_))
    ));
}

#[tokio::test]
async fn fork_uses_codex_sanitize_profile_for_internal_items() {
    let (_temp, _event_log_writer, store, core) = persistent_core();
    let parent = core
        .start_session(start_params(
            "sanitize-parent-session",
            "sanitize-parent-thread",
        ))
        .expect("parent")
        .session;
    let turn_id = "sanitize-parent-turn";
    let payloads = [
        ThreadItemPayload::Reasoning {
            summary: vec!["summary".to_string()],
            content: vec!["private reasoning".to_string()],
        },
        ThreadItemPayload::Tool {
            call_id: "tool-call".to_string(),
            name: "exec_command".to_string(),
            arguments: Vec::new(),
            output: Some(ToolOutput {
                text: Some("tool output".to_string()),
                ..Default::default()
            }),
        },
        ThreadItemPayload::McpToolCall {
            call_id: "mcp-call".to_string(),
            server_name: "docs".to_string(),
            tool_name: "search".to_string(),
            app_context: None,
            mcp_app_resource_uri: None,
            plugin_id: None,
            arguments: Vec::new(),
            output: Some(ToolOutput {
                text: Some("mcp output".to_string()),
                ..Default::default()
            }),
        },
        ThreadItemPayload::Media {
            uri: "sidecar://media/result".to_string(),
            mime_type: "image/png".to_string(),
            preview: None,
        },
    ];
    let events = payloads
        .into_iter()
        .enumerate()
        .flat_map(|(index, payload)| {
            item_lifecycle(completed_item(
                &parent,
                turn_id,
                &format!("sanitize-{index}"),
                payload,
            ))
        })
        .collect();
    append_parent_turn(
        &core,
        &parent,
        turn_id,
        AgentInput {
            text: "parent task".to_string(),
            attachments: Vec::new(),
        },
        events,
    )
    .await;

    core.create_open_agent_control_child_for_test(AgentControlSpawnRequest {
        parent_session_id: parent.session_id,
        child_session_id: Some("sanitize-child-session".to_string()),
        child_thread_id: Some("sanitize-child-thread".to_string()),
        fork_mode: SpawnAgentForkMode::FullHistory,
    })
    .await
    .expect("Codex-sanitized fork");

    let child = store
        .read_thread(ReadThreadParams {
            thread_id: ThreadId::new("sanitize-child-thread"),
            include_archived: true,
            turns_view: ThreadTurnsView::Full,
        })
        .await
        .expect("read child")
        .expect("child thread");
    assert!(child
        .turns
        .iter()
        .flat_map(|turn| &turn.items)
        .all(|item| { matches!(item.kind, ItemKind::UserMessage | ItemKind::AgentMessage) }));
}

#[tokio::test]
async fn full_history_compaction_fails_before_child_side_effects() {
    let (_temp, event_log_writer, store, core) = persistent_core();
    let parent = core
        .start_session(start_params(
            "compact-parent-session",
            "compact-parent-thread",
        ))
        .expect("parent")
        .session;
    append_parent_turn(
        &core,
        &parent,
        "compact-parent-turn",
        AgentInput {
            text: "parent task".to_string(),
            attachments: Vec::new(),
        },
        vec![
            RuntimeEvent::new(
                "context.compaction.started",
                json!({ "compactionId": "compact-1" }),
            ),
            RuntimeEvent::new(
                "context.compaction.completed",
                json!({
                    "compactionId": "compact-1",
                    "summary": "bounded summary",
                }),
            ),
        ],
    )
    .await;

    let error = core
        .create_open_agent_control_child_for_test(AgentControlSpawnRequest {
            parent_session_id: parent.session_id.clone(),
            child_session_id: Some("compact-child-session".to_string()),
            child_thread_id: Some("compact-child-thread".to_string()),
            fork_mode: SpawnAgentForkMode::FullHistory,
        })
        .await
        .expect_err("full history cannot discard compaction lineage");
    assert!(error
        .to_string()
        .contains("compaction without replacement history"));
    assert_rejected_child_is_clean(
        &core,
        &event_log_writer,
        &store,
        &parent.thread_id,
        "compact-child-session",
        "compact-child-thread",
    )
    .await;

    core.create_open_agent_control_child_for_test(AgentControlSpawnRequest {
        parent_session_id: parent.session_id,
        child_session_id: Some("bounded-child-session".to_string()),
        child_thread_id: Some("bounded-child-thread".to_string()),
        fork_mode: SpawnAgentForkMode::LastNTurns(1),
    })
    .await
    .expect("bounded fork rebuilds from its selected turn");
}

#[tokio::test]
async fn full_history_rewrites_compaction_lineage_and_replays_after_restart() {
    let (_temp, event_log_writer, store, core) = persistent_core();
    let parent = core
        .start_session(start_params(
            "compact-valid-parent-session",
            "compact-valid-parent-thread",
        ))
        .expect("parent")
        .session;
    let first_window_id = Uuid::now_v7().to_string();
    let window_id = Uuid::now_v7().to_string();
    append_parent_turn(
        &core,
        &parent,
        "compact-valid-parent-turn",
        AgentInput {
            text: "parent task".to_string(),
            attachments: Vec::new(),
        },
        vec![RuntimeEvent::new(
            "context.compaction.completed",
            json!({
                "compactionId": "compact-valid-1",
                "tailStartTurnId": "compact-valid-parent-turn",
                "replacementHistory": [{
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "earlier context"}]
                }],
                "windowNumber": 1,
                "firstWindowId": first_window_id,
                "previousWindowId": null,
                "windowId": window_id,
                "artifact": {
                    "tailStartTurnId": "compact-valid-parent-turn",
                    "replacementHistory": [{
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "earlier context"}]
                    }],
                    "windowNumber": 1,
                    "firstWindowId": first_window_id,
                    "previousWindowId": null,
                    "windowId": window_id,
                },
            }),
        )],
    )
    .await;

    core.create_open_agent_control_child_for_test(AgentControlSpawnRequest {
        parent_session_id: parent.session_id,
        child_session_id: Some("compact-valid-child-session".to_string()),
        child_thread_id: Some("compact-valid-child-thread".to_string()),
        fork_mode: SpawnAgentForkMode::FullHistory,
    })
    .await
    .expect("valid compaction lineage is forkable");

    let child = store
        .read_thread(ReadThreadParams {
            thread_id: ThreadId::new("compact-valid-child-thread"),
            include_archived: true,
            turns_view: ThreadTurnsView::Full,
        })
        .await
        .expect("read child")
        .expect("child thread");
    let child_turn_id = child
        .turns
        .first()
        .expect("forked child turn")
        .turn_id
        .as_str()
        .to_string();
    let marker = event_log_writer
        .read_session_events("compact-valid-child-session")
        .expect("read child events")
        .into_iter()
        .find(|event| event.event.event_type == "context.compaction.completed")
        .expect("rewritten child compaction marker");
    assert_eq!(marker.event.turn_id, None);
    assert_eq!(marker.event.payload["tailStartTurnId"], child_turn_id);
    assert_eq!(
        marker.event.payload["artifact"]["tailStartTurnId"],
        child_turn_id
    );
    assert_eq!(
        marker.event.payload["replacementHistory"][0]["content"][0]["text"],
        "earlier context"
    );

    drop(core);
    let restarted = RuntimeCore::default()
        .with_event_log_writer(event_log_writer)
        .with_projection_store(store);
    restarted
        .ensure_current_session_hydrated("compact-valid-child-session")
        .await
        .expect("hydrate compacted child after restart");
    let stored = restarted
        .state
        .lock()
        .expect("runtime core state mutex poisoned")
        .sessions
        .get("compact-valid-child-session")
        .cloned()
        .expect("hydrated child session");
    let provider_history =
        crate::runtime::provider_history::provider_history_excluding_current_turn_input(
            &stored,
            None,
            "future-turn",
        )
        .expect("rebuild provider history after restart");
    assert!(provider_history.iter().any(|message| {
        message.role == CurrentProviderRole::User
            && matches!(
                &message.content[..],
                [CurrentProviderContent::Text(text)] if text == "earlier context"
            )
    }));
    assert!(provider_history.iter().any(|message| {
        message.role == CurrentProviderRole::Assistant
            && matches!(
                &message.content[..],
                [CurrentProviderContent::Text(text)] if text == "answer for compact-valid-parent-turn"
            )
    }));
}

#[tokio::test]
async fn last_n_validates_only_the_selected_turn_window() {
    let (_temp, event_log_writer, store, core) = persistent_core();
    let parent = core
        .start_session(start_params(
            "window-parent-session",
            "window-parent-thread",
        ))
        .expect("parent")
        .session;
    append_parent_turn(
        &core,
        &parent,
        "window-old-turn",
        AgentInput {
            text: "old task".to_string(),
            attachments: vec![app_server_protocol::AgentAttachment {
                kind: "image".to_string(),
                uri: Some("https://example.invalid/input.png".to_string()),
                metadata: None,
            }],
        },
        Vec::new(),
    )
    .await;
    append_completed_parent_turn(
        &core,
        &parent,
        "window-current-turn",
        "current task",
        "current answer",
    )
    .await;

    let error = core
        .create_open_agent_control_child_for_test(AgentControlSpawnRequest {
            parent_session_id: parent.session_id.clone(),
            child_session_id: Some("window-full-session".to_string()),
            child_thread_id: Some("window-full-thread".to_string()),
            fork_mode: SpawnAgentForkMode::FullHistory,
        })
        .await
        .expect_err("full history must reject lossy user input");
    assert!(error
        .to_string()
        .contains("user input that cannot be copied without loss"));
    assert_rejected_child_is_clean(
        &core,
        &event_log_writer,
        &store,
        &parent.thread_id,
        "window-full-session",
        "window-full-thread",
    )
    .await;

    core.create_open_agent_control_child_for_test(AgentControlSpawnRequest {
        parent_session_id: parent.session_id,
        child_session_id: Some("window-last-session".to_string()),
        child_thread_id: Some("window-last-thread".to_string()),
        fork_mode: SpawnAgentForkMode::LastNTurns(1),
    })
    .await
    .expect("last-N must not validate turns outside the selected window");
    let child = store
        .read_thread(ReadThreadParams {
            thread_id: ThreadId::new("window-last-thread"),
            include_archived: true,
            turns_view: ThreadTurnsView::Full,
        })
        .await
        .expect("read bounded child")
        .expect("bounded child");
    assert_eq!(child.turns.len(), 1);
    assert!(child.turns[0].items.iter().any(|item| matches!(
        &item.payload,
        ThreadItemPayload::UserMessage { content, .. }
            if content == &vec![agent_protocol::AgentInput::text("current task")]
    )));
}

#[tokio::test]
async fn final_answer_media_fails_closed_while_none_stays_fresh() {
    let (_temp, event_log_writer, store, core) = persistent_core();
    let parent = core
        .start_session(start_params("media-parent-session", "media-parent-thread"))
        .expect("parent")
        .session;
    let media_parts = vec![agent_protocol::MessageContentPart::Media {
        kind: "image".to_string(),
        reference: agent_protocol::MessageContentReference {
            uri: "sidecar://media/final".to_string(),
            mime_type: "image/png".to_string(),
            title: None,
            source_uri: None,
            source_path: None,
            preview_url: None,
            sidecar_ref: None,
            sha256: None,
            byte_size: None,
        },
        caption: None,
    }];
    append_parent_turn(
        &core,
        &parent,
        "media-parent-turn",
        AgentInput {
            text: "parent task".to_string(),
            attachments: Vec::new(),
        },
        vec![
            RuntimeEvent::new(
                "message.delta",
                json!({
                    "itemId": "media-final",
                    "phase": "final_answer",
                    "text": "result with image",
                    "contentParts": media_parts,
                }),
            ),
            RuntimeEvent::new(
                "message.completed",
                json!({
                    "itemId": "media-final",
                    "phase": "final_answer",
                    "status": "completed",
                    "text": "result with image",
                    "contentParts": media_parts,
                }),
            ),
        ],
    )
    .await;

    let error = core
        .create_open_agent_control_child_for_test(AgentControlSpawnRequest {
            parent_session_id: parent.session_id.clone(),
            child_session_id: Some("media-full-session".to_string()),
            child_thread_id: Some("media-full-thread".to_string()),
            fork_mode: SpawnAgentForkMode::FullHistory,
        })
        .await
        .expect_err("final-answer media must not be dropped");
    assert!(error
        .to_string()
        .contains("AgentMessage Item item_media-final that cannot be copied without loss"));
    assert_rejected_child_is_clean(
        &core,
        &event_log_writer,
        &store,
        &parent.thread_id,
        "media-full-session",
        "media-full-thread",
    )
    .await;

    core.create_open_agent_control_child_for_test(AgentControlSpawnRequest {
        parent_session_id: parent.session_id,
        child_session_id: Some("media-none-session".to_string()),
        child_thread_id: Some("media-none-thread".to_string()),
        fork_mode: SpawnAgentForkMode::None,
    })
    .await
    .expect("fork_turns=none must not inspect or copy parent history");
    let child = store
        .read_thread(ReadThreadParams {
            thread_id: ThreadId::new("media-none-thread"),
            include_archived: true,
            turns_view: ThreadTurnsView::Full,
        })
        .await
        .expect("read fresh child")
        .expect("fresh child");
    assert!(child.turns.is_empty());
    assert_eq!(child.forked_from_id, None);
}
