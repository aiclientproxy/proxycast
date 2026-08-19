use super::*;

#[tokio::test]
async fn append_external_runtime_events_keeps_sequence_and_turn_scope() {
    let core = RuntimeCore::default();
    let session = core
        .start_session(AgentSessionStartParams {
            session_id: Some("sess_external".to_string()),
            thread_id: Some("thread_external".to_string()),
            app_id: "content-studio".to_string(),
            workspace_id: Some("default".to_string()),
            business_object_ref: None,
            locale: None,
        })
        .expect("session")
        .session;
    let output = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: session.session_id.clone(),
                turn_id: None,
                input: AgentInput {
                    text: "hello".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("turn");
    let turn_id = output.response.turn.turn_id;

    let appended = core
        .append_external_runtime_events(
            &session.session_id,
            Some(&turn_id),
            vec![RuntimeEvent::new(
                "message.delta",
                json!({ "text": "delta" }),
            )],
        )
        .expect("append");

    assert_eq!(appended.len(), 2);
    assert_eq!(appended[0].sequence, 5);
    assert_eq!(appended[0].event_type, "item.started");
    assert_eq!(appended[0].payload["item"]["kind"], "agentMessage");
    assert_eq!(appended[1].sequence, 6);
    assert_eq!(appended[1].session_id, "sess_external");
    assert_eq!(appended[1].thread_id.as_deref(), Some("thread_external"));
    assert_eq!(appended[1].turn_id.as_deref(), Some(turn_id.as_str()));
    assert_eq!(appended[1].event_type, "message.delta");
    assert_eq!(appended[1].payload["text"], "delta");
}

#[tokio::test]
async fn runtime_error_does_not_preempt_a_later_turn_failed_terminal() {
    let (core, session_id, turn_id) = runtime_with_active_turn(
        "sess_runtime_error_before_terminal",
        "thread_runtime_error_before_terminal",
        "turn_runtime_error_before_terminal",
    )
    .await;

    core.append_external_runtime_events(
        &session_id,
        Some(&turn_id),
        vec![RuntimeEvent::new(
            "runtime.error",
            json!({ "message": "diagnostic before terminal" }),
        )],
    )
    .expect("append runtime diagnostic");

    let after_diagnostic = core
        .read_session(AgentSessionReadParams {
            session_id: session_id.clone(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read active session after diagnostic");
    assert_eq!(after_diagnostic.session.status, AgentSessionStatus::Running);

    let terminal_events = core
        .append_external_runtime_events(
            &session_id,
            Some(&turn_id),
            vec![RuntimeEvent::new(
                "turn.failed",
                json!({ "message": "canonical terminal" }),
            )],
        )
        .expect("append canonical failure terminal");
    assert!(
        terminal_events
            .iter()
            .any(|event| event.event_type == "turn.failed"),
        "turn.failed must not be discarded after a runtime.error diagnostic"
    );

    let after_terminal = core
        .read_session(AgentSessionReadParams {
            session_id,
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read failed session after terminal");
    assert_eq!(after_terminal.session.status, AgentSessionStatus::Failed);
}

#[tokio::test]
async fn regular_canonical_projection_failure_is_returned_after_event_log_append() {
    let temp = tempfile::tempdir().expect("tempdir");
    let database_path = temp.path().join("projection.sqlite");
    let event_log_root = temp.path().join("event-log");
    let projection_store =
        Arc::new(ProjectionStore::initialize(&database_path).expect("projection store"));
    let event_log_writer = Arc::new(EventLogWriter::new(&event_log_root).expect("event log"));
    let core = RuntimeCore::default()
        .with_projection_store(projection_store)
        .with_event_log_writer(event_log_writer.clone());
    let session = core
        .start_session(AgentSessionStartParams {
            session_id: Some("sess_projection_failure".to_string()),
            thread_id: Some("thread_projection_failure".to_string()),
            app_id: "content-studio".to_string(),
            workspace_id: Some("default".to_string()),
            business_object_ref: None,
            locale: None,
        })
        .expect("session")
        .session;
    let turn = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: session.session_id.clone(),
                turn_id: Some("turn_projection_failure".to_string()),
                input: AgentInput {
                    text: "hello".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("turn")
        .response
        .turn;
    let memory_event_count = core
        .events_for_session(&session.session_id)
        .expect("events before projection failure")
        .len();

    let next_item_ordinal = u64::try_from(memory_event_count).expect("event count") + 1;
    rusqlite::Connection::open(&database_path)
        .expect("open projection database")
        .execute(
            "INSERT INTO canonical_items (
                thread_id, turn_id, item_id, ordinal, sequence, item_json
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![
                session.thread_id.as_str(),
                turn.turn_id.as_str(),
                "forced-ordinal-conflict",
                next_item_ordinal,
                memory_event_count,
                "{}"
            ],
        )
        .expect("inject canonical ordinal conflict");

    let error = core
        .append_external_runtime_events(
            &session.session_id,
            Some(&turn.turn_id),
            vec![RuntimeEvent::new(
                "message.delta",
                json!({ "itemId": "projection-failure-message", "text": "durable" }),
            )],
        )
        .expect_err("canonical projection failure must be returned");
    let RuntimeCoreError::Backend(message) = error else {
        panic!("expected canonical projection backend error");
    };
    assert!(message.contains("canonical ThreadStore projection failed after EventLog append"));
    assert_eq!(
        core.events_for_session(&session.session_id)
            .expect("events after projection failure")
            .len(),
        memory_event_count,
        "failed canonical projection must not advance in-memory session history"
    );
    assert!(event_log_writer
        .read_session_events(&session.session_id)
        .expect("durable EventLog-first tail")
        .iter()
        .any(|record| {
            record.event.event_type == "message.delta" && record.event.payload["text"] == "durable"
        }));
}

#[tokio::test]
async fn append_external_runtime_events_keeps_text_delta_fast_path_and_terminal_guards() {
    let core = RuntimeCore::default();
    let session = core
        .start_session(AgentSessionStartParams {
            session_id: Some("sess_text_delta_fast_path".to_string()),
            thread_id: Some("thread_text_delta_fast_path".to_string()),
            app_id: "content-studio".to_string(),
            workspace_id: Some("default".to_string()),
            business_object_ref: None,
            locale: None,
        })
        .expect("session")
        .session;
    let output = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: session.session_id.clone(),
                turn_id: Some("turn_text_delta_fast_path".to_string()),
                input: AgentInput {
                    text: "hello".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("turn");
    let turn_id = output.response.turn.turn_id;

    let text_events = (0..128)
        .map(|index| {
            RuntimeEvent::new(
                "message.delta",
                json!({ "text": format!("chunk-{index};") }),
            )
        })
        .collect::<Vec<_>>();
    let appended = core
        .append_external_runtime_events(&session.session_id, Some(&turn_id), text_events)
        .expect("append text deltas");
    assert_eq!(appended.len(), 129);
    assert_eq!(appended[0].sequence, 5);
    assert_eq!(appended[0].event_type, "item.started");
    assert_eq!(appended[1].sequence, 6);
    assert_eq!(appended[1].event_type, "message.delta");
    assert_eq!(appended[128].sequence, 133);

    core.append_external_runtime_events(
        &session.session_id,
        Some(&turn_id),
        vec![RuntimeEvent::new(
            "item.started",
            json!({
                "item": {
                    "sessionId": session.session_id.clone(),
                    "threadId": session.thread_id.clone(),
                    "turnId": turn_id.clone(),
                    "itemId": "item_tool_after_text_history",
                    "sequence": 134,
                    "ordinal": 134,
                    "createdAtMs": 1,
                    "updatedAtMs": 1,
                    "kind": "tool",
                    "status": "inProgress",
                    "payload": {
                        "type": "tool",
                        "call_id": "tool_after_text_history",
                        "name": "WebFetch",
                        "arguments": []
                    },
                    "metadata": {}
                }
            }),
        )],
    )
    .expect("tool start should remain accepted after text delta history");

    let text_events_after_tool = (0..64)
        .map(|index| {
            RuntimeEvent::new(
                "message.delta",
                json!({ "text": format!("post-tool-chunk-{index};") }),
            )
        })
        .collect::<Vec<_>>();
    core.append_external_runtime_events(
        &session.session_id,
        Some(&turn_id),
        text_events_after_tool,
    )
    .expect("text deltas after tool start should remain on the fast path");

    let error = core
        .append_external_runtime_events(
            &session.session_id,
            Some(&turn_id),
            vec![RuntimeEvent::new("turn.completed", json!({}))],
        )
        .expect_err("terminal turn with active tool must still fail closed");
    match error {
        RuntimeCoreError::Backend(message) => {
            assert!(message.contains("tool_unclosed_at_turn_end"));
        }
        other => panic!("expected backend sequence validation error, got {other:?}"),
    }
}

#[tokio::test]
async fn append_external_runtime_events_rejects_invalid_state_delta_before_storage() {
    let core = RuntimeCore::default();
    let session = core
        .start_session(AgentSessionStartParams {
            session_id: Some("sess_state_delta_schema".to_string()),
            thread_id: Some("thread_state_delta_schema".to_string()),
            app_id: "content-studio".to_string(),
            workspace_id: Some("default".to_string()),
            business_object_ref: None,
            locale: None,
        })
        .expect("session")
        .session;
    let output = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: session.session_id.clone(),
                turn_id: Some("turn_state_delta_schema".to_string()),
                input: AgentInput {
                    text: "hello".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("turn");
    let turn_id = output.response.turn.turn_id;
    let before_event_count = {
        let state = core
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        state
            .sessions
            .get(&session.session_id)
            .expect("stored session before")
            .events
            .len()
    };

    let error = core
        .append_external_runtime_events(
            &session.session_id,
            Some(&turn_id),
            vec![RuntimeEvent::new(
                "state.delta",
                json!({
                    "target": "projection",
                    "patch": [{ "op": "remove" }]
                }),
            )],
        )
        .expect_err("invalid state.delta must fail closed");

    match error {
        RuntimeCoreError::Backend(message) => {
            assert!(message.contains("agent runtime state delta schema validation failed"));
        }
        other => panic!("expected backend validation error, got {other:?}"),
    }

    let after_event_count = {
        let state = core
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        state
            .sessions
            .get(&session.session_id)
            .expect("stored session after")
            .events
            .len()
    };
    assert_eq!(after_event_count, before_event_count);
}
