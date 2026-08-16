use super::*;

#[tokio::test]
async fn append_external_runtime_events_rejects_coding_terminal_without_start() {
    let (core, session_id, turn_id) = runtime_with_active_turn(
        "sess_coding_terminal_without_start",
        "thread_coding_terminal_without_start",
        "turn_coding_terminal_without_start",
    )
    .await;
    let before = read_session(&core, &session_id);
    let before_event_count = event_count(&core, &session_id);

    let error = core
        .append_external_runtime_events(
            &session_id,
            Some(&turn_id),
            vec![RuntimeEvent::new(
                "command.exited",
                json!({
                    "commandId": "cmd_without_start",
                    "exitCode": 0
                }),
            )],
        )
        .expect_err("command.exited without command.started must fail closed");

    match error {
        RuntimeCoreError::Backend(message) => {
            assert!(message.contains("agent runtime event sequence validation failed"));
            assert!(message.contains("command_exited_without_start"));
        }
        other => panic!("expected backend sequence validation error, got {other:?}"),
    }

    let after = read_session(&core, &session_id);
    assert_eq!(event_count(&core, &session_id), before_event_count);
    assert_eq!(after.turns[0].status, before.turns[0].status);
    assert_eq!(after.session.status, before.session.status);
}

#[tokio::test]
async fn append_external_runtime_events_rejects_incomplete_coding_payload() {
    let (core, session_id, turn_id) = runtime_with_active_turn(
        "sess_coding_payload_guard",
        "thread_coding_payload_guard",
        "turn_coding_payload_guard",
    )
    .await;
    let before = read_session(&core, &session_id);
    let before_event_count = event_count(&core, &session_id);

    let error = core
        .append_external_runtime_events(
            &session_id,
            Some(&turn_id),
            vec![RuntimeEvent::new(
                "file.changed",
                json!({ "path": "src/App.tsx" }),
            )],
        )
        .expect_err("file.changed without artifact reference must fail closed");

    match error {
        RuntimeCoreError::Backend(message) => {
            assert!(message.contains("file.changed events must include artifactId or artifactRefs"));
        }
        other => panic!("expected backend payload validation error, got {other:?}"),
    }

    let after = read_session(&core, &session_id);
    assert_eq!(event_count(&core, &session_id), before_event_count);
    assert_eq!(after.turns[0].status, before.turns[0].status);
    assert_eq!(after.session.status, before.session.status);
}

#[tokio::test]
async fn append_external_runtime_events_ignores_coding_execution_after_terminal_turn() {
    let (core, session_id, turn_id) = runtime_with_active_turn(
        "sess_coding_terminal_ignore",
        "thread_coding_terminal_ignore",
        "turn_coding_terminal_ignore",
    )
    .await;

    core.append_external_runtime_events(
        &session_id,
        Some(&turn_id),
        vec![RuntimeEvent::new("turn.completed", json!({}))],
    )
    .expect("turn completion should append");
    let before = read_session(&core, &session_id);
    let before_event_count = event_count(&core, &session_id);

    let appended = core
        .append_external_runtime_events(
            &session_id,
            Some(&turn_id),
            vec![RuntimeEvent::new(
                "file.changed",
                json!({
                    "path": "src/App.tsx",
                    "artifactId": "late_artifact"
                }),
            )],
        )
        .expect("terminal turns ignore late runtime events");

    let after = read_session(&core, &session_id);
    assert!(appended.is_empty());
    assert_eq!(event_count(&core, &session_id), before_event_count);
    assert_eq!(before.turns[0].status, AgentTurnStatus::Completed);
    assert_eq!(after.turns[0].status, AgentTurnStatus::Completed);
    assert_eq!(after.session.status, before.session.status);
}

#[tokio::test]
async fn start_turn_accepts_backend_emitted_coding_lifecycle() {
    let core = RuntimeCore::with_backend(Arc::new(CodingLifecycleBackend));
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_backend_coding_lifecycle".to_string()),
        thread_id: Some("thread_backend_coding_lifecycle".to_string()),
        app_id: "content-studio".to_string(),
        workspace_id: Some("default".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    let output = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_backend_coding_lifecycle".to_string(),
                turn_id: Some("turn_backend_coding_lifecycle".to_string()),
                input: AgentInput {
                    text: "update the project".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("backend coding lifecycle should complete");

    let event_types = output
        .events
        .iter()
        .map(|event| event.event_type.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        event_types,
        vec![
            "item.started",
            "message.created",
            "item.completed",
            "turn.accepted",
            "file.changed",
            "patch.started",
            "patch.applied",
            "command.started",
            "command.output",
            "command.exited",
            "test.started",
            "test.completed",
            "turn.completed",
        ]
    );
    assert_eq!(output.response.turn.status, AgentTurnStatus::Completed);

    let read = read_session(&core, "sess_backend_coding_lifecycle");
    assert_eq!(read.turns[0].status, AgentTurnStatus::Completed);
    assert_eq!(read.session.status, AgentSessionStatus::Completed);
}

#[tokio::test]
async fn start_turn_rejects_invalid_backend_coding_payload_before_storage() {
    let core = RuntimeCore::with_backend(Arc::new(InvalidCodingPayloadBackend));
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_backend_coding_payload_guard".to_string()),
        thread_id: Some("thread_backend_coding_payload_guard".to_string()),
        app_id: "content-studio".to_string(),
        workspace_id: Some("default".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    let error = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_backend_coding_payload_guard".to_string(),
                turn_id: Some("turn_backend_coding_payload_guard".to_string()),
                input: AgentInput {
                    text: "update the project".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect_err("invalid backend coding event should fail closed");

    assert!(
        error
            .to_string()
            .contains("file.changed events must include artifactId or artifactRefs"),
        "{error}"
    );

    let read = read_session(&core, "sess_backend_coding_payload_guard");
    assert!(read.turns.is_empty());
    assert_eq!(read.session.status, AgentSessionStatus::Idle);
    let events = core
        .events_for_session("sess_backend_coding_payload_guard")
        .expect("admission events");
    assert_eq!(events.len(), 4);
    assert_eq!(events[3].event_type, "turn.accepted");
    assert_eq!(events[3].payload["backend"], "app_server");
}
