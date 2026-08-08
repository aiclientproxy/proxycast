use super::*;
use crate::RuntimeCore;
use std::sync::Arc;

fn image_command_request(workspace: &TempDir) -> ExecutionRequest {
    request_for_test(
        "画一张广州夏天的图",
        None,
        Some(json!({
            "harness": {
                "projectRoot": workspace.path().to_string_lossy(),
                "image_command_intent": {
                    "kind": "image_command",
                    "image_task": {
                        "prompt": "画一张广州夏天的图",
                        "mode": "generate",
                        "count": 1,
                        "provider_id": "openai",
                        "model": "gpt-image-2",
                        "executor_mode": "images_api"
                    }
                }
            }
        })),
    )
}

#[tokio::test]
async fn cancel_turn_cancels_runtime_stream_token() {
    let backend = RuntimeBackend::new();
    let request = request_for_test("请持续输出", None, None);
    let cancel_token = backend
        .agent_state
        .create_cancel_token(&request.session.session_id)
        .await;
    let mut sink = TestRuntimeEventSink::default();

    ExecutionBackend::cancel_turn(
        &backend,
        CancelExecutionRequest {
            host: RuntimeHostContext::default(),
            session: request.session,
            turn: request.turn,
        },
        &mut sink,
    )
    .await
    .expect("cancel should emit a runtime event");

    assert!(cancel_token.is_cancelled());
    assert_eq!(sink.events.len(), 1);
    assert_eq!(sink.events[0].event_type, "turn.canceled");
}

#[tokio::test]
async fn runtime_backend_image_command_short_circuits_before_chat_model_routing() {
    let backend = RuntimeBackend::new();
    let workspace = TempDir::new().expect("workspace");
    let request = image_command_request(&workspace);
    let mut sink = TestRuntimeEventSink::default();

    backend
        .handle_turn_start(request, &mut sink)
        .await
        .expect("image command should not require chat model routing");

    let event_types = sink
        .events
        .iter()
        .filter(|event| !event.event_type.starts_with("workflow."))
        .map(|event| event.event_type.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        event_types,
        vec![
            "runtime.status",
            "item.started",
            "image_task.create_failed",
            "item.completed",
            "turn.completed"
        ]
    );
    let started_item = &sink
        .events
        .iter()
        .find(|event| event.event_type == "item.started")
        .expect("canonical tool item started")
        .payload["item"];
    let completed_item = &sink
        .events
        .iter()
        .find(|event| event.event_type == "item.completed")
        .expect("canonical tool item completed")
        .payload["item"];
    assert_eq!(started_item["itemId"], completed_item["itemId"]);
    assert_eq!(started_item["itemId"], started_item["payload"]["call_id"]);
    assert_eq!(started_item["status"], "inProgress");
    assert_eq!(completed_item["status"], "failed");
    assert_eq!(
        completed_item["payload"]["output"]["error"].as_str(),
        Some("App Server image command workflow requires AppDataSource")
    );
    assert_eq!(
        sink.events
            .iter()
            .find(|event| event.event_type == "image_task.create_failed")
            .and_then(|event| event.payload["reasonCode"].as_str()),
        Some("app_data_source_unavailable")
    );
    let workflow_event_types = sink
        .events
        .iter()
        .filter(|event| event.event_type.starts_with("workflow."))
        .map(|event| event.event_type.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        workflow_event_types,
        vec![
            "workflow.run.started",
            "workflow.step.completed",
            "workflow.step.completed",
            "workflow.run.completed"
        ]
    );
    assert!(
        sink.events
            .iter()
            .all(|event| event.event_type != "routing.decision.made"),
        "image command should bypass chat model route events"
    );
}

#[tokio::test]
async fn runtime_core_image_command_bypasses_chat_route_preflight() {
    let workspace = TempDir::new().expect("workspace");
    let request = image_command_request(&workspace);
    let core = RuntimeCore::with_backend(Arc::new(RuntimeBackend::new()));
    let session = core
        .start_session(app_server_protocol::AgentSessionStartParams {
            session_id: Some(request.session.session_id.clone()),
            thread_id: Some(request.session.thread_id.clone()),
            app_id: request.session.app_id.clone(),
            workspace_id: request.session.workspace_id.clone(),
            business_object_ref: request.session.business_object_ref.clone(),
            locale: None,
        })
        .expect("session")
        .session;

    let output = core
        .start_turn(
            app_server_protocol::AgentSessionTurnStartParams {
                session_id: session.session_id,
                turn_id: Some(request.turn.turn_id),
                input: AgentInput {
                    text: request.input.concat_text(),
                    attachments: Vec::new(),
                },
                runtime_options: request.runtime_options,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("image command should bypass RuntimeCore chat route preflight");

    assert_eq!(output.response.turn.status, AgentTurnStatus::Completed);
    assert_eq!(
        output
            .events
            .iter()
            .find(|event| event.event_type == "image_task.create_failed")
            .and_then(|event| event.payload["reasonCode"].as_str()),
        Some("app_data_source_unavailable")
    );
    assert!(output
        .events
        .iter()
        .all(|event| event.event_type != "routing.decision.made"));
}

#[tokio::test]
async fn respond_action_without_pending_descriptor_fails_closed() {
    let db: lime_core::database::DbConnection = std::sync::Arc::new(std::sync::Mutex::new(
        rusqlite::Connection::open_in_memory().expect("db"),
    ));
    {
        let conn = db.lock().expect("db lock");
        lime_core::database::schema::create_tables(&conn).expect("schema");
    }
    let backend = RuntimeBackend::with_db(db);
    let request = request_for_test("hello", None, None);
    let mut sink = TestRuntimeEventSink::default();

    let error = ExecutionBackend::respond_action(
        &backend,
        ActionRespondRequest {
            host: RuntimeHostContext::default(),
            session: request.session,
            turn: Some(request.turn),
            request_id: "ask-1".to_string(),
            action_type: AgentSessionActionType::AskUser,
            decision: None,
            confirmed: false,
            response: None,
            user_data: None,
            metadata: None,
            event_name: None,
            action_scope: Some(AgentSessionActionScope {
                session_id: Some("session-1".to_string()),
                thread_id: Some("thread-1".to_string()),
                turn_id: Some("turn-1".to_string()),
            }),
            pending_action_descriptor: None,
        },
        &mut sink,
    )
    .await
    .expect_err("missing pending waiter must fail closed");

    assert!(matches!(
        error,
        RuntimeCoreError::ActionResponse { ref code, ref request_id }
            if code == "action_descriptor_invalid" && request_id == "ask-1"
    ));
    assert!(sink.events.is_empty());
}

#[tokio::test]
async fn runtime_backend_registers_current_gateway_tools_in_agent_registry() {
    let db: lime_core::database::DbConnection = std::sync::Arc::new(std::sync::Mutex::new(
        rusqlite::Connection::open_in_memory().expect("db"),
    ));
    {
        let conn = db.lock().expect("db lock");
        lime_core::database::schema::create_tables(&conn).expect("schema");
    }
    let backend = RuntimeBackend::with_db(db.clone());
    ExecutionBackend::set_app_data_source(&backend, std::sync::Arc::new(NoopAppDataSource))
        .expect("app data source should be accepted");
    backend
        .agent_state
        .init_agent_with_db(&db)
        .await
        .expect("agent should initialize");
    backend
        .register_current_native_tools_if_available()
        .await
        .expect("current gateway tools should register");

    for tool_name in [
        MEMORY_LIST_TOOL_NAME,
        MEMORY_READ_TOOL_NAME,
        MEMORY_SEARCH_TOOL_NAME,
        MEMORY_ADD_NOTE_TOOL_NAME,
    ] {
        assert!(
            backend.agent_state.contains_native_tool(tool_name).await,
            "{tool_name} should be registered as a native memory tool"
        );
    }
    assert!(
        backend
            .agent_state
            .contains_native_tool(TOOL_SEARCH_TOOL_NAME)
            .await,
        "{TOOL_SEARCH_TOOL_NAME} should be registered as the current deferred tool search native tool"
    );
}
