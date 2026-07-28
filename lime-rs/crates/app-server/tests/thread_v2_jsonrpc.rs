use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use app_server::{
    ActionRespondRequest, AppServer, CancelExecutionRequest, ExecutionBackend, ExecutionRequest,
    MockBackend, ProjectionStore, RuntimeCore, RuntimeCoreError, RuntimeEvent, RuntimeEventSink,
};
use app_server_protocol::protocol::v2::{
    METHOD_THREAD_ARCHIVE, METHOD_THREAD_GOAL_CLEAR, METHOD_THREAD_GOAL_CLEARED,
    METHOD_THREAD_GOAL_GET, METHOD_THREAD_GOAL_SET, METHOD_THREAD_GOAL_UPDATED, METHOD_THREAD_LIST,
    METHOD_THREAD_UNARCHIVE,
};
use app_server_protocol::{
    error_codes, METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_READ, METHOD_THREAD_RESUME,
    METHOD_THREAD_START, METHOD_THREAD_TURNS_LIST, METHOD_TURN_START, PROTOCOL_VERSION,
};
use async_trait::async_trait;
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::sync::Notify;
use tokio::time::{sleep, timeout};

struct BlockingTurnBackend {
    started: Arc<Notify>,
    release: Arc<Notify>,
}

struct RejectingThreadStartBackend;

#[async_trait]
impl ExecutionBackend for RejectingThreadStartBackend {
    fn requires_provider_selection(&self) -> bool {
        true
    }

    async fn preflight_thread_settings(
        &self,
        session: &app_server_protocol::AgentSession,
        settings: &app_server_protocol::protocol::v2::ThreadSettings,
    ) -> Result<(), RuntimeCoreError> {
        Err(RuntimeCoreError::RouteRejected {
            session_id: session.session_id.clone(),
            provider: Some(settings.model_provider.clone()),
            model: Some(settings.model.clone()),
            category: app_server_protocol::RouteFailureCategory::ModelUnavailable,
            reason_code: "provider_not_ready".to_string(),
        })
    }

    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn cancel_turn(
        &self,
        _request: CancelExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn respond_action(
        &self,
        _request: ActionRespondRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }
}

#[async_trait]
impl ExecutionBackend for BlockingTurnBackend {
    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        self.started.notify_one();
        self.release.notified().await;
        sink.emit(RuntimeEvent::new("turn.completed", json!({})))
    }

    async fn cancel_turn(
        &self,
        _request: CancelExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn respond_action(
        &self,
        _request: ActionRespondRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }
}

#[tokio::test]
async fn thread_start_returns_the_v2_thread_envelope() {
    let (_temp, server) = test_server();
    initialize_server(&server).await;

    let lines = request_lines(
        &server,
        2,
        METHOD_THREAD_START,
        json!({
            "model": "gpt-5.4",
            "modelProvider": "openai",
            "cwd": "/tmp/lime-thread-v2",
            "historyMode": "paginated",
            "multiAgentMode": "proactive"
        }),
    )
    .await;
    let response = lines
        .iter()
        .find(|value| value.get("id") == Some(&json!(2)))
        .expect("thread/start response");
    let started = lines
        .iter()
        .find(|value| value.get("method") == Some(&json!("thread/started")))
        .unwrap_or_else(|| panic!("thread/started notification: {lines:#?}"));
    assert_eq!(
        started.pointer("/params/thread/id"),
        response.pointer("/result/thread/id")
    );

    let thread_id = response
        .pointer("/result/thread/id")
        .and_then(Value::as_str)
        .expect("thread/start must return a canonical thread id");
    assert!(!thread_id.is_empty());
    assert_eq!(response.pointer("/result/model"), Some(&json!("gpt-5.4")));
    assert_eq!(
        response.pointer("/result/modelProvider"),
        Some(&json!("openai"))
    );
    assert_eq!(
        response.pointer("/result/thread/historyMode"),
        Some(&json!("paginated"))
    );
    assert_eq!(
        response.pointer("/result/thread/cwd"),
        Some(&json!("/tmp/lime-thread-v2"))
    );
    assert_eq!(
        response.pointer("/result/multiAgentMode"),
        Some(&json!("explicitRequestOnly"))
    );
    assert!(response.pointer("/result/session").is_none());

    let read = request(
        &server,
        3,
        METHOD_THREAD_READ,
        json!({"threadId": thread_id, "includeTurns": false}),
    )
    .await;
    assert_eq!(read.pointer("/result/thread/id"), Some(&json!(thread_id)));
    assert_eq!(
        read.pointer("/result/thread/extra/providerName"),
        Some(&json!("openai"))
    );
    assert_eq!(
        read.pointer("/result/thread/extra/modelName"),
        Some(&json!("gpt-5.4"))
    );
    assert_eq!(
        read.pointer("/result/thread/extra/workingDir"),
        Some(&json!("/tmp/lime-thread-v2"))
    );
    assert!(read
        .pointer("/result/thread/extra/multiAgentMode")
        .is_none());
    assert_eq!(
        read.pointer("/result/thread/extra/historyMode"),
        Some(&json!("paginated"))
    );
}

#[tokio::test]
async fn thread_start_rejects_partial_or_blank_explicit_route() {
    let (_temp, server) = test_server();
    initialize_server(&server).await;

    for (id, params) in [
        (2, json!({"modelProvider": "openai"})),
        (3, json!({"model": "gpt-5.4"})),
        (4, json!({"model": " ", "modelProvider": "openai"})),
    ] {
        let response = request_raw(&server, id, METHOD_THREAD_START, params).await;
        assert_eq!(
            response.pointer("/error/code"),
            Some(&json!(error_codes::INVALID_PARAMS))
        );
        assert!(response.get("result").is_none());
    }
}

#[tokio::test]
async fn thread_start_preflights_route_before_persisting_thread() {
    let temp = TempDir::new().expect("thread preflight temp dir");
    let projection_store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("thread preflight projection store"),
    );
    let runtime = RuntimeCore::with_backend(Arc::new(RejectingThreadStartBackend))
        .with_projection_store(projection_store);
    let server = AppServer::with_runtime(runtime);
    initialize_server(&server).await;

    let response = request_raw(
        &server,
        2,
        METHOD_THREAD_START,
        json!({"model": "model-a", "modelProvider": "provider-a"}),
    )
    .await;
    assert!(response.get("error").is_some());
    assert!(response.get("result").is_none());

    let listed = request(&server, 3, METHOD_THREAD_LIST, json!({})).await;
    assert_eq!(listed.pointer("/result/data"), Some(&json!([])));
}

#[tokio::test]
async fn plan_delta_uses_one_typed_item_lifecycle_in_public_jsonrpc_messages() {
    let (_temp, server) = test_server();
    initialize_server(&server).await;

    let thread = request(
        &server,
        2,
        METHOD_THREAD_START,
        json!({
            "model": "fixture-model",
            "modelProvider": "fixture-provider"
        }),
    )
    .await;
    let thread_id = thread
        .pointer("/result/thread/id")
        .and_then(Value::as_str)
        .expect("thread id");
    let session_id = thread
        .pointer("/result/thread/sessionId")
        .and_then(Value::as_str)
        .expect("session id");
    let turn = request(
        &server,
        3,
        METHOD_TURN_START,
        json!({
            "threadId": thread_id,
            "input": [{"type": "text", "text": "create a plan"}]
        }),
    )
    .await;
    let turn_id = turn
        .pointer("/result/turn/id")
        .and_then(Value::as_str)
        .expect("turn id");

    let messages = server
        .append_external_runtime_events(
            session_id,
            Some(turn_id),
            vec![
                RuntimeEvent::new(
                    "plan.delta",
                    json!({
                        "text": "- [ ] Read the protocol",
                        "delta": "- [ ] Read the protocol",
                        "revisionId": "public-plan:1"
                    }),
                ),
                RuntimeEvent::new(
                    "plan.delta",
                    json!({
                        "text": "- [ ] Read the protocol\n- [ ] Project the GUI",
                        "delta": "\n- [ ] Project the GUI",
                        "revisionId": "public-plan:1"
                    }),
                ),
                RuntimeEvent::new(
                    "plan.final",
                    json!({
                        "text": "- [ ] Read the protocol\n- [ ] Project the GUI",
                        "revisionId": "public-plan:1"
                    }),
                ),
            ],
        )
        .await
        .expect("plan notifications");
    let notifications = messages
        .into_iter()
        .map(|message| serde_json::to_value(message).expect("serialize JSON-RPC message"))
        .collect::<Vec<_>>();

    assert_eq!(
        notifications
            .iter()
            .map(|message| message["method"].as_str())
            .collect::<Vec<_>>(),
        vec![
            Some("item/started"),
            Some("item/plan/delta"),
            Some("item/plan/delta"),
            Some("item/completed"),
        ]
    );
    let item_id = notifications[0]
        .pointer("/params/item/id")
        .and_then(Value::as_str)
        .expect("plan item id");
    for notification in &notifications[1..3] {
        assert_eq!(
            notification.pointer("/params/itemId"),
            Some(&json!(item_id))
        );
    }
    assert_eq!(
        notifications[3].pointer("/params/item/id"),
        Some(&json!(item_id))
    );
    assert_eq!(
        notifications[2].pointer("/params/delta"),
        Some(&json!("\n- [ ] Project the GUI"))
    );
}

#[tokio::test]
async fn command_output_delta_uses_one_typed_item_lifecycle_in_public_jsonrpc_messages() {
    let (_temp, server) = test_server();
    initialize_server(&server).await;

    let thread = request(
        &server,
        2,
        METHOD_THREAD_START,
        json!({
            "model": "fixture-model",
            "modelProvider": "fixture-provider"
        }),
    )
    .await;
    let thread_id = thread
        .pointer("/result/thread/id")
        .and_then(Value::as_str)
        .expect("thread id");
    let session_id = thread
        .pointer("/result/thread/sessionId")
        .and_then(Value::as_str)
        .expect("session id");
    let turn = request(
        &server,
        3,
        METHOD_TURN_START,
        json!({
            "threadId": thread_id,
            "input": [{"type": "text", "text": "run a command"}]
        }),
    )
    .await;
    let turn_id = turn
        .pointer("/result/turn/id")
        .and_then(Value::as_str)
        .expect("turn id");

    let messages = server
        .append_external_runtime_events(
            session_id,
            Some(turn_id),
            vec![
                RuntimeEvent::new(
                    "command.started",
                    json!({
                        "commandId": "command-1",
                        "toolCallId": "command-1",
                        "command": "printf ready",
                        "cwd": "/workspace"
                    }),
                ),
                RuntimeEvent::new(
                    "command.output",
                    json!({
                        "commandId": "command-1",
                        "toolCallId": "command-1",
                        "outputRef": "output://command-1",
                        "delta": "ready\n"
                    }),
                ),
                RuntimeEvent::new(
                    "command.exited",
                    json!({
                        "commandId": "command-1",
                        "toolCallId": "command-1",
                        "command": "printf ready",
                        "cwd": "/workspace",
                        "exitCode": 0,
                        "status": "passed",
                        "success": true
                    }),
                ),
            ],
        )
        .await
        .expect("command notifications");
    let notifications = messages
        .into_iter()
        .map(|message| serde_json::to_value(message).expect("serialize JSON-RPC message"))
        .collect::<Vec<_>>();

    assert_eq!(
        notifications
            .iter()
            .map(|message| message["method"].as_str())
            .collect::<Vec<_>>(),
        vec![
            Some("item/started"),
            Some("item/commandExecution/outputDelta"),
            Some("item/completed")
        ]
    );
    let item_id = notifications[0]
        .pointer("/params/item/id")
        .and_then(Value::as_str)
        .expect("command item id");
    assert_eq!(item_id, "item_command-1");
    assert_eq!(
        notifications[1].pointer("/params/itemId"),
        Some(&json!(item_id))
    );
    assert_eq!(
        notifications[1].pointer("/params/delta"),
        Some(&json!("ready\n"))
    );
    assert_eq!(
        notifications[2].pointer("/params/item/id"),
        Some(&json!(item_id))
    );
}

#[tokio::test]
async fn file_change_patch_update_uses_one_typed_item_lifecycle_in_public_jsonrpc_messages() {
    let (_temp, server) = test_server();
    initialize_server(&server).await;

    let thread = request(
        &server,
        2,
        METHOD_THREAD_START,
        json!({
            "model": "fixture-model",
            "modelProvider": "fixture-provider"
        }),
    )
    .await;
    let thread_id = thread
        .pointer("/result/thread/id")
        .and_then(Value::as_str)
        .expect("thread id");
    let session_id = thread
        .pointer("/result/thread/sessionId")
        .and_then(Value::as_str)
        .expect("session id");
    let turn = request(
        &server,
        3,
        METHOD_TURN_START,
        json!({
            "threadId": thread_id,
            "input": [{"type": "text", "text": "apply a patch"}]
        }),
    )
    .await;
    let turn_id = turn
        .pointer("/result/turn/id")
        .and_then(Value::as_str)
        .expect("turn id");
    let changes = json!([{
        "path": "src/lib.rs",
        "kind": "update",
        "movePath": "src/main.rs",
        "diff": "-old\n+new"
    }]);

    let messages = server
        .append_external_runtime_events(
            session_id,
            Some(turn_id),
            vec![
                RuntimeEvent::new(
                    "patch.started",
                    json!({"patchId": "patch-1", "changes": changes}),
                ),
                RuntimeEvent::new(
                    "patch.applied",
                    json!({"patchId": "patch-1", "changes": changes}),
                ),
            ],
        )
        .await
        .expect("file change notifications");
    let notifications = messages
        .into_iter()
        .map(|message| serde_json::to_value(message).expect("serialize JSON-RPC message"))
        .collect::<Vec<_>>();

    assert_eq!(
        notifications
            .iter()
            .map(|message| message["method"].as_str())
            .collect::<Vec<_>>(),
        vec![
            Some("item/started"),
            Some("item/fileChange/patchUpdated"),
            Some("item/completed")
        ]
    );
    let item_id = notifications[0]
        .pointer("/params/item/id")
        .and_then(Value::as_str)
        .expect("file change item id");
    assert_eq!(item_id, "item_patch-1");
    assert_eq!(
        notifications[1].pointer("/params/itemId"),
        Some(&json!(item_id))
    );
    assert_eq!(
        notifications[1].pointer("/params/changes/0/kind"),
        Some(&json!({"type": "update", "move_path": "src/main.rs"}))
    );
    assert_eq!(
        notifications[2].pointer("/params/item/id"),
        Some(&json!(item_id))
    );
}

#[tokio::test]
async fn mcp_progress_uses_one_typed_item_lifecycle_in_public_jsonrpc_messages() {
    let (_temp, server) = test_server();
    initialize_server(&server).await;
    let (session_id, thread_id, turn_id) = start_thread_turn(&server, "call an MCP tool").await;

    let messages = server
        .append_external_runtime_events(
            &session_id,
            Some(&turn_id),
            vec![
                RuntimeEvent::new(
                    "item.started",
                    canonical_mcp_item("mcp-call-1", "inProgress"),
                ),
                RuntimeEvent::new(
                    "tool.progress",
                    mcp_progress_payload("mcp-call-1", "正在检索文档", Some("mcp_progress")),
                ),
                RuntimeEvent::new(
                    "item.completed",
                    canonical_mcp_item("mcp-call-1", "completed"),
                ),
            ],
        )
        .await
        .expect("MCP progress notifications");
    let notifications = messages
        .into_iter()
        .map(|message| serde_json::to_value(message).expect("serialize JSON-RPC message"))
        .collect::<Vec<_>>();

    assert_eq!(
        notifications
            .iter()
            .map(|message| message["method"].as_str())
            .collect::<Vec<_>>(),
        vec![
            Some("item/started"),
            Some("item/mcpToolCall/progress"),
            Some("item/completed")
        ]
    );
    let item_id = notifications[0]
        .pointer("/params/item/id")
        .and_then(Value::as_str)
        .expect("MCP item id");
    assert_eq!(item_id, "item_mcp-call-1");
    assert_eq!(
        notifications[1].pointer("/params/itemId"),
        Some(&json!(item_id))
    );
    assert_eq!(
        notifications[2].pointer("/params/item/id"),
        Some(&json!(item_id))
    );
    assert_eq!(
        notifications[1].pointer("/params/threadId"),
        Some(&json!(thread_id))
    );
    assert_eq!(
        notifications[1].pointer("/params/turnId"),
        Some(&json!(turn_id))
    );
}

#[tokio::test]
async fn mcp_progress_fails_closed_outside_its_typed_lifecycle() {
    for (scenario, events) in [
        (
            "before start",
            vec![RuntimeEvent::new(
                "tool.progress",
                mcp_progress_payload("mcp-call-1", "正在检索文档", Some("mcp_progress")),
            )],
        ),
        (
            "after completed",
            vec![
                RuntimeEvent::new(
                    "item.started",
                    canonical_mcp_item("mcp-call-1", "inProgress"),
                ),
                RuntimeEvent::new(
                    "item.completed",
                    canonical_mcp_item("mcp-call-1", "completed"),
                ),
                RuntimeEvent::new(
                    "tool.progress",
                    mcp_progress_payload("mcp-call-1", "正在检索文档", Some("mcp_progress")),
                ),
            ],
        ),
        (
            "generic tool",
            vec![
                RuntimeEvent::new(
                    "item.started",
                    canonical_dynamic_tool_item("mcp-call-1", "inProgress"),
                ),
                RuntimeEvent::new(
                    "tool.progress",
                    mcp_progress_payload("mcp-call-1", "正在检索文档", Some("mcp_progress")),
                ),
            ],
        ),
        (
            "empty message",
            vec![
                RuntimeEvent::new(
                    "item.started",
                    canonical_mcp_item("mcp-call-1", "inProgress"),
                ),
                RuntimeEvent::new(
                    "tool.progress",
                    mcp_progress_payload("mcp-call-1", "   ", Some("mcp_progress")),
                ),
            ],
        ),
        (
            "missing provenance",
            vec![
                RuntimeEvent::new(
                    "item.started",
                    canonical_mcp_item("mcp-call-1", "inProgress"),
                ),
                RuntimeEvent::new(
                    "tool.progress",
                    mcp_progress_payload("mcp-call-1", "正在检索文档", None),
                ),
            ],
        ),
        (
            "wrong provenance",
            vec![
                RuntimeEvent::new(
                    "item.started",
                    canonical_mcp_item("mcp-call-1", "inProgress"),
                ),
                RuntimeEvent::new(
                    "tool.progress",
                    mcp_progress_payload("mcp-call-1", "正在检索文档", Some("mcp_log")),
                ),
            ],
        ),
    ] {
        let (_temp, server) = test_server();
        initialize_server(&server).await;
        let (session_id, _, turn_id) = start_thread_turn(&server, scenario).await;
        let error = match server
            .append_external_runtime_events(&session_id, Some(&turn_id), events)
            .await
        {
            Ok(_) => panic!("{scenario} must fail closed"),
            Err(error) => error,
        };
        assert!(
            !error.message.trim().is_empty(),
            "{scenario} must return a diagnostic error"
        );
    }
}

#[tokio::test]
async fn thread_goal_lifecycle_is_durable_and_emits_ordered_notifications() {
    let temp = TempDir::new().expect("thread goal temp dir");
    let projection_path = temp.path().join("projection.sqlite");
    let server = AppServer::with_runtime(RuntimeCore::default().with_projection_store(Arc::new(
        ProjectionStore::initialize(&projection_path).expect("thread goal projection store"),
    )));
    initialize_server(&server).await;

    let started = request(
        &server,
        2,
        METHOD_THREAD_START,
        json!({
            "model": "fixture-model",
            "modelProvider": "fixture-provider",
            "historyMode": "legacy"
        }),
    )
    .await;
    let thread_id = started
        .pointer("/result/thread/id")
        .and_then(Value::as_str)
        .expect("thread/start id")
        .to_string();

    let set_lines = request_lines(
        &server,
        3,
        METHOD_THREAD_GOAL_SET,
        json!({
            "threadId": thread_id,
            "objective": "finish the durable goal",
            "status": "active",
            "tokenBudget": 1_000
        }),
    )
    .await;
    let set = set_lines
        .iter()
        .find(|value| value.get("id") == Some(&json!(3)))
        .expect("thread/goal/set response");
    let updated = set_lines
        .iter()
        .find(|value| value.get("method") == Some(&json!(METHOD_THREAD_GOAL_UPDATED)))
        .unwrap_or_else(|| panic!("thread/goal/updated notification: {set_lines:#?}"));
    assert_eq!(updated.pointer("/params/goal"), set.pointer("/result/goal"));
    assert_eq!(set.pointer("/result/goal/tokensUsed"), Some(&json!(0)));
    assert_eq!(set.pointer("/result/goal/timeUsedSeconds"), Some(&json!(0)));
    let created_at = set
        .pointer("/result/goal/createdAt")
        .cloned()
        .expect("goal createdAt");

    let edit = request(
        &server,
        4,
        METHOD_THREAD_GOAL_SET,
        json!({
            "threadId": thread_id,
            "objective": "finish the edited durable goal",
            "status": "blocked",
            "tokenBudget": null
        }),
    )
    .await;
    assert_eq!(
        edit.pointer("/result/goal/objective"),
        Some(&json!("finish the edited durable goal"))
    );
    assert_eq!(edit.pointer("/result/goal/status"), Some(&json!("blocked")));
    assert_eq!(edit.pointer("/result/goal/tokenBudget"), Some(&Value::Null));
    assert_eq!(edit.pointer("/result/goal/createdAt"), Some(&created_at));

    drop(server);
    let restarted = AppServer::with_runtime(
        RuntimeCore::default().with_projection_store(Arc::new(
            ProjectionStore::initialize(&projection_path)
                .expect("reopen thread goal projection store"),
        )),
    );
    initialize_server(&restarted).await;

    let read = request(
        &restarted,
        5,
        METHOD_THREAD_GOAL_GET,
        json!({ "threadId": thread_id }),
    )
    .await;
    assert_eq!(
        read.pointer("/result/goal/objective"),
        Some(&json!("finish the edited durable goal"))
    );
    assert_eq!(read.pointer("/result/goal/status"), Some(&json!("blocked")));

    let clear_lines = request_lines(
        &restarted,
        6,
        METHOD_THREAD_GOAL_CLEAR,
        json!({ "threadId": thread_id }),
    )
    .await;
    assert_eq!(
        clear_lines
            .iter()
            .find(|value| value.get("id") == Some(&json!(6)))
            .and_then(|value| value.pointer("/result/cleared")),
        Some(&json!(true))
    );
    assert!(clear_lines
        .iter()
        .any(|value| value.get("method") == Some(&json!(METHOD_THREAD_GOAL_CLEARED))));

    let read_after_clear = request(
        &restarted,
        7,
        METHOD_THREAD_GOAL_GET,
        json!({ "threadId": thread_id }),
    )
    .await;
    assert_eq!(read_after_clear.pointer("/result/goal"), Some(&Value::Null));

    let clear_again = request_lines(
        &restarted,
        8,
        METHOD_THREAD_GOAL_CLEAR,
        json!({ "threadId": thread_id }),
    )
    .await;
    assert_eq!(
        clear_again
            .iter()
            .find(|value| value.get("id") == Some(&json!(8)))
            .and_then(|value| value.pointer("/result/cleared")),
        Some(&json!(false))
    );
    assert!(clear_again
        .iter()
        .all(|value| value.get("method") != Some(&json!(METHOD_THREAD_GOAL_CLEARED))));
}

#[tokio::test]
async fn thread_resume_rehydrates_the_same_identity_and_bootstraps_turns_page() {
    let (_temp, server) = test_server();
    initialize_server(&server).await;

    let started = request(
        &server,
        2,
        METHOD_THREAD_START,
        json!({
            "model": "gpt-5.4",
            "modelProvider": "openai",
            "cwd": "/tmp/lime-thread-resume-v2",
            "historyMode": "legacy"
        }),
    )
    .await;
    let thread_id = started
        .pointer("/result/thread/id")
        .and_then(Value::as_str)
        .expect("thread/start id");
    let session_id = started
        .pointer("/result/thread/sessionId")
        .and_then(Value::as_str)
        .expect("thread/start session id");

    let resumed_lines = request_lines(
        &server,
        3,
        METHOD_THREAD_RESUME,
        json!({
            "threadId": thread_id,
            "path": "",
            "excludeTurns": true,
            "initialTurnsPage": {
                "limit": 10,
                "sortDirection": "desc",
                "itemsView": "summary"
            }
        }),
    )
    .await;
    let resumed = resumed_lines
        .iter()
        .find(|value| value.get("id") == Some(&json!(3)))
        .expect("thread/resume response");
    assert_eq!(
        resumed.pointer("/result/thread/id"),
        Some(&json!(thread_id))
    );
    assert_eq!(
        resumed.pointer("/result/thread/sessionId"),
        Some(&json!(session_id))
    );
    assert_eq!(resumed.pointer("/result/model"), Some(&json!("gpt-5.4")));
    assert_eq!(
        resumed.pointer("/result/modelProvider"),
        Some(&json!("openai"))
    );
    assert_eq!(
        resumed.pointer("/result/cwd"),
        Some(&json!("/tmp/lime-thread-resume-v2"))
    );
    assert_eq!(resumed.pointer("/result/thread/turns"), Some(&json!([])));
    assert!(
        resumed_lines
            .iter()
            .all(|value| value.get("method") != Some(&json!("thread/started"))),
        "thread/resume must not emit thread/started: {resumed_lines:#?}"
    );

    let turns = request(
        &server,
        4,
        METHOD_THREAD_TURNS_LIST,
        json!({
            "threadId": thread_id,
            "limit": 10,
            "sortDirection": "desc",
            "itemsView": "summary"
        }),
    )
    .await;
    assert_eq!(
        resumed.pointer("/result/initialTurnsPage"),
        turns.get("result")
    );
}

#[tokio::test]
async fn thread_resume_projects_the_loaded_actor_active_turn() {
    let temp = TempDir::new().expect("loaded thread resume temp dir");
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let projection_store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("loaded thread resume projection store"),
    );
    let runtime = RuntimeCore::with_backend(Arc::new(BlockingTurnBackend {
        started: Arc::clone(&started),
        release: Arc::clone(&release),
    }))
    .with_projection_store(projection_store);
    let server = AppServer::with_runtime(runtime);
    initialize_server(&server).await;

    let thread_start = request(
        &server,
        2,
        METHOD_THREAD_START,
        json!({
            "model": "fixture-model",
            "modelProvider": "fixture-provider",
            "historyMode": "legacy"
        }),
    )
    .await;
    let thread_id = thread_start
        .pointer("/result/thread/id")
        .and_then(Value::as_str)
        .expect("thread/start id")
        .to_string();
    let session_id = thread_start
        .pointer("/result/thread/sessionId")
        .and_then(Value::as_str)
        .expect("thread/start session id")
        .to_string();

    let turn_start = request(
        &server,
        3,
        METHOD_TURN_START,
        json!({
            "threadId": thread_id,
            "input": [{"type": "text", "text": "hold the active turn"}],
            "model": "fixture-model",
            "approvalPolicy": "never",
            "sandboxPolicy": "workspace-write"
        }),
    )
    .await;
    timeout(Duration::from_secs(2), started.notified())
        .await
        .expect("backend must hold an active turn");
    let turn_id = turn_start
        .pointer("/result/turn/id")
        .and_then(Value::as_str)
        .expect("turn/start id")
        .to_string();

    let resumed = request(
        &server,
        4,
        METHOD_THREAD_RESUME,
        json!({"threadId": thread_id}),
    )
    .await;
    assert_eq!(
        resumed.pointer("/result/thread/id"),
        Some(&json!(thread_id))
    );
    assert_eq!(
        resumed.pointer("/result/thread/sessionId"),
        Some(&json!(session_id))
    );
    assert_eq!(
        resumed.pointer("/result/thread/status/type"),
        Some(&json!("active"))
    );
    assert_eq!(
        resumed.pointer("/result/thread/turns/0/id"),
        Some(&json!(turn_id))
    );
    assert_eq!(
        resumed.pointer("/result/thread/turns/0/status"),
        Some(&json!("inProgress"))
    );

    release.notify_one();
    timeout(Duration::from_secs(2), async {
        let mut request_id = 5;
        loop {
            let read = request(
                &server,
                request_id,
                METHOD_THREAD_READ,
                json!({"threadId": thread_id, "includeTurns": true}),
            )
            .await;
            if read.pointer("/result/thread/turns/0/status") == Some(&json!("completed")) {
                break;
            }
            request_id += 1;
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("released turn must reach canonical completed state");
}

#[tokio::test]
async fn thread_resume_enforces_paginated_history_constraints() {
    let (_temp, server) = test_server();
    initialize_server(&server).await;
    let started = request(
        &server,
        2,
        METHOD_THREAD_START,
        json!({
            "model": "gpt-5.4",
            "modelProvider": "openai",
            "historyMode": "paginated"
        }),
    )
    .await;
    let thread_id = started
        .pointer("/result/thread/id")
        .and_then(Value::as_str)
        .expect("thread/start id");

    let full = request_raw(
        &server,
        3,
        METHOD_THREAD_RESUME,
        json!({"threadId": thread_id}),
    )
    .await;
    assert_eq!(
        full.pointer("/error/code"),
        Some(&json!(error_codes::INVALID_REQUEST))
    );

    let initial_page = request_raw(
        &server,
        4,
        METHOD_THREAD_RESUME,
        json!({
            "threadId": thread_id,
            "excludeTurns": true,
            "initialTurnsPage": {}
        }),
    )
    .await;
    assert_eq!(
        initial_page.pointer("/error/code"),
        Some(&json!(error_codes::INVALID_REQUEST))
    );

    let metadata_only = request(
        &server,
        5,
        METHOD_THREAD_RESUME,
        json!({"threadId": thread_id, "excludeTurns": true}),
    )
    .await;
    assert_eq!(
        metadata_only.pointer("/result/thread/historyMode"),
        Some(&json!("paginated"))
    );
}

#[tokio::test]
async fn thread_resume_rejects_legacy_shape_and_unimplemented_sources_or_overrides() {
    let (_temp, server) = test_server();
    initialize_server(&server).await;
    let started = request(
        &server,
        2,
        METHOD_THREAD_START,
        json!({"model": "gpt-5.4", "modelProvider": "openai"}),
    )
    .await;
    let thread_id = started
        .pointer("/result/thread/id")
        .and_then(Value::as_str)
        .expect("thread/start id");

    let legacy = request_raw(
        &server,
        3,
        METHOD_THREAD_RESUME,
        json!({"sessionId": "session-1"}),
    )
    .await;
    assert_eq!(
        legacy.pointer("/error/code"),
        Some(&json!(error_codes::INVALID_PARAMS))
    );

    for (id, params) in [
        (4, json!({"threadId": thread_id, "history": []})),
        (
            5,
            json!({"threadId": thread_id, "history": [{"type": "message"}]}),
        ),
        (6, json!({"threadId": thread_id, "path": "/tmp/rollout"})),
        (7, json!({"threadId": thread_id, "model": "gpt-5.4-mini"})),
    ] {
        let response = request_raw(&server, id, METHOD_THREAD_RESUME, params).await;
        assert_eq!(
            response.pointer("/error/code"),
            Some(&json!(error_codes::INVALID_REQUEST)),
            "request {id} must fail closed: {response:#?}"
        );
        assert!(response.get("result").is_none());
    }
}

#[tokio::test]
async fn thread_archive_moves_the_dated_rollout_and_unarchive_restores_it() {
    let (temp, server) = test_server();
    initialize_server(&server).await;
    let started = request(
        &server,
        2,
        METHOD_THREAD_START,
        json!({
            "model": "gpt-5.4",
            "modelProvider": "openai",
            "historyMode": "paginated"
        }),
    )
    .await;
    let thread_id = started
        .pointer("/result/thread/id")
        .and_then(Value::as_str)
        .expect("thread/start id")
        .to_string();
    let agent_root = temp.path().join("agent-root");
    let active_files = rollout_files(&agent_root.join("sessions"));
    assert_eq!(active_files.len(), 1, "one dated rollout must be created");
    let active_path = active_files[0].clone();
    assert_eq!(
        active_path
            .strip_prefix(agent_root.join("sessions"))
            .expect("dated rollout relative path")
            .components()
            .count(),
        4,
        "rollout path must be YYYY/MM/DD/<file>"
    );

    let archive_lines = request_lines(
        &server,
        3,
        METHOD_THREAD_ARCHIVE,
        json!({"threadId": thread_id}),
    )
    .await;
    assert_eq!(
        archive_lines
            .iter()
            .find(|line| line.get("id") == Some(&json!(3)))
            .and_then(|line| line.get("result")),
        Some(&json!({}))
    );
    assert_eq!(
        archive_lines
            .iter()
            .find(|line| line.get("method") == Some(&json!("thread/archived")))
            .and_then(|line| line.pointer("/params/threadId")),
        Some(&json!(thread_id))
    );
    assert!(!active_path.exists());
    let archived_files = rollout_files(&agent_root.join("archived_sessions"));
    assert_eq!(archived_files.len(), 1);
    assert_eq!(archived_files[0].file_name(), active_path.file_name());

    let active = request(&server, 4, METHOD_THREAD_LIST, json!({"archived": false})).await;
    assert_eq!(active.pointer("/result/data"), Some(&json!([])));
    let archived = request(&server, 5, METHOD_THREAD_LIST, json!({"archived": true})).await;
    assert_eq!(
        archived.pointer("/result/data/0/id"),
        Some(&json!(thread_id))
    );

    let duplicate_archive = request_lines(
        &server,
        6,
        METHOD_THREAD_ARCHIVE,
        json!({"threadId": thread_id}),
    )
    .await;
    assert!(duplicate_archive
        .iter()
        .all(|line| line.get("method") != Some(&json!("thread/archived"))));

    let unarchive_lines = request_lines(
        &server,
        7,
        METHOD_THREAD_UNARCHIVE,
        json!({"threadId": thread_id}),
    )
    .await;
    assert_eq!(
        unarchive_lines
            .iter()
            .find(|line| line.get("id") == Some(&json!(7)))
            .and_then(|line| line.pointer("/result/thread/id")),
        Some(&json!(thread_id))
    );
    assert_eq!(
        unarchive_lines
            .iter()
            .find(|line| line.get("method") == Some(&json!("thread/unarchived")))
            .and_then(|line| line.pointer("/params/threadId")),
        Some(&json!(thread_id))
    );
    assert!(active_path.exists());
    assert!(rollout_files(&agent_root.join("archived_sessions")).is_empty());
}

#[tokio::test]
async fn retired_agent_session_start_is_not_a_production_method() {
    let (_temp, server) = test_server();
    initialize_server(&server).await;

    let response = request_raw(
        &server,
        2,
        "agentSession/start",
        json!({
            "appId": "agent-chat"
        }),
    )
    .await;

    assert_eq!(
        response.pointer("/error/code"),
        Some(&json!(error_codes::METHOD_NOT_FOUND))
    );
    assert!(response.get("result").is_none());
}

async fn start_thread_turn(server: &AppServer, input: &str) -> (String, String, String) {
    let thread = request(
        server,
        2,
        METHOD_THREAD_START,
        json!({
            "model": "fixture-model",
            "modelProvider": "fixture-provider"
        }),
    )
    .await;
    let thread_id = thread
        .pointer("/result/thread/id")
        .and_then(Value::as_str)
        .expect("thread id")
        .to_string();
    let session_id = thread
        .pointer("/result/thread/sessionId")
        .and_then(Value::as_str)
        .expect("session id")
        .to_string();
    let turn = request(
        server,
        3,
        METHOD_TURN_START,
        json!({
            "threadId": thread_id,
            "input": [{"type": "text", "text": input}]
        }),
    )
    .await;
    let turn_id = turn
        .pointer("/result/turn/id")
        .and_then(Value::as_str)
        .expect("turn id")
        .to_string();
    (session_id, thread_id, turn_id)
}

fn canonical_mcp_item(call_id: &str, status: &str) -> Value {
    canonical_tool_item(
        call_id,
        status,
        "mcpToolCall",
        json!({
            "type": "mcpToolCall",
            "call_id": call_id,
            "server_name": "docs",
            "tool_name": "search",
            "arguments": [],
            "output": (status != "inProgress").then(|| json!({"text": "done"}))
        }),
    )
}

fn canonical_dynamic_tool_item(call_id: &str, status: &str) -> Value {
    canonical_tool_item(
        call_id,
        status,
        "tool",
        json!({
            "type": "tool",
            "call_id": call_id,
            "name": "search",
            "arguments": [],
            "output": (status != "inProgress").then(|| json!({"text": "done"}))
        }),
    )
}

fn canonical_tool_item(call_id: &str, status: &str, kind: &str, payload: Value) -> Value {
    json!({
        "item": {
            "sessionId": "session-fixture",
            "threadId": "thread-fixture",
            "turnId": "turn-fixture",
            "itemId": format!("item_{call_id}"),
            "sequence": 1,
            "ordinal": 1,
            "createdAtMs": 1,
            "updatedAtMs": 2,
            "completedAtMs": (status != "inProgress").then_some(2),
            "kind": kind,
            "status": status,
            "payload": payload,
            "metadata": {}
        }
    })
}

fn mcp_progress_payload(call_id: &str, message: &str, notification_kind: Option<&str>) -> Value {
    json!({
        "tool_id": call_id,
        "serverName": "docs",
        "toolName": "search",
        "progress": {
            "message": message,
            "metadata": notification_kind.map(|kind| json!({
                "notification_kind": kind,
                "server_name": "docs",
                "tool_name": "search",
                "runtime_tool_name": "mcp__docs__search"
            })).unwrap_or_else(|| json!({}))
        }
    })
}

fn test_server() -> (TempDir, AppServer) {
    let temp = TempDir::new().expect("thread v2 temp dir");
    let agent_root = temp.path().join("agent-root");
    let projection_store = Arc::new(
        ProjectionStore::initialize_with_agent_root(
            agent_root.join("runtime").join("projection.sqlite"),
            &agent_root,
        )
        .expect("thread v2 projection store"),
    );
    let runtime =
        RuntimeCore::with_backend(Arc::new(MockBackend)).with_projection_store(projection_store);
    (temp, AppServer::with_runtime(runtime))
}

fn rollout_files(root: &Path) -> Vec<PathBuf> {
    if !root.exists() {
        return Vec::new();
    }
    let mut pending = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(path) = pending.pop() {
        for entry in fs::read_dir(&path).expect("read rollout directory") {
            let path = entry.expect("read rollout entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().and_then(|value| value.to_str()) == Some("jsonl") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

async fn initialize_server(server: &AppServer) {
    let response = request(
        server,
        1,
        METHOD_INITIALIZE,
        json!({
            "clientInfo": {
                "name": "thread-v2-jsonrpc-test",
                "version": "1.0.0"
            }
        }),
    )
    .await;
    assert_eq!(
        response.pointer("/result/serverInfo/protocolVersion"),
        Some(&json!(PROTOCOL_VERSION))
    );

    let lines = server
        .handle_json_line(
            &json!({
                "jsonrpc": "2.0",
                "method": METHOD_INITIALIZED,
                "params": {}
            })
            .to_string(),
        )
        .await
        .expect("handle initialized notification");
    assert!(lines.is_empty());
}

async fn request(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    let response = request_raw(server, id, method, params).await;
    if let Some(error) = response.get("error") {
        panic!("{method} failed: {error}");
    }
    response
}

async fn request_raw(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    let lines = request_lines(server, id, method, params).await;
    lines
        .into_iter()
        .find(|value| value.get("id") == Some(&json!(id)))
        .expect("JSON-RPC response")
}

async fn request_lines(server: &AppServer, id: u64, method: &str, params: Value) -> Vec<Value> {
    let lines = server
        .handle_json_line(
            &json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": method,
                "params": params
            })
            .to_string(),
        )
        .await
        .expect("handle JSON-RPC request");
    let values = lines
        .iter()
        .map(|line| serde_json::from_str(line).expect("decode JSON-RPC response"))
        .collect::<Vec<Value>>();
    assert_eq!(
        values
            .iter()
            .filter(|value| value.get("id") == Some(&json!(id)))
            .count(),
        1,
        "{method} must return exactly one response"
    );
    values
}
