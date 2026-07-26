use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use app_server::{
    ActionRespondRequest, AppServer, CancelExecutionRequest, EventLogWriter, ExecutionBackend,
    ExecutionRequest, MockBackend, ProjectionStore, RuntimeCore, RuntimeCoreError, RuntimeEvent,
    RuntimeEventSink, RuntimeHostContext,
};
use app_server_protocol::protocol::v2::{
    METHOD_THREAD_MEMORY_MODE_SET, METHOD_THREAD_SETTINGS_UPDATE,
};
use app_server_protocol::{
    error_codes, AgentInput, AgentSessionStartParams, AgentSessionTurnStartParams,
    BusinessObjectRef, METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_ITEMS_LIST,
    METHOD_THREAD_READ, METHOD_THREAD_RESUME, METHOD_THREAD_START, METHOD_THREAD_TURNS_LIST,
    METHOD_TURN_START, PROTOCOL_VERSION,
};
use async_trait::async_trait;
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::sync::Notify;
use tokio::time::{timeout, Duration};

struct BlockingCaptureBackend {
    requests: Mutex<Vec<ExecutionRequest>>,
    starts: AtomicUsize,
    first_started: Notify,
    release_first: Notify,
}

#[async_trait]
impl ExecutionBackend for BlockingCaptureBackend {
    async fn start_turn(
        &self,
        request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        let index = self.starts.fetch_add(1, Ordering::SeqCst);
        self.requests
            .lock()
            .expect("captured requests mutex poisoned")
            .push(request);
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        if index == 0 {
            self.first_started.notify_one();
            self.release_first.notified().await;
        }
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
async fn settings_update_preserves_the_active_turn_and_only_changes_subsequent_turns() {
    let temp = TempDir::new().expect("turn inheritance temp dir");
    let store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("turn inheritance projection store"),
    );
    let backend = Arc::new(BlockingCaptureBackend {
        requests: Mutex::new(Vec::new()),
        starts: AtomicUsize::new(0),
        first_started: Notify::new(),
        release_first: Notify::new(),
    });
    let core = RuntimeCore::with_backend(backend.clone()).with_projection_store(store);
    core.start_session(AgentSessionStartParams {
        session_id: Some("session-settings-inheritance".to_string()),
        thread_id: Some("thread-settings-inheritance".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: None,
        business_object_ref: Some(BusinessObjectRef {
            kind: "agent.thread".to_string(),
            id: "thread-settings-inheritance".to_string(),
            title: None,
            uri: None,
            metadata: Some(json!({
                "providerSelector": "provider-a",
                "providerName": "provider-a",
                "modelName": "model-a",
                "agentControlRoute": {
                    "schemaVersion": 2,
                    "providerPreference": "provider-a",
                    "modelPreference": "model-a",
                    "providerConfig": {
                        "providerId": "provider-a",
                        "providerName": "provider-a",
                        "modelName": "model-a"
                    },
                    "routeProtocol": "openai_responses",
                    "authKind": "api_key_ref",
                    "credentialRef": "credential-a",
                    "effectiveGeneration": 1
                },
                "workingDir": "/tmp/settings-a",
                "approvalPolicy": "on-request",
                "sandboxPolicy": "workspace-write",
                "reasoningEffort": "low",
                "serviceTier": "standard"
            })),
        }),
        locale: None,
    })
    .expect("start inheritance session");

    let turn_core = core.clone();
    let first_turn = tokio::spawn(async move {
        turn_core
            .start_turn(
                turn_params("turn-settings-first", "first"),
                RuntimeHostContext::default(),
            )
            .await
    });
    timeout(Duration::from_secs(2), backend.first_started.notified())
        .await
        .expect("first turn must reach backend");

    let settings = core
        .update_thread_settings(
            serde_json::from_value(json!({
                "threadId": "thread-settings-inheritance",
                "model": "model-b",
                "modelProvider": "provider-b",
                "cwd": "/tmp/settings-b",
                "approvalPolicy": "never",
                "sandboxPolicy": "danger-full-access",
                "effort": "high",
                "serviceTier": "priority",
                "collaborationMode": {
                    "mode": "plan",
                    "settings": {
                        "model": "model-b",
                        "reasoning_effort": "high",
                        "developer_instructions": "Plan before editing."
                    }
                },
                "personality": "pragmatic"
            }))
            .expect("settings params"),
        )
        .await
        .expect("update active session settings");
    assert_eq!(settings.model, "model-b");
    assert_eq!(settings.model_provider, "provider-b");
    assert!(
        !first_turn.is_finished(),
        "settings must not interrupt active turn"
    );
    {
        let requests = backend
            .requests
            .lock()
            .expect("captured requests mutex poisoned");
        let first = requests[0]
            .runtime_options
            .as_ref()
            .and_then(|options| options.runtime_request.as_ref())
            .expect("first runtime request");
        assert_eq!(first.model_preference.as_deref(), Some("model-a"));
        assert_eq!(first.provider_preference.as_deref(), Some("provider-a"));
        assert_eq!(first.working_dir.as_deref(), Some("/tmp/settings-a"));
        assert_eq!(first.approval_policy.as_deref(), Some("on-request"));
        assert_eq!(first.sandbox_policy.as_deref(), Some("workspace-write"));
    }

    backend.release_first.notify_one();
    timeout(Duration::from_secs(2), first_turn)
        .await
        .expect("first turn completion timeout")
        .expect("first turn task")
        .expect("first turn result");
    core.start_turn(
        turn_params("turn-settings-second", "second"),
        RuntimeHostContext::default(),
    )
    .await
    .expect("second turn");

    let requests = backend
        .requests
        .lock()
        .expect("captured requests mutex poisoned");
    let second = requests[1]
        .runtime_options
        .as_ref()
        .and_then(|options| options.runtime_request.as_ref())
        .expect("second runtime request");
    assert_eq!(second.model_preference.as_deref(), Some("model-b"));
    assert_eq!(second.provider_preference.as_deref(), Some("provider-b"));
    assert_eq!(second.working_dir.as_deref(), Some("/tmp/settings-b"));
    assert_eq!(second.approval_policy.as_deref(), Some("never"));
    assert_eq!(second.sandbox_policy.as_deref(), Some("danger-full-access"));
    assert_eq!(second.reasoning_effort.as_deref(), Some("high"));
    assert_eq!(
        second.system_prompt.as_deref(),
        Some("Plan before editing.")
    );
    assert_eq!(
        second
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("serviceTier")),
        Some(&json!("priority"))
    );
    assert_eq!(
        second.collaboration_mode.as_ref().map(|mode| mode.mode),
        Some(agent_protocol::ModeKind::Plan)
    );
}

fn turn_params(turn_id: &str, text: &str) -> AgentSessionTurnStartParams {
    AgentSessionTurnStartParams {
        session_id: "session-settings-inheritance".to_string(),
        turn_id: Some(turn_id.to_string()),
        input: AgentInput {
            text: text.to_string(),
            attachments: Vec::new(),
        },
        runtime_options: None,
        queue_if_busy: false,
        skip_pre_submit_resume: false,
    }
}

#[tokio::test]
async fn settings_update_runs_through_the_actor_and_persists_without_items() {
    let temp = TempDir::new().expect("thread settings persistence temp dir");
    let store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("thread settings projection store"),
    );
    let event_log_root = temp.path().join("event-log");
    let server = AppServer::with_runtime(
        RuntimeCore::with_backend(Arc::new(MockBackend))
            .with_projection_store(Arc::clone(&store))
            .with_event_log_writer(Arc::new(
                EventLogWriter::new(&event_log_root).expect("thread settings event log"),
            )),
    );
    initialize_server(&server).await;
    let started = request(
        &server,
        2,
        METHOD_THREAD_START,
        json!({
            "model": "model-a",
            "modelProvider": "provider-a",
            "cwd": "/tmp/thread-control-a",
            "approvalPolicy": "on-request",
            "sandbox": "workspace-write",
            "serviceTier": "priority"
        }),
    )
    .await;
    let thread_id = started
        .pointer("/result/thread/id")
        .and_then(Value::as_str)
        .expect("thread id")
        .to_string();

    let lines = request_lines(
        &server,
        3,
        METHOD_THREAD_SETTINGS_UPDATE,
        json!({
            "threadId": thread_id,
            "cwd": "/tmp/thread-control-b",
            "model": "model-b",
            "modelProvider": "provider-b",
            "serviceTier": null,
            "approvalPolicy": "never",
            "sandboxPolicy": "danger-full-access",
            "effort": "high",
            "summary": "concise",
            "collaborationMode": {
                "mode": "plan",
                "settings": {
                    "model": "model-b",
                    "reasoning_effort": "high",
                    "developer_instructions": "Plan before editing."
                }
            },
            "personality": "pragmatic"
        }),
    )
    .await;
    let response = response_for(&lines, 3);
    assert_eq!(response.pointer("/result"), Some(&json!({})));
    let updated = lines
        .iter()
        .find(|value| value.get("method") == Some(&json!("thread/settings/updated")))
        .unwrap_or_else(|| panic!("missing settings notification: {lines:#?}"));
    assert_eq!(
        updated.pointer("/params/threadSettings/model"),
        Some(&json!("model-b"))
    );
    assert_eq!(
        updated.pointer("/params/threadSettings/modelProvider"),
        Some(&json!("provider-b"))
    );
    assert_eq!(
        updated.pointer("/params/threadSettings/cwd"),
        Some(&json!("/tmp/thread-control-b"))
    );
    assert_eq!(
        updated.pointer("/params/threadSettings/collaborationMode"),
        Some(&json!({
            "mode": "plan",
            "settings": {
                "model": "model-b",
                "reasoning_effort": "high",
                "developer_instructions": "Plan before editing."
            }
        }))
    );
    assert!(
        updated
            .pointer("/params/threadSettings/serviceTier")
            .is_none(),
        "explicit null must clear the persisted service tier"
    );

    let items = request(
        &server,
        4,
        METHOD_THREAD_ITEMS_LIST,
        json!({"threadId": thread_id}),
    )
    .await;
    assert_eq!(items.pointer("/result/data"), Some(&json!([])));

    let seed_turn = request(
        &server,
        40,
        METHOD_TURN_START,
        json!({
            "threadId": thread_id,
            "input": [{"type": "text", "text": "persist the switched route"}]
        }),
    )
    .await;
    let seed_turn_id = seed_turn
        .pointer("/result/turn/id")
        .and_then(Value::as_str)
        .expect("seed turn id");
    wait_for_terminal_turn(&server, &thread_id, seed_turn_id).await;

    let restarted_backend = Arc::new(BlockingCaptureBackend {
        requests: Mutex::new(Vec::new()),
        starts: AtomicUsize::new(0),
        first_started: Notify::new(),
        release_first: Notify::new(),
    });
    restarted_backend.release_first.notify_one();
    let restarted = AppServer::with_runtime(
        RuntimeCore::with_backend(restarted_backend.clone())
            .with_projection_store(store)
            .with_event_log_writer(Arc::new(
                EventLogWriter::new(&event_log_root).expect("restarted thread settings event log"),
            )),
    );
    initialize_server(&restarted).await;
    let read = request(
        &restarted,
        5,
        METHOD_THREAD_READ,
        json!({"threadId": thread_id}),
    )
    .await;
    assert_eq!(
        read.pointer("/result/thread/extra/modelName"),
        Some(&json!("model-b"))
    );
    assert_eq!(
        read.pointer("/result/thread/extra/providerSelector"),
        Some(&json!("provider-b"))
    );
    assert_eq!(
        read.pointer("/result/thread/extra/providerName"),
        Some(&json!("provider-b"))
    );
    assert_eq!(
        read.pointer("/result/thread/extra/workingDir"),
        Some(&json!("/tmp/thread-control-b"))
    );
    assert_eq!(
        read.pointer("/result/thread/extra/reasoningEffort"),
        Some(&json!("high"))
    );
    assert_eq!(
        read.pointer("/result/thread/extra/collaborationMode/settings/model"),
        Some(&json!("model-b"))
    );
    assert!(read.pointer("/result/thread/extra/serviceTier").is_none());

    request(
        &restarted,
        6,
        METHOD_THREAD_RESUME,
        json!({"threadId": thread_id}),
    )
    .await;
    request(
        &restarted,
        7,
        METHOD_TURN_START,
        json!({
            "threadId": thread_id,
            "input": [{"type": "text", "text": "use the persisted route"}]
        }),
    )
    .await;
    timeout(
        Duration::from_secs(2),
        restarted_backend.first_started.notified(),
    )
    .await
    .expect("restarted turn must reach backend");
    let requests = restarted_backend
        .requests
        .lock()
        .expect("restarted requests mutex poisoned");
    let runtime_request = requests[0]
        .runtime_options
        .as_ref()
        .and_then(|options| options.runtime_request.as_ref())
        .expect("restarted runtime request");
    assert_eq!(
        runtime_request.provider_preference.as_deref(),
        Some("provider-b")
    );
    assert_eq!(runtime_request.model_preference.as_deref(), Some("model-b"));
}

#[tokio::test]
async fn memory_mode_is_typed_actor_serialized_and_survives_restart() {
    let (_temp, store, server) = test_server();
    initialize_server(&server).await;
    let started = request(
        &server,
        2,
        METHOD_THREAD_START,
        json!({"model": "model-a", "modelProvider": "provider-a"}),
    )
    .await;
    let thread_id = started
        .pointer("/result/thread/id")
        .and_then(Value::as_str)
        .expect("thread id")
        .to_string();

    let response = request(
        &server,
        3,
        METHOD_THREAD_MEMORY_MODE_SET,
        json!({"threadId": thread_id, "mode": "disabled"}),
    )
    .await;
    assert_eq!(response.pointer("/result"), Some(&json!({})));
    let turns = request(
        &server,
        30,
        METHOD_THREAD_TURNS_LIST,
        json!({"threadId": thread_id}),
    )
    .await;
    assert_eq!(turns.pointer("/result/data"), Some(&json!([])));

    let restarted = AppServer::with_runtime(
        RuntimeCore::with_backend(Arc::new(MockBackend)).with_projection_store(store),
    );
    initialize_server(&restarted).await;
    let read = request(
        &restarted,
        4,
        METHOD_THREAD_READ,
        json!({"threadId": thread_id}),
    )
    .await;
    assert_eq!(
        read.pointer("/result/thread/extra/memoryMode"),
        Some(&json!("disabled"))
    );
}

#[tokio::test]
async fn rollout_rebuild_restores_thread_settings_and_memory_mode() {
    let temp = TempDir::new().expect("rollout rebuild temp dir");
    let agent_root = temp.path().join("agent-root");
    let source_store = Arc::new(
        ProjectionStore::initialize_with_agent_root(
            agent_root.join("runtime").join("source.sqlite"),
            &agent_root,
        )
        .expect("source rollout store"),
    );
    let source = AppServer::with_runtime(
        RuntimeCore::with_backend(Arc::new(MockBackend))
            .with_projection_store(Arc::clone(&source_store)),
    );
    initialize_server(&source).await;
    let started = request(
        &source,
        2,
        METHOD_THREAD_START,
        json!({"model": "model-a", "modelProvider": "provider-a"}),
    )
    .await;
    let thread_id = started
        .pointer("/result/thread/id")
        .and_then(Value::as_str)
        .expect("thread id")
        .to_string();
    request(
        &source,
        3,
        METHOD_THREAD_SETTINGS_UPDATE,
        json!({
            "threadId": thread_id,
            "model": "model-b",
            "cwd": "/tmp/rollout-rebuild",
            "serviceTier": "priority"
        }),
    )
    .await;
    request(
        &source,
        4,
        METHOD_THREAD_MEMORY_MODE_SET,
        json!({"threadId": thread_id, "mode": "disabled"}),
    )
    .await;
    drop(source);
    drop(source_store);

    let rebuilt_store = Arc::new(
        ProjectionStore::initialize_with_agent_root(
            agent_root.join("runtime").join("rebuilt.sqlite"),
            &agent_root,
        )
        .expect("rebuild projection from rollout"),
    );
    let rebuilt = AppServer::with_runtime(
        RuntimeCore::with_backend(Arc::new(MockBackend)).with_projection_store(rebuilt_store),
    );
    initialize_server(&rebuilt).await;
    let read = request(
        &rebuilt,
        5,
        METHOD_THREAD_READ,
        json!({"threadId": thread_id}),
    )
    .await;
    assert_eq!(
        read.pointer("/result/thread/extra/modelName"),
        Some(&json!("model-b"))
    );
    assert_eq!(
        read.pointer("/result/thread/extra/workingDir"),
        Some(&json!("/tmp/rollout-rebuild"))
    );
    assert_eq!(
        read.pointer("/result/thread/extra/serviceTier"),
        Some(&json!("priority"))
    );
    assert_eq!(
        read.pointer("/result/thread/extra/memoryMode"),
        Some(&json!("disabled"))
    );
}

#[tokio::test]
async fn thread_control_requests_fail_closed_before_actor_mutation() {
    let (_temp, _store, server) = test_server();
    initialize_server(&server).await;
    let started = request(
        &server,
        2,
        METHOD_THREAD_START,
        json!({"model": "model-a", "modelProvider": "provider-a"}),
    )
    .await;
    let thread_id = started
        .pointer("/result/thread/id")
        .and_then(Value::as_str)
        .expect("thread id");

    for (id, method, params, expected_code) in [
        (
            3,
            METHOD_THREAD_SETTINGS_UPDATE,
            json!({"threadId": thread_id}),
            error_codes::INVALID_REQUEST,
        ),
        (
            4,
            METHOD_THREAD_SETTINGS_UPDATE,
            json!({
                "threadId": thread_id,
                "sandboxPolicy": "workspace-write",
                "permissions": "workspace"
            }),
            error_codes::INVALID_REQUEST,
        ),
        (
            5,
            METHOD_THREAD_SETTINGS_UPDATE,
            json!({"threadId": "missing-thread", "model": "model-b"}),
            error_codes::RUNTIME_ERROR,
        ),
        (
            7,
            METHOD_THREAD_SETTINGS_UPDATE,
            json!({"threadId": thread_id, "modelProvider": "provider-b"}),
            error_codes::INVALID_REQUEST,
        ),
        (
            8,
            METHOD_THREAD_SETTINGS_UPDATE,
            json!({
                "threadId": thread_id,
                "model": "model-b",
                "modelProvider": " "
            }),
            error_codes::INVALID_REQUEST,
        ),
        (
            6,
            METHOD_THREAD_MEMORY_MODE_SET,
            json!({"threadId": thread_id, "mode": "automatic"}),
            error_codes::INVALID_PARAMS,
        ),
    ] {
        let response = request_raw(&server, id, method, params).await;
        assert_eq!(
            response.pointer("/error/code"),
            Some(&json!(expected_code)),
            "request {id} must fail closed: {response:#?}"
        );
        assert!(response.get("result").is_none());
    }
}

fn test_server() -> (TempDir, Arc<ProjectionStore>, AppServer) {
    let temp = TempDir::new().expect("thread control temp dir");
    let store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("thread control projection store"),
    );
    let runtime =
        RuntimeCore::with_backend(Arc::new(MockBackend)).with_projection_store(Arc::clone(&store));
    (temp, store, AppServer::with_runtime(runtime))
}

async fn initialize_server(server: &AppServer) {
    let response = request(
        server,
        1,
        METHOD_INITIALIZE,
        json!({
            "clientInfo": {
                "name": "thread-control-jsonrpc-test",
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
    response_for(&lines, id).clone()
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

fn response_for(lines: &[Value], id: u64) -> &Value {
    lines
        .iter()
        .find(|value| value.get("id") == Some(&json!(id)))
        .expect("JSON-RPC response")
}

async fn wait_for_terminal_turn(server: &AppServer, thread_id: &str, turn_id: &str) {
    timeout(Duration::from_secs(2), async {
        loop {
            let turns = request(
                server,
                90,
                METHOD_THREAD_TURNS_LIST,
                json!({"threadId": thread_id}),
            )
            .await;
            let terminal = turns
                .pointer("/result/data")
                .and_then(Value::as_array)
                .and_then(|turns| {
                    turns
                        .iter()
                        .find(|turn| turn.pointer("/id").and_then(Value::as_str) == Some(turn_id))
                })
                .and_then(|turn| turn.pointer("/status").and_then(Value::as_str))
                .is_some_and(|status| {
                    matches!(status, "completed" | "failed" | "interrupted" | "canceled")
                });
            if terminal {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("turn {turn_id} did not become terminal"));
}
