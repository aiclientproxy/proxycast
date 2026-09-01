use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_protocol::{
    ImageDetail, ItemId, ItemStatus, SessionId, ThreadId, ThreadItem, ThreadItemPayload,
    ToolArgument, ToolOutput, TurnId,
};
use app_server::{
    ActionRespondRequest, AppServer, CancelExecutionRequest, EventLogWriter, ExecutionBackend,
    ExecutionRequest, ProjectionStore, ProviderTurnHistory, RuntimeCore, RuntimeCoreError,
    RuntimeEvent, RuntimeEventSink, SidecarStore,
};
use app_server_protocol::protocol::v2::{
    METHOD_THREAD_DELETE, METHOD_THREAD_FORK, METHOD_THREAD_GOAL_GET, METHOD_THREAD_GOAL_SET,
    METHOD_THREAD_LIST, METHOD_THREAD_TURNS_LIST,
};
use app_server_protocol::{
    error_codes, METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_READ, METHOD_THREAD_RESUME,
    METHOD_THREAD_START, METHOD_TURN_START,
};
use async_trait::async_trait;
use model_provider::current_client::{
    CurrentProviderContent, CurrentProviderMessage, CurrentProviderRole,
};
use rusqlite::{params, Connection};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::sync::Notify;
use tokio::time::timeout;

struct ForkBackend {
    calls: AtomicUsize,
    first_completed: Notify,
    explicit_started: Notify,
    release_explicit: Notify,
}

impl ForkBackend {
    fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
            first_completed: Notify::new(),
            explicit_started: Notify::new(),
            release_explicit: Notify::new(),
        }
    }
}

#[async_trait]
impl ExecutionBackend for ForkBackend {
    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        match call {
            0 => {
                sink.emit(RuntimeEvent::new("turn.completed", json!({})))?;
                self.first_completed.notify_one();
                Ok(())
            }
            1 => {
                self.explicit_started.notify_one();
                self.release_explicit.notified().await;
                sink.emit(RuntimeEvent::new("turn.completed", json!({})))
            }
            _ => Err(RuntimeCoreError::Backend(format!(
                "unexpected automatic fork continuation call {}",
                call + 1
            ))),
        }
    }

    async fn cancel_turn(
        &self,
        _request: CancelExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.release_explicit.notify_one();
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

struct ImmediateBackend;

#[async_trait]
impl ExecutionBackend for ImmediateBackend {
    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
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

#[derive(Default)]
struct HistoryCaptureBackend {
    calls: AtomicUsize,
    histories: Mutex<Vec<Vec<CurrentProviderMessage>>>,
}

#[async_trait]
impl ExecutionBackend for HistoryCaptureBackend {
    async fn start_turn(
        &self,
        request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.start_turn_with_provider_history(request, ProviderTurnHistory::default(), sink)
            .await
    }

    async fn start_turn_with_provider_history(
        &self,
        request: ExecutionRequest,
        provider_history: ProviderTurnHistory,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        self.histories
            .lock()
            .expect("fork history capture mutex poisoned")
            .push(provider_history.messages_for_route("fixture-provider", "fixture-model"));
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        if call == 0 {
            sink.emit(RuntimeEvent::new(
                "item.started",
                canonical_tool_payload(&request, ItemStatus::InProgress, None),
            ))?;
            sink.emit(RuntimeEvent::new(
                "item.completed",
                canonical_tool_payload(
                    &request,
                    ItemStatus::Completed,
                    Some(ToolOutput {
                        text: Some("source tool output".to_string()),
                        ..ToolOutput::default()
                    }),
                ),
            ))?;
            sink.emit(RuntimeEvent::new(
                "message.delta",
                json!({"itemId": "source-answer", "text": "source assistant answer"}),
            ))?;
            sink.emit(RuntimeEvent::new(
                "message.completed",
                json!({
                    "itemId": "source-answer",
                    "phase": "final_answer",
                    "status": "completed",
                    "text": "source assistant answer"
                }),
            ))?;
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

fn canonical_tool_payload(
    request: &ExecutionRequest,
    status: ItemStatus,
    output: Option<ToolOutput>,
) -> Value {
    let mut item = ThreadItem::new(
        SessionId::new(request.session.session_id.clone()),
        ThreadId::new(request.session.thread_id.clone()),
        TurnId::new(request.turn.turn_id.clone()),
        1,
        1,
        ThreadItemPayload::Tool {
            call_id: "source-tool-call".to_string(),
            name: "read_file".to_string(),
            arguments: vec![ToolArgument {
                name: "path".to_string(),
                value: "\"README.md\"".to_string(),
            }],
            output,
        },
    );
    item.item_id = ItemId::new("source-tool-item");
    item.status = status;
    json!({"item": item})
}

#[tokio::test]
async fn thread_fork_preserves_paginated_history_boundary_across_restart() {
    const SOURCE_TITLE: &str = "Paginated fork source";

    let temp = TempDir::new().expect("paginated fork temp dir");
    let projection_path = temp.path().join("projection.sqlite");
    let runtime = || {
        RuntimeCore::with_backend(Arc::new(ImmediateBackend)).with_projection_store(Arc::new(
            ProjectionStore::initialize(&projection_path).expect("paginated fork store"),
        ))
    };
    let server = AppServer::with_runtime(runtime());
    initialize(&server, 1).await;

    let started = request(
        &server,
        2,
        METHOD_THREAD_START,
        json!({
            "model": "fixture-model",
            "modelProvider": "fixture-provider",
            "cwd": temp.path(),
            "historyMode": "paginated",
            "serviceName": SOURCE_TITLE
        }),
    )
    .await;
    let source_thread_id = required_string(&started, "/result/thread/id");
    let source_session_id = required_string(&started, "/result/thread/sessionId");
    assert_eq!(
        started.pointer("/result/thread/name"),
        Some(&json!(SOURCE_TITLE))
    );
    for (id, text, expected_turns) in [
        (3, "first paginated turn", 1),
        (4, "second paginated turn", 2),
    ] {
        request(
            &server,
            id,
            METHOD_TURN_START,
            json!({
                "threadId": source_thread_id,
                "input": [{"type": "text", "text": text}],
                "model": "fixture-model",
                "approvalPolicy": "never",
                "sandboxPolicy": "workspace-write"
            }),
        )
        .await;
        wait_for_completed_turn_count(&server, &source_thread_id, expected_turns).await;
    }
    let source_turns = request(
        &server,
        5,
        METHOD_THREAD_TURNS_LIST,
        json!({"threadId": source_thread_id, "sortDirection": "asc"}),
    )
    .await;
    let first_turn_id = required_string(&source_turns, "/result/data/0/id");

    let forked = request(
        &server,
        6,
        METHOD_THREAD_FORK,
        json!({"threadId": source_thread_id, "lastTurnId": first_turn_id}),
    )
    .await;
    let target_thread_id = required_string(&forked, "/result/thread/id");
    let target_session_id = required_string(&forked, "/result/thread/sessionId");
    assert_ne!(target_thread_id, source_thread_id);
    assert_ne!(target_session_id, source_session_id);
    assert_eq!(
        forked.pointer("/result/thread/historyMode"),
        Some(&json!("paginated"))
    );
    assert_eq!(
        forked.pointer("/result/thread/forkedFromId"),
        Some(&json!(source_thread_id))
    );
    assert_eq!(
        forked.pointer("/result/thread/name"),
        Some(&json!(SOURCE_TITLE))
    );
    assert_eq!(turn_count(&forked), 1);

    let target_read = request(
        &server,
        7,
        METHOD_THREAD_READ,
        json!({"threadId": target_thread_id, "includeTurns": false}),
    )
    .await;
    assert_eq!(
        target_read.pointer("/result/thread/name"),
        Some(&json!(SOURCE_TITLE))
    );
    let listed = request(&server, 8, METHOD_THREAD_LIST, json!({"archived": false})).await;
    let listed_threads = listed["result"]["data"]
        .as_array()
        .expect("thread/list data");
    assert_eq!(
        listed_threads
            .iter()
            .filter(|thread| {
                thread["id"] == source_thread_id || thread["id"] == target_thread_id
            })
            .filter(|thread| thread["name"] == SOURCE_TITLE)
            .count(),
        2
    );

    let target_turns = request(
        &server,
        9,
        METHOD_THREAD_TURNS_LIST,
        json!({"threadId": target_thread_id, "sortDirection": "asc"}),
    )
    .await;
    assert_eq!(
        target_turns["result"]["data"].as_array().map(Vec::len),
        Some(1)
    );
    assert_eq!(
        target_turns.pointer("/result/data/0/id"),
        Some(&json!(first_turn_id))
    );
    drop(server);

    let restarted = AppServer::with_runtime(runtime());
    initialize(&restarted, 10).await;
    let resumed = request(
        &restarted,
        11,
        METHOD_THREAD_RESUME,
        json!({"threadId": target_thread_id, "excludeTurns": true}),
    )
    .await;
    assert_eq!(
        resumed.pointer("/result/thread/id"),
        Some(&json!(target_thread_id))
    );
    assert_eq!(
        resumed.pointer("/result/thread/forkedFromId"),
        Some(&json!(source_thread_id))
    );
    assert_eq!(
        resumed.pointer("/result/thread/name"),
        Some(&json!(SOURCE_TITLE))
    );
    let cold_turns = request(
        &restarted,
        12,
        METHOD_THREAD_TURNS_LIST,
        json!({"threadId": target_thread_id, "sortDirection": "asc"}),
    )
    .await;
    assert_eq!(
        cold_turns["result"]["data"].as_array().map(Vec::len),
        Some(1)
    );
    assert_eq!(
        cold_turns.pointer("/result/data/0/id"),
        Some(&json!(first_turn_id))
    );
    let cold_read = request(
        &restarted,
        13,
        METHOD_THREAD_READ,
        json!({"threadId": target_thread_id, "includeTurns": false}),
    )
    .await;
    assert_eq!(
        cold_read.pointer("/result/thread/name"),
        Some(&json!(SOURCE_TITLE))
    );
    let cold_listed = request(
        &restarted,
        14,
        METHOD_THREAD_LIST,
        json!({"archived": false}),
    )
    .await;
    assert_eq!(
        cold_listed["result"]["data"]
            .as_array()
            .expect("cold thread/list data")
            .iter()
            .filter(|thread| thread["name"] == SOURCE_TITLE)
            .count(),
        2
    );
}

#[tokio::test]
async fn thread_fork_defers_goal_across_restart_until_explicit_turn_admission() {
    let temp = TempDir::new().expect("thread fork temp dir");
    let projection_path = temp.path().join("projection.sqlite");
    let event_log_root = temp.path().join("event-log");
    let backend = Arc::new(ForkBackend::new());
    let server = AppServer::with_runtime(
        RuntimeCore::with_backend(backend.clone())
            .with_projection_store(Arc::new(
                ProjectionStore::initialize(&projection_path).expect("thread fork store"),
            ))
            .with_event_log_writer(Arc::new(
                EventLogWriter::new(&event_log_root).expect("thread fork event log"),
            )),
    );
    initialize(&server, 1).await;

    let started = request(
        &server,
        2,
        METHOD_THREAD_START,
        json!({
            "model": "fixture-model",
            "modelProvider": "fixture-provider",
            "cwd": temp.path()
        }),
    )
    .await;
    let source_thread_id = required_string(&started, "/result/thread/id");

    request(
        &server,
        3,
        METHOD_TURN_START,
        json!({
            "threadId": source_thread_id,
            "input": [{"type": "text", "text": "create fork history"}],
            "model": "fixture-model",
            "approvalPolicy": "never",
            "sandboxPolicy": "workspace-write"
        }),
    )
    .await;
    timeout(Duration::from_secs(3), backend.first_completed.notified())
        .await
        .expect("source turn should complete");
    wait_for_completed_turn(&server, &source_thread_id).await;

    request(
        &server,
        4,
        METHOD_THREAD_GOAL_SET,
        json!({
            "threadId": source_thread_id,
            "objective": "finish the forked goal",
            "tokenBudget": 500
        }),
    )
    .await;
    let forked = request(
        &server,
        5,
        METHOD_THREAD_FORK,
        json!({
            "threadId": source_thread_id,
            "deferGoalContinuation": true
        }),
    )
    .await;
    let target_thread_id = required_string(&forked, "/result/thread/id");
    assert_ne!(target_thread_id, source_thread_id);
    assert_eq!(
        forked.pointer("/result/thread/forkedFromId"),
        Some(&json!(source_thread_id))
    );
    assert_eq!(
        forked
            .pointer("/result/thread/turns")
            .and_then(Value::as_array)
            .map(Vec::len),
        Some(1)
    );
    assert_eq!(deferral_count(&projection_path, &target_thread_id), 1);

    let inherited_goal = request(
        &server,
        6,
        METHOD_THREAD_GOAL_GET,
        json!({"threadId": target_thread_id}),
    )
    .await;
    assert_eq!(
        inherited_goal.pointer("/result/goal/objective"),
        Some(&json!("finish the forked goal"))
    );
    drop(server);

    let restarted = AppServer::with_runtime(
        RuntimeCore::with_backend(backend.clone())
            .with_projection_store(Arc::new(
                ProjectionStore::initialize(&projection_path).expect("reopen thread fork store"),
            ))
            .with_event_log_writer(Arc::new(
                EventLogWriter::new(&event_log_root).expect("reopen thread fork event log"),
            )),
    );
    initialize(&restarted, 10).await;
    request(
        &restarted,
        11,
        METHOD_THREAD_RESUME,
        json!({"threadId": target_thread_id}),
    )
    .await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(backend.calls.load(Ordering::SeqCst), 1);
    assert_eq!(deferral_count(&projection_path, &target_thread_id), 1);

    request(
        &restarted,
        12,
        METHOD_TURN_START,
        json!({
            "threadId": target_thread_id,
            "input": [{"type": "text", "text": "explicit fork turn"}],
            "model": "fixture-model",
            "approvalPolicy": "never",
            "sandboxPolicy": "workspace-write"
        }),
    )
    .await;
    timeout(Duration::from_secs(3), backend.explicit_started.notified())
        .await
        .expect("explicit fork turn should start");
    assert_eq!(deferral_count(&projection_path, &target_thread_id), 0);

    request(
        &restarted,
        13,
        METHOD_THREAD_GOAL_SET,
        json!({"threadId": target_thread_id, "status": "paused"}),
    )
    .await;
    backend.release_explicit.notify_one();
}

#[tokio::test]
async fn thread_fork_preserves_boundaries_and_goal_deferral_lifecycle() {
    let temp = TempDir::new().expect("thread fork boundary temp dir");
    let projection_path = temp.path().join("projection.sqlite");
    let server = AppServer::with_runtime(
        RuntimeCore::with_backend(Arc::new(ImmediateBackend)).with_projection_store(Arc::new(
            ProjectionStore::initialize(&projection_path).expect("thread fork boundary store"),
        )),
    );
    initialize(&server, 100).await;

    let started = request(
        &server,
        101,
        METHOD_THREAD_START,
        json!({
            "model": "fixture-model",
            "modelProvider": "fixture-provider",
            "cwd": temp.path()
        }),
    )
    .await;
    let source_thread_id = required_string(&started, "/result/thread/id");
    for (id, text, expected_turns) in [
        (102, "first source turn", 1),
        (103, "second source turn", 2),
    ] {
        request(
            &server,
            id,
            METHOD_TURN_START,
            json!({
                "threadId": source_thread_id,
                "input": [{"type": "text", "text": text}],
                "model": "fixture-model",
                "approvalPolicy": "never",
                "sandboxPolicy": "workspace-write"
            }),
        )
        .await;
        wait_for_completed_turn_count(&server, &source_thread_id, expected_turns).await;
    }

    let source = request(
        &server,
        104,
        METHOD_THREAD_READ,
        json!({"threadId": source_thread_id, "includeTurns": true}),
    )
    .await;
    let first_turn_id = required_string(&source, "/result/thread/turns/0/id");

    let ordinary = request(
        &server,
        105,
        METHOD_THREAD_FORK,
        json!({"threadId": source_thread_id}),
    )
    .await;
    let ordinary_thread_id = required_string(&ordinary, "/result/thread/id");
    assert_eq!(turn_count(&ordinary), 2);
    assert_eq!(goal_count(&projection_path, &ordinary_thread_id), 0);
    assert_eq!(deferral_count(&projection_path, &ordinary_thread_id), 0);

    let last_turn = request(
        &server,
        106,
        METHOD_THREAD_FORK,
        json!({"threadId": source_thread_id, "lastTurnId": first_turn_id}),
    )
    .await;
    assert_eq!(turn_count(&last_turn), 1);

    let before_turn = request(
        &server,
        107,
        METHOD_THREAD_FORK,
        json!({"threadId": source_thread_id, "beforeTurnId": first_turn_id}),
    )
    .await;
    assert_eq!(turn_count(&before_turn), 0);

    let without_goal = request(
        &server,
        108,
        METHOD_THREAD_FORK,
        json!({
            "threadId": source_thread_id,
            "deferGoalContinuation": true
        }),
    )
    .await;
    let without_goal_thread_id = required_string(&without_goal, "/result/thread/id");
    assert_eq!(goal_count(&projection_path, &without_goal_thread_id), 0);
    assert_eq!(deferral_count(&projection_path, &without_goal_thread_id), 0);

    request(
        &server,
        109,
        METHOD_THREAD_GOAL_SET,
        json!({
            "threadId": source_thread_id,
            "objective": "preserve the paused source goal",
            "status": "paused",
            "tokenBudget": 500
        }),
    )
    .await;
    let deferred = request(
        &server,
        110,
        METHOD_THREAD_FORK,
        json!({
            "threadId": source_thread_id,
            "deferGoalContinuation": true
        }),
    )
    .await;
    let deferred_thread_id = required_string(&deferred, "/result/thread/id");
    let inherited_goal = request(
        &server,
        111,
        METHOD_THREAD_GOAL_GET,
        json!({"threadId": deferred_thread_id}),
    )
    .await;
    assert_eq!(
        inherited_goal.pointer("/result/goal/status"),
        Some(&json!("paused"))
    );
    assert_eq!(
        inherited_goal.pointer("/result/goal/objective"),
        Some(&json!("preserve the paused source goal"))
    );
    assert_eq!(deferral_count(&projection_path, &deferred_thread_id), 1);

    request(
        &server,
        112,
        METHOD_THREAD_DELETE,
        json!({"threadId": deferred_thread_id}),
    )
    .await;
    assert_eq!(goal_count(&projection_path, &deferred_thread_id), 0);
    assert_eq!(deferral_count(&projection_path, &deferred_thread_id), 0);

    for (id, params, expected_message) in [
        (
            113,
            json!({
                "threadId": source_thread_id,
                "lastTurnId": first_turn_id,
                "beforeTurnId": first_turn_id
            }),
            "thread/fork beforeTurnId cannot be combined with lastTurnId",
        ),
        (
            114,
            json!({
                "threadId": source_thread_id,
                "ephemeral": true,
                "deferGoalContinuation": true
            }),
            "thread/fork deferGoalContinuation cannot be combined with ephemeral",
        ),
    ] {
        let error = request_error(&server, id, METHOD_THREAD_FORK, params).await;
        assert_eq!(
            error.pointer("/error/message"),
            Some(&json!(expected_message))
        );
    }
}

#[tokio::test]
async fn thread_fork_rebuilds_provider_history_across_restarts_without_duplicate_prefix() {
    let temp = TempDir::new().expect("thread fork provider history temp dir");
    let projection_path = temp.path().join("projection.sqlite");
    let event_log_root = temp.path().join("event-log");
    let sidecar_root = temp.path().join("sidecar");
    let local_image_path = temp.path().join("local-input.png");
    std::fs::write(&local_image_path, b"\x89PNG\r\n\x1a\nfixture")
        .expect("write local image fixture");
    let source_input = json!([
        {
            "type": "text",
            "text": "source user prompt",
            "text_elements": [{
                "byteRange": {"start": 0, "end": 6},
                "placeholder": "source"
            }]
        },
        {
            "type": "image",
            "url": "https://example.com/remote.png",
            "detail": "high"
        },
        {
            "type": "localImage",
            "path": local_image_path,
            "detail": "original"
        },
        {
            "type": "skill",
            "name": "review",
            "path": "/skills/review/SKILL.md"
        },
        {
            "type": "mention",
            "name": "docs",
            "path": "app://docs"
        }
    ]);
    let backend = Arc::new(HistoryCaptureBackend::default());
    let runtime = || {
        RuntimeCore::with_backend(backend.clone())
            .with_projection_store(Arc::new(
                ProjectionStore::initialize(&projection_path)
                    .expect("thread fork provider history store"),
            ))
            .with_event_log_writer(Arc::new(
                EventLogWriter::new(&event_log_root)
                    .expect("thread fork provider history event log"),
            ))
            .with_sidecar_store(Arc::new(
                SidecarStore::new(&sidecar_root).expect("thread fork sidecar store"),
            ))
    };

    let server = AppServer::with_runtime(runtime());
    initialize(&server, 300).await;
    let started = request(
        &server,
        301,
        METHOD_THREAD_START,
        json!({
            "model": "fixture-model",
            "modelProvider": "fixture-provider",
            "cwd": temp.path()
        }),
    )
    .await;
    let source_thread_id = required_string(&started, "/result/thread/id");
    request(
        &server,
        302,
        METHOD_TURN_START,
        json!({
            "threadId": source_thread_id,
            "input": source_input,
            "clientUserMessageId": "client-1",
            "model": "fixture-model",
            "approvalPolicy": "never",
            "sandboxPolicy": "workspace-write"
        }),
    )
    .await;
    wait_for_completed_turn_count(&server, &source_thread_id, 1).await;
    let source_read = request(
        &server,
        303,
        METHOD_THREAD_READ,
        json!({"threadId": source_thread_id, "includeTurns": true}),
    )
    .await;
    let source_content = source_read
        .pointer("/result/thread/turns/0/items/0/content")
        .cloned()
        .expect("canonical source user content");
    assert_canonical_multimodal_input(&source_content);
    assert_eq!(
        source_read.pointer("/result/thread/turns/0/items/0/clientId"),
        Some(&json!("client-1"))
    );
    let forked = request(
        &server,
        304,
        METHOD_THREAD_FORK,
        json!({"threadId": source_thread_id}),
    )
    .await;
    let target_thread_id = required_string(&forked, "/result/thread/id");
    let canonical_file_name = source_content[2]["url"]
        .as_str()
        .and_then(|uri| uri.strip_prefix("sidecar://media/"))
        .expect("canonical input media URI");
    assert!(sidecar_root
        .join("sessions")
        .join(&target_thread_id)
        .join("media")
        .join(canonical_file_name)
        .is_file());
    drop(server);

    let restarted = AppServer::with_runtime(runtime());
    initialize(&restarted, 310).await;
    request(
        &restarted,
        311,
        METHOD_THREAD_RESUME,
        json!({"threadId": target_thread_id}),
    )
    .await;
    let target_read = request(
        &restarted,
        312,
        METHOD_THREAD_READ,
        json!({"threadId": target_thread_id, "includeTurns": true}),
    )
    .await;
    assert_eq!(
        target_read.pointer("/result/thread/turns/0/items/0/content"),
        Some(&source_content)
    );
    request(
        &restarted,
        313,
        METHOD_TURN_START,
        json!({
            "threadId": target_thread_id,
            "input": [{"type": "text", "text": "first target prompt"}],
            "model": "fixture-model",
            "approvalPolicy": "never",
            "sandboxPolicy": "workspace-write"
        }),
    )
    .await;
    wait_for_completed_turn_count(&restarted, &target_thread_id, 2).await;
    drop(restarted);

    let restarted_again = AppServer::with_runtime(runtime());
    initialize(&restarted_again, 320).await;
    let cold_turn = request_error(
        &restarted_again,
        321,
        METHOD_TURN_START,
        json!({
            "threadId": target_thread_id,
            "input": [{"type": "text", "text": "second target prompt"}],
            "model": "fixture-model",
            "approvalPolicy": "never",
            "sandboxPolicy": "workspace-write"
        }),
    )
    .await;
    assert_eq!(
        cold_turn.pointer("/error/code"),
        Some(&json!(error_codes::INVALID_REQUEST))
    );
    request(
        &restarted_again,
        322,
        METHOD_THREAD_RESUME,
        json!({"threadId": target_thread_id}),
    )
    .await;
    request(
        &restarted_again,
        323,
        METHOD_TURN_START,
        json!({
            "threadId": target_thread_id,
            "input": [{"type": "text", "text": "second target prompt"}],
            "model": "fixture-model",
            "approvalPolicy": "never",
            "sandboxPolicy": "workspace-write"
        }),
    )
    .await;
    wait_for_completed_turn_count(&restarted_again, &target_thread_id, 3).await;

    let histories = backend
        .histories
        .lock()
        .expect("fork history capture mutex poisoned");
    assert_eq!(histories.len(), 3);
    assert_source_provider_history(&histories[1]);
    assert_eq!(&histories[2][..histories[1].len()], histories[1].as_slice());
    assert!(matches!(
        &histories[2][histories[1].len()],
        CurrentProviderMessage {
            role: CurrentProviderRole::User,
            content,
        } if matches!(
            &content[..],
            [CurrentProviderContent::Text(text)] if text == "first target prompt"
        )
    ));
}

fn assert_canonical_multimodal_input(content: &Value) {
    let parts = content.as_array().expect("canonical input parts");
    assert_eq!(parts.len(), 5);
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[0]["text"], "source user prompt");
    assert_eq!(parts[1]["type"], "image");
    assert_eq!(parts[1]["url"], "https://example.com/remote.png");
    assert_eq!(parts[2]["type"], "image");
    assert!(parts[2]["url"]
        .as_str()
        .is_some_and(|url| url.starts_with("sidecar://media/input-") && url.ends_with(".png")));
    assert!(parts[2].get("path").is_none());
    assert_eq!(parts[3]["type"], "skill");
    assert_eq!(parts[4]["type"], "mention");
}

fn assert_source_provider_history(history: &[CurrentProviderMessage]) {
    assert_eq!(
        history.len(),
        4,
        "unexpected provider history: {history:#?}"
    );
    assert!(matches!(
        &history[0],
        CurrentProviderMessage { role: CurrentProviderRole::User, content }
            if matches!(&content[..], [
                CurrentProviderContent::Text(text),
                CurrentProviderContent::Image {
                    uri: remote_uri,
                    detail: Some(ImageDetail::High),
                    ..
                },
                CurrentProviderContent::Image {
                    uri: local_uri,
                    provider_data: Some(local_data),
                    detail: Some(ImageDetail::Original),
                    ..
                },
            ] if text == "source user prompt"
                && remote_uri == "https://example.com/remote.png"
                && local_uri.starts_with("sidecar://media/")
                && local_data.starts_with("data:image/png;base64,"))
    ));
    assert!(matches!(
        &history[1],
        CurrentProviderMessage { role: CurrentProviderRole::Assistant, content }
            if matches!(&content[..], [CurrentProviderContent::ToolCall(call)]
                if call.id == "source-tool-call"
                    && call.name == "read_file"
                    && call.arguments == json!({"path": "README.md"}))
    ));
    assert!(matches!(
        &history[2],
        CurrentProviderMessage { role: CurrentProviderRole::Tool, content }
            if matches!(&content[..], [CurrentProviderContent::ToolResult(result)]
                if result.call_id == "source-tool-call"
                    && result.name == "read_file"
                    && result.success
                    && result.output == "source tool output")
    ));
    assert!(matches!(
        &history[3],
        CurrentProviderMessage { role: CurrentProviderRole::Assistant, content }
            if matches!(&content[..], [CurrentProviderContent::Text(text)] if text == "source assistant answer")
    ));
}

async fn initialize(server: &AppServer, id: u64) {
    request(
        server,
        id,
        METHOD_INITIALIZE,
        json!({
            "clientInfo": {"name": "thread-fork-jsonrpc-test", "version": "1.0.0"}
        }),
    )
    .await;
    let messages = server
        .handle_json_line(
            &json!({"jsonrpc": "2.0", "method": METHOD_INITIALIZED, "params": {}}).to_string(),
        )
        .await
        .expect("initialized notification");
    assert!(messages.is_empty());
}

async fn request(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    let messages = server
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
    let response = messages
        .iter()
        .filter_map(|message| serde_json::from_str::<Value>(message).ok())
        .find(|message| message.get("id") == Some(&json!(id)))
        .unwrap_or_else(|| panic!("missing {method} response: {messages:#?}"));
    if let Some(error) = response.get("error") {
        panic!("{method} failed: {error}");
    }
    response
}

async fn request_error(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    let messages = server
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
        .expect("handle JSON-RPC error request");
    let response = messages
        .iter()
        .filter_map(|message| serde_json::from_str::<Value>(message).ok())
        .find(|message| message.get("id") == Some(&json!(id)))
        .unwrap_or_else(|| panic!("missing {method} error response: {messages:#?}"));
    assert!(response.get("error").is_some(), "expected {method} failure");
    response
}

fn required_string(response: &Value, pointer: &str) -> String {
    response
        .pointer(pointer)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("missing string at {pointer}: {response:#}"))
        .to_string()
}

fn deferral_count(path: &std::path::Path, thread_id: &str) -> i64 {
    Connection::open(path)
        .expect("open thread fork store")
        .query_row(
            "SELECT COUNT(*) FROM thread_goal_continuation_deferrals WHERE thread_id = ?1",
            params![thread_id],
            |row| row.get(0),
        )
        .expect("count thread goal deferrals")
}

fn goal_count(path: &std::path::Path, thread_id: &str) -> i64 {
    Connection::open(path)
        .expect("open thread fork store")
        .query_row(
            "SELECT COUNT(*) FROM thread_goals WHERE thread_id = ?1",
            params![thread_id],
            |row| row.get(0),
        )
        .expect("count thread goals")
}

fn turn_count(response: &Value) -> usize {
    response
        .pointer("/result/thread/turns")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or_else(|| panic!("missing thread turns: {response:#}"))
}

async fn wait_for_completed_turn(server: &AppServer, thread_id: &str) {
    for id in 20..70 {
        let read = request(
            server,
            id,
            METHOD_THREAD_READ,
            json!({"threadId": thread_id, "includeTurns": true}),
        )
        .await;
        if read
            .pointer("/result/thread/turns/0/status")
            .is_some_and(|status| status == "completed")
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("source turn did not reach durable completed state");
}

async fn wait_for_completed_turn_count(server: &AppServer, thread_id: &str, count: usize) {
    let mut last_read = Value::Null;
    for id in 200..260 {
        let read = request(
            server,
            id,
            METHOD_THREAD_READ,
            json!({"threadId": thread_id, "includeTurns": true}),
        )
        .await;
        last_read = read.clone();
        let completed = read
            .pointer("/result/thread/turns")
            .and_then(Value::as_array)
            .map(|turns| {
                turns
                    .iter()
                    .filter(|turn| turn.get("status") == Some(&json!("completed")))
                    .count()
            })
            .unwrap_or_default();
        if completed == count {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("source thread did not reach {count} durable completed turns: {last_read:#}");
}
