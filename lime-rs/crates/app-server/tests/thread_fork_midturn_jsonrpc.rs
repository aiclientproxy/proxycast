use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use app_server::{
    ActionRespondRequest, AppServer, CancelExecutionRequest, EventLogWriter, ExecutionBackend,
    ExecutionRequest, ProjectionStore, RuntimeCore, RuntimeCoreError, RuntimeEvent,
    RuntimeEventSink,
};
use app_server_protocol::protocol::v2::METHOD_THREAD_FORK;
use app_server_protocol::{
    error_codes, METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_READ, METHOD_THREAD_RESUME,
    METHOD_THREAD_START, METHOD_TURN_START,
};
use async_trait::async_trait;
use model_provider::current_client::{
    CurrentProviderContent, CurrentProviderMessage, CurrentProviderRole,
};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::sync::Notify;
use tokio::time::timeout;

struct MidTurnBackend {
    calls: AtomicUsize,
    source_started: Notify,
    release_source: Notify,
    histories: Mutex<Vec<Vec<CurrentProviderMessage>>>,
    fork_lineages: Mutex<Vec<Option<String>>>,
}

impl MidTurnBackend {
    fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
            source_started: Notify::new(),
            release_source: Notify::new(),
            histories: Mutex::new(Vec::new()),
            fork_lineages: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl ExecutionBackend for MidTurnBackend {
    async fn start_turn(
        &self,
        request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.start_turn_with_provider_history(request, Vec::new(), sink)
            .await
    }

    async fn start_turn_with_provider_history(
        &self,
        request: ExecutionRequest,
        provider_history: Vec<CurrentProviderMessage>,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        self.histories
            .lock()
            .expect("mid-turn history mutex poisoned")
            .push(provider_history);
        self.fork_lineages
            .lock()
            .expect("mid-turn lineage mutex poisoned")
            .push(request.forked_from_thread_id);
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        if call == 0 {
            sink.emit(RuntimeEvent::new(
                "message.delta",
                json!({"itemId": "partial-answer", "text": "partial source answer"}),
            ))?;
            sink.emit(RuntimeEvent::new(
                "message.completed",
                json!({
                    "itemId": "partial-answer",
                    "phase": "final_answer",
                    "status": "completed",
                    "text": "partial source answer"
                }),
            ))?;
            self.source_started.notify_one();
            self.release_source.notified().await;
        }
        sink.emit(RuntimeEvent::new("turn.completed", json!({})))
    }

    async fn cancel_turn(
        &self,
        _request: CancelExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.release_source.notify_one();
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
async fn thread_fork_snapshots_active_turn_and_replays_one_interrupted_marker() {
    let temp = TempDir::new().expect("mid-turn fork temp dir");
    let projection_path = temp.path().join("projection.sqlite");
    let event_log_root = temp.path().join("event-log");
    let backend = Arc::new(MidTurnBackend::new());
    let runtime = || {
        RuntimeCore::with_backend(backend.clone())
            .with_projection_store(Arc::new(
                ProjectionStore::initialize(&projection_path).expect("mid-turn fork store"),
            ))
            .with_event_log_writer(Arc::new(
                EventLogWriter::new(&event_log_root).expect("mid-turn fork event log"),
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
            "cwd": temp.path()
        }),
    )
    .await;
    let source_thread_id = required_string(&started, "/result/thread/id");
    let turn_started = request(
        &server,
        3,
        METHOD_TURN_START,
        json!({
            "threadId": source_thread_id,
            "input": [{"type": "text", "text": "source user prompt"}],
            "model": "fixture-model",
            "approvalPolicy": "never",
            "sandboxPolicy": "workspace-write"
        }),
    )
    .await;
    let source_turn_id = required_string(&turn_started, "/result/turn/id");
    timeout(Duration::from_secs(3), backend.source_started.notified())
        .await
        .expect("source turn should become active");
    wait_for_agent_message(&server, &source_thread_id, "partial source answer").await;

    let fork_lines = request_lines(
        &server,
        4,
        METHOD_THREAD_FORK,
        json!({"threadId": source_thread_id}),
    )
    .await;
    let forked = successful_response(&fork_lines, 4, METHOD_THREAD_FORK);
    let target_thread_id = required_string(forked, "/result/thread/id");
    assert_eq!(
        forked.pointer("/result/thread/turns/0/id"),
        Some(&json!(source_turn_id))
    );
    assert_eq!(
        forked.pointer("/result/thread/turns/0/status"),
        Some(&json!("interrupted"))
    );
    let started_notification = fork_lines
        .iter()
        .find(|message| message.get("method") == Some(&json!("thread/started")))
        .unwrap_or_else(|| panic!("missing fork thread/started: {fork_lines:#?}"));
    assert_eq!(
        started_notification.pointer("/params/thread/id"),
        Some(&json!(target_thread_id))
    );
    assert_eq!(
        started_notification.pointer("/params/thread/turns"),
        Some(&json!([]))
    );

    let source = request(
        &server,
        5,
        METHOD_THREAD_READ,
        json!({"threadId": source_thread_id, "includeTurns": true}),
    )
    .await;
    assert_eq!(
        source.pointer("/result/thread/turns/0/status"),
        Some(&json!("inProgress"))
    );
    let last_turn_error = request_error(
        &server,
        6,
        METHOD_THREAD_FORK,
        json!({"threadId": source_thread_id, "lastTurnId": source_turn_id}),
    )
    .await;
    assert_eq!(
        last_turn_error.pointer("/error/code"),
        Some(&json!(error_codes::INVALID_REQUEST))
    );
    let before_turn = request(
        &server,
        7,
        METHOD_THREAD_FORK,
        json!({"threadId": source_thread_id, "beforeTurnId": source_turn_id}),
    )
    .await;
    assert_eq!(turn_count(&before_turn), 0);

    backend.release_source.notify_one();
    wait_for_status(&server, &source_thread_id, 1, "completed").await;
    drop(server);

    let restarted_runtime = runtime();
    let restarted = AppServer::with_runtime(restarted_runtime.clone());
    initialize(&restarted, 20).await;
    request(
        &restarted,
        21,
        METHOD_THREAD_RESUME,
        json!({"threadId": target_thread_id}),
    )
    .await;
    assert_contiguous_events(
        &restarted_runtime
            .events_for_session(&target_thread_id)
            .expect("resumed target events"),
    );
    let target_turn_lines = request_lines(
        &restarted,
        22,
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
    let target_turn_events = restarted_runtime
        .events_for_session(&target_thread_id)
        .expect("target events after turn admission");
    if target_turn_lines
        .iter()
        .any(|message| message.get("error").is_some())
    {
        panic!(
            "target turn admission failed with events {:#?}: {target_turn_lines:#?}",
            target_turn_events
                .iter()
                .map(|event| (event.sequence, event.event_type.as_str()))
                .collect::<Vec<_>>()
        );
    }
    successful_response(&target_turn_lines, 22, METHOD_TURN_START);
    wait_for_status(&restarted, &target_thread_id, 2, "completed").await;
    let reforked = request(
        &restarted,
        23,
        METHOD_THREAD_FORK,
        json!({"threadId": target_thread_id}),
    )
    .await;
    let reforked_thread_id = required_string(&reforked, "/result/thread/id");
    assert_eq!(turn_count(&reforked), 2);
    drop(restarted);

    let restarted_again = AppServer::with_runtime(runtime());
    initialize(&restarted_again, 30).await;
    request(
        &restarted_again,
        31,
        METHOD_THREAD_RESUME,
        json!({"threadId": reforked_thread_id}),
    )
    .await;
    request(
        &restarted_again,
        32,
        METHOD_TURN_START,
        json!({
            "threadId": reforked_thread_id,
            "input": [{"type": "text", "text": "second target prompt"}],
            "model": "fixture-model",
            "approvalPolicy": "never",
            "sandboxPolicy": "workspace-write"
        }),
    )
    .await;
    wait_for_status(&restarted_again, &reforked_thread_id, 3, "completed").await;

    let histories = backend
        .histories
        .lock()
        .expect("mid-turn history mutex poisoned");
    assert_eq!(
        histories.len(),
        3,
        "unexpected provider calls: {histories:#?}"
    );
    assert_interrupted_history(&histories[1], None);
    assert_interrupted_history(&histories[2], Some("first target prompt"));
    assert_eq!(
        *backend
            .fork_lineages
            .lock()
            .expect("mid-turn lineage mutex poisoned"),
        vec![None, Some(source_thread_id), Some(target_thread_id),]
    );
}

fn assert_interrupted_history(history: &[CurrentProviderMessage], tail_user: Option<&str>) {
    let developer_indexes = history
        .iter()
        .enumerate()
        .filter_map(|(index, message)| {
            (message.role == CurrentProviderRole::Developer).then_some(index)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        developer_indexes.len(),
        1,
        "interrupted marker must appear exactly once: {history:#?}"
    );
    let partial_index = history
        .iter()
        .position(|message| message_text(message) == Some("partial source answer"))
        .unwrap_or_else(|| panic!("missing partial source answer: {history:#?}"));
    assert!(
        partial_index < developer_indexes[0],
        "developer marker must follow partial assistant history: {history:#?}"
    );
    match tail_user {
        Some(expected) => assert!(
            history
                .iter()
                .skip(developer_indexes[0] + 1)
                .any(|message| message_text(message) == Some(expected)),
            "missing reforked target input after marker: {history:#?}"
        ),
        None => assert_eq!(
            history.len(),
            developer_indexes[0] + 1,
            "first target input must be excluded from provider history: {history:#?}"
        ),
    }
}

fn message_text(message: &CurrentProviderMessage) -> Option<&str> {
    match &message.content[..] {
        [CurrentProviderContent::Text(text)] => Some(text),
        _ => None,
    }
}

fn assert_contiguous_events(events: &[app_server_protocol::AgentEvent]) {
    let sequences = events
        .iter()
        .map(|event| (event.sequence, event.event_type.as_str()))
        .collect::<Vec<_>>();
    assert!(
        events
            .iter()
            .enumerate()
            .all(|(index, event)| event.sequence == index as u64 + 1),
        "fork events must be a unique contiguous sequence: {sequences:#?}"
    );
}

async fn initialize(server: &AppServer, id: u64) {
    request(
        server,
        id,
        METHOD_INITIALIZE,
        json!({
            "clientInfo": {"name": "thread-fork-midturn-test", "version": "1.0.0"}
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
    let lines = request_lines(server, id, method, params).await;
    successful_response(&lines, id, method).clone()
}

async fn request_error(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    let lines = request_lines(server, id, method, params).await;
    let response = lines
        .into_iter()
        .find(|message| message.get("id") == Some(&json!(id)))
        .unwrap_or_else(|| panic!("missing {method} error response"));
    assert!(response.get("error").is_some(), "expected {method} failure");
    response
}

async fn request_lines(server: &AppServer, id: u64, method: &str, params: Value) -> Vec<Value> {
    server
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
        .expect("handle JSON-RPC request")
        .into_iter()
        .map(|line| serde_json::from_str(&line).expect("JSON-RPC line"))
        .collect()
}

fn successful_response<'a>(lines: &'a [Value], id: u64, method: &str) -> &'a Value {
    let response = lines
        .iter()
        .find(|message| message.get("id") == Some(&json!(id)))
        .unwrap_or_else(|| panic!("missing {method} response: {lines:#?}"));
    if let Some(error) = response.get("error") {
        panic!("{method} failed: {error}");
    }
    response
}

fn required_string(response: &Value, pointer: &str) -> String {
    response
        .pointer(pointer)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("missing string at {pointer}: {response:#}"))
        .to_string()
}

fn turn_count(response: &Value) -> usize {
    response
        .pointer("/result/thread/turns")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or_else(|| panic!("missing thread turns: {response:#}"))
}

async fn wait_for_status(server: &AppServer, thread_id: &str, turn_count: usize, status: &str) {
    let mut last_read = Value::Null;
    for id in 100..180 {
        let read = request(
            server,
            id,
            METHOD_THREAD_READ,
            json!({"threadId": thread_id, "includeTurns": true}),
        )
        .await;
        last_read = read.clone();
        let turns = read
            .pointer("/result/thread/turns")
            .and_then(Value::as_array);
        if turns.is_some_and(|turns| {
            turns.len() == turn_count
                && turns
                    .last()
                    .and_then(|turn| turn.get("status"))
                    .and_then(Value::as_str)
                    == Some(status)
        }) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("thread {thread_id} did not reach {turn_count} turns with {status}: {last_read:#}");
}

async fn wait_for_agent_message(server: &AppServer, thread_id: &str, expected_text: &str) {
    let mut last_read = Value::Null;
    for id in 180..240 {
        let read = request(
            server,
            id,
            METHOD_THREAD_READ,
            json!({"threadId": thread_id, "includeTurns": true}),
        )
        .await;
        last_read = read.clone();
        let visible = read
            .pointer("/result/thread/turns")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|turn| turn.get("items").and_then(Value::as_array))
            .flatten()
            .any(|item| {
                item.get("type").and_then(Value::as_str) == Some("agentMessage")
                    && item.get("text").and_then(Value::as_str) == Some(expected_text)
            });
        if visible {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!("thread {thread_id} did not expose agent message {expected_text:?}: {last_read:#}");
}
