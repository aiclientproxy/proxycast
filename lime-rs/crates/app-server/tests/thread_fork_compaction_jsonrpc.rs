use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use app_server::{
    ActionRespondRequest, AppServer, CancelExecutionRequest, EventLogWriter, ExecutionBackend,
    ExecutionRequest, ProjectionStore, ProviderTurnHistory, RuntimeCore, RuntimeCoreError,
    RuntimeEvent, RuntimeEventSink,
};
use app_server_protocol::protocol::v2::{METHOD_THREAD_COMPACT_START, METHOD_THREAD_FORK};
use app_server_protocol::{
    AgentEvent, METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_READ, METHOD_THREAD_RESUME,
    METHOD_THREAD_START, METHOD_TURN_START,
};
use async_trait::async_trait;
use model_provider::current_client::{
    CurrentProviderContent, CurrentProviderMessage, CurrentProviderRole,
};
use serde_json::{json, Value};
use tempfile::TempDir;

#[derive(Default)]
struct CompactionForkBackend {
    calls: AtomicUsize,
    histories: Mutex<Vec<Vec<CurrentProviderMessage>>>,
}

#[async_trait]
impl ExecutionBackend for CompactionForkBackend {
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
            .expect("compaction fork histories mutex poisoned")
            .push(provider_history.messages_for_route("fixture-provider", "fixture-model"));
        let item_id = format!("answer-{}", request.turn.turn_id);
        let text = format!("source assistant {call}");
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        sink.emit(RuntimeEvent::new(
            "message.delta",
            json!({"itemId": item_id, "text": text}),
        ))?;
        sink.emit(RuntimeEvent::new(
            "message.completed",
            json!({
                "itemId": item_id,
                "phase": "final_answer",
                "status": "completed",
                "text": text
            }),
        ))?;
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
async fn compacted_thread_fork_replays_replacement_and_surviving_tail_after_restart() {
    let temp = TempDir::new().expect("thread fork compaction temp dir");
    let projection_path = temp.path().join("projection.sqlite");
    let event_log = Arc::new(
        EventLogWriter::new(temp.path().join("event-log"))
            .expect("thread fork compaction event log"),
    );
    let backend = Arc::new(CompactionForkBackend::default());
    let runtime = || {
        RuntimeCore::with_backend(backend.clone())
            .with_projection_store(Arc::new(
                ProjectionStore::initialize(&projection_path)
                    .expect("thread fork compaction projection store"),
            ))
            .with_event_log_writer(event_log.clone())
    };

    let source_runtime = runtime();
    let source_runtime_inspection = source_runtime.clone();
    let server = AppServer::with_runtime(source_runtime);
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
    let source_session_id = required_string(&started, "/result/thread/sessionId");

    for index in 0..6 {
        request(
            &server,
            10 + index,
            METHOD_TURN_START,
            json!({
                "threadId": source_thread_id,
                "input": [{"type": "text", "text": format!("source prompt {index}")}],
                "model": "fixture-model",
                "approvalPolicy": "never",
                "sandboxPolicy": "workspace-write"
            }),
        )
        .await;
        wait_for_completed_turn_count(&server, &source_thread_id, index as usize + 1).await;
    }

    request(
        &server,
        30,
        METHOD_THREAD_COMPACT_START,
        json!({"threadId": source_thread_id}),
    )
    .await;
    wait_for_completed_turn_count(&server, &source_thread_id, 7).await;
    let compacted_source = request(
        &server,
        31,
        METHOD_THREAD_READ,
        json!({"threadId": source_thread_id, "includeTurns": true}),
    )
    .await;
    let source_compaction_item = context_compaction_item(&compacted_source);
    assert_context_compaction_is_id_only(source_compaction_item);
    let source_compaction_item_id = required_value_string(source_compaction_item, "/id");

    let source_events_before_fork = source_runtime_inspection
        .events_for_session(&source_session_id)
        .expect("source events before fork");
    let source_marker = single_compaction_marker(&source_events_before_fork);
    let mut expected_history = replacement_messages(&source_marker.payload);
    assert_eq!(expected_history.len(), 3, "unexpected replacement history");
    for index in 2..6 {
        expected_history.push(text_message(
            CurrentProviderRole::User,
            format!("source prompt {index}"),
        ));
        expected_history.push(text_message(
            CurrentProviderRole::Assistant,
            format!("source assistant {index}"),
        ));
    }

    let forked = request(
        &server,
        32,
        METHOD_THREAD_FORK,
        json!({"threadId": source_thread_id}),
    )
    .await;
    let target_thread_id = required_string(&forked, "/result/thread/id");
    let target_session_id = required_string(&forked, "/result/thread/sessionId");
    let forked_compaction_item = context_compaction_item(&forked);
    assert_context_compaction_is_id_only(forked_compaction_item);
    assert_eq!(
        forked_compaction_item.get("id"),
        Some(&json!(source_compaction_item_id))
    );
    assert_eq!(
        source_runtime_inspection
            .events_for_session(&source_session_id)
            .expect("source events after fork"),
        source_events_before_fork,
        "thread/fork must not rewrite the source rollout"
    );
    let target_events = source_runtime_inspection
        .events_for_session(&target_session_id)
        .expect("target fork seed events");
    let target_marker = single_compaction_marker(&target_events);
    assert_eq!(
        target_marker.payload.get("itemId"),
        Some(&json!(source_compaction_item_id))
    );
    assert_eq!(
        target_marker.payload.get("tailStartTurnId"),
        source_marker.payload.get("tailStartTurnId")
    );
    drop(server);

    let restarted_runtime = runtime();
    let restarted_runtime_inspection = restarted_runtime.clone();
    let restarted = AppServer::with_runtime(restarted_runtime);
    initialize(&restarted, 40).await;
    request(
        &restarted,
        41,
        METHOD_THREAD_RESUME,
        json!({"threadId": target_thread_id}),
    )
    .await;
    let restarted_target = request(
        &restarted,
        42,
        METHOD_THREAD_READ,
        json!({"threadId": target_thread_id, "includeTurns": true}),
    )
    .await;
    let restarted_compaction_item = context_compaction_item(&restarted_target);
    assert_context_compaction_is_id_only(restarted_compaction_item);
    assert_eq!(
        restarted_compaction_item.get("id"),
        Some(&json!(source_compaction_item_id))
    );
    let restarted_events = restarted_runtime_inspection
        .events_for_session(&target_session_id)
        .expect("restarted target fork seed events");
    assert_eq!(
        restarted_events
            .iter()
            .filter(|event| event.event_type == "context.compaction.completed")
            .count(),
        1,
        "cold hydration must not duplicate the compaction marker"
    );

    request(
        &restarted,
        43,
        METHOD_TURN_START,
        json!({
            "threadId": target_thread_id,
            "input": [{"type": "text", "text": "target prompt"}],
            "model": "fixture-model",
            "approvalPolicy": "never",
            "sandboxPolicy": "workspace-write"
        }),
    )
    .await;
    wait_for_completed_turn_count(&restarted, &target_thread_id, 8).await;

    let histories = backend
        .histories
        .lock()
        .expect("compaction fork histories mutex poisoned");
    assert_eq!(histories.len(), 7);
    assert_eq!(
        histories.last(),
        Some(&expected_history),
        "forked provider history must be exactly replacement + surviving tail"
    );
}

fn context_compaction_item(response: &Value) -> &Value {
    response
        .pointer("/result/thread/turns")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .flat_map(|turn| {
            turn.get("items")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .find(|item| item.get("type") == Some(&json!("contextCompaction")))
        .unwrap_or_else(|| panic!("missing public contextCompaction item: {response:#}"))
}

fn assert_context_compaction_is_id_only(item: &Value) {
    let object = item.as_object().expect("contextCompaction object");
    assert_eq!(
        object.len(),
        2,
        "private compaction lineage leaked: {item:#}"
    );
    assert!(object.contains_key("id"));
    assert_eq!(object.get("type"), Some(&json!("contextCompaction")));
}

fn single_compaction_marker(events: &[AgentEvent]) -> &AgentEvent {
    let markers = events
        .iter()
        .filter(|event| event.event_type == "context.compaction.completed")
        .collect::<Vec<_>>();
    assert_eq!(
        markers.len(),
        1,
        "unexpected compaction markers: {events:#?}"
    );
    markers[0]
}

fn replacement_messages(payload: &Value) -> Vec<CurrentProviderMessage> {
    payload["replacementHistory"]
        .as_array()
        .expect("replacementHistory array")
        .iter()
        .map(|item| {
            let role = match item["role"].as_str().expect("replacement role") {
                "user" => CurrentProviderRole::User,
                "assistant" => CurrentProviderRole::Assistant,
                "tool" => CurrentProviderRole::Tool,
                role => panic!("unexpected replacement role: {role}"),
            };
            let content = item["content"]
                .as_array()
                .expect("replacement content array")
                .iter()
                .map(|part| {
                    CurrentProviderContent::Text(
                        part.get("text")
                            .and_then(Value::as_str)
                            .expect("replacement text")
                            .to_string(),
                    )
                })
                .collect();
            CurrentProviderMessage { role, content }
        })
        .collect()
}

fn text_message(role: CurrentProviderRole, text: String) -> CurrentProviderMessage {
    CurrentProviderMessage {
        role,
        content: vec![CurrentProviderContent::Text(text)],
    }
}

async fn initialize(server: &AppServer, id: u64) {
    request(
        server,
        id,
        METHOD_INITIALIZE,
        json!({"clientInfo": {"name": "thread-fork-compaction-test", "version": "1.0.0"}}),
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

fn required_string(response: &Value, pointer: &str) -> String {
    required_value_string(response, pointer)
}

fn required_value_string(value: &Value, pointer: &str) -> String {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("missing string at {pointer}: {value:#}"))
        .to_string()
}

async fn wait_for_completed_turn_count(server: &AppServer, thread_id: &str, count: usize) {
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
    panic!("thread did not reach {count} completed turns: {last_read:#}");
}
