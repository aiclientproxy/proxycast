use std::sync::Arc;
use std::time::Duration;

use app_server::{
    run_json_lines, ActionRespondRequest, AppServer, CancelExecutionRequest, EventLogWriter,
    ExecutionBackend, ExecutionRequest, ProjectionStore, RuntimeCore, RuntimeCoreError,
    RuntimeEvent, RuntimeEventSink,
};
use app_server_protocol::protocol::v2::{
    METHOD_THREAD_GOAL_SET, METHOD_THREAD_GOAL_UPDATED, METHOD_THREAD_TOKEN_USAGE_UPDATED,
    METHOD_TURN_COMPLETED, METHOD_TURN_STARTED,
};
use app_server_protocol::{
    METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_READ, METHOD_THREAD_RESUME,
    METHOD_THREAD_START, METHOD_TURN_START, PROTOCOL_VERSION,
};
use async_trait::async_trait;
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, DuplexStream, Lines};
use tokio::sync::Notify;
use tokio::time::{sleep, timeout};

struct UsageHistoryBackend {
    completed: Arc<Notify>,
}

#[async_trait]
impl ExecutionBackend for UsageHistoryBackend {
    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        sink.emit(RuntimeEvent::new(
            "provider.usage",
            json!({
                "backend": "runtime",
                "attempt": 0,
                "usage": {
                    "input_tokens": 120,
                    "cached_input_tokens": 20,
                    "cache_write_input_tokens": 7,
                    "output_tokens": 30,
                    "reasoning_output_tokens": 10,
                    "model_context_window": 128_000
                }
            }),
        ))?;
        sink.emit(RuntimeEvent::new("turn.completed", json!({})))?;
        self.completed.notify_one();
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

struct BlockingContinuationBackend {
    started: Arc<Notify>,
    release: Arc<Notify>,
}

#[async_trait]
impl ExecutionBackend for BlockingContinuationBackend {
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
        self.release.notify_one();
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
async fn cold_resume_replays_token_usage_and_goal_before_live_events_over_jsonl() {
    let temp = TempDir::new().expect("cold resume replay temp dir");
    let projection_path = temp.path().join("projection.sqlite");
    let event_log_root = temp.path().join("event-log");
    let initial_completed = Arc::new(Notify::new());
    let initial_runtime = RuntimeCore::with_backend(Arc::new(UsageHistoryBackend {
        completed: Arc::clone(&initial_completed),
    }))
    .with_projection_store(Arc::new(
        ProjectionStore::initialize(&projection_path).expect("initial projection store"),
    ))
    .with_event_log_writer(Arc::new(
        EventLogWriter::new(&event_log_root).expect("initial event log"),
    ));
    let initial_server = AppServer::with_runtime(initial_runtime);
    initialize_direct(&initial_server).await;

    let started = request_direct(
        &initial_server,
        2,
        METHOD_THREAD_START,
        json!({
            "model": "fixture-model",
            "modelProvider": "fixture-provider",
            "cwd": temp.path()
        }),
    )
    .await;
    let thread_id = required_string(&started, "/result/thread/id", "initial thread id");

    let turn = request_direct(
        &initial_server,
        3,
        METHOD_TURN_START,
        json!({
            "threadId": thread_id,
            "input": [{"type": "text", "text": "persist usage for cold resume"}],
            "model": "fixture-model",
            "approvalPolicy": "never",
            "sandboxPolicy": "workspace-write"
        }),
    )
    .await;
    let original_turn_id = required_string(&turn, "/result/turn/id", "initial turn id");
    timeout(Duration::from_secs(2), initial_completed.notified())
        .await
        .expect("initial usage turn must complete");
    wait_for_terminal_turn(&initial_server, &thread_id, &original_turn_id).await;
    drop(initial_server);

    let connection =
        rusqlite::Connection::open(&projection_path).expect("open cold resume projection database");
    connection
        .execute(
            r#"INSERT INTO thread_goals (
                   thread_id, goal_id, objective, status, token_budget,
                   tokens_used, time_used_seconds, created_at_ms, updated_at_ms
               ) VALUES (?1, 'goal-cold-resume-replay', ?2, 'active', 1000, 0, 0, 1, 1)"#,
            rusqlite::params![thread_id, "continue after replaying durable state"],
        )
        .expect("seed active durable goal");
    drop(connection);

    let continuation_started = Arc::new(Notify::new());
    let continuation_release = Arc::new(Notify::new());
    let restarted_runtime = RuntimeCore::with_backend(Arc::new(BlockingContinuationBackend {
        started: Arc::clone(&continuation_started),
        release: Arc::clone(&continuation_release),
    }))
    .with_projection_store(Arc::new(
        ProjectionStore::initialize(&projection_path).expect("restarted projection store"),
    ))
    .with_event_log_writer(Arc::new(
        EventLogWriter::new(&event_log_root).expect("restarted event log"),
    ));
    let restarted_server = AppServer::with_runtime(restarted_runtime);
    let (mut input_client, input_server) = tokio::io::duplex(32 * 1024);
    let (output_server, output_client) = tokio::io::duplex(32 * 1024);
    let runner = tokio::spawn(run_json_lines(
        restarted_server,
        input_server,
        output_server,
    ));
    let mut output_lines = BufReader::new(output_client).lines();
    initialize_jsonl(&mut input_client, &mut output_lines, 10).await;

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 11,
            "method": METHOD_THREAD_RESUME,
            "params": { "threadId": thread_id }
        }),
    )
    .await;

    let mut response_seen = false;
    let mut token_usage_seen = false;
    let mut goal_seen = false;
    let continuation_turn_id = timeout(Duration::from_secs(3), async {
        loop {
            let message = next_message(&mut output_lines).await;
            assert_ne!(
                message.get("method"),
                Some(&json!("thread/started")),
                "thread/resume must not emit thread/started: {message:#?}"
            );
            if message.get("id") == Some(&json!(11)) {
                assert!(!response_seen, "thread/resume response must be unique");
                assert!(
                    !token_usage_seen && !goal_seen,
                    "resume response must lead replay"
                );
                assert_eq!(
                    message.pointer("/result/thread/id"),
                    Some(&json!(thread_id))
                );
                response_seen = true;
                continue;
            }

            assert!(
                response_seen,
                "resume response must precede replay: {message:#?}"
            );
            match message.get("method").and_then(Value::as_str) {
                Some(METHOD_THREAD_TOKEN_USAGE_UPDATED) => {
                    assert!(!token_usage_seen, "token usage snapshot must be unique");
                    assert!(!goal_seen, "token usage must precede goal snapshot");
                    assert_eq!(message.pointer("/params/threadId"), Some(&json!(thread_id)));
                    assert_eq!(
                        message.pointer("/params/turnId"),
                        Some(&json!(original_turn_id))
                    );
                    assert_eq!(
                        message.pointer("/params/tokenUsage/total/inputTokens"),
                        Some(&json!(120))
                    );
                    assert_eq!(
                        message.pointer("/params/tokenUsage/total/cachedInputTokens"),
                        Some(&json!(20))
                    );
                    assert_eq!(
                        message.pointer("/params/tokenUsage/total/cacheWriteInputTokens"),
                        Some(&json!(7))
                    );
                    assert_eq!(
                        message.pointer("/params/tokenUsage/last/cacheWriteInputTokens"),
                        Some(&json!(7))
                    );
                    assert_eq!(
                        message.pointer("/params/tokenUsage/last/reasoningOutputTokens"),
                        Some(&json!(10))
                    );
                    assert_eq!(
                        message.pointer("/params/tokenUsage/modelContextWindow"),
                        Some(&json!(128_000))
                    );
                    token_usage_seen = true;
                }
                Some(METHOD_THREAD_GOAL_UPDATED) => {
                    assert!(token_usage_seen, "goal snapshot must follow token usage");
                    assert!(!goal_seen, "goal snapshot must be unique");
                    assert_eq!(message.pointer("/params/threadId"), Some(&json!(thread_id)));
                    assert_eq!(
                        message.pointer("/params/goal/objective"),
                        Some(&json!("continue after replaying durable state"))
                    );
                    assert_eq!(
                        message.pointer("/params/goal/status"),
                        Some(&json!("active"))
                    );
                    goal_seen = true;
                }
                Some(METHOD_TURN_STARTED) => {
                    assert!(goal_seen, "live turn event must follow replay snapshots");
                    assert_eq!(message.pointer("/params/threadId"), Some(&json!(thread_id)));
                    break required_string(&message, "/params/turn/id", "continuation turn id");
                }
                _ => {}
            }
        }
    })
    .await
    .expect("cold resume replay and live event timeout");

    timeout(Duration::from_secs(2), continuation_started.notified())
        .await
        .expect("cold-resumed continuation must reach the backend");

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 12,
            "method": METHOD_THREAD_GOAL_SET,
            "params": { "threadId": thread_id, "status": "paused" }
        }),
    )
    .await;
    let paused = read_response(&mut output_lines, 12).await;
    assert!(
        paused.get("error").is_none(),
        "pause active goal: {paused:#?}"
    );

    continuation_release.notify_one();
    read_turn_notification(
        &mut output_lines,
        METHOD_TURN_COMPLETED,
        &continuation_turn_id,
    )
    .await;

    drop(input_client);
    timeout(Duration::from_secs(2), runner)
        .await
        .expect("restarted JSONL runner should stop")
        .expect("restarted JSONL runner task")
        .expect("restarted JSONL runner result");
}

async fn initialize_direct(server: &AppServer) {
    let response = request_direct(
        server,
        1,
        METHOD_INITIALIZE,
        json!({
            "clientInfo": {
                "name": "thread-resume-replay-jsonrpc-test",
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

async fn request_direct(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
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
        .expect("handle direct JSON-RPC request");
    let response = lines
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).expect("decode direct JSON-RPC message"))
        .find(|message| message.get("id") == Some(&json!(id)))
        .unwrap_or_else(|| panic!("missing {method} response: {lines:#?}"));
    assert!(
        response.get("error").is_none(),
        "{method} failed: {response:#?}"
    );
    response
}

async fn wait_for_terminal_turn(server: &AppServer, thread_id: &str, turn_id: &str) {
    timeout(Duration::from_secs(2), async {
        let mut request_id = 4;
        loop {
            let read = request_direct(
                server,
                request_id,
                METHOD_THREAD_READ,
                json!({ "threadId": thread_id, "includeTurns": true }),
            )
            .await;
            if read.pointer("/result/thread/turns/0/id") == Some(&json!(turn_id))
                && read.pointer("/result/thread/turns/0/status") == Some(&json!("completed"))
            {
                break;
            }
            request_id += 1;
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("usage turn must become durable before restart");
}

async fn initialize_jsonl(
    input: &mut DuplexStream,
    output: &mut Lines<BufReader<DuplexStream>>,
    id: u64,
) {
    write_message(
        input,
        json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": METHOD_INITIALIZE,
            "params": {
                "clientInfo": {
                    "name": "thread-resume-replay-jsonrpc-test",
                    "version": "1.0.0"
                }
            }
        }),
    )
    .await;
    let response = read_response(output, id).await;
    assert_eq!(
        response.pointer("/result/serverInfo/protocolVersion"),
        Some(&json!(PROTOCOL_VERSION))
    );
    write_message(
        input,
        json!({ "jsonrpc": "2.0", "method": METHOD_INITIALIZED, "params": {} }),
    )
    .await;
}

async fn write_message(client: &mut DuplexStream, message: Value) {
    client
        .write_all(format!("{}\n", message).as_bytes())
        .await
        .expect("write JSONL message");
}

async fn read_response(lines: &mut Lines<BufReader<DuplexStream>>, id: u64) -> Value {
    loop {
        let message = next_message(lines).await;
        if message.get("id") == Some(&json!(id)) {
            return message;
        }
    }
}

async fn read_turn_notification(
    lines: &mut Lines<BufReader<DuplexStream>>,
    method: &str,
    turn_id: &str,
) {
    loop {
        let message = next_message(lines).await;
        if message.get("method") == Some(&json!(method))
            && message.pointer("/params/turn/id") == Some(&json!(turn_id))
        {
            return;
        }
    }
}

async fn next_message(lines: &mut Lines<BufReader<DuplexStream>>) -> Value {
    timeout(Duration::from_secs(2), lines.next_line())
        .await
        .expect("JSONL output timeout")
        .expect("read JSONL output")
        .map(|line| serde_json::from_str(&line).expect("decode JSONL output"))
        .expect("JSONL output must remain open")
}

fn required_string(message: &Value, pointer: &str, context: &str) -> String {
    message
        .pointer(pointer)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("missing {context}: {message:#?}"))
        .to_string()
}
