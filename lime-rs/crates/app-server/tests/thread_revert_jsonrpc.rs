use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use app_server::{
    run_json_lines, ActionRespondRequest, AppServer, AppServerError, CancelExecutionRequest,
    EventLogWriter, ExecutionBackend, ExecutionRequest, ProjectionStore, RuntimeCore,
    RuntimeCoreError, RuntimeEvent, RuntimeEventSink,
};
use app_server_protocol::protocol::v2::{
    METHOD_THREAD_ITEMS_LIST, METHOD_THREAD_REVERT, METHOD_THREAD_START, METHOD_THREAD_TURNS_LIST,
    METHOD_TURN_START,
};
use app_server_protocol::{
    error_codes, METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_RESUME, PROTOCOL_VERSION,
};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, DuplexStream, Lines};
use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tokio::time::{timeout, Duration};

struct ImmediateBackend;

struct BlockingAfterFirstBackend {
    active_started: Arc<Notify>,
    turn_count: AtomicUsize,
}

#[async_trait::async_trait]
impl ExecutionBackend for BlockingAfterFirstBackend {
    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        let turn_index = self.turn_count.fetch_add(1, Ordering::SeqCst);
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        if turn_index == 0 {
            sink.emit(RuntimeEvent::new("turn.completed", json!({})))
        } else {
            self.active_started.notify_one();
            std::future::pending::<()>().await;
            Ok(())
        }
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

#[async_trait::async_trait]
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

#[tokio::test]
async fn thread_revert_is_experimental_and_replaces_paginated_history() {
    let (_temp, server) = test_server();
    initialize_server(&server, true).await;

    let started = request(
        &server,
        2,
        METHOD_THREAD_START,
        json!({
            "model": "fixture-model",
            "modelProvider": "fixture-provider",
            "historyMode": "paginated"
        }),
    )
    .await;
    let thread_id = required_string(&started, "/result/thread/id");
    let first_turn = start_turn(&server, 3, &thread_id, "first").await;
    let second_turn = start_turn(&server, 4, &thread_id, "second").await;
    let _third_turn = start_turn(&server, 5, &thread_id, "third").await;

    let lines = request_lines(
        &server,
        6,
        METHOD_THREAD_REVERT,
        json!({"threadId": thread_id, "beforeTurnId": second_turn}),
    )
    .await;
    let response = response_for(&lines, 6);
    assert_eq!(
        response.pointer("/result/thread/id"),
        Some(&json!(thread_id))
    );
    assert_eq!(
        response.pointer("/result/thread/turns"),
        Some(&json!([])),
        "unexpected revert response: {response:#}"
    );
    assert!(response.pointer("/result/turnsBackwardsCursor").is_some());
    assert!(response.pointer("/result/itemsBackwardsCursor").is_some());
    assert!(lines.iter().any(|value| {
        value.get("method") == Some(&json!("thread/reverted"))
            && value.pointer("/params/threadId") == Some(&json!(thread_id))
    }));

    let turns = request(
        &server,
        7,
        METHOD_THREAD_TURNS_LIST,
        json!({"threadId": thread_id}),
    )
    .await;
    let data = turns
        .pointer("/result/data")
        .and_then(Value::as_array)
        .expect("turns page data");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], json!(first_turn));

    let items = request(
        &server,
        8,
        METHOD_THREAD_ITEMS_LIST,
        json!({
            "threadId": thread_id,
            "cursor": response["result"]["itemsBackwardsCursor"]
        }),
    )
    .await;
    let item_data = items
        .pointer("/result/data")
        .and_then(Value::as_array)
        .expect("items page data");
    assert!(!item_data.is_empty());
    assert!(item_data
        .iter()
        .all(|item| item["turnId"] == json!(first_turn)));

    let repeated = request(
        &server,
        9,
        METHOD_THREAD_REVERT,
        json!({"threadId": thread_id, "beforeTurnId": first_turn}),
    )
    .await;
    assert_eq!(
        repeated.pointer("/result/thread/id"),
        Some(&json!(thread_id))
    );
    assert_eq!(repeated.pointer("/result/thread/turns"), Some(&json!([])));
    let empty_turns = request(
        &server,
        10,
        METHOD_THREAD_TURNS_LIST,
        json!({"threadId": thread_id}),
    )
    .await;
    assert_eq!(empty_turns["result"]["data"], json!([]));
}

#[tokio::test]
async fn thread_revert_reports_missing_turn_exactly() {
    let (_temp, server) = test_server();
    initialize_server(&server, true).await;
    let started = request(
        &server,
        2,
        METHOD_THREAD_START,
        json!({"model": "fixture-model", "modelProvider": "fixture-provider", "historyMode": "paginated"}),
    )
    .await;
    let thread_id = required_string(&started, "/result/thread/id");
    let missing = request_raw(
        &server,
        3,
        METHOD_THREAD_REVERT,
        json!({"threadId": thread_id, "beforeTurnId": "missing-turn"}),
    )
    .await;
    assert_eq!(
        missing.pointer("/error/code"),
        Some(&json!(error_codes::INVALID_REQUEST))
    );
    assert_eq!(
        missing.pointer("/error/message"),
        Some(&json!("turn not found: missing-turn"))
    );
}

#[tokio::test]
async fn thread_revert_interrupts_active_turn_and_keeps_prefix() {
    let temp = TempDir::new().expect("active revert temp dir");
    let active_started = Arc::new(Notify::new());
    let runtime = RuntimeCore::with_backend(Arc::new(BlockingAfterFirstBackend {
        active_started: Arc::clone(&active_started),
        turn_count: AtomicUsize::new(0),
    }))
    .with_event_log_writer(Arc::new(
        EventLogWriter::new(temp.path().join("events")).expect("active revert event log"),
    ))
    .with_projection_store(Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("active revert projection store"),
    ));
    let mut client = TransportClient::start(
        AppServer::with_runtime(runtime),
        "active-thread-revert",
        true,
    )
    .await;

    let started = client
        .request_ok(
            2,
            METHOD_THREAD_START,
            json!({
                "model": "fixture-model",
                "modelProvider": "fixture-provider",
                "historyMode": "paginated"
            }),
        )
        .await;
    let thread_id = required_string(&started, "/result/thread/id");
    let first_turn = start_transport_turn(&mut client, 3, &thread_id, "first").await;
    let first_completed = client.take_notification("turn/completed").await;
    assert_eq!(
        first_completed.pointer("/params/turn/status"),
        Some(&json!("completed"))
    );
    let active = client
        .request_ok(
            4,
            METHOD_TURN_START,
            json!({"threadId": thread_id, "input": [{"type": "text", "text": "active"}]}),
        )
        .await;
    let active_turn = required_string(&active, "/result/turn/id");
    timeout(Duration::from_secs(2), active_started.notified())
        .await
        .expect("active backend turn should start");

    let response = timeout(
        Duration::from_secs(2),
        client.request_ok(
            5,
            METHOD_THREAD_REVERT,
            json!({"threadId": thread_id, "beforeTurnId": active_turn}),
        ),
    )
    .await
    .expect("revert should interrupt the active turn");
    assert_eq!(
        response.pointer("/result/thread/turns"),
        Some(&json!([])),
        "unexpected revert response: {response:#}"
    );
    let interrupted = client.take_notification("turn/completed").await;
    assert_eq!(
        interrupted.pointer("/params/turn/status"),
        Some(&json!("interrupted"))
    );
    let reverted = client.take_notification("thread/reverted").await;
    assert_eq!(
        reverted.pointer("/params/threadId"),
        Some(&json!(thread_id))
    );

    let turns = client
        .request_ok(6, METHOD_THREAD_TURNS_LIST, json!({"threadId": thread_id}))
        .await;
    let data = turns["result"]["data"].as_array().expect("turn page data");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], json!(first_turn));
    client.shutdown().await;
}

#[tokio::test]
async fn thread_revert_cold_resume_keeps_identity_and_workspace_files() {
    let (temp, server) = test_server();
    let workspace = temp.path().join("workspace");
    std::fs::create_dir_all(&workspace).expect("workspace directory");
    let local_file = workspace.join("keep.txt");
    std::fs::write(&local_file, "unchanged").expect("workspace file");
    initialize_server(&server, true).await;

    let started = request(
        &server,
        2,
        METHOD_THREAD_START,
        json!({
            "model": "fixture-model",
            "modelProvider": "fixture-provider",
            "historyMode": "paginated",
            "cwd": workspace
        }),
    )
    .await;
    let thread_id = required_string(&started, "/result/thread/id");
    let first_turn = start_turn(&server, 3, &thread_id, "first").await;
    let second_turn = start_turn(&server, 4, &thread_id, "second").await;
    let reverted = request(
        &server,
        5,
        METHOD_THREAD_REVERT,
        json!({"threadId": thread_id, "beforeTurnId": second_turn}),
    )
    .await;
    assert_eq!(reverted["result"]["thread"]["id"], json!(thread_id));
    assert_eq!(
        std::fs::read_to_string(&local_file).expect("read workspace file"),
        "unchanged"
    );

    drop(server);
    let restarted = AppServer::with_runtime(
        RuntimeCore::with_backend(Arc::new(ImmediateBackend))
            .with_event_log_writer(Arc::new(
                EventLogWriter::new(temp.path().join("events")).expect("event log restart"),
            ))
            .with_projection_store(Arc::new(
                ProjectionStore::initialize(temp.path().join("projection.sqlite"))
                    .expect("projection restart"),
            )),
    );
    initialize_server(&restarted, true).await;
    let resumed = request(
        &restarted,
        6,
        METHOD_THREAD_RESUME,
        json!({"threadId": thread_id, "excludeTurns": true}),
    )
    .await;
    assert_eq!(resumed["result"]["thread"]["id"], json!(thread_id));
    let turns = request(
        &restarted,
        7,
        METHOD_THREAD_TURNS_LIST,
        json!({"threadId": thread_id}),
    )
    .await;
    assert_eq!(turns["result"]["data"].as_array().map(Vec::len), Some(1));
    assert_eq!(turns["result"]["data"][0]["id"], json!(first_turn));
    let third_turn = start_turn(&restarted, 8, &thread_id, "third").await;
    let turns_after = request(
        &restarted,
        9,
        METHOD_THREAD_TURNS_LIST,
        json!({"threadId": thread_id, "sortDirection": "asc"}),
    )
    .await;
    let ids = turns_after["result"]["data"]
        .as_array()
        .expect("resumed turns page")
        .iter()
        .map(|turn| turn["id"].as_str().unwrap_or_default().to_string())
        .collect::<Vec<_>>();
    assert_eq!(ids, vec![first_turn, third_turn]);
    assert_eq!(
        std::fs::read_to_string(&local_file).expect("read workspace file"),
        "unchanged"
    );
}

#[tokio::test]
async fn thread_revert_transport_gate_is_connection_scoped() {
    let (_temp, server) = test_server();
    let mut denied = TransportClient::start(server.clone(), "thread-revert-denied", false).await;
    let denied_response = denied
        .request_raw(
            2,
            METHOD_THREAD_REVERT,
            json!({"threadId": "thread-1", "beforeTurnId": "turn-1"}),
        )
        .await;
    assert_eq!(
        denied_response.pointer("/error/code"),
        Some(&json!(error_codes::INVALID_REQUEST))
    );
    assert_eq!(
        denied_response.pointer("/error/message"),
        Some(&json!(
            "experimental method requires initialize capabilities.experimentalApi"
        ))
    );
    let mut enabled = TransportClient::start(server, "thread-revert-enabled", true).await;
    let started = enabled
        .request_ok(
            3,
            METHOD_THREAD_START,
            json!({
                "model": "fixture-model",
                "modelProvider": "fixture-provider",
                "historyMode": "paginated"
            }),
        )
        .await;
    let thread_id = required_string(&started, "/result/thread/id");
    let allowed_response = enabled
        .request_raw(
            4,
            METHOD_THREAD_REVERT,
            json!({"threadId": thread_id, "beforeTurnId": "missing-turn"}),
        )
        .await;
    assert_eq!(
        allowed_response.pointer("/error/message"),
        Some(&json!("turn not found: missing-turn"))
    );
    denied.shutdown().await;
    enabled.shutdown().await;
}

fn test_server() -> (TempDir, AppServer) {
    let temp = TempDir::new().expect("thread revert temp dir");
    let projection_store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("thread revert projection store"),
    );
    let event_log = Arc::new(EventLogWriter::new(temp.path().join("events")).expect("event log"));
    let runtime = RuntimeCore::with_backend(Arc::new(ImmediateBackend))
        .with_event_log_writer(event_log)
        .with_projection_store(projection_store);
    (temp, AppServer::with_runtime(runtime))
}

async fn initialize_server(server: &AppServer, experimental_api: bool) {
    let response = request(
        server,
        1,
        METHOD_INITIALIZE,
        json!({
            "clientInfo": {"name": "thread-revert-jsonrpc-test", "version": "1.0.0"},
            "capabilities": {"experimentalApi": experimental_api}
        }),
    )
    .await;
    assert_eq!(
        response.pointer("/result/serverInfo/protocolVersion"),
        Some(&json!(PROTOCOL_VERSION))
    );
    let lines = server
        .handle_json_line(
            &json!({"jsonrpc": "2.0", "method": METHOD_INITIALIZED, "params": {}}).to_string(),
        )
        .await
        .expect("handle initialized notification");
    assert!(lines.is_empty());
}

async fn start_turn(server: &AppServer, id: u64, thread_id: &str, text: &str) -> String {
    let response = request(
        server,
        id,
        METHOD_TURN_START,
        json!({"threadId": thread_id, "input": [{"type": "text", "text": text}]}),
    )
    .await;
    let turn_id = required_string(&response, "/result/turn/id");
    for attempt in 0..20_u64 {
        let turns = request(
            server,
            100 + id * 20 + attempt,
            METHOD_THREAD_TURNS_LIST,
            json!({"threadId": thread_id}),
        )
        .await;
        if turns
            .pointer("/result/data")
            .and_then(Value::as_array)
            .and_then(|data| data.iter().find(|turn| turn["id"] == json!(turn_id)))
            .is_some_and(|turn| turn["status"] != json!("inProgress"))
        {
            return turn_id;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("turn did not complete: {turn_id}");
}

async fn start_transport_turn(
    client: &mut TransportClient,
    id: u64,
    thread_id: &str,
    text: &str,
) -> String {
    let response = client
        .request_ok(
            id,
            METHOD_TURN_START,
            json!({"threadId": thread_id, "input": [{"type": "text", "text": text}]}),
        )
        .await;
    let turn_id = required_string(&response, "/result/turn/id");
    for attempt in 0..20_u64 {
        let turns = client
            .request_ok(
                100 + id * 20 + attempt,
                METHOD_THREAD_TURNS_LIST,
                json!({"threadId": thread_id}),
            )
            .await;
        if turns
            .pointer("/result/data")
            .and_then(Value::as_array)
            .and_then(|data| data.iter().find(|turn| turn["id"] == json!(turn_id)))
            .is_some_and(|turn| turn["status"] != json!("inProgress"))
        {
            return turn_id;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("transport turn did not complete: {turn_id}");
}

async fn request(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    let response = request_raw(server, id, method, params).await;
    if let Some(error) = response.get("error") {
        panic!("{method} failed: {error}");
    }
    response
}

async fn request_raw(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    response_for(&request_lines(server, id, method, params).await, id).clone()
}

async fn request_lines(server: &AppServer, id: u64, method: &str, params: Value) -> Vec<Value> {
    server
        .handle_json_line(
            &json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string(),
        )
        .await
        .expect("handle JSON-RPC request")
        .iter()
        .map(|line| serde_json::from_str(line).expect("decode JSON-RPC response"))
        .collect()
}

fn response_for(lines: &[Value], id: u64) -> &Value {
    lines
        .iter()
        .find(|value| value.get("id") == Some(&json!(id)))
        .expect("JSON-RPC response")
}

fn required_string(value: &Value, pointer: &str) -> String {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .expect("non-empty string")
        .to_string()
}

struct TransportClient {
    input: DuplexStream,
    lines: Lines<BufReader<DuplexStream>>,
    pending_messages: VecDeque<Value>,
    runner: JoinHandle<Result<(), AppServerError>>,
}

impl TransportClient {
    async fn start(server: AppServer, name: &str, experimental_api: bool) -> Self {
        let (input, input_server) = tokio::io::duplex(64 * 1024);
        let (output_server, output) = tokio::io::duplex(64 * 1024);
        let runner = tokio::spawn(run_json_lines(server, input_server, output_server));
        let mut client = Self {
            input,
            lines: BufReader::new(output).lines(),
            pending_messages: VecDeque::new(),
            runner,
        };
        let mut params = json!({"clientInfo": {"name": name, "version": "1.0.0"}});
        if experimental_api {
            params["capabilities"] = json!({"experimentalApi": true});
        }
        client.request_ok(1, METHOD_INITIALIZE, params).await;
        client
            .write(json!({
                "jsonrpc": "2.0",
                "method": METHOD_INITIALIZED,
                "params": {},
            }))
            .await;
        client
    }

    async fn request_ok(&mut self, id: u64, method: &str, params: Value) -> Value {
        let response = self.request_raw(id, method, params).await;
        assert!(
            response.get("error").is_none(),
            "{method} failed: {response}"
        );
        response
    }

    async fn request_raw(&mut self, id: u64, method: &str, params: Value) -> Value {
        self.write(json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        }))
        .await;
        loop {
            let message = self.next_wire_message(method).await;
            if message.get("id") == Some(&json!(id)) {
                return message;
            }
            self.pending_messages.push_back(message);
        }
    }

    async fn take_notification(&mut self, method: &str) -> Value {
        if let Some(index) = self
            .pending_messages
            .iter()
            .position(|message| message.get("method") == Some(&json!(method)))
        {
            return self
                .pending_messages
                .remove(index)
                .expect("pending notification index");
        }
        loop {
            let message = self.next_wire_message(method).await;
            if message.get("method") == Some(&json!(method)) {
                return message;
            }
            self.pending_messages.push_back(message);
        }
    }

    async fn next_wire_message(&mut self, context: &str) -> Value {
        let line = timeout(Duration::from_secs(5), self.lines.next_line())
            .await
            .unwrap_or_else(|_| panic!("timed out waiting for {context}"))
            .expect("read JSON-RPC response")
            .expect("JSON-RPC output closed");
        serde_json::from_str(&line).expect("decode JSON-RPC response")
    }

    async fn write(&mut self, message: Value) {
        self.input
            .write_all(format!("{message}\n").as_bytes())
            .await
            .expect("write JSON-RPC message");
    }

    async fn shutdown(self) {
        drop(self.input);
        timeout(Duration::from_secs(2), self.runner)
            .await
            .expect("JSON lines runner should stop after input closes")
            .expect("run_json_lines task")
            .expect("run_json_lines result");
    }
}
