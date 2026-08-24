use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use app_server::{
    run_json_lines, ActionRespondRequest, AppServer, AppServerError, CancelExecutionRequest,
    EventLogWriter, ExecutionBackend, ExecutionRequest, ProjectionStore, RuntimeCore,
    RuntimeCoreError, RuntimeEvent, RuntimeEventSink, SidecarStore,
};
use app_server_protocol::protocol::v2::{
    METHOD_THREAD_QUEUE_ADD, METHOD_THREAD_QUEUE_CHANGED, METHOD_THREAD_QUEUE_DELETE,
    METHOD_THREAD_QUEUE_LIST, METHOD_THREAD_QUEUE_REORDER, METHOD_THREAD_QUEUE_START,
    METHOD_THREAD_QUEUE_UPDATE, METHOD_TURN_COMPLETED,
};
use app_server_protocol::{
    error_codes, METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_RESUME, METHOD_THREAD_START,
    METHOD_TURN_INTERRUPT, METHOD_TURN_START,
};
use async_trait::async_trait;
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, DuplexStream, Lines};
use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tokio::time::{sleep, timeout, Duration};

const INLINE_PNG_DATA_URL: &str =
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==";

struct HoldFirstBackend {
    first_started: Arc<Notify>,
    release_first: Arc<Notify>,
    start_count: AtomicUsize,
    turn_ids: Mutex<Vec<String>>,
}

impl HoldFirstBackend {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            first_started: Arc::new(Notify::new()),
            release_first: Arc::new(Notify::new()),
            start_count: AtomicUsize::new(0),
            turn_ids: Mutex::new(Vec::new()),
        })
    }

    async fn wait_for_first_start(&self) {
        timeout(Duration::from_secs(2), self.first_started.notified())
            .await
            .expect("first backend turn should start");
    }

    async fn wait_for_start_count(&self, expected: usize) {
        timeout(Duration::from_secs(2), async {
            loop {
                if self.start_count.load(Ordering::SeqCst) >= expected {
                    break;
                }
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("backend should start {expected} turns"));
    }
}

#[async_trait]
impl ExecutionBackend for HoldFirstBackend {
    async fn start_turn(
        &self,
        request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        let index = self.start_count.fetch_add(1, Ordering::SeqCst);
        self.turn_ids
            .lock()
            .expect("turn id mutex poisoned")
            .push(request.turn.turn_id.clone());
        sink.emit(RuntimeEvent::new("turn.started", json!({})))?;
        if index == 0 {
            self.first_started.notify_one();
            self.release_first.notified().await;
            sink.emit(RuntimeEvent::new("turn.completed", json!({})))?;
        } else {
            self.release_first.notified().await;
        }
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

#[tokio::test]
async fn thread_queue_methods_require_experimental_api() {
    let mut client = TransportClient::start(
        AppServer::with_runtime(RuntimeCore::default()),
        "thread-queue-gate-test",
        false,
    )
    .await;

    for (id, method, params) in [
        (2, METHOD_THREAD_QUEUE_ADD, json!({})),
        (3, METHOD_THREAD_QUEUE_LIST, json!({})),
        (4, METHOD_THREAD_QUEUE_UPDATE, json!({})),
        (5, METHOD_THREAD_QUEUE_DELETE, json!({})),
        (6, METHOD_THREAD_QUEUE_REORDER, json!({})),
        (7, METHOD_THREAD_QUEUE_START, json!({})),
    ] {
        let response = client.request_raw(id, method, params).await;
        assert_eq!(response["error"]["code"], error_codes::INVALID_REQUEST);
        assert_eq!(
            response["error"]["message"],
            "thread queue methods require initialize capabilities.experimentalApi"
        );
    }

    client.shutdown().await;
}

#[tokio::test]
async fn thread_queue_public_jsonrpc_covers_crud_pagination_notifications_and_fifo_start() {
    let temp = TempDir::new().expect("thread queue CRUD tempdir");
    let projection_store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("thread queue CRUD projection store"),
    );
    let backend = HoldFirstBackend::new();
    let mut client = TransportClient::start(
        AppServer::with_runtime(
            RuntimeCore::with_backend(backend.clone()).with_projection_store(projection_store),
        ),
        "thread-queue-jsonrpc-test",
        true,
    )
    .await;
    let thread_id = start_thread(&mut client, 2).await;
    let active_turn_id = start_turn(&mut client, 3, &thread_id, "hold the active turn").await;
    backend.wait_for_first_start().await;

    let mut ids = Vec::new();
    for index in 0..30_u64 {
        let client_id = if index < 2 {
            "duplicate-client".to_string()
        } else {
            format!("client-{index}")
        };
        let response = client
            .request_ok(
                10 + index,
                METHOD_THREAD_QUEUE_ADD,
                json!({
                    "threadId": thread_id,
                    "input": [{"type": "text", "text": format!("queued-{index}")}],
                    "clientUserMessageId": client_id,
                }),
            )
            .await;
        ids.push(required_string(
            &response,
            "/result/queuedSubmission/id",
            "queued submission id",
        ));
    }
    assert_ne!(ids[0], ids[1]);

    let changed = client.take_notification(METHOD_THREAD_QUEUE_CHANGED).await;
    assert_eq!(changed["params"], json!({"threadId": thread_id}));

    let first_page = client
        .request_ok(50, METHOD_THREAD_QUEUE_LIST, json!({"threadId": thread_id}))
        .await;
    assert_eq!(
        first_page["result"]["data"].as_array().map(Vec::len),
        Some(25)
    );
    assert_eq!(first_page["result"]["nextCursor"], "25");
    assert_eq!(
        first_page["result"]["data"][0]["clientUserMessageId"],
        "duplicate-client"
    );
    assert_eq!(
        first_page["result"]["data"][1]["clientUserMessageId"],
        "duplicate-client"
    );

    let second_page = client
        .request_ok(
            51,
            METHOD_THREAD_QUEUE_LIST,
            json!({"threadId": thread_id, "cursor": "25", "limit": 1_000}),
        )
        .await;
    assert_eq!(
        second_page["result"]["data"].as_array().map(Vec::len),
        Some(5)
    );
    assert_eq!(second_page["result"]["nextCursor"], Value::Null);

    let beyond_end = client
        .request_ok(
            52,
            METHOD_THREAD_QUEUE_LIST,
            json!({"threadId": thread_id, "cursor": usize::MAX.to_string()}),
        )
        .await;
    assert_eq!(
        beyond_end["result"],
        json!({"data": [], "nextCursor": null})
    );
    let invalid_cursor = client
        .request_raw(
            53,
            METHOD_THREAD_QUEUE_LIST,
            json!({"threadId": thread_id, "cursor": "not-a-cursor"}),
        )
        .await;
    assert_eq!(
        invalid_cursor["error"]["code"],
        error_codes::INVALID_REQUEST
    );

    let updated = client
        .request_ok(
            54,
            METHOD_THREAD_QUEUE_UPDATE,
            json!({
                "threadId": thread_id,
                "queuedSubmissionId": ids[1],
                "input": [{"type": "text", "text": "updated-second"}],
            }),
        )
        .await;
    assert_eq!(updated["result"]["queuedSubmission"]["id"], ids[1]);
    assert_eq!(
        updated["result"]["queuedSubmission"]["clientUserMessageId"],
        "duplicate-client"
    );

    let remote_image = client
        .request_raw(
            55,
            METHOD_THREAD_QUEUE_ADD,
            json!({
                "threadId": thread_id,
                "input": [{"type": "image", "url": "https://example.test/image.png"}],
                "clientUserMessageId": "remote-image",
            }),
        )
        .await;
    assert_eq!(remote_image["error"]["code"], error_codes::INVALID_REQUEST);
    assert_eq!(
        remote_image["error"]["message"],
        "remote image URLs are not supported; use an inline data URL instead"
    );

    ids.reverse();
    client
        .request_ok(
            56,
            METHOD_THREAD_QUEUE_REORDER,
            json!({"threadId": thread_id, "queuedSubmissionIds": ids}),
        )
        .await;
    let reordered = client
        .request_ok(
            57,
            METHOD_THREAD_QUEUE_LIST,
            json!({"threadId": thread_id, "limit": 100}),
        )
        .await;
    assert_eq!(reordered["result"]["data"][0]["id"], ids[0]);

    let deleted_id = ids.pop().expect("queue id to delete");
    let deleted = client
        .request_ok(
            58,
            METHOD_THREAD_QUEUE_DELETE,
            json!({"threadId": thread_id, "queuedSubmissionId": deleted_id}),
        )
        .await;
    assert_eq!(deleted["result"], json!({"deleted": true}));
    let deleted_again = client
        .request_ok(
            59,
            METHOD_THREAD_QUEUE_DELETE,
            json!({"threadId": thread_id, "queuedSubmissionId": deleted_id}),
        )
        .await;
    assert_eq!(deleted_again["result"], json!({"deleted": false}));

    let active_start = client
        .request_raw(
            60,
            METHOD_THREAD_QUEUE_START,
            json!({"threadId": thread_id}),
        )
        .await;
    assert_eq!(active_start["error"]["code"], error_codes::INVALID_REQUEST);
    let before_release = client
        .request_ok(
            61,
            METHOD_THREAD_QUEUE_LIST,
            json!({"threadId": thread_id, "limit": 100}),
        )
        .await;
    assert_eq!(
        before_release["result"]["data"].as_array().map(Vec::len),
        Some(29)
    );

    backend.release_first.notify_one();
    backend.wait_for_start_count(2).await;
    let started_ids = backend
        .turn_ids
        .lock()
        .expect("turn id mutex poisoned")
        .clone();
    assert_eq!(started_ids[0], active_turn_id);
    assert_eq!(started_ids[1], ids[0]);
    let after_completion = client
        .request_ok(
            62,
            METHOD_THREAD_QUEUE_LIST,
            json!({"threadId": thread_id, "limit": 100}),
        )
        .await;
    assert_eq!(
        after_completion["result"]["data"].as_array().map(Vec::len),
        Some(28)
    );

    client.shutdown().await;
}

#[tokio::test]
async fn interrupted_thread_keeps_queue_paused_until_explicit_start() {
    let temp = TempDir::new().expect("thread queue interrupt tempdir");
    let projection_store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("thread queue interrupt projection store"),
    );
    let backend = HoldFirstBackend::new();
    let mut client = TransportClient::start(
        AppServer::with_runtime(
            RuntimeCore::with_backend(backend.clone()).with_projection_store(projection_store),
        ),
        "thread-queue-interrupt-test",
        true,
    )
    .await;
    let thread_id = start_thread(&mut client, 2).await;
    let turn_id = start_turn(&mut client, 3, &thread_id, "interrupt this turn").await;
    backend.wait_for_first_start().await;
    let queued = client
        .request_ok(
            4,
            METHOD_THREAD_QUEUE_ADD,
            json!({
                "threadId": thread_id,
                "input": [{"type": "text", "text": "start explicitly later"}],
                "clientUserMessageId": "client-paused",
            }),
        )
        .await;
    let queued_id = required_string(
        &queued,
        "/result/queuedSubmission/id",
        "paused queued submission id",
    );

    client
        .request_ok(
            5,
            METHOD_TURN_INTERRUPT,
            json!({"threadId": thread_id, "turnId": turn_id}),
        )
        .await;
    let completed = client.take_notification(METHOD_TURN_COMPLETED).await;
    assert_eq!(completed["params"]["turn"]["status"], "interrupted");
    client
        .request_ok(6, METHOD_THREAD_RESUME, json!({"threadId": thread_id}))
        .await;
    sleep(Duration::from_millis(50)).await;
    assert_eq!(backend.start_count.load(Ordering::SeqCst), 1);
    let paused = client
        .request_ok(7, METHOD_THREAD_QUEUE_LIST, json!({"threadId": thread_id}))
        .await;
    assert_eq!(paused["result"]["data"].as_array().map(Vec::len), Some(1));

    let started = client
        .request_ok(
            8,
            METHOD_THREAD_QUEUE_START,
            json!({"threadId": thread_id, "queuedSubmissionId": queued_id}),
        )
        .await;
    assert_eq!(started["result"]["turn"]["id"], queued_id);
    backend.wait_for_start_count(2).await;
    let empty = client
        .request_ok(9, METHOD_THREAD_QUEUE_LIST, json!({"threadId": thread_id}))
        .await;
    assert_eq!(empty["result"], json!({"data": [], "nextCursor": null}));

    client.shutdown().await;
}

#[tokio::test]
async fn thread_queue_public_jsonrpc_restores_multimodal_input_after_cold_restart() {
    let temp = TempDir::new().expect("thread queue restart temp");
    let event_log_root = temp.path().join("event-log");
    let projection_path = temp.path().join("projection.sqlite");
    let sidecar_root = temp.path().join("sidecar");
    let event_log_writer = Arc::new(EventLogWriter::new(&event_log_root).expect("event writer"));
    let projection_store =
        Arc::new(ProjectionStore::initialize(&projection_path).expect("projection store"));
    let sidecar_store = Arc::new(SidecarStore::new(&sidecar_root).expect("sidecar store"));
    let backend = HoldFirstBackend::new();
    let runtime = RuntimeCore::with_backend(backend.clone())
        .with_event_log_writer(event_log_writer.clone())
        .with_projection_store(projection_store.clone())
        .with_sidecar_store(sidecar_store.clone());
    let mut client = TransportClient::start(
        AppServer::with_runtime(runtime),
        "thread-queue-restart-source",
        true,
    )
    .await;
    let thread_id = start_thread(&mut client, 2).await;
    start_turn(&mut client, 3, &thread_id, "keep queue pending").await;
    backend.wait_for_first_start().await;
    let first = client
        .request_ok(
            4,
            METHOD_THREAD_QUEUE_ADD,
            json!({
                "threadId": thread_id,
                "input": [{"type": "text", "text": "first"}],
                "clientUserMessageId": "client-first",
            }),
        )
        .await;
    let second = client
        .request_ok(
            5,
            METHOD_THREAD_QUEUE_ADD,
            json!({
                "threadId": thread_id,
                "input": [
                    {"type": "text", "text": "inspect after restart"},
                    {"type": "image", "url": INLINE_PNG_DATA_URL, "detail": "high"}
                ],
                "clientUserMessageId": "client-image",
            }),
        )
        .await;
    let first_id = required_string(
        &first,
        "/result/queuedSubmission/id",
        "first queued submission id",
    );
    let second_id = required_string(
        &second,
        "/result/queuedSubmission/id",
        "second queued submission id",
    );
    client
        .request_ok(
            6,
            METHOD_THREAD_QUEUE_REORDER,
            json!({
                "threadId": thread_id,
                "queuedSubmissionIds": [second_id, first_id],
            }),
        )
        .await;
    client.shutdown().await;

    let restarted_runtime = RuntimeCore::default()
        .with_event_log_writer(event_log_writer)
        .with_projection_store(projection_store)
        .with_sidecar_store(sidecar_store);
    let mut restarted = TransportClient::start(
        AppServer::with_runtime(restarted_runtime),
        "thread-queue-restart-target",
        true,
    )
    .await;
    let listed = restarted
        .request_ok(
            2,
            METHOD_THREAD_QUEUE_LIST,
            json!({"threadId": thread_id, "limit": 100}),
        )
        .await;
    assert_eq!(listed["result"]["data"][0]["id"], second_id);
    assert_eq!(listed["result"]["data"][1]["id"], first_id);
    assert_eq!(
        listed["result"]["data"][0]["clientUserMessageId"],
        "client-image"
    );
    assert_eq!(
        listed["result"]["data"][0]["input"][0]["text"],
        "inspect after restart"
    );
    assert_eq!(listed["result"]["data"][0]["input"][1]["detail"], "high");
    assert!(listed["result"]["data"][0]["input"][1]["url"]
        .as_str()
        .is_some_and(|url| url.starts_with("sidecar://media/input-")));

    restarted.shutdown().await;
}

#[tokio::test]
async fn cold_thread_resume_dispatches_persisted_queued_submission() {
    let temp = TempDir::new().expect("thread queue cold resume temp");
    let event_log_writer =
        Arc::new(EventLogWriter::new(temp.path().join("event-log")).expect("event writer"));
    let projection_store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store"),
    );
    let source_runtime = RuntimeCore::default()
        .with_event_log_writer(event_log_writer.clone())
        .with_projection_store(projection_store.clone());
    let mut source = TransportClient::start(
        AppServer::with_runtime(source_runtime),
        "thread-queue-cold-resume-source",
        true,
    )
    .await;
    let thread_id = start_thread(&mut source, 2).await;
    source.shutdown().await;

    let backend = HoldFirstBackend::new();
    let resumed_runtime = RuntimeCore::with_backend(backend.clone())
        .with_event_log_writer(event_log_writer)
        .with_projection_store(projection_store);
    let mut resumed = TransportClient::start(
        AppServer::with_runtime(resumed_runtime),
        "thread-queue-cold-resume-target",
        true,
    )
    .await;
    let queued = resumed
        .request_ok(
            2,
            METHOD_THREAD_QUEUE_ADD,
            json!({
                "threadId": thread_id,
                "input": [{"type": "text", "text": "dispatch after cold resume"}],
                "clientUserMessageId": "client-cold-resume",
            }),
        )
        .await;
    let queued_id = required_string(
        &queued,
        "/result/queuedSubmission/id",
        "cold-resume queued submission id",
    );
    let before_resume = resumed
        .request_ok(3, METHOD_THREAD_QUEUE_LIST, json!({"threadId": thread_id}))
        .await;
    assert_eq!(before_resume["result"]["data"][0]["id"], queued_id);
    assert_eq!(backend.start_count.load(Ordering::SeqCst), 0);

    resumed
        .request_ok(4, METHOD_THREAD_RESUME, json!({"threadId": thread_id}))
        .await;
    backend.wait_for_start_count(1).await;
    assert_eq!(
        backend
            .turn_ids
            .lock()
            .expect("turn id mutex poisoned")
            .as_slice(),
        [queued_id.as_str()]
    );
    let after_resume = resumed
        .request_ok(5, METHOD_THREAD_QUEUE_LIST, json!({"threadId": thread_id}))
        .await;
    assert_eq!(
        after_resume["result"],
        json!({"data": [], "nextCursor": null})
    );

    backend.release_first.notify_one();
    let completed = resumed.take_notification(METHOD_TURN_COMPLETED).await;
    assert_eq!(completed["params"]["turn"]["id"], queued_id);
    resumed.shutdown().await;
}

async fn start_thread(client: &mut TransportClient, id: u64) -> String {
    let response = client
        .request_ok(
            id,
            METHOD_THREAD_START,
            json!({"model": "fixture-model", "modelProvider": "fixture-provider"}),
        )
        .await;
    required_string(&response, "/result/thread/id", "thread id")
}

async fn start_turn(client: &mut TransportClient, id: u64, thread_id: &str, text: &str) -> String {
    let response = client
        .request_ok(
            id,
            METHOD_TURN_START,
            json!({
                "threadId": thread_id,
                "input": [{"type": "text", "text": text}],
                "model": "fixture-model",
                "approvalPolicy": "never",
                "sandboxPolicy": "workspace-write",
            }),
        )
        .await;
    required_string(&response, "/result/turn/id", "turn id")
}

fn required_string(value: &Value, pointer: &str, label: &str) -> String {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("{label} missing at {pointer}: {value:#}"))
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
        let (input, input_server) = tokio::io::duplex(128 * 1024);
        let (output_server, output) = tokio::io::duplex(128 * 1024);
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
            "{method} returned an error: {response:#}"
        );
        assert!(
            response.get("result").is_some(),
            "{method} returned no result"
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

    async fn next_wire_message(&mut self, scenario: &str) -> Value {
        let line = timeout(Duration::from_secs(5), self.lines.next_line())
            .await
            .unwrap_or_else(|_| panic!("timed out waiting for JSON-RPC message: {scenario}"))
            .expect("read JSON-RPC message")
            .expect("JSON-RPC output closed");
        serde_json::from_str(&line).expect("decode JSON-RPC message")
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
            .expect("JSON lines runner task")
            .expect("JSON lines runner result");
    }
}
