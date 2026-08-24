use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use app_server::{
    run_json_lines, ActionRespondRequest, AppServer, AppServerError, CancelExecutionRequest,
    ExecutionBackend, ExecutionRequest, ProjectionStore, RuntimeCore, RuntimeCoreError,
    RuntimeEvent, RuntimeEventSink,
};
use app_server_protocol::{
    METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_RESUME, METHOD_THREAD_START,
    METHOD_TURN_START,
};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, DuplexStream, Lines};
use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tokio::time::{timeout, Duration};

struct BlockingBackend {
    started: Arc<Notify>,
    release: Arc<Notify>,
    turn_count: AtomicUsize,
}

#[async_trait::async_trait]
impl ExecutionBackend for BlockingBackend {
    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.turn_count.fetch_add(1, Ordering::SeqCst);
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
async fn strict_review_notification_uses_public_transport_and_connection_gate() {
    let temp = TempDir::new().expect("strict review temp");
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let runtime = RuntimeCore::with_backend(Arc::new(BlockingBackend {
        started: Arc::clone(&started),
        release: Arc::clone(&release),
        turn_count: AtomicUsize::new(0),
    }))
    .with_projection_store(Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("strict review projection store"),
    ));
    let server = AppServer::with_runtime(runtime);

    let mut denied = TransportClient::start(server.clone(), "strict-review-denied", false).await;
    let mut enabled = TransportClient::start(server.clone(), "strict-review-enabled", true).await;
    let started_thread = enabled
        .request_ok(
            2,
            METHOD_THREAD_START,
            json!({"model": "fixture-model", "modelProvider": "fixture-provider"}),
        )
        .await;
    let thread = &started_thread["result"]["thread"];
    let thread_id = thread["id"].as_str().expect("thread id").to_string();
    let session_id = thread["sessionId"]
        .as_str()
        .expect("session id")
        .to_string();
    denied
        .request_ok(
            2,
            METHOD_THREAD_RESUME,
            json!({"threadId": thread_id, "excludeTurns": true}),
        )
        .await;
    let turn = enabled
        .request_ok(
            3,
            METHOD_TURN_START,
            json!({"threadId": thread_id, "input": [{"type": "text", "text": "review"}]}),
        )
        .await;
    let turn_id = turn["result"]["turn"]["id"]
        .as_str()
        .expect("turn id")
        .to_string();
    timeout(Duration::from_secs(2), started.notified())
        .await
        .expect("blocking backend should start");

    server
        .append_external_runtime_events(
            &session_id,
            Some(&turn_id),
            vec![RuntimeEvent::new(
                "guardian.review.started",
                json!({
                    "reviewId": "strict-review-1",
                    "targetItemId": "item-command-1",
                    "startedAtMs": 1_783_814_400_100i64,
                    "action": {
                        "type": "command",
                        "source": "shell",
                        "command": "git status --short",
                        "cwd": "/workspace"
                    }
                }),
            )],
        )
        .await
        .expect("append guardian review event");

    let item_started = enabled
        .take_notification("item/autoApprovalReview/started")
        .await;
    assert_eq!(item_started["params"]["threadId"], json!(thread_id));
    assert_eq!(item_started["params"]["turnId"], json!(turn_id));
    assert_eq!(item_started["params"]["review"]["status"], "inProgress");
    let strict = enabled
        .take_notification("autoApprovalReview/strictReviewRequired")
        .await;
    assert_eq!(strict["params"]["threadId"], json!(thread_id));
    assert_eq!(strict["params"]["turnId"], json!(turn_id));
    assert_eq!(strict["params"]["startedAtMs"], 1_783_814_400_100i64);

    assert!(timeout(
        Duration::from_millis(300),
        denied.take_notification("autoApprovalReview/strictReviewRequired")
    )
    .await
    .is_err());

    release.notify_one();
    let _ = enabled.take_notification("turn/completed").await;
    enabled.shutdown().await;
    denied.shutdown().await;
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
        self.write(json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))
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
                .expect("pending notification");
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
            .expect("JSON lines runner should stop")
            .expect("run_json_lines task")
            .expect("run_json_lines result");
    }
}
