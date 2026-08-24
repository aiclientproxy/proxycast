use std::collections::VecDeque;

use app_server::{run_json_lines, AppServer, AppServerError, MockBackend, RuntimeCore};
use app_server_protocol::protocol::v2::{
    METHOD_ENVIRONMENT_ADD, METHOD_ENVIRONMENT_INFO, METHOD_ENVIRONMENT_STATUS,
};
use app_server_protocol::{error_codes, METHOD_INITIALIZE, METHOD_INITIALIZED};
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, DuplexStream, Lines};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio::time::{timeout, Duration};
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;

#[tokio::test]
async fn environment_methods_use_public_jsonrpc_and_local_owner() {
    let _temp = TempDir::new().expect("environment integration temp");
    let mut client = TransportClient::start(
        AppServer::with_runtime(RuntimeCore::with_backend(std::sync::Arc::new(MockBackend))),
        "environment-jsonrpc-test",
        true,
    )
    .await;

    let info = client
        .request_ok(
            2,
            METHOD_ENVIRONMENT_INFO,
            json!({"environmentId": "local"}),
        )
        .await;
    assert!(info
        .pointer("/result/shell/name")
        .is_some_and(Value::is_string));
    assert!(info
        .pointer("/result/shell/path")
        .is_some_and(Value::is_string));
    assert!(info
        .pointer("/result/cwd")
        .and_then(Value::as_str)
        .is_some_and(|cwd| cwd.starts_with("file://")));

    let status = client
        .request_ok(
            3,
            METHOD_ENVIRONMENT_STATUS,
            json!({"environmentId": "local"}),
        )
        .await;
    assert_eq!(status["result"], json!({"status": "ready"}));

    let unknown_status = client
        .request_ok(
            4,
            METHOD_ENVIRONMENT_STATUS,
            json!({"environmentId": "missing"}),
        )
        .await;
    assert_eq!(unknown_status["result"]["status"], "unknown");
    assert_eq!(
        unknown_status["result"]["error"],
        "environment 'missing' is not configured"
    );

    let unknown_info = client
        .request_raw(
            5,
            METHOD_ENVIRONMENT_INFO,
            json!({"environmentId": "missing"}),
        )
        .await;
    assert_eq!(unknown_info["error"]["code"], error_codes::INVALID_REQUEST);

    let remote_add = client
        .request_raw(
            6,
            METHOD_ENVIRONMENT_ADD,
            json!({
                "environmentId": "remote",
                "execServerUrl": "https://exec.example.test",
            }),
        )
        .await;
    assert_eq!(remote_add["error"]["code"], error_codes::INVALID_PARAMS);
    assert_eq!(
        remote_add["error"]["message"],
        "execServerUrl must be a ws:// or wss:// URL with a host"
    );

    client.shutdown().await;
}

#[tokio::test]
async fn environment_remote_exec_server_uses_websocket_registry() {
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("bind fixture");
    let address = listener.local_addr().expect("fixture address");
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept fixture");
        let mut socket = accept_async(stream).await.expect("websocket handshake");
        while let Some(message) = socket.next().await {
            let Message::Text(text) = message.expect("fixture websocket message") else {
                continue;
            };
            let request: Value = serde_json::from_str(&text).expect("fixture JSON-RPC request");
            let Some(id) = request.get("id") else {
                continue;
            };
            let result = match request["method"].as_str().expect("fixture method") {
                "initialize" => json!({"sessionId": "fixture-session"}),
                "environment/info" => json!({
                    "shell": {"name": "fixture-sh", "path": "/bin/fixture-sh"},
                    "cwd": "file:///tmp/fixture"
                }),
                "environment/status" => json!({"status": "ready"}),
                method => panic!("unexpected fixture method: {method}"),
            };
            socket
                .send(Message::Text(
                    json!({"jsonrpc": "2.0", "id": id, "result": result}).to_string(),
                ))
                .await
                .expect("send fixture response");
        }
    });

    let mut client = TransportClient::start(
        AppServer::with_runtime(RuntimeCore::with_backend(std::sync::Arc::new(MockBackend))),
        "environment-remote-jsonrpc-test",
        true,
    )
    .await;
    let add = client
        .request_ok(
            2,
            METHOD_ENVIRONMENT_ADD,
            json!({
                "environmentId": "fixture",
                "execServerUrl": format!("ws://{address}"),
            }),
        )
        .await;
    assert_eq!(add["result"], json!({}));

    let mut status = Value::Null;
    for id in 3..=20 {
        status = client
            .request_ok(
                id,
                METHOD_ENVIRONMENT_STATUS,
                json!({"environmentId": "fixture"}),
            )
            .await;
        if status["result"]["status"] == "ready" {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert_eq!(status["result"], json!({"status": "ready"}));

    let info = client
        .request_ok(
            21,
            METHOD_ENVIRONMENT_INFO,
            json!({"environmentId": "fixture"}),
        )
        .await;
    assert_eq!(
        info["result"]["shell"],
        json!({
            "name": "fixture-sh",
            "path": "/bin/fixture-sh"
        })
    );
    assert_eq!(info["result"]["cwd"], "file:///tmp/fixture");

    server.abort();
    let _ = server.await;
    let disconnected = client
        .request_ok(
            22,
            METHOD_ENVIRONMENT_STATUS,
            json!({"environmentId": "fixture"}),
        )
        .await;
    assert_eq!(disconnected["result"]["status"], "disconnected");
    client.shutdown().await;
}

#[tokio::test]
async fn environment_methods_fail_closed_without_experimental_api_capability() {
    let mut client = TransportClient::start(
        AppServer::with_runtime(RuntimeCore::with_backend(std::sync::Arc::new(MockBackend))),
        "environment-capability-gate-test",
        false,
    )
    .await;

    let response = client
        .request_raw(
            2,
            METHOD_ENVIRONMENT_STATUS,
            json!({"environmentId": "local"}),
        )
        .await;
    assert_eq!(response["error"]["code"], error_codes::INVALID_REQUEST);
    assert_eq!(
        response["error"]["message"],
        "environment methods require initialize capabilities.experimentalApi"
    );

    client.shutdown().await;
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
            let message = self.next_message(method).await;
            if message.get("id") == Some(&json!(id)) {
                return message;
            }
            self.pending_messages.push_back(message);
        }
    }

    async fn next_message(&mut self, scenario: &str) -> Value {
        if let Some(message) = self.pending_messages.pop_front() {
            return message;
        }
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
