use std::sync::Arc;
use std::time::Duration;

use app_server::{run_json_lines, AppServer, MockBackend, ProjectionStore, RuntimeCore};
use app_server_protocol::protocol::v2::{METHOD_THREAD_LOADED_LIST, METHOD_THREAD_UNSUBSCRIBE};
use app_server_protocol::{METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_START};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, DuplexStream, Lines};
use tokio::time::timeout;
use uuid::Uuid;

#[tokio::test]
async fn thread_unsubscribe_is_connection_scoped_and_keeps_thread_loaded() {
    let temp = TempDir::new().expect("thread unsubscribe temp dir");
    let runtime = RuntimeCore::with_backend(Arc::new(MockBackend)).with_projection_store(Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("thread unsubscribe projection store"),
    ));
    let server = AppServer::with_runtime(runtime);
    let (mut input_client, input_server) = tokio::io::duplex(16 * 1024);
    let (output_server, output_client) = tokio::io::duplex(16 * 1024);
    let runner = tokio::spawn(run_json_lines(server, input_server, output_server));
    let mut output_lines = BufReader::new(output_client).lines();
    let mut notifications = Vec::new();

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": METHOD_INITIALIZE,
            "params": {
                "clientInfo": {
                    "name": "thread-unsubscribe-jsonrpc-test",
                    "version": "1.0.0"
                }
            }
        }),
    )
    .await;
    let initialized = read_response(&mut output_lines, 1, &mut notifications).await;
    assert!(initialized.get("error").is_none(), "{initialized:#?}");
    write_message(
        &mut input_client,
        json!({"jsonrpc": "2.0", "method": METHOD_INITIALIZED, "params": {}}),
    )
    .await;

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": METHOD_THREAD_START,
            "params": {
                "model": "fixture-model",
                "modelProvider": "fixture-provider"
            }
        }),
    )
    .await;
    let started = read_response(&mut output_lines, 2, &mut notifications).await;
    let thread_id = started
        .pointer("/result/thread/id")
        .and_then(Value::as_str)
        .expect("thread/start id")
        .to_string();

    let first = unsubscribe(
        &mut input_client,
        &mut output_lines,
        3,
        &thread_id,
        &mut notifications,
    )
    .await;
    assert_eq!(first["result"], json!({"status": "unsubscribed"}));

    let second = unsubscribe(
        &mut input_client,
        &mut output_lines,
        4,
        &thread_id,
        &mut notifications,
    )
    .await;
    assert_eq!(second["result"], json!({"status": "notSubscribed"}));

    let cold_thread_id = Uuid::now_v7().to_string();
    let cold = unsubscribe(
        &mut input_client,
        &mut output_lines,
        5,
        &cold_thread_id,
        &mut notifications,
    )
    .await;
    assert_eq!(cold["result"], json!({"status": "notLoaded"}));

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 6,
            "method": METHOD_THREAD_LOADED_LIST,
            "params": {}
        }),
    )
    .await;
    let loaded = read_response(&mut output_lines, 6, &mut notifications).await;
    assert_eq!(loaded["result"]["data"], json!([thread_id]));
    assert!(!notifications.iter().any(|method| method == "thread/closed"));

    let next = timeout(Duration::from_millis(100), output_lines.next_line()).await;
    if let Ok(Ok(Some(line))) = next {
        let message: Value = serde_json::from_str(&line).expect("trailing JSON-RPC message");
        assert_ne!(message["method"], json!("thread/closed"));
    }

    drop(input_client);
    timeout(Duration::from_secs(2), runner)
        .await
        .expect("JSON lines runner should stop after input closes")
        .expect("JSON lines runner task")
        .expect("JSON lines runner result");
}

async fn unsubscribe(
    input: &mut DuplexStream,
    output: &mut Lines<BufReader<DuplexStream>>,
    id: u64,
    thread_id: &str,
    notifications: &mut Vec<String>,
) -> Value {
    write_message(
        input,
        json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": METHOD_THREAD_UNSUBSCRIBE,
            "params": {"threadId": thread_id}
        }),
    )
    .await;
    read_response(output, id, notifications).await
}

async fn write_message(input: &mut DuplexStream, message: Value) {
    input
        .write_all(format!("{message}\n").as_bytes())
        .await
        .expect("write JSON-RPC message");
}

async fn read_response(
    output: &mut Lines<BufReader<DuplexStream>>,
    id: u64,
    notifications: &mut Vec<String>,
) -> Value {
    loop {
        let line = timeout(Duration::from_secs(2), output.next_line())
            .await
            .expect("JSON-RPC response timeout")
            .expect("read JSON-RPC response")
            .expect("JSON-RPC output closed");
        let message: Value = serde_json::from_str(&line).expect("decode JSON-RPC response");
        if message["id"] == json!(id) {
            return message;
        }
        if let Some(method) = message.get("method").and_then(Value::as_str) {
            notifications.push(method.to_string());
        }
    }
}
