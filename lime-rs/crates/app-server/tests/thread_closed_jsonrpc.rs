use std::sync::Arc;
use std::time::Duration;

use app_server::{run_json_lines, AppServer, MockBackend, ProjectionStore, RuntimeCore};
use app_server_protocol::protocol::v2::{
    METHOD_THREAD_CLOSED, METHOD_THREAD_LOADED_LIST, METHOD_THREAD_STATUS_CHANGED,
    METHOD_THREAD_UNSUBSCRIBE,
};
use app_server_protocol::{METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_START};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, DuplexStream, Lines};
use tokio::time::timeout;

#[tokio::test]
async fn idle_unsubscribed_thread_unloads_before_broadcasting_closed() {
    let temp = TempDir::new().expect("thread closed temp dir");
    let runtime = RuntimeCore::with_backend(Arc::new(MockBackend)).with_projection_store(Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("thread closed projection store"),
    ));
    let server =
        AppServer::with_runtime(runtime).with_thread_unloading_delay(Duration::from_millis(25));
    let (mut input_client, input_server) = tokio::io::duplex(16 * 1024);
    let (output_server, output_client) = tokio::io::duplex(16 * 1024);
    let runner = tokio::spawn(run_json_lines(server, input_server, output_server));
    let mut output_lines = BufReader::new(output_client).lines();

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": METHOD_INITIALIZE,
            "params": {
                "clientInfo": {
                    "name": "thread-closed-jsonrpc-test",
                    "version": "1.0.0"
                }
            }
        }),
    )
    .await;
    let initialized = read_response(&mut output_lines, 1).await;
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
    let started = read_response(&mut output_lines, 2).await;
    let thread_id = started
        .pointer("/result/thread/id")
        .and_then(Value::as_str)
        .expect("thread/start id")
        .to_string();

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": METHOD_THREAD_UNSUBSCRIBE,
            "params": {"threadId": thread_id}
        }),
    )
    .await;
    let unsubscribed = read_response(&mut output_lines, 3).await;
    assert_eq!(unsubscribed["result"], json!({"status": "unsubscribed"}));

    let status = read_notification(&mut output_lines, METHOD_THREAD_STATUS_CHANGED).await;
    assert_eq!(status["params"]["threadId"], json!(thread_id));
    assert_eq!(status["params"]["status"], json!({"type": "notLoaded"}));
    let closed = read_notification(&mut output_lines, METHOD_THREAD_CLOSED).await;
    assert_eq!(closed["params"], json!({"threadId": thread_id}));

    write_message(
        &mut input_client,
        json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": METHOD_THREAD_LOADED_LIST,
            "params": {}
        }),
    )
    .await;
    let loaded = read_response(&mut output_lines, 4).await;
    assert_eq!(loaded["result"]["data"], json!([]));

    drop(input_client);
    timeout(Duration::from_secs(2), runner)
        .await
        .expect("JSON lines runner should stop after input closes")
        .expect("JSON lines runner task")
        .expect("JSON lines runner result");
}

async fn write_message(input: &mut DuplexStream, message: Value) {
    input
        .write_all(format!("{message}\n").as_bytes())
        .await
        .expect("write JSON-RPC message");
}

async fn read_response(output: &mut Lines<BufReader<DuplexStream>>, id: u64) -> Value {
    loop {
        let message = read_message(output).await;
        if message["id"] == json!(id) {
            return message;
        }
    }
}

async fn read_notification(output: &mut Lines<BufReader<DuplexStream>>, method: &str) -> Value {
    loop {
        let message = read_message(output).await;
        if message["method"] == json!(method) {
            return message;
        }
    }
}

async fn read_message(output: &mut Lines<BufReader<DuplexStream>>) -> Value {
    let line = timeout(Duration::from_secs(2), output.next_line())
        .await
        .expect("JSON-RPC message timeout")
        .expect("read JSON-RPC message")
        .expect("JSON-RPC output closed");
    serde_json::from_str(&line).expect("decode JSON-RPC message")
}
