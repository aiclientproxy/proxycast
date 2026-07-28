mod support;

use app_server::AppServer;
use app_server_protocol::protocol::v2::{METHOD_THREAD_ARCHIVE, METHOD_THREAD_SHELL_COMMAND};
use app_server_protocol::{
    error_codes, JsonRpcMessage, METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_READ,
    METHOD_THREAD_START, PROTOCOL_VERSION,
};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::time::{sleep, timeout, Duration};

#[tokio::test]
async fn shell_command_runs_through_jsonrpc_and_projects_user_shell_items() {
    let temp = TempDir::new().expect("shell JSON-RPC temp dir");
    let core = support::runtime_core_with_chat_provider(&temp, "provider-test", "model-test");
    let server = AppServer::with_runtime(core.clone());
    initialize_server(&server).await;

    let started = request(
        &server,
        2,
        METHOD_THREAD_START,
        json!({
            "model": "model-test",
            "modelProvider": "provider-test",
            "cwd": temp.path().to_string_lossy(),
            "historyMode": "paginated"
        }),
    )
    .await;
    let thread_id = started
        .pointer("/result/thread/id")
        .and_then(Value::as_str)
        .expect("thread id")
        .to_string();
    let session_id = started
        .pointer("/result/thread/sessionId")
        .and_then(Value::as_str)
        .expect("session id")
        .to_string();

    let empty = request_raw(
        &server,
        3,
        METHOD_THREAD_SHELL_COMMAND,
        json!({"threadId": thread_id, "command": "   "}),
    )
    .await;
    assert_eq!(
        empty.pointer("/error/code"),
        Some(&json!(error_codes::INVALID_PARAMS))
    );

    let unknown = request_raw(
        &server,
        4,
        METHOD_THREAD_SHELL_COMMAND,
        json!({"threadId": "missing", "command": "printf unreachable"}),
    )
    .await;
    assert_eq!(
        unknown.pointer("/error/code"),
        Some(&json!(error_codes::RUNTIME_ERROR))
    );

    let mut outbound = server.subscribe_outbound_messages();
    let shell_lines = request_lines(
        &server,
        5,
        METHOD_THREAD_SHELL_COMMAND,
        json!({
            "threadId": thread_id,
            "command": "printf jsonrpc-shell-ready"
        }),
    )
    .await;
    let shell_response = shell_lines
        .iter()
        .find(|value| value.get("id") == Some(&json!(5)))
        .expect("shell response");
    assert_eq!(shell_response.get("result"), Some(&json!({})));

    timeout(Duration::from_secs(3), async {
        loop {
            if core
                .events_for_session(&session_id)
                .expect("shell events")
                .iter()
                .any(|event| event.event_type == "turn.completed")
            {
                break;
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("shell completion timeout");

    let lifecycle_methods = timeout(Duration::from_secs(3), async {
        let mut methods = Vec::new();
        loop {
            let message = outbound.recv().await.expect("outbound shell lifecycle");
            let JsonRpcMessage::Notification(notification) = message else {
                continue;
            };
            let completed = notification.method == "item/completed";
            methods.push(notification.method);
            if completed {
                break methods;
            }
        }
    })
    .await
    .expect("outbound shell lifecycle timeout");
    assert!(lifecycle_methods
        .iter()
        .any(|method| method == "item/started"));
    assert!(lifecycle_methods
        .iter()
        .any(|method| method == "item/completed"));

    let read_lines = request_lines(
        &server,
        6,
        METHOD_THREAD_READ,
        json!({"threadId": thread_id, "includeTurns": true}),
    )
    .await;
    let read = read_lines
        .iter()
        .find(|value| value.get("id") == Some(&json!(6)))
        .expect("thread read response");
    let item = read
        .pointer("/result/thread/turns/0/items/0")
        .expect("shell item");
    assert_eq!(item["type"], "commandExecution");
    assert_eq!(item["source"], "userShell");
    assert_eq!(item["status"], "completed");
    assert_eq!(item["aggregatedOutput"], "jsonrpc-shell-ready");
    assert_eq!(item["exitCode"], 0);
    assert!(item["processId"].as_str().is_some());

    request(
        &server,
        7,
        METHOD_THREAD_ARCHIVE,
        json!({"threadId": thread_id}),
    )
    .await;
    let archived = request_raw(
        &server,
        8,
        METHOD_THREAD_SHELL_COMMAND,
        json!({"threadId": thread_id, "command": "printf archived"}),
    )
    .await;
    assert_eq!(
        archived.pointer("/error/code"),
        Some(&json!(error_codes::INVALID_PARAMS))
    );
}

async fn initialize_server(server: &AppServer) {
    let response = request(
        server,
        1,
        METHOD_INITIALIZE,
        json!({
            "clientInfo": {
                "name": "thread-shell-jsonrpc-test",
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

async fn request(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    let response = request_raw(server, id, method, params).await;
    if let Some(error) = response.get("error") {
        panic!("{method} failed: {error}");
    }
    response
}

async fn request_raw(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    request_lines(server, id, method, params)
        .await
        .into_iter()
        .find(|value| value.get("id") == Some(&json!(id)))
        .expect("JSON-RPC response")
}

async fn request_lines(server: &AppServer, id: u64, method: &str, params: Value) -> Vec<Value> {
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
        .expect("handle JSON-RPC request");
    lines
        .iter()
        .map(|line| serde_json::from_str(line).expect("decode JSON-RPC response"))
        .collect()
}
