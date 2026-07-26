use std::sync::Arc;

use app_server::{AppServer, JsonRpcMessage, MockBackend, ProjectionStore, RuntimeCore};
use app_server_protocol::protocol::v2::METHOD_THREAD_COMPACT_START;
use app_server_protocol::{
    error_codes, METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_START, PROTOCOL_VERSION,
};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::time::{timeout, Duration};

#[tokio::test]
async fn thread_compact_start_uses_the_public_jsonrpc_contract() {
    let (_temp, server) = test_server();
    initialize_server(&server).await;

    let started = request(
        &server,
        2,
        METHOD_THREAD_START,
        json!({
            "model": "fixture-model",
            "modelProvider": "fixture-provider",
            "cwd": "/tmp/thread-compact-jsonrpc"
        }),
    )
    .await;
    let thread_id = started
        .pointer("/result/thread/id")
        .and_then(Value::as_str)
        .expect("thread id");

    let mut outbound = server.subscribe_outbound_messages();
    let lines = request_lines(
        &server,
        3,
        METHOD_THREAD_COMPACT_START,
        json!({"threadId": thread_id}),
    )
    .await;
    assert_eq!(
        lines,
        vec![json!({
            "id": 3,
            "result": {}
        })]
    );

    let notifications = timeout(Duration::from_secs(3), async {
        let mut notifications = Vec::new();
        loop {
            let message = outbound
                .recv()
                .await
                .expect("outbound compaction lifecycle");
            let JsonRpcMessage::Notification(notification) = message else {
                continue;
            };
            let value = serde_json::to_value(notification)
                .expect("serialize compaction lifecycle notification");
            if value.pointer("/params/threadId") != Some(&json!(thread_id)) {
                continue;
            }
            let completed = value.get("method") == Some(&json!("turn/completed"));
            notifications.push(value);
            if completed {
                break notifications;
            }
        }
    })
    .await
    .expect("compaction lifecycle timeout");

    assert_eq!(
        notifications
            .iter()
            .map(|notification| notification["method"].as_str())
            .collect::<Vec<_>>(),
        vec![
            Some("thread/status/changed"),
            Some("turn/started"),
            Some("item/started"),
            Some("item/completed"),
            Some("thread/status/changed"),
            Some("turn/completed"),
        ]
    );
    assert_eq!(
        notifications[0].pointer("/params/status"),
        Some(&json!({"type": "active", "activeFlags": []}))
    );
    assert_eq!(
        notifications[4].pointer("/params/status"),
        Some(&json!({"type": "idle"}))
    );
    let turn_id = notifications[1]
        .pointer("/params/turn/id")
        .and_then(Value::as_str)
        .expect("compaction turn id");
    let started_item = notifications[2]
        .pointer("/params/item")
        .expect("started compaction item");
    let completed_item = notifications[3]
        .pointer("/params/item")
        .expect("completed compaction item");
    assert_eq!(started_item["type"], "contextCompaction");
    assert_eq!(completed_item["type"], "contextCompaction");
    assert_eq!(started_item["id"], completed_item["id"]);
    assert_eq!(
        notifications[2].pointer("/params/turnId"),
        Some(&json!(turn_id))
    );
    assert_eq!(
        notifications[3].pointer("/params/turnId"),
        Some(&json!(turn_id))
    );
    assert_eq!(
        notifications[5].pointer("/params/turn/id"),
        Some(&json!(turn_id))
    );
}

#[tokio::test]
async fn retired_agent_session_compact_is_not_a_production_method() {
    let (_temp, server) = test_server();
    initialize_server(&server).await;

    let response = request_raw(
        &server,
        2,
        "agentSession/compact",
        json!({"sessionId": "retired-session"}),
    )
    .await;

    assert_eq!(
        response.pointer("/error/code"),
        Some(&json!(error_codes::METHOD_NOT_FOUND))
    );
    assert!(response.get("result").is_none());
}

#[tokio::test]
async fn thread_compact_start_rejects_missing_malformed_and_empty_thread_ids() {
    let (_temp, server) = test_server();
    initialize_server(&server).await;

    for (id, params) in [
        (2, json!({})),
        (3, json!({"threadId": null})),
        (4, json!({"threadId": 42})),
    ] {
        let response = request_raw(&server, id, METHOD_THREAD_COMPACT_START, params).await;
        assert_eq!(
            response.pointer("/error/code"),
            Some(&json!(error_codes::INVALID_PARAMS))
        );
        assert!(response.get("result").is_none());
    }

    for (id, thread_id) in [(5, ""), (6, "   ")] {
        let response = request_raw(
            &server,
            id,
            METHOD_THREAD_COMPACT_START,
            json!({"threadId": thread_id}),
        )
        .await;
        assert!(response.get("error").is_some());
        assert!(response.get("result").is_none());
    }
}

fn test_server() -> (TempDir, AppServer) {
    let temp = TempDir::new().expect("thread compact JSON-RPC temp dir");
    let projection_store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("thread compact JSON-RPC projection store"),
    );
    let runtime =
        RuntimeCore::with_backend(Arc::new(MockBackend)).with_projection_store(projection_store);
    (temp, AppServer::with_runtime(runtime))
}

async fn initialize_server(server: &AppServer) {
    let response = request(
        server,
        1,
        METHOD_INITIALIZE,
        json!({
            "clientInfo": {
                "name": "thread-compact-jsonrpc-test",
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
    let responses = request_lines(server, id, method, params)
        .await
        .into_iter()
        .filter(|value| value.get("id") == Some(&json!(id)))
        .collect::<Vec<_>>();
    assert_eq!(responses.len(), 1, "{method} must return one response");
    responses.into_iter().next().expect("JSON-RPC response")
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
        .map(|line| serde_json::from_str::<Value>(line).expect("decode JSON-RPC response"))
        .collect()
}
