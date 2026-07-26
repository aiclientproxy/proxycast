use std::sync::Arc;

use agent_protocol::ThreadId;
use app_server::{AppServer, MockBackend, ProjectionStore, RuntimeCore};
use app_server_protocol::protocol::v2::{
    METHOD_THREAD_COMPACT_START, METHOD_THREAD_MEMORY_MODE_SET, METHOD_THREAD_SETTINGS_UPDATE,
    METHOD_THREAD_SHELL_COMMAND,
};
use app_server_protocol::{
    error_codes, METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_READ, METHOD_THREAD_START,
    METHOD_TURN_START, METHOD_TURN_STEER, PROTOCOL_VERSION,
};
use serde_json::{json, Value};
use thread_store::{AgentGraphStore, ThreadSpawnEdgeStatus};

#[tokio::test]
async fn parent_owned_thread_projects_policy_and_rejects_direct_turn_input() {
    let temp = tempfile::tempdir().expect("direct-input policy temp dir");
    let store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("direct-input policy projection store"),
    );
    let runtime =
        RuntimeCore::with_backend(Arc::new(MockBackend)).with_projection_store(Arc::clone(&store));
    let server = AppServer::with_runtime(runtime);
    initialize_server(&server).await;

    let parent_thread_id = start_thread(&server, 2).await;
    let child_thread_id = start_thread(&server, 3).await;
    store
        .upsert_thread_spawn_edge(
            ThreadId::new(parent_thread_id.clone()),
            ThreadId::new(child_thread_id.clone()),
            ThreadSpawnEdgeStatus::Open,
        )
        .await
        .expect("persist canonical spawn edge");

    let parent_read = request(
        &server,
        4,
        METHOD_THREAD_READ,
        json!({"threadId": parent_thread_id, "includeTurns": false}),
    )
    .await;
    assert_eq!(
        parent_read.pointer("/result/thread/canAcceptDirectInput"),
        Some(&json!(true))
    );

    let child_read = request(
        &server,
        5,
        METHOD_THREAD_READ,
        json!({"threadId": child_thread_id, "includeTurns": false}),
    )
    .await;
    assert_eq!(
        child_read.pointer("/result/thread/parentThreadId"),
        Some(&json!(parent_thread_id))
    );
    assert_eq!(
        child_read.pointer("/result/thread/canAcceptDirectInput"),
        Some(&json!(false))
    );

    for (id, method, params) in [
        (
            6,
            METHOD_TURN_START,
            json!({
                "threadId": child_thread_id,
                "input": [{"type": "text", "text": "do not submit directly"}]
            }),
        ),
        (
            7,
            METHOD_TURN_STEER,
            json!({
                "threadId": child_thread_id,
                "expectedTurnId": "turn-parent-owned",
                "input": [{"type": "text", "text": "do not steer directly"}]
            }),
        ),
        (
            8,
            METHOD_THREAD_COMPACT_START,
            json!({"threadId": child_thread_id}),
        ),
        (
            9,
            METHOD_THREAD_SETTINGS_UPDATE,
            json!({"threadId": child_thread_id, "model": "other-model"}),
        ),
        (
            10,
            METHOD_THREAD_MEMORY_MODE_SET,
            json!({"threadId": child_thread_id, "mode": "enabled"}),
        ),
        (
            11,
            METHOD_THREAD_SHELL_COMMAND,
            json!({"threadId": child_thread_id, "command": "pwd"}),
        ),
    ] {
        let response = request_raw(&server, id, method, params).await;
        assert_eq!(
            response.pointer("/error/code"),
            Some(&json!(error_codes::INVALID_REQUEST))
        );
        assert_eq!(
            response.pointer("/error/message"),
            Some(&json!(
                "direct app-server input is not allowed for parent-owned threads"
            ))
        );
        assert!(response.get("result").is_none());
    }
}

async fn start_thread(server: &AppServer, id: u64) -> String {
    request(
        server,
        id,
        METHOD_THREAD_START,
        json!({
            "model": "fixture-model",
            "modelProvider": "fixture-provider"
        }),
    )
    .await
    .pointer("/result/thread/id")
    .and_then(Value::as_str)
    .expect("thread id")
    .to_string()
}

async fn initialize_server(server: &AppServer) {
    let response = request(
        server,
        1,
        METHOD_INITIALIZE,
        json!({
            "clientInfo": {
                "name": "thread-direct-input-policy-jsonrpc-test",
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
        .find(|value| value.get("id") == Some(&json!(id)))
        .expect("JSON-RPC response")
}
