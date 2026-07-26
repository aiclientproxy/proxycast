use std::sync::Arc;

use app_server::{AppServer, MockBackend, ProjectionStore, RuntimeCore};
use app_server_protocol::protocol::v2::METHOD_THREAD_APPROVE_GUARDIAN_DENIED_ACTION;
use app_server_protocol::{
    error_codes, METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_START,
};
use serde_json::{json, Value};
use tempfile::TempDir;
use uuid::Uuid;

#[tokio::test]
async fn guardian_denial_approval_validates_the_exact_event_and_loaded_thread() {
    let temp = TempDir::new().expect("Guardian JSON-RPC temp dir");
    let projection = ProjectionStore::initialize(temp.path().join("projection.sqlite"))
        .expect("Guardian projection store");
    let runtime = RuntimeCore::with_backend(Arc::new(MockBackend))
        .with_projection_store(Arc::new(projection));
    let server = AppServer::with_runtime(runtime);
    initialize(&server).await;

    let thread_id = start_thread(&server, 2).await;
    let response = request_raw(
        &server,
        3,
        METHOD_THREAD_APPROVE_GUARDIAN_DENIED_ACTION,
        json!({
            "threadId": thread_id,
            "event": denied_command_event()
        }),
    )
    .await;
    assert_eq!(response.get("result"), Some(&json!({})));

    let invalid_event = request_raw(
        &server,
        4,
        METHOD_THREAD_APPROVE_GUARDIAN_DENIED_ACTION,
        json!({
            "threadId": thread_id,
            "event": {
                "id": "guardian-invalid",
                "status": "denied",
                "action": {
                    "type": "command",
                    "source": "shell",
                    "command": "pwd"
                }
            }
        }),
    )
    .await;
    assert_eq!(
        invalid_event.pointer("/error/code"),
        Some(&json!(error_codes::INVALID_REQUEST))
    );

    let unknown_thread = request_raw(
        &server,
        5,
        METHOD_THREAD_APPROVE_GUARDIAN_DENIED_ACTION,
        json!({
            "threadId": Uuid::now_v7().to_string(),
            "event": denied_command_event()
        }),
    )
    .await;
    assert_eq!(
        unknown_thread.pointer("/error/code"),
        Some(&json!(error_codes::INVALID_REQUEST))
    );
}

fn denied_command_event() -> Value {
    json!({
        "id": "guardian-review-1",
        "target_item_id": "item-command-1",
        "turn_id": "turn-1",
        "started_at_ms": 1,
        "completed_at_ms": 2,
        "status": "denied",
        "risk_level": "high",
        "user_authorization": "low",
        "rationale": "The exact command was not previously authorized.",
        "decision_source": "agent",
        "action": {
            "type": "command",
            "source": "shell",
            "command": "git status --short",
            "cwd": "/workspace"
        }
    })
}

async fn initialize(server: &AppServer) {
    request_raw(
        server,
        1,
        METHOD_INITIALIZE,
        json!({"clientInfo": {"name": "thread-guardian-test", "version": "1"}}),
    )
    .await;
    server
        .handle_json_line(
            &json!({"jsonrpc": "2.0", "method": METHOD_INITIALIZED, "params": {}}).to_string(),
        )
        .await
        .expect("initialized notification");
}

async fn start_thread(server: &AppServer, id: u64) -> String {
    request_raw(
        server,
        id,
        METHOD_THREAD_START,
        json!({"model": "fixture-model", "modelProvider": "fixture-provider"}),
    )
    .await["result"]["thread"]["id"]
        .as_str()
        .expect("thread id")
        .to_string()
}

async fn request_raw(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    server
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
        .expect("handle JSON-RPC request")
        .into_iter()
        .map(|line| serde_json::from_str::<Value>(&line).expect("valid JSON-RPC response"))
        .find(|message| message.get("id") == Some(&json!(id)))
        .expect("matching JSON-RPC response")
}
