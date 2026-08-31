use std::sync::Arc;

use app_server::{AppServer, MockBackend, ProjectionStore, RuntimeCore};
use app_server_protocol::protocol::v2::METHOD_THREAD_SETTINGS_UPDATE;
use app_server_protocol::{
    error_codes, METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_PERMISSION_PROFILE_LIST,
    METHOD_THREAD_READ, METHOD_THREAD_START, PROTOCOL_VERSION,
};
use serde_json::{json, Value};
use tempfile::TempDir;

#[tokio::test]
async fn permission_profile_list_returns_desktop_catalog_over_public_jsonrpc() {
    let temp = TempDir::new().expect("permission profile config temp dir");
    let server = server_with_config(&temp);
    initialize(&server).await;

    let first = request(
        &server,
        2,
        METHOD_PERMISSION_PROFILE_LIST,
        json!({"limit": 2, "cwd": "/workspace"}),
    )
    .await;
    assert_eq!(
        first.pointer("/result"),
        Some(&json!({
            "data": [
                {"id": ":read-only", "allowed": true},
                {"id": ":workspace", "allowed": true}
            ],
            "nextCursor": "2"
        }))
    );

    let second = request(
        &server,
        3,
        METHOD_PERMISSION_PROFILE_LIST,
        json!({"cursor": "2", "limit": 2}),
    )
    .await;
    assert_eq!(
        second.pointer("/result"),
        Some(&json!({
            "data": [{"id": ":danger-full-access", "allowed": true}]
        }))
    );

    let rejected = request_messages(
        &server,
        4,
        METHOD_PERMISSION_PROFILE_LIST,
        json!({"cwd": "  "}),
    )
    .await;
    let response = rejected
        .iter()
        .find(|message| message.get("id") == Some(&json!(4)))
        .expect("permissionProfile/list cwd error response");
    assert_eq!(
        response.pointer("/error/code"),
        Some(&json!(error_codes::INVALID_REQUEST))
    );
}

#[tokio::test]
async fn thread_start_returns_permission_profile_provenance_over_public_jsonrpc() {
    let temp = TempDir::new().expect("permission profile projection temp dir");
    let server = server_with_projection_store(&temp);
    initialize(&server).await;

    let response = request(
        &server,
        2,
        METHOD_THREAD_START,
        json!({
            "model": "fixture-model",
            "modelProvider": "fixture-provider",
            "permissions": ":workspace"
        }),
    )
    .await;

    assert_eq!(
        response.pointer("/result/activePermissionProfile"),
        Some(&json!({"id": ":workspace"}))
    );
    assert_eq!(
        response.pointer("/result/thread/extra/activePermissionProfile"),
        Some(&json!({"id": ":workspace"}))
    );
    assert_eq!(
        response.pointer("/result/thread/extra/sandboxPolicy"),
        Some(&json!("workspace-write"))
    );
}

#[tokio::test]
async fn thread_start_rejects_unknown_permission_profile_over_public_jsonrpc() {
    let temp = TempDir::new().expect("permission profile config temp dir");
    let server = server_with_config(&temp);
    initialize(&server).await;

    let lines = server
        .handle_json_line(
            &json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": METHOD_THREAD_START,
                "params": {
                    "model": "fixture-model",
                    "modelProvider": "fixture-provider",
                    "permissions": ":custom"
                }
            })
            .to_string(),
        )
        .await
        .expect("handle thread/start JSON-RPC request");
    let response = lines
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).expect("decode JSON-RPC response"))
        .find(|message| message.get("id") == Some(&json!(2)))
        .expect("thread/start must return matching error response");
    assert_eq!(response.pointer("/error/code"), Some(&json!(-32602)));
    assert!(response
        .pointer("/error/message")
        .and_then(Value::as_str)
        .is_some_and(|message| message.contains("unknown permission profile")));
}

#[tokio::test]
async fn thread_settings_update_persists_permission_profile_provenance_over_public_jsonrpc() {
    let temp = TempDir::new().expect("permission profile projection temp dir");
    let server = server_with_projection_store(&temp);
    initialize(&server).await;

    let started = request(
        &server,
        2,
        METHOD_THREAD_START,
        json!({
            "model": "fixture-model",
            "modelProvider": "fixture-provider",
            "permissions": ":workspace"
        }),
    )
    .await;
    let thread_id = started
        .pointer("/result/thread/id")
        .and_then(Value::as_str)
        .expect("thread id");

    let updated = request_messages(
        &server,
        3,
        METHOD_THREAD_SETTINGS_UPDATE,
        json!({
            "threadId": thread_id,
            "approvalPolicy": "on-request",
            "permissions": ":read-only"
        }),
    )
    .await;
    let notification = updated
        .iter()
        .find(|message| message.get("method") == Some(&json!("thread/settings/updated")))
        .unwrap_or_else(|| panic!("missing settings notification: {updated:#?}"));
    assert_eq!(
        notification.pointer("/params/threadSettings/activePermissionProfile"),
        Some(&json!({"id": ":read-only"}))
    );
    assert_eq!(
        notification.pointer("/params/threadSettings/sandboxPolicy"),
        Some(&json!("read-only"))
    );

    let read = request(
        &server,
        4,
        METHOD_THREAD_READ,
        json!({"threadId": thread_id}),
    )
    .await;
    assert_eq!(
        read.pointer("/result/thread/extra/permissions"),
        Some(&json!(":read-only"))
    );
    assert_eq!(
        read.pointer("/result/thread/extra/activePermissionProfile"),
        Some(&json!({"id": ":read-only"}))
    );
    assert_eq!(
        read.pointer("/result/thread/extra/sandboxPolicy"),
        Some(&json!("read-only"))
    );

    let rejected = request_messages(
        &server,
        5,
        METHOD_THREAD_SETTINGS_UPDATE,
        json!({"threadId": thread_id, "permissions": ":unknown"}),
    )
    .await;
    let response = rejected
        .iter()
        .find(|message| message.get("id") == Some(&json!(5)))
        .expect("settings update error response");
    assert_eq!(
        response.pointer("/error/code"),
        Some(&json!(error_codes::INVALID_REQUEST))
    );

    let unchanged = request(
        &server,
        6,
        METHOD_THREAD_READ,
        json!({"threadId": thread_id}),
    )
    .await;
    assert_eq!(
        unchanged.pointer("/result/thread/extra/activePermissionProfile"),
        Some(&json!({"id": ":read-only"}))
    );

    let legacy = request_messages(
        &server,
        7,
        METHOD_THREAD_SETTINGS_UPDATE,
        json!({"threadId": thread_id, "sandboxPolicy": "danger-full-access"}),
    )
    .await;
    let legacy_notification = legacy
        .iter()
        .find(|message| message.get("method") == Some(&json!("thread/settings/updated")))
        .unwrap_or_else(|| panic!("missing legacy settings notification: {legacy:#?}"));
    assert!(legacy_notification
        .pointer("/params/threadSettings/activePermissionProfile")
        .is_none());

    let legacy_read = request(
        &server,
        8,
        METHOD_THREAD_READ,
        json!({"threadId": thread_id}),
    )
    .await;
    assert!(legacy_read
        .pointer("/result/thread/extra/activePermissionProfile")
        .is_none());
    assert!(legacy_read
        .pointer("/result/thread/extra/permissions")
        .is_none());
}

fn server_with_projection_store(temp: &TempDir) -> AppServer {
    let store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("permission profile projection store"),
    );
    AppServer::with_runtime(runtime_with_config(temp).with_projection_store(store))
}

fn server_with_config(temp: &TempDir) -> AppServer {
    AppServer::with_runtime(runtime_with_config(temp))
}

fn runtime_with_config(temp: &TempDir) -> RuntimeCore {
    RuntimeCore::with_backend(Arc::new(MockBackend))
        .with_app_config_path(temp.path().join("config.yaml"))
}

async fn initialize(server: &AppServer) {
    let response = request(
        server,
        1,
        METHOD_INITIALIZE,
        json!({
            "clientInfo": {"name": "permission-profile-jsonrpc-test", "version": "1.0.0"}
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

async fn request(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    let messages = request_messages(server, id, method, params).await;
    let response = messages
        .into_iter()
        .find(|message| message.get("id") == Some(&json!(id)))
        .unwrap_or_else(|| panic!("{method} should return matching response"));
    if let Some(error) = response.get("error") {
        panic!("{method} failed: {error}");
    }
    response
}

async fn request_messages(server: &AppServer, id: u64, method: &str, params: Value) -> Vec<Value> {
    server
        .handle_json_line(
            &json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string(),
        )
        .await
        .expect("handle JSON-RPC request")
        .into_iter()
        .map(|line| serde_json::from_str::<Value>(&line).expect("decode JSON-RPC response"))
        .collect()
}
