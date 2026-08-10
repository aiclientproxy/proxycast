use std::sync::Arc;

use app_server::{AppServer, MockBackend, RuntimeCore};
use app_server_protocol::{
    METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_PERMISSION_PROFILE_LIST, PROTOCOL_VERSION,
};
use serde_json::{json, Value};

#[tokio::test]
async fn permission_profile_list_returns_desktop_catalog_over_public_jsonrpc() {
    let server = AppServer::with_runtime(RuntimeCore::with_backend(Arc::new(MockBackend)));
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
    let lines = server
        .handle_json_line(
            &json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string(),
        )
        .await
        .expect("handle JSON-RPC request");
    let response = lines
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).expect("decode JSON-RPC response"))
        .find(|message| message.get("id") == Some(&json!(id)))
        .unwrap_or_else(|| panic!("{method} should return matching response: {lines:#?}"));
    if let Some(error) = response.get("error") {
        panic!("{method} failed: {error}");
    }
    response
}
