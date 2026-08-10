use std::sync::Arc;

use app_server::{AppServer, MockBackend, RuntimeCore};
use app_server_protocol::{
    METHOD_COLLABORATION_MODE_LIST, METHOD_INITIALIZE, METHOD_INITIALIZED, PROTOCOL_VERSION,
};
use serde_json::{json, Value};

#[tokio::test]
async fn collaboration_mode_list_returns_desktop_presets_over_public_jsonrpc() {
    let server = AppServer::with_runtime(RuntimeCore::with_backend(Arc::new(MockBackend)));
    initialize(&server).await;

    let response = request(&server, 2, METHOD_COLLABORATION_MODE_LIST, json!({})).await;
    assert_eq!(
        response.pointer("/result/data"),
        Some(&json!([
            {
                "name": "Plan",
                "mode": "plan",
                "model": null,
                "reasoning_effort": "medium"
            },
            {
                "name": "Default",
                "mode": "default",
                "model": null,
                "reasoning_effort": null
            }
        ]))
    );
}

async fn initialize(server: &AppServer) {
    let response = request(
        server,
        1,
        METHOD_INITIALIZE,
        json!({
            "clientInfo": {
                "name": "collaboration-mode-jsonrpc-test",
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
    let response = lines
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).expect("decode JSON-RPC message"))
        .find(|message| message.get("id") == Some(&json!(id)))
        .unwrap_or_else(|| panic!("{method} should return the matching response: {lines:#?}"));
    if let Some(error) = response.get("error") {
        panic!("{method} failed: {error}");
    }
    response
}
