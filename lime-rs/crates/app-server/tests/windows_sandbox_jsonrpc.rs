use std::sync::Arc;

use app_server::{AppServer, MockBackend, RuntimeCore};
use app_server_protocol::{
    METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_WINDOWS_SANDBOX_READINESS, PROTOCOL_VERSION,
};
use serde_json::{json, Value};

#[tokio::test]
async fn windows_sandbox_readiness_uses_current_config_and_tool_runtime_state() {
    let server = AppServer::with_runtime(RuntimeCore::with_backend(Arc::new(MockBackend)));
    initialize(&server).await;

    let response = request(&server, 2, METHOD_WINDOWS_SANDBOX_READINESS, None).await;
    assert!(
        matches!(
            response.pointer("/result/status").and_then(Value::as_str),
            Some("ready" | "notConfigured" | "updateRequired")
        ),
        "readiness must return the current typed status: {response:#}"
    );

    let invalid = request(
        &server,
        3,
        METHOD_WINDOWS_SANDBOX_READINESS,
        Some(json!({"mode": "elevated"})),
    )
    .await;
    assert_eq!(invalid.pointer("/error/code"), Some(&json!(-32602)));
}

async fn initialize(server: &AppServer) {
    let response = request(
        server,
        1,
        METHOD_INITIALIZE,
        Some(json!({
            "clientInfo": {
                "name": "windows-sandbox-jsonrpc-test",
                "version": "1.0.0"
            }
        })),
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

async fn request(server: &AppServer, id: u64, method: &str, params: Option<Value>) -> Value {
    let mut payload = json!({"jsonrpc": "2.0", "id": id, "method": method});
    if let Some(params) = params {
        payload["params"] = params;
    }
    let lines = server
        .handle_json_line(&payload.to_string())
        .await
        .expect("handle JSON-RPC request");
    lines
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).expect("decode response"))
        .find(|message| message.get("id") == Some(&json!(id)))
        .unwrap_or_else(|| panic!("{method} should return matching response: {lines:#?}"))
}
