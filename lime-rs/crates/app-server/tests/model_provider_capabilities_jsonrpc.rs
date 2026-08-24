use std::sync::Arc;

use app_server::{AppServer, MockBackend, RuntimeCore};
use app_server_protocol::{
    METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_MODEL_PROVIDER_CAPABILITIES_READ,
};
use lime_core::config::{Config, ConfigManager};
use serde_json::{json, Value};
use tempfile::TempDir;

#[tokio::test]
async fn model_provider_capabilities_read_uses_runtime_config_route() {
    let temp = TempDir::new().expect("model provider capabilities temp");
    let config_path = temp.path().join("config.yaml");
    let mut config = Config::default();
    config.default_provider = "openai-response".to_string();
    config.providers.openai.base_url = Some("https://api.openai.com/v1".to_string());
    ConfigManager::with_config(config, config_path.clone())
        .save()
        .expect("write official provider config");

    let runtime =
        RuntimeCore::with_backend(Arc::new(MockBackend)).with_app_config_path(config_path.clone());
    let server = AppServer::with_runtime(runtime);
    initialize(&server).await;

    let official = request(
        &server,
        2,
        METHOD_MODEL_PROVIDER_CAPABILITIES_READ,
        json!({}),
    )
    .await;
    assert_eq!(
        official["result"],
        json!({"namespaceTools": false, "imageGeneration": true, "webSearch": true})
    );

    let mut compatible = Config::default();
    compatible.default_provider = "openai-response".to_string();
    compatible.providers.openai.base_url = Some("https://gateway.example/v1".to_string());
    ConfigManager::with_config(compatible, config_path)
        .save()
        .expect("write compatible provider config");
    let compatible_response = request(
        &server,
        3,
        METHOD_MODEL_PROVIDER_CAPABILITIES_READ,
        json!({}),
    )
    .await;
    assert_eq!(
        compatible_response["result"],
        json!({"namespaceTools": false, "imageGeneration": false, "webSearch": false})
    );
}

async fn initialize(server: &AppServer) {
    request(
        server,
        1,
        METHOD_INITIALIZE,
        json!({"clientInfo": {"name": "model-provider-capabilities-jsonrpc-test", "version": "1"}}),
    )
    .await;
    server
        .handle_json_line(
            &json!({"jsonrpc": "2.0", "method": METHOD_INITIALIZED, "params": {}}).to_string(),
        )
        .await
        .expect("initialized notification");
}

async fn request(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    let response = server
        .handle_json_line(
            &json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string(),
        )
        .await
        .expect("JSON-RPC request")
        .iter()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|message| message.get("id") == Some(&json!(id)))
        .expect("JSON-RPC response");
    assert!(
        response.get("error").is_none(),
        "request failed: {response:#}"
    );
    response
}
