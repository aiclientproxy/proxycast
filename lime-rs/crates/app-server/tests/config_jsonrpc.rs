use app_server::{AppServer, MockBackend, RuntimeCore};
use app_server_protocol::protocol::v2::{
    METHOD_CONFIG_BATCH_WRITE, METHOD_CONFIG_READ, METHOD_CONFIG_VALUE_WRITE,
};
use app_server_protocol::{METHOD_INITIALIZE, METHOD_INITIALIZED, PROTOCOL_VERSION};
use lime_core::config::ConfigManager;
use serde_json::{json, Value};
use std::sync::Arc;
use tempfile::TempDir;

#[tokio::test]
async fn config_control_plane_uses_the_single_desktop_yaml_layer() {
    let temp = TempDir::new().expect("config temp");
    let config_path = temp.path().join("config.yaml");
    std::fs::write(
        &config_path,
        "server:\n  api_key: config-test-key\nlanguage: zh-CN\n",
    )
    .expect("write config");
    let previous = std::env::var_os("LIME_CONFIG_PATH");
    let _restore = scopeguard::guard(previous, |value| {
        if let Some(value) = value {
            std::env::set_var("LIME_CONFIG_PATH", value);
        } else {
            std::env::remove_var("LIME_CONFIG_PATH");
        }
    });
    std::env::set_var("LIME_CONFIG_PATH", &config_path);

    let server = AppServer::with_runtime(RuntimeCore::with_backend(Arc::new(MockBackend)));
    initialize(&server).await;

    let read = request(
        &server,
        2,
        METHOD_CONFIG_READ,
        json!({"includeLayers": true}),
    )
    .await;
    assert_eq!(read["result"]["config"]["language"], "zh-CN");
    assert_eq!(
        read["result"]["layers"][0]["name"],
        json!({
            "type": "user",
            "file": config_path.to_string_lossy(),
            "profile": null
        })
    );
    let version = read["result"]["layers"][0]["version"]
        .as_str()
        .expect("config version")
        .to_string();
    assert_eq!(read["result"]["origins"]["language"]["version"], version);

    let batch = request(
        &server,
        3,
        METHOD_CONFIG_BATCH_WRITE,
        json!({
            "edits": [{
                "keyPath": "language",
                "value": "en-US",
                "mergeStrategy": "replace"
            }],
            "expectedVersion": version,
            "reloadUserConfig": true
        }),
    )
    .await;
    assert_eq!(batch["result"]["status"], "ok");
    assert_eq!(
        batch["result"]["filePath"],
        json!(config_path.to_string_lossy())
    );
    let next_version = batch["result"]["version"]
        .as_str()
        .expect("next version")
        .to_string();

    let value_write = request(
        &server,
        4,
        METHOD_CONFIG_VALUE_WRITE,
        json!({
            "keyPath": "minimize_to_tray",
            "value": true,
            "mergeStrategy": "replace",
            "expectedVersion": next_version
        }),
    )
    .await;
    assert_eq!(value_write["result"]["status"], "ok");
    let persisted = ConfigManager::load(&config_path).expect("load persisted config");
    assert_eq!(persisted.config().language, "en-US");
    assert!(persisted.config().minimize_to_tray);

    let stale = request_error(
        &server,
        5,
        METHOD_CONFIG_VALUE_WRITE,
        json!({
            "keyPath": "language",
            "value": "ja-JP",
            "mergeStrategy": "replace",
            "expectedVersion": "stale-version"
        }),
    )
    .await;
    assert_eq!(stale["error"]["code"], -32600);
    assert_eq!(
        stale["error"]["data"]["config_write_error_code"],
        "configVersionConflict"
    );

    let foreign_path = request_error(
        &server,
        6,
        METHOD_CONFIG_VALUE_WRITE,
        json!({
            "keyPath": "language",
            "value": "ko-KR",
            "mergeStrategy": "replace",
            "filePath": temp.path().join("other.yaml").to_string_lossy()
        }),
    )
    .await;
    assert_eq!(
        foreign_path["error"]["data"]["config_write_error_code"],
        "configLayerReadonly"
    );

    let unknown_key = request_error(
        &server,
        7,
        METHOD_CONFIG_VALUE_WRITE,
        json!({
            "keyPath": "unknown_product_config",
            "value": true,
            "mergeStrategy": "replace"
        }),
    )
    .await;
    assert_eq!(
        unknown_key["error"]["data"]["config_write_error_code"],
        "configSchemaUnknownKey"
    );

    let project_layer = request_error(
        &server,
        8,
        METHOD_CONFIG_READ,
        json!({"cwd": temp.path().to_string_lossy()}),
    )
    .await;
    assert_eq!(project_layer["error"]["code"], -32602);
}

async fn initialize(server: &AppServer) {
    let response = request(
        server,
        1,
        METHOD_INITIALIZE,
        json!({
            "clientInfo": {"name": "config-jsonrpc-test", "version": "1.0.0"}
        }),
    )
    .await;
    assert_eq!(
        response["result"]["serverInfo"]["protocolVersion"],
        PROTOCOL_VERSION
    );
    let lines = server
        .handle_json_line(
            &json!({"jsonrpc": "2.0", "method": METHOD_INITIALIZED, "params": {}}).to_string(),
        )
        .await
        .expect("initialized notification");
    assert!(lines.is_empty());
}

async fn request(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    let response = request_error(server, id, method, params).await;
    if let Some(error) = response.get("error") {
        panic!("{method} failed: {error}");
    }
    response
}

async fn request_error(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    let lines = server
        .handle_json_line(
            &json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string(),
        )
        .await
        .expect("handle JSON-RPC request");
    lines
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).expect("decode response"))
        .find(|message| message.get("id") == Some(&json!(id)))
        .unwrap_or_else(|| panic!("{method} should return matching response: {lines:#?}"))
}
