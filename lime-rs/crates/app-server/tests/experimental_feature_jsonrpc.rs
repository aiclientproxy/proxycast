use app_server::{AppServer, MockBackend, RuntimeCore};
use app_server_protocol::protocol::v2::{
    METHOD_EXPERIMENTAL_FEATURE_ENABLEMENT_SET, METHOD_EXPERIMENTAL_FEATURE_LIST,
};
use app_server_protocol::{METHOD_INITIALIZE, METHOD_INITIALIZED, PROTOCOL_VERSION};
use lime_core::config::ConfigManager;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex, OnceLock};
use tempfile::TempDir;

#[tokio::test]
async fn experimental_feature_catalog_and_enablement_use_current_config_owner() {
    let _config_lock = config_env_lock()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let temp = TempDir::new().expect("experimental feature config temp");
    let config_path = temp.path().join("config.yaml");
    std::fs::write(
        &config_path,
        "experimental:\n  webmcp:\n    enabled: true\n",
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

    let listed = request(&server, 2, METHOD_EXPERIMENTAL_FEATURE_LIST, json!({})).await;
    assert_eq!(listed["result"]["data"][0]["name"], "webmcp");
    assert_eq!(listed["result"]["data"][0]["enabled"], true);
    assert_eq!(listed["result"]["data"][0]["defaultEnabled"], false);

    let updated = request(
        &server,
        3,
        METHOD_EXPERIMENTAL_FEATURE_ENABLEMENT_SET,
        json!({"enablement": {"webmcp": false, "unknown": true}}),
    )
    .await;
    assert_eq!(updated["result"]["enablement"], json!({"webmcp": false}));

    let listed = request(
        &server,
        4,
        METHOD_EXPERIMENTAL_FEATURE_LIST,
        json!({"cursor": "0", "limit": 1}),
    )
    .await;
    assert_eq!(listed["result"]["data"][0]["enabled"], false);
    assert_eq!(listed["result"]["nextCursor"], Value::Null);
    assert!(
        !ConfigManager::load(&config_path)
            .expect("load persisted config")
            .config()
            .experimental
            .webmcp
            .enabled
    );

    let unknown_thread = request_error(
        &server,
        5,
        METHOD_EXPERIMENTAL_FEATURE_LIST,
        json!({"threadId": "missing-thread"}),
    )
    .await;
    assert_eq!(unknown_thread["error"]["code"], -32600);
    assert_eq!(
        unknown_thread["error"]["message"],
        "thread not found: missing-thread"
    );
}

async fn initialize(server: &AppServer) {
    let response = request(
        server,
        1,
        METHOD_INITIALIZE,
        json!({
            "clientInfo": {"name": "experimental-feature-jsonrpc-test", "version": "1.0.0"}
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
    let lines = server
        .handle_json_line(
            &json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string(),
        )
        .await
        .expect("handle JSON-RPC request");
    let response = lines
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).expect("decode response"))
        .find(|message| message.get("id") == Some(&json!(id)))
        .unwrap_or_else(|| panic!("{method} should return matching response: {lines:#?}"));
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

fn config_env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}
