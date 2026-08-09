use std::fs;
use std::sync::{Arc, Mutex};

use app_server::{AppServer, LocalAppDataSource, MockBackend, RuntimeCore};
use app_server_protocol::error_codes;
use app_server_protocol::protocol::v2::{
    METHOD_APP_INSTALLED, METHOD_APP_LIST, METHOD_APP_LIST_UPDATED, METHOD_APP_READ,
    METHOD_PLUGIN_ENABLED_SET, METHOD_PLUGIN_INSTALL,
};
use app_server_protocol::{JsonRpcMessage, METHOD_INITIALIZE, METHOD_INITIALIZED};
use lime_core::database::schema::create_tables;
use rusqlite::Connection;
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::sync::broadcast;
use tokio::time::{timeout, Duration};

#[tokio::test]
async fn apps_exact_contract_uses_plugin_catalog_and_publishes_updates() {
    let temp = TempDir::new().expect("apps temp dir");
    let server = server(&temp).await;
    let mut outbound = server.subscribe_outbound_messages();
    initialize(&server).await;

    install_plugin(&server, &temp, 2, "beta-plugin", "beta-app").await;
    let beta_update = next_app_update(&mut outbound).await;
    assert_eq!(beta_update["params"]["data"][0]["id"], "beta-app");

    install_plugin(&server, &temp, 3, "alpha-plugin", "alpha-app").await;
    let install_update = next_app_update(&mut outbound).await;
    assert_eq!(
        install_update["params"]["data"]
            .as_array()
            .expect("install app update")
            .len(),
        2
    );

    let first = request_ok(
        &server,
        4,
        METHOD_APP_LIST,
        json!({"limit": 1, "forceRefetch": true}),
    )
    .await;
    assert_eq!(first["result"]["data"][0]["id"], "alpha-app");
    assert_eq!(first["result"]["nextCursor"], "1");
    let list_update = next_app_update(&mut outbound).await;
    assert_eq!(
        list_update["params"]["data"]
            .as_array()
            .expect("list app update")
            .len(),
        2
    );

    let second = request_ok(
        &server,
        5,
        METHOD_APP_LIST,
        json!({"cursor": "1", "limit": 1}),
    )
    .await;
    assert_eq!(second["result"]["data"][0]["id"], "beta-app");
    assert_eq!(second["result"]["nextCursor"], Value::Null);

    let read = request_ok(
        &server,
        6,
        METHOD_APP_READ,
        json!({
            "appIds": ["beta-app", "missing", "alpha-app", "beta-app"],
            "includeTools": true
        }),
    )
    .await;
    assert_eq!(read["result"]["apps"][0]["id"], "beta-app");
    assert_eq!(read["result"]["apps"][1]["id"], "alpha-app");
    assert_eq!(read["result"]["apps"][0]["toolSummaries"], json!([]));
    assert_eq!(read["result"]["missingAppIds"], json!(["missing"]));

    let installed = request_ok(&server, 7, METHOD_APP_INSTALLED, json!({})).await;
    let installed_apps = installed["result"]["apps"]
        .as_array()
        .expect("installed apps");
    assert_eq!(installed_apps.len(), 2);
    assert!(installed_apps
        .iter()
        .all(|app| app["enabled"] == true && app["callable"] == false));

    let too_many_ids = (0..101)
        .map(|index| format!("app-{index}"))
        .collect::<Vec<_>>();
    let too_many = request_raw(&server, 8, METHOD_APP_READ, json!({"appIds": too_many_ids})).await;
    assert_eq!(
        too_many["error"]["code"],
        json!(error_codes::INVALID_PARAMS)
    );

    let missing_thread = request_raw(
        &server,
        9,
        METHOD_APP_LIST,
        json!({"threadId": "missing-thread"}),
    )
    .await;
    assert_eq!(
        missing_thread["error"]["code"],
        json!(error_codes::SESSION_NOT_FOUND)
    );

    request_ok(
        &server,
        10,
        METHOD_PLUGIN_ENABLED_SET,
        json!({"pluginId": "alpha-plugin", "enabled": false}),
    )
    .await;
    let disabled_update = next_app_update(&mut outbound).await;
    let disabled_alpha = disabled_update["params"]["data"]
        .as_array()
        .expect("disabled app update")
        .iter()
        .find(|app| app["id"] == "alpha-app")
        .expect("alpha app");
    assert_eq!(disabled_alpha["isEnabled"], false);
    assert_eq!(disabled_alpha["isAccessible"], false);
}

async fn server(temp: &TempDir) -> AppServer {
    let connection = Connection::open_in_memory().expect("apps product db");
    create_tables(&connection).expect("apps product schema");
    let app_data_source = LocalAppDataSource::initialize_with_roots(
        Arc::new(Mutex::new(connection)),
        temp.path(),
        temp.path().join("agent-root"),
    )
    .await
    .expect("apps app data source");
    let runtime = RuntimeCore::with_backend(Arc::new(MockBackend))
        .with_app_data_source(Arc::new(app_data_source));
    AppServer::with_runtime(runtime)
}

async fn install_plugin(
    server: &AppServer,
    temp: &TempDir,
    id: u64,
    plugin_id: &str,
    app_id: &str,
) {
    let source = temp.path().join(plugin_id);
    fs::create_dir_all(source.join(".codex-plugin")).expect("plugin manifest directory");
    fs::write(
        source.join("plugin.json"),
        serde_json::to_vec_pretty(&json!({
            "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
            "name": plugin_id,
            "version": "1.0.0",
            "description": format!("{plugin_id} description")
        }))
        .expect("serialize standard plugin manifest"),
    )
    .expect("write standard plugin manifest");
    fs::write(
        source.join(".codex-plugin/plugin.json"),
        serde_json::to_vec_pretty(&json!({
            "interface": {"displayName": plugin_id},
            "apps": "./apps.json"
        }))
        .expect("serialize Codex plugin extension"),
    )
    .expect("write Codex plugin extension");
    fs::write(
        source.join("apps.json"),
        serde_json::to_vec_pretty(&json!({
            "apps": {
                (app_id): {"id": app_id}
            }
        }))
        .expect("serialize plugin apps"),
    )
    .expect("write plugin apps");
    request_ok(
        server,
        id,
        METHOD_PLUGIN_INSTALL,
        json!({
            "sourcePath": source,
            "marketplaceId": "workspace-market",
            "source": "repo"
        }),
    )
    .await;
}

async fn initialize(server: &AppServer) {
    request_ok(
        server,
        1,
        METHOD_INITIALIZE,
        json!({"clientInfo": {"name": "apps-jsonrpc-test", "version": "1"}}),
    )
    .await;
    server
        .handle_json_line(
            &json!({"jsonrpc": "2.0", "method": METHOD_INITIALIZED, "params": {}}).to_string(),
        )
        .await
        .expect("initialized notification");
}

async fn next_app_update(receiver: &mut broadcast::Receiver<JsonRpcMessage>) -> Value {
    timeout(Duration::from_secs(2), async {
        loop {
            let message = receiver.recv().await.expect("outbound notification");
            let value = serde_json::to_value(message).expect("serialize outbound notification");
            if value["method"] == METHOD_APP_LIST_UPDATED {
                return value;
            }
        }
    })
    .await
    .expect("app/list/updated timeout")
}

async fn request_ok(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    let response = request_raw(server, id, method, params).await;
    assert!(
        response.get("error").is_none(),
        "{method} request failed: {response:#}"
    );
    response
}

async fn request_raw(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    server
        .handle_json_line(
            &json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string(),
        )
        .await
        .expect("JSON-RPC request")
        .iter()
        .filter_map(|message| serde_json::from_str::<Value>(message).ok())
        .find(|message| message.get("id") == Some(&json!(id)))
        .unwrap_or_else(|| panic!("{method} should return response id {id}"))
}
