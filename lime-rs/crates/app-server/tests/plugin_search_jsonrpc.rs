use std::fs;
use std::sync::{Arc, Mutex};

use app_server::{AppServer, LocalAppDataSource, MockBackend, RuntimeCore};
use app_server_protocol::protocol::v2::{METHOD_PLUGIN_INSTALL, METHOD_PLUGIN_SEARCH};
use app_server_protocol::{METHOD_INITIALIZE, METHOD_INITIALIZED};
use lime_core::database::schema::create_tables;
use rusqlite::Connection;
use serde_json::{json, Value};
use tempfile::TempDir;

#[tokio::test]
async fn plugin_search_uses_current_catalog_and_codex_wire_contract() {
    let temp = TempDir::new().expect("plugin search temp dir");
    let server = server(&temp).await;
    initialize(&server).await;

    for (id, name, display_name) in [
        (2, "alpha-plugin", "Alpha Plugin"),
        (3, "beta-plugin", "Beta Plugin"),
    ] {
        let source = temp.path().join(name);
        write_plugin(&source, name, display_name);
        request(
            &server,
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

    let first = request(
        &server,
        4,
        METHOD_PLUGIN_SEARCH,
        json!({
            "searchTerm": "plugin",
            "scope": "workspace",
            "limit": 1
        }),
    )
    .await;
    assert_eq!(first["result"]["data"].as_array().unwrap().len(), 1);
    assert_eq!(first["result"]["nextCursor"], "1");
    assert_eq!(first["result"]["data"][0]["plugin"]["id"], "alpha-plugin");
    assert_eq!(
        first["result"]["data"][0]["marketplaceName"],
        "workspace-market"
    );
    assert_eq!(
        first["result"]["data"][0]["plugin"]["source"]["type"],
        "local"
    );
    assert_eq!(
        first["result"]["data"][0]["plugin"]["installPolicy"],
        "AVAILABLE"
    );
    assert_eq!(first["result"]["data"][0]["plugin"]["authPolicy"], "ON_USE");

    let second = request(
        &server,
        5,
        METHOD_PLUGIN_SEARCH,
        json!({
            "searchTerm": "plugin",
            "scope": "workspace",
            "cursor": "1",
            "limit": 1
        }),
    )
    .await;
    assert_eq!(second["result"]["data"][0]["plugin"]["id"], "beta-plugin");
    assert_eq!(second["result"]["nextCursor"], Value::Null);

    let personal = request(
        &server,
        6,
        METHOD_PLUGIN_SEARCH,
        json!({"searchTerm": "plugin", "scope": "personal"}),
    )
    .await;
    assert_eq!(personal["result"], json!({"data": [], "nextCursor": null}));
}

async fn server(temp: &TempDir) -> AppServer {
    let connection = Connection::open_in_memory().expect("plugin search product db");
    create_tables(&connection).expect("plugin search product schema");
    let app_data_source = LocalAppDataSource::initialize_with_roots(
        Arc::new(Mutex::new(connection)),
        temp.path(),
        temp.path().join("agent-root"),
    )
    .await
    .expect("plugin search app data source");
    let runtime = RuntimeCore::with_backend(Arc::new(MockBackend))
        .with_app_data_source(Arc::new(app_data_source));
    AppServer::with_runtime(runtime)
}

fn write_plugin(root: &std::path::Path, name: &str, display_name: &str) {
    fs::create_dir_all(root.join(".codex-plugin")).expect("create plugin manifest directory");
    fs::write(
        root.join("plugin.json"),
        serde_json::to_vec_pretty(&json!({
            "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
            "name": name,
            "version": "1.0.0",
            "description": format!("{display_name} description")
        }))
        .expect("serialize standard plugin manifest"),
    )
    .expect("write standard plugin manifest");
    fs::write(
        root.join(".codex-plugin/plugin.json"),
        serde_json::to_vec_pretty(&json!({
            "name": name,
            "version": "1.0.0",
            "description": format!("{display_name} description"),
            "interface": {"displayName": display_name}
        }))
        .expect("serialize plugin manifest"),
    )
    .expect("write plugin manifest");
}

async fn initialize(server: &AppServer) {
    request(
        server,
        1,
        METHOD_INITIALIZE,
        json!({"clientInfo": {"name": "plugin-search-jsonrpc-test", "version": "1"}}),
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
        .filter_map(|message| serde_json::from_str::<Value>(message).ok())
        .find(|message| message.get("id") == Some(&json!(id)))
        .expect("JSON-RPC response");
    assert!(
        response.get("error").is_none(),
        "plugin search request failed: {response:#}"
    );
    response
}
