use app_server::{AppServer, RuntimeCore};
use app_server_protocol::protocol::v2::METHOD_FUZZY_FILE_SEARCH;
use app_server_protocol::{error_codes, METHOD_INITIALIZE, METHOD_INITIALIZED};
use serde_json::{json, Value};
use tempfile::TempDir;

#[tokio::test]
async fn fuzzy_file_search_round_trips_over_public_jsonrpc() {
    let root = TempDir::new().expect("fuzzy file search root");
    std::fs::create_dir_all(root.path().join("src")).expect("create src");
    std::fs::write(root.path().join("src/app.rs"), "app").expect("write app.rs");
    std::fs::write(root.path().join("src/apple.rs"), "apple").expect("write apple.rs");
    let server = AppServer::with_runtime(RuntimeCore::default());
    initialize(&server).await;

    let response = request(
        &server,
        2,
        json!({
            "query": "app",
            "roots": [root.path()],
            "cancellationToken": "composer"
        }),
    )
    .await;
    assert_eq!(response["result"]["files"][0]["path"], "src/app.rs");
    assert_eq!(response["result"]["files"][0]["match_type"], "file");
    assert_eq!(response["result"]["files"][0]["file_name"], "app.rs");
    assert_eq!(response["result"]["files"][0]["indices"], json!([4, 5, 6]));

    let empty = request(
        &server,
        3,
        json!({"query": "", "roots": [root.path()], "cancellationToken": null}),
    )
    .await;
    assert_eq!(empty["result"], json!({"files": []}));

    let invalid = request_raw(
        &server,
        4,
        json!({"query": "app", "roots": ["relative"], "cancellationToken": null}),
    )
    .await;
    assert_eq!(invalid["error"]["code"], json!(error_codes::INVALID_PARAMS));
}

async fn initialize(server: &AppServer) {
    request_raw(
        server,
        1,
        json!({"clientInfo": {"name": "fuzzy-file-search-test", "version": "1"}}),
    )
    .await;
    server
        .handle_json_line(
            &json!({"jsonrpc": "2.0", "method": METHOD_INITIALIZED, "params": {}}).to_string(),
        )
        .await
        .expect("initialized notification");
}

async fn request(server: &AppServer, id: u64, params: Value) -> Value {
    let response = request_raw(server, id, params).await;
    assert!(
        response.get("error").is_none(),
        "fuzzy file search request failed: {response:#}"
    );
    response
}

async fn request_raw(server: &AppServer, id: u64, params: Value) -> Value {
    let method = if id == 1 {
        METHOD_INITIALIZE
    } else {
        METHOD_FUZZY_FILE_SEARCH
    };
    server
        .handle_json_line(
            &json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string(),
        )
        .await
        .expect("JSON-RPC request")
        .iter()
        .filter_map(|message| serde_json::from_str::<Value>(message).ok())
        .find(|message| message.get("id") == Some(&json!(id)))
        .expect("JSON-RPC response")
}
