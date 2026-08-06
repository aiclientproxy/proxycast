use std::sync::Arc;

use app_server::{AppServer, MockBackend, ProjectionStore, RuntimeCore};
use app_server_protocol::protocol::v2::{METHOD_THREAD_ARCHIVE, METHOD_THREAD_METADATA_UPDATE};
use app_server_protocol::{
    error_codes, METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_LIST, METHOD_THREAD_READ,
    METHOD_THREAD_START,
};
use serde_json::{json, Value};
use tempfile::TempDir;

#[tokio::test]
async fn thread_metadata_update_persists_git_patch_and_supports_archived_threads() {
    let temp = TempDir::new().expect("thread metadata temp dir");
    let projection_path = temp.path().join("projection.sqlite");
    let initial = server(&projection_path);
    initialize(&initial).await;

    let started = request(
        &initial,
        2,
        METHOD_THREAD_START,
        json!({"model": "fixture-model", "modelProvider": "fixture-provider"}),
    )
    .await;
    let thread_id = started["result"]["thread"]["id"]
        .as_str()
        .expect("thread id")
        .to_string();
    let updated = request(
        &initial,
        3,
        METHOD_THREAD_METADATA_UPDATE,
        json!({
            "threadId": thread_id,
            "gitInfo": {
                "sha": "  abc123  ",
                "branch": " main ",
                "originUrl": " https://example.test/repo.git "
            }
        }),
    )
    .await;
    assert_eq!(
        updated["result"]["thread"]["gitInfo"],
        json!({
            "sha": "abc123",
            "branch": "main",
            "originUrl": "https://example.test/repo.git"
        })
    );

    request(
        &initial,
        5,
        METHOD_THREAD_ARCHIVE,
        json!({"threadId": thread_id}),
    )
    .await;
    let archived_update = request(
        &initial,
        6,
        METHOD_THREAD_METADATA_UPDATE,
        json!({
            "threadId": thread_id,
            "gitInfo": {"branch": null}
        }),
    )
    .await;
    assert_eq!(
        archived_update["result"]["thread"]["gitInfo"],
        json!({
            "sha": "abc123",
            "originUrl": "https://example.test/repo.git"
        })
    );

    drop(initial);
    let restarted = server(&projection_path);
    initialize(&restarted).await;
    let cold = request(
        &restarted,
        7,
        METHOD_THREAD_READ,
        json!({"threadId": thread_id}),
    )
    .await;
    assert_eq!(
        cold["result"]["thread"]["gitInfo"],
        json!({
            "sha": "abc123",
            "originUrl": "https://example.test/repo.git"
        })
    );
    let archived = request(&restarted, 8, METHOD_THREAD_LIST, json!({"archived": true})).await;
    assert_eq!(archived["result"]["data"][0]["id"], json!(thread_id));
}

#[tokio::test]
async fn thread_metadata_update_rejects_empty_patches_and_blank_git_fields() {
    let temp = TempDir::new().expect("thread metadata validation temp dir");
    let server = server(&temp.path().join("projection.sqlite"));
    initialize(&server).await;
    let started = request(
        &server,
        2,
        METHOD_THREAD_START,
        json!({"model": "fixture-model", "modelProvider": "fixture-provider"}),
    )
    .await;
    let thread_id = started["result"]["thread"]["id"]
        .as_str()
        .expect("thread id");

    for (id, params) in [
        (3, json!({"threadId": thread_id})),
        (4, json!({"threadId": thread_id, "gitInfo": {}})),
        (5, json!({"threadId": thread_id, "gitInfo": {"sha": "   "}})),
    ] {
        let response = request_raw(&server, id, METHOD_THREAD_METADATA_UPDATE, params).await;
        assert_eq!(
            response.pointer("/error/code"),
            Some(&json!(error_codes::INVALID_REQUEST))
        );
    }
}

fn server(path: &std::path::Path) -> AppServer {
    let projection = ProjectionStore::initialize(path).expect("thread metadata projection store");
    AppServer::with_runtime(
        RuntimeCore::with_backend(Arc::new(MockBackend))
            .with_projection_store(Arc::new(projection)),
    )
}

async fn initialize(server: &AppServer) {
    request(
        server,
        1,
        METHOD_INITIALIZE,
        json!({"clientInfo": {"name": "thread-metadata-jsonrpc-test", "version": "1"}}),
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
    let response = request_raw(server, id, method, params).await;
    assert!(
        response.get("error").is_none(),
        "request failed: {response:#}"
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
        .expect("JSON-RPC response")
}
