use std::sync::Arc;

use app_server::{AppServer, MockBackend, ProjectionStore, RuntimeCore};
use app_server_protocol::protocol::v2::{
    METHOD_THREAD_SECTION_CREATE, METHOD_THREAD_SECTION_DELETE, METHOD_THREAD_SECTION_LIST,
    METHOD_THREAD_SECTION_MOVE, METHOD_THREAD_SECTION_UPDATE,
};
use app_server_protocol::{
    error_codes, METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_LIST, METHOD_THREAD_START,
};
use serde_json::{json, Value};
use tempfile::TempDir;
use thread_store::PINNED_THREAD_SECTION_ID;

#[tokio::test]
async fn thread_sections_persist_order_and_replace_legacy_pinning() {
    let temp = TempDir::new().expect("thread sections temp dir");
    let projection_path = temp.path().join("projection.sqlite");
    let initial = server(&projection_path);
    initialize(&initial).await;

    let sections = request(&initial, 2, METHOD_THREAD_SECTION_LIST, json!({})).await;
    assert_eq!(
        sections["result"]["data"],
        json!([{"id": PINNED_THREAD_SECTION_ID, "name": "Pinned"}])
    );

    let created = request(
        &initial,
        3,
        METHOD_THREAD_SECTION_CREATE,
        json!({"name": "  Active  "}),
    )
    .await;
    let section_id = created["result"]["section"]["id"]
        .as_str()
        .expect("created section id")
        .to_string();
    assert_eq!(created["result"]["section"]["name"], "Active");

    let renamed = request(
        &initial,
        4,
        METHOD_THREAD_SECTION_UPDATE,
        json!({"sectionId": section_id, "name": "Current"}),
    )
    .await;
    assert_eq!(renamed["result"]["section"]["name"], "Current");

    let first_thread = start_thread(&initial, 5).await;
    let second_thread = start_thread(&initial, 6).await;
    request(
        &initial,
        7,
        METHOD_THREAD_SECTION_MOVE,
        json!({"threadId": first_thread, "sectionId": section_id}),
    )
    .await;
    request(
        &initial,
        8,
        METHOD_THREAD_SECTION_MOVE,
        json!({
            "threadId": second_thread,
            "sectionId": section_id,
            "beforeThreadId": first_thread
        }),
    )
    .await;

    let ordered = list_section(&initial, 9, &section_id).await;
    assert_eq!(
        thread_ids(&ordered),
        vec![second_thread.as_str(), first_thread.as_str()]
    );
    for thread in ordered["result"]["data"]
        .as_array()
        .expect("section thread page")
    {
        assert_eq!(thread["section"]["id"], section_id);
        assert_eq!(thread["section"]["name"], "Current");
        assert!(thread["sectionEnteredAt"].is_number());
        assert!(thread.get("isPinned").is_none());
    }

    drop(initial);
    let restarted = server(&projection_path);
    initialize(&restarted).await;
    let restored = list_section(&restarted, 10, &section_id).await;
    assert_eq!(
        thread_ids(&restored),
        vec![second_thread.as_str(), first_thread.as_str()]
    );

    request(
        &restarted,
        11,
        METHOD_THREAD_SECTION_MOVE,
        json!({"threadId": second_thread, "sectionId": null}),
    )
    .await;
    let unsectioned = request(
        &restarted,
        12,
        METHOD_THREAD_LIST,
        json!({"sectionId": null}),
    )
    .await;
    assert!(thread_ids(&unsectioned).contains(&second_thread.as_str()));

    request(
        &restarted,
        13,
        METHOD_THREAD_SECTION_DELETE,
        json!({"sectionId": section_id}),
    )
    .await;
    let unsectioned = request(
        &restarted,
        14,
        METHOD_THREAD_LIST,
        json!({"sectionId": null}),
    )
    .await;
    let ids = thread_ids(&unsectioned);
    assert!(ids.contains(&first_thread.as_str()));
    assert!(ids.contains(&second_thread.as_str()));

    for (id, method, params) in [
        (
            15,
            METHOD_THREAD_SECTION_UPDATE,
            json!({"sectionId": PINNED_THREAD_SECTION_ID, "name": "Other"}),
        ),
        (
            16,
            METHOD_THREAD_SECTION_DELETE,
            json!({"sectionId": PINNED_THREAD_SECTION_ID}),
        ),
    ] {
        let response = request_raw(&restarted, id, method, params).await;
        assert_eq!(
            response.pointer("/error/code"),
            Some(&json!(error_codes::INVALID_REQUEST))
        );
    }
}

fn server(path: &std::path::Path) -> AppServer {
    let projection = ProjectionStore::initialize(path).expect("thread sections projection store");
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
        json!({"clientInfo": {"name": "thread-sections-jsonrpc-test", "version": "1"}}),
    )
    .await;
    server
        .handle_json_line(
            &json!({"jsonrpc": "2.0", "method": METHOD_INITIALIZED, "params": {}}).to_string(),
        )
        .await
        .expect("initialized notification");
}

async fn start_thread(server: &AppServer, id: u64) -> String {
    request(
        server,
        id,
        METHOD_THREAD_START,
        json!({"model": "fixture-model", "modelProvider": "fixture-provider"}),
    )
    .await["result"]["thread"]["id"]
        .as_str()
        .expect("started thread id")
        .to_string()
}

async fn list_section(server: &AppServer, id: u64, section_id: &str) -> Value {
    request(
        server,
        id,
        METHOD_THREAD_LIST,
        json!({"sectionId": section_id, "sortKey": "section_position"}),
    )
    .await
}

fn thread_ids(response: &Value) -> Vec<&str> {
    response["result"]["data"]
        .as_array()
        .expect("thread list data")
        .iter()
        .map(|thread| thread["id"].as_str().expect("thread id"))
        .collect()
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
