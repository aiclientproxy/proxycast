use std::sync::Arc;

use app_server::{AppServer, MockBackend, ProjectionStore, RuntimeCore};
use app_server_protocol::protocol::v2::{
    METHOD_THREAD_ARCHIVE, METHOD_THREAD_DELETE, METHOD_THREAD_FORK,
};
use app_server_protocol::{
    error_codes, AgentSessionStartParams, BusinessObjectRef, METHOD_INITIALIZE, METHOD_INITIALIZED,
    METHOD_THREAD_LOADED_LIST, METHOD_THREAD_READ, METHOD_THREAD_RESUME, METHOD_THREAD_START,
};
use serde_json::{json, Value};
use tempfile::TempDir;
use uuid::Uuid;

#[tokio::test]
async fn thread_loaded_list_tracks_memory_owner_and_paginates_like_codex() {
    let temp = TempDir::new().expect("loaded thread temp dir");
    let projection = ProjectionStore::initialize(temp.path().join("projection.sqlite"))
        .expect("loaded thread projection store");
    let runtime = RuntimeCore::with_backend(Arc::new(MockBackend))
        .with_projection_store(Arc::new(projection));
    runtime
        .start_session(AgentSessionStartParams {
            session_id: Some("hidden-loaded-session".to_string()),
            thread_id: Some("hidden-loaded-thread".to_string()),
            app_id: "thread-loaded-list-test".to_string(),
            workspace_id: None,
            business_object_ref: Some(BusinessObjectRef {
                kind: "agent.thread".to_string(),
                id: "hidden-loaded-thread".to_string(),
                title: None,
                uri: None,
                metadata: Some(json!({
                    "harness": {"hiddenFromUserRecents": true}
                })),
            }),
            locale: None,
        })
        .expect("hidden loaded session");
    let server = AppServer::with_runtime(runtime);
    initialize(&server).await;

    let empty = request(&server, 2, METHOD_THREAD_LOADED_LIST, json!({})).await;
    assert_eq!(empty["result"], json!({"data": [], "nextCursor": null}));

    let first = start_thread(&server, 3).await;
    let second = start_thread(&server, 4).await;
    let mut expected = [first, second];
    expected.sort();

    let first_page = request(&server, 5, METHOD_THREAD_LOADED_LIST, json!({"limit": 1})).await;
    assert_eq!(first_page["result"]["data"], json!([expected[0]]));
    assert_eq!(first_page["result"]["nextCursor"], json!(expected[0]));

    let second_page = request(
        &server,
        6,
        METHOD_THREAD_LOADED_LIST,
        json!({"cursor": expected[0], "limit": 1}),
    )
    .await;
    assert_eq!(second_page["result"]["data"], json!([expected[1]]));
    assert_eq!(second_page["result"]["nextCursor"], Value::Null);

    let stale_cursor = "00000000-0000-7000-8000-000000000000";
    let stale_page = request(
        &server,
        7,
        METHOD_THREAD_LOADED_LIST,
        json!({"cursor": stale_cursor, "limit": 1}),
    )
    .await;
    assert_eq!(stale_page["result"]["data"], json!([expected[0]]));

    let zero_limit = request(&server, 8, METHOD_THREAD_LOADED_LIST, json!({"limit": 0})).await;
    assert_eq!(zero_limit["result"]["data"], json!([expected[0]]));
    assert_eq!(zero_limit["result"]["nextCursor"], json!(expected[0]));

    let invalid = request_raw(
        &server,
        9,
        METHOD_THREAD_LOADED_LIST,
        json!({"cursor": "not-a-thread-id"}),
    )
    .await;
    assert_eq!(
        invalid.pointer("/error/code"),
        Some(&json!(error_codes::INVALID_REQUEST))
    );

    request(
        &server,
        10,
        METHOD_THREAD_DELETE,
        json!({"threadId": expected[0]}),
    )
    .await;
    let after_delete = request(&server, 11, METHOD_THREAD_LOADED_LIST, json!({})).await;
    assert_eq!(after_delete["result"]["data"], json!([expected[1]]));

    request(
        &server,
        12,
        METHOD_THREAD_ARCHIVE,
        json!({"threadId": expected[1]}),
    )
    .await;
    let after_archive = request(&server, 13, METHOD_THREAD_LOADED_LIST, json!({})).await;
    assert_eq!(
        after_archive["result"],
        json!({"data": [], "nextCursor": null})
    );
}

#[tokio::test]
async fn thread_read_does_not_load_a_cold_fork_but_resume_does() {
    let temp = TempDir::new().expect("cold fork read temp dir");
    let projection_path = temp.path().join("projection.sqlite");
    let initial = AppServer::with_runtime(
        RuntimeCore::with_backend(Arc::new(MockBackend)).with_projection_store(Arc::new(
            ProjectionStore::initialize(&projection_path).expect("initial projection store"),
        )),
    );
    initialize(&initial).await;
    let source_thread_id = start_thread(&initial, 2).await;
    let forked = request(
        &initial,
        3,
        METHOD_THREAD_FORK,
        json!({"threadId": source_thread_id}),
    )
    .await;
    let forked_thread_id = forked["result"]["thread"]["id"]
        .as_str()
        .expect("forked thread id")
        .to_string();
    drop(initial);

    let restarted = AppServer::with_runtime(
        RuntimeCore::with_backend(Arc::new(MockBackend)).with_projection_store(Arc::new(
            ProjectionStore::initialize(&projection_path).expect("restarted projection store"),
        )),
    );
    initialize(&restarted).await;
    let before_read = request(&restarted, 4, METHOD_THREAD_LOADED_LIST, json!({})).await;
    assert_eq!(
        before_read["result"],
        json!({"data": [], "nextCursor": null})
    );

    let read = request(
        &restarted,
        5,
        METHOD_THREAD_READ,
        json!({"threadId": forked_thread_id, "includeTurns": true}),
    )
    .await;
    assert_eq!(read["result"]["thread"]["id"], json!(forked_thread_id));
    let after_read = request(&restarted, 6, METHOD_THREAD_LOADED_LIST, json!({})).await;
    assert_eq!(
        after_read["result"],
        json!({"data": [], "nextCursor": null}),
        "thread/read must not hydrate a cold fork into the loaded owner"
    );

    request(
        &restarted,
        7,
        METHOD_THREAD_RESUME,
        json!({"threadId": forked_thread_id}),
    )
    .await;
    let after_resume = request(&restarted, 8, METHOD_THREAD_LOADED_LIST, json!({})).await;
    assert_eq!(after_resume["result"]["data"], json!([forked_thread_id]));
}

async fn initialize(server: &AppServer) {
    request(
        server,
        1,
        METHOD_INITIALIZE,
        json!({"clientInfo": {"name": "thread-loaded-list-test", "version": "1"}}),
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
    let response = request(
        server,
        id,
        METHOD_THREAD_START,
        json!({"model": "fixture-model", "modelProvider": "fixture-provider"}),
    )
    .await;
    let thread_id = response["result"]["thread"]["id"]
        .as_str()
        .expect("thread id")
        .to_string();
    assert_eq!(response["result"]["thread"]["sessionId"], json!(thread_id));
    assert_eq!(
        Uuid::parse_str(&thread_id)
            .expect("thread id UUID")
            .get_version_num(),
        7
    );
    thread_id
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
