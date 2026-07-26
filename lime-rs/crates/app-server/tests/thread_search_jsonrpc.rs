use std::sync::Arc;

use agent_protocol::{
    AgentInput, ItemId, ItemStatus, SessionId, Thread, ThreadHistoryChangeSet, ThreadId,
    ThreadItem, ThreadItemPayload, ThreadStatus, ThreadTurnsView, Turn, TurnAdmissionState,
    TurnApprovalState, TurnId, TurnItemsView, TurnQueueState, TurnStatus,
};
use app_server::{AppServer, MockBackend, ProjectionStore, RuntimeCore};
use app_server_protocol::protocol::v2::METHOD_THREAD_SEARCH;
use app_server_protocol::{error_codes, METHOD_INITIALIZE, METHOD_INITIALIZED};
use serde_json::{json, Value};
use thread_store::{
    ApplyThreadHistoryParams, ArchiveThreadParams, CreateThreadParams, ThreadStore,
};

#[tokio::test]
async fn thread_search_reads_canonical_content_with_exact_filters_and_cursors() {
    let temp = tempfile::tempdir().expect("thread search tempdir");
    let projection = ProjectionStore::initialize(temp.path().join("projection.sqlite"))
        .expect("thread search projection store");
    for timestamp in 1..=3 {
        seed_thread(
            &projection,
            &format!("app-{timestamp}"),
            timestamp,
            "appServer",
            false,
            "find the Needle in canonical content",
        )
        .await;
    }
    seed_thread(&projection, "cli-4", 4, "cli", false, "CLI needle content").await;
    seed_thread(
        &projection,
        "archived-5",
        5,
        "appServer",
        true,
        "archived needle content",
    )
    .await;

    let server = AppServer::with_runtime(
        RuntimeCore::with_backend(Arc::new(MockBackend))
            .with_projection_store(Arc::new(projection)),
    );
    initialize(&server).await;

    let interactive = request(
        &server,
        2,
        METHOD_THREAD_SEARCH,
        json!({"searchTerm": "needle"}),
    )
    .await;
    assert_eq!(interactive["result"]["data"][0]["thread"]["id"], "cli-4");
    assert_eq!(
        interactive["result"]["data"][0]["snippet"],
        "CLI needle content"
    );

    let first = request(
        &server,
        3,
        METHOD_THREAD_SEARCH,
        json!({
            "searchTerm": "needle",
            "sourceKinds": ["appServer"],
            "limit": 2
        }),
    )
    .await;
    assert_eq!(thread_ids(&first), vec!["app-3", "app-2"]);
    assert_eq!(
        first["result"]["data"][0]["snippet"],
        "find the Needle in canonical content"
    );
    assert_eq!(
        first["result"]["data"][0]["thread"]["status"]["type"],
        "idle"
    );
    let next_cursor = first["result"]["nextCursor"]
        .as_str()
        .expect("next cursor")
        .to_string();

    let second = request(
        &server,
        4,
        METHOD_THREAD_SEARCH,
        json!({
            "searchTerm": "needle",
            "sourceKinds": ["appServer"],
            "cursor": next_cursor,
            "limit": 2
        }),
    )
    .await;
    assert_eq!(thread_ids(&second), vec!["app-1"]);
    assert_eq!(second["result"]["nextCursor"], Value::Null);
    let backwards_cursor = second["result"]["backwardsCursor"]
        .as_str()
        .expect("backwards cursor")
        .to_string();

    let backwards = request(
        &server,
        5,
        METHOD_THREAD_SEARCH,
        json!({
            "searchTerm": "needle",
            "sourceKinds": ["appServer"],
            "cursor": backwards_cursor,
            "limit": 2,
            "sortDirection": "asc"
        }),
    )
    .await;
    assert_eq!(thread_ids(&backwards), vec!["app-2", "app-3"]);

    let archived = request(
        &server,
        6,
        METHOD_THREAD_SEARCH,
        json!({
            "searchTerm": "needle",
            "sourceKinds": ["appServer"],
            "archived": true
        }),
    )
    .await;
    assert_eq!(thread_ids(&archived), vec!["archived-5"]);

    let title_only = request(
        &server,
        7,
        METHOD_THREAD_SEARCH,
        json!({"searchTerm": "title-only", "sourceKinds": ["appServer"]}),
    )
    .await;
    assert_eq!(title_only["result"]["data"], json!([]));

    let zero_limit = request(
        &server,
        8,
        METHOD_THREAD_SEARCH,
        json!({
            "searchTerm": "needle",
            "sourceKinds": ["appServer"],
            "limit": 0
        }),
    )
    .await;
    assert_eq!(
        zero_limit["result"]["data"].as_array().map(Vec::len),
        Some(1)
    );
}

#[tokio::test]
async fn thread_search_rejects_blank_terms_and_mismatched_cursors() {
    let temp = tempfile::tempdir().expect("thread search validation tempdir");
    let projection = ProjectionStore::initialize(temp.path().join("projection.sqlite"))
        .expect("thread search validation projection store");
    for timestamp in 1..=2 {
        seed_thread(
            &projection,
            &format!("app-{timestamp}"),
            timestamp,
            "appServer",
            false,
            "needle haystack",
        )
        .await;
    }
    let server = AppServer::with_runtime(
        RuntimeCore::with_backend(Arc::new(MockBackend))
            .with_projection_store(Arc::new(projection)),
    );
    initialize(&server).await;

    let page = request(
        &server,
        2,
        METHOD_THREAD_SEARCH,
        json!({"searchTerm": "needle", "sourceKinds": ["appServer"], "limit": 1}),
    )
    .await;
    let cursor = page["result"]["nextCursor"].clone();
    for (id, params) in [
        (3, json!({"searchTerm": "   "})),
        (
            4,
            json!({
                "searchTerm": "haystack",
                "sourceKinds": ["appServer"],
                "cursor": cursor
            }),
        ),
        (
            5,
            json!({
                "searchTerm": "needle",
                "sourceKinds": ["appServer"],
                "cursor": "invalid"
            }),
        ),
    ] {
        let response = request_raw(&server, id, METHOD_THREAD_SEARCH, params).await;
        assert_eq!(
            response.pointer("/error/code"),
            Some(&json!(error_codes::INVALID_REQUEST)),
            "unexpected response: {response:#}"
        );
    }
}

async fn seed_thread(
    store: &ProjectionStore,
    thread_id: &str,
    timestamp: i64,
    source: &str,
    archived: bool,
    content: &str,
) {
    let thread = Thread {
        session_id: SessionId::new(format!("session-{thread_id}")),
        thread_id: ThreadId::new(thread_id),
        status: ThreadStatus::Idle,
        created_at_ms: timestamp,
        updated_at_ms: timestamp,
        archived: false,
        recency_at_ms: Some(timestamp),
        parent_thread_id: None,
        agent_path: None,
        agent_nickname: None,
        agent_role: None,
        last_task_message: None,
        agent_state: None,
        forked_from_id: None,
        preview: format!("title-only-{thread_id}"),
        model_provider: "openai".to_string(),
        product: None,
        name: None,
        metadata: json!({"source": source}),
        turns: Vec::new(),
        turns_view: ThreadTurnsView::NotLoaded,
    };
    store
        .create_thread(CreateThreadParams {
            thread: thread.clone(),
        })
        .await
        .expect("create searchable thread");
    store
        .apply_history(ApplyThreadHistoryParams {
            session_id: thread.session_id.clone(),
            thread_id: thread.thread_id.clone(),
            changes: ThreadHistoryChangeSet {
                sequence: 1,
                changed_turns: vec![Turn {
                    session_id: thread.session_id.clone(),
                    thread_id: thread.thread_id.clone(),
                    turn_id: TurnId::new("turn-1"),
                    status: TurnStatus::Completed,
                    admission: TurnAdmissionState::Accepted,
                    queue: TurnQueueState::NotQueued,
                    approval: TurnApprovalState::NotRequired,
                    items: Vec::new(),
                    items_view: TurnItemsView::NotLoaded,
                    error: None,
                    created_at_ms: timestamp,
                    updated_at_ms: timestamp,
                    started_at_ms: Some(timestamp),
                    completed_at_ms: Some(timestamp),
                    duration_ms: Some(0),
                }],
                changed_items: vec![ThreadItem {
                    session_id: thread.session_id.clone(),
                    thread_id: thread.thread_id.clone(),
                    turn_id: TurnId::new("turn-1"),
                    item_id: ItemId::new("user-1"),
                    sequence: 1,
                    ordinal: 1,
                    created_at_ms: timestamp,
                    updated_at_ms: timestamp,
                    completed_at_ms: Some(timestamp),
                    kind: agent_protocol::ItemKind::UserMessage,
                    status: ItemStatus::Completed,
                    payload: ThreadItemPayload::UserMessage {
                        content: vec![AgentInput::text(content)],
                        client_id: None,
                    },
                    metadata: json!({}),
                }],
                ..Default::default()
            },
        })
        .await
        .expect("persist searchable history");
    if archived {
        store
            .archive_thread(ArchiveThreadParams {
                thread_id: thread.thread_id,
            })
            .await
            .expect("archive searchable thread");
    }
}

fn thread_ids(response: &Value) -> Vec<&str> {
    response["result"]["data"]
        .as_array()
        .expect("thread search data")
        .iter()
        .map(|result| result["thread"]["id"].as_str().expect("thread id"))
        .collect()
}

async fn initialize(server: &AppServer) {
    request(
        server,
        1,
        METHOD_INITIALIZE,
        json!({"clientInfo": {"name": "thread-search-jsonrpc-test", "version": "1"}}),
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
