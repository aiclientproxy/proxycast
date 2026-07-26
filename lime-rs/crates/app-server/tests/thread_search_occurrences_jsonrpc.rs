use std::sync::Arc;

use agent_protocol::{
    AgentInput, ItemId, ItemStatus, SessionId, Thread, ThreadHistoryChangeSet, ThreadId,
    ThreadItem, ThreadItemPayload, ThreadStatus, ThreadTurnsView, Turn, TurnAdmissionState,
    TurnApprovalState, TurnId, TurnItemsView, TurnQueueState, TurnStatus,
};
use app_server::{AppServer, MockBackend, ProjectionStore, RuntimeCore};
use app_server_protocol::protocol::v2::{
    METHOD_THREAD_SEARCH_OCCURRENCES, METHOD_THREAD_TURNS_LIST,
};
use app_server_protocol::{error_codes, METHOD_INITIALIZE, METHOD_INITIALIZED};
use serde_json::{json, Value};
use thread_store::{
    ApplyThreadHistoryParams, ArchiveThreadParams, CreateThreadParams, ThreadStore,
};

const THREAD_ID: &str = "019f9b19-17a2-78b2-84d7-ce881fcf0617";

#[tokio::test]
async fn thread_search_occurrences_reads_archived_cold_history_and_paginates() {
    let temp = tempfile::tempdir().expect("thread search tempdir");
    let projection = ProjectionStore::initialize(temp.path().join("projection.sqlite"))
        .expect("thread search projection store");
    seed_history(&projection).await;
    projection
        .archive_thread(ArchiveThreadParams {
            thread_id: ThreadId::new(THREAD_ID),
        })
        .await
        .expect("archive searchable thread");

    let server = AppServer::with_runtime(
        RuntimeCore::with_backend(Arc::new(MockBackend))
            .with_projection_store(Arc::new(projection)),
    );
    initialize(&server).await;

    let first = request(
        &server,
        2,
        METHOD_THREAD_SEARCH_OCCURRENCES,
        json!({"threadId": THREAD_ID, "searchTerm": "needle", "limit": 3}),
    )
    .await;
    assert_eq!(
        first["result"]["data"]
            .as_array()
            .expect("first occurrence page")
            .iter()
            .map(|item| item["itemId"].as_str().expect("item id"))
            .collect::<Vec<_>>(),
        vec!["item_user-1", "item_user-1", "item_user-1"]
    );
    assert_eq!(
        first["result"]["data"]
            .as_array()
            .expect("first occurrence page")
            .iter()
            .map(|item| item["snippetMatchRange"]["start"].as_u64())
            .collect::<Vec<_>>(),
        vec![Some(0), Some(7), Some(14)]
    );
    let cursor = first["result"]["nextCursor"]
        .as_str()
        .expect("next cursor")
        .to_string();

    let second = request(
        &server,
        3,
        METHOD_THREAD_SEARCH_OCCURRENCES,
        json!({
            "threadId": THREAD_ID,
            "searchTerm": "needle",
            "cursor": cursor,
            "limit": 3
        }),
    )
    .await;
    assert_eq!(
        second["result"]["data"]
            .as_array()
            .expect("second occurrence page")
            .iter()
            .map(|item| item["itemId"].as_str().expect("item id"))
            .collect::<Vec<_>>(),
        vec!["item_user-1", "item_steer-1", "item_final-1"]
    );
    assert_eq!(second["result"]["data"][2]["snippet"], "😀 Final needle");
    assert_eq!(
        second["result"]["data"][2]["snippetMatchRange"],
        json!({"start": 9, "end": 15})
    );
    assert_eq!(second["result"]["nextCursor"], Value::Null);

    let turn_cursor = second["result"]["data"][2]["turnCursor"]
        .as_str()
        .expect("inclusive turn cursor");
    let turn_page = request(
        &server,
        4,
        METHOD_THREAD_TURNS_LIST,
        json!({
            "threadId": THREAD_ID,
            "cursor": turn_cursor,
            "limit": 1,
            "sortDirection": "asc",
            "itemsView": "notLoaded"
        }),
    )
    .await;
    assert_eq!(turn_page["result"]["data"][0]["id"], "turn-1");

    for (id, search_term) in [(5, "commentary"), (6, "obsolete")] {
        let excluded = request(
            &server,
            id,
            METHOD_THREAD_SEARCH_OCCURRENCES,
            json!({"threadId": THREAD_ID, "searchTerm": search_term}),
        )
        .await;
        assert_eq!(excluded["result"], json!({"data": [], "nextCursor": null}));
    }

    let zero_limit = request(
        &server,
        7,
        METHOD_THREAD_SEARCH_OCCURRENCES,
        json!({"threadId": THREAD_ID, "searchTerm": "needle", "limit": 0}),
    )
    .await;
    assert_eq!(
        zero_limit["result"]["data"].as_array().map(Vec::len),
        Some(1)
    );
}

#[tokio::test]
async fn thread_search_occurrences_rejects_invalid_scope_and_missing_threads() {
    let temp = tempfile::tempdir().expect("thread search validation tempdir");
    let projection = ProjectionStore::initialize(temp.path().join("projection.sqlite"))
        .expect("thread search validation projection store");
    seed_history(&projection).await;
    let server = AppServer::with_runtime(
        RuntimeCore::with_backend(Arc::new(MockBackend))
            .with_projection_store(Arc::new(projection)),
    );
    initialize(&server).await;

    let page = request(
        &server,
        2,
        METHOD_THREAD_SEARCH_OCCURRENCES,
        json!({"threadId": THREAD_ID, "searchTerm": "needle", "limit": 1}),
    )
    .await;
    let cursor = page["result"]["nextCursor"].clone();

    for (id, params) in [
        (3, json!({"threadId": "not-a-uuid", "searchTerm": "needle"})),
        (4, json!({"threadId": THREAD_ID, "searchTerm": "   "})),
        (
            5,
            json!({"threadId": THREAD_ID, "searchTerm": "Needle", "cursor": cursor}),
        ),
        (
            6,
            json!({"threadId": THREAD_ID, "searchTerm": "needle", "cursor": "invalid"}),
        ),
        (
            7,
            json!({
                "threadId": "019f9b19-17a2-78b2-84d7-ce881fcf0618",
                "searchTerm": "needle"
            }),
        ),
    ] {
        let response = request_raw(&server, id, METHOD_THREAD_SEARCH_OCCURRENCES, params).await;
        assert_eq!(
            response.pointer("/error/code"),
            Some(&json!(error_codes::INVALID_REQUEST)),
            "unexpected response: {response:#}"
        );
    }
}

async fn seed_history(store: &ProjectionStore) {
    let thread = Thread {
        session_id: SessionId::new(THREAD_ID),
        thread_id: ThreadId::new(THREAD_ID),
        status: ThreadStatus::Idle,
        created_at_ms: 1,
        updated_at_ms: 2,
        archived: false,
        recency_at_ms: Some(2),
        parent_thread_id: None,
        agent_path: None,
        agent_nickname: None,
        agent_role: None,
        last_task_message: None,
        agent_state: None,
        forked_from_id: None,
        preview: "search fixture".to_string(),
        model_provider: "fixture-provider".to_string(),
        product: None,
        name: None,
        metadata: json!({}),
        turns: Vec::new(),
        turns_view: ThreadTurnsView::NotLoaded,
    };
    store
        .create_thread(CreateThreadParams {
            thread: thread.clone(),
        })
        .await
        .expect("create searchable thread");
    let turn = Turn {
        session_id: thread.session_id.clone(),
        thread_id: thread.thread_id.clone(),
        turn_id: TurnId::new("turn-1"),
        status: TurnStatus::Completed,
        admission: TurnAdmissionState::Accepted,
        queue: TurnQueueState::Running,
        approval: TurnApprovalState::NotRequired,
        items: Vec::new(),
        items_view: TurnItemsView::NotLoaded,
        error: None,
        created_at_ms: 10,
        updated_at_ms: 20,
        started_at_ms: Some(10),
        completed_at_ms: Some(20),
        duration_ms: Some(10),
    };
    let items = vec![
        user_item(
            &thread,
            "user-1",
            1,
            vec![
                AgentInput::text("Nee"),
                AgentInput::text("dle needle needle needle"),
            ],
        ),
        user_item(
            &thread,
            "steer-1",
            2,
            vec![AgentInput::text("steer toward needle")],
        ),
        agent_item(
            &thread,
            "commentary-1",
            3,
            "commentary needle",
            "commentary",
        ),
        agent_item(&thread, "final-old", 4, "obsolete needle", "final_answer"),
        agent_item(
            &thread,
            "final-1",
            5,
            "😀 **Final**  \nneedle",
            "final_answer",
        ),
    ];
    store
        .apply_history(ApplyThreadHistoryParams {
            session_id: thread.session_id.clone(),
            thread_id: thread.thread_id.clone(),
            changes: ThreadHistoryChangeSet {
                sequence: 5,
                changed_turns: vec![turn],
                changed_items: items,
                ..Default::default()
            },
        })
        .await
        .expect("persist searchable history");
}

fn user_item(thread: &Thread, item_id: &str, ordinal: u64, content: Vec<AgentInput>) -> ThreadItem {
    item(
        thread,
        item_id,
        ordinal,
        ThreadItemPayload::UserMessage {
            content,
            client_id: None,
        },
    )
}

fn agent_item(thread: &Thread, item_id: &str, ordinal: u64, text: &str, phase: &str) -> ThreadItem {
    item(
        thread,
        item_id,
        ordinal,
        ThreadItemPayload::AgentMessage {
            text: text.to_string(),
            phase: Some(phase.to_string()),
            content_parts: Vec::new(),
        },
    )
}

fn item(thread: &Thread, item_id: &str, ordinal: u64, payload: ThreadItemPayload) -> ThreadItem {
    ThreadItem {
        session_id: thread.session_id.clone(),
        thread_id: thread.thread_id.clone(),
        turn_id: TurnId::new("turn-1"),
        item_id: ItemId::new(item_id),
        sequence: ordinal,
        ordinal,
        created_at_ms: ordinal as i64,
        updated_at_ms: ordinal as i64,
        completed_at_ms: Some(ordinal as i64),
        kind: payload.kind(),
        status: ItemStatus::Completed,
        payload,
        metadata: json!({}),
    }
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
