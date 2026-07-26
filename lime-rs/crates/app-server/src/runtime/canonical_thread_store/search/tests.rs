use super::super::ProjectionStore;
use agent_protocol::{
    AgentInput, ItemId, ItemKind, ItemStatus, SessionId, SortDirection, Thread,
    ThreadHistoryChangeSet, ThreadId, ThreadItem, ThreadItemPayload, ThreadStatus, ThreadTurnsView,
    Turn, TurnAdmissionState, TurnApprovalState, TurnId, TurnItemsView, TurnQueueState, TurnStatus,
};
use futures::executor::block_on;
use serde_json::json;
use thread_store::{
    ApplyThreadHistoryParams, CreateThreadParams, SearchThreadsParams, ThreadSearchSortKey,
    ThreadSearchSourceKind, ThreadStore,
};

#[test]
fn searches_conversation_content_with_stable_forward_and_reverse_cursors() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ProjectionStore::initialize(temp.path().join("projection.sqlite"))
        .expect("projection store");
    for timestamp in 1..=3 {
        let thread = canonical_thread(timestamp);
        block_on(store.create_thread(CreateThreadParams {
            thread: thread.clone(),
        }))
        .expect("create thread");
        block_on(store.apply_history(ApplyThreadHistoryParams {
            session_id: thread.session_id.clone(),
            thread_id: thread.thread_id.clone(),
            changes: ThreadHistoryChangeSet {
                sequence: 1,
                changed_turns: vec![canonical_turn(&thread)],
                changed_items: vec![user_message(&thread, "find the Needle in content")],
                ..Default::default()
            },
        }))
        .expect("apply history");
    }

    let first = block_on(store.search_threads(search_params(None, 2, SortDirection::Desc)))
        .expect("first search page");
    assert_eq!(
        first
            .data
            .iter()
            .map(|result| result.thread.thread_id.as_str())
            .collect::<Vec<_>>(),
        vec!["thread-3", "thread-2"]
    );
    assert_eq!(first.data[0].snippet, "find the Needle in content");
    assert!(first.next_cursor.is_some());

    let second =
        block_on(store.search_threads(search_params(first.next_cursor, 2, SortDirection::Desc)))
            .expect("second search page");
    assert_eq!(second.data[0].thread.thread_id.as_str(), "thread-1");

    let backwards = block_on(store.search_threads(search_params(
        second.backwards_cursor,
        2,
        SortDirection::Asc,
    )))
    .expect("reverse search page");
    assert_eq!(
        backwards
            .data
            .iter()
            .map(|result| result.thread.thread_id.as_str())
            .collect::<Vec<_>>(),
        vec!["thread-2", "thread-3"]
    );
}

#[test]
fn rejects_cursor_reuse_for_a_different_search_term() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ProjectionStore::initialize(temp.path().join("projection.sqlite"))
        .expect("projection store");
    for timestamp in 1..=2 {
        let thread = canonical_thread(timestamp);
        block_on(store.create_thread(CreateThreadParams {
            thread: thread.clone(),
        }))
        .expect("create thread");
        block_on(store.apply_history(ApplyThreadHistoryParams {
            session_id: thread.session_id.clone(),
            thread_id: thread.thread_id.clone(),
            changes: ThreadHistoryChangeSet {
                sequence: 1,
                changed_turns: vec![canonical_turn(&thread)],
                changed_items: vec![user_message(&thread, "needle and haystack")],
                ..Default::default()
            },
        }))
        .expect("apply history");
    }
    let page = block_on(store.search_threads(search_params(None, 1, SortDirection::Desc)))
        .expect("search page");
    let error = block_on(store.search_threads(SearchThreadsParams {
        cursor: page.next_cursor,
        search_term: "haystack".to_string(),
        ..search_params(None, 1, SortDirection::Desc)
    }))
    .expect_err("cursor must be bound to the search term");
    assert!(error.to_string().contains("invalid cursor"));
}

fn search_params(
    cursor: Option<thread_store::StoreCursor>,
    page_size: usize,
    sort_direction: SortDirection,
) -> SearchThreadsParams {
    SearchThreadsParams {
        cursor,
        page_size,
        sort_key: ThreadSearchSortKey::CreatedAt,
        sort_direction,
        source_kinds: vec![ThreadSearchSourceKind::AppServer],
        archived: false,
        search_term: "needle".to_string(),
    }
}

fn canonical_thread(timestamp: i64) -> Thread {
    Thread {
        session_id: SessionId::new(format!("session-{timestamp}")),
        thread_id: ThreadId::new(format!("thread-{timestamp}")),
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
        preview: format!("title-only-{timestamp}"),
        model_provider: "openai".to_string(),
        product: None,
        name: None,
        metadata: json!({"source": "appServer"}),
        turns: Vec::new(),
        turns_view: ThreadTurnsView::NotLoaded,
    }
}

fn canonical_turn(thread: &Thread) -> Turn {
    Turn {
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
        created_at_ms: 1,
        updated_at_ms: 1,
        started_at_ms: Some(1),
        completed_at_ms: Some(1),
        duration_ms: Some(0),
    }
}

fn user_message(thread: &Thread, text: &str) -> ThreadItem {
    ThreadItem {
        session_id: thread.session_id.clone(),
        thread_id: thread.thread_id.clone(),
        turn_id: TurnId::new("turn-1"),
        item_id: ItemId::new("user-1"),
        sequence: 1,
        ordinal: 1,
        created_at_ms: 1,
        updated_at_ms: 1,
        completed_at_ms: Some(1),
        kind: ItemKind::UserMessage,
        status: ItemStatus::Completed,
        payload: ThreadItemPayload::UserMessage {
            content: vec![AgentInput::text(text)],
            client_id: None,
        },
        metadata: json!({}),
    }
}
