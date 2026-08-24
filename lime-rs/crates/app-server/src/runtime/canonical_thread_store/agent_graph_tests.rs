use agent_protocol::{
    CollabAgentState, CollabAgentStatus, SessionId, SortDirection, Thread, ThreadId, ThreadStatus,
    ThreadTurnsView, Turn, TurnAdmissionState, TurnApprovalState, TurnError, TurnId, TurnItemsView,
    TurnQueueState, TurnStatus,
};
use futures::executor::block_on;
use thread_store::{
    AgentGraphStore, AgentIdentity, AgentIdentityStore, CreateThreadParams, ListThreadsParams,
    PageRequest, ReadThreadParams, ThreadSpawnEdgeStatus, ThreadSpawnParent,
};

use super::super::derive_agent_state;
use super::ProjectionStore;

fn store() -> (tempfile::TempDir, ProjectionStore) {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ProjectionStore::initialize(temp.path().join("projection.sqlite"))
        .expect("projection store");
    (temp, store)
}

fn thread_id(id: &str) -> ThreadId {
    ThreadId::new(id)
}

fn canonical_thread(id: &str) -> Thread {
    Thread {
        session_id: SessionId::new(format!("session-{id}")),
        thread_id: thread_id(id),
        status: ThreadStatus::Idle,
        created_at_ms: 100,
        updated_at_ms: 200,
        archived: false,
        recency_at_ms: Some(200),
        parent_thread_id: None,
        agent_path: None,
        agent_nickname: None,
        agent_role: None,
        last_task_message: None,
        agent_state: None,
        forked_from_id: None,
        preview: String::new(),
        model_provider: "openai".to_string(),
        product: None,
        name: None,
        metadata: serde_json::Value::Null,
        turns: Vec::new(),
        turns_view: ThreadTurnsView::NotLoaded,
    }
}

fn canonical_turn(thread: &Thread, status: TurnStatus, message: Option<&str>) -> Turn {
    Turn {
        session_id: thread.session_id.clone(),
        thread_id: thread.thread_id.clone(),
        turn_id: TurnId::new("turn-latest"),
        status,
        admission: TurnAdmissionState::Accepted,
        queue: TurnQueueState::NotQueued,
        approval: TurnApprovalState::NotRequired,
        items: Vec::new(),
        items_view: TurnItemsView::NotLoaded,
        error: message.map(|message| TurnError {
            message: message.to_string(),
            code: None,
            details: None,
        }),
        created_at_ms: 201,
        updated_at_ms: 202,
        started_at_ms: Some(201),
        completed_at_ms: status.is_terminal().then_some(202),
        duration_ms: status.is_terminal().then_some(1),
    }
}

#[test]
fn canonical_agent_state_uses_edge_thread_and_latest_turn_facts() {
    let mut thread = canonical_thread("child-status");
    assert_eq!(
        derive_agent_state(&thread, ThreadSpawnEdgeStatus::Open).status,
        CollabAgentStatus::PendingInit
    );

    thread.status = ThreadStatus::Active {
        active_flags: Vec::new(),
    };
    assert_eq!(
        derive_agent_state(&thread, ThreadSpawnEdgeStatus::Open).status,
        CollabAgentStatus::Running
    );

    thread.status = ThreadStatus::Idle;
    for (turn_status, expected) in [
        (TurnStatus::InProgress, CollabAgentStatus::Running),
        (TurnStatus::Interrupted, CollabAgentStatus::Interrupted),
        (TurnStatus::Completed, CollabAgentStatus::Completed),
    ] {
        thread.turns = vec![canonical_turn(&thread, turn_status, None)];
        assert_eq!(
            derive_agent_state(&thread, ThreadSpawnEdgeStatus::Open).status,
            expected
        );
    }

    thread.turns = vec![canonical_turn(
        &thread,
        TurnStatus::Failed,
        Some("provider failed"),
    )];
    assert_eq!(
        derive_agent_state(&thread, ThreadSpawnEdgeStatus::Open),
        CollabAgentState {
            status: CollabAgentStatus::Errored,
            message: Some("provider failed".to_string()),
        }
    );
    assert_eq!(
        derive_agent_state(&thread, ThreadSpawnEdgeStatus::Closed).status,
        CollabAgentStatus::Shutdown
    );
    assert_eq!(
        derive_agent_state(&thread, ThreadSpawnEdgeStatus::Pending).status,
        CollabAgentStatus::NotFound
    );
}

#[test]
fn pending_spawn_is_hidden_until_atomic_commit() {
    let (_temp, store) = store();
    let parent = canonical_thread("pending-parent");
    let child = canonical_thread("pending-child");
    store
        .create_thread_sync(CreateThreadParams {
            thread: parent.clone(),
        })
        .expect("create parent");
    block_on(store.create_pending_thread_spawn_edge(
        parent.thread_id.clone(),
        child.thread_id.clone(),
        child.session_id.to_string(),
    ))
    .expect("reserve pending child");
    store
        .create_thread_sync(CreateThreadParams {
            thread: child.clone(),
        })
        .expect("create hidden child");

    assert!(store
        .read_thread_sync(ReadThreadParams {
            thread_id: child.thread_id.clone(),
            include_archived: true,
            turns_view: ThreadTurnsView::NotLoaded,
        })
        .expect("read pending child")
        .is_none());
    assert_eq!(
        store
            .list_threads_sync(ListThreadsParams {
                include_archived: true,
                section: None,
                project: None,
                sort_by_section_position: false,
                page: PageRequest {
                    cursor: None,
                    limit: 10,
                    sort_direction: SortDirection::Asc,
                },
            })
            .expect("list visible threads")
            .data
            .into_iter()
            .map(|thread| thread.thread_id)
            .collect::<Vec<_>>(),
        vec![parent.thread_id.clone()]
    );
    assert_eq!(
        block_on(store.read_thread_spawn_parent(child.thread_id.clone()))
            .expect("read pending parent")
            .expect("pending parent")
            .status,
        ThreadSpawnEdgeStatus::Pending
    );
    assert_eq!(
        store
            .list_pending_thread_spawn_intents_sync()
            .expect("list pending intents"),
        vec![(
            parent.thread_id.clone(),
            child.thread_id.clone(),
            child.session_id.to_string(),
        )]
    );

    assert!(block_on(
        store.commit_pending_thread_spawn_edge(
            child.thread_id.clone(),
            child.session_id.to_string(),
        )
    )
    .expect("commit pending child"));
    let visible = store
        .read_thread_sync(ReadThreadParams {
            thread_id: child.thread_id,
            include_archived: true,
            turns_view: ThreadTurnsView::NotLoaded,
        })
        .expect("read committed child")
        .expect("committed child");
    assert_eq!(visible.parent_thread_id, Some(parent.thread_id));
    assert!(store
        .list_pending_thread_spawn_intents_sync()
        .expect("pending intents after commit")
        .is_empty());
}

#[test]
fn canonical_thread_spawn_edge_upsert_reparents_and_replaces_status() {
    let (_temp, store) = store();
    let first_parent = thread_id("parent-1");
    let second_parent = thread_id("parent-2");
    let child = thread_id("child-1");

    block_on(store.upsert_thread_spawn_edge(
        first_parent.clone(),
        child.clone(),
        ThreadSpawnEdgeStatus::Open,
    ))
    .expect("insert edge");
    block_on(store.upsert_thread_spawn_edge(
        second_parent.clone(),
        child.clone(),
        ThreadSpawnEdgeStatus::Closed,
    ))
    .expect("reparent edge");

    assert_eq!(
        block_on(store.list_thread_spawn_children(first_parent, None))
            .expect("old parent children"),
        Vec::<ThreadId>::new()
    );
    assert_eq!(
        block_on(
            store.list_thread_spawn_children(
                second_parent.clone(),
                Some(ThreadSpawnEdgeStatus::Open),
            )
        )
        .expect("open children"),
        Vec::<ThreadId>::new()
    );
    assert_eq!(
        block_on(
            store.list_thread_spawn_children(second_parent, Some(ThreadSpawnEdgeStatus::Closed),)
        )
        .expect("closed children"),
        vec![child]
    );
}

#[test]
fn canonical_thread_spawn_parent_reads_open_and_closed_status() {
    let (_temp, store) = store();
    let parent = thread_id("parent");
    let child = thread_id("child");

    block_on(store.upsert_thread_spawn_edge(
        parent.clone(),
        child.clone(),
        ThreadSpawnEdgeStatus::Open,
    ))
    .expect("insert edge");
    assert_eq!(
        block_on(store.read_thread_spawn_parent(child.clone())).expect("read open parent"),
        Some(ThreadSpawnParent {
            parent_thread_id: parent.clone(),
            status: ThreadSpawnEdgeStatus::Open,
        })
    );

    block_on(store.set_thread_spawn_edge_status(child.clone(), ThreadSpawnEdgeStatus::Closed))
        .expect("close edge");
    assert_eq!(
        block_on(store.read_thread_spawn_parent(child)).expect("read closed parent"),
        Some(ThreadSpawnParent {
            parent_thread_id: parent,
            status: ThreadSpawnEdgeStatus::Closed,
        })
    );
}

#[test]
fn canonical_thread_spawn_parent_returns_none_for_missing_child() {
    let (_temp, store) = store();

    assert_eq!(
        block_on(store.read_thread_spawn_parent(thread_id("missing")))
            .expect("read missing parent"),
        None
    );
}

#[test]
fn canonical_thread_reads_join_durable_parent_and_agent_identity() {
    let (_temp, store) = store();
    let parent = thread_id("parent");
    let child = thread_id("child");
    store
        .create_thread_sync(CreateThreadParams {
            thread: canonical_thread("parent"),
        })
        .expect("create parent thread");
    store
        .create_thread_sync(CreateThreadParams {
            thread: canonical_thread("child"),
        })
        .expect("create child thread");
    block_on(store.upsert_thread_spawn_edge(
        parent.clone(),
        child.clone(),
        ThreadSpawnEdgeStatus::Open,
    ))
    .expect("insert child edge");
    block_on(store.upsert_agent_identity(AgentIdentity {
        root_thread_id: parent.clone(),
        thread_id: child.clone(),
        agent_path: "/root/research".to_string(),
        nickname: Some("Euclid".to_string()),
        role: Some("researcher".to_string()),
        last_task_message: Some("audit canonical roster".to_string()),
    }))
    .expect("insert child identity");

    let read = store
        .read_thread_sync(ReadThreadParams {
            thread_id: child.clone(),
            include_archived: false,
            turns_view: ThreadTurnsView::NotLoaded,
        })
        .expect("read child")
        .expect("child exists");
    assert_eq!(read.parent_thread_id, Some(parent.clone()));
    assert_eq!(read.agent_path.as_deref(), Some("/root/research"));
    assert_eq!(read.agent_nickname.as_deref(), Some("Euclid"));
    assert_eq!(read.agent_role.as_deref(), Some("researcher"));
    assert_eq!(
        read.last_task_message.as_deref(),
        Some("audit canonical roster")
    );
    assert_eq!(
        read.agent_state,
        Some(CollabAgentState {
            status: CollabAgentStatus::PendingInit,
            message: None,
        })
    );
    assert!(read.metadata.is_null());

    let page = store
        .list_threads_sync(ListThreadsParams {
            include_archived: false,
            section: None,
            project: None,
            sort_by_section_position: false,
            page: PageRequest {
                cursor: None,
                limit: 10,
                sort_direction: SortDirection::Asc,
            },
        })
        .expect("list threads");
    let listed = page
        .data
        .iter()
        .find(|thread| thread.thread_id == child)
        .expect("listed child");
    assert_eq!(listed.parent_thread_id, Some(parent));
    assert_eq!(listed.agent_path.as_deref(), Some("/root/research"));
    assert_eq!(
        listed.agent_state.as_ref().map(|state| state.status),
        Some(CollabAgentStatus::PendingInit)
    );
}

#[test]
fn canonical_thread_reads_project_closed_spawn_edge_without_erasing_identity() {
    let (_temp, store) = store();
    let parent = thread_id("parent");
    let child = thread_id("child");
    store
        .create_thread_sync(CreateThreadParams {
            thread: canonical_thread("child"),
        })
        .expect("create child thread");
    block_on(store.upsert_thread_spawn_edge(
        parent.clone(),
        child.clone(),
        ThreadSpawnEdgeStatus::Closed,
    ))
    .expect("insert closed child edge");
    block_on(store.upsert_agent_identity(AgentIdentity {
        root_thread_id: parent.clone(),
        thread_id: child.clone(),
        agent_path: "/root/closed".to_string(),
        nickname: None,
        role: None,
        last_task_message: None,
    }))
    .expect("insert closed child identity");

    let read = store
        .read_thread_sync(ReadThreadParams {
            thread_id: child,
            include_archived: false,
            turns_view: ThreadTurnsView::NotLoaded,
        })
        .expect("read closed child")
        .expect("closed child exists");
    assert_eq!(read.parent_thread_id, Some(parent));
    assert_eq!(read.agent_path.as_deref(), Some("/root/closed"));
    assert_eq!(
        read.agent_state,
        Some(CollabAgentState {
            status: CollabAgentStatus::Shutdown,
            message: None,
        })
    );
}

#[test]
fn canonical_thread_spawn_edge_lists_stable_breadth_first_descendants() {
    let (_temp, store) = store();
    let root = thread_id("root");
    let child_a = thread_id("child-a");
    let child_b = thread_id("child-b");
    let child_closed = thread_id("child-c");
    let grandchild_a = thread_id("grandchild-a");
    let grandchild_closed = thread_id("grandchild-b");
    let hidden_below_closed = thread_id("great-grandchild-a");

    for (parent, child, status) in [
        (root.clone(), child_b.clone(), ThreadSpawnEdgeStatus::Open),
        (root.clone(), child_a.clone(), ThreadSpawnEdgeStatus::Open),
        (
            root.clone(),
            child_closed.clone(),
            ThreadSpawnEdgeStatus::Closed,
        ),
        (
            child_b.clone(),
            grandchild_closed.clone(),
            ThreadSpawnEdgeStatus::Closed,
        ),
        (
            child_a.clone(),
            grandchild_a.clone(),
            ThreadSpawnEdgeStatus::Open,
        ),
        (
            child_closed.clone(),
            hidden_below_closed.clone(),
            ThreadSpawnEdgeStatus::Open,
        ),
    ] {
        block_on(store.upsert_thread_spawn_edge(parent, child, status)).expect("insert edge");
    }

    assert_eq!(
        block_on(store.list_thread_spawn_children(root.clone(), None)).expect("all children"),
        vec![child_a.clone(), child_b.clone(), child_closed.clone()]
    );
    assert_eq!(
        block_on(store.list_thread_spawn_descendants(root.clone(), None)).expect("all descendants"),
        vec![
            child_a.clone(),
            child_b.clone(),
            child_closed,
            grandchild_a.clone(),
            grandchild_closed,
            hidden_below_closed,
        ]
    );
    assert_eq!(
        block_on(store.list_thread_spawn_descendants(root, Some(ThreadSpawnEdgeStatus::Open),))
            .expect("open descendants"),
        vec![child_a, child_b, grandchild_a]
    );
}

#[test]
fn canonical_thread_spawn_edge_close_is_durable_and_preserves_audit_state() {
    let (temp, store) = store();
    let path = temp.path().join("projection.sqlite");
    let parent = thread_id("parent");
    let child = thread_id("child");

    block_on(store.upsert_thread_spawn_edge(
        parent.clone(),
        child.clone(),
        ThreadSpawnEdgeStatus::Open,
    ))
    .expect("insert edge");
    block_on(store.set_thread_spawn_edge_status(child.clone(), ThreadSpawnEdgeStatus::Closed))
        .expect("close edge");
    drop(store);

    let reopened = ProjectionStore::initialize(path).expect("reopen projection store");
    assert_eq!(
        block_on(reopened.read_thread_spawn_parent(child.clone()))
            .expect("read parent after reopen"),
        Some(ThreadSpawnParent {
            parent_thread_id: parent.clone(),
            status: ThreadSpawnEdgeStatus::Closed,
        })
    );
    assert_eq!(
        block_on(
            reopened.list_thread_spawn_children(parent.clone(), Some(ThreadSpawnEdgeStatus::Open),)
        )
        .expect("open children after reopen"),
        Vec::<ThreadId>::new()
    );
    assert_eq!(
        block_on(reopened.list_thread_spawn_children(parent, Some(ThreadSpawnEdgeStatus::Closed),))
            .expect("closed children after reopen"),
        vec![child]
    );
}

#[test]
fn canonical_thread_spawn_edge_missing_status_update_is_a_no_op() {
    let (_temp, store) = store();

    block_on(
        store.set_thread_spawn_edge_status(thread_id("missing"), ThreadSpawnEdgeStatus::Closed),
    )
    .expect("missing edge update");
}

#[test]
fn canonical_thread_spawn_edge_rejects_self_edge_without_mutation() {
    let (_temp, store) = store();
    let outside = thread_id("outside");
    let thread = thread_id("thread");

    block_on(store.upsert_thread_spawn_edge(
        outside.clone(),
        thread.clone(),
        ThreadSpawnEdgeStatus::Closed,
    ))
    .expect("insert existing edge");
    let error = block_on(store.upsert_thread_spawn_edge(
        thread.clone(),
        thread.clone(),
        ThreadSpawnEdgeStatus::Open,
    ))
    .expect_err("self edge must fail closed");

    assert!(error.to_string().contains("cannot target itself"));
    assert_eq!(
        block_on(store.list_thread_spawn_children(outside, Some(ThreadSpawnEdgeStatus::Closed),))
            .expect("existing parent after failure"),
        vec![thread.clone()]
    );
    assert_eq!(
        block_on(store.list_thread_spawn_children(thread, None))
            .expect("self children after failure"),
        Vec::<ThreadId>::new()
    );
}

#[test]
fn canonical_thread_spawn_edge_rejects_cycle_and_preserves_existing_parent() {
    let (_temp, store) = store();
    let outside = thread_id("outside");
    let root = thread_id("root");
    let child = thread_id("child");
    let grandchild = thread_id("grandchild");

    for (parent, child, status) in [
        (outside.clone(), root.clone(), ThreadSpawnEdgeStatus::Closed),
        (root.clone(), child.clone(), ThreadSpawnEdgeStatus::Open),
        (
            child.clone(),
            grandchild.clone(),
            ThreadSpawnEdgeStatus::Open,
        ),
    ] {
        block_on(store.upsert_thread_spawn_edge(parent, child, status)).expect("insert edge");
    }

    let error = block_on(store.upsert_thread_spawn_edge(
        grandchild.clone(),
        root.clone(),
        ThreadSpawnEdgeStatus::Open,
    ))
    .expect_err("cycle attempt must fail closed");

    assert!(error.to_string().contains("create a cycle"));
    assert_eq!(
        block_on(
            store.list_thread_spawn_children(outside.clone(), Some(ThreadSpawnEdgeStatus::Closed),)
        )
        .expect("existing parent after failure"),
        vec![root.clone()]
    );
    assert_eq!(
        block_on(store.list_thread_spawn_descendants(outside, None))
            .expect("descendants after failure"),
        vec![root, child, grandchild]
    );
}
