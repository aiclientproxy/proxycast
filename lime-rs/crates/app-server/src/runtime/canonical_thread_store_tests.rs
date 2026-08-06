use agent_protocol::{
    AgentInput, ImageDetail, ItemId, ItemStatus, PlanStepStatus, SessionId, SortDirection,
    TextElement, Thread, ThreadHistoryChangeSet, ThreadId, ThreadItem, ThreadItemPayload,
    ThreadStatus, ThreadTurnsView, Turn, TurnAdmissionState, TurnApprovalState, TurnId,
    TurnItemsView, TurnQueueState, TurnStatus,
};
use app_server_protocol::{
    AgentEvent, AgentSession, AgentSessionListParams, AgentSessionStatus, BusinessObjectRef,
};
use futures::executor::block_on;
use rusqlite::Connection;
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use thread_store::{
    AppendThreadItemsParams, ApplyThreadHistoryParams, ArchiveThreadParams, CreateThreadParams,
    ListItemsParams, ListThreadsParams, ListTurnsParams, PageRequest, ReadThreadParams,
    ThreadMetadataPatch, ThreadStore, UpdateThreadMetadataParams,
};

use super::canonical_rollout::RolloutStore;
use super::{ProjectionStore, StorageRoots, StoredSession};

fn store() -> (tempfile::TempDir, ProjectionStore) {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = ProjectionStore::initialize(temp.path().join("projection.sqlite"))
        .expect("projection store");
    (temp, store)
}

fn store_with_rollout() -> (tempfile::TempDir, PathBuf, ProjectionStore) {
    let temp = tempfile::tempdir().expect("tempdir");
    let agent_root = temp.path().join("agent-root");
    let store = ProjectionStore::initialize_with_agent_root(
        agent_root.join("runtime").join("projection.sqlite"),
        &agent_root,
    )
    .expect("projection store with rollout");
    (temp, agent_root, store)
}

fn user_tables(path: &Path) -> Vec<String> {
    let connection = Connection::open(path).expect("open SQLite database");
    let mut statement = connection
        .prepare(
            "SELECT name FROM sqlite_master
             WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
             ORDER BY name",
        )
        .expect("prepare table inventory");
    let tables = statement
        .query_map([], |row| row.get::<_, String>(0))
        .expect("query table inventory")
        .collect::<Result<Vec<_>, _>>()
        .expect("read table inventory");
    tables
}

#[test]
fn production_storage_paths_keep_physical_table_owners_separate() {
    let temp = tempfile::tempdir().expect("tempdir");
    let roots = StorageRoots::initialize(temp.path(), temp.path().join("agent-root"))
        .expect("storage roots");
    let store = ProjectionStore::initialize_with_storage_paths(
        &roots.projection_db_path,
        &roots.state_db_path,
        &roots.thread_history_db_path,
        &roots.data_root,
    )
    .expect("split projection store");
    drop(store);

    assert_eq!(
        user_tables(&roots.state_db_path),
        vec![
            "canonical_thread_spawn_edges".to_string(),
            "canonical_threads".to_string(),
            "thread_goal_continuation_deferrals".to_string(),
            "thread_goal_turn_accounting".to_string(),
            "thread_goal_update_outbox".to_string(),
            "thread_goals".to_string(),
            "thread_section_members".to_string(),
            "thread_sections".to_string(),
        ]
    );
    assert_eq!(
        user_tables(&roots.thread_history_db_path),
        vec![
            "canonical_history_applies".to_string(),
            "canonical_items".to_string(),
            "canonical_turns".to_string(),
        ]
    );
    assert_eq!(
        user_tables(&roots.projection_db_path),
        vec![
            "projected_items".to_string(),
            "projected_sessions".to_string(),
            "projected_turns".to_string(),
            "projection_watermarks".to_string(),
        ]
    );
}

#[test]
fn canonical_store_connections_wait_for_transient_sqlite_locks() {
    let (_temp, store) = store();
    let conn = store.open_thread_store().expect("open canonical store");
    let busy_timeout_ms = conn
        .query_row("PRAGMA busy_timeout", [], |row| row.get::<_, u64>(0))
        .expect("read SQLite busy timeout");

    assert_eq!(busy_timeout_ms, 5_000);
}

#[test]
fn queued_recovery_requires_a_current_canonical_thread_owner() {
    let temp = tempfile::tempdir().expect("tempdir");
    let roots = StorageRoots::initialize(temp.path(), temp.path().join("agent-root"))
        .expect("storage roots");
    let store = ProjectionStore::initialize_with_storage_paths(
        &roots.projection_db_path,
        &roots.state_db_path,
        &roots.thread_history_db_path,
        &roots.data_root,
    )
    .expect("split projection store");
    let projection = Connection::open(&roots.projection_db_path).expect("projection database");
    projection
        .execute(
            "INSERT INTO projected_sessions
             (session_id, thread_id, status, updated_at, last_event_sequence)
             VALUES ('stale-session', 'stale-thread', 'idle', '2026-07-20', 1)",
            [],
        )
        .expect("stale projected session");
    projection
        .execute(
            "INSERT INTO projected_turns
             (turn_id, session_id, thread_id, status, last_event_sequence)
             VALUES ('stale-turn', 'stale-session', 'stale-thread', 'queued', 1)",
            [],
        )
        .expect("stale projected turn");
    assert!(store
        .list_queued_session_ids()
        .expect("ignore stale projected queue")
        .is_empty());

    let current = thread("current-queued", 1_700_000_000_000);
    create(&store, &current);
    projection
        .execute(
            "INSERT INTO projected_sessions
             (session_id, thread_id, status, updated_at, last_event_sequence)
             VALUES (?1, ?2, 'idle', '2026-07-20', 1)",
            rusqlite::params![current.session_id.as_str(), current.thread_id.as_str()],
        )
        .expect("current projected session");
    projection
        .execute(
            "INSERT INTO projected_turns
             (turn_id, session_id, thread_id, status, last_event_sequence)
             VALUES (?1, ?2, ?3, 'queued', 1)",
            rusqlite::params![
                "current-queued-turn",
                current.session_id.as_str(),
                current.thread_id.as_str()
            ],
        )
        .expect("current projected turn");
    assert_eq!(
        store
            .list_queued_session_ids()
            .expect("list current projected queue"),
        vec![current.session_id.as_str().to_string()]
    );
}

#[test]
fn empty_projection_rebuilds_from_existing_split_state_and_history() {
    let temp = tempfile::tempdir().expect("tempdir");
    let roots = StorageRoots::initialize(temp.path(), temp.path().join("agent-root"))
        .expect("storage roots");
    let store = ProjectionStore::initialize_with_storage_paths(
        &roots.projection_db_path,
        &roots.state_db_path,
        &roots.thread_history_db_path,
        &roots.data_root,
    )
    .expect("split projection store");
    let source = thread("split-projection-rebuild", 1_700_000_000_000);
    create(&store, &source);
    let completed_turn = turn(&source, "turn-1", TurnStatus::Completed);
    let assistant = item(&source, "turn-1", "item-1", 1, 1);
    block_on(store.apply_history(ApplyThreadHistoryParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        changes: ThreadHistoryChangeSet {
            sequence: 1,
            changed_turns: vec![completed_turn],
            changed_items: vec![assistant],
            ..Default::default()
        },
    }))
    .expect("persist split canonical history");
    drop(store);

    let rebuilt_projection_path = roots.runtime_root.join("projection_rebuilt.sqlite");
    let rebuilt = ProjectionStore::initialize_with_storage_paths(
        &rebuilt_projection_path,
        &roots.state_db_path,
        &roots.thread_history_db_path,
        &roots.data_root,
    )
    .expect("rebuild empty projection from split canonical stores");
    let projection = rebuilt
        .read_session_projection(
            source.session_id.as_str(),
            super::projection_store::ProjectionReadWindow::tail(None),
        )
        .expect("read rebuilt projection")
        .expect("rebuilt projection exists");

    assert_eq!(projection.session.thread_id, source.thread_id.as_str());
    assert_eq!(projection.turns.len(), 1);
    assert_eq!(projection.messages.len(), 1);
    assert_eq!(user_tables(&rebuilt_projection_path).len(), 4);
}

fn thread(id: &str, updated_at_ms: i64) -> Thread {
    Thread {
        session_id: SessionId::new(format!("session-{id}")),
        thread_id: ThreadId::new(id),
        status: ThreadStatus::Idle,
        created_at_ms: updated_at_ms,
        updated_at_ms,
        archived: false,
        recency_at_ms: None,
        parent_thread_id: None,
        agent_path: None,
        agent_nickname: None,
        agent_role: None,
        last_task_message: None,
        agent_state: None,
        forked_from_id: None,
        preview: format!("preview-{id}"),
        model_provider: "test".to_string(),
        product: None,
        name: None,
        metadata: json!({}),
        turns: Vec::new(),
        turns_view: ThreadTurnsView::NotLoaded,
    }
}

fn turn(thread: &Thread, id: &str, status: TurnStatus) -> Turn {
    Turn {
        session_id: thread.session_id.clone(),
        thread_id: thread.thread_id.clone(),
        turn_id: TurnId::new(id),
        status,
        admission: TurnAdmissionState::Accepted,
        queue: TurnQueueState::Running,
        approval: TurnApprovalState::NotRequired,
        items: Vec::new(),
        items_view: TurnItemsView::NotLoaded,
        error: None,
        created_at_ms: 10,
        updated_at_ms: 10,
        started_at_ms: Some(10),
        completed_at_ms: status.is_terminal().then_some(11),
        duration_ms: status.is_terminal().then_some(1),
    }
}

fn item(thread: &Thread, turn_id: &str, id: &str, sequence: u64, ordinal: u64) -> ThreadItem {
    ThreadItem {
        session_id: thread.session_id.clone(),
        thread_id: thread.thread_id.clone(),
        turn_id: TurnId::new(turn_id),
        item_id: ItemId::new(id),
        sequence,
        ordinal,
        created_at_ms: sequence as i64,
        updated_at_ms: sequence as i64,
        completed_at_ms: Some(sequence as i64),
        kind: agent_protocol::ItemKind::AgentMessage,
        status: ItemStatus::Completed,
        payload: ThreadItemPayload::AgentMessage {
            text: id.to_string(),
            phase: None,
            content_parts: Vec::new(),
        },
        metadata: json!({}),
    }
}

fn user_item(
    thread: &Thread,
    turn_id: &str,
    id: &str,
    content: String,
    sequence: u64,
    ordinal: u64,
) -> ThreadItem {
    ThreadItem {
        session_id: thread.session_id.clone(),
        thread_id: thread.thread_id.clone(),
        turn_id: TurnId::new(turn_id),
        item_id: ItemId::new(id),
        sequence,
        ordinal,
        created_at_ms: sequence as i64,
        updated_at_ms: sequence as i64,
        completed_at_ms: Some(sequence as i64),
        kind: agent_protocol::ItemKind::UserMessage,
        status: ItemStatus::Completed,
        payload: ThreadItemPayload::UserMessage {
            content: vec![agent_protocol::AgentInput::text(content)],
            client_id: None,
        },
        metadata: json!({}),
    }
}

fn create(store: &ProjectionStore, thread: &Thread) {
    block_on(store.create_thread(CreateThreadParams {
        thread: thread.clone(),
    }))
    .expect("create thread");
}

#[test]
fn multimodal_user_message_survives_store_restart_exactly() {
    let (temp, store) = store();
    let path = temp.path().join("projection.sqlite");
    let source = thread("multimodal-restart", 1_700_000_000_000);
    create(&store, &source);
    let content = vec![
        AgentInput::Text {
            text: "inspect".to_string(),
            text_elements: vec![TextElement::new(0..7, Some("inspect".to_string()))],
        },
        AgentInput::Image {
            uri: "https://example.com/remote.png".to_string(),
            detail: Some(ImageDetail::High),
        },
        AgentInput::LocalImage {
            path: "/tmp/local.png".to_string(),
            detail: Some(ImageDetail::Original),
        },
        AgentInput::Skill {
            name: "review".to_string(),
            path: "/skills/review/SKILL.md".to_string(),
        },
        AgentInput::Mention {
            name: "docs".to_string(),
            path: "app://docs".to_string(),
        },
    ];
    let mut user = user_item(&source, "turn-1", "user-1", "unused".to_string(), 1, 1);
    user.payload = ThreadItemPayload::UserMessage {
        content: content.clone(),
        client_id: Some("client-1".to_string()),
    };
    block_on(store.apply_history(ApplyThreadHistoryParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        changes: ThreadHistoryChangeSet {
            sequence: 1,
            changed_turns: vec![turn(&source, "turn-1", TurnStatus::Completed)],
            changed_items: vec![user],
            ..Default::default()
        },
    }))
    .expect("persist multimodal canonical history");
    drop(store);

    let reopened = ProjectionStore::initialize(path).expect("reopen canonical store");
    let thread = block_on(reopened.read_thread(ReadThreadParams {
        thread_id: source.thread_id,
        include_archived: false,
        turns_view: ThreadTurnsView::Full,
    }))
    .expect("read canonical thread")
    .expect("canonical thread exists");
    assert_eq!(
        thread.turns[0].items[0].payload,
        ThreadItemPayload::UserMessage {
            content,
            client_id: Some("client-1".to_string()),
        }
    );
}

fn page(direction: SortDirection, limit: u32) -> PageRequest {
    PageRequest {
        cursor: None,
        limit,
        sort_direction: direction,
    }
}

fn rollout_path(store: &ProjectionStore, agent_root: &Path, thread_id: &ThreadId) -> PathBuf {
    let conn = Connection::open(store.path()).expect("open projection DB");
    let relative = conn
        .query_row(
            "SELECT rollout_path FROM canonical_threads WHERE thread_id = ?1",
            [thread_id.as_str()],
            |row| row.get::<_, String>(0),
        )
        .expect("stored rollout path");
    agent_root.join(relative)
}

fn rollout_lines(path: &Path) -> Vec<serde_json::Value> {
    fs::read_to_string(path)
        .expect("read rollout")
        .lines()
        .map(|line| serde_json::from_str(line).expect("valid rollout JSONL"))
        .collect()
}

#[test]
fn canonical_rollout_uses_creation_date_and_survives_cross_day_restart() {
    let (_temp, agent_root, store) = store_with_rollout();
    let created_at = chrono::DateTime::parse_from_rfc3339("2026-07-19T23:30:00Z")
        .expect("timestamp")
        .timestamp_millis();
    let source = thread("dated-thread", created_at);
    create(&store, &source);
    let first = ApplyThreadHistoryParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        changes: ThreadHistoryChangeSet {
            sequence: 1,
            changed_turns: vec![turn(&source, "turn-1", TurnStatus::InProgress)],
            ..Default::default()
        },
    };
    block_on(store.apply_history(first)).expect("first rollout append");

    let path = rollout_path(&store, &agent_root, &source.thread_id);
    let local = chrono::DateTime::<chrono::Utc>::from_timestamp_millis(created_at)
        .expect("created at")
        .with_timezone(&chrono::Local);
    let expected_dir = agent_root
        .join("sessions")
        .join(local.format("%Y").to_string())
        .join(local.format("%m").to_string())
        .join(local.format("%d").to_string());
    assert_eq!(path.parent(), Some(expected_dir.as_path()));
    assert!(path
        .file_name()
        .and_then(|value| value.to_str())
        .is_some_and(
            |value| value.starts_with("rollout-") && value.ends_with("-dated-thread.jsonl")
        ));
    let first_lines = rollout_lines(&path);
    assert_eq!(first_lines.len(), 2);
    assert_eq!(first_lines[0]["type"], "session_meta");
    assert_eq!(first_lines[0]["thread_id"], source.thread_id.as_str());
    assert_eq!(first_lines[1]["type"], "thread_history");
    assert_eq!(first_lines[1]["sequence"], 1);
    assert_eq!(
        first_lines[1]["changes"]["changedTurns"]
            .as_array()
            .map(Vec::len),
        Some(1)
    );

    let mut next_day_turn = turn(&source, "turn-1", TurnStatus::Completed);
    next_day_turn.updated_at_ms = created_at + 2 * 24 * 60 * 60 * 1_000;
    let second = ApplyThreadHistoryParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        changes: ThreadHistoryChangeSet {
            sequence: 2,
            changed_turns: vec![next_day_turn],
            ..Default::default()
        },
    };
    block_on(store.apply_history(second.clone())).expect("cross-day append");
    assert_eq!(rollout_path(&store, &agent_root, &source.thread_id), path);
    drop(store);

    let restarted = ProjectionStore::initialize_with_agent_root(
        agent_root.join("runtime").join("projection.sqlite"),
        &agent_root,
    )
    .expect("restart projection store");
    let retry = block_on(restarted.apply_history(second)).expect("idempotent restart retry");
    assert!(!retry.applied);
    assert_eq!(
        rollout_path(&restarted, &agent_root, &source.thread_id),
        path
    );
    assert_eq!(rollout_lines(&path).len(), 3);
}

#[test]
fn rollout_append_is_idempotent_when_projection_commit_is_retried() {
    let (_temp, agent_root, store) = store_with_rollout();
    let source = thread("commit-retry", 1_700_000_000_000);
    create(&store, &source);
    let conn = Connection::open(store.path()).expect("open projection DB");
    conn.execute_batch(
        "PRAGMA foreign_keys = ON;
         CREATE TABLE rollout_commit_guard_parent (id INTEGER PRIMARY KEY);
         CREATE TABLE rollout_commit_guard_child (
             id INTEGER PRIMARY KEY,
             parent_id INTEGER NOT NULL,
             FOREIGN KEY (parent_id) REFERENCES rollout_commit_guard_parent(id)
                 DEFERRABLE INITIALLY DEFERRED
         );
         CREATE TRIGGER fail_rollout_projection_commit
         AFTER INSERT ON canonical_history_applies
         BEGIN
             INSERT INTO rollout_commit_guard_child (id, parent_id) VALUES (1, 999);
         END;",
    )
    .expect("install deferred commit failure");
    drop(conn);

    let params = ApplyThreadHistoryParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        changes: ThreadHistoryChangeSet {
            sequence: 1,
            changed_turns: vec![turn(&source, "turn-1", TurnStatus::InProgress)],
            ..Default::default()
        },
    };
    let error = block_on(store.apply_history(params.clone()))
        .expect_err("deferred foreign key must fail DB commit");
    assert!(error.to_string().contains("FOREIGN KEY constraint failed"));
    let path = rollout_path(&store, &agent_root, &source.thread_id);
    assert_eq!(rollout_lines(&path).len(), 2);
    assert_eq!(
        block_on(store.history_sequence(source.thread_id.clone())).expect("history sequence"),
        None
    );

    let conn = Connection::open(store.path()).expect("reopen projection DB");
    conn.execute_batch("DROP TRIGGER fail_rollout_projection_commit;")
        .expect("remove commit failure");
    drop(conn);
    let mut conflict = params.clone();
    conflict.changes.changed_turns[0].status = TurnStatus::Completed;
    let error = block_on(store.apply_history(conflict)).expect_err("rollout collision");
    assert!(error
        .to_string()
        .contains("rollout history sequence collision"));
    assert_eq!(rollout_lines(&path).len(), 2);

    let applied = block_on(store.apply_history(params)).expect("retry projection commit");
    assert!(applied.applied);
    assert_eq!(rollout_lines(&path).len(), 2);
    assert_eq!(
        block_on(store.history_sequence(source.thread_id)).expect("history sequence"),
        Some(1)
    );
}

#[test]
fn rollout_metadata_retry_reconciles_projection_and_rejects_divergence() {
    let (_temp, agent_root, store) = store_with_rollout();
    let mut source = thread("metadata-commit-retry", 1_700_000_000_000);
    source.metadata = json!({"modelName": "model-a"});
    create(&store, &source);
    let path = rollout_path(&store, &agent_root, &source.thread_id);
    let conn = Connection::open(store.path()).expect("open projection DB");
    conn.execute_batch(
        "PRAGMA foreign_keys = ON;
         CREATE TABLE rollout_metadata_guard_parent (id INTEGER PRIMARY KEY);
         CREATE TABLE rollout_metadata_guard_child (
             id INTEGER PRIMARY KEY,
             parent_id INTEGER NOT NULL,
             FOREIGN KEY (parent_id) REFERENCES rollout_metadata_guard_parent(id)
                 DEFERRABLE INITIALLY DEFERRED
         );
         CREATE TRIGGER fail_rollout_metadata_commit
         AFTER UPDATE ON canonical_threads
         BEGIN
             INSERT INTO rollout_metadata_guard_child (id, parent_id) VALUES (1, 999);
         END;",
    )
    .expect("install deferred metadata commit failure");
    drop(conn);

    let session = |model: &str, updated_at: &str| AgentSession {
        session_id: source.session_id.as_str().to_string(),
        thread_id: source.thread_id.as_str().to_string(),
        app_id: "agent-chat".to_string(),
        workspace_id: None,
        business_object_ref: Some(BusinessObjectRef {
            kind: "agent.thread".to_string(),
            id: source.thread_id.as_str().to_string(),
            title: None,
            uri: None,
            metadata: Some(json!({"modelName": model})),
        }),
        status: AgentSessionStatus::Idle,
        created_at: "2023-11-14T22:13:20.000Z".to_string(),
        updated_at: updated_at.to_string(),
    };
    let mut first = session("model-b", "2023-11-14T22:13:21.000Z");
    let error = store
        .persist_session_metadata(&mut first)
        .expect_err("deferred foreign key must fail metadata commit");
    assert!(error.contains("FOREIGN KEY constraint failed"));
    assert_eq!(rollout_lines(&path).len(), 2);

    let mut divergent = session("model-c", "2023-11-14T22:13:22.000Z");
    let error = store
        .persist_session_metadata(&mut divergent)
        .expect_err("a different update must not overwrite uncommitted rollout metadata");
    assert!(error.contains("rollout metadata state conflict"));
    assert_eq!(rollout_lines(&path).len(), 2);

    let conn = Connection::open(store.path()).expect("reopen projection DB");
    conn.execute_batch("DROP TRIGGER fail_rollout_metadata_commit;")
        .expect("remove metadata commit failure");
    drop(conn);
    let mut retry = session("model-b", "2023-11-14T22:13:23.000Z");
    store
        .persist_session_metadata(&mut retry)
        .expect("retry metadata projection commit");
    assert_eq!(retry.updated_at, "2023-11-14T22:13:21.000Z");
    assert_eq!(rollout_lines(&path).len(), 2);

    let rebuilt = ProjectionStore::initialize_with_agent_root(
        agent_root.join("runtime").join("metadata-rebuilt.sqlite"),
        &agent_root,
    )
    .expect("rebuild metadata projection from rollout");
    let rebuilt_thread = block_on(rebuilt.read_thread(ReadThreadParams {
        thread_id: source.thread_id,
        include_archived: false,
        turns_view: ThreadTurnsView::NotLoaded,
    }))
    .expect("read rebuilt metadata thread")
    .expect("rebuilt metadata thread");
    assert_eq!(rebuilt_thread.metadata["modelName"], "model-b");
    assert_eq!(rebuilt_thread.updated_at_ms, 1_700_000_001_000);
}

#[test]
fn projection_rebuild_rejects_tampered_rollout_metadata() {
    let (_temp, agent_root, store) = store_with_rollout();
    let source = thread("tampered-metadata", 1_700_000_000_000);
    create(&store, &source);
    let mut session = AgentSession {
        session_id: source.session_id.as_str().to_string(),
        thread_id: source.thread_id.as_str().to_string(),
        app_id: "agent-chat".to_string(),
        workspace_id: None,
        business_object_ref: Some(BusinessObjectRef {
            kind: "agent.thread".to_string(),
            id: source.thread_id.as_str().to_string(),
            title: None,
            uri: None,
            metadata: Some(json!({"memoryMode": "disabled"})),
        }),
        status: AgentSessionStatus::Idle,
        created_at: "2023-11-14T22:13:20.000Z".to_string(),
        updated_at: "2023-11-14T22:13:21.000Z".to_string(),
    };
    store
        .persist_session_metadata(&mut session)
        .expect("persist metadata before tamper");
    let path = rollout_path(&store, &agent_root, &source.thread_id);
    drop(store);

    let mut lines = rollout_lines(&path);
    lines[1]["metadata"]["memoryMode"] = json!("enabled");
    let tampered = lines
        .into_iter()
        .map(|line| serde_json::to_string(&line).expect("encode tampered metadata line"))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    fs::write(&path, tampered).expect("write tampered metadata rollout");

    let error = ProjectionStore::initialize_with_agent_root(
        agent_root.join("runtime").join("metadata-tampered.sqlite"),
        &agent_root,
    )
    .expect_err("tampered metadata rollout must fail projection rebuild");
    assert!(error.contains("invalid rollout metadata record"));
}

#[test]
fn canonical_rollout_archive_moves_and_restores_the_original_date_path() {
    let (_temp, agent_root, store) = store_with_rollout();
    let source = thread("archive-thread", 1_700_000_000_000);
    create(&store, &source);
    block_on(store.apply_history(ApplyThreadHistoryParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        changes: ThreadHistoryChangeSet {
            sequence: 1,
            changed_turns: vec![turn(&source, "turn-1", TurnStatus::Completed)],
            ..Default::default()
        },
    }))
    .expect("seed rollout history");
    let active_path = rollout_path(&store, &agent_root, &source.thread_id);

    block_on(store.archive_thread(ArchiveThreadParams {
        thread_id: source.thread_id.clone(),
    }))
    .expect("archive rollout");
    let archived_path = rollout_path(&store, &agent_root, &source.thread_id);
    assert_eq!(
        archived_path.parent(),
        Some(agent_root.join("archived_sessions").as_path())
    );
    assert!(!active_path.exists());
    assert!(archived_path.exists());
    block_on(store.archive_thread(ArchiveThreadParams {
        thread_id: source.thread_id.clone(),
    }))
    .expect("idempotent archive");
    let error = block_on(store.apply_history(ApplyThreadHistoryParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        changes: ThreadHistoryChangeSet {
            sequence: 2,
            changed_turns: vec![turn(&source, "turn-2", TurnStatus::InProgress)],
            ..Default::default()
        },
    }))
    .expect_err("archived thread must reject append");
    assert!(error.to_string().contains("is archived"));
    drop(store);

    let restarted = ProjectionStore::initialize_with_agent_root(
        agent_root.join("runtime").join("projection.sqlite"),
        &agent_root,
    )
    .expect("restart archived store");
    let restored = block_on(restarted.unarchive_thread(ArchiveThreadParams {
        thread_id: source.thread_id.clone(),
    }))
    .expect("unarchive rollout");
    assert!(!restored.archived);
    assert!(!archived_path.exists());
    assert!(active_path.exists());
    assert_eq!(
        rollout_path(&restarted, &agent_root, &source.thread_id),
        active_path
    );
    block_on(restarted.unarchive_thread(ArchiveThreadParams {
        thread_id: source.thread_id,
    }))
    .expect("idempotent unarchive");
    assert_eq!(rollout_lines(&active_path).len(), 2);
}

#[test]
fn rollout_archive_and_unarchive_retry_after_projection_commit_failure() {
    let (_temp, agent_root, store) = store_with_rollout();
    let source = thread("archive-retry", 1_700_000_000_000);
    create(&store, &source);
    let active_path = rollout_path(&store, &agent_root, &source.thread_id);
    let archived_path = agent_root
        .join("archived_sessions")
        .join(active_path.file_name().expect("rollout filename"));
    let conn = Connection::open(store.path()).expect("open projection DB");
    conn.execute_batch(
        "PRAGMA foreign_keys = ON;
         CREATE TABLE rollout_archive_guard_parent (id INTEGER PRIMARY KEY);
         CREATE TABLE rollout_archive_guard_child (
             id INTEGER PRIMARY KEY,
             parent_id INTEGER NOT NULL,
             FOREIGN KEY (parent_id) REFERENCES rollout_archive_guard_parent(id)
                 DEFERRABLE INITIALLY DEFERRED
         );
         CREATE TRIGGER fail_rollout_archive_commit
         AFTER UPDATE OF archived ON canonical_threads
         WHEN NEW.archived = 1
         BEGIN
             INSERT INTO rollout_archive_guard_child (id, parent_id) VALUES (1, 999);
         END;",
    )
    .expect("install archive commit failure");
    drop(conn);

    let error = block_on(store.archive_thread(ArchiveThreadParams {
        thread_id: source.thread_id.clone(),
    }))
    .expect_err("archive projection commit must fail");
    assert!(error.to_string().contains("FOREIGN KEY constraint failed"));
    assert!(!active_path.exists());
    assert!(archived_path.exists());
    assert_eq!(
        rollout_path(&store, &agent_root, &source.thread_id),
        active_path
    );

    let conn = Connection::open(store.path()).expect("reopen projection DB");
    conn.execute_batch("DROP TRIGGER fail_rollout_archive_commit;")
        .expect("remove archive failure");
    drop(conn);
    block_on(store.archive_thread(ArchiveThreadParams {
        thread_id: source.thread_id.clone(),
    }))
    .expect("retry archive");
    assert_eq!(
        rollout_path(&store, &agent_root, &source.thread_id),
        archived_path
    );

    let conn = Connection::open(store.path()).expect("open projection DB");
    conn.execute_batch(
        "CREATE TRIGGER fail_rollout_unarchive_commit
         AFTER UPDATE OF archived ON canonical_threads
         WHEN NEW.archived = 0
         BEGIN
             INSERT INTO rollout_archive_guard_child (id, parent_id) VALUES (1, 999);
         END;",
    )
    .expect("install unarchive commit failure");
    drop(conn);
    let error = block_on(store.unarchive_thread(ArchiveThreadParams {
        thread_id: source.thread_id.clone(),
    }))
    .expect_err("unarchive projection commit must fail");
    assert!(error.to_string().contains("FOREIGN KEY constraint failed"));
    assert!(active_path.exists());
    assert!(!archived_path.exists());
    assert_eq!(
        rollout_path(&store, &agent_root, &source.thread_id),
        archived_path
    );

    let conn = Connection::open(store.path()).expect("reopen projection DB");
    conn.execute_batch("DROP TRIGGER fail_rollout_unarchive_commit;")
        .expect("remove unarchive failure");
    drop(conn);
    block_on(store.unarchive_thread(ArchiveThreadParams {
        thread_id: source.thread_id.clone(),
    }))
    .expect("retry unarchive");
    assert_eq!(
        rollout_path(&store, &agent_root, &source.thread_id),
        active_path
    );
    assert_eq!(rollout_lines(&active_path).len(), 1);
}

#[test]
fn empty_projection_rebuilds_active_and_archived_threads_from_rollouts() {
    let (_temp, agent_root, store) = store_with_rollout();
    let mut active = thread("rebuild-active", 1_700_000_000_000);
    active.name = Some("Rebuilt Active".to_string());
    active.metadata = json!({
        "modelName": "rebuild-model",
        "workspaceId": "rebuild-workspace",
        "workingDir": "/tmp/rebuild-active",
        "executionStrategy": "rebuild"
    });
    create(&store, &active);
    let long_user_message = "u".repeat(700);
    block_on(store.apply_history(ApplyThreadHistoryParams {
        session_id: active.session_id.clone(),
        thread_id: active.thread_id.clone(),
        changes: ThreadHistoryChangeSet {
            sequence: 1,
            changed_turns: vec![turn(&active, "turn-active", TurnStatus::InProgress)],
            changed_items: vec![
                user_item(
                    &active,
                    "turn-active",
                    "user-active",
                    long_user_message.clone(),
                    1,
                    1,
                ),
                item(&active, "turn-active", "item-active", 1, 2),
            ],
            ..Default::default()
        },
    }))
    .expect("write active rollout");
    let archived = thread("rebuild-archived", 1_700_000_001_000);
    create(&store, &archived);
    block_on(store.apply_history(ApplyThreadHistoryParams {
        session_id: archived.session_id.clone(),
        thread_id: archived.thread_id.clone(),
        changes: ThreadHistoryChangeSet {
            sequence: 1,
            changed_turns: vec![turn(&archived, "turn-archived", TurnStatus::Completed)],
            changed_items: vec![item(&archived, "turn-archived", "item-archived", 1, 1)],
            ..Default::default()
        },
    }))
    .expect("write archived rollout");
    block_on(store.archive_thread(ArchiveThreadParams {
        thread_id: archived.thread_id.clone(),
    }))
    .expect("archive rollout before rebuild");
    let read_full = |store: &ProjectionStore, thread_id: ThreadId| {
        block_on(store.read_thread(ReadThreadParams {
            thread_id,
            include_archived: true,
            turns_view: ThreadTurnsView::Full,
        }))
        .expect("read canonical thread")
        .expect("canonical thread")
    };
    let active_before = read_full(&store, active.thread_id.clone());
    let archived_before = read_full(&store, archived.thread_id.clone());
    drop(store);

    let rebuilt_path = agent_root.join("runtime").join("projection-rebuilt.sqlite");
    let rebuilt = ProjectionStore::initialize_with_agent_root(&rebuilt_path, &agent_root)
        .expect("rebuild empty projection from rollouts");
    assert_eq!(read_full(&rebuilt, active.thread_id.clone()), active_before);
    assert_eq!(
        read_full(&rebuilt, archived.thread_id.clone()),
        archived_before
    );
    assert!(!read_full(&rebuilt, active.thread_id.clone()).archived);
    assert!(read_full(&rebuilt, archived.thread_id.clone()).archived);
    assert_eq!(
        block_on(rebuilt.history_sequence(active.thread_id.clone())).expect("active sequence"),
        Some(1)
    );
    assert_eq!(
        block_on(rebuilt.history_sequence(archived.thread_id.clone())).expect("archived sequence"),
        Some(1)
    );
    let active_projection = rebuilt
        .read_session_projection(
            active.session_id.as_str(),
            super::projection_store::ProjectionReadWindow::tail(None),
        )
        .expect("read rebuilt active projection")
        .expect("rebuilt active projection");
    assert_eq!(
        active_projection.session.thread_id,
        active.thread_id.as_str()
    );
    assert_eq!(
        active_projection
            .session
            .business_object_ref
            .as_ref()
            .and_then(|reference| reference.title.as_deref()),
        Some("Rebuilt Active")
    );
    assert_eq!(active_projection.messages.len(), 2);
    assert_eq!(active_projection.messages[0]["role"], "user");
    assert_ne!(
        active_projection.messages[0]["content"][0]["text"],
        long_user_message
    );
    assert_eq!(active_projection.messages[1]["role"], "assistant");
    assert_eq!(
        active_projection.messages[1]["content"][0]["text"],
        "item-active"
    );
    let user_event_id = active_projection.messages[0]["metadata"]["source_event_id"]
        .as_str()
        .expect("rebuilt user source event id");
    let user_summary = rebuilt
        .read_item_summary_for_test(user_event_id)
        .expect("read rebuilt user summary")
        .expect("rebuilt user summary");
    let user_summary: serde_json::Value =
        serde_json::from_str(&user_summary).expect("parse rebuilt user summary");
    assert_eq!(
        serde_json::from_value::<Vec<AgentInput>>(user_summary["input"].clone())
            .expect("rebuilt typed user input"),
        vec![AgentInput::text(long_user_message.clone())]
    );
    assert_ne!(user_summary["text"], long_user_message);
    let active_overviews = rebuilt
        .list_session_overviews(&AgentSessionListParams {
            workspace_id: Some("rebuild-workspace".to_string()),
            include_archived: Some(false),
            ..Default::default()
        })
        .expect("list rebuilt active projection");
    assert_eq!(active_overviews.len(), 1);
    assert_eq!(active_overviews[0].session_id, active.session_id.as_str());
    let archived_overviews = rebuilt
        .list_session_overviews(&AgentSessionListParams {
            archived_only: Some(true),
            ..Default::default()
        })
        .expect("list rebuilt archived projection");
    assert_eq!(archived_overviews.len(), 1);
    assert_eq!(
        archived_overviews[0].session_id,
        archived.session_id.as_str()
    );
    assert!(archived_overviews[0].archived_at.is_some());
    drop(rebuilt);

    ProjectionStore::initialize_with_agent_root(&rebuilt_path, &agent_root)
        .expect("restart rebuilt projection without duplicate import");
}

#[test]
fn canonical_rebuild_does_not_overwrite_existing_projected_read_model() {
    let temp = tempfile::tempdir().expect("tempdir");
    let agent_root = temp.path().join("agent-root");
    let rollout_db = agent_root.join("runtime").join("rollout-source.sqlite");
    let rollout_source = ProjectionStore::initialize_with_agent_root(&rollout_db, &agent_root)
        .expect("rollout source store");
    let source = thread("projection-no-clobber", 1_700_000_000_000);
    create(&rollout_source, &source);
    drop(rollout_source);

    let rebuilt_path = agent_root
        .join("runtime")
        .join("projection-existing.sqlite");
    let legacy = ProjectionStore::initialize(&rebuilt_path).expect("legacy projection store");
    legacy
        .apply_event(&AgentEvent {
            event_id: "legacy-projection-event".to_string(),
            sequence: 1,
            session_id: "legacy-session".to_string(),
            thread_id: Some("legacy-thread".to_string()),
            turn_id: None,
            event_type: "session.created".to_string(),
            timestamp: "2026-07-19T00:00:00Z".to_string(),
            payload: json!({ "session": { "title": "Legacy Projection" } }),
        })
        .expect("seed legacy projection");
    drop(legacy);

    let rebuilt = ProjectionStore::initialize_with_agent_root(&rebuilt_path, &agent_root)
        .expect("rebuild canonical with existing projection");
    assert!(block_on(rebuilt.read_thread(ReadThreadParams {
        thread_id: source.thread_id.clone(),
        include_archived: false,
        turns_view: ThreadTurnsView::NotLoaded,
    }))
    .expect("read rebuilt canonical thread")
    .is_some());
    assert!(rebuilt
        .read_session_projection(
            "legacy-session",
            super::projection_store::ProjectionReadWindow::tail(None),
        )
        .expect("read retained legacy projection")
        .is_some());
    assert!(rebuilt
        .read_session_projection(
            source.session_id.as_str(),
            super::projection_store::ProjectionReadWindow::tail(None),
        )
        .expect("read skipped rollout projection")
        .is_none());
}

#[test]
fn projection_rebuild_rejects_tampered_rollout_content() {
    let (_temp, agent_root, store) = store_with_rollout();
    let source = thread("tampered-rollout", 1_700_000_000_000);
    create(&store, &source);
    block_on(store.apply_history(ApplyThreadHistoryParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        changes: ThreadHistoryChangeSet {
            sequence: 1,
            changed_turns: vec![turn(&source, "turn-1", TurnStatus::InProgress)],
            ..Default::default()
        },
    }))
    .expect("write rollout before tamper");
    let path = rollout_path(&store, &agent_root, &source.thread_id);
    drop(store);

    let mut lines = rollout_lines(&path);
    lines[1]["changes"]["changedTurns"][0]["updatedAtMs"] = json!(999_999);
    let tampered = lines
        .into_iter()
        .map(|line| serde_json::to_string(&line).expect("encode tampered line"))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    fs::write(&path, tampered).expect("write tampered temp rollout");

    let error = ProjectionStore::initialize_with_agent_root(
        agent_root
            .join("runtime")
            .join("projection-tampered.sqlite"),
        &agent_root,
    )
    .expect_err("tampered rollout must fail projection rebuild");
    assert!(error.contains("invalid rollout history record"));
}

#[test]
fn production_rollout_store_rejects_legacy_thread_without_rollout_path() {
    let temp = tempfile::tempdir().expect("tempdir");
    let agent_root = temp.path().join("agent-root");
    let database_path = agent_root.join("runtime").join("projection.sqlite");
    let legacy = ProjectionStore::initialize(&database_path).expect("legacy projection store");
    let source = thread("legacy-thread", 1_700_000_000_000);
    create(&legacy, &source);
    drop(legacy);

    let production = ProjectionStore::initialize_with_agent_root(&database_path, &agent_root)
        .expect("production projection store");
    let error = block_on(production.apply_history(ApplyThreadHistoryParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        changes: ThreadHistoryChangeSet {
            sequence: 1,
            changed_turns: vec![turn(&source, "turn-1", TurnStatus::InProgress)],
            ..Default::default()
        },
    }))
    .expect_err("legacy row must not continue without canonical rollout");
    assert!(error.to_string().contains("migration is required"));
    assert!(!agent_root.join("sessions").exists());
}

#[test]
fn rollout_thread_creation_rejects_unsafe_file_identity_atomically() {
    let (_temp, agent_root, store) = store_with_rollout();
    let source = thread("../outside", 1_700_000_000_000);
    let error = block_on(store.create_thread(CreateThreadParams {
        thread: source.clone(),
    }))
    .expect_err("unsafe thread id must fail");
    assert!(error
        .to_string()
        .contains("not safe for a rollout filename"));
    assert!(block_on(store.read_thread(ReadThreadParams {
        thread_id: source.thread_id,
        include_archived: true,
        turns_view: ThreadTurnsView::NotLoaded,
    }))
    .expect("read rejected thread")
    .is_none());
    assert!(!agent_root.join("sessions").exists());
}

#[test]
fn rollout_metadata_retry_requires_the_same_initial_thread() {
    let (_temp, agent_root, store) = store_with_rollout();
    let source = thread("create-retry", 1_700_000_000_000);
    let rollout_path = agent_root.join(
        RolloutStore::new(&agent_root)
            .path_for_thread(&source)
            .expect("rollout path"),
    );
    let conn = Connection::open(store.path()).expect("open projection DB");
    conn.execute_batch(
        "PRAGMA foreign_keys = ON;
         CREATE TABLE rollout_create_guard_parent (id INTEGER PRIMARY KEY);
         CREATE TABLE rollout_create_guard_child (
             id INTEGER PRIMARY KEY,
             parent_id INTEGER NOT NULL,
             FOREIGN KEY (parent_id) REFERENCES rollout_create_guard_parent(id)
                 DEFERRABLE INITIALLY DEFERRED
         );
         CREATE TRIGGER fail_rollout_thread_commit
         AFTER INSERT ON canonical_threads
         BEGIN
             INSERT INTO rollout_create_guard_child (id, parent_id) VALUES (1, 999);
         END;",
    )
    .expect("install deferred create failure");
    drop(conn);

    let error = block_on(store.create_thread(CreateThreadParams {
        thread: source.clone(),
    }))
    .expect_err("deferred foreign key must fail thread commit");
    assert!(error.to_string().contains("FOREIGN KEY constraint failed"));
    assert_eq!(rollout_lines(&rollout_path).len(), 1);

    let conn = Connection::open(store.path()).expect("reopen projection DB");
    conn.execute_batch("DROP TRIGGER fail_rollout_thread_commit;")
        .expect("remove create failure");
    drop(conn);
    let mut changed = source.clone();
    changed.preview = "different metadata".to_string();
    let error = block_on(store.create_thread(CreateThreadParams { thread: changed }))
        .expect_err("metadata divergence must fail closed");
    assert!(error
        .to_string()
        .contains("existing rollout metadata does not match"));
    create(&store, &source);
    assert_eq!(rollout_lines(&rollout_path).len(), 1);
}

#[test]
fn projection_store_is_the_canonical_thread_store_owner() {
    let (_temp, store) = store();
    let source = thread("thread-1", 1);
    create(&store, &source);

    let updated = block_on(store.update_thread_metadata(UpdateThreadMetadataParams {
        thread_id: source.thread_id.clone(),
        patch: ThreadMetadataPatch {
            name: Some(Some("renamed".to_string())),
            preview: Some("next".to_string()),
            advance_recency_at_ms: Some(9),
            metadata: Some(Some(json!({"source": "test"}))),
            ..Default::default()
        },
        include_archived: false,
    }))
    .expect("update metadata");
    assert_eq!(updated.name.as_deref(), Some("renamed"));
    assert_eq!(updated.preview, "next");
    assert_eq!(updated.recency_at_ms, Some(9));

    block_on(store.archive_thread(ArchiveThreadParams {
        thread_id: source.thread_id.clone(),
    }))
    .expect("archive");
    let archived = block_on(store.list_threads(ListThreadsParams {
        include_archived: true,
        section: None,
        sort_by_section_position: false,
        page: page(SortDirection::Asc, 10),
    }))
    .expect("list archived threads");
    assert!(archived.data[0].archived);
    assert!(block_on(store.read_thread(ReadThreadParams {
        thread_id: source.thread_id.clone(),
        include_archived: false,
        turns_view: ThreadTurnsView::NotLoaded,
    }))
    .expect("read hidden archive")
    .is_none());
    let restored = block_on(store.unarchive_thread(ArchiveThreadParams {
        thread_id: source.thread_id.clone(),
    }))
    .expect("unarchive");
    assert!(!restored.archived);
    assert_eq!(restored.name.as_deref(), Some("renamed"));
}

#[test]
fn projection_store_persists_explicit_fork_lineage_without_inventing_it() {
    let temp = tempfile::tempdir().expect("tempdir");
    let database_path = temp.path().join("projection.sqlite");
    let store = ProjectionStore::initialize(database_path.clone()).expect("projection store");
    let parent_thread_id = ThreadId::new("thread-parent");
    let forked = thread("thread-forked", 1);
    let plain = thread("thread-plain", 2);
    create(&store, &forked);
    create(&store, &plain);

    let updated_fork = block_on(store.update_thread_metadata(UpdateThreadMetadataParams {
        thread_id: forked.thread_id.clone(),
        patch: ThreadMetadataPatch {
            forked_from_id: Some(parent_thread_id.clone()),
            ..Default::default()
        },
        include_archived: false,
    }))
    .expect("persist fork lineage");
    assert_eq!(updated_fork.forked_from_id, Some(parent_thread_id.clone()));

    let updated_plain = block_on(store.update_thread_metadata(UpdateThreadMetadataParams {
        thread_id: plain.thread_id.clone(),
        patch: ThreadMetadataPatch {
            preview: Some("plain-updated".to_string()),
            ..Default::default()
        },
        include_archived: false,
    }))
    .expect("update plain thread");
    assert_eq!(updated_plain.forked_from_id, None);

    drop(store);
    let reopened = ProjectionStore::initialize(database_path).expect("reopen projection store");
    let persisted_fork = block_on(reopened.read_thread(ReadThreadParams {
        thread_id: forked.thread_id,
        include_archived: false,
        turns_view: ThreadTurnsView::NotLoaded,
    }))
    .expect("read persisted fork")
    .expect("forked thread");
    let persisted_plain = block_on(reopened.read_thread(ReadThreadParams {
        thread_id: plain.thread_id,
        include_archived: false,
        turns_view: ThreadTurnsView::NotLoaded,
    }))
    .expect("read persisted plain thread")
    .expect("plain thread");

    assert_eq!(persisted_fork.forked_from_id, Some(parent_thread_id));
    assert_eq!(persisted_plain.forked_from_id, None);
}

#[test]
fn history_apply_is_typed_atomic_idempotent_and_rollback_capable() {
    let (_temp, store) = store();
    let source = thread("thread-1", 1);
    create(&store, &source);
    let first_turn = turn(&source, "turn-1", TurnStatus::InProgress);
    let first_item = item(&source, "turn-1", "message-1", 1, 1);
    let first = ApplyThreadHistoryParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        changes: ThreadHistoryChangeSet {
            sequence: 1,
            changed_turns: vec![first_turn],
            changed_items: vec![first_item.clone()],
            ..Default::default()
        },
    };
    assert!(
        block_on(store.apply_history(first.clone()))
            .expect("first apply")
            .applied
    );
    assert!(
        !block_on(store.apply_history(first.clone()))
            .expect("exact retry")
            .applied
    );

    let mut collision = first;
    collision
        .changes
        .removed_item_ids
        .push(first_item.item_id.clone());
    assert!(block_on(store.apply_history(collision))
        .expect_err("collision")
        .to_string()
        .contains("sequence collision"));

    let second = ApplyThreadHistoryParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        changes: ThreadHistoryChangeSet {
            sequence: 2,
            changed_turns: vec![turn(&source, "turn-2", TurnStatus::Completed)],
            changed_items: vec![item(&source, "turn-2", "message-2", 2, 2)],
            ..Default::default()
        },
    };
    block_on(store.apply_history(second)).expect("second apply");

    let rollback = ApplyThreadHistoryParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        changes: ThreadHistoryChangeSet {
            sequence: 3,
            removed_turn_ids: vec![TurnId::new("turn-2")],
            rollback_to_sequence: Some(1),
            ..Default::default()
        },
    };
    block_on(store.apply_history(rollback)).expect("rollback");
    assert_eq!(
        block_on(store.history_sequence(source.thread_id.clone())).expect("sequence"),
        Some(3)
    );
    let read = block_on(store.read_thread(ReadThreadParams {
        thread_id: source.thread_id.clone(),
        include_archived: false,
        turns_view: ThreadTurnsView::Full,
    }))
    .expect("read")
    .expect("thread");
    assert_eq!(read.turns.len(), 1);
    assert_eq!(read.turns[0].items, vec![first_item]);
}

#[test]
fn append_items_is_idempotent_and_rejects_sequence_collisions() {
    let (_temp, store) = store();
    let source = thread("append-idempotent", 1);
    create(&store, &source);
    block_on(store.apply_history(ApplyThreadHistoryParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        changes: ThreadHistoryChangeSet {
            sequence: 1,
            changed_turns: vec![turn(&source, "turn-1", TurnStatus::InProgress)],
            ..Default::default()
        },
    }))
    .expect("seed turn");

    let params = AppendThreadItemsParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        sequence: 2,
        items: vec![item(&source, "turn-1", "append-1", 2, 7)],
    };
    assert!(
        block_on(store.append_items(params.clone()))
            .expect("first append")
            .applied
    );
    assert_eq!(
        block_on(store.append_items(params))
            .expect("exact append retry")
            .applied,
        false
    );

    let mut collision = item(&source, "turn-1", "append-2", 2, 8);
    collision.status = ItemStatus::InProgress;
    let error = block_on(store.append_items(AppendThreadItemsParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        sequence: 2,
        items: vec![collision],
    }))
    .expect_err("different payload at an applied sequence must collide");
    assert!(error.to_string().contains("history sequence collision"));
    assert_eq!(
        block_on(store.history_sequence(source.thread_id.clone())).expect("sequence"),
        Some(2)
    );
    assert_eq!(
        block_on(store.list_items(ListItemsParams {
            thread_id: source.thread_id,
            turn_id: None,
            include_archived: false,
            page: page(SortDirection::Asc, 10),
        }))
        .expect("list appended items")
        .data
        .len(),
        1
    );
}

#[test]
fn append_items_does_not_refresh_thread_metadata_or_turn_snapshot() {
    let (_temp, store) = store();
    let mut source = thread("append-metadata", 100);
    source.name = Some("explicit-name".to_string());
    source.preview = "explicit-preview".to_string();
    source.model_provider = "explicit-provider".to_string();
    source.product = Some("explicit-product".to_string());
    source.metadata = json!({"owner": "explicit"});
    create(&store, &source);
    block_on(store.apply_history(ApplyThreadHistoryParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        changes: ThreadHistoryChangeSet {
            sequence: 1,
            changed_turns: vec![turn(&source, "turn-1", TurnStatus::InProgress)],
            ..Default::default()
        },
    }))
    .expect("seed turn");

    let before = block_on(store.read_thread(ReadThreadParams {
        thread_id: source.thread_id.clone(),
        include_archived: false,
        turns_view: ThreadTurnsView::Summary,
    }))
    .expect("read before append")
    .expect("thread before append");
    let before_turn = before.turns.first().expect("seeded turn");

    block_on(store.append_items(AppendThreadItemsParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        sequence: 2,
        items: vec![item(&source, "turn-1", "append-item", 2, 9)],
    }))
    .expect("append canonical item");

    let after = block_on(store.read_thread(ReadThreadParams {
        thread_id: source.thread_id,
        include_archived: false,
        turns_view: ThreadTurnsView::Full,
    }))
    .expect("read after append")
    .expect("thread after append");
    assert_eq!(after.name, before.name);
    assert_eq!(after.preview, before.preview);
    assert_eq!(after.model_provider, before.model_provider);
    assert_eq!(after.product, before.product);
    assert_eq!(after.metadata, before.metadata);
    assert_eq!(after.created_at_ms, before.created_at_ms);
    assert_eq!(after.updated_at_ms, before.updated_at_ms);
    assert_eq!(after.recency_at_ms, before.recency_at_ms);
    assert_eq!(after.status, before.status);
    let after_turn = after.turns.first().expect("turn after append");
    assert_eq!(after_turn.turn_id, before_turn.turn_id);
    assert_eq!(after_turn.status, before_turn.status);
    assert_eq!(after_turn.updated_at_ms, before_turn.updated_at_ms);
    assert_eq!(after_turn.completed_at_ms, before_turn.completed_at_ms);
    assert_eq!(after_turn.items.len(), 1);
}

#[test]
fn append_items_rejects_identity_and_foreign_key_errors_atomically() {
    let (_temp, store) = store();
    let source = thread("append-validation", 1);
    create(&store, &source);
    block_on(store.apply_history(ApplyThreadHistoryParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        changes: ThreadHistoryChangeSet {
            sequence: 1,
            changed_turns: vec![turn(&source, "turn-1", TurnStatus::InProgress)],
            ..Default::default()
        },
    }))
    .expect("seed turn");

    let mut wrong_thread = item(&source, "turn-1", "wrong-thread", 2, 1);
    wrong_thread.thread_id = ThreadId::new("different-thread");
    let error = block_on(store.append_items(AppendThreadItemsParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        sequence: 2,
        items: vec![wrong_thread],
    }))
    .expect_err("item identity mismatch must fail");
    assert!(error.to_string().contains("item identity"));
    assert_eq!(
        block_on(store.history_sequence(source.thread_id.clone()))
            .expect("sequence after identity error"),
        Some(1)
    );

    let missing_turn = item(&source, "missing-turn", "orphan", 2, 2);
    let error = block_on(store.append_items(AppendThreadItemsParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        sequence: 2,
        items: vec![missing_turn],
    }))
    .expect_err("missing turn must fail atomically");
    assert!(error.to_string().contains("missing turn"));
    assert_eq!(
        block_on(store.history_sequence(source.thread_id.clone()))
            .expect("sequence after foreign key error"),
        Some(1)
    );
    assert!(block_on(store.list_items(ListItemsParams {
        thread_id: source.thread_id,
        turn_id: None,
        include_archived: false,
        page: page(SortDirection::Asc, 10),
    }))
    .expect("items after failed append")
    .data
    .is_empty());
}

#[test]
fn append_items_rejects_item_sequence_after_batch_sequence() {
    let (_temp, store) = store();
    let source = thread("append-sequence-validation", 1);
    create(&store, &source);
    block_on(store.apply_history(ApplyThreadHistoryParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        changes: ThreadHistoryChangeSet {
            sequence: 1,
            changed_turns: vec![turn(&source, "turn-1", TurnStatus::InProgress)],
            ..Default::default()
        },
    }))
    .expect("seed turn");

    let error = block_on(store.append_items(AppendThreadItemsParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        sequence: 2,
        items: vec![item(&source, "turn-1", "future-item", 3, 1)],
    }))
    .expect_err("future item sequence must fail closed");
    assert!(error.to_string().contains("exceeds history sequence"));
    assert_eq!(
        block_on(store.history_sequence(source.thread_id.clone()))
            .expect("sequence after invalid append"),
        Some(1)
    );
    assert!(block_on(store.list_items(ListItemsParams {
        thread_id: source.thread_id,
        turn_id: None,
        include_archived: false,
        page: page(SortDirection::Asc, 10),
    }))
    .expect("items after invalid append")
    .data
    .is_empty());
}

#[test]
fn history_apply_preserves_thread_metadata_until_an_explicit_patch() {
    let temp = tempfile::tempdir().expect("tempdir");
    let database_path = temp.path().join("metadata.sqlite");
    let store = ProjectionStore::initialize(database_path.clone()).expect("projection store");
    let mut source = thread("metadata-history", 100);
    source.name = Some("explicit-name".to_string());
    source.preview = "explicit-preview".to_string();
    source.model_provider = "explicit-provider".to_string();
    source.metadata = json!({"owner": "metadata"});
    create(&store, &source);

    let mut first_turn = turn(&source, "turn-1", TurnStatus::Completed);
    first_turn.updated_at_ms = 1_000;
    block_on(store.apply_history(ApplyThreadHistoryParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        changes: ThreadHistoryChangeSet {
            sequence: 1,
            changed_turns: vec![first_turn],
            ..Default::default()
        },
    }))
    .expect("first history apply");

    let mut second_turn = turn(&source, "turn-2", TurnStatus::Completed);
    second_turn.updated_at_ms = 2_000;
    block_on(store.apply_history(ApplyThreadHistoryParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        changes: ThreadHistoryChangeSet {
            sequence: 2,
            changed_turns: vec![second_turn],
            ..Default::default()
        },
    }))
    .expect("second history apply");

    let after_history = block_on(store.read_thread(ReadThreadParams {
        thread_id: source.thread_id.clone(),
        include_archived: false,
        turns_view: ThreadTurnsView::NotLoaded,
    }))
    .expect("read after history")
    .expect("thread after history");
    assert_eq!(after_history.updated_at_ms, 100);
    assert_eq!(after_history.name.as_deref(), Some("explicit-name"));
    assert_eq!(after_history.preview, "explicit-preview");
    assert_eq!(after_history.model_provider, "explicit-provider");
    assert_eq!(after_history.metadata, json!({"owner": "metadata"}));

    block_on(store.apply_history(ApplyThreadHistoryParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        changes: ThreadHistoryChangeSet {
            sequence: 3,
            rollback_to_sequence: Some(1),
            ..Default::default()
        },
    }))
    .expect("rollback history");
    drop(store);

    let reopened = ProjectionStore::initialize(database_path).expect("reopen projection store");
    let after_reopen = block_on(reopened.read_thread(ReadThreadParams {
        thread_id: source.thread_id.clone(),
        include_archived: false,
        turns_view: ThreadTurnsView::Full,
    }))
    .expect("read after reopen")
    .expect("thread after reopen");
    assert_eq!(after_reopen.updated_at_ms, 100);
    assert_eq!(after_reopen.name.as_deref(), Some("explicit-name"));
    assert_eq!(after_reopen.preview, "explicit-preview");
    assert_eq!(after_reopen.model_provider, "explicit-provider");
    assert_eq!(after_reopen.metadata, json!({"owner": "metadata"}));
    assert_eq!(after_reopen.turns.len(), 1);

    let patched = block_on(reopened.update_thread_metadata(UpdateThreadMetadataParams {
        thread_id: source.thread_id,
        patch: ThreadMetadataPatch {
            name: Some(Some("patched-name".to_string())),
            updated_at_ms: Some(9_000),
            metadata: Some(Some(json!({"owner": "patch"}))),
            ..Default::default()
        },
        include_archived: false,
    }))
    .expect("explicit metadata patch");
    assert_eq!(patched.updated_at_ms, 9_000);
    assert_eq!(patched.name.as_deref(), Some("patched-name"));
    assert_eq!(patched.metadata, json!({"owner": "patch"}));
}

#[test]
fn failed_history_apply_rolls_back_without_advancing_sequence() {
    let (_temp, store) = store();
    let source = thread("thread-1", 1);
    create(&store, &source);
    let orphan = item(&source, "missing-turn", "orphan", 1, 1);

    let error = block_on(store.apply_history(ApplyThreadHistoryParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        changes: ThreadHistoryChangeSet {
            sequence: 1,
            changed_items: vec![orphan],
            ..Default::default()
        },
    }))
    .expect_err("missing turn must fail atomically");
    assert!(error.to_string().contains("missing turn"));
    assert_eq!(
        block_on(store.history_sequence(source.thread_id.clone())).expect("history sequence"),
        None
    );
    assert!(block_on(store.list_items(ListItemsParams {
        thread_id: source.thread_id,
        turn_id: None,
        include_archived: false,
        page: page(SortDirection::Asc, 10),
    }))
    .expect("items after rollback")
    .data
    .is_empty());
}

#[test]
fn opaque_cursors_page_threads_turns_and_items_stably() {
    let (_temp, store) = store();
    for index in 1..=3 {
        create(&store, &thread(&format!("thread-{index}"), index));
    }
    let first = block_on(store.list_threads(ListThreadsParams {
        include_archived: false,
        section: None,
        sort_by_section_position: false,
        page: page(SortDirection::Asc, 2),
    }))
    .expect("first thread page");
    assert_eq!(first.data.len(), 2);
    assert!(first.next_cursor.is_some());
    let second = block_on(store.list_threads(ListThreadsParams {
        include_archived: false,
        section: None,
        sort_by_section_position: false,
        page: PageRequest {
            cursor: first.next_cursor,
            limit: 2,
            sort_direction: SortDirection::Asc,
        },
    }))
    .expect("second thread page");
    assert_eq!(second.data.len(), 1);
    assert!(second.backwards_cursor.is_some());

    let source = thread("history", 10);
    create(&store, &source);
    block_on(store.apply_history(ApplyThreadHistoryParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        changes: ThreadHistoryChangeSet {
            sequence: 2,
            changed_turns: vec![
                turn(&source, "turn-1", TurnStatus::Completed),
                turn(&source, "turn-2", TurnStatus::Completed),
            ],
            changed_items: vec![
                item(&source, "turn-1", "item-1", 1, 1),
                item(&source, "turn-2", "item-2", 2, 2),
            ],
            ..Default::default()
        },
    }))
    .expect("apply history");
    let turns = block_on(store.list_turns(ListTurnsParams {
        thread_id: source.thread_id.clone(),
        include_archived: false,
        page: page(SortDirection::Asc, 1),
        items_view: TurnItemsView::NotLoaded,
    }))
    .expect("turn page");
    assert_eq!(turns.data[0].turn_id.as_str(), "turn-1");
    assert!(turns.next_cursor.is_some());
    let items = block_on(store.list_items(ListItemsParams {
        thread_id: source.thread_id,
        turn_id: None,
        include_archived: false,
        page: page(SortDirection::Desc, 1),
    }))
    .expect("item page");
    assert_eq!(items.data[0].item_id.as_str(), "item_item-2");
    assert!(items.next_cursor.is_some());
}

#[test]
fn production_event_batches_create_and_incrementally_update_canonical_history() {
    let (_temp, store) = store();
    let stored = StoredSession {
        session: app_server_protocol::AgentSession {
            session_id: "session-production".to_string(),
            thread_id: "thread-production".to_string(),
            app_id: "agent-chat".to_string(),
            workspace_id: None,
            business_object_ref: Some(app_server_protocol::BusinessObjectRef {
                kind: "agent.session".to_string(),
                id: "session-production".to_string(),
                title: Some("Production thread".to_string()),
                uri: None,
                metadata: Some(json!({"providerName": "openai"})),
            }),
            status: app_server_protocol::AgentSessionStatus::Running,
            created_at: "2026-07-12T00:00:00Z".to_string(),
            updated_at: "2026-07-12T00:00:01Z".to_string(),
        },
        turns: Vec::new(),
        turn_inputs: HashMap::new(),
        turn_runtime_options: HashMap::new(),
        events: Vec::new(),
        output_blobs: HashMap::new(),
    };
    let event = |sequence, text: &str| app_server_protocol::AgentEvent {
        event_id: format!("event-{sequence}"),
        sequence,
        session_id: stored.session.session_id.clone(),
        thread_id: Some(stored.session.thread_id.clone()),
        turn_id: Some("turn-production".to_string()),
        event_type: "message.delta".to_string(),
        timestamp: format!("2026-07-12T00:00:0{sequence}Z"),
        payload: json!({
            "id": "message-production",
            "text": text,
        }),
    };

    store
        .apply_canonical_events(&stored, &[event(1, "hello")])
        .expect("first canonical batch");
    store
        .apply_canonical_events(&stored, &[event(2, " world")])
        .expect("second canonical batch");

    let thread = block_on(store.read_thread(ReadThreadParams {
        thread_id: ThreadId::new("thread-production"),
        include_archived: false,
        turns_view: ThreadTurnsView::Full,
    }))
    .expect("read canonical production thread")
    .expect("canonical production thread");
    assert_eq!(thread.name.as_deref(), Some("Production thread"));
    assert_eq!(thread.model_provider, "openai");
    assert_eq!(thread.turns.len(), 1);
    assert_eq!(thread.turns[0].items.len(), 1);
    assert!(matches!(
        &thread.turns[0].items[0].payload,
        ThreadItemPayload::AgentMessage { text, .. } if text == "hello world"
    ));
}

#[test]
fn reasoning_deltas_persist_linearly_in_canonical_history_and_rollout() {
    let (_temp, agent_root, store) = store_with_rollout();
    let stored = StoredSession {
        session: AgentSession {
            session_id: "session-reasoning-linear".to_string(),
            thread_id: "thread-reasoning-linear".to_string(),
            app_id: "agent-chat".to_string(),
            workspace_id: None,
            business_object_ref: None,
            status: AgentSessionStatus::Running,
            created_at: "2026-08-05T00:00:00Z".to_string(),
            updated_at: "2026-08-05T00:00:01Z".to_string(),
        },
        turns: Vec::new(),
        turn_inputs: HashMap::new(),
        turn_runtime_options: HashMap::new(),
        events: Vec::new(),
        output_blobs: HashMap::new(),
    };
    let fragments = [
        "The", " user", " said", " hello", ".", " Respond", " briefly", " now",
    ];

    for (index, fragment) in fragments.iter().enumerate() {
        let sequence = (index + 1) as u64;
        store
            .apply_canonical_events(
                &stored,
                &[AgentEvent {
                    event_id: format!("reasoning-event-{sequence}"),
                    sequence,
                    session_id: stored.session.session_id.clone(),
                    thread_id: Some(stored.session.thread_id.clone()),
                    turn_id: Some("turn-reasoning-linear".to_string()),
                    event_type: "reasoning.delta".to_string(),
                    timestamp: format!("2026-08-05T00:00:{sequence:02}Z"),
                    payload: json!({
                        "reasoningId": "reasoning-linear",
                        "delta": fragment,
                    }),
                }],
            )
            .expect("persist reasoning delta");
    }

    let thread = block_on(store.read_thread(ReadThreadParams {
        thread_id: ThreadId::new(&stored.session.thread_id),
        include_archived: false,
        turns_view: ThreadTurnsView::Full,
    }))
    .expect("read canonical reasoning thread")
    .expect("canonical reasoning thread");
    let reasoning_content = thread.turns[0]
        .items
        .iter()
        .find_map(|item| match &item.payload {
            ThreadItemPayload::Reasoning { content, .. } => Some(content.clone()),
            _ => None,
        })
        .expect("canonical reasoning item");
    assert_eq!(reasoning_content, fragments.map(str::to_string).to_vec());

    let path = rollout_path(
        &store,
        &agent_root,
        &ThreadId::new(&stored.session.thread_id),
    );
    let lines = rollout_lines(&path);
    assert_eq!(lines.len(), fragments.len() + 1);
    for (index, line) in lines.iter().skip(1).enumerate() {
        let changes: ThreadHistoryChangeSet = serde_json::from_value(line["changes"].clone())
            .expect("decode rollout history changes");
        let content = changes
            .changed_items
            .iter()
            .find_map(|item| match &item.payload {
                ThreadItemPayload::Reasoning { content, .. } => Some(content),
                _ => None,
            })
            .expect("rollout reasoning item");
        assert_eq!(content.len(), index + 1);
        assert_eq!(content.last().map(String::as_str), Some(fragments[index]));
    }
}

#[test]
fn empty_reasoning_completion_preserves_canonical_content_across_persistence() {
    let (_temp, store) = store();
    let stored = StoredSession {
        session: AgentSession {
            session_id: "session-reasoning-empty-completion".to_string(),
            thread_id: "thread-reasoning-empty-completion".to_string(),
            app_id: "agent-chat".to_string(),
            workspace_id: None,
            business_object_ref: None,
            status: AgentSessionStatus::Running,
            created_at: "2026-08-05T00:00:00Z".to_string(),
            updated_at: "2026-08-05T00:00:01Z".to_string(),
        },
        turns: Vec::new(),
        turn_inputs: HashMap::new(),
        turn_runtime_options: HashMap::new(),
        events: Vec::new(),
        output_blobs: HashMap::new(),
    };
    let reasoning_item_id = "item_reasoning-turn-reasoning-empty-completion";
    let event = |sequence, event_type: &str, payload| AgentEvent {
        event_id: format!("reasoning-empty-event-{sequence}"),
        sequence,
        session_id: stored.session.session_id.clone(),
        thread_id: Some(stored.session.thread_id.clone()),
        turn_id: Some("turn-reasoning-empty-completion".to_string()),
        event_type: event_type.to_string(),
        timestamp: format!("2026-08-05T00:00:{sequence:02}Z"),
        payload,
    };

    store
        .apply_canonical_events(
            &stored,
            &[event(
                1,
                "reasoning.delta",
                json!({
                    "reasoningId": "reasoning-empty-completion",
                    "text": "inspect inputs"
                }),
            )],
        )
        .expect("persist reasoning content");
    store
        .apply_canonical_events(
            &stored,
            &[event(
                2,
                "item.completed",
                json!({
                    "item": {
                        "sessionId": stored.session.session_id.clone(),
                        "threadId": stored.session.thread_id.clone(),
                        "turnId": "turn-reasoning-empty-completion",
                        "itemId": reasoning_item_id,
                        "sequence": 2,
                        "ordinal": 2,
                        "createdAtMs": 2,
                        "updatedAtMs": 2,
                        "completedAtMs": 2,
                        "kind": "reasoning",
                        "status": "completed",
                        "payload": {
                            "type": "reasoning",
                            "summary": [],
                            "content": []
                        },
                        "metadata": {}
                    }
                }),
            )],
        )
        .expect("persist empty reasoning completion");

    let thread = block_on(store.read_thread(ReadThreadParams {
        thread_id: ThreadId::new(&stored.session.thread_id),
        include_archived: false,
        turns_view: ThreadTurnsView::Full,
    }))
    .expect("read persisted reasoning thread")
    .expect("persisted reasoning thread");
    let reasoning = thread.turns[0]
        .items
        .iter()
        .find(|item| matches!(item.payload, ThreadItemPayload::Reasoning { .. }))
        .expect("persisted reasoning item");
    assert_eq!(reasoning.status, ItemStatus::Completed);
    assert_eq!(
        reasoning.payload,
        ThreadItemPayload::Reasoning {
            summary: Vec::new(),
            content: vec!["inspect inputs".to_string()],
        }
    );
}

#[test]
fn canonical_live_cold_and_replay_projections_are_equivalent() {
    let stored = StoredSession {
        session: app_server_protocol::AgentSession {
            session_id: "session-equivalent".to_string(),
            thread_id: "thread-equivalent".to_string(),
            app_id: "agent-chat".to_string(),
            workspace_id: None,
            business_object_ref: None,
            status: app_server_protocol::AgentSessionStatus::Running,
            created_at: "2026-07-12T00:00:00Z".to_string(),
            updated_at: "2026-07-12T00:00:02Z".to_string(),
        },
        turns: Vec::new(),
        turn_inputs: HashMap::new(),
        turn_runtime_options: HashMap::new(),
        events: Vec::new(),
        output_blobs: HashMap::new(),
    };
    let event = |sequence: u64, text: &str| app_server_protocol::AgentEvent {
        event_id: format!("equivalent-event-{sequence}"),
        sequence,
        session_id: stored.session.session_id.clone(),
        thread_id: Some(stored.session.thread_id.clone()),
        turn_id: Some("turn-equivalent".to_string()),
        event_type: "message.delta".to_string(),
        timestamp: format!("2026-07-12T00:00:0{sequence}Z"),
        payload: json!({"id": "message-equivalent", "text": text}),
    };
    let events = vec![event(1, "hello"), event(2, " world")];

    let (live_temp, live) = store();
    live.apply_canonical_events(&stored, &events[..1])
        .expect("live first batch");
    live.apply_canonical_events(&stored, &events[1..])
        .expect("live second batch");
    let live_path = live_temp.path().join("projection.sqlite");
    drop(live);
    let cold = ProjectionStore::initialize(live_path).expect("cold reopen");

    let (_replay_temp, replay) = store();
    replay
        .repair_canonical_history(&stored, &events)
        .expect("full replay");
    let read = |store: &ProjectionStore| {
        block_on(store.read_thread(ReadThreadParams {
            thread_id: ThreadId::new("thread-equivalent"),
            include_archived: false,
            turns_view: ThreadTurnsView::Full,
        }))
        .expect("read canonical projection")
        .expect("canonical thread")
    };
    let cold_thread = read(&cold);
    let replay_thread = read(&replay);
    assert_eq!(cold_thread.turns, replay_thread.turns);
    assert_eq!(cold_thread.status, replay_thread.status);
    assert_eq!(cold_thread.preview, replay_thread.preview);
}

#[test]
fn canonical_builder_rejects_item_turn_identity_conflict_atomically() {
    let (_temp, store) = store();
    let source = thread("builder-conflict", 1);
    create(&store, &source);
    let first_turn = turn(&source, "turn-1", TurnStatus::InProgress);
    let first_item = item(&source, "turn-1", "message-1", 1, 1);
    block_on(store.apply_history(ApplyThreadHistoryParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        changes: ThreadHistoryChangeSet {
            sequence: 1,
            changed_turns: vec![first_turn],
            changed_items: vec![first_item],
            ..Default::default()
        },
    }))
    .expect("seed canonical history");

    let moved_item = item(&source, "turn-2", "message-1", 2, 2);
    let error = block_on(store.apply_history(ApplyThreadHistoryParams {
        session_id: source.session_id.clone(),
        thread_id: source.thread_id.clone(),
        changes: ThreadHistoryChangeSet {
            sequence: 2,
            changed_turns: vec![turn(&source, "turn-2", TurnStatus::InProgress)],
            changed_items: vec![moved_item],
            ..Default::default()
        },
    }))
    .expect_err("item must not move between turns");
    assert!(error.to_string().contains("changed turn identity"));
    assert_eq!(
        block_on(store.history_sequence(source.thread_id.clone())).expect("history sequence"),
        Some(1)
    );
    let thread = block_on(store.read_thread(ReadThreadParams {
        thread_id: source.thread_id,
        include_archived: false,
        turns_view: ThreadTurnsView::Full,
    }))
    .expect("read after conflict")
    .expect("thread after conflict");
    assert_eq!(thread.turns.len(), 1);
    assert_eq!(thread.turns[0].items.len(), 1);
}

#[test]
fn imported_and_live_items_share_event_log_ordinal_domain() {
    let temp = tempfile::tempdir().expect("tempdir");
    let database_path = temp.path().join("projection.sqlite");
    let store = ProjectionStore::initialize(database_path.clone()).expect("projection store");
    let stored = StoredSession {
        session: app_server_protocol::AgentSession {
            session_id: "session-live-ordinal".to_string(),
            thread_id: "thread-live-ordinal".to_string(),
            app_id: "agent-chat".to_string(),
            workspace_id: None,
            business_object_ref: None,
            status: app_server_protocol::AgentSessionStatus::Running,
            created_at: "2026-07-15T00:00:00Z".to_string(),
            updated_at: "2026-07-15T00:00:03Z".to_string(),
        },
        turns: Vec::new(),
        turn_inputs: HashMap::new(),
        turn_runtime_options: HashMap::new(),
        events: Vec::new(),
        output_blobs: HashMap::new(),
    };
    let event = |sequence: u64, event_type: &str, payload| app_server_protocol::AgentEvent {
        event_id: format!("live-ordinal-event-{sequence}"),
        sequence,
        session_id: stored.session.session_id.clone(),
        thread_id: Some(stored.session.thread_id.clone()),
        turn_id: Some("turn-live-ordinal".to_string()),
        event_type: event_type.to_string(),
        timestamp: format!("2026-07-15T00:00:{sequence:02}Z"),
        payload,
    };
    let tool_item = |status: &str| {
        json!({
            "sessionId": "producer-session",
            "threadId": "producer-thread",
            "turnId": "producer-turn",
            "itemId": "tool-live-ordinal",
            "sequence": 1,
            "ordinal": 1,
            "createdAtMs": 1,
            "updatedAtMs": 1,
            "completedAtMs": if status == "completed" { Some(1) } else { None },
            "kind": "tool",
            "status": status,
            "payload": {
                "type": "tool",
                "call_id": "tool-live-ordinal",
                "name": "read_file",
                "arguments": [],
                "output": null
            },
            "metadata": {}
        })
    };

    store
        .apply_canonical_events(
            &stored,
            &[event(
                1,
                "item.completed",
                json!({
                    "imported": true,
                    "sourceClient": "codex",
                    "importVersion": 2,
                    "sourceEventSeq": 20,
                    "item": {
                        "sessionId": "producer-session",
                        "threadId": "producer-thread",
                        "turnId": "producer-turn",
                        "itemId": "message-imported-ordinal",
                        "sequence": 0,
                        "ordinal": 20,
                        "createdAtMs": 0,
                        "updatedAtMs": 0,
                        "completedAtMs": 0,
                        "kind": "agentMessage",
                        "status": "completed",
                        "payload": {
                            "type": "agentMessage",
                            "text": "imported",
                            "content_parts": []
                        },
                        "metadata": {"source_event_seq": 20}
                    }
                }),
            )],
        )
        .expect("imported Item must use Lime EventLog ordinal");
    store
        .apply_canonical_events(
            &stored,
            &[event(
                20,
                "item.started",
                json!({"item": tool_item("inProgress")}),
            )],
        )
        .expect("live Item must not collide with imported source ordinal");
    store
        .apply_canonical_events(
            &stored,
            &[event(
                21,
                "item.completed",
                json!({"item": tool_item("completed")}),
            )],
        )
        .expect("tool completion must preserve the first live ordinal");
    drop(store);

    let store = ProjectionStore::initialize(database_path).expect("reopen projection store");
    let thread = block_on(store.read_thread(ReadThreadParams {
        thread_id: ThreadId::new("thread-live-ordinal"),
        include_archived: false,
        turns_view: ThreadTurnsView::Full,
    }))
    .expect("read live ordinal thread")
    .expect("live ordinal thread");
    let items = &thread.turns[0].items;
    assert_eq!(items.len(), 2);
    assert_eq!(items[0].ordinal, 1);
    assert_eq!(items[0].item_id.as_str(), "message-imported-ordinal");
    assert_eq!(items[1].ordinal, 20);
    assert_eq!(items[1].item_id.as_str(), "tool-live-ordinal");
    assert_eq!(items[1].status, ItemStatus::Completed);
}

#[test]
fn plan_revision_identity_survives_canonical_store_restart() {
    let temp = tempfile::tempdir().expect("tempdir");
    let database_path = temp.path().join("projection.sqlite");
    let stored = StoredSession {
        session: app_server_protocol::AgentSession {
            session_id: "session-plan-restart".to_string(),
            thread_id: "thread-plan-restart".to_string(),
            app_id: "agent-chat".to_string(),
            workspace_id: None,
            business_object_ref: None,
            status: app_server_protocol::AgentSessionStatus::Completed,
            created_at: "2026-07-14T00:00:00Z".to_string(),
            updated_at: "2026-07-14T00:00:03Z".to_string(),
        },
        turns: Vec::new(),
        turn_inputs: HashMap::new(),
        turn_runtime_options: HashMap::new(),
        events: Vec::new(),
        output_blobs: HashMap::new(),
    };
    let event = |sequence: u64, event_type: &str, payload| app_server_protocol::AgentEvent {
        event_id: format!("plan-restart-event-{sequence}"),
        sequence,
        session_id: stored.session.session_id.clone(),
        thread_id: Some(stored.session.thread_id.clone()),
        turn_id: Some("turn-plan-restart".to_string()),
        event_type: event_type.to_string(),
        timestamp: format!("2026-07-14T00:00:{sequence:02}Z"),
        payload,
    };

    let store = ProjectionStore::initialize(database_path.clone()).expect("projection store");
    store
        .apply_canonical_events(
            &stored,
            &[event(
                1,
                "plan.delta",
                json!({
                    "text": "- [ ] inspect",
                    "revisionId": "proposed_plan:1",
                    "source": "proposed_plan"
                }),
            )],
        )
        .expect("apply plan delta");
    store
        .apply_canonical_events(
            &stored,
            &[event(
                2,
                "plan.final",
                json!({
                    "text": "- [x] inspect\n- [ ] verify",
                    "revisionId": "proposed_plan:1",
                    "source": "proposed_plan",
                    "plan": [
                        {"step": "inspect", "status": "completed"},
                        {"step": "verify", "status": "in_progress"}
                    ]
                }),
            )],
        )
        .expect("apply plan final");
    drop(store);

    let store = ProjectionStore::initialize(database_path).expect("reopen projection store");
    let thread = block_on(store.read_thread(ReadThreadParams {
        thread_id: ThreadId::new("thread-plan-restart"),
        include_archived: false,
        turns_view: ThreadTurnsView::Full,
    }))
    .expect("read restarted plan thread")
    .expect("restarted plan thread");
    assert_eq!(thread.turns.len(), 1);
    assert_eq!(thread.turns[0].items.len(), 1);
    let item = &thread.turns[0].items[0];
    assert_eq!(
        item.item_id.as_str(),
        "plan_turn-plan-restart_proposed_plan:1"
    );
    assert_eq!(item.ordinal, 1);
    assert_eq!(item.sequence, 2);
    assert_eq!(item.status, ItemStatus::Completed);
    let ThreadItemPayload::Plan {
        revision_id,
        source,
        plan,
        ..
    } = &item.payload
    else {
        panic!("restarted plan payload");
    };
    assert_eq!(revision_id, "proposed_plan:1");
    assert_eq!(source.as_deref(), Some("proposed_plan"));
    assert_eq!(plan[1].status, PlanStepStatus::InProgress);
}

#[test]
fn approval_session_cache_hit_stays_audit_only_after_restart_read() {
    let temp = tempfile::tempdir().expect("tempdir");
    let database_path = temp.path().join("projection.sqlite");
    let stored = StoredSession {
        session: app_server_protocol::AgentSession {
            session_id: "session-approval-cache".to_string(),
            thread_id: "thread-approval-cache".to_string(),
            app_id: "agent-chat".to_string(),
            workspace_id: None,
            business_object_ref: None,
            status: app_server_protocol::AgentSessionStatus::Running,
            created_at: "2026-07-13T00:00:00Z".to_string(),
            updated_at: "2026-07-13T00:00:01Z".to_string(),
        },
        turns: Vec::new(),
        turn_inputs: HashMap::new(),
        turn_runtime_options: HashMap::new(),
        events: Vec::new(),
        output_blobs: HashMap::new(),
    };
    let event = |sequence: u64, event_type: &str, payload| app_server_protocol::AgentEvent {
        event_id: format!("approval-cache-event-{sequence}"),
        sequence,
        session_id: stored.session.session_id.clone(),
        thread_id: Some(stored.session.thread_id.clone()),
        turn_id: Some("turn-approval-cache".to_string()),
        event_type: event_type.to_string(),
        timestamp: format!("2026-07-13T00:00:{sequence:02}Z"),
        payload,
    };
    let read_full = |store: &ProjectionStore| {
        block_on(store.read_thread(ReadThreadParams {
            thread_id: ThreadId::new("thread-approval-cache"),
            include_archived: false,
            turns_view: ThreadTurnsView::Full,
        }))
        .expect("read canonical approval thread")
        .expect("canonical approval thread")
    };

    let store = ProjectionStore::initialize(&database_path).expect("projection store");
    store
        .apply_canonical_events(
            &stored,
            &[
                event(
                    1,
                    "approval.session_cache.hit",
                    json!({
                        "request_id": "provider-request-1",
                        "sourceRequestId": "approval-turn-initial",
                        "decision": "allow_for_session",
                        "decisionScope": "session",
                    }),
                ),
                event(
                    2,
                    "action.resolved",
                    json!({
                        "requestId": "permission-turn-approval-cache",
                        "actionId": "permission-turn-approval-cache",
                        "actionType": "tool_confirmation",
                        "source": "approval_session_cache",
                        "decision": "allow_for_session",
                        "decisionScope": "session",
                    }),
                ),
            ],
        )
        .expect("apply cache-backed approval resolution");

    let assert_terminal_approval = |thread: &Thread| {
        assert_eq!(thread.turns.len(), 1);
        assert_eq!(thread.turns[0].items.len(), 1);
        assert_eq!(thread.turns[0].approval, TurnApprovalState::Approved);
        assert_eq!(thread.turns[0].items[0].status, ItemStatus::Completed);
        assert!(matches!(
            &thread.turns[0].items[0].payload,
            ThreadItemPayload::Approval {
                request_id,
                decision: Some(agent_protocol::ApprovalDecision::ApprovedForSession),
                ..
            } if request_id == "permission-turn-approval-cache"
        ));
    };

    assert_terminal_approval(&read_full(&store));
    drop(store);

    let restarted = ProjectionStore::initialize(&database_path).expect("reopen projection store");
    assert_terminal_approval(&read_full(&restarted));
}

#[test]
fn canonical_queue_state_survives_incremental_apply_restart_remove_and_start() {
    let temp = tempfile::tempdir().expect("tempdir");
    let database_path = temp.path().join("projection.sqlite");
    let stored = StoredSession {
        session: app_server_protocol::AgentSession {
            session_id: "session-queue".to_string(),
            thread_id: "thread-queue".to_string(),
            app_id: "agent-chat".to_string(),
            workspace_id: None,
            business_object_ref: None,
            status: app_server_protocol::AgentSessionStatus::Running,
            created_at: "2026-07-12T00:00:00Z".to_string(),
            updated_at: "2026-07-12T00:00:01Z".to_string(),
        },
        turns: Vec::new(),
        turn_inputs: HashMap::new(),
        turn_runtime_options: HashMap::new(),
        events: Vec::new(),
        output_blobs: HashMap::new(),
    };
    let event = |sequence: u64,
                 event_type: &str,
                 turn_id: Option<&str>,
                 queued_turn_id: Option<&str>| app_server_protocol::AgentEvent {
        event_id: format!("queue-event-{sequence}"),
        sequence,
        session_id: stored.session.session_id.clone(),
        thread_id: Some(stored.session.thread_id.clone()),
        turn_id: turn_id.map(str::to_string),
        event_type: event_type.to_string(),
        timestamp: format!("2026-07-12T00:00:{sequence:02}Z"),
        payload: queued_turn_id
            .map(|queued_turn_id| json!({"queuedTurnId": queued_turn_id}))
            .unwrap_or_else(|| json!({})),
    };
    let read_full = |store: &ProjectionStore| {
        block_on(store.read_thread(ReadThreadParams {
            thread_id: ThreadId::new("thread-queue"),
            include_archived: false,
            turns_view: ThreadTurnsView::Full,
        }))
        .expect("read canonical queue thread")
        .expect("canonical queue thread")
    };

    let store = ProjectionStore::initialize(&database_path).expect("projection store");
    store
        .apply_canonical_events(
            &stored,
            &[event(1, "turn.started", Some("turn-active"), None)],
        )
        .expect("apply active turn");
    store
        .apply_canonical_events(
            &stored,
            &[event(
                2,
                "queue.added",
                Some("turn-queued"),
                Some("turn-queued"),
            )],
        )
        .expect("apply queued turn");

    let initial = read_full(&store);
    let active = initial
        .turns
        .iter()
        .find(|turn| turn.turn_id.as_str() == "turn-active")
        .expect("active turn");
    assert_eq!(active.status, TurnStatus::InProgress);
    assert_eq!(active.queue, TurnQueueState::Running);
    let queued = initial
        .turns
        .iter()
        .find(|turn| turn.turn_id.as_str() == "turn-queued")
        .expect("queued turn");
    assert_eq!(queued.status, TurnStatus::InProgress);
    assert!(matches!(queued.queue, TurnQueueState::Queued { .. }));

    drop(store);
    let store = ProjectionStore::initialize(&database_path).expect("reopen projection store");
    let restarted = read_full(&store);
    assert_eq!(restarted.turns.len(), 2);
    assert!(restarted.turns.iter().any(|turn| {
        turn.turn_id.as_str() == "turn-queued"
            && turn.status == TurnStatus::InProgress
            && matches!(turn.queue, TurnQueueState::Queued { .. })
    }));

    store
        .apply_canonical_events(
            &stored,
            &[event(3, "queue.promoted", None, Some("turn-queued"))],
        )
        .expect("apply queue promotion");
    let promoted = read_full(&store);
    assert!(promoted.turns.iter().any(|turn| {
        turn.turn_id.as_str() == "turn-queued"
            && turn.status == TurnStatus::InProgress
            && matches!(turn.queue, TurnQueueState::Queued { .. })
    }));

    store
        .apply_canonical_events(
            &stored,
            &[event(4, "turn.completed", Some("turn-active"), None)],
        )
        .expect("complete active turn");
    store
        .apply_canonical_events(
            &stored,
            &[event(5, "turn.started", Some("turn-queued"), None)],
        )
        .expect("start promoted turn");
    let started = read_full(&store);
    let promoted = started
        .turns
        .iter()
        .find(|turn| turn.turn_id.as_str() == "turn-queued")
        .expect("started promoted turn");
    assert_eq!(promoted.status, TurnStatus::InProgress);
    assert_eq!(promoted.queue, TurnQueueState::Running);

    store
        .apply_canonical_events(
            &stored,
            &[event(
                6,
                "queue.added",
                Some("turn-remove"),
                Some("turn-remove"),
            )],
        )
        .expect("apply removable queued turn");
    store
        .apply_canonical_events(
            &stored,
            &[event(
                7,
                "queue.removed",
                Some("wrong-outer-turn"),
                Some("turn-remove"),
            )],
        )
        .expect("remove queued turn by payload identity");
    assert!(!read_full(&store)
        .turns
        .iter()
        .any(|turn| turn.turn_id.as_str() == "turn-remove"));
}

#[test]
fn canonical_read_keeps_active_turn_when_long_stream_precedes_queue() {
    let temp = tempfile::tempdir().expect("tempdir");
    let stored = StoredSession {
        session: app_server_protocol::AgentSession {
            session_id: "session-active-read".to_string(),
            thread_id: "thread-active-read".to_string(),
            app_id: "agent-chat".to_string(),
            workspace_id: None,
            business_object_ref: None,
            status: app_server_protocol::AgentSessionStatus::Running,
            created_at: "2026-07-13T00:00:00Z".to_string(),
            updated_at: "2026-07-13T00:00:01Z".to_string(),
        },
        turns: Vec::new(),
        turn_inputs: HashMap::new(),
        turn_runtime_options: HashMap::new(),
        events: Vec::new(),
        output_blobs: HashMap::new(),
    };
    let event = |sequence: u64,
                 event_type: &str,
                 turn_id: Option<&str>,
                 payload: serde_json::Value|
     -> app_server_protocol::AgentEvent {
        app_server_protocol::AgentEvent {
            event_id: format!("active-read-event-{sequence}"),
            sequence,
            session_id: stored.session.session_id.clone(),
            thread_id: Some(stored.session.thread_id.clone()),
            turn_id: turn_id.map(str::to_string),
            event_type: event_type.to_string(),
            timestamp: format!("2026-07-13T00:00:{sequence:02}Z"),
            payload,
        }
    };
    let store = ProjectionStore::initialize(temp.path().join("projection.sqlite"))
        .expect("projection store");

    store
        .apply_canonical_events(
            &stored,
            &[event(
                1,
                "turn.accepted",
                Some("turn-active"),
                json!({"source": "turn/start"}),
            )],
        )
        .expect("apply active admission");
    store
        .apply_canonical_events(
            &stored,
            &[event(
                2,
                "message.delta",
                Some("turn-active"),
                json!({"itemId": "message-active", "role": "assistant", "text": "stream"}),
            )],
        )
        .expect("apply active stream");
    store
        .apply_canonical_events(
            &stored,
            &[event(
                3,
                "queue.added",
                Some("turn-queued"),
                json!({"queuedTurnId": "turn-queued", "position": 0}),
            )],
        )
        .expect("apply queued turn");

    let thread = block_on(store.read_thread(ReadThreadParams {
        thread_id: ThreadId::new("thread-active-read"),
        include_archived: false,
        turns_view: ThreadTurnsView::Full,
    }))
    .expect("read canonical active thread")
    .expect("canonical active thread");
    assert_eq!(thread.turns.len(), 2);
    let active = thread
        .turns
        .iter()
        .find(|turn| turn.turn_id.as_str() == "turn-active")
        .expect("active turn is durable");
    assert_eq!(active.status, TurnStatus::InProgress);
    assert_eq!(active.queue, TurnQueueState::Running);
    assert!(thread
        .turns
        .iter()
        .any(|turn| turn.turn_id.as_str() == "turn-queued"
            && matches!(turn.queue, TurnQueueState::Queued { .. })));
}

#[test]
fn completed_turn_goal_accounting_is_exactly_once_across_retry_and_restart() {
    let temp = tempfile::tempdir().expect("tempdir");
    let path = temp.path().join("state.sqlite");
    let store = ProjectionStore::initialize(&path).expect("projection store");
    let source = thread("goal-accounting-replay", 1);
    create(&store, &source);
    store
        .set_thread_goal_sync(app_server_protocol::protocol::v2::ThreadGoalSetParams {
            thread_id: source.thread_id.to_string(),
            objective: Some("finish usage accounting".to_string()),
            status: None,
            token_budget: Some(Some(80)),
        })
        .expect("set thread goal");

    let stored = StoredSession {
        session: AgentSession {
            session_id: source.session_id.to_string(),
            thread_id: source.thread_id.to_string(),
            app_id: "agent-chat".to_string(),
            workspace_id: None,
            business_object_ref: None,
            status: AgentSessionStatus::Running,
            created_at: "2026-07-20T00:00:00Z".to_string(),
            updated_at: "2026-07-20T00:00:05Z".to_string(),
        },
        turns: Vec::new(),
        turn_inputs: HashMap::new(),
        turn_runtime_options: HashMap::new(),
        events: Vec::new(),
        output_blobs: HashMap::new(),
    };
    let event = |sequence: u64, event_type: &str, timestamp: &str, payload| AgentEvent {
        event_id: format!("goal-accounting-event-{sequence}"),
        sequence,
        session_id: stored.session.session_id.clone(),
        thread_id: Some(stored.session.thread_id.clone()),
        turn_id: Some("turn-goal-accounting".to_string()),
        event_type: event_type.to_string(),
        timestamp: timestamp.to_string(),
        payload,
    };
    let accepted = event(
        1,
        "turn.accepted",
        "2026-07-20T00:00:00Z",
        json!({"source": "turn/start"}),
    );
    let completed = event(
        2,
        "turn.completed",
        "2026-07-20T00:00:05Z",
        json!({
            "usage": {
                "total_token_usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 30,
                    "output_tokens": 20,
                    "reasoning_output_tokens": 5,
                    "total_tokens": 120
                },
                "last_token_usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 30,
                    "output_tokens": 20,
                    "reasoning_output_tokens": 5,
                    "total_tokens": 120
                },
                "model_context_window": 128000
            }
        }),
    );

    store
        .apply_canonical_events(&stored, std::slice::from_ref(&accepted))
        .expect("bind accepted turn");
    store
        .apply_canonical_events(&stored, std::slice::from_ref(&completed))
        .expect("account completed turn");
    store
        .apply_canonical_events(&stored, std::slice::from_ref(&completed))
        .expect("retry completed turn");

    let goal = store
        .get_thread_goal_sync(source.thread_id.as_str())
        .expect("read accounted goal")
        .expect("accounted goal");
    assert_eq!(goal.tokens_used, 90);
    assert_eq!(goal.time_used_seconds, 5);
    assert_eq!(
        goal.status,
        app_server_protocol::protocol::v2::ThreadGoalStatus::BudgetLimited
    );
    let conn = store.open_thread_store().expect("open state store");
    assert_eq!(
        conn.query_row(
            "SELECT COUNT(*) FROM thread_goal_turn_accounting",
            [],
            |row| row.get::<_, i64>(0)
        )
        .expect("count accounting watermarks"),
        1
    );
    assert_eq!(
        conn.query_row(
            "SELECT COUNT(*) FROM thread_goal_update_outbox",
            [],
            |row| { row.get::<_, i64>(0) }
        )
        .expect("count goal update outbox"),
        1
    );
    drop(conn);
    drop(store);

    let reopened = ProjectionStore::initialize(&path).expect("reopen projection store");
    reopened
        .apply_canonical_events(&stored, std::slice::from_ref(&completed))
        .expect("retry completed turn after restart");
    let goal = reopened
        .get_thread_goal_sync(source.thread_id.as_str())
        .expect("read goal after restart")
        .expect("goal after restart");
    assert_eq!(goal.tokens_used, 90);
    assert_eq!(goal.time_used_seconds, 5);
}

#[test]
fn plan_turn_goal_accounting_advances_watermark_without_charging() {
    let (_temp, store) = store();
    let source = thread("goal-accounting-plan", 1);
    create(&store, &source);
    store
        .set_thread_goal_sync(app_server_protocol::protocol::v2::ThreadGoalSetParams {
            thread_id: source.thread_id.to_string(),
            objective: Some("plan without charging".to_string()),
            status: None,
            token_budget: None,
        })
        .expect("set plan goal");
    let mut turn_runtime_options = HashMap::new();
    turn_runtime_options.insert(
        "turn-plan".to_string(),
        app_server_protocol::RuntimeOptions {
            runtime_request: Some(app_server_protocol::RuntimeRequest {
                collaboration_mode: Some(agent_protocol::CollaborationMode {
                    mode: agent_protocol::ModeKind::Plan,
                    settings: agent_protocol::CollaborationModeSettings {
                        model: "goal-plan-model".to_string(),
                        reasoning_effort: None,
                        developer_instructions: None,
                    },
                }),
                ..Default::default()
            }),
            ..Default::default()
        },
    );
    let stored = StoredSession {
        session: AgentSession {
            session_id: source.session_id.to_string(),
            thread_id: source.thread_id.to_string(),
            app_id: "agent-chat".to_string(),
            workspace_id: None,
            business_object_ref: None,
            status: AgentSessionStatus::Running,
            created_at: "2026-07-20T00:00:00Z".to_string(),
            updated_at: "2026-07-20T00:00:03Z".to_string(),
        },
        turns: Vec::new(),
        turn_inputs: HashMap::new(),
        turn_runtime_options,
        events: Vec::new(),
        output_blobs: HashMap::new(),
    };
    let event = |sequence: u64, event_type: &str, timestamp: &str, payload| AgentEvent {
        event_id: format!("goal-plan-event-{sequence}"),
        sequence,
        session_id: stored.session.session_id.clone(),
        thread_id: Some(stored.session.thread_id.clone()),
        turn_id: Some("turn-plan".to_string()),
        event_type: event_type.to_string(),
        timestamp: timestamp.to_string(),
        payload,
    };
    store
        .apply_canonical_events(
            &stored,
            &[
                event(
                    1,
                    "turn.accepted",
                    "2026-07-20T00:00:00Z",
                    json!({"source": "turn/start"}),
                ),
                event(
                    2,
                    "turn.completed",
                    "2026-07-20T00:00:03Z",
                    json!({
                        "usage": {
                            "total_token_usage": {
                                "input_tokens": 50,
                                "cached_input_tokens": 10,
                                "output_tokens": 20,
                                "reasoning_output_tokens": 0,
                                "total_tokens": 70
                            },
                            "last_token_usage": {
                                "input_tokens": 50,
                                "cached_input_tokens": 10,
                                "output_tokens": 20,
                                "reasoning_output_tokens": 0,
                                "total_tokens": 70
                            },
                            "model_context_window": 128000
                        }
                    }),
                ),
            ],
        )
        .expect("apply plan turn accounting events");

    let goal = store
        .get_thread_goal_sync(source.thread_id.as_str())
        .expect("read plan goal")
        .expect("plan goal");
    assert_eq!(goal.tokens_used, 0);
    assert_eq!(goal.time_used_seconds, 0);
    let conn = store.open_thread_store().expect("open plan state store");
    assert_eq!(
        conn.query_row(
            "SELECT COUNT(*) FROM thread_goal_update_outbox",
            [],
            |row| { row.get::<_, i64>(0) }
        )
        .expect("count plan outbox"),
        0
    );
}
