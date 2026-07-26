use super::*;
use crate::runtime::projection_store::ProjectionReadWindow;
use crate::{EventLogWriter, ProjectionStore, SidecarStore, StorageRoots};
use app_server_protocol::ConversationImportThreadStatus;

#[test]
fn committing_same_codex_thread_reuses_existing_imported_session() {
    let temp = tempfile::tempdir().expect("tempdir");
    let rollout_path = temp.path().join("rollout-thread-idempotent.jsonl");
    fs::write(
        &rollout_path,
        [
            codex_session_meta_line(
                "thread-idempotent",
                "/workspace/idempotent",
                "idempotent import",
            ),
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:02.000Z",
                "type": "event_msg",
                "payload": {
                    "type": "agent_message",
                    "message": "idempotent reply"
                }
            })
            .to_string(),
        ]
        .join("\n"),
    )
    .expect("write rollout");
    let core = RuntimeCore::default();

    let first = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect("first commit");
    let second = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect("second commit");

    assert_eq!(second.session.session_id, first.session.session_id);
    assert_eq!(
        second.thread.import_status,
        ConversationImportThreadStatus::Imported
    );
    assert_eq!(second.imported_messages, first.imported_messages);
    assert_eq!(second.imported_turns, first.imported_turns);
    assert_eq!(
        core.state
            .lock()
            .expect("runtime core state mutex poisoned")
            .sessions
            .len(),
        1
    );
}

#[tokio::test]
async fn committing_same_codex_thread_with_replace_existing_reimports_source() {
    let temp = tempfile::tempdir().expect("tempdir");
    let roots =
        StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
    let event_log_writer = Arc::new(EventLogWriter::new(&roots.event_log_root).expect("writer"));
    let projection_store =
        Arc::new(ProjectionStore::initialize(&roots.projection_db_path).expect("projection"));
    let rollout_path = temp.path().join("rollout-thread-replace.jsonl");
    write_rollout_with_assistant_reply(
        &rollout_path,
        "thread-replace",
        "/workspace/replace",
        "replace import",
        "old imported reply",
    );
    let core = RuntimeCore::default()
        .with_event_log_writer(event_log_writer.clone())
        .with_projection_store(projection_store.clone());

    let first = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect("first commit");
    assert!(projection_store
        .read_session_projection(&first.session.session_id, ProjectionReadWindow::default())
        .expect("old projection read")
        .is_some());
    assert!(!event_log_writer
        .read_session_events(&first.session.session_id)
        .expect("old events")
        .is_empty());
    write_rollout_with_assistant_reply(
        &rollout_path,
        "thread-replace",
        "/workspace/replace",
        "replace import",
        "new imported reply",
    );
    let second = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            confirmed: true,
            replace_existing: Some(true),
            ..Default::default()
        },
    )
    .expect("replace commit");

    assert_ne!(second.session.session_id, first.session.session_id);
    assert_eq!(
        core.state
            .lock()
            .expect("runtime core state mutex poisoned")
            .sessions
            .len(),
        1
    );
    assert!(projection_store
        .read_session_projection(&first.session.session_id, ProjectionReadWindow::default())
        .expect("old projection after replace")
        .is_none());
    assert!(event_log_writer
        .read_session_events(&first.session.session_id)
        .expect("old event log after replace")
        .is_empty());
    assert!(projection_store
        .read_session_projection(&second.session.session_id, ProjectionReadWindow::default())
        .expect("new projection after replace")
        .is_some());

    let missing_old = core
        .read_session_current(AgentSessionReadParams {
            session_id: first.session.session_id.clone(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .await
        .expect_err("old imported session should be cleared");
    assert!(matches!(
        missing_old,
        RuntimeCoreError::SessionNotFound(session_id) if session_id == first.session.session_id
    ));

    let read = core
        .read_session_current(AgentSessionReadParams {
            session_id: second.session.session_id.clone(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .await
        .expect("read replaced session");
    let messages = read
        .detail
        .as_ref()
        .and_then(|detail| detail.get("messages"))
        .and_then(serde_json::Value::as_array)
        .expect("read messages");
    let assistant_texts = messages
        .iter()
        .filter(|message| {
            message.get("role").and_then(serde_json::Value::as_str) == Some("assistant")
        })
        .filter_map(|message| {
            message
                .pointer("/content/0/text")
                .and_then(serde_json::Value::as_str)
        })
        .collect::<Vec<_>>();
    assert!(assistant_texts.contains(&"new imported reply"));
    assert!(!assistant_texts.contains(&"old imported reply"));
}

#[tokio::test]
async fn committing_same_codex_thread_after_restart_reuses_projected_session() {
    let temp = tempfile::tempdir().expect("tempdir");
    let roots =
        StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
    let event_log_writer = Arc::new(EventLogWriter::new(&roots.event_log_root).expect("writer"));
    let projection_store =
        Arc::new(ProjectionStore::initialize(&roots.projection_db_path).expect("projection"));
    let db_path = temp.path().join("state_5.sqlite");
    let rollout_path = temp.path().join("rollout-thread-restarted.jsonl");
    fs::write(
        &rollout_path,
        [
            codex_session_meta_line("thread-restarted", "/workspace/restarted", "restart import"),
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:02.000Z",
                "type": "event_msg",
                "payload": {
                    "type": "agent_message",
                    "message": "restart reply"
                }
            })
            .to_string(),
        ]
        .join("\n"),
    )
    .expect("write rollout");
    let conn = Connection::open(&db_path).expect("db");
    create_legacy_threads_table(&conn);
    insert_thread(
        &conn,
        "thread-restarted",
        "Restart import",
        "/workspace/restarted",
        &rollout_path.to_string_lossy(),
        1,
        2,
        false,
    );
    let first_core = RuntimeCore::default()
        .with_event_log_writer(event_log_writer.clone())
        .with_projection_store(projection_store.clone());

    let first = commit::commit_conversation_import_thread(
        &first_core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_thread_id: Some("thread-restarted".to_string()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect("first commit");
    assert!(projection_store
        .read_session_projection(&first.session.session_id, ProjectionReadWindow::default())
        .expect("projection read")
        .is_some());

    let backend = Arc::new(RecordingBackend::default());
    let restarted_core = RuntimeCore::with_backend(backend.clone())
        .with_event_log_writer(event_log_writer)
        .with_projection_store(projection_store);
    let second = commit::commit_conversation_import_thread(
        &restarted_core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_thread_id: Some("thread-restarted".to_string()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect("second commit after restart");

    assert_eq!(second.session.session_id, first.session.session_id);
    assert_eq!(
        second.thread.import_status,
        ConversationImportThreadStatus::Imported
    );
    assert_eq!(second.imported_messages, first.imported_messages);
    assert_eq!(second.imported_turns, first.imported_turns);
    assert_eq!(
        restarted_core
            .state
            .lock()
            .expect("runtime core state mutex poisoned")
            .sessions
            .len(),
        0
    );

    let rescan = restarted_core
        .scan_conversation_import_source(ConversationImportSourceScanParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            project_path: Some("/workspace/restarted".to_string()),
            ..Default::default()
        })
        .await
        .expect("rescan after restart");
    assert_eq!(
        rescan.threads[0].import_status,
        ConversationImportThreadStatus::Imported
    );

    let preview = restarted_core
        .preview_conversation_import_thread(ConversationImportThreadPreviewParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_thread_id: Some("thread-restarted".to_string()),
            ..Default::default()
        })
        .await
        .expect("preview after restart");
    assert_eq!(
        preview.thread.import_status,
        ConversationImportThreadStatus::Imported
    );
    assert!(!preview.summary.dry_run.will_create_session);
    assert!(preview.summary.dry_run.will_append_to_existing_session);

    let read = restarted_core
        .read_session_current(AgentSessionReadParams {
            session_id: first.session.session_id.clone(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .await
        .expect("read imported session after restart");
    let business_object_ref = read
        .session
        .business_object_ref
        .as_ref()
        .expect("import business object ref after restart");
    assert_eq!(
        business_object_ref.kind,
        import_status::IMPORTED_CONVERSATION_KIND
    );
    assert_eq!(business_object_ref.id, "thread-restarted");
    assert_eq!(
        business_object_ref
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("sourceThreadId"))
            .and_then(serde_json::Value::as_str),
        Some("thread-restarted")
    );

    restarted_core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: first.session.session_id.clone(),
                turn_id: Some("turn-after-restart".to_string()),
                input: AgentInput {
                    text: "continue after restart".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("continue imported session after restart");
    let requests = backend.requests.lock().expect("requests mutex poisoned");
    let request = requests.last().expect("recorded continuation request");
    assert_eq!(request.provider_preference(), None);
    assert_eq!(request.model_preference(), None);
    assert_eq!(
        request
            .runtime_options
            .as_ref()
            .and_then(|options| options.runtime_request.as_ref())
            .and_then(|runtime_request| runtime_request.working_dir.as_deref()),
        Some("/workspace/restarted")
    );
}

#[tokio::test]
async fn scan_and_preview_mark_previously_imported_codex_thread() {
    let temp = tempfile::tempdir().expect("tempdir");
    let db_path = temp.path().join("state_5.sqlite");
    let rollout_path = temp.path().join("rollout-thread-status.jsonl");
    fs::write(
        &rollout_path,
        codex_session_meta_line("thread-status", "/workspace/status", "status import"),
    )
    .expect("write rollout");
    let conn = Connection::open(&db_path).expect("db");
    create_legacy_threads_table(&conn);
    insert_thread(
        &conn,
        "thread-status",
        "Status import",
        "/workspace/status",
        &rollout_path.to_string_lossy(),
        1,
        2,
        false,
    );
    let core = RuntimeCore::default();

    let initial_scan = core
        .scan_conversation_import_source(ConversationImportSourceScanParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            project_path: Some("/workspace/status".to_string()),
            ..Default::default()
        })
        .await
        .expect("initial scan");
    assert_eq!(
        initial_scan.threads[0].import_status,
        ConversationImportThreadStatus::NotImported
    );

    commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_thread_id: Some("thread-status".to_string()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect("commit");

    let rescan = core
        .scan_conversation_import_source(ConversationImportSourceScanParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            project_path: Some("/workspace/status".to_string()),
            ..Default::default()
        })
        .await
        .expect("rescan");
    assert_eq!(
        rescan.threads[0].import_status,
        ConversationImportThreadStatus::Imported
    );

    let preview = core
        .preview_conversation_import_thread(ConversationImportThreadPreviewParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_thread_id: Some("thread-status".to_string()),
            ..Default::default()
        })
        .await
        .expect("preview");
    assert_eq!(
        preview.thread.import_status,
        ConversationImportThreadStatus::Imported
    );
}

#[cfg(unix)]
#[test]
fn event_log_append_failure_rolls_back_all_import_state() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().expect("tempdir");
    let roots =
        StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
    let event_log_writer = Arc::new(EventLogWriter::new(&roots.event_log_root).expect("writer"));
    let projection_store =
        Arc::new(ProjectionStore::initialize(&roots.projection_db_path).expect("projection"));
    let sidecar_store = Arc::new(SidecarStore::new(&roots.sidecar_root).expect("sidecar"));
    let event_sessions = roots.event_log_root.join("sessions");
    fs::create_dir_all(&event_sessions).expect("event sessions directory");
    fs::set_permissions(&event_sessions, fs::Permissions::from_mode(0o555))
        .expect("make event sessions read only");

    let rollout_path = temp.path().join("rollout-event-log-failure.jsonl");
    write_rollout_with_assistant_reply(
        &rollout_path,
        "thread-event-log-failure",
        "/workspace/event-log-failure",
        "event log failure import",
        "reply must roll back",
    );
    let core = RuntimeCore::default()
        .with_event_log_writer(event_log_writer)
        .with_projection_store(projection_store)
        .with_sidecar_store(sidecar_store);

    let result = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            confirmed: true,
            ..Default::default()
        },
    );
    fs::set_permissions(&event_sessions, fs::Permissions::from_mode(0o755))
        .expect("restore event sessions permissions");

    let error = result.expect_err("event log append failure must fail the import");
    assert!(error.to_string().contains("event log"));
    assert_import_storage_is_empty(&core, &roots);
}

#[test]
fn canonical_projection_failure_after_event_log_append_rolls_back_all_import_state() {
    let temp = tempfile::tempdir().expect("tempdir");
    let roots =
        StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
    let event_log_writer = Arc::new(EventLogWriter::new(&roots.event_log_root).expect("writer"));
    let projection_store =
        Arc::new(ProjectionStore::initialize(&roots.projection_db_path).expect("projection"));
    let sidecar_store = Arc::new(SidecarStore::new(&roots.sidecar_root).expect("sidecar"));
    Connection::open(&roots.projection_db_path)
        .expect("projection connection")
        .execute_batch(
            "CREATE TRIGGER fail_import_item_insert
             BEFORE INSERT ON canonical_items
             BEGIN
               SELECT RAISE(ABORT, 'injected canonical item failure');
             END;",
        )
        .expect("projection failure trigger");

    let rollout_path = temp.path().join("rollout-projection-failure.jsonl");
    write_rollout_with_assistant_reply(
        &rollout_path,
        "thread-projection-failure",
        "/workspace/projection-failure",
        "projection failure import",
        "reply must roll back",
    );
    let core = RuntimeCore::default()
        .with_event_log_writer(event_log_writer)
        .with_projection_store(projection_store)
        .with_sidecar_store(sidecar_store);

    let error = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect_err("canonical projection failure must fail the import");

    assert!(error
        .to_string()
        .contains("injected canonical item failure"));
    assert_import_storage_is_empty(&core, &roots);
}

#[test]
fn cleanup_failure_reports_both_import_and_compensation_errors() {
    let temp = tempfile::tempdir().expect("tempdir");
    let roots =
        StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
    let event_log_writer = Arc::new(EventLogWriter::new(&roots.event_log_root).expect("writer"));
    let projection_store =
        Arc::new(ProjectionStore::initialize(&roots.projection_db_path).expect("projection"));
    let sidecar_store = Arc::new(SidecarStore::new(&roots.sidecar_root).expect("sidecar"));
    Connection::open(&roots.projection_db_path)
        .expect("projection connection")
        .execute_batch(
            "CREATE TRIGGER fail_import_item_insert
             BEFORE INSERT ON canonical_items
             BEGIN
               SELECT RAISE(ABORT, 'injected import projection failure');
             END;
             CREATE TRIGGER fail_import_thread_cleanup
             BEFORE DELETE ON canonical_threads
             BEGIN
               SELECT RAISE(ABORT, 'injected projection cleanup failure');
             END;",
        )
        .expect("projection failure triggers");

    let rollout_path = temp.path().join("rollout-cleanup-failure.jsonl");
    write_rollout_with_assistant_reply(
        &rollout_path,
        "thread-cleanup-failure",
        "/workspace/cleanup-failure",
        "cleanup failure import",
        "reply must fail closed",
    );
    let core = RuntimeCore::default()
        .with_event_log_writer(event_log_writer)
        .with_projection_store(projection_store)
        .with_sidecar_store(sidecar_store);

    let error = commit::commit_conversation_import_thread(
        &core,
        ConversationImportThreadCommitParams {
            source_root: Some(temp.path().to_string_lossy().into_owned()),
            source_path: Some(rollout_path.to_string_lossy().into_owned()),
            confirmed: true,
            ..Default::default()
        },
    )
    .expect_err("cleanup failure must remain fail closed")
    .to_string();

    assert!(error.contains("injected import projection failure"));
    assert!(error.contains("compensating cleanup also failed"));
    assert!(error.contains("injected projection cleanup failure"));
    assert!(
        core.state
            .lock()
            .expect("runtime core state mutex poisoned")
            .sessions
            .is_empty(),
        "best-effort cleanup must still remove in-memory state"
    );
    assert_directory_is_empty(&roots.event_log_root.join("sessions"));
    assert_directory_is_empty(&roots.sidecar_root.join("sessions"));
}

fn assert_import_storage_is_empty(core: &RuntimeCore, roots: &StorageRoots) {
    assert!(
        core.state
            .lock()
            .expect("runtime core state mutex poisoned")
            .sessions
            .is_empty(),
        "failed import must not leave an in-memory session"
    );

    let connection = Connection::open(&roots.projection_db_path).expect("projection connection");
    for table in [
        "canonical_threads",
        "canonical_turns",
        "canonical_items",
        "projected_sessions",
        "projected_turns",
        "projected_items",
        "projection_watermarks",
    ] {
        let count = connection
            .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap_or_else(|error| panic!("read {table} count: {error}"));
        assert_eq!(count, 0, "failed import left rows in {table}");
    }
    assert_directory_is_empty(&roots.event_log_root.join("sessions"));
    assert_directory_is_empty(&roots.sidecar_root.join("sessions"));
}

fn assert_directory_is_empty(path: &std::path::Path) {
    if !path.exists() {
        return;
    }
    assert!(
        fs::read_dir(path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
            .next()
            .is_none(),
        "{} must be empty",
        path.display()
    );
}

fn write_rollout_with_assistant_reply(
    path: &std::path::Path,
    thread_id: &str,
    cwd: &str,
    user_message: &str,
    assistant_reply: &str,
) {
    fs::write(
        path,
        [
            codex_session_meta_line(thread_id, cwd, user_message),
            serde_json::json!({
                "timestamp": "2026-06-16T00:00:02.000Z",
                "type": "event_msg",
                "payload": {
                    "type": "agent_message",
                    "message": assistant_reply
                }
            })
            .to_string(),
        ]
        .join("\n"),
    )
    .expect("write rollout with assistant reply");
}
