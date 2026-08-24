use super::support::*;
use super::*;
use agent_protocol::{AgentInput as RuntimeInput, ImageDetail};
use std::sync::atomic::AtomicUsize;
use tokio::time::{sleep, timeout};

const INLINE_PNG_DATA_URL: &str =
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==";

fn start_queue_session(core: &RuntimeCore, session_id: &str, thread_id: &str) {
    core.start_session(AgentSessionStartParams {
        session_id: Some(session_id.to_string()),
        thread_id: Some(thread_id.to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("queue session");
}

fn text_input(text: impl Into<String>) -> Vec<RuntimeInput> {
    vec![RuntimeInput::text(text)]
}

#[tokio::test]
async fn thread_queue_crud_reorder_and_duplicate_client_ids_are_durable() {
    let projection_root = tempfile::tempdir().expect("queue CRUD projection root");
    let projection_store = Arc::new(
        ProjectionStore::initialize(projection_root.path().join("projection.sqlite"))
            .expect("queue CRUD projection store"),
    );
    let core = RuntimeCore::default().with_projection_store(projection_store);
    start_queue_session(&core, "sess_thread_queue_crud", "thread_queue_crud");

    let first = core
        .add_thread_queue_submission(
            "thread_queue_crud",
            text_input("first"),
            "client-duplicate".to_string(),
        )
        .await
        .expect("first queued submission");
    let second = core
        .add_thread_queue_submission(
            "thread_queue_crud",
            text_input("second"),
            "client-duplicate".to_string(),
        )
        .await
        .expect("second queued submission");
    let third = core
        .add_thread_queue_submission(
            "thread_queue_crud",
            text_input("third"),
            "client-third".to_string(),
        )
        .await
        .expect("third queued submission");

    assert_ne!(first.id, second.id);
    assert_eq!(first.client_user_message_id, second.client_user_message_id);
    assert_eq!(
        core.list_thread_queue_submissions("thread_queue_crud")
            .await
            .expect("initial queue")
            .into_iter()
            .map(|submission| submission.id)
            .collect::<Vec<_>>(),
        vec![first.id.clone(), second.id.clone(), third.id.clone()]
    );

    let updated = core
        .update_thread_queue_submission(
            "thread_queue_crud",
            &second.id,
            text_input("second updated"),
        )
        .await
        .expect("updated submission");
    assert_eq!(updated.id, second.id);
    assert_eq!(updated.client_user_message_id, "client-duplicate");
    assert_eq!(updated.input, text_input("second updated"));

    core.reorder_thread_queue_submissions(
        "thread_queue_crud",
        vec![third.id.clone(), second.id.clone(), first.id.clone()],
    )
    .await
    .expect("reordered queue");
    let reordered = core
        .list_thread_queue_submissions("thread_queue_crud")
        .await
        .expect("reordered queue list");
    assert_eq!(
        reordered
            .iter()
            .map(|submission| submission.id.as_str())
            .collect::<Vec<_>>(),
        vec![third.id.as_str(), second.id.as_str(), first.id.as_str()]
    );
    assert_eq!(reordered[1].input, text_input("second updated"));

    assert!(core
        .delete_thread_queue_submission("thread_queue_crud", &second.id)
        .await
        .expect("delete queued submission"));
    assert!(!core
        .delete_thread_queue_submission("thread_queue_crud", &second.id)
        .await
        .expect("delete missing queued submission"));
    assert_eq!(
        core.list_thread_queue_submissions("thread_queue_crud")
            .await
            .expect("queue after delete")
            .into_iter()
            .map(|submission| submission.id)
            .collect::<Vec<_>>(),
        vec![third.id, first.id]
    );
}

#[tokio::test]
async fn thread_queue_cold_restart_restores_order_multimodal_input_and_client_id() {
    let temp = tempfile::tempdir().expect("queue restart tempdir");
    let roots =
        StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
    let event_log_writer = Arc::new(EventLogWriter::new(&roots.event_log_root).expect("writer"));
    let projection_store =
        Arc::new(ProjectionStore::initialize(&roots.projection_db_path).expect("projection"));
    let sidecar_store = Arc::new(SidecarStore::new(&roots.sidecar_root).expect("sidecar"));
    let core = RuntimeCore::default()
        .with_event_log_writer(event_log_writer.clone())
        .with_projection_store(projection_store.clone())
        .with_sidecar_store(sidecar_store.clone());
    start_queue_session(&core, "sess_thread_queue_restart", "thread_queue_restart");

    let first = core
        .add_thread_queue_submission(
            "thread_queue_restart",
            text_input("first after restart"),
            "client-first".to_string(),
        )
        .await
        .expect("first queued submission");
    let second = core
        .add_thread_queue_submission(
            "thread_queue_restart",
            vec![
                RuntimeInput::text("inspect this image after restart"),
                RuntimeInput::Image {
                    uri: INLINE_PNG_DATA_URL.to_string(),
                    detail: Some(ImageDetail::High),
                },
            ],
            "client-multimodal".to_string(),
        )
        .await
        .expect("multimodal queued submission");
    core.reorder_thread_queue_submissions(
        "thread_queue_restart",
        vec![second.id.clone(), first.id.clone()],
    )
    .await
    .expect("persist reordered queue");
    drop(core);

    let restarted = RuntimeCore::default()
        .with_event_log_writer(event_log_writer)
        .with_projection_store(projection_store)
        .with_sidecar_store(sidecar_store);
    let restored = restarted
        .list_thread_queue_submissions("thread_queue_restart")
        .await
        .expect("restored queue");

    assert_eq!(
        restored
            .iter()
            .map(|submission| submission.id.as_str())
            .collect::<Vec<_>>(),
        vec![second.id.as_str(), first.id.as_str()]
    );
    assert_eq!(restored[0].client_user_message_id, "client-multimodal");
    assert_eq!(
        restored[0].input[0],
        RuntimeInput::text("inspect this image after restart")
    );
    match &restored[0].input[1] {
        RuntimeInput::Image { uri, detail } => {
            assert!(uri.starts_with("sidecar://media/input-"));
            assert_eq!(*detail, Some(ImageDetail::High));
        }
        input => panic!("restored multimodal input changed shape: {input:?}"),
    }
}

#[tokio::test]
async fn archive_unarchive_preserves_queue_and_delete_clears_it() {
    let temp = tempfile::tempdir().expect("queue lifecycle tempdir");
    let roots =
        StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
    let event_log_writer = Arc::new(EventLogWriter::new(&roots.event_log_root).expect("writer"));
    let projection_store =
        Arc::new(ProjectionStore::initialize(&roots.projection_db_path).expect("projection"));
    let core = RuntimeCore::default()
        .with_event_log_writer(event_log_writer)
        .with_projection_store(projection_store);
    start_queue_session(
        &core,
        "sess_thread_queue_lifecycle",
        "thread_queue_lifecycle",
    );
    let queued = core
        .add_thread_queue_submission(
            "thread_queue_lifecycle",
            text_input("persist across archive"),
            "client-lifecycle".to_string(),
        )
        .await
        .expect("queued lifecycle submission");

    assert!(core
        .archive_thread(agent_protocol::ThreadId::new("thread_queue_lifecycle"))
        .await
        .expect("archive queued thread"));
    assert!(matches!(
        core.list_thread_queue_submissions("thread_queue_lifecycle")
            .await,
        Err(RuntimeCoreError::InvalidRequest(message))
            if message.contains("archived thread does not support queued submissions")
    ));

    let (_, changed) = core
        .unarchive_thread(agent_protocol::ThreadId::new("thread_queue_lifecycle"))
        .await
        .expect("unarchive queued thread");
    assert!(changed);
    assert_eq!(
        core.list_thread_queue_submissions("thread_queue_lifecycle")
            .await
            .expect("queue after unarchive")
            .into_iter()
            .map(|submission| submission.id)
            .collect::<Vec<_>>(),
        vec![queued.id]
    );

    core.delete_thread(agent_protocol::ThreadId::new("thread_queue_lifecycle"))
        .await
        .expect("delete queued thread");
    start_queue_session(
        &core,
        "sess_thread_queue_lifecycle",
        "thread_queue_lifecycle",
    );
    assert!(core
        .list_thread_queue_submissions("thread_queue_lifecycle")
        .await
        .expect("queue after delete and recreate")
        .is_empty());
}

#[tokio::test]
async fn cold_thread_resume_wakes_persisted_queue_unless_interrupted() {
    let temp = tempfile::tempdir().expect("queue resume tempdir");
    let roots =
        StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
    let event_log_writer = Arc::new(EventLogWriter::new(&roots.event_log_root).expect("writer"));
    let projection_store =
        Arc::new(ProjectionStore::initialize(&roots.projection_db_path).expect("projection"));
    let source = RuntimeCore::default()
        .with_event_log_writer(event_log_writer.clone())
        .with_projection_store(projection_store.clone());
    start_queue_session(&source, "sess_thread_queue_resume", "thread_queue_resume");
    source
        .add_thread_queue_submission(
            "thread_queue_resume",
            text_input("dispatch after cold resume"),
            "client-cold-resume".to_string(),
        )
        .await
        .expect("persist queued resume submission");
    drop(source);

    let backend = Arc::new(RunningCountingBackend {
        start_count: AtomicUsize::new(0),
    });
    let resumed = RuntimeCore::with_backend(backend.clone())
        .with_event_log_writer(event_log_writer)
        .with_projection_store(projection_store);
    let snapshot = resumed
        .resume_thread(agent_protocol::ThreadId::new("thread_queue_resume"))
        .await
        .expect("resume queued thread");
    assert!(snapshot.active_turn_id.is_none());
    resumed.wake_thread_queue_if_idle("thread_queue_resume", RuntimeHostContext::default());
    timeout(Duration::from_secs(2), async {
        loop {
            if backend.start_count.load(Ordering::SeqCst) == 1 {
                break;
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("cold resume should start queued submission");
    assert!(resumed
        .list_thread_queue_submissions("thread_queue_resume")
        .await
        .expect("queue after cold resume")
        .is_empty());
}

#[tokio::test]
async fn active_thread_rejects_queue_start_and_interruption_keeps_queue_paused() {
    let backend = Arc::new(RunningCountingBackend {
        start_count: AtomicUsize::new(0),
    });
    let projection_root = tempfile::tempdir().expect("paused queue projection root");
    let projection_store = Arc::new(
        ProjectionStore::initialize(projection_root.path().join("projection.sqlite"))
            .expect("paused queue projection store"),
    );
    let core = RuntimeCore::with_backend(backend.clone()).with_projection_store(projection_store);
    start_queue_session(&core, "sess_thread_queue_pause", "thread_queue_pause");
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_thread_queue_pause".to_string(),
            turn_id: Some("turn-active".to_string()),
            input: AgentInput {
                text: "active".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("active turn");
    let queued = core
        .add_thread_queue_submission(
            "thread_queue_pause",
            text_input("queued after interruption"),
            "client-paused".to_string(),
        )
        .await
        .expect("queued submission");

    let error = core
        .start_thread_queue_submission("thread_queue_pause", None, RuntimeHostContext::default())
        .await
        .expect_err("active thread must reject queue start");
    assert!(error.to_string().contains("active or pending turn"));
    assert_eq!(
        core.list_thread_queue_submissions("thread_queue_pause")
            .await
            .expect("queue remains intact")
            .len(),
        1
    );

    core.append_external_runtime_events(
        "sess_thread_queue_pause",
        Some("turn-active"),
        vec![RuntimeEvent::new("turn.canceled", json!({}))],
    )
    .expect("interrupt active turn");
    sleep(Duration::from_millis(50)).await;
    assert_eq!(backend.start_count.load(Ordering::SeqCst), 1);
    assert_eq!(
        core.list_thread_queue_submissions("thread_queue_pause")
            .await
            .expect("paused queue")
            .len(),
        1
    );

    core.wake_thread_queue_if_idle("thread_queue_pause", RuntimeHostContext::default());
    sleep(Duration::from_millis(50)).await;
    assert_eq!(backend.start_count.load(Ordering::SeqCst), 1);

    let started = core
        .start_thread_queue_submission(
            "thread_queue_pause",
            Some(&queued.id),
            RuntimeHostContext::default(),
        )
        .await
        .expect("explicit queue start after interruption");
    assert_eq!(started.turn_id, queued.id);
    assert_eq!(backend.start_count.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn completed_and_failed_turns_advance_thread_queue_fifo() {
    for (terminal_event, suffix) in [("turn.completed", "completed"), ("turn.failed", "failed")] {
        let backend = Arc::new(ExternallyTerminatedBackend::default());
        let projection_root = tempfile::tempdir().expect("FIFO queue projection root");
        let projection_store = Arc::new(
            ProjectionStore::initialize(projection_root.path().join("projection.sqlite"))
                .expect("FIFO queue projection store"),
        );
        let core =
            RuntimeCore::with_backend(backend.clone()).with_projection_store(projection_store);
        let session_id = format!("sess_thread_queue_{suffix}");
        let thread_id = format!("thread_queue_{suffix}");
        start_queue_session(&core, &session_id, &thread_id);
        core.start_turn_admitted(
            AgentSessionTurnStartParams {
                session_id: session_id.clone(),
                turn_id: Some(format!("turn-{suffix}")),
                input: AgentInput {
                    text: "active".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("active turn");
        let first = core
            .add_thread_queue_submission(
                &thread_id,
                text_input("first queued"),
                format!("client-first-{suffix}"),
            )
            .await
            .expect("first queued submission");
        let second = core
            .add_thread_queue_submission(
                &thread_id,
                text_input("second queued"),
                format!("client-second-{suffix}"),
            )
            .await
            .expect("second queued submission");

        core.append_external_runtime_events(
            &session_id,
            Some(&format!("turn-{suffix}")),
            vec![RuntimeEvent::new(terminal_event, json!({}))],
        )
        .expect("finish active turn");
        backend.release_initial.notify_one();
        timeout(Duration::from_secs(2), async {
            loop {
                if backend.start_count.load(Ordering::SeqCst) >= 2 {
                    break;
                }
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("queue head should start");

        let remaining = core
            .list_thread_queue_submissions(&thread_id)
            .await
            .expect("remaining queue");
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].id, second.id);
        assert_ne!(remaining[0].id, first.id);
    }
}
