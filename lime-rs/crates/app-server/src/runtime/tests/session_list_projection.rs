use super::support::*;
use super::*;

struct ProjectionTestHarness {
    _temp: tempfile::TempDir,
    event_log_writer: Arc<EventLogWriter>,
    projection_store: Arc<ProjectionStore>,
    core: RuntimeCore,
}

fn projection_test_core() -> ProjectionTestHarness {
    let temp = tempfile::tempdir().expect("tempdir");
    let roots =
        StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
    let event_log_writer = Arc::new(EventLogWriter::new(&roots.event_log_root).expect("writer"));
    let projection_store =
        Arc::new(ProjectionStore::initialize(&roots.projection_db_path).expect("projection"));
    let core = RuntimeCore::with_backend(Arc::new(CompletedBackend))
        .with_event_log_writer(event_log_writer.clone())
        .with_projection_store(projection_store.clone());
    ProjectionTestHarness {
        _temp: temp,
        event_log_writer,
        projection_store,
        core,
    }
}

fn projection_running_test_core() -> ProjectionTestHarness {
    let temp = tempfile::tempdir().expect("tempdir");
    let roots =
        StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
    let event_log_writer = Arc::new(EventLogWriter::new(&roots.event_log_root).expect("writer"));
    let projection_store =
        Arc::new(ProjectionStore::initialize(&roots.projection_db_path).expect("projection"));
    let core = RuntimeCore::with_backend(Arc::new(RunningCountingBackend {
        start_count: AtomicUsize::new(0),
    }))
    .with_event_log_writer(event_log_writer.clone())
    .with_projection_store(projection_store.clone());
    ProjectionTestHarness {
        _temp: temp,
        event_log_writer,
        projection_store,
        core,
    }
}

async fn seed_projected_session(core: &RuntimeCore, session_id: &str, thread_id: &str) {
    seed_projected_session_with_working_dir(core, session_id, thread_id, "/tmp/projection").await;
}

async fn seed_projected_session_with_working_dir(
    core: &RuntimeCore,
    session_id: &str,
    thread_id: &str,
    working_dir: &str,
) {
    core.start_session(AgentSessionStartParams {
        session_id: Some(session_id.to_string()),
        thread_id: Some(thread_id.to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: Some(app_server_protocol::BusinessObjectRef {
            kind: "agent.session".to_string(),
            id: session_id.to_string(),
            title: Some("Projection Title".to_string()),
            uri: None,
            metadata: Some(json!({
                "title": "Projection Title",
                "modelName": "projection-model",
                "workingDir": working_dir,
                "executionStrategy": "projection",
                "providerSelector": "fixture-provider",
                "recentAccessMode": "read-only",
            })),
        }),
        locale: None,
    })
    .expect("session");

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: session_id.to_string(),
            turn_id: Some(format!("{thread_id}_turn")),
            input: AgentInput {
                text: "Projection DB 应该是列表事实源".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("turn");
}

async fn seed_projected_running_session(core: &RuntimeCore, session_id: &str, thread_id: &str) {
    core.start_session(AgentSessionStartParams {
        session_id: Some(session_id.to_string()),
        thread_id: Some(thread_id.to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: Some(app_server_protocol::BusinessObjectRef {
            kind: "agent.session".to_string(),
            id: session_id.to_string(),
            title: Some("Running Projection".to_string()),
            uri: None,
            metadata: Some(json!({
                "title": "Running Projection",
                "modelName": "projection-model",
            })),
        }),
        locale: None,
    })
    .expect("session");

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: session_id.to_string(),
            turn_id: Some(format!("{thread_id}_turn")),
            input: AgentInput {
                text: "这轮还在运行".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("turn");
}

fn projected_event(
    sequence: u64,
    session_id: &str,
    thread_id: &str,
    turn_id: Option<&str>,
    event_type: &str,
    timestamp: &str,
    payload: serde_json::Value,
) -> AgentEvent {
    AgentEvent {
        event_id: format!("{session_id}-{sequence}"),
        sequence,
        session_id: session_id.to_string(),
        thread_id: Some(thread_id.to_string()),
        turn_id: turn_id.map(ToString::to_string),
        event_type: event_type.to_string(),
        timestamp: timestamp.to_string(),
        payload,
    }
}

#[tokio::test]
async fn list_agent_sessions_reads_projection_as_current_truth() {
    let harness = projection_test_core();
    seed_projected_session(
        &harness.core,
        "sess_projection_wins",
        "thread_projection_wins",
    )
    .await;

    let restarted_core = RuntimeCore::default()
        .with_event_log_writer(harness.event_log_writer)
        .with_projection_store(harness.projection_store);

    let listed = restarted_core
        .list_agent_sessions(AgentSessionListParams {
            workspace_id: Some("workspace-current".to_string()),
            limit: Some(20),
            ..AgentSessionListParams::default()
        })
        .await
        .expect("list sessions");

    assert_eq!(listed.sessions.len(), 1);
    let projected = listed
        .sessions
        .iter()
        .find(|session| session.session_id == "sess_projection_wins")
        .expect("projected session");
    assert_eq!(
        projected.thread_id.as_deref(),
        Some("thread_projection_wins")
    );
    assert_eq!(projected.title.as_deref(), Some("Projection Title"));
    assert_eq!(projected.model, "projection-model");
    assert_eq!(projected.working_dir.as_deref(), Some("/tmp/projection"));
    assert_eq!(projected.execution_strategy.as_deref(), Some("projection"));
    assert_eq!(projected.messages_count, 2);
    assert_eq!(projected.thread_status, "completed");
    assert_eq!(projected.latest_turn_status.as_deref(), Some("completed"));
    assert_eq!(projected.active_turn_id, None);
    assert_eq!(projected.queued_turn_count, 0);
}

#[tokio::test]
async fn list_agent_sessions_downgrades_stale_orphan_running_turn() {
    let harness = projection_test_core();
    let session_id = "sess_projection_orphan_running";
    let thread_id = "thread_projection_orphan_running";
    let turn_id = "turn_projection_orphan_running";
    harness
        .projection_store
        .apply_events(&[
            projected_event(
                1,
                session_id,
                thread_id,
                None,
                "session.started",
                "2026-03-29T00:00:00.000Z",
                json!({
                    "session": {
                        "title": "Stale Running Projection",
                        "workspaceId": "workspace-current",
                        "metadata": {
                            "title": "Stale Running Projection",
                            "modelName": "projection-model"
                        }
                    }
                }),
            ),
            projected_event(
                2,
                session_id,
                thread_id,
                Some(turn_id),
                "turn.started",
                "2026-03-29T00:00:01.000Z",
                json!({}),
            ),
        ])
        .expect("apply stale projection events");

    let listed = harness
        .projection_store
        .list_session_overviews(&AgentSessionListParams {
            workspace_id: Some("workspace-current".to_string()),
            limit: Some(20),
            ..AgentSessionListParams::default()
        })
        .expect("list sessions");

    let projected = listed
        .iter()
        .find(|session| session.session_id == session_id)
        .expect("projected stale running session");
    assert_eq!(projected.thread_status, "idle");
    assert_eq!(projected.latest_turn_status.as_deref(), Some("canceled"));
    assert_eq!(projected.active_turn_id, None);
    assert_eq!(projected.queued_turn_count, 0);
}

#[tokio::test]
async fn list_agent_sessions_interrupts_recent_running_projection_after_restart() {
    let harness = projection_running_test_core();
    seed_projected_running_session(
        &harness.core,
        "sess_projection_running",
        "thread_projection_running",
    )
    .await;

    let restarted_core = RuntimeCore::default()
        .with_event_log_writer(harness.event_log_writer)
        .with_projection_store(harness.projection_store);

    let listed = restarted_core
        .list_agent_sessions(AgentSessionListParams {
            workspace_id: Some("workspace-current".to_string()),
            limit: Some(20),
            ..AgentSessionListParams::default()
        })
        .await
        .expect("list sessions");

    let projected = listed
        .sessions
        .iter()
        .find(|session| session.session_id == "sess_projection_running")
        .expect("projected running session");
    assert_eq!(projected.thread_status, "idle");
    assert_eq!(projected.latest_turn_status.as_deref(), Some("canceled"));
    assert_eq!(projected.active_turn_id, None);
    assert_eq!(projected.queued_turn_count, 0);
}

#[tokio::test]
async fn list_agent_sessions_filters_projection_by_cwd() {
    let harness = projection_test_core();
    seed_projected_session_with_working_dir(
        &harness.core,
        "sess_projection_cwd_a",
        "thread_projection_cwd_a",
        "/tmp/projection-a",
    )
    .await;
    seed_projected_session_with_working_dir(
        &harness.core,
        "sess_projection_cwd_b",
        "thread_projection_cwd_b",
        "/tmp/projection-b",
    )
    .await;

    let restarted_core = RuntimeCore::default()
        .with_event_log_writer(harness.event_log_writer)
        .with_projection_store(harness.projection_store);

    let listed = restarted_core
        .list_agent_sessions(AgentSessionListParams {
            cwd: Some(AgentSessionCwdFilter::One("/tmp/projection-a".to_string())),
            limit: Some(20),
            ..AgentSessionListParams::default()
        })
        .await
        .expect("list sessions");

    assert!(listed
        .sessions
        .iter()
        .any(|session| session.session_id == "sess_projection_cwd_a"));
    assert!(!listed
        .sessions
        .iter()
        .any(|session| session.session_id == "sess_projection_cwd_b"));
}

#[tokio::test]
async fn list_agent_sessions_filters_projection_by_workspace_or_workspace_root() {
    let harness = projection_test_core();
    seed_projected_session_with_working_dir(
        &harness.core,
        "sess_projection_workspace_only",
        "thread_projection_workspace_only",
        "",
    )
    .await;
    seed_projected_session_with_working_dir(
        &harness.core,
        "sess_projection_cwd_only",
        "thread_projection_cwd_only",
        "/tmp/projection-root",
    )
    .await;

    let app_data_source = Arc::new(TestSessionDataSource::new().with_workspace(json!({
        "id": "workspace-current",
        "rootPath": "/tmp/projection-root",
    })));
    let restarted_core = RuntimeCore::default()
        .with_event_log_writer(harness.event_log_writer)
        .with_projection_store(harness.projection_store)
        .with_app_data_source(app_data_source);

    let listed = restarted_core
        .list_agent_sessions(AgentSessionListParams {
            workspace_id: Some("workspace-current".to_string()),
            limit: Some(20),
            ..AgentSessionListParams::default()
        })
        .await
        .expect("list sessions");
    let ids = listed
        .sessions
        .iter()
        .map(|session| session.session_id.as_str())
        .collect::<Vec<_>>();

    assert!(ids.contains(&"sess_projection_workspace_only"));
    assert!(ids.contains(&"sess_projection_cwd_only"));
}
