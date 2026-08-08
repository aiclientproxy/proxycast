use super::support::*;
use super::*;
use agent_protocol::{SortDirection, ThreadId, ThreadStatus, ThreadTurnsView, TurnItemsView};
use app_server_protocol::protocol::v2::{ThreadCompactStartParams, ThreadCompactStartResponse};
use futures::executor::block_on;
use thread_store::{
    ListItemsParams, ListThreadsParams, ListTurnsParams, PageRequest, ReadThreadParams, ThreadStore,
};

fn canonical_page() -> PageRequest {
    PageRequest {
        cursor: None,
        limit: 20,
        sort_direction: SortDirection::Asc,
    }
}

fn empty_thread_session_params(session_id: &str, thread_id: &str) -> AgentSessionStartParams {
    AgentSessionStartParams {
        session_id: Some(session_id.to_string()),
        thread_id: Some(thread_id.to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    }
}

#[derive(Default)]
struct CloseSessionRecordingBackend {
    close_requests: Mutex<Vec<(String, String)>>,
}

#[async_trait]
impl ExecutionBackend for CloseSessionRecordingBackend {
    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn cancel_turn(
        &self,
        _request: CancelExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn close_session(
        &self,
        session_id: &str,
        thread_id: &str,
    ) -> Result<(), RuntimeCoreError> {
        self.close_requests
            .lock()
            .expect("close session requests mutex poisoned")
            .push((session_id.to_string(), thread_id.to_string()));
        Ok(())
    }

    async fn respond_action(
        &self,
        _request: ActionRespondRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }
}

#[test]
fn start_session_creates_empty_canonical_thread_before_returning() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store"),
    );
    let core = RuntimeCore::default().with_projection_store(store.clone());

    let response = core
        .start_session(empty_thread_session_params(
            "sess_empty_canonical",
            "thread_empty_canonical",
        ))
        .expect("start session");

    let session_read = block_on(core.read_session_current(AgentSessionReadParams {
        session_id: "sess_empty_canonical".to_string(),
        history_limit: None,
        history_offset: None,
        history_before_message_id: None,
    }))
    .expect("read empty session through current boundary");
    assert!(session_read.turns.is_empty());
    assert!(session_read
        .detail
        .as_ref()
        .and_then(|detail| detail["items"].as_array())
        .is_some_and(Vec::is_empty));

    let thread = block_on(store.read_thread(ReadThreadParams {
        thread_id: ThreadId::new("thread_empty_canonical"),
        include_archived: false,
        turns_view: ThreadTurnsView::Full,
    }))
    .expect("read canonical thread")
    .expect("canonical thread exists");
    assert_eq!(thread.session_id.as_str(), response.session.session_id);
    assert_eq!(thread.thread_id.as_str(), response.session.thread_id);
    assert_eq!(thread.status, ThreadStatus::Idle);
    assert!(thread.turns.is_empty());

    let threads = block_on(store.list_threads(ListThreadsParams {
        include_archived: false,
        section: None,
        sort_by_section_position: false,
        page: canonical_page(),
    }))
    .expect("list canonical threads");
    assert_eq!(threads.data.len(), 1);
    assert_eq!(threads.data[0].thread_id, thread.thread_id);
    assert!(block_on(store.list_turns(ListTurnsParams {
        thread_id: thread.thread_id.clone(),
        include_archived: false,
        page: canonical_page(),
        items_view: TurnItemsView::NotLoaded,
    }))
    .expect("list canonical turns")
    .data
    .is_empty());
    assert!(block_on(store.list_items(ListItemsParams {
        thread_id: thread.thread_id,
        turn_id: None,
        include_archived: false,
        page: canonical_page(),
    }))
    .expect("list canonical items")
    .data
    .is_empty());
}

#[test]
fn start_session_rejects_duplicate_session_and_thread_identity_atomically() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store"),
    );
    let core = RuntimeCore::default().with_projection_store(store.clone());
    core.start_session(empty_thread_session_params(
        "sess_identity_owner",
        "thread_identity_owner",
    ))
    .expect("first session");

    let duplicate_session = core
        .start_session(empty_thread_session_params(
            "sess_identity_owner",
            "thread_duplicate_session",
        ))
        .expect_err("duplicate session must fail");
    assert!(matches!(
        duplicate_session,
        RuntimeCoreError::SessionAlreadyExists(ref id) if id == "sess_identity_owner"
    ));
    assert!(block_on(store.read_thread(ReadThreadParams {
        thread_id: ThreadId::new("thread_duplicate_session"),
        include_archived: false,
        turns_view: ThreadTurnsView::NotLoaded,
    }))
    .expect("read duplicate session thread")
    .is_none());

    let duplicate_thread = core
        .start_session(empty_thread_session_params(
            "sess_duplicate_thread",
            "thread_identity_owner",
        ))
        .expect_err("duplicate thread identity must fail");
    assert!(matches!(duplicate_thread, RuntimeCoreError::Backend(_)));
    assert!(matches!(
        core.read_session(AgentSessionReadParams {
            session_id: "sess_duplicate_thread".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        }),
        Err(RuntimeCoreError::SessionNotFound(ref id)) if id == "sess_duplicate_thread"
    ));

    let restarted_core = RuntimeCore::default().with_projection_store(store.clone());
    assert!(matches!(
        restarted_core.start_session(empty_thread_session_params(
            "sess_identity_owner",
            "thread_duplicate_persisted_session",
        )),
        Err(RuntimeCoreError::Backend(_))
    ));
    assert!(block_on(store.read_thread(ReadThreadParams {
        thread_id: ThreadId::new("thread_duplicate_persisted_session"),
        include_archived: false,
        turns_view: ThreadTurnsView::NotLoaded,
    }))
    .expect("read persisted duplicate session thread")
    .is_none());
}

#[test]
fn start_session_store_failure_leaves_no_memory_session_and_allows_retry() {
    let temp = tempfile::tempdir().expect("tempdir");
    let database_path = temp.path().join("projection.sqlite");
    let backup_path = temp.path().join("projection.sqlite.backup");
    let blocked_path = temp.path().join("projection.sqlite.blocked");
    let store = Arc::new(ProjectionStore::initialize(&database_path).expect("projection store"));
    let core = RuntimeCore::default().with_projection_store(store.clone());
    std::fs::rename(&database_path, &backup_path).expect("move database aside");
    std::fs::create_dir(&database_path).expect("block database path");

    let params = empty_thread_session_params("sess_store_retry", "thread_store_retry");
    assert!(matches!(
        core.start_session(params.clone()),
        Err(RuntimeCoreError::Backend(_))
    ));
    assert!(matches!(
        core.read_session(AgentSessionReadParams {
            session_id: "sess_store_retry".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        }),
        Err(RuntimeCoreError::SessionNotFound(ref id)) if id == "sess_store_retry"
    ));

    std::fs::rename(&database_path, &blocked_path).expect("unblock database path");
    std::fs::rename(&backup_path, &database_path).expect("restore database");
    core.start_session(params).expect("retry session start");
    assert!(block_on(store.read_thread(ReadThreadParams {
        thread_id: ThreadId::new("thread_store_retry"),
        include_archived: false,
        turns_view: ThreadTurnsView::NotLoaded,
    }))
    .expect("read retried canonical thread")
    .is_some());
}

#[tokio::test]
async fn delete_thread_removes_empty_canonical_thread_and_allows_recreate() {
    let temp = tempfile::tempdir().expect("tempdir");
    let database_path = temp.path().join("projection.sqlite");
    let backup_path = temp.path().join("projection.sqlite.backup");
    let blocked_path = temp.path().join("projection.sqlite.blocked");
    let store = Arc::new(ProjectionStore::initialize(&database_path).expect("projection store"));
    let core = RuntimeCore::default().with_projection_store(store.clone());
    let params = empty_thread_session_params("sess_empty_delete", "thread_empty_delete");
    core.start_session(params.clone()).expect("start session");

    std::fs::rename(&database_path, &backup_path).expect("move database aside");
    std::fs::create_dir(&database_path).expect("block database path");
    assert!(matches!(
        core.delete_thread(ThreadId::new("thread_empty_delete"))
            .await,
        Err(RuntimeCoreError::Backend(_))
    ));
    std::fs::rename(&database_path, &blocked_path).expect("unblock database path");
    std::fs::rename(&backup_path, &database_path).expect("restore database");
    core.read_session(AgentSessionReadParams {
        session_id: "sess_empty_delete".to_string(),
        history_limit: None,
        history_offset: None,
        history_before_message_id: None,
    })
    .expect("failed delete keeps memory session");

    let deleted = core
        .delete_thread(ThreadId::new("thread_empty_delete"))
        .await
        .expect("delete thread");
    assert_eq!(deleted.len(), 1);
    assert!(block_on(store.read_thread(ReadThreadParams {
        thread_id: ThreadId::new("thread_empty_delete"),
        include_archived: false,
        turns_view: ThreadTurnsView::NotLoaded,
    }))
    .expect("read deleted canonical thread")
    .is_none());

    core.start_session(params).expect("recreate session");
    assert!(block_on(store.read_thread(ReadThreadParams {
        thread_id: ThreadId::new("thread_empty_delete"),
        include_archived: false,
        turns_view: ThreadTurnsView::NotLoaded,
    }))
    .expect("read recreated canonical thread")
    .is_some());
}

#[tokio::test]
async fn delete_thread_closes_exact_runtime_owner_before_removal() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store"),
    );
    let backend = Arc::new(CloseSessionRecordingBackend::default());
    let core = RuntimeCore::with_backend(backend.clone()).with_projection_store(store);
    core.start_session(empty_thread_session_params(
        "sess_close_runtime",
        "thread_close_runtime",
    ))
    .expect("start session");

    let deleted = core
        .delete_thread(ThreadId::new("thread_close_runtime"))
        .await
        .expect("delete thread");

    assert_eq!(deleted.len(), 1);
    assert_eq!(
        *backend
            .close_requests
            .lock()
            .expect("close session requests mutex poisoned"),
        vec![(
            "sess_close_runtime".to_string(),
            "thread_close_runtime".to_string()
        )]
    );
}

#[tokio::test]
async fn compact_thread_writes_session_context_artifact() {
    let sidecar_root = tempfile::tempdir().expect("sidecar root");
    let sidecar_store = Arc::new(SidecarStore::new(sidecar_root.path()).expect("sidecar store"));
    let projection_store = Arc::new(
        ProjectionStore::initialize(sidecar_root.path().join("projection.sqlite"))
            .expect("projection store"),
    );
    let core = RuntimeCore::with_backend(Arc::new(CompletedBackend))
        .with_sidecar_store(sidecar_store.clone())
        .with_projection_store(projection_store);
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_compact".to_string()),
        thread_id: Some("thread_compact".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_compact".to_string(),
            turn_id: Some("turn_compact_1".to_string()),
            input: AgentInput {
                text: "请总结上下文".to_string(),
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

    let mut runtime_events = core.take_event_receiver().expect("runtime event receiver");
    let output = core
        .compact_thread(ThreadCompactStartParams {
            thread_id: "thread_compact".to_string(),
        })
        .await
        .expect("compact");

    assert_eq!(output.response, ThreadCompactStartResponse {});
    assert!(output.events.is_empty());
    let completed =
        wait_for_runtime_event(&mut runtime_events, "context.compaction.completed").await;
    assert_eq!(completed.payload["contextEpoch"].as_u64(), Some(1));
    assert_eq!(
        completed.payload["tailStartTurnId"].as_str(),
        Some("turn_compact_1")
    );
    assert_eq!(
        completed.payload["artifact"]["policy"]["durableHistoryRewrite"].as_bool(),
        Some(false)
    );
    assert_eq!(
        completed.payload["artifact"]["policy"]["providerHistoryRewrite"].as_bool(),
        Some(true)
    );
    assert_eq!(
        completed.payload["artifact"]["policy"]["longTermMemoryWrite"].as_bool(),
        Some(false)
    );
    let relative_path = completed.payload["sidecarRef"]["relativePath"]
        .as_str()
        .expect("sidecar relative path");
    let sidecar = sidecar_store
        .read_text(relative_path)
        .expect("sidecar content");
    assert!(sidecar.contains("\"schema\": \"session_context_compaction.v2\""));
    assert!(sidecar.contains("No earlier turns were removed"));
    assert!(!sidecar.contains("请总结上下文"));

    let read = core
        .read_session(AgentSessionReadParams {
            session_id: "sess_compact".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read session");
    let detail = read.detail.expect("detail");
    let items = detail["items"].as_array().expect("items");
    assert!(items.iter().any(|item| {
        item["type"] == "context_compaction"
            && item["id"] == completed.payload["compactionId"]
            && item["status"] == "completed"
    }));
}

#[tokio::test]
async fn compact_thread_uses_replacement_history_without_duplicate_prompt_context() {
    let sidecar_root = tempfile::tempdir().expect("sidecar root");
    let sidecar_store = Arc::new(SidecarStore::new(sidecar_root.path()).expect("sidecar store"));
    let backend = Arc::new(TurnCompletedRecordingBackend {
        requests: Mutex::new(Vec::new()),
    });
    let projection_store = Arc::new(
        ProjectionStore::initialize(sidecar_root.path().join("projection.sqlite"))
            .expect("projection store"),
    );
    let core = RuntimeCore::with_backend(backend.clone())
        .with_sidecar_store(sidecar_store.clone())
        .with_projection_store(projection_store);
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_compact_next_turn".to_string()),
        thread_id: Some("thread_compact_next_turn".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_compact_next_turn".to_string(),
            turn_id: Some("turn_compact_next_1".to_string()),
            input: AgentInput {
                text: "第一轮需要保留的事实".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("first turn");
    let mut runtime_events = core.take_event_receiver().expect("runtime event receiver");
    core.compact_thread(ThreadCompactStartParams {
        thread_id: "thread_compact_next_turn".to_string(),
    })
    .await
    .expect("compact");
    wait_for_runtime_event(&mut runtime_events, "context.compaction.completed").await;

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_compact_next_turn".to_string(),
            turn_id: Some("turn_compact_next_2".to_string()),
            input: AgentInput {
                text: "继续".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: Some(RuntimeOptions {
                runtime_request: Some(app_server_protocol::RuntimeRequest {
                    metadata: Some(json!({
                    "system_prompt": "base prompt",
                    "harness": {
                        "model_request_policy": {
                            "context_policy": {
                                "context_window": 8000
                            }
                        }
                    }
                    })),
                    ..app_server_protocol::RuntimeRequest::default()
                }),
                ..RuntimeOptions::default()
            }),
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("second turn");

    let requests = backend
        .requests
        .lock()
        .expect("test backend requests mutex poisoned");
    assert_eq!(requests.len(), 2);
    let metadata = requests[1]
        .runtime_options
        .as_ref()
        .and_then(app_server_protocol::RuntimeOptions::runtime_metadata)
        .expect("runtime metadata");
    assert!(
        metadata
            .get(crate::runtime::memory_prompt::SESSION_COMPACTION_PROMPT_CONTEXT_KEY)
            .is_none(),
        "replacement history must own the compaction summary"
    );
    assert!(
        metadata
            .get(crate::runtime::memory_prompt::CONTEXT_PACKET_TELEMETRY_KEY)
            .is_none(),
        "skipped prompt context must not emit packet telemetry"
    );

    let prompt = crate::runtime::memory_prompt::append_memory_context_to_system_prompt(
        Some("base prompt".to_string()),
        Some(metadata),
    )
    .expect("system prompt");
    assert_eq!(prompt, "base prompt");
}

#[tokio::test]
async fn list_agent_sessions_projects_runtime_core_sessions_only() {
    let core = RuntimeCore::default();
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_old".to_string()),
        thread_id: Some("thread_old".to_string()),
        app_id: "content-studio".to_string(),
        workspace_id: Some("workspace-old".to_string()),
        business_object_ref: Some(app_server_protocol::BusinessObjectRef {
            kind: "project".to_string(),
            id: "old".to_string(),
            title: Some("Old Workspace Session".to_string()),
            uri: None,
            metadata: Some(json!({
                "model": "gpt-test",
                "workingDir": "/tmp/old",
                "executionStrategy": "runtime-core"
            })),
        }),
        locale: None,
    })
    .expect("old workspace session");
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_current".to_string()),
        thread_id: Some("thread_current".to_string()),
        app_id: "content-studio".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: Some(app_server_protocol::BusinessObjectRef {
            kind: "project".to_string(),
            id: "current".to_string(),
            title: Some("Current Workspace Session".to_string()),
            uri: None,
            metadata: Some(json!({
                "modelName": "claude-test",
                "working_dir": "/tmp/current",
                "execution_strategy": "runtime-core"
            })),
        }),
        locale: None,
    })
    .expect("current workspace session");

    let response = core
        .list_agent_sessions(AgentSessionListParams {
            workspace_id: Some("workspace-current".to_string()),
            limit: Some(1),
            ..AgentSessionListParams::default()
        })
        .await
        .expect("list sessions");

    assert_eq!(response.sessions.len(), 1);
    assert_eq!(response.sessions[0].session_id, "sess_current");
    assert_eq!(
        response.sessions[0].thread_id.as_deref(),
        Some("thread_current")
    );
    assert_eq!(
        response.sessions[0].title.as_deref(),
        Some("Current Workspace Session")
    );
    assert_eq!(response.sessions[0].model, "claude-test");
    assert_eq!(
        response.sessions[0].working_dir.as_deref(),
        Some("/tmp/current")
    );
    assert_eq!(
        response.sessions[0].execution_strategy.as_deref(),
        Some("runtime-core")
    );
}

#[tokio::test]
async fn list_agent_sessions_keeps_workspace_id_filter_when_workspace_root_is_available() {
    let app_data_source = Arc::new(TestSessionDataSource::new().with_workspace(json!({
        "id": "workspace-current",
        "rootPath": "/tmp/current",
    })));
    let core = RuntimeCore::default().with_app_data_source(app_data_source);
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_workspace_only".to_string()),
        thread_id: Some("thread_workspace_only".to_string()),
        app_id: "desktop".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: Some(app_server_protocol::BusinessObjectRef {
            kind: "agent.session".to_string(),
            id: "sess_workspace_only".to_string(),
            title: Some("Workspace Only Session".to_string()),
            uri: None,
            metadata: Some(json!({
                "modelName": "fixture-model",
                "executionStrategy": "react"
            })),
        }),
        locale: None,
    })
    .expect("workspace-only session");
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_cwd_only".to_string()),
        thread_id: Some("thread_cwd_only".to_string()),
        app_id: "desktop".to_string(),
        workspace_id: None,
        business_object_ref: Some(app_server_protocol::BusinessObjectRef {
            kind: "agent.session".to_string(),
            id: "sess_cwd_only".to_string(),
            title: Some("Cwd Only Session".to_string()),
            uri: None,
            metadata: Some(json!({
                "workingDir": "/tmp/current",
                "modelName": "fixture-model",
                "executionStrategy": "react"
            })),
        }),
        locale: None,
    })
    .expect("cwd-only session");

    let response = core
        .list_agent_sessions(AgentSessionListParams {
            workspace_id: Some("workspace-current".to_string()),
            limit: Some(20),
            ..AgentSessionListParams::default()
        })
        .await
        .expect("list sessions");
    let ids = response
        .sessions
        .iter()
        .map(|session| session.session_id.as_str())
        .collect::<Vec<_>>();

    assert!(ids.contains(&"sess_workspace_only"));
    assert!(ids.contains(&"sess_cwd_only"));
}

#[tokio::test]
async fn list_agent_sessions_derives_placeholder_title_from_first_user_message() {
    let core = RuntimeCore::default();
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_auto_title".to_string()),
        thread_id: Some("thread_auto_title".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: Some(app_server_protocol::BusinessObjectRef {
            kind: "agent.session".to_string(),
            id: "sess_auto_title".to_string(),
            title: Some("新对话".to_string()),
            uri: None,
            metadata: Some(json!({ "title": "新对话" })),
        }),
        locale: None,
    })
    .expect("session");

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_auto_title".to_string(),
            turn_id: Some("turn_auto_title".to_string()),
            input: AgentInput {
                text: "整理今天的国际新闻".to_string(),
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

    let listed = core
        .list_agent_sessions(AgentSessionListParams {
            workspace_id: Some("workspace-current".to_string()),
            limit: Some(20),
            ..AgentSessionListParams::default()
        })
        .await
        .expect("list sessions");

    assert_eq!(listed.sessions.len(), 1);
    assert_eq!(
        listed.sessions[0].title.as_deref(),
        Some("整理今天的国际新闻")
    );
}

#[tokio::test]
async fn list_agent_sessions_preserves_explicit_title_when_user_message_exists() {
    let core = RuntimeCore::default();
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_manual_title".to_string()),
        thread_id: Some("thread_manual_title".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: Some(app_server_protocol::BusinessObjectRef {
            kind: "agent.session".to_string(),
            id: "sess_manual_title".to_string(),
            title: Some("手动标题".to_string()),
            uri: None,
            metadata: Some(json!({ "title": "手动标题" })),
        }),
        locale: None,
    })
    .expect("session");

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_manual_title".to_string(),
            turn_id: Some("turn_manual_title".to_string()),
            input: AgentInput {
                text: "不要覆盖我".to_string(),
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

    let listed = core
        .list_agent_sessions(AgentSessionListParams {
            workspace_id: Some("workspace-current".to_string()),
            limit: Some(20),
            ..AgentSessionListParams::default()
        })
        .await
        .expect("list sessions");

    assert_eq!(listed.sessions.len(), 1);
    assert_eq!(listed.sessions[0].title.as_deref(), Some("手动标题"));
}

#[tokio::test]
async fn list_agent_sessions_excludes_hidden_runtime_core_sessions() {
    let core = RuntimeCore::default();
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_hidden".to_string()),
        thread_id: Some("thread_hidden".to_string()),
        app_id: "desktop".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: Some(app_server_protocol::BusinessObjectRef {
            kind: "agent.session".to_string(),
            id: "hidden".to_string(),
            title: Some("Internal Smoke Session".to_string()),
            uri: None,
            metadata: Some(json!({
                "harness": {
                    "hiddenFromUserRecents": true,
                    "source": "unit"
                },
                "model": "gpt-test"
            })),
        }),
        locale: None,
    })
    .expect("hidden session");
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_visible".to_string()),
        thread_id: Some("thread_visible".to_string()),
        app_id: "desktop".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: Some(app_server_protocol::BusinessObjectRef {
            kind: "agent.session".to_string(),
            id: "visible".to_string(),
            title: Some("Visible Session".to_string()),
            uri: None,
            metadata: Some(json!({
                "model": "gpt-test"
            })),
        }),
        locale: None,
    })
    .expect("visible session");

    let response = core
        .list_agent_sessions(AgentSessionListParams {
            workspace_id: Some("workspace-current".to_string()),
            limit: Some(20),
            ..AgentSessionListParams::default()
        })
        .await
        .expect("list sessions");

    let ids = response
        .sessions
        .iter()
        .map(|session| session.session_id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(ids, vec!["sess_visible"]);

    let hidden = core
        .read_session_current(AgentSessionReadParams {
            session_id: "sess_hidden".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .await
        .expect("hidden session remains readable by id");
    assert_eq!(hidden.session.session_id, "sess_hidden");
}

#[tokio::test]
async fn read_session_current_does_not_fallback_to_persistent_history() {
    let core = RuntimeCore::default();
    let error = core
        .read_session_current(AgentSessionReadParams {
            session_id: "missing_legacy_session".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .await
        .expect_err("missing session should fail closed");

    assert_eq!(
        error.into_jsonrpc_error().code,
        error_codes::SESSION_NOT_FOUND
    );
}

#[tokio::test]
async fn read_session_current_repairs_and_reads_jsonl_projection() {
    let temp = tempfile::tempdir().expect("tempdir");
    let roots =
        StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
    let event_log_writer = Arc::new(EventLogWriter::new(&roots.event_log_root).expect("writer"));
    let projection_store =
        Arc::new(ProjectionStore::initialize(&roots.projection_db_path).expect("projection"));
    let core = RuntimeCore::with_backend(Arc::new(CompletedBackend))
        .with_event_log_writer(event_log_writer.clone())
        .with_projection_store(projection_store.clone());
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_projection_read".to_string()),
        thread_id: Some("thread_projection_read".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_projection_read".to_string(),
            turn_id: Some("turn_projection_read".to_string()),
            input: AgentInput {
                text: "hello".to_string(),
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
    projection_store
        .clear_session("sess_projection_read")
        .expect("simulate missing projection");

    let app_data_source = Arc::new(TestSessionDataSource::new());
    let restarted_core = RuntimeCore::default()
        .with_event_log_writer(event_log_writer)
        .with_projection_store(projection_store.clone())
        .with_app_data_source(app_data_source);

    let read = restarted_core
        .read_session_current(AgentSessionReadParams {
            session_id: "sess_projection_read".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .await
        .expect("read from projection");

    assert_eq!(read.session.session_id, "sess_projection_read");
    assert_eq!(read.session.thread_id, "thread_projection_read");
    assert_eq!(read.session.status, AgentSessionStatus::Completed);
    assert_eq!(read.turns.len(), 1);
    assert_eq!(read.turns[0].turn_id, "turn_projection_read");
    assert_eq!(read.turns[0].status, AgentTurnStatus::Completed);
    let detail = read.detail.expect("projection detail");
    assert_eq!(detail["projection_source"], "runtime.projection_1");
    assert_eq!(detail["thread_read"]["status"].as_str(), Some("completed"));
    let projected = projection_store
        .read_session("sess_projection_read")
        .expect("read repaired projection")
        .expect("projection row");
    assert_eq!(projected.last_event_sequence, 8);
}

#[tokio::test]
async fn read_session_respects_history_limit_for_runtime_core_session() {
    let core = RuntimeCore::with_backend(Arc::new(CompletedBackend));
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_runtime_history_limit".to_string()),
        thread_id: Some("thread_runtime_history_limit".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    for index in 0..3 {
        core.start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_runtime_history_limit".to_string(),
                turn_id: Some(format!("turn_runtime_history_limit_{index}")),
                input: AgentInput {
                    text: format!("hello {index}"),
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

    let read = core
        .read_session(AgentSessionReadParams {
            session_id: "sess_runtime_history_limit".to_string(),
            history_limit: Some(2),
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read");
    let detail = read.detail.expect("detail");

    assert_eq!(detail["messages_count"].as_u64(), Some(6));
    assert_eq!(detail["history_limit"].as_u64(), Some(2));
    assert_eq!(detail["history_cursor"]["loaded_count"].as_u64(), Some(2));
    assert_eq!(detail["history_truncated"].as_bool(), Some(true));
    assert_eq!(detail["messages"].as_array().expect("messages").len(), 2);
}

#[tokio::test]
async fn read_session_current_uses_projection_summary_for_limited_history() {
    let temp = tempfile::tempdir().expect("tempdir");
    let roots =
        StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
    let event_log_writer = Arc::new(EventLogWriter::new(&roots.event_log_root).expect("writer"));
    let projection_store =
        Arc::new(ProjectionStore::initialize(&roots.projection_db_path).expect("projection"));
    let core = RuntimeCore::with_backend(Arc::new(CompletedBackend))
        .with_event_log_writer(event_log_writer.clone())
        .with_projection_store(projection_store.clone());
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_projection_fast_read".to_string()),
        thread_id: Some("thread_projection_fast_read".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_projection_fast_read".to_string(),
            turn_id: Some("turn_projection_fast_read_1".to_string()),
            input: AgentInput {
                text: "first".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("first turn");
    let watermark_before = projection_store
        .read_watermark("sess_projection_fast_read")
        .expect("watermark")
        .expect("watermark")
        .last_sequence;
    let extra_event = AgentEvent {
        event_id: "evt_projection_fast_read_extra".to_string(),
        sequence: watermark_before + 1,
        session_id: "sess_projection_fast_read".to_string(),
        thread_id: Some("thread_projection_fast_read".to_string()),
        turn_id: Some("turn_projection_fast_read_extra".to_string()),
        event_type: "message.created".to_string(),
        timestamp: timestamp(),
        payload: json!({
            "input": [{
                "type": "text",
                "text": "current projection should support limited read"
            }]
        }),
    };
    event_log_writer
        .append(&extra_event)
        .expect("append projection event");
    projection_store
        .apply_event(&extra_event)
        .expect("apply current projection event");

    let restarted_core = RuntimeCore::default()
        .with_event_log_writer(event_log_writer)
        .with_projection_store(projection_store.clone());
    let read = restarted_core
        .read_session_current(AgentSessionReadParams {
            session_id: "sess_projection_fast_read".to_string(),
            history_limit: Some(1),
            history_offset: None,
            history_before_message_id: None,
        })
        .await
        .expect("read from projection summary");
    let detail = read.detail.expect("detail");
    let watermark_after = projection_store
        .read_watermark("sess_projection_fast_read")
        .expect("watermark after")
        .expect("watermark after")
        .last_sequence;

    assert_eq!(watermark_after, watermark_before + 1);
    assert_eq!(detail["projection_source"], "runtime.projection_1");
    assert_eq!(detail["messages_count"].as_u64(), Some(3));
    assert_eq!(detail["history_cursor"]["loaded_count"].as_u64(), Some(1));
    assert_eq!(
        detail["messages"][0]["metadata"]["source"].as_str(),
        Some("projection_summary")
    );
    assert!(detail["messages"][0]["timestamp"].as_f64().is_some());
}

#[tokio::test]
async fn read_session_current_projection_summary_projects_turn_usage() {
    let temp = tempfile::tempdir().expect("tempdir");
    let roots =
        StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
    let event_log_writer = Arc::new(EventLogWriter::new(&roots.event_log_root).expect("writer"));
    let projection_store =
        Arc::new(ProjectionStore::initialize(&roots.projection_db_path).expect("projection"));
    let core = RuntimeCore::with_backend(Arc::new(CompletedBackend))
        .with_event_log_writer(event_log_writer.clone())
        .with_projection_store(projection_store.clone());
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_projection_usage_read".to_string()),
        thread_id: Some("thread_projection_usage_read".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_projection_usage_read".to_string(),
            turn_id: Some("turn_projection_usage_read".to_string()),
            input: AgentInput {
                text: "@配图 画一张深圳夏天的图".to_string(),
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
    let watermark_before = projection_store
        .read_watermark("sess_projection_usage_read")
        .expect("watermark")
        .expect("watermark")
        .last_sequence;
    let usage_event = AgentEvent {
        event_id: "evt_projection_usage_terminal".to_string(),
        sequence: watermark_before + 1,
        session_id: "sess_projection_usage_read".to_string(),
        thread_id: Some("thread_projection_usage_read".to_string()),
        turn_id: Some("turn_projection_usage_read".to_string()),
        event_type: "turn.completed".to_string(),
        timestamp: timestamp(),
        payload: json!({
            "usage": {
                "input_tokens": 1175,
                "output_tokens": 112,
                "cached_input_tokens": 0
            }
        }),
    };
    event_log_writer
        .append(&usage_event)
        .expect("append usage terminal event");
    projection_store
        .apply_event(&usage_event)
        .expect("apply usage projection event");

    let restarted_core = RuntimeCore::default()
        .with_event_log_writer(event_log_writer)
        .with_projection_store(projection_store.clone());
    let read = restarted_core
        .read_session_current(AgentSessionReadParams {
            session_id: "sess_projection_usage_read".to_string(),
            history_limit: Some(1),
            history_offset: None,
            history_before_message_id: None,
        })
        .await
        .expect("read projection summary");
    let detail = read.detail.expect("detail");
    let watermark_after = projection_store
        .read_watermark("sess_projection_usage_read")
        .expect("watermark after")
        .expect("watermark after")
        .last_sequence;

    assert_eq!(watermark_after, watermark_before + 1);
    assert_eq!(detail["projection_source"], "runtime.projection_1");
    assert_eq!(
        detail["messages"][0]["metadata"]["source"].as_str(),
        Some("projection_summary")
    );
    assert_eq!(
        detail["turns"][0]["usage"]["input_tokens"].as_u64(),
        Some(1175)
    );
    assert_eq!(
        detail["thread_read"]["turns"][0]["usage"]["output_tokens"].as_u64(),
        Some(112)
    );
    assert_eq!(
        detail["thread_read"]["diagnostics"]["latest_turn_usage"]["cached_input_tokens"].as_u64(),
        Some(0)
    );
    assert_eq!(
        detail["thread_read"]["runtime_summary"]["latestTurnUsage"]["input_tokens"].as_u64(),
        Some(1175)
    );
}

#[tokio::test]
async fn read_session_current_projection_summary_preserves_process_items() {
    let temp = tempfile::tempdir().expect("tempdir");
    let roots =
        StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
    let event_log_writer = Arc::new(EventLogWriter::new(&roots.event_log_root).expect("writer"));
    let projection_store =
        Arc::new(ProjectionStore::initialize(&roots.projection_db_path).expect("projection"));
    let core = RuntimeCore::default().with_projection_store(projection_store.clone());
    core.start_session(empty_thread_session_params(
        "sess_projection_process",
        "thread_projection_process",
    ))
    .expect("start canonical session");
    let events = vec![
        AgentEvent {
            event_id: "evt_projection_process_user".to_string(),
            sequence: 1,
            session_id: "sess_projection_process".to_string(),
            thread_id: Some("thread_projection_process".to_string()),
            turn_id: Some("turn_projection_process".to_string()),
            event_type: "message.created".to_string(),
            timestamp: timestamp(),
            payload: json!({
                "input": [{
                    "type": "text",
                    "text": "打开历史时保留工具调用和思考"
                }],
                "session": {
                    "workspaceId": "workspace-current",
                    "modelName": "fixture-model"
                }
            }),
        },
        AgentEvent {
            event_id: "evt_projection_process_reasoning".to_string(),
            sequence: 2,
            session_id: "sess_projection_process".to_string(),
            thread_id: Some("thread_projection_process".to_string()),
            turn_id: Some("turn_projection_process".to_string()),
            event_type: "reasoning.delta".to_string(),
            timestamp: timestamp(),
            payload: json!({
                "text": "我会先查找历史事件，再汇总结论。",
                "metadata": {
                    "source": "reasoning"
                }
            }),
        },
        AgentEvent {
            event_id: "evt_projection_process_tool".to_string(),
            sequence: 3,
            session_id: "sess_projection_process".to_string(),
            thread_id: Some("thread_projection_process".to_string()),
            turn_id: Some("turn_projection_process".to_string()),
            event_type: "item.completed".to_string(),
            timestamp: timestamp(),
            payload: json!({
                "item": {
                    "sessionId": "sess_projection_process",
                    "threadId": "thread_projection_process",
                    "turnId": "turn_projection_process",
                    "itemId": "item_tool-history-web-search",
                    "sequence": 3,
                    "ordinal": 3,
                    "createdAtMs": 1_784_000_000_000_i64,
                    "updatedAtMs": 1_784_000_000_003_i64,
                    "completedAtMs": 1_784_000_000_003_i64,
                    "kind": "tool",
                    "status": "completed",
                    "payload": {
                        "type": "tool",
                        "call_id": "tool-history-web-search",
                        "name": "WebSearch",
                        "arguments": [{
                            "name": "query",
                            "value": "history process preservation"
                        }],
                        "output": {
                            "text": "搜索完成",
                            "truncated": false
                        }
                    },
                    "metadata": {}
                }
            }),
        },
        AgentEvent {
            event_id: "evt_projection_process_commentary".to_string(),
            sequence: 4,
            session_id: "sess_projection_process".to_string(),
            thread_id: Some("thread_projection_process".to_string()),
            turn_id: Some("turn_projection_process".to_string()),
            event_type: "message.delta".to_string(),
            timestamp: timestamp(),
            payload: json!({
                "itemId": "agent-commentary",
                "text": "上面工具还在查找。",
                "phase": "commentary"
            }),
        },
        AgentEvent {
            event_id: "evt_projection_process_routing".to_string(),
            sequence: 5,
            session_id: "sess_projection_process".to_string(),
            thread_id: Some("thread_projection_process".to_string()),
            turn_id: Some("turn_projection_process".to_string()),
            event_type: "routing.decision.made".to_string(),
            timestamp: timestamp(),
            payload: json!({
                "routingDecision": {
                    "decisionSource": "profile_model_slot",
                    "decisionReason": "profile_slot_selected",
                    "serviceModelSlot": "coding",
                    "selectedProvider": "custom-coding",
                    "selectedModel": "coder-large"
                }
            }),
        },
        AgentEvent {
            event_id: "evt_projection_process_commentary_batch".to_string(),
            sequence: 6,
            session_id: "sess_projection_process".to_string(),
            thread_id: Some("thread_projection_process".to_string()),
            turn_id: Some("turn_projection_process".to_string()),
            event_type: "message.delta".to_string(),
            timestamp: timestamp(),
            payload: json!({
                "itemId": "agent-commentary",
                "text": "批量过程输出也要保留。不能混进最终结论。",
                "phase": "commentary"
            }),
        },
        AgentEvent {
            event_id: "evt_projection_process_test_started".to_string(),
            sequence: 7,
            session_id: "sess_projection_process".to_string(),
            thread_id: Some("thread_projection_process".to_string()),
            turn_id: Some("turn_projection_process".to_string()),
            event_type: "test.started".to_string(),
            timestamp: timestamp(),
            payload: json!({
                "testRunId": "test-history-read",
                "suite": "app-server",
                "commandSummary": "cargo test read_session_current_"
            }),
        },
        AgentEvent {
            event_id: "evt_projection_process_test_completed".to_string(),
            sequence: 8,
            session_id: "sess_projection_process".to_string(),
            thread_id: Some("thread_projection_process".to_string()),
            turn_id: Some("turn_projection_process".to_string()),
            event_type: "test.completed".to_string(),
            timestamp: timestamp(),
            payload: json!({
                "testRunId": "test-history-read",
                "suite": "app-server",
                "commandSummary": "cargo test read_session_current_",
                "result": "passed",
                "passed": 5,
                "failed": 0
            }),
        },
        AgentEvent {
            event_id: "evt_projection_process_ask".to_string(),
            sequence: 9,
            session_id: "sess_projection_process".to_string(),
            thread_id: Some("thread_projection_process".to_string()),
            turn_id: Some("turn_projection_process".to_string()),
            event_type: "action.required".to_string(),
            timestamp: timestamp(),
            payload: json!({
                "requestId": "ask-history-process",
                "actionType": "ask_user",
                "prompt": "是否继续对齐 Codex？",
                "questions": [
                    {
                        "question": "是否继续保留非工具过程项？",
                        "header": "历史",
                        "options": [
                            {
                                "label": "继续",
                                "description": "保留 reasoning、测试、告警和产物。"
                            }
                        ]
                    }
                ]
            }),
        },
        AgentEvent {
            event_id: "evt_projection_process_ask_resolved".to_string(),
            sequence: 10,
            session_id: "sess_projection_process".to_string(),
            thread_id: Some("thread_projection_process".to_string()),
            turn_id: Some("turn_projection_process".to_string()),
            event_type: "action.resolved".to_string(),
            timestamp: timestamp(),
            payload: json!({
                "requestId": "ask-history-process",
                "actionType": "ask_user",
                "decision": "submitted"
            }),
        },
        AgentEvent {
            event_id: "evt_projection_process_warning".to_string(),
            sequence: 11,
            session_id: "sess_projection_process".to_string(),
            thread_id: Some("thread_projection_process".to_string()),
            turn_id: Some("turn_projection_process".to_string()),
            event_type: "runtime.warning".to_string(),
            timestamp: timestamp(),
            payload: json!({
                "code": "skill_not_available",
                "message": "模型路由发生降级，但任务继续。"
            }),
        },
        AgentEvent {
            event_id: "evt_projection_process_artifact".to_string(),
            sequence: 12,
            session_id: "sess_projection_process".to_string(),
            thread_id: Some("thread_projection_process".to_string()),
            turn_id: Some("turn_projection_process".to_string()),
            event_type: "artifact.snapshot".to_string(),
            timestamp: timestamp(),
            payload: json!({
                "artifact": {
                    "id": "artifact-history-workspace",
                    "kind": "article_workspace",
                    "title": "历史过程工作台",
                    "status": "completed",
                    "metadata": {
                        "articleWorkspace": {
                            "objects": [
                                {
                                    "ref": {
                                        "appId": "agent-runtime",
                                        "sessionId": "sess_projection_process",
                                        "kind": "articleDraft",
                                        "id": "draft-history"
                                    },
                                    "kind": "articleDraft",
                                    "title": "历史过程草稿",
                                    "documentText": "过程项已保留"
                                }
                            ]
                        }
                    }
                }
            }),
        },
        AgentEvent {
            event_id: "evt_projection_process_assistant".to_string(),
            sequence: 13,
            session_id: "sess_projection_process".to_string(),
            thread_id: Some("thread_projection_process".to_string()),
            turn_id: Some("turn_projection_process".to_string()),
            event_type: "message.delta_batch".to_string(),
            timestamp: timestamp(),
            payload: json!({
                "itemId": "agent-final",
                "deltas": [
                    {
                        "text": "结论：历史打开后应保留工具调用"
                    },
                    {
                        "text": "和思考。"
                    }
                ],
                "phase": "final"
            }),
        },
        AgentEvent {
            event_id: "evt_projection_process_completed".to_string(),
            sequence: 14,
            session_id: "sess_projection_process".to_string(),
            thread_id: Some("thread_projection_process".to_string()),
            turn_id: Some("turn_projection_process".to_string()),
            event_type: "turn.completed".to_string(),
            timestamp: timestamp(),
            payload: json!({}),
        },
    ];
    event_log_writer
        .append_events(&events)
        .expect("append events");
    let stored = core
        .state
        .lock()
        .expect("runtime core state mutex poisoned")
        .sessions
        .get("sess_projection_process")
        .expect("stored canonical session")
        .clone();
    projection_store
        .apply_canonical_events(&stored, &events)
        .expect("apply canonical events");
    projection_store
        .apply_events(&events)
        .expect("apply events");

    let restarted_core = RuntimeCore::default()
        .with_event_log_writer(event_log_writer)
        .with_projection_store(projection_store);
    let read = restarted_core
        .read_session_current(AgentSessionReadParams {
            session_id: "sess_projection_process".to_string(),
            history_limit: Some(2),
            history_offset: None,
            history_before_message_id: None,
        })
        .await
        .expect("read projection summary");
    let detail = read.detail.expect("detail");
    let items = detail["items"].as_array().expect("items");

    assert_eq!(detail["projection_source"], "runtime.projection_1");
    assert_eq!(
        detail["messages"][1]["metadata"]["source"].as_str(),
        Some("projection_summary")
    );
    assert_eq!(
        detail["messages"][1]["content"][0]["text"].as_str(),
        Some("结论：历史打开后应保留工具调用和思考。")
    );
    assert!(items.iter().any(|item| {
        item["type"].as_str() == Some("reasoning")
            && item["text"]
                .as_str()
                .is_some_and(|text| text.contains("查找历史事件"))
    }));
    assert!(items.iter().any(|item| {
        item["type"].as_str() == Some("web_search")
            && item["tool_name"].as_str() == Some("WebSearch")
    }));
    assert!(items.iter().any(|item| {
        item["type"].as_str() == Some("agent_message")
            && item["phase"].as_str() == Some("commentary")
            && item["text"].as_str().is_some_and(|text| {
                text.contains("上面工具还在查找。")
                    && text.contains("批量过程输出也要保留。")
                    && text.contains("不能混进最终结论。")
            })
    }));
    assert!(items.iter().any(|item| {
        item["type"].as_str() == Some("approval_request")
            && item["action_type"].as_str() == Some("ask_user")
            && item["status"].as_str() == Some("completed")
            && item["response"].is_null()
            && item["prompt"].as_str() == Some("是否继续对齐 Codex？")
    }));
    assert!(items.iter().any(|item| {
        item["type"].as_str() == Some("warning")
            && item["message"].as_str() == Some("模型路由发生降级，但任务继续。")
            && item["code"].as_str() == Some("skill_not_available")
    }));
    assert_eq!(detail["thread_read"]["thread_items"], detail["items"]);
    assert_eq!(
        detail["thread_read"]["model_routing"]["decisionSource"].as_str(),
        Some("profile_model_slot")
    );
    assert_eq!(
        detail["thread_read"]["service_model_slot"].as_str(),
        Some("coding")
    );
    let tests = detail["thread_read"]["tests"].as_array().expect("tests");
    assert!(tests.iter().any(|test| {
        test["test_run_id"].as_str() == Some("test-history-read")
            && test["status"].as_str() == Some("completed")
            && test["result"].as_str() == Some("passed")
            && test["passed"].as_i64() == Some(5)
    }));
    let artifacts = detail["artifacts"].as_array().expect("artifacts");
    assert!(artifacts.iter().any(|artifact| {
        artifact["artifactRef"].as_str() == Some("artifact-history-workspace")
            && artifact["kind"].as_str() == Some("article_workspace")
    }));
    assert_eq!(
        detail["article_workspace"]["sourceArtifacts"][0]["eventId"].as_str(),
        Some("evt_projection_process_artifact")
    );
    let tool_calls = detail["thread_read"]["tool_calls"]
        .as_array()
        .expect("tool calls");
    assert!(tool_calls.iter().any(|tool| {
        tool["tool_name"].as_str() == Some("WebSearch")
            && tool["status"].as_str() == Some("completed")
    }));
}

#[tokio::test]
async fn read_session_current_pages_projection_summary_with_cursor() {
    let temp = tempfile::tempdir().expect("tempdir");
    let roots =
        StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
    let event_log_writer = Arc::new(EventLogWriter::new(&roots.event_log_root).expect("writer"));
    let projection_store =
        Arc::new(ProjectionStore::initialize(&roots.projection_db_path).expect("projection"));
    let core = RuntimeCore::with_backend(Arc::new(CompletedBackend))
        .with_event_log_writer(event_log_writer.clone())
        .with_projection_store(projection_store.clone());
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_projection_cursor_read".to_string()),
        thread_id: Some("thread_projection_cursor_read".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    for index in 0..3 {
        core.start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_projection_cursor_read".to_string(),
                turn_id: Some(format!("turn_projection_cursor_read_{index}")),
                input: AgentInput {
                    text: format!("user {index}"),
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

    let restarted_core = RuntimeCore::default()
        .with_event_log_writer(event_log_writer)
        .with_projection_store(projection_store);
    let tail = restarted_core
        .read_session_current(AgentSessionReadParams {
            session_id: "sess_projection_cursor_read".to_string(),
            history_limit: Some(2),
            history_offset: None,
            history_before_message_id: None,
        })
        .await
        .expect("read tail");
    let tail_detail = tail.detail.expect("tail detail");
    let oldest_message_id = tail_detail["history_cursor"]["oldest_message_id"]
        .as_i64()
        .expect("oldest message id");
    assert_eq!(tail_detail["messages_count"].as_u64(), Some(6));
    assert_eq!(
        tail_detail["history_cursor"]["start_index"].as_u64(),
        Some(4)
    );
    assert_eq!(
        tail_detail["messages"][0]["content"][0]["text"].as_str(),
        Some("user 2")
    );

    let page = restarted_core
        .read_session_current(AgentSessionReadParams {
            session_id: "sess_projection_cursor_read".to_string(),
            history_limit: Some(2),
            history_offset: None,
            history_before_message_id: Some(oldest_message_id),
        })
        .await
        .expect("read cursor page");
    let page_detail = page.detail.expect("page detail");

    assert_eq!(
        page_detail["history_cursor"]["start_index"].as_u64(),
        Some(2)
    );
    assert_eq!(
        page_detail["messages"][0]["content"][0]["text"].as_str(),
        Some("user 1")
    );
    assert_eq!(
        page_detail["messages"][1]["content"][0]["text"].as_str(),
        Some("你好！有什么可以帮你的吗？")
    );
}

#[tokio::test]
async fn list_agent_sessions_derives_projection_title_from_first_user_message() {
    let temp = tempfile::tempdir().expect("tempdir");
    let roots =
        StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
    let event_log_writer = Arc::new(EventLogWriter::new(&roots.event_log_root).expect("writer"));
    let projection_store =
        Arc::new(ProjectionStore::initialize(&roots.projection_db_path).expect("projection"));
    let core = RuntimeCore::with_backend(Arc::new(CompletedBackend))
        .with_event_log_writer(event_log_writer.clone())
        .with_projection_store(projection_store.clone());
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_projection_title".to_string()),
        thread_id: Some("thread_projection_title".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: Some(app_server_protocol::BusinessObjectRef {
            kind: "agent.session".to_string(),
            id: "sess_projection_title".to_string(),
            title: Some("未命名对话".to_string()),
            uri: None,
            metadata: Some(json!({
                "title": "未命名对话",
                "modelName": "fixture-model",
            })),
        }),
        locale: None,
    })
    .expect("session");

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_projection_title".to_string(),
            turn_id: Some("turn_projection_title".to_string()),
            input: AgentInput {
                text: "根据原生方式生成标题".to_string(),
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

    let restarted_core = RuntimeCore::default()
        .with_event_log_writer(event_log_writer)
        .with_projection_store(projection_store);
    let listed = restarted_core
        .list_agent_sessions(AgentSessionListParams {
            workspace_id: Some("workspace-current".to_string()),
            limit: Some(20),
            ..AgentSessionListParams::default()
        })
        .await
        .expect("list projected sessions");

    assert_eq!(listed.sessions.len(), 1);
    assert_eq!(listed.sessions[0].session_id, "sess_projection_title");
    assert_eq!(
        listed.sessions[0].title.as_deref(),
        Some("根据原生方式生成标题")
    );
}
