use super::support::*;
use super::*;
use crate::runtime::session_control::QueuedTurnResume;

#[tokio::test]
async fn queued_turn_blocks_then_resumes_after_active_turn_completes() {
    let core = RuntimeCore::default();
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_queue".to_string()),
        thread_id: Some("thread_queue".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_queue".to_string(),
            turn_id: Some("turn_running".to_string()),
            input: AgentInput {
                text: "running".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: Some(RuntimeOptions {
                runtime_request: Some(RuntimeRequest {
                    provider_config: Some(RuntimeProviderConfig {
                        provider_id: Some("fixture-provider".to_string()),
                        provider_name: Some("openai".to_string()),
                        model_name: Some("fixture-model".to_string()),
                        api_key: Some("fixture-key".to_string()),
                        base_url: Some("http://127.0.0.1:65535".to_string()),
                        ..RuntimeProviderConfig::default()
                    }),
                    provider_preference: Some("fixture-provider".to_string()),
                    model_preference: Some("fixture-model".to_string()),
                    ..RuntimeRequest::default()
                }),
                ..RuntimeOptions::default()
            }),
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("running turn");
    let queued = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_queue".to_string(),
                turn_id: Some("turn_queued".to_string()),
                input: AgentInput {
                    text: "queued".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: true,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("queued turn");
    assert_eq!(queued.response.turn.status, AgentTurnStatus::Queued);
    assert!(queued
        .events
        .iter()
        .any(|event| event.event_type == "queue.added"));

    let blocked_resume = core
        .resume_next_queued_turn_if_idle("sess_queue", RuntimeHostContext::default())
        .await
        .expect("blocked resume");
    assert!(matches!(blocked_resume, QueuedTurnResume::Blocked));

    core.append_external_runtime_events(
        "sess_queue",
        Some("turn_running"),
        vec![RuntimeEvent::new("turn.completed", json!({}))],
    )
    .expect("complete running");
    let resumed = core
        .resume_next_queued_turn_if_idle("sess_queue", RuntimeHostContext::default())
        .await
        .expect("resume queued");
    match resumed {
        QueuedTurnResume::Started {
            queued_turn_id,
            events,
        } => {
            assert_eq!(queued_turn_id, "turn_queued");
            assert!(events
                .iter()
                .any(|event| event.event_type == "turn.accepted"));
        }
        QueuedTurnResume::Empty | QueuedTurnResume::Blocked => {
            panic!("queued turn helper did not start turn_queued")
        }
    }
}

#[tokio::test]
async fn queue_if_busy_turn_starts_after_previous_turn_completed() {
    let backend = Arc::new(RecordingBackend::default());
    let core = RuntimeCore::with_backend(backend.clone());
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_queue_after_completed".to_string()),
        thread_id: Some("thread_queue_after_completed".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_queue_after_completed".to_string(),
            turn_id: Some("turn_completed".to_string()),
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
    core.append_external_runtime_events(
        "sess_queue_after_completed",
        Some("turn_completed"),
        vec![RuntimeEvent::new("turn.completed", json!({}))],
    )
    .expect("complete first turn");

    let output = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_queue_after_completed".to_string(),
                turn_id: Some("turn_followup".to_string()),
                input: AgentInput {
                    text: "follow up".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: true,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("follow-up turn");

    assert_ne!(output.response.turn.status, AgentTurnStatus::Queued);
    assert!(output
        .events
        .iter()
        .all(|event| event.event_type != "queue.added"));

    let requests = backend
        .requests
        .lock()
        .expect("test backend requests mutex poisoned");
    assert!(requests
        .iter()
        .any(|request| request.turn.turn_id == "turn_followup"));

    let read = core
        .read_session_current(AgentSessionReadParams {
            session_id: "sess_queue_after_completed".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .await
        .expect("read session");
    assert!(!read
        .turns
        .iter()
        .any(|turn| turn.status == AgentTurnStatus::Queued));
}

#[tokio::test]
async fn duplicate_queue_if_busy_input_reuses_active_turn() {
    let backend = Arc::new(RunningCountingBackend {
        start_count: AtomicUsize::new(0),
    });
    let core = RuntimeCore::with_backend(backend.clone());
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_duplicate_queue".to_string()),
        thread_id: Some("thread_duplicate_queue".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    let input = AgentInput {
        text: "同一条专家首轮请求".to_string(),
        attachments: Vec::new(),
    };
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_duplicate_queue".to_string(),
            turn_id: Some("turn_active".to_string()),
            input: input.clone(),
            runtime_options: None,
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("first turn");

    let duplicate = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_duplicate_queue".to_string(),
                turn_id: Some("turn_duplicate".to_string()),
                input,
                runtime_options: None,
                queue_if_busy: true,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("duplicate turn should be idempotent");

    assert_eq!(duplicate.response.turn.turn_id, "turn_active");
    assert!(duplicate.events.is_empty());
    assert_eq!(backend.start_count.load(Ordering::SeqCst), 1);

    let read = core
        .read_session_current(AgentSessionReadParams {
            session_id: "sess_duplicate_queue".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .await
        .expect("read session");
    assert_eq!(read.turns.len(), 1);
    assert!(!read
        .turns
        .iter()
        .any(|turn| turn.status == AgentTurnStatus::Queued));
    assert_eq!(
        read.detail
            .as_ref()
            .and_then(|detail| detail.get("queued_turns"))
            .and_then(serde_json::Value::as_array)
            .map(Vec::len),
        Some(0)
    );
}

#[tokio::test]
async fn duplicate_queue_if_busy_input_reuses_existing_queued_turn() {
    let backend = Arc::new(RunningCountingBackend {
        start_count: AtomicUsize::new(0),
    });
    let core = RuntimeCore::with_backend(backend.clone());
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_duplicate_existing_queue".to_string()),
        thread_id: Some("thread_duplicate_existing_queue".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_duplicate_existing_queue".to_string(),
            turn_id: Some("turn_active".to_string()),
            input: AgentInput {
                text: "第一条还在运行".to_string(),
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

    let input = AgentInput {
        text: "同一条专家首轮请求".to_string(),
        attachments: Vec::new(),
    };
    let queued = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_duplicate_existing_queue".to_string(),
                turn_id: Some("turn_queued".to_string()),
                input: input.clone(),
                runtime_options: None,
                queue_if_busy: true,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("queued turn");
    assert_eq!(queued.response.turn.turn_id, "turn_queued");
    assert_eq!(queued.response.turn.status, AgentTurnStatus::Queued);
    assert!(queued
        .events
        .iter()
        .any(|event| event.event_type == "queue.added"));

    let duplicate = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_duplicate_existing_queue".to_string(),
                turn_id: Some("turn_duplicate".to_string()),
                input,
                runtime_options: None,
                queue_if_busy: true,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("duplicate queued turn should be idempotent");

    assert_eq!(duplicate.response.turn.turn_id, "turn_queued");
    assert_eq!(duplicate.response.turn.status, AgentTurnStatus::Queued);
    assert!(duplicate.events.is_empty());
    assert_eq!(backend.start_count.load(Ordering::SeqCst), 1);

    let read = core
        .read_session_current(AgentSessionReadParams {
            session_id: "sess_duplicate_existing_queue".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .await
        .expect("read session");
    let queued_turns = read
        .turns
        .iter()
        .filter(|turn| turn.status == AgentTurnStatus::Queued)
        .collect::<Vec<_>>();
    assert_eq!(queued_turns.len(), 1);
    assert_eq!(queued_turns[0].turn_id, "turn_queued");
    assert_eq!(
        read.detail
            .as_ref()
            .and_then(|detail| detail.get("queued_turns"))
            .and_then(serde_json::Value::as_array)
            .map(Vec::len),
        Some(1)
    );
}

#[tokio::test]
async fn read_session_projects_queued_turn_input_snapshot() {
    let backend = Arc::new(RecordingBackend::default());
    let core = RuntimeCore::with_backend(backend);
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_queue_snapshot".to_string()),
        thread_id: Some("thread_queue_snapshot".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_queue_snapshot".to_string(),
            turn_id: Some("turn_active".to_string()),
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

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_queue_snapshot".to_string(),
            turn_id: Some("turn_queued".to_string()),
            input: AgentInput {
                text: "请继续执行排队任务，并保留完整输入".to_string(),
                attachments: vec![AgentAttachment {
                    kind: "image".to_string(),
                    uri: Some("file://queued.png".to_string()),
                    metadata: Some(json!({
                        "mediaType": "image/png",
                        "sourcePath": "/project/queued.png"
                    })),
                }],
            },
            runtime_options: Some(RuntimeOptions {
                runtime_request: Some(RuntimeRequest {
                    metadata: Some(json!({
                    "path_references": [
                        {
                            "path": "/project/report.md",
                            "name": "report.md",
                            "is_dir": false,
                            "isDir": false,
                            "source": "file_manager"
                        }
                    ],
                    "text_elements": [
                        {
                            "type": "text",
                            "text": "请继续执行排队任务，并保留完整输入"
                        }
                    ],
                    "input_capability_route": {
                        "kind": "installed_skill",
                        "skillKey": "code-review",
                        "skillName": "Code Review"
                    }
                    })),
                    ..RuntimeRequest::default()
                }),
                ..RuntimeOptions::default()
            }),
            queue_if_busy: true,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("queued turn");

    let read = core
        .read_session_current(AgentSessionReadParams {
            session_id: "sess_queue_snapshot".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .await
        .expect("read session");
    let detail = read.detail.expect("read detail");
    let queued = &detail["queued_turns"][0];
    assert_eq!(queued["queued_turn_id"], "turn_queued");
    assert_eq!(queued["turn_id"], "turn_queued");
    assert_eq!(queued["status"], "queued");
    assert_eq!(queued["message_text"], "请继续执行排队任务，并保留完整输入");
    assert_eq!(
        queued["message_preview"],
        "请继续执行排队任务，并保留完整输入"
    );
    assert_eq!(queued["image_count"], 1);
    assert_eq!(queued["attachments"][0]["kind"], "image");
    assert_eq!(queued["attachments"][0]["uri"], "file://queued.png");
    assert_eq!(
        queued["input_attachments"][0]["detail"],
        serde_json::Value::Null
    );
    assert_eq!(queued["path_references"][0]["path"], "/project/report.md");
    assert_eq!(queued["pathReferences"][0]["name"], "report.md");
    assert_eq!(
        queued["text_elements"][0]["text"],
        "请继续执行排队任务，并保留完整输入"
    );
    assert_eq!(queued["input_capability_route"]["skillKey"], "code-review");
    assert_eq!(queued["inputCapabilityRoute"]["skillName"], "Code Review");
    assert_eq!(queued["position"], 0);
    assert_eq!(detail["thread_read"]["queued_turns"][0], *queued);
}

#[tokio::test]
async fn queued_turn_snapshot_recovers_message_text_from_event_log() {
    let temp = tempfile::tempdir().expect("tempdir");
    let roots = StorageRoots::initialize(temp.path().join("app-server")).expect("roots");
    let event_log_writer = Arc::new(EventLogWriter::new(&roots.event_log_root).expect("writer"));
    let projection_store =
        Arc::new(ProjectionStore::initialize(&roots.projection_db_path).expect("projection"));
    let core = RuntimeCore::with_backend(Arc::new(RunningCountingBackend {
        start_count: AtomicUsize::new(0),
    }))
    .with_event_log_writer(event_log_writer.clone())
    .with_projection_store(projection_store.clone());
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_queue_projection".to_string()),
        thread_id: Some("thread_queue_projection".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_queue_projection".to_string(),
            turn_id: Some("turn_running".to_string()),
            input: AgentInput {
                text: "running".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("running turn");
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_queue_projection".to_string(),
            turn_id: Some("turn_queued".to_string()),
            input: AgentInput {
                text: "请继续以「代码文学专家」身份，使用刚添加的技能再做一次最小代码审查。"
                    .to_string(),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: true,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("queued turn");
    projection_store
        .clear_session("sess_queue_projection")
        .expect("simulate missing projection");

    let restarted_core = RuntimeCore::default()
        .with_event_log_writer(event_log_writer)
        .with_projection_store(projection_store);
    let read = restarted_core
        .read_session_current(AgentSessionReadParams {
            session_id: "sess_queue_projection".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .await
        .expect("read recovered session");

    let thread_read = read
        .detail
        .as_ref()
        .and_then(|detail| detail.get("thread_read"))
        .expect("thread read");
    let queued_text = thread_read["queued_turns"][0]["message_text"]
        .as_str()
        .unwrap_or_else(|| panic!("queued turn message text missing: {thread_read}"));
    assert_eq!(
        queued_text,
        "请继续以「代码文学专家」身份，使用刚添加的技能再做一次最小代码审查。"
    );
}

#[tokio::test]
async fn resume_queued_turn_preserves_runtime_options_for_backend() {
    let backend = Arc::new(RecordingBackend::default());
    let core = RuntimeCore::with_backend(backend.clone());
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_queue_runtime_options".to_string()),
        thread_id: Some("thread_queue_runtime_options".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_queue_runtime_options".to_string(),
            turn_id: Some("turn_running".to_string()),
            input: AgentInput {
                text: "running".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: None,
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("running turn");

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_queue_runtime_options".to_string(),
            turn_id: Some("turn_queued".to_string()),
            input: AgentInput {
                text: "queued".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: Some(RuntimeOptions {
                runtime_request: Some(RuntimeRequest {
                    provider_preference: Some("fixture-provider".to_string()),
                    model_preference: Some("fixture-model".to_string()),
                    metadata: Some(json!({
                        "expert": {
                            "expertId": "code-literature",
                            "skillRefs": ["skill:code-review", "skill:local:capability-report"]
                        }
                    })),
                    ..RuntimeRequest::default()
                }),
                ..RuntimeOptions::default()
            }),
            queue_if_busy: true,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("queued turn");

    core.append_external_runtime_events(
        "sess_queue_runtime_options",
        Some("turn_running"),
        vec![RuntimeEvent::new("turn.completed", json!({}))],
    )
    .expect("complete running");

    let resumed = core
        .resume_next_queued_turn_if_idle(
            "sess_queue_runtime_options",
            RuntimeHostContext::default(),
        )
        .await
        .expect("resume queued");
    assert!(matches!(
        resumed,
        QueuedTurnResume::Started { queued_turn_id, .. } if queued_turn_id == "turn_queued"
    ));

    let requests = backend
        .requests
        .lock()
        .expect("test backend requests mutex poisoned");
    let resumed_request = requests
        .iter()
        .find(|request| request.turn.turn_id == "turn_queued")
        .expect("resumed queued request");
    assert_eq!(
        resumed_request.provider_preference(),
        Some("fixture-provider")
    );
    assert_eq!(resumed_request.model_preference(), Some("fixture-model"));
    assert_eq!(
        resumed_request.queued_turn_id.as_deref(),
        Some("turn_queued")
    );
    assert_eq!(
        resumed_request
            .runtime_metadata()
            .and_then(|metadata| metadata.pointer("/expert/skillRefs/1"))
            .and_then(serde_json::Value::as_str),
        Some("skill:local:capability-report")
    );
    assert_eq!(
        resumed_request
            .runtime_options
            .as_ref()
            .and_then(|options| options.runtime_request.as_ref())
            .and_then(|runtime_request| runtime_request.metadata.as_ref())
            .and_then(|metadata| metadata.pointer("/expert/skillRefs/0"))
            .and_then(serde_json::Value::as_str),
        Some("skill:code-review")
    );
}

#[tokio::test]
async fn second_active_turn_without_queue_fails_closed() {
    let backend = Arc::new(RunningCountingBackend {
        start_count: AtomicUsize::new(0),
    });
    let core = RuntimeCore::with_backend(backend.clone());
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_single_active".to_string()),
        thread_id: Some("thread_single_active".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_single_active".to_string(),
            turn_id: Some("turn_active".to_string()),
            input: AgentInput {
                text: "running".to_string(),
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

    let error = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_single_active".to_string(),
                turn_id: Some("turn_parallel".to_string()),
                input: AgentInput {
                    text: "parallel".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect_err("parallel turn must fail closed");

    match error {
        RuntimeCoreError::TurnAlreadyActive(turn_id) => {
            assert_eq!(turn_id, "turn_active");
        }
        other => panic!("expected active turn error, got {other:?}"),
    }
    assert_eq!(backend.start_count.load(Ordering::SeqCst), 1);

    let read = core
        .read_session_current(AgentSessionReadParams {
            session_id: "sess_single_active".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .await
        .expect("read session");
    assert_eq!(read.turns.len(), 1);
    assert_eq!(read.turns[0].turn_id, "turn_active");
    let events = core
        .events_for_session("sess_single_active")
        .expect("runtime events");
    assert!(events
        .iter()
        .all(|event| event.turn_id.as_deref() != Some("turn_parallel")));
}

#[tokio::test]
async fn resume_queued_turn_restores_queue_when_backend_fails_before_emit() {
    let core = RuntimeCore::with_backend(Arc::new(FailBeforeEmitBackend {
        start_count: AtomicUsize::new(0),
    }));
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_queue_rollback".to_string()),
        thread_id: Some("thread_queue_rollback".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_queue_rollback".to_string(),
            turn_id: Some("turn_running".to_string()),
            input: AgentInput {
                text: "running".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: Some(RuntimeOptions {
                runtime_request: Some(RuntimeRequest {
                    provider_config: Some(RuntimeProviderConfig {
                        provider_id: Some("fixture-provider".to_string()),
                        provider_name: Some("openai".to_string()),
                        model_name: Some("fixture-model".to_string()),
                        api_key: Some("fixture-key".to_string()),
                        base_url: Some("http://127.0.0.1:65535".to_string()),
                        ..RuntimeProviderConfig::default()
                    }),
                    provider_preference: Some("fixture-provider".to_string()),
                    model_preference: Some("fixture-model".to_string()),
                    ..RuntimeRequest::default()
                }),
                ..RuntimeOptions::default()
            }),
            queue_if_busy: false,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("running turn");
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_queue_rollback".to_string(),
            turn_id: Some("turn_queued".to_string()),
            input: AgentInput {
                text: "queued".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: Some(RuntimeOptions {
                runtime_request: Some(RuntimeRequest {
                    provider_preference: Some("fixture-provider".to_string()),
                    model_preference: Some("fixture-model".to_string()),
                    ..RuntimeRequest::default()
                }),
                ..RuntimeOptions::default()
            }),
            queue_if_busy: true,
            skip_pre_submit_resume: false,
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("queued turn");
    core.append_external_runtime_events(
        "sess_queue_rollback",
        Some("turn_running"),
        vec![RuntimeEvent::new("turn.completed", json!({}))],
    )
    .expect("complete running");

    let error = core
        .resume_next_queued_turn_if_idle("sess_queue_rollback", RuntimeHostContext::default())
        .await
        .expect_err("resume should fail before backend emits");
    assert!(matches!(error, RuntimeCoreError::Backend(_)));

    let read = core
        .read_session_current(AgentSessionReadParams {
            session_id: "sess_queue_rollback".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .await
        .expect("read session");
    assert!(read
        .turns
        .iter()
        .any(|turn| turn.turn_id == "turn_queued" && turn.status == AgentTurnStatus::Queued));
}
