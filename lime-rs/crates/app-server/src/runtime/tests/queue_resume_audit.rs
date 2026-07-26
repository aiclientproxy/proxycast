use super::support::*;
use super::*;
use crate::runtime::session_control::QueuedTurnResume;

async fn create_resume_audit_core(
    session_id: &str,
    thread_id: &str,
    event_log_writer: Arc<EventLogWriter>,
) -> RuntimeCore {
    let core = RuntimeCore::with_backend(Arc::new(RecordingBackend::default()))
        .with_event_log_writer(event_log_writer);
    core.start_session(AgentSessionStartParams {
        session_id: Some(session_id.to_string()),
        thread_id: Some(thread_id.to_string()),
        app_id: "content-studio".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: session_id.to_string(),
            turn_id: Some("turn_running".to_string()),
            input: AgentInput {
                text: "running content workflow".to_string(),
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
            session_id: session_id.to_string(),
            turn_id: Some("turn_queued".to_string()),
            input: AgentInput {
                text: "queued follow-up".to_string(),
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

    core
}

fn complete_running_turn(core: &RuntimeCore, session_id: &str) {
    core.append_external_runtime_events(
        session_id,
        Some("turn_running"),
        vec![RuntimeEvent::new("turn.completed", json!({}))],
    )
    .expect("complete running turn");
}

#[tokio::test]
async fn queued_resume_helper_reports_idle_boundary_without_workflow_audit() {
    let temp = tempfile::tempdir().expect("tempdir");
    let roots =
        StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
    let event_log_writer = Arc::new(EventLogWriter::new(&roots.event_log_root).expect("writer"));
    let core = create_resume_audit_core(
        "sess_queue_resume_helper",
        "thread_queue_resume_helper",
        event_log_writer,
    )
    .await;

    assert!(matches!(
        core.resume_next_queued_turn_if_idle(
            "sess_queue_resume_helper",
            RuntimeHostContext::default(),
        )
        .await
        .expect("blocked queued resume"),
        QueuedTurnResume::Blocked
    ));

    complete_running_turn(&core, "sess_queue_resume_helper");
    let started = core
        .resume_next_queued_turn_if_idle("sess_queue_resume_helper", RuntimeHostContext::default())
        .await
        .expect("start queued resume");
    match started {
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
            panic!("queued resume helper did not start the queued turn")
        }
    }

    let empty_core = RuntimeCore::with_backend(Arc::new(RecordingBackend::default()));
    empty_core
        .start_session(AgentSessionStartParams {
            session_id: Some("sess_queue_resume_empty".to_string()),
            thread_id: Some("thread_queue_resume_empty".to_string()),
            app_id: "content-studio".to_string(),
            workspace_id: Some("workspace-current".to_string()),
            business_object_ref: None,
            locale: None,
        })
        .expect("empty session");
    assert!(matches!(
        empty_core
            .resume_next_queued_turn_if_idle(
                "sess_queue_resume_empty",
                RuntimeHostContext::default(),
            )
            .await
            .expect("empty queued resume"),
        QueuedTurnResume::Empty
    ));
}

#[tokio::test]
async fn queued_turn_helper_does_not_forge_workflow_resume_audit() {
    let temp = tempfile::tempdir().expect("tempdir");
    let roots =
        StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
    let event_log_writer = Arc::new(EventLogWriter::new(&roots.event_log_root).expect("writer"));
    let core = create_resume_audit_core(
        "sess_queue_resume_audit",
        "thread_queue_resume_audit",
        event_log_writer.clone(),
    )
    .await;

    let workflow_output = core
        .append_external_runtime_events(
            "sess_queue_resume_audit",
            Some("turn_running"),
            vec![
                RuntimeEvent::new(
                    "workflow.run.started",
                    json!({
                        "workflowRunId": "turn_running:content-article",
                        "workflowKey": "content_article_workflow",
                        "status": "running",
                        "metadata": {
                            "pluginWorkflow": {
                                "status": "running"
                            }
                        }
                    }),
                ),
                RuntimeEvent::new(
                    "workflow.run.completed",
                    json!({
                        "workflowRunId": "turn_running:content-article",
                        "workflowKey": "content_article_workflow",
                        "status": "completed",
                        "metadata": {
                            "pluginWorkflow": {
                                "status": "completed"
                            }
                        }
                    }),
                ),
            ],
        )
        .expect("append workflow audit events");
    assert!(
        workflow_output.is_empty(),
        "workflow audit events must not enter regular runtime output: {workflow_output:?}"
    );
    complete_running_turn(&core, "sess_queue_resume_audit");

    let resumed = core
        .resume_next_queued_turn_if_idle("sess_queue_resume_audit", RuntimeHostContext::default())
        .await
        .expect("resume queued");
    assert!(matches!(
        resumed,
        QueuedTurnResume::Started { queued_turn_id, .. } if queued_turn_id == "turn_queued"
    ));

    let regular_records = event_log_writer
        .read_session_events("sess_queue_resume_audit")
        .expect("regular records");
    assert!(
        regular_records
            .iter()
            .all(|record| !record.event.event_type.starts_with("workflow.")),
        "workflow audit events must stay out of regular session JSONL"
    );
    let audit_records = event_log_writer
        .read_session_workflow_audit_events("sess_queue_resume_audit")
        .expect("workflow audit records");
    let event_types = audit_records
        .iter()
        .map(|record| record.event.event_type.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        event_types,
        vec!["workflow.run.started", "workflow.run.completed"]
    );
    assert!(
        audit_records
            .iter()
            .all(|record| !record.event.event_type.contains("resum")),
        "queued turn helper must not forge plugin worker resume audit events"
    );
}

#[tokio::test]
async fn pending_work_recovery_scans_in_memory_queued_turns() {
    let backend = Arc::new(RecordingBackend::default());
    let core = RuntimeCore::with_backend(backend.clone());
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_pending_work_memory".to_string()),
        thread_id: Some("thread_pending_work_memory".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: None,
        business_object_ref: None,
        locale: None,
    })
    .expect("session");
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_pending_work_memory".to_string(),
            turn_id: Some("turn_pending_work_active".to_string()),
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
            session_id: "sess_pending_work_memory".to_string(),
            turn_id: Some("turn_pending_work_queued".to_string()),
            input: AgentInput {
                text: "queued recovery input".to_string(),
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
    core.append_external_runtime_events(
        "sess_pending_work_memory",
        Some("turn_pending_work_active"),
        vec![RuntimeEvent::new("turn.completed", json!({}))],
    )
    .expect("complete active turn");

    core.recover_agent_control_spawns(RuntimeHostContext::default(), None)
        .await
        .expect("recover pending work");

    let requests = backend
        .requests
        .lock()
        .expect("backend requests")
        .iter()
        .map(|request| (request.turn.turn_id.clone(), request.input.concat_text()))
        .collect::<Vec<_>>();
    assert_eq!(
        requests,
        vec![
            ("turn_pending_work_active".to_string(), "active".to_string()),
            (
                "turn_pending_work_queued".to_string(),
                "queued recovery input".to_string()
            ),
        ]
    );
}

#[tokio::test]
async fn action_response_writes_workflow_resume_audit_with_worker_lifecycle_metadata() {
    let temp = tempfile::tempdir().expect("tempdir");
    let roots =
        StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
    let event_log_writer = Arc::new(EventLogWriter::new(&roots.event_log_root).expect("writer"));
    let core = RuntimeCore::with_backend(Arc::new(RecordingBackend::default()))
        .with_event_log_writer(event_log_writer.clone());
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_action_resume_lifecycle".to_string()),
        thread_id: Some("thread_action_resume_lifecycle".to_string()),
        app_id: "content-studio".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_action_resume_lifecycle".to_string(),
            turn_id: Some("turn_action".to_string()),
            input: AgentInput {
                text: "waiting for plugin host response".to_string(),
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
    core.append_external_runtime_events(
        "sess_action_resume_lifecycle",
        Some("turn_action"),
        vec![RuntimeEvent::new(
            "action.required",
            json!({
                "requestId": "article-draft-review",
                "actionType": "ask_user",
                "prompt": "Review the draft",
                "deadlineAtMs": 1_999_999_999_999_u64,
            }),
        )],
    )
    .expect("append pending action");

    core.respond_action(
        AgentSessionActionRespondParams {
            session_id: "sess_action_resume_lifecycle".to_string(),
            request_id: "article-draft-review".to_string(),
            action_type: AgentSessionActionType::AskUser,
            decision: None,
            confirmed: Some(true),
            response: Some("decision text must stay out of workflow audit".to_string()),
            user_data: Some(json!({
                "notes": "private reviewer details"
            })),
            metadata: Some(json!({
                "workflowResume": {
                    "workflowRunId": "turn_action:content-article",
                    "workflowKey": "content_article_workflow",
                    "stepId": "draft"
                },
                "plugin_runtime": {
                    "app_id": "content-factory-app",
                    "task_id": "plugin-task-1"
                }
            })),
            event_name: None,
            action_scope: Some(AgentSessionActionScope {
                session_id: Some("sess_action_resume_lifecycle".to_string()),
                thread_id: Some("thread_action_resume_lifecycle".to_string()),
                turn_id: Some("turn_action".to_string()),
            }),
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("respond action");

    let regular_records = event_log_writer
        .read_session_events("sess_action_resume_lifecycle")
        .expect("regular records");
    assert!(
        regular_records
            .iter()
            .all(|record| !record.event.event_type.starts_with("workflow.")),
        "workflow action response audit events must stay out of regular session JSONL"
    );

    let audit_records = event_log_writer
        .read_session_workflow_audit_events("sess_action_resume_lifecycle")
        .expect("workflow audit records");
    let event_types = audit_records
        .iter()
        .map(|record| record.event.event_type.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        event_types,
        vec!["workflow.step.resuming", "workflow.run.resuming"]
    );
    for record in audit_records {
        assert_eq!(
            record.event.payload["workflowRunId"],
            json!("turn_action:content-article")
        );
        assert_eq!(
            record.event.payload["workflowKey"],
            json!("content_article_workflow")
        );
        assert_eq!(record.event.payload["stepId"], json!("draft"));
        assert_eq!(
            record.event.payload["actionId"],
            json!("article-draft-review")
        );
        assert_eq!(
            record.event.payload["source"],
            json!("agentSession/action/respond")
        );
        assert_eq!(
            record.event.payload["redaction"]["policy"],
            json!("workflow_audit_metadata_only")
        );
        let serialized = record.event.payload.to_string();
        assert!(!serialized.contains("decision text"));
        assert!(!serialized.contains("private reviewer details"));
    }
}

#[tokio::test]
async fn workflow_respond_writes_resume_lifecycle_before_progress_audit() {
    let temp = tempfile::tempdir().expect("tempdir");
    let roots =
        StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
    let event_log_writer = Arc::new(EventLogWriter::new(&roots.event_log_root).expect("writer"));
    let core = RuntimeCore::with_backend(Arc::new(RecordingBackend::default()))
        .with_event_log_writer(event_log_writer.clone());
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_workflow_respond_lifecycle".to_string()),
        thread_id: Some("thread_workflow_respond_lifecycle".to_string()),
        app_id: "content-studio".to_string(),
        workspace_id: Some("workspace-current".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_workflow_respond_lifecycle".to_string(),
            turn_id: Some("turn_waiting".to_string()),
            input: AgentInput {
                text: "waiting workflow action".to_string(),
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
    core.append_external_runtime_events(
        "sess_workflow_respond_lifecycle",
        Some("turn_waiting"),
        vec![RuntimeEvent::new(
            "action.required",
            json!({
                "requestId": "article-draft-review",
                "actionType": "ask_user",
                "prompt": "Review the draft",
                "deadlineAtMs": 1_999_999_999_999_u64,
            }),
        )],
    )
    .expect("append workflow pending action");
    core.append_external_runtime_events(
        "sess_workflow_respond_lifecycle",
        Some("turn_waiting"),
        vec![
            RuntimeEvent::new(
                "workflow.run.started",
                json!({
                    "workflowRunId": "turn_waiting:content-article",
                    "workflowKey": "content_article_workflow",
                    "status": "running",
                    "turnId": "turn_waiting",
                }),
            ),
            RuntimeEvent::new(
                "workflow.step.progress",
                json!({
                    "workflowRunId": "turn_waiting:content-article",
                    "workflowKey": "content_article_workflow",
                    "stepId": "draft",
                    "stepTitle": "正文写作",
                    "status": "waiting",
                    "requestId": "article-draft-review",
                    "actionType": "ask_user",
                }),
            ),
        ],
    )
    .expect("workflow waiting");

    core.respond_workflow_current(
        WorkflowRespondParams {
            session_id: "sess_workflow_respond_lifecycle".to_string(),
            workflow_run_id: "turn_waiting:content-article".to_string(),
            step_id: Some("draft".to_string()),
            request_id: Some("article-draft-review".to_string()),
            action_type: Some(AgentSessionActionType::AskUser),
            confirmed: Some(true),
            response: Some(json!({
                "response": "raw workflow response must stay redacted"
            })),
        },
        RuntimeHostContext::default(),
    )
    .await
    .expect("workflow respond");

    let regular_records = event_log_writer
        .read_session_events("sess_workflow_respond_lifecycle")
        .expect("regular records");
    assert!(
        regular_records
            .iter()
            .all(|record| !record.event.event_type.starts_with("workflow.")),
        "workflow/respond audit events must stay out of regular session JSONL"
    );

    let audit_records = event_log_writer
        .read_session_workflow_audit_events("sess_workflow_respond_lifecycle")
        .expect("workflow audit records");
    let event_types = audit_records
        .iter()
        .map(|record| record.event.event_type.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        event_types,
        vec![
            "workflow.run.started",
            "workflow.step.progress",
            "workflow.step.resuming",
            "workflow.run.resuming",
            "workflow.step.progress"
        ]
    );
    let resume_records = audit_records
        .iter()
        .filter(|record| record.event.event_type.contains("resuming"))
        .collect::<Vec<_>>();
    assert_eq!(resume_records.len(), 2);
    for record in resume_records {
        assert_eq!(
            record.event.payload["workflowRunId"],
            json!("turn_waiting:content-article")
        );
        assert_eq!(
            record.event.payload["workflowKey"],
            json!("content_article_workflow")
        );
        assert_eq!(record.event.payload["stepId"], json!("draft"));
        assert_eq!(
            record.event.payload["actionId"],
            json!("article-draft-review")
        );
        assert_eq!(
            record.event.payload["source"],
            json!("agentSession/action/respond")
        );
        assert_eq!(
            record.event.payload["redaction"]["policy"],
            json!("workflow_audit_metadata_only")
        );
        assert!(
            !record
                .event
                .payload
                .to_string()
                .contains("raw workflow response"),
            "workflow response must not be copied into resume lifecycle audit"
        );
    }
}
