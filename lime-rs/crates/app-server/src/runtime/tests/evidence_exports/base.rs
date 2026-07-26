use super::*;

#[tokio::test]
async fn export_evidence_reads_session_turn_events_and_artifact_summaries() {
    let sidecar_root = tempfile::tempdir().expect("sidecar root");
    let core = RuntimeCore::default().with_sidecar_store(Arc::new(
        SidecarStore::new(sidecar_root.path()).expect("sidecar store"),
    ));
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_evidence".to_string()),
        thread_id: Some("thread_evidence".to_string()),
        app_id: "content-studio".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_evidence".to_string(),
            turn_id: Some("turn_evidence".to_string()),
            input: AgentInput {
                text: "生成 evidence".to_string(),
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
        "sess_evidence",
        Some("turn_evidence"),
        vec![
            RuntimeEvent::new(
                "message.delta",
                json!({
                    "text": "draft",
                    "evidenceRefs": ["evidence://sess_evidence/runtime"]
                }),
            ),
            RuntimeEvent::new(
                "artifact.snapshot",
                json!({
                    "artifactId": "artifact-report",
                    "path": ".app-server/artifacts/report.md",
                    "content": "# Report"
                }),
            ),
        ],
    )
    .expect("append evidence events");

    let response = core
        .export_evidence(EvidenceExportParams {
            session_id: "sess_evidence".to_string(),
            turn_id: Some("turn_evidence".to_string()),
            include_events: Some(true),
            include_artifacts: Some(true),
            include_evidence_pack: None,
        })
        .await
        .expect("export evidence");

    assert_eq!(response.session.session_id, "sess_evidence");
    assert_eq!(response.turns.len(), 1);
    assert_eq!(response.turns[0].turn_id, "turn_evidence");
    assert_eq!(response.events.len(), 7);
    let user_message = response
        .events
        .iter()
        .find(|event| event.event_type == "message.created")
        .expect("user message event");
    assert_eq!(user_message.payload["input"][0]["text"], "生成 evidence");
    assert!(response
        .events
        .iter()
        .any(|event| event.event_type == "message.delta"));
    assert_eq!(response.artifacts.len(), 1);
    assert_eq!(response.artifacts[0].artifact_ref, "artifact-report");
    assert_eq!(response.artifacts[0].content, None);
    let artifact_event = response
        .events
        .iter()
        .find(|event| event.event_type == "artifact.snapshot")
        .expect("artifact snapshot event");
    assert!(artifact_event.payload["content"].as_str().is_none());
    assert!(artifact_event.payload["sidecarRef"]["sha256"]
        .as_str()
        .is_some_and(|value| value.starts_with("sha256:")));
    assert_eq!(
        response.artifacts[0].content_status,
        ArtifactContentStatus::NotRequested
    );
    assert!(!response.exported_at.is_empty());
    let evidence_pack = response.evidence_pack.expect("basic evidence pack");
    assert_eq!(evidence_pack.thread_status, "running");
    assert_eq!(
        evidence_pack.latest_turn_status.as_deref(),
        Some("accepted")
    );
    assert_eq!(evidence_pack.turn_count, 1);
    assert_eq!(evidence_pack.item_count, 7);
    assert_eq!(evidence_pack.recent_artifact_count, 1);
    assert_eq!(
        evidence_pack
            .completion_audit_summary
            .as_ref()
            .and_then(|summary| summary.get("decision"))
            .and_then(serde_json::Value::as_str),
        Some("in_progress")
    );

    let summary_only = core
        .export_evidence(EvidenceExportParams {
            session_id: "sess_evidence".to_string(),
            turn_id: Some("turn_evidence".to_string()),
            include_events: Some(false),
            include_artifacts: Some(false),
            include_evidence_pack: Some(false),
        })
        .await
        .expect("export summary-only evidence");
    assert_eq!(summary_only.events.len(), 0);
    assert_eq!(summary_only.artifacts.len(), 0);
    assert_eq!(summary_only.turns.len(), 1);
    assert_eq!(summary_only.evidence_pack, None);
}

#[tokio::test]
async fn basic_evidence_pack_downgrades_stale_running_turn() {
    let provider = BasicEvidenceExportProvider;
    let evidence_pack = provider
        .export_evidence_pack(&EvidencePackRequest {
            session: AgentSession {
                session_id: "sess_stale_evidence".to_string(),
                thread_id: "thread_stale_evidence".to_string(),
                app_id: "content-studio".to_string(),
                workspace_id: Some("workspace-main".to_string()),
                business_object_ref: None,
                status: AgentSessionStatus::Running,
                created_at: "2000-01-01T00:00:00Z".to_string(),
                updated_at: "2000-01-01T00:00:00Z".to_string(),
            },
            turns: vec![AgentTurn {
                turn_id: "turn_stale_evidence".to_string(),
                session_id: "sess_stale_evidence".to_string(),
                thread_id: "thread_stale_evidence".to_string(),
                status: AgentTurnStatus::Running,
                started_at: Some("2000-01-01T00:00:00Z".to_string()),
                completed_at: None,
            }],
            events: vec![AgentEvent {
                event_id: "evt_stale_evidence_started".to_string(),
                sequence: 1,
                session_id: "sess_stale_evidence".to_string(),
                thread_id: Some("thread_stale_evidence".to_string()),
                turn_id: Some("turn_stale_evidence".to_string()),
                event_type: "turn.started".to_string(),
                timestamp: "2000-01-01T00:00:00Z".to_string(),
                payload: json!({}),
            }],
            artifacts: Vec::new(),
            turn_runtime_metadata: std::collections::BTreeMap::new(),
            request_logs: Vec::new(),
            workflow_audit_events: Vec::new(),
        })
        .await
        .expect("basic evidence provider")
        .expect("evidence pack");

    assert_eq!(evidence_pack.thread_status, "idle");
    assert_eq!(evidence_pack.latest_turn_status.as_deref(), Some("running"));
    let completion_audit = evidence_pack
        .completion_audit_summary
        .as_ref()
        .expect("completion audit");
    assert_eq!(
        completion_audit
            .get("decision")
            .and_then(serde_json::Value::as_str),
        Some("verifying")
    );
    assert_eq!(
        completion_audit
            .get("runningTurnCount")
            .and_then(serde_json::Value::as_u64),
        Some(0)
    );
}

#[tokio::test]
async fn export_evidence_summarizes_workflow_audit_jsonl_metadata_only() {
    let temp = tempfile::tempdir().expect("tempdir");
    let event_log_writer = Arc::new(EventLogWriter::new(temp.path()).expect("writer"));
    let core = RuntimeCore::default().with_event_log_writer(event_log_writer.clone());
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_workflow_evidence".to_string()),
        thread_id: Some("thread_workflow_evidence".to_string()),
        app_id: "content-studio".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_workflow_evidence".to_string(),
            turn_id: Some("turn_workflow_evidence".to_string()),
            input: AgentInput {
                text: "生成 workflow evidence".to_string(),
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

    let appended = core
        .append_external_runtime_events(
            "sess_workflow_evidence",
            Some("turn_workflow_evidence"),
            vec![
                RuntimeEvent::new(
                    "workflow.run.started",
                    json!({
                        "workflowRunId": "run-article",
                        "workflowKey": "content_article_workflow",
                        "status": "running",
                        "prompt": "写一篇包含敏感素材的文章",
                    }),
                ),
                RuntimeEvent::new(
                    "workflow.step.completed",
                    json!({
                        "workflowRunId": "run-article",
                        "workflowKey": "content_article_workflow",
                        "stepId": "research",
                        "connectorRef": "web-search",
                        "toolName": "WebSearch",
                        "status": "completed",
                        "query": "secret launch plan",
                        "result": {
                            "summary": "raw search result",
                        },
                    }),
                ),
                RuntimeEvent::new(
                    "workflow.run.completed",
                    json!({
                        "workflowRunId": "run-article",
                        "workflowKey": "content_article_workflow",
                        "status": "completed",
                        "providerConfig": {
                            "apiKey": "sk-live-secret",
                        },
                    }),
                ),
            ],
        )
        .expect("append workflow audit runtime events");
    assert!(
        appended.is_empty(),
        "workflow audit events must not enter regular event output: {appended:?}"
    );
    let compaction = event_log_writer
        .compact_session_workflow_audit_events("sess_workflow_evidence", 1)
        .expect("compact workflow audit");
    assert_eq!(compaction.archived_count, 2);
    assert_eq!(compaction.retained_count, 1);

    let audit_records = event_log_writer
        .read_session_workflow_audit_events("sess_workflow_evidence")
        .expect("workflow audit records");
    assert_eq!(audit_records.len(), 3);
    assert_eq!(audit_records[0].event.payload["prompt"]["redacted"], true);
    assert_eq!(audit_records[1].event.payload["query"]["redacted"], true);
    assert_eq!(audit_records[1].event.payload["result"]["redacted"], true);
    assert_eq!(
        audit_records[2].event.payload["providerConfig"]["redacted"],
        true
    );

    let response = core
        .export_evidence(EvidenceExportParams {
            session_id: "sess_workflow_evidence".to_string(),
            turn_id: Some("turn_workflow_evidence".to_string()),
            include_events: Some(true),
            include_artifacts: Some(true),
            include_evidence_pack: Some(true),
        })
        .await
        .expect("export evidence");

    let workflow_audit = response
        .evidence_pack
        .expect("evidence pack")
        .observability_summary
        .expect("observability summary")
        .get("workflow_audit")
        .cloned()
        .expect("workflow audit summary");
    assert_eq!(workflow_audit["status"], "exported");
    assert_eq!(workflow_audit["source"], "workflow-events.jsonl");
    assert_eq!(workflow_audit["eventCount"], json!(3));
    assert_eq!(workflow_audit["metadataOnly"], true);
    assert_eq!(workflow_audit["rawContentIncluded"], false);
    assert_eq!(
        workflow_audit["redactionPolicy"],
        "workflow_audit_metadata_only"
    );
    assert_eq!(workflow_audit["redactionPolicyEventCount"], json!(3));
    assert_eq!(
        workflow_audit["eventTypeBreakdown"]["workflow.run.started"],
        json!(1)
    );
    assert_eq!(
        workflow_audit["eventTypeBreakdown"]["workflow.step.completed"],
        json!(1)
    );
    assert_eq!(
        workflow_audit["eventTypeBreakdown"]["workflow.run.completed"],
        json!(1)
    );
    assert_eq!(workflow_audit["workflowRunIds"], json!(["run-article"]));
    assert_eq!(
        workflow_audit["workflowKeys"],
        json!(["content_article_workflow"])
    );
    assert_eq!(workflow_audit["turnIds"], json!(["turn_workflow_evidence"]));
    assert_eq!(workflow_audit["stepIds"], json!(["research"]));
    assert_eq!(workflow_audit["connectorRefs"], json!(["web-search"]));
    assert_eq!(workflow_audit["toolNames"], json!(["WebSearch"]));

    let workflow_audit_json = serde_json::to_string(&workflow_audit).expect("summary json");
    assert!(!workflow_audit_json.contains("写一篇包含敏感素材的文章"));
    assert!(!workflow_audit_json.contains("secret launch plan"));
    assert!(!workflow_audit_json.contains("raw search result"));
    assert!(!workflow_audit_json.contains("sk-live-secret"));
}

#[tokio::test]
async fn export_evidence_repairs_and_reads_jsonl_projection() {
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
        session_id: Some("sess_evidence_projection".to_string()),
        thread_id: Some("thread_evidence_projection".to_string()),
        app_id: "content-studio".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_evidence_projection".to_string(),
            turn_id: Some("turn_evidence_projection".to_string()),
            input: AgentInput {
                text: "生成 projection evidence".to_string(),
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
        .clear_session("sess_evidence_projection")
        .expect("simulate missing projection");

    let app_data_source = Arc::new(TestSessionDataSource::new());
    let restarted_core = RuntimeCore::default()
        .with_event_log_writer(event_log_writer)
        .with_projection_store(projection_store)
        .with_app_data_source(app_data_source);

    let response = restarted_core
        .export_evidence(EvidenceExportParams {
            session_id: "sess_evidence_projection".to_string(),
            turn_id: Some("turn_evidence_projection".to_string()),
            include_events: Some(true),
            include_artifacts: Some(true),
            include_evidence_pack: Some(true),
        })
        .await
        .expect("export evidence from projection");

    assert_eq!(response.session.session_id, "sess_evidence_projection");
    assert_eq!(response.session.thread_id, "thread_evidence_projection");
    assert_eq!(response.session.status, AgentSessionStatus::Completed);
    assert_eq!(response.turns.len(), 1);
    assert_eq!(response.turns[0].turn_id, "turn_evidence_projection");
    assert_eq!(response.turns[0].status, AgentTurnStatus::Completed);
    assert_eq!(response.events.len(), 8);
    let user_message = response
        .events
        .iter()
        .find(|event| event.event_type == "message.created")
        .expect("user message event");
    assert_eq!(
        user_message.payload["input"][0]["text"],
        "生成 projection evidence"
    );
    assert!(response
        .events
        .iter()
        .any(|event| event.event_type == "message.delta"));
    assert!(response.evidence_pack.is_some());
}
