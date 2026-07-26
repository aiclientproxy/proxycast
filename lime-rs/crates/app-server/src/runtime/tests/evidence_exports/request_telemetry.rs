use super::*;

#[tokio::test]
async fn export_evidence_request_telemetry_ignores_unmatched_session_and_turn_logs() {
    let temp = tempfile::tempdir().expect("tempdir");
    let roots =
        StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
    let telemetry_store =
        TelemetryStore::initialize(&roots.telemetry_db_path).expect("telemetry store");

    let same_session_other_turn = request_log(
        "request-other-turn",
        "sess_request_telemetry_guard",
        "thread_request_telemetry_guard",
        "turn_other",
    );
    telemetry_store
        .upsert_request_log(&same_session_other_turn)
        .expect("upsert same session other turn log");
    let other_session_same_turn = request_log(
        "request-other-session",
        "sess_other",
        "thread_other",
        "turn_request_telemetry_guard",
    );
    telemetry_store
        .upsert_request_log(&other_session_same_turn)
        .expect("upsert other session same turn log");

    let core = RuntimeCore::default()
        .with_telemetry_store(Arc::new(telemetry_store))
        .with_event_log_writer(Arc::new(
            EventLogWriter::new(&roots.event_log_root).expect("writer"),
        ))
        .with_projection_store(Arc::new(
            ProjectionStore::initialize(&roots.projection_db_path).expect("projection"),
        ));
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_request_telemetry_guard".to_string()),
        thread_id: Some("thread_request_telemetry_guard".to_string()),
        app_id: "agent-runtime".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_request_telemetry_guard".to_string(),
            turn_id: Some("turn_request_telemetry_guard".to_string()),
            input: AgentInput {
                text: "导出 requestTelemetry 负向 evidence".to_string(),
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

    let response = core
        .export_evidence(EvidenceExportParams {
            session_id: "sess_request_telemetry_guard".to_string(),
            turn_id: Some("turn_request_telemetry_guard".to_string()),
            include_events: Some(true),
            include_artifacts: Some(true),
            include_evidence_pack: Some(true),
        })
        .await
        .expect("export evidence");

    let evidence_pack = response.evidence_pack.expect("evidence pack");
    let request_telemetry = evidence_pack
        .observability_summary
        .as_ref()
        .and_then(|summary| summary.get("request_telemetry"))
        .expect("request telemetry summary");

    assert_request_telemetry_missing(request_telemetry);
    assert!(!request_telemetry.to_string().contains("unlinked"));
}

fn request_log(
    request_id: &str,
    session_id: &str,
    thread_id: &str,
    turn_id: &str,
) -> lime_infra::telemetry::RequestLog {
    let mut log = lime_infra::telemetry::RequestLog::new(
        request_id.to_string(),
        lime_core::ProviderType::OpenAI,
        "gpt-4o".to_string(),
        true,
    );
    log.session_id = Some(session_id.to_string());
    log.thread_id = Some(thread_id.to_string());
    log.turn_id = Some(turn_id.to_string());
    log.mark_success(100, 200);
    log
}

fn assert_request_telemetry_missing(request_telemetry: &serde_json::Value) {
    assert_eq!(
        request_telemetry
            .get("status")
            .and_then(serde_json::Value::as_str),
        Some("missing")
    );
    assert_eq!(
        request_telemetry
            .get("requestCount")
            .and_then(serde_json::Value::as_u64),
        Some(0)
    );
    assert_eq!(
        request_telemetry
            .get("sessionRequestCount")
            .and_then(serde_json::Value::as_u64),
        Some(0)
    );
    assert_eq!(
        request_telemetry
            .get("turnRequestCount")
            .and_then(serde_json::Value::as_u64),
        Some(0)
    );
}
