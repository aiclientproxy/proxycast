use super::*;
use serde_json::Value;

fn canonical_tool_event(event_type: &str, payload: Value) -> RuntimeEvent {
    let call_id = payload["toolCallId"].as_str().expect("tool call id");
    let name = payload["toolName"].as_str().expect("tool name");
    let arguments: Vec<Value> = payload
        .get("arguments")
        .and_then(Value::as_object)
        .map(|arguments| {
            arguments
                .iter()
                .map(|(name, value)| json!({ "name": name, "value": value.to_string() }))
                .collect()
        })
        .unwrap_or_default();
    let structured_content: Option<Value> = payload
        .get("structuredContent")
        .cloned()
        .or_else(|| {
            payload
                .get("result")
                .and_then(|result| result.get("structuredContent"))
                .cloned()
        })
        .or_else(|| payload.get("result").cloned())
        .or_else(|| {
            payload
                .get("metadata")
                .cloned()
                .map(|metadata| json!({ "metadata": metadata }))
        });
    let output = (event_type == "item.completed").then(|| {
        json!({
            "structuredContent": structured_content,
        })
    });
    RuntimeEvent::new(
        event_type,
        json!({
            "item": {
                "sessionId": "evidence-session",
                "threadId": "evidence-thread",
                "turnId": "evidence-turn",
                "itemId": call_id,
                "sequence": 1,
                "ordinal": 1,
                "createdAtMs": 1,
                "updatedAtMs": 2,
                "completedAtMs": (event_type == "item.completed").then_some(2),
                "kind": "tool",
                "status": if event_type == "item.completed" { "completed" } else { "inProgress" },
                "payload": {
                    "type": "tool",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                    "output": output,
                },
                "metadata": payload.get("metadata").cloned().unwrap_or(Value::Null)
            }
        }),
    )
}

#[tokio::test]
async fn basic_evidence_pack_exports_canonical_provenance() {
    let mut request_log = lime_infra::telemetry::RequestLog::new(
        "request-provenance-1".to_string(),
        lime_core::ProviderType::OpenAI,
        "gpt-4.1".to_string(),
        true,
    );
    request_log.session_id = Some("sess_provenance".to_string());
    request_log.thread_id = Some("thread_provenance".to_string());
    request_log.turn_id = Some("turn_provenance".to_string());
    request_log.set_credential_id("secret-credential-id".to_string());
    request_log.set_tokens(Some(12), Some(8));
    request_log.increment_retry();
    request_log.mark_success(125, 200);

    let evidence_pack = BasicEvidenceExportProvider
        .export_evidence_pack(&EvidencePackRequest {
            session: AgentSession {
                session_id: "sess_provenance".to_string(),
                thread_id: "thread_provenance".to_string(),
                app_id: "agent-chat".to_string(),
                workspace_id: None,
                business_object_ref: None,
                status: AgentSessionStatus::Idle,
                created_at: "2026-07-19T00:00:00Z".to_string(),
                updated_at: "2026-07-19T00:00:01Z".to_string(),
            },
            turns: Vec::new(),
            events: vec![
                AgentEvent {
                    event_id: "event-item-1".to_string(),
                    sequence: 1,
                    session_id: "sess_provenance".to_string(),
                    thread_id: Some("thread_provenance".to_string()),
                    turn_id: Some("turn_provenance".to_string()),
                    event_type: "item.completed".to_string(),
                    timestamp: "2026-07-19T00:00:01Z".to_string(),
                    payload: json!({
                        "item": {
                            "sessionId": "sess_provenance",
                            "threadId": "thread_provenance",
                            "turnId": "turn_provenance",
                            "itemId": "item-provenance-1",
                            "sequence": 1,
                            "ordinal": 1,
                            "createdAtMs": 1,
                            "updatedAtMs": 2,
                            "completedAtMs": 2,
                            "kind": "agentMessage",
                            "status": "completed",
                            "payload": {
                                "type": "agentMessage",
                                "text": "canonical answer",
                                "content_parts": []
                            },
                            "metadata": {"source": "runtime"}
                        }
                    }),
                },
                AgentEvent {
                    event_id: "event-provider-1".to_string(),
                    sequence: 2,
                    session_id: "sess_provenance".to_string(),
                    thread_id: Some("thread_provenance".to_string()),
                    turn_id: Some("turn_provenance".to_string()),
                    event_type: "provider.request.started".to_string(),
                    timestamp: "2026-07-19T00:00:02Z".to_string(),
                    payload: json!({
                        "runtimeEvent": {
                            "stage": "request_started",
                            "provider": "openai",
                            "model": "gpt-4.1",
                            "attempt": 2,
                            "status": "running",
                            "runtime_provider_backend": "current",
                            "runtime_provider_selector": "codex",
                            "runtime_provider_protocol": "responses",
                            "runtime_provider_active_model": "gpt-4.1"
                        }
                    }),
                },
            ],
            artifacts: Vec::new(),
            turn_runtime_metadata: [(
                "resolvedRoute".to_string(),
                json!({"protocol": "openai_responses", "source": "profile_slot"}),
            )]
            .into_iter()
            .collect(),
            request_logs: vec![request_log],
            workflow_audit_events: Vec::new(),
        })
        .await
        .expect("basic evidence provider")
        .expect("evidence pack");

    let provenance = evidence_pack
        .observability_summary
        .as_ref()
        .and_then(|summary| summary.get("canonical_provenance"))
        .expect("canonical provenance");
    assert_eq!(provenance["schemaVersion"], "canonical-provenance.v1");
    assert_eq!(provenance["canonicalItemCount"], 1);
    assert_eq!(
        provenance["canonicalItems"][0]["itemId"],
        "item-provenance-1"
    );
    assert_eq!(
        provenance["canonicalItems"][0]["item"]["payload"]["text"],
        "canonical answer"
    );
    assert_eq!(provenance["sequence"]["monotonic"], true);
    assert_eq!(
        provenance["routeMetadata"]["resolvedRoute"]["protocol"],
        "openai_responses"
    );
    assert!(provenance["replayDigest"]
        .as_str()
        .is_some_and(|value| value.starts_with("sha256:")));

    let attempts = provenance["providerAttempts"]
        .as_array()
        .expect("provider attempts");
    assert!(attempts.iter().any(|attempt| {
        attempt["source"] == "provider_trace"
            && attempt["attempt"] == 2
            && attempt["route"]["protocol"] == "responses"
    }));
    let request_attempt = attempts
        .iter()
        .find(|attempt| attempt["source"] == "request_log")
        .expect("request log provenance");
    assert_eq!(request_attempt["retryCount"], 1);
    assert_eq!(request_attempt["inputTokens"], 12);
    assert!(request_attempt["credentialFingerprint"]
        .as_str()
        .is_some_and(|value| value.starts_with("sha256:")));
    assert!(!serde_json::to_string(provenance)
        .expect("serialize provenance")
        .contains("secret-credential-id"));
}

#[tokio::test]
async fn export_evidence_uses_injected_evidence_pack_provider() {
    let provider = Arc::new(TestEvidenceExportProvider::default());
    let core = RuntimeCore::with_backend_capability_source_artifact_content_provider_and_evidence_export_provider(
            Arc::new(MockBackend),
            Arc::new(CapabilityInventorySource::default()),
            Arc::new(InlineArtifactContentProvider),
            provider.clone(),
        );
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
        vec![RuntimeEvent::new(
            "artifact.snapshot",
            json!({
                "artifactId": "artifact-report",
                "path": ".app-server/artifacts/report.md"
            }),
        )],
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

    assert_eq!(provider.call_count.load(Ordering::SeqCst), 1);
    let requests = provider
        .requests
        .lock()
        .expect("test evidence requests mutex poisoned");
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].session.session_id, "sess_evidence");
    assert_eq!(requests[0].turns[0].turn_id, "turn_evidence");
    assert_eq!(requests[0].events.len(), 5);
    assert_eq!(requests[0].artifacts[0].artifact_ref, "artifact-report");

    let evidence_pack = response.evidence_pack.expect("evidence pack");
    assert_eq!(evidence_pack.thread_status, "running");
    assert_eq!(
        evidence_pack.latest_turn_status.as_deref(),
        Some("accepted")
    );
    assert_eq!(evidence_pack.turn_count, 1);
    assert_eq!(evidence_pack.recent_artifact_count, 1);
    assert_eq!(
        evidence_pack
            .completion_audit_summary
            .as_ref()
            .and_then(|summary| summary.get("decision"))
            .and_then(|decision| decision.as_str()),
        Some("in_progress")
    );
}

#[tokio::test]
async fn export_evidence_reads_request_logs_from_telemetry_store() {
    let temp = tempfile::tempdir().expect("tempdir");
    let roots =
        StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");
    let telemetry_store =
        TelemetryStore::initialize(&roots.telemetry_db_path).expect("telemetry store");
    let mut request_log = lime_infra::telemetry::RequestLog::new(
        "request-telemetry-1".to_string(),
        lime_core::ProviderType::OpenAI,
        "gpt-4o".to_string(),
        true,
    );
    request_log.session_id = Some("sess_telemetry_export".to_string());
    request_log.thread_id = Some("thread_telemetry_export".to_string());
    request_log.turn_id = Some("turn_telemetry_export".to_string());
    request_log.mark_success(125, 200);
    telemetry_store
        .upsert_request_log(&request_log)
        .expect("upsert telemetry log");

    let core = RuntimeCore::default()
        .with_telemetry_store(Arc::new(telemetry_store))
        .with_event_log_writer(Arc::new(
            EventLogWriter::new(&roots.event_log_root).expect("writer"),
        ))
        .with_projection_store(Arc::new(
            ProjectionStore::initialize(&roots.projection_db_path).expect("projection"),
        ));
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_telemetry_export".to_string()),
        thread_id: Some("thread_telemetry_export".to_string()),
        app_id: "content-studio".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_telemetry_export".to_string(),
            turn_id: Some("turn_telemetry_export".to_string()),
            input: AgentInput {
                text: "生成 telemetry evidence".to_string(),
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
            session_id: "sess_telemetry_export".to_string(),
            turn_id: Some("turn_telemetry_export".to_string()),
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
    assert_eq!(
        request_telemetry
            .get("status")
            .and_then(serde_json::Value::as_str),
        Some("exported")
    );
    assert_eq!(
        request_telemetry
            .get("requestCount")
            .and_then(serde_json::Value::as_u64),
        Some(1)
    );
    assert_eq!(
        request_telemetry
            .get("sessionRequestCount")
            .and_then(serde_json::Value::as_u64),
        Some(1)
    );
    assert_eq!(
        request_telemetry
            .get("turnRequestCount")
            .and_then(serde_json::Value::as_u64),
        Some(1)
    );
    assert_eq!(
        request_telemetry
            .get("statusBreakdown")
            .and_then(|value| value.get("success"))
            .and_then(serde_json::Value::as_u64),
        Some(1)
    );
    assert_eq!(
        request_telemetry
            .get("statusBreakdown")
            .and_then(|value| value.get("failed"))
            .and_then(serde_json::Value::as_u64),
        Some(0)
    );
    assert_eq!(
        request_telemetry
            .get("statusBreakdown")
            .and_then(|value| value.get("timeout"))
            .and_then(serde_json::Value::as_u64),
        Some(0)
    );
    assert_eq!(
        request_telemetry
            .get("statusBreakdown")
            .and_then(|value| value.get("cancelled"))
            .and_then(serde_json::Value::as_u64),
        Some(0)
    );
}

#[tokio::test]
async fn export_evidence_records_skill_invocation_from_tool_metadata() {
    let core = RuntimeCore::default();
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_skill_invocation_evidence".to_string()),
        thread_id: Some("thread_skill_invocation_evidence".to_string()),
        app_id: "agent-runtime".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_skill_invocation_evidence".to_string(),
            turn_id: Some("turn_skill_invocation_evidence".to_string()),
            input: AgentInput {
                text: "用 $capability-report 生成报告".to_string(),
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
        "sess_skill_invocation_evidence",
        Some("turn_skill_invocation_evidence"),
        vec![
            canonical_tool_event(
                "item.started",
                json!({
                    "toolCallId": "skill-call-1",
                    "toolName": "Skill",
                    "arguments": {
                        "skill": "project:capability-report"
                    }
                }),
            ),
            canonical_tool_event(
                "item.started",
                json!({
                    "toolCallId": "skill-search-call-1",
                    "toolName": "skill_search",
                    "arguments": {
                        "query": "capability report"
                    }
                }),
            ),
            canonical_tool_event(
                "item.completed",
                json!({
                    "toolCallId": "skill-search-call-1",
                    "toolName": "skill_search",
                    "arguments": {
                        "query": "capability report"
                    },
                    "outputPreview": "{\"results\":[]}",
                    "success": true,
                    "metadata": {
                        "tool_family": "skill_search",
                        "skill_search_query": "capability report",
                        "skill_search_snapshot_skill_count": 7,
                        "skill_search_result_count": 2
                    }
                }),
            ),
            canonical_tool_event(
                "item.started",
                json!({
                    "toolCallId": "mcp-search-call-1",
                    "toolName": "mcp__docs__search_docs",
                    "arguments": {
                        "query": "mcp structured content"
                    }
                }),
            ),
            canonical_tool_event(
                "item.completed",
                json!({
                    "toolCallId": "mcp-search-call-1",
                    "toolName": "mcp__docs__search_docs",
                    "success": true,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": "docs found"
                            }
                        ],
                        "structuredContent": {
                            "answer": "ok",
                            "ids": ["doc-1"]
                        },
                        "isError": false
                    }
                }),
            ),
            canonical_tool_event(
                "item.started",
                json!({
                    "toolCallId": "mcp-resource-call-1",
                    "toolName": "ReadMcpResourceTool",
                    "arguments": {
                        "server": "docs",
                        "uri": "file:///docs/intro.md"
                    }
                }),
            ),
            canonical_tool_event(
                "item.completed",
                json!({
                    "toolCallId": "mcp-resource-call-1",
                    "toolName": "ReadMcpResourceTool",
                    "arguments": {
                        "server": "docs",
                        "uri": "file:///docs/intro.md"
                    },
                    "success": true,
                    "result": {
                        "uri": "file:///docs/intro.md",
                        "mime_type": "text/markdown",
                        "text": "# Intro"
                    }
                }),
            ),
            canonical_tool_event(
                "item.completed",
                json!({
                    "toolCallId": "skill-call-1",
                    "toolName": "Skill",
                    "arguments": {
                        "skill": "project:capability-report"
                    },
                    "outputPreview": "报告已生成",
                    "success": true,
                    "metadata": {
                        "tool_family": "skill",
                        "skill_name": "project:capability-report",
                        "workspace_skill_source": {
                            "workspaceRoot": "/tmp/workspace",
                            "source": "manual_session_enable",
                            "approval": "manual",
                            "authorizationScope": "session",
                            "directory": "capability-report",
                            "registeredSkillDirectory": "capability-report",
                            "skillName": "project:capability-report"
                        },
                        "workspace_skill_runtime_enable": {
                            "source": "manual_session_enable",
                            "approval": "manual",
                            "authorization_scope": "session",
                            "workspace_root": "/tmp/workspace",
                            "directory": "capability-report",
                            "skill": "project:capability-report"
                        }
                    }
                }),
            ),
        ],
    )
    .expect("append skill invocation event");

    let response = core
        .export_evidence(EvidenceExportParams {
            session_id: "sess_skill_invocation_evidence".to_string(),
            turn_id: Some("turn_skill_invocation_evidence".to_string()),
            include_events: Some(true),
            include_artifacts: Some(true),
            include_evidence_pack: Some(true),
        })
        .await
        .expect("export evidence");

    let evidence_pack = response.evidence_pack.expect("evidence pack");
    let skill_invocations = evidence_pack
        .observability_summary
        .as_ref()
        .and_then(|summary| summary.get("skill_invocations"))
        .and_then(serde_json::Value::as_array)
        .expect("skill invocations");
    assert_eq!(skill_invocations.len(), 1);
    assert_eq!(
        skill_invocations[0]
            .get("event")
            .and_then(serde_json::Value::as_str),
        Some("skill_invocation")
    );
    assert_eq!(
        skill_invocations[0]
            .get("skillName")
            .and_then(serde_json::Value::as_str),
        Some("project:capability-report")
    );
    assert_eq!(
        skill_invocations[0]
            .get("status")
            .and_then(serde_json::Value::as_str),
        Some("completed")
    );
    assert_eq!(
        skill_invocations[0]
            .get("workspaceSkillRuntimeEnable")
            .and_then(|value| value.get("approval"))
            .and_then(serde_json::Value::as_str),
        Some("manual")
    );
    let skill_searches = evidence_pack
        .observability_summary
        .as_ref()
        .and_then(|summary| summary.get("skill_searches"))
        .and_then(serde_json::Value::as_array)
        .expect("skill searches");
    assert_eq!(skill_searches.len(), 1);
    assert_eq!(
        skill_searches[0]
            .get("event")
            .and_then(serde_json::Value::as_str),
        Some("skill_search")
    );
    assert_eq!(
        skill_searches[0]
            .get("query")
            .and_then(serde_json::Value::as_str),
        Some("capability report")
    );
    assert_eq!(
        skill_searches[0]
            .get("resultCount")
            .and_then(serde_json::Value::as_u64),
        Some(2)
    );
    assert_eq!(
        skill_searches[0]
            .get("snapshotSkillCount")
            .and_then(serde_json::Value::as_u64),
        Some(7)
    );
    assert_eq!(
        skill_searches[0]
            .get("toolCallId")
            .and_then(serde_json::Value::as_str),
        Some("skill-search-call-1")
    );
    let mcp_tool_results = evidence_pack
        .observability_summary
        .as_ref()
        .and_then(|summary| summary.get("mcp_tool_results"))
        .and_then(serde_json::Value::as_array)
        .expect("mcp tool results");
    assert_eq!(mcp_tool_results.len(), 1);
    assert_eq!(
        mcp_tool_results[0]
            .get("event")
            .and_then(serde_json::Value::as_str),
        Some("mcp_tool_result")
    );
    assert_eq!(
        mcp_tool_results[0]
            .get("toolName")
            .and_then(serde_json::Value::as_str),
        Some("mcp__docs__search_docs")
    );
    assert_eq!(
        mcp_tool_results[0]
            .get("hasStructuredContent")
            .and_then(serde_json::Value::as_bool),
        Some(true)
    );
    assert_eq!(
        mcp_tool_results[0]
            .get("structuredContentKeys")
            .and_then(serde_json::Value::as_array)
            .map(|keys| keys.len()),
        Some(2)
    );
    assert_eq!(
        mcp_tool_results[0]
            .get("toolCallId")
            .and_then(serde_json::Value::as_str),
        Some("mcp-search-call-1")
    );
    assert!(mcp_tool_results[0].get("structuredContent").is_none());
    let mcp_resource_reads = evidence_pack
        .observability_summary
        .as_ref()
        .and_then(|summary| summary.get("mcp_resource_reads"))
        .and_then(serde_json::Value::as_array)
        .expect("mcp resource reads");
    assert_eq!(mcp_resource_reads.len(), 1);
    assert_eq!(
        mcp_resource_reads[0]
            .get("event")
            .and_then(serde_json::Value::as_str),
        Some("mcp_resource_read")
    );
    assert_eq!(
        mcp_resource_reads[0]
            .get("server")
            .and_then(serde_json::Value::as_str),
        Some("docs")
    );
    assert_eq!(
        mcp_resource_reads[0]
            .get("uri")
            .and_then(serde_json::Value::as_str),
        Some("file:///docs/intro.md")
    );
    assert_eq!(
        mcp_resource_reads[0]
            .get("mimeTypes")
            .and_then(serde_json::Value::as_array)
            .and_then(|items| items.first())
            .and_then(serde_json::Value::as_str),
        Some("text/markdown")
    );
    assert_eq!(
        mcp_resource_reads[0]
            .get("contentRefs")
            .and_then(serde_json::Value::as_array)
            .and_then(|items| items.first())
            .and_then(|item| item.get("textCharCount"))
            .and_then(serde_json::Value::as_u64),
        Some(7)
    );
    assert!(mcp_resource_reads[0].get("text").is_none());
    let audit = evidence_pack
        .completion_audit_summary
        .as_ref()
        .expect("completion audit");
    assert_eq!(
        audit
            .get("workspaceSkillToolCallCount")
            .and_then(serde_json::Value::as_u64),
        Some(1)
    );
    assert_eq!(
        audit
            .get("requiredEvidence")
            .and_then(|value| value.get("workspaceSkillToolCall"))
            .and_then(serde_json::Value::as_bool),
        Some(true)
    );
}

#[tokio::test]
async fn export_evidence_marks_completed_with_workspace_skill_and_artifact() {
    let core = RuntimeCore::default();
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_skill_completion_evidence".to_string()),
        thread_id: Some("thread_skill_completion_evidence".to_string()),
        app_id: "agent-runtime".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_skill_completion_evidence".to_string(),
            turn_id: Some("turn_skill_completion_evidence".to_string()),
            input: AgentInput {
                text: "运行 workspace skill 并生成报告".to_string(),
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
        "sess_skill_completion_evidence",
        Some("turn_skill_completion_evidence"),
        vec![
            canonical_tool_event(
                "item.started",
                json!({
                    "toolCallId": "skill-call-completion-1",
                    "toolName": "Skill",
                    "arguments": {
                        "skill": "project:capability-report"
                    }
                }),
            ),
            canonical_tool_event(
                "item.completed",
                json!({
                    "toolCallId": "skill-call-completion-1",
                    "toolName": "Skill",
                    "outputPreview": "报告已生成",
                    "success": true,
                    "metadata": {
                        "canonical": false,
                        "compat": true,
                        "source": "legacy_message_tool_response",
                        "sourceType": "tool_end"
                    },
                    "structuredContent": {
                        "tool_family": "skill",
                        "skill_name": "project:capability-report",
                        "workspace_skill_runtime_enable": {
                            "source": "manual_session_enable",
                            "approval": "manual",
                            "authorization_scope": "session",
                            "workspace_root": "/tmp/workspace",
                            "directory": "capability-report",
                            "skill": "project:capability-report"
                        }
                    }
                }),
            ),
            RuntimeEvent::new(
                "artifact.snapshot",
                json!({
                    "artifactId": "artifact-workspace-skill-evidence",
                    "path": "reports/workspace-skill-evidence.md",
                    "title": "Workspace Skill Evidence Report"
                }),
            ),
            RuntimeEvent::new("turn.completed", json!({})),
        ],
    )
    .expect("append skill completion evidence events");

    let response = core
        .export_evidence(EvidenceExportParams {
            session_id: "sess_skill_completion_evidence".to_string(),
            turn_id: Some("turn_skill_completion_evidence".to_string()),
            include_events: Some(true),
            include_artifacts: Some(true),
            include_evidence_pack: Some(true),
        })
        .await
        .expect("export evidence");

    let evidence_pack = response.evidence_pack.expect("evidence pack");
    assert_eq!(
        evidence_pack.latest_turn_status.as_deref(),
        Some("completed")
    );
    assert_eq!(evidence_pack.recent_artifact_count, 1);
    let audit = evidence_pack
        .completion_audit_summary
        .as_ref()
        .expect("completion audit");
    assert_eq!(
        audit.get("decision").and_then(serde_json::Value::as_str),
        Some("completed")
    );
    assert!(audit
        .get("ownerAuditStatuses")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|statuses| statuses
            .iter()
            .any(|status| status.as_str() == Some("audit_input_ready"))));
    assert_eq!(
        audit
            .get("workspaceSkillToolCallCount")
            .and_then(serde_json::Value::as_u64),
        Some(1)
    );
    assert_eq!(
        audit
            .get("artifactCount")
            .and_then(serde_json::Value::as_u64),
        Some(1)
    );
    assert_eq!(
        audit
            .get("requiredEvidence")
            .and_then(|value| value.get("workspaceSkillToolCall"))
            .and_then(serde_json::Value::as_bool),
        Some(true)
    );
}

#[tokio::test]
async fn export_evidence_can_skip_injected_evidence_pack_provider() {
    let provider = Arc::new(TestEvidenceExportProvider::default());
    let core = RuntimeCore::with_backend_capability_source_artifact_content_provider_and_evidence_export_provider(
            Arc::new(MockBackend),
            Arc::new(CapabilityInventorySource::default()),
            Arc::new(InlineArtifactContentProvider),
            provider.clone(),
        );
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_evidence".to_string()),
        thread_id: Some("thread_evidence".to_string()),
        app_id: "content-studio".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    let response = core
        .export_evidence(EvidenceExportParams {
            session_id: "sess_evidence".to_string(),
            turn_id: None,
            include_events: Some(true),
            include_artifacts: Some(true),
            include_evidence_pack: Some(false),
        })
        .await
        .expect("export evidence");

    assert_eq!(provider.call_count.load(Ordering::SeqCst), 0);
    assert_eq!(response.evidence_pack, None);
}

#[tokio::test]
async fn default_runtime_exports_basic_evidence_pack_without_desktop_provider() {
    let core = RuntimeCore::default();
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_basic_evidence".to_string()),
        thread_id: Some("thread_basic_evidence".to_string()),
        app_id: "agent-runtime".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_basic_evidence".to_string(),
            turn_id: Some("turn_basic_evidence".to_string()),
            input: AgentInput {
                text: "生成基础 evidence".to_string(),
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
            session_id: "sess_basic_evidence".to_string(),
            turn_id: None,
            include_events: Some(true),
            include_artifacts: Some(true),
            include_evidence_pack: Some(true),
        })
        .await
        .expect("export evidence");

    let evidence_pack = response.evidence_pack.expect("basic evidence pack");
    assert_eq!(
        evidence_pack.pack_relative_root,
        ".lime/harness/sessions/sess_basic_evidence/evidence"
    );
    assert_eq!(evidence_pack.thread_status, "running");
    assert_eq!(
        evidence_pack
            .completion_audit_summary
            .as_ref()
            .and_then(|summary| summary.get("decision"))
            .and_then(serde_json::Value::as_str),
        Some("in_progress")
    );
    assert_eq!(
        evidence_pack
            .observability_summary
            .as_ref()
            .and_then(|summary| summary.get("source"))
            .and_then(serde_json::Value::as_str),
        Some("app-server-basic")
    );
}
