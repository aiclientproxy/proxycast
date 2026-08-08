use super::*;

#[tokio::test]
async fn article_workspace_reads_embedded_worker_evidence_from_workspace_patch() {
    let core = RuntimeCore::default();
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_worker_evidence_merge".to_string()),
        thread_id: Some("thread_worker_evidence_merge".to_string()),
        app_id: "content-factory-app".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");
    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_worker_evidence_merge".to_string(),
            turn_id: Some("turn_worker_evidence_merge".to_string()),
            input: AgentInput {
                text: "生成文章".to_string(),
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
        "sess_worker_evidence_merge",
        Some("turn_worker_evidence_merge"),
        vec![RuntimeEvent::new(
            "artifact.snapshot",
            json!({
                "artifact": {
                    "artifactId": "task-article:workspace-patch",
                    "artifactRef": "task-article:workspace-patch",
                    "kind": "workspace_patch",
                    "title": "Content Factory workspace patch",
                    "metadata": {
                        "workspacePatch": {
                            "schemaVersion": "article-workspace.v1",
                            "appId": "content-factory-app",
                            "sessionId": "sess_worker_evidence_merge",
                            "objects": [
                                {
                                    "ref": {
                                        "appId": "content-factory-app",
                                        "kind": "articleDraft",
                                        "id": "article-1",
                                        "sessionId": "sess_worker_evidence_merge",
                                        "sourceTaskId": "task-article"
                                    },
                                    "title": "公众号文章草稿",
                                    "status": "needs_review",
                                    "source": {
                                        "taskKind": "content.article.generate",
                                        "taskId": "task-article",
                                        "turnId": "turn_worker_evidence_merge"
                                    }
                                }
                            ],
                            "workerEvidence": [
                                {
                                    "taskId": "task-article",
                                    "taskKind": "content.article.generate",
                                    "turnId": "turn_worker_evidence_merge",
                                    "status": "completed",
                                    "artifactKind": "workspace_patch",
                                    "outputObjectCount": 1,
                                    "workflowKey": "content_article_workflow",
                                    "subagents": ["article-writer"],
                                    "skillRefs": ["article-writing", "article-image-plan"],
                                    "cliRefs": ["content-factory"],
                                    "connectorRefs": ["web-research"],
                                    "hookPolicy": {
                                        "prompt": ["prompt-submit"],
                                        "task": ["task-complete"]
                                    },
                                    "orchestration": [
                                        {
                                            "id": "draft",
                                            "subagent": "article-writer"
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                }
            }),
        )],
    )
    .expect("append worker artifact");

    let detail = core
        .read_session(AgentSessionReadParams {
            session_id: "sess_worker_evidence_merge".to_string(),
            history_limit: Some(20),
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read session")
        .detail
        .expect("detail");
    let worker_evidence = &detail["article_workspace"]["workerEvidence"][0];
    assert_eq!(worker_evidence["taskId"], "task-article");
    assert_eq!(worker_evidence["outputObjectCount"], 1);
    assert_worker_evidence_audit_fields_hidden(worker_evidence);
}

fn assert_worker_evidence_audit_fields_hidden(worker_evidence: &serde_json::Value) {
    for key in [
        "workflowKey",
        "subagents",
        "skillRefs",
        "cliRefs",
        "connectorRefs",
        "hookPolicy",
        "orchestration",
        "workerEntrypoint",
        "inputSummary",
        "outputSummary",
    ] {
        assert!(
            worker_evidence.get(key).is_none(),
            "worker evidence must hide audit-only field {key}"
        );
    }
}
