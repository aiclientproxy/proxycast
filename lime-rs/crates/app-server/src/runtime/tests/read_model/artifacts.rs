use super::*;
use serde_json::Value;

fn assert_worker_evidence_audit_fields_hidden(worker_evidence: &Value) {
    for key in [
        "workerEntrypoint",
        "inputSummary",
        "outputSummary",
        "workflowKey",
        "subagents",
        "skillRefs",
        "cliRefs",
        "connectorRefs",
        "hookPolicy",
        "runtimeRegistries",
        "orchestration",
    ] {
        assert!(
            worker_evidence.get(key).is_none(),
            "worker evidence must hide audit-only field {key}"
        );
    }
}

fn article_workspace_search_snapshot_payload(search_evidence: Value) -> Value {
    let legacy_host_search_evidence = search_evidence.clone();
    let host_tool_evidence = search_evidence.clone();

    let mut article_source = serde_json::Map::new();
    article_source.insert("taskKind".to_string(), json!("content.article.generate"));
    article_source.insert("taskId".to_string(), json!("task-article-1"));
    article_source.insert("turnId".to_string(), json!("turn_article_workspace"));
    article_source.insert("artifactIds".to_string(), json!(["artifact-article-1"]));
    article_source.insert(
        "hostToolRequests".to_string(),
        json!([
            {
                "id": "search-request-1",
                "toolName": "WebSearch",
                "params": {
                    "query": "Lime 写文章"
                },
                "query": "Lime 写文章",
                "purpose": "验证宿主真实检索回填"
            }
        ]),
    );
    article_source.insert("searchEvidence".to_string(), search_evidence);
    article_source.insert(
        "hostSearchEvidence".to_string(),
        legacy_host_search_evidence,
    );
    article_source.insert("hostSearchStatus".to_string(), json!("completed"));
    article_source.insert("hostToolEvidence".to_string(), host_tool_evidence);
    article_source.insert("hostToolStatus".to_string(), json!("completed"));
    article_source.insert(
        "documentText".to_string(),
        json!("# 公众号文章草稿\n\n这是正文。"),
    );
    article_source.insert(
        "finalMarkdown".to_string(),
        json!("# 公众号文章草稿\n\n这是正文。"),
    );
    article_source.insert(
        "researchRounds".to_string(),
        json!([
            {
                "id": "research-1",
                "title": "检索行业背景"
            }
        ]),
    );
    article_source.insert(
        "outline".to_string(),
        json!([
            {
                "id": "intro",
                "title": "开场：为什么要把写作变成工作流"
            }
        ]),
    );
    article_source.insert(
        "imageSlots".to_string(),
        json!([
            {
                "id": "hero",
                "title": "首图",
                "prompt": "桌面端内容工厂写作流程图，中文标签"
            }
        ]),
    );
    article_source.insert("evidenceIds".to_string(), json!(["evidence-1"]));

    let mut article_ref = serde_json::Map::new();
    article_ref.insert("appId".to_string(), json!("content-factory-app"));
    article_ref.insert("kind".to_string(), json!("articleDraft"));
    article_ref.insert("id".to_string(), json!("article-1"));
    article_ref.insert("sessionId".to_string(), json!("sess_article_workspace"));
    article_ref.insert("artifactIds".to_string(), json!(["artifact-article-1"]));
    article_ref.insert("sourceTurnId".to_string(), json!("turn_article_workspace"));

    let mut article_object = serde_json::Map::new();
    article_object.insert("ref".to_string(), Value::Object(article_ref));
    article_object.insert("title".to_string(), json!("公众号文章草稿"));
    article_object.insert("status".to_string(), json!("ready"));
    article_object.insert("summary".to_string(), json!("已生成首版文章"));
    article_object.insert("previewArtifactId".to_string(), json!("artifact-article-1"));
    article_object.insert("source".to_string(), Value::Object(article_source));

    let mut image_ref = serde_json::Map::new();
    image_ref.insert("appId".to_string(), json!("content-factory-app"));
    image_ref.insert("kind".to_string(), json!("imageGenerationSet"));
    image_ref.insert("id".to_string(), json!("image-set-1"));
    image_ref.insert("sessionId".to_string(), json!("sess_article_workspace"));
    image_ref.insert("artifactIds".to_string(), json!(["artifact-image-1"]));
    image_ref.insert("sourceTurnId".to_string(), json!("turn_article_workspace"));

    let mut image_source = serde_json::Map::new();
    image_source.insert("taskKind".to_string(), json!("content.image.generate"));
    image_source.insert("taskId".to_string(), json!("task-image-1"));
    image_source.insert("turnId".to_string(), json!("turn_article_workspace"));
    image_source.insert("artifactIds".to_string(), json!(["artifact-image-1"]));
    image_source.insert("evidenceIds".to_string(), json!(["evidence-2"]));

    let mut image_object = serde_json::Map::new();
    image_object.insert("ref".to_string(), Value::Object(image_ref));
    image_object.insert("title".to_string(), json!("配图组"));
    image_object.insert("status".to_string(), json!("needs_review"));
    image_object.insert("summary".to_string(), json!("等待选择主图"));
    image_object.insert("previewArtifactId".to_string(), json!("artifact-image-1"));
    image_object.insert("source".to_string(), Value::Object(image_source));

    let mut patch = serde_json::Map::new();
    patch.insert("schemaVersion".to_string(), json!(1));
    patch.insert("appId".to_string(), json!("content-factory-app"));
    patch.insert("sessionId".to_string(), json!("sess_article_workspace"));
    patch.insert(
        "primaryObjectRef".to_string(),
        json!({
            "appId": "content-factory-app",
            "kind": "articleDraft",
            "id": "article-1",
            "sessionId": "sess_article_workspace",
            "artifactIds": ["artifact-article-1"],
            "sourceTurnId": "turn_article_workspace"
        }),
    );
    patch.insert(
        "selectedObjectRef".to_string(),
        json!({
            "appId": "content-factory-app",
            "kind": "articleDraft",
            "id": "article-1",
            "sessionId": "sess_article_workspace"
        }),
    );
    patch.insert(
        "objects".to_string(),
        Value::Array(vec![
            Value::Object(article_object),
            Value::Object(image_object),
        ]),
    );
    patch.insert(
        "layoutState".to_string(),
        json!({
            "activeTabKind": "articleWorkspace",
            "activePaneKind": "documentCanvas",
            "openTabKinds": ["articleWorkspace", "files"],
            "splitMode": "chat-right-dock"
        }),
    );

    let mut worker = serde_json::Map::new();
    worker.insert("appId".to_string(), json!("content-factory-app"));
    worker.insert("taskId".to_string(), json!("task-article-1"));
    worker.insert("taskKind".to_string(), json!("content.article.generate"));
    worker.insert("turnId".to_string(), json!("turn_article_workspace"));
    worker.insert(
        "workerEntrypoint".to_string(),
        json!("./runtime/content-factory-worker.mjs"),
    );
    worker.insert("status".to_string(), json!("completed"));
    worker.insert(
        "inputSummary".to_string(),
        json!("prompt=生成文章; inputKeys=topic"),
    );
    worker.insert(
        "outputSummary".to_string(),
        json!("2 objects: 公众号文章草稿, 配图组"),
    );
    worker.insert("outputObjectCount".to_string(), json!(2));
    worker.insert("outputArtifactKind".to_string(), json!("workspace_patch"));
    worker.insert("artifactKind".to_string(), json!("workspace_patch"));
    worker.insert("workflowKey".to_string(), json!("content_article_workflow"));
    worker.insert(
        "subagents".to_string(),
        json!(["content-researcher", "article-writer", "image-planner"]),
    );
    worker.insert(
        "skillRefs".to_string(),
        json!(["article-research", "article-writing", "article-image-plan"]),
    );
    worker.insert("cliRefs".to_string(), json!(["content-factory"]));
    worker.insert(
        "connectorRefs".to_string(),
        json!(["lime-knowledge", "web-research", "media-generation"]),
    );
    worker.insert(
        "hookPolicy".to_string(),
        json!({
            "prompt": ["prompt-submit"],
            "task": ["task-complete"]
        }),
    );
    worker.insert(
        "orchestration".to_string(),
        json!([
            {
                "id": "research",
                "title": "资料检索",
                "subagent": "content-researcher",
                "skillRefs": ["article-research"],
                "status": "completed",
                "summary": "整理资料"
            },
            {
                "id": "draft",
                "title": "正文写作",
                "subagent": "article-writer",
                "skillRefs": ["article-writing"],
                "status": "completed",
                "summary": "生成文章草稿"
            },
            {
                "id": "image-plan",
                "title": "配图规划",
                "subagent": "image-planner",
                "skillRefs": ["article-image-plan"],
                "status": "completed",
                "summary": "生成配图规划"
            }
        ]),
    );
    patch.insert(
        "workerEvidence".to_string(),
        Value::Array(vec![
            Value::Object(worker),
            json!({
                "taskId": "task-image-1",
                "taskKind": "content.image.generate",
                "turnId": "turn_article_workspace",
                "status": "failed",
                "source": "workspace_patch",
                "artifactKind": "workspace_patch",
                "errorCode": "worker_invalid_json_output"
            }),
        ]),
    );

    let mut metadata = serde_json::Map::new();
    metadata.insert("workspacePatch".to_string(), Value::Object(patch));

    let mut artifact = serde_json::Map::new();
    artifact.insert(
        "artifactId".to_string(),
        json!("artifact-workspace-patch-1"),
    );
    artifact.insert(
        "path".to_string(),
        json!(".lime/artifacts/workspace-patch.json"),
    );
    artifact.insert("title".to_string(), json!("内容工厂工作区补丁"));
    artifact.insert("kind".to_string(), json!("workspace_patch"));
    artifact.insert("status".to_string(), json!("ready"));
    artifact.insert("metadata".to_string(), Value::Object(metadata));

    let mut payload = serde_json::Map::new();
    payload.insert("artifact".to_string(), Value::Object(artifact));
    Value::Object(payload)
}

fn article_workspace_selection_snapshot_payload() -> Value {
    json!({
        "artifact": {
            "artifactId": "artifact-workspace-selection",
            "kind": "article_workspace_selection",
            "metadata": {
                "workspacePatch": {
                    "schemaVersion": "article-workspace.v1",
                    "appId": "content-factory-app",
                    "sessionId": "sess_article_workspace",
                    "selectedObjectRef": {
                        "appId": "content-factory-app",
                        "kind": "imageGenerationSet",
                        "id": "image-set-1",
                        "sessionId": "sess_article_workspace"
                    },
                    "objects": [{
                        "ref": {
                            "appId": "content-factory-app",
                            "kind": "imageGenerationSet",
                            "id": "image-set-1",
                            "sessionId": "sess_article_workspace"
                        }
                    }]
                }
            }
        }
    })
}

fn article_workspace_edited_draft_snapshot_payload(markdown: &str, updated_at: &str) -> Value {
    let object_ref = json!({
        "appId": "content-factory-app",
        "kind": "articleDraft",
        "id": "article-1",
        "sessionId": "sess_article_workspace",
        "artifactIds": ["artifact-article-1"],
        "sourceTurnId": "turn_article_workspace"
    });
    let object_key = "content-factory-app:sess_article_workspace:articleDraft:article-1";
    json!({
        "artifact": {
            "artifactId": "artifact-article-1",
            "artifactRef": "artifact-article-1",
            "artifactDocumentId": "article-1",
            "title": "公众号文章草稿",
            "kind": "article_workspace_draft",
            "status": "draft",
            "metadata": {
                "workspacePatch": {
                    "schemaVersion": "article-workspace.v1",
                    "appId": "content-factory-app",
                    "sessionId": "sess_article_workspace",
                    "objects": [{
                        "ref": object_ref,
                        "title": "公众号文章草稿",
                        "status": "draft",
                        "source": {
                            "documentText": markdown,
                            "finalMarkdown": markdown,
                            "edited": true,
                            "updatedAt": updated_at
                        }
                    }],
                    "updatedAt": updated_at
                },
                "editedDraft": {
                    "objectKey": object_key,
                    "objectRef": object_ref,
                    "markdown": markdown,
                    "updatedAt": updated_at
                }
            }
        }
    })
}

#[tokio::test]
async fn read_session_keeps_workspace_patch_in_thread_read_artifacts_only() {
    let core = RuntimeCore::default();
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_thread_read_artifacts".to_string()),
        thread_id: Some("thread_read_artifacts".to_string()),
        app_id: "content-studio".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    let turn = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_thread_read_artifacts".to_string(),
                turn_id: Some("turn_thread_read_artifacts".to_string()),
                input: AgentInput {
                    text: "生成内容工厂产物".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("turn")
        .response
        .turn;
    core.append_external_runtime_events(
        "sess_thread_read_artifacts",
        Some(&turn.turn_id),
        vec![RuntimeEvent::new(
            "artifact.snapshot",
            json!({
                "artifact": {
                    "artifactId": "artifact-content-batch",
                    "path": ".lime/artifacts/content-batch.json",
                    "title": "Content Batch",
                    "kind": "workspace_patch",
                    "status": "ready",
                    "metadata": {
                        "workspacePatch": {
                            "kind": "content_batch",
                            "contentBatch": {
                                "count": 1
                            }
                        }
                    }
                }
            }),
        )],
    )
    .expect("append artifact event");

    let read = core
        .read_session(AgentSessionReadParams {
            session_id: "sess_thread_read_artifacts".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read session");
    let detail = read.detail.expect("session detail");
    let artifacts = detail["thread_read"]["artifacts"]
        .as_array()
        .expect("thread read artifacts");

    assert!(
        detail["artifacts"]
            .as_array()
            .expect("user visible artifacts")
            .is_empty(),
        "top-level user visible artifacts should hide internal workspace patches"
    );
    assert_eq!(artifacts.len(), 1);
    assert_eq!(artifacts[0]["artifactRef"], "artifact-content-batch");
    assert_eq!(artifacts[0]["path"], ".lime/artifacts/content-batch.json");
    assert_eq!(artifacts[0]["kind"], "workspace_patch");
    assert_eq!(artifacts[0]["status"], "ready");
    assert_eq!(
        artifacts[0]["metadata"]["workspacePatch"]["kind"],
        "content_batch"
    );
    assert!(artifacts[0]["content"].is_null());
    assert_eq!(artifacts[0]["contentStatus"], "notRequested");

    let artifact_read = core
        .read_artifacts(ArtifactReadParams {
            session_id: "sess_thread_read_artifacts".to_string(),
            turn_id: Some("turn_thread_read_artifacts".to_string()),
            artifact_ref: Some("artifact-content-batch".to_string()),
            include_content: Some(false),
            cursor: None,
            limit: Some(1),
        })
        .expect("read internal workspace patch artifact");
    assert_eq!(artifact_read.artifacts.len(), 1);
    assert_eq!(
        artifact_read.artifacts[0].artifact_ref,
        "artifact-content-batch"
    );
    assert_eq!(
        artifact_read.artifacts[0].path.as_deref(),
        Some(".lime/artifacts/content-batch.json")
    );
    assert_eq!(
        artifact_read.artifacts[0].kind.as_deref(),
        Some("workspace_patch")
    );
    assert_eq!(
        artifact_read.artifacts[0]
            .metadata
            .as_ref()
            .expect("workspace patch metadata")["workspacePatch"]["kind"],
        "content_batch"
    );
}

#[tokio::test]
async fn article_workspace_artifact_documents_merge_version_history_across_turns() {
    let core = RuntimeCore::default();
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_product_artifact_versions".to_string()),
        thread_id: Some("thread_product_artifact_versions".to_string()),
        app_id: "content-factory-app".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    for (version, body) in [
        ("1", "第一版正文"),
        ("2", "第二版正文，包含编辑后的交付内容"),
    ] {
        let turn_id = format!("turn_article_v{version}");
        let turn = core
            .start_turn(
                AgentSessionTurnStartParams {
                    session_id: "sess_product_artifact_versions".to_string(),
                    turn_id: Some(turn_id.clone()),
                    input: AgentInput {
                        text: format!("生成文章草稿 v{version}"),
                        attachments: Vec::new(),
                    },
                    runtime_options: None,
                    queue_if_busy: false,
                    skip_pre_submit_resume: false,
                },
                RuntimeHostContext::default(),
            )
            .await
            .expect("turn")
            .response
            .turn;

        core.append_external_runtime_events(
            "sess_product_artifact_versions",
            Some(&turn.turn_id),
            vec![
                RuntimeEvent::new(
                    "artifact.snapshot",
                    json!({
                        "artifact": {
                            "artifactId": format!("artifact-workspace-patch-v{version}"),
                            "path": format!(".lime/artifacts/content-factory-workspace-patch-v{version}.json"),
                            "title": format!("内容工厂工作区补丁 v{version}"),
                            "kind": "workspace_patch",
                            "status": "ready",
                            "metadata": {
                                "workspacePatch": {
                                    "schemaVersion": 1,
                                    "appId": "content-factory-app",
                                    "sessionId": "sess_product_artifact_versions",
                                    "objects": [
                                        {
                                            "ref": {
                                                "appId": "content-factory-app",
                                                "kind": "articleDraft",
                                                "id": "article-1",
                                                "sessionId": "sess_product_artifact_versions",
                                                "artifactIds": ["artifact-article-1"],
                                                "sourceTurnId": turn_id,
                                                "version": version
                                            },
                                            "title": format!("公众号文章草稿 v{version}"),
                                            "status": "ready",
                                            "summary": format!("文章草稿第 {version} 版"),
                                            "previewArtifactId": "artifact-article-1",
                                            "source": {
                                                "taskKind": "content.article.generate",
                                                "taskId": format!("task-article-v{version}"),
                                                "turnId": turn.turn_id,
                                                "artifactIds": ["artifact-article-1"],
                                                "markdown": body
                                            }
                                        }
                                    ],
                                    "layoutState": {
                                        "activeTabKind": "articleWorkspace",
                                        "openTabKinds": ["articleWorkspace"]
                                    }
                                }
                            }
                        }
                    }),
                ),
                RuntimeEvent::new("turn.completed", json!({})),
            ],
        )
        .expect("append artifact event");
    }

    let read = core
        .read_session(AgentSessionReadParams {
            session_id: "sess_product_artifact_versions".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read session");
    let detail = read.detail.expect("session detail");
    let artifacts = detail["thread_read"]["artifacts"]
        .as_array()
        .expect("thread read artifacts");
    assert!(
        !detail["artifacts"]
            .as_array()
            .expect("user visible artifacts")
            .iter()
            .any(|artifact| artifact["kind"] == "workspace_patch"),
        "workspace patch must stay out of top-level user-visible artifacts"
    );
    assert!(
        artifacts.iter().any(
            |artifact| artifact["artifactRef"] == "artifact-workspace-patch-v1"
                && artifact["kind"] == "workspace_patch"
        ),
        "thread read artifacts must keep workspace patch for Article Editor projection"
    );
    let article_artifact = artifacts
        .iter()
        .find(|artifact| artifact["artifactRef"] == "artifact-article-1")
        .expect("article artifact document");

    assert_eq!(
        article_artifact["metadata"]["artifactDocument"]["metadata"]["currentVersionNo"],
        2
    );
    assert_eq!(
        article_artifact["metadata"]["artifactDocument"]["metadata"]["versionHistory"]
            .as_array()
            .expect("version history")
            .len(),
        2
    );
    assert_eq!(
        article_artifact["metadata"]["artifactDocument"]["metadata"]["versionHistory"][0]["id"],
        "artifact-document:content-factory-app:artifact-article-1:v1"
    );
    assert_eq!(
        article_artifact["metadata"]["artifactDocument"]["metadata"]["versionHistory"][1]["id"],
        "artifact-document:content-factory-app:artifact-article-1:v2"
    );

    let artifact_read = core
        .read_artifacts(ArtifactReadParams {
            session_id: "sess_product_artifact_versions".to_string(),
            turn_id: None,
            artifact_ref: Some("artifact-article-1".to_string()),
            include_content: Some(true),
            cursor: None,
            limit: Some(1),
        })
        .expect("read merged artifact document");
    assert_eq!(artifact_read.artifacts.len(), 1);
    assert_eq!(
        artifact_read.artifacts[0].content_status,
        ArtifactContentStatus::Available
    );
    let content: serde_json::Value = serde_json::from_str(
        artifact_read.artifacts[0]
            .content
            .as_deref()
            .expect("artifact document content"),
    )
    .expect("artifact document json");

    assert_eq!(content["metadata"]["currentVersionNo"], 2);
    assert_eq!(
        content["metadata"]["versionHistory"]
            .as_array()
            .expect("content version history")
            .len(),
        2
    );
    assert_eq!(
        content["blocks"][0]["content"],
        "第二版正文，包含编辑后的交付内容"
    );
}

#[tokio::test]
async fn artifact_workbench_save_snapshot_merges_with_article_workspace_artifact_document_history()
{
    let sidecar_root = tempfile::tempdir().expect("sidecar root");
    let core = RuntimeCore::default().with_sidecar_store(Arc::new(
        SidecarStore::new(sidecar_root.path()).expect("sidecar store"),
    ));
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_product_artifact_workbench_save".to_string()),
        thread_id: Some("thread_product_artifact_workbench_save".to_string()),
        app_id: "content-factory-app".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    let first_turn = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_product_artifact_workbench_save".to_string(),
                turn_id: Some("turn_article_generated".to_string()),
                input: AgentInput {
                    text: "生成文章草稿".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("first turn")
        .response
        .turn;

    core.append_external_runtime_events(
        "sess_product_artifact_workbench_save",
        Some(&first_turn.turn_id),
        vec![
            RuntimeEvent::new(
                "artifact.snapshot",
                json!({
                    "artifact": {
                        "artifactId": "artifact-workspace-patch-generated",
                        "path": ".lime/artifacts/content-factory-workspace-patch-generated.json",
                        "title": "内容工厂工作区补丁",
                        "kind": "workspace_patch",
                        "status": "ready",
                        "metadata": {
                            "workspacePatch": {
                                "schemaVersion": 1,
                                "appId": "content-factory-app",
                                "sessionId": "sess_product_artifact_workbench_save",
                                "objects": [
                                    {
                                        "ref": {
                                            "appId": "content-factory-app",
                                            "kind": "articleDraft",
                                            "id": "article-1",
                                            "sessionId": "sess_product_artifact_workbench_save",
                                            "artifactIds": ["artifact-article-1"],
                                            "sourceTurnId": "turn_article_generated",
                                            "version": "1"
                                        },
                                        "title": "公众号文章草稿",
                                        "status": "ready",
                                        "summary": "首版文章草稿",
                                        "previewArtifactId": "artifact-article-1",
                                        "source": {
                                            "taskKind": "content.article.generate",
                                            "taskId": "task-article-v1",
                                            "turnId": "turn_article_generated",
                                            "artifactIds": ["artifact-article-1"],
                                            "markdown": "生成的首版正文"
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }),
            ),
            RuntimeEvent::new("turn.completed", json!({})),
        ],
    )
    .expect("append generated artifact event");

    let saved_turn = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_product_artifact_workbench_save".to_string(),
                turn_id: Some("turn_article_workbench_save".to_string()),
                input: AgentInput {
                    text: "保存 Artifact Workbench 编辑稿".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("save turn")
        .response
        .turn;

    let saved_document = json!({
        "schemaVersion": "artifact_document.v1",
        "artifactId": "artifact-document:content-factory-app:artifact-article-1",
        "workspaceId": "workspace-main",
        "threadId": null,
        "turnId": "turn_article_workbench_save",
        "kind": "report",
        "title": "公众号文章草稿",
        "status": "ready",
        "language": "zh-CN",
        "summary": "用户在 Artifact Workbench 保存的第二版",
        "blocks": [
            {
                "id": "body",
                "type": "rich_text",
                "contentFormat": "markdown",
                "content": "用户编辑后的第二版正文",
                "markdown": "用户编辑后的第二版正文"
            }
        ],
        "sources": [],
        "metadata": {
            "generatedBy": "user",
            "currentVersionId": "artifact-document:content-factory-app:artifact-article-1:v2",
            "currentVersionNo": 2,
            "versionHistory": [
                {
                    "id": "artifact-document:content-factory-app:artifact-article-1:v2",
                    "artifactId": "artifact-document:content-factory-app:artifact-article-1",
                    "versionNo": 2,
                    "title": "公众号文章草稿",
                    "status": "ready",
                    "createdBy": "user"
                }
            ],
            "articleWorkspace": {
                "appId": "content-factory-app",
                "sessionId": "sess_product_artifact_workbench_save",
                "workspaceId": "workspace-main",
                "objectKind": "articleDraft",
                "objectId": "article-1",
                "artifactIds": ["artifact-article-1"]
            }
        }
    });
    let saved_content =
        serde_json::to_string_pretty(&saved_document).expect("saved document content");

    core.append_external_runtime_events(
        "sess_product_artifact_workbench_save",
        Some(&saved_turn.turn_id),
        vec![
            RuntimeEvent::new(
                "artifact.snapshot",
                json!({
                    "artifact": {
                        "artifactId": "artifact-article-1",
                        "artifactRef": "artifact-article-1",
                        "artifactDocumentId": "artifact-document:content-factory-app:artifact-article-1",
                        "filePath": "article.md",
                        "path": "article.md",
                        "title": "公众号文章草稿",
                        "kind": "artifact_document",
                        "status": "ready",
                        "content": saved_content,
                        "metadata": {
                            "artifactSchema": "artifact_document.v1",
                            "artifactKind": "report",
                            "artifactDocument": saved_document,
                            "artifactTitle": "公众号文章草稿",
                            "artifactDocumentId": "artifact-document:content-factory-app:artifact-article-1",
                            "artifactVersionId": "artifact-document:content-factory-app:artifact-article-1:v2",
                            "artifactVersionNo": 2,
                            "artifactRef": "artifact-article-1",
                            "filePath": "article.md",
                            "articleWorkspace": {
                                "appId": "content-factory-app",
                                "sessionId": "sess_product_artifact_workbench_save",
                                "workspaceId": "workspace-main",
                                "objectKind": "articleDraft",
                                "objectId": "article-1",
                                "artifactIds": ["artifact-article-1"]
                            }
                        }
                    }
                }),
            ),
            RuntimeEvent::new("turn.completed", json!({})),
        ],
    )
    .expect("append saved artifact event");

    let artifact_read = core
        .read_artifacts(ArtifactReadParams {
            session_id: "sess_product_artifact_workbench_save".to_string(),
            turn_id: None,
            artifact_ref: Some("artifact-article-1".to_string()),
            include_content: Some(true),
            cursor: None,
            limit: Some(1),
        })
        .expect("read saved artifact document");
    assert_eq!(artifact_read.artifacts.len(), 1);
    assert_eq!(
        artifact_read.artifacts[0].content_status,
        ArtifactContentStatus::Available
    );
    let content: serde_json::Value = serde_json::from_str(
        artifact_read.artifacts[0]
            .content
            .as_deref()
            .expect("saved artifact document content"),
    )
    .expect("saved artifact document json");

    assert_eq!(content["metadata"]["currentVersionNo"], 2);
    assert_eq!(content["blocks"][0]["content"], "用户编辑后的第二版正文");
    assert_eq!(
        content["metadata"]["versionHistory"]
            .as_array()
            .expect("merged version history")
            .len(),
        2
    );
    assert_eq!(
        content["metadata"]["versionHistory"][0]["id"],
        "artifact-document:content-factory-app:artifact-article-1:v1"
    );
    assert_eq!(
        content["metadata"]["versionHistory"][1]["id"],
        "artifact-document:content-factory-app:artifact-article-1:v2"
    );
}

#[tokio::test]
async fn read_session_materializes_workspace_patch_into_article_workspace() {
    let core = RuntimeCore::default();
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_article_workspace".to_string()),
        thread_id: Some("thread_article_workspace".to_string()),
        app_id: "content-factory-app".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    let turn = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_article_workspace".to_string(),
                turn_id: Some("turn_article_workspace".to_string()),
                input: AgentInput {
                    text: "生成文章草稿".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("turn")
        .response
        .turn;

    core.append_external_runtime_events(
        "sess_article_workspace",
        Some(&turn.turn_id),
        vec![
            RuntimeEvent::new(
                "artifact.snapshot",
                article_workspace_search_snapshot_payload(json!([
                    {
                        "id": "host-tool-evidence-search-request-1",
                        "requestId": "search-request-1",
                        "tool": "WebSearch",
                        "toolCallId": "workspace-patch-host-tool-websearch-search-request-1",
                        "source": "workspace_patch_host_tool_requests",
                        "status": "completed",
                        "query": "Lime 写文章",
                        "purpose": "验证宿主真实检索回填",
                        "summary": "session=sess_article_workspace query=Lime 写文章 result=found",
                        "output": "session=sess_article_workspace query=Lime 写文章 result=found",
                        "error": null,
                        "confidence": "host_verified"
                    }
                ])),
            ),
            RuntimeEvent::new("turn.completed", json!({})),
        ],
    )
    .expect("append artifact event");

    let read = core
        .read_session(AgentSessionReadParams {
            session_id: "sess_article_workspace".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read session");
    let detail = read.detail.expect("session detail");
    let article_workspace = &detail["article_workspace"];

    assert_eq!(article_workspace["schemaVersion"], "article-workspace.v1");
    assert_eq!(article_workspace["appId"], "content-factory-app");
    assert_eq!(article_workspace["sessionId"], "sess_article_workspace");
    assert_eq!(article_workspace["workspaceId"], "workspace-main");
    assert_eq!(article_workspace["objectCount"], 2);
    assert_eq!(
        article_workspace["primaryObjectRef"]["kind"],
        "articleDraft"
    );
    assert_eq!(article_workspace["selectedObjectRef"]["id"], "article-1");
    assert_eq!(article_workspace["objects"][0]["title"], "公众号文章草稿");
    assert_eq!(article_workspace["layoutState"]["openTabKinds"][1], "files");
    assert_eq!(
        article_workspace["objects"][0]["source"]["searchEvidence"][0]["tool"],
        "WebSearch"
    );
    assert_eq!(
        article_workspace["objects"][0]["source"]["searchEvidence"][0]["status"],
        "completed"
    );
    assert_eq!(
        article_workspace["objects"][0]["source"]["hostSearchStatus"],
        "completed"
    );
    assert_eq!(
        article_workspace["objects"][0]["source"]["hostSearchEvidence"],
        article_workspace["objects"][0]["source"]["searchEvidence"]
    );
    assert_eq!(
        article_workspace["objects"][0]["source"]["hostToolStatus"],
        "completed"
    );
    assert_eq!(
        article_workspace["objects"][0]["source"]["hostToolEvidence"],
        article_workspace["objects"][0]["source"]["searchEvidence"]
    );
    assert_eq!(
        article_workspace["sourceArtifacts"][0]["artifactRef"],
        "artifact-workspace-patch-1"
    );
    let completed_worker_evidence = &article_workspace["workerEvidence"][0];
    assert_eq!(completed_worker_evidence["taskId"], "task-article-1");
    assert_eq!(completed_worker_evidence["status"], "completed");
    assert_eq!(completed_worker_evidence["outputObjectCount"], 2);
    assert_worker_evidence_audit_fields_hidden(completed_worker_evidence);
    assert_eq!(
        article_workspace["workerEvidence"][1]["taskId"],
        "task-image-1"
    );
    assert_eq!(article_workspace["workerEvidence"][1]["status"], "failed");
    assert_eq!(
        article_workspace["workerEvidence"][1]["errorCode"],
        "worker_invalid_json_output"
    );
    assert_worker_evidence_audit_fields_hidden(&article_workspace["workerEvidence"][1]);
    assert_eq!(
        detail["thread_read"]["article_workspace"],
        detail["article_workspace"]
    );
    assert_eq!(
        detail["thread_read"]["articleWorkspace"],
        detail["articleWorkspace"]
    );
    let artifacts = detail["thread_read"]["artifacts"]
        .as_array()
        .expect("thread read artifacts");
    let article_artifact = artifacts
        .iter()
        .find(|artifact| artifact["artifactRef"] == "artifact-article-1")
        .expect("article product object artifact document");
    assert_eq!(article_artifact["kind"], "artifact_document");
    assert_eq!(article_artifact["title"], "公众号文章草稿");
    assert_eq!(
        article_artifact["metadata"]["artifactSchema"],
        "artifact_document.v1"
    );
    assert_eq!(
        article_artifact["metadata"]["artifactDocumentId"],
        "artifact-document:content-factory-app:artifact-article-1"
    );
    assert_eq!(
        article_artifact["metadata"]["artifactDocument"]["metadata"]["currentVersionId"],
        "artifact-document:content-factory-app:artifact-article-1:v1"
    );
    assert_eq!(
        article_artifact["metadata"]["artifactDocument"]["metadata"]["sourceRunBinding"]["taskId"],
        "task-article-1"
    );
    assert_eq!(
        article_artifact["metadata"]["articleWorkspace"]["objectKind"],
        "articleDraft"
    );
    assert!(article_artifact["content"].is_null());
    assert_eq!(article_artifact["contentStatus"], "notRequested");

    let artifact_read = core
        .read_artifacts(ArtifactReadParams {
            session_id: "sess_article_workspace".to_string(),
            turn_id: Some("turn_article_workspace".to_string()),
            artifact_ref: Some("artifact-article-1".to_string()),
            include_content: Some(true),
            cursor: None,
            limit: Some(1),
        })
        .expect("read product object artifact document");
    assert_eq!(artifact_read.artifacts.len(), 1);
    assert_eq!(
        artifact_read.artifacts[0].content_status,
        ArtifactContentStatus::Available
    );
    let content = artifact_read.artifacts[0]
        .content
        .as_deref()
        .expect("artifact document content");
    assert!(content.contains("\"schemaVersion\": \"artifact_document.v1\""));
    assert!(content
        .contains("\"artifactId\": \"artifact-document:content-factory-app:artifact-article-1\""));

    core.append_artifact_snapshot(
        "sess_article_workspace",
        Some(&turn.turn_id),
        article_workspace_selection_snapshot_payload(),
    )
    .expect("append selected product object snapshot");

    let updated_read = core
        .read_session(AgentSessionReadParams {
            session_id: "sess_article_workspace".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read updated session");
    let updated_detail = updated_read.detail.expect("updated session detail");

    assert_eq!(
        updated_detail["article_workspace"]["selectedObjectRef"]["id"],
        "image-set-1"
    );
    assert_eq!(
        updated_detail["thread_read"]["article_workspace"]["selectedObjectRef"]["kind"],
        "imageGenerationSet"
    );

    core.append_artifact_snapshot(
        "sess_article_workspace",
        Some(&turn.turn_id),
        article_workspace_edited_draft_snapshot_payload(
            "# 用户编辑稿\n\n这是 Article Editor 画布写回后的正文。",
            "2026-06-29T10:00:00.000Z",
        ),
    )
    .expect("append edited article draft snapshot");

    let edited_read = core
        .read_session(AgentSessionReadParams {
            session_id: "sess_article_workspace".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read edited session");
    let edited_detail = edited_read.detail.expect("edited session detail");
    assert_eq!(
        edited_detail["article_workspace"]["objects"][0]["source"]["documentText"],
        "# 用户编辑稿\n\n这是 Article Editor 画布写回后的正文。"
    );
    assert_eq!(
        edited_detail["article_workspace"]["objects"][0]["source"]["finalMarkdown"],
        "# 用户编辑稿\n\n这是 Article Editor 画布写回后的正文。"
    );
    assert_eq!(
        edited_detail["article_workspace"]["objects"][0]["source"]["researchRounds"][0]["title"],
        "检索行业背景"
    );
    assert_eq!(
        edited_detail["article_workspace"]["objects"][0]["source"]["edited"],
        true
    );
    assert_eq!(
        edited_detail["article_workspace"]["editedDraft"]["objectRef"]["id"],
        "article-1"
    );
    assert_eq!(
        edited_detail["thread_read"]["article_workspace"]["objects"][0]["source"]["documentText"],
        "# 用户编辑稿\n\n这是 Article Editor 画布写回后的正文。"
    );

    core.append_artifact_snapshot(
        "sess_article_workspace",
        Some(&turn.turn_id),
        article_workspace_edited_draft_snapshot_payload(
            "# 用户编辑稿\n\n![正文配图](pending-image-task://task-inline?status=running&prompt=%E9%85%8D%E5%9B%BE)\n<!-- lime:image-task-slot:article-image-slot-1 -->",
            "2026-06-29T10:01:00.000Z",
        ),
    )
    .expect("append inline image slot article snapshot");

    core.append_artifact_snapshot(
        "sess_article_workspace",
        Some(&turn.turn_id),
        article_workspace_edited_draft_snapshot_payload(
            "# 用户编辑稿\n\n正文被陈旧编辑器写回覆盖，已经没有 inline slot。",
            "2026-06-29T10:02:00.000Z",
        ),
    )
    .expect("append stale inline image downgrade snapshot");

    let inline_slot_read = core
        .read_session(AgentSessionReadParams {
            session_id: "sess_article_workspace".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read inline image slot session");
    let inline_slot_detail = inline_slot_read
        .detail
        .expect("inline image slot session detail");
    assert!(
        inline_slot_detail["article_workspace"]["objects"][0]["source"]["documentText"]
            .as_str()
            .expect("inline slot document text")
            .contains("lime:image-task-slot:article-image-slot-1")
    );
    assert_eq!(
        inline_slot_detail["article_workspace"]["editedDraft"]["updatedAt"],
        "2026-06-29T10:01:00.000Z"
    );

    core.append_artifact_snapshot(
        "sess_article_workspace",
        Some(&turn.turn_id),
        article_workspace_edited_draft_snapshot_payload(
            "# 用户编辑稿\n\n![正文配图](https://example.com/article-image.png)",
            "2026-06-29T10:03:00.000Z",
        ),
    )
    .expect("append resolved inline image article snapshot");

    let resolved_inline_slot_read = core
        .read_session(AgentSessionReadParams {
            session_id: "sess_article_workspace".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read resolved inline image slot session");
    let resolved_inline_slot_detail = resolved_inline_slot_read
        .detail
        .expect("resolved inline image slot session detail");
    let resolved_inline_slot_markdown = resolved_inline_slot_detail["article_workspace"]["objects"]
        [0]["source"]["documentText"]
        .as_str()
        .expect("resolved inline slot document text");
    assert!(resolved_inline_slot_markdown.contains("https://example.com/article-image.png"));
    assert!(!resolved_inline_slot_markdown.contains("pending-image-task://"));
    assert_eq!(
        resolved_inline_slot_detail["article_workspace"]["editedDraft"]["updatedAt"],
        "2026-06-29T10:03:00.000Z"
    );

    core.start_turn(
        AgentSessionTurnStartParams {
            session_id: "sess_article_workspace".to_string(),
            turn_id: Some("turn_article_workspace_action".to_string()),
            input: AgentInput {
                text: "请重新生成「配图组」".to_string(),
                attachments: Vec::new(),
            },
            runtime_options: Some(RuntimeOptions {
                runtime_request: Some(app_server_protocol::RuntimeRequest {
                    metadata: Some(json!({
                    "plugin": {
                        "source": "read_model_fixture",
                        "app_id": "content-factory-app",
                        "session_id": "sess_article_workspace",
                        "workspace_id": "workspace-main",
                        "article_workspace_action": {
                            "key": "regenerate",
                            "intent": "regenerate",
                            "risk": "write",
                            "task_kind": "content.image.generate",
                            "output_artifact_kind": "workspace_patch",
                            "prompt": "请重新生成「配图组」",
                            "object": {
                                "app_id": "content-factory-app",
                                "kind": "imageGenerationSet",
                                "id": "image-set-1",
                                "session_id": "sess_article_workspace",
                                "title": "配图组",
                                "status": "needs_review",
                                "artifact_ids": ["artifact-image-1"],
                                "source_turn_id": "turn_article_workspace"
                            }
                        }
                    },
                    "right_surface": {
                        "surface_kind": "articleWorkspace",
                        "source": "threadRead",
                        "action_key": "regenerate"
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
    .expect("start article workspace action turn");

    core.append_external_runtime_events(
        "sess_article_workspace",
        Some("turn_article_workspace_action"),
        vec![
            RuntimeEvent::new(
                "artifact.snapshot",
                json!({
                    "artifact": {
                        "artifactId": "artifact-image-regenerate-workspace-patch",
                        "artifactRef": "artifact-image-regenerate-workspace-patch",
                        "path": ".lime/artifacts/article-workspace/image-regenerate-workspace-patch.json",
                        "title": "配图组重新生成结果",
                        "kind": "workspace_patch",
                        "status": "ready",
                        "metadata": {
                            "workspacePatch": {
                                "schemaVersion": 1,
                                "appId": "content-factory-app",
                                "sessionId": "sess_article_workspace",
                                "selectedObjectRef": {
                                    "appId": "content-factory-app",
                                    "kind": "imageGenerationSet",
                                    "id": "image-set-1",
                                    "sessionId": "sess_article_workspace"
                                },
                                "objects": [
                                    {
                                        "ref": {
                                            "appId": "content-factory-app",
                                            "kind": "imageGenerationSet",
                                            "id": "image-set-1",
                                            "sessionId": "sess_article_workspace",
                                            "artifactIds": ["artifact-image-regenerated"],
                                            "sourceTurnId": "turn_article_workspace_action",
                                            "sourceTaskId": "task-image-regenerate-1",
                                            "version": "2"
                                        },
                                        "title": "配图组",
                                        "status": "ready",
                                        "summary": "已重新生成 2 张候选图",
                                        "previewArtifactId": "artifact-image-regenerated",
                                        "source": {
                                            "taskKind": "content.image.generate",
                                            "taskId": "task-image-regenerate-1",
                                            "turnId": "turn_article_workspace_action",
                                            "artifactIds": ["artifact-image-regenerated"],
                                            "images": [
                                                {
                                                    "id": "image-regenerated-1",
                                                    "title": "厨房台面主图",
                                                    "url": "file:///tmp/content-factory/image-regenerated-1.png",
                                                    "prompt": "厨房台面主图，明亮自然光"
                                                },
                                                {
                                                    "id": "image-regenerated-2",
                                                    "title": "产品细节图",
                                                    "url": "file:///tmp/content-factory/image-regenerated-2.png",
                                                    "prompt": "产品细节图，突出质感"
                                                }
                                            ]
                                        }
                                    }
                                ],
                                "layoutState": {
                                    "activeTabKind": "articleWorkspace",
                                    "activePaneKind": "imageGrid",
                                    "openTabKinds": ["articleWorkspace", "files"],
                                    "splitMode": "chat-right-dock"
                                },
                                "workerEvidence": [{
                                    "taskId": "task-image-regenerate-1",
                                    "taskKind": "content.image.generate",
                                    "turnId": "turn_article_workspace_action",
                                    "status": "completed",
                                    "source": "workspace_patch"
                                }]
                            }
                        }
                    }
                }),
            ),
            RuntimeEvent::new("turn.completed", json!({})),
        ],
    )
    .expect("complete article workspace action turn");

    let action_read = core
        .read_session(AgentSessionReadParams {
            session_id: "sess_article_workspace".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read action history session");
    let action_detail = action_read.detail.expect("action history detail");
    let action_history = action_detail["thread_read"]["article_workspace_actions"]
        .as_array()
        .expect("article workspace action history");

    assert_eq!(action_history.len(), 1);
    assert_eq!(action_history[0]["key"], "regenerate");
    assert_eq!(action_history[0]["status"], "completed");
    assert_eq!(action_history[0]["turnId"], "turn_article_workspace_action");
    assert_eq!(action_history[0]["objectRef"]["id"], "image-set-1");
    assert_eq!(action_history[0]["objectTitle"], "配图组");
    assert_eq!(action_history[0]["taskKind"], "content.image.generate");
    let action_result_artifacts = action_history[0]["resultArtifacts"]
        .as_array()
        .expect("article workspace action result artifacts");
    assert!(action_result_artifacts
        .iter()
        .any(
            |artifact| artifact["artifactRef"] == "artifact-image-regenerated"
                && artifact["kind"] == "artifact_document"
                && artifact["title"] == "配图组"
        ));
    assert!(action_result_artifacts
        .iter()
        .any(
            |artifact| artifact["artifactRef"] == "artifact-image-regenerate-workspace-patch"
                && artifact["kind"] == "workspace_patch"
        ));
    assert_eq!(
        action_detail["article_workspace"]["actionHistory"][0],
        action_history[0]
    );
    let image_object = action_detail["article_workspace"]["objects"]
        .as_array()
        .expect("article workspace objects")
        .iter()
        .find(|object| object["ref"]["id"] == "image-set-1")
        .expect("updated image object");
    assert_eq!(image_object["status"], "ready");
    assert_eq!(image_object["summary"], "已重新生成 2 张候选图");
    assert_eq!(
        image_object["previewArtifactId"],
        "artifact-image-regenerated"
    );
    assert_eq!(
        image_object["ref"]["sourceTurnId"],
        "turn_article_workspace_action"
    );
    assert_eq!(
        image_object["ref"]["sourceTaskId"],
        "task-image-regenerate-1"
    );
    assert_eq!(
        action_detail["article_workspace"]["selectedObjectRef"]["id"],
        "image-set-1"
    );
    assert_eq!(
        action_detail["article_workspace"]["workerEvidence"][2]["taskId"],
        "task-image-regenerate-1"
    );
    assert_eq!(
        action_detail["article_workspace"]["workerEvidence"][2]["status"],
        "completed"
    );
    let article_object = action_detail["article_workspace"]["objects"]
        .as_array()
        .expect("article workspace objects")
        .iter()
        .find(|object| object["ref"]["id"] == "article-1")
        .expect("article object");
    assert_eq!(
        article_object["source"]["researchRounds"][0]["title"],
        "检索行业背景"
    );
    assert_eq!(
        article_object["source"]["outline"][0]["title"],
        "开场：为什么要把写作变成工作流"
    );
    assert_eq!(
        article_object["source"]["imageSlots"][0]["prompt"],
        "桌面端内容工厂写作流程图，中文标签"
    );
}

#[tokio::test]
async fn read_session_marks_failed_article_draft_as_non_deliverable_when_article_worker_errors() {
    let core = RuntimeCore::default();
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_article_workspace_failed".to_string()),
        thread_id: Some("thread_article_workspace_failed".to_string()),
        app_id: "content-factory-app".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    let turn = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_article_workspace_failed".to_string(),
                turn_id: Some("turn_article_workspace_failed".to_string()),
                input: AgentInput {
                    text: "生成文章草稿".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            RuntimeHostContext::default(),
        )
        .await
        .expect("turn")
        .response
        .turn;

    core.append_external_runtime_events(
        "sess_article_workspace_failed",
        Some(&turn.turn_id),
        vec![
            RuntimeEvent::new(
                "artifact.snapshot",
                article_workspace_search_snapshot_payload(json!([
                    {
                        "id": "host-tool-evidence-search-request-1",
                        "requestId": "search-request-1",
                        "tool": "WebSearch",
                        "toolCallId": "workspace-patch-host-tool-websearch-search-request-1",
                        "source": "workspace_patch_host_tool_requests",
                        "status": "completed",
                        "query": "Lime 写文章",
                        "purpose": "验证宿主真实检索回填",
                        "summary": "session=sess_article_workspace_failed query=Lime 写文章 result=found",
                        "output": "session=sess_article_workspace_failed query=Lime 写文章 result=found",
                        "error": null,
                        "confidence": "host_verified"
                    }
                ])),
            ),
            RuntimeEvent::new(
                "artifact.snapshot",
                json!({
                    "artifact": {
                        "artifactId": "artifact-workspace-patch-failed",
                        "kind": "workspace_patch",
                        "metadata": {
                            "workspacePatch": {
                                "appId": "content-factory-app",
                                "sessionId": "sess_article_workspace_failed",
                                "objects": [{
                                    "ref": {
                                        "appId": "content-factory-app",
                                        "kind": "articleDraft",
                                        "id": "article-1",
                                        "sessionId": "sess_article_workspace",
                                        "sourceTaskId": "task-article-1"
                                    },
                                    "status": "ready",
                                    "source": {
                                        "taskId": "task-article-1",
                                        "taskKind": "content.article.generate"
                                    }
                                }],
                                "workerEvidence": [{
                                    "source": "workspace_patch",
                                    "eventType": "artifact.snapshot",
                                    "artifactKind": "workspace_patch",
                                    "errorCode": "worker_invalid_json_output",
                                    "errorMessage": "Workspace patch generation failed",
                                    "failureCategory": "invalid_output",
                                    "appId": "content-factory-app",
                                    "taskId": "task-article-1",
                                    "taskKind": "content.article.generate",
                                    "turnId": "turn_article_workspace_failed",
                                    "status": "failed"
                                }]
                            }
                        }
                    }
                }),
            ),
            RuntimeEvent::new("turn.completed", json!({})),
        ],
    )
    .expect("append artifact event");

    let read = core
        .read_session(AgentSessionReadParams {
            session_id: "sess_article_workspace_failed".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read session");
    let detail = read.detail.expect("session detail");
    let article_workspace = &detail["article_workspace"];

    assert_eq!(article_workspace["objects"][0]["status"], "failed");
    assert_eq!(
        article_workspace["objects"][0]["summary"],
        "写作失败，文章草稿未达到可交付状态"
    );
    assert_eq!(article_workspace["objects"][1]["status"], "needs_review");
    let failure = article_workspace["workerEvidence"]
        .as_array()
        .expect("worker evidence")
        .iter()
        .find(|item| item["taskKind"] == "content.article.generate" && item["status"] == "failed")
        .expect("article failure evidence");
    assert_eq!(failure["source"], "workspace_patch");
}
