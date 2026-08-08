use super::*;

#[tokio::test]
async fn article_workspace_artifact_documents_cover_media_storyboard_and_checklist_objects() {
    let core = RuntimeCore::default();
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_product_documents_mvp".to_string()),
        thread_id: Some("thread_product_documents_mvp".to_string()),
        app_id: "content-factory-app".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");

    let turn = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_product_documents_mvp".to_string(),
                turn_id: Some("turn_product_documents_mvp".to_string()),
                input: AgentInput {
                    text: "生成配图、分镜和交付清单".to_string(),
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
        "sess_product_documents_mvp",
        Some(&turn.turn_id),
        vec![
            RuntimeEvent::new(
                "artifact.snapshot",
                json!({
                    "artifact": {
                        "artifactId": "artifact-workspace-patch-mvp",
                        "path": ".lime/artifacts/content-factory-workspace-patch-mvp.json",
                        "title": "内容工厂交付包",
                        "kind": "workspace_patch",
                        "status": "ready",
                        "metadata": {
                            "workspacePatch": content_factory_mvp_workspace_patch()
                        }
                    }
                }),
            ),
            RuntimeEvent::new("turn.completed", json!({})),
        ],
    )
    .expect("append workspace patch");

    let read = core
        .read_session(AgentSessionReadParams {
            session_id: "sess_product_documents_mvp".to_string(),
            history_limit: None,
            history_offset: None,
            history_before_message_id: None,
        })
        .expect("read session");
    let detail = read.detail.expect("session detail");
    let artifacts = detail["thread_read"]["artifacts"]
        .as_array()
        .expect("thread read artifacts");

    assert_product_artifact(
        artifacts,
        "artifact-image-set",
        "配图组",
        "brief",
        "imageGrid",
        "image",
        "imageGenerationSet",
    );
    assert_product_artifact(
        artifacts,
        "artifact-video-storyboard",
        "视频分镜",
        "brief",
        "storyboard",
        "rich_text",
        "videoStoryboard",
    );
    assert_product_artifact(
        artifacts,
        "artifact-delivery-checklist",
        "交付检查清单",
        "plan",
        "checklist",
        "checklist",
        "deliveryChecklist",
    );

    let image_content = read_artifact_document_content(&core, "artifact-image-set");
    assert_eq!(image_content["kind"], "brief");
    assert_eq!(
        image_content["metadata"]["articleWorkspace"]["surfaceKind"],
        "imageGrid"
    );
    assert_eq!(image_content["blocks"][0]["type"], "image");
    assert_eq!(
        image_content["blocks"][0]["url"],
        "file:///tmp/content-factory/image-1.png"
    );

    let storyboard_content = read_artifact_document_content(&core, "artifact-video-storyboard");
    assert_eq!(storyboard_content["kind"], "brief");
    assert_eq!(
        storyboard_content["metadata"]["articleWorkspace"]["surfaceKind"],
        "storyboard"
    );
    assert_eq!(storyboard_content["blocks"][0]["type"], "rich_text");
    assert!(storyboard_content["blocks"][0]["markdown"]
        .as_str()
        .expect("storyboard markdown")
        .contains("厨房开场"));

    let checklist_content = read_artifact_document_content(&core, "artifact-delivery-checklist");
    assert_eq!(checklist_content["kind"], "plan");
    assert_eq!(
        checklist_content["metadata"]["articleWorkspace"]["surfaceKind"],
        "checklist"
    );
    assert_eq!(checklist_content["blocks"][0]["type"], "checklist");
    assert_eq!(checklist_content["blocks"][0]["items"][0]["state"], "done");
    assert_eq!(checklist_content["blocks"][0]["items"][1]["state"], "todo");
}

fn assert_product_artifact(
    artifacts: &[serde_json::Value],
    artifact_ref: &str,
    title: &str,
    artifact_kind: &str,
    surface_kind: &str,
    first_block_type: &str,
    object_kind: &str,
) {
    let artifact = artifacts
        .iter()
        .find(|artifact| artifact["artifactRef"] == artifact_ref)
        .expect("product artifact document");
    assert_eq!(artifact["kind"], "artifact_document");
    assert_eq!(artifact["title"], title);
    assert_eq!(artifact["metadata"]["artifactKind"], artifact_kind);
    assert_eq!(artifact["metadata"]["surfaceKind"], surface_kind);
    assert_eq!(artifact["metadata"]["layout"], surface_kind);
    assert_eq!(
        artifact["metadata"]["artifactDocument"]["blocks"][0]["type"],
        first_block_type
    );
    assert_eq!(
        artifact["metadata"]["articleWorkspace"]["objectKind"],
        object_kind
    );
    assert_eq!(
        artifact["metadata"]["articleWorkspace"]["surfaceKind"],
        surface_kind
    );
}

fn read_artifact_document_content(core: &RuntimeCore, artifact_ref: &str) -> serde_json::Value {
    let artifact_read = core
        .read_artifacts(ArtifactReadParams {
            session_id: "sess_product_documents_mvp".to_string(),
            turn_id: Some("turn_product_documents_mvp".to_string()),
            artifact_ref: Some(artifact_ref.to_string()),
            include_content: Some(true),
            cursor: None,
            limit: Some(1),
        })
        .expect("read artifact document");
    assert_eq!(artifact_read.artifacts.len(), 1);
    assert_eq!(
        artifact_read.artifacts[0].content_status,
        ArtifactContentStatus::Available
    );
    serde_json::from_str(
        artifact_read.artifacts[0]
            .content
            .as_deref()
            .expect("artifact document content"),
    )
    .expect("artifact document json")
}

fn content_factory_mvp_workspace_patch() -> serde_json::Value {
    json!({
        "schemaVersion": 1,
        "appId": "content-factory-app",
        "sessionId": "sess_product_documents_mvp",
        "workspaceId": "workspace-main",
        "objects": [
            {
                "ref": {
                    "appId": "content-factory-app",
                    "kind": "imageGenerationSet",
                    "id": "image-set-1",
                    "sessionId": "sess_product_documents_mvp",
                    "artifactIds": ["artifact-image-set"],
                    "sourceTurnId": "turn_product_documents_mvp",
                    "sourceTaskId": "task-image-1"
                },
                "title": "配图组",
                "status": "ready",
                "summary": "已生成 2 张候选图",
                "previewArtifactId": "artifact-image-set",
                "source": {
                    "taskKind": "content.image.generate",
                    "taskId": "task-image-1",
                    "turnId": "turn_product_documents_mvp",
                    "artifactIds": ["artifact-image-set"],
                    "images": [
                        {
                            "id": "image-1",
                            "title": "厨房主图",
                            "url": "file:///tmp/content-factory/image-1.png",
                            "prompt": "自然光厨房主图"
                        },
                        {
                            "id": "image-2",
                            "title": "细节图",
                            "url": "file:///tmp/content-factory/image-2.png",
                            "prompt": "产品细节"
                        }
                    ]
                }
            },
            {
                "ref": {
                    "appId": "content-factory-app",
                    "kind": "videoStoryboard",
                    "id": "storyboard-1",
                    "sessionId": "sess_product_documents_mvp",
                    "artifactIds": ["artifact-video-storyboard"],
                    "sourceTurnId": "turn_product_documents_mvp",
                    "sourceTaskId": "task-storyboard-1"
                },
                "title": "视频分镜",
                "status": "ready",
                "summary": "3 镜头短视频分镜",
                "previewArtifactId": "artifact-video-storyboard",
                "source": {
                    "taskKind": "content.video.storyboard.generate",
                    "taskId": "task-storyboard-1",
                    "turnId": "turn_product_documents_mvp",
                    "artifactIds": ["artifact-video-storyboard"],
                    "scenes": [
                        {
                            "title": "厨房开场",
                            "description": "镜头推近产品",
                            "visualPrompt": "明亮厨房，自然光",
                            "duration": "3s"
                        }
                    ]
                }
            },
            {
                "ref": {
                    "appId": "content-factory-app",
                    "kind": "deliveryChecklist",
                    "id": "delivery-checklist-1",
                    "sessionId": "sess_product_documents_mvp",
                    "artifactIds": ["artifact-delivery-checklist"],
                    "sourceTurnId": "turn_product_documents_mvp",
                    "sourceTaskId": "task-checklist-1"
                },
                "title": "交付检查清单",
                "status": "ready",
                "summary": "发布前检查项",
                "previewArtifactId": "artifact-delivery-checklist",
                "source": {
                    "taskKind": "content.delivery.review",
                    "taskId": "task-checklist-1",
                    "turnId": "turn_product_documents_mvp",
                    "artifactIds": ["artifact-delivery-checklist"],
                    "items": [
                        {
                            "id": "article",
                            "title": "文章已生成",
                            "status": "done"
                        },
                        {
                            "id": "image-license",
                            "title": "确认图片授权",
                            "notes": "发布前需复核",
                            "status": "todo"
                        }
                    ]
                }
            }
        ],
        "layoutState": {
            "activeTabKind": "articleWorkspace",
            "activePaneKind": "imageGrid",
            "openTabKinds": ["articleWorkspace", "files"]
        }
    })
}
