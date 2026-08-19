use super::*;
use crate::runtime::artifact_content::DEFAULT_ARTIFACT_CONTENT_MAX_BYTES;
use app_server_protocol::ArtifactContentStatus;
use app_server_protocol::ArtifactReadParams;
use std::fs;

#[tokio::test]
async fn read_artifacts_indexes_latest_artifact_events_for_session() {
    let sidecar_root = tempfile::tempdir().expect("sidecar root");
    let core = RuntimeCore::default().with_sidecar_store(Arc::new(
        SidecarStore::new(sidecar_root.path()).expect("sidecar store"),
    ));
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_artifacts".to_string()),
        thread_id: Some("thread_artifacts".to_string()),
        app_id: "content-studio".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");
    let turn = core
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_artifacts".to_string(),
                turn_id: Some("turn_artifacts".to_string()),
                input: AgentInput {
                    text: "生成产物".to_string(),
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
        "sess_artifacts",
        Some(&turn.turn_id),
        vec![
            RuntimeEvent::new(
                "artifact.snapshot",
                json!({
                    "artifactId": "artifact-report",
                    "filePath": ".lime/artifacts/report-v1.md",
                    "title": "Report",
                    "kind": "markdown_report",
                    "status": "ready",
                    "metadata": {
                        "version": 1
                    }
                }),
            ),
            RuntimeEvent::new(
                "artifact.snapshot",
                json!({
                    "artifactId": "artifact-report",
                    "filePath": ".lime/artifacts/report-v2.md",
                    "title": "Report",
                    "kind": "markdown_report",
                    "status": "ready",
                    "metadata": {
                        "version": 2
                    }
                }),
            ),
            RuntimeEvent::new(
                "artifact.snapshot",
                json!({
                    "artifact": {
                        "id": "artifact-outline",
                        "path": ".lime/artifacts/outline.md",
                        "content": "# Outline"
                    }
                }),
            ),
        ],
    )
    .expect("append artifact events");

    let response = core
        .read_artifacts(ArtifactReadParams {
            session_id: "sess_artifacts".to_string(),
            turn_id: Some("turn_artifacts".to_string()),
            artifact_ref: None,
            include_content: None,
            cursor: None,
            limit: Some(1),
        })
        .expect("read artifacts");

    assert_eq!(response.artifacts.len(), 1);
    assert_eq!(response.next_cursor.as_deref(), Some("1"));
    assert_eq!(response.artifacts[0].artifact_ref, "artifact-outline");
    assert_eq!(
        response.artifacts[0].path.as_deref(),
        Some(".lime/artifacts/outline.md")
    );
    assert_eq!(response.artifacts[0].content, None);
    assert_eq!(
        response.artifacts[0].content_status,
        ArtifactContentStatus::NotRequested
    );
    assert!(response.artifacts[0]
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("content"))
        .is_none());
    assert!(response.artifacts[0]
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("sidecarRef"))
        .and_then(|sidecar_ref| sidecar_ref.get("sha256"))
        .and_then(serde_json::Value::as_str)
        .is_some_and(|value| value.starts_with("sha256:")));

    let filtered = core
        .read_artifacts(ArtifactReadParams {
            session_id: "sess_artifacts".to_string(),
            turn_id: None,
            artifact_ref: Some("artifact-report".to_string()),
            include_content: Some(true),
            cursor: None,
            limit: None,
        })
        .expect("filtered artifacts");
    assert_eq!(filtered.artifacts.len(), 1);
    assert_eq!(
        filtered.artifacts[0].path.as_deref(),
        Some(".lime/artifacts/report-v2.md")
    );
    assert_eq!(
        filtered.artifacts[0]
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("version")),
        Some(&json!(2))
    );
    assert_eq!(
        filtered.artifacts[0].content_status,
        ArtifactContentStatus::Unavailable
    );

    let outline = core
        .read_artifacts(ArtifactReadParams {
            session_id: "sess_artifacts".to_string(),
            turn_id: None,
            artifact_ref: Some("artifact-outline".to_string()),
            include_content: Some(true),
            cursor: None,
            limit: None,
        })
        .expect("outline artifact");
    assert_eq!(outline.artifacts.len(), 1);
    assert_eq!(outline.artifacts[0].content.as_deref(), Some("# Outline"));
    assert_eq!(
        outline.artifacts[0].content_status,
        ArtifactContentStatus::Available
    );
}

#[test]
fn read_artifacts_uses_injected_content_provider_for_current_page() {
    #[derive(Debug)]
    struct TestArtifactContentProvider;

    impl ArtifactContentProvider for TestArtifactContentProvider {
        fn read_content(&self, request: &ArtifactContentRequest) -> Option<String> {
            Some(format!(
                "{}:{}",
                request.session.app_id, request.artifact.artifact_ref
            ))
        }
    }

    let sidecar_root = tempfile::tempdir().expect("sidecar root");
    let core = RuntimeCore::with_backend_capability_source_and_artifact_content_provider(
        Arc::new(MockBackend),
        Arc::new(CapabilityInventorySource::default()),
        Arc::new(TestArtifactContentProvider),
    )
    .with_sidecar_store(Arc::new(
        SidecarStore::new(sidecar_root.path()).expect("sidecar store"),
    ));
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_content".to_string()),
        thread_id: None,
        app_id: "content-studio".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");
    core.append_external_runtime_events(
        "sess_content",
        None,
        vec![RuntimeEvent::new(
            "artifact.snapshot",
            json!({
                "artifactId": "artifact-provider",
                "path": ".app-server/artifacts/provider.md",
                "content": "inline content"
            }),
        )],
    )
    .expect("append artifact event");

    let without_content = core
        .read_artifacts(ArtifactReadParams {
            session_id: "sess_content".to_string(),
            turn_id: None,
            artifact_ref: Some("artifact-provider".to_string()),
            include_content: None,
            cursor: None,
            limit: None,
        })
        .expect("read summary");
    assert_eq!(without_content.artifacts[0].content, None);
    assert_eq!(
        without_content.artifacts[0].content_status,
        ArtifactContentStatus::NotRequested
    );

    let with_content = core
        .read_artifacts(ArtifactReadParams {
            session_id: "sess_content".to_string(),
            turn_id: None,
            artifact_ref: Some("artifact-provider".to_string()),
            include_content: Some(true),
            cursor: None,
            limit: None,
        })
        .expect("read content");
    assert_eq!(
        with_content.artifacts[0].content.as_deref(),
        Some("inline content")
    );
    assert_eq!(
        with_content.artifacts[0].content_status,
        ArtifactContentStatus::Available
    );
    assert!(with_content.artifacts[0]
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("content"))
        .is_none());
    assert!(with_content.artifacts[0]
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("sidecarRef"))
        .and_then(|sidecar_ref| sidecar_ref.get("sha256"))
        .and_then(serde_json::Value::as_str)
        .is_some_and(|value| value.starts_with("sha256:")));
}

#[test]
fn filesystem_artifact_content_provider_reads_allowed_relative_path() {
    let temp = tempfile::tempdir().expect("temp dir");
    let artifact_dir = temp.path().join(".app-server").join("artifacts");
    fs::create_dir_all(&artifact_dir).expect("artifact dir");
    fs::write(artifact_dir.join("provider.md"), "# Provider").expect("artifact file");

    let core = RuntimeCore::with_backend_capability_source_and_artifact_content_provider(
        Arc::new(MockBackend),
        Arc::new(CapabilityInventorySource::default()),
        Arc::new(FilesystemArtifactContentProvider::new(temp.path())),
    );
    core.start_session(AgentSessionStartParams {
        session_id: Some("sess_file_content".to_string()),
        thread_id: None,
        app_id: "content-studio".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("session");
    core.append_external_runtime_events(
        "sess_file_content",
        None,
        vec![RuntimeEvent::new(
            "artifact.snapshot",
            json!({
                "artifactId": "artifact-file",
                "path": ".app-server/artifacts/provider.md"
            }),
        )],
    )
    .expect("append artifact event");

    let response = core
        .read_artifacts(ArtifactReadParams {
            session_id: "sess_file_content".to_string(),
            turn_id: None,
            artifact_ref: Some("artifact-file".to_string()),
            include_content: Some(true),
            cursor: None,
            limit: None,
        })
        .expect("read file content");

    assert_eq!(response.artifacts.len(), 1);
    assert_eq!(response.artifacts[0].content.as_deref(), Some("# Provider"));
    assert_eq!(
        response.artifacts[0].content_status,
        ArtifactContentStatus::Available
    );
}

#[test]
fn filesystem_artifact_content_provider_rejects_escape_and_oversized_files() {
    let temp = tempfile::tempdir().expect("temp dir");
    let artifact_dir = temp.path().join("artifacts");
    fs::create_dir_all(&artifact_dir).expect("artifact dir");
    fs::write(artifact_dir.join("small.md"), "ok").expect("small file");
    fs::write(artifact_dir.join("large.md"), "too-large").expect("large file");
    let outside = tempfile::tempdir().expect("outside dir");
    fs::write(outside.path().join("outside.md"), "outside").expect("outside file");

    let provider = FilesystemArtifactContentProvider::new(temp.path()).with_max_bytes(2);
    let session = AgentSession {
        session_id: "sess_fs".to_string(),
        thread_id: "thread_fs".to_string(),
        app_id: "content-studio".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: None,
        status: AgentSessionStatus::Idle,
        created_at: timestamp(),
        updated_at: timestamp(),
    };

    let small = provider.read_content(&ArtifactContentRequest {
        session: session.clone(),
        artifact: ArtifactSummary {
            artifact_ref: "small".to_string(),
            event_id: "evt-small".to_string(),
            sequence: 1,
            turn_id: None,
            artifact_id: Some("small".to_string()),
            path: Some("artifacts/small.md".to_string()),
            title: None,
            kind: None,
            status: None,
            content: None,
            content_status: ArtifactContentStatus::NotRequested,
            metadata: None,
        },
    });
    assert_eq!(small.as_deref(), Some("ok"));

    let oversized = provider.read_content(&ArtifactContentRequest {
        session: session.clone(),
        artifact: ArtifactSummary {
            artifact_ref: "large".to_string(),
            event_id: "evt-large".to_string(),
            sequence: 2,
            turn_id: None,
            artifact_id: Some("large".to_string()),
            path: Some("artifacts/large.md".to_string()),
            title: None,
            kind: None,
            status: None,
            content: Some("inline fallback".to_string()),
            content_status: ArtifactContentStatus::NotRequested,
            metadata: None,
        },
    });
    assert_eq!(oversized.as_deref(), Some("inline fallback"));

    let escaped = provider.read_content(&ArtifactContentRequest {
        session,
        artifact: ArtifactSummary {
            artifact_ref: "escape".to_string(),
            event_id: "evt-escape".to_string(),
            sequence: 3,
            turn_id: None,
            artifact_id: Some("escape".to_string()),
            path: Some(format!(
                "../{}/outside.md",
                outside
                    .path()
                    .file_name()
                    .expect("outside file name")
                    .to_string_lossy()
            )),
            title: None,
            kind: None,
            status: None,
            content: Some("inline fallback".to_string()),
            content_status: ArtifactContentStatus::NotRequested,
            metadata: None,
        },
    });
    assert_eq!(escaped.as_deref(), Some("inline fallback"));
}

#[test]
fn workspace_artifact_content_provider_reads_declared_workspace_files() {
    let workspace = tempfile::tempdir().expect("workspace");
    let source_dir = workspace.path().join("src");
    fs::create_dir_all(&source_dir).expect("source dir");
    let source_path = source_dir.join("lib.rs");
    fs::write(&source_path, "pub fn answer() -> u32 { 42 }").expect("source file");

    let session = AgentSession {
        session_id: "sess_workspace".to_string(),
        thread_id: "thread_workspace".to_string(),
        app_id: "coding".to_string(),
        workspace_id: Some("workspace-main".to_string()),
        business_object_ref: None,
        status: AgentSessionStatus::Idle,
        created_at: timestamp(),
        updated_at: timestamp(),
    };
    let provider = WorkspaceArtifactContentProvider;
    let metadata = json!({
        "environments": [{ "cwd": workspace.path().to_string_lossy() }]
    });

    let relative = provider.read_content(&ArtifactContentRequest {
        session: session.clone(),
        artifact: ArtifactSummary {
            artifact_ref: "relative".to_string(),
            event_id: "evt-relative".to_string(),
            sequence: 1,
            turn_id: None,
            artifact_id: Some("relative".to_string()),
            path: Some("src/lib.rs".to_string()),
            title: None,
            kind: None,
            status: None,
            content: None,
            content_status: ArtifactContentStatus::NotRequested,
            metadata: Some(metadata.clone()),
        },
    });
    assert_eq!(relative.as_deref(), Some("pub fn answer() -> u32 { 42 }"));

    let absolute = provider.read_content(&ArtifactContentRequest {
        session: session.clone(),
        artifact: ArtifactSummary {
            artifact_ref: "absolute".to_string(),
            event_id: "evt-absolute".to_string(),
            sequence: 2,
            turn_id: None,
            artifact_id: Some("absolute".to_string()),
            path: Some(source_path.to_string_lossy().into_owned()),
            title: None,
            kind: None,
            status: None,
            content: None,
            content_status: ArtifactContentStatus::NotRequested,
            metadata: Some(metadata.clone()),
        },
    });
    assert_eq!(absolute.as_deref(), Some("pub fn answer() -> u32 { 42 }"));

    let outside = tempfile::NamedTempFile::new_in(workspace.path().parent().expect("temp parent"))
        .expect("outside file");
    fs::write(outside.path(), "outside").expect("outside content");
    let escaped = provider.read_content(&ArtifactContentRequest {
        session: session.clone(),
        artifact: ArtifactSummary {
            artifact_ref: "escaped".to_string(),
            event_id: "evt-escaped".to_string(),
            sequence: 3,
            turn_id: None,
            artifact_id: Some("escaped".to_string()),
            path: Some(format!(
                "../{}",
                outside
                    .path()
                    .file_name()
                    .expect("outside file name")
                    .to_string_lossy()
            )),
            title: None,
            kind: None,
            status: None,
            content: None,
            content_status: ArtifactContentStatus::NotRequested,
            metadata: Some(metadata.clone()),
        },
    });
    assert_eq!(escaped, None);

    let large_path = workspace.path().join("large.bin");
    fs::write(
        &large_path,
        vec![b'x'; (DEFAULT_ARTIFACT_CONTENT_MAX_BYTES + 1) as usize],
    )
    .expect("large file");
    let oversized = provider.read_content(&ArtifactContentRequest {
        session: session.clone(),
        artifact: ArtifactSummary {
            artifact_ref: "oversized".to_string(),
            event_id: "evt-oversized".to_string(),
            sequence: 4,
            turn_id: None,
            artifact_id: Some("oversized".to_string()),
            path: Some("large.bin".to_string()),
            title: None,
            kind: None,
            status: None,
            content: None,
            content_status: ArtifactContentStatus::NotRequested,
            metadata: Some(metadata),
        },
    });
    assert_eq!(oversized, None);

    let inline = provider.read_content(&ArtifactContentRequest {
        session: session.clone(),
        artifact: ArtifactSummary {
            artifact_ref: "inline".to_string(),
            event_id: "evt-inline".to_string(),
            sequence: 5,
            turn_id: None,
            artifact_id: Some("inline".to_string()),
            path: None,
            title: None,
            kind: None,
            status: None,
            content: Some("inline content".to_string()),
            content_status: ArtifactContentStatus::NotRequested,
            metadata: None,
        },
    });
    assert_eq!(inline.as_deref(), Some("inline content"));

    let empty_inline = provider.read_content(&ArtifactContentRequest {
        session,
        artifact: ArtifactSummary {
            artifact_ref: "empty-inline".to_string(),
            event_id: "evt-empty-inline".to_string(),
            sequence: 6,
            turn_id: None,
            artifact_id: Some("empty-inline".to_string()),
            path: Some("missing.txt".to_string()),
            title: None,
            kind: None,
            status: None,
            content: Some(String::new()),
            content_status: ArtifactContentStatus::NotRequested,
            metadata: Some(json!({ "cwd": workspace.path().to_string_lossy() })),
        },
    });
    assert_eq!(empty_inline.as_deref(), Some(""));
}
