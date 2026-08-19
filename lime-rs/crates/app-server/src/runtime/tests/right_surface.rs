use super::support::*;
use super::*;
use std::sync::Arc;

#[tokio::test]
async fn workspace_right_surface_request_registers_pending_intent() {
    let core = RuntimeCore::default();

    let response = core
        .request_workspace_right_surface(WorkspaceRightSurfaceRequestParams {
            workspace_id: Some(" workspace-main ".to_string()),
            workspace_root: Some(" /workspace/project ".to_string()),
            session_id: Some(" sess-main ".to_string()),
            surface_kind: " browser ".to_string(),
            origin: " agent ".to_string(),
            reason: Some(" Browser candidate ".to_string()),
            priority: None,
            candidate_id: Some(" candidate-1 ".to_string()),
            ttl_ms: Some(60_000),
            metadata: Some(json!({ "source": "browser" })),
        })
        .await
        .expect("right surface request");

    assert_eq!(response.status, "pending");
    assert_eq!(response.pending.request_id, response.request_id);
    assert_eq!(
        response.pending.workspace_id.as_deref(),
        Some("workspace-main")
    );
    assert_eq!(
        response.pending.workspace_root.as_deref(),
        Some("/workspace/project")
    );
    assert_eq!(response.pending.session_id.as_deref(), Some("sess-main"));
    assert_eq!(response.pending.surface_kind, "browser");
    assert_eq!(response.pending.origin, "agent");
    assert_eq!(response.pending.priority, "normal");
    assert_eq!(
        response.pending.candidate_id.as_deref(),
        Some("candidate-1")
    );
    assert!(response.pending.expires_at.is_some());

    let pending = core
        .list_workspace_right_surface_pending(WorkspaceRightSurfacePendingListParams {
            workspace_id: Some("workspace-main".to_string()),
            workspace_root: Some("/workspace/project".to_string()),
            session_id: Some("sess-main".to_string()),
            surface_kind: Some("browser".to_string()),
            limit: None,
        })
        .await
        .expect("pending list");

    assert_eq!(pending.pending, vec![response.pending]);
}

#[tokio::test]
async fn browser_identity_is_created_by_app_server_and_reused_per_thread() {
    let core = RuntimeCore::default();
    core.start_session(AgentSessionStartParams {
        session_id: Some("session-browser".to_string()),
        thread_id: Some("thread-browser".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-browser".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("start Browser owner session");
    let open = |browser: serde_json::Value| WorkspaceRightSurfaceRequestParams {
        session_id: Some("session-browser".to_string()),
        surface_kind: "browser".to_string(),
        origin: "renderer".to_string(),
        metadata: Some(json!({ "browser": browser })),
        ..WorkspaceRightSurfaceRequestParams::default()
    };

    let first = core
        .request_workspace_right_surface(open(json!({
            "action": "open",
            "threadId": "thread-browser"
        })))
        .await
        .expect("open Browser identity");
    let second = core
        .request_workspace_right_surface(open(json!({
            "action": "open",
            "threadId": "thread-browser"
        })))
        .await
        .expect("reuse Browser identity");
    let first_browser = first
        .pending
        .metadata
        .as_ref()
        .and_then(|value| value.get("browser"))
        .expect("first Browser identity");
    let second_browser = second
        .pending
        .metadata
        .as_ref()
        .and_then(|value| value.get("browser"))
        .expect("second Browser identity");
    assert_eq!(
        first_browser.get("browserSessionId"),
        second_browser.get("browserSessionId")
    );
    assert_eq!(first_browser.get("tabId"), second_browser.get("tabId"));

    let created = core
        .request_workspace_right_surface(open(json!({
            "action": "createTab",
            "threadId": "thread-browser",
            "browserSessionId": first_browser["browserSessionId"]
        })))
        .await
        .expect("create Browser tab identity");
    let created_browser = created
        .pending
        .metadata
        .as_ref()
        .and_then(|value| value.get("browser"))
        .expect("created Browser identity");
    assert_eq!(
        created_browser.get("browserSessionId"),
        first_browser.get("browserSessionId")
    );
    assert_ne!(created_browser.get("tabId"), first_browser.get("tabId"));
}

#[tokio::test]
async fn browser_identity_rejects_a_thread_outside_the_runtime_session() {
    let core = RuntimeCore::default();
    core.start_session(AgentSessionStartParams {
        session_id: Some("session-browser-owner".to_string()),
        thread_id: Some("thread-browser-owner".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: Some("workspace-browser".to_string()),
        business_object_ref: None,
        locale: None,
    })
    .expect("start Browser owner session");

    let error = core
        .request_workspace_right_surface(WorkspaceRightSurfaceRequestParams {
            session_id: Some("session-browser-owner".to_string()),
            surface_kind: "browser".to_string(),
            origin: "renderer".to_string(),
            metadata: Some(json!({
                "browser": {
                    "action": "open",
                    "threadId": "thread-forged"
                }
            })),
            ..WorkspaceRightSurfaceRequestParams::default()
        })
        .await
        .expect_err("forged Browser thread must fail closed");

    assert!(error
        .to_string()
        .contains("does not belong to the runtime session"));
}

#[tokio::test]
async fn workspace_right_surface_pending_list_filters_and_limits_requests() {
    let core = RuntimeCore::default();
    for (workspace_id, surface_kind, priority) in [
        ("workspace-a", "files", "normal"),
        ("workspace-a", "browser", "high"),
        ("workspace-b", "browser", "normal"),
    ] {
        core.request_workspace_right_surface(WorkspaceRightSurfaceRequestParams {
            workspace_id: Some(workspace_id.to_string()),
            workspace_root: Some(format!("/repo/{workspace_id}")),
            session_id: Some("sess-filter".to_string()),
            surface_kind: surface_kind.to_string(),
            origin: "agent".to_string(),
            reason: None,
            priority: Some(priority.to_string()),
            candidate_id: None,
            ttl_ms: None,
            metadata: None,
        })
        .await
        .expect("right surface request");
    }

    let pending = core
        .list_workspace_right_surface_pending(WorkspaceRightSurfacePendingListParams {
            workspace_id: Some("workspace-a".to_string()),
            workspace_root: None,
            session_id: Some("sess-filter".to_string()),
            surface_kind: Some("browser".to_string()),
            limit: Some(1),
        })
        .await
        .expect("filtered pending");

    assert_eq!(pending.pending.len(), 1);
    assert_eq!(
        pending.pending[0].workspace_id.as_deref(),
        Some("workspace-a")
    );
    assert_eq!(pending.pending[0].surface_kind, "browser");
    assert_eq!(pending.pending[0].priority, "high");
}

#[tokio::test]
async fn workspace_right_surface_pending_list_prunes_expired_requests() {
    let core = RuntimeCore::default();
    core.request_workspace_right_surface(WorkspaceRightSurfaceRequestParams {
        workspace_id: Some("workspace-expired".to_string()),
        workspace_root: None,
        session_id: None,
        surface_kind: "files".to_string(),
        origin: "agent".to_string(),
        reason: None,
        priority: None,
        candidate_id: None,
        ttl_ms: Some(0),
        metadata: None,
    })
    .await
    .expect("expired right surface request");

    let pending = core
        .list_workspace_right_surface_pending(WorkspaceRightSurfacePendingListParams {
            workspace_id: Some("workspace-expired".to_string()),
            workspace_root: None,
            session_id: None,
            surface_kind: None,
            limit: None,
        })
        .await
        .expect("pending list");

    assert!(pending.pending.is_empty());
}

#[tokio::test]
async fn workspace_right_surface_pending_consume_removes_registered_request() {
    let core = RuntimeCore::default();
    let response = core
        .request_workspace_right_surface(WorkspaceRightSurfaceRequestParams {
            workspace_id: Some("workspace-consume".to_string()),
            workspace_root: None,
            session_id: None,
            surface_kind: "files".to_string(),
            origin: "agent".to_string(),
            reason: None,
            priority: None,
            candidate_id: None,
            ttl_ms: None,
            metadata: None,
        })
        .await
        .expect("right surface request");

    let consumed = core
        .consume_workspace_right_surface_pending(WorkspaceRightSurfacePendingConsumeParams {
            request_id: Some(format!(" {} ", response.request_id)),
            request_ids: vec![response.request_id.clone()],
        })
        .await
        .expect("consume pending request");
    assert_eq!(consumed.status, "consumed");
    assert_eq!(consumed.consumed_request_ids, vec![response.request_id]);
    assert!(consumed.missing_request_ids.is_empty());

    let pending = core
        .list_workspace_right_surface_pending(WorkspaceRightSurfacePendingListParams {
            workspace_id: Some("workspace-consume".to_string()),
            ..WorkspaceRightSurfacePendingListParams::default()
        })
        .await
        .expect("pending list");
    assert!(pending.pending.is_empty());
}

#[tokio::test]
async fn workspace_right_surface_pending_consume_reports_missing_ids() {
    let core = RuntimeCore::default();

    let consumed = core
        .consume_workspace_right_surface_pending(WorkspaceRightSurfacePendingConsumeParams {
            request_id: Some(" right-surface:missing ".to_string()),
            request_ids: vec!["right-surface:missing".to_string()],
        })
        .await
        .expect("consume missing pending request");

    assert_eq!(consumed.status, "consumed");
    assert!(consumed.consumed_request_ids.is_empty());
    assert_eq!(
        consumed.missing_request_ids,
        vec!["right-surface:missing".to_string()]
    );
}

#[tokio::test]
async fn workspace_right_surface_pending_consume_requires_request_id() {
    let core = RuntimeCore::default();

    let error = core
        .consume_workspace_right_surface_pending(WorkspaceRightSurfacePendingConsumeParams {
            request_id: Some(" ".to_string()),
            request_ids: vec![" ".to_string()],
        })
        .await
        .expect_err("missing request id");

    assert!(matches!(error, RuntimeCoreError::Backend(message) if message.contains("requestId")));
}

#[tokio::test]
async fn workspace_right_surface_pending_dismiss_removes_registered_request() {
    let core = RuntimeCore::default();
    let response = core
        .request_workspace_right_surface(WorkspaceRightSurfaceRequestParams {
            workspace_id: Some("workspace-dismiss".to_string()),
            workspace_root: None,
            session_id: None,
            surface_kind: "browser".to_string(),
            origin: "mcpTool".to_string(),
            reason: None,
            priority: None,
            candidate_id: None,
            ttl_ms: None,
            metadata: None,
        })
        .await
        .expect("right surface request");

    let dismissed = core
        .dismiss_workspace_right_surface_pending(WorkspaceRightSurfacePendingDismissParams {
            request_id: Some(format!(" {} ", response.request_id)),
            request_ids: vec![response.request_id.clone()],
            reason: Some("user_closed_surface".to_string()),
        })
        .await
        .expect("dismiss pending request");
    assert_eq!(dismissed.status, "dismissed");
    assert_eq!(dismissed.dismissed_request_ids, vec![response.request_id]);
    assert!(dismissed.missing_request_ids.is_empty());

    let pending = core
        .list_workspace_right_surface_pending(WorkspaceRightSurfacePendingListParams {
            workspace_id: Some("workspace-dismiss".to_string()),
            ..WorkspaceRightSurfacePendingListParams::default()
        })
        .await
        .expect("pending list");
    assert!(pending.pending.is_empty());
}

#[tokio::test]
async fn workspace_right_surface_pending_recovers_from_app_data_source() {
    let app_data_source = Arc::new(TestSessionDataSource::new());
    let core1 = RuntimeCore::default().with_app_data_source(app_data_source.clone());
    let response = core1
        .request_workspace_right_surface(WorkspaceRightSurfaceRequestParams {
            workspace_id: Some("workspace-recovery".to_string()),
            workspace_root: Some("/workspace/recovery".to_string()),
            session_id: Some("sess-recovery".to_string()),
            surface_kind: "browser".to_string(),
            origin: "mcpTool".to_string(),
            reason: Some("recovered from app data source".to_string()),
            priority: None,
            candidate_id: Some("candidate-recovery".to_string()),
            ttl_ms: None,
            metadata: Some(json!({ "source": "right-surface-test" })),
        })
        .await
        .expect("right surface request");

    let core2 = RuntimeCore::default().with_app_data_source(app_data_source.clone());
    let recovered = core2
        .list_workspace_right_surface_pending(WorkspaceRightSurfacePendingListParams {
            workspace_id: Some("workspace-recovery".to_string()),
            workspace_root: Some("/workspace/recovery".to_string()),
            session_id: Some("sess-recovery".to_string()),
            surface_kind: Some("browser".to_string()),
            limit: None,
        })
        .await
        .expect("recovered pending list");
    assert_eq!(recovered.pending, vec![response.pending.clone()]);

    let consumed = core2
        .consume_workspace_right_surface_pending(WorkspaceRightSurfacePendingConsumeParams {
            request_id: Some(response.request_id.clone()),
            request_ids: Vec::new(),
        })
        .await
        .expect("consume recovered pending request");
    assert_eq!(
        consumed.consumed_request_ids,
        vec![response.request_id.clone()]
    );
    assert!(consumed.missing_request_ids.is_empty());

    for core in [&core1, &core2] {
        let pending = core
            .list_workspace_right_surface_pending(WorkspaceRightSurfacePendingListParams {
                workspace_id: Some("workspace-recovery".to_string()),
                workspace_root: Some("/workspace/recovery".to_string()),
                session_id: Some("sess-recovery".to_string()),
                surface_kind: Some("browser".to_string()),
                limit: None,
            })
            .await
            .expect("pending list after consume");
        assert!(pending.pending.is_empty());
    }
}

#[tokio::test]
async fn workspace_right_surface_request_requires_surface_kind_and_origin() {
    let core = RuntimeCore::default();

    let missing_surface = core
        .request_workspace_right_surface(WorkspaceRightSurfaceRequestParams {
            surface_kind: " ".to_string(),
            origin: "agent".to_string(),
            ..WorkspaceRightSurfaceRequestParams::default()
        })
        .await
        .expect_err("missing surface kind");
    assert!(
        matches!(missing_surface, RuntimeCoreError::Backend(message) if message.contains("surfaceKind"))
    );

    let missing_origin = core
        .request_workspace_right_surface(WorkspaceRightSurfaceRequestParams {
            surface_kind: "files".to_string(),
            origin: " ".to_string(),
            ..WorkspaceRightSurfaceRequestParams::default()
        })
        .await
        .expect_err("missing origin");
    assert!(
        matches!(missing_origin, RuntimeCoreError::Backend(message) if message.contains("origin"))
    );
}
