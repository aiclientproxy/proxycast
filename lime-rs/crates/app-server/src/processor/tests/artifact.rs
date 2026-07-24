//! artifact request processor tests.

use super::super::*;
use crate::{ProjectionStore, SidecarStore};
use app_server_protocol::{
    protocol::v2::METHOD_ARTIFACT_WRITE, AgentInput, AgentSessionStartParams,
    AgentSessionTurnStartParams, ClientCapabilities, JsonRpcMessage, RequestId,
    METHOD_ARTIFACT_READ, METHOD_INITIALIZE, METHOD_INITIALIZED,
};
use serde_json::json;
use std::sync::Arc;

#[tokio::test]
async fn artifact_read_requires_initialized_and_returns_artifact_summaries() {
    let sidecar_root = tempfile::tempdir().expect("sidecar root");
    let runtime = RuntimeCore::default()
        .with_sidecar_store(Arc::new(
            SidecarStore::new(sidecar_root.path()).expect("sidecar store"),
        ))
        .with_projection_store(Arc::new(
            ProjectionStore::initialize(sidecar_root.path().join("projection.sqlite"))
                .expect("projection store"),
        ));
    runtime
        .start_session(AgentSessionStartParams {
            session_id: Some("sess_artifact".to_string()),
            thread_id: Some("thread_artifact".to_string()),
            app_id: "content-studio".to_string(),
            workspace_id: Some("workspace-main".to_string()),
            business_object_ref: None,
            locale: None,
        })
        .expect("session");
    runtime
        .append_external_runtime_events(
            "sess_artifact",
            None,
            vec![crate::RuntimeEvent::new(
                "artifact.snapshot",
                json!({
                    "artifactId": "artifact-report",
                    "filePath": ".app-server/artifacts/report.md",
                    "title": "Report",
                    "kind": "markdown",
                    "status": "ready",
                    "content": "# Report",
                }),
            )],
        )
        .expect("artifact event");

    let processor = RequestProcessor::new(runtime);
    let blocked = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(1),
            METHOD_ARTIFACT_READ,
            Some(json!({ "sessionId": "sess_artifact" })),
        ))
        .await
        .expect("blocked response");
    assert!(matches!(
        &blocked[0],
        JsonRpcMessage::Error(error) if error.error.code == error_codes::NOT_INITIALIZED
    ));

    processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(2),
            METHOD_INITIALIZE,
            Some(
                serde_json::to_value(InitializeParams {
                    client_info: ClientInfo {
                        name: "test-client".to_string(),
                        title: None,
                        version: None,
                    },
                    capabilities: ClientCapabilities::default(),
                })
                .expect("initialize params"),
            ),
        ))
        .await
        .expect("initialize");
    processor.handle_notification(JsonRpcNotification::new(
        METHOD_INITIALIZED,
        Some(json!({})),
    ));

    let messages = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(3),
            METHOD_ARTIFACT_READ,
            Some(json!({
                "sessionId": "sess_artifact",
                "artifactRef": "artifact-report",
            })),
        ))
        .await
        .expect("artifact read response");

    match &messages[0] {
        JsonRpcMessage::Response(response) => {
            assert_eq!(
                response.result["artifacts"][0]["artifactRef"],
                "artifact-report"
            );
            assert_eq!(
                response.result["artifacts"][0]["path"],
                ".app-server/artifacts/report.md"
            );
            assert_eq!(response.result["artifacts"][0]["title"], "Report");
            assert_eq!(
                response.result["artifacts"][0]["contentStatus"],
                "notRequested"
            );
            assert!(response.result["artifacts"][0].get("content").is_none());
        }
        other => panic!("expected response, got {other:?}"),
    }
}

#[tokio::test]
async fn artifact_write_persists_typed_snapshot_after_turn_completion() {
    let sidecar_root = tempfile::tempdir().expect("sidecar root");
    let runtime = RuntimeCore::default()
        .with_sidecar_store(Arc::new(
            SidecarStore::new(sidecar_root.path()).expect("sidecar store"),
        ))
        .with_projection_store(Arc::new(
            ProjectionStore::initialize(sidecar_root.path().join("projection.sqlite"))
                .expect("projection store"),
        ));
    runtime
        .start_session(AgentSessionStartParams {
            session_id: Some("sess_artifact_write".to_string()),
            thread_id: Some("thread_artifact_write".to_string()),
            app_id: "content-studio".to_string(),
            workspace_id: Some("workspace-main".to_string()),
            business_object_ref: None,
            locale: None,
        })
        .expect("session");
    let turn = runtime
        .start_turn(
            AgentSessionTurnStartParams {
                session_id: "sess_artifact_write".to_string(),
                turn_id: Some("turn_artifact_write".to_string()),
                input: AgentInput {
                    text: "draft article".to_string(),
                    attachments: Vec::new(),
                },
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: false,
            },
            crate::RuntimeHostContext::default(),
        )
        .await
        .expect("turn")
        .response
        .turn;
    runtime
        .append_external_runtime_events(
            "sess_artifact_write",
            Some(&turn.turn_id),
            vec![crate::RuntimeEvent::new("turn.completed", json!({}))],
        )
        .expect("complete turn");

    let processor = RequestProcessor::new(runtime);
    processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(1),
            METHOD_INITIALIZE,
            Some(
                serde_json::to_value(InitializeParams {
                    client_info: ClientInfo {
                        name: "test-client".to_string(),
                        title: None,
                        version: None,
                    },
                    capabilities: ClientCapabilities::default(),
                })
                .expect("initialize params"),
            ),
        ))
        .await
        .expect("initialize");
    processor.handle_notification(JsonRpcNotification::new(
        METHOD_INITIALIZED,
        Some(json!({})),
    ));

    let messages = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(2),
            METHOD_ARTIFACT_WRITE,
            Some(json!({
                "threadId": "thread_artifact_write",
                "turnId": "turn_artifact_write",
                "artifact": {
                    "artifactRef": "artifact-document-1",
                    "artifactDocumentId": "document-1",
                    "path": "drafts/article.json",
                    "title": "Draft",
                    "kind": "artifact_document",
                    "status": "ready",
                    "content": "{\"schemaVersion\":\"artifact-document/v1\"}",
                    "metadata": {"versionNo": 1}
                }
            })),
        ))
        .await
        .expect("artifact write response");

    assert_eq!(
        messages.len(),
        1,
        "typed write must not emit legacy wrappers"
    );
    match &messages[0] {
        JsonRpcMessage::Response(response) => {
            assert_eq!(response.result["threadId"], "thread_artifact_write");
            assert_eq!(response.result["turnId"], "turn_artifact_write");
            assert_eq!(response.result["artifactRef"], "artifact-document-1");
            assert_eq!(response.result["artifactDocumentId"], "document-1");
            assert!(response.result["eventId"]
                .as_str()
                .is_some_and(|value| value.starts_with("evt_")));
            assert!(response.result["sidecar"]["relativePath"]
                .as_str()
                .is_some_and(|value| value.contains("runtime-artifacts")));
            assert!(
                response.result["sidecar"]["bytes"]
                    .as_u64()
                    .unwrap_or_default()
                    > 0
            );
            assert_eq!(response.result["sidecar"]["contentStatus"], "available");
        }
        other => panic!("expected response, got {other:?}"),
    }

    let read = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(3),
            METHOD_ARTIFACT_READ,
            Some(json!({
                "sessionId": "sess_artifact_write",
                "artifactRef": "artifact-document-1",
                "includeContent": true
            })),
        ))
        .await
        .expect("artifact read response");
    match &read[0] {
        JsonRpcMessage::Response(response) => {
            assert_eq!(
                response.result["artifacts"][0]["artifactRef"],
                "artifact-document-1"
            );
            assert_eq!(
                response.result["artifacts"][0]["content"],
                "{\"schemaVersion\":\"artifact-document/v1\"}"
            );
            assert_eq!(
                response.result["artifacts"][0]["contentStatus"],
                "available"
            );
        }
        other => panic!("expected response, got {other:?}"),
    }
}
