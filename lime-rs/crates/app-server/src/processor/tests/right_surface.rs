//! right surface request processor tests.

use super::super::*;
use super::tests_support::*;
use app_server_protocol::{
    JsonRpcMessage, RequestId, METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CONSUME,
    METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_DISMISS, METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_LIST,
    METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST,
};
use serde_json::json;

#[tokio::test]
async fn workspace_right_surface_methods_register_and_list_pending_requests() {
    let processor = RequestProcessor::new(RuntimeCore::default());
    initialize_processor(&processor).await;

    let requested = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(20),
            METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST,
            Some(json!({
                "workspaceId": "workspace-main",
                "workspaceRoot": "/workspace/project",
                "sessionId": "sess-main",
                "surfaceKind": "browser",
                "origin": "mcp:browser",
                "reason": "Browser candidate",
                "priority": "high",
                "candidateId": "candidate-1",
                "metadata": { "source": "browser-assist" },
            })),
        ))
        .await
        .expect("right surface request response");
    assert_eq!(requested.len(), 2);

    let request_id = match &requested[0] {
        JsonRpcMessage::Response(response) => {
            assert_eq!(response.result["status"], "pending");
            assert_eq!(response.result["pending"]["surfaceKind"], "browser");
            assert_eq!(response.result["pending"]["origin"], "mcp:browser");
            assert_eq!(response.result["pending"]["priority"], "high");
            assert_eq!(
                response.result["pending"]["metadata"],
                json!({ "source": "browser-assist" })
            );
            response.result["requestId"]
                .as_str()
                .expect("request id")
                .to_string()
        }
        other => panic!("expected response, got {other:?}"),
    };
    assert_right_surface_pending_changed_notification(
        &requested[1],
        "requested",
        json!([request_id.clone()]),
    );
    match &requested[1] {
        JsonRpcMessage::Notification(notification) => {
            let params = notification.params.as_ref().expect("notification params");
            assert_eq!(params["surfaceKind"], "browser");
            assert_eq!(params["pending"][0]["requestId"], request_id);
            assert_eq!(params["pending"][0]["workspaceRoot"], "/workspace/project");
        }
        other => panic!("expected right surface notification, got {other:?}"),
    }

    let listed = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(21),
            METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_LIST,
            Some(json!({
                "workspaceId": "workspace-main",
                "surfaceKind": "browser",
                "limit": 5,
            })),
        ))
        .await
        .expect("right surface pending list response");

    match &listed[0] {
        JsonRpcMessage::Response(response) => {
            let pending = response.result["pending"]
                .as_array()
                .expect("pending array");
            assert_eq!(pending.len(), 1);
            assert_eq!(pending[0]["workspaceRoot"], "/workspace/project");
            assert_eq!(pending[0]["sessionId"], "sess-main");
            assert_eq!(pending[0]["candidateId"], "candidate-1");
        }
        other => panic!("expected response, got {other:?}"),
    }

    let consumed = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(22),
            METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CONSUME,
            Some(json!({
                "requestId": request_id.clone(),
                "requestIds": ["right-surface:missing"],
            })),
        ))
        .await
        .expect("right surface pending consume response");
    assert_eq!(consumed.len(), 2);

    match &consumed[0] {
        JsonRpcMessage::Response(response) => {
            assert_eq!(response.result["status"], "consumed");
            assert_eq!(
                response.result["consumedRequestIds"],
                json!([request_id.clone()])
            );
            assert_eq!(
                response.result["missingRequestIds"],
                json!(["right-surface:missing"])
            );
        }
        other => panic!("expected response, got {other:?}"),
    }
    assert_right_surface_pending_changed_notification(
        &consumed[1],
        "consumed",
        json!([request_id.clone()]),
    );
    match &consumed[1] {
        JsonRpcMessage::Notification(notification) => {
            let params = notification.params.as_ref().expect("notification params");
            assert_eq!(params["consumedRequestIds"], json!([request_id.clone()]));
            assert_eq!(
                params["missingRequestIds"],
                json!(["right-surface:missing"])
            );
        }
        other => panic!("expected right surface notification, got {other:?}"),
    }

    let listed_after_consume = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(23),
            METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_LIST,
            Some(json!({
                "workspaceId": "workspace-main",
                "surfaceKind": "browser",
            })),
        ))
        .await
        .expect("right surface pending list response after consume");

    match &listed_after_consume[0] {
        JsonRpcMessage::Response(response) => {
            let pending = response.result["pending"]
                .as_array()
                .expect("pending array");
            assert!(pending.is_empty());
        }
        other => panic!("expected response, got {other:?}"),
    }
}

#[tokio::test]
async fn workspace_right_surface_pending_dismiss_removes_pending_request() {
    let processor = RequestProcessor::new(RuntimeCore::default());
    initialize_processor(&processor).await;

    let requested = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(24),
            METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST,
            Some(json!({
                "workspaceId": "workspace-dismiss",
                "surfaceKind": "files",
                "origin": "skill",
                "reason": "file preview ready",
            })),
        ))
        .await
        .expect("right surface request response");
    assert_eq!(requested.len(), 2);

    let request_id = match &requested[0] {
        JsonRpcMessage::Response(response) => response.result["requestId"]
            .as_str()
            .expect("request id")
            .to_string(),
        other => panic!("expected response, got {other:?}"),
    };

    let dismissed = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(25),
            METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_DISMISS,
            Some(json!({
                "requestId": request_id.clone(),
                "requestIds": ["right-surface:missing"],
                "reason": "user_closed_surface",
            })),
        ))
        .await
        .expect("right surface pending dismiss response");
    assert_eq!(dismissed.len(), 2);

    match &dismissed[0] {
        JsonRpcMessage::Response(response) => {
            assert_eq!(response.result["status"], "dismissed");
            assert_eq!(
                response.result["dismissedRequestIds"],
                json!([request_id.clone()])
            );
            assert_eq!(
                response.result["missingRequestIds"],
                json!(["right-surface:missing"])
            );
        }
        other => panic!("expected response, got {other:?}"),
    }
    assert_right_surface_pending_changed_notification(
        &dismissed[1],
        "dismissed",
        json!([request_id.clone()]),
    );
    match &dismissed[1] {
        JsonRpcMessage::Notification(notification) => {
            let params = notification.params.as_ref().expect("notification params");
            assert_eq!(params["dismissedRequestIds"], json!([request_id.clone()]));
            assert_eq!(
                params["missingRequestIds"],
                json!(["right-surface:missing"])
            );
        }
        other => panic!("expected right surface notification, got {other:?}"),
    }

    let listed_after_dismiss = processor
        .handle_request(JsonRpcRequest::new(
            RequestId::Integer(26),
            METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_LIST,
            Some(json!({
                "workspaceId": "workspace-dismiss",
                "surfaceKind": "files",
            })),
        ))
        .await
        .expect("right surface pending list response after dismiss");

    match &listed_after_dismiss[0] {
        JsonRpcMessage::Response(response) => {
            let pending = response.result["pending"]
                .as_array()
                .expect("pending array");
            assert!(pending.is_empty());
        }
        other => panic!("expected response, got {other:?}"),
    }
}
