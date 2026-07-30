use crate::protocol::AgentEvent;
use agent_runtime::action_required::ActionRequiredState;
use agent_runtime::request_permissions::{
    request_permissions_and_wait_with_notification, RequestPermissionsRequest,
    RequestPermissionsResponse,
};
use agent_runtime::session_loop::RuntimeSessionInputHandle;
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedSender;

pub(crate) async fn request_permissions(
    state: Arc<ActionRequiredState>,
    response_handle: RuntimeSessionInputHandle,
    request: RequestPermissionsRequest,
    event_sender: UnboundedSender<AgentEvent>,
) -> Result<RequestPermissionsResponse, agent_runtime::request_permissions::RequestPermissionsError>
{
    request_permissions_and_wait_with_notification(
        state.as_ref(),
        response_handle,
        request,
        move |action| {
            let _ = event_sender.send(AgentEvent::ActionRequired {
                request_id: action.request_id.clone(),
                action_type: action.action_type.clone(),
                data: serde_json::json!({
                    "actionType": action.action_type,
                    "toolCallId": action.identity.item_id,
                    "environmentId": action.environment_id,
                    "cwd": action.cwd,
                    "reason": action.reason,
                    "permissions": action.permissions,
                    "createdAtMs": action.created_at_ms,
                    "deadlineAtMs": action.deadline_at_ms,
                }),
                scope: Some(action.scope.clone()),
            });
        },
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_runtime::request_permissions::{
        AdditionalNetworkPermissions, GrantedPermissionProfile, PermissionGrantScope,
        RequestPermissionProfile, RequestPermissionsIdentity,
    };
    use agent_runtime::session_loop::{RuntimeSessionClosureTask, RuntimeSessionRegistry};
    use std::sync::Mutex as StdMutex;
    use std::time::Duration;

    #[tokio::test]
    async fn bridge_emits_typed_permission_action_and_resumes_exact_waiter() {
        let state = Arc::new(ActionRequiredState::default());
        let registry = RuntimeSessionRegistry::default();
        let session = registry.get_or_create("session-1").await;
        let (event_sender, mut event_receiver) = tokio::sync::mpsc::unbounded_channel();
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let result_tx = Arc::new(StdMutex::new(Some(result_tx)));
        let task_state = Arc::clone(&state);
        let task = RuntimeSessionClosureTask::new(
            "turn-1",
            Vec::new(),
            move |context, _input, _cancel| {
                let result_tx = Arc::clone(&result_tx);
                let request = RequestPermissionsRequest::new(
                    RequestPermissionsIdentity::new("session-1", "thread-1", "turn-1", "call-1"),
                    Some("local".to_string()),
                    "/tmp/workspace",
                    Some("Allow network".to_string()),
                    RequestPermissionProfile {
                        network: Some(AdditionalNetworkPermissions {
                            enabled: Some(true),
                        }),
                        file_system: None,
                    },
                )
                .with_timeout(Duration::from_secs(1));
                let state = Arc::clone(&task_state);
                let event_sender = event_sender.clone();
                Box::pin(async move {
                    let result =
                        request_permissions(state, context.input_handle(), request, event_sender)
                            .await;
                    if let Some(sender) = result_tx.lock().expect("result sender").take() {
                        let _ = sender.send(result);
                    }
                    Ok(())
                })
            },
        );
        let submission = session.submit(Arc::new(task), false).await.expect("task");
        let event = event_receiver.recv().await.expect("permission action");
        let AgentEvent::ActionRequired {
            request_id,
            action_type,
            data,
            scope,
        } = event
        else {
            panic!("expected permission action");
        };
        assert_eq!(action_type, "request_permissions");
        assert_eq!(data["toolCallId"], "call-1");
        assert_eq!(data["environmentId"], "local");
        assert_eq!(data["cwd"], "/tmp/workspace");
        let scope = scope.expect("scope");
        state
            .resolve_action(&request_id, Some(&scope))
            .await
            .expect("resolve");
        session
            .respond_permission(
                Some("turn-1"),
                request_id,
                serde_json::to_value(RequestPermissionsResponse {
                    permissions: GrantedPermissionProfile {
                        network: Some(AdditionalNetworkPermissions {
                            enabled: Some(true),
                        }),
                        file_system: None,
                    },
                    scope: PermissionGrantScope::Turn,
                    strict_auto_review: None,
                })
                .expect("response"),
            )
            .await
            .expect("respond");
        assert!(result_rx.await.expect("result sender").is_ok());
        submission
            .completion
            .await
            .expect("completion")
            .expect("task");
        registry.shutdown("session-1").await.expect("shutdown");
    }
}
