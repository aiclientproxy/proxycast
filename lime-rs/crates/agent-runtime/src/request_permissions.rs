use crate::action_required::{ActionRequiredRequest, ActionRequiredState};
use crate::session_loop::{RuntimeSessionInputHandle, RuntimeSessionResponseKind};
use agent_protocol::action_required::ActionRequiredScope;
use serde_json::Value;
use std::fmt;
use std::time::Duration;

pub use app_server_protocol::protocol::v2::{
    AdditionalFileSystemPermissions, AdditionalNetworkPermissions, FileSystemAccessMode,
    FileSystemPath, FileSystemSandboxEntry, FileSystemSpecialPath, GrantedPermissionProfile,
    PermissionGrantScope, PermissionsRequestApprovalResponse, RequestPermissionProfile,
};

pub const REQUEST_PERMISSIONS_ACTION_TYPE: &str = "request_permissions";
pub const DEFAULT_REQUEST_PERMISSIONS_TIMEOUT: Duration = Duration::from_secs(300);

pub type RequestPermissionsResponse = PermissionsRequestApprovalResponse;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestPermissionsIdentity {
    pub session_id: String,
    pub thread_id: String,
    pub turn_id: String,
    pub item_id: String,
}

impl RequestPermissionsIdentity {
    pub fn new(
        session_id: impl Into<String>,
        thread_id: impl Into<String>,
        turn_id: impl Into<String>,
        item_id: impl Into<String>,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            thread_id: thread_id.into(),
            turn_id: turn_id.into(),
            item_id: item_id.into(),
        }
    }

    fn validate(&self) -> Result<(), RequestPermissionsError> {
        for (name, value) in [
            ("session_id", self.session_id.as_str()),
            ("thread_id", self.thread_id.as_str()),
            ("turn_id", self.turn_id.as_str()),
            ("item_id", self.item_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(RequestPermissionsError::invalid_request(format!(
                    "request_permissions requires {name}"
                )));
            }
        }
        Ok(())
    }

    fn action_scope(&self) -> ActionRequiredScope {
        ActionRequiredScope {
            session_id: Some(self.session_id.clone()),
            thread_id: Some(self.thread_id.clone()),
            turn_id: Some(self.turn_id.clone()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestPermissionsRequest {
    pub identity: RequestPermissionsIdentity,
    pub environment_id: Option<String>,
    pub cwd: String,
    pub reason: Option<String>,
    pub permissions: RequestPermissionProfile,
    pub timeout: Duration,
}

impl RequestPermissionsRequest {
    pub fn new(
        identity: RequestPermissionsIdentity,
        environment_id: Option<String>,
        cwd: impl Into<String>,
        reason: Option<String>,
        permissions: RequestPermissionProfile,
    ) -> Self {
        Self {
            identity,
            environment_id,
            cwd: cwd.into(),
            reason,
            permissions,
            timeout: DEFAULT_REQUEST_PERMISSIONS_TIMEOUT,
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    fn validate(&self) -> Result<(), RequestPermissionsError> {
        self.identity.validate()?;
        if self.cwd.trim().is_empty() || !std::path::Path::new(&self.cwd).is_absolute() {
            return Err(RequestPermissionsError::invalid_request(
                "request_permissions requires an absolute cwd",
            ));
        }
        if permission_profile_is_empty(&self.permissions) {
            return Err(RequestPermissionsError::invalid_request(
                "request_permissions requires at least one permission",
            ));
        }
        if self.timeout.is_zero() {
            return Err(RequestPermissionsError::invalid_request(
                "request_permissions timeout must be greater than zero",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RequestPermissionsAction {
    pub request_id: String,
    pub action_type: String,
    pub identity: RequestPermissionsIdentity,
    pub environment_id: Option<String>,
    pub cwd: String,
    pub reason: Option<String>,
    pub permissions: RequestPermissionProfile,
    pub requested_schema: Value,
    pub available_decisions: Vec<String>,
    pub scope: ActionRequiredScope,
    pub created_at_ms: Option<u64>,
    pub deadline_at_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestPermissionsError {
    code: &'static str,
    message: String,
}

impl RequestPermissionsError {
    fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            code: "permission_request_invalid",
            message: message.into(),
        }
    }

    fn invalid_response(message: impl Into<String>) -> Self {
        Self {
            code: "permission_response_invalid",
            message: message.into(),
        }
    }

    fn wait_failed(message: impl Into<String>) -> Self {
        Self {
            code: "permission_response_wait_failed",
            message: message.into(),
        }
    }

    pub fn code(&self) -> &'static str {
        self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for RequestPermissionsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for RequestPermissionsError {}

pub async fn request_permissions_and_wait_with_notification<F>(
    state: &ActionRequiredState,
    response_handle: RuntimeSessionInputHandle,
    request: RequestPermissionsRequest,
    notify: F,
) -> Result<RequestPermissionsResponse, RequestPermissionsError>
where
    F: FnOnce(&RequestPermissionsAction),
{
    request.validate()?;
    let RequestPermissionsRequest {
        identity,
        environment_id,
        cwd,
        reason,
        permissions,
        timeout,
    } = request;
    let scope = identity.action_scope();
    let requested_schema = response_schema();
    let message = reason
        .clone()
        .unwrap_or_else(|| "Request additional permissions".to_string());
    let requested_permissions = permissions.clone();
    let response = state
        .request_action_and_wait_with_notification(
            response_handle,
            RuntimeSessionResponseKind::Permission,
            REQUEST_PERMISSIONS_ACTION_TYPE,
            Some(identity.item_id.clone()),
            vec![
                "allow_once".to_string(),
                "allow_for_session".to_string(),
                "decline".to_string(),
            ],
            Some(scope.clone()),
            message,
            requested_schema.clone(),
            timeout,
            move |queued| {
                notify(&materialize_action(
                    queued,
                    identity,
                    environment_id,
                    cwd,
                    reason,
                    permissions,
                    requested_schema,
                    scope,
                ));
            },
        )
        .await
        .map_err(|error| RequestPermissionsError::wait_failed(error.to_string()))?;
    let response: RequestPermissionsResponse =
        serde_json::from_value(response).map_err(|error| {
            RequestPermissionsError::invalid_response(format!(
                "request_permissions response does not match the canonical shape: {error}"
            ))
        })?;
    normalize_response(requested_permissions, response)
}

fn materialize_action(
    queued: &ActionRequiredRequest,
    identity: RequestPermissionsIdentity,
    environment_id: Option<String>,
    cwd: String,
    reason: Option<String>,
    permissions: RequestPermissionProfile,
    requested_schema: Value,
    scope: ActionRequiredScope,
) -> RequestPermissionsAction {
    RequestPermissionsAction {
        request_id: queued.id.clone(),
        action_type: queued.action_type.clone(),
        identity,
        environment_id,
        cwd,
        reason,
        permissions,
        requested_schema,
        available_decisions: queued.available_decisions.clone(),
        scope,
        created_at_ms: queued.created_at_ms,
        deadline_at_ms: queued.deadline_at_ms,
    }
}

fn normalize_response(
    requested: RequestPermissionProfile,
    response: RequestPermissionsResponse,
) -> Result<RequestPermissionsResponse, RequestPermissionsError> {
    if response.strict_auto_review == Some(true) && response.scope == PermissionGrantScope::Session
    {
        return Err(RequestPermissionsError::invalid_response(
            "strict_auto_review cannot be granted for a session",
        ));
    }
    if !permission_profile_is_subset(&response.permissions, &requested) {
        return Err(RequestPermissionsError::invalid_response(
            "granted permissions must be a subset of the request",
        ));
    }
    Ok(response)
}

fn permission_profile_is_subset(
    granted: &GrantedPermissionProfile,
    requested: &RequestPermissionProfile,
) -> bool {
    granted.network.as_ref().is_none_or(|granted| {
        requested
            .network
            .as_ref()
            .is_some_and(|requested| granted == requested)
    }) && granted.file_system.as_ref().is_none_or(|granted| {
        requested
            .file_system
            .as_ref()
            .is_some_and(|requested| file_system_permissions_are_subset(granted, requested))
    })
}

fn permission_profile_is_empty(profile: &RequestPermissionProfile) -> bool {
    profile.network.is_none() && profile.file_system.is_none()
}

fn file_system_permissions_are_subset(
    granted: &AdditionalFileSystemPermissions,
    requested: &AdditionalFileSystemPermissions,
) -> bool {
    optional_values_are_subset(granted.read.as_deref(), requested.read.as_deref())
        && optional_values_are_subset(granted.write.as_deref(), requested.write.as_deref())
        && granted.glob_scan_max_depth.is_none_or(|depth| {
            requested
                .glob_scan_max_depth
                .is_some_and(|requested_depth| depth <= requested_depth)
        })
        && optional_values_are_subset(granted.entries.as_deref(), requested.entries.as_deref())
}

fn optional_values_are_subset<T: PartialEq>(
    granted: Option<&[T]>,
    requested: Option<&[T]>,
) -> bool {
    granted.is_none_or(|granted| {
        requested.is_some_and(|requested| granted.iter().all(|value| requested.contains(value)))
    })
}

fn response_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "permissions": { "type": "object" },
            "scope": { "type": "string", "enum": ["turn", "session"] },
            "strictAutoReview": { "type": "boolean" }
        },
        "required": ["permissions"]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session_loop::{
        RuntimeSessionClosureTask, RuntimeSessionRegistry, RuntimeSessionTaskOutcome,
    };
    use std::sync::{Arc, Mutex as StdMutex};
    use tokio::sync::oneshot;

    fn identity() -> RequestPermissionsIdentity {
        RequestPermissionsIdentity::new(
            "session-permissions",
            "thread-permissions",
            "turn-permissions",
            "item-permissions",
        )
    }

    fn requested_permissions() -> RequestPermissionProfile {
        RequestPermissionProfile {
            network: Some(AdditionalNetworkPermissions {
                enabled: Some(true),
            }),
            file_system: None,
        }
    }

    fn granted_permissions() -> GrantedPermissionProfile {
        GrantedPermissionProfile {
            network: requested_permissions().network,
            file_system: None,
        }
    }

    #[tokio::test]
    async fn permission_action_preserves_identity_and_resolves_exactly_once() {
        let state = Arc::new(ActionRequiredState::default());
        let registry = RuntimeSessionRegistry::default();
        let session = registry
            .get_or_create("session-permissions", "thread-permissions")
            .await
            .expect("bind permission session actor");
        let (action_tx, action_rx) = oneshot::channel();
        let action_tx = Arc::new(StdMutex::new(Some(action_tx)));
        let (result_tx, result_rx) = oneshot::channel();
        let result_tx = Arc::new(StdMutex::new(Some(result_tx)));
        let task_state = Arc::clone(&state);
        let task = RuntimeSessionClosureTask::new(
            "turn-permissions",
            Vec::new(),
            move |context, _input, _cancel| {
                let state = Arc::clone(&task_state);
                let action_tx = Arc::clone(&action_tx);
                let result_tx = Arc::clone(&result_tx);
                Box::pin(async move {
                    let result = request_permissions_and_wait_with_notification(
                        &state,
                        context.input_handle(),
                        RequestPermissionsRequest::new(
                            identity(),
                            Some("local".to_string()),
                            "/tmp/workspace",
                            Some("Allow network access?".to_string()),
                            requested_permissions(),
                        )
                        .with_timeout(Duration::from_secs(1)),
                        move |action| {
                            if let Some(sender) = action_tx.lock().expect("action sender").take() {
                                let _ = sender.send(action.clone());
                            }
                        },
                    )
                    .await;
                    if let Some(sender) = result_tx.lock().expect("result sender").take() {
                        let _ = sender.send(result);
                    }
                    Ok(())
                })
            },
        );
        let submission = session
            .submit(Arc::new(task), false)
            .await
            .expect("permission task");
        let action = action_rx.await.expect("permission action");

        assert_eq!(action.action_type, REQUEST_PERMISSIONS_ACTION_TYPE);
        assert_eq!(action.identity, identity());
        assert_eq!(action.identity.thread_id, "thread-permissions");
        assert_eq!(action.identity.turn_id, "turn-permissions");
        assert_eq!(action.identity.item_id, "item-permissions");
        assert_eq!(action.permissions, requested_permissions());
        assert_eq!(action.scope, identity().action_scope());

        state
            .resolve_action(&action.request_id, Some(&action.scope))
            .await
            .expect("resolve permission action");
        let response = RequestPermissionsResponse {
            permissions: granted_permissions(),
            scope: PermissionGrantScope::Turn,
            strict_auto_review: None,
        };
        session
            .respond_permission(
                Some("turn-permissions"),
                &action.request_id,
                serde_json::to_value(&response).expect("serialize permission response"),
            )
            .await
            .expect("permission response");
        assert!(session
            .respond_permission(
                Some("turn-permissions"),
                &action.request_id,
                serde_json::to_value(&response).expect("serialize duplicate response"),
            )
            .await
            .is_err());
        assert!(state
            .resolve_action(&action.request_id, Some(&action.scope))
            .await
            .is_err());

        assert_eq!(
            result_rx
                .await
                .expect("permission result sender")
                .expect("permission result"),
            response
        );
        assert_eq!(
            submission.completion.await.expect("task completion"),
            Ok(RuntimeSessionTaskOutcome::Completed)
        );
        registry
            .shutdown("session-permissions")
            .await
            .expect("shutdown");
    }

    #[tokio::test]
    async fn permission_response_fails_closed_for_unsafe_session_review() {
        let state = Arc::new(ActionRequiredState::default());
        let registry = RuntimeSessionRegistry::default();
        let session = registry
            .get_or_create("session-permissions", "thread-permissions")
            .await
            .expect("bind permission session actor");
        let (action_tx, action_rx) = oneshot::channel();
        let action_tx = Arc::new(StdMutex::new(Some(action_tx)));
        let (result_tx, result_rx) = oneshot::channel();
        let result_tx = Arc::new(StdMutex::new(Some(result_tx)));
        let task_state = Arc::clone(&state);
        let task = RuntimeSessionClosureTask::new(
            "turn-permissions",
            Vec::new(),
            move |context, _input, _cancel| {
                let state = Arc::clone(&task_state);
                let action_tx = Arc::clone(&action_tx);
                let result_tx = Arc::clone(&result_tx);
                Box::pin(async move {
                    let result = request_permissions_and_wait_with_notification(
                        &state,
                        context.input_handle(),
                        RequestPermissionsRequest::new(
                            identity(),
                            None,
                            "/tmp/workspace",
                            None,
                            requested_permissions(),
                        )
                        .with_timeout(Duration::from_secs(1)),
                        move |action| {
                            if let Some(sender) = action_tx.lock().expect("action sender").take() {
                                let _ = sender.send(action.clone());
                            }
                        },
                    )
                    .await;
                    if let Some(sender) = result_tx.lock().expect("result sender").take() {
                        let _ = sender.send(result);
                    }
                    Ok(())
                })
            },
        );
        let submission = session
            .submit(Arc::new(task), false)
            .await
            .expect("permission task");
        let action = action_rx.await.expect("permission action");
        state
            .resolve_action(&action.request_id, Some(&action.scope))
            .await
            .expect("resolve permission action");
        session
            .respond_permission(
                Some("turn-permissions"),
                &action.request_id,
                serde_json::json!({
                    "permissions": { "network": { "enabled": true } },
                    "scope": "session",
                    "strictAutoReview": true
                }),
            )
            .await
            .expect("deliver invalid permission response");

        let error = result_rx
            .await
            .expect("permission result sender")
            .expect_err("unsafe response must fail closed");
        assert_eq!(error.code(), "permission_response_invalid");
        assert_eq!(
            error.message(),
            "strict_auto_review cannot be granted for a session"
        );
        assert_eq!(
            submission.completion.await.expect("task completion"),
            Ok(RuntimeSessionTaskOutcome::Completed)
        );
        registry
            .shutdown("session-permissions")
            .await
            .expect("shutdown");
    }

    #[test]
    fn permission_request_rejects_missing_identity_and_permission_escalation() {
        let request = RequestPermissionsRequest::new(
            RequestPermissionsIdentity::new("session-1", "thread-1", "turn-1", " "),
            None,
            "/tmp/workspace",
            None,
            requested_permissions(),
        );
        assert_eq!(
            request.validate().expect_err("missing item id").code(),
            "permission_request_invalid"
        );

        let error = normalize_response(
            requested_permissions(),
            RequestPermissionsResponse {
                permissions: GrantedPermissionProfile {
                    network: None,
                    file_system: Some(AdditionalFileSystemPermissions {
                        read: None,
                        write: Some(vec!["/tmp".to_string()]),
                        glob_scan_max_depth: None,
                        entries: None,
                    }),
                },
                scope: PermissionGrantScope::Turn,
                strict_auto_review: None,
            },
        )
        .expect_err("unrequested permission must fail closed");
        assert_eq!(error.code(), "permission_response_invalid");
    }
}
