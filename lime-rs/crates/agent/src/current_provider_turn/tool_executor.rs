//! 当前 provider sampling step 的工具执行适配器。

use super::{is_web_tool, mcp_step_snapshot};
use crate::guardian_review;
use crate::protocol::{AgentEvent, AgentToolProgressPayload};
use crate::request_tool_policy::RequestToolPolicy;
use crate::runtime_state::{AgentRuntimeState, EffectivePermissionGrant};
use agent_protocol::ThreadId;
use agent_runtime::session_loop::RuntimeSessionInputHandle;
use futures::StreamExt;
use rmcp::model::{CallToolResult, ErrorData, ServerNotification};
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;
use tokio::sync::mpsc::UnboundedSender;
use tokio_util::sync::CancellationToken;
use tool_runtime::tool_executor::{
    RuntimeToolExecutionError, RuntimeToolExecutionFuture, RuntimeToolExecutionRequest,
    RuntimeToolExecutionResult, RuntimeToolExecutor, RuntimeToolPolicyErrorKind,
};

const GRANTED_PERMISSIONS_METADATA_KEY: &str = "grantedPermissions";
const STRICT_AUTO_REVIEW_METADATA_KEY: &str = "strictAutoReview";

pub(super) mod orchestration;

#[derive(Clone)]
pub(super) struct CurrentTurnToolExecutor {
    pub(super) state: AgentRuntimeState,
    pub(super) policy: RequestToolPolicy,
    pub(super) event_sender: UnboundedSender<AgentEvent>,
    pub(super) thread_id: ThreadId,
    pub(super) mcp_snapshot: tool_runtime::mcp_connection::McpStepSnapshot,
    pub(super) deferred_tools: mcp_step_snapshot::DeferredToolSelections,
    pub(super) agent_control_gateway:
        Option<tool_runtime::agent_control::AgentControlGatewayHandle>,
    pub(super) pending_input: Option<RuntimeSessionInputHandle>,
    pub(super) dynamic_tool_routes: mcp_step_snapshot::DynamicToolRoutes,
}

impl RuntimeToolExecutor for CurrentTurnToolExecutor {
    fn execute<'a>(
        &'a self,
        request: RuntimeToolExecutionRequest<'a>,
    ) -> RuntimeToolExecutionFuture<'a> {
        Box::pin(async move {
            if self.policy.matches_any_disallowed_tool(request.tool_name) {
                return Err(RuntimeToolExecutionError::new(
                    format!("当前请求策略禁止工具调用: {}", request.tool_name),
                    Some(RuntimeToolPolicyErrorKind::PermissionDenied(
                        "request_tool_policy".to_string(),
                    )),
                )
                .before_handler());
            }
            if !self.policy.allows_web_search() && is_web_tool(request.tool_name) {
                return Err(RuntimeToolExecutionError::new(
                    format!("当前请求未启用联网工具: {}", request.tool_name),
                    Some(RuntimeToolPolicyErrorKind::PermissionDenied(
                        "web_search_disabled".to_string(),
                    )),
                )
                .before_handler());
            }

            if tool_runtime::request_permissions::is_request_permissions_tool(request.tool_name) {
                let args = tool_runtime::request_permissions::parse_request_permissions_args(
                    request.params,
                    request.context.working_directory(),
                )
                .map_err(|error| {
                    RuntimeToolExecutionError::new(
                        error,
                        Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
                            "request_permissions_invalid".to_string(),
                        )),
                    )
                    .before_handler()
                })?;
                if args
                    .environment_id
                    .as_deref()
                    .is_some_and(|environment_id| environment_id != "local")
                {
                    return Err(RuntimeToolExecutionError::new(
                        "request_permissions references an unknown environment_id",
                        Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
                            "request_permissions_environment_unknown".to_string(),
                        )),
                    )
                    .before_handler());
                }
                let (scope, request_call_id) = action_scope(request, &self.thread_id)
                    .map_err(RuntimeToolExecutionError::before_handler)?;
                let scope = scope.ok_or_else(|| {
                    RuntimeToolExecutionError::new(
                        "request_permissions requires canonical action scope",
                        Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
                            "request_permissions_scope_missing".to_string(),
                        )),
                    )
                    .before_handler()
                })?;
                let session_id = scope.session_id.clone().ok_or_else(|| {
                    RuntimeToolExecutionError::new(
                        "request_permissions requires canonical session_id",
                        Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
                            "request_permissions_scope_missing".to_string(),
                        )),
                    )
                    .before_handler()
                })?;
                let turn_id = scope.turn_id.clone().ok_or_else(|| {
                    RuntimeToolExecutionError::new(
                        "request_permissions requires canonical turn_id",
                        Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
                            "request_permissions_scope_missing".to_string(),
                        )),
                    )
                    .before_handler()
                })?;
                let response_handle = self.pending_input.clone().ok_or_else(|| {
                    RuntimeToolExecutionError::new(
                        "request_permissions requires the active session response owner",
                        Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
                            "session_response_owner_missing".to_string(),
                        )),
                    )
                    .before_handler()
                })?;
                let permission_request =
                    agent_runtime::request_permissions::RequestPermissionsRequest::new(
                        agent_runtime::request_permissions::RequestPermissionsIdentity::new(
                            session_id.clone(),
                            self.thread_id.as_str(),
                            turn_id.clone(),
                            request_call_id,
                        ),
                        args.environment_id,
                        request.context.working_directory().to_string_lossy(),
                        args.reason,
                        args.permissions,
                    );
                let response = crate::request_permissions_bridge::request_permissions(
                    self.state.action_required_state(),
                    response_handle,
                    permission_request,
                    self.event_sender.clone(),
                )
                .await
                .map_err(|error| {
                    RuntimeToolExecutionError::new(
                        error.to_string(),
                        Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
                            error.code().to_string(),
                        )),
                    )
                })?;
                self.state
                    .record_permission_grant(&session_id, &turn_id, &response)
                    .await;
                let output = serde_json::to_string(&response).map_err(|error| {
                    RuntimeToolExecutionError::new(
                        format!("failed to serialize request_permissions response: {error}"),
                        Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
                            "request_permissions_response".to_string(),
                        )),
                    )
                })?;
                return Ok(RuntimeToolExecutionResult::new(
                    true,
                    output,
                    None,
                    HashMap::from([
                        (
                            "permission_scope".to_string(),
                            serde_json::to_value(response.scope).unwrap_or(Value::Null),
                        ),
                        (
                            "granted_permissions".to_string(),
                            serde_json::to_value(response.permissions).unwrap_or(Value::Null),
                        ),
                        (
                            "strict_auto_review".to_string(),
                            Value::Bool(response.strict_auto_review.unwrap_or(false)),
                        ),
                    ]),
                ));
            }

            if tool_runtime::request_user_input::request_user_input_canonical_tool_name(
                request.tool_name,
            )
            .is_some()
            {
                let (scope, request_call_id) = action_scope(request, &self.thread_id)
                    .map_err(RuntimeToolExecutionError::before_handler)?;
                let response_handle = self.pending_input.clone().ok_or_else(|| {
                    RuntimeToolExecutionError::new(
                        "request_user_input requires the active session response owner",
                        Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
                            "session_response_owner_missing".to_string(),
                        )),
                    )
                    .before_handler()
                })?;
                let callback = crate::request_user_input_bridge::create_request_user_input_callback(
                    self.state.action_required_state(),
                    response_handle,
                    request_call_id,
                    scope,
                    self.event_sender.clone(),
                );
                let projection = tool_runtime::request_user_input::execute_request_user_input(
                    request.params.clone(),
                    Some(&callback),
                    Duration::from_secs(
                        tool_runtime::request_user_input::DEFAULT_REQUEST_USER_INPUT_TIMEOUT_SECS,
                    ),
                )
                .await
                .map_err(|error| {
                    RuntimeToolExecutionError::new(
                        error.to_string(),
                        Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
                            "request_user_input".to_string(),
                        )),
                    )
                })?;
                return Ok(RuntimeToolExecutionResult::new(
                    true,
                    projection.output,
                    None,
                    projection.metadata.into_iter().collect(),
                ));
            }

            if let Some(route) = self.dynamic_tool_routes.get(request.tool_name) {
                return crate::current_provider_turn::dynamic_tool_bridge::call_dynamic_tool(
                    request,
                    &self.thread_id,
                    route,
                    self.pending_input.clone(),
                    &self.event_sender,
                    &self.state,
                )
                .await;
            }

            let permission_grant = if let Some(identity) = request.context.tool_identity() {
                self.state
                    .effective_permission_grant(request.context.session_id(), identity.turn_id())
                    .await
            } else {
                EffectivePermissionGrant::default()
            };
            let trusted_turn_context =
                trusted_permission_turn_context(request.turn_context, &permission_grant);
            let request = RuntimeToolExecutionRequest {
                turn_context: trusted_turn_context.as_ref(),
                ..request
            };

            orchestration::orchestrate_current_tool_execution(self, request, permission_grant).await
        })
    }
}

async fn run_guardian_tool_review(
    state: &AgentRuntimeState,
    event_sender: &UnboundedSender<AgentEvent>,
    request: RuntimeToolExecutionRequest<'_>,
    thread_id: &ThreadId,
) -> Result<(), RuntimeToolExecutionError> {
    let identity = request.context.tool_identity().ok_or_else(|| {
        RuntimeToolExecutionError::new(
            "Guardian review requires canonical tool identity",
            Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
                "guardian_review_identity_missing".to_string(),
            )),
        )
    })?;
    let turn_id = identity.turn_id().to_string();
    let command = request
        .params
        .get("cmd")
        .or_else(|| request.params.get("command"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    let cwd = request
        .context
        .working_directory()
        .to_string_lossy()
        .to_string();
    if command.is_empty() {
        return Err(RuntimeToolExecutionError::new(
            "Guardian review requires a non-empty shell command",
            Some(RuntimeToolPolicyErrorKind::PermissionDenied(
                "guardian_review_command_missing".to_string(),
            )),
        ));
    }
    let review_request = guardian_review::GuardianReviewRequest {
        session_id: request.context.session_id().to_string(),
        thread_id: thread_id.clone(),
        turn_id: turn_id.clone(),
        target_item_id: Some(identity.call_id().to_string()),
        tool_name: request.tool_name.to_string(),
        command,
        cwd,
        started_at_ms: chrono::Utc::now().timestamp_millis(),
    };
    let review_id = guardian_review::review_id();
    let action = guardian_review::action_value(&review_request);
    let _ = event_sender.send(guardian_review::started_event(
        &review_request,
        &review_id,
        action.clone(),
    ));
    let provider = state
        .provider_for_session(request.context.session_id())
        .await
        .ok_or_else(|| {
            RuntimeToolExecutionError::new(
                "Guardian review provider is not ready; the action was denied",
                Some(RuntimeToolPolicyErrorKind::PermissionDenied(
                    "guardian_review_provider_unavailable".to_string(),
                )),
            )
        });
    let result = match provider {
        Ok(provider) => {
            guardian_review::run(
                provider,
                &review_request,
                request_cancel_token(request.context.cancel_token()),
            )
            .await
        }
        Err(error) => {
            let rationale = "Guardian review provider is not ready; the action was denied.";
            let _ = event_sender.send(guardian_review::completed_event(
                &review_request,
                &review_id,
                action,
                guardian_review::GuardianReviewResult {
                    status: crate::protocol::GuardianReviewStatus::Denied,
                    risk_level: Some(crate::protocol::GuardianRiskLevel::High),
                    user_authorization: Some(crate::protocol::GuardianUserAuthorization::Unknown),
                    rationale: rationale.to_string(),
                },
                chrono::Utc::now().timestamp_millis(),
            ));
            let _ = record_guardian_denial(state, event_sender, request.context, &turn_id).await;
            return Err(error);
        }
    };
    let status = result.status;
    let rationale = result.rationale.clone();
    let _ = event_sender.send(guardian_review::completed_event(
        &review_request,
        &review_id,
        action,
        result.clone(),
        chrono::Utc::now().timestamp_millis(),
    ));
    if matches!(status, crate::protocol::GuardianReviewStatus::Approved) {
        state
            .record_guardian_non_denial(request.context.session_id(), &turn_id)
            .await;
        return Ok(());
    }
    if matches!(status, crate::protocol::GuardianReviewStatus::Denied) {
        let _ = record_guardian_denial(state, event_sender, request.context, &turn_id).await;
    }
    Err(RuntimeToolExecutionError::new(
        format!("Guardian denied shell execution: {rationale}"),
        Some(RuntimeToolPolicyErrorKind::PermissionDenied(
            "guardian_review_denied".to_string(),
        )),
    ))
}

async fn record_guardian_denial(
    state: &AgentRuntimeState,
    event_sender: &UnboundedSender<AgentEvent>,
    context: &tool_runtime::tool_executor::RuntimeToolExecutionContext,
    turn_id: &str,
) -> Option<String> {
    let (consecutive_denials, recent_denials) = state
        .record_guardian_denial(context.session_id(), turn_id)
        .await?;
    let message = format!(
        "Automatic approval review rejected too many approval requests for this turn ({consecutive_denials} consecutive, {recent_denials} in the last 5 reviews); interrupting the turn."
    );
    let _ = event_sender.send(AgentEvent::GuardianWarning {
        message: message.clone(),
    });
    if let Some(cancel_token) = context.cancel_token() {
        cancel_token.cancel();
    }
    Some(message)
}

fn trusted_permission_turn_context(
    source: Option<&tool_runtime::tool_executor::RuntimeToolTurnContext>,
    grant: &EffectivePermissionGrant,
) -> Option<tool_runtime::tool_executor::RuntimeToolTurnContext> {
    let mut context = source.cloned();
    if let Some(context) = context.as_mut() {
        context.metadata.remove(GRANTED_PERMISSIONS_METADATA_KEY);
        context.metadata.remove(STRICT_AUTO_REVIEW_METADATA_KEY);
    }
    if grant.strict_auto_review {
        context
            .get_or_insert_with(Default::default)
            .metadata
            .insert(
                STRICT_AUTO_REVIEW_METADATA_KEY.to_string(),
                Value::Bool(true),
            );
    }
    context
}

fn mcp_tool_id(
    request: RuntimeToolExecutionRequest<'_>,
) -> Result<String, RuntimeToolExecutionError> {
    let identity = request
        .context
        .tool_identity()
        .ok_or_else(|| mcp_identity_error("tool identity"))?;
    mcp_identity_value(identity.call_id(), "call_id")
}

async fn await_mcp_call(
    event_sender: &UnboundedSender<AgentEvent>,
    tool_id: &str,
    route: &tool_runtime::mcp_connection::McpStepRouteIdentity,
    mut call: tool_runtime::mcp_connection::McpConnectionCall,
) -> Result<RuntimeToolExecutionResult, RuntimeToolExecutionError> {
    let mut notifications_open = true;
    loop {
        tokio::select! {
            biased;
            notification = call.notifications.next(), if notifications_open => {
                match notification {
                    Some(notification) => {
                        emit_mcp_progress(event_sender, tool_id, route, notification)
                    }
                    None => notifications_open = false,
                }
            }
            result = &mut call.response => return project_call_result(result),
        }
    }
}

fn emit_mcp_progress(
    event_sender: &UnboundedSender<AgentEvent>,
    tool_id: &str,
    route: &tool_runtime::mcp_connection::McpStepRouteIdentity,
    notification: ServerNotification,
) {
    if !matches!(notification, ServerNotification::ProgressNotification(_)) {
        return;
    }
    for projection in
        tool_runtime::mcp_notification::project_mcp_notification(tool_id, notification)
    {
        let tool_runtime::mcp_notification::McpNotificationProjection::ToolProgress {
            tool_id,
            progress,
        } = projection
        else {
            continue;
        };
        let Some(message) = progress
            .message
            .map(|message| message.trim().to_string())
            .filter(|message| !message.is_empty())
        else {
            continue;
        };
        let mut metadata = progress.metadata.unwrap_or_default();
        metadata.insert(
            "server_name".to_string(),
            Value::String(route.server_name.clone()),
        );
        metadata.insert(
            "tool_name".to_string(),
            Value::String(route.tool_name.clone()),
        );
        metadata.insert(
            "runtime_tool_name".to_string(),
            Value::String(route.runtime_tool_name.clone()),
        );
        let _ = event_sender.send(AgentEvent::ToolProgress {
            tool_id,
            progress: AgentToolProgressPayload {
                message: Some(message),
                progress: progress.progress,
                total: progress.total,
                metadata: Some(metadata),
            },
        });
    }
}

pub(super) fn mcp_call_scope(
    request: RuntimeToolExecutionRequest<'_>,
) -> Result<tool_runtime::mcp_connection::McpCallScope, RuntimeToolExecutionError> {
    let identity = request
        .context
        .tool_identity()
        .ok_or_else(|| mcp_identity_error("tool identity"))?;
    let turn_id = mcp_identity_value(identity.turn_id(), "turn_id")?;
    tool_runtime::mcp_connection::McpCallScope::new(Some(turn_id)).map_err(mcp_identity_error)
}

fn mcp_identity_value(value: &str, field: &str) -> Result<String, RuntimeToolExecutionError> {
    (!value.trim().is_empty())
        .then(|| value.to_string())
        .ok_or_else(|| mcp_identity_error(field))
}

fn mcp_identity_error(field: &str) -> RuntimeToolExecutionError {
    RuntimeToolExecutionError::new(
        format!("MCP call requires canonical {field}"),
        Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
            "mcp_call_scope_missing".to_string(),
        )),
    )
}

pub(super) fn action_scope(
    request: RuntimeToolExecutionRequest<'_>,
    thread_id: &ThreadId,
) -> Result<
    (
        Option<agent_protocol::action_required::ActionRequiredScope>,
        String,
    ),
    RuntimeToolExecutionError,
> {
    let session_id = canonical_identity_value(request.context.session_id(), "session_id")?;
    let thread_id = canonical_identity_value(thread_id.as_str(), "thread_id")?;
    let identity = request
        .context
        .tool_identity()
        .ok_or_else(|| approval_identity_error("tool identity"))?;
    let turn_id = canonical_identity_value(identity.turn_id(), "turn_id")?;
    let tool_call_id = canonical_identity_value(identity.call_id(), "call_id")?;
    Ok((
        agent_protocol::action_required::ActionRequiredScope::from_parts(
            Some(session_id),
            Some(thread_id),
            Some(turn_id),
        ),
        tool_call_id,
    ))
}

fn canonical_identity_value(value: &str, field: &str) -> Result<String, RuntimeToolExecutionError> {
    (!value.trim().is_empty())
        .then(|| value.to_string())
        .ok_or_else(|| approval_identity_error(field))
}

fn approval_identity_error(field: &str) -> RuntimeToolExecutionError {
    RuntimeToolExecutionError::new(
        format!("tool approval requires canonical {field}"),
        Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
            "tool_approval_identity_missing".to_string(),
        )),
    )
}

fn request_cancel_token(token: Option<&CancellationToken>) -> CancellationToken {
    token.cloned().unwrap_or_default()
}

fn project_mcp_error(error: ErrorData) -> RuntimeToolExecutionError {
    RuntimeToolExecutionError::new(
        error.message.to_string(),
        Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
            "mcp_dispatch".to_string(),
        )),
    )
}

pub(super) fn project_call_result(
    result: Result<CallToolResult, ErrorData>,
) -> Result<RuntimeToolExecutionResult, RuntimeToolExecutionError> {
    let result = result.map_err(project_mcp_error)?;
    let mut metadata = HashMap::new();
    if let Some(meta) = result.meta.clone() {
        metadata.insert("meta".to_string(), Value::Object(meta.0));
    }
    let output = call_result_text(&result);
    let success = !result.is_error.unwrap_or(false);
    let error = (!success)
        .then(|| output.clone())
        .filter(|value| !value.is_empty());
    let projection = RuntimeToolExecutionResult::new(success, output, error, metadata);
    Ok(match result.structured_content {
        Some(content) => projection.with_structured_content(content),
        None => projection,
    })
}

fn project_runtime_dispatch_result(
    result: Result<CallToolResult, ErrorData>,
) -> Result<RuntimeToolExecutionResult, RuntimeToolExecutionError> {
    result
        .map_err(project_runtime_dispatch_error)
        .and_then(|result| project_call_result(Ok(result)))
}

fn project_runtime_dispatch_error(error: ErrorData) -> RuntimeToolExecutionError {
    let handler_executed = error
        .data
        .as_ref()
        .and_then(|data| {
            data.get(tool_runtime::tool_result_projection::TOOL_HANDLER_EXECUTED_METADATA_KEY)
        })
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let error = project_mcp_error(error);
    if handler_executed {
        error
    } else {
        error.before_handler()
    }
}

fn call_result_text(result: &CallToolResult) -> String {
    let value = serde_json::to_value(result).unwrap_or(Value::Null);
    let mut text = Vec::new();
    collect_text_fields(&value, &mut text);
    if text.is_empty() {
        serde_json::to_string(&value).unwrap_or_default()
    } else {
        text.join("\n")
    }
}

fn collect_text_fields(value: &Value, target: &mut Vec<String>) {
    match value {
        Value::Object(object) => {
            if object.get("type").and_then(Value::as_str) == Some("text") {
                if let Some(text) = object.get("text").and_then(Value::as_str) {
                    target.push(text.to_string());
                    return;
                }
            }
            for value in object.values() {
                collect_text_fields(value, target);
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_text_fields(value, target);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_runtime::action_required::ActionRequiredRequest;
    use app_server_protocol::protocol::v2::GrantedPermissionProfile;
    use rmcp::model::{
        Content, NumberOrString, ProgressNotification, ProgressNotificationMethod,
        ProgressNotificationParam, ProgressToken,
    };

    #[test]
    fn shell_tool_approval_materializes_current_projection() {
        let metadata = HashMap::from([
            (
                "reasonCode".to_string(),
                serde_json::json!("shell_command_requires_approval"),
            ),
            ("cwd".to_string(), serde_json::json!("/Users/coso/project")),
        ]);
        let approval = tool_runtime::execution_approval::execution_approval_projection(
            "exec_command",
            &metadata,
        );
        let queued = ActionRequiredRequest {
            id: "approval-1".to_string(),
            action_type: agent_protocol::action_required::TOOL_CONFIRMATION_ACTION_TYPE.to_string(),
            tool_id: Some("call-1".to_string()),
            message: "Allow?".to_string(),
            requested_schema: serde_json::json!({}),
            available_decisions: approval.available_decisions.clone(),
            scope: None,
            created_at_ms: Some(1),
            deadline_at_ms: Some(2),
        };

        let projection = orchestration::materialize_tool_approval_action(
            &queued,
            "exec_command",
            &serde_json::json!({ "cmd": "cargo test" }),
            "Allow?",
            &approval,
        );

        assert_eq!(projection.data["actionKind"], "tool_execution_policy");
        assert_eq!(projection.data["action_kind"], "tool_execution_policy");
        assert_eq!(projection.data["toolFamily"], "shell_command");
        assert_eq!(projection.data["tool_family"], "shell_command");
        assert_eq!(projection.data["contractKey"], "shell_command");
        assert_eq!(projection.data["contract_key"], "shell_command");
        assert_eq!(
            projection.data["runtime_contract"]["session_cache_supported"],
            false
        );
        assert_eq!(
            projection.data["availableDecisions"],
            serde_json::json!(["allow_once", "decline", "cancel"])
        );
        assert_eq!(projection.data["arguments"]["cmd"], "cargo test");
        assert_eq!(projection.data["prompt"], "Allow?");
        assert!(projection.data["approvalScope"].get("cwd").is_none());
        assert!(!projection.data["approvalScope"]
            .to_string()
            .contains("/Users/coso/project"));
        assert_eq!(
            projection.data["approvalScope"],
            projection.data["approval_scope"]
        );
    }

    #[test]
    fn permission_context_accepts_only_recorded_strict_review_state() {
        let source = tool_runtime::tool_executor::RuntimeToolTurnContext {
            metadata: HashMap::from([
                (
                    GRANTED_PERMISSIONS_METADATA_KEY.to_string(),
                    serde_json::json!({ "network": { "enabled": true } }),
                ),
                (
                    STRICT_AUTO_REVIEW_METADATA_KEY.to_string(),
                    Value::Bool(true),
                ),
                ("safe".to_string(), Value::String("kept".to_string())),
            ]),
            ..Default::default()
        };
        let sanitized =
            trusted_permission_turn_context(Some(&source), &EffectivePermissionGrant::default())
                .expect("sanitized turn context");
        assert!(!sanitized
            .metadata
            .contains_key(GRANTED_PERMISSIONS_METADATA_KEY));
        assert!(!sanitized
            .metadata
            .contains_key(STRICT_AUTO_REVIEW_METADATA_KEY));
        assert_eq!(
            sanitized.metadata.get("safe"),
            Some(&serde_json::json!("kept"))
        );

        let trusted = trusted_permission_turn_context(
            Some(&source),
            &EffectivePermissionGrant {
                permissions: GrantedPermissionProfile::default(),
                strict_auto_review: true,
            },
        )
        .expect("trusted turn context");
        assert_eq!(
            trusted.metadata.get(STRICT_AUTO_REVIEW_METADATA_KEY),
            Some(&Value::Bool(true))
        );
    }

    #[tokio::test]
    async fn mcp_call_emits_only_non_empty_progress_notifications() {
        let call = tool_runtime::mcp_connection::McpConnectionCall {
            response: Box::pin(async {
                tokio::task::yield_now().await;
                Ok(CallToolResult::success(vec![Content::text("done")]))
            }),
            notifications: Box::pin(futures::stream::iter([
                progress_notification(Some("   ")),
                progress_notification(Some("正在检索文档")),
            ])),
        };
        let (event_sender, mut event_receiver) = tokio::sync::mpsc::unbounded_channel();
        let route = tool_runtime::mcp_connection::McpStepRouteIdentity {
            server_name: "docs".to_string(),
            tool_name: "search".to_string(),
            runtime_tool_name: "mcp__docs__search".to_string(),
            mcp_app_resource_uri: None,
            plugin_id: None,
        };

        let result = await_mcp_call(&event_sender, "mcp-call-1", &route, call)
            .await
            .expect("MCP result");

        assert_eq!(result.output, "done");
        let AgentEvent::ToolProgress { tool_id, progress } =
            event_receiver.try_recv().expect("MCP progress event")
        else {
            panic!("expected MCP progress event");
        };
        assert_eq!(tool_id, "mcp-call-1");
        assert_eq!(progress.message.as_deref(), Some("正在检索文档"));
        assert_eq!(
            progress
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.get("notification_kind"))
                .and_then(Value::as_str),
            Some("mcp_progress")
        );
        assert_eq!(
            progress
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.get("server_name"))
                .and_then(Value::as_str),
            Some("docs")
        );
        assert_eq!(
            progress
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.get("tool_name"))
                .and_then(Value::as_str),
            Some("search")
        );
        assert!(event_receiver.try_recv().is_err());
    }

    fn progress_notification(message: Option<&str>) -> ServerNotification {
        ServerNotification::ProgressNotification(ProgressNotification {
            method: ProgressNotificationMethod,
            params: ProgressNotificationParam {
                progress_token: ProgressToken(NumberOrString::Number(1)),
                progress: 1.0,
                total: Some(2.0),
                message: message.map(str::to_string),
            },
            extensions: Default::default(),
        })
    }
}
