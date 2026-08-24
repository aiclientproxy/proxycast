use super::{
    action_scope, await_mcp_call, mcp_call_scope, mcp_identity_error, mcp_tool_id,
    project_mcp_error, project_runtime_dispatch_result, request_cancel_token,
    run_guardian_tool_review, CurrentTurnToolExecutor,
};
use crate::agent_tools::execution::{
    decide_tool_execution, persisted_tool_execution_policy_from_metadata, ToolExecutionDecision,
    ToolExecutionDecisionInput, ToolExecutionDecisionKind, ToolExecutionResolverInput,
};
use crate::protocol::AgentEvent;
use crate::request_tool_policy::is_same_tool;
use crate::runtime_state::{AgentRuntimeState, EffectivePermissionGrant};
use agent_protocol::action_required::{tool_confirmation_action, ActionRequiredProjection};
use agent_protocol::ThreadId;
use agent_runtime::action_required::ActionRequiredRequest;
use agent_runtime::session_loop::{RuntimeSessionInputHandle, RuntimeSessionResponseKind};
use app_server_protocol::protocol::v2::{
    DynamicToolCallApproval, FileSystemAccessMode, GrantedPermissionProfile,
};
use rmcp::model::CallToolRequestParam;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Duration;
use tokio::sync::mpsc::UnboundedSender;
use tool_runtime::execution_approval::{BROWSER_ACTION_CONTRACT_KEY, BROWSER_ACTION_TOOL_FAMILY};
use tool_runtime::execution_orchestrator::{
    orchestrate_runtime_tool_execution, RuntimeToolApprovalFuture, RuntimeToolApprovalHandler,
    RuntimeToolApprovalKind, RuntimeToolApprovalPhase, RuntimeToolApprovalPolicy,
    RuntimeToolApprovalRequest, RuntimeToolApprovalSource, RuntimeToolAttemptFuture,
    RuntimeToolAttemptRunner, RuntimeToolExecutionAttempt, RuntimeToolInitialApproval,
    RuntimeToolOrchestrationInput, RuntimeToolSandboxPolicy,
};
use tool_runtime::gateway_dispatch_execution::{
    execute_runtime_gateway_dispatch_tool, RuntimeGatewayDispatchToolRequest,
};
use tool_runtime::native_dispatch_execution::{
    execute_runtime_native_dispatch_tool_typed, RuntimeNativeDispatchToolRequest,
};
use tool_runtime::tool_executor::{
    RuntimeToolExecutionError, RuntimeToolExecutionRequest, RuntimeToolExecutionResult,
    RuntimeToolPolicyErrorKind, TOOL_APPROVAL_GRANTED_METADATA_KEY,
};

const TOOL_CONFIRMATION_TIMEOUT: Duration = Duration::from_secs(300);

pub(super) async fn orchestrate_current_tool_execution(
    executor: &CurrentTurnToolExecutor,
    request: RuntimeToolExecutionRequest<'_>,
    permission_grant: EffectivePermissionGrant,
) -> Result<RuntimeToolExecutionResult, RuntimeToolExecutionError> {
    let mut decision = current_tool_execution_decision(request);
    normalize_deferred_sandbox_decision(&mut decision, request);
    annotate_shell_approval_contract(&mut decision, request);

    match decision.kind {
        ToolExecutionDecisionKind::Allow | ToolExecutionDecisionKind::RequiresApproval => {}
        ToolExecutionDecisionKind::Deny => {
            return Err(RuntimeToolExecutionError::new(
                decision.reason,
                Some(RuntimeToolPolicyErrorKind::PermissionDenied(
                    decision.reason_code,
                )),
            )
            .before_handler());
        }
        ToolExecutionDecisionKind::SandboxBlocked => {
            return Err(RuntimeToolExecutionError::new(
                decision.reason,
                Some(RuntimeToolPolicyErrorKind::SandboxDenied(
                    decision.reason_code,
                )),
            )
            .before_handler());
        }
    }

    let identity = request.context.tool_identity().cloned().ok_or_else(|| {
        RuntimeToolExecutionError::new(
            "orchestrated tool execution requires canonical tool identity",
            Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
                "tool_execution_identity_missing".to_string(),
            )),
        )
        .before_handler()
    })?;
    let approval_policy = RuntimeToolApprovalPolicy::from_label(
        request
            .turn_context
            .and_then(|context| context.approval_policy.as_deref()),
    );
    let strict_guardian = permission_grant.strict_auto_review
        && tool_runtime::shell::is_shell_tool_name(request.tool_name);
    let explicit_sandbox_escalation = explicit_sandbox_escalation(request);
    let ambient_sandbox = ambient_sandbox_policy(request, &decision);
    let requested_sandbox = if explicit_sandbox_escalation {
        RuntimeToolSandboxPolicy::DangerFullAccess
    } else {
        ambient_sandbox
    };
    let effective_sandbox = if explicit_sandbox_escalation
        && has_denied_file_system_permissions(&permission_grant.permissions)
    {
        ambient_sandbox
    } else {
        requested_sandbox
    };
    let approval_key = shell_approval_key(request, requested_sandbox);
    let cached_approval = match approval_key.as_deref() {
        Some(key) => {
            executor
                .state
                .has_shell_approval(request.context.session_id(), key)
                .await
        }
        None => false,
    };
    let initial_approval = if strict_guardian {
        RuntimeToolInitialApproval::Required(RuntimeToolApprovalKind::Guardian)
    } else if decision.kind == ToolExecutionDecisionKind::RequiresApproval && cached_approval {
        RuntimeToolInitialApproval::Cached
    } else if decision.kind == ToolExecutionDecisionKind::RequiresApproval {
        RuntimeToolInitialApproval::Required(RuntimeToolApprovalKind::User)
    } else {
        RuntimeToolInitialApproval::NotRequired
    };
    let managed_network_host = decision
        .metadata
        .get("networkHost")
        .and_then(Value::as_str)
        .map(str::to_string);
    let input = RuntimeToolOrchestrationInput {
        identity,
        approval_policy,
        initial_approval,
        initial_approval_reason: Some(decision.reason.clone()),
        requested_sandbox_policy: requested_sandbox,
        effective_sandbox_policy: effective_sandbox,
        granted_permissions: permission_grant.permissions,
        managed_network_host,
        strict_guardian,
        explicit_sandbox_escalation,
        sandbox_denial_retry_allowed: sandbox_retry_allowed(request.tool_name, approval_policy),
        network_denial_retry_allowed: network_retry_allowed(request.tool_name, approval_policy),
        cancel_token: request.context.cancel_token().cloned(),
    };
    let approvals = CurrentToolApprovalHandler {
        executor,
        request,
        decision,
        approval_key,
    };
    let runner = CurrentToolAttemptRunner { executor, request };
    orchestrate_runtime_tool_execution(input, &approvals, &runner).await
}

struct CurrentToolApprovalHandler<'a> {
    executor: &'a CurrentTurnToolExecutor,
    request: RuntimeToolExecutionRequest<'a>,
    decision: ToolExecutionDecision,
    approval_key: Option<String>,
}

impl RuntimeToolApprovalHandler for CurrentToolApprovalHandler<'_> {
    fn approve<'a>(
        &'a self,
        approval_request: RuntimeToolApprovalRequest,
    ) -> RuntimeToolApprovalFuture<'a> {
        Box::pin(async move {
            match approval_request.kind {
                RuntimeToolApprovalKind::Guardian => run_guardian_tool_review(
                    &self.executor.state,
                    &self.executor.event_sender,
                    self.request,
                    &self.executor.thread_id,
                )
                .await
                .map_err(RuntimeToolExecutionError::before_handler),
                RuntimeToolApprovalKind::User => {
                    let decision = approval_decision(&self.decision, &approval_request);
                    wait_for_tool_approval(
                        &self.executor.state,
                        &self.executor.event_sender,
                        self.request,
                        &self.executor.thread_id,
                        self.executor.pending_input.as_ref(),
                        &decision.reason,
                        &decision.metadata,
                        self.approval_key.as_deref(),
                    )
                    .await
                    .map_err(RuntimeToolExecutionError::before_handler)
                }
            }
        })
    }
}

struct CurrentToolAttemptRunner<'a> {
    executor: &'a CurrentTurnToolExecutor,
    request: RuntimeToolExecutionRequest<'a>,
}

impl RuntimeToolAttemptRunner for CurrentToolAttemptRunner<'_> {
    fn run<'a>(&'a self, attempt: RuntimeToolExecutionAttempt) -> RuntimeToolAttemptFuture<'a> {
        Box::pin(async move {
            let mut turn_context = self.request.turn_context.cloned();
            if attempt.approval_source() != RuntimeToolApprovalSource::Config {
                turn_context
                    .get_or_insert_with(Default::default)
                    .metadata
                    .insert(
                        TOOL_APPROVAL_GRANTED_METADATA_KEY.to_string(),
                        Value::Bool(true),
                    );
            }
            let context = self.request.context.clone().with_execution_attempt(attempt);
            execute_current_tool_attempt(
                self.executor,
                RuntimeToolExecutionRequest {
                    tool_name: self.request.tool_name,
                    params: self.request.params,
                    context: &context,
                    turn_context: turn_context.as_ref().or(self.request.turn_context),
                },
            )
            .await
        })
    }
}

async fn execute_current_tool_attempt(
    executor: &CurrentTurnToolExecutor,
    request: RuntimeToolExecutionRequest<'_>,
) -> Result<RuntimeToolExecutionResult, RuntimeToolExecutionError> {
    let attempt = request
        .context
        .execution_attempt()
        .cloned()
        .ok_or_else(|| {
            RuntimeToolExecutionError::new(
                "tool attempt context is missing",
                Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
                    "tool_execution_attempt_missing".to_string(),
                )),
            )
            .before_handler()
        })?;

    if tool_runtime::unified_exec::is_unified_exec_tool_name(request.tool_name) {
        let gateway = executor
            .state
            .live_execution_process_gateway()
            .await
            .ok_or_else(|| {
                RuntimeToolExecutionError::new(
                    "unified exec process gateway is unavailable",
                    Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
                        "unified_exec_gateway_unavailable".to_string(),
                    )),
                )
                .before_handler()
            })?;
        return tool_runtime::unified_exec::execute_runtime_unified_exec_tool(
            gateway,
            tool_runtime::unified_exec::RuntimeUnifiedExecToolRequest {
                tool_name: request.tool_name,
                params: request.params,
                thread_id: executor.thread_id.as_str(),
                environment_id: request.context.environment_id().unwrap_or("local"),
                working_directory: request.context.working_directory().clone(),
                environment: request.context.environment().clone(),
                tool_call_id: attempt.identity().call_id().to_string(),
                cancel_token: request.context.cancel_token().cloned(),
                turn_context: request.turn_context,
                attempt: Some(attempt),
            },
        )
        .await;
    }

    if let Some(agent_control_gateway) = executor.agent_control_gateway.as_ref() {
        if let Some(result) = tool_runtime::agent_control::execute_agent_control_tool(
            agent_control_gateway.gateway(),
            executor.thread_id.as_str(),
            request,
        )
        .await
        {
            return result;
        }
    }

    if let Some(mut result) = execute_runtime_gateway_dispatch_tool(
        executor.state.gateway_tools(),
        RuntimeGatewayDispatchToolRequest {
            tool_name: request.tool_name,
            params: request.params,
            working_directory: request.context.working_directory().clone(),
            session_id: request.context.session_id().to_string(),
            cancel_token: request.context.cancel_token().cloned(),
            turn_context: request.turn_context,
        },
    )
    .await
    {
        if is_same_tool(
            request.tool_name,
            tool_runtime::tool_search::TOOL_SEARCH_TOOL_NAME,
        ) {
            if let Ok(result) = &mut result {
                executor
                    .deferred_tools
                    .activate_from_tool_search_result(result)
                    .await;
            }
        }
        return project_runtime_dispatch_result(result);
    }

    if let Some(result) =
        execute_runtime_native_dispatch_tool_typed(RuntimeNativeDispatchToolRequest {
            tool_name: request.tool_name,
            params: request.params,
            working_directory: request.context.working_directory().clone(),
            session_id: request.context.session_id().to_string(),
            cancel_token: request.context.cancel_token().cloned(),
            turn_context: request.turn_context,
            attempt: Some(attempt),
            filesystem_gateway: request.context.filesystem_gateway().cloned(),
        })
        .await
    {
        return result;
    }

    let cancel_token = request_cancel_token(request.context.cancel_token());
    let mcp_request = CallToolRequestParam {
        name: request.tool_name.to_string().into(),
        arguments: request.params.as_object().cloned(),
    };
    let mcp_scope = mcp_call_scope(request).map_err(RuntimeToolExecutionError::before_handler)?;
    let tool_id = mcp_tool_id(request).map_err(RuntimeToolExecutionError::before_handler)?;
    let mcp_route = executor
        .mcp_snapshot
        .route_identity(request.tool_name)
        .ok_or_else(|| mcp_identity_error("route identity"))
        .map_err(RuntimeToolExecutionError::before_handler)?;
    let call = executor
        .mcp_snapshot
        .dispatch(mcp_request, mcp_scope, cancel_token)
        .await
        .map_err(|error| project_mcp_error(error).before_handler())?;
    await_mcp_call(&executor.event_sender, &tool_id, &mcp_route, call).await
}

fn current_tool_execution_decision(
    request: RuntimeToolExecutionRequest<'_>,
) -> ToolExecutionDecision {
    let request_metadata = request
        .turn_context
        .map(|context| serde_json::to_value(&context.metadata).unwrap_or(Value::Null));
    let persisted_policy = persisted_tool_execution_policy_from_metadata(request_metadata.as_ref());
    decide_tool_execution(ToolExecutionDecisionInput {
        tool_name: request.tool_name,
        params: request.params,
        working_directory: request.context.working_directory(),
        surface: "current_provider_turn",
        auto_mode: false,
        bypass_restrictions: false,
        approval_policy: request
            .turn_context
            .and_then(|context| context.approval_policy.as_deref()),
        requested_sandbox_policy: request
            .turn_context
            .and_then(|context| context.sandbox_policy.as_deref()),
        resolver_input: ToolExecutionResolverInput {
            persisted_policy: persisted_policy.as_ref(),
            request_metadata: request_metadata.as_ref(),
        },
    })
}

fn annotate_shell_approval_contract(
    decision: &mut ToolExecutionDecision,
    request: RuntimeToolExecutionRequest<'_>,
) {
    if decision.kind != ToolExecutionDecisionKind::RequiresApproval
        || request.tool_name != tool_runtime::unified_exec::EXEC_COMMAND_TOOL_NAME
    {
        return;
    }
    decision.metadata.insert(
        "availableDecisions".to_string(),
        json!(["allow_once", "allow_for_session", "decline", "cancel"]),
    );
    decision.metadata.insert(
        "runtime_contract".to_string(),
        json!({
            "contract_key": tool_runtime::execution_approval::SHELL_COMMAND_CONTRACT_KEY,
            "tool_family": tool_runtime::execution_approval::SHELL_TOOL_FAMILY,
            "session_cache_supported": true,
        }),
    );
}

fn shell_approval_key(
    request: RuntimeToolExecutionRequest<'_>,
    requested_sandbox: RuntimeToolSandboxPolicy,
) -> Option<String> {
    if request.tool_name != tool_runtime::unified_exec::EXEC_COMMAND_TOOL_NAME {
        return None;
    }
    let command = request
        .params
        .get("cmd")
        .or_else(|| request.params.get("command"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|command| !command.is_empty())?;
    let requested_permissions = [
        request.params.get("sandbox_permissions"),
        request.params.get("additional_permissions"),
        request.params.get("prefix_rule"),
    ]
    .into_iter()
    .flatten()
    .map(Value::to_string)
    .collect::<Vec<_>>()
    .join(",");
    Some(format!(
        "shell-v1\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}",
        command,
        request.context.working_directory().to_string_lossy(),
        requested_sandbox.label().unwrap_or("none"),
        requested_permissions
    ))
}

fn normalize_deferred_sandbox_decision(
    decision: &mut ToolExecutionDecision,
    request: RuntimeToolExecutionRequest<'_>,
) {
    if decision.kind == ToolExecutionDecisionKind::SandboxBlocked
        && decision.reason_code == "read_only_sandbox_blocks_shell_command"
    {
        let approval_policy = RuntimeToolApprovalPolicy::from_label(
            request
                .turn_context
                .and_then(|context| context.approval_policy.as_deref()),
        );
        decision.kind = if tool_runtime::shell::is_shell_tool_name(request.tool_name)
            && matches!(
                approval_policy,
                RuntimeToolApprovalPolicy::OnRequest
                    | RuntimeToolApprovalPolicy::UnlessTrusted
                    | RuntimeToolApprovalPolicy::Granular
            ) {
            ToolExecutionDecisionKind::RequiresApproval
        } else {
            ToolExecutionDecisionKind::Allow
        };
        decision.reason_code = "sandbox_decision_deferred".to_string();
        decision.reason = "sandbox enforcement is deferred to the execution attempt".to_string();
        decision
            .metadata
            .insert("decisionKind".to_string(), json!(decision.kind));
        decision.metadata.insert(
            "reasonCode".to_string(),
            json!(decision.reason_code.clone()),
        );
        decision
            .metadata
            .insert("reason".to_string(), json!(decision.reason.clone()));
    }
}

fn explicit_sandbox_escalation(request: RuntimeToolExecutionRequest<'_>) -> bool {
    request.tool_name == tool_runtime::unified_exec::EXEC_COMMAND_TOOL_NAME
        && request
            .params
            .get("sandbox_permissions")
            .and_then(Value::as_str)
            .is_some_and(|value| value.trim() == "require_escalated")
}

fn ambient_sandbox_policy(
    request: RuntimeToolExecutionRequest<'_>,
    decision: &ToolExecutionDecision,
) -> RuntimeToolSandboxPolicy {
    use tool_runtime::execution_policy::ToolExecutionSandboxProfile;

    if decision.policy_resolution.policy.sandbox_profile
        != ToolExecutionSandboxProfile::WorkspaceCommand
    {
        return RuntimeToolSandboxPolicy::None;
    }
    let requested = RuntimeToolSandboxPolicy::from_label(
        request
            .turn_context
            .and_then(|context| context.sandbox_policy.as_deref()),
    );
    if requested == RuntimeToolSandboxPolicy::None {
        RuntimeToolSandboxPolicy::WorkspaceWrite
    } else {
        requested
    }
}

fn sandbox_retry_allowed(tool_name: &str, policy: RuntimeToolApprovalPolicy) -> bool {
    if tool_name == tool_runtime::apply_patch::APPLY_PATCH_TOOL_NAME {
        return matches!(
            policy,
            RuntimeToolApprovalPolicy::OnRequest
                | RuntimeToolApprovalPolicy::UnlessTrusted
                | RuntimeToolApprovalPolicy::Granular
        );
    }
    tool_name == tool_runtime::unified_exec::EXEC_COMMAND_TOOL_NAME
        && matches!(
            policy,
            RuntimeToolApprovalPolicy::UnlessTrusted | RuntimeToolApprovalPolicy::Granular
        )
}

fn network_retry_allowed(tool_name: &str, policy: RuntimeToolApprovalPolicy) -> bool {
    tool_name == tool_runtime::unified_exec::EXEC_COMMAND_TOOL_NAME
        && matches!(
            policy,
            RuntimeToolApprovalPolicy::OnRequest
                | RuntimeToolApprovalPolicy::UnlessTrusted
                | RuntimeToolApprovalPolicy::Granular
        )
}

fn has_denied_file_system_permissions(permissions: &GrantedPermissionProfile) -> bool {
    permissions
        .file_system
        .as_ref()
        .and_then(|file_system| file_system.entries.as_deref())
        .is_some_and(|entries| {
            entries
                .iter()
                .any(|entry| entry.access == FileSystemAccessMode::Deny)
        })
}

fn approval_decision(
    base: &ToolExecutionDecision,
    request: &RuntimeToolApprovalRequest,
) -> ToolExecutionDecision {
    let mut decision = base.clone();
    if let Some(reason) = request.reason.as_deref() {
        decision.reason = reason.to_string();
    }
    let phase = match request.phase {
        RuntimeToolApprovalPhase::Initial => "initial",
        RuntimeToolApprovalPhase::Escalation => "escalation",
    };
    decision
        .metadata
        .insert("approvalPhase".to_string(), json!(phase));
    if request.phase == RuntimeToolApprovalPhase::Escalation {
        decision
            .metadata
            .insert("retryReason".to_string(), json!(decision.reason.clone()));
    }
    if let Some(host) = request.network_host.as_deref() {
        decision
            .metadata
            .insert("networkHost".to_string(), json!(host));
    }
    if let Some(denial_kind) = request.denial_kind {
        let denial_kind = match denial_kind {
            tool_runtime::execution_orchestrator::RuntimeToolDenialKind::Sandbox => "sandbox",
            tool_runtime::execution_orchestrator::RuntimeToolDenialKind::ManagedNetwork => {
                "managed_network"
            }
        };
        decision
            .metadata
            .insert("denialKind".to_string(), json!(denial_kind));
    }
    decision
}

async fn wait_for_tool_approval(
    state: &AgentRuntimeState,
    event_sender: &UnboundedSender<AgentEvent>,
    request: RuntimeToolExecutionRequest<'_>,
    thread_id: &ThreadId,
    pending_input: Option<&RuntimeSessionInputHandle>,
    prompt: &str,
    metadata: &HashMap<String, Value>,
    approval_key: Option<&str>,
) -> Result<(), RuntimeToolExecutionError> {
    let response_handle = pending_input.cloned().ok_or_else(|| {
        RuntimeToolExecutionError::new(
            "tool approval requires the active session response owner",
            Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
                "session_response_owner_missing".to_string(),
            )),
        )
    })?;
    let (scope, tool_call_id) = action_scope(request, thread_id)?;
    let tool_name = request.tool_name.to_string();
    let arguments = request.params.clone();
    let approval = tool_runtime::execution_approval::execution_approval_projection(
        request.tool_name,
        metadata,
    );
    let response = state
        .action_required_state()
        .request_action_and_wait_with_notification(
            response_handle,
            RuntimeSessionResponseKind::Approval,
            agent_protocol::action_required::TOOL_CONFIRMATION_ACTION_TYPE,
            Some(tool_call_id),
            approval.available_decisions.clone(),
            scope,
            prompt.to_string(),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "confirmed": { "type": "boolean" }
                },
                "required": ["confirmed"]
            }),
            TOOL_CONFIRMATION_TIMEOUT,
            {
                let event_sender = event_sender.clone();
                move |queued| {
                    let projection = materialize_tool_approval_action(
                        queued, &tool_name, &arguments, prompt, &approval,
                    );
                    let _ = event_sender.send(AgentEvent::ActionRequired {
                        request_id: projection.id,
                        action_type: projection.action_type,
                        data: projection.data,
                        scope: projection.scope,
                    });
                }
            },
        )
        .await
        .map_err(|error| {
            RuntimeToolExecutionError::new(
                format!("工具审批等待失败: {error}"),
                Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
                    "tool_approval_wait_failed".to_string(),
                )),
            )
        })?;

    if response.get("confirmed").and_then(Value::as_bool) == Some(true) {
        if approval_key.is_some()
            && response
                .get("decision")
                .and_then(Value::as_str)
                .is_some_and(|decision| decision == "allow_for_session")
        {
            state
                .record_shell_approval(
                    request.context.session_id(),
                    approval_key.expect("checked above").to_string(),
                )
                .await;
        }
        return Ok(());
    }
    Err(RuntimeToolExecutionError::new(
        "用户拒绝工具执行",
        Some(RuntimeToolPolicyErrorKind::PermissionDenied(
            "tool_approval_declined".to_string(),
        )),
    ))
}

pub(in crate::current_provider_turn) async fn wait_for_browser_action_approval(
    state: &AgentRuntimeState,
    event_sender: &UnboundedSender<AgentEvent>,
    request: RuntimeToolExecutionRequest<'_>,
    thread_id: &ThreadId,
    pending_input: Option<&RuntimeSessionInputHandle>,
    descriptor: &DynamicToolCallApproval,
) -> Result<(), RuntimeToolExecutionError> {
    let approval_scope = json!({
        "contractKey": BROWSER_ACTION_CONTRACT_KEY,
        "contract_key": BROWSER_ACTION_CONTRACT_KEY,
        "toolFamily": BROWSER_ACTION_TOOL_FAMILY,
        "tool_family": BROWSER_ACTION_TOOL_FAMILY,
        "riskClass": descriptor.risk_class,
        "risk_class": descriptor.risk_class,
        "browserActionKind": descriptor.action_kind,
        "browserSessionId": descriptor.browser_session_id,
        "tabId": descriptor.tab_id,
        "viewId": descriptor.view_id,
        "webContentsId": descriptor.web_contents_id,
        "snapshotId": descriptor.snapshot_id,
        "backendNodeId": descriptor.backend_node_id,
    });
    let metadata = HashMap::from([
        ("actionKind".to_string(), json!("browser_action")),
        ("toolFamily".to_string(), json!(BROWSER_ACTION_TOOL_FAMILY)),
        (
            "contractKey".to_string(),
            json!(BROWSER_ACTION_CONTRACT_KEY),
        ),
        ("riskClass".to_string(), json!(descriptor.risk_class)),
        ("approvalScope".to_string(), approval_scope),
        (
            "availableDecisions".to_string(),
            json!(["allow_once", "decline", "cancel"]),
        ),
        (
            "runtime_contract".to_string(),
            json!({
                "contract_key": BROWSER_ACTION_CONTRACT_KEY,
                "tool_family": BROWSER_ACTION_TOOL_FAMILY,
                "session_cache_supported": false,
            }),
        ),
    ]);
    wait_for_tool_approval(
        state,
        event_sender,
        request,
        thread_id,
        pending_input,
        &descriptor.reason,
        &metadata,
        None,
    )
    .await
    .map_err(RuntimeToolExecutionError::before_handler)
}

pub(super) fn materialize_tool_approval_action(
    queued: &ActionRequiredRequest,
    tool_name: &str,
    arguments: &Value,
    prompt: &str,
    approval: &tool_runtime::execution_approval::ExecutionApprovalProjection,
) -> ActionRequiredProjection {
    let mut projection = tool_confirmation_action(
        queued.id.clone(),
        tool_name.to_string(),
        arguments.clone(),
        Some(prompt.to_string()),
        queued.scope.clone(),
    );
    if let Some(data) = projection.data.as_object_mut() {
        data.insert("actionType".to_string(), queued.action_type.clone().into());
        data.insert("toolCallId".to_string(), queued.tool_id.clone().into());
        data.insert(
            "availableDecisions".to_string(),
            queued.available_decisions.clone().into(),
        );
        data.insert("createdAtMs".to_string(), queued.created_at_ms.into());
        data.insert("deadlineAtMs".to_string(), queued.deadline_at_ms.into());
        data.insert(
            "actionKind".to_string(),
            approval.action_kind.clone().into(),
        );
        data.insert(
            "action_kind".to_string(),
            approval.action_kind.clone().into(),
        );
        data.insert(
            "toolFamily".to_string(),
            approval.tool_family.clone().into(),
        );
        data.insert(
            "tool_family".to_string(),
            approval.tool_family.clone().into(),
        );
        data.insert(
            "runtime_contract".to_string(),
            approval.runtime_contract.clone(),
        );
        data.insert(
            "contractKey".to_string(),
            approval.contract_key.clone().into(),
        );
        data.insert(
            "contract_key".to_string(),
            approval.contract_key.clone().into(),
        );
        data.insert("approvalScope".to_string(), approval.approval_scope.clone());
        data.insert(
            "approval_scope".to_string(),
            approval.approval_scope.clone(),
        );
    }
    projection
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn execution_request(
        params: Value,
        working_directory: &str,
    ) -> (
        Value,
        tool_runtime::tool_executor::RuntimeToolExecutionContext,
    ) {
        let context = tool_runtime::tool_executor::RuntimeToolExecutionContext::new(
            tool_runtime::tool_executor::RuntimeToolExecutionContextInput {
                working_directory: PathBuf::from(working_directory),
                session_id: "approval-key-test-session".to_string(),
                cancel_token: None,
                workspace_sandbox: None,
            },
        );
        (params, context)
    }

    #[test]
    fn shell_and_apply_patch_retry_policies_match_codex_contract() {
        assert!(!sandbox_retry_allowed(
            tool_runtime::unified_exec::EXEC_COMMAND_TOOL_NAME,
            RuntimeToolApprovalPolicy::OnRequest,
        ));
        assert!(sandbox_retry_allowed(
            tool_runtime::unified_exec::EXEC_COMMAND_TOOL_NAME,
            RuntimeToolApprovalPolicy::UnlessTrusted,
        ));
        assert!(sandbox_retry_allowed(
            tool_runtime::apply_patch::APPLY_PATCH_TOOL_NAME,
            RuntimeToolApprovalPolicy::OnRequest,
        ));
        assert!(!sandbox_retry_allowed(
            tool_runtime::apply_patch::APPLY_PATCH_TOOL_NAME,
            RuntimeToolApprovalPolicy::Never,
        ));
    }

    #[test]
    fn shell_approval_key_is_scoped_to_command_cwd_sandbox_and_permissions() {
        let (params, context) = execution_request(
            json!({
                "cmd": "cargo test",
                "sandbox_permissions": "require_escalated",
                "additional_permissions": {"network": true}
            }),
            "/workspace/project",
        );
        let request = RuntimeToolExecutionRequest {
            tool_name: tool_runtime::unified_exec::EXEC_COMMAND_TOOL_NAME,
            params: &params,
            context: &context,
            turn_context: None,
        };
        let base = shell_approval_key(request, RuntimeToolSandboxPolicy::WorkspaceWrite)
            .expect("shell approval key");

        let (different_command, _) =
            execution_request(json!({"cmd": "cargo check"}), "/workspace/project");
        let different_command_request = RuntimeToolExecutionRequest {
            tool_name: tool_runtime::unified_exec::EXEC_COMMAND_TOOL_NAME,
            params: &different_command,
            context: &context,
            turn_context: None,
        };
        assert_ne!(
            base,
            shell_approval_key(
                different_command_request,
                RuntimeToolSandboxPolicy::WorkspaceWrite
            )
            .expect("different command key")
        );

        let (same_command_different_cwd, different_cwd_context) = execution_request(
            json!({
                "cmd": "cargo test",
                "sandbox_permissions": "require_escalated",
                "additional_permissions": {"network": true}
            }),
            "/workspace/other-project",
        );
        let different_cwd_request = RuntimeToolExecutionRequest {
            tool_name: tool_runtime::unified_exec::EXEC_COMMAND_TOOL_NAME,
            params: &same_command_different_cwd,
            context: &different_cwd_context,
            turn_context: None,
        };
        assert_ne!(
            base,
            shell_approval_key(
                different_cwd_request,
                RuntimeToolSandboxPolicy::WorkspaceWrite
            )
            .expect("different cwd key")
        );
        assert_ne!(
            base,
            shell_approval_key(request, RuntimeToolSandboxPolicy::DangerFullAccess)
                .expect("different sandbox key")
        );

        let (different_permissions, different_permissions_context) = execution_request(
            json!({
                "cmd": "cargo test",
                "sandbox_permissions": "require_escalated",
                "additional_permissions": {"network": false}
            }),
            "/workspace/project",
        );
        let different_permissions_request = RuntimeToolExecutionRequest {
            tool_name: tool_runtime::unified_exec::EXEC_COMMAND_TOOL_NAME,
            params: &different_permissions,
            context: &different_permissions_context,
            turn_context: None,
        };
        assert_ne!(
            base,
            shell_approval_key(
                different_permissions_request,
                RuntimeToolSandboxPolicy::WorkspaceWrite
            )
            .expect("different permissions key")
        );

        let (browser_params, browser_context) =
            execution_request(json!({"cmd": "cargo test"}), "/workspace/project");
        let browser_request = RuntimeToolExecutionRequest {
            tool_name: "browser_click",
            params: &browser_params,
            context: &browser_context,
            turn_context: None,
        };
        assert!(
            shell_approval_key(browser_request, RuntimeToolSandboxPolicy::WorkspaceWrite).is_none()
        );
    }

    #[test]
    fn shell_approval_contract_exposes_session_scope_only_for_shell() {
        let params = json!({"cmd": "cargo test"});
        let context = tool_runtime::tool_executor::RuntimeToolExecutionContext::new(
            tool_runtime::tool_executor::RuntimeToolExecutionContextInput {
                working_directory: PathBuf::from("/workspace/project"),
                session_id: "approval-contract-test-session".to_string(),
                cancel_token: None,
                workspace_sandbox: None,
            },
        );
        let request = RuntimeToolExecutionRequest {
            tool_name: tool_runtime::unified_exec::EXEC_COMMAND_TOOL_NAME,
            params: &params,
            context: &context,
            turn_context: None,
        };
        let mut decision = current_tool_execution_decision(request);
        decision.kind = ToolExecutionDecisionKind::RequiresApproval;
        annotate_shell_approval_contract(&mut decision, request);
        let projection = tool_runtime::execution_approval::execution_approval_projection(
            request.tool_name,
            &decision.metadata,
        );
        assert_eq!(
            projection.available_decisions,
            vec![
                "allow_once".to_string(),
                "allow_for_session".to_string(),
                "decline".to_string(),
                "cancel".to_string()
            ]
        );
        assert_eq!(
            projection.runtime_contract["session_cache_supported"],
            Value::Bool(true)
        );

        let mut browser_metadata = HashMap::from([(
            "runtime_contract".to_string(),
            json!({
                "contract_key": BROWSER_ACTION_CONTRACT_KEY,
                "tool_family": BROWSER_ACTION_TOOL_FAMILY,
                "session_cache_supported": false,
            }),
        )]);
        browser_metadata.insert(
            "availableDecisions".to_string(),
            json!(["allow_once", "allow_for_session", "decline", "cancel"]),
        );
        let browser_projection = tool_runtime::execution_approval::execution_approval_projection(
            "browser_click",
            &browser_metadata,
        );
        assert_eq!(
            browser_projection.available_decisions,
            vec!["allow_once", "decline", "cancel"]
        );
    }
}
