use super::request_context::{
    host_approval_policy, host_metadata_value, host_sandbox_policy, request_workspace_scope,
    RuntimeSessionScope,
};
use crate::runtime::approval_cache::{
    approval_scope_payload_from_parts, metadata_has_session_approval_cache_hit,
    session_approval_cache_hit_payload,
};
use crate::runtime::{ExecutionRequest, RuntimeEvent};
use app_server_protocol::RuntimeRequest;
use serde_json::{json, Value};

const APPROVAL_ON_REQUEST: &str = "on-request";
const SANDBOX_WORKSPACE_WRITE: &str = "workspace-write";
const BROWSER_CONTROL_CONTRACT: &str = "browser_control";

pub(super) enum PermissionPreflightOutcome {
    Required(RuntimeEvent),
    Cached(RuntimeEvent),
}

pub(super) fn browser_control_permission_event(
    request: &ExecutionRequest,
    host_request: Option<&RuntimeRequest>,
    scope: &RuntimeSessionScope,
) -> Option<PermissionPreflightOutcome> {
    let host_request = host_request?;
    let approval_policy = host_approval_policy(host_request)?;
    let sandbox_policy = host_sandbox_policy(host_request)?;
    if approval_policy != APPROVAL_ON_REQUEST || sandbox_policy != SANDBOX_WORKSPACE_WRITE {
        return None;
    }

    let host_metadata = host_metadata_value(host_request);
    let metadata = host_metadata.as_ref()?;
    let contract = metadata.pointer("/harness/browser_assist/runtime_contract")?;
    if contract.get("contract_key").and_then(Value::as_str) != Some(BROWSER_CONTROL_CONTRACT) {
        return None;
    }

    let request_id = format!("permission-{}", scope.turn_id);
    let permission_state = json!({
        "status": "requires_confirmation",
        "confirmation_status": "requested",
        "confirmation_request_id": request_id,
        "confirmation_source": "runtime_preflight",
        "ask_profile_keys": ["browser_control_profile"],
        "required_profile_keys": ["browser_control_profile"],
        "approval_policy": approval_policy,
        "sandbox_policy": sandbox_policy,
    });
    let scope_payload = json!({
        "sessionId": scope.session_id,
        "threadId": scope.thread_id,
        "turnId": scope.turn_id,
        "workspaceId": scope.workspace_id,
    });
    let workspace_scope = request_workspace_scope(request, Some(host_request));
    let approval_scope = approval_scope_payload_from_parts(
        BROWSER_CONTROL_CONTRACT,
        scope.workspace_id.as_deref(),
        workspace_scope
            .working_dir
            .as_ref()
            .and_then(|path| path.to_str()),
        workspace_scope
            .project_root
            .as_ref()
            .and_then(|path| path.to_str()),
        Some(metadata),
    );

    if metadata_has_session_approval_cache_hit(host_metadata.as_ref()) {
        let cache = session_approval_cache_hit_payload(host_metadata.as_ref());
        return Some(PermissionPreflightOutcome::Cached(RuntimeEvent::new(
            "action.resolved",
            json!({
                "backend": "runtime_core",
                "source": "approval_session_cache",
                "requestId": request_id,
                "actionId": request_id,
                "actionType": "tool_confirmation",
                "actionKind": "permission_preflight",
                "toolName": "browser_control",
                "approvalScope": approval_scope,
                "confirmed": true,
                "decision": "allow_for_session",
                "decisionScope": "session",
                "scope": scope_payload,
                "cache": cache,
            }),
        )));
    }

    Some(PermissionPreflightOutcome::Required(RuntimeEvent::new(
        "action.required",
        json!({
            "backend": "runtime_core",
            "source": "runtime_preflight",
            "requestId": request_id,
            "actionId": request_id,
            "actionType": "tool_confirmation",
            "actionKind": "permission_preflight",
            "availableDecisions": ["allow_once", "allow_for_session", "decline", "cancel"],
            "toolName": "browser_control",
            "prompt": "需要确认浏览器控制权限",
            "message": "需要确认浏览器控制权限",
            "permission_state": permission_state,
            "data": {
                "toolName": "browser_control",
                "prompt": "需要确认浏览器控制权限",
                "availableDecisions": ["allow_once", "allow_for_session", "decline", "cancel"],
                "permission_state": permission_state,
                "runtime_contract": contract,
                "approvalScope": approval_scope.clone(),
            },
            "scope": scope_payload,
            "runtime_contract": contract,
            "approvalScope": approval_scope,
            "approval_policy": approval_policy,
            "sandbox_policy": sandbox_policy,
        }),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{ExecutionRequest, RuntimeHostContext};
    use app_server_protocol::{AgentSession, AgentSessionStatus, AgentTurn, AgentTurnStatus};

    #[test]
    fn browser_control_preflight_cache_hint_returns_cached_outcome() {
        let contract = json!({
            "contract_key": "browser_control",
            "routing_slot": "browser_reasoning_model"
        });
        let metadata = json!({
            "harness": {
                "browser_assist": {
                    "runtime_contract": contract
                },
                "approval_session_cache": {
                    "decision": "allow_for_session",
                    "decisionScope": "session",
                    "sourceRequestId": "permission-turn-1"
                }
            }
        });
        let host_request = RuntimeRequest {
            approval_policy: Some("on-request".to_string()),
            sandbox_policy: Some("workspace-write".to_string()),
            metadata: Some(metadata.clone()),
            ..RuntimeRequest::default()
        };
        let request = ExecutionRequest {
            host: RuntimeHostContext::default(),
            session: AgentSession {
                session_id: "session-cache".to_string(),
                thread_id: "thread-cache".to_string(),
                app_id: "agent-chat".to_string(),
                workspace_id: Some("workspace-cache".to_string()),
                business_object_ref: None,
                status: AgentSessionStatus::Running,
                created_at: "2026-07-09T00:00:00.000Z".to_string(),
                updated_at: "2026-07-09T00:00:00.000Z".to_string(),
            },
            turn: AgentTurn {
                turn_id: "turn-2".to_string(),
                session_id: "session-cache".to_string(),
                thread_id: "thread-cache".to_string(),
                status: AgentTurnStatus::Accepted,
                started_at: None,
                completed_at: None,
            },
            forked_from_thread_id: None,
            input: agent_runtime::reply_input::RuntimeReplyInput::text("打开浏览器"),
            runtime_options: None,
            expected_output: None,
            structured_output: None,
            output_schema: None,
            event_name: None,
            queued_turn_id: None,
            queue_if_busy: false,
            skip_pre_submit_resume: true,
            agent_control_gateway: None,
        };
        let scope = RuntimeSessionScope {
            session_id: "session-cache".to_string(),
            thread_id: "thread-cache".to_string(),
            turn_id: "turn-2".to_string(),
            workspace_id: Some("workspace-cache".to_string()),
        };

        let outcome = browser_control_permission_event(&request, Some(&host_request), &scope)
            .expect("preflight outcome");

        match outcome {
            PermissionPreflightOutcome::Cached(event) => {
                assert_eq!(event.event_type, "action.resolved");
                assert_eq!(
                    event.payload["decision"].as_str(),
                    Some("allow_for_session")
                );
                assert_eq!(event.payload["decisionScope"].as_str(), Some("session"));
                assert_eq!(
                    event.payload["source"].as_str(),
                    Some("approval_session_cache")
                );
            }
            PermissionPreflightOutcome::Required(_) => {
                panic!("cache hit must not request approval")
            }
        }
    }
}
