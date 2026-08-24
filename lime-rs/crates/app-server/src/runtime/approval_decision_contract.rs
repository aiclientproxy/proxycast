use super::{event_request_id, RuntimeCoreError, StoredSession};
use app_server_protocol::AgentSessionApprovalDecision;
use serde_json::Value;

enum ToolConfirmationDecisionAvailability {
    Declared(Vec<AgentSessionApprovalDecision>),
    OnceOnlyFallback,
    Missing,
}

pub(super) fn validate_tool_confirmation_decision(
    stored: &StoredSession,
    request_id: &str,
    decision: AgentSessionApprovalDecision,
) -> Result<(), RuntimeCoreError> {
    match tool_confirmation_decision_availability(stored, request_id) {
        ToolConfirmationDecisionAvailability::Declared(available_decisions) => {
            if decision == AgentSessionApprovalDecision::AllowForSession
                && !tool_confirmation_supports_session_approval(stored, request_id)
            {
                return Err(unavailable_decision_error(request_id, decision));
            }
            if available_decisions.contains(&decision) {
                return Ok(());
            }
            Err(unavailable_decision_error(request_id, decision))
        }
        ToolConfirmationDecisionAvailability::OnceOnlyFallback
        | ToolConfirmationDecisionAvailability::Missing => {
            if decision == AgentSessionApprovalDecision::AllowForSession {
                return Err(unavailable_decision_error(request_id, decision));
            }
            Ok(())
        }
    }
}

fn tool_confirmation_supports_session_approval(stored: &StoredSession, request_id: &str) -> bool {
    stored
        .events
        .iter()
        .rev()
        .find_map(|event| {
            if event_request_id(&event.payload).as_deref() != Some(request_id)
                || event.event_type != "action.required"
                || action_type(&event.payload) != Some("tool_confirmation")
            {
                return None;
            }
            Some(
                event
                    .payload
                    .get("runtime_contract")
                    .or_else(|| event.payload.pointer("/data/runtime_contract"))
                    .and_then(|contract| contract.get("session_cache_supported"))
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            )
        })
        .unwrap_or(false)
}

fn tool_confirmation_decision_availability(
    stored: &StoredSession,
    request_id: &str,
) -> ToolConfirmationDecisionAvailability {
    let request_id = request_id.trim();
    if request_id.is_empty() {
        return ToolConfirmationDecisionAvailability::Missing;
    }

    for event in stored.events.iter().rev() {
        if event_request_id(&event.payload).as_deref() != Some(request_id) {
            continue;
        }

        match event.event_type.as_str() {
            "action.required" if action_type(&event.payload) == Some("tool_confirmation") => {
                return available_decisions_from_action_required(&event.payload)
                    .map(ToolConfirmationDecisionAvailability::Declared)
                    .unwrap_or(ToolConfirmationDecisionAvailability::OnceOnlyFallback);
            }
            "action.resolved" | "action.canceled" | "action.cancelled" | "action.expired" => {
                return ToolConfirmationDecisionAvailability::Missing;
            }
            _ => {}
        }
    }

    ToolConfirmationDecisionAvailability::Missing
}

fn available_decisions_from_action_required(
    payload: &Value,
) -> Option<Vec<AgentSessionApprovalDecision>> {
    let data = payload.get("data").unwrap_or(payload);
    let values = data
        .get("availableDecisions")
        .or_else(|| data.get("available_decisions"))
        .or_else(|| payload.get("availableDecisions"))
        .or_else(|| payload.get("available_decisions"))?;
    let decisions = values
        .as_array()?
        .iter()
        .filter_map(Value::as_str)
        .filter_map(approval_decision_from_str)
        .collect::<Vec<_>>();
    Some(decisions)
}

fn approval_decision_from_str(value: &str) -> Option<AgentSessionApprovalDecision> {
    match value {
        "allow_once" => Some(AgentSessionApprovalDecision::AllowOnce),
        "allow_for_session" => Some(AgentSessionApprovalDecision::AllowForSession),
        "decline" => Some(AgentSessionApprovalDecision::Decline),
        "cancel" => Some(AgentSessionApprovalDecision::Cancel),
        _ => None,
    }
}

fn action_type(payload: &Value) -> Option<&str> {
    let data = payload.get("data").unwrap_or(payload);
    data.get("actionType")
        .or_else(|| data.get("action_type"))
        .or_else(|| payload.get("actionType"))
        .or_else(|| payload.get("action_type"))
        .and_then(Value::as_str)
}

fn unavailable_decision_error(
    request_id: &str,
    decision: AgentSessionApprovalDecision,
) -> RuntimeCoreError {
    RuntimeCoreError::Backend(format!(
        "approval decision '{}' is not available for tool_confirmation request '{}'",
        decision.as_str(),
        request_id
    ))
}
