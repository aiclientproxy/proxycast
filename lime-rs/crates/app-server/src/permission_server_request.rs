use crate::approval_server_request::WaitServerRequestError;
use crate::{AppServer, PermissionRespondRequest};
use app_server_protocol::protocol::v2::{
    GrantedPermissionProfile, PermissionGrantScope, PermissionsRequestApprovalParams,
    PermissionsRequestApprovalResponse, RequestPermissionProfile,
    METHOD_ITEM_PERMISSIONS_REQUEST_APPROVAL,
};
use app_server_protocol::AgentEvent;
use serde_json::Value;
use std::path::Path;

impl AppServer {
    pub(crate) async fn handle_permission_server_request(&self, event: AgentEvent) -> bool {
        let request = match permission_server_request(&event) {
            Ok(Some(request)) => request,
            Ok(None) => return false,
            Err(error) => {
                tracing::warn!(event_id = %event.event_id, %error, "invalid permission server request event");
                return true;
            }
        };
        let response = match self
            .wait_server_request::<_, PermissionsRequestApprovalResponse>(
                METHOD_ITEM_PERMISSIONS_REQUEST_APPROVAL,
                &request.params.thread_id,
                request.params.clone(),
            )
            .await
        {
            Ok(response) => response,
            Err(WaitServerRequestError::Transition) => return true,
            Err(WaitServerRequestError::Failed(error)) => {
                tracing::warn!(%error, "permission server request failed closed");
                PermissionsRequestApprovalResponse {
                    permissions: GrantedPermissionProfile::default(),
                    scope: PermissionGrantScope::Turn,
                    strict_auto_review: None,
                }
            }
        };
        let runtime = self.processor.runtime();
        if let Err(error) = runtime
            .respond_permission(PermissionRespondRequest {
                session_id: request.session_id,
                thread_id: request.params.thread_id,
                turn_id: request.params.turn_id,
                request_id: request.request_id,
                response,
            })
            .await
        {
            tracing::warn!(event_id = %event.event_id, %error, "typed permission response rejected by runtime");
        }
        true
    }
}

struct PermissionServerRequest {
    request_id: String,
    session_id: String,
    params: PermissionsRequestApprovalParams,
}

fn permission_server_request(
    event: &AgentEvent,
) -> Result<Option<PermissionServerRequest>, String> {
    if event.event_type != "action.required"
        || payload_string(&event.payload, &["actionType", "action_type"]).as_deref()
            != Some("request_permissions")
    {
        return Ok(None);
    }
    let session_id = required_text(&event.session_id, "sessionId")?;
    let thread_id = required_optional_text(event.thread_id.as_deref(), "threadId")?;
    let turn_id = required_optional_text(event.turn_id.as_deref(), "turnId")?;
    let request_id = required_payload_text(
        &event.payload,
        &["requestId", "request_id", "actionId", "action_id"],
        "requestId",
    )?;
    let item_id = required_payload_text(&event.payload, &["toolCallId", "tool_call_id"], "itemId")?;
    let cwd = required_payload_text(&event.payload, &["cwd"], "cwd")?;
    if !Path::new(&cwd).is_absolute() {
        return Err("action.required request_permissions cwd must be absolute".to_string());
    }
    let environment_id = payload_value(&event.payload, &["environmentId", "environment_id"])
        .map(|value| {
            value
                .as_str()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .ok_or_else(|| {
                    "action.required request_permissions environmentId is invalid".to_string()
                })
        })
        .transpose()?;
    let permissions = payload_value(&event.payload, &["permissions"])
        .cloned()
        .ok_or_else(|| "action.required request_permissions has no permissions".to_string())
        .and_then(|permissions| {
            serde_json::from_value::<RequestPermissionProfile>(permissions)
                .map_err(|error| format!("invalid request_permissions profile: {error}"))
        })?;
    if permissions.network.is_none() && permissions.file_system.is_none() {
        return Err("action.required request_permissions profile is empty".to_string());
    }
    let started_at_ms = payload_value(
        &event.payload,
        &[
            "createdAtMs",
            "created_at_ms",
            "startedAtMs",
            "started_at_ms",
        ],
    )
    .and_then(Value::as_i64)
    .ok_or_else(|| "action.required request_permissions has no startedAtMs".to_string())?;
    Ok(Some(PermissionServerRequest {
        request_id,
        session_id,
        params: PermissionsRequestApprovalParams {
            thread_id,
            turn_id,
            item_id,
            environment_id,
            started_at_ms,
            cwd,
            reason: payload_string(&event.payload, &["reason"]),
            permissions,
        },
    }))
}

fn payload_value<'a>(payload: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().find_map(|key| {
        payload
            .get(key)
            .or_else(|| payload.get("data").and_then(|data| data.get(key)))
    })
}

fn payload_string(payload: &Value, keys: &[&str]) -> Option<String> {
    payload_value(payload, keys).and_then(|value| {
        value
            .as_str()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    })
}

fn required_payload_text(payload: &Value, keys: &[&str], field: &str) -> Result<String, String> {
    payload_string(payload, keys)
        .ok_or_else(|| format!("action.required request_permissions has no {field}"))
}

fn required_text(value: &str, field: &str) -> Result<String, String> {
    let value = value.trim();
    (!value.is_empty())
        .then(|| value.to_string())
        .ok_or_else(|| format!("action.required request_permissions has no {field}"))
}

fn required_optional_text(value: Option<&str>, field: &str) -> Result<String, String> {
    value
        .map(|value| required_text(value, field))
        .transpose()?
        .ok_or_else(|| format!("action.required request_permissions has no {field}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::{AgentSessionStatus, AgentTurnStatus};

    fn event() -> AgentEvent {
        AgentEvent {
            event_id: "event-permission-1".to_string(),
            session_id: "session-1".to_string(),
            thread_id: Some("thread-1".to_string()),
            turn_id: Some("turn-1".to_string()),
            sequence: 1,
            event_type: "action.required".to_string(),
            timestamp: "2026-07-30T00:00:00Z".to_string(),
            payload: serde_json::json!({
                "request_id": "permission-1",
                "action_type": "request_permissions",
                "data": {
                    "actionType": "request_permissions",
                    "toolCallId": "call-1",
                    "environmentId": "local",
                    "createdAtMs": 1_783_860_000_123_i64,
                    "cwd": "/tmp/workspace",
                    "reason": "Allow output",
                    "permissions": {
                        "fileSystem": { "read": null, "write": ["/tmp/workspace/output"] }
                    }
                },
                "sessionStatus": AgentSessionStatus::WaitingAction,
                "turnStatus": AgentTurnStatus::WaitingAction
            }),
        }
    }

    #[test]
    fn parses_typed_permission_event_and_rejects_relative_cwd() {
        let request = permission_server_request(&event())
            .expect("parse")
            .expect("permission request");
        assert_eq!(request.request_id, "permission-1");
        assert_eq!(request.params.item_id, "call-1");
        assert_eq!(request.params.environment_id.as_deref(), Some("local"));
        assert_eq!(request.params.started_at_ms, 1_783_860_000_123);

        let mut invalid = event();
        invalid.payload["data"]["cwd"] = Value::String("relative".to_string());
        assert!(permission_server_request(&invalid).is_err());
    }
}
