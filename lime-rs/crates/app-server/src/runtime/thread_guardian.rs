use super::status::agent_turn_is_active;
use super::{RuntimeCore, RuntimeCoreError, RuntimeEvent};
use agent_runtime::session_loop::RuntimeSessionInput;
use app_server_protocol::protocol::v2::{
    ThreadApproveGuardianDeniedActionParams, ThreadApproveGuardianDeniedActionResponse,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;

pub(super) const GUARDIAN_APPROVAL_EVENT_TYPE: &str = "guardian.denied_action.approved";
const APPROVAL_PREFIX: &str =
    "The user has manually approved a specific action that was previously `Rejected`.";

#[derive(Debug, Deserialize)]
struct GuardianAssessmentEvent {
    id: String,
    status: GuardianAssessmentStatus,
    action: GuardianAssessmentAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum GuardianAssessmentStatus {
    InProgress,
    Approved,
    Denied,
    TimedOut,
    Aborted,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum GuardianAssessmentAction {
    Command {
        source: GuardianCommandSource,
        command: String,
        cwd: String,
    },
    Execve {
        source: GuardianCommandSource,
        program: String,
        argv: Vec<String>,
        cwd: String,
    },
    ApplyPatch {
        cwd: String,
        files: Vec<String>,
    },
    NetworkAccess {
        target: String,
        host: String,
        protocol: NetworkApprovalProtocol,
        port: u16,
    },
    McpToolCall {
        server: String,
        tool_name: String,
        connector_id: Option<String>,
        connector_name: Option<String>,
        tool_title: Option<String>,
    },
    RequestPermissions {
        reason: Option<String>,
        permissions: Value,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum GuardianCommandSource {
    Shell,
    UnifiedExec,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum NetworkApprovalProtocol {
    Http,
    #[serde(alias = "https_connect", alias = "http-connect")]
    Https,
    Socks5Tcp,
    Socks5Udp,
}

impl RuntimeCore {
    pub async fn approve_guardian_denied_action(
        &self,
        params: ThreadApproveGuardianDeniedActionParams,
    ) -> Result<ThreadApproveGuardianDeniedActionResponse, RuntimeCoreError> {
        let thread_id = params.thread_id.trim();
        if thread_id.is_empty() {
            return Err(RuntimeCoreError::InvalidRequest(
                "thread/approveGuardianDeniedAction requires threadId".to_string(),
            ));
        }
        let event: GuardianAssessmentEvent =
            serde_json::from_value(params.event).map_err(|error| {
                RuntimeCoreError::InvalidRequest(format!("invalid Guardian denial event: {error}"))
            })?;
        event.action.validate()?;

        let (session_id, active_turn_id) = {
            let state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .values()
                .find(|stored| stored.session.thread_id == thread_id)
                .ok_or_else(|| {
                    RuntimeCoreError::InvalidRequest(format!("thread not found: {thread_id}"))
                })?;
            let active_turn_id = stored
                .turns
                .iter()
                .find(|turn| agent_turn_is_active(turn.status))
                .map(|turn| turn.turn_id.clone());
            (stored.session.session_id.clone(), active_turn_id)
        };

        if event.status != GuardianAssessmentStatus::Denied {
            return Ok(ThreadApproveGuardianDeniedActionResponse {});
        }

        let action = serde_json::to_value(&event.action).map_err(|error| {
            RuntimeCoreError::Backend(format!(
                "failed to serialize approved Guardian action: {error}"
            ))
        })?;
        let approved_action = json!({
            "action": action,
            "outcome": "allowed",
        });
        let approved_action = serde_json::to_string_pretty(&approved_action).map_err(|error| {
            RuntimeCoreError::Backend(format!(
                "failed to serialize approved Guardian action: {error}"
            ))
        })?;
        let text = format!(
            "{APPROVAL_PREFIX}\n\nTreat this as approval to perform that exact action in the same context in which it was originally requested.\nDo not assume this also authorizes similar operations with different payloads.\n\nApproved action:\n{approved_action}"
        );

        self.append_runtime_events(
            &session_id,
            thread_id,
            active_turn_id.as_deref(),
            vec![RuntimeEvent::new(
                GUARDIAN_APPROVAL_EVENT_TYPE,
                json!({
                    "reviewId": event.id,
                    "role": "developer",
                    "visibility": "provider_only",
                    "source": "thread/approveGuardianDeniedAction",
                    "text": text,
                }),
            )],
        )?;

        if let Some(active_turn_id) = active_turn_id {
            if let Some(session) = self.session_loops.get_existing(&session_id).await {
                let _ = session
                    .steer_for_turn_id(
                        Some(&active_turn_id),
                        vec![RuntimeSessionInput::Developer(text)],
                    )
                    .await;
            }
        }

        Ok(ThreadApproveGuardianDeniedActionResponse {})
    }
}

impl GuardianAssessmentAction {
    fn validate(&self) -> Result<(), RuntimeCoreError> {
        match self {
            Self::Command { cwd, .. } | Self::Execve { cwd, .. } => validate_absolute(cwd),
            Self::ApplyPatch { cwd, files } => {
                validate_absolute(cwd)?;
                for file in files {
                    validate_absolute(file)?;
                }
                Ok(())
            }
            Self::NetworkAccess { .. }
            | Self::McpToolCall { .. }
            | Self::RequestPermissions { .. } => Ok(()),
        }
    }
}

fn validate_absolute(path: &str) -> Result<(), RuntimeCoreError> {
    if Path::new(path).is_absolute() {
        return Ok(());
    }
    Err(RuntimeCoreError::InvalidRequest(format!(
        "invalid Guardian denial event: path must be absolute: {path}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::AgentSessionStartParams;
    use model_provider::current_client::{CurrentProviderContent, CurrentProviderRole};

    #[tokio::test]
    async fn denied_action_is_persisted_as_provider_only_developer_history() {
        let runtime = RuntimeCore::default();
        let session = runtime
            .start_session(AgentSessionStartParams {
                session_id: Some("session-guardian".to_string()),
                thread_id: Some("thread-guardian".to_string()),
                app_id: "test".to_string(),
                workspace_id: None,
                business_object_ref: None,
                locale: None,
            })
            .expect("guardian session")
            .session;

        runtime
            .approve_guardian_denied_action(ThreadApproveGuardianDeniedActionParams {
                thread_id: session.thread_id.clone(),
                event: json!({
                    "id": "guardian-review-1",
                    "status": "denied",
                    "action": {
                        "type": "command",
                        "source": "shell",
                        "command": "git status --short",
                        "cwd": "/workspace"
                    }
                }),
            })
            .await
            .expect("approve denied action");

        let history = {
            let state = runtime
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .get(&session.session_id)
                .expect("stored guardian session");
            super::super::provider_history::provider_history_excluding_current_turn_input(
                stored,
                None,
                "future-turn",
            )
            .expect("provider history")
        };

        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, CurrentProviderRole::Developer);
        assert!(matches!(
            history[0].content.as_slice(),
            [CurrentProviderContent::Text(text)]
                if text.starts_with(APPROVAL_PREFIX)
                    && text.contains("git status --short")
                    && text.contains("\"outcome\": \"allowed\"")
        ));
    }

    #[tokio::test]
    async fn non_denied_assessment_is_ignored() {
        let runtime = RuntimeCore::default();
        let session = runtime
            .start_session(AgentSessionStartParams {
                session_id: Some("session-guardian-approved".to_string()),
                thread_id: Some("thread-guardian-approved".to_string()),
                app_id: "test".to_string(),
                workspace_id: None,
                business_object_ref: None,
                locale: None,
            })
            .expect("guardian session")
            .session;

        runtime
            .approve_guardian_denied_action(ThreadApproveGuardianDeniedActionParams {
                thread_id: session.thread_id.clone(),
                event: json!({
                    "id": "guardian-review-2",
                    "status": "approved",
                    "action": {
                        "type": "apply_patch",
                        "cwd": "/workspace",
                        "files": ["/workspace/file.rs"]
                    }
                }),
            })
            .await
            .expect("ignore non-denied action");

        let state = runtime
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        let stored = state
            .sessions
            .get(&session.session_id)
            .expect("stored guardian session");
        assert!(stored
            .events
            .iter()
            .all(|event| event.event_type != GUARDIAN_APPROVAL_EVENT_TYPE));
    }
}
