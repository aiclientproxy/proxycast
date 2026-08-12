use crate::session_loop::{
    RuntimeSessionInputHandle, RuntimeSessionResponseKind, RuntimeSessionTaskFailure,
};
use agent_protocol::action_required::{ActionRequiredScope, TOOL_CONFIRMATION_ACTION_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;
use tokio::time::timeout;
use uuid::Uuid;

pub const ACTION_NOT_RESUMABLE_CODE: &str = "action_not_resumable";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PendingActionStatus {
    Pending,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingActionDescriptor {
    pub request_id: String,
    pub action_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_schema: Option<Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub available_decisions: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<ActionRequiredScope>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deadline_at_ms: Option<u64>,
    pub status: PendingActionStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PendingActionRestoreOutcome {
    Restored,
    AlreadyPresent,
    Expired,
    Terminal,
    Invalid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionTerminalStatus {
    Resolved,
    Canceled,
    Expired,
    ContinuationClosed,
    NotResumable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionRequiredError {
    NotFound(String),
    ScopeMissing(String),
    ScopeMismatch(String),
    NotResumable(String),
    AlreadyResolved(String),
    Terminal(String, ActionTerminalStatus),
    ContinuationClosed(String),
}

impl fmt::Display for ActionRequiredError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(request_id) => {
                write!(formatter, "action_not_found: {request_id}")
            }
            Self::ScopeMissing(request_id) => {
                write!(formatter, "action_scope_missing: {request_id}")
            }
            Self::ScopeMismatch(request_id) => {
                write!(formatter, "action_scope_mismatch: {request_id}")
            }
            Self::NotResumable(request_id) => {
                write!(formatter, "{ACTION_NOT_RESUMABLE_CODE}: {request_id}")
            }
            Self::AlreadyResolved(request_id) => {
                write!(formatter, "action_already_resolved: {request_id}")
            }
            Self::Terminal(request_id, status) => {
                write!(
                    formatter,
                    "action_terminal:{}: {request_id}",
                    status.as_str()
                )
            }
            Self::ContinuationClosed(request_id) => {
                write!(formatter, "action_continuation_closed: {request_id}")
            }
        }
    }
}

impl ActionRequiredError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::NotFound(_) => "action_not_found",
            Self::ScopeMissing(_) => "action_scope_missing",
            Self::ScopeMismatch(_) => "action_scope_mismatch",
            Self::NotResumable(_) => ACTION_NOT_RESUMABLE_CODE,
            Self::AlreadyResolved(_) => "action_already_resolved",
            Self::Terminal(_, _) => "action_terminal",
            Self::ContinuationClosed(_) => "action_continuation_closed",
        }
    }

    pub fn request_id(&self) -> &str {
        match self {
            Self::NotFound(request_id)
            | Self::ScopeMissing(request_id)
            | Self::ScopeMismatch(request_id)
            | Self::NotResumable(request_id)
            | Self::AlreadyResolved(request_id)
            | Self::Terminal(request_id, _)
            | Self::ContinuationClosed(request_id) => request_id,
        }
    }
}

impl ActionTerminalStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Resolved => "resolved",
            Self::Canceled => "canceled",
            Self::Expired => "expired",
            Self::ContinuationClosed => "continuation_closed",
            Self::NotResumable => "not_resumable",
        }
    }
}

impl std::error::Error for ActionRequiredError {}

struct PendingAction {
    descriptor: PendingActionDescriptor,
    restored: bool,
}

#[derive(Debug, Clone)]
struct TerminalAction {
    status: ActionTerminalStatus,
    scope: Option<ActionRequiredScope>,
}

#[derive(Default)]
struct ActionRequiredEntries {
    pending: HashMap<String, PendingAction>,
    terminal: HashMap<String, TerminalAction>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActionRequiredRequest {
    pub id: String,
    pub action_type: String,
    pub tool_id: Option<String>,
    pub message: String,
    pub requested_schema: Value,
    pub available_decisions: Vec<String>,
    pub scope: Option<ActionRequiredScope>,
    pub created_at_ms: Option<u64>,
    pub deadline_at_ms: Option<u64>,
}

/// 单个 Agent session runtime 的交互等待状态。
///
/// 对齐 `TurnState.pending_user_input`：pending sender 归 session/turn
/// 实例所有，不使用进程级 singleton。队列仅用于 Agent reply adapter 退场期间投影事件。
#[derive(Default)]
pub struct ActionRequiredState {
    entries: Mutex<ActionRequiredEntries>,
}

impl ActionRequiredState {
    pub async fn request_action_and_wait_with_notification<F>(
        &self,
        response_handle: RuntimeSessionInputHandle,
        response_kind: RuntimeSessionResponseKind,
        action_type: impl Into<String>,
        tool_id: Option<String>,
        available_decisions: Vec<String>,
        scope: Option<ActionRequiredScope>,
        message: String,
        requested_schema: Value,
        timeout_duration: Duration,
        notify: F,
    ) -> anyhow::Result<Value>
    where
        F: FnOnce(&ActionRequiredRequest),
    {
        let id = Uuid::now_v7().to_string();
        let pending_response = response_handle
            .register_response(response_kind, id.clone())
            .await
            .map_err(action_wait_error)?;
        let created_at_ms = now_millis();
        let deadline_at_ms = created_at_ms
            .map(|created_at_ms| created_at_ms.saturating_add(duration_millis(timeout_duration)));
        let descriptor = PendingActionDescriptor {
            request_id: id.clone(),
            tool_id,
            action_type: action_type.into(),
            message: Some(message.clone()),
            requested_schema: Some(requested_schema.clone()),
            available_decisions,
            scope: scope.clone(),
            created_at_ms,
            deadline_at_ms,
            status: PendingActionStatus::Pending,
        };

        self.entries.lock().await.pending.insert(
            id.clone(),
            PendingAction {
                descriptor: descriptor.clone(),
                restored: false,
            },
        );
        let action = ActionRequiredRequest {
            id: id.clone(),
            action_type: descriptor.action_type.clone(),
            tool_id: descriptor.tool_id.clone(),
            message,
            requested_schema,
            available_decisions: descriptor.available_decisions.clone(),
            scope,
            created_at_ms: descriptor.created_at_ms,
            deadline_at_ms: descriptor.deadline_at_ms,
        };
        notify(&action);

        let (result, terminal_status) =
            match timeout(timeout_duration, pending_response.wait()).await {
                Ok(Ok(user_data)) => return Ok(user_data),
                Ok(Err(error)) => (
                    Err(action_wait_error(error)),
                    ActionTerminalStatus::Canceled,
                ),
                Err(_) => (
                    Err(anyhow::anyhow!("Timeout waiting for user response")),
                    ActionTerminalStatus::Expired,
                ),
            };
        let mut entries = self.entries.lock().await;
        let scope = entries
            .pending
            .remove(&id)
            .and_then(|action| action.descriptor.scope);
        entries
            .terminal
            .entry(id.clone())
            .or_insert(TerminalAction {
                status: terminal_status,
                scope,
            });
        drop(entries);
        result
    }

    pub async fn resolve_action(
        &self,
        request_id: &str,
        scope: Option<&ActionRequiredScope>,
    ) -> Result<(), ActionRequiredError> {
        let mut entries = self.entries.lock().await;
        expire_action(&mut entries, request_id);
        if let Some(terminal) = entries.terminal.get(request_id) {
            return Err(terminal_error(request_id, terminal.status));
        }
        let action = entries
            .pending
            .remove(request_id)
            .ok_or_else(|| ActionRequiredError::NotFound(request_id.to_string()))?;
        if !scope_matches(action.descriptor.scope.as_ref(), scope) {
            entries.pending.insert(request_id.to_string(), action);
            return Err(ActionRequiredError::ScopeMismatch(request_id.to_string()));
        }
        let action_scope = action.descriptor.scope.clone();
        if action.restored {
            entries.terminal.insert(
                request_id.to_string(),
                TerminalAction {
                    status: ActionTerminalStatus::NotResumable,
                    scope: action_scope,
                },
            );
            return Err(ActionRequiredError::NotResumable(request_id.to_string()));
        }
        entries.terminal.insert(
            request_id.to_string(),
            TerminalAction {
                status: ActionTerminalStatus::Resolved,
                scope: action_scope,
            },
        );
        Ok(())
    }

    pub async fn ensure_resumable(
        &self,
        request_id: &str,
        scope: Option<&ActionRequiredScope>,
    ) -> Result<(), ActionRequiredError> {
        let mut entries = self.entries.lock().await;
        expire_action(&mut entries, request_id);
        if let Some(terminal) = entries.terminal.get(request_id) {
            return Err(terminal_error(request_id, terminal.status));
        }
        let action = entries
            .pending
            .get(request_id)
            .ok_or_else(|| ActionRequiredError::NotFound(request_id.to_string()))?;
        if !scope_matches(action.descriptor.scope.as_ref(), scope) {
            return Err(ActionRequiredError::ScopeMismatch(request_id.to_string()));
        }
        if !action.restored {
            return Ok(());
        }
        let scope = action.descriptor.scope.clone();
        entries.pending.remove(request_id);
        entries.terminal.insert(
            request_id.to_string(),
            TerminalAction {
                status: ActionTerminalStatus::NotResumable,
                scope,
            },
        );
        Err(ActionRequiredError::NotResumable(request_id.to_string()))
    }

    pub async fn cancel_action(
        &self,
        request_id: &str,
        scope: Option<&ActionRequiredScope>,
    ) -> Result<(), ActionRequiredError> {
        let mut entries = self.entries.lock().await;
        expire_action(&mut entries, request_id);
        if let Some(terminal) = entries.terminal.get(request_id) {
            return Err(terminal_error(request_id, terminal.status));
        }
        let action = entries
            .pending
            .remove(request_id)
            .ok_or_else(|| ActionRequiredError::NotFound(request_id.to_string()))?;
        if !scope_matches(action.descriptor.scope.as_ref(), scope) {
            entries.pending.insert(request_id.to_string(), action);
            return Err(ActionRequiredError::ScopeMismatch(request_id.to_string()));
        }
        entries.terminal.insert(
            request_id.to_string(),
            TerminalAction {
                status: ActionTerminalStatus::Canceled,
                scope: action.descriptor.scope,
            },
        );
        Ok(())
    }

    pub async fn restore_pending_action(
        &self,
        descriptor: PendingActionDescriptor,
    ) -> PendingActionRestoreOutcome {
        if descriptor.request_id.trim().is_empty()
            || descriptor.action_type.trim().is_empty()
            || !scope_is_complete(descriptor.scope.as_ref())
            || descriptor.deadline_at_ms.is_none()
            || (descriptor.action_type == TOOL_CONFIRMATION_ACTION_TYPE
                && (descriptor.tool_id.as_deref().is_none_or(str::is_empty)
                    || descriptor.available_decisions.is_empty()))
        {
            return PendingActionRestoreOutcome::Invalid;
        }
        let mut entries = self.entries.lock().await;
        if descriptor
            .deadline_at_ms
            .zip(now_millis())
            .is_some_and(|(deadline, now)| deadline <= now)
        {
            entries.terminal.insert(
                descriptor.request_id.clone(),
                TerminalAction {
                    status: ActionTerminalStatus::Expired,
                    scope: descriptor.scope,
                },
            );
            return PendingActionRestoreOutcome::Expired;
        }
        if entries.terminal.contains_key(&descriptor.request_id) {
            return PendingActionRestoreOutcome::Terminal;
        }
        if entries.pending.contains_key(&descriptor.request_id) {
            return PendingActionRestoreOutcome::AlreadyPresent;
        }
        entries.pending.insert(
            descriptor.request_id.clone(),
            PendingAction {
                descriptor,
                restored: true,
            },
        );
        PendingActionRestoreOutcome::Restored
    }

    pub async fn restore_pending_actions(
        &self,
        descriptors: impl IntoIterator<Item = PendingActionDescriptor>,
    ) -> Vec<PendingActionRestoreOutcome> {
        let mut outcomes = Vec::new();
        for descriptor in descriptors {
            outcomes.push(self.restore_pending_action(descriptor).await);
        }
        outcomes
    }

    pub async fn pending_action_descriptors(&self) -> Vec<PendingActionDescriptor> {
        let mut entries = self.entries.lock().await;
        expire_pending_actions(&mut entries);
        let mut descriptors = entries
            .pending
            .values()
            .map(|action| action.descriptor.clone())
            .collect::<Vec<_>>();
        descriptors.sort_by(|left, right| left.request_id.cmp(&right.request_id));
        descriptors
    }

    pub async fn contains_action(&self, request_id: &str) -> bool {
        let mut entries = self.entries.lock().await;
        expire_action(&mut entries, request_id);
        entries.pending.contains_key(request_id)
    }

    pub async fn terminal_status(&self, request_id: &str) -> Option<ActionTerminalStatus> {
        let mut entries = self.entries.lock().await;
        expire_action(&mut entries, request_id);
        entries.terminal.get(request_id).map(|entry| entry.status)
    }

    pub async fn clear_for_scope(&self, scope: &ActionRequiredScope) {
        let mut entries = self.entries.lock().await;
        entries
            .pending
            .retain(|_, action| !scope_matches(Some(scope), action.descriptor.scope.as_ref()));
        entries
            .terminal
            .retain(|_, action| !scope_matches(Some(scope), action.scope.as_ref()));
        drop(entries);
    }

    pub async fn cancel_for_scope(&self, scope: &ActionRequiredScope) {
        let mut entries = self.entries.lock().await;
        let request_ids = entries
            .pending
            .iter()
            .filter(|(_, action)| scope_matches(Some(scope), action.descriptor.scope.as_ref()))
            .map(|(request_id, _)| request_id.clone())
            .collect::<Vec<_>>();
        for request_id in request_ids {
            let action_scope = entries
                .pending
                .remove(&request_id)
                .and_then(|action| action.descriptor.scope);
            entries.terminal.insert(
                request_id,
                TerminalAction {
                    status: ActionTerminalStatus::Canceled,
                    scope: action_scope,
                },
            );
        }
        drop(entries);
    }
}

fn now_millis() -> Option<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_millis()).ok())
}

fn duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn action_wait_error(error: RuntimeSessionTaskFailure) -> anyhow::Error {
    anyhow::anyhow!(error.message)
}

fn expire_action(entries: &mut ActionRequiredEntries, request_id: &str) {
    let expired = entries
        .pending
        .get(request_id)
        .and_then(|action| action.descriptor.deadline_at_ms)
        .zip(now_millis())
        .is_some_and(|(deadline, now)| deadline <= now);
    if !expired {
        return;
    }
    let scope = entries
        .pending
        .remove(request_id)
        .and_then(|action| action.descriptor.scope);
    entries
        .terminal
        .entry(request_id.to_string())
        .or_insert(TerminalAction {
            status: ActionTerminalStatus::Expired,
            scope,
        });
}

fn expire_pending_actions(entries: &mut ActionRequiredEntries) {
    let request_ids = entries.pending.keys().cloned().collect::<Vec<_>>();
    for request_id in request_ids {
        expire_action(entries, &request_id);
    }
}

fn terminal_error(request_id: &str, status: ActionTerminalStatus) -> ActionRequiredError {
    match status {
        ActionTerminalStatus::Resolved => {
            ActionRequiredError::AlreadyResolved(request_id.to_string())
        }
        ActionTerminalStatus::ContinuationClosed => {
            ActionRequiredError::ContinuationClosed(request_id.to_string())
        }
        status => ActionRequiredError::Terminal(request_id.to_string(), status),
    }
}

fn scope_is_complete(scope: Option<&ActionRequiredScope>) -> bool {
    let Some(scope) = scope else {
        return false;
    };
    [&scope.session_id, &scope.thread_id, &scope.turn_id]
        .into_iter()
        .all(|field| {
            field
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty())
        })
}

fn scope_matches(
    expected: Option<&ActionRequiredScope>,
    actual: Option<&ActionRequiredScope>,
) -> bool {
    let (Some(expected), Some(actual)) = (expected, actual) else {
        return expected.is_none() && actual.is_none();
    };

    expected.session_id == actual.session_id
        && expected.thread_id == actual.thread_id
        && expected.turn_id == actual.turn_id
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session_loop::{
        RuntimeSessionClosureTask, RuntimeSessionHandle, RuntimeSessionRegistry,
        RuntimeSessionSubmission,
    };
    use std::sync::{Arc, Mutex as StdMutex};
    use tokio::sync::oneshot;

    fn scope(session_id: &str) -> ActionRequiredScope {
        ActionRequiredScope {
            session_id: Some(session_id.to_string()),
            thread_id: Some(format!("thread-{session_id}")),
            turn_id: Some(format!("turn-{session_id}")),
        }
    }

    async fn start_live_request(
        state: Arc<ActionRequiredState>,
        session_id: &'static str,
        response_kind: RuntimeSessionResponseKind,
        action_type: &'static str,
        timeout_duration: Duration,
    ) -> (
        RuntimeSessionRegistry,
        RuntimeSessionHandle,
        RuntimeSessionSubmission,
        String,
        oneshot::Receiver<anyhow::Result<Value>>,
    ) {
        let registry = RuntimeSessionRegistry::default();
        let thread_id = format!("thread-{session_id}");
        let session = registry
            .get_or_create(session_id, &thread_id)
            .await
            .expect("bind action-required session actor");
        let (id_tx, id_rx) = oneshot::channel();
        let id_tx = Arc::new(StdMutex::new(Some(id_tx)));
        let (result_tx, result_rx) = oneshot::channel();
        let result_tx = Arc::new(StdMutex::new(Some(result_tx)));
        let task = RuntimeSessionClosureTask::new(
            format!("turn-{session_id}"),
            Vec::new(),
            move |context, _input, _cancel| {
                let state = Arc::clone(&state);
                let id_tx = Arc::clone(&id_tx);
                let result_tx = Arc::clone(&result_tx);
                Box::pin(async move {
                    let result = state
                        .request_action_and_wait_with_notification(
                            context.input_handle(),
                            response_kind,
                            action_type,
                            None,
                            Vec::new(),
                            Some(scope(session_id)),
                            "Continue?".to_string(),
                            serde_json::json!({ "type": "string" }),
                            timeout_duration,
                            move |request| {
                                if let Some(id_tx) = id_tx.lock().expect("id sender lock").take() {
                                    let _ = id_tx.send(request.id.clone());
                                }
                            },
                        )
                        .await;
                    if let Some(result_tx) = result_tx.lock().expect("result sender lock").take() {
                        let _ = result_tx.send(result);
                    }
                    Ok(())
                })
            },
        );
        let submission = session
            .submit(Arc::new(task), false)
            .await
            .expect("live request task");
        let request_id = tokio::time::timeout(Duration::from_secs(1), id_rx)
            .await
            .expect("request id timeout")
            .expect("request id sender");
        (registry, session, submission, request_id, result_rx)
    }

    #[tokio::test]
    async fn state_is_scoped_and_routes_response_to_pending_turn() {
        let state = Arc::new(ActionRequiredState::default());
        let (registry, session, submission, request_id, result_rx) = start_live_request(
            Arc::clone(&state),
            "session-1",
            RuntimeSessionResponseKind::AskUser,
            "ask_user",
            Duration::from_secs(1),
        )
        .await;
        state
            .resolve_action(&request_id, Some(&scope("session-1")))
            .await
            .expect("response should be accepted");
        session
            .answer_user_input(
                Some("turn-session-1"),
                &request_id,
                serde_json::json!({ "answer": "yes" }),
            )
            .await
            .expect("typed response");

        assert_eq!(
            result_rx.await.expect("response sender").expect("response"),
            serde_json::json!({ "answer": "yes" })
        );
        assert_eq!(
            submission.completion.await.expect("task completion"),
            Ok(crate::session_loop::RuntimeSessionTaskOutcome::Completed)
        );
        registry.shutdown("session-1").await.expect("shutdown");
    }

    fn restored_descriptor(request_id: &str) -> PendingActionDescriptor {
        PendingActionDescriptor {
            request_id: request_id.to_string(),
            action_type: "tool_confirmation".to_string(),
            tool_id: Some("tool-1".to_string()),
            message: Some("Allow?".to_string()),
            requested_schema: Some(serde_json::json!({ "type": "boolean" })),
            available_decisions: vec!["allow_once".to_string(), "decline".to_string()],
            scope: Some(scope("session-1")),
            created_at_ms: Some(1),
            deadline_at_ms: now_millis().map(|now| now.saturating_add(60_000)),
            status: PendingActionStatus::Pending,
        }
    }

    #[tokio::test]
    async fn restored_descriptor_is_serializable_and_fails_closed_without_continuation() {
        let descriptor = restored_descriptor("restored-1");
        let encoded = serde_json::to_value(&descriptor).expect("serialize descriptor");
        let decoded: PendingActionDescriptor =
            serde_json::from_value(encoded).expect("deserialize descriptor");
        assert_eq!(decoded, descriptor);

        let state = ActionRequiredState::default();
        assert_eq!(
            state.restore_pending_action(decoded.clone()).await,
            PendingActionRestoreOutcome::Restored
        );
        assert_eq!(state.pending_action_descriptors().await, vec![decoded]);

        let error = state
            .resolve_action("restored-1", Some(&scope("session-1")))
            .await
            .expect_err("restored action must not fake a live continuation");
        assert_eq!(
            error,
            ActionRequiredError::NotResumable("restored-1".to_string())
        );
        assert!(error.to_string().starts_with(ACTION_NOT_RESUMABLE_CODE));
        assert_eq!(
            state.terminal_status("restored-1").await,
            Some(ActionTerminalStatus::NotResumable)
        );
        assert_eq!(
            state
                .resolve_action("restored-1", Some(&scope("session-1")))
                .await,
            Err(ActionRequiredError::Terminal(
                "restored-1".to_string(),
                ActionTerminalStatus::NotResumable,
            ))
        );
    }

    #[tokio::test]
    async fn restored_descriptor_preserves_scope_and_rejects_expired_or_duplicate_state() {
        let state = ActionRequiredState::default();
        assert_eq!(
            state
                .restore_pending_action(restored_descriptor("restored-2"))
                .await,
            PendingActionRestoreOutcome::Restored
        );
        assert_eq!(
            state
                .restore_pending_action(restored_descriptor("restored-2"))
                .await,
            PendingActionRestoreOutcome::AlreadyPresent
        );
        let scope_error = state
            .ensure_resumable("restored-2", Some(&scope("session-2")))
            .await
            .expect_err("scope mismatch must fail before resumability");
        assert_eq!(
            scope_error,
            ActionRequiredError::ScopeMismatch("restored-2".to_string())
        );

        let mut expired = restored_descriptor("expired-1");
        expired.deadline_at_ms = Some(0);
        assert_eq!(
            state.restore_pending_action(expired).await,
            PendingActionRestoreOutcome::Expired
        );
        assert!(!state.contains_action("expired-1").await);
        assert_eq!(
            state.terminal_status("expired-1").await,
            Some(ActionTerminalStatus::Expired)
        );
    }

    #[tokio::test]
    async fn cancel_action_consumes_restored_pending_state() {
        let state = ActionRequiredState::default();
        state
            .restore_pending_action(restored_descriptor("cancel-restored"))
            .await;

        state
            .cancel_action("cancel-restored", Some(&scope("session-1")))
            .await
            .expect("cancel restored action");

        assert!(!state.contains_action("cancel-restored").await);
        assert_eq!(
            state.terminal_status("cancel-restored").await,
            Some(ActionTerminalStatus::Canceled)
        );
    }

    #[tokio::test]
    async fn cancel_action_does_not_own_or_fake_the_live_response() {
        let state = Arc::new(ActionRequiredState::default());
        let (registry, session, submission, request_id, mut result_rx) = start_live_request(
            Arc::clone(&state),
            "session-cancel-live",
            RuntimeSessionResponseKind::AskUser,
            "ask_user",
            Duration::from_secs(1),
        )
        .await;

        state
            .cancel_action(&request_id, Some(&scope("session-cancel-live")))
            .await
            .expect("cancel live action");

        assert!(
            tokio::time::timeout(Duration::from_millis(10), &mut result_rx)
                .await
                .is_err()
        );
        assert!(session.interrupt().await.expect("interrupt waiter"));
        result_rx
            .await
            .expect_err("interrupted waiter must close without a response");
        assert_eq!(
            state.terminal_status(&request_id).await,
            Some(ActionTerminalStatus::Canceled)
        );
        assert_eq!(
            submission.completion.await.expect("task completion"),
            Ok(crate::session_loop::RuntimeSessionTaskOutcome::Interrupted)
        );
        registry
            .shutdown("session-cancel-live")
            .await
            .expect("shutdown");
    }

    #[tokio::test]
    async fn live_tool_notification_uses_typed_descriptor_and_terminalizes_response() {
        let state = Arc::new(ActionRequiredState::default());
        let (registry, session, submission, request_id, result_rx) = start_live_request(
            Arc::clone(&state),
            "session-tool",
            RuntimeSessionResponseKind::Approval,
            TOOL_CONFIRMATION_ACTION_TYPE,
            Duration::from_secs(1),
        )
        .await;
        let descriptors = state.pending_action_descriptors().await;
        assert_eq!(descriptors.len(), 1);
        let descriptor = &descriptors[0];
        assert_eq!(descriptor.request_id, request_id);
        assert_eq!(descriptor.action_type, TOOL_CONFIRMATION_ACTION_TYPE);
        assert_eq!(descriptor.tool_id, None);
        assert!(descriptor.available_decisions.is_empty());
        assert_eq!(descriptor.scope, Some(scope("session-tool")));
        assert!(descriptor.deadline_at_ms.is_some());

        state
            .resolve_action(&request_id, Some(&scope("session-tool")))
            .await
            .expect("live continuation response");
        session
            .approve(
                Some("turn-session-tool"),
                &request_id,
                serde_json::json!({ "confirmed": true }),
            )
            .await
            .expect("typed approval");
        result_rx
            .await
            .expect("request result sender")
            .expect("request response");
        assert_eq!(
            state.terminal_status(&request_id).await,
            Some(ActionTerminalStatus::Resolved)
        );
        assert_eq!(
            submission.completion.await.expect("task completion"),
            Ok(crate::session_loop::RuntimeSessionTaskOutcome::Completed)
        );
        registry.shutdown("session-tool").await.expect("shutdown");
    }

    #[tokio::test]
    async fn scope_is_required_and_all_fields_are_checked() {
        let state = ActionRequiredState::default();
        let request_id = "scope-live".to_string();
        let expected = scope("session-scope");
        let mut descriptor = restored_descriptor(&request_id);
        descriptor.scope = Some(expected.clone());
        state.entries.lock().await.pending.insert(
            request_id.clone(),
            PendingAction {
                descriptor,
                restored: false,
            },
        );
        let wrong_scopes = [
            None,
            Some(ActionRequiredScope {
                session_id: Some("wrong-session".to_string()),
                ..expected.clone()
            }),
            Some(ActionRequiredScope {
                thread_id: Some("wrong-thread".to_string()),
                ..expected.clone()
            }),
            Some(ActionRequiredScope {
                turn_id: Some("wrong-turn".to_string()),
                ..expected.clone()
            }),
        ];
        for actual in wrong_scopes {
            assert_eq!(
                state.ensure_resumable(&request_id, actual.as_ref()).await,
                Err(ActionRequiredError::ScopeMismatch(request_id.clone()))
            );
        }
        state
            .resolve_action(&request_id, Some(&expected))
            .await
            .expect("matching scope");
    }

    #[tokio::test]
    async fn timeout_and_task_abort_become_terminal_without_fake_success() {
        let timed_out = Arc::new(ActionRequiredState::default());
        let (timeout_registry, _session, timeout_submission, timeout_id, timeout_result) =
            start_live_request(
                Arc::clone(&timed_out),
                "session-timeout",
                RuntimeSessionResponseKind::AskUser,
                "ask_user",
                Duration::from_millis(1),
            )
            .await;
        assert!(timeout_result
            .await
            .expect("timeout result sender")
            .is_err());
        assert_eq!(
            timed_out.terminal_status(&timeout_id).await,
            Some(ActionTerminalStatus::Expired)
        );
        assert_eq!(
            timeout_submission
                .completion
                .await
                .expect("timeout task completion"),
            Ok(crate::session_loop::RuntimeSessionTaskOutcome::Completed)
        );
        timeout_registry
            .shutdown("session-timeout")
            .await
            .expect("timeout shutdown");

        let aborted = Arc::new(ActionRequiredState::default());
        let (registry, session, submission, request_id, result_rx) = start_live_request(
            Arc::clone(&aborted),
            "session-aborted",
            RuntimeSessionResponseKind::AskUser,
            "ask_user",
            Duration::from_secs(1),
        )
        .await;
        assert!(session.interrupt().await.expect("interrupt live task"));
        aborted.cancel_for_scope(&scope("session-aborted")).await;
        result_rx
            .await
            .expect_err("aborted task must close the response waiter");
        assert_eq!(
            submission.completion.await.expect("aborted completion"),
            Ok(crate::session_loop::RuntimeSessionTaskOutcome::Interrupted)
        );
        assert_eq!(
            aborted.terminal_status(&request_id).await,
            Some(ActionTerminalStatus::Canceled)
        );
        registry
            .shutdown("session-aborted")
            .await
            .expect("aborted shutdown");
    }
}
