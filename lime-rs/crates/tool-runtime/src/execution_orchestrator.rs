use crate::tool_executor::{
    RuntimeToolExecutionError, RuntimeToolExecutionIdentity, RuntimeToolExecutionResult,
    RuntimeToolPolicyErrorKind,
};
use app_server_protocol::protocol::v2::{
    AdditionalNetworkPermissions, FileSystemAccessMode, GrantedPermissionProfile,
};
use serde_json::{json, Value};
use std::future::Future;
use std::pin::Pin;
use tokio_util::sync::CancellationToken;

pub const TOOL_ATTEMPT_COUNT_METADATA_KEY: &str = "toolAttemptCount";
pub const TOOL_ATTEMPT_NUMBER_METADATA_KEY: &str = "toolAttemptNumber";
pub const TOOL_ESCALATED_METADATA_KEY: &str = "toolEscalated";
pub const TOOL_APPROVAL_SOURCE_METADATA_KEY: &str = "toolApprovalSource";
pub const TOOL_REQUESTED_SANDBOX_METADATA_KEY: &str = "requestedSandboxPolicy";
pub const TOOL_EFFECTIVE_SANDBOX_METADATA_KEY: &str = "effectiveSandboxPolicy";
pub const TOOL_FIRST_ATTEMPT_OUTCOME_METADATA_KEY: &str = "firstAttemptOutcome";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeToolApprovalPolicy {
    Never,
    OnRequest,
    UnlessTrusted,
    Granular,
    Unknown,
}

impl RuntimeToolApprovalPolicy {
    pub fn from_label(value: Option<&str>) -> Self {
        match value
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| value.to_ascii_lowercase().replace('_', "-"))
            .as_deref()
        {
            Some("never") => Self::Never,
            Some("on-request") => Self::OnRequest,
            Some("unless-trusted") => Self::UnlessTrusted,
            Some("granular") => Self::Granular,
            _ => Self::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeToolSandboxPolicy {
    None,
    ReadOnly,
    WorkspaceWrite,
    DangerFullAccess,
}

impl RuntimeToolSandboxPolicy {
    pub fn from_label(value: Option<&str>) -> Self {
        match value
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| value.to_ascii_lowercase().replace('_', "-"))
            .as_deref()
        {
            Some("read-only") => Self::ReadOnly,
            Some("danger-full-access") => Self::DangerFullAccess,
            Some("workspace-write") => Self::WorkspaceWrite,
            _ => Self::None,
        }
    }

    pub fn label(self) -> Option<&'static str> {
        match self {
            Self::None => None,
            Self::ReadOnly => Some("read-only"),
            Self::WorkspaceWrite => Some("workspace-write"),
            Self::DangerFullAccess => Some("danger-full-access"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeToolAttemptNumber {
    Initial,
    Escalated,
}

impl RuntimeToolAttemptNumber {
    pub fn ordinal(self) -> u8 {
        match self {
            Self::Initial => 1,
            Self::Escalated => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeToolApprovalKind {
    User,
    Guardian,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeToolApprovalSource {
    Config,
    User,
    Guardian,
    Reused,
}

impl RuntimeToolApprovalSource {
    pub fn label(self) -> &'static str {
        match self {
            Self::Config => "config",
            Self::User => "user",
            Self::Guardian => "guardian",
            Self::Reused => "reused",
        }
    }

    fn from_kind(kind: RuntimeToolApprovalKind) -> Self {
        match kind {
            RuntimeToolApprovalKind::User => Self::User,
            RuntimeToolApprovalKind::Guardian => Self::Guardian,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeToolInitialApproval {
    NotRequired,
    Required(RuntimeToolApprovalKind),
    Cached,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeToolApprovalPhase {
    Initial,
    Escalation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeToolDenialKind {
    Sandbox,
    ManagedNetwork,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeToolApprovalRequest {
    pub phase: RuntimeToolApprovalPhase,
    pub kind: RuntimeToolApprovalKind,
    pub denial_kind: Option<RuntimeToolDenialKind>,
    pub reason: Option<String>,
    pub network_host: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RuntimeToolExecutionAttempt {
    identity: RuntimeToolExecutionIdentity,
    number: RuntimeToolAttemptNumber,
    approval_policy: RuntimeToolApprovalPolicy,
    approval_source: RuntimeToolApprovalSource,
    requested_sandbox_policy: RuntimeToolSandboxPolicy,
    effective_sandbox_policy: RuntimeToolSandboxPolicy,
    granted_permissions: GrantedPermissionProfile,
    managed_network_host: Option<String>,
    cancel_token: Option<CancellationToken>,
}

impl RuntimeToolExecutionAttempt {
    pub fn identity(&self) -> &RuntimeToolExecutionIdentity {
        &self.identity
    }

    pub fn number(&self) -> RuntimeToolAttemptNumber {
        self.number
    }

    pub fn approval_policy(&self) -> RuntimeToolApprovalPolicy {
        self.approval_policy
    }

    pub fn approval_source(&self) -> RuntimeToolApprovalSource {
        self.approval_source
    }

    pub fn requested_sandbox_policy(&self) -> RuntimeToolSandboxPolicy {
        self.requested_sandbox_policy
    }

    pub fn effective_sandbox_policy(&self) -> RuntimeToolSandboxPolicy {
        self.effective_sandbox_policy
    }

    pub fn granted_permissions(&self) -> &GrantedPermissionProfile {
        &self.granted_permissions
    }

    pub fn managed_network_host(&self) -> Option<&str> {
        self.managed_network_host.as_deref()
    }

    pub fn cancel_token(&self) -> Option<&CancellationToken> {
        self.cancel_token.as_ref()
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel_token
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeToolOrchestrationInput {
    pub identity: RuntimeToolExecutionIdentity,
    pub approval_policy: RuntimeToolApprovalPolicy,
    pub initial_approval: RuntimeToolInitialApproval,
    pub initial_approval_reason: Option<String>,
    pub requested_sandbox_policy: RuntimeToolSandboxPolicy,
    pub effective_sandbox_policy: RuntimeToolSandboxPolicy,
    pub granted_permissions: GrantedPermissionProfile,
    pub managed_network_host: Option<String>,
    pub strict_guardian: bool,
    pub explicit_sandbox_escalation: bool,
    pub sandbox_denial_retry_allowed: bool,
    pub network_denial_retry_allowed: bool,
    pub cancel_token: Option<CancellationToken>,
}

pub type RuntimeToolAttemptFuture<'a> = Pin<
    Box<
        dyn Future<Output = Result<RuntimeToolExecutionResult, RuntimeToolExecutionError>>
            + Send
            + 'a,
    >,
>;

pub trait RuntimeToolAttemptRunner: Send + Sync {
    fn run<'a>(&'a self, attempt: RuntimeToolExecutionAttempt) -> RuntimeToolAttemptFuture<'a>;
}

pub type RuntimeToolApprovalFuture<'a> =
    Pin<Box<dyn Future<Output = Result<(), RuntimeToolExecutionError>> + Send + 'a>>;

pub trait RuntimeToolApprovalHandler: Send + Sync {
    fn approve<'a>(&'a self, request: RuntimeToolApprovalRequest) -> RuntimeToolApprovalFuture<'a>;
}

pub async fn orchestrate_runtime_tool_execution(
    input: RuntimeToolOrchestrationInput,
    approvals: &dyn RuntimeToolApprovalHandler,
    runner: &dyn RuntimeToolAttemptRunner,
) -> Result<RuntimeToolExecutionResult, RuntimeToolExecutionError> {
    if input.explicit_sandbox_escalation
        && matches!(
            input.approval_policy,
            RuntimeToolApprovalPolicy::Never | RuntimeToolApprovalPolicy::Unknown
        )
    {
        return Err(policy_denied(
            "sandbox escalation is forbidden by the active approval policy",
            "sandbox_escalation_forbidden",
        ));
    }

    let initial_approval_source = match input.initial_approval {
        RuntimeToolInitialApproval::NotRequired => RuntimeToolApprovalSource::Config,
        RuntimeToolInitialApproval::Cached => RuntimeToolApprovalSource::Reused,
        RuntimeToolInitialApproval::Required(kind) => {
            await_approval(
                approvals,
                RuntimeToolApprovalRequest {
                    phase: RuntimeToolApprovalPhase::Initial,
                    kind,
                    denial_kind: None,
                    reason: input.initial_approval_reason.clone(),
                    network_host: None,
                },
                input.cancel_token.as_ref(),
            )
            .await?;
            RuntimeToolApprovalSource::from_kind(kind)
        }
    };

    let initial_attempt = RuntimeToolExecutionAttempt {
        identity: input.identity.clone(),
        number: RuntimeToolAttemptNumber::Initial,
        approval_policy: input.approval_policy,
        approval_source: initial_approval_source,
        requested_sandbox_policy: input.requested_sandbox_policy,
        effective_sandbox_policy: input.effective_sandbox_policy,
        granted_permissions: input.granted_permissions.clone(),
        managed_network_host: input.managed_network_host.clone(),
        cancel_token: input.cancel_token.clone(),
    };
    let first_result = run_attempt(runner, initial_attempt.clone()).await;
    let first_error = match first_result {
        Ok(mut result) => {
            insert_attempt_metadata(&mut result, &initial_attempt, 1, None);
            return Ok(result);
        }
        Err(error) => error,
    };

    let Some(retry) = retry_plan(&input, &initial_attempt, &first_error) else {
        return Err(first_error);
    };
    let retry_approval_source = match retry.approval_kind {
        None => RuntimeToolApprovalSource::Reused,
        Some(kind) => {
            await_approval(
                approvals,
                RuntimeToolApprovalRequest {
                    phase: RuntimeToolApprovalPhase::Escalation,
                    kind,
                    denial_kind: Some(retry.denial_kind),
                    reason: Some(first_error.message().to_string()),
                    network_host: retry.network_host.clone(),
                },
                input.cancel_token.as_ref(),
            )
            .await?;
            RuntimeToolApprovalSource::from_kind(kind)
        }
    };
    let retry_attempt = RuntimeToolExecutionAttempt {
        identity: input.identity,
        number: RuntimeToolAttemptNumber::Escalated,
        approval_policy: input.approval_policy,
        approval_source: retry_approval_source,
        requested_sandbox_policy: input.requested_sandbox_policy,
        effective_sandbox_policy: retry.effective_sandbox_policy,
        granted_permissions: retry.granted_permissions,
        managed_network_host: input.managed_network_host,
        cancel_token: input.cancel_token,
    };
    let mut result = run_attempt(runner, retry_attempt.clone()).await?;
    insert_attempt_metadata(
        &mut result,
        &retry_attempt,
        2,
        first_error.policy_kind().map(policy_error_label),
    );
    Ok(result)
}

struct RuntimeToolRetryPlan {
    denial_kind: RuntimeToolDenialKind,
    approval_kind: Option<RuntimeToolApprovalKind>,
    effective_sandbox_policy: RuntimeToolSandboxPolicy,
    granted_permissions: GrantedPermissionProfile,
    network_host: Option<String>,
}

fn retry_plan(
    input: &RuntimeToolOrchestrationInput,
    initial_attempt: &RuntimeToolExecutionAttempt,
    error: &RuntimeToolExecutionError,
) -> Option<RuntimeToolRetryPlan> {
    match error.policy_kind()? {
        RuntimeToolPolicyErrorKind::SandboxDenied(_)
            if input.sandbox_denial_retry_allowed
                && initial_attempt.effective_sandbox_policy
                    != RuntimeToolSandboxPolicy::DangerFullAccess
                && !has_denied_file_system_permissions(&input.granted_permissions) =>
        {
            Some(RuntimeToolRetryPlan {
                denial_kind: RuntimeToolDenialKind::Sandbox,
                approval_kind: retry_approval_kind(input, initial_attempt, false),
                effective_sandbox_policy: RuntimeToolSandboxPolicy::DangerFullAccess,
                granted_permissions: input.granted_permissions.clone(),
                network_host: None,
            })
        }
        RuntimeToolPolicyErrorKind::ManagedNetworkDenied { host, .. }
            if input.network_denial_retry_allowed
                && !network_is_granted(&input.granted_permissions) =>
        {
            Some(RuntimeToolRetryPlan {
                denial_kind: RuntimeToolDenialKind::ManagedNetwork,
                approval_kind: retry_approval_kind(input, initial_attempt, true),
                effective_sandbox_policy: initial_attempt.effective_sandbox_policy,
                granted_permissions: with_network_grant(input.granted_permissions.clone()),
                network_host: host.clone(),
            })
        }
        _ => None,
    }
}

fn retry_approval_kind(
    input: &RuntimeToolOrchestrationInput,
    initial_attempt: &RuntimeToolExecutionAttempt,
    network_denial: bool,
) -> Option<RuntimeToolApprovalKind> {
    if input.strict_guardian {
        return Some(RuntimeToolApprovalKind::Guardian);
    }
    if !network_denial && initial_attempt.approval_source == RuntimeToolApprovalSource::User {
        return None;
    }
    Some(RuntimeToolApprovalKind::User)
}

async fn await_approval(
    approvals: &dyn RuntimeToolApprovalHandler,
    request: RuntimeToolApprovalRequest,
    cancel_token: Option<&CancellationToken>,
) -> Result<(), RuntimeToolExecutionError> {
    if cancel_token.is_some_and(CancellationToken::is_cancelled) {
        return Err(cancelled_error());
    }
    match cancel_token.cloned() {
        Some(cancel_token) => {
            tokio::select! {
                biased;
                _ = cancel_token.cancelled() => Err(cancelled_error()),
                result = approvals.approve(request) => result,
            }
        }
        None => approvals.approve(request).await,
    }
}

async fn run_attempt(
    runner: &dyn RuntimeToolAttemptRunner,
    attempt: RuntimeToolExecutionAttempt,
) -> Result<RuntimeToolExecutionResult, RuntimeToolExecutionError> {
    if attempt.is_cancelled() {
        return Err(cancelled_error());
    }
    match attempt.cancel_token().cloned() {
        Some(cancel_token) => {
            tokio::select! {
                biased;
                _ = cancel_token.cancelled() => Err(cancelled_error()),
                result = runner.run(attempt) => result,
            }
        }
        None => runner.run(attempt).await,
    }
}

fn insert_attempt_metadata(
    result: &mut RuntimeToolExecutionResult,
    attempt: &RuntimeToolExecutionAttempt,
    attempt_count: u8,
    first_attempt_outcome: Option<&'static str>,
) {
    result.metadata.insert(
        TOOL_ATTEMPT_COUNT_METADATA_KEY.to_string(),
        json!(attempt_count),
    );
    result.metadata.insert(
        TOOL_ATTEMPT_NUMBER_METADATA_KEY.to_string(),
        json!(attempt.number.ordinal()),
    );
    result.metadata.insert(
        TOOL_ESCALATED_METADATA_KEY.to_string(),
        Value::Bool(attempt.number == RuntimeToolAttemptNumber::Escalated),
    );
    result.metadata.insert(
        TOOL_APPROVAL_SOURCE_METADATA_KEY.to_string(),
        json!(attempt.approval_source.label()),
    );
    if let Some(policy) = attempt.requested_sandbox_policy.label() {
        result.metadata.insert(
            TOOL_REQUESTED_SANDBOX_METADATA_KEY.to_string(),
            json!(policy),
        );
    }
    if let Some(policy) = attempt.effective_sandbox_policy.label() {
        result.metadata.insert(
            TOOL_EFFECTIVE_SANDBOX_METADATA_KEY.to_string(),
            json!(policy),
        );
    }
    if let Some(outcome) = first_attempt_outcome {
        result.metadata.insert(
            TOOL_FIRST_ATTEMPT_OUTCOME_METADATA_KEY.to_string(),
            json!(outcome),
        );
    }
}

fn policy_error_label(kind: &RuntimeToolPolicyErrorKind) -> &'static str {
    match kind {
        RuntimeToolPolicyErrorKind::PermissionDenied(_) => "permission_denied",
        RuntimeToolPolicyErrorKind::SafetyCheckFailed(_) => "policy_denied",
        RuntimeToolPolicyErrorKind::SandboxDenied(_) => "sandbox_denied",
        RuntimeToolPolicyErrorKind::ManagedNetworkDenied { .. } => "managed_network_denied",
        RuntimeToolPolicyErrorKind::Timeout(_) => "timeout",
        RuntimeToolPolicyErrorKind::Canceled(_) => "canceled",
        RuntimeToolPolicyErrorKind::ExecutionFailed(_) => "handler_failure",
    }
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

fn network_is_granted(permissions: &GrantedPermissionProfile) -> bool {
    permissions
        .network
        .as_ref()
        .and_then(|network| network.enabled)
        .unwrap_or(false)
}

fn with_network_grant(mut permissions: GrantedPermissionProfile) -> GrantedPermissionProfile {
    permissions.network = Some(AdditionalNetworkPermissions {
        enabled: Some(true),
    });
    permissions
}

fn cancelled_error() -> RuntimeToolExecutionError {
    RuntimeToolExecutionError::new(
        "Tool execution cancelled",
        Some(RuntimeToolPolicyErrorKind::Canceled(
            "tool_execution_cancelled".to_string(),
        )),
    )
}

fn policy_denied(message: &str, reason_code: &str) -> RuntimeToolExecutionError {
    RuntimeToolExecutionError::new(
        message,
        Some(RuntimeToolPolicyErrorKind::PermissionDenied(
            reason_code.to_string(),
        )),
    )
    .before_handler()
}

#[cfg(test)]
pub(crate) fn test_runtime_tool_execution_attempt(
    identity: RuntimeToolExecutionIdentity,
    sandbox_policy: RuntimeToolSandboxPolicy,
    granted_permissions: GrantedPermissionProfile,
) -> RuntimeToolExecutionAttempt {
    RuntimeToolExecutionAttempt {
        identity,
        number: RuntimeToolAttemptNumber::Initial,
        approval_policy: RuntimeToolApprovalPolicy::UnlessTrusted,
        approval_source: RuntimeToolApprovalSource::Config,
        requested_sandbox_policy: sandbox_policy,
        effective_sandbox_policy: sandbox_policy,
        granted_permissions,
        managed_network_host: None,
        cancel_token: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_executor::{
        RuntimeToolExecutionContext, RuntimeToolExecutionContextInput, RuntimeToolExecutionRequest,
    };
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct RecordingApprovals {
        requests: Mutex<Vec<RuntimeToolApprovalRequest>>,
    }

    impl RuntimeToolApprovalHandler for RecordingApprovals {
        fn approve<'a>(
            &'a self,
            request: RuntimeToolApprovalRequest,
        ) -> RuntimeToolApprovalFuture<'a> {
            Box::pin(async move {
                self.requests.lock().unwrap().push(request);
                Ok(())
            })
        }
    }

    struct SequenceRunner {
        calls: Mutex<Vec<RuntimeToolExecutionAttempt>>,
        first_error: Option<RuntimeToolExecutionError>,
    }

    impl RuntimeToolAttemptRunner for SequenceRunner {
        fn run<'a>(&'a self, attempt: RuntimeToolExecutionAttempt) -> RuntimeToolAttemptFuture<'a> {
            Box::pin(async move {
                let mut calls = self.calls.lock().unwrap();
                calls.push(attempt);
                if calls.len() == 1 {
                    if let Some(error) = &self.first_error {
                        return Err(error.clone());
                    }
                }
                Ok(RuntimeToolExecutionResult::new(
                    true,
                    "ok".to_string(),
                    None,
                    HashMap::new(),
                ))
            })
        }
    }

    struct ApplyPatchAttemptRunner {
        working_directory: PathBuf,
        params: Value,
        calls: Mutex<Vec<RuntimeToolExecutionAttempt>>,
    }

    impl RuntimeToolAttemptRunner for ApplyPatchAttemptRunner {
        fn run<'a>(&'a self, attempt: RuntimeToolExecutionAttempt) -> RuntimeToolAttemptFuture<'a> {
            Box::pin(async move {
                self.calls.lock().unwrap().push(attempt.clone());
                let context = RuntimeToolExecutionContext::new(RuntimeToolExecutionContextInput {
                    working_directory: self.working_directory.clone(),
                    session_id: "apply-patch-orchestrator-test".to_string(),
                    cancel_token: None,
                    workspace_sandbox: None,
                })
                .with_tool_identity(attempt.identity().clone())
                .with_execution_attempt(attempt);
                crate::apply_patch::runtime_apply_patch_executor_handle()
                    .execute(RuntimeToolExecutionRequest {
                        tool_name: crate::apply_patch::APPLY_PATCH_TOOL_NAME,
                        params: &self.params,
                        context: &context,
                        turn_context: None,
                    })
                    .await
            })
        }
    }

    fn input() -> RuntimeToolOrchestrationInput {
        RuntimeToolOrchestrationInput {
            identity: RuntimeToolExecutionIdentity::new("call-1", "turn-1"),
            approval_policy: RuntimeToolApprovalPolicy::UnlessTrusted,
            initial_approval: RuntimeToolInitialApproval::NotRequired,
            initial_approval_reason: None,
            requested_sandbox_policy: RuntimeToolSandboxPolicy::WorkspaceWrite,
            effective_sandbox_policy: RuntimeToolSandboxPolicy::WorkspaceWrite,
            granted_permissions: GrantedPermissionProfile::default(),
            managed_network_host: None,
            strict_guardian: false,
            explicit_sandbox_escalation: false,
            sandbox_denial_retry_allowed: true,
            network_denial_retry_allowed: true,
            cancel_token: None,
        }
    }

    #[tokio::test]
    async fn sandbox_denial_approves_once_then_retries_with_same_identity() {
        let approvals = RecordingApprovals::default();
        let runner = SequenceRunner {
            calls: Mutex::new(Vec::new()),
            first_error: Some(RuntimeToolExecutionError::new(
                "sandbox denied write",
                Some(RuntimeToolPolicyErrorKind::SandboxDenied(
                    "sandbox_denied".to_string(),
                )),
            )),
        };

        let result = orchestrate_runtime_tool_execution(input(), &approvals, &runner)
            .await
            .expect("escalated result");

        let calls = runner.calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].identity(), calls[1].identity());
        assert_eq!(
            calls[1].effective_sandbox_policy(),
            RuntimeToolSandboxPolicy::DangerFullAccess
        );
        assert_eq!(approvals.requests.lock().unwrap().len(), 1);
        assert_eq!(result.metadata[TOOL_ATTEMPT_COUNT_METADATA_KEY], json!(2));
        assert_eq!(
            result.metadata[TOOL_FIRST_ATTEMPT_OUTCOME_METADATA_KEY],
            json!("sandbox_denied")
        );
    }

    #[tokio::test]
    async fn ordinary_initial_approval_is_reused_for_sandbox_retry() {
        let approvals = RecordingApprovals::default();
        let runner = SequenceRunner {
            calls: Mutex::new(Vec::new()),
            first_error: Some(RuntimeToolExecutionError::new(
                "sandbox denied write",
                Some(RuntimeToolPolicyErrorKind::SandboxDenied(
                    "sandbox_denied".to_string(),
                )),
            )),
        };
        let mut input = input();
        input.initial_approval =
            RuntimeToolInitialApproval::Required(RuntimeToolApprovalKind::User);

        orchestrate_runtime_tool_execution(input, &approvals, &runner)
            .await
            .expect("approved retry");

        assert_eq!(approvals.requests.lock().unwrap().len(), 1);
        assert_eq!(
            runner.calls.lock().unwrap()[1].approval_source(),
            RuntimeToolApprovalSource::Reused
        );
    }

    #[tokio::test]
    async fn cached_initial_approval_skips_approval_handler() {
        let approvals = RecordingApprovals::default();
        let runner = SequenceRunner {
            calls: Mutex::new(Vec::new()),
            first_error: None,
        };
        let mut input = input();
        input.initial_approval = RuntimeToolInitialApproval::Cached;

        let result = orchestrate_runtime_tool_execution(input, &approvals, &runner)
            .await
            .expect("cached approval should execute");

        assert_eq!(approvals.requests.lock().unwrap().len(), 0);
        assert_eq!(runner.calls.lock().unwrap().len(), 1);
        assert_eq!(
            runner.calls.lock().unwrap()[0].approval_source(),
            RuntimeToolApprovalSource::Reused
        );
        assert_eq!(
            result.metadata[TOOL_APPROVAL_SOURCE_METADATA_KEY],
            json!("reused")
        );
    }

    #[tokio::test]
    async fn strict_guardian_reviews_sandbox_retry_again() {
        let approvals = RecordingApprovals::default();
        let runner = SequenceRunner {
            calls: Mutex::new(Vec::new()),
            first_error: Some(RuntimeToolExecutionError::new(
                "sandbox denied write",
                Some(RuntimeToolPolicyErrorKind::SandboxDenied(
                    "sandbox_denied".to_string(),
                )),
            )),
        };
        let mut input = input();
        input.strict_guardian = true;
        input.initial_approval =
            RuntimeToolInitialApproval::Required(RuntimeToolApprovalKind::Guardian);

        orchestrate_runtime_tool_execution(input, &approvals, &runner)
            .await
            .expect("guardian retry");

        let requests = approvals.requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].kind, RuntimeToolApprovalKind::Guardian);
        assert_eq!(requests[1].kind, RuntimeToolApprovalKind::Guardian);
        assert_eq!(requests[1].phase, RuntimeToolApprovalPhase::Escalation);
    }

    #[tokio::test]
    async fn managed_network_denial_requires_scoped_approval_and_keeps_sandbox() {
        let approvals = RecordingApprovals::default();
        let runner = SequenceRunner {
            calls: Mutex::new(Vec::new()),
            first_error: Some(RuntimeToolExecutionError::new(
                "network blocked",
                Some(RuntimeToolPolicyErrorKind::ManagedNetworkDenied {
                    reason_code: "managed_network_denied".to_string(),
                    host: Some("https://example.com".to_string()),
                }),
            )),
        };

        orchestrate_runtime_tool_execution(input(), &approvals, &runner)
            .await
            .expect("network retry");

        let calls = runner.calls.lock().unwrap();
        assert_eq!(
            calls[1].effective_sandbox_policy(),
            RuntimeToolSandboxPolicy::WorkspaceWrite
        );
        assert!(network_is_granted(calls[1].granted_permissions()));
        let requests = approvals.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(
            requests[0].denial_kind,
            Some(RuntimeToolDenialKind::ManagedNetwork)
        );
        assert_eq!(
            requests[0].network_host.as_deref(),
            Some("https://example.com")
        );
    }

    #[tokio::test]
    async fn never_policy_rejects_explicit_escalation_before_approval_or_attempt() {
        let approvals = RecordingApprovals::default();
        let runner = SequenceRunner {
            calls: Mutex::new(Vec::new()),
            first_error: None,
        };
        let mut input = input();
        input.approval_policy = RuntimeToolApprovalPolicy::Never;
        input.explicit_sandbox_escalation = true;

        let error = orchestrate_runtime_tool_execution(input, &approvals, &runner)
            .await
            .expect_err("never must reject escalation");

        assert!(matches!(
            error.policy_kind(),
            Some(RuntimeToolPolicyErrorKind::PermissionDenied(reason))
                if reason == "sandbox_escalation_forbidden"
        ));
        assert!(approvals.requests.lock().unwrap().is_empty());
        assert!(runner.calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn ordinary_handler_failure_never_retries() {
        let approvals = Arc::new(RecordingApprovals::default());
        let runner = SequenceRunner {
            calls: Mutex::new(Vec::new()),
            first_error: Some(RuntimeToolExecutionError::new(
                "handler failed",
                Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
                    "handler_failure".to_string(),
                )),
            )),
        };

        orchestrate_runtime_tool_execution(input(), approvals.as_ref(), &runner)
            .await
            .expect_err("ordinary failure");

        assert_eq!(runner.calls.lock().unwrap().len(), 1);
        assert!(approvals.requests.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn apply_patch_outside_workspace_retries_after_approval() {
        let root = tempfile::tempdir().expect("tempdir");
        let workspace = root.path().join("workspace");
        std::fs::create_dir_all(&workspace).expect("workspace");
        let approvals = RecordingApprovals::default();
        let runner = ApplyPatchAttemptRunner {
            working_directory: workspace,
            params: json!({
                "patch": "*** Begin Patch\n*** Add File: ../outside.md\n+approved\n*** End Patch"
            }),
            calls: Mutex::new(Vec::new()),
        };
        let mut input = input();
        input.approval_policy = RuntimeToolApprovalPolicy::OnRequest;
        input.sandbox_denial_retry_allowed = true;

        let result = orchestrate_runtime_tool_execution(input, &approvals, &runner)
            .await
            .expect("approved apply_patch retry");

        assert!(result.success);
        assert_eq!(
            std::fs::read_to_string(root.path().join("outside.md")).expect("outside patch"),
            "approved\n"
        );
        let attempts = runner.calls.lock().unwrap();
        assert_eq!(attempts.len(), 2);
        assert_eq!(
            attempts[0].effective_sandbox_policy(),
            RuntimeToolSandboxPolicy::WorkspaceWrite
        );
        assert_eq!(
            attempts[1].effective_sandbox_policy(),
            RuntimeToolSandboxPolicy::DangerFullAccess
        );
        let approvals = approvals.requests.lock().unwrap();
        assert_eq!(approvals.len(), 1);
        assert_eq!(approvals[0].phase, RuntimeToolApprovalPhase::Escalation);
        assert_eq!(
            approvals[0].denial_kind,
            Some(RuntimeToolDenialKind::Sandbox)
        );
    }

    #[tokio::test]
    async fn timeout_never_retries() {
        let approvals = RecordingApprovals::default();
        let runner = SequenceRunner {
            calls: Mutex::new(Vec::new()),
            first_error: Some(RuntimeToolExecutionError::new(
                "execution timed out",
                Some(RuntimeToolPolicyErrorKind::Timeout(
                    "execution_timeout".to_string(),
                )),
            )),
        };

        let error = orchestrate_runtime_tool_execution(input(), &approvals, &runner)
            .await
            .expect_err("timeout must not retry");

        assert!(matches!(
            error.policy_kind(),
            Some(RuntimeToolPolicyErrorKind::Timeout(reason)) if reason == "execution_timeout"
        ));
        assert_eq!(runner.calls.lock().unwrap().len(), 1);
        assert!(approvals.requests.lock().unwrap().is_empty());
    }
}
