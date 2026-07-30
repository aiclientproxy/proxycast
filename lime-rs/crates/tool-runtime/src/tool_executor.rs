use crate::tool_call::ToolCall;
use crate::tool_definition::{RuntimeToolDefinition, RuntimeToolExposure};
use crate::tool_io::{ToolOutputReference, ToolOutputTruncation};
use crate::tool_result_projection::NormalizedToolOutput;
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone)]
pub struct RuntimeToolExecutionContextInput {
    pub working_directory: PathBuf,
    pub session_id: String,
    pub cancel_token: Option<CancellationToken>,
    pub workspace_sandbox: Option<RuntimeWorkspaceSandboxInput>,
}

#[derive(Debug, Clone)]
pub struct RuntimeToolExecutionContext {
    working_directory: PathBuf,
    session_id: String,
    cancel_token: Option<CancellationToken>,
    workspace_sandbox: Option<RuntimeWorkspaceSandboxInput>,
    environment: HashMap<String, String>,
    tool_identity: Option<RuntimeToolExecutionIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeToolExecutionIdentity {
    call_id: String,
    turn_id: String,
}

impl RuntimeToolExecutionIdentity {
    pub fn new(call_id: impl Into<String>, turn_id: impl Into<String>) -> Self {
        Self {
            call_id: call_id.into(),
            turn_id: turn_id.into(),
        }
    }

    pub fn call_id(&self) -> &str {
        &self.call_id
    }

    pub fn turn_id(&self) -> &str {
        &self.turn_id
    }
}

impl RuntimeToolExecutionContext {
    pub fn new(input: RuntimeToolExecutionContextInput) -> Self {
        Self {
            working_directory: input.working_directory,
            session_id: input.session_id,
            cancel_token: input.cancel_token,
            workspace_sandbox: input.workspace_sandbox,
            environment: HashMap::new(),
            tool_identity: None,
        }
    }

    pub fn working_directory(&self) -> &PathBuf {
        &self.working_directory
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn cancel_token(&self) -> Option<&CancellationToken> {
        self.cancel_token.as_ref()
    }

    pub fn workspace_sandbox(&self) -> Option<&RuntimeWorkspaceSandboxInput> {
        self.workspace_sandbox.as_ref()
    }

    pub fn has_workspace_sandbox(&self) -> bool {
        self.workspace_sandbox.is_some()
    }

    pub fn environment(&self) -> &HashMap<String, String> {
        &self.environment
    }

    pub fn tool_identity(&self) -> Option<&RuntimeToolExecutionIdentity> {
        self.tool_identity.as_ref()
    }

    pub fn with_tool_identity(mut self, identity: RuntimeToolExecutionIdentity) -> Self {
        self.tool_identity = Some(identity);
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeWorkspaceSandboxInput {
    metadata: HashMap<String, Value>,
}

impl RuntimeWorkspaceSandboxInput {
    pub fn from_policy_metadata(metadata: &HashMap<String, Value>) -> Self {
        Self {
            metadata: metadata.clone(),
        }
    }

    pub fn metadata(&self) -> &HashMap<String, Value> {
        &self.metadata
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RuntimeToolExecutionRequest<'a> {
    pub tool_name: &'a str,
    pub params: &'a Value,
    pub context: &'a RuntimeToolExecutionContext,
    pub turn_context: Option<&'a RuntimeToolTurnContext>,
}

pub type RuntimeToolTurnContext = agent_protocol::turn_context::TurnContextOverride;

pub const TOOL_APPROVAL_GRANTED_METADATA_KEY: &str = "tool_approval_granted";

pub fn turn_context_has_tool_approval(turn_context: Option<&RuntimeToolTurnContext>) -> bool {
    turn_context
        .and_then(|context| context.metadata.get(TOOL_APPROVAL_GRANTED_METADATA_KEY))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

pub type RuntimeToolExecutionFuture<'a> = Pin<
    Box<
        dyn Future<Output = Result<RuntimeToolExecutionResult, RuntimeToolExecutionError>>
            + Send
            + 'a,
    >,
>;

pub trait RuntimeToolExecutor: Send + Sync {
    fn execute<'a>(
        &'a self,
        request: RuntimeToolExecutionRequest<'a>,
    ) -> RuntimeToolExecutionFuture<'a>;

    fn execute_call<'a>(
        &'a self,
        call: &'a ToolCall,
        context: &'a RuntimeToolExecutionContext,
        turn_context: Option<&'a RuntimeToolTurnContext>,
    ) -> RuntimeToolExecutionFuture<'a> {
        Box::pin(async move {
            self.execute(RuntimeToolExecutionRequest {
                tool_name: call.tool_name(),
                params: call.arguments(),
                context,
                turn_context,
            })
            .await
        })
    }
}

#[derive(Clone)]
pub struct RuntimeToolExecutorHandle {
    executor: Arc<dyn RuntimeToolExecutor>,
}

impl RuntimeToolExecutorHandle {
    pub fn new(executor: Arc<dyn RuntimeToolExecutor>) -> Self {
        Self { executor }
    }

    pub fn inner_executor(&self) -> Arc<dyn RuntimeToolExecutor> {
        Arc::clone(&self.executor)
    }

    pub async fn execute(
        &self,
        request: RuntimeToolExecutionRequest<'_>,
    ) -> Result<RuntimeToolExecutionResult, RuntimeToolExecutionError> {
        self.executor.execute(request).await
    }

    pub async fn execute_call(
        &self,
        call: &ToolCall,
        context: &RuntimeToolExecutionContext,
        turn_context: Option<&RuntimeToolTurnContext>,
    ) -> Result<RuntimeToolExecutionResult, RuntimeToolExecutionError> {
        let context = context
            .clone()
            .with_tool_identity(RuntimeToolExecutionIdentity::new(
                call.call_id(),
                call.turn_id(),
            ));
        self.executor
            .execute_call(call, &context, turn_context)
            .await
    }

    pub fn bind(
        self,
        definition: RuntimeToolDefinition,
        exposure: RuntimeToolExposure,
    ) -> RuntimeTool {
        RuntimeTool::new(definition, exposure, self)
    }
}

/// Binds the model-visible definition and exposure to its executable runtime.
#[derive(Clone)]
pub struct RuntimeTool {
    definition: RuntimeToolDefinition,
    exposure: RuntimeToolExposure,
    executor: RuntimeToolExecutorHandle,
}

impl RuntimeTool {
    pub fn new(
        definition: RuntimeToolDefinition,
        exposure: RuntimeToolExposure,
        executor: RuntimeToolExecutorHandle,
    ) -> Self {
        Self {
            definition,
            exposure,
            executor,
        }
    }

    pub fn definition(&self) -> &RuntimeToolDefinition {
        &self.definition
    }

    pub fn exposure(&self) -> RuntimeToolExposure {
        self.exposure
    }

    pub async fn execute_call(
        &self,
        call: &ToolCall,
        context: &RuntimeToolExecutionContext,
        turn_context: Option<&RuntimeToolTurnContext>,
    ) -> NormalizedToolOutput {
        call.emit_started().await;
        let started_at = Instant::now();
        let outcome = if call.tool_name() == self.definition.name {
            match context.cancel_token().cloned() {
                Some(cancel_token) if cancel_token.is_cancelled() => {
                    RuntimeToolExecutionOutcome::Aborted
                }
                Some(cancel_token) => {
                    let execution = self.executor.execute_call(call, context, turn_context);
                    tokio::select! {
                        biased;
                        _ = cancel_token.cancelled() => RuntimeToolExecutionOutcome::Aborted,
                        result = execution => RuntimeToolExecutionOutcome::from_execution_result(result),
                    }
                }
                None => RuntimeToolExecutionOutcome::from_execution_result(
                    self.executor
                        .execute_call(call, context, turn_context)
                        .await,
                ),
            }
        } else {
            RuntimeToolExecutionOutcome::Error(RuntimeToolExecutionFailure::from_error(
                RuntimeToolExecutionError::new(
                    format!(
                        "tool call name '{}' does not match bound runtime '{}'",
                        call.tool_name(),
                        self.definition.name
                    ),
                    None,
                )
                .before_handler(),
            ))
        };
        let duration_ms = u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX);
        let output = NormalizedToolOutput::from_execution_outcome(outcome, duration_ms);
        call.emit_completed(output.clone()).await;
        output
    }
}

impl std::fmt::Debug for RuntimeTool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RuntimeTool")
            .field("definition", &self.definition)
            .field("exposure", &self.exposure)
            .field("executor", &"<runtime tool executor>")
            .finish()
    }
}

pub async fn run_runtime_tool_execution(
    executor: &RuntimeToolExecutorHandle,
    request: RuntimeToolExecutionRequest<'_>,
) -> RuntimeToolExecutionOutcome {
    RuntimeToolExecutionOutcome::from_execution_result(executor.execute(request).await)
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeToolExecutionResult {
    pub success: bool,
    pub output: String,
    pub structured_content: Option<Value>,
    pub error: Option<String>,
    pub truncation: Option<ToolOutputTruncation>,
    pub sidecar_reference: Option<ToolOutputReference>,
    pub metadata: HashMap<String, Value>,
    pub agent_control_projection_facts: Vec<crate::agent_control::SubAgentProjectionFact>,
    pub agent_control_state_facts: Vec<crate::agent_control::AgentStateProjectionFact>,
}

impl RuntimeToolExecutionResult {
    pub fn new(
        success: bool,
        output: String,
        error: Option<String>,
        metadata: HashMap<String, Value>,
    ) -> Self {
        Self {
            success,
            output,
            structured_content: None,
            error,
            truncation: None,
            sidecar_reference: None,
            metadata,
            agent_control_projection_facts: Vec::new(),
            agent_control_state_facts: Vec::new(),
        }
    }

    pub fn with_structured_content(mut self, structured_content: Value) -> Self {
        self.structured_content = Some(structured_content);
        self
    }

    pub fn with_truncation(mut self, truncation: ToolOutputTruncation) -> Self {
        self.truncation = Some(truncation);
        self
    }

    pub fn with_sidecar_reference(mut self, sidecar_reference: ToolOutputReference) -> Self {
        self.sidecar_reference = Some(sidecar_reference);
        self
    }

    pub fn with_agent_control_projection_facts(
        mut self,
        facts: Vec<crate::agent_control::SubAgentProjectionFact>,
    ) -> Self {
        self.agent_control_projection_facts = facts;
        self
    }

    pub fn with_agent_control_state_facts(
        mut self,
        facts: Vec<crate::agent_control::AgentStateProjectionFact>,
    ) -> Self {
        self.agent_control_state_facts = facts;
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeToolExecutionOutcome {
    Result(RuntimeToolExecutionResult),
    Error(RuntimeToolExecutionFailure),
    Aborted,
}

impl RuntimeToolExecutionOutcome {
    pub fn from_execution_result(
        result: Result<RuntimeToolExecutionResult, RuntimeToolExecutionError>,
    ) -> Self {
        match result {
            Ok(result) => Self::Result(result),
            Err(error) => Self::Error(RuntimeToolExecutionFailure::from_error(error)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeToolExecutionError {
    message: String,
    policy_kind: Option<RuntimeToolPolicyErrorKind>,
    handler_executed: bool,
}

impl RuntimeToolExecutionError {
    pub fn new(
        message: impl Into<String>,
        policy_kind: Option<RuntimeToolPolicyErrorKind>,
    ) -> Self {
        Self {
            message: message.into(),
            policy_kind,
            handler_executed: true,
        }
    }

    pub fn before_handler(mut self) -> Self {
        self.handler_executed = false;
        self
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn policy_kind(&self) -> Option<&RuntimeToolPolicyErrorKind> {
        self.policy_kind.as_ref()
    }

    pub fn handler_executed(&self) -> bool {
        self.handler_executed
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeToolExecutionFailure {
    message: String,
    kind: RuntimeToolExecutionFailureKind,
    reason_code: Option<String>,
    handler_executed: bool,
}

impl RuntimeToolExecutionFailure {
    pub fn from_error(error: RuntimeToolExecutionError) -> Self {
        let reason_code = error.policy_kind().map(|kind| match kind {
            RuntimeToolPolicyErrorKind::PermissionDenied(reason)
            | RuntimeToolPolicyErrorKind::SafetyCheckFailed(reason)
            | RuntimeToolPolicyErrorKind::ExecutionFailed(reason) => reason.clone(),
        });
        let kind = match error.policy_kind() {
            Some(RuntimeToolPolicyErrorKind::PermissionDenied(_)) => {
                RuntimeToolExecutionFailureKind::PermissionDenied
            }
            Some(RuntimeToolPolicyErrorKind::SafetyCheckFailed(_)) => {
                RuntimeToolExecutionFailureKind::SafetyCheckFailed
            }
            Some(RuntimeToolPolicyErrorKind::ExecutionFailed(_)) | None => {
                RuntimeToolExecutionFailureKind::ExecutionFailed
            }
        };

        Self {
            message: error.message().to_string(),
            kind,
            reason_code,
            handler_executed: error.handler_executed(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn kind(&self) -> RuntimeToolExecutionFailureKind {
        self.kind
    }

    pub fn reason_code(&self) -> Option<&str> {
        self.reason_code.as_deref()
    }

    pub fn handler_executed(&self) -> bool {
        self.handler_executed
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeToolExecutionFailureKind {
    PermissionDenied,
    SafetyCheckFailed,
    ExecutionFailed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeToolPolicyErrorKind {
    PermissionDenied(String),
    SafetyCheckFailed(String),
    ExecutionFailed(String),
}

impl RuntimeToolPolicyErrorKind {
    pub fn classification(&self) -> Option<RuntimeToolPolicyErrorClassification<'_>> {
        match self {
            RuntimeToolPolicyErrorKind::PermissionDenied(reason) => {
                Some(RuntimeToolPolicyErrorClassification {
                    event_class: "permission.denied",
                    failure_category: "permission_denied",
                    reason_code: "permission_denied",
                    reason,
                })
            }
            RuntimeToolPolicyErrorKind::SafetyCheckFailed(reason) => {
                Some(RuntimeToolPolicyErrorClassification {
                    event_class: "permission.denied",
                    failure_category: "policy_denied",
                    reason_code: "safety_check_failed",
                    reason,
                })
            }
            RuntimeToolPolicyErrorKind::ExecutionFailed(reason)
                if looks_like_sandbox_block(reason) =>
            {
                Some(RuntimeToolPolicyErrorClassification {
                    event_class: "sandbox.blocked",
                    failure_category: "sandbox_blocked",
                    reason_code: "sandbox_blocked",
                    reason,
                })
            }
            RuntimeToolPolicyErrorKind::ExecutionFailed(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeToolPolicyErrorClassification<'a> {
    pub event_class: &'static str,
    pub failure_category: &'static str,
    pub reason_code: &'static str,
    pub reason: &'a str,
}

fn looks_like_sandbox_block(reason: &str) -> bool {
    let normalized = reason.to_ascii_lowercase();
    normalized.contains("sandbox")
        && (normalized.contains("block")
            || normalized.contains("denied")
            || normalized.contains("not permitted"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn runtime_tool_result_keeps_terminal_fields() {
        let result = RuntimeToolExecutionResult::new(
            true,
            "ok".to_string(),
            None,
            HashMap::from([("source".to_string(), json!("runtime"))]),
        );

        assert!(result.success);
        assert_eq!(result.output, "ok");
        assert_eq!(result.metadata.get("source"), Some(&json!("runtime")));
    }

    #[test]
    fn runtime_tool_context_keeps_execution_inputs_without_agent_types() {
        let metadata = HashMap::from([("sandboxBackend".to_string(), json!("seatbelt"))]);
        let context = RuntimeToolExecutionContext::new(RuntimeToolExecutionContextInput {
            working_directory: PathBuf::from("/tmp/workspace"),
            session_id: "session-1".to_string(),
            cancel_token: None,
            workspace_sandbox: Some(RuntimeWorkspaceSandboxInput::from_policy_metadata(
                &metadata,
            )),
        });

        assert_eq!(
            context.working_directory(),
            &PathBuf::from("/tmp/workspace")
        );
        assert_eq!(context.session_id(), "session-1");
        assert!(context.has_workspace_sandbox());
        assert_eq!(
            context
                .workspace_sandbox()
                .map(|sandbox| sandbox.metadata()),
            Some(&metadata)
        );
        assert!(context.environment().is_empty());
        assert!(context.tool_identity().is_none());
    }

    struct IdentityExecutor;

    impl RuntimeToolExecutor for IdentityExecutor {
        fn execute<'a>(
            &'a self,
            request: RuntimeToolExecutionRequest<'a>,
        ) -> RuntimeToolExecutionFuture<'a> {
            Box::pin(async move {
                let identity = request
                    .context
                    .tool_identity()
                    .expect("execute_call must bind canonical tool identity");
                Ok(RuntimeToolExecutionResult::new(
                    true,
                    format!("{}:{}", identity.call_id(), identity.turn_id()),
                    None,
                    HashMap::new(),
                ))
            })
        }
    }

    struct NoopLifecycleEmitter;

    impl crate::tool_lifecycle::ToolLifecycleEmitter for NoopLifecycleEmitter {
        fn emit<'a>(
            &'a self,
            _event: crate::tool_lifecycle::ToolLifecycleEvent,
        ) -> crate::tool_lifecycle::ToolLifecycleEmissionFuture<'a> {
            Box::pin(async {})
        }
    }

    #[tokio::test]
    async fn execute_call_binds_typed_identity_without_mutating_turn_metadata() {
        let context = RuntimeToolExecutionContext::new(RuntimeToolExecutionContextInput {
            working_directory: PathBuf::from("/tmp/workspace"),
            session_id: "session-identity".to_string(),
            cancel_token: None,
            workspace_sandbox: None,
        });
        let call = ToolCall::new(
            "turn-canonical",
            "call-canonical",
            "Echo",
            json!({}),
            Vec::new(),
            Arc::new(NoopLifecycleEmitter),
        );
        let turn_context = RuntimeToolTurnContext {
            metadata: HashMap::from([
                ("tool_call_id".to_string(), json!("call-metadata")),
                ("turn_id".to_string(), json!("turn-metadata")),
            ]),
            ..RuntimeToolTurnContext::default()
        };
        let executor = RuntimeToolExecutorHandle::new(Arc::new(IdentityExecutor));

        let result = executor
            .execute_call(&call, &context, Some(&turn_context))
            .await
            .expect("identified call should execute");

        assert_eq!(result.output, "call-canonical:turn-canonical");
        assert_eq!(
            turn_context.metadata.get("tool_call_id"),
            Some(&json!("call-metadata"))
        );
        assert_eq!(
            turn_context.metadata.get("turn_id"),
            Some(&json!("turn-metadata"))
        );
    }

    struct EchoExecutor;

    impl RuntimeToolExecutor for EchoExecutor {
        fn execute<'a>(
            &'a self,
            request: RuntimeToolExecutionRequest<'a>,
        ) -> RuntimeToolExecutionFuture<'a> {
            Box::pin(async move {
                Ok(RuntimeToolExecutionResult::new(
                    true,
                    format!("{}:{}", request.context.session_id(), request.tool_name),
                    None,
                    HashMap::new(),
                ))
            })
        }
    }

    #[tokio::test]
    async fn runtime_tool_executor_handle_dispatches_current_request() {
        let context = RuntimeToolExecutionContext::new(RuntimeToolExecutionContextInput {
            working_directory: PathBuf::from("/tmp/workspace"),
            session_id: "session-2".to_string(),
            cancel_token: None,
            workspace_sandbox: None,
        });
        let params = json!({});
        let executor = RuntimeToolExecutorHandle::new(Arc::new(EchoExecutor));

        let result = executor
            .execute(RuntimeToolExecutionRequest {
                tool_name: "Echo",
                params: &params,
                context: &context,
                turn_context: None,
            })
            .await
            .expect("runtime executor should return result");

        assert!(result.success);
        assert_eq!(result.output, "session-2:Echo");
    }

    #[tokio::test]
    async fn runtime_tool_runner_materializes_current_result_outcome() {
        let context = RuntimeToolExecutionContext::new(RuntimeToolExecutionContextInput {
            working_directory: PathBuf::from("/tmp/workspace"),
            session_id: "session-3".to_string(),
            cancel_token: None,
            workspace_sandbox: None,
        });
        let params = json!({});
        let executor = RuntimeToolExecutorHandle::new(Arc::new(EchoExecutor));

        let outcome = run_runtime_tool_execution(
            &executor,
            RuntimeToolExecutionRequest {
                tool_name: "Echo",
                params: &params,
                context: &context,
                turn_context: None,
            },
        )
        .await;

        match outcome {
            RuntimeToolExecutionOutcome::Result(result) => {
                assert!(result.success);
                assert_eq!(result.output, "session-3:Echo");
            }
            RuntimeToolExecutionOutcome::Error(error) => {
                panic!("runtime runner should not fail: {error:?}");
            }
            RuntimeToolExecutionOutcome::Aborted => {
                panic!("runtime runner should not abort without a cancellation token");
            }
        }
    }

    struct FailingExecutor {
        error: RuntimeToolExecutionError,
    }

    impl RuntimeToolExecutor for FailingExecutor {
        fn execute<'a>(
            &'a self,
            _request: RuntimeToolExecutionRequest<'a>,
        ) -> RuntimeToolExecutionFuture<'a> {
            Box::pin(async move { Err(self.error.clone()) })
        }
    }

    #[tokio::test]
    async fn runtime_tool_runner_materializes_policy_failure_outcome() {
        let context = RuntimeToolExecutionContext::new(RuntimeToolExecutionContextInput {
            working_directory: PathBuf::from("/tmp/workspace"),
            session_id: "session-4".to_string(),
            cancel_token: None,
            workspace_sandbox: None,
        });
        let params = json!({});
        let executor = RuntimeToolExecutorHandle::new(Arc::new(FailingExecutor {
            error: RuntimeToolExecutionError::new(
                "blocked by policy",
                Some(RuntimeToolPolicyErrorKind::PermissionDenied(
                    "permission_denied".to_string(),
                )),
            ),
        }));

        let outcome = run_runtime_tool_execution(
            &executor,
            RuntimeToolExecutionRequest {
                tool_name: "Echo",
                params: &params,
                context: &context,
                turn_context: None,
            },
        )
        .await;

        match outcome {
            RuntimeToolExecutionOutcome::Result(result) => {
                panic!("runtime runner should fail, got {result:?}");
            }
            RuntimeToolExecutionOutcome::Error(error) => {
                assert_eq!(error.message(), "blocked by policy");
                assert_eq!(
                    error.kind(),
                    RuntimeToolExecutionFailureKind::PermissionDenied
                );
            }
            RuntimeToolExecutionOutcome::Aborted => {
                panic!("runtime runner should not abort without a cancellation token");
            }
        }
    }

    #[test]
    fn policy_error_classification_projects_gui_metadata() {
        let permission_error = RuntimeToolPolicyErrorKind::PermissionDenied("blocked".to_string());
        let permission = permission_error
            .classification()
            .expect("permission denial should classify");
        assert_eq!(permission.event_class, "permission.denied");
        assert_eq!(permission.failure_category, "permission_denied");
        assert_eq!(permission.reason_code, "permission_denied");

        let sandbox_error =
            RuntimeToolPolicyErrorKind::ExecutionFailed("sandbox denied file write".to_string());
        let sandbox = sandbox_error
            .classification()
            .expect("sandbox execution failure should classify");
        assert_eq!(sandbox.event_class, "sandbox.blocked");
        assert_eq!(sandbox.failure_category, "sandbox_blocked");
        assert_eq!(sandbox.reason_code, "sandbox_blocked");

        assert!(
            RuntimeToolPolicyErrorKind::ExecutionFailed("command failed".to_string())
                .classification()
                .is_none()
        );
    }
}
