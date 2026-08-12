//! Session submission loop。
//!
//! Session 是串行调度边界：同一个 session 同时只运行一个 task，新的提交要么
//! 进入 FIFO 队列，要么显式返回 busy。steer 与 inter-agent mailbox 保持两条
//! 输入队列，task 可以在 sampling step 之间主动 drain pending input。

use futures::future::BoxFuture;
use serde_json::Value;
use std::fmt;
use std::sync::Arc;
use tokio::sync::oneshot;
use uuid::Uuid;

mod actor;
mod handle;
mod input_queue;
mod inter_agent;
mod registry;
mod resources;
mod step;

pub use handle::RuntimeSessionHandle;
pub use input_queue::{
    RuntimeSessionClosureTask, RuntimeSessionInput, RuntimeSessionInputActivity,
    RuntimeSessionInputHandle, RuntimeSessionMailboxDeliveryPhase, RuntimeSessionMailboxLoader,
    RuntimeSessionPendingResponse, RuntimeSessionResponseKind, RuntimeSessionTask,
    RuntimeSessionTaskContext, RuntimeSessionTaskKind, RuntimeSessionTaskOutcome,
};
pub use inter_agent::{
    RuntimeSessionInterAgentDeliveryMode, RuntimeSessionInterAgentInput,
    RuntimeSessionInterAgentMessageKind, RuntimeSessionInterAgentResultStatus,
};
pub use registry::RuntimeSessionRegistry;
pub use step::{RuntimeSessionStepContext, RuntimeSessionTokenUsage};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RuntimeSessionTaskFailure {
    pub message: String,
    pub reason_code: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeSessionSubmitResult {
    Started,
    Queued { position: usize },
    Busy,
}

pub struct RuntimeSessionSubmission {
    pub id: String,
    pub client_user_message_id: Option<String>,
    pub trace: Option<RuntimeSessionTraceContext>,
    pub result: RuntimeSessionSubmitResult,
    pub completion: oneshot::Receiver<Result<RuntimeSessionTaskOutcome, RuntimeSessionTaskFailure>>,
}

pub enum RuntimeSessionUserInputResult {
    Submitted(RuntimeSessionSubmission),
    Steered { id: String, turn_id: String },
}

/// W3C trace carrier propagated with a session operation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RuntimeSessionTraceContext {
    pub traceparent: Option<String>,
    pub tracestate: Option<String>,
}

/// Actor-ordered state used by App Server read/resume projections.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RuntimeSessionSnapshot {
    pub active_turn_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeSessionOperationContext {
    pub session_id: String,
    pub thread_id: String,
    pub submission_id: String,
    pub active_turn_id: Option<String>,
    pub client_user_message_id: Option<String>,
    pub trace: Option<RuntimeSessionTraceContext>,
}

#[derive(Clone)]
pub struct RuntimeSessionHandler {
    run: Arc<
        dyn Fn(
                RuntimeSessionOperationContext,
                tokio_util::sync::CancellationToken,
            ) -> BoxFuture<'static, Result<(), String>>
            + Send
            + Sync,
    >,
}

impl RuntimeSessionHandler {
    pub fn new(
        run: impl Fn(RuntimeSessionOperationContext) -> BoxFuture<'static, Result<(), String>>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        Self::new_with_cancellation(move |context, _cancellation_token| run(context))
    }

    pub fn new_with_cancellation(
        run: impl Fn(
                RuntimeSessionOperationContext,
                tokio_util::sync::CancellationToken,
            ) -> BoxFuture<'static, Result<(), String>>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        Self { run: Arc::new(run) }
    }

    async fn execute(&self, context: RuntimeSessionOperationContext) -> Result<(), String> {
        self.execute_with_cancellation(context, tokio_util::sync::CancellationToken::new())
            .await
    }

    async fn execute_with_cancellation(
        &self,
        context: RuntimeSessionOperationContext,
        cancellation_token: tokio_util::sync::CancellationToken,
    ) -> Result<(), String> {
        (self.run)(context, cancellation_token).await
    }
}

/// Operations accepted by the session dispatcher.
///
/// Existing handle methods lower into this operation type. Callers that need
/// operation identity can submit the envelope directly and receive a stable
/// operation receipt.
#[derive(Clone)]
pub enum RuntimeSessionOperation {
    StartTask {
        task: Arc<dyn RuntimeSessionTask>,
        queue_if_busy: bool,
        replace_active: bool,
    },
    UserInput {
        expected_turn_id: Option<String>,
        input: Vec<RuntimeSessionInput>,
        task: Option<Arc<dyn RuntimeSessionTask>>,
        queue_if_busy: bool,
    },
    Review {
        task: Arc<dyn RuntimeSessionTask>,
    },
    Compact {
        task: Arc<dyn RuntimeSessionTask>,
    },
    ThreadSettings {
        handler: RuntimeSessionHandler,
    },
    SetMemoryMode {
        handler: RuntimeSessionHandler,
    },
    RefreshMcp {
        handler: RuntimeSessionHandler,
    },
    ReloadConfig {
        handler: RuntimeSessionHandler,
    },
    RunShell {
        auxiliary: RuntimeSessionHandler,
        task: Arc<dyn RuntimeSessionTask>,
    },
    InterAgentCommunication {
        input: RuntimeSessionInterAgentInput,
    },
    ApprovalResponse {
        expected_turn_id: Option<String>,
        request_id: String,
        response: Value,
    },
    UserInputResponse {
        expected_turn_id: Option<String>,
        request_id: String,
        response: Value,
    },
    PermissionResponse {
        expected_turn_id: Option<String>,
        request_id: String,
        response: Value,
    },
    DynamicToolResponse {
        expected_turn_id: Option<String>,
        request_id: String,
        response: Value,
    },
    McpElicitationResponse {
        expected_turn_id: Option<String>,
        request_id: String,
        response: Value,
    },
    Interrupt {
        expected_turn_id: Option<String>,
    },
    Shutdown,
}

impl fmt::Debug for RuntimeSessionOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StartTask {
                task,
                queue_if_busy,
                replace_active,
            } => f
                .debug_struct("StartTask")
                .field("turn_id", &task.turn_id())
                .field("kind", &task.kind())
                .field("queue_if_busy", queue_if_busy)
                .field("replace_active", replace_active)
                .finish(),
            Self::UserInput {
                expected_turn_id,
                input,
                task,
                queue_if_busy,
            } => f
                .debug_struct("UserInput")
                .field("expected_turn_id", expected_turn_id)
                .field("input_count", &input.len())
                .field(
                    "candidate_turn_id",
                    &task.as_ref().map(|task| task.turn_id()),
                )
                .field("queue_if_busy", queue_if_busy)
                .finish(),
            Self::Review { task } | Self::Compact { task } => f
                .debug_struct("ReplacingSessionTaskOperation")
                .field("turn_id", &task.turn_id())
                .field("kind", &task.kind())
                .finish(),
            Self::ThreadSettings { .. } => f.write_str("ThreadSettings"),
            Self::SetMemoryMode { .. } => f.write_str("SetMemoryMode"),
            Self::RefreshMcp { .. } => f.write_str("RefreshMcp"),
            Self::ReloadConfig { .. } => f.write_str("ReloadConfig"),
            Self::RunShell { task, .. } => f
                .debug_struct("RunShell")
                .field("turn_id", &task.turn_id())
                .field("kind", &task.kind())
                .finish(),
            Self::InterAgentCommunication { input } => f
                .debug_struct("InterAgentCommunication")
                .field("message_id", &input.message_id)
                .field("sender_thread_id", &input.sender_thread_id)
                .field("recipient_thread_id", &input.recipient_thread_id)
                .field("delivery_mode", &input.delivery_mode)
                .finish(),
            Self::ApprovalResponse {
                expected_turn_id,
                request_id,
                ..
            } => f
                .debug_struct("ApprovalResponse")
                .field("expected_turn_id", expected_turn_id)
                .field("request_id", request_id)
                .finish(),
            Self::UserInputResponse {
                expected_turn_id,
                request_id,
                ..
            } => f
                .debug_struct("UserInputResponse")
                .field("expected_turn_id", expected_turn_id)
                .field("request_id", request_id)
                .finish(),
            Self::PermissionResponse {
                expected_turn_id,
                request_id,
                ..
            } => f
                .debug_struct("PermissionResponse")
                .field("expected_turn_id", expected_turn_id)
                .field("request_id", request_id)
                .finish(),
            Self::DynamicToolResponse {
                expected_turn_id,
                request_id,
                ..
            } => f
                .debug_struct("DynamicToolResponse")
                .field("expected_turn_id", expected_turn_id)
                .field("request_id", request_id)
                .finish(),
            Self::McpElicitationResponse {
                expected_turn_id,
                request_id,
                ..
            } => f
                .debug_struct("McpElicitationResponse")
                .field("expected_turn_id", expected_turn_id)
                .field("request_id", request_id)
                .finish(),
            Self::Interrupt { expected_turn_id } => f
                .debug_struct("Interrupt")
                .field("expected_turn_id", expected_turn_id)
                .finish(),
            Self::Shutdown => f.write_str("Shutdown"),
        }
    }
}

/// Envelope carrying operation identity and propagation metadata.
#[derive(Clone, Debug)]
pub struct RuntimeSessionOperationSubmission {
    pub id: String,
    pub operation: RuntimeSessionOperation,
    pub client_user_message_id: Option<String>,
    pub trace: Option<RuntimeSessionTraceContext>,
}

impl RuntimeSessionOperationSubmission {
    pub fn new(operation: RuntimeSessionOperation) -> Self {
        Self::with_id(Uuid::now_v7().to_string(), operation, None, None)
    }

    pub fn with_metadata(
        operation: RuntimeSessionOperation,
        client_user_message_id: Option<String>,
        trace: Option<RuntimeSessionTraceContext>,
    ) -> Self {
        Self::with_id(
            Uuid::now_v7().to_string(),
            operation,
            client_user_message_id,
            trace,
        )
    }

    pub fn with_id(
        id: impl Into<String>,
        operation: RuntimeSessionOperation,
        client_user_message_id: Option<String>,
        trace: Option<RuntimeSessionTraceContext>,
    ) -> Self {
        Self {
            id: id.into(),
            operation,
            client_user_message_id,
            trace,
        }
    }
}

/// Result returned by the unified session operation dispatcher.
pub enum RuntimeSessionOperationResult {
    Submission(RuntimeSessionSubmission),
    Accepted { id: String, turn_id: Option<String> },
    Interrupted { id: String, interrupted: bool },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeSessionLoopError {
    Closed,
    InvalidTask(String),
    OperationFailed(String),
}

impl fmt::Display for RuntimeSessionLoopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Closed => f.write_str("runtime session submission loop is closed"),
            Self::InvalidTask(message) => f.write_str(message),
            Self::OperationFailed(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for RuntimeSessionLoopError {}

#[cfg(test)]
use actor::RuntimeSessionActor;
#[cfg(test)]
mod activity_tests;
#[cfg(test)]
mod tests;
