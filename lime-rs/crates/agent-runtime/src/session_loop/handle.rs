use super::actor::RuntimeSessionCommand;
use super::resources::RuntimeSessionResources;
use super::{
    RuntimeSessionInput, RuntimeSessionInterAgentInput, RuntimeSessionLoopError,
    RuntimeSessionOperation, RuntimeSessionOperationResult, RuntimeSessionOperationSubmission,
    RuntimeSessionSnapshot, RuntimeSessionSubmission, RuntimeSessionSubmitResult,
    RuntimeSessionTask, RuntimeSessionTaskFailure, RuntimeSessionTaskOutcome,
    RuntimeSessionTraceContext, RuntimeSessionUserInputResult,
};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, watch};

#[derive(Clone)]
pub struct RuntimeSessionHandle {
    pub(super) tx: mpsc::Sender<RuntimeSessionCommand>,
    pub(super) termination: watch::Receiver<bool>,
    pub(super) resources: Arc<RuntimeSessionResources>,
}

impl RuntimeSessionHandle {
    pub fn thread_id(&self) -> &str {
        self.resources.thread_id()
    }

    pub async fn snapshot(&self) -> Result<RuntimeSessionSnapshot, RuntimeSessionLoopError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(RuntimeSessionCommand::Snapshot { reply: reply_tx })
            .await
            .map_err(|_| RuntimeSessionLoopError::Closed)?;
        reply_rx.await.map_err(|_| RuntimeSessionLoopError::Closed)
    }

    pub async fn submit(
        &self,
        task: Arc<dyn RuntimeSessionTask>,
        queue_if_busy: bool,
    ) -> Result<RuntimeSessionSubmission, RuntimeSessionLoopError> {
        self.submit_with_policy(task, queue_if_busy, false).await
    }

    pub async fn submit_replacing(
        &self,
        task: Arc<dyn RuntimeSessionTask>,
    ) -> Result<RuntimeSessionSubmission, RuntimeSessionLoopError> {
        self.submit_with_policy(task, false, true).await
    }

    async fn submit_with_policy(
        &self,
        task: Arc<dyn RuntimeSessionTask>,
        queue_if_busy: bool,
        replace_active: bool,
    ) -> Result<RuntimeSessionSubmission, RuntimeSessionLoopError> {
        if task.turn_id().trim().is_empty() {
            return Err(RuntimeSessionLoopError::InvalidTask(
                "runtime session task requires a canonical turn_id".to_string(),
            ));
        }
        match self
            .dispatch(RuntimeSessionOperationSubmission::new(
                RuntimeSessionOperation::StartTask {
                    task,
                    queue_if_busy,
                    replace_active,
                },
            ))
            .await?
        {
            RuntimeSessionOperationResult::Submission(submission) => Ok(submission),
            RuntimeSessionOperationResult::Accepted { .. }
            | RuntimeSessionOperationResult::Interrupted { .. } => {
                Err(RuntimeSessionLoopError::InvalidTask(
                    "runtime session submit returned an invalid operation result".to_string(),
                ))
            }
        }
    }

    pub async fn submit_user_input_with_metadata(
        &self,
        task: Arc<dyn RuntimeSessionTask>,
        input: Vec<RuntimeSessionInput>,
        queue_if_busy: bool,
        client_user_message_id: Option<String>,
        trace: Option<RuntimeSessionTraceContext>,
    ) -> Result<RuntimeSessionUserInputResult, RuntimeSessionLoopError> {
        match self
            .dispatch(RuntimeSessionOperationSubmission::with_metadata(
                RuntimeSessionOperation::UserInput {
                    expected_turn_id: None,
                    input,
                    task: Some(task),
                    queue_if_busy,
                },
                client_user_message_id,
                trace,
            ))
            .await?
        {
            RuntimeSessionOperationResult::Submission(submission) => {
                Ok(RuntimeSessionUserInputResult::Submitted(submission))
            }
            RuntimeSessionOperationResult::Accepted {
                id,
                turn_id: Some(turn_id),
            } => Ok(RuntimeSessionUserInputResult::Steered { id, turn_id }),
            RuntimeSessionOperationResult::Accepted { turn_id: None, .. }
            | RuntimeSessionOperationResult::Interrupted { .. } => {
                Err(RuntimeSessionLoopError::InvalidTask(
                    "runtime session user input returned an invalid operation result".to_string(),
                ))
            }
        }
    }

    /// Dispatch a typed operation through the session's single command queue.
    pub async fn dispatch(
        &self,
        submission: RuntimeSessionOperationSubmission,
    ) -> Result<RuntimeSessionOperationResult, RuntimeSessionLoopError> {
        validate_operation_submission(&submission)?;
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(RuntimeSessionCommand::Operation {
                submission,
                reply: reply_tx,
            })
            .await
            .map_err(|_| RuntimeSessionLoopError::Closed)?;
        reply_rx
            .await
            .map_err(|_| RuntimeSessionLoopError::Closed)?
    }

    pub async fn steer(
        &self,
        input: Vec<RuntimeSessionInput>,
    ) -> Result<(), RuntimeSessionLoopError> {
        self.steer_for_turn(None, input).await
    }

    pub async fn steer_for_turn(
        &self,
        expected_turn_id: Option<&str>,
        input: Vec<RuntimeSessionInput>,
    ) -> Result<(), RuntimeSessionLoopError> {
        self.steer_for_turn_id(expected_turn_id, input)
            .await
            .map(|_| ())
    }

    pub async fn steer_for_turn_id(
        &self,
        expected_turn_id: Option<&str>,
        input: Vec<RuntimeSessionInput>,
    ) -> Result<String, RuntimeSessionLoopError> {
        self.steer_for_turn_id_with_metadata(expected_turn_id, input, None, None)
            .await
    }

    pub async fn steer_for_turn_id_with_metadata(
        &self,
        expected_turn_id: Option<&str>,
        input: Vec<RuntimeSessionInput>,
        client_user_message_id: Option<String>,
        trace: Option<RuntimeSessionTraceContext>,
    ) -> Result<String, RuntimeSessionLoopError> {
        if input.is_empty() {
            return Err(RuntimeSessionLoopError::InvalidTask(
                "runtime session steer input must not be empty".to_string(),
            ));
        }
        match self
            .dispatch(RuntimeSessionOperationSubmission::with_metadata(
                RuntimeSessionOperation::UserInput {
                    expected_turn_id: expected_turn_id.map(str::to_string),
                    input,
                    task: None,
                    queue_if_busy: false,
                },
                client_user_message_id,
                trace,
            ))
            .await?
        {
            RuntimeSessionOperationResult::Accepted {
                turn_id: Some(turn_id),
                ..
            } => Ok(turn_id),
            RuntimeSessionOperationResult::Accepted { turn_id: None, .. } => {
                Err(RuntimeSessionLoopError::InvalidTask(
                    "runtime session steer result requires an active turn id".to_string(),
                ))
            }
            RuntimeSessionOperationResult::Submission(_)
            | RuntimeSessionOperationResult::Interrupted { .. } => {
                Err(RuntimeSessionLoopError::InvalidTask(
                    "runtime session steer returned an invalid operation result".to_string(),
                ))
            }
        }
    }

    pub async fn notify_inter_agent_communication(
        &self,
        input: RuntimeSessionInterAgentInput,
    ) -> Result<(), RuntimeSessionLoopError> {
        match self
            .dispatch(RuntimeSessionOperationSubmission::new(
                RuntimeSessionOperation::InterAgentCommunication { input },
            ))
            .await?
        {
            RuntimeSessionOperationResult::Accepted { .. } => Ok(()),
            RuntimeSessionOperationResult::Submission(_)
            | RuntimeSessionOperationResult::Interrupted { .. } => {
                Err(RuntimeSessionLoopError::InvalidTask(
                    "runtime session inter-agent communication returned an invalid operation result"
                        .to_string(),
                ))
            }
        }
    }

    pub async fn subscribe_input_activity(
        &self,
    ) -> Result<
        (
            watch::Receiver<super::RuntimeSessionInputActivity>,
            Option<super::RuntimeSessionInputActivity>,
        ),
        RuntimeSessionLoopError,
    > {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(RuntimeSessionCommand::SubscribeInputActivity { reply: reply_tx })
            .await
            .map_err(|_| RuntimeSessionLoopError::Closed)?;
        reply_rx.await.map_err(|_| RuntimeSessionLoopError::Closed)
    }

    pub async fn approve(
        &self,
        expected_turn_id: Option<&str>,
        request_id: impl Into<String>,
        response: Value,
    ) -> Result<(), RuntimeSessionLoopError> {
        self.dispatch_response(RuntimeSessionOperation::ApprovalResponse {
            expected_turn_id: expected_turn_id.map(str::to_string),
            request_id: request_id.into(),
            response,
        })
        .await
    }

    pub async fn answer_user_input(
        &self,
        expected_turn_id: Option<&str>,
        request_id: impl Into<String>,
        response: Value,
    ) -> Result<(), RuntimeSessionLoopError> {
        self.dispatch_response(RuntimeSessionOperation::UserInputResponse {
            expected_turn_id: expected_turn_id.map(str::to_string),
            request_id: request_id.into(),
            response,
        })
        .await
    }

    pub async fn respond_permission(
        &self,
        expected_turn_id: Option<&str>,
        request_id: impl Into<String>,
        response: Value,
    ) -> Result<(), RuntimeSessionLoopError> {
        self.dispatch_response(RuntimeSessionOperation::PermissionResponse {
            expected_turn_id: expected_turn_id.map(str::to_string),
            request_id: request_id.into(),
            response,
        })
        .await
    }

    pub async fn respond_dynamic_tool(
        &self,
        expected_turn_id: Option<&str>,
        request_id: impl Into<String>,
        response: Value,
    ) -> Result<(), RuntimeSessionLoopError> {
        self.dispatch_response(RuntimeSessionOperation::DynamicToolResponse {
            expected_turn_id: expected_turn_id.map(str::to_string),
            request_id: request_id.into(),
            response,
        })
        .await
    }

    pub async fn resolve_mcp_elicitation(
        &self,
        expected_turn_id: Option<&str>,
        request_id: impl Into<String>,
        response: Value,
    ) -> Result<(), RuntimeSessionLoopError> {
        self.dispatch_response(RuntimeSessionOperation::McpElicitationResponse {
            expected_turn_id: expected_turn_id.map(str::to_string),
            request_id: request_id.into(),
            response,
        })
        .await
    }

    async fn dispatch_response(
        &self,
        operation: RuntimeSessionOperation,
    ) -> Result<(), RuntimeSessionLoopError> {
        match self
            .dispatch(RuntimeSessionOperationSubmission::new(operation))
            .await?
        {
            RuntimeSessionOperationResult::Accepted { .. } => Ok(()),
            RuntimeSessionOperationResult::Submission(_)
            | RuntimeSessionOperationResult::Interrupted { .. } => {
                Err(RuntimeSessionLoopError::InvalidTask(
                    "runtime session response returned an invalid operation result".to_string(),
                ))
            }
        }
    }

    pub async fn interrupt(&self) -> Result<bool, RuntimeSessionLoopError> {
        self.interrupt_for_turn(None).await
    }

    pub async fn interrupt_for_turn(
        &self,
        expected_turn_id: Option<&str>,
    ) -> Result<bool, RuntimeSessionLoopError> {
        match self
            .dispatch(RuntimeSessionOperationSubmission::new(
                RuntimeSessionOperation::Interrupt {
                    expected_turn_id: expected_turn_id.map(str::to_string),
                },
            ))
            .await?
        {
            RuntimeSessionOperationResult::Interrupted { interrupted, .. } => Ok(interrupted),
            RuntimeSessionOperationResult::Submission(_)
            | RuntimeSessionOperationResult::Accepted { .. } => {
                Err(RuntimeSessionLoopError::InvalidTask(
                    "runtime session interrupt returned an invalid operation result".to_string(),
                ))
            }
        }
    }

    pub async fn shutdown(&self) -> Result<(), RuntimeSessionLoopError> {
        self.shutdown_and_wait().await
    }

    pub async fn shutdown_and_wait(&self) -> Result<(), RuntimeSessionLoopError> {
        let result = self
            .dispatch(RuntimeSessionOperationSubmission::new(
                RuntimeSessionOperation::Shutdown,
            ))
            .await;
        let operation_result = match result {
            Ok(RuntimeSessionOperationResult::Accepted { .. }) => Ok(()),
            Ok(RuntimeSessionOperationResult::Submission(_))
            | Ok(RuntimeSessionOperationResult::Interrupted { .. }) => {
                Err(RuntimeSessionLoopError::InvalidTask(
                    "runtime session shutdown returned an invalid operation result".to_string(),
                ))
            }
            Err(RuntimeSessionLoopError::Closed) => Ok(()),
            Err(error) => Err(error),
        };
        self.wait_for_termination().await;
        operation_result
    }

    async fn wait_for_termination(&self) {
        let mut termination = self.termination.clone();
        while !*termination.borrow() {
            if termination.changed().await.is_err() {
                break;
            }
        }
    }
}

fn validate_operation_submission(
    submission: &RuntimeSessionOperationSubmission,
) -> Result<(), RuntimeSessionLoopError> {
    if submission.id.trim().is_empty() {
        return Err(RuntimeSessionLoopError::InvalidTask(
            "runtime session operation requires a submission id".to_string(),
        ));
    }
    match &submission.operation {
        RuntimeSessionOperation::StartTask { task, .. } if task.turn_id().trim().is_empty() => {
            Err(RuntimeSessionLoopError::InvalidTask(
                "runtime session task requires a canonical turn_id".to_string(),
            ))
        }
        RuntimeSessionOperation::UserInput { input, .. } if input.is_empty() => {
            Err(RuntimeSessionLoopError::InvalidTask(
                "runtime session user input must not be empty".to_string(),
            ))
        }
        RuntimeSessionOperation::UserInput {
            task: Some(task), ..
        } if task.turn_id().trim().is_empty() => Err(RuntimeSessionLoopError::InvalidTask(
            "runtime session user input task requires a canonical turn_id".to_string(),
        )),
        RuntimeSessionOperation::UserInput {
            task: Some(task), ..
        } if task.kind() != super::RuntimeSessionTaskKind::Regular => {
            Err(RuntimeSessionLoopError::InvalidTask(
                "runtime session user input requires a regular task".to_string(),
            ))
        }
        RuntimeSessionOperation::Review { task, .. }
            if task.kind() != super::RuntimeSessionTaskKind::Review =>
        {
            Err(RuntimeSessionLoopError::InvalidTask(
                "runtime session review operation requires a review task".to_string(),
            ))
        }
        RuntimeSessionOperation::Compact { task, .. }
            if task.kind() != super::RuntimeSessionTaskKind::Compact =>
        {
            Err(RuntimeSessionLoopError::InvalidTask(
                "runtime session compact operation requires a compact task".to_string(),
            ))
        }
        RuntimeSessionOperation::Review { task, .. }
        | RuntimeSessionOperation::Compact { task, .. }
            if task.turn_id().trim().is_empty() =>
        {
            Err(RuntimeSessionLoopError::InvalidTask(
                "runtime session operation task requires a canonical turn_id".to_string(),
            ))
        }
        RuntimeSessionOperation::RunShell { task, .. }
            if task.kind() != super::RuntimeSessionTaskKind::RunShell =>
        {
            Err(RuntimeSessionLoopError::InvalidTask(
                "runtime session shell operation requires a shell task".to_string(),
            ))
        }
        RuntimeSessionOperation::RunShell { task, .. } if task.turn_id().trim().is_empty() => {
            Err(RuntimeSessionLoopError::InvalidTask(
                "runtime session shell operation requires a canonical turn_id".to_string(),
            ))
        }
        RuntimeSessionOperation::InterAgentCommunication { input }
            if input.message_id.trim().is_empty()
                || input.root_thread_id.trim().is_empty()
                || input.sender_thread_id.trim().is_empty()
                || input.recipient_thread_id.trim().is_empty() =>
        {
            Err(RuntimeSessionLoopError::InvalidTask(
                "runtime session inter-agent communication requires durable identity".to_string(),
            ))
        }
        RuntimeSessionOperation::ApprovalResponse { request_id, .. }
        | RuntimeSessionOperation::UserInputResponse { request_id, .. }
        | RuntimeSessionOperation::PermissionResponse { request_id, .. }
        | RuntimeSessionOperation::DynamicToolResponse { request_id, .. }
        | RuntimeSessionOperation::McpElicitationResponse { request_id, .. }
            if request_id.trim().is_empty() =>
        {
            Err(RuntimeSessionLoopError::InvalidTask(
                "runtime session response requires a request id".to_string(),
            ))
        }
        _ => Ok(()),
    }
}

pub(super) fn runtime_session_submission(
    id: &str,
    client_user_message_id: &Option<String>,
    trace: &Option<RuntimeSessionTraceContext>,
    result: RuntimeSessionSubmitResult,
    completion: oneshot::Receiver<Result<RuntimeSessionTaskOutcome, RuntimeSessionTaskFailure>>,
) -> RuntimeSessionSubmission {
    RuntimeSessionSubmission {
        id: id.to_string(),
        client_user_message_id: client_user_message_id.clone(),
        trace: trace.clone(),
        result,
        completion,
    }
}
