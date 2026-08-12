use super::input_queue::{
    PendingInputQueue, QueuedTask, RuntimeSessionTask, RuntimeSessionTaskContext,
    RuntimeSessionTaskMetadata, RuntimeSessionTaskOutcome, RuntimeSessionTaskState,
};
use super::resources::RuntimeSessionResources;
use super::{
    RuntimeSessionHandle, RuntimeSessionLoopError, RuntimeSessionOperation,
    RuntimeSessionOperationContext, RuntimeSessionOperationResult,
    RuntimeSessionOperationSubmission, RuntimeSessionResponseKind, RuntimeSessionSnapshot,
    RuntimeSessionSubmitResult, RuntimeSessionTaskFailure,
};
use crate::code_mode::RuntimeCodeModeServiceFactory;
use crate::session_loop::handle::runtime_session_submission;
use futures::FutureExt;
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use tokio::time::{timeout_at, Duration, Instant};
use tokio_util::sync::CancellationToken;

const SESSION_COMMAND_BUFFER: usize = 512;
const TASK_ABORT_GRACE: Duration = Duration::from_millis(100);

struct ActiveTask {
    key: u64,
    task: Arc<dyn RuntimeSessionTask>,
    context: RuntimeSessionTaskContext,
    cancellation_token: CancellationToken,
    completion: oneshot::Sender<Result<RuntimeSessionTaskOutcome, RuntimeSessionTaskFailure>>,
    join: JoinHandle<()>,
}

pub(super) enum RuntimeSessionCommand {
    Operation {
        submission: RuntimeSessionOperationSubmission,
        reply: oneshot::Sender<Result<RuntimeSessionOperationResult, RuntimeSessionLoopError>>,
    },
    SubscribeInputActivity {
        reply: oneshot::Sender<(
            watch::Receiver<super::RuntimeSessionInputActivity>,
            Option<super::RuntimeSessionInputActivity>,
        )>,
    },
    Snapshot {
        reply: oneshot::Sender<RuntimeSessionSnapshot>,
    },
    TaskFinished {
        task_key: u64,
        result: Result<(), RuntimeSessionTaskFailure>,
    },
}

struct TaskFinishedMessage {
    task_key: u64,
    result: Result<(), RuntimeSessionTaskFailure>,
}

pub(super) struct RuntimeSessionActor;

impl RuntimeSessionActor {
    pub(super) fn spawn(
        session_id: String,
        thread_id: String,
        code_mode_factory: Option<&RuntimeCodeModeServiceFactory>,
    ) -> RuntimeSessionHandle {
        let (tx, rx) = mpsc::channel(SESSION_COMMAND_BUFFER);
        let (finished_tx, finished_rx) = mpsc::unbounded_channel();
        let (termination_tx, termination) = watch::channel(false);
        let resources = Arc::new(RuntimeSessionResources::new(thread_id, code_mode_factory));
        tokio::spawn(run_session_loop(
            session_id,
            Arc::clone(&resources),
            rx,
            finished_rx,
            finished_tx,
            termination_tx,
        ));
        RuntimeSessionHandle {
            tx,
            termination,
            resources,
        }
    }
}

async fn run_session_loop(
    session_id: String,
    resources: Arc<RuntimeSessionResources>,
    mut rx: mpsc::Receiver<RuntimeSessionCommand>,
    mut finished_rx: mpsc::UnboundedReceiver<TaskFinishedMessage>,
    finished_tx: mpsc::UnboundedSender<TaskFinishedMessage>,
    termination_tx: watch::Sender<bool>,
) {
    let session_id: Arc<str> = Arc::from(session_id);
    let pending_input = Arc::new(PendingInputQueue::default());
    let mut active: Option<ActiveTask> = None;
    let mut queued = VecDeque::new();
    let mut next_task_key = 1_u64;

    loop {
        let command = tokio::select! {
            biased;
            finished = finished_rx.recv() => finished.map(|finished| RuntimeSessionCommand::TaskFinished {
                task_key: finished.task_key,
                result: finished.result,
            }),
            command = rx.recv() => command,
        };
        let Some(command) = command else {
            break;
        };
        match command {
            RuntimeSessionCommand::Snapshot { reply } => {
                let _ = reply.send(RuntimeSessionSnapshot {
                    active_turn_id: active
                        .as_ref()
                        .map(|active_task| active_task.task.turn_id().to_string()),
                });
            }
            RuntimeSessionCommand::SubscribeInputActivity { reply } => {
                let (activity, mut pending_activity) = match active.as_ref() {
                    Some(active_task) => active_task.context.subscribe_activity().await,
                    None => pending_input.subscribe_activity_snapshot().await,
                };
                if pending_activity.is_none() && !queued.is_empty() {
                    pending_activity = Some(super::RuntimeSessionInputActivity::Steer);
                }
                let _ = reply.send((activity, pending_activity));
            }
            RuntimeSessionCommand::Operation { submission, reply } => {
                let RuntimeSessionOperationSubmission {
                    id,
                    operation,
                    client_user_message_id,
                    trace,
                } = submission;
                match operation {
                    RuntimeSessionOperation::StartTask {
                        task,
                        queue_if_busy,
                        replace_active,
                    } => {
                        let input = task.initial_input();
                        let submission = submit_task(
                            &id,
                            &client_user_message_id,
                            &trace,
                            task,
                            input,
                            queue_if_busy,
                            replace_active,
                            &mut active,
                            &mut queued,
                            Arc::clone(&session_id),
                            Arc::clone(&resources),
                            Arc::clone(&pending_input),
                            finished_tx.clone(),
                            &mut next_task_key,
                        )
                        .await;
                        if matches!(submission.result, RuntimeSessionSubmitResult::Queued { .. }) {
                            pending_input.publish_steer_activity();
                        }
                        let _ =
                            reply.send(Ok(RuntimeSessionOperationResult::Submission(submission)));
                    }
                    RuntimeSessionOperation::UserInput {
                        expected_turn_id,
                        input,
                        task,
                        queue_if_busy,
                    } => {
                        if let Some(active_task) = active.as_ref() {
                            if active_task.task.kind().accepts_steer() {
                                let active_turn_id = active_task.task.turn_id().to_string();
                                if expected_turn_id
                                    .as_deref()
                                    .is_some_and(|expected| expected != active_turn_id)
                                {
                                    let _ = reply.send(Err(RuntimeSessionLoopError::InvalidTask(
                                        "runtime session user input target is no longer active"
                                            .to_string(),
                                    )));
                                    continue;
                                }
                                if !active_task.context.input_handle().push_steer(input).await {
                                    let _ = reply.send(Err(RuntimeSessionLoopError::InvalidTask(
                                        "runtime session turn is already finishing".to_string(),
                                    )));
                                    continue;
                                }
                                let _ = reply.send(Ok(RuntimeSessionOperationResult::Accepted {
                                    id,
                                    turn_id: Some(active_turn_id),
                                }));
                                continue;
                            }
                        }
                        if expected_turn_id.is_some() {
                            let _ = reply.send(Err(RuntimeSessionLoopError::InvalidTask(
                                "runtime session has no matching active turn for user input"
                                    .to_string(),
                            )));
                            continue;
                        }
                        let Some(task) = task else {
                            let message = if active.is_some() {
                                "runtime session task does not accept user input"
                            } else {
                                "runtime session has no active turn for user input"
                            };
                            let _ = reply.send(Err(RuntimeSessionLoopError::InvalidTask(
                                message.to_string(),
                            )));
                            continue;
                        };
                        let submission = submit_task(
                            &id,
                            &client_user_message_id,
                            &trace,
                            task,
                            input,
                            queue_if_busy,
                            false,
                            &mut active,
                            &mut queued,
                            Arc::clone(&session_id),
                            Arc::clone(&resources),
                            Arc::clone(&pending_input),
                            finished_tx.clone(),
                            &mut next_task_key,
                        )
                        .await;
                        let _ =
                            reply.send(Ok(RuntimeSessionOperationResult::Submission(submission)));
                    }
                    RuntimeSessionOperation::Review { task }
                    | RuntimeSessionOperation::Compact { task } => {
                        let input = task.initial_input();
                        let submission = submit_task(
                            &id,
                            &client_user_message_id,
                            &trace,
                            task,
                            input,
                            false,
                            true,
                            &mut active,
                            &mut queued,
                            Arc::clone(&session_id),
                            Arc::clone(&resources),
                            Arc::clone(&pending_input),
                            finished_tx.clone(),
                            &mut next_task_key,
                        )
                        .await;
                        let _ =
                            reply.send(Ok(RuntimeSessionOperationResult::Submission(submission)));
                    }
                    RuntimeSessionOperation::ThreadSettings { handler }
                    | RuntimeSessionOperation::SetMemoryMode { handler }
                    | RuntimeSessionOperation::RefreshMcp { handler }
                    | RuntimeSessionOperation::ReloadConfig { handler } => {
                        let context = RuntimeSessionOperationContext {
                            session_id: session_id.to_string(),
                            thread_id: resources.thread_id().to_string(),
                            submission_id: id.clone(),
                            active_turn_id: active
                                .as_ref()
                                .map(|active| active.task.turn_id().to_string()),
                            client_user_message_id,
                            trace,
                        };
                        let result = handler
                            .execute(context)
                            .await
                            .map_err(RuntimeSessionLoopError::OperationFailed);
                        let _ = reply.send(result.map(|()| {
                            RuntimeSessionOperationResult::Accepted { id, turn_id: None }
                        }));
                    }
                    RuntimeSessionOperation::RunShell { auxiliary, task } => {
                        if let Some(active_task) = active.as_ref() {
                            let turn_id = active_task.task.turn_id().to_string();
                            let context = RuntimeSessionOperationContext {
                                session_id: session_id.to_string(),
                                thread_id: resources.thread_id().to_string(),
                                submission_id: id.clone(),
                                active_turn_id: Some(turn_id.clone()),
                                client_user_message_id,
                                trace,
                            };
                            let result = auxiliary
                                .execute_with_cancellation(
                                    context,
                                    active_task.cancellation_token.clone(),
                                )
                                .await
                                .map_err(RuntimeSessionLoopError::OperationFailed);
                            let _ = reply.send(result.map(|()| {
                                RuntimeSessionOperationResult::Accepted {
                                    id,
                                    turn_id: Some(turn_id),
                                }
                            }));
                        } else {
                            let input = task.initial_input();
                            let submission = submit_task(
                                &id,
                                &client_user_message_id,
                                &trace,
                                task,
                                input,
                                false,
                                false,
                                &mut active,
                                &mut queued,
                                Arc::clone(&session_id),
                                Arc::clone(&resources),
                                Arc::clone(&pending_input),
                                finished_tx.clone(),
                                &mut next_task_key,
                            )
                            .await;
                            let _ = reply
                                .send(Ok(RuntimeSessionOperationResult::Submission(submission)));
                        }
                    }
                    RuntimeSessionOperation::InterAgentCommunication { input: _ } => {
                        let _ = (client_user_message_id, trace);
                        pending_input.notify_mailbox_activity().await;
                        let _ = reply.send(Ok(RuntimeSessionOperationResult::Accepted {
                            id,
                            turn_id: None,
                        }));
                    }
                    RuntimeSessionOperation::ApprovalResponse {
                        expected_turn_id,
                        request_id,
                        response,
                    } => {
                        let _ = (client_user_message_id, trace);
                        let result = resolve_response_operation(
                            active.as_ref(),
                            expected_turn_id.as_deref(),
                            RuntimeSessionResponseKind::Approval,
                            &request_id,
                            response,
                        )
                        .await;
                        let _ = reply.send(result.map(|()| {
                            RuntimeSessionOperationResult::Accepted { id, turn_id: None }
                        }));
                    }
                    RuntimeSessionOperation::UserInputResponse {
                        expected_turn_id,
                        request_id,
                        response,
                    } => {
                        let _ = (client_user_message_id, trace);
                        let result = resolve_response_operation(
                            active.as_ref(),
                            expected_turn_id.as_deref(),
                            RuntimeSessionResponseKind::AskUser,
                            &request_id,
                            response,
                        )
                        .await;
                        let _ = reply.send(result.map(|()| {
                            RuntimeSessionOperationResult::Accepted { id, turn_id: None }
                        }));
                    }
                    RuntimeSessionOperation::PermissionResponse {
                        expected_turn_id,
                        request_id,
                        response,
                    } => {
                        let _ = (client_user_message_id, trace);
                        let result = resolve_response_operation(
                            active.as_ref(),
                            expected_turn_id.as_deref(),
                            RuntimeSessionResponseKind::Permission,
                            &request_id,
                            response,
                        )
                        .await;
                        let _ = reply.send(result.map(|()| {
                            RuntimeSessionOperationResult::Accepted { id, turn_id: None }
                        }));
                    }
                    RuntimeSessionOperation::DynamicToolResponse {
                        expected_turn_id,
                        request_id,
                        response,
                    } => {
                        let _ = (client_user_message_id, trace);
                        let result = resolve_response_operation(
                            active.as_ref(),
                            expected_turn_id.as_deref(),
                            RuntimeSessionResponseKind::DynamicTool,
                            &request_id,
                            response,
                        )
                        .await;
                        let _ = reply.send(result.map(|()| {
                            RuntimeSessionOperationResult::Accepted { id, turn_id: None }
                        }));
                    }
                    RuntimeSessionOperation::McpElicitationResponse {
                        expected_turn_id,
                        request_id,
                        response,
                    } => {
                        let _ = (client_user_message_id, trace);
                        let result = resolve_response_operation(
                            active.as_ref(),
                            expected_turn_id.as_deref(),
                            RuntimeSessionResponseKind::McpElicitation,
                            &request_id,
                            response,
                        )
                        .await;
                        let _ = reply.send(result.map(|()| {
                            RuntimeSessionOperationResult::Accepted { id, turn_id: None }
                        }));
                    }
                    RuntimeSessionOperation::Interrupt { expected_turn_id } => {
                        let _ = (client_user_message_id, trace);
                        if let Some(expected_turn_id) = expected_turn_id.as_deref() {
                            let active_turn_id = active.as_ref().map(|task| task.task.turn_id());
                            if active_turn_id != Some(expected_turn_id) {
                                let _ =
                                    reply.send(Ok(RuntimeSessionOperationResult::Interrupted {
                                        id,
                                        interrupted: false,
                                    }));
                                continue;
                            }
                        }
                        let interrupted = if let Some(active_task) = active.take() {
                            resources.interrupt_code_mode().await;
                            stop_active_task(active_task, RuntimeSessionTaskOutcome::Interrupted)
                                .await;
                            true
                        } else {
                            false
                        };
                        if interrupted {
                            start_next_task(
                                &mut active,
                                &mut queued,
                                Arc::clone(&session_id),
                                Arc::clone(&resources),
                                Arc::clone(&pending_input),
                                finished_tx.clone(),
                                &mut next_task_key,
                            );
                        }
                        let _ = reply.send(Ok(RuntimeSessionOperationResult::Interrupted {
                            id,
                            interrupted,
                        }));
                    }
                    RuntimeSessionOperation::Shutdown => {
                        let _ = (client_user_message_id, trace);
                        if let Some(active_task) = active.take() {
                            stop_active_task(active_task, RuntimeSessionTaskOutcome::Shutdown)
                                .await;
                        }
                        while let Some(task) = queued.pop_front() {
                            let _ = task
                                .completion
                                .send(Ok(RuntimeSessionTaskOutcome::Shutdown));
                        }
                        pending_input.clear().await;
                        let result = resources
                            .shutdown_code_mode()
                            .await
                            .map_err(RuntimeSessionLoopError::OperationFailed)
                            .map(|()| RuntimeSessionOperationResult::Accepted {
                                id,
                                turn_id: None,
                            });
                        let _ = reply.send(result);
                        break;
                    }
                }
            }
            RuntimeSessionCommand::TaskFinished { task_key, result } => {
                let Some(active_task) = active.take() else {
                    continue;
                };
                if active_task.key != task_key {
                    active = Some(active_task);
                    continue;
                }
                active_task.context.input_handle().clear_turn_state().await;
                let outcome = match result {
                    Ok(()) => Ok(RuntimeSessionTaskOutcome::Completed),
                    Err(error) => Err(error),
                };
                let _ = active_task.completion.send(outcome);
                start_next_task(
                    &mut active,
                    &mut queued,
                    Arc::clone(&session_id),
                    Arc::clone(&resources),
                    Arc::clone(&pending_input),
                    finished_tx.clone(),
                    &mut next_task_key,
                );
            }
        }
    }

    // A dropped command channel is equivalent to shutdown. Do not leave an in-flight
    // provider task or queued completion receiver hanging forever.
    if let Some(active_task) = active.take() {
        stop_active_task(active_task, RuntimeSessionTaskOutcome::Shutdown).await;
    }
    while let Some(task) = queued.pop_front() {
        let _ = task
            .completion
            .send(Ok(RuntimeSessionTaskOutcome::Shutdown));
    }
    pending_input.clear().await;
    let _ = resources.shutdown_code_mode().await;
    termination_tx.send_replace(true);
}

async fn resolve_response_operation(
    active: Option<&ActiveTask>,
    expected_turn_id: Option<&str>,
    kind: RuntimeSessionResponseKind,
    request_id: &str,
    response: Value,
) -> Result<(), RuntimeSessionLoopError> {
    let Some(active_task) = active else {
        return Err(RuntimeSessionLoopError::InvalidTask(
            "runtime session has no active turn for response".to_string(),
        ));
    };
    if expected_turn_id.is_some_and(|expected| expected != active_task.task.turn_id()) {
        return Err(RuntimeSessionLoopError::InvalidTask(
            "runtime session response target is no longer active".to_string(),
        ));
    }
    active_task
        .context
        .input_handle()
        .resolve_response(kind, request_id, response)
        .await
}

async fn stop_active_task(active_task: ActiveTask, outcome: RuntimeSessionTaskOutcome) {
    let ActiveTask {
        task,
        context,
        cancellation_token,
        completion,
        mut join,
        ..
    } = active_task;
    cancellation_token.cancel();
    let deadline = Instant::now() + TASK_ABORT_GRACE;
    if timeout_at(deadline, &mut join).await.is_err() {
        join.abort();
        let _ = join.await;
    }
    let _ = tokio::time::timeout(TASK_ABORT_GRACE, task.abort(context.clone())).await;
    context.input_handle().clear_turn_state().await;
    let _ = completion.send(Ok(outcome));
}

#[allow(clippy::too_many_arguments)]
async fn submit_task(
    id: &str,
    client_user_message_id: &Option<String>,
    trace: &Option<super::RuntimeSessionTraceContext>,
    task: Arc<dyn RuntimeSessionTask>,
    input: Vec<super::RuntimeSessionInput>,
    queue_if_busy: bool,
    replace_active: bool,
    active: &mut Option<ActiveTask>,
    queued: &mut VecDeque<QueuedTask>,
    session_id: Arc<str>,
    resources: Arc<RuntimeSessionResources>,
    pending_input: Arc<PendingInputQueue>,
    finished_tx: mpsc::UnboundedSender<TaskFinishedMessage>,
    next_task_key: &mut u64,
) -> super::RuntimeSessionSubmission {
    let metadata = RuntimeSessionTaskMetadata::new(
        id.to_string(),
        client_user_message_id.clone(),
        trace.clone(),
    );
    let (completion_tx, completion_rx) = oneshot::channel();
    if replace_active {
        if let Some(active_task) = active.take() {
            resources.interrupt_code_mode().await;
            stop_active_task(active_task, RuntimeSessionTaskOutcome::Replaced).await;
        }
    } else if active.is_some() {
        if !queue_if_busy {
            let _ = completion_tx.send(Err(RuntimeSessionTaskFailure {
                message: "runtime session is busy".to_string(),
                ..Default::default()
            }));
            return runtime_session_submission(
                id,
                client_user_message_id,
                trace,
                RuntimeSessionSubmitResult::Busy,
                completion_rx,
            );
        }
        queued.push_back(QueuedTask {
            task,
            input,
            completion: completion_tx,
            metadata,
        });
        return runtime_session_submission(
            id,
            client_user_message_id,
            trace,
            RuntimeSessionSubmitResult::Queued {
                position: queued.len(),
            },
            completion_rx,
        );
    }

    *active = Some(spawn_task(
        session_id,
        resources,
        pending_input,
        task,
        input,
        metadata,
        completion_tx,
        finished_tx,
        *next_task_key,
    ));
    *next_task_key = next_task_key.saturating_add(1);
    runtime_session_submission(
        id,
        client_user_message_id,
        trace,
        RuntimeSessionSubmitResult::Started,
        completion_rx,
    )
}

fn spawn_task(
    session_id: Arc<str>,
    resources: Arc<RuntimeSessionResources>,
    pending_input: Arc<PendingInputQueue>,
    task: Arc<dyn RuntimeSessionTask>,
    initial_input: Vec<super::RuntimeSessionInput>,
    metadata: RuntimeSessionTaskMetadata,
    completion: oneshot::Sender<Result<RuntimeSessionTaskOutcome, RuntimeSessionTaskFailure>>,
    finished_tx: mpsc::UnboundedSender<TaskFinishedMessage>,
    task_key: u64,
) -> ActiveTask {
    let turn_id: Arc<str> = Arc::from(task.turn_id().to_string());
    let state = Arc::new(RuntimeSessionTaskState::default());
    let context = RuntimeSessionTaskContext::new(
        session_id,
        turn_id,
        task.kind(),
        metadata,
        Arc::clone(&pending_input),
        task.mailbox_loader(),
        resources,
        state,
    );
    let cancellation_token = CancellationToken::new();
    let task_for_run = Arc::clone(&task);
    let task_key_for_run = task_key;
    let token_for_run = cancellation_token.clone();
    let context_for_run = context.clone();
    let join = tokio::spawn(async move {
        let result = std::panic::AssertUnwindSafe(task_for_run.run(
            context_for_run,
            initial_input,
            token_for_run,
        ))
        .catch_unwind()
        .await
        .unwrap_or_else(|_| {
            Err(RuntimeSessionTaskFailure {
                message: "runtime session task panicked".to_string(),
                ..Default::default()
            })
        });
        let _ = finished_tx.send(TaskFinishedMessage {
            task_key: task_key_for_run,
            result,
        });
    });
    ActiveTask {
        key: task_key,
        task,
        context,
        cancellation_token,
        completion,
        join,
    }
}

fn start_next_task(
    active: &mut Option<ActiveTask>,
    queued: &mut VecDeque<QueuedTask>,
    session_id: Arc<str>,
    resources: Arc<RuntimeSessionResources>,
    pending_input: Arc<PendingInputQueue>,
    finished_tx: mpsc::UnboundedSender<TaskFinishedMessage>,
    next_task_key: &mut u64,
) {
    let Some(task) = queued.pop_front() else {
        return;
    };
    *active = Some(spawn_task(
        session_id,
        resources,
        pending_input,
        task.task,
        task.input,
        task.metadata,
        task.completion,
        finished_tx,
        *next_task_key,
    ));
    *next_task_key = next_task_key.saturating_add(1);
}
