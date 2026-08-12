use super::status::agent_turn_is_active;
use super::{RuntimeCore, RuntimeCoreError, RuntimeEvent};
use agent_runtime::session_loop::{
    RuntimeSessionClosureTask, RuntimeSessionHandler, RuntimeSessionOperation,
    RuntimeSessionOperationResult, RuntimeSessionOperationSubmission, RuntimeSessionSubmitResult,
    RuntimeSessionTaskFailure, RuntimeSessionTaskKind,
};
use app_server_protocol::protocol::v2::{ThreadShellCommandParams, ThreadShellCommandResponse};
use app_server_protocol::{AgentTurn, AgentTurnStatus};
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tool_runtime::execution_process::{live::LiveExecutionRequest, ExecutionProcessStatus};
use uuid::Uuid;

const PROCESS_POLL_INTERVAL: Duration = Duration::from_millis(25);

#[derive(Debug)]
struct ShellProcessOutcome {
    output: String,
    exit_code: Option<i32>,
    elapsed_ms: u64,
    terminal: ShellTerminal,
}

#[derive(Clone, Copy, Debug)]
enum ShellTerminal {
    Completed,
    Failed,
    Canceled,
}

impl RuntimeCore {
    pub async fn run_thread_shell_command(
        &self,
        params: ThreadShellCommandParams,
    ) -> Result<ThreadShellCommandResponse, RuntimeCoreError> {
        let thread_id = required_value(&params.thread_id, "thread/shellCommand threadId")?;
        let command = required_value(&params.command, "thread/shellCommand command")?;
        if self.execution_process_server().is_none() {
            return Err(RuntimeCoreError::Backend(
                "local execution environment is not configured".to_string(),
            ));
        }
        let thread = self
            .read_thread(agent_protocol::thread::ThreadReadParams {
                thread_id: agent_protocol::ThreadId::new(thread_id.clone()),
                turns_view: agent_protocol::ThreadTurnsView::NotLoaded,
            })
            .await?;
        if thread.thread.archived {
            return Err(RuntimeCoreError::InvalidRequest(format!(
                "thread is archived: {thread_id}"
            )));
        }
        let session_id = thread.thread.session_id.as_str().to_string();
        self.ensure_current_session_hydrated(&session_id).await?;
        let cwd = shell_working_directory(&thread.thread)?;
        let standalone_turn_id = format!("turn_{}", Uuid::new_v4().simple());
        let item_id = format!("shell_{}", Uuid::new_v4().simple());
        let process_id = format!("shell-process-{}", Uuid::new_v4().simple());

        let auxiliary_runtime = self.clone();
        let auxiliary_session_id = session_id.clone();
        let auxiliary_thread_id = thread_id.clone();
        let auxiliary_command = command.clone();
        let auxiliary_cwd = cwd.clone();
        let auxiliary_item_id = item_id.clone();
        let auxiliary_process_id = process_id.clone();
        let auxiliary =
            RuntimeSessionHandler::new_with_cancellation(move |context, cancellation_token| {
                let runtime = auxiliary_runtime.clone();
                let session_id = auxiliary_session_id.clone();
                let thread_id = auxiliary_thread_id.clone();
                let command = auxiliary_command.clone();
                let cwd = auxiliary_cwd.clone();
                let item_id = auxiliary_item_id.clone();
                let process_id = auxiliary_process_id.clone();
                Box::pin(async move {
                    if context.session_id != session_id {
                        return Err(
                            "session actor identity changed during shell command".to_string()
                        );
                    }
                    let turn_id = context
                        .active_turn_id
                        .ok_or_else(|| "active shell command has no turn identity".to_string())?;
                    tokio::spawn(async move {
                        if let Err(error) = runtime
                            .execute_shell_on_turn(
                                &session_id,
                                &thread_id,
                                &turn_id,
                                &item_id,
                                &process_id,
                                &command,
                                &cwd,
                                cancellation_token,
                                false,
                            )
                            .await
                        {
                            tracing::warn!("active user shell command failed: {error}");
                        }
                    });
                    Ok(())
                })
            });

        let task_runtime = self.clone();
        let task_session_id = session_id.clone();
        let task_thread_id = thread_id.clone();
        let task_command = command.clone();
        let task_cwd = cwd.clone();
        let task_item_id = item_id.clone();
        let task_process_id = process_id.clone();
        let task = RuntimeSessionClosureTask::new(
            standalone_turn_id,
            Vec::new(),
            move |context, _input, cancellation_token| {
                let runtime = task_runtime.clone();
                let session_id = task_session_id.clone();
                let thread_id = task_thread_id.clone();
                let command = task_command.clone();
                let cwd = task_cwd.clone();
                let item_id = task_item_id.clone();
                let process_id = task_process_id.clone();
                Box::pin(async move {
                    runtime
                        .execute_shell_on_turn(
                            &session_id,
                            &thread_id,
                            context.turn_id(),
                            &item_id,
                            &process_id,
                            &command,
                            &cwd,
                            cancellation_token,
                            true,
                        )
                        .await
                        .map_err(|error| RuntimeSessionTaskFailure {
                            message: error.to_string(),
                            reason_code: None,
                        })
                })
            },
        )
        .with_kind(RuntimeSessionTaskKind::RunShell);

        let session = self
            .session_loops
            .get_or_create(&session_id, &thread_id)
            .await
            .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;
        let result = session
            .dispatch(RuntimeSessionOperationSubmission::new(
                RuntimeSessionOperation::RunShell {
                    auxiliary,
                    task: Arc::new(task),
                },
            ))
            .await
            .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;
        match result {
            RuntimeSessionOperationResult::Accepted {
                turn_id: Some(_), ..
            } => {}
            RuntimeSessionOperationResult::Submission(submission)
                if matches!(submission.result, RuntimeSessionSubmitResult::Started) => {}
            RuntimeSessionOperationResult::Submission(submission) => {
                return Err(RuntimeCoreError::Backend(format!(
                    "shell operation was not started: {:?}",
                    submission.result
                )));
            }
            _ => {
                return Err(RuntimeCoreError::Backend(
                    "shell operation returned an invalid receipt".to_string(),
                ));
            }
        }
        Ok(ThreadShellCommandResponse {})
    }

    #[allow(clippy::too_many_arguments)]
    async fn execute_shell_on_turn(
        &self,
        session_id: &str,
        thread_id: &str,
        turn_id: &str,
        item_id: &str,
        process_id: &str,
        command: &str,
        cwd: &Path,
        cancellation_token: CancellationToken,
        standalone: bool,
    ) -> Result<(), RuntimeCoreError> {
        if standalone {
            self.create_shell_turn(session_id, thread_id, turn_id)?;
            self.append_and_publish_shell_events(
                session_id,
                thread_id,
                turn_id,
                vec![
                    RuntimeEvent::new("turn.accepted", json!({"source": "thread/shellCommand"})),
                    RuntimeEvent::new("turn.started", json!({"source": "thread/shellCommand"})),
                ],
            )?;
        }

        let server = match self
            .start_shell_process(thread_id, process_id, item_id, command, cwd)
            .await
        {
            Ok(server) => server,
            Err(error) => {
                let outcome = ShellProcessOutcome {
                    output: error.clone(),
                    exit_code: Some(-1),
                    elapsed_ms: 0,
                    terminal: ShellTerminal::Failed,
                };
                let mut events = vec![
                    RuntimeEvent::new(
                        "command.started",
                        shell_event_payload(item_id, process_id, command, cwd, None),
                    ),
                    RuntimeEvent::new(
                        "command.exited",
                        shell_event_payload(item_id, process_id, command, cwd, Some(&outcome)),
                    ),
                ];
                if standalone {
                    events.push(RuntimeEvent::new(
                        "turn.failed",
                        json!({"source": "thread/shellCommand", "error": error}),
                    ));
                }
                self.append_and_publish_shell_events(session_id, thread_id, turn_id, events)?;
                return Err(RuntimeCoreError::Backend(error));
            }
        };

        self.append_and_publish_shell_events(
            session_id,
            thread_id,
            turn_id,
            vec![RuntimeEvent::new(
                "command.started",
                shell_event_payload(item_id, process_id, command, cwd, None),
            )],
        )?;

        let outcome = self
            .poll_shell_process(server, process_id, cancellation_token)
            .await;
        match outcome {
            Ok(outcome) => {
                let mut events = vec![RuntimeEvent::new(
                    "command.exited",
                    shell_event_payload(item_id, process_id, command, cwd, Some(&outcome)),
                )];
                if standalone {
                    events.push(RuntimeEvent::new(
                        match outcome.terminal {
                            ShellTerminal::Completed => "turn.completed",
                            ShellTerminal::Failed => "turn.failed",
                            ShellTerminal::Canceled => "turn.canceled",
                        },
                        json!({"source": "thread/shellCommand"}),
                    ));
                }
                self.append_and_publish_shell_events(session_id, thread_id, turn_id, events)?;
                Ok(())
            }
            Err(error) => {
                let outcome = ShellProcessOutcome {
                    output: error.clone(),
                    exit_code: Some(-1),
                    elapsed_ms: 0,
                    terminal: ShellTerminal::Failed,
                };
                let mut events = vec![RuntimeEvent::new(
                    "command.exited",
                    shell_event_payload(item_id, process_id, command, cwd, Some(&outcome)),
                )];
                if standalone {
                    events.push(RuntimeEvent::new(
                        "turn.failed",
                        json!({"source": "thread/shellCommand", "error": error}),
                    ));
                }
                self.append_and_publish_shell_events(session_id, thread_id, turn_id, events)?;
                Err(RuntimeCoreError::Backend(error))
            }
        }
    }

    fn create_shell_turn(
        &self,
        session_id: &str,
        thread_id: &str,
        turn_id: &str,
    ) -> Result<(), RuntimeCoreError> {
        let mut state = self
            .state
            .lock()
            .expect("runtime core state mutex poisoned");
        let stored = state
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.to_string()))?;
        if stored.session.thread_id != thread_id {
            return Err(RuntimeCoreError::InvalidRequest(
                "session/thread identity mismatch for shell command".to_string(),
            ));
        }
        if stored
            .turns
            .iter()
            .any(|turn| agent_turn_is_active(turn.status))
        {
            return Err(RuntimeCoreError::InvalidRequest(
                "session actor and runtime turn state disagree for shell command".to_string(),
            ));
        }
        let now = super::timestamp();
        stored.session.status = app_server_protocol::AgentSessionStatus::Running;
        stored.session.updated_at = now.clone();
        stored.turns.push(AgentTurn {
            turn_id: turn_id.to_string(),
            session_id: session_id.to_string(),
            thread_id: thread_id.to_string(),
            status: AgentTurnStatus::Accepted,
            started_at: Some(now),
            completed_at: None,
        });
        Ok(())
    }

    async fn start_shell_process(
        &self,
        thread_id: &str,
        process_id: &str,
        item_id: &str,
        command: &str,
        cwd: &Path,
    ) -> Result<crate::execution_process::ExecutionProcessServer, String> {
        let server = self
            .execution_process_server()
            .ok_or_else(|| "local execution environment is not configured".to_string())?;
        server
            .start_thread_process(
                thread_id,
                command,
                LiveExecutionRequest {
                    process_id: process_id.to_string(),
                    tool_id: item_id.to_string(),
                    tool_name: tool_runtime::unified_exec::EXEC_COMMAND_TOOL_NAME.to_string(),
                    command: tool_runtime::shell_runtime::platform_shell_argv(command),
                    working_directory: cwd.to_path_buf(),
                    tty: false,
                    approval_policy: Some("never".to_string()),
                    sandbox_policy: Some("danger-full-access".to_string()),
                    runtime_metadata: Some(json!({
                        "surface": "user_shell",
                        "explicitUserCommand": true,
                    })),
                    env: HashMap::new(),
                },
            )
            .await
            .map_err(|error| error.to_string())?;
        Ok(server)
    }

    async fn poll_shell_process(
        &self,
        server: crate::execution_process::ExecutionProcessServer,
        process_id: &str,
        cancellation_token: CancellationToken,
    ) -> Result<ShellProcessOutcome, String> {
        let mut canceled = false;
        loop {
            if cancellation_token.is_cancelled() && !canceled {
                canceled = true;
                let _ = server.terminate(process_id);
            }
            let snapshot = server
                .status(process_id)
                .map_err(|error| error.to_string())?;
            if matches!(
                snapshot.status,
                ExecutionProcessStatus::Exited
                    | ExecutionProcessStatus::Interrupted
                    | ExecutionProcessStatus::Terminated
                    | ExecutionProcessStatus::Failed
            ) {
                let terminal = if canceled
                    || matches!(
                        snapshot.status,
                        ExecutionProcessStatus::Interrupted | ExecutionProcessStatus::Terminated
                    ) {
                    ShellTerminal::Canceled
                } else if snapshot.status == ExecutionProcessStatus::Exited
                    && snapshot.exit_code == Some(0)
                {
                    ShellTerminal::Completed
                } else {
                    ShellTerminal::Failed
                };
                return Ok(ShellProcessOutcome {
                    output: snapshot.retained_output,
                    exit_code: snapshot.exit_code,
                    elapsed_ms: snapshot.elapsed_ms,
                    terminal,
                });
            }
            tokio::time::sleep(PROCESS_POLL_INTERVAL).await;
        }
    }

    fn append_and_publish_shell_events(
        &self,
        session_id: &str,
        thread_id: &str,
        turn_id: &str,
        events: Vec<RuntimeEvent>,
    ) -> Result<(), RuntimeCoreError> {
        let events = self.append_runtime_events(session_id, thread_id, Some(turn_id), events)?;
        for event in events {
            self.event_hub.publish(event);
        }
        Ok(())
    }
}

fn required_value(value: &str, field: &str) -> Result<String, RuntimeCoreError> {
    let value = value.trim();
    if value.is_empty() {
        Err(RuntimeCoreError::InvalidRequest(format!(
            "{field} must not be empty"
        )))
    } else {
        Ok(value.to_string())
    }
}

fn shell_working_directory(thread: &agent_protocol::Thread) -> Result<PathBuf, RuntimeCoreError> {
    let configured = ["workingDir", "cwd"]
        .into_iter()
        .find_map(|key| thread.metadata.get(key).and_then(serde_json::Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let path = match configured {
        Some(value) => PathBuf::from(value),
        None => std::env::current_dir().map_err(|error| {
            RuntimeCoreError::Backend(format!("cannot resolve shell working directory: {error}"))
        })?,
    };
    if !path.is_absolute() {
        return Err(RuntimeCoreError::InvalidRequest(
            "thread shell working directory must be absolute".to_string(),
        ));
    }
    std::fs::canonicalize(&path).map_err(|error| {
        RuntimeCoreError::InvalidRequest(format!(
            "thread shell working directory is unavailable: {}: {error}",
            path.display()
        ))
    })
}

fn shell_event_payload(
    item_id: &str,
    process_id: &str,
    command: &str,
    cwd: &Path,
    outcome: Option<&ShellProcessOutcome>,
) -> serde_json::Value {
    let exit_code = outcome.and_then(|outcome| {
        outcome
            .exit_code
            .or_else(|| (!matches!(outcome.terminal, ShellTerminal::Completed)).then_some(-1))
    });
    json!({
        "commandId": item_id,
        "itemId": item_id,
        "processId": process_id,
        "command": command,
        "cwd": cwd.to_string_lossy(),
        "output": outcome.map(|outcome| outcome.output.as_str()),
        "exitCode": exit_code,
        "durationMs": outcome.map(|outcome| outcome.elapsed_ms),
        "source": "user_shell",
        "metadata": {
            "commandExecutionSource": "userShell",
            "processId": process_id,
            "durationMs": outcome.map(|outcome| outcome.elapsed_ms),
        },
    })
}
