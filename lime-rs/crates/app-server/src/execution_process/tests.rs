use super::*;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use tool_runtime::execution_orchestrator::{
    orchestrate_runtime_tool_execution, RuntimeToolApprovalFuture, RuntimeToolApprovalHandler,
    RuntimeToolApprovalKind, RuntimeToolApprovalPolicy, RuntimeToolApprovalRequest,
    RuntimeToolAttemptFuture, RuntimeToolAttemptRunner, RuntimeToolExecutionAttempt,
    RuntimeToolInitialApproval, RuntimeToolOrchestrationInput, RuntimeToolSandboxPolicy,
};
use tool_runtime::execution_process::{
    live::{
        LiveExecutionOutputBatch, LiveExecutionOutputQuery, LiveExecutionRequest,
        RuntimeLiveExecutionGateway,
    },
    ExecutionProcessStatus,
};
use tool_runtime::tool_executor::{RuntimeToolExecutionIdentity, RuntimeToolExecutionResult};

#[derive(Default)]
struct RecordingProcessApprovals {
    calls: AtomicUsize,
}

impl RuntimeToolApprovalHandler for RecordingProcessApprovals {
    fn approve<'a>(
        &'a self,
        _request: RuntimeToolApprovalRequest,
    ) -> RuntimeToolApprovalFuture<'a> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
    }
}

struct ProcessAttemptRunner {
    server: ExecutionProcessServer,
    calls: AtomicUsize,
    runtime_metadata: Option<Value>,
}

impl RuntimeToolAttemptRunner for ProcessAttemptRunner {
    fn run<'a>(&'a self, attempt: RuntimeToolExecutionAttempt) -> RuntimeToolAttemptFuture<'a> {
        Box::pin(async move {
            let attempt_number = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            let process_id = format!("orchestrated-process-{attempt_number}");
            let snapshot = RuntimeLiveExecutionGateway::start_process(
                &self.server,
                "test-thread",
                "printf orchestrated",
                LiveExecutionRequest {
                    process_id,
                    tool_id: attempt.identity().call_id().to_string(),
                    tool_name: "exec_command".to_string(),
                    command: shell_output_command("orchestrated"),
                    working_directory: current_directory(),
                    tty: false,
                    approval_policy: Some("on-request".to_string()),
                    sandbox_policy: attempt
                        .effective_sandbox_policy()
                        .label()
                        .map(str::to_string),
                    runtime_metadata: self.runtime_metadata.clone(),
                    env: HashMap::new(),
                    attempt: Some(attempt),
                },
            )
            .await?;
            Ok(RuntimeToolExecutionResult::new(
                true,
                snapshot.process_id,
                None,
                HashMap::new(),
            ))
        })
    }
}

fn process_orchestration_input(
    initial_approval: RuntimeToolInitialApproval,
    sandbox_policy: RuntimeToolSandboxPolicy,
    managed_network_host: Option<String>,
    network_denial_retry_allowed: bool,
) -> RuntimeToolOrchestrationInput {
    RuntimeToolOrchestrationInput {
        identity: RuntimeToolExecutionIdentity::new("orchestrated-call", "orchestrated-turn"),
        approval_policy: RuntimeToolApprovalPolicy::OnRequest,
        initial_approval,
        initial_approval_reason: Some("approval required".to_string()),
        requested_sandbox_policy: sandbox_policy,
        effective_sandbox_policy: sandbox_policy,
        granted_permissions: Default::default(),
        managed_network_host,
        strict_guardian: false,
        explicit_sandbox_escalation: false,
        sandbox_denial_retry_allowed: false,
        network_denial_retry_allowed,
        cancel_token: None,
    }
}

#[tokio::test]
async fn orchestrated_process_does_not_repeat_policy_approval() {
    let approvals = RecordingProcessApprovals::default();
    let runner = ProcessAttemptRunner {
        server: ExecutionProcessServer::default(),
        calls: AtomicUsize::new(0),
        runtime_metadata: None,
    };

    let result = orchestrate_runtime_tool_execution(
        process_orchestration_input(
            RuntimeToolInitialApproval::Required(RuntimeToolApprovalKind::User),
            RuntimeToolSandboxPolicy::DangerFullAccess,
            None,
            false,
        ),
        &approvals,
        &runner,
    )
    .await
    .expect("orchestrated process should start after one approval");

    assert_eq!(result.output, "orchestrated-process-1");
    assert_eq!(approvals.calls.load(Ordering::SeqCst), 1);
    assert_eq!(runner.calls.load(Ordering::SeqCst), 1);
    let snapshot = wait_for_terminal_snapshot(&runner.server, "orchestrated-process-1").await;
    assert_eq!(snapshot.status, ExecutionProcessStatus::Exited);
}

#[tokio::test]
async fn managed_network_denial_precedes_sandbox_fallback_and_retries_with_network_grant() {
    let approvals = RecordingProcessApprovals::default();
    let runner = ProcessAttemptRunner {
        server: ExecutionProcessServer::default(),
        calls: AtomicUsize::new(0),
        runtime_metadata: None,
    };

    let result = orchestrate_runtime_tool_execution(
        process_orchestration_input(
            RuntimeToolInitialApproval::NotRequired,
            RuntimeToolSandboxPolicy::WorkspaceWrite,
            Some("example.com".to_string()),
            true,
        ),
        &approvals,
        &runner,
    )
    .await
    .expect("managed network approval should retry the process");

    assert_eq!(result.output, "orchestrated-process-2");
    assert_eq!(approvals.calls.load(Ordering::SeqCst), 1);
    assert_eq!(runner.calls.load(Ordering::SeqCst), 2);
    let snapshot = wait_for_terminal_snapshot(&runner.server, "orchestrated-process-2").await;
    assert_eq!(snapshot.status, ExecutionProcessStatus::Exited);
}

#[tokio::test]
async fn execution_process_server_streams_output_and_status() {
    let server = ExecutionProcessServer::default();
    let response = server
        .start_thread_process(
            "test-thread",
            "test command",
            LiveExecutionRequest {
                process_id: "process-test".to_string(),
                tool_id: "tool-test".to_string(),
                tool_name: "exec_command".to_string(),
                command: vec![
                    "sh".to_string(),
                    "-c".to_string(),
                    "printf hello".to_string(),
                ],
                working_directory: current_directory(),
                tty: false,
                approval_policy: Some("never".to_string()),
                sandbox_policy: Some("danger-full-access".to_string()),
                runtime_metadata: None,
                env: HashMap::new(),
                attempt: None,
            },
        )
        .await
        .expect("process should start");
    assert_eq!(response.status, ExecutionProcessStatus::Running);

    let output = wait_for_output(&server, "process-test", "hello").await;
    assert!(output
        .deltas
        .iter()
        .any(|delta| delta.delta.contains("hello")));
    let snapshot = wait_for_terminal_snapshot(&server, "process-test").await;
    assert_eq!(snapshot.status, ExecutionProcessStatus::Exited);
}

#[tokio::test]
async fn execution_process_output_replays_until_cursor_advances() {
    let server = ExecutionProcessServer::default();
    server
        .start_thread_process(
            "test-thread",
            "test command",
            LiveExecutionRequest {
                process_id: "process-replay".to_string(),
                tool_id: "tool-replay".to_string(),
                tool_name: "exec_command".to_string(),
                command: vec![
                    "sh".to_string(),
                    "-c".to_string(),
                    "printf replay".to_string(),
                ],
                working_directory: current_directory(),
                tty: false,
                approval_policy: Some("never".to_string()),
                sandbox_policy: Some("danger-full-access".to_string()),
                runtime_metadata: None,
                env: HashMap::new(),
                attempt: None,
            },
        )
        .await
        .expect("process should start");

    let first = wait_for_output(&server, "process-replay", "replay").await;
    let cursor = first.next_sequence.expect("cursor should advance");
    let repeated = server
        .drain_output(LiveExecutionOutputQuery {
            process_id: Some("process-replay".to_string()),
            after_sequence: None,
            limit: None,
            max_bytes: None,
        })
        .expect("output should remain replayable");
    assert_eq!(repeated.deltas, first.deltas);

    let after_cursor = server
        .drain_output(LiveExecutionOutputQuery {
            process_id: Some("process-replay".to_string()),
            after_sequence: Some(cursor),
            limit: None,
            max_bytes: None,
        })
        .expect("cursor read should succeed");
    assert!(after_cursor.deltas.is_empty());
    assert_eq!(after_cursor.next_sequence, Some(cursor));
}

#[tokio::test]
async fn execution_process_server_tracks_registered_live_process() {
    let server = ExecutionProcessServer::default();
    let mut handle = start_local_execution_process(LocalExecutionRequest {
        process_id: "process-registered".to_string(),
        tool_id: "tool-registered".to_string(),
        tool_name: "exec_command".to_string(),
        command: shell_output_command("registered-output"),
        cwd: Some(std::env::current_dir().unwrap_or_default()),
        env: HashMap::new(),
        tty: false,
        stdin: true,
        env_clear: false,
        pty_size: None,
        sandbox: None,
    })
    .expect("local process should start");

    server
        .register_live_process(handle.control_handle(), handle.status())
        .expect("registered process should attach");
    assert_eq!(
        server
            .status("process-registered")
            .expect("registered status should read")
            .status,
        ExecutionProcessStatus::Running
    );

    let mut saw_output = false;
    while let Some(delta) = handle.recv_output().await {
        saw_output |= delta.delta.contains("registered-output");
        server
            .record_live_process_output(delta)
            .expect("registered output should record");
    }
    assert!(saw_output);

    let final_snapshot = handle.wait().await.expect("process should finish");
    server
        .finish_live_process(final_snapshot)
        .expect("registered process should finish");
    let output = server
        .drain_output(LiveExecutionOutputQuery {
            process_id: Some("process-registered".to_string()),
            after_sequence: None,
            limit: None,
            max_bytes: None,
        })
        .expect("registered output should drain");
    assert!(output
        .deltas
        .iter()
        .any(|delta| delta.delta.contains("registered-output")));
    let status = server
        .status("process-registered")
        .expect("final registered status should read");
    assert_eq!(status.status, ExecutionProcessStatus::Exited);
    assert_eq!(status.exit_code, Some(0));
}

#[tokio::test]
async fn execution_process_server_rejects_dangerous_shell_command() {
    let error = ExecutionProcessServer::default()
        .start_thread_process(
            "test-thread",
            "test command",
            LiveExecutionRequest {
                process_id: "process-danger".to_string(),
                tool_id: "tool-danger".to_string(),
                tool_name: "exec_command".to_string(),
                command: vec!["sh".to_string(), "-c".to_string(), "rm -rf /".to_string()],
                working_directory: current_directory(),
                tty: false,
                approval_policy: Some("never".to_string()),
                sandbox_policy: Some("danger-full-access".to_string()),
                runtime_metadata: None,
                env: HashMap::new(),
                attempt: None,
            },
        )
        .await
        .expect_err("dangerous command should be rejected");

    assert!(matches!(error, ExecutionProcessError::Policy(_)));
}

#[tokio::test]
async fn execution_process_server_uses_current_unsandboxed_fallback_when_backend_is_disabled() {
    let response = ExecutionProcessServer::default()
        .start_thread_process(
            "test-thread",
            "test command",
            LiveExecutionRequest {
                process_id: "process-sandbox".to_string(),
                tool_id: "tool-sandbox".to_string(),
                tool_name: "exec_command".to_string(),
                command: vec![
                    "sh".to_string(),
                    "-c".to_string(),
                    "printf allowed".to_string(),
                ],
                working_directory: current_directory(),
                tty: false,
                approval_policy: Some("never".to_string()),
                sandbox_policy: Some("workspace-write".to_string()),
                runtime_metadata: None,
                env: HashMap::new(),
                attempt: None,
            },
        )
        .await
        .expect("disabled workspace sandbox backend should preserve configured fallback policy");

    assert_eq!(response.tool_name, "exec_command");
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn execution_process_server_enforces_seatbelt_workspace_boundaries() {
    let root = tempfile::tempdir().expect("sandbox temp root");
    let workspace = root.path().join("workspace");
    std::fs::create_dir_all(&workspace).expect("sandbox workspace");
    let outside_path = root.path().join("outside.txt");
    let server = ExecutionProcessServer::default();
    server
        .start_thread_process(
            "test-thread",
            "test command",
            LiveExecutionRequest {
                process_id: "process-seatbelt".to_string(),
                tool_id: "tool-seatbelt".to_string(),
                tool_name: "exec_command".to_string(),
                command: vec![
                    "sh".to_string(),
                    "-c".to_string(),
                    concat!(
                        "printf allowed > inside.txt; ",
                        "printf denied > \"$OUTSIDE_PATH\" 2>/dev/null || true"
                    )
                    .to_string(),
                ],
                working_directory: workspace.clone(),
                tty: false,
                approval_policy: Some("never".to_string()),
                sandbox_policy: Some("workspace-write".to_string()),
                runtime_metadata: Some(json!({
                    "workspaceSandbox": { "enabled": true, "strict": true },
                    "metadata": {
                        "grantedPermissions": {
                            "fileSystem": {
                                "write": [outside_path.to_string_lossy().to_string()]
                            }
                        }
                    }
                })),
                env: HashMap::from([(
                    "OUTSIDE_PATH".to_string(),
                    outside_path.to_string_lossy().to_string(),
                )]),
                attempt: None,
            },
        )
        .await
        .expect("seatbelt process should start");

    let final_snapshot = wait_for_terminal_snapshot(&server, "process-seatbelt").await;
    assert_eq!(final_snapshot.status, ExecutionProcessStatus::Exited);
    assert_eq!(final_snapshot.exit_code, Some(0));
    assert_eq!(
        std::fs::read_to_string(workspace.join("inside.txt")).expect("workspace write"),
        "allowed"
    );
    assert!(!outside_path.exists());
}

async fn wait_for_output(
    server: &ExecutionProcessServer,
    process_id: &str,
    marker: &str,
) -> LiveExecutionOutputBatch {
    for _ in 0..80 {
        let output = server
            .drain_output(LiveExecutionOutputQuery {
                process_id: Some(process_id.to_string()),
                after_sequence: None,
                limit: None,
                max_bytes: None,
            })
            .expect("execution process output");
        if output
            .deltas
            .iter()
            .any(|delta| delta.delta.contains(marker))
        {
            return output;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    panic!("execution process did not emit marker '{marker}': {process_id}");
}

async fn wait_for_terminal_snapshot(
    server: &ExecutionProcessServer,
    process_id: &str,
) -> ExecutionProcessSnapshot {
    for _ in 0..80 {
        let snapshot = server.status(process_id).expect("execution process status");
        if matches!(
            snapshot.status,
            ExecutionProcessStatus::Exited
                | ExecutionProcessStatus::Interrupted
                | ExecutionProcessStatus::Terminated
                | ExecutionProcessStatus::Failed
        ) {
            return snapshot;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    panic!("execution process did not reach terminal status: {process_id}");
}

fn current_directory() -> PathBuf {
    std::env::current_dir().unwrap_or_default()
}

fn shell_output_command(output: &str) -> Vec<String> {
    if cfg!(windows) {
        vec![
            "cmd".to_string(),
            "/D".to_string(),
            "/S".to_string(),
            "/C".to_string(),
            format!("echo {output}"),
        ]
    } else {
        vec![
            "sh".to_string(),
            "-c".to_string(),
            format!("printf {output}"),
        ]
    }
}
