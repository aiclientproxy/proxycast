use super::*;
use futures::{SinkExt, StreamExt};
use serde_json::json;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;
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
use tool_runtime::unified_exec::{
    execute_runtime_unified_exec_tool, RuntimeUnifiedExecToolRequest, EXEC_COMMAND_TOOL_NAME,
    WRITE_STDIN_TOOL_NAME,
};

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
                    environment_id: "local".to_string(),
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
                environment_id: "local".to_string(),
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
async fn unified_exec_yields_active_session_then_poll_observes_terminal_process() {
    let server = Arc::new(ExecutionProcessServer::default());
    let command = if cfg!(target_os = "windows") {
        "Start-Sleep -Milliseconds 750; Write-Output done"
    } else {
        "sleep 1; printf done"
    };
    let exec_params = json!({
        "cmd": command,
        "login": false,
        "yield_time_ms": 250,
        "sandbox_permissions": "require_escalated",
        "justification": "exercise the local execution process lifecycle"
    });
    let running = execute_runtime_unified_exec_tool(
        server.clone(),
        RuntimeUnifiedExecToolRequest {
            tool_name: EXEC_COMMAND_TOOL_NAME,
            params: &exec_params,
            thread_id: "unified-exec-thread",
            environment_id: "local",
            working_directory: current_directory(),
            environment: HashMap::new(),
            tool_call_id: "unified-exec-call".to_string(),
            cancel_token: None,
            turn_context: None,
            attempt: None,
        },
    )
    .await
    .expect("long command should yield an active session");
    let running = running
        .structured_content
        .as_ref()
        .expect("active command structured output");
    let session_id = running["session_id"].as_i64().expect("active session id");
    assert_eq!(running["observation"]["process_active"], json!(true));
    assert_eq!(running["observation"]["kind"], json!("waiting"));

    let poll_params = json!({
        "session_id": session_id,
        "chars": "",
        "yield_time_ms": 2_000
    });
    let terminal = execute_runtime_unified_exec_tool(
        server,
        RuntimeUnifiedExecToolRequest {
            tool_name: WRITE_STDIN_TOOL_NAME,
            params: &poll_params,
            thread_id: "unified-exec-thread",
            environment_id: "local",
            working_directory: current_directory(),
            environment: HashMap::new(),
            tool_call_id: "unified-exec-poll".to_string(),
            cancel_token: None,
            turn_context: None,
            attempt: None,
        },
    )
    .await
    .expect("poll should observe terminal process state");
    let structured = terminal
        .structured_content
        .as_ref()
        .expect("terminal command structured output");
    assert_eq!(structured["observation"]["process_active"], json!(false));
    assert_eq!(structured["observation"]["kind"], json!("terminal"));
    assert_eq!(structured["exit_code"], json!(0));
    assert!(structured["output"]
        .as_str()
        .unwrap_or_default()
        .contains("done"));
    assert_eq!(
        terminal.metadata.get("exec_command_call_id"),
        Some(&json!("unified-exec-call"))
    );
}

#[tokio::test]
async fn unified_exec_reports_silent_non_zero_exit_as_terminal() {
    let server = Arc::new(ExecutionProcessServer::default());
    let params = json!({
        "cmd": "exit 7",
        "login": false,
        "yield_time_ms": 1_000,
        "sandbox_permissions": "require_escalated",
        "justification": "exercise silent non-zero command termination"
    });

    let result = execute_runtime_unified_exec_tool(
        server,
        RuntimeUnifiedExecToolRequest {
            tool_name: EXEC_COMMAND_TOOL_NAME,
            params: &params,
            thread_id: "unified-exec-silent-failure-thread",
            environment_id: "local",
            working_directory: current_directory(),
            environment: HashMap::new(),
            tool_call_id: "unified-exec-silent-failure-call".to_string(),
            cancel_token: None,
            turn_context: None,
            attempt: None,
        },
    )
    .await
    .expect("silent non-zero command should produce a terminal result");
    let structured = result
        .structured_content
        .as_ref()
        .expect("terminal command structured output");

    assert!(!result.success);
    assert_eq!(structured["observation"]["kind"], json!("terminal"));
    assert_eq!(structured["observation"]["process_active"], json!(false));
    assert_eq!(structured["exit_code"], json!(7));
    assert_eq!(structured["output"], json!(""));
}

#[tokio::test]
async fn unified_exec_starts_short_command_after_repeated_terminal_commands() {
    let server = Arc::new(ExecutionProcessServer::default());
    let directory = tempfile::tempdir().expect("temp directory");
    let file_path = directory.path().join("structural_selector.rs");
    std::fs::write(&file_path, "selector fixture\n").expect("write selector fixture");

    for index in 0..8 {
        let params = json!({
            "cmd": format!("printf terminal-{index}"),
            "login": false,
            "yield_time_ms": 1_000,
            "sandbox_permissions": "require_escalated",
            "justification": "exercise repeated terminal command lifecycle"
        });
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            execute_runtime_unified_exec_tool(
                server.clone(),
                RuntimeUnifiedExecToolRequest {
                    tool_name: EXEC_COMMAND_TOOL_NAME,
                    params: &params,
                    thread_id: "unified-exec-sequence-thread",
                    environment_id: "local",
                    working_directory: current_directory(),
                    environment: HashMap::new(),
                    tool_call_id: format!("unified-exec-sequence-{index}"),
                    cancel_token: None,
                    turn_context: None,
                    attempt: None,
                },
            ),
        )
        .await
        .expect("terminal command must not block")
        .expect("terminal command should succeed");
        assert_eq!(result.structured_content.as_ref().unwrap()["exit_code"], 0);
    }

    let params = json!({
        "cmd": format!("cat {}", file_path.to_string_lossy()),
        "login": false,
        "yield_time_ms": 1_000,
        "sandbox_permissions": "require_escalated",
        "justification": "exercise command start after repeated terminal commands"
    });
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        execute_runtime_unified_exec_tool(
            server,
            RuntimeUnifiedExecToolRequest {
                tool_name: EXEC_COMMAND_TOOL_NAME,
                params: &params,
                thread_id: "unified-exec-sequence-thread",
                environment_id: "local",
                working_directory: current_directory(),
                environment: HashMap::new(),
                tool_call_id: "unified-exec-sequence-read".to_string(),
                cancel_token: None,
                turn_context: None,
                attempt: None,
            },
        ),
    )
    .await
    .expect("short command must start after repeated terminal commands")
    .expect("short command should succeed");
    assert!(result.output.contains("selector fixture"));
}

#[tokio::test]
async fn execution_process_server_never_lowers_remote_environment_to_local_process() {
    let error = ExecutionProcessServer::default()
        .start_thread_process(
            "test-thread",
            "printf remote",
            LiveExecutionRequest {
                process_id: "process-remote".to_string(),
                tool_id: "tool-remote".to_string(),
                tool_name: "exec_command".to_string(),
                environment_id: "remote-tools".to_string(),
                command: shell_output_command("remote"),
                working_directory: PathBuf::from("/remote/workspace"),
                tty: false,
                approval_policy: Some("never".to_string()),
                sandbox_policy: Some("danger-full-access".to_string()),
                runtime_metadata: None,
                env: HashMap::new(),
                attempt: None,
            },
        )
        .await
        .expect_err("remote Environment must not fall back to local execution");

    assert!(matches!(
        error,
        ExecutionProcessError::UnsupportedEnvironment(environment_id)
            if environment_id == "remote-tools"
    ));
}

#[tokio::test]
async fn execution_process_server_uses_environment_process_transport() {
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("remote exec fixture bind");
    let address = listener.local_addr().expect("remote exec fixture address");
    let read_count = Arc::new(AtomicUsize::new(0));
    let fixture_read_count = Arc::clone(&read_count);
    let fixture = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("remote exec fixture accept");
        let mut socket = accept_async(stream)
            .await
            .expect("remote exec fixture websocket");
        while let Some(message) = socket.next().await {
            let Message::Text(text) = message.expect("remote exec fixture message") else {
                continue;
            };
            let request: Value = serde_json::from_str(&text).expect("remote exec fixture JSON");
            let Some(id) = request.get("id") else {
                continue;
            };
            let method = request["method"]
                .as_str()
                .expect("remote exec fixture method");
            let result = match method {
                "initialize" => json!({"sessionId": "remote-process-fixture"}),
                "environment/info" => json!({
                    "shell": {"name": "fixture-sh", "path": "/bin/fixture-sh"},
                    "cwd": "file:///remote/workspace"
                }),
                "environment/status" => json!({"status": "ready"}),
                "process/start" => json!({
                    "processId": request["params"]["processId"].clone()
                }),
                "process/read" => {
                    let read = fixture_read_count.fetch_add(1, Ordering::SeqCst);
                    if read == 0 {
                        json!({
                            "chunks": [{
                                "seq": 0,
                                "stream": "stdout",
                                "chunk": base64::engine::general_purpose::STANDARD.encode("remote-output")
                            }],
                            "nextSeq": 1,
                            "exited": true,
                            "exitCode": 0,
                            "closed": false,
                            "failure": null,
                            "sandboxDenied": false
                        })
                    } else {
                        json!({"chunks": [], "nextSeq": 1, "exited": true, "exitCode": 0})
                    }
                }
                "process/write" | "process/signal" | "process/terminate" => json!({}),
                method => panic!("unexpected remote exec method: {method}"),
            };
            socket
                .send(Message::Text(
                    json!({"jsonrpc": "2.0", "id": id, "result": result}).to_string(),
                ))
                .await
                .expect("remote exec fixture response");
        }
    });

    let registry = Arc::new(EnvironmentRegistry::new());
    registry
        .upsert(
            "remote-fixture".to_string(),
            format!("ws://{address}"),
            None,
        )
        .await
        .expect("remote fixture registry entry");
    registry.start();

    let server = ExecutionProcessServer::default();
    server.attach_environment_registry(Arc::clone(&registry));
    let mut client_ready = false;
    for _ in 0..80 {
        if registry.execution_client("remote-fixture").await.is_ok() {
            client_ready = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    assert!(
        client_ready,
        "remote fixture should complete Environment handshake"
    );

    let response = server
        .start_thread_process(
            "test-thread",
            "printf remote-output",
            LiveExecutionRequest {
                process_id: "process-remote-fixture".to_string(),
                tool_id: "tool-remote-fixture".to_string(),
                tool_name: "exec_command".to_string(),
                environment_id: "remote-fixture".to_string(),
                command: shell_output_command("remote-output"),
                working_directory: PathBuf::from(r"C:\remote\workspace"),
                tty: false,
                approval_policy: Some("never".to_string()),
                sandbox_policy: Some("workspace-write".to_string()),
                runtime_metadata: None,
                env: HashMap::new(),
                attempt: None,
            },
        )
        .await
        .expect("remote process should start through Environment transport");
    assert_eq!(response.status, ExecutionProcessStatus::Running);
    server
        .write_stdin("process-remote-fixture", b"input")
        .expect("remote stdin should use process/write");
    server
        .signal("process-remote-fixture")
        .expect("remote signal should use process/signal");

    let output = wait_for_output(&server, "process-remote-fixture", "remote-output").await;
    assert!(output
        .deltas
        .iter()
        .any(|delta| delta.delta.contains("remote-output")));
    let snapshot = wait_for_terminal_snapshot(&server, "process-remote-fixture").await;
    assert_eq!(snapshot.status, ExecutionProcessStatus::Exited);
    assert_eq!(snapshot.exit_code, Some(0));
    assert!(read_count.load(Ordering::SeqCst) >= 1);

    fixture.abort();
    let _ = fixture.await;
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
                environment_id: "local".to_string(),
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
                environment_id: "local".to_string(),
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
                environment_id: "local".to_string(),
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
                environment_id: "local".to_string(),
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

#[test]
fn remote_sandbox_context_lowers_codex_environment_wire() {
    let cwd = app_server_protocol::protocol::v2::PathUri::from_host_path(current_directory())
        .expect("test cwd should be representable as file URI");
    let request = LiveExecutionRequest {
        process_id: "remote-process".to_string(),
        tool_id: "tool".to_string(),
        tool_name: "exec_command".to_string(),
        environment_id: "remote".to_string(),
        command: shell_output_command("hello"),
        working_directory: current_directory(),
        tty: false,
        approval_policy: None,
        sandbox_policy: Some("workspace-write".to_string()),
        runtime_metadata: None,
        env: HashMap::new(),
        attempt: None,
    };
    let context = remote_sandbox_context(&request, &cwd).expect("workspace sandbox lowering");
    assert_eq!(context["permissions"]["type"], "managed");
    assert_eq!(context["permissions"]["fileSystem"]["type"], "restricted");
    assert_eq!(
        context["permissions"]["fileSystem"]["entries"][0]["access"],
        "write"
    );
    assert_eq!(context["cwd"], cwd.as_str());
}

#[test]
fn runtime_windows_sandbox_mode_reads_persisted_metadata() {
    let metadata = json!({
        "agent": { "workspaceSandbox": { "mode": "unelevated" } }
    });
    assert_eq!(
        runtime_windows_sandbox_mode(Some(&metadata)),
        Some(tool_runtime::sandbox::WindowsSandboxExecutionMode::Unelevated)
    );
}

#[test]
fn remote_sandbox_context_rejects_unknown_policy() {
    let cwd = app_server_protocol::protocol::v2::PathUri::from_host_path(current_directory())
        .expect("test cwd should be representable as file URI");
    let request = LiveExecutionRequest {
        process_id: "remote-process".to_string(),
        tool_id: "tool".to_string(),
        tool_name: "exec_command".to_string(),
        environment_id: "remote".to_string(),
        command: shell_output_command("hello"),
        working_directory: current_directory(),
        tty: false,
        approval_policy: None,
        sandbox_policy: Some("future-policy".to_string()),
        runtime_metadata: None,
        env: HashMap::new(),
        attempt: None,
    };
    assert!(matches!(
        remote_sandbox_context(&request, &cwd),
        Err(ExecutionProcessError::SandboxDenied { .. })
    ));
}

#[test]
fn remote_path_uri_preserves_foreign_windows_drive_paths() {
    let uri = remote_path_uri(Path::new(r"C:\workspace\project"))
        .expect("Windows drive path should lower to a file URI");
    assert_eq!(uri.as_str(), "file:///C:/workspace/project");
}

#[test]
fn remote_path_uri_rejects_relative_paths() {
    assert!(remote_path_uri(Path::new("relative/project")).is_err());
}
