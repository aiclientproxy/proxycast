use super::*;
use environment::{resolve_child_environment_with_semantics, EnvironmentKeySemantics};

fn start_process() -> ExecutionProcess {
    ExecutionProcess::start(ExecutionProcessStart {
        process_id: "process-1".to_string(),
        tool_id: "tool-1".to_string(),
        tool_name: "exec_command".to_string(),
        command: Some("npm test".to_string()),
        cwd: Some("/tmp/project".to_string()),
    })
}

#[test]
fn process_tracks_output_delta_metadata() {
    let mut process = start_process();
    let delta = process.append_output(ExecutionOutputKind::Stdout, b"hello");

    assert_eq!(delta.sequence, 1);
    assert_eq!(delta.delta, "hello");
    assert_eq!(delta.bytes, 5);
    assert_eq!(delta.omitted_bytes, 0);
    assert!(!delta.truncated);

    let metadata = delta.metadata();
    assert_eq!(metadata.get("processId"), Some(&json!("process-1")));
    assert_eq!(metadata.get("outputBytes"), Some(&json!(5)));
    assert_eq!(metadata.get("outputTruncated"), Some(&json!(false)));
    assert_eq!(metadata.get("stdinWritable"), Some(&json!(true)));
    assert_eq!(metadata.get("stdin_writable"), Some(&json!(true)));
}

#[test]
fn process_bounds_retained_output() {
    let mut output = BoundedProcessOutput::new(8);
    output.push(b"12345");
    output.push(b"67890");

    let snapshot = output.snapshot();
    assert_eq!(snapshot.bytes, 10);
    assert_eq!(snapshot.omitted_bytes, 2);
    assert!(snapshot.truncated);
    assert_eq!(snapshot.text, "34567890");
}

#[test]
fn process_status_terminal_transitions_do_not_regress() {
    let mut process = start_process();
    process.interrupt();
    process.exit(0);

    let snapshot = process.snapshot();
    assert_eq!(snapshot.status, ExecutionProcessStatus::Interrupted);
    assert_eq!(snapshot.exit_code, None);
    let metadata = snapshot.metadata();
    assert_eq!(metadata.get("stdinWritable"), Some(&json!(false)));
    assert_eq!(metadata.get("stdin_writable"), Some(&json!(false)));
}

#[test]
fn manager_controls_process_lifecycle() {
    let mut manager = ExecutionProcessManager::default();
    let snapshot = manager.start(ExecutionProcessStart {
        process_id: "process-1".to_string(),
        tool_id: "tool-1".to_string(),
        tool_name: "exec_command".to_string(),
        command: Some("cargo test".to_string()),
        cwd: None,
    });
    assert_eq!(snapshot.status, ExecutionProcessStatus::Running);

    let delta = manager
        .append_output("process-1", ExecutionOutputKind::Combined, b"running")
        .expect("process should exist");
    assert_eq!(delta.sequence, 1);

    let snapshot = manager
        .terminate("process-1")
        .expect("process should terminate");
    assert_eq!(snapshot.status, ExecutionProcessStatus::Terminated);
    assert_eq!(snapshot.retained_output, "running");
}

#[tokio::test]
async fn local_process_emits_stdout_stderr_and_exit_snapshot() {
    let mut handle = start_local_execution_process(LocalExecutionRequest::new(
        "process-local-1",
        "tool-local-1",
        "exec_command",
        shell_command("printf stdout; printf stderr 1>&2"),
    ))
    .expect("local process should start");

    let mut observed = Vec::new();
    while let Ok(Some(delta)) =
        tokio::time::timeout(Duration::from_secs(2), handle.recv_output()).await
    {
        observed.push(delta);
    }

    let final_snapshot = handle.wait().await.expect("process should finish");
    assert_eq!(final_snapshot.status, ExecutionProcessStatus::Exited);
    assert_eq!(final_snapshot.exit_code, Some(0));
    assert!(final_snapshot.retained_output.contains("stdout"));
    assert!(final_snapshot.retained_output.contains("stderr"));
    assert!(observed
        .iter()
        .any(|delta| delta.kind == ExecutionOutputKind::Stdout && delta.delta == "stdout"));
    assert!(observed
        .iter()
        .any(|delta| delta.kind == ExecutionOutputKind::Stderr && delta.delta == "stderr"));
}

#[cfg(not(target_os = "windows"))]
#[tokio::test]
async fn local_process_does_not_inherit_sensitive_parent_environment() {
    const SECRET_NAME: &str = "LIME_EXECUTION_PROCESS_TEST_SECRET";
    let inherited_environment = [
        ("PATH".to_string(), "/usr/bin:/bin".to_string()),
        (SECRET_NAME.to_string(), "inherited-secret".to_string()),
    ];
    let mut handle = start_local_execution_process_with_inherited_environment(
        LocalExecutionRequest::new(
            "process-local-environment",
            "tool-local-environment",
            "exec_command",
            shell_command(
                "if [ -n \"${LIME_EXECUTION_PROCESS_TEST_SECRET+x}\" ]; then printf leaked; else printf filtered; fi",
            ),
        ),
        inherited_environment,
    )
    .expect("local process should start");

    let final_snapshot = handle.wait().await.expect("process should finish");
    assert_eq!(final_snapshot.status, ExecutionProcessStatus::Exited);
    assert_eq!(final_snapshot.exit_code, Some(0));
    assert_eq!(final_snapshot.retained_output, "filtered");
}

#[tokio::test]
async fn local_process_next_event_delivers_output_before_exit() {
    let mut handle = start_local_execution_process(LocalExecutionRequest::new(
        "process-local-events",
        "tool-local-events",
        "exec_command",
        shell_command("printf stdout"),
    ))
    .expect("local process should start");

    let output = tokio::time::timeout(Duration::from_secs(2), handle.next_event())
        .await
        .expect("output timeout")
        .expect("output event");
    let LocalExecutionProcessEvent::Output(output) = output else {
        panic!("output must precede exit");
    };
    assert_eq!(output.raw_bytes, b"stdout");

    let exited = tokio::time::timeout(Duration::from_secs(2), handle.next_event())
        .await
        .expect("exit timeout")
        .expect("exit event");
    let LocalExecutionProcessEvent::Exited(exited) = exited else {
        panic!("expected exit event");
    };
    assert_eq!(exited.exit_code, Some(0));
    assert!(handle.next_event().await.is_none());
}

#[tokio::test]
async fn local_process_terminate_sets_terminal_status() {
    let mut handle = start_local_execution_process(LocalExecutionRequest::new(
        "process-local-terminate",
        "tool-local-terminate",
        "exec_command",
        shell_command("sleep 5"),
    ))
    .expect("local process should start");

    handle.terminate().expect("terminate signal should send");
    let final_snapshot = handle.wait().await.expect("process should finish");

    assert_eq!(final_snapshot.status, ExecutionProcessStatus::Terminated);
}

#[tokio::test]
async fn local_pty_process_accepts_stdin_and_emits_combined_output() {
    let mut request = LocalExecutionRequest::new(
        "process-local-pty",
        "tool-local-pty",
        "exec_command",
        interactive_shell_command(),
    );
    request.tty = true;
    let mut handle = start_local_execution_process(request).expect("PTY process should start");

    handle
        .write_stdin("hello-from-pty\n")
        .expect("PTY stdin should remain writable");
    let mut observed = Vec::new();
    loop {
        match tokio::time::timeout(Duration::from_secs(5), handle.recv_output()).await {
            Ok(Some(delta)) => observed.push(delta),
            Ok(None) => break,
            Err(_) => panic!("timed out waiting for PTY output"),
        }
    }

    let final_snapshot = tokio::time::timeout(Duration::from_secs(5), handle.wait())
        .await
        .expect("PTY process should terminate")
        .expect("PTY final snapshot should be available");
    assert_eq!(final_snapshot.status, ExecutionProcessStatus::Exited);
    assert_eq!(final_snapshot.exit_code, Some(0));
    assert!(final_snapshot.retained_output.contains("PTY_READY"));
    assert!(final_snapshot
        .retained_output
        .contains("PTY_ECHO:hello-from-pty"));
    assert!(observed
        .iter()
        .all(|delta| delta.kind == ExecutionOutputKind::Combined));
}

#[cfg(not(target_os = "windows"))]
#[test]
fn restricted_token_sandbox_fails_closed_off_windows() {
    let mut request = LocalExecutionRequest::new(
        "process-restricted-token",
        "tool-restricted-token",
        "exec_command",
        shell_command("printf restricted"),
    );
    request.sandbox = Some(LocalExecutionSandbox {
        backend: SandboxBackend::RestrictedToken,
        requested_policy: Some("workspace-write".to_string()),
        granted_permissions: None,
    });

    let error =
        start_local_execution_process(request).expect_err("restricted token must fail closed");
    assert_eq!(error.kind(), std::io::ErrorKind::Unsupported);
    assert!(error.to_string().contains("only available on Windows"));
}

#[test]
fn windows_environment_inherits_and_applies_case_insensitive_overrides() {
    let inherited = [
        ("Path".to_string(), "C:\\Windows".to_string()),
        ("SystemRoot".to_string(), "C:\\Windows".to_string()),
        ("OPENAI_API_KEY".to_string(), "inherited-key".to_string()),
        ("service_secret".to_string(), "inherited-secret".to_string()),
        ("Access_Token".to_string(), "inherited-token".to_string()),
    ];
    let overrides = HashMap::from([
        ("PATH".to_string(), "C:\\Tools".to_string()),
        ("custom_value".to_string(), "enabled".to_string()),
    ]);

    let environment = resolve_child_environment_with_semantics(
        false,
        inherited,
        &overrides,
        EnvironmentKeySemantics::CaseInsensitive,
    );

    assert_eq!(
        environment.get("PATH").map(String::as_str),
        Some("C:\\Tools")
    );
    assert_eq!(
        environment.get("SYSTEMROOT").map(String::as_str),
        Some("C:\\Windows")
    );
    assert_eq!(
        environment.get("CUSTOM_VALUE").map(String::as_str),
        Some("enabled")
    );
    assert!(!environment.contains_key("OPENAI_API_KEY"));
    assert!(!environment.contains_key("SERVICE_SECRET"));
    assert!(!environment.contains_key("ACCESS_TOKEN"));
    assert_eq!(environment.len(), 3);
}

#[test]
fn windows_environment_clear_drops_inherited_values() {
    let inherited = [("SystemRoot".to_string(), "C:\\Windows".to_string())];
    let overrides = HashMap::from([("Path".to_string(), "C:\\Tools".to_string())]);

    let environment = resolve_child_environment_with_semantics(
        true,
        inherited,
        &overrides,
        EnvironmentKeySemantics::CaseInsensitive,
    );

    assert_eq!(
        environment.get("PATH").map(String::as_str),
        Some("C:\\Tools")
    );
    assert!(!environment.contains_key("SYSTEMROOT"));
}

#[test]
fn native_environment_preserves_key_case_and_filters_sensitive_inheritance() {
    let inherited = [
        ("PATH".to_string(), "/usr/bin".to_string()),
        ("Path".to_string(), "/custom/bin".to_string()),
        ("HOME".to_string(), "/home/test".to_string()),
        ("LANG".to_string(), "en_US.UTF-8".to_string()),
        ("api_key_hint".to_string(), "sensitive".to_string()),
        ("SERVICE_SECRET".to_string(), "sensitive".to_string()),
        ("AccessToken".to_string(), "sensitive".to_string()),
    ];

    let environment = resolve_child_environment_with_semantics(
        false,
        inherited,
        &HashMap::new(),
        EnvironmentKeySemantics::Native,
    );

    assert_eq!(
        environment.get("PATH").map(String::as_str),
        Some("/usr/bin")
    );
    assert_eq!(
        environment.get("Path").map(String::as_str),
        Some("/custom/bin")
    );
    assert_eq!(
        environment.get("HOME").map(String::as_str),
        Some("/home/test")
    );
    assert_eq!(
        environment.get("LANG").map(String::as_str),
        Some("en_US.UTF-8")
    );
    assert!(!environment.contains_key("api_key_hint"));
    assert!(!environment.contains_key("SERVICE_SECRET"));
    assert!(!environment.contains_key("AccessToken"));
}

#[test]
fn explicit_environment_overrides_can_restore_filtered_values() {
    let inherited = [
        ("OPENAI_API_KEY".to_string(), "inherited-key".to_string()),
        ("SERVICE_SECRET".to_string(), "inherited-secret".to_string()),
        ("ACCESS_TOKEN".to_string(), "inherited-token".to_string()),
    ];
    let overrides = HashMap::from([
        ("OPENAI_API_KEY".to_string(), "explicit-key".to_string()),
        ("SERVICE_SECRET".to_string(), "explicit-secret".to_string()),
        ("ACCESS_TOKEN".to_string(), "explicit-token".to_string()),
    ]);

    let environment = resolve_child_environment_with_semantics(
        false,
        inherited,
        &overrides,
        EnvironmentKeySemantics::Native,
    );

    assert_eq!(
        environment.get("OPENAI_API_KEY").map(String::as_str),
        Some("explicit-key")
    );
    assert_eq!(
        environment.get("SERVICE_SECRET").map(String::as_str),
        Some("explicit-secret")
    );
    assert_eq!(
        environment.get("ACCESS_TOKEN").map(String::as_str),
        Some("explicit-token")
    );
}

#[test]
fn windows_job_preserves_only_after_normal_root_exit() {
    assert!(should_preserve_windows_job(
        false,
        ExecutionProcessStatus::Running
    ));
    assert!(!should_preserve_windows_job(
        true,
        ExecutionProcessStatus::Running
    ));
    assert!(!should_preserve_windows_job(
        false,
        ExecutionProcessStatus::Interrupted
    ));
    assert!(!should_preserve_windows_job(
        false,
        ExecutionProcessStatus::Terminated
    ));
}

fn shell_command(script: &str) -> Vec<String> {
    if cfg!(windows) {
        vec![
            "cmd".to_string(),
            "/C".to_string(),
            script
                .replace("printf stdout", "echo|set /p=stdout")
                .replace("printf stderr 1>&2", "echo|set /p=stderr 1>&2")
                .replace("sleep 5", "timeout /T 5 /NOBREAK >NUL")
                .to_string(),
        ]
    } else {
        vec!["sh".to_string(), "-c".to_string(), script.to_string()]
    }
}

fn interactive_shell_command() -> Vec<String> {
    if cfg!(windows) {
        vec![
            "cmd.exe".to_string(),
            "/D".to_string(),
            "/V:ON".to_string(),
            "/S".to_string(),
            "/C".to_string(),
            "echo PTY_READY & set /p PTY_VALUE= & echo PTY_ECHO:!PTY_VALUE!".to_string(),
        ]
    } else {
        vec![
            "sh".to_string(),
            "-c".to_string(),
            "printf PTY_READY; IFS= read -r value; printf 'PTY_ECHO:%s' \"$value\"".to_string(),
        ]
    }
}
