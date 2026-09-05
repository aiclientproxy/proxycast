use super::*;
#[cfg(unix)]
use app_server_protocol::protocol::v2::CommandExecTerminalSize;
#[cfg(unix)]
use tokio::sync::mpsc;

#[cfg(unix)]
fn server_with_notifications() -> (
    CommandExecServer,
    mpsc::UnboundedReceiver<(ConnectionId, JsonRpcNotification)>,
) {
    let (tx, rx) = mpsc::unbounded_channel();
    let hook: CommandExecNotificationHook = Arc::new(move |connection_id, notification| {
        let tx = tx.clone();
        Box::pin(async move {
            let _ = tx.send((connection_id, notification));
        })
    });
    (
        CommandExecServer::default().with_notification_hook(hook),
        rx,
    )
}

#[cfg(unix)]
fn exec_params(process_id: Option<&str>, script: &str) -> CommandExecParams {
    CommandExecParams {
        command: vec!["/bin/sh".to_string(), "-c".to_string(), script.to_string()],
        process_id: process_id.map(str::to_string),
        tty: false,
        stream_stdin: false,
        stream_stdout_stderr: false,
        output_bytes_cap: Some(1024),
        disable_output_cap: false,
        disable_timeout: false,
        timeout_ms: Some(2_000),
        cwd: Some(std::env::current_dir().expect("current directory")),
        env: None,
        size: None,
        sandbox_policy: None,
        permission_profile: None,
    }
}

#[cfg(windows)]
fn windows_exec_params(script: &str) -> CommandExecParams {
    CommandExecParams {
        command: vec![
            "powershell.exe".to_string(),
            "-NoProfile".to_string(),
            "-NonInteractive".to_string(),
            "-Command".to_string(),
            script.to_string(),
        ],
        process_id: None,
        tty: false,
        stream_stdin: false,
        stream_stdout_stderr: false,
        output_bytes_cap: Some(1024),
        disable_output_cap: false,
        disable_timeout: false,
        timeout_ms: Some(20),
        cwd: Some(std::env::current_dir().expect("current directory")),
        env: None,
        size: None,
        sandbox_policy: None,
        permission_profile: None,
    }
}

#[cfg(unix)]
async fn wait_for_session(
    server: &CommandExecServer,
    connection_id: ConnectionId,
    process_id: &str,
) {
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        loop {
            if server
                .session(&key(connection_id, process_id))
                .await
                .is_ok()
            {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("command/exec session registration timeout");
}

#[cfg(unix)]
#[tokio::test]
async fn one_off_command_captures_stdout_stderr_and_exit_code() {
    let response = CommandExecServer::default()
        .exec(
            ConnectionId(1),
            exec_params(None, "printf stdout; printf stderr >&2; exit 7"),
            None,
        )
        .await
        .expect("execute command");

    assert_eq!(response.exit_code, 7);
    assert_eq!(response.stdout, "stdout");
    assert_eq!(response.stderr, "stderr");
}

#[cfg(unix)]
#[tokio::test]
async fn explicit_read_only_policy_fails_closed_for_mutating_command() {
    let temp = tempfile::tempdir().expect("tempdir");
    let mut params = exec_params(None, "touch command-exec-must-not-exist");
    params.cwd = Some(temp.path().to_path_buf());
    params.sandbox_policy = Some(serde_json::json!("read-only"));

    let error = CommandExecServer::default()
        .exec(ConnectionId(10), params, None)
        .await
        .expect_err("read-only command/exec must reject a mutating command");

    assert_eq!(error.code, error_codes::RUNTIME_ERROR);
    assert!(error
        .message
        .contains("read_only_sandbox_blocks_shell_command"));
    assert!(!temp.path().join("command-exec-must-not-exist").exists());
}

#[cfg(unix)]
#[tokio::test]
async fn streaming_command_emits_output_before_the_final_response() {
    let (server, mut notifications) = server_with_notifications();
    let connection_id = ConnectionId(2);
    let mut params = exec_params(Some("streaming"), "printf streamed");
    params.stream_stdout_stderr = true;
    let task = tokio::spawn({
        let server = server.clone();
        async move { server.exec(connection_id, params, None).await }
    });

    let (actual_connection, notification) =
        tokio::time::timeout(std::time::Duration::from_secs(2), notifications.recv())
            .await
            .expect("output timeout")
            .expect("output notification");
    assert_eq!(actual_connection, connection_id);
    let ServerNotification::CommandExecOutputDelta(delta) =
        ServerNotification::try_from(notification).expect("typed notification")
    else {
        panic!("expected command/exec/outputDelta notification");
    };
    assert_eq!(delta.process_id, "streaming");
    assert_eq!(delta.stream, CommandExecOutputStream::Stdout);
    assert_eq!(
        STANDARD.decode(delta.delta_base64).expect("base64"),
        b"streamed"
    );
    assert!(!delta.cap_reached);

    let response = task.await.expect("command task").expect("command response");
    assert_eq!(response.exit_code, 0);
    assert!(response.stdout.is_empty());
    assert!(response.stderr.is_empty());
}

#[cfg(unix)]
#[tokio::test]
async fn stdin_write_and_close_are_connection_scoped_and_fail_closed() {
    let server = CommandExecServer::default();
    let connection_id = ConnectionId(3);
    let mut params = exec_params(Some("stdin"), "cat");
    params.stream_stdin = true;
    let task = tokio::spawn({
        let server = server.clone();
        async move { server.exec(connection_id, params, None).await }
    });
    wait_for_session(&server, connection_id, "stdin").await;

    let foreign = server
        .write(
            ConnectionId(4),
            CommandExecWriteParams {
                process_id: "stdin".to_string(),
                delta_base64: Some(STANDARD.encode(b"foreign")),
                close_stdin: false,
            },
        )
        .await
        .expect_err("other connections must not access the session");
    assert_eq!(foreign.code, error_codes::INVALID_REQUEST);

    server
        .write(
            connection_id,
            CommandExecWriteParams {
                process_id: "stdin".to_string(),
                delta_base64: Some(STANDARD.encode(b"hello")),
                close_stdin: true,
            },
        )
        .await
        .expect("write and close stdin");
    let closed = server
        .write(
            connection_id,
            CommandExecWriteParams {
                process_id: "stdin".to_string(),
                delta_base64: Some(STANDARD.encode(b"late")),
                close_stdin: false,
            },
        )
        .await
        .expect_err("writes after close must fail");
    assert_eq!(closed.code, error_codes::INVALID_REQUEST);

    let response = task.await.expect("command task").expect("command response");
    assert_eq!(response.stdout, "hello");
}

#[cfg(unix)]
#[tokio::test]
async fn tty_resize_and_terminate_control_the_active_process() {
    let server = CommandExecServer::default();
    let connection_id = ConnectionId(5);
    let mut params = exec_params(Some("pty"), "sleep 10");
    params.tty = true;
    params.disable_timeout = true;
    params.timeout_ms = None;
    params.size = Some(CommandExecTerminalSize { rows: 24, cols: 80 });
    let task = tokio::spawn({
        let server = server.clone();
        async move { server.exec(connection_id, params, None).await }
    });
    wait_for_session(&server, connection_id, "pty").await;

    server
        .resize(
            connection_id,
            CommandExecResizeParams {
                process_id: "pty".to_string(),
                size: CommandExecTerminalSize {
                    rows: 40,
                    cols: 120,
                },
            },
        )
        .await
        .expect("resize pty");
    server
        .terminate(
            connection_id,
            CommandExecTerminateParams {
                process_id: "pty".to_string(),
            },
        )
        .await
        .expect("terminate pty");

    tokio::time::timeout(std::time::Duration::from_secs(2), task)
        .await
        .expect("terminated command timeout")
        .expect("command task")
        .expect("command response");
}

#[cfg(unix)]
#[tokio::test]
async fn duplicate_process_ids_fail_only_within_the_same_connection() {
    let server = CommandExecServer::default();
    let connection_id = ConnectionId(6);
    let mut first_params = exec_params(Some("shared"), "sleep 10");
    first_params.disable_timeout = true;
    first_params.timeout_ms = None;
    let first = tokio::spawn({
        let server = server.clone();
        async move { server.exec(connection_id, first_params, None).await }
    });
    wait_for_session(&server, connection_id, "shared").await;

    let duplicate = server
        .exec(
            connection_id,
            exec_params(Some("shared"), "printf duplicate"),
            None,
        )
        .await
        .expect_err("same-connection duplicate must fail");
    assert_eq!(duplicate.code, error_codes::INVALID_REQUEST);

    let other_connection = ConnectionId(7);
    let other = server
        .exec(
            other_connection,
            exec_params(Some("shared"), "printf other"),
            None,
        )
        .await
        .expect("other connection may reuse process id");
    assert_eq!(other.stdout, "other");

    server.connection_closed(connection_id).await;
    tokio::time::timeout(std::time::Duration::from_secs(2), first)
        .await
        .expect("connection cleanup timeout")
        .expect("command task")
        .expect("command response");
}

#[cfg(unix)]
#[tokio::test]
async fn output_cap_and_timeout_apply_to_one_off_commands() {
    let server = CommandExecServer::default();
    let mut capped = exec_params(None, "printf 12345");
    capped.output_bytes_cap = Some(3);
    let response = server
        .exec(ConnectionId(8), capped, None)
        .await
        .expect("capped command");
    assert_eq!(response.stdout, "123");

    let mut timed = exec_params(None, "sleep 10");
    timed.timeout_ms = Some(20);
    let response = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        server.exec(ConnectionId(8), timed, None),
    )
    .await
    .expect("timeout enforcement")
    .expect("timeout response");
    assert_eq!(response.exit_code, COMMAND_EXEC_TIMEOUT_EXIT_CODE);
}

#[cfg(windows)]
#[tokio::test]
async fn windows_timeout_returns_canonical_exit_code() {
    let response = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        CommandExecServer::default().exec(
            ConnectionId(9),
            windows_exec_params("[System.Threading.Thread]::Sleep(10000)"),
            None,
        ),
    )
    .await
    .expect("Windows timeout enforcement should finish")
    .expect("Windows timeout response");

    assert_eq!(response.exit_code, COMMAND_EXEC_TIMEOUT_EXIT_CODE);
}
