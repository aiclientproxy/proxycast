use super::*;
#[cfg(unix)]
use app_server_protocol::protocol::v2::ProcessTerminalSize;
#[cfg(unix)]
use tokio::sync::mpsc;

#[cfg(unix)]
fn server_with_notifications() -> (
    ProcessServer,
    mpsc::UnboundedReceiver<(ConnectionId, JsonRpcNotification)>,
) {
    let (tx, rx) = mpsc::unbounded_channel();
    let hook: ProcessNotificationHook = Arc::new(move |connection_id, notification| {
        let tx = tx.clone();
        Box::pin(async move {
            let _ = tx.send((connection_id, notification));
        })
    });
    (ProcessServer::default().with_notification_hook(hook), rx)
}

#[cfg(unix)]
fn spawn_params(process_handle: &str, script: &str) -> ProcessSpawnParams {
    ProcessSpawnParams {
        command: vec!["/bin/sh".to_string(), "-c".to_string(), script.to_string()],
        process_handle: process_handle.to_string(),
        cwd: std::env::current_dir()
            .expect("current directory")
            .to_string_lossy()
            .to_string(),
        tty: false,
        stream_stdin: false,
        stream_stdout_stderr: false,
        output_bytes_cap: Some(Some(1024)),
        timeout_ms: Some(Some(2_000)),
        env: None,
        size: None,
    }
}

#[cfg(unix)]
#[tokio::test]
async fn spawn_waits_for_activation_before_sending_notifications() {
    let (server, mut notifications) = server_with_notifications();
    let connection_id = ConnectionId(1);
    let process_handle = "response-barrier";
    let mut params = spawn_params(process_handle, "printf ready");
    params.stream_stdout_stderr = true;

    server
        .spawn(connection_id, params)
        .await
        .expect("spawn process");
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(50), notifications.recv())
            .await
            .is_err(),
        "process notifications must wait until the response is queued"
    );

    server.activate(connection_id, process_handle).await;
    let (_, output) = tokio::time::timeout(std::time::Duration::from_secs(2), notifications.recv())
        .await
        .expect("output timeout")
        .expect("output notification");
    let (_, exited) = tokio::time::timeout(std::time::Duration::from_secs(2), notifications.recv())
        .await
        .expect("exit timeout")
        .expect("exit notification");

    assert_eq!(output.method, "process/outputDelta");
    assert_eq!(exited.method, "process/exited");
}

#[cfg(unix)]
#[tokio::test]
async fn stdin_close_drains_bytes_before_process_exit() {
    let (server, mut notifications) = server_with_notifications();
    let connection_id = ConnectionId(2);
    let process_handle = "stdin-close";
    let mut params = spawn_params(process_handle, "cat");
    params.stream_stdin = true;

    server
        .spawn(connection_id, params)
        .await
        .expect("spawn process");
    server.activate(connection_id, process_handle).await;
    server
        .write_stdin(
            connection_id,
            ProcessWriteStdinParams {
                process_handle: process_handle.to_string(),
                delta_base64: Some(STANDARD.encode(b"hello")),
                close_stdin: true,
            },
        )
        .await
        .expect("write and close stdin");
    let closed = server
        .write_stdin(
            connection_id,
            ProcessWriteStdinParams {
                process_handle: process_handle.to_string(),
                delta_base64: Some(STANDARD.encode(b"late")),
                close_stdin: false,
            },
        )
        .await
        .expect_err("non-empty writes after close must fail");
    assert_eq!(closed.code, error_codes::INVALID_REQUEST);

    let (_, notification) =
        tokio::time::timeout(std::time::Duration::from_secs(2), notifications.recv())
            .await
            .expect("exit timeout")
            .expect("exit notification");
    let ServerNotification::ProcessExited(exited) =
        ServerNotification::try_from(notification).expect("typed notification")
    else {
        panic!("expected process/exited notification");
    };
    assert_eq!(exited.exit_code, 0);
    assert_eq!(exited.stdout, "hello");
    assert!(!exited.stdout_cap_reached);
}

#[cfg(unix)]
#[tokio::test]
async fn process_handles_are_unique_per_connection() {
    let server = ProcessServer::default();
    let first_connection = ConnectionId(3);
    let second_connection = ConnectionId(4);

    server
        .spawn(first_connection, spawn_params("shared", "exit 0"))
        .await
        .expect("first spawn");
    let duplicate = server
        .spawn(first_connection, spawn_params("shared", "exit 0"))
        .await
        .expect_err("same-connection duplicate must fail");
    assert_eq!(duplicate.code, error_codes::INVALID_REQUEST);
    server
        .spawn(second_connection, spawn_params("shared", "exit 0"))
        .await
        .expect("different connection may reuse handle");

    server.connection_closed(first_connection).await;
    server.connection_closed(second_connection).await;
}

#[tokio::test]
async fn controls_fail_closed_for_unknown_or_invalid_sessions() {
    let server = ProcessServer::default();
    let connection_id = ConnectionId(5);
    let unknown = server
        .kill(
            connection_id,
            ProcessKillParams {
                process_handle: "missing".to_string(),
            },
        )
        .await
        .expect_err("unknown handle must fail");
    assert_eq!(unknown.code, error_codes::INVALID_REQUEST);

    #[cfg(unix)]
    {
        let mut params = spawn_params("invalid-size", "exit 0");
        params.tty = true;
        params.size = Some(ProcessTerminalSize { rows: 0, cols: 80 });
        let error = server
            .spawn(connection_id, params)
            .await
            .expect_err("zero terminal size must fail");
        assert_eq!(error.code, error_codes::INVALID_PARAMS);
    }
}
