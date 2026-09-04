use super::*;
use std::ffi::OsString;
use std::io::{Read, Write};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use portable_pty::{native_pty_system, CommandBuilder, PtySize};

#[test]
fn real_pty_restores_terminal_after_visible_turn_completion() {
    if std::env::var_os("LIME_TEST_TUI_GATE_B").is_none() {
        return;
    }
    let cli_bin = required_test_path("LIME_TEST_CLI_BIN");
    let app_server_bin = required_test_path("LIME_TEST_APP_SERVER_BIN");
    let backend_path = required_test_path("LIME_TEST_TERMINAL_BACKEND");
    let ledger_path = required_test_path("LIME_TEST_TERMINAL_LEDGER");
    let cwd = required_test_path("LIME_TEST_TERMINAL_CWD");
    let node_bin = required_test_path("LIME_TEST_NODE_BIN");
    let prompt = std::env::var("LIME_TEST_TERMINAL_PROMPT").expect("terminal prompt");
    let completed_text =
        std::env::var("LIME_TEST_TERMINAL_COMPLETED_TEXT").expect("completed text");
    let scenario =
        std::env::var("LIME_TEST_TERMINAL_SCENARIO").unwrap_or_else(|_| "complete".to_string());
    // Keep the interrupt backend alive long enough for the real PTY event
    // loop to deliver turn/interrupt before the fixture timeout fires.
    let backend_timeout_ms = if scenario == "interrupt" {
        "30000"
    } else {
        "5000"
    };
    let data_dir = cwd.join("data");
    let app_data_dir = cwd.join("app-data");

    let mut command = CommandBuilder::new(cli_bin);
    for argument in [
        OsString::from("tui"),
        OsString::from("--cwd"),
        cwd.as_os_str().to_os_string(),
        OsString::from("--model"),
        OsString::from("fixture-model"),
        OsString::from("--provider"),
        OsString::from("fixture-provider"),
        OsString::from("--app-server"),
        app_server_bin.as_os_str().to_os_string(),
        OsString::from("--app-server-arg=--backend"),
        OsString::from("--app-server-arg=external"),
        OsString::from("--app-server-arg=--backend-command"),
        OsString::from(format!("--app-server-arg={}", node_bin.to_string_lossy())),
        OsString::from("--app-server-arg=--backend-arg"),
        OsString::from(format!(
            "--app-server-arg={}",
            backend_path.to_string_lossy()
        )),
        OsString::from("--app-server-arg=--backend-arg"),
        OsString::from(format!(
            "--app-server-arg={}",
            ledger_path.to_string_lossy()
        )),
        OsString::from("--app-server-arg=--backend-timeout-ms"),
        OsString::from(format!("--app-server-arg={backend_timeout_ms}")),
        OsString::from("--app-server-arg=--data-dir"),
        OsString::from(format!("--app-server-arg={}", data_dir.to_string_lossy())),
        OsString::from("--app-server-arg=--app-data-dir"),
        OsString::from(format!(
            "--app-server-arg={}",
            app_data_dir.to_string_lossy()
        )),
    ] {
        command.arg(argument);
    }
    command.cwd(&cwd);
    command.env("TERM", "xterm-256color");

    let pair = native_pty_system()
        .openpty(PtySize {
            rows: 24,
            cols: 100,
            pixel_width: 0,
            pixel_height: 0,
        })
        .expect("open PTY");
    let mut reader = pair.master.try_clone_reader().expect("clone PTY reader");
    let mut writer = pair.master.take_writer().expect("take PTY writer");
    let mut child = pair.slave.spawn_command(command).expect("spawn lime TUI");
    drop(pair.slave);
    let master = pair.master;
    let (output_tx, output_rx) = mpsc::channel();
    let reader_thread = thread::spawn(move || {
        let mut buffer = [0_u8; 4096];
        loop {
            match reader.read(&mut buffer) {
                Ok(0) => break,
                Ok(read) => {
                    if output_tx.send(buffer[..read].to_vec()).is_err() {
                        break;
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            }
        }
    });
    let mut output = String::new();

    wait_for_marker(&output_rx, &mut output, "ready", Duration::from_secs(10));
    writer.write_all(prompt.as_bytes()).expect("write prompt");
    writer.write_all(b"\r").expect("submit prompt");
    writer.flush().expect("flush prompt");
    let marker = match scenario.as_str() {
        "approval" => "Allow terminal command?",
        // Cursor-addressed renders may split the full question across writes; the
        // stable question title is sufficient to prove the prompt is visible.
        "user-input" => "Choose",
        "interrupt" => "INTERRUPT_READY",
        "failure" => "fixture backend failure",
        _ => &completed_text,
    };
    let visible_result = match scenario.as_str() {
        "interrupt" => "INTERRUPT_READY",
        "failure" => "fixture backend failure",
        _ => &completed_text,
    };
    wait_for_marker(&output_rx, &mut output, marker, Duration::from_secs(10));
    if scenario == "approval" {
        writer.write_all(b"y").expect("approve command");
        writer.flush().expect("flush approval");
        wait_for_marker(
            &output_rx,
            &mut output,
            &completed_text,
            Duration::from_secs(10),
        );
    } else if scenario == "user-input" {
        writer.write_all(b"\r").expect("answer user input");
        writer.flush().expect("flush user input");
        wait_for_marker(
            &output_rx,
            &mut output,
            &completed_text,
            Duration::from_secs(10),
        );
    }
    let exit_key = if scenario == "interrupt" { 3 } else { 4 };
    writer.write_all(&[exit_key]).expect("exit TUI");
    writer.flush().expect("flush TUI exit");

    if scenario == "interrupt" {
        // Codex-style Ctrl-C first cancels the active turn. The canonical
        // interrupted projection proves the request crossed App Server;
        // backend cleanup is intentionally best-effort and asynchronous.
        wait_for_marker(
            &output_rx,
            &mut output,
            "interrupting",
            Duration::from_secs(5),
        );
        wait_for_ledger_kind(&ledger_path, "turnCancel", Duration::from_secs(5));
        writer.write_all(&[3]).expect("send quit Ctrl-C");
        writer.flush().expect("flush quit Ctrl-C");
    }

    let deadline = Instant::now() + Duration::from_secs(5);
    let status = loop {
        if let Some(status) = child.try_wait().expect("poll TUI process") {
            break status;
        }
        if Instant::now() >= deadline {
            child.kill().expect("terminate timed out TUI");
            panic!("TUI did not exit after Ctrl-C; output: {output}");
        }
        thread::sleep(Duration::from_millis(20));
    };
    drop(writer);
    drop(master);
    reader_thread.join().expect("join PTY reader");
    while let Ok(chunk) = output_rx.try_recv() {
        output.push_str(&String::from_utf8_lossy(&chunk));
    }

    assert!(status.success(), "TUI exited with {status:?}: {output}");
    assert!(
        output.contains("\u{1b}[?1049h"),
        "alternate screen not entered"
    );
    assert!(
        visible_terminal_text(&output).contains(visible_result),
        "terminal result was not visible"
    );
    if scenario == "interrupt" {
        assert!(
            visible_terminal_text(&output).contains("interrupting"),
            "interrupt action was not rendered"
        );
    }
    assert!(
        output.contains("\u{1b}[?1049l"),
        "alternate screen not restored"
    );
}

fn required_test_path(name: &str) -> PathBuf {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("missing {name}"))
}

fn wait_for_marker(
    output_rx: &mpsc::Receiver<Vec<u8>>,
    output: &mut String,
    marker: &str,
    timeout: Duration,
) {
    let deadline = Instant::now() + timeout;
    while !visible_terminal_text(output).contains(marker) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            panic!("timed out waiting for {marker:?}; output: {output}");
        }
        let chunk = output_rx
            .recv_timeout(remaining)
            .unwrap_or_else(|_| panic!("PTY closed before {marker:?}; output: {output}"));
        output.push_str(&String::from_utf8_lossy(&chunk));
    }
}

fn wait_for_ledger_kind(path: &PathBuf, kind: &str, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    loop {
        if let Ok(contents) = std::fs::read_to_string(path) {
            let found = contents
                .lines()
                .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
                .any(|entry| entry.get("kind").and_then(serde_json::Value::as_str) == Some(kind));
            if found {
                return;
            }
        }
        if Instant::now() >= deadline {
            panic!("timed out waiting for backend ledger kind {kind:?}");
        }
        thread::sleep(Duration::from_millis(20));
    }
}

fn visible_terminal_text(output: &str) -> String {
    let mut visible = String::new();
    let mut chars = output.chars();
    while let Some(character) = chars.next() {
        if character != '\u{1b}' {
            visible.push(character);
            continue;
        }
        let Some(control) = chars.next() else {
            break;
        };
        match control {
            '[' => {
                for character in chars.by_ref() {
                    if ('@'..='~').contains(&character) {
                        break;
                    }
                }
            }
            ']' => {
                let mut previous = '\0';
                for character in chars.by_ref() {
                    if character == '\u{7}' || (previous == '\u{1b}' && character == '\\') {
                        break;
                    }
                    previous = character;
                }
            }
            _ => {}
        }
    }
    visible
}

#[test]
fn visible_terminal_text_ignores_cursor_controls() {
    let output = "Choose\u{1b}[19;8Ha\u{1b}[19;10H mode";
    assert_eq!(visible_terminal_text(output), "Choosea mode");
}
