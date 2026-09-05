use std::ffi::OsString;
use std::io::{Read, Write};
use std::path::Path;
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use portable_pty::{CommandBuilder, PtySize, native_pty_system};

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
    let queue_prompt = std::env::var("LIME_TEST_TERMINAL_QUEUE_PROMPT")
        .unwrap_or_else(|_| "queued follow-up for editing".to_string());
    let completed_text =
        std::env::var("LIME_TEST_TERMINAL_COMPLETED_TEXT").expect("completed text");
    let permission_config = std::env::var_os("LIME_TEST_PERMISSION_CONFIG");
    let expected_permission_profile = std::env::var("LIME_TEST_PERMISSION_PROFILE").ok();
    let scenario =
        std::env::var("LIME_TEST_TERMINAL_SCENARIO").unwrap_or_else(|_| "complete".to_string());
    // Keep the interrupt backend alive long enough for the real PTY event
    // loop to deliver turn/interrupt before the fixture timeout fires.
    let backend_timeout_ms = if matches!(scenario.as_str(), "interrupt" | "queue-edit") {
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
    command.env("LIME_LOCALE", "en-US");
    if let Some(permission_config) = permission_config {
        command.env("LIME_CONFIG_PATH", permission_config);
    }
    if scenario == "complete" {
        configure_external_editor(&mut command, &cwd, &prompt);
    }

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
    if scenario == "complete" {
        writer.write_all(b"/status\r").expect("open status pager");
        writer.flush().expect("flush status command");
        wait_for_marker(&output_rx, &mut output, "/ STATUS", Duration::from_secs(10));
        let return_to_composer_at = output.len();
        writer.write_all(b"q").expect("close status pager");
        writer.flush().expect("flush status pager close");
        wait_for_marker_after(
            &output_rx,
            &mut output,
            return_to_composer_at,
            "ready",
            Duration::from_secs(10),
        );
        writer
            .write_all(b"before external edit")
            .expect("write draft");
        writer.write_all(&[7]).expect("open external editor");
        writer.flush().expect("flush editor shortcut");
        wait_for_marker(
            &output_rx,
            &mut output,
            "EDITOR_JOB_CONTROL_OK",
            Duration::from_secs(10),
        );
        // Re-entering the alternate screen must not synchronously query the PTY. The next
        // draw is responsible for reconciling the restored surface and showing the composer.
        wait_for_marker(&output_rx, &mut output, "prompt", Duration::from_secs(10));
    } else {
        writer.write_all(prompt.as_bytes()).expect("write prompt");
    }
    writer.write_all(b"\r").expect("submit prompt");
    writer.flush().expect("flush prompt");
    let marker = match scenario.as_str() {
        "approval" => "Allow terminal command?",
        // Cursor-addressed renders may split the full question across writes; the
        // stable question title is sufficient to prove the prompt is visible.
        "user-input" => "Choose",
        "interrupt" => "INTERRUPT_READY",
        "queue-edit" => "QUEUE_EDIT_READY",
        "failure" => "fixture backend failure",
        _ => &completed_text,
    };
    let visible_result = match scenario.as_str() {
        "interrupt" => "INTERRUPT_READY",
        "queue-edit" => &queue_prompt,
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
    if scenario == "complete" {
        let transcript_at = output.len();
        writer.write_all(&[20]).expect("open transcript Ctrl-T");
        writer.flush().expect("flush transcript shortcut");
        wait_for_marker_after(
            &output_rx,
            &mut output,
            transcript_at,
            "Ctrl+T/Esc/Q close",
            Duration::from_secs(10),
        );
        wait_for_marker_after(
            &output_rx,
            &mut output,
            transcript_at,
            &completed_text,
            Duration::from_secs(10),
        );
        writer.write_all(&[20]).expect("close transcript Ctrl-T");
        writer.flush().expect("flush transcript close");
    }
    if scenario == "queue-edit" {
        let queued_at = output.len();
        writer
            .write_all(queue_prompt.as_bytes())
            .expect("write queued follow-up");
        writer.write_all(b"\t").expect("queue follow-up with Tab");
        writer.flush().expect("flush queued follow-up");
        wait_for_marker_after(
            &output_rx,
            &mut output,
            queued_at,
            "queued (1)",
            Duration::from_secs(10),
        );
        wait_for_marker_after(
            &output_rx,
            &mut output,
            queued_at,
            &queue_prompt,
            Duration::from_secs(10),
        );
        wait_for_marker_after(
            &output_rx,
            &mut output,
            queued_at,
            "Alt+Up edit last queued input",
            Duration::from_secs(10),
        );

        let edit_at = output.len();
        writer
            .write_all(b"\x1b[1;3A")
            .expect("edit queued follow-up with Alt-Up");
        writer.flush().expect("flush Alt-Up queue edit");
        wait_for_marker_after(
            &output_rx,
            &mut output,
            edit_at,
            "editing queued",
            Duration::from_secs(10),
        );
        wait_for_marker_after(
            &output_rx,
            &mut output,
            edit_at,
            "follow-up",
            Duration::from_secs(10),
        );
    }
    if matches!(scenario.as_str(), "interrupt" | "queue-edit") {
        wait_for_marker(
            &output_rx,
            &mut output,
            "esc to interrupt",
            Duration::from_secs(5),
        );
        writer.write_all(b"\x1b").expect("interrupt with Escape");
        writer.flush().expect("flush Escape interrupt");
        wait_for_marker(
            &output_rx,
            &mut output,
            "interrupting",
            Duration::from_secs(5),
        );
        wait_for_ledger_kind_and_scenario(
            &ledger_path,
            "turnCancel",
            &scenario,
            Duration::from_secs(5),
        );
        writer.write_all(&[3]).expect("send quit Ctrl-C");
        writer.flush().expect("flush quit Ctrl-C");
    } else {
        writer.write_all(&[4]).expect("exit TUI");
        writer.flush().expect("flush TUI exit");
    }

    let deadline = Instant::now() + Duration::from_secs(5);
    let status = loop {
        if let Some(status) = child.try_wait().expect("poll TUI process") {
            break status;
        }
        if Instant::now() >= deadline {
            child.kill().expect("terminate timed out TUI");
            panic!("TUI did not exit after terminal exit input; output: {output}");
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
        assert!(
            visible_terminal_text(&output).contains("esc to interrupt"),
            "Codex-style Escape interrupt hint was not rendered"
        );
    }
    if scenario == "queue-edit" {
        let visible = visible_terminal_text(&output);
        assert!(
            visible.contains("queued (1)"),
            "canonical queued preview was not rendered"
        );
        assert!(
            visible.contains("Alt+Up edit last queued input"),
            "queue edit affordance was not rendered"
        );
        assert!(
            visible.contains("editing queued"),
            "queued input was not restored to the composer"
        );
    }
    assert!(
        output.contains("\u{1b}[?1049l"),
        "alternate screen not restored"
    );
    if scenario == "complete" {
        assert!(
            output.contains("EDITOR_JOB_CONTROL_OK"),
            "external editor did not inherit the PTY"
        );
        assert!(
            visible_terminal_text(&output).contains("/ STATUS"),
            "status pager was not visible"
        );
        if let Some(expected_permission_profile) = expected_permission_profile.as_deref() {
            assert!(
                visible_terminal_text(&output).contains(expected_permission_profile),
                "configured permission profile was not visible: {expected_permission_profile}"
            );
        }
        assert!(
            visible_terminal_text(&output).contains("Ctrl+T/Esc/Q close"),
            "transcript overlay was not visible"
        );
    }
}

#[cfg(unix)]
fn configure_external_editor(command: &mut CommandBuilder, cwd: &Path, prompt: &str) {
    use std::os::unix::fs::PermissionsExt;

    let script = cwd.join("tui-editor.sh");
    std::fs::write(
        &script,
        "#!/bin/sh\ntest -t 0 && test -t 1 && test -t 2 || exit 9\nprintf '%s' \"$LIME_TEST_EDITOR_REPLACEMENT\" > \"$1\"\nprintf 'EDITOR_JOB_CONTROL_OK\\n'\n",
    )
    .expect("write editor fixture");
    let mut permissions = std::fs::metadata(&script)
        .expect("editor fixture metadata")
        .permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&script, permissions).expect("editor fixture permissions");
    command.env("VISUAL", script);
    command.env("LIME_TEST_EDITOR_REPLACEMENT", prompt);
}

#[cfg(windows)]
fn configure_external_editor(command: &mut CommandBuilder, cwd: &Path, prompt: &str) {
    let script = cwd.join("tui-editor.cmd");
    std::fs::write(
        &script,
        "@echo off\r\n<nul set /p \"=%LIME_TEST_EDITOR_REPLACEMENT%\" > \"%~1\"\r\necho EDITOR_JOB_CONTROL_OK\r\n",
    )
    .expect("write editor fixture");
    command.env("VISUAL", script);
    command.env("LIME_TEST_EDITOR_REPLACEMENT", prompt);
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

fn wait_for_marker_after(
    output_rx: &mpsc::Receiver<Vec<u8>>,
    output: &mut String,
    start: usize,
    marker: &str,
    timeout: Duration,
) {
    let deadline = Instant::now() + timeout;
    loop {
        let recent = output.get(start..).unwrap_or_default();
        if visible_terminal_text(recent).contains(marker) {
            return;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            panic!("timed out waiting for new {marker:?}; output: {output}");
        }
        let chunk = output_rx
            .recv_timeout(remaining)
            .unwrap_or_else(|_| panic!("PTY closed before new {marker:?}; output: {output}"));
        output.push_str(&String::from_utf8_lossy(&chunk));
    }
}

fn wait_for_ledger_kind_and_scenario(
    path: &PathBuf,
    kind: &str,
    scenario: &str,
    timeout: Duration,
) {
    let deadline = Instant::now() + timeout;
    loop {
        if let Ok(contents) = std::fs::read_to_string(path) {
            let found = contents
                .lines()
                .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
                .any(|entry| {
                    entry.get("kind").and_then(serde_json::Value::as_str) == Some(kind)
                        && entry.get("scenario").and_then(serde_json::Value::as_str)
                            == Some(scenario)
                });
            if found {
                return;
            }
        }
        if Instant::now() >= deadline {
            panic!("timed out waiting for backend ledger kind {kind:?} in scenario {scenario:?}");
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
