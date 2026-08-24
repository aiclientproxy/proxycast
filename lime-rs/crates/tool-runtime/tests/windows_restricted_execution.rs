#![cfg(windows)]

use std::path::Path;
use std::time::Duration;

use app_server_protocol::protocol::v2::GrantedPermissionProfile;
use tool_runtime::execution_process::{
    start_local_execution_process, ExecutionProcessStatus, LocalExecutionProcessHandle,
    LocalExecutionRequest, LocalExecutionSandbox,
};
use tool_runtime::sandbox::SandboxBackend;

fn powershell_script(script: &str) -> Vec<String> {
    vec![
        "powershell.exe".to_string(),
        "-NoProfile".to_string(),
        "-NonInteractive".to_string(),
        "-Command".to_string(),
        script.to_string(),
    ]
}

fn restricted_request(root: &Path, command: Vec<String>) -> LocalExecutionRequest {
    let mut request = LocalExecutionRequest::new(
        format!("windows-restricted-{}", uuid::Uuid::new_v4()),
        "windows-restricted-test",
        "exec_command",
        command,
    );
    request.cwd = Some(root.to_path_buf());
    request.sandbox = Some(LocalExecutionSandbox {
        backend: SandboxBackend::RestrictedToken,
        requested_policy: Some("workspace-write".to_string()),
        granted_permissions: Some(GrantedPermissionProfile::default()),
    });
    request
}

async fn run_to_terminal(
    mut handle: LocalExecutionProcessHandle,
) -> tool_runtime::execution_process::ExecutionProcessSnapshot {
    tokio::time::timeout(Duration::from_secs(20), handle.wait())
        .await
        .expect("restricted process should reach a terminal state")
        .expect("restricted process supervisor should remain available")
}

fn ps_literal(path: &Path) -> String {
    path.to_string_lossy().replace('\'', "''")
}

#[tokio::test]
async fn workspace_write_allows_workspace_and_denies_metadata_and_external_paths() {
    let fixture = tempfile::tempdir().expect("fixture root");
    let workspace = fixture.path().join("workspace");
    let outside = fixture.path().join("outside.txt");
    std::fs::create_dir(&workspace).expect("workspace directory");
    for name in [".git", ".codex", ".agents"] {
        std::fs::create_dir(workspace.join(name)).expect("metadata directory");
    }

    let inside = workspace.join("inside.txt");
    let script = format!(
        "Set-Content -LiteralPath '{}' -Value allowed",
        ps_literal(&inside)
    );
    let snapshot = run_to_terminal(
        start_local_execution_process(restricted_request(&workspace, powershell_script(&script)))
            .expect("restricted workspace process should start"),
    )
    .await;
    assert_eq!(snapshot.status, ExecutionProcessStatus::Exited);
    assert!(inside.is_file(), "workspace write must be allowed");

    for (label, target) in [
        ("git", workspace.join(".git").join("blocked.txt")),
        ("codex", workspace.join(".codex").join("blocked.txt")),
        ("agents", workspace.join(".agents").join("blocked.txt")),
        ("outside", outside.clone()),
    ] {
        let script = format!(
            "$ErrorActionPreference='Stop'; Set-Content -LiteralPath '{}' -Value blocked",
            ps_literal(&target)
        );
        let snapshot = run_to_terminal(
            start_local_execution_process(restricted_request(
                &workspace,
                powershell_script(&script),
            ))
            .unwrap_or_else(|error| panic!("{label} denial process should start: {error}")),
        )
        .await;
        assert_eq!(
            snapshot.status,
            ExecutionProcessStatus::Exited,
            "denial should be observed as a normal process exit"
        );
        assert_ne!(
            snapshot.exit_code,
            Some(0),
            "denied write must fail the command"
        );
        assert!(!target.exists(), "{label} path must remain unwritable");
    }
}

#[tokio::test]
async fn restricted_execution_bounds_large_output() {
    let fixture = tempfile::tempdir().expect("fixture root");
    let script = "[Console]::Out.Write(('x' * 400000))";
    let snapshot = run_to_terminal(
        start_local_execution_process(restricted_request(
            fixture.path(),
            powershell_script(script),
        ))
        .expect("large-output process should start"),
    )
    .await;

    assert_eq!(snapshot.status, ExecutionProcessStatus::Exited);
    assert!(snapshot.output_truncated, "large output must be truncated");
    assert!(
        snapshot.output_omitted_bytes > 0,
        "omitted byte count must be reported"
    );
    assert!(snapshot.retained_output.len() <= 128 * 1024);
}

#[tokio::test]
async fn terminate_ends_restricted_process_and_its_job() {
    let fixture = tempfile::tempdir().expect("fixture root");
    let marker = fixture.path().join("descendant-marker.txt");
    let script = format!(
        "$child = Start-Process -FilePath powershell.exe -ArgumentList '-NoProfile','-Command','Start-Sleep -Seconds 10; Set-Content -LiteralPath ''{}'' -Value leaked' -PassThru; Start-Sleep -Seconds 10",
        ps_literal(&marker)
    );
    let mut handle = start_local_execution_process(restricted_request(
        fixture.path(),
        powershell_script(&script),
    ))
    .expect("long-running process should start");

    tokio::time::sleep(Duration::from_millis(250)).await;
    handle
        .terminate()
        .expect("terminate should reach supervisor");
    let snapshot = tokio::time::timeout(Duration::from_secs(10), handle.wait())
        .await
        .expect("terminated process should finish")
        .expect("terminated process supervisor should remain available");
    assert_eq!(snapshot.status, ExecutionProcessStatus::Terminated);
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(
        !marker.exists(),
        "job termination must clean up descendants"
    );
}
