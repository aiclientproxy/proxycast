#![cfg(windows)]

use std::path::Path;
use std::process::Command;
use std::ptr;
use std::time::Duration;
use std::{collections::HashMap, collections::HashSet, env};

use app_server_protocol::protocol::v2::{AdditionalNetworkPermissions, GrantedPermissionProfile};
use tool_runtime::execution_process::{
    audit_windows_world_writable, start_local_execution_process, ExecutionProcessStatus,
    LocalExecutionProcessHandle, LocalExecutionRequest, LocalExecutionSandbox,
};
use tool_runtime::sandbox::SandboxBackend;
use windows_sys::Win32::Foundation::{LocalFree, ERROR_SUCCESS, HLOCAL};
use windows_sys::Win32::Security::Authorization::{
    ConvertSecurityDescriptorToStringSecurityDescriptorW, GetNamedSecurityInfoW,
};
use windows_sys::Win32::Security::{
    DACL_SECURITY_INFORMATION, GROUP_SECURITY_INFORMATION, OWNER_SECURITY_INFORMATION,
};

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
        granted_permissions: Some(GrantedPermissionProfile {
            network: Some(AdditionalNetworkPermissions {
                enabled: Some(true),
            }),
            file_system: None,
        }),
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

fn acl_sddl(path: &Path) -> String {
    let wide = path
        .as_os_str()
        .to_string_lossy()
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    unsafe {
        let mut descriptor = ptr::null_mut();
        let mut dacl = ptr::null_mut();
        let result = GetNamedSecurityInfoW(
            wide.as_ptr(),
            1,
            OWNER_SECURITY_INFORMATION | GROUP_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
            ptr::null_mut(),
            ptr::null_mut(),
            &mut dacl,
            ptr::null_mut(),
            &mut descriptor,
        );
        assert_eq!(
            result,
            ERROR_SUCCESS,
            "ACL SDDL query should succeed for {}",
            path.display()
        );
        let mut text = ptr::null_mut();
        let mut length = 0;
        let converted = ConvertSecurityDescriptorToStringSecurityDescriptorW(
            descriptor,
            1,
            OWNER_SECURITY_INFORMATION | GROUP_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
            &mut text,
            &mut length,
        );
        assert_ne!(
            converted,
            0,
            "ACL SDDL conversion should succeed for {}",
            path.display()
        );
        let value = String::from_utf16_lossy(std::slice::from_raw_parts(text, length as usize));
        LocalFree(text as HLOCAL);
        LocalFree(descriptor as HLOCAL);
        value
    }
}

fn account_sid(account: &str) -> String {
    let script = format!(
        "(New-Object Security.Principal.NTAccount('{}')).Translate([Security.Principal.SecurityIdentifier]).Value",
        account.replace('\'', "''")
    );
    let output = Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", &script])
        .output()
        .expect("account SID query should start");
    assert!(
        output.status.success(),
        "account SID query should succeed for {account}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn sddl_sids(sddl: &str) -> HashSet<String> {
    regex::Regex::new(r"S-1-[0-9]+(?:-[0-9]+)+")
        .expect("SID regex")
        .find_iter(sddl)
        .map(|value| value.as_str().to_string())
        .collect()
}

async fn wait_for_path(handle: &LocalExecutionProcessHandle, path: &Path) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while !path.exists() {
            let snapshot = handle.status();
            assert!(
                !snapshot.status.is_terminal(),
                "restricted process exited before creating {}: {snapshot:#?}",
                path.display()
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .unwrap_or_else(|_| {
        panic!(
            "path should be created by restricted process: {}",
            path.display()
        )
    });
}

async fn wait_for_acl_restore(path: &Path, expected: &str) {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if acl_sddl(path) == expected {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("ACL should be restored for {}", path.display()));
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
    let acl_baselines = [
        workspace.clone(),
        workspace.join(".git"),
        workspace.join(".codex"),
        workspace.join(".agents"),
    ]
    .map(|path| {
        let sddl = acl_sddl(&path);
        (path, sddl)
    });

    let native_probe = workspace.join("native-probe.txt");
    let native_probe_snapshot = run_to_terminal(
        start_local_execution_process(restricted_request(
            &workspace,
            vec![
                "cmd.exe".to_string(),
                "/D".to_string(),
                "/S".to_string(),
                "/C".to_string(),
                "echo allowed>native-probe.txt".to_string(),
            ],
        ))
        .expect("restricted native workspace probe should start"),
    )
    .await;
    assert_eq!(
        native_probe_snapshot.exit_code,
        Some(0),
        "restricted native workspace probe must exit successfully: {native_probe_snapshot:#?}"
    );
    assert!(
        native_probe.is_file(),
        "restricted native workspace probe must create {}",
        native_probe.display()
    );
    std::fs::remove_file(&native_probe).expect("remove native workspace probe");

    let inside = workspace.join("inside.txt");
    let script = format!(
        "Write-Output ('identity=' + [Security.Principal.WindowsIdentity]::GetCurrent().Name); Set-Content -LiteralPath '{}' -Value allowed; [Console]::In.ReadLine() | Out-Null",
        ps_literal(&inside)
    );
    let mut request = restricted_request(&workspace, powershell_script(&script));
    request.stdin = true;
    let mut handle =
        start_local_execution_process(request).expect("restricted workspace process should start");
    let baseline_workspace_sddl = &acl_baselines[0].1;
    let active_workspace_sddl = acl_sddl(&workspace);
    eprintln!("active restricted workspace SDDL: {active_workspace_sddl}");
    let group_sid = account_sid("LimeSandboxUsers");
    let added_sids = sddl_sids(&active_workspace_sddl)
        .difference(&sddl_sids(baseline_workspace_sddl))
        .cloned()
        .collect::<HashSet<_>>();
    assert!(
        added_sids.contains(&group_sid),
        "active workspace ACL must grant the ordinary token through {group_sid}: {active_workspace_sddl}"
    );
    assert!(
        added_sids.iter().any(|sid| sid != &group_sid),
        "active workspace ACL must also grant a short-lived capability SID: {active_workspace_sddl}"
    );
    wait_for_path(&handle, &inside).await;

    handle
        .close_stdin()
        .expect("workspace process stdin should close");
    let snapshot = run_to_terminal(handle).await;
    assert_eq!(snapshot.status, ExecutionProcessStatus::Exited);
    assert!(
        snapshot
            .retained_output
            .to_ascii_lowercase()
            .contains("limesandboxonline"),
        "network-enabled execution must use the online sandbox account: {:?}",
        snapshot.retained_output
    );
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

    for (path, baseline) in acl_baselines {
        wait_for_acl_restore(&path, &baseline).await;
    }
}

#[tokio::test]
async fn restricted_execution_uses_offline_account_and_blocks_network() {
    let fixture = tempfile::tempdir().expect("fixture root");
    let marker = fixture.path().join("offline-process-ran.txt");
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("loopback listener");
    let port = listener.local_addr().expect("listener address").port();
    tool_runtime::windows_firewall::verify_offline_rules("LimeSandboxOffline")
        .expect("offline firewall rules must pass read-back before execution");
    let mut request = restricted_request(
        fixture.path(),
        powershell_script(&format!(
            "$ErrorActionPreference='Stop'; Write-Output ('identity=' + [Security.Principal.WindowsIdentity]::GetCurrent().Name); Set-Content -LiteralPath '{}' -Value ran; $client = [Net.Sockets.TcpClient]::new(); try {{ $task = $client.ConnectAsync('127.0.0.1', {}); if (-not $task.Wait(2000)) {{ throw 'connect timeout' }}; if ($client.Connected) {{ Write-Output 'network-connected'; exit 9 }} }} catch {{ Write-Output 'network-blocked' }} finally {{ $client.Dispose() }}",
            ps_literal(&marker),
            port
        )),
    );
    request
        .sandbox
        .as_mut()
        .expect("restricted sandbox")
        .granted_permissions = Some(GrantedPermissionProfile::default());

    let snapshot = run_to_terminal(
        start_local_execution_process(request)
            .expect("network-restricted process should use the enforced offline identity"),
    )
    .await;
    assert_eq!(snapshot.status, ExecutionProcessStatus::Exited);
    assert_eq!(
        snapshot.exit_code,
        Some(0),
        "offline restricted process must exit successfully: {snapshot:#?}"
    );
    assert!(
        marker.is_file(),
        "offline process must execute inside the workspace"
    );
    assert!(
        snapshot
            .retained_output
            .to_ascii_lowercase()
            .contains("limesandboxoffline"),
        "network-restricted execution must use the offline account: {:?}",
        snapshot.retained_output
    );
    assert!(
        snapshot.retained_output.contains("network-blocked"),
        "offline account must not connect to loopback: {:?}",
        snapshot.retained_output
    );
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
    assert_eq!(
        snapshot.exit_code,
        Some(0),
        "large-output restricted process must exit successfully: {snapshot:#?}"
    );
    assert!(snapshot.output_truncated, "large output must be truncated");
    assert!(
        snapshot.output_omitted_bytes > 0,
        "omitted byte count must be reported"
    );
    assert!(snapshot.retained_output.len() <= 128 * 1024);
}

#[tokio::test]
async fn restricted_execution_preserves_allowlisted_stdin_handle() {
    let fixture = tempfile::tempdir().expect("fixture root");
    let mut handle = start_local_execution_process(restricted_request(
        fixture.path(),
        powershell_script("$line = [Console]::In.ReadLine(); Write-Output ('received:' + $line)"),
    ))
    .expect("stdin process should start");

    handle
        .write_stdin(b"sandbox-input\n".to_vec())
        .expect("stdin should be writable through the restricted handle");
    handle.close_stdin().expect("stdin should close cleanly");
    let snapshot = run_to_terminal(handle).await;

    assert_eq!(snapshot.status, ExecutionProcessStatus::Exited);
    assert_eq!(
        snapshot.exit_code,
        Some(0),
        "stdin restricted process must exit successfully: {snapshot:#?}"
    );
    assert!(
        snapshot.retained_output.contains("received:sandbox-input"),
        "child output should contain the stdin payload: {:?}",
        snapshot.retained_output
    );
}

#[tokio::test]
async fn restricted_conpty_supports_stdin_resize_and_combined_output() {
    let fixture = tempfile::tempdir().expect("fixture root");
    let mut request = restricted_request(
        fixture.path(),
        powershell_script(
            "$line = [Console]::In.ReadLine(); [Console]::Out.WriteLine('out:' + $line); [Console]::Error.WriteLine('err:' + $line)",
        ),
    );
    request.tty = true;
    request.stdin = true;
    request.pty_size = Some((24, 80));
    let mut handle =
        start_local_execution_process(request).expect("restricted ConPTY process should start");

    handle
        .resize(40, 132)
        .expect("ConPTY resize should reach the restricted supervisor");
    handle
        .write_stdin(b"conpty-input\r\n".to_vec())
        .expect("ConPTY stdin should be writable");
    handle.close_stdin().expect("ConPTY stdin should close");
    let snapshot = run_to_terminal(handle).await;

    assert_eq!(snapshot.status, ExecutionProcessStatus::Exited);
    assert_eq!(snapshot.exit_code, Some(0));
    assert!(
        snapshot.retained_output.contains("out:conpty-input"),
        "ConPTY output should contain stdout: {:?}",
        snapshot.retained_output
    );
    assert!(
        snapshot.retained_output.contains("err:conpty-input"),
        "ConPTY output should contain stderr on the combined stream: {:?}",
        snapshot.retained_output
    );
}

#[tokio::test]
async fn world_writable_audit_reports_everyone_write_acl() {
    let fixture = tempfile::tempdir().expect("fixture root");
    let world_writable = fixture.path().join("world-writable");
    std::fs::create_dir(&world_writable).expect("world-writable directory");
    let world_writable_text = world_writable.to_string_lossy().to_string();
    let grant = Command::new("icacls.exe")
        .args([
            world_writable_text.as_str(),
            "/grant",
            "Everyone:(OI)(CI)(M)",
        ])
        .output()
        .expect("icacls should start");
    assert!(
        grant.status.success(),
        "icacls grant should succeed: stdout={}, stderr={}",
        String::from_utf8_lossy(&grant.stdout),
        String::from_utf8_lossy(&grant.stderr)
    );

    let environment = env::vars().collect::<HashMap<_, _>>();
    let audit = audit_windows_world_writable(fixture.path(), &environment);
    assert!(
        audit
            .sample_paths
            .iter()
            .any(|path| Path::new(path) == world_writable.as_path()),
        "audit should report Everyone-writable fixture: {audit:?}"
    );
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
    let before_terminate = handle.status();
    assert!(
        !before_terminate.status.is_terminal(),
        "terminate target exited before control could be sent: {before_terminate:#?}"
    );
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
