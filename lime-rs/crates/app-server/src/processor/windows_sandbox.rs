use super::{dispatch_result, parse_params, ConnectionRequestId, RequestProcessor, RpcDispatch};
use app_server_protocol::protocol::v2::ServerNotification;
use app_server_protocol::protocol::v2::{
    WindowsSandboxReadiness, WindowsSandboxReadinessParams, WindowsSandboxReadinessResponse,
    WindowsSandboxSetupCompletedNotification, WindowsSandboxSetupMode,
    WindowsSandboxSetupStartParams, WindowsSandboxSetupStartResponse,
    WindowsWorldWritableWarningNotification,
};
use app_server_protocol::{error_codes, JsonRpcError};
use lime_core::config::{load_config, Config};
use serde_json::Value;
use std::path::PathBuf;
#[cfg(windows)]
use std::process::{Command, Stdio};
#[cfg(windows)]
use std::time::{Duration, Instant};
use tool_runtime::execution_policy::ToolExecutionSandboxProfile;
use tool_runtime::execution_process::{audit_windows_world_writable, WindowsWorldWritableAudit};
use tool_runtime::sandbox::{
    plan_sandbox_backend, SandboxBackendPlanInput, SandboxBackendPlatform, SandboxBackendStatus,
};
use tool_runtime::windows_setup::inspect_default_windows_sandbox_setup;

impl RequestProcessor {
    pub(super) async fn handle_windows_sandbox_readiness_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let _: WindowsSandboxReadinessParams = parse_params(params)?;
        let config = load_config().map_err(|error| {
            JsonRpcError::new(
                error_codes::RUNTIME_ERROR,
                format!("failed to load Lime config for Windows sandbox readiness: {error}"),
            )
        })?;
        dispatch_result(WindowsSandboxReadinessResponse {
            status: windows_sandbox_readiness(&config, SandboxBackendPlatform::current()),
        })
    }

    pub(super) async fn handle_windows_sandbox_setup_start_impl(
        &self,
        params: Option<Value>,
        connection_request_id: Option<ConnectionRequestId>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: WindowsSandboxSetupStartParams = parse_params(params)?;
        let connection_id = connection_request_id
            .map(|request| request.connection_id)
            .ok_or_else(|| {
                JsonRpcError::new(
                    error_codes::INVALID_REQUEST,
                    "windowsSandbox/setupStart requires a transport connection",
                )
            })?;
        if let Some(cwd) = params.cwd.as_ref() {
            if !cwd.is_absolute() {
                return Err(JsonRpcError::new(
                    error_codes::INVALID_REQUEST,
                    "windowsSandbox/setupStart cwd must be absolute",
                ));
            }
        }

        let config = load_config().map_err(|error| {
            JsonRpcError::new(
                error_codes::RUNTIME_ERROR,
                format!("failed to load Lime config for Windows sandbox setup: {error}"),
            )
        })?;
        let cwd = params.cwd.clone().or_else(|| std::env::current_dir().ok());
        let mode = params.mode;
        let setup_config = config.clone();
        let audit_cwd = cwd.clone();
        let processor = self.clone();
        tokio::spawn(async move {
            let setup_error = tokio::task::spawn_blocking(move || {
                windows_sandbox_setup_for_request(&setup_config, mode)
            })
            .await
            .unwrap_or_else(|_| Some("Windows sandbox setup worker panicked".to_string()));
            let audit = if let Some(audit_cwd) = audit_cwd {
                let environment = std::env::vars().collect::<std::collections::HashMap<_, _>>();
                tokio::task::spawn_blocking(move || {
                    audit_windows_world_writable(&audit_cwd, &environment)
                })
                .await
                .unwrap_or_else(|_| WindowsWorldWritableAudit {
                    sample_paths: Vec::new(),
                    extra_count: 0,
                    failed_scan: true,
                })
            } else {
                WindowsWorldWritableAudit {
                    sample_paths: Vec::new(),
                    extra_count: 0,
                    failed_scan: true,
                }
            };
            if audit.failed_scan || !audit.sample_paths.is_empty() || audit.extra_count > 0 {
                processor
                    .publish_connection_server_notification(
                        connection_id,
                        ServerNotification::WindowsWorldWritableWarning(
                            WindowsWorldWritableWarningNotification {
                                sample_paths: audit.sample_paths,
                                extra_count: audit.extra_count,
                                failed_scan: audit.failed_scan,
                            },
                        ),
                    )
                    .await;
            }
            let notification = WindowsSandboxSetupCompletedNotification {
                mode,
                success: setup_error.is_none(),
                error: setup_error,
            };
            processor
                .publish_connection_server_notification(
                    connection_id,
                    ServerNotification::WindowsSandboxSetupCompleted(notification),
                )
                .await;
        });

        dispatch_result(WindowsSandboxSetupStartResponse { started: true })
    }
}

fn windows_sandbox_setup_for_request(
    config: &Config,
    mode: WindowsSandboxSetupMode,
) -> Option<String> {
    if SandboxBackendPlatform::current() != SandboxBackendPlatform::Windows {
        return windows_sandbox_setup_error(config, None);
    }
    if !config.agent.workspace_sandbox.enabled {
        return Some("workspace sandbox is disabled".to_string());
    }

    let agent_root = match lime_core::app_paths::preferred_agent_root() {
        Ok(root) => root,
        Err(error) => return Some(format!("failed to resolve the sandbox data root: {error}")),
    };
    if let Err(error) = run_windows_sandbox_setup_helper(&agent_root, mode) {
        return Some(format!("Windows sandbox setup helper failed: {error}"));
    }
    windows_sandbox_setup_error(config, None)
}

#[cfg(not(windows))]
fn run_windows_sandbox_setup_helper(
    _agent_root: &std::path::Path,
    _mode: WindowsSandboxSetupMode,
) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "Windows sandbox setup helper is only available on Windows",
    ))
}

#[cfg(windows)]
fn run_windows_sandbox_setup_helper(
    agent_root: &std::path::Path,
    mode: WindowsSandboxSetupMode,
) -> std::io::Result<()> {
    let current_exe = std::env::current_exe()?;
    let helper_name = "windows-sandbox-setup.exe";
    let helper = current_exe
        .parent()
        .ok_or_else(|| std::io::Error::other("app-server has no executable parent"))?
        .join(helper_name);
    if !helper.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "Windows sandbox setup helper is missing: {}",
                helper.display()
            ),
        ));
    }
    let username = std::env::var("USERNAME").map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "USERNAME is missing; cannot protect Windows sandbox artifacts",
        )
    })?;
    if username.trim().is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "USERNAME is empty; cannot protect Windows sandbox artifacts",
        ));
    }
    let owner = std::env::var("USERDOMAIN")
        .ok()
        .filter(|domain| !domain.trim().is_empty())
        .map(|domain| format!(r"{domain}\{username}"))
        .unwrap_or(username);

    let mut child = match mode {
        WindowsSandboxSetupMode::Unelevated => Command::new(&helper)
            .arg("--agent-root")
            .arg(agent_root)
            .arg("--owner")
            .arg(&owner)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?,
        WindowsSandboxSetupMode::Elevated => {
            let powershell = std::env::var_os("SystemRoot")
                .map(PathBuf::from)
                .map(|root| root.join("System32/WindowsPowerShell/v1.0/powershell.exe"))
                .filter(|path| path.is_file())
                .ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::NotFound, "PowerShell is missing")
                })?;
            let script = format!(
                "$p = Start-Process -FilePath '{}' -ArgumentList @('--agent-root','{}','--owner','{}') -Verb RunAs -Wait -PassThru; exit $p.ExitCode",
                powershell_escape(&helper.to_string_lossy()),
                powershell_escape(&agent_root.to_string_lossy()),
                powershell_escape(&owner),
            );
            let encoded = encode_powershell_command(&script);
            Command::new(powershell)
                .args([
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-EncodedCommand",
                    &encoded,
                ])
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()?
        }
    };
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| std::io::Error::other("setup helper stdout is unavailable"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| std::io::Error::other("setup helper stderr is unavailable"))?;
    let stdout_reader = std::thread::spawn(move || read_bounded_setup_output(stdout));
    let stderr_reader = std::thread::spawn(move || read_bounded_setup_output(stderr));
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if status.success() {
                    join_setup_output(stdout_reader)?;
                    join_setup_output(stderr_reader)?;
                    return Ok(());
                }
                let stdout = join_setup_output(stdout_reader)?;
                let stderr = join_setup_output(stderr_reader)?;
                return Err(std::io::Error::other(format!(
                    "helper exited with {status}; stdout={}; stderr={}",
                    stdout, stderr
                )));
            }
            Ok(None) => {}
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = join_setup_output(stdout_reader);
                let _ = join_setup_output(stderr_reader);
                return Err(error);
            }
        }
        if started.elapsed() >= Duration::from_secs(120) {
            let _ = child.kill();
            let _ = child.wait();
            let _ = join_setup_output(stdout_reader);
            let _ = join_setup_output(stderr_reader);
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "Windows sandbox setup helper timed out after 120 seconds",
            ));
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

#[cfg(windows)]
fn powershell_escape(value: &str) -> String {
    value.replace('\'', "''")
}

#[cfg(windows)]
fn encode_powershell_command(script: &str) -> String {
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    let mut utf16 = Vec::with_capacity(script.len() * 2);
    for unit in script.encode_utf16() {
        utf16.extend_from_slice(&unit.to_le_bytes());
    }
    STANDARD.encode(utf16)
}

#[cfg(windows)]
fn read_bounded_setup_output(mut reader: impl std::io::Read) -> std::io::Result<String> {
    const MAX_RETAINED_BYTES: usize = 2048;
    let mut retained = Vec::with_capacity(MAX_RETAINED_BYTES);
    let mut buffer = [0u8; 1024];
    let mut truncated = false;
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        let remaining = MAX_RETAINED_BYTES.saturating_sub(retained.len());
        retained.extend_from_slice(&buffer[..read.min(remaining)]);
        truncated |= read > remaining;
    }
    let mut output = String::from_utf8_lossy(&retained).trim().to_string();
    if truncated {
        output.push_str("...");
    }
    Ok(output)
}

#[cfg(windows)]
fn join_setup_output(
    reader: std::thread::JoinHandle<std::io::Result<String>>,
) -> std::io::Result<String> {
    reader
        .join()
        .map_err(|_| std::io::Error::other("setup helper output reader panicked"))?
}

fn windows_sandbox_setup_error(config: &Config, cwd: Option<&PathBuf>) -> Option<String> {
    windows_sandbox_setup_error_for_platform(config, cwd, SandboxBackendPlatform::current())
}

fn windows_sandbox_setup_error_for_platform(
    config: &Config,
    cwd: Option<&PathBuf>,
    platform: SandboxBackendPlatform,
) -> Option<String> {
    if platform != SandboxBackendPlatform::Windows {
        return Some("Windows sandbox setup is only available on Windows".to_string());
    }
    if !config.agent.workspace_sandbox.enabled {
        return Some("workspace sandbox is disabled".to_string());
    }
    if platform == SandboxBackendPlatform::current() {
        let inspection = inspect_default_windows_sandbox_setup();
        if !inspection.is_valid() {
            return Some(format!(
                "Windows sandbox setup is fail-closed: {}",
                inspection.reason
            ));
        }
    }
    let metadata = serde_json::to_value(config).unwrap_or(Value::Null);
    let plan = plan_sandbox_backend(SandboxBackendPlanInput {
        sandbox_profile: ToolExecutionSandboxProfile::WorkspaceCommand,
        requested_policy: Some("workspace-write"),
        request_metadata: Some(&metadata),
        bypass_restrictions: false,
        platform,
    });
    let cwd = cwd
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "current working directory".to_string());
    if plan.status == SandboxBackendStatus::Ready && plan.enforced {
        return None;
    }

    Some(format!(
        "Windows sandbox setup is fail-closed for {cwd}: {} ({})",
        plan.reason,
        plan.status.label()
    ))
}

fn windows_sandbox_readiness(
    config: &Config,
    platform: SandboxBackendPlatform,
) -> WindowsSandboxReadiness {
    if platform != SandboxBackendPlatform::Windows || !config.agent.workspace_sandbox.enabled {
        return WindowsSandboxReadiness::NotConfigured;
    }
    if platform == SandboxBackendPlatform::current()
        && !inspect_default_windows_sandbox_setup().is_valid()
    {
        return WindowsSandboxReadiness::UpdateRequired;
    }

    let metadata = serde_json::to_value(config).unwrap_or(Value::Null);
    let plan = plan_sandbox_backend(SandboxBackendPlanInput {
        sandbox_profile: ToolExecutionSandboxProfile::WorkspaceCommand,
        requested_policy: Some("workspace-write"),
        request_metadata: Some(&metadata),
        bypass_restrictions: false,
        platform,
    });
    match (plan.status, plan.enforced) {
        (SandboxBackendStatus::Ready, true) => WindowsSandboxReadiness::Ready,
        (SandboxBackendStatus::Disabled, _)
        | (SandboxBackendStatus::NotRequired, _)
        | (SandboxBackendStatus::Bypassed, _) => WindowsSandboxReadiness::NotConfigured,
        _ => WindowsSandboxReadiness::UpdateRequired,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_windows_hosts_never_report_windows_sandbox_ready() {
        let mut config = Config::default();
        config.agent.workspace_sandbox.enabled = true;

        assert_eq!(
            windows_sandbox_readiness(&config, SandboxBackendPlatform::Macos),
            WindowsSandboxReadiness::NotConfigured
        );
    }

    #[test]
    fn disabled_windows_sandbox_is_not_configured() {
        assert_eq!(
            windows_sandbox_readiness(&Config::default(), SandboxBackendPlatform::Windows),
            WindowsSandboxReadiness::NotConfigured
        );
    }

    #[test]
    fn enabled_windows_sandbox_exposes_runner_gap_as_update_required() {
        let mut config = Config::default();
        config.agent.workspace_sandbox.enabled = true;

        assert_eq!(
            windows_sandbox_readiness(&config, SandboxBackendPlatform::Windows),
            WindowsSandboxReadiness::UpdateRequired
        );
    }

    #[test]
    fn setup_remains_fail_closed_when_the_backend_is_not_enforced() {
        let mut config = Config::default();
        config.agent.workspace_sandbox.enabled = true;

        let error = windows_sandbox_setup_error(&config, Some(&PathBuf::from("C:/workspace")))
            .expect("non-Windows setup must fail closed");
        assert!(error.contains("only available on Windows"));

        let error = windows_sandbox_setup_error_for_platform(
            &config,
            Some(&PathBuf::from("C:/workspace")),
            SandboxBackendPlatform::Windows,
        )
        .expect("setup must not claim success before runner evidence");
        assert!(error.contains("fail-closed"));
    }
}
