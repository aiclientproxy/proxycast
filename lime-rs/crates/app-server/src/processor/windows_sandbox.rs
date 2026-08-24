use super::{dispatch_result, parse_params, ConnectionRequestId, RequestProcessor, RpcDispatch};
use app_server_protocol::protocol::v2::ServerNotification;
use app_server_protocol::protocol::v2::{
    WindowsSandboxReadiness, WindowsSandboxReadinessParams, WindowsSandboxReadinessResponse,
    WindowsSandboxSetupCompletedNotification, WindowsSandboxSetupStartParams,
    WindowsSandboxSetupStartResponse,
};
use app_server_protocol::{error_codes, JsonRpcError};
use lime_core::config::{load_config, Config};
use serde_json::Value;
use std::path::PathBuf;
use tool_runtime::execution_policy::ToolExecutionSandboxProfile;
use tool_runtime::sandbox::{
    plan_sandbox_backend, SandboxBackendPlanInput, SandboxBackendPlatform, SandboxBackendStatus,
};

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
        let setup_error = windows_sandbox_setup_error(&config, cwd.as_ref());
        let processor = self.clone();
        tokio::spawn(async move {
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
