use super::{dispatch_result, parse_params, RequestProcessor, RpcDispatch};
use app_server_protocol::protocol::v2::{
    WindowsSandboxReadiness, WindowsSandboxReadinessParams, WindowsSandboxReadinessResponse,
};
use app_server_protocol::{error_codes, JsonRpcError};
use lime_core::config::{load_config, Config};
use serde_json::Value;
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
}
