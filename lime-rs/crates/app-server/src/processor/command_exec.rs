use super::{dispatch_result, parse_params, ConnectionRequestId, RequestProcessor, RpcDispatch};
use crate::permission_profile::resolve_permission_profile_for_request;
use app_server_protocol::protocol::v2::CommandExecParams;
use app_server_protocol::{error_codes, JsonRpcError};
use serde_json::Value;

impl RequestProcessor {
    pub(super) async fn handle_command_exec_impl(
        &self,
        params: Option<Value>,
        request: Option<ConnectionRequestId>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        reject_client_granted_permissions(params.as_ref())?;
        let connection_id = request
            .map(|request| request.connection_id)
            .ok_or_else(|| {
                JsonRpcError::new(
                    error_codes::INVALID_REQUEST,
                    "command/exec requires transport connection",
                )
            })?;
        let mut params: CommandExecParams = parse_params(params)?;
        let granted_permissions = self.resolve_command_exec_permission_profile(&mut params)?;
        dispatch_result(
            self.command_exec
                .exec(connection_id, params, granted_permissions)
                .await?,
        )
    }

    pub(super) async fn handle_command_exec_write_impl(
        &self,
        params: Option<Value>,
        request: Option<ConnectionRequestId>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let connection_id = connection_id(request)?;
        dispatch_result(
            self.command_exec
                .write(connection_id, parse_params(params)?)
                .await?,
        )
    }

    pub(super) async fn handle_command_exec_resize_impl(
        &self,
        params: Option<Value>,
        request: Option<ConnectionRequestId>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let connection_id = connection_id(request)?;
        dispatch_result(
            self.command_exec
                .resize(connection_id, parse_params(params)?)
                .await?,
        )
    }

    pub(super) async fn handle_command_exec_terminate_impl(
        &self,
        params: Option<Value>,
        request: Option<ConnectionRequestId>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let connection_id = connection_id(request)?;
        dispatch_result(
            self.command_exec
                .terminate(connection_id, parse_params(params)?)
                .await?,
        )
    }

    fn resolve_command_exec_permission_profile(
        &self,
        params: &mut CommandExecParams,
    ) -> Result<Option<app_server_protocol::protocol::v2::GrantedPermissionProfile>, JsonRpcError>
    {
        if params.permission_profile.is_none() && params.sandbox_policy.is_some() {
            return Ok(None);
        }
        let cwd = params
            .cwd
            .as_ref()
            .map(|cwd| {
                cwd.to_str().ok_or_else(|| {
                    JsonRpcError::new(
                        error_codes::INVALID_PARAMS,
                        "command/exec cwd must be valid UTF-8 when resolving permissionProfile",
                    )
                })
            })
            .transpose()?;
        let policy = self
            .runtime
            .current_permission_profile_policy(cwd)
            .map_err(crate::processor::to_jsonrpc_error)?;
        resolve_command_exec_permission_profile_with_policy(&policy, params)
    }
}

fn resolve_command_exec_permission_profile_with_policy(
    policy: &crate::permission_profile::PermissionProfilePolicy,
    params: &mut CommandExecParams,
) -> Result<Option<app_server_protocol::protocol::v2::GrantedPermissionProfile>, JsonRpcError> {
    if params.permission_profile.is_none() && params.sandbox_policy.is_some() {
        return Ok(None);
    }
    let requested = params.permission_profile.as_deref();
    let profile = resolve_permission_profile_for_request(&policy, requested)
        .map_err(|message| JsonRpcError::new(error_codes::INVALID_PARAMS, message))?;
    let Some(profile) = profile else {
        return Ok(None);
    };
    if params.sandbox_policy.is_some() {
        return Err(JsonRpcError::new(
            error_codes::INVALID_PARAMS,
            "permissionProfile cannot be combined with sandboxPolicy",
        ));
    }
    params.sandbox_policy = Some(Value::String(profile.sandbox_policy));
    params.permission_profile = None;
    Ok(profile.granted_permissions)
}

fn reject_client_granted_permissions(params: Option<&Value>) -> Result<(), JsonRpcError> {
    if params
        .and_then(Value::as_object)
        .is_some_and(|object| object.contains_key("grantedPermissions"))
    {
        return Err(JsonRpcError::new(
            error_codes::INVALID_PARAMS,
            "grantedPermissions is managed by permissionProfile",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::permission_profile::permission_profile_policy;
    use lime_core::config::{Config, PermissionFilesystemConfig, PermissionProfileConfig};
    use std::collections::BTreeMap;
    use tool_runtime::sandbox::SandboxBackendPlatform;

    fn params() -> CommandExecParams {
        CommandExecParams {
            command: vec!["printf".to_string(), "ok".to_string()],
            process_id: None,
            tty: false,
            stream_stdin: false,
            stream_stdout_stderr: false,
            output_bytes_cap: None,
            disable_output_cap: false,
            disable_timeout: false,
            timeout_ms: None,
            cwd: None,
            env: None,
            size: None,
            sandbox_policy: None,
            permission_profile: None,
        }
    }

    #[test]
    fn command_exec_named_profile_lowers_policy_and_grants() {
        let mut config = Config::default();
        config.permissions.insert(
            "docs".to_string(),
            PermissionProfileConfig {
                filesystem: Some(PermissionFilesystemConfig {
                    entries: BTreeMap::from([(":minimal".to_string(), "read".to_string())]),
                    ..PermissionFilesystemConfig::default()
                }),
                ..PermissionProfileConfig::default()
            },
        );
        let policy = permission_profile_policy(&config, None, SandboxBackendPlatform::Unsupported)
            .expect("permission profile policy");
        let mut params = params();
        params.permission_profile = Some("docs".to_string());

        let grants = resolve_command_exec_permission_profile_with_policy(&policy, &mut params)
            .expect("named profile");
        assert_eq!(params.permission_profile, None);
        assert_eq!(
            params.sandbox_policy,
            Some(Value::String("workspace-write".into()))
        );
        assert!(grants.is_some());
    }

    #[test]
    fn command_exec_uses_configured_default_profile() {
        let mut config = Config::default();
        config.default_permissions = Some(":read-only".to_string());
        let policy = permission_profile_policy(&config, None, SandboxBackendPlatform::Unsupported)
            .expect("permission profile policy");
        let mut params = params();

        let grants = resolve_command_exec_permission_profile_with_policy(&policy, &mut params)
            .expect("default profile");
        assert!(grants.is_none());
        assert_eq!(
            params.sandbox_policy,
            Some(Value::String("read-only".into()))
        );
    }

    #[test]
    fn command_exec_rejects_client_granted_permissions() {
        let error = reject_client_granted_permissions(Some(&serde_json::json!({
            "grantedPermissions": {"network": {"enabled": true}}
        })))
        .expect_err("client grants must fail closed");
        assert_eq!(error.code, error_codes::INVALID_PARAMS);
        assert!(error.message.contains("managed by permissionProfile"));
    }
}

fn connection_id(
    request: Option<ConnectionRequestId>,
) -> Result<app_server_transport::ConnectionId, JsonRpcError> {
    request.map(|request| request.connection_id).ok_or_else(|| {
        JsonRpcError::new(
            error_codes::INVALID_REQUEST,
            "command/exec requires transport connection",
        )
    })
}
