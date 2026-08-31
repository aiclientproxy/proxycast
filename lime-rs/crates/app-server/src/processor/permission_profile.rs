use super::{dispatch_result, parse_params, to_jsonrpc_error, RequestProcessor, RpcDispatch};
use crate::permission_profile::{
    builtin_permission_profiles, permission_profile_is_allowed, PermissionProfilePolicy,
};
use app_server_protocol::protocol::v2::{
    PermissionProfileListParams, PermissionProfileListResponse, PermissionProfileSummary,
};
use app_server_protocol::{error_codes, JsonRpcError};

impl RequestProcessor {
    pub(super) async fn handle_permission_profile_list_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let PermissionProfileListParams { cursor, limit, cwd } = parse_params(params)?;
        let policy = self
            .runtime
            .current_permission_profile_policy(cwd.as_deref())
            .map_err(to_jsonrpc_error)?;
        let data = permission_profile_catalog(&policy);
        let start = parse_cursor(cursor.as_deref(), data.len())?;
        let limit = limit.unwrap_or(data.len() as u32).max(1) as usize;
        let end = start.saturating_add(limit).min(data.len());
        dispatch_result(PermissionProfileListResponse {
            data: data[start..end].to_vec(),
            next_cursor: (end < data.len()).then(|| end.to_string()),
        })
    }

    pub(super) fn resolve_allowed_permission_profile(
        &self,
        id: &str,
        cwd: Option<&str>,
    ) -> Result<crate::permission_profile::ResolvedPermissionProfile, JsonRpcError> {
        self.runtime
            .resolve_allowed_permission_profile(id, cwd)
            .map_err(|error| match error {
                crate::RuntimeCoreError::InvalidRequest(message) => {
                    JsonRpcError::new(error_codes::INVALID_PARAMS, message)
                }
                other => to_jsonrpc_error(other),
            })
    }
}

pub(super) fn resolve_permission_profile(
    id: &str,
) -> Result<crate::permission_profile::ResolvedPermissionProfile, JsonRpcError> {
    crate::permission_profile::resolve_permission_profile(id)
        .map_err(|message| JsonRpcError::new(error_codes::INVALID_PARAMS, message))
}

fn permission_profile_catalog(policy: &PermissionProfilePolicy) -> Vec<PermissionProfileSummary> {
    builtin_permission_profiles()
        .into_iter()
        .map(|profile| PermissionProfileSummary {
            id: profile.id.to_string(),
            description: None,
            allowed: permission_profile_is_allowed(policy, profile),
        })
        .collect()
}

fn parse_cursor(cursor: Option<&str>, total: usize) -> Result<usize, JsonRpcError> {
    let Some(cursor) = cursor else {
        return Ok(0);
    };
    let start = cursor.parse::<usize>().map_err(|_| {
        JsonRpcError::new(
            error_codes::INVALID_PARAMS,
            format!("invalid cursor: {cursor}"),
        )
    })?;
    if start > total {
        return Err(JsonRpcError::new(
            error_codes::INVALID_PARAMS,
            format!("cursor {start} exceeds total permission profiles {total}"),
        ));
    }
    Ok(start)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::permission_profile::{
        DANGER_FULL_ACCESS_PROFILE_ID, READ_ONLY_PROFILE_ID, WORKSPACE_PROFILE_ID,
    };

    #[test]
    fn catalog_keeps_codex_builtin_order_and_runtime_mapping() {
        let policy = crate::permission_profile::permission_profile_policy(
            &lime_core::config::Config::default(),
            None,
            tool_runtime::sandbox::SandboxBackendPlatform::Unsupported,
        )
        .expect("default permission policy");
        let catalog = permission_profile_catalog(&policy);
        assert_eq!(
            catalog
                .iter()
                .map(|profile| profile.id.as_str())
                .collect::<Vec<_>>(),
            vec![
                READ_ONLY_PROFILE_ID,
                WORKSPACE_PROFILE_ID,
                DANGER_FULL_ACCESS_PROFILE_ID
            ]
        );
        assert!(catalog.iter().all(|profile| profile.allowed));
        assert_eq!(
            resolve_permission_profile(WORKSPACE_PROFILE_ID)
                .unwrap()
                .sandbox_policy,
            "workspace-write"
        );
        assert!(resolve_permission_profile("custom").is_err());
    }
}
