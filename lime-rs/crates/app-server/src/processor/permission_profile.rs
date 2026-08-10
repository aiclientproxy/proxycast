use super::{dispatch_result, parse_params, RequestProcessor, RpcDispatch};
use app_server_protocol::protocol::v2::{
    PermissionProfileListParams, PermissionProfileListResponse, PermissionProfileSummary,
};
use app_server_protocol::{error_codes, JsonRpcError};

pub(super) const READ_ONLY_PROFILE_ID: &str = ":read-only";
pub(super) const WORKSPACE_PROFILE_ID: &str = ":workspace";
pub(super) const DANGER_FULL_ACCESS_PROFILE_ID: &str = ":danger-full-access";

pub(super) struct ResolvedPermissionProfile {
    pub(super) id: &'static str,
    pub(super) sandbox_policy: &'static str,
}

impl RequestProcessor {
    pub(super) async fn handle_permission_profile_list_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let PermissionProfileListParams {
            cursor,
            limit,
            cwd: _,
        } = parse_params(params)?;
        let data = permission_profile_catalog();
        let start = parse_cursor(cursor.as_deref(), data.len())?;
        let limit = limit.unwrap_or(data.len() as u32).max(1) as usize;
        let end = start.saturating_add(limit).min(data.len());
        dispatch_result(PermissionProfileListResponse {
            data: data[start..end].to_vec(),
            next_cursor: (end < data.len()).then(|| end.to_string()),
        })
    }
}

pub(super) fn resolve_permission_profile(
    id: &str,
) -> Result<ResolvedPermissionProfile, JsonRpcError> {
    match id.trim() {
        READ_ONLY_PROFILE_ID => Ok(ResolvedPermissionProfile {
            id: READ_ONLY_PROFILE_ID,
            sandbox_policy: "read-only",
        }),
        WORKSPACE_PROFILE_ID => Ok(ResolvedPermissionProfile {
            id: WORKSPACE_PROFILE_ID,
            sandbox_policy: "workspace-write",
        }),
        DANGER_FULL_ACCESS_PROFILE_ID => Ok(ResolvedPermissionProfile {
            id: DANGER_FULL_ACCESS_PROFILE_ID,
            sandbox_policy: "danger-full-access",
        }),
        value => Err(JsonRpcError::new(
            error_codes::INVALID_PARAMS,
            format!("unknown permission profile: {value}"),
        )),
    }
}

fn permission_profile_catalog() -> Vec<PermissionProfileSummary> {
    [
        READ_ONLY_PROFILE_ID,
        WORKSPACE_PROFILE_ID,
        DANGER_FULL_ACCESS_PROFILE_ID,
    ]
    .into_iter()
    .map(|id| PermissionProfileSummary {
        id: id.to_string(),
        description: None,
        allowed: true,
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

    #[test]
    fn catalog_keeps_codex_builtin_order_and_runtime_mapping() {
        let catalog = permission_profile_catalog();
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
