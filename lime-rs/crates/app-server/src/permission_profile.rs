use lime_core::config::Config;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use tool_runtime::execution_policy::ToolExecutionSandboxProfile;
use tool_runtime::sandbox::{
    plan_sandbox_backend, SandboxBackendPlanInput, SandboxBackendPlatform,
};
use tool_runtime::windows_setup::inspect_default_windows_sandbox_setup;

pub(crate) const READ_ONLY_PROFILE_ID: &str = ":read-only";
pub(crate) const WORKSPACE_PROFILE_ID: &str = ":workspace";
pub(crate) const DANGER_FULL_ACCESS_PROFILE_ID: &str = ":danger-full-access";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedPermissionProfile {
    pub(crate) id: &'static str,
    pub(crate) sandbox_policy: &'static str,
}

#[derive(Clone, Debug)]
pub(crate) struct PermissionProfilePolicy {
    runtime_metadata: Value,
    platform: SandboxBackendPlatform,
}

pub(crate) fn builtin_permission_profiles() -> [ResolvedPermissionProfile; 3] {
    [
        ResolvedPermissionProfile {
            id: READ_ONLY_PROFILE_ID,
            sandbox_policy: "read-only",
        },
        ResolvedPermissionProfile {
            id: WORKSPACE_PROFILE_ID,
            sandbox_policy: "workspace-write",
        },
        ResolvedPermissionProfile {
            id: DANGER_FULL_ACCESS_PROFILE_ID,
            sandbox_policy: "danger-full-access",
        },
    ]
}

pub(crate) fn resolve_permission_profile(id: &str) -> Result<ResolvedPermissionProfile, String> {
    let id = id.trim();
    builtin_permission_profiles()
        .into_iter()
        .find(|profile| profile.id == id)
        .ok_or_else(|| format!("unknown permission profile: {id}"))
}

pub(crate) fn permission_profile_policy(
    config: &Config,
    cwd: Option<&str>,
    platform: SandboxBackendPlatform,
) -> Result<PermissionProfilePolicy, String> {
    materialize_permission_profile_cwd(cwd)?;
    let runtime_metadata = json!({
        "agent": {
            "workspaceSandbox": &config.agent.workspace_sandbox,
        }
    });
    Ok(PermissionProfilePolicy {
        runtime_metadata,
        platform,
    })
}

pub(crate) fn permission_profile_is_allowed(
    policy: &PermissionProfilePolicy,
    profile: ResolvedPermissionProfile,
) -> bool {
    let plan = plan_sandbox_backend(SandboxBackendPlanInput {
        sandbox_profile: ToolExecutionSandboxProfile::WorkspaceCommand,
        requested_policy: Some(profile.sandbox_policy),
        request_metadata: Some(&policy.runtime_metadata),
        bypass_restrictions: profile.id == DANGER_FULL_ACCESS_PROFILE_ID,
        platform: policy.platform,
    });
    if plan.strict_fallback_blocks_execution() {
        return false;
    }
    if policy.platform == SandboxBackendPlatform::Windows
        && plan.required
        && plan.config.enabled
        && plan.config.strict
    {
        return inspect_default_windows_sandbox_setup().is_valid();
    }
    true
}

pub(crate) fn resolve_allowed_permission_profile(
    policy: &PermissionProfilePolicy,
    id: &str,
) -> Result<ResolvedPermissionProfile, String> {
    let profile = resolve_permission_profile(id)?;
    if permission_profile_is_allowed(policy, profile) {
        return Ok(profile);
    }
    Err(format!(
        "permission profile is disallowed by the current sandbox policy: {}",
        profile.id
    ))
}

fn materialize_permission_profile_cwd(cwd: Option<&str>) -> Result<PathBuf, String> {
    let current_dir = std::env::current_dir()
        .map_err(|error| format!("resolve current working directory: {error}"))?;
    let Some(cwd) = cwd else {
        return Ok(current_dir);
    };
    let cwd = cwd.trim();
    if cwd.is_empty() {
        return Err("permission profile cwd must not be empty".to_string());
    }
    if cwd.contains('\0') {
        return Err("permission profile cwd must not contain NUL".to_string());
    }
    let cwd = Path::new(cwd);
    Ok(if cwd.is_absolute() {
        cwd.to_path_buf()
    } else {
        current_dir.join(cwd)
    })
}

pub(crate) fn apply_permission_profile_to_metadata(
    metadata: &mut Map<String, Value>,
    id: &str,
) -> Result<(), String> {
    let profile = resolve_permission_profile(id)?;
    metadata.insert(
        "permissions".to_string(),
        Value::String(profile.id.to_string()),
    );
    metadata.insert(
        "activePermissionProfile".to_string(),
        json!({ "id": profile.id }),
    );
    metadata.insert(
        "sandboxPolicy".to_string(),
        Value::String(profile.sandbox_policy.to_string()),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_catalog_keeps_codex_order_and_runtime_mapping() {
        let catalog = builtin_permission_profiles();
        assert_eq!(
            catalog.iter().map(|profile| profile.id).collect::<Vec<_>>(),
            vec![
                READ_ONLY_PROFILE_ID,
                WORKSPACE_PROFILE_ID,
                DANGER_FULL_ACCESS_PROFILE_ID
            ]
        );
        assert_eq!(
            resolve_permission_profile(WORKSPACE_PROFILE_ID)
                .unwrap()
                .sandbox_policy,
            "workspace-write"
        );
        assert!(resolve_permission_profile("custom").is_err());
    }

    #[test]
    fn metadata_projection_keeps_profile_and_lowered_sandbox_together() {
        let mut metadata = Map::new();
        apply_permission_profile_to_metadata(&mut metadata, READ_ONLY_PROFILE_ID)
            .expect("known permission profile");

        assert_eq!(
            metadata.get("activePermissionProfile"),
            Some(&json!({ "id": READ_ONLY_PROFILE_ID }))
        );
        assert_eq!(metadata.get("sandboxPolicy"), Some(&json!("read-only")));
    }

    #[test]
    fn strict_unavailable_sandbox_disallows_restricted_profiles() {
        let mut config = Config::default();
        config.agent.workspace_sandbox.enabled = true;
        config.agent.workspace_sandbox.strict = true;
        let policy = permission_profile_policy(
            &config,
            Some("/workspace"),
            SandboxBackendPlatform::Unsupported,
        )
        .expect("permission profile policy");

        assert!(!permission_profile_is_allowed(
            &policy,
            resolve_permission_profile(READ_ONLY_PROFILE_ID).unwrap()
        ));
        assert!(!permission_profile_is_allowed(
            &policy,
            resolve_permission_profile(WORKSPACE_PROFILE_ID).unwrap()
        ));
        assert!(permission_profile_is_allowed(
            &policy,
            resolve_permission_profile(DANGER_FULL_ACCESS_PROFILE_ID).unwrap()
        ));
        assert!(
            resolve_allowed_permission_profile(&policy, WORKSPACE_PROFILE_ID)
                .unwrap_err()
                .contains("disallowed")
        );
    }

    #[test]
    fn non_strict_policy_keeps_profiles_selectable_without_a_backend() {
        let mut config = Config::default();
        config.agent.workspace_sandbox.enabled = true;
        config.agent.workspace_sandbox.strict = false;
        let policy = permission_profile_policy(
            &config,
            Some("workspace"),
            SandboxBackendPlatform::Unsupported,
        )
        .expect("permission profile policy");

        assert!(builtin_permission_profiles()
            .into_iter()
            .all(|profile| permission_profile_is_allowed(&policy, profile)));
    }

    #[cfg(not(windows))]
    #[test]
    fn strict_windows_policy_requires_valid_setup_artifacts() {
        let mut config = Config::default();
        config.agent.workspace_sandbox.enabled = true;
        config.agent.workspace_sandbox.strict = true;
        let policy = permission_profile_policy(
            &config,
            Some("C:/workspace"),
            SandboxBackendPlatform::Windows,
        )
        .expect("permission profile policy");

        assert!(!permission_profile_is_allowed(
            &policy,
            resolve_permission_profile(WORKSPACE_PROFILE_ID).unwrap()
        ));
        assert!(permission_profile_is_allowed(
            &policy,
            resolve_permission_profile(DANGER_FULL_ACCESS_PROFILE_ID).unwrap()
        ));
    }

    #[test]
    fn cwd_materialization_rejects_empty_values_and_resolves_relative_paths() {
        assert!(materialize_permission_profile_cwd(Some("  ")).is_err());
        let cwd = materialize_permission_profile_cwd(Some("workspace"))
            .expect("relative cwd should materialize");
        assert!(cwd.is_absolute());
        assert!(cwd.ends_with("workspace"));
    }
}
