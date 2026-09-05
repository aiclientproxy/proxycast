use app_server_protocol::protocol::v2::{
    AdditionalFileSystemPermissions, AdditionalNetworkPermissions, FileSystemAccessMode,
    FileSystemPath, FileSystemSandboxEntry, FileSystemSpecialPath, GrantedPermissionProfile,
};
use lime_core::config::{
    Config, PermissionFilesystemConfig, PermissionNetworkConfig, PermissionProfileConfig,
};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use tool_runtime::execution_policy::ToolExecutionSandboxProfile;
use tool_runtime::sandbox::{
    plan_sandbox_backend, SandboxBackendPlanInput, SandboxBackendPlatform,
};
use tool_runtime::windows_setup::inspect_default_windows_sandbox_setup;

pub(crate) const READ_ONLY_PROFILE_ID: &str = ":read-only";
pub(crate) const WORKSPACE_PROFILE_ID: &str = ":workspace";
pub(crate) const DANGER_FULL_ACCESS_PROFILE_ID: &str = ":danger-full-access";

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedPermissionProfile {
    pub(crate) id: String,
    pub(crate) sandbox_policy: String,
    pub(crate) description: Option<String>,
    pub(crate) extends: Option<String>,
    pub(crate) granted_permissions: Option<GrantedPermissionProfile>,
}

#[derive(Clone, Debug)]
pub(crate) struct PermissionProfilePolicy {
    runtime_metadata: Value,
    platform: SandboxBackendPlatform,
    default_permissions: Option<String>,
    profiles: BTreeMap<String, PermissionProfileConfig>,
}

pub(crate) fn builtin_permission_profiles() -> [ResolvedPermissionProfile; 3] {
    [
        ResolvedPermissionProfile {
            id: READ_ONLY_PROFILE_ID.to_string(),
            sandbox_policy: "read-only".to_string(),
            description: None,
            extends: None,
            granted_permissions: None,
        },
        ResolvedPermissionProfile {
            id: WORKSPACE_PROFILE_ID.to_string(),
            sandbox_policy: "workspace-write".to_string(),
            description: None,
            extends: None,
            granted_permissions: None,
        },
        ResolvedPermissionProfile {
            id: DANGER_FULL_ACCESS_PROFILE_ID.to_string(),
            sandbox_policy: "danger-full-access".to_string(),
            description: None,
            extends: None,
            granted_permissions: None,
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

pub(crate) fn resolve_permission_profile_for_request(
    policy: &PermissionProfilePolicy,
    requested_id: Option<&str>,
) -> Result<Option<ResolvedPermissionProfile>, String> {
    requested_id
        .or(policy.default_permissions.as_deref())
        .map(|id| resolve_configured_permission_profile(policy, id))
        .transpose()
}

pub(crate) fn permission_profile_catalog(
    policy: &PermissionProfilePolicy,
) -> Vec<ResolvedPermissionProfile> {
    let mut profiles = builtin_permission_profiles().to_vec();
    for id in policy.profiles.keys() {
        if let Ok(profile) = resolve_configured_permission_profile(policy, id) {
            if !profiles.iter().any(|entry| entry.id == profile.id) {
                profiles.push(profile);
            }
        }
    }
    profiles
}

fn resolve_configured_permission_profile_inner(
    policy: &PermissionProfilePolicy,
    id: &str,
    stack: &mut Vec<String>,
) -> Result<ResolvedPermissionProfile, String> {
    let id = id.trim();
    if let Some(profile) = builtin_permission_profiles()
        .into_iter()
        .find(|profile| profile.id == id)
    {
        return Ok(profile);
    }
    if id.starts_with(':') {
        return Err(format!("unknown permission profile: {id}"));
    }
    if let Some(cycle_start) = stack.iter().position(|parent| parent == id) {
        let cycle = stack[cycle_start..]
            .iter()
            .cloned()
            .chain(std::iter::once(id.to_string()))
            .collect::<Vec<_>>();
        return Err(format!(
            "permission profile inheritance cycle: {}",
            cycle.join(" -> ")
        ));
    }
    let config = policy
        .profiles
        .get(id)
        .ok_or_else(|| format!("unknown permission profile: {id}"))?;
    stack.push(id.to_string());
    let mut resolved = if let Some(parent) = config.extends.as_deref() {
        resolve_configured_permission_profile_inner(policy, parent, stack)?
    } else {
        ResolvedPermissionProfile {
            id: id.to_string(),
            sandbox_policy: "workspace-write".to_string(),
            description: None,
            extends: None,
            granted_permissions: None,
        }
    };
    stack.pop();
    resolved.id = id.to_string();
    resolved.description = config.description.clone();
    resolved.extends = config.extends.clone();
    if config.filesystem.is_some() || config.network.is_some() {
        let mut permissions = resolved.granted_permissions.unwrap_or_default();
        if let Some(filesystem) = config.filesystem.as_ref() {
            permissions.file_system = Some(merge_filesystem_permissions(
                permissions.file_system.take(),
                filesystem,
            )?);
        }
        if let Some(network) = config.network.as_ref() {
            permissions.network = Some(merge_network_permissions(
                permissions.network.take(),
                network,
            ));
        }
        resolved.granted_permissions = Some(permissions);
    }
    Ok(resolved)
}

fn resolve_configured_permission_profile(
    policy: &PermissionProfilePolicy,
    id: &str,
) -> Result<ResolvedPermissionProfile, String> {
    resolve_configured_permission_profile_inner(policy, id, &mut Vec::new())
}

fn merge_network_permissions(
    parent: Option<AdditionalNetworkPermissions>,
    config: &PermissionNetworkConfig,
) -> AdditionalNetworkPermissions {
    AdditionalNetworkPermissions {
        enabled: config.enabled.or(parent.and_then(|value| value.enabled)),
    }
}

fn merge_filesystem_permissions(
    parent: Option<AdditionalFileSystemPermissions>,
    config: &PermissionFilesystemConfig,
) -> Result<AdditionalFileSystemPermissions, String> {
    let mut entries = BTreeMap::new();
    let mut glob_scan_max_depth = None;
    if let Some(parent) = parent {
        glob_scan_max_depth = parent.glob_scan_max_depth;
        for entry in parent.entries.unwrap_or_default() {
            let key = filesystem_entry_key(&entry.path);
            entries.insert(key, entry.access);
        }
        for path in parent.read.unwrap_or_default() {
            entries.insert(path, FileSystemAccessMode::Read);
        }
        for path in parent.write.unwrap_or_default() {
            entries.insert(path, FileSystemAccessMode::Write);
        }
    }
    for (path, access) in &config.entries {
        entries.insert(path.clone(), parse_filesystem_access(access)?);
    }
    let entries = entries
        .into_iter()
        .map(|(path, access)| FileSystemSandboxEntry {
            path: parse_filesystem_path(&path),
            access,
        })
        .collect::<Vec<_>>();
    Ok(AdditionalFileSystemPermissions {
        read: None,
        write: None,
        glob_scan_max_depth: config.glob_scan_max_depth.or(glob_scan_max_depth),
        entries: Some(entries),
    })
}

fn filesystem_entry_key(path: &FileSystemPath) -> String {
    match path {
        FileSystemPath::Path { path } => path.clone(),
        FileSystemPath::GlobPattern { pattern } => pattern.clone(),
        FileSystemPath::Special { value } => match value {
            FileSystemSpecialPath::Root => ":root".to_string(),
            FileSystemSpecialPath::Minimal => ":minimal".to_string(),
            FileSystemSpecialPath::ProjectRoots { subpath } => subpath
                .as_deref()
                .map(|path| format!(":workspace_roots/{path}"))
                .unwrap_or_else(|| ":workspace_roots".to_string()),
            FileSystemSpecialPath::Tmpdir => ":tmpdir".to_string(),
            FileSystemSpecialPath::SlashTmp => ":slash_tmp".to_string(),
            FileSystemSpecialPath::Unknown { path, subpath } => subpath
                .as_deref()
                .map(|subpath| format!("{path}/{subpath}"))
                .unwrap_or_else(|| path.clone()),
        },
    }
}

fn parse_filesystem_access(value: &str) -> Result<FileSystemAccessMode, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "read" => Ok(FileSystemAccessMode::Read),
        "write" => Ok(FileSystemAccessMode::Write),
        "none" | "deny" => Ok(FileSystemAccessMode::Deny),
        _ => Err(format!("unsupported filesystem permission: {value}")),
    }
}

fn parse_filesystem_path(path: &str) -> FileSystemPath {
    match path {
        ":root" => FileSystemPath::Special {
            value: FileSystemSpecialPath::Root,
        },
        ":minimal" => FileSystemPath::Special {
            value: FileSystemSpecialPath::Minimal,
        },
        ":workspace_roots" => FileSystemPath::Special {
            value: FileSystemSpecialPath::ProjectRoots { subpath: None },
        },
        ":tmpdir" => FileSystemPath::Special {
            value: FileSystemSpecialPath::Tmpdir,
        },
        ":slash_tmp" => FileSystemPath::Special {
            value: FileSystemSpecialPath::SlashTmp,
        },
        value if value.starts_with(":workspace_roots/") => FileSystemPath::Special {
            value: FileSystemSpecialPath::ProjectRoots {
                subpath: Some(value[":workspace_roots/".len()..].to_string()),
            },
        },
        value if value.contains('*') || value.contains('?') => FileSystemPath::GlobPattern {
            pattern: value.to_string(),
        },
        value => FileSystemPath::Path {
            path: value.to_string(),
        },
    }
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
        default_permissions: config.default_permissions.clone(),
        profiles: config.permissions.clone(),
    })
}

pub(crate) fn permission_profile_is_allowed(
    policy: &PermissionProfilePolicy,
    profile: ResolvedPermissionProfile,
) -> bool {
    let plan = plan_sandbox_backend(SandboxBackendPlanInput {
        sandbox_profile: ToolExecutionSandboxProfile::WorkspaceCommand,
        requested_policy: Some(profile.sandbox_policy.as_str()),
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
    let profile = resolve_configured_permission_profile(policy, id)?;
    if permission_profile_is_allowed(policy, profile.clone()) {
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
    apply_resolved_permission_profile_to_metadata(metadata, profile)
}

pub(crate) fn apply_resolved_permission_profile_to_metadata(
    metadata: &mut Map<String, Value>,
    profile: ResolvedPermissionProfile,
) -> Result<(), String> {
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
        Value::String(profile.sandbox_policy),
    );
    metadata.remove("grantedPermissions");
    if let Some(permissions) = profile.granted_permissions {
        metadata.insert(
            "grantedPermissions".to_string(),
            serde_json::to_value(permissions).map_err(|error| error.to_string())?,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_catalog_keeps_codex_order_and_runtime_mapping() {
        let catalog = builtin_permission_profiles();
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
    fn configured_profile_resolves_inheritance_and_permission_grants() {
        let mut config = Config::default();
        config.permissions.insert(
            "limited-read".to_string(),
            PermissionProfileConfig {
                extends: Some(WORKSPACE_PROFILE_ID.to_string()),
                description: Some("Read selected documentation".to_string()),
                filesystem: Some(PermissionFilesystemConfig {
                    entries: BTreeMap::from([
                        (":minimal".to_string(), "read".to_string()),
                        ("/docs".to_string(), "read".to_string()),
                        ("/docs/private".to_string(), "none".to_string()),
                    ]),
                    glob_scan_max_depth: Some(4),
                }),
                network: Some(PermissionNetworkConfig {
                    enabled: Some(true),
                }),
            },
        );
        let policy = permission_profile_policy(
            &config,
            Some("/workspace"),
            SandboxBackendPlatform::Unsupported,
        )
        .expect("permission profile policy");

        let profile = resolve_configured_permission_profile(&policy, "limited-read")
            .expect("configured profile");
        assert_eq!(profile.id, "limited-read");
        assert_eq!(profile.extends.as_deref(), Some(WORKSPACE_PROFILE_ID));
        assert_eq!(
            profile.description.as_deref(),
            Some("Read selected documentation")
        );
        assert_eq!(profile.sandbox_policy, "workspace-write");
        let grants = profile.granted_permissions.expect("permission grants");
        assert_eq!(
            grants.network.and_then(|network| network.enabled),
            Some(true)
        );
        let filesystem = grants.file_system.expect("filesystem grants");
        assert_eq!(filesystem.glob_scan_max_depth, Some(4));
        let entries = filesystem.entries.expect("filesystem entries");
        assert!(entries.iter().any(|entry| {
            matches!(
                (&entry.path, entry.access),
                (FileSystemPath::Path { path }, FileSystemAccessMode::Read)
                    if path == "/docs"
            )
        }));
        assert!(entries.iter().any(|entry| {
            matches!(
                (&entry.path, entry.access),
                (FileSystemPath::Path { path }, FileSystemAccessMode::Deny)
                    if path == "/docs/private"
            )
        }));
    }

    #[test]
    fn configured_profile_catalog_keeps_builtins_before_named_profiles() {
        let mut config = Config::default();
        config
            .permissions
            .insert("zeta".to_string(), PermissionProfileConfig::default());
        config
            .permissions
            .insert("alpha".to_string(), PermissionProfileConfig::default());
        let policy = permission_profile_policy(&config, None, SandboxBackendPlatform::Unsupported)
            .expect("permission profile policy");
        let ids = permission_profile_catalog(&policy)
            .into_iter()
            .map(|profile| profile.id)
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            vec![
                READ_ONLY_PROFILE_ID,
                WORKSPACE_PROFILE_ID,
                DANGER_FULL_ACCESS_PROFILE_ID,
                "alpha",
                "zeta"
            ]
        );
    }

    #[test]
    fn configured_profile_rejects_unknown_parent_and_cycles() {
        let mut config = Config::default();
        config.permissions.insert(
            "unknown-parent".to_string(),
            PermissionProfileConfig {
                extends: Some("missing".to_string()),
                ..PermissionProfileConfig::default()
            },
        );
        config.permissions.insert(
            "cycle-a".to_string(),
            PermissionProfileConfig {
                extends: Some("cycle-b".to_string()),
                ..PermissionProfileConfig::default()
            },
        );
        config.permissions.insert(
            "cycle-b".to_string(),
            PermissionProfileConfig {
                extends: Some("cycle-a".to_string()),
                ..PermissionProfileConfig::default()
            },
        );
        let policy = permission_profile_policy(&config, None, SandboxBackendPlatform::Unsupported)
            .expect("permission profile policy");

        assert!(
            resolve_configured_permission_profile(&policy, "unknown-parent")
                .expect_err("unknown parent")
                .contains("unknown permission profile: missing")
        );
        assert!(resolve_configured_permission_profile(&policy, "cycle-a")
            .expect_err("cycle")
            .contains("cycle-a -> cycle-b -> cycle-a"));
    }

    #[test]
    fn request_profile_overrides_configured_default_and_unknown_default_fails_closed() {
        let mut config = Config::default();
        config.default_permissions = Some(READ_ONLY_PROFILE_ID.to_string());
        config
            .permissions
            .insert("custom".to_string(), PermissionProfileConfig::default());
        let policy = permission_profile_policy(&config, None, SandboxBackendPlatform::Unsupported)
            .expect("permission profile policy");

        assert_eq!(
            resolve_permission_profile_for_request(&policy, None)
                .expect("configured default")
                .expect("default profile")
                .id,
            READ_ONLY_PROFILE_ID
        );
        assert_eq!(
            resolve_permission_profile_for_request(&policy, Some("custom"))
                .expect("explicit profile")
                .expect("explicit profile")
                .id,
            "custom"
        );

        let mut invalid = config;
        invalid.default_permissions = Some("missing".to_string());
        let invalid_policy =
            permission_profile_policy(&invalid, None, SandboxBackendPlatform::Unsupported)
                .expect("invalid default is resolved at request time");
        assert!(
            resolve_permission_profile_for_request(&invalid_policy, None)
                .expect_err("unknown default must fail closed")
                .contains("unknown permission profile: missing")
        );
    }

    #[test]
    fn metadata_projection_clears_grants_when_switching_to_builtin_profile() {
        let mut metadata = Map::from_iter([(
            "grantedPermissions".to_string(),
            json!({"network": {"enabled": true}}),
        )]);
        apply_resolved_permission_profile_to_metadata(
            &mut metadata,
            ResolvedPermissionProfile {
                id: READ_ONLY_PROFILE_ID.to_string(),
                sandbox_policy: "read-only".to_string(),
                description: None,
                extends: None,
                granted_permissions: None,
            },
        )
        .expect("builtin profile projection");
        assert!(metadata.get("grantedPermissions").is_none());
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
