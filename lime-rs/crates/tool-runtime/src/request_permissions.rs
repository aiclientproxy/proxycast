use crate::tool_definition::RuntimeToolDefinition;
use app_server_protocol::protocol::v2::{
    AdditionalFileSystemPermissions, FileSystemPath, FileSystemSpecialPath,
    RequestPermissionProfile,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Component, Path, PathBuf};

pub const REQUEST_PERMISSIONS_TOOL_NAME: &str = "request_permissions";

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestPermissionsArgs {
    #[serde(default, alias = "environmentId")]
    pub environment_id: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    pub permissions: RequestPermissionProfile,
}

pub fn request_permissions_tool_definition() -> RuntimeToolDefinition {
    RuntimeToolDefinition::new(
        REQUEST_PERMISSIONS_TOOL_NAME,
        "Request additional filesystem or network permissions from the user and wait for the client to grant a subset. Relative filesystem paths resolve against the active turn cwd. Granted permissions apply to later commands in the current turn, or for the rest of the session when approved at session scope.",
        json!({
            "type": "object",
            "properties": {
                "environment_id": {
                    "type": "string",
                    "description": "Environment id from the active environment context. Omit to use the primary environment."
                },
                "reason": {
                    "type": "string",
                    "description": "Optional short explanation for why the permissions are needed."
                },
                "permissions": {
                    "type": "object",
                    "properties": {
                        "network": {
                            "type": "object",
                            "properties": { "enabled": { "type": "boolean" } },
                            "additionalProperties": false
                        },
                        "file_system": {
                            "type": "object",
                            "properties": {
                                "read": { "type": "array", "items": { "type": "string" } },
                                "write": { "type": "array", "items": { "type": "string" } },
                                "glob_scan_max_depth": { "type": "integer", "minimum": 1 },
                                "entries": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "path": { "type": "object" },
                                            "access": { "type": "string", "enum": ["read", "write", "deny"] }
                                        },
                                        "required": ["path", "access"],
                                        "additionalProperties": false
                                    }
                                }
                            },
                            "additionalProperties": false
                        }
                    },
                    "additionalProperties": false
                }
            },
            "required": ["permissions"],
            "additionalProperties": false
        }),
    )
}

pub fn is_request_permissions_tool(tool_name: &str) -> bool {
    tool_name
        .trim()
        .eq_ignore_ascii_case(REQUEST_PERMISSIONS_TOOL_NAME)
}

pub fn parse_request_permissions_args(
    params: &Value,
    cwd: &Path,
) -> Result<RequestPermissionsArgs, String> {
    if !cwd.is_absolute() {
        return Err("request_permissions requires an absolute turn cwd".to_string());
    }
    let mut args: RequestPermissionsArgs = serde_json::from_value(params.clone())
        .map_err(|error| format!("request_permissions arguments are invalid: {error}"))?;
    if args
        .environment_id
        .as_deref()
        .is_some_and(|environment_id| environment_id.trim().is_empty())
    {
        return Err("request_permissions environment_id must not be empty".to_string());
    }
    if let Some(reason) = args.reason.as_mut() {
        *reason = reason.trim().to_string();
        if reason.is_empty() {
            args.reason = None;
        }
    }
    normalize_permission_profile(&mut args.permissions, cwd)?;
    if !permission_profile_has_effective_grant(&args.permissions) {
        return Err("request_permissions requires at least one permission".to_string());
    }
    Ok(args)
}

fn normalize_permission_profile(
    profile: &mut RequestPermissionProfile,
    cwd: &Path,
) -> Result<(), String> {
    let Some(file_system) = profile.file_system.as_mut() else {
        return Ok(());
    };
    normalize_legacy_paths(file_system.read.as_mut(), cwd)?;
    normalize_legacy_paths(file_system.write.as_mut(), cwd)?;
    if file_system.glob_scan_max_depth == Some(0) {
        return Err(
            "request_permissions glob_scan_max_depth must be greater than zero".to_string(),
        );
    }
    if let Some(entries) = file_system.entries.as_mut() {
        for entry in entries {
            match &mut entry.path {
                FileSystemPath::Path { path } => *path = normalize_path(path, cwd)?,
                FileSystemPath::GlobPattern { pattern } => {
                    *pattern = normalize_path(pattern, cwd)?;
                }
                FileSystemPath::Special { value } => normalize_special_path(value)?,
            }
        }
    }
    Ok(())
}

fn normalize_legacy_paths(paths: Option<&mut Vec<String>>, cwd: &Path) -> Result<(), String> {
    let Some(paths) = paths else {
        return Ok(());
    };
    for path in paths {
        *path = normalize_path(path, cwd)?;
    }
    Ok(())
}

fn normalize_special_path(value: &mut FileSystemSpecialPath) -> Result<(), String> {
    let subpath = match value {
        FileSystemSpecialPath::ProjectRoots { subpath }
        | FileSystemSpecialPath::Unknown { subpath, .. } => subpath,
        _ => return Ok(()),
    };
    let Some(subpath) = subpath.as_mut() else {
        return Ok(());
    };
    let normalized = lexical_normalize(Path::new(subpath))?;
    if normalized.is_absolute() {
        return Err("request_permissions special subpath must be relative".to_string());
    }
    *subpath = normalized.to_string_lossy().into_owned();
    Ok(())
}

fn normalize_path(path: &str, cwd: &Path) -> Result<String, String> {
    let path = path.trim();
    if path.is_empty() {
        return Err("request_permissions filesystem path must not be empty".to_string());
    }
    let path = PathBuf::from(path);
    let path = if path.is_absolute() {
        path
    } else {
        cwd.join(path)
    };
    Ok(lexical_normalize(&path)?.to_string_lossy().into_owned())
}

fn lexical_normalize(path: &Path) -> Result<PathBuf, String> {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() {
                    return Err("request_permissions path escapes its filesystem root".to_string());
                }
            }
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::Normal(value) => normalized.push(value),
        }
    }
    Ok(normalized)
}

fn permission_profile_has_effective_grant(profile: &RequestPermissionProfile) -> bool {
    profile
        .network
        .as_ref()
        .and_then(|network| network.enabled)
        .unwrap_or(false)
        || profile
            .file_system
            .as_ref()
            .is_some_and(file_system_has_effective_grant)
}

fn file_system_has_effective_grant(file_system: &AdditionalFileSystemPermissions) -> bool {
    file_system
        .read
        .as_ref()
        .is_some_and(|paths| !paths.is_empty())
        || file_system
            .write
            .as_ref()
            .is_some_and(|paths| !paths.is_empty())
        || file_system.glob_scan_max_depth.is_some()
        || file_system
            .entries
            .as_ref()
            .is_some_and(|entries| !entries.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_permissions_normalizes_relative_paths_and_aliases() {
        let args = parse_request_permissions_args(
            &json!({
                "environment_id": "local",
                "permissions": {
                    "file_system": {
                        "write": ["generated/output.txt"],
                        "glob_scan_max_depth": 3
                    }
                }
            }),
            Path::new("/tmp/workspace"),
        )
        .expect("typed permissions");
        let file_system = args.permissions.file_system.expect("filesystem");
        assert_eq!(
            file_system.write,
            Some(vec!["/tmp/workspace/generated/output.txt".to_string()])
        );
        assert_eq!(file_system.glob_scan_max_depth, Some(3));
    }

    #[test]
    fn request_permissions_rejects_empty_or_escaping_profiles() {
        assert!(parse_request_permissions_args(
            &json!({ "permissions": { "network": { "enabled": false } } }),
            Path::new("/tmp/workspace"),
        )
        .is_err());
        assert!(parse_request_permissions_args(
            &json!({
                "permissions": { "file_system": { "write": ["../../../../../../escape"] } }
            }),
            Path::new("/tmp/workspace"),
        )
        .is_err());
    }
}
