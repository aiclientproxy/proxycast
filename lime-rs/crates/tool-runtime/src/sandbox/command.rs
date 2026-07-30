use super::{parse_requested_sandbox_policy, RequestedSandboxPolicy, SandboxBackend};
use app_server_protocol::protocol::v2::{
    AdditionalFileSystemPermissions, FileSystemAccessMode, FileSystemPath, FileSystemSpecialPath,
    GrantedPermissionProfile,
};
use std::fmt;
use std::path::{Path, PathBuf};

const MACOS_SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";
const PROTECTED_METADATA_NAMES: [&str; 3] = [".git", ".codex", ".agents"];

const SEATBELT_BASE_POLICY: &str = r#"
(version 1)
(deny default)
(allow process-exec)
(allow process-fork)
(allow signal (target same-sandbox))
(allow process-info* (target same-sandbox))
(allow file-read*)
(allow file-map-executable)
(allow file-write-data (literal "/dev/null"))
(allow file-read* file-write* (literal "/dev/null"))
(allow file-read* file-write* (subpath "/dev/fd"))
(allow file-read* file-write* (subpath "/tmp"))
(allow file-read* file-write* (subpath "/private/tmp"))
(allow file-read* file-write* (subpath "/var/tmp"))
(allow file-read* file-write* (subpath "/private/var/tmp"))
(allow sysctl-read)
(allow mach-lookup)
(allow ipc-posix-sem)
(allow ipc-posix-shm-read*)
(allow ipc-posix-shm-write-create)
(allow ipc-posix-shm-write-data)
(allow ipc-posix-shm-write-unlink)
(allow user-preference-read)
(allow pseudo-tty)
(allow file-read* file-write* file-ioctl (literal "/dev/ptmx"))
(allow file-read* file-write* file-ioctl (regex #"^/dev/ttys[0-9]+"))
(allow network-outbound (literal "/private/var/run/syslog"))
"#;

#[derive(Debug)]
pub struct SandboxCommandRequest<'a> {
    pub backend: SandboxBackend,
    pub requested_policy: Option<&'a str>,
    pub command: Vec<String>,
    pub working_directory: &'a Path,
    pub granted_permissions: Option<&'a GrantedPermissionProfile>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SandboxCommandError {
    EmptyCommand,
    UnsupportedBackend(SandboxBackend),
    UnsupportedPolicy(String),
}

impl fmt::Display for SandboxCommandError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCommand => formatter.write_str("sandbox command must not be empty"),
            Self::UnsupportedBackend(backend) => {
                write!(
                    formatter,
                    "sandbox backend '{}' is unavailable",
                    backend.label()
                )
            }
            Self::UnsupportedPolicy(policy) => {
                write!(formatter, "sandbox policy '{policy}' is unsupported")
            }
        }
    }
}

impl std::error::Error for SandboxCommandError {}

pub fn prepare_sandbox_command(
    request: SandboxCommandRequest<'_>,
) -> Result<Vec<String>, SandboxCommandError> {
    if request.command.is_empty() {
        return Err(SandboxCommandError::EmptyCommand);
    }
    let policy = parse_requested_sandbox_policy(request.requested_policy)
        .unwrap_or(RequestedSandboxPolicy::WorkspaceWrite);
    if policy == RequestedSandboxPolicy::DangerFullAccess {
        return Err(SandboxCommandError::UnsupportedPolicy(
            policy.label().to_string(),
        ));
    }

    match request.backend {
        SandboxBackend::Seatbelt => Ok(prepare_seatbelt_command(
            request.command,
            request.working_directory,
            policy,
            request.granted_permissions,
        )),
        SandboxBackend::LinuxSandbox => Ok(prepare_bubblewrap_command(
            request.command,
            request.working_directory,
            policy,
            request.granted_permissions,
        )),
        SandboxBackend::None | SandboxBackend::RestrictedToken => {
            Err(SandboxCommandError::UnsupportedBackend(request.backend))
        }
    }
}

fn prepare_seatbelt_command(
    command: Vec<String>,
    working_directory: &Path,
    policy: RequestedSandboxPolicy,
    granted_permissions: Option<&GrantedPermissionProfile>,
) -> Vec<String> {
    let mut profile = SEATBELT_BASE_POLICY.to_string();
    if policy == RequestedSandboxPolicy::WorkspaceWrite {
        profile.push_str("\n(allow file-write* (subpath (param \"WORKSPACE\")))\n");
        for name in PROTECTED_METADATA_NAMES {
            profile.push_str(&format!(
                "(deny file-write* (regex #\"{}/{}(/.*)?$\"))\n",
                regex::escape(&working_directory.to_string_lossy()),
                regex::escape(name),
            ));
        }
    }
    if let Some(permissions) = granted_permissions {
        append_seatbelt_permissions(&mut profile, permissions, working_directory);
    }

    let mut wrapped = vec![
        MACOS_SANDBOX_EXEC.to_string(),
        "-p".to_string(),
        profile,
        format!("-DWORKSPACE={}", working_directory.to_string_lossy()),
        "--".to_string(),
    ];
    wrapped.extend(command);
    wrapped
}

fn prepare_bubblewrap_command(
    command: Vec<String>,
    working_directory: &Path,
    policy: RequestedSandboxPolicy,
    granted_permissions: Option<&GrantedPermissionProfile>,
) -> Vec<String> {
    let cwd = working_directory.to_string_lossy().to_string();
    let mut wrapped = vec![
        "bwrap".to_string(),
        "--die-with-parent".to_string(),
        "--new-session".to_string(),
        "--unshare-user".to_string(),
        "--unshare-pid".to_string(),
        "--unshare-uts".to_string(),
        "--unshare-ipc".to_string(),
        "--ro-bind".to_string(),
        "/".to_string(),
        "/".to_string(),
        "--dev-bind".to_string(),
        "/dev".to_string(),
        "/dev".to_string(),
        "--proc".to_string(),
        "/proc".to_string(),
    ];
    if !granted_permissions
        .and_then(|permissions| permissions.network.as_ref())
        .and_then(|network| network.enabled)
        .unwrap_or(false)
    {
        wrapped.insert(7, "--unshare-net".to_string());
    }
    for temporary_root in ["/tmp", "/var/tmp"] {
        if Path::new(temporary_root).is_dir() {
            wrapped.extend([
                "--bind".to_string(),
                temporary_root.to_string(),
                temporary_root.to_string(),
            ]);
        }
    }
    if policy == RequestedSandboxPolicy::WorkspaceWrite {
        wrapped.extend(["--bind".to_string(), cwd.clone(), cwd.clone()]);
        for name in PROTECTED_METADATA_NAMES {
            let path = working_directory.join(name);
            if path.exists() {
                let path = path.to_string_lossy().to_string();
                wrapped.extend(["--ro-bind".to_string(), path.clone(), path]);
            }
        }
    }
    if let Some(file_system) = granted_permissions.and_then(|value| value.file_system.as_ref()) {
        append_bubblewrap_permissions(&mut wrapped, file_system, working_directory);
    }
    wrapped.extend(["--chdir".to_string(), cwd, "--".to_string()]);
    wrapped.extend(command);
    wrapped
}

fn append_seatbelt_permissions(
    profile: &mut String,
    permissions: &GrantedPermissionProfile,
    working_directory: &Path,
) {
    if permissions
        .network
        .as_ref()
        .and_then(|network| network.enabled)
        .unwrap_or(false)
    {
        profile.push_str("\n(allow network*)\n");
    }
    let Some(file_system) = permissions.file_system.as_ref() else {
        return;
    };
    for path in file_system.read.as_deref().unwrap_or_default() {
        append_seatbelt_path(profile, Path::new(path), FileSystemAccessMode::Read);
    }
    for path in file_system.write.as_deref().unwrap_or_default() {
        append_seatbelt_path(profile, Path::new(path), FileSystemAccessMode::Write);
    }
    for entry in file_system.entries.as_deref().unwrap_or_default() {
        match &entry.path {
            FileSystemPath::Path { path } => {
                append_seatbelt_path(profile, Path::new(path), entry.access)
            }
            FileSystemPath::GlobPattern { pattern } => {
                append_seatbelt_pattern(profile, pattern, entry.access)
            }
            FileSystemPath::Special { value } => {
                if let Some(path) = special_path(value, working_directory) {
                    append_seatbelt_path(profile, &path, entry.access);
                }
            }
        }
    }
}

fn append_seatbelt_path(profile: &mut String, path: &Path, access: FileSystemAccessMode) {
    let escaped = regex::escape(&path.to_string_lossy());
    append_seatbelt_regex(profile, &format!("^{escaped}(/.*)?$"), access);
}

fn append_seatbelt_pattern(profile: &mut String, pattern: &str, access: FileSystemAccessMode) {
    append_seatbelt_regex(profile, &glob_pattern_regex(pattern), access);
}

fn append_seatbelt_regex(profile: &mut String, pattern: &str, access: FileSystemAccessMode) {
    let operation = match access {
        FileSystemAccessMode::Read => "allow file-read*",
        FileSystemAccessMode::Write => "allow file-read* file-write*",
        FileSystemAccessMode::Deny => "deny file-read* file-write*",
    };
    profile.push_str(&format!("\n({operation} (regex #\"{pattern}\"))\n"));
}

fn glob_pattern_regex(pattern: &str) -> String {
    let mut regex = String::from("^");
    let mut chars = pattern.chars().peekable();
    while let Some(character) = chars.next() {
        match character {
            '*' if chars.peek() == Some(&'*') => {
                chars.next();
                regex.push_str(".*");
            }
            '*' => regex.push_str("[^/]*"),
            '?' => regex.push_str("[^/]"),
            character => regex.push_str(&regex::escape(&character.to_string())),
        }
    }
    regex.push('$');
    regex
}

fn append_bubblewrap_permissions(
    wrapped: &mut Vec<String>,
    file_system: &AdditionalFileSystemPermissions,
    working_directory: &Path,
) {
    for path in file_system.read.as_deref().unwrap_or_default() {
        append_bubblewrap_path(wrapped, Path::new(path), false);
    }
    for path in file_system.write.as_deref().unwrap_or_default() {
        append_bubblewrap_path(wrapped, Path::new(path), true);
    }
    for entry in file_system.entries.as_deref().unwrap_or_default() {
        let path = match &entry.path {
            FileSystemPath::Path { path } => Some(PathBuf::from(path)),
            FileSystemPath::Special { value } => special_path(value, working_directory),
            FileSystemPath::GlobPattern { .. } => None,
        };
        if let Some(path) = path {
            match entry.access {
                FileSystemAccessMode::Read => append_bubblewrap_path(wrapped, &path, false),
                FileSystemAccessMode::Write => append_bubblewrap_path(wrapped, &path, true),
                FileSystemAccessMode::Deny => {}
            }
        }
    }
}

fn append_bubblewrap_path(wrapped: &mut Vec<String>, path: &Path, writable: bool) {
    if !path.exists() {
        return;
    }
    let path = path.to_string_lossy().into_owned();
    wrapped.extend([
        if writable { "--bind" } else { "--ro-bind" }.to_string(),
        path.clone(),
        path,
    ]);
}

fn special_path(value: &FileSystemSpecialPath, working_directory: &Path) -> Option<PathBuf> {
    match value {
        FileSystemSpecialPath::Root => Some(PathBuf::from("/")),
        FileSystemSpecialPath::ProjectRoots { subpath } => Some(
            subpath
                .as_deref()
                .map(|subpath| working_directory.join(subpath))
                .unwrap_or_else(|| working_directory.to_path_buf()),
        ),
        FileSystemSpecialPath::Tmpdir => Some(std::env::temp_dir()),
        FileSystemSpecialPath::SlashTmp => Some(PathBuf::from("/tmp")),
        FileSystemSpecialPath::Unknown { path, subpath } => {
            let path = PathBuf::from(path);
            path.is_absolute().then(|| {
                subpath
                    .as_deref()
                    .map(|subpath| path.join(subpath))
                    .unwrap_or(path)
            })
        }
        FileSystemSpecialPath::Minimal => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::protocol::v2::AdditionalNetworkPermissions;

    #[test]
    fn seatbelt_workspace_write_wraps_command_and_protects_metadata() {
        let command = prepare_sandbox_command(SandboxCommandRequest {
            backend: SandboxBackend::Seatbelt,
            requested_policy: Some("workspace-write"),
            command: vec!["sh".to_string(), "-c".to_string(), "pwd".to_string()],
            working_directory: Path::new("/tmp/workspace"),
            granted_permissions: None,
        })
        .expect("seatbelt command");

        assert_eq!(command[0], MACOS_SANDBOX_EXEC);
        assert!(command[2].contains("(allow file-write* (subpath (param \"WORKSPACE\")))"));
        assert!(command[2].contains(r"/\.git(/.*)?$"));
        assert_eq!(command[3], "-DWORKSPACE=/tmp/workspace");
        assert_eq!(&command[5..], ["sh", "-c", "pwd"]);
    }

    #[test]
    fn bubblewrap_read_only_does_not_bind_workspace_writable() {
        let command = prepare_sandbox_command(SandboxCommandRequest {
            backend: SandboxBackend::LinuxSandbox,
            requested_policy: Some("read-only"),
            command: vec!["pwd".to_string()],
            working_directory: Path::new("/workspace"),
            granted_permissions: None,
        })
        .expect("bubblewrap command");

        assert_eq!(command[0], "bwrap");
        assert!(command
            .windows(2)
            .any(|args| args == ["--chdir", "/workspace"]));
        assert!(!command
            .windows(3)
            .any(|args| args == ["--bind", "/workspace", "/workspace"]));
    }

    #[test]
    fn restricted_token_backend_fails_closed_without_runner() {
        let error = prepare_sandbox_command(SandboxCommandRequest {
            backend: SandboxBackend::RestrictedToken,
            requested_policy: Some("workspace-write"),
            command: vec!["cmd.exe".to_string()],
            working_directory: Path::new("C:/workspace"),
            granted_permissions: None,
        })
        .expect_err("restricted token must not fall back to unsandboxed execution");

        assert_eq!(
            error,
            SandboxCommandError::UnsupportedBackend(SandboxBackend::RestrictedToken)
        );
    }

    #[test]
    fn seatbelt_lowers_granted_network_and_write_permissions() {
        let permissions = GrantedPermissionProfile {
            network: Some(AdditionalNetworkPermissions {
                enabled: Some(true),
            }),
            file_system: Some(AdditionalFileSystemPermissions {
                read: None,
                write: Some(vec!["/tmp/shared-output".to_string()]),
                glob_scan_max_depth: None,
                entries: None,
            }),
        };
        let command = prepare_sandbox_command(SandboxCommandRequest {
            backend: SandboxBackend::Seatbelt,
            requested_policy: Some("workspace-write"),
            command: vec!["pwd".to_string()],
            working_directory: Path::new("/tmp/workspace"),
            granted_permissions: Some(&permissions),
        })
        .expect("seatbelt command");

        assert!(command[2].contains("(allow network*)"));
        assert!(command[2].contains("allow file-read* file-write*"));
        assert!(command[2].contains("^/tmp/shared\\-output(/.*)?$"));
    }

    #[test]
    fn bubblewrap_keeps_network_isolated_until_explicitly_granted() {
        let root = tempfile::tempdir().expect("sandbox root");
        let writable = root.path().join("shared-output");
        std::fs::create_dir(&writable).expect("writable directory");
        let permissions = GrantedPermissionProfile {
            network: Some(AdditionalNetworkPermissions {
                enabled: Some(true),
            }),
            file_system: Some(AdditionalFileSystemPermissions {
                read: None,
                write: Some(vec![writable.to_string_lossy().into_owned()]),
                glob_scan_max_depth: None,
                entries: None,
            }),
        };
        let granted = prepare_sandbox_command(SandboxCommandRequest {
            backend: SandboxBackend::LinuxSandbox,
            requested_policy: Some("workspace-write"),
            command: vec!["pwd".to_string()],
            working_directory: root.path(),
            granted_permissions: Some(&permissions),
        })
        .expect("granted bubblewrap command");
        assert!(!granted.iter().any(|argument| argument == "--unshare-net"));
        let writable = writable.to_string_lossy();
        assert!(granted.windows(3).any(|arguments| {
            arguments[0] == "--bind"
                && arguments[1] == writable.as_ref()
                && arguments[2] == writable.as_ref()
        }));

        let isolated = prepare_sandbox_command(SandboxCommandRequest {
            backend: SandboxBackend::LinuxSandbox,
            requested_policy: Some("workspace-write"),
            command: vec!["pwd".to_string()],
            working_directory: root.path(),
            granted_permissions: None,
        })
        .expect("isolated bubblewrap command");
        assert!(isolated.iter().any(|argument| argument == "--unshare-net"));
    }
}
