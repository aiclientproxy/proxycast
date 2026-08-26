use super::*;
#[cfg(test)]
use app_server_protocol::protocol::v2::{AdditionalFileSystemPermissions, FileSystemSandboxEntry};
use app_server_protocol::protocol::v2::{
    FileSystemAccessMode, FileSystemPath, FileSystemSpecialPath, GrantedPermissionProfile,
};
use std::os::windows::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use windows_sys::Win32::Foundation::ERROR_INSUFFICIENT_BUFFER;
use windows_sys::Win32::Security::{
    GetFileSecurityW, GetSecurityDescriptorControl, SetFileSecurityW, DACL_SECURITY_INFORMATION,
    PROTECTED_DACL_SECURITY_INFORMATION, SE_DACL_PROTECTED, UNPROTECTED_DACL_SECURITY_INFORMATION,
};
use windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT;

const PROTECTED_METADATA_NAMES: [&str; 3] = [".git", ".codex", ".agents"];
const ICACLS_TIMEOUT: Duration = Duration::from_secs(15);

#[derive(Debug)]
pub(super) struct AclPlan {
    read_grants: Vec<PathBuf>,
    write_grants: Vec<PathBuf>,
    write_denies: Vec<PathBuf>,
    access_denies: Vec<PathBuf>,
}

pub(super) fn build_acl_plan(
    cwd: &Path,
    policy: RequestedSandboxPolicy,
    permissions: Option<&GrantedPermissionProfile>,
) -> io::Result<AclPlan> {
    let mut read_grants = vec![canonical_existing(cwd)?];
    let mut write_grants = Vec::new();
    let mut write_denies = Vec::new();
    let mut access_denies = Vec::new();
    if policy == RequestedSandboxPolicy::WorkspaceWrite {
        write_grants.push(canonical_existing(cwd)?);
        for name in PROTECTED_METADATA_NAMES {
            let path = cwd.join(name);
            if path.exists() {
                write_denies.push(canonical_existing(&path)?);
            }
        }
    }

    if let Some(file_system) = permissions.and_then(|profile| profile.file_system.as_ref()) {
        for path in file_system.write.as_deref().unwrap_or_default() {
            write_grants.push(canonical_absolute(Path::new(path))?);
        }
        for entry in file_system.entries.as_deref().unwrap_or_default() {
            let path = resolve_permission_path(&entry.path, cwd)?;
            let Some(path) = path else {
                continue;
            };
            match entry.access {
                FileSystemAccessMode::Read => {
                    read_grants.push(path.clone());
                    write_denies.push(path);
                }
                FileSystemAccessMode::Write => write_grants.push(path),
                FileSystemAccessMode::Deny => access_denies.push(path),
            }
        }
    }
    read_grants.sort();
    read_grants.dedup();
    write_grants.sort();
    write_grants.dedup();
    access_denies.sort();
    access_denies.dedup();
    write_grants.retain(|path| access_denies.binary_search(path).is_err());
    read_grants.retain(|path| write_grants.binary_search(path).is_err());
    read_grants.retain(|path| access_denies.binary_search(path).is_err());
    write_denies.sort();
    write_denies.dedup();
    write_denies.retain(|path| access_denies.binary_search(path).is_err());
    Ok(AclPlan {
        read_grants,
        write_grants,
        write_denies,
        access_denies,
    })
}

fn resolve_permission_path(path: &FileSystemPath, cwd: &Path) -> io::Result<Option<PathBuf>> {
    match path {
        FileSystemPath::Path { path } => canonical_absolute(Path::new(path)).map(Some),
        FileSystemPath::GlobPattern { .. } => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Windows restricted token sandbox does not support glob filesystem permissions",
        )),
        FileSystemPath::Special { value } => match value {
            FileSystemSpecialPath::ProjectRoots { subpath } => {
                let path = subpath
                    .as_deref()
                    .map(|subpath| cwd.join(subpath))
                    .unwrap_or_else(|| cwd.to_path_buf());
                canonical_existing(&path).map(Some)
            }
            FileSystemSpecialPath::Tmpdir => canonical_existing(&std::env::temp_dir()).map(Some),
            FileSystemSpecialPath::Unknown { path, subpath } => {
                let base = PathBuf::from(path);
                if !base.is_absolute() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "Windows sandbox permission path must be absolute",
                    ));
                }
                let path = subpath
                    .as_deref()
                    .map(|subpath| base.join(subpath))
                    .unwrap_or(base);
                canonical_existing(&path).map(Some)
            }
            FileSystemSpecialPath::Root | FileSystemSpecialPath::SlashTmp => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Unix filesystem special path is unsupported by the Windows sandbox",
            )),
            FileSystemSpecialPath::Minimal => Ok(None),
        },
    }
}

fn canonical_absolute(path: &Path) -> io::Result<PathBuf> {
    if !path.is_absolute() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "Windows sandbox permission path must be absolute: {}",
                path.display()
            ),
        ));
    }
    canonical_existing(path)
}

fn canonical_existing(path: &Path) -> io::Result<PathBuf> {
    reject_reparse_components(path)?;
    std::fs::canonicalize(path).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "Windows sandbox path is invalid ({}): {error}",
                path.display()
            ),
        )
    })
}

fn reject_reparse_components(path: &Path) -> io::Result<()> {
    for ancestor in path.ancestors() {
        if ancestor.as_os_str().is_empty() {
            continue;
        }
        let metadata = std::fs::symlink_metadata(ancestor).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!(
                    "Windows sandbox cannot inspect path component {}: {error}",
                    ancestor.display()
                ),
            )
        })?;
        if metadata.file_type().is_symlink()
            || metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!(
                    "Windows sandbox rejects reparse-point path component: {}",
                    ancestor.display()
                ),
            ));
        }
    }
    Ok(())
}

#[derive(Debug)]
pub(super) struct AclLease {
    snapshots: Vec<DaclSnapshot>,
}

#[derive(Debug)]
struct DaclSnapshot {
    path: PathBuf,
    descriptor: Vec<u8>,
    protected: bool,
}

impl AclLease {
    pub(super) fn acquire(
        sandbox_group_sid: &str,
        capability_sid: &str,
        plan: AclPlan,
    ) -> io::Result<Self> {
        let mut lease = Self {
            snapshots: Vec::new(),
        };
        for path in plan.read_grants {
            if let Err(error) = lease.capture(&path).and_then(|()| {
                grant_access(&path, sandbox_group_sid, capability_sid, "(OI)(CI)(RX)")
            }) {
                return Err(lease.rollback_after(error));
            }
        }
        for path in plan.write_grants {
            if let Err(error) = lease.capture(&path).and_then(|()| {
                grant_access(&path, sandbox_group_sid, capability_sid, "(OI)(CI)(RX,W,D)")
            }) {
                return Err(lease.rollback_after(error));
            }
        }
        for path in plan.write_denies {
            if let Err(error) = lease.capture(&path).and_then(|()| {
                let deny = format!("*{capability_sid}:(OI)(CI)(W,D,DC)");
                run_icacls(&path, &["/deny", &deny, "/C", "/Q"])
            }) {
                return Err(lease.rollback_after(error));
            }
        }
        for path in plan.access_denies {
            if let Err(error) = lease
                .capture(&path)
                .and_then(|()| deny_access(&path, sandbox_group_sid, capability_sid))
            {
                return Err(lease.rollback_after(error));
            }
        }
        Ok(lease)
    }

    fn capture(&mut self, path: &Path) -> io::Result<()> {
        if self.snapshots.iter().any(|snapshot| snapshot.path == path) {
            return Ok(());
        }
        self.snapshots.push(DaclSnapshot::capture(path)?);
        Ok(())
    }

    fn rollback_after(&mut self, error: io::Error) -> io::Error {
        match self.rollback() {
            Ok(()) => error,
            Err(rollback_error) => io::Error::other(format!(
                "{error}; Windows sandbox ACL rollback also failed: {rollback_error}"
            )),
        }
    }

    fn rollback(&mut self) -> io::Result<()> {
        let mut failures = Vec::new();
        for snapshot in self.snapshots.iter().rev() {
            if let Err(error) = snapshot.restore() {
                failures.push(format!("{}: {error}", snapshot.path.display()));
            }
        }
        self.snapshots.clear();
        if failures.is_empty() {
            Ok(())
        } else {
            Err(io::Error::other(failures.join("; ")))
        }
    }
}

impl Drop for AclLease {
    fn drop(&mut self) {
        if let Err(error) = self.rollback() {
            tracing::warn!(%error, "failed to restore Windows sandbox ACL lease");
        }
    }
}

impl DaclSnapshot {
    fn capture(path: &Path) -> io::Result<Self> {
        let path_w = to_wide(path.as_os_str());
        let mut needed = 0;
        let first = unsafe {
            GetFileSecurityW(
                path_w.as_ptr(),
                DACL_SECURITY_INFORMATION,
                ptr::null_mut(),
                0,
                &mut needed,
            )
        };
        if first != 0 || unsafe { GetLastError() } != ERROR_INSUFFICIENT_BUFFER || needed == 0 {
            return Err(last_os_error("GetFileSecurityW size"));
        }
        let mut descriptor = vec![0u8; needed as usize];
        if unsafe {
            GetFileSecurityW(
                path_w.as_ptr(),
                DACL_SECURITY_INFORMATION,
                descriptor.as_mut_ptr() as *mut c_void,
                needed,
                &mut needed,
            )
        } == 0
        {
            return Err(last_os_error("GetFileSecurityW"));
        }
        descriptor.truncate(needed as usize);
        let mut control = 0;
        let mut revision = 0;
        if unsafe {
            GetSecurityDescriptorControl(
                descriptor.as_mut_ptr() as *mut c_void,
                &mut control,
                &mut revision,
            )
        } == 0
        {
            return Err(last_os_error("GetSecurityDescriptorControl"));
        }
        Ok(Self {
            path: path.to_path_buf(),
            descriptor,
            protected: control & SE_DACL_PROTECTED != 0,
        })
    }

    fn restore(&self) -> io::Result<()> {
        let path_w = to_wide(self.path.as_os_str());
        let inheritance = if self.protected {
            PROTECTED_DACL_SECURITY_INFORMATION
        } else {
            UNPROTECTED_DACL_SECURITY_INFORMATION
        };
        if unsafe {
            SetFileSecurityW(
                path_w.as_ptr(),
                DACL_SECURITY_INFORMATION | inheritance,
                self.descriptor.as_ptr() as *mut c_void,
            )
        } == 0
        {
            Err(last_os_error("SetFileSecurityW"))
        } else {
            Ok(())
        }
    }
}

fn grant_access(
    path: &Path,
    sandbox_group_sid: &str,
    capability_sid: &str,
    permissions: &str,
) -> io::Result<()> {
    let group_grant = format!("*{sandbox_group_sid}:{permissions}");
    run_icacls(path, &["/grant:r", &group_grant, "/C", "/Q"])?;
    let capability_grant = format!("*{capability_sid}:{permissions}");
    run_icacls(path, &["/grant:r", &capability_grant, "/C", "/Q"])
}

fn deny_access(path: &Path, sandbox_group_sid: &str, capability_sid: &str) -> io::Result<()> {
    let group_deny = format!("*{sandbox_group_sid}:(OI)(CI)(F)");
    run_icacls(path, &["/deny", &group_deny, "/C", "/Q"])?;
    let capability_deny = format!("*{capability_sid}:(OI)(CI)(F)");
    run_icacls(path, &["/deny", &capability_deny, "/C", "/Q"])
}

fn run_icacls(path: &Path, args: &[&str]) -> io::Result<()> {
    let mut child = Command::new(icacls_path()?)
        .arg(path)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    let status = wait_for_icacls(&mut child, path, ICACLS_TIMEOUT)?;
    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "icacls failed for {} with status {status}",
            path.display()
        )))
    }
}

fn icacls_path() -> io::Result<PathBuf> {
    let system_root = std::env::var_os("SystemRoot")
        .or_else(|| std::env::var_os("windir"))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "Windows sandbox cannot locate SystemRoot for icacls.exe",
            )
        })?;
    let path = PathBuf::from(system_root)
        .join("System32")
        .join("icacls.exe");
    if !path.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Windows sandbox cannot locate {}", path.display()),
        ));
    }
    Ok(path)
}

fn wait_for_icacls(child: &mut Child, path: &Path, timeout: Duration) -> io::Result<ExitStatus> {
    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!(
                    "icacls timed out for {} after {:?}",
                    path.display(),
                    timeout
                ),
            ));
        }
        thread::sleep(Duration::from_millis(25));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_acl_plan_grants_workspace_and_denies_metadata() {
        let root = tempfile::tempdir().expect("workspace");
        std::fs::create_dir(root.path().join(".git")).expect("git metadata");
        let plan = build_acl_plan(root.path(), RequestedSandboxPolicy::WorkspaceWrite, None)
            .expect("ACL plan");

        assert_eq!(
            plan.write_grants,
            vec![std::fs::canonicalize(root.path()).unwrap()]
        );
        assert!(plan.read_grants.is_empty());
        assert_eq!(
            plan.write_denies,
            vec![std::fs::canonicalize(root.path().join(".git")).unwrap()]
        );
        assert!(plan.access_denies.is_empty());
    }

    #[test]
    fn read_only_acl_plan_grants_workspace_read_access() {
        let root = tempfile::tempdir().expect("workspace");
        let plan =
            build_acl_plan(root.path(), RequestedSandboxPolicy::ReadOnly, None).expect("ACL plan");

        assert_eq!(
            plan.read_grants,
            vec![std::fs::canonicalize(root.path()).unwrap()]
        );
        assert!(plan.write_grants.is_empty());
        assert!(plan.write_denies.is_empty());
        assert!(plan.access_denies.is_empty());
    }

    #[test]
    fn explicit_read_and_deny_entries_create_distinct_carveouts() {
        let root = tempfile::tempdir().expect("workspace");
        let read_only = root.path().join("read-only");
        let denied = root.path().join("denied");
        std::fs::create_dir(&read_only).expect("read-only path");
        std::fs::create_dir(&denied).expect("denied path");
        let permissions = GrantedPermissionProfile {
            network: None,
            file_system: Some(AdditionalFileSystemPermissions {
                read: None,
                write: None,
                glob_scan_max_depth: None,
                entries: Some(vec![
                    FileSystemSandboxEntry {
                        path: FileSystemPath::Path {
                            path: read_only.to_string_lossy().to_string(),
                        },
                        access: FileSystemAccessMode::Read,
                    },
                    FileSystemSandboxEntry {
                        path: FileSystemPath::Path {
                            path: denied.to_string_lossy().to_string(),
                        },
                        access: FileSystemAccessMode::Deny,
                    },
                ]),
            }),
        };
        let plan = build_acl_plan(
            root.path(),
            RequestedSandboxPolicy::WorkspaceWrite,
            Some(&permissions),
        )
        .expect("ACL plan");

        assert!(plan
            .read_grants
            .contains(&std::fs::canonicalize(&read_only).unwrap()));
        assert!(plan
            .write_denies
            .contains(&std::fs::canonicalize(&read_only).unwrap()));
        assert!(plan
            .access_denies
            .contains(&std::fs::canonicalize(&denied).unwrap()));
    }

    #[test]
    fn ordinary_existing_paths_are_not_reparse_points() {
        let root = tempfile::tempdir().expect("workspace");
        reject_reparse_components(root.path()).expect("ordinary path should be accepted");
    }

    #[test]
    fn icacls_wait_returns_completed_child_status() {
        let mut child = Command::new("cmd.exe")
            .args(["/C", "exit", "0"])
            .spawn()
            .expect("cmd should start");
        let status = wait_for_icacls(
            &mut child,
            Path::new("C:\\workspace"),
            Duration::from_secs(2),
        )
        .expect("completed child should return status");
        assert!(status.success());
    }
}
