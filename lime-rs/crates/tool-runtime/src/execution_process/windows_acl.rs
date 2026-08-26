use super::*;
#[cfg(test)]
use app_server_protocol::protocol::v2::{AdditionalFileSystemPermissions, FileSystemSandboxEntry};
use app_server_protocol::protocol::v2::{
    FileSystemAccessMode, FileSystemPath, FileSystemSpecialPath, GrantedPermissionProfile,
};
use std::collections::HashMap;
use std::os::windows::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use windows_sys::Win32::Foundation::ERROR_INSUFFICIENT_BUFFER;
use windows_sys::Win32::Foundation::ERROR_SUCCESS;
use windows_sys::Win32::Security::Authorization::{
    GetNamedSecurityInfoW, SetEntriesInAclW, SetNamedSecurityInfoW, DENY_ACCESS, EXPLICIT_ACCESS_W,
    SET_ACCESS, TRUSTEE_IS_SID, TRUSTEE_IS_UNKNOWN, TRUSTEE_W,
};
use windows_sys::Win32::Security::{
    GetFileSecurityW, GetSecurityDescriptorControl, SetFileSecurityW, DACL_SECURITY_INFORMATION,
    PROTECTED_DACL_SECURITY_INFORMATION, SE_DACL_PROTECTED, UNPROTECTED_DACL_SECURITY_INFORMATION,
};
use windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT;
use windows_sys::Win32::Storage::FileSystem::{
    DELETE, FILE_ALL_ACCESS, FILE_DELETE_CHILD, FILE_GENERIC_EXECUTE, FILE_GENERIC_READ,
    FILE_GENERIC_WRITE,
};

const PROTECTED_METADATA_NAMES: [&str; 3] = [".git", ".codex", ".agents"];
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
    environment: &HashMap<String, String>,
) -> io::Result<AclPlan> {
    let mut read_grants = vec![canonical_existing(cwd)?];
    let mut write_grants = Vec::new();
    let mut write_denies = Vec::new();
    let mut access_denies = Vec::new();
    if policy == RequestedSandboxPolicy::WorkspaceWrite {
        write_grants.push(canonical_existing(cwd)?);
        write_grants.extend(windows_temp_write_roots(environment)?);
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

fn windows_temp_write_roots(environment: &HashMap<String, String>) -> io::Result<Vec<PathBuf>> {
    ["TEMP", "TMP"]
        .into_iter()
        .filter_map(|key| {
            environment
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case(key))
                .map(|(_, value)| PathBuf::from(value.as_str()))
        })
        .filter(|path| path.is_absolute())
        .map(|path| canonical_existing(&path))
        .collect()
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
    paths: Vec<PathBuf>,
}

#[derive(Debug)]
struct DaclSnapshot {
    path: PathBuf,
    descriptor: Vec<u8>,
    protected: bool,
}

#[derive(Debug)]
struct SharedAclState {
    snapshot: DaclSnapshot,
    leases: usize,
}

fn acl_registry() -> &'static Mutex<HashMap<PathBuf, SharedAclState>> {
    static REGISTRY: OnceLock<Mutex<HashMap<PathBuf, SharedAclState>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

impl AclLease {
    pub(super) fn acquire(
        sandbox_group_sid: &str,
        capability_sid: &str,
        plan: AclPlan,
    ) -> io::Result<Self> {
        let mut paths = plan
            .read_grants
            .iter()
            .chain(&plan.write_grants)
            .chain(&plan.write_denies)
            .chain(&plan.access_denies)
            .cloned()
            .collect::<Vec<_>>();
        paths.sort();
        paths.dedup();
        let mut lease = Self { paths: Vec::new() };
        if let Err(error) = lease.reserve(&paths) {
            return Err(lease.rollback_after(error));
        }
        if let Err(error) = lease.apply(sandbox_group_sid, capability_sid, plan) {
            return Err(lease.rollback_after(error));
        }
        Ok(lease)
    }

    fn apply(
        &self,
        sandbox_group_sid: &str,
        capability_sid: &str,
        plan: AclPlan,
    ) -> io::Result<()> {
        // Serialize read-modify-write DACL updates with final snapshot restore.
        let _registry = acl_registry()
            .lock()
            .map_err(|_| io::Error::other("Windows sandbox ACL registry lock is poisoned"))?;
        for path in plan.read_grants {
            grant_access(
                &path,
                sandbox_group_sid,
                capability_sid,
                FILE_GENERIC_READ | FILE_GENERIC_EXECUTE,
            )?;
        }
        for path in plan.write_grants {
            grant_access(
                &path,
                sandbox_group_sid,
                capability_sid,
                FILE_GENERIC_READ | FILE_GENERIC_WRITE | FILE_GENERIC_EXECUTE | DELETE,
            )?;
        }
        for path in plan.write_denies {
            deny_write_access(&path, capability_sid)?;
        }
        for path in plan.access_denies {
            deny_access(&path, sandbox_group_sid, capability_sid)?;
        }
        Ok(())
    }

    fn reserve(&mut self, paths: &[PathBuf]) -> io::Result<()> {
        let mut registry = acl_registry()
            .lock()
            .map_err(|_| io::Error::other("Windows sandbox ACL registry lock is poisoned"))?;
        for path in paths {
            if let Some(state) = registry.get_mut(path) {
                state.leases = state.leases.checked_add(1).ok_or_else(|| {
                    io::Error::other(format!(
                        "Windows sandbox ACL lease count overflow for {}",
                        path.display()
                    ))
                })?;
            } else {
                registry.insert(
                    path.clone(),
                    SharedAclState {
                        snapshot: DaclSnapshot::capture(path)?,
                        leases: 1,
                    },
                );
            }
            self.paths.push(path.clone());
        }
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
        let mut registry = acl_registry()
            .lock()
            .map_err(|_| io::Error::other("Windows sandbox ACL registry lock is poisoned"))?;
        for path in self.paths.iter().rev() {
            let should_restore = match registry.get_mut(path) {
                Some(state) if state.leases > 0 => {
                    state.leases -= 1;
                    state.leases == 0
                }
                Some(_) => {
                    failures.push(format!(
                        "{}: ACL lease count is already zero",
                        path.display()
                    ));
                    false
                }
                None => {
                    failures.push(format!(
                        "{}: ACL lease registry entry is missing",
                        path.display()
                    ));
                    false
                }
            };
            if !should_restore {
                continue;
            }
            let restore_result = registry
                .get(path)
                .expect("ACL registry entry checked above")
                .snapshot
                .restore();
            match restore_result {
                Ok(()) => {
                    registry.remove(path);
                }
                Err(error) => failures.push(format!("{}: {error}", path.display())),
            }
        }
        self.paths.clear();
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
    permissions: u32,
) -> io::Result<()> {
    let group_sid = LocalSid::parse(sandbox_group_sid)?;
    let capability_sid = LocalSid::parse(capability_sid)?;
    set_acl_entries(
        path,
        &[
            explicit_access(group_sid.raw(), permissions, SET_ACCESS),
            explicit_access(capability_sid.raw(), permissions, SET_ACCESS),
        ],
    )
}

fn deny_access(path: &Path, sandbox_group_sid: &str, capability_sid: &str) -> io::Result<()> {
    let group_sid = LocalSid::parse(sandbox_group_sid)?;
    let capability_sid = LocalSid::parse(capability_sid)?;
    set_acl_entries(
        path,
        &[
            explicit_access(group_sid.raw(), FILE_ALL_ACCESS, DENY_ACCESS),
            explicit_access(capability_sid.raw(), FILE_ALL_ACCESS, DENY_ACCESS),
        ],
    )
}

fn deny_write_access(path: &Path, capability_sid: &str) -> io::Result<()> {
    let capability_sid = LocalSid::parse(capability_sid)?;
    set_acl_entries(
        path,
        &[explicit_access(
            capability_sid.raw(),
            FILE_GENERIC_WRITE | DELETE | FILE_DELETE_CHILD,
            DENY_ACCESS,
        )],
    )
}

fn explicit_access(sid: *mut c_void, permissions: u32, mode: i32) -> EXPLICIT_ACCESS_W {
    EXPLICIT_ACCESS_W {
        grfAccessPermissions: permissions,
        grfAccessMode: mode,
        grfInheritance: 0x3, // CONTAINER_INHERIT_ACE | OBJECT_INHERIT_ACE
        Trustee: TRUSTEE_W {
            pMultipleTrustee: std::ptr::null_mut(),
            MultipleTrusteeOperation: 0,
            TrusteeForm: TRUSTEE_IS_SID,
            TrusteeType: TRUSTEE_IS_UNKNOWN,
            ptstrName: sid as *mut u16,
        },
    }
}

fn set_acl_entries(path: &Path, entries: &[EXPLICIT_ACCESS_W]) -> io::Result<()> {
    let path_w = to_wide(path.as_os_str());
    let mut security_descriptor = ptr::null_mut();
    let mut dacl = ptr::null_mut();
    let result = unsafe {
        GetNamedSecurityInfoW(
            path_w.as_ptr(),
            1, // SE_FILE_OBJECT
            DACL_SECURITY_INFORMATION,
            ptr::null_mut(),
            ptr::null_mut(),
            &mut dacl,
            ptr::null_mut(),
            &mut security_descriptor,
        )
    };
    if result != ERROR_SUCCESS {
        return Err(io::Error::other(format!(
            "GetNamedSecurityInfoW failed for {}: {result}",
            path.display()
        )));
    }

    let mut new_dacl = ptr::null_mut();
    let result =
        unsafe { SetEntriesInAclW(entries.len() as u32, entries.as_ptr(), dacl, &mut new_dacl) };
    if result != ERROR_SUCCESS {
        unsafe {
            if !security_descriptor.is_null() {
                LocalFree(security_descriptor as HLOCAL);
            }
        }
        return Err(io::Error::other(format!(
            "SetEntriesInAclW failed for {}: {result}",
            path.display()
        )));
    }

    let result = unsafe {
        SetNamedSecurityInfoW(
            path_w.as_ptr() as *mut u16,
            1, // SE_FILE_OBJECT
            DACL_SECURITY_INFORMATION,
            ptr::null_mut(),
            ptr::null_mut(),
            new_dacl,
            ptr::null_mut(),
        )
    };
    unsafe {
        if !new_dacl.is_null() {
            LocalFree(new_dacl as HLOCAL);
        }
        if !security_descriptor.is_null() {
            LocalFree(security_descriptor as HLOCAL);
        }
    }
    if result != ERROR_SUCCESS {
        return Err(io::Error::other(format!(
            "SetNamedSecurityInfoW failed for {}: {result}",
            path.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_acl_plan_grants_workspace_and_denies_metadata() {
        let root = tempfile::tempdir().expect("workspace");
        std::fs::create_dir(root.path().join(".git")).expect("git metadata");
        let plan = build_acl_plan(
            root.path(),
            RequestedSandboxPolicy::WorkspaceWrite,
            None,
            &HashMap::new(),
        )
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
        let plan = build_acl_plan(
            root.path(),
            RequestedSandboxPolicy::ReadOnly,
            None,
            &HashMap::new(),
        )
        .expect("ACL plan");

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
            &HashMap::new(),
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
    fn workspace_acl_plan_keeps_nested_workspace_and_windows_temp_roots() {
        let root = tempfile::tempdir().expect("fixture root");
        let workspace = root.path().join("workspace");
        std::fs::create_dir(&workspace).expect("workspace directory");
        let environment = HashMap::from([
            (
                "Temp".to_string(),
                root.path().to_string_lossy().into_owned(),
            ),
            (
                "TMP".to_string(),
                root.path().to_string_lossy().into_owned(),
            ),
        ]);

        let plan = build_acl_plan(
            &workspace,
            RequestedSandboxPolicy::WorkspaceWrite,
            None,
            &environment,
        )
        .expect("ACL plan");

        assert_eq!(
            plan.write_grants,
            vec![
                std::fs::canonicalize(root.path()).unwrap(),
                std::fs::canonicalize(&workspace).unwrap(),
            ]
        );
    }
}
