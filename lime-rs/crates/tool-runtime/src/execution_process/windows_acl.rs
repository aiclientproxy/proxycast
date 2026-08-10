use super::*;
use app_server_protocol::protocol::v2::{
    FileSystemAccessMode, FileSystemPath, FileSystemSpecialPath, GrantedPermissionProfile,
};
use std::path::{Path, PathBuf};
use std::process::Stdio;

const PROTECTED_METADATA_NAMES: [&str; 3] = [".git", ".codex", ".agents"];

#[derive(Debug)]
pub(super) struct AclPlan {
    grants: Vec<PathBuf>,
    denies: Vec<PathBuf>,
}

pub(super) fn build_acl_plan(
    cwd: &Path,
    policy: RequestedSandboxPolicy,
    permissions: Option<&GrantedPermissionProfile>,
) -> io::Result<AclPlan> {
    let mut grants = Vec::new();
    let mut denies = Vec::new();
    if policy == RequestedSandboxPolicy::WorkspaceWrite {
        grants.push(canonical_existing(cwd)?);
        for name in PROTECTED_METADATA_NAMES {
            let path = cwd.join(name);
            if path.exists() {
                denies.push(canonical_existing(&path)?);
            }
        }
    }

    if let Some(file_system) = permissions.and_then(|profile| profile.file_system.as_ref()) {
        for path in file_system.write.as_deref().unwrap_or_default() {
            grants.push(canonical_absolute(Path::new(path))?);
        }
        for entry in file_system.entries.as_deref().unwrap_or_default() {
            let path = resolve_permission_path(&entry.path, cwd)?;
            let Some(path) = path else {
                continue;
            };
            match entry.access {
                FileSystemAccessMode::Read => {}
                FileSystemAccessMode::Write => grants.push(path),
                FileSystemAccessMode::Deny => denies.push(path),
            }
        }
    }
    grants.sort();
    grants.dedup();
    denies.sort();
    denies.dedup();
    Ok(AclPlan { grants, denies })
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

#[derive(Debug)]
pub(super) struct AclLease {
    sid: String,
    grants: Vec<PathBuf>,
    denies: Vec<PathBuf>,
}

impl AclLease {
    pub(super) fn acquire(sid: &str, plan: AclPlan) -> io::Result<Self> {
        let mut lease = Self {
            sid: sid.to_string(),
            grants: Vec::new(),
            denies: Vec::new(),
        };
        for path in plan.grants {
            let grant = format!("*{sid}:(OI)(CI)(RX,W,D)");
            lease.grants.push(path.clone());
            if let Err(error) = run_icacls(&path, &["/grant:r", &grant, "/C", "/Q"]) {
                lease.rollback();
                return Err(error);
            }
        }
        for path in plan.denies {
            let deny = format!("*{sid}:(OI)(CI)(W,D,DC)");
            lease.denies.push(path.clone());
            if let Err(error) = run_icacls(&path, &["/deny", &deny, "/C", "/Q"]) {
                lease.rollback();
                return Err(error);
            }
        }
        Ok(lease)
    }

    fn rollback(&mut self) {
        for path in self.denies.iter().rev() {
            let sid = format!("*{}", self.sid);
            let _ = run_icacls(path, &["/remove:d", &sid, "/C", "/Q"]);
        }
        for path in self.grants.iter().rev() {
            let sid = format!("*{}", self.sid);
            let _ = run_icacls(path, &["/remove:g", &sid, "/C", "/Q"]);
        }
        self.denies.clear();
        self.grants.clear();
    }
}

impl Drop for AclLease {
    fn drop(&mut self) {
        self.rollback();
    }
}

fn run_icacls(path: &Path, args: &[&str]) -> io::Result<()> {
    let status = std::process::Command::new("icacls.exe")
        .arg(path)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "icacls failed for {} with status {status}",
            path.display()
        )))
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
            plan.grants,
            vec![std::fs::canonicalize(root.path()).unwrap()]
        );
        assert_eq!(
            plan.denies,
            vec![std::fs::canonicalize(root.path().join(".git")).unwrap()]
        );
    }
}
