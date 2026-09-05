use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
#[cfg(windows)]
use uuid::Uuid;

#[cfg(windows)]
#[path = "windows_setup/accounts.rs"]
mod accounts;
#[cfg(windows)]
#[path = "windows_setup/read_access.rs"]
mod read_access;

#[cfg(windows)]
use accounts::{
    ensure_builtin_users_group_member, ensure_local_account, ensure_local_group,
    ensure_local_group_member, ensure_sandbox_accounts_null_device_access,
    validate_sandbox_accounts_null_device_access, validate_windows_sandbox_group_membership,
};
#[cfg(windows)]
pub(crate) use accounts::{
    resolve_windows_account_sid, verify_windows_sandbox_group_membership,
    windows_sandbox_users_group_sid,
};
#[cfg(windows)]
use read_access::{ensure_default_read_access, validate_default_read_access};

pub const WINDOWS_SANDBOX_SETUP_VERSION: u32 = 2;
pub const WINDOWS_SANDBOX_OFFLINE_USERNAME: &str = "LimeSandboxOffline";
pub const WINDOWS_SANDBOX_ONLINE_USERNAME: &str = "LimeSandboxOnline";
pub const WINDOWS_SANDBOX_USERS_GROUP: &str = "LimeSandboxUsers";

const SANDBOX_DIR_NAME: &str = ".sandbox";
const SANDBOX_SECRETS_DIR_NAME: &str = ".sandbox-secrets";
const SETUP_MARKER_FILE_NAME: &str = "setup_marker.json";
const SANDBOX_USERS_FILE_NAME: &str = "sandbox_users.json";
const MAX_PROTECTED_PASSWORD_BYTES: usize = 16 * 1024;
#[cfg(windows)]
const WINDOWS_SANDBOX_USERS_GROUP_COMMENT: &str = "Lime sandbox internal group (managed)";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct WindowsSandboxSetupMarker {
    pub version: u32,
    pub offline_username: String,
    pub online_username: String,
    #[serde(default)]
    pub created_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct WindowsSandboxUserRecord {
    pub username: String,
    /// DPAPI-encrypted password blob, encoded as base64.
    pub password: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct WindowsSandboxUsersFile {
    pub version: u32,
    pub offline: WindowsSandboxUserRecord,
    pub online: WindowsSandboxUserRecord,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowsSandboxSetupArtifactStatus {
    Valid,
    Missing,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowsSandboxSetupArtifactInspection {
    pub status: WindowsSandboxSetupArtifactStatus,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowsSandboxSetupResult {
    pub marker_path: PathBuf,
    pub users_path: PathBuf,
    pub installed_firewall_rule_count: usize,
    pub installed_wfp_filter_count: usize,
}

/// Creates the local accounts and commits the setup artifacts.
///
/// The implementation is deliberately target-gated: a non-Windows build can
/// inspect fixture-shaped artifacts but can never claim that setup was run.
#[cfg(not(windows))]
pub fn run_windows_sandbox_setup(
    _agent_root: &Path,
    _owner: &str,
) -> io::Result<WindowsSandboxSetupResult> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "Windows sandbox setup is only available on Windows",
    ))
}

#[cfg(windows)]
pub fn run_windows_sandbox_setup(
    agent_root: &Path,
    owner: &str,
) -> io::Result<WindowsSandboxSetupResult> {
    if !agent_root.is_absolute() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Windows sandbox setup root must be absolute",
        ));
    }
    if owner.trim().is_empty() || owner.contains('\0') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Windows sandbox setup owner must be non-empty",
        ));
    }
    reject_reparse_ancestors(agent_root)?;
    let marker_path = windows_sandbox_setup_marker_path(agent_root);
    let users_path = windows_sandbox_users_path(agent_root);
    if let Some(parent) = marker_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = users_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Machine-scope DPAPI blobs are only confidential while their containing
    // directory is private. Lock both directories before changing credentials,
    // then invalidate the old marker until every setup step succeeds.
    protect_setup_directory(marker_path.parent().expect("marker parent"), owner)?;
    protect_setup_directory(users_path.parent().expect("users parent"), owner)?;
    remove_file_if_exists(&marker_path)?;

    let offline_password = generate_setup_password();
    let online_password = generate_setup_password();
    ensure_local_group(
        WINDOWS_SANDBOX_USERS_GROUP,
        WINDOWS_SANDBOX_USERS_GROUP_COMMENT,
    )?;
    ensure_local_account(WINDOWS_SANDBOX_OFFLINE_USERNAME, &offline_password)?;
    ensure_local_account(WINDOWS_SANDBOX_ONLINE_USERNAME, &online_password)?;
    ensure_local_group_member(
        WINDOWS_SANDBOX_USERS_GROUP,
        WINDOWS_SANDBOX_OFFLINE_USERNAME,
    )?;
    ensure_local_group_member(WINDOWS_SANDBOX_USERS_GROUP, WINDOWS_SANDBOX_ONLINE_USERNAME)?;
    ensure_builtin_users_group_member(WINDOWS_SANDBOX_OFFLINE_USERNAME)?;
    ensure_builtin_users_group_member(WINDOWS_SANDBOX_ONLINE_USERNAME)?;
    ensure_default_read_access()?;
    ensure_sandbox_accounts_null_device_access()?;

    let offline_blob = dpapi_protect(offline_password.as_bytes())?;
    let online_blob = dpapi_protect(online_password.as_bytes())?;
    let users = WindowsSandboxUsersFile {
        version: WINDOWS_SANDBOX_SETUP_VERSION,
        offline: WindowsSandboxUserRecord {
            username: WINDOWS_SANDBOX_OFFLINE_USERNAME.to_string(),
            password: BASE64_STANDARD.encode(offline_blob),
        },
        online: WindowsSandboxUserRecord {
            username: WINDOWS_SANDBOX_ONLINE_USERNAME.to_string(),
            password: BASE64_STANDARD.encode(online_blob),
        },
    };
    write_json_atomic(&users_path, &users)?;

    let installed_firewall_rule_count =
        crate::windows_firewall::install_offline_rules(WINDOWS_SANDBOX_OFFLINE_USERNAME)?;
    let installed_wfp_filter_count =
        crate::windows_wfp::install_filters(WINDOWS_SANDBOX_OFFLINE_USERNAME)?;

    let marker = WindowsSandboxSetupMarker {
        version: WINDOWS_SANDBOX_SETUP_VERSION,
        offline_username: WINDOWS_SANDBOX_OFFLINE_USERNAME.to_string(),
        online_username: WINDOWS_SANDBOX_ONLINE_USERNAME.to_string(),
        created_at: Some(chrono::Utc::now().to_rfc3339()),
    };
    write_json_atomic(&marker_path, &marker)?;

    Ok(WindowsSandboxSetupResult {
        marker_path,
        users_path,
        installed_firewall_rule_count,
        installed_wfp_filter_count,
    })
}

#[cfg(windows)]
fn remove_file_if_exists(path: &Path) -> io::Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

#[cfg(windows)]
fn reject_reparse_ancestors(path: &Path) -> io::Result<()> {
    use std::os::windows::fs::MetadataExt;
    use windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT;

    for ancestor in path.ancestors() {
        if ancestor.as_os_str().is_empty() {
            continue;
        }
        match fs::symlink_metadata(ancestor) {
            Ok(metadata)
                if metadata.file_type().is_symlink()
                    || metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 =>
            {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    format!(
                        "Windows sandbox setup rejects reparse-point path component: {}",
                        ancestor.display()
                    ),
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

#[cfg(windows)]
fn generate_setup_password() -> String {
    format!("LimeSandbox-{}", Uuid::new_v4().simple())
}

#[cfg(windows)]
fn protect_setup_directory(path: &Path, owner: &str) -> io::Result<()> {
    use std::process::{Command, Stdio};
    use std::thread;
    use std::time::{Duration, Instant};

    let system_root = std::env::var_os("SystemRoot").ok_or_else(|| {
        io::Error::new(io::ErrorKind::NotFound, "SystemRoot is missing for icacls")
    })?;
    let icacls = PathBuf::from(system_root)
        .join("System32")
        .join("icacls.exe");
    if !icacls.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("icacls is missing: {}", icacls.display()),
        ));
    }
    let owner_grant = format!("{owner}:(OI)(CI)(F)");
    let system_grant = "*S-1-5-18:(OI)(CI)(F)";
    let administrators_grant = "*S-1-5-32-544:(OI)(CI)(F)";
    let mut child = Command::new(icacls)
        .arg(path)
        .args([
            "/inheritance:r",
            "/grant:r",
            owner_grant.as_str(),
            system_grant,
            administrators_grant,
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            if status.success() {
                return Ok(());
            }
            return Err(io::Error::other(format!(
                "icacls failed for {} with {status}",
                path.display()
            )));
        }
        if started.elapsed() >= Duration::from_secs(15) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("icacls timed out for {}", path.display()),
            ));
        }
        thread::sleep(Duration::from_millis(25));
    }
}

#[cfg(windows)]
fn dpapi_protect(plaintext: &[u8]) -> io::Result<Vec<u8>> {
    use windows_sys::Win32::Foundation::{GetLastError, LocalFree, HLOCAL};
    use windows_sys::Win32::Security::Cryptography::{
        CryptProtectData, CRYPTPROTECT_LOCAL_MACHINE, CRYPTPROTECT_UI_FORBIDDEN, CRYPT_INTEGER_BLOB,
    };

    let mut input = CRYPT_INTEGER_BLOB {
        cbData: plaintext.len() as u32,
        pbData: plaintext.as_ptr() as *mut u8,
    };
    let mut output = CRYPT_INTEGER_BLOB {
        cbData: 0,
        pbData: std::ptr::null_mut(),
    };
    let ok = unsafe {
        CryptProtectData(
            &mut input,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            CRYPTPROTECT_UI_FORBIDDEN | CRYPTPROTECT_LOCAL_MACHINE,
            &mut output,
        )
    };
    if ok == 0 {
        return Err(io::Error::from_raw_os_error(
            unsafe { GetLastError() } as i32
        ));
    }
    if output.pbData.is_null() || output.cbData == 0 {
        return Err(io::Error::other("CryptProtectData returned an empty blob"));
    }
    let protected = unsafe { std::slice::from_raw_parts(output.pbData, output.cbData as usize) };
    let result = protected.to_vec();
    unsafe {
        LocalFree(output.pbData as HLOCAL);
    }
    Ok(result)
}

#[cfg(windows)]
fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> io::Result<()> {
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    let temporary = path.with_extension(format!("tmp-{}", Uuid::new_v4().simple()));
    let encoded = serde_json::to_vec(value).map_err(io::Error::other)?;
    fs::write(&temporary, encoded)?;
    let source = to_wide(temporary.as_os_str());
    let destination = to_wide(path.as_os_str());
    let moved = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if moved == 0 {
        let error = io::Error::last_os_error();
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    Ok(())
}

#[cfg(windows)]
fn to_wide(value: impl AsRef<std::ffi::OsStr>) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    value
        .as_ref()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect()
}

impl WindowsSandboxSetupArtifactInspection {
    pub fn is_valid(&self) -> bool {
        self.status == WindowsSandboxSetupArtifactStatus::Valid
    }
}

pub fn windows_sandbox_setup_marker_path(agent_root: &Path) -> PathBuf {
    agent_root
        .join(SANDBOX_DIR_NAME)
        .join(SETUP_MARKER_FILE_NAME)
}

pub fn windows_sandbox_users_path(agent_root: &Path) -> PathBuf {
    agent_root
        .join(SANDBOX_SECRETS_DIR_NAME)
        .join(SANDBOX_USERS_FILE_NAME)
}

pub fn inspect_default_windows_sandbox_setup() -> WindowsSandboxSetupArtifactInspection {
    match lime_core::app_paths::preferred_agent_root() {
        Ok(agent_root) => match load_validated_setup_artifacts(&agent_root) {
            Ok(artifacts) => {
                #[cfg(windows)]
                if let Err(reason) = validate_windows_runtime_artifacts(&artifacts) {
                    return invalid_inspection(reason);
                }
                #[cfg(not(windows))]
                let _ = artifacts;
                valid_inspection()
            }
            Err(error) => inspection_from_error(error),
        },
        Err(error) => invalid_inspection(format!(
            "failed to resolve the Windows sandbox data root: {error}"
        )),
    }
}

pub fn inspect_windows_sandbox_setup_at(
    agent_root: &Path,
) -> WindowsSandboxSetupArtifactInspection {
    match load_validated_setup_artifacts(agent_root) {
        Ok(_) => valid_inspection(),
        Err(error) => inspection_from_error(error),
    }
}

/// Verifies the complete setup required by the restricted-token runner.
///
/// Unlike `inspect_windows_sandbox_setup_at`, this probe performs the native
/// Windows checks (DPAPI, account logon, group/read/NUL ACLs and network
/// policy). It is intentionally unavailable off Windows so an unelevated
/// setup request can never claim readiness on another platform.
#[cfg(not(windows))]
pub fn verify_windows_sandbox_setup_at(_agent_root: &Path) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "Windows sandbox setup verification is only available on Windows",
    ))
}

#[cfg(windows)]
pub fn verify_windows_sandbox_setup_at(agent_root: &Path) -> io::Result<()> {
    let artifacts = load_validated_setup_artifacts(agent_root).map_err(|error| {
        let inspection = inspection_from_error(error);
        let kind = match inspection.status {
            WindowsSandboxSetupArtifactStatus::Missing => io::ErrorKind::NotFound,
            WindowsSandboxSetupArtifactStatus::Invalid => io::ErrorKind::PermissionDenied,
            WindowsSandboxSetupArtifactStatus::Valid => io::ErrorKind::Other,
        };
        io::Error::new(kind, inspection.reason)
    })?;
    validate_windows_runtime_artifacts(&artifacts)
        .map_err(|reason| io::Error::new(io::ErrorKind::PermissionDenied, reason))
}

struct ValidatedSetupArtifacts {
    #[cfg(windows)]
    users: WindowsSandboxUsersFile,
}

fn load_validated_setup_artifacts(
    agent_root: &Path,
) -> Result<ValidatedSetupArtifacts, ArtifactReadError> {
    let marker = read_json::<WindowsSandboxSetupMarker>(
        &windows_sandbox_setup_marker_path(agent_root),
        "setup marker",
    )?;
    let users = read_json::<WindowsSandboxUsersFile>(
        &windows_sandbox_users_path(agent_root),
        "sandbox users file",
    )?;

    if marker.version != WINDOWS_SANDBOX_SETUP_VERSION
        || users.version != WINDOWS_SANDBOX_SETUP_VERSION
    {
        return Err(ArtifactReadError::Invalid(format!(
            "Windows sandbox setup version mismatch: expected {}, marker={}, users={}",
            WINDOWS_SANDBOX_SETUP_VERSION, marker.version, users.version
        )));
    }
    if marker.offline_username != WINDOWS_SANDBOX_OFFLINE_USERNAME
        || marker.online_username != WINDOWS_SANDBOX_ONLINE_USERNAME
    {
        return Err(ArtifactReadError::Invalid(
            "Windows sandbox setup marker account names do not match".to_string(),
        ));
    }
    if users.offline.username != marker.offline_username
        || users.online.username != marker.online_username
    {
        return Err(ArtifactReadError::Invalid(
            "Windows sandbox user records do not match the setup marker".to_string(),
        ));
    }
    validate_protected_password("offline", &users.offline.password)
        .map_err(ArtifactReadError::Invalid)?;
    validate_protected_password("online", &users.online.password)
        .map_err(ArtifactReadError::Invalid)?;

    Ok(ValidatedSetupArtifacts {
        #[cfg(windows)]
        users,
    })
}

enum ArtifactReadError {
    Missing(String),
    Invalid(String),
}

fn read_json<T>(path: &Path, label: &str) -> Result<T, ArtifactReadError>
where
    T: for<'de> Deserialize<'de>,
{
    let contents = fs::read(path).map_err(|error| match error.kind() {
        io::ErrorKind::NotFound => ArtifactReadError::Missing(format!("{label} is missing")),
        _ => ArtifactReadError::Invalid(format!("failed to read {label}: {error}")),
    })?;
    serde_json::from_slice(&contents)
        .map_err(|error| ArtifactReadError::Invalid(format!("failed to parse {label}: {error}")))
}

fn validate_protected_password(label: &str, value: &str) -> Result<(), String> {
    let blob = BASE64_STANDARD
        .decode(value)
        .map_err(|_| format!("{label} sandbox password is not valid base64"))?;
    if blob.is_empty() || blob.len() > MAX_PROTECTED_PASSWORD_BYTES {
        return Err(format!(
            "{label} sandbox password blob length is outside the accepted range"
        ));
    }
    Ok(())
}

#[cfg(windows)]
fn validate_windows_runtime_artifacts(artifacts: &ValidatedSetupArtifacts) -> Result<(), String> {
    validate_windows_sandbox_group_membership(WINDOWS_SANDBOX_OFFLINE_USERNAME)?;
    validate_windows_sandbox_group_membership(WINDOWS_SANDBOX_ONLINE_USERNAME)?;
    validate_default_read_access()?;
    validate_sandbox_accounts_null_device_access()?;
    validate_windows_sandbox_user("offline", &artifacts.users.offline)?;
    validate_windows_sandbox_user("online", &artifacts.users.online)?;
    crate::windows_firewall::verify_offline_rules(WINDOWS_SANDBOX_OFFLINE_USERNAME)
        .map_err(|error| format!("offline Windows Firewall validation failed: {error}"))?;
    crate::windows_wfp::verify_filters(WINDOWS_SANDBOX_OFFLINE_USERNAME)
        .map_err(|error| format!("offline Windows network isolation validation failed: {error}"))
}

#[cfg(windows)]
pub fn read_windows_sandbox_password(account: &str) -> io::Result<String> {
    let agent_root = lime_core::app_paths::preferred_agent_root().map_err(io::Error::other)?;
    let artifacts = load_validated_setup_artifacts(&agent_root).map_err(|error| {
        let inspection = inspection_from_error(error);
        io::Error::other(format!(
            "Windows sandbox setup artifacts are not valid: {}",
            inspection.reason
        ))
    })?;
    let record = match account {
        WINDOWS_SANDBOX_OFFLINE_USERNAME => &artifacts.users.offline,
        WINDOWS_SANDBOX_ONLINE_USERNAME => &artifacts.users.online,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "unknown Windows sandbox account",
            ))
        }
    };
    let protected = BASE64_STANDARD
        .decode(&record.password)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid DPAPI blob encoding"))?;
    let plaintext = dpapi_unprotect(&protected).map_err(io::Error::other)?;
    let password = std::str::from_utf8(&plaintext)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "DPAPI password is not UTF-8"))?;
    if password.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "DPAPI password is empty",
        ));
    }
    Ok(password.to_string())
}

#[cfg(windows)]
fn validate_windows_sandbox_user(
    label: &str,
    record: &WindowsSandboxUserRecord,
) -> Result<(), String> {
    let protected_password = BASE64_STANDARD
        .decode(&record.password)
        .map_err(|_| format!("{label} sandbox password is not valid base64"))?;
    let password = dpapi_unprotect(&protected_password)
        .map_err(|error| format!("{label} sandbox password DPAPI validation failed: {error}"))?;
    let password = std::str::from_utf8(&password)
        .map_err(|_| format!("{label} sandbox password is not UTF-8"))?;
    if password.is_empty() {
        return Err(format!("{label} sandbox password is empty"));
    }
    resolve_windows_account_sid(&record.username)
        .map_err(|error| format!("{label} sandbox account validation failed: {error}"))?;
    validate_windows_account_logon(&record.username, password)
        .map_err(|error| format!("{label} sandbox credential validation failed: {error}"))
}

#[cfg(windows)]
fn validate_windows_account_logon(account: &str, password: &str) -> io::Result<()> {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::Security::{
        LogonUserW, LOGON32_LOGON_INTERACTIVE, LOGON32_PROVIDER_DEFAULT,
    };

    let account = to_wide(account);
    let domain = to_wide(".");
    let password = to_wide(password);
    let mut token = 0;
    let ok = unsafe {
        LogonUserW(
            account.as_ptr(),
            domain.as_ptr(),
            password.as_ptr(),
            LOGON32_LOGON_INTERACTIVE,
            LOGON32_PROVIDER_DEFAULT,
            &mut token,
        )
    };
    if ok == 0 {
        return Err(io::Error::last_os_error());
    }
    unsafe {
        CloseHandle(token);
    }
    Ok(())
}

#[cfg(windows)]
fn dpapi_unprotect(blob: &[u8]) -> Result<Vec<u8>, String> {
    use windows_sys::Win32::Foundation::{GetLastError, LocalFree, HLOCAL};
    use windows_sys::Win32::Security::Cryptography::{
        CryptUnprotectData, CRYPTPROTECT_LOCAL_MACHINE, CRYPTPROTECT_UI_FORBIDDEN,
        CRYPT_INTEGER_BLOB,
    };

    let mut input = CRYPT_INTEGER_BLOB {
        cbData: blob.len() as u32,
        pbData: blob.as_ptr() as *mut u8,
    };
    let mut output = CRYPT_INTEGER_BLOB {
        cbData: 0,
        pbData: std::ptr::null_mut(),
    };
    let ok = unsafe {
        CryptUnprotectData(
            &mut input,
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            CRYPTPROTECT_UI_FORBIDDEN | CRYPTPROTECT_LOCAL_MACHINE,
            &mut output,
        )
    };
    if ok == 0 {
        return Err(format!("CryptUnprotectData failed: {}", unsafe {
            GetLastError()
        }));
    }
    let plaintext = if output.pbData.is_null()
        || output.cbData == 0
        || output.cbData as usize > MAX_PROTECTED_PASSWORD_BYTES
    {
        if !output.pbData.is_null() {
            unsafe {
                LocalFree(output.pbData as HLOCAL);
            }
        }
        return Err("CryptUnprotectData returned an invalid plaintext blob".to_string());
    } else {
        unsafe { std::slice::from_raw_parts(output.pbData, output.cbData as usize).to_vec() }
    };
    if !output.pbData.is_null() {
        unsafe {
            LocalFree(output.pbData as HLOCAL);
        }
    }
    Ok(plaintext)
}

fn inspection_from_error(error: ArtifactReadError) -> WindowsSandboxSetupArtifactInspection {
    match error {
        ArtifactReadError::Missing(reason) => missing_inspection(reason),
        ArtifactReadError::Invalid(reason) => invalid_inspection(reason),
    }
}

fn valid_inspection() -> WindowsSandboxSetupArtifactInspection {
    WindowsSandboxSetupArtifactInspection {
        status: WindowsSandboxSetupArtifactStatus::Valid,
        reason: "Windows sandbox setup artifacts are structurally valid".to_string(),
    }
}

fn missing_inspection(reason: impl Into<String>) -> WindowsSandboxSetupArtifactInspection {
    WindowsSandboxSetupArtifactInspection {
        status: WindowsSandboxSetupArtifactStatus::Missing,
        reason: reason.into(),
    }
}

fn invalid_inspection(reason: impl Into<String>) -> WindowsSandboxSetupArtifactInspection {
    WindowsSandboxSetupArtifactInspection {
        status: WindowsSandboxSetupArtifactStatus::Invalid,
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(windows))]
    #[test]
    fn setup_helper_is_fail_closed_off_windows() {
        let root = tempfile::tempdir().expect("tempdir");
        let error = run_windows_sandbox_setup(root.path(), "owner")
            .expect_err("non-Windows setup must not run");
        assert_eq!(error.kind(), io::ErrorKind::Unsupported);
    }

    #[cfg(not(windows))]
    #[test]
    fn unelevated_preflight_is_fail_closed_off_windows() {
        let root = tempfile::tempdir().expect("tempdir");
        let error = verify_windows_sandbox_setup_at(root.path())
            .expect_err("non-Windows setup verification must not claim readiness");
        assert_eq!(error.kind(), io::ErrorKind::Unsupported);
    }

    fn write_valid_artifacts(root: &Path) {
        let marker_path = windows_sandbox_setup_marker_path(root);
        let users_path = windows_sandbox_users_path(root);
        fs::create_dir_all(marker_path.parent().expect("marker parent"))
            .expect("create marker parent");
        fs::create_dir_all(users_path.parent().expect("users parent"))
            .expect("create users parent");
        fs::write(
            marker_path,
            serde_json::to_vec(&WindowsSandboxSetupMarker {
                version: WINDOWS_SANDBOX_SETUP_VERSION,
                offline_username: WINDOWS_SANDBOX_OFFLINE_USERNAME.to_string(),
                online_username: WINDOWS_SANDBOX_ONLINE_USERNAME.to_string(),
                created_at: Some("2026-08-26T00:00:00Z".to_string()),
            })
            .expect("serialize marker"),
        )
        .expect("write marker");
        fs::write(
            users_path,
            serde_json::to_vec(&WindowsSandboxUsersFile {
                version: WINDOWS_SANDBOX_SETUP_VERSION,
                offline: WindowsSandboxUserRecord {
                    username: WINDOWS_SANDBOX_OFFLINE_USERNAME.to_string(),
                    password: BASE64_STANDARD.encode([1, 2, 3]),
                },
                online: WindowsSandboxUserRecord {
                    username: WINDOWS_SANDBOX_ONLINE_USERNAME.to_string(),
                    password: BASE64_STANDARD.encode([4, 5, 6]),
                },
            })
            .expect("serialize users"),
        )
        .expect("write users");
    }

    #[test]
    fn missing_marker_is_not_ready() {
        let root = tempfile::tempdir().expect("tempdir");
        let inspection = inspect_windows_sandbox_setup_at(root.path());

        assert_eq!(
            inspection.status,
            WindowsSandboxSetupArtifactStatus::Missing
        );
        assert!(inspection.reason.contains("setup marker is missing"));
    }

    #[test]
    fn structurally_valid_artifacts_are_accepted() {
        let root = tempfile::tempdir().expect("tempdir");
        write_valid_artifacts(root.path());

        let inspection = inspect_windows_sandbox_setup_at(root.path());
        assert!(inspection.is_valid(), "{}", inspection.reason);
    }

    #[test]
    fn version_drift_fails_closed() {
        let root = tempfile::tempdir().expect("tempdir");
        write_valid_artifacts(root.path());
        let marker_path = windows_sandbox_setup_marker_path(root.path());
        fs::write(
            marker_path,
            r#"{"version":3,"offline_username":"Other","online_username":"LimeSandboxOnline"}"#,
        )
        .expect("rewrite marker");

        let inspection = inspect_windows_sandbox_setup_at(root.path());
        assert_eq!(
            inspection.status,
            WindowsSandboxSetupArtifactStatus::Invalid
        );
        assert!(inspection.reason.contains("version mismatch"));
    }

    #[test]
    fn account_drift_fails_closed() {
        let root = tempfile::tempdir().expect("tempdir");
        write_valid_artifacts(root.path());
        let marker_path = windows_sandbox_setup_marker_path(root.path());
        fs::write(
            marker_path,
            r#"{"version":2,"offline_username":"Other","online_username":"LimeSandboxOnline"}"#,
        )
        .expect("rewrite marker");

        let inspection = inspect_windows_sandbox_setup_at(root.path());
        assert_eq!(
            inspection.status,
            WindowsSandboxSetupArtifactStatus::Invalid
        );
        assert!(inspection.reason.contains("account names"));
    }

    #[test]
    fn unknown_artifact_fields_fail_closed() {
        let root = tempfile::tempdir().expect("tempdir");
        write_valid_artifacts(root.path());
        let marker_path = windows_sandbox_setup_marker_path(root.path());
        fs::write(
            marker_path,
            r#"{"version":2,"offline_username":"LimeSandboxOffline","online_username":"LimeSandboxOnline","unexpected":true}"#,
        )
        .expect("rewrite marker");

        let inspection = inspect_windows_sandbox_setup_at(root.path());
        assert_eq!(
            inspection.status,
            WindowsSandboxSetupArtifactStatus::Invalid
        );
        assert!(inspection.reason.contains("unknown field"));
    }

    #[test]
    fn malformed_or_empty_protected_password_fails_closed() {
        let root = tempfile::tempdir().expect("tempdir");
        write_valid_artifacts(root.path());
        let users_path = windows_sandbox_users_path(root.path());
        let mut users: serde_json::Value =
            serde_json::from_slice(&fs::read(&users_path).expect("read users"))
                .expect("parse users");
        users["offline"]["password"] = serde_json::Value::String(String::new());
        fs::write(
            users_path,
            serde_json::to_vec(&users).expect("serialize users"),
        )
        .expect("rewrite users");

        let inspection = inspect_windows_sandbox_setup_at(root.path());
        assert_eq!(
            inspection.status,
            WindowsSandboxSetupArtifactStatus::Invalid
        );
        assert!(inspection.reason.contains("blob length"));
    }
}
