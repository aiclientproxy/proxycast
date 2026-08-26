use super::accounts::windows_sandbox_users_group_sid;
use super::to_wide;
use std::ffi::c_void;
use std::io;
use std::path::{Path, PathBuf};
use std::ptr;
use windows_sys::Win32::Foundation::{LocalFree, ERROR_SUCCESS, HLOCAL};
use windows_sys::Win32::Security::Authorization::{
    ConvertStringSidToSidW, GetEffectiveRightsFromAclW, GetNamedSecurityInfoW, SetEntriesInAclW,
    SetNamedSecurityInfoW, EXPLICIT_ACCESS_W, GRANT_ACCESS, SE_FILE_OBJECT, TRUSTEE_IS_SID,
    TRUSTEE_IS_UNKNOWN, TRUSTEE_W,
};
use windows_sys::Win32::Security::{ACL, DACL_SECURITY_INFORMATION};
use windows_sys::Win32::Storage::FileSystem::{FILE_GENERIC_EXECUTE, FILE_GENERIC_READ};

const BUILTIN_USERS_SID: &str = "S-1-5-32-545";
const AUTHENTICATED_USERS_SID: &str = "S-1-5-11";
const EVERYONE_SID: &str = "S-1-1-0";
const PROFILE_ROOT_EXCLUSIONS: &[&str] = &[
    ".ssh",
    ".tsh",
    ".brev",
    ".gnupg",
    ".aws",
    ".azure",
    ".kube",
    ".docker",
    ".config",
    ".npm",
    ".pki",
    ".terraform.d",
];

pub(super) fn ensure_default_read_access() -> io::Result<()> {
    let sandbox_group = LocalSid::parse(&windows_sandbox_users_group_sid()?)?;
    let builtin_users = LocalSid::parse(BUILTIN_USERS_SID)?;
    let authenticated_users = LocalSid::parse(AUTHENTICATED_USERS_SID)?;
    let everyone = LocalSid::parse(EVERYONE_SID)?;
    let readable_principals = [
        sandbox_group.raw(),
        builtin_users.raw(),
        authenticated_users.raw(),
        everyone.raw(),
    ];

    for root in default_read_roots()? {
        with_path_dacl(&root, |dacl| {
            if principals_allow_read_execute(dacl, &readable_principals)? {
                return Ok(());
            }
            grant_read_execute(&root, dacl, sandbox_group.raw())
        })?;
    }
    Ok(())
}

pub(super) fn validate_default_read_access() -> Result<(), String> {
    let sandbox_group_sid = windows_sandbox_users_group_sid()
        .map_err(|error| format!("sandbox group SID validation failed: {error}"))?;
    let sandbox_group = LocalSid::parse(&sandbox_group_sid)
        .map_err(|error| format!("sandbox group SID validation failed: {error}"))?;
    let builtin_users = LocalSid::parse(BUILTIN_USERS_SID)
        .map_err(|error| format!("built-in Users SID validation failed: {error}"))?;
    let authenticated_users = LocalSid::parse(AUTHENTICATED_USERS_SID)
        .map_err(|error| format!("Authenticated Users SID validation failed: {error}"))?;
    let everyone = LocalSid::parse(EVERYONE_SID)
        .map_err(|error| format!("Everyone SID validation failed: {error}"))?;
    let principals = [
        sandbox_group.raw(),
        builtin_users.raw(),
        authenticated_users.raw(),
        everyone.raw(),
    ];

    for root in default_read_roots().map_err(|error| error.to_string())? {
        let allowed = with_path_dacl(&root, |dacl| {
            principals_allow_read_execute(dacl, &principals)
        })
        .map_err(|error| {
            format!(
                "default read root {} validation failed: {error}",
                root.display()
            )
        })?;
        if !allowed {
            return Err(format!(
                "Windows sandbox cannot read and execute default root {}",
                root.display()
            ));
        }
    }
    Ok(())
}

fn default_read_roots() -> io::Result<Vec<PathBuf>> {
    let mut roots = Vec::new();
    for (name, fallback) in [
        ("SystemRoot", Some(r"C:\Windows")),
        ("ProgramFiles", Some(r"C:\Program Files")),
        ("ProgramFiles(x86)", Some(r"C:\Program Files (x86)")),
        ("ProgramData", Some(r"C:\ProgramData")),
    ] {
        let path = std::env::var_os(name)
            .map(PathBuf::from)
            .or_else(|| fallback.map(PathBuf::from));
        if let Some(path) = path.filter(|path| path.exists()) {
            roots.push(path);
        }
    }

    if let Some(profile) = std::env::var_os("USERPROFILE").map(PathBuf::from) {
        let entries = std::fs::read_dir(&profile).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!(
                    "failed to enumerate Windows user profile {}: {error}",
                    profile.display()
                ),
            )
        })?;
        for entry in entries {
            let entry = entry.map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!(
                        "failed to enumerate Windows user profile {}: {error}",
                        profile.display()
                    ),
                )
            })?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if !PROFILE_ROOT_EXCLUSIONS
                .iter()
                .any(|excluded| name.eq_ignore_ascii_case(excluded))
            {
                roots.push(entry.path());
            }
        }
    }

    roots.sort_by(|left, right| {
        left.to_string_lossy()
            .to_ascii_lowercase()
            .cmp(&right.to_string_lossy().to_ascii_lowercase())
    });
    roots.dedup_by(|left, right| {
        left.to_string_lossy()
            .eq_ignore_ascii_case(right.to_string_lossy().as_ref())
    });
    Ok(roots)
}

fn with_path_dacl<T>(
    path: &Path,
    operation: impl FnOnce(*mut ACL) -> io::Result<T>,
) -> io::Result<T> {
    let path_wide = to_wide(path.as_os_str());
    let mut descriptor = ptr::null_mut();
    let mut dacl = ptr::null_mut();
    let result = unsafe {
        GetNamedSecurityInfoW(
            path_wide.as_ptr(),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION,
            ptr::null_mut(),
            ptr::null_mut(),
            &mut dacl,
            ptr::null_mut(),
            &mut descriptor,
        )
    };
    if result != ERROR_SUCCESS {
        return Err(win32_result_error(
            &format!("GetNamedSecurityInfoW({})", path.display()),
            result,
        ));
    }
    if dacl.is_null() {
        unsafe {
            LocalFree(descriptor as HLOCAL);
        }
        return Err(io::Error::other(format!(
            "Windows default read root {} has no DACL",
            path.display()
        )));
    }
    let operation_result = operation(dacl);
    unsafe {
        if !descriptor.is_null() {
            LocalFree(descriptor as HLOCAL);
        }
    }
    operation_result
}

fn principals_allow_read_execute(dacl: *mut ACL, principals: &[*mut c_void]) -> io::Result<bool> {
    let required = FILE_GENERIC_READ | FILE_GENERIC_EXECUTE;
    for principal in principals {
        let trustee = trustee(*principal);
        let mut effective = 0;
        let result = unsafe { GetEffectiveRightsFromAclW(dacl, &trustee, &mut effective) };
        if result != ERROR_SUCCESS {
            return Err(win32_result_error(
                "GetEffectiveRightsFromAclW(default read root)",
                result,
            ));
        }
        if effective & required == required {
            return Ok(true);
        }
    }
    Ok(false)
}

fn grant_read_execute(path: &Path, dacl: *mut ACL, sandbox_group: *mut c_void) -> io::Result<()> {
    let entry = EXPLICIT_ACCESS_W {
        grfAccessPermissions: FILE_GENERIC_READ | FILE_GENERIC_EXECUTE,
        grfAccessMode: GRANT_ACCESS,
        grfInheritance: 0x3,
        Trustee: trustee(sandbox_group),
    };
    let mut updated_dacl = ptr::null_mut();
    let result = unsafe { SetEntriesInAclW(1, &entry, dacl, &mut updated_dacl) };
    if result != ERROR_SUCCESS {
        return Err(win32_result_error(
            "SetEntriesInAclW(default read root)",
            result,
        ));
    }
    if updated_dacl.is_null() {
        return Err(io::Error::other(
            "SetEntriesInAclW(default read root) returned a null DACL",
        ));
    }
    let path_wide = to_wide(path.as_os_str());
    let applied = unsafe {
        SetNamedSecurityInfoW(
            path_wide.as_ptr() as *mut u16,
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION,
            ptr::null_mut(),
            ptr::null_mut(),
            updated_dacl,
            ptr::null_mut(),
        )
    };
    unsafe {
        LocalFree(updated_dacl as HLOCAL);
    }
    if applied == ERROR_SUCCESS {
        Ok(())
    } else {
        Err(win32_result_error(
            &format!("SetNamedSecurityInfoW({})", path.display()),
            applied,
        ))
    }
}

fn trustee(sid: *mut c_void) -> TRUSTEE_W {
    TRUSTEE_W {
        pMultipleTrustee: ptr::null_mut(),
        MultipleTrusteeOperation: 0,
        TrusteeForm: TRUSTEE_IS_SID,
        TrusteeType: TRUSTEE_IS_UNKNOWN,
        ptstrName: sid as *mut u16,
    }
}

struct LocalSid(*mut c_void);

impl LocalSid {
    fn parse(value: &str) -> io::Result<Self> {
        let mut sid = ptr::null_mut();
        let value = to_wide(value);
        if unsafe { ConvertStringSidToSidW(value.as_ptr(), &mut sid) } == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self(sid))
    }

    fn raw(&self) -> *mut c_void {
        self.0
    }
}

impl Drop for LocalSid {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                LocalFree(self.0 as HLOCAL);
            }
            self.0 = ptr::null_mut();
        }
    }
}

fn win32_result_error(context: &str, code: u32) -> io::Error {
    io::Error::other(format!(
        "{context} failed: {}",
        io::Error::from_raw_os_error(code as i32)
    ))
}
