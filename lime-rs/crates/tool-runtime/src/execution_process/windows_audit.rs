use super::{to_wide, world_sid};
use crate::execution_process::WindowsWorldWritableAudit;
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::io;
use std::path::Path;
use std::ptr;
use std::time::{Duration, Instant};
use windows_sys::Win32::Foundation::{LocalFree, ERROR_SUCCESS, HLOCAL};
use windows_sys::Win32::Security::Authorization::GetNamedSecurityInfoW;
use windows_sys::Win32::Security::{
    AclSizeInformation, EqualSid, GetAce, GetAclInformation, MapGenericMask, ACCESS_ALLOWED_ACE,
    ACE_HEADER, ACL, ACL_SIZE_INFORMATION, DACL_SECURITY_INFORMATION, GENERIC_MAPPING,
};
use windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT;

const ACCESS_ALLOWED_ACE_TYPE: u8 = 0;
const INHERIT_ONLY_ACE: u8 = 0x08;
const SE_FILE_OBJECT: i32 = 1;

const AUDIT_MAX_ITEMS_PER_DIRECTORY: usize = 1_000;
const AUDIT_MAX_CHECKED_ITEMS: usize = 50_000;
const AUDIT_MAX_DURATION: Duration = Duration::from_secs(2);
const AUDIT_SAMPLE_LIMIT: usize = 5;
const AUDIT_SKIP_SUFFIXES: [&str; 3] = [
    "/windows/installer",
    "/windows/registration",
    "/programdata",
];

/// Perform the bounded Everyone-write audit used by the Windows setup warning.
///
/// This deliberately only observes ACLs. The restricted-token ACL lease remains
/// the execution owner and is responsible for applying and rolling back grants.
pub(crate) fn audit_world_writable(
    cwd: &Path,
    environment: &HashMap<String, String>,
) -> WindowsWorldWritableAudit {
    let started = Instant::now();
    let mut candidates = Vec::new();
    let mut seen_candidates = HashSet::new();
    push_audit_candidate(cwd, &mut candidates, &mut seen_candidates);
    for key in ["TEMP", "TMP"] {
        if let Some(value) = environment
            .get(key)
            .cloned()
            .or_else(|| std::env::var(key).ok())
        {
            push_audit_candidate(Path::new(&value), &mut candidates, &mut seen_candidates);
        }
    }
    for key in ["USERPROFILE", "PUBLIC"] {
        if let Some(value) = environment
            .get(key)
            .cloned()
            .or_else(|| std::env::var(key).ok())
        {
            push_audit_candidate(Path::new(&value), &mut candidates, &mut seen_candidates);
        }
    }
    if let Some(path) = environment
        .get("PATH")
        .cloned()
        .or_else(|| std::env::var("PATH").ok())
    {
        for entry in std::env::split_paths(std::ffi::OsStr::new(&path)) {
            if !entry.as_os_str().is_empty() {
                push_audit_candidate(&entry, &mut candidates, &mut seen_candidates);
            }
        }
    }
    for path in [Path::new("C:/"), Path::new("C:/Windows")] {
        push_audit_candidate(path, &mut candidates, &mut seen_candidates);
    }

    let mut flagged = Vec::new();
    let mut seen_flagged = HashSet::new();
    let mut checked = 0usize;
    let mut failed_scan = false;

    // Check immediate workspace children first so a local permission problem is
    // surfaced before the broader best-effort candidate sweep.
    match std::fs::read_dir(cwd) {
        Ok(entries) => {
            for (index, entry) in entries
                .take(AUDIT_MAX_ITEMS_PER_DIRECTORY.saturating_add(1))
                .enumerate()
            {
                if index >= AUDIT_MAX_ITEMS_PER_DIRECTORY {
                    failed_scan = true;
                    break;
                }
                if started.elapsed() > AUDIT_MAX_DURATION || checked >= AUDIT_MAX_CHECKED_ITEMS {
                    break;
                }
                let Ok(entry) = entry else {
                    failed_scan = true;
                    continue;
                };
                let Ok(file_type) = entry.file_type() else {
                    failed_scan = true;
                    continue;
                };
                if is_reparse_point(&entry.path(), &file_type, &mut failed_scan)
                    || !file_type.is_dir()
                {
                    continue;
                }
                check_audit_path(
                    &entry.path(),
                    started,
                    &mut checked,
                    &mut failed_scan,
                    &mut flagged,
                    &mut seen_flagged,
                );
            }
        }
        Err(_) => failed_scan = true,
    }

    for root in candidates {
        if started.elapsed() > AUDIT_MAX_DURATION || checked >= AUDIT_MAX_CHECKED_ITEMS {
            break;
        }
        check_audit_path(
            &root,
            started,
            &mut checked,
            &mut failed_scan,
            &mut flagged,
            &mut seen_flagged,
        );
        let Ok(entries) = std::fs::read_dir(&root) else {
            failed_scan = true;
            continue;
        };
        for (index, entry) in entries
            .take(AUDIT_MAX_ITEMS_PER_DIRECTORY.saturating_add(1))
            .enumerate()
        {
            if index >= AUDIT_MAX_ITEMS_PER_DIRECTORY {
                failed_scan = true;
                break;
            }
            if started.elapsed() > AUDIT_MAX_DURATION || checked >= AUDIT_MAX_CHECKED_ITEMS {
                break;
            }
            let Ok(entry) = entry else {
                failed_scan = true;
                continue;
            };
            let Ok(file_type) = entry.file_type() else {
                failed_scan = true;
                continue;
            };
            if is_reparse_point(&entry.path(), &file_type, &mut failed_scan) || !file_type.is_dir()
            {
                continue;
            }
            let normalized = entry.path().to_string_lossy().replace('\\', "/");
            if AUDIT_SKIP_SUFFIXES
                .iter()
                .any(|suffix| normalized.to_ascii_lowercase().ends_with(suffix))
            {
                continue;
            }
            check_audit_path(
                &entry.path(),
                started,
                &mut checked,
                &mut failed_scan,
                &mut flagged,
                &mut seen_flagged,
            );
        }
    }

    let sample_paths = flagged
        .iter()
        .take(AUDIT_SAMPLE_LIMIT)
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>();
    if checked >= AUDIT_MAX_CHECKED_ITEMS || started.elapsed() > AUDIT_MAX_DURATION {
        failed_scan = true;
    }
    WindowsWorldWritableAudit {
        sample_paths,
        extra_count: flagged.len().saturating_sub(AUDIT_SAMPLE_LIMIT),
        failed_scan,
    }
}

fn check_audit_path(
    path: &Path,
    started: Instant,
    checked: &mut usize,
    failed_scan: &mut bool,
    flagged: &mut Vec<std::path::PathBuf>,
    seen_flagged: &mut HashSet<String>,
) -> bool {
    if started.elapsed() > AUDIT_MAX_DURATION || *checked >= AUDIT_MAX_CHECKED_ITEMS {
        return false;
    }
    *checked += 1;
    match path_is_world_writable(path) {
        Ok(true) => {
            let key = canonical_audit_key(path);
            if seen_flagged.insert(key) {
                flagged.push(path.to_path_buf());
            }
            true
        }
        Ok(false) => false,
        Err(_) => {
            *failed_scan = true;
            false
        }
    }
}

fn is_reparse_point(path: &Path, file_type: &std::fs::FileType, failed_scan: &mut bool) -> bool {
    if file_type.is_symlink() {
        return true;
    }
    use std::os::windows::fs::MetadataExt;
    match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0,
        Err(_) => {
            *failed_scan = true;
            true
        }
    }
}

fn push_audit_candidate(
    path: &Path,
    candidates: &mut Vec<std::path::PathBuf>,
    seen: &mut HashSet<String>,
) {
    if has_reparse_component(path) {
        return;
    }
    let Ok(canonical) = path.canonicalize() else {
        return;
    };
    if seen.insert(canonical_audit_key(&canonical)) {
        candidates.push(canonical);
    }
}

fn has_reparse_component(path: &Path) -> bool {
    use std::os::windows::fs::MetadataExt;

    path.ancestors().any(|ancestor| {
        if ancestor.as_os_str().is_empty() {
            return false;
        }
        let Ok(metadata) = std::fs::symlink_metadata(ancestor) else {
            return true;
        };
        metadata.file_type().is_symlink()
            || metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
    })
}

fn canonical_audit_key(path: &Path) -> String {
    path.to_string_lossy().to_ascii_lowercase()
}

fn path_is_world_writable(path: &Path) -> io::Result<bool> {
    unsafe {
        let mut world = world_sid()?;
        let world_sid = world.as_mut_ptr() as *mut c_void;
        let wide = to_wide(path.as_os_str());
        let mut dacl = ptr::null_mut();
        let mut descriptor = ptr::null_mut();
        let result = GetNamedSecurityInfoW(
            wide.as_ptr(),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION,
            ptr::null_mut(),
            ptr::null_mut(),
            &mut dacl,
            ptr::null_mut(),
            &mut descriptor,
        );
        if result != ERROR_SUCCESS {
            return Err(io::Error::from_raw_os_error(result as i32));
        }
        let writable = dacl_has_world_write(dacl, world_sid);
        if !descriptor.is_null() {
            LocalFree(descriptor as HLOCAL);
        }
        Ok(writable)
    }
}

unsafe fn dacl_has_world_write(dacl: *mut ACL, world_sid: *mut c_void) -> bool {
    if dacl.is_null() {
        return false;
    }
    let mut info: ACL_SIZE_INFORMATION = std::mem::zeroed();
    if GetAclInformation(
        dacl as *const ACL,
        &mut info as *mut _ as *mut c_void,
        std::mem::size_of::<ACL_SIZE_INFORMATION>() as u32,
        AclSizeInformation,
    ) == 0
    {
        return false;
    }
    let mapping = GENERIC_MAPPING {
        GenericRead: 0x0012_0089,
        GenericWrite: 0x0012_0116,
        GenericExecute: 0x0012_00A0,
        GenericAll: 0x001F_01FF,
    };
    let write_mask = 0x0000_0002u32 | 0x0000_0004 | 0x0000_0010 | 0x0000_0100;
    for index in 0..info.AceCount {
        let mut raw_ace = ptr::null_mut();
        if GetAce(dacl as *const ACL, index as u32, &mut raw_ace) == 0 || raw_ace.is_null() {
            continue;
        }
        let header = &*(raw_ace as *const ACE_HEADER);
        // INHERIT_ONLY_ACE applies only to children, not to this path itself.
        // INHERITED_ACE remains effective and must still be audited.
        if header.AceType != ACCESS_ALLOWED_ACE_TYPE || header.AceFlags & INHERIT_ONLY_ACE != 0 {
            continue;
        }
        let ace = &*(raw_ace as *const ACCESS_ALLOWED_ACE);
        let sid_start = (raw_ace as usize
            + std::mem::size_of::<ACE_HEADER>()
            + std::mem::size_of::<u32>()) as *mut c_void;
        if EqualSid(sid_start, world_sid) == 0 {
            continue;
        }
        let mut mask = ace.Mask;
        MapGenericMask(&mut mask, &mapping);
        if mask & write_mask != 0 {
            return true;
        }
    }
    false
}
