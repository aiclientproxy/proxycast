use super::{
    to_wide, WINDOWS_SANDBOX_OFFLINE_USERNAME, WINDOWS_SANDBOX_ONLINE_USERNAME,
    WINDOWS_SANDBOX_USERS_GROUP,
};
use std::io;

pub(super) fn ensure_local_account(username: &str, password: &str) -> io::Result<()> {
    use windows_sys::Win32::NetworkManagement::NetManagement::{
        NERR_UserExists, NetUserAdd, NetUserSetInfo, UF_DONT_EXPIRE_PASSWD, UF_SCRIPT, USER_INFO_1,
        USER_INFO_1003, USER_PRIV_USER,
    };

    let username_w = to_wide(username);
    let mut password_w = to_wide(password);
    let mut info = USER_INFO_1 {
        usri1_name: username_w.as_ptr() as *mut _,
        usri1_password: password_w.as_mut_ptr(),
        usri1_password_age: 0,
        usri1_priv: USER_PRIV_USER,
        usri1_home_dir: std::ptr::null_mut(),
        usri1_comment: std::ptr::null_mut(),
        usri1_flags: UF_SCRIPT | UF_DONT_EXPIRE_PASSWD,
        usri1_script_path: std::ptr::null_mut(),
    };
    let result = unsafe {
        NetUserAdd(
            std::ptr::null(),
            1,
            &mut info as *mut _ as *const u8,
            std::ptr::null_mut(),
        )
    };
    if result == 0 {
        return Ok(());
    }
    if result != NERR_UserExists {
        return Err(io::Error::other(format!(
            "NetUserAdd({username}) failed: {result}"
        )));
    }

    let mut password_update = USER_INFO_1003 {
        usri1003_password: password_w.as_mut_ptr(),
    };
    let result = unsafe {
        NetUserSetInfo(
            std::ptr::null(),
            username_w.as_ptr(),
            1003,
            &mut password_update as *mut _ as *const u8,
            std::ptr::null_mut(),
        )
    };
    if result == 0 {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "NetUserSetInfo({username}) failed: {result}"
        )))
    }
}

pub(super) fn ensure_local_group(name: &str, comment: &str) -> io::Result<()> {
    use windows_sys::Win32::Foundation::ERROR_ALIAS_EXISTS;
    use windows_sys::Win32::NetworkManagement::NetManagement::{
        NERR_GroupExists, NERR_Success, NetLocalGroupAdd, LOCALGROUP_INFO_1,
    };

    let mut name_w = to_wide(name);
    let mut comment_w = to_wide(comment);
    let info = LOCALGROUP_INFO_1 {
        lgrpi1_name: name_w.as_mut_ptr(),
        lgrpi1_comment: comment_w.as_mut_ptr(),
    };
    let mut parameter_error = 0;
    let result = unsafe {
        NetLocalGroupAdd(
            std::ptr::null(),
            1,
            &info as *const _ as *const u8,
            &mut parameter_error,
        )
    };
    if matches!(result, NERR_Success | ERROR_ALIAS_EXISTS | NERR_GroupExists) {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "NetLocalGroupAdd({name}) failed: {result} (parameter {parameter_error})"
        )))
    }
}

pub(super) fn ensure_local_group_member(group: &str, account: &str) -> io::Result<()> {
    use windows_sys::Win32::Foundation::ERROR_MEMBER_IN_ALIAS;
    use windows_sys::Win32::NetworkManagement::NetManagement::{
        NERR_Success, NetLocalGroupAddMembers, LOCALGROUP_MEMBERS_INFO_3,
    };

    let group_w = to_wide(group);
    let mut account_w = to_wide(account);
    let member = LOCALGROUP_MEMBERS_INFO_3 {
        lgrmi3_domainandname: account_w.as_mut_ptr(),
    };
    let result = unsafe {
        NetLocalGroupAddMembers(
            std::ptr::null(),
            group_w.as_ptr(),
            3,
            &member as *const _ as *const u8,
            1,
        )
    };
    if matches!(result, NERR_Success | ERROR_MEMBER_IN_ALIAS) {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "NetLocalGroupAddMembers({group}, {account}) failed: {result}"
        )))
    }
}

pub(crate) fn verify_windows_sandbox_group_membership(account: &str) -> io::Result<()> {
    if !matches!(
        account,
        WINDOWS_SANDBOX_OFFLINE_USERNAME | WINDOWS_SANDBOX_ONLINE_USERNAME
    ) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "unknown Windows sandbox account",
        ));
    }
    validate_windows_sandbox_group_membership(account).map_err(io::Error::other)
}

pub(super) fn validate_windows_sandbox_group_membership(account: &str) -> Result<(), String> {
    let account_sid = resolve_windows_account_sid(account)
        .map_err(|error| format!("sandbox account SID validation failed: {error}"))?;
    windows_sandbox_users_group_sid()
        .map_err(|error| format!("sandbox users group validation failed: {error}"))?;
    if local_group_contains_sid(WINDOWS_SANDBOX_USERS_GROUP, &account_sid)? {
        Ok(())
    } else {
        Err(format!(
            "Windows sandbox account {account} is not a member of {WINDOWS_SANDBOX_USERS_GROUP}"
        ))
    }
}

pub(crate) fn windows_sandbox_users_group_sid() -> io::Result<String> {
    use std::ffi::c_void;
    use windows_sys::Win32::Foundation::{LocalFree, HLOCAL};
    use windows_sys::Win32::Security::Authorization::ConvertSidToStringSidW;

    let sid = resolve_windows_account_sid(WINDOWS_SANDBOX_USERS_GROUP).map_err(io::Error::other)?;
    let mut text = std::ptr::null_mut();
    if unsafe { ConvertSidToStringSidW(sid.as_ptr() as *mut c_void, &mut text) } == 0 {
        return Err(io::Error::last_os_error());
    }
    if text.is_null() {
        return Err(io::Error::other(
            "ConvertSidToStringSidW returned a null SID string",
        ));
    }
    let length = (0..184).find(|index| unsafe { *text.add(*index) == 0 });
    let result = match length {
        Some(length) => String::from_utf16(unsafe { std::slice::from_raw_parts(text, length) })
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "group SID is not UTF-16")),
        None => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "group SID string exceeds the accepted length",
        )),
    };
    unsafe {
        LocalFree(text as HLOCAL);
    }
    result
}

fn local_group_contains_sid(group: &str, account_sid: &[u8]) -> Result<bool, String> {
    use std::ffi::c_void;
    use windows_sys::Win32::NetworkManagement::NetManagement::{
        NERR_Success, NetApiBufferFree, NetLocalGroupGetMembers, LOCALGROUP_MEMBERS_INFO_0,
        MAX_PREFERRED_LENGTH,
    };
    use windows_sys::Win32::Security::EqualSid;

    let group_w = to_wide(group);
    let mut buffer = std::ptr::null_mut();
    let mut entries_read = 0;
    let mut total_entries = 0;
    let result = unsafe {
        NetLocalGroupGetMembers(
            std::ptr::null(),
            group_w.as_ptr(),
            0,
            &mut buffer,
            MAX_PREFERRED_LENGTH,
            &mut entries_read,
            &mut total_entries,
            std::ptr::null_mut(),
        )
    };
    if result != NERR_Success {
        if !buffer.is_null() {
            unsafe {
                NetApiBufferFree(buffer as *const c_void);
            }
        }
        return Err(format!("NetLocalGroupGetMembers({group}) failed: {result}"));
    }
    if entries_read > total_entries || (entries_read > 0 && buffer.is_null()) {
        if !buffer.is_null() {
            unsafe {
                NetApiBufferFree(buffer as *const c_void);
            }
        }
        return Err(format!(
            "NetLocalGroupGetMembers({group}) returned an invalid member buffer"
        ));
    }

    let found = if entries_read == 0 {
        false
    } else {
        let members = unsafe {
            std::slice::from_raw_parts(
                buffer as *const LOCALGROUP_MEMBERS_INFO_0,
                entries_read as usize,
            )
        };
        members.iter().any(|member| unsafe {
            EqualSid(member.lgrmi0_sid, account_sid.as_ptr() as *mut c_void) != 0
        })
    };
    if !buffer.is_null() {
        unsafe {
            NetApiBufferFree(buffer as *const c_void);
        }
    }
    Ok(found)
}

pub(crate) fn resolve_windows_account_sid(account: &str) -> Result<Vec<u8>, String> {
    use std::ffi::{c_void, OsStr};
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::{GetLastError, ERROR_INSUFFICIENT_BUFFER};
    use windows_sys::Win32::Security::{LookupAccountNameW, SID_NAME_USE};

    let account: Vec<u16> = OsStr::new(account)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let mut sid = vec![0u8; 68];
    let mut sid_len = sid.len() as u32;
    let mut domain = Vec::<u16>::new();
    let mut domain_len = 0u32;
    let mut sid_type: SID_NAME_USE = 0;
    loop {
        let ok = unsafe {
            LookupAccountNameW(
                std::ptr::null(),
                account.as_ptr(),
                sid.as_mut_ptr() as *mut c_void,
                &mut sid_len,
                domain.as_mut_ptr(),
                &mut domain_len,
                &mut sid_type,
            )
        };
        if ok != 0 {
            sid.truncate(sid_len as usize);
            return Ok(sid);
        }
        let error = unsafe { GetLastError() };
        if error != ERROR_INSUFFICIENT_BUFFER {
            return Err(format!("LookupAccountNameW failed: {error}"));
        }
        sid.resize(sid_len as usize, 0);
        domain.resize(domain_len as usize, 0);
    }
}
