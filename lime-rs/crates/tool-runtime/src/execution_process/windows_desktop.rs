use super::{last_os_error, logon_sid, to_wide, OwnedHandle};
use std::io;
use std::ptr;
use windows_sys::Win32::Foundation::{LocalFree, ERROR_SUCCESS, HLOCAL};
use windows_sys::Win32::Security::Authorization::{
    SetEntriesInAclW, SetSecurityInfo, EXPLICIT_ACCESS_W, GRANT_ACCESS, SE_WINDOW_OBJECT,
    TRUSTEE_IS_SID, TRUSTEE_IS_UNKNOWN, TRUSTEE_W,
};
use windows_sys::Win32::Security::{DACL_SECURITY_INFORMATION, TOKEN_QUERY};
use windows_sys::Win32::System::StationsAndDesktops::{
    CloseDesktop, CreateDesktopW, DESKTOP_CREATEMENU, DESKTOP_CREATEWINDOW, DESKTOP_DELETE,
    DESKTOP_ENUMERATE, DESKTOP_HOOKCONTROL, DESKTOP_JOURNALPLAYBACK, DESKTOP_JOURNALRECORD,
    DESKTOP_READOBJECTS, DESKTOP_READ_CONTROL, DESKTOP_SWITCHDESKTOP, DESKTOP_WRITEOBJECTS,
    DESKTOP_WRITE_DAC, DESKTOP_WRITE_OWNER, HDESK,
};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

const DESKTOP_ALL_ACCESS: u32 = DESKTOP_READOBJECTS
    | DESKTOP_CREATEWINDOW
    | DESKTOP_CREATEMENU
    | DESKTOP_HOOKCONTROL
    | DESKTOP_JOURNALRECORD
    | DESKTOP_JOURNALPLAYBACK
    | DESKTOP_ENUMERATE
    | DESKTOP_WRITEOBJECTS
    | DESKTOP_SWITCHDESKTOP
    | DESKTOP_DELETE
    | DESKTOP_READ_CONTROL
    | DESKTOP_WRITE_DAC
    | DESKTOP_WRITE_OWNER;

pub(super) struct LaunchDesktop {
    handle: HDESK,
    startup_name: Vec<u16>,
}

impl LaunchDesktop {
    pub(super) fn prepare() -> io::Result<Self> {
        let name = format!("LimeSandboxDesktop-{}", uuid::Uuid::new_v4().simple());
        let name_wide = to_wide(&name);
        let handle = unsafe {
            CreateDesktopW(
                name_wide.as_ptr(),
                ptr::null(),
                ptr::null_mut(),
                0,
                DESKTOP_ALL_ACCESS,
                ptr::null_mut(),
            )
        };
        if handle == 0 {
            return Err(last_os_error("CreateDesktopW(Windows sandbox)"));
        }

        let desktop = Self {
            handle,
            startup_name: to_wide(format!("Winsta0\\{name}")),
        };
        if let Err(error) = desktop.grant_logon_access() {
            drop(desktop);
            return Err(error);
        }
        Ok(desktop)
    }

    pub(super) fn startup_name(&mut self) -> *mut u16 {
        self.startup_name.as_mut_ptr()
    }

    fn grant_logon_access(&self) -> io::Result<()> {
        let token = current_process_token()?;
        let mut sid = unsafe { logon_sid(token.raw())? };
        let entry = EXPLICIT_ACCESS_W {
            grfAccessPermissions: DESKTOP_ALL_ACCESS,
            grfAccessMode: GRANT_ACCESS,
            grfInheritance: 0,
            Trustee: TRUSTEE_W {
                pMultipleTrustee: ptr::null_mut(),
                MultipleTrusteeOperation: 0,
                TrusteeForm: TRUSTEE_IS_SID,
                TrusteeType: TRUSTEE_IS_UNKNOWN,
                ptstrName: sid.as_mut_ptr() as *mut u16,
            },
        };
        let mut dacl = ptr::null_mut();
        let result = unsafe { SetEntriesInAclW(1, &entry, ptr::null_mut(), &mut dacl) };
        if result != ERROR_SUCCESS {
            return Err(io::Error::other(format!(
                "SetEntriesInAclW(Windows sandbox desktop) failed: {result}"
            )));
        }
        let result = unsafe {
            SetSecurityInfo(
                self.handle,
                SE_WINDOW_OBJECT,
                DACL_SECURITY_INFORMATION,
                ptr::null_mut(),
                ptr::null_mut(),
                dacl,
                ptr::null_mut(),
            )
        };
        unsafe {
            if !dacl.is_null() {
                LocalFree(dacl as HLOCAL);
            }
        }
        if result != ERROR_SUCCESS {
            return Err(io::Error::other(format!(
                "SetSecurityInfo(Windows sandbox desktop) failed: {result}"
            )));
        }
        Ok(())
    }
}

impl Drop for LaunchDesktop {
    fn drop(&mut self) {
        if self.handle != 0 {
            unsafe {
                CloseDesktop(self.handle);
            }
            self.handle = 0;
        }
    }
}

fn current_process_token() -> io::Result<OwnedHandle> {
    let mut token = 0;
    if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) } == 0 {
        return Err(last_os_error("OpenProcessToken(Windows sandbox desktop)"));
    }
    OwnedHandle::new(token, "OpenProcessToken(Windows sandbox desktop)")
}
