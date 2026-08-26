use super::{last_os_error, to_wide, LocalSid};
use std::io;
use std::ptr;
use windows_sys::Win32::Foundation::{
    CloseHandle, LocalFree, ERROR_SUCCESS, HANDLE, HLOCAL, INVALID_HANDLE_VALUE, WAIT_ABANDONED,
    WAIT_OBJECT_0,
};
use windows_sys::Win32::Security::Authorization::{
    GetSecurityInfo, SetEntriesInAclW, SetSecurityInfo, EXPLICIT_ACCESS_W, REVOKE_ACCESS,
    SET_ACCESS, SE_KERNEL_OBJECT, TRUSTEE_IS_SID, TRUSTEE_IS_UNKNOWN, TRUSTEE_W,
};
use windows_sys::Win32::Security::{ACL, DACL_SECURITY_INFORMATION};
use windows_sys::Win32::Storage::FileSystem::{
    CreateFileW, FILE_GENERIC_EXECUTE, FILE_GENERIC_READ, FILE_GENERIC_WRITE, FILE_SHARE_READ,
    FILE_SHARE_WRITE, OPEN_EXISTING,
};
use windows_sys::Win32::System::Threading::{
    CreateMutexW, ReleaseMutex, WaitForSingleObject, INFINITE,
};

const READ_CONTROL: u32 = 0x0002_0000;
const WRITE_DAC: u32 = 0x0004_0000;
const NULL_DEVICE_MUTEX: &str = "Local\\LimeSandboxNullDeviceAcl";

/// Temporarily grants one restricted-token capability access to the Windows NUL device.
///
/// PowerShell may open NUL while initializing redirected standard streams. The capability ACE
/// is removed when this lease ends so per-execution SIDs do not accumulate in the device DACL.
pub(super) struct NullDeviceLease {
    capability: LocalSid,
    released: bool,
}

impl NullDeviceLease {
    pub(super) fn acquire(capability_sid: &str) -> io::Result<Self> {
        let capability = LocalSid::parse(capability_sid)?;
        let _guard = NullDeviceAclLock::acquire()?;
        update_null_device_acl(capability.raw(), SET_ACCESS)?;
        Ok(Self {
            capability,
            released: false,
        })
    }

    pub(super) fn release(&mut self) -> io::Result<()> {
        if self.released {
            return Ok(());
        }
        let _guard = NullDeviceAclLock::acquire()?;
        update_null_device_acl(self.capability.raw(), REVOKE_ACCESS)?;
        self.released = true;
        Ok(())
    }
}

impl Drop for NullDeviceLease {
    fn drop(&mut self) {
        let _ = self.release();
    }
}

struct NullDeviceAclLock(HANDLE);

impl NullDeviceAclLock {
    fn acquire() -> io::Result<Self> {
        let name = to_wide(NULL_DEVICE_MUTEX);
        let handle = unsafe { CreateMutexW(ptr::null_mut(), 0, name.as_ptr()) };
        if handle == 0 {
            return Err(last_os_error("CreateMutexW(Windows sandbox NUL)"));
        }
        let wait = unsafe { WaitForSingleObject(handle, INFINITE) };
        if wait != WAIT_OBJECT_0 && wait != WAIT_ABANDONED {
            unsafe {
                CloseHandle(handle);
            }
            return Err(last_os_error("WaitForSingleObject(Windows sandbox NUL)"));
        }
        Ok(Self(handle))
    }
}

impl Drop for NullDeviceAclLock {
    fn drop(&mut self) {
        unsafe {
            let _ = ReleaseMutex(self.0);
            CloseHandle(self.0);
        }
    }
}

fn update_null_device_acl(capability: *mut std::ffi::c_void, mode: i32) -> io::Result<()> {
    let path = to_wide(r"\\.\NUL");
    let handle = unsafe {
        CreateFileW(
            path.as_ptr(),
            READ_CONTROL | WRITE_DAC,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            ptr::null_mut(),
            OPEN_EXISTING,
            0,
            0,
        )
    };
    if handle == 0 || handle == INVALID_HANDLE_VALUE {
        return Err(last_os_error("CreateFileW(Windows sandbox NUL)"));
    }
    let result = update_null_device_handle_acl(handle, capability, mode);
    unsafe {
        CloseHandle(handle);
    }
    result
}

fn update_null_device_handle_acl(
    handle: HANDLE,
    capability: *mut std::ffi::c_void,
    mode: i32,
) -> io::Result<()> {
    let mut descriptor = ptr::null_mut();
    let mut dacl: *mut ACL = ptr::null_mut();
    let security = unsafe {
        GetSecurityInfo(
            handle,
            SE_KERNEL_OBJECT,
            DACL_SECURITY_INFORMATION,
            ptr::null_mut(),
            ptr::null_mut(),
            &mut dacl,
            ptr::null_mut(),
            &mut descriptor,
        )
    };
    if security != ERROR_SUCCESS {
        return Err(io::Error::from_raw_os_error(security as i32));
    }
    let result = (|| {
        if dacl.is_null() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "Windows sandbox NUL device has no editable DACL",
            ));
        }
        let entry = EXPLICIT_ACCESS_W {
            grfAccessPermissions: FILE_GENERIC_READ | FILE_GENERIC_WRITE | FILE_GENERIC_EXECUTE,
            grfAccessMode: mode,
            grfInheritance: 0,
            Trustee: TRUSTEE_W {
                pMultipleTrustee: ptr::null_mut(),
                MultipleTrusteeOperation: 0,
                TrusteeForm: TRUSTEE_IS_SID,
                TrusteeType: TRUSTEE_IS_UNKNOWN,
                ptstrName: capability as *mut u16,
            },
        };
        let mut updated_dacl: *mut ACL = ptr::null_mut();
        let update = unsafe { SetEntriesInAclW(1, &entry, dacl, &mut updated_dacl) };
        if update != ERROR_SUCCESS {
            return Err(io::Error::from_raw_os_error(update as i32));
        }
        if updated_dacl.is_null() {
            return Err(io::Error::other(
                "SetEntriesInAclW(Windows sandbox NUL) returned a null DACL",
            ));
        }
        let applied = unsafe {
            SetSecurityInfo(
                handle,
                SE_KERNEL_OBJECT,
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
            Err(io::Error::from_raw_os_error(applied as i32))
        }
    })();
    unsafe {
        if !descriptor.is_null() {
            LocalFree(descriptor as HLOCAL);
        }
    }
    result
}
