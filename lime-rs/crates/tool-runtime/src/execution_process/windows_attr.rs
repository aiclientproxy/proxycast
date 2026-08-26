use std::ffi::c_void;
use std::io;
use std::ptr;

use windows_sys::Win32::Foundation::{GetLastError, HANDLE};
use windows_sys::Win32::System::Threading::{
    DeleteProcThreadAttributeList, InitializeProcThreadAttributeList, UpdateProcThreadAttribute,
    LPPROC_THREAD_ATTRIBUTE_LIST,
};

const PROC_THREAD_ATTRIBUTE_HANDLE_LIST: usize = 0x0002_0002;
const PROC_THREAD_ATTRIBUTE_JOB_LIST: usize = 0x0002_000D;
const PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE: usize = 0x0002_0016;

/// Owns the native attribute buffer and the backing handle arrays referenced by it.
pub(super) struct ProcessAttributeList {
    buffer: Vec<u8>,
    handle_list: Vec<HANDLE>,
    job_list: Vec<HANDLE>,
    initialized: bool,
}

impl ProcessAttributeList {
    pub(super) fn new(attribute_count: u32) -> io::Result<Self> {
        let mut size = 0usize;
        unsafe {
            InitializeProcThreadAttributeList(ptr::null_mut(), attribute_count, 0, &mut size);
        }
        if size == 0 {
            return Err(io::Error::from_raw_os_error(
                unsafe { GetLastError() } as i32
            ));
        }

        let mut attributes = Self {
            buffer: vec![0; size],
            handle_list: Vec::new(),
            job_list: Vec::new(),
            initialized: false,
        };
        let ok = unsafe {
            InitializeProcThreadAttributeList(
                attributes.as_mut_ptr(),
                attribute_count,
                0,
                &mut size,
            )
        };
        if ok == 0 {
            return Err(io::Error::from_raw_os_error(
                unsafe { GetLastError() } as i32
            ));
        }
        attributes.initialized = true;
        Ok(attributes)
    }

    pub(super) fn set_handle_list(&mut self, handles: &[HANDLE]) -> io::Result<()> {
        self.handle_list = handles.to_vec();
        let value = self.handle_list.as_ptr() as *const c_void;
        let size = std::mem::size_of_val(self.handle_list.as_slice());
        self.update(PROC_THREAD_ATTRIBUTE_HANDLE_LIST, value, size)
    }

    pub(super) fn set_job(&mut self, job: HANDLE) -> io::Result<()> {
        self.job_list = vec![job];
        let value = self.job_list.as_ptr() as *const c_void;
        let size = std::mem::size_of_val(self.job_list.as_slice());
        self.update(PROC_THREAD_ATTRIBUTE_JOB_LIST, value, size)
    }

    pub(super) fn set_pseudoconsole(&mut self, pseudoconsole: isize) -> io::Result<()> {
        self.update(
            PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE,
            pseudoconsole as *const c_void,
            std::mem::size_of::<HANDLE>(),
        )
    }

    pub(super) fn as_mut_ptr(&mut self) -> LPPROC_THREAD_ATTRIBUTE_LIST {
        self.buffer.as_mut_ptr() as LPPROC_THREAD_ATTRIBUTE_LIST
    }

    fn update(&mut self, attribute: usize, value: *const c_void, size: usize) -> io::Result<()> {
        let ok = unsafe {
            UpdateProcThreadAttribute(
                self.as_mut_ptr(),
                0,
                attribute,
                value,
                size,
                ptr::null_mut(),
                ptr::null(),
            )
        };
        if ok == 0 {
            return Err(io::Error::from_raw_os_error(
                unsafe { GetLastError() } as i32
            ));
        }
        Ok(())
    }
}

impl Drop for ProcessAttributeList {
    fn drop(&mut self) {
        if self.initialized {
            unsafe {
                DeleteProcThreadAttributeList(self.as_mut_ptr());
            }
        }
    }
}
