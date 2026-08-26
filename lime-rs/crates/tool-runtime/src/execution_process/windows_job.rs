use super::*;

pub(super) fn create_kill_on_close_job() -> io::Result<OwnedHandle> {
    let job = OwnedHandle::new(
        unsafe { CreateJobObjectW(ptr::null(), ptr::null()) },
        "CreateJobObjectW",
    )?;
    // Keep breakaway disabled while the restricted process tree is running.
    // It is enabled only after a normal root exit, when the reaper has taken
    // ownership of preserving descendants.
    set_job_limit_flags(job.raw(), JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)?;
    Ok(job)
}

fn set_job_limit_flags(job: HANDLE, flags: u32) -> io::Result<()> {
    let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
    limits.BasicLimitInformation.LimitFlags = flags;
    if unsafe {
        SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &limits as *const _ as *const c_void,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        )
    } == 0
    {
        Err(last_os_error("SetInformationJobObject"))
    } else {
        Ok(())
    }
}

pub(super) fn preserve_job_descendants(job: &OwnedHandle) -> io::Result<()> {
    set_job_limit_flags(job.raw(), JOB_OBJECT_LIMIT_BREAKAWAY_OK)
}

pub(super) fn active_job_processes(job: &OwnedHandle) -> io::Result<u32> {
    let mut accounting: JOBOBJECT_BASIC_ACCOUNTING_INFORMATION = unsafe { std::mem::zeroed() };
    if unsafe {
        QueryInformationJobObject(
            job.raw(),
            JobObjectBasicAccountingInformation,
            &mut accounting as *mut _ as *mut c_void,
            std::mem::size_of::<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>() as u32,
            ptr::null_mut(),
        )
    } == 0
    {
        Err(last_os_error("QueryInformationJobObject"))
    } else {
        Ok(accounting.ActiveProcesses)
    }
}

pub(super) fn write_pipe(handle: HANDLE, mut bytes: &[u8]) -> io::Result<()> {
    while !bytes.is_empty() {
        let mut written = 0;
        if unsafe {
            WriteFile(
                handle,
                bytes.as_ptr(),
                bytes.len().min(u32::MAX as usize) as u32,
                &mut written,
                ptr::null_mut(),
            )
        } == 0
        {
            return Err(last_os_error("WriteFile"));
        }
        if written == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "WriteFile wrote zero bytes",
            ));
        }
        bytes = &bytes[written as usize..];
    }
    Ok(())
}
