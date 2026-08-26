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

fn active_job_processes(job: &OwnedHandle) -> io::Result<u32> {
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

fn reap_preserved_job(job: OwnedHandle, acl_lease: AclLease) {
    loop {
        match active_job_processes(&job) {
            Ok(0) => break,
            Ok(_) => thread::sleep(Duration::from_millis(JOB_REAPER_POLL_MILLIS)),
            Err(_) => {
                unsafe {
                    TerminateJobObject(job.raw(), 1);
                }
                break;
            }
        }
    }
    drop(acl_lease);
}

#[allow(clippy::too_many_arguments)]
pub(super) fn supervise_restricted_process(
    process_handle: OwnedHandle,
    thread_handle: OwnedHandle,
    job: OwnedHandle,
    mut stdin: Option<OwnedHandle>,
    pseudoconsole: Option<RestrictedConpty>,
    stdout_reader: thread::JoinHandle<()>,
    stderr_reader: Option<thread::JoinHandle<()>>,
    acl_lease: AclLease,
    process: Arc<Mutex<ExecutionProcess>>,
    state_tx: watch::Sender<ExecutionProcessSnapshot>,
    final_tx: oneshot::Sender<ExecutionProcessSnapshot>,
    control_rx: std::sync::mpsc::Receiver<LocalExecutionControl>,
) {
    let mut wait_error = None;
    loop {
        let wait = unsafe { WaitForSingleObject(process_handle.raw(), CONTROL_POLL_MILLIS) };
        if wait == WAIT_OBJECT_0 {
            break;
        }
        drain_controls(
            &job,
            pseudoconsole.as_ref(),
            &mut stdin,
            &process,
            &state_tx,
            &control_rx,
        );
        if wait != WAIT_TIMEOUT {
            wait_error = Some(last_os_error("WaitForSingleObject").to_string());
            unsafe {
                TerminateJobObject(job.raw(), 1);
            }
            break;
        }
    }

    stdin.take();
    let preserve_descendants =
        should_preserve_windows_job(wait_error.is_some(), process.blocking_lock().status());
    if preserve_descendants && preserve_job_descendants(&job).is_ok() {
        thread::spawn(move || reap_preserved_job(job, acl_lease));
    } else {
        unsafe {
            TerminateJobObject(job.raw(), 1);
        }
        drop(job);
        drop(acl_lease);
    }
    drop(pseudoconsole);
    let _ = stdout_reader.join();
    if let Some(stderr_reader) = stderr_reader {
        let _ = stderr_reader.join();
    }
    drop(thread_handle);

    let mut exit_code = 1;
    let exit_result = unsafe { GetExitCodeProcess(process_handle.raw(), &mut exit_code) };
    let final_snapshot = {
        let mut guard = process.blocking_lock();
        if !guard.status().is_terminal() {
            if let Some(error) = wait_error {
                guard.fail(error);
            } else if exit_result == 0 {
                guard.fail(last_os_error("GetExitCodeProcess").to_string());
            } else {
                guard.exit(i32::try_from(exit_code).unwrap_or(-1));
            }
        }
        guard.snapshot()
    };
    let _ = state_tx.send(final_snapshot.clone());
    let _ = final_tx.send(final_snapshot);
}

fn drain_controls(
    job: &OwnedHandle,
    pseudoconsole: Option<&RestrictedConpty>,
    stdin: &mut Option<OwnedHandle>,
    process: &Arc<Mutex<ExecutionProcess>>,
    state_tx: &watch::Sender<ExecutionProcessSnapshot>,
    control_rx: &std::sync::mpsc::Receiver<LocalExecutionControl>,
) {
    loop {
        match control_rx.try_recv() {
            Ok(LocalExecutionControl::WriteStdin(bytes)) => {
                if let Some(stdin) = stdin.as_ref() {
                    let _ = write_pipe(stdin.raw(), &bytes);
                }
            }
            Ok(LocalExecutionControl::CloseStdin) => {
                stdin.take();
            }
            Ok(LocalExecutionControl::Resize { rows, cols }) => {
                if let Some(pseudoconsole) = pseudoconsole {
                    let _ = pseudoconsole.resize(rows, cols);
                }
            }
            Ok(LocalExecutionControl::Interrupt) => {
                update_status_blocking(process, state_tx, ExecutionProcessStatus::Interrupted);
                unsafe {
                    TerminateJobObject(job.raw(), 1);
                }
            }
            Ok(LocalExecutionControl::Terminate) => {
                update_status_blocking(process, state_tx, ExecutionProcessStatus::Terminated);
                unsafe {
                    TerminateJobObject(job.raw(), 1);
                }
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => break,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                update_status_blocking(process, state_tx, ExecutionProcessStatus::Terminated);
                unsafe {
                    TerminateJobObject(job.raw(), 1);
                }
                break;
            }
        }
    }
}

fn write_pipe(handle: HANDLE, mut bytes: &[u8]) -> io::Result<()> {
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

pub(super) fn read_pipe_stream(
    handle: OwnedHandle,
    kind: ExecutionOutputKind,
    process: Arc<Mutex<ExecutionProcess>>,
    output_tx: mpsc::UnboundedSender<ExecutionOutputDelta>,
    state_tx: watch::Sender<ExecutionProcessSnapshot>,
) {
    let mut buffer = vec![0u8; PROCESS_OUTPUT_CHUNK_BYTES];
    loop {
        let mut read = 0;
        let ok = unsafe {
            ReadFile(
                handle.raw(),
                buffer.as_mut_ptr(),
                buffer.len() as u32,
                &mut read,
                ptr::null_mut(),
            )
        };
        if ok == 0 || read == 0 {
            break;
        }
        let (delta, snapshot) = {
            let mut guard = process.blocking_lock();
            let delta = guard.append_output(kind, &buffer[..read as usize]);
            let snapshot = guard.snapshot();
            (delta, snapshot)
        };
        let _ = output_tx.send(delta);
        let _ = state_tx.send(snapshot);
    }
}

fn update_status_blocking(
    process: &Arc<Mutex<ExecutionProcess>>,
    state_tx: &watch::Sender<ExecutionProcessSnapshot>,
    status: ExecutionProcessStatus,
) {
    let snapshot = {
        let mut guard = process.blocking_lock();
        match status {
            ExecutionProcessStatus::Interrupted => guard.interrupt(),
            ExecutionProcessStatus::Terminated => guard.terminate(),
            ExecutionProcessStatus::Failed => guard.fail("process failed"),
            ExecutionProcessStatus::Exited => guard.exit(-1),
            ExecutionProcessStatus::Starting | ExecutionProcessStatus::Running => {}
        }
        guard.snapshot()
    };
    let _ = state_tx.send(snapshot);
}
