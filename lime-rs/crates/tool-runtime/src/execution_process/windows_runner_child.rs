use super::windows_job::write_pipe;
use super::windows_job::{active_job_processes, preserve_job_descendants};
use super::windows_runner_protocol::{
    decode_bytes, encode_bytes, read_frame, write_frame, RunnerMessage, RunnerSpawnRequest,
};
use super::{
    create_restricted_token, last_os_error, spawn_restricted_process, to_wide,
    LocalExecutionRequest, LocalSid, OwnedHandle, RestrictedConpty,
};
use std::fs::File;
use std::io;
use std::os::windows::io::FromRawHandle;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;
use windows_sys::Win32::Foundation::{EqualSid, WAIT_OBJECT_0};
use windows_sys::Win32::Storage::FileSystem::{
    CreateFileW, ReadFile, FILE_GENERIC_READ, FILE_GENERIC_WRITE, OPEN_EXISTING,
};
use windows_sys::Win32::System::JobObjects::TerminateJobObject;
use windows_sys::Win32::System::Threading::{
    GetExitCodeProcess, GetProcessId, WaitForSingleObject, INFINITE,
};

const OUTPUT_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);
const JOB_REAPER_POLL: Duration = Duration::from_millis(250);

pub(super) fn run() -> io::Result<()> {
    let (pipe_in, pipe_out) = parse_pipe_arguments()?;
    let pipe_read = open_pipe(&pipe_in, FILE_GENERIC_READ)?;
    let pipe_write = open_pipe(&pipe_out, FILE_GENERIC_WRITE)?;
    let mut pipe_read = unsafe { File::from_raw_handle(pipe_read.into_raw() as _) };
    let pipe_write = Arc::new(Mutex::new(unsafe {
        File::from_raw_handle(pipe_write.into_raw() as _)
    }));

    let request = match read_frame(&mut pipe_read) {
        Ok(Some(RunnerMessage::Spawn { payload })) => payload,
        Ok(Some(_)) => {
            return send_start_error(
                &pipe_write,
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "runner expected a spawn request",
                ),
            )
        }
        Ok(None) => {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "runner pipe closed before spawn request",
            ))
        }
        Err(error) => return send_start_error(&pipe_write, error),
    };

    if let Err(error) = run_spawned_process(request, pipe_read, Arc::clone(&pipe_write)) {
        let _ = write_locked(
            &pipe_write,
            RunnerMessage::Error {
                windows_error_code: error.raw_os_error().map(|value| value as u32),
                message: error.to_string(),
            },
        );
        return Err(error);
    }
    Ok(())
}

fn run_spawned_process(
    request: RunnerSpawnRequest,
    pipe_read: File,
    pipe_write: Arc<Mutex<File>>,
) -> io::Result<()> {
    verify_runner_identity(&request.expected_account_sid)?;
    let token = create_restricted_token(&request.capability_sid)?;
    let local_request = LocalExecutionRequest {
        process_id: "windows-sandbox-runner-child".to_string(),
        tool_id: "windows-sandbox-runner-child".to_string(),
        tool_name: "exec_command".to_string(),
        command: request.command,
        cwd: Some(request.cwd.clone()),
        env: request.env,
        tty: request.tty,
        stdin: request.stdin_open,
        env_clear: true,
        pty_size: request.pty_size,
        sandbox: None,
    };
    let spawned = spawn_restricted_process(&local_request, &request.cwd, token.raw())?;
    drop(token);

    let super::SpawnedRestrictedProcess {
        process,
        thread: process_thread,
        job,
        stdin_write,
        stdout_read,
        stderr_read,
        pseudoconsole,
    } = spawned;
    let process_id = unsafe { GetProcessId(process.raw()) };
    write_locked(&pipe_write, RunnerMessage::Ready { process_id })?;

    let (reader_done_tx, reader_done_rx) = mpsc::channel();
    let output_kind = if pseudoconsole.is_some() {
        super::ExecutionOutputKind::Combined
    } else {
        super::ExecutionOutputKind::Stdout
    };
    spawn_output_reader(
        stdout_read,
        output_kind,
        Arc::clone(&pipe_write),
        reader_done_tx.clone(),
    );
    if let Some(stderr) = stderr_read {
        spawn_output_reader(
            stderr,
            super::ExecutionOutputKind::Stderr,
            Arc::clone(&pipe_write),
            reader_done_tx,
        );
    }

    let job = Arc::new(job);
    let stdin = Arc::new(Mutex::new(stdin_write));
    let pseudoconsole = pseudoconsole.map(Arc::new);
    let terminated = Arc::new(AtomicBool::new(false));
    spawn_control_reader(
        pipe_read,
        Arc::clone(&job),
        Arc::clone(&stdin),
        pseudoconsole.as_ref().map(Arc::clone),
        Arc::clone(&terminated),
    );

    let wait = unsafe { WaitForSingleObject(process.raw(), INFINITE) };
    if wait != WAIT_OBJECT_0 {
        unsafe {
            TerminateJobObject(job.raw(), 1);
        }
        return Err(last_os_error("WaitForSingleObject(restricted child)"));
    }
    stdin.lock().ok().and_then(|mut value| value.take());
    let preserve_descendants = !terminated.load(Ordering::Acquire);
    if preserve_descendants {
        preserve_job_descendants(&job)?;
    } else {
        unsafe {
            TerminateJobObject(job.raw(), 1);
        }
    }

    drop(pseudoconsole);
    let reader_count = if request.tty { 1 } else { 2 };
    for _ in 0..reader_count {
        let _ = reader_done_rx.recv_timeout(OUTPUT_DRAIN_TIMEOUT);
    }

    let mut exit_code = 1;
    if unsafe { GetExitCodeProcess(process.raw(), &mut exit_code) } == 0 {
        return Err(last_os_error("GetExitCodeProcess(restricted child)"));
    }
    drop(process_thread);
    drop(process);
    write_locked(
        &pipe_write,
        RunnerMessage::Exit {
            exit_code: i32::try_from(exit_code).unwrap_or(-1),
        },
    )?;

    if preserve_descendants {
        while active_job_processes(&job)? != 0 {
            thread::sleep(JOB_REAPER_POLL);
        }
    }
    Ok(())
}

fn spawn_output_reader(
    handle: OwnedHandle,
    kind: super::ExecutionOutputKind,
    pipe_write: Arc<Mutex<File>>,
    done: mpsc::Sender<()>,
) {
    thread::spawn(move || {
        let mut buffer = vec![0u8; super::PROCESS_OUTPUT_CHUNK_BYTES];
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
            if write_locked(
                &pipe_write,
                RunnerMessage::Output {
                    kind,
                    data_base64: encode_bytes(&buffer[..read as usize]),
                },
            )
            .is_err()
            {
                break;
            }
        }
        let _ = done.send(());
    });
}

fn spawn_control_reader(
    mut pipe_read: File,
    job: Arc<OwnedHandle>,
    stdin: Arc<Mutex<Option<OwnedHandle>>>,
    pseudoconsole: Option<Arc<RestrictedConpty>>,
    terminated: Arc<AtomicBool>,
) {
    thread::spawn(move || loop {
        match read_frame(&mut pipe_read) {
            Ok(Some(RunnerMessage::Stdin { data_base64 })) => {
                let Ok(bytes) = decode_bytes(&data_base64) else {
                    continue;
                };
                if let Ok(guard) = stdin.lock() {
                    if let Some(handle) = guard.as_ref() {
                        let _ = write_pipe(handle.raw(), &bytes);
                    }
                }
            }
            Ok(Some(RunnerMessage::CloseStdin)) => {
                if let Ok(mut guard) = stdin.lock() {
                    guard.take();
                }
            }
            Ok(Some(RunnerMessage::Resize { rows, cols })) => {
                if let Some(pseudoconsole) = pseudoconsole.as_ref() {
                    let _ = pseudoconsole.resize(rows, cols);
                }
            }
            Ok(Some(RunnerMessage::Terminate)) => {
                terminated.store(true, Ordering::Release);
                unsafe {
                    TerminateJobObject(job.raw(), 1);
                }
            }
            Ok(Some(_)) => {}
            Ok(None) | Err(_) => {
                terminated.store(true, Ordering::Release);
                unsafe {
                    TerminateJobObject(job.raw(), 1);
                }
                break;
            }
        }
    });
}

fn parse_pipe_arguments() -> io::Result<(String, String)> {
    let mut pipe_in = None;
    let mut pipe_out = None;
    for argument in std::env::args().skip(1) {
        if let Some(value) = argument.strip_prefix("--pipe-in=") {
            pipe_in = Some(value.to_string());
        } else if let Some(value) = argument.strip_prefix("--pipe-out=") {
            pipe_out = Some(value.to_string());
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unknown windows sandbox runner argument: {argument}"),
            ));
        }
    }
    match (pipe_in, pipe_out) {
        (Some(pipe_in), Some(pipe_out)) => Ok((pipe_in, pipe_out)),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "windows sandbox runner requires --pipe-in and --pipe-out",
        )),
    }
}

fn open_pipe(name: &str, access: u32) -> io::Result<OwnedHandle> {
    let name = to_wide(name);
    let handle = unsafe {
        CreateFileW(
            name.as_ptr(),
            access,
            0,
            ptr::null_mut(),
            OPEN_EXISTING,
            0,
            0,
        )
    };
    OwnedHandle::new(handle, "CreateFileW(windows sandbox runner pipe)")
}

fn verify_runner_identity(expected_sid: &str) -> io::Result<()> {
    let token = super::current_process_token_for_restriction()?;
    let actual_sid = unsafe { super::token_user_sid(token.raw())? };
    let expected_sid = LocalSid::parse(expected_sid)?;
    if unsafe { EqualSid(actual_sid.as_ptr() as *mut _, expected_sid.raw()) } == 0 {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "Windows sandbox runner identity does not match the requested account",
        ));
    }
    Ok(())
}

fn write_locked(writer: &Arc<Mutex<File>>, message: RunnerMessage) -> io::Result<()> {
    let mut writer = writer
        .lock()
        .map_err(|_| io::Error::other("Windows sandbox runner writer lock poisoned"))?;
    write_frame(&mut *writer, message)
}

fn send_start_error(writer: &Arc<Mutex<File>>, error: io::Error) -> io::Result<()> {
    let _ = write_locked(
        writer,
        RunnerMessage::Error {
            windows_error_code: error.raw_os_error().map(|value| value as u32),
            message: error.to_string(),
        },
    );
    Err(error)
}
