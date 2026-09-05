use super::windows_runner_protocol::{read_frame, write_frame, RunnerMessage, RunnerSpawnRequest};
use super::{argv_to_command_line, last_os_error, to_wide, LocalExecutionRequest, OwnedHandle};
use crate::windows_setup::{read_windows_sandbox_password, resolve_windows_account_sid};
use std::ffi::c_void;
use std::fs::File;
use std::io;
use std::os::windows::io::FromRawHandle;
use std::path::{Path, PathBuf};
use std::ptr;
use std::thread;
use std::time::{Duration, Instant};
use windows_sys::Win32::Foundation::{
    GetLastError, LocalFree, ERROR_PIPE_CONNECTED, ERROR_PIPE_LISTENING, HANDLE, HLOCAL,
    WAIT_OBJECT_0, WAIT_TIMEOUT,
};
use windows_sys::Win32::Security::Authorization::{
    ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW,
};
use windows_sys::Win32::Security::{PSECURITY_DESCRIPTOR, SECURITY_ATTRIBUTES};
use windows_sys::Win32::System::Diagnostics::Debug::SetErrorMode;
use windows_sys::Win32::System::Pipes::{
    ConnectNamedPipe, CreateNamedPipeW, GetNamedPipeClientProcessId, PeekNamedPipe,
    SetNamedPipeHandleState, PIPE_NOWAIT, PIPE_READMODE_BYTE, PIPE_REJECT_REMOTE_CLIENTS,
    PIPE_TYPE_BYTE, PIPE_WAIT,
};
use windows_sys::Win32::System::Threading::{
    CreateProcessW, CreateProcessWithLogonW, TerminateProcess, WaitForSingleObject,
    CREATE_NO_WINDOW, CREATE_UNICODE_ENVIRONMENT, PROCESS_INFORMATION, STARTUPINFOW,
};

const PIPE_ACCESS_OUTBOUND: u32 = 0x0000_0002;
const PIPE_ACCESS_DUPLEX: u32 = 0x0000_0003;
const RUNNER_START_TIMEOUT: Duration = Duration::from_secs(15);
const RUNNER_POLL_INTERVAL: Duration = Duration::from_millis(5);
const RUNNER_ERROR_MODE_FLAGS: u32 = 0x0001 | 0x0002;

pub(super) struct RunnerTransport {
    pub(super) pipe_write: File,
    pub(super) pipe_read: File,
    pub(super) process: OwnedHandle,
}

pub(super) fn spawn_runner_transport(
    request: &LocalExecutionRequest,
    cwd: &Path,
    account: &str,
    capability_sid: String,
) -> io::Result<RunnerTransport> {
    let account_sid = resolve_windows_account_sid(account).map_err(io::Error::other)?;
    let account_sid = sid_string(&account_sid)?;
    spawn_runner_transport_inner(request, cwd, account_sid, capability_sid, Some(account))
}

pub(super) fn spawn_current_user_runner_transport(
    request: &LocalExecutionRequest,
    cwd: &Path,
    capability_sid: String,
) -> io::Result<RunnerTransport> {
    let token = super::current_process_token_for_restriction()?;
    let account_sid = unsafe { super::token_user_sid(token.raw())? };
    let account_sid = sid_string(&account_sid)?;
    spawn_runner_transport_inner(request, cwd, account_sid, capability_sid, None)
}

fn spawn_runner_transport_inner(
    request: &LocalExecutionRequest,
    cwd: &Path,
    account_sid: String,
    capability_sid: String,
    account: Option<&str>,
) -> io::Result<RunnerTransport> {
    let nonce = uuid::Uuid::new_v4().simple();
    let pipe_in_name = format!(r"\\.\pipe\lime-sandbox-runner-{nonce}-in");
    let pipe_out_name = format!(r"\\.\pipe\lime-sandbox-runner-{nonce}-out");
    let pipe_in = create_named_pipe(&pipe_in_name, PIPE_ACCESS_OUTBOUND, &account_sid)?;
    // The host reads this pipe, but needs GENERIC_WRITE to switch the bounded
    // nonblocking connection handle back to PIPE_WAIT after the runner connects.
    let pipe_out = create_named_pipe(&pipe_out_name, PIPE_ACCESS_DUPLEX, &account_sid)?;

    let runner = resolve_runner_executable()?;
    let runner_args = vec![
        runner.to_string_lossy().into_owned(),
        format!("--pipe-in={pipe_in_name}"),
        format!("--pipe-out={pipe_out_name}"),
    ];
    let mut command_line = to_wide(argv_to_command_line(&runner_args));
    let runner_wide = to_wide(runner.as_os_str());
    let cwd_wide = to_wide(cwd.as_os_str());
    let mut startup: STARTUPINFOW = unsafe { std::mem::zeroed() };
    startup.cb = std::mem::size_of::<STARTUPINFOW>() as u32;
    let mut process_info: PROCESS_INFORMATION = unsafe { std::mem::zeroed() };
    let previous_error_mode = unsafe { SetErrorMode(RUNNER_ERROR_MODE_FLAGS) };
    let created = if let Some(account) = account {
        let username = to_wide(account);
        let domain = to_wide(".");
        let password = to_wide(read_windows_sandbox_password(account)?);
        unsafe {
            CreateProcessWithLogonW(
                username.as_ptr(),
                domain.as_ptr(),
                password.as_ptr(),
                0,
                runner_wide.as_ptr(),
                command_line.as_mut_ptr(),
                CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT,
                ptr::null(),
                cwd_wide.as_ptr(),
                &startup,
                &mut process_info,
            )
        }
    } else {
        unsafe {
            CreateProcessW(
                ptr::null(),
                command_line.as_mut_ptr(),
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT,
                ptr::null(),
                cwd_wide.as_ptr(),
                &startup,
                &mut process_info,
            )
        }
    };
    unsafe {
        SetErrorMode(previous_error_mode);
    }
    if created == 0 {
        return Err(last_os_error(if account.is_some() {
            "CreateProcessWithLogonW(windows-sandbox-runner)"
        } else {
            "CreateProcessW(windows-sandbox-runner)"
        }));
    }
    let process = OwnedHandle::new(process_info.hProcess, "windows sandbox runner process")?;
    drop(OwnedHandle::new(
        process_info.hThread,
        "windows sandbox runner thread",
    )?);

    if let Err(error) = connect_named_pipe(&pipe_in, process.raw(), process_info.dwProcessId)
        .and_then(|_| connect_named_pipe(&pipe_out, process.raw(), process_info.dwProcessId))
    {
        unsafe {
            TerminateProcess(process.raw(), 1);
        }
        return Err(error);
    }

    let mut transport = RunnerTransport {
        pipe_write: unsafe { File::from_raw_handle(pipe_in.into_raw() as _) },
        pipe_read: unsafe { File::from_raw_handle(pipe_out.into_raw() as _) },
        process,
    };
    let startup_result = (|| {
        write_frame(
            &mut transport.pipe_write,
            RunnerMessage::Spawn {
                payload: RunnerSpawnRequest {
                    command: request.command.clone(),
                    cwd: cwd.to_path_buf(),
                    env: request.env.clone(),
                    capability_sid,
                    expected_account_sid: account_sid,
                    tty: request.tty,
                    stdin_open: request.stdin,
                    pty_size: request.pty_size,
                },
            },
        )?;
        wait_for_complete_frame(
            &transport.pipe_read,
            transport.process.raw(),
            RUNNER_START_TIMEOUT,
        )?;
        match read_frame(&mut transport.pipe_read)? {
            Some(RunnerMessage::Ready { .. }) => Ok(()),
            Some(RunnerMessage::Error {
                message,
                windows_error_code,
            }) => Err(io::Error::other(format_runner_error(
                &message,
                windows_error_code,
            ))),
            Some(_) => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Windows sandbox runner sent an unexpected startup frame",
            )),
            None => Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Windows sandbox runner closed before ready",
            )),
        }
    })();
    if let Err(error) = startup_result {
        unsafe {
            TerminateProcess(transport.process.raw(), 1);
        }
        return Err(error);
    }
    Ok(transport)
}

fn resolve_runner_executable() -> io::Result<PathBuf> {
    let current = std::env::current_exe()?;
    let name = "windows-sandbox-runner.exe";
    let mut candidates = Vec::new();
    if let Some(parent) = current.parent() {
        candidates.push(parent.join(name));
        if parent.file_name().is_some_and(|value| value == "deps") {
            if let Some(target_profile) = parent.parent() {
                candidates.push(target_profile.join(name));
            }
        }
    }
    candidates
        .into_iter()
        .find(|candidate| candidate.is_file())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "windows-sandbox-runner.exe was not found beside {}",
                    current.display()
                ),
            )
        })
}

fn create_named_pipe(name: &str, access: u32, account_sid: &str) -> io::Result<OwnedHandle> {
    let sddl = to_wide(format!("D:(A;;GA;;;{account_sid})"));
    let mut descriptor: PSECURITY_DESCRIPTOR = ptr::null_mut();
    let mut descriptor_size = 0;
    if unsafe {
        ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl.as_ptr(),
            1,
            &mut descriptor,
            &mut descriptor_size,
        )
    } == 0
    {
        return Err(last_os_error(
            "ConvertStringSecurityDescriptorToSecurityDescriptorW(runner pipe)",
        ));
    }
    let mut attributes = SECURITY_ATTRIBUTES {
        nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
        lpSecurityDescriptor: descriptor,
        bInheritHandle: 0,
    };
    let name = to_wide(name);
    let handle = unsafe {
        CreateNamedPipeW(
            name.as_ptr(),
            access,
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_NOWAIT | PIPE_REJECT_REMOTE_CLIENTS,
            1,
            64 * 1024,
            64 * 1024,
            0,
            &mut attributes,
        )
    };
    unsafe {
        LocalFree(descriptor as HLOCAL);
    }
    OwnedHandle::new(handle, "CreateNamedPipeW(windows sandbox runner)")
}

fn connect_named_pipe(pipe: &OwnedHandle, process: HANDLE, expected_pid: u32) -> io::Result<()> {
    let deadline = Instant::now() + RUNNER_START_TIMEOUT;
    loop {
        let connected = unsafe { ConnectNamedPipe(pipe.raw(), ptr::null_mut()) };
        let error = if connected == 0 {
            unsafe { GetLastError() }
        } else {
            0
        };
        if connected != 0 || error == ERROR_PIPE_CONNECTED {
            break;
        }
        if error != ERROR_PIPE_LISTENING {
            return Err(io::Error::from_raw_os_error(error as i32));
        }
        if unsafe { WaitForSingleObject(process, 0) } == WAIT_OBJECT_0 {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "Windows sandbox runner exited before connecting",
            ));
        }
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "timed out connecting Windows sandbox runner pipe",
            ));
        }
        thread::sleep(RUNNER_POLL_INTERVAL);
    }
    let wait_mode = PIPE_READMODE_BYTE | PIPE_WAIT;
    if unsafe { SetNamedPipeHandleState(pipe.raw(), &wait_mode, ptr::null(), ptr::null()) } == 0 {
        return Err(last_os_error("SetNamedPipeHandleState(runner pipe)"));
    }
    let mut client_pid = 0;
    if unsafe { GetNamedPipeClientProcessId(pipe.raw(), &mut client_pid) } == 0 {
        return Err(last_os_error("GetNamedPipeClientProcessId"));
    }
    if client_pid != expected_pid {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!("runner pipe client pid {client_pid} did not match {expected_pid}"),
        ));
    }
    Ok(())
}

fn wait_for_complete_frame(pipe: &File, process: HANDLE, timeout: Duration) -> io::Result<()> {
    use std::os::windows::io::AsRawHandle;

    let handle = pipe.as_raw_handle() as HANDLE;
    let deadline = Instant::now() + timeout;
    let mut length = [0u8; 4];
    loop {
        let mut bytes_read = 0;
        let mut available = 0;
        if unsafe {
            PeekNamedPipe(
                handle,
                length.as_mut_ptr().cast::<c_void>(),
                length.len() as u32,
                &mut bytes_read,
                &mut available,
                ptr::null_mut(),
            )
        } == 0
        {
            return Err(last_os_error("PeekNamedPipe(runner ready)"));
        }
        if bytes_read == length.len() as u32 {
            let frame_bytes = u32::from_le_bytes(length) as usize + length.len();
            if available as usize >= frame_bytes {
                return Ok(());
            }
        }
        match unsafe { WaitForSingleObject(process, 0) } {
            WAIT_OBJECT_0 => {
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "Windows sandbox runner exited before ready",
                ))
            }
            WAIT_TIMEOUT => {}
            _ => return Err(last_os_error("WaitForSingleObject(runner ready)")),
        }
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "timed out waiting for Windows sandbox runner ready frame",
            ));
        }
        thread::sleep(RUNNER_POLL_INTERVAL);
    }
}

fn sid_string(sid: &[u8]) -> io::Result<String> {
    let mut value = ptr::null_mut();
    if unsafe { ConvertSidToStringSidW(sid.as_ptr() as *mut c_void, &mut value) } == 0 {
        return Err(last_os_error("ConvertSidToStringSidW(runner account)"));
    }
    let mut length = 0;
    while unsafe { *value.add(length) } != 0 {
        length += 1;
    }
    let result = String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(value, length) });
    unsafe {
        LocalFree(value as HLOCAL);
    }
    Ok(result)
}

pub(super) fn format_runner_error(message: &str, code: Option<u32>) -> String {
    match code {
        Some(code) => format!("{message} (Windows error {code})"),
        None => message.to_string(),
    }
}
