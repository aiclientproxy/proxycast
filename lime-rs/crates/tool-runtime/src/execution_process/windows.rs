use super::*;
use crate::sandbox::{parse_requested_sandbox_policy, RequestedSandboxPolicy};
use app_server_protocol::protocol::v2::GrantedPermissionProfile;
use std::ffi::c_void;
use std::os::windows::ffi::OsStrExt;
use std::path::Path;
use std::ptr;
use std::thread;
use windows_sys::Win32::Foundation::{
    CloseHandle, GetLastError, LocalFree, SetHandleInformation, ERROR_SUCCESS, HANDLE,
    HANDLE_FLAG_INHERIT, HLOCAL, INVALID_HANDLE_VALUE, LUID, WAIT_OBJECT_0, WAIT_TIMEOUT,
};
use windows_sys::Win32::Security::Authorization::{
    ConvertStringSidToSidW, SetEntriesInAclW, EXPLICIT_ACCESS_W, GRANT_ACCESS, TRUSTEE_IS_SID,
    TRUSTEE_IS_UNKNOWN, TRUSTEE_W,
};
use windows_sys::Win32::Security::{
    AdjustTokenPrivileges, CopySid, CreateRestrictedToken, CreateWellKnownSid, GetLengthSid,
    GetTokenInformation, LookupPrivilegeValueW, SetTokenInformation, TokenDefaultDacl, TokenGroups,
    ACL, SECURITY_ATTRIBUTES, SID_AND_ATTRIBUTES, TOKEN_ADJUST_DEFAULT, TOKEN_ADJUST_PRIVILEGES,
    TOKEN_ADJUST_SESSIONID, TOKEN_ASSIGN_PRIMARY, TOKEN_DUPLICATE, TOKEN_PRIVILEGES, TOKEN_QUERY,
};
use windows_sys::Win32::Storage::FileSystem::{ReadFile, WriteFile};
use windows_sys::Win32::System::JobObjects::{
    CreateJobObjectW, JobObjectBasicAccountingInformation, JobObjectExtendedLimitInformation,
    QueryInformationJobObject, SetInformationJobObject, TerminateJobObject,
    JOBOBJECT_BASIC_ACCOUNTING_INFORMATION, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOB_OBJECT_LIMIT_BREAKAWAY_OK, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};
use windows_sys::Win32::System::Pipes::CreatePipe;
use windows_sys::Win32::System::Threading::{
    CreateProcessAsUserW, GetCurrentProcess, GetExitCodeProcess, OpenProcessToken, ResumeThread,
    WaitForSingleObject, CREATE_NO_WINDOW, CREATE_SUSPENDED, CREATE_UNICODE_ENVIRONMENT,
    EXTENDED_STARTUPINFO_PRESENT, PROCESS_INFORMATION, STARTF_USESTDHANDLES, STARTUPINFOEXW,
};

#[path = "windows_acl.rs"]
mod windows_acl;
#[path = "windows_attr.rs"]
mod windows_attr;
use windows_acl::{build_acl_plan, AclLease};
use windows_attr::ProcessAttributeList;

const DISABLE_MAX_PRIVILEGE: u32 = 0x01;
const LUA_TOKEN: u32 = 0x04;
const WRITE_RESTRICTED: u32 = 0x08;
const GENERIC_ALL: u32 = 0x1000_0000;
const WIN_WORLD_SID: i32 = 1;
const SE_GROUP_LOGON_ID: u32 = 0xC000_0000;
const CONTROL_POLL_MILLIS: u32 = 25;
const JOB_REAPER_POLL_MILLIS: u64 = 250;

pub(super) fn start_windows_restricted_execution_process(
    mut request: LocalExecutionRequest,
    sandbox: LocalExecutionSandbox,
) -> io::Result<LocalExecutionProcessHandle> {
    if request.tty {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Windows restricted token sandbox does not support TTY sessions yet",
        ));
    }
    if request.command.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "local execution command must not be empty",
        ));
    }
    let cwd = request.cwd.clone().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Windows restricted token sandbox requires a working directory",
        )
    })?;
    let policy = parse_requested_sandbox_policy(sandbox.requested_policy.as_deref())
        .unwrap_or(RequestedSandboxPolicy::WorkspaceWrite);
    if policy == RequestedSandboxPolicy::DangerFullAccess {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "danger-full-access must bypass the restricted token sandbox",
        ));
    }

    apply_offline_environment(&mut request.env, sandbox.granted_permissions.as_ref());
    let acl_plan = build_acl_plan(&cwd, policy, sandbox.granted_permissions.as_ref())?;
    let capability_sid = capability_sid();
    let acl_lease = AclLease::acquire(&capability_sid, acl_plan)?;
    let token = create_restricted_token(&capability_sid)?;
    let spawned = spawn_restricted_process(&request, &cwd, token.raw())?;
    drop(token);

    let start = ExecutionProcessStart {
        process_id: request.process_id.clone(),
        tool_id: request.tool_id.clone(),
        tool_name: request.tool_name.clone(),
        command: Some(request.command.join(" ")),
        cwd: Some(cwd.to_string_lossy().to_string()),
    };
    let process_state = ExecutionProcess::start(start);
    let initial_snapshot = process_state.snapshot();
    let process = Arc::new(Mutex::new(process_state));
    let (output_tx, output_rx) = mpsc::unbounded_channel();
    let (control_tx, control_rx) = std::sync::mpsc::channel();
    let (state_tx, state_rx) = watch::channel(initial_snapshot);
    let (final_tx, final_rx) = oneshot::channel();

    let stdout_process = Arc::clone(&process);
    let stdout_state = state_tx.clone();
    let stdout_output = output_tx.clone();
    let stdout_reader = thread::spawn(move || {
        read_pipe_stream(
            spawned.stdout_read,
            ExecutionOutputKind::Stdout,
            stdout_process,
            stdout_output,
            stdout_state,
        )
    });
    let stderr_process = Arc::clone(&process);
    let stderr_state = state_tx.clone();
    let stderr_reader = thread::spawn(move || {
        read_pipe_stream(
            spawned.stderr_read,
            ExecutionOutputKind::Stderr,
            stderr_process,
            output_tx,
            stderr_state,
        )
    });
    thread::spawn(move || {
        supervise_restricted_process(
            spawned.process,
            spawned.thread,
            spawned.job,
            spawned.stdin_write,
            stdout_reader,
            stderr_reader,
            acl_lease,
            process,
            state_tx,
            final_tx,
            control_rx,
        )
    });

    Ok(LocalExecutionProcessHandle {
        process_id: request.process_id,
        control_tx: LocalExecutionControlSender::Blocking(control_tx),
        output_rx,
        state_rx,
        final_rx: Some(final_rx),
        final_snapshot: None,
    })
}

fn apply_offline_environment(
    env: &mut HashMap<String, String>,
    permissions: Option<&GrantedPermissionProfile>,
) {
    let network_enabled = permissions
        .and_then(|profile| profile.network.as_ref())
        .and_then(|network| network.enabled)
        .unwrap_or(false);
    if network_enabled {
        return;
    }
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "GIT_HTTP_PROXY",
        "GIT_HTTPS_PROXY",
    ] {
        env.entry(key.to_string())
            .or_insert_with(|| "http://127.0.0.1:9".to_string());
    }
    env.entry("NO_PROXY".to_string())
        .or_insert_with(|| "localhost,127.0.0.1,::1".to_string());
    env.insert("SBX_NONET_ACTIVE".to_string(), "1".to_string());
    env.entry("PIP_NO_INDEX".to_string())
        .or_insert_with(|| "1".to_string());
    env.entry("PIP_DISABLE_PIP_VERSION_CHECK".to_string())
        .or_insert_with(|| "1".to_string());
    env.entry("NPM_CONFIG_OFFLINE".to_string())
        .or_insert_with(|| "true".to_string());
    env.entry("CARGO_NET_OFFLINE".to_string())
        .or_insert_with(|| "true".to_string());
    env.entry("GIT_SSH_COMMAND".to_string())
        .or_insert_with(|| "cmd /c exit 1".to_string());
    env.entry("GIT_ALLOW_PROTOCOLS".to_string()).or_default();
}

fn capability_sid() -> String {
    let bytes = *uuid::Uuid::new_v4().as_bytes();
    let parts = [
        u32::from_le_bytes(bytes[0..4].try_into().expect("uuid segment")),
        u32::from_le_bytes(bytes[4..8].try_into().expect("uuid segment")),
        u32::from_le_bytes(bytes[8..12].try_into().expect("uuid segment")),
        u32::from_le_bytes(bytes[12..16].try_into().expect("uuid segment")),
    ];
    format!(
        "S-1-5-21-{}-{}-{}-{}",
        parts[0], parts[1], parts[2], parts[3]
    )
}

#[derive(Debug)]
struct OwnedHandle(HANDLE);

impl OwnedHandle {
    fn new(handle: HANDLE, context: &str) -> io::Result<Self> {
        if handle == 0 || handle == INVALID_HANDLE_VALUE {
            Err(last_os_error(context))
        } else {
            Ok(Self(handle))
        }
    }

    fn raw(&self) -> HANDLE {
        self.0
    }
}

impl Drop for OwnedHandle {
    fn drop(&mut self) {
        if self.0 != 0 && self.0 != INVALID_HANDLE_VALUE {
            unsafe {
                CloseHandle(self.0);
            }
            self.0 = 0;
        }
    }
}

struct LocalSid(*mut c_void);

impl LocalSid {
    fn parse(sid: &str) -> io::Result<Self> {
        let mut value = ptr::null_mut();
        let wide = to_wide(sid);
        if unsafe { ConvertStringSidToSidW(wide.as_ptr(), &mut value) } == 0 {
            return Err(last_os_error("ConvertStringSidToSidW"));
        }
        Ok(Self(value))
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

#[repr(C)]
struct TokenDefaultDaclInfo {
    default_dacl: *mut ACL,
}

fn create_restricted_token(capability_sid: &str) -> io::Result<OwnedHandle> {
    unsafe {
        let desired = TOKEN_DUPLICATE
            | TOKEN_QUERY
            | TOKEN_ASSIGN_PRIMARY
            | TOKEN_ADJUST_DEFAULT
            | TOKEN_ADJUST_SESSIONID
            | TOKEN_ADJUST_PRIVILEGES;
        let mut base = 0;
        if OpenProcessToken(GetCurrentProcess(), desired, &mut base) == 0 {
            return Err(last_os_error("OpenProcessToken"));
        }
        let base = OwnedHandle::new(base, "OpenProcessToken")?;
        let capability = LocalSid::parse(capability_sid)?;
        let mut logon = logon_sid(base.raw())?;
        let mut everyone = world_sid()?;
        let mut entries = [
            SID_AND_ATTRIBUTES {
                Sid: capability.raw(),
                Attributes: 0,
            },
            SID_AND_ATTRIBUTES {
                Sid: logon.as_mut_ptr() as *mut c_void,
                Attributes: 0,
            },
            SID_AND_ATTRIBUTES {
                Sid: everyone.as_mut_ptr() as *mut c_void,
                Attributes: 0,
            },
        ];
        let mut restricted = 0;
        if CreateRestrictedToken(
            base.raw(),
            DISABLE_MAX_PRIVILEGE | LUA_TOKEN | WRITE_RESTRICTED,
            0,
            ptr::null(),
            0,
            ptr::null(),
            entries.len() as u32,
            entries.as_mut_ptr(),
            &mut restricted,
        ) == 0
        {
            return Err(last_os_error("CreateRestrictedToken"));
        }
        let restricted = OwnedHandle::new(restricted, "CreateRestrictedToken")?;
        set_default_dacl(
            restricted.raw(),
            &[
                logon.as_mut_ptr() as *mut c_void,
                everyone.as_mut_ptr() as *mut c_void,
                capability.raw(),
            ],
        )?;
        enable_privilege(restricted.raw(), "SeChangeNotifyPrivilege")?;
        Ok(restricted)
    }
}

unsafe fn set_default_dacl(token: HANDLE, sids: &[*mut c_void]) -> io::Result<()> {
    let entries = sids
        .iter()
        .map(|sid| EXPLICIT_ACCESS_W {
            grfAccessPermissions: GENERIC_ALL,
            grfAccessMode: GRANT_ACCESS,
            grfInheritance: 0,
            Trustee: TRUSTEE_W {
                pMultipleTrustee: ptr::null_mut(),
                MultipleTrusteeOperation: 0,
                TrusteeForm: TRUSTEE_IS_SID,
                TrusteeType: TRUSTEE_IS_UNKNOWN,
                ptstrName: *sid as *mut u16,
            },
        })
        .collect::<Vec<_>>();
    let mut dacl = ptr::null_mut();
    let result = SetEntriesInAclW(
        entries.len() as u32,
        entries.as_ptr(),
        ptr::null_mut(),
        &mut dacl,
    );
    if result != 0 {
        return Err(io::Error::other(format!(
            "SetEntriesInAclW failed: {result}"
        )));
    }
    let mut info = TokenDefaultDaclInfo { default_dacl: dacl };
    let set_result = SetTokenInformation(
        token,
        TokenDefaultDacl,
        &mut info as *mut _ as *mut c_void,
        std::mem::size_of::<TokenDefaultDaclInfo>() as u32,
    );
    LocalFree(dacl as HLOCAL);
    if set_result == 0 {
        return Err(last_os_error("SetTokenInformation(TokenDefaultDacl)"));
    }
    Ok(())
}

unsafe fn enable_privilege(token: HANDLE, name: &str) -> io::Result<()> {
    let mut luid = LUID {
        LowPart: 0,
        HighPart: 0,
    };
    if LookupPrivilegeValueW(ptr::null(), to_wide(name).as_ptr(), &mut luid) == 0 {
        return Err(last_os_error("LookupPrivilegeValueW"));
    }
    let mut privileges: TOKEN_PRIVILEGES = std::mem::zeroed();
    privileges.PrivilegeCount = 1;
    privileges.Privileges[0].Luid = luid;
    privileges.Privileges[0].Attributes = 0x0000_0002;
    if AdjustTokenPrivileges(token, 0, &privileges, 0, ptr::null_mut(), ptr::null_mut()) == 0 {
        return Err(last_os_error("AdjustTokenPrivileges"));
    }
    let error = GetLastError();
    if error != ERROR_SUCCESS {
        return Err(io::Error::from_raw_os_error(error as i32));
    }
    Ok(())
}

unsafe fn scan_logon_sid(token: HANDLE) -> Option<Vec<u8>> {
    let mut needed = 0;
    GetTokenInformation(token, TokenGroups, ptr::null_mut(), 0, &mut needed);
    if needed == 0 {
        return None;
    }
    let mut buffer = vec![0u8; needed as usize];
    if GetTokenInformation(
        token,
        TokenGroups,
        buffer.as_mut_ptr() as *mut c_void,
        needed,
        &mut needed,
    ) == 0
    {
        return None;
    }
    let count = ptr::read_unaligned(buffer.as_ptr() as *const u32) as usize;
    let after_count = buffer.as_ptr().add(std::mem::size_of::<u32>()) as usize;
    let align = std::mem::align_of::<SID_AND_ATTRIBUTES>();
    let groups = ((after_count + align - 1) & !(align - 1)) as *const SID_AND_ATTRIBUTES;
    for index in 0..count {
        let entry = ptr::read_unaligned(groups.add(index));
        if entry.Attributes & SE_GROUP_LOGON_ID == SE_GROUP_LOGON_ID {
            let length = GetLengthSid(entry.Sid);
            if length == 0 {
                return None;
            }
            let mut sid = vec![0u8; length as usize];
            if CopySid(length, sid.as_mut_ptr() as *mut c_void, entry.Sid) == 0 {
                return None;
            }
            return Some(sid);
        }
    }
    None
}

unsafe fn logon_sid(token: HANDLE) -> io::Result<Vec<u8>> {
    if let Some(sid) = scan_logon_sid(token) {
        return Ok(sid);
    }

    // UAC-filtered tokens can expose the logon SID only through their linked token.
    const TOKEN_LINKED_TOKEN: i32 = 19;
    #[repr(C)]
    struct LinkedToken {
        token: HANDLE,
    }
    let mut needed = 0;
    GetTokenInformation(token, TOKEN_LINKED_TOKEN, ptr::null_mut(), 0, &mut needed);
    if needed >= std::mem::size_of::<LinkedToken>() as u32 {
        let mut buffer = vec![0u8; needed as usize];
        if GetTokenInformation(
            token,
            TOKEN_LINKED_TOKEN,
            buffer.as_mut_ptr() as *mut c_void,
            needed,
            &mut needed,
        ) != 0
        {
            let linked = ptr::read_unaligned(buffer.as_ptr() as *const LinkedToken).token;
            if linked != 0 && linked != INVALID_HANDLE_VALUE {
                let result = scan_logon_sid(linked);
                CloseHandle(linked);
                if let Some(sid) = result {
                    return Ok(sid);
                }
            }
        }
    }
    Err(io::Error::other("current process token has no logon SID"))
}

unsafe fn world_sid() -> io::Result<Vec<u8>> {
    let mut size = 0;
    CreateWellKnownSid(WIN_WORLD_SID, ptr::null_mut(), ptr::null_mut(), &mut size);
    let mut sid = vec![0u8; size as usize];
    if CreateWellKnownSid(
        WIN_WORLD_SID,
        ptr::null_mut(),
        sid.as_mut_ptr() as *mut c_void,
        &mut size,
    ) == 0
    {
        return Err(last_os_error("CreateWellKnownSid(World)"));
    }
    Ok(sid)
}

struct SpawnedRestrictedProcess {
    process: OwnedHandle,
    thread: OwnedHandle,
    job: OwnedHandle,
    stdin_write: Option<OwnedHandle>,
    stdout_read: OwnedHandle,
    stderr_read: OwnedHandle,
}

fn spawn_restricted_process(
    request: &LocalExecutionRequest,
    cwd: &Path,
    token: HANDLE,
) -> io::Result<SpawnedRestrictedProcess> {
    let (stdin_read, stdin_write) = create_pipe_pair(true)?;
    let (stdout_read, stdout_write) = create_pipe_pair(false)?;
    let (stderr_read, stderr_write) = create_pipe_pair(false)?;
    let job = create_kill_on_close_job()?;
    let mut command_line = to_wide(argv_to_command_line(&request.command));
    let mut env_block = environment_block(&request.env)?;
    let cwd = to_wide(cwd.as_os_str());
    let mut desktop = to_wide("winsta0\\default");
    let mut attributes = ProcessAttributeList::new(2)?;
    attributes.set_job(job.raw())?;
    attributes.set_handle_list(&[stdin_read.raw(), stdout_write.raw(), stderr_write.raw()])?;
    let mut startup: STARTUPINFOEXW = unsafe { std::mem::zeroed() };
    startup.StartupInfo.cb = std::mem::size_of::<STARTUPINFOEXW>() as u32;
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = stdin_read.raw();
    startup.StartupInfo.hStdOutput = stdout_write.raw();
    startup.StartupInfo.hStdError = stderr_write.raw();
    startup.StartupInfo.lpDesktop = desktop.as_mut_ptr();
    startup.lpAttributeList = attributes.as_mut_ptr();
    let mut process_info: PROCESS_INFORMATION = unsafe { std::mem::zeroed() };
    let created = unsafe {
        CreateProcessAsUserW(
            token,
            ptr::null(),
            command_line.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut(),
            1,
            CREATE_UNICODE_ENVIRONMENT
                | CREATE_SUSPENDED
                | CREATE_NO_WINDOW
                | EXTENDED_STARTUPINFO_PRESENT,
            env_block.as_mut_ptr() as *mut c_void,
            cwd.as_ptr(),
            &startup.StartupInfo,
            &mut process_info,
        )
    };
    if created == 0 {
        return Err(last_os_error("CreateProcessAsUserW"));
    }
    let process = OwnedHandle::new(process_info.hProcess, "CreateProcessAsUserW process")?;
    let thread = OwnedHandle::new(process_info.hThread, "CreateProcessAsUserW thread")?;
    if unsafe { ResumeThread(thread.raw()) } == u32::MAX {
        unsafe {
            TerminateJobObject(job.raw(), 1);
        }
        return Err(last_os_error("ResumeThread"));
    }
    drop(stdin_read);
    drop(stdout_write);
    drop(stderr_write);
    let stdin_write = request.stdin.then_some(stdin_write);
    Ok(SpawnedRestrictedProcess {
        process,
        thread,
        job,
        stdin_write,
        stdout_read,
        stderr_read,
    })
}

fn create_pipe_pair(parent_writes: bool) -> io::Result<(OwnedHandle, OwnedHandle)> {
    let mut attributes = SECURITY_ATTRIBUTES {
        nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
        lpSecurityDescriptor: ptr::null_mut(),
        bInheritHandle: 1,
    };
    let mut read = 0;
    let mut write = 0;
    if unsafe { CreatePipe(&mut read, &mut write, &mut attributes, 0) } == 0 {
        return Err(last_os_error("CreatePipe"));
    }
    let read = OwnedHandle::new(read, "CreatePipe read")?;
    let write = OwnedHandle::new(write, "CreatePipe write")?;
    let parent = if parent_writes {
        write.raw()
    } else {
        read.raw()
    };
    if unsafe { SetHandleInformation(parent, HANDLE_FLAG_INHERIT, 0) } == 0 {
        return Err(last_os_error("SetHandleInformation"));
    }
    Ok((read, write))
}

fn create_kill_on_close_job() -> io::Result<OwnedHandle> {
    let job = OwnedHandle::new(
        unsafe { CreateJobObjectW(ptr::null(), ptr::null()) },
        "CreateJobObjectW",
    )?;
    set_job_limit_flags(
        job.raw(),
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_BREAKAWAY_OK,
    )?;
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

fn preserve_job_descendants(job: &OwnedHandle) -> io::Result<()> {
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
fn supervise_restricted_process(
    process_handle: OwnedHandle,
    thread_handle: OwnedHandle,
    job: OwnedHandle,
    mut stdin: Option<OwnedHandle>,
    stdout_reader: thread::JoinHandle<()>,
    stderr_reader: thread::JoinHandle<()>,
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
        drain_controls(&job, &mut stdin, &process, &state_tx, &control_rx);
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
    let _ = stdout_reader.join();
    let _ = stderr_reader.join();
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
            Ok(LocalExecutionControl::Resize { .. }) => {}
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

fn read_pipe_stream(
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

fn environment_block(env: &HashMap<String, String>) -> io::Result<Vec<u16>> {
    let mut entries = env.iter().collect::<Vec<_>>();
    entries.sort_by(|(left, _), (right, _)| {
        left.to_uppercase()
            .cmp(&right.to_uppercase())
            .then(left.cmp(right))
    });
    let mut block = Vec::new();
    for (key, value) in entries {
        if key.is_empty()
            || key.contains('\0')
            || (key.contains('=') && !key.starts_with('='))
            || value.contains('\0')
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid Windows environment variable: {key}"),
            ));
        }
        block.extend(format!("{key}={value}").encode_utf16());
        block.push(0);
    }
    block.push(0);
    if block.len() == 1 {
        block.push(0);
    }
    Ok(block)
}

fn argv_to_command_line(argv: &[String]) -> String {
    argv.iter()
        .map(|arg| quote_windows_arg(arg))
        .collect::<Vec<_>>()
        .join(" ")
}

fn quote_windows_arg(argument: &str) -> String {
    let needs_quotes = argument.is_empty()
        || argument
            .chars()
            .any(|character| matches!(character, ' ' | '\t' | '\n' | '\r' | '"'));
    if !needs_quotes {
        return argument.to_string();
    }
    let mut quoted = String::with_capacity(argument.len() + 2);
    quoted.push('"');
    let mut backslashes = 0;
    for character in argument.chars() {
        match character {
            '\\' => backslashes += 1,
            '"' => {
                quoted.push_str(&"\\".repeat(backslashes * 2 + 1));
                quoted.push('"');
                backslashes = 0;
            }
            _ => {
                quoted.push_str(&"\\".repeat(backslashes));
                backslashes = 0;
                quoted.push(character);
            }
        }
    }
    quoted.push_str(&"\\".repeat(backslashes * 2));
    quoted.push('"');
    quoted
}

fn to_wide(value: impl AsRef<std::ffi::OsStr>) -> Vec<u16> {
    value
        .as_ref()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect()
}

fn last_os_error(context: &str) -> io::Error {
    let code = unsafe { GetLastError() } as i32;
    io::Error::new(
        io::ErrorKind::Other,
        format!("{context} failed: {}", io::Error::from_raw_os_error(code)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_line_quoting_preserves_spaces_quotes_and_trailing_slashes() {
        assert_eq!(quote_windows_arg("plain"), "plain");
        assert_eq!(quote_windows_arg("two words"), "\"two words\"");
        assert_eq!(quote_windows_arg("a\\\"b"), "\"a\\\\\\\"b\"");
        assert_eq!(
            quote_windows_arg("C:\\Program Files\\"),
            "\"C:\\Program Files\\\\\""
        );
    }

    #[test]
    fn offline_environment_does_not_override_explicit_proxy() {
        let mut env = HashMap::from([("HTTP_PROXY".to_string(), "http://proxy:8080".to_string())]);
        apply_offline_environment(&mut env, None);

        assert_eq!(
            env.get("HTTP_PROXY").map(String::as_str),
            Some("http://proxy:8080")
        );
        assert_eq!(
            env.get("HTTPS_PROXY").map(String::as_str),
            Some("http://127.0.0.1:9")
        );
        assert_eq!(env.get("SBX_NONET_ACTIVE").map(String::as_str), Some("1"));
        assert_eq!(env.get("GIT_ALLOW_PROTOCOLS").map(String::as_str), Some(""));
    }

    #[test]
    fn environment_block_is_sorted_and_double_null_terminated() {
        let block = environment_block(&HashMap::from([
            ("Z_VALUE".to_string(), "last".to_string()),
            ("A_VALUE".to_string(), "first".to_string()),
        ]))
        .expect("environment block");
        let expected = "A_VALUE=first\0Z_VALUE=last\0\0"
            .encode_utf16()
            .collect::<Vec<_>>();

        assert_eq!(block, expected);
        assert_eq!(environment_block(&HashMap::new()).unwrap(), vec![0, 0]);
    }

    #[test]
    fn environment_block_rejects_embedded_nulls() {
        let error = environment_block(&HashMap::from([(
            "VALUE".to_string(),
            "before\0after".to_string(),
        )]))
        .expect_err("embedded null must fail closed");

        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }
}
