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
    HANDLE_FLAG_INHERIT, HLOCAL, INVALID_HANDLE_VALUE, LUID,
};
use windows_sys::Win32::Security::Authorization::{
    ConvertStringSidToSidW, SetEntriesInAclW, EXPLICIT_ACCESS_W, GRANT_ACCESS, TRUSTEE_IS_SID,
    TRUSTEE_IS_UNKNOWN, TRUSTEE_W,
};
use windows_sys::Win32::Security::{
    AdjustTokenPrivileges, CopySid, CreateRestrictedToken, CreateWellKnownSid, GetLengthSid,
    GetTokenInformation, LookupPrivilegeValueW, SetTokenInformation, TokenDefaultDacl, TokenGroups,
    TokenUser, ACL, SECURITY_ATTRIBUTES, SID_AND_ATTRIBUTES, TOKEN_ADJUST_DEFAULT,
    TOKEN_ADJUST_PRIVILEGES, TOKEN_ADJUST_SESSIONID, TOKEN_ASSIGN_PRIMARY, TOKEN_DUPLICATE,
    TOKEN_PRIVILEGES, TOKEN_QUERY, TOKEN_USER,
};
use windows_sys::Win32::Storage::FileSystem::WriteFile;
use windows_sys::Win32::System::JobObjects::{
    CreateJobObjectW, JobObjectBasicAccountingInformation, JobObjectExtendedLimitInformation,
    QueryInformationJobObject, SetInformationJobObject, JOBOBJECT_BASIC_ACCOUNTING_INFORMATION,
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_BREAKAWAY_OK,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};
use windows_sys::Win32::System::Pipes::CreatePipe;
use windows_sys::Win32::System::Threading::{
    CreateProcessAsUserW, GetCurrentProcess, OpenProcessToken, CREATE_UNICODE_ENVIRONMENT,
    EXTENDED_STARTUPINFO_PRESENT, PROCESS_INFORMATION, STARTF_USESTDHANDLES, STARTUPINFOEXW,
};

#[path = "windows_acl.rs"]
mod windows_acl;
#[path = "windows_attr.rs"]
mod windows_attr;
#[path = "windows_audit.rs"]
mod windows_audit;
#[path = "windows_conpty.rs"]
mod windows_conpty;
#[path = "windows_job.rs"]
mod windows_job;
#[path = "windows_null.rs"]
mod windows_null;
#[path = "windows_runner_child.rs"]
mod windows_runner_child;
#[path = "windows_runner_host.rs"]
mod windows_runner_host;
#[path = "windows_runner_protocol.rs"]
mod windows_runner_protocol;
#[path = "windows_runner_supervisor.rs"]
mod windows_runner_supervisor;
use crate::windows_setup::{
    verify_windows_sandbox_group_membership, windows_sandbox_users_group_sid,
    WINDOWS_SANDBOX_OFFLINE_USERNAME, WINDOWS_SANDBOX_ONLINE_USERNAME,
};
use windows_acl::{build_acl_plan, AclLease};
use windows_attr::ProcessAttributeList;
pub(crate) use windows_audit::audit_world_writable;
use windows_conpty::RestrictedConpty;
use windows_job::create_kill_on_close_job;
#[cfg(test)]
use windows_job::preserve_job_descendants;
use windows_null::NullDeviceLease;

const DISABLE_MAX_PRIVILEGE: u32 = 0x01;
const LUA_TOKEN: u32 = 0x04;
const WRITE_RESTRICTED: u32 = 0x08;
const GENERIC_ALL: u32 = 0x1000_0000;
const WIN_WORLD_SID: i32 = 1;
const SE_GROUP_LOGON_ID: u32 = 0xC000_0000;
const CONTROL_POLL_MILLIS: u32 = 25;

pub(super) fn run_windows_sandbox_runner() -> io::Result<()> {
    windows_runner_child::run()
}
pub(super) fn start_windows_restricted_execution_process(
    mut request: LocalExecutionRequest,
    sandbox: LocalExecutionSandbox,
) -> io::Result<LocalExecutionProcessHandle> {
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

    let sandbox_account = sandbox_account_for_permissions(sandbox.granted_permissions.as_ref());
    verify_sandbox_account_network_policy(sandbox_account)?;
    verify_windows_sandbox_group_membership(sandbox_account)?;
    if request.tty {
        request
            .env
            .entry("TERM".to_string())
            .or_insert_with(|| "xterm-256color".to_string());
    }
    let acl_plan = build_acl_plan(
        &cwd,
        policy,
        sandbox.granted_permissions.as_ref(),
        &request.env,
    )?;
    let sandbox_group_sid = windows_sandbox_users_group_sid()?;
    let capability_sid = capability_sid();
    let acl_lease = AclLease::acquire(&sandbox_group_sid, &capability_sid, acl_plan)?;
    let null_device_lease = NullDeviceLease::acquire(&capability_sid)?;
    let transport = windows_runner_host::spawn_runner_transport(
        &request,
        &cwd,
        sandbox_account,
        capability_sid,
    )?;

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
    thread::spawn(move || {
        windows_runner_supervisor::supervise(
            transport,
            acl_lease,
            null_device_lease,
            process,
            output_tx,
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

fn sandbox_account_for_permissions(permissions: Option<&GrantedPermissionProfile>) -> &'static str {
    let network_enabled = permissions
        .and_then(|profile| profile.network.as_ref())
        .and_then(|network| network.enabled)
        .unwrap_or(false);
    if network_enabled {
        WINDOWS_SANDBOX_ONLINE_USERNAME
    } else {
        WINDOWS_SANDBOX_OFFLINE_USERNAME
    }
}

fn verify_sandbox_account_network_policy(account: &str) -> io::Result<()> {
    if account != WINDOWS_SANDBOX_OFFLINE_USERNAME {
        return Ok(());
    }
    crate::windows_firewall::verify_offline_rules(account)?;
    crate::windows_wfp::verify_filters(account)
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

    fn into_raw(mut self) -> HANDLE {
        let handle = self.0;
        self.0 = 0;
        handle
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

fn current_process_token_for_restriction() -> io::Result<OwnedHandle> {
    let desired = TOKEN_DUPLICATE
        | TOKEN_QUERY
        | TOKEN_ASSIGN_PRIMARY
        | TOKEN_ADJUST_DEFAULT
        | TOKEN_ADJUST_SESSIONID
        | TOKEN_ADJUST_PRIVILEGES;
    let mut token = 0;
    if unsafe { OpenProcessToken(GetCurrentProcess(), desired, &mut token) } == 0 {
        return Err(last_os_error("OpenProcessToken(runner)"));
    }
    OwnedHandle::new(token, "OpenProcessToken(runner)")
}

fn create_restricted_token(capability_sid: &str) -> io::Result<OwnedHandle> {
    unsafe {
        let base = current_process_token_for_restriction()?;
        let capability = LocalSid::parse(capability_sid)?;
        let mut user = token_user_sid(base.raw())?;
        let mut logon = logon_sid(base.raw())?;
        let mut everyone = world_sid()?;
        let mut entries = [
            SID_AND_ATTRIBUTES {
                Sid: capability.raw(),
                Attributes: 0,
            },
            SID_AND_ATTRIBUTES {
                Sid: user.as_mut_ptr() as *mut c_void,
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
        // The account SID constrains the restricted access check but is not a
        // capability, so it must not grant access through the default DACL.
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

unsafe fn token_user_sid(token: HANDLE) -> io::Result<Vec<u8>> {
    let mut needed = 0;
    GetTokenInformation(token, TokenUser, ptr::null_mut(), 0, &mut needed);
    if needed < std::mem::size_of::<TOKEN_USER>() as u32 {
        return Err(last_os_error("GetTokenInformation(TokenUser) size"));
    }
    let mut buffer = vec![0u8; needed as usize];
    if GetTokenInformation(
        token,
        TokenUser,
        buffer.as_mut_ptr() as *mut c_void,
        needed,
        &mut needed,
    ) == 0
    {
        return Err(last_os_error("GetTokenInformation(TokenUser)"));
    }
    let user = ptr::read_unaligned(buffer.as_ptr() as *const TOKEN_USER);
    let length = GetLengthSid(user.User.Sid);
    if length == 0 {
        return Err(last_os_error("GetLengthSid(TokenUser)"));
    }
    let mut sid = vec![0u8; length as usize];
    if CopySid(length, sid.as_mut_ptr() as *mut c_void, user.User.Sid) == 0 {
        return Err(last_os_error("CopySid(TokenUser)"));
    }
    Ok(sid)
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
    stderr_read: Option<OwnedHandle>,
    pseudoconsole: Option<RestrictedConpty>,
}

fn spawn_restricted_process(
    request: &LocalExecutionRequest,
    cwd: &Path,
    token: HANDLE,
) -> io::Result<SpawnedRestrictedProcess> {
    if request.tty {
        return spawn_restricted_conpty_process(request, cwd, token);
    }
    spawn_restricted_pipe_process(request, cwd, token)
}

fn spawn_restricted_pipe_process(
    request: &LocalExecutionRequest,
    cwd: &Path,
    token: HANDLE,
) -> io::Result<SpawnedRestrictedProcess> {
    let (stdin_read, stdin_write) = create_pipe_pair(true)?;
    let (stdout_read, stdout_write) = create_pipe_pair(false)?;
    let (stderr_read, stderr_write) = create_pipe_pair(false)?;
    let job = create_kill_on_close_job()?;
    let command_line_text = argv_to_command_line(&request.command);
    let mut command_line = to_wide(&command_line_text);
    let mut env_block = environment_block(&request.env)?;
    let cwd = to_wide(cwd.as_os_str());
    let mut desktop = to_wide("winsta0\\default");
    let mut attributes = ProcessAttributeList::new(2)?;
    attributes.set_handle_list(&[stdin_read.raw(), stdout_write.raw(), stderr_write.raw()])?;
    attributes.set_job(job.raw())?;
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
            CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT,
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
        stderr_read: Some(stderr_read),
        pseudoconsole: None,
    })
}

fn spawn_restricted_conpty_process(
    request: &LocalExecutionRequest,
    cwd: &Path,
    token: HANDLE,
) -> io::Result<SpawnedRestrictedProcess> {
    let (rows, cols) = request.pty_size.unwrap_or((24, 120));
    let (pseudoconsole, stdin_write, stdout_read) = RestrictedConpty::create(rows, cols)?;
    let job = create_kill_on_close_job()?;
    let command_line_text = argv_to_command_line(&request.command);
    let mut command_line = to_wide(&command_line_text);
    let mut env_block = environment_block(&request.env)?;
    let cwd = to_wide(cwd.as_os_str());
    let mut desktop = to_wide("winsta0\\default");
    let mut attributes = ProcessAttributeList::new(2)?;
    attributes.set_pseudoconsole(pseudoconsole.raw())?;
    attributes.set_job(job.raw())?;
    let mut startup: STARTUPINFOEXW = unsafe { std::mem::zeroed() };
    startup.StartupInfo.cb = std::mem::size_of::<STARTUPINFOEXW>() as u32;
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = INVALID_HANDLE_VALUE;
    startup.StartupInfo.hStdOutput = INVALID_HANDLE_VALUE;
    startup.StartupInfo.hStdError = INVALID_HANDLE_VALUE;
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
            0,
            CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT,
            env_block.as_mut_ptr() as *mut c_void,
            cwd.as_ptr(),
            &startup.StartupInfo,
            &mut process_info,
        )
    };
    if created == 0 {
        return Err(last_os_error("CreateProcessAsUserW ConPTY"));
    }
    let process = OwnedHandle::new(process_info.hProcess, "CreateProcessAsUserW process")?;
    let thread = OwnedHandle::new(process_info.hThread, "CreateProcessAsUserW thread")?;
    let stdin_write = request.stdin.then_some(stdin_write);
    Ok(SpawnedRestrictedProcess {
        process,
        thread,
        job,
        stdin_write,
        stdout_read,
        stderr_read: None,
        pseudoconsole: Some(pseudoconsole),
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

    fn job_flags(job: &OwnedHandle) -> u32 {
        let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        let result = unsafe {
            QueryInformationJobObject(
                job.raw(),
                JobObjectExtendedLimitInformation,
                &mut limits as *mut _ as *mut c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                ptr::null_mut(),
            )
        };
        assert_ne!(result, 0, "job limit query should succeed");
        limits.BasicLimitInformation.LimitFlags
    }

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
    fn network_permission_selects_sandbox_account() {
        assert_eq!(
            sandbox_account_for_permissions(None),
            WINDOWS_SANDBOX_OFFLINE_USERNAME
        );
        let permissions = GrantedPermissionProfile {
            network: Some(
                app_server_protocol::protocol::v2::AdditionalNetworkPermissions {
                    enabled: Some(true),
                },
            ),
            file_system: None,
        };
        assert_eq!(
            sandbox_account_for_permissions(Some(&permissions)),
            WINDOWS_SANDBOX_ONLINE_USERNAME
        );
    }

    #[test]
    fn restricted_job_disables_breakaway_until_descendant_preservation() {
        let job = create_kill_on_close_job().expect("job should start without breakaway");
        let initial_flags = job_flags(&job);
        assert_ne!(initial_flags & JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, 0);
        assert_eq!(initial_flags & JOB_OBJECT_LIMIT_BREAKAWAY_OK, 0);

        preserve_job_descendants(&job).expect("preservation should enable breakaway");
        assert_ne!(job_flags(&job) & JOB_OBJECT_LIMIT_BREAKAWAY_OK, 0);
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
