use std::env;
use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use app_server_client::{ClientSession, SessionEvent, StdioTransportConfig};
use app_server_protocol::protocol::v2::{
    CommandExecOutputStream, CommandExecParams, CommandExecResponse, CommandExecWriteParams,
    ServerNotification, ServerRequest, METHOD_COMMAND_EXEC, METHOD_COMMAND_EXEC_WRITE,
};
use app_server_protocol::{
    error_codes, ClientCapabilities, ClientInfo, InitializeParams, JsonRpcError,
};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use uuid::Uuid;

use crate::exit_status::handle_exit_status;
use crate::{LandlockCommand, SandboxStateArgs, SeatbeltCommand, WindowsCommand};

const APP_SERVER_BIN_ENV: &str = "LIME_APP_SERVER_BIN";

#[cfg(target_os = "macos")]
pub async fn run_command_under_seatbelt(command: SeatbeltCommand) -> Result<ExitCode> {
    let SeatbeltCommand {
        sandbox_state,
        permissions_profile,
        config_profile,
        cwd,
        include_managed_config,
        allow_unix_sockets,
        log_denials,
        command,
    } = command;
    run_command_under_sandbox(
        DebugSandboxConfigOptions {
            sandbox_state,
            permissions_profile,
            config_profile,
            cwd,
            include_managed_config,
        },
        command,
        SandboxType::Seatbelt,
        log_denials,
        &allow_unix_sockets,
    )
    .await
}

#[cfg(not(target_os = "macos"))]
pub async fn run_command_under_seatbelt(_command: SeatbeltCommand) -> Result<ExitCode> {
    bail!("Seatbelt sandbox is only available on macOS")
}

pub async fn run_command_under_landlock(command: LandlockCommand) -> Result<ExitCode> {
    let LandlockCommand {
        sandbox_state,
        permissions_profile,
        config_profile,
        cwd,
        include_managed_config,
        command,
    } = command;
    run_command_under_sandbox(
        DebugSandboxConfigOptions {
            sandbox_state,
            permissions_profile,
            config_profile,
            cwd,
            include_managed_config,
        },
        command,
        SandboxType::Landlock,
        false,
        &[],
    )
    .await
}

pub async fn run_command_under_windows_sandbox(command: WindowsCommand) -> Result<ExitCode> {
    let WindowsCommand {
        sandbox_state,
        permissions_profile,
        config_profile,
        cwd,
        include_managed_config,
        command,
    } = command;
    run_command_under_sandbox(
        DebugSandboxConfigOptions {
            sandbox_state,
            permissions_profile,
            config_profile,
            cwd,
            include_managed_config,
        },
        command,
        SandboxType::Windows,
        false,
        &[],
    )
    .await
}

enum SandboxType {
    #[cfg(target_os = "macos")]
    Seatbelt,
    Landlock,
    Windows,
}

#[derive(Debug)]
struct DebugSandboxConfigOptions {
    sandbox_state: SandboxStateArgs,
    permissions_profile: Option<String>,
    config_profile: Option<String>,
    cwd: Option<PathBuf>,
    include_managed_config: bool,
}

#[derive(Debug, Clone, Copy)]
enum ManagedRequirementsMode {
    Include,
    Ignore,
}

impl ManagedRequirementsMode {
    fn for_profile_invocation(
        permissions_profile: &Option<String>,
        include_managed_config: bool,
    ) -> Self {
        if permissions_profile.is_some() && !include_managed_config {
            Self::Ignore
        } else {
            Self::Include
        }
    }
}

async fn run_command_under_sandbox(
    config_options: DebugSandboxConfigOptions,
    command: Vec<String>,
    sandbox_type: SandboxType,
    log_denials: bool,
    allow_unix_sockets: &[PathBuf],
) -> Result<ExitCode> {
    ensure_host_sandbox(sandbox_type)?;
    let mut params = command_exec_params(config_options, command, log_denials, allow_unix_sockets)?;
    let mut session = ClientSession::start_stdio(
        StdioTransportConfig::runtime(resolve_app_server_bin()),
        cli_initialize_params(),
    )
    .await
    .context("failed to start App Server for sandbox execution")?;
    let inherit_stdio = io::stdin().is_terminal() && io::stdout().is_terminal();
    let response = if inherit_stdio {
        let process_id = format!("lime-sandbox-{}", Uuid::new_v4());
        enable_inherited_stdio(&mut params, process_id);
        run_with_inherited_stdio(&mut session, params).await
    } else {
        session
            .request_handle()
            .request::<_, CommandExecResponse>(METHOD_COMMAND_EXEC, params)
            .await
            .map_err(Into::into)
    };
    let shutdown = session.shutdown().await;
    let response = response.context("App Server command/exec failed")?;
    shutdown.context("failed to shut down App Server sandbox session")?;

    io::stdout().write_all(response.stdout.as_bytes())?;
    io::stdout().flush()?;
    io::stderr().write_all(response.stderr.as_bytes())?;
    io::stderr().flush()?;
    Ok(handle_exit_status(response.exit_code))
}

fn enable_inherited_stdio(params: &mut CommandExecParams, process_id: String) {
    params.process_id = Some(process_id);
    params.tty = true;
    params.stream_stdin = true;
    params.stream_stdout_stderr = true;
}

async fn run_with_inherited_stdio(
    session: &mut ClientSession,
    params: CommandExecParams,
) -> Result<CommandExecResponse> {
    let process_id = params
        .process_id
        .clone()
        .expect("inherited stdio requires a process id");
    let request_handle = session.request_handle();
    let request = request_handle.request::<_, CommandExecResponse>(METHOD_COMMAND_EXEC, params);
    tokio::pin!(request);
    let mut stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    let mut stderr = tokio::io::stderr();
    let mut input = vec![0_u8; 8192];
    let mut stdin_open = true;

    loop {
        tokio::select! {
            response = &mut request => {
                return response.context("App Server command/exec failed");
            }
            event = session.next_event() => match event {
                Some(SessionEvent::Notification(notification)) => {
                    if let ServerNotification::CommandExecOutputDelta(delta) = *notification {
                        if delta.process_id != process_id {
                            continue;
                        }
                        let bytes = decode_output_delta(&delta.delta_base64)?;
                        match delta.stream {
                            CommandExecOutputStream::Stdout => {
                                stdout.write_all(&bytes).await?;
                                stdout.flush().await?;
                            }
                            CommandExecOutputStream::Stderr => {
                                stderr.write_all(&bytes).await?;
                                stderr.flush().await?;
                            }
                        }
                    }
                }
                Some(SessionEvent::ServerRequest(request)) => {
                    reject_unexpected_server_request(&request_handle, &request).await?;
                }
                Some(SessionEvent::RawServerRequest(request)) => {
                    request_handle
                        .reject(
                            request.id,
                            JsonRpcError::new(
                                error_codes::METHOD_NOT_FOUND,
                                "lime sandbox does not support server requests",
                            ),
                        )
                        .await
                        .context("failed to reject unexpected App Server request")?;
                }
                Some(SessionEvent::Disconnected { message }) => {
                    bail!("App Server disconnected during sandbox execution: {message}");
                }
                Some(SessionEvent::RawNotification(_)) => {}
                None => bail!("App Server event stream closed during sandbox execution"),
            },
            read = stdin.read(&mut input), if stdin_open => {
                let read = read?;
                if read == 0 {
                    request_handle
                        .request::<_, app_server_protocol::protocol::v2::CommandExecWriteResponse>(
                            METHOD_COMMAND_EXEC_WRITE,
                            CommandExecWriteParams {
                                process_id: process_id.clone(),
                                delta_base64: None,
                                close_stdin: true,
                            },
                        )
                        .await
                        .context("failed to close sandbox stdin")?;
                    stdin_open = false;
                } else {
                    request_handle
                        .request::<_, app_server_protocol::protocol::v2::CommandExecWriteResponse>(
                            METHOD_COMMAND_EXEC_WRITE,
                            CommandExecWriteParams {
                                process_id: process_id.clone(),
                                delta_base64: Some(STANDARD.encode(&input[..read])),
                                close_stdin: false,
                            },
                        )
                        .await
                        .context("failed to forward sandbox stdin")?;
                }
            }
        }
    }
}

fn decode_output_delta(delta_base64: &str) -> Result<Vec<u8>> {
    STANDARD
        .decode(delta_base64)
        .context("invalid command/exec output delta")
}

async fn reject_unexpected_server_request(
    request_handle: &app_server_client::RequestHandle,
    request: &ServerRequest,
) -> Result<()> {
    request_handle
        .reject(
            request.id().clone(),
            JsonRpcError::new(
                error_codes::METHOD_NOT_FOUND,
                "lime sandbox does not support server requests",
            ),
        )
        .await
        .context("failed to reject unexpected App Server request")?;
    Ok(())
}

fn command_exec_params(
    options: DebugSandboxConfigOptions,
    command: Vec<String>,
    log_denials: bool,
    allow_unix_sockets: &[PathBuf],
) -> Result<CommandExecParams> {
    if command.is_empty() {
        bail!("sandbox command must not be empty");
    }
    if options.config_profile.is_some() {
        bail!("--profile is not supported by the App Server-owned Lime configuration");
    }
    if options.include_managed_config {
        bail!("--include-managed-config requires a managed configuration owner");
    }
    if options.sandbox_state.sandbox_state_json.is_some()
        || !options.sandbox_state.sandbox_state_readable_root.is_empty()
        || options.sandbox_state.sandbox_state_disable_network
    {
        bail!("sandbox state replay is not exposed by App Server command/exec");
    }
    if log_denials {
        bail!("--log-denials is not exposed by App Server command/exec");
    }
    if !allow_unix_sockets.is_empty() {
        bail!("--allow-unix-socket is not exposed by App Server command/exec");
    }

    let _managed_requirements_mode = ManagedRequirementsMode::for_profile_invocation(
        &options.permissions_profile,
        options.include_managed_config,
    );
    if options
        .permissions_profile
        .as_deref()
        .is_some_and(|profile| profile.trim() == ":danger-full-access")
    {
        bail!("danger-full-access bypasses the sandbox and is not valid for `lime sandbox`");
    }
    let cwd = options
        .cwd
        .map(Ok)
        .unwrap_or_else(env::current_dir)
        .context("failed to resolve sandbox working directory")?;
    if !cwd.is_absolute() {
        bail!(
            "sandbox working directory must be absolute: {}",
            cwd.display()
        );
    }

    Ok(CommandExecParams {
        command,
        process_id: None,
        tty: false,
        stream_stdin: false,
        stream_stdout_stderr: false,
        output_bytes_cap: None,
        disable_output_cap: true,
        disable_timeout: true,
        timeout_ms: None,
        cwd: Some(cwd),
        env: None,
        size: None,
        sandbox_policy: None,
        permission_profile: options.permissions_profile,
    })
}

fn ensure_host_sandbox(sandbox_type: SandboxType) -> Result<()> {
    let supported = match sandbox_type {
        #[cfg(target_os = "macos")]
        SandboxType::Seatbelt => cfg!(target_os = "macos"),
        SandboxType::Landlock => cfg!(target_os = "linux"),
        SandboxType::Windows => cfg!(target_os = "windows"),
    };
    if supported {
        Ok(())
    } else {
        bail!("host sandbox is not available on this operating system")
    }
}

fn resolve_app_server_bin() -> PathBuf {
    if let Some(path) = env::var_os(APP_SERVER_BIN_ENV).filter(|value| !value.is_empty()) {
        return PathBuf::from(path);
    }
    if let Ok(current_exe) = env::current_exe() {
        if let Some(parent) = current_exe.parent() {
            let sibling = parent.join(if cfg!(windows) {
                "app-server.exe"
            } else {
                "app-server"
            });
            if sibling.is_file() {
                return sibling;
            }
        }
    }
    PathBuf::from(if cfg!(windows) {
        "app-server.exe"
    } else {
        "app-server"
    })
}

fn cli_initialize_params() -> InitializeParams {
    InitializeParams {
        client_info: ClientInfo {
            name: "lime".to_string(),
            title: Some("Lime CLI".to_string()),
            version: Some(env!("CARGO_PKG_VERSION").to_string()),
        },
        capabilities: ClientCapabilities {
            event_methods: Vec::new(),
            experimental_api: true,
            opt_out_notification_methods: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(
        permission_profile: Option<&str>,
        cwd: Option<PathBuf>,
    ) -> DebugSandboxConfigOptions {
        DebugSandboxConfigOptions {
            sandbox_state: SandboxStateArgs::default(),
            permissions_profile: permission_profile.map(str::to_string),
            config_profile: None,
            cwd,
            include_managed_config: false,
        }
    }

    #[test]
    fn debug_sandbox_honors_explicit_builtin_permission_profile() {
        let params = command_exec_params(
            options(Some(":read-only"), None),
            vec!["pwd".to_string()],
            false,
            &[],
        )
        .expect("build command/exec params");

        assert_eq!(params.permission_profile.as_deref(), Some(":read-only"));
        assert!(params.sandbox_policy.is_none());
    }

    #[test]
    fn debug_sandbox_honors_active_permission_profiles() {
        let params = command_exec_params(options(None, None), vec!["pwd".to_string()], false, &[])
            .expect("build command/exec params");

        assert!(params.permission_profile.is_none());
        assert!(params.sandbox_policy.is_none());
    }

    #[test]
    fn debug_sandbox_honors_explicit_named_permission_profile() {
        let params = command_exec_params(
            options(Some("limited-read-test"), None),
            vec!["pwd".to_string()],
            false,
            &[],
        )
        .expect("build command/exec params");

        assert_eq!(
            params.permission_profile.as_deref(),
            Some("limited-read-test")
        );
        assert!(params.sandbox_policy.is_none());
    }

    #[test]
    fn debug_sandbox_uses_explicit_cwd() {
        let cwd = env::current_dir().unwrap().join("sandbox-cwd");
        let params = command_exec_params(
            options(Some(":workspace"), Some(cwd.clone())),
            vec!["pwd".to_string()],
            false,
            &[],
        )
        .expect("build command/exec params");

        assert_eq!(params.cwd.as_ref(), Some(&cwd));
    }

    #[test]
    fn danger_full_access_is_rejected_instead_of_bypassing_sandbox() {
        let error = command_exec_params(
            options(Some(":danger-full-access"), None),
            vec!["pwd".to_string()],
            false,
            &[],
        )
        .expect_err("sandbox bypass must fail closed");

        assert!(error.to_string().contains("bypasses the sandbox"));
    }

    #[test]
    fn unsupported_codex_config_sources_fail_closed() {
        let mut unsupported = options(Some(":workspace"), None);
        unsupported.config_profile = Some("work".to_string());
        let error = command_exec_params(unsupported, vec!["pwd".to_string()], false, &[])
            .expect_err("a second config owner must not be accepted");

        assert!(error.to_string().contains("App Server-owned"));
    }

    #[test]
    fn inherited_stdio_enables_the_existing_streaming_command_contract() {
        let mut params = command_exec_params(
            options(Some(":workspace"), None),
            vec!["sh".to_string(), "-i".to_string()],
            false,
            &[],
        )
        .expect("build command/exec params");

        enable_inherited_stdio(&mut params, "process-1".to_string());

        assert_eq!(params.process_id.as_deref(), Some("process-1"));
        assert!(params.tty);
        assert!(params.stream_stdin);
        assert!(params.stream_stdout_stderr);
        assert!(params.disable_timeout);
        assert!(params.disable_output_cap);
    }

    #[test]
    fn output_delta_decoding_is_binary_safe_and_fail_closed() {
        assert_eq!(
            decode_output_delta(&STANDARD.encode([0, 1, 255])).unwrap(),
            [0, 1, 255]
        );
        let error = decode_output_delta("not-base64").expect_err("invalid output must fail");
        assert!(error
            .to_string()
            .contains("invalid command/exec output delta"));
    }
}
