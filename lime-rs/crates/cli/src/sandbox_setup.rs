use std::path::PathBuf;
#[cfg(windows)]
use std::time::Duration;

use anyhow::{bail, Context};
#[cfg(windows)]
use app_server_client::SessionEvent;
#[cfg(windows)]
use app_server_protocol::protocol::v2::{
    ServerNotification, WindowsSandboxSetupMode, WindowsSandboxSetupStartParams,
    WindowsSandboxSetupStartResponse, METHOD_WINDOWS_SANDBOX_SETUP_START,
};
use clap::{ArgAction, ArgGroup, Parser};

#[cfg(windows)]
use crate::{start_session, ConnectionArgs};

#[derive(Debug, Parser)]
#[command(group(
    ArgGroup::new("sandbox_user")
        .required(true)
        .args(["user", "current_user"])
))]
pub(crate) struct SandboxSetupCommand {
    /// Set up the elevated Windows sandbox.
    #[arg(long = "elevated", action = ArgAction::SetTrue)]
    elevated_sandbox_level: bool,

    /// Windows user that will run Lime after managed deployment.
    #[arg(
        long = "user",
        value_name = "USER",
        conflicts_with = "current_user",
        requires = "codex_home"
    )]
    user: Option<String>,

    /// Use the current Windows user as the Lime user.
    #[arg(
        long = "current-user",
        default_value_t = false,
        conflicts_with = "user"
    )]
    current_user: bool,

    /// CODEX_HOME for the managed user. Required with --user.
    #[arg(long = "codex-home", value_name = "DIR")]
    codex_home: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SandboxSetupLevel {
    Elevated,
}

impl SandboxSetupCommand {
    fn setup_level(&self) -> anyhow::Result<SandboxSetupLevel> {
        if self.elevated_sandbox_level {
            Ok(SandboxSetupLevel::Elevated)
        } else {
            bail!("`lime sandbox setup` currently requires --elevated");
        }
    }
}

pub(crate) async fn run(
    cmd: SandboxSetupCommand,
    config_profile: Option<String>,
) -> anyhow::Result<()> {
    match cmd.setup_level()? {
        SandboxSetupLevel::Elevated => run_elevated(cmd, config_profile).await,
    }
}

pub(crate) fn parse_setup_command(
    sandbox_command: &[String],
) -> anyhow::Result<Option<SandboxSetupCommand>> {
    if sandbox_command
        .first()
        .is_none_or(|command| command != "setup")
    {
        return Ok(None);
    }

    SandboxSetupCommand::try_parse_from(sandbox_command.iter().map(String::as_str))
        .map(Some)
        .map_err(anyhow::Error::from)
}

#[cfg(not(windows))]
async fn run_elevated(
    cmd: SandboxSetupCommand,
    config_profile: Option<String>,
) -> anyhow::Result<()> {
    let identity = resolve_sandbox_setup_identity(&cmd)?;
    let _ = (&identity.real_user, &identity.codex_home);
    let _ = config_profile;
    bail!("Windows elevated sandbox setup is only available on Windows")
}

#[cfg(windows)]
async fn run_elevated(
    cmd: SandboxSetupCommand,
    config_profile: Option<String>,
) -> anyhow::Result<()> {
    if config_profile.is_some() {
        bail!("--profile is not supported by the App Server-owned Lime configuration");
    }
    let identity = resolve_sandbox_setup_identity(&cmd)?;
    if !cmd.current_user {
        bail!("managed-user sandbox setup is not exposed by the App Server; use --current-user");
    }

    let mode = WindowsSandboxSetupMode::Elevated;
    let mut session = start_session(&ConnectionArgs::default())
        .await
        .context("failed to start App Server for Windows sandbox setup")?;
    let response = session
        .request_handle()
        .request::<_, WindowsSandboxSetupStartResponse>(
            METHOD_WINDOWS_SANDBOX_SETUP_START,
            WindowsSandboxSetupStartParams {
                mode,
                cwd: Some(std::env::current_dir()?),
            },
        )
        .await;
    let completion = match response {
        Ok(response) if response.started => tokio::time::timeout(Duration::from_secs(130), async {
            loop {
                match session.next_event().await {
                    Some(SessionEvent::Notification(notification)) => match *notification {
                        ServerNotification::WindowsSandboxSetupCompleted(completed)
                            if completed.mode == mode =>
                        {
                            break Ok(completed)
                        }
                        _ => {}
                    },
                    Some(SessionEvent::Disconnected { message }) => {
                        break Err(anyhow::anyhow!(
                            "App Server disconnected during Windows sandbox setup: {message}"
                        ));
                    }
                    Some(_) => {}
                    None => {
                        break Err(anyhow::anyhow!(
                            "App Server closed during Windows sandbox setup"
                        ));
                    }
                }
            }
        })
        .await
        .context("Windows sandbox setup timed out")?,
        Ok(_) => Err(anyhow::anyhow!(
            "App Server did not start Windows sandbox setup"
        )),
        Err(error) => Err(error.into()),
    };
    let shutdown = session.shutdown().await;
    let completion = completion?;
    shutdown.context("failed to shut down App Server Windows sandbox setup session")?;
    if !completion.success {
        bail!(
            "Windows elevated sandbox setup failed: {}",
            completion
                .error
                .unwrap_or_else(|| "unknown setup error".to_string())
        );
    }

    println!(
        "Windows elevated sandbox setup completed for {} at {}.",
        identity.real_user,
        identity.codex_home.display()
    );
    Ok(())
}

struct SandboxSetupIdentity {
    real_user: String,
    codex_home: PathBuf,
}

fn resolve_sandbox_setup_identity(
    cmd: &SandboxSetupCommand,
) -> anyhow::Result<SandboxSetupIdentity> {
    if cmd.current_user {
        let real_user = std::env::var("USERNAME")
            .or_else(|_| std::env::var("USER"))
            .map_err(|error| {
                anyhow::anyhow!("failed to determine current user from environment: {error}")
            })?;
        let codex_home = match cmd.codex_home.clone() {
            Some(codex_home) => codex_home,
            None => std::env::current_dir().context("failed to resolve current directory")?,
        };
        return Ok(SandboxSetupIdentity {
            real_user,
            codex_home,
        });
    }

    let real_user = cmd
        .user
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--user or --current-user is required"))?;
    let codex_home = cmd
        .codex_home
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--codex-home is required with --user"))?;
    Ok(SandboxSetupIdentity {
        real_user,
        codex_home,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_managed_user_identity() {
        let command = SandboxSetupCommand::try_parse_from([
            "setup",
            "--elevated",
            "--user",
            "DOMAIN\\alice",
            "--codex-home",
            r"C:\Users\alice\.codex",
        ])
        .expect("parse");

        assert!(command.elevated_sandbox_level);
        assert_eq!(command.user.as_deref(), Some(r"DOMAIN\alice"));
        assert!(!command.current_user);
        assert_eq!(
            command.codex_home.as_deref(),
            Some(std::path::Path::new(r"C:\Users\alice\.codex"))
        );
    }

    #[test]
    fn requires_explicit_user_identity() {
        let error = SandboxSetupCommand::try_parse_from(["setup", "--elevated"])
            .expect_err("parse should fail");

        assert_eq!(
            error.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );
    }

    #[test]
    fn requires_codex_home_for_managed_user() {
        let error =
            SandboxSetupCommand::try_parse_from(["setup", "--elevated", "--user", "DOMAIN\\alice"])
                .expect_err("parse should fail");

        assert_eq!(
            error.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );
    }

    #[test]
    fn parses_setup_from_sandbox_command_args() {
        let command = parse_setup_command(&[
            "setup".to_string(),
            "--elevated".to_string(),
            "--user".to_string(),
            r"DOMAIN\alice".to_string(),
            "--codex-home".to_string(),
            r"C:\Users\alice\.codex".to_string(),
        ])
        .expect("parse")
        .expect("setup command");

        assert_eq!(command.user.as_deref(), Some(r"DOMAIN\alice"));
    }

    #[test]
    fn ignores_non_setup_sandbox_command_args() {
        let command =
            parse_setup_command(&["echo".to_string(), "hello".to_string()]).expect("parse");

        assert!(command.is_none());
    }
}
