mod commands;

use std::future::Future;
use std::io::{self, Write};
use std::process::ExitCode;

use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::{generate, Shell};

/// Lime CLI
///
/// If no subcommand is specified, options are forwarded to the interactive TUI.
#[derive(Debug, Parser)]
#[command(
    name = "lime",
    version,
    about = "Lime CLI",
    subcommand_negates_reqs = true,
    override_usage = "lime [OPTIONS]\n       lime [OPTIONS] <COMMAND> [ARGS]"
)]
struct Cli {
    #[command(flatten)]
    interactive: commands::TuiArgs,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Start the interactive TUI.
    Tui(commands::TuiArgs),

    /// Run Lime non-interactively.
    #[command(visible_alias = "e")]
    Exec(commands::ExecArgs),

    /// Resume a previous interactive thread.
    Resume(commands::ResumeArgs),

    /// Manage canonical threads through App Server.
    Thread(commands::ThreadCommand),

    /// Inspect MCP server status through App Server.
    Mcp(commands::McpCommand),

    /// Inspect executable skills through App Server.
    Skills(commands::SkillsCommand),

    /// Generate shell completion for the canonical `lime` command tree.
    Completion(CompletionArgs),
}

#[derive(Debug, clap::Args)]
struct CompletionArgs {
    #[arg(value_enum)]
    shell: Shell,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    run_async(dispatch(cli))
}

async fn dispatch(cli: Cli) -> ExitCode {
    match cli.command {
        None => commands::run_tui(cli.interactive).await,
        Some(Command::Tui(args)) => commands::run_tui(args).await,
        Some(Command::Exec(args)) => commands::run_exec(args).await,
        Some(Command::Resume(args)) => commands::run_resume(args).await,
        Some(Command::Thread(args)) => commands::run_thread(args).await,
        Some(Command::Mcp(args)) => commands::run_mcp(args).await,
        Some(Command::Skills(args)) => commands::run_skills(args).await,
        Some(Command::Completion(args)) => {
            let mut command = Cli::command();
            let mut output = io::stdout();
            generate(args.shell, &mut command, "lime", &mut output);
            let _ = output.flush();
            ExitCode::SUCCESS
        }
    }
}

fn run_async(future: impl Future<Output = ExitCode>) -> ExitCode {
    match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime.block_on(future),
        Err(error) => {
            let error = io::Error::other(format!("failed to initialize CLI runtime: {error}"));
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_subcommand_selects_the_tui() {
        let cli = Cli::try_parse_from(["lime"]).expect("parse default TUI");
        assert!(cli.command.is_none());
    }

    #[test]
    fn tui_is_the_explicit_interactive_command() {
        let cli = Cli::try_parse_from(["lime", "tui"]).expect("parse TUI command");
        assert!(matches!(cli.command, Some(Command::Tui(_))));
    }

    #[test]
    fn old_direct_runtime_commands_stay_absent() {
        for command in ["task", "media", "skill", "doctor"] {
            let error = Cli::try_parse_from(["lime", command]).expect_err("retired command");
            assert_eq!(error.kind(), clap::error::ErrorKind::InvalidSubcommand);
        }
    }

    #[test]
    fn exec_alias_matches_codex_cli_shape() {
        let cli = Cli::try_parse_from(["lime", "e", "check", "this"]).expect("parse exec alias");
        assert!(matches!(cli.command, Some(Command::Exec(_))));
    }

    #[test]
    fn completion_uses_the_same_command_tree() {
        let cli = Cli::try_parse_from(["lime", "completion", "zsh"]).expect("parse zsh completion");
        assert!(matches!(
            cli.command,
            Some(Command::Completion(CompletionArgs { shell: Shell::Zsh }))
        ));
    }
}
