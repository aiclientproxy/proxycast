use std::collections::HashSet;
use std::env;
use std::ffi::OsString;
use std::io::{self, IsTerminal, Read};
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use app_server_client::{ClientSession, StdioTransportConfig};
use app_server_protocol::protocol::v2::{
    ListMcpServerStatusParams, ListMcpServerStatusResponse, McpServerStatusDetail,
    SkillsListParams, SkillsListResponse, ThreadArchiveParams, ThreadArchiveResponse,
    ThreadDeleteParams, ThreadDeleteResponse, ThreadForkParams, ThreadForkResponse,
    ThreadListParams, ThreadListResponse, ThreadReadParams, ThreadReadResponse,
    ThreadUnarchiveParams, ThreadUnarchiveResponse, METHOD_MCP_SERVER_STATUS_LIST,
    METHOD_SKILLS_LIST, METHOD_THREAD_ARCHIVE, METHOD_THREAD_DELETE, METHOD_THREAD_FORK,
    METHOD_THREAD_LIST, METHOD_THREAD_READ, METHOD_THREAD_UNARCHIVE,
};
use app_server_protocol::{ClientCapabilities, ClientInfo, InitializeParams};
use clap::Args;
use serde_json::json;
use tui::{ExecOptions, TuiOptions};

const APP_SERVER_BIN_ENV: &str = "LIME_APP_SERVER_BIN";

#[derive(Debug, Args, Clone)]
pub(crate) struct ConnectionArgs {
    #[arg(long = "app-server", value_name = "PATH")]
    app_server: Option<PathBuf>,
    #[arg(
        long = "app-server-arg",
        value_name = "ARG",
        allow_hyphen_values = true,
        hide = true
    )]
    app_server_args: Vec<OsString>,
    #[arg(long, value_name = "DIR")]
    cwd: Option<PathBuf>,
    #[arg(long)]
    model: Option<String>,
    #[arg(long, visible_alias = "model-provider")]
    provider: Option<String>,
    #[arg(long, value_name = "EFFORT")]
    effort: Option<String>,
    #[arg(long, value_name = "PROFILE")]
    permissions: Option<String>,
}

#[derive(Debug, Args)]
pub(crate) struct TuiArgs {
    #[command(flatten)]
    connection: ConnectionArgs,
}

#[derive(Debug, Args)]
pub(crate) struct ExecArgs {
    #[arg(value_name = "PROMPT")]
    prompt: Vec<String>,
    /// 输出稳定 JSON envelope。
    #[arg(long, conflicts_with = "jsonl")]
    json: bool,
    /// 输出稳定单行 JSONL envelope，便于脚本逐行消费。
    #[arg(long, conflicts_with = "json")]
    jsonl: bool,
    #[command(flatten)]
    connection: ConnectionArgs,
}

#[derive(Debug, Args)]
pub(crate) struct ResumeArgs {
    /// 要恢复的 canonical Thread id。
    thread_id: Option<String>,
    #[command(flatten)]
    connection: ConnectionArgs,
}

#[derive(Debug, Args)]
pub(crate) struct ThreadCommand {
    #[command(subcommand)]
    command: ThreadSubcommand,
}

#[derive(Debug, Args)]
pub(crate) struct McpCommand {
    #[command(subcommand)]
    command: McpSubcommand,
}

#[derive(Debug, clap::Subcommand)]
enum McpSubcommand {
    /// 读取 current MCP server status catalog。
    List(McpListArgs),
}

#[derive(Debug, Args)]
struct McpListArgs {
    #[arg(long)]
    cursor: Option<String>,
    #[arg(long)]
    limit: Option<u32>,
    /// 仅读取工具和认证状态，不读取完整资源详情。
    #[arg(long)]
    tools_only: bool,
    #[command(flatten)]
    connection: ConnectionArgs,
}

#[derive(Debug, Args)]
pub(crate) struct SkillsCommand {
    #[command(subcommand)]
    command: SkillsSubcommand,
}

#[derive(Debug, clap::Subcommand)]
enum SkillsSubcommand {
    /// 读取 current executable Skill catalog。
    List(SkillsListArgs),
}

#[derive(Debug, Args)]
struct SkillsListArgs {
    #[arg(long = "skill-cwd", visible_alias = "cwds", value_name = "DIR")]
    skill_cwds: Vec<PathBuf>,
    #[arg(long)]
    force_reload: bool,
    #[command(flatten)]
    connection: ConnectionArgs,
}

#[derive(Debug, clap::Subcommand)]
enum ThreadSubcommand {
    /// 列出当前 App Server 可见的 Thread。
    List(ThreadListArgs),
    /// 读取一个 Thread 的 canonical metadata，可选完整 turns。
    Show(ThreadReadArgs),
    /// 归档一个 Thread。
    Archive(ThreadIdArgs),
    /// 取消归档一个 Thread。
    Unarchive(ThreadIdArgs),
    /// 删除一个 Thread 及其 canonical descendants。
    Delete(ThreadIdArgs),
    /// 从一个 Thread fork 出新 Thread。
    Fork(ThreadForkArgs),
}

#[derive(Debug, Args)]
struct ThreadListArgs {
    #[arg(long)]
    limit: Option<u32>,
    #[arg(long)]
    search: Option<String>,
    #[command(flatten)]
    connection: ConnectionArgs,
}

#[derive(Debug, Args)]
struct ThreadReadArgs {
    thread_id: String,
    #[arg(long)]
    include_turns: bool,
    #[command(flatten)]
    connection: ConnectionArgs,
}

#[derive(Debug, Args)]
struct ThreadIdArgs {
    thread_id: String,
    #[command(flatten)]
    connection: ConnectionArgs,
}

#[derive(Debug, Args)]
struct ThreadForkArgs {
    thread_id: String,
    #[arg(long)]
    last_turn_id: Option<String>,
    #[arg(long)]
    before_turn_id: Option<String>,
    #[arg(long)]
    exclude_turns: bool,
    #[command(flatten)]
    connection: ConnectionArgs,
}

impl Default for TuiArgs {
    fn default() -> Self {
        Self {
            connection: ConnectionArgs {
                app_server: None,
                app_server_args: Vec::new(),
                cwd: None,
                model: None,
                provider: None,
                effort: None,
                permissions: None,
            },
        }
    }
}

pub(crate) async fn run_tui(args: TuiArgs) -> ExitCode {
    match tui::run_tui(tui_options(args.connection, None)).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::FAILURE
        }
    }
}

pub(crate) async fn run_resume(args: ResumeArgs) -> ExitCode {
    match tui::run_resume(tui_options(args.connection, args.thread_id)).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::FAILURE
        }
    }
}

pub(crate) async fn run_thread(command: ThreadCommand) -> ExitCode {
    let result = run_thread_request(command).await;
    match result {
        Ok(value) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&value).unwrap_or_else(|_| "{}".to_string())
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::FAILURE
        }
    }
}

pub(crate) async fn run_mcp(command: McpCommand) -> ExitCode {
    let result = run_mcp_request(command).await;
    match result {
        Ok(value) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&value).unwrap_or_else(|_| "{}".to_string())
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run_mcp_request(command: McpCommand) -> Result<serde_json::Value> {
    let McpSubcommand::List(args) = command.command;
    let config = thread_stdio_config(&args.connection);
    let session = ClientSession::start_stdio(config, cli_initialize_params()).await?;
    let request_handle = session.request_handle();
    let mut params = ListMcpServerStatusParams {
        cursor: args.cursor,
        limit: args.limit,
        detail: args
            .tools_only
            .then_some(McpServerStatusDetail::ToolsAndAuthOnly),
        thread_id: None,
    };
    let mut data = Vec::new();
    let mut seen_cursors = HashSet::new();
    let mut page_count = 0usize;
    let response = loop {
        page_count += 1;
        if page_count > 256 {
            let _ = session.shutdown().await;
            bail!("MCP status pagination exceeded 256 pages");
        }
        let page: Result<ListMcpServerStatusResponse, _> = request_handle
            .request(METHOD_MCP_SERVER_STATUS_LIST, params.clone())
            .await;
        let page = match page {
            Ok(page) => page,
            Err(error) => {
                let _ = session.shutdown().await;
                return Err(error.into());
            }
        };
        data.extend(page.data);
        let Some(next_cursor) = page.next_cursor else {
            break ListMcpServerStatusResponse {
                data,
                next_cursor: None,
            };
        };
        if !seen_cursors.insert(next_cursor.clone()) {
            let _ = session.shutdown().await;
            bail!("MCP status pagination repeated cursor {next_cursor}");
        }
        params.cursor = Some(next_cursor);
    };
    session.shutdown().await?;
    Ok(serde_json::to_value(response)?)
}

pub(crate) async fn run_skills(command: SkillsCommand) -> ExitCode {
    let result = run_skills_request(command).await;
    match result {
        Ok(value) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&value).unwrap_or_else(|_| "{}".to_string())
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run_skills_request(command: SkillsCommand) -> Result<serde_json::Value> {
    let SkillsSubcommand::List(args) = command.command;
    let cwd = if args.skill_cwds.is_empty() {
        vec![args
            .connection
            .cwd
            .clone()
            .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))]
    } else {
        args.skill_cwds
    };
    let config = thread_stdio_config(&args.connection);
    let session = ClientSession::start_stdio(config, cli_initialize_params()).await?;
    let request_handle = session.request_handle();
    let response: Result<SkillsListResponse, _> = request_handle
        .request(
            METHOD_SKILLS_LIST,
            SkillsListParams {
                cwds: cwd,
                force_reload: args.force_reload,
            },
        )
        .await;
    let shutdown = session.shutdown().await;
    let response = response?;
    shutdown?;
    Ok(serde_json::to_value(response)?)
}

async fn run_thread_request(command: ThreadCommand) -> Result<serde_json::Value> {
    let (connection, method, params) = match command.command {
        ThreadSubcommand::List(args) => (
            args.connection,
            METHOD_THREAD_LIST,
            serde_json::to_value(ThreadListParams {
                limit: args.limit,
                search_term: args.search,
                ..ThreadListParams::default()
            })?,
        ),
        ThreadSubcommand::Show(args) => (
            args.connection,
            METHOD_THREAD_READ,
            serde_json::to_value(ThreadReadParams {
                thread_id: args.thread_id,
                include_turns: args.include_turns,
            })?,
        ),
        ThreadSubcommand::Archive(args) => (
            args.connection,
            METHOD_THREAD_ARCHIVE,
            serde_json::to_value(ThreadArchiveParams {
                thread_id: args.thread_id,
            })?,
        ),
        ThreadSubcommand::Unarchive(args) => (
            args.connection,
            METHOD_THREAD_UNARCHIVE,
            serde_json::to_value(ThreadUnarchiveParams {
                thread_id: args.thread_id,
            })?,
        ),
        ThreadSubcommand::Delete(args) => (
            args.connection,
            METHOD_THREAD_DELETE,
            serde_json::to_value(ThreadDeleteParams {
                thread_id: args.thread_id,
            })?,
        ),
        ThreadSubcommand::Fork(args) => (
            args.connection,
            METHOD_THREAD_FORK,
            serde_json::to_value(ThreadForkParams {
                thread_id: args.thread_id,
                last_turn_id: args.last_turn_id,
                before_turn_id: args.before_turn_id,
                exclude_turns: args.exclude_turns,
                ..ThreadForkParams::default()
            })?,
        ),
    };

    let config = thread_stdio_config(&connection);
    let session = ClientSession::start_stdio(config, cli_initialize_params()).await?;
    let request_handle = session.request_handle();
    let response = request_handle.request_value(method, params).await;
    let shutdown = session.shutdown().await;
    let value = response?;
    shutdown?;

    // Deserialize known responses here so a server drift cannot silently become
    // an untyped CLI success envelope.
    let value = match method {
        METHOD_THREAD_LIST => {
            serde_json::to_value(serde_json::from_value::<ThreadListResponse>(value)?)?
        }
        METHOD_THREAD_READ => {
            serde_json::to_value(serde_json::from_value::<ThreadReadResponse>(value)?)?
        }
        METHOD_THREAD_ARCHIVE => {
            serde_json::to_value(serde_json::from_value::<ThreadArchiveResponse>(value)?)?
        }
        METHOD_THREAD_UNARCHIVE => {
            serde_json::to_value(serde_json::from_value::<ThreadUnarchiveResponse>(value)?)?
        }
        METHOD_THREAD_DELETE => {
            serde_json::to_value(serde_json::from_value::<ThreadDeleteResponse>(value)?)?
        }
        METHOD_THREAD_FORK => {
            serde_json::to_value(serde_json::from_value::<ThreadForkResponse>(value)?)?
        }
        _ => unreachable!("thread method is exhaustive"),
    };
    Ok(value)
}

fn thread_stdio_config(connection: &ConnectionArgs) -> StdioTransportConfig {
    let mut config =
        StdioTransportConfig::runtime(resolve_app_server_bin(connection.app_server.clone()));
    config
        .args
        .extend(connection.app_server_args.iter().cloned());
    config
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

pub(crate) async fn run_exec(args: ExecArgs) -> ExitCode {
    let json_output = args.json;
    let jsonl_output = args.jsonl;
    let result = read_prompt(args.prompt).map(|prompt| ExecOptions {
        tui: tui_options(args.connection, None),
        prompt,
    });
    let result = match result {
        Ok(options) => tui::run_exec(options).await,
        Err(error) => Err(error),
    };

    match result {
        Ok(output) => {
            if json_output || jsonl_output {
                let value = json!({ "ok": true, "result": output });
                println!("{}", render_json_envelope(&value, jsonl_output));
            } else {
                println!("{}", output.output);
            }
            match output.status.as_str() {
                "ready" => ExitCode::SUCCESS,
                "interrupted" => ExitCode::from(130),
                _ => ExitCode::FAILURE,
            }
        }
        Err(error) => {
            if json_output || jsonl_output {
                let value = json!({
                    "ok": false,
                    "error": {
                        "kind": "runtime",
                        "message": format!("{error:#}"),
                    }
                });
                println!("{}", render_json_envelope(&value, jsonl_output));
            } else {
                eprintln!("{error:#}");
            }
            ExitCode::FAILURE
        }
    }
}

fn render_json_envelope(value: &serde_json::Value, jsonl: bool) -> String {
    if jsonl {
        serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string())
    } else {
        serde_json::to_string_pretty(value).unwrap_or_else(|_| "{}".to_string())
    }
}

fn tui_options(args: ConnectionArgs, resume_thread: Option<String>) -> TuiOptions {
    TuiOptions {
        app_server_bin: resolve_app_server_bin(args.app_server),
        app_server_args: args.app_server_args,
        cwd: args
            .cwd
            .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from("."))),
        model: args.model,
        model_provider: args.provider,
        reasoning_effort: args.effort,
        permissions: args.permissions,
        resume_thread,
    }
}

fn resolve_app_server_bin(explicit: Option<PathBuf>) -> PathBuf {
    if let Some(path) = explicit {
        return path;
    }
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

fn read_prompt(parts: Vec<String>) -> Result<String> {
    if !parts.is_empty() {
        return Ok(parts.join(" "));
    }
    if io::stdin().is_terminal() {
        bail!("prompt is required when stdin is a terminal");
    }
    let mut prompt = String::new();
    io::stdin()
        .read_to_string(&mut prompt)
        .context("failed to read prompt from stdin")?;
    if prompt.trim().is_empty() {
        bail!("prompt must not be empty");
    }
    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn explicit_app_server_path_has_highest_precedence() {
        let path = resolve_app_server_bin(Some(PathBuf::from("/tmp/custom-app-server")));
        assert_eq!(path, PathBuf::from("/tmp/custom-app-server"));
    }

    #[test]
    fn prompt_parts_preserve_shell_token_boundaries() {
        assert_eq!(
            read_prompt(vec!["review".to_string(), "this diff".to_string()]).unwrap(),
            "review this diff"
        );
    }

    #[test]
    fn exec_parser_keeps_options_out_of_multi_word_prompt() {
        let cli = crate::Cli::try_parse_from([
            "lime",
            "exec",
            "review",
            "this diff",
            "--json",
            "--cwd",
            "/tmp/worktree",
            "--model",
            "gpt-test",
            "--provider",
            "openai-test",
            "--effort",
            "high",
            "--permissions",
            "workspace-write",
            "--app-server-arg=--backend",
            "--app-server-arg=external",
        ])
        .expect("parse exec");
        let Some(crate::Command::Exec(args)) = cli.command else {
            panic!("expected exec command");
        };

        assert_eq!(args.prompt, vec!["review", "this diff"]);
        assert!(args.json);
        assert!(!args.jsonl);
        assert_eq!(args.connection.cwd, Some(PathBuf::from("/tmp/worktree")));
        assert_eq!(args.connection.model.as_deref(), Some("gpt-test"));
        assert_eq!(args.connection.provider.as_deref(), Some("openai-test"));
        assert_eq!(args.connection.effort.as_deref(), Some("high"));
        assert_eq!(
            args.connection.permissions.as_deref(),
            Some("workspace-write")
        );
        assert_eq!(
            args.connection.app_server_args,
            vec![OsString::from("--backend"), OsString::from("external")]
        );
    }

    #[test]
    fn exec_parser_supports_jsonl_as_a_mutually_exclusive_format() {
        let cli = crate::Cli::try_parse_from(["lime", "exec", "check", "--jsonl"])
            .expect("parse JSONL exec");
        let Some(crate::Command::Exec(args)) = cli.command else {
            panic!("expected exec command");
        };
        assert!(args.jsonl);
        assert!(!args.json);

        let error = crate::Cli::try_parse_from(["lime", "exec", "check", "--json", "--jsonl"])
            .expect_err("JSON and JSONL must not be combined");
        assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    #[test]
    fn jsonl_envelope_is_single_line_while_json_remains_pretty() {
        let value = json!({"ok": true, "result": {"output": "done"}});
        let jsonl = render_json_envelope(&value, true);
        let pretty = render_json_envelope(&value, false);
        assert!(!jsonl.contains('\n'));
        assert!(pretty.contains('\n'));
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&jsonl).unwrap(),
            value
        );
    }

    #[test]
    fn resume_parser_keeps_thread_identity_separate_from_connection_options() {
        let cli = crate::Cli::try_parse_from([
            "lime",
            "resume",
            "thread-42",
            "--cwd",
            "/tmp/worktree",
            "--app-server",
            "/tmp/app-server",
        ])
        .expect("parse resume");
        let Some(crate::Command::Resume(args)) = cli.command else {
            panic!("expected resume command");
        };

        assert_eq!(args.thread_id.as_deref(), Some("thread-42"));
        assert_eq!(args.connection.cwd, Some(PathBuf::from("/tmp/worktree")));
        assert_eq!(
            args.connection.app_server,
            Some(PathBuf::from("/tmp/app-server"))
        );
    }

    #[test]
    fn resume_without_id_selects_from_the_tui_session_picker() {
        let cli = crate::Cli::try_parse_from(["lime", "resume"]).expect("parse resume picker");
        let Some(crate::Command::Resume(args)) = cli.command else {
            panic!("expected resume command");
        };
        assert!(args.thread_id.is_none());
    }

    #[test]
    fn thread_fork_parser_keeps_v2_identity_and_history_flags() {
        let cli = crate::Cli::try_parse_from([
            "lime",
            "thread",
            "fork",
            "thread-42",
            "--last-turn-id",
            "turn-7",
            "--exclude-turns",
        ])
        .expect("parse thread fork");
        let Some(crate::Command::Thread(command)) = cli.command else {
            panic!("expected thread command");
        };
        let ThreadSubcommand::Fork(args) = command.command else {
            panic!("expected thread fork command");
        };

        assert_eq!(args.thread_id, "thread-42");
        assert_eq!(args.last_turn_id.as_deref(), Some("turn-7"));
        assert!(args.exclude_turns);
    }

    #[test]
    fn mcp_list_parser_keeps_status_pagination_and_detail_flags() {
        let cli = crate::Cli::try_parse_from([
            "lime",
            "mcp",
            "list",
            "--cursor",
            "page-2",
            "--limit",
            "25",
            "--tools-only",
            "--app-server",
            "/tmp/app-server",
        ])
        .expect("parse mcp list");
        let Some(crate::Command::Mcp(command)) = cli.command else {
            panic!("expected mcp command");
        };
        let McpSubcommand::List(args) = command.command;
        assert_eq!(args.cursor.as_deref(), Some("page-2"));
        assert_eq!(args.limit, Some(25));
        assert!(args.tools_only);
        assert_eq!(
            args.connection.app_server,
            Some(PathBuf::from("/tmp/app-server"))
        );
    }

    #[test]
    fn skills_list_parser_accepts_multiple_cwds_and_force_reload() {
        let cli = crate::Cli::try_parse_from([
            "lime",
            "skills",
            "list",
            "--skill-cwd",
            "/tmp/one",
            "--skill-cwd",
            "/tmp/two",
            "--force-reload",
        ])
        .expect("parse skills list");
        let Some(crate::Command::Skills(command)) = cli.command else {
            panic!("expected skills command");
        };
        let SkillsSubcommand::List(args) = command.command;
        assert_eq!(
            args.skill_cwds,
            vec![PathBuf::from("/tmp/one"), PathBuf::from("/tmp/two")]
        );
        assert!(args.force_reload);
    }
}
