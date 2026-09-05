mod mcp_cmd;
mod plugin_cmd;
mod queue_cmd;
mod sandbox_setup;
#[cfg(not(windows))]
mod wsl_paths;

use std::collections::{BTreeMap, HashSet};
use std::env;
use std::ffi::OsString;
use std::future::Future;
use std::io::{self, IsTerminal, Read, Write};
use std::path::PathBuf;
use std::process::{Command, ExitCode};

use anyhow::{bail, Context, Result};
use app_server_client::{ClientSession, RemoteTransportConfig, StdioTransportConfig};
use app_server_protocol::protocol::v2::{
    ExperimentalFeature, ExperimentalFeatureEnablementSetParams,
    ExperimentalFeatureEnablementSetResponse, ExperimentalFeatureListParams,
    ExperimentalFeatureListResponse, ExperimentalFeatureStage, MemoryResetResponse, Model,
    ModelListParams, ModelListResponse, SkillsListParams, SkillsListResponse, ThreadArchiveParams,
    ThreadArchiveResponse, ThreadDeleteParams, ThreadDeleteResponse, ThreadForkParams,
    ThreadForkResponse, ThreadListParams, ThreadListResponse, ThreadReadParams, ThreadReadResponse,
    ThreadUnarchiveParams, ThreadUnarchiveResponse, METHOD_EXPERIMENTAL_FEATURE_ENABLEMENT_SET,
    METHOD_EXPERIMENTAL_FEATURE_LIST, METHOD_MEMORY_RESET, METHOD_MODEL_LIST, METHOD_SKILLS_LIST,
    METHOD_THREAD_ARCHIVE, METHOD_THREAD_DELETE, METHOD_THREAD_FORK, METHOD_THREAD_LIST,
    METHOD_THREAD_READ, METHOD_THREAD_UNARCHIVE,
};
use app_server_protocol::{ClientCapabilities, ClientInfo, InitializeParams};
use clap::{Args, CommandFactory, Parser, Subcommand as ClapSubcommand};
use clap_complete::{generate, Shell};
use execpolicy::ExecPolicyCheckCommand;
use serde_json::json;
use tui::{ExecOptions, TuiOptions};

const APP_SERVER_BIN_ENV: &str = "LIME_APP_SERVER_BIN";

#[derive(Debug, Default, Args, Clone)]
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
    #[command(flatten)]
    remote: InteractiveRemoteOptions,
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

#[derive(Debug, Default, Args, Clone)]
pub(crate) struct InteractiveRemoteOptions {
    /// Connect to an App Server WebSocket instead of spawning a local process.
    #[arg(long = "remote", value_name = "URL")]
    remote: Option<String>,
    /// Environment variable containing the remote App Server bearer token.
    #[arg(long = "remote-auth-token-env", value_name = "ENV_VAR")]
    remote_auth_token_env: Option<String>,
}

#[derive(Debug, Args)]
pub(crate) struct TuiCli {
    #[command(flatten)]
    connection: ConnectionArgs,
    #[arg(long, value_name = "LOCALE")]
    locale: Option<String>,
}

#[derive(Debug, Args)]
pub(crate) struct ExecCli {
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
pub(crate) struct ResumeCommand {
    /// 要恢复的 canonical Thread id。
    thread_id: Option<String>,
    #[command(flatten)]
    connection: ConnectionArgs,
    #[arg(long, value_name = "LOCALE")]
    locale: Option<String>,
}

#[derive(Debug, Args)]
pub(crate) struct ThreadCommand {
    #[command(subcommand)]
    command: ThreadSubcommand,
}

#[derive(Debug, Args)]
pub(crate) struct SkillsCli {
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
pub(crate) struct SessionArchiveCommand {
    thread_id: String,
    #[command(flatten)]
    connection: ConnectionArgs,
}

#[derive(Debug, Args)]
pub(crate) struct DeleteCommand {
    thread_id: String,
    #[command(flatten)]
    connection: ConnectionArgs,
}

#[derive(Debug, Args)]
pub(crate) struct ForkCommand {
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

impl Default for TuiCli {
    fn default() -> Self {
        Self {
            connection: ConnectionArgs {
                app_server: None,
                app_server_args: Vec::new(),
                remote: InteractiveRemoteOptions::default(),
                cwd: None,
                model: None,
                provider: None,
                effort: None,
                permissions: None,
            },
            locale: None,
        }
    }
}

pub(crate) async fn run_interactive_tui(args: TuiCli) -> ExitCode {
    let options = match tui_options(args.connection, None, args.locale) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("{error:#}");
            return ExitCode::FAILURE;
        }
    };
    match tui::run_tui(options).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::FAILURE
        }
    }
}

pub(crate) async fn run_resume(args: ResumeCommand) -> ExitCode {
    let options = match tui_options(args.connection, args.thread_id, args.locale) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("{error:#}");
            return ExitCode::FAILURE;
        }
    };
    match tui::run_resume(options).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::FAILURE
        }
    }
}

pub(crate) async fn run_thread(command: ThreadCommand) -> ExitCode {
    report_json_result(run_thread_request(command).await)
}

fn report_json_result(result: Result<serde_json::Value>) -> ExitCode {
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

pub(crate) async fn run_archive(args: SessionArchiveCommand) -> ExitCode {
    run_thread_operation(
        args.connection,
        METHOD_THREAD_ARCHIVE,
        ThreadArchiveParams {
            thread_id: args.thread_id,
        },
    )
    .await
}

pub(crate) async fn run_unarchive(args: SessionArchiveCommand) -> ExitCode {
    run_thread_operation(
        args.connection,
        METHOD_THREAD_UNARCHIVE,
        ThreadUnarchiveParams {
            thread_id: args.thread_id,
        },
    )
    .await
}

pub(crate) async fn run_delete(args: DeleteCommand) -> ExitCode {
    run_thread_operation(
        args.connection,
        METHOD_THREAD_DELETE,
        ThreadDeleteParams {
            thread_id: args.thread_id,
        },
    )
    .await
}

pub(crate) async fn run_fork(args: ForkCommand) -> ExitCode {
    run_thread_operation(
        args.connection,
        METHOD_THREAD_FORK,
        ThreadForkParams {
            thread_id: args.thread_id,
            last_turn_id: args.last_turn_id,
            before_turn_id: args.before_turn_id,
            exclude_turns: args.exclude_turns,
            ..ThreadForkParams::default()
        },
    )
    .await
}

async fn run_thread_operation(
    connection: ConnectionArgs,
    method: &str,
    params: impl serde::Serialize,
) -> ExitCode {
    let result = match serde_json::to_value(params) {
        Ok(params) => request_thread_value(&connection, method, params).await,
        Err(error) => Err(error.into()),
    };
    report_json_result(result)
}

pub(crate) async fn run_skills(command: SkillsCli) -> ExitCode {
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

pub(crate) async fn run_skills_request(command: SkillsCli) -> Result<serde_json::Value> {
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
    let session = start_session(&args.connection).await?;
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
    };

    request_thread_value(&connection, method, params).await
}

async fn request_thread_value(
    connection: &ConnectionArgs,
    method: &str,
    params: serde_json::Value,
) -> Result<serde_json::Value> {
    let value = request_value(connection, method, params).await?;

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

pub(crate) async fn request_value(
    connection: &ConnectionArgs,
    method: &str,
    params: serde_json::Value,
) -> Result<serde_json::Value> {
    let session = start_session(connection).await?;
    let response = session.request_handle().request_value(method, params).await;
    let shutdown = session.shutdown().await;
    let response = response?;
    shutdown?;
    Ok(response)
}

pub(crate) async fn start_session(connection: &ConnectionArgs) -> Result<ClientSession> {
    if let Some(config) = resolve_remote_endpoint(connection)? {
        return Ok(ClientSession::start_remote(config, cli_initialize_params()).await?);
    }
    Ok(
        ClientSession::start_stdio(thread_stdio_config(connection), cli_initialize_params())
            .await?,
    )
}

fn resolve_remote_endpoint(connection: &ConnectionArgs) -> Result<Option<RemoteTransportConfig>> {
    let Some(remote) = connection.remote.remote.clone() else {
        if connection.remote.remote_auth_token_env.is_some() {
            bail!("`--remote-auth-token-env` requires `--remote`.");
        }
        return Ok(None);
    };

    if connection.app_server.is_some() || !connection.app_server_args.is_empty() {
        bail!("`--remote` cannot be combined with `--app-server` or `--app-server-arg`.");
    }

    let auth_token = connection
        .remote
        .remote_auth_token_env
        .as_deref()
        .map(read_remote_auth_token_from_env_var)
        .transpose()?;
    Ok(Some(
        RemoteTransportConfig::new(remote).with_optional_auth_token(auth_token),
    ))
}

fn read_remote_auth_token_from_env_var_with<F>(env_var_name: &str, get_var: F) -> Result<String>
where
    F: FnOnce(&str) -> std::result::Result<String, env::VarError>,
{
    let auth_token = get_var(env_var_name)
        .with_context(|| format!("environment variable `{env_var_name}` is not set"))?;
    let auth_token = auth_token.trim().to_string();
    if auth_token.is_empty() {
        bail!("environment variable `{env_var_name}` is empty");
    }
    Ok(auth_token)
}

fn read_remote_auth_token_from_env_var(env_var_name: &str) -> Result<String> {
    read_remote_auth_token_from_env_var_with(env_var_name, |name| env::var(name))
}

pub(crate) fn thread_stdio_config(connection: &ConnectionArgs) -> StdioTransportConfig {
    let mut config =
        StdioTransportConfig::runtime(resolve_app_server_bin(connection.app_server.clone()));
    config
        .args
        .extend(connection.app_server_args.iter().cloned());
    config
}

pub(crate) fn cli_initialize_params() -> InitializeParams {
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

pub(crate) async fn run_exec(args: ExecCli) -> ExitCode {
    let json_output = args.json;
    let jsonl_output = args.jsonl;
    let result = read_prompt(args.prompt).and_then(|prompt| {
        tui_options(args.connection, None, None).map(|tui| ExecOptions { tui, prompt })
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

fn tui_options(
    args: ConnectionArgs,
    resume_thread: Option<String>,
    locale: Option<String>,
) -> Result<TuiOptions> {
    let remote = resolve_remote_endpoint(&args)?;
    Ok(TuiOptions {
        app_server_bin: resolve_app_server_bin(args.app_server),
        app_server_args: args.app_server_args,
        remote,
        cwd: args
            .cwd
            .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from("."))),
        model: args.model,
        model_provider: args.provider,
        reasoning_effort: args.effort,
        permissions: args.permissions,
        locale,
        resume_thread,
    })
}

pub(crate) fn resolve_app_server_bin(explicit: Option<PathBuf>) -> PathBuf {
    if let Some(path) = explicit {
        #[cfg(not(windows))]
        {
            return PathBuf::from(wsl_paths::normalize_for_wsl(path));
        }
        #[cfg(windows)]
        {
            return path;
        }
    }
    if let Some(path) = env::var_os(APP_SERVER_BIN_ENV).filter(|value| !value.is_empty()) {
        #[cfg(not(windows))]
        {
            return PathBuf::from(wsl_paths::normalize_for_wsl(path));
        }
        #[cfg(windows)]
        {
            return PathBuf::from(path);
        }
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

#[derive(Debug, Args)]
#[command(disable_help_flag = true)]
pub(crate) struct AppServerCommand {
    /// Override the sibling app-server binary. Intended for packaging and tests.
    #[arg(long = "app-server-bin", value_name = "PATH", hide = true)]
    app_server_bin: Option<PathBuf>,

    /// Arguments forwarded unchanged to app-server.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    args: Vec<OsString>,
}

pub(crate) fn run(command: AppServerCommand) -> ExitCode {
    let app_server_bin = resolve_app_server_bin(command.app_server_bin);
    let mut child = Command::new(app_server_bin);
    child.args(command.args);

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;

        let error = child.exec();
        eprintln!("failed to start app-server: {error}");
        ExitCode::FAILURE
    }

    #[cfg(not(unix))]
    {
        match child.status() {
            Ok(status) if status.success() => ExitCode::SUCCESS,
            Ok(status) => ExitCode::from(
                status
                    .code()
                    .and_then(|code| u8::try_from(code).ok())
                    .unwrap_or(1),
            ),
            Err(error) => {
                eprintln!("failed to start app-server: {error}");
                ExitCode::FAILURE
            }
        }
    }
}

#[derive(Debug, Args)]
pub(crate) struct DebugCommand {
    #[command(subcommand)]
    subcommand: DebugSubcommand,
}

#[derive(Debug, clap::Subcommand)]
enum DebugSubcommand {
    /// Render the current App Server model catalog as JSON.
    Models(DebugModelsCommand),
    /// Reset global memory state through App Server.
    #[command(hide = true)]
    ClearMemories(DebugClearMemoriesCommand),
}

#[derive(Debug, Args)]
struct DebugModelsCommand {
    /// Include hidden models from the App Server catalog.
    #[arg(long)]
    bundled: bool,
    #[command(flatten)]
    connection: ConnectionArgs,
}

#[derive(Debug, Args)]
struct DebugClearMemoriesCommand {
    #[command(flatten)]
    connection: ConnectionArgs,
}

pub(crate) async fn run_debug_command(command: DebugCommand) -> ExitCode {
    match run_inner(command).await {
        Ok(output) => {
            println!("{output}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run_inner(command: DebugCommand) -> Result<String> {
    match command.subcommand {
        DebugSubcommand::Models(args) => run_debug_models_command(args).await,
        DebugSubcommand::ClearMemories(args) => run_debug_clear_memories_command(args).await,
    }
}

async fn run_debug_models_command(command: DebugModelsCommand) -> Result<String> {
    let models = list_models(&command.connection, command.bundled).await?;
    Ok(serde_json::to_string_pretty(
        &serde_json::json!({ "models": models }),
    )?)
}

async fn run_debug_clear_memories_command(command: DebugClearMemoriesCommand) -> Result<String> {
    let response = request_value(
        &command.connection,
        METHOD_MEMORY_RESET,
        serde_json::Value::Null,
    )
    .await?;
    let _: MemoryResetResponse = serde_json::from_value(response)?;
    Ok("Cleared memory state.".to_string())
}

async fn list_models(connection: &ConnectionArgs, include_hidden: bool) -> Result<Vec<Model>> {
    let mut cursor = None;
    let mut models = Vec::new();
    let mut seen_cursors = HashSet::new();
    for _ in 0..256 {
        let response = request_value(
            connection,
            METHOD_MODEL_LIST,
            serde_json::to_value(ModelListParams {
                cursor,
                limit: None,
                include_hidden: Some(include_hidden),
            })?,
        )
        .await?;
        let response: ModelListResponse = serde_json::from_value(response)?;
        models.extend(response.data);
        let Some(next_cursor) = response.next_cursor else {
            return Ok(models);
        };
        if !seen_cursors.insert(next_cursor.clone()) {
            bail!("model pagination repeated cursor {next_cursor}");
        }
        cursor = Some(next_cursor);
    }
    bail!("model pagination exceeded 256 pages")
}

#[derive(Debug, Args)]
pub(crate) struct FeaturesCli {
    #[command(subcommand)]
    sub: FeaturesSubcommand,
}

#[derive(Debug, clap::Subcommand)]
enum FeaturesSubcommand {
    /// List known features with their stage and effective state.
    List(FeatureListArgs),
    /// Enable a feature in the current Lime configuration.
    Enable(FeatureSetArgs),
    /// Disable a feature in the current Lime configuration.
    Disable(FeatureSetArgs),
}

#[derive(Debug, Args)]
struct FeatureListArgs {
    /// Emit the complete typed response as JSON.
    #[arg(long)]
    json: bool,
    #[command(flatten)]
    connection: ConnectionArgs,
}

#[derive(Debug, Args)]
struct FeatureSetArgs {
    /// Feature key to update.
    feature: String,
    #[command(flatten)]
    connection: ConnectionArgs,
}

pub(crate) async fn run_features_command(command: FeaturesCli) -> ExitCode {
    match run_features_inner(command).await {
        Ok(output) => {
            println!("{output}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run_features_inner(command: FeaturesCli) -> Result<String> {
    match command.sub {
        FeaturesSubcommand::List(args) => {
            let features = list_features(&args.connection).await?;
            if args.json {
                return Ok(serde_json::to_string_pretty(&serde_json::json!({
                    "data": features,
                    "nextCursor": null,
                }))?);
            }
            Ok(render_features(features))
        }
        FeaturesSubcommand::Enable(args) => set_feature(args, true).await,
        FeaturesSubcommand::Disable(args) => set_feature(args, false).await,
    }
}

async fn set_feature(args: FeatureSetArgs, enabled: bool) -> Result<String> {
    let features = list_features(&args.connection).await?;
    let Some(feature) = features.iter().find(|feature| feature.name == args.feature) else {
        bail!("Unknown feature flag: {}", args.feature);
    };
    let stage = feature.stage;
    let feature_name = feature.name.clone();
    let response = request_value(
        &args.connection,
        METHOD_EXPERIMENTAL_FEATURE_ENABLEMENT_SET,
        serde_json::to_value(ExperimentalFeatureEnablementSetParams {
            enablement: BTreeMap::from([(feature_name.clone(), enabled)]),
        })?,
    )
    .await?;
    let response: ExperimentalFeatureEnablementSetResponse = serde_json::from_value(response)?;
    if response.enablement.get(&feature_name) != Some(&enabled) {
        bail!("App Server did not apply feature flag: {feature_name}");
    }
    if enabled && stage == ExperimentalFeatureStage::UnderDevelopment {
        eprintln!("Under-development features enabled: {feature_name}.");
    }
    Ok(format!(
        "{} feature `{feature_name}`.",
        if enabled { "Enabled" } else { "Disabled" }
    ))
}

async fn list_features(connection: &ConnectionArgs) -> Result<Vec<ExperimentalFeature>> {
    let mut cursor = None;
    let mut features = Vec::new();
    let mut seen_cursors = HashSet::new();
    for _ in 0..256 {
        let response = request_value(
            connection,
            METHOD_EXPERIMENTAL_FEATURE_LIST,
            serde_json::to_value(ExperimentalFeatureListParams {
                cursor,
                limit: None,
                thread_id: None,
            })?,
        )
        .await?;
        let response: ExperimentalFeatureListResponse = serde_json::from_value(response)?;
        features.extend(response.data);
        let Some(next_cursor) = response.next_cursor else {
            features.sort_by(|left, right| left.name.cmp(&right.name));
            return Ok(features);
        };
        if !seen_cursors.insert(next_cursor.clone()) {
            bail!("feature pagination repeated cursor {next_cursor}");
        }
        cursor = Some(next_cursor);
    }
    bail!("feature pagination exceeded 256 pages")
}

fn render_features(features: Vec<ExperimentalFeature>) -> String {
    let name_width = features
        .iter()
        .map(|feature| feature.name.len())
        .max()
        .unwrap_or_default();
    features
        .into_iter()
        .map(|feature| {
            format!(
                "{:<name_width$}  {:<17}  {}",
                feature.name,
                stage_str(feature.stage),
                feature.enabled,
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn stage_str(stage: ExperimentalFeatureStage) -> &'static str {
    match stage {
        ExperimentalFeatureStage::Beta => "beta",
        ExperimentalFeatureStage::UnderDevelopment => "under development",
        ExperimentalFeatureStage::Stable => "stable",
        ExperimentalFeatureStage::Deprecated => "deprecated",
        ExperimentalFeatureStage::Removed => "removed",
    }
}

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
struct MultitoolCli {
    #[command(flatten)]
    interactive: TuiCli,

    #[command(subcommand)]
    subcommand: Option<Subcommand>,
}

#[derive(Debug, Parser)]
struct ExecpolicyCommand {
    #[command(subcommand)]
    sub: ExecpolicySubcommand,
}

#[derive(Debug, ClapSubcommand)]
enum ExecpolicySubcommand {
    /// Check execpolicy files against a command.
    #[command(name = "check")]
    Check(ExecPolicyCheckCommand),
}

#[derive(Debug, ClapSubcommand)]
enum Subcommand {
    /// Start the interactive TUI.
    Tui(TuiCli),

    /// Run Lime non-interactively.
    #[command(visible_alias = "e")]
    Exec(ExecCli),

    /// Resume a previous interactive thread.
    Resume(ResumeCommand),

    /// Queue a message for an existing Thread.
    Queue(queue_cmd::QueueCommand),

    /// Archive a saved Thread by id.
    Archive(SessionArchiveCommand),

    /// Permanently delete a saved Thread by id.
    Delete(DeleteCommand),

    /// Unarchive a saved Thread by id.
    Unarchive(SessionArchiveCommand),

    /// Fork a saved Thread by id.
    Fork(ForkCommand),

    /// Manage canonical threads through App Server.
    Thread(ThreadCommand),

    /// Inspect MCP server status through App Server.
    Mcp(mcp_cmd::McpCli),

    /// Inspect executable skills through App Server.
    Skills(SkillsCli),

    /// Manage plugins through App Server.
    Plugin(plugin_cmd::PluginCli),

    /// Start the sibling App Server with forwarded arguments.
    #[command(name = "app-server")]
    AppServer(AppServerCommand),

    /// Debug current App Server state.
    Debug(DebugCommand),

    /// Inspect and update experimental features.
    Features(FeaturesCli),

    /// Run commands within the host sandbox.
    Sandbox(HostSandboxArgs),

    /// Check execpolicy files against a command.
    #[command(name = "execpolicy")]
    Execpolicy(ExecpolicyCommand),

    /// Generate shell completion for the canonical `lime` command tree.
    Completion(CompletionCommand),
}

#[derive(Debug, clap::Args)]
struct CompletionCommand {
    #[arg(value_enum)]
    shell: Shell,
}

fn main() -> ExitCode {
    let cli = MultitoolCli::parse();
    run_async(cli_main(cli))
}

async fn cli_main(cli: MultitoolCli) -> ExitCode {
    match cli.subcommand {
        None => run_interactive_tui(cli.interactive).await,
        Some(Subcommand::Tui(args)) => run_interactive_tui(args).await,
        Some(Subcommand::Exec(args)) => run_exec(args).await,
        Some(Subcommand::Resume(args)) => run_resume(args).await,
        Some(Subcommand::Queue(args)) => queue_cmd::run_queue_command(args).await,
        Some(Subcommand::Archive(args)) => run_archive(args).await,
        Some(Subcommand::Delete(args)) => run_delete(args).await,
        Some(Subcommand::Unarchive(args)) => run_unarchive(args).await,
        Some(Subcommand::Fork(args)) => run_fork(args).await,
        Some(Subcommand::Thread(args)) => run_thread(args).await,
        Some(Subcommand::Mcp(args)) => args.run().await,
        Some(Subcommand::Skills(args)) => run_skills(args).await,
        Some(Subcommand::Plugin(args)) => args.run().await,
        Some(Subcommand::AppServer(args)) => run(args),
        Some(Subcommand::Debug(args)) => run_debug_command(args).await,
        Some(Subcommand::Features(args)) => run_features_command(args).await,
        Some(Subcommand::Sandbox(sandbox_cli)) => {
            let setup_command = sandbox_setup::parse_setup_command(&sandbox_cli.command);
            let result = match setup_command {
                Ok(Some(setup_cli)) => {
                    sandbox_setup::run(setup_cli, sandbox_cli.config_profile.clone())
                        .await
                        .map(|()| ExitCode::SUCCESS)
                }
                Ok(None) => {
                    #[cfg(target_os = "macos")]
                    let result = cli::run_command_under_seatbelt(sandbox_cli).await;
                    #[cfg(target_os = "linux")]
                    let result = cli::run_command_under_landlock(sandbox_cli).await;
                    #[cfg(target_os = "windows")]
                    let result = cli::run_command_under_windows_sandbox(sandbox_cli).await;
                    #[cfg(not(any(
                        target_os = "macos",
                        target_os = "linux",
                        target_os = "windows"
                    )))]
                    let result: anyhow::Result<ExitCode> = {
                        let _ = sandbox_cli.command;
                        Err(anyhow::anyhow!(
                            "`lime sandbox` is not supported on this operating system"
                        ))
                    };
                    result
                }
                Err(error) => Err(error),
            };

            match result {
                Ok(exit_code) => exit_code,
                Err(error) => {
                    eprintln!("{error:#}");
                    ExitCode::FAILURE
                }
            }
        }
        Some(Subcommand::Execpolicy(ExecpolicyCommand { sub })) => match sub {
            ExecpolicySubcommand::Check(command) => run_execpolicycheck(command),
        },
        Some(Subcommand::Completion(args)) => {
            let mut command = MultitoolCli::command();
            let mut output = io::stdout();
            generate(args.shell, &mut command, "lime", &mut output);
            let _ = output.flush();
            ExitCode::SUCCESS
        }
    }
}

#[cfg(target_os = "macos")]
type HostSandboxArgs = cli::SeatbeltCommand;
#[cfg(target_os = "linux")]
type HostSandboxArgs = cli::LandlockCommand;
#[cfg(target_os = "windows")]
type HostSandboxArgs = cli::WindowsCommand;

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
#[derive(Debug, clap::Args)]
struct HostSandboxArgs {
    #[arg(trailing_var_arg = true, required = true)]
    command: Vec<String>,
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

fn run_execpolicycheck(command: ExecPolicyCheckCommand) -> ExitCode {
    match command.run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod command_tests {
    use std::ffi::OsString;
    use std::path::PathBuf;

    use clap::Parser;
    use serde_json::json;

    use super::*;

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
        let cli = crate::MultitoolCli::try_parse_from([
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
            ":workspace",
            "--app-server-arg=--backend",
            "--app-server-arg=external",
        ])
        .expect("parse exec");
        let Some(crate::Subcommand::Exec(args)) = cli.subcommand else {
            panic!("expected exec command");
        };

        assert_eq!(args.prompt, vec!["review", "this diff"]);
        assert!(args.json);
        assert!(!args.jsonl);
        assert_eq!(args.connection.cwd, Some(PathBuf::from("/tmp/worktree")));
        assert_eq!(args.connection.model.as_deref(), Some("gpt-test"));
        assert_eq!(args.connection.provider.as_deref(), Some("openai-test"));
        assert_eq!(args.connection.effort.as_deref(), Some("high"));
        assert_eq!(args.connection.permissions.as_deref(), Some(":workspace"));
        assert_eq!(
            args.connection.app_server_args,
            vec![OsString::from("--backend"), OsString::from("external")]
        );
    }

    #[test]
    fn remote_connection_flags_match_codex_shape() {
        let cli = crate::MultitoolCli::try_parse_from([
            "lime",
            "exec",
            "check",
            "--remote",
            "wss://cloud.example/rpc",
            "--remote-auth-token-env",
            "LIME_REMOTE_TOKEN",
        ])
        .expect("parse remote exec");
        let Some(crate::Subcommand::Exec(args)) = cli.subcommand else {
            panic!("expected exec command");
        };
        assert_eq!(
            args.connection.remote.remote.as_deref(),
            Some("wss://cloud.example/rpc")
        );
        assert_eq!(
            args.connection.remote.remote_auth_token_env.as_deref(),
            Some("LIME_REMOTE_TOKEN")
        );
    }

    #[test]
    fn remote_auth_token_env_requires_remote_endpoint() {
        let connection = ConnectionArgs {
            app_server: None,
            app_server_args: Vec::new(),
            remote: InteractiveRemoteOptions {
                remote: None,
                remote_auth_token_env: Some("LIME_REMOTE_TOKEN".to_string()),
            },
            cwd: None,
            model: None,
            provider: None,
            effort: None,
            permissions: None,
        };
        let error = resolve_remote_endpoint(&connection)
            .expect_err("remote auth without endpoint must fail");
        assert!(error.to_string().contains("requires `--remote`"));
    }

    #[test]
    fn remote_auth_token_env_missing_fails_before_connecting() {
        let connection = ConnectionArgs {
            app_server: None,
            app_server_args: Vec::new(),
            remote: InteractiveRemoteOptions {
                remote: Some("wss://cloud.example/rpc".to_string()),
                remote_auth_token_env: Some("LIME_REMOTE_TOKEN_MISSING_FOR_TEST_9A7C".to_string()),
            },
            cwd: None,
            model: None,
            provider: None,
            effort: None,
            permissions: None,
        };
        let error = resolve_remote_endpoint(&connection)
            .expect_err("missing remote auth env must fail before transport startup");
        assert!(error.to_string().contains("is not set"));
    }

    #[test]
    fn tui_options_lower_remote_url_without_local_process_settings() {
        let connection = ConnectionArgs {
            app_server: None,
            app_server_args: Vec::new(),
            remote: InteractiveRemoteOptions {
                remote: Some("ws://127.0.0.1:4500/rpc".to_string()),
                remote_auth_token_env: None,
            },
            cwd: Some(PathBuf::from("/tmp/worktree")),
            model: None,
            provider: None,
            effort: None,
            permissions: None,
        };
        let options = tui_options(connection, None, None).expect("remote options");
        assert_eq!(
            options
                .remote
                .as_ref()
                .map(|config| config.websocket_url.as_str()),
            Some("ws://127.0.0.1:4500/rpc")
        );
        assert!(options.remote.as_ref().unwrap().auth_token.is_none());
    }

    #[test]
    fn remote_connection_rejects_local_app_server_overrides() {
        let connection = ConnectionArgs {
            app_server: Some(PathBuf::from("app-server")),
            app_server_args: Vec::new(),
            remote: InteractiveRemoteOptions {
                remote: Some("wss://cloud.example/rpc".to_string()),
                remote_auth_token_env: None,
            },
            cwd: None,
            model: None,
            provider: None,
            effort: None,
            permissions: None,
        };
        let error = resolve_remote_endpoint(&connection)
            .expect_err("remote and local app-server overrides must be exclusive");
        assert!(error.to_string().contains("cannot be combined"));
    }

    #[test]
    fn read_remote_auth_token_from_env_var_trims_values() {
        let token = read_remote_auth_token_from_env_var_with("LIME_REMOTE_TOKEN", |_| {
            Ok("  bearer-secret  ".to_string())
        })
        .expect("trim remote token");
        assert_eq!(token, "bearer-secret");
    }

    #[test]
    fn read_remote_auth_token_from_env_var_reports_missing_values() {
        let error = read_remote_auth_token_from_env_var_with("LIME_REMOTE_TOKEN", |_| {
            Err(env::VarError::NotPresent)
        })
        .expect_err("missing token");
        assert!(error.to_string().contains("is not set"));
    }

    #[test]
    fn read_remote_auth_token_from_env_var_rejects_empty_values() {
        let error = read_remote_auth_token_from_env_var_with("LIME_REMOTE_TOKEN", |_| {
            Ok("  \n".to_string())
        })
        .expect_err("empty token");
        assert!(error.to_string().contains("is empty"));
    }

    #[test]
    fn exec_parser_supports_jsonl_as_a_mutually_exclusive_format() {
        let cli = crate::MultitoolCli::try_parse_from(["lime", "exec", "check", "--jsonl"])
            .expect("parse exec");
        let Some(crate::Subcommand::Exec(args)) = cli.subcommand else {
            panic!("expected exec command");
        };
        assert!(args.jsonl);
        assert!(!args.json);

        let error =
            crate::MultitoolCli::try_parse_from(["lime", "exec", "check", "--json", "--jsonl"])
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
        let cli = crate::MultitoolCli::try_parse_from([
            "lime",
            "resume",
            "thread-42",
            "--cwd",
            "/tmp/worktree",
            "--app-server",
            "/tmp/app-server",
            "--locale",
            "zh-CN",
        ])
        .expect("parse resume");
        let Some(crate::Subcommand::Resume(args)) = cli.subcommand else {
            panic!("expected resume command");
        };

        assert_eq!(args.thread_id.as_deref(), Some("thread-42"));
        assert_eq!(args.locale.as_deref(), Some("zh-CN"));
        assert_eq!(args.connection.cwd, Some(PathBuf::from("/tmp/worktree")));
        assert_eq!(
            args.connection.app_server,
            Some(PathBuf::from("/tmp/app-server"))
        );
    }

    #[test]
    fn resume_without_id_selects_from_the_tui_session_picker() {
        let cli =
            crate::MultitoolCli::try_parse_from(["lime", "resume"]).expect("parse resume picker");
        let Some(crate::Subcommand::Resume(args)) = cli.subcommand else {
            panic!("expected resume command");
        };
        assert!(args.thread_id.is_none());
    }

    #[test]
    fn top_level_fork_keeps_v2_identity_and_history_flags() {
        let cli = crate::MultitoolCli::try_parse_from([
            "lime",
            "fork",
            "thread-42",
            "--last-turn-id",
            "turn-7",
            "--exclude-turns",
        ])
        .expect("parse top-level fork");
        let Some(crate::Subcommand::Fork(args)) = cli.subcommand else {
            panic!("expected fork command");
        };

        assert_eq!(args.thread_id, "thread-42");
        assert_eq!(args.last_turn_id.as_deref(), Some("turn-7"));
        assert!(args.exclude_turns);
    }

    #[test]
    fn mcp_list_parser_matches_codex_command_shape() {
        let cli = crate::MultitoolCli::try_parse_from([
            "lime",
            "mcp",
            "list",
            "--app-server",
            "/tmp/app-server",
        ])
        .expect("parse mcp list");
        let Some(crate::Subcommand::Mcp(command)) = cli.subcommand else {
            panic!("expected mcp command");
        };
        let crate::mcp_cmd::McpSubcommand::List(args) = command.subcommand else {
            panic!("expected mcp list command");
        };
        assert!(!args.json);
        assert_eq!(
            args.connection.app_server,
            Some(PathBuf::from("/tmp/app-server"))
        );
    }

    #[test]
    fn skills_list_parser_accepts_multiple_cwds_and_force_reload() {
        let cli = crate::MultitoolCli::try_parse_from([
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
        let Some(crate::Subcommand::Skills(command)) = cli.subcommand else {
            panic!("expected skills command");
        };
        let SkillsSubcommand::List(args) = command.command;
        assert_eq!(
            args.skill_cwds,
            vec![PathBuf::from("/tmp/one"), PathBuf::from("/tmp/two")]
        );
        assert!(args.force_reload);
    }

    #[test]
    fn execpolicy_check_parser_matches_codex_command_shape() {
        let cli = crate::MultitoolCli::try_parse_from([
            "lime",
            "execpolicy",
            "check",
            "--rules",
            "/tmp/policy.rules",
            "--pretty",
            "git",
            "push",
            "origin",
            "main",
        ])
        .expect("parse execpolicy check");
        let Some(crate::Subcommand::Execpolicy(command)) = cli.subcommand else {
            panic!("expected execpolicy command");
        };
        let crate::ExecpolicySubcommand::Check(check) = command.sub;
        assert_eq!(check.rules, [PathBuf::from("/tmp/policy.rules")]);
        assert!(check.pretty);
        assert_eq!(check.command, ["git", "push", "origin", "main"]);
    }
}

#[cfg(test)]
mod app_server_tests {
    use clap::Parser;

    #[test]
    fn app_server_forwards_help_and_transport_flags() {
        let cli = crate::MultitoolCli::try_parse_from([
            "lime",
            "app-server",
            "--help",
            "--listen",
            "ws://127.0.0.1:4512",
        ])
        .expect("parse app-server passthrough");
        let Some(crate::Subcommand::AppServer(command)) = cli.subcommand else {
            panic!("expected app-server command");
        };
        assert_eq!(
            command.args,
            ["--help", "--listen", "ws://127.0.0.1:4512"].map(std::ffi::OsString::from)
        );
    }
}

#[cfg(test)]
mod debug_tests {
    use clap::Parser;

    #[test]
    fn debug_models_parses_bundled_flag() {
        let cli = crate::MultitoolCli::try_parse_from(["lime", "debug", "models", "--bundled"])
            .expect("parse debug models");
        assert!(matches!(cli.subcommand, Some(crate::Subcommand::Debug(_))));
    }

    #[test]
    fn debug_clear_memories_is_registered() {
        let cli = crate::MultitoolCli::try_parse_from(["lime", "debug", "clear-memories"])
            .expect("parse debug clear-memories");
        assert!(matches!(cli.subcommand, Some(crate::Subcommand::Debug(_))));
    }
}

#[cfg(test)]
mod features_tests {
    use app_server_protocol::protocol::v2::ExperimentalFeature;
    use clap::Parser;

    use super::*;

    #[test]
    fn features_enable_and_disable_parse_feature_name() {
        for action in ["enable", "disable"] {
            let cli = crate::MultitoolCli::try_parse_from(["lime", "features", action, "webmcp"])
                .expect("parse feature toggle");
            assert!(matches!(
                cli.subcommand,
                Some(crate::Subcommand::Features(_))
            ));
        }
    }

    #[test]
    fn feature_list_is_sorted_alphabetically() {
        let feature = |name: &str| ExperimentalFeature {
            name: name.to_string(),
            stage: ExperimentalFeatureStage::Stable,
            display_name: None,
            description: None,
            announcement: None,
            enabled: false,
            default_enabled: false,
        };
        let mut features = vec![feature("zeta"), feature("alpha")];
        features.sort_by(|left, right| left.name.cmp(&right.name));
        let output = render_features(features);
        assert_eq!(
            output
                .lines()
                .map(|line| line.split_whitespace().next().unwrap())
                .collect::<Vec<_>>(),
            vec!["alpha", "zeta"]
        );
    }
}
