//! MCP CLI commands.
//!
//! The command names and argument types intentionally follow Codex's
//! `mcp_cmd.rs`. Configuration mutations still go through App Server JSON-RPC
//! so Desktop, CLI and TUI share one MCP owner.

use std::collections::HashMap;
use std::process::ExitCode;

use anyhow::{bail, Result};
use app_server_protocol::{
    McpServerCreateParams, McpServerDeleteParams, McpServerListResponse, McpServerOauthLoginParams,
    McpServerOauthLoginResponse, McpServerStartParams, McpServerStopParams,
    METHOD_MCP_SERVER_CREATE, METHOD_MCP_SERVER_DELETE, METHOD_MCP_SERVER_LIST,
    METHOD_MCP_SERVER_OAUTH_LOGIN, METHOD_MCP_SERVER_START, METHOD_MCP_SERVER_STOP,
};
use clap::Args;
use serde_json::{json, Value};

use crate::ConnectionArgs;

#[derive(Debug, Args)]
pub(crate) struct McpCli {
    #[command(subcommand)]
    pub(crate) subcommand: McpSubcommand,
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum McpSubcommand {
    List(ListArgs),
    Get(GetArgs),
    Add(AddArgs),
    Remove(RemoveArgs),
    Start(ServerNameArgs),
    Stop(ServerNameArgs),
    Login(LoginArgs),
    Logout(LogoutArgs),
}

#[derive(Debug, Args)]
pub(crate) struct ListArgs {
    #[arg(long)]
    pub(crate) json: bool,
    #[command(flatten)]
    pub(crate) connection: ConnectionArgs,
}

#[derive(Debug, Args)]
pub(crate) struct GetArgs {
    pub(crate) name: String,
    #[arg(long)]
    pub(crate) json: bool,
    #[command(flatten)]
    pub(crate) connection: ConnectionArgs,
}

#[derive(Debug, Args)]
#[command(override_usage = "lime mcp add [OPTIONS] <NAME> (--url <URL> | -- <COMMAND>...)")]
pub(crate) struct AddArgs {
    pub(crate) name: String,
    #[command(flatten)]
    pub(crate) transport_args: AddMcpTransportArgs,
    #[command(flatten)]
    pub(crate) connection: ConnectionArgs,
}

#[derive(Debug, Args)]
#[command(group(
    clap::ArgGroup::new("transport")
        .args(["command", "url"])
        .required(true)
        .multiple(false)
))]
pub(crate) struct AddMcpTransportArgs {
    #[command(flatten)]
    pub(crate) stdio: Option<AddMcpStdioArgs>,
    #[command(flatten)]
    pub(crate) streamable_http: Option<AddMcpStreamableHttpArgs>,
}

#[derive(Debug, Args)]
pub(crate) struct AddMcpStdioArgs {
    #[arg(trailing_var_arg = true, num_args = 1..)]
    pub(crate) command: Vec<String>,
    #[arg(long, value_parser = parse_env_pair, value_name = "KEY=VALUE")]
    pub(crate) env: Vec<(String, String)>,
}

#[derive(Debug, Args)]
pub(crate) struct AddMcpStreamableHttpArgs {
    #[arg(long)]
    pub(crate) url: String,
    #[arg(long = "bearer-token-env-var", value_name = "ENV_VAR")]
    pub(crate) bearer_token_env_var: Option<String>,
}

#[derive(Debug, Args)]
pub(crate) struct RemoveArgs {
    pub(crate) name: String,
    #[command(flatten)]
    pub(crate) connection: ConnectionArgs,
}

#[derive(Debug, Args)]
pub(crate) struct ServerNameArgs {
    pub(crate) name: String,
    #[command(flatten)]
    pub(crate) connection: ConnectionArgs,
}

#[derive(Debug, Args)]
pub(crate) struct LoginArgs {
    pub(crate) name: String,
    #[arg(long, value_delimiter = ',', value_name = "SCOPE,SCOPE")]
    pub(crate) scopes: Vec<String>,
    #[command(flatten)]
    pub(crate) connection: ConnectionArgs,
}

#[derive(Debug, Args)]
pub(crate) struct LogoutArgs {
    pub(crate) name: String,
    #[command(flatten)]
    pub(crate) connection: ConnectionArgs,
}

impl McpCli {
    pub(crate) async fn run(self) -> ExitCode {
        match run_inner(self).await {
            Ok(output) => {
                if !output.is_empty() {
                    println!("{output}");
                }
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("{error:#}");
                ExitCode::FAILURE
            }
        }
    }
}

async fn run_inner(command: McpCli) -> Result<String> {
    match command.subcommand {
        McpSubcommand::List(args) => run_list(args).await,
        McpSubcommand::Get(args) => run_get(args).await,
        McpSubcommand::Add(args) => run_add(args).await,
        McpSubcommand::Remove(args) => run_remove(args).await,
        McpSubcommand::Start(args) => run_lifecycle(args, METHOD_MCP_SERVER_START).await,
        McpSubcommand::Stop(args) => run_lifecycle(args, METHOD_MCP_SERVER_STOP).await,
        McpSubcommand::Login(args) => run_login(args).await,
        McpSubcommand::Logout(args) => run_logout(args).await,
    }
}

async fn run_list(args: ListArgs) -> Result<String> {
    let response = request(&args.connection, METHOD_MCP_SERVER_LIST, json!({})).await?;
    let response: McpServerListResponse = serde_json::from_value(response)?;
    if args.json {
        return Ok(serde_json::to_string_pretty(&response)?);
    }
    if response.servers.is_empty() {
        return Ok("No MCP servers configured.".to_string());
    }
    let mut servers = response.servers;
    servers.sort_by_key(server_name);
    let mut output = vec!["NAME  STATUS  TRANSPORT".to_string()];
    output.extend(servers.iter().map(|server| {
        let enabled = server
            .get("enabled_lime")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let transport = server
            .get("server_config")
            .and_then(|config| config.get("type"))
            .and_then(Value::as_str)
            .unwrap_or("stdio");
        format!(
            "{}  {}  {transport}",
            server_name(server),
            if enabled { "enabled" } else { "disabled" }
        )
    }));
    Ok(output.join("\n"))
}

async fn run_get(args: GetArgs) -> Result<String> {
    let response = request(&args.connection, METHOD_MCP_SERVER_LIST, json!({})).await?;
    let response: McpServerListResponse = serde_json::from_value(response)?;
    let Some(server) = response
        .servers
        .into_iter()
        .find(|server| server_name(server) == args.name)
    else {
        bail!("No MCP server named '{}' found.", args.name);
    };
    if args.json {
        return Ok(serde_json::to_string_pretty(&server)?);
    }
    Ok(render_server(&server))
}

async fn run_add(args: AddArgs) -> Result<String> {
    validate_server_name(&args.name)?;
    let server_config = match args.transport_args {
        AddMcpTransportArgs {
            stdio: Some(stdio), ..
        } => {
            let mut command = stdio.command.into_iter();
            let command_bin = command
                .next()
                .ok_or_else(|| anyhow::anyhow!("command is required"))?;
            let command_args = command.collect::<Vec<_>>();
            let env: HashMap<_, _> = stdio.env.into_iter().collect();
            json!({ "command": command_bin, "args": command_args, "env": env })
        }
        AddMcpTransportArgs {
            streamable_http: Some(http),
            ..
        } => json!({
            "type": "streamable_http",
            "url": http.url,
            "bearer_token_env_var": http.bearer_token_env_var,
        }),
        AddMcpTransportArgs { .. } => bail!("exactly one MCP transport is required"),
    };
    let server = json!({
        "id": uuid::Uuid::new_v4().to_string(),
        "name": args.name,
        "description": "",
        "server_config": server_config,
        "enabled_lime": true,
        "enabled_claude": false,
        "enabled_codex": false,
        "enabled_gemini": false,
        "created_at": 0,
    });
    let response = request(
        &args.connection,
        METHOD_MCP_SERVER_CREATE,
        serde_json::to_value(McpServerCreateParams { server })?,
    )
    .await?;
    let _: McpServerListResponse = serde_json::from_value(response)?;
    Ok(format!("Added MCP server `{}`.", args.name))
}

async fn run_remove(args: RemoveArgs) -> Result<String> {
    let listed = request(&args.connection, METHOD_MCP_SERVER_LIST, json!({})).await?;
    let listed: McpServerListResponse = serde_json::from_value(listed)?;
    let id = listed
        .servers
        .iter()
        .find(|server| server_name(server) == args.name)
        .and_then(|server| server.get("id"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("No MCP server named '{}' found.", args.name))?;
    let response = request(
        &args.connection,
        METHOD_MCP_SERVER_DELETE,
        serde_json::to_value(McpServerDeleteParams { id })?,
    )
    .await?;
    let _: McpServerListResponse = serde_json::from_value(response)?;
    Ok(format!("Removed MCP server `{}`.", args.name))
}

async fn run_lifecycle(args: ServerNameArgs, method: &str) -> Result<String> {
    let params = if method == METHOD_MCP_SERVER_START {
        serde_json::to_value(McpServerStartParams {
            name: args.name.clone(),
        })?
    } else {
        serde_json::to_value(McpServerStopParams {
            name: args.name.clone(),
        })?
    };
    request(&args.connection, method, params).await?;
    Ok(format!(
        "{} MCP server `{}`.",
        if method == METHOD_MCP_SERVER_START {
            "Started"
        } else {
            "Stopped"
        },
        args.name
    ))
}

async fn run_login(args: LoginArgs) -> Result<String> {
    let response = request(
        &args.connection,
        METHOD_MCP_SERVER_OAUTH_LOGIN,
        serde_json::to_value(McpServerOauthLoginParams {
            name: args.name.clone(),
            scopes: (!args.scopes.is_empty()).then_some(args.scopes),
            timeout_secs: None,
        })?,
    )
    .await?;
    let response: McpServerOauthLoginResponse = serde_json::from_value(response)?;
    Ok(format!(
        "OAuth login started for `{}`. Open: {}",
        args.name, response.authorization_url
    ))
}

async fn run_logout(_args: LogoutArgs) -> Result<String> {
    bail!("MCP OAuth logout is not exposed by the current App Server protocol; refusing to delete credentials from the CLI")
}

async fn request(connection: &ConnectionArgs, method: &str, params: Value) -> Result<Value> {
    crate::request_value(connection, method, params).await
}

fn server_name(server: &Value) -> String {
    server
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("<unnamed>")
        .to_string()
}

fn render_server(server: &Value) -> String {
    format!(
        "{}\n  enabled: {}\n  config: {}",
        server_name(server),
        server
            .get("enabled_lime")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        server.get("server_config").unwrap_or(&Value::Null)
    )
}

fn parse_env_pair(raw: &str) -> Result<(String, String), String> {
    let Some((key, value)) = raw.split_once('=') else {
        return Err("environment entries must be in KEY=VALUE form".to_string());
    };
    if key.trim().is_empty() {
        return Err("environment entries must have a non-empty key".to_string());
    }
    Ok((key.trim().to_string(), value.to_string()))
}

fn validate_server_name(name: &str) -> Result<()> {
    if !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | ':' | '@' | '/' | '.'))
    {
        Ok(())
    } else {
        bail!("invalid MCP server name `{name}`")
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    #[test]
    fn mcp_cli_matches_codex_subcommand_names() {
        for args in [
            ["lime", "mcp", "list"].as_slice(),
            ["lime", "mcp", "get", "docs"].as_slice(),
            ["lime", "mcp", "add", "docs", "--", "node", "server.js"].as_slice(),
            ["lime", "mcp", "remove", "docs"].as_slice(),
            ["lime", "mcp", "login", "docs"].as_slice(),
            ["lime", "mcp", "logout", "docs"].as_slice(),
        ] {
            let cli = crate::MultitoolCli::try_parse_from(args).expect("parse MCP command");
            assert!(matches!(cli.subcommand, Some(crate::Subcommand::Mcp(_))));
        }
    }

    #[test]
    fn mcp_add_rejects_missing_transport() {
        let error = crate::MultitoolCli::try_parse_from(["lime", "mcp", "add", "docs"])
            .expect_err("transport is required");
        assert_eq!(
            error.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );
    }

    #[test]
    fn mcp_add_rejects_malformed_environment_pair() {
        let error = crate::MultitoolCli::try_parse_from([
            "lime", "mcp", "add", "docs", "--env", "BROKEN", "--", "node",
        ])
        .expect_err("invalid env");
        assert_eq!(error.kind(), clap::error::ErrorKind::ValueValidation);
    }
}
