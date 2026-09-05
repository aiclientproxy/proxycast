use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use app_server_protocol::protocol::v2::{
    PluginCatalogEnabledSetParams, PluginCatalogEnabledSetResponse, PluginCatalogInstallParams,
    PluginCatalogInstallResponse, PluginCatalogListParams, PluginCatalogListResponse,
    PluginCatalogReadParams, PluginCatalogReadResponse, PluginCatalogUninstallParams,
    PluginCatalogUninstallResponse, PluginSearchParams, PluginSearchResponse, PluginSearchScope,
    METHOD_PLUGIN_ENABLED_SET, METHOD_PLUGIN_INSTALL, METHOD_PLUGIN_LIST, METHOD_PLUGIN_READ,
    METHOD_PLUGIN_SEARCH, METHOD_PLUGIN_UNINSTALL,
};
use clap::{Args, ValueEnum};

use crate::ConnectionArgs;

#[derive(Debug, Args)]
pub(crate) struct PluginCli {
    #[command(subcommand)]
    subcommand: PluginSubcommand,
}

#[derive(Debug, clap::Subcommand)]
enum PluginSubcommand {
    /// Install a plugin package from a local source path.
    Add(AddPluginArgs),
    /// List plugins visible to the current App Server.
    List(ListPluginsArgs),
    /// Read one plugin and all of its declared capabilities.
    Read(PluginIdArgs),
    /// Search the current plugin catalog.
    Search(PluginSearchArgs),
    /// Uninstall a plugin.
    Remove(RemovePluginArgs),
    /// Enable an installed plugin.
    Enable(PluginIdArgs),
    /// Disable an installed plugin.
    Disable(PluginIdArgs),
}

#[derive(Debug, Args)]
struct AddPluginArgs {
    #[arg(value_name = "SOURCE_PATH")]
    source_path: PathBuf,
    #[arg(long)]
    marketplace: Option<String>,
    #[arg(long)]
    source: Option<String>,
    #[arg(long)]
    expected_digest: Option<String>,
    #[arg(long)]
    json: bool,
    #[command(flatten)]
    connection: ConnectionArgs,
}

#[derive(Debug, Args)]
struct ListPluginsArgs {
    #[arg(long)]
    query: Option<String>,
    #[arg(long)]
    source: Option<String>,
    #[arg(long = "marketplace-path", value_name = "PATH")]
    marketplace_paths: Vec<String>,
    #[arg(long)]
    json: bool,
    #[command(flatten)]
    connection: ConnectionArgs,
}

#[derive(Debug, Args)]
struct PluginIdArgs {
    #[arg(value_name = "PLUGIN")]
    plugin_id: String,
    #[arg(long)]
    json: bool,
    #[command(flatten)]
    connection: ConnectionArgs,
}

#[derive(Debug, Args)]
struct RemovePluginArgs {
    #[arg(value_name = "PLUGIN")]
    plugin_id: String,
    #[arg(long)]
    json: bool,
    #[command(flatten)]
    connection: ConnectionArgs,
}

#[derive(Debug, Args)]
struct PluginSearchArgs {
    #[arg(value_name = "QUERY")]
    query: String,
    #[arg(long, value_enum)]
    scope: Option<SearchScope>,
    #[arg(long = "plugin-cwd", value_name = "DIR")]
    cwds: Vec<String>,
    #[arg(long)]
    cursor: Option<String>,
    #[arg(long)]
    limit: Option<u32>,
    #[command(flatten)]
    connection: ConnectionArgs,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SearchScope {
    Global,
    Workspace,
    Personal,
}

impl From<SearchScope> for PluginSearchScope {
    fn from(value: SearchScope) -> Self {
        match value {
            SearchScope::Global => Self::Global,
            SearchScope::Workspace => Self::Workspace,
            SearchScope::Personal => Self::Personal,
        }
    }
}

impl PluginCli {
    pub(crate) async fn run(self) -> ExitCode {
        match run_inner(self).await {
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
}

async fn run_inner(command: PluginCli) -> Result<String> {
    match command.subcommand {
        PluginSubcommand::Add(args) => run_plugin_add(args).await,
        PluginSubcommand::List(args) => run_plugin_list(args).await,
        PluginSubcommand::Read(args) => run_plugin_read(args).await,
        PluginSubcommand::Search(args) => run_plugin_search(args).await,
        PluginSubcommand::Remove(args) => run_plugin_remove(args).await,
        PluginSubcommand::Enable(args) => set_enabled(args, true).await,
        PluginSubcommand::Disable(args) => set_enabled(args, false).await,
    }
}

async fn run_plugin_add(args: AddPluginArgs) -> Result<String> {
    let response = request(
        &args.connection,
        METHOD_PLUGIN_INSTALL,
        PluginCatalogInstallParams {
            source_path: args.source_path.to_string_lossy().into_owned(),
            marketplace_id: args.marketplace,
            source: args.source,
            expected_digest: args.expected_digest,
        },
    )
    .await?;
    let response: PluginCatalogInstallResponse = serde_json::from_value(response)?;
    if args.json {
        return Ok(serde_json::to_string_pretty(&response)?);
    }
    Ok(format!(
        "Added plugin `{}` from marketplace `{}`.\nInstalled plugin source: {}",
        response.plugin.name, response.plugin.marketplace_id, response.plugin.source_uri
    ))
}

async fn run_plugin_list(args: ListPluginsArgs) -> Result<String> {
    let response = request(
        &args.connection,
        METHOD_PLUGIN_LIST,
        PluginCatalogListParams {
            query: args.query,
            source: args.source,
            marketplace_paths: args.marketplace_paths,
        },
    )
    .await?;
    let response: PluginCatalogListResponse = serde_json::from_value(response)?;
    if args.json {
        return Ok(serde_json::to_string_pretty(&response)?);
    }
    Ok(render_plugin_table(response))
}

async fn run_plugin_read(args: PluginIdArgs) -> Result<String> {
    let response = request(
        &args.connection,
        METHOD_PLUGIN_READ,
        PluginCatalogReadParams {
            plugin_id: args.plugin_id,
        },
    )
    .await?;
    let response: PluginCatalogReadResponse = serde_json::from_value(response)?;
    Ok(serde_json::to_string_pretty(&response)?)
}

async fn run_plugin_search(args: PluginSearchArgs) -> Result<String> {
    let response = request(
        &args.connection,
        METHOD_PLUGIN_SEARCH,
        PluginSearchParams {
            search_term: args.query,
            scope: args.scope.map(Into::into),
            cwds: (!args.cwds.is_empty()).then_some(args.cwds),
            cursor: args.cursor,
            limit: args.limit,
        },
    )
    .await?;
    let response: PluginSearchResponse = serde_json::from_value(response)?;
    Ok(serde_json::to_string_pretty(&response)?)
}

async fn run_plugin_remove(args: RemovePluginArgs) -> Result<String> {
    let response = request(
        &args.connection,
        METHOD_PLUGIN_UNINSTALL,
        PluginCatalogUninstallParams {
            plugin_id: args.plugin_id,
        },
    )
    .await?;
    let response: PluginCatalogUninstallResponse = serde_json::from_value(response)?;
    if args.json {
        return Ok(serde_json::to_string_pretty(&response)?);
    }
    Ok(if response.uninstalled {
        format!("Removed plugin `{}`.", response.plugin_id)
    } else {
        format!("No installed plugin named `{}` found.", response.plugin_id)
    })
}

async fn set_enabled(args: PluginIdArgs, enabled: bool) -> Result<String> {
    let response = request(
        &args.connection,
        METHOD_PLUGIN_ENABLED_SET,
        PluginCatalogEnabledSetParams {
            plugin_id: args.plugin_id,
            enabled,
        },
    )
    .await?;
    let response: PluginCatalogEnabledSetResponse = serde_json::from_value(response)?;
    if args.json {
        return Ok(serde_json::to_string_pretty(&response)?);
    }
    Ok(format!(
        "{} plugin `{}`.",
        if enabled { "Enabled" } else { "Disabled" },
        response.plugin.id
    ))
}

async fn request(
    connection: &ConnectionArgs,
    method: &str,
    params: impl serde::Serialize,
) -> Result<serde_json::Value> {
    crate::request_value(connection, method, serde_json::to_value(params)?).await
}

fn render_plugin_table(response: PluginCatalogListResponse) -> String {
    if response.plugins.is_empty() {
        return "No plugins found.".to_string();
    }
    let name_width = response
        .plugins
        .iter()
        .map(|plugin| plugin.name.len())
        .chain(["PLUGIN".len()])
        .max()
        .unwrap_or_default();
    let status_width = "installed, disabled".len();
    let mut lines = vec![format!(
        "{:<name_width$}  {:<status_width$}  VERSION  SOURCE",
        "PLUGIN", "STATUS"
    )];
    lines.extend(response.plugins.into_iter().map(|plugin| {
        let status = if plugin.installed && plugin.enabled {
            "installed, enabled"
        } else if plugin.installed {
            "installed, disabled"
        } else {
            "not installed"
        };
        format!(
            "{:<name_width$}  {:<status_width$}  {:<7}  {}",
            plugin.name, status, plugin.version, plugin.source_uri
        )
    }));
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    #[test]
    fn plugin_add_list_and_remove_parse_under_plugin() {
        for args in [
            vec!["lime", "plugin", "add", "/tmp/plugin"],
            vec!["lime", "plugin", "list", "--json"],
            vec!["lime", "plugin", "remove", "sample", "--json"],
        ] {
            let cli = crate::MultitoolCli::try_parse_from(args).expect("parse plugin command");
            assert!(matches!(cli.subcommand, Some(crate::Subcommand::Plugin(_))));
        }
    }

    #[test]
    fn plugin_list_available_is_not_an_unbacked_flag() {
        let error = crate::MultitoolCli::try_parse_from(["lime", "plugin", "list", "--available"])
            .expect_err("available needs a distinct App Server contract");
        assert_eq!(error.kind(), clap::error::ErrorKind::UnknownArgument);
    }

    #[test]
    fn plugin_search_uses_a_distinct_marketplace_cwd_flag() {
        let cli = crate::MultitoolCli::try_parse_from([
            "lime",
            "plugin",
            "search",
            "surface",
            "--plugin-cwd",
            "/tmp/workspace",
        ])
        .expect("parse plugin search");
        assert!(matches!(cli.subcommand, Some(crate::Subcommand::Plugin(_))));
    }
}
