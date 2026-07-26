use app_server::capability_source_from_app_policy_json;
use app_server::init_app_server_otel_from_env;
use app_server::run_transport;
use app_server::AppDataSource;
use app_server::AppServer;
use app_server::AppServerBackendMode;
use app_server::AppServerRuntimeFactory;
use app_server::EventLogWriter;
use app_server::ExternalBackendConfig;
use app_server::FilesystemFileCheckpointSnapshotStore;
use app_server::FilesystemOutputSnapshotStore;
use app_server::LocalAppDataSource;
use app_server::ProjectionStore;
use app_server::RuntimeHostContext;
use app_server::SidecarStore;
use app_server::StorageRoots;
use app_server::TraceEventWriter;
use app_server::DEFAULT_EXTERNAL_BACKEND_TIMEOUT_MS;
use app_server_transport::AppServerTransport;
use app_server_transport::DEFAULT_LISTEN_URL;
use lime_core::app_paths;
use lime_core::database;
use lime_core::database::DbConnection;
use lime_infra::telemetry::TelemetryStore;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const APP_SERVER_DATA_DIR_ENV: &str = "APP_SERVER_DATA_DIR";
const APP_SERVER_APP_DATA_DIR_ENV: &str = "APP_SERVER_APP_DATA_DIR";

struct InitializedDatabase {
    db: DbConnection,
    storage_roots: Option<StorageRoots>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = parse_args()?;
    let transport_base_dir = match config.data_dir.clone() {
        Some(data_dir) => data_dir,
        None => app_paths::preferred_agent_root()
            .map_err(|error| anyhow::anyhow!("failed to resolve AgentRoot: {error}"))?,
    };
    let transport =
        AppServerTransport::from_listen_url_with_base(&config.listen, &transport_base_dir)
            .map_err(|error| anyhow::anyhow!(error))?;

    if config.model_control_source.is_some() && transport == AppServerTransport::Off {
        drop(initialize_database(&config)?);
        return Ok(());
    }

    let _otel_guard = init_app_server_otel_from_env()?;
    run_transport(build_app_server(&config).await?, transport).await?;

    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliConfig {
    listen: String,
    backend_mode: AppServerBackendMode,
    app_policy_path: Option<String>,
    backend_command: Option<String>,
    backend_args: Vec<String>,
    backend_timeout_ms: u64,
    data_dir: Option<PathBuf>,
    app_data_dir: Option<PathBuf>,
    model_control_source: Option<PathBuf>,
}

impl CliConfig {
    fn external_backend_config(&self) -> anyhow::Result<ExternalBackendConfig> {
        let command = self.backend_command.as_ref().ok_or_else(|| {
            anyhow::anyhow!("--backend-command is required when --backend external")
        })?;
        Ok(ExternalBackendConfig::new(command.clone())
            .with_args(self.backend_args.clone())
            .with_timeout_ms(self.backend_timeout_ms))
    }
}

fn parse_args() -> anyhow::Result<CliConfig> {
    parse_args_from_with_env(
        std::env::args().skip(1),
        std::env::var_os(APP_SERVER_DATA_DIR_ENV),
        std::env::var_os(APP_SERVER_APP_DATA_DIR_ENV),
    )
}

async fn build_app_server(config: &CliConfig) -> anyhow::Result<AppServer> {
    let initialized = initialize_database(config)?;
    let db = initialized.db;
    let sidecar_store = initialized
        .storage_roots
        .as_ref()
        .map(|storage_roots| {
            SidecarStore::new(&storage_roots.sidecar_root)
                .map(Arc::new)
                .map_err(|error| anyhow::anyhow!("failed to initialize sidecar store: {error}"))
        })
        .transpose()?;
    let _image_task_worker_scheduler =
        app_server::spawn_image_task_worker_scheduler(db.clone(), sidecar_store.clone());
    let (app_data_root, data_root) = match initialized.storage_roots.as_ref() {
        Some(storage_roots) => (
            storage_roots.app_data_root.clone(),
            storage_roots.data_root.clone(),
        ),
        None => (
            resolve_app_data_root(config)?,
            app_paths::preferred_agent_root()
                .map_err(|error| anyhow::anyhow!("failed to resolve AgentRoot: {error}"))?,
        ),
    };
    // 组合根一次性注入产品资产根，soul style pack 消费链不再解析平台默认目录。
    app_server::install_style_pack_data_root(&app_data_root)
        .map_err(|error| anyhow::anyhow!("failed to install style pack data root: {error}"))?;
    let mut app_data_source =
        LocalAppDataSource::initialize_with_roots(db.clone(), app_data_root, data_root)
            .await
            .map_err(|error| {
                anyhow::anyhow!("failed to initialize local app data source: {error}")
            })?;
    if let Some(sidecar_store) = sidecar_store.clone() {
        app_data_source = app_data_source.with_sidecar_store(sidecar_store);
    }
    let mcp_elicitation_router = app_data_source.mcp_elicitation_router();
    let app_data_source: Arc<dyn AppDataSource> = Arc::new(app_data_source);
    let capability_source = config
        .app_policy_path
        .as_deref()
        .map(load_app_policy_source)
        .transpose()?
        .map(Arc::new);

    let mut runtime = match (config.backend_mode, capability_source) {
        (AppServerBackendMode::External, Some(capability_source)) => {
            AppServerRuntimeFactory::external_runtime_core_with_capability_source(
                config.external_backend_config()?,
                capability_source,
            )
        }
        (AppServerBackendMode::External, None) => {
            AppServerRuntimeFactory::external_runtime_core(config.external_backend_config()?)
        }
        (AppServerBackendMode::Runtime, Some(capability_source)) => {
            AppServerRuntimeFactory::runtime_backend_core_with_db_and_capability_source(
                db,
                capability_source,
            )
        }
        (AppServerBackendMode::Runtime, None) => {
            AppServerRuntimeFactory::runtime_backend_core_with_db(db)
        }
        (AppServerBackendMode::Mock, Some(capability_source)) => {
            AppServerRuntimeFactory::mock_runtime_core_with_capability_source(capability_source)
        }
        (AppServerBackendMode::Mock, None) => AppServerRuntimeFactory::mock_runtime_core(),
        (AppServerBackendMode::Unavailable, Some(capability_source)) => {
            AppServerRuntimeFactory::unavailable_runtime_core_with_capability_source(
                capability_source,
            )
        }
        (AppServerBackendMode::Unavailable, None) => {
            AppServerRuntimeFactory::unavailable_runtime_core()
        }
    }
    .with_app_data_source(app_data_source);
    if let Some(storage_roots) = initialized.storage_roots {
        runtime = runtime
            .with_file_checkpoint_snapshot_store(Arc::new(
                FilesystemFileCheckpointSnapshotStore::with_sidecar_root(
                    &storage_roots.sidecar_root,
                ),
            ))
            .with_output_snapshot_store(Arc::new(
                FilesystemOutputSnapshotStore::with_sidecar_root(&storage_roots.sidecar_root),
            ));
        if let Some(sidecar_store) = sidecar_store.clone() {
            runtime = runtime.with_sidecar_store(sidecar_store);
        }
        let event_log_writer = EventLogWriter::new(&storage_roots.event_log_root)
            .map_err(|error| anyhow::anyhow!("failed to initialize event log writer: {error}"))?;
        let trace_event_writer = TraceEventWriter::new(&storage_roots.trace_log_root)
            .map_err(|error| anyhow::anyhow!("failed to initialize trace event writer: {error}"))?;
        let projection_store = ProjectionStore::initialize_with_storage_paths(
            &storage_roots.projection_db_path,
            &storage_roots.state_db_path,
            &storage_roots.thread_history_db_path,
            &storage_roots.data_root,
        )
        .map_err(|error| anyhow::anyhow!("failed to initialize projection store: {error}"))?;
        let telemetry_store = TelemetryStore::initialize(&storage_roots.telemetry_db_path)
            .map_err(|error| anyhow::anyhow!("failed to initialize telemetry store: {error}"))?;
        runtime = runtime
            .with_event_log_writer(Arc::new(event_log_writer))
            .with_trace_event_writer(Arc::new(trace_event_writer))
            .with_projection_store(Arc::new(projection_store))
            .with_telemetry_store(Arc::new(telemetry_store));
    }

    if let Err(error) = runtime
        .recover_agent_control_spawns(RuntimeHostContext::default(), None)
        .await
    {
        if error.is_provider_selection_required() {
            tracing::warn!(
                error = %error,
                "deferring agent control recovery until provider/model selection is available"
            );
        } else {
            return Err(anyhow::anyhow!(
                "failed to recover agent control spawns: {error}"
            ));
        }
    }

    AppServer::with_runtime(runtime)
        .with_mcp_elicitation_router(mcp_elicitation_router)
        .map_err(|error| anyhow::anyhow!("failed to attach MCP elicitation router: {error}"))
}

/// 组合根：唯一允许解析平台默认 AppDataRoot 的位置。
/// 叶子 writer 一律通过构造注入拿到已解析的根，不得自行解析。
fn resolve_app_data_root(config: &CliConfig) -> anyhow::Result<PathBuf> {
    match config.app_data_dir.as_ref() {
        Some(app_data_dir) => Ok(app_data_dir.clone()),
        None => app_paths::preferred_data_dir()
            .map_err(|error| anyhow::anyhow!("failed to resolve AppDataRoot: {error}")),
    }
}

fn initialize_database(config: &CliConfig) -> anyhow::Result<InitializedDatabase> {
    let app_data_root = resolve_app_data_root(config)?;
    match config.data_dir.as_ref() {
        Some(data_dir) => {
            let storage_roots = StorageRoots::initialize(&app_data_root, data_dir)
                .map_err(|error| anyhow::anyhow!("failed to initialize storage roots: {error}"))?;
            let db = database::init_database_at_path(&storage_roots.product_db_path).map_err(
                |error| {
                    anyhow::anyhow!(
                        "failed to initialize app-server database at {}: {error}",
                        storage_roots.product_db_path.display()
                    )
                },
            )?;
            migrate_model_control_data_if_available(
                data_dir,
                &storage_roots.product_db_path,
                config.model_control_source.as_deref(),
            )?;
            Ok(InitializedDatabase {
                db,
                storage_roots: Some(storage_roots),
            })
        }
        None => {
            let data_root = app_paths::preferred_agent_root()
                .map_err(|error| anyhow::anyhow!("failed to resolve AgentRoot: {error}"))?;
            let storage_roots = StorageRoots::initialize(&app_data_root, &data_root)
                .map_err(|error| anyhow::anyhow!("failed to initialize storage roots: {error}"))?;
            let db = database::init_database_at_path(&storage_roots.product_db_path).map_err(
                |error| {
                    anyhow::anyhow!(
                        "failed to initialize app-server database at {}: {error}",
                        storage_roots.product_db_path.display()
                    )
                },
            )?;
            migrate_model_control_data_if_available(
                &data_root,
                &storage_roots.product_db_path,
                config.model_control_source.as_deref(),
            )?;
            Ok(InitializedDatabase {
                db,
                storage_roots: Some(storage_roots),
            })
        }
    }
}

fn migrate_model_control_data_if_available(
    data_root: &Path,
    target_path: &Path,
    explicit_source: Option<&Path>,
) -> anyhow::Result<()> {
    if let Some(source) = explicit_source {
        validate_explicit_model_control_source(data_root, source)?;
    }
    let candidates = explicit_source
        .map(|source| vec![source.to_path_buf()])
        .unwrap_or_else(|| app_paths::model_control_migration_source_paths(data_root));
    let Some((source_path, signal)) =
        select_model_control_source(data_root, candidates, target_path)?
    else {
        if let Some(source) = explicit_source {
            anyhow::bail!(
                "explicit model control source has no migratable model state: {}",
                source.display()
            );
        }
        return Ok(());
    };
    let report =
        database::model_control_migration::migrate_model_control_data(&source_path, target_path)
            .map_err(|error| {
                anyhow::anyhow!(
                    "failed to migrate model control data from {}: {error}",
                    source_path.display()
                )
            })?;
    if report.changed() {
        tracing::info!(
            source = %source_path.display(),
            source_api_keys = signal.api_keys,
            source_custom_providers = signal.custom_providers,
            source_model_preferences = signal.model_preferences,
            providers = report.providers,
            api_keys = report.api_keys,
            provider_ui_state = report.provider_ui_state,
            model_preferences = report.model_preferences,
            settings = report.settings,
            "migrated model control data without importing Product DB"
        );
    }
    Ok(())
}

fn validate_explicit_model_control_source(data_root: &Path, source: &Path) -> anyhow::Result<()> {
    let allowed_sources = app_paths::model_control_migration_source_paths(data_root);
    let Some(source_root) = allowed_sources
        .iter()
        .find(|candidate| candidate.as_path() == source)
        .and_then(|candidate| candidate.parent())
    else {
        anyhow::bail!(
            "explicit model control source is outside the same-product Lime legacy boundary: {}",
            source.display()
        );
    };

    validate_model_control_source_path(source_root, source)
}

fn validate_model_control_source_path(source_root: &Path, source: &Path) -> anyhow::Result<()> {
    let root_metadata = std::fs::symlink_metadata(source_root).map_err(|error| {
        anyhow::anyhow!(
            "failed to inspect model control source root {}: {error}",
            source_root.display()
        )
    })?;
    if root_metadata.file_type().is_symlink() || !root_metadata.is_dir() {
        anyhow::bail!(
            "model control source root must be a regular non-symlink directory: {}",
            source_root.display()
        );
    }

    let metadata = std::fs::symlink_metadata(source).map_err(|error| {
        anyhow::anyhow!(
            "failed to inspect explicit model control source {}: {error}",
            source.display()
        )
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        anyhow::bail!(
            "explicit model control source must be a regular non-symlink file: {}",
            source.display()
        );
    }

    let canonical_root = std::fs::canonicalize(source_root).map_err(|error| {
        anyhow::anyhow!(
            "failed to canonicalize model control source root {}: {error}",
            source_root.display()
        )
    })?;
    let canonical_source = std::fs::canonicalize(source).map_err(|error| {
        anyhow::anyhow!(
            "failed to canonicalize model control source {}: {error}",
            source.display()
        )
    })?;
    if canonical_source.parent() != Some(canonical_root.as_path()) {
        anyhow::bail!(
            "model control source escapes its same-product Lime legacy root: {}",
            source.display()
        );
    }
    Ok(())
}

fn select_model_control_source(
    data_root: &Path,
    candidates: impl IntoIterator<Item = PathBuf>,
    target_path: &Path,
) -> anyhow::Result<
    Option<(
        PathBuf,
        database::model_control_migration::ModelControlSourceSignal,
    )>,
> {
    let mut selected = None;
    for source_path in candidates {
        if source_path == target_path {
            continue;
        }
        match std::fs::symlink_metadata(&source_path) {
            Ok(_) => validate_explicit_model_control_source(data_root, &source_path)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(anyhow::anyhow!(
                    "failed to inspect model control data at {}: {error}",
                    source_path.display()
                ));
            }
        }
        let signal = database::model_control_migration::inspect_model_control_source(&source_path)
            .map_err(|error| {
                anyhow::anyhow!(
                    "failed to inspect model control data at {}: {error}",
                    source_path.display()
                )
            })?;
        let Some(signal) = signal else {
            continue;
        };
        let should_replace = selected.as_ref().is_none_or(
            |(_, current): &(
                PathBuf,
                database::model_control_migration::ModelControlSourceSignal,
            )| { signal.priority() > current.priority() },
        );
        if should_replace {
            selected = Some((source_path, signal));
        }
    }
    Ok(selected)
}

#[cfg(test)]
fn parse_args_from(args: impl IntoIterator<Item = String>) -> anyhow::Result<CliConfig> {
    parse_args_from_with_env(args, None, None)
}

fn parse_args_from_with_env(
    args: impl IntoIterator<Item = String>,
    data_dir_env: Option<OsString>,
    app_data_dir_env: Option<OsString>,
) -> anyhow::Result<CliConfig> {
    let mut args = args.into_iter();
    let mut listen = DEFAULT_LISTEN_URL.to_string();
    let mut backend_mode = AppServerBackendMode::Unavailable;
    let mut app_policy_path = None;
    let mut backend_command = None;
    let mut backend_args = Vec::new();
    let mut backend_timeout_ms = DEFAULT_EXTERNAL_BACKEND_TIMEOUT_MS;
    let mut data_dir_from_cli = false;
    let mut data_dir = data_dir_env
        .filter(|value| !value.is_empty())
        .map(PathBuf::from);
    let mut app_data_dir = app_data_dir_env
        .filter(|value| !value.is_empty())
        .map(PathBuf::from);
    let mut model_control_source = None;

    while let Some(arg) = args.next() {
        if let Some(value) = arg.strip_prefix("--data-dir=") {
            if value.is_empty() {
                anyhow::bail!("--data-dir requires a path");
            }
            data_dir = Some(PathBuf::from(value));
            data_dir_from_cli = true;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--app-data-dir=") {
            if value.is_empty() {
                anyhow::bail!("--app-data-dir requires a path");
            }
            app_data_dir = Some(PathBuf::from(value));
            continue;
        }
        if let Some(value) = arg.strip_prefix("--model-control-source=") {
            if value.is_empty() {
                anyhow::bail!("--model-control-source requires a path");
            }
            model_control_source = Some(PathBuf::from(value));
            continue;
        }
        match arg.as_str() {
            "--stdio" => listen = DEFAULT_LISTEN_URL.to_string(),
            "--listen" => {
                listen = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--listen requires a URL"))?;
            }
            "--backend" => {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--backend requires a value"))?;
                backend_mode = AppServerBackendMode::parse(&value)?;
            }
            "--backend-command" => {
                backend_command = Some(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("--backend-command requires a path"))?,
                );
            }
            "--backend-arg" => {
                backend_args.push(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("--backend-arg requires a value"))?,
                );
            }
            "--backend-timeout-ms" => {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--backend-timeout-ms requires a value"))?;
                backend_timeout_ms = value
                    .parse::<u64>()
                    .map_err(|error| anyhow::anyhow!("invalid --backend-timeout-ms: {error}"))?;
            }
            "--app-policy" => {
                app_policy_path = Some(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("--app-policy requires a path"))?,
                );
            }
            "--data-dir" => {
                data_dir =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        anyhow::anyhow!("--data-dir requires a path")
                    })?));
                data_dir_from_cli = true;
            }
            "--app-data-dir" => {
                app_data_dir =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        anyhow::anyhow!("--app-data-dir requires a path")
                    })?));
            }
            "--model-control-source" => {
                model_control_source =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        anyhow::anyhow!("--model-control-source requires a path")
                    })?));
            }
            "--help" | "-h" => {
                println!(
                    "Usage: app-server [--stdio] [--listen stdio://|unix://PATH|ws://127.0.0.1:PORT|off] [--backend external|runtime|mock|unavailable] [--backend-command path] [--backend-arg value] [--backend-timeout-ms ms] [--app-policy path] [--data-dir path] [--app-data-dir path] [--model-control-source path]"
                );
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }

    if model_control_source.is_some() && !data_dir_from_cli {
        anyhow::bail!("--model-control-source requires an explicit --data-dir");
    }

    Ok(CliConfig {
        listen,
        backend_mode,
        app_policy_path,
        backend_command,
        backend_args,
        backend_timeout_ms,
        data_dir,
        app_data_dir,
        model_control_source,
    })
}

fn load_app_policy_source(path: &str) -> anyhow::Result<app_server::CapabilityInventorySource> {
    let json = std::fs::read_to_string(path)
        .map_err(|error| anyhow::anyhow!("failed to read app policy {path}: {error}"))?;
    capability_source_from_app_policy_json(&json)
        .map_err(|error| anyhow::anyhow!("failed to load app policy {path}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_args_defaults_to_stdio_unavailable_backend() {
        let config = parse_args_from(Vec::new()).expect("config");

        assert_eq!(config.listen, DEFAULT_LISTEN_URL);
        assert_eq!(config.backend_mode, AppServerBackendMode::Unavailable);
        assert_eq!(config.app_policy_path, None);
        assert_eq!(config.backend_command, None);
        assert!(config.backend_args.is_empty());
        assert_eq!(
            config.backend_timeout_ms,
            DEFAULT_EXTERNAL_BACKEND_TIMEOUT_MS
        );
        assert_eq!(config.data_dir, None);
        assert_eq!(config.model_control_source, None);
    }

    #[test]
    fn parse_args_accepts_explicit_mock_backend() {
        let config =
            parse_args_from(["--stdio", "--backend", "mock"].map(str::to_string)).expect("config");

        assert_eq!(config.listen, DEFAULT_LISTEN_URL);
        assert_eq!(config.backend_mode, AppServerBackendMode::Mock);
        assert_eq!(config.app_policy_path, None);
        assert_eq!(config.backend_command, None);
        assert!(config.backend_args.is_empty());
    }

    #[test]
    fn parse_args_accepts_explicit_unavailable_backend() {
        let config = parse_args_from(["--stdio", "--backend", "unavailable"].map(str::to_string))
            .expect("config");

        assert_eq!(config.listen, DEFAULT_LISTEN_URL);
        assert_eq!(config.backend_mode, AppServerBackendMode::Unavailable);
        assert_eq!(config.app_policy_path, None);
        assert_eq!(config.backend_command, None);
        assert!(config.backend_args.is_empty());
    }

    #[test]
    fn parse_args_accepts_explicit_runtime_backend() {
        let config = parse_args_from(["--stdio", "--backend", "runtime"].map(str::to_string))
            .expect("config");

        assert_eq!(config.listen, DEFAULT_LISTEN_URL);
        assert_eq!(config.backend_mode, AppServerBackendMode::Runtime);
        assert_eq!(config.app_policy_path, None);
        assert_eq!(config.backend_command, None);
        assert!(config.backend_args.is_empty());
    }

    #[test]
    fn parse_args_accepts_data_dir() {
        let config = parse_args_from(
            ["--stdio", "--data-dir", "/tmp/platform-app-server"].map(str::to_string),
        )
        .expect("config");

        assert_eq!(
            config.data_dir,
            Some(PathBuf::from("/tmp/platform-app-server"))
        );
    }

    #[test]
    fn parse_args_accepts_data_dir_equals_form() {
        let config = parse_args_from(["--data-dir=/tmp/platform-app-server"].map(str::to_string))
            .expect("config");

        assert_eq!(
            config.data_dir,
            Some(PathBuf::from("/tmp/platform-app-server"))
        );
    }

    #[test]
    fn parse_args_accepts_app_data_dir_both_forms() {
        let spaced =
            parse_args_from(["--app-data-dir", "/tmp/platform-app-data"].map(str::to_string))
                .expect("config");
        let equals = parse_args_from(["--app-data-dir=/tmp/platform-app-data"].map(str::to_string))
            .expect("config");

        let expected = Some(PathBuf::from("/tmp/platform-app-data"));
        assert_eq!(spaced.app_data_dir, expected);
        assert_eq!(equals.app_data_dir, expected);
    }

    #[test]
    fn parse_args_app_data_dir_cli_wins_over_env() {
        let config = parse_args_from_with_env(
            ["--app-data-dir", "/tmp/cli-app-data"].map(str::to_string),
            None,
            Some(OsString::from("/tmp/env-app-data")),
        )
        .expect("config");

        assert_eq!(
            config.app_data_dir,
            Some(PathBuf::from("/tmp/cli-app-data"))
        );
    }

    #[test]
    fn parse_args_accepts_app_data_dir_env_fallback() {
        let config = parse_args_from_with_env(
            ["--stdio"].map(str::to_string),
            None,
            Some(OsString::from("/tmp/env-app-data")),
        )
        .expect("config");

        assert_eq!(
            config.app_data_dir,
            Some(PathBuf::from("/tmp/env-app-data"))
        );
    }

    #[test]
    fn parse_args_rejects_empty_app_data_dir() {
        let error = parse_args_from(["--app-data-dir="].map(str::to_string))
            .expect_err("empty --app-data-dir must fail closed");

        assert!(error.to_string().contains("--app-data-dir requires a path"));
    }

    #[test]
    fn parse_args_accepts_explicit_model_control_source_with_data_dir() {
        let config = parse_args_from(
            [
                "--data-dir=/tmp/platform-app-server",
                "--model-control-source=/tmp/lime-legacy/app.db",
            ]
            .map(str::to_string),
        )
        .expect("config");

        assert_eq!(
            config.model_control_source,
            Some(PathBuf::from("/tmp/lime-legacy/app.db"))
        );
    }

    #[test]
    fn parse_args_rejects_model_control_source_without_explicit_data_dir() {
        let error = parse_args_from_with_env(
            ["--model-control-source", "/tmp/lime-legacy/app.db"].map(str::to_string),
            Some(OsString::from("/tmp/env-app-server")),
            None,
        )
        .expect_err("model source must not target an ambient data dir");

        assert!(error
            .to_string()
            .contains("requires an explicit --data-dir"));
    }

    #[test]
    fn parse_args_data_dir_cli_wins_over_env() {
        let config = parse_args_from_with_env(
            ["--data-dir", "/tmp/cli-app-server"].map(str::to_string),
            Some(OsString::from("/tmp/env-app-server")),
            None,
        )
        .expect("config");

        assert_eq!(config.data_dir, Some(PathBuf::from("/tmp/cli-app-server")));
    }

    #[test]
    fn parse_args_accepts_data_dir_env_fallback() {
        let config = parse_args_from_with_env(
            Vec::new(),
            Some(OsString::from("/tmp/env-app-server")),
            None,
        )
        .expect("config");

        assert_eq!(config.data_dir, Some(PathBuf::from("/tmp/env-app-server")));
    }

    #[test]
    fn parse_args_rejects_retired_product_db_migration_cleanup_policy() {
        let error =
            parse_args_from(["--product-db-migration-cleanup", "delete-file"].map(str::to_string))
                .expect_err("retired startup cleanup option must be rejected");

        assert!(error.to_string().contains("unknown argument"));

        let error = parse_args_from(["--product-db-migration-cleanup=retain"].map(str::to_string))
            .expect_err("retired startup cleanup option must be rejected");

        assert!(error.to_string().contains("unknown argument"));
    }

    #[test]
    fn initialize_database_uses_configured_data_dir() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let app_data_dir = temp_dir.path().join("desktop-platform");
        let data_dir = app_data_dir.join("app-server");
        let config = CliConfig {
            listen: DEFAULT_LISTEN_URL.to_string(),
            backend_mode: AppServerBackendMode::Unavailable,
            app_policy_path: None,
            backend_command: None,
            backend_args: Vec::new(),
            backend_timeout_ms: DEFAULT_EXTERNAL_BACKEND_TIMEOUT_MS,
            data_dir: Some(data_dir.clone()),
            app_data_dir: Some(app_data_dir.clone()),
            model_control_source: None,
        };

        let db = initialize_database(&config).expect("initialize database");
        drop(db);

        assert!(data_dir.join("lime.db").is_file());
        let storage_roots =
            StorageRoots::initialize(&app_data_dir, &data_dir).expect("storage roots");
        assert_eq!(storage_roots.app_data_root, app_data_dir);
        assert_eq!(
            storage_roots.projection_db_path,
            data_dir.join("runtime").join("projection_1.sqlite")
        );
        assert_eq!(
            storage_roots.state_db_path,
            data_dir.join("sqlite").join("state.sqlite")
        );
        assert_eq!(
            storage_roots.thread_history_db_path,
            data_dir.join("sqlite").join("thread_history.sqlite")
        );
        assert_eq!(
            storage_roots.event_log_root,
            data_dir.join("runtime").join("events")
        );
        assert_eq!(
            storage_roots.trace_log_root,
            data_dir.join("runtime").join("traces")
        );
    }

    #[test]
    fn initialize_database_migrates_only_model_control_data_from_same_product_source() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let user_data_root = temp_dir.path().join("desktop-platform");
        let app_server_data_dir = user_data_root.join("app-server");
        let previous_db = user_data_root.join("app.db");
        let previous_db_conn =
            database::init_database_at_path(&previous_db).expect("previous product database");
        {
            let conn = previous_db_conn.lock().expect("previous db lock");
            conn.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?1, ?2)",
                ("providers.active_tab", "service-providers"),
            )
            .expect("insert previous setting");
            conn.execute(
                "INSERT INTO api_key_providers
                 (id, name, type, api_host, is_system, group_name, enabled, sort_order, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, 0, ?5, 1, 1, ?6, ?6)",
                (
                    "custom-openai-compatible",
                    "Custom OpenAI Compatible",
                    "openai",
                    "https://models.example.invalid/v1",
                    "custom",
                    "2026-06-15T00:00:00Z",
                ),
            )
            .expect("insert previous provider");
            conn.execute(
                "INSERT INTO api_keys (id, provider_id, api_key_encrypted, created_at)
                 VALUES ('provider-key', 'custom-openai-compatible', 'ciphertext', '2026-06-15T00:00:00Z')",
                [],
            )
            .expect("insert previous API key");
            conn.execute(
                "INSERT INTO settings (key, value) VALUES ('request_logs.retention', '7d')",
                [],
            )
            .expect("insert non-model setting");
        }
        drop(previous_db_conn);

        let config = CliConfig {
            listen: DEFAULT_LISTEN_URL.to_string(),
            backend_mode: AppServerBackendMode::Unavailable,
            app_policy_path: None,
            backend_command: None,
            backend_args: Vec::new(),
            backend_timeout_ms: DEFAULT_EXTERNAL_BACKEND_TIMEOUT_MS,
            data_dir: Some(app_server_data_dir.clone()),
            app_data_dir: Some(user_data_root.clone()),
            model_control_source: Some(previous_db.clone()),
        };

        let db = initialize_database(&config).expect("initialize app-server database");
        drop(db);

        let current = rusqlite::Connection::open(app_server_data_dir.join("lime.db"))
            .expect("current app-server db");
        let setting_value: String = current
            .query_row(
                "SELECT value FROM settings WHERE key = ?1",
                ["providers.active_tab"],
                |row| row.get(0),
            )
            .expect("migrated provider setting");
        let provider_count: i64 = current
            .query_row(
                "SELECT COUNT(*) FROM api_key_providers WHERE id = ?1",
                ["custom-openai-compatible"],
                |row| row.get(0),
            )
            .expect("migrated custom provider count");
        let api_key_count: i64 = current
            .query_row(
                "SELECT COUNT(*) FROM api_keys WHERE id = 'provider-key'",
                [],
                |row| row.get(0),
            )
            .expect("migrated API key count");
        let non_model_setting_count: i64 = current
            .query_row(
                "SELECT COUNT(*) FROM settings WHERE key = 'request_logs.retention'",
                [],
                |row| row.get(0),
            )
            .expect("non-model setting count");

        assert_eq!(setting_value, "service-providers");
        assert_eq!(provider_count, 1);
        assert_eq!(api_key_count, 1);
        assert_eq!(non_model_setting_count, 0);

        let previous = rusqlite::Connection::open(&previous_db).expect("retained product db");
        let old_user_table_count: i64 = previous
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'",
                [],
                |row| row.get(0),
            )
            .expect("retained product db schema");
        assert!(old_user_table_count > 0);
        let retained_setting_value: String = previous
            .query_row(
                "SELECT value FROM settings WHERE key = ?1",
                ["providers.active_tab"],
                |row| row.get(0),
            )
            .expect("retained provider setting");
        assert_eq!(retained_setting_value, "service-providers");
        let retained_provider_count: i64 = previous
            .query_row(
                "SELECT COUNT(*) FROM api_key_providers WHERE id = ?1",
                ["custom-openai-compatible"],
                |row| row.get(0),
            )
            .expect("retained custom provider");
        assert_eq!(retained_provider_count, 1);
    }

    #[test]
    fn explicit_model_control_source_rejects_sibling_product_path() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let data_root = temp_dir.path().join("lime").join("app-server");
        let sibling_source = temp_dir
            .path()
            .join("content-studio")
            .join("app-server")
            .join("lime.db");
        std::fs::create_dir_all(sibling_source.parent().expect("source parent"))
            .expect("create source parent");
        std::fs::write(&sibling_source, []).expect("create source file");

        let error = validate_explicit_model_control_source(&data_root, &sibling_source)
            .expect_err("sibling product source must be rejected");

        assert!(error
            .to_string()
            .contains("outside the same-product Lime legacy boundary"));
    }

    #[cfg(unix)]
    #[test]
    fn explicit_model_control_source_rejects_symlink_file() {
        use std::os::unix::fs::symlink;

        let temp_dir = tempfile::tempdir().expect("temp dir");
        let product_root = temp_dir.path().join("lime");
        let data_root = product_root.join("app-server");
        let sibling_source = temp_dir.path().join("content-studio").join("app.db");
        let linked_source = product_root.join("app.db");
        std::fs::create_dir_all(&product_root).expect("create product root");
        std::fs::create_dir_all(sibling_source.parent().expect("sibling parent"))
            .expect("create sibling root");
        std::fs::write(&sibling_source, []).expect("create sibling source");
        symlink(&sibling_source, &linked_source).expect("link model source");

        let error = validate_explicit_model_control_source(&data_root, &linked_source)
            .expect_err("symlink source must be rejected");

        assert!(error.to_string().contains("regular non-symlink file"));
    }

    #[cfg(unix)]
    #[test]
    fn explicit_model_control_source_rejects_symlink_root() {
        use std::os::unix::fs::symlink;

        let temp_dir = tempfile::tempdir().expect("temp dir");
        let sibling_root = temp_dir.path().join("content-studio");
        let product_root = temp_dir.path().join("lime");
        let data_root = product_root.join("app-server");
        let source = product_root.join("app.db");
        std::fs::create_dir_all(&sibling_root).expect("create sibling root");
        std::fs::write(sibling_root.join("app.db"), []).expect("create sibling source");
        symlink(&sibling_root, &product_root).expect("link product root");

        let error = validate_explicit_model_control_source(&data_root, &source)
            .expect_err("symlink source root must be rejected");

        assert!(error.to_string().contains("regular non-symlink directory"));
    }

    #[test]
    fn model_control_source_selection_prefers_credentials_over_ui_state() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let product_root = temp_dir.path().join("lime");
        let data_root = product_root.join("app-server");
        let ui_source = data_root.join("app.db");
        let credential_source = product_root.join("proxycast.db");
        let target = data_root.join("lime.db");
        let ui_db = database::init_database_at_path(&ui_source).expect("ui source");
        {
            let conn = ui_db.lock().expect("ui source lock");
            conn.execute(
                "INSERT INTO provider_ui_state (key, value) VALUES ('selected_provider', 'openai')",
                [],
            )
            .expect("ui state");
        }
        let credential_db =
            database::init_database_at_path(&credential_source).expect("credential source");
        {
            let conn = credential_db.lock().expect("credential source lock");
            conn.execute(
                "INSERT INTO api_key_providers
                 (id, name, type, api_host, is_system, group_name, enabled, sort_order, created_at, updated_at)
                 VALUES ('custom', 'Custom', 'openai', 'https://example.invalid/v1', 0, 'custom', 1, 1, '2026-07-19', '2026-07-19')",
                [],
            )
            .expect("provider");
            conn.execute(
                "INSERT INTO api_keys (id, provider_id, api_key_encrypted, created_at)
                 VALUES ('key', 'custom', 'ciphertext', '2026-07-19')",
                [],
            )
            .expect("api key");
        }

        let (selected, signal) = select_model_control_source(
            &data_root,
            [ui_source.clone(), credential_source.clone()],
            &target,
        )
        .expect("select source")
        .expect("selected source");

        assert_eq!(selected, credential_source);
        assert_eq!(signal.api_keys, 1);
        assert_eq!(signal.custom_providers, 1);
    }

    #[test]
    fn parse_args_accepts_external_backend_command() {
        let config = parse_args_from(
            [
                "--stdio",
                "--backend",
                "external",
                "--backend-command",
                "/tmp/agent-backend",
                "--backend-arg",
                "--profile",
                "--backend-arg",
                "content",
                "--backend-timeout-ms",
                "1234",
            ]
            .map(str::to_string),
        )
        .expect("config");

        assert_eq!(config.listen, DEFAULT_LISTEN_URL);
        assert_eq!(config.backend_mode, AppServerBackendMode::External);
        assert_eq!(
            config.backend_command.as_deref(),
            Some("/tmp/agent-backend")
        );
        assert_eq!(
            config.backend_args,
            vec!["--profile".to_string(), "content".to_string()]
        );
        assert_eq!(config.backend_timeout_ms, 1234);

        let backend_config = config.external_backend_config().expect("backend config");
        assert_eq!(backend_config.command, "/tmp/agent-backend");
        assert_eq!(
            backend_config.args,
            vec!["--profile".to_string(), "content".to_string()]
        );
        assert_eq!(backend_config.timeout_ms, 1234);
    }

    #[test]
    fn external_backend_requires_command() {
        let config =
            parse_args_from(["--backend", "external"].map(str::to_string)).expect("config");
        let error = config.external_backend_config().expect_err("error");

        assert!(error
            .to_string()
            .contains("--backend-command is required when --backend external"));
    }

    #[test]
    fn parse_args_accepts_app_policy_path() {
        let config = parse_args_from(
            ["--stdio", "--app-policy", "/tmp/content-studio.policy.json"].map(str::to_string),
        )
        .expect("config");

        assert_eq!(config.listen, DEFAULT_LISTEN_URL);
        assert_eq!(config.backend_mode, AppServerBackendMode::Unavailable);
        assert_eq!(
            config.app_policy_path.as_deref(),
            Some("/tmp/content-studio.policy.json")
        );
    }

    #[test]
    fn load_app_policy_source_reads_scoped_capabilities() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let policy_path = temp_dir.path().join("app-policy.json");
        std::fs::write(
            &policy_path,
            r#"{
              "capabilities": [
                {
                  "id": "content.draft.generate",
                  "title": "Generate Draft",
                  "methods": ["turn/start"],
                  "appIds": ["content-studio"]
                }
              ]
            }"#,
        )
        .expect("write policy");

        let source = load_app_policy_source(policy_path.to_str().expect("policy path"))
            .expect("policy source");
        let capabilities = app_server::CapabilitySource::list_capabilities(
            &source,
            &app_server::CapabilityListContext {
                app_id: Some("content-studio".to_string()),
                workspace_id: None,
                session_id: None,
            },
        );

        assert_eq!(capabilities.len(), 1);
        assert_eq!(capabilities[0].id, "content.draft.generate");
    }

    #[test]
    fn parse_args_rejects_runtime_backend_for_standalone_binary() {
        let error = parse_args_from(["--backend", "agent"].map(str::to_string)).expect_err("error");

        assert!(error
            .to_string()
            .contains("unsupported app-server backend mode: agent"));
    }
}
