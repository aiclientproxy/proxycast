use app_server_protocol::protocol::v2::{
    AppInfo, AppsInstalledParams, AppsInstalledResponse, AppsListParams, AppsListResponse,
    AppsReadParams, AppsReadResponse, ConnectorMetadata, InstalledApp, PluginAuthPolicy,
    PluginAvailability, PluginCatalogCapability, PluginCatalogDetail, PluginCatalogHook,
    PluginCatalogInstallParams, PluginCatalogInstallResponse, PluginCatalogInstalledParams,
    PluginCatalogListParams, PluginCatalogListResponse, PluginCatalogReadParams,
    PluginCatalogReadResponse, PluginCatalogSummary, PluginCatalogUiResource,
    PluginCatalogUninstallParams, PluginCatalogUninstallResponse, PluginInstallPolicy,
    PluginInterface, PluginSearchParams, PluginSearchResponse, PluginSearchResult,
    PluginSearchScope, PluginSource, PluginSummary,
};
use chrono::Utc;
use lime_mcp::agent_plugin_config::parse_agent_plugin_mcp_config;
use lime_mcp::{McpRuntimeServerSpec, McpServerConfig};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

const STORE_DIR: &str = "v3";
const PACKAGES_DIR: &str = "packages";
const INSTALLED_DIR: &str = "installed";
const STAGING_DIR: &str = "staging";
const STANDARD_MCP_CONFIG_PATH: &str = "mcp.json";
const STANDARD_MANIFEST_SCHEMA: &str = "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json";
const CODEX_EXTENSION_NAMESPACE: &str = "com.openai";
const CODEX_EXTENSION_MANIFEST_PATH: &str = ".codex-plugin/plugin.json";
const STANDARD_MANIFEST_FIELDS: &[&str] = &[
    "$schema",
    "name",
    "version",
    "description",
    "author",
    "homepage",
    "repository",
    "license",
    "keywords",
    "extensions",
];
const INSTALLED_SCHEMA_VERSION: u64 = 1;
const MAX_MANIFEST_BYTES: u64 = 2 * 1024 * 1024;
const MAX_PACKAGE_FILES: usize = 10_000;
const MAX_PACKAGE_BYTES: u64 = 256 * 1024 * 1024;

pub(crate) fn list(
    plugin_data_root: &Path,
    params: PluginCatalogListParams,
) -> Result<PluginCatalogListResponse, String> {
    let installed = read_installed_records(plugin_data_root)?;
    let mut summaries = Vec::new();
    let mut source_paths = params
        .marketplace_paths
        .into_iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    let explicit_source_paths = !source_paths.is_empty();
    if source_paths.is_empty() {
        source_paths = implicit_marketplace_paths(plugin_data_root);
    }

    for source_path in &source_paths {
        for package_root in discover_package_roots(source_path)? {
            let source_uri = package_root.to_string_lossy().into_owned();
            let installed_record = read_manifest(&package_root)
                .ok()
                .and_then(|manifest| manifest_name(&manifest).ok())
                .and_then(|plugin_id| installed.get(&plugin_id).cloned());
            let summary = build_summary(
                &package_root,
                source_kind_for_marketplace_path(source_path),
                &source_uri,
                installed_record.as_ref(),
            )?;
            let mut summary = summary;
            if installed_record.is_none() {
                summary.marketplace_id = marketplace_id_for_path(source_path)
                    .unwrap_or_else(|| source_kind_for_marketplace_path(source_path).to_string());
            }
            if matches_query(&summary, params.query.as_deref())
                && matches_source(&summary, params.source.as_deref())
            {
                summaries.push(summary);
            }
        }
    }

    if !explicit_source_paths {
        for record in installed.values() {
            let package_root = installed_package_root(plugin_data_root, record)?;
            let already_visible = summaries.iter().any(|summary: &PluginCatalogSummary| {
                summary.id == record_string(record, "pluginId").unwrap_or_default()
                    && summary.version
                        == record_string(record, "installedVersion").unwrap_or_default()
            });
            if already_visible {
                continue;
            }
            let summary = build_summary(
                &package_root,
                record_string(record, "sourceKind").unwrap_or_else(|| "local".to_string()),
                record_string(record, "sourceUri").unwrap_or_default(),
                Some(record),
            )?;
            if matches_query(&summary, params.query.as_deref())
                && matches_source(&summary, params.source.as_deref())
            {
                summaries.push(summary);
            }
        }
    }

    summaries.sort_by(|left, right| left.id.cmp(&right.id));
    summaries.dedup_by(|left, right| left.id == right.id && left.version == right.version);
    Ok(PluginCatalogListResponse {
        plugins: summaries,
        generated_at: now_iso(),
    })
}

pub(crate) fn installed(
    plugin_data_root: &Path,
    _params: PluginCatalogInstalledParams,
) -> Result<PluginCatalogListResponse, String> {
    let mut summaries = Vec::new();
    for record in read_installed_records(plugin_data_root)?.values() {
        let package_root = installed_package_root(plugin_data_root, record)?;
        summaries.push(build_summary(
            &package_root,
            record_string(record, "sourceKind").unwrap_or_else(|| "local".to_string()),
            record_string(record, "sourceUri").unwrap_or_default(),
            Some(record),
        )?);
    }
    summaries.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(PluginCatalogListResponse {
        plugins: summaries,
        generated_at: now_iso(),
    })
}

pub(crate) fn list_apps(
    plugin_data_root: &Path,
    params: AppsListParams,
) -> Result<AppsListResponse, String> {
    const DEFAULT_LIMIT: u32 = 50;
    const MAX_LIMIT: u32 = 100;

    let apps = app_catalog(plugin_data_root)?;
    let offset = params
        .cursor
        .as_deref()
        .unwrap_or("0")
        .parse::<usize>()
        .map_err(|_| "app/list cursor 必须是非负整数。".to_string())?;
    let limit = params.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT) as usize;
    let start = offset.min(apps.len());
    let end = start.saturating_add(limit).min(apps.len());

    Ok(AppsListResponse {
        data: apps[start..end].to_vec(),
        next_cursor: (end < apps.len()).then(|| end.to_string()),
    })
}

pub(crate) fn read_apps(
    plugin_data_root: &Path,
    params: AppsReadParams,
) -> Result<AppsReadResponse, String> {
    const MAX_APP_IDS: usize = 100;

    if params.app_ids.len() > MAX_APP_IDS {
        return Err(format!(
            "app/read appIds 最多允许 {MAX_APP_IDS} 项，实际为 {}。",
            params.app_ids.len()
        ));
    }

    let catalog = app_catalog(plugin_data_root)?
        .into_iter()
        .map(|app| (app.id.clone(), app))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut seen = std::collections::BTreeSet::new();
    let mut apps = Vec::new();
    let mut missing_app_ids = Vec::new();

    for app_id in params.app_ids {
        if !seen.insert(app_id.clone()) {
            continue;
        }
        let Some(app) = catalog.get(&app_id) else {
            missing_app_ids.push(app_id);
            continue;
        };
        apps.push(ConnectorMetadata {
            id: app.id.clone(),
            name: app.name.clone(),
            description: app.description.clone(),
            icon_url: app.logo_url.clone(),
            icon_url_dark: app.logo_url_dark.clone(),
            distribution_channel: app.distribution_channel.clone(),
            install_url: app.install_url.clone(),
            plugin_display_names: app.plugin_display_names.clone(),
            tool_summaries: params.include_tools.then(Vec::new),
        });
    }

    Ok(AppsReadResponse {
        apps,
        missing_app_ids,
    })
}

pub(crate) fn installed_apps(
    plugin_data_root: &Path,
    _params: AppsInstalledParams,
) -> Result<AppsInstalledResponse, String> {
    Ok(AppsInstalledResponse {
        apps: app_catalog(plugin_data_root)?
            .into_iter()
            .map(|app| InstalledApp {
                id: app.id,
                runtime_name: Some(app.name),
                enabled: app.is_enabled,
                // Local Plugin apps currently have no hosted connector tool snapshot.
                callable: false,
            })
            .collect(),
    })
}

fn app_catalog(plugin_data_root: &Path) -> Result<Vec<AppInfo>, String> {
    let installed = installed(plugin_data_root, PluginCatalogInstalledParams::default())?;
    let mut apps = std::collections::BTreeMap::<String, AppInfo>::new();

    for plugin in installed.plugins {
        let detail = read(
            plugin_data_root,
            PluginCatalogReadParams {
                plugin_id: plugin.id.clone(),
            },
        )?
        .plugin;

        for capability in detail.apps {
            let app = apps
                .entry(capability.id.clone())
                .or_insert_with(|| AppInfo {
                    id: capability.id.clone(),
                    name: capability.name.clone(),
                    description: (!capability.description.is_empty())
                        .then_some(capability.description.clone()),
                    logo_url: None,
                    logo_url_dark: None,
                    icon_assets: None,
                    icon_dark_assets: None,
                    distribution_channel: Some(plugin.source.clone()),
                    branding: None,
                    app_metadata: None,
                    labels: None,
                    install_url: None,
                    is_accessible: plugin.enabled && !capability.requires_auth,
                    is_enabled: plugin.enabled,
                    plugin_display_names: Vec::new(),
                });
            app.is_enabled |= plugin.enabled;
            app.is_accessible |= plugin.enabled && !capability.requires_auth;
            if !app.plugin_display_names.contains(&plugin.name) {
                app.plugin_display_names.push(plugin.name.clone());
            }
        }
    }

    Ok(apps.into_values().collect())
}

pub(crate) fn search(
    plugin_data_root: &Path,
    params: PluginSearchParams,
) -> Result<PluginSearchResponse, String> {
    const DEFAULT_LIMIT: u32 = 16;
    const MAX_LIMIT: u32 = 1_000;

    let search_term = params.search_term.trim();
    if search_term.is_empty() {
        return Ok(PluginSearchResponse {
            data: Vec::new(),
            next_cursor: None,
        });
    }

    let source = match params.scope {
        Some(PluginSearchScope::Global) => Some("bundled".to_string()),
        Some(PluginSearchScope::Workspace) => Some("repo".to_string()),
        Some(PluginSearchScope::Personal) => Some("personal".to_string()),
        None => None,
    };
    let marketplace_paths = params
        .cwds
        .unwrap_or_default()
        .into_iter()
        .map(PathBuf::from)
        .map(|cwd| cwd.join(".agents/plugins/marketplace.json"))
        .filter(|path| path.is_file())
        .map(|path| path.to_string_lossy().into_owned())
        .collect();
    let catalog = list(
        plugin_data_root,
        PluginCatalogListParams {
            query: Some(search_term.to_string()),
            source,
            marketplace_paths,
        },
    )?;

    let offset = params
        .cursor
        .as_deref()
        .unwrap_or("0")
        .parse::<usize>()
        .map_err(|_| "plugin/search cursor 必须是非负整数。".to_string())?;
    let limit = params.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT) as usize;
    let end = offset.saturating_add(limit).min(catalog.plugins.len());
    let data = catalog.plugins[offset.min(catalog.plugins.len())..end]
        .iter()
        .cloned()
        .map(plugin_search_result)
        .collect::<Vec<_>>();
    let next_cursor = (end < catalog.plugins.len()).then(|| end.to_string());

    Ok(PluginSearchResponse { data, next_cursor })
}

fn plugin_search_result(summary: PluginCatalogSummary) -> PluginSearchResult {
    let marketplace_name = summary.marketplace_id.clone();
    PluginSearchResult {
        marketplace_name,
        marketplace_path: None,
        plugin: PluginSummary {
            id: summary.id,
            remote_plugin_id: None,
            version: Some(summary.version.clone()),
            local_version: summary.local_version,
            name: summary.name.clone(),
            share_context: None,
            source: PluginSource::Local {
                path: summary.source_uri,
            },
            installed: summary.installed,
            installed_at: None,
            enabled: summary.enabled,
            install_policy: match summary.install_policy.as_str() {
                "NOT_AVAILABLE" => PluginInstallPolicy::NotAvailable,
                "INSTALLED_BY_DEFAULT" => PluginInstallPolicy::InstalledByDefault,
                _ => PluginInstallPolicy::Available,
            },
            install_policy_source: None,
            must_show_installation_interstitial: None,
            auth_policy: match summary.auth_policy.as_str() {
                "ON_INSTALL" => PluginAuthPolicy::OnInstall,
                _ => PluginAuthPolicy::OnUse,
            },
            availability: PluginAvailability::Available,
            disabled_reason: None,
            eligible_plan_types: None,
            interface: Some(PluginInterface {
                display_name: Some(summary.name),
                short_description: Some(summary.description),
                long_description: None,
                developer_name: None,
                category: None,
                capabilities: Vec::new(),
                website_url: None,
                privacy_policy_url: None,
                terms_of_service_url: None,
                default_prompt: None,
                brand_color: None,
                composer_icon: None,
                composer_icon_url: None,
                logo: None,
                logo_dark: None,
                logo_url: None,
                logo_url_dark: None,
                screenshots: Vec::new(),
                screenshot_urls: Vec::new(),
            }),
            keywords: Vec::new(),
        },
    }
}

pub(crate) fn enabled_plugin_turn_snapshots(
    plugin_data_root: &Path,
) -> Result<Vec<crate::runtime::PluginTurnSnapshot>, String> {
    let mut snapshots = Vec::new();
    for record in read_installed_records(plugin_data_root)?.values() {
        if !record
            .get("enabled")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            continue;
        }

        let package_root = installed_package_root(plugin_data_root, record)?;
        let manifest = read_manifest(&package_root)?;
        let id = manifest_name(&manifest)?;
        let marketplace_id = record_string(record, "marketplaceId").unwrap_or_else(|| {
            record_string(record, "sourceKind").unwrap_or_else(|| "local".to_string())
        });
        let config_name = format!("{id}@{marketplace_id}");
        let display_name = interface_string(&manifest, "displayName").unwrap_or_else(|| id.clone());
        let skill_names = skill_capabilities(&package_root)?
            .into_iter()
            .map(|skill| skill.id)
            .collect::<Vec<_>>();
        let mcp_server_names = plugin_mcp_server_descriptors(plugin_data_root, &package_root, &id)
            .unwrap_or_else(|error| {
                tracing::warn!(
                    plugin_id = %id,
                    %error,
                    "禁用包含非法 mcp.json 的 Agent Plugin MCP 组件"
                );
                Vec::new()
            })
            .into_iter()
            .map(|server| server.runtime_name)
            .collect::<Vec<_>>();
        snapshots.push(crate::runtime::PluginTurnSnapshot {
            id,
            config_name,
            display_name,
            package_root,
            skill_names,
            mcp_server_names,
        });
    }
    snapshots.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(snapshots)
}

pub(crate) fn list_plugin_mcp_runtime_server_specs(
    plugin_data_root: &Path,
) -> Result<Vec<McpRuntimeServerSpec>, String> {
    let mut specs = Vec::new();
    for record in read_installed_records(plugin_data_root)?.values() {
        if !record
            .get("enabled")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            continue;
        }

        let package_root = installed_package_root(plugin_data_root, record)?;
        let manifest = read_manifest(&package_root)?;
        let plugin_id = manifest_name(&manifest)?;
        let servers =
            match plugin_mcp_server_descriptors(plugin_data_root, &package_root, &plugin_id) {
                Ok(servers) => servers,
                Err(error) => {
                    tracing::warn!(
                        plugin_id,
                        %error,
                        "禁用包含非法 mcp.json 的 Agent Plugin MCP 组件"
                    );
                    continue;
                }
            };
        for server in servers {
            specs.push(McpRuntimeServerSpec {
                name: server.runtime_name,
                plugin_id: Some(plugin_id.clone()),
                config: server.config,
            });
        }
    }
    specs.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(specs)
}

struct PluginMcpServerDescriptor {
    id: String,
    runtime_name: String,
    config: McpServerConfig,
}

fn plugin_mcp_server_descriptors(
    plugin_data_root: &Path,
    package_root: &Path,
    plugin_id: &str,
) -> Result<Vec<PluginMcpServerDescriptor>, String> {
    let package_root = canonical_package_root(package_root)?;
    let path = package_root.join(STANDARD_MCP_CONFIG_PATH);
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let data_root = plugin_data_root.join("data").join(plugin_id);
    let content =
        fs::read_to_string(&path).map_err(io_error("读取 Agent Plugins mcp.json 失败"))?;
    let outcome = parse_agent_plugin_mcp_config(&package_root, &data_root, &content)?;
    let mut descriptors = Vec::new();
    for (id, config) in outcome.servers {
        let runtime_name = plugin_mcp_runtime_name(plugin_id, &id)?;
        descriptors.push(PluginMcpServerDescriptor {
            id: id.clone(),
            runtime_name,
            config,
        });
    }
    for error in outcome.errors {
        tracing::warn!(
            plugin_id,
            server_id = error.name,
            message = error.message,
            "跳过非法 Agent Plugins MCP server 配置"
        );
    }
    descriptors.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(descriptors)
}

fn plugin_mcp_runtime_name(plugin_id: &str, server_id: &str) -> Result<String, String> {
    let server_id = server_id.trim();
    if server_id.is_empty() || server_id.len() > 96 || server_id.contains('\0') {
        return Err(format!("Plugin MCP server id 非法: {server_id}"));
    }
    Ok(format!("plugin__{plugin_id}__{server_id}"))
}

pub(crate) fn set_enabled(
    plugin_data_root: &Path,
    params: app_server_protocol::protocol::v2::PluginCatalogEnabledSetParams,
) -> Result<app_server_protocol::protocol::v2::PluginCatalogEnabledSetResponse, String> {
    validate_plugin_id(&params.plugin_id)?;
    let records = read_installed_records(plugin_data_root)?;
    let mut record = records
        .get(&params.plugin_id)
        .cloned()
        .ok_or_else(|| format!("Plugin 未安装: {}", params.plugin_id))?;
    record
        .as_object_mut()
        .ok_or_else(|| "Plugin installed record 必须是对象。".to_string())?
        .insert("enabled".to_string(), Value::Bool(params.enabled));
    write_installed_record(plugin_data_root, &record)?;
    let package_root = installed_package_root(plugin_data_root, &record)?;
    let summary = build_summary(
        &package_root,
        record_string(&record, "sourceKind").unwrap_or_else(|| "local".to_string()),
        record_string(&record, "sourceUri").unwrap_or_default(),
        Some(&record),
    )?;
    Ok(app_server_protocol::protocol::v2::PluginCatalogEnabledSetResponse { plugin: summary })
}

pub(crate) fn read(
    plugin_data_root: &Path,
    params: PluginCatalogReadParams,
) -> Result<PluginCatalogReadResponse, String> {
    validate_plugin_id(&params.plugin_id)?;
    let records = read_installed_records(plugin_data_root)?;
    let record = records
        .get(&params.plugin_id)
        .ok_or_else(|| format!("Plugin 未安装: {}", params.plugin_id))?;
    let package_root = installed_package_root(plugin_data_root, record)?;
    let summary = build_summary(
        &package_root,
        record_string(record, "sourceKind").unwrap_or_else(|| "local".to_string()),
        record_string(record, "sourceUri").unwrap_or_default(),
        Some(record),
    )?;
    Ok(PluginCatalogReadResponse {
        plugin: build_detail(&package_root, summary)?,
    })
}

pub(crate) fn install(
    plugin_data_root: &Path,
    params: PluginCatalogInstallParams,
) -> Result<PluginCatalogInstallResponse, String> {
    let source_root = canonical_package_root(Path::new(&params.source_path))?;
    let store_root = plugin_data_root.join(STORE_DIR);
    let canonical_store =
        fs::canonicalize(plugin_data_root).unwrap_or_else(|_| plugin_data_root.to_path_buf());
    if source_root.starts_with(&canonical_store) {
        return Err("Plugin source 不能位于 installed store 内。".to_string());
    }
    let manifest = read_manifest(&source_root)?;
    let plugin_id = manifest_name(&manifest)?;
    let version = manifest_version(&manifest)?;
    let content_digest = package_digest(&source_root)?;
    if params
        .expected_digest
        .as_deref()
        .is_some_and(|expected| expected != content_digest)
    {
        return Err(format!(
            "Plugin package digest 不一致: expected {}, actual {content_digest}",
            params.expected_digest.as_deref().unwrap_or_default()
        ));
    }
    let installed_records = read_installed_records(plugin_data_root)?;
    let previous_record = installed_records.get(&plugin_id).cloned();
    if let Some(record) = previous_record.as_ref() {
        let installed_version = record_string(record, "installedVersion").unwrap_or_default();
        let installed_digest = record_string(record, "contentDigest")
            .or_else(|| {
                installed_package_root(plugin_data_root, record)
                    .ok()
                    .and_then(|root| package_digest(&root).ok())
            })
            .unwrap_or_default();
        if installed_version == version {
            if installed_digest == content_digest {
                let package_root = installed_package_root(plugin_data_root, record)?;
                return Ok(PluginCatalogInstallResponse {
                    plugin: build_summary(
                        &package_root,
                        record_string(record, "sourceKind").unwrap_or_else(|| "local".to_string()),
                        record_string(record, "sourceUri").unwrap_or_default(),
                        Some(record),
                    )?,
                });
            }
            return Err(format!(
                "Plugin package identity 冲突: {plugin_id}@{version} 的 content digest 不同"
            ));
        }
    }
    let final_root = store_root
        .join(PACKAGES_DIR)
        .join(&plugin_id)
        .join(&version);
    if final_root.exists() {
        let existing_digest = package_digest(&final_root)?;
        if existing_digest != content_digest {
            return Err(format!(
                "Plugin package 已存在且 digest 冲突: {plugin_id}@{version}"
            ));
        }
    }

    let staging_root = store_root.join(STAGING_DIR).join(format!(
        "{plugin_id}-{version}-{}",
        Utc::now().timestamp_nanos_opt().unwrap_or_default()
    ));
    fs::create_dir_all(&staging_root).map_err(io_error("创建 Plugin staging 目录失败"))?;
    if let Err(error) = copy_package_tree(&source_root, &staging_root) {
        let _ = fs::remove_dir_all(&staging_root);
        return Err(error);
    }
    let _staging_manifest = match read_manifest(&staging_root) {
        Ok(manifest) => manifest,
        Err(error) => {
            let _ = fs::remove_dir_all(&staging_root);
            return Err(format!("校验 Plugin staging manifest 失败: {error}"));
        }
    };
    if let Err(error) = build_summary(&staging_root, "local", &params.source_path, None) {
        let _ = fs::remove_dir_all(&staging_root);
        return Err(format!("校验 Plugin staging capability 失败: {error}"));
    }
    fs::create_dir_all(final_root.parent().expect("version has parent"))
        .map_err(io_error("创建 Plugin package 目录失败"))?;
    if package_digest(&staging_root)? != content_digest {
        let _ = fs::remove_dir_all(&staging_root);
        return Err("Plugin staging digest 与 source 不一致。".to_string());
    }
    if !final_root.exists() {
        fs::rename(&staging_root, &final_root).map_err(io_error("提交 Plugin package 失败"))?;
    } else {
        fs::remove_dir_all(&staging_root).map_err(io_error("清理 Plugin staging 目录失败"))?;
    }

    let package_root_locator = format!("{PACKAGES_DIR}/{plugin_id}/{version}");
    let source_kind = params.source.as_deref().unwrap_or("local");
    let marketplace_id = params.marketplace_id.as_deref().unwrap_or(source_kind);
    let record = serde_json::json!({
        "schemaVersion": INSTALLED_SCHEMA_VERSION,
        "pluginId": plugin_id.clone(),
        "installedVersion": version.clone(),
        "marketplaceId": marketplace_id,
        "contentDigest": content_digest,
        "sourceKind": source_kind,
        "sourceUri": params.source_path,
        "enabled": true,
        "installedAt": now_iso(),
        "packageRoot": package_root_locator,
    });
    if let Err(error) = write_installed_record(plugin_data_root, &record) {
        if previous_record.is_none() {
            let _ = fs::remove_dir_all(&final_root);
        }
        return Err(error);
    }
    if let Some(previous_record) = previous_record.as_ref() {
        if let Ok(previous_root) = installed_package_root(plugin_data_root, previous_record) {
            if previous_root != final_root && previous_root.exists() {
                fs::remove_dir_all(previous_root)
                    .map_err(io_error("清理 Plugin 旧版本 package 失败"))?;
            }
        }
    }
    let summary = build_summary(
        &final_root,
        "local",
        record_string(&record, "sourceUri")
            .as_deref()
            .unwrap_or_default(),
        Some(&record),
    )?;
    Ok(PluginCatalogInstallResponse { plugin: summary })
}

pub(crate) fn uninstall(
    plugin_data_root: &Path,
    params: PluginCatalogUninstallParams,
) -> Result<PluginCatalogUninstallResponse, String> {
    validate_plugin_id(&params.plugin_id)?;
    let records = read_installed_records(plugin_data_root)?;
    let Some(record) = records.get(&params.plugin_id) else {
        return Ok(PluginCatalogUninstallResponse {
            plugin_id: params.plugin_id,
            uninstalled: false,
        });
    };
    let package_root = installed_package_root(plugin_data_root, record)?;
    if package_root.exists() {
        fs::remove_dir_all(&package_root).map_err(io_error("删除 Plugin package 失败"))?;
    }
    let record_path = installed_record_path(plugin_data_root, &params.plugin_id)?;
    if record_path.exists() {
        fs::remove_file(record_path).map_err(io_error("删除 Plugin installed record 失败"))?;
    }
    Ok(PluginCatalogUninstallResponse {
        plugin_id: params.plugin_id,
        uninstalled: true,
    })
}

fn build_summary<S: AsRef<str>, U: AsRef<str>>(
    package_root: &Path,
    source: S,
    source_uri: U,
    installed_record: Option<&Value>,
) -> Result<PluginCatalogSummary, String> {
    let manifest = read_manifest(package_root)?;
    let plugin_id = manifest_name(&manifest)?;
    let version = manifest_version(&manifest)?;
    let detail = build_capability_detail(package_root, &manifest)?;
    let installed = installed_record.is_some();
    let enabled = installed_record
        .and_then(|record| record.get("enabled"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Ok(PluginCatalogSummary {
        name: interface_string(&manifest, "displayName").unwrap_or_else(|| plugin_id.clone()),
        id: plugin_id,
        version,
        marketplace_id: installed_record
            .and_then(|record| record_string(record, "marketplaceId"))
            .unwrap_or_else(|| source.as_ref().to_string()),
        content_digest: match installed_record
            .and_then(|record| record_string(record, "contentDigest"))
        {
            Some(content_digest) => content_digest,
            None => package_digest(package_root)?,
        },
        description: interface_string(&manifest, "shortDescription")
            .or_else(|| manifest_string(&manifest, "description"))
            .unwrap_or_default(),
        source: source.as_ref().to_string(),
        source_uri: source_uri.as_ref().to_string(),
        installed,
        enabled,
        install_policy: "AVAILABLE".to_string(),
        auth_policy: "ON_USE".to_string(),
        availability: if installed { "installed" } else { "available" }.to_string(),
        disabled_reason: if installed && !enabled {
            Some("Plugin 当前未启用。".to_string())
        } else {
            None
        },
        local_version: installed_record
            .and_then(|record| record_string(record, "installedVersion")),
        skills_count: detail.skills.len() as u32,
        mcp_servers_count: detail.mcp_servers.len() as u32,
        apps_count: detail.apps.len() as u32,
        hooks_count: detail.hooks.len() as u32,
    })
}

fn build_detail(
    package_root: &Path,
    summary: PluginCatalogSummary,
) -> Result<PluginCatalogDetail, String> {
    let manifest = read_manifest(package_root)?;
    let detail = build_capability_detail(package_root, &manifest)?;
    Ok(PluginCatalogDetail {
        summary,
        skills: detail.skills,
        mcp_servers: detail.mcp_servers,
        apps: detail.apps,
        hooks: detail.hooks,
        ui_resources: detail.ui_resources,
    })
}

struct CapabilityDetail {
    skills: Vec<PluginCatalogCapability>,
    mcp_servers: Vec<PluginCatalogCapability>,
    apps: Vec<PluginCatalogCapability>,
    hooks: Vec<PluginCatalogHook>,
    ui_resources: Vec<PluginCatalogUiResource>,
}

fn build_capability_detail(
    package_root: &Path,
    _manifest: &Value,
) -> Result<CapabilityDetail, String> {
    let skills = skill_capabilities(package_root)?;
    let mcp_servers = mcp_capabilities(package_root)?;
    Ok(CapabilityDetail {
        skills,
        mcp_servers,
        apps: Vec::new(),
        hooks: Vec::new(),
        ui_resources: Vec::new(),
    })
}

fn skill_capabilities(package_root: &Path) -> Result<Vec<PluginCatalogCapability>, String> {
    let root = package_root.join("skills");
    if !root.is_dir() {
        return Ok(Vec::new());
    }
    let mut capabilities = Vec::new();
    for entry in fs::read_dir(&root).map_err(io_error("读取 Agent Plugins skills 目录失败"))?
    {
        let entry = entry.map_err(io_error("读取 Agent Plugins skill 条目失败"))?;
        let path = entry.path();
        if path.is_dir() && path.join("SKILL.md").is_file() {
            let id = entry.file_name().to_string_lossy().into_owned();
            capabilities.push(PluginCatalogCapability {
                id: id.clone(),
                name: id,
                description: String::new(),
                requires_auth: false,
            });
        }
    }
    capabilities.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(capabilities)
}

fn mcp_capabilities(package_root: &Path) -> Result<Vec<PluginCatalogCapability>, String> {
    let path = package_root.join(STANDARD_MCP_CONFIG_PATH);
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let content =
        fs::read_to_string(&path).map_err(io_error("读取 Agent Plugins mcp.json 失败"))?;
    let value: Value = serde_json::from_str(&content)
        .map_err(|error| format!("解析 Agent Plugins mcp.json 失败: {error}"))?;
    if value.get("$schema").and_then(Value::as_str)
        != Some(lime_mcp::agent_plugin_config::AGENT_PLUGIN_MCP_SCHEMA_URI)
    {
        return Ok(Vec::new());
    }
    let Some(servers) = value.get("mcpServers").and_then(Value::as_object) else {
        return Ok(Vec::new());
    };
    Ok(servers
        .keys()
        .map(|id| PluginCatalogCapability {
            id: id.clone(),
            name: id.clone(),
            description: String::new(),
            requires_auth: false,
        })
        .collect())
}

fn discover_package_roots(path: &Path) -> Result<Vec<PathBuf>, String> {
    let path = fs::canonicalize(path).map_err(io_error("解析 Plugin catalog source 失败"))?;
    if path.is_file() {
        let content = fs::read_to_string(&path).map_err(io_error("读取 marketplace 文件失败"))?;
        let value: Value = serde_json::from_str(&content)
            .map_err(|error| format!("解析 marketplace 文件失败: {error}"))?;
        let source_root = marketplace_source_root(&path)?;
        let canonical_source_root =
            fs::canonicalize(&source_root).map_err(io_error("解析 marketplace root 失败"))?;
        let mut roots = Vec::new();
        for entry in value
            .get("plugins")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(relative) = entry
                .get("source")
                .and_then(Value::as_object)
                .and_then(|source| source.get("path"))
                .and_then(Value::as_str)
            else {
                continue;
            };
            let package_root = canonical_package_root(&source_root.join(relative))?;
            if !package_root.starts_with(&canonical_source_root) {
                return Err(format!("Plugin marketplace local source 越界: {relative}"));
            }
            roots.push(package_root);
        }
        return Ok(roots);
    }
    if path.join("plugin.json").is_file() {
        return Ok(vec![path]);
    }
    let mut roots = Vec::new();
    for entry in fs::read_dir(&path).map_err(io_error("读取 Plugin catalog 目录失败"))? {
        let entry = entry.map_err(io_error("读取 Plugin catalog 条目失败"))?;
        let child = entry.path();
        if child.is_dir()
            && !entry.file_name().to_string_lossy().starts_with('.')
            && child.join("plugin.json").is_file()
        {
            roots.push(child);
        }
    }
    Ok(roots)
}

fn marketplace_source_root(path: &Path) -> Result<PathBuf, String> {
    for relative_path in [
        ".agents/plugins/marketplace.json",
        ".agents/plugins/api_marketplace.json",
        ".claude-plugin/marketplace.json",
        ".cursor-plugin/marketplace.json",
    ] {
        let relative_path = Path::new(relative_path);
        if !path.ends_with(relative_path) {
            continue;
        }
        let mut root = path.to_path_buf();
        for _ in relative_path.components() {
            root.pop();
        }
        return Ok(root);
    }
    path.parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "Plugin marketplace 文件缺少父目录。".to_string())
}

fn implicit_marketplace_paths(plugin_data_root: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(path) = bundled_marketplace_path_from_env() {
        paths.push(path);
    }
    if let Some(home) = dirs::home_dir() {
        paths.push(home.join(".agents/plugins/marketplace.json"));
    }
    if let Ok(current_dir) = std::env::current_dir() {
        paths.push(current_dir.join(".agents/plugins/marketplace.json"));
    }
    let configured_root = plugin_data_root.join(STORE_DIR).join("marketplaces");
    if let Ok(entries) = fs::read_dir(configured_root) {
        paths.extend(entries.flatten().map(|entry| entry.path()).filter(|path| {
            path.extension().and_then(|extension| extension.to_str()) == Some("json")
        }));
    }
    paths.retain(|path| path.is_file());
    paths.sort();
    paths.dedup();
    paths
}

fn bundled_marketplace_path_from_env() -> Option<PathBuf> {
    std::env::var_os("LIME_BUNDLED_PLUGIN_MARKETPLACE")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
}

fn source_kind_for_marketplace_path(path: &Path) -> &'static str {
    if bundled_marketplace_path_from_env().is_some_and(|bundled_path| path == bundled_path) {
        return "bundled";
    }
    if dirs::home_dir()
        .map(|home| path == home.join(".agents/plugins/marketplace.json"))
        .unwrap_or(false)
    {
        return "personal";
    }
    if std::env::current_dir()
        .map(|current_dir| path == current_dir.join(".agents/plugins/marketplace.json"))
        .unwrap_or(false)
    {
        return "repo";
    }
    "local"
}

fn marketplace_id_for_path(path: &Path) -> Option<String> {
    let content = fs::read_to_string(path).ok()?;
    let value: Value = serde_json::from_str(&content).ok()?;
    value
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(ToString::to_string)
}

fn read_manifest(package_root: &Path) -> Result<Value, String> {
    let root = canonical_package_root(package_root)?;
    let path = root.join("plugin.json");
    let metadata =
        fs::symlink_metadata(&path).map_err(io_error("读取 Plugin manifest 元数据失败"))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err("Agent Plugins 根 plugin.json 必须是普通文件。".to_string());
    }
    if metadata.len() > MAX_MANIFEST_BYTES {
        return Err("Plugin manifest 超过大小限制。".to_string());
    }
    let content = fs::read_to_string(&path).map_err(io_error("读取 Plugin manifest 失败"))?;
    let value: Value = serde_json::from_str(&content)
        .map_err(|error| format!("解析 Agent Plugins plugin.json 失败: {error}"))?;
    let Value::Object(mut object) = value else {
        return Err("Agent Plugins plugin.json 必须是 object。".to_string());
    };
    for field in object.keys() {
        if !STANDARD_MANIFEST_FIELDS.contains(&field.as_str()) {
            tracing::warn!(
                path = %path.display(),
                field,
                "忽略未知 Agent Plugins manifest 字段"
            );
        }
    }
    object.retain(|field, _| STANDARD_MANIFEST_FIELDS.contains(&field.as_str()));
    if object
        .get("extensions")
        .is_some_and(|extensions| !extensions.is_object())
    {
        tracing::warn!(
            path = %path.display(),
            "忽略非 object 的 Agent Plugins extensions 字段"
        );
        object.remove("extensions");
    }
    if object.get("$schema").and_then(Value::as_str) != Some(STANDARD_MANIFEST_SCHEMA) {
        return Err(format!(
            "plugin.json 必须声明标准 schema `{STANDARD_MANIFEST_SCHEMA}`。"
        ));
    }
    validate_standard_manifest_fields(&object)?;

    let mut manifest = Value::Object(object);
    apply_codex_manifest_extension(&root, &path, &mut manifest)?;
    manifest_name(&manifest)?;
    manifest_version(&manifest)?;
    Ok(manifest)
}

fn validate_standard_manifest_fields(
    object: &serde_json::Map<String, Value>,
) -> Result<(), String> {
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| "Plugin manifest 缺少合法 name。".to_string())?;
    validate_plugin_id(name)?;

    for field in [
        "version",
        "description",
        "homepage",
        "repository",
        "license",
    ] {
        if object.get(field).is_some_and(|value| !value.is_string()) {
            return Err(format!("Agent Plugins `{field}` 必须是 string。"));
        }
    }

    if let Some(author) = object.get("author") {
        let author = author
            .as_object()
            .ok_or_else(|| "Agent Plugins `author` 必须是 object。".to_string())?;
        for (field, value) in author {
            if !["name", "email", "url"].contains(&field.as_str()) {
                return Err(format!("Agent Plugins `author` 包含未知字段 `{field}`。"));
            }
            if !value.is_string() {
                return Err(format!("Agent Plugins `author.{field}` 必须是 string。"));
            }
        }
    }

    if let Some(keywords) = object.get("keywords") {
        let keywords = keywords
            .as_array()
            .ok_or_else(|| "Agent Plugins `keywords` 必须是 string array。".to_string())?;
        if keywords.iter().any(|keyword| !keyword.is_string()) {
            return Err("Agent Plugins `keywords` 必须是 string array。".to_string());
        }
    }
    Ok(())
}

fn apply_codex_manifest_extension(
    package_root: &Path,
    manifest_path: &Path,
    manifest: &mut Value,
) -> Result<(), String> {
    let inline_extension = manifest
        .get("extensions")
        .and_then(Value::as_object)
        .and_then(|extensions| extensions.get(CODEX_EXTENSION_NAMESPACE))
        .and_then(|extension| {
            if extension.is_object() {
                Some(extension.clone())
            } else {
                tracing::warn!(
                    path = %manifest_path.display(),
                    namespace = CODEX_EXTENSION_NAMESPACE,
                    "忽略非 object 的 Agent Plugins client extension"
                );
                None
            }
        });
    let extension = match inline_extension {
        Some(extension) => Some(extension),
        None => {
            let overlay_path = package_root.join(CODEX_EXTENSION_MANIFEST_PATH);
            match fs::read_to_string(&overlay_path) {
                Ok(contents) => Some(
                    serde_json::from_str(&contents)
                        .map_err(|error| format!("解析 Codex Plugin extension 失败: {error}"))?,
                ),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                Err(error) => return Err(format!("读取 Codex Plugin extension 失败: {error}")),
            }
        }
    };
    let Some(extension) = extension else {
        return Ok(());
    };
    let extension = extension
        .as_object()
        .ok_or_else(|| "Codex Plugin extension 必须是 object。".to_string())?;
    if let Some(interface) = extension.get("interface") {
        if !interface.is_object() {
            return Err("Codex Plugin extension interface 必须是 object。".to_string());
        }
        manifest
            .as_object_mut()
            .expect("validated manifest object")
            .insert("interface".to_string(), interface.clone());
    }
    Ok(())
}

fn manifest_name(manifest: &Value) -> Result<String, String> {
    let name = manifest_string(manifest, "name")
        .ok_or_else(|| "Plugin manifest 缺少 name。".to_string())?;
    validate_plugin_id(&name)?;
    Ok(name)
}

fn manifest_version(manifest: &Value) -> Result<String, String> {
    let version = manifest_string(manifest, "version").unwrap_or_else(|| "0.0.0".to_string());
    if version.len() > 128
        || version == "."
        || version == ".."
        || version.contains('/')
        || version.contains('\\')
        || version.contains('\0')
    {
        return Err("Plugin version 不能作为安全的 package 路径。".to_string());
    }
    Ok(version)
}

fn package_digest(package_root: &Path) -> Result<String, String> {
    let package_root = canonical_package_root(package_root)?;
    let mut files = Vec::new();
    collect_package_files(&package_root, &package_root, &mut files)?;
    files.sort_by(|left, right| left.0.cmp(&right.0));
    let mut hasher = Sha256::new();
    for (relative, path) in files {
        hasher.update((relative.len() as u64).to_be_bytes());
        hasher.update(relative.as_bytes());
        let mut file = fs::File::open(path).map_err(io_error("读取 Plugin digest 文件失败"))?;
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let read = file
                .read(&mut buffer)
                .map_err(io_error("计算 Plugin package digest 失败"))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
    }
    Ok(format!("sha256:{}", hex::encode(hasher.finalize())))
}

fn collect_package_files(
    package_root: &Path,
    current: &Path,
    files: &mut Vec<(String, PathBuf)>,
) -> Result<(), String> {
    for entry in fs::read_dir(current).map_err(io_error("读取 Plugin package 目录失败"))? {
        let entry = entry.map_err(io_error("读取 Plugin package 条目失败"))?;
        let path = entry.path();
        let metadata =
            fs::symlink_metadata(&path).map_err(io_error("读取 Plugin package 元数据失败"))?;
        if metadata.file_type().is_symlink() {
            return Err(format!(
                "Plugin package 禁止包含符号链接: {}",
                path.display()
            ));
        }
        if metadata.is_dir() {
            collect_package_files(package_root, &path, files)?;
        } else if metadata.is_file() {
            let relative = path
                .strip_prefix(package_root)
                .map_err(|error| format!("计算 Plugin package 相对路径失败: {error}"))?
                .components()
                .map(|component| component.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/");
            files.push((relative, path));
        }
    }
    Ok(())
}

fn manifest_string(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn interface_string(manifest: &Value, key: &str) -> Option<String> {
    manifest
        .get("interface")
        .and_then(Value::as_object)
        .and_then(|interface| interface.get(key))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn validate_plugin_id(id: &str) -> Result<(), String> {
    if id.is_empty()
        || id.len() > 64
        || !id
            .chars()
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-' || ch == '.')
        || !id.as_bytes().first().is_some_and(u8::is_ascii_alphanumeric)
        || !id.as_bytes().last().is_some_and(u8::is_ascii_lowercase)
            && !id.as_bytes().last().is_some_and(u8::is_ascii_digit)
        || id.contains("--")
        || id.contains("..")
    {
        return Err(format!(
            "Plugin name 不符合 Agent Plugins identity 规则: {id}"
        ));
    }
    Ok(())
}

fn canonical_package_root(path: &Path) -> Result<PathBuf, String> {
    let root = fs::canonicalize(path).map_err(io_error("解析 Plugin package root 失败"))?;
    if !root.is_dir() {
        return Err(format!("Plugin package root 不是目录: {}", root.display()));
    }
    Ok(root)
}

fn resource_path(package_root: &Path, relative: &str) -> Result<PathBuf, String> {
    let relative_path = Path::new(relative);
    if relative_path.is_absolute()
        || relative_path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(format!("Plugin resource path 越界: {relative}"));
    }
    let package_root =
        fs::canonicalize(package_root).map_err(io_error("解析 Plugin package root 失败"))?;
    let resolved = fs::canonicalize(package_root.join(relative_path))
        .map_err(io_error("解析 Plugin resource 失败"))?;
    if !resolved.starts_with(&package_root) {
        return Err(format!("Plugin resource path 越界: {relative}"));
    }
    Ok(resolved)
}

fn copy_package_tree(source_root: &Path, destination_root: &Path) -> Result<(), String> {
    let mut files = 0usize;
    let mut bytes = 0u64;
    copy_package_entry(source_root, destination_root, &mut files, &mut bytes)
}

fn copy_package_entry(
    source: &Path,
    destination: &Path,
    files: &mut usize,
    bytes: &mut u64,
) -> Result<(), String> {
    let metadata =
        fs::symlink_metadata(source).map_err(io_error("读取 Plugin package 条目失败"))?;
    if metadata.file_type().is_symlink() {
        return Err(format!(
            "Plugin package 禁止包含符号链接: {}",
            source.display()
        ));
    }
    if metadata.is_dir() {
        fs::create_dir_all(destination).map_err(io_error("创建 Plugin package 条目失败"))?;
        for entry in fs::read_dir(source).map_err(io_error("读取 Plugin package 目录失败"))? {
            let entry = entry.map_err(io_error("读取 Plugin package 条目失败"))?;
            let entry_path = entry.path();
            let destination_path = destination.join(entry.file_name());
            copy_package_entry(&entry_path, &destination_path, files, bytes)?;
        }
    } else if metadata.is_file() {
        *files += 1;
        *bytes = bytes.saturating_add(metadata.len());
        if *files > MAX_PACKAGE_FILES || *bytes > MAX_PACKAGE_BYTES {
            return Err("Plugin package 超过文件数量或总大小限制。".to_string());
        }
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).map_err(io_error("创建 Plugin package 文件目录失败"))?;
        }
        fs::copy(source, destination).map_err(io_error("复制 Plugin package 文件失败"))?;
    }
    Ok(())
}

fn read_installed_records(
    plugin_data_root: &Path,
) -> Result<std::collections::BTreeMap<String, Value>, String> {
    let directory = plugin_data_root.join(STORE_DIR).join(INSTALLED_DIR);
    fs::create_dir_all(&directory).map_err(io_error("创建 Agent Plugins installed 目录失败"))?;
    let mut records = std::collections::BTreeMap::new();
    for entry in
        fs::read_dir(directory).map_err(io_error("读取 Agent Plugins installed 目录失败"))?
    {
        let entry = entry.map_err(io_error("读取 Plugin installed record 失败"))?;
        if entry.path().extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let content = fs::read_to_string(entry.path())
            .map_err(io_error("读取 Plugin installed record 失败"))?;
        let record: Value = serde_json::from_str(&content)
            .map_err(|error| format!("解析 Plugin installed record 失败: {error}"))?;
        if record.get("schemaVersion").and_then(Value::as_u64) != Some(INSTALLED_SCHEMA_VERSION) {
            return Err(format!(
                "不支持的 Agent Plugins installed schema: {}",
                entry.path().display()
            ));
        }
        let id = record_string(&record, "pluginId")
            .ok_or_else(|| "Plugin installed record 缺少 pluginId。".to_string())?;
        validate_plugin_id(&id)?;
        records.insert(id, record);
    }
    Ok(records)
}

fn installed_record_path(plugin_data_root: &Path, plugin_id: &str) -> Result<PathBuf, String> {
    validate_plugin_id(plugin_id)?;
    Ok(plugin_data_root
        .join(STORE_DIR)
        .join(INSTALLED_DIR)
        .join(format!("{plugin_id}.json")))
}

fn installed_package_root(plugin_data_root: &Path, record: &Value) -> Result<PathBuf, String> {
    let relative = record_string(record, "packageRoot")
        .ok_or_else(|| "Plugin installed record 缺少 packageRoot。".to_string())?;
    let path = resource_path(&plugin_data_root.join(STORE_DIR), &relative)?;
    let packages_root = fs::canonicalize(plugin_data_root.join(STORE_DIR).join(PACKAGES_DIR))
        .map_err(io_error("解析 Plugin package store 失败"))?;
    if !path.starts_with(&packages_root) {
        return Err("Plugin installed record 的 packageRoot 越界。".to_string());
    }
    Ok(path)
}

fn write_installed_record(plugin_data_root: &Path, record: &Value) -> Result<(), String> {
    let id = record_string(record, "pluginId")
        .ok_or_else(|| "Plugin installed record 缺少 pluginId。".to_string())?;
    let path = installed_record_path(plugin_data_root, &id)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(io_error("创建 Agent Plugins installed 目录失败"))?;
    }
    let temporary = path.with_extension("json.tmp");
    fs::write(
        &temporary,
        serde_json::to_vec_pretty(record)
            .map_err(|error| format!("序列化 Plugin installed record 失败: {error}"))?,
    )
    .map_err(io_error("写入 Plugin installed record 失败"))?;
    fs::rename(temporary, path).map_err(io_error("提交 Plugin installed record 失败"))
}

fn record_string(record: &Value, key: &str) -> Option<String> {
    record
        .get(key)
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

fn matches_query(summary: &PluginCatalogSummary, query: Option<&str>) -> bool {
    let Some(query) = query.map(str::trim).filter(|query| !query.is_empty()) else {
        return true;
    };
    let query = query.to_ascii_lowercase();
    summary.id.to_ascii_lowercase().contains(&query)
        || summary.name.to_ascii_lowercase().contains(&query)
        || summary.description.to_ascii_lowercase().contains(&query)
}

fn matches_source(summary: &PluginCatalogSummary, source: Option<&str>) -> bool {
    source
        .map(str::trim)
        .filter(|source| !source.is_empty())
        .is_none_or(|source| summary.source == source)
}

fn io_error(prefix: &'static str) -> impl FnOnce(std::io::Error) -> String {
    move |error| format!("{prefix}: {error}")
}

fn now_iso() -> String {
    Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

#[cfg(test)]
mod tests;
