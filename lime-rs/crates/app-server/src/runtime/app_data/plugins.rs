use super::unavailable;
use super::NoopAppDataSource;
use super::RuntimeCoreError;
use app_server_protocol::protocol::v2::{
    AppsInstalledParams, AppsInstalledResponse, AppsListParams, AppsListResponse, AppsReadParams,
    AppsReadResponse, PluginCatalogEnabledSetParams, PluginCatalogEnabledSetResponse,
    PluginCatalogInstallParams, PluginCatalogInstallResponse, PluginCatalogInstalledParams,
    PluginCatalogListParams, PluginCatalogListResponse, PluginCatalogReadParams,
    PluginCatalogReadResponse, PluginCatalogUninstallParams, PluginCatalogUninstallResponse,
    PluginSearchParams, PluginSearchResponse,
};
use async_trait::async_trait;
use std::path::PathBuf;

/// Snapshot of an enabled Agent Plugin used to build one current turn.
///
/// This is intentionally smaller than an activation payload: package discovery and
/// component loading stay owned by App Server, while RuntimeCore receives only the
/// stable inputs needed for Skills and MCP routing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginTurnSnapshot {
    pub id: String,
    pub config_name: String,
    pub display_name: String,
    pub package_root: PathBuf,
    pub skill_names: Vec<String>,
    pub mcp_server_names: Vec<String>,
}
#[async_trait]
pub trait PluginDataSource: Send + Sync {
    async fn list_plugin_catalog(
        &self,
        _params: PluginCatalogListParams,
    ) -> Result<PluginCatalogListResponse, RuntimeCoreError> {
        Err(unavailable("plugin/list"))
    }

    async fn search_plugins(
        &self,
        _params: PluginSearchParams,
    ) -> Result<PluginSearchResponse, RuntimeCoreError> {
        Err(unavailable("plugin/search"))
    }

    async fn read_plugin_catalog(
        &self,
        _params: PluginCatalogReadParams,
    ) -> Result<PluginCatalogReadResponse, RuntimeCoreError> {
        Err(unavailable("plugin/read"))
    }

    async fn install_plugin_catalog(
        &self,
        _params: PluginCatalogInstallParams,
    ) -> Result<PluginCatalogInstallResponse, RuntimeCoreError> {
        Err(unavailable("plugin/install"))
    }

    async fn uninstall_plugin_catalog(
        &self,
        _params: PluginCatalogUninstallParams,
    ) -> Result<PluginCatalogUninstallResponse, RuntimeCoreError> {
        Err(unavailable("plugin/uninstall"))
    }

    async fn list_plugin_catalog_installed(
        &self,
        _params: PluginCatalogInstalledParams,
    ) -> Result<PluginCatalogListResponse, RuntimeCoreError> {
        Err(unavailable("plugin/installed"))
    }

    async fn list_enabled_plugin_turn_snapshots(
        &self,
    ) -> Result<Vec<PluginTurnSnapshot>, RuntimeCoreError> {
        Ok(Vec::new())
    }

    async fn list_apps(
        &self,
        _params: AppsListParams,
    ) -> Result<AppsListResponse, RuntimeCoreError> {
        Err(unavailable("app/list"))
    }

    async fn read_apps(
        &self,
        _params: AppsReadParams,
    ) -> Result<AppsReadResponse, RuntimeCoreError> {
        Err(unavailable("app/read"))
    }

    async fn list_installed_apps(
        &self,
        _params: AppsInstalledParams,
    ) -> Result<AppsInstalledResponse, RuntimeCoreError> {
        Err(unavailable("app/installed"))
    }

    async fn set_plugin_catalog_enabled(
        &self,
        _params: PluginCatalogEnabledSetParams,
    ) -> Result<PluginCatalogEnabledSetResponse, RuntimeCoreError> {
        Err(unavailable("plugin/enabled/set"))
    }
}

impl PluginDataSource for NoopAppDataSource {}
