use super::super::*;
use app_server_protocol::protocol::v2::{
    AppsInstalledParams, AppsInstalledResponse, AppsListParams, AppsListResponse, AppsReadParams,
    AppsReadResponse, PluginCatalogEnabledSetParams, PluginCatalogEnabledSetResponse,
    PluginCatalogInstallParams, PluginCatalogInstallResponse, PluginCatalogInstalledParams,
    PluginCatalogListParams, PluginCatalogListResponse, PluginCatalogReadParams,
    PluginCatalogReadResponse, PluginCatalogUninstallParams, PluginCatalogUninstallResponse,
    PluginSearchParams, PluginSearchResponse,
};
use async_trait::async_trait;

#[async_trait]
impl PluginDataSource for LocalAppDataSource {
    async fn list_plugin_catalog(
        &self,
        params: PluginCatalogListParams,
    ) -> Result<PluginCatalogListResponse, RuntimeCoreError> {
        crate::local_data_source::plugin_catalog::list(&self.plugin_data_root, params)
            .map_err(data_error)
    }

    async fn search_plugins(
        &self,
        params: PluginSearchParams,
    ) -> Result<PluginSearchResponse, RuntimeCoreError> {
        crate::local_data_source::plugin_catalog::search(&self.plugin_data_root, params)
            .map_err(data_error)
    }

    async fn read_plugin_catalog(
        &self,
        params: PluginCatalogReadParams,
    ) -> Result<PluginCatalogReadResponse, RuntimeCoreError> {
        crate::local_data_source::plugin_catalog::read(&self.plugin_data_root, params)
            .map_err(data_error)
    }

    async fn install_plugin_catalog(
        &self,
        params: PluginCatalogInstallParams,
    ) -> Result<PluginCatalogInstallResponse, RuntimeCoreError> {
        crate::local_data_source::plugin_catalog::install(&self.plugin_data_root, params)
            .map_err(data_error)
    }

    async fn uninstall_plugin_catalog(
        &self,
        params: PluginCatalogUninstallParams,
    ) -> Result<PluginCatalogUninstallResponse, RuntimeCoreError> {
        crate::local_data_source::plugin_catalog::uninstall(&self.plugin_data_root, params)
            .map_err(data_error)
    }

    async fn list_plugin_catalog_installed(
        &self,
        params: PluginCatalogInstalledParams,
    ) -> Result<PluginCatalogListResponse, RuntimeCoreError> {
        crate::local_data_source::plugin_catalog::installed(&self.plugin_data_root, params)
            .map_err(data_error)
    }

    async fn list_enabled_plugin_turn_snapshots(
        &self,
    ) -> Result<Vec<crate::runtime::PluginTurnSnapshot>, RuntimeCoreError> {
        crate::local_data_source::plugin_catalog::enabled_plugin_turn_snapshots(
            &self.plugin_data_root,
        )
        .map_err(data_error)
    }

    async fn list_apps(
        &self,
        params: AppsListParams,
    ) -> Result<AppsListResponse, RuntimeCoreError> {
        crate::local_data_source::plugin_catalog::list_apps(&self.plugin_data_root, params)
            .map_err(data_error)
    }

    async fn read_apps(
        &self,
        params: AppsReadParams,
    ) -> Result<AppsReadResponse, RuntimeCoreError> {
        crate::local_data_source::plugin_catalog::read_apps(&self.plugin_data_root, params)
            .map_err(data_error)
    }

    async fn list_installed_apps(
        &self,
        params: AppsInstalledParams,
    ) -> Result<AppsInstalledResponse, RuntimeCoreError> {
        crate::local_data_source::plugin_catalog::installed_apps(&self.plugin_data_root, params)
            .map_err(data_error)
    }

    async fn set_plugin_catalog_enabled(
        &self,
        params: PluginCatalogEnabledSetParams,
    ) -> Result<PluginCatalogEnabledSetResponse, RuntimeCoreError> {
        crate::local_data_source::plugin_catalog::set_enabled(&self.plugin_data_root, params)
            .map_err(data_error)
    }
}
