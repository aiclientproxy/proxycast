use super::super::*;
use app_server_protocol::protocol::v2::{
    PluginCatalogEnabledSetParams, PluginCatalogEnabledSetResponse, PluginCatalogInstallParams,
    PluginCatalogInstallResponse, PluginCatalogInstalledParams, PluginCatalogListParams,
    PluginCatalogListResponse, PluginCatalogReadParams, PluginCatalogReadResponse,
    PluginCatalogUninstallParams, PluginCatalogUninstallResponse, PluginSearchParams,
    PluginSearchResponse,
};
use async_trait::async_trait;

#[async_trait]
impl PluginDataSource for LocalAppDataSource {
    fn plugin_data_root(&self) -> Result<std::path::PathBuf, RuntimeCoreError> {
        Ok(self.plugin_data_root.clone())
    }

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

    async fn list_plugin_catalog_activations(&self) -> Result<Vec<Value>, RuntimeCoreError> {
        crate::local_data_source::plugin_catalog::enabled_activation_descriptors(
            &self.plugin_data_root,
        )
        .map_err(data_error)
    }

    async fn set_plugin_catalog_enabled(
        &self,
        params: PluginCatalogEnabledSetParams,
    ) -> Result<PluginCatalogEnabledSetResponse, RuntimeCoreError> {
        crate::local_data_source::plugin_catalog::set_enabled(&self.plugin_data_root, params)
            .map_err(data_error)
    }

    async fn list_plugin_installed(&self) -> Result<PluginInstalledListResponse, RuntimeCoreError> {
        plugins::list_plugin_installed_state(&self.plugin_data_root).map_err(data_error)
    }

    async fn inspect_plugin_local_package(
        &self,
        params: PluginLocalPackageInspectParams,
    ) -> Result<PluginLocalPackageInspectResponse, RuntimeCoreError> {
        plugins::inspect_plugin_local_package(params).map_err(data_error)
    }

    async fn export_plugin_local_package(
        &self,
        params: PluginLocalPackageExportParams,
    ) -> Result<PluginLocalPackageExportResponse, RuntimeCoreError> {
        plugins::export_plugin_local_package(params).map_err(data_error)
    }

    async fn fetch_plugin_cloud_package(
        &self,
        params: PluginFetchCloudPackageParams,
    ) -> Result<PluginPackageCacheEntry, RuntimeCoreError> {
        plugins::fetch_plugin_cloud_package(&self.plugin_data_root, params)
            .await
            .map_err(data_error)
    }

    async fn save_plugin_installed(
        &self,
        params: PluginInstalledSaveParams,
    ) -> Result<Value, RuntimeCoreError> {
        plugins::save_plugin_installed_state(&self.plugin_data_root, params).map_err(data_error)
    }

    async fn set_plugin_installed_disabled(
        &self,
        params: PluginInstalledDisabledSetParams,
    ) -> Result<PluginInstalledListResponse, RuntimeCoreError> {
        plugins::set_plugin_installed_disabled(&self.plugin_data_root, params).map_err(data_error)
    }

    async fn preview_plugin_uninstall(
        &self,
        params: PluginUninstallRehearsalParams,
    ) -> Result<PluginUninstallRehearsalResponse, RuntimeCoreError> {
        plugins::build_plugin_uninstall_rehearsal(
            &self.plugin_data_root,
            params.app_id,
            params.mode,
        )
        .map_err(data_error)
    }

    async fn uninstall_plugin(
        &self,
        params: PluginUninstallParams,
    ) -> Result<PluginUninstallResponse, RuntimeCoreError> {
        plugins::uninstall_plugin(&self.plugin_data_root, params).map_err(data_error)
    }
}
