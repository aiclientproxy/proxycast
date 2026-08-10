use super::RuntimeCore;
use super::RuntimeCoreError;
use app_server_protocol::protocol::v2::{
    AppsInstalledParams, AppsInstalledResponse, AppsListParams, AppsListResponse, AppsReadParams,
    AppsReadResponse, PluginCatalogEnabledSetParams, PluginCatalogEnabledSetResponse,
    PluginCatalogInstallParams, PluginCatalogInstallResponse, PluginCatalogInstalledParams,
    PluginCatalogListParams, PluginCatalogListResponse, PluginCatalogReadParams,
    PluginCatalogReadResponse, PluginCatalogUninstallParams, PluginCatalogUninstallResponse,
    PluginSearchParams, PluginSearchResponse,
};

impl RuntimeCore {
    pub async fn list_apps(
        &self,
        params: AppsListParams,
    ) -> Result<AppsListResponse, RuntimeCoreError> {
        self.ensure_optional_apps_thread_loaded(params.thread_id.as_deref())?;
        self.app_data_source.list_apps(params).await
    }

    pub async fn read_apps(
        &self,
        params: AppsReadParams,
    ) -> Result<AppsReadResponse, RuntimeCoreError> {
        self.app_data_source.read_apps(params).await
    }

    pub async fn list_installed_apps(
        &self,
        params: AppsInstalledParams,
    ) -> Result<AppsInstalledResponse, RuntimeCoreError> {
        self.ensure_optional_apps_thread_loaded(params.thread_id.as_deref())?;
        self.app_data_source.list_installed_apps(params).await
    }

    fn ensure_optional_apps_thread_loaded(
        &self,
        thread_id: Option<&str>,
    ) -> Result<(), RuntimeCoreError> {
        let Some(thread_id) = thread_id else {
            return Ok(());
        };
        let thread_id = thread_id.trim();
        if thread_id.is_empty() || self.loaded_session_id_for_thread(thread_id).is_none() {
            return Err(RuntimeCoreError::SessionNotFound(thread_id.to_string()));
        }
        Ok(())
    }

    pub async fn list_plugin_catalog(
        &self,
        params: PluginCatalogListParams,
    ) -> Result<PluginCatalogListResponse, RuntimeCoreError> {
        self.app_data_source.list_plugin_catalog(params).await
    }

    pub async fn search_plugins(
        &self,
        params: PluginSearchParams,
    ) -> Result<PluginSearchResponse, RuntimeCoreError> {
        self.app_data_source.search_plugins(params).await
    }

    pub async fn read_plugin_catalog(
        &self,
        params: PluginCatalogReadParams,
    ) -> Result<PluginCatalogReadResponse, RuntimeCoreError> {
        self.app_data_source.read_plugin_catalog(params).await
    }

    pub async fn install_plugin_catalog(
        &self,
        params: PluginCatalogInstallParams,
    ) -> Result<PluginCatalogInstallResponse, RuntimeCoreError> {
        self.backend.invalidate_mcp_runtimes().await;
        self.app_data_source.install_plugin_catalog(params).await
    }

    pub async fn uninstall_plugin_catalog(
        &self,
        params: PluginCatalogUninstallParams,
    ) -> Result<PluginCatalogUninstallResponse, RuntimeCoreError> {
        self.backend.invalidate_mcp_runtimes().await;
        self.app_data_source.uninstall_plugin_catalog(params).await
    }

    pub async fn list_plugin_catalog_installed(
        &self,
        params: PluginCatalogInstalledParams,
    ) -> Result<PluginCatalogListResponse, RuntimeCoreError> {
        self.app_data_source
            .list_plugin_catalog_installed(params)
            .await
    }

    pub async fn set_plugin_catalog_enabled(
        &self,
        params: PluginCatalogEnabledSetParams,
    ) -> Result<PluginCatalogEnabledSetResponse, RuntimeCoreError> {
        let response = self
            .app_data_source
            .set_plugin_catalog_enabled(params)
            .await?;
        self.backend.invalidate_mcp_runtimes().await;
        Ok(response)
    }
}
