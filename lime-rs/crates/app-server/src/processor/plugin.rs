//! plugin domain handlers for the App Server processor.

use super::{dispatch_result, parse_params, to_jsonrpc_error, RequestProcessor, RpcDispatch};
use app_server_protocol::protocol::v2::{
    PluginCatalogEnabledSetParams, PluginCatalogInstallParams, PluginCatalogInstalledParams,
    PluginCatalogListParams, PluginCatalogReadParams, PluginCatalogUninstallParams,
    PluginSearchParams,
};
use app_server_protocol::JsonRpcError;

impl RequestProcessor {
    pub(super) async fn handle_plugin_catalog_list_v2_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: PluginCatalogListParams = parse_params(params)?;
        let response = self
            .runtime
            .list_plugin_catalog(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_plugin_search_v2_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: PluginSearchParams = parse_params(params)?;
        let response = self
            .runtime
            .search_plugins(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_plugin_catalog_read_v2_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: PluginCatalogReadParams = parse_params(params)?;
        let response = self
            .runtime
            .read_plugin_catalog(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_plugin_catalog_install_v2_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: PluginCatalogInstallParams = parse_params(params)?;
        let response = self
            .runtime
            .install_plugin_catalog(params)
            .await
            .map_err(to_jsonrpc_error)?;
        self.publish_app_list_updated().await;
        dispatch_result(response)
    }

    pub(super) async fn handle_plugin_catalog_uninstall_v2_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: PluginCatalogUninstallParams = parse_params(params)?;
        let response = self
            .runtime
            .uninstall_plugin_catalog(params)
            .await
            .map_err(to_jsonrpc_error)?;
        self.publish_app_list_updated().await;
        dispatch_result(response)
    }

    pub(super) async fn handle_plugin_catalog_installed_v2_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: PluginCatalogInstalledParams = parse_params(params)?;
        let response = self
            .runtime
            .list_plugin_catalog_installed(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_plugin_catalog_enabled_set_v2_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: PluginCatalogEnabledSetParams = parse_params(params)?;
        let response = self
            .runtime
            .set_plugin_catalog_enabled(params)
            .await
            .map_err(to_jsonrpc_error)?;
        self.publish_app_list_updated().await;
        dispatch_result(response)
    }
}
