//! plugin domain handlers for the App Server processor.

use super::{dispatch_result, parse_params, to_jsonrpc_error, RequestProcessor, RpcDispatch};
use app_server_protocol::protocol::v2::{
    PluginCatalogEnabledSetParams, PluginCatalogInstallParams, PluginCatalogInstalledParams,
    PluginCatalogListParams, PluginCatalogReadParams, PluginCatalogUninstallParams,
};
use app_server_protocol::{
    JsonRpcError, PluginFetchCloudPackageParams, PluginInstalledDisabledSetParams,
    PluginInstalledSaveParams, PluginLocalPackageExportParams, PluginLocalPackageInspectParams,
    PluginShellPrepareParams, PluginUiRuntimeStartParams, PluginUiRuntimeStatusParams,
    PluginUiRuntimeStopParams, PluginUninstallParams, PluginUninstallRehearsalParams,
};

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
        dispatch_result(response)
    }

    pub(super) async fn handle_plugin_installed_list_impl(
        &self,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let response = self
            .runtime
            .list_plugin_installed()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_plugin_local_package_inspect_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: PluginLocalPackageInspectParams = parse_params(params)?;
        let response = self
            .runtime
            .inspect_plugin_local_package(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_plugin_local_package_export_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: PluginLocalPackageExportParams = parse_params(params)?;
        let response = self
            .runtime
            .export_plugin_local_package(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_plugin_package_fetch_cloud_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: PluginFetchCloudPackageParams = parse_params(params)?;
        let response = self
            .runtime
            .fetch_plugin_cloud_package(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_plugin_installed_save_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: PluginInstalledSaveParams = parse_params(params)?;
        let response = self
            .runtime
            .save_plugin_installed(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_plugin_installed_disabled_set_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: PluginInstalledDisabledSetParams = parse_params(params)?;
        let response = self
            .runtime
            .set_plugin_installed_disabled(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_plugin_installed_uninstall_rehearsal_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: PluginUninstallRehearsalParams = parse_params(params)?;
        let response = self
            .runtime
            .preview_plugin_uninstall(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_plugin_installed_uninstall_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: PluginUninstallParams = parse_params(params)?;
        let response = self
            .runtime
            .uninstall_plugin(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_plugin_host_lifecycle_list_impl(
        &self,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let response = self
            .runtime
            .list_plugin_host_lifecycle()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_plugin_shell_prepare_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: PluginShellPrepareParams = parse_params(params)?;
        let response = self
            .runtime
            .prepare_plugin_shell(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_plugin_ui_runtime_start_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: PluginUiRuntimeStartParams = parse_params(params)?;
        let response = self
            .runtime
            .start_plugin_ui_runtime(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_plugin_ui_runtime_status_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: PluginUiRuntimeStatusParams = parse_params(params)?;
        let response = self
            .runtime
            .plugin_ui_runtime_status(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_plugin_ui_runtime_stop_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: PluginUiRuntimeStopParams = parse_params(params)?;
        let response = self
            .runtime
            .stop_plugin_ui_runtime(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }
}
