//! model domain handlers for the App Server processor.

use super::{dispatch_result, parse_params, to_jsonrpc_error, RequestProcessor, RpcDispatch};
use app_server_protocol::protocol::v2::{
    ModelProviderCapabilitiesReadParams, ModelProviderCapabilitiesReadResponse,
};
use app_server_protocol::{
    JsonRpcError, ModelListParams, ModelProviderAliasReadParams, ModelProviderConfigExportParams,
    ModelProviderConfigImportParams, ModelProviderCreateParams, ModelProviderDeleteParams,
    ModelProviderFetchModelsParams, ModelProviderKeyCreateParams, ModelProviderKeyDeleteParams,
    ModelProviderKeyUpdateParams, ModelProviderReadParams, ModelProviderSortOrdersUpdateParams,
    ModelProviderTestChatParams, ModelProviderTestConnectionParams, ModelProviderUiStateReadParams,
    ModelProviderUiStateWriteParams, ModelProviderUpdateParams,
};

impl RequestProcessor {
    pub(super) async fn handle_model_list_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelListParams = parse_params(params)?;
        let response = self
            .runtime
            .list_models(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_model_preferences_list_impl(
        &self,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let response = self
            .runtime
            .list_model_preferences()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_model_sync_state_read_impl(
        &self,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let response = self
            .runtime
            .read_model_sync_state()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_list_impl(
        &self,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let response = self
            .runtime
            .list_model_providers()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_capabilities_read_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let _: ModelProviderCapabilitiesReadParams = parse_params(params)?;
        let response: ModelProviderCapabilitiesReadResponse = self
            .runtime
            .read_model_provider_capabilities()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_catalog_list_impl(
        &self,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let response = self
            .runtime
            .list_model_provider_catalog()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_read_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderReadParams = parse_params(params)?;
        let response = self
            .runtime
            .read_model_provider(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_create_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderCreateParams = parse_params(params)?;
        let response = self
            .runtime
            .create_model_provider(params)
            .await
            .map_err(to_jsonrpc_error)?;
        self.runtime
            .schedule_pending_route_recovery(self.runtime_host_context());
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_update_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderUpdateParams = parse_params(params)?;
        let response = self
            .runtime
            .update_model_provider(params)
            .await
            .map_err(to_jsonrpc_error)?;
        self.runtime
            .schedule_pending_route_recovery(self.runtime_host_context());
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_delete_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderDeleteParams = parse_params(params)?;
        let response = self
            .runtime
            .delete_model_provider(params)
            .await
            .map_err(to_jsonrpc_error)?;
        self.runtime
            .schedule_pending_route_recovery(self.runtime_host_context());
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_sort_orders_update_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderSortOrdersUpdateParams = parse_params(params)?;
        let response = self
            .runtime
            .update_model_provider_sort_orders(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_config_export_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderConfigExportParams = parse_params(params)?;
        let response = self
            .runtime
            .export_model_provider_config(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_config_import_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderConfigImportParams = parse_params(params)?;
        let response = self
            .runtime
            .import_model_provider_config(params)
            .await
            .map_err(to_jsonrpc_error)?;
        self.runtime
            .schedule_pending_route_recovery(self.runtime_host_context());
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_test_connection_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderTestConnectionParams = parse_params(params)?;
        let response = self
            .runtime
            .test_model_provider_connection(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_test_chat_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderTestChatParams = parse_params(params)?;
        let response = self
            .runtime
            .test_model_provider_chat(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_fetch_models_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderFetchModelsParams = parse_params(params)?;
        let response = self
            .runtime
            .fetch_model_provider_models(params)
            .await
            .map_err(to_jsonrpc_error)?;
        self.runtime
            .schedule_pending_route_recovery(self.runtime_host_context());
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_key_create_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderKeyCreateParams = parse_params(params)?;
        let response = self
            .runtime
            .create_model_provider_key(params)
            .await
            .map_err(to_jsonrpc_error)?;
        self.runtime
            .schedule_pending_route_recovery(self.runtime_host_context());
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_key_update_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderKeyUpdateParams = parse_params(params)?;
        let response = self
            .runtime
            .update_model_provider_key(params)
            .await
            .map_err(to_jsonrpc_error)?;
        self.runtime
            .schedule_pending_route_recovery(self.runtime_host_context());
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_key_delete_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderKeyDeleteParams = parse_params(params)?;
        let response = self
            .runtime
            .delete_model_provider_key(params)
            .await
            .map_err(to_jsonrpc_error)?;
        self.runtime
            .schedule_pending_route_recovery(self.runtime_host_context());
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_ui_state_read_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderUiStateReadParams = parse_params(params)?;
        let response = self
            .runtime
            .read_model_provider_ui_state(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_ui_state_write_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderUiStateWriteParams = parse_params(params)?;
        let response = self
            .runtime
            .write_model_provider_ui_state(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_alias_read_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ModelProviderAliasReadParams = parse_params(params)?;
        let response = self
            .runtime
            .read_model_provider_alias(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_model_provider_alias_list_impl(
        &self,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let response = self
            .runtime
            .list_model_provider_aliases()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }
}
