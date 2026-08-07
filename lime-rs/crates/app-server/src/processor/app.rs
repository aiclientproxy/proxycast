use super::{dispatch_result, parse_params, to_jsonrpc_error, RequestProcessor, RpcDispatch};
use app_server_protocol::error_codes;
use app_server_protocol::protocol::v2::{
    AppListUpdatedNotification, AppsInstalledParams, AppsListParams, AppsReadParams,
    ServerNotification,
};
use app_server_protocol::JsonRpcError;

impl RequestProcessor {
    pub(super) async fn handle_app_list_v2_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: AppsListParams = parse_params(params)?;
        let publish_update = params.cursor.is_none();
        let response = self
            .runtime
            .list_apps(params)
            .await
            .map_err(to_jsonrpc_error)?;
        if publish_update {
            self.publish_app_list_updated().await;
        }
        dispatch_result(response)
    }

    pub(super) async fn handle_app_read_v2_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: AppsReadParams = parse_params(params)?;
        if params.app_ids.len() > 100 {
            return Err(JsonRpcError::new(
                error_codes::INVALID_PARAMS,
                "app/read appIds 最多允许 100 项。",
            ));
        }
        let response = self
            .runtime
            .read_apps(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_app_installed_v2_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: AppsInstalledParams = parse_params(params)?;
        let response = self
            .runtime
            .list_installed_apps(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn publish_app_list_updated(&self) {
        let response = match self
            .runtime
            .list_apps(AppsListParams {
                limit: Some(100),
                ..AppsListParams::default()
            })
            .await
        {
            Ok(response) => response,
            Err(error) => {
                tracing::warn!(error = %error, "app registry changed but its list could not be read");
                return;
            }
        };
        self.publish_server_notification(ServerNotification::AppListUpdated(
            AppListUpdatedNotification {
                data: response.data,
            },
        ))
        .await;
    }
}
