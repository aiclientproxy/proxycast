use super::{dispatch_result, parse_params, to_jsonrpc_error, RequestProcessor, RpcDispatch};
use app_server_protocol::protocol::v2::HooksListParams;
use app_server_protocol::JsonRpcError;

impl RequestProcessor {
    pub(super) async fn handle_hooks_list_v2_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: HooksListParams = parse_params(params)?;
        let response = self
            .runtime
            .list_hooks(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }
}
