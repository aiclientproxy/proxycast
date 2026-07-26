use super::super::{
    dispatch_result, parse_params, to_jsonrpc_error, RequestProcessor, RpcDispatch,
};
use app_server_protocol::protocol::v2::ThreadInjectItemsParams;
use app_server_protocol::JsonRpcError;
use uuid::Uuid;

impl RequestProcessor {
    pub(in crate::processor) async fn handle_thread_inject_items_v2(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let mut params: ThreadInjectItemsParams = parse_params(params)?;
        params.thread_id = params.thread_id.trim().to_string();
        if params.thread_id.is_empty() {
            return Err(super::invalid_request(
                "thread/inject_items requires threadId",
            ));
        }
        Uuid::parse_str(&params.thread_id)
            .map_err(|error| super::invalid_request(format!("invalid thread id: {error}")))?;
        let response =
            self.runtime
                .inject_thread_items(params)
                .await
                .map_err(|error| match error {
                    crate::RuntimeCoreError::InvalidRequest(message) => {
                        super::invalid_request(message)
                    }
                    other => to_jsonrpc_error(other),
                })?;
        dispatch_result(response)
    }
}
