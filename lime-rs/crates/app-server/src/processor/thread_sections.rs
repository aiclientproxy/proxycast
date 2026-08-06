use super::{dispatch_result, parse_params, to_jsonrpc_error, RequestProcessor, RpcDispatch};
use app_server_protocol::protocol::v2::{
    ThreadSectionCreateParams, ThreadSectionDeleteParams, ThreadSectionListParams,
    ThreadSectionMoveParams, ThreadSectionUpdateParams,
};
use app_server_protocol::JsonRpcError;

impl RequestProcessor {
    pub(super) async fn handle_section_list(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ThreadSectionListParams = parse_params(params)?;
        let response = self
            .runtime
            .list_thread_sections(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_section_create(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ThreadSectionCreateParams = parse_params(params)?;
        let response = self
            .runtime
            .create_thread_section(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_section_update(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ThreadSectionUpdateParams = parse_params(params)?;
        let response = self
            .runtime
            .update_thread_section(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_section_delete(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ThreadSectionDeleteParams = parse_params(params)?;
        let response = self
            .runtime
            .delete_thread_section(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_section_move(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ThreadSectionMoveParams = parse_params(params)?;
        let response = self
            .runtime
            .move_thread_to_section(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }
}
