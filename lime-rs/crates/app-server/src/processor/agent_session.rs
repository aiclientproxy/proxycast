//! agent_session domain handlers for the App Server processor.

use super::{dispatch_result, parse_params, to_jsonrpc_error, RequestProcessor, RpcDispatch};
use app_server_protocol::protocol::v2::MediaReadParams;
use app_server_protocol::{
    AgentSessionFileCheckpointDiffParams, AgentSessionFileCheckpointGetParams,
    AgentSessionFileCheckpointListParams, AgentSessionFileCheckpointRestoreParams,
    AgentSessionToolInventoryReadParams, JsonRpcError, JsonRpcMessage, RequestId,
};

impl RequestProcessor {
    pub(super) async fn handle_media_read_v2_impl(
        &self,
        request_id: &RequestId,
        params: Option<serde_json::Value>,
        _event_callback: Option<&mut (dyn FnMut(JsonRpcMessage) + Send)>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: MediaReadParams = parse_params(params)?;
        let response = self
            .runtime
            .read_media_with_cancel(params, || self.is_request_canceled(request_id))
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_file_checkpoint_list_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: AgentSessionFileCheckpointListParams = parse_params(params)?;
        let response = self
            .runtime
            .list_agent_session_file_checkpoints(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_file_checkpoint_get_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: AgentSessionFileCheckpointGetParams = parse_params(params)?;
        let response = self
            .runtime
            .get_agent_session_file_checkpoint(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_file_checkpoint_diff_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: AgentSessionFileCheckpointDiffParams = parse_params(params)?;
        let response = self
            .runtime
            .diff_agent_session_file_checkpoint(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_file_checkpoint_restore_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: AgentSessionFileCheckpointRestoreParams = parse_params(params)?;
        let response = self
            .runtime
            .restore_agent_session_file_checkpoint(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_tool_inventory_read_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: AgentSessionToolInventoryReadParams = parse_params(params)?;
        let response = self
            .runtime
            .read_agent_session_tool_inventory(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }
}
