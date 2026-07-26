//! agent_session domain handlers for the App Server processor.

use super::{
    dispatch_result, parse_params, project_event_notifications_jsonrpc, to_jsonrpc_error,
    v2_notifications::V2NotificationProjector, RequestProcessor, RpcDispatch,
};
use app_server_protocol::{
    AgentSessionFileCheckpointDiffParams, AgentSessionFileCheckpointGetParams,
    AgentSessionFileCheckpointListParams, AgentSessionFileCheckpointRestoreParams,
    AgentSessionMediaReadParams, AgentSessionToolInventoryReadParams, JsonRpcError, JsonRpcMessage,
    RequestId,
};

impl RequestProcessor {
    pub(super) async fn handle_session_media_read_impl(
        &self,
        request_id: &RequestId,
        params: Option<serde_json::Value>,
        event_callback: Option<&mut (dyn FnMut(JsonRpcMessage) + Send)>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: AgentSessionMediaReadParams = parse_params(params)?;
        let response = if params.stream {
            if let Some(event_callback) = event_callback {
                let mut event_projector = V2NotificationProjector::default();
                let mut runtime_event_callback = |event| {
                    let messages = project_event_notifications_jsonrpc(&mut event_projector, event)
                        .map_err(|error| {
                            crate::RuntimeCoreError::Backend(format!(
                                "failed to serialize media read streaming event: {}",
                                error.message
                            ))
                        })?;
                    for message in messages {
                        event_callback(message);
                    }
                    Ok(())
                };
                self.runtime
                    .read_agent_session_media_streaming_with_cancel(
                        params,
                        || self.is_request_canceled(request_id),
                        &mut runtime_event_callback,
                    )
                    .map_err(to_jsonrpc_error)?
            } else {
                self.runtime
                    .read_agent_session_media_with_cancel(params, || {
                        self.is_request_canceled(request_id)
                    })
                    .map_err(to_jsonrpc_error)?
            }
        } else {
            self.runtime
                .read_agent_session_media_with_cancel(params, || {
                    self.is_request_canceled(request_id)
                })
                .map_err(to_jsonrpc_error)?
        };
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
