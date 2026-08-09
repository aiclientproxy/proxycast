use super::{dispatch_result, parse_params, to_jsonrpc_error, RequestProcessor, RpcDispatch};
use app_server_protocol::protocol::v2::{ReviewStartParams, ReviewStartResponse};
use serde_json::Value;

impl RequestProcessor {
    pub(super) async fn handle_review_start_v2_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, app_server_protocol::JsonRpcError> {
        self.ensure_initialized()?;
        let params: ReviewStartParams = parse_params(params)?;
        self.ensure_direct_input_allowed(&params.thread_id).await?;
        let output = self
            .runtime
            .start_review(&params.thread_id, params.target, params.delivery)
            .await
            .map_err(to_jsonrpc_error)?;
        let response = ReviewStartResponse {
            turn: super::turn::v2_turn_from_agent_turn(output.response.turn),
            review_thread_id: params.thread_id,
        };
        dispatch_result(response)
    }
}
