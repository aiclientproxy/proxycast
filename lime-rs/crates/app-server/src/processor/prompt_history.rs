use super::{dispatch_result, parse_params, RequestProcessor, RpcDispatch};
use app_server_protocol::protocol::v2::{PromptHistoryAppendParams, PromptHistoryReadParams};
use app_server_protocol::{error_codes, JsonRpcError};
use serde_json::Value;

impl RequestProcessor {
    pub(super) async fn handle_prompt_history_read_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: PromptHistoryReadParams = parse_params(params)?;
        let store = self.prompt_history.clone();
        let response = tokio::task::spawn_blocking(move || {
            store.read(
                params.cursor.as_deref(),
                params.limit,
                params.log_id.as_deref(),
            )
        })
        .await
        .map_err(|error| history_error(format!("prompt history read task failed: {error}")))?
        .map_err(|error| history_error(format!("prompt history read failed: {error}")))?;
        dispatch_result(response)
    }

    pub(super) async fn handle_prompt_history_append_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: PromptHistoryAppendParams = parse_params(params)?;
        if params.session_id.trim().is_empty() {
            return Err(JsonRpcError::new(
                error_codes::INVALID_PARAMS,
                "promptHistory/append sessionId must not be empty",
            ));
        }
        if params.text.is_empty() {
            return Err(JsonRpcError::new(
                error_codes::INVALID_PARAMS,
                "promptHistory/append text must not be empty",
            ));
        }
        let store = self.prompt_history.clone();
        let response =
            tokio::task::spawn_blocking(move || store.append(&params.session_id, &params.text))
                .await
                .map_err(|error| {
                    history_error(format!("prompt history append task failed: {error}"))
                })?
                .map_err(|error| history_error(format!("prompt history append failed: {error}")))?;
        dispatch_result(response)
    }
}

fn history_error(message: String) -> JsonRpcError {
    JsonRpcError::new(error_codes::RUNTIME_ERROR, message)
}
