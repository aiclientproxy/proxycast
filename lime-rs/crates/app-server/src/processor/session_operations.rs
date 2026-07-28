use super::*;
use app_server_protocol::protocol::v2::{
    ServerNotification as V2ServerNotification, ThreadMemoryModeSetParams,
    ThreadSettingsUpdateParams, ThreadSettingsUpdateResponse, ThreadSettingsUpdatedNotification,
    ThreadShellCommandParams,
};

impl RequestProcessor {
    pub(super) async fn handle_thread_settings_update_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ThreadSettingsUpdateParams = parse_params(params)?;
        let thread_id = params.thread_id.clone();
        self.ensure_direct_input_allowed(&thread_id).await?;
        let thread_settings = self
            .runtime
            .update_thread_settings(params)
            .await
            .map_err(to_jsonrpc_error)?;
        let notification: JsonRpcNotification =
            V2ServerNotification::ThreadSettingsUpdated(ThreadSettingsUpdatedNotification {
                thread_id,
                thread_settings,
            })
            .into();
        dispatch_result(ThreadSettingsUpdateResponse {})
            .map(|dispatch| dispatch.with_notification(notification))
    }

    pub(super) async fn handle_thread_memory_mode_set_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ThreadMemoryModeSetParams = parse_params(params)?;
        self.ensure_direct_input_allowed(&params.thread_id).await?;
        let response = self
            .runtime
            .set_thread_memory_mode(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_thread_shell_command_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ThreadShellCommandParams = parse_params(params)?;
        self.ensure_direct_input_allowed(&params.thread_id).await?;
        let response = self
            .runtime
            .run_thread_shell_command(params)
            .await
            .map_err(|error| match error {
                crate::RuntimeCoreError::InvalidRequest(message) => {
                    JsonRpcError::new(error_codes::INVALID_PARAMS, message)
                }
                other => to_jsonrpc_error(other),
            })?;
        dispatch_result(response)
    }
}
