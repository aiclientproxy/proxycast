//! mcp domain handlers for the App Server processor.

use super::{dispatch_result, parse_params, to_jsonrpc_error, RequestProcessor, RpcDispatch};
use app_server_protocol::protocol::v2::{
    McpServerOauthLoginCompletedNotification, ServerNotification,
};
use app_server_protocol::{
    JsonRpcError, McpPromptGetParams, McpResourceReadParams, McpResourceSubscribeParams,
    McpResourceUnsubscribeParams, McpServerCreateParams, McpServerDeleteParams,
    McpServerEnabledSetParams, McpServerImportFromAppParams, McpServerOauthLoginParams,
    McpServerOauthLoginResponse, McpServerStartParams, McpServerStopParams, McpServerUpdateParams,
    McpToolCallParams, McpToolCallWithCallerParams, McpToolListForContextParams,
    McpToolSearchParams,
};

impl RequestProcessor {
    pub(super) async fn handle_mcp_server_list_impl(&self) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let response = self
            .runtime
            .list_mcp_servers()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_server_status_list_impl(
        &self,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let response = self
            .runtime
            .list_mcp_servers_with_status()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_server_create_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpServerCreateParams = parse_params(params)?;
        let response = self
            .runtime
            .create_mcp_server(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_server_update_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpServerUpdateParams = parse_params(params)?;
        let response = self
            .runtime
            .update_mcp_server(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_server_delete_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpServerDeleteParams = parse_params(params)?;
        let response = self
            .runtime
            .delete_mcp_server(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_server_enabled_set_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpServerEnabledSetParams = parse_params(params)?;
        let response = self
            .runtime
            .set_mcp_server_enabled(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_server_import_from_app_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpServerImportFromAppParams = parse_params(params)?;
        let response = self
            .runtime
            .import_mcp_servers_from_app(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_server_sync_all_to_live_impl(
        &self,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let response = self
            .runtime
            .sync_all_mcp_servers_to_live()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_server_start_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpServerStartParams = parse_params(params)?;
        let response = self
            .runtime
            .start_mcp_server(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_server_stop_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpServerStopParams = parse_params(params)?;
        let response = self
            .runtime
            .stop_mcp_server(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_server_oauth_login_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpServerOauthLoginParams = parse_params(params)?;
        let server_name = params.name.clone();
        let handle = self
            .runtime
            .login_mcp_server_oauth(params)
            .await
            .map_err(to_jsonrpc_error)?;
        let response = McpServerOauthLoginResponse {
            authorization_url: handle.authorization_url.clone(),
            state: handle.state.clone(),
        };
        let processor = self.clone();
        tokio::spawn(async move {
            let (success, error) = match handle.wait().await {
                Ok(()) => (true, None),
                Err(error) => (false, Some(error.to_string())),
            };
            processor
                .publish_server_notification(ServerNotification::McpServerOauthLoginCompleted(
                    McpServerOauthLoginCompletedNotification {
                        name: server_name,
                        thread_id: None,
                        success,
                        error,
                    },
                ))
                .await;
        });
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_tool_list_impl(&self) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let response = self
            .runtime
            .list_mcp_tools()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_tool_list_for_context_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpToolListForContextParams = parse_params(params)?;
        let response = self
            .runtime
            .list_mcp_tools_for_context(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_tool_search_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpToolSearchParams = parse_params(params)?;
        let response = self
            .runtime
            .search_mcp_tools(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_tool_call_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpToolCallParams = parse_params(params)?;
        let response = self
            .runtime
            .call_mcp_tool(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_tool_call_with_caller_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpToolCallWithCallerParams = parse_params(params)?;
        let response = self
            .runtime
            .call_mcp_tool_with_caller(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_prompt_list_impl(&self) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let response = self
            .runtime
            .list_mcp_prompts()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_prompt_get_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpPromptGetParams = parse_params(params)?;
        let response = self
            .runtime
            .get_mcp_prompt(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_resource_list_impl(&self) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let response = self
            .runtime
            .list_mcp_resources()
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_resource_read_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpResourceReadParams = parse_params(params)?;
        let response = self
            .runtime
            .read_mcp_resource(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_resource_subscribe_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpResourceSubscribeParams = parse_params(params)?;
        let response = self
            .runtime
            .subscribe_mcp_resource(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_mcp_resource_unsubscribe_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: McpResourceUnsubscribeParams = parse_params(params)?;
        let response = self
            .runtime
            .unsubscribe_mcp_resource(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }
}
