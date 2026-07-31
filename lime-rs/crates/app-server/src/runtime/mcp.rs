use super::{RuntimeCore, RuntimeCoreError};
use app_server_protocol::*;

impl RuntimeCore {
    pub async fn list_mcp_servers(&self) -> Result<McpServerListResponse, RuntimeCoreError> {
        self.app_data_source.list_mcp_servers().await
    }

    pub async fn list_mcp_servers_with_status(
        &self,
    ) -> Result<McpServerStatusListResponse, RuntimeCoreError> {
        self.app_data_source.list_mcp_servers_with_status().await
    }

    pub async fn create_mcp_server(
        &self,
        params: McpServerCreateParams,
    ) -> Result<McpServerListResponse, RuntimeCoreError> {
        self.app_data_source.create_mcp_server(params).await
    }

    pub async fn update_mcp_server(
        &self,
        params: McpServerUpdateParams,
    ) -> Result<McpServerListResponse, RuntimeCoreError> {
        self.app_data_source.update_mcp_server(params).await
    }

    pub async fn delete_mcp_server(
        &self,
        params: McpServerDeleteParams,
    ) -> Result<McpServerListResponse, RuntimeCoreError> {
        self.app_data_source.delete_mcp_server(params).await
    }

    pub async fn set_mcp_server_enabled(
        &self,
        params: McpServerEnabledSetParams,
    ) -> Result<McpServerListResponse, RuntimeCoreError> {
        self.app_data_source.set_mcp_server_enabled(params).await
    }

    pub async fn import_mcp_servers_from_app(
        &self,
        params: McpServerImportFromAppParams,
    ) -> Result<McpServerImportFromAppResponse, RuntimeCoreError> {
        self.app_data_source
            .import_mcp_servers_from_app(params)
            .await
    }

    pub async fn sync_all_mcp_servers_to_live(
        &self,
    ) -> Result<McpServerListResponse, RuntimeCoreError> {
        self.app_data_source.sync_all_mcp_servers_to_live().await
    }

    pub async fn start_mcp_server(
        &self,
        params: McpServerStartParams,
    ) -> Result<McpServerLifecycleResponse, RuntimeCoreError> {
        self.app_data_source.start_mcp_server(params).await
    }

    pub async fn stop_mcp_server(
        &self,
        params: McpServerStopParams,
    ) -> Result<McpServerLifecycleResponse, RuntimeCoreError> {
        self.app_data_source.stop_mcp_server(params).await
    }

    pub async fn login_mcp_server_oauth(
        &self,
        params: McpServerOauthLoginParams,
    ) -> Result<lime_mcp::McpOAuthLoginHandle, RuntimeCoreError> {
        self.app_data_source.login_mcp_server_oauth(params).await
    }

    pub async fn list_mcp_tools(&self) -> Result<McpToolListResponse, RuntimeCoreError> {
        self.app_data_source.list_mcp_tools().await
    }

    pub async fn list_mcp_tools_for_context(
        &self,
        params: McpToolListForContextParams,
    ) -> Result<McpToolListResponse, RuntimeCoreError> {
        self.app_data_source
            .list_mcp_tools_for_context(params)
            .await
    }

    pub async fn search_mcp_tools(
        &self,
        params: McpToolSearchParams,
    ) -> Result<McpToolListResponse, RuntimeCoreError> {
        self.app_data_source.search_mcp_tools(params).await
    }

    pub async fn call_mcp_tool(
        &self,
        params: McpToolCallParams,
    ) -> Result<McpToolCallResponse, RuntimeCoreError> {
        self.app_data_source.call_mcp_tool(params).await
    }

    pub async fn call_mcp_tool_with_caller(
        &self,
        params: McpToolCallWithCallerParams,
    ) -> Result<McpToolCallResponse, RuntimeCoreError> {
        self.app_data_source.call_mcp_tool_with_caller(params).await
    }

    pub async fn list_mcp_prompts(&self) -> Result<McpPromptListResponse, RuntimeCoreError> {
        self.app_data_source.list_mcp_prompts().await
    }

    pub async fn get_mcp_prompt(
        &self,
        params: McpPromptGetParams,
    ) -> Result<McpPromptGetResponse, RuntimeCoreError> {
        self.app_data_source.get_mcp_prompt(params).await
    }

    pub async fn list_mcp_resources(&self) -> Result<McpResourceListResponse, RuntimeCoreError> {
        self.app_data_source.list_mcp_resources().await
    }

    pub async fn read_mcp_resource(
        &self,
        params: McpResourceReadParams,
    ) -> Result<McpResourceReadResponse, RuntimeCoreError> {
        self.app_data_source.read_mcp_resource(params).await
    }

    pub async fn subscribe_mcp_resource(
        &self,
        params: McpResourceSubscribeParams,
    ) -> Result<McpResourceSubscriptionResponse, RuntimeCoreError> {
        self.app_data_source.subscribe_mcp_resource(params).await
    }

    pub async fn unsubscribe_mcp_resource(
        &self,
        params: McpResourceUnsubscribeParams,
    ) -> Result<McpResourceSubscriptionResponse, RuntimeCoreError> {
        self.app_data_source.unsubscribe_mcp_resource(params).await
    }
}
