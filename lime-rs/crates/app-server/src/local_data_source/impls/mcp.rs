use super::super::*;
use app_server_protocol::McpServerOauthLoginParams;
use async_trait::async_trait;

#[async_trait]
impl McpAppDataSource for LocalAppDataSource {
    fn mcp_elicitation_router(&self) -> lime_mcp::ElicitationRequestRouter {
        self.mcp_elicitation_router.clone()
    }

    async fn list_mcp_servers(&self) -> Result<McpServerListResponse, RuntimeCoreError> {
        mcp::list_mcp_servers(&self.db)
    }

    async fn list_mcp_servers_with_status_v2(
        &self,
        params: app_server_protocol::protocol::v2::ListMcpServerStatusParams,
    ) -> Result<app_server_protocol::protocol::v2::ListMcpServerStatusResponse, RuntimeCoreError>
    {
        mcp::list_mcp_servers_with_status_v2(&self.db, &self.mcp_manager, params).await
    }

    async fn create_mcp_server(
        &self,
        params: McpServerCreateParams,
    ) -> Result<McpServerListResponse, RuntimeCoreError> {
        mcp::create_mcp_server(&self.db, params)
    }

    async fn update_mcp_server(
        &self,
        params: McpServerUpdateParams,
    ) -> Result<McpServerListResponse, RuntimeCoreError> {
        mcp::update_mcp_server(&self.db, params)
    }

    async fn delete_mcp_server(
        &self,
        params: McpServerDeleteParams,
    ) -> Result<McpServerListResponse, RuntimeCoreError> {
        mcp::delete_mcp_server(&self.db, params)
    }

    async fn set_mcp_server_enabled(
        &self,
        params: McpServerEnabledSetParams,
    ) -> Result<McpServerListResponse, RuntimeCoreError> {
        mcp::set_mcp_server_enabled(&self.db, params)
    }

    async fn import_mcp_servers_from_app(
        &self,
        params: McpServerImportFromAppParams,
    ) -> Result<McpServerImportFromAppResponse, RuntimeCoreError> {
        mcp::import_mcp_servers_from_app(&self.db, params)
    }

    async fn sync_all_mcp_servers_to_live(
        &self,
    ) -> Result<McpServerListResponse, RuntimeCoreError> {
        mcp::sync_all_mcp_servers_to_live(&self.db)
    }

    async fn start_mcp_server(
        &self,
        params: McpServerStartParams,
    ) -> Result<McpServerLifecycleResponse, RuntimeCoreError> {
        mcp::start_mcp_server(&self.db, &self.mcp_manager, params).await
    }

    async fn stop_mcp_server(
        &self,
        params: McpServerStopParams,
    ) -> Result<McpServerLifecycleResponse, RuntimeCoreError> {
        mcp::stop_mcp_server(&self.mcp_manager, params).await
    }

    async fn login_mcp_server_oauth(
        &self,
        params: McpServerOauthLoginParams,
    ) -> Result<lime_mcp::McpOAuthLoginHandle, RuntimeCoreError> {
        mcp::login_mcp_server_oauth(&self.db, &self.mcp_manager, params).await
    }

    async fn list_mcp_tools(&self) -> Result<McpToolListResponse, RuntimeCoreError> {
        mcp::list_mcp_tools(&self.mcp_manager).await
    }

    async fn list_mcp_runtime_server_specs(
        &self,
    ) -> Result<Vec<lime_mcp::McpRuntimeServerSpec>, RuntimeCoreError> {
        mcp::list_mcp_runtime_server_specs(&self.db, &self.plugin_data_root)
    }

    async fn list_mcp_tools_for_context(
        &self,
        params: McpToolListForContextParams,
    ) -> Result<McpToolListResponse, RuntimeCoreError> {
        mcp::list_mcp_tools_for_context(&self.mcp_manager, params).await
    }

    async fn search_mcp_tools(
        &self,
        params: McpToolSearchParams,
    ) -> Result<McpToolListResponse, RuntimeCoreError> {
        mcp::search_mcp_tools(&self.mcp_manager, params).await
    }

    async fn list_mcp_prompts(&self) -> Result<McpPromptListResponse, RuntimeCoreError> {
        mcp::list_mcp_prompts(&self.mcp_manager).await
    }

    async fn get_mcp_prompt(
        &self,
        params: McpPromptGetParams,
    ) -> Result<McpPromptGetResponse, RuntimeCoreError> {
        mcp::get_mcp_prompt(&self.mcp_manager, params).await
    }

    async fn list_mcp_resources(&self) -> Result<McpResourceListResponse, RuntimeCoreError> {
        mcp::list_mcp_resources(&self.mcp_manager).await
    }

    async fn read_mcp_server_resource(
        &self,
        params: McpServerResourceReadParams,
    ) -> Result<McpServerResourceReadResponse, RuntimeCoreError> {
        mcp::read_mcp_server_resource(&self.mcp_manager, params).await
    }

    async fn subscribe_mcp_resource(
        &self,
        params: McpResourceSubscribeParams,
    ) -> Result<McpResourceSubscriptionResponse, RuntimeCoreError> {
        mcp::subscribe_mcp_resource(&self.mcp_manager, params).await
    }

    async fn unsubscribe_mcp_resource(
        &self,
        params: McpResourceUnsubscribeParams,
    ) -> Result<McpResourceSubscriptionResponse, RuntimeCoreError> {
        mcp::unsubscribe_mcp_resource(&self.mcp_manager, params).await
    }
}
