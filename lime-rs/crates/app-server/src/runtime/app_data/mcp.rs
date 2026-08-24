use super::requires_current;
use super::NoopAppDataSource;
use super::RuntimeCoreError;
use app_server_protocol::protocol::v2::{
    McpServerResourceReadParams, McpServerResourceReadResponse,
};
use app_server_protocol::*;
use async_trait::async_trait;
use lime_mcp::{McpOAuthLoginHandle, McpRuntimeServerSpec};

#[async_trait]
pub trait McpAppDataSource: Send + Sync {
    fn mcp_elicitation_router(&self) -> lime_mcp::ElicitationRequestRouter {
        lime_mcp::ElicitationRequestRouter::default()
    }

    async fn list_mcp_servers(&self) -> Result<McpServerListResponse, RuntimeCoreError> {
        Ok(McpServerListResponse::default())
    }

    async fn list_mcp_servers_with_status_v2(
        &self,
        _params: app_server_protocol::protocol::v2::ListMcpServerStatusParams,
    ) -> Result<app_server_protocol::protocol::v2::ListMcpServerStatusResponse, RuntimeCoreError>
    {
        Ok(
            app_server_protocol::protocol::v2::ListMcpServerStatusResponse {
                data: Vec::new(),
                next_cursor: None,
            },
        )
    }

    async fn create_mcp_server(
        &self,
        _params: McpServerCreateParams,
    ) -> Result<McpServerListResponse, RuntimeCoreError> {
        Err(requires_current("mcpServer/create"))
    }

    async fn update_mcp_server(
        &self,
        _params: McpServerUpdateParams,
    ) -> Result<McpServerListResponse, RuntimeCoreError> {
        Err(requires_current("mcpServer/update"))
    }

    async fn delete_mcp_server(
        &self,
        _params: McpServerDeleteParams,
    ) -> Result<McpServerListResponse, RuntimeCoreError> {
        Err(requires_current("mcpServer/delete"))
    }

    async fn set_mcp_server_enabled(
        &self,
        _params: McpServerEnabledSetParams,
    ) -> Result<McpServerListResponse, RuntimeCoreError> {
        Err(requires_current("mcpServer/enabled/set"))
    }

    async fn import_mcp_servers_from_app(
        &self,
        _params: McpServerImportFromAppParams,
    ) -> Result<McpServerImportFromAppResponse, RuntimeCoreError> {
        Err(requires_current("mcpServer/importFromApp"))
    }

    async fn sync_all_mcp_servers_to_live(
        &self,
    ) -> Result<McpServerListResponse, RuntimeCoreError> {
        Err(requires_current("mcpServer/syncAllToLive"))
    }

    async fn start_mcp_server(
        &self,
        _params: McpServerStartParams,
    ) -> Result<McpServerLifecycleResponse, RuntimeCoreError> {
        Err(requires_current("mcpServer/start"))
    }

    async fn stop_mcp_server(
        &self,
        _params: McpServerStopParams,
    ) -> Result<McpServerLifecycleResponse, RuntimeCoreError> {
        Err(requires_current("mcpServer/stop"))
    }

    async fn login_mcp_server_oauth(
        &self,
        _params: McpServerOauthLoginParams,
    ) -> Result<McpOAuthLoginHandle, RuntimeCoreError> {
        Err(requires_current("mcpServer/oauth/login"))
    }

    async fn list_mcp_tools(&self) -> Result<McpToolListResponse, RuntimeCoreError> {
        Ok(McpToolListResponse::default())
    }

    async fn list_mcp_runtime_server_specs(
        &self,
    ) -> Result<Vec<McpRuntimeServerSpec>, RuntimeCoreError> {
        Ok(Vec::new())
    }

    async fn list_mcp_tools_for_context(
        &self,
        _params: McpToolListForContextParams,
    ) -> Result<McpToolListResponse, RuntimeCoreError> {
        Ok(McpToolListResponse::default())
    }

    async fn search_mcp_tools(
        &self,
        _params: McpToolSearchParams,
    ) -> Result<McpToolListResponse, RuntimeCoreError> {
        Ok(McpToolListResponse::default())
    }

    async fn list_mcp_prompts(&self) -> Result<McpPromptListResponse, RuntimeCoreError> {
        Ok(McpPromptListResponse::default())
    }

    async fn get_mcp_prompt(
        &self,
        _params: McpPromptGetParams,
    ) -> Result<McpPromptGetResponse, RuntimeCoreError> {
        Err(requires_current("mcpPrompt/get"))
    }

    async fn list_mcp_resources(&self) -> Result<McpResourceListResponse, RuntimeCoreError> {
        Ok(McpResourceListResponse::default())
    }

    async fn read_mcp_server_resource(
        &self,
        _params: McpServerResourceReadParams,
    ) -> Result<McpServerResourceReadResponse, RuntimeCoreError> {
        Err(requires_current("mcpServer/resource/read"))
    }

    async fn subscribe_mcp_resource(
        &self,
        _params: McpResourceSubscribeParams,
    ) -> Result<McpResourceSubscriptionResponse, RuntimeCoreError> {
        Err(requires_current("mcpResource/subscribe"))
    }

    async fn unsubscribe_mcp_resource(
        &self,
        _params: McpResourceUnsubscribeParams,
    ) -> Result<McpResourceSubscriptionResponse, RuntimeCoreError> {
        Err(requires_current("mcpResource/unsubscribe"))
    }
}

impl McpAppDataSource for NoopAppDataSource {}
