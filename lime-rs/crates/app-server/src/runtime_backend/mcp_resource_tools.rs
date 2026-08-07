use crate::AppDataSource;
use app_server_protocol::{
    protocol::v2::{McpServerResourceReadParams, McpServerResourceReadResponse},
    McpResourceListResponse,
};
use async_trait::async_trait;
use std::sync::Arc;
use tool_runtime::mcp_resource::McpResourceGateway;

pub(crate) fn mcp_resource_gateway(
    app_data_source: Arc<dyn AppDataSource>,
) -> Arc<dyn McpResourceGateway> {
    Arc::new(AppServerMcpResourceGateway { app_data_source })
}

struct AppServerMcpResourceGateway {
    app_data_source: Arc<dyn AppDataSource>,
}

#[async_trait]
impl McpResourceGateway for AppServerMcpResourceGateway {
    async fn list_mcp_resources(&self) -> Result<McpResourceListResponse, String> {
        self.app_data_source
            .list_mcp_resources()
            .await
            .map_err(|error| error.to_string())
    }

    async fn read_mcp_resource(
        &self,
        params: McpServerResourceReadParams,
    ) -> Result<McpServerResourceReadResponse, String> {
        self.app_data_source
            .read_mcp_server_resource(params)
            .await
            .map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NoopAppDataSource;

    #[tokio::test]
    async fn mcp_resource_gateway_uses_app_data_source_resource_list() {
        let gateway = mcp_resource_gateway(Arc::new(NoopAppDataSource));
        let response = gateway
            .list_mcp_resources()
            .await
            .expect("noop app data source returns empty resource list");

        assert!(response.resources.is_empty());
        assert!(response.resource_templates.is_empty());
    }
}
