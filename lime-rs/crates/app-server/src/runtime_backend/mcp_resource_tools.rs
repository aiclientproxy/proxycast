use app_server_protocol::{
    protocol::v2::{McpServerResourceReadParams, McpServerResourceReadResponse},
    McpResourceListResponse,
};
use async_trait::async_trait;
use lime_agent::AgentRuntimeState;
use std::sync::Arc;
use tool_runtime::mcp_resource::McpResourceGateway;

pub(crate) fn mcp_resource_gateway(agent_state: AgentRuntimeState) -> Arc<dyn McpResourceGateway> {
    Arc::new(AppServerMcpResourceGateway { agent_state })
}

struct AppServerMcpResourceGateway {
    agent_state: AgentRuntimeState,
}

#[async_trait]
impl McpResourceGateway for AppServerMcpResourceGateway {
    async fn list_mcp_resources(
        &self,
        session_id: &str,
        thread_id: &str,
        server: Option<&str>,
        cursor: Option<String>,
    ) -> Result<McpResourceListResponse, String> {
        if let Some(server) = server {
            let page = self
                .agent_state
                .list_mcp_resource_page(session_id, thread_id, server, cursor)
                .await?;
            return Ok(McpResourceListResponse {
                resources: serializable_values(page.resources)?,
                resource_templates: Vec::new(),
                next_cursor: page.next_cursor,
            });
        }
        if cursor.is_some() {
            return Err("MCP resource cursor requires an exact server".to_string());
        }
        let (resources, templates) = self
            .agent_state
            .list_mcp_resources(session_id, thread_id)
            .await?;
        Ok(McpResourceListResponse {
            resources: serializable_values(resources)?,
            resource_templates: serializable_values(templates)?,
            next_cursor: None,
        })
    }

    async fn read_mcp_resource(
        &self,
        session_id: &str,
        thread_id: &str,
        params: McpServerResourceReadParams,
    ) -> Result<McpServerResourceReadResponse, String> {
        if params
            .thread_id
            .as_deref()
            .is_some_and(|value| value != thread_id)
        {
            return Err("MCP resource thread identity mismatch".to_string());
        }
        let content = self
            .agent_state
            .read_mcp_resource(session_id, thread_id, &params.server, &params.uri)
            .await?;
        let content = match (content.text, content.blob) {
            (Some(text), None) => Some(
                app_server_protocol::protocol::v2::McpServerResourceContent::Text {
                    uri: content.uri,
                    mime_type: content.mime_type,
                    text,
                    meta: content.meta,
                },
            ),
            (None, Some(blob)) => Some(
                app_server_protocol::protocol::v2::McpServerResourceContent::Blob {
                    uri: content.uri,
                    mime_type: content.mime_type,
                    blob,
                    meta: content.meta,
                },
            ),
            (None, None) => None,
            (Some(_), Some(_)) => {
                return Err("MCP resource response contained both text and blob".to_string());
            }
        };
        Ok(McpServerResourceReadResponse {
            contents: content.into_iter().collect(),
        })
    }
}

fn serializable_values<T: serde::Serialize>(
    values: Vec<T>,
) -> Result<Vec<serde_json::Value>, String> {
    values
        .into_iter()
        .map(|value| serde_json::to_value(value).map_err(|error| error.to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn mcp_resource_gateway_requires_session_runtime() {
        let gateway = mcp_resource_gateway(AgentRuntimeState::new());
        let error = gateway
            .list_mcp_resources("session", "thread", None, None)
            .await
            .expect_err("missing session runtime must fail closed");

        assert!(error.contains("not initialized"));
    }
}
