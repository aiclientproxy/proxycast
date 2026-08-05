use crate::tool_definition::RuntimeToolDefinition;
use crate::tool_executor::{
    RuntimeToolExecutionError, RuntimeToolExecutionFuture, RuntimeToolExecutionRequest,
    RuntimeToolExecutionResult, RuntimeToolExecutor, RuntimeToolExecutorHandle,
    RuntimeToolPolicyErrorKind,
};
use app_server_protocol::{
    McpResourceListResponse, McpResourceReadParams, McpResourceReadResponse,
};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

pub const LIST_MCP_RESOURCES_TOOL_NAME: &str = "list_mcp_resources";
pub const READ_MCP_RESOURCE_TOOL_NAME: &str = "read_mcp_resource";
pub const LIST_MCP_RESOURCES_LOOKUP_ALIASES: &[&str] =
    &["ListMcpResources", "ListMcpResourcesTool"];
pub const READ_MCP_RESOURCE_LOOKUP_ALIASES: &[&str] = &["ReadMcpResource", "ReadMcpResourceTool"];

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ListMcpResourcesInput {
    #[serde(default)]
    server: Option<String>,
    #[serde(default)]
    cursor: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadMcpResourceInput {
    server: String,
    uri: String,
}

#[async_trait]
pub trait McpResourceGateway: Send + Sync {
    async fn list_mcp_resources(&self) -> Result<McpResourceListResponse, String>;

    async fn read_mcp_resource(
        &self,
        params: McpResourceReadParams,
    ) -> Result<McpResourceReadResponse, String>;
}

pub struct RuntimeMcpResourceExecutor {
    gateway: Arc<dyn McpResourceGateway>,
}

impl RuntimeMcpResourceExecutor {
    pub fn new(gateway: Arc<dyn McpResourceGateway>) -> Self {
        Self { gateway }
    }
}

impl RuntimeToolExecutor for RuntimeMcpResourceExecutor {
    fn execute<'a>(
        &'a self,
        request: RuntimeToolExecutionRequest<'a>,
    ) -> RuntimeToolExecutionFuture<'a> {
        Box::pin(async move {
            match request.tool_name {
                LIST_MCP_RESOURCES_TOOL_NAME => {
                    let input = parse_list_input(request.params)?;
                    if input
                        .cursor
                        .as_deref()
                        .is_some_and(|value| !value.trim().is_empty())
                    {
                        return Err(RuntimeToolExecutionError::new(
                            "cursor 暂不支持；当前 Lime MCP resource gateway 一次返回完整资源列表",
                            Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
                                "unsupported_mcp_resource_cursor".to_string(),
                            )),
                        ));
                    }
                    let response = self.gateway.list_mcp_resources().await.map_err(|error| {
                        RuntimeToolExecutionError::new(
                            error,
                            Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
                                "mcp_resource_list_gateway_failed".to_string(),
                            )),
                        )
                    })?;
                    Ok(list_mcp_resources_result(input.server.as_deref(), response))
                }
                READ_MCP_RESOURCE_TOOL_NAME => {
                    let input = parse_read_input(request.params)?;
                    let server = input.server.trim();
                    if server.is_empty() {
                        return Err(RuntimeToolExecutionError::new(
                            "server 不能为空",
                            Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
                                "invalid_mcp_resource_server".to_string(),
                            )),
                        ));
                    }
                    let uri = input.uri.trim();
                    if uri.is_empty() {
                        return Err(RuntimeToolExecutionError::new(
                            "uri 不能为空",
                            Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
                                "invalid_mcp_resource_uri".to_string(),
                            )),
                        ));
                    }
                    let response = self
                        .gateway
                        .read_mcp_resource(McpResourceReadParams {
                            server: server.to_string(),
                            uri: uri.to_string(),
                            session_id: None,
                            thread_id: None,
                        })
                        .await
                        .map_err(|error| {
                            RuntimeToolExecutionError::new(
                                error,
                                Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
                                    "mcp_resource_read_gateway_failed".to_string(),
                                )),
                            )
                        })?;
                    Ok(read_mcp_resource_result(server, response))
                }
                other => Err(RuntimeToolExecutionError::new(
                    format!("unsupported MCP resource tool: {other}"),
                    Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
                        "unsupported_mcp_resource_tool".to_string(),
                    )),
                )),
            }
        })
    }
}

pub fn list_mcp_resources_tool_definition() -> RuntimeToolDefinition {
    RuntimeToolDefinition::new(
        LIST_MCP_RESOURCES_TOOL_NAME,
        "Lists resources provided by MCP servers. Prefer resources over web search when possible.",
        json!({
            "type": "object",
            "properties": {
                "server": {
                    "type": "string",
                    "description": "MCP server name. Omit to list resources from every configured server."
                },
                "cursor": {
                    "type": "string",
                    "description": "Opaque cursor from a previous list_mcp_resources call; omit for the first page."
                }
            },
            "additionalProperties": false
        }),
    )
}

pub fn read_mcp_resource_tool_definition() -> RuntimeToolDefinition {
    RuntimeToolDefinition::new(
        READ_MCP_RESOURCE_TOOL_NAME,
        "Read a specific resource from an MCP server given the server name and resource URI.",
        json!({
            "type": "object",
            "properties": {
                "server": {
                    "type": "string",
                    "description": "MCP server name exactly as configured. Must match the server field returned by list_mcp_resources."
                },
                "uri": {
                    "type": "string",
                    "description": "Resource URI to read. Must be one of the URIs returned by list_mcp_resources."
                }
            },
            "required": ["server", "uri"],
            "additionalProperties": false
        }),
    )
}

pub fn mcp_resource_tool_definitions() -> Vec<RuntimeToolDefinition> {
    vec![
        list_mcp_resources_tool_definition(),
        read_mcp_resource_tool_definition(),
    ]
}

pub fn runtime_mcp_resource_executor_handle(
    gateway: Arc<dyn McpResourceGateway>,
) -> RuntimeToolExecutorHandle {
    RuntimeToolExecutorHandle::new(Arc::new(RuntimeMcpResourceExecutor::new(gateway)))
}

pub fn check_runtime_mcp_resource_permissions() -> Result<(), RuntimeToolExecutionError> {
    Ok(())
}

fn parse_list_input(params: &Value) -> Result<ListMcpResourcesInput, RuntimeToolExecutionError> {
    serde_json::from_value(params.clone()).map_err(|error| {
        RuntimeToolExecutionError::new(
            format!("参数解析失败: {error}"),
            Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
                "invalid_mcp_resource_list_params".to_string(),
            )),
        )
    })
}

fn parse_read_input(params: &Value) -> Result<ReadMcpResourceInput, RuntimeToolExecutionError> {
    serde_json::from_value(params.clone()).map_err(|error| {
        RuntimeToolExecutionError::new(
            format!("参数解析失败: {error}"),
            Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
                "invalid_mcp_resource_read_params".to_string(),
            )),
        )
    })
}

fn list_mcp_resources_result(
    server: Option<&str>,
    response: McpResourceListResponse,
) -> RuntimeToolExecutionResult {
    let server = server.map(str::trim).filter(|value| !value.is_empty());
    let resources = filter_resource_values(response.resources, server);
    let resource_templates = filter_resource_values(response.resource_templates, server);
    let resource_count = resources.len();
    let template_count = resource_templates.len();
    let output = json!({
        "resources": resources,
        "resource_templates": resource_templates,
        "server": server,
        "resource_count": resource_count,
        "resource_template_count": template_count,
    });
    let mut metadata = HashMap::new();
    metadata.insert("tool_family".to_string(), json!("mcp_resource"));
    metadata.insert("operation".to_string(), json!("list"));
    metadata.insert("resource_count".to_string(), json!(resource_count));
    metadata.insert("resource_template_count".to_string(), json!(template_count));
    if let Some(server) = server {
        metadata.insert("server".to_string(), json!(server));
    }

    RuntimeToolExecutionResult::new(true, output.to_string(), None, metadata)
}

fn read_mcp_resource_result(
    server: &str,
    response: McpResourceReadResponse,
) -> RuntimeToolExecutionResult {
    let uri = response.uri.clone();
    let has_text = response.text.is_some();
    let has_blob = response.blob.is_some();
    let output = serde_json::to_value(response).unwrap_or_else(|_| json!({ "uri": uri }));
    let mut metadata = HashMap::new();
    metadata.insert("tool_family".to_string(), json!("mcp_resource"));
    metadata.insert("operation".to_string(), json!("read"));
    metadata.insert("uri".to_string(), json!(uri));
    metadata.insert("has_text".to_string(), json!(has_text));
    metadata.insert("has_blob".to_string(), json!(has_blob));
    metadata.insert("server".to_string(), json!(server));

    RuntimeToolExecutionResult::new(true, output.to_string(), None, metadata)
}

fn filter_resource_values(values: Vec<Value>, server: Option<&str>) -> Vec<Value> {
    let Some(server) = server else {
        return values;
    };
    values
        .into_iter()
        .filter(|value| resource_server(value) == Some(server))
        .collect()
}

fn resource_server(value: &Value) -> Option<&str> {
    value
        .get("server_name")
        .or_else(|| value.get("serverName"))
        .or_else(|| value.get("server"))
        .and_then(Value::as_str)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_executor::{RuntimeToolExecutionContext, RuntimeToolExecutionContextInput};
    use std::path::PathBuf;

    struct FakeMcpResourceGateway;

    #[async_trait]
    impl McpResourceGateway for FakeMcpResourceGateway {
        async fn list_mcp_resources(&self) -> Result<McpResourceListResponse, String> {
            Ok(McpResourceListResponse {
                resources: vec![json!({
                    "uri": "docs://readme",
                    "name": "README",
                    "server_name": "docs"
                })],
                resource_templates: Vec::new(),
            })
        }

        async fn read_mcp_resource(
            &self,
            params: McpResourceReadParams,
        ) -> Result<McpResourceReadResponse, String> {
            Ok(McpResourceReadResponse {
                uri: params.uri,
                mime_type: Some("text/markdown".to_string()),
                text: Some("hello".to_string()),
                blob: None,
                meta: None,
            })
        }
    }

    fn context() -> RuntimeToolExecutionContext {
        RuntimeToolExecutionContext::new(RuntimeToolExecutionContextInput {
            working_directory: PathBuf::from("."),
            session_id: "mcp-resource-test".to_string(),
            cancel_token: None,
            workspace_sandbox: None,
        })
    }

    #[tokio::test]
    async fn list_mcp_resources_uses_gateway() {
        let handle = runtime_mcp_resource_executor_handle(Arc::new(FakeMcpResourceGateway));
        let params = json!({ "server": "docs" });
        let result = handle
            .execute(RuntimeToolExecutionRequest {
                tool_name: LIST_MCP_RESOURCES_TOOL_NAME,
                params: &params,
                context: &context(),
                turn_context: None,
            })
            .await
            .expect("list resources should succeed");

        assert!(result.success);
        assert_eq!(result.metadata.get("resource_count"), Some(&json!(1)));
        assert!(result.output.contains("docs://readme"));
    }

    #[tokio::test]
    async fn read_mcp_resource_uses_gateway() {
        let handle = runtime_mcp_resource_executor_handle(Arc::new(FakeMcpResourceGateway));
        let params = json!({
            "server": "docs",
            "uri": "docs://readme"
        });
        let result = handle
            .execute(RuntimeToolExecutionRequest {
                tool_name: READ_MCP_RESOURCE_TOOL_NAME,
                params: &params,
                context: &context(),
                turn_context: None,
            })
            .await
            .expect("read resource should succeed");

        assert!(result.success);
        assert_eq!(result.metadata.get("uri"), Some(&json!("docs://readme")));
        assert_eq!(result.metadata.get("server"), Some(&json!("docs")));
        assert!(result.output.contains("hello"));
    }
}
