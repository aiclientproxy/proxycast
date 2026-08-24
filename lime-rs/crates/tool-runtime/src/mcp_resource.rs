use crate::tool_definition::RuntimeToolDefinition;
use crate::tool_executor::{
    RuntimeToolExecutionError, RuntimeToolExecutionFuture, RuntimeToolExecutionRequest,
    RuntimeToolExecutionResult, RuntimeToolExecutor, RuntimeToolExecutorHandle,
    RuntimeToolPolicyErrorKind,
};
use app_server_protocol::{
    protocol::v2::{McpServerResourceReadParams, McpServerResourceReadResponse},
    McpResourceListResponse,
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
    async fn list_mcp_resources(
        &self,
        session_id: &str,
        thread_id: &str,
        server: Option<&str>,
        cursor: Option<String>,
    ) -> Result<McpResourceListResponse, String>;

    async fn read_mcp_resource(
        &self,
        session_id: &str,
        thread_id: &str,
        params: McpServerResourceReadParams,
    ) -> Result<McpServerResourceReadResponse, String>;
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
                    let server = input
                        .server
                        .as_deref()
                        .map(str::trim)
                        .filter(|value| !value.is_empty());
                    let cursor = input
                        .cursor
                        .as_deref()
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(ToOwned::to_owned);
                    if cursor.is_some() && server.is_none() {
                        return Err(RuntimeToolExecutionError::new(
                            "cursor 只能在指定 server 时使用",
                            Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
                                "mcp_resource_cursor_requires_server".to_string(),
                            )),
                        ));
                    }
                    if server == Some(lime_skills::APPS_MCP_SERVER_NAME)
                        && !orchestrator_feature_enabled(request.turn_context, "mcp")
                    {
                        return Err(mcp_resource_error(
                            "MCP server 'codex_apps' 已由 orchestrator.mcp.enabled 禁用",
                            "orchestrator_mcp_disabled",
                        ));
                    }
                    let thread_id = current_thread_id(request.turn_context)?;
                    let mut response = self
                        .gateway
                        .list_mcp_resources(request.context.session_id(), thread_id, server, cursor)
                        .await
                        .map_err(|error| {
                            mcp_resource_error(error, "mcp_resource_list_gateway_failed")
                        })?;
                    if server.is_none()
                        && !orchestrator_feature_enabled(request.turn_context, "mcp")
                    {
                        response.resources = response
                            .resources
                            .into_iter()
                            .filter(|value| {
                                resource_server(value) != Some(lime_skills::APPS_MCP_SERVER_NAME)
                            })
                            .collect();
                        response.resource_templates = response
                            .resource_templates
                            .into_iter()
                            .filter(|value| {
                                resource_server(value) != Some(lime_skills::APPS_MCP_SERVER_NAME)
                            })
                            .collect();
                    }
                    Ok(list_mcp_resources_result(server, response))
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
                    if server == lime_skills::APPS_MCP_SERVER_NAME
                        && !orchestrator_feature_enabled(request.turn_context, "mcp")
                        && !orchestrator_skill_resource_allowed(request.turn_context, uri)
                    {
                        return Err(mcp_resource_error(
                            "MCP server 'codex_apps' 已由 orchestrator.mcp.enabled 禁用",
                            "orchestrator_mcp_disabled",
                        ));
                    }
                    let thread_id = current_thread_id(request.turn_context)?;
                    let response = self
                        .gateway
                        .read_mcp_resource(
                            request.context.session_id(),
                            thread_id,
                            McpServerResourceReadParams {
                                thread_id: Some(thread_id.to_string()),
                                origin_call_id: None,
                                server: server.to_string(),
                                uri: uri.to_string(),
                                connector_id: None,
                            },
                        )
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
        "next_cursor": response.next_cursor,
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

fn current_thread_id(
    turn_context: Option<&crate::tool_executor::RuntimeToolTurnContext>,
) -> Result<&str, RuntimeToolExecutionError> {
    turn_context
        .and_then(|context| context.metadata.get("app_server_runtime_backend"))
        .and_then(|metadata| metadata.get("threadId"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|thread_id| !thread_id.is_empty())
        .ok_or_else(|| {
            mcp_resource_error(
                "MCP resource tool requires canonical thread identity",
                "mcp_resource_thread_identity_missing",
            )
        })
}

fn orchestrator_feature_enabled(
    turn_context: Option<&crate::tool_executor::RuntimeToolTurnContext>,
    feature: &str,
) -> bool {
    let Some(config) = turn_context.and_then(|context| context.metadata.get("config")) else {
        return true;
    };
    if config.pointer("/orchestrator/loadError").is_some() {
        return false;
    }
    config
        .pointer(&format!("/orchestrator/{feature}/enabled"))
        .and_then(Value::as_bool)
        .unwrap_or(true)
}

fn orchestrator_skill_resource_allowed(
    turn_context: Option<&crate::tool_executor::RuntimeToolTurnContext>,
    uri: &str,
) -> bool {
    if !orchestrator_feature_enabled(turn_context, "skills") {
        return false;
    }
    let Some(snapshot) = turn_context
        .and_then(|context| {
            context
                .metadata
                .get(lime_skills::SKILL_SNAPSHOT_TURN_METADATA_KEY)
        })
        .and_then(|value| {
            serde_json::from_value::<lime_skills::AgentSkillSnapshot>(value.clone()).ok()
        })
    else {
        return false;
    };
    snapshot.skills.iter().any(|skill| {
        skill.enabled
            && skill.authority == lime_skills::AgentSkillAuthority::Orchestrator
            && skill.skill_file_path.to_str() == Some(uri)
    })
}

fn mcp_resource_error(message: impl Into<String>, reason: &str) -> RuntimeToolExecutionError {
    RuntimeToolExecutionError::new(
        message,
        Some(RuntimeToolPolicyErrorKind::ExecutionFailed(
            reason.to_string(),
        )),
    )
}

fn read_mcp_resource_result(
    server: &str,
    response: McpServerResourceReadResponse,
) -> RuntimeToolExecutionResult {
    let uri = response.contents.first().map(|content| match content {
        app_server_protocol::protocol::v2::McpServerResourceContent::Text { uri, .. }
        | app_server_protocol::protocol::v2::McpServerResourceContent::Blob { uri, .. } => {
            uri.clone()
        }
    });
    let has_text = response.contents.iter().any(|content| {
        matches!(
            content,
            app_server_protocol::protocol::v2::McpServerResourceContent::Text { .. }
        )
    });
    let has_blob = response.contents.iter().any(|content| {
        matches!(
            content,
            app_server_protocol::protocol::v2::McpServerResourceContent::Blob { .. }
        )
    });
    let output = serde_json::to_value(response).unwrap_or_else(|_| json!({}));
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
        async fn list_mcp_resources(
            &self,
            _session_id: &str,
            _thread_id: &str,
            _server: Option<&str>,
            _cursor: Option<String>,
        ) -> Result<McpResourceListResponse, String> {
            Ok(McpResourceListResponse {
                resources: vec![json!({
                    "uri": "docs://readme",
                    "name": "README",
                    "server_name": "docs"
                })],
                resource_templates: Vec::new(),
                next_cursor: None,
            })
        }

        async fn read_mcp_resource(
            &self,
            _session_id: &str,
            _thread_id: &str,
            params: McpServerResourceReadParams,
        ) -> Result<McpServerResourceReadResponse, String> {
            Ok(McpServerResourceReadResponse {
                contents: vec![
                    app_server_protocol::protocol::v2::McpServerResourceContent::Text {
                        uri: params.uri,
                        mime_type: Some("text/markdown".to_string()),
                        text: "hello".to_string(),
                        meta: None,
                    },
                ],
                origin_call_id: None,
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

    fn turn_context() -> agent_protocol::turn_context::TurnContextOverride {
        agent_protocol::turn_context::TurnContextOverride {
            metadata: HashMap::from([(
                "app_server_runtime_backend".to_string(),
                json!({"threadId": "thread-1"}),
            )]),
            ..Default::default()
        }
    }

    fn orchestrator_disabled_turn_context(
        skill_uri: Option<&str>,
    ) -> agent_protocol::turn_context::TurnContextOverride {
        let mut metadata = HashMap::from([
            (
                "app_server_runtime_backend".to_string(),
                json!({"threadId": "thread-1"}),
            ),
            (
                "config".to_string(),
                json!({
                    "orchestrator": {
                        "skills": {"enabled": true},
                        "mcp": {"enabled": false}
                    }
                }),
            ),
        ]);
        if let Some(skill_uri) = skill_uri {
            metadata.insert(
                lime_skills::SKILL_SNAPSHOT_TURN_METADATA_KEY.to_string(),
                json!({
                    "roots": [],
                    "skills": [{
                        "skill_id": "orchestrator:release-notes",
                        "name": "release-notes",
                        "description": "Release notes",
                        "scope": "orchestrator",
                        "source": "orchestrator",
                        "authority": "orchestrator",
                        "enabled": true,
                        "interface": {
                            "display_name": "Release notes",
                            "execution_mode": "mcp_resource",
                            "provider": "codex_apps",
                            "model": null,
                            "argument_hint": null
                        },
                        "dependencies": {"tools": []},
                        "policy": {
                            "allow_implicit_invocation": true,
                            "when_to_use": null
                        },
                        "capabilities": [],
                        "directory": "skill://delivery/release-notes",
                        "skill_file_path": skill_uri
                    }]
                }),
            );
        }
        agent_protocol::turn_context::TurnContextOverride {
            metadata,
            ..Default::default()
        }
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
                turn_context: Some(&turn_context()),
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
                turn_context: Some(&turn_context()),
            })
            .await
            .expect("read resource should succeed");

        assert!(result.success);
        assert_eq!(result.metadata.get("uri"), Some(&json!("docs://readme")));
        assert_eq!(result.metadata.get("server"), Some(&json!("docs")));
        assert!(result.output.contains("hello"));
    }

    #[tokio::test]
    async fn orchestrator_mcp_gate_rejects_apps_catalog_but_allows_snapshot_skill_read() {
        let handle = runtime_mcp_resource_executor_handle(Arc::new(FakeMcpResourceGateway));
        let list_params = json!({"server": lime_skills::APPS_MCP_SERVER_NAME});
        let error = handle
            .execute(RuntimeToolExecutionRequest {
                tool_name: LIST_MCP_RESOURCES_TOOL_NAME,
                params: &list_params,
                context: &context(),
                turn_context: Some(&orchestrator_disabled_turn_context(None)),
            })
            .await
            .expect_err("disabled Apps MCP catalog must fail closed");
        assert!(error.message().contains("orchestrator.mcp.enabled"));

        let uri = "skill://delivery/release-notes/SKILL.md";
        let read_params = json!({
            "server": lime_skills::APPS_MCP_SERVER_NAME,
            "uri": uri
        });
        let result = handle
            .execute(RuntimeToolExecutionRequest {
                tool_name: READ_MCP_RESOURCE_TOOL_NAME,
                params: &read_params,
                context: &context(),
                turn_context: Some(&orchestrator_disabled_turn_context(Some(uri))),
            })
            .await
            .expect("snapshot-owned Skill resource stays readable");
        assert!(result.success);

        let other_uri = json!({
            "server": lime_skills::APPS_MCP_SERVER_NAME,
            "uri": "skill://delivery/other/SKILL.md"
        });
        assert!(handle
            .execute(RuntimeToolExecutionRequest {
                tool_name: READ_MCP_RESOURCE_TOOL_NAME,
                params: &other_uri,
                context: &context(),
                turn_context: Some(&orchestrator_disabled_turn_context(Some(uri))),
            })
            .await
            .is_err());
    }
}
