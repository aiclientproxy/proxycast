use crate::runtime::ToolInventoryReadRequest;
use crate::AppDataSource;
use crate::RuntimeCoreError;
use app_server_protocol::protocol::v2::{ListMcpServerStatusParams, ListMcpServerStatusResponse};
use app_server_protocol::McpToolListResponse;
use lime_agent::agent_tools::catalog::WorkspaceToolSurface;
use lime_agent::agent_tools::execution::persisted_tool_execution_policy_from_metadata;
use lime_agent::agent_tools::{read_agent_tool_inventory, AgentToolInventoryReadInput};
use lime_agent::AgentRuntimeState;
use lime_core::tool_calling::extract_tool_surface_metadata;
use lime_mcp::McpToolDefinition;
use serde_json::{Map, Value};
use std::sync::Arc;

pub(crate) async fn read_tool_inventory(
    agent_state: &AgentRuntimeState,
    request: ToolInventoryReadRequest,
    config_metadata: Option<Value>,
    app_data_source: Option<Arc<dyn AppDataSource>>,
) -> Result<serde_json::Value, RuntimeCoreError> {
    let caller = request
        .caller
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("assistant")
        .to_string();
    let request_metadata = merged_inventory_metadata(request.metadata, config_metadata);
    let persisted_execution_policy =
        persisted_tool_execution_policy_from_metadata(request_metadata.as_ref());
    let mcp_snapshot = read_mcp_inventory_snapshot(app_data_source).await;
    let inventory = read_agent_tool_inventory(
        agent_state,
        AgentToolInventoryReadInput {
            surface: WorkspaceToolSurface {
                workbench: request.workbench,
            },
            caller,
            warnings: mcp_snapshot.warnings.clone(),
            persisted_execution_policy,
            request_metadata,
            mcp_server_names: mcp_snapshot.server_names.clone(),
            mcp_tools: mcp_snapshot.tools.clone(),
            resource_helpers_supported: mcp_snapshot.resource_helpers_supported,
        },
    )
    .await;
    let value = serde_json::to_value(inventory)
        .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?;
    Ok(value)
}

#[derive(Debug, Clone, Default)]
struct McpInventorySnapshot {
    server_names: Vec<String>,
    tools: Vec<McpToolDefinition>,
    resource_helpers_supported: bool,
    warnings: Vec<String>,
}

async fn read_mcp_inventory_snapshot(
    app_data_source: Option<Arc<dyn AppDataSource>>,
) -> McpInventorySnapshot {
    let Some(app_data_source) = app_data_source else {
        return McpInventorySnapshot::default();
    };

    let mut snapshot = McpInventorySnapshot::default();
    match app_data_source
        .list_mcp_servers_with_status_v2(ListMcpServerStatusParams {
            cursor: None,
            limit: None,
            detail: Some(
                app_server_protocol::protocol::v2::McpServerStatusDetail::ToolsAndAuthOnly,
            ),
            thread_id: None,
        })
        .await
    {
        Ok(response) => apply_mcp_server_status_snapshot(&mut snapshot, response),
        Err(error) => snapshot
            .warnings
            .push(format!("MCP server status snapshot unavailable: {error}")),
    }
    match app_data_source.list_mcp_tools().await {
        Ok(response) => apply_mcp_tool_snapshot(&mut snapshot, response),
        Err(error) => snapshot
            .warnings
            .push(format!("MCP tool snapshot unavailable: {error}")),
    }

    snapshot.server_names.sort();
    snapshot.server_names.dedup();
    snapshot
}

fn apply_mcp_server_status_snapshot(
    snapshot: &mut McpInventorySnapshot,
    response: ListMcpServerStatusResponse,
) {
    for server in response.data {
        snapshot.server_names.push(server.name);
        snapshot.resource_helpers_supported |= server.runtime_status
            == Some(app_server_protocol::protocol::v2::McpServerConnectionStatus::Connected)
            && (!server.resources.is_empty() || !server.resource_templates.is_empty());
    }
}

fn apply_mcp_tool_snapshot(snapshot: &mut McpInventorySnapshot, response: McpToolListResponse) {
    for tool in response.tools {
        match serde_json::from_value::<McpToolDefinition>(tool) {
            Ok(mut tool) => {
                hydrate_mcp_tool_metadata(&mut tool);
                snapshot.server_names.push(tool.server_name.clone());
                snapshot.tools.push(tool);
            }
            Err(error) => snapshot
                .warnings
                .push(format!("MCP tool snapshot entry skipped: {error}")),
        }
    }
}

fn hydrate_mcp_tool_metadata(tool: &mut McpToolDefinition) {
    if tool.deferred_loading.is_some()
        && tool.always_visible.is_some()
        && tool.allowed_callers.is_some()
        && tool.tags.is_some()
        && tool.input_examples.is_some()
    {
        return;
    }
    let metadata = extract_tool_surface_metadata(&tool.name, &tool.input_schema);
    if tool.deferred_loading.is_none() {
        tool.deferred_loading = metadata.deferred_loading;
    }
    if tool.always_visible.is_none() {
        tool.always_visible = metadata.always_visible;
    }
    if tool.allowed_callers.is_none() {
        tool.allowed_callers = metadata.allowed_callers;
    }
    if tool.tags.is_none() {
        tool.tags = metadata.tags;
    }
    if tool.input_examples.is_none() && !metadata.input_examples.is_empty() {
        tool.input_examples = Some(metadata.input_examples);
    }
}

fn merged_inventory_metadata(
    request_metadata: Option<Value>,
    config_metadata: Option<Value>,
) -> Option<Value> {
    match (request_metadata, config_metadata) {
        (Some(Value::Object(mut request_object)), Some(config_metadata)) => {
            request_object
                .entry("config".to_string())
                .or_insert(config_metadata);
            Some(Value::Object(request_object))
        }
        (Some(request_metadata), _) => Some(request_metadata),
        (None, Some(config_metadata)) => {
            let mut object = Map::new();
            object.insert("config".to_string(), config_metadata);
            Some(Value::Object(object))
        }
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn inventory_request(metadata: Option<Value>) -> ToolInventoryReadRequest {
        ToolInventoryReadRequest {
            caller: Some("assistant".to_string()),
            workbench: true,
            metadata,
        }
    }

    async fn read_snapshot(
        request_metadata: Option<Value>,
        config_metadata: Option<Value>,
    ) -> Value {
        read_tool_inventory(
            &AgentRuntimeState::new(),
            inventory_request(request_metadata),
            config_metadata,
            None,
        )
        .await
        .expect("tool inventory should be readable")
    }

    fn catalog_tool<'a>(inventory: &'a Value, name: &str) -> &'a Value {
        inventory
            .get("catalog_tools")
            .and_then(Value::as_array)
            .and_then(|entries| {
                entries
                    .iter()
                    .find(|entry| entry.get("name").and_then(Value::as_str) == Some(name))
            })
            .expect("catalog entry should exist")
    }

    #[tokio::test]
    async fn read_tool_inventory_uses_persisted_config_metadata_without_request_metadata() {
        let inventory = read_snapshot(
            None,
            Some(json!({
                "agent": {
                    "toolExecution": {
                        "toolOverrides": {
                            "exec_command": {
                                "warningPolicy": "none",
                                "sandboxProfile": "none"
                            }
                        }
                    }
                }
            })),
        )
        .await;

        assert_eq!(
            catalog_tool(&inventory, "exec_command")
                .get("execution_warning_policy")
                .and_then(Value::as_str),
            Some("none")
        );
        assert_eq!(
            catalog_tool(&inventory, "exec_command")
                .get("execution_warning_policy_source")
                .and_then(Value::as_str),
            Some("persisted")
        );
        assert_eq!(
            catalog_tool(&inventory, "exec_command")
                .get("execution_sandbox_profile")
                .and_then(Value::as_str),
            Some("none")
        );
        assert_eq!(
            catalog_tool(&inventory, "exec_command")
                .get("execution_sandbox_profile_source")
                .and_then(Value::as_str),
            Some("persisted")
        );
    }

    #[tokio::test]
    async fn read_tool_inventory_runtime_metadata_overrides_persisted_config_metadata() {
        let inventory = read_snapshot(
            Some(json!({
                "harness": {
                    "executionPolicy": {
                        "toolOverrides": {
                            "exec_command": {
                                "warningPolicy": "shell_command_risk",
                                "sandboxProfile": "workspace_command"
                            }
                        }
                    }
                }
            })),
            Some(json!({
                "agent": {
                    "toolExecution": {
                        "toolOverrides": {
                            "exec_command": {
                                "warningPolicy": "none",
                                "sandboxProfile": "none"
                            }
                        }
                    }
                }
            })),
        )
        .await;

        assert_eq!(
            catalog_tool(&inventory, "exec_command")
                .get("execution_warning_policy")
                .and_then(Value::as_str),
            Some("shell_command_risk")
        );
        assert_eq!(
            catalog_tool(&inventory, "exec_command")
                .get("execution_warning_policy_source")
                .and_then(Value::as_str),
            Some("runtime")
        );
        assert_eq!(
            catalog_tool(&inventory, "exec_command")
                .get("execution_sandbox_profile")
                .and_then(Value::as_str),
            Some("workspace_command")
        );
        assert_eq!(
            catalog_tool(&inventory, "exec_command")
                .get("execution_sandbox_profile_source")
                .and_then(Value::as_str),
            Some("runtime")
        );
    }
}
