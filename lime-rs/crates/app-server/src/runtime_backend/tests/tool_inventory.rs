use super::*;
use crate::AutomationManagementAppDataSource;
use crate::AutomationOverviewAppDataSource;
use crate::ConnectAppDataSource;
use crate::DiagnosticsAppDataSource;
use crate::GatewayAppDataSource;
use crate::KnowledgeAppDataSource;
use crate::McpAppDataSource;
use crate::MediaAppDataSource;
use crate::MemoryAppDataSource;
use crate::ModelProviderAppDataSource;
use crate::PluginDataSource;
use crate::RightSurfaceAppDataSource;
use crate::SessionAppDataSource;
use crate::SkillAppDataSource;
use crate::UsageStatsAppDataSource;
use crate::VoiceAppDataSource;
use crate::WorkspaceAppDataSource;
use crate::WorkspaceSkillBindingAppDataSource;
use app_server_protocol::McpServerStatusListResponse;
use app_server_protocol::McpToolListResponse;
use async_trait::async_trait;
use serde_json::json;
use std::sync::Mutex;

#[derive(Default)]
struct TestMcpInventoryDataSource {
    server_status_response: Mutex<Option<Result<McpServerStatusListResponse, String>>>,
    tool_list_response: Mutex<Option<Result<McpToolListResponse, String>>>,
}

impl TestMcpInventoryDataSource {
    fn with_server_status_response(
        self,
        response: Result<McpServerStatusListResponse, String>,
    ) -> Self {
        *self
            .server_status_response
            .lock()
            .expect("test MCP server status mutex poisoned") = Some(response);
        self
    }

    fn with_tool_list_response(self, response: Result<McpToolListResponse, String>) -> Self {
        *self
            .tool_list_response
            .lock()
            .expect("test MCP tool list mutex poisoned") = Some(response);
        self
    }
}

impl SessionAppDataSource for TestMcpInventoryDataSource {}
impl WorkspaceAppDataSource for TestMcpInventoryDataSource {}
impl SkillAppDataSource for TestMcpInventoryDataSource {}
impl WorkspaceSkillBindingAppDataSource for TestMcpInventoryDataSource {}
impl GatewayAppDataSource for TestMcpInventoryDataSource {}
impl MediaAppDataSource for TestMcpInventoryDataSource {}
impl VoiceAppDataSource for TestMcpInventoryDataSource {}
impl PluginDataSource for TestMcpInventoryDataSource {}
impl KnowledgeAppDataSource for TestMcpInventoryDataSource {}
impl AutomationOverviewAppDataSource for TestMcpInventoryDataSource {}
impl AutomationManagementAppDataSource for TestMcpInventoryDataSource {}
impl MemoryAppDataSource for TestMcpInventoryDataSource {}
impl DiagnosticsAppDataSource for TestMcpInventoryDataSource {}
impl UsageStatsAppDataSource for TestMcpInventoryDataSource {}
impl ModelProviderAppDataSource for TestMcpInventoryDataSource {}
impl ConnectAppDataSource for TestMcpInventoryDataSource {}
impl RightSurfaceAppDataSource for TestMcpInventoryDataSource {}

#[async_trait]
impl McpAppDataSource for TestMcpInventoryDataSource {
    async fn list_mcp_servers_with_status(
        &self,
    ) -> Result<McpServerStatusListResponse, RuntimeCoreError> {
        let response = self
            .server_status_response
            .lock()
            .expect("test MCP server status mutex poisoned")
            .clone();
        match response {
            Some(Ok(response)) => Ok(response),
            Some(Err(message)) => Err(RuntimeCoreError::Backend(message)),
            None => Ok(McpServerStatusListResponse::default()),
        }
    }

    async fn list_mcp_tools(&self) -> Result<McpToolListResponse, RuntimeCoreError> {
        let response = self
            .tool_list_response
            .lock()
            .expect("test MCP tool list mutex poisoned")
            .clone();
        match response {
            Some(Ok(response)) => Ok(response),
            Some(Err(message)) => Err(RuntimeCoreError::Backend(message)),
            None => Ok(McpToolListResponse::default()),
        }
    }
}

fn in_memory_db() -> lime_core::database::DbConnection {
    let db: lime_core::database::DbConnection = std::sync::Arc::new(std::sync::Mutex::new(
        rusqlite::Connection::open_in_memory().expect("db"),
    ));
    {
        let conn = db.lock().expect("db lock");
        lime_core::database::schema::create_tables(&conn).expect("schema");
    }
    db
}

async fn read_inventory(data_source: TestMcpInventoryDataSource) -> serde_json::Value {
    read_inventory_with_metadata(data_source, None).await
}

async fn read_inventory_with_metadata(
    data_source: TestMcpInventoryDataSource,
    metadata: Option<serde_json::Value>,
) -> serde_json::Value {
    let db = in_memory_db();
    let backend = RuntimeBackend::with_db(db.clone());
    ExecutionBackend::set_app_data_source(&backend, std::sync::Arc::new(data_source))
        .expect("app data source should be accepted");
    backend
        .agent_state
        .init_agent_with_db(&db)
        .await
        .expect("agent should initialize");

    ExecutionBackend::read_tool_inventory(
        &backend,
        ToolInventoryReadRequest {
            caller: Some("assistant".to_string()),
            workbench: true,
            metadata,
        },
    )
    .await
    .expect("tool inventory should be readable")
}

async fn read_inventory_without_initialized_agent(
    data_source: TestMcpInventoryDataSource,
) -> serde_json::Value {
    let db = in_memory_db();
    let backend = RuntimeBackend::with_db(db);
    ExecutionBackend::set_app_data_source(&backend, std::sync::Arc::new(data_source))
        .expect("app data source should be accepted");

    ExecutionBackend::read_tool_inventory(
        &backend,
        ToolInventoryReadRequest {
            caller: Some("assistant".to_string()),
            workbench: true,
            metadata: None,
        },
    )
    .await
    .expect("tool inventory should be readable")
}

#[tokio::test]
async fn runtime_backend_tool_inventory_reads_current_mcp_snapshot() {
    let data_source = TestMcpInventoryDataSource::default()
        .with_server_status_response(Ok(McpServerStatusListResponse {
            servers: vec![json!({
                "name": "context7",
                "is_running": true,
                "runtime_status": {
                    "supports_resources": true
                }
            })],
        }))
        .with_tool_list_response(Ok(McpToolListResponse {
            tools: vec![
                json!({
                    "server_name": "context7",
                    "name": "mcp__context7__resolve-library-id",
                    "description": "Resolve a package name to a Context7 library id",
                    "input_schema": {
                        "type": "object",
                        "x-lime": {
                            "deferred_loading": true,
                            "always_visible": true,
                            "allowed_callers": ["assistant"],
                            "tags": ["context7", "docs"]
                        }
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "libraryId": { "type": "string" }
                        }
                    }
                }),
                json!({
                    "server_name": "context7",
                    "name": "mcp__context7__query-docs",
                    "description": "Query Context7 documentation",
                    "input_schema": {
                        "type": "object",
                        "x-lime": {
                            "deferred_loading": true,
                            "always_visible": false,
                            "allowed_callers": ["assistant"],
                            "tags": ["context7", "docs"]
                        }
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "answer": { "type": "string" }
                        }
                    }
                }),
            ],
        }));

    let inventory = read_inventory(data_source).await;

    assert_eq!(inventory["mcp_servers"], json!(["context7"]));
    assert_eq!(inventory["counts"]["mcp_server_total"], json!(1));
    assert_eq!(inventory["counts"]["mcp_tool_total"], json!(2));
    assert_eq!(inventory["counts"]["mcp_tool_visible_total"], json!(1));
    assert_eq!(inventory["counts"]["extension_mcp_bridge_total"], json!(1));
    assert!(inventory["counts"]["extension_tool_total"]
        .as_u64()
        .is_some_and(|count| count >= 2));
    assert!(inventory["counts"]["extension_tool_visible_total"]
        .as_u64()
        .is_some_and(|count| count >= 1));
    let mcp_tools = inventory["mcp_tools"].as_array().expect("mcp tools");
    let resolve_tool = mcp_tools
        .iter()
        .find(|entry| entry["name"] == "mcp__context7__resolve-library-id")
        .expect("resolve-library-id MCP tool");
    assert_eq!(resolve_tool["server_name"], json!("context7"));
    assert_eq!(resolve_tool["deferred_loading"], json!(true));
    assert_eq!(resolve_tool["always_visible"], json!(true));
    assert_eq!(resolve_tool["allowed_callers"], json!(["assistant"]));
    assert_eq!(resolve_tool["tags"], json!(["context7", "docs"]));
    assert_eq!(resolve_tool["has_output_schema"], json!(true));
    assert!(mcp_tools
        .iter()
        .any(|entry| entry["name"] == "mcp__context7__query-docs"));
    let context7_surface = inventory["extension_surfaces"]
        .as_array()
        .expect("extension surfaces")
        .iter()
        .find(|entry| entry["extension_name"] == "mcp__context7")
        .expect("context7 extension surface");
    assert_eq!(context7_surface["source_kind"], json!("mcp_bridge"));
    assert_eq!(
        context7_surface["available_tools"],
        json!(["query-docs", "resolve-library-id"])
    );
    assert_eq!(
        context7_surface["always_expose_tools"],
        json!(["resolve-library-id"])
    );
    assert_eq!(
        context7_surface["searchable_tools"],
        json!([
            "mcp__context7__query-docs",
            "mcp__context7__resolve-library-id"
        ])
    );
    let extension_tools = inventory["extension_tools"]
        .as_array()
        .expect("extension tools");
    assert!(extension_tools.iter().any(|entry| {
        entry["name"] == "mcp__context7__query-docs"
            && entry["status"] == "deferred"
            && entry["visible_in_context"] == false
    }));
    assert!(extension_tools.iter().any(|entry| {
        entry["name"] == "mcp__context7__resolve-library-id"
            && entry["status"] == "visible"
            && entry["visible_in_context"] == true
    }));
    assert!(inventory["runtime_tools"]
        .as_array()
        .expect("runtime tools")
        .iter()
        .any(|entry| {
            entry["name"] == "mcp__context7__resolve-library-id"
                && entry["source_kind"] == "runtime_extension"
                && entry["source_label"] == "mcp__context7"
                && entry["status"] == "visible"
                && entry["visible_in_context"] == true
        }));
    let runtime_tools = inventory["runtime_tools"]
        .as_array()
        .expect("runtime tools");
    assert!(runtime_tools.iter().any(|entry| {
        entry["name"] == "list_mcp_resources"
            && entry["source_kind"] == "current_surface"
            && entry["visible_in_context"] == true
    }));
    assert!(runtime_tools.iter().any(|entry| {
        entry["name"] == "read_mcp_resource"
            && entry["source_kind"] == "current_surface"
            && entry["visible_in_context"] == true
    }));
    assert!(!inventory["native_tools"]
        .as_array()
        .expect("native tools")
        .iter()
        .any(|entry| {
            entry["name"] == "ListMcpResourcesTool" || entry["name"] == "ReadMcpResourceTool"
        }));
}

#[tokio::test]
async fn runtime_backend_tool_inventory_does_not_project_plugin_private_targets() {
    let data_source = TestMcpInventoryDataSource::default();
    let metadata = json!({
        "harness": {
            "plugin_runtime_capabilities": {
                "pluginId": "docs-plugin",
                "mcpBindings": [{"serverId": "context7", "toolKey": "context7/search"}]
            }
        }
    });

    let inventory = read_inventory_with_metadata(data_source, Some(metadata)).await;

    assert!(inventory.get("plugin_mcp_targets").is_none());
}

#[tokio::test]
async fn runtime_backend_tool_inventory_projects_mcp_bridge_before_agent_initialization() {
    let data_source = TestMcpInventoryDataSource::default()
        .with_server_status_response(Ok(McpServerStatusListResponse {
            servers: vec![json!({
                "name": "context7",
                "is_running": true,
                "runtime_status": {
                    "supports_resources": true
                }
            })],
        }))
        .with_tool_list_response(Ok(McpToolListResponse {
            tools: vec![
                json!({
                    "server_name": "context7",
                    "name": "mcp__context7__resolve-library-id",
                    "description": "Resolve a package name to a Context7 library id",
                    "input_schema": {
                        "type": "object",
                        "x-lime": {
                            "deferred_loading": true,
                            "always_visible": true,
                            "allowed_callers": ["assistant"],
                            "tags": ["context7", "docs"]
                        }
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "libraryId": { "type": "string" }
                        }
                    }
                }),
                json!({
                    "server_name": "context7",
                    "name": "mcp__context7__query-docs",
                    "description": "Query Context7 documentation",
                    "input_schema": {
                        "type": "object",
                        "x-lime": {
                            "deferred_loading": true,
                            "always_visible": false,
                            "allowed_callers": ["assistant"],
                            "tags": ["context7", "docs"]
                        }
                    }
                }),
            ],
        }));

    let inventory = read_inventory_without_initialized_agent(data_source).await;

    assert_eq!(inventory["agent_initialized"], json!(false));
    assert!(inventory["warnings"]
        .as_array()
        .expect("warnings")
        .iter()
        .any(|warning| warning == "Agent runtime is not initialized"));
    assert_eq!(inventory["mcp_servers"], json!(["context7"]));
    assert_eq!(inventory["counts"]["mcp_tool_total"], json!(2));
    assert_eq!(inventory["counts"]["extension_mcp_bridge_total"], json!(1));
    let context7_surface = inventory["extension_surfaces"]
        .as_array()
        .expect("extension surfaces")
        .iter()
        .find(|entry| entry["extension_name"] == "mcp__context7")
        .expect("context7 extension surface");
    assert_eq!(context7_surface["source_kind"], json!("mcp_bridge"));
    assert_eq!(
        context7_surface["available_tools"],
        json!(["query-docs", "resolve-library-id"])
    );
    assert_eq!(
        context7_surface["searchable_tools"],
        json!([
            "mcp__context7__query-docs",
            "mcp__context7__resolve-library-id"
        ])
    );
    assert!(inventory["runtime_tools"]
        .as_array()
        .expect("runtime tools")
        .iter()
        .any(|entry| {
            entry["name"] == "mcp__context7__resolve-library-id"
                && entry["source_kind"] == "runtime_extension"
                && entry["source_label"] == "mcp__context7"
                && entry["status"] == "visible"
        }));
}

#[tokio::test]
async fn runtime_backend_tool_inventory_keeps_warning_for_invalid_mcp_snapshot() {
    let data_source = TestMcpInventoryDataSource::default()
        .with_server_status_response(Ok(McpServerStatusListResponse {
            servers: vec![json!({
                "name": "docs",
                "isRunning": true,
                "runtimeStatus": {
                    "supportsResources": true
                }
            })],
        }))
        .with_tool_list_response(Ok(McpToolListResponse {
            tools: vec![json!({
                "server_name": "docs",
                "name": "mcp__docs__broken"
            })],
        }));

    let inventory = read_inventory(data_source).await;

    assert_eq!(inventory["mcp_servers"], json!(["docs"]));
    assert_eq!(inventory["counts"]["mcp_tool_total"], json!(0));
    assert!(inventory["warnings"]
        .as_array()
        .expect("warnings")
        .iter()
        .any(|warning| warning
            .as_str()
            .is_some_and(|value| value.starts_with("MCP tool snapshot entry skipped:"))));
}
