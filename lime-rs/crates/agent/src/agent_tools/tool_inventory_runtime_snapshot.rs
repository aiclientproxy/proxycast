use crate::agent_tools::inventory::{
    build_tool_inventory, AgentToolInventoryBuildInput, AgentToolInventorySnapshot,
    ExtensionToolInventorySeed,
};
use crate::agent_tools::tool_inventory_runtime_adapter::read_agent_tool_inventory_runtime_seed;
use crate::mcp::McpToolDefinition;
use crate::AgentRuntimeState;
use code_mode_protocol::RuntimeToolDefinition;
use lime_core::config::ToolExecutionPolicyConfig as ConfigToolExecutionPolicyConfig;
use lime_mcp::{build_runtime_extension_surface, runtime_extension_name};
use std::collections::{BTreeMap, HashSet};
use tool_runtime::tool_extension::RuntimeExtensionConfig;

#[derive(Debug, Clone)]
pub struct AgentToolInventoryReadInput {
    pub surface: crate::agent_tools::catalog::WorkspaceToolSurface,
    pub caller: String,
    pub warnings: Vec<String>,
    pub persisted_execution_policy: Option<ConfigToolExecutionPolicyConfig>,
    pub request_metadata: Option<serde_json::Value>,
    pub mcp_server_names: Vec<String>,
    pub mcp_tools: Vec<McpToolDefinition>,
    pub resource_helpers_supported: bool,
}

struct AgentToolInventoryRuntimeSnapshot {
    agent_initialized: bool,
    warnings: Vec<String>,
    current_tool_definitions: Vec<RuntimeToolDefinition>,
    extension_configs: Vec<RuntimeExtensionConfig>,
    visible_extension_tools: Vec<ExtensionToolInventorySeed>,
    searchable_extension_tools: Vec<ExtensionToolInventorySeed>,
}

pub async fn read_agent_tool_inventory(
    agent_state: &AgentRuntimeState,
    input: AgentToolInventoryReadInput,
) -> AgentToolInventorySnapshot {
    let AgentToolInventoryReadInput {
        surface,
        caller,
        mut warnings,
        persisted_execution_policy,
        request_metadata,
        mcp_server_names,
        mcp_tools,
        resource_helpers_supported,
    } = input;
    let runtime_snapshot =
        read_agent_tool_inventory_runtime_snapshot(agent_state, &mcp_tools).await;
    warnings.extend(runtime_snapshot.warnings);

    build_tool_inventory(AgentToolInventoryBuildInput {
        surface,
        caller,
        agent_initialized: runtime_snapshot.agent_initialized,
        warnings,
        persisted_execution_policy,
        request_metadata,
        mcp_server_names,
        mcp_tools,
        current_tool_definitions: runtime_snapshot.current_tool_definitions,
        resource_helpers_supported,
        extension_configs: runtime_snapshot.extension_configs,
        visible_extension_tools: runtime_snapshot.visible_extension_tools,
        searchable_extension_tools: runtime_snapshot.searchable_extension_tools,
    })
}

async fn read_agent_tool_inventory_runtime_snapshot(
    agent_state: &AgentRuntimeState,
    mcp_tools: &[McpToolDefinition],
) -> AgentToolInventoryRuntimeSnapshot {
    let Some(mut seed) = read_agent_tool_inventory_runtime_seed(agent_state).await else {
        let mut extension_configs = Vec::new();
        let mut visible_extension_tools = Vec::new();
        let mut searchable_extension_tools = Vec::new();
        merge_mcp_extension_projection(
            &mut extension_configs,
            &mut visible_extension_tools,
            &mut searchable_extension_tools,
            mcp_tools,
        );
        return AgentToolInventoryRuntimeSnapshot {
            agent_initialized: false,
            warnings: vec!["Agent runtime is not initialized".to_string()],
            current_tool_definitions: Vec::new(),
            extension_configs,
            visible_extension_tools,
            searchable_extension_tools,
        };
    };
    merge_mcp_extension_projection(
        &mut seed.extension_configs,
        &mut seed.visible_extension_tools,
        &mut seed.searchable_extension_tools,
        mcp_tools,
    );

    AgentToolInventoryRuntimeSnapshot {
        agent_initialized: true,
        warnings: Vec::new(),
        current_tool_definitions: seed.current_tool_definitions,
        extension_configs: seed.extension_configs,
        visible_extension_tools: seed.visible_extension_tools,
        searchable_extension_tools: seed.searchable_extension_tools,
    }
}

fn merge_mcp_extension_projection(
    extension_configs: &mut Vec<RuntimeExtensionConfig>,
    visible_extension_tools: &mut Vec<ExtensionToolInventorySeed>,
    searchable_extension_tools: &mut Vec<ExtensionToolInventorySeed>,
    mcp_tools: &[McpToolDefinition],
) {
    if mcp_tools.is_empty() {
        return;
    }

    let mut existing_extensions = extension_configs
        .iter()
        .map(|config| config.name.clone())
        .collect::<HashSet<_>>();
    let mut tools_by_extension: BTreeMap<String, Vec<McpToolDefinition>> = BTreeMap::new();
    for tool in mcp_tools {
        let extension_name = runtime_extension_name(&tool.server_name);
        tools_by_extension
            .entry(extension_name)
            .or_default()
            .push(tool.clone());
        searchable_extension_tools.push(ExtensionToolInventorySeed {
            name: tool.name.clone(),
            description: tool.description.clone(),
        });
        if mcp_tool_visible_by_default(tool) {
            visible_extension_tools.push(ExtensionToolInventorySeed {
                name: tool.name.clone(),
                description: tool.description.clone(),
            });
        }
    }

    for (extension_name, tools) in tools_by_extension {
        if existing_extensions.contains(&extension_name) {
            continue;
        }
        let server_name = tools
            .first()
            .map(|tool| tool.server_name.as_str())
            .unwrap_or(extension_name.as_str());
        let surface = build_runtime_extension_surface(
            &extension_name,
            format!("MCP server {server_name} tools"),
            &tools,
        );
        if !surface.has_tools() {
            continue;
        }
        let bridge_name = surface.name.clone();
        extension_configs.push(surface);
        existing_extensions.insert(bridge_name);
    }

    dedupe_extension_tool_seeds(visible_extension_tools);
    dedupe_extension_tool_seeds(searchable_extension_tools);
}

fn mcp_tool_visible_by_default(tool: &McpToolDefinition) -> bool {
    tool.always_visible.unwrap_or(false) || !tool.deferred_loading.unwrap_or(false)
}

fn dedupe_extension_tool_seeds(seeds: &mut Vec<ExtensionToolInventorySeed>) {
    let mut seen = HashSet::new();
    seeds.retain(|seed| seen.insert(seed.name.clone()));
    seeds.sort_by(|left, right| left.name.cmp(&right.name));
}
