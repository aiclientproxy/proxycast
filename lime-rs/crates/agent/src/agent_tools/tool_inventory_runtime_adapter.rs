use super::inventory::ExtensionToolInventorySeed;
use crate::AgentRuntimeState;
use code_mode_protocol::RuntimeToolDefinition;
use tool_runtime::tool_extension::RuntimeExtensionConfig;

pub(super) struct AgentToolInventoryRuntimeSeed {
    pub(super) current_tool_definitions: Vec<RuntimeToolDefinition>,
    pub(super) extension_configs: Vec<RuntimeExtensionConfig>,
    pub(super) visible_extension_tools: Vec<ExtensionToolInventorySeed>,
    pub(super) searchable_extension_tools: Vec<ExtensionToolInventorySeed>,
}

pub(super) async fn read_agent_tool_inventory_runtime_seed(
    agent_state: &AgentRuntimeState,
) -> Option<AgentToolInventoryRuntimeSeed> {
    if !agent_state.is_initialized().await {
        return None;
    }
    let current_tool_definitions = agent_state.native_tool_definitions_snapshot().await;
    Some(AgentToolInventoryRuntimeSeed {
        current_tool_definitions,
        // Tool inventory is management/configuration projection. It must not
        // borrow a live Session-owned MCP runtime just to render a dashboard.
        extension_configs: Vec::new(),
        visible_extension_tools: Vec::new(),
        searchable_extension_tools: Vec::new(),
    })
}
