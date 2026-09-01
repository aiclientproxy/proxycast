use crate::agent_tools::catalog::{
    tool_catalog_entries_for_surface, tool_catalog_entry, workspace_default_allowed_tool_names,
    ToolCapability, ToolCatalogEntry, ToolLifecycle, ToolPermissionPlane, ToolSourceKind,
    ToolSurfaceProfile, WorkspaceToolSurface, LIST_MCP_RESOURCES_TOOL_NAME,
    READ_MCP_RESOURCE_TOOL_NAME,
};
use crate::agent_tools::execution::{
    resolve_tool_execution_policy_resolution, ToolExecutionPolicySource,
    ToolExecutionResolverInput, ToolExecutionRestrictionProfile, ToolExecutionSandboxProfile,
    ToolExecutionWarningPolicy,
};
use crate::agent_tools::native_tool_policy_gate::NativeToolPolicyGate;
use crate::mcp::McpToolDefinition;
use lime_core::config::ToolExecutionPolicyConfig as ConfigToolExecutionPolicyConfig;
use lime_core::tool_calling::{
    extract_tool_surface_metadata, tool_matches_caller, tool_visible_in_context,
};
use lime_mcp::runtime_extension_name;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashSet};
use tool_runtime::tool_definition::RuntimeToolDefinition;
use tool_runtime::tool_extension::RuntimeExtensionConfig;

fn resource_helper_visibility_allowed(tool_name: &str, resources_supported: bool) -> bool {
    !matches!(
        tool_name,
        LIST_MCP_RESOURCES_TOOL_NAME | READ_MCP_RESOURCE_TOOL_NAME
    ) || resources_supported
}

fn catalog_runtime_visibility_allowed(entry: Option<&ToolCatalogEntry>) -> bool {
    entry.is_none_or(|item| item.lifecycle == ToolLifecycle::Current)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeExtensionSourceKind {
    McpBridge,
    RuntimeExtension,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolInventorySurfaceSnapshot {
    pub workbench: bool,
}

impl From<WorkspaceToolSurface> for ToolInventorySurfaceSnapshot {
    fn from(value: WorkspaceToolSurface) -> Self {
        Self {
            workbench: value.workbench,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolInventoryRequestSnapshot {
    pub caller: String,
    pub surface: ToolInventorySurfaceSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolInventoryCounts {
    pub catalog_total: usize,
    pub catalog_current_total: usize,
    pub catalog_compat_total: usize,
    pub catalog_deprecated_total: usize,
    pub default_allowed_total: usize,
    pub runtime_total: usize,
    pub runtime_visible_total: usize,
    pub native_total: usize,
    pub native_visible_total: usize,
    pub native_catalog_unmapped_total: usize,
    pub extension_surface_total: usize,
    pub extension_mcp_bridge_total: usize,
    pub extension_runtime_total: usize,
    pub extension_tool_total: usize,
    pub extension_tool_visible_total: usize,
    pub mcp_server_total: usize,
    pub mcp_tool_total: usize,
    pub mcp_tool_visible_total: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCatalogInventoryEntry {
    pub name: String,
    pub profiles: Vec<ToolSurfaceProfile>,
    pub capabilities: Vec<ToolCapability>,
    pub lifecycle: ToolLifecycle,
    pub source: ToolSourceKind,
    pub permission_plane: ToolPermissionPlane,
    pub workspace_default_allow: bool,
    pub execution_warning_policy: ToolExecutionWarningPolicy,
    pub execution_warning_policy_source: ToolExecutionPolicySource,
    pub execution_restriction_profile: ToolExecutionRestrictionProfile,
    pub execution_restriction_profile_source: ToolExecutionPolicySource,
    pub execution_sandbox_profile: ToolExecutionSandboxProfile,
    pub execution_sandbox_profile_source: ToolExecutionPolicySource,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeToolInventoryEntry {
    pub name: String,
    pub description: String,
    pub catalog_entry_name: Option<String>,
    pub catalog_source: Option<ToolSourceKind>,
    pub catalog_lifecycle: Option<ToolLifecycle>,
    pub catalog_permission_plane: Option<ToolPermissionPlane>,
    pub catalog_workspace_default_allow: Option<bool>,
    pub catalog_execution_warning_policy: Option<ToolExecutionWarningPolicy>,
    pub catalog_execution_warning_policy_source: Option<ToolExecutionPolicySource>,
    pub catalog_execution_restriction_profile: Option<ToolExecutionRestrictionProfile>,
    pub catalog_execution_restriction_profile_source: Option<ToolExecutionPolicySource>,
    pub catalog_execution_sandbox_profile: Option<ToolExecutionSandboxProfile>,
    pub catalog_execution_sandbox_profile_source: Option<ToolExecutionPolicySource>,
    pub deferred_loading: bool,
    pub always_visible: bool,
    pub allowed_callers: Vec<String>,
    pub tags: Vec<String>,
    pub input_examples_count: usize,
    pub has_output_schema: bool,
    pub caller_allowed: bool,
    pub visible_in_context: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeToolSourceKind {
    CurrentSurface,
    RuntimeExtension,
    Mcp,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeToolInventoryEntry {
    pub name: String,
    pub description: String,
    pub source_kind: RuntimeToolSourceKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    pub catalog_entry_name: Option<String>,
    pub catalog_source: Option<ToolSourceKind>,
    pub catalog_lifecycle: Option<ToolLifecycle>,
    pub catalog_permission_plane: Option<ToolPermissionPlane>,
    pub catalog_workspace_default_allow: Option<bool>,
    pub deferred_loading: bool,
    pub always_visible: bool,
    pub allowed_callers: Vec<String>,
    pub tags: Vec<String>,
    pub input_examples_count: usize,
    pub has_output_schema: bool,
    pub caller_allowed: bool,
    pub visible_in_context: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeExtensionSurfaceInventoryEntry {
    pub extension_name: String,
    pub description: String,
    pub source_kind: RuntimeExtensionSourceKind,
    pub deferred_loading: bool,
    pub allowed_caller: Option<String>,
    pub available_tools: Vec<String>,
    pub always_expose_tools: Vec<String>,
    pub loaded_tools: Vec<String>,
    pub searchable_tools: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeExtensionToolInventoryEntry {
    pub name: String,
    pub description: String,
    pub extension_name: Option<String>,
    pub source_kind: RuntimeExtensionSourceKind,
    pub deferred_loading: bool,
    pub allowed_caller: Option<String>,
    pub status: String,
    pub caller_allowed: bool,
    pub visible_in_context: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtensionToolInventorySeed {
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtensionToolRuntimeStatus {
    pub status: &'static str,
    pub deferred_loading: bool,
    pub extension_name: Option<String>,
    pub allowed_caller: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpToolInventoryEntry {
    pub server_name: String,
    pub name: String,
    pub description: String,
    pub deferred_loading: bool,
    pub always_visible: bool,
    pub allowed_callers: Vec<String>,
    pub tags: Vec<String>,
    pub input_examples_count: usize,
    pub has_output_schema: bool,
    pub caller_allowed: bool,
    pub visible_in_context: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentToolInventorySnapshot {
    pub request: ToolInventoryRequestSnapshot,
    pub agent_initialized: bool,
    pub warnings: Vec<String>,
    pub mcp_servers: Vec<String>,
    pub default_allowed_tools: Vec<String>,
    pub counts: ToolInventoryCounts,
    pub catalog_tools: Vec<ToolCatalogInventoryEntry>,
    pub native_tools: Vec<NativeToolInventoryEntry>,
    pub runtime_tools: Vec<RuntimeToolInventoryEntry>,
    pub extension_surfaces: Vec<RuntimeExtensionSurfaceInventoryEntry>,
    pub extension_tools: Vec<RuntimeExtensionToolInventoryEntry>,
    pub mcp_tools: Vec<McpToolInventoryEntry>,
}

#[derive(Debug, Clone)]
pub(crate) struct AgentToolInventoryBuildInput {
    pub surface: WorkspaceToolSurface,
    pub caller: String,
    pub agent_initialized: bool,
    pub warnings: Vec<String>,
    pub persisted_execution_policy: Option<ConfigToolExecutionPolicyConfig>,
    pub request_metadata: Option<serde_json::Value>,
    pub mcp_server_names: Vec<String>,
    pub mcp_tools: Vec<McpToolDefinition>,
    pub current_tool_definitions: Vec<RuntimeToolDefinition>,
    pub resource_helpers_supported: bool,
    pub extension_configs: Vec<RuntimeExtensionConfig>,
    pub visible_extension_tools: Vec<ExtensionToolInventorySeed>,
    pub searchable_extension_tools: Vec<ExtensionToolInventorySeed>,
}

pub(crate) fn build_tool_inventory(
    input: AgentToolInventoryBuildInput,
) -> AgentToolInventorySnapshot {
    let AgentToolInventoryBuildInput {
        surface,
        caller,
        agent_initialized,
        warnings,
        persisted_execution_policy,
        request_metadata,
        mcp_server_names,
        mcp_tools,
        current_tool_definitions,
        resource_helpers_supported,
        extension_configs,
        visible_extension_tools,
        searchable_extension_tools,
    } = input;

    let execution_policy_input = ToolExecutionResolverInput {
        persisted_policy: persisted_execution_policy.as_ref(),
        request_metadata: request_metadata.as_ref(),
    };
    let native_tool_policy_gate =
        NativeToolPolicyGate::from_request_metadata(request_metadata.as_ref());
    let update_plan_enabled =
        tool_runtime::update_plan::update_plan_enabled_from_metadata(request_metadata.as_ref());

    let mut mcp_servers = mcp_server_names
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    mcp_servers.sort();
    let mcp_extension_lookup = mcp_servers
        .iter()
        .map(|server_name| runtime_extension_name(server_name))
        .collect::<HashSet<_>>();

    let mut default_allowed_tools = workspace_default_allowed_tool_names(surface)
        .into_iter()
        .filter(|name| native_tool_policy_gate.allows_tool_name(name))
        .filter(|name| *name != tool_runtime::update_plan::UPDATE_PLAN_NAME || update_plan_enabled)
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    default_allowed_tools.sort();

    let catalog_tools = tool_catalog_entries_for_surface(surface)
        .into_iter()
        .filter(|entry| native_tool_policy_gate.allows_catalog_entry(entry))
        .filter(|entry| {
            entry.name != tool_runtime::update_plan::UPDATE_PLAN_NAME || update_plan_enabled
        })
        .map(|entry| {
            let resolution =
                resolve_tool_execution_policy_resolution(entry.name, execution_policy_input);

            ToolCatalogInventoryEntry {
                name: entry.name.to_string(),
                profiles: entry.profiles.to_vec(),
                capabilities: entry.capabilities.to_vec(),
                lifecycle: entry.lifecycle,
                source: entry.source,
                permission_plane: entry.permission_plane,
                workspace_default_allow: entry.workspace_default_allow,
                execution_warning_policy: resolution.policy.warning_policy,
                execution_warning_policy_source: resolution.warning_policy_source,
                execution_restriction_profile: resolution.policy.restriction_profile,
                execution_restriction_profile_source: resolution.restriction_profile_source,
                execution_sandbox_profile: resolution.policy.sandbox_profile,
                execution_sandbox_profile_source: resolution.sandbox_profile_source,
            }
        })
        .collect::<Vec<_>>();

    let current_tool_definitions = current_tool_definitions
        .into_iter()
        .filter(|definition| {
            definition.name != tool_runtime::update_plan::UPDATE_PLAN_NAME || update_plan_enabled
        })
        .collect::<Vec<_>>();
    let native_tools = build_native_inventory(
        &current_tool_definitions,
        &caller,
        execution_policy_input,
        resource_helpers_supported,
        &native_tool_policy_gate,
    );
    let extension_surfaces = build_extension_surface_inventory(
        &extension_configs,
        &visible_extension_tools,
        &searchable_extension_tools,
        &mcp_extension_lookup,
    );
    let extension_tools = build_extension_tool_inventory(
        &extension_configs,
        &visible_extension_tools,
        &searchable_extension_tools,
        &caller,
        &mcp_extension_lookup,
    );
    let mcp_tools = build_mcp_inventory(&mcp_tools, &caller);
    let runtime_tools = build_runtime_tool_inventory(
        &native_tools,
        &extension_tools,
        &mcp_tools,
        &native_tool_policy_gate,
    );

    let counts = ToolInventoryCounts {
        catalog_total: catalog_tools.len(),
        catalog_current_total: catalog_tools
            .iter()
            .filter(|entry| entry.lifecycle == ToolLifecycle::Current)
            .count(),
        catalog_compat_total: catalog_tools
            .iter()
            .filter(|entry| entry.lifecycle == ToolLifecycle::Compat)
            .count(),
        catalog_deprecated_total: catalog_tools
            .iter()
            .filter(|entry| entry.lifecycle == ToolLifecycle::Deprecated)
            .count(),
        default_allowed_total: default_allowed_tools.len(),
        runtime_total: runtime_tools.len(),
        runtime_visible_total: runtime_tools
            .iter()
            .filter(|entry| entry.visible_in_context)
            .count(),
        native_total: native_tools.len(),
        native_visible_total: native_tools
            .iter()
            .filter(|entry| entry.visible_in_context)
            .count(),
        native_catalog_unmapped_total: native_tools
            .iter()
            .filter(|entry| entry.catalog_entry_name.is_none())
            .count(),
        extension_surface_total: extension_surfaces.len(),
        extension_mcp_bridge_total: extension_surfaces
            .iter()
            .filter(|entry| entry.source_kind == RuntimeExtensionSourceKind::McpBridge)
            .count(),
        extension_runtime_total: extension_surfaces
            .iter()
            .filter(|entry| entry.source_kind == RuntimeExtensionSourceKind::RuntimeExtension)
            .count(),
        extension_tool_total: extension_tools.len(),
        extension_tool_visible_total: extension_tools
            .iter()
            .filter(|entry| entry.visible_in_context)
            .count(),
        mcp_server_total: mcp_servers.len(),
        mcp_tool_total: mcp_tools.len(),
        mcp_tool_visible_total: mcp_tools
            .iter()
            .filter(|entry| entry.visible_in_context)
            .count(),
    };

    AgentToolInventorySnapshot {
        request: ToolInventoryRequestSnapshot {
            caller,
            surface: surface.into(),
        },
        agent_initialized,
        warnings,
        mcp_servers,
        default_allowed_tools,
        counts,
        catalog_tools,
        native_tools,
        runtime_tools,
        extension_surfaces,
        extension_tools,
        mcp_tools,
    }
}

fn build_native_inventory(
    definitions: &[RuntimeToolDefinition],
    caller: &str,
    execution_policy_input: ToolExecutionResolverInput<'_>,
    resource_helpers_supported: bool,
    native_tool_policy_gate: &NativeToolPolicyGate,
) -> Vec<NativeToolInventoryEntry> {
    let mut result = definitions
        .iter()
        .filter(|definition| native_tool_policy_gate.allows_tool_name(&definition.name))
        .map(|definition| {
            let metadata =
                extract_tool_surface_metadata(&definition.name, &definition.input_schema);
            let catalog_entry = tool_catalog_entry(&definition.name);
            let caller_allowed = tool_matches_caller(&metadata, Some(caller));
            let visible_in_context = caller_allowed
                && resource_helper_visibility_allowed(&definition.name, resource_helpers_supported)
                && catalog_runtime_visibility_allowed(catalog_entry)
                && tool_visible_in_context(&metadata, false);
            let catalog_execution_policy = catalog_entry.map(|entry| {
                resolve_tool_execution_policy_resolution(entry.name, execution_policy_input)
            });

            NativeToolInventoryEntry {
                name: definition.name.clone(),
                description: definition.description.clone(),
                catalog_entry_name: catalog_entry.map(|entry| entry.name.to_string()),
                catalog_source: catalog_entry.map(|entry| entry.source),
                catalog_lifecycle: catalog_entry.map(|entry| entry.lifecycle),
                catalog_permission_plane: catalog_entry.map(|entry| entry.permission_plane),
                catalog_workspace_default_allow: catalog_entry
                    .map(|entry| entry.workspace_default_allow),
                catalog_execution_warning_policy: catalog_execution_policy
                    .map(|resolution| resolution.policy.warning_policy),
                catalog_execution_warning_policy_source: catalog_execution_policy
                    .map(|resolution| resolution.warning_policy_source),
                catalog_execution_restriction_profile: catalog_execution_policy
                    .map(|resolution| resolution.policy.restriction_profile),
                catalog_execution_restriction_profile_source: catalog_execution_policy
                    .map(|resolution| resolution.restriction_profile_source),
                catalog_execution_sandbox_profile: catalog_execution_policy
                    .map(|resolution| resolution.policy.sandbox_profile),
                catalog_execution_sandbox_profile_source: catalog_execution_policy
                    .map(|resolution| resolution.sandbox_profile_source),
                deferred_loading: metadata.deferred_loading.unwrap_or(false),
                always_visible: metadata.always_visible.unwrap_or(false),
                allowed_callers: metadata.allowed_callers.unwrap_or_default(),
                tags: metadata.tags.unwrap_or_default(),
                input_examples_count: metadata.input_examples.len(),
                has_output_schema: false,
                caller_allowed,
                visible_in_context,
            }
        })
        .collect::<Vec<_>>();

    result.sort_by(|left, right| left.name.cmp(&right.name));
    result
}

fn build_extension_surface_inventory(
    configs: &[RuntimeExtensionConfig],
    visible_extension_tools: &[ExtensionToolInventorySeed],
    searchable_extension_tools: &[ExtensionToolInventorySeed],
    mcp_server_lookup: &HashSet<String>,
) -> Vec<RuntimeExtensionSurfaceInventoryEntry> {
    let loaded_tool_names = visible_extension_tools
        .iter()
        .map(|tool| tool.name.clone())
        .collect::<HashSet<_>>();
    let searchable_tool_names = searchable_extension_tools
        .iter()
        .map(|tool| tool.name.clone())
        .collect::<HashSet<_>>();

    let mut result = configs
        .iter()
        .map(|config| {
            let extension_name = config.name.clone();
            let available_tools = extension_available_tools(config);
            let always_expose_tools = extension_always_expose_tools(config);
            let loaded_tools = prefixed_tool_names(
                &extension_name,
                loaded_tool_names.iter().map(String::as_str),
            );
            let searchable_tools = prefixed_tool_names(
                &extension_name,
                searchable_tool_names.iter().map(String::as_str),
            );

            RuntimeExtensionSurfaceInventoryEntry {
                extension_name: extension_name.clone(),
                description: extension_description(config),
                source_kind: extension_source_kind(&extension_name, mcp_server_lookup),
                deferred_loading: config.deferred_loading,
                allowed_caller: config.allowed_caller.clone(),
                available_tools,
                always_expose_tools,
                loaded_tools,
                searchable_tools,
            }
        })
        .collect::<Vec<_>>();

    result.sort_by(|left, right| left.extension_name.cmp(&right.extension_name));
    result
}

fn build_extension_tool_inventory(
    configs: &[RuntimeExtensionConfig],
    visible_extension_tools: &[ExtensionToolInventorySeed],
    searchable_extension_tools: &[ExtensionToolInventorySeed],
    caller: &str,
    mcp_server_lookup: &HashSet<String>,
) -> Vec<RuntimeExtensionToolInventoryEntry> {
    let visible_tool_names = visible_extension_tools
        .iter()
        .map(|tool| tool.name.clone())
        .collect::<HashSet<_>>();

    let mut result = searchable_extension_tools
        .iter()
        .map(|tool| {
            let runtime_status =
                resolve_extension_tool_runtime_status(configs, &visible_tool_names, &tool.name);
            let source_kind = runtime_status
                .extension_name
                .as_ref()
                .map(|name| extension_source_kind(name, mcp_server_lookup))
                .unwrap_or(RuntimeExtensionSourceKind::RuntimeExtension);
            let caller_allowed = runtime_status
                .allowed_caller
                .as_deref()
                .is_none_or(|value| value == caller);

            RuntimeExtensionToolInventoryEntry {
                name: tool.name.clone(),
                description: tool.description.clone(),
                extension_name: runtime_status.extension_name.clone(),
                source_kind,
                deferred_loading: runtime_status.deferred_loading,
                allowed_caller: runtime_status.allowed_caller.clone(),
                status: runtime_status.status.to_string(),
                caller_allowed,
                visible_in_context: caller_allowed && runtime_status.status != "deferred",
            }
        })
        .collect::<Vec<_>>();

    result.sort_by(|left, right| left.name.cmp(&right.name));
    result
}

fn build_mcp_inventory(tools: &[McpToolDefinition], caller: &str) -> Vec<McpToolInventoryEntry> {
    let mut result = tools
        .iter()
        .map(|tool| {
            let metadata = extract_tool_surface_metadata(&tool.name, &tool.input_schema);
            let caller_allowed = tool_matches_caller(&metadata, Some(caller));
            let visible_in_context = caller_allowed && tool_visible_in_context(&metadata, false);

            McpToolInventoryEntry {
                server_name: tool.server_name.clone(),
                name: tool.name.clone(),
                description: tool.description.clone(),
                deferred_loading: metadata.deferred_loading.unwrap_or(false),
                always_visible: metadata.always_visible.unwrap_or(false),
                allowed_callers: metadata.allowed_callers.unwrap_or_default(),
                tags: metadata.tags.unwrap_or_default(),
                input_examples_count: metadata.input_examples.len(),
                has_output_schema: tool.output_schema.is_some(),
                caller_allowed,
                visible_in_context,
            }
        })
        .collect::<Vec<_>>();

    result.sort_by(|left, right| {
        left.server_name
            .cmp(&right.server_name)
            .then_with(|| left.name.cmp(&right.name))
    });
    result
}

fn runtime_tool_source_rank(kind: RuntimeToolSourceKind) -> u8 {
    match kind {
        RuntimeToolSourceKind::CurrentSurface => 0,
        RuntimeToolSourceKind::RuntimeExtension => 1,
        RuntimeToolSourceKind::Mcp => 2,
    }
}

fn insert_runtime_tool_entry(
    entries: &mut Vec<RuntimeToolInventoryEntry>,
    entry: RuntimeToolInventoryEntry,
) {
    if entries
        .iter()
        .any(|existing| existing.name.eq_ignore_ascii_case(&entry.name))
    {
        return;
    }

    entries.push(entry);
}

fn build_runtime_tool_inventory(
    native_tools: &[NativeToolInventoryEntry],
    extension_tools: &[RuntimeExtensionToolInventoryEntry],
    mcp_tools: &[McpToolInventoryEntry],
    native_tool_policy_gate: &NativeToolPolicyGate,
) -> Vec<RuntimeToolInventoryEntry> {
    let mut result = Vec::new();
    let mcp_output_schema_tool_names = mcp_tools
        .iter()
        .filter(|entry| entry.has_output_schema)
        .map(|entry| entry.name.clone())
        .collect::<HashSet<_>>();

    for entry in native_tools {
        if !native_tool_policy_gate.allows_tool_name(&entry.name) {
            continue;
        }
        insert_runtime_tool_entry(
            &mut result,
            RuntimeToolInventoryEntry {
                name: entry.name.clone(),
                description: entry.description.clone(),
                source_kind: RuntimeToolSourceKind::CurrentSurface,
                source_label: None,
                status: None,
                catalog_entry_name: entry.catalog_entry_name.clone(),
                catalog_source: entry.catalog_source,
                catalog_lifecycle: entry.catalog_lifecycle,
                catalog_permission_plane: entry.catalog_permission_plane,
                catalog_workspace_default_allow: entry.catalog_workspace_default_allow,
                deferred_loading: entry.deferred_loading,
                always_visible: entry.always_visible,
                allowed_callers: entry.allowed_callers.clone(),
                tags: entry.tags.clone(),
                input_examples_count: entry.input_examples_count,
                has_output_schema: entry.has_output_schema,
                caller_allowed: entry.caller_allowed,
                visible_in_context: entry.visible_in_context,
            },
        );
    }

    for entry in extension_tools {
        if !native_tool_policy_gate.allows_tool_name(&entry.name) {
            continue;
        }
        let catalog_entry = tool_catalog_entry(&entry.name);
        let source_kind = entry.source_kind;
        insert_runtime_tool_entry(
            &mut result,
            RuntimeToolInventoryEntry {
                name: entry.name.clone(),
                description: entry.description.clone(),
                source_kind: RuntimeToolSourceKind::RuntimeExtension,
                source_label: entry.extension_name.clone(),
                status: Some(entry.status.clone()),
                catalog_entry_name: catalog_entry.map(|item| item.name.to_string()),
                catalog_source: catalog_entry.map(|item| item.source),
                catalog_lifecycle: catalog_entry.map(|item| item.lifecycle),
                catalog_permission_plane: catalog_entry.map(|item| item.permission_plane),
                catalog_workspace_default_allow: catalog_entry
                    .map(|item| item.workspace_default_allow),
                deferred_loading: entry.deferred_loading,
                always_visible: false,
                allowed_callers: entry.allowed_caller.clone().into_iter().collect::<Vec<_>>(),
                tags: Vec::new(),
                input_examples_count: 0,
                has_output_schema: source_kind == RuntimeExtensionSourceKind::McpBridge
                    && mcp_output_schema_tool_names.contains(&entry.name),
                caller_allowed: entry.caller_allowed,
                visible_in_context: entry.visible_in_context,
            },
        );
    }

    for entry in mcp_tools {
        if !native_tool_policy_gate.allows_tool_name(&entry.name) {
            continue;
        }
        let catalog_entry = tool_catalog_entry(&entry.name);
        insert_runtime_tool_entry(
            &mut result,
            RuntimeToolInventoryEntry {
                name: entry.name.clone(),
                description: entry.description.clone(),
                source_kind: RuntimeToolSourceKind::Mcp,
                source_label: Some(entry.server_name.clone()),
                status: None,
                catalog_entry_name: catalog_entry.map(|item| item.name.to_string()),
                catalog_source: catalog_entry.map(|item| item.source),
                catalog_lifecycle: catalog_entry.map(|item| item.lifecycle),
                catalog_permission_plane: catalog_entry.map(|item| item.permission_plane),
                catalog_workspace_default_allow: catalog_entry
                    .map(|item| item.workspace_default_allow),
                deferred_loading: entry.deferred_loading,
                always_visible: entry.always_visible,
                allowed_callers: entry.allowed_callers.clone(),
                tags: entry.tags.clone(),
                input_examples_count: entry.input_examples_count,
                has_output_schema: entry.has_output_schema,
                caller_allowed: entry.caller_allowed,
                visible_in_context: entry.visible_in_context,
            },
        );
    }

    result.sort_by(|left, right| {
        runtime_tool_source_rank(left.source_kind)
            .cmp(&runtime_tool_source_rank(right.source_kind))
            .then_with(|| right.visible_in_context.cmp(&left.visible_in_context))
            .then_with(|| left.source_label.cmp(&right.source_label))
            .then_with(|| left.name.cmp(&right.name))
    });
    result
}

fn extension_available_tools(config: &RuntimeExtensionConfig) -> Vec<String> {
    let mut tools = config.available_tools.clone();
    tools.sort();
    tools.dedup();
    tools
}

fn extension_always_expose_tools(config: &RuntimeExtensionConfig) -> Vec<String> {
    let mut tools = config.always_expose_tools.clone();
    tools.sort();
    tools.dedup();
    tools
}

fn extension_description(config: &RuntimeExtensionConfig) -> String {
    config.description.clone()
}

fn extension_source_kind(
    extension_name: &str,
    mcp_server_lookup: &HashSet<String>,
) -> RuntimeExtensionSourceKind {
    if mcp_server_lookup.contains(extension_name) {
        RuntimeExtensionSourceKind::McpBridge
    } else {
        RuntimeExtensionSourceKind::RuntimeExtension
    }
}

fn prefixed_tool_names<'a>(
    extension_name: &str,
    tool_names: impl Iterator<Item = &'a str>,
) -> Vec<String> {
    let prefix = format!("{extension_name}__");
    let mut names = tool_names
        .filter(|name| name.starts_with(&prefix))
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    names
}

pub fn resolve_extension_tool_runtime_status(
    configs: &[RuntimeExtensionConfig],
    visible_tool_names: &HashSet<String>,
    tool_name: &str,
) -> ExtensionToolRuntimeStatus {
    let matched = configs
        .iter()
        .filter_map(|config| {
            let extension_name = config.name.clone();
            tool_name
                .strip_prefix(extension_name.as_str())
                .and_then(|rest| rest.strip_prefix("__"))
                .map(|inner_tool_name| (extension_name, config, inner_tool_name.to_string()))
        })
        .max_by_key(|(extension_name, _, _)| extension_name.len());

    let Some((extension_name, config, inner_tool_name)) = matched else {
        return ExtensionToolRuntimeStatus {
            status: "visible",
            deferred_loading: false,
            extension_name: None,
            allowed_caller: None,
        };
    };

    if !config.deferred_loading || config.is_tool_exposed_by_default(&inner_tool_name) {
        return ExtensionToolRuntimeStatus {
            status: "visible",
            deferred_loading: false,
            extension_name: Some(extension_name),
            allowed_caller: config.allowed_caller.clone(),
        };
    }

    if visible_tool_names.contains(tool_name) {
        ExtensionToolRuntimeStatus {
            status: "loaded",
            deferred_loading: false,
            extension_name: Some(extension_name),
            allowed_caller: config.allowed_caller.clone(),
        }
    } else {
        ExtensionToolRuntimeStatus {
            status: "deferred",
            deferred_loading: true,
            extension_name: Some(extension_name),
            allowed_caller: config.allowed_caller.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_tools::catalog::{
        APPLY_PATCH_TOOL_NAME, MEMORY_ADD_NOTE_TOOL_NAME, MEMORY_LIST_TOOL_NAME,
        MEMORY_READ_TOOL_NAME, MEMORY_SEARCH_TOOL_NAME, SKILL_SEARCH_TOOL_NAME,
        TOOL_SEARCH_TOOL_NAME,
    };
    use lime_core::config::{
        ToolExecutionOverrideConfig as ConfigToolExecutionOverrideConfig,
        ToolExecutionPolicyConfig as ConfigToolExecutionPolicyConfig,
        ToolExecutionWarningPolicyConfig as ConfigToolExecutionWarningPolicyConfig,
    };
    use serde_json::json;

    fn builtin_extension(
        name: &str,
        available_tools: Vec<&str>,
        deferred_loading: bool,
        always_expose_tools: Vec<&str>,
        allowed_caller: Option<&str>,
    ) -> RuntimeExtensionConfig {
        RuntimeExtensionConfig::new(
            name,
            format!("{name} tools"),
            available_tools
                .into_iter()
                .map(|item| item.to_string())
                .collect(),
            deferred_loading,
            always_expose_tools
                .into_iter()
                .map(|item| item.to_string())
                .collect(),
            allowed_caller.map(ToString::to_string),
        )
    }

    fn definition(
        name: &str,
        description: &str,
        schema: serde_json::Value,
    ) -> RuntimeToolDefinition {
        RuntimeToolDefinition::new(name, description, schema)
    }

    fn seed(name: &str, description: &str) -> ExtensionToolInventorySeed {
        ExtensionToolInventorySeed {
            name: name.to_string(),
            description: description.to_string(),
        }
    }

    fn mcp_tool(
        server_name: &str,
        name: &str,
        deferred_loading: bool,
        always_visible: bool,
        allowed_callers: Vec<&str>,
    ) -> McpToolDefinition {
        McpToolDefinition {
            server_name: server_name.to_string(),
            name: name.to_string(),
            description: format!("{name} desc"),
            input_schema: json!({
                "type": "object",
                "x-lime": {
                    "deferred_loading": deferred_loading,
                    "always_visible": always_visible,
                    "allowed_callers": allowed_callers
                }
            }),
            output_schema: None,
            deferred_loading: Some(deferred_loading),
            always_visible: Some(always_visible),
            allowed_callers: Some(
                allowed_callers
                    .into_iter()
                    .map(|item| item.to_string())
                    .collect(),
            ),
            input_examples: None,
            tags: None,
        }
    }

    fn mcp_tool_with_output_schema(
        server_name: &str,
        name: &str,
        deferred_loading: bool,
        always_visible: bool,
        allowed_callers: Vec<&str>,
    ) -> McpToolDefinition {
        let mut tool = mcp_tool(
            server_name,
            name,
            deferred_loading,
            always_visible,
            allowed_callers,
        );
        tool.output_schema = Some(json!({
            "type": "object",
            "properties": {
                "content": { "type": "array" },
                "structuredContent": {
                    "type": "object",
                    "properties": {
                        "answer": { "type": "string" }
                    }
                }
            },
            "required": ["content"]
        }));
        tool
    }

    #[test]
    fn test_build_tool_inventory_marks_visibility_and_mappings() {
        let expected_catalog = tool_catalog_entries_for_surface(WorkspaceToolSurface::core());
        let inventory = build_tool_inventory(AgentToolInventoryBuildInput {
            surface: WorkspaceToolSurface::core(),
            caller: "assistant".to_string(),
            agent_initialized: true,
            warnings: Vec::new(),
            persisted_execution_policy: None,
            request_metadata: None,
            mcp_server_names: vec!["docs".to_string()],
            mcp_tools: vec![mcp_tool(
                "docs",
                "search_docs",
                true,
                false,
                vec!["assistant"],
            )],
            current_tool_definitions: vec![
                definition(
                    TOOL_SEARCH_TOOL_NAME,
                    "search tools",
                    json!({ "type": "object" }),
                ),
                definition(
                    "Read",
                    "read file",
                    json!({
                        "type": "object",
                        "x-lime": { "allowed_callers": ["assistant"] }
                    }),
                ),
                definition(
                    "StructuredOutput",
                    "structured response",
                    json!({ "type": "object" }),
                ),
                definition(
                    "admin_secret",
                    "secret",
                    json!({
                        "type": "object",
                        "x-lime": {
                            "deferred_loading": true,
                            "allowed_callers": ["code_execution"]
                        }
                    }),
                ),
            ],
            resource_helpers_supported: false,
            extension_configs: vec![builtin_extension(
                "mcp__docs",
                vec!["search_docs", "read_docs"],
                true,
                vec!["search_docs"],
                Some("assistant"),
            )],
            visible_extension_tools: vec![ExtensionToolInventorySeed {
                name: "mcp__docs__search_docs".to_string(),
                description: "visible docs tool".to_string(),
            }],
            searchable_extension_tools: vec![
                ExtensionToolInventorySeed {
                    name: "mcp__docs__search_docs".to_string(),
                    description: "visible docs tool".to_string(),
                },
                ExtensionToolInventorySeed {
                    name: "mcp__docs__read_docs".to_string(),
                    description: "deferred docs tool".to_string(),
                },
            ],
        });

        assert_eq!(inventory.counts.catalog_total, expected_catalog.len());
        assert_eq!(
            inventory.counts.catalog_current_total,
            expected_catalog
                .iter()
                .filter(|entry| entry.lifecycle == ToolLifecycle::Current)
                .count()
        );
        assert_eq!(
            inventory.counts.catalog_compat_total,
            expected_catalog
                .iter()
                .filter(|entry| entry.lifecycle == ToolLifecycle::Compat)
                .count()
        );
        assert_eq!(inventory.counts.native_total, 4);
        assert_eq!(inventory.counts.native_visible_total, 3);
        assert_eq!(inventory.counts.native_catalog_unmapped_total, 1);
        assert_eq!(inventory.counts.extension_surface_total, 1);
        assert_eq!(inventory.counts.extension_mcp_bridge_total, 1);
        assert_eq!(inventory.counts.extension_tool_total, 2);
        assert_eq!(inventory.counts.extension_tool_visible_total, 1);
        assert_eq!(inventory.counts.mcp_tool_total, 1);
        assert_eq!(inventory.counts.mcp_tool_visible_total, 0);
        assert!(inventory
            .default_allowed_tools
            .contains(&TOOL_SEARCH_TOOL_NAME.to_string()));
        assert!(inventory
            .default_allowed_tools
            .contains(&SKILL_SEARCH_TOOL_NAME.to_string()));
        for memory_tool_name in [
            MEMORY_LIST_TOOL_NAME,
            MEMORY_READ_TOOL_NAME,
            MEMORY_SEARCH_TOOL_NAME,
            MEMORY_ADD_NOTE_TOOL_NAME,
        ] {
            let memory_catalog = inventory
                .catalog_tools
                .iter()
                .find(|entry| entry.name == memory_tool_name)
                .expect("memory catalog entry should exist");
            assert_eq!(memory_catalog.lifecycle, ToolLifecycle::Current);
            assert_eq!(memory_catalog.capabilities, vec![ToolCapability::Memory]);
        }
        let exec_catalog = inventory
            .catalog_tools
            .iter()
            .find(|entry| entry.name == "exec_command")
            .expect("exec_command catalog entry should exist");
        assert_eq!(
            exec_catalog.execution_warning_policy,
            ToolExecutionWarningPolicy::ShellCommandRisk
        );
        assert_eq!(
            exec_catalog.execution_sandbox_profile,
            ToolExecutionSandboxProfile::WorkspaceCommand
        );

        let admin_tool = inventory
            .native_tools
            .iter()
            .find(|entry| entry.name == "admin_secret")
            .expect("admin tool should exist");
        assert!(!admin_tool.caller_allowed);
        assert!(!admin_tool.visible_in_context);
        assert!(admin_tool.catalog_entry_name.is_none());

        let structured_output_tool = inventory
            .native_tools
            .iter()
            .find(|entry| entry.name == "StructuredOutput")
            .expect("StructuredOutput should exist");
        assert_eq!(
            structured_output_tool.catalog_entry_name.as_deref(),
            Some("StructuredOutput")
        );
        assert_eq!(
            structured_output_tool.catalog_source,
            Some(ToolSourceKind::RuntimeBuiltin)
        );
        assert_eq!(
            structured_output_tool.catalog_lifecycle,
            Some(ToolLifecycle::Current)
        );

        let docs_surface = inventory
            .extension_surfaces
            .iter()
            .find(|entry| entry.extension_name == "mcp__docs")
            .expect("docs surface should exist");
        assert_eq!(
            docs_surface.source_kind,
            RuntimeExtensionSourceKind::McpBridge
        );
        assert_eq!(
            docs_surface.loaded_tools,
            vec!["mcp__docs__search_docs".to_string()]
        );
        assert_eq!(
            docs_surface.searchable_tools,
            vec![
                "mcp__docs__read_docs".to_string(),
                "mcp__docs__search_docs".to_string()
            ]
        );

        let deferred_extension_tool = inventory
            .extension_tools
            .iter()
            .find(|entry| entry.name == "mcp__docs__read_docs")
            .expect("deferred extension tool should exist");
        assert_eq!(deferred_extension_tool.status, "deferred");
        assert!(!deferred_extension_tool.visible_in_context);
    }

    #[test]
    fn test_build_tool_inventory_defaults_unknown_caller_normalization_upstream() {
        let caller = lime_core::tool_calling::normalize_tool_caller(Some(" Assistant "))
            .expect("caller should normalize upstream");
        assert_eq!(caller, "assistant");
    }

    #[test]
    fn test_build_tool_inventory_runtime_tools_dedup_and_marks_current_surface() {
        let inventory = build_tool_inventory(AgentToolInventoryBuildInput {
            surface: WorkspaceToolSurface::core(),
            caller: "assistant".to_string(),
            agent_initialized: true,
            warnings: Vec::new(),
            persisted_execution_policy: None,
            request_metadata: None,
            mcp_server_names: vec!["docs".to_string()],
            mcp_tools: vec![mcp_tool_with_output_schema(
                "docs",
                "mcp__docs__search_docs",
                true,
                false,
                vec!["assistant"],
            )],
            current_tool_definitions: vec![
                definition("Agent", "delegate work", json!({ "type": "object" })),
                definition(
                    "exec_command",
                    "workspace command",
                    json!({ "type": "object" }),
                ),
            ],
            resource_helpers_supported: false,
            extension_configs: vec![builtin_extension(
                "mcp__docs",
                vec!["search_docs"],
                true,
                vec!["search_docs"],
                Some("assistant"),
            )],
            visible_extension_tools: vec![seed("mcp__docs__search_docs", "search docs")],
            searchable_extension_tools: vec![seed("mcp__docs__search_docs", "search docs")],
        });

        assert_eq!(inventory.counts.runtime_total, 3);
        assert_eq!(inventory.counts.runtime_visible_total, 3);

        let agent_tool = inventory
            .runtime_tools
            .iter()
            .find(|entry| entry.name == "Agent")
            .expect("Agent runtime tool should exist");
        assert_eq!(
            agent_tool.source_kind,
            RuntimeToolSourceKind::CurrentSurface
        );
        assert!(agent_tool.visible_in_context);

        let exec_tool = inventory
            .runtime_tools
            .iter()
            .find(|entry| entry.name == "exec_command")
            .expect("exec_command runtime tool should exist");
        assert_eq!(exec_tool.source_kind, RuntimeToolSourceKind::CurrentSurface);

        let docs_tool = inventory
            .runtime_tools
            .iter()
            .find(|entry| entry.name == "mcp__docs__search_docs")
            .expect("docs runtime tool should exist");
        assert_eq!(
            docs_tool.source_kind,
            RuntimeToolSourceKind::RuntimeExtension
        );
        assert_eq!(docs_tool.source_label.as_deref(), Some("mcp__docs"));
        assert_eq!(docs_tool.status.as_deref(), Some("visible"));
        assert!(docs_tool.has_output_schema);
        let docs_mcp_tool = inventory
            .mcp_tools
            .iter()
            .find(|entry| entry.name == "mcp__docs__search_docs")
            .expect("docs MCP tool should exist");
        assert!(docs_mcp_tool.has_output_schema);
        assert_eq!(
            inventory
                .runtime_tools
                .iter()
                .filter(|entry| entry.name == "mcp__docs__search_docs")
                .count(),
            1
        );
    }

    #[test]
    fn test_build_tool_inventory_gates_native_shell_and_apply_patch_by_request_policy() {
        let inventory = build_tool_inventory(AgentToolInventoryBuildInput {
            surface: WorkspaceToolSurface::core(),
            caller: "assistant".to_string(),
            agent_initialized: true,
            warnings: Vec::new(),
            persisted_execution_policy: None,
            request_metadata: Some(json!({
                "harness": {
                    "model_request_policy": {
                        "native_tool_policy": {
                            "shell_type": "disabled",
                            "shell_tool_enabled": false,
                            "apply_patch_tool_enabled": false
                        }
                    }
                }
            })),
            mcp_server_names: Vec::new(),
            mcp_tools: Vec::new(),
            current_tool_definitions: vec![
                definition("Read", "read file", json!({ "type": "object" })),
                definition(
                    "exec_command",
                    "workspace command",
                    json!({ "type": "object" }),
                ),
                definition(
                    "write_stdin",
                    "workspace powershell",
                    json!({ "type": "object" }),
                ),
                definition(
                    "apply_patch",
                    "workspace patch",
                    json!({ "type": "object" }),
                ),
            ],
            resource_helpers_supported: false,
            extension_configs: Vec::new(),
            visible_extension_tools: Vec::new(),
            searchable_extension_tools: Vec::new(),
        });

        for blocked_tool_name in ["exec_command", "write_stdin", APPLY_PATCH_TOOL_NAME] {
            assert!(
                !inventory
                    .catalog_tools
                    .iter()
                    .any(|entry| entry.name == blocked_tool_name),
                "{blocked_tool_name} should be hidden from catalog inventory"
            );
            assert!(
                !inventory
                    .native_tools
                    .iter()
                    .any(|entry| entry.name == blocked_tool_name),
                "{blocked_tool_name} should be hidden from native inventory"
            );
            assert!(
                !inventory
                    .runtime_tools
                    .iter()
                    .any(|entry| entry.name == blocked_tool_name),
                "{blocked_tool_name} should be hidden from runtime inventory"
            );
        }

        assert!(inventory
            .catalog_tools
            .iter()
            .any(|entry| entry.name == "Read"));
        assert!(inventory
            .native_tools
            .iter()
            .any(|entry| entry.name == "Read"));
        assert!(inventory
            .runtime_tools
            .iter()
            .any(|entry| entry.name == "Read"));
    }

    #[test]
    fn test_build_tool_inventory_hides_deprecated_catalog_native_tools() {
        let inventory = build_tool_inventory(AgentToolInventoryBuildInput {
            surface: WorkspaceToolSurface::workbench(),
            caller: "assistant".to_string(),
            agent_initialized: true,
            warnings: Vec::new(),
            persisted_execution_policy: None,
            request_metadata: None,
            mcp_server_names: Vec::new(),
            mcp_tools: Vec::new(),
            current_tool_definitions: vec![definition(
                "lime_create_video_generation_task",
                "legacy video task facade",
                json!({ "type": "object" }),
            )],
            resource_helpers_supported: false,
            extension_configs: Vec::new(),
            visible_extension_tools: Vec::new(),
            searchable_extension_tools: Vec::new(),
        });

        assert!(!inventory
            .default_allowed_tools
            .contains(&"lime_create_video_generation_task".to_string()));

        let native_tool = inventory
            .native_tools
            .iter()
            .find(|entry| entry.name == "lime_create_video_generation_task")
            .expect("deprecated video tool should stay inventoried for governance");
        assert_eq!(
            native_tool.catalog_lifecycle,
            Some(ToolLifecycle::Deprecated)
        );
        assert!(!native_tool.visible_in_context);

        let runtime_tool = inventory
            .runtime_tools
            .iter()
            .find(|entry| entry.name == "lime_create_video_generation_task")
            .expect("deprecated video runtime tool should stay inventoried");
        assert_eq!(
            runtime_tool.catalog_lifecycle,
            Some(ToolLifecycle::Deprecated)
        );
        assert!(!runtime_tool.visible_in_context);
    }

    #[test]
    fn test_build_tool_inventory_uninitialized_agent_keeps_sorted_servers_and_mcp_visibility() {
        let inventory = build_tool_inventory(AgentToolInventoryBuildInput {
            surface: WorkspaceToolSurface::core(),
            caller: "assistant".to_string(),
            agent_initialized: false,
            warnings: vec!["agent not initialized".to_string()],
            persisted_execution_policy: None,
            request_metadata: None,
            mcp_server_names: vec!["docs".to_string(), "alpha".to_string(), "docs".to_string()],
            mcp_tools: vec![
                mcp_tool("docs", "search_docs", true, true, vec!["assistant"]),
                mcp_tool("alpha", "read_alpha", false, false, vec![]),
            ],
            current_tool_definitions: Vec::new(),
            resource_helpers_supported: false,
            extension_configs: Vec::new(),
            visible_extension_tools: Vec::new(),
            searchable_extension_tools: Vec::new(),
        });

        assert!(!inventory.agent_initialized);
        assert_eq!(
            inventory.warnings,
            vec!["agent not initialized".to_string()]
        );
        assert_eq!(
            inventory.mcp_servers,
            vec!["alpha".to_string(), "docs".to_string()]
        );
        assert_eq!(inventory.counts.native_total, 0);
        assert_eq!(inventory.counts.extension_surface_total, 0);
        assert_eq!(inventory.counts.mcp_server_total, 2);
        assert_eq!(inventory.counts.mcp_tool_total, 2);
        assert_eq!(inventory.counts.mcp_tool_visible_total, 2);
        assert_eq!(inventory.mcp_tools[0].server_name, "alpha");
        assert!(inventory.mcp_tools.iter().any(|entry| {
            entry.name == "search_docs" && entry.always_visible && entry.visible_in_context
        }));
    }

    #[test]
    fn test_build_tool_inventory_workbench_keeps_small_default_allowlist() {
        let expected_catalog = tool_catalog_entries_for_surface(WorkspaceToolSurface::workbench());
        let inventory = build_tool_inventory(AgentToolInventoryBuildInput {
            surface: WorkspaceToolSurface::workbench(),
            caller: "assistant".to_string(),
            agent_initialized: true,
            warnings: Vec::new(),
            persisted_execution_policy: None,
            request_metadata: None,
            mcp_server_names: Vec::new(),
            mcp_tools: Vec::new(),
            current_tool_definitions: Vec::new(),
            resource_helpers_supported: false,
            extension_configs: Vec::new(),
            visible_extension_tools: Vec::new(),
            searchable_extension_tools: Vec::new(),
        });
        let expected_default_allowed =
            workspace_default_allowed_tool_names(WorkspaceToolSurface::workbench())
                .into_iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>();

        assert_eq!(inventory.counts.catalog_total, expected_catalog.len());
        assert_eq!(
            inventory.counts.catalog_current_total,
            expected_catalog
                .iter()
                .filter(|entry| entry.lifecycle == ToolLifecycle::Current)
                .count()
        );
        assert_eq!(
            inventory.counts.catalog_compat_total,
            expected_catalog
                .iter()
                .filter(|entry| entry.lifecycle == ToolLifecycle::Compat)
                .count()
        );
        assert_eq!(inventory.default_allowed_tools, expected_default_allowed);
        assert_eq!(
            inventory.counts.default_allowed_total,
            inventory.default_allowed_tools.len()
        );
        assert!(inventory
            .default_allowed_tools
            .contains(&TOOL_SEARCH_TOOL_NAME.to_string()));
        assert!(inventory
            .default_allowed_tools
            .contains(&LIST_MCP_RESOURCES_TOOL_NAME.to_string()));
        assert!(inventory
            .default_allowed_tools
            .contains(&READ_MCP_RESOURCE_TOOL_NAME.to_string()));
        assert!(inventory
            .default_allowed_tools
            .contains(&"social_generate_cover_image".to_string()));
    }

    #[test]
    fn test_build_tool_inventory_removes_update_plan_when_turn_disables_it() {
        let inventory = build_tool_inventory(AgentToolInventoryBuildInput {
            surface: WorkspaceToolSurface::core(),
            caller: "assistant".to_string(),
            agent_initialized: true,
            warnings: Vec::new(),
            persisted_execution_policy: None,
            request_metadata: Some(json!({
                "harness": { "update_plan_enabled": false }
            })),
            mcp_server_names: Vec::new(),
            mcp_tools: Vec::new(),
            current_tool_definitions: vec![
                definition(
                    tool_runtime::update_plan::UPDATE_PLAN_NAME,
                    "update checklist",
                    json!({ "type": "object" }),
                ),
                definition("Read", "read file", json!({ "type": "object" })),
            ],
            resource_helpers_supported: false,
            extension_configs: Vec::new(),
            visible_extension_tools: Vec::new(),
            searchable_extension_tools: Vec::new(),
        });

        assert!(!inventory
            .default_allowed_tools
            .iter()
            .any(|name| name == tool_runtime::update_plan::UPDATE_PLAN_NAME));
        assert!(!inventory
            .catalog_tools
            .iter()
            .any(|entry| entry.name == tool_runtime::update_plan::UPDATE_PLAN_NAME));
        assert!(!inventory
            .native_tools
            .iter()
            .any(|entry| entry.name == tool_runtime::update_plan::UPDATE_PLAN_NAME));
        assert!(!inventory
            .runtime_tools
            .iter()
            .any(|entry| entry.name == tool_runtime::update_plan::UPDATE_PLAN_NAME));
        assert!(inventory
            .native_tools
            .iter()
            .any(|entry| entry.name == "Read"));
    }

    #[test]
    fn test_build_tool_inventory_uses_effective_execution_policy_provenance() {
        let inventory = build_tool_inventory(AgentToolInventoryBuildInput {
            surface: WorkspaceToolSurface::core(),
            caller: "assistant".to_string(),
            agent_initialized: true,
            warnings: Vec::new(),
            persisted_execution_policy: Some(ConfigToolExecutionPolicyConfig {
                tool_overrides: std::collections::HashMap::from([(
                    "exec_command".to_string(),
                    ConfigToolExecutionOverrideConfig {
                        warning_policy: Some(ConfigToolExecutionWarningPolicyConfig::None),
                        restriction_profile: None,
                        sandbox_profile: None,
                    },
                )]),
                ..Default::default()
            }),
            request_metadata: Some(json!({
                "harness": {
                    "executionPolicy": {
                        "toolOverrides": {
                            "exec_command": {
                                "sandboxProfile": "none"
                            }
                        }
                    }
                }
            })),
            mcp_server_names: Vec::new(),
            mcp_tools: Vec::new(),
            current_tool_definitions: vec![definition(
                "exec_command",
                "workspace command",
                json!({
                    "type": "object",
                    "x-lime": { "allowed_callers": ["assistant"] }
                }),
            )],
            resource_helpers_supported: false,
            extension_configs: Vec::new(),
            visible_extension_tools: Vec::new(),
            searchable_extension_tools: Vec::new(),
        });

        let exec_catalog = inventory
            .catalog_tools
            .iter()
            .find(|entry| entry.name == "exec_command")
            .expect("exec_command catalog entry should exist");
        assert_eq!(
            exec_catalog.execution_warning_policy,
            ToolExecutionWarningPolicy::None
        );
        assert_eq!(
            exec_catalog.execution_restriction_profile,
            ToolExecutionRestrictionProfile::WorkspaceShellCommand
        );
        assert_eq!(
            exec_catalog.execution_warning_policy_source,
            ToolExecutionPolicySource::Persisted
        );
        assert_eq!(
            exec_catalog.execution_restriction_profile_source,
            ToolExecutionPolicySource::Default
        );
        assert_eq!(
            exec_catalog.execution_sandbox_profile,
            ToolExecutionSandboxProfile::None
        );
        assert_eq!(
            exec_catalog.execution_sandbox_profile_source,
            ToolExecutionPolicySource::Runtime
        );

        let exec_native = inventory
            .native_tools
            .iter()
            .find(|entry| entry.name == "exec_command")
            .expect("exec_command native entry should exist");
        assert_eq!(
            exec_native.catalog_execution_warning_policy,
            Some(ToolExecutionWarningPolicy::None)
        );
        assert_eq!(
            exec_native.catalog_execution_restriction_profile,
            Some(ToolExecutionRestrictionProfile::WorkspaceShellCommand)
        );
        assert_eq!(
            exec_native.catalog_execution_sandbox_profile,
            Some(ToolExecutionSandboxProfile::None)
        );
        assert_eq!(
            exec_native.catalog_execution_warning_policy_source,
            Some(ToolExecutionPolicySource::Persisted)
        );
        assert_eq!(
            exec_native.catalog_execution_restriction_profile_source,
            Some(ToolExecutionPolicySource::Default)
        );
        assert_eq!(
            exec_native.catalog_execution_sandbox_profile_source,
            Some(ToolExecutionPolicySource::Runtime)
        );
    }

    #[test]
    fn test_build_tool_inventory_marks_extension_sources_and_statuses() {
        let inventory = build_tool_inventory(AgentToolInventoryBuildInput {
            surface: WorkspaceToolSurface::core(),
            caller: "assistant".to_string(),
            agent_initialized: true,
            warnings: Vec::new(),
            persisted_execution_policy: None,
            request_metadata: None,
            mcp_server_names: vec!["docs".to_string()],
            mcp_tools: Vec::new(),
            current_tool_definitions: Vec::new(),
            resource_helpers_supported: false,
            extension_configs: vec![
                builtin_extension(
                    "mcp__docs",
                    vec!["search_docs", "read_docs"],
                    true,
                    vec!["search_docs"],
                    Some("assistant"),
                ),
                builtin_extension("fs", vec!["list"], false, vec![], Some("code_execution")),
            ],
            visible_extension_tools: vec![seed("mcp__docs__read_docs", "loaded docs tool")],
            searchable_extension_tools: vec![
                seed("mcp__docs__search_docs", "search docs"),
                seed("mcp__docs__read_docs", "loaded docs tool"),
                seed("fs__list", "list files"),
            ],
        });

        let docs_surface = inventory
            .extension_surfaces
            .iter()
            .find(|entry| entry.extension_name == "mcp__docs")
            .expect("docs surface should exist");
        assert_eq!(
            docs_surface.source_kind,
            RuntimeExtensionSourceKind::McpBridge
        );
        assert_eq!(
            docs_surface.loaded_tools,
            vec!["mcp__docs__read_docs".to_string()]
        );
        assert_eq!(
            docs_surface.searchable_tools,
            vec![
                "mcp__docs__read_docs".to_string(),
                "mcp__docs__search_docs".to_string()
            ]
        );

        let fs_surface = inventory
            .extension_surfaces
            .iter()
            .find(|entry| entry.extension_name == "fs")
            .expect("fs surface should exist");
        assert_eq!(
            fs_surface.source_kind,
            RuntimeExtensionSourceKind::RuntimeExtension
        );

        let visible_tool = inventory
            .extension_tools
            .iter()
            .find(|entry| entry.name == "mcp__docs__search_docs")
            .expect("visible tool should exist");
        assert_eq!(visible_tool.status, "visible");
        assert!(!visible_tool.deferred_loading);
        assert!(visible_tool.visible_in_context);

        let loaded_tool = inventory
            .extension_tools
            .iter()
            .find(|entry| entry.name == "mcp__docs__read_docs")
            .expect("loaded tool should exist");
        assert_eq!(loaded_tool.status, "loaded");
        assert!(!loaded_tool.deferred_loading);
        assert!(loaded_tool.visible_in_context);

        let caller_filtered_tool = inventory
            .extension_tools
            .iter()
            .find(|entry| entry.name == "fs__list")
            .expect("caller filtered tool should exist");
        assert_eq!(caller_filtered_tool.status, "visible");
        assert_eq!(
            caller_filtered_tool.source_kind,
            RuntimeExtensionSourceKind::RuntimeExtension
        );
        assert_eq!(
            caller_filtered_tool.allowed_caller.as_deref(),
            Some("code_execution")
        );
        assert!(!caller_filtered_tool.caller_allowed);
        assert!(!caller_filtered_tool.visible_in_context);
    }

    #[test]
    fn test_build_tool_inventory_prefers_longest_extension_name_match() {
        let inventory = build_tool_inventory(AgentToolInventoryBuildInput {
            surface: WorkspaceToolSurface::core(),
            caller: "assistant".to_string(),
            agent_initialized: true,
            warnings: Vec::new(),
            persisted_execution_policy: None,
            request_metadata: None,
            mcp_server_names: Vec::new(),
            mcp_tools: Vec::new(),
            current_tool_definitions: Vec::new(),
            resource_helpers_supported: false,
            extension_configs: vec![
                builtin_extension("mcp__docs", vec!["search"], true, vec![], Some("assistant")),
                builtin_extension(
                    "mcp__docs__admin",
                    vec!["search"],
                    true,
                    vec![],
                    Some("code_execution"),
                ),
            ],
            visible_extension_tools: Vec::new(),
            searchable_extension_tools: vec![seed("mcp__docs__admin__search", "admin search")],
        });

        let tool = inventory
            .extension_tools
            .iter()
            .find(|entry| entry.name == "mcp__docs__admin__search")
            .expect("nested extension tool should exist");
        assert_eq!(tool.extension_name.as_deref(), Some("mcp__docs__admin"));
        assert_eq!(tool.allowed_caller.as_deref(), Some("code_execution"));
        assert_eq!(tool.status, "deferred");
        assert!(tool.deferred_loading);
        assert!(!tool.caller_allowed);
    }

    #[test]
    fn test_resolve_extension_tool_runtime_status_defaults_unknown_tools_visible() {
        let status = resolve_extension_tool_runtime_status(
            &[builtin_extension(
                "mcp__docs",
                vec!["search"],
                true,
                vec![],
                Some("assistant"),
            )],
            &HashSet::new(),
            "unmapped__tool",
        );

        assert_eq!(
            status,
            ExtensionToolRuntimeStatus {
                status: "visible",
                deferred_loading: false,
                extension_name: None,
                allowed_caller: None,
            }
        );
    }

    #[test]
    fn test_build_tool_inventory_native_marks_always_visible_deferred_tools_visible() {
        let inventory = build_tool_inventory(AgentToolInventoryBuildInput {
            surface: WorkspaceToolSurface::core(),
            caller: "assistant".to_string(),
            agent_initialized: true,
            warnings: Vec::new(),
            persisted_execution_policy: None,
            request_metadata: None,
            mcp_server_names: Vec::new(),
            mcp_tools: Vec::new(),
            current_tool_definitions: vec![
                definition(
                    "exec_command",
                    "workspace command",
                    json!({
                        "type": "object",
                        "x-lime": {
                            "deferred_loading": true,
                            "allowed_callers": ["assistant"]
                        }
                    }),
                ),
                definition(
                    "review_docs",
                    "review docs",
                    json!({
                        "type": "object",
                        "x-lime": {
                            "deferred_loading": true,
                            "always_visible": true,
                            "allowed_callers": ["assistant"],
                            "tags": ["docs"],
                            "input_examples": [{"query": "rust"}]
                        }
                    }),
                ),
                definition(
                    "admin_secret",
                    "admin only",
                    json!({
                        "type": "object",
                        "x-lime": {
                            "deferred_loading": true,
                            "allowed_callers": ["code_execution"]
                        }
                    }),
                ),
            ],
            resource_helpers_supported: false,
            extension_configs: Vec::new(),
            visible_extension_tools: Vec::new(),
            searchable_extension_tools: Vec::new(),
        });

        let review_docs = inventory
            .native_tools
            .iter()
            .find(|entry| entry.name == "review_docs")
            .expect("review_docs should exist");
        assert!(review_docs.deferred_loading);
        assert!(review_docs.always_visible);
        assert_eq!(review_docs.tags, vec!["docs".to_string()]);
        assert_eq!(review_docs.input_examples_count, 1);
        assert!(review_docs.caller_allowed);
        assert!(review_docs.visible_in_context);

        let admin_secret = inventory
            .native_tools
            .iter()
            .find(|entry| entry.name == "admin_secret")
            .expect("admin_secret should exist");
        assert!(!admin_secret.caller_allowed);
        assert!(!admin_secret.visible_in_context);
        assert_eq!(
            inventory
                .native_tools
                .iter()
                .find(|entry| entry.name == "exec_command")
                .and_then(|entry| entry.catalog_execution_sandbox_profile),
            Some(ToolExecutionSandboxProfile::WorkspaceCommand)
        );
        assert_eq!(inventory.counts.native_visible_total, 1);
    }

    #[test]
    fn test_build_tool_inventory_gates_resource_helpers_by_resource_support() {
        for (resource_helpers_supported, expected_visible) in [(false, false), (true, true)] {
            let inventory = build_tool_inventory(AgentToolInventoryBuildInput {
                surface: WorkspaceToolSurface::core(),
                caller: "assistant".to_string(),
                agent_initialized: true,
                warnings: Vec::new(),
                persisted_execution_policy: None,
                request_metadata: None,
                mcp_server_names: Vec::new(),
                mcp_tools: Vec::new(),
                current_tool_definitions: vec![
                    definition(
                        TOOL_SEARCH_TOOL_NAME,
                        "search tools",
                        json!({ "type": "object" }),
                    ),
                    definition(
                        LIST_MCP_RESOURCES_TOOL_NAME,
                        "list mcp resources",
                        json!({ "type": "object" }),
                    ),
                    definition(
                        READ_MCP_RESOURCE_TOOL_NAME,
                        "read mcp resource",
                        json!({ "type": "object" }),
                    ),
                ],
                resource_helpers_supported,
                extension_configs: Vec::new(),
                visible_extension_tools: Vec::new(),
                searchable_extension_tools: Vec::new(),
            });

            let tool_search = inventory
                .runtime_tools
                .iter()
                .find(|entry| entry.name == TOOL_SEARCH_TOOL_NAME)
                .expect("tool_search should stay in inventory");
            assert!(tool_search.visible_in_context);

            for tool_name in [LIST_MCP_RESOURCES_TOOL_NAME, READ_MCP_RESOURCE_TOOL_NAME] {
                let native_tool = inventory
                    .native_tools
                    .iter()
                    .find(|entry| entry.name == tool_name)
                    .expect("resource helper should stay listed in native inventory");
                assert_eq!(
                    native_tool.visible_in_context, expected_visible,
                    "{tool_name} native visibility should follow resource support"
                );

                let runtime_tool = inventory
                    .runtime_tools
                    .iter()
                    .find(|entry| entry.name == tool_name)
                    .expect("resource helper should stay listed in runtime inventory");
                assert_eq!(
                    runtime_tool.visible_in_context, expected_visible,
                    "{tool_name} runtime visibility should follow resource support"
                );
            }
        }
    }
}
