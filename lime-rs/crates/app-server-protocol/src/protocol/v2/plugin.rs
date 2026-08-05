use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PluginCatalogListParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default)]
    pub marketplace_paths: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PluginCatalogListResponse {
    pub plugins: Vec<PluginCatalogSummary>,
    pub generated_at: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PluginCatalogInstalledParams {}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PluginCatalogEnabledSetParams {
    pub plugin_id: String,
    pub enabled: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PluginCatalogEnabledSetResponse {
    pub plugin: PluginCatalogSummary,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PluginCatalogSummary {
    pub id: String,
    pub name: String,
    pub version: String,
    pub marketplace_id: String,
    pub content_digest: String,
    pub description: String,
    pub source: String,
    pub source_uri: String,
    pub installed: bool,
    pub enabled: bool,
    pub install_policy: String,
    pub auth_policy: String,
    pub availability: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disabled_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_version: Option<String>,
    pub skills_count: u32,
    pub mcp_servers_count: u32,
    pub apps_count: u32,
    pub hooks_count: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PluginCatalogReadParams {
    pub plugin_id: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PluginCatalogReadResponse {
    pub plugin: PluginCatalogDetail,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PluginCatalogDetail {
    pub summary: PluginCatalogSummary,
    pub skills: Vec<PluginCatalogCapability>,
    pub mcp_servers: Vec<PluginCatalogCapability>,
    pub apps: Vec<PluginCatalogCapability>,
    pub hooks: Vec<PluginCatalogHook>,
    pub ui_resources: Vec<PluginCatalogUiResource>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PluginCatalogCapability {
    pub id: String,
    pub name: String,
    pub description: String,
    pub requires_auth: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PluginCatalogHook {
    pub id: String,
    pub event: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PluginCatalogUiResource {
    pub id: String,
    pub resource_uri: String,
    pub kind: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PluginCatalogInstallParams {
    pub source_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub marketplace_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_digest: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PluginCatalogInstallResponse {
    pub plugin: PluginCatalogSummary,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PluginCatalogUninstallParams {
    pub plugin_id: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PluginCatalogUninstallResponse {
    pub plugin_id: String,
    pub uninstalled: bool,
}
