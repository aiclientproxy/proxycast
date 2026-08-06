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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PluginSearchParams {
    pub search_term: String,
    #[serde(default)]
    pub scope: Option<PluginSearchScope>,
    #[serde(default)]
    pub cwds: Option<Vec<String>>,
    #[serde(default)]
    pub cursor: Option<String>,
    #[serde(default)]
    pub limit: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum PluginSearchScope {
    Global,
    Workspace,
    Personal,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PluginSearchResult {
    pub plugin: PluginSummary,
    pub marketplace_name: String,
    #[serde(default)]
    pub marketplace_path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PluginSearchResponse {
    pub data: Vec<PluginSearchResult>,
    #[serde(default)]
    pub next_cursor: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum PluginInstallPolicy {
    #[serde(rename = "NOT_AVAILABLE")]
    NotAvailable,
    #[serde(rename = "AVAILABLE")]
    Available,
    #[serde(rename = "INSTALLED_BY_DEFAULT")]
    InstalledByDefault,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum PluginInstallPolicySource {
    #[serde(rename = "WORKSPACE_SETTING")]
    WorkspaceSetting,
    #[serde(rename = "IMPLICIT_CANONICAL_APP")]
    ImplicitCanonicalApp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum PluginAuthPolicy {
    #[serde(rename = "ON_INSTALL")]
    OnInstall,
    #[serde(rename = "ON_USE")]
    OnUse,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum PluginAvailability {
    #[serde(rename = "AVAILABLE", alias = "ENABLED")]
    #[default]
    Available,
    #[serde(rename = "DISABLED_BY_ADMIN")]
    DisabledByAdmin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PluginDisabledReason {
    DisabledByAdmin,
    PlanNotEligible,
    RequiredAppUnavailable,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PluginSummary {
    pub id: String,
    #[serde(default)]
    pub remote_plugin_id: Option<String>,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub local_version: Option<String>,
    pub name: String,
    #[serde(default)]
    pub share_context: Option<PluginShareContext>,
    pub source: PluginSource,
    pub installed: bool,
    #[serde(default)]
    pub installed_at: Option<i64>,
    pub enabled: bool,
    pub install_policy: PluginInstallPolicy,
    #[serde(default)]
    pub install_policy_source: Option<PluginInstallPolicySource>,
    #[serde(default)]
    pub must_show_installation_interstitial: Option<bool>,
    pub auth_policy: PluginAuthPolicy,
    #[serde(default)]
    pub availability: PluginAvailability,
    #[serde(default)]
    pub disabled_reason: Option<PluginDisabledReason>,
    #[serde(default)]
    pub eligible_plan_types: Option<Vec<String>>,
    #[serde(default)]
    pub interface: Option<PluginInterface>,
    #[serde(default)]
    pub keywords: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PluginShareContext {
    pub remote_plugin_id: String,
    #[serde(default)]
    pub remote_version: Option<String>,
    #[serde(default)]
    pub discoverability: Option<PluginShareDiscoverability>,
    #[serde(default)]
    pub share_url: Option<String>,
    #[serde(default)]
    pub creator_account_user_id: Option<String>,
    #[serde(default)]
    pub creator_name: Option<String>,
    #[serde(default)]
    pub share_principals: Option<Vec<PluginSharePrincipal>>,
    #[serde(default)]
    pub can_publish_to_workspace: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum PluginShareDiscoverability {
    #[serde(rename = "LISTED")]
    Listed,
    #[serde(rename = "UNLISTED")]
    Unlisted,
    #[serde(rename = "PRIVATE")]
    Private,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum PluginSharePrincipalType {
    #[serde(rename = "user")]
    User,
    #[serde(rename = "group")]
    Group,
    #[serde(rename = "workspace")]
    Workspace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum PluginSharePrincipalRole {
    Reader,
    Editor,
    Owner,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PluginSharePrincipal {
    pub principal_type: PluginSharePrincipalType,
    pub principal_id: String,
    pub role: PluginSharePrincipalRole,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PluginInterface {
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub short_description: Option<String>,
    #[serde(default)]
    pub long_description: Option<String>,
    #[serde(default)]
    pub developer_name: Option<String>,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub website_url: Option<String>,
    #[serde(default)]
    pub privacy_policy_url: Option<String>,
    #[serde(default)]
    pub terms_of_service_url: Option<String>,
    #[serde(default)]
    pub default_prompt: Option<Vec<String>>,
    #[serde(default)]
    pub brand_color: Option<String>,
    #[serde(default)]
    pub composer_icon: Option<String>,
    #[serde(default)]
    pub composer_icon_url: Option<String>,
    #[serde(default)]
    pub logo: Option<String>,
    #[serde(default)]
    pub logo_dark: Option<String>,
    #[serde(default)]
    pub logo_url: Option<String>,
    #[serde(default)]
    pub logo_url_dark: Option<String>,
    #[serde(default)]
    pub screenshots: Vec<String>,
    #[serde(default)]
    pub screenshot_urls: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum PluginSource {
    #[serde(rename_all = "camelCase")]
    Local {
        path: String,
    },
    #[serde(rename_all = "camelCase")]
    Git {
        url: String,
        #[serde(default)]
        path: Option<String>,
        #[serde(default)]
        ref_name: Option<String>,
        #[serde(default)]
        sha: Option<String>,
    },
    #[serde(rename_all = "camelCase")]
    Npm {
        package: String,
        #[serde(default)]
        version: Option<String>,
        #[serde(default)]
        registry: Option<String>,
    },
    Remote,
}
