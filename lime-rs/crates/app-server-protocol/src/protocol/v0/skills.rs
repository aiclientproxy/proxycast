use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillReadParams {
    pub skill_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SkillDetailScope {
    Project,
    User,
    App,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SkillSource {
    Project,
    User,
    App,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SkillAuthority {
    Workspace,
    User,
    Application,
    External,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillDetailInterface {
    pub display_name: String,
    pub execution_mode: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub argument_hint: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillDetailToolDependency {
    #[serde(rename = "type")]
    pub dependency_type: String,
    pub value: String,
    pub required: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillDetailDependencies {
    pub tools: Vec<SkillDetailToolDependency>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillPolicy {
    pub allow_implicit_invocation: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when_to_use: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillLocator {
    pub directory: String,
    pub skill_file_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillSummary {
    pub skill_id: String,
    pub name: String,
    pub description: String,
    pub scope: SkillDetailScope,
    pub source: SkillSource,
    pub authority: SkillAuthority,
    pub enabled: bool,
    pub interface: SkillDetailInterface,
    pub dependencies: SkillDetailDependencies,
    pub policy: SkillPolicy,
    pub capabilities: Vec<String>,
    pub locator: SkillLocator,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillWorkflowStep {
    pub id: String,
    pub name: String,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillDetail {
    pub metadata: SkillSummary,
    pub markdown_content: String,
    pub workflow_steps: Vec<SkillWorkflowStep>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillReadResponse {
    pub skill: SkillDetail,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillManagementListResponse {
    pub skills: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillManagementListParams {
    pub app: String,
    #[serde(default)]
    pub refresh_remote: bool,
    #[serde(default)]
    pub scope: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillManagementInstallParams {
    pub app: String,
    pub directory: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillManagementUninstallParams {
    pub app: String,
    pub directory: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillRepositoryEntry {
    pub owner: String,
    pub name: String,
    pub branch: String,
    pub enabled: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillRepositorySaveParams {
    pub repo: SkillRepositoryEntry,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillRepositoryDeleteParams {
    pub owner: String,
    pub name: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillLocalInspectParams {
    pub app: String,
    pub directory: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillScaffoldCreateParams {
    pub app: String,
    pub request: serde_json::Value,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillLocalImportParams {
    pub app: String,
    pub source_path: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillRemoteInspectParams {
    pub owner: String,
    pub name: String,
    pub branch: String,
    pub directory: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillManagementWriteResponse {
    pub success: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillRepositoryListResponse {
    #[serde(default)]
    pub repos: Vec<SkillRepositoryEntry>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillInstalledDirectoriesListResponse {
    #[serde(default)]
    pub directories: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillLocalInspectResponse {
    pub inspection: serde_json::Value,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillScaffoldCreateResponse {
    pub inspection: serde_json::Value,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillLocalImportResponse {
    pub directory: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillRemoteInspectResponse {
    pub inspection: serde_json::Value,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillPackageLocalInspectParams {
    pub app: String,
    pub source_path: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillLocalDetailInspectParams {
    pub app: String,
    pub directory: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillLocalRenameParams {
    pub app: String,
    pub directory: String,
    pub new_directory: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillPackageLocalReplaceParams {
    pub app: String,
    pub directory: String,
    pub source_path: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillPackageLocalInstallParams {
    pub app: String,
    pub source_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skill_name: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillPackageExportParams {
    pub app: String,
    pub directory: String,
    pub target_path: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillMarketplaceBundleFile {
    pub path: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillMarketplaceInstallParams {
    pub app: String,
    pub manifest_version: String,
    pub name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub version: String,
    #[serde(default)]
    pub content_hash: String,
    #[serde(default)]
    pub file_count: u64,
    #[serde(default)]
    pub files: Vec<SkillMarketplaceBundleFile>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillDownloadInstallParams {
    pub app: String,
    pub skill_name: String,
    pub download_url: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillPackageLocalInspectResponse {
    pub directory: String,
    pub inspection: serde_json::Value,
    #[serde(default)]
    pub files: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillLocalDetailInspectResponse {
    pub directory: String,
    pub inspection: serde_json::Value,
    #[serde(default)]
    pub files: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillLocalRenameResponse {
    pub directory: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillPackageLocalInstallResponse {
    pub directory: String,
    pub inspection: serde_json::Value,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillPackageLocalReplaceResponse {
    pub directory: String,
    pub inspection: serde_json::Value,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillMarketplaceInstallResponse {
    pub directory: String,
    pub inspection: serde_json::Value,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillDownloadInstallResponse {
    pub directory: String,
    pub inspection: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn summary() -> SkillSummary {
        SkillSummary {
            skill_id: "project:writer".to_string(),
            name: "writer".to_string(),
            description: "Write clearly.".to_string(),
            scope: SkillDetailScope::Project,
            source: SkillSource::Project,
            authority: SkillAuthority::Workspace,
            enabled: true,
            interface: SkillDetailInterface {
                display_name: "Writer".to_string(),
                execution_mode: "prompt".to_string(),
                provider: None,
                model: None,
                argument_hint: None,
            },
            dependencies: SkillDetailDependencies {
                tools: vec![SkillDetailToolDependency {
                    dependency_type: "runtime_tool".to_string(),
                    value: "Read".to_string(),
                    required: true,
                }],
            },
            policy: SkillPolicy {
                allow_implicit_invocation: true,
                when_to_use: Some("Use for writing.".to_string()),
            },
            capabilities: vec!["Read".to_string()],
            locator: SkillLocator {
                directory: "/workspace/.agents/skills/writer".to_string(),
                skill_file_path: "/workspace/.agents/skills/writer/SKILL.md".to_string(),
            },
        }
    }

    #[test]
    fn skill_detail_metadata_serializes_typed_identity_and_locator() {
        let value = serde_json::to_value(summary()).expect("serialize skill metadata");

        assert_eq!(value["skillId"], json!("project:writer"));
        assert_eq!(value["authority"], json!("workspace"));
        assert_eq!(
            value["dependencies"]["tools"][0]["type"],
            json!("runtime_tool")
        );
        assert_eq!(
            value["locator"]["skillFilePath"],
            json!("/workspace/.agents/skills/writer/SKILL.md")
        );
    }

    #[test]
    fn skill_read_params_serialize_stable_identity() {
        let value = serde_json::to_value(SkillReadParams {
            skill_id: "project:writer".to_string(),
        })
        .expect("serialize skill read params");

        assert_eq!(value, json!({ "skillId": "project:writer" }));
    }

    #[test]
    fn skill_read_params_reject_legacy_name_only_shape() {
        let error = serde_json::from_value::<SkillReadParams>(json!({
            "skillName": "writer"
        }))
        .expect_err("legacy name-only params must fail closed");

        assert!(error.to_string().contains("skillId"));
    }

    #[test]
    fn detail_and_management_responses_keep_distinct_item_contracts() {
        let detail = summary();
        let management = SkillManagementListResponse {
            skills: vec![json!({
                "directory": "writer",
                "installed": true,
                "catalogSource": "project"
            })],
        };

        assert_eq!(detail.skill_id, "project:writer");
        assert_eq!(management.skills[0]["installed"], json!(true));
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillPackageExportResponse {
    pub directory: String,
    pub output_path: String,
    pub file_count: u64,
    pub bytes_written: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct WorkspaceSkillBindingsListParams {
    pub workspace_root: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caller: Option<String>,
    #[serde(default)]
    pub workbench: bool,
    #[serde(default)]
    pub browser_assist: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct WorkspaceSkillBindingsListResponse {
    pub bindings: serde_json::Value,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct WorkspaceRegisteredSkillsListParams {
    pub workspace_root: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct WorkspaceRegisteredSkillsListResponse {
    #[serde(default)]
    pub skills: Vec<serde_json::Value>,
}
