use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillsListParams {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cwds: Vec<PathBuf>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub force_reload: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillsListResponse {
    pub data: Vec<SkillsListEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillsExtraRootsSetParams {
    pub extra_roots: Vec<PathBuf>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillsExtraRootsSetResponse {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillsConfigWriteParams {
    #[serde(default)]
    pub path: Option<PathBuf>,
    #[serde(default)]
    pub name: Option<String>,
    pub enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillsConfigWriteResponse {
    pub effective_enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SkillScope {
    User,
    Repo,
    System,
    Admin,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillMetadata {
    pub name: String,
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub short_description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interface: Option<SkillInterface>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dependencies: Option<SkillDependencies>,
    pub path: PathBuf,
    pub scope: SkillScope,
    pub enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillInterface {
    pub display_name: Option<String>,
    pub short_description: Option<String>,
    pub icon_small: Option<PathBuf>,
    pub icon_large: Option<PathBuf>,
    pub icon_small_url: Option<String>,
    pub icon_large_url: Option<String>,
    pub brand_color: Option<String>,
    pub default_prompt: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillDependencies {
    pub tools: Vec<SkillToolDependency>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillToolDependency {
    #[serde(rename = "type")]
    pub dependency_type: String,
    pub value: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transport: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillErrorInfo {
    pub path: PathBuf,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SkillsListEntry {
    pub cwd: PathBuf,
    pub skills: Vec<SkillMetadata>,
    pub errors: Vec<SkillErrorInfo>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn skills_list_wire_matches_codex_shape() {
        assert_eq!(
            serde_json::to_value(SkillsListParams {
                cwds: vec![PathBuf::from("/tmp/repo")],
                force_reload: true,
            })
            .expect("params wire"),
            json!({"cwds":["/tmp/repo"],"forceReload":true})
        );
        assert_eq!(
            serde_json::to_value(SkillScope::Repo).expect("scope wire"),
            json!("repo")
        );
        assert_eq!(
            serde_json::to_value(SkillsConfigWriteParams {
                path: Some(PathBuf::from("/tmp/skill/SKILL.md")),
                name: None,
                enabled: false,
            })
            .expect("config write wire"),
            json!({
                "path": "/tmp/skill/SKILL.md",
                "name": null,
                "enabled": false
            })
        );
    }
}
