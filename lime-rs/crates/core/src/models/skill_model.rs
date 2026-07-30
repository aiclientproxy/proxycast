use super::app_type::AppType;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

const SKILL_FRONTMATTER_NAME: &str = "name";
const SKILL_FRONTMATTER_DESCRIPTION: &str = "description";
const SKILL_FRONTMATTER_LICENSE: &str = "license";
const SKILL_FRONTMATTER_COMPATIBILITY: &str = "compatibility";
const SKILL_FRONTMATTER_METADATA: &str = "metadata";
const SKILL_FRONTMATTER_ALLOWED_TOOLS: &str = "allowed-tools";
const SKILL_FRONTMATTER_ALLOWED_TOOLS_ALIAS: &str = "allowed_tools";
const LEGACY_LIME_TOP_LEVEL_FIELDS: &[&str] = &[
    "argument-hint",
    "argument_hint",
    "when-to-use",
    "when_to_use",
    "execution-mode",
    "steps-json",
    "provider",
    "disable-model-invocation",
];

pub const VIDEO_GENERATE_SKILL_DIRECTORY: &str = "video_generate";
pub const TRANSCRIPTION_GENERATE_SKILL_DIRECTORY: &str = "transcription_generate";
pub const BROADCAST_GENERATE_SKILL_DIRECTORY: &str = "broadcast_generate";
pub const COVER_GENERATE_SKILL_DIRECTORY: &str = "cover_generate";
pub const MODAL_RESOURCE_SEARCH_SKILL_DIRECTORY: &str = "modal_resource_search";
pub const IMAGE_GENERATE_SKILL_DIRECTORY: &str = "image_generate";
pub const LIBRARY_SKILL_DIRECTORY: &str = "library";
pub const URL_PARSE_SKILL_DIRECTORY: &str = "url_parse";
pub const RESEARCH_SKILL_DIRECTORY: &str = "research";
pub const REPORT_GENERATE_SKILL_DIRECTORY: &str = "report_generate";
pub const SITE_SEARCH_SKILL_DIRECTORY: &str = "site_search";
pub const PDF_READ_SKILL_DIRECTORY: &str = "pdf_read";
pub const PRESENTATION_GENERATE_SKILL_DIRECTORY: &str = "presentation_generate";
pub const FORM_GENERATE_SKILL_DIRECTORY: &str = "form_generate";
pub const SUMMARY_SKILL_DIRECTORY: &str = "summary";
pub const TRANSLATION_SKILL_DIRECTORY: &str = "translation";
pub const ANALYSIS_SKILL_DIRECTORY: &str = "analysis";
pub const TYPESETTING_SKILL_DIRECTORY: &str = "typesetting";
pub const WEBPAGE_GENERATE_SKILL_DIRECTORY: &str = "webpage_generate";
pub const CONTENT_POST_WITH_COVER_SKILL_DIRECTORY: &str = "content_post_with_cover";
pub const CONTENT_FACTORY_KNOWLEDGE_BUILDER_SKILL_DIRECTORY: &str = "knowledge-builder";
pub const CONTENT_FACTORY_ARTICLE_WRITER_SKILL_DIRECTORY: &str = "article-writer";
pub const CONTENT_FACTORY_CONTENT_REVIEWER_SKILL_DIRECTORY: &str = "content-reviewer";
pub const KNOWLEDGE_BUILDER_SKILL_DIRECTORY: &str = "knowledge_builder";
pub const PERSONAL_IP_KNOWLEDGE_BUILDER_SKILL_DIRECTORY: &str = "personal-ip-knowledge-builder";
pub const BRAND_PERSONA_KNOWLEDGE_BUILDER_SKILL_DIRECTORY: &str = "brand-persona-knowledge-builder";
pub const CONTENT_OPERATIONS_KNOWLEDGE_BUILDER_SKILL_DIRECTORY: &str =
    "content-operations-knowledge-builder";
pub const PRIVATE_DOMAIN_OPERATIONS_KNOWLEDGE_BUILDER_SKILL_DIRECTORY: &str =
    "private-domain-operations-knowledge-builder";
pub const LIVE_COMMERCE_OPERATIONS_KNOWLEDGE_BUILDER_SKILL_DIRECTORY: &str =
    "live-commerce-operations-knowledge-builder";
pub const CAMPAIGN_OPERATIONS_KNOWLEDGE_BUILDER_SKILL_DIRECTORY: &str =
    "campaign-operations-knowledge-builder";
pub const BRAND_PRODUCT_KNOWLEDGE_BUILDER_SKILL_DIRECTORY: &str = "brand-product-knowledge-builder";
pub const ORGANIZATION_KNOWHOW_KNOWLEDGE_BUILDER_SKILL_DIRECTORY: &str =
    "organization-knowhow-knowledge-builder";
pub const GROWTH_STRATEGY_KNOWLEDGE_BUILDER_SKILL_DIRECTORY: &str =
    "growth-strategy-knowledge-builder";

pub const DEFAULT_LIME_SKILL_DIRECTORIES: [&str; 33] = [
    VIDEO_GENERATE_SKILL_DIRECTORY,
    TRANSCRIPTION_GENERATE_SKILL_DIRECTORY,
    BROADCAST_GENERATE_SKILL_DIRECTORY,
    COVER_GENERATE_SKILL_DIRECTORY,
    MODAL_RESOURCE_SEARCH_SKILL_DIRECTORY,
    IMAGE_GENERATE_SKILL_DIRECTORY,
    LIBRARY_SKILL_DIRECTORY,
    URL_PARSE_SKILL_DIRECTORY,
    RESEARCH_SKILL_DIRECTORY,
    REPORT_GENERATE_SKILL_DIRECTORY,
    SITE_SEARCH_SKILL_DIRECTORY,
    PDF_READ_SKILL_DIRECTORY,
    PRESENTATION_GENERATE_SKILL_DIRECTORY,
    FORM_GENERATE_SKILL_DIRECTORY,
    SUMMARY_SKILL_DIRECTORY,
    TRANSLATION_SKILL_DIRECTORY,
    ANALYSIS_SKILL_DIRECTORY,
    TYPESETTING_SKILL_DIRECTORY,
    WEBPAGE_GENERATE_SKILL_DIRECTORY,
    CONTENT_POST_WITH_COVER_SKILL_DIRECTORY,
    CONTENT_FACTORY_KNOWLEDGE_BUILDER_SKILL_DIRECTORY,
    CONTENT_FACTORY_ARTICLE_WRITER_SKILL_DIRECTORY,
    CONTENT_FACTORY_CONTENT_REVIEWER_SKILL_DIRECTORY,
    KNOWLEDGE_BUILDER_SKILL_DIRECTORY,
    PERSONAL_IP_KNOWLEDGE_BUILDER_SKILL_DIRECTORY,
    BRAND_PERSONA_KNOWLEDGE_BUILDER_SKILL_DIRECTORY,
    CONTENT_OPERATIONS_KNOWLEDGE_BUILDER_SKILL_DIRECTORY,
    PRIVATE_DOMAIN_OPERATIONS_KNOWLEDGE_BUILDER_SKILL_DIRECTORY,
    LIVE_COMMERCE_OPERATIONS_KNOWLEDGE_BUILDER_SKILL_DIRECTORY,
    CAMPAIGN_OPERATIONS_KNOWLEDGE_BUILDER_SKILL_DIRECTORY,
    BRAND_PRODUCT_KNOWLEDGE_BUILDER_SKILL_DIRECTORY,
    ORGANIZATION_KNOWHOW_KNOWLEDGE_BUILDER_SKILL_DIRECTORY,
    GROWTH_STRATEGY_KNOWLEDGE_BUILDER_SKILL_DIRECTORY,
];

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SkillSourceKind {
    Builtin,
    Other,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SkillCatalogSource {
    Project,
    User,
    Remote,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    pub key: String,
    pub name: String,
    pub description: String,
    pub directory: String,
    #[serde(rename = "localDirectoryPath", skip_serializing_if = "Option::is_none")]
    pub local_directory_path: Option<String>,
    #[serde(rename = "readmeUrl", skip_serializing_if = "Option::is_none")]
    pub readme_url: Option<String>,
    pub installed: bool,
    #[serde(rename = "sourceKind")]
    pub source_kind: SkillSourceKind,
    #[serde(rename = "catalogSource")]
    pub catalog_source: SkillCatalogSource,
    #[serde(rename = "repoOwner", skip_serializing_if = "Option::is_none")]
    pub repo_owner: Option<String>,
    #[serde(rename = "repoName", skip_serializing_if = "Option::is_none")]
    pub repo_name: Option<String>,
    #[serde(rename = "repoBranch", skip_serializing_if = "Option::is_none")]
    pub repo_branch: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compatibility: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
    #[serde(
        rename = "allowedTools",
        default,
        skip_serializing_if = "Vec::is_empty"
    )]
    pub allowed_tools: Vec<String>,
    #[serde(rename = "resourceSummary", skip_serializing_if = "Option::is_none")]
    pub resource_summary: Option<SkillResourceSummary>,
    #[serde(rename = "standardCompliance", skip_serializing_if = "Option::is_none")]
    pub standard_compliance: Option<SkillStandardCompliance>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillRepo {
    pub owner: String,
    pub name: String,
    pub branch: String,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillState {
    pub installed: bool,
    pub installed_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillMetadata {
    pub name: Option<String>,
    pub description: Option<String>,
    pub license: Option<String>,
    pub compatibility: Option<String>,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
    #[serde(default)]
    pub allowed_tools: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct SkillResourceSummary {
    #[serde(rename = "hasScripts")]
    pub has_scripts: bool,
    #[serde(rename = "hasReferences")]
    pub has_references: bool,
    #[serde(rename = "hasAssets")]
    pub has_assets: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct SkillStandardCompliance {
    #[serde(rename = "isStandard")]
    pub is_standard: bool,
    #[serde(
        rename = "validationErrors",
        default,
        skip_serializing_if = "Vec::is_empty"
    )]
    pub validation_errors: Vec<String>,
    #[serde(
        rename = "deprecatedFields",
        default,
        skip_serializing_if = "Vec::is_empty"
    )]
    pub deprecated_fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct SkillPackageInspection {
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compatibility: Option<String>,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
    #[serde(rename = "allowedTools", default)]
    pub allowed_tools: Vec<String>,
    #[serde(rename = "resourceSummary")]
    pub resource_summary: SkillResourceSummary,
    #[serde(rename = "standardCompliance")]
    pub standard_compliance: SkillStandardCompliance,
}

#[derive(Debug, Clone)]
pub struct ParsedSkillManifest {
    pub metadata: SkillMetadata,
    pub compliance: SkillStandardCompliance,
    pub raw_frontmatter: serde_yaml::Value,
}

impl ParsedSkillManifest {
    pub fn metadata_value(&self, key: &str) -> Option<&str> {
        self.metadata.metadata.get(key).map(|value| value.as_str())
    }

    pub fn raw_string(&self, key: &str) -> Option<String> {
        let mapping = self.raw_frontmatter.as_mapping()?;
        yaml_mapping_get(mapping, key).and_then(yaml_scalar_to_string)
    }

    pub fn raw_bool(&self, key: &str) -> Option<bool> {
        let mapping = self.raw_frontmatter.as_mapping()?;
        yaml_mapping_get(mapping, key).and_then(yaml_scalar_to_bool)
    }
}

pub fn split_skill_frontmatter(content: &str) -> Option<(&str, &str)> {
    let content = content.trim_start_matches('\u{feff}');
    let regex = regex::Regex::new(r"(?s)\A---\s*\n(?P<frontmatter>.*?)\n---\s*(?:\n|$)").ok()?;
    let captures = regex.captures(content)?;
    let frontmatter = captures.name("frontmatter")?.as_str();
    let body_start = captures.get(0)?.end();
    let body = content.get(body_start..).unwrap_or("");
    Some((frontmatter, body))
}

pub fn parse_skill_manifest_from_content(content: &str) -> Result<ParsedSkillManifest, String> {
    let Some((frontmatter, _body)) = split_skill_frontmatter(content) else {
        return Ok(ParsedSkillManifest {
            metadata: SkillMetadata {
                name: None,
                description: None,
                license: None,
                compatibility: None,
                metadata: HashMap::new(),
                allowed_tools: Vec::new(),
            },
            compliance: SkillStandardCompliance {
                is_standard: false,
                validation_errors: vec!["缺少以 --- 包裹的 YAML frontmatter".to_string()],
                deprecated_fields: Vec::new(),
            },
            raw_frontmatter: serde_yaml::Value::Null,
        });
    };

    let raw_frontmatter = serde_yaml::from_str::<serde_yaml::Value>(frontmatter)
        .map_err(|error| format!("解析 YAML frontmatter 失败: {error}"))?;
    let mapping = raw_frontmatter
        .as_mapping()
        .ok_or_else(|| "YAML frontmatter 顶层必须是对象".to_string())?;

    let mut validation_errors = Vec::new();
    let mut deprecated_fields = Vec::new();

    let name = required_string_field(
        mapping,
        SKILL_FRONTMATTER_NAME,
        &mut validation_errors,
        "缺少必填字段 `name`",
    );
    let description = required_string_field(
        mapping,
        SKILL_FRONTMATTER_DESCRIPTION,
        &mut validation_errors,
        "缺少必填字段 `description`",
    );
    let license = optional_string_field(mapping, SKILL_FRONTMATTER_LICENSE, &mut validation_errors);
    let compatibility = optional_string_field(
        mapping,
        SKILL_FRONTMATTER_COMPATIBILITY,
        &mut validation_errors,
    );
    let allowed_tools = parse_allowed_tools_field(mapping, &mut validation_errors);
    let metadata = parse_metadata_field(mapping, &mut validation_errors);

    for field in LEGACY_LIME_TOP_LEVEL_FIELDS {
        if yaml_mapping_get(mapping, field).is_some() {
            deprecated_fields.push((*field).to_string());
        }
    }

    deprecated_fields.sort();
    deprecated_fields.dedup();

    let compliance = SkillStandardCompliance {
        is_standard: validation_errors.is_empty(),
        validation_errors,
        deprecated_fields,
    };

    Ok(ParsedSkillManifest {
        metadata: SkillMetadata {
            name,
            description,
            license,
            compatibility,
            metadata,
            allowed_tools,
        },
        compliance,
        raw_frontmatter,
    })
}

pub fn summarize_skill_resources_dir(skill_dir: &Path) -> SkillResourceSummary {
    SkillResourceSummary {
        has_scripts: skill_dir.join("scripts").is_dir(),
        has_references: skill_dir.join("references").is_dir(),
        has_assets: skill_dir.join("assets").is_dir(),
    }
}

fn required_string_field(
    mapping: &serde_yaml::Mapping,
    key: &str,
    validation_errors: &mut Vec<String>,
    missing_error: &str,
) -> Option<String> {
    match yaml_mapping_get(mapping, key) {
        Some(value) => match yaml_scalar_to_string(value) {
            Some(parsed) if !parsed.trim().is_empty() => Some(parsed),
            Some(_) => {
                validation_errors.push(format!("字段 `{key}` 不能为空"));
                None
            }
            None => {
                validation_errors.push(format!("字段 `{key}` 必须是字符串"));
                None
            }
        },
        None => {
            validation_errors.push(missing_error.to_string());
            None
        }
    }
}

fn optional_string_field(
    mapping: &serde_yaml::Mapping,
    key: &str,
    validation_errors: &mut Vec<String>,
) -> Option<String> {
    let value = yaml_mapping_get(mapping, key)?;

    match yaml_scalar_to_string(value) {
        Some(parsed) if !parsed.trim().is_empty() => Some(parsed),
        Some(_) => None,
        None => {
            validation_errors.push(format!("字段 `{key}` 必须是字符串"));
            None
        }
    }
}

fn parse_allowed_tools_field(
    mapping: &serde_yaml::Mapping,
    validation_errors: &mut Vec<String>,
) -> Vec<String> {
    let Some(value) = yaml_mapping_get(mapping, SKILL_FRONTMATTER_ALLOWED_TOOLS)
        .or_else(|| yaml_mapping_get(mapping, SKILL_FRONTMATTER_ALLOWED_TOOLS_ALIAS))
    else {
        return Vec::new();
    };

    match value {
        serde_yaml::Value::String(single) => split_allowed_tools_string(single),
        serde_yaml::Value::Sequence(values) => {
            let mut tools = Vec::new();
            for item in values {
                match yaml_scalar_to_string(item) {
                    Some(tool) if !tool.trim().is_empty() => tools.push(tool),
                    _ => validation_errors
                        .push("字段 `allowed-tools` 只能包含字符串条目".to_string()),
                }
            }
            tools
        }
        _ => {
            validation_errors.push("字段 `allowed-tools` 必须是字符串或字符串数组".to_string());
            Vec::new()
        }
    }
}

fn split_allowed_tools_string(value: &str) -> Vec<String> {
    let separator: &[_] = if value.contains(',') {
        &[',']
    } else {
        &[' ', '\t', '\n', '\r']
    };
    value
        .split(separator)
        .map(|item| item.trim())
        .filter(|item| !item.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn parse_metadata_field(
    mapping: &serde_yaml::Mapping,
    validation_errors: &mut Vec<String>,
) -> HashMap<String, String> {
    let Some(value) = yaml_mapping_get(mapping, SKILL_FRONTMATTER_METADATA) else {
        return HashMap::new();
    };

    let Some(meta_mapping) = value.as_mapping() else {
        validation_errors.push("字段 `metadata` 必须是键值对象".to_string());
        return HashMap::new();
    };

    let mut metadata = HashMap::new();
    for (raw_key, raw_value) in meta_mapping {
        let Some(key) = yaml_scalar_to_string(raw_key) else {
            validation_errors.push("字段 `metadata` 的 key 必须是字符串".to_string());
            continue;
        };

        match yaml_scalar_to_string(raw_value) {
            Some(value) => {
                metadata.insert(key, value);
            }
            None => {
                validation_errors.push(format!("字段 `metadata.{key}` 必须是字符串、数字或布尔值"))
            }
        }
    }

    metadata
}

fn yaml_mapping_get<'a>(
    mapping: &'a serde_yaml::Mapping,
    key: &str,
) -> Option<&'a serde_yaml::Value> {
    mapping.get(serde_yaml::Value::String(key.to_string()))
}

fn yaml_scalar_to_string(value: &serde_yaml::Value) -> Option<String> {
    match value {
        serde_yaml::Value::String(value) => Some(value.clone()),
        serde_yaml::Value::Number(value) => Some(value.to_string()),
        serde_yaml::Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn yaml_scalar_to_bool(value: &serde_yaml::Value) -> Option<bool> {
    match value {
        serde_yaml::Value::Bool(value) => Some(*value),
        serde_yaml::Value::String(value) => match value.trim().to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "enabled" => Some(true),
            "false" | "0" | "no" | "disabled" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

impl Default for SkillRepo {
    fn default() -> Self {
        Self {
            owner: String::new(),
            name: String::new(),
            branch: "main".to_string(),
            enabled: true,
        }
    }
}

#[allow(dead_code)]
impl SkillRepo {
    pub fn new(owner: String, name: String, branch: String) -> Self {
        Self {
            owner,
            name,
            branch,
            enabled: true,
        }
    }

    pub fn github_url(&self) -> String {
        format!("https://github.com/{}/{}", self.owner, self.name)
    }

    pub fn zip_url(&self) -> String {
        format!(
            "https://github.com/{}/{}/archive/refs/heads/{}.zip",
            self.owner, self.name, self.branch
        )
    }
}

pub fn get_default_skill_repos() -> Vec<SkillRepo> {
    vec![
        // Lime 官方仓库（排第一位）
        SkillRepo {
            owner: "lime".to_string(),
            name: "skills".to_string(),
            branch: "main".to_string(),
            enabled: true,
        },
        SkillRepo {
            owner: "ComposioHQ".to_string(),
            name: "awesome-claude-skills".to_string(),
            branch: "main".to_string(),
            enabled: true,
        },
        SkillRepo {
            owner: "anthropics".to_string(),
            name: "skills".to_string(),
            branch: "main".to_string(),
            enabled: true,
        },
        SkillRepo {
            owner: "cexll".to_string(),
            name: "myclaude".to_string(),
            branch: "master".to_string(),
            enabled: true,
        },
    ]
}

pub fn is_default_lime_skill(directory: &str) -> bool {
    DEFAULT_LIME_SKILL_DIRECTORIES.contains(&directory)
}

pub fn resolve_skill_source_kind(app_type: &AppType, directory: &str) -> SkillSourceKind {
    if matches!(app_type, AppType::Lime) && is_default_lime_skill(directory) {
        SkillSourceKind::Builtin
    } else {
        SkillSourceKind::Other
    }
}

pub type SkillStates = HashMap<String, SkillState>;

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Feature: skills-platform-mvp, Property 1: Default Repositories Include Lime Official
    /// Validates: Requirements 1.1, 1.2, 1.3
    #[test]
    fn test_default_repos_include_lime_official() {
        let repos = get_default_skill_repos();

        // 验证列表非空
        assert!(!repos.is_empty(), "默认仓库列表不应为空");

        // 验证第一个仓库是 Lime 官方仓库
        let first_repo = &repos[0];
        assert_eq!(first_repo.owner, "lime", "第一个仓库的 owner 应为 lime");
        assert_eq!(first_repo.name, "skills", "第一个仓库的 name 应为 skills");
        assert_eq!(first_repo.branch, "main", "第一个仓库的 branch 应为 main");
        assert!(first_repo.enabled, "Lime 官方仓库应默认启用");
    }

    // Property 1: Default Repositories Include Lime Official (Property-Based Test)
    // For any call to get_default_skill_repos(), the returned list SHALL contain
    // a SkillRepo with owner="lime", name="skills", branch="main", and enabled=true,
    // and this repo SHALL be the first item in the list.
    // Validates: Requirements 1.1, 1.2, 1.3
    proptest! {
        #[test]
        fn prop_default_repos_lime_first(_seed in 0u64..1000) {
            // 无论调用多少次，结果应该一致
            let repos = get_default_skill_repos();

            // Property: 列表非空
            prop_assert!(!repos.is_empty());

            // Property: 第一个仓库是 Lime 官方仓库
            let first = &repos[0];
            prop_assert_eq!(&first.owner, "lime");
            prop_assert_eq!(&first.name, "skills");
            prop_assert_eq!(&first.branch, "main");
            prop_assert!(first.enabled);
        }
    }

    #[test]
    fn test_lime_repo_exists_in_list() {
        let repos = get_default_skill_repos();

        // 验证 Lime 仓库存在于列表中
        let lime_repo = repos
            .iter()
            .find(|r| r.owner == "lime" && r.name == "skills");
        assert!(lime_repo.is_some(), "Lime 官方仓库应存在于默认列表中");

        let repo = lime_repo.unwrap();
        assert_eq!(repo.branch, "main");
        assert!(repo.enabled);
    }

    #[test]
    fn test_default_lime_skill_directories_include_embedded_defaults() {
        assert!(is_default_lime_skill(VIDEO_GENERATE_SKILL_DIRECTORY));
        assert!(is_default_lime_skill(REPORT_GENERATE_SKILL_DIRECTORY));
        assert!(is_default_lime_skill(SITE_SEARCH_SKILL_DIRECTORY));
        assert!(is_default_lime_skill(PDF_READ_SKILL_DIRECTORY));
        assert!(is_default_lime_skill(FORM_GENERATE_SKILL_DIRECTORY));
        assert!(is_default_lime_skill(SUMMARY_SKILL_DIRECTORY));
        assert!(is_default_lime_skill(WEBPAGE_GENERATE_SKILL_DIRECTORY));
        assert!(is_default_lime_skill(
            CONTENT_POST_WITH_COVER_SKILL_DIRECTORY
        ));
        assert!(is_default_lime_skill(KNOWLEDGE_BUILDER_SKILL_DIRECTORY));
        assert!(is_default_lime_skill(
            PERSONAL_IP_KNOWLEDGE_BUILDER_SKILL_DIRECTORY
        ));
        assert!(is_default_lime_skill(
            BRAND_PERSONA_KNOWLEDGE_BUILDER_SKILL_DIRECTORY
        ));
        assert!(is_default_lime_skill(
            BRAND_PRODUCT_KNOWLEDGE_BUILDER_SKILL_DIRECTORY
        ));
        assert!(is_default_lime_skill(
            ORGANIZATION_KNOWHOW_KNOWLEDGE_BUILDER_SKILL_DIRECTORY
        ));
        assert!(is_default_lime_skill(
            GROWTH_STRATEGY_KNOWLEDGE_BUILDER_SKILL_DIRECTORY
        ));
        assert!(!is_default_lime_skill("custom-skill"));
    }

    #[test]
    fn test_resolve_skill_source_kind_only_marks_lime_defaults_as_builtin() {
        assert_eq!(
            resolve_skill_source_kind(&AppType::Lime, VIDEO_GENERATE_SKILL_DIRECTORY),
            SkillSourceKind::Builtin
        );
        assert_eq!(
            resolve_skill_source_kind(&AppType::Lime, "custom-skill"),
            SkillSourceKind::Other
        );
        assert_eq!(
            resolve_skill_source_kind(&AppType::Claude, VIDEO_GENERATE_SKILL_DIRECTORY),
            SkillSourceKind::Other
        );
    }

    #[test]
    fn parse_manifest_supports_agent_skills_standard_optional_fields() {
        let manifest = parse_skill_manifest_from_content(
            r#"---
name: docx
description: Use when working with Word .docx files.
license: Proprietary. LICENSE.txt has complete terms
compatibility: Requires Node.js and Python.
metadata:
  author: anthropic
allowed-tools: Bash(git:*) Bash(jq:*) Read
---

# DOCX
"#,
        )
        .unwrap();

        assert_eq!(manifest.metadata.name.as_deref(), Some("docx"));
        assert_eq!(
            manifest.metadata.compatibility.as_deref(),
            Some("Requires Node.js and Python.")
        );
        assert_eq!(
            manifest.metadata.allowed_tools,
            vec![
                "Bash(git:*)".to_string(),
                "Bash(jq:*)".to_string(),
                "Read".to_string()
            ]
        );
        assert!(manifest.compliance.validation_errors.is_empty());
    }

    #[test]
    fn parse_manifest_keeps_legacy_comma_allowed_tools_compatibility() {
        let manifest = parse_skill_manifest_from_content(
            r#"---
name: image-generate
description: Generate images.
allowed-tools: search_query, lime_create_image_generation_task
---
"#,
        )
        .unwrap();

        assert_eq!(
            manifest.metadata.allowed_tools,
            vec![
                "search_query".to_string(),
                "lime_create_image_generation_task".to_string()
            ]
        );
    }
}
