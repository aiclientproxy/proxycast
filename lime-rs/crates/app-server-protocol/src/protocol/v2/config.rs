use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Source of one effective configuration layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ConfigLayerSource {
    Mdm {
        domain: String,
        key: String,
    },
    System {
        file: String,
    },
    EnterpriseManaged {
        id: String,
        name: String,
    },
    User {
        file: String,
        profile: Option<String>,
    },
    Project {
        #[serde(rename = "dotCodexFolder")]
        dot_codex_folder: String,
    },
    SessionFlags,
    LegacyManagedConfigTomlFromFile {
        file: String,
    },
    LegacyManagedConfigTomlFromMdm,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ConfigLayerMetadata {
    pub name: ConfigLayerSource,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ConfigLayer {
    pub name: ConfigLayerSource,
    pub version: String,
    pub config: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disabled_reason: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ConfigReadParams {
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub include_layers: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ConfigReadResponse {
    pub config: serde_json::Value,
    pub origins: BTreeMap<String, ConfigLayerMetadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layers: Option<Vec<ConfigLayer>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum MergeStrategy {
    Replace,
    Upsert,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum WriteStatus {
    Ok,
    OkOverridden,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct OverriddenMetadata {
    pub message: String,
    pub overriding_layer: ConfigLayerMetadata,
    pub effective_value: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ConfigWriteResponse {
    pub status: WriteStatus,
    pub version: String,
    pub file_path: String,
    pub overridden_metadata: Option<OverriddenMetadata>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum ConfigWriteErrorCode {
    ConfigLayerReadonly,
    ConfigRequirementReadonly,
    ConfigVersionConflict,
    ConfigValidationError,
    ConfigPathNotFound,
    ConfigSchemaUnknownKey,
    UserLayerNotFound,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ConfigValueWriteParams {
    pub key_path: String,
    pub value: serde_json::Value,
    pub merge_strategy: MergeStrategy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_version: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ConfigEdit {
    pub key_path: String,
    pub value: serde_json::Value,
    pub merge_strategy: MergeStrategy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ConfigBatchWriteParams {
    pub edits: Vec<ConfigEdit>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_version: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub reload_user_config: bool,
}

/// Codex-compatible lifecycle stage for an experimental feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum ExperimentalFeatureStage {
    Beta,
    UnderDevelopment,
    Stable,
    Deprecated,
    Removed,
}

/// Feature metadata and current enablement state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ExperimentalFeature {
    pub name: String,
    pub stage: ExperimentalFeatureStage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub announcement: Option<String>,
    pub enabled: bool,
    pub default_enabled: bool,
}

/// Parameters for `experimentalFeature/list`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ExperimentalFeatureListParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
}

/// Response for `experimentalFeature/list`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ExperimentalFeatureListResponse {
    pub data: Vec<ExperimentalFeature>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}

/// Process-wide feature enablement updates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ExperimentalFeatureEnablementSetParams {
    pub enablement: BTreeMap<String, bool>,
}

/// Applied feature enablement updates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ExperimentalFeatureEnablementSetResponse {
    pub enablement: BTreeMap<String, bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TextPosition {
    /// 1-based line number.
    pub line: usize,
    /// 1-based column number (in Unicode scalar values).
    pub column: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TextRange {
    pub start: TextPosition,
    pub end: TextPosition,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ConfigWarningNotification {
    /// Concise summary of the warning.
    pub summary: String,
    /// Optional extra guidance or error details.
    pub details: Option<String>,
    /// Optional path to the config file that triggered the warning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// Optional range for the error location inside the config file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<TextRange>,
}
