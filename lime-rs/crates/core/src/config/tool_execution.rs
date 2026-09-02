use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ToolExecutionWarningPolicyConfig {
    #[default]
    None,
    ShellCommandRisk,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ToolExecutionRestrictionProfileConfig {
    #[default]
    None,
    WorkspacePathRequired,
    WorkspacePathOptional,
    WorkspaceAbsolutePathRequired,
    WorkspaceShellCommand,
    AnalyzeImageInput,
    SafeHttpsUrlRequired,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ToolExecutionSandboxProfileConfig {
    #[default]
    None,
    WorkspaceCommand,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ToolExecutionCommandRiskLevelConfig {
    Low,
    #[default]
    Medium,
    High,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ToolExecutionCommandRuleMatchTypeConfig {
    #[default]
    Regex,
    Prefix,
    Exact,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ToolExecutionCommandRuleConfig {
    #[serde(default, alias = "ruleId", alias = "id")]
    pub rule_id: String,
    #[serde(default, alias = "matchType")]
    pub match_type: ToolExecutionCommandRuleMatchTypeConfig,
    #[serde(default)]
    pub pattern: String,
    #[serde(default, alias = "riskLevel")]
    pub risk_level: ToolExecutionCommandRiskLevelConfig,
    #[serde(default, alias = "reasonCode")]
    pub reason_code: String,
    #[serde(default)]
    pub reason: String,
}

impl ToolExecutionCommandRuleConfig {
    pub fn is_default(value: &Self) -> bool {
        value.rule_id.is_empty()
            && value.pattern.is_empty()
            && value.match_type == ToolExecutionCommandRuleMatchTypeConfig::Regex
            && value.risk_level == ToolExecutionCommandRiskLevelConfig::Medium
            && value.reason_code.is_empty()
            && value.reason.is_empty()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ToolExecutionNetworkRuleTargetConfig {
    #[default]
    Url,
    Host,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ToolExecutionNetworkRuleConfig {
    #[serde(default, alias = "ruleId", alias = "id")]
    pub rule_id: String,
    #[serde(default, alias = "matchType")]
    pub match_type: ToolExecutionCommandRuleMatchTypeConfig,
    #[serde(default)]
    pub target: ToolExecutionNetworkRuleTargetConfig,
    #[serde(default)]
    pub pattern: String,
    #[serde(default, alias = "riskLevel")]
    pub risk_level: ToolExecutionCommandRiskLevelConfig,
    #[serde(default, alias = "reasonCode")]
    pub reason_code: String,
    #[serde(default)]
    pub reason: String,
}

impl ToolExecutionNetworkRuleConfig {
    pub fn is_default(value: &Self) -> bool {
        value.rule_id.is_empty()
            && value.pattern.is_empty()
            && value.match_type == ToolExecutionCommandRuleMatchTypeConfig::Regex
            && value.target == ToolExecutionNetworkRuleTargetConfig::Url
            && value.risk_level == ToolExecutionCommandRiskLevelConfig::Medium
            && value.reason_code.is_empty()
            && value.reason.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ToolExecutionOverrideConfig {
    #[serde(
        default,
        alias = "warningPolicy",
        skip_serializing_if = "Option::is_none"
    )]
    pub warning_policy: Option<ToolExecutionWarningPolicyConfig>,
    #[serde(
        default,
        alias = "restrictionProfile",
        skip_serializing_if = "Option::is_none"
    )]
    pub restriction_profile: Option<ToolExecutionRestrictionProfileConfig>,
    #[serde(
        default,
        alias = "sandboxProfile",
        skip_serializing_if = "Option::is_none"
    )]
    pub sandbox_profile: Option<ToolExecutionSandboxProfileConfig>,
}

impl ToolExecutionOverrideConfig {
    pub fn is_default(value: &Self) -> bool {
        value.warning_policy.is_none()
            && value.restriction_profile.is_none()
            && value.sandbox_profile.is_none()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ToolExecutionPolicyConfig {
    /// Whether the Codex-compatible checklist tool is exposed to the model.
    /// The feature is opt-in, matching Codex's `tools.update_plan.enabled` default.
    #[serde(default, alias = "updatePlanEnabled", skip_serializing_if = "is_false")]
    pub update_plan_enabled: bool,
    #[serde(
        default,
        alias = "toolOverrides",
        skip_serializing_if = "HashMap::is_empty"
    )]
    pub tool_overrides: HashMap<String, ToolExecutionOverrideConfig>,
    #[serde(
        default,
        alias = "shellCommandRules",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub shell_command_rules: Vec<ToolExecutionCommandRuleConfig>,
    #[serde(default, alias = "networkRules", skip_serializing_if = "Vec::is_empty")]
    pub network_rules: Vec<ToolExecutionNetworkRuleConfig>,
}

impl ToolExecutionPolicyConfig {
    pub fn is_default(value: &Self) -> bool {
        !value.update_plan_enabled
            && value.tool_overrides.is_empty()
            && value.shell_command_rules.is_empty()
            && value.network_rules.is_empty()
    }
}

fn is_false(value: &bool) -> bool {
    !*value
}
