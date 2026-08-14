use crate::execution_policy::{
    ToolExecutionPolicy, ToolExecutionPolicyResolution, ToolExecutionPolicySource,
    ToolExecutionRestrictionProfile, ToolExecutionSandboxProfile, ToolExecutionWarningPolicy,
};
use crate::execution_rules::{
    classify_network_access, classify_shell_command_with_rules, NetworkRule, NetworkRuleMatch,
    NetworkRuleTarget, ShellCommandRiskLevel, ShellCommandRule, ShellCommandRuleMatch,
    ShellCommandRuleMatchType, ShellCommandRuleSource,
};
use lime_core::config::{
    ToolExecutionCommandRiskLevelConfig as ConfigToolExecutionCommandRiskLevelConfig,
    ToolExecutionCommandRuleConfig as ConfigToolExecutionCommandRuleConfig,
    ToolExecutionCommandRuleMatchTypeConfig as ConfigToolExecutionCommandRuleMatchTypeConfig,
    ToolExecutionNetworkRuleConfig as ConfigToolExecutionNetworkRuleConfig,
    ToolExecutionNetworkRuleTargetConfig as ConfigToolExecutionNetworkRuleTargetConfig,
    ToolExecutionOverrideConfig as ConfigToolExecutionOverrideConfig,
    ToolExecutionPolicyConfig as ConfigToolExecutionPolicyConfig,
    ToolExecutionRestrictionProfileConfig as ConfigToolExecutionRestrictionProfileConfig,
    ToolExecutionSandboxProfileConfig as ConfigToolExecutionSandboxProfileConfig,
    ToolExecutionWarningPolicyConfig as ConfigToolExecutionWarningPolicyConfig,
};
use serde_json::{json, Map as JsonMap, Value as JsonValue};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Default)]
pub struct ToolExecutionResolverInput<'a> {
    pub persisted_policy: Option<&'a ConfigToolExecutionPolicyConfig>,
    pub request_metadata: Option<&'a JsonValue>,
}

#[derive(Debug, Clone, Copy)]
pub struct ToolExecutionPolicyServiceOptions {
    pub default_policy_for_tool: fn(&str) -> ToolExecutionPolicy,
    pub tool_names_match: fn(&str, &str) -> bool,
}

#[derive(Debug, Clone, Copy)]
pub struct ToolExecutionPolicyService<'a> {
    input: ToolExecutionResolverInput<'a>,
    options: ToolExecutionPolicyServiceOptions,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ToolExecutionPolicyOverride {
    warning_policy: Option<ToolExecutionWarningPolicy>,
    restriction_profile: Option<ToolExecutionRestrictionProfile>,
    sandbox_profile: Option<ToolExecutionSandboxProfile>,
}

#[derive(Debug, Clone, Copy)]
struct RuntimeExecutionPolicyLayer<'a> {
    policy: &'a JsonMap<String, JsonValue>,
    policy_source: ToolExecutionPolicySource,
    rule_source: ShellCommandRuleSource,
}

impl<'a> ToolExecutionPolicyService<'a> {
    pub fn new(
        input: ToolExecutionResolverInput<'a>,
        options: ToolExecutionPolicyServiceOptions,
    ) -> Self {
        Self { input, options }
    }

    pub fn resolve(&self, tool_name: &str) -> ToolExecutionPolicyResolution {
        let default_policy = (self.options.default_policy_for_tool)(tool_name);
        let persisted_override =
            self.extract_persisted_tool_execution_override(tool_name, self.input.persisted_policy);
        let runtime_layers = extract_runtime_execution_policy_layers(self.input.request_metadata);
        let mut resolution = apply_tool_execution_override(
            ToolExecutionPolicyResolution {
                policy: default_policy,
                warning_policy_source: ToolExecutionPolicySource::Default,
                restriction_profile_source: ToolExecutionPolicySource::Default,
                sandbox_profile_source: ToolExecutionPolicySource::Default,
            },
            persisted_override,
            ToolExecutionPolicySource::Persisted,
        );

        for layer in runtime_layers {
            resolution = apply_tool_execution_override(
                resolution,
                self.extract_runtime_execution_policy_override(tool_name, layer.policy),
                layer.policy_source,
            );
        }

        resolution
    }

    pub fn metadata(&self, tool_name: &str, surface: &str) -> HashMap<String, JsonValue> {
        self.metadata_for_resolution(tool_name, surface, self.resolve(tool_name))
    }

    pub fn classify_shell_command(&self, command: &str) -> Option<ShellCommandRuleMatch> {
        let configured_rules = self.shell_command_rules();
        classify_shell_command_with_rules(command, &configured_rules)
    }

    pub fn classify_network_access(
        &self,
        tool_name: &str,
        params: &JsonValue,
        command: Option<&str>,
    ) -> Option<NetworkRuleMatch> {
        let configured_rules = self.network_rules();
        classify_network_access(tool_name, params, command, &configured_rules)
    }

    pub fn shell_command_rules(&self) -> Vec<ShellCommandRule> {
        let persisted_rules = self
            .input
            .persisted_policy
            .map(|policy| policy.shell_command_rules.as_slice())
            .unwrap_or_default();
        let runtime_layers = extract_runtime_execution_policy_layers(self.input.request_metadata);

        let mut rules =
            convert_shell_command_rules(persisted_rules, ShellCommandRuleSource::Persisted);
        for layer in runtime_layers {
            let layer_rules = extract_runtime_shell_command_rules(layer.policy);
            rules.extend(convert_shell_command_rules(&layer_rules, layer.rule_source));
        }
        rules
    }

    pub fn network_rules(&self) -> Vec<NetworkRule> {
        let persisted_rules = self
            .input
            .persisted_policy
            .map(|policy| policy.network_rules.as_slice())
            .unwrap_or_default();
        let runtime_layers = extract_runtime_execution_policy_layers(self.input.request_metadata);

        let mut rules = convert_network_rules(persisted_rules, ShellCommandRuleSource::Persisted);
        for layer in runtime_layers {
            let layer_rules = extract_runtime_network_rules(layer.policy);
            rules.extend(convert_network_rules(&layer_rules, layer.rule_source));
        }
        rules
    }

    pub fn metadata_for_resolution(
        &self,
        tool_name: &str,
        surface: &str,
        resolution: ToolExecutionPolicyResolution,
    ) -> HashMap<String, JsonValue> {
        HashMap::from([
            ("policyName".to_string(), json!("workspace_tool_execution")),
            ("policyProfile".to_string(), json!("workspace")),
            ("toolSurface".to_string(), json!(surface)),
            ("toolName".to_string(), json!(tool_name)),
            (
                "warningPolicy".to_string(),
                json!(resolution.policy.warning_policy),
            ),
            (
                "restrictionProfile".to_string(),
                json!(resolution.policy.restriction_profile),
            ),
            (
                "sandboxPolicy".to_string(),
                json!(resolution.policy.sandbox_profile),
            ),
            (
                "warningPolicySource".to_string(),
                json!(resolution.warning_policy_source),
            ),
            (
                "restrictionProfileSource".to_string(),
                json!(resolution.restriction_profile_source),
            ),
            (
                "sandboxPolicySource".to_string(),
                json!(resolution.sandbox_profile_source),
            ),
        ])
    }

    fn extract_persisted_tool_execution_override(
        &self,
        tool_name: &str,
        persisted_policy: Option<&ConfigToolExecutionPolicyConfig>,
    ) -> ToolExecutionPolicyOverride {
        let Some(tool_override) = persisted_policy
            .and_then(|policy| self.find_tool_override_config(&policy.tool_overrides, tool_name))
        else {
            return ToolExecutionPolicyOverride::default();
        };

        ToolExecutionPolicyOverride {
            warning_policy: tool_override
                .warning_policy
                .map(convert_warning_policy_config),
            restriction_profile: tool_override
                .restriction_profile
                .map(convert_restriction_profile_config),
            sandbox_profile: tool_override
                .sandbox_profile
                .map(convert_sandbox_profile_config),
        }
    }

    fn extract_runtime_execution_policy_override(
        &self,
        tool_name: &str,
        execution_policy: &JsonMap<String, JsonValue>,
    ) -> ToolExecutionPolicyOverride {
        let tool_overrides =
            find_named_object(execution_policy, &["tool_overrides", "toolOverrides"])
                .unwrap_or(execution_policy);
        let Some(tool_override) = self.find_case_insensitive_object(tool_overrides, tool_name)
        else {
            return ToolExecutionPolicyOverride::default();
        };

        ToolExecutionPolicyOverride {
            warning_policy: extract_named_string(
                tool_override,
                &["warning_policy", "warningPolicy"],
            )
            .and_then(parse_warning_policy),
            restriction_profile: extract_named_string(
                tool_override,
                &["restriction_profile", "restrictionProfile"],
            )
            .and_then(parse_restriction_profile),
            sandbox_profile: extract_named_string(
                tool_override,
                &["sandbox_profile", "sandboxProfile"],
            )
            .and_then(parse_sandbox_profile),
        }
    }

    fn find_case_insensitive_object<'b>(
        &self,
        object: &'b JsonMap<String, JsonValue>,
        key: &str,
    ) -> Option<&'b JsonMap<String, JsonValue>> {
        let normalized_key = key.trim();
        object.iter().find_map(|(candidate, value)| {
            (self.options.tool_names_match)(candidate, normalized_key)
                .then_some(value)
                .and_then(JsonValue::as_object)
        })
    }

    fn find_tool_override_config<'b>(
        &self,
        tool_overrides: &'b HashMap<String, ConfigToolExecutionOverrideConfig>,
        tool_name: &str,
    ) -> Option<&'b ConfigToolExecutionOverrideConfig> {
        let normalized_name = tool_name.trim();
        tool_overrides
            .iter()
            .find_map(|(candidate, override_config)| {
                (self.options.tool_names_match)(candidate, normalized_name)
                    .then_some(override_config)
            })
    }

    pub fn persisted_policy_from_metadata(
        request_metadata: Option<&JsonValue>,
    ) -> Option<ConfigToolExecutionPolicyConfig> {
        let policy_value = find_persisted_tool_execution_policy_value(request_metadata?)?;
        serde_json::from_value::<ConfigToolExecutionPolicyConfig>(policy_value.clone())
            .ok()
            .filter(|policy| !ConfigToolExecutionPolicyConfig::is_default(policy))
    }
}

fn extract_runtime_shell_command_rules(
    execution_policy: &JsonMap<String, JsonValue>,
) -> Vec<ConfigToolExecutionCommandRuleConfig> {
    let Some(rules) = find_named_array(
        execution_policy,
        &[
            "shell_command_rules",
            "shellCommandRules",
            "command_rules",
            "commandRules",
        ],
    ) else {
        return Vec::new();
    };

    rules
        .iter()
        .filter_map(|value| {
            serde_json::from_value::<ConfigToolExecutionCommandRuleConfig>(value.clone()).ok()
        })
        .collect()
}

fn extract_runtime_network_rules(
    execution_policy: &JsonMap<String, JsonValue>,
) -> Vec<ConfigToolExecutionNetworkRuleConfig> {
    let Some(rules) = find_named_array(
        execution_policy,
        &["network_rules", "networkRules", "url_rules", "urlRules"],
    ) else {
        return Vec::new();
    };

    rules
        .iter()
        .filter_map(|value| {
            serde_json::from_value::<ConfigToolExecutionNetworkRuleConfig>(value.clone()).ok()
        })
        .collect()
}

fn extract_runtime_execution_policy_layers(
    request_metadata: Option<&JsonValue>,
) -> Vec<RuntimeExecutionPolicyLayer<'_>> {
    let Some(value) = request_metadata else {
        return Vec::new();
    };
    let mut layers = Vec::new();
    collect_runtime_execution_policy_layers(value, &mut layers);
    layers
}

fn collect_runtime_execution_policy_layers<'a>(
    value: &'a JsonValue,
    layers: &mut Vec<RuntimeExecutionPolicyLayer<'a>>,
) {
    let Some(object) = value.as_object() else {
        return;
    };

    push_named_policy_layer(
        object,
        &[
            "organization_execution_policy",
            "organizationExecutionPolicy",
        ],
        ToolExecutionPolicySource::Organization,
        ShellCommandRuleSource::Organization,
        layers,
    );
    if let Some(policy_container) = find_named_object(object, &["policy", "policies"]) {
        push_named_policy_layer(
            policy_container,
            &["organization", "org"],
            ToolExecutionPolicySource::Organization,
            ShellCommandRuleSource::Organization,
            layers,
        );
        push_named_policy_layer(
            policy_container,
            &["user"],
            ToolExecutionPolicySource::User,
            ShellCommandRuleSource::User,
            layers,
        );
        push_named_policy_layer(
            policy_container,
            &["request"],
            ToolExecutionPolicySource::Request,
            ShellCommandRuleSource::Request,
            layers,
        );
    }
    if let Some(harness) = object.get("harness").and_then(JsonValue::as_object) {
        push_named_policy_layer(
            harness,
            &[
                "organization_execution_policy",
                "organizationExecutionPolicy",
            ],
            ToolExecutionPolicySource::Organization,
            ShellCommandRuleSource::Organization,
            layers,
        );
        push_named_policy_layer(
            harness,
            &["user_execution_policy", "userExecutionPolicy"],
            ToolExecutionPolicySource::User,
            ShellCommandRuleSource::User,
            layers,
        );
        push_named_policy_layer(
            harness,
            &["execution_policy", "executionPolicy"],
            ToolExecutionPolicySource::Runtime,
            ShellCommandRuleSource::Runtime,
            layers,
        );
        push_named_policy_layer(
            harness,
            &["request_execution_policy", "requestExecutionPolicy"],
            ToolExecutionPolicySource::Request,
            ShellCommandRuleSource::Request,
            layers,
        );
    }
    push_named_policy_layer(
        object,
        &["user_execution_policy", "userExecutionPolicy"],
        ToolExecutionPolicySource::User,
        ShellCommandRuleSource::User,
        layers,
    );
    push_named_policy_layer(
        object,
        &["execution_policy", "executionPolicy"],
        ToolExecutionPolicySource::Runtime,
        ShellCommandRuleSource::Runtime,
        layers,
    );
    push_named_policy_layer(
        object,
        &["request_execution_policy", "requestExecutionPolicy"],
        ToolExecutionPolicySource::Request,
        ShellCommandRuleSource::Request,
        layers,
    );

    [
        "runtime_options",
        "runtimeOptions",
        "runtime_request",
        "runtimeRequest",
        "metadata",
    ]
    .into_iter()
    .filter_map(|key| object.get(key))
    .for_each(|nested| collect_runtime_execution_policy_layers(nested, layers));
}

fn push_named_policy_layer<'a>(
    object: &'a JsonMap<String, JsonValue>,
    keys: &[&str],
    policy_source: ToolExecutionPolicySource,
    rule_source: ShellCommandRuleSource,
    layers: &mut Vec<RuntimeExecutionPolicyLayer<'a>>,
) {
    if let Some(policy) = find_named_object(object, keys) {
        layers.push(RuntimeExecutionPolicyLayer {
            policy,
            policy_source,
            rule_source,
        });
    }
}

fn find_persisted_tool_execution_policy_value(value: &JsonValue) -> Option<&JsonValue> {
    let object = value.as_object()?;
    if let Some(policy) = find_agent_tool_execution_policy_value(object) {
        return Some(policy);
    }

    [
        "config",
        "native_agent",
        "nativeAgent",
        "runtime_options",
        "runtimeOptions",
        "runtime_request",
        "runtimeRequest",
        "metadata",
    ]
    .into_iter()
    .filter_map(|key| object.get(key))
    .find_map(find_persisted_tool_execution_policy_value)
}

fn find_agent_tool_execution_policy_value(
    object: &JsonMap<String, JsonValue>,
) -> Option<&JsonValue> {
    find_named_value(object, &["tool_execution", "toolExecution"]).or_else(|| {
        find_named_object(object, &["agent", "native_agent", "nativeAgent"])
            .and_then(|agent| find_named_value(agent, &["tool_execution", "toolExecution"]))
    })
}

fn find_named_object<'a>(
    object: &'a JsonMap<String, JsonValue>,
    keys: &[&str],
) -> Option<&'a JsonMap<String, JsonValue>> {
    keys.iter()
        .filter_map(|key| object.get(*key))
        .find_map(JsonValue::as_object)
}

fn find_named_value<'a>(
    object: &'a JsonMap<String, JsonValue>,
    keys: &[&str],
) -> Option<&'a JsonValue> {
    keys.iter().find_map(|key| object.get(*key))
}

fn find_named_array<'a>(
    object: &'a JsonMap<String, JsonValue>,
    keys: &[&str],
) -> Option<&'a Vec<JsonValue>> {
    keys.iter()
        .filter_map(|key| object.get(*key))
        .find_map(JsonValue::as_array)
}

fn extract_named_string<'a>(
    object: &'a JsonMap<String, JsonValue>,
    keys: &[&str],
) -> Option<&'a str> {
    keys.iter()
        .filter_map(|key| object.get(*key))
        .find_map(JsonValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn apply_tool_execution_override(
    mut base: ToolExecutionPolicyResolution,
    tool_override: ToolExecutionPolicyOverride,
    source: ToolExecutionPolicySource,
) -> ToolExecutionPolicyResolution {
    if let Some(value) = tool_override.warning_policy {
        base.policy.warning_policy = value;
        base.warning_policy_source = source;
    }
    if let Some(value) = tool_override.restriction_profile {
        base.policy.restriction_profile = value;
        base.restriction_profile_source = source;
    }
    if let Some(value) = tool_override.sandbox_profile {
        base.policy.sandbox_profile = value;
        base.sandbox_profile_source = source;
    }
    base
}

fn convert_warning_policy_config(
    value: ConfigToolExecutionWarningPolicyConfig,
) -> ToolExecutionWarningPolicy {
    match value {
        ConfigToolExecutionWarningPolicyConfig::None => ToolExecutionWarningPolicy::None,
        ConfigToolExecutionWarningPolicyConfig::ShellCommandRisk => {
            ToolExecutionWarningPolicy::ShellCommandRisk
        }
    }
}

fn convert_restriction_profile_config(
    value: ConfigToolExecutionRestrictionProfileConfig,
) -> ToolExecutionRestrictionProfile {
    match value {
        ConfigToolExecutionRestrictionProfileConfig::None => ToolExecutionRestrictionProfile::None,
        ConfigToolExecutionRestrictionProfileConfig::WorkspacePathRequired => {
            ToolExecutionRestrictionProfile::WorkspacePathRequired
        }
        ConfigToolExecutionRestrictionProfileConfig::WorkspacePathOptional => {
            ToolExecutionRestrictionProfile::WorkspacePathOptional
        }
        ConfigToolExecutionRestrictionProfileConfig::WorkspaceAbsolutePathRequired => {
            ToolExecutionRestrictionProfile::WorkspaceAbsolutePathRequired
        }
        ConfigToolExecutionRestrictionProfileConfig::WorkspaceShellCommand => {
            ToolExecutionRestrictionProfile::WorkspaceShellCommand
        }
        ConfigToolExecutionRestrictionProfileConfig::AnalyzeImageInput => {
            ToolExecutionRestrictionProfile::AnalyzeImageInput
        }
        ConfigToolExecutionRestrictionProfileConfig::SafeHttpsUrlRequired => {
            ToolExecutionRestrictionProfile::SafeHttpsUrlRequired
        }
    }
}

fn convert_sandbox_profile_config(
    value: ConfigToolExecutionSandboxProfileConfig,
) -> ToolExecutionSandboxProfile {
    match value {
        ConfigToolExecutionSandboxProfileConfig::None => ToolExecutionSandboxProfile::None,
        ConfigToolExecutionSandboxProfileConfig::WorkspaceCommand => {
            ToolExecutionSandboxProfile::WorkspaceCommand
        }
    }
}

fn convert_shell_command_rules(
    configs: &[ConfigToolExecutionCommandRuleConfig],
    source: ShellCommandRuleSource,
) -> Vec<ShellCommandRule> {
    configs
        .iter()
        .filter_map(|config| {
            let rule_id = config.rule_id.trim();
            let pattern = config.pattern.trim();
            if rule_id.is_empty() || pattern.is_empty() {
                return None;
            }

            Some(ShellCommandRule {
                rule_id: rule_id.to_string(),
                match_type: convert_command_rule_match_type_config(config.match_type),
                pattern: pattern.to_string(),
                risk_level: convert_command_risk_level_config(config.risk_level),
                reason_code: config.reason_code.trim().to_string(),
                reason: config.reason.trim().to_string(),
                source,
            })
        })
        .collect()
}

fn convert_network_rules(
    configs: &[ConfigToolExecutionNetworkRuleConfig],
    source: ShellCommandRuleSource,
) -> Vec<NetworkRule> {
    configs
        .iter()
        .filter_map(|config| {
            let rule_id = config.rule_id.trim();
            let pattern = config.pattern.trim();
            if rule_id.is_empty() || pattern.is_empty() {
                return None;
            }

            Some(NetworkRule {
                rule_id: rule_id.to_string(),
                match_type: convert_command_rule_match_type_config(config.match_type),
                target: convert_network_rule_target_config(config.target),
                pattern: pattern.to_string(),
                risk_level: convert_command_risk_level_config(config.risk_level),
                reason_code: config.reason_code.trim().to_string(),
                reason: config.reason.trim().to_string(),
                source,
            })
        })
        .collect()
}

fn convert_command_rule_match_type_config(
    value: ConfigToolExecutionCommandRuleMatchTypeConfig,
) -> ShellCommandRuleMatchType {
    match value {
        ConfigToolExecutionCommandRuleMatchTypeConfig::Regex => ShellCommandRuleMatchType::Regex,
        ConfigToolExecutionCommandRuleMatchTypeConfig::Prefix => ShellCommandRuleMatchType::Prefix,
        ConfigToolExecutionCommandRuleMatchTypeConfig::Exact => ShellCommandRuleMatchType::Exact,
    }
}

fn convert_network_rule_target_config(
    value: ConfigToolExecutionNetworkRuleTargetConfig,
) -> NetworkRuleTarget {
    match value {
        ConfigToolExecutionNetworkRuleTargetConfig::Url => NetworkRuleTarget::Url,
        ConfigToolExecutionNetworkRuleTargetConfig::Host => NetworkRuleTarget::Host,
    }
}

fn convert_command_risk_level_config(
    value: ConfigToolExecutionCommandRiskLevelConfig,
) -> ShellCommandRiskLevel {
    match value {
        ConfigToolExecutionCommandRiskLevelConfig::Low => ShellCommandRiskLevel::Low,
        ConfigToolExecutionCommandRiskLevelConfig::Medium => ShellCommandRiskLevel::Medium,
        ConfigToolExecutionCommandRiskLevelConfig::High => ShellCommandRiskLevel::High,
    }
}

fn parse_warning_policy(value: &str) -> Option<ToolExecutionWarningPolicy> {
    match value.trim() {
        "none" => Some(ToolExecutionWarningPolicy::None),
        "shell_command_risk" => Some(ToolExecutionWarningPolicy::ShellCommandRisk),
        _ => None,
    }
}

fn parse_restriction_profile(value: &str) -> Option<ToolExecutionRestrictionProfile> {
    match value.trim() {
        "none" => Some(ToolExecutionRestrictionProfile::None),
        "workspace_path_required" => Some(ToolExecutionRestrictionProfile::WorkspacePathRequired),
        "workspace_path_optional" => Some(ToolExecutionRestrictionProfile::WorkspacePathOptional),
        "workspace_absolute_path_required" => {
            Some(ToolExecutionRestrictionProfile::WorkspaceAbsolutePathRequired)
        }
        "workspace_shell_command" => Some(ToolExecutionRestrictionProfile::WorkspaceShellCommand),
        "analyze_image_input" => Some(ToolExecutionRestrictionProfile::AnalyzeImageInput),
        "safe_https_url_required" => Some(ToolExecutionRestrictionProfile::SafeHttpsUrlRequired),
        _ => None,
    }
}

fn parse_sandbox_profile(value: &str) -> Option<ToolExecutionSandboxProfile> {
    match value.trim() {
        "none" => Some(ToolExecutionSandboxProfile::None),
        "workspace_command" => Some(ToolExecutionSandboxProfile::WorkspaceCommand),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn default_policy_for_tool(tool_name: &str) -> ToolExecutionPolicy {
        if tool_name.eq_ignore_ascii_case("exec_command") {
            ToolExecutionPolicy {
                warning_policy: ToolExecutionWarningPolicy::ShellCommandRisk,
                restriction_profile: ToolExecutionRestrictionProfile::WorkspaceShellCommand,
                sandbox_profile: ToolExecutionSandboxProfile::WorkspaceCommand,
            }
        } else {
            ToolExecutionPolicy::default()
        }
    }

    fn tool_names_match(left: &str, right: &str) -> bool {
        left.eq_ignore_ascii_case(right)
    }

    fn service<'a>(input: ToolExecutionResolverInput<'a>) -> ToolExecutionPolicyService<'a> {
        ToolExecutionPolicyService::new(
            input,
            ToolExecutionPolicyServiceOptions {
                default_policy_for_tool,
                tool_names_match,
            },
        )
    }

    #[test]
    fn runtime_override_beats_persisted_policy() {
        let persisted_policy = ConfigToolExecutionPolicyConfig {
            tool_overrides: HashMap::from([(
                "exec_command".to_string(),
                ConfigToolExecutionOverrideConfig {
                    warning_policy: Some(ConfigToolExecutionWarningPolicyConfig::None),
                    restriction_profile: Some(
                        ConfigToolExecutionRestrictionProfileConfig::WorkspacePathRequired,
                    ),
                    sandbox_profile: Some(ConfigToolExecutionSandboxProfileConfig::None),
                },
            )]),
            ..Default::default()
        };
        let request_metadata = json!({
            "harness": {
                "executionPolicy": {
                    "toolOverrides": {
                        "EXEC_COMMAND": {
                            "warningPolicy": "shell_command_risk",
                            "restrictionProfile": "workspace_shell_command",
                            "sandboxProfile": "workspace_command"
                        }
                    }
                }
            }
        });

        let resolution = service(ToolExecutionResolverInput {
            persisted_policy: Some(&persisted_policy),
            request_metadata: Some(&request_metadata),
        })
        .resolve("exec_command");

        assert_eq!(
            resolution.policy.warning_policy,
            ToolExecutionWarningPolicy::ShellCommandRisk
        );
        assert_eq!(
            resolution.warning_policy_source,
            ToolExecutionPolicySource::Runtime
        );
        assert_eq!(
            resolution.policy.restriction_profile,
            ToolExecutionRestrictionProfile::WorkspaceShellCommand
        );
        assert_eq!(
            resolution.policy.sandbox_profile,
            ToolExecutionSandboxProfile::WorkspaceCommand
        );
    }

    #[test]
    fn merges_persisted_and_runtime_shell_rules() {
        let persisted_policy = ConfigToolExecutionPolicyConfig {
            shell_command_rules: vec![ConfigToolExecutionCommandRuleConfig {
                rule_id: "persisted_publish".to_string(),
                match_type: ConfigToolExecutionCommandRuleMatchTypeConfig::Prefix,
                pattern: "cargo publish".to_string(),
                risk_level: ConfigToolExecutionCommandRiskLevelConfig::High,
                reason_code: "persisted_publish".to_string(),
                reason: "持久化策略要求确认发布".to_string(),
            }],
            ..Default::default()
        };
        let request_metadata = json!({
            "harness": {
                "requestExecutionPolicy": {
                    "shellCommandRules": [
                        {
                            "ruleId": "request_npm_publish",
                            "matchType": "prefix",
                            "pattern": "npm publish",
                            "riskLevel": "high",
                            "reasonCode": "request_npm_publish",
                            "reason": "请求策略要求确认 npm publish"
                        }
                    ]
                }
            }
        });

        let policy_service = service(ToolExecutionResolverInput {
            persisted_policy: Some(&persisted_policy),
            request_metadata: Some(&request_metadata),
        });

        let cargo_rule = policy_service
            .classify_shell_command("cargo publish --dry-run")
            .expect("persisted rule should match");
        let npm_rule = policy_service
            .classify_shell_command("npm publish --dry-run")
            .expect("request rule should match");

        assert_eq!(cargo_rule.rule_id, "persisted_publish");
        assert_eq!(cargo_rule.source.label(), "persisted");
        assert_eq!(npm_rule.rule_id, "request_npm_publish");
        assert_eq!(npm_rule.source.label(), "request");
    }

    #[test]
    fn reads_request_network_rules_from_turn_context_runtime_request() {
        let request_metadata = json!({
            "runtime_request": {
                "harness": {
                    "requestExecutionPolicy": {
                        "networkRules": [
                            {
                                "ruleId": "managed_network_gate_b",
                                "matchType": "exact",
                                "target": "host",
                                "pattern": "127.0.0.1",
                                "riskLevel": "high",
                                "reasonCode": "managed_network_gate_b",
                                "reason": "Gate B requires managed-network approval"
                            }
                        ]
                    }
                }
            }
        });

        let rule_match = service(ToolExecutionResolverInput {
            persisted_policy: None,
            request_metadata: Some(&request_metadata),
        })
        .classify_network_access(
            "exec_command",
            &json!({ "cmd": "curl http://127.0.0.1:43123/proof" }),
            Some("curl http://127.0.0.1:43123/proof"),
        )
        .expect("request network rule should cross the runtime_request wrapper");

        assert_eq!(rule_match.rule_id, "managed_network_gate_b");
        assert_eq!(rule_match.source.label(), "request");
        assert_eq!(rule_match.host.as_deref(), Some("127.0.0.1"));
    }

    #[test]
    fn persisted_policy_from_metadata_reads_agent_config_shape() {
        let metadata = json!({
            "config": {
                "nativeAgent": {
                    "toolExecution": {
                        "toolOverrides": {
                            "exec_command": {
                                "warningPolicy": "none"
                            }
                        }
                    }
                }
            }
        });

        let policy = ToolExecutionPolicyService::persisted_policy_from_metadata(Some(&metadata))
            .expect("persisted policy should be decoded");

        assert!(policy.tool_overrides.contains_key("exec_command"));
    }
}
