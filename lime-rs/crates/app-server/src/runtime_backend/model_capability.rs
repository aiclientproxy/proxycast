use app_server_protocol::{CapabilitySnapshot, ModelReasoningEffortSupportInfo};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[cfg(test)]
use serde_json::Map;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelRef {
    pub provider_id: String,
    pub model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCapability {
    pub model: ModelRef,
    pub supports_tool_calling: bool,
    pub supports_tool_streaming: bool,
    pub supports_reasoning: bool,
    pub supports_reasoning_summary: bool,
    pub supported_reasoning_levels: Vec<String>,
    pub plan_strategy: PlanStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanStrategy {
    UpdatePlan,
    ProposedPlan,
    Hybrid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReasoningPolicy {
    pub supported: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_level: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_level: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub downgrade_reason: Option<String>,
}

impl ModelRef {
    pub fn new(provider_id: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self {
            provider_id: provider_id.into(),
            model_id: model_id.into(),
            variant: None,
        }
    }
}

fn unknown_model_capability(model: ModelRef) -> ModelCapability {
    ModelCapability {
        model,
        supports_tool_calling: false,
        supports_tool_streaming: false,
        supports_reasoning: false,
        supports_reasoning_summary: false,
        supported_reasoning_levels: Vec::new(),
        plan_strategy: PlanStrategy::Hybrid,
    }
}

pub fn resolve_model_capability(
    model: ModelRef,
    capability_snapshot: Option<&CapabilitySnapshot>,
) -> ModelCapability {
    match capability_snapshot {
        Some(snapshot) => resolve_model_capability_from_snapshot(model, snapshot),
        None => unknown_model_capability(model),
    }
}

pub fn resolve_model_capability_from_snapshot(
    model: ModelRef,
    snapshot: &CapabilitySnapshot,
) -> ModelCapability {
    let explicit_reasoning_support =
        reasoning_effort_supported(snapshot.capabilities.reasoning_effort.as_ref());
    let supports_reasoning = explicit_reasoning_support.unwrap_or_else(|| {
        snapshot.capabilities.reasoning
            || has_case_insensitive_value(&snapshot.runtime_features, "reasoning")
            || has_case_insensitive_value(&snapshot.task_families, "reasoning")
    });
    let supported_reasoning_levels = if supports_reasoning {
        reasoning_levels_from_snapshot(snapshot)
    } else {
        Vec::new()
    };
    let supports_tool_calling = snapshot.capabilities.tools
        || snapshot.capabilities.function_calling
        || has_case_insensitive_value(&snapshot.runtime_features, "tool_calling")
        || has_case_insensitive_value(&snapshot.runtime_features, "tools");
    let supports_tool_streaming = snapshot.capabilities.streaming
        || has_case_insensitive_value(&snapshot.runtime_features, "streaming");

    ModelCapability {
        model,
        supports_tool_calling,
        supports_tool_streaming,
        supports_reasoning,
        supports_reasoning_summary: supports_reasoning,
        supported_reasoning_levels,
        plan_strategy: PlanStrategy::Hybrid,
    }
}

pub fn resolve_reasoning_policy(
    capability: &ModelCapability,
    requested_level: Option<&str>,
) -> ReasoningPolicy {
    let requested_level = requested_level.and_then(non_empty);
    if !capability.supports_reasoning {
        return ReasoningPolicy {
            supported: false,
            effective_level: None,
            downgrade_reason: requested_level
                .as_ref()
                .map(|_| "selected model does not support reasoning".to_string()),
            requested_level,
        };
    }

    let effective_level = requested_level.as_ref().and_then(|requested| {
        capability
            .supported_reasoning_levels
            .iter()
            .find(|supported| *supported == requested)
            .cloned()
    });
    let downgrade_reason = requested_level
        .as_ref()
        .filter(|_| effective_level.is_none())
        .map(|_| "requested reasoning level is not advertised by selected model".to_string());

    ReasoningPolicy {
        supported: true,
        requested_level,
        effective_level,
        downgrade_reason,
    }
}

pub fn model_effective_payload(capability: &ModelCapability, policy: &ReasoningPolicy) -> Value {
    json!({
        "model": capability.model,
        "modelRef": capability.model,
        "capability": capability,
        "reasoning": policy,
        "toolCalling": {
            "supported": capability.supports_tool_calling,
            "streaming": capability.supports_tool_streaming,
        },
    })
}

fn has_case_insensitive_value(values: &[String], expected: &str) -> bool {
    values
        .iter()
        .any(|value| value.trim().eq_ignore_ascii_case(expected))
}

fn reasoning_effort_supported(value: Option<&ModelReasoningEffortSupportInfo>) -> Option<bool> {
    value.map(|support| support.supported)
}

fn reasoning_levels_from_snapshot(snapshot: &CapabilitySnapshot) -> Vec<String> {
    let Some(support) = snapshot.capabilities.reasoning_effort.as_ref() else {
        return Vec::new();
    };
    let mut levels = Vec::new();
    for value in support
        .levels
        .iter()
        .map(String::as_str)
        .chain(support.options.iter().map(|option| option.value.as_str()))
    {
        let Some(value) = non_empty(value) else {
            continue;
        };
        if !levels.contains(&value) {
            levels.push(value);
        }
    }
    levels
}

fn non_empty(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_string())
}

#[cfg(test)]
fn provider_request_options_skeleton(policy: &ReasoningPolicy) -> Value {
    let mut options = Map::new();
    if let Some(level) = policy.effective_level.as_ref() {
        options.insert("reasoningLevel".to_string(), json!(level));
    }
    if let Some(reason) = policy.downgrade_reason.as_ref() {
        options.insert("downgradeReason".to_string(), json!(reason));
    }
    Value::Object(options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::ModelReasoningEffortOptionInfo;

    #[test]
    fn model_names_never_infer_reasoning_without_snapshot() {
        for (provider, model) in [
            ("openai", "gpt-codex"),
            ("anthropic", "claude-sonnet-4-5"),
            ("google", "gemini-2.5-pro"),
            ("openai-compatible", "o3-mini"),
        ] {
            let capability = resolve_model_capability(ModelRef::new(provider, model), None);
            let policy = resolve_reasoning_policy(&capability, Some("high"));

            assert!(!capability.supports_reasoning, "{provider}/{model}");
            assert!(!policy.supported, "{provider}/{model}");
            assert_eq!(policy.effective_level, None, "{provider}/{model}");
        }
    }

    #[test]
    fn capability_snapshot_keeps_declared_canonical_values_in_provider_order() {
        let snapshot = reasoning_snapshot(
            &["minimal", "high", "ultra", "high", ""],
            &[("deep", "xhigh"), ("future", "provider-native")],
        );
        let capability = resolve_model_capability(
            ModelRef::new("custom-provider", "plain-chat"),
            Some(&snapshot),
        );

        assert_eq!(
            capability.supported_reasoning_levels,
            vec!["minimal", "high", "ultra", "xhigh", "provider-native"]
        );
        assert_eq!(
            resolve_reasoning_policy(&capability, Some("minimal")).effective_level,
            Some("minimal".to_string())
        );
        assert_eq!(
            resolve_reasoning_policy(&capability, Some("ultra")).effective_level,
            Some("ultra".to_string())
        );
        assert_eq!(
            resolve_reasoning_policy(&capability, Some("provider-native")).effective_level,
            Some("provider-native".to_string())
        );
    }

    #[test]
    fn unsupported_effort_is_omitted_instead_of_silently_switching_levels() {
        let snapshot = reasoning_snapshot(&["low", "high"], &[]);
        let capability =
            resolve_model_capability(ModelRef::new("openai", "gpt-codex"), Some(&snapshot));
        let policy = resolve_reasoning_policy(&capability, Some("medium"));

        assert!(policy.supported);
        assert_eq!(policy.requested_level.as_deref(), Some("medium"));
        assert_eq!(policy.effective_level, None);
        assert_eq!(
            policy.downgrade_reason.as_deref(),
            Some("requested reasoning level is not advertised by selected model")
        );
        assert!(provider_request_options_skeleton(&policy)
            .get("reasoningLevel")
            .is_none());
    }

    #[test]
    fn explicit_reasoning_disable_overrides_reasoning_named_features() {
        let snapshot = CapabilitySnapshot {
            runtime_features: vec!["reasoning".to_string(), "streaming".to_string()],
            capabilities: app_server_protocol::ModelCapabilitiesInfo {
                reasoning: true,
                reasoning_effort: Some(ModelReasoningEffortSupportInfo {
                    supported: false,
                    levels: vec!["high".to_string()],
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
        };
        let capability =
            resolve_model_capability(ModelRef::new("openai", "gpt-codex"), Some(&snapshot));

        assert!(!capability.supports_reasoning);
        assert!(capability.supported_reasoning_levels.is_empty());
        assert_eq!(
            resolve_reasoning_policy(&capability, Some("high")).effective_level,
            None
        );
    }

    #[test]
    fn missing_request_does_not_invent_default_effort() {
        let snapshot = reasoning_snapshot(&["low", "medium", "high"], &[]);
        let capability =
            resolve_model_capability(ModelRef::new("openai", "gpt-codex"), Some(&snapshot));
        let policy = resolve_reasoning_policy(&capability, None);

        assert!(policy.supported);
        assert_eq!(policy.requested_level, None);
        assert_eq!(policy.effective_level, None);
    }

    #[test]
    fn provider_options_keep_custom_wire_value_stable() {
        let policy = ReasoningPolicy {
            supported: true,
            requested_level: Some("ultra".to_string()),
            effective_level: Some("ultra".to_string()),
            downgrade_reason: None,
        };

        assert_eq!(
            provider_request_options_skeleton(&policy)["reasoningLevel"],
            "ultra"
        );
    }

    fn reasoning_snapshot(levels: &[&str], options: &[(&str, &str)]) -> CapabilitySnapshot {
        CapabilitySnapshot {
            runtime_features: vec!["reasoning".to_string(), "streaming".to_string()],
            capabilities: app_server_protocol::ModelCapabilitiesInfo {
                tools: true,
                streaming: true,
                reasoning: true,
                reasoning_effort: Some(ModelReasoningEffortSupportInfo {
                    supported: true,
                    levels: levels.iter().map(|value| (*value).to_string()).collect(),
                    options: options
                        .iter()
                        .map(|(id, value)| ModelReasoningEffortOptionInfo {
                            id: (*id).to_string(),
                            value: (*value).to_string(),
                            label: (*id).to_string(),
                            ..Default::default()
                        })
                        .collect(),
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
        }
    }
}
