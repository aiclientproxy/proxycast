use app_server_protocol::ModelCapabilitiesInfo;
use app_server_protocol::ModelInfo;
use app_server_protocol::ModelReasoningEffortSupportInfo;
use app_server_protocol::ModelServiceTierInfo;
use app_server_protocol::ProviderInfo;
use app_server_protocol::ProviderKeyInfo;
use serde_json::Value;

pub(super) fn model_info_from_value(value: &Value) -> ModelInfo {
    let capabilities = value.get("capabilities");
    let reasoning_effort: Option<ModelReasoningEffortSupportInfo> = capabilities
        .and_then(|capabilities| field_value(capabilities, "reasoning_effort"))
        .and_then(|support| serde_json::from_value(support.to_owned()).ok());
    let default_reasoning_level = string_field(value, "default_reasoning_level").or_else(|| {
        reasoning_effort
            .as_ref()
            .and_then(|support| support.default.clone())
    });
    let mut supported_reasoning_levels = value_vec_field(value, "supported_reasoning_levels");
    if supported_reasoning_levels.is_empty() {
        supported_reasoning_levels = reasoning_effort
            .as_ref()
            .map(|support| {
                if support.options.is_empty() {
                    support.levels.iter().cloned().map(Value::String).collect()
                } else {
                    support
                        .options
                        .iter()
                        .filter_map(|option| serde_json::to_value(option).ok())
                        .collect()
                }
            })
            .unwrap_or_default();
    }
    ModelInfo {
        id: string_field(value, "id").unwrap_or_default(),
        provider_id: string_field(value, "provider_id").unwrap_or_default(),
        display_name: string_field(value, "display_name")
            .or_else(|| string_field(value, "id"))
            .unwrap_or_default(),
        provider_name: string_field(value, "provider_name").unwrap_or_default(),
        family: string_field(value, "family"),
        tier: string_field(value, "tier").unwrap_or_else(|| "pro".to_string()),
        capabilities: ModelCapabilitiesInfo {
            vision: bool_field(capabilities, "vision").unwrap_or(false),
            tools: bool_field(capabilities, "tools").unwrap_or(false),
            streaming: bool_field(capabilities, "streaming").unwrap_or(false),
            json_mode: bool_field(capabilities, "json_mode").unwrap_or(false),
            function_calling: bool_field(capabilities, "function_calling").unwrap_or(false),
            reasoning: bool_field(capabilities, "reasoning").unwrap_or(false),
            reasoning_effort,
        },
        task_families: string_vec_field(value, "task_families"),
        input_modalities: string_vec_field(value, "input_modalities"),
        tool_mode: string_field(value, "tool_mode"),
        supports_search_tool: bool_field(Some(value), "supports_search_tool").unwrap_or(false),
        web_search_tool_type: string_field(value, "web_search_tool_type"),
        supports_image_detail_original: bool_field(Some(value), "supports_image_detail_original")
            .unwrap_or(false),
        context_window: i64_field(value, "context_window"),
        max_context_window: i64_field(value, "max_context_window"),
        auto_compact_token_limit: i64_field(value, "auto_compact_token_limit"),
        effective_context_window_percent: i64_field(value, "effective_context_window_percent"),
        visibility: string_field(value, "visibility"),
        service_tiers: model_service_tiers_field(value, "service_tiers"),
        default_service_tier: string_field(value, "default_service_tier"),
        supports_parallel_tool_calls: bool_field(Some(value), "supports_parallel_tool_calls")
            .unwrap_or(false),
        default_reasoning_level,
        supported_reasoning_levels,
        supports_reasoning_summaries: bool_field(
            Some(value),
            "supports_reasoning_summary_parameter",
        )
        .or_else(|| bool_field(Some(value), "supports_reasoning_summaries"))
        .unwrap_or(false),
        default_reasoning_summary: string_field(value, "default_reasoning_summary"),
        support_verbosity: bool_field(Some(value), "support_verbosity").unwrap_or(false),
        default_verbosity: string_field(value, "default_verbosity"),
        use_responses_lite: bool_field(Some(value), "use_responses_lite").unwrap_or(false),
        truncation_policy: field_value(value, "truncation_policy").cloned(),
        shell_type: string_field(value, "shell_type"),
        apply_patch_tool_type: string_field(value, "apply_patch_tool_type"),
        experimental_supported_tools: string_vec_field(value, "experimental_supported_tools"),
        output_modalities: string_vec_field(value, "output_modalities"),
        runtime_features: string_vec_field(value, "runtime_features"),
        deployment_source: string_field(value, "deployment_source")
            .unwrap_or_else(|| "user_cloud".to_string()),
        management_plane: string_field(value, "management_plane")
            .unwrap_or_else(|| "local_settings".to_string()),
        canonical_model_id: string_field(value, "canonical_model_id"),
        provider_model_id: string_field(value, "provider_model_id"),
        alias_source: string_field(value, "alias_source"),
        status: string_field(value, "status").unwrap_or_else(|| "active".to_string()),
        source: string_field(value, "source").unwrap_or_else(|| "local".to_string()),
        release_date: string_field(value, "release_date"),
        is_latest: bool_field(Some(value), "is_latest").unwrap_or(false),
        description: string_field(value, "description"),
        pricing: field_value(value, "pricing")
            .cloned()
            .unwrap_or(Value::Null),
        limits: field_value(value, "limits")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({})),
        created_at: i64_field(value, "created_at").unwrap_or_default(),
        updated_at: i64_field(value, "updated_at").unwrap_or_default(),
    }
}

pub(super) fn provider_info_from_value(value: &Value) -> ProviderInfo {
    ProviderInfo {
        id: string_field(value, "id").unwrap_or_default(),
        name: string_field(value, "name")
            .or_else(|| string_field(value, "id"))
            .unwrap_or_default(),
        provider_type: string_field(value, "type")
            .or_else(|| string_field(value, "provider_type"))
            .unwrap_or_default(),
        api_host: string_field(value, "api_host").unwrap_or_default(),
        group: string_field(value, "group").unwrap_or_default(),
        enabled: bool_field(Some(value), "enabled").unwrap_or(true),
        is_system: bool_field(Some(value), "is_system").unwrap_or(false),
        sort_order: i32_field(value, "sort_order").unwrap_or_default(),
        api_version: string_field(value, "api_version"),
        project: string_field(value, "project"),
        location: string_field(value, "location"),
        region: string_field(value, "region"),
        custom_models: string_vec_field(value, "custom_models"),
        prompt_cache_mode: string_field(value, "prompt_cache_mode"),
        api_key_count: usize_field(value, "api_key_count").unwrap_or_default(),
        api_keys: field_value(value, "api_keys")
            .and_then(Value::as_array)
            .map(|items| items.iter().map(provider_key_info_from_value).collect())
            .unwrap_or_default(),
        legacy_ids: string_vec_field(value, "legacy_ids"),
        created_at: string_field(value, "created_at"),
        updated_at: string_field(value, "updated_at"),
    }
}

pub(super) fn provider_key_info_from_value(value: &Value) -> ProviderKeyInfo {
    ProviderKeyInfo {
        id: string_field(value, "id").unwrap_or_default(),
        provider_id: string_field(value, "provider_id").unwrap_or_default(),
        api_key_masked: string_field(value, "api_key_masked").unwrap_or_default(),
        alias: string_field(value, "alias"),
        enabled: bool_field(Some(value), "enabled").unwrap_or(true),
        usage_count: i64_field(value, "usage_count").unwrap_or_default(),
        error_count: i64_field(value, "error_count").unwrap_or_default(),
        last_used_at: string_field(value, "last_used_at"),
        created_at: string_field(value, "created_at").unwrap_or_default(),
    }
}

fn field_value<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    value
        .get(key)
        .or_else(|| value.get(to_camel_case(key).as_str()))
}

fn string_field(value: &Value, key: &str) -> Option<String> {
    field_value(value, key)
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn bool_field(value: Option<&Value>, key: &str) -> Option<bool> {
    value
        .and_then(|value| field_value(value, key))
        .and_then(Value::as_bool)
}

fn i32_field(value: &Value, key: &str) -> Option<i32> {
    field_value(value, key)
        .and_then(Value::as_i64)
        .and_then(|number| i32::try_from(number).ok())
}

fn i64_field(value: &Value, key: &str) -> Option<i64> {
    field_value(value, key).and_then(Value::as_i64)
}

fn usize_field(value: &Value, key: &str) -> Option<usize> {
    field_value(value, key)
        .and_then(Value::as_u64)
        .and_then(|number| usize::try_from(number).ok())
}

fn string_vec_field(value: &Value, key: &str) -> Vec<String> {
    field_value(value, key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn value_vec_field(value: &Value, key: &str) -> Vec<Value> {
    field_value(value, key)
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn model_service_tiers_field(value: &Value, key: &str) -> Vec<ModelServiceTierInfo> {
    value_vec_field(value, key)
        .into_iter()
        .filter_map(|item| match item {
            Value::String(id) => {
                let id = id.trim();
                (!id.is_empty()).then(|| ModelServiceTierInfo {
                    id: id.to_string(),
                    name: id.to_string(),
                    description: String::new(),
                })
            }
            Value::Object(fields) => {
                let id = fields
                    .get("id")
                    .or_else(|| fields.get("value"))
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|id| !id.is_empty())?;
                let name = fields
                    .get("name")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|name| !name.is_empty())
                    .unwrap_or(id);
                let description = fields
                    .get("description")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .unwrap_or_default();
                Some(ModelServiceTierInfo {
                    id: id.to_string(),
                    name: name.to_string(),
                    description: description.to_string(),
                })
            }
            _ => None,
        })
        .collect()
}

fn to_camel_case(key: &str) -> String {
    let mut result = String::new();
    let mut uppercase_next = false;
    for ch in key.chars() {
        if ch == '_' {
            uppercase_next = true;
        } else if uppercase_next {
            result.extend(ch.to_uppercase());
            uppercase_next = false;
        } else {
            result.push(ch);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::model_info_from_value;

    #[test]
    fn model_info_from_value_preserves_extended_policy_fields() {
        let value = serde_json::json!({
            "id": "gpt-5-codex",
            "providerId": "openai",
            "displayName": "GPT-5 Codex",
            "providerName": "OpenAI",
            "toolMode": "responses",
            "supportsSearchTool": true,
            "webSearchToolType": "web_search_preview",
            "supportsImageDetailOriginal": true,
            "contextWindow": 272000,
            "maxContextWindow": 400000,
            "autoCompactTokenLimit": 180000,
            "effectiveContextWindowPercent": 90,
            "visibility": "public",
            "serviceTiers": [{"id": "default"}],
            "defaultServiceTier": "default",
            "supportsParallelToolCalls": true,
            "defaultReasoningLevel": "medium",
            "supportedReasoningLevels": ["low", "medium", "high"],
            "supportsReasoningSummaryParameter": true,
            "defaultReasoningSummary": "auto",
            "supportVerbosity": true,
            "defaultVerbosity": "medium",
            "useResponsesLite": true,
            "truncationPolicy": {"type": "auto"},
            "shellType": "bash",
            "applyPatchToolType": "apply_patch",
            "experimentalSupportedTools": ["web_search"],
            "deploymentSource": "user_cloud",
            "managementPlane": "local_settings",
            "status": "active",
            "source": "local",
            "pricing": null,
            "limits": {},
            "createdAt": 1,
            "updatedAt": 2
        });

        let info = model_info_from_value(&value);

        assert_eq!(info.provider_id, "openai");
        assert_eq!(info.tool_mode.as_deref(), Some("responses"));
        assert!(info.supports_search_tool);
        assert_eq!(
            info.web_search_tool_type.as_deref(),
            Some("web_search_preview")
        );
        assert!(info.supports_image_detail_original);
        assert_eq!(info.context_window, Some(272000));
        assert_eq!(info.max_context_window, Some(400000));
        assert_eq!(info.auto_compact_token_limit, Some(180000));
        assert_eq!(info.effective_context_window_percent, Some(90));
        assert_eq!(info.visibility.as_deref(), Some("public"));
        assert_eq!(info.service_tiers.len(), 1);
        assert_eq!(info.service_tiers[0].id, "default");
        assert_eq!(info.service_tiers[0].name, "default");
        assert_eq!(info.service_tiers[0].description, "");
        assert_eq!(info.default_service_tier.as_deref(), Some("default"));
        assert!(info.supports_parallel_tool_calls);
        assert_eq!(info.default_reasoning_level.as_deref(), Some("medium"));
        assert_eq!(info.supported_reasoning_levels.len(), 3);
        assert!(info.supports_reasoning_summaries);
        assert_eq!(info.default_reasoning_summary.as_deref(), Some("auto"));
        assert!(info.support_verbosity);
        assert_eq!(info.default_verbosity.as_deref(), Some("medium"));
        assert!(info.use_responses_lite);
        assert!(info.truncation_policy.is_some());
        assert_eq!(info.shell_type.as_deref(), Some("bash"));
        assert_eq!(info.apply_patch_tool_type.as_deref(), Some("apply_patch"));
        assert_eq!(info.experimental_supported_tools, vec!["web_search"]);
    }

    #[test]
    fn model_info_from_value_projects_reasoning_menu_from_capability_fact() {
        let value = serde_json::json!({
            "id": "grok-4.5",
            "providerId": "xai",
            "displayName": "Grok 4.5",
            "providerName": "xAI",
            "capabilities": {
                "reasoning": true,
                "reasoningEffort": {
                    "supported": true,
                    "levels": ["xhigh", "medium"],
                    "options": [
                        {
                            "id": "deep",
                            "value": "xhigh",
                            "label": "Deep",
                            "description": "Maximum reasoning",
                            "default": true
                        },
                        {
                            "id": "balanced",
                            "value": "medium",
                            "label": "Balanced",
                            "default": false
                        }
                    ],
                    "default": "xhigh",
                    "source": "api"
                }
            },
            "deploymentSource": "user_cloud",
            "managementPlane": "local_settings",
            "status": "active",
            "source": "api",
            "pricing": null,
            "limits": {},
            "createdAt": 1,
            "updatedAt": 2
        });

        let info = model_info_from_value(&value);

        let support = info
            .capabilities
            .reasoning_effort
            .as_ref()
            .expect("typed reasoning effort support");
        assert!(support.supported);
        assert_eq!(support.default.as_deref(), Some("xhigh"));
        assert_eq!(support.source.as_deref(), Some("api"));
        assert_eq!(support.options[0].id, "deep");
        assert_eq!(support.options[0].value, "xhigh");
        assert_eq!(support.options[0].label, "Deep");
        assert_eq!(info.default_reasoning_level.as_deref(), Some("xhigh"));
        assert_eq!(
            info.supported_reasoning_levels,
            vec![
                serde_json::json!({
                    "id": "deep",
                    "value": "xhigh",
                    "label": "Deep",
                    "description": "Maximum reasoning",
                    "default": true
                }),
                serde_json::json!({
                    "id": "balanced",
                    "value": "medium",
                    "label": "Balanced",
                    "default": false
                })
            ]
        );
    }
}
