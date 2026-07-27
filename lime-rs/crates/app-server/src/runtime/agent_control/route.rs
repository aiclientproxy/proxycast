use super::trimmed_option;
use app_server_protocol::{
    AgentSession, AuthKind, ProtocolKind, RuntimeOptions, RuntimeProviderConfig, RuntimeRequest,
};
use serde_json::json;
use std::collections::HashSet;

pub(super) const AGENT_CONTROL_ROUTE_KEY: &str = "agentControlRoute";
const AGENT_CONTROL_ROUTE_SCHEMA_VERSION: u64 = 2;

pub(super) fn agent_control_route_snapshot(
    runtime_options: Option<&RuntimeOptions>,
) -> Option<serde_json::Value> {
    let request = runtime_options?.runtime_request.as_ref()?;
    request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get(AGENT_CONTROL_ROUTE_KEY))
        .and_then(normalize_route_snapshot)
}

fn normalize_route_snapshot(value: &serde_json::Value) -> Option<serde_json::Value> {
    let object = value.as_object()?;
    const TOP_LEVEL_KEYS: &[&str] = &[
        "schemaVersion",
        "routeSource",
        "providerPreference",
        "modelPreference",
        "serviceTier",
        "providerConfig",
        "routeProtocol",
        "authKind",
        "credentialRef",
        "effectiveGeneration",
        "modelRegistry",
    ];
    if object
        .keys()
        .any(|key| !TOP_LEVEL_KEYS.contains(&key.as_str()))
    {
        return None;
    }
    let schema_version = object.get("schemaVersion")?.as_u64()?;
    if schema_version != AGENT_CONTROL_ROUTE_SCHEMA_VERSION {
        return None;
    }
    let route_source = route_string(object, "routeSource")
        .filter(|source| matches!(source.as_str(), "catalog" | "direct_provider_config"));
    let provider_config = object.get("providerConfig")?.as_object()?;
    const PROVIDER_CONFIG_KEYS: &[&str] = &[
        "providerId",
        "providerName",
        "modelName",
        "reasoningEffort",
        "toolshim",
        "toolshimModel",
        "supportsWebsockets",
    ];
    if provider_config
        .keys()
        .any(|key| !PROVIDER_CONFIG_KEYS.contains(&key.as_str()))
    {
        return None;
    }
    let provider_id = route_string(provider_config, "providerId")?;
    let model_name = route_string(provider_config, "modelName")?;
    let provider_name =
        route_string(provider_config, "providerName").unwrap_or_else(|| provider_id.clone());
    let route_protocol: ProtocolKind =
        serde_json::from_value(object.get("routeProtocol")?.clone()).ok()?;
    let auth_kind: AuthKind = object
        .get("authKind")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .ok()?
        .unwrap_or(AuthKind::ApiKeyRef);
    let effective_generation = object.get("effectiveGeneration")?.as_u64()?;
    let credential_ref = object
        .get("credentialRef")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string);
    let model_registry = normalize_model_registry_snapshot(object.get("modelRegistry"));
    Some(json!({
        "schemaVersion": AGENT_CONTROL_ROUTE_SCHEMA_VERSION,
        "routeSource": route_source,
        "providerPreference": route_string(object, "providerPreference")
            .unwrap_or_else(|| provider_id.clone()),
        "modelPreference": route_string(object, "modelPreference")
            .unwrap_or_else(|| model_name.clone()),
        "serviceTier": route_string(object, "serviceTier"),
        "providerConfig": {
            "providerId": provider_id,
            "providerName": provider_name,
            "modelName": model_name,
            "reasoningEffort": route_string(provider_config, "reasoningEffort"),
            "toolshim": provider_config.get("toolshim").and_then(serde_json::Value::as_bool),
            "toolshimModel": route_string(provider_config, "toolshimModel"),
            "supportsWebsockets": provider_config
                .get("supportsWebsockets")
                .and_then(serde_json::Value::as_bool)
        },
        "routeProtocol": route_protocol,
        "authKind": auth_kind,
        "credentialRef": credential_ref,
        "effectiveGeneration": effective_generation,
        "modelRegistry": model_registry
    }))
}

fn normalize_model_registry_snapshot(
    value: Option<&serde_json::Value>,
) -> Option<serde_json::Value> {
    let registry = value?.as_object()?;
    if registry.get("status").and_then(serde_json::Value::as_str) != Some("matched") {
        return None;
    }
    let model = registry.get("model")?.as_object()?;
    let mut seen = HashSet::new();
    let service_tiers = model
        .get("service_tiers")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|tier| {
            let tier = tier.as_object()?;
            let id = route_string(tier, "id")?;
            if !seen.insert(id.clone()) {
                return None;
            }
            let name = route_string(tier, "name").unwrap_or_else(|| id.clone());
            let description = route_string(tier, "description").unwrap_or_default();
            Some(json!({
                "id": id,
                "name": name,
                "description": description,
            }))
        })
        .collect::<Vec<_>>();
    let default_service_tier =
        route_string(model, "default_service_tier").filter(|default| seen.contains(default));

    Some(json!({
        "status": "matched",
        "model": {
            "service_tiers": service_tiers,
            "default_service_tier": default_service_tier,
        }
    }))
}

pub(super) fn has_complete_agent_control_route_snapshot(
    runtime_options: Option<&RuntimeOptions>,
) -> bool {
    runtime_options
        .and_then(|options| agent_control_route_snapshot(Some(options)))
        .and_then(|snapshot| normalize_route_snapshot(&snapshot))
        .is_some_and(|snapshot| {
            snapshot
                .get("schemaVersion")
                .and_then(serde_json::Value::as_u64)
                == Some(AGENT_CONTROL_ROUTE_SCHEMA_VERSION)
        })
}

pub(super) fn agent_control_route_snapshot_is_direct(route_snapshot: &serde_json::Value) -> bool {
    normalize_route_snapshot(route_snapshot).is_some_and(|snapshot| {
        snapshot
            .get("routeSource")
            .and_then(serde_json::Value::as_str)
            == Some("direct_provider_config")
    })
}

pub(super) fn agent_control_route_snapshot_from_session(
    session: &AgentSession,
) -> Option<&serde_json::Value> {
    session
        .business_object_ref
        .as_ref()
        .and_then(|reference| reference.metadata.as_ref())
        .and_then(|metadata| metadata.get(AGENT_CONTROL_ROUTE_KEY))
}

pub(super) fn runtime_provider_config_from_route_snapshot(
    route_snapshot: &serde_json::Value,
) -> Option<RuntimeProviderConfig> {
    let config = route_snapshot
        .get("providerConfig")
        .and_then(serde_json::Value::as_object)?;
    Some(RuntimeProviderConfig {
        provider_id: route_string(config, "providerId"),
        provider_name: route_string(config, "providerName"),
        model_name: route_string(config, "modelName"),
        api_key: None,
        base_url: None,
        tool_call_strategy: None,
        toolshim_model: route_string(config, "toolshimModel"),
        model_capabilities: None,
        supports_websockets: config
            .get("supportsWebsockets")
            .and_then(serde_json::Value::as_bool),
    })
}

fn route_string(object: &serde_json::Map<String, serde_json::Value>, key: &str) -> Option<String> {
    trimmed_option(object.get(key).and_then(serde_json::Value::as_str))
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::{RuntimeOptions, RuntimeProviderConfig, RuntimeRequest};

    #[test]
    fn route_snapshot_redacts_direct_secret_and_endpoint() {
        let options = RuntimeOptions {
            runtime_request: Some(RuntimeRequest {
                provider_preference: Some("provider-id".to_string()),
                model_preference: Some("model-a".to_string()),
                provider_config: Some(RuntimeProviderConfig {
                    provider_id: Some("provider-id".to_string()),
                    provider_name: Some("fixture".to_string()),
                    model_name: Some("model-a".to_string()),
                    api_key: Some("route-secret".to_string()),
                    base_url: Some("https://user:token@example.test/v1?secret=1".to_string()),
                    ..RuntimeProviderConfig::default()
                }),
                metadata: Some(json!({
                    "agentControlRoute": {
                        "schemaVersion": AGENT_CONTROL_ROUTE_SCHEMA_VERSION,
                        "providerPreference": "provider-id",
                        "modelPreference": "model-a",
                        "providerConfig": {
                            "providerId": "provider-id",
                            "providerName": "fixture",
                            "modelName": "model-a"
                        },
                        "routeProtocol": "openai_responses",
                        "authKind": "api_key_ref",
                        "credentialRef": "credential-1",
                        "effectiveGeneration": 7,
                        "modelRegistry": {
                            "status": "matched",
                            "endpoint": "https://user:token@example.test/v1?secret=1",
                            "model": {
                                "service_tiers": [
                                    {
                                        "id": "priority",
                                        "name": "Priority",
                                        "description": "Fast queue",
                                        "apiKey": "catalog-secret"
                                    },
                                    { "id": "priority", "name": "Duplicate" },
                                    { "name": "Missing id" }
                                ],
                                "default_service_tier": "priority",
                                "base_url": "https://catalog-secret.example.test"
                            }
                        }
                    }
                })),
                ..RuntimeRequest::default()
            }),
            ..RuntimeOptions::default()
        };

        let snapshot = agent_control_route_snapshot(Some(&options)).expect("route snapshot");
        let encoded = snapshot.to_string();
        assert!(!encoded.contains("route-secret"));
        assert!(!encoded.contains("example.test"));
        assert!(!encoded.contains("baseUrl"));
        assert!(!encoded.contains("apiKeyPresent"));
        assert!(!encoded.contains("catalog-secret"));
        assert_eq!(
            snapshot["modelRegistry"],
            json!({
                "status": "matched",
                "model": {
                    "service_tiers": [{
                        "id": "priority",
                        "name": "Priority",
                        "description": "Fast queue"
                    }],
                    "default_service_tier": "priority"
                }
            })
        );
    }

    #[test]
    fn implicit_provider_config_is_not_promoted_to_durable_route() {
        let options = RuntimeOptions {
            runtime_request: Some(RuntimeRequest {
                provider_preference: Some("provider-id".to_string()),
                model_preference: Some("model-a".to_string()),
                provider_config: Some(RuntimeProviderConfig {
                    provider_id: Some("provider-id".to_string()),
                    provider_name: Some("fixture".to_string()),
                    model_name: Some("model-a".to_string()),
                    model_capabilities: Some(json!({
                        "baseUrl": "https://user:token@example.test/v1?secret=1"
                    })),
                    ..RuntimeProviderConfig::default()
                }),
                ..RuntimeRequest::default()
            }),
            ..RuntimeOptions::default()
        };

        assert!(agent_control_route_snapshot(Some(&options)).is_none());
    }

    #[test]
    fn durable_route_fields_win_over_explicit_provider_config() {
        let stored = RuntimeProviderConfig {
            provider_id: Some("durable-provider".to_string()),
            provider_name: Some("durable".to_string()),
            model_name: Some("durable-model".to_string()),
            base_url: None,
            ..RuntimeProviderConfig::default()
        };
        let explicit = RuntimeProviderConfig {
            provider_id: Some("explicit-provider".to_string()),
            provider_name: Some("explicit".to_string()),
            model_name: Some("explicit-model".to_string()),
            api_key: Some("must-not-restore".to_string()),
            base_url: Some("https://explicit.invalid".to_string()),
            ..RuntimeProviderConfig::default()
        };
        let merged = merge_runtime_provider_config(stored, Some(explicit));
        assert_eq!(merged.provider_id.as_deref(), Some("durable-provider"));
        assert_eq!(merged.provider_name.as_deref(), Some("durable"));
        assert_eq!(merged.model_name.as_deref(), Some("durable-model"));
        assert!(merged.api_key.is_none());
        assert!(merged.base_url.is_none());
    }

    #[test]
    fn incomplete_route_snapshot_is_not_admitted_as_durable_route() {
        let options = RuntimeOptions {
            runtime_request: Some(RuntimeRequest {
                metadata: Some(json!({
                    "agentControlRoute": {
                        "schemaVersion": AGENT_CONTROL_ROUTE_SCHEMA_VERSION,
                        "providerPreference": "provider-id"
                    }
                })),
                ..RuntimeRequest::default()
            }),
            ..RuntimeOptions::default()
        };
        assert!(!has_complete_agent_control_route_snapshot(Some(&options)));
    }
}

pub(super) fn merge_runtime_provider_config(
    stored: RuntimeProviderConfig,
    explicit: Option<RuntimeProviderConfig>,
) -> RuntimeProviderConfig {
    let Some(explicit) = explicit else {
        return stored;
    };
    RuntimeProviderConfig {
        provider_id: stored.provider_id.or(explicit.provider_id),
        provider_name: stored.provider_name.or(explicit.provider_name),
        model_name: stored.model_name.or(explicit.model_name),
        api_key: None,
        base_url: stored.base_url,
        tool_call_strategy: stored.tool_call_strategy.or(explicit.tool_call_strategy),
        toolshim_model: stored.toolshim_model.or(explicit.toolshim_model),
        model_capabilities: stored.model_capabilities.or(explicit.model_capabilities),
        supports_websockets: stored.supports_websockets.or(explicit.supports_websockets),
    }
}

pub(super) fn restore_agent_control_route_metadata(
    request: &mut RuntimeRequest,
    route_snapshot: &serde_json::Value,
) {
    if let Some(normalized) = normalize_route_snapshot(route_snapshot) {
        let metadata = request
            .metadata
            .get_or_insert_with(|| serde_json::Value::Object(Default::default()));
        if !metadata.is_object() {
            *metadata = serde_json::Value::Object(Default::default());
        }
        metadata
            .as_object_mut()
            .expect("runtime metadata object")
            .insert(AGENT_CONTROL_ROUTE_KEY.to_string(), normalized);
    }
}
