use super::request_context::RuntimeModelSelection;
use lime_agent::SessionProviderConfig;
use lime_core::database::DbConnection;
use lime_core::models::model_registry::{EnhancedModelMetadata, ModelCapabilityProvenance};
use lime_core::models::RuntimeProviderCredential;
use lime_services::api_key_provider_service::ApiKeyProviderService;
use lime_services::model_registry_service::{ModelRegistryService, ProviderModelCacheAccess};
use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub(super) struct RuntimeModelRegistryMetadata {
    payload: Value,
}

impl RuntimeModelRegistryMetadata {
    #[cfg(test)]
    pub(super) fn from_payload(payload: Value) -> Self {
        Self { payload }
    }

    pub(super) fn payload(&self) -> &Value {
        &self.payload
    }
}

pub(super) async fn resolve_runtime_model_registry_metadata(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
    selection: &RuntimeModelSelection,
    direct_provider_config: Option<&SessionProviderConfig>,
    route_credential: Option<&RuntimeProviderCredential>,
) -> Result<RuntimeModelRegistryMetadata, String> {
    if let Some(config) = direct_provider_config {
        let model_capabilities = config.model_capabilities.as_ref().map(|value| {
            let snapshot = runtime_core::capability_snapshot_from_model_capabilities(value);
            capability_payload(&snapshot, ModelCapabilityProvenance::ProviderExplicit)
        });
        return Ok(RuntimeModelRegistryMetadata {
            payload: json!({
                "source": "direct_provider_config",
                "sourceLabel": "direct_provider_config",
                "source_label": "direct_provider_config",
                "status": "runtime_selection_only",
                "reasonCode": "direct_provider_config_not_in_registry",
                "reason_code": "direct_provider_config_not_in_registry",
                "providerId": selection.provider,
                "provider_id": selection.provider,
                "requestedModelId": selection.model,
                "requested_model_id": selection.model,
                "matchedModelId": null,
                "matched_model_id": null,
                "model": null,
                "modelCapabilities": model_capabilities.clone(),
                "model_capabilities": model_capabilities,
                "modelAlias": null,
                "model_alias": null,
                "reasoning": null,
            }),
        });
    }

    let provider = api_key_provider_service.get_provider(db, &selection.provider)?;
    let cache_access = match (provider.as_ref(), route_credential) {
        (_, Some(credential)) => ProviderModelCacheAccess::Credential(credential),
        (Some(provider), None)
            if !ModelRegistryService::requires_api_key_for_runtime(
                &provider.provider.id,
                &provider.provider.api_host,
                provider.provider.effective_provider_type(),
            ) =>
        {
            ProviderModelCacheAccess::Keyless
        }
        _ => ProviderModelCacheAccess::Unavailable,
    };
    let registry = ModelRegistryService::new(db.clone());
    let metadata = registry.resolve_provider_model_metadata(
        provider.as_ref(),
        &selection.provider,
        &selection.model,
        cache_access,
    )?;
    let model = metadata
        .model
        .as_ref()
        .map(serde_json::to_value)
        .transpose()
        .map_err(|error| format!("序列化模型注册 metadata 失败: {error}"))?;
    let model_capabilities = metadata.model.as_ref().map(model_capability_payload);
    let model_alias = metadata.model.as_ref().map(|model| {
        json!({
            "canonicalModelId": model.canonical_model_id,
            "canonical_model_id": model.canonical_model_id,
            "providerModelId": model.provider_model_id,
            "provider_model_id": model.provider_model_id,
            "aliasSource": model.alias_source,
            "alias_source": model.alias_source,
        })
    });
    let reasoning = metadata.model.as_ref().map(|model| {
        json!({
            "supported": model.capabilities.reasoning,
            "reasoningEffort": model.capabilities.reasoning_effort,
            "reasoning_effort": model.capabilities.reasoning_effort,
        })
    });

    Ok(RuntimeModelRegistryMetadata {
        payload: json!({
            "source": metadata.source.as_str(),
            "sourceLabel": metadata.source.as_str(),
            "source_label": metadata.source.as_str(),
            "status": if metadata.model.is_some() { "matched" } else { "missing" },
            "reasonCode": metadata.reason_code,
            "reason_code": metadata.reason_code,
            "providerId": metadata.provider_id,
            "provider_id": metadata.provider_id,
            "requestedModelId": metadata.requested_model_id,
            "requested_model_id": metadata.requested_model_id,
            "matchedModelId": metadata.matched_model_id,
            "matched_model_id": metadata.matched_model_id,
            "cachedModelCount": metadata.cached_model_count,
            "cached_model_count": metadata.cached_model_count,
            "fromCache": metadata.from_cache,
            "from_cache": metadata.from_cache,
            "providerDeclaredModel": metadata.provider_declared_model,
            "provider_declared_model": metadata.provider_declared_model,
            "model": model,
            "modelCapabilities": model_capabilities,
            "model_capabilities": model_capabilities,
            "modelAlias": model_alias,
            "model_alias": model_alias,
            "reasoning": reasoning,
        }),
    })
}

fn model_capability_payload(model: &EnhancedModelMetadata) -> Value {
    json!({
        "provenance": model.capability_provenance,
        "capabilities": model.capabilities,
        "taskFamilies": model.task_families,
        "task_families": model.task_families,
        "runtimeFeatures": model.runtime_features,
        "runtime_features": model.runtime_features,
        "inputModalities": model.input_modalities,
        "input_modalities": model.input_modalities,
        "outputModalities": model.output_modalities,
        "output_modalities": model.output_modalities,
    })
}

fn capability_payload(
    snapshot: &app_server_protocol::CapabilitySnapshot,
    provenance: ModelCapabilityProvenance,
) -> Value {
    json!({
        "provenance": provenance,
        "capabilities": snapshot.capabilities,
        "taskFamilies": snapshot.task_families,
        "runtimeFeatures": snapshot.runtime_features,
        "inputModalities": snapshot.input_modalities,
        "outputModalities": snapshot.output_modalities,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_backend::request_context::RuntimeModelSelection;
    use lime_core::database::dao::api_key_provider::ApiProviderType;
    use lime_core::database::schema::create_tables;
    use rusqlite::Connection;
    use std::sync::{Arc, Mutex};

    fn test_db() -> lime_core::database::DbConnection {
        let conn = Connection::open_in_memory().expect("open in-memory db");
        create_tables(&conn).expect("create schema");
        Arc::new(Mutex::new(conn))
    }

    #[tokio::test]
    async fn custom_provider_declared_model_does_not_infer_reasoning_from_name() {
        let db = test_db();
        let provider_service = ApiKeyProviderService::new();
        let provider = provider_service
            .add_custom_provider(
                &db,
                "Coding Gateway".to_string(),
                ApiProviderType::Openai,
                "https://gateway.example.com/v1".to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .expect("create provider");
        provider_service
            .update_provider(
                &db,
                &provider.id,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(vec![
                    lime_core::models::model_registry::ProviderModelConfig::hint(
                        "coder-reasoning-large",
                    ),
                ]),
            )
            .expect("set custom models");

        let metadata = resolve_runtime_model_registry_metadata(
            &db,
            &provider_service,
            &RuntimeModelSelection {
                provider: provider.id.clone(),
                model: "coder-reasoning-large".to_string(),
                source: "profile_model_slot",
                reasoning_effort: None,
            },
            None,
            None,
        )
        .await
        .expect("metadata");

        assert_eq!(
            metadata.payload()["source"].as_str(),
            Some("provider_declared_model")
        );
        assert_eq!(metadata.payload()["status"].as_str(), Some("matched"));
        assert_eq!(
            metadata
                .payload()
                .pointer("/modelCapabilities/provenance")
                .and_then(Value::as_str),
            Some("inferred_hint")
        );
        assert_eq!(
            metadata
                .payload()
                .pointer("/modelCapabilities/capabilities/reasoning")
                .and_then(Value::as_bool),
            Some(false)
        );
        assert!(metadata.payload()["reasoning"]["reasoningEffort"].is_null());
        assert_eq!(
            metadata
                .payload()
                .pointer("/modelAlias/providerModelId")
                .and_then(Value::as_str),
            Some("coder-reasoning-large")
        );
    }

    #[tokio::test]
    async fn direct_provider_config_is_marked_runtime_selection_only() {
        let db = test_db();
        let provider_service = ApiKeyProviderService::new();

        let metadata = resolve_runtime_model_registry_metadata(
            &db,
            &provider_service,
            &RuntimeModelSelection {
                provider: "fixture-openai".to_string(),
                model: "fixture-model".to_string(),
                source: "runtime_request_provider_config",
                reasoning_effort: None,
            },
            Some(&SessionProviderConfig {
                provider_name: "openai".to_string(),
                provider_selector: Some("fixture-openai".to_string()),
                model_name: "fixture-model".to_string(),
                api_key: Some("fixture-key".to_string()),
                base_url: Some("http://127.0.0.1:56599".to_string()),
                credential_uuid: None,
                reasoning_effort: None,
                service_tier: None,
                route_protocol: None,
                toolshim: false,
                toolshim_model: None,
                model_capabilities: Some(json!({
                    "taskFamilies": ["chat"],
                    "runtimeFeatures": ["streaming"],
                    "capabilities": {
                        "streaming": true,
                        "apiKey": "must-not-persist"
                    },
                    "baseUrl": "https://user:token@example.test/v1?secret=1"
                })),
                supports_websockets: false,
            }),
            None,
        )
        .await
        .expect("metadata");

        assert_eq!(
            metadata.payload()["source"].as_str(),
            Some("direct_provider_config")
        );
        assert_eq!(
            metadata.payload()["reasonCode"].as_str(),
            Some("direct_provider_config_not_in_registry")
        );
        assert_eq!(
            metadata
                .payload()
                .pointer("/modelCapabilities/provenance")
                .and_then(Value::as_str),
            Some("provider_explicit")
        );
        assert!(metadata.payload()["model"].is_null());
        assert_eq!(
            metadata
                .payload()
                .pointer("/modelCapabilities/capabilities/streaming")
                .and_then(Value::as_bool),
            Some(true)
        );
        let encoded = metadata.payload().to_string();
        assert!(!encoded.contains("must-not-persist"));
        assert!(!encoded.contains("example.test"));
    }
}
