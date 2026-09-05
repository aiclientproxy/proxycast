use super::request_context::RuntimeModelSelection;
use code_mode::RuntimeToolMode;
use lime_agent::SessionProviderConfig;
use lime_core::database::DbConnection;
use lime_core::models::model_registry::{
    EnhancedModelMetadata, ModelCapabilityProvenance, ModelModality, ModelRuntimeFeature,
};
use lime_core::models::RuntimeProviderCredential;
use lime_services::api_key_provider_service::ApiKeyProviderService;
use lime_services::model_registry_service::{ModelRegistryService, ProviderModelCacheAccess};
use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub(super) struct RuntimeModelRegistryMetadata {
    payload: Value,
    tool_mode: RuntimeToolMode,
    supports_custom_tools: bool,
}

impl RuntimeModelRegistryMetadata {
    #[cfg(test)]
    pub(super) fn from_payload(payload: Value) -> Self {
        let tool_mode = runtime_tool_mode_from_payload(&payload);
        let supports_custom_tools = payload_supports_custom_tools(&payload);
        Self {
            payload,
            tool_mode,
            supports_custom_tools,
        }
    }

    pub(super) fn payload(&self) -> &Value {
        &self.payload
    }

    pub(super) fn tool_mode(&self) -> RuntimeToolMode {
        self.tool_mode
    }

    pub(super) fn supports_custom_tools(&self) -> bool {
        self.supports_custom_tools
    }
}

fn runtime_tool_mode(value: Option<&str>) -> RuntimeToolMode {
    match value.map(str::trim) {
        Some("code_mode") => RuntimeToolMode::CodeMode,
        Some("code_mode_only") => RuntimeToolMode::CodeModeOnly,
        _ => RuntimeToolMode::Direct,
    }
}

#[cfg(test)]
fn runtime_tool_mode_from_payload(payload: &Value) -> RuntimeToolMode {
    runtime_tool_mode(
        payload
            .pointer("/model/tool_mode")
            .or_else(|| payload.pointer("/model/toolMode"))
            .or_else(|| payload.get("tool_mode"))
            .or_else(|| payload.get("toolMode"))
            .and_then(Value::as_str),
    )
}

#[cfg(test)]
fn payload_supports_custom_tools(payload: &Value) -> bool {
    [
        "/model/runtime_features",
        "/model/runtimeFeatures",
        "/model_capabilities/runtime_features",
        "/modelCapabilities/runtimeFeatures",
    ]
    .into_iter()
    .filter_map(|pointer| payload.pointer(pointer))
    .filter_map(Value::as_array)
    .flatten()
    .filter_map(Value::as_str)
    .any(|feature| feature == "custom_tools")
}

pub(super) async fn resolve_runtime_model_registry_metadata(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
    selection: &RuntimeModelSelection,
    direct_provider_config: Option<&SessionProviderConfig>,
    route_credential: Option<&RuntimeProviderCredential>,
) -> Result<RuntimeModelRegistryMetadata, String> {
    if let Some(config) = direct_provider_config {
        let tool_mode = runtime_tool_mode(
            config
                .model_capabilities
                .as_ref()
                .and_then(|value| value.get("tool_mode").or_else(|| value.get("toolMode")))
                .and_then(Value::as_str),
        );
        let capability_snapshot = config.model_capabilities.as_ref().map(|value| {
            let snapshot = runtime_core::capability_snapshot_from_model_capabilities(value);
            snapshot
        });
        let supports_custom_tools = capability_snapshot.as_ref().is_some_and(|snapshot| {
            snapshot
                .runtime_features
                .iter()
                .any(|feature| feature == "custom_tools")
        });
        let chat_wire_was_lowered = capability_snapshot.as_ref().is_some_and(|snapshot| {
            snapshot
                .input_modalities
                .iter()
                .any(|modality| !matches!(modality.as_str(), "text" | "image"))
        });
        let model_capabilities = capability_snapshot.as_ref().map(|snapshot| {
            capability_payload(snapshot, ModelCapabilityProvenance::ProviderExplicit)
        });
        let reason_code = if chat_wire_was_lowered {
            "chat_wire_text_image_only"
        } else {
            "direct_provider_config_not_in_registry"
        };
        return Ok(RuntimeModelRegistryMetadata {
            payload: json!({
                "source": "direct_provider_config",
                "sourceLabel": "direct_provider_config",
                "source_label": "direct_provider_config",
                "status": "runtime_selection_only",
                "reasonCode": reason_code,
                "reason_code": reason_code,
                "providerId": selection.provider,
                "provider_id": selection.provider,
                "requestedModelId": selection.model,
                "requested_model_id": selection.model,
                "matchedModelId": null,
                "matched_model_id": null,
                "model": null,
                "modelCapabilities": model_capabilities.clone(),
                "model_capabilities": model_capabilities,
                "modelMessages": model_messages_from_provider_config(config.model_capabilities.as_ref()),
                "model_messages": model_messages_from_provider_config(config.model_capabilities.as_ref()),
                "multiAgentVersion": multi_agent_version_from_provider_config(config.model_capabilities.as_ref()),
                "multi_agent_version": multi_agent_version_from_provider_config(config.model_capabilities.as_ref()),
                "multiAgentReasoningEffort": multi_agent_reasoning_effort_from_provider_config(config.model_capabilities.as_ref()),
                "multi_agent_reasoning_effort": multi_agent_reasoning_effort_from_provider_config(config.model_capabilities.as_ref()),
                "modelAlias": null,
                "model_alias": null,
                "reasoning": null,
                "toolMode": match tool_mode {
                    RuntimeToolMode::Direct => "direct",
                    RuntimeToolMode::CodeMode => "code_mode",
                    RuntimeToolMode::CodeModeOnly => "code_mode_only",
                },
            }),
            tool_mode,
            supports_custom_tools,
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
    let chat_wire_model = metadata.model.as_ref().map(chat_wire_model_metadata);
    let chat_wire_was_lowered = metadata.model.as_ref().is_some_and(|model| {
        chat_wire_model
            .as_ref()
            .is_some_and(|lowered| lowered.input_modalities != model.input_modalities)
    });
    let reason_code = if chat_wire_was_lowered {
        "chat_wire_text_image_only"
    } else {
        metadata.reason_code
    };
    let model = chat_wire_model
        .as_ref()
        .map(serde_json::to_value)
        .transpose()
        .map_err(|error| format!("序列化模型注册 metadata 失败: {error}"))?;
    let model_capabilities = chat_wire_model.as_ref().map(model_capability_payload);
    let model_alias = chat_wire_model.as_ref().map(|model| {
        json!({
            "canonicalModelId": model.canonical_model_id,
            "canonical_model_id": model.canonical_model_id,
            "providerModelId": model.provider_model_id,
            "provider_model_id": model.provider_model_id,
            "aliasSource": model.alias_source,
            "alias_source": model.alias_source,
        })
    });
    let reasoning = chat_wire_model.as_ref().map(|model| {
        json!({
            "supported": model.capabilities.reasoning,
            "reasoningEffort": model.capabilities.reasoning_effort,
            "reasoning_effort": model.capabilities.reasoning_effort,
        })
    });
    let tool_mode = runtime_tool_mode(
        chat_wire_model
            .as_ref()
            .and_then(|model| model.tool_mode.as_deref()),
    );
    let supports_custom_tools = chat_wire_model.as_ref().is_some_and(|model| {
        model
            .runtime_features
            .contains(&ModelRuntimeFeature::CustomTools)
    });

    Ok(RuntimeModelRegistryMetadata {
        payload: json!({
            "source": metadata.source.as_str(),
            "sourceLabel": metadata.source.as_str(),
            "source_label": metadata.source.as_str(),
            "status": if metadata.model.is_some() { "matched" } else { "missing" },
            "reasonCode": reason_code,
            "reason_code": reason_code,
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
            "modelMessages": chat_wire_model
                .as_ref()
                .and_then(|model| model.model_messages.clone()),
            "model_messages": chat_wire_model
                .as_ref()
                .and_then(|model| model.model_messages.clone()),
            "multiAgentVersion": chat_wire_model
                .as_ref()
                .and_then(|model| model.multi_agent_version.map(model_multi_agent_version_name)),
            "multi_agent_version": chat_wire_model
                .as_ref()
                .and_then(|model| model.multi_agent_version.map(model_multi_agent_version_name)),
            "multiAgentReasoningEffort": chat_wire_model
                .as_ref()
                .and_then(|model| model.multi_agent_reasoning_effort.clone()),
            "multi_agent_reasoning_effort": chat_wire_model
                .as_ref()
                .and_then(|model| model.multi_agent_reasoning_effort.clone()),
            "modelAlias": model_alias,
            "model_alias": model_alias,
            "reasoning": reasoning,
            "toolMode": chat_wire_model.as_ref().and_then(|model| model.tool_mode.clone()),
        }),
        tool_mode,
        supports_custom_tools,
    })
}

fn chat_wire_model_metadata(model: &EnhancedModelMetadata) -> EnhancedModelMetadata {
    let mut model = model.clone();
    model
        .input_modalities
        .retain(|modality| matches!(modality, ModelModality::Text | ModelModality::Image));
    model
}

fn model_multi_agent_version_name(
    version: lime_core::models::model_registry::ModelMultiAgentVersion,
) -> &'static str {
    match version {
        lime_core::models::model_registry::ModelMultiAgentVersion::Disabled => "disabled",
        lime_core::models::model_registry::ModelMultiAgentVersion::V1 => "v1",
        lime_core::models::model_registry::ModelMultiAgentVersion::V2 => "v2",
    }
}

fn model_messages_from_provider_config(capabilities: Option<&Value>) -> Option<Value> {
    capabilities.and_then(|value| {
        ["modelMessages", "model_messages"]
            .into_iter()
            .find_map(|key| value.get(key).cloned())
    })
}

fn multi_agent_version_from_provider_config(capabilities: Option<&Value>) -> Option<Value> {
    capabilities.and_then(|value| {
        ["multiAgentVersion", "multi_agent_version"]
            .into_iter()
            .find_map(|key| value.get(key).cloned())
            .filter(|value| matches!(value.as_str(), Some("disabled" | "v1" | "v2")))
    })
}

fn multi_agent_reasoning_effort_from_provider_config(
    capabilities: Option<&Value>,
) -> Option<Value> {
    capabilities.and_then(|value| {
        ["multiAgentReasoningEffort", "multi_agent_reasoning_effort"]
            .into_iter()
            .find_map(|key| value.get(key).cloned())
            .filter(Value::is_string)
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
    let input_modalities = snapshot
        .input_modalities
        .iter()
        .filter(|modality| matches!(modality.as_str(), "text" | "image"))
        .collect::<Vec<_>>();
    json!({
        "provenance": provenance,
        "capabilities": snapshot.capabilities,
        "taskFamilies": snapshot.task_families,
        "runtimeFeatures": snapshot.runtime_features,
        "inputModalities": input_modalities,
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

    #[test]
    fn chat_route_metadata_keeps_only_executable_text_and_image_inputs() {
        let mut model = EnhancedModelMetadata::new(
            "model-a".to_string(),
            "Model A".to_string(),
            "provider-a".to_string(),
            "Provider A".to_string(),
        );
        model.input_modalities = vec![
            ModelModality::Text,
            ModelModality::Image,
            ModelModality::Audio,
            ModelModality::Video,
            ModelModality::File,
        ];

        let lowered = chat_wire_model_metadata(&model);

        assert_eq!(
            lowered.input_modalities,
            vec![ModelModality::Text, ModelModality::Image]
        );
        assert_eq!(model.input_modalities.len(), 5);
    }

    #[test]
    fn registry_payload_requires_explicit_tool_mode_and_custom_tools_capability() {
        let executable = RuntimeModelRegistryMetadata::from_payload(json!({
            "model": {
                "toolMode": "code_mode_only",
                "runtimeFeatures": ["streaming", "custom_tools"]
            }
        }));
        assert_eq!(executable.tool_mode(), RuntimeToolMode::CodeModeOnly);
        assert!(executable.supports_custom_tools());

        let unknown_mode = RuntimeModelRegistryMetadata::from_payload(json!({
            "model": {
                "tool_mode": "code_interpreter",
                "runtime_features": ["custom_tools"]
            }
        }));
        assert_eq!(unknown_mode.tool_mode(), RuntimeToolMode::Direct);
        assert!(unknown_mode.supports_custom_tools());

        let missing_capability = RuntimeModelRegistryMetadata::from_payload(json!({
            "model": {
                "tool_mode": "code_mode",
                "runtime_features": ["tool_calling"]
            }
        }));
        assert_eq!(missing_capability.tool_mode(), RuntimeToolMode::CodeMode);
        assert!(!missing_capability.supports_custom_tools());
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
                api_version: None,
                credential_uuid: None,
                reasoning_effort: None,
                service_tier: None,
                route_protocol: None,
                toolshim: false,
                toolshim_model: None,
                model_capabilities: Some(json!({
                    "taskFamilies": ["chat"],
                    "inputModalities": ["text", "image", "audio"],
                    "outputModalities": ["text"],
                    "runtimeFeatures": ["streaming"],
                    "multiAgentVersion": "v2",
                    "multiAgentReasoningEffort": "high",
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
            Some("chat_wire_text_image_only")
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
                .pointer("/modelCapabilities/inputModalities"),
            Some(&json!(["text", "image"]))
        );
        assert_eq!(
            metadata
                .payload()
                .pointer("/modelCapabilities/capabilities/streaming")
                .and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(metadata.payload()["multiAgentVersion"], "v2");
        assert_eq!(metadata.payload()["multiAgentReasoningEffort"], "high");
        assert_eq!(metadata.tool_mode(), RuntimeToolMode::Direct);
        assert!(!metadata.supports_custom_tools());
        let encoded = metadata.payload().to_string();
        assert!(!encoded.contains("must-not-persist"));
        assert!(!encoded.contains("example.test"));
    }
}
