use super::model_registry_metadata::RuntimeModelRegistryMetadata;
use super::request_context::RuntimeModelSelection;
use crate::ExecutionRequest;
use lime_agent::{supports_direct_route, supports_provider_type, SessionProviderConfig};
use lime_core::database::dao::api_key_provider::{ApiProviderType, ProviderWithKeys};
use lime_core::database::DbConnection;
use lime_services::api_key_provider_service::ApiKeyProviderService;
use lime_services::model_registry_service::ModelRegistryService;
use runtime_core::{
    resolve_ready_model_routing_with_candidate_set, ModelRouteExclusion, ModelRoutingDecision,
    ProviderReadiness, RoutingAttempt, RoutingResolution,
};
use serde_json::Value;

#[cfg(test)]
pub(super) fn selection_from_profile_model_slot(
    request: &ExecutionRequest,
) -> Option<RuntimeModelSelection> {
    let metadata_values = metadata_candidates(request);
    runtime_core::selection_from_profile_model_slot(
        &metadata_values,
        super::request_context::reasoning_effort_from_request(request),
        None,
    )
}

pub(super) fn resolve_ready_routing(
    db: &DbConnection,
    service: &ApiKeyProviderService,
    request: &ExecutionRequest,
    selection: &RuntimeModelSelection,
    direct_provider_config: Option<&SessionProviderConfig>,
    excluded_routes: &[ModelRouteExclusion],
) -> Result<RoutingResolution, String> {
    let metadata_values = metadata_candidates(request);
    if let Some(config) = direct_provider_config {
        return Ok(resolve_direct_routing(&metadata_values, selection, config));
    }
    let candidate_set = super::model_candidate_set::build(db, service, request, selection)?;
    resolve_ready_model_routing_with_candidate_set(
        &metadata_values,
        selection,
        &candidate_set,
        excluded_routes,
        |candidate| resolve_provider_readiness(db, service, candidate, None),
    )
}

fn resolve_direct_routing(
    metadata_values: &[&Value],
    selection: &RuntimeModelSelection,
    config: &SessionProviderConfig,
) -> RoutingResolution {
    let mut routing = runtime_core::resolve_model_routing_for_candidate(metadata_values, selection);
    let readiness = direct_provider_readiness(config);
    let attempt = RoutingAttempt {
        slot: routing.service_model_slot.clone(),
        provider: selection.provider.clone(),
        model: selection.model.clone(),
        source: selection.source.to_string(),
        readiness: readiness.clone(),
        runtime_failure: None,
    };
    routing.fallback_chain = vec![format!("{}/{}", selection.provider, selection.model)];
    RoutingResolution {
        selection: selection.clone(),
        routing,
        readiness,
        attempted: vec![attempt],
    }
}

pub(super) fn resolve_provider_readiness(
    db: &DbConnection,
    service: &ApiKeyProviderService,
    selection: &RuntimeModelSelection,
    direct_provider_config: Option<&SessionProviderConfig>,
) -> Result<ProviderReadiness, String> {
    if let Some(config) = direct_provider_config {
        return Ok(direct_provider_readiness(config));
    }

    let providers = service.get_all_providers(db)?;
    if let Some(provider) = providers
        .iter()
        .find(|provider| provider.provider.id == selection.provider)
    {
        return Ok(configured_provider_readiness(provider));
    }

    Ok(ProviderReadiness::provider_not_configured())
}

pub(super) fn routing_decision_payload(
    selection: &RuntimeModelSelection,
    routing: &ModelRoutingDecision,
    readiness: &ProviderReadiness,
    model_registry: &RuntimeModelRegistryMetadata,
) -> Value {
    runtime_core::routing_decision_payload(selection, routing, readiness, model_registry.payload())
}

pub(crate) fn configured_provider_readiness(provider: &ProviderWithKeys) -> ProviderReadiness {
    let enabled_key_count = provider.api_keys.iter().filter(|key| key.enabled).count();
    let total_key_count = provider.api_keys.len();
    let effective_provider_type = provider.provider.effective_provider_type().to_string();
    let provider_type = Some(effective_provider_type.clone());
    if !supports_provider_type(&effective_provider_type) {
        return ProviderReadiness::provider_store_blocked(
            "unsupported_protocol",
            provider_type,
            Some(provider.provider.enabled),
            enabled_key_count,
            total_key_count,
        );
    }
    if !provider.provider.enabled {
        return ProviderReadiness::provider_store_needs_setup(
            "provider_disabled",
            provider_type,
            Some(false),
            enabled_key_count,
            total_key_count,
        );
    }
    if provider.provider.provider_type == ApiProviderType::Vertexai {
        for (field, value) in [
            ("missing_project", provider.provider.project.as_deref()),
            ("missing_location", provider.provider.location.as_deref()),
        ] {
            if !value.is_some_and(|value| !value.trim().is_empty()) {
                return ProviderReadiness::provider_store_needs_setup(
                    field,
                    provider_type,
                    Some(true),
                    enabled_key_count,
                    total_key_count,
                );
            }
        }
    }
    if enabled_key_count == 0 && provider_requires_enabled_api_key(provider) {
        return ProviderReadiness::provider_store_needs_setup(
            "missing_enabled_api_key",
            provider_type,
            Some(true),
            enabled_key_count,
            total_key_count,
        );
    }

    ProviderReadiness::provider_store_ready(provider_type, enabled_key_count, total_key_count)
}

fn provider_requires_enabled_api_key(provider: &ProviderWithKeys) -> bool {
    ModelRegistryService::requires_api_key_for_runtime(
        &provider.provider.id,
        &provider.provider.api_host,
        provider.provider.provider_type,
    )
}

fn metadata_candidates(request: &ExecutionRequest) -> Vec<&Value> {
    request.runtime_metadata().into_iter().collect()
}

fn direct_provider_readiness(config: &SessionProviderConfig) -> ProviderReadiness {
    let protocol = config
        .route_protocol
        .clone()
        .unwrap_or_else(|| runtime_core::protocol_from_provider_name(&config.provider_name));
    let provider_name = config
        .provider_name
        .trim()
        .to_ascii_lowercase()
        .replace(['-', ' '], "_");
    let requires_api_key = matches!(provider_name.as_str(), "azure" | "azure_openai");
    let has_api_key = config
        .api_key
        .as_deref()
        .is_some_and(|value| !value.trim().is_empty());
    if requires_api_key && !has_api_key {
        ProviderReadiness::direct_request_blocked("missing_enabled_api_key")
    } else if supports_direct_route(&config.provider_name, &protocol) {
        ProviderReadiness::direct_request_ready()
    } else {
        ProviderReadiness::direct_request_blocked("unsupported_protocol")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_backend::tests::request_for_test;
    use lime_core::database::dao::api_key_provider::ApiProviderType;
    use lime_core::database::schema::create_tables;
    use rusqlite::Connection;
    use serde_json::json;
    use std::sync::{Arc, Mutex};

    fn test_db() -> DbConnection {
        let conn = Connection::open_in_memory().expect("open in-memory db");
        create_tables(&conn).expect("create schema");
        Arc::new(Mutex::new(conn))
    }

    #[test]
    fn selection_from_coding_profile_slot_reads_harness_metadata() {
        let request = request_for_test(
            "hello",
            None,
            Some(json!({
                "harness": {
                    "coding_model_slots": {
                        "base": {
                            "provider": "openai",
                            "model": "gpt-4.1-mini"
                        },
                        "coding": {
                            "provider": "custom-coding",
                            "model": "coder-large",
                            "reason": "workspace_coding_profile",
                            "capabilityTags": ["coding", "tools"]
                        },
                        "review": {
                            "provider": "custom-review",
                            "model": "review-small"
                        }
                    }
                }
            })),
        );

        let selection = selection_from_profile_model_slot(&request).expect("slot selection");

        assert_eq!(selection.provider, "custom-coding");
        assert_eq!(selection.model, "coder-large");
        assert_eq!(selection.source, "profile_model_slot");
    }

    #[test]
    fn routing_payload_keeps_review_fast_local_as_diagnostics_only() {
        let request = request_for_test(
            "hello",
            None,
            Some(json!({
                "harness": {
                    "modelSlots": {
                        "coding": {
                            "providerPreference": "custom-coding",
                            "modelPreference": "coder-large"
                        },
                        "review": {
                            "providerPreference": "custom-review",
                            "modelPreference": "review-small"
                        },
                        "fast": {
                            "providerPreference": "openai",
                            "modelPreference": "gpt-4.1-mini"
                        },
                        "local": {
                            "providerPreference": "ollama",
                            "modelPreference": "qwen-coder"
                        }
                    }
                }
            })),
        );
        let selection = RuntimeModelSelection {
            provider: "custom-coding".to_string(),
            model: "coder-large".to_string(),
            source: "profile_model_slot",
            reasoning_effort: None,
        };
        let metadata_values = metadata_candidates(&request);
        let routing =
            runtime_core::resolve_model_routing_for_candidate(&metadata_values, &selection);
        let readiness = ProviderReadiness {
            ready: true,
            status: "ready",
            source: "direct_provider_config",
            reason_code: None,
            provider_type: None,
            enabled: None,
            enabled_key_count: None,
            total_key_count: None,
            direct_request_config: true,
        };
        let model_registry = RuntimeModelRegistryMetadata::from_payload(json!({
            "source": "provider_declared_model",
            "status": "matched",
            "reasonCode": "matched_provider_models",
            "reason_code": "matched_provider_models",
            "modelCapabilities": {
                "provenance": "inferred_hint",
                "capabilities": {
                    "tools": true,
                    "streaming": true,
                    "reasoning": true
                },
                "taskFamilies": ["chat", "reasoning"],
                "runtimeFeatures": ["streaming", "tool_calling", "reasoning"]
            },
            "modelAlias": {
                "canonicalModelId": "coder-large",
                "providerModelId": "coder-large",
                "aliasSource": "local"
            },
            "reasoning": {
                "supported": true,
                "reasoningEffort": {
                    "supported": true,
                    "levels": ["low", "medium", "high"],
                    "default": "medium",
                    "source": "api"
                }
            }
        }));

        let payload = routing_decision_payload(&selection, &routing, &readiness, &model_registry);

        assert_eq!(payload["serviceModelSlot"].as_str(), Some("coding"));
        assert_eq!(payload["selectedProvider"].as_str(), Some("custom-coding"));
        assert_eq!(payload["selectedModel"].as_str(), Some("coder-large"));
        assert_eq!(payload["modelSlot"]["slots"].as_array().unwrap().len(), 4);
        assert!(payload["fallbackChain"].as_array().unwrap().is_empty());
        assert_eq!(
            payload["modelRegistry"]["reasonCode"].as_str(),
            Some("matched_provider_models")
        );
        assert_eq!(
            payload["modelRegistry"]["modelCapabilities"]["capabilities"]["reasoning"].as_bool(),
            Some(true)
        );
    }

    #[test]
    fn ready_routing_falls_back_from_unready_coding_slot_to_base_slot() {
        let db = test_db();
        let service = ApiKeyProviderService::new();
        let custom = service
            .add_custom_provider(
                &db,
                "Workspace Coding Gateway".to_string(),
                ApiProviderType::Openai,
                "https://coding.example.com/v1".to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .expect("custom provider");
        let custom_id = custom.id.clone();
        service
            .initialize_system_providers(&db)
            .expect("system providers");
        service
            .add_api_key(
                &db,
                "openai",
                "sk-test",
                Some("OpenAI test".to_string()),
                true,
            )
            .expect("openai api key");
        let request = request_for_test(
            "hello",
            None,
            Some(json!({
                "harness": {
                    "coding_model_slots": {
                        "coding": {
                            "provider": custom_id,
                            "model": "missing-key-coder"
                        },
                        "base": {
                            "provider": "openai",
                            "model": "gpt-4.1-mini"
                        }
                    }
                }
            })),
        );
        let requested = selection_from_profile_model_slot(&request).expect("requested selection");

        let resolution = resolve_ready_routing(&db, &service, &request, &requested, None, &[])
            .expect("routing resolution");

        assert_eq!(requested.provider, custom.id);
        assert_eq!(resolution.selection.provider, "openai");
        assert_eq!(resolution.selection.model, "gpt-4.1-mini");
        assert!(resolution.readiness.ready);
        assert_eq!(resolution.routing.service_model_slot, "base");
        assert_eq!(resolution.attempted.len(), 2);
        assert_eq!(resolution.attempted[0].slot, "coding");
        assert!(!resolution.attempted[0].readiness.ready);
        assert_eq!(
            resolution.attempted[0].readiness.reason_code,
            Some("missing_enabled_api_key")
        );
        assert_eq!(
            resolution.routing.fallback_chain,
            vec![
                format!("{}/missing-key-coder", requested.provider),
                "openai/gpt-4.1-mini".to_string()
            ]
        );
    }

    #[test]
    fn ready_routing_uses_configured_gemini_adapter() {
        let db = test_db();
        let service = ApiKeyProviderService::new();
        let gemini = service
            .add_custom_provider(
                &db,
                "Gemini route".to_string(),
                ApiProviderType::Gemini,
                "https://generativelanguage.googleapis.com".to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .expect("custom Gemini provider");
        service
            .add_api_key(&db, &gemini.id, "gemini-test", None, true)
            .expect("Gemini API key");
        service
            .initialize_system_providers(&db)
            .expect("system providers");
        service
            .add_api_key(&db, "openai", "sk-test", None, true)
            .expect("OpenAI API key");
        let request = request_for_test(
            "hello",
            None,
            Some(json!({
                "harness": {
                    "coding_model_slots": {
                        "coding": {
                            "provider": gemini.id,
                            "model": "gemini-2.5-pro"
                        },
                        "base": {
                            "provider": "openai",
                            "model": "gpt-4.1-mini"
                        }
                    }
                }
            })),
        );
        let requested = selection_from_profile_model_slot(&request).expect("requested selection");

        let resolution = resolve_ready_routing(&db, &service, &request, &requested, None, &[])
            .expect("routing resolution");

        assert_eq!(resolution.selection.provider, gemini.id);
        assert_eq!(resolution.selection.model, "gemini-2.5-pro");
        assert_eq!(resolution.routing.service_model_slot, "coding");
        assert_eq!(resolution.attempted.len(), 1);
        assert!(resolution.readiness.ready);
    }

    #[test]
    fn custom_openai_localhost_without_key_is_unready_for_runtime() {
        let db = test_db();
        let service = ApiKeyProviderService::new();
        let provider = service
            .add_custom_provider(
                &db,
                "Local OpenAI-compatible Gateway".to_string(),
                ApiProviderType::Openai,
                "http://127.0.0.1:56599/v1".to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .expect("custom provider");
        service
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
                        "pending-route-model",
                    ),
                ]),
            )
            .expect("declared model");

        let readiness = resolve_provider_readiness(
            &db,
            &service,
            &RuntimeModelSelection {
                provider: provider.id.clone(),
                model: "pending-route-model".to_string(),
                source: "runtime_request",
                reasoning_effort: None,
            },
            None,
        )
        .expect("readiness");

        assert!(!readiness.ready);
        assert_eq!(readiness.reason_code, Some("missing_enabled_api_key"));
    }

    #[test]
    fn direct_ollama_responses_route_is_ready_without_api_key() {
        let direct = SessionProviderConfig {
            provider_name: "ollama".to_string(),
            provider_selector: Some("local-ollama".to_string()),
            model_name: "qwen3:14b".to_string(),
            api_key: None,
            base_url: Some("http://127.0.0.1:11434".to_string()),
            api_version: None,
            credential_uuid: None,
            reasoning_effort: None,
            service_tier: None,
            route_protocol: Some(app_server_protocol::ProtocolKind::OpenaiResponses),
            toolshim: false,
            toolshim_model: None,
            model_capabilities: Some(json!({
                "capabilities": { "streaming": true },
                "taskFamilies": ["chat"],
                "runtimeFeatures": ["streaming"]
            })),
            supports_websockets: false,
        };

        let direct_readiness = direct_provider_readiness(&direct);
        assert!(direct_readiness.ready);
        assert_eq!(direct_readiness.source, "direct_provider_config");
        assert_eq!(direct_readiness.reason_code, None);
        assert!(direct_readiness.direct_request_config);
    }

    #[test]
    fn ready_direct_ollama_route_stays_on_requested_selection() {
        let db = test_db();
        let service = ApiKeyProviderService::new();
        let request = request_for_test(
            "hello",
            None,
            Some(json!({
                "harness": {
                    "coding_model_slots": {
                        "coding": {
                            "provider": "local-ollama",
                            "model": "qwen3:14b"
                        },
                        "base": {
                            "provider": "openai",
                            "model": "gpt-4.1-mini"
                        }
                    }
                }
            })),
        );
        let selection = selection_from_profile_model_slot(&request).expect("direct selection");
        let direct = SessionProviderConfig {
            provider_name: "ollama".to_string(),
            provider_selector: Some("local-ollama".to_string()),
            model_name: "qwen3:14b".to_string(),
            api_key: None,
            base_url: Some("http://127.0.0.1:11434".to_string()),
            api_version: None,
            credential_uuid: None,
            reasoning_effort: None,
            service_tier: None,
            route_protocol: Some(app_server_protocol::ProtocolKind::OpenaiResponses),
            toolshim: false,
            toolshim_model: None,
            model_capabilities: Some(json!({
                "capabilities": { "streaming": true },
                "taskFamilies": ["chat"],
                "runtimeFeatures": ["streaming"]
            })),
            supports_websockets: false,
        };

        let resolution =
            resolve_ready_routing(&db, &service, &request, &selection, Some(&direct), &[])
                .expect("direct routing resolution");

        assert_eq!(resolution.selection, selection);
        assert!(resolution.readiness.ready);
        assert_eq!(resolution.readiness.reason_code, None);
        assert_eq!(resolution.attempted.len(), 1);
        assert_eq!(resolution.attempted[0].provider, "local-ollama");
        assert_eq!(
            resolution.routing.fallback_chain,
            vec!["local-ollama/qwen3:14b"]
        );
    }

    #[test]
    fn configured_provider_readiness_rejects_protocols_without_current_adapters() {
        let db = test_db();
        let service = ApiKeyProviderService::new();

        for (provider_type, label) in [
            (ApiProviderType::AwsBedrock, "Bedrock"),
            (ApiProviderType::Fal, "Fal"),
        ] {
            let provider = service
                .add_custom_provider(
                    &db,
                    format!("{label} route"),
                    provider_type,
                    provider_type.runtime_spec().default_api_host.to_string(),
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .expect("create unsupported provider route");
            let provider = service
                .get_provider(&db, &provider.id)
                .expect("read provider")
                .expect("stored provider");

            let readiness = configured_provider_readiness(&provider);
            assert!(!readiness.ready, "provider_type={provider_type}");
            assert_eq!(
                readiness.reason_code,
                Some("unsupported_protocol"),
                "provider_type={provider_type}"
            );
        }
    }

    #[test]
    fn configured_vertex_provider_requires_context_key_and_current_adapter() {
        let db = test_db();
        let service = ApiKeyProviderService::new();
        let provider = service
            .add_custom_provider(
                &db,
                "Vertex Gemini route".to_string(),
                ApiProviderType::Vertexai,
                String::new(),
                None,
                None,
                None,
                None,
                None,
            )
            .expect("create Vertex provider route");
        let stored = service
            .get_provider(&db, &provider.id)
            .expect("read Vertex provider")
            .expect("stored Vertex provider");
        assert_eq!(
            configured_provider_readiness(&stored).reason_code,
            Some("missing_project")
        );

        service
            .update_provider(
                &db,
                &provider.id,
                None,
                None,
                None,
                None,
                None,
                None,
                Some("project-alpha".to_string()),
                Some("us-central1".to_string()),
                None,
                None,
                None,
            )
            .expect("configure Vertex context");
        let stored = service
            .get_provider(&db, &provider.id)
            .expect("read configured Vertex provider")
            .expect("stored configured Vertex provider");
        assert_eq!(
            configured_provider_readiness(&stored).reason_code,
            Some("missing_enabled_api_key")
        );

        service
            .add_api_key(&db, &provider.id, "vertex-token", None, true)
            .expect("add Vertex access token");
        let stored = service
            .get_provider(&db, &provider.id)
            .expect("read ready Vertex provider")
            .expect("stored ready Vertex provider");
        assert!(configured_provider_readiness(&stored).ready);
    }

    #[test]
    fn configured_azure_provider_requires_and_accepts_enabled_api_key() {
        let db = test_db();
        let service = ApiKeyProviderService::new();
        let provider = service
            .add_custom_provider(
                &db,
                "Azure Responses route".to_string(),
                ApiProviderType::AzureOpenai,
                "https://resource.openai.azure.com".to_string(),
                Some("v1".to_string()),
                None,
                None,
                None,
                None,
            )
            .expect("create Azure provider route");

        let stored = service
            .get_provider(&db, &provider.id)
            .expect("read Azure provider")
            .expect("stored Azure provider");
        let readiness = configured_provider_readiness(&stored);
        assert!(!readiness.ready);
        assert_eq!(readiness.reason_code, Some("missing_enabled_api_key"));

        service
            .add_api_key(&db, &provider.id, "azure-test-key", None, true)
            .expect("add Azure API key");
        let stored = service
            .get_provider(&db, &provider.id)
            .expect("read ready Azure provider")
            .expect("stored ready Azure provider");
        let readiness = configured_provider_readiness(&stored);
        assert!(readiness.ready);
        assert_eq!(readiness.reason_code, None);
    }

    #[test]
    fn direct_azure_route_without_api_key_is_missing_credential() {
        let direct = SessionProviderConfig {
            provider_name: "azure".to_string(),
            provider_selector: Some("azure-openai".to_string()),
            model_name: "gpt-5.4".to_string(),
            api_key: None,
            base_url: Some("https://resource.openai.azure.com".to_string()),
            api_version: Some("v1".to_string()),
            credential_uuid: None,
            reasoning_effort: None,
            service_tier: None,
            route_protocol: Some(app_server_protocol::ProtocolKind::OpenaiResponses),
            toolshim: false,
            toolshim_model: None,
            model_capabilities: None,
            supports_websockets: false,
        };

        let readiness = direct_provider_readiness(&direct);
        assert!(!readiness.ready);
        assert_eq!(readiness.reason_code, Some("missing_enabled_api_key"));
    }

    #[test]
    fn configured_keyless_ollama_provider_is_runtime_ready() {
        let db = test_db();
        let service = ApiKeyProviderService::new();
        let provider = service
            .add_custom_provider(
                &db,
                "Ollama route".to_string(),
                ApiProviderType::Ollama,
                "http://127.0.0.1:11434".to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .expect("create Ollama provider route");
        let provider = service
            .get_provider(&db, &provider.id)
            .expect("read Ollama provider")
            .expect("stored Ollama provider");

        let readiness = configured_provider_readiness(&provider);
        assert!(readiness.ready);
        assert_eq!(readiness.reason_code, None);
        assert_eq!(readiness.enabled_key_count, Some(0));
    }

    #[test]
    fn configured_gemini_requires_an_enabled_api_key() {
        let db = test_db();
        let service = ApiKeyProviderService::new();
        let provider = service
            .add_custom_provider(
                &db,
                "Gemini route".to_string(),
                ApiProviderType::Gemini,
                "https://generativelanguage.googleapis.com".to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .expect("create Gemini provider route");
        let provider = service
            .get_provider(&db, &provider.id)
            .expect("read Gemini provider")
            .expect("stored Gemini provider");

        let readiness = configured_provider_readiness(&provider);
        assert!(!readiness.ready);
        assert_eq!(readiness.reason_code, Some("missing_enabled_api_key"));
    }

    #[test]
    fn unknown_stored_provider_type_is_not_runtime_ready() {
        let db = test_db();
        {
            let conn = db.lock().expect("lock db");
            conn.execute(
                "INSERT INTO api_key_providers (
                    id, name, type, api_host, is_system, group_name, enabled, sort_order,
                    models, created_at, updated_at
                 ) VALUES (?1, ?2, ?3, ?4, 0, 'cloud', 1, 0, '[]', ?5, ?5)",
                rusqlite::params![
                    "future-route",
                    "Future Route",
                    "future-provider",
                    "https://future.invalid/v1",
                    "2026-07-26T00:00:00Z",
                ],
            )
            .expect("insert unknown provider type");
        }
        let service = ApiKeyProviderService::new();

        let readiness = resolve_provider_readiness(
            &db,
            &service,
            &RuntimeModelSelection {
                provider: "future-route".to_string(),
                model: "future-model".to_string(),
                source: "runtime_request",
                reasoning_effort: None,
            },
            None,
        )
        .expect("resolve unknown provider readiness");

        assert!(!readiness.ready);
        assert_eq!(readiness.reason_code, Some("provider_not_configured"));
    }

    #[test]
    fn known_provider_names_without_store_records_are_not_runtime_ready() {
        let db = test_db();
        let service = ApiKeyProviderService::new();
        let providers = service
            .get_all_providers(&db)
            .expect("read provider fixture");

        for provider in ["claude", "gemini_api_key"] {
            assert!(
                providers
                    .iter()
                    .all(|stored| stored.provider.id != provider),
                "fixture must not contain provider={provider}"
            );
            let readiness = resolve_provider_readiness(
                &db,
                &service,
                &RuntimeModelSelection {
                    provider: provider.to_string(),
                    model: "test-model".to_string(),
                    source: "runtime_request",
                    reasoning_effort: None,
                },
                None,
            )
            .expect("resolve provider readiness");

            assert!(!readiness.ready, "provider={provider}");
            assert_eq!(
                readiness.reason_code,
                Some("provider_not_configured"),
                "provider={provider}"
            );
        }
    }
}
