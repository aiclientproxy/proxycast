use super::model_registry_metadata;
use super::model_route_contract;
use super::model_route_credential::{self, RouteCredential};
use super::model_routing;
use super::request_context::{selection_with_capability_reasoning, RuntimeModelSelection};
use crate::ExecutionRequest;
use app_server_protocol::{ModelTaskRequest, ResolvedModelRoute};
use lime_agent::SessionProviderConfig;
use lime_core::database::dao::api_key_provider::ProviderWithKeys;
use lime_core::database::DbConnection;
use lime_services::api_key_provider_service::ApiKeyProviderService;
use runtime_core::RoutingResolution;
use serde_json::Value;

#[derive(Debug, Clone)]
pub(super) struct ChatModelRouteResolution {
    pub(super) selection: RuntimeModelSelection,
    pub(super) model_task_request: ModelTaskRequest,
    pub(super) resolved_route: ResolvedModelRoute,
    pub(super) decision_payload: Value,
    pub(super) fallback_payload: Option<Value>,
    pub(super) not_possible_payload: Option<Value>,
}

impl ChatModelRouteResolution {
    pub(super) fn service_model_slot(&self) -> &str {
        self.resolved_route
            .decision
            .service_model_slot
            .as_deref()
            .or(self.model_task_request.routing_slot.as_deref())
            .unwrap_or("coding")
    }
}

pub(super) struct PreparedChatModelRoute {
    routing_resolution: RoutingResolution,
    provider_record: Option<ProviderWithKeys>,
}

impl PreparedChatModelRoute {
    pub(super) fn selection(&self) -> &RuntimeModelSelection {
        &self.routing_resolution.selection
    }
}

pub(super) fn prepare_chat_model_route(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
    request: &ExecutionRequest,
    requested_selection: &RuntimeModelSelection,
    direct_provider_config: Option<&SessionProviderConfig>,
) -> Result<PreparedChatModelRoute, String> {
    let routing_resolution = model_routing::resolve_ready_routing(
        db,
        api_key_provider_service,
        request,
        requested_selection,
        direct_provider_config,
    )?;
    let provider_record = if direct_provider_config.is_some() {
        None
    } else {
        api_key_provider_service.get_provider(db, &routing_resolution.selection.provider)?
    };

    Ok(PreparedChatModelRoute {
        routing_resolution,
        provider_record,
    })
}

pub(super) async fn assemble_chat_model_route(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
    request: &ExecutionRequest,
    requested_selection: &RuntimeModelSelection,
    direct_provider_config: Option<&SessionProviderConfig>,
    prepared: PreparedChatModelRoute,
    preferred_credential_ref: Option<&str>,
) -> Result<ChatModelRouteResolution, String> {
    let routing_resolution = prepared.routing_resolution;
    let requested_route_selection = &routing_resolution.selection;
    let model_routing = &routing_resolution.routing;
    let provider_readiness = &routing_resolution.readiness;
    let route_credential = if provider_readiness.ready {
        model_route_credential::resolve_route_credential(
            db,
            api_key_provider_service,
            &requested_route_selection.provider,
            prepared.provider_record.as_ref(),
            direct_provider_config,
            preferred_credential_ref,
        )
        .await?
    } else {
        RouteCredential::unavailable()
    };
    let model_registry = model_registry_metadata::resolve_runtime_model_registry_metadata(
        db,
        api_key_provider_service,
        requested_route_selection,
        direct_provider_config,
        route_credential.runtime_credential(),
    )
    .await?;
    let routing_payload = model_routing::routing_decision_payload(
        requested_route_selection,
        model_routing,
        provider_readiness,
        &model_registry,
    );
    let model_task_request = model_route_contract::chat_task_request_from_runtime(
        request,
        requested_route_selection,
        &routing_payload,
    );
    let mut resolved_route = model_route_contract::resolved_route_from_runtime(
        &model_task_request,
        requested_route_selection,
        &routing_payload,
        prepared.provider_record.as_ref(),
        route_credential.credential_ref(),
        direct_provider_config,
    );
    let selection = selection_with_capability_reasoning(
        requested_route_selection,
        &resolved_route.capability_snapshot,
    );
    resolved_route.defaults.reasoning_effort = selection.reasoning_effort.clone();
    if direct_provider_config.is_none()
        && resolved_route.auth.kind == app_server_protocol::AuthKind::ApiKeyRef
    {
        resolved_route.auth.credential_ref =
            route_credential.credential_ref().map(ToString::to_string);
        if resolved_route.failure.is_none() && resolved_route.auth.credential_ref.is_none() {
            return Err("resolved_credential_unavailable".to_string());
        }
    }
    let evidence = runtime_core::route_resolution_evidence_payloads(
        requested_selection,
        &routing_resolution,
        model_registry.payload(),
        &model_task_request,
        &resolved_route,
    );
    Ok(ChatModelRouteResolution {
        selection,
        model_task_request,
        resolved_route,
        decision_payload: evidence.decision_payload,
        fallback_payload: evidence.fallback_payload,
        not_possible_payload: evidence.not_possible_payload,
    })
}

pub(super) async fn resolve_chat_model_route(
    db: &DbConnection,
    api_key_provider_service: &ApiKeyProviderService,
    request: &ExecutionRequest,
    requested_selection: &RuntimeModelSelection,
    direct_provider_config: Option<&SessionProviderConfig>,
) -> Result<ChatModelRouteResolution, String> {
    let prepared = prepare_chat_model_route(
        db,
        api_key_provider_service,
        request,
        requested_selection,
        direct_provider_config,
    )?;
    assemble_chat_model_route(
        db,
        api_key_provider_service,
        request,
        requested_selection,
        direct_provider_config,
        prepared,
        None,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_backend::tests::request_for_test;
    use app_server_protocol::{AuthKind, EndpointKind, ProtocolKind, RouteFailureCategory};
    use lime_core::database::dao::api_key_provider::ApiProviderType;
    use lime_core::database::schema::create_tables;
    use lime_core::database::DbConnection;
    use lime_services::model_registry_service::ModelRegistryService;
    use rusqlite::Connection;
    use serde_json::json;
    use std::sync::{Arc, Mutex};

    fn test_db() -> DbConnection {
        let conn = Connection::open_in_memory().expect("open in-memory db");
        create_tables(&conn).expect("create schema");
        Arc::new(Mutex::new(conn))
    }

    fn selection(provider: &str, model: &str) -> RuntimeModelSelection {
        RuntimeModelSelection {
            provider: provider.to_string(),
            model: model.to_string(),
            source: "runtime_options",
            reasoning_effort: None,
        }
    }

    #[tokio::test]
    async fn direct_provider_config_without_capability_snapshot_fails_closed() {
        let db = test_db();
        let service = ApiKeyProviderService::new();
        let request = request_for_test("hello", None, None);
        let requested_selection = selection("fixture-openai", "fixture-model");
        let direct_provider_config = SessionProviderConfig {
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
            model_capabilities: None,
            supports_websockets: false,
        };

        let route = resolve_chat_model_route(
            &db,
            &service,
            &request,
            &requested_selection,
            Some(&direct_provider_config),
        )
        .await
        .expect("route");

        let failure = route
            .resolved_route
            .failure
            .as_ref()
            .expect("missing capability snapshot failure");
        assert_eq!(failure.category, RouteFailureCategory::CapabilityGap);
        assert_eq!(failure.reason_code, "capability_snapshot_missing");
        assert_eq!(
            failure.capability_gap.as_deref(),
            Some("capability_snapshot:missing")
        );
        assert_eq!(route.resolved_route.protocol, ProtocolKind::OpenaiChat);
        assert_eq!(
            route.resolved_route.endpoint.kind,
            EndpointKind::DirectRequest
        );
        assert_eq!(
            route.resolved_route.endpoint.base_url.as_deref(),
            Some("http://127.0.0.1:56599")
        );
        assert_eq!(route.resolved_route.auth.kind, AuthKind::DirectApiKey);
        assert_eq!(
            route
                .decision_payload
                .pointer("/modelRegistry/reasonCode")
                .and_then(|value| value.as_str()),
            Some("direct_provider_config_not_in_registry")
        );
        assert_eq!(
            route
                .not_possible_payload
                .as_ref()
                .and_then(|payload| payload.pointer("/routeFailure/reasonCode"))
                .and_then(Value::as_str),
            Some("capability_snapshot_missing")
        );
    }

    #[tokio::test]
    async fn direct_provider_config_can_declare_fixture_tool_capabilities() {
        let db = test_db();
        let service = ApiKeyProviderService::new();
        let request = request_for_test("hello", None, None);
        let requested_selection = selection("fixture-openai", "fixture-model");
        let direct_provider_config = SessionProviderConfig {
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
                "capabilities": {
                    "tools": true,
                    "streaming": true,
                    "jsonMode": true,
                    "functionCalling": true
                },
                "taskFamilies": ["chat"],
                "inputModalities": ["text"],
                "outputModalities": ["text"],
                "runtimeFeatures": ["streaming", "tool_calling"]
            })),
            supports_websockets: false,
        };

        let route = resolve_chat_model_route(
            &db,
            &service,
            &request,
            &requested_selection,
            Some(&direct_provider_config),
        )
        .await
        .expect("route");

        assert!(route.resolved_route.failure.is_none());
        assert!(route.resolved_route.capability_snapshot.capabilities.tools);
        assert!(
            route
                .resolved_route
                .capability_snapshot
                .capabilities
                .function_calling
        );
        assert!(route
            .resolved_route
            .capability_snapshot
            .runtime_features
            .contains(&"tool_calling".to_string()));
        assert_eq!(
            route
                .decision_payload
                .pointer("/modelRegistry/modelCapabilities/capabilities/tools")
                .and_then(|value| value.as_bool()),
            Some(true)
        );
    }

    #[tokio::test]
    async fn resolved_route_only_keeps_snapshot_declared_custom_reasoning_effort() {
        let db = test_db();
        let service = ApiKeyProviderService::new();
        let request = request_for_test("hello", None, None);
        let mut requested_selection = selection("fixture-openai", "fixture-model");
        requested_selection.reasoning_effort = Some("ultra".to_string());
        let direct_provider_config = SessionProviderConfig {
            provider_name: "openai".to_string(),
            provider_selector: Some("fixture-openai".to_string()),
            model_name: "fixture-model".to_string(),
            api_key: Some("fixture-key".to_string()),
            base_url: Some("http://127.0.0.1:56599".to_string()),
            credential_uuid: None,
            reasoning_effort: Some("ultra".to_string()),
            service_tier: None,
            route_protocol: None,
            toolshim: false,
            toolshim_model: None,
            model_capabilities: Some(json!({
                "capabilities": {
                    "tools": true,
                    "streaming": true,
                    "reasoning": true,
                    "reasoningEffort": {
                        "supported": true,
                        "levels": ["low", "high", "ultra"]
                    }
                },
                "taskFamilies": ["chat"],
                "inputModalities": ["text"],
                "outputModalities": ["text"],
                "runtimeFeatures": ["streaming", "reasoning"]
            })),
            supports_websockets: false,
        };

        let route = resolve_chat_model_route(
            &db,
            &service,
            &request,
            &requested_selection,
            Some(&direct_provider_config),
        )
        .await
        .expect("route");

        assert!(route.resolved_route.failure.is_none());
        assert_eq!(route.selection.reasoning_effort.as_deref(), Some("ultra"));
        assert_eq!(
            route.resolved_route.defaults.reasoning_effort.as_deref(),
            Some("ultra")
        );
    }

    #[tokio::test]
    async fn resolved_route_drops_reasoning_effort_not_declared_by_snapshot() {
        let db = test_db();
        let service = ApiKeyProviderService::new();
        let request = request_for_test("hello", None, None);
        let mut requested_selection = selection("fixture-openai", "fixture-model");
        requested_selection.reasoning_effort = Some("medium".to_string());
        let direct_provider_config = SessionProviderConfig {
            provider_name: "openai".to_string(),
            provider_selector: Some("fixture-openai".to_string()),
            model_name: "fixture-model".to_string(),
            api_key: Some("fixture-key".to_string()),
            base_url: Some("http://127.0.0.1:56599".to_string()),
            credential_uuid: None,
            reasoning_effort: Some("medium".to_string()),
            service_tier: None,
            route_protocol: None,
            toolshim: false,
            toolshim_model: None,
            model_capabilities: Some(json!({
                "capabilities": {
                    "tools": true,
                    "streaming": true,
                    "reasoning": true,
                    "reasoningEffort": {
                        "supported": true,
                        "levels": ["low", "high"]
                    }
                },
                "taskFamilies": ["chat"],
                "inputModalities": ["text"],
                "outputModalities": ["text"],
                "runtimeFeatures": ["streaming", "reasoning"]
            })),
            supports_websockets: false,
        };

        let route = resolve_chat_model_route(
            &db,
            &service,
            &request,
            &requested_selection,
            Some(&direct_provider_config),
        )
        .await
        .expect("route");

        assert!(route.resolved_route.failure.is_none());
        assert_eq!(route.selection.reasoning_effort, None);
        assert_eq!(route.resolved_route.defaults.reasoning_effort, None);
    }

    #[tokio::test]
    async fn configured_provider_with_unknown_model_fails_closed() {
        let db = test_db();
        let service = ApiKeyProviderService::new();
        let provider = service
            .add_custom_provider(
                &db,
                "Unknown Model Gateway".to_string(),
                ApiProviderType::Openai,
                "https://unknown-model.example.com/v1".to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .expect("custom provider");
        service
            .add_api_key(&db, &provider.id, "sk-unknown", None, true)
            .expect("api key");
        let request = request_for_test("hello", None, None);

        let route = resolve_chat_model_route(
            &db,
            &service,
            &request,
            &selection(&provider.id, "unknown-model"),
            None,
        )
        .await
        .expect("route");

        let failure = route.resolved_route.failure.expect("unknown model failure");
        assert_eq!(failure.category, RouteFailureCategory::ModelUnavailable);
        assert_eq!(failure.reason_code, "model_registry_metadata_missing");
        assert_eq!(failure.provider_id.as_deref(), Some(provider.id.as_str()));
        assert_eq!(failure.model_id.as_deref(), Some("unknown-model"));
        assert_eq!(
            route
                .not_possible_payload
                .as_ref()
                .and_then(|payload| payload.pointer("/routeFailure/reasonCode"))
                .and_then(Value::as_str),
            Some("model_registry_metadata_missing")
        );
    }

    #[tokio::test]
    async fn disabled_provider_returns_provider_disabled_failure() {
        let db = test_db();
        let service = ApiKeyProviderService::new();
        let provider = service
            .add_custom_provider(
                &db,
                "Disabled Gateway".to_string(),
                ApiProviderType::Openai,
                "https://disabled.example.com/v1".to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .expect("custom provider");
        service
            .add_api_key(&db, &provider.id, "sk-disabled", None, true)
            .expect("api key");
        service
            .update_provider(
                &db,
                &provider.id,
                None,
                None,
                None,
                Some(false),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .expect("disable provider");
        let request = request_for_test("hello", None, None);

        let route = resolve_chat_model_route(
            &db,
            &service,
            &request,
            &selection(&provider.id, "gpt-4.1-mini"),
            None,
        )
        .await
        .expect("route");

        let failure = route.resolved_route.failure.expect("route failure");
        assert_eq!(failure.category, RouteFailureCategory::ProviderDisabled);
        assert_eq!(failure.reason_code, "provider_disabled");
        assert_eq!(failure.provider_id.as_deref(), Some(provider.id.as_str()));
        assert_eq!(
            route
                .not_possible_payload
                .as_ref()
                .and_then(|payload| payload.pointer("/routeFailure/reasonCode"))
                .and_then(|value| value.as_str()),
            Some("provider_disabled")
        );
    }

    #[tokio::test]
    async fn declared_capability_gap_blocks_image_chat_before_provider_call() {
        let db = test_db();
        let service = ApiKeyProviderService::new();
        let provider = service
            .add_custom_provider(
                &db,
                "Text Gateway".to_string(),
                ApiProviderType::Openai,
                "https://text.example.com/v1".to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .expect("custom provider");
        service
            .add_api_key(&db, &provider.id, "sk-text-fixture", None, true)
            .expect("api key");
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
                Some(vec!["text-only-model".to_string()]),
            )
            .expect("declared model");
        let mut request = request_for_test("看图", None, None);
        request
            .input
            .push_image(agent_runtime::reply_input::RuntimeReplyInputImage {
                uri: "file:///tmp/poster.png".to_string(),
                media_type: "image/png".to_string(),
                provider_data: None,
                detail: None,
            });

        let route = resolve_chat_model_route(
            &db,
            &service,
            &request,
            &selection(&provider.id, "text-only-model"),
            None,
        )
        .await
        .expect("route");

        let failure = route.resolved_route.failure.expect("capability gap");
        assert_eq!(failure.category, RouteFailureCategory::CapabilityGap);
        assert_eq!(failure.reason_code, "capability_gap");
        assert_eq!(
            failure.capability_gap.as_deref(),
            Some("task_family:vision_understanding")
        );
        assert_eq!(
            route
                .decision_payload
                .pointer("/modelTaskRequest/requirements/inputModalities")
                .and_then(|value| value.as_array())
                .map(|items| items.iter().any(|item| item.as_str() == Some("image"))),
            Some(true)
        );
        assert_eq!(
            route
                .not_possible_payload
                .as_ref()
                .and_then(|payload| payload.pointer("/routeFailure/capabilityGap"))
                .and_then(|value| value.as_str()),
            Some("task_family:vision_understanding")
        );
    }

    #[tokio::test]
    async fn declared_agnes_vision_model_accepts_image_chat_route() {
        let db = test_db();
        let service = ApiKeyProviderService::new();
        let provider = service
            .add_custom_provider(
                &db,
                "Agnes".to_string(),
                ApiProviderType::Openai,
                "https://apihub.agnes-ai.com/v1".to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .expect("custom provider");
        service
            .add_api_key(&db, &provider.id, "sk-agnes", None, true)
            .expect("api key");
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
                Some(vec!["agnes-2.0-flash".to_string()]),
            )
            .expect("declared model");
        let mut request = request_for_test("看图", None, None);
        request
            .input
            .push_image(agent_runtime::reply_input::RuntimeReplyInputImage {
                uri: "file:///tmp/poster.png".to_string(),
                media_type: "image/png".to_string(),
                provider_data: None,
                detail: None,
            });

        let route = resolve_chat_model_route(
            &db,
            &service,
            &request,
            &selection(&provider.id, "agnes-2.0-flash"),
            None,
        )
        .await
        .expect("route");

        assert!(route.resolved_route.failure.is_none());
        assert!(route.resolved_route.capability_snapshot.capabilities.vision);
        assert!(route
            .resolved_route
            .capability_snapshot
            .input_modalities
            .contains(&"image".to_string()));
        assert!(route.not_possible_payload.is_none());
    }

    #[tokio::test]
    async fn provider_type_maps_to_route_protocol_endpoint_and_auth() {
        let db = test_db();
        let service = ApiKeyProviderService::new();
        let provider = service
            .add_custom_provider(
                &db,
                "Anthropic Gateway".to_string(),
                ApiProviderType::Anthropic,
                "https://api.anthropic.com".to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .expect("custom provider");
        service
            .add_api_key(&db, &provider.id, "sk-anthropic-fixture", None, true)
            .expect("api key");
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
                Some(vec!["claude-sonnet-4".to_string()]),
            )
            .expect("declared model");
        let request = request_for_test("hello", None, None);

        let route = resolve_chat_model_route(
            &db,
            &service,
            &request,
            &selection(&provider.id, "claude-sonnet-4"),
            None,
        )
        .await
        .expect("route");

        assert!(route.resolved_route.failure.is_none());
        assert_eq!(
            route.resolved_route.protocol,
            ProtocolKind::AnthropicMessages
        );
        assert_eq!(
            route.resolved_route.endpoint.kind,
            EndpointKind::ProviderBaseUrl
        );
        assert_eq!(
            route.resolved_route.endpoint.base_url.as_deref(),
            Some("https://api.anthropic.com")
        );
        assert_eq!(route.resolved_route.auth.kind, AuthKind::ApiKeyRef);
        assert_eq!(
            route.resolved_route.auth.provider_id.as_deref(),
            Some(provider.id.as_str())
        );
    }

    #[tokio::test]
    async fn managed_lime_hub_tenant_provider_without_key_resolves_no_auth_route() {
        let db = test_db();
        let service = ApiKeyProviderService::new();
        let provider = service
            .add_custom_provider(
                &db,
                "Lime Hub".to_string(),
                ApiProviderType::Openai,
                "https://llm.limeai.run/v1#lime_tenant_id=tenant-0001".to_string(),
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
                Some(vec!["agnes-2.0-flash".to_string()]),
            )
            .expect("declared model");
        let request = request_for_test("hello", None, None);

        let route = resolve_chat_model_route(
            &db,
            &service,
            &request,
            &selection(&provider.id, "agnes-2.0-flash"),
            None,
        )
        .await
        .expect("route");

        assert!(route.resolved_route.failure.is_none());
        assert_eq!(route.resolved_route.protocol, ProtocolKind::OpenaiChat);
        assert_eq!(
            route.resolved_route.endpoint.kind,
            EndpointKind::OpenaiCompatible
        );
        assert_eq!(
            route.resolved_route.endpoint.base_url.as_deref(),
            Some("https://llm.limeai.run/v1#lime_tenant_id=tenant-0001")
        );
        assert_eq!(route.resolved_route.auth.kind, AuthKind::NoAuth);
        assert_eq!(
            route
                .decision_payload
                .pointer("/providerReadiness/reasonCode")
                .and_then(|value| value.as_str()),
            None
        );
    }

    #[tokio::test]
    async fn unready_coding_slot_emits_fallback_payload_for_ready_base_slot() {
        let db = test_db();
        let service = ApiKeyProviderService::new();
        let coding = service
            .add_custom_provider(
                &db,
                "Coding Gateway".to_string(),
                ApiProviderType::Openai,
                "https://coding.example.com/v1".to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .expect("custom provider");
        service
            .initialize_system_providers(&db)
            .expect("system providers");
        service
            .add_api_key(&db, "openai", "sk-openai", None, true)
            .expect("openai api key");
        service
            .update_provider(
                &db,
                "openai",
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
                Some(vec!["gpt-4.1-mini".to_string()]),
            )
            .expect("declared base model");
        let request = request_for_test(
            "hello",
            None,
            Some(json!({
                "harness": {
                    "coding_model_slots": {
                        "coding": {
                            "provider": coding.id,
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
        let requested_selection = model_routing::selection_from_profile_model_slot(&request)
            .expect("requested selection");

        let route = resolve_chat_model_route(&db, &service, &request, &requested_selection, None)
            .await
            .expect("route");

        assert_eq!(route.selection.provider, "openai");
        assert_eq!(route.selection.model, "gpt-4.1-mini");
        assert!(route.resolved_route.failure.is_none());
        assert_eq!(route.service_model_slot(), "base");
        assert_eq!(
            route
                .fallback_payload
                .as_ref()
                .and_then(|payload| payload.pointer("/fallbackApplied"))
                .and_then(|value| value.as_bool()),
            Some(true)
        );
        assert_eq!(
            route
                .fallback_payload
                .as_ref()
                .and_then(|payload| payload.pointer("/routingAttempts"))
                .and_then(|value| value.as_array())
                .map(Vec::len),
            Some(2)
        );
    }

    #[tokio::test]
    async fn durable_credential_binds_scoped_metadata_route_and_evidence() {
        let db = test_db();
        let service = ApiKeyProviderService::new();
        let api_host = "https://api.xiaomimimo.com/v1";
        let provider = service
            .add_custom_provider(
                &db,
                "Scoped Route Fixture".to_string(),
                ApiProviderType::Openai,
                api_host.to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .expect("custom provider");
        let key_a = service
            .add_api_key(&db, &provider.id, "scope-key-a", None, true)
            .expect("key A");
        let key_b = service
            .add_api_key(&db, &provider.id, "scope-key-b", None, false)
            .expect("key B");
        ModelRegistryService::new(db.clone())
            .fetch_models_from_api_with_hints(
                &provider.id,
                api_host,
                "scope-key-b",
                Some(ApiProviderType::Openai),
                &["scoped-route-model".to_string()],
            )
            .await
            .expect("seed key B scoped cache");

        let request = request_for_test("hello", None, None);
        let requested_selection = selection(&provider.id, "scoped-route-model");
        let ref_a = lime_core::models::runtime_api_key_credential_uuid(&key_a.id);
        let ref_b = lime_core::models::runtime_api_key_credential_uuid(&key_b.id);
        assert!(service
            .select_runtime_credential_by_ref(&db, &provider.id, &ref_a)
            .expect("read key A")
            .is_some());
        assert!(service
            .select_runtime_credential_by_ref(&db, &provider.id, &ref_b)
            .expect("read key B")
            .is_some());

        let route_b = assemble_chat_model_route(
            &db,
            &service,
            &request,
            &requested_selection,
            None,
            prepare_chat_model_route(&db, &service, &request, &requested_selection, None)
                .expect("prepare key B route"),
            Some(&ref_b),
        )
        .await
        .expect("assemble key B route");

        assert!(route_b.resolved_route.failure.is_none());
        assert_eq!(
            route_b.resolved_route.auth.credential_ref.as_deref(),
            Some(ref_b.as_str())
        );
        assert_eq!(
            route_b
                .decision_payload
                .pointer("/modelRegistry/source")
                .and_then(Value::as_str),
            Some("provider_models_cache")
        );
        assert_eq!(
            route_b
                .decision_payload
                .pointer("/resolvedRoute/auth/credentialRef")
                .and_then(Value::as_str),
            Some(ref_b.as_str())
        );
        let evidence = route_b.decision_payload.to_string();
        assert!(!evidence.contains("scope-key-a"));
        assert!(!evidence.contains("scope-key-b"));

        let route_a = assemble_chat_model_route(
            &db,
            &service,
            &request,
            &requested_selection,
            None,
            prepare_chat_model_route(&db, &service, &request, &requested_selection, None)
                .expect("prepare key A route"),
            Some(&ref_a),
        )
        .await
        .expect("assemble key A route");
        assert_eq!(
            route_a
                .resolved_route
                .failure
                .as_ref()
                .map(|failure| failure.reason_code.as_str()),
            Some("model_registry_metadata_missing")
        );
        assert_eq!(
            route_a
                .decision_payload
                .pointer("/modelRegistry/source")
                .and_then(Value::as_str),
            Some("runtime_selection_only")
        );
    }
}
