use super::request_context::RuntimeModelSelection;
use crate::model_route_assembly::{DirectRouteConfig, ModelRouteSelection};
use crate::model_task_contract::{build_model_task_request, ModelTaskRequestInput};
use crate::ExecutionRequest;
use agent_protocol::ModelId;
use agent_runtime::turn_executor::TurnProviderConfiguration;
use app_server_protocol::{
    AuthKind, CapabilitySnapshot, ModelRefSource, ModelTaskKind, ModelTaskRequest, ModelTaskSource,
    ProtocolKind, ResolvedModelRoute,
};
use lime_agent::{
    route_protocol_from_session_provider_config, supports_direct_route,
    ModelRouteProviderConfiguration, SessionProviderConfig,
};
use lime_core::database::dao::api_key_provider::ProviderWithKeys;
use model_provider::runtime_provider::RuntimeProviderAuth;
use model_provider::{ModelProviderProtocol, ModelRoute};
use serde_json::{json, Value};

pub(super) fn chat_task_request_from_runtime(
    request: &ExecutionRequest,
    selection: &RuntimeModelSelection,
    routing_payload: &Value,
) -> ModelTaskRequest {
    let routing_slot = string_field(routing_payload, &["serviceModelSlot", "service_model_slot"])
        .unwrap_or_else(|| "coding".to_string());
    let mut task_families = vec!["chat".to_string()];
    let mut input_modalities = vec!["text".to_string()];
    let output_modalities = vec!["text".to_string()];
    let runtime_features = vec!["streaming".to_string()];
    let mut capabilities = string_array_field(
        routing_payload,
        &["requiredCapabilities", "required_capabilities"],
    );

    if request.input.has_images() {
        push_unique(&mut task_families, "vision_understanding".to_string());
        push_unique(&mut input_modalities, "image".to_string());
        push_unique(&mut capabilities, "vision".to_string());
    }
    push_unique(&mut capabilities, "streaming".to_string());

    build_model_task_request(ModelTaskRequestInput {
        task_kind: ModelTaskKind::Chat,
        source: ModelTaskSource::AgentTurn,
        provider_id: Some(selection.provider.clone()),
        model_id: Some(selection.model.clone()),
        model_ref_source: model_ref_source(selection.source),
        modality_contract_key: Some("chat".to_string()),
        routing_slot: Some(routing_slot),
        task_families,
        input_modalities,
        output_modalities,
        runtime_features,
        capabilities,
        session_id: Some(request.session.session_id.clone()),
        thread_id: non_empty(Some(&request.turn.thread_id))
            .or_else(|| non_empty(Some(&request.session.thread_id))),
        turn_id: Some(request.turn.turn_id.clone()),
        content_id: None,
        trace_id: None,
    })
}

pub(super) fn resolved_route_from_runtime(
    task_request: &ModelTaskRequest,
    selection: &RuntimeModelSelection,
    routing_payload: &Value,
    provider: Option<&ProviderWithKeys>,
    credential_ref: Option<&str>,
    direct_provider_config: Option<&SessionProviderConfig>,
) -> ResolvedModelRoute {
    crate::model_route_assembly::resolved_route_from_task_with_credential(
        task_request,
        ModelRouteSelection {
            provider_id: &selection.provider,
            model_id: &selection.model,
            model_ref_source: model_ref_source(selection.source),
            reasoning_effort: selection.reasoning_effort.as_deref(),
        },
        routing_payload,
        provider,
        credential_ref,
        direct_provider_config.map(direct_route_config),
    )
}

pub(super) fn model_route_from_runtime(
    selection: &RuntimeModelSelection,
    resolved_route: &ResolvedModelRoute,
) -> ModelRoute {
    let service_model_slot = resolved_route.decision.service_model_slot.clone();
    ModelRoute {
        provider: selection.provider.clone(),
        model: ModelId::new(selection.model.clone()),
        protocol: model_provider_protocol_from_route_protocol(&resolved_route.protocol),
        capabilities: model_route_capabilities(&resolved_route.capability_snapshot),
        metadata: json!({
            "source": selection.source,
            "serviceModelSlot": service_model_slot.clone(),
            "service_model_slot": service_model_slot,
        }),
    }
}

pub(super) fn provider_configuration_from_runtime(
    selection: &RuntimeModelSelection,
    resolved_route: &ResolvedModelRoute,
    direct_provider_config: Option<SessionProviderConfig>,
    service_tier: Option<String>,
) -> ModelRouteProviderConfiguration {
    let mut direct_provider_config = direct_provider_config
        .or_else(|| no_auth_direct_provider_config_from_route(selection, resolved_route));
    if let Some(config) = direct_provider_config.as_mut() {
        config.reasoning_effort = selection.reasoning_effort.clone();
        config.service_tier = service_tier.clone();
    }

    ModelRouteProviderConfiguration {
        turn_provider: TurnProviderConfiguration {
            route: model_route_from_runtime(selection, resolved_route),
            reasoning_effort: selection.reasoning_effort.clone(),
        },
        service_tier,
        route_protocol: Some(resolved_route.protocol.clone()),
        credential_ref: resolved_route.auth.credential_ref.clone(),
        direct_provider_config,
        auth: runtime_provider_auth(&resolved_route.auth.kind),
        api_version: resolved_route.endpoint.api_version.clone(),
    }
}

fn runtime_provider_auth(auth: &AuthKind) -> RuntimeProviderAuth {
    match auth {
        AuthKind::NoAuth => RuntimeProviderAuth::NoAuth,
        AuthKind::ApiKeyRef | AuthKind::DirectApiKey => RuntimeProviderAuth::ApiKey,
        AuthKind::OemManaged => RuntimeProviderAuth::OemManaged,
    }
}

fn no_auth_direct_provider_config_from_route(
    selection: &RuntimeModelSelection,
    resolved_route: &ResolvedModelRoute,
) -> Option<SessionProviderConfig> {
    if resolved_route.auth.kind != AuthKind::NoAuth {
        return None;
    }

    supports_direct_route(&selection.provider, &resolved_route.protocol).then_some(())?;
    let base_url = resolved_route
        .endpoint
        .base_url
        .as_deref()
        .and_then(|value| non_empty(Some(value)))?;

    Some(SessionProviderConfig {
        provider_name: selection.provider.clone(),
        provider_selector: Some(selection.provider.clone()),
        model_name: selection.model.clone(),
        api_key: None,
        base_url: Some(base_url),
        api_version: resolved_route.endpoint.api_version.clone(),
        credential_uuid: None,
        reasoning_effort: selection.reasoning_effort.clone(),
        service_tier: None,
        route_protocol: Some(resolved_route.protocol.clone()),
        toolshim: false,
        toolshim_model: None,
        model_capabilities: Some(serde_json::to_value(&resolved_route.capability_snapshot).ok()?),
        supports_websockets: false,
    })
}

fn model_ref_source(source: &str) -> ModelRefSource {
    match source {
        "runtime_options" => ModelRefSource::RuntimeOptions,
        "runtime_request_provider_config" => ModelRefSource::RuntimeRequest,
        "profile_model_slot" => ModelRefSource::ProfileSlot,
        "session_default" => ModelRefSource::SessionDefault,
        "direct_provider_config" => ModelRefSource::DirectProviderConfig,
        _ => ModelRefSource::Explicit,
    }
}

fn model_provider_protocol_from_route_protocol(protocol: &ProtocolKind) -> ModelProviderProtocol {
    match protocol {
        ProtocolKind::OpenaiResponses | ProtocolKind::CodexResponses => {
            ModelProviderProtocol::Responses
        }
        ProtocolKind::OpenaiChat => ModelProviderProtocol::ChatCompletions,
        ProtocolKind::GeminiGenerateContent => ModelProviderProtocol::GeminiGenerateContent,
        ProtocolKind::VertexGemini => ModelProviderProtocol::GeminiGenerateContent,
        other => ModelProviderProtocol::Custom(route_protocol_name(other).to_string()),
    }
}

fn route_protocol_name(protocol: &ProtocolKind) -> &'static str {
    match protocol {
        ProtocolKind::OpenaiChat => "openai_chat",
        ProtocolKind::OpenaiResponses => "openai_responses",
        ProtocolKind::OpenaiImages => "openai_images",
        ProtocolKind::AnthropicMessages => "anthropic_messages",
        ProtocolKind::GeminiGenerateContent => "gemini_generate_content",
        ProtocolKind::Fal => "fal",
        ProtocolKind::XaiVideo => "xai_video",
        ProtocolKind::BedrockConverse => "bedrock_converse",
        ProtocolKind::VertexGemini => "vertex_gemini",
        ProtocolKind::CodexResponses => "codex_responses",
        ProtocolKind::Unknown => "unknown",
    }
}

fn model_route_capabilities(capability_snapshot: &CapabilitySnapshot) -> Vec<String> {
    let mut capabilities = capability_snapshot.runtime_features.clone();
    let model_capabilities = &capability_snapshot.capabilities;
    push_capability(&mut capabilities, "vision", model_capabilities.vision);
    push_capability(&mut capabilities, "tools", model_capabilities.tools);
    push_capability(&mut capabilities, "streaming", model_capabilities.streaming);
    push_capability(&mut capabilities, "json_mode", model_capabilities.json_mode);
    push_capability(
        &mut capabilities,
        "function_calling",
        model_capabilities.function_calling,
    );
    push_capability(&mut capabilities, "reasoning", model_capabilities.reasoning);
    capabilities
}

fn push_capability(capabilities: &mut Vec<String>, capability: &str, enabled: bool) {
    if enabled {
        push_unique(capabilities, capability.to_string());
    }
}

fn push_unique(values: &mut Vec<String>, value: String) {
    if !values.iter().any(|existing| existing == &value) {
        values.push(value);
    }
}

fn non_empty(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn direct_route_config(config: &SessionProviderConfig) -> DirectRouteConfig<'_> {
    DirectRouteConfig {
        provider_name: &config.provider_name,
        api_key_present: config
            .api_key
            .as_deref()
            .is_some_and(|key| !key.trim().is_empty()),
        base_url: config.base_url.as_deref(),
        api_version: config.api_version.as_deref(),
        credential_ref: config.credential_uuid.as_deref(),
        protocol: route_protocol_from_session_provider_config(config),
        toolshim: config.toolshim,
        toolshim_model: config.toolshim_model.as_deref(),
    }
}

fn string_field(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|key| value.get(*key))
        .find_map(Value::as_str)
        .and_then(|value| non_empty(Some(value)))
}

fn string_array_field(value: &Value, keys: &[&str]) -> Vec<String> {
    keys.iter()
        .filter_map(|key| value.get(*key))
        .find_map(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .filter_map(|value| non_empty(Some(value)))
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_backend::tests::request_for_test;
    use app_server_protocol::RouteFailureCategory;
    use serde_json::json;

    #[test]
    fn chat_task_request_adds_vision_requirement_for_image_attachments() {
        let mut request = request_for_test("看图", None, None);
        request
            .input
            .push_image(agent_runtime::reply_input::RuntimeReplyInputImage {
                uri: "file:///tmp/poster.png".to_string(),
                media_type: "image/png".to_string(),
                provider_data: None,
                detail: None,
            });
        let selection = RuntimeModelSelection {
            provider: "openai".to_string(),
            model: "gpt-4.1".to_string(),
            source: "runtime_options",
            reasoning_effort: None,
        };
        let task_request = chat_task_request_from_runtime(
            &request,
            &selection,
            &json!({
                "serviceModelSlot": "coding",
                "requiredCapabilities": ["tools", "streaming"]
            }),
        );

        assert_eq!(task_request.task_kind, ModelTaskKind::Chat);
        assert!(task_request
            .requirements
            .task_families
            .contains(&"vision_understanding".to_string()));
        assert!(task_request
            .requirements
            .input_modalities
            .contains(&"image".to_string()));
        assert!(task_request
            .requirements
            .capabilities
            .contains(&"vision".to_string()));
    }

    #[test]
    fn resolved_route_reports_capability_gap_for_image_input_without_vision() {
        let mut request = request_for_test("看图", None, None);
        request
            .input
            .push_image(agent_runtime::reply_input::RuntimeReplyInputImage {
                uri: "file:///tmp/poster.png".to_string(),
                media_type: "image/png".to_string(),
                provider_data: None,
                detail: None,
            });
        let selection = RuntimeModelSelection {
            provider: "openai".to_string(),
            model: "text-only".to_string(),
            source: "runtime_options",
            reasoning_effort: None,
        };
        let routing_payload = json!({
            "providerReadiness": {
                "ready": true,
                "status": "ready"
            },
            "serviceModelSlot": "coding",
            "modelRegistry": {
                "source": "api",
                "reasonCode": "matched",
                "modelCapabilities": {
                    "provenance": "provider_explicit",
                    "capabilities": {
                        "vision": false,
                        "tools": true,
                        "streaming": true
                    },
                    "taskFamilies": ["chat"],
                    "inputModalities": ["text"],
                    "outputModalities": ["text"],
                    "runtimeFeatures": ["streaming", "tool_calling"]
                }
            }
        });
        let task_request = chat_task_request_from_runtime(&request, &selection, &routing_payload);
        let route = resolved_route_from_runtime(
            &task_request,
            &selection,
            &routing_payload,
            None,
            None,
            None,
        );

        let failure = route.failure.expect("capability failure");
        assert_eq!(failure.category, RouteFailureCategory::CapabilityGap);
        assert_eq!(failure.reason_code, "capability_gap");
        assert_eq!(
            failure.capability_gap.as_deref(),
            Some("task_family:vision_understanding")
        );
        assert_eq!(
            route.decision.capability_gap.as_deref(),
            Some("task_family:vision_understanding")
        );
    }

    #[test]
    fn resolved_route_blocks_inferred_capability_snapshot() {
        let selection = RuntimeModelSelection {
            provider: "fixture-openai".to_string(),
            model: "fixture-model".to_string(),
            source: "runtime_request_provider_config",
            reasoning_effort: None,
        };
        let task_request = build_model_task_request(ModelTaskRequestInput {
            task_kind: ModelTaskKind::Chat,
            source: ModelTaskSource::AgentTurn,
            provider_id: Some(selection.provider.clone()),
            model_id: Some(selection.model.clone()),
            model_ref_source: ModelRefSource::RuntimeRequest,
            modality_contract_key: Some("chat".to_string()),
            routing_slot: Some("coding".to_string()),
            task_families: vec!["chat".to_string(), "vision_understanding".to_string()],
            input_modalities: vec!["text".to_string(), "image".to_string()],
            output_modalities: vec!["text".to_string()],
            runtime_features: vec!["streaming".to_string()],
            capabilities: vec!["vision".to_string(), "streaming".to_string()],
            session_id: None,
            thread_id: None,
            turn_id: None,
            content_id: None,
            trace_id: None,
        });
        let routing_payload = json!({
            "providerReadiness": {
                "ready": true,
                "status": "ready"
            },
            "modelRegistry": {
                "source": "provider_declared_model",
                "status": "matched",
                "reasonCode": "matched_provider_models",
                "modelCapabilities": {
                    "provenance": "inferred_hint",
                    "capabilities": {
                        "vision": true,
                        "streaming": true
                    },
                    "taskFamilies": ["chat", "vision_understanding"],
                    "inputModalities": ["text", "image"],
                    "outputModalities": ["text"],
                    "runtimeFeatures": ["streaming"]
                }
            }
        });
        let route = resolved_route_from_runtime(
            &task_request,
            &selection,
            &routing_payload,
            None,
            None,
            None,
        );

        let failure = route.failure.expect("missing capability snapshot failure");
        assert_eq!(failure.category, RouteFailureCategory::CapabilityGap);
        assert_eq!(failure.reason_code, "capability_snapshot_missing");
        assert_eq!(
            failure.capability_gap.as_deref(),
            Some("capability_snapshot:missing")
        );
        assert_eq!(
            route.decision.capability_gap.as_deref(),
            Some("capability_snapshot:missing")
        );
    }

    #[test]
    fn model_route_from_runtime_projects_lime_provider_route() {
        let selection = RuntimeModelSelection {
            provider: "openai".to_string(),
            model: "gpt-4.1".to_string(),
            source: "runtime_options",
            reasoning_effort: None,
        };
        let task_request = build_model_task_request(ModelTaskRequestInput {
            task_kind: ModelTaskKind::Chat,
            source: ModelTaskSource::AgentTurn,
            provider_id: Some(selection.provider.clone()),
            model_id: Some(selection.model.clone()),
            model_ref_source: ModelRefSource::RuntimeOptions,
            modality_contract_key: Some("chat".to_string()),
            routing_slot: Some("coding".to_string()),
            task_families: vec!["chat".to_string()],
            input_modalities: vec!["text".to_string()],
            output_modalities: vec!["text".to_string()],
            runtime_features: vec!["streaming".to_string()],
            capabilities: vec!["tools".to_string(), "streaming".to_string()],
            session_id: None,
            thread_id: None,
            turn_id: None,
            content_id: None,
            trace_id: None,
        });
        let routing_payload = json!({
            "providerReadiness": {
                "ready": true,
                "status": "ready"
            },
            "serviceModelSlot": "coding",
            "modelRegistry": {
                "source": "api",
                "reasonCode": "matched",
                "modelCapabilities": {
                    "provenance": "provider_explicit",
                    "capabilities": {
                        "vision": false,
                        "tools": true,
                        "streaming": true
                    },
                    "taskFamilies": ["chat"],
                    "inputModalities": ["text"],
                    "outputModalities": ["text"],
                    "runtimeFeatures": ["streaming"]
                }
            }
        });
        let resolved_route = resolved_route_from_runtime(
            &task_request,
            &selection,
            &routing_payload,
            None,
            None,
            None,
        );
        let model_route = model_route_from_runtime(&selection, &resolved_route);

        assert_eq!(model_route.provider, "openai");
        assert_eq!(model_route.model.as_str(), "gpt-4.1");
        assert_eq!(model_route.protocol, ModelProviderProtocol::ChatCompletions);
        assert!(model_route.capabilities.contains(&"tools".to_string()));
        assert!(model_route.capabilities.contains(&"streaming".to_string()));
        assert_eq!(
            model_route
                .metadata
                .get("serviceModelSlot")
                .and_then(Value::as_str),
            Some("coding")
        );
    }

    #[test]
    fn model_route_protocol_projects_gemini_to_dedicated_adapter() {
        assert_eq!(
            model_provider_protocol_from_route_protocol(&ProtocolKind::GeminiGenerateContent),
            ModelProviderProtocol::GeminiGenerateContent
        );
    }

    #[test]
    fn provider_configuration_projects_no_auth_openai_route_to_direct_config() {
        let selection = RuntimeModelSelection {
            provider: "lime-hub".to_string(),
            model: "agnes-2.0-flash".to_string(),
            source: "runtime_options",
            reasoning_effort: None,
        };
        let task_request = build_model_task_request(ModelTaskRequestInput {
            task_kind: ModelTaskKind::Chat,
            source: ModelTaskSource::AgentTurn,
            provider_id: Some(selection.provider.clone()),
            model_id: Some(selection.model.clone()),
            model_ref_source: ModelRefSource::RuntimeOptions,
            modality_contract_key: Some("chat".to_string()),
            routing_slot: Some("coding".to_string()),
            task_families: vec!["chat".to_string()],
            input_modalities: vec!["text".to_string()],
            output_modalities: vec!["text".to_string()],
            runtime_features: vec!["streaming".to_string()],
            capabilities: vec!["streaming".to_string()],
            session_id: None,
            thread_id: None,
            turn_id: None,
            content_id: None,
            trace_id: None,
        });
        let routing_payload = json!({
            "providerReadiness": {
                "ready": true,
                "status": "ready"
            },
            "modelRegistry": {
                "source": "provider_declared_model",
                "reasonCode": "matched_provider_models",
                "modelCapabilities": {
                    "provenance": "provider_explicit",
                    "capabilities": {
                        "streaming": true
                    },
                    "taskFamilies": ["chat"],
                    "inputModalities": ["text"],
                    "outputModalities": ["text"],
                    "runtimeFeatures": ["streaming"]
                }
            }
        });
        let route_seed = SessionProviderConfig {
            provider_name: "openai".to_string(),
            provider_selector: Some("lime-hub".to_string()),
            model_name: "agnes-2.0-flash".to_string(),
            api_key: None,
            base_url: Some("https://llm.limeai.run/v1#lime_tenant_id=tenant-0001".to_string()),
            api_version: None,
            credential_uuid: None,
            reasoning_effort: None,
            service_tier: None,
            route_protocol: Some(ProtocolKind::OpenaiChat),
            toolshim: false,
            toolshim_model: None,
            model_capabilities: None,
            supports_websockets: false,
        };
        let resolved_route = resolved_route_from_runtime(
            &task_request,
            &selection,
            &routing_payload,
            None,
            None,
            Some(&route_seed),
        );

        let configuration =
            provider_configuration_from_runtime(&selection, &resolved_route, None, None);
        let direct_config = configuration
            .direct_provider_config
            .expect("no-auth route should create direct provider config");

        assert_eq!(resolved_route.auth.kind, AuthKind::NoAuth);
        assert_eq!(configuration.auth, RuntimeProviderAuth::NoAuth);
        assert_eq!(direct_config.provider_name, "lime-hub");
        assert_eq!(direct_config.provider_selector.as_deref(), Some("lime-hub"));
        assert_eq!(direct_config.model_name, "agnes-2.0-flash");
        assert!(direct_config.api_key.is_none());
        assert_eq!(
            direct_config.base_url.as_deref(),
            Some("https://llm.limeai.run/v1#lime_tenant_id=tenant-0001")
        );
        assert_eq!(direct_config.route_protocol, Some(ProtocolKind::OpenaiChat));
    }

    #[test]
    fn provider_configuration_preserves_resolved_credential_ref() {
        let selection = RuntimeModelSelection {
            provider: "openai".to_string(),
            model: "gpt-4.1".to_string(),
            source: "runtime_options",
            reasoning_effort: None,
        };
        let resolved_route = ResolvedModelRoute {
            auth: app_server_protocol::AuthMaterialRef {
                credential_ref: Some("runtime-api-key:credential-a".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };

        let configuration =
            provider_configuration_from_runtime(&selection, &resolved_route, None, None);

        assert_eq!(
            configuration.credential_ref.as_deref(),
            Some("runtime-api-key:credential-a")
        );
    }

    #[test]
    fn provider_configuration_preserves_azure_responses_wire_metadata() {
        let selection = RuntimeModelSelection {
            provider: "azure-openai".to_string(),
            model: "gpt-5.4".to_string(),
            source: "runtime_options",
            reasoning_effort: None,
        };
        let resolved_route = ResolvedModelRoute {
            protocol: ProtocolKind::OpenaiResponses,
            endpoint: app_server_protocol::EndpointInfo {
                base_url: Some("https://resource.openai.azure.com".to_string()),
                api_version: Some("2025-04-01-preview".to_string()),
                ..Default::default()
            },
            auth: app_server_protocol::AuthMaterialRef {
                kind: AuthKind::ApiKeyRef,
                provider_id: Some("azure-openai".to_string()),
                credential_ref: Some("runtime-api-key:azure-key".to_string()),
                header_name: Some("api-key".to_string()),
                header_prefix: None,
            },
            ..Default::default()
        };

        let configuration =
            provider_configuration_from_runtime(&selection, &resolved_route, None, None);

        assert_eq!(
            configuration.route_protocol,
            Some(ProtocolKind::OpenaiResponses)
        );
        assert_eq!(
            configuration.api_version.as_deref(),
            Some("2025-04-01-preview")
        );
        assert_eq!(configuration.auth, RuntimeProviderAuth::ApiKey);
        assert_eq!(
            configuration.credential_ref.as_deref(),
            Some("runtime-api-key:azure-key")
        );
    }

    #[test]
    fn provider_configuration_overwrites_stale_direct_reasoning_with_effective_selection() {
        let selection = RuntimeModelSelection {
            provider: "openai".to_string(),
            model: "gpt-codex".to_string(),
            source: "runtime_options",
            reasoning_effort: None,
        };
        let direct_config = SessionProviderConfig {
            provider_name: "openai".to_string(),
            provider_selector: Some("openai".to_string()),
            model_name: "gpt-codex".to_string(),
            api_key: Some("test-key".to_string()),
            base_url: Some("https://api.example.test/v1".to_string()),
            api_version: None,
            credential_uuid: None,
            reasoning_effort: Some("high".to_string()),
            service_tier: None,
            route_protocol: Some(ProtocolKind::OpenaiResponses),
            toolshim: false,
            toolshim_model: None,
            model_capabilities: None,
            supports_websockets: false,
        };

        let configuration = provider_configuration_from_runtime(
            &selection,
            &ResolvedModelRoute::default(),
            Some(direct_config),
            None,
        );

        assert_eq!(configuration.turn_provider.reasoning_effort, None);
        assert_eq!(
            configuration
                .direct_provider_config
                .expect("direct config")
                .reasoning_effort,
            None
        );
    }
}
