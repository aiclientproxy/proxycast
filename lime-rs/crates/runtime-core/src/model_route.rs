use crate::model_routing::{
    routing_decision_payload, routing_fallback_applied_payload,
    routing_not_possible_payload_with_attempts, RoutingResolution, RuntimeModelSelection,
};
use crate::model_task::{capability_snapshot_from_model_capabilities, route_capability_gap};
use app_server_protocol::{
    AuthKind, AuthMaterialRef, CapabilitySnapshot, EndpointInfo, EndpointKind, FramingKind,
    ModelRef, ModelRefSource, ModelTaskKind, ModelTaskRequest, ProtocolKind, ResolvedModelRoute,
    RouteDefaults, RouteFailure, RouteFailureCategory, RoutingDecision, TransportKind,
};
use serde_json::{json, Value};
use std::borrow::Cow;

pub struct ModelRouteSelection<'a> {
    pub provider_id: &'a str,
    pub model_id: &'a str,
    pub model_ref_source: ModelRefSource,
    pub reasoning_effort: Option<&'a str>,
}

pub struct DirectRouteConfig<'a> {
    pub provider_name: &'a str,
    pub api_key_present: bool,
    pub base_url: Option<&'a str>,
    pub credential_ref: Option<&'a str>,
    pub protocol: Option<ProtocolKind>,
    pub toolshim: bool,
    pub toolshim_model: Option<&'a str>,
}

pub struct ModelRouteProvider<'a> {
    pub provider_id: &'a str,
    pub provider_type: Cow<'a, str>,
    pub base_url: Option<&'a str>,
    pub api_version: Option<&'a str>,
    pub project: Option<&'a str>,
    pub location: Option<&'a str>,
    pub region: Option<&'a str>,
    pub credential_ref: Option<String>,
    pub auth_header: &'a str,
    pub auth_prefix: Option<&'a str>,
    pub prompt_cache_mode: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RouteResolutionEvidencePayloads {
    pub decision_payload: Value,
    pub fallback_payload: Option<Value>,
    pub not_possible_payload: Option<Value>,
}

pub fn resolved_route_from_task(
    task_request: &ModelTaskRequest,
    selection: ModelRouteSelection<'_>,
    routing_payload: &Value,
    provider: Option<&ModelRouteProvider<'_>>,
    direct_config: Option<DirectRouteConfig<'_>>,
) -> ResolvedModelRoute {
    let readiness = routing_payload
        .get("providerReadiness")
        .or_else(|| routing_payload.get("provider_readiness"));
    let ready = readiness
        .and_then(|value| value.get("ready"))
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let capability_snapshot = capability_snapshot(routing_payload);
    let has_capability_snapshot = has_declared_capability_snapshot(routing_payload);
    let protocol = resolve_protocol(
        task_request,
        &capability_snapshot,
        &selection,
        provider,
        direct_config.as_ref(),
    );
    let capability_gap = if has_capability_snapshot {
        route_capability_gap(task_request, &capability_snapshot)
    } else {
        None
    };
    let unresolved_registry_failure = (ready && !has_capability_snapshot)
        .then(|| unknown_model_or_capability_failure(&selection, routing_payload));
    let mut decision = routing_decision(routing_payload);
    decision.capability_gap = capability_gap.clone().or_else(|| {
        unresolved_registry_failure
            .as_ref()
            .and_then(|failure| failure.capability_gap.clone())
    });
    let unsupported_protocol_failure =
        (!protocol_supported_for_task(task_request.task_kind.clone(), &protocol))
            .then(|| unsupported_protocol_route_failure(&selection, &protocol));
    let failure = if !ready {
        Some(route_failure(&selection, readiness))
    } else {
        unresolved_registry_failure
            .or(unsupported_protocol_failure)
            .or_else(|| capability_gap.map(|gap| capability_route_failure(&selection, gap)))
    };

    ResolvedModelRoute {
        model_ref: task_request.model_ref.clone().unwrap_or_else(|| ModelRef {
            provider_id: selection.provider_id.to_string(),
            model_id: selection.model_id.to_string(),
            variant: None,
            routing_slot: task_request.routing_slot.clone(),
            source: selection.model_ref_source.clone(),
        }),
        protocol: protocol.clone(),
        endpoint: endpoint_info(provider, direct_config.as_ref()),
        auth: auth_ref(&selection, provider, direct_config.as_ref()),
        transport: TransportKind::Http,
        framing: framing_for_protocol(&protocol),
        defaults: route_defaults(&selection, provider, direct_config.as_ref()),
        capability_snapshot,
        decision,
        failure,
    }
}

pub fn route_resolution_evidence_payloads(
    requested_selection: &RuntimeModelSelection,
    routing_resolution: &RoutingResolution,
    model_registry_payload: &Value,
    task_request: &ModelTaskRequest,
    resolved_route: &ResolvedModelRoute,
) -> RouteResolutionEvidencePayloads {
    let routing_payload = routing_decision_payload(
        &routing_resolution.selection,
        &routing_resolution.routing,
        &routing_resolution.readiness,
        model_registry_payload,
    );
    let decision_payload = route_evidence_payload(routing_payload, task_request, resolved_route);
    let fallback_payload = (requested_selection != &routing_resolution.selection).then(|| {
        route_evidence_payload(
            routing_fallback_applied_payload(
                requested_selection,
                &routing_resolution.selection,
                &routing_resolution.routing,
                &routing_resolution.readiness,
                model_registry_payload,
                &routing_resolution.attempted,
            ),
            task_request,
            resolved_route,
        )
    });
    let not_possible_payload = resolved_route.failure.as_ref().map(|_| {
        route_evidence_payload(
            routing_not_possible_payload_with_attempts(
                &routing_resolution.selection,
                &routing_resolution.routing,
                &routing_resolution.readiness,
                model_registry_payload,
                &routing_resolution.attempted,
            ),
            task_request,
            resolved_route,
        )
    });

    RouteResolutionEvidencePayloads {
        decision_payload,
        fallback_payload,
        not_possible_payload,
    }
}

pub fn route_evidence_payload(
    mut payload: Value,
    task_request: &ModelTaskRequest,
    resolved_route: &ResolvedModelRoute,
) -> Value {
    let task_request_value = serde_json::to_value(task_request).unwrap_or_else(|_| json!({}));
    let resolved_route_value = safe_route_evidence_value(resolved_route);
    if let Some(object) = payload.as_object_mut() {
        object.insert("modelTaskRequest".to_string(), task_request_value.clone());
        object.insert("model_task_request".to_string(), task_request_value);
        object.insert("resolvedRoute".to_string(), resolved_route_value.clone());
        object.insert("resolved_route".to_string(), resolved_route_value);
        if let Some(failure) = resolved_route.failure.as_ref() {
            let failure_value = serde_json::to_value(failure).unwrap_or_else(|_| json!({}));
            object.insert("routeFailure".to_string(), failure_value.clone());
            object.insert("route_failure".to_string(), failure_value);
            object.insert("failureCategory".to_string(), json!(failure.category));
            object.insert("failure_category".to_string(), json!(failure.category));
            object.insert(
                "reasonCode".to_string(),
                Value::String(failure.reason_code.clone()),
            );
            object.insert(
                "reason_code".to_string(),
                Value::String(failure.reason_code.clone()),
            );
            if let Some(capability_gap) = failure.capability_gap.as_ref() {
                object.insert(
                    "capabilityGap".to_string(),
                    Value::String(capability_gap.clone()),
                );
                object.insert(
                    "capability_gap".to_string(),
                    Value::String(capability_gap.clone()),
                );
            }
        }
    }
    payload
}

fn safe_route_evidence_value(resolved_route: &ResolvedModelRoute) -> Value {
    let mut value = serde_json::to_value(resolved_route).unwrap_or_else(|_| json!({}));
    if let Some(endpoint) = value.get_mut("endpoint").and_then(Value::as_object_mut) {
        // Routing evidence is durable and may be exported. The endpoint is an execution
        // detail, so retain its kind and non-sensitive routing hints but never persist its URL.
        endpoint.remove("baseUrl");
    }
    if let Some(auth) = value.get_mut("auth").and_then(Value::as_object_mut) {
        auth.remove("credentialRef");
    }
    value
}

fn resolve_protocol(
    task_request: &ModelTaskRequest,
    capability_snapshot: &CapabilitySnapshot,
    selection: &ModelRouteSelection<'_>,
    provider: Option<&ModelRouteProvider<'_>>,
    direct_config: Option<&DirectRouteConfig<'_>>,
) -> ProtocolKind {
    if let Some(config) = direct_config {
        return config
            .protocol
            .clone()
            .unwrap_or_else(|| protocol_from_provider_name(config.provider_name));
    }

    if route_has_runtime_feature(capability_snapshot, "responses_api") {
        return ProtocolKind::OpenaiResponses;
    }

    if task_request.task_kind == ModelTaskKind::ImageGenerate {
        if route_has_runtime_feature(capability_snapshot, "images_api") {
            return match provider {
                Some(provider) if provider_type_is(&provider.provider_type, "fal") => {
                    ProtocolKind::Fal
                }
                None if provider_name_is_fal(selection.provider_id) => ProtocolKind::Fal,
                _ => ProtocolKind::OpenaiImages,
            };
        }
    }

    if let Some(provider) = provider {
        return protocol_from_provider_type(&provider.provider_type);
    }

    protocol_from_provider_name(selection.provider_id)
}

fn protocol_from_provider_type(provider_type: &str) -> ProtocolKind {
    match normalize_token(provider_type).as_deref() {
        Some("openai_response") | Some("openai_responses") | Some("responses") => {
            ProtocolKind::OpenaiResponses
        }
        Some("codex") => ProtocolKind::CodexResponses,
        Some("anthropic") | Some("anthropic_compatible") => ProtocolKind::AnthropicMessages,
        Some("gemini") => ProtocolKind::GeminiGenerateContent,
        Some("vertexai") | Some("vertex_ai") | Some("gcpvertexai") => ProtocolKind::VertexGemini,
        Some("aws_bedrock") | Some("bedrock") => ProtocolKind::BedrockConverse,
        Some("ollama") => ProtocolKind::OllamaChat,
        Some("fal") => ProtocolKind::Fal,
        Some("openai") | Some("azure_openai") | Some("newapi") | Some("new_api")
        | Some("gateway") => ProtocolKind::OpenaiChat,
        _ => ProtocolKind::Unknown,
    }
}

fn route_has_runtime_feature(snapshot: &CapabilitySnapshot, feature: &str) -> bool {
    normalize_string_values(&snapshot.runtime_features)
        .iter()
        .any(|value| value == feature)
}

fn normalize_string_values(values: &[String]) -> Vec<String> {
    values
        .iter()
        .filter_map(|value| normalize_token(value))
        .collect()
}

fn normalize_token(value: &str) -> Option<String> {
    let normalized = value.trim().to_ascii_lowercase().replace(['-', ' '], "_");
    if normalized.is_empty() {
        None
    } else {
        Some(normalized)
    }
}

pub fn protocol_from_provider_name(provider: &str) -> ProtocolKind {
    match normalize_token(provider).as_deref() {
        Some("openai_response") | Some("openai_responses") | Some("responses") => {
            ProtocolKind::OpenaiResponses
        }
        Some("codex") => ProtocolKind::CodexResponses,
        Some("anthropic") | Some("claude") | Some("anthropic_compatible") => {
            ProtocolKind::AnthropicMessages
        }
        Some("gemini") | Some("gemini_api_key") | Some("google") => {
            ProtocolKind::GeminiGenerateContent
        }
        Some("vertex") | Some("vertexai") | Some("vertex_ai") | Some("gcpvertexai") => {
            ProtocolKind::VertexGemini
        }
        Some("aws_bedrock") | Some("bedrock") => ProtocolKind::BedrockConverse,
        Some("ollama") => ProtocolKind::OllamaChat,
        Some("fal") => ProtocolKind::Fal,
        Some("openai") | Some("azure_openai") | Some("newapi") | Some("new_api")
        | Some("gateway") => ProtocolKind::OpenaiChat,
        _ => ProtocolKind::Unknown,
    }
}

fn provider_type_is(provider_type: &str, expected: &str) -> bool {
    normalize_token(provider_type).as_deref() == Some(expected)
}

fn provider_name_is_fal(provider: &str) -> bool {
    let normalized = provider.trim().to_ascii_lowercase();
    normalized == "fal" || normalized.contains("fal.ai") || normalized.contains("fal-ai")
}

fn endpoint_info(
    provider: Option<&ModelRouteProvider<'_>>,
    direct_config: Option<&DirectRouteConfig<'_>>,
) -> EndpointInfo {
    if let Some(config) = direct_config {
        return EndpointInfo {
            kind: EndpointKind::DirectRequest,
            base_url: config.base_url.and_then(|value| non_empty(Some(value))),
            api_version: None,
            project: None,
            location: None,
            region: None,
        };
    }

    if let Some(provider) = provider {
        let base_url = provider.base_url.and_then(|value| non_empty(Some(value)));
        return EndpointInfo {
            kind: endpoint_kind_for_provider(&provider.provider_type, &base_url),
            base_url,
            api_version: provider.api_version.map(ToString::to_string),
            project: provider.project.map(ToString::to_string),
            location: provider.location.map(ToString::to_string),
            region: provider.region.map(ToString::to_string),
        };
    }

    EndpointInfo {
        kind: EndpointKind::ProviderBaseUrl,
        base_url: None,
        api_version: None,
        project: None,
        location: None,
        region: None,
    }
}

fn endpoint_kind_for_provider(provider_type: &str, base_url: &Option<String>) -> EndpointKind {
    if base_url.as_deref().is_some_and(is_local_base_url) {
        return EndpointKind::Local;
    }
    match normalize_token(provider_type).as_deref() {
        Some("openai")
        | Some("openai_response")
        | Some("openai_responses")
        | Some("codex")
        | Some("newapi")
        | Some("new_api")
        | Some("gateway")
        | Some("fal") => EndpointKind::OpenaiCompatible,
        _ => EndpointKind::ProviderBaseUrl,
    }
}

fn auth_ref(
    selection: &ModelRouteSelection<'_>,
    provider: Option<&ModelRouteProvider<'_>>,
    direct_config: Option<&DirectRouteConfig<'_>>,
) -> AuthMaterialRef {
    if let Some(config) = direct_config {
        return AuthMaterialRef {
            kind: if config.api_key_present {
                AuthKind::DirectApiKey
            } else {
                AuthKind::NoAuth
            },
            provider_id: Some(selection.provider_id.to_string()),
            credential_ref: config
                .credential_ref
                .and_then(|value| non_empty(Some(value))),
            header_name: Some("Authorization".to_string()),
            header_prefix: Some("Bearer".to_string()),
        };
    }

    if let Some(provider) = provider {
        return AuthMaterialRef {
            kind: if provider.credential_ref.is_some() {
                AuthKind::ApiKeyRef
            } else {
                AuthKind::NoAuth
            },
            provider_id: Some(provider.provider_id.to_string()),
            credential_ref: provider.credential_ref.clone(),
            header_name: Some(provider.auth_header.to_string()),
            header_prefix: provider.auth_prefix.map(ToString::to_string),
        };
    }

    AuthMaterialRef {
        kind: AuthKind::NoAuth,
        provider_id: Some(selection.provider_id.to_string()),
        credential_ref: None,
        header_name: None,
        header_prefix: None,
    }
}

fn route_defaults(
    selection: &ModelRouteSelection<'_>,
    provider: Option<&ModelRouteProvider<'_>>,
    direct_config: Option<&DirectRouteConfig<'_>>,
) -> RouteDefaults {
    RouteDefaults {
        reasoning_effort: selection.reasoning_effort.map(ToString::to_string),
        prompt_cache_mode: provider.and_then(|provider| provider.prompt_cache_mode.clone()),
        toolshim: direct_config.map(|config| config.toolshim),
        toolshim_model: direct_config
            .and_then(|config| config.toolshim_model)
            .map(ToString::to_string),
    }
}

fn capability_snapshot(routing_payload: &Value) -> CapabilitySnapshot {
    let registry = routing_payload
        .get("modelRegistry")
        .or_else(|| routing_payload.get("model_registry"));
    let model_capabilities = registry
        .and_then(|value| value.get("modelCapabilities"))
        .or_else(|| registry.and_then(|value| value.get("model_capabilities")));
    let mut snapshot =
        capability_snapshot_from_model_capabilities(model_capabilities.unwrap_or(&Value::Null));
    snapshot.source = registry.and_then(|value| string_field(value, &["source"]));
    snapshot.reason_code =
        registry.and_then(|value| string_field(value, &["reasonCode", "reason_code"]));
    snapshot
}

fn routing_decision(routing_payload: &Value) -> RoutingDecision {
    let fallback_chain = string_array_field(routing_payload, &["fallbackChain", "fallback_chain"]);
    let routing_attempts = routing_payload
        .get("routingAttempts")
        .or_else(|| routing_payload.get("routing_attempts"))
        .and_then(Value::as_array)
        .map(|items| items.len() as u32)
        .unwrap_or(0);
    RoutingDecision {
        routing_mode: string_field(routing_payload, &["routingMode", "routing_mode"])
            .unwrap_or_else(|| "profile_slot".to_string()),
        decision_source: string_field(routing_payload, &["decisionSource", "decision_source"])
            .unwrap_or_else(|| "runtime_selection".to_string()),
        decision_reason: string_field(routing_payload, &["decisionReason", "decision_reason"])
            .unwrap_or_default(),
        settings_source: string_field(routing_payload, &["settingsSource", "settings_source"]),
        service_model_slot: string_field(
            routing_payload,
            &["serviceModelSlot", "service_model_slot"],
        ),
        fallback_chain,
        candidate_count: if routing_attempts > 0 {
            routing_attempts
        } else {
            1
        },
        capability_gap: string_field(routing_payload, &["capabilityGap", "capability_gap"]),
    }
}

fn route_failure(selection: &ModelRouteSelection<'_>, readiness: Option<&Value>) -> RouteFailure {
    let reason_code = readiness
        .and_then(|value| string_field(value, &["reasonCode", "reason_code"]))
        .unwrap_or_else(|| "routing_not_ready".to_string());
    RouteFailure {
        category: route_failure_category(&reason_code),
        reason_code: reason_code.clone(),
        message: Some(reason_code),
        provider_id: Some(selection.provider_id.to_string()),
        model_id: Some(selection.model_id.to_string()),
        capability_gap: None,
        retryable: false,
    }
}

fn capability_route_failure(
    selection: &ModelRouteSelection<'_>,
    capability_gap: String,
) -> RouteFailure {
    RouteFailure {
        category: RouteFailureCategory::CapabilityGap,
        reason_code: "capability_gap".to_string(),
        message: Some(format!("model capability gap: {capability_gap}")),
        provider_id: Some(selection.provider_id.to_string()),
        model_id: Some(selection.model_id.to_string()),
        capability_gap: Some(capability_gap),
        retryable: false,
    }
}

fn unsupported_protocol_route_failure(
    selection: &ModelRouteSelection<'_>,
    protocol: &ProtocolKind,
) -> RouteFailure {
    let protocol_name = format!("{protocol:?}");
    RouteFailure {
        category: RouteFailureCategory::UnsupportedProtocol,
        reason_code: "unsupported_protocol".to_string(),
        message: Some(format!(
            "provider route resolved to unsupported protocol: {protocol_name}"
        )),
        provider_id: Some(selection.provider_id.to_string()),
        model_id: Some(selection.model_id.to_string()),
        capability_gap: None,
        retryable: false,
    }
}

fn protocol_supported_for_task(task_kind: ModelTaskKind, protocol: &ProtocolKind) -> bool {
    match task_kind {
        // Current agent turns are backed by model-provider's three wire adapters. Other
        // ProtocolKind variants must not be silently lowered to Custom/ChatCompletions.
        ModelTaskKind::Chat => matches!(
            protocol,
            ProtocolKind::OpenaiChat
                | ProtocolKind::OpenaiResponses
                | ProtocolKind::AnthropicMessages
                | ProtocolKind::CodexResponses
        ),
        // Media tasks have dedicated lowerings and are validated by their own execution owner.
        _ => true,
    }
}

fn unknown_model_or_capability_failure(
    selection: &ModelRouteSelection<'_>,
    routing_payload: &Value,
) -> RouteFailure {
    let registry = model_registry(routing_payload);
    let registry_source = registry.and_then(|value| string_field(value, &["source"]));
    let registry_status = registry.and_then(|value| string_field(value, &["status"]));
    let registry_reason =
        registry.and_then(|value| string_field(value, &["reasonCode", "reason_code"]));
    let model_is_known = registry_source.as_deref() == Some("direct_provider_config")
        || registry_status.as_deref() == Some("matched")
        || registry_reason
            .as_deref()
            .is_some_and(|reason| reason.starts_with("matched_"));

    if model_is_known {
        let capability_gap = "capability_snapshot:missing".to_string();
        return RouteFailure {
            category: RouteFailureCategory::CapabilityGap,
            reason_code: "capability_snapshot_missing".to_string(),
            message: Some("model capability snapshot is required for routing".to_string()),
            provider_id: Some(selection.provider_id.to_string()),
            model_id: Some(selection.model_id.to_string()),
            capability_gap: Some(capability_gap),
            retryable: false,
        };
    }

    let reason_code =
        registry_reason.unwrap_or_else(|| "model_registry_metadata_missing".to_string());
    RouteFailure {
        category: RouteFailureCategory::ModelUnavailable,
        message: Some(format!("model is unavailable: {reason_code}")),
        reason_code,
        provider_id: Some(selection.provider_id.to_string()),
        model_id: Some(selection.model_id.to_string()),
        capability_gap: None,
        retryable: false,
    }
}

fn route_failure_category(reason_code: &str) -> RouteFailureCategory {
    match reason_code {
        "provider_disabled" => RouteFailureCategory::ProviderDisabled,
        "missing_enabled_api_key" => RouteFailureCategory::MissingCredential,
        "provider_not_chat_capable" => RouteFailureCategory::CapabilityGap,
        "provider_not_configured" => RouteFailureCategory::ProviderNeedsSetup,
        _ => RouteFailureCategory::ProviderNeedsSetup,
    }
}

fn has_declared_capability_snapshot(routing_payload: &Value) -> bool {
    model_registry(routing_payload)
        .and_then(|registry| {
            registry
                .get("modelCapabilities")
                .or_else(|| registry.get("model_capabilities"))
        })
        .is_some_and(Value::is_object)
}

fn model_registry(routing_payload: &Value) -> Option<&Value> {
    routing_payload
        .get("modelRegistry")
        .or_else(|| routing_payload.get("model_registry"))
}

fn framing_for_protocol(protocol: &ProtocolKind) -> FramingKind {
    match protocol {
        ProtocolKind::OllamaChat => FramingKind::Ndjson,
        ProtocolKind::Unknown => FramingKind::Json,
        _ => FramingKind::Sse,
    }
}

fn is_local_base_url(value: &str) -> bool {
    let value = value.trim().to_ascii_lowercase();
    value.starts_with("http://127.0.0.1")
        || value.starts_with("http://localhost")
        || value.starts_with("http://[::1]")
}

fn non_empty(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
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
    use crate::model_task::{build_model_task_request, ModelTaskRequestInput};
    use app_server_protocol::{ModelTaskKind, ModelTaskSource};
    use serde_json::json;

    fn chat_task_request(provider_id: &str, model_id: &str) -> ModelTaskRequest {
        build_model_task_request(ModelTaskRequestInput {
            task_kind: ModelTaskKind::Chat,
            source: ModelTaskSource::AgentTurn,
            provider_id: Some(provider_id.to_string()),
            model_id: Some(model_id.to_string()),
            model_ref_source: ModelRefSource::RuntimeRequest,
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
        })
    }

    #[test]
    fn direct_route_without_capability_snapshot_fails_closed() {
        let task_request = chat_task_request("fixture-openai", "fixture-model");
        let route = resolved_route_from_task(
            &task_request,
            ModelRouteSelection {
                provider_id: "fixture-openai",
                model_id: "fixture-model",
                model_ref_source: ModelRefSource::RuntimeRequest,
                reasoning_effort: None,
            },
            &json!({
                "providerReadiness": {
                    "ready": true,
                    "status": "ready"
                },
                "modelRegistry": {
                    "source": "direct_provider_config",
                    "status": "runtime_selection_only",
                    "reasonCode": "direct_provider_config_not_in_registry",
                    "modelCapabilities": null
                }
            }),
            None,
            Some(DirectRouteConfig {
                provider_name: "openai",
                api_key_present: true,
                base_url: Some("http://127.0.0.1:56599"),
                credential_ref: None,
                protocol: Some(ProtocolKind::OpenaiChat),
                toolshim: false,
                toolshim_model: None,
            }),
        );

        let failure = route.failure.expect("missing capability snapshot failure");
        assert_eq!(failure.category, RouteFailureCategory::CapabilityGap);
        assert_eq!(failure.reason_code, "capability_snapshot_missing");
        assert_eq!(
            failure.capability_gap.as_deref(),
            Some("capability_snapshot:missing")
        );
    }

    #[test]
    fn unknown_registry_model_fails_closed() {
        let task_request = chat_task_request("openai", "unknown-model");
        let route = resolved_route_from_task(
            &task_request,
            ModelRouteSelection {
                provider_id: "openai",
                model_id: "unknown-model",
                model_ref_source: ModelRefSource::RuntimeOptions,
                reasoning_effort: None,
            },
            &json!({
                "providerReadiness": {
                    "ready": true,
                    "status": "ready"
                },
                "modelRegistry": {
                    "source": "runtime_selection_only",
                    "status": "missing",
                    "reasonCode": "model_registry_metadata_missing",
                    "modelCapabilities": null
                }
            }),
            None,
            None,
        );

        let failure = route.failure.expect("unknown model failure");
        assert_eq!(failure.category, RouteFailureCategory::ModelUnavailable);
        assert_eq!(failure.reason_code, "model_registry_metadata_missing");
        assert!(failure.capability_gap.is_none());
    }

    #[test]
    fn unknown_provider_protocol_fails_closed_instead_of_defaulting_to_chat() {
        let task_request = chat_task_request("unregistered-gateway", "gateway-model");
        let route = resolved_route_from_task(
            &task_request,
            ModelRouteSelection {
                provider_id: "unregistered-gateway",
                model_id: "gateway-model",
                model_ref_source: ModelRefSource::RuntimeRequest,
                reasoning_effort: None,
            },
            &json!({
                "providerReadiness": {
                    "ready": true,
                    "status": "ready"
                },
                "modelRegistry": {
                    "source": "direct_provider_config",
                    "status": "matched",
                    "reasonCode": "matched_direct_provider_config",
                    "modelCapabilities": {
                        "taskFamilies": ["chat"],
                        "inputModalities": ["text"],
                        "outputModalities": ["text"],
                        "runtimeFeatures": ["streaming"],
                        "capabilities": {
                            "streaming": true
                        }
                    }
                }
            }),
            None,
            None,
        );

        assert_eq!(route.protocol, ProtocolKind::Unknown);
        let failure = route.failure.expect("unsupported protocol failure");
        assert_eq!(failure.category, RouteFailureCategory::UnsupportedProtocol);
        assert_eq!(failure.reason_code, "unsupported_protocol");
        assert_eq!(failure.provider_id.as_deref(), Some("unregistered-gateway"));
        assert_eq!(failure.model_id.as_deref(), Some("gateway-model"));
    }

    #[test]
    fn known_non_chat_wire_protocol_fails_closed_for_agent_turns() {
        let task_request = chat_task_request("gemini", "gemini-pro");
        let route = resolved_route_from_task(
            &task_request,
            ModelRouteSelection {
                provider_id: "gemini",
                model_id: "gemini-pro",
                model_ref_source: ModelRefSource::RuntimeRequest,
                reasoning_effort: None,
            },
            &json!({
                "providerReadiness": {
                    "ready": true,
                    "status": "ready"
                },
                "modelRegistry": {
                    "source": "api",
                    "status": "matched",
                    "reasonCode": "matched_model",
                    "modelCapabilities": {
                        "taskFamilies": ["chat"],
                        "inputModalities": ["text"],
                        "outputModalities": ["text"],
                        "runtimeFeatures": ["streaming"],
                        "capabilities": {
                            "streaming": true
                        }
                    }
                }
            }),
            Some(&ModelRouteProvider {
                provider_id: "gemini",
                provider_type: Cow::Borrowed("gemini"),
                base_url: Some("https://generativelanguage.googleapis.com"),
                api_version: None,
                project: None,
                location: None,
                region: None,
                credential_ref: Some("credential-gemini".to_string()),
                auth_header: "x-goog-api-key",
                auth_prefix: None,
                prompt_cache_mode: None,
            }),
            None,
        );

        assert_eq!(route.protocol, ProtocolKind::GeminiGenerateContent);
        let failure = route.failure.expect("unsupported chat protocol failure");
        assert_eq!(failure.category, RouteFailureCategory::UnsupportedProtocol);
        assert_eq!(failure.reason_code, "unsupported_protocol");
    }

    #[test]
    fn image_task_route_preserves_task_model_ref_and_reports_capability_gap() {
        let task_request = build_model_task_request(ModelTaskRequestInput {
            task_kind: ModelTaskKind::ImageGenerate,
            source: ModelTaskSource::MediaTaskArtifact,
            provider_id: Some("openai".to_string()),
            model_id: Some("text-only".to_string()),
            model_ref_source: ModelRefSource::Task,
            modality_contract_key: Some("image_generation".to_string()),
            routing_slot: Some("image_generation_model".to_string()),
            task_families: vec!["image_generation".to_string()],
            input_modalities: vec!["text".to_string()],
            output_modalities: vec!["image".to_string()],
            runtime_features: Vec::new(),
            capabilities: vec!["image_generation".to_string()],
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
            "routingMode": "task_route",
            "decisionSource": "media_task_artifact",
            "decisionReason": "explicit_task_model",
            "serviceModelSlot": "image_generation_model",
            "modelRegistry": {
                "source": "provider_declared_model",
                "reasonCode": "matched_provider_custom_models",
                "modelCapabilities": {
                    "capabilities": {
                        "vision": false,
                        "streaming": true
                    },
                    "taskFamilies": ["chat"],
                    "inputModalities": ["text"],
                    "outputModalities": ["text"],
                    "runtimeFeatures": ["streaming"]
                }
            }
        });

        let route = resolved_route_from_task(
            &task_request,
            ModelRouteSelection {
                provider_id: "openai",
                model_id: "text-only",
                model_ref_source: ModelRefSource::Task,
                reasoning_effort: None,
            },
            &routing_payload,
            None,
            None,
        );

        assert_eq!(route.model_ref.source, ModelRefSource::Task);
        assert_eq!(
            route.model_ref.routing_slot.as_deref(),
            Some("image_generation_model")
        );
        assert_eq!(route.decision.routing_mode, "task_route");
        assert_eq!(
            route.decision.capability_gap.as_deref(),
            Some("task_family:image_generation")
        );
        let failure = route.failure.expect("capability gap");
        assert_eq!(failure.category, RouteFailureCategory::CapabilityGap);
        assert_eq!(
            failure.capability_gap.as_deref(),
            Some("task_family:image_generation")
        );
    }

    #[test]
    fn resolved_route_does_not_project_picker_dtos_as_execution_facts() {
        let task_request = build_model_task_request(ModelTaskRequestInput {
            task_kind: ModelTaskKind::Chat,
            source: ModelTaskSource::AgentTurn,
            provider_id: Some("openai".to_string()),
            model_id: Some("gpt-4.1".to_string()),
            model_ref_source: ModelRefSource::Explicit,
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
        let provider = ModelRouteProvider {
            provider_id: "openai",
            provider_type: Cow::Borrowed("openai"),
            base_url: Some("https://api.openai.com/v1"),
            api_version: None,
            project: None,
            location: None,
            region: None,
            credential_ref: Some("runtime-api-key-openai".to_string()),
            auth_header: "Authorization",
            auth_prefix: Some("Bearer"),
            prompt_cache_mode: None,
        };

        let route = resolved_route_from_task(
            &task_request,
            ModelRouteSelection {
                provider_id: "openai",
                model_id: "gpt-4.1",
                model_ref_source: ModelRefSource::Explicit,
                reasoning_effort: None,
            },
            &json!({
                "providerReadiness": {
                    "ready": true,
                    "status": "ready"
                },
                "routingMode": "profile_slot",
                "decisionSource": "runtime_selection",
                "decisionReason": "explicit_model",
                "modelRegistry": {
                    "displayName": "GPT 4.1",
                    "providerName": "OpenAI",
                    "tier": "pro",
                    "status": "active",
                    "pricing": {
                        "inputPerMillion": 2.0,
                        "outputPerMillion": 8.0
                    },
                    "modelCapabilities": {
                        "taskFamilies": ["chat"],
                        "inputModalities": ["text"],
                        "outputModalities": ["text"],
                        "runtimeFeatures": ["streaming"],
                        "capabilities": {
                            "streaming": true
                        }
                    }
                }
            }),
            Some(&provider),
            None,
        );

        assert_eq!(route.model_ref.model_id, "gpt-4.1");
        assert_eq!(
            route.endpoint.base_url.as_deref(),
            Some("https://api.openai.com/v1")
        );
        assert_eq!(
            route.auth.credential_ref.as_deref(),
            Some("runtime-api-key-openai")
        );
        assert_eq!(
            route.capability_snapshot.task_families,
            vec!["chat".to_string()]
        );
    }

    #[test]
    fn image_task_route_uses_openai_images_protocol_for_images_api_feature() {
        let task_request = build_model_task_request(ModelTaskRequestInput {
            task_kind: ModelTaskKind::ImageGenerate,
            source: ModelTaskSource::MediaTaskArtifact,
            provider_id: Some("openai".to_string()),
            model_id: Some("gpt-image-2".to_string()),
            model_ref_source: ModelRefSource::Task,
            modality_contract_key: Some("image_generation".to_string()),
            routing_slot: Some("image_generation_model".to_string()),
            task_families: vec!["image_generation".to_string()],
            input_modalities: vec!["text".to_string()],
            output_modalities: vec!["image".to_string()],
            runtime_features: Vec::new(),
            capabilities: vec!["image_generation".to_string()],
            session_id: None,
            thread_id: None,
            turn_id: None,
            content_id: None,
            trace_id: None,
        });
        let route = resolved_route_from_task(
            &task_request,
            ModelRouteSelection {
                provider_id: "openai",
                model_id: "gpt-image-2",
                model_ref_source: ModelRefSource::Task,
                reasoning_effort: None,
            },
            &json!({
                "providerReadiness": {
                    "ready": true,
                    "status": "ready"
                },
                "routingMode": "task_route",
                "decisionSource": "media_task_artifact",
                "decisionReason": "explicit_task_model",
                "serviceModelSlot": "image_generation_model",
                "modelRegistry": {
                    "source": "api",
                    "reasonCode": "matched_media_task_model",
                    "modelCapabilities": {
                        "taskFamilies": ["image_generation"],
                        "inputModalities": ["text"],
                        "outputModalities": ["image"],
                        "runtimeFeatures": ["images_api"],
                        "capabilities": {
                            "vision": false
                        }
                    }
                }
            }),
            None,
            None,
        );

        assert_eq!(route.protocol, ProtocolKind::OpenaiImages);
        assert!(route.failure.is_none());
    }

    #[test]
    fn image_task_route_uses_openai_responses_protocol_for_responses_feature() {
        let task_request = build_model_task_request(ModelTaskRequestInput {
            task_kind: ModelTaskKind::ImageGenerate,
            source: ModelTaskSource::MediaTaskArtifact,
            provider_id: Some("openai-responses".to_string()),
            model_id: Some("gpt-images-2".to_string()),
            model_ref_source: ModelRefSource::Task,
            modality_contract_key: Some("image_generation".to_string()),
            routing_slot: Some("image_generation_model".to_string()),
            task_families: vec!["image_generation".to_string()],
            input_modalities: vec!["text".to_string()],
            output_modalities: vec!["image".to_string()],
            runtime_features: Vec::new(),
            capabilities: vec!["image_generation".to_string()],
            session_id: None,
            thread_id: None,
            turn_id: None,
            content_id: None,
            trace_id: None,
        });
        let route = resolved_route_from_task(
            &task_request,
            ModelRouteSelection {
                provider_id: "openai-responses",
                model_id: "gpt-images-2",
                model_ref_source: ModelRefSource::Task,
                reasoning_effort: None,
            },
            &json!({
                "providerReadiness": {
                    "ready": true,
                    "status": "ready"
                },
                "routingMode": "task_route",
                "decisionSource": "media_task_artifact",
                "decisionReason": "explicit_task_model",
                "serviceModelSlot": "image_generation_model",
                "modelRegistry": {
                    "source": "api",
                    "reasonCode": "matched_media_task_model",
                    "modelCapabilities": {
                        "taskFamilies": ["image_generation"],
                        "inputModalities": ["text"],
                        "outputModalities": ["image"],
                        "runtimeFeatures": ["responses_api"],
                        "capabilities": {
                            "vision": false
                        }
                    }
                }
            }),
            None,
            None,
        );

        assert_eq!(route.protocol, ProtocolKind::OpenaiResponses);
        assert!(route.failure.is_none());
    }

    #[test]
    fn image_task_route_uses_gemini_protocol_for_gemini_provider() {
        let task_request = build_model_task_request(ModelTaskRequestInput {
            task_kind: ModelTaskKind::ImageGenerate,
            source: ModelTaskSource::MediaTaskArtifact,
            provider_id: Some("google".to_string()),
            model_id: Some("gemini-2.5-flash-image".to_string()),
            model_ref_source: ModelRefSource::Task,
            modality_contract_key: Some("image_generation".to_string()),
            routing_slot: Some("image_generation_model".to_string()),
            task_families: vec!["image_generation".to_string()],
            input_modalities: vec!["text".to_string()],
            output_modalities: vec!["image".to_string()],
            runtime_features: Vec::new(),
            capabilities: vec!["image_generation".to_string()],
            session_id: None,
            thread_id: None,
            turn_id: None,
            content_id: None,
            trace_id: None,
        });
        let provider = ModelRouteProvider {
            provider_id: "google",
            provider_type: Cow::Borrowed("gemini"),
            base_url: Some("https://generativelanguage.googleapis.com"),
            api_version: None,
            project: None,
            location: None,
            region: None,
            credential_ref: Some("runtime-api-key-google".to_string()),
            auth_header: "x-goog-api-key",
            auth_prefix: None,
            prompt_cache_mode: None,
        };
        let route = resolved_route_from_task(
            &task_request,
            ModelRouteSelection {
                provider_id: "google",
                model_id: "gemini-2.5-flash-image",
                model_ref_source: ModelRefSource::Task,
                reasoning_effort: None,
            },
            &json!({
                "providerReadiness": {
                    "ready": true,
                    "status": "ready"
                },
                "routingMode": "task_route",
                "decisionSource": "media_task_artifact",
                "decisionReason": "explicit_task_model",
                "serviceModelSlot": "image_generation_model",
                "modelRegistry": {
                    "source": "api",
                    "reasonCode": "matched_media_task_model",
                    "modelCapabilities": {
                        "taskFamilies": ["image_generation"],
                        "inputModalities": ["text"],
                        "outputModalities": ["image"],
                        "runtimeFeatures": [],
                        "capabilities": {
                            "vision": false
                        }
                    }
                }
            }),
            Some(&provider),
            None,
        );

        assert_eq!(route.protocol, ProtocolKind::GeminiGenerateContent);
        assert_eq!(route.auth.header_name.as_deref(), Some("x-goog-api-key"));
        assert!(route.failure.is_none());
    }

    #[test]
    fn chat_task_route_uses_responses_protocol_when_runtime_feature_declares_it() {
        let task_request = build_model_task_request(ModelTaskRequestInput {
            task_kind: ModelTaskKind::Chat,
            source: ModelTaskSource::AgentTurn,
            provider_id: Some("openai".to_string()),
            model_id: Some("gpt-4.1".to_string()),
            model_ref_source: ModelRefSource::Task,
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
        let route = resolved_route_from_task(
            &task_request,
            ModelRouteSelection {
                provider_id: "openai",
                model_id: "gpt-4.1",
                model_ref_source: ModelRefSource::Task,
                reasoning_effort: None,
            },
            &json!({
                "providerReadiness": {
                    "ready": true,
                    "status": "ready"
                },
                "routingMode": "task_route",
                "decisionSource": "runtime_selection",
                "decisionReason": "explicit_task_model",
                "serviceModelSlot": "coding",
                "modelRegistry": {
                    "source": "api",
                    "reasonCode": "matched_model",
                    "modelCapabilities": {
                        "taskFamilies": ["chat"],
                        "inputModalities": ["text"],
                        "outputModalities": ["text"],
                        "runtimeFeatures": ["responses_api", "streaming"],
                        "capabilities": {
                            "vision": false,
                            "streaming": true
                        }
                    }
                }
            }),
            None,
            None,
        );

        assert_eq!(route.protocol, ProtocolKind::OpenaiResponses);
        assert!(route.failure.is_none());
    }

    #[test]
    fn route_evidence_payload_preserves_task_and_route_failures() {
        let task_request = build_model_task_request(ModelTaskRequestInput {
            task_kind: ModelTaskKind::Chat,
            source: ModelTaskSource::AgentTurn,
            provider_id: Some("openai".to_string()),
            model_id: Some("gpt-4.1".to_string()),
            model_ref_source: ModelRefSource::Explicit,
            modality_contract_key: Some("chat".to_string()),
            routing_slot: Some("coding".to_string()),
            task_families: vec!["chat".to_string()],
            input_modalities: vec!["text".to_string()],
            output_modalities: vec!["text".to_string()],
            runtime_features: vec!["streaming".to_string()],
            capabilities: vec!["streaming".to_string()],
            session_id: Some("session-1".to_string()),
            thread_id: Some("thread-1".to_string()),
            turn_id: Some("turn-1".to_string()),
            content_id: None,
            trace_id: Some("trace-1".to_string()),
        });
        let route = ResolvedModelRoute {
            model_ref: task_request.model_ref.clone().expect("model ref"),
            protocol: ProtocolKind::OpenaiChat,
            endpoint: EndpointInfo {
                kind: EndpointKind::ProviderBaseUrl,
                base_url: Some("https://example.com".to_string()),
                api_version: None,
                project: None,
                location: None,
                region: None,
            },
            auth: AuthMaterialRef {
                kind: app_server_protocol::AuthKind::ApiKeyRef,
                provider_id: Some("openai".to_string()),
                credential_ref: Some("credential-secret-ref".to_string()),
                header_name: Some("Authorization".to_string()),
                header_prefix: Some("Bearer".to_string()),
            },
            transport: TransportKind::Http,
            framing: FramingKind::Sse,
            defaults: RouteDefaults::default(),
            capability_snapshot: CapabilitySnapshot {
                task_families: vec!["chat".to_string()],
                input_modalities: vec!["text".to_string()],
                output_modalities: vec!["text".to_string()],
                runtime_features: vec!["streaming".to_string()],
                capabilities: Default::default(),
                source: Some("api".to_string()),
                reason_code: Some("declared".to_string()),
            },
            decision: RoutingDecision {
                routing_mode: "profile_slot".to_string(),
                decision_source: "runtime_selection".to_string(),
                decision_reason: "explicit".to_string(),
                settings_source: Some("runtime_options".to_string()),
                service_model_slot: Some("coding".to_string()),
                fallback_chain: vec!["coding".to_string()],
                candidate_count: 1,
                capability_gap: Some("task_family:chat".to_string()),
            },
            failure: Some(RouteFailure {
                category: RouteFailureCategory::CapabilityGap,
                reason_code: "capability_gap".to_string(),
                message: Some("model capability gap: task_family:chat".to_string()),
                provider_id: Some("openai".to_string()),
                model_id: Some("gpt-4.1".to_string()),
                capability_gap: Some("task_family:chat".to_string()),
                retryable: false,
            }),
        };

        let payload = route_evidence_payload(json!({"backend":"runtime"}), &task_request, &route);

        assert_eq!(
            payload
                .pointer("/modelTaskRequest/modelRef/providerId")
                .and_then(Value::as_str),
            Some("openai")
        );
        assert_eq!(
            payload
                .pointer("/resolvedRoute/protocol")
                .and_then(Value::as_str),
            Some("openai_chat")
        );
        assert!(payload.pointer("/resolvedRoute/endpoint/baseUrl").is_none());
        assert!(payload
            .pointer("/resolvedRoute/auth/credentialRef")
            .is_none());
        assert!(!payload.to_string().contains("https://example.com"));
        assert!(!payload.to_string().contains("credential-secret-ref"));
        assert_eq!(
            payload
                .pointer("/routeFailure/reasonCode")
                .and_then(Value::as_str),
            Some("capability_gap")
        );
        assert_eq!(
            payload.pointer("/capabilityGap").and_then(Value::as_str),
            Some("task_family:chat")
        );
    }

    #[test]
    fn route_resolution_evidence_payloads_builds_fallback_and_failure_payloads() {
        let requested_selection = RuntimeModelSelection {
            provider: "workspace-coding".to_string(),
            model: "coder-large".to_string(),
            source: crate::model_routing::PROFILE_MODEL_SLOT_SOURCE,
            reasoning_effort: None,
        };
        let routing_resolution = RoutingResolution {
            selection: RuntimeModelSelection {
                provider: "openai".to_string(),
                model: "gpt-4.1-mini".to_string(),
                source: crate::model_routing::PROFILE_MODEL_SLOT_SOURCE,
                reasoning_effort: None,
            },
            routing: crate::model_routing::ModelRoutingDecision {
                service_model_slot: "base".to_string(),
                requested_provider: Some("workspace-coding".to_string()),
                requested_model: Some("coder-large".to_string()),
                settings_source: "profile_model_slot".to_string(),
                decision_reason: "profile_slot_selected".to_string(),
                fallback_chain: vec![
                    "workspace-coding/coder-large".to_string(),
                    "openai/gpt-4.1-mini".to_string(),
                ],
                profile_slots: Vec::new(),
            },
            readiness: crate::model_routing::ProviderReadiness::provider_store_ready(
                Some("openai".to_string()),
                1,
                1,
            ),
            attempted: vec![
                crate::model_routing::RoutingAttempt {
                    slot: "coding".to_string(),
                    provider: "workspace-coding".to_string(),
                    model: "coder-large".to_string(),
                    source: crate::model_routing::PROFILE_MODEL_SLOT_SOURCE.to_string(),
                    readiness: crate::model_routing::ProviderReadiness::provider_store_needs_setup(
                        "missing_enabled_api_key",
                        Some("openai".to_string()),
                        Some(true),
                        0,
                        1,
                    ),
                    runtime_failure: None,
                },
                crate::model_routing::RoutingAttempt {
                    slot: "base".to_string(),
                    provider: "openai".to_string(),
                    model: "gpt-4.1-mini".to_string(),
                    source: crate::model_routing::PROFILE_MODEL_SLOT_SOURCE.to_string(),
                    readiness: crate::model_routing::ProviderReadiness::provider_store_ready(
                        Some("openai".to_string()),
                        1,
                        1,
                    ),
                    runtime_failure: None,
                },
            ],
        };
        let task_request = build_model_task_request(ModelTaskRequestInput {
            task_kind: ModelTaskKind::Chat,
            source: ModelTaskSource::AgentTurn,
            provider_id: Some("openai".to_string()),
            model_id: Some("gpt-4.1-mini".to_string()),
            model_ref_source: ModelRefSource::ProfileSlot,
            modality_contract_key: Some("chat".to_string()),
            routing_slot: Some("base".to_string()),
            task_families: vec!["chat".to_string()],
            input_modalities: vec!["text".to_string()],
            output_modalities: vec!["text".to_string()],
            runtime_features: vec!["streaming".to_string()],
            capabilities: vec!["streaming".to_string()],
            session_id: Some("session-1".to_string()),
            thread_id: Some("thread-1".to_string()),
            turn_id: Some("turn-1".to_string()),
            content_id: None,
            trace_id: None,
        });
        let route = resolved_route_from_task(
            &task_request,
            ModelRouteSelection {
                provider_id: "openai",
                model_id: "gpt-4.1-mini",
                model_ref_source: ModelRefSource::ProfileSlot,
                reasoning_effort: None,
            },
            &routing_decision_payload(
                &routing_resolution.selection,
                &routing_resolution.routing,
                &routing_resolution.readiness,
                &json!({
                    "source": "api",
                    "reasonCode": "matched_model",
                    "modelCapabilities": {
                        "taskFamilies": ["chat"],
                        "inputModalities": ["text"],
                        "outputModalities": ["text"],
                        "runtimeFeatures": ["streaming"],
                        "capabilities": {
                            "streaming": true
                        }
                    }
                }),
            ),
            None,
            None,
        );

        let payloads = route_resolution_evidence_payloads(
            &requested_selection,
            &routing_resolution,
            &json!({
                "source": "api",
                "reasonCode": "matched_model",
                "modelCapabilities": {
                    "taskFamilies": ["chat"],
                    "inputModalities": ["text"],
                    "outputModalities": ["text"],
                    "runtimeFeatures": ["streaming"],
                    "capabilities": {
                        "streaming": true
                    }
                }
            }),
            &task_request,
            &route,
        );

        assert_eq!(
            payloads
                .decision_payload
                .pointer("/resolvedRoute/protocol")
                .and_then(Value::as_str),
            Some("openai_chat")
        );
        let fallback = payloads.fallback_payload.expect("fallback payload");
        assert_eq!(
            fallback
                .pointer("/fallbackApplied")
                .and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(
            fallback
                .pointer("/routingAttempts")
                .and_then(Value::as_array)
                .map(Vec::len),
            Some(2)
        );
        assert!(payloads.not_possible_payload.is_none());
    }
}
