use super::{
    ModelRoutingDecision, ProviderReadiness, RoutingAttempt, RuntimeModelSelection,
    REQUIRED_CODING_CAPABILITIES,
};
use serde_json::{json, Value};

pub fn routing_decision_payload(
    selection: &RuntimeModelSelection,
    routing: &ModelRoutingDecision,
    readiness: &ProviderReadiness,
    model_registry_payload: &Value,
) -> Value {
    let selected_provider = selection.provider.clone();
    let selected_model = selection.model.clone();
    let requested_provider = routing
        .requested_provider
        .clone()
        .unwrap_or_else(|| selected_provider.clone());
    let requested_model = routing
        .requested_model
        .clone()
        .unwrap_or_else(|| selected_model.clone());
    let routing_decision = json!({
        "routingMode": "profile_slot",
        "routing_mode": "profile_slot",
        "decisionSource": selection.source,
        "decision_source": selection.source,
        "decisionReason": routing.decision_reason,
        "decision_reason": routing.decision_reason,
        "settingsSource": routing.settings_source,
        "settings_source": routing.settings_source,
        "serviceModelSlot": routing.service_model_slot,
        "service_model_slot": routing.service_model_slot,
        "selectedProvider": selected_provider,
        "selected_provider": selected_provider,
        "selectedModel": selected_model,
        "selected_model": selected_model,
        "requestedProvider": requested_provider,
        "requested_provider": requested_provider,
        "requestedModel": requested_model,
        "requested_model": requested_model,
        "fallbackChain": routing.fallback_chain,
        "fallback_chain": routing.fallback_chain,
        "requiredCapabilities": REQUIRED_CODING_CAPABILITIES,
        "required_capabilities": REQUIRED_CODING_CAPABILITIES,
        "modelRegistry": model_registry_payload,
        "model_registry": model_registry_payload,
    });
    let model_slot = model_slot_payload(routing, selection);

    json!({
        "backend": "runtime",
        "routingDecision": routing_decision,
        "routing_decision": routing_decision,
        "modelSlot": model_slot,
        "model_slot": model_slot,
        "providerReadiness": readiness.to_payload(),
        "provider_readiness": readiness.to_payload(),
        "modelRegistry": model_registry_payload,
        "model_registry": model_registry_payload,
        "provider": selected_provider,
        "model": selected_model,
        "source": selection.source,
        "decisionSource": selection.source,
        "decision_source": selection.source,
        "decisionReason": routing.decision_reason,
        "decision_reason": routing.decision_reason,
        "settingsSource": routing.settings_source,
        "settings_source": routing.settings_source,
        "serviceModelSlot": routing.service_model_slot,
        "service_model_slot": routing.service_model_slot,
        "selectedProvider": selected_provider,
        "selected_provider": selected_provider,
        "selectedModel": selected_model,
        "selected_model": selected_model,
        "requestedProvider": requested_provider,
        "requested_provider": requested_provider,
        "requestedModel": requested_model,
        "requested_model": requested_model,
        "fallbackChain": routing.fallback_chain,
        "fallback_chain": routing.fallback_chain,
        "requiredCapabilities": REQUIRED_CODING_CAPABILITIES,
        "required_capabilities": REQUIRED_CODING_CAPABILITIES,
    })
}

pub fn routing_fallback_applied_payload(
    requested_selection: &RuntimeModelSelection,
    selection: &RuntimeModelSelection,
    routing: &ModelRoutingDecision,
    readiness: &ProviderReadiness,
    model_registry_payload: &Value,
    attempted: &[RoutingAttempt],
) -> Value {
    let mut payload =
        routing_decision_payload(selection, routing, readiness, model_registry_payload);
    if let Some(object) = payload.as_object_mut() {
        object.insert("status".to_string(), Value::String("ready".to_string()));
        object.insert(
            "fallbackApplied".to_string(),
            Value::Bool(requested_selection != selection),
        );
        object.insert(
            "fallback_applied".to_string(),
            Value::Bool(requested_selection != selection),
        );
        object.insert(
            "requestedSelection".to_string(),
            selection_payload(requested_selection),
        );
        object.insert(
            "requested_selection".to_string(),
            selection_payload(requested_selection),
        );
        object.insert(
            "routingAttempts".to_string(),
            routing_attempts_payload(attempted),
        );
        object.insert(
            "routing_attempts".to_string(),
            routing_attempts_payload(attempted),
        );
        if let Some(runtime_failure) = attempted
            .iter()
            .find_map(|attempt| attempt.runtime_failure.as_ref())
        {
            object.insert(
                "fallbackReason".to_string(),
                Value::String("runtime_provider_failure".to_string()),
            );
            object.insert(
                "fallback_reason".to_string(),
                Value::String("runtime_provider_failure".to_string()),
            );
            let runtime_failure = serde_json::json!({
                "provider": runtime_failure.provider,
                "model": runtime_failure.model,
                "reasonCode": runtime_failure.reason_code,
                "reason_code": runtime_failure.reason_code,
                "classification": runtime_failure.classification,
                "retryable": runtime_failure.retryable,
            });
            object.insert("runtimeFailure".to_string(), runtime_failure.clone());
            object.insert("runtime_failure".to_string(), runtime_failure);
        }
    }
    payload
}

pub fn routing_not_possible_payload(
    selection: &RuntimeModelSelection,
    routing: &ModelRoutingDecision,
    readiness: &ProviderReadiness,
    model_registry_payload: &Value,
) -> Value {
    let mut payload =
        routing_decision_payload(selection, routing, readiness, model_registry_payload);
    if let Some(object) = payload.as_object_mut() {
        object.insert("status".to_string(), Value::String("blocked".to_string()));
        object.insert(
            "failureCategory".to_string(),
            Value::String("provider_needs_setup".to_string()),
        );
        object.insert(
            "failure_category".to_string(),
            Value::String("provider_needs_setup".to_string()),
        );
        if let Some(reason_code) = readiness.reason_code {
            object.insert(
                "reasonCode".to_string(),
                Value::String(reason_code.to_string()),
            );
            object.insert(
                "reason_code".to_string(),
                Value::String(reason_code.to_string()),
            );
        }
    }
    payload
}

pub fn routing_not_possible_payload_with_attempts(
    selection: &RuntimeModelSelection,
    routing: &ModelRoutingDecision,
    readiness: &ProviderReadiness,
    model_registry_payload: &Value,
    attempted: &[RoutingAttempt],
) -> Value {
    let mut payload =
        routing_not_possible_payload(selection, routing, readiness, model_registry_payload);
    if let Some(object) = payload.as_object_mut() {
        object.insert(
            "routingAttempts".to_string(),
            routing_attempts_payload(attempted),
        );
        object.insert(
            "routing_attempts".to_string(),
            routing_attempts_payload(attempted),
        );
    }
    payload
}

fn model_slot_payload(routing: &ModelRoutingDecision, selection: &RuntimeModelSelection) -> Value {
    json!({
        "serviceModelSlot": routing.service_model_slot,
        "service_model_slot": routing.service_model_slot,
        "selected": {
            "provider": selection.provider,
            "model": selection.model,
            "source": selection.source,
        },
        "requested": {
            "provider": routing.requested_provider,
            "model": routing.requested_model,
            "source": routing.settings_source,
        },
        "slots": routing
            .profile_slots
            .iter()
            .map(profile_slot_payload)
            .collect::<Vec<_>>(),
        "requiredCapabilities": REQUIRED_CODING_CAPABILITIES,
        "required_capabilities": REQUIRED_CODING_CAPABILITIES,
    })
}

fn profile_slot_payload(slot: &super::ProfileModelSlot) -> Value {
    json!({
        "slot": slot.slot,
        "provider": slot.provider,
        "model": slot.model,
        "source": slot.source,
        "decisionReason": slot.decision_reason,
        "decision_reason": slot.decision_reason,
        "capabilityTags": slot.capability_tags,
        "capability_tags": slot.capability_tags,
    })
}

fn routing_attempts_payload(attempted: &[RoutingAttempt]) -> Value {
    Value::Array(attempted.iter().map(RoutingAttempt::to_payload).collect())
}

fn selection_payload(selection: &RuntimeModelSelection) -> Value {
    json!({
        "provider": selection.provider,
        "model": selection.model,
        "source": selection.source,
        "reasoningEffort": selection.reasoning_effort,
        "reasoning_effort": selection.reasoning_effort,
    })
}
