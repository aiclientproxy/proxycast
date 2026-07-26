use super::{
    ModelRouteExclusion, ModelRoutingDecision, ProfileModelSlot, ProviderReadiness, RoutingAttempt,
    RoutingResolution, RuntimeModelSelection, DEFAULT_CODING_SLOT, DERIVED_MODEL_SLOT_SOURCE,
    KNOWN_CODING_SLOTS, PROFILE_MODEL_SLOT_SOURCE,
};
use serde_json::{Map, Value};

pub fn selection_from_profile_model_slot(
    metadata_values: &[&Value],
    reasoning_effort: Option<String>,
    preferred_slot: Option<&str>,
) -> Option<RuntimeModelSelection> {
    let slot = primary_profile_model_slot(metadata_values, preferred_slot)?;
    Some(RuntimeModelSelection {
        provider: slot.provider?,
        model: slot.model?,
        source: PROFILE_MODEL_SLOT_SOURCE,
        reasoning_effort,
    })
}

pub fn resolve_ready_model_routing<F>(
    metadata_values: &[&Value],
    selection: &RuntimeModelSelection,
    resolve_readiness: F,
) -> Result<RoutingResolution, String>
where
    F: FnMut(&RuntimeModelSelection) -> Result<ProviderReadiness, String>,
{
    resolve_ready_model_routing_with_exclusions(metadata_values, selection, &[], resolve_readiness)
}

pub fn resolve_ready_model_routing_with_exclusions<F>(
    metadata_values: &[&Value],
    selection: &RuntimeModelSelection,
    excluded_routes: &[ModelRouteExclusion],
    mut resolve_readiness: F,
) -> Result<RoutingResolution, String>
where
    F: FnMut(&RuntimeModelSelection) -> Result<ProviderReadiness, String>,
{
    let candidates = routing_candidates_from_metadata(metadata_values, selection);
    let mut attempted = Vec::new();
    let mut blocked_resolution = None;

    for candidate in candidates {
        let mut routing = resolve_model_routing_for_candidate(metadata_values, &candidate);
        if let Some(runtime_failure) = excluded_routes.iter().find(|excluded| {
            excluded.provider == candidate.provider && excluded.model == candidate.model
        }) {
            let readiness = ProviderReadiness::runtime_failure(runtime_failure.reason_code);
            attempted.push(RoutingAttempt {
                slot: routing.service_model_slot.clone(),
                provider: candidate.provider.clone(),
                model: candidate.model.clone(),
                source: candidate.source.to_string(),
                readiness: readiness.clone(),
                runtime_failure: Some(runtime_failure.clone()),
            });
            routing.fallback_chain = fallback_chain_from_attempts(&attempted);
            blocked_resolution = Some(RoutingResolution {
                selection: candidate,
                routing,
                readiness,
                attempted: attempted.clone(),
            });
            continue;
        }
        let readiness = resolve_readiness(&candidate)?;
        attempted.push(RoutingAttempt {
            slot: routing.service_model_slot.clone(),
            provider: candidate.provider.clone(),
            model: candidate.model.clone(),
            source: candidate.source.to_string(),
            readiness: readiness.clone(),
            runtime_failure: None,
        });
        routing.fallback_chain = fallback_chain_from_attempts(&attempted);
        let resolution = RoutingResolution {
            selection: candidate,
            routing,
            readiness,
            attempted: attempted.clone(),
        };
        if resolution.readiness.ready {
            return Ok(resolution);
        }
        blocked_resolution = Some(resolution);
    }

    blocked_resolution
        .ok_or_else(|| "RuntimeCore could not build a model routing candidate".to_string())
}

pub fn resolve_model_routing_for_candidate(
    metadata_values: &[&Value],
    selection: &RuntimeModelSelection,
) -> ModelRoutingDecision {
    let profile_slots = profile_model_slots_from_metadata_values(metadata_values);
    let primary_slot = profile_slots
        .iter()
        .find(|slot| slot.slot == DEFAULT_CODING_SLOT && slot_matches_selection(slot, selection))
        .or_else(|| {
            profile_slots
                .iter()
                .find(|slot| slot_matches_selection(slot, selection))
        })
        .or_else(|| {
            profile_slots
                .iter()
                .find(|slot| slot.slot == DEFAULT_CODING_SLOT)
        })
        .or_else(|| profile_slots.iter().find(|slot| slot.slot == "base"));
    let service_model_slot = primary_slot
        .map(|slot| slot.slot.clone())
        .unwrap_or_else(|| DEFAULT_CODING_SLOT.to_string());
    let requested_provider = primary_slot
        .and_then(|slot| slot.provider.clone())
        .or_else(|| Some(selection.provider.clone()));
    let requested_model = primary_slot
        .and_then(|slot| slot.model.clone())
        .or_else(|| Some(selection.model.clone()));
    let settings_source = primary_slot
        .map(|slot| slot.source.clone())
        .unwrap_or_else(|| DERIVED_MODEL_SLOT_SOURCE.to_string());
    let requested_pair =
        provider_model_pair(requested_provider.as_deref(), requested_model.as_deref());
    let selected_pair = provider_model_pair(Some(&selection.provider), Some(&selection.model));
    let fallback_chain = match (requested_pair.as_ref(), selected_pair.as_ref()) {
        (Some(requested), Some(selected)) if requested != selected => {
            vec![requested.clone(), selected.clone()]
        }
        _ => Vec::new(),
    };
    let decision_reason = primary_slot
        .and_then(|slot| slot.decision_reason.clone())
        .unwrap_or_else(|| {
            if profile_slots.is_empty() {
                "selection_derived_as_coding_slot".to_string()
            } else if selection.source == PROFILE_MODEL_SLOT_SOURCE {
                "profile_slot_selected".to_string()
            } else if fallback_chain.is_empty() {
                "selection_matches_profile_slot".to_string()
            } else {
                "selection_overrode_profile_slot".to_string()
            }
        });

    ModelRoutingDecision {
        service_model_slot,
        requested_provider,
        requested_model,
        settings_source,
        decision_reason,
        fallback_chain,
        profile_slots,
    }
}

fn routing_candidates_from_metadata(
    metadata_values: &[&Value],
    selection: &RuntimeModelSelection,
) -> Vec<RuntimeModelSelection> {
    let mut candidates = Vec::new();
    push_unique_selection(&mut candidates, selection.clone());

    for slot in profile_model_slots_from_metadata_values(metadata_values) {
        if !candidate_fallback_slot(&slot.slot) {
            continue;
        }
        let Some(provider) = slot.provider else {
            continue;
        };
        let Some(model) = slot.model else {
            continue;
        };
        push_unique_selection(
            &mut candidates,
            RuntimeModelSelection {
                provider,
                model,
                source: PROFILE_MODEL_SLOT_SOURCE,
                reasoning_effort: selection.reasoning_effort.clone(),
            },
        );
    }

    candidates
}

fn push_unique_selection(
    candidates: &mut Vec<RuntimeModelSelection>,
    selection: RuntimeModelSelection,
) {
    if candidates.iter().any(|candidate| {
        candidate.provider == selection.provider && candidate.model == selection.model
    }) {
        return;
    }
    candidates.push(selection);
}

fn candidate_fallback_slot(slot: &str) -> bool {
    matches!(slot, DEFAULT_CODING_SLOT | "base" | "fast" | "local")
}

fn fallback_chain_from_attempts(attempts: &[RoutingAttempt]) -> Vec<String> {
    attempts
        .iter()
        .map(|attempt| format!("{}/{}", attempt.provider, attempt.model))
        .collect()
}

fn slot_matches_selection(slot: &ProfileModelSlot, selection: &RuntimeModelSelection) -> bool {
    slot.provider.as_deref() == Some(selection.provider.as_str())
        && slot.model.as_deref() == Some(selection.model.as_str())
}

fn slot_is_selectable(slot: &ProfileModelSlot) -> bool {
    slot.provider.is_some() && slot.model.is_some()
}

fn primary_profile_model_slot(
    metadata_values: &[&Value],
    preferred_slot: Option<&str>,
) -> Option<ProfileModelSlot> {
    let slots = profile_model_slots_from_metadata_values(metadata_values);
    if let Some(preferred_slot) = preferred_slot {
        if let Some(slot) = slots
            .iter()
            .find(|slot| slot.slot == preferred_slot && slot_is_selectable(slot))
        {
            return Some(slot.clone());
        }
    }

    slots
        .iter()
        .find(|slot| slot.slot == DEFAULT_CODING_SLOT && slot_is_selectable(slot))
        .or_else(|| {
            slots
                .iter()
                .find(|slot| slot.slot == "base" && slot_is_selectable(slot))
        })
        .cloned()
}

fn profile_model_slots_from_metadata_values(metadata_values: &[&Value]) -> Vec<ProfileModelSlot> {
    metadata_values
        .iter()
        .find_map(|metadata| profile_model_slots_from_metadata(metadata))
        .unwrap_or_default()
}

fn profile_model_slots_from_metadata(metadata: &Value) -> Option<Vec<ProfileModelSlot>> {
    let container = [
        "/harness/coding_model_slots",
        "/harness/codingModelSlots",
        "/harness/model_slots",
        "/harness/modelSlots",
        "/coding_model_slots",
        "/codingModelSlots",
        "/model_slots",
        "/modelSlots",
        "/coding_profile/model_slots",
        "/codingProfile/modelSlots",
    ]
    .iter()
    .find_map(|pointer| metadata.pointer(pointer))?;

    match container {
        Value::Object(object) => Some(slots_from_object(object)),
        Value::Array(items) => Some(slots_from_array(items)),
        _ => None,
    }
    .filter(|slots| !slots.is_empty())
}

fn slots_from_object(object: &Map<String, Value>) -> Vec<ProfileModelSlot> {
    KNOWN_CODING_SLOTS
        .iter()
        .filter_map(|slot| object.get(*slot).map(|value| (*slot, value)))
        .filter_map(|(slot, value)| profile_slot_from_value(slot, value))
        .collect()
}

fn slots_from_array(items: &[Value]) -> Vec<ProfileModelSlot> {
    items
        .iter()
        .filter_map(|value| {
            let slot = string_field(
                value,
                &[
                    "slot",
                    "id",
                    "name",
                    "serviceModelSlot",
                    "service_model_slot",
                ],
            )?;
            profile_slot_from_value(&slot, value)
        })
        .filter(|slot| KNOWN_CODING_SLOTS.contains(&slot.slot.as_str()))
        .collect()
}

fn profile_slot_from_value(slot: &str, value: &Value) -> Option<ProfileModelSlot> {
    let slot = normalized_slot_name(slot)?;
    let source = string_field(value, &["source", "settingsSource", "settings_source"])
        .unwrap_or_else(|| PROFILE_MODEL_SLOT_SOURCE.to_string());
    let capability_tags = string_array_field(
        value,
        &[
            "capabilityTags",
            "capability_tags",
            "capabilities",
            "requiredCapabilities",
            "required_capabilities",
        ],
    );
    Some(ProfileModelSlot {
        slot,
        provider: string_field(
            value,
            &[
                "provider",
                "providerId",
                "provider_id",
                "providerPreference",
                "provider_preference",
                "selectedProvider",
                "selected_provider",
            ],
        ),
        model: string_field(
            value,
            &[
                "model",
                "modelName",
                "model_name",
                "modelPreference",
                "model_preference",
                "selectedModel",
                "selected_model",
            ],
        ),
        source,
        decision_reason: string_field(
            value,
            &[
                "reason",
                "reasonCode",
                "reason_code",
                "decisionReason",
                "decision_reason",
            ],
        ),
        capability_tags,
    })
}

fn provider_model_pair(provider: Option<&str>, model: Option<&str>) -> Option<String> {
    Some(format!("{}/{}", non_empty(provider)?, non_empty(model)?))
}

fn normalized_slot_name(value: &str) -> Option<String> {
    let value = value.trim().to_ascii_lowercase();
    KNOWN_CODING_SLOTS
        .contains(&value.as_str())
        .then_some(value)
}

fn string_field(value: &Value, keys: &[&str]) -> Option<String> {
    let object = value.as_object()?;
    keys.iter()
        .filter_map(|key| object.get(*key))
        .find_map(|value| value.as_str().and_then(|value| non_empty(Some(value))))
}

fn string_array_field(value: &Value, keys: &[&str]) -> Vec<String> {
    let Some(object) = value.as_object() else {
        return Vec::new();
    };
    keys.iter()
        .filter_map(|key| object.get(*key))
        .find_map(|value| {
            value.as_array().map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .filter_map(|value| non_empty(Some(value)))
                    .collect::<Vec<_>>()
            })
        })
        .unwrap_or_default()
}

fn non_empty(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}
