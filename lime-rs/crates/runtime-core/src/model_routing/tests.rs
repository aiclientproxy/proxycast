use super::*;
use crate::FailureClassification;
use serde_json::json;

fn selection(provider: &str, model: &str) -> RuntimeModelSelection {
    RuntimeModelSelection {
        provider: provider.to_string(),
        model: model.to_string(),
        source: PROFILE_MODEL_SLOT_SOURCE,
        reasoning_effort: None,
    }
}

#[test]
fn selection_from_profile_slot_reads_harness_metadata() {
    let metadata = json!({
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
    });

    let selection =
        selection_from_profile_model_slot(&[&metadata], Some("medium".to_string()), None)
            .expect("slot selection");

    assert_eq!(selection.provider, "custom-coding");
    assert_eq!(selection.model, "coder-large");
    assert_eq!(selection.source, PROFILE_MODEL_SLOT_SOURCE);
    assert_eq!(selection.reasoning_effort.as_deref(), Some("medium"));
}

#[test]
fn selection_from_profile_slot_honors_runtime_preferred_slot() {
    let metadata = json!({
        "harness": {
            "model_slots": {
                "coding": {
                    "provider": "custom-coding",
                    "model": "coder-large"
                },
                "fast": {
                    "provider": "responsive-provider",
                    "model": "fast-chat"
                },
                "base": {
                    "provider": "openai",
                    "model": "gpt-4.1-mini"
                }
            }
        }
    });

    let selection = selection_from_profile_model_slot(&[&metadata], None, Some("fast"))
        .expect("fast slot selection");

    assert_eq!(selection.provider, "responsive-provider");
    assert_eq!(selection.model, "fast-chat");
    let routing = resolve_model_routing_for_candidate(&[&metadata], &selection);
    assert_eq!(routing.service_model_slot, "fast");
    assert_eq!(routing.decision_reason, "profile_slot_selected".to_string());
}

#[test]
fn selection_from_profile_slot_ignores_fast_slot_without_runtime_preference() {
    let metadata = json!({
        "harness": {
            "model_slots": {
                "coding": {
                    "provider": "custom-coding",
                    "model": "coder-large"
                },
                "fast": {
                    "provider": "responsive-provider",
                    "model": "fast-chat"
                }
            }
        }
    });

    let selection =
        selection_from_profile_model_slot(&[&metadata], None, None).expect("coding slot selection");

    assert_eq!(selection.provider, "custom-coding");
    assert_eq!(selection.model, "coder-large");
}

#[test]
fn routing_payload_keeps_review_fast_local_as_diagnostics_only() {
    let metadata = json!({
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
    });
    let selection = selection("custom-coding", "coder-large");
    let routing = resolve_model_routing_for_candidate(&[&metadata], &selection);
    let readiness = ProviderReadiness::direct_request_ready();
    let model_registry = json!({
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
    });

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
    let metadata = json!({
        "harness": {
            "coding_model_slots": {
                "coding": {
                    "provider": "custom-coding",
                    "model": "missing-key-coder"
                },
                "base": {
                    "provider": "openai",
                    "model": "gpt-4.1-mini"
                }
            }
        }
    });
    let requested =
        selection_from_profile_model_slot(&[&metadata], None, None).expect("requested selection");

    let resolution = resolve_ready_model_routing(&[&metadata], &requested, |candidate| {
        if candidate.provider == "openai" {
            Ok(ProviderReadiness::provider_store_ready(
                Some("openai".to_string()),
                1,
                1,
            ))
        } else {
            Ok(ProviderReadiness::provider_store_needs_setup(
                "missing_enabled_api_key",
                Some("openai".to_string()),
                Some(true),
                0,
                0,
            ))
        }
    })
    .expect("routing resolution");

    assert_eq!(requested.provider, "custom-coding");
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
            "custom-coding/missing-key-coder".to_string(),
            "openai/gpt-4.1-mini".to_string()
        ]
    );
}

#[test]
fn ready_routing_falls_back_from_unready_fast_slot_to_coding_slot() {
    let metadata = json!({
        "harness": {
            "model_slots": {
                "fast": {
                    "provider": "responsive-provider",
                    "model": "fast-chat"
                },
                "coding": {
                    "provider": "custom-coding",
                    "model": "coder-large"
                }
            }
        }
    });
    let requested = selection_from_profile_model_slot(&[&metadata], None, Some("fast"))
        .expect("requested fast selection");

    let resolution = resolve_ready_model_routing(&[&metadata], &requested, |candidate| {
        if candidate.provider == "custom-coding" {
            Ok(ProviderReadiness::provider_store_ready(
                Some("custom".to_string()),
                1,
                1,
            ))
        } else {
            Ok(ProviderReadiness::provider_store_needs_setup(
                "missing_enabled_api_key",
                Some("custom".to_string()),
                Some(true),
                0,
                0,
            ))
        }
    })
    .expect("routing resolution");

    assert_eq!(requested.provider, "responsive-provider");
    assert_eq!(resolution.selection.provider, "custom-coding");
    assert_eq!(resolution.selection.model, "coder-large");
    assert!(resolution.readiness.ready);
    assert_eq!(resolution.routing.service_model_slot, "coding");
    assert_eq!(resolution.attempted.len(), 2);
    assert_eq!(resolution.attempted[0].slot, "fast");
    assert!(!resolution.attempted[0].readiness.ready);
    assert_eq!(resolution.attempted[1].slot, "coding");
    assert_eq!(
        resolution.routing.fallback_chain,
        vec![
            "responsive-provider/fast-chat".to_string(),
            "custom-coding/coder-large".to_string()
        ]
    );
}

#[test]
fn ready_routing_excludes_runtime_failed_route_and_records_reroute_evidence() {
    let metadata = json!({
        "harness": {
            "coding_model_slots": {
                "coding": {
                    "provider": "primary-provider",
                    "model": "primary-model"
                },
                "base": {
                    "provider": "backup-provider",
                    "model": "backup-model"
                }
            }
        }
    });
    let requested = selection("primary-provider", "primary-model");
    let excluded = ModelRouteExclusion::new(
        "primary-provider",
        "primary-model",
        FailureClassification::ProviderInternal,
    );

    let resolution = resolve_ready_model_routing_with_exclusions(
        &[&metadata],
        &requested,
        &[excluded],
        |candidate| {
            assert_eq!(candidate.provider, "backup-provider");
            Ok(ProviderReadiness::provider_store_ready(
                Some("openai".to_string()),
                1,
                1,
            ))
        },
    )
    .expect("rerouted resolution");

    assert_eq!(resolution.selection.provider, "backup-provider");
    assert_eq!(resolution.selection.model, "backup-model");
    assert_eq!(resolution.attempted.len(), 2);
    assert_eq!(resolution.attempted[0].readiness.source, "runtime_failure");
    assert_eq!(
        resolution.attempted[0].readiness.reason_code,
        Some("provider_internal_failure")
    );
    assert_eq!(
        resolution.attempted[0]
            .runtime_failure
            .as_ref()
            .map(|failure| failure.classification),
        Some(FailureClassification::ProviderInternal)
    );
    assert!(resolution.attempted[1].runtime_failure.is_none());

    let payload = routing_fallback_applied_payload(
        &requested,
        &resolution.selection,
        &resolution.routing,
        &resolution.readiness,
        &json!({}),
        &resolution.attempted,
    );
    assert_eq!(
        payload["fallbackReason"].as_str(),
        Some("runtime_provider_failure")
    );
    assert_eq!(
        payload["runtimeFailure"]["classification"].as_str(),
        Some("provider-internal")
    );
    assert_eq!(
        payload["routingAttempts"][0]["runtimeFailure"]["reasonCode"].as_str(),
        Some("provider_internal_failure")
    );
}

#[test]
fn credential_failure_keeps_route_available_without_exposing_credential_ref() {
    let requested = selection("primary-provider", "primary-model");
    let credential_ref = "runtime-api-key-secret-key-id";
    let excluded = ModelRouteExclusion::for_credential(
        "primary-provider",
        "primary-model",
        credential_ref,
        FailureClassification::Authentication,
    );

    assert!(excluded.matches_route("primary-provider", "primary-model"));
    assert!(!excluded.excludes_entire_route());
    assert_eq!(excluded.credential_ref(), Some(credential_ref));
    assert!(!excluded.to_payload().to_string().contains(credential_ref));
    assert!(!format!("{excluded:?}").contains(credential_ref));

    let resolution =
        resolve_ready_model_routing_with_exclusions(&[], &requested, &[excluded], |_| {
            Ok(ProviderReadiness::provider_store_ready(
                Some("openai".to_string()),
                2,
                2,
            ))
        })
        .expect("same route remains ready through another credential");

    assert_eq!(resolution.selection, requested);
    assert_eq!(resolution.attempted.len(), 1);
    assert!(resolution.attempted[0].runtime_failure.is_none());
}
