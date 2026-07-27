use app_server_protocol::{ModelTaskRequest, ResolvedModelRoute};
use lime_core::database::dao::api_key_provider::ProviderWithKeys;
use runtime_core::ModelRouteProvider;
pub(crate) use runtime_core::{DirectRouteConfig, ModelRouteSelection};
use serde_json::Value;
use std::borrow::Cow;

pub(crate) fn resolved_route_from_task_with_credential(
    task_request: &ModelTaskRequest,
    selection: ModelRouteSelection<'_>,
    routing_payload: &Value,
    provider: Option<&ProviderWithKeys>,
    credential_ref: Option<&str>,
    direct_config: Option<DirectRouteConfig<'_>>,
) -> ResolvedModelRoute {
    let route_provider = provider.map(|provider| model_route_provider(provider, credential_ref));
    runtime_core::resolved_route_from_task(
        task_request,
        selection,
        routing_payload,
        route_provider.as_ref(),
        direct_config,
    )
}

fn model_route_provider<'a>(
    provider: &'a ProviderWithKeys,
    credential_ref: Option<&'a str>,
) -> ModelRouteProvider<'a> {
    let effective_provider_type = provider.provider.effective_provider_type();
    let spec = effective_provider_type.runtime_spec();
    ModelRouteProvider {
        provider_id: &provider.provider.id,
        provider_type: Cow::Owned(effective_provider_type.to_string()),
        base_url: Some(&provider.provider.api_host),
        api_version: provider.provider.api_version.as_deref(),
        project: provider.provider.project.as_deref(),
        location: provider.provider.location.as_deref(),
        region: provider.provider.region.as_deref(),
        credential_ref: credential_ref.map(ToString::to_string),
        auth_header: spec.auth_header,
        auth_prefix: spec.auth_prefix,
        prompt_cache_mode: provider
            .provider
            .prompt_cache_mode
            .map(|value| format!("{value:?}").to_ascii_lowercase()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_task_contract::{build_model_task_request, ModelTaskRequestInput};
    use app_server_protocol::{
        ModelRefSource, ModelTaskKind, ModelTaskSource, ProtocolKind, RouteFailureCategory,
    };
    use serde_json::json;

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
                "reasonCode": "matched_provider_models",
                "modelCapabilities": {
                    "provenance": "provider_explicit",
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

        let route = resolved_route_from_task_with_credential(
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
        let route = resolved_route_from_task_with_credential(
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
                        "provenance": "provider_explicit",
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
        let route = resolved_route_from_task_with_credential(
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
                        "provenance": "provider_explicit",
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
            None,
        );

        assert_eq!(route.protocol, ProtocolKind::OpenaiResponses);
        assert!(route.failure.is_none());
    }
}
