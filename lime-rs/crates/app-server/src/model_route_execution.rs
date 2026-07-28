use app_server_protocol::{ModelTaskKind, ResolvedModelRoute};
use serde_json::{json, Value};

pub(crate) fn media_route_execution_binding(
    payload: &Value,
    route: &ResolvedModelRoute,
) -> Option<Value> {
    let task_kind = task_kind_from_payload(payload)?;
    let binding = media_task_binding_for_task(task_kind)?;
    let provider_id = non_empty(Some(&route.model_ref.provider_id))?;
    let model_id = non_empty(Some(&route.model_ref.model_id))?;
    let protocol = serde_json::to_value(&route.protocol).unwrap_or_else(|_| json!("unknown"));

    Some(json!({
        "version": 1,
        "status": "ready",
        "executionOwner": "media_task_worker",
        "executor": {
            "kind": "media_task_worker",
            "bindingKey": binding.binding_key,
            "endpointSource": "resolved_route"
        },
        "credentialResolver": {
            "owner": "media_task_worker",
            "source": "resolved_route_credential_ref",
            "providerId": provider_id,
            "credentialRef": route.auth.credential_ref,
            "secretMaterialStatus": "not_embedded",
            "authHeaderName": route.auth.header_name,
            "authHeaderPrefix": route.auth.header_prefix
        },
        "route": {
            "providerId": provider_id,
            "modelId": model_id,
            "protocol": protocol
        }
    }))
}

struct MediaTaskBinding {
    binding_key: &'static str,
}

fn media_task_binding_for_task(task_kind: ModelTaskKind) -> Option<MediaTaskBinding> {
    match task_kind {
        ModelTaskKind::ImageGenerate => Some(MediaTaskBinding {
            binding_key: "mediaTaskArtifact/image/create",
        }),
        ModelTaskKind::VideoGenerate => Some(MediaTaskBinding {
            binding_key: "mediaTaskArtifact/video/create",
        }),
        _ => None,
    }
}

fn task_kind_from_payload(payload: &Value) -> Option<ModelTaskKind> {
    let value = payload
        .pointer("/model_task_request/taskKind")
        .or_else(|| payload.pointer("/modelTaskRequest/taskKind"))
        .and_then(Value::as_str)?;
    serde_json::from_value(json!(value)).ok()
}

fn non_empty(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::{
        AuthKind, AuthMaterialRef, CapabilitySnapshot, EndpointInfo, EndpointKind, FramingKind,
        ModelRef, ModelRefSource, ProtocolKind, RouteDefaults, RoutingDecision, TransportKind,
    };

    #[test]
    fn media_route_execution_binding_delegates_supported_tasks_to_current_worker() {
        let route = ResolvedModelRoute {
            model_ref: ModelRef {
                provider_id: "openai-images".to_string(),
                model_id: "gpt-image-2".to_string(),
                variant: None,
                routing_slot: Some("image_generation_model".to_string()),
                source: ModelRefSource::Task,
            },
            protocol: ProtocolKind::OpenaiImages,
            endpoint: EndpointInfo {
                kind: EndpointKind::OpenaiCompatible,
                base_url: Some("https://api.openai.com/v1".to_string()),
                api_version: None,
                project: None,
                location: None,
                region: None,
            },
            auth: AuthMaterialRef {
                kind: AuthKind::ApiKeyRef,
                provider_id: Some("openai-images".to_string()),
                credential_ref: Some("runtime-api-key-key-1".to_string()),
                header_name: Some("Authorization".to_string()),
                header_prefix: Some("Bearer".to_string()),
            },
            transport: TransportKind::Http,
            framing: FramingKind::Json,
            defaults: RouteDefaults::default(),
            capability_snapshot: CapabilitySnapshot::default(),
            decision: RoutingDecision {
                routing_mode: "task_route".to_string(),
                decision_source: "media_task_artifact".to_string(),
                decision_reason: "explicit_task_model".to_string(),
                settings_source: None,
                service_model_slot: Some("image_generation_model".to_string()),
                fallback_chain: Vec::new(),
                candidate_count: 1,
                capability_gap: None,
            },
            failure: None,
        };

        let binding = media_route_execution_binding(
            &json!({
                "model_task_request": {
                    "taskKind": "image_generate"
                }
            }),
            &route,
        )
        .expect("binding");

        assert_eq!(
            binding["executor"]["bindingKey"].as_str(),
            Some("mediaTaskArtifact/image/create")
        );
        assert_eq!(
            binding["credentialResolver"]["owner"].as_str(),
            Some("media_task_worker")
        );
        assert_eq!(
            binding["credentialResolver"]["secretMaterialStatus"].as_str(),
            Some("not_embedded")
        );
        assert_eq!(
            binding["credentialResolver"]["credentialRef"].as_str(),
            Some("runtime-api-key-key-1")
        );
        assert_eq!(
            binding["credentialResolver"]["authHeaderName"].as_str(),
            Some("Authorization")
        );
        assert_eq!(
            binding["credentialResolver"]["authHeaderPrefix"].as_str(),
            Some("Bearer")
        );
        assert!(binding["executor"].get("baseUrl").is_none());
        assert!(binding["executor"].get("path").is_none());

        let video_binding = media_route_execution_binding(
            &json!({
                "model_task_request": {
                    "taskKind": "video_generate"
                }
            }),
            &route,
        )
        .expect("video binding");
        assert_eq!(
            video_binding["executor"]["bindingKey"].as_str(),
            Some("mediaTaskArtifact/video/create")
        );
    }
}
