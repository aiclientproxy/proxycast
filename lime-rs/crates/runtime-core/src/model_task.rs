use app_server_protocol::{
    CapabilityRequirement, CapabilitySnapshot, ModelCapabilitiesInfo, ModelRef, ModelRefSource,
    ModelTaskKind, ModelTaskRequest, ModelTaskSource,
};
use serde_json::{json, Value};

pub struct ModelTaskRequestInput {
    pub task_kind: ModelTaskKind,
    pub source: ModelTaskSource,
    pub provider_id: Option<String>,
    pub model_id: Option<String>,
    pub model_ref_source: ModelRefSource,
    pub modality_contract_key: Option<String>,
    pub routing_slot: Option<String>,
    pub task_families: Vec<String>,
    pub input_modalities: Vec<String>,
    pub output_modalities: Vec<String>,
    pub runtime_features: Vec<String>,
    pub capabilities: Vec<String>,
    pub session_id: Option<String>,
    pub thread_id: Option<String>,
    pub turn_id: Option<String>,
    pub content_id: Option<String>,
    pub trace_id: Option<String>,
}

pub fn build_model_task_request(input: ModelTaskRequestInput) -> ModelTaskRequest {
    let routing_slot = normalize_optional_string(input.routing_slot);
    let model_ref = match (
        normalize_optional_string(input.provider_id),
        normalize_optional_string(input.model_id),
    ) {
        (Some(provider_id), Some(model_id)) => Some(ModelRef {
            provider_id,
            model_id,
            variant: None,
            routing_slot: routing_slot.clone(),
            source: input.model_ref_source,
        }),
        _ => None,
    };

    ModelTaskRequest {
        task_kind: input.task_kind,
        source: input.source,
        model_ref,
        modality_contract_key: normalize_optional_string(input.modality_contract_key),
        routing_slot,
        requirements: CapabilityRequirement {
            task_families: normalize_string_list(input.task_families),
            input_modalities: normalize_string_list(input.input_modalities),
            output_modalities: normalize_string_list(input.output_modalities),
            runtime_features: normalize_string_list(input.runtime_features),
            capabilities: normalize_string_list(input.capabilities),
        },
        session_id: normalize_optional_string(input.session_id),
        thread_id: normalize_optional_string(input.thread_id),
        turn_id: normalize_optional_string(input.turn_id),
        content_id: normalize_optional_string(input.content_id),
        trace_id: normalize_optional_string(input.trace_id),
    }
}

pub fn model_task_request_value(request: &ModelTaskRequest) -> Value {
    serde_json::to_value(request).unwrap_or_else(|_| json!({}))
}

pub fn capability_snapshot_from_model_capabilities(value: &Value) -> CapabilitySnapshot {
    let capabilities_value = value.get("capabilities");

    CapabilitySnapshot {
        task_families: string_array_field(value, &["taskFamilies", "task_families"]),
        input_modalities: string_array_field(value, &["inputModalities", "input_modalities"]),
        output_modalities: string_array_field(value, &["outputModalities", "output_modalities"]),
        runtime_features: string_array_field(value, &["runtimeFeatures", "runtime_features"]),
        capabilities: ModelCapabilitiesInfo {
            vision: bool_field(capabilities_value, &["vision"]).unwrap_or(false),
            tools: bool_field(capabilities_value, &["tools"]).unwrap_or(false),
            streaming: bool_field(capabilities_value, &["streaming"]).unwrap_or(false),
            json_mode: bool_field(capabilities_value, &["jsonMode", "json_mode"]).unwrap_or(false),
            function_calling: bool_field(
                capabilities_value,
                &["functionCalling", "function_calling"],
            )
            .unwrap_or(false),
            reasoning: bool_field(capabilities_value, &["reasoning"]).unwrap_or(false),
            reasoning_effort: capabilities_value
                .and_then(|capabilities| {
                    capabilities
                        .get("reasoningEffort")
                        .or_else(|| capabilities.get("reasoning_effort"))
                })
                .cloned()
                .and_then(|support| serde_json::from_value(support).ok()),
        },
        source: None,
        reason_code: None,
    }
}

pub fn route_capability_gap(
    task_request: &ModelTaskRequest,
    snapshot: &CapabilitySnapshot,
) -> Option<String> {
    first_missing_from(
        "task_family",
        &task_request.requirements.task_families,
        &snapshot.task_families,
    )
    .or_else(|| {
        first_missing_from(
            "input_modality",
            &task_request.requirements.input_modalities,
            &snapshot.input_modalities,
        )
    })
    .or_else(|| {
        first_missing_from(
            "output_modality",
            &task_request.requirements.output_modalities,
            &snapshot.output_modalities,
        )
    })
    .or_else(|| {
        first_missing_from(
            "runtime_feature",
            &task_request.requirements.runtime_features,
            &snapshot.runtime_features,
        )
    })
    .or_else(|| first_missing_capability(&task_request.requirements.capabilities, snapshot))
}

fn normalize_optional_string(value: Option<String>) -> Option<String> {
    value.and_then(|raw| {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn normalize_string_list(values: Vec<String>) -> Vec<String> {
    values
        .into_iter()
        .filter_map(|value| normalize_optional_string(Some(value)))
        .collect()
}

fn bool_field(value: Option<&Value>, keys: &[&str]) -> Option<bool> {
    let value = value?;
    keys.iter()
        .filter_map(|key| value.get(*key))
        .find_map(Value::as_bool)
}

fn string_array_field(value: &Value, keys: &[&str]) -> Vec<String> {
    keys.iter()
        .filter_map(|key| value.get(*key))
        .find_map(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .filter_map(normalize_token)
                .collect()
        })
        .unwrap_or_default()
}

fn first_missing_from(kind: &str, required: &[String], available: &[String]) -> Option<String> {
    let available = normalized_values(available);
    required
        .iter()
        .filter_map(|value| normalize_token(value))
        .find(|required| !available.iter().any(|available| available == required))
        .map(|missing| format!("{kind}:{missing}"))
}

fn first_missing_capability(required: &[String], snapshot: &CapabilitySnapshot) -> Option<String> {
    required
        .iter()
        .filter_map(|value| normalize_token(value))
        .find(|required| !capability_satisfied(required, snapshot))
        .map(|missing| format!("capability:{missing}"))
}

fn capability_satisfied(required: &str, snapshot: &CapabilitySnapshot) -> bool {
    match required {
        "coding" => true,
        "vision" | "image_understanding" => snapshot.capabilities.vision,
        "tools" | "tool_calling" => {
            snapshot.capabilities.tools
                || normalized_values(&snapshot.runtime_features)
                    .iter()
                    .any(|feature| feature == "tool_calling")
        }
        "function_calling" => snapshot.capabilities.function_calling,
        "streaming" => {
            snapshot.capabilities.streaming
                || normalized_values(&snapshot.runtime_features)
                    .iter()
                    .any(|feature| feature == "streaming")
        }
        "json_mode" | "json_schema" => {
            snapshot.capabilities.json_mode
                || normalized_values(&snapshot.runtime_features)
                    .iter()
                    .any(|feature| feature == "json_schema")
        }
        "reasoning" => {
            snapshot.capabilities.reasoning
                || normalized_values(&snapshot.task_families)
                    .iter()
                    .any(|family| family == "reasoning")
        }
        _ => true,
    }
}

fn normalized_values(values: &[String]) -> Vec<String> {
    values
        .iter()
        .filter_map(|value| normalize_token(value))
        .collect()
}

fn normalize_token(value: &str) -> Option<String> {
    let normalized = value.trim().to_ascii_lowercase().replace('-', "_");
    if normalized.is_empty() {
        None
    } else {
        Some(normalized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_model_task_request_trims_identity_fields() {
        let request = build_model_task_request(ModelTaskRequestInput {
            task_kind: ModelTaskKind::ImageGenerate,
            source: ModelTaskSource::MediaTaskArtifact,
            provider_id: Some(" openai ".to_string()),
            model_id: Some(" gpt-image-2 ".to_string()),
            model_ref_source: ModelRefSource::Task,
            modality_contract_key: Some(" image_generation ".to_string()),
            routing_slot: Some(" image_generation_model ".to_string()),
            task_families: vec![" image_generation ".to_string()],
            input_modalities: vec![" text ".to_string()],
            output_modalities: vec![" image ".to_string()],
            runtime_features: vec![],
            capabilities: vec![" image_generation ".to_string()],
            session_id: Some(" sess ".to_string()),
            thread_id: Some(" thread ".to_string()),
            turn_id: Some(" turn ".to_string()),
            content_id: Some(" content ".to_string()),
            trace_id: None,
        });

        let model_ref = request.model_ref.expect("model ref");
        assert_eq!(model_ref.provider_id, "openai");
        assert_eq!(model_ref.model_id, "gpt-image-2");
        assert_eq!(
            model_ref.routing_slot.as_deref(),
            Some("image_generation_model")
        );
        assert_eq!(
            request.modality_contract_key.as_deref(),
            Some("image_generation")
        );
        assert_eq!(
            request.routing_slot.as_deref(),
            Some("image_generation_model")
        );
        assert_eq!(request.session_id.as_deref(), Some("sess"));
        assert_eq!(
            request.requirements.task_families,
            vec!["image_generation".to_string()]
        );
    }

    #[test]
    fn build_model_task_request_omits_incomplete_model_ref() {
        let request = build_model_task_request(ModelTaskRequestInput {
            task_kind: ModelTaskKind::Chat,
            source: ModelTaskSource::AgentTurn,
            provider_id: Some("openai".to_string()),
            model_id: None,
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

        assert!(request.model_ref.is_none());
    }

    #[test]
    fn capability_snapshot_ignores_picker_and_catalog_metadata() {
        let snapshot = capability_snapshot_from_model_capabilities(&json!({
            "displayName": "Vision Pro",
            "providerName": "Provider UI Name",
            "tier": "max",
            "status": "active",
            "source": "models.dev",
            "releaseDate": "2026-07-05",
            "isLatest": true,
            "pricing": {
                "inputPerMillion": 1.0,
                "outputPerMillion": 2.0
            },
            "deploymentSource": "oem_cloud",
            "managementPlane": "oem_control_plane",
            "aliasSource": "official",
            "capabilities": {
                "vision": false,
                "tools": false,
                "streaming": true,
                "jsonMode": false,
                "functionCalling": false,
                "reasoning": false
            },
            "taskFamilies": ["chat"],
            "inputModalities": ["text"],
            "outputModalities": ["text"],
            "runtimeFeatures": ["streaming"]
        }));

        assert_eq!(snapshot.task_families, vec!["chat".to_string()]);
        assert_eq!(snapshot.input_modalities, vec!["text".to_string()]);
        assert_eq!(snapshot.output_modalities, vec!["text".to_string()]);
        assert_eq!(snapshot.runtime_features, vec!["streaming".to_string()]);
        assert!(!snapshot.capabilities.vision);
        assert!(!snapshot.capabilities.tools);
        assert!(!snapshot.capabilities.reasoning);
        assert_eq!(snapshot.source, None);
        assert_eq!(snapshot.reason_code, None);
    }

    #[test]
    fn route_capability_gap_reports_first_missing_requirement() {
        let request = build_model_task_request(ModelTaskRequestInput {
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
        let snapshot = capability_snapshot_from_model_capabilities(&json!({
            "capabilities": {
                "vision": false,
                "streaming": true
            },
            "taskFamilies": ["chat"],
            "inputModalities": ["text"],
            "outputModalities": ["text"],
            "runtimeFeatures": ["streaming"]
        }));

        assert_eq!(
            route_capability_gap(&request, &snapshot).as_deref(),
            Some("task_family:image_generation")
        );
    }
}
