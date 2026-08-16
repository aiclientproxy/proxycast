use serde::Serialize;
use serde_json::{json, Value};

pub(crate) const IMAGE_GENERATION_CONTRACT_KEY: &str = "image_generation";
pub(crate) const VIDEO_GENERATION_CONTRACT_KEY: &str = "video_generation";
pub(crate) const VOICE_GENERATION_CONTRACT_KEY: &str = "voice_generation";
pub(crate) const AUDIO_TRANSCRIPTION_CONTRACT_KEY: &str = "audio_transcription";
pub(crate) const IMAGE_GENERATION_ROUTING_SLOT: &str = "image_generation_model";
pub(crate) const VIDEO_GENERATION_ROUTING_SLOT: &str = "video_generation_model";
pub(crate) const VOICE_GENERATION_ROUTING_SLOT: &str = "voice_generation_model";
pub(crate) const AUDIO_TRANSCRIPTION_ROUTING_SLOT: &str = "audio_transcription_model";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MediaRuntimeContractKind {
    ImageGeneration,
    VideoGeneration,
    VoiceGeneration,
    AudioTranscription,
}

pub(crate) fn runtime_contract_or_default(
    override_contract: Option<&Value>,
    kind: MediaRuntimeContractKind,
) -> Value {
    override_contract
        .cloned()
        .unwrap_or_else(|| default_runtime_contract(kind))
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct MediaRuntimeContractProjection {
    pub(crate) contract_key: Option<String>,
    pub(crate) execution_profile_key: Option<String>,
    pub(crate) executor_adapter_key: Option<String>,
    pub(crate) executor_kind: Option<String>,
    pub(crate) executor_binding_key: Option<String>,
    pub(crate) policy_refs: Vec<String>,
}

pub(crate) fn runtime_contract_projection_from_payload(
    payload: &Value,
) -> MediaRuntimeContractProjection {
    let runtime_contract = runtime_contract_value(payload);
    let contract_key = string_field(payload, &["modality_contract_key", "modalityContractKey"])
        .or_else(|| {
            runtime_contract.and_then(|value| string_field(value, &["contract_key", "contractKey"]))
        });
    let route_execution_status = runtime_contract
        .and_then(|value| string_field(value, &["route_execution_status", "routeExecutionStatus"]));
    let suppress_executor_projection =
        is_metadata_only_contract(contract_key.as_deref(), route_execution_status.as_deref());

    MediaRuntimeContractProjection {
        contract_key,
        execution_profile_key: nested_string_field(
            runtime_contract,
            &["execution_profile", "executionProfile"],
            &["profile_key", "profileKey"],
        ),
        executor_adapter_key: if suppress_executor_projection {
            None
        } else {
            nested_string_field(
                runtime_contract,
                &["executor_adapter", "executorAdapter"],
                &["adapter_key", "adapterKey"],
            )
        },
        executor_kind: if suppress_executor_projection {
            None
        } else {
            nested_string_field(
                runtime_contract,
                &["executor_binding", "executorBinding"],
                &["executor_kind", "executorKind"],
            )
        },
        executor_binding_key: if suppress_executor_projection {
            None
        } else {
            nested_string_field(
                runtime_contract,
                &["executor_binding", "executorBinding"],
                &["binding_key", "bindingKey"],
            )
        },
        policy_refs: runtime_contract
            .and_then(|value| {
                value
                    .get("limecore_policy_refs")
                    .or_else(|| value.get("limecorePolicyRefs"))
            })
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .filter_map(normalize_string)
                    .collect()
            })
            .unwrap_or_default(),
    }
}

fn is_metadata_only_contract(
    _contract_key: Option<&str>,
    route_execution_status: Option<&str>,
) -> bool {
    route_execution_status == Some("metadata_only")
}

fn default_runtime_contract(kind: MediaRuntimeContractKind) -> Value {
    serde_json::to_value(RuntimeContractSpec::from(kind)).unwrap_or_else(|_| json!({}))
}

fn runtime_contract_value(payload: &Value) -> Option<&Value> {
    payload
        .get("runtime_contract")
        .or_else(|| payload.get("runtimeContract"))
}

fn nested_string_field(
    value: Option<&Value>,
    parent_keys: &[&str],
    child_keys: &[&str],
) -> Option<String> {
    value
        .and_then(|value| value_from_keys(value, parent_keys))
        .and_then(|value| string_field(value, child_keys))
}

fn string_field(value: &Value, keys: &[&str]) -> Option<String> {
    value_from_keys(value, keys)
        .and_then(Value::as_str)
        .and_then(normalize_string)
}

fn value_from_keys<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().find_map(|key| value.get(*key))
}

fn normalize_string(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

#[derive(Debug, Clone, Copy)]
struct RuntimeContractSpec {
    contract_key: &'static str,
    modality: &'static str,
    routing_slot: &'static str,
    required_capability: &'static str,
    execution_profile_key: &'static str,
    executor_adapter_key: Option<&'static str>,
    executor_binding_key: Option<&'static str>,
    route_execution_status: &'static str,
    route_execution_exit_condition: Option<&'static str>,
}

impl From<MediaRuntimeContractKind> for RuntimeContractSpec {
    fn from(kind: MediaRuntimeContractKind) -> Self {
        match kind {
            MediaRuntimeContractKind::ImageGeneration => Self {
                contract_key: IMAGE_GENERATION_CONTRACT_KEY,
                modality: "image",
                routing_slot: IMAGE_GENERATION_ROUTING_SLOT,
                required_capability: "image_generation",
                execution_profile_key: "image_generation_profile",
                executor_adapter_key: Some("app-server:media_task_artifact:image"),
                executor_binding_key: Some("mediaTaskArtifact/image/create"),
                route_execution_status: "executable",
                route_execution_exit_condition: None,
            },
            MediaRuntimeContractKind::VideoGeneration => Self {
                contract_key: VIDEO_GENERATION_CONTRACT_KEY,
                modality: "video",
                routing_slot: VIDEO_GENERATION_ROUTING_SLOT,
                required_capability: "video_generation",
                execution_profile_key: "video_generation_profile",
                executor_adapter_key: Some("app-server:media_task_artifact:video"),
                executor_binding_key: Some("mediaTaskArtifact/video/create"),
                route_execution_status: "executable",
                route_execution_exit_condition: None,
            },
            MediaRuntimeContractKind::VoiceGeneration => Self {
                contract_key: VOICE_GENERATION_CONTRACT_KEY,
                modality: "audio",
                routing_slot: VOICE_GENERATION_ROUTING_SLOT,
                required_capability: "voice_generation",
                execution_profile_key: "voice_generation_profile",
                executor_adapter_key: Some("app_server:mediaTaskArtifact/audio/create"),
                executor_binding_key: Some("mediaTaskArtifact/audio/create"),
                route_execution_status: "executable",
                route_execution_exit_condition: None,
            },
            MediaRuntimeContractKind::AudioTranscription => Self {
                contract_key: AUDIO_TRANSCRIPTION_CONTRACT_KEY,
                modality: "audio",
                routing_slot: AUDIO_TRANSCRIPTION_ROUTING_SLOT,
                required_capability: "audio_transcription",
                execution_profile_key: "audio_transcription_profile",
                executor_adapter_key: Some("app_server:mediaTaskArtifact/transcription/create"),
                executor_binding_key: Some("mediaTaskArtifact/transcription/create"),
                route_execution_status: "executable",
                route_execution_exit_condition: None,
            },
        }
    }
}

impl Serialize for RuntimeContractSpec {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        RuntimeContractValue {
            contract_key: self.contract_key,
            modality: self.modality,
            routing_slot: self.routing_slot,
            required_capabilities: [self.required_capability],
            execution_profile: ExecutionProfile {
                profile_key: self.execution_profile_key,
            },
            executor_adapter: self
                .executor_adapter_key
                .map(|adapter_key| ExecutorAdapter { adapter_key }),
            executor_binding: self
                .executor_binding_key
                .map(|binding_key| ExecutorBinding {
                    executor_kind: "app_server",
                    binding_key,
                }),
            route_execution_status: self.route_execution_status,
            route_execution_exit_condition: self.route_execution_exit_condition,
            limecore_policy_refs: ["model_catalog", "provider_offer", "tenant_feature_flags"],
        }
        .serialize(serializer)
    }
}

#[derive(Serialize)]
struct RuntimeContractValue {
    contract_key: &'static str,
    modality: &'static str,
    routing_slot: &'static str,
    required_capabilities: [&'static str; 1],
    execution_profile: ExecutionProfile,
    #[serde(skip_serializing_if = "Option::is_none")]
    executor_adapter: Option<ExecutorAdapter>,
    #[serde(skip_serializing_if = "Option::is_none")]
    executor_binding: Option<ExecutorBinding>,
    route_execution_status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    route_execution_exit_condition: Option<&'static str>,
    limecore_policy_refs: [&'static str; 3],
}

#[derive(Serialize)]
struct ExecutionProfile {
    profile_key: &'static str,
}

#[derive(Serialize)]
struct ExecutorAdapter {
    adapter_key: &'static str,
}

#[derive(Serialize)]
struct ExecutorBinding {
    executor_kind: &'static str,
    binding_key: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_image_contract_preserves_existing_json_shape() {
        let contract = runtime_contract_or_default(None, MediaRuntimeContractKind::ImageGeneration);

        assert_eq!(
            contract["contract_key"].as_str(),
            Some(IMAGE_GENERATION_CONTRACT_KEY)
        );
        assert_eq!(contract["modality"].as_str(), Some("image"));
        assert_eq!(
            contract["routing_slot"].as_str(),
            Some(IMAGE_GENERATION_ROUTING_SLOT)
        );
        assert_eq!(
            contract["required_capabilities"][0].as_str(),
            Some("image_generation")
        );
        assert_eq!(
            contract["execution_profile"]["profile_key"].as_str(),
            Some("image_generation_profile")
        );
        assert_eq!(
            contract["executor_adapter"]["adapter_key"].as_str(),
            Some("app-server:media_task_artifact:image")
        );
        assert_eq!(
            contract["executor_binding"]["executor_kind"].as_str(),
            Some("app_server")
        );
        assert_eq!(
            contract["executor_binding"]["binding_key"].as_str(),
            Some("mediaTaskArtifact/image/create")
        );
        assert_eq!(
            contract["limecore_policy_refs"].as_array().map(Vec::len),
            Some(3)
        );
        assert_eq!(
            contract["route_execution_status"].as_str(),
            Some("executable")
        );
    }

    #[test]
    fn default_voice_contract_is_executable() {
        let contract = runtime_contract_or_default(None, MediaRuntimeContractKind::VoiceGeneration);

        assert_eq!(
            contract["contract_key"].as_str(),
            Some(VOICE_GENERATION_CONTRACT_KEY)
        );
        assert_eq!(
            contract["routing_slot"].as_str(),
            Some(VOICE_GENERATION_ROUTING_SLOT)
        );
        assert_eq!(
            contract["route_execution_status"].as_str(),
            Some("executable")
        );
        assert_eq!(
            contract["executor_adapter"]["adapter_key"].as_str(),
            Some("app_server:mediaTaskArtifact/audio/create")
        );
        assert_eq!(
            contract["executor_binding"]["binding_key"].as_str(),
            Some("mediaTaskArtifact/audio/create")
        );
        assert!(contract.get("route_execution_exit_condition").is_none());
        assert!(contract.get("resolved_route").is_none());
        assert!(contract.get("model_route_execution").is_none());
    }

    #[test]
    fn override_contract_is_preserved() {
        let override_contract = json!({
            "contract_key": "custom_runtime",
            "modality": "image",
            "routing_slot": "custom_slot"
        });

        assert_eq!(
            runtime_contract_or_default(
                Some(&override_contract),
                MediaRuntimeContractKind::ImageGeneration
            ),
            override_contract
        );
    }

    #[test]
    fn projection_reads_top_level_and_contract_metadata() {
        let projection = runtime_contract_projection_from_payload(&json!({
            "modality_contract_key": " image_generation ",
            "runtime_contract": {
                "execution_profile": {
                    "profile_key": " image_generation_profile "
                },
                "executor_adapter": {
                    "adapter_key": "app-server:media_task_artifact:image"
                },
                "executor_binding": {
                    "executor_kind": "app_server",
                    "binding_key": "mediaTaskArtifact/image/create"
                },
                "limecore_policy_refs": [
                    "model_catalog",
                    "",
                    "provider_offer"
                ]
            }
        }));

        assert_eq!(
            projection.contract_key.as_deref(),
            Some(IMAGE_GENERATION_CONTRACT_KEY)
        );
        assert_eq!(
            projection.execution_profile_key.as_deref(),
            Some("image_generation_profile")
        );
        assert_eq!(
            projection.executor_adapter_key.as_deref(),
            Some("app-server:media_task_artifact:image")
        );
        assert_eq!(projection.executor_kind.as_deref(), Some("app_server"));
        assert_eq!(
            projection.executor_binding_key.as_deref(),
            Some("mediaTaskArtifact/image/create")
        );
        assert_eq!(
            projection.policy_refs,
            vec!["model_catalog".to_string(), "provider_offer".to_string()]
        );
    }

    #[test]
    fn projection_falls_back_to_camel_runtime_contract_key() {
        let projection = runtime_contract_projection_from_payload(&json!({
            "runtimeContract": {
                "contractKey": "custom_runtime",
                "executionProfile": {
                    "profileKey": "custom_profile"
                },
                "executorAdapter": {
                    "adapterKey": "custom_adapter"
                },
                "executorBinding": {
                    "executorKind": "custom_executor",
                    "bindingKey": "custom_binding"
                },
                "limecorePolicyRefs": ["tenant_feature_flags"]
            }
        }));

        assert_eq!(projection.contract_key.as_deref(), Some("custom_runtime"));
        assert_eq!(
            projection.execution_profile_key.as_deref(),
            Some("custom_profile")
        );
        assert_eq!(
            projection.executor_adapter_key.as_deref(),
            Some("custom_adapter")
        );
        assert_eq!(projection.executor_kind.as_deref(), Some("custom_executor"));
        assert_eq!(
            projection.executor_binding_key.as_deref(),
            Some("custom_binding")
        );
        assert_eq!(projection.policy_refs, vec!["tenant_feature_flags"]);
    }

    #[test]
    fn projection_suppresses_legacy_voice_executor_fields() {
        let projection = runtime_contract_projection_from_payload(&json!({
            "modality_contract_key": VOICE_GENERATION_CONTRACT_KEY,
            "runtime_contract": {
                "contract_key": VOICE_GENERATION_CONTRACT_KEY,
                "route_execution_status": "metadata_only",
                "execution_profile": {
                    "profile_key": "voice_generation_profile"
                },
                "executor_adapter": {
                    "adapter_key": "service_skill:voice_runtime"
                },
                "executor_binding": {
                    "executor_kind": "service_skill",
                    "binding_key": "voice_runtime"
                }
            }
        }));

        assert_eq!(
            projection.contract_key.as_deref(),
            Some(VOICE_GENERATION_CONTRACT_KEY)
        );
        assert_eq!(
            projection.execution_profile_key.as_deref(),
            Some("voice_generation_profile")
        );
        assert!(projection.executor_adapter_key.is_none());
        assert!(projection.executor_kind.is_none());
        assert!(projection.executor_binding_key.is_none());
    }
}
