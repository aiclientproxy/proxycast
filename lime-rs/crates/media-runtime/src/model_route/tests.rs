use super::*;
use serde_json::json;

#[test]
fn route_projection_reads_top_level_resolved_route() {
    let route = resolved_model_route_from_payload(&json!({
        "resolvedRoute": {
            "modelRef": {
                "providerId": "openai-images",
                "modelId": "gpt-image-2"
            },
            "protocol": "openai_images"
        }
    }))
    .expect("route");

    assert_eq!(route.provider_id.as_deref(), Some("openai-images"));
    assert_eq!(route.model_id.as_deref(), Some("gpt-image-2"));
    assert_eq!(route.protocol.as_deref(), Some("openai_images"));
}

#[test]
fn route_projection_reads_nested_assessment_route() {
    let route = resolved_model_route_from_payload(&json!({
        "model_route_assessment": {
            "resolvedRoute": {
                "modelRef": {
                    "providerId": "fal",
                    "modelId": "fal-ai/nano-banana-pro"
                },
                "protocol": "fal"
            }
        }
    }))
    .expect("route");

    assert_eq!(route.provider_id.as_deref(), Some("fal"));
    assert_eq!(route.model_id.as_deref(), Some("fal-ai/nano-banana-pro"));
    assert_eq!(
        image_executor_mode_from_route_protocol(route.protocol.as_deref()),
        Some("images_api")
    );
}

#[test]
fn route_failure_reads_capability_gap_message() {
    let failure = route_failure_from_payload(&json!({
        "failure_code": "capability_gap",
        "route_failure": {
            "category": "capability_gap",
            "reasonCode": "capability_gap",
            "capabilityGap": "task_family:image_generation"
        }
    }))
    .expect("failure");

    assert_eq!(failure.code, "capability_gap");
    assert!(failure.message.contains("task_family:image_generation"));
    assert!(!failure.retryable);
}

#[test]
fn protocol_support_rejects_chat_for_media_tasks() {
    assert!(supports_image_generation_route_protocol(Some(
        "openai_images"
    )));
    assert!(supports_image_generation_route_protocol(Some(
        "gemini_generate_content"
    )));
    assert!(supports_image_generation_route_protocol(Some(
        "dashscope_multimodal_generation"
    )));
    assert_eq!(
        image_executor_mode_from_route_protocol(Some("gemini_generate_content")),
        Some("gemini_generate_content")
    );
    assert_eq!(
        image_executor_mode_from_route_protocol(Some("dashscope_multimodal_generation")),
        Some("dashscope_images")
    );
    assert!(supports_video_generation_route_protocol(Some("fal")));
    assert!(supports_video_generation_route_protocol(Some("xai_video")));
    assert_eq!(
        video_executor_mode_from_route_protocol(Some("fal")),
        Some("fal_video_generation")
    );
    assert_eq!(
        video_executor_mode_from_route_protocol(Some("xai_video")),
        Some("xai_video_generation")
    );
    assert!(!supports_image_generation_route_protocol(Some(
        "openai_chat"
    )));
    assert!(!supports_video_generation_route_protocol(Some(
        "openai_chat"
    )));
}

#[test]
fn video_route_preflight_accepts_xai_video_protocol() {
    let preflight = video_route_payload_preflight(&json!({
        "resolvedRoute": {
            "modelRef": {
                "providerId": "xai-video",
                "modelId": "grok-imagine-video"
            },
            "protocol": "xai_video"
        }
    }));

    assert!(preflight.failure.is_none());
    assert_eq!(
        preflight
            .payload_patch
            .as_ref()
            .and_then(|patch| patch.pointer("/modelRouteExecution/route/protocol"))
            .and_then(Value::as_str),
        Some("xai_video")
    );
}

#[test]
fn route_execution_patch_binds_route_only_image_payload_to_media_worker() {
    let patch = route_execution_patch_from_payload(
        &json!({
            "resolvedRoute": {
                "modelRef": {
                    "providerId": "openai-images",
                    "modelId": "gpt-image-2"
                },
                "protocol": "openai_images",
                "auth": {
                    "credentialRef": "runtime-api-key-key-1",
                    "headerName": "X-Api-Key"
                }
            }
        }),
        &image_generation_execution_spec(),
    )
    .expect("patch");

    assert_eq!(
        patch["model_route_execution"]["executor"]["bindingKey"].as_str(),
        Some("mediaTaskArtifact/image/create")
    );
    assert_eq!(
        patch["model_route_execution"]["credentialResolver"]["secretMaterialStatus"].as_str(),
        Some("not_embedded")
    );
    assert_eq!(
        patch["model_route_execution"]["credentialResolver"]["credentialRef"].as_str(),
        Some("runtime-api-key-key-1")
    );
    assert_eq!(
        patch["model_route_execution"]["credentialResolver"]["authHeaderName"].as_str(),
        Some("X-Api-Key")
    );
    assert!(patch["model_route_execution"]["credentialResolver"]["authHeaderPrefix"].is_null());
}

#[test]
fn image_route_preflight_migrates_route_only_payload() {
    let preflight = image_route_payload_preflight(&json!({
        "resolvedRoute": {
            "modelRef": {
                "providerId": "openai-images",
                "modelId": "gpt-image-2"
            },
            "protocol": "openai_images"
        }
    }));

    assert!(preflight.failure.is_none());
    assert_eq!(
        preflight
            .payload_patch
            .as_ref()
            .and_then(|patch| patch.pointer("/modelRouteExecution/executor/kind"))
            .and_then(Value::as_str),
        Some("media_task_worker")
    );
}

#[test]
fn video_route_preflight_rejects_chat_protocol() {
    let preflight = video_route_payload_preflight(&json!({
        "resolvedRoute": {
            "modelRef": {
                "providerId": "openai",
                "modelId": "gpt-4.1"
            },
            "protocol": "openai_chat"
        }
    }));

    assert!(preflight.payload_patch.is_none());
    assert_eq!(
        preflight
            .failure
            .as_ref()
            .map(|failure| failure.code.as_str()),
        Some("unsupported_protocol")
    );
}

#[test]
fn route_execution_validation_rejects_embedded_secret() {
    let failure = route_execution_failure_from_payload(
        &json!({
            "resolvedRoute": {
                "modelRef": {
                    "providerId": "openai-images",
                    "modelId": "gpt-image-2"
                },
                "protocol": "openai_images"
            },
            "modelRouteExecution": {
                "executor": {
                    "kind": "media_task_worker",
                    "bindingKey": "mediaTaskArtifact/image/create",
                    "endpointSource": "resolved_route"
                },
                "credentialResolver": {
                    "owner": "media_task_worker",
                    "secretMaterialStatus": "embedded",
                    "apiKey": "sk-test"
                },
                "route": {
                    "providerId": "openai-images",
                    "modelId": "gpt-image-2",
                    "protocol": "openai_images"
                }
            }
        }),
        &image_generation_execution_spec(),
    )
    .expect("failure");

    assert_eq!(failure.code, "unsupported_route_execution");
    assert!(failure.message.contains("明文凭证"));
}
