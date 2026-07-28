use std::sync::{Arc, Mutex};

use axum::{
    extract::Json,
    http::{HeaderMap, StatusCode},
    routing::post,
    Router,
};
use lime_media_runtime::{
    execute_image_generation_task, execute_video_generation_task, load_task_output,
    write_task_artifact, ImageGenerationRunnerConfig, TaskType, TaskWriteOptions,
    VideoGenerationRunnerConfig,
};
use serde_json::{json, Value};
use tokio::net::TcpListener;

#[tokio::test]
async fn image_worker_uses_resolved_route_provider_and_model() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let captured_provider_id = Arc::new(Mutex::new(None::<String>));
    let captured_model = Arc::new(Mutex::new(None::<String>));
    let created = write_task_artifact(
        temp_dir.path(),
        TaskType::ImageGenerate,
        Some("青柠主视觉".to_string()),
        json!({
            "prompt": "未来感青柠实验室",
            "size": "1024x1024",
            "count": 1,
            "provider_id": "legacy-provider",
            "model": "legacy-image-model",
            "resolvedRoute": {
                "modelRef": {
                    "providerId": "route-provider",
                    "modelId": "route-image-model"
                },
                "protocol": "openai_images"
            }
        }),
        TaskWriteOptions::default(),
    )
    .expect("create task");

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind image api");
    let address = listener.local_addr().expect("resolve address");
    let captured_provider_id_for_server = Arc::clone(&captured_provider_id);
    let captured_model_for_server = Arc::clone(&captured_model);
    let server = tokio::spawn(async move {
        let app = Router::new().route(
            "/v1/images/generations",
            post(move |headers: HeaderMap, Json(body): Json<Value>| {
                let captured_provider_id = Arc::clone(&captured_provider_id_for_server);
                let captured_model = Arc::clone(&captured_model_for_server);
                async move {
                    *captured_provider_id.lock().expect("lock provider id") = headers
                        .get("x-provider-id")
                        .and_then(|value| value.to_str().ok())
                        .map(ToOwned::to_owned);
                    *captured_model.lock().expect("lock model") = body
                        .get("model")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned);
                    (
                        StatusCode::OK,
                        Json(json!({
                            "created": 1_717_200_000i64,
                            "data": [
                                {
                                    "b64_json": "ZmFrZS1saW1lLWltYWdl",
                                    "revised_prompt": "未来感青柠实验室主视觉"
                                }
                            ]
                        })),
                    )
                }
            }),
        );
        axum::serve(listener, app).await.expect("serve image api");
    });

    let result = execute_image_generation_task(
        temp_dir.path(),
        &created.task_id,
        &ImageGenerationRunnerConfig {
            endpoint: format!("http://{address}/v1/images/generations"),
            api_key: "test-key".to_string(),
            request_body_format: Default::default(),
        },
    )
    .await
    .expect("execute image task");

    assert_eq!(result.normalized_status, "succeeded");
    assert_eq!(
        captured_provider_id
            .lock()
            .expect("lock provider id")
            .as_deref(),
        Some("route-provider")
    );
    assert_eq!(
        captured_model.lock().expect("lock model").as_deref(),
        Some("route-image-model")
    );
    let migrated = load_task_output(temp_dir.path(), &created.task_id, None)
        .expect("load migrated image task");
    assert_eq!(
        migrated
            .record
            .payload
            .pointer("/modelRouteExecution/executor/kind")
            .and_then(Value::as_str),
        Some("media_task_worker")
    );
    assert_eq!(
        migrated
            .record
            .payload
            .pointer("/modelRouteExecution/credentialResolver/secretMaterialStatus")
            .and_then(Value::as_str),
        Some("not_embedded")
    );
    assert_route_runtime_evidence(
        &migrated.record.payload,
        "image_generation",
        "route-provider",
        "route-image-model",
        "openai_images",
    );

    server.abort();
}

#[tokio::test]
async fn image_worker_uses_nested_route_execution_without_route_only_migration() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let captured_provider_id = Arc::new(Mutex::new(None::<String>));
    let captured_model = Arc::new(Mutex::new(None::<String>));
    let created = write_task_artifact(
        temp_dir.path(),
        TaskType::ImageGenerate,
        Some("青柠主视觉".to_string()),
        json!({
            "prompt": "未来感青柠实验室",
            "size": "1024x1024",
            "provider_id": "legacy-provider",
            "model": "legacy-image-model",
            "modelRouteAssessment": {
                "status": "accepted",
                "resolvedRoute": {
                    "modelRef": {
                        "providerId": "route-provider",
                        "modelId": "route-image-model"
                    },
                    "protocol": "openai_images"
                },
                "routeExecution": {
                    "executor": {
                        "kind": "media_task_worker",
                        "bindingKey": "mediaTaskArtifact/image/create",
                        "endpointSource": "resolved_route"
                    },
                    "credentialResolver": {
                        "owner": "media_task_worker",
                        "secretMaterialStatus": "not_embedded"
                    },
                    "route": {
                        "providerId": "route-provider",
                        "modelId": "route-image-model",
                        "protocol": "openai_images"
                    }
                }
            }
        }),
        TaskWriteOptions::default(),
    )
    .expect("create task");

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind image api");
    let address = listener.local_addr().expect("resolve address");
    let captured_provider_id_for_server = Arc::clone(&captured_provider_id);
    let captured_model_for_server = Arc::clone(&captured_model);
    let server = tokio::spawn(async move {
        let app = Router::new().route(
            "/v1/images/generations",
            post(move |headers: HeaderMap, Json(body): Json<Value>| {
                let captured_provider_id = Arc::clone(&captured_provider_id_for_server);
                let captured_model = Arc::clone(&captured_model_for_server);
                async move {
                    *captured_provider_id.lock().expect("lock provider id") = headers
                        .get("x-provider-id")
                        .and_then(|value| value.to_str().ok())
                        .map(ToOwned::to_owned);
                    *captured_model.lock().expect("lock model") = body
                        .get("model")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned);
                    (
                        StatusCode::OK,
                        Json(json!({
                            "created": 1_717_200_000i64,
                            "data": [
                                {
                                    "b64_json": "ZmFrZS1saW1lLWltYWdl",
                                    "revised_prompt": "未来感青柠实验室主视觉"
                                }
                            ]
                        })),
                    )
                }
            }),
        );
        axum::serve(listener, app).await.expect("serve image api");
    });

    let result = execute_image_generation_task(
        temp_dir.path(),
        &created.task_id,
        &ImageGenerationRunnerConfig {
            endpoint: format!("http://{address}/v1/images/generations"),
            api_key: "test-key".to_string(),
            request_body_format: Default::default(),
        },
    )
    .await
    .expect("execute image task");

    assert_eq!(result.normalized_status, "succeeded");
    assert_eq!(
        captured_provider_id
            .lock()
            .expect("lock provider id")
            .as_deref(),
        Some("route-provider")
    );
    assert_eq!(
        captured_model.lock().expect("lock model").as_deref(),
        Some("route-image-model")
    );
    let persisted = load_task_output(temp_dir.path(), &created.task_id, None)
        .expect("load persisted image task");
    assert!(persisted
        .record
        .payload
        .pointer("/modelRouteExecution")
        .is_none());
    assert_eq!(
        persisted
            .record
            .payload
            .pointer("/modelRouteAssessment/routeExecution/executor/kind")
            .and_then(Value::as_str),
        Some("media_task_worker")
    );

    server.abort();
}

#[tokio::test]
async fn video_worker_uses_resolved_route_provider_and_model() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let captured_provider_id = Arc::new(Mutex::new(None::<String>));
    let captured_body = Arc::new(Mutex::new(None::<Value>));
    let created = write_task_artifact(
        temp_dir.path(),
        TaskType::VideoGenerate,
        Some("短视频".to_string()),
        json!({
            "prompt": "生成一段青柠实验室短视频",
            "provider_id": "legacy-video-provider",
            "model": "legacy-video-model",
            "aspect_ratio": "16:9",
            "resolvedRoute": {
                "modelRef": {
                    "providerId": "route-video-provider",
                    "modelId": "route-video-model"
                },
                "protocol": "fal"
            }
        }),
        TaskWriteOptions {
            status: Some("pending_submit".to_string()),
            ..TaskWriteOptions::default()
        },
    )
    .expect("create video task");

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind video api");
    let address = listener.local_addr().expect("resolve address");
    let captured_provider_id_for_server = Arc::clone(&captured_provider_id);
    let captured_body_for_server = Arc::clone(&captured_body);
    let server = tokio::spawn(async move {
        let app = Router::new().route(
            "/v1/videos/generations",
            post(move |headers: HeaderMap, Json(body): Json<Value>| {
                let captured_provider_id = Arc::clone(&captured_provider_id_for_server);
                let captured_body = Arc::clone(&captured_body_for_server);
                async move {
                    *captured_provider_id.lock().expect("lock provider id") = headers
                        .get("x-provider-id")
                        .and_then(|value| value.to_str().ok())
                        .map(ToOwned::to_owned);
                    *captured_body.lock().expect("lock body") = Some(body);
                    (
                        StatusCode::OK,
                        Json(json!({
                            "id": "video-job-1",
                            "data": [
                                {
                                    "id": "generated-video-1",
                                    "url": "https://cdn.example.test/generated.mp4",
                                    "mime_type": "video/mp4",
                                    "duration": 8
                                }
                            ]
                        })),
                    )
                }
            }),
        );
        axum::serve(listener, app).await.expect("serve video api");
    });

    let result = execute_video_generation_task(
        temp_dir.path(),
        &created.task_id,
        &VideoGenerationRunnerConfig::fal(
            format!("http://{address}/v1/videos/generations"),
            "test-key".to_string(),
            "Authorization".to_string(),
            Some("Bearer".to_string()),
        ),
    )
    .await
    .expect("execute video task");

    assert_eq!(result.normalized_status, "succeeded");
    assert_eq!(
        captured_provider_id
            .lock()
            .expect("lock provider id")
            .as_deref(),
        Some("route-video-provider")
    );
    let body = captured_body
        .lock()
        .expect("lock body")
        .clone()
        .expect("captured body");
    assert_eq!(body.pointer("/model"), Some(&json!("route-video-model")));
    assert_eq!(
        body.pointer("/provider_id"),
        Some(&json!("route-video-provider"))
    );
    let migrated = load_task_output(temp_dir.path(), &created.task_id, None)
        .expect("load migrated video task");
    assert_eq!(
        migrated
            .record
            .payload
            .pointer("/modelRouteExecution/executor/bindingKey")
            .and_then(Value::as_str),
        Some("mediaTaskArtifact/video/create")
    );
    assert_route_runtime_evidence(
        &migrated.record.payload,
        "video_generation",
        "route-video-provider",
        "route-video-model",
        "fal",
    );

    server.abort();
}

fn assert_route_runtime_evidence(
    payload: &Value,
    task_family: &str,
    provider_id: &str,
    model_id: &str,
    protocol: &str,
) {
    assert_eq!(
        payload
            .pointer("/llm_events/1/type")
            .and_then(Value::as_str),
        Some("turn.completed")
    );
    assert_eq!(
        payload.pointer("/llmEvents/1/type").and_then(Value::as_str),
        Some("turn.completed")
    );

    for diagnostics_root in ["/provider_diagnostics", "/providerDiagnostics"] {
        assert_eq!(
            payload
                .pointer(&format!("{diagnostics_root}/taskFamily"))
                .and_then(Value::as_str),
            Some(task_family)
        );
        assert_eq!(
            payload
                .pointer(&format!("{diagnostics_root}/providerId"))
                .and_then(Value::as_str),
            Some(provider_id)
        );
        assert_eq!(
            payload
                .pointer(&format!("{diagnostics_root}/modelId"))
                .and_then(Value::as_str),
            Some(model_id)
        );
        assert_eq!(
            payload
                .pointer(&format!("{diagnostics_root}/protocol"))
                .and_then(Value::as_str),
            Some(protocol)
        );
        assert_eq!(
            payload
                .pointer(&format!("{diagnostics_root}/transport"))
                .and_then(Value::as_str),
            Some("provider_http")
        );
        assert_eq!(
            payload
                .pointer(&format!("{diagnostics_root}/credential"))
                .and_then(Value::as_str),
            Some("not_embedded")
        );
    }
}

#[tokio::test]
async fn image_worker_fails_closed_on_capability_gap() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let captured_updates = Arc::new(Mutex::new(Vec::<String>::new()));
    let created = write_task_artifact(
        temp_dir.path(),
        TaskType::ImageGenerate,
        Some("青柠主视觉".to_string()),
        json!({
            "prompt": "未来感青柠实验室",
            "size": "1024x1024",
            "count": 1,
            "provider_id": "openai",
            "model": "text-only",
            "failure_code": "capability_gap",
            "capability_gap": "task_family:image_generation",
            "route_failure": {
                "category": "capability_gap",
                "reasonCode": "capability_gap",
                "capabilityGap": "task_family:image_generation"
            }
        }),
        TaskWriteOptions::default(),
    )
    .expect("create task");

    let updates_for_hook = Arc::clone(&captured_updates);
    let result = lime_media_runtime::execute_image_generation_task_with_hook(
        temp_dir.path(),
        &created.task_id,
        &ImageGenerationRunnerConfig {
            endpoint: "http://127.0.0.1:1/v1/images/generations".to_string(),
            api_key: "test-key".to_string(),
            request_body_format: Default::default(),
        },
        move |output| {
            updates_for_hook
                .lock()
                .expect("lock updates")
                .push(output.normalized_status.clone());
        },
    )
    .await
    .expect("image task should fail closed");

    assert_eq!(result.normalized_status, "failed");
    assert_eq!(
        result.last_error.as_ref().map(|value| value.code.as_str()),
        Some("capability_gap")
    );
    assert_eq!(
        captured_updates.lock().expect("lock updates").as_slice(),
        ["queued", "failed"]
    );
}

#[tokio::test]
async fn video_worker_fails_closed_on_capability_gap() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let captured_updates = Arc::new(Mutex::new(Vec::<String>::new()));
    let created = write_task_artifact(
        temp_dir.path(),
        TaskType::VideoGenerate,
        Some("短视频".to_string()),
        json!({
            "prompt": "生成一段青柠实验室短视频",
            "provider_id": "openai",
            "model": "text-only",
            "failure_code": "capability_gap",
            "capability_gap": "task_family:video_generation",
            "route_failure": {
                "category": "capability_gap",
                "reasonCode": "capability_gap",
                "capabilityGap": "task_family:video_generation"
            }
        }),
        TaskWriteOptions {
            status: Some("pending_submit".to_string()),
            ..TaskWriteOptions::default()
        },
    )
    .expect("create video task");

    let updates_for_hook = Arc::clone(&captured_updates);
    let result = lime_media_runtime::execute_video_generation_task_with_hook(
        temp_dir.path(),
        &created.task_id,
        &VideoGenerationRunnerConfig::fal(
            "http://127.0.0.1:1/v1/videos/generations".to_string(),
            "test-key".to_string(),
            "Authorization".to_string(),
            Some("Bearer".to_string()),
        ),
        move |output| {
            updates_for_hook
                .lock()
                .expect("lock updates")
                .push(output.normalized_status.clone());
        },
    )
    .await
    .expect("video task should fail closed");

    assert_eq!(result.normalized_status, "failed");
    assert_eq!(
        result.last_error.as_ref().map(|value| value.code.as_str()),
        Some("capability_gap")
    );
    assert_eq!(
        captured_updates.lock().expect("lock updates").as_slice(),
        ["queued", "failed"]
    );
}

#[tokio::test]
async fn image_worker_fails_closed_on_embedded_route_secret() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let created = write_task_artifact(
        temp_dir.path(),
        TaskType::ImageGenerate,
        Some("青柠主视觉".to_string()),
        json!({
            "prompt": "未来感青柠实验室",
            "size": "1024x1024",
            "provider_id": "openai",
            "model": "gpt-image-2",
            "resolvedRoute": {
                "modelRef": {
                    "providerId": "openai",
                    "modelId": "gpt-image-2"
                },
                "protocol": "openai_images"
            },
            "modelRouteExecution": {
                "executor": {
                    "kind": "direct_provider",
                    "bindingKey": "direct:https://api.openai.com/v1/images/generations",
                    "endpointSource": "resolved_route"
                },
                "credentialResolver": {
                    "owner": "media_runtime_worker",
                    "secretMaterialStatus": "embedded",
                    "apiKey": "sk-test"
                },
                "route": {
                    "providerId": "openai",
                    "modelId": "gpt-image-2",
                    "protocol": "openai_images"
                }
            }
        }),
        TaskWriteOptions::default(),
    )
    .expect("create task");

    let result = execute_image_generation_task(
        temp_dir.path(),
        &created.task_id,
        &ImageGenerationRunnerConfig {
            endpoint: "http://127.0.0.1:1/v1/images/generations".to_string(),
            api_key: "test-key".to_string(),
            request_body_format: Default::default(),
        },
    )
    .await
    .expect("execute image task");

    assert_eq!(result.normalized_status, "failed");
    assert_eq!(
        result.last_error.as_ref().map(|value| value.code.as_str()),
        Some("unsupported_route_execution")
    );
}

#[tokio::test]
async fn video_worker_fails_closed_on_non_local_route_execution() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let created = write_task_artifact(
        temp_dir.path(),
        TaskType::VideoGenerate,
        Some("短视频".to_string()),
        json!({
            "prompt": "生成一段青柠实验室短视频",
            "provider_id": "fal",
            "model": "fal-ai/video",
            "resolvedRoute": {
                "modelRef": {
                    "providerId": "fal",
                    "modelId": "fal-ai/video"
                },
                "protocol": "fal"
            },
            "modelRouteExecution": {
                "executor": {
                    "kind": "direct_provider",
                    "bindingKey": "direct:https://fal.run",
                    "endpointSource": "resolved_route"
                },
                "credentialResolver": {
                    "owner": "media_runtime_worker",
                    "secretMaterialStatus": "not_embedded"
                },
                "route": {
                    "providerId": "fal",
                    "modelId": "fal-ai/video",
                    "protocol": "fal"
                }
            }
        }),
        TaskWriteOptions {
            status: Some("pending_submit".to_string()),
            ..TaskWriteOptions::default()
        },
    )
    .expect("create video task");

    let result = execute_video_generation_task(
        temp_dir.path(),
        &created.task_id,
        &VideoGenerationRunnerConfig::fal(
            "http://127.0.0.1:1/v1/videos/generations".to_string(),
            "test-key".to_string(),
            "Authorization".to_string(),
            Some("Bearer".to_string()),
        ),
    )
    .await
    .expect("execute video task");

    assert_eq!(result.normalized_status, "failed");
    assert_eq!(
        result.last_error.as_ref().map(|value| value.code.as_str()),
        Some("unsupported_route_execution")
    );
}

#[tokio::test]
async fn image_worker_fails_closed_on_unsupported_route_protocol() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let created = write_task_artifact(
        temp_dir.path(),
        TaskType::ImageGenerate,
        Some("青柠主视觉".to_string()),
        json!({
            "prompt": "未来感青柠实验室",
            "size": "1024x1024",
            "provider_id": "openai",
            "model": "gpt-4.1",
            "resolvedRoute": {
                "modelRef": {
                    "providerId": "openai",
                    "modelId": "gpt-4.1"
                },
                "protocol": "openai_chat"
            }
        }),
        TaskWriteOptions::default(),
    )
    .expect("create task");

    let result = execute_image_generation_task(
        temp_dir.path(),
        &created.task_id,
        &ImageGenerationRunnerConfig {
            endpoint: "http://127.0.0.1:1/v1/images/generations".to_string(),
            api_key: "test-key".to_string(),
            request_body_format: Default::default(),
        },
    )
    .await
    .expect("execute image task");

    assert_eq!(result.normalized_status, "failed");
    assert_eq!(
        result.last_error.as_ref().map(|value| value.code.as_str()),
        Some("unsupported_protocol")
    );
}

#[tokio::test]
async fn video_worker_fails_closed_on_unsupported_route_protocol() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let created = write_task_artifact(
        temp_dir.path(),
        TaskType::VideoGenerate,
        Some("短视频".to_string()),
        json!({
            "prompt": "生成一段青柠实验室短视频",
            "provider_id": "openai",
            "model": "gpt-4.1",
            "resolvedRoute": {
                "modelRef": {
                    "providerId": "openai",
                    "modelId": "gpt-4.1"
                },
                "protocol": "openai_chat"
            }
        }),
        TaskWriteOptions {
            status: Some("pending_submit".to_string()),
            ..TaskWriteOptions::default()
        },
    )
    .expect("create video task");

    let result = execute_video_generation_task(
        temp_dir.path(),
        &created.task_id,
        &VideoGenerationRunnerConfig::fal(
            "http://127.0.0.1:1/v1/videos/generations".to_string(),
            "test-key".to_string(),
            "Authorization".to_string(),
            Some("Bearer".to_string()),
        ),
    )
    .await
    .expect("execute video task");

    assert_eq!(result.normalized_status, "failed");
    assert_eq!(
        result.last_error.as_ref().map(|value| value.code.as_str()),
        Some("unsupported_protocol")
    );
}
