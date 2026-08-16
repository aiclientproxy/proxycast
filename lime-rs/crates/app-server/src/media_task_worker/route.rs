use super::MediaTaskWorkerContext;
use lime_core::models::{runtime_api_key_id_from_credential_uuid, RuntimeCredentialData};
use lime_media_runtime::{
    patch_task_artifact, ImageGenerationRequestBodyFormat, ImageGenerationRunnerConfig,
    TaskArtifactPatch, VideoGenerationRunnerConfig, IMAGE_TASK_RUNNER_WORKER_ID,
    VIDEO_TASK_RUNNER_WORKER_ID,
};
use lime_services::api_key_provider_service::ApiKeyProviderService;
use model_provider::audio::AudioProviderConfig;
use serde_json::Value;
use std::path::Path;

pub(super) fn image_generation_runner_config_from_resolved_route(
    workspace_root: &Path,
    task_id: &str,
    context: &MediaTaskWorkerContext,
) -> Result<Option<ImageGenerationRunnerConfig>, String> {
    let task = lime_media_runtime::load_task_output(workspace_root, task_id, None)
        .map_err(|error| error.to_string())?;
    let Some(route) = task.record.payload.get("resolved_route") else {
        return Ok(None);
    };
    if route_failure_present(&task.record.payload) {
        return Ok(None);
    }
    let Some(provider_id) = route_model_ref_string(route, &["providerId", "provider_id"]) else {
        return Ok(None);
    };
    let Some(model_id) = route_model_ref_string(route, &["modelId", "model_id"]) else {
        return Ok(None);
    };
    let Some(protocol) = read_value_string(route, &["protocol"]) else {
        return Ok(None);
    };
    let Some(endpoint) = image_generation_endpoint_from_route(route, &protocol) else {
        return Ok(None);
    };
    let request_body_format = image_request_body_format_from_route(route, &protocol);
    let api_key_service = ApiKeyProviderService::new();
    let (key_id, api_key) =
        image_api_key_from_resolved_route(route, &context.db, &api_key_service, &provider_id)?;
    record_credential_usage(
        &context.db,
        &api_key_service,
        &provider_id,
        key_id.as_deref(),
    );

    patch_task_artifact(
        workspace_root,
        task_id,
        None,
        TaskArtifactPatch {
            payload_patch: Some(serde_json::json!({
                "executor_mode": image_executor_mode_from_route(route, &protocol),
                "provider_id": provider_id,
                "model": model_id,
                "request_body_format": request_body_format.as_str(),
            })),
            current_attempt_worker_id: Some(Some(IMAGE_TASK_RUNNER_WORKER_ID.to_string())),
            ..TaskArtifactPatch::default()
        },
    )
    .map_err(|error| error.to_string())?;

    Ok(Some(ImageGenerationRunnerConfig {
        endpoint,
        api_key,
        request_body_format,
    }))
}

pub(super) struct AudioSpeechRunnerConfig {
    pub(super) provider: AudioProviderConfig,
    pub(super) provider_id: String,
    pub(super) model_id: String,
}

pub(super) struct AudioTranscriptionRunnerConfig {
    pub(super) provider: AudioProviderConfig,
    pub(super) provider_id: String,
    pub(super) model_id: String,
}

pub(super) fn audio_transcription_runner_config_from_resolved_route(
    workspace_root: &Path,
    task_id: &str,
    context: &MediaTaskWorkerContext,
) -> Result<Option<AudioTranscriptionRunnerConfig>, String> {
    let task = lime_media_runtime::load_task_output(workspace_root, task_id, None)
        .map_err(|error| error.to_string())?;
    let Some(route) = task.record.payload.get("resolved_route") else {
        return Ok(None);
    };
    if route_failure_present(&task.record.payload) {
        return Ok(None);
    }
    let Some(provider_id) = route_model_ref_string(route, &["providerId", "provider_id"]) else {
        return Ok(None);
    };
    let Some(model_id) = route_model_ref_string(route, &["modelId", "model_id"]) else {
        return Ok(None);
    };
    let protocol = read_value_string(route, &["protocol"]);
    if protocol.as_deref() != Some("openai_audio_transcription") {
        return Ok(None);
    }
    let Some(base_url) = route
        .get("endpoint")
        .and_then(|endpoint| read_value_string(endpoint, &["baseUrl", "base_url"]))
    else {
        return Ok(None);
    };
    let api_key_service = ApiKeyProviderService::new();
    let credential = route_credential_from_resolved_route(
        route,
        &context.db,
        &api_key_service,
        &provider_id,
        "音频转写",
    )?;
    record_credential_usage(
        &context.db,
        &api_key_service,
        &provider_id,
        credential.key_id.as_deref(),
    );
    let endpoint = audio_transcription_endpoint_from_base(&base_url);
    patch_task_artifact(
        workspace_root,
        task_id,
        None,
        TaskArtifactPatch {
            payload_patch: Some(serde_json::json!({
                "provider_id": provider_id,
                "model": model_id,
                "executor_mode": "openai_audio_transcription",
                "audio_transcription_endpoint": endpoint,
            })),
            ..TaskArtifactPatch::default()
        },
    )
    .map_err(|error| error.to_string())?;
    Ok(Some(AudioTranscriptionRunnerConfig {
        provider: AudioProviderConfig {
            endpoint,
            api_key: credential.api_key,
            auth_header: credential.auth_header,
            auth_prefix: credential.auth_prefix,
            timeout: std::time::Duration::from_secs(300),
        },
        provider_id,
        model_id,
    }))
}

fn audio_transcription_endpoint_from_base(base_url: &str) -> String {
    let normalized = base_url.trim().trim_end_matches('/');
    if normalized.ends_with("/audio/transcriptions") {
        normalized.to_string()
    } else if normalized.ends_with("/v1") {
        format!("{normalized}/audio/transcriptions")
    } else {
        format!("{normalized}/v1/audio/transcriptions")
    }
}

pub(super) fn audio_speech_runner_config_from_resolved_route(
    workspace_root: &Path,
    task_id: &str,
    context: &MediaTaskWorkerContext,
) -> Result<Option<AudioSpeechRunnerConfig>, String> {
    let task = lime_media_runtime::load_task_output(workspace_root, task_id, None)
        .map_err(|error| error.to_string())?;
    let Some(route) = task.record.payload.get("resolved_route") else {
        return Ok(None);
    };
    if route_failure_present(&task.record.payload) {
        return Ok(None);
    }
    let Some(provider_id) = route_model_ref_string(route, &["providerId", "provider_id"]) else {
        return Ok(None);
    };
    let Some(model_id) = route_model_ref_string(route, &["modelId", "model_id"]) else {
        return Ok(None);
    };
    let protocol = read_value_string(route, &["protocol"]);
    if protocol.as_deref() != Some("openai_audio_speech") {
        return Ok(None);
    }
    let Some(base_url) = route
        .get("endpoint")
        .and_then(|endpoint| read_value_string(endpoint, &["baseUrl", "base_url"]))
    else {
        return Ok(None);
    };
    let credential = route_credential_from_resolved_route(
        route,
        &context.db,
        &ApiKeyProviderService::new(),
        &provider_id,
        "语音合成",
    )?;
    let endpoint = audio_speech_endpoint_from_base(&base_url);
    let api_key = credential.api_key;
    let auth_header = credential.auth_header;
    let auth_prefix = credential.auth_prefix;
    let api_key_service = ApiKeyProviderService::new();
    record_credential_usage(
        &context.db,
        &api_key_service,
        &provider_id,
        credential.key_id.as_deref(),
    );
    patch_task_artifact(
        workspace_root,
        task_id,
        None,
        TaskArtifactPatch {
            payload_patch: Some(serde_json::json!({
                "provider_id": provider_id,
                "model": model_id,
                "executor_mode": "openai_audio_speech",
                "audio_endpoint": endpoint,
            })),
            ..TaskArtifactPatch::default()
        },
    )
    .map_err(|error| error.to_string())?;
    Ok(Some(AudioSpeechRunnerConfig {
        provider: AudioProviderConfig {
            endpoint,
            api_key,
            auth_header,
            auth_prefix,
            timeout: std::time::Duration::from_secs(300),
        },
        provider_id,
        model_id,
    }))
}

fn audio_speech_endpoint_from_base(base_url: &str) -> String {
    let normalized = base_url.trim().trim_end_matches('/');
    if normalized.ends_with("/audio/speech") {
        normalized.to_string()
    } else if normalized.ends_with("/v1") {
        format!("{normalized}/audio/speech")
    } else {
        format!("{normalized}/v1/audio/speech")
    }
}

pub(super) fn video_generation_runner_config_from_resolved_route(
    workspace_root: &Path,
    task_id: &str,
    context: &MediaTaskWorkerContext,
) -> Result<Option<VideoGenerationRunnerConfig>, String> {
    let task = lime_media_runtime::load_task_output(workspace_root, task_id, None)
        .map_err(|error| error.to_string())?;
    let Some(route) = task.record.payload.get("resolved_route") else {
        return Ok(None);
    };
    if route_failure_present(&task.record.payload) {
        return Ok(None);
    }
    let Some(provider_id) = route_model_ref_string(route, &["providerId", "provider_id"]) else {
        return Ok(None);
    };
    let Some(model_id) = route_model_ref_string(route, &["modelId", "model_id"]) else {
        return Ok(None);
    };
    let Some(protocol) = read_value_string(route, &["protocol"]) else {
        return Ok(None);
    };
    let Some(endpoint) = video_generation_endpoint_from_route(route, &protocol, &model_id) else {
        return Ok(None);
    };
    let api_key_service = ApiKeyProviderService::new();
    let credential = route_credential_from_resolved_route(
        route,
        &context.db,
        &api_key_service,
        &provider_id,
        "视频",
    )?;
    record_credential_usage(
        &context.db,
        &api_key_service,
        &provider_id,
        credential.key_id.as_deref(),
    );

    let executor_mode = match protocol.as_str() {
        "fal" => "fal_video_generation",
        "xai_video" => "xai_video_generation",
        _ => return Ok(None),
    };
    patch_task_artifact(
        workspace_root,
        task_id,
        None,
        TaskArtifactPatch {
            payload_patch: Some(serde_json::json!({
                "executor_mode": executor_mode,
                "provider_id": provider_id,
                "model": model_id,
            })),
            current_attempt_worker_id: Some(Some(VIDEO_TASK_RUNNER_WORKER_ID.to_string())),
            ..TaskArtifactPatch::default()
        },
    )
    .map_err(|error| error.to_string())?;

    Ok(Some(match protocol.as_str() {
        "fal" => VideoGenerationRunnerConfig::fal(
            endpoint,
            credential.api_key,
            credential.auth_header,
            credential.auth_prefix,
        ),
        "xai_video" => VideoGenerationRunnerConfig::xai(
            endpoint,
            credential.api_key,
            credential.auth_header,
            credential.auth_prefix,
        ),
        _ => unreachable!("video protocol was admitted above"),
    }))
}

fn image_api_key_from_resolved_route(
    route: &Value,
    db: &lime_core::database::DbConnection,
    api_key_service: &ApiKeyProviderService,
    provider_id: &str,
) -> Result<(Option<String>, String), String> {
    let credential =
        route_credential_from_resolved_route(route, db, api_key_service, provider_id, "图片")?;
    Ok((credential.key_id, credential.api_key))
}

struct ResolvedRouteCredential {
    key_id: Option<String>,
    api_key: String,
    auth_header: String,
    auth_prefix: Option<String>,
}

fn route_credential_from_resolved_route(
    route: &Value,
    db: &lime_core::database::DbConnection,
    api_key_service: &ApiKeyProviderService,
    provider_id: &str,
    task_label: &str,
) -> Result<ResolvedRouteCredential, String> {
    let auth = route.get("auth").ok_or_else(|| {
        format!("{task_label} Provider {provider_id} 的 resolved route 缺少 auth")
    })?;
    let auth_header = read_value_string(auth, &["headerName", "header_name"])
        .unwrap_or_else(|| "Authorization".to_string());
    let auth_prefix = read_value_string(auth, &["headerPrefix", "header_prefix"]);
    let credential_ref = read_value_string(auth, &["credentialRef", "credential_ref"]);
    if credential_ref.is_none() && read_value_string(auth, &["kind"]).as_deref() == Some("no_auth")
    {
        return Ok(ResolvedRouteCredential {
            key_id: None,
            api_key: String::new(),
            auth_header,
            auth_prefix,
        });
    }
    let credential_ref = credential_ref.ok_or_else(|| {
        format!("{task_label} Provider {provider_id} 的 resolved route 缺少 credentialRef")
    })?;
    let credential = api_key_service
        .select_runtime_credential_by_ref(db, provider_id, &credential_ref)
        .map_err(|error| format!("读取{task_label} Provider 精确凭证失败: {error}"))?
        .ok_or_else(|| {
            format!("{task_label} Provider {provider_id} 的 resolved credential 不可用")
        })?;
    let key_id = runtime_api_key_id_from_credential_uuid(&credential.uuid)
        .ok_or_else(|| format!("{task_label} Provider resolved credentialRef 格式无效"))?
        .to_string();
    let api_key = match credential.credential {
        RuntimeCredentialData::OpenAIKey { api_key, .. }
        | RuntimeCredentialData::ClaudeKey { api_key, .. }
        | RuntimeCredentialData::VertexKey { api_key, .. }
        | RuntimeCredentialData::GeminiApiKey { api_key, .. }
        | RuntimeCredentialData::AnthropicKey { api_key, .. } => api_key,
    };
    Ok(ResolvedRouteCredential {
        key_id: Some(key_id),
        api_key,
        auth_header,
        auth_prefix,
    })
}

fn record_credential_usage(
    db: &lime_core::database::DbConnection,
    api_key_service: &ApiKeyProviderService,
    provider_id: &str,
    key_id: Option<&str>,
) {
    let Some(key_id) = key_id else {
        return;
    };
    if let Err(error) = api_key_service.record_usage(db, key_id) {
        tracing::warn!(
            provider_id = %provider_id,
            key_id = %key_id,
            error = %error,
            "failed to record media provider api key usage"
        );
    }
}

fn route_failure_present(payload: &Value) -> bool {
    payload.get("route_failure").is_some()
        || payload.get("routeFailure").is_some()
        || payload
            .get("model_route_assessment")
            .or_else(|| payload.get("modelRouteAssessment"))
            .and_then(|assessment| read_value_string(assessment, &["status"]))
            .as_deref()
            == Some("blocked")
}

fn route_model_ref_string(route: &Value, keys: &[&str]) -> Option<String> {
    route
        .get("modelRef")
        .or_else(|| route.get("model_ref"))
        .and_then(|model_ref| read_value_string(model_ref, keys))
}

fn image_generation_endpoint_from_route(route: &Value, protocol: &str) -> Option<String> {
    if is_zhipu_image_route_for_protocol(route, protocol) {
        let base_url = route
            .get("endpoint")
            .and_then(|endpoint| read_value_string(endpoint, &["baseUrl", "base_url"]))
            .unwrap_or_else(|| "https://open.bigmodel.cn/api/paas/v4".to_string());
        return Some(image_generation_endpoint_from_zhipu_base(&base_url));
    }
    if is_dashscope_image_route_for_protocol(route, protocol) {
        let base_url = route
            .get("endpoint")
            .and_then(|endpoint| read_value_string(endpoint, &["baseUrl", "base_url"]))
            .unwrap_or_else(|| "https://dashscope.aliyuncs.com/compatible-mode/v1".to_string());
        return Some(image_generation_endpoint_from_dashscope_base(&base_url));
    }

    match protocol {
        "dashscope_multimodal_generation" => {
            let base_url = route
                .get("endpoint")
                .and_then(|endpoint| read_value_string(endpoint, &["baseUrl", "base_url"]))
                .unwrap_or_else(|| "https://dashscope.aliyuncs.com/compatible-mode/v1".to_string());
            Some(image_generation_endpoint_from_dashscope_base(&base_url))
        }
        "openai_images" | "openai_responses" | "codex_responses" => {
            let base_url = route
                .get("endpoint")
                .and_then(|endpoint| read_value_string(endpoint, &["baseUrl", "base_url"]))?;
            Some(image_generation_endpoint_from_openai_base(&base_url))
        }
        "gemini_generate_content" => {
            let base_url = route
                .get("endpoint")
                .and_then(|endpoint| read_value_string(endpoint, &["baseUrl", "base_url"]))?;
            Some(image_generation_endpoint_from_gemini_base(&base_url))
        }
        _ => None,
    }
}

fn image_generation_endpoint_from_openai_base(base_url: &str) -> String {
    let normalized = base_url.trim().trim_end_matches('/');
    if normalized.ends_with("/v1/images/generations") || normalized.ends_with("/images/generations")
    {
        return normalized.to_string();
    }
    if normalized.ends_with("/v1") {
        format!("{normalized}/images/generations")
    } else {
        format!("{normalized}/v1/images/generations")
    }
}

fn video_generation_endpoint_from_route(
    route: &Value,
    protocol: &str,
    model_id: &str,
) -> Option<String> {
    let base_url = route
        .get("endpoint")
        .and_then(|endpoint| read_value_string(endpoint, &["baseUrl", "base_url"]))?;
    let normalized = normalize_urlish_base(&base_url);
    if normalized.is_empty() {
        return None;
    }
    match protocol {
        "fal" => {
            if normalized.ends_with("/videos/generations") || normalized.ends_with(model_id) {
                return Some(normalized);
            }
            Some(format!(
                "{}/{}",
                normalized.trim_end_matches('/'),
                model_id.trim_start_matches('/')
            ))
        }
        "xai_video" => Some(xai_video_generation_endpoint(&normalized)),
        _ => None,
    }
}

fn xai_video_generation_endpoint(base_url: &str) -> String {
    let normalized = base_url.trim().trim_end_matches('/');
    if normalized.ends_with("/videos/generations") {
        return normalized.to_string();
    }
    if normalized.ends_with("/v1") {
        format!("{normalized}/videos/generations")
    } else {
        format!("{normalized}/v1/videos/generations")
    }
}

fn image_generation_endpoint_from_gemini_base(base_url: &str) -> String {
    let normalized = base_url.trim().trim_end_matches('/');
    if normalized.contains(":generateContent") {
        return normalized.to_string();
    }
    if normalized.ends_with("/v1") || normalized.ends_with("/v1beta") {
        return normalized.to_string();
    }
    format!("{normalized}/v1beta")
}

fn image_generation_endpoint_from_zhipu_base(base_url: &str) -> String {
    let normalized = normalize_urlish_base(base_url);
    if normalized.is_empty() {
        return "https://open.bigmodel.cn/api/paas/v4/images/generations".to_string();
    }
    let Ok(mut url) = reqwest::Url::parse(&normalized) else {
        let base = normalized
            .strip_suffix("/images/generations")
            .unwrap_or(&normalized)
            .trim_end_matches('/');
        return if base.eq_ignore_ascii_case("https://open.bigmodel.cn") {
            "https://open.bigmodel.cn/api/paas/v4/images/generations".to_string()
        } else {
            format!("{base}/images/generations")
        };
    };

    let mut segments: Vec<String> = url
        .path_segments()
        .map(|items| {
            items
                .filter(|segment| !segment.is_empty())
                .map(ToString::to_string)
                .collect()
        })
        .unwrap_or_default();
    if segments.len() >= 2
        && segments[segments.len() - 2] == "images"
        && segments[segments.len() - 1] == "generations"
    {
        segments.truncate(segments.len() - 2);
    }
    if segments.is_empty() && url.host_str() == Some("open.bigmodel.cn") {
        segments = ["api", "paas", "v4"]
            .into_iter()
            .map(ToString::to_string)
            .collect();
    }
    let base_path = if segments.is_empty() {
        "/".to_string()
    } else {
        format!("/{}", segments.join("/"))
    };
    url.set_path(&base_path);
    url.set_query(None);
    url.set_fragment(None);

    let base = url.to_string().trim_end_matches('/').to_string();
    if base.eq_ignore_ascii_case("https://open.bigmodel.cn") {
        "https://open.bigmodel.cn/api/paas/v4/images/generations".to_string()
    } else {
        format!("{base}/images/generations")
    }
}

fn image_generation_endpoint_from_dashscope_base(base_url: &str) -> String {
    let normalized = normalize_urlish_base(base_url);
    if normalized.is_empty() {
        return "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
            .to_string();
    }
    let Ok(mut url) = reqwest::Url::parse(&normalized) else {
        let base = normalized
            .strip_suffix("/api/v1/services/aigc/multimodal-generation/generation")
            .unwrap_or(&normalized)
            .trim_end_matches('/');
        return format!("{base}/api/v1/services/aigc/multimodal-generation/generation");
    };

    if url
        .path()
        .trim_end_matches('/')
        .ends_with("/api/v1/services/aigc/multimodal-generation/generation")
    {
        url.set_query(None);
        url.set_fragment(None);
        return url.to_string().trim_end_matches('/').to_string();
    }

    url.set_path("/api/v1/services/aigc/multimodal-generation/generation");
    url.set_query(None);
    url.set_fragment(None);
    url.to_string().trim_end_matches('/').to_string()
}

fn normalize_urlish_base(base_url: &str) -> String {
    let trimmed = base_url.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return String::new();
    }
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_string()
    } else {
        format!("https://{trimmed}")
    }
}

fn image_executor_mode_from_route(route: &Value, protocol: &str) -> &'static str {
    if is_zhipu_image_route_for_protocol(route, protocol) {
        return "zhipu_images";
    }
    if is_dashscope_image_route_for_protocol(route, protocol)
        || protocol == "dashscope_multimodal_generation"
    {
        return "dashscope_images";
    }

    match protocol {
        "openai_responses" | "codex_responses" => "responses_image_generation",
        "gemini_generate_content" => "gemini_generate_content",
        _ => "images_api",
    }
}

fn image_request_body_format_from_route(
    route: &Value,
    protocol: &str,
) -> ImageGenerationRequestBodyFormat {
    if matches!(
        protocol,
        "openai_images" | "openai_responses" | "codex_responses"
    ) && is_agnes_image_route(route)
    {
        return ImageGenerationRequestBodyFormat::AgnesImages;
    }

    ImageGenerationRequestBodyFormat::OpenaiImages
}

fn is_zhipu_image_route(route: &Value) -> bool {
    let provider_id = route_model_ref_string(route, &["providerId", "provider_id"])
        .or_else(|| {
            route
                .get("provider")
                .and_then(|value| read_value_string(value, &["id", "providerId", "provider_id"]))
        })
        .unwrap_or_default()
        .to_ascii_lowercase();
    let model_id = route_model_ref_string(route, &["modelId", "model_id"])
        .or_else(|| {
            route
                .get("model")
                .and_then(|value| read_value_string(value, &["id", "modelId", "model_id"]))
        })
        .unwrap_or_default()
        .to_ascii_lowercase();
    let base_url = route
        .get("endpoint")
        .and_then(|endpoint| read_value_string(endpoint, &["baseUrl", "base_url"]))
        .unwrap_or_default()
        .to_ascii_lowercase();

    provider_id.contains("zhipu")
        || provider_id.contains("bigmodel")
        || base_url.contains("bigmodel.cn/api/paas")
        || matches!(
            model_id.as_str(),
            "glm-image" | "cogview-4-250304" | "cogview-4" | "cogview-3-flash"
        )
        || model_id.contains("cogview")
}

fn is_dashscope_image_route(route: &Value) -> bool {
    let provider_id = route_model_ref_string(route, &["providerId", "provider_id"])
        .or_else(|| {
            route
                .get("provider")
                .and_then(|value| read_value_string(value, &["id", "providerId", "provider_id"]))
        })
        .unwrap_or_default()
        .to_ascii_lowercase();
    let model_id = route_model_ref_string(route, &["modelId", "model_id"])
        .or_else(|| {
            route
                .get("model")
                .and_then(|value| read_value_string(value, &["id", "modelId", "model_id"]))
        })
        .unwrap_or_default();
    let base_url = route
        .get("endpoint")
        .and_then(|endpoint| read_value_string(endpoint, &["baseUrl", "base_url"]))
        .unwrap_or_default()
        .to_ascii_lowercase();
    let provider_matches = provider_id.contains("dashscope")
        || provider_id.contains("alibaba")
        || provider_id.contains("qwen")
        || provider_id.contains("tongyi")
        || base_url.contains("dashscope.aliyuncs.com")
        || base_url.contains("dashscope-intl.aliyuncs.com")
        || base_url.contains("maas.aliyuncs.com");

    provider_matches && is_dashscope_image_model_id(&model_id)
}

fn is_dashscope_image_model_id(model_id: &str) -> bool {
    let normalized = model_id.trim().to_ascii_lowercase();
    normalized.contains("qwen-image")
        || normalized.contains("wanx")
        || normalized.contains("wan2.")
        || normalized.contains("wan2-")
}

fn is_agnes_image_route(route: &Value) -> bool {
    let provider_id = route_model_ref_string(route, &["providerId", "provider_id"])
        .or_else(|| {
            route
                .get("provider")
                .and_then(|value| read_value_string(value, &["id", "providerId", "provider_id"]))
        })
        .unwrap_or_default()
        .to_ascii_lowercase();
    let model_id = route_model_ref_string(route, &["modelId", "model_id"])
        .or_else(|| {
            route
                .get("model")
                .and_then(|value| read_value_string(value, &["id", "modelId", "model_id"]))
        })
        .unwrap_or_default();
    let base_url = route
        .get("endpoint")
        .and_then(|endpoint| read_value_string(endpoint, &["baseUrl", "base_url"]))
        .unwrap_or_default()
        .to_ascii_lowercase();

    provider_id.contains("agnes")
        || base_url.contains("agnes-ai.com")
        || is_agnes_image_model_id(&model_id)
}

fn is_agnes_image_model_id(model_id: &str) -> bool {
    model_id
        .trim()
        .to_ascii_lowercase()
        .starts_with("agnes-image-")
}

fn is_zhipu_image_route_for_protocol(route: &Value, protocol: &str) -> bool {
    matches!(
        protocol,
        "openai_images" | "openai_responses" | "codex_responses"
    ) && is_zhipu_image_route(route)
}

fn is_dashscope_image_route_for_protocol(route: &Value, protocol: &str) -> bool {
    matches!(
        protocol,
        "openai_images" | "openai_responses" | "codex_responses"
    ) && is_dashscope_image_route(route)
}

fn read_value_string(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|key| value.get(*key))
        .find_map(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lime_core::database::dao::api_key_provider::ApiProviderType;
    use lime_core::database::schema::create_tables;
    use lime_core::models::runtime_api_key_credential_uuid;
    use rusqlite::Connection;
    use std::sync::{Arc, Mutex};

    #[test]
    fn resolved_route_uses_exact_credential_ref_instead_of_round_robin() {
        let conn = Connection::open_in_memory().expect("open in-memory db");
        create_tables(&conn).expect("create schema");
        let db = Arc::new(Mutex::new(conn));
        let service = ApiKeyProviderService::new();
        let provider = service
            .add_custom_provider(
                &db,
                "Exact Image Credential".to_string(),
                ApiProviderType::Openai,
                "https://images.example/v1".to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .expect("create provider");
        service
            .add_api_key(&db, &provider.id, "image-key-a", None, false)
            .expect("add key A");
        let key_b = service
            .add_api_key(&db, &provider.id, "image-key-b", None, false)
            .expect("add key B");
        let credential_ref = runtime_api_key_credential_uuid(&key_b.id);
        let route = serde_json::json!({
            "auth": {
                "credentialRef": credential_ref
            }
        });

        let (key_id, api_key) =
            image_api_key_from_resolved_route(&route, &db, &service, &provider.id)
                .expect("resolve exact image credential");

        assert_eq!(key_id.as_deref(), Some(key_b.id.as_str()));
        assert_eq!(api_key, "image-key-b");
    }

    #[test]
    fn resolved_no_auth_route_does_not_select_provider_credential() {
        let conn = Connection::open_in_memory().expect("open in-memory db");
        create_tables(&conn).expect("create schema");
        let db = Arc::new(Mutex::new(conn));
        let service = ApiKeyProviderService::new();
        let route = serde_json::json!({
            "auth": {
                "kind": "no_auth"
            }
        });

        let (key_id, api_key) =
            image_api_key_from_resolved_route(&route, &db, &service, "keyless-images")
                .expect("resolve no-auth image route");

        assert!(key_id.is_none());
        assert!(api_key.is_empty());
    }

    #[test]
    fn image_generation_endpoint_from_openai_base_normalizes_common_shapes() {
        assert_eq!(
            image_generation_endpoint_from_openai_base("https://api.openai.com/v1"),
            "https://api.openai.com/v1/images/generations"
        );
        assert_eq!(
            image_generation_endpoint_from_openai_base(
                "https://gateway.example.com/proxy/images/generations"
            ),
            "https://gateway.example.com/proxy/images/generations"
        );
        assert_eq!(
            image_generation_endpoint_from_openai_base("https://gateway.example.com/proxy"),
            "https://gateway.example.com/proxy/v1/images/generations"
        );
    }

    #[test]
    fn video_generation_endpoint_uses_fal_model_path_or_explicit_generation_endpoint() {
        let fal_route = serde_json::json!({
            "endpoint": { "baseUrl": "https://fal.run" }
        });
        assert_eq!(
            video_generation_endpoint_from_route(
                &fal_route,
                "fal",
                "fal-ai/kling-video/v2.1/master/image-to-video"
            )
            .as_deref(),
            Some("https://fal.run/fal-ai/kling-video/v2.1/master/image-to-video")
        );

        let explicit_route = serde_json::json!({
            "endpoint": { "baseUrl": "https://video.example/v1/videos/generations" }
        });
        assert_eq!(
            video_generation_endpoint_from_route(&explicit_route, "fal", "video-model").as_deref(),
            Some("https://video.example/v1/videos/generations")
        );
        assert!(
            video_generation_endpoint_from_route(&fal_route, "openai_chat", "video-model")
                .is_none()
        );

        let xai_route = serde_json::json!({
            "endpoint": { "baseUrl": "https://api.x.ai/v1" }
        });
        assert_eq!(
            video_generation_endpoint_from_route(&xai_route, "xai_video", "grok-imagine-video")
                .as_deref(),
            Some("https://api.x.ai/v1/videos/generations")
        );
    }

    #[test]
    fn resolved_route_credential_preserves_provider_auth_shape() {
        let conn = Connection::open_in_memory().expect("open in-memory db");
        create_tables(&conn).expect("create schema");
        let db = Arc::new(Mutex::new(conn));
        let service = ApiKeyProviderService::new();
        let provider = service
            .add_custom_provider(
                &db,
                "Fal Video".to_string(),
                ApiProviderType::Openai,
                "https://fal.run".to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .expect("create provider");
        service
            .add_api_key(&db, &provider.id, "fal-key-a", None, false)
            .expect("add key A");
        let key = service
            .add_api_key(&db, &provider.id, "fal-key-b", None, false)
            .expect("add key B");
        let route = serde_json::json!({
            "auth": {
                "credentialRef": runtime_api_key_credential_uuid(&key.id),
                "headerName": "Authorization",
                "headerPrefix": "Key"
            }
        });

        let credential =
            route_credential_from_resolved_route(&route, &db, &service, &provider.id, "视频")
                .expect("video credential");

        assert_eq!(credential.api_key, "fal-key-b");
        assert_eq!(credential.auth_header, "Authorization");
        assert_eq!(credential.auth_prefix.as_deref(), Some("Key"));
    }

    #[test]
    fn image_generation_endpoint_from_route_requires_image_protocol() {
        let route = serde_json::json!({
            "protocol": "openai_images",
            "endpoint": {
                "baseUrl": "https://api.openai.com/v1"
            }
        });
        assert_eq!(
            image_generation_endpoint_from_route(&route, "openai_images").as_deref(),
            Some("https://api.openai.com/v1/images/generations")
        );
        let gemini_route = serde_json::json!({
            "protocol": "gemini_generate_content",
            "endpoint": {
                "baseUrl": "https://generativelanguage.googleapis.com"
            }
        });
        assert_eq!(
            image_generation_endpoint_from_route(&gemini_route, "gemini_generate_content")
                .as_deref(),
            Some("https://generativelanguage.googleapis.com/v1beta")
        );
        let zhipu_route = serde_json::json!({
            "protocol": "openai_images",
            "modelRef": {
                "providerId": "zhipuai",
                "modelId": "glm-image"
            },
            "endpoint": {
                "baseUrl": "https://open.bigmodel.cn/api/paas/v4"
            }
        });
        assert_eq!(
            image_generation_endpoint_from_route(&zhipu_route, "openai_images").as_deref(),
            Some("https://open.bigmodel.cn/api/paas/v4/images/generations")
        );
        assert_eq!(
            image_executor_mode_from_route(&zhipu_route, "openai_images"),
            "zhipu_images"
        );
        let dashscope_route = serde_json::json!({
            "protocol": "openai_images",
            "modelRef": {
                "providerId": "alibaba",
                "modelId": "qwen-image-plus"
            },
            "endpoint": {
                "baseUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1"
            }
        });
        assert_eq!(
            image_generation_endpoint_from_route(&dashscope_route, "openai_images").as_deref(),
            Some(
                "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
            )
        );
        assert_eq!(
            image_executor_mode_from_route(&dashscope_route, "openai_images"),
            "dashscope_images"
        );
        let agnes_route = serde_json::json!({
            "protocol": "openai_images",
            "modelRef": {
                "providerId": "agnes",
                "modelId": "agnes-image-2.1-flash"
            },
            "endpoint": {
                "baseUrl": "https://apihub.agnes-ai.com/v1"
            }
        });
        assert_eq!(
            image_generation_endpoint_from_route(&agnes_route, "openai_images").as_deref(),
            Some("https://apihub.agnes-ai.com/v1/images/generations")
        );
        assert_eq!(
            image_executor_mode_from_route(&agnes_route, "openai_images"),
            "images_api"
        );
        assert_eq!(
            image_request_body_format_from_route(&agnes_route, "openai_images"),
            ImageGenerationRequestBodyFormat::AgnesImages
        );
        assert!(image_generation_endpoint_from_route(&zhipu_route, "anthropic_messages").is_none());
        assert!(image_generation_endpoint_from_route(&route, "anthropic_messages").is_none());
    }
}
