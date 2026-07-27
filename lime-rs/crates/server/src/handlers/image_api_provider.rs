mod gemini_image;
mod provider_routing;

use std::{collections::BTreeMap, time::Duration};

use axum::http::StatusCode;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use futures::StreamExt;
use lime_core::api_host_utils::normalize_openai_compatible_api_host;
use lime_core::config::ConfigManager;
use lime_core::database::dao::api_key_provider::{ApiKeyProvider, ApiKeyProviderDao};
use lime_core::models::model_registry::ProviderModelConfig;
use lime_core::models::openai::{ImageData, ImageGenerationRequest, ImageGenerationResponse};
use model_provider::current_client::responses_endpoint;
use reqwest::{header::CONTENT_TYPE, Client};
use serde_json::{json, Value};

use self::gemini_image::{request_gemini_interactions_image, resolve_gemini_image_model};
use self::provider_routing::{
    resolve_compatible_image_model, resolve_configured_image_provider_kind, resolve_fal_model,
    resolve_openai_responses_image_orchestration_model, should_prefer_openai_responses_image_api,
};
use crate::AppState;

const FAL_DEFAULT_HOST: &str = "https://fal.run";
const FAL_DEFAULT_MODEL: &str = "fal-ai/nano-banana-pro";
const FAL_QUEUE_DEFAULT_HOST: &str = "https://queue.fal.run";
const OPENAI_RESPONSES_IMAGE_ORCHESTRATOR_MODEL: &str = "gpt-5.4";
const IMAGE_PROVIDER_REQUEST_TIMEOUT_SECS: u64 = 240;
const FAL_QUEUE_TIMEOUT_SECS: u64 = 180;
const FAL_QUEUE_POLL_INTERVAL_MS: u64 = 1500;

pub(crate) struct ConfiguredImageProviderError {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: String,
}

struct ImageProviderRoutingConfig {
    provider_id: String,
    preferred_model_id: Option<String>,
    allow_fallback: bool,
    default_size: Option<String>,
    is_explicit: bool,
}

enum ConfiguredImageProviderKind {
    Fal,
    Gemini,
    Zhipu,
    OpenAiCompatible,
}

#[derive(Debug)]
struct OpenAiImageEndpointError {
    message: String,
    is_endpoint_not_found: bool,
}

pub(crate) async fn try_generate_with_configured_provider(
    state: &AppState,
    request: &ImageGenerationRequest,
    explicit_provider_id: Option<&str>,
) -> Result<Option<ImageGenerationResponse>, ConfiguredImageProviderError> {
    let Some(routing) = load_image_provider_routing(explicit_provider_id) else {
        return Ok(None);
    };

    let db = state
        .db
        .as_ref()
        .ok_or_else(|| ConfiguredImageProviderError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "database_unavailable",
            message: "Database not available".to_string(),
        })?;

    let provider = match load_api_key_provider(db, &routing.provider_id) {
        Ok(Some(provider)) if provider.enabled => provider,
        Ok(Some(_)) | Ok(None) => {
            return handle_routing_failure(
                state,
                &routing,
                format!("默认图片服务 {} 当前不可用", routing.provider_id),
                "configured_provider_unavailable",
            )
        }
        Err(error) => {
            return handle_routing_failure(
                state,
                &routing,
                format!("读取默认图片服务失败: {error}"),
                "configured_provider_lookup_failed",
            )
        }
    };

    let provider_kind = match resolve_configured_image_provider_kind(&provider, request, &routing) {
        Some(kind) => kind,
        None => {
            return handle_routing_failure(
                state,
                &routing,
                format!(
                    "当前默认图片服务 {} 尚未接入 /v1/images/generations",
                    provider.id
                ),
                "configured_provider_not_supported",
            )
        }
    };

    let Some((key_id, api_key)) = state
        .api_key_service
        .get_next_api_key_entry(db, &provider.id)
        .map_err(|error| ConfiguredImageProviderError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "configured_provider_key_failed",
            message: format!("读取默认图片服务 API Key 失败: {error}"),
        })?
    else {
        return handle_routing_failure(
            state,
            &routing,
            format!("默认图片服务 {} 没有可用的 API Key", provider.id),
            "configured_provider_missing_key",
        );
    };

    let client = Client::builder()
        .timeout(Duration::from_secs(IMAGE_PROVIDER_REQUEST_TIMEOUT_SECS))
        .build()
        .unwrap_or_else(|_| Client::new());
    let request_size = request
        .size
        .clone()
        .or_else(|| routing.default_size.clone())
        .unwrap_or_else(|| "1024x1024".to_string());

    state.logs.write().await.add(
        "info",
        &format!(
            "[IMAGE] 图片服务命中 API Provider: provider_id={}, kind={}, requested_model={}, size={}, explicit={}",
            provider.id,
            match provider_kind {
                ConfiguredImageProviderKind::Fal => "fal",
                ConfiguredImageProviderKind::Gemini => "gemini-image",
                ConfiguredImageProviderKind::Zhipu => "zhipu-image",
                ConfiguredImageProviderKind::OpenAiCompatible => "openai-images",
            },
            request.model,
            request_size,
            routing.is_explicit
        ),
    );

    let response = match provider_kind {
        ConfiguredImageProviderKind::Fal => {
            let request_model = resolve_fal_model(
                request.model.as_str(),
                routing.preferred_model_id.as_deref(),
            );
            match request_fal_images(
                &client,
                &provider.api_host,
                &api_key,
                &request.prompt,
                &request_model,
                &request_size,
                request.n.max(1),
                &request.reference_images,
            )
            .await
            {
                Ok(image_urls) => {
                    build_openai_response(&client, &image_urls, &request.response_format, None)
                        .await
                }
                Err(error) => Err(error),
            }
        }
        ConfiguredImageProviderKind::Gemini => {
            let Some(request_model) = resolve_gemini_image_model(
                request.model.as_str(),
                routing.preferred_model_id.as_deref(),
                &provider.models,
            ) else {
                return handle_routing_failure(
                    state,
                    &routing,
                    format!("默认图片服务 {} 缺少可用 Gemini 图片模型", provider.id),
                    "configured_provider_missing_model",
                );
            };

            request_gemini_interactions_image(
                &client,
                &provider.api_host,
                &api_key,
                request,
                &request_model,
                &request_size,
            )
            .await
        }
        ConfiguredImageProviderKind::Zhipu => {
            let Some(request_model) = resolve_zhipu_image_model(
                request.model.as_str(),
                routing.preferred_model_id.as_deref(),
                &provider.models,
            ) else {
                return handle_routing_failure(
                    state,
                    &routing,
                    format!("默认图片服务 {} 缺少可用智谱图片模型", provider.id),
                    "configured_provider_missing_model",
                );
            };

            request_zhipu_images(
                &client,
                &provider.api_host,
                &api_key,
                request,
                &request_model,
                &request_size,
            )
            .await
        }
        ConfiguredImageProviderKind::OpenAiCompatible => {
            let Some(request_model) = resolve_compatible_image_model(
                request.model.as_str(),
                routing.preferred_model_id.as_deref(),
                &provider.models,
            ) else {
                return handle_routing_failure(
                    state,
                    &routing,
                    format!("默认图片服务 {} 缺少可用图片模型", provider.id),
                    "configured_provider_missing_model",
                );
            };

            request_openai_compatible_images(
                &client,
                &provider,
                &provider.api_host,
                &api_key,
                request,
                &request_model,
                &request_size,
                !request.reference_images.is_empty(),
            )
            .await
        }
    };

    let response = match response {
        Ok(response) => response,
        Err(error) => {
            let _ = state.api_key_service.record_error(db, &key_id);
            return handle_routing_failure(
                state,
                &routing,
                format!("默认图片服务调用失败: {error}"),
                "configured_provider_request_failed",
            );
        }
    };

    let _ = state.api_key_service.record_usage(db, &key_id);

    Ok(Some(response))
}

fn handle_routing_failure(
    _state: &AppState,
    routing: &ImageProviderRoutingConfig,
    message: String,
    code: &'static str,
) -> Result<Option<ImageGenerationResponse>, ConfiguredImageProviderError> {
    if routing.allow_fallback {
        tracing::warn!("[IMAGE] 默认图片服务失败，允许回退: {}", message);
        return Ok(None);
    }

    Err(ConfiguredImageProviderError {
        status: StatusCode::SERVICE_UNAVAILABLE,
        code,
        message,
    })
}

fn load_image_provider_routing(
    explicit_provider_id: Option<&str>,
) -> Option<ImageProviderRoutingConfig> {
    if let Some(provider_id) = explicit_provider_id
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| item.to_string())
    {
        return Some(ImageProviderRoutingConfig {
            provider_id,
            preferred_model_id: None,
            allow_fallback: false,
            default_size: None,
            is_explicit: true,
        });
    }

    let config_path = ConfigManager::default_config_path();
    let manager = ConfigManager::load(&config_path).ok()?;
    let image_preference = manager
        .config()
        .workspace_preferences
        .media_defaults
        .image
        .clone();
    let provider_id = image_preference.preferred_provider_id?.trim().to_string();

    if provider_id.is_empty() {
        return None;
    }

    Some(ImageProviderRoutingConfig {
        provider_id,
        preferred_model_id: normalize_optional_string(image_preference.preferred_model_id),
        allow_fallback: image_preference.allow_fallback,
        default_size: normalize_optional_string(manager.config().image_gen.default_size.clone()),
        is_explicit: false,
    })
}

fn normalize_optional_string(value: Option<String>) -> Option<String> {
    value.and_then(|item| {
        let trimmed = item.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn load_api_key_provider(
    db: &lime_core::database::DbConnection,
    provider_id: &str,
) -> Result<Option<ApiKeyProvider>, String> {
    let conn = lime_core::database::lock_db(db)?;
    ApiKeyProviderDao::get_provider_by_id(&conn, provider_id).map_err(|error| error.to_string())
}

fn build_openai_image_endpoint_url(api_host: &str, endpoint_path: &str) -> String {
    let normalized_host = normalize_openai_compatible_api_host(api_host);
    let trimmed = normalized_host.trim().trim_end_matches('/');
    let normalized = if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_string()
    } else if trimmed.is_empty() {
        "https://api.openai.com".to_string()
    } else {
        format!("https://{trimmed}")
    };

    let has_version = normalized
        .rsplit('/')
        .next()
        .map(|segment| {
            segment.starts_with('v')
                && segment.len() >= 2
                && segment[1..].chars().all(|char| char.is_ascii_digit())
        })
        .unwrap_or(false);

    let endpoint_path = endpoint_path.trim();
    let endpoint_path = if endpoint_path.starts_with('/') {
        endpoint_path.to_string()
    } else {
        format!("/{endpoint_path}")
    };

    if has_version {
        format!("{normalized}{endpoint_path}")
    } else {
        format!("{normalized}/v1{endpoint_path}")
    }
}

fn build_openai_images_url(api_host: &str) -> String {
    build_openai_image_endpoint_url(api_host, "/images/generations")
}

fn build_openai_image_request_payload(
    request: &ImageGenerationRequest,
    model: &str,
    request_size: &str,
    use_edit_endpoint: bool,
) -> Value {
    let mut payload = json!({
        "model": model,
        "prompt": request.prompt.trim(),
        "n": request.n.max(1),
        "response_format": request.response_format,
        "size": request_size,
    });

    if let Some(quality) = request
        .quality
        .as_ref()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
    {
        payload["quality"] = Value::String(quality.to_string());
    }

    if let Some(style) = request
        .style
        .as_ref()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
    {
        payload["style"] = Value::String(style.to_string());
    }

    if let Some(user) = request
        .user
        .as_ref()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
    {
        payload["user"] = Value::String(user.to_string());
    }

    if use_edit_endpoint && !request.reference_images.is_empty() {
        payload["images"] = Value::Array(
            request
                .reference_images
                .iter()
                .filter_map(|image| {
                    let trimmed = image.trim();
                    if trimmed.is_empty() {
                        None
                    } else {
                        Some(json!({ "image_url": trimmed }))
                    }
                })
                .collect(),
        );
    }

    payload
}

fn looks_like_zhipu_image_model(model: &str) -> bool {
    let normalized = model.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return false;
    }

    matches!(
        normalized.as_str(),
        "glm-image" | "cogview-4-250304" | "cogview-4" | "cogview-3-flash"
    ) || normalized.starts_with("glm-image")
        || normalized.contains("cogview")
}

fn resolve_zhipu_image_model(
    request_model: &str,
    preferred_model_id: Option<&str>,
    models: &[ProviderModelConfig],
) -> Option<String> {
    let trimmed = request_model.trim();
    if looks_like_zhipu_image_model(trimmed) {
        return Some(trimmed.to_string());
    }

    if let Some(preferred_model) = preferred_model_id
        .map(str::trim)
        .filter(|value| looks_like_zhipu_image_model(value))
    {
        return Some(preferred_model.to_string());
    }

    models
        .iter()
        .map(|model| model.id.trim())
        .find(|candidate| looks_like_zhipu_image_model(candidate))
        .map(ToString::to_string)
}

fn normalize_zhipu_model(model: &str) -> String {
    let trimmed = model.trim();
    if trimmed.is_empty() {
        return "glm-image".to_string();
    }

    if matches!(
        trimmed,
        "cogview-4-250304" | "cogview-4" | "cogview-3-flash"
    ) {
        trimmed.to_string()
    } else {
        "glm-image".to_string()
    }
}

fn normalize_zhipu_quality(model: &str, quality: Option<&str>) -> Option<String> {
    let normalized_model = normalize_zhipu_model(model);
    if normalized_model == "glm-image" {
        return Some("hd".to_string());
    }

    let trimmed = quality?.trim();
    if trimmed.is_empty() {
        return None;
    }

    if trimmed.eq_ignore_ascii_case("hd") {
        return Some("hd".to_string());
    }

    if trimmed.eq_ignore_ascii_case("standard") {
        return Some("standard".to_string());
    }

    None
}

fn normalize_zhipu_size(model: &str, request_size: &str) -> String {
    let trimmed = request_size.trim();
    if !trimmed.is_empty() {
        return trimmed.to_string();
    }

    if normalize_zhipu_model(model) == "glm-image" {
        "1280x1280".to_string()
    } else {
        "1024x1024".to_string()
    }
}

fn build_zhipu_image_request_payload(
    request: &ImageGenerationRequest,
    model: &str,
    request_size: &str,
) -> Value {
    let mut payload = json!({
        "model": normalize_zhipu_model(model),
        "prompt": request.prompt.trim(),
        "size": normalize_zhipu_size(model, request_size),
    });

    if let Some(quality) = normalize_zhipu_quality(model, request.quality.as_deref()) {
        payload["quality"] = Value::String(quality);
    }

    if let Some(user) = request
        .user
        .as_ref()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .filter(|value| value.len() >= 6 && value.len() <= 128)
    {
        payload["user_id"] = Value::String(user.to_string());
    }

    if request.n.max(1) > 1 {
        tracing::debug!(
            "[IMAGE] Zhipu images API 仅按单图处理，已将 n={} 收敛为 1",
            request.n
        );
    }

    payload
}

fn build_zhipu_images_url(api_host: &str) -> String {
    let trimmed = api_host.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return "https://open.bigmodel.cn/api/paas/v4/images/generations".to_string();
    }

    let normalized = if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_string()
    } else {
        format!("https://{trimmed}")
    };

    let Ok(mut url) = reqwest::Url::parse(&normalized) else {
        let base = normalized
            .strip_suffix("/images/generations")
            .unwrap_or(&normalized)
            .trim_end_matches('/')
            .to_string();
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
                .map(|segment| segment.to_string())
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
        segments = vec!["api", "paas", "v4"]
            .into_iter()
            .map(str::to_string)
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

async fn request_zhipu_images(
    client: &Client,
    api_host: &str,
    api_key: &str,
    request: &ImageGenerationRequest,
    model: &str,
    request_size: &str,
) -> Result<ImageGenerationResponse, String> {
    if !request.reference_images.is_empty() {
        return Err("智谱图片服务当前暂不支持参考图或修图请求".to_string());
    }

    let endpoint = build_zhipu_images_url(api_host);
    let payload = build_zhipu_image_request_payload(request, model, request_size);
    let response = client
        .post(&endpoint)
        .header("Authorization", format!("Bearer {api_key}"))
        .header(CONTENT_TYPE, "application/json")
        .json(&payload)
        .send()
        .await
        .map_err(|error| format!("智谱图片接口请求失败: {error}"))?;

    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| format!("智谱图片接口响应读取失败: {error}"))?;

    if !status.is_success() {
        return Err(format!(
            "智谱图片接口 HTTP {}: {}",
            status.as_u16(),
            summarize_image_service_error_body(&body)
        ));
    }

    let payload = serde_json::from_str::<Value>(&body).map_err(|error| {
        format!(
            "智谱图片接口 JSON 解析失败: {error}; body={}",
            preview_text(&body, 240)
        )
    })?;

    let created_at = payload.get("created").and_then(Value::as_i64);
    let image_urls = collect_image_urls(&payload);
    if image_urls.is_empty() {
        return Err(format!(
            "智谱图片接口未返回可解析图片地址: {}",
            preview_text(&body, 240)
        ));
    }

    build_openai_response(client, &image_urls, &request.response_format, created_at).await
}

fn build_openai_responses_image_request_payload(
    request: &ImageGenerationRequest,
    orchestration_model: &str,
    image_model: &str,
    request_size: &str,
    use_input_list: bool,
) -> Value {
    let tool_model = normalize_openai_responses_image_tool_model(image_model);
    // 一些 Codex/OpenAI relay 对 n 字段兼容性较差，默认依赖上游单图默认值。
    let mut tool = json!({
        "type": "image_generation",
        "model": tool_model,
        "output_format": "png",
    });

    if !request_size.trim().is_empty() {
        tool["size"] = Value::String(request_size.trim().to_string());
    }

    json!({
        "model": orchestration_model,
        "input": build_openai_responses_image_input(request, use_input_list),
        "tools": [tool],
        "stream": true,
    })
}

fn build_openai_responses_image_input(
    request: &ImageGenerationRequest,
    use_input_list: bool,
) -> Value {
    let prompt = request.prompt.trim();
    let count = request.n.max(1);
    let input_text = if count <= 1 {
        prompt.to_string()
    } else {
        format!(
            concat!(
                "你是一名图片生成编排器。请调用 image_generation 工具恰好 {count} 次，",
                "每次只生成 1 张独立图片。\n",
                "- 不要只调用一次工具。\n",
                "- 不要把多张图片拼成一张拼贴、九宫格、海报或联系单。\n",
                "- 所有图片要保持同一主题与风格，但每张图都必须能单独使用，并且构图、主体或镜头要有变化。\n",
                "- 如果原始要求里包含多个角色、物体或场景，请把它们合理分配到不同图片里，不要全部塞进同一张。\n",
                "- 最终只返回工具结果，不要额外输出解释文本。\n\n",
                "原始创作要求：{prompt}"
            ),
            count = count,
            prompt = prompt,
        )
    };

    if use_input_list || !request.reference_images.is_empty() {
        let mut content = vec![json!({
            "type": "input_text",
            "text": input_text,
        })];

        if !request.reference_images.is_empty() {
            content.extend(request.reference_images.iter().filter_map(|image| {
                let trimmed = image.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(json!({
                        "type": "input_image",
                        "image_url": trimmed,
                    }))
                }
            }));
        }

        return json!([
            {
                "role": "user",
                "content": content,
            }
        ]);
    }

    json!(input_text)
}

fn normalize_openai_responses_image_tool_model(model: &str) -> String {
    let trimmed = model.trim();
    if let Some(version) = trimmed.strip_prefix("gpt-images-") {
        return format!("gpt-image-{version}");
    }

    trimmed.to_string()
}

fn is_not_found_status(status: StatusCode, body: &str) -> bool {
    if status == StatusCode::NOT_FOUND {
        return true;
    }

    let normalized = body.to_ascii_lowercase();
    normalized.contains("not found")
        || normalized.contains("page not found")
        || body.contains("请求的接口不存在")
}

fn should_retry_openai_responses_image_with_input_list(status: StatusCode, body: &str) -> bool {
    status == StatusCode::BAD_REQUEST && body.to_ascii_lowercase().contains("input must be a list")
}

fn build_openai_responses_data_url(output_format: Option<&str>, b64: &str) -> String {
    let mime = match output_format
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("png")
        .to_ascii_lowercase()
        .as_str()
    {
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        _ => "image/png",
    };

    format!("data:{mime};base64,{b64}")
}

fn buffer_contains_responses_event(buffer: &str, event_name: &str) -> bool {
    buffer.contains(event_name) || buffer.contains(&format!("event: {event_name}"))
}

fn try_finalize_openai_responses_image_stream(
    buffer: &str,
    response_format: &str,
    expected_count: usize,
) -> Result<Option<ImageGenerationResponse>, String> {
    let has_completed = buffer_contains_responses_event(buffer, "response.completed");
    let has_output_item_done = buffer_contains_responses_event(buffer, "response.output_item.done");

    if !has_completed && !has_output_item_done {
        return Ok(None);
    }

    let parsed = normalize_openai_responses_image_sse(buffer, response_format)?;
    let expected = expected_count.max(1);

    if has_completed || parsed.data.len() >= expected {
        return Ok(Some(parsed));
    }

    Ok(None)
}

fn try_extract_partial_openai_responses_images(
    buffer: &str,
    response_format: &str,
) -> Result<Option<ImageGenerationResponse>, String> {
    if !buffer_contains_responses_event(buffer, "response.output_item.done") {
        return Ok(None);
    }

    let parsed = normalize_openai_responses_image_sse(buffer, response_format)?;
    if parsed.data.is_empty() {
        return Ok(None);
    }

    Ok(Some(parsed))
}

fn normalize_openai_responses_image_payload(
    payload: &Value,
    response_format: &str,
) -> Result<ImageGenerationResponse, String> {
    let created_at = payload
        .get("created_at")
        .and_then(Value::as_i64)
        .unwrap_or_else(|| chrono::Utc::now().timestamp());
    let mut data = Vec::new();

    if let Some(items) = payload.get("output").and_then(Value::as_array) {
        for item in items {
            if item.get("type").and_then(Value::as_str) != Some("image_generation_call") {
                continue;
            }

            let revised_prompt = item
                .get("revised_prompt")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(|value| value.to_string());
            let output_format = item
                .get("output_format")
                .and_then(Value::as_str)
                .map(|value| value.to_string());
            let b64 = item
                .get("result")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(|value| value.to_string())
                .or_else(|| {
                    item.get("partial_image_b64")
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(|value| value.to_string())
                });

            let Some(b64) = b64 else {
                continue;
            };

            if response_format == "b64_json" {
                data.push(ImageData {
                    b64_json: Some(b64),
                    url: None,
                    revised_prompt,
                });
            } else {
                data.push(ImageData {
                    b64_json: None,
                    url: Some(build_openai_responses_data_url(
                        output_format.as_deref(),
                        &b64,
                    )),
                    revised_prompt,
                });
            }
        }
    }

    if data.is_empty() {
        return Err("Responses 图片接口未返回可解析图片字段".to_string());
    }

    Ok(ImageGenerationResponse {
        created: created_at,
        data,
    })
}

fn normalize_openai_responses_image_sse(
    body: &str,
    response_format: &str,
) -> Result<ImageGenerationResponse, String> {
    #[derive(Debug)]
    struct ParsedSseEvent {
        event: Option<String>,
        data: String,
    }

    fn parse_sse_events(body: &str) -> Vec<ParsedSseEvent> {
        let mut events = Vec::new();
        let mut current_event: Option<String> = None;
        let mut data_lines: Vec<String> = Vec::new();

        let flush_event = |events: &mut Vec<ParsedSseEvent>,
                           current_event: &mut Option<String>,
                           data_lines: &mut Vec<String>| {
            if current_event.is_none() && data_lines.is_empty() {
                return;
            }
            events.push(ParsedSseEvent {
                event: current_event.take(),
                data: data_lines.join("\n"),
            });
            data_lines.clear();
        };

        for raw_line in body.lines() {
            let line = raw_line.trim_end();
            if line.trim().is_empty() {
                flush_event(&mut events, &mut current_event, &mut data_lines);
                continue;
            }

            let trimmed = line.trim_start();
            if let Some(value) = trimmed.strip_prefix("event:") {
                current_event = Some(value.trim().to_string());
                continue;
            }

            if let Some(value) = trimmed.strip_prefix("data:") {
                data_lines.push(value.trim_start().to_string());
            }
        }

        flush_event(&mut events, &mut current_event, &mut data_lines);
        events
    }

    fn upsert_openai_responses_image_output(
        latest_partial_images: &mut BTreeMap<usize, Value>,
        output_index: usize,
        output_item: &Value,
    ) {
        let mut merged = latest_partial_images
            .remove(&output_index)
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_default();

        if let Some(record) = output_item.as_object() {
            for (key, value) in record {
                if key == "type"
                    && value.as_str() == Some("response.image_generation_call.partial_image")
                {
                    continue;
                }
                merged.insert(key.clone(), value.clone());
            }
        }

        merged.insert(
            "type".to_string(),
            Value::String("image_generation_call".to_string()),
        );
        latest_partial_images.insert(output_index, Value::Object(merged));
    }

    let mut latest_partial_images: BTreeMap<usize, Value> = BTreeMap::new();
    let mut completed_response: Option<Value> = None;

    for sse_event in parse_sse_events(body) {
        let data = sse_event.data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }

        let Ok(event) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        let event_type = event
            .get("type")
            .and_then(Value::as_str)
            .or(sse_event.event.as_deref());

        match event_type {
            Some("response.image_generation_call.partial_image") => {
                let output_index = event
                    .get("output_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize;
                upsert_openai_responses_image_output(
                    &mut latest_partial_images,
                    output_index,
                    &event,
                );
            }
            Some("response.output_item.done") => {
                let item = event.get("item");
                let is_image_generation_item = item
                    .and_then(|value| value.get("type"))
                    .and_then(Value::as_str)
                    == Some("image_generation_call")
                    || item.and_then(|value| value.get("result")).is_some()
                    || item
                        .and_then(|value| value.get("partial_image_b64"))
                        .is_some();
                if is_image_generation_item {
                    let output_index = event
                        .get("output_index")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as usize;
                    upsert_openai_responses_image_output(
                        &mut latest_partial_images,
                        output_index,
                        item.unwrap_or(&Value::Null),
                    );
                }
            }
            Some("response.completed") => {
                completed_response = event
                    .get("response")
                    .cloned()
                    .or_else(|| Some(event.clone()));
            }
            _ => {}
        }
    }

    if !latest_partial_images.is_empty() {
        let created_at = completed_response
            .as_ref()
            .and_then(|response| response.get("created_at"))
            .and_then(Value::as_i64)
            .unwrap_or_else(|| chrono::Utc::now().timestamp());
        let output = latest_partial_images.into_values().collect::<Vec<_>>();
        return normalize_openai_responses_image_payload(
            &json!({
                "created_at": created_at,
                "output": output,
            }),
            response_format,
        );
    }

    if let Some(response) = completed_response {
        return normalize_openai_responses_image_payload(&response, response_format);
    }

    Err("Responses 图片接口未返回任何图片事件".to_string())
}

async fn request_openai_responses_images(
    client: &Client,
    provider: &ApiKeyProvider,
    api_host: &str,
    api_key: &str,
    request: &ImageGenerationRequest,
    image_model: &str,
    request_size: &str,
) -> Result<ImageGenerationResponse, OpenAiImageEndpointError> {
    let endpoint = responses_endpoint(api_host);
    let orchestration_model =
        resolve_openai_responses_image_orchestration_model(provider, image_model);
    let expected_count = request.n.max(1) as usize;
    let mut use_input_list = false;

    loop {
        let payload = build_openai_responses_image_request_payload(
            request,
            &orchestration_model,
            image_model,
            request_size,
            use_input_list,
        );
        let response = client
            .post(&endpoint)
            .header("Authorization", format!("Bearer {api_key}"))
            .header(CONTENT_TYPE, "application/json")
            .header("Accept", "text/event-stream")
            .header("Accept-Encoding", "identity")
            .json(&payload)
            .send()
            .await
            .map_err(|error| OpenAiImageEndpointError {
                message: format!("OpenAI Responses 图片接口请求失败: {error}"),
                is_endpoint_not_found: false,
            })?;

        let status = response.status();
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    buffer.push_str(&String::from_utf8_lossy(&bytes));

                    if !status.is_success() {
                        continue;
                    }

                    if let Ok(Some(parsed)) = try_finalize_openai_responses_image_stream(
                        &buffer,
                        &request.response_format,
                        expected_count,
                    ) {
                        return Ok(parsed);
                    }
                }
                Err(error) => {
                    if let Ok(Some(parsed)) = try_finalize_openai_responses_image_stream(
                        &buffer,
                        &request.response_format,
                        expected_count,
                    ) {
                        return Ok(parsed);
                    }

                    if let Ok(Some(parsed)) = try_extract_partial_openai_responses_images(
                        &buffer,
                        &request.response_format,
                    ) {
                        return Ok(parsed);
                    }

                    return Err(OpenAiImageEndpointError {
                        message: format!("OpenAI Responses 图片接口流读取失败: {error}"),
                        is_endpoint_not_found: false,
                    });
                }
            }
        }

        if !status.is_success() {
            if !use_input_list
                && should_retry_openai_responses_image_with_input_list(status, &buffer)
            {
                use_input_list = true;
                continue;
            }

            return Err(OpenAiImageEndpointError {
                is_endpoint_not_found: is_not_found_status(status, &buffer),
                message: format!(
                    "OpenAI Responses 图片接口 HTTP {}: {}",
                    status.as_u16(),
                    summarize_image_service_error_body(&buffer)
                ),
            });
        }

        if let Some(parsed) = try_finalize_openai_responses_image_stream(
            &buffer,
            &request.response_format,
            expected_count,
        )
        .map_err(|error| OpenAiImageEndpointError {
            message: format!("OpenAI Responses 图片接口解析失败: {error}"),
            is_endpoint_not_found: false,
        })? {
            return Ok(parsed);
        }

        if let Some(parsed) =
            try_extract_partial_openai_responses_images(&buffer, &request.response_format).map_err(
                |error| OpenAiImageEndpointError {
                    message: format!("OpenAI Responses 图片接口解析失败: {error}"),
                    is_endpoint_not_found: false,
                },
            )?
        {
            return Ok(parsed);
        }

        return Err(OpenAiImageEndpointError {
            message: "OpenAI Responses 图片接口解析失败: 未收齐所需图片事件".to_string(),
            is_endpoint_not_found: false,
        });
    }
}

fn normalize_fal_api_host(api_host: &str) -> String {
    let trimmed = api_host.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return FAL_DEFAULT_HOST.to_string();
    }

    let normalized = if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_string()
    } else {
        format!("https://{trimmed}")
    };

    match reqwest::Url::parse(&normalized) {
        Ok(url) if matches!(url.host_str(), Some("fal.run" | "queue.fal.run")) => {
            format!("{}://{}", url.scheme(), url.host_str().unwrap_or("fal.run"))
        }
        Ok(url) => url.to_string().trim_end_matches('/').to_string(),
        Err(_) => normalized,
    }
}

fn resolve_fal_queue_host(api_host: &str) -> String {
    let normalized = normalize_fal_api_host(api_host);
    match reqwest::Url::parse(&normalized) {
        Ok(url) if url.host_str() == Some("queue.fal.run") => {
            format!("{}://queue.fal.run", url.scheme())
        }
        Ok(url) if url.host_str() == Some("fal.run") => format!("{}://queue.fal.run", url.scheme()),
        Ok(url) => url.to_string().trim_end_matches('/').to_string(),
        _ => FAL_QUEUE_DEFAULT_HOST.to_string(),
    }
}

const FAL_SUPPORTED_ASPECT_RATIOS: [(&str, f64); 10] = [
    ("21:9", 21.0 / 9.0),
    ("16:9", 16.0 / 9.0),
    ("3:2", 3.0 / 2.0),
    ("4:3", 4.0 / 3.0),
    ("5:4", 5.0 / 4.0),
    ("1:1", 1.0),
    ("4:5", 4.0 / 5.0),
    ("3:4", 3.0 / 4.0),
    ("2:3", 2.0 / 3.0),
    ("9:16", 9.0 / 16.0),
];

fn size_to_aspect_ratio(size: &str) -> Option<String> {
    let (width_raw, height_raw) = size.split_once('x')?;
    let width = width_raw.parse::<u32>().ok()?;
    let height = height_raw.parse::<u32>().ok()?;
    if width == 0 || height == 0 {
        return None;
    }

    let gcd = greatest_common_divisor(width, height);
    let exact_ratio = format!("{}:{}", width / gcd, height / gcd);
    if FAL_SUPPORTED_ASPECT_RATIOS
        .iter()
        .any(|(label, _)| *label == exact_ratio)
    {
        return Some(exact_ratio);
    }

    let numeric_ratio = width as f64 / height as f64;
    let nearest = FAL_SUPPORTED_ASPECT_RATIOS
        .iter()
        .map(|(label, ratio)| (*label, (numeric_ratio - ratio).abs()))
        .min_by(|left, right| left.1.total_cmp(&right.1));

    match nearest {
        Some((label, diff)) if diff <= 0.08 => Some(label.to_string()),
        Some(_) => Some("auto".to_string()),
        None => None,
    }
}

fn greatest_common_divisor(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let temp = right;
        right = left % right;
        left = temp;
    }
    left.max(1)
}

fn build_fal_payload(prompt: &str, size: &str, count: u32, reference_images: &[String]) -> Value {
    let mut payload = json!({
        "prompt": prompt.trim(),
        "num_images": count.max(1),
        "output_format": "png",
        "safety_tolerance": "4",
    });

    if let Some(aspect_ratio) = size_to_aspect_ratio(size) {
        payload["aspect_ratio"] = Value::String(aspect_ratio);
    }

    let cleaned_references: Vec<String> = reference_images
        .iter()
        .filter_map(|image| {
            let trimmed = image.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        })
        .collect();

    if !cleaned_references.is_empty() {
        payload["image_urls"] = Value::Array(
            cleaned_references
                .iter()
                .map(|image| Value::String(image.clone()))
                .collect(),
        );
        payload["image_url"] = Value::String(cleaned_references[0].clone());
    }

    payload
}

async fn request_openai_compatible_images(
    client: &Client,
    provider: &ApiKeyProvider,
    api_host: &str,
    api_key: &str,
    request: &ImageGenerationRequest,
    model: &str,
    request_size: &str,
    use_edit_endpoint: bool,
) -> Result<ImageGenerationResponse, String> {
    if should_prefer_openai_responses_image_api(provider) {
        return request_openai_responses_images(
            client,
            provider,
            api_host,
            api_key,
            request,
            model,
            request_size,
        )
        .await
        .map_err(|error| error.message);
    }

    let endpoint = if use_edit_endpoint {
        build_openai_image_endpoint_url(api_host, "/images/edits")
    } else {
        build_openai_images_url(api_host)
    };
    let payload =
        build_openai_image_request_payload(request, model, request_size, use_edit_endpoint);
    let response = client
        .post(&endpoint)
        .header("Authorization", format!("Bearer {api_key}"))
        .header(CONTENT_TYPE, "application/json")
        .json(&payload)
        .send()
        .await
        .map_err(|error| format!("OpenAI 图片接口请求失败: {error}"))?;

    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| format!("OpenAI 图片接口响应读取失败: {error}"))?;

    if !status.is_success() {
        let image_error = OpenAiImageEndpointError {
            is_endpoint_not_found: is_not_found_status(status, &body),
            message: format!(
                "OpenAI 图片接口 HTTP {}: {}",
                status.as_u16(),
                summarize_image_service_error_body(&body)
            ),
        };

        if image_error.is_endpoint_not_found {
            match request_openai_responses_images(
                client,
                provider,
                api_host,
                api_key,
                request,
                model,
                request_size,
            )
            .await
            {
                Ok(response) => return Ok(response),
                Err(responses_error) => {
                    return Err(format!(
                        "{}；回退 Responses 后仍失败: {}",
                        image_error.message, responses_error.message
                    ))
                }
            }
        }

        return Err(image_error.message);
    }

    let payload = serde_json::from_str::<Value>(&body).map_err(|error| {
        format!(
            "OpenAI 图片接口 JSON 解析失败: {error}; body={}",
            preview_text(&body, 240)
        )
    })?;

    normalize_openai_compatible_image_response(client, &payload, &request.response_format).await
}

async fn normalize_openai_compatible_image_response(
    client: &Client,
    payload: &Value,
    response_format: &str,
) -> Result<ImageGenerationResponse, String> {
    let mut data = Vec::new();

    if let Some(items) = payload.get("data").and_then(Value::as_array) {
        for item in items {
            let revised_prompt = item
                .get("revised_prompt")
                .and_then(Value::as_str)
                .map(|value| value.trim())
                .filter(|value| !value.is_empty())
                .map(|value| value.to_string());
            let b64_json = item
                .get("b64_json")
                .and_then(Value::as_str)
                .map(|value| value.trim())
                .filter(|value| !value.is_empty())
                .map(|value| value.to_string());
            let image_url = item
                .get("url")
                .and_then(Value::as_str)
                .map(|value| value.trim())
                .filter(|value| !value.is_empty())
                .map(|value| value.to_string());

            if response_format == "b64_json" {
                if let Some(b64_json) = b64_json.clone() {
                    data.push(ImageData {
                        b64_json: Some(b64_json),
                        url: None,
                        revised_prompt,
                    });
                    continue;
                }

                if let Some(image_url) = image_url.clone() {
                    match download_image_as_base64(client, &image_url).await {
                        Ok(b64_json) => {
                            data.push(ImageData {
                                b64_json: Some(b64_json),
                                url: None,
                                revised_prompt,
                            });
                        }
                        Err(error) => {
                            tracing::warn!(
                                "[IMAGE] OpenAI 图片 URL 转 base64 失败，回退原始 URL: url={}, error={}",
                                image_url,
                                error
                            );
                            data.push(ImageData {
                                b64_json: None,
                                url: Some(image_url),
                                revised_prompt,
                            });
                        }
                    }
                    continue;
                }
            } else {
                if let Some(image_url) = image_url {
                    data.push(ImageData {
                        b64_json: None,
                        url: Some(image_url),
                        revised_prompt,
                    });
                    continue;
                }

                if let Some(b64_json) = b64_json {
                    data.push(ImageData {
                        b64_json: None,
                        url: Some(format!("data:image/png;base64,{b64_json}")),
                        revised_prompt,
                    });
                    continue;
                }
            }
        }
    }

    if !data.is_empty() {
        return Ok(ImageGenerationResponse {
            created: chrono::Utc::now().timestamp(),
            data,
        });
    }

    let image_urls = collect_image_urls(payload);
    if !image_urls.is_empty() {
        return build_openai_response(client, &image_urls, response_format, None).await;
    }

    Err("默认图片服务未返回可解析图片字段".to_string())
}

async fn request_fal_images(
    client: &Client,
    api_host: &str,
    api_key: &str,
    prompt: &str,
    model: &str,
    size: &str,
    count: u32,
    reference_images: &[String],
) -> Result<Vec<String>, String> {
    let endpoint_model = model.trim().trim_start_matches('/');
    let sync_endpoint = format!(
        "{}/{}",
        normalize_fal_api_host(api_host).trim_end_matches('/'),
        endpoint_model
    );
    let payload = build_fal_payload(prompt, size, count, reference_images);
    let sync_result = post_fal_json(client, &sync_endpoint, &payload, api_key).await;

    if let Ok(sync_response) = &sync_result {
        let urls = collect_image_urls(sync_response);
        if !urls.is_empty() {
            return Ok(urls.into_iter().take(count as usize).collect());
        }
    }

    let queue_error_context = sync_result
        .err()
        .unwrap_or_else(|| "Fal 同步接口未返回可解析图片，改走队列模式".to_string());
    tracing::warn!("[IMAGE] {}", queue_error_context);

    request_fal_queue_images(client, api_host, api_key, endpoint_model, &payload, count)
        .await
        .map_err(|queue_error| format!("{queue_error}; sync_context={queue_error_context}"))
}

async fn post_fal_json(
    client: &Client,
    endpoint: &str,
    payload: &Value,
    api_key: &str,
) -> Result<Value, String> {
    let response = client
        .post(endpoint)
        .header("Authorization", format!("Key {api_key}"))
        .json(payload)
        .send()
        .await
        .map_err(|error| format!("Fal 请求失败: {error}"))?;

    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| format!("Fal 响应读取失败: {error}"))?;

    if !status.is_success() {
        return Err(format!(
            "Fal HTTP {}: {}",
            status.as_u16(),
            summarize_image_service_error_body(&body)
        ));
    }

    serde_json::from_str::<Value>(&body).map_err(|error| {
        format!(
            "Fal JSON 解析失败: {error}; body={}",
            preview_text(&body, 240)
        )
    })
}

async fn request_fal_queue_images(
    client: &Client,
    api_host: &str,
    api_key: &str,
    endpoint_model: &str,
    payload: &Value,
    count: u32,
) -> Result<Vec<String>, String> {
    request_fal_queue_images_with_options(
        client,
        api_host,
        api_key,
        endpoint_model,
        payload,
        count,
        Duration::from_secs(FAL_QUEUE_TIMEOUT_SECS),
        Duration::from_millis(FAL_QUEUE_POLL_INTERVAL_MS),
    )
    .await
}

async fn request_fal_queue_images_with_options(
    client: &Client,
    api_host: &str,
    api_key: &str,
    endpoint_model: &str,
    payload: &Value,
    count: u32,
    queue_timeout: Duration,
    poll_interval: Duration,
) -> Result<Vec<String>, String> {
    let queue_endpoint = format!(
        "{}/{}",
        resolve_fal_queue_host(api_host).trim_end_matches('/'),
        endpoint_model
    );
    let submit = post_fal_json(client, &queue_endpoint, payload, api_key).await?;
    let mut urls = collect_image_urls(&submit);
    if !urls.is_empty() {
        return Ok(urls.into_iter().take(count as usize).collect());
    }

    let request_id = submit
        .get("request_id")
        .and_then(Value::as_str)
        .map(str::to_string);
    let mut status_url = submit
        .get("status_url")
        .and_then(Value::as_str)
        .map(str::to_string);
    let mut response_url = submit
        .get("response_url")
        .and_then(Value::as_str)
        .map(str::to_string);

    if let Some(request_id) = request_id {
        let request_base = format!(
            "{queue_endpoint}/requests/{}",
            urlencoding::encode(&request_id)
        );
        if status_url.is_none() {
            status_url = Some(format!("{request_base}/status"));
        }
        if response_url.is_none() {
            response_url = Some(format!("{request_base}/response"));
        }
    }

    let Some(status_url) = status_url else {
        return Err("Fal 队列提交成功，但缺少 status_url".to_string());
    };

    let deadline = tokio::time::Instant::now() + queue_timeout;
    let mut last_status_payload: Option<Value> = None;
    let mut completed = false;
    while tokio::time::Instant::now() < deadline {
        let status_payload = get_fal_json(client, &status_url, api_key).await?;
        if let Some(next_response_url) = status_payload
            .get("response_url")
            .and_then(Value::as_str)
            .map(str::to_string)
        {
            response_url = Some(next_response_url);
        }

        let status = status_payload
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_ascii_uppercase();

        if status == "COMPLETED" {
            completed = true;
            break;
        }

        if matches!(status.as_str(), "FAILED" | "ERROR" | "CANCELLED") {
            return Err(format!(
                "Fal 队列任务失败: {}",
                preview_text(&status_payload.to_string(), 240)
            ));
        }

        last_status_payload = Some(status_payload);
        tokio::time::sleep(poll_interval).await;
    }

    if !completed {
        let status_summary = last_status_payload
            .map(|value| preview_text(&value.to_string(), 240))
            .unwrap_or_else(|| "未收到 Fal 队列状态响应".to_string());
        return Err(format!(
            "Fal 队列任务超过 {} 秒仍未完成: {}",
            queue_timeout.as_secs(),
            status_summary
        ));
    }

    let Some(response_url) = response_url else {
        return Err("Fal 队列完成后缺少 response_url".to_string());
    };

    let result_payload = get_fal_json(client, &response_url, api_key).await?;
    urls = collect_image_urls(&result_payload);
    if urls.is_empty() {
        return Err(format!(
            "Fal 队列结果中未找到图片地址: {}",
            preview_text(&result_payload.to_string(), 240)
        ));
    }

    Ok(urls.into_iter().take(count as usize).collect())
}

async fn get_fal_json(client: &Client, endpoint: &str, api_key: &str) -> Result<Value, String> {
    let response = client
        .get(endpoint)
        .header("Authorization", format!("Key {api_key}"))
        .send()
        .await
        .map_err(|error| format!("Fal GET 请求失败: {error}"))?;

    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| format!("Fal GET 响应读取失败: {error}"))?;

    if !status.is_success() {
        return Err(format!(
            "Fal GET HTTP {}: {}",
            status.as_u16(),
            summarize_image_service_error_body(&body)
        ));
    }

    serde_json::from_str::<Value>(&body).map_err(|error| {
        format!(
            "Fal GET JSON 解析失败: {error}; body={}",
            preview_text(&body, 240)
        )
    })
}

async fn build_openai_response(
    client: &Client,
    image_urls: &[String],
    response_format: &str,
    created_at: Option<i64>,
) -> Result<ImageGenerationResponse, String> {
    if image_urls.is_empty() {
        return Err("默认图片服务未返回图片地址".to_string());
    }

    let mut data = Vec::with_capacity(image_urls.len());
    for image_url in image_urls {
        if response_format == "b64_json" {
            match download_image_as_base64(client, image_url).await {
                Ok(b64_json) => data.push(ImageData {
                    b64_json: Some(b64_json),
                    url: None,
                    revised_prompt: None,
                }),
                Err(error) => {
                    tracing::warn!(
                        "[IMAGE] 图片二次下载失败，回退原始 URL: url={}, error={}",
                        image_url,
                        error
                    );
                    data.push(ImageData {
                        b64_json: None,
                        url: Some(image_url.clone()),
                        revised_prompt: None,
                    });
                }
            }
        } else {
            data.push(ImageData {
                b64_json: None,
                url: Some(image_url.clone()),
                revised_prompt: None,
            });
        }
    }

    Ok(ImageGenerationResponse {
        created: created_at.unwrap_or_else(|| chrono::Utc::now().timestamp()),
        data,
    })
}

async fn download_image_as_base64(client: &Client, image_url: &str) -> Result<String, String> {
    let response = client
        .get(image_url)
        .send()
        .await
        .map_err(|error| format!("下载图片失败: {error}"))?;

    let status = response.status();
    if !status.is_success() {
        return Err(format!("下载图片失败: HTTP {}", status.as_u16()));
    }

    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("image/png")
        .to_string();
    let bytes = response
        .bytes()
        .await
        .map_err(|error| format!("读取图片字节失败: {error}"))?;
    tracing::debug!(
        "[IMAGE] 默认图片服务下载成功: content_type={}, bytes={}",
        content_type,
        bytes.len()
    );
    Ok(BASE64.encode(bytes))
}

fn collect_image_urls(value: &Value) -> Vec<String> {
    let mut urls = Vec::new();
    collect_image_urls_inner(value, None, &mut urls);
    urls
}

fn should_skip_control_url_key(key: &str) -> bool {
    matches!(
        key,
        "status_url" | "statusUrl" | "response_url" | "responseUrl" | "cancel_url" | "cancelUrl"
    )
}

fn collect_image_urls_inner(value: &Value, parent_key: Option<&str>, urls: &mut Vec<String>) {
    match value {
        Value::String(text) => {
            if parent_key.is_some_and(should_skip_control_url_key) {
                return;
            }
            let trimmed = text.trim();
            if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
                push_unique(urls, trimmed.to_string());
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_image_urls_inner(item, None, urls);
            }
        }
        Value::Object(map) => {
            for key in [
                "url",
                "uri",
                "href",
                "image",
                "image_url",
                "imageUrl",
                "image_uri",
                "imageUri",
                "file_url",
                "fileUrl",
                "download_url",
                "downloadUrl",
            ] {
                if let Some(Value::String(url)) = map.get(key) {
                    let trimmed = url.trim();
                    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
                        push_unique(urls, trimmed.to_string());
                    }
                }
            }

            for (key, nested) in map {
                collect_image_urls_inner(nested, Some(key.as_str()), urls);
            }
        }
        _ => {}
    }
}

fn push_unique(urls: &mut Vec<String>, candidate: String) {
    if !urls.iter().any(|existing| existing == &candidate) {
        urls.push(candidate);
    }
}

fn summarize_image_service_error_body(body: &str) -> String {
    let normalized = body.trim();
    if normalized.is_empty() {
        return "图片服务返回了空响应。".to_string();
    }

    if normalized.contains("aspect_ratio") {
        return "当前图片服务不支持这个画幅比例，请改用 21:9、16:9、3:2、4:3、5:4、1:1、4:5、3:4、2:3 或 9:16。".to_string();
    }

    if let Ok(parsed) = serde_json::from_str::<Value>(normalized) {
        if let Some(message) = parsed
            .get("message")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return message.to_string();
        }

        if let Some(message) = parsed
            .get("error")
            .and_then(Value::as_object)
            .and_then(|error| error.get("message"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return message.to_string();
        }

        if let Some(message) = parsed
            .get("detail")
            .and_then(Value::as_array)
            .and_then(|items| items.first())
            .and_then(Value::as_object)
            .and_then(|record| record.get("msg"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return message.to_string();
        }
    }

    preview_text(normalized, 240)
}

fn preview_text(text: &str, max_len: usize) -> String {
    let normalized = text.trim();
    let normalized_chars = normalized.chars().count();
    if normalized_chars <= max_len {
        return normalized.to_string();
    }
    let preview: String = normalized.chars().take(max_len).collect();
    format!("{preview}...")
}

#[cfg(test)]
mod tests {
    use super::provider_routing::{
        is_gemini_image_provider, supports_openai_compatible_image_provider,
    };
    use super::{
        build_fal_payload, build_openai_image_request_payload, build_openai_images_url,
        build_openai_responses_image_input, build_openai_responses_image_request_payload,
        build_zhipu_image_request_payload, build_zhipu_images_url, collect_image_urls,
        load_image_provider_routing, normalize_fal_api_host,
        normalize_openai_compatible_image_response, normalize_openai_responses_image_payload,
        normalize_openai_responses_image_sse, normalize_openai_responses_image_tool_model,
        request_fal_queue_images_with_options, request_openai_responses_images,
        request_zhipu_images, resolve_compatible_image_model,
        resolve_configured_image_provider_kind, resolve_fal_model,
        resolve_openai_responses_image_orchestration_model, resolve_zhipu_image_model,
        should_prefer_openai_responses_image_api, size_to_aspect_ratio,
        try_extract_partial_openai_responses_images, ConfiguredImageProviderKind,
        ImageProviderRoutingConfig,
    };
    use async_stream::stream;
    use axum::{
        response::sse::{Event, Sse},
        routing::post,
        Router,
    };
    use chrono::Utc;
    use lime_core::database::dao::api_key_provider::{
        ApiKeyProvider, ApiProviderType, ProviderGroup,
    };
    use lime_core::image_generation_matcher::is_likely_image_generation_model_id;
    use lime_core::models::model_registry::ProviderModelConfig;
    use lime_core::models::openai::ImageGenerationRequest;
    use reqwest::Client;
    use serde_json::{json, Value};
    use std::sync::{Arc, Mutex};
    use std::{convert::Infallible, time::Duration};
    use tokio::net::TcpListener;

    #[test]
    fn normalize_fal_api_host_strips_builtin_path_suffix() {
        assert_eq!(
            normalize_fal_api_host("https://fal.run/fal-ai"),
            "https://fal.run"
        );
        assert_eq!(
            normalize_fal_api_host("https://queue.fal.run/fal-ai"),
            "https://queue.fal.run"
        );
    }

    #[test]
    fn resolve_fal_model_prefers_config_when_request_uses_openai_default() {
        assert_eq!(
            resolve_fal_model("dall-e-3", Some("fal-ai/nano-banana-pro")),
            "fal-ai/nano-banana-pro"
        );
        assert_eq!(
            resolve_fal_model("nano-banana-pro", None),
            "fal-ai/nano-banana-pro"
        );
    }

    #[test]
    fn size_to_aspect_ratio_maps_to_supported_fal_values() {
        assert_eq!(size_to_aspect_ratio("1024x1024"), Some("1:1".to_string()));
        assert_eq!(size_to_aspect_ratio("1792x1024"), Some("16:9".to_string()));
        assert_eq!(size_to_aspect_ratio("1024x1792"), Some("9:16".to_string()));
        assert_eq!(size_to_aspect_ratio("1000x100"), Some("auto".to_string()));
        assert_eq!(size_to_aspect_ratio("invalid"), None);
    }

    #[test]
    fn collect_image_urls_reads_nested_payloads() {
        let payload = json!({
            "images": [
                { "url": "https://example.com/a.png" },
                { "imageUrl": "https://example.com/b.png" }
            ],
            "nested": {
                "downloadUrl": "https://example.com/c.png"
            }
        });

        assert_eq!(
            collect_image_urls(&payload),
            vec![
                "https://example.com/a.png".to_string(),
                "https://example.com/b.png".to_string(),
                "https://example.com/c.png".to_string(),
            ]
        );
    }

    #[test]
    fn collect_image_urls_ignores_fal_queue_control_urls() {
        let payload = json!({
            "status_url": "https://queue.fal.run/fal-ai/nano-banana/requests/req-1/status",
            "response_url": "https://queue.fal.run/fal-ai/nano-banana/requests/req-1/response",
            "cancel_url": "https://queue.fal.run/fal-ai/nano-banana/requests/req-1/cancel",
            "data": [
                {
                    "url": "https://cdn.example.com/final-image.png"
                }
            ]
        });

        assert_eq!(
            collect_image_urls(&payload),
            vec!["https://cdn.example.com/final-image.png".to_string()]
        );
    }

    #[tokio::test]
    async fn request_fal_queue_images_should_fail_when_queue_times_out() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind fal queue test app");
        let address = listener.local_addr().expect("resolve address");
        let server = tokio::spawn(async move {
            let app = Router::new()
                .route(
                    "/fal-ai/nano-banana-pro",
                    post(|| async move {
                        axum::Json(json!({
                            "request_id": "req-timeout",
                        }))
                    }),
                )
                .route(
                    "/fal-ai/nano-banana-pro/requests/req-timeout/status",
                    axum::routing::get(|| async move { axum::Json(json!({"status": "IN_QUEUE"})) }),
                );
            axum::serve(listener, app)
                .await
                .expect("serve fal queue test app");
        });

        let client = Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(2))
            .build()
            .expect("build client");
        let result = request_fal_queue_images_with_options(
            &client,
            &format!("http://{address}"),
            "test-key",
            "fal-ai/nano-banana-pro",
            &json!({"prompt": "青柠"}),
            1,
            Duration::from_millis(25),
            Duration::from_millis(1),
        )
        .await;

        let error = result.expect_err("queue timeout should fail");
        assert!(
            error.contains("Fal 队列任务超过"),
            "unexpected queue timeout error: {error}"
        );
        assert!(
            error.contains("IN_QUEUE"),
            "unexpected queue status error: {error}"
        );

        server.abort();
    }

    #[test]
    fn load_image_provider_routing_prefers_explicit_provider_without_fallback() {
        let routing = load_image_provider_routing(Some("fal")).expect("explicit provider routing");

        assert_eq!(routing.provider_id, "fal");
        assert_eq!(routing.preferred_model_id, None);
        assert_eq!(routing.default_size, None);
        assert!(!routing.allow_fallback);
        assert!(routing.is_explicit);
    }

    #[test]
    fn resolve_compatible_image_model_prefers_request_then_provider_defaults() {
        assert_eq!(
            resolve_compatible_image_model("gpt-images-2", Some("dall-e-3"), &[]),
            Some("gpt-images-2".to_string())
        );
        assert_eq!(
            resolve_compatible_image_model("gpt-5.2", Some("gpt-images-2"), &[]),
            Some("gpt-images-2".to_string())
        );
        assert_eq!(
            resolve_compatible_image_model(
                "gpt-5.2",
                None,
                &[
                    ProviderModelConfig::hint("gpt-images-2"),
                    ProviderModelConfig::hint("gpt-5.2"),
                ],
            ),
            Some("gpt-images-2".to_string())
        );
        assert_eq!(
            resolve_compatible_image_model("claude-sonnet-4-5", Some("cogview-3-flash"), &[]),
            Some("cogview-3-flash".to_string())
        );
        assert_eq!(
            resolve_compatible_image_model(
                "claude-sonnet-4-5",
                None,
                &[
                    ProviderModelConfig::hint("black-forest-labs/FLUX.1-schnell"),
                    ProviderModelConfig::hint("glm-5"),
                ],
            ),
            Some("black-forest-labs/FLUX.1-schnell".to_string())
        );
        assert_eq!(resolve_compatible_image_model("gpt-5.2", None, &[]), None);
    }

    #[test]
    fn provider_kind_should_route_catalog_image_providers_to_current_adapters() {
        fn provider(
            id: &str,
            provider_type: ApiProviderType,
            api_host: &str,
            models: Vec<ProviderModelConfig>,
        ) -> ApiKeyProvider {
            ApiKeyProvider {
                id: id.to_string(),
                name: id.to_string(),
                provider_type,
                api_host: api_host.to_string(),
                is_system: false,
                group: ProviderGroup::Custom,
                enabled: true,
                sort_order: 100,
                api_version: None,
                project: None,
                location: None,
                region: None,
                models,
                prompt_cache_mode: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            }
        }

        let request = ImageGenerationRequest {
            model: "gemini-3.1-flash-image".to_string(),
            prompt: "青柠".to_string(),
            n: 1,
            size: Some("1024x1024".to_string()),
            response_format: "url".to_string(),
            quality: None,
            style: None,
            reference_images: vec![],
            user: None,
        };
        let routing = ImageProviderRoutingConfig {
            provider_id: "gemini".to_string(),
            preferred_model_id: None,
            allow_fallback: false,
            default_size: None,
            is_explicit: true,
        };
        let gemini_provider = provider(
            "gemini",
            ApiProviderType::Gemini,
            "https://generativelanguage.googleapis.com",
            vec![ProviderModelConfig::hint("gemini-3.1-flash-image")],
        );

        assert!(is_gemini_image_provider(&gemini_provider));
        assert!(matches!(
            resolve_configured_image_provider_kind(&gemini_provider, &request, &routing),
            Some(ConfiguredImageProviderKind::Gemini)
        ));
        assert!(!supports_openai_compatible_image_provider(&gemini_provider));
        let zhipu_provider = provider(
            "zhipuai",
            ApiProviderType::Openai,
            "https://open.bigmodel.cn/api/paas/v4",
            vec![ProviderModelConfig::hint("cogview-4-250304")],
        );
        assert!(matches!(
            resolve_configured_image_provider_kind(&zhipu_provider, &request, &routing),
            Some(ConfiguredImageProviderKind::Zhipu)
        ));
        assert!(!supports_openai_compatible_image_provider(&zhipu_provider));
        assert!(supports_openai_compatible_image_provider(&provider(
            "siliconflow",
            ApiProviderType::Openai,
            "https://api.siliconflow.cn/v1",
            vec![ProviderModelConfig::hint(
                "black-forest-labs/FLUX.1-schnell",
            )],
        )));
        assert!(matches!(
            resolve_configured_image_provider_kind(
                &provider(
                    "google-vertex",
                    ApiProviderType::Vertexai,
                    "",
                    vec![ProviderModelConfig::hint("gemini-3.1-flash-image")],
                ),
                &request,
                &routing,
            ),
            Some(ConfiguredImageProviderKind::Gemini)
        ));
    }

    #[test]
    fn zhipu_image_model_resolution_prefers_native_models() {
        assert_eq!(
            resolve_zhipu_image_model(
                "gpt-images-2",
                Some("cogview-4-250304"),
                &[ProviderModelConfig::hint("glm-image")]
            ),
            Some("cogview-4-250304".to_string())
        );
        assert_eq!(
            resolve_zhipu_image_model("glm-image", None, &[]),
            Some("glm-image".to_string())
        );
        assert_eq!(
            resolve_zhipu_image_model("gpt-5.2", None, &[ProviderModelConfig::hint("glm-image")],),
            Some("glm-image".to_string())
        );
    }

    #[test]
    fn build_zhipu_images_url_uses_official_sync_endpoint() {
        assert_eq!(
            build_zhipu_images_url("https://open.bigmodel.cn"),
            "https://open.bigmodel.cn/api/paas/v4/images/generations"
        );
        assert_eq!(
            build_zhipu_images_url("https://open.bigmodel.cn/api/paas/v4"),
            "https://open.bigmodel.cn/api/paas/v4/images/generations"
        );
    }

    #[test]
    fn build_zhipu_image_request_payload_defaults_glm_image_quality_to_hd() {
        let request = ImageGenerationRequest {
            model: "glm-image".to_string(),
            prompt: "生成一个苹果".to_string(),
            n: 1,
            size: None,
            response_format: "url".to_string(),
            quality: None,
            style: None,
            reference_images: vec![],
            user: None,
        };

        let payload = build_zhipu_image_request_payload(&request, "glm-image", "");

        assert_eq!(payload["quality"].as_str(), Some("hd"));
        assert_eq!(payload["size"].as_str(), Some("1280x1280"));
    }

    #[tokio::test]
    async fn request_zhipu_images_sends_native_payload_and_normalizes_url_response() {
        let captured_body: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
        let captured_body_for_handler: Arc<Mutex<Option<Value>>> = Arc::clone(&captured_body);

        let app = Router::new().route(
            "/api/paas/v4/images/generations",
            post(
                move |headers: axum::http::HeaderMap, axum::Json(body): axum::Json<Value>| {
                    let captured_body = Arc::clone(&captured_body_for_handler);
                    async move {
                        assert_eq!(
                            headers
                                .get("authorization")
                                .and_then(|value| value.to_str().ok()),
                            Some("Bearer test-key")
                        );
                        *captured_body.lock().expect("capture body mutex") = Some(body);
                        axum::Json(json!({
                            "created": 1_777_000_002i64,
                            "data": [
                                {
                                    "url": "https://cdn.example.com/zhipu.png"
                                }
                            ]
                        }))
                    }
                },
            ),
        );

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind listener");
        let addr = listener.local_addr().expect("listener addr");
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve test app");
        });

        let client = Client::builder().no_proxy().build().expect("client");
        let request = ImageGenerationRequest {
            model: "glm-image".to_string(),
            prompt: "生成一个苹果".to_string(),
            n: 3,
            size: None,
            response_format: "url".to_string(),
            quality: Some("standard".to_string()),
            style: None,
            reference_images: vec![],
            user: Some("user-123456".to_string()),
        };

        let response = request_zhipu_images(
            &client,
            &format!("http://{addr}/api/paas/v4"),
            "test-key",
            &request,
            "glm-image",
            "",
        )
        .await
        .expect("zhipu image request");

        assert_eq!(response.created, 1_777_000_002);
        assert_eq!(
            response.data[0].url.as_deref(),
            Some("https://cdn.example.com/zhipu.png")
        );
        assert_eq!(response.data[0].b64_json, None);
        assert_eq!(
            captured_body
                .lock()
                .expect("capture body mutex")
                .clone()
                .expect("captured body"),
            json!({
                "model": "glm-image",
                "prompt": "生成一个苹果",
                "size": "1280x1280",
                "quality": "hd",
                "user_id": "user-123456",
            })
        );
    }

    #[test]
    fn build_openai_images_url_reuses_existing_version_path() {
        assert_eq!(
            build_openai_images_url("https://airgate.k8ray.com/v1"),
            "https://airgate.k8ray.com/v1/images/generations"
        );
        assert_eq!(
            build_openai_images_url("https://api.openai.com"),
            "https://api.openai.com/v1/images/generations"
        );
    }

    #[tokio::test]
    async fn normalize_openai_compatible_image_response_supports_b64_and_url_modes() {
        let client = Client::builder().no_proxy().build().expect("client");
        let payload = json!({
            "data": [
                {
                    "b64_json": "dGVzdA==",
                    "revised_prompt": "refined prompt"
                }
            ]
        });

        let b64_response =
            normalize_openai_compatible_image_response(&client, &payload, "b64_json")
                .await
                .expect("b64 response");
        assert_eq!(b64_response.data.len(), 1);
        assert_eq!(b64_response.data[0].b64_json.as_deref(), Some("dGVzdA=="));
        assert_eq!(b64_response.data[0].url, None);
        assert_eq!(
            b64_response.data[0].revised_prompt.as_deref(),
            Some("refined prompt")
        );

        let url_response = normalize_openai_compatible_image_response(&client, &payload, "url")
            .await
            .expect("url response");
        assert_eq!(
            url_response.data[0].url.as_deref(),
            Some("data:image/png;base64,dGVzdA==")
        );
        assert_eq!(url_response.data[0].b64_json, None);
    }

    #[test]
    fn normalize_openai_responses_image_payload_supports_b64_and_url_modes() {
        let payload = json!({
            "created_at": 1_777_000_000i64,
            "output": [
                {
                    "type": "image_generation_call",
                    "result": "dGVzdA==",
                    "revised_prompt": "refined prompt",
                    "output_format": "png"
                }
            ]
        });

        let b64_response =
            normalize_openai_responses_image_payload(&payload, "b64_json").expect("b64 response");
        assert_eq!(b64_response.created, 1_777_000_000);
        assert_eq!(b64_response.data[0].b64_json.as_deref(), Some("dGVzdA=="));
        assert_eq!(b64_response.data[0].url, None);

        let url_response =
            normalize_openai_responses_image_payload(&payload, "url").expect("url response");
        assert_eq!(
            url_response.data[0].url.as_deref(),
            Some("data:image/png;base64,dGVzdA==")
        );
        assert_eq!(url_response.data[0].b64_json, None);
        assert_eq!(
            url_response.data[0].revised_prompt.as_deref(),
            Some("refined prompt")
        );
    }

    #[test]
    fn normalize_openai_responses_image_sse_reads_partial_image_events() {
        let sse = r#"
event: response.image_generation_call.partial_image
data: {"type":"response.image_generation_call.partial_image","output_index":0,"partial_image_b64":"dGVzdA==","revised_prompt":"refined prompt","output_format":"png"}

event: response.completed
data: {"type":"response.completed","response":{"created_at":1777000000,"output":[]}}
"#;

        let response =
            normalize_openai_responses_image_sse(sse, "b64_json").expect("sse image response");
        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0].b64_json.as_deref(), Some("dGVzdA=="));
        assert_eq!(
            response.data[0].revised_prompt.as_deref(),
            Some("refined prompt")
        );
    }

    #[test]
    fn normalize_openai_responses_image_sse_reads_multiple_done_events() {
        let sse = r#"
event: response.output_item.done
data: {"type":"response.output_item.done","output_index":0,"item":{"type":"image_generation_call","result":"Zmlyc3Q=","revised_prompt":"first prompt","output_format":"png"}}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":1,"item":{"type":"image_generation_call","result":"c2Vjb25k","revised_prompt":"second prompt","output_format":"png"}}

event: response.completed
data: {"type":"response.completed","response":{"created_at":1777000001,"output":[]}}
"#;

        let response =
            normalize_openai_responses_image_sse(sse, "b64_json").expect("sse image response");
        assert_eq!(response.data.len(), 2);
        assert_eq!(response.data[0].b64_json.as_deref(), Some("Zmlyc3Q="));
        assert_eq!(response.data[1].b64_json.as_deref(), Some("c2Vjb25k"));
        assert_eq!(
            response.data[0].revised_prompt.as_deref(),
            Some("first prompt")
        );
        assert_eq!(
            response.data[1].revised_prompt.as_deref(),
            Some("second prompt")
        );
    }

    #[test]
    fn normalize_openai_responses_image_sse_uses_event_name_when_json_type_missing() {
        let sse = r#"
event: response.output_item.done
data: {"output_index":0,"item":{"type":"image_generation_call","result":"Zmlyc3Q=","revised_prompt":"first prompt","output_format":"png"}}

event: response.completed
data: {"response":{"created_at":1777000001,"output":[]}}
"#;

        let response =
            normalize_openai_responses_image_sse(sse, "b64_json").expect("sse image response");
        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0].b64_json.as_deref(), Some("Zmlyc3Q="));
        assert_eq!(
            response.data[0].revised_prompt.as_deref(),
            Some("first prompt")
        );
    }

    #[test]
    fn normalize_openai_responses_image_sse_merges_multiline_data_blocks() {
        let sse = r#"
event: response.output_item.done
data: {"output_index":0,
data: "item":{"type":"image_generation_call","result":"Zmlyc3Q=","revised_prompt":"first prompt","output_format":"png"}}

event: response.completed
data: {"response":{"created_at":1777000001,"output":[]}}
"#;

        let response =
            normalize_openai_responses_image_sse(sse, "b64_json").expect("sse image response");
        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0].b64_json.as_deref(), Some("Zmlyc3Q="));
    }

    #[test]
    fn normalize_openai_responses_image_tool_model_rewrites_gpt_images_alias() {
        assert_eq!(
            normalize_openai_responses_image_tool_model("gpt-images-2"),
            "gpt-image-2"
        );
        assert_eq!(
            normalize_openai_responses_image_tool_model("gpt-image-1"),
            "gpt-image-1"
        );
    }

    #[test]
    fn build_openai_responses_image_request_payload_uses_tool_model_and_omits_n() {
        let request = ImageGenerationRequest {
            model: "gpt-images-2".to_string(),
            prompt: "生成一个苹果".to_string(),
            n: 3,
            size: Some("1024x1024".to_string()),
            response_format: "b64_json".to_string(),
            quality: Some("high".to_string()),
            style: None,
            reference_images: vec![],
            user: None,
        };

        let payload = build_openai_responses_image_request_payload(
            &request,
            "gpt-5.4",
            "gpt-images-2",
            "1024x1024",
            false,
        );
        let tool = payload["tools"][0].as_object().expect("tool object");

        assert_eq!(payload["model"].as_str(), Some("gpt-5.4"));
        let input = payload["input"].as_str().expect("responses input");
        assert!(input.contains("恰好 3 次"));
        assert!(input.contains("每次只生成 1 张独立图片"));
        assert!(input.contains("原始创作要求：生成一个苹果"));
        assert_eq!(
            tool.get("type").and_then(|value| value.as_str()),
            Some("image_generation")
        );
        assert_eq!(
            tool.get("model").and_then(|value| value.as_str()),
            Some("gpt-image-2")
        );
        assert_eq!(
            tool.get("size").and_then(|value| value.as_str()),
            Some("1024x1024")
        );
        assert_eq!(
            tool.get("output_format").and_then(|value| value.as_str()),
            Some("png")
        );
        assert!(tool.get("n").is_none());
        assert!(tool.get("quality").is_none());
    }

    #[test]
    fn build_openai_responses_image_request_payload_keeps_plain_prompt_for_single_image() {
        let request = ImageGenerationRequest {
            model: "gpt-images-2".to_string(),
            prompt: "生成一个苹果".to_string(),
            n: 1,
            size: Some("1024x1024".to_string()),
            response_format: "b64_json".to_string(),
            quality: None,
            style: None,
            reference_images: vec![],
            user: None,
        };

        let payload = build_openai_responses_image_request_payload(
            &request,
            "gpt-5.4",
            "gpt-images-2",
            "1024x1024",
            false,
        );

        assert_eq!(payload["input"].as_str(), Some("生成一个苹果"));
    }

    #[test]
    fn build_openai_responses_image_request_payload_can_use_input_list() {
        let request = ImageGenerationRequest {
            model: "gpt-images-2".to_string(),
            prompt: "生成一个苹果".to_string(),
            n: 1,
            size: Some("1024x1024".to_string()),
            response_format: "b64_json".to_string(),
            quality: None,
            style: None,
            reference_images: vec![],
            user: None,
        };

        let payload = build_openai_responses_image_request_payload(
            &request,
            "gpt-5.4",
            "gpt-images-2",
            "1024x1024",
            true,
        );

        assert_eq!(
            payload["input"][0]["content"][0]["type"].as_str(),
            Some("input_text")
        );
        assert_eq!(
            payload["input"][0]["content"][0]["text"].as_str(),
            Some("生成一个苹果")
        );
    }

    #[test]
    fn resolve_openai_responses_image_orchestration_model_prefers_non_image_custom_model() {
        let now = Utc::now();
        let provider = ApiKeyProvider {
            id: "custom-openai-images".to_string(),
            name: "Custom".to_string(),
            provider_type: ApiProviderType::Openai,
            api_host: "https://gateway.example.com/codex".to_string(),
            is_system: false,
            group: ProviderGroup::Custom,
            enabled: true,
            sort_order: 0,
            api_version: None,
            project: None,
            location: None,
            region: None,
            models: vec![
                ProviderModelConfig::hint("gpt-images-2"),
                ProviderModelConfig::hint("gpt-5.4"),
            ],
            prompt_cache_mode: None,
            created_at: now,
            updated_at: now,
        };

        assert_eq!(
            resolve_openai_responses_image_orchestration_model(&provider, "gpt-images-2"),
            "gpt-5.4"
        );
    }

    #[test]
    fn build_openai_image_request_payload_includes_reference_images() {
        let request = ImageGenerationRequest {
            model: "gpt-image-1".to_string(),
            prompt: "编辑这张图".to_string(),
            n: 1,
            size: Some("1024x1024".to_string()),
            response_format: "url".to_string(),
            quality: Some("high".to_string()),
            style: Some("vivid".to_string()),
            reference_images: vec![
                "  https://cdn.example.com/ref-a.png  ".to_string(),
                " ".to_string(),
                "https://cdn.example.com/ref-b.png".to_string(),
            ],
            user: Some("user-1".to_string()),
        };

        let payload =
            build_openai_image_request_payload(&request, "gpt-image-1", "1024x1024", true);

        assert_eq!(
            payload["images"],
            json!([
                { "image_url": "https://cdn.example.com/ref-a.png" },
                { "image_url": "https://cdn.example.com/ref-b.png" },
            ])
        );
        assert_eq!(payload["user"].as_str(), Some("user-1"));
    }

    #[test]
    fn build_openai_responses_image_input_includes_reference_images() {
        let request = ImageGenerationRequest {
            model: "gpt-images-2".to_string(),
            prompt: "编辑这张图".to_string(),
            n: 1,
            size: Some("1024x1024".to_string()),
            response_format: "url".to_string(),
            quality: None,
            style: None,
            reference_images: vec![
                "https://cdn.example.com/ref-a.png".to_string(),
                "  ".to_string(),
                "https://cdn.example.com/ref-b.png".to_string(),
            ],
            user: None,
        };

        let payload = build_openai_responses_image_input(&request, false);
        let content = payload[0]["content"].as_array().expect("content array");

        assert_eq!(content.len(), 3);
        assert_eq!(content[0]["type"].as_str(), Some("input_text"));
        assert_eq!(content[1]["type"].as_str(), Some("input_image"));
        assert_eq!(
            content[1]["image_url"].as_str(),
            Some("https://cdn.example.com/ref-a.png")
        );
        assert_eq!(
            content[2]["image_url"].as_str(),
            Some("https://cdn.example.com/ref-b.png")
        );
    }

    #[test]
    fn build_fal_payload_includes_reference_images() {
        let payload = build_fal_payload(
            "编辑这张图",
            "1024x1024",
            1,
            &[
                "https://cdn.example.com/ref-a.png".to_string(),
                " ".to_string(),
                "https://cdn.example.com/ref-b.png".to_string(),
            ],
        );

        assert_eq!(
            payload["image_urls"],
            json!([
                "https://cdn.example.com/ref-a.png",
                "https://cdn.example.com/ref-b.png",
            ])
        );
        assert_eq!(
            payload["image_url"].as_str(),
            Some("https://cdn.example.com/ref-a.png")
        );
    }

    #[tokio::test]
    async fn request_openai_responses_images_waits_for_completed_event_before_returning() {
        async fn responses_handler() -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>>
        {
            let stream = stream! {
                yield Ok(Event::default().event("response.output_item.done").data(
                    r#"{"type":"response.output_item.done","output_index":0,"item":{"type":"image_generation_call","result":"Zmlyc3Q=","revised_prompt":"first prompt","output_format":"png"}}"#,
                ));
                tokio::time::sleep(Duration::from_millis(15)).await;
                yield Ok(Event::default().event("response.output_item.done").data(
                    r#"{"type":"response.output_item.done","output_index":1,"item":{"type":"image_generation_call","result":"c2Vjb25k","revised_prompt":"second prompt","output_format":"png"}}"#,
                ));
                tokio::time::sleep(Duration::from_millis(15)).await;
                yield Ok(Event::default().event("response.completed").data(
                    r#"{"type":"response.completed","response":{"created_at":1777000001,"output":[]}}"#,
                ));
            };

            Sse::new(stream)
        }

        let app = Router::new().route("/v1/responses", post(responses_handler));
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind listener");
        let addr = listener.local_addr().expect("listener addr");
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve test app");
        });

        let client = Client::builder().no_proxy().build().expect("client");
        let now = Utc::now();
        let provider = ApiKeyProvider {
            id: "custom-openai-images".to_string(),
            name: "Custom".to_string(),
            provider_type: ApiProviderType::Openai,
            api_host: format!("http://{addr}"),
            is_system: false,
            group: ProviderGroup::Custom,
            enabled: true,
            sort_order: 0,
            api_version: None,
            project: None,
            location: None,
            region: None,
            models: vec![ProviderModelConfig::hint("gpt-images-2")],
            prompt_cache_mode: None,
            created_at: now,
            updated_at: now,
        };
        let request = ImageGenerationRequest {
            model: "gpt-images-2".to_string(),
            prompt: "生成两个苹果".to_string(),
            n: 2,
            size: Some("1024x1024".to_string()),
            response_format: "b64_json".to_string(),
            quality: None,
            style: None,
            reference_images: vec![],
            user: None,
        };

        let response = request_openai_responses_images(
            &client,
            &provider,
            &format!("http://{addr}"),
            "test-key",
            &request,
            "gpt-images-2",
            "1024x1024",
        )
        .await
        .expect("responses image request");

        assert_eq!(response.data.len(), 2);
        assert_eq!(response.data[0].b64_json.as_deref(), Some("Zmlyc3Q="));
        assert_eq!(response.data[1].b64_json.as_deref(), Some("c2Vjb25k"));
    }

    #[tokio::test]
    async fn request_openai_responses_images_accepts_event_only_completion_markers() {
        async fn responses_handler() -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>>
        {
            let stream = stream! {
                yield Ok(Event::default().event("response.output_item.done").data(
                    r#"{"output_index":0,"item":{"type":"image_generation_call","result":"Zmlyc3Q=","revised_prompt":"first prompt","output_format":"png"}}"#,
                ));
                tokio::time::sleep(Duration::from_millis(15)).await;
                yield Ok(Event::default().event("response.completed").data(
                    r#"{"response":{"created_at":1777000001,"output":[]}}"#,
                ));
            };

            Sse::new(stream)
        }

        let app = Router::new().route("/v1/responses", post(responses_handler));
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind listener");
        let addr = listener.local_addr().expect("listener addr");
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve test app");
        });

        let client = Client::builder().no_proxy().build().expect("client");
        let now = Utc::now();
        let provider = ApiKeyProvider {
            id: "custom-openai-images".to_string(),
            name: "Custom".to_string(),
            provider_type: ApiProviderType::Openai,
            api_host: format!("http://{addr}"),
            is_system: false,
            group: ProviderGroup::Custom,
            enabled: true,
            sort_order: 0,
            api_version: None,
            project: None,
            location: None,
            region: None,
            models: vec![ProviderModelConfig::hint("gpt-images-2")],
            prompt_cache_mode: None,
            created_at: now,
            updated_at: now,
        };
        let request = ImageGenerationRequest {
            model: "gpt-images-2".to_string(),
            prompt: "生成一个苹果".to_string(),
            n: 1,
            size: Some("1024x1024".to_string()),
            response_format: "b64_json".to_string(),
            quality: None,
            style: None,
            reference_images: vec![],
            user: None,
        };

        let response = request_openai_responses_images(
            &client,
            &provider,
            &format!("http://{addr}"),
            "test-key",
            &request,
            "gpt-images-2",
            "1024x1024",
        )
        .await
        .expect("responses image request");

        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0].b64_json.as_deref(), Some("Zmlyc3Q="));
    }

    #[tokio::test]
    async fn request_openai_responses_images_returns_after_expected_done_events_without_completed()
    {
        async fn responses_handler() -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>>
        {
            let stream = stream! {
                yield Ok(Event::default().event("response.output_item.done").data(
                    r#"{"type":"response.output_item.done","output_index":0,"item":{"type":"image_generation_call","result":"Zmlyc3Q=","revised_prompt":"first prompt","output_format":"png"}}"#,
                ));
                tokio::time::sleep(Duration::from_millis(15)).await;
                yield Ok(Event::default().event("response.output_item.done").data(
                    r#"{"type":"response.output_item.done","output_index":1,"item":{"type":"image_generation_call","result":"c2Vjb25k","revised_prompt":"second prompt","output_format":"png"}}"#,
                ));
                tokio::time::sleep(Duration::from_secs(5)).await;
                yield Ok(Event::default().event("keepalive").data(r#"{"type":"keepalive"}"#));
            };

            Sse::new(stream)
        }

        let app = Router::new().route("/v1/responses", post(responses_handler));
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind listener");
        let addr = listener.local_addr().expect("listener addr");
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve test app");
        });

        let client = Client::builder().no_proxy().build().expect("client");
        let now = Utc::now();
        let provider = ApiKeyProvider {
            id: "custom-openai-images".to_string(),
            name: "Custom".to_string(),
            provider_type: ApiProviderType::Openai,
            api_host: format!("http://{addr}"),
            is_system: false,
            group: ProviderGroup::Custom,
            enabled: true,
            sort_order: 0,
            api_version: None,
            project: None,
            location: None,
            region: None,
            models: vec![ProviderModelConfig::hint("gpt-images-2")],
            prompt_cache_mode: None,
            created_at: now,
            updated_at: now,
        };
        let request = ImageGenerationRequest {
            model: "gpt-images-2".to_string(),
            prompt: "生成两个苹果".to_string(),
            n: 2,
            size: Some("1024x1024".to_string()),
            response_format: "b64_json".to_string(),
            quality: None,
            style: None,
            reference_images: vec![],
            user: None,
        };

        let response = tokio::time::timeout(
            Duration::from_millis(250),
            request_openai_responses_images(
                &client,
                &provider,
                &format!("http://{addr}"),
                "test-key",
                &request,
                "gpt-images-2",
                "1024x1024",
            ),
        )
        .await
        .expect("responses image request should finish before keepalive")
        .expect("responses image request");

        assert_eq!(response.data.len(), 2);
        assert_eq!(response.data[0].b64_json.as_deref(), Some("Zmlyc3Q="));
        assert_eq!(response.data[1].b64_json.as_deref(), Some("c2Vjb25k"));
    }

    #[test]
    fn try_extract_partial_openai_responses_images_returns_done_images_without_completed() {
        let sse = r#"
event: response.output_item.done
data: {"type":"response.output_item.done","output_index":0,"item":{"type":"image_generation_call","result":"Zmlyc3Q=","revised_prompt":"first prompt","output_format":"png"}}
"#;

        let response = try_extract_partial_openai_responses_images(sse, "b64_json")
            .expect("partial extraction should parse")
            .expect("partial extraction should keep at least one image");

        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0].b64_json.as_deref(), Some("Zmlyc3Q="));
    }

    #[test]
    fn should_prefer_openai_responses_image_api_for_generic_codex_base_path() {
        let now = Utc::now();
        let provider = ApiKeyProvider {
            id: "custom-openai-images".to_string(),
            name: "Custom OpenAI Images".to_string(),
            provider_type: ApiProviderType::Openai,
            api_host: "https://gateway.example.com/codex".to_string(),
            is_system: false,
            group: ProviderGroup::Custom,
            enabled: true,
            sort_order: 0,
            api_version: None,
            project: None,
            location: None,
            region: None,
            models: vec![ProviderModelConfig::hint("gpt-images-2")],
            prompt_cache_mode: None,
            created_at: now,
            updated_at: now,
        };

        assert!(should_prefer_openai_responses_image_api(&provider));

        let standard_provider = ApiKeyProvider {
            api_host: "https://api.openai.com/v1".to_string(),
            ..provider
        };
        assert!(!should_prefer_openai_responses_image_api(
            &standard_provider
        ));
    }

    #[test]
    fn image_generation_matcher_matches_supported_families() {
        assert!(is_likely_image_generation_model_id("gpt-images-2"));
        assert!(is_likely_image_generation_model_id("gpt-image-1"));
        assert!(is_likely_image_generation_model_id("dall-e-3"));
        assert!(is_likely_image_generation_model_id("cogview-3-flash"));
        assert!(is_likely_image_generation_model_id(
            "black-forest-labs/FLUX.1-schnell"
        ));
        assert!(is_likely_image_generation_model_id("nano-banana-pro"));
        assert!(!is_likely_image_generation_model_id("gpt-5.2"));
    }

    #[test]
    fn image_api_provider_should_not_restore_local_image_keyword_matchers() {
        let source = include_str!("image_api_provider.rs");
        let looks_like_marker = ["fn looks_like_", "image_generation_model("].concat();
        let fal_marker = ["fn looks_like_", "fal_model("].concat();
        let keywords_marker = ["IMAGE_MODEL", "_KEYWORDS"].concat();

        assert!(!source.contains(&looks_like_marker));
        assert!(!source.contains(&keywords_marker));
        assert!(!source.contains(&fal_marker));
    }
}
