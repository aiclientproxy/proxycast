use axum::http::header::CONTENT_TYPE;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use lime_core::models::model_registry::ProviderModelConfig;
use lime_core::models::openai::{ImageData, ImageGenerationRequest, ImageGenerationResponse};
use reqwest::Client;
use serde_json::{json, Value};

const GEMINI_DEFAULT_HOST: &str = "https://generativelanguage.googleapis.com";

#[derive(Debug, Clone, PartialEq, Eq)]
struct GeminiImageReference {
    data: String,
    mime_type: String,
}

fn looks_like_gemini_image_model(model: &str) -> bool {
    let normalized = model.trim().to_ascii_lowercase();
    !normalized.is_empty()
        && normalized.starts_with("gemini-")
        && (normalized.contains("-image")
            || normalized.contains("image-")
            || normalized.contains("nano-banana"))
}

pub(super) fn resolve_gemini_image_model(
    request_model: &str,
    preferred_model_id: Option<&str>,
    models: &[ProviderModelConfig],
) -> Option<String> {
    let request_model = request_model.trim();
    if looks_like_gemini_image_model(request_model) {
        return Some(request_model.to_string());
    }

    if let Some(preferred_model) = preferred_model_id
        .map(str::trim)
        .filter(|value| looks_like_gemini_image_model(value))
    {
        return Some(preferred_model.to_string());
    }

    models
        .iter()
        .map(|model| model.id.trim())
        .find(|model| looks_like_gemini_image_model(model))
        .map(ToString::to_string)
}

fn normalize_gemini_api_host(api_host: &str) -> String {
    let trimmed = api_host.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return GEMINI_DEFAULT_HOST.to_string();
    }

    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_string()
    } else {
        format!("https://{trimmed}")
    }
}

fn build_gemini_interactions_url(api_host: &str) -> String {
    let normalized = normalize_gemini_api_host(api_host);
    if normalized.ends_with("/v1beta") {
        return format!("{normalized}/interactions");
    }

    format!("{normalized}/v1beta/interactions")
}

fn parse_data_url_image(value: &str) -> Option<GeminiImageReference> {
    let trimmed = value.trim();
    let rest = trimmed.strip_prefix("data:")?;
    let (mime_type, data) = rest.split_once(";base64,")?;
    let mime_type = mime_type.trim();
    let data = data.trim().replace(char::is_whitespace, "");

    if !mime_type.starts_with("image/") || data.is_empty() {
        return None;
    }

    Some(GeminiImageReference {
        data,
        mime_type: mime_type.to_string(),
    })
}

fn looks_like_base64_image_data(value: &str) -> bool {
    let normalized = value.trim().replace(char::is_whitespace, "");
    normalized.len() >= 64
        && normalized
            .chars()
            .all(|char| char.is_ascii_alphanumeric() || matches!(char, '+' | '/' | '='))
}

fn parse_inline_reference_image(value: &str) -> Option<GeminiImageReference> {
    parse_data_url_image(value).or_else(|| {
        let normalized = value.trim().replace(char::is_whitespace, "");
        if looks_like_base64_image_data(&normalized) {
            Some(GeminiImageReference {
                data: normalized,
                mime_type: "image/png".to_string(),
            })
        } else {
            None
        }
    })
}

async fn download_reference_image(
    client: &Client,
    image_url: &str,
) -> Result<GeminiImageReference, String> {
    let response = client
        .get(image_url)
        .send()
        .await
        .map_err(|error| format!("Gemini 参考图下载失败: {error}"))?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("Gemini 参考图下载失败: HTTP {}", status.as_u16()));
    }

    let mime_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| value.starts_with("image/"))
        .unwrap_or("image/png")
        .to_string();
    let bytes = response
        .bytes()
        .await
        .map_err(|error| format!("Gemini 参考图读取失败: {error}"))?;

    Ok(GeminiImageReference {
        data: BASE64.encode(bytes),
        mime_type,
    })
}

async fn resolve_gemini_references(
    client: &Client,
    reference_images: &[String],
) -> Result<Vec<GeminiImageReference>, String> {
    let mut references = Vec::new();

    for image in reference_images {
        let trimmed = image.trim();
        if trimmed.is_empty() {
            continue;
        }

        if let Some(reference) = parse_inline_reference_image(trimmed) {
            references.push(reference);
            continue;
        }

        if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
            references.push(download_reference_image(client, trimmed).await?);
        }
    }

    Ok(references)
}

fn size_to_aspect_ratio(size: &str) -> Option<String> {
    let (width, height) = size.split_once('x')?;
    let width = width.parse::<u32>().ok()?;
    let height = height.parse::<u32>().ok()?;

    match (width, height) {
        (0, _) | (_, 0) => None,
        (width, height) if width == height => Some("1:1".to_string()),
        (width, height) if width > height => Some("16:9".to_string()),
        _ => Some("9:16".to_string()),
    }
}

fn resolve_gemini_image_size(size: &str) -> Option<&'static str> {
    let (width, height) = size.split_once('x')?;
    let width = width.parse::<u32>().ok()?;
    let height = height.parse::<u32>().ok()?;
    let longest_edge = width.max(height);

    if longest_edge >= 3072 {
        Some("4K")
    } else if longest_edge >= 1536 {
        Some("2K")
    } else if longest_edge > 0 {
        Some("1K")
    } else {
        None
    }
}

fn resolve_gemini_thinking_level(quality: Option<&str>) -> Option<&'static str> {
    let normalized = quality?.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return None;
    }

    if matches!(normalized.as_str(), "high" | "hd") {
        Some("high")
    } else {
        None
    }
}

fn build_gemini_interactions_payload(
    request: &ImageGenerationRequest,
    model: &str,
    request_size: &str,
    reference_images: &[GeminiImageReference],
) -> Value {
    let mut input = vec![json!({
        "type": "text",
        "text": request.prompt.trim(),
    })];

    input.extend(reference_images.iter().map(|image| {
        json!({
            "type": "image",
            "mime_type": image.mime_type,
            "data": image.data,
        })
    }));

    let mut response_format = json!({
        "type": "image",
        "mime_type": "image/png",
    });

    if let Some(aspect_ratio) = size_to_aspect_ratio(request_size) {
        response_format["aspect_ratio"] = Value::String(aspect_ratio);
    }

    if let Some(image_size) = resolve_gemini_image_size(request_size) {
        response_format["image_size"] = Value::String(image_size.to_string());
    }

    let mut payload = json!({
        "model": model,
        "input": input,
        "response_format": response_format,
    });

    if let Some(thinking_level) = resolve_gemini_thinking_level(request.quality.as_deref()) {
        payload["generation_config"] = json!({
            "thinking_level": thinking_level,
        });
    }

    payload
}

fn read_image_block(value: &Value) -> Option<(String, String)> {
    let object = value.as_object()?;
    let data = object
        .get("data")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())?;
    let mime_type = object
        .get("mime_type")
        .or_else(|| object.get("mimeType"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| value.starts_with("image/"))
        .unwrap_or("image/png");

    Some((data.replace(char::is_whitespace, ""), mime_type.to_string()))
}

fn push_unique_gemini_image(images: &mut Vec<(String, String)>, image: (String, String)) {
    if images.iter().any(|existing| existing == &image) {
        return;
    }

    images.push(image);
}

fn collect_gemini_output_images(value: &Value, images: &mut Vec<(String, String)>) {
    match value {
        Value::Object(map) => {
            if let Some(image) = read_image_block(value) {
                push_unique_gemini_image(images, image);
            }

            for key in [
                "interaction",
                "response",
                "output",
                "model_output",
                "steps",
                "candidates",
                "content",
                "parts",
                "images",
                "generated_images",
                "output_image",
                "outputImage",
                "inline_data",
                "inlineData",
            ] {
                if let Some(child) = map.get(key) {
                    collect_gemini_output_images(child, images);
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_gemini_output_images(item, images);
            }
        }
        _ => {}
    }
}

fn normalize_gemini_interactions_image_response(
    payload: &Value,
    response_format: &str,
) -> Result<ImageGenerationResponse, String> {
    let mut images = Vec::new();
    collect_gemini_output_images(payload, &mut images);

    if images.is_empty() {
        return Err("Gemini Interactions 图片接口未返回可解析图片字段".to_string());
    }

    let data = images
        .into_iter()
        .map(|(image_data, mime_type)| {
            if response_format == "b64_json" {
                ImageData {
                    b64_json: Some(image_data),
                    url: None,
                    revised_prompt: None,
                }
            } else {
                ImageData {
                    b64_json: None,
                    url: Some(format!("data:{mime_type};base64,{image_data}")),
                    revised_prompt: None,
                }
            }
        })
        .collect();

    Ok(ImageGenerationResponse {
        created: chrono::Utc::now().timestamp(),
        data,
    })
}

fn summarize_gemini_error_body(body: &str) -> String {
    let normalized = body.trim();
    if normalized.is_empty() {
        return "Gemini Interactions 图片接口返回了空响应。".to_string();
    }

    if let Ok(parsed) = serde_json::from_str::<Value>(normalized) {
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
            .get("message")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return message.to_string();
        }
    }

    let mut current_event: Option<String> = None;
    let mut data_lines: Vec<String> = Vec::new();

    let flush_event = |current_event: &mut Option<String>, data_lines: &mut Vec<String>| {
        let event = current_event.take();
        let data = data_lines.join("\n");
        data_lines.clear();
        event.map(|event_name| (event_name, data))
    };

    for raw_line in normalized.lines() {
        let line = raw_line.trim_end();
        if line.trim().is_empty() {
            if let Some((event_name, data)) = flush_event(&mut current_event, &mut data_lines) {
                if event_name == "error" || event_name == "event:error" {
                    if let Ok(parsed) = serde_json::from_str::<Value>(data.trim()) {
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
                    }
                }
            }
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

    if let Some((event_name, data)) = flush_event(&mut current_event, &mut data_lines) {
        if event_name == "error" || event_name == "event:error" {
            if let Ok(parsed) = serde_json::from_str::<Value>(data.trim()) {
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
            }
        }
    }

    preview_text(normalized, 240)
}

pub(super) async fn request_gemini_interactions_image(
    client: &Client,
    api_host: &str,
    api_key: &str,
    request: &ImageGenerationRequest,
    model: &str,
    request_size: &str,
) -> Result<ImageGenerationResponse, String> {
    let endpoint = build_gemini_interactions_url(api_host);
    let references = resolve_gemini_references(client, &request.reference_images).await?;
    let payload = build_gemini_interactions_payload(request, model, request_size, &references);
    let response = client
        .post(&endpoint)
        .header("x-goog-api-key", api_key)
        .header(CONTENT_TYPE, "application/json")
        .json(&payload)
        .send()
        .await
        .map_err(|error| format!("Gemini Interactions 图片接口请求失败: {error}"))?;

    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| format!("Gemini Interactions 图片接口响应读取失败: {error}"))?;

    if !status.is_success() {
        return Err(format!(
            "Gemini Interactions 图片接口 HTTP {}: {}",
            status.as_u16(),
            summarize_gemini_error_body(&body)
        ));
    }

    let payload = serde_json::from_str::<Value>(&body).map_err(|error| {
        format!(
            "Gemini Interactions 图片接口 JSON 解析失败: {error}; body={}",
            preview_text(&body, 240)
        )
    })?;

    normalize_gemini_interactions_image_response(&payload, &request.response_format)
}

fn preview_text(text: &str, max_len: usize) -> String {
    let normalized = text.trim();
    if normalized.chars().count() <= max_len {
        return normalized.to_string();
    }

    let preview: String = normalized.chars().take(max_len).collect();
    format!("{preview}...")
}

#[cfg(test)]
mod tests {
    use super::{
        build_gemini_interactions_payload, normalize_gemini_interactions_image_response,
        parse_inline_reference_image, request_gemini_interactions_image,
        resolve_gemini_image_model, summarize_gemini_error_body, GeminiImageReference,
    };
    use axum::{
        http::header::CONTENT_TYPE,
        routing::{get, post},
        Router,
    };
    use lime_core::models::model_registry::ProviderModelConfig;
    use lime_core::models::openai::ImageGenerationRequest;
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    use tokio::net::TcpListener;

    fn request(reference_images: Vec<String>) -> ImageGenerationRequest {
        request_with_quality(reference_images, None)
    }

    fn request_with_quality(
        reference_images: Vec<String>,
        quality: Option<&str>,
    ) -> ImageGenerationRequest {
        ImageGenerationRequest {
            model: "gemini-3.1-flash-image".to_string(),
            prompt: "生成一张青柠图片".to_string(),
            n: 1,
            size: Some("1024x1024".to_string()),
            response_format: "url".to_string(),
            quality: quality.map(ToString::to_string),
            style: None,
            reference_images,
            user: None,
        }
    }

    #[test]
    fn resolve_gemini_image_model_prefers_request_then_preference_then_models() {
        assert_eq!(
            resolve_gemini_image_model("gemini-3.1-flash-image", None, &[]),
            Some("gemini-3.1-flash-image".to_string())
        );
        assert_eq!(
            resolve_gemini_image_model(
                "gpt-5.2",
                Some("gemini-3-pro-image"),
                &[ProviderModelConfig::hint("gemini-2.5-flash-image")],
            ),
            Some("gemini-3-pro-image".to_string())
        );
        assert_eq!(
            resolve_gemini_image_model(
                "gpt-5.2",
                None,
                &[ProviderModelConfig::hint("gemini-2.5-flash-image")],
            ),
            Some("gemini-2.5-flash-image".to_string())
        );
        assert_eq!(
            resolve_gemini_image_model(
                "gpt-5.2",
                None,
                &[
                    ProviderModelConfig::hint("gemini-3.1-flash-image"),
                    ProviderModelConfig::hint("gemini-3-pro-image"),
                ],
            ),
            Some("gemini-3.1-flash-image".to_string())
        );
        assert_eq!(resolve_gemini_image_model("gpt-5.2", None, &[]), None);
    }

    #[test]
    fn build_gemini_interactions_payload_uses_current_image_request_shape() {
        let reference = GeminiImageReference {
            data: "cmVm".to_string(),
            mime_type: "image/png".to_string(),
        };
        let payload = build_gemini_interactions_payload(
            &request(vec![]),
            "gemini-3.1-flash-image",
            "1792x1024",
            &[reference],
        );

        assert_eq!(payload["model"].as_str(), Some("gemini-3.1-flash-image"));
        assert_eq!(payload["input"][0]["type"].as_str(), Some("text"));
        assert_eq!(payload["input"][1]["type"].as_str(), Some("image"));
        assert_eq!(payload["input"][1]["mime_type"].as_str(), Some("image/png"));
        assert_eq!(payload["input"][1]["data"].as_str(), Some("cmVm"));
        assert_eq!(payload["response_format"]["type"].as_str(), Some("image"));
        assert_eq!(
            payload["response_format"]["aspect_ratio"].as_str(),
            Some("16:9")
        );
        assert_eq!(
            payload["response_format"]["image_size"].as_str(),
            Some("2K")
        );
        assert!(payload.get("generation_config").is_none());
    }

    #[test]
    fn build_gemini_interactions_payload_maps_high_quality_to_thinking_level_high() {
        let payload = build_gemini_interactions_payload(
            &request_with_quality(vec![], Some("hd")),
            "gemini-3.1-flash-image",
            "1024x1024",
            &[],
        );

        assert_eq!(
            payload["generation_config"]["thinking_level"].as_str(),
            Some("high")
        );
    }

    #[test]
    fn parse_inline_reference_image_accepts_data_url_and_raw_base64() {
        assert_eq!(
            parse_inline_reference_image("data:image/jpeg;base64, YWJjZA== "),
            Some(GeminiImageReference {
                data: "YWJjZA==".to_string(),
                mime_type: "image/jpeg".to_string(),
            })
        );

        let raw = "a".repeat(80);
        assert_eq!(
            parse_inline_reference_image(&raw),
            Some(GeminiImageReference {
                data: raw,
                mime_type: "image/png".to_string(),
            })
        );
    }

    #[test]
    fn normalize_gemini_interactions_image_response_supports_output_image() {
        let response = normalize_gemini_interactions_image_response(
            &json!({
                "interaction": {
                    "output_image": {
                        "data": "aW1hZ2U=",
                        "mime_type": "image/png"
                    }
                }
            }),
            "url",
        )
        .expect("gemini output image");

        assert_eq!(response.data.len(), 1);
        assert_eq!(
            response.data[0].url.as_deref(),
            Some("data:image/png;base64,aW1hZ2U=")
        );
    }

    #[test]
    fn normalize_gemini_interactions_image_response_supports_nested_inline_data() {
        let response = normalize_gemini_interactions_image_response(
            &json!({
                "steps": [
                    {
                        "model_output": {
                            "parts": [
                                {
                                    "inline_data": {
                                        "data": "bmVzdGVk",
                                        "mime_type": "image/webp"
                                    }
                                }
                            ]
                        }
                    }
                ]
            }),
            "b64_json",
        )
        .expect("gemini nested inline image");

        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0].b64_json.as_deref(), Some("bmVzdGVk"));
        assert_eq!(response.data[0].url, None);
    }

    #[test]
    fn normalize_gemini_interactions_image_response_supports_direct_step_image_blocks() {
        let response = normalize_gemini_interactions_image_response(
            &json!({
                "steps": [
                    {
                        "type": "model_output",
                        "content": [
                            {
                                "type": "image",
                                "data": "Z2VtaW5pLXN0ZXA=",
                                "mime_type": "image/webp"
                            }
                        ]
                    }
                ]
            }),
            "url",
        )
        .expect("gemini direct step image");

        assert_eq!(response.data.len(), 1);
        assert_eq!(
            response.data[0].url.as_deref(),
            Some("data:image/webp;base64,Z2VtaW5pLXN0ZXA=")
        );
    }

    #[test]
    fn summarize_gemini_error_body_prefers_structured_json_messages() {
        assert_eq!(
            summarize_gemini_error_body(
                r#"{"error":{"message":"  quota exhausted  ","code":"RESOURCE_EXHAUSTED"}}"#
            ),
            "quota exhausted"
        );
        assert_eq!(
            summarize_gemini_error_body(r#"{"message":"  missing api key  "}"#),
            "missing api key"
        );
    }

    #[test]
    fn summarize_gemini_error_body_handles_sse_error_event_and_fallbacks() {
        assert_eq!(
            summarize_gemini_error_body(
                "event: error\n\
                 data: {\"error\":{\"message\":\"  invalid prompt  \"}}\n\n"
            ),
            "invalid prompt"
        );
        assert_eq!(
            summarize_gemini_error_body("   "),
            "Gemini Interactions 图片接口返回了空响应。"
        );
        assert_eq!(
            summarize_gemini_error_body("  plain upstream error body  "),
            "plain upstream error body"
        );
    }

    #[tokio::test]
    async fn request_gemini_interactions_image_includes_reference_images_from_data_url_and_http() {
        let captured_body: Arc<Mutex<Option<serde_json::Value>>> = Arc::new(Mutex::new(None));
        let captured_body_for_handler = Arc::clone(&captured_body);

        let app = Router::new()
            .route(
                "/ref.png",
                get(|| async { ([(CONTENT_TYPE, "image/webp")], "downloaded-ref") }),
            )
            .route(
                "/v1beta/interactions",
                post(
                    move |headers: axum::http::HeaderMap,
                          axum::Json(body): axum::Json<serde_json::Value>| {
                        let captured_body = Arc::clone(&captured_body_for_handler);
                        async move {
                            assert_eq!(
                                headers
                                    .get("x-goog-api-key")
                                    .and_then(|value| value.to_str().ok()),
                                Some("test-key")
                            );
                            *captured_body.lock().expect("capture body mutex") = Some(body);
                            axum::Json(json!({
                                "interaction": {
                                    "output_image": {
                                        "data": "Z2VtaW5pLXRlc3Q=",
                                        "mime_type": "image/png"
                                    }
                                }
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

        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client");
        let request = ImageGenerationRequest {
            model: "gemini-3.1-flash-image".to_string(),
            prompt: "生成一张青柠".to_string(),
            n: 1,
            size: Some("1536x1024".to_string()),
            response_format: "url".to_string(),
            quality: Some("hd".to_string()),
            style: None,
            reference_images: vec![
                " data:image/jpeg;base64, YWJjZA== ".to_string(),
                format!(" http://{addr}/ref.png "),
            ],
            user: None,
        };

        let response = request_gemini_interactions_image(
            &client,
            &format!("http://{addr}"),
            "test-key",
            &request,
            "gemini-3.1-flash-image",
            "1536x1024",
        )
        .await
        .expect("gemini image request");

        assert_eq!(response.data.len(), 1);
        assert_eq!(
            response.data[0].url.as_deref(),
            Some("data:image/png;base64,Z2VtaW5pLXRlc3Q=")
        );

        let body = captured_body
            .lock()
            .expect("capture body mutex")
            .clone()
            .expect("captured body");
        assert_eq!(body["input"].as_array().map(|input| input.len()), Some(3));
        assert_eq!(body["input"][0]["type"].as_str(), Some("text"));
        assert_eq!(body["input"][1]["type"].as_str(), Some("image"));
        assert_eq!(body["input"][1]["mime_type"].as_str(), Some("image/jpeg"));
        assert_eq!(body["input"][1]["data"].as_str(), Some("YWJjZA=="));
        assert_eq!(body["input"][2]["type"].as_str(), Some("image"));
        assert_eq!(body["input"][2]["mime_type"].as_str(), Some("image/webp"));
        assert_eq!(
            body["input"][2]["data"].as_str(),
            Some("ZG93bmxvYWRlZC1yZWY=")
        );
        assert_eq!(
            body["response_format"]["aspect_ratio"].as_str(),
            Some("16:9")
        );
        assert_eq!(body["response_format"]["image_size"].as_str(), Some("2K"));
        assert_eq!(
            body["generation_config"]["thinking_level"].as_str(),
            Some("high")
        );
    }
}
