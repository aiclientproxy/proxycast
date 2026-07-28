use super::common::{
    canonical_generation_prompt, non_empty, unsupported_option, ProtocolMappingError,
};
use app_server_protocol::ProtocolKind;
use runtime_core::CanonicalRequest;
use serde_json::{json, Map, Value};

const UNSUPPORTED_OPTIONS: &[&str] = &["end_image_url", "seed", "generate_audio", "camera_fixed"];

pub(crate) fn body_for_model(
    model_id: &str,
    request: &CanonicalRequest,
) -> Result<Value, ProtocolMappingError> {
    for option in UNSUPPORTED_OPTIONS {
        if request.provider_options.contains_key(*option) {
            return Err(unsupported_option(ProtocolKind::XaiVideo, option));
        }
    }

    let prompt = canonical_generation_prompt(request, ProtocolKind::XaiVideo, false)?;
    let mut body = Map::from_iter([
        ("model".to_string(), json!(model_id.trim())),
        ("prompt".to_string(), json!(prompt)),
    ]);
    insert_string_option(&mut body, request, "aspect_ratio");
    insert_string_option(&mut body, request, "resolution");
    insert_u64_option(&mut body, request, "duration");
    if let Some(image_url) = request
        .provider_options
        .get("image_url")
        .and_then(Value::as_str)
        .and_then(|value| non_empty(Some(value)))
    {
        body.insert("image".to_string(), json!({ "url": image_url }));
    }
    Ok(Value::Object(body))
}

fn insert_string_option(body: &mut Map<String, Value>, request: &CanonicalRequest, key: &str) {
    if let Some(value) = request
        .provider_options
        .get(key)
        .and_then(Value::as_str)
        .and_then(|value| non_empty(Some(value)))
    {
        body.insert(key.to_string(), json!(value));
    }
}

fn insert_u64_option(body: &mut Map<String, Value>, request: &CanonicalRequest, key: &str) {
    let value = request.provider_options.get(key).and_then(|value| {
        value
            .as_u64()
            .or_else(|| value.as_str().and_then(|raw| raw.trim().parse().ok()))
    });
    if let Some(value) = value {
        body.insert(key.to_string(), json!(value));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowers_xai_image_to_video_body() {
        let mut request = CanonicalRequest::text("grok-imagine-video", "让画面缓慢向前推进");
        request.provider_options.insert(
            "image_url".to_string(),
            json!("https://example.test/start.png"),
        );
        request
            .provider_options
            .insert("duration".to_string(), json!(6));
        request
            .provider_options
            .insert("aspect_ratio".to_string(), json!("16:9"));
        request
            .provider_options
            .insert("resolution".to_string(), json!("720p"));

        let body = body_for_model("grok-imagine-video", &request).expect("xAI video body");

        assert_eq!(body["model"], "grok-imagine-video");
        assert_eq!(body["prompt"], "让画面缓慢向前推进");
        assert_eq!(body["image"]["url"], "https://example.test/start.png");
        assert_eq!(body["duration"], 6);
        assert_eq!(body["aspect_ratio"], "16:9");
        assert_eq!(body["resolution"], "720p");
    }

    #[test]
    fn rejects_options_not_supported_by_xai_video_wire() {
        let mut request = CanonicalRequest::text("grok-imagine-video", "生成视频");
        request.provider_options.insert(
            "end_image_url".to_string(),
            json!("https://example.test/end.png"),
        );

        assert_eq!(
            body_for_model("grok-imagine-video", &request),
            Err(ProtocolMappingError::UnsupportedOption {
                protocol: ProtocolKind::XaiVideo,
                option: "end_image_url",
            })
        );
    }
}
