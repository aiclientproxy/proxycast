use super::common::{canonical_generation_prompt, non_empty, ProtocolMappingError};
use app_server_protocol::ProtocolKind;
use runtime_core::CanonicalRequest;
#[cfg(test)]
use runtime_core::ContentPart;
use serde_json::{json, Map, Value};

pub(crate) fn body_for_model(
    model_id: &str,
    request: &CanonicalRequest,
) -> Result<Value, ProtocolMappingError> {
    let input = canonical_generation_prompt(request, ProtocolKind::OpenaiAudioSpeech, false)?;
    let mut body = Map::new();
    body.insert("model".to_string(), json!(model_id.trim()));
    body.insert("input".to_string(), json!(input));
    insert_string_option(&mut body, request, "voice");
    insert_string_option(&mut body, request, "response_format");
    insert_string_option(&mut body, request, "instructions");
    insert_number_option(&mut body, request, "speed");
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

fn insert_number_option(body: &mut Map<String, Value>, request: &CanonicalRequest, key: &str) {
    let Some(value) = request.provider_options.get(key) else {
        return;
    };
    if value.is_number() {
        body.insert(key.to_string(), value.clone());
    } else if let Some(raw) = value
        .as_str()
        .and_then(|raw| raw.trim().parse::<f64>().ok())
    {
        body.insert(key.to_string(), json!(raw));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowers_tts_options() {
        let mut request = CanonicalRequest::text("gpt-4o-mini-tts", "你好，Lime");
        request
            .provider_options
            .insert("voice".to_string(), json!("alloy"));
        request
            .provider_options
            .insert("response_format".to_string(), json!("wav"));
        request
            .provider_options
            .insert("speed".to_string(), json!(1.1));

        let body = body_for_model("gpt-4o-mini-tts", &request).expect("tts body");
        assert_eq!(body["model"], "gpt-4o-mini-tts");
        assert_eq!(body["input"], "你好，Lime");
        assert_eq!(body["voice"], "alloy");
        assert_eq!(body["response_format"], "wav");
        assert_eq!(body["speed"], 1.1);
    }

    #[test]
    fn rejects_non_text_tts_input() {
        let mut request = CanonicalRequest::text("tts", "hello");
        request.messages[0]
            .content
            .push(ContentPart::media("asset://audio", "audio/wav").expect("media"));
        assert!(matches!(
            body_for_model("tts", &request),
            Err(ProtocolMappingError::UnsupportedInputPart {
                protocol: ProtocolKind::OpenaiAudioSpeech,
                part_type: "media"
            })
        ));
    }
}
