use super::stream::{sse_frames_with_idle_timeout, DEFAULT_STREAM_IDLE_TIMEOUT};
use super::CurrentProviderError;
use async_stream::try_stream;
use futures::{Stream, StreamExt};
use reqwest::Response;
use runtime_core::{
    CanonicalLlmEvent as LlmEvent, CanonicalRequest, CanonicalRole, CanonicalToolDefinition,
    ContentPart, FailureClassification, FinishReason, ToolResultValue, Usage,
};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

pub(super) fn endpoint(base_url: Option<&str>, model: &str) -> String {
    let base = base_url
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_BASE_URL)
        .trim_end_matches('/');
    if base.contains(":streamGenerateContent") {
        return with_sse_query(base);
    }

    let versioned_base = if url::Url::parse(base)
        .ok()
        .is_some_and(|url| url.path().trim_matches('/').is_empty())
    {
        format!("{base}/v1beta")
    } else {
        base.to_string()
    };
    let encoded_model =
        url::form_urlencoded::byte_serialize(model.trim().as_bytes()).collect::<String>();
    with_sse_query(&format!(
        "{versioned_base}/models/{encoded_model}:streamGenerateContent"
    ))
}

fn with_sse_query(endpoint: &str) -> String {
    if endpoint
        .split_once('?')
        .is_some_and(|(_, query)| query.split('&').any(|pair| pair == "alt=sse"))
    {
        endpoint.to_string()
    } else if endpoint.contains('?') {
        format!("{endpoint}&alt=sse")
    } else {
        format!("{endpoint}?alt=sse")
    }
}

pub(super) fn request(
    request: &CanonicalRequest,
    media_payloads: &BTreeMap<String, String>,
) -> Result<Value, CurrentProviderError> {
    let mut contents = Vec::new();
    for message in &request.messages {
        let (role, parts) = match message.role {
            CanonicalRole::Assistant => {
                ("model", assistant_parts(&message.content, media_payloads)?)
            }
            CanonicalRole::Tool => ("user", tool_result_parts(&message.content)?),
            CanonicalRole::User | CanonicalRole::System | CanonicalRole::Developer => {
                ("user", user_parts(&message.content, media_payloads)?)
            }
        };
        push_content(&mut contents, role, parts);
    }

    let mut object = Map::from_iter([("contents".to_string(), Value::Array(contents))]);
    let system = text_from_parts(&request.system);
    if !system.is_empty() {
        object.insert(
            "systemInstruction".to_string(),
            json!({ "parts": [{ "text": system }] }),
        );
    }
    if !request.tools.is_empty() {
        let declarations = request
            .tools
            .iter()
            .map(|tool| {
                let CanonicalToolDefinition::Function {
                    name,
                    description,
                    input_schema,
                    ..
                } = tool
                else {
                    return Err(CurrentProviderError::invalid_request(
                        "custom tools require a Responses provider route",
                    ));
                };
                let mut declaration = Map::from_iter([
                    ("name".to_string(), json!(name)),
                    ("description".to_string(), json!(description)),
                ]);
                if let Some(parameters) = gemini_tool_schema(input_schema) {
                    declaration.insert("parameters".to_string(), parameters);
                }
                Ok(Value::Object(declaration))
            })
            .collect::<Result<Vec<_>, _>>()?;
        object.insert(
            "tools".to_string(),
            json!([{ "functionDeclarations": declarations }]),
        );
    }

    let mut generation = Map::new();
    if let Some(max_tokens) = request.generation.max_tokens {
        generation.insert("maxOutputTokens".to_string(), json!(max_tokens));
    }
    if let Some(temperature) = request.generation.temperature {
        generation.insert("temperature".to_string(), json!(temperature));
    }
    if let Some(top_p) = request.generation.top_p {
        generation.insert("topP".to_string(), json!(top_p));
    }
    if let Some(top_k) = request.generation.top_k {
        generation.insert("topK".to_string(), json!(top_k));
    }
    if !generation.is_empty() {
        object.insert("generationConfig".to_string(), Value::Object(generation));
    }
    Ok(Value::Object(object))
}

fn push_content(contents: &mut Vec<Value>, role: &str, parts: Vec<Value>) {
    if parts.is_empty() {
        return;
    }
    if let Some(previous) = contents.last_mut().and_then(Value::as_object_mut) {
        if previous.get("role").and_then(Value::as_str) == Some(role) {
            if let Some(previous_parts) = previous.get_mut("parts").and_then(Value::as_array_mut) {
                previous_parts.extend(parts);
                return;
            }
        }
    }
    contents.push(json!({ "role": role, "parts": parts }));
}

fn user_parts(
    parts: &[ContentPart],
    media_payloads: &BTreeMap<String, String>,
) -> Result<Vec<Value>, CurrentProviderError> {
    parts
        .iter()
        .map(|part| match part {
            ContentPart::Text { text, .. } => Ok(json!({ "text": text })),
            ContentPart::Media {
                uri, media_type, ..
            } => media_part(provider_media_uri(uri, media_payloads), media_type),
            other => Err(unsupported_content("user", other)),
        })
        .collect()
}

fn assistant_parts(
    parts: &[ContentPart],
    media_payloads: &BTreeMap<String, String>,
) -> Result<Vec<Value>, CurrentProviderError> {
    parts
        .iter()
        .map(|part| match part {
            ContentPart::Text { text, .. } => Ok(json!({ "text": text })),
            ContentPart::Reasoning { text, metadata, .. } => {
                let mut value = json!({ "text": text, "thought": true });
                if let Some(signature) = thought_signature(metadata) {
                    value["thoughtSignature"] = json!(signature);
                }
                Ok(value)
            }
            ContentPart::ToolCall {
                name,
                input,
                metadata,
                ..
            } => {
                let mut value = json!({ "functionCall": { "name": name, "args": input } });
                if let Some(signature) = thought_signature(metadata) {
                    value["thoughtSignature"] = json!(signature);
                }
                Ok(value)
            }
            ContentPart::Media {
                uri, media_type, ..
            } => media_part(provider_media_uri(uri, media_payloads), media_type),
            other => Err(unsupported_content("assistant", other)),
        })
        .collect()
}

fn tool_result_parts(parts: &[ContentPart]) -> Result<Vec<Value>, CurrentProviderError> {
    parts
        .iter()
        .map(|part| match part {
            ContentPart::ToolResult {
                name,
                result,
                error,
                ..
            } => Ok(json!({
                "functionResponse": {
                    "name": name,
                    "response": {
                        "name": name,
                        "content": tool_result_text(result, error.as_deref()),
                    }
                }
            })),
            other => Err(unsupported_content("tool", other)),
        })
        .collect()
}

fn unsupported_content(role: &str, part: &ContentPart) -> CurrentProviderError {
    CurrentProviderError::invalid_request(format!(
        "Gemini GenerateContent does not support canonical {role} content {part:?}"
    ))
}

fn media_part(uri: &str, media_type: &str) -> Result<Value, CurrentProviderError> {
    if let Some((metadata, data)) = uri
        .strip_prefix("data:")
        .and_then(|value| value.split_once(','))
    {
        if !metadata
            .split(';')
            .any(|part| part.eq_ignore_ascii_case("base64"))
        {
            return Err(CurrentProviderError::invalid_request(
                "Gemini inline media must use base64 encoding",
            ));
        }
        let declared_type = metadata.split(';').next().unwrap_or_default().trim();
        return Ok(json!({
            "inlineData": {
                "mimeType": if declared_type.is_empty() { media_type } else { declared_type },
                "data": data,
            }
        }));
    }
    if uri.starts_with("gs://") || is_google_file_uri(uri) {
        return Ok(json!({
            "fileData": { "mimeType": media_type, "fileUri": uri }
        }));
    }
    Err(CurrentProviderError::invalid_request(format!(
        "Gemini media requires base64 inlineData or a Google file URI: {uri}"
    )))
}

fn is_google_file_uri(uri: &str) -> bool {
    url::Url::parse(uri).ok().is_some_and(|url| {
        url.scheme() == "https" && url.host_str() == Some("generativelanguage.googleapis.com")
    })
}

fn thought_signature(metadata: &BTreeMap<String, Value>) -> Option<&str> {
    metadata
        .get("google")
        .and_then(|value| value.get("thoughtSignature"))
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
}

fn provider_media_uri<'a>(uri: &'a str, payloads: &'a BTreeMap<String, String>) -> &'a str {
    payloads.get(uri).map(String::as_str).unwrap_or(uri)
}

fn text_from_parts(parts: &[ContentPart]) -> String {
    parts
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

fn tool_result_text(result: &ToolResultValue, error: Option<&str>) -> String {
    if let Some(error) = error.filter(|value| !value.trim().is_empty()) {
        return error.to_string();
    }
    match result {
        ToolResultValue::Text { value } => value.clone(),
        ToolResultValue::Json { value } | ToolResultValue::Error { value } => value.to_string(),
        ToolResultValue::Content { value } => text_from_parts(value),
    }
}

fn gemini_tool_schema(schema: &Value) -> Option<Value> {
    let object = schema.as_object()?;
    if object.get("type").and_then(Value::as_str) == Some("object")
        && object
            .get("properties")
            .and_then(Value::as_object)
            .is_none_or(Map::is_empty)
        && !object
            .get("additionalProperties")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    {
        return None;
    }

    let mut projected = Map::new();
    for key in ["description", "format", "minLength"] {
        if let Some(value) = object.get(key) {
            projected.insert(key.to_string(), value.clone());
        }
    }
    if let Some(schema_type) = projected_schema_type(object.get("type")) {
        projected.insert("type".to_string(), json!(schema_type));
    }
    if object
        .get("type")
        .and_then(Value::as_array)
        .is_some_and(|types| types.iter().any(|value| value.as_str() == Some("null")))
    {
        projected.insert("nullable".to_string(), Value::Bool(true));
    }
    if let Some(constant) = object.get("const") {
        projected.insert("enum".to_string(), Value::Array(vec![constant.clone()]));
    } else if let Some(values) = object.get("enum").and_then(Value::as_array) {
        let numeric = matches!(
            projected.get("type").and_then(Value::as_str),
            Some("integer" | "number")
        );
        if numeric {
            projected.insert("type".to_string(), json!("string"));
        }
        projected.insert(
            "enum".to_string(),
            Value::Array(
                values
                    .iter()
                    .map(|value| {
                        if numeric {
                            Value::String(match value {
                                Value::String(value) => value.clone(),
                                _ => value.to_string(),
                            })
                        } else {
                            value.clone()
                        }
                    })
                    .collect(),
            ),
        );
    }
    if let Some(properties) = object.get("properties").and_then(Value::as_object) {
        let properties = properties
            .iter()
            .filter_map(|(name, schema)| {
                gemini_tool_schema(schema).map(|schema| (name.clone(), schema))
            })
            .collect::<Map<_, _>>();
        if !properties.is_empty() {
            if let Some(required) = object.get("required").and_then(Value::as_array) {
                let required = required
                    .iter()
                    .filter_map(Value::as_str)
                    .filter(|name| properties.contains_key(*name))
                    .map(|name| Value::String(name.to_string()))
                    .collect::<Vec<_>>();
                if !required.is_empty() {
                    projected.insert("required".to_string(), Value::Array(required));
                }
            }
            projected.insert("properties".to_string(), Value::Object(properties));
        }
    }
    if let Some(items) = object.get("items") {
        if let Some(items) = gemini_tool_schema(items) {
            projected.insert("items".to_string(), items);
        }
    } else if projected.get("type").and_then(Value::as_str) == Some("array") {
        projected.insert("items".to_string(), json!({ "type": "string" }));
    }
    for key in ["allOf", "anyOf", "oneOf"] {
        if let Some(values) = object.get(key).and_then(Value::as_array) {
            let values = values
                .iter()
                .filter_map(gemini_tool_schema)
                .collect::<Vec<_>>();
            if !values.is_empty() {
                projected.insert(key.to_string(), Value::Array(values));
            }
        }
    }
    (!projected.is_empty()).then_some(Value::Object(projected))
}

fn projected_schema_type(value: Option<&Value>) -> Option<&str> {
    match value? {
        Value::String(value) if value != "null" => Some(value),
        Value::Array(values) => values
            .iter()
            .filter_map(Value::as_str)
            .find(|value| *value != "null"),
        _ => None,
    }
}

#[derive(Default)]
struct GeminiStreamState {
    text_open: bool,
    reasoning_open: bool,
    next_tool_call_id: u64,
    emitted_tool_call: bool,
    usage: Option<Usage>,
    finish_reason: Option<FinishReason>,
}

pub(super) fn stream(
    response: Response,
) -> impl Stream<Item = Result<LlmEvent, CurrentProviderError>> + Send {
    try_stream! {
        let mut state = GeminiStreamState::default();
        let mut frames = Box::pin(sse_frames_with_idle_timeout(
            response,
            DEFAULT_STREAM_IDLE_TIMEOUT,
        ));
        while let Some(frame) = frames.next().await {
            let frame = frame?;
            let event = serde_json::from_str::<Value>(&frame.data).map_err(|error| {
                CurrentProviderError::new(format!("解析 Gemini SSE event 失败: {error}"))
            })?;
            let events = reduce_event(&mut state, &event)?;
            let provider_error = events
                .iter()
                .any(|event| matches!(event, LlmEvent::ProviderError { .. }));
            for event in events {
                yield event;
            }
            if provider_error {
                drop(frames);
                for event in close_content(&mut state) {
                    yield event;
                }
                return;
            }
        }
        if state.finish_reason.is_some() || state.usage.is_some() {
            for event in finish_stream(&mut state) {
                yield event;
            }
            return;
        }
        for event in close_content(&mut state) {
            yield event;
        }
        yield LlmEvent::ProviderError {
            message: "Gemini GenerateContent stream ended before its terminal event".to_string(),
            classification: Some(FailureClassification::Transport),
            retryable: Some(true),
        };
    }
}

fn reduce_event(
    state: &mut GeminiStreamState,
    event: &Value,
) -> Result<Vec<LlmEvent>, CurrentProviderError> {
    if let Some(reason) = event
        .pointer("/promptFeedback/blockReason")
        .and_then(Value::as_str)
    {
        return Ok(vec![LlmEvent::ProviderError {
            message: format!("Gemini blocked the prompt: {reason}"),
            classification: Some(FailureClassification::ContentPolicy),
            retryable: Some(false),
        }]);
    }

    let mut events = Vec::new();
    if let Some(usage) = event.get("usageMetadata") {
        let usage = gemini_usage(usage);
        state.usage = Some(usage.clone());
        events.push(LlmEvent::Usage { usage });
    }
    let Some(candidate) = event
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|values| values.first())
    else {
        return Ok(events);
    };
    if let Some(parts) = candidate
        .pointer("/content/parts")
        .and_then(Value::as_array)
    {
        for part in parts {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                if text.is_empty() {
                    continue;
                }
                if part.get("thought").and_then(Value::as_bool) == Some(true) {
                    if state.text_open {
                        state.text_open = false;
                        events.push(LlmEvent::TextEnd {
                            id: "text-0".to_string(),
                        });
                    }
                    if !state.reasoning_open {
                        state.reasoning_open = true;
                        events.push(LlmEvent::ReasoningStart {
                            id: "reasoning-0".to_string(),
                        });
                    }
                    events.push(LlmEvent::ReasoningContentDelta {
                        id: "reasoning-0".to_string(),
                        text: text.to_string(),
                        content_index: 0,
                    });
                } else {
                    if state.reasoning_open {
                        state.reasoning_open = false;
                        events.push(LlmEvent::ReasoningEnd {
                            id: "reasoning-0".to_string(),
                        });
                    }
                    if !state.text_open {
                        state.text_open = true;
                        events.push(LlmEvent::TextStart {
                            id: "text-0".to_string(),
                        });
                    }
                    events.push(LlmEvent::TextDelta {
                        id: "text-0".to_string(),
                        text: text.to_string(),
                    });
                }
                continue;
            }
            if let Some(call) = part.get("functionCall").and_then(Value::as_object) {
                events.extend(close_content(state));
                let name = call
                    .get("name")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| {
                        CurrentProviderError::new("Gemini tool call omitted function name")
                    })?;
                let input = call.get("args").cloned().unwrap_or_else(|| json!({}));
                let id = format!("tool_{}", state.next_tool_call_id);
                state.next_tool_call_id += 1;
                state.emitted_tool_call = true;
                let arguments = serde_json::to_string(&input).map_err(|error| {
                    CurrentProviderError::new(format!("序列化 Gemini tool arguments 失败: {error}"))
                })?;
                events.push(LlmEvent::ToolInputStart {
                    id: id.clone(),
                    name: name.to_string(),
                });
                events.push(LlmEvent::ToolInputDelta {
                    id: id.clone(),
                    name: name.to_string(),
                    text: arguments.clone(),
                });
                events.push(LlmEvent::ToolInputEnd {
                    id: id.clone(),
                    name: name.to_string(),
                });
                events.push(LlmEvent::ToolCall {
                    id,
                    name: name.to_string(),
                    input,
                    raw_arguments: Some(arguments),
                    provider_executed: None,
                    provider_metadata: part
                        .get("thoughtSignature")
                        .and_then(Value::as_str)
                        .filter(|value| !value.trim().is_empty())
                        .map(|signature| {
                            BTreeMap::from([(
                                "google".to_string(),
                                json!({ "thoughtSignature": signature }),
                            )])
                        })
                        .unwrap_or_default(),
                });
                continue;
            }
            return Err(CurrentProviderError::invalid_request(
                "Gemini returned an unsupported output content part",
            ));
        }
    }
    if let Some(reason) = candidate.get("finishReason").and_then(Value::as_str) {
        state.finish_reason = Some(gemini_finish_reason(reason, state.emitted_tool_call));
    }
    Ok(events)
}

fn close_content(state: &mut GeminiStreamState) -> Vec<LlmEvent> {
    let mut events = Vec::new();
    if state.reasoning_open {
        state.reasoning_open = false;
        events.push(LlmEvent::ReasoningEnd {
            id: "reasoning-0".to_string(),
        });
    }
    if state.text_open {
        state.text_open = false;
        events.push(LlmEvent::TextEnd {
            id: "text-0".to_string(),
        });
    }
    events
}

fn finish_stream(state: &mut GeminiStreamState) -> Vec<LlmEvent> {
    let mut events = close_content(state);
    events.push(LlmEvent::Finish {
        reason: state.finish_reason.take().unwrap_or(FinishReason::Unknown),
        usage: state.usage.take(),
        response_id: None,
    });
    events
}

fn gemini_finish_reason(reason: &str, has_tool_calls: bool) -> FinishReason {
    match reason {
        "STOP" if has_tool_calls => FinishReason::ToolCall,
        "STOP" => FinishReason::Stop,
        "MAX_TOKENS" => FinishReason::Length,
        "IMAGE_SAFETY" | "RECITATION" | "SAFETY" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" => {
            FinishReason::ContentFilter
        }
        "MALFORMED_FUNCTION_CALL" => FinishReason::Error,
        _ => FinishReason::Unknown,
    }
}

fn gemini_usage(value: &Value) -> Usage {
    let input = u64_field(value, "promptTokenCount");
    let cached = u64_field(value, "cachedContentTokenCount");
    let visible_output = u64_field(value, "candidatesTokenCount");
    let reasoning = u64_field(value, "thoughtsTokenCount");
    let output = visible_output.map(|tokens| tokens.saturating_add(reasoning.unwrap_or_default()));
    Usage {
        input_tokens: input,
        output_tokens: output,
        non_cached_input_tokens: input
            .map(|tokens| tokens.saturating_sub(cached.unwrap_or_default())),
        cache_read_input_tokens: cached,
        reasoning_tokens: reasoning,
        total_tokens: u64_field(value, "totalTokenCount").or_else(|| {
            input
                .zip(output)
                .map(|(input, output)| input.saturating_add(output))
        }),
        provider_metadata: BTreeMap::from([("google".to_string(), value.clone())]),
        ..Usage::default()
    }
}

fn u64_field(value: &Value, key: &str) -> Option<u64> {
    value.get(key).and_then(Value::as_u64)
}
