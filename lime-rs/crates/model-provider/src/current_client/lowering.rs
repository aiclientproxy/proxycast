use super::{
    REQUEST_KIND_METADATA_KEY, SESSION_ID_METADATA_KEY, THREAD_ID_METADATA_KEY,
    TURN_ID_METADATA_KEY, X_CODEX_TURN_METADATA_KEY,
};
use crate::provider_capabilities::ProviderCapabilities;
use crate::provider_stream::RuntimeReplyProviderRequestWireShape;
use crate::runtime_provider::RuntimeProviderConfig;
use agent_protocol::ImageDetail;
use runtime_core::{CanonicalRequest, CanonicalRole, ContentPart, ToolResultValue};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

pub(super) fn chat_completions_request(
    config: &RuntimeProviderConfig,
    request: &CanonicalRequest,
    wire_shape: &RuntimeReplyProviderRequestWireShape,
    media_payloads: &BTreeMap<String, String>,
) -> Value {
    let mut messages = Vec::new();
    let system = text_from_parts(&request.system);
    if !system.is_empty() {
        messages.push(json!({ "role": "system", "content": system }));
    }
    messages.extend(
        request
            .messages
            .iter()
            .flat_map(|message| chat_message(message, media_payloads)),
    );
    let mut object = Map::from_iter([
        ("model".to_string(), json!(config.model_name)),
        ("messages".to_string(), Value::Array(messages)),
        ("stream".to_string(), Value::Bool(true)),
        (
            "stream_options".to_string(),
            json!({ "include_usage": true }),
        ),
    ]);
    if !request.tools.is_empty() {
        object.insert(
            "tools".to_string(),
            Value::Array(request.tools.iter().map(chat_tool).collect()),
        );
        if let Some(parallel_tool_calls) = wire_shape.parallel_tool_calls {
            object.insert(
                "parallel_tool_calls".to_string(),
                json!(parallel_tool_calls),
            );
        }
    }
    apply_generation_options(&mut object, request, "max_tokens", false);
    apply_chat_reasoning_effort(&mut object, config);
    apply_service_tier(&mut object, config);
    if let Some(enable_thinking) = request
        .provider_options
        .get("enable_thinking")
        .and_then(Value::as_bool)
    {
        object.insert(
            "chat_template_kwargs".to_string(),
            json!({ "enable_thinking": enable_thinking }),
        );
    }
    Value::Object(object)
}

pub(super) fn responses_request(
    config: &RuntimeProviderConfig,
    request: &CanonicalRequest,
    wire_shape: &RuntimeReplyProviderRequestWireShape,
    media_payloads: &BTreeMap<String, String>,
) -> Value {
    let mut input = Vec::new();
    for message in &request.messages {
        input.extend(responses_message(message, media_payloads));
    }
    if wire_shape.use_responses_lite {
        input
            .iter_mut()
            .for_each(strip_responses_lite_image_details);
    }

    let instructions = text_from_parts(&request.system);
    let instructions_in_input = wire_shape.instructions_location.as_deref() == Some("input_prefix")
        || (wire_shape.use_responses_lite && wire_shape.instructions_location.is_none());
    let tools_in_input = wire_shape.tools_location.as_deref() == Some("input_prefix")
        || (wire_shape.use_responses_lite && wire_shape.tools_location.is_none());
    let response_tools = request
        .tools
        .iter()
        .map(|tool| responses_tool(config, tool))
        .collect::<Vec<_>>();
    let mut input_prefix = Vec::new();
    if tools_in_input {
        input_prefix.push(json!({
            "type": "additional_tools",
            "role": "developer",
            "tools": response_tools,
        }));
    }
    if instructions_in_input && !instructions.is_empty() {
        input_prefix.push(json!({
            "type": "message",
            "role": "developer",
            "content": [{ "type": "input_text", "text": instructions }],
        }));
    }
    input.splice(0..0, input_prefix);

    let mut object = Map::from_iter([
        ("model".to_string(), json!(config.model_name)),
        ("input".to_string(), Value::Array(input)),
        ("stream".to_string(), Value::Bool(true)),
        ("store".to_string(), Value::Bool(false)),
    ]);
    if !instructions_in_input && !instructions.is_empty() {
        object.insert("instructions".to_string(), json!(instructions));
    }
    if !tools_in_input && !response_tools.is_empty() {
        object.insert("tools".to_string(), Value::Array(response_tools));
    }
    if let Some(parallel_tool_calls) = wire_shape.parallel_tool_calls {
        object.insert(
            "parallel_tool_calls".to_string(),
            json!(parallel_tool_calls),
        );
    }
    apply_generation_options(&mut object, request, "max_output_tokens", false);
    apply_responses_reasoning(&mut object, config, wire_shape);
    apply_text_verbosity(&mut object, wire_shape);
    apply_service_tier(&mut object, config);
    if let Some(client_metadata) = responses_client_metadata(request) {
        object.insert(
            "client_metadata".to_string(),
            Value::Object(client_metadata),
        );
    }
    Value::Object(object)
}

fn responses_client_metadata(request: &CanonicalRequest) -> Option<Map<String, Value>> {
    let session_id = request.metadata.get(SESSION_ID_METADATA_KEY)?.as_str()?;
    let thread_id = request.metadata.get(THREAD_ID_METADATA_KEY)?.as_str()?;
    let turn_id = request.metadata.get(TURN_ID_METADATA_KEY)?.as_str()?;

    let mut turn_metadata = request
        .metadata
        .iter()
        .filter_map(|(key, value)| {
            value
                .as_str()
                .map(|value| (key.clone(), Value::String(value.to_string())))
        })
        .collect::<Map<String, Value>>();
    turn_metadata.insert(
        REQUEST_KIND_METADATA_KEY.to_string(),
        Value::String("turn".to_string()),
    );
    let serialized = serde_json::to_string(&turn_metadata).ok()?;

    let client_metadata = Map::from_iter([
        (
            SESSION_ID_METADATA_KEY.to_string(),
            Value::String(session_id.to_string()),
        ),
        (
            THREAD_ID_METADATA_KEY.to_string(),
            Value::String(thread_id.to_string()),
        ),
        (
            TURN_ID_METADATA_KEY.to_string(),
            Value::String(turn_id.to_string()),
        ),
        (
            X_CODEX_TURN_METADATA_KEY.to_string(),
            Value::String(serialized),
        ),
    ]);
    Some(client_metadata)
}

pub(super) fn anthropic_request(
    config: &RuntimeProviderConfig,
    request: &CanonicalRequest,
    media_payloads: &BTreeMap<String, String>,
) -> Value {
    let messages = request
        .messages
        .iter()
        .flat_map(|message| anthropic_message(message, media_payloads))
        .collect::<Vec<_>>();
    let mut object = Map::from_iter([
        ("model".to_string(), json!(config.model_name)),
        ("messages".to_string(), Value::Array(messages)),
        ("max_tokens".to_string(), json!(4096)),
        ("stream".to_string(), Value::Bool(true)),
    ]);
    let system = text_from_parts(&request.system);
    if !system.is_empty() {
        object.insert("system".to_string(), json!(system));
    }
    if !request.tools.is_empty() {
        object.insert(
            "tools".to_string(),
            Value::Array(
                request
                    .tools
                    .iter()
                    .map(|tool| {
                        json!({
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": tool.input_schema,
                        })
                    })
                    .collect(),
            ),
        );
    }
    apply_generation_options(&mut object, request, "max_tokens", true);
    Value::Object(object)
}

fn apply_generation_options(
    object: &mut Map<String, Value>,
    request: &CanonicalRequest,
    max_tokens_key: &str,
    supports_top_k: bool,
) {
    if let Some(max_tokens) = request.generation.max_tokens {
        object.insert(max_tokens_key.to_string(), json!(max_tokens));
    }
    if let Some(temperature) = request.generation.temperature {
        object.insert("temperature".to_string(), json!(temperature));
    }
    if let Some(top_p) = request.generation.top_p {
        object.insert("top_p".to_string(), json!(top_p));
    }
    if supports_top_k {
        if let Some(top_k) = request.generation.top_k {
            object.insert("top_k".to_string(), json!(top_k));
        }
    }
}

fn apply_chat_reasoning_effort(object: &mut Map<String, Value>, config: &RuntimeProviderConfig) {
    let Some(effort) = config
        .reasoning_effort
        .as_deref()
        .filter(|effort| !effort.trim().is_empty())
    else {
        return;
    };

    object.insert("reasoning_effort".to_string(), json!(effort));
}

fn apply_responses_reasoning(
    object: &mut Map<String, Value>,
    config: &RuntimeProviderConfig,
    wire_shape: &RuntimeReplyProviderRequestWireShape,
) {
    let mut reasoning = Map::new();
    if let Some(effort) = config
        .reasoning_effort
        .as_deref()
        .map(str::trim)
        .filter(|effort| !effort.is_empty())
    {
        reasoning.insert("effort".to_string(), json!(effort));
    }
    if let Some(summary) = wire_shape
        .reasoning_summary
        .as_deref()
        .map(str::trim)
        .filter(|summary| !summary.is_empty() && *summary != "none")
    {
        reasoning.insert("summary".to_string(), json!(summary));
    }
    if let Some(context) = wire_shape
        .reasoning_context
        .as_deref()
        .map(str::trim)
        .filter(|context| matches!(*context, "all_turns" | "current_turn" | "auto"))
    {
        reasoning.insert("context".to_string(), json!(context));
    }
    if !reasoning.is_empty() {
        object.insert("reasoning".to_string(), Value::Object(reasoning));
    }
}

fn apply_text_verbosity(
    object: &mut Map<String, Value>,
    wire_shape: &RuntimeReplyProviderRequestWireShape,
) {
    let Some(verbosity) = wire_shape
        .text_verbosity
        .as_deref()
        .map(str::trim)
        .filter(|verbosity| matches!(*verbosity, "low" | "medium" | "high"))
    else {
        return;
    };
    object.insert("text".to_string(), json!({ "verbosity": verbosity }));
}

fn apply_service_tier(object: &mut Map<String, Value>, config: &RuntimeProviderConfig) {
    let Some(service_tier) = config
        .service_tier
        .as_deref()
        .map(str::trim)
        .filter(|service_tier| !service_tier.is_empty())
    else {
        return;
    };
    object.insert("service_tier".to_string(), json!(service_tier));
}

fn chat_message(
    message: &runtime_core::CanonicalMessage,
    media_payloads: &BTreeMap<String, String>,
) -> Vec<Value> {
    match message.role {
        CanonicalRole::Tool => message
            .content
            .iter()
            .filter_map(|content| match content {
                ContentPart::ToolResult {
                    id, result, error, ..
                } => Some(json!({
                    "role": "tool",
                    "tool_call_id": id,
                    "content": tool_result_text(result, error.as_deref()),
                })),
                _ => None,
            })
            .collect(),
        CanonicalRole::Assistant => {
            let text = text_from_parts(&message.content);
            let reasoning = reasoning_from_parts(&message.content);
            let tool_calls = message
                .content
                .iter()
                .filter_map(|content| match content {
                    ContentPart::ToolCall {
                        id, name, input, ..
                    } => Some(json!({
                        "id": id,
                        "type": "function",
                        "function": { "name": name, "arguments": input.to_string() },
                    })),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let mut value = json!({ "role": "assistant", "content": text });
            if !reasoning.is_empty() {
                value["reasoning_content"] = json!(reasoning);
            }
            if !tool_calls.is_empty() {
                value["tool_calls"] = Value::Array(tool_calls);
            }
            vec![value]
        }
        CanonicalRole::User | CanonicalRole::System | CanonicalRole::Developer => vec![json!({
            "role": wire_role(message.role),
            "content": chat_content(&message.content, media_payloads),
        })],
    }
}

fn wire_role(role: CanonicalRole) -> &'static str {
    match role {
        CanonicalRole::System => "system",
        CanonicalRole::Developer => "developer",
        CanonicalRole::User => "user",
        CanonicalRole::Assistant => "assistant",
        CanonicalRole::Tool => "tool",
    }
}

fn chat_content(content: &[ContentPart], media_payloads: &BTreeMap<String, String>) -> Value {
    let has_media = content
        .iter()
        .any(|part| matches!(part, ContentPart::Media { .. }));
    if !has_media {
        return json!(text_from_parts(content));
    }
    Value::Array(
        content
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text { text, .. } => Some(json!({ "type": "text", "text": text })),
                ContentPart::Media { uri, detail, .. } => {
                    let mut image_url = Map::from_iter([(
                        "url".to_string(),
                        json!(provider_media_uri(uri, media_payloads)),
                    )]);
                    if let Some(detail) = detail.map(openai_image_detail) {
                        image_url.insert("detail".to_string(), json!(detail));
                    }
                    Some(json!({
                        "type": "image_url",
                        "image_url": Value::Object(image_url),
                    }))
                }
                _ => None,
            })
            .collect(),
    )
}

fn chat_tool(tool: &runtime_core::CanonicalToolDefinition) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
            "strict": false,
        }
    })
}

fn responses_tool(
    config: &RuntimeProviderConfig,
    tool: &runtime_core::CanonicalToolDefinition,
) -> Value {
    let capabilities = ProviderCapabilities::from_runtime_config(config);
    if capabilities.web_search && is_web_search_tool_name(&tool.name) {
        return json!({
            "type": "web_search",
            "external_web_access": true,
        });
    }
    if capabilities.image_generation && is_image_generation_tool_name(&tool.name) {
        return json!({ "type": "image_generation" });
    }
    json!({
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema,
        "strict": false,
    })
}

fn is_web_search_tool_name(name: &str) -> bool {
    name == "WebSearch"
}

fn is_image_generation_tool_name(name: &str) -> bool {
    name == "ImageGeneration"
}

fn strip_responses_lite_image_details(value: &mut Value) {
    match value {
        Value::Object(object) => {
            if object.get("type").and_then(Value::as_str) == Some("input_image") {
                object.remove("detail");
            }
            object
                .values_mut()
                .for_each(strip_responses_lite_image_details);
        }
        Value::Array(values) => values
            .iter_mut()
            .for_each(strip_responses_lite_image_details),
        _ => {}
    }
}

fn responses_message(
    message: &runtime_core::CanonicalMessage,
    media_payloads: &BTreeMap<String, String>,
) -> Vec<Value> {
    match message.role {
        CanonicalRole::Tool => message
            .content
            .iter()
            .filter_map(|content| match content {
                ContentPart::ToolResult {
                    id, result, error, ..
                } => Some(json!({
                    "type": "function_call_output",
                    "call_id": id,
                    "output": tool_result_text(result, error.as_deref()),
                })),
                _ => None,
            })
            .collect(),
        CanonicalRole::Assistant => {
            let mut items = Vec::new();
            let text = text_from_parts(&message.content);
            if !text.is_empty() {
                items.push(json!({
                    "type": "message",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "text": text }],
                }));
            }
            for part in &message.content {
                match part {
                    ContentPart::ToolCall {
                        id, name, input, ..
                    } => items.push(json!({
                        "type": "function_call",
                        "call_id": id,
                        "name": name,
                        "arguments": input.to_string(),
                    })),
                    ContentPart::RawResponseItem { item } => items.push(item.clone()),
                    _ => {}
                }
            }
            items
        }
        CanonicalRole::User | CanonicalRole::System | CanonicalRole::Developer => vec![json!({
            "type": "message",
            "role": wire_role(message.role),
            "content": responses_input_content(&message.content, media_payloads),
        })],
    }
}

fn responses_input_content(
    content: &[ContentPart],
    media_payloads: &BTreeMap<String, String>,
) -> Vec<Value> {
    content
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text { text, .. } => Some(json!({ "type": "input_text", "text": text })),
            ContentPart::Media { uri, detail, .. } => {
                let mut image = Map::from_iter([
                    ("type".to_string(), json!("input_image")),
                    (
                        "image_url".to_string(),
                        json!(provider_media_uri(uri, media_payloads)),
                    ),
                ]);
                if let Some(detail) = detail.map(openai_image_detail) {
                    image.insert("detail".to_string(), json!(detail));
                }
                Some(Value::Object(image))
            }
            _ => None,
        })
        .collect()
}

fn anthropic_message(
    message: &runtime_core::CanonicalMessage,
    media_payloads: &BTreeMap<String, String>,
) -> Vec<Value> {
    let role = match message.role {
        CanonicalRole::Assistant => "assistant",
        _ => "user",
    };
    let content = message
        .content
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text { text, .. } => Some(json!({ "type": "text", "text": text })),
            ContentPart::Reasoning { text, .. } => {
                Some(json!({ "type": "thinking", "thinking": text }))
            }
            ContentPart::Media {
                uri, media_type, ..
            } => Some(json!({
                "type": "image",
                "source": anthropic_media_source(
                    provider_media_uri(uri, media_payloads),
                    media_type,
                ),
            })),
            ContentPart::ToolCall {
                id, name, input, ..
            } => Some(json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": input,
            })),
            ContentPart::ToolResult {
                id, result, error, ..
            } => Some(json!({
                "type": "tool_result",
                "tool_use_id": id,
                "content": tool_result_text(result, error.as_deref()),
            })),
            ContentPart::RawResponseItem { .. } => None,
        })
        .collect::<Vec<_>>();
    vec![json!({ "role": role, "content": content })]
}

fn provider_media_uri<'a>(uri: &'a str, media_payloads: &'a BTreeMap<String, String>) -> &'a str {
    media_payloads.get(uri).map(String::as_str).unwrap_or(uri)
}

fn openai_image_detail(detail: ImageDetail) -> &'static str {
    match detail {
        ImageDetail::Auto => "auto",
        ImageDetail::Low => "low",
        ImageDetail::High => "high",
        ImageDetail::Original => "original",
    }
}

fn anthropic_media_source(uri: &str, media_type: &str) -> Value {
    if let Some(encoded) = uri
        .strip_prefix("data:")
        .and_then(|value| value.split_once(','))
        .filter(|(metadata, _)| {
            metadata
                .split(';')
                .any(|part| part.eq_ignore_ascii_case("base64"))
        })
        .map(|(_, encoded)| encoded)
    {
        return json!({
            "type": "base64",
            "media_type": media_type,
            "data": encoded,
        });
    }
    json!({ "type": "url", "url": uri })
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

fn reasoning_from_parts(parts: &[ContentPart]) -> String {
    parts
        .iter()
        .filter_map(|part| match part {
            ContentPart::Reasoning { text, .. } => Some(text.as_str()),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn image_content() -> Vec<ContentPart> {
        vec![ContentPart::media("sidecar://image-1", "image/png").expect("canonical media")]
    }

    fn detailed_image_content(detail: ImageDetail) -> Vec<ContentPart> {
        vec![
            ContentPart::media_with_detail("sidecar://image-1", "image/png", Some(detail))
                .expect("canonical media"),
        ]
    }

    fn media_payloads() -> BTreeMap<String, String> {
        BTreeMap::from([(
            "sidecar://image-1".to_string(),
            "data:image/png;base64,abc".to_string(),
        )])
    }

    #[test]
    fn openai_compatible_image_parts_use_only_native_wire_fields() {
        let content = image_content();
        let payloads = media_payloads();

        let chat = chat_content(&content, &payloads);
        let responses = responses_input_content(&content, &payloads);

        assert_eq!(chat[0]["type"], "image_url");
        assert_eq!(
            chat[0]["image_url"],
            json!({ "url": "data:image/png;base64,abc" })
        );
        assert!(chat[0]["image_url"].get("media_type").is_none());
        assert_eq!(
            responses[0],
            json!({
                "type": "input_image",
                "image_url": "data:image/png;base64,abc"
            })
        );
        assert!(responses[0].get("media_type").is_none());
    }

    #[test]
    fn anthropic_base64_image_keeps_required_media_type() {
        let message = runtime_core::CanonicalMessage {
            id: None,
            role: CanonicalRole::User,
            content: image_content(),
            metadata: Default::default(),
        };

        let lowered = anthropic_message(&message, &media_payloads());

        assert_eq!(
            lowered[0]["content"][0]["source"],
            json!({
                "type": "base64",
                "media_type": "image/png",
                "data": "abc"
            })
        );
    }

    #[test]
    fn image_detail_is_lowered_only_to_supported_openai_fields() {
        let content = detailed_image_content(ImageDetail::Original);
        let payloads = media_payloads();

        let chat = chat_content(&content, &payloads);
        let responses = responses_input_content(&content, &payloads);
        let anthropic = anthropic_message(
            &runtime_core::CanonicalMessage {
                id: None,
                role: CanonicalRole::User,
                content,
                metadata: Default::default(),
            },
            &payloads,
        );

        assert_eq!(chat[0]["image_url"]["detail"], "original");
        assert_eq!(responses[0]["detail"], "original");
        assert!(anthropic[0]["content"][0].get("detail").is_none());
        assert!(anthropic[0]["content"][0]["source"].get("detail").is_none());
    }

    fn config(reasoning_effort: Option<&str>) -> RuntimeProviderConfig {
        RuntimeProviderConfig {
            provider_name: "openai".to_string(),
            provider_selector: Some("openai".to_string()),
            model_name: "gpt-5-codex".to_string(),
            api_key: Some("test".to_string()),
            auth: crate::runtime_provider::RuntimeProviderAuth::ApiKey,
            base_url: Some("https://gateway.example.com/v1".to_string()),
            api_version: None,
            credential_uuid: "credential-1".to_string(),
            reasoning_effort: reasoning_effort.map(str::to_string),
            service_tier: None,
            protocol: Some(crate::runtime_provider::RuntimeProviderProtocol::ChatCompletions),
            supports_websockets: false,
            toolshim: false,
            toolshim_model: None,
        }
    }

    #[test]
    fn openai_reasoning_controls_follow_native_wire_shapes() {
        let request = CanonicalRequest::text("gpt-5-codex", "hello");

        let chat = chat_completions_request(
            &config(Some("high")),
            &request,
            &RuntimeReplyProviderRequestWireShape::default(),
            &BTreeMap::new(),
        );
        assert_eq!(chat["reasoning_effort"], "high");

        let responses = responses_request(
            &config(Some("high")),
            &request,
            &RuntimeReplyProviderRequestWireShape::default(),
            &BTreeMap::new(),
        );
        assert_eq!(responses["reasoning"], json!({ "effort": "high" }));
    }

    fn request_with_tool() -> CanonicalRequest {
        let mut request = CanonicalRequest::text("gpt-5-codex", "hello");
        request.tools = vec![runtime_core::CanonicalToolDefinition {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            }),
            output_schema: None,
            metadata: Default::default(),
        }];
        request
    }

    #[test]
    fn parallel_tool_calls_preserves_true_false_and_unknown_wire_states() {
        let request = request_with_tool();

        for (parallel_tool_calls, expected) in [
            (Some(true), Some(true)),
            (Some(false), Some(false)),
            (None, None),
        ] {
            let wire_shape = RuntimeReplyProviderRequestWireShape {
                parallel_tool_calls,
                ..Default::default()
            };
            let responses =
                responses_request(&config(None), &request, &wire_shape, &BTreeMap::new());
            assert_eq!(
                responses
                    .get("parallel_tool_calls")
                    .and_then(Value::as_bool),
                expected,
                "parallel_tool_calls={parallel_tool_calls:?}"
            );
        }
    }

    #[test]
    fn responses_reasoning_summary_context_and_verbosity_share_native_objects() {
        let request = CanonicalRequest::text("gpt-5-codex", "hello");
        let wire_shape = RuntimeReplyProviderRequestWireShape {
            reasoning_context: Some("all_turns".to_string()),
            reasoning_summary: Some("detailed".to_string()),
            text_verbosity: Some("low".to_string()),
            ..Default::default()
        };

        let responses = responses_request(
            &config(Some("high")),
            &request,
            &wire_shape,
            &BTreeMap::new(),
        );

        assert_eq!(
            responses["reasoning"],
            json!({
                "effort": "high",
                "summary": "detailed",
                "context": "all_turns"
            })
        );
        assert_eq!(responses["text"], json!({ "verbosity": "low" }));
    }

    #[test]
    fn responses_lite_uses_input_prefix_and_strips_image_detail() {
        let mut request = request_with_tool();
        request.system = vec![ContentPart::text("Follow repository rules")];
        request.messages[0].content = detailed_image_content(ImageDetail::Original);
        let wire_shape = RuntimeReplyProviderRequestWireShape {
            use_responses_lite: true,
            instructions_location: Some("input_prefix".to_string()),
            tools_location: Some("input_prefix".to_string()),
            reasoning_context: Some("all_turns".to_string()),
            parallel_tool_calls: Some(false),
            ..Default::default()
        };

        let responses = responses_request(&config(None), &request, &wire_shape, &media_payloads());

        assert_eq!(
            responses,
            json!({
                "model": "gpt-5-codex",
                "input": [
                    {
                        "type": "additional_tools",
                        "role": "developer",
                        "tools": [{
                            "type": "function",
                            "name": "read_file",
                            "description": "Read a file",
                            "parameters": {
                                "type": "object",
                                "properties": { "path": { "type": "string" } },
                                "required": ["path"]
                            },
                            "strict": false
                        }]
                    },
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [{
                            "type": "input_text",
                            "text": "Follow repository rules"
                        }]
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{
                            "type": "input_image",
                            "image_url": "data:image/png;base64,abc"
                        }]
                    }
                ],
                "stream": true,
                "store": false,
                "parallel_tool_calls": false,
                "reasoning": { "context": "all_turns" }
            })
        );
    }

    #[test]
    fn responses_only_emits_codex_turn_identity_and_protects_reserved_metadata() {
        let metadata = super::super::CurrentProviderRequestMetadata::new(
            "session-1",
            "thread-1",
            "turn-1",
            Some("thread-source".to_string()),
        )
        .with_extra(runtime_core::ProviderMetadata::from([
            ("session_id".to_string(), json!("overridden-session")),
            (
                "forked_from_thread_id".to_string(),
                json!("overridden-source"),
            ),
            ("request_kind".to_string(), json!("overridden-kind")),
            ("workspace_kind".to_string(), json!("local")),
            ("ignored_object".to_string(), json!({"value": true})),
        ]));
        let provider_request = super::super::CurrentProviderRequest::new(vec![
            super::super::CurrentProviderMessage::user(vec![
                super::super::CurrentProviderContent::Text("hello".to_string()),
            ]),
        ])
        .with_metadata(metadata);
        let canonical = provider_request
            .into_canonical("gpt-5-codex")
            .expect("canonical request");

        let responses = responses_request(
            &config(Some("high")),
            &canonical,
            &RuntimeReplyProviderRequestWireShape::default(),
            &BTreeMap::new(),
        );
        let client_metadata = responses["client_metadata"]
            .as_object()
            .expect("Responses client_metadata");
        assert_eq!(client_metadata["session_id"], "session-1");
        assert_eq!(client_metadata["thread_id"], "thread-1");
        assert_eq!(client_metadata["turn_id"], "turn-1");
        assert!(!client_metadata.contains_key("forked_from_thread_id"));
        let turn_metadata: Value = serde_json::from_str(
            client_metadata["x-codex-turn-metadata"]
                .as_str()
                .expect("serialized turn metadata"),
        )
        .expect("valid turn metadata JSON");
        assert_eq!(turn_metadata["session_id"], "session-1");
        assert_eq!(turn_metadata["thread_id"], "thread-1");
        assert_eq!(turn_metadata["turn_id"], "turn-1");
        assert_eq!(turn_metadata["forked_from_thread_id"], "thread-source");
        assert_eq!(turn_metadata["request_kind"], "turn");
        assert_eq!(turn_metadata["workspace_kind"], "local");
        assert!(turn_metadata.get("ignored_object").is_none());

        let chat = chat_completions_request(
            &config(Some("high")),
            &canonical,
            &RuntimeReplyProviderRequestWireShape::default(),
            &BTreeMap::new(),
        );
        let anthropic = anthropic_request(&config(None), &canonical, &BTreeMap::new());
        assert!(chat.get("client_metadata").is_none());
        assert!(anthropic.get("client_metadata").is_none());
    }

    #[test]
    fn blank_reasoning_effort_is_omitted() {
        let request = CanonicalRequest::text("gpt-5-codex", "hello");
        let value = chat_completions_request(
            &config(Some("  ")),
            &request,
            &RuntimeReplyProviderRequestWireShape::default(),
            &BTreeMap::new(),
        );

        assert!(value.get("reasoning_effort").is_none());
    }

    #[test]
    fn openai_service_tier_uses_native_wire_field() {
        let request = CanonicalRequest::text("gpt-5-codex", "hello");
        let mut config = config(None);
        config.service_tier = Some("priority".to_string());

        let chat = chat_completions_request(
            &config,
            &request,
            &RuntimeReplyProviderRequestWireShape::default(),
            &BTreeMap::new(),
        );
        let responses = responses_request(
            &config,
            &request,
            &RuntimeReplyProviderRequestWireShape::default(),
            &BTreeMap::new(),
        );

        assert_eq!(chat["service_tier"], "priority");
        assert_eq!(responses["service_tier"], "priority");
    }

    #[test]
    fn chat_assistant_history_preserves_reasoning_content() {
        let message = runtime_core::CanonicalMessage {
            id: None,
            role: CanonicalRole::Assistant,
            content: vec![
                ContentPart::Reasoning {
                    text: "先检查输入。".to_string(),
                    encrypted: Some("opaque-reasoning".to_string()),
                    metadata: Default::default(),
                },
                ContentPart::text("结果"),
            ],
            metadata: Default::default(),
        };

        let lowered = chat_message(&message, &BTreeMap::new());

        assert_eq!(lowered[0]["reasoning_content"], "先检查输入。");
        assert_eq!(lowered[0]["content"], "结果");
    }
}
