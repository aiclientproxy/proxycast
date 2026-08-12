//! Current provider 调用边界。
//!
//! HTTP server 只负责把公开的 OpenAI/Anthropic 请求映射到 provider-neutral
//! `CurrentProviderRequest`，网络请求和 provider wire lowering 统一由
//! `model-provider::CurrentProviderClient` 承担。

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use futures::StreamExt;
use lime_core::models::anthropic::AnthropicMessagesRequest;
use lime_core::models::openai::{
    ChatCompletionRequest, ChatMessage, ContentPart as OpenAiContentPart, MessageContent,
};
use lime_core::models::{RuntimeCredentialData, RuntimeProviderCredential, RuntimeProviderType};
use model_provider::current_client::{
    CanonicalLlmEvent, CurrentProviderClient, CurrentProviderContent, CurrentProviderError,
    CurrentProviderMessage, CurrentProviderRequest, CurrentProviderRole, CurrentProviderTool,
    CurrentProviderToolCall, CurrentProviderToolResult, FinishReason, GenerationOptions,
};
use model_provider::current_client::{CurrentProviderStream, Usage};
use model_provider::runtime_provider::{
    RuntimeProviderAuth, RuntimeProviderConfig, RuntimeProviderProtocol,
};
use serde_json::{json, Value};

use crate::AppState;

mod streaming;
use streaming::stream_provider_response;

#[derive(Clone, Copy)]
enum OutputFormat {
    OpenAi,
    Anthropic,
}

struct CollectedProviderOutput {
    text: String,
    reasoning: String,
    tool_calls: Vec<CollectedToolCall>,
    usage: Usage,
    finish_reason: FinishReason,
    response_id: Option<String>,
}

impl Default for CollectedProviderOutput {
    fn default() -> Self {
        Self {
            text: String::new(),
            reasoning: String::new(),
            tool_calls: Vec::new(),
            usage: Usage::default(),
            finish_reason: FinishReason::Stop,
            response_id: None,
        }
    }
}

struct CollectedToolCall {
    id: String,
    name: String,
    input: Value,
}

/// 根据凭证建立唯一 current provider client。
///
/// Provider 类型没有 current wire adapter 时直接在边界返回错误。
fn provider_client_for_credential(
    credential: &RuntimeProviderCredential,
    model: &str,
) -> Result<CurrentProviderClient, CurrentProviderError> {
    let (provider_name, protocol, api_key, base_url) = match &credential.credential {
        RuntimeCredentialData::OpenAIKey { api_key, base_url } => {
            if credential.provider_type == RuntimeProviderType::AzureOpenai {
                (
                    "azure",
                    RuntimeProviderProtocol::AzureResponses,
                    api_key,
                    base_url,
                )
            } else {
                (
                    "openai",
                    RuntimeProviderProtocol::ChatCompletions,
                    api_key,
                    base_url,
                )
            }
        }
        RuntimeCredentialData::ClaudeKey { api_key, base_url }
        | RuntimeCredentialData::AnthropicKey { api_key, base_url } => (
            "anthropic",
            RuntimeProviderProtocol::AnthropicMessages,
            api_key,
            base_url,
        ),
        RuntimeCredentialData::GeminiApiKey {
            api_key, base_url, ..
        } => (
            "google",
            RuntimeProviderProtocol::GeminiGenerateContent,
            api_key,
            base_url,
        ),
        RuntimeCredentialData::VertexKey {
            api_key, base_url, ..
        } => (
            "gcpvertexai",
            RuntimeProviderProtocol::VertexGemini,
            api_key,
            base_url,
        ),
    };

    CurrentProviderClient::new(RuntimeProviderConfig {
        provider_name: provider_name.to_string(),
        provider_selector: Some(credential.provider_type.to_string()),
        model_name: model.to_string(),
        api_key: Some(api_key.clone()),
        auth: RuntimeProviderAuth::ApiKey,
        base_url: base_url.clone(),
        api_version: None,
        credential_uuid: credential.uuid.clone(),
        reasoning_effort: None,
        service_tier: None,
        protocol: Some(protocol),
        supports_websockets: false,
        toolshim: false,
        toolshim_model: None,
    })
}

fn error_status(error: &CurrentProviderError) -> StatusCode {
    if let Some(status) = error
        .status
        .and_then(|status| StatusCode::from_u16(status).ok())
    {
        return status;
    }
    match error.classification {
        Some(model_provider::current_client::FailureClassification::InvalidRequest)
        | Some(model_provider::current_client::FailureClassification::ContextOverflow)
        | Some(model_provider::current_client::FailureClassification::ContentPolicy) => {
            StatusCode::BAD_REQUEST
        }
        Some(model_provider::current_client::FailureClassification::Authentication) => {
            StatusCode::UNAUTHORIZED
        }
        Some(model_provider::current_client::FailureClassification::Permission) => {
            StatusCode::FORBIDDEN
        }
        Some(model_provider::current_client::FailureClassification::RateLimit) => {
            StatusCode::TOO_MANY_REQUESTS
        }
        _ => StatusCode::BAD_GATEWAY,
    }
}

fn provider_error_response(error: &CurrentProviderError) -> Response {
    let status = error_status(error);
    (
        status,
        Json(json!({
            "error": {
                "message": error.message,
                "type": "provider_error"
            }
        })),
    )
        .into_response()
}

async fn collect_provider_output(
    mut stream: CurrentProviderStream,
) -> Result<CollectedProviderOutput, CurrentProviderError> {
    let mut output = CollectedProviderOutput {
        finish_reason: FinishReason::Stop,
        ..CollectedProviderOutput::default()
    };

    while let Some(event) = stream.next().await {
        let event = event?;
        match event {
            CanonicalLlmEvent::TextDelta { text, .. } => output.text.push_str(&text),
            CanonicalLlmEvent::ReasoningContentDelta { text, .. } => {
                output.reasoning.push_str(&text)
            }
            CanonicalLlmEvent::ToolCall {
                id, name, input, ..
            } => output
                .tool_calls
                .push(CollectedToolCall { id, name, input }),
            CanonicalLlmEvent::CustomToolCall { .. } => {
                return Err(CurrentProviderError::invalid_request(
                    "legacy provider call endpoint cannot expose custom tool calls",
                ));
            }
            CanonicalLlmEvent::Usage { usage } => output.usage = usage,
            CanonicalLlmEvent::Finish {
                reason,
                usage,
                response_id,
            } => {
                output.finish_reason = reason;
                if let Some(usage) = usage {
                    output.usage = usage;
                }
                output.response_id = response_id;
            }
            CanonicalLlmEvent::StepFinish { reason, usage, .. } => {
                output.finish_reason = reason;
                if let Some(usage) = usage {
                    output.usage = usage;
                }
            }
            CanonicalLlmEvent::ProviderError {
                message,
                classification,
                retryable,
            } => {
                return Err(CurrentProviderError {
                    message,
                    status: None,
                    classification,
                    retryable: retryable.unwrap_or(false),
                    retry_after: None,
                });
            }
            CanonicalLlmEvent::TextStart { .. }
            | CanonicalLlmEvent::TextEnd { .. }
            | CanonicalLlmEvent::ReasoningStart { .. }
            | CanonicalLlmEvent::ReasoningSummaryDelta { .. }
            | CanonicalLlmEvent::ReasoningSummaryPartAdded { .. }
            | CanonicalLlmEvent::ReasoningEnd { .. }
            | CanonicalLlmEvent::ToolInputStart { .. }
            | CanonicalLlmEvent::ToolInputDelta { .. }
            | CanonicalLlmEvent::ToolInputEnd { .. }
            | CanonicalLlmEvent::ToolResult { .. }
            | CanonicalLlmEvent::ToolError { .. }
            | CanonicalLlmEvent::ServerModel { .. }
            | CanonicalLlmEvent::ModelReroute { .. }
            | CanonicalLlmEvent::ModelVerification { .. }
            | CanonicalLlmEvent::TurnModerationMetadata { .. }
            | CanonicalLlmEvent::StepStart { .. } => {}
        }
    }

    Ok(output)
}

fn usage_value(usage: &Usage) -> (u64, u64) {
    (
        usage.input_tokens.unwrap_or_default(),
        usage.output_tokens.unwrap_or_default(),
    )
}

fn finish_reason_openai(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::ToolCall => "tool_calls",
        FinishReason::Length => "length",
        FinishReason::ContentFilter => "content_filter",
        FinishReason::Error => "error",
        FinishReason::Stop | FinishReason::Unknown => "stop",
    }
}

fn finish_reason_anthropic(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::ToolCall => "tool_use",
        FinishReason::Length => "max_tokens",
        FinishReason::ContentFilter => "refusal",
        FinishReason::Error => "end_turn",
        FinishReason::Stop | FinishReason::Unknown => "end_turn",
    }
}

fn openai_response(model: &str, output: &CollectedProviderOutput) -> Value {
    let (input_tokens, output_tokens) = usage_value(&output.usage);
    let tool_calls = (!output.tool_calls.is_empty()).then(|| {
        output
            .tool_calls
            .iter()
            .map(|call| {
                json!({
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.input.to_string()
                    }
                })
            })
            .collect::<Vec<_>>()
    });
    json!({
        "id": output.response_id.clone().unwrap_or_else(|| format!("chatcmpl-{}", uuid::Uuid::new_v4())),
        "object": "chat.completion",
        "created": chrono::Utc::now().timestamp().max(0) as u64,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": (!output.text.is_empty()).then_some(output.text.clone()),
                "tool_calls": tool_calls,
                "reasoning_content": (!output.reasoning.is_empty()).then_some(output.reasoning.clone())
            },
            "finish_reason": finish_reason_openai(output.finish_reason)
        }],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens.saturating_add(output_tokens)
        }
    })
}

fn anthropic_response(model: &str, output: &CollectedProviderOutput) -> Value {
    let (input_tokens, output_tokens) = usage_value(&output.usage);
    let mut content = Vec::new();
    if !output.text.is_empty() {
        content.push(json!({ "type": "text", "text": output.text }));
    }
    if !output.reasoning.is_empty() {
        content.push(json!({ "type": "thinking", "thinking": output.reasoning, "signature": "" }));
    }
    content.extend(output.tool_calls.iter().map(|call| {
        json!({
            "type": "tool_use",
            "id": call.id,
            "name": call.name,
            "input": call.input
        })
    }));
    json!({
        "id": output.response_id.clone().unwrap_or_else(|| format!("msg_{}", uuid::Uuid::new_v4())),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": finish_reason_anthropic(output.finish_reason),
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    })
}

async fn call_current_provider(
    state: &AppState,
    credential: &RuntimeProviderCredential,
    request: CurrentProviderRequest,
    model: &str,
    stream: bool,
    format: OutputFormat,
) -> Response {
    let client = match provider_client_for_credential(credential, model) {
        Ok(client) => client,
        Err(error) => return provider_error_response(&error),
    };
    let provider_stream = match client.stream(request).await {
        Ok(stream) => stream,
        Err(error) => {
            if let Some(db) = &state.db {
                let _ = state.mark_credential_unhealthy(db, &credential.uuid, Some(&error.message));
            }
            return provider_error_response(&error);
        }
    };

    if stream {
        if let Some(db) = &state.db {
            let _ = state.mark_credential_healthy(db, &credential.uuid, Some(model));
            let _ = state.record_credential_usage(db, &credential.uuid);
        }
        return stream_provider_response(provider_stream, model, format);
    }

    let output = match collect_provider_output(provider_stream).await {
        Ok(output) => output,
        Err(error) => {
            if let Some(db) = &state.db {
                let _ = state.mark_credential_unhealthy(db, &credential.uuid, Some(&error.message));
            }
            return provider_error_response(&error);
        }
    };
    if let Some(db) = &state.db {
        let _ = state.mark_credential_healthy(db, &credential.uuid, Some(model));
        let _ = state.record_credential_usage(db, &credential.uuid);
    }

    let body = match format {
        OutputFormat::OpenAi => openai_response(model, &output),
        OutputFormat::Anthropic => anthropic_response(model, &output),
    };
    (StatusCode::OK, Json(body)).into_response()
}

fn image_part(url: &str, index: usize) -> CurrentProviderContent {
    let (uri, provider_data, media_type) = if let Some((metadata, _)) = url
        .strip_prefix("data:")
        .and_then(|value| value.split_once(','))
    {
        let media_type = metadata.split(';').next().unwrap_or("image/png");
        (
            format!("attachment://request-image-{index}"),
            Some(url.to_string()),
            media_type.to_string(),
        )
    } else {
        (url.to_string(), None, "image/*".to_string())
    };
    CurrentProviderContent::Image {
        uri,
        media_type,
        provider_data,
        detail: None,
    }
}

fn openai_message_content(
    message: &ChatMessage,
    image_index: &mut usize,
) -> Result<Vec<CurrentProviderContent>, String> {
    let mut content = Vec::new();
    if let Some(message_content) = &message.content {
        match message_content {
            MessageContent::Text(text) => content.push(CurrentProviderContent::Text(text.clone())),
            MessageContent::Parts(parts) => {
                for part in parts {
                    match part {
                        OpenAiContentPart::Text { text } => {
                            content.push(CurrentProviderContent::Text(text.clone()))
                        }
                        OpenAiContentPart::ImageUrl { image_url } => {
                            content.push(image_part(&image_url.url, *image_index));
                            *image_index = image_index.saturating_add(1);
                        }
                    }
                }
            }
        }
    }
    if let Some(tool_calls) = &message.tool_calls {
        for call in tool_calls {
            let arguments = serde_json::from_str(&call.function.arguments).unwrap_or_else(|_| {
                json!({
                    "raw_arguments": call.function.arguments
                })
            });
            content.push(CurrentProviderContent::ToolCall(
                CurrentProviderToolCall::new(
                    call.id.clone(),
                    call.function.name.clone(),
                    arguments,
                ),
            ));
        }
    }
    if let Some(reasoning) = &message.reasoning_content {
        if !reasoning.is_empty() {
            content.push(CurrentProviderContent::Reasoning(reasoning.clone()));
        }
    }
    if message.role.eq_ignore_ascii_case("tool") {
        let output = message.get_content_text();
        content.clear();
        content.push(CurrentProviderContent::ToolResult(
            CurrentProviderToolResult {
                call_id: message
                    .tool_call_id
                    .clone()
                    .unwrap_or_else(|| "tool".to_string()),
                name: "tool".to_string(),
                success: true,
                output,
                error: None,
            },
        ));
    }
    Ok(content)
}

fn openai_request_to_current(
    request: &ChatCompletionRequest,
) -> Result<CurrentProviderRequest, String> {
    let mut system = Vec::new();
    let mut messages = Vec::new();
    let mut image_index = 0usize;
    for message in &request.messages {
        let content = openai_message_content(message, &mut image_index)?;
        if message.role.eq_ignore_ascii_case("system") {
            if let Some(text) = content.iter().find_map(|part| match part {
                CurrentProviderContent::Text(text) => Some(text.as_str()),
                _ => None,
            }) {
                system.push(text.to_string());
            }
            continue;
        }
        let role = if message.role.eq_ignore_ascii_case("assistant") {
            CurrentProviderRole::Assistant
        } else if message.role.eq_ignore_ascii_case("developer") {
            CurrentProviderRole::Developer
        } else if message.role.eq_ignore_ascii_case("tool") {
            CurrentProviderRole::Tool
        } else {
            CurrentProviderRole::User
        };
        messages.push(CurrentProviderMessage { role, content });
    }
    let tools = request
        .tools
        .as_ref()
        .into_iter()
        .flat_map(|tools| tools.iter())
        .filter_map(|tool| match tool {
            agent_protocol::openai::Tool::Function { function } => {
                Some(CurrentProviderTool::function(
                    function.name.clone(),
                    function.description.clone().unwrap_or_default(),
                    function.parameters.clone().unwrap_or_else(|| json!({})),
                ))
            }
            agent_protocol::openai::Tool::WebSearch
            | agent_protocol::openai::Tool::WebSearch20250305 => None,
        })
        .collect();
    Ok(CurrentProviderRequest::new(messages)
        .with_system_prompt((!system.is_empty()).then(|| system.join("\n\n")))
        .with_tools(tools)
        .with_generation(GenerationOptions {
            max_tokens: request.max_tokens,
            temperature: request.temperature.map(f64::from),
            top_p: request.top_p.map(f64::from),
            top_k: None,
        }))
}

fn anthropic_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn anthropic_content(
    value: &Value,
    image_index: &mut usize,
) -> Result<Vec<CurrentProviderContent>, String> {
    let mut content = Vec::new();
    match value {
        Value::String(text) => content.push(CurrentProviderContent::Text(text.clone())),
        Value::Array(parts) => {
            for part in parts {
                let kind = part.get("type").and_then(Value::as_str).unwrap_or("text");
                match kind {
                    "text" | "thinking" => {
                        if let Some(text) = part
                            .get("text")
                            .or_else(|| part.get("thinking"))
                            .and_then(Value::as_str)
                        {
                            content.push(if kind == "thinking" {
                                CurrentProviderContent::Reasoning(text.to_string())
                            } else {
                                CurrentProviderContent::Text(text.to_string())
                            });
                        }
                    }
                    "image" => {
                        let source = part.get("source").cloned().unwrap_or_default();
                        let media_type = source
                            .get("media_type")
                            .and_then(Value::as_str)
                            .unwrap_or("image/png");
                        let source_type = source.get("type").and_then(Value::as_str).unwrap_or("");
                        let data = source.get("data").and_then(Value::as_str);
                        let uri = if source_type == "base64" {
                            format!("attachment://request-image-{image_index}")
                        } else {
                            source
                                .get("url")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string()
                        };
                        content.push(CurrentProviderContent::Image {
                            uri,
                            media_type: media_type.to_string(),
                            provider_data: data
                                .map(|data| format!("data:{media_type};base64,{data}")),
                            detail: None,
                        });
                        *image_index = image_index.saturating_add(1);
                    }
                    "tool_use" => content.push(CurrentProviderContent::ToolCall(
                        CurrentProviderToolCall::new(
                            part.get("id")
                                .and_then(Value::as_str)
                                .unwrap_or("tool")
                                .to_string(),
                            part.get("name")
                                .and_then(Value::as_str)
                                .unwrap_or("tool")
                                .to_string(),
                            part.get("input").cloned().unwrap_or_else(|| json!({})),
                        ),
                    )),
                    "tool_result" => content.push(CurrentProviderContent::ToolResult(
                        CurrentProviderToolResult {
                            call_id: part
                                .get("tool_use_id")
                                .and_then(Value::as_str)
                                .unwrap_or("tool")
                                .to_string(),
                            name: "tool".to_string(),
                            success: true,
                            output: anthropic_text(part.get("content").unwrap_or(&Value::Null)),
                            error: None,
                        },
                    )),
                    _ => return Err(format!("unsupported Anthropic content block: {kind}")),
                }
            }
        }
        _ => return Err("Anthropic message content must be string or array".to_string()),
    }
    Ok(content)
}

fn anthropic_request_to_current(
    request: &AnthropicMessagesRequest,
) -> Result<CurrentProviderRequest, String> {
    let mut messages = Vec::new();
    let mut image_index = 0usize;
    for message in &request.messages {
        let role = if message.role.eq_ignore_ascii_case("assistant") {
            CurrentProviderRole::Assistant
        } else {
            CurrentProviderRole::User
        };
        messages.push(CurrentProviderMessage {
            role,
            content: anthropic_content(&message.content, &mut image_index)?,
        });
    }
    let tools = request
        .tools
        .as_ref()
        .into_iter()
        .flat_map(|tools| tools.iter())
        .map(|tool| {
            CurrentProviderTool::function(
                tool.name.clone(),
                tool.description.clone().unwrap_or_default(),
                tool.input_schema.clone().unwrap_or_else(|| json!({})),
            )
        })
        .collect();
    Ok(CurrentProviderRequest::new(messages)
        .with_system_prompt(request.system.as_ref().map(anthropic_text))
        .with_tools(tools)
        .with_generation(GenerationOptions {
            max_tokens: request.max_tokens,
            temperature: request.temperature.map(f64::from),
            top_p: None,
            top_k: None,
        }))
}

/// 根据凭证调用 Provider（Anthropic wire）。
pub async fn call_provider_anthropic(
    state: &AppState,
    credential: &RuntimeProviderCredential,
    request: &AnthropicMessagesRequest,
    _flow_id: Option<&str>,
) -> Response {
    let current_request = match anthropic_request_to_current(request) {
        Ok(request) => request,
        Err(message) => {
            return provider_error_response(&CurrentProviderError::invalid_request(message));
        }
    };
    call_current_provider(
        state,
        credential,
        current_request,
        &request.model,
        request.stream,
        OutputFormat::Anthropic,
    )
    .await
}

/// 根据凭证调用 Provider（OpenAI wire）。
pub async fn call_provider_openai(
    state: &AppState,
    credential: &RuntimeProviderCredential,
    request: &ChatCompletionRequest,
    _flow_id: Option<&str>,
) -> Response {
    let current_request = match openai_request_to_current(request) {
        Ok(request) => request,
        Err(message) => {
            return provider_error_response(&CurrentProviderError::invalid_request(message));
        }
    };
    call_current_provider(
        state,
        credential,
        current_request,
        &request.model,
        request.stream,
        OutputFormat::OpenAi,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::header;
    use futures::stream;
    use lime_core::models::openai::ChatMessage;
    use model_provider::current_client::{FailureClassification, ModelVerification};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::{mpsc, oneshot};
    use tokio::time::timeout;

    fn receiver_stream(
        receiver: mpsc::Receiver<Result<CanonicalLlmEvent, CurrentProviderError>>,
    ) -> CurrentProviderStream {
        Box::pin(stream::unfold(receiver, |mut receiver| async move {
            receiver.recv().await.map(|event| (event, receiver))
        }))
    }

    async fn next_body_chunk(body: &mut axum::body::BodyDataStream) -> String {
        let bytes = timeout(Duration::from_secs(1), body.next())
            .await
            .expect("stream chunk timeout")
            .expect("stream ended before chunk")
            .expect("body chunk");
        String::from_utf8(bytes.to_vec()).expect("SSE chunk must be UTF-8")
    }

    #[test]
    fn openai_system_message_is_projected_to_current_system_prompt() {
        let request = ChatCompletionRequest {
            model: "gpt-5".to_string(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: Some(MessageContent::Text("be concise".to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: Some(MessageContent::Text("hello".to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                },
            ],
            temperature: None,
            max_tokens: None,
            top_p: None,
            stream: false,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
        };
        let current = openai_request_to_current(&request).expect("request projection");
        assert_eq!(current.system_prompt.as_deref(), Some("be concise"));
        assert_eq!(current.messages.len(), 1);
    }

    #[test]
    fn gemini_credential_uses_current_generate_content_adapter() {
        let credential = RuntimeProviderCredential::new(
            lime_core::models::RuntimeProviderType::GeminiApiKey,
            RuntimeCredentialData::GeminiApiKey {
                api_key: "key".to_string(),
                base_url: None,
                excluded_models: Vec::new(),
            },
        );
        let client = provider_client_for_credential(&credential, "gemini-2.5-flash")
            .expect("Gemini current provider client");
        assert_eq!(client.config().provider_name, "google");
        assert_eq!(
            client.protocol().expect("Gemini protocol"),
            model_provider::ModelProviderProtocol::GeminiGenerateContent
        );
    }

    #[tokio::test]
    async fn openai_stream_emits_delta_before_terminal() {
        let (sender, receiver) = mpsc::channel(4);
        let response =
            stream_provider_response(receiver_stream(receiver), "gpt-5", OutputFormat::OpenAi);
        assert_eq!(
            response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("text/event-stream")
        );
        let mut body = response.into_body().into_data_stream();

        assert!(next_body_chunk(&mut body).await.contains("assistant"));
        sender
            .send(Ok(CanonicalLlmEvent::TextDelta {
                id: "text-0".to_string(),
                text: "first".to_string(),
            }))
            .await
            .expect("send text delta");
        let delta = next_body_chunk(&mut body).await;
        assert!(delta.contains("first"));
        assert!(!delta.contains("[DONE]"));
        assert!(timeout(Duration::from_millis(25), body.next())
            .await
            .is_err());

        sender
            .send(Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::Stop,
                usage: Some(Usage {
                    input_tokens: Some(3),
                    output_tokens: Some(1),
                    ..Usage::default()
                }),
                response_id: Some("upstream-response".to_string()),
            }))
            .await
            .expect("send finish");
        assert!(next_body_chunk(&mut body).await.contains("finish_reason"));
        assert!(next_body_chunk(&mut body).await.contains("prompt_tokens"));
        assert_eq!(next_body_chunk(&mut body).await, "data: [DONE]\n\n");
    }

    #[tokio::test]
    async fn anthropic_stream_preserves_block_and_terminal_order() {
        let events = vec![
            Ok(CanonicalLlmEvent::TextStart {
                id: "text-0".to_string(),
            }),
            Ok(CanonicalLlmEvent::TextDelta {
                id: "text-0".to_string(),
                text: "hello".to_string(),
            }),
            Ok(CanonicalLlmEvent::TextEnd {
                id: "text-0".to_string(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::Stop,
                usage: None,
                response_id: None,
            }),
        ];
        let response = stream_provider_response(
            Box::pin(stream::iter(events)),
            "claude-sonnet",
            OutputFormat::Anthropic,
        );
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .expect("collect finite SSE body");
        let body = String::from_utf8(body.to_vec()).expect("SSE body UTF-8");

        let message_start = body.find("event: message_start").expect("message start");
        let block_start = body
            .find("event: content_block_start")
            .expect("block start");
        let delta = body
            .find("event: content_block_delta")
            .expect("block delta");
        let block_stop = body.find("event: content_block_stop").expect("block stop");
        let message_stop = body.find("event: message_stop").expect("message stop");
        assert!(message_start < block_start);
        assert!(block_start < delta);
        assert!(delta < block_stop);
        assert!(block_stop < message_stop);
        assert!(body.contains("hello"));
    }

    #[tokio::test]
    async fn provider_outputs_only_lower_raw_reasoning_content() {
        let events = vec![
            Ok(CanonicalLlmEvent::ReasoningStart {
                id: "reasoning-0".to_string(),
            }),
            Ok(CanonicalLlmEvent::ReasoningSummaryPartAdded {
                id: "reasoning-0".to_string(),
                summary_index: 0,
            }),
            Ok(CanonicalLlmEvent::ReasoningSummaryDelta {
                id: "reasoning-0".to_string(),
                text: "summary-only".to_string(),
                summary_index: 0,
            }),
            Ok(CanonicalLlmEvent::ReasoningContentDelta {
                id: "reasoning-0".to_string(),
                text: "raw-reasoning".to_string(),
                content_index: 0,
            }),
            Ok(CanonicalLlmEvent::ReasoningEnd {
                id: "reasoning-0".to_string(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::Stop,
                usage: None,
                response_id: None,
            }),
        ];

        let collected = collect_provider_output(Box::pin(stream::iter(events.clone())))
            .await
            .expect("collect provider output");
        assert_eq!(collected.reasoning, "raw-reasoning");

        for format in [OutputFormat::OpenAi, OutputFormat::Anthropic] {
            let response = stream_provider_response(
                Box::pin(stream::iter(events.clone())),
                "reasoning-model",
                format,
            );
            let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
                .await
                .expect("collect reasoning SSE body");
            let body = String::from_utf8(body.to_vec()).expect("reasoning SSE UTF-8");
            assert!(body.contains("raw-reasoning"));
            assert!(!body.contains("summary-only"));
        }
    }

    #[tokio::test]
    async fn provider_metadata_events_do_not_leak_into_compat_outputs() {
        let events = vec![
            Ok(CanonicalLlmEvent::ServerModel {
                model: "internal-routed-model".to_string(),
            }),
            Ok(CanonicalLlmEvent::ModelVerification {
                verifications: vec![ModelVerification::TrustedAccessForCyber],
            }),
            Ok(CanonicalLlmEvent::TextDelta {
                id: "text-0".to_string(),
                text: "visible".to_string(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::Stop,
                usage: None,
                response_id: None,
            }),
        ];

        let collected = collect_provider_output(Box::pin(stream::iter(events.clone())))
            .await
            .expect("collect provider output");
        assert_eq!(collected.text, "visible");

        for format in [OutputFormat::OpenAi, OutputFormat::Anthropic] {
            let response = stream_provider_response(
                Box::pin(stream::iter(events.clone())),
                "requested-model",
                format,
            );
            let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
                .await
                .expect("collect metadata SSE body");
            let body = String::from_utf8(body.to_vec()).expect("metadata SSE UTF-8");
            assert!(body.contains("visible"));
            assert!(!body.contains("internal-routed-model"));
            assert!(!body.contains("trustedAccessForCyber"));
        }
    }

    #[tokio::test]
    async fn legacy_provider_output_rejects_custom_tool_calls() {
        let error = match collect_provider_output(Box::pin(stream::iter(vec![Ok(
            CanonicalLlmEvent::CustomToolCall {
                id: "custom-call-1".to_string(),
                name: "run_code".to_string(),
                input: "return 42;".to_string(),
                namespace: Some("codemode".to_string()),
                provider_metadata: Default::default(),
            },
        )])))
        .await
        {
            Ok(_) => panic!("legacy provider output must reject custom calls"),
            Err(error) => error,
        };
        assert_eq!(
            error.classification,
            Some(FailureClassification::InvalidRequest)
        );
        assert!(error.message.contains("cannot expose custom tool calls"));
    }

    #[tokio::test]
    async fn anthropic_tool_call_does_not_restart_after_input_end() {
        let events = vec![
            Ok(CanonicalLlmEvent::ToolInputStart {
                id: "call-1".to_string(),
                name: "lookup".to_string(),
            }),
            Ok(CanonicalLlmEvent::ToolInputDelta {
                id: "call-1".to_string(),
                name: "lookup".to_string(),
                text: "{}".to_string(),
            }),
            Ok(CanonicalLlmEvent::ToolInputEnd {
                id: "call-1".to_string(),
                name: "lookup".to_string(),
            }),
            Ok(CanonicalLlmEvent::ToolCall {
                id: "call-1".to_string(),
                name: "lookup".to_string(),
                input: json!({}),
                raw_arguments: None,
                provider_executed: None,
                provider_metadata: Default::default(),
            }),
            Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::ToolCall,
                usage: None,
                response_id: None,
            }),
        ];
        let response = stream_provider_response(
            Box::pin(stream::iter(events)),
            "claude-sonnet",
            OutputFormat::Anthropic,
        );
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .expect("collect tool SSE body");
        let body = String::from_utf8(body.to_vec()).expect("Anthropic SSE UTF-8");
        assert_eq!(body.matches("event: content_block_start").count(), 1);
        assert_eq!(body.matches("event: content_block_stop").count(), 1);
        assert!(body.contains("lookup"));
    }

    #[tokio::test]
    async fn provider_errors_are_encoded_as_terminal_sse_events() {
        let openai_response = stream_provider_response(
            Box::pin(stream::iter(vec![Ok(CanonicalLlmEvent::ProviderError {
                message: "rate limited".to_string(),
                classification: Some(FailureClassification::RateLimit),
                retryable: Some(true),
            })])),
            "gpt-5",
            OutputFormat::OpenAi,
        );
        let openai_body = axum::body::to_bytes(openai_response.into_body(), 64 * 1024)
            .await
            .expect("collect OpenAI error stream");
        let openai_body = String::from_utf8(openai_body.to_vec()).expect("OpenAI SSE UTF-8");
        assert!(openai_body.contains("rate_limit_error"));
        assert!(openai_body.contains("rate limited"));
        assert!(openai_body.ends_with("data: [DONE]\n\n"));

        let anthropic_response = stream_provider_response(
            Box::pin(stream::iter(vec![Err(CurrentProviderError {
                message: "connection reset".to_string(),
                status: None,
                classification: Some(FailureClassification::Transport),
                retryable: true,
                retry_after: None,
            })])),
            "claude-sonnet",
            OutputFormat::Anthropic,
        );
        let anthropic_body = axum::body::to_bytes(anthropic_response.into_body(), 64 * 1024)
            .await
            .expect("collect Anthropic error stream");
        let anthropic_body =
            String::from_utf8(anthropic_body.to_vec()).expect("Anthropic SSE UTF-8");
        assert!(anthropic_body.contains("event: error"));
        assert!(anthropic_body.contains("connection reset"));
        assert!(!anthropic_body.contains("event: message_stop"));
    }

    struct DropSignal {
        dropped: Arc<AtomicBool>,
        sender: Option<oneshot::Sender<()>>,
    }

    impl Drop for DropSignal {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::Release);
            if let Some(sender) = self.sender.take() {
                let _ = sender.send(());
            }
        }
    }

    #[tokio::test]
    async fn dropping_response_body_drops_upstream_provider_stream() {
        let dropped = Arc::new(AtomicBool::new(false));
        let (drop_sender, drop_receiver) = oneshot::channel();
        let guard = DropSignal {
            dropped: dropped.clone(),
            sender: Some(drop_sender),
        };
        let provider_stream: CurrentProviderStream = Box::pin(async_stream::stream! {
            let _guard = guard;
            yield Ok(CanonicalLlmEvent::TextDelta {
                id: "text-0".to_string(),
                text: "visible".to_string(),
            });
            futures::future::pending::<()>().await;
        });
        let response = stream_provider_response(provider_stream, "gpt-5", OutputFormat::OpenAi);
        let mut body = response.into_body().into_data_stream();

        assert!(next_body_chunk(&mut body).await.contains("assistant"));
        assert!(next_body_chunk(&mut body).await.contains("visible"));
        assert!(!dropped.load(Ordering::Acquire));
        drop(body);

        timeout(Duration::from_secs(1), drop_receiver)
            .await
            .expect("provider stream was not dropped")
            .expect("drop signal sender closed");
        assert!(dropped.load(Ordering::Acquire));
    }
}
