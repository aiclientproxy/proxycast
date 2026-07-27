//! Public SSE lowering for current provider events.

use super::{finish_reason_anthropic, finish_reason_openai, usage_value, OutputFormat};
use axum::{
    body::Body,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures::StreamExt;
use model_provider::current_client::{
    CanonicalLlmEvent, CurrentProviderStream, FailureClassification, Usage,
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;

/// 将 canonical provider event 逐事件 lower 到公开 OpenAI SSE。
///
/// 这个 adapter 只负责 HTTP proxy 的输出 wire；provider 请求 lowering 和上游 SSE
/// reducer 仍由 `model-provider` 持有。它不缓存正文，也不在 terminal 前等待完整响应。
struct OpenAiStreamEncoder {
    id: String,
    created: u64,
    model: String,
    usage: Usage,
    finish_reason: model_provider::current_client::FinishReason,
    tool_indices: BTreeMap<String, usize>,
    next_tool_index: usize,
    terminal: bool,
}

impl OpenAiStreamEncoder {
    fn new(model: &str) -> Self {
        Self {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            created: chrono::Utc::now().timestamp().max(0) as u64,
            model: model.to_string(),
            usage: Usage::default(),
            finish_reason: model_provider::current_client::FinishReason::Stop,
            tool_indices: BTreeMap::new(),
            next_tool_index: 0,
            terminal: false,
        }
    }

    fn start(&self) -> String {
        self.frame(json!({
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
        }))
    }

    fn frame(&self, mut payload: Value) -> String {
        let object = payload
            .as_object_mut()
            .expect("stream payload must be an object");
        object.insert("id".to_string(), Value::String(self.id.clone()));
        object.insert(
            "object".to_string(),
            Value::String("chat.completion.chunk".to_string()),
        );
        object.insert("created".to_string(), json!(self.created));
        object.insert("model".to_string(), Value::String(self.model.clone()));
        format!("data: {payload}\n\n")
    }

    fn tool_index(&mut self, id: &str) -> usize {
        if let Some(index) = self.tool_indices.get(id) {
            return *index;
        }
        let index = self.next_tool_index;
        self.next_tool_index = self.next_tool_index.saturating_add(1);
        self.tool_indices.insert(id.to_string(), index);
        index
    }

    fn encode(&mut self, event: &CanonicalLlmEvent) -> Vec<String> {
        match event {
            CanonicalLlmEvent::TextDelta { text, .. } if !text.is_empty() => vec![self.frame(
                json!({"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]}),
            )],
            CanonicalLlmEvent::ReasoningContentDelta { text, .. } if !text.is_empty() => vec![self.frame(
                json!({"choices": [{"index": 0, "delta": {"reasoning_content": text}, "finish_reason": null}]}),
            )],
            CanonicalLlmEvent::ToolInputStart { id, name } => {
                let index = self.tool_index(id);
                vec![self.frame(json!({
                    "choices": [{"index": 0, "delta": {"tool_calls": [{"index": index, "id": id, "type": "function", "function": {"name": name, "arguments": ""}}]}, "finish_reason": null}]
                }))]
            }
            CanonicalLlmEvent::ToolInputDelta { id, name, text } => {
                let index = self.tool_index(id);
                vec![self.frame(json!({
                    "choices": [{"index": 0, "delta": {"tool_calls": [{"index": index, "id": id, "type": "function", "function": {"name": name, "arguments": text}}]}, "finish_reason": null}]
                }))]
            }
            CanonicalLlmEvent::ToolCall {
                id, name, input, ..
            } if !self.tool_indices.contains_key(id) => {
                let index = self.tool_index(id);
                vec![self.frame(json!({
                    "choices": [{"index": 0, "delta": {"tool_calls": [{"index": index, "id": id, "type": "function", "function": {"name": name, "arguments": input.to_string()}}]}, "finish_reason": null}]
                }))]
            }
            CanonicalLlmEvent::Usage { usage } => {
                self.usage = usage.clone();
                Vec::new()
            }
            CanonicalLlmEvent::StepFinish { reason, usage, .. } => {
                self.finish_reason = *reason;
                if let Some(usage) = usage {
                    self.usage = usage.clone();
                }
                Vec::new()
            }
            CanonicalLlmEvent::Finish { reason, usage, .. } => {
                self.terminal = true;
                self.finish_reason = *reason;
                if let Some(usage) = usage {
                    self.usage = usage.clone();
                }
                let (input_tokens, output_tokens) = usage_value(&self.usage);
                vec![
                    self.frame(json!({
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason_openai(self.finish_reason)}]
                    })),
                    self.frame(json!({
                        "choices": [],
                        "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens, "total_tokens": input_tokens.saturating_add(output_tokens)}
                    })),
                    "data: [DONE]\n\n".to_string(),
                ]
            }
            CanonicalLlmEvent::TextStart { .. }
            | CanonicalLlmEvent::TextEnd { .. }
            | CanonicalLlmEvent::ReasoningStart { .. }
            | CanonicalLlmEvent::ReasoningSummaryDelta { .. }
            | CanonicalLlmEvent::ReasoningSummaryPartAdded { .. }
            | CanonicalLlmEvent::ReasoningEnd { .. }
            | CanonicalLlmEvent::ToolInputEnd { .. }
            | CanonicalLlmEvent::ToolResult { .. }
            | CanonicalLlmEvent::ToolError { .. }
            | CanonicalLlmEvent::ServerModel { .. }
            | CanonicalLlmEvent::ModelReroute { .. }
            | CanonicalLlmEvent::ModelVerification { .. }
            | CanonicalLlmEvent::StepStart { .. }
            | CanonicalLlmEvent::ToolCall { .. }
            | CanonicalLlmEvent::TextDelta { .. }
            | CanonicalLlmEvent::ReasoningContentDelta { .. }
            | CanonicalLlmEvent::ProviderError { .. } => Vec::new(),
        }
    }

    fn error(
        &mut self,
        message: &str,
        classification: Option<FailureClassification>,
    ) -> Vec<String> {
        self.terminal = true;
        vec![
            self.frame(json!({
                "choices": [],
                "error": {"message": message, "type": provider_error_type(classification)}
            })),
            "data: [DONE]\n\n".to_string(),
        ]
    }

    fn truncated(&mut self) -> Vec<String> {
        self.error(
            "provider stream ended before its terminal event",
            Some(FailureClassification::Transport),
        )
    }
}

struct AnthropicStreamEncoder {
    id: String,
    model: String,
    usage: Usage,
    finish_reason: model_provider::current_client::FinishReason,
    blocks: BTreeMap<String, (u32, &'static str)>,
    completed_tools: BTreeSet<String>,
    next_block_index: u32,
    terminal: bool,
}

impl AnthropicStreamEncoder {
    fn new(model: &str) -> Self {
        Self {
            id: format!("msg_{}", uuid::Uuid::new_v4()),
            model: model.to_string(),
            usage: Usage::default(),
            finish_reason: model_provider::current_client::FinishReason::Stop,
            blocks: BTreeMap::new(),
            completed_tools: BTreeSet::new(),
            next_block_index: 0,
            terminal: false,
        }
    }

    fn frame(event: &str, payload: Value) -> String {
        format!("event: {event}\ndata: {payload}\n\n")
    }

    fn start(&self) -> String {
        Self::frame(
            "message_start",
            json!({
                "type": "message_start",
                "message": {"id": self.id, "type": "message", "role": "assistant", "model": self.model, "content": [], "usage": {"input_tokens": 0, "output_tokens": 0}}
            }),
        )
    }

    fn ensure_block(&mut self, id: &str, kind: &'static str, name: Option<&str>) -> Vec<String> {
        if self.blocks.contains_key(id) {
            return Vec::new();
        }
        let index = self.next_block_index;
        self.next_block_index = self.next_block_index.saturating_add(1);
        self.blocks.insert(id.to_string(), (index, kind));
        let content_block = match kind {
            "tool_use" => {
                json!({"type": "tool_use", "id": id, "name": name.unwrap_or("tool"), "input": {}})
            }
            "thinking" => json!({"type": "thinking", "thinking": ""}),
            _ => json!({"type": "text", "text": ""}),
        };
        vec![Self::frame(
            "content_block_start",
            json!({"type": "content_block_start", "index": index, "content_block": content_block}),
        )]
    }

    fn block_index(&self, id: &str) -> Option<u32> {
        self.blocks.get(id).map(|(index, _)| *index)
    }

    fn encode(&mut self, event: &CanonicalLlmEvent) -> Vec<String> {
        match event {
            CanonicalLlmEvent::TextStart { id } => self.ensure_block(id, "text", None),
            CanonicalLlmEvent::TextDelta { id, text } if !text.is_empty() => {
                let mut frames = self.ensure_block(id, "text", None);
                let index = self.block_index(id).expect("text block inserted");
                frames.push(Self::frame(
                    "content_block_delta",
                    json!({"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}}),
                ));
                frames
            }
            CanonicalLlmEvent::ReasoningStart { id } => self.ensure_block(id, "thinking", None),
            CanonicalLlmEvent::ReasoningContentDelta { id, text, .. } if !text.is_empty() => {
                let mut frames = self.ensure_block(id, "thinking", None);
                let index = self.block_index(id).expect("thinking block inserted");
                frames.push(Self::frame(
                    "content_block_delta",
                    json!({"type": "content_block_delta", "index": index, "delta": {"type": "thinking_delta", "thinking": text}}),
                ));
                frames
            }
            CanonicalLlmEvent::ToolInputStart { id, name } => {
                self.ensure_block(id, "tool_use", Some(name))
            }
            CanonicalLlmEvent::ToolInputDelta { id, name, text } => {
                let mut frames = self.ensure_block(id, "tool_use", Some(name));
                let index = self.block_index(id).expect("tool block inserted");
                frames.push(Self::frame(
                    "content_block_delta",
                    json!({"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": text}}),
                ));
                frames
            }
            CanonicalLlmEvent::ToolCall {
                id, name, input, ..
            } if self.block_index(id).is_none() && !self.completed_tools.contains(id) => {
                let mut frames = self.ensure_block(id, "tool_use", Some(name));
                let index = self.block_index(id).expect("tool block inserted");
                frames.push(Self::frame(
                    "content_block_delta",
                    json!({"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": input.to_string()}}),
                ));
                frames.push(Self::frame(
                    "content_block_stop",
                    json!({"type": "content_block_stop", "index": index}),
                ));
                self.blocks.remove(id);
                frames
            }
            CanonicalLlmEvent::ToolCall { .. } => Vec::new(),
            CanonicalLlmEvent::TextEnd { id }
            | CanonicalLlmEvent::ReasoningEnd { id }
            | CanonicalLlmEvent::ToolInputEnd { id, .. } => self.close_block(id),
            CanonicalLlmEvent::Usage { usage } => {
                self.usage = usage.clone();
                Vec::new()
            }
            CanonicalLlmEvent::StepFinish { reason, usage, .. } => {
                self.finish_reason = *reason;
                if let Some(usage) = usage {
                    self.usage = usage.clone();
                }
                Vec::new()
            }
            CanonicalLlmEvent::Finish { reason, usage, .. } => {
                self.finish_reason = *reason;
                if let Some(usage) = usage {
                    self.usage = usage.clone();
                }
                self.finish()
            }
            CanonicalLlmEvent::ToolResult { .. }
            | CanonicalLlmEvent::ToolError { .. }
            | CanonicalLlmEvent::ServerModel { .. }
            | CanonicalLlmEvent::ModelReroute { .. }
            | CanonicalLlmEvent::ModelVerification { .. }
            | CanonicalLlmEvent::StepStart { .. }
            | CanonicalLlmEvent::TextDelta { .. }
            | CanonicalLlmEvent::ReasoningSummaryDelta { .. }
            | CanonicalLlmEvent::ReasoningSummaryPartAdded { .. }
            | CanonicalLlmEvent::ReasoningContentDelta { .. }
            | CanonicalLlmEvent::ProviderError { .. } => Vec::new(),
        }
    }

    fn close_block(&mut self, id: &str) -> Vec<String> {
        let Some((index, kind)) = self.blocks.remove(id) else {
            return Vec::new();
        };
        if kind == "tool_use" {
            self.completed_tools.insert(id.to_string());
        }
        vec![Self::frame(
            "content_block_stop",
            json!({"type": "content_block_stop", "index": index}),
        )]
    }

    fn finish(&mut self) -> Vec<String> {
        self.terminal = true;
        let mut frames = Vec::new();
        let open_ids = self.blocks.keys().cloned().collect::<Vec<_>>();
        for id in open_ids {
            frames.extend(self.close_block(&id));
        }
        let (input_tokens, output_tokens) = usage_value(&self.usage);
        frames.push(Self::frame(
            "message_delta",
            json!({"type": "message_delta", "delta": {"stop_reason": finish_reason_anthropic(self.finish_reason)}, "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}}),
        ));
        frames.push(Self::frame("message_stop", json!({"type": "message_stop"})));
        frames
    }

    fn error(
        &mut self,
        message: &str,
        classification: Option<FailureClassification>,
    ) -> Vec<String> {
        self.terminal = true;
        vec![Self::frame(
            "error",
            json!({"type": "error", "error": {"type": provider_error_type(classification), "message": message}}),
        )]
    }

    fn truncated(&mut self) -> Vec<String> {
        self.error(
            "provider stream ended before its terminal event",
            Some(FailureClassification::Transport),
        )
    }
}

fn provider_error_type(classification: Option<FailureClassification>) -> &'static str {
    match classification {
        Some(FailureClassification::Authentication) => "authentication_error",
        Some(FailureClassification::RateLimit) => "rate_limit_error",
        Some(FailureClassification::InvalidRequest) => "invalid_request_error",
        Some(FailureClassification::ContentPolicy) => "content_policy_error",
        _ => "provider_error",
    }
}

/// 把 current client 的真实事件流暴露为公开 API 的 SSE Body。
pub(super) fn stream_provider_response(
    stream: CurrentProviderStream,
    model: &str,
    format: OutputFormat,
) -> Response {
    let model = model.to_string();
    let body_stream = async_stream::stream! {
        match format {
            OutputFormat::OpenAi => {
                let mut encoder = OpenAiStreamEncoder::new(&model);
                yield Ok::<Bytes, Infallible>(Bytes::from(encoder.start()));
                futures::pin_mut!(stream);
                while let Some(event) = stream.next().await {
                    match event {
                        Ok(CanonicalLlmEvent::ProviderError {
                            message,
                            classification,
                            ..
                        }) => {
                            for frame in encoder.error(&message, classification) {
                                yield Ok(Bytes::from(frame));
                            }
                            return;
                        }
                        Ok(event) => {
                            for frame in encoder.encode(&event) {
                                yield Ok(Bytes::from(frame));
                            }
                            if encoder.terminal { return; }
                        }
                        Err(error) => {
                            for frame in encoder.error(&error.message, error.classification) {
                                yield Ok(Bytes::from(frame));
                            }
                            return;
                        }
                    }
                }
                for frame in encoder.truncated() {
                    yield Ok(Bytes::from(frame));
                }
            }
            OutputFormat::Anthropic => {
                let mut encoder = AnthropicStreamEncoder::new(&model);
                yield Ok::<Bytes, Infallible>(Bytes::from(encoder.start()));
                futures::pin_mut!(stream);
                while let Some(event) = stream.next().await {
                    match event {
                        Ok(CanonicalLlmEvent::ProviderError {
                            message,
                            classification,
                            ..
                        }) => {
                            for frame in encoder.error(&message, classification) {
                                yield Ok(Bytes::from(frame));
                            }
                            return;
                        }
                        Ok(event) => {
                            for frame in encoder.encode(&event) {
                                yield Ok(Bytes::from(frame));
                            }
                            if encoder.terminal { return; }
                        }
                        Err(error) => {
                            for frame in encoder.error(&error.message, error.classification) {
                                yield Ok(Bytes::from(frame));
                            }
                            return;
                        }
                    }
                }
                for frame in encoder.truncated() {
                    yield Ok(Bytes::from(frame));
                }
            }
        }
    };
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(Body::from_stream(body_stream))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}
