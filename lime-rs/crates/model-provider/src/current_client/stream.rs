use super::{CurrentProviderCustomToolCall, CurrentProviderError, CurrentProviderToolCall};
use agent_protocol::{anthropic, openai};
use async_stream::try_stream;
use futures::{Stream, StreamExt};
use reqwest::Response;
use runtime_core::{
    CanonicalLlmEvent as LlmEvent, FailureClassification, FinishReason, ModelVerification, Usage,
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Duration;

mod hosted_tools;

use hosted_tools::{hosted_tool_events, HostedToolState};

pub(super) const DEFAULT_STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(300);
const OPENAI_FINISH_TRAILER_TIMEOUT: Duration = Duration::from_secs(1);

#[derive(Debug, Default)]
struct ToolCallAccumulator {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
    emitted: bool,
    started: bool,
}

impl ToolCallAccumulator {
    fn call_id(&self, fallback: &str) -> String {
        self.id.clone().unwrap_or_else(|| fallback.to_string())
    }

    fn into_call(&mut self, fallback: &str) -> Option<CurrentProviderToolCall> {
        if self.emitted {
            return None;
        }
        let call = CurrentProviderToolCall::from_raw(
            self.call_id(fallback),
            self.name.clone().unwrap_or_default(),
            self.arguments.clone(),
        );
        self.emitted = true;
        Some(call)
    }

    fn begin_input_if_ready(&mut self, fallback: &str) -> Option<(String, String, String)> {
        if self.started || self.arguments.is_empty() {
            return None;
        }
        let name = self.name.as_deref()?.trim();
        if name.is_empty() {
            return None;
        }
        self.started = true;
        Some((
            self.call_id(fallback),
            name.to_string(),
            self.arguments.clone(),
        ))
    }
}

#[derive(Debug, Default)]
struct CustomToolCallAccumulator {
    id: Option<String>,
    name: Option<String>,
    namespace: Option<String>,
    input: String,
    emitted: bool,
    started: bool,
}

impl CustomToolCallAccumulator {
    fn call_id(&self, fallback: &str) -> String {
        self.id.clone().unwrap_or_else(|| fallback.to_string())
    }

    fn into_call(&mut self, fallback: &str) -> Option<CurrentProviderCustomToolCall> {
        if self.emitted {
            return None;
        }
        let call = CurrentProviderCustomToolCall {
            id: self.call_id(fallback),
            name: self.name.clone().unwrap_or_default(),
            input: self.input.clone(),
            namespace: self.namespace.clone(),
            provider_metadata: Default::default(),
        };
        self.emitted = true;
        Some(call)
    }

    fn begin_input_if_ready(&mut self, fallback: &str) -> Option<(String, String, String)> {
        if self.started || self.input.is_empty() {
            return None;
        }
        let name = self.name.as_deref()?.trim();
        if name.is_empty() {
            return None;
        }
        self.started = true;
        Some((self.call_id(fallback), name.to_string(), self.input.clone()))
    }
}

#[derive(Debug, Default)]
struct OpenAiStreamState {
    response_id: Option<String>,
    calls: BTreeMap<u32, ToolCallAccumulator>,
    emitted_tool_call: bool,
    usage: Option<Usage>,
    text_ids: HashSet<String>,
    reasoning_ids: HashSet<String>,
    finish_reason: Option<FinishReason>,
}

pub(super) fn openai_chat_sse(
    response: Response,
) -> impl Stream<Item = Result<LlmEvent, CurrentProviderError>> + Send {
    try_stream! {
        let mut state = OpenAiStreamState::default();
        let mut frames = Box::pin(sse_frames(response));
        while let Some(frame) = frames.next().await {
            let frame = frame?;
            if frame.data.trim() == "[DONE]" {
                drop(frames);
                for event in finish_openai_stream(&mut state)? {
                    yield event;
                }
                return;
            }
            let chunk = serde_json::from_str::<openai::ChatCompletionChunk>(&frame.data)
                .map_err(|error| CurrentProviderError::new(format!("解析 OpenAI SSE chunk 失败: {error}")))?;
            if let Some(response_id) = chunk.id.filter(|value| !value.trim().is_empty()) {
                state.response_id = Some(response_id);
            }
            if let Some(usage) = chunk.usage {
                let usage = openai_usage(usage);
                yield LlmEvent::Usage { usage: usage.clone() };
                state.usage = Some(usage);
            }
            for choice in chunk.choices {
                let text_id = format!("text-{}", choice.index);
                if let Some(text) = choice.delta.content.filter(|value| !value.is_empty()) {
                    if state.text_ids.insert(text_id.clone()) {
                        yield LlmEvent::TextStart { id: text_id.clone() };
                    }
                    yield LlmEvent::TextDelta { id: text_id, text };
                }
                let reasoning_id = format!("reasoning-{}", choice.index);
                if let Some(reasoning) = choice.delta.reasoning_content.filter(|value| !value.is_empty()) {
                    if state.reasoning_ids.insert(reasoning_id.clone()) {
                        yield LlmEvent::ReasoningStart { id: reasoning_id.clone() };
                    }
                    yield LlmEvent::ReasoningContentDelta {
                        id: reasoning_id,
                        text: reasoning,
                        content_index: 0,
                    };
                }
                for delta in choice.delta.tool_calls.unwrap_or_default() {
                    let index = delta.index;
                    let id = delta.id.filter(|value| !value.trim().is_empty());
                    let name = delta
                        .function
                        .name
                        .filter(|value| !value.trim().is_empty());
                    let arguments = delta
                        .function
                        .arguments
                        .filter(|value| !value.is_empty());
                    if id.is_none() && name.is_none() && arguments.is_none() {
                        continue;
                    }
                    let call = state.calls.entry(index).or_default();
                    if id.is_some() { call.id = id; }
                    if name.is_some() { call.name = name; }
                    if let Some(arguments) = arguments.as_ref() {
                        call.arguments.push_str(&arguments);
                    }
                    if let Some((call_id, name, buffered_arguments)) =
                        call.begin_input_if_ready(&format!("call_{index}"))
                    {
                        yield LlmEvent::ToolInputStart { id: call_id.clone(), name: name.clone() };
                        yield LlmEvent::ToolInputDelta {
                            id: call_id,
                            name,
                            text: buffered_arguments,
                        };
                    } else if call.started {
                        if let (Some(name), Some(arguments)) = (call.name.clone(), arguments) {
                            yield LlmEvent::ToolInputDelta {
                                id: call.call_id(&format!("call_{index}")),
                                name,
                                text: arguments,
                            };
                        }
                    }
                }
                if choice.finish_reason.as_deref() == Some("tool_calls") {
                    for event in take_openai_calls(&mut state)? {
                        yield event;
                    }
                }
                if let Some(reason) = choice.finish_reason.as_deref() {
                    state.finish_reason = Some(openai_finish_reason(reason));
                }
            }
            if state.finish_reason.is_some() {
                if let Ok(Some(Ok(frame))) = tokio::time::timeout(
                    OPENAI_FINISH_TRAILER_TIMEOUT,
                    frames.next(),
                )
                .await
                {
                    absorb_openai_usage_frame(&mut state, &frame.data);
                }
                drop(frames);
                for event in finish_openai_stream(&mut state)? {
                    yield event;
                }
                return;
            }
        }
        for id in state.text_ids.drain() { yield LlmEvent::TextEnd { id }; }
        for id in state.reasoning_ids.drain() { yield LlmEvent::ReasoningEnd { id }; }
        yield truncated_stream_error("OpenAI Chat Completions");
    }
}

fn openai_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" => FinishReason::Stop,
        "tool_calls" | "function_call" => FinishReason::ToolCall,
        "length" => FinishReason::Length,
        "content_filter" => FinishReason::ContentFilter,
        _ => FinishReason::Unknown,
    }
}

fn take_openai_calls(state: &mut OpenAiStreamState) -> Result<Vec<LlmEvent>, CurrentProviderError> {
    let mut events = Vec::new();
    for (index, call) in &mut state.calls {
        if let Some((id, name, arguments)) = call.begin_input_if_ready(&format!("call_{index}")) {
            events.push(LlmEvent::ToolInputStart {
                id: id.clone(),
                name: name.clone(),
            });
            events.push(LlmEvent::ToolInputDelta {
                id,
                name,
                text: arguments,
            });
        }
        let Some(tool_call) = call.into_call(&format!("call_{index}")) else {
            continue;
        };
        state.emitted_tool_call = true;
        events.push(LlmEvent::ToolInputEnd {
            id: tool_call.id.clone(),
            name: tool_call.name.clone(),
        });
        events.push(LlmEvent::ToolCall {
            id: tool_call.id,
            name: tool_call.name,
            input: tool_call.arguments,
            raw_arguments: Some(tool_call.raw_arguments),
            provider_executed: None,
            provider_metadata: Default::default(),
        });
    }
    Ok(events)
}

fn finish_openai_stream(
    state: &mut OpenAiStreamState,
) -> Result<Vec<LlmEvent>, CurrentProviderError> {
    let mut events = Vec::new();
    events.extend(state.text_ids.drain().map(|id| LlmEvent::TextEnd { id }));
    events.extend(
        state
            .reasoning_ids
            .drain()
            .map(|id| LlmEvent::ReasoningEnd { id }),
    );
    events.extend(take_openai_calls(state)?);
    events.push(LlmEvent::Finish {
        reason: state.finish_reason.unwrap_or_else(|| {
            if state.emitted_tool_call {
                FinishReason::ToolCall
            } else {
                FinishReason::Stop
            }
        }),
        usage: state.usage.take(),
        response_id: state.response_id.clone(),
    });
    Ok(events)
}

fn absorb_openai_usage_frame(state: &mut OpenAiStreamState, data: &str) {
    if data.trim() == "[DONE]" {
        return;
    }

    let Ok(chunk) = serde_json::from_str::<openai::ChatCompletionChunk>(data) else {
        return;
    };
    if let Some(usage) = chunk.usage {
        state.usage = Some(openai_usage(usage));
    }
}

fn openai_usage(usage: openai::StreamUsage) -> Usage {
    Usage {
        input_tokens: Some(usage.prompt_tokens as u64),
        output_tokens: Some(usage.completion_tokens as u64),
        cache_read_input_tokens: usage
            .prompt_tokens_details
            .and_then(|details| details.cached_tokens)
            .map(u64::from),
        total_tokens: Some(usage.total_tokens as u64),
        ..Usage::default()
    }
}

#[derive(Debug, Default)]
struct ResponsesStreamState {
    response_id: Option<String>,
    calls: HashMap<String, ToolCallAccumulator>,
    custom_calls: HashMap<String, CustomToolCallAccumulator>,
    emitted_calls: HashSet<String>,
    hosted_tools: HostedToolState,
    emitted_tool_call: bool,
    usage: Option<Usage>,
    text_ids: HashSet<String>,
    reasoning_ids: HashSet<String>,
    active_reasoning_id: Option<String>,
    server_model: Option<String>,
    emitted_verifications: HashSet<ModelVerification>,
    allow_openai_response_metadata: bool,
}

pub(super) struct ResponsesEventBatch {
    pub events: Vec<LlmEvent>,
    pub terminal: bool,
}

#[derive(Debug, Default)]
pub(super) struct ResponsesEventReducer {
    state: ResponsesStreamState,
}

impl ResponsesEventReducer {
    pub(super) fn new(server_model: Option<String>, allow_openai_response_metadata: bool) -> Self {
        Self {
            state: ResponsesStreamState {
                server_model,
                allow_openai_response_metadata,
                ..ResponsesStreamState::default()
            },
        }
    }

    pub fn push(&mut self, payload: &Value) -> Result<ResponsesEventBatch, CurrentProviderError> {
        let mut events = Vec::new();
        let mut terminal = false;
        let event_type = payload
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if let Some(model) = response_server_model(payload) {
            if self.state.server_model.as_deref() != Some(model.as_str()) {
                self.state.server_model = Some(model.clone());
                events.push(LlmEvent::ServerModel { model });
            }
        }
        match event_type {
            "response.metadata" if self.state.allow_openai_response_metadata => {
                let verifications = response_model_verifications(payload)
                    .into_iter()
                    .filter(|verification| self.state.emitted_verifications.insert(*verification))
                    .collect::<Vec<_>>();
                if !verifications.is_empty() {
                    events.push(LlmEvent::ModelVerification { verifications });
                }
                if let Some(metadata) = response_turn_moderation_metadata(payload) {
                    events.push(LlmEvent::TurnModerationMetadata { metadata });
                }
            }
            "response.output_text.delta" => {
                let id = response_block_id(payload, "text");
                if let Some(delta) = payload
                    .get("delta")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                {
                    if self.state.text_ids.insert(id.clone()) {
                        events.push(LlmEvent::TextStart { id: id.clone() });
                    }
                    events.push(LlmEvent::TextDelta {
                        id,
                        text: delta.to_string(),
                    });
                }
            }
            "response.reasoning_summary_text.delta" => {
                let id = response_reasoning_id(payload, &self.state);
                if let (Some(delta), Some(summary_index)) = (
                    payload
                        .get("delta")
                        .and_then(Value::as_str)
                        .filter(|value| !value.is_empty()),
                    payload.get("summary_index").and_then(Value::as_i64),
                ) {
                    if self.state.reasoning_ids.insert(id.clone()) {
                        events.push(LlmEvent::ReasoningStart { id: id.clone() });
                    }
                    events.push(LlmEvent::ReasoningSummaryDelta {
                        id,
                        text: delta.to_string(),
                        summary_index,
                    });
                }
            }
            "response.reasoning_summary_part.added" => {
                let id = response_reasoning_id(payload, &self.state);
                if let Some(summary_index) = payload.get("summary_index").and_then(Value::as_i64) {
                    if self.state.reasoning_ids.insert(id.clone()) {
                        events.push(LlmEvent::ReasoningStart { id: id.clone() });
                    }
                    events.push(LlmEvent::ReasoningSummaryPartAdded { id, summary_index });
                }
            }
            "response.reasoning_text.delta" => {
                let id = response_reasoning_id(payload, &self.state);
                if let (Some(delta), Some(content_index)) = (
                    payload
                        .get("delta")
                        .and_then(Value::as_str)
                        .filter(|value| !value.is_empty()),
                    payload.get("content_index").and_then(Value::as_i64),
                ) {
                    if self.state.reasoning_ids.insert(id.clone()) {
                        events.push(LlmEvent::ReasoningStart { id: id.clone() });
                    }
                    events.push(LlmEvent::ReasoningContentDelta {
                        id,
                        text: delta.to_string(),
                        content_index,
                    });
                }
            }
            "response.output_item.added" => {
                if let Some(item) = payload.get("item") {
                    observe_active_responses_reasoning_item(item, &mut self.state);
                    absorb_responses_call(item, &mut self.state);
                    absorb_responses_custom_call(item, &mut self.state);
                    events.extend(hosted_tool_events(
                        item,
                        &mut self.state.hosted_tools,
                        false,
                    )?);
                }
            }
            "response.function_call_arguments.delta" => {
                let key = response_call_key(payload);
                let call = self.state.calls.entry(key.clone()).or_default();
                if let Some(call_id) = payload.get("call_id").and_then(Value::as_str) {
                    call.id = Some(call_id.to_string());
                }
                if let Some(name) = payload.get("name").and_then(Value::as_str) {
                    call.name = Some(name.to_string());
                }
                if let Some(delta) = payload.get("delta").and_then(Value::as_str) {
                    call.arguments.push_str(delta);
                    if let Some((id, name, arguments)) = call.begin_input_if_ready(&key) {
                        events.push(LlmEvent::ToolInputStart {
                            id: id.clone(),
                            name: name.clone(),
                        });
                        events.push(LlmEvent::ToolInputDelta {
                            id,
                            name,
                            text: arguments,
                        });
                    } else if call.started {
                        if let Some(name) = call.name.clone() {
                            events.push(LlmEvent::ToolInputDelta {
                                id: call.call_id(&key),
                                name,
                                text: delta.to_string(),
                            });
                        }
                    }
                }
            }
            "response.function_call_arguments.done" => {
                let key = response_call_key(payload);
                let call = self.state.calls.entry(key.clone()).or_default();
                if let Some(call_id) = payload.get("call_id").and_then(Value::as_str) {
                    call.id = Some(call_id.to_string());
                }
                if let Some(name) = payload.get("name").and_then(Value::as_str) {
                    call.name = Some(name.to_string());
                }
                if let Some(arguments) = payload.get("arguments").and_then(Value::as_str) {
                    call.arguments = arguments.to_string();
                }
                events.extend(take_responses_call(&mut self.state, &key)?);
            }
            "response.custom_tool_call_input.delta" => {
                let key = response_call_key(payload);
                let call = self.state.custom_calls.entry(key.clone()).or_default();
                if let Some(call_id) = payload.get("call_id").and_then(Value::as_str) {
                    call.id = Some(call_id.to_string());
                }
                if let Some(name) = payload.get("name").and_then(Value::as_str) {
                    call.name = Some(name.to_string());
                }
                if let Some(namespace) = payload.get("namespace").and_then(Value::as_str) {
                    call.namespace = Some(namespace.to_string());
                }
                if let Some(delta) = payload.get("delta").and_then(Value::as_str) {
                    call.input.push_str(delta);
                    if let Some((id, name, input)) = call.begin_input_if_ready(&key) {
                        events.push(LlmEvent::ToolInputStart {
                            id: id.clone(),
                            name: name.clone(),
                        });
                        events.push(LlmEvent::ToolInputDelta {
                            id,
                            name,
                            text: input,
                        });
                    } else if call.started {
                        if let Some(name) = call.name.clone() {
                            events.push(LlmEvent::ToolInputDelta {
                                id: call.call_id(&key),
                                name,
                                text: delta.to_string(),
                            });
                        }
                    }
                }
            }
            "response.custom_tool_call_input.done" => {
                let key = response_call_key(payload);
                let call = self.state.custom_calls.entry(key.clone()).or_default();
                if let Some(call_id) = payload.get("call_id").and_then(Value::as_str) {
                    call.id = Some(call_id.to_string());
                }
                if let Some(name) = payload.get("name").and_then(Value::as_str) {
                    call.name = Some(name.to_string());
                }
                if let Some(input) = payload.get("input").and_then(Value::as_str) {
                    call.input = input.to_string();
                }
                events.extend(take_responses_custom_call(&mut self.state, &key)?);
            }
            "response.output_item.done" => {
                if let Some(item) = payload.get("item") {
                    clear_active_responses_reasoning_item(item, &mut self.state);
                    absorb_responses_call(item, &mut self.state);
                    absorb_responses_custom_call(item, &mut self.state);
                    events.extend(hosted_tool_events(
                        item,
                        &mut self.state.hosted_tools,
                        true,
                    )?);
                    if item.get("type").and_then(Value::as_str) == Some("function_call") {
                        let key = response_call_key(item);
                        events.extend(take_responses_call(&mut self.state, &key)?);
                    }
                    if item.get("type").and_then(Value::as_str) == Some("custom_tool_call") {
                        let key = response_call_key(item);
                        events.extend(take_responses_custom_call(&mut self.state, &key)?);
                    }
                }
            }
            "response.completed" => {
                let response = payload.get("response").unwrap_or(payload);
                self.state.response_id = response
                    .get("id")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                if let Some(usage) = response.get("usage") {
                    let usage = responses_usage(usage);
                    events.push(LlmEvent::Usage {
                        usage: usage.clone(),
                    });
                    self.state.usage = Some(usage);
                }
                for item in response
                    .get("output")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    absorb_responses_call(item, &mut self.state);
                    absorb_responses_custom_call(item, &mut self.state);
                    events.extend(hosted_tool_events(
                        item,
                        &mut self.state.hosted_tools,
                        true,
                    )?);
                }
                events.extend(take_responses_calls(&mut self.state)?);
                events.extend(take_responses_custom_calls(&mut self.state)?);
                events.extend(
                    self.state
                        .text_ids
                        .drain()
                        .map(|id| LlmEvent::TextEnd { id }),
                );
                events.extend(
                    self.state
                        .reasoning_ids
                        .drain()
                        .map(|id| LlmEvent::ReasoningEnd { id }),
                );
                events.push(LlmEvent::Finish {
                    reason: if self.state.emitted_tool_call {
                        FinishReason::ToolCall
                    } else {
                        FinishReason::Stop
                    },
                    usage: self.state.usage.take(),
                    response_id: self.state.response_id.clone(),
                });
                terminal = true;
            }
            "error" | "response.failed" => {
                let message = payload
                    .pointer("/error/message")
                    .or_else(|| payload.get("message"))
                    .and_then(Value::as_str)
                    .unwrap_or("Responses provider stream failed");
                events.push(LlmEvent::ProviderError {
                    message: message.to_string(),
                    classification: Some(FailureClassification::ProviderInternal),
                    retryable: Some(false),
                });
                terminal = true;
            }
            _ => {}
        }
        Ok(ResponsesEventBatch { events, terminal })
    }

    pub fn finish_incomplete(mut self) -> Vec<LlmEvent> {
        let mut events = Vec::new();
        events.extend(
            self.state
                .text_ids
                .drain()
                .map(|id| LlmEvent::TextEnd { id }),
        );
        events.extend(
            self.state
                .reasoning_ids
                .drain()
                .map(|id| LlmEvent::ReasoningEnd { id }),
        );
        events.push(LlmEvent::ProviderError {
            message: "stream closed before response.completed".to_string(),
            classification: Some(FailureClassification::Transport),
            retryable: Some(true),
        });
        events
    }
}

pub(super) fn responses_sse(
    response: Response,
    allow_openai_response_metadata: bool,
) -> impl Stream<Item = Result<LlmEvent, CurrentProviderError>> + Send {
    let server_model = response_header_model(response.headers());
    try_stream! {
        let mut reducer = ResponsesEventReducer::new(
            server_model.clone(),
            allow_openai_response_metadata,
        );
        if let Some(model) = server_model {
            yield LlmEvent::ServerModel { model };
        }
        let mut frames = Box::pin(sse_frames(response));
        while let Some(frame) = frames.next().await {
            let frame = frame?;
            let payload: Value = serde_json::from_str(&frame.data)
                .map_err(|error| CurrentProviderError::new(format!("解析 Responses SSE event 失败: {error}")))?;
            let batch = reducer.push(&payload)?;
            if batch.terminal {
                drop(frames);
                for event in batch.events {
                    yield event;
                }
                return;
            }
            for event in batch.events {
                yield event;
            }
        }
        for event in reducer.finish_incomplete() {
            yield event;
        }
    }
}

pub(super) fn response_header_model(headers: &reqwest::header::HeaderMap) -> Option<String> {
    ["openai-model", "x-openai-model"]
        .into_iter()
        .find_map(|name| headers.get(name))
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn response_server_model(payload: &Value) -> Option<String> {
    payload
        .pointer("/response/headers")
        .and_then(response_model_from_json_headers)
        .or_else(|| {
            payload
                .get("headers")
                .and_then(response_model_from_json_headers)
        })
}

fn response_model_from_json_headers(headers: &Value) -> Option<String> {
    headers.as_object()?.iter().find_map(|(name, value)| {
        if !name.eq_ignore_ascii_case("openai-model")
            && !name.eq_ignore_ascii_case("x-openai-model")
        {
            return None;
        }
        json_header_string(value)
    })
}

fn json_header_string(value: &Value) -> Option<String> {
    let value = match value {
        Value::String(value) => Some(value.as_str()),
        Value::Array(values) => values.first().and_then(Value::as_str),
        _ => None,
    }?;
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn response_model_verifications(payload: &Value) -> Vec<ModelVerification> {
    payload
        .pointer("/metadata/openai_verification_recommendation")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter_map(|value| match value {
            "trusted_access_for_cyber" => Some(ModelVerification::TrustedAccessForCyber),
            _ => None,
        })
        .collect()
}

fn response_turn_moderation_metadata(payload: &Value) -> Option<Value> {
    payload
        .pointer("/metadata/openai_chatgpt_moderation_metadata")
        .cloned()
}

fn take_responses_calls(
    state: &mut ResponsesStreamState,
) -> Result<Vec<LlmEvent>, CurrentProviderError> {
    let keys = state.calls.keys().cloned().collect::<Vec<_>>();
    let mut events = Vec::new();
    for key in keys {
        events.extend(take_responses_call(state, &key)?);
    }
    Ok(events)
}

fn take_responses_custom_calls(
    state: &mut ResponsesStreamState,
) -> Result<Vec<LlmEvent>, CurrentProviderError> {
    let keys = state.custom_calls.keys().cloned().collect::<Vec<_>>();
    let mut events = Vec::new();
    for key in keys {
        events.extend(take_responses_custom_call(state, &key)?);
    }
    Ok(events)
}

fn take_responses_custom_call(
    state: &mut ResponsesStreamState,
    key: &str,
) -> Result<Vec<LlmEvent>, CurrentProviderError> {
    let call = state
        .custom_calls
        .get_mut(key)
        .expect("response custom tool call must be accumulated before emission");
    let mut events = Vec::new();
    if let Some((id, name, input)) = call.begin_input_if_ready(key) {
        events.push(LlmEvent::ToolInputStart {
            id: id.clone(),
            name: name.clone(),
        });
        events.push(LlmEvent::ToolInputDelta {
            id,
            name,
            text: input,
        });
    }
    let Some(call) = call.into_call(key) else {
        return Ok(events);
    };
    state.emitted_tool_call = true;
    events.push(LlmEvent::ToolInputEnd {
        id: call.id.clone(),
        name: call.name.clone(),
    });
    events.push(LlmEvent::CustomToolCall {
        id: call.id,
        name: call.name,
        input: call.input,
        namespace: call.namespace,
        provider_metadata: call.provider_metadata,
    });
    Ok(events)
}

fn take_responses_call(
    state: &mut ResponsesStreamState,
    key: &str,
) -> Result<Vec<LlmEvent>, CurrentProviderError> {
    let mut events = Vec::new();
    let call = state
        .calls
        .get_mut(key)
        .expect("response tool call must be accumulated before emission");
    if let Some((id, name, arguments)) = call.begin_input_if_ready(key) {
        events.push(LlmEvent::ToolInputStart {
            id: id.clone(),
            name: name.clone(),
        });
        events.push(LlmEvent::ToolInputDelta {
            id,
            name,
            text: arguments,
        });
    }
    let Some(call) = call.into_call(key) else {
        return Ok(events);
    };
    if state.emitted_calls.insert(call.id.clone()) {
        state.emitted_tool_call = true;
        events.push(LlmEvent::ToolInputEnd {
            id: call.id.clone(),
            name: call.name.clone(),
        });
        events.push(LlmEvent::ToolCall {
            id: call.id,
            name: call.name,
            input: call.arguments,
            raw_arguments: Some(call.raw_arguments),
            provider_executed: None,
            provider_metadata: Default::default(),
        });
    }
    Ok(events)
}

fn response_block_id(payload: &Value, prefix: &str) -> String {
    payload
        .get("item_id")
        .or_else(|| payload.get("output_index"))
        .map(|value| {
            value
                .as_str()
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| value.to_string())
        })
        .map(|id| format!("{prefix}-{id}"))
        .unwrap_or_else(|| format!("{prefix}-0"))
}

fn response_reasoning_id(payload: &Value, state: &ResponsesStreamState) -> String {
    payload
        .get("item_id")
        .map(|_| response_block_id(payload, "reasoning"))
        .or_else(|| state.active_reasoning_id.clone())
        .unwrap_or_else(|| response_block_id(payload, "reasoning"))
}

fn observe_active_responses_reasoning_item(item: &Value, state: &mut ResponsesStreamState) {
    if item.get("type").and_then(Value::as_str) != Some("reasoning") {
        return;
    }
    state.active_reasoning_id = Some(response_item_id(item, "reasoning"));
}

fn clear_active_responses_reasoning_item(item: &Value, state: &mut ResponsesStreamState) {
    if item.get("type").and_then(Value::as_str) != Some("reasoning") {
        return;
    }
    let item_id = response_item_id(item, "reasoning");
    if state.active_reasoning_id.as_deref() == Some(item_id.as_str()) {
        state.active_reasoning_id = None;
    }
}

fn response_item_id(item: &Value, prefix: &str) -> String {
    item.get("id")
        .or_else(|| item.get("item_id"))
        .or_else(|| item.get("output_index"))
        .map(|value| {
            value
                .as_str()
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| value.to_string())
        })
        .map(|id| format!("{prefix}-{id}"))
        .unwrap_or_else(|| format!("{prefix}-0"))
}

fn response_call_key(payload: &Value) -> String {
    payload
        .get("call_id")
        .or_else(|| payload.get("item_id"))
        .or_else(|| payload.get("output_index"))
        .map(|value| match value {
            Value::String(value) => value.clone(),
            other => other.to_string(),
        })
        .unwrap_or_else(|| "response_call".to_string())
}

fn absorb_responses_call(item: &Value, state: &mut ResponsesStreamState) {
    if item.get("type").and_then(Value::as_str) != Some("function_call") {
        return;
    }
    let key = response_call_key(item);
    let call = state.calls.entry(key).or_default();
    call.id = item
        .get("call_id")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .or(call.id.clone());
    call.name = item
        .get("name")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .or(call.name.clone());
    if let Some(arguments) = item.get("arguments").and_then(Value::as_str) {
        call.arguments = arguments.to_string();
    }
}

fn absorb_responses_custom_call(item: &Value, state: &mut ResponsesStreamState) {
    if item.get("type").and_then(Value::as_str) != Some("custom_tool_call") {
        return;
    }
    let key = response_call_key(item);
    let call = state.custom_calls.entry(key).or_default();
    call.id = item
        .get("call_id")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .or(call.id.clone());
    call.name = item
        .get("name")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .or(call.name.clone());
    call.namespace = item
        .get("namespace")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .or(call.namespace.clone());
    if let Some(input) = item.get("input").and_then(Value::as_str) {
        call.input = input.to_string();
    }
}

#[cfg(test)]
pub(super) fn response_item_tool_call(
    item: &Value,
) -> Result<Option<CurrentProviderToolCall>, CurrentProviderError> {
    if item.get("type").and_then(Value::as_str) != Some("function_call") {
        return Ok(None);
    }
    let id = item
        .get("call_id")
        .and_then(Value::as_str)
        .ok_or_else(|| CurrentProviderError::new("Responses function_call omitted call_id"))?
        .to_string();
    let name = item
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| CurrentProviderError::new("Responses function_call omitted name"))?
        .to_string();
    let arguments = item
        .get("arguments")
        .and_then(Value::as_str)
        .unwrap_or("{}")
        .to_string();
    Ok(Some(CurrentProviderToolCall::from_raw(id, name, arguments)))
}

fn responses_usage(value: &Value) -> Usage {
    let number = |key: &str| value.get(key).and_then(Value::as_u64);
    Usage {
        input_tokens: number("input_tokens"),
        output_tokens: number("output_tokens"),
        cache_read_input_tokens: value
            .pointer("/input_tokens_details/cached_tokens")
            .and_then(Value::as_u64),
        total_tokens: number("total_tokens"),
        ..Usage::default()
    }
}

#[derive(Debug, Default)]
struct AnthropicStreamState {
    response_id: Option<String>,
    calls: BTreeMap<u32, ToolCallAccumulator>,
    emitted_tool_call: bool,
    usage: Option<Usage>,
    text_ids: HashSet<String>,
    reasoning_ids: HashSet<String>,
    finish_reason: Option<FinishReason>,
}

pub(super) fn anthropic_sse(
    response: Response,
) -> impl Stream<Item = Result<LlmEvent, CurrentProviderError>> + Send {
    try_stream! {
        let mut state = AnthropicStreamState::default();
        let mut frames = Box::pin(sse_frames(response));
        while let Some(frame) = frames.next().await {
            let frame = frame?;
            let event: anthropic::AnthropicStreamEvent = serde_json::from_str(&frame.data)
                .map_err(|error| CurrentProviderError::new(format!("解析 Anthropic SSE event 失败: {error}")))?;
            match event {
                anthropic::AnthropicStreamEvent::MessageStart { message } => {
                    state.response_id = Some(message.id);
                    if let Some(usage) = message.usage {
                        let usage = anthropic_usage(usage);
                        yield LlmEvent::Usage { usage: usage.clone() };
                        state.usage = Some(usage);
                    }
                }
                anthropic::AnthropicStreamEvent::ContentBlockStart { index, content_block } => match content_block {
                    anthropic::AnthropicContentBlock::Text { text } => {
                        let id = format!("text-{index}");
                        state.text_ids.insert(id.clone());
                        yield LlmEvent::TextStart { id: id.clone() };
                        if !text.is_empty() { yield LlmEvent::TextDelta { id, text }; }
                    }
                    anthropic::AnthropicContentBlock::Thinking { thinking, .. } => {
                        let id = format!("reasoning-{index}");
                        state.reasoning_ids.insert(id.clone());
                        yield LlmEvent::ReasoningStart { id: id.clone() };
                        if !thinking.is_empty() {
                            yield LlmEvent::ReasoningContentDelta {
                                id,
                                text: thinking,
                                content_index: 0,
                            };
                        }
                    }
                    anthropic::AnthropicContentBlock::ToolUse { id, name, input } => {
                        let call = state.calls.entry(index).or_default();
                        call.id = Some(id.clone()); call.name = Some(name.clone()); call.started = true;
                        if input != Value::Null && input != json!({}) { call.arguments = serde_json::to_string(&input).unwrap_or_else(|_| "{}".to_string()); }
                        yield LlmEvent::ToolInputStart { id, name };
                    }
                    anthropic::AnthropicContentBlock::ToolResult { .. } | anthropic::AnthropicContentBlock::Image { .. } => {
                        yield LlmEvent::ProviderError { message: "unsupported Anthropic output content block".to_string(), classification: Some(FailureClassification::InvalidRequest), retryable: Some(false) };
                        return;
                    }
                },
                anthropic::AnthropicStreamEvent::ContentBlockDelta { index, delta } => match delta {
                    anthropic::AnthropicDelta::TextDelta { text } => if !text.is_empty() {
                        let id = format!("text-{index}");
                        if state.text_ids.insert(id.clone()) { yield LlmEvent::TextStart { id: id.clone() }; }
                        yield LlmEvent::TextDelta { id, text };
                    },
                    anthropic::AnthropicDelta::ThinkingDelta { thinking } => if !thinking.is_empty() {
                        let id = format!("reasoning-{index}");
                        if state.reasoning_ids.insert(id.clone()) { yield LlmEvent::ReasoningStart { id: id.clone() }; }
                        yield LlmEvent::ReasoningContentDelta {
                            id,
                            text: thinking,
                            content_index: 0,
                        };
                    },
                    anthropic::AnthropicDelta::InputJsonDelta { partial_json } => {
                        let call = state.calls.entry(index).or_default();
                        let Some(name) = call.name.clone() else {
                            yield LlmEvent::ProviderError { message: "Anthropic tool call stream omitted tool name".to_string(), classification: Some(FailureClassification::InvalidRequest), retryable: Some(false) };
                            return;
                        };
                        let id = call.call_id(&format!("tool_{index}"));
                        if !call.started { call.started = true; yield LlmEvent::ToolInputStart { id: id.clone(), name: name.clone() }; }
                        call.arguments.push_str(&partial_json);
                        yield LlmEvent::ToolInputDelta { id, name, text: partial_json };
                    }
                    anthropic::AnthropicDelta::SignatureDelta { .. } => {}
                },
                anthropic::AnthropicStreamEvent::ContentBlockStop { index } => {
                    let text_id = format!("text-{index}");
                    if state.text_ids.remove(&text_id) { yield LlmEvent::TextEnd { id: text_id }; }
                    let reasoning_id = format!("reasoning-{index}");
                    if state.reasoning_ids.remove(&reasoning_id) { yield LlmEvent::ReasoningEnd { id: reasoning_id }; }
                    if let Some(call) = state
                        .calls
                        .get_mut(&index)
                        .and_then(|call| call.into_call(&format!("tool_{index}")))
                    {
                        state.emitted_tool_call = true;
                        yield LlmEvent::ToolInputEnd { id: call.id.clone(), name: call.name.clone() };
                        yield LlmEvent::ToolCall { id: call.id, name: call.name, input: call.arguments, raw_arguments: Some(call.raw_arguments), provider_executed: None, provider_metadata: Default::default() };
                    }
                }
                anthropic::AnthropicStreamEvent::MessageDelta { delta, usage } => {
                    let usage = anthropic_usage(usage);
                    yield LlmEvent::Usage { usage: usage.clone() };
                    state.usage = Some(usage);
                    state.finish_reason = delta.stop_reason.as_deref().map(anthropic_finish_reason);
                }
                anthropic::AnthropicStreamEvent::MessageStop => {
                    drop(frames);
                    for id in state.text_ids.drain() { yield LlmEvent::TextEnd { id }; }
                    for id in state.reasoning_ids.drain() { yield LlmEvent::ReasoningEnd { id }; }
                    yield LlmEvent::Finish {
                        reason: state.finish_reason.unwrap_or_else(|| {
                            if state.emitted_tool_call { FinishReason::ToolCall } else { FinishReason::Stop }
                        }),
                        usage: state.usage,
                        response_id: state.response_id,
                    };
                    return;
                }
            }
        }
        for id in state.text_ids.drain() { yield LlmEvent::TextEnd { id }; }
        for id in state.reasoning_ids.drain() { yield LlmEvent::ReasoningEnd { id }; }
        yield truncated_stream_error("Anthropic Messages");
    }
}

fn anthropic_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "end_turn" | "stop_sequence" => FinishReason::Stop,
        "tool_use" => FinishReason::ToolCall,
        "max_tokens" => FinishReason::Length,
        "refusal" => FinishReason::ContentFilter,
        _ => FinishReason::Unknown,
    }
}

fn truncated_stream_error(protocol: &str) -> LlmEvent {
    LlmEvent::ProviderError {
        message: format!("{protocol} stream ended before its terminal event"),
        classification: Some(FailureClassification::Transport),
        retryable: Some(true),
    }
}

fn anthropic_usage(usage: anthropic::AnthropicUsage) -> Usage {
    Usage {
        input_tokens: Some(usage.input_tokens as u64),
        output_tokens: Some(usage.output_tokens as u64),
        cache_read_input_tokens: usage.cache_read_input_tokens.map(u64::from),
        cache_write_input_tokens: usage.cache_creation_input_tokens.map(u64::from),
        ..Usage::default()
    }
}

#[derive(Debug)]
pub(super) struct SseFrame {
    pub(super) data: String,
}

fn sse_frames(
    response: Response,
) -> impl Stream<Item = Result<SseFrame, CurrentProviderError>> + Send {
    sse_frames_with_idle_timeout(response, DEFAULT_STREAM_IDLE_TIMEOUT)
}

pub(super) fn sse_frames_with_idle_timeout(
    response: Response,
    idle_timeout: Duration,
) -> impl Stream<Item = Result<SseFrame, CurrentProviderError>> + Send {
    try_stream! {
        let mut pending = Vec::new();
        let mut bytes = response.bytes_stream();
        loop {
            let next = tokio::time::timeout(idle_timeout, bytes.next())
                .await
                .map_err(|_| CurrentProviderError::new(format!(
                    "读取 provider SSE 超时: {} ms 内未收到数据",
                    idle_timeout.as_millis()
                )))?;
            let Some(next) = next else { break; };
            let next = next.map_err(|error| CurrentProviderError::new(format!(
                "读取 provider SSE 失败: {}",
                super::transport::error_chain(&error)
            )))?;
            pending.extend_from_slice(&next);
            for frame in drain_sse_frames(&mut pending)? { yield frame; }
        }
        let pending = std::str::from_utf8(&pending).map_err(|error| CurrentProviderError::new(format!("解析 provider SSE UTF-8 失败: {error}")))?;
        if let Some(frame) = parse_sse_frame(pending) { yield frame; }
    }
}

pub(super) fn drain_sse_frames(
    pending: &mut Vec<u8>,
) -> Result<Vec<SseFrame>, CurrentProviderError> {
    let mut frames = Vec::new();
    while let Some(frame_end) = find_sse_frame_end(pending) {
        let frame = std::str::from_utf8(&pending[..frame_end])
            .map_err(|error| {
                CurrentProviderError::new(format!("解析 provider SSE UTF-8 失败: {error}"))
            })?
            .to_string();
        let next_start = skip_sse_frame_end(pending, frame_end);
        pending.drain(..next_start);
        if let Some(frame) = parse_sse_frame(&frame) {
            frames.push(frame);
        }
    }
    Ok(frames)
}

fn find_sse_frame_end(value: &[u8]) -> Option<usize> {
    value
        .windows(2)
        .position(|window| window == b"\n\n")
        .or_else(|| value.windows(4).position(|window| window == b"\r\n\r\n"))
}

fn skip_sse_frame_end(value: &[u8], frame_end: usize) -> usize {
    if value[frame_end..].starts_with(b"\r\n\r\n") {
        frame_end + 4
    } else {
        frame_end + 2
    }
}

pub(super) fn parse_sse_frame(frame: &str) -> Option<SseFrame> {
    let data = frame
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim_start)
        .collect::<Vec<_>>()
        .join("\n");
    (!data.is_empty()).then_some(SseFrame { data })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finish_reason_mapping_preserves_terminal_semantics() {
        assert_eq!(openai_finish_reason("tool_calls"), FinishReason::ToolCall);
        assert_eq!(openai_finish_reason("length"), FinishReason::Length);
        assert_eq!(anthropic_finish_reason("max_tokens"), FinishReason::Length);
        assert_eq!(
            anthropic_finish_reason("refusal"),
            FinishReason::ContentFilter
        );
    }

    #[test]
    fn truncated_stream_is_retryable_transport_failure() {
        assert!(matches!(
            truncated_stream_error("OpenAI Responses"),
            LlmEvent::ProviderError {
                classification: Some(FailureClassification::Transport),
                retryable: Some(true),
                ..
            }
        ));
    }

    #[test]
    fn trusted_responses_metadata_preserves_each_moderation_update_as_opaque_json() {
        let mut reducer = ResponsesEventReducer::new(None, true);
        let first = reducer
            .push(&json!({
                "type": "response.metadata",
                "metadata": {
                    "openai_chatgpt_moderation_metadata": {
                        "presentation": "inline"
                    }
                }
            }))
            .expect("first moderation metadata");
        let second = reducer
            .push(&json!({
                "type": "response.metadata",
                "metadata": {
                    "openai_chatgpt_moderation_metadata": ["updated", 2]
                }
            }))
            .expect("second moderation metadata");

        assert_eq!(
            first.events,
            vec![LlmEvent::TurnModerationMetadata {
                metadata: json!({ "presentation": "inline" }),
            }]
        );
        assert_eq!(
            second.events,
            vec![LlmEvent::TurnModerationMetadata {
                metadata: json!(["updated", 2]),
            }]
        );
    }

    #[test]
    fn untrusted_responses_metadata_does_not_emit_moderation_metadata() {
        let events = ResponsesEventReducer::new(None, false)
            .push(&json!({
                "type": "response.metadata",
                "metadata": {
                    "openai_chatgpt_moderation_metadata": {
                        "presentation": "inline"
                    }
                }
            }))
            .expect("ignored moderation metadata")
            .events;

        assert!(events.is_empty());
    }

    #[test]
    fn openai_tool_call_with_blank_name_preserves_runtime_repair_input() {
        let call = CurrentProviderToolCall::from_raw(
            "call-blank-name".to_string(),
            "   ".to_string(),
            "{}".to_string(),
        );

        assert_eq!(call.name, "");
        assert_eq!(call.arguments, json!({}));
        assert_eq!(call.raw_arguments, "{}");
    }

    #[test]
    fn tool_call_accumulator_accepts_arguments_before_name() {
        let mut call = ToolCallAccumulator {
            id: Some("call-1".to_string()),
            arguments: "{\"query\":\"latest Rust release\"}".to_string(),
            ..Default::default()
        };

        assert!(call.begin_input_if_ready("fallback").is_none());

        call.name = Some("WebSearch".to_string());
        assert_eq!(
            call.begin_input_if_ready("fallback"),
            Some((
                "call-1".to_string(),
                "WebSearch".to_string(),
                "{\"query\":\"latest Rust release\"}".to_string(),
            ))
        );
        let call = call.into_call("fallback").expect("tool call");
        assert_eq!(call.name, "WebSearch");
        assert_eq!(call.arguments["query"], "latest Rust release");
    }

    #[test]
    fn tool_call_accumulator_preserves_terminal_arguments_without_name() {
        let mut call = ToolCallAccumulator {
            arguments: "{}".to_string(),
            ..Default::default()
        };

        let call = call
            .into_call("call-1")
            .expect("incomplete tool call must reach runtime repair");

        assert_eq!(call.id, "call-1");
        assert_eq!(call.name, "");
        assert_eq!(call.arguments, json!({}));
    }

    #[test]
    fn completed_custom_tool_call_emits_complete_input_lifecycle() {
        let events = ResponsesEventReducer::new(None, false)
            .push(&json!({
                "type": "response.completed",
                "response": {
                    "id": "response-1",
                    "output": [{
                        "type": "custom_tool_call",
                        "call_id": "custom-call-1",
                        "name": "run_code",
                        "namespace": "codemode",
                        "input": "return 42;"
                    }]
                }
            }))
            .expect("completed custom tool call")
            .events;

        assert_eq!(
            events,
            vec![
                LlmEvent::ToolInputStart {
                    id: "custom-call-1".to_string(),
                    name: "run_code".to_string(),
                },
                LlmEvent::ToolInputDelta {
                    id: "custom-call-1".to_string(),
                    name: "run_code".to_string(),
                    text: "return 42;".to_string(),
                },
                LlmEvent::ToolInputEnd {
                    id: "custom-call-1".to_string(),
                    name: "run_code".to_string(),
                },
                LlmEvent::CustomToolCall {
                    id: "custom-call-1".to_string(),
                    name: "run_code".to_string(),
                    input: "return 42;".to_string(),
                    namespace: Some("codemode".to_string()),
                    provider_metadata: Default::default(),
                },
                LlmEvent::Finish {
                    reason: FinishReason::ToolCall,
                    usage: None,
                    response_id: Some("response-1".to_string()),
                },
            ]
        );
    }
}
