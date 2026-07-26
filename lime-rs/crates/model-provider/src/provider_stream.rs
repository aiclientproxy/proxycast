//! Provider reply stream 的 current contract。
//!
//! 该模块只描述 Lime runtime 侧可传递的 provider handle 和 stream request，
//! 不持有具体 provider trait object，不能把具体 client 类型暴露给 current 调用面。

use crate::runtime_provider::{RuntimeProviderConfig, RuntimeProviderProtocol};
use crate::safety::{
    parse_safety_buffering_runtime_event_payload, ProviderSafetyBufferingRuntimeEventPayload,
    SAFETY_BUFFERING_RUNTIME_EVENT_KIND,
};
use crate::ModelProviderProtocol;
use agent_protocol::provider_trace::ProviderTraceEvent;
use serde::{Deserialize, Serialize};

mod failure;
mod message_output;
mod model_change;
mod notification;
mod plaintext_tool_use;
mod poll;
mod progress;
mod response_content;
mod response_context;
mod response_event;
mod sampling;
mod sampling_output;
mod text_delta;
mod tool_input_delta;
mod usage;

pub use agent_protocol::provider_trace::{
    ProviderTraceEvent as RuntimeReplyProviderTraceEvent,
    ProviderTraceFailure as RuntimeReplyProviderTraceFailure,
    ProviderTraceStage as RuntimeReplyProviderTraceStage,
};
pub use failure::{
    provider_stream_failure_message_should_log_as_warning,
    provider_stream_failure_should_log_as_error, provider_stream_trace_failure,
    RuntimeReplyProviderFailure, RuntimeReplyProviderFailureKind,
};
pub use message_output::{
    provider_stream_message_outputs, provider_stream_single_message_output,
    RuntimeReplyProviderMessageOutput,
};
pub use model_change::{
    provider_stream_model_change, RuntimeReplyProviderModelChange,
    RuntimeReplyProviderModelChangeMode,
};
pub use notification::{
    provider_stream_has_notification_text, provider_stream_notification_payload_from_text,
    provider_stream_notification_payload_from_texts, provider_stream_notification_text,
    PROVIDER_STREAM_EVENT_NOTIFICATION_PREFIX,
};
pub use plaintext_tool_use::{
    provider_stream_plaintext_tool_use_is_complete, provider_stream_plaintext_tool_use_progress,
    provider_stream_plaintext_tool_use_start, provider_stream_plaintext_tool_uses,
    RuntimeReplyProviderPlaintextToolCall, RuntimeReplyProviderPlaintextToolUse,
    RuntimeReplyProviderPlaintextToolUseProgress, RuntimeReplyProviderPlaintextToolUseStream,
    RuntimeReplyProviderPlaintextToolUseStreamEvent, PROVIDER_STREAM_PLAINTEXT_TOOL_USE_PROVIDER,
};
pub use poll::{
    provider_stream_cancel_poll_interval, provider_stream_event_poll, provider_stream_timeout_poll,
    ProviderStreamCancelReason, ProviderStreamPoll, PROVIDER_STREAM_CANCEL_BEFORE_EVENT_REASON,
    PROVIDER_STREAM_CANCEL_POLL_INTERVAL, PROVIDER_STREAM_CANCEL_WHILE_WAITING_REASON,
};
pub use progress::RuntimeReplyProviderStreamProgress;
pub use response_content::{
    provider_stream_direct_answer_should_bypass_tool_execution,
    provider_stream_direct_answer_should_strip_response_content,
    provider_stream_response_first_text_delta_chars,
    provider_stream_response_has_notification_text, provider_stream_response_outcome,
    provider_stream_response_route, provider_stream_response_text_chars,
    provider_stream_response_tool_input_delta_events, RuntimeReplyProviderLeadWorkerModels,
    RuntimeReplyProviderResponseContent, RuntimeReplyProviderResponseOutcome,
    RuntimeReplyProviderResponseRoute, RuntimeReplyProviderResponseSession,
};
pub use response_context::{
    provider_stream_response_context_from_header_pairs, RuntimeReplyProviderResponseContext,
};
pub use response_event::{
    RuntimeReplyResponseEvent, RuntimeReplyResponseItem, RuntimeReplyResponseItemPayload,
};
pub use sampling::{
    provider_stream_should_retry_empty_first_content, RuntimeReplyProviderSamplingFailureLogLevel,
    RuntimeReplyProviderSamplingMode, RuntimeReplyProviderSamplingRequest,
    RuntimeReplyProviderSamplingSession, RuntimeReplyProviderSamplingStreamItem,
    PROVIDER_EMPTY_STREAM_RETRY_MARKER,
};
pub use sampling_output::{
    provider_stream_open_sampled_message_outputs, provider_stream_sampled_message_outputs,
    RuntimeReplyProviderPlaintextMessageNormalizer, RuntimeReplyProviderSampledMessageStream,
};
pub use text_delta::provider_stream_first_text_delta_chars;
pub use tool_input_delta::{
    provider_stream_tool_input_delta_events, RuntimeReplyProviderToolInputDelta,
};
pub use usage::{RuntimeReplyProviderTokenUsage, RuntimeReplyProviderUsage};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeProviderBackend {
    Current,
}

impl RuntimeProviderBackend {
    pub fn as_wire_str(self) -> &'static str {
        match self {
            Self::Current => "current",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeReplyInputKind {
    UserMessage,
    ActionRequiredResponse,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeReplyProviderIdentity {
    pub provider_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_selector: Option<String>,
    pub model_name: String,
    pub credential_uuid: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protocol: Option<ModelProviderProtocol>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(default)]
    pub toolshim: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub toolshim_model: Option<String>,
}

impl RuntimeReplyProviderIdentity {
    pub fn from_config(config: &RuntimeProviderConfig) -> Self {
        Self {
            provider_name: config.provider_name.clone(),
            provider_selector: config.provider_selector.clone(),
            model_name: config.model_name.clone(),
            credential_uuid: config.credential_uuid.clone(),
            protocol: config
                .protocol
                .map(RuntimeProviderProtocol::to_model_provider_protocol),
            reasoning_effort: config.reasoning_effort.clone(),
            service_tier: config.service_tier.clone(),
            toolshim: config.toolshim,
            toolshim_model: config.toolshim_model.clone(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeReplyProviderCapabilities {
    pub supports_streaming: bool,
    pub supports_embeddings: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_model_name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeReplyProviderHandle {
    pub identity: RuntimeReplyProviderIdentity,
    pub backend: RuntimeProviderBackend,
    #[serde(default)]
    pub capabilities: RuntimeReplyProviderCapabilities,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeReplyProviderTraceMetadata {
    pub provider_name: String,
    pub model_name: String,
    pub runtime_provider_backend: String,
    pub runtime_provider_selector: Option<String>,
    pub runtime_provider_protocol: Option<String>,
    pub runtime_provider_active_model: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeReplyProviderBinding<B> {
    handle: RuntimeReplyProviderHandle,
    backend: B,
}

impl RuntimeReplyProviderHandle {
    pub fn from_config(config: &RuntimeProviderConfig, backend: RuntimeProviderBackend) -> Self {
        Self {
            identity: RuntimeReplyProviderIdentity::from_config(config),
            backend,
            capabilities: RuntimeReplyProviderCapabilities::default(),
        }
    }

    pub fn with_capabilities(mut self, capabilities: RuntimeReplyProviderCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    pub fn provider_name(&self) -> &str {
        &self.identity.provider_name
    }

    pub fn model_name(&self) -> &str {
        &self.identity.model_name
    }

    pub fn provider_trace_metadata(&self) -> RuntimeReplyProviderTraceMetadata {
        RuntimeReplyProviderTraceMetadata::from_handle(self)
    }
}

impl RuntimeReplyProviderTraceMetadata {
    pub fn from_handle(provider: &RuntimeReplyProviderHandle) -> Self {
        Self {
            provider_name: provider.identity.provider_name.clone(),
            model_name: provider.identity.model_name.clone(),
            runtime_provider_backend: provider.backend.as_wire_str().to_string(),
            runtime_provider_selector: provider.identity.provider_selector.clone(),
            runtime_provider_protocol: provider
                .identity
                .protocol
                .as_ref()
                .map(model_provider_protocol_wire_value),
            runtime_provider_active_model: provider.capabilities.active_model_name.clone(),
        }
    }

    pub fn apply_to_provider_trace_event(&self, event: &mut ProviderTraceEvent) {
        if event.provider.trim().is_empty() {
            event.provider = self.provider_name.clone();
        }
        if event.model.trim().is_empty() {
            event.model = self.model_name.clone();
        }
        event.runtime_provider_backend = Some(self.runtime_provider_backend.clone());
        event.runtime_provider_selector = self.runtime_provider_selector.clone();
        event.runtime_provider_protocol = self.runtime_provider_protocol.clone();
        event.runtime_provider_active_model = self.runtime_provider_active_model.clone();
    }
}

pub fn apply_runtime_provider_metadata(
    event: &mut ProviderTraceEvent,
    provider: Option<&RuntimeReplyProviderHandle>,
) {
    let Some(provider) = provider else {
        return;
    };
    provider
        .provider_trace_metadata()
        .apply_to_provider_trace_event(event);
}

fn model_provider_protocol_wire_value(protocol: &ModelProviderProtocol) -> String {
    match protocol {
        ModelProviderProtocol::Responses => "responses".to_string(),
        ModelProviderProtocol::ChatCompletions => "chat_completions".to_string(),
        ModelProviderProtocol::AnthropicMessages => "anthropic_messages".to_string(),
        ModelProviderProtocol::Custom(value) => value.clone(),
    }
}

impl<B> RuntimeReplyProviderBinding<B> {
    pub fn new(handle: RuntimeReplyProviderHandle, backend: B) -> Self {
        Self { handle, backend }
    }

    pub fn handle(&self) -> &RuntimeReplyProviderHandle {
        &self.handle
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn into_backend(self) -> B {
        self.backend
    }

    pub fn into_parts(self) -> (RuntimeReplyProviderHandle, B) {
        (self.handle, self.backend)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeReplyStreamRequest {
    pub session_id: String,
    pub input_kind: RuntimeReplyInputKind,
    pub message_chars: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<RuntimeReplyProviderHandle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_request_policy: Option<RuntimeReplyModelRequestPolicy>,
}

impl RuntimeReplyStreamRequest {
    pub fn new(
        session_id: impl Into<String>,
        input_kind: RuntimeReplyInputKind,
        message_chars: usize,
        provider: Option<RuntimeReplyProviderHandle>,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            input_kind,
            message_chars,
            provider,
            model_request_policy: None,
        }
    }

    pub fn with_model_request_policy(
        mut self,
        model_request_policy: Option<RuntimeReplyModelRequestPolicy>,
    ) -> Self {
        self.model_request_policy = model_request_policy;
        self
    }

    pub fn provider_backend(&self) -> Option<RuntimeProviderBackend> {
        self.provider.as_ref().map(|provider| provider.backend)
    }

    pub fn provider_name(&self) -> Option<&str> {
        self.provider
            .as_ref()
            .map(RuntimeReplyProviderHandle::provider_name)
    }

    pub fn model_name(&self) -> Option<&str> {
        self.provider
            .as_ref()
            .map(RuntimeReplyProviderHandle::model_name)
    }

    pub fn provider_request_wire_shape(&self) -> RuntimeReplyProviderRequestWireShape {
        RuntimeReplyProviderRequestWireShape::from_model_request_policy(
            self.model_request_policy.as_ref(),
        )
    }

    pub fn provider_request_wire_support_issue(
        &self,
    ) -> Option<RuntimeReplyProviderWireSupportIssue> {
        let wire_shape = self.provider_request_wire_shape();
        if provider_supports_request_wire_shape(self.provider.as_ref(), &wire_shape) {
            return None;
        }
        Some(RuntimeReplyProviderWireSupportIssue {
            provider_backend: self.provider_backend(),
            provider_name: self.provider_name().map(ToOwned::to_owned),
            model_name: self.model_name().map(ToOwned::to_owned),
            wire_shape,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeReplyProviderStreamStart {
    stream_request: RuntimeReplyStreamRequest,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeReplyProviderStreamTrace<'a> {
    pub session_id: &'a str,
    pub input_kind: RuntimeReplyInputKind,
    pub message_chars: usize,
    pub provider_backend: Option<RuntimeProviderBackend>,
    pub provider_name: Option<&'a str>,
    pub model_name: Option<&'a str>,
}

impl RuntimeReplyProviderStreamStart {
    pub fn new(
        stream_request: RuntimeReplyStreamRequest,
        expected_provider: &RuntimeReplyProviderHandle,
    ) -> Result<Self, RuntimeReplyProviderStartError> {
        let Some(provider) = stream_request.provider.as_ref() else {
            return Err(RuntimeReplyProviderStartError::new(format!(
                "Provider stream start requires a configured provider handle for session {}",
                stream_request.session_id
            )));
        };
        if provider != expected_provider {
            return Err(RuntimeReplyProviderStartError::new(format!(
                "Provider stream handle mismatch for session {}: expected {}/{}, got {}/{}",
                stream_request.session_id,
                expected_provider.provider_name(),
                expected_provider.model_name(),
                provider.provider_name(),
                provider.model_name()
            )));
        }

        Ok(Self { stream_request })
    }

    pub fn stream_request(&self) -> &RuntimeReplyStreamRequest {
        &self.stream_request
    }

    pub fn trace(&self) -> RuntimeReplyProviderStreamTrace<'_> {
        RuntimeReplyProviderStreamTrace {
            session_id: &self.stream_request.session_id,
            input_kind: self.stream_request.input_kind,
            message_chars: self.stream_request.message_chars,
            provider_backend: self.stream_request.provider_backend(),
            provider_name: self.stream_request.provider_name(),
            model_name: self.stream_request.model_name(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeReplyProviderStartError {
    pub message: String,
}

impl RuntimeReplyProviderStartError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

fn provider_supports_request_wire_shape(
    provider: Option<&RuntimeReplyProviderHandle>,
    wire_shape: &RuntimeReplyProviderRequestWireShape,
) -> bool {
    if !wire_shape.requires_responses_lite_wire_support() {
        return true;
    }

    let Some(provider) = provider else {
        return false;
    };

    provider.identity.provider_name == "openai"
        && provider
            .identity
            .protocol
            .as_ref()
            .is_some_and(ModelProviderProtocol::uses_responses_api)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeReplyProviderWireSupportIssue {
    pub provider_backend: Option<RuntimeProviderBackend>,
    pub provider_name: Option<String>,
    pub model_name: Option<String>,
    pub wire_shape: RuntimeReplyProviderRequestWireShape,
}

impl RuntimeReplyProviderWireSupportIssue {
    pub const MESSAGE: &'static str = "Provider request policy requires Responses Lite wire support, but the configured provider backend cannot safely apply the required header/reasoning payload yet; refusing to stream instead of silently dropping the policy.";

    pub fn message(&self) -> &'static str {
        Self::MESSAGE
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeReplyModelRequestPolicy {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub responses: Option<RuntimeReplyResponsesPolicy>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call: Option<RuntimeReplyToolCallPolicy>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_output: Option<RuntimeReplyReasoningOutputPolicy>,
}

impl RuntimeReplyModelRequestPolicy {
    pub fn new(
        responses: Option<RuntimeReplyResponsesPolicy>,
        tool_call: Option<RuntimeReplyToolCallPolicy>,
        reasoning_output: Option<RuntimeReplyReasoningOutputPolicy>,
    ) -> Option<Self> {
        (responses.is_some() || tool_call.is_some() || reasoning_output.is_some()).then_some(Self {
            responses,
            tool_call,
            reasoning_output,
        })
    }

    pub fn use_responses_lite(&self) -> bool {
        self.responses
            .as_ref()
            .is_some_and(|policy| policy.use_responses_lite)
    }

    pub fn reasoning_context(&self) -> Option<&str> {
        self.responses
            .as_ref()
            .map(|policy| policy.reasoning_context.as_str())
    }

    pub fn requires_responses_lite_header(&self) -> bool {
        self.responses
            .as_ref()
            .is_some_and(|policy| policy.requires_responses_lite_header)
    }

    pub fn parallel_tool_calls(&self) -> Option<bool> {
        self.tool_call
            .as_ref()
            .map(|policy| policy.parallel_tool_calls)
    }

    pub fn reasoning_summary(&self) -> Option<&str> {
        self.reasoning_output
            .as_ref()
            .and_then(RuntimeReplyReasoningOutputPolicy::reasoning_summary)
    }

    pub fn text_verbosity(&self) -> Option<&str> {
        self.reasoning_output
            .as_ref()
            .and_then(RuntimeReplyReasoningOutputPolicy::text_verbosity)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeReplyProviderRequestWireShape {
    #[serde(default)]
    pub use_responses_lite: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_context: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_summary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_verbosity: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub headers: Vec<RuntimeReplyProviderRequestHeader>,
}

impl RuntimeReplyProviderRequestWireShape {
    pub const TURN_CONTEXT_METADATA_KEY: &'static str = "provider_request_wire_shape";
    pub const RESPONSES_LITE_HEADER_NAME: &'static str = "x-openai-internal-codex-responses-lite";
    pub const RESPONSES_LITE_HEADER_VALUE: &'static str = "true";

    pub fn from_model_request_policy(policy: Option<&RuntimeReplyModelRequestPolicy>) -> Self {
        let Some(policy) = policy else {
            return Self::default();
        };

        let use_responses_lite = policy.use_responses_lite();
        let reasoning_context = policy.reasoning_context().map(ToOwned::to_owned);
        let reasoning_summary = policy.reasoning_summary().map(ToOwned::to_owned);
        let text_verbosity = policy.text_verbosity().map(ToOwned::to_owned);
        let parallel_tool_calls = policy.parallel_tool_calls();
        let headers = if policy.requires_responses_lite_header() {
            vec![RuntimeReplyProviderRequestHeader {
                name: Self::RESPONSES_LITE_HEADER_NAME.to_string(),
                value: Self::RESPONSES_LITE_HEADER_VALUE.to_string(),
            }]
        } else {
            Vec::new()
        };

        Self {
            use_responses_lite,
            reasoning_context,
            reasoning_summary,
            text_verbosity,
            parallel_tool_calls,
            headers,
        }
    }

    pub fn requires_responses_lite_wire_support(&self) -> bool {
        self.use_responses_lite
            || self.reasoning_context.as_deref() == Some("all_turns")
            || self
                .headers
                .iter()
                .any(RuntimeReplyProviderRequestHeader::is_responses_lite_header)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "payload")]
pub enum RuntimeReplyProviderStreamEvent {
    SafetyBuffering(ProviderSafetyBufferingRuntimeEventPayload),
}

impl RuntimeReplyProviderStreamEvent {
    pub const NOTIFICATION_KIND_SAFETY_BUFFERING: &'static str =
        "openai_responses.safety_buffering";

    pub fn from_notification_payload(
        stream_request: &RuntimeReplyStreamRequest,
        payload: &serde_json::Value,
    ) -> Option<Self> {
        let event_kind = payload
            .get("eventKind")
            .and_then(serde_json::Value::as_str)?;
        match event_kind {
            Self::NOTIFICATION_KIND_SAFETY_BUFFERING => {
                let response_event = payload.get("responseEvent")?;
                let headers = provider_stream_event_headers(payload);
                Self::safety_buffering_from_response_event(
                    stream_request,
                    response_event,
                    headers
                        .iter()
                        .map(|(name, value)| (name.as_str(), value.as_str())),
                )
            }
            _ => None,
        }
    }

    pub fn safety_buffering_from_response_event<'a>(
        stream_request: &RuntimeReplyStreamRequest,
        response_event: &serde_json::Value,
        headers: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Option<Self> {
        parse_safety_buffering_runtime_event_payload(
            response_event,
            headers,
            stream_request.provider_name(),
            stream_request.model_name(),
        )
        .map(Self::SafetyBuffering)
    }

    pub fn runtime_event_kind(&self) -> &'static str {
        match self {
            Self::SafetyBuffering(_) => SAFETY_BUFFERING_RUNTIME_EVENT_KIND,
        }
    }

    pub fn payload_json_value(&self) -> serde_json::Value {
        match self {
            Self::SafetyBuffering(payload) => payload.to_json_value(),
        }
    }
}

fn provider_stream_event_headers(payload: &serde_json::Value) -> Vec<(String, String)> {
    payload
        .get("headers")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|header| {
            let name = header.get("name").and_then(serde_json::Value::as_str)?;
            let value = header.get("value").and_then(serde_json::Value::as_str)?;
            Some((name.to_string(), value.to_string()))
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeReplyProviderRequestHeader {
    pub name: String,
    pub value: String,
}

impl RuntimeReplyProviderRequestHeader {
    pub fn is_responses_lite_header(&self) -> bool {
        self.name
            .eq_ignore_ascii_case(RuntimeReplyProviderRequestWireShape::RESPONSES_LITE_HEADER_NAME)
            && self.value == RuntimeReplyProviderRequestWireShape::RESPONSES_LITE_HEADER_VALUE
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeReplyResponsesPolicy {
    pub use_responses_lite: bool,
    pub request_mode: String,
    pub instructions_location: String,
    pub tools_location: String,
    pub reasoning_context: String,
    pub parallel_tool_calls_allowed: bool,
    pub requires_responses_lite_header: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeReplyToolCallPolicy {
    pub supports_parallel_tool_calls: bool,
    pub parallel_tool_calls: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeReplyReasoningOutputPolicy {
    pub default_reasoning_summary: String,
    pub support_verbosity: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_verbosity: Option<String>,
    pub can_set_verbosity: bool,
}

impl RuntimeReplyReasoningOutputPolicy {
    pub fn reasoning_summary(&self) -> Option<&str> {
        let summary = self.default_reasoning_summary.trim();
        (!summary.is_empty() && summary != "none").then_some(summary)
    }

    pub fn text_verbosity(&self) -> Option<&str> {
        if !(self.support_verbosity && self.can_set_verbosity) {
            return None;
        }
        self.default_verbosity
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
    }
}

#[cfg(test)]
mod sampling_output_tests;
#[cfg(test)]
mod tests;
