//! 当前 provider client。
//!
//! 该模块是固定模型回合的唯一网络边界。它以 Lime 自己的 message/tool contract
//! lower 到 OpenAI Chat Completions、OpenAI Responses 或 Anthropic Messages，并将
//! SSE 重新物化为统一的 response event。这里不依赖 Agent provider、Message 或
//! session 类型。

use crate::provider_stream::{
    RuntimeProviderBackend, RuntimeReplyModelRequestPolicy, RuntimeReplyProviderCapabilities,
    RuntimeReplyProviderHandle, RuntimeReplyProviderRequestWireShape,
};
use crate::runtime_provider::{
    RuntimeProviderAuth, RuntimeProviderConfig, RuntimeProviderProtocol,
};
use crate::ModelProviderProtocol;
use agent_protocol::ImageDetail;
use async_stream::stream;
use futures::future::BoxFuture;
use futures::Stream;
use reqwest::{Client, Response, StatusCode};
pub use runtime_core::{
    CanonicalLlmEvent, FailureClassification, FinishReason, FreeformToolFormat, GenerationOptions,
    ModelRerouteReason, ModelVerification, ProviderMetadata, ToolResultValue, Usage,
};
use runtime_core::{CanonicalRequest, CanonicalRole, CanonicalToolDefinition, ContentPart};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::{HeaderName, HeaderValue};
use tokio_tungstenite::tungstenite::Error as WebSocketError;

#[cfg(test)]
mod azure_responses_tests;
#[cfg(test)]
mod code_mode_tests;
mod gemini;
#[cfg(test)]
mod gemini_tests;
mod health;
#[cfg(test)]
mod hosted_web_search_tests;
mod lowering;
#[cfg(test)]
mod ollama_responses_tests;
#[cfg(test)]
mod request_capture_tests;
mod stream;
#[cfg(test)]
mod stream_tests;
mod transport;
#[cfg(test)]
mod vertex_gemini_tests;
mod websocket;

#[cfg(test)]
use health::HealthConfig;
use health::{CircuitBreaker, CircuitOpen, CircuitPermit};
pub use health::{
    CurrentProviderHealthRegistry, CurrentProviderHealthSnapshot, CurrentProviderHealthState,
};
use lowering::{anthropic_request, chat_completions_request, responses_request};
use stream::{anthropic_sse, openai_chat_sse, responses_sse};
use transport::{
    observed_retry_delay, provider_retry_after, request_failure, request_retry_reason,
    server_disallows_retry, should_retry_stream_request_status, MAX_STREAM_REQUEST_ATTEMPTS,
};
use websocket::{responses_websocket, ResponsesSocket};

pub type CurrentProviderStream =
    Pin<Box<dyn Stream<Item = Result<CanonicalLlmEvent, CurrentProviderError>> + Send>>;

/// Turn executor 依赖的 current provider stream contract。
///
/// HTTP client 只是其中一个实现；运行时通过该窄接口消费统一 response event，测试和
/// 其他 current transport 不需要伪造 HTTP 或引入 provider-specific trait object。
pub trait CurrentProvider: Send + Sync {
    fn stream<'a>(
        &'a self,
        request: CurrentProviderRequest,
    ) -> BoxFuture<'a, Result<CurrentProviderStream, CurrentProviderError>>;
}

const SESSION_ID_METADATA_KEY: &str = "session_id";
const THREAD_ID_METADATA_KEY: &str = "thread_id";
const TURN_ID_METADATA_KEY: &str = "turn_id";
const FORKED_FROM_THREAD_ID_METADATA_KEY: &str = "forked_from_thread_id";
const REQUEST_KIND_METADATA_KEY: &str = "request_kind";
const X_CODEX_TURN_METADATA_KEY: &str = "x-codex-turn-metadata";
const RESERVED_TURN_METADATA_KEYS: &[&str] = &[
    SESSION_ID_METADATA_KEY,
    THREAD_ID_METADATA_KEY,
    TURN_ID_METADATA_KEY,
    FORKED_FROM_THREAD_ID_METADATA_KEY,
    REQUEST_KIND_METADATA_KEY,
    X_CODEX_TURN_METADATA_KEY,
];

#[derive(Clone, Debug, PartialEq)]
pub struct CurrentProviderRequestMetadata {
    pub session_id: String,
    pub thread_id: String,
    pub turn_id: String,
    pub forked_from_thread_id: Option<String>,
    extra: ProviderMetadata,
}

impl CurrentProviderRequestMetadata {
    pub fn new(
        session_id: impl Into<String>,
        thread_id: impl Into<String>,
        turn_id: impl Into<String>,
        forked_from_thread_id: Option<String>,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            thread_id: thread_id.into(),
            turn_id: turn_id.into(),
            forked_from_thread_id,
            extra: ProviderMetadata::new(),
        }
    }

    pub fn with_extra(mut self, extra: ProviderMetadata) -> Self {
        self.extra = extra;
        self
    }

    fn canonical_metadata(&self) -> ProviderMetadata {
        let mut metadata = self
            .extra
            .iter()
            .filter(|(key, value)| {
                !RESERVED_TURN_METADATA_KEYS.contains(&key.as_str()) && value.is_string()
            })
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect::<ProviderMetadata>();
        metadata.insert(
            SESSION_ID_METADATA_KEY.to_string(),
            Value::String(self.session_id.clone()),
        );
        metadata.insert(
            THREAD_ID_METADATA_KEY.to_string(),
            Value::String(self.thread_id.clone()),
        );
        metadata.insert(
            TURN_ID_METADATA_KEY.to_string(),
            Value::String(self.turn_id.clone()),
        );
        if let Some(forked_from_thread_id) = &self.forked_from_thread_id {
            metadata.insert(
                FORKED_FROM_THREAD_ID_METADATA_KEY.to_string(),
                Value::String(forked_from_thread_id.clone()),
            );
        }
        metadata
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CurrentProviderRequest {
    pub system_prompt: Option<String>,
    pub messages: Vec<CurrentProviderMessage>,
    pub tools: Vec<CurrentProviderTool>,
    pub generation: GenerationOptions,
    pub provider_options: ProviderMetadata,
    pub metadata: Option<CurrentProviderRequestMetadata>,
    pub model_request_policy: Option<RuntimeReplyModelRequestPolicy>,
}

impl CurrentProviderRequest {
    pub fn new(messages: Vec<CurrentProviderMessage>) -> Self {
        Self {
            system_prompt: None,
            messages,
            tools: Vec::new(),
            generation: GenerationOptions::default(),
            provider_options: ProviderMetadata::new(),
            metadata: None,
            model_request_policy: None,
        }
    }

    pub fn with_system_prompt(mut self, system_prompt: Option<String>) -> Self {
        self.system_prompt = system_prompt;
        self
    }

    pub fn with_tools(mut self, tools: Vec<CurrentProviderTool>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_generation(mut self, generation: GenerationOptions) -> Self {
        self.generation = generation;
        self
    }

    pub fn with_provider_options(mut self, provider_options: ProviderMetadata) -> Self {
        self.provider_options = provider_options;
        self
    }

    pub fn with_metadata(mut self, metadata: CurrentProviderRequestMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn with_model_request_policy(
        mut self,
        model_request_policy: Option<RuntimeReplyModelRequestPolicy>,
    ) -> Self {
        self.model_request_policy = model_request_policy;
        self
    }

    /// 将回合边界的历史消息转换为唯一的 provider-neutral request contract。
    ///
    /// `CurrentProviderMessage` 仍由上层 transcript 使用，模型名由 current client
    /// 的 route config 注入；wire lowering 不再读取这些旧消息结构。
    pub(crate) fn into_canonical(
        &self,
        model: impl Into<String>,
    ) -> Result<CanonicalRequest, CurrentProviderError> {
        let mut canonical = CanonicalRequest::text(model, "");
        canonical.messages.clear();
        canonical.system = self
            .system_prompt
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .map(|value| vec![ContentPart::text(value)])
            .unwrap_or_default();
        canonical.messages = self
            .messages
            .iter()
            .map(canonical_message)
            .collect::<Result<Vec<_>, _>>()?;
        canonical.tools = self
            .tools
            .iter()
            .map(|tool| match tool {
                CurrentProviderTool::Function {
                    name,
                    description,
                    input_schema,
                } => CanonicalToolDefinition::function(
                    name.clone(),
                    description.clone(),
                    input_schema.clone(),
                ),
                CurrentProviderTool::Custom {
                    name,
                    description,
                    format,
                } => CanonicalToolDefinition::custom(
                    name.clone(),
                    description.clone(),
                    format.clone(),
                ),
            })
            .collect();
        canonical.generation = self.generation.clone();
        canonical.provider_options = self.provider_options.clone();
        if let Some(metadata) = &self.metadata {
            canonical.metadata = metadata.canonical_metadata();
        }
        Ok(canonical)
    }

    fn media_payloads(&self) -> Result<BTreeMap<String, String>, CurrentProviderError> {
        let mut payloads = BTreeMap::new();
        for content in self.messages.iter().flat_map(|message| &message.content) {
            let CurrentProviderContent::Image {
                uri, provider_data, ..
            } = content
            else {
                continue;
            };
            if let Some(provider_data) = provider_data {
                if let Some(previous) = payloads.insert(uri.clone(), provider_data.clone()) {
                    if previous != *provider_data {
                        return Err(CurrentProviderError::invalid_request(format!(
                            "canonical media reference {uri} maps to conflicting provider payloads"
                        )));
                    }
                }
                continue;
            }
            if is_local_media_reference(uri) {
                return Err(CurrentProviderError::invalid_request(format!(
                    "canonical media reference {uri} has no provider-readable payload"
                )));
            }
        }
        Ok(payloads)
    }

    fn contains_raw_response_items(&self) -> bool {
        self.messages.iter().any(|message| {
            message
                .content
                .iter()
                .any(|content| matches!(content, CurrentProviderContent::RawResponseItem(_)))
        })
    }
}

fn canonical_message(
    message: &CurrentProviderMessage,
) -> Result<runtime_core::CanonicalMessage, CurrentProviderError> {
    let role = match message.role {
        CurrentProviderRole::User => CanonicalRole::User,
        CurrentProviderRole::Developer => CanonicalRole::Developer,
        CurrentProviderRole::Assistant => CanonicalRole::Assistant,
        CurrentProviderRole::Tool => CanonicalRole::Tool,
    };
    let content = message
        .content
        .iter()
        .map(canonical_content)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(runtime_core::CanonicalMessage {
        id: None,
        role,
        content,
        metadata: Default::default(),
    })
}

fn canonical_content(
    content: &CurrentProviderContent,
) -> Result<ContentPart, CurrentProviderError> {
    match content {
        CurrentProviderContent::Text(text) => Ok(ContentPart::text(text)),
        CurrentProviderContent::Reasoning(text) => Ok(ContentPart::Reasoning {
            text: text.clone(),
            encrypted: None,
            metadata: Default::default(),
        }),
        CurrentProviderContent::Image {
            uri,
            media_type,
            detail,
            ..
        } => ContentPart::media_with_detail(uri.clone(), media_type.clone(), *detail).map_err(
            |error| CurrentProviderError::new(format!("canonical media input rejected: {error}")),
        ),
        CurrentProviderContent::ToolCall(call) => Ok(ContentPart::ToolCall {
            id: call.id.clone(),
            name: call.name.clone(),
            input: call.arguments.clone(),
            provider_executed: None,
            metadata: call.provider_metadata.clone(),
        }),
        CurrentProviderContent::CustomToolCall(call) => Ok(ContentPart::CustomToolCall {
            id: call.id.clone(),
            name: call.name.clone(),
            input: call.input.clone(),
            namespace: call.namespace.clone(),
            metadata: call.provider_metadata.clone(),
        }),
        CurrentProviderContent::ToolResult(result) => Ok(ContentPart::ToolResult {
            id: result.call_id.clone(),
            name: result.name.clone(),
            result: if result.success {
                ToolResultValue::text(result.output.clone())
            } else {
                ToolResultValue::Error {
                    value: serde_json::json!({
                        "output": result.output,
                        "error": result.error,
                    }),
                }
            },
            error: result.error.clone(),
            provider_executed: Some(false),
            metadata: Default::default(),
        }),
        CurrentProviderContent::CustomToolResult(result) => Ok(ContentPart::CustomToolResult {
            id: result.call_id.clone(),
            name: result.name.clone(),
            result: if result.success {
                ToolResultValue::text(result.output.clone())
            } else {
                ToolResultValue::Error {
                    value: serde_json::json!({
                        "output": result.output,
                        "error": result.error,
                    }),
                }
            },
            error: result.error.clone(),
            metadata: Default::default(),
        }),
        CurrentProviderContent::RawResponseItem(item) => {
            Ok(ContentPart::RawResponseItem { item: item.clone() })
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CurrentProviderMessage {
    pub role: CurrentProviderRole,
    pub content: Vec<CurrentProviderContent>,
}

impl CurrentProviderMessage {
    pub fn user(content: Vec<CurrentProviderContent>) -> Self {
        Self {
            role: CurrentProviderRole::User,
            content,
        }
    }

    pub fn assistant(content: Vec<CurrentProviderContent>) -> Self {
        Self {
            role: CurrentProviderRole::Assistant,
            content,
        }
    }

    pub fn developer(content: Vec<CurrentProviderContent>) -> Self {
        Self {
            role: CurrentProviderRole::Developer,
            content,
        }
    }

    pub fn tool(content: Vec<CurrentProviderContent>) -> Self {
        Self {
            role: CurrentProviderRole::Tool,
            content,
        }
    }

    pub fn raw_response_item(item: Value) -> Self {
        Self::assistant(vec![CurrentProviderContent::RawResponseItem(item)])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurrentProviderRole {
    User,
    Developer,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CurrentProviderContent {
    Text(String),
    Reasoning(String),
    Image {
        uri: String,
        media_type: String,
        provider_data: Option<String>,
        detail: Option<ImageDetail>,
    },
    ToolCall(CurrentProviderToolCall),
    CustomToolCall(CurrentProviderCustomToolCall),
    ToolResult(CurrentProviderToolResult),
    CustomToolResult(CurrentProviderToolResult),
    RawResponseItem(Value),
}

#[derive(Clone, Debug, PartialEq)]
pub struct CurrentProviderToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
    pub raw_arguments: String,
    pub provider_metadata: ProviderMetadata,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CurrentProviderCustomToolCall {
    pub id: String,
    pub name: String,
    pub input: String,
    pub namespace: Option<String>,
    pub provider_metadata: ProviderMetadata,
}

impl CurrentProviderCustomToolCall {
    pub fn new(id: impl Into<String>, name: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            input: input.into(),
            namespace: None,
            provider_metadata: ProviderMetadata::new(),
        }
    }
}

impl CurrentProviderToolCall {
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: Value) -> Self {
        let raw_arguments = serde_json::to_string(&arguments).unwrap_or_else(|_| "{}".to_string());
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
            raw_arguments,
            provider_metadata: ProviderMetadata::new(),
        }
    }

    pub fn from_raw(id: String, name: String, raw_arguments: String) -> Self {
        let arguments = serde_json::from_str(&raw_arguments)
            .unwrap_or_else(|_| Value::String(raw_arguments.clone()));
        Self {
            id,
            name: name.trim().to_string(),
            arguments,
            raw_arguments,
            provider_metadata: ProviderMetadata::new(),
        }
    }

    pub fn with_provider_metadata(mut self, provider_metadata: ProviderMetadata) -> Self {
        self.provider_metadata = provider_metadata;
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CurrentProviderToolResult {
    pub call_id: String,
    pub name: String,
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CurrentProviderTool {
    Function {
        name: String,
        description: String,
        input_schema: Value,
    },
    Custom {
        name: String,
        description: String,
        format: FreeformToolFormat,
    },
}

impl CurrentProviderTool {
    pub fn name(&self) -> &str {
        match self {
            Self::Function { name, .. } | Self::Custom { name, .. } => name,
        }
    }

    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
    ) -> Self {
        Self::Function {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }

    pub fn custom(
        name: impl Into<String>,
        description: impl Into<String>,
        format: FreeformToolFormat,
    ) -> Self {
        Self::Custom {
            name: name.into(),
            description: description.into(),
            format,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CurrentProviderUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cached_input_tokens: Option<u32>,
    pub cache_creation_input_tokens: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentProviderError {
    pub message: String,
    pub status: Option<u16>,
    pub classification: Option<FailureClassification>,
    pub retryable: bool,
    pub retry_after: Option<Duration>,
}

impl CurrentProviderError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            status: None,
            classification: None,
            retryable: false,
            retry_after: None,
        }
    }

    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            status: None,
            classification: Some(FailureClassification::InvalidRequest),
            retryable: false,
            retry_after: None,
        }
    }

    fn with_status(
        status: StatusCode,
        message: impl Into<String>,
        retry_after: Option<Duration>,
    ) -> Self {
        Self {
            message: message.into(),
            status: Some(status.as_u16()),
            classification: Some(classification_from_status(status)),
            retryable: status_failure_is_retryable(status),
            retry_after,
        }
    }

    pub(super) fn transport(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            status: None,
            classification: Some(FailureClassification::Transport),
            retryable: true,
            retry_after: None,
        }
    }
}

impl fmt::Display for CurrentProviderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for CurrentProviderError {}

#[derive(Clone)]
pub struct CurrentProviderClient {
    config: RuntimeProviderConfig,
    client: Client,
    health: Arc<CircuitBreaker>,
    http_fallback: Arc<AtomicBool>,
    websocket: Arc<Mutex<Option<ResponsesSocket>>>,
    websocket_server_model: Arc<Mutex<Option<String>>>,
}

impl CurrentProviderClient {
    pub fn new(config: RuntimeProviderConfig) -> Result<Self, CurrentProviderError> {
        Self::new_with_health_registry(config, &CurrentProviderHealthRegistry::new())
    }

    pub fn new_with_health_registry(
        config: RuntimeProviderConfig,
        health_registry: &CurrentProviderHealthRegistry,
    ) -> Result<Self, CurrentProviderError> {
        runtime_protocol(&config)?;
        let mut client_builder = Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .tcp_keepalive(Duration::from_secs(60))
            .gzip(true)
            .brotli(true)
            .deflate(true);
        if config
            .base_url
            .as_deref()
            .is_some_and(crate::http::should_bypass_system_proxy)
        {
            client_builder = client_builder.no_proxy();
        }
        let client = client_builder.build().map_err(|error| {
            CurrentProviderError::new(format!("创建 provider HTTP client 失败: {error}"))
        })?;
        Ok(Self {
            health: health_registry.circuit_for(&config),
            config,
            client,
            http_fallback: Arc::new(AtomicBool::new(false)),
            websocket: Arc::new(Mutex::new(None)),
            websocket_server_model: Arc::new(Mutex::new(None)),
        })
    }

    pub fn with_client(config: RuntimeProviderConfig, client: Client) -> Self {
        Self::with_client_and_health_registry(config, client, &CurrentProviderHealthRegistry::new())
    }

    fn with_client_and_health_registry(
        config: RuntimeProviderConfig,
        client: Client,
        health_registry: &CurrentProviderHealthRegistry,
    ) -> Self {
        Self {
            health: health_registry.circuit_for(&config),
            config,
            client,
            http_fallback: Arc::new(AtomicBool::new(false)),
            websocket: Arc::new(Mutex::new(None)),
            websocket_server_model: Arc::new(Mutex::new(None)),
        }
    }

    pub fn config(&self) -> &RuntimeProviderConfig {
        &self.config
    }

    pub fn runtime_handle(&self) -> RuntimeReplyProviderHandle {
        RuntimeReplyProviderHandle::from_config(&self.config, RuntimeProviderBackend::Current)
            .with_capabilities(RuntimeReplyProviderCapabilities {
                supports_streaming: true,
                supports_embeddings: false,
                active_model_name: Some(self.config.model_name.clone()),
            })
    }

    pub fn protocol(&self) -> Result<ModelProviderProtocol, CurrentProviderError> {
        runtime_protocol(&self.config)
    }

    pub async fn stream(
        &self,
        request: CurrentProviderRequest,
    ) -> Result<CurrentProviderStream, CurrentProviderError> {
        let protocol = self.protocol()?;
        ensure_supported_protocol(&protocol)?;
        if request.contains_raw_response_items()
            && !matches!(protocol, ModelProviderProtocol::Responses)
        {
            return Err(CurrentProviderError::invalid_request(
                "raw Responses API history items require a Responses provider route",
            ));
        }
        let mut permit = self.health.acquire().map_err(circuit_open_error)?;
        let media_payloads = request.media_payloads()?;
        let canonical_request = request.into_canonical(&self.config.model_name)?;
        let wire_shape = RuntimeReplyProviderRequestWireShape::from_model_request_policy(
            request.model_request_policy.as_ref(),
        );
        let payload = match protocol {
            ModelProviderProtocol::Responses => responses_request(
                &self.config,
                &canonical_request,
                &wire_shape,
                &media_payloads,
            ),
            ModelProviderProtocol::AnthropicMessages => {
                anthropic_request(&self.config, &canonical_request, &media_payloads)
            }
            ModelProviderProtocol::GeminiGenerateContent => {
                gemini::request(&canonical_request, &media_payloads)
            }
            ModelProviderProtocol::ChatCompletions | ModelProviderProtocol::Custom(_) => {
                chat_completions_request(
                    &self.config,
                    &canonical_request,
                    &wire_shape,
                    &media_payloads,
                )
            }
        }?;
        if matches!(protocol, ModelProviderProtocol::Responses)
            && self.responses_websocket_enabled()
        {
            match self
                .send_responses_websocket(payload.clone(), &wire_shape)
                .await
            {
                Ok(stream) => {
                    return Ok(self.tracked_stream(
                        self.websocket_with_http_fallback(stream, payload, wire_shape),
                        permit,
                    ));
                }
                Err(error) => {
                    self.http_fallback.store(true, Ordering::Release);
                    tracing::warn!(
                        error = %error,
                        "Responses WebSocket unavailable; falling back to HTTP for this session"
                    );
                }
            }
        }
        let response = match self
            .send_stream_request(&protocol, payload, &wire_shape)
            .await
        {
            Ok(response) => response,
            Err(error) => {
                if request_error_is_health_failure(&error) {
                    permit.failure();
                } else {
                    permit.ignore();
                }
                return Err(error);
            }
        };
        let stream: CurrentProviderStream = match protocol {
            ModelProviderProtocol::Responses => Box::pin(responses_sse(
                response,
                self.trusts_openai_response_metadata(),
            )),
            ModelProviderProtocol::AnthropicMessages => Box::pin(anthropic_sse(response)),
            ModelProviderProtocol::GeminiGenerateContent => Box::pin(gemini::stream(response)),
            ModelProviderProtocol::ChatCompletions | ModelProviderProtocol::Custom(_) => {
                Box::pin(openai_chat_sse(response))
            }
        };
        Ok(self.tracked_stream(stream, permit))
    }

    fn tracked_stream(
        &self,
        mut stream: CurrentProviderStream,
        permit: CircuitPermit,
    ) -> CurrentProviderStream {
        let requested_model = self.config.model_name.clone();
        let trust_server_model = self.trusts_openai_response_metadata();
        Box::pin(stream! {
            let mut permit = Some(permit);
            let mut model_reroute_emitted = false;
            while let Some(item) = futures::StreamExt::next(&mut stream).await {
                match &item {
                    Ok(CanonicalLlmEvent::Finish { .. }) => {
                        if let Some(mut permit) = permit.take() {
                            permit.success();
                        }
                    }
                    Ok(CanonicalLlmEvent::ProviderError {
                        classification,
                        retryable,
                        ..
                    }) => {
                        if let Some(mut permit) = permit.take() {
                            if stream_event_is_health_failure(*classification, *retryable) {
                                permit.failure();
                            } else {
                                permit.ignore();
                            }
                        }
                    }
                    Err(error) => {
                        if let Some(mut permit) = permit.take() {
                            if stream_error_is_health_failure(error) {
                                permit.failure();
                            } else {
                                permit.ignore();
                            }
                        }
                    }
                    Ok(_) => {}
                }
                let reroute = match &item {
                    Ok(CanonicalLlmEvent::ServerModel { model })
                        if trust_server_model
                            && !model_reroute_emitted
                            && !requested_model.eq_ignore_ascii_case(model) =>
                    {
                        model_reroute_emitted = true;
                        Some(CanonicalLlmEvent::ModelReroute {
                            from_model: requested_model.clone(),
                            to_model: model.clone(),
                            reason: ModelRerouteReason::HighRiskCyberActivity,
                        })
                    }
                    _ => None,
                };
                yield item;
                if let Some(reroute) = reroute {
                    yield Ok(reroute);
                }
            }
            if let Some(mut permit) = permit {
                permit.failure();
            }
        })
    }

    pub fn responses_websocket_enabled(&self) -> bool {
        !matches!(
            self.config.protocol,
            Some(RuntimeProviderProtocol::AzureResponses | RuntimeProviderProtocol::VertexGemini)
        ) && self.config.supports_websockets
            && !self.http_fallback.load(Ordering::Acquire)
    }

    fn websocket_with_http_fallback(
        &self,
        mut websocket: CurrentProviderStream,
        payload: Value,
        wire_shape: RuntimeReplyProviderRequestWireShape,
    ) -> CurrentProviderStream {
        let client = self.clone();
        Box::pin(stream! {
            let mut emitted_replay_sensitive_event = false;
            while let Some(event) = futures::StreamExt::next(&mut websocket).await {
                match event {
                    Ok(event) => {
                        emitted_replay_sensitive_event |= !matches!(
                            &event,
                            CanonicalLlmEvent::ServerModel { .. }
                                | CanonicalLlmEvent::ModelReroute { .. }
                                | CanonicalLlmEvent::ModelVerification { .. }
                                | CanonicalLlmEvent::TurnModerationMetadata { .. }
                        );
                        yield Ok(event);
                    }
                    Err(error) if !emitted_replay_sensitive_event => {
                        client.http_fallback.store(true, Ordering::Release);
                        tracing::warn!(
                            error = %error,
                            "Responses WebSocket ended before visible output; replaying over HTTP"
                        );
                        match client
                            .send_stream_request(
                                &ModelProviderProtocol::Responses,
                                payload,
                                &wire_shape,
                            )
                            .await
                        {
                            Ok(response) => {
                                let mut http = Box::pin(responses_sse(
                                    response,
                                    client.trusts_openai_response_metadata(),
                                ));
                                while let Some(event) = futures::StreamExt::next(&mut http).await {
                                    yield event;
                                }
                            }
                            Err(http_error) => yield Err(http_error),
                        }
                        return;
                    }
                    Err(error) => {
                        yield Err(error);
                        return;
                    }
                }
            }
        })
    }

    async fn send_responses_websocket(
        &self,
        payload: Value,
        wire_shape: &RuntimeReplyProviderRequestWireShape,
    ) -> Result<CurrentProviderStream, CurrentProviderError> {
        let api_key = self.request_api_key()?;
        let url = responses_websocket_url(self.config.base_url.as_deref())?;
        let mut attempts = 0;
        let mut websocket = self.websocket.lock().await;

        while websocket.is_none() {
            attempts += 1;
            let mut request = url.as_str().into_client_request().map_err(|error| {
                CurrentProviderError::invalid_request(format!(
                    "创建 Responses WebSocket request 失败: {error}"
                ))
            })?;
            if let Some(api_key) = api_key {
                request.headers_mut().insert(
                    "Authorization",
                    HeaderValue::from_str(&format!("Bearer {api_key}")).map_err(|error| {
                        CurrentProviderError::invalid_request(format!(
                            "Responses WebSocket authorization header 无效: {error}"
                        ))
                    })?,
                );
            }
            request.headers_mut().insert(
                "OpenAI-Beta",
                HeaderValue::from_static("responses_websockets=2026-02-06"),
            );
            for header in &wire_shape.headers {
                let name = header.name.parse::<HeaderName>().map_err(|error| {
                    CurrentProviderError::invalid_request(format!(
                        "Responses WebSocket header name 无效 ({}): {error}",
                        header.name
                    ))
                })?;
                let value = HeaderValue::from_str(&header.value).map_err(|error| {
                    CurrentProviderError::invalid_request(format!(
                        "Responses WebSocket header value 无效 ({}): {error}",
                        header.name
                    ))
                })?;
                request.headers_mut().insert(name, value);
            }

            match tokio_tungstenite::connect_async(request).await {
                Ok((socket, response)) => {
                    let server_model = ["openai-model", "x-openai-model"]
                        .into_iter()
                        .find_map(|name| response.headers().get(name))
                        .and_then(|value| value.to_str().ok())
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_string);
                    *self.websocket_server_model.lock().await = server_model;
                    *websocket = Some(socket);
                }
                Err(error)
                    if websocket_error_status(&error) == Some(StatusCode::UPGRADE_REQUIRED) =>
                {
                    return Err(websocket_connect_error(error));
                }
                Err(error) if websocket_error_disallows_retry(&error) => {
                    return Err(websocket_connect_error(error));
                }
                Err(error) if attempts < MAX_STREAM_REQUEST_ATTEMPTS => {
                    let delay = observed_retry_delay(
                        &self.health,
                        &reqwest::header::HeaderMap::new(),
                        attempts,
                        "websocket",
                        "connect_error",
                        websocket_error_status(&error).map(|status| status.as_u16()),
                    );
                    tokio::time::sleep(delay).await;
                }
                Err(error) => return Err(websocket_connect_error(error)),
            }
        }
        drop(websocket);
        let server_model = self.websocket_server_model.lock().await.clone();
        Ok(responses_websocket(
            Arc::clone(&self.websocket),
            payload,
            Arc::clone(&self.http_fallback),
            server_model,
            self.trusts_openai_response_metadata(),
        ))
    }

    async fn send_stream_request(
        &self,
        protocol: &ModelProviderProtocol,
        payload: Value,
        wire_shape: &RuntimeReplyProviderRequestWireShape,
    ) -> Result<Response, CurrentProviderError> {
        let api_key = self.request_api_key()?;
        let urls = match self.config.protocol {
            Some(RuntimeProviderProtocol::AzureResponses) => vec![azure_responses_endpoint(
                self.config.base_url.as_deref().ok_or_else(|| {
                    CurrentProviderError::invalid_request(
                        "Azure OpenAI provider requires a resource base URL",
                    )
                })?,
                self.config.api_version.as_deref(),
            )?],
            Some(RuntimeProviderProtocol::VertexGemini) => vec![gemini::endpoint(
                Some(self.config.base_url.as_deref().ok_or_else(|| {
                    CurrentProviderError::invalid_request(
                        "Vertex Gemini provider requires a resolved project endpoint",
                    )
                })?),
                &self.config.model_name,
            )],
            _ => provider_urls(
                protocol,
                self.config.base_url.as_deref(),
                Some(&self.config.model_name),
            ),
        };
        let mut last_response = None;
        let mut attempts = 0;

        for url in urls {
            while attempts < MAX_STREAM_REQUEST_ATTEMPTS {
                attempts += 1;
                let mut request = self
                    .client
                    .post(&url)
                    .header("Content-Type", "application/json")
                    .header("Accept", "text/event-stream")
                    .json(&payload);
                if matches!(protocol, ModelProviderProtocol::AnthropicMessages) {
                    request = request.header("anthropic-version", "2023-06-01");
                }
                if let Some(api_key) = api_key {
                    request = match protocol {
                        ModelProviderProtocol::AnthropicMessages => {
                            request.header("x-api-key", api_key)
                        }
                        _ if self.config.protocol
                            == Some(RuntimeProviderProtocol::VertexGemini) =>
                        {
                            request.header("Authorization", format!("Bearer {api_key}"))
                        }
                        ModelProviderProtocol::GeminiGenerateContent => {
                            request.header("x-goog-api-key", api_key)
                        }
                        _ if self.config.protocol
                            == Some(RuntimeProviderProtocol::AzureResponses) =>
                        {
                            request.header("api-key", api_key)
                        }
                        _ => request.header("Authorization", format!("Bearer {api_key}")),
                    };
                }
                for header in &wire_shape.headers {
                    request = request.header(&header.name, &header.value);
                }
                let response = match request.send().await {
                    Ok(response) => response,
                    Err(error) if attempts < MAX_STREAM_REQUEST_ATTEMPTS => {
                        let reason = request_retry_reason(&error);
                        let delay = observed_retry_delay(
                            &self.health,
                            &reqwest::header::HeaderMap::new(),
                            attempts,
                            "http",
                            reason,
                            None,
                        );
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                    Err(error) => return Err(request_failure(error)),
                };
                if response.status() == StatusCode::NOT_FOUND {
                    last_response = Some(response);
                    break;
                }
                if should_retry_stream_request_status(response.status())
                    && !server_disallows_retry(response.headers())
                    && attempts < MAX_STREAM_REQUEST_ATTEMPTS
                {
                    let delay = observed_retry_delay(
                        &self.health,
                        response.headers(),
                        attempts,
                        "http",
                        "server_error",
                        Some(response.status().as_u16()),
                    );
                    drop(response);
                    tokio::time::sleep(delay).await;
                    continue;
                }
                return ensure_success_response(response).await;
            }
        }

        let response =
            last_response.ok_or_else(|| CurrentProviderError::new("Provider 未生成请求地址"))?;
        ensure_success_response(response).await
    }

    fn request_api_key(&self) -> Result<Option<&str>, CurrentProviderError> {
        match self.config.auth {
            RuntimeProviderAuth::NoAuth
                if self.config.protocol == Some(RuntimeProviderProtocol::AzureResponses) =>
            {
                return Err(CurrentProviderError::invalid_request(
                    "Azure OpenAI Responses requires API-key authentication",
                ));
            }
            RuntimeProviderAuth::NoAuth
                if self.config.protocol == Some(RuntimeProviderProtocol::VertexGemini) =>
            {
                return Err(CurrentProviderError::invalid_request(
                    "Vertex Gemini requires Bearer access-token authentication",
                ));
            }
            RuntimeProviderAuth::NoAuth => return Ok(None),
            RuntimeProviderAuth::OemManaged => {
                return Err(CurrentProviderError::invalid_request(
                    "OEM-managed authentication has no current model-provider adapter",
                ));
            }
            RuntimeProviderAuth::ApiKey => {}
        }
        self.config
            .api_key
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(Some)
            .ok_or_else(|| CurrentProviderError::new("Provider API key 未配置"))
    }

    fn trusts_openai_response_metadata(&self) -> bool {
        if self.config.protocol != Some(RuntimeProviderProtocol::Responses) {
            return false;
        }
        let provider = self
            .config
            .provider_name
            .trim()
            .to_ascii_lowercase()
            .replace(['-', ' '], "_");
        if !matches!(provider.as_str(), "openai" | "codex") {
            return false;
        }
        self.config.base_url.as_deref().map_or(true, |base_url| {
            url::Url::parse(base_url)
                .ok()
                .and_then(|url| url.host_str().map(str::to_string))
                .is_some_and(|host| host.eq_ignore_ascii_case("api.openai.com"))
        })
    }
}

impl CurrentProvider for CurrentProviderClient {
    fn stream<'a>(
        &'a self,
        request: CurrentProviderRequest,
    ) -> BoxFuture<'a, Result<CurrentProviderStream, CurrentProviderError>> {
        Box::pin(async move { CurrentProviderClient::stream(self, request).await })
    }
}

fn circuit_open_error(error: CircuitOpen) -> CurrentProviderError {
    CurrentProviderError::transport(error.to_string())
}

fn request_error_is_health_failure(error: &CurrentProviderError) -> bool {
    health_failure(error.classification, Some(error.retryable), false)
}

fn stream_error_is_health_failure(error: &CurrentProviderError) -> bool {
    health_failure(error.classification, Some(error.retryable), true)
}

fn stream_event_is_health_failure(
    classification: Option<FailureClassification>,
    retryable: Option<bool>,
) -> bool {
    health_failure(classification, retryable, true)
}

fn health_failure(
    classification: Option<FailureClassification>,
    retryable: Option<bool>,
    unknown_is_failure: bool,
) -> bool {
    if retryable == Some(false) {
        return false;
    }
    match classification {
        Some(
            FailureClassification::Authentication
            | FailureClassification::Permission
            | FailureClassification::Quota
            | FailureClassification::InvalidRequest
            | FailureClassification::ContextOverflow
            | FailureClassification::ContentPolicy,
        ) => false,
        Some(
            FailureClassification::RateLimit
            | FailureClassification::ProviderInternal
            | FailureClassification::Transport,
        ) => true,
        Some(FailureClassification::Unknown) => retryable.unwrap_or(unknown_is_failure),
        None => retryable.unwrap_or(unknown_is_failure),
    }
}

async fn ensure_success_response(response: Response) -> Result<Response, CurrentProviderError> {
    if response.status().is_success() {
        return Ok(response);
    }
    let status = response.status();
    let retry_after = provider_retry_after(response.headers());
    let mut error = CurrentProviderError::with_status(
        status,
        format!("Provider 请求失败 ({status})"),
        retry_after,
    );
    if server_disallows_retry(response.headers()) {
        error.retryable = false;
    }
    Err(error)
}

fn classification_from_status(status: StatusCode) -> FailureClassification {
    match status {
        StatusCode::UNAUTHORIZED => FailureClassification::Authentication,
        StatusCode::FORBIDDEN => FailureClassification::Permission,
        StatusCode::PAYMENT_REQUIRED => FailureClassification::Quota,
        StatusCode::TOO_MANY_REQUESTS => FailureClassification::RateLimit,
        StatusCode::PAYLOAD_TOO_LARGE => FailureClassification::ContextOverflow,
        status if status.is_server_error() => FailureClassification::ProviderInternal,
        status if status.is_client_error() => FailureClassification::InvalidRequest,
        _ => FailureClassification::Unknown,
    }
}

fn status_failure_is_retryable(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

fn is_local_media_reference(uri: &str) -> bool {
    let uri = uri.trim().to_ascii_lowercase();
    uri.starts_with("sidecar://") || uri.starts_with("asset://") || uri.starts_with("file://")
}

fn runtime_protocol(
    config: &RuntimeProviderConfig,
) -> Result<ModelProviderProtocol, CurrentProviderError> {
    match config.protocol {
        Some(RuntimeProviderProtocol::Responses) => Ok(ModelProviderProtocol::Responses),
        Some(RuntimeProviderProtocol::AzureResponses) => Ok(ModelProviderProtocol::Responses),
        Some(RuntimeProviderProtocol::AnthropicMessages) => {
            Ok(ModelProviderProtocol::AnthropicMessages)
        }
        Some(RuntimeProviderProtocol::ChatCompletions) => {
            Ok(ModelProviderProtocol::ChatCompletions)
        }
        Some(RuntimeProviderProtocol::GeminiGenerateContent) => {
            Ok(ModelProviderProtocol::GeminiGenerateContent)
        }
        Some(RuntimeProviderProtocol::VertexGemini) => {
            Ok(ModelProviderProtocol::GeminiGenerateContent)
        }
        None => Err(CurrentProviderError::invalid_request(format!(
            "provider route for `{}` is missing an explicit protocol",
            config
                .provider_selector
                .as_deref()
                .unwrap_or_else(|| config.provider_name.as_str())
        ))),
    }
}

fn responses_websocket_url(base_url: Option<&str>) -> Result<url::Url, CurrentProviderError> {
    let http_url = provider_urls(&ModelProviderProtocol::Responses, base_url, None)
        .into_iter()
        .next()
        .ok_or_else(|| CurrentProviderError::invalid_request("Provider 未生成 Responses 地址"))?;
    let mut url = url::Url::parse(&http_url).map_err(|error| {
        CurrentProviderError::invalid_request(format!(
            "Responses WebSocket URL 无效 ({http_url}): {error}"
        ))
    })?;
    let scheme = match url.scheme() {
        "http" => "ws",
        "https" => "wss",
        "ws" => "ws",
        "wss" => "wss",
        other => {
            return Err(CurrentProviderError::invalid_request(format!(
                "Responses WebSocket 不支持 URL scheme: {other}"
            )))
        }
    };
    url.set_scheme(scheme).map_err(|_| {
        CurrentProviderError::invalid_request("Responses WebSocket URL scheme 转换失败")
    })?;
    Ok(url)
}

fn websocket_error_status(error: &WebSocketError) -> Option<StatusCode> {
    let WebSocketError::Http(response) = error else {
        return None;
    };
    StatusCode::from_u16(response.status().as_u16()).ok()
}

fn websocket_error_disallows_retry(error: &WebSocketError) -> bool {
    let WebSocketError::Http(response) = error else {
        return false;
    };
    server_disallows_retry(response.headers())
}

fn websocket_connect_error(error: WebSocketError) -> CurrentProviderError {
    let retryable = !websocket_error_disallows_retry(&error);
    if let Some(status) = websocket_error_status(&error) {
        let retry_after = match &error {
            WebSocketError::Http(response) => provider_retry_after(response.headers()),
            _ => None,
        };
        let mut error = CurrentProviderError::with_status(
            status,
            format!("Responses WebSocket upgrade 失败 ({status})"),
            retry_after,
        );
        error.retryable &= retryable;
        return error;
    }
    CurrentProviderError::transport("Responses WebSocket 连接失败")
}

fn ensure_supported_protocol(protocol: &ModelProviderProtocol) -> Result<(), CurrentProviderError> {
    if let ModelProviderProtocol::Custom(name) = protocol {
        return Err(CurrentProviderError::invalid_request(format!(
            "unsupported provider protocol `{name}`; no current model-provider wire adapter is registered"
        )));
    }
    Ok(())
}

fn provider_urls(
    protocol: &ModelProviderProtocol,
    base_url: Option<&str>,
    model: Option<&str>,
) -> Vec<String> {
    if matches!(protocol, ModelProviderProtocol::GeminiGenerateContent) {
        return vec![gemini::endpoint(base_url, model.unwrap_or_default())];
    }
    let base_url = base_url
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| match protocol {
            ModelProviderProtocol::AnthropicMessages => "https://api.anthropic.com",
            ModelProviderProtocol::GeminiGenerateContent => {
                "https://generativelanguage.googleapis.com/v1beta"
            }
            _ => "https://api.openai.com",
        });
    let endpoint = match protocol {
        ModelProviderProtocol::Responses => "responses",
        ModelProviderProtocol::AnthropicMessages => "messages",
        ModelProviderProtocol::GeminiGenerateContent => unreachable!("handled above"),
        ModelProviderProtocol::ChatCompletions | ModelProviderProtocol::Custom(_) => {
            "chat/completions"
        }
    };
    endpoint_urls(base_url, endpoint)
}

pub fn azure_responses_endpoint(
    base_url: &str,
    api_version: Option<&str>,
) -> Result<String, CurrentProviderError> {
    let base_url = base_url.trim();
    if base_url.is_empty() {
        return Err(CurrentProviderError::invalid_request(
            "Azure OpenAI provider requires a resource base URL",
        ));
    }
    let mut url = url::Url::parse(base_url).map_err(|error| {
        CurrentProviderError::invalid_request(format!(
            "Azure OpenAI resource base URL is invalid ({base_url}): {error}"
        ))
    })?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(CurrentProviderError::invalid_request(format!(
            "Azure OpenAI resource base URL uses unsupported scheme `{}`",
            url.scheme()
        )));
    }
    if url.host_str().is_none() {
        return Err(CurrentProviderError::invalid_request(
            "Azure OpenAI resource base URL requires a host",
        ));
    }
    url.set_fragment(None);

    let path = match url.path().trim_end_matches('/') {
        "" => "/openai/v1/responses",
        "/openai" => "/openai/v1/responses",
        "/openai/v1" => "/openai/v1/responses",
        "/openai/v1/responses" => "/openai/v1/responses",
        path => {
            return Err(CurrentProviderError::invalid_request(format!(
                "Azure OpenAI resource base URL has unsupported path `{path}`; expected resource root or /openai/v1"
            )))
        }
    };
    url.set_path(path);
    let existing_query = url
        .query_pairs()
        .filter(|(name, _)| !name.eq_ignore_ascii_case("api-version"))
        .map(|(name, value)| (name.into_owned(), value.into_owned()))
        .collect::<Vec<_>>();
    url.set_query(None);
    let api_version = api_version
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("v1");
    {
        let mut query = url.query_pairs_mut();
        for (name, value) in existing_query {
            query.append_pair(&name, &value);
        }
        query.append_pair("api-version", api_version);
    }
    Ok(url.to_string())
}

pub fn vertex_gemini_base_url(
    base_url: Option<&str>,
    project: Option<&str>,
    location: Option<&str>,
) -> Result<String, CurrentProviderError> {
    let project = required_vertex_context("project", project)?;
    let location = required_vertex_context("location", location)?;
    if !location
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
    {
        return Err(CurrentProviderError::invalid_request(
            "Vertex Gemini location contains unsupported characters",
        ));
    }
    let default_base = if location.eq_ignore_ascii_case("global") {
        "https://aiplatform.googleapis.com".to_string()
    } else {
        format!("https://{location}-aiplatform.googleapis.com")
    };
    let base_url = base_url
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(&default_base);
    let mut url = url::Url::parse(base_url).map_err(|error| {
        CurrentProviderError::invalid_request(format!(
            "Vertex Gemini base URL is invalid ({base_url}): {error}"
        ))
    })?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err(CurrentProviderError::invalid_request(
            "Vertex Gemini base URL requires an HTTP(S) host",
        ));
    }
    if url.query().is_some() || url.fragment().is_some() {
        return Err(CurrentProviderError::invalid_request(
            "Vertex Gemini base URL must not contain query or fragment components",
        ));
    }
    if !url.path().trim_matches('/').is_empty() {
        return Err(CurrentProviderError::invalid_request(format!(
            "Vertex Gemini base URL has unsupported path `{}`; expected an origin URL",
            url.path()
        )));
    }
    {
        let mut segments = url.path_segments_mut().map_err(|_| {
            CurrentProviderError::invalid_request(
                "Vertex Gemini base URL cannot carry path segments",
            )
        })?;
        segments.clear();
        for segment in [
            "v1",
            "projects",
            project,
            "locations",
            location,
            "publishers",
            "google",
        ] {
            segments.push(segment);
        }
    }
    Ok(url.to_string().trim_end_matches('/').to_string())
}

fn required_vertex_context<'a>(
    field: &str,
    value: Option<&'a str>,
) -> Result<&'a str, CurrentProviderError> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            CurrentProviderError::invalid_request(format!(
                "Vertex Gemini provider requires {field}"
            ))
        })
}

/// 返回 Responses API 的首选 endpoint。
///
/// 图片编排等非回合消费者也必须复用 current provider 的 endpoint 规则，不能重新
/// 按 current provider 的 Responses 路径规则构造 endpoint。
pub fn responses_endpoint(base_url: &str) -> String {
    provider_urls(&ModelProviderProtocol::Responses, Some(base_url), None)
        .into_iter()
        .next()
        .unwrap_or_else(|| format!("{}/v1/responses", base_url.trim_end_matches('/')))
}

fn endpoint_urls(base_url: &str, endpoint: &str) -> Vec<String> {
    let base = base_url.trim_end_matches('/');
    if base.ends_with(endpoint) {
        return vec![base.to_string()];
    }
    let ends_with_version = base.rsplit('/').next().is_some_and(|segment| {
        segment.starts_with('v')
            && segment.len() > 1
            && segment[1..]
                .chars()
                .all(|character| character.is_ascii_digit())
    });
    let primary = if ends_with_version {
        format!("{base}/{endpoint}")
    } else if url::Url::parse(base)
        .ok()
        .is_some_and(|url| url.path().trim_matches('/').is_empty())
    {
        format!("{base}/v1/{endpoint}")
    } else {
        format!("{base}/{endpoint}")
    };
    let mut urls = vec![primary.clone()];
    if primary.contains("/v1/") {
        let without_v1 = primary.replacen("/v1/", "/", 1);
        if without_v1 != primary {
            urls.push(without_v1);
        }
    }
    urls
}

#[cfg(test)]
mod tests {
    use super::lowering::{anthropic_request, chat_completions_request, responses_request};
    use super::stream::{
        drain_sse_frames, openai_chat_sse, parse_sse_frame, response_item_tool_call, responses_sse,
        sse_frames_with_idle_timeout,
    };
    use super::*;
    use crate::runtime_provider::RuntimeProviderConfig;
    use futures::{SinkExt, StreamExt};
    use serde_json::json;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
        sync::oneshot,
        task::JoinHandle,
    };
    use tokio_tungstenite::tungstenite::Message;

    fn config(protocol: Option<RuntimeProviderProtocol>) -> RuntimeProviderConfig {
        RuntimeProviderConfig {
            provider_name: "openai".to_string(),
            provider_selector: Some("openai".to_string()),
            model_name: "gpt-5-codex".to_string(),
            api_key: Some("test".to_string()),
            auth: RuntimeProviderAuth::ApiKey,
            base_url: Some("https://gateway.example.com/v1".to_string()),
            api_version: None,
            credential_uuid: "credential-1".to_string(),
            reasoning_effort: Some("medium".to_string()),
            service_tier: None,
            protocol,
            supports_websockets: false,
            toolshim: false,
            toolshim_model: None,
        }
    }

    fn text_request() -> CurrentProviderRequest {
        CurrentProviderRequest::new(vec![CurrentProviderMessage::user(vec![
            CurrentProviderContent::Text("hello".to_string()),
        ])])
    }

    #[test]
    fn raw_response_items_are_preserved_by_responses_lowering() {
        let item = json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "injected context"}],
            "provider_extension": {"keep": true}
        });
        let request = CurrentProviderRequest::new(vec![CurrentProviderMessage::raw_response_item(
            item.clone(),
        )]);
        let canonical = request
            .into_canonical("gpt-5-codex")
            .expect("canonical raw response item");
        let payload = responses_request(
            &config(Some(RuntimeProviderProtocol::Responses)),
            &canonical,
            &RuntimeReplyProviderRequestWireShape::default(),
            &Default::default(),
        )
        .expect("Responses lowering");

        assert_eq!(payload["input"], json!([item]));
    }

    #[tokio::test]
    async fn raw_response_items_fail_closed_for_non_responses_routes() {
        let client =
            CurrentProviderClient::new(config(Some(RuntimeProviderProtocol::ChatCompletions)))
                .expect("chat client");
        let request =
            CurrentProviderRequest::new(vec![CurrentProviderMessage::raw_response_item(json!({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "injected context"}]
            }))]);

        let error = match client.stream(request).await {
            Ok(_) => panic!("raw response item must not reach chat-completions transport"),
            Err(error) => error,
        };
        assert_eq!(
            error.classification,
            Some(FailureClassification::InvalidRequest)
        );
        assert!(error.message.contains("require a Responses provider route"));
    }

    async fn stream_error(client: &CurrentProviderClient, scenario: &str) -> CurrentProviderError {
        match client.stream(text_request()).await {
            Ok(_) => panic!("{scenario}: expected provider request to fail"),
            Err(error) => error,
        }
    }

    #[test]
    fn endpoint_urls_keep_versioned_and_custom_provider_paths() {
        assert_eq!(
            endpoint_urls("https://api.openai.com", "chat/completions"),
            vec![
                "https://api.openai.com/v1/chat/completions".to_string(),
                "https://api.openai.com/chat/completions".to_string(),
            ]
        );
        assert_eq!(
            endpoint_urls(
                "https://gateway.example.com/compatible-mode/v2",
                "chat/completions"
            ),
            vec!["https://gateway.example.com/compatible-mode/v2/chat/completions".to_string()]
        );
    }

    #[test]
    fn responses_endpoint_uses_current_provider_path_rules() {
        assert_eq!(
            responses_endpoint("https://api.openai.com"),
            "https://api.openai.com/v1/responses"
        );
        assert_eq!(
            responses_endpoint("https://api.openai.com/v1"),
            "https://api.openai.com/v1/responses"
        );
        assert_eq!(
            responses_endpoint("https://gateway.example.com/codex"),
            "https://gateway.example.com/codex/responses"
        );
        assert_eq!(
            responses_endpoint("https://gateway.example.com/proxy/responses"),
            "https://gateway.example.com/proxy/responses"
        );
    }

    #[test]
    fn client_rejects_missing_route_protocol_instead_of_inferring_from_provider_name() {
        let mut config = config(None);
        config.provider_name = "anthropic".to_string();
        config.provider_selector = Some("anthropic".to_string());
        let construction_error = match CurrentProviderClient::new(config.clone()) {
            Ok(_) => panic!("client construction must reject a missing route protocol"),
            Err(error) => error,
        };
        assert!(construction_error.message.contains("anthropic"));

        let client = CurrentProviderClient::with_client(config, Client::new());
        let error = client
            .protocol()
            .expect_err("missing route protocol must fail closed");

        assert_eq!(
            error.classification,
            Some(FailureClassification::InvalidRequest)
        );
        assert!(!error.retryable);
        assert!(error.message.contains("missing an explicit protocol"));
        assert!(error.message.contains("anthropic"));
    }

    #[test]
    fn custom_protocol_is_rejected_before_wire_lowering() {
        let error = ensure_supported_protocol(&ModelProviderProtocol::Custom(
            "gemini_generate_content".to_string(),
        ))
        .expect_err("custom protocols require an explicit wire adapter");

        assert_eq!(
            error.classification,
            Some(FailureClassification::InvalidRequest)
        );
        assert!(!error.retryable);
        assert!(error.message.contains("unsupported provider protocol"));
    }

    #[test]
    fn runtime_handle_reports_selected_model_and_streaming_capability() {
        let client = CurrentProviderClient::with_client(
            config(Some(RuntimeProviderProtocol::Responses)),
            Client::new(),
        );

        let handle = client.runtime_handle();

        assert!(handle.capabilities.supports_streaming);
        assert!(!handle.capabilities.supports_embeddings);
        assert_eq!(
            handle.capabilities.active_model_name.as_deref(),
            Some("gpt-5-codex")
        );
        assert_eq!(
            handle
                .provider_trace_metadata()
                .runtime_provider_active_model
                .as_deref(),
            Some("gpt-5-codex")
        );
    }

    #[test]
    fn verification_metadata_is_trusted_only_for_first_party_routes() {
        let mut official = config(Some(RuntimeProviderProtocol::Responses));
        official.base_url = Some("https://api.openai.com/v1".to_string());
        assert!(CurrentProviderClient::with_client(official, Client::new())
            .trusts_openai_response_metadata());

        let mut custom_official = config(Some(RuntimeProviderProtocol::Responses));
        custom_official.provider_selector = Some("custom-provider-id".to_string());
        custom_official.base_url = Some("https://api.openai.com/v1".to_string());
        assert!(
            CurrentProviderClient::with_client(custom_official, Client::new())
                .trusts_openai_response_metadata()
        );

        let gateway = config(Some(RuntimeProviderProtocol::Responses));
        assert!(!CurrentProviderClient::with_client(gateway, Client::new())
            .trusts_openai_response_metadata());

        let mut codex_labeled_gateway = config(Some(RuntimeProviderProtocol::Responses));
        codex_labeled_gateway.provider_selector = Some("codex".to_string());
        assert!(
            !CurrentProviderClient::with_client(codex_labeled_gateway, Client::new())
                .trusts_openai_response_metadata()
        );

        let mut chat = config(Some(RuntimeProviderProtocol::ChatCompletions));
        chat.base_url = Some("https://api.openai.com/v1".to_string());
        assert!(!CurrentProviderClient::with_client(chat, Client::new())
            .trusts_openai_response_metadata());
    }

    #[test]
    fn chat_lowering_preserves_images_prior_tool_calls_and_results() {
        let request = CurrentProviderRequest::new(vec![
            CurrentProviderMessage::user(vec![
                CurrentProviderContent::Text("look".to_string()),
                CurrentProviderContent::Image {
                    uri: "sidecar://image-1".to_string(),
                    media_type: "image/png".to_string(),
                    provider_data: Some("data:image/png;base64,abc".to_string()),
                    detail: Some(ImageDetail::High),
                },
            ]),
            CurrentProviderMessage::assistant(vec![CurrentProviderContent::ToolCall(
                CurrentProviderToolCall::new("call-1", "Read", json!({ "path": "README.md" })),
            )]),
            CurrentProviderMessage::tool(vec![CurrentProviderContent::ToolResult(
                CurrentProviderToolResult {
                    call_id: "call-1".to_string(),
                    name: "Read".to_string(),
                    success: true,
                    output: "content".to_string(),
                    error: None,
                },
            )]),
        ]);

        let canonical = request
            .into_canonical("gpt-5-codex")
            .expect("canonical request");
        let media_payloads = request.media_payloads().expect("media payloads");
        let value = chat_completions_request(
            &config(Some(RuntimeProviderProtocol::ChatCompletions)),
            &canonical,
            &RuntimeReplyProviderRequestWireShape::default(),
            &media_payloads,
        )
        .expect("Chat Completions lowering");

        assert_eq!(value["messages"][0]["content"][1]["type"], "image_url");
        assert_eq!(
            value["messages"][0]["content"][1]["image_url"]["url"],
            "data:image/png;base64,abc"
        );
        assert_eq!(
            value["messages"][0]["content"][1]["image_url"]["detail"],
            "high"
        );
        assert_eq!(
            value["messages"][1]["tool_calls"][0]["function"]["name"],
            "Read"
        );
        assert_eq!(value["messages"][2]["tool_call_id"], "call-1");
    }

    #[test]
    fn chat_lowering_preserves_generation_and_thinking_controls() {
        let request = text_request()
            .with_generation(GenerationOptions {
                max_tokens: Some(128),
                temperature: Some(0.2),
                top_p: Some(0.8),
                top_k: None,
            })
            .with_provider_options(ProviderMetadata::from([(
                "enable_thinking".to_string(),
                json!(false),
            )]));
        let canonical = request
            .into_canonical("agnes-2.0-flash")
            .expect("canonical request");
        let value = chat_completions_request(
            &config(Some(RuntimeProviderProtocol::ChatCompletions)),
            &canonical,
            &RuntimeReplyProviderRequestWireShape::default(),
            &Default::default(),
        )
        .expect("Chat Completions lowering");

        assert_eq!(value["max_tokens"], 128);
        assert_eq!(value["temperature"], 0.2);
        assert_eq!(value["top_p"], 0.8);
        assert_eq!(value["chat_template_kwargs"]["enable_thinking"], false);
    }

    #[test]
    fn responses_and_anthropic_lowering_resolve_provider_media_payload() {
        let request = CurrentProviderRequest::new(vec![CurrentProviderMessage::user(vec![
            CurrentProviderContent::Text("look".to_string()),
            CurrentProviderContent::Image {
                uri: "sidecar://image-1".to_string(),
                media_type: "image/png".to_string(),
                provider_data: Some("data:image/png;base64,abc".to_string()),
                detail: Some(ImageDetail::Original),
            },
        ])]);
        let canonical = request
            .into_canonical("vision-model")
            .expect("canonical request");
        let media_payloads = request.media_payloads().expect("media payloads");

        let responses = responses_request(
            &config(Some(RuntimeProviderProtocol::Responses)),
            &canonical,
            &RuntimeReplyProviderRequestWireShape::default(),
            &media_payloads,
        )
        .expect("Responses lowering");
        let anthropic = anthropic_request(
            &config(Some(RuntimeProviderProtocol::AnthropicMessages)),
            &canonical,
            &media_payloads,
        )
        .expect("Anthropic lowering");

        assert_eq!(
            responses["input"][0]["content"][1]["image_url"],
            "data:image/png;base64,abc"
        );
        assert_eq!(responses["input"][0]["content"][1]["detail"], "original");
        assert_eq!(
            anthropic["messages"][0]["content"][1]["source"],
            json!({
                "type": "base64",
                "media_type": "image/png",
                "data": "abc"
            })
        );
    }

    #[test]
    fn canonical_request_keeps_provider_media_payload_out_of_reference() {
        let request = CurrentProviderRequest::new(vec![CurrentProviderMessage::user(vec![
            CurrentProviderContent::Image {
                uri: "sidecar://image-1".to_string(),
                media_type: "image/png".to_string(),
                provider_data: Some("data:image/png;base64,abc".to_string()),
                detail: Some(ImageDetail::Low),
            },
        ])]);

        let canonical = request
            .into_canonical("gpt-5-codex")
            .expect("reference-only canonical request");

        assert!(matches!(
            &canonical.messages[0].content[0],
            ContentPart::Media {
                uri,
                detail: Some(ImageDetail::Low),
                ..
            } if uri == "sidecar://image-1"
        ));
        assert!(!serde_json::to_string(&canonical)
            .expect("serialize canonical request")
            .contains("base64,abc"));
    }

    #[test]
    fn local_media_reference_without_provider_payload_fails_before_network() {
        let request = CurrentProviderRequest::new(vec![CurrentProviderMessage::user(vec![
            CurrentProviderContent::Image {
                uri: "sidecar://image-1".to_string(),
                media_type: "image/png".to_string(),
                provider_data: None,
                detail: None,
            },
        ])]);

        let error = request
            .media_payloads()
            .expect_err("local reference must resolve before provider request");

        assert_eq!(
            error.classification,
            Some(FailureClassification::InvalidRequest)
        );
        assert!(!error.retryable);
        assert!(error.message.contains("no provider-readable payload"));
    }

    #[test]
    fn responses_tool_call_is_normalized_from_final_item() {
        let item = json!({
            "type": "function_call",
            "call_id": "call-7",
            "name": "apply_patch",
            "arguments": "{\"patch\":\"*** Begin Patch\"}"
        });
        let call = response_item_tool_call(&item)
            .expect("valid tool call")
            .expect("tool call");

        assert_eq!(call.id, "call-7");
        assert_eq!(call.name, "apply_patch");
        assert_eq!(call.arguments["patch"], "*** Begin Patch");
    }

    #[test]
    fn responses_tool_call_preserves_invalid_json_for_runtime_repair() {
        let item = json!({
            "type": "function_call",
            "call_id": "call-invalid",
            "name": "apply_patch",
            "arguments": "{not-json"
        });

        let call = response_item_tool_call(&item)
            .expect("tool call envelope")
            .expect("tool call");

        assert_eq!(call.name, "apply_patch");
        assert_eq!(call.arguments, json!("{not-json"));
        assert_eq!(call.raw_arguments, "{not-json");
    }

    #[test]
    fn responses_tool_call_preserves_blank_tool_name_for_runtime_repair() {
        let item = json!({
            "type": "function_call",
            "call_id": "call-blank-name",
            "name": "  ",
            "arguments": "{}"
        });

        let call = response_item_tool_call(&item)
            .expect("tool call envelope")
            .expect("tool call");

        assert_eq!(call.name, "");
        assert_eq!(call.arguments, json!({}));
    }

    #[test]
    fn sse_frame_parser_keeps_multiline_data_without_comments() {
        let frame =
            parse_sse_frame(": keepalive\ndata: {\"type\":\"response.created\"}\ndata: second")
                .expect("frame");
        assert_eq!(frame.data, "{\"type\":\"response.created\"}\nsecond");
    }

    #[test]
    fn sse_frame_buffer_preserves_utf8_split_across_chunks() {
        let mut pending = b"data: {\"delta\":\"".to_vec();
        pending.extend_from_slice(&[0xE4, 0xB8]);

        assert!(drain_sse_frames(&mut pending)
            .expect("incomplete UTF-8 must stay buffered")
            .is_empty());

        pending.extend_from_slice(&[0xAD, b'\"', b'}', b'\n', b'\n']);
        let frames = drain_sse_frames(&mut pending).expect("valid UTF-8 frame");

        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].data, "{\"delta\":\"中\"}");
        assert!(pending.is_empty());
    }

    #[tokio::test]
    async fn openai_tool_stream_accepts_arguments_before_name() {
        let body = concat!(
            "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-5.5\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"type\":\"function\",\"function\":{\"arguments\":\"{\\\"query\\\":\\\"\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-5.5\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"WebSearch\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-5.5\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"Rust release\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-5.5\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let (base_url, _requests, server) = spawn_http_fixture(vec![fixture_response(
            "200 OK",
            "Content-Type: text/event-stream\r\n",
            body,
        )])
        .await;
        let response = Client::builder()
            .no_proxy()
            .build()
            .expect("HTTP client")
            .get(base_url)
            .send()
            .await
            .expect("SSE response");

        let events = openai_chat_sse(response)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("out-of-order tool fields must normalize");

        server.await.expect("fixture server");
        assert!(events.iter().any(|event| matches!(
            event,
            CanonicalLlmEvent::ToolCall { name, input, .. }
                if name == "WebSearch" && input == &json!({ "query": "Rust release" })
        )));
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, CanonicalLlmEvent::ToolCall { .. }))
                .count(),
            1
        );
        let lifecycle = events
            .iter()
            .filter_map(|event| match event {
                CanonicalLlmEvent::ToolInputStart { .. } => Some("start"),
                CanonicalLlmEvent::ToolInputDelta { .. } => Some("delta"),
                CanonicalLlmEvent::ToolInputEnd { .. } => Some("end"),
                CanonicalLlmEvent::ToolCall { .. } => Some("call"),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(lifecycle, ["start", "delta", "delta", "end", "call"]);
    }

    #[tokio::test]
    async fn openai_text_stream_ignores_empty_tool_call_placeholder() {
        let body = concat!(
            "data: {\"id\":\"chatcmpl-empty-tool\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-5.5\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"done\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl-empty-tool\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-5.5\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"\",\"type\":\"function\",\"function\":{\"name\":\"\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl-empty-tool\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-5.5\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let (base_url, _requests, server) = spawn_http_fixture(vec![fixture_response(
            "200 OK",
            "Content-Type: text/event-stream\r\n",
            body,
        )])
        .await;
        let response = Client::builder()
            .no_proxy()
            .build()
            .expect("HTTP client")
            .get(base_url)
            .send()
            .await
            .expect("SSE response");

        let events = openai_chat_sse(response)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("empty tool placeholder must not fail a text response");

        server.await.expect("fixture server");
        assert!(events.iter().any(|event| matches!(
            event,
            CanonicalLlmEvent::TextDelta { text, .. } if text == "done"
        )));
        assert!(!events
            .iter()
            .any(|event| matches!(event, CanonicalLlmEvent::ToolCall { .. })));
        assert!(events.iter().any(|event| matches!(
            event,
            CanonicalLlmEvent::Finish {
                reason: FinishReason::Stop,
                ..
            }
        )));
    }

    #[tokio::test]
    async fn openai_tool_stream_preserves_arguments_without_name() {
        let body = concat!(
            "data: {\"id\":\"chatcmpl-missing-name\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-5.5\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"type\":\"function\",\"function\":{\"arguments\":\"{}\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl-missing-name\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-5.5\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"
        );
        let (base_url, _requests, server) = spawn_http_fixture(vec![fixture_response(
            "200 OK",
            "Content-Type: text/event-stream\r\n",
            body,
        )])
        .await;
        let response = Client::builder()
            .no_proxy()
            .build()
            .expect("HTTP client")
            .get(base_url)
            .send()
            .await
            .expect("SSE response");

        let events = openai_chat_sse(response)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("incomplete tool call must reach runtime repair");

        server.await.expect("fixture server");
        assert!(events.iter().any(|event| matches!(
            event,
            CanonicalLlmEvent::ToolCall {
                name,
                raw_arguments: Some(arguments),
                ..
            } if name.is_empty() && arguments == "{}"
        )));
    }

    #[tokio::test]
    async fn provider_stream_timeout_is_idle_based() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind idle fixture");
        let address = listener.local_addr().expect("idle fixture address");
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("accept idle request");
            read_http_headers(&mut stream).await;
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 10\r\n\r\n",
                )
                .await
                .expect("write idle response headers");
            tokio::time::sleep(Duration::from_millis(100)).await;
        });
        let response = Client::builder()
            .no_proxy()
            .build()
            .expect("HTTP client")
            .get(format!("http://{address}"))
            .send()
            .await
            .expect("idle SSE response");

        let mut frames = Box::pin(sse_frames_with_idle_timeout(
            response,
            Duration::from_millis(10),
        ));
        let error = frames
            .next()
            .await
            .expect("idle stream result")
            .expect_err("idle stream must time out");

        assert_eq!(error.message, "读取 provider SSE 超时: 10 ms 内未收到数据");
        server.await.expect("idle fixture server");
    }

    #[tokio::test]
    async fn responses_tool_stream_accepts_arguments_before_name() {
        let body = concat!(
            "data: {\"type\":\"response.function_call_arguments.delta\",\"call_id\":\"call-1\",\"delta\":\"{\\\"query\\\":\\\"\"}\n\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"call_id\":\"call-1\",\"name\":\"WebSearch\",\"delta\":\"Rust release\\\"}\"}\n\n",
            "data: {\"type\":\"response.function_call_arguments.done\",\"call_id\":\"call-1\",\"arguments\":\"{\\\"query\\\":\\\"Rust release\\\"}\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-1\",\"output\":[]}}\n\n"
        );
        let (base_url, _requests, server) = spawn_http_fixture(vec![fixture_response(
            "200 OK",
            "Content-Type: text/event-stream\r\n",
            body,
        )])
        .await;
        let response = Client::builder()
            .no_proxy()
            .build()
            .expect("HTTP client")
            .get(base_url)
            .send()
            .await
            .expect("SSE response");

        let events = responses_sse(response, true)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("out-of-order tool fields must normalize");

        server.await.expect("fixture server");
        assert!(events.iter().any(|event| matches!(
            event,
            CanonicalLlmEvent::ToolCall { name, input, .. }
                if name == "WebSearch" && input == &json!({ "query": "Rust release" })
        )));
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, CanonicalLlmEvent::ToolCall { .. }))
                .count(),
            1
        );
        let lifecycle = events
            .iter()
            .filter_map(|event| match event {
                CanonicalLlmEvent::ToolInputStart { .. } => Some("start"),
                CanonicalLlmEvent::ToolInputDelta { .. } => Some("delta"),
                CanonicalLlmEvent::ToolInputEnd { .. } => Some("end"),
                CanonicalLlmEvent::ToolCall { .. } => Some("call"),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(lifecycle, ["start", "delta", "end", "call"]);
    }

    #[tokio::test]
    async fn responses_tool_stream_preserves_terminal_arguments_without_name() {
        let body = concat!(
            "data: {\"type\":\"response.function_call_arguments.delta\",\"call_id\":\"call-1\",\"delta\":\"{}\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-1\",\"output\":[]}}\n\n"
        );
        let (base_url, _requests, server) = spawn_http_fixture(vec![fixture_response(
            "200 OK",
            "Content-Type: text/event-stream\r\n",
            body,
        )])
        .await;
        let response = Client::builder()
            .no_proxy()
            .build()
            .expect("HTTP client")
            .get(base_url)
            .send()
            .await
            .expect("SSE response");

        let events = responses_sse(response, true)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("incomplete tool call must reach runtime repair");

        server.await.expect("fixture server");
        assert!(events.iter().any(|event| matches!(
            event,
            CanonicalLlmEvent::ToolCall {
                name,
                raw_arguments: Some(arguments),
                ..
            } if name.is_empty() && arguments == "{}"
        )));
    }

    #[tokio::test]
    async fn stream_request_retries_server_statuses_until_success() {
        let (base_url, requests, server) = spawn_http_fixture(vec![
            fixture_response("501 Not Implemented", "Retry-After: 0\r\n", "first"),
            fixture_response("505 HTTP Version Not Supported", "", "second"),
            fixture_response("200 OK", "", ""),
        ])
        .await;
        let mut runtime_config = config(Some(RuntimeProviderProtocol::ChatCompletions));
        runtime_config.base_url = Some(base_url);
        let client = CurrentProviderClient::with_client(
            runtime_config,
            Client::builder().no_proxy().build().expect("HTTP client"),
        );

        let _stream = client
            .stream(text_request())
            .await
            .expect("third attempt succeeds");

        server.await.expect("fixture server");
        assert_eq!(requests.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn no_auth_http_request_omits_authorization_header() {
        let (base_url, headers, server) = spawn_http_headers_fixture(fixture_response(
            "200 OK",
            "Content-Type: text/event-stream\r\n",
            "",
        ))
        .await;
        let mut runtime_config = config(Some(RuntimeProviderProtocol::ChatCompletions));
        runtime_config.auth = RuntimeProviderAuth::NoAuth;
        runtime_config.api_key = None;
        runtime_config.base_url = Some(base_url);
        runtime_config.credential_uuid.clear();
        let client = CurrentProviderClient::with_client(
            runtime_config,
            Client::builder().no_proxy().build().expect("HTTP client"),
        );

        client
            .send_stream_request(
                &ModelProviderProtocol::ChatCompletions,
                json!({ "stream": true }),
                &RuntimeReplyProviderRequestWireShape::default(),
            )
            .await
            .expect("no-auth provider request");

        let headers = headers
            .await
            .expect("captured HTTP request")
            .to_ascii_lowercase();
        server.await.expect("fixture server");
        assert!(!headers.contains("\nauthorization:"));
        assert!(!headers.contains("\nx-api-key:"));
    }

    #[tokio::test]
    async fn api_key_auth_still_rejects_missing_key_before_network() {
        let mut runtime_config = config(Some(RuntimeProviderProtocol::ChatCompletions));
        runtime_config.api_key = None;
        let client = CurrentProviderClient::with_client(runtime_config, Client::new());

        let error = client
            .send_stream_request(
                &ModelProviderProtocol::ChatCompletions,
                json!({ "stream": true }),
                &RuntimeReplyProviderRequestWireShape::default(),
            )
            .await
            .expect_err("API-key route must fail closed");

        assert_eq!(error.message, "Provider API key 未配置");
    }

    #[tokio::test]
    async fn responses_websocket_capability_uses_upgrade_transport() {
        let (base_url, capture, server) = spawn_websocket_fixture().await;
        let mut runtime_config = config(Some(RuntimeProviderProtocol::Responses));
        runtime_config.base_url = Some(base_url);
        runtime_config.supports_websockets = true;
        let client = CurrentProviderClient::with_client(
            runtime_config,
            Client::builder().no_proxy().build().expect("HTTP client"),
        );

        let events = client
            .stream(text_request())
            .await
            .expect("fixture accepts provider request")
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("websocket events");
        let capture = capture.await.expect("captured websocket request");

        server.await.expect("fixture server");
        assert_eq!(capture.method, "GET");
        assert_eq!(capture.path, "/v1/responses");
        assert_eq!(capture.authorization.as_deref(), Some("Bearer test"));
        assert_eq!(
            capture.beta.as_deref(),
            Some("responses_websockets=2026-02-06")
        );
        assert_eq!(capture.payload["type"], "response.create");
        assert!(capture.payload.get("stream").is_none());
        assert!(events.iter().any(|event| matches!(
            event,
            CanonicalLlmEvent::ServerModel { model } if model == "gpt-5-codex"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            CanonicalLlmEvent::Finish {
                response_id: Some(response_id),
                ..
            } if response_id == "resp-ws-1"
        )));
    }

    #[tokio::test]
    async fn no_auth_responses_websocket_omits_authorization_header() {
        let (base_url, capture, server) = spawn_websocket_fixture().await;
        let mut runtime_config = config(Some(RuntimeProviderProtocol::Responses));
        runtime_config.auth = RuntimeProviderAuth::NoAuth;
        runtime_config.api_key = None;
        runtime_config.base_url = Some(base_url);
        runtime_config.credential_uuid.clear();
        runtime_config.supports_websockets = true;
        let client = CurrentProviderClient::with_client(
            runtime_config,
            Client::builder().no_proxy().build().expect("HTTP client"),
        );

        client
            .stream(text_request())
            .await
            .expect("no-auth websocket request")
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("websocket events");
        let capture = capture.await.expect("captured websocket request");

        server.await.expect("fixture server");
        assert!(capture.authorization.is_none());
        assert_eq!(
            capture.beta.as_deref(),
            Some("responses_websockets=2026-02-06")
        );
    }

    #[tokio::test]
    async fn responses_websocket_ignores_verification_for_codex_labeled_third_party_route() {
        let (base_url, _capture, server) = spawn_websocket_fixture().await;
        let mut runtime_config = config(Some(RuntimeProviderProtocol::Responses));
        runtime_config.provider_selector = Some("codex".to_string());
        runtime_config.base_url = Some(base_url);
        runtime_config.supports_websockets = true;
        let client = CurrentProviderClient::with_client(
            runtime_config,
            Client::builder().no_proxy().build().expect("HTTP client"),
        );

        let events = client
            .stream(text_request())
            .await
            .expect("third-party websocket request")
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("websocket events");

        server.await.expect("fixture server");
        assert!(!events
            .iter()
            .any(|event| matches!(event, CanonicalLlmEvent::ModelVerification { .. })));
    }

    #[tokio::test]
    async fn responses_websocket_426_fallback_is_sticky_for_client_session() {
        let body = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-http\",\"output\":[]}}\n\n";
        let responses = vec![
            fixture_response("426 Upgrade Required", "", "upgrade unsupported"),
            fixture_response("200 OK", "Content-Type: text/event-stream\r\n", body),
            fixture_response("200 OK", "Content-Type: text/event-stream\r\n", body),
        ];
        let (base_url, methods, server) = spawn_http_method_fixture(responses).await;
        let mut runtime_config = config(Some(RuntimeProviderProtocol::Responses));
        runtime_config.base_url = Some(base_url);
        runtime_config.supports_websockets = true;
        let client = CurrentProviderClient::with_client(
            runtime_config,
            Client::builder().no_proxy().build().expect("HTTP client"),
        );

        for _ in 0..2 {
            client
                .stream(text_request())
                .await
                .expect("HTTP fallback stream")
                .collect::<Vec<_>>()
                .await
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .expect("HTTP fallback events");
        }

        server.await.expect("fixture server");
        assert_eq!(
            methods.lock().expect("method capture").as_slice(),
            ["GET", "POST", "POST"]
        );
        assert!(!client.responses_websocket_enabled());
    }

    #[tokio::test]
    async fn responses_without_websocket_capability_stays_on_http() {
        let body = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-http\",\"output\":[]}}\n\n";
        let (base_url, methods, server) = spawn_http_method_fixture(vec![fixture_response(
            "200 OK",
            "Content-Type: text/event-stream\r\n",
            body,
        )])
        .await;
        let mut runtime_config = config(Some(RuntimeProviderProtocol::Responses));
        runtime_config.base_url = Some(base_url);
        let client = CurrentProviderClient::with_client(
            runtime_config,
            Client::builder().no_proxy().build().expect("HTTP client"),
        );

        client
            .stream(text_request())
            .await
            .expect("HTTP stream")
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("HTTP events");

        server.await.expect("fixture server");
        assert_eq!(methods.lock().expect("method capture").as_slice(), ["POST"]);
    }

    #[tokio::test]
    async fn responses_websocket_retry_exhaustion_replays_over_http() {
        let body = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-http\",\"output\":[]}}\n\n";
        let mut responses = (0..MAX_STREAM_REQUEST_ATTEMPTS)
            .map(|_| fixture_response("500 Internal Server Error", "", "upgrade failed"))
            .collect::<Vec<_>>();
        responses.push(fixture_response(
            "200 OK",
            "Content-Type: text/event-stream\r\n",
            body,
        ));
        let (base_url, methods, server) = spawn_http_method_fixture(responses).await;
        let mut runtime_config = config(Some(RuntimeProviderProtocol::Responses));
        runtime_config.base_url = Some(base_url);
        runtime_config.supports_websockets = true;
        let client = CurrentProviderClient::with_client(
            runtime_config,
            Client::builder().no_proxy().build().expect("HTTP client"),
        );

        client
            .stream(text_request())
            .await
            .expect("HTTP replay stream")
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("HTTP replay events");

        server.await.expect("fixture server");
        let methods = methods.lock().expect("method capture");
        assert_eq!(
            methods
                .iter()
                .filter(|method| method.as_str() == "GET")
                .count(),
            usize::from(MAX_STREAM_REQUEST_ATTEMPTS)
        );
        assert_eq!(methods.last().map(String::as_str), Some("POST"));
    }

    #[tokio::test]
    async fn responses_websocket_respects_explicit_server_retry_false() {
        let body = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-http\",\"output\":[]}}\n\n";
        let responses = vec![
            fixture_response(
                "503 Service Unavailable",
                "X-Should-Retry: false\r\n",
                "upgrade failed",
            ),
            fixture_response("200 OK", "Content-Type: text/event-stream\r\n", body),
        ];
        let (base_url, methods, server) = spawn_http_method_fixture(responses).await;
        let mut runtime_config = config(Some(RuntimeProviderProtocol::Responses));
        runtime_config.base_url = Some(base_url);
        runtime_config.supports_websockets = true;
        let client = CurrentProviderClient::with_client(
            runtime_config,
            Client::builder().no_proxy().build().expect("HTTP client"),
        );

        client
            .stream(text_request())
            .await
            .expect("HTTP replay stream")
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("HTTP replay events");

        server.await.expect("fixture server");
        assert_eq!(
            methods.lock().expect("method capture").as_slice(),
            ["GET", "POST"]
        );
    }

    #[tokio::test]
    async fn responses_websocket_close_before_output_replays_over_http() {
        let (base_url, methods, server) = spawn_websocket_drop_then_http_fixture().await;
        let mut runtime_config = config(Some(RuntimeProviderProtocol::Responses));
        runtime_config.base_url = Some(base_url);
        runtime_config.supports_websockets = true;
        let client = CurrentProviderClient::with_client(
            runtime_config,
            Client::builder().no_proxy().build().expect("HTTP client"),
        );

        let events = client
            .stream(text_request())
            .await
            .expect("websocket stream")
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("HTTP replay events");

        server.await.expect("fixture server");
        assert_eq!(
            methods.lock().expect("method capture").as_slice(),
            ["GET", "POST"]
        );
        assert!(events.iter().any(|event| matches!(
            event,
            CanonicalLlmEvent::Finish {
                response_id: Some(response_id),
                ..
            } if response_id == "resp-http-replay"
        )));
        assert!(!client.responses_websocket_enabled());
    }

    #[tokio::test]
    async fn stream_request_does_not_retry_non_retryable_statuses() {
        let (base_url, requests, server) = spawn_http_fixture(vec![fixture_response(
            "400 Bad Request",
            "",
            "invalid model",
        )])
        .await;
        let mut runtime_config = config(Some(RuntimeProviderProtocol::ChatCompletions));
        runtime_config.base_url = Some(base_url);
        let client = CurrentProviderClient::with_client(
            runtime_config,
            Client::builder().no_proxy().build().expect("HTTP client"),
        );

        let error = stream_error(&client, "bad request must fail immediately").await;

        assert_eq!(error.status, Some(StatusCode::BAD_REQUEST.as_u16()));
        assert_eq!(
            error.classification,
            Some(FailureClassification::InvalidRequest)
        );
        assert!(!error.retryable);
        assert_eq!(error.message, "Provider 请求失败 (400 Bad Request)");
        assert!(!error.message.contains("invalid model"));
        server.await.expect("fixture server");
        assert_eq!(requests.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn stream_request_classifies_authentication_without_retrying() {
        let (base_url, requests, server) = spawn_http_fixture(vec![fixture_response(
            "401 Unauthorized",
            "",
            r#"{"error":{"message":"invalid token"}}"#,
        )])
        .await;
        let mut runtime_config = config(Some(RuntimeProviderProtocol::ChatCompletions));
        runtime_config.base_url = Some(base_url);
        let client = CurrentProviderClient::with_client(
            runtime_config,
            Client::builder().no_proxy().build().expect("HTTP client"),
        );

        let error = stream_error(&client, "authentication failure must remain visible").await;

        assert_eq!(error.status, Some(StatusCode::UNAUTHORIZED.as_u16()));
        assert_eq!(
            error.classification,
            Some(FailureClassification::Authentication)
        );
        assert!(!error.retryable);
        assert_eq!(error.message, "Provider 请求失败 (401 Unauthorized)");
        assert!(!error.message.contains("invalid token"));
        server.await.expect("fixture server");
        assert_eq!(requests.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn stream_request_classifies_rate_limit_without_request_retry() {
        let (base_url, requests, server) = spawn_http_fixture(vec![fixture_response(
            "429 Too Many Requests",
            "Retry-After: 0\r\n",
            r#"{"error":{"message":"rate limited"}}"#,
        )])
        .await;
        let mut runtime_config = config(Some(RuntimeProviderProtocol::ChatCompletions));
        runtime_config.base_url = Some(base_url);
        let client = CurrentProviderClient::with_client(
            runtime_config,
            Client::builder().no_proxy().build().expect("HTTP client"),
        );

        let error = stream_error(
            &client,
            "rate limit must remain visible without request retry",
        )
        .await;

        assert_eq!(error.status, Some(StatusCode::TOO_MANY_REQUESTS.as_u16()));
        assert_eq!(error.classification, Some(FailureClassification::RateLimit));
        assert!(error.retryable);
        assert_eq!(error.message, "Provider 请求失败 (429 Too Many Requests)");
        assert!(!error.message.contains("rate limited"));
        server.await.expect("fixture server");
        assert_eq!(requests.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn stream_request_returns_final_transient_status_after_retry_budget() {
        let (base_url, requests, server) = spawn_http_fixture(vec![
            fixture_response("503 Service Unavailable", "Retry-After: 0\r\n", "first"),
            fixture_response("503 Service Unavailable", "Retry-After: 0\r\n", "second"),
            fixture_response("503 Service Unavailable", "Retry-After: 0\r\n", "third"),
            fixture_response("503 Service Unavailable", "Retry-After: 0\r\n", "fourth"),
            fixture_response("503 Service Unavailable", "Retry-After: 0\r\n", "final"),
        ])
        .await;
        let mut runtime_config = config(Some(RuntimeProviderProtocol::ChatCompletions));
        runtime_config.base_url = Some(base_url);
        let client = CurrentProviderClient::with_client(
            runtime_config,
            Client::builder().no_proxy().build().expect("HTTP client"),
        );

        let error = stream_error(&client, "all transient failures must remain visible").await;

        assert_eq!(error.status, Some(StatusCode::SERVICE_UNAVAILABLE.as_u16()));
        assert_eq!(
            error.classification,
            Some(FailureClassification::ProviderInternal)
        );
        assert!(error.retryable);
        assert_eq!(error.message, "Provider 请求失败 (503 Service Unavailable)");
        server.await.expect("fixture server");
        assert_eq!(
            requests.load(Ordering::SeqCst),
            usize::from(MAX_STREAM_REQUEST_ATTEMPTS)
        );
    }

    #[tokio::test]
    async fn stream_request_respects_explicit_server_retry_false() {
        for status in ["503 Service Unavailable", "429 Too Many Requests"] {
            let (base_url, requests, server) = spawn_http_fixture(vec![fixture_response(
                status,
                "X-Should-Retry: false\r\n",
                "do not retry",
            )])
            .await;
            let mut runtime_config = config(Some(RuntimeProviderProtocol::ChatCompletions));
            runtime_config.base_url = Some(base_url);
            let client = CurrentProviderClient::with_client(
                runtime_config,
                Client::builder().no_proxy().build().expect("HTTP client"),
            );

            let error = stream_error(&client, "explicit retry false must fail immediately").await;

            assert!(!error.retryable, "status={status}");
            server.await.expect("fixture server");
            assert_eq!(requests.load(Ordering::SeqCst), 1, "status={status}");
        }
    }

    #[tokio::test]
    async fn stream_request_shares_retry_budget_with_compatible_endpoint_probe() {
        let (base_url, requests, server) = spawn_http_fixture(vec![
            fixture_response("404 Not Found", "", "not found"),
            fixture_response("503 Service Unavailable", "Retry-After: 0\r\n", "second"),
            fixture_response("503 Service Unavailable", "Retry-After: 0\r\n", "third"),
            fixture_response("503 Service Unavailable", "Retry-After: 0\r\n", "fourth"),
            fixture_response("503 Service Unavailable", "Retry-After: 0\r\n", "final"),
        ])
        .await;
        let mut runtime_config = config(Some(RuntimeProviderProtocol::ChatCompletions));
        runtime_config.base_url = Some(base_url);
        let client = CurrentProviderClient::with_client(
            runtime_config,
            Client::builder().no_proxy().build().expect("HTTP client"),
        );

        let error = client
            .send_stream_request(
                &ModelProviderProtocol::ChatCompletions,
                json!({ "stream": true }),
                &RuntimeReplyProviderRequestWireShape::default(),
            )
            .await
            .expect_err("shared retry budget is exhausted");

        assert_eq!(error.status, Some(StatusCode::SERVICE_UNAVAILABLE.as_u16()));
        assert_eq!(error.message, "Provider 请求失败 (503 Service Unavailable)");
        server.await.expect("fixture server");
        assert_eq!(
            requests.load(Ordering::SeqCst),
            usize::from(MAX_STREAM_REQUEST_ATTEMPTS)
        );
    }

    fn health_test_client() -> CurrentProviderClient {
        let mut client = CurrentProviderClient::with_client(
            config(Some(RuntimeProviderProtocol::ChatCompletions)),
            Client::new(),
        );
        client.health = Arc::new(CircuitBreaker::new(HealthConfig {
            window_duration: Duration::from_secs(60),
            min_samples: 1,
            error_rate_threshold: 1.0,
            open_duration: Duration::from_secs(60),
        }));
        client
    }

    #[test]
    fn circuit_health_failure_filters_provider_rejections() {
        assert!(health_failure(
            Some(FailureClassification::RateLimit),
            Some(true),
            false
        ));
        assert!(health_failure(
            Some(FailureClassification::ProviderInternal),
            Some(true),
            false
        ));
        assert!(health_failure(
            Some(FailureClassification::Transport),
            Some(true),
            false
        ));
        for classification in [
            FailureClassification::RateLimit,
            FailureClassification::ProviderInternal,
            FailureClassification::Transport,
        ] {
            assert!(!health_failure(Some(classification), Some(false), false));
        }
        assert!(!health_failure(
            Some(FailureClassification::Authentication),
            Some(false),
            false
        ));
        assert!(!health_failure(
            Some(FailureClassification::InvalidRequest),
            Some(false),
            false
        ));
        assert!(!health_failure(
            Some(FailureClassification::ContextOverflow),
            Some(false),
            false
        ));
    }

    #[tokio::test]
    async fn tracked_stream_records_finish_success_and_failure_outcomes() {
        let success_client = health_test_client();
        let success_stream = success_client.tracked_stream(
            Box::pin(futures::stream::iter([Ok(CanonicalLlmEvent::Finish {
                reason: FinishReason::Stop,
                usage: None,
                response_id: None,
            })])),
            success_client.health.acquire().expect("success permit"),
        );
        let success_events = success_stream.collect::<Vec<_>>().await;
        assert_eq!(success_events.len(), 1);
        assert!(success_client.health.acquire().is_ok());

        let provider_error_client = health_test_client();
        let provider_error_stream = provider_error_client.tracked_stream(
            Box::pin(futures::stream::iter([Ok(
                CanonicalLlmEvent::ProviderError {
                    message: "provider unavailable".to_string(),
                    classification: Some(FailureClassification::ProviderInternal),
                    retryable: Some(true),
                },
            )])),
            provider_error_client
                .health
                .acquire()
                .expect("provider error permit"),
        );
        let _ = provider_error_stream.collect::<Vec<_>>().await;
        assert!(provider_error_client.health.acquire().is_err());

        let transport_error_client = health_test_client();
        let transport_error_stream = transport_error_client.tracked_stream(
            Box::pin(futures::stream::iter([Err(
                CurrentProviderError::transport("connection reset"),
            )])),
            transport_error_client
                .health
                .acquire()
                .expect("transport error permit"),
        );
        let _ = transport_error_stream.collect::<Vec<_>>().await;
        assert!(transport_error_client.health.acquire().is_err());

        let eof_client = health_test_client();
        let eof_stream = eof_client.tracked_stream(
            Box::pin(futures::stream::empty()),
            eof_client.health.acquire().expect("EOF permit"),
        );
        assert!(eof_stream.collect::<Vec<_>>().await.is_empty());
        assert!(eof_client.health.acquire().is_err());
    }

    #[tokio::test]
    async fn tracked_stream_emits_one_reroute_for_trusted_model_mismatch() {
        let mut runtime_config = config(Some(RuntimeProviderProtocol::Responses));
        runtime_config.base_url = Some("https://api.openai.com/v1".to_string());
        let client = CurrentProviderClient::with_client(runtime_config, Client::new());
        let stream = client.tracked_stream(
            Box::pin(futures::stream::iter([
                Ok(CanonicalLlmEvent::ServerModel {
                    model: "gpt-5.1-codex".to_string(),
                }),
                Ok(CanonicalLlmEvent::ServerModel {
                    model: "gpt-5.1-codex".to_string(),
                }),
                Ok(CanonicalLlmEvent::Finish {
                    reason: FinishReason::Stop,
                    usage: None,
                    response_id: None,
                }),
            ])),
            client.health.acquire().expect("trusted route permit"),
        );
        let events = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("tracked events");

        let reroutes = events
            .iter()
            .filter_map(|event| match event {
                CanonicalLlmEvent::ModelReroute {
                    from_model,
                    to_model,
                    reason,
                } => Some((from_model, to_model, reason)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(reroutes.len(), 1);
        assert_eq!(reroutes[0].0, "gpt-5-codex");
        assert_eq!(reroutes[0].1, "gpt-5.1-codex");
        assert_eq!(*reroutes[0].2, ModelRerouteReason::HighRiskCyberActivity);
    }

    #[tokio::test]
    async fn tracked_stream_ignores_case_only_match_and_untrusted_gateway() {
        for (base_url, server_model) in [
            ("https://api.openai.com/v1", "gpt-5-codex"),
            ("https://api.openai.com/v1", "GPT-5-CODEX"),
            ("https://gateway.example.com/v1", "gpt-5.1-codex"),
        ] {
            let mut runtime_config = config(Some(RuntimeProviderProtocol::Responses));
            runtime_config.provider_selector = Some("codex".to_string());
            runtime_config.base_url = Some(base_url.to_string());
            let client = CurrentProviderClient::with_client(runtime_config, Client::new());
            let stream = client.tracked_stream(
                Box::pin(futures::stream::iter([
                    Ok(CanonicalLlmEvent::ServerModel {
                        model: server_model.to_string(),
                    }),
                    Ok(CanonicalLlmEvent::Finish {
                        reason: FinishReason::Stop,
                        usage: None,
                        response_id: None,
                    }),
                ])),
                client.health.acquire().expect("route permit"),
            );
            let events = stream.collect::<Vec<_>>().await;
            assert!(!events
                .iter()
                .any(|event| matches!(event, Ok(CanonicalLlmEvent::ModelReroute { .. }))));
        }
    }

    #[tokio::test]
    async fn tracked_stream_drop_does_not_record_cancellation() {
        let client = health_test_client();
        let stream = client.tracked_stream(
            Box::pin(futures::stream::pending()),
            client.health.acquire().expect("pending permit"),
        );
        drop(stream);
        assert!(client.health.acquire().is_ok());
    }

    #[tokio::test]
    async fn stream_preflight_rejects_open_circuit_before_network() {
        let client = health_test_client();
        let mut permit = client.health.acquire().expect("initial permit");
        permit.failure();

        let error = match client.stream(text_request()).await {
            Err(error) => error,
            Ok(_) => panic!("open circuit must reject before network"),
        };
        assert!(error.message.contains("health circuit is open"));
        assert_eq!(error.classification, Some(FailureClassification::Transport));
        assert!(error.retryable);
    }

    async fn spawn_http_fixture(
        responses: Vec<String>,
    ) -> (String, Arc<AtomicUsize>, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind fixture server");
        let address = listener.local_addr().expect("fixture address");
        let requests = Arc::new(AtomicUsize::new(0));
        let request_count = Arc::clone(&requests);
        let server = tokio::spawn(async move {
            for response in responses {
                let (mut stream, _) = listener.accept().await.expect("accept fixture request");
                request_count.fetch_add(1, Ordering::SeqCst);
                read_http_headers(&mut stream).await;
                stream
                    .write_all(response.as_bytes())
                    .await
                    .expect("write fixture response");
                stream.shutdown().await.expect("close fixture response");
            }
        });
        (format!("http://{address}"), requests, server)
    }

    async fn spawn_http_headers_fixture(
        response: String,
    ) -> (String, oneshot::Receiver<String>, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind header fixture server");
        let address = listener.local_addr().expect("header fixture address");
        let (headers_tx, headers_rx) = oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("accept fixture request");
            let headers = read_http_headers_text(&mut stream).await;
            let _ = headers_tx.send(headers);
            stream
                .write_all(response.as_bytes())
                .await
                .expect("write fixture response");
            stream.shutdown().await.expect("close fixture response");
        });
        (format!("http://{address}"), headers_rx, server)
    }

    #[derive(Debug)]
    struct WebSocketCapture {
        method: String,
        path: String,
        authorization: Option<String>,
        beta: Option<String>,
        payload: Value,
    }

    async fn spawn_websocket_fixture(
    ) -> (String, oneshot::Receiver<WebSocketCapture>, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind websocket fixture");
        let address = listener.local_addr().expect("websocket fixture address");
        let (capture_tx, capture_rx) = oneshot::channel();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept websocket request");
            let handshake = Arc::new(std::sync::Mutex::new(None));
            let handshake_capture = Arc::clone(&handshake);
            let mut socket = tokio_tungstenite::accept_hdr_async(stream, move |request: &tokio_tungstenite::tungstenite::handshake::server::Request, mut response: tokio_tungstenite::tungstenite::handshake::server::Response| {
                *handshake_capture.lock().expect("handshake capture") = Some((
                    request.method().to_string(),
                    request.uri().path().to_string(),
                    request.headers().get("Authorization").and_then(|value| value.to_str().ok()).map(str::to_string),
                    request.headers().get("OpenAI-Beta").and_then(|value| value.to_str().ok()).map(str::to_string),
                ));
                response.headers_mut().insert(
                    "OpenAI-Model",
                    HeaderValue::from_static("gpt-5-codex"),
                );
                Ok(response)
            })
            .await
            .expect("websocket handshake");
            let request = socket
                .next()
                .await
                .expect("websocket request frame")
                .expect("valid websocket request frame");
            let Message::Text(request) = request else {
                panic!("expected text websocket request");
            };
            let payload = serde_json::from_str(&request).expect("websocket request json");
            let (method, path, authorization, beta) = handshake
                .lock()
                .expect("handshake capture")
                .take()
                .expect("captured websocket handshake");
            let _ = capture_tx.send(WebSocketCapture {
                method,
                path,
                authorization,
                beta,
                payload,
            });
            socket
                .send(Message::Text(
                    json!({
                        "type": "response.metadata",
                        "headers": { "X-OpenAI-Model": ["gpt-5-codex"] },
                        "metadata": {
                            "openai_verification_recommendation": [
                                "trusted_access_for_cyber",
                                "unknown"
                            ]
                        }
                    })
                    .to_string(),
                ))
                .await
                .expect("send websocket metadata");
            socket
                .send(Message::Text(
                    json!({
                        "type": "response.completed",
                        "response": { "id": "resp-ws-1", "output": [] }
                    })
                    .to_string(),
                ))
                .await
                .expect("send websocket response");
        });
        (format!("http://{address}"), capture_rx, server)
    }

    async fn spawn_http_method_fixture(
        responses: Vec<String>,
    ) -> (String, Arc<std::sync::Mutex<Vec<String>>>, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind method fixture");
        let address = listener.local_addr().expect("method fixture address");
        let methods = Arc::new(std::sync::Mutex::new(Vec::new()));
        let method_capture = Arc::clone(&methods);
        let server = tokio::spawn(async move {
            for response in responses {
                let (mut stream, _) = listener.accept().await.expect("accept fixture request");
                let headers = read_http_headers_text(&mut stream).await;
                let method = headers
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().next())
                    .unwrap_or_default()
                    .to_string();
                method_capture.lock().expect("method capture").push(method);
                stream
                    .write_all(response.as_bytes())
                    .await
                    .expect("write fixture response");
                stream.shutdown().await.expect("close fixture response");
            }
        });
        (format!("http://{address}"), methods, server)
    }

    async fn spawn_websocket_drop_then_http_fixture(
    ) -> (String, Arc<std::sync::Mutex<Vec<String>>>, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind websocket replay fixture");
        let address = listener
            .local_addr()
            .expect("websocket replay fixture address");
        let methods = Arc::new(std::sync::Mutex::new(Vec::new()));
        let method_capture = Arc::clone(&methods);
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept websocket request");
            let first_capture = Arc::clone(&method_capture);
            let socket = tokio_tungstenite::accept_hdr_async(stream, move |request: &tokio_tungstenite::tungstenite::handshake::server::Request, response: tokio_tungstenite::tungstenite::handshake::server::Response| {
                first_capture
                    .lock()
                    .expect("method capture")
                    .push(request.method().to_string());
                Ok(response)
            })
            .await
            .expect("websocket handshake");
            drop(socket);

            let (mut stream, _) = listener.accept().await.expect("accept HTTP replay");
            let headers = read_http_headers_text(&mut stream).await;
            let method = headers
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().next())
                .unwrap_or_default()
                .to_string();
            method_capture.lock().expect("method capture").push(method);
            let body = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-http-replay\",\"output\":[]}}\n\n";
            let response = fixture_response("200 OK", "Content-Type: text/event-stream\r\n", body);
            stream
                .write_all(response.as_bytes())
                .await
                .expect("write HTTP replay response");
            stream.shutdown().await.expect("close HTTP replay response");
        });
        (format!("http://{address}"), methods, server)
    }

    async fn read_http_headers(stream: &mut tokio::net::TcpStream) {
        let _ = read_http_headers_text(stream).await;
    }

    async fn read_http_headers_text(stream: &mut tokio::net::TcpStream) -> String {
        let mut received = Vec::new();
        let mut buffer = [0_u8; 1024];
        while !received.windows(4).any(|window| window == b"\r\n\r\n") {
            let read = stream
                .read(&mut buffer)
                .await
                .expect("read fixture request");
            if read == 0 {
                break;
            }
            received.extend_from_slice(&buffer[..read]);
        }
        String::from_utf8_lossy(&received).into_owned()
    }

    fn fixture_response(status: &str, extra_headers: &str, body: &str) -> String {
        format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n{extra_headers}\r\n{body}",
            body.len()
        )
    }
}
