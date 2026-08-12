//! Provider-neutral content, request, and stream contracts.
//!
//! This module is deliberately independent of any provider wire format.  A
//! provider adapter may lower these values to a native request, but neither
//! the GUI nor the runtime history should carry that native shape.

use agent_protocol::ImageDetail;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub type ProviderMetadata = BTreeMap<String, Value>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentPartError {
    EmptyMediaUri,
    EmptyMediaType,
    InlineMediaForbidden,
}

impl std::fmt::Display for ContentPartError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::EmptyMediaUri => "canonical media URI must not be empty",
            Self::EmptyMediaType => "canonical media type must not be empty",
            Self::InlineMediaForbidden => "canonical media must use a URI or sidecar reference",
        })
    }
}

impl std::error::Error for ContentPartError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ContentPart {
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        metadata: ProviderMetadata,
    },
    Media {
        media_type: String,
        /// A URI or sidecar reference. Inline bytes are intentionally absent.
        uri: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        byte_size: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sha256: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sidecar_ref: Option<String>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        metadata: ProviderMetadata,
    },
    Reasoning {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        encrypted: Option<String>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        metadata: ProviderMetadata,
    },
    ToolCall {
        id: String,
        name: String,
        input: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_executed: Option<bool>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        metadata: ProviderMetadata,
    },
    CustomToolCall {
        id: String,
        name: String,
        input: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        namespace: Option<String>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        metadata: ProviderMetadata,
    },
    ToolResult {
        id: String,
        name: String,
        result: ToolResultValue,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_executed: Option<bool>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        metadata: ProviderMetadata,
    },
    CustomToolResult {
        id: String,
        name: String,
        result: ToolResultValue,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        metadata: ProviderMetadata,
    },
    /// A validated Responses API item that must retain its exact provider wire shape.
    RawResponseItem { item: Value },
}

impl ContentPart {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            metadata: ProviderMetadata::new(),
        }
    }

    pub fn media(
        uri: impl Into<String>,
        media_type: impl Into<String>,
    ) -> Result<Self, ContentPartError> {
        Self::media_with_detail(uri, media_type, None)
    }

    pub fn media_with_detail(
        uri: impl Into<String>,
        media_type: impl Into<String>,
        detail: Option<ImageDetail>,
    ) -> Result<Self, ContentPartError> {
        let uri = uri.into();
        let media_type = media_type.into();
        if uri.trim().is_empty() {
            return Err(ContentPartError::EmptyMediaUri);
        }
        if media_type.trim().is_empty() {
            return Err(ContentPartError::EmptyMediaType);
        }
        if uri.trim_start().starts_with("data:") {
            return Err(ContentPartError::InlineMediaForbidden);
        }
        Ok(Self::Media {
            media_type,
            uri,
            detail,
            filename: None,
            byte_size: None,
            sha256: None,
            sidecar_ref: None,
            metadata: ProviderMetadata::new(),
        })
    }

    pub fn is_inline_data_uri(&self) -> bool {
        matches!(self, Self::Media { uri, .. } if uri.trim_start().starts_with("data:"))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultValue {
    Json { value: Value },
    Text { value: String },
    Error { value: Value },
    Content { value: Vec<ContentPart> },
}

impl ToolResultValue {
    pub fn text(value: impl Into<String>) -> Self {
        Self::Text {
            value: value.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub role: Role,
    #[serde(default)]
    pub content: Vec<ContentPart>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: ProviderMetadata,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            id: None,
            role: Role::User,
            content: vec![ContentPart::text(content)],
            metadata: ProviderMetadata::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreeformToolFormat {
    pub r#type: String,
    pub syntax: String,
    pub definition: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolDefinition {
    Function {
        name: String,
        #[serde(default)]
        description: String,
        input_schema: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output_schema: Option<Value>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        metadata: ProviderMetadata,
    },
    Custom {
        name: String,
        #[serde(default)]
        description: String,
        format: FreeformToolFormat,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        metadata: ProviderMetadata,
    },
}

impl ToolDefinition {
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
    ) -> Self {
        Self::Function {
            name: name.into(),
            description: description.into(),
            input_schema,
            output_schema: None,
            metadata: ProviderMetadata::new(),
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
            metadata: ProviderMetadata::new(),
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Function { name, .. } | Self::Custom { name, .. } => name,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct GenerationOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Request {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub model: String,
    #[serde(default)]
    pub system: Vec<ContentPart>,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    #[serde(default)]
    pub generation: GenerationOptions,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub provider_options: ProviderMetadata,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: ProviderMetadata,
}

impl Request {
    pub fn text(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            id: None,
            model: model.into(),
            system: Vec::new(),
            messages: vec![Message::user(prompt)],
            tools: Vec::new(),
            tool_choice: None,
            generation: GenerationOptions::default(),
            provider_options: ProviderMetadata::new(),
            metadata: ProviderMetadata::new(),
        }
    }

    pub fn has_input(&self) -> bool {
        !self.system.is_empty()
            || self
                .messages
                .iter()
                .any(|message| !message.content.is_empty())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    ToolCall,
    Length,
    ContentFilter,
    Error,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Usage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub non_cached_input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub provider_metadata: ProviderMetadata,
}

impl Usage {
    pub fn breakdown_is_consistent(&self) -> bool {
        let Some(input) = self.input_tokens else {
            return true;
        };
        let breakdown = self
            .non_cached_input_tokens
            .unwrap_or_default()
            .saturating_add(self.cache_read_input_tokens.unwrap_or_default())
            .saturating_add(self.cache_write_input_tokens.unwrap_or_default());
        breakdown == input
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelVerification {
    TrustedAccessForCyber,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelRerouteReason {
    HighRiskCyberActivity,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum LlmEvent {
    StepStart {
        index: u32,
    },
    TextStart {
        id: String,
    },
    TextDelta {
        id: String,
        text: String,
    },
    TextEnd {
        id: String,
    },
    ReasoningStart {
        id: String,
    },
    ReasoningSummaryDelta {
        id: String,
        text: String,
        summary_index: i64,
    },
    ReasoningSummaryPartAdded {
        id: String,
        summary_index: i64,
    },
    ReasoningContentDelta {
        id: String,
        text: String,
        content_index: i64,
    },
    ReasoningEnd {
        id: String,
    },
    ToolInputStart {
        id: String,
        name: String,
    },
    ToolInputDelta {
        id: String,
        name: String,
        text: String,
    },
    ToolInputEnd {
        id: String,
        name: String,
    },
    ToolCall {
        id: String,
        name: String,
        input: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        raw_arguments: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_executed: Option<bool>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        provider_metadata: ProviderMetadata,
    },
    CustomToolCall {
        id: String,
        name: String,
        input: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        namespace: Option<String>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        provider_metadata: ProviderMetadata,
    },
    ToolResult {
        id: String,
        name: String,
        result: ToolResultValue,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_executed: Option<bool>,
    },
    ToolError {
        id: String,
        name: String,
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        classification: Option<FailureClassification>,
    },
    Usage {
        usage: Usage,
    },
    ServerModel {
        model: String,
    },
    ModelReroute {
        from_model: String,
        to_model: String,
        reason: ModelRerouteReason,
    },
    ModelVerification {
        verifications: Vec<ModelVerification>,
    },
    TurnModerationMetadata {
        metadata: Value,
    },
    StepFinish {
        index: u32,
        reason: FinishReason,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
    },
    Finish {
        reason: FinishReason,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        response_id: Option<String>,
    },
    ProviderError {
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        classification: Option<FailureClassification>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retryable: Option<bool>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FailureClassification {
    Authentication,
    Permission,
    RateLimit,
    Quota,
    InvalidRequest,
    ContextOverflow,
    ContentPolicy,
    ProviderInternal,
    Transport,
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn media_part_is_reference_only() {
        let part = ContentPart::media("sidecar://asset-1", "image/png").expect("media");
        let value = serde_json::to_value(&part).expect("media serializes");
        assert_eq!(value["type"], json!("media"));
        assert_eq!(value["uri"], json!("sidecar://asset-1"));
        assert!(value.get("detail").is_none());
        assert!(value.get("data").is_none());
        assert!(!part.is_inline_data_uri());
    }

    #[test]
    fn media_part_preserves_typed_image_detail() {
        let part = ContentPart::media_with_detail(
            "sidecar://asset-1",
            "image/png",
            Some(ImageDetail::Original),
        )
        .expect("media");
        let value = serde_json::to_value(&part).expect("media serializes");

        assert_eq!(value["detail"], json!("original"));
        assert_eq!(
            serde_json::from_value::<ContentPart>(value).expect("media deserializes"),
            part
        );
    }

    #[test]
    fn media_rejects_inline_data_and_empty_references() {
        assert_eq!(
            ContentPart::media("data:image/png;base64,AAAA", "image/png"),
            Err(ContentPartError::InlineMediaForbidden)
        );
        assert_eq!(
            ContentPart::media("", "image/png"),
            Err(ContentPartError::EmptyMediaUri)
        );
        assert_eq!(
            ContentPart::media("sidecar://asset-1", ""),
            Err(ContentPartError::EmptyMediaType)
        );
    }

    #[test]
    fn request_and_tool_parts_preserve_provider_execution_metadata() {
        let part = ContentPart::ToolCall {
            id: "call-1".to_string(),
            name: "search".to_string(),
            input: json!({"q": "lime"}),
            provider_executed: Some(true),
            metadata: ProviderMetadata::new(),
        };
        let request = Request {
            messages: vec![Message {
                id: Some("message-1".to_string()),
                role: Role::Assistant,
                content: vec![part],
                metadata: ProviderMetadata::new(),
            }],
            ..Request::text("model-1", "hello")
        };
        let value = serde_json::to_value(&request).expect("request serializes");
        assert_eq!(
            value["messages"][0]["content"][0]["provider_executed"],
            json!(true)
        );
    }

    #[test]
    fn usage_breakdown_is_checked_without_subtraction() {
        let usage = Usage {
            input_tokens: Some(10),
            non_cached_input_tokens: Some(6),
            cache_read_input_tokens: Some(3),
            cache_write_input_tokens: Some(1),
            ..Usage::default()
        };
        assert!(usage.breakdown_is_consistent());
        assert!(!Usage {
            cache_read_input_tokens: Some(2),
            ..usage
        }
        .breakdown_is_consistent());
    }

    #[test]
    fn event_algebra_keeps_block_ids_and_failure_classification() {
        let event = LlmEvent::ProviderError {
            message: "rate limited".to_string(),
            classification: Some(FailureClassification::RateLimit),
            retryable: Some(true),
        };
        let value = serde_json::to_value(event).expect("event serializes");
        assert_eq!(value["type"], json!("provider-error"));
        assert_eq!(value["classification"], json!("rate-limit"));
        assert_eq!(value["retryable"], json!(true));
    }
}
