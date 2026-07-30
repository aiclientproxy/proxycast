//! Codex Responses API response-item union used by history-only injection.
//!
//! The public App Server method intentionally keeps `items` as JSON values. This
//! module owns the runtime validation contract while retaining the original JSON
//! for provider lowering, so unknown provider fields are not rewritten on write.

use crate::ImageDetail;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MessagePhase {
    Commentary,
    FinalAnswer,
}

impl MessagePhase {
    pub fn from_wire(value: &str) -> Option<Self> {
        match value.trim() {
            "commentary" => Some(Self::Commentary),
            "final" | "finalAnswer" | "final_answer" => Some(Self::FinalAnswer),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct InternalChatMessageMetadataPassthrough {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentItem {
    InputText {
        text: String,
    },
    InputImage {
        image_url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
    },
    InputAudio {
        audio_url: String,
    },
    OutputText {
        text: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentMessageInputContent {
    InputText { text: String },
    EncryptedContent { encrypted_content: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum LocalShellStatus {
    Completed,
    InProgress,
    Incomplete,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LocalShellAction {
    Exec(LocalShellExecAction),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LocalShellExecAction {
    pub command: Vec<String>,
    pub timeout_ms: Option<u64>,
    pub working_directory: Option<String>,
    pub env: Option<std::collections::HashMap<String, String>>,
    pub user: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WebSearchAction {
    Search {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        query: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        queries: Option<Vec<String>>,
    },
    OpenPage {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        url: Option<String>,
    },
    FindInPage {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pattern: Option<String>,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningItemReasoningSummary {
    SummaryText { text: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningItemContent {
    ReasoningText { text: String },
    Text { text: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FunctionCallOutputContentItem {
    InputText {
        text: String,
    },
    InputImage {
        image_url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
    },
    InputAudio {
        audio_url: String,
    },
    EncryptedContent {
        encrypted_content: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum FunctionCallOutputBody {
    Text(String),
    ContentItems(Vec<FunctionCallOutputContentItem>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseItem {
    #[serde(rename = "custom_tool_call")]
    CustomToolCall {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        call_id: String,
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        namespace: Option<String>,
        input: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        internal_chat_message_metadata_passthrough: Option<InternalChatMessageMetadataPassthrough>,
    },
    AdditionalTools {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        role: String,
        tools: Vec<Value>,
    },
    Message {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        role: String,
        content: Vec<ContentItem>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        phase: Option<MessagePhase>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        internal_chat_message_metadata_passthrough: Option<InternalChatMessageMetadataPassthrough>,
    },
    AgentMessage {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        author: String,
        recipient: String,
        content: Vec<AgentMessageInputContent>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        internal_chat_message_metadata_passthrough: Option<InternalChatMessageMetadataPassthrough>,
    },
    Reasoning {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        summary: Vec<ReasoningItemReasoningSummary>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<Vec<ReasoningItemContent>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        internal_chat_message_metadata_passthrough: Option<InternalChatMessageMetadataPassthrough>,
    },
    LocalShellCall {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
        status: LocalShellStatus,
        action: LocalShellAction,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        internal_chat_message_metadata_passthrough: Option<InternalChatMessageMetadataPassthrough>,
    },
    FunctionCall {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        namespace: Option<String>,
        arguments: String,
        call_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        internal_chat_message_metadata_passthrough: Option<InternalChatMessageMetadataPassthrough>,
    },
    ToolSearchCall {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        execution: String,
        arguments: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        internal_chat_message_metadata_passthrough: Option<InternalChatMessageMetadataPassthrough>,
    },
    FunctionCallOutput {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        call_id: String,
        output: FunctionCallOutputBody,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        internal_chat_message_metadata_passthrough: Option<InternalChatMessageMetadataPassthrough>,
    },
    CustomToolCallOutput {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        call_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        output: FunctionCallOutputBody,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        internal_chat_message_metadata_passthrough: Option<InternalChatMessageMetadataPassthrough>,
    },
    ToolSearchOutput {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        call_id: Option<String>,
        status: String,
        execution: String,
        tools: Vec<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        internal_chat_message_metadata_passthrough: Option<InternalChatMessageMetadataPassthrough>,
    },
    WebSearchCall {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        action: Option<WebSearchAction>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        internal_chat_message_metadata_passthrough: Option<InternalChatMessageMetadataPassthrough>,
    },
    ImageGenerationCall {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        status: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        revised_prompt: Option<String>,
        result: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        internal_chat_message_metadata_passthrough: Option<InternalChatMessageMetadataPassthrough>,
    },
    Compaction {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        encrypted_content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        internal_chat_message_metadata_passthrough: Option<InternalChatMessageMetadataPassthrough>,
    },
    CompactionTrigger {},
    ContextCompaction {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        internal_chat_message_metadata_passthrough: Option<InternalChatMessageMetadataPassthrough>,
    },
    #[serde(other)]
    Other,
}

impl ResponseItem {
    /// Mirrors Codex's image guard before an item reaches a provider.
    pub fn contains_remote_image_url(&self) -> bool {
        let is_remote = |url: &str| {
            let url = url.trim().to_ascii_lowercase();
            url.starts_with("http://") || url.starts_with("https://")
        };
        match self {
            Self::Message { content, .. } => content.iter().any(|item| {
                matches!(item, ContentItem::InputImage { image_url, .. } if is_remote(image_url))
            }),
            Self::FunctionCallOutput { output, .. } | Self::CustomToolCallOutput { output, .. } => {
                matches!(output, FunctionCallOutputBody::ContentItems(items) if items.iter().any(|item| matches!(item, FunctionCallOutputContentItem::InputImage { image_url, .. } if is_remote(image_url))))
            }
            _ => false,
        }
    }
}
