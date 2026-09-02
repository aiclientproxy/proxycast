pub mod audio;
pub mod canonical;
pub mod current_client;
pub mod embedding;
pub mod http;
pub mod lowering;
pub mod provider_capabilities;
pub mod provider_stream;
pub mod provider_url;
pub mod reasoning_effort;
pub mod runtime_provider;
pub mod safety;
pub mod video;

use agent_protocol::ModelId;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelProviderProtocol {
    Responses,
    ChatCompletions,
    AnthropicMessages,
    GeminiGenerateContent,
    Custom(String),
}

impl ModelProviderProtocol {
    pub fn uses_responses_api(&self) -> bool {
        matches!(self, Self::Responses)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelRoute {
    pub provider: String,
    pub model: ModelId,
    pub protocol: ModelProviderProtocol,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub metadata: Value,
}
