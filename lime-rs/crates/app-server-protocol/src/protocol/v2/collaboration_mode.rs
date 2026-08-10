use agent_protocol::ModeKind;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// EXPERIMENTAL - list collaboration mode presets.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CollaborationModeListParams {}

/// EXPERIMENTAL - collaboration mode preset metadata for clients.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CollaborationModeMask {
    pub name: String,
    pub mode: Option<ModeKind>,
    pub model: Option<String>,
    #[serde(rename = "reasoning_effort")]
    pub reasoning_effort: Option<Option<String>>,
}

/// EXPERIMENTAL - collaboration mode presets response.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CollaborationModeListResponse {
    pub data: Vec<CollaborationModeMask>,
}
