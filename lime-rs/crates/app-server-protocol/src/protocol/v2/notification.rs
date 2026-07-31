use super::TurnError;
use crate::RequestId;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Signals that the process-wide Skill catalog must be read again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SkillsChangedNotification {}

/// Reports a transient retry or terminal error for the current turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct ErrorNotification {
    pub error: TurnError,
    /// True only when App Server has already decided to retry this turn.
    pub will_retry: bool,
    pub thread_id: String,
    pub turn_id: String,
}

/// Reports the latest turn-scoped execution checklist.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct TurnPlanUpdatedNotification {
    pub thread_id: String,
    pub turn_id: String,
    #[serde(default)]
    pub explanation: Option<String>,
    pub plan: Vec<TurnPlanStep>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct TurnPlanStep {
    pub step: String,
    pub status: TurnPlanStepStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum TurnPlanStepStatus {
    Pending,
    InProgress,
    Completed,
}

/// A user-visible runtime warning projected from the current event stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct WarningNotification {
    /// Optional thread target when the warning applies to a specific thread.
    pub thread_id: Option<String>,
    /// Concise warning message for the user.
    pub message: String,
    /// Lime-owned warning code used for localized presentation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

/// Emitted after a server-initiated request reaches a terminal state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ServerRequestResolvedNotification {
    pub thread_id: String,
    pub request_id: RequestId,
}
