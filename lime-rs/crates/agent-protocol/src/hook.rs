// Adapted from Codex hook protocol contracts
// (c4f42d161ae44a8d696ee9fb595709661979d187), Apache-2.0.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum HookEventName {
    PreToolUse,
    PermissionRequest,
    PostToolUse,
    PreCompact,
    PostCompact,
    SessionStart,
    SessionEnd,
    UserPromptSubmit,
    SubagentStart,
    SubagentStop,
    Stop,
}

impl HookEventName {
    pub fn key_label(self) -> &'static str {
        match self {
            Self::PreToolUse => "pre_tool_use",
            Self::PermissionRequest => "permission_request",
            Self::PostToolUse => "post_tool_use",
            Self::PreCompact => "pre_compact",
            Self::PostCompact => "post_compact",
            Self::SessionStart => "session_start",
            Self::SessionEnd => "session_end",
            Self::UserPromptSubmit => "user_prompt_submit",
            Self::SubagentStart => "subagent_start",
            Self::SubagentStop => "subagent_stop",
            Self::Stop => "stop",
        }
    }

    pub fn scope(self) -> HookScope {
        match self {
            Self::SessionStart | Self::SessionEnd | Self::SubagentStart => HookScope::Thread,
            Self::PreToolUse
            | Self::PermissionRequest
            | Self::PostToolUse
            | Self::PreCompact
            | Self::PostCompact
            | Self::UserPromptSubmit
            | Self::SubagentStop
            | Self::Stop => HookScope::Turn,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum HookHandlerType {
    Command,
    Prompt,
    Agent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum HookExecutionMode {
    Sync,
    Async,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum HookScope {
    Thread,
    Turn,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum HookSource {
    System,
    User,
    Project,
    Mdm,
    SessionFlags,
    Plugin,
    CloudRequirements,
    CloudManagedConfig,
    LegacyManagedConfigFile,
    LegacyManagedConfigMdm,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum HookTrustStatus {
    Managed,
    Untrusted,
    Trusted,
    Modified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum HookRunStatus {
    Running,
    Completed,
    Failed,
    Blocked,
    Stopped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum HookOutputEntryKind {
    Warning,
    Stop,
    Feedback,
    Context,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct HookOutputEntry {
    pub kind: HookOutputEntryKind,
    pub text: String,
}

/// Immutable definition captured for one provider sampling step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct HookSnapshot {
    pub key: String,
    pub event_name: HookEventName,
    pub handler_type: HookHandlerType,
    pub execution_mode: HookExecutionMode,
    pub matcher: Option<String>,
    pub command: Option<String>,
    pub timeout_sec: u64,
    pub status_message: Option<String>,
    pub additional_context_limit: Option<usize>,
    pub source_path: PathBuf,
    pub source: HookSource,
    pub plugin_id: Option<String>,
    pub display_order: i64,
    pub enabled: bool,
    pub is_managed: bool,
    pub current_hash: String,
    pub trust_status: HookTrustStatus,
}

impl HookSnapshot {
    pub fn scope(&self) -> HookScope {
        self.event_name.scope()
    }

    pub fn run_id(&self) -> String {
        format!(
            "{}:{}:{}",
            self.event_name.key_label().replace('_', "-"),
            self.display_order,
            self.source_path.display()
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct HookRunSummary {
    pub id: String,
    pub event_name: HookEventName,
    pub handler_type: HookHandlerType,
    pub execution_mode: HookExecutionMode,
    pub scope: HookScope,
    pub source_path: PathBuf,
    #[serde(default)]
    pub source: HookSource,
    pub display_order: i64,
    pub status: HookRunStatus,
    pub status_message: Option<String>,
    pub started_at: i64,
    pub completed_at: Option<i64>,
    pub duration_ms: Option<i64>,
    pub entries: Vec<HookOutputEntry>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn canonical_hook_wire_matches_codex_core_contract() {
        assert_eq!(
            serde_json::to_value(HookEventName::PreToolUse).expect("event wire"),
            json!("pre_tool_use")
        );
        assert_eq!(HookEventName::SessionEnd.scope(), HookScope::Thread);
        assert_eq!(HookEventName::PostToolUse.scope(), HookScope::Turn);
    }
}
