use agent_protocol::hook as core;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

macro_rules! hook_enum {
    ($name:ident { $($variant:ident),+ $(,)? }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
        #[serde(rename_all = "camelCase")]
        pub enum $name { $($variant),+ }

        impl From<core::$name> for $name {
            fn from(value: core::$name) -> Self {
                match value { $(core::$name::$variant => Self::$variant),+ }
            }
        }

        impl From<$name> for core::$name {
            fn from(value: $name) -> Self {
                match value { $($name::$variant => Self::$variant),+ }
            }
        }
    };
}

hook_enum!(HookEventName {
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
});
hook_enum!(HookHandlerType {
    Command,
    Prompt,
    Agent,
});
hook_enum!(HookExecutionMode { Sync, Async });

impl Default for HookExecutionMode {
    fn default() -> Self {
        Self::Sync
    }
}
hook_enum!(HookScope { Thread, Turn });
hook_enum!(HookSource {
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
    Unknown,
});
hook_enum!(HookTrustStatus {
    Managed,
    Untrusted,
    Trusted,
    Modified,
});
hook_enum!(HookRunStatus {
    Running,
    Completed,
    Failed,
    Blocked,
    Stopped,
});
hook_enum!(HookOutputEntryKind {
    Warning,
    Stop,
    Feedback,
    Context,
    Error,
});

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct HooksListParams {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cwds: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct HooksListResponse {
    pub data: Vec<HooksListEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct HooksListEntry {
    pub cwd: PathBuf,
    pub hooks: Vec<HookMetadata>,
    pub warnings: Vec<String>,
    pub errors: Vec<HookErrorInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct HookMetadata {
    pub key: String,
    pub event_name: HookEventName,
    pub handler_type: HookHandlerType,
    #[serde(default)]
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

impl From<&core::HookSnapshot> for HookMetadata {
    fn from(value: &core::HookSnapshot) -> Self {
        Self {
            key: value.key.clone(),
            event_name: value.event_name.into(),
            handler_type: value.handler_type.into(),
            execution_mode: value.execution_mode.into(),
            matcher: value.matcher.clone(),
            command: value.command.clone(),
            timeout_sec: value.timeout_sec,
            status_message: value.status_message.clone(),
            additional_context_limit: value.additional_context_limit,
            source_path: value.source_path.clone(),
            source: value.source.into(),
            plugin_id: value.plugin_id.clone(),
            display_order: value.display_order,
            enabled: value.enabled,
            is_managed: value.is_managed,
            current_hash: value.current_hash.clone(),
            trust_status: value.trust_status.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct HookErrorInfo {
    pub path: PathBuf,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct HookOutputEntry {
    pub kind: HookOutputEntryKind,
    pub text: String,
}

impl From<&core::HookOutputEntry> for HookOutputEntry {
    fn from(value: &core::HookOutputEntry) -> Self {
        Self {
            kind: value.kind.into(),
            text: value.text.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct HookRunSummary {
    pub id: String,
    pub event_name: HookEventName,
    pub handler_type: HookHandlerType,
    pub execution_mode: HookExecutionMode,
    pub scope: HookScope,
    pub source_path: PathBuf,
    pub source: HookSource,
    pub display_order: i64,
    pub status: HookRunStatus,
    pub status_message: Option<String>,
    pub started_at: i64,
    pub completed_at: Option<i64>,
    pub duration_ms: Option<i64>,
    pub entries: Vec<HookOutputEntry>,
}

impl From<&core::HookRunSummary> for HookRunSummary {
    fn from(value: &core::HookRunSummary) -> Self {
        Self {
            id: value.id.clone(),
            event_name: value.event_name.into(),
            handler_type: value.handler_type.into(),
            execution_mode: value.execution_mode.into(),
            scope: value.scope.into(),
            source_path: value.source_path.clone(),
            source: value.source.into(),
            display_order: value.display_order,
            status: value.status.into(),
            status_message: value.status_message.clone(),
            started_at: value.started_at,
            completed_at: value.completed_at,
            duration_ms: value.duration_ms,
            entries: value.entries.iter().map(Into::into).collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct HookStartedNotification {
    pub thread_id: String,
    pub turn_id: Option<String>,
    pub run: HookRunSummary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct HookCompletedNotification {
    pub thread_id: String,
    pub turn_id: Option<String>,
    pub run: HookRunSummary,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn v2_hook_wire_uses_exact_camel_case_contract() {
        let params = HooksListParams {
            cwds: vec![PathBuf::from("/tmp/project")],
        };
        assert_eq!(
            serde_json::to_value(params).expect("params wire"),
            json!({"cwds":["/tmp/project"]})
        );
        assert_eq!(
            serde_json::to_value(HookEventName::PreToolUse).expect("event wire"),
            json!("preToolUse")
        );
        assert_eq!(
            serde_json::to_value(HookTrustStatus::Modified).expect("trust wire"),
            json!("modified")
        );
    }
}
