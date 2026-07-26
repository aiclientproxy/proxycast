pub mod action_required;
pub mod anthropic;
pub mod collaboration_mode;
pub mod context_trace;
pub mod input;
pub mod message_content;
pub mod model_context;
pub mod multi_agent_mode;
pub mod openai;
pub mod provider_trace;
pub mod response_item;
pub mod session_context;
pub mod thread;
pub mod turn_context;
pub mod world_state;

pub use collaboration_mode::{CollaborationMode, CollaborationModeSettings, ModeKind};
pub use input::{AgentInput, AgentInputError, ByteRange, ImageDetail, TextElement};
pub use multi_agent_mode::MultiAgentMode;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

macro_rules! id_type {
    ($name:ident) => {
        #[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self::new(value)
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self::new(value)
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                self.as_str()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(self.as_str())
            }
        }
    };
}

id_type!(SessionId);
id_type!(ThreadId);
id_type!(TurnId);
id_type!(RuntimeEventId);
id_type!(ToolCallId);
id_type!(ActionId);
id_type!(ArtifactId);
id_type!(ModelId);

/// Stable canonical identity for a persisted ThreadItem.
///
/// Adapted from Codex `ResponseItemId` (`5c19155cbd93bfa099016e7487259f61669823ff`),
/// Apache-2.0; see the repository's recorded provenance and NOTICE. New IDs are
/// always prefixed, while deserialization remains permissive for legacy history.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, JsonSchema)]
#[serde(transparent)]
pub struct ItemId(String);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ItemIdParseError {
    value: String,
}

impl fmt::Display for ItemIdParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "item id must contain a non-empty domain prefix: {}",
            self.value
        )
    }
}

impl std::error::Error for ItemIdParseError {}

static NEXT_ITEM_ID: AtomicU64 = AtomicU64::new(0);

impl ItemId {
    pub const PREFIX: &'static str = "item";

    /// Canonicalize a caller-provided suffix into the `item_` domain.
    ///
    /// This deliberately accepts an already-prefixed value so event adapters can
    /// pass through canonical IDs without creating a second prefix.
    pub fn new(value: impl Into<String>) -> Self {
        let value = value.into();
        if Self::is_prefixed_value(&value) {
            Self(value)
        } else {
            let suffix = value.trim_matches('_');
            if suffix.is_empty() {
                return Self::generated();
            }
            Self(format!("{}_{}", Self::PREFIX, suffix))
        }
    }

    /// Generate a fresh prefixed ID without relying on a provider-specific ID.
    pub fn generated() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_default();
        let counter = NEXT_ITEM_ID.fetch_add(1, Ordering::Relaxed);
        Self(format!("{}_{}_{counter:x}", Self::PREFIX, nanos))
    }

    pub fn with_suffix(suffix: impl fmt::Display) -> Self {
        Self::new(suffix.to_string())
    }

    /// Explicit legacy boundary. Legacy IDs are readable but must not be used
    /// for newly materialized items without passing through `new`/`generated`.
    pub fn from_legacy(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn parse(value: impl Into<String>) -> Result<Self, ItemIdParseError> {
        let value = value.into();
        if Self::is_prefixed_value(&value) {
            Ok(Self(value))
        } else {
            Err(ItemIdParseError { value })
        }
    }

    pub fn is_prefixed(&self) -> bool {
        Self::is_prefixed_value(&self.0)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn is_prefixed_value(value: &str) -> bool {
        value
            .split_once('_')
            .is_some_and(|(prefix, suffix)| !prefix.is_empty() && !suffix.is_empty())
    }
}

impl<'de> Deserialize<'de> for ItemId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // History imports may contain IDs from before the prefix contract. Keep
        // that read path permissive; all new writes use `new` or `generated`.
        String::deserialize(deserializer).map(Self::from_legacy)
    }
}

impl From<String> for ItemId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for ItemId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl AsRef<str> for ItemId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for ItemId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeEventKind {
    SessionStarted,
    TurnAccepted,
    TurnQueued,
    TurnStarted,
    TurnInterrupted,
    TurnCancelled,
    ModelDelta,
    ItemCreated,
    ItemUpdated,
    ItemRemoved,
    ToolCallStarted,
    ToolCallCompleted,
    ActionRequested,
    TurnCompleted,
    TurnFailed,
    Diagnostic,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeEvent {
    pub id: RuntimeEventId,
    pub session_id: SessionId,
    pub thread_id: ThreadId,
    pub turn_id: Option<TurnId>,
    pub sequence: u64,
    pub kind: RuntimeEventKind,
    #[serde(default)]
    pub payload: Value,
}

impl RuntimeEvent {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.kind,
            RuntimeEventKind::TurnCompleted
                | RuntimeEventKind::TurnFailed
                | RuntimeEventKind::TurnInterrupted
                | RuntimeEventKind::TurnCancelled
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AgentTurnInput {
    pub session_id: SessionId,
    pub thread_id: ThreadId,
    pub turn_id: TurnId,
    pub text: String,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeSnapshot {
    pub session_id: SessionId,
    pub thread_id: ThreadId,
    pub latest_turn_id: Option<TurnId>,
    #[serde(default)]
    pub events: Vec<RuntimeEvent>,
    #[serde(default)]
    pub metadata: Value,
}

pub use message_content::{MessageContentPart, MessageContentReference};
pub use response_item::ResponseItem;
pub use thread::{
    ApprovalAction, ApprovalDecision, ApprovalScope, CollabAgentOperation, CollabAgentState,
    CollabAgentStatus, FileChange, FileChangeKind, FileChangeStatus, ItemKind, ItemStatus,
    PageCursor, PlanStep, PlanStepStatus, SortDirection, SubAgentActivityKind, Thread,
    ThreadActiveFlag, ThreadHistoryChange, ThreadHistoryChangeSet, ThreadItem, ThreadItemPayload,
    ThreadItemsListParams, ThreadItemsListResponse, ThreadListParams, ThreadListResponse,
    ThreadStatus, ThreadTurnsListParams, ThreadTurnsListResponse, ThreadTurnsView, ToolArgument,
    ToolOutput, Turn, TurnAdmissionState, TurnApprovalState, TurnError, TurnItemsView,
    TurnQueueState, TurnStatus,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_serialize_as_plain_strings() {
        let id = SessionId::new("session-1");
        let encoded = serde_json::to_string(&id).expect("session id should serialize");

        assert_eq!(encoded, "\"session-1\"");
        assert_eq!(id.as_str(), "session-1");
    }

    #[test]
    fn item_ids_canonicalize_new_writes_and_keep_legacy_explicit() {
        let generated = ItemId::generated();
        assert!(generated.is_prefixed());
        assert!(ItemId::new("suffix").is_prefixed());
        assert_eq!(ItemId::from_legacy("legacy-id").as_str(), "legacy-id");
        assert!(ItemId::parse("legacy-id").is_err());
        assert_eq!(
            ItemId::parse("msg_123").expect("prefixed id").as_str(),
            "msg_123"
        );
    }
}
