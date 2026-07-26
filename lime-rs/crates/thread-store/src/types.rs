// Adapted from Codex thread-store/types.rs
// (5c19155cbd93bfa099016e7487259f61669823ff), Apache-2.0; see repository NOTICE.

use std::fmt;

use agent_protocol::{
    SessionId, SortDirection, Thread, ThreadHistoryChangeSet, ThreadId, ThreadItem,
    ThreadTurnsView, Turn, TurnId, TurnItemsView,
};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

mod optional_option {
    use super::*;

    pub fn serialize<T, S>(value: &Option<Option<T>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        T: Serialize,
        S: Serializer,
    {
        match value {
            Some(value) => value.serialize(serializer),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, T, D>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
    where
        T: Deserialize<'de>,
        D: Deserializer<'de>,
    {
        Option::<T>::deserialize(deserializer).map(Some)
    }
}

/// Opaque continuation token minted and interpreted only by a store implementation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct StoreCursor(String);

impl StoreCursor {
    pub fn new(value: impl Into<String>) -> Result<Self, &'static str> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err("store cursor must not be empty");
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for StoreCursor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for StoreCursor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(D::Error::custom)
    }
}

/// Shared paging request. Cursor contents are never parsed outside the store implementation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageRequest {
    pub cursor: Option<StoreCursor>,
    pub limit: u32,
    pub sort_direction: SortDirection,
}

/// Parameters for creating the durable row for a canonical thread.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CreateThreadParams {
    /// The initial thread snapshot. Implementations must reject embedded turns; history is written
    /// through [`ApplyThreadHistoryParams`] so ordering and idempotency stay explicit.
    pub thread: Thread,
}

/// Parameters for reading a canonical thread.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadThreadParams {
    pub thread_id: ThreadId,
    pub include_archived: bool,
    pub turns_view: ThreadTurnsView,
}

/// Parameters for listing canonical threads.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListThreadsParams {
    pub include_archived: bool,
    pub page: PageRequest,
}

/// Parameters for listing turns in one thread.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListTurnsParams {
    pub thread_id: ThreadId,
    pub include_archived: bool,
    pub page: PageRequest,
    pub items_view: TurnItemsView,
}

/// Parameters for listing items in a thread, optionally restricted to one turn.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListItemsParams {
    pub thread_id: ThreadId,
    pub turn_id: Option<TurnId>,
    pub include_archived: bool,
    pub page: PageRequest,
}

/// A page of canonical threads.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThreadPage {
    pub data: Vec<Thread>,
    pub next_cursor: Option<StoreCursor>,
    pub backwards_cursor: Option<StoreCursor>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThreadSearchSortKey {
    CreatedAt,
    UpdatedAt,
    RecencyAt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThreadSearchSourceKind {
    Cli,
    VsCode,
    Exec,
    AppServer,
    SubAgent,
    SubAgentReview,
    SubAgentCompact,
    SubAgentThreadSpawn,
    SubAgentOther,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchThreadsParams {
    pub cursor: Option<StoreCursor>,
    pub page_size: usize,
    pub sort_key: ThreadSearchSortKey,
    pub sort_direction: SortDirection,
    pub source_kinds: Vec<ThreadSearchSourceKind>,
    pub archived: bool,
    pub search_term: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredThreadSearchResult {
    pub thread: Thread,
    pub snippet: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThreadSearchPage {
    pub data: Vec<StoredThreadSearchResult>,
    pub next_cursor: Option<StoreCursor>,
    pub backwards_cursor: Option<StoreCursor>,
}

/// A page of canonical turns.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TurnPage {
    pub data: Vec<Turn>,
    pub next_cursor: Option<StoreCursor>,
    pub backwards_cursor: Option<StoreCursor>,
}

/// A page of canonical items.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ItemPage {
    pub data: Vec<ThreadItem>,
    pub next_cursor: Option<StoreCursor>,
    pub backwards_cursor: Option<StoreCursor>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchThreadOccurrencesParams {
    pub thread_id: ThreadId,
    pub search_term: String,
    pub cursor: Option<StoreCursor>,
    pub page_size: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchTextRange {
    pub start: u32,
    pub end: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredThreadOccurrence {
    pub turn_id: TurnId,
    pub item_id: agent_protocol::ItemId,
    pub snippet: String,
    pub snippet_match_range: SearchTextRange,
    pub turn_cursor: StoreCursor,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThreadOccurrenceSearchPage {
    pub data: Vec<StoredThreadOccurrence>,
    pub next_cursor: Option<StoreCursor>,
}

/// Parameters for appending already-canonical items to one thread.
///
/// The outer sequence is the caller-owned durable event sequence used for
/// idempotency. Item contents are persisted as supplied; metadata and turn
/// snapshots must be changed through their dedicated APIs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppendThreadItemsParams {
    pub session_id: SessionId,
    pub thread_id: ThreadId,
    pub sequence: u64,
    pub items: Vec<ThreadItem>,
}

/// Parameters for atomically applying one canonical materializer change set.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ApplyThreadHistoryParams {
    pub session_id: SessionId,
    pub thread_id: ThreadId,
    pub changes: ThreadHistoryChangeSet,
}

/// Result of an idempotent history application.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApplyThreadHistoryResult {
    pub sequence: u64,
    pub applied: bool,
}

/// Optional field patch where omission leaves a value unchanged and `Some(None)` clears it.
pub type ClearableField<T> = Option<Option<T>>;

/// Literal patch for mutable thread metadata.
///
/// `None` leaves a field unchanged. Clearable fields use `Some(None)` to clear the value.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ThreadMetadataPatch {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "optional_option"
    )]
    pub name: ClearableField<String>,
    pub preview: Option<String>,
    pub model_provider: Option<String>,
    pub forked_from_id: Option<ThreadId>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "optional_option"
    )]
    pub product: ClearableField<String>,
    pub updated_at_ms: Option<i64>,
    pub advance_recency_at_ms: Option<i64>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "optional_option"
    )]
    pub metadata: ClearableField<Value>,
}

impl ThreadMetadataPatch {
    /// Merges `next` using field-presence semantics.
    pub fn merge(&mut self, next: Self) {
        if next.name.is_some() {
            self.name = next.name;
        }
        if next.preview.is_some() {
            self.preview = next.preview;
        }
        if next.model_provider.is_some() {
            self.model_provider = next.model_provider;
        }
        if next.forked_from_id.is_some() {
            self.forked_from_id = next.forked_from_id;
        }
        if next.product.is_some() {
            self.product = next.product;
        }
        if next.updated_at_ms.is_some() {
            self.updated_at_ms = next.updated_at_ms;
        }
        if next.advance_recency_at_ms.is_some() {
            self.advance_recency_at_ms = next.advance_recency_at_ms;
        }
        if next.metadata.is_some() {
            self.metadata = next.metadata;
        }
    }

    pub fn is_empty(&self) -> bool {
        self.name.is_none()
            && self.preview.is_none()
            && self.model_provider.is_none()
            && self.forked_from_id.is_none()
            && self.product.is_none()
            && self.updated_at_ms.is_none()
            && self.advance_recency_at_ms.is_none()
            && self.metadata.is_none()
    }
}

/// Parameters for patching mutable thread metadata.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UpdateThreadMetadataParams {
    pub thread_id: ThreadId,
    pub patch: ThreadMetadataPatch,
    pub include_archived: bool,
}

/// Parameters for archiving or unarchiving a thread.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchiveThreadParams {
    pub thread_id: ThreadId,
}

/// Parameters for deleting all persistence owned by a thread.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeleteThreadParams {
    pub thread_id: ThreadId,
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_protocol::ThreadItemPayload;
    use serde_json::json;

    #[test]
    fn cursor_is_opaque_and_rejects_empty_values() {
        let cursor = StoreCursor::new("store-owned-token").expect("valid cursor");
        assert_eq!(cursor.as_str(), "store-owned-token");
        assert_eq!(
            serde_json::to_value(&cursor).unwrap(),
            json!("store-owned-token")
        );
        assert!(StoreCursor::new("  ").is_err());
        assert!(serde_json::from_value::<StoreCursor>(json!("")).is_err());
    }

    #[test]
    fn metadata_patch_round_trips_clear_operations() {
        let forked_from_id = ThreadId::new("thread-parent");
        let patch = ThreadMetadataPatch {
            name: Some(None),
            product: Some(Some("chat".to_string())),
            forked_from_id: Some(forked_from_id.clone()),
            metadata: Some(None),
            ..Default::default()
        };

        let value = serde_json::to_value(&patch).expect("serialize patch");
        assert_eq!(value["name"], Value::Null);
        assert_eq!(value["product"], "chat");
        assert_eq!(value["forked_from_id"], forked_from_id.as_str());
        assert_eq!(value["metadata"], Value::Null);

        let decoded: ThreadMetadataPatch =
            serde_json::from_value(value).expect("deserialize patch");
        assert_eq!(decoded.name, Some(None));
        assert_eq!(decoded.product, Some(Some("chat".to_string())));
        assert_eq!(decoded.forked_from_id, Some(forked_from_id));
        assert_eq!(decoded.metadata, Some(None));
    }

    #[test]
    fn metadata_patch_merge_uses_presence_semantics() {
        let original_fork = ThreadId::new("thread-original-parent");
        let next_fork = ThreadId::new("thread-next-parent");
        let mut current = ThreadMetadataPatch {
            name: Some(Some("old".to_string())),
            preview: Some("keep".to_string()),
            forked_from_id: Some(original_fork),
            metadata: Some(Some(json!({"old": true}))),
            ..Default::default()
        };

        current.merge(ThreadMetadataPatch {
            name: Some(None),
            product: Some(Some("agent".to_string())),
            forked_from_id: Some(next_fork.clone()),
            metadata: Some(Some(json!({"new": true}))),
            ..Default::default()
        });

        assert_eq!(current.name, Some(None));
        assert_eq!(current.preview.as_deref(), Some("keep"));
        assert_eq!(current.product, Some(Some("agent".to_string())));
        assert_eq!(current.forked_from_id, Some(next_fork.clone()));
        assert_eq!(current.metadata, Some(Some(json!({"new": true}))));

        current.merge(ThreadMetadataPatch::default());
        assert_eq!(current.forked_from_id, Some(next_fork));
    }

    #[test]
    fn append_params_round_trip_canonical_items() {
        let params = AppendThreadItemsParams {
            session_id: SessionId::new("session-append"),
            thread_id: ThreadId::new("thread-append"),
            sequence: 7,
            items: vec![ThreadItem::new(
                SessionId::new("session-append"),
                ThreadId::new("thread-append"),
                TurnId::new("turn-append"),
                7,
                11,
                ThreadItemPayload::AgentMessage {
                    text: "canonical".to_string(),
                    phase: None,
                    content_parts: Vec::new(),
                },
            )],
        };
        let value = serde_json::to_value(&params).expect("serialize append params");
        let decoded: AppendThreadItemsParams =
            serde_json::from_value(value).expect("deserialize append params");
        assert_eq!(decoded, params);
    }
}
