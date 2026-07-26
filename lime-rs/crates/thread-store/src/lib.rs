pub mod agent_graph;
pub mod agent_identity;
pub mod agent_mailbox;
pub mod history;
pub mod runtime_snapshot;
pub mod session_record;
pub mod session_repository;
pub mod store;
pub mod task_board;
pub mod types;

use std::error::Error;
use std::fmt;

pub use agent_graph::{
    AgentGraphStore, AgentGraphStoreFuture, ThreadSpawnEdgeStatus, ThreadSpawnParent,
};
pub use agent_identity::{
    canonical_agent_path_task_name, AgentIdentity, AgentIdentityStore, AgentIdentityStoreFuture,
};
pub use agent_mailbox::{
    AgentMailboxDeliveryMode, AgentMailboxDeliveryStatus, AgentMailboxMessage,
    AgentMailboxMessageKind, AgentMailboxResultStatus, AgentMailboxStore, AgentMailboxStoreFuture,
    AppendAgentMailboxMessageParams, PendingAgentMailboxTriggerRecipient,
};
pub use history::{
    CanonicalHistory, ThreadHistoryBuilder, ThreadHistoryBuilderError, ThreadHistoryPage,
};
pub use store::{ThreadStore, ThreadStoreFuture};
pub use types::{
    AppendThreadItemsParams, ApplyThreadHistoryParams, ApplyThreadHistoryResult,
    ArchiveThreadParams, ClearableField, CreateThreadParams, DeleteThreadParams, ItemPage,
    ListItemsParams, ListThreadsParams, ListTurnsParams, PageRequest, ReadThreadParams,
    SearchTextRange, SearchThreadOccurrencesParams, SearchThreadsParams, StoreCursor,
    StoredThreadOccurrence, StoredThreadSearchResult, ThreadMetadataPatch,
    ThreadOccurrenceSearchPage, ThreadPage, ThreadSearchPage, ThreadSearchSortKey,
    ThreadSearchSourceKind, TurnPage, UpdateThreadMetadataParams,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreadStoreErrorKind {
    InvalidRequest,
    Unsupported,
    ThreadNotFound,
    Internal,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThreadStoreError {
    kind: ThreadStoreErrorKind,
    message: String,
}

impl ThreadStoreError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            kind: ThreadStoreErrorKind::Internal,
            message: message.into(),
        }
    }

    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            kind: ThreadStoreErrorKind::InvalidRequest,
            message: message.into(),
        }
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self {
            kind: ThreadStoreErrorKind::Unsupported,
            message: message.into(),
        }
    }

    pub fn thread_not_found(message: impl Into<String>) -> Self {
        Self {
            kind: ThreadStoreErrorKind::ThreadNotFound,
            message: message.into(),
        }
    }

    pub fn kind(&self) -> ThreadStoreErrorKind {
        self.kind
    }
}

impl fmt::Display for ThreadStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for ThreadStoreError {}

pub type ThreadStoreResult<T> = Result<T, ThreadStoreError>;
