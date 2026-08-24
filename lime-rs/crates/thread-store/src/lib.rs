pub mod agent_graph;
pub mod agent_identity;
pub mod agent_mailbox;
pub mod history;
pub mod projects;
pub mod runtime_snapshot;
pub mod session_record;
pub mod session_repository;
pub mod store;
pub mod task_board;
pub mod types;

/// Stable UUIDv7 identifying the built-in pinned thread section.
pub const PINNED_THREAD_SECTION_ID: &str = "01984de2-8f74-7c91-a3b2-5c5e937cf318";

/// User-facing name of the built-in pinned thread section.
pub const PINNED_THREAD_SECTION_NAME: &str = "Pinned";

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
pub use projects::{
    CreateProjectParams, CreatedProject, DeletedProject, ListProjectsParams, MoveProjectParams,
    ProjectMoveOutcome, StoredProject, StoredProjectRoot, StoredProjectsPage, UpdateProjectParams,
    UpdatedProject,
};
pub use store::{ThreadStore, ThreadStoreFuture};
pub use types::{
    AppendThreadItemsParams, ApplyThreadHistoryParams, ApplyThreadHistoryResult,
    ArchiveThreadParams, ClearableField, CreateThreadParams, CreateThreadSectionParams,
    DeleteThreadParams, DeleteThreadSectionParams, ItemPage, ListItemsParams,
    ListThreadSectionsParams, ListThreadsParams, ListTurnsParams, MoveThreadToSectionParams,
    PageRequest, ReadThreadParams, RenameThreadSectionParams, SearchTextRange,
    SearchThreadOccurrencesParams, SearchThreadsParams, StoreCursor, StoredThreadOccurrence,
    StoredThreadSearchResult, StoredThreadSection, ThreadMetadataPatch, ThreadOccurrenceSearchPage,
    ThreadPage, ThreadSearchPage, ThreadSearchSortKey, ThreadSearchSourceKind, ThreadSectionPage,
    TurnPage, UpdateThreadMetadataParams,
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
