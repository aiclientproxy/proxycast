// Adapted from Codex app-server-protocol v2 thread_data/turn/item contracts
// (5c19155cbd93bfa099016e7487259f61669823ff), Apache-2.0; see repository NOTICE.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use crate::response_item::MessagePhase;

use crate::{AgentInput, ItemId, MessageContentPart, SessionId, ThreadId, TurnId};

/// Codex runtime status for a thread. Persistence operations such as archive or
/// delete are commands, not runtime status values.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ThreadStatus {
    NotLoaded,
    Idle,
    SystemError,
    #[serde(rename_all = "camelCase")]
    Active {
        #[serde(default)]
        active_flags: Vec<ThreadActiveFlag>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum ThreadActiveFlag {
    WaitingOnApproval,
    WaitingOnUserInput,
}

impl ThreadStatus {
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Active { .. })
    }

    pub fn active_flags(&self) -> &[ThreadActiveFlag] {
        match self {
            Self::Active { active_flags } => active_flags,
            _ => &[],
        }
    }
}

/// Canonical turn result. Admission, queue and approval controls are carried by
/// their own fields on `Turn` and never become terminal result variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum TurnStatus {
    InProgress,
    Completed,
    Interrupted,
    Failed,
}

impl TurnStatus {
    pub fn is_terminal(self) -> bool {
        !matches!(self, Self::InProgress)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum TurnAdmissionState {
    #[default]
    Accepted,
    Rejected,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "state", rename_all = "camelCase")]
pub enum TurnQueueState {
    NotQueued,
    Queued {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        position: Option<u32>,
    },
    Running,
}

impl Default for TurnQueueState {
    fn default() -> Self {
        Self::NotQueued
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum TurnApprovalState {
    #[default]
    NotRequired,
    Pending,
    Resolved,
    Approved,
    Denied,
    Cancelled,
    TimedOut,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum ItemStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Interrupted,
    Cancelled,
}

impl ItemStatus {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Interrupted | Self::Cancelled
        )
    }
}

/// Stable family name used by projections and selectors. The payload below is
/// tagged and typed; consumers must not infer this value from rendered text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum ItemKind {
    UserMessage,
    AgentMessage,
    Plan,
    Reasoning,
    Tool,
    DynamicToolCall,
    McpToolCall,
    CollabAgentToolCall,
    Approval,
    Command,
    File,
    Media,
    SubAgent,
    ContextCompaction,
    Unknown,
    Extension,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ToolArgument {
    pub name: String,
    pub value: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ToolOutput {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_content: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    #[serde(default)]
    pub truncated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_ref: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct McpToolCallAppContext {
    pub connector_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource_uri: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub app_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action_name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum DynamicToolCallContentItem {
    InputText { text: String },
    InputImage { image_url: String },
    InputAudio { audio_url: String },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum ApprovalScope {
    #[default]
    Once,
    Turn,
    Session,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum CollabAgentOperation {
    Spawn,
    SendMessage,
    FollowUp,
    Wait,
    Interrupt,
    Resume,
    Close,
}

/// Last known runtime status for an AgentControl child.
///
/// This mirrors Codex app-server's `CollabAgentStatus`; the optional message is
/// carried separately so the wire shape stays stable across terminal states.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum CollabAgentStatus {
    PendingInit,
    Running,
    Interrupted,
    Completed,
    Errored,
    Shutdown,
    NotFound,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CollabAgentState {
    pub status: CollabAgentStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ApprovalAction {
    pub kind: String,
    pub description: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum ApprovalDecision {
    Approved,
    ApprovedForSession,
    Denied,
    TimedOut,
    Abort,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum FileChangeStatus {
    Proposed,
    Applied,
    Rejected,
    Failed,
}

/// Canonical patch change retained by the Thread/Turn/Item history owner.
///
/// `path` is always the source path for an update/move. `move_path` is the
/// destination, mirroring Codex's v2 `PatchChangeKind::Update` contract.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct FileChange {
    pub path: String,
    pub kind: FileChangeKind,
    pub diff: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum FileChangeKind {
    Add,
    Delete,
    Update {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        move_path: Option<String>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum SubAgentActivityKind {
    Started,
    Interacted,
    Interrupted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PlanStepStatus {
    Pending,
    InProgress,
    Completed,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct PlanStep {
    pub step: String,
    pub status: PlanStepStatus,
}

/// Typed Item payload union. Unknown upstream items retain only their type and
/// sanitized field names; raw values must not cross the canonical boundary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ThreadItemPayload {
    UserMessage {
        content: Vec<AgentInput>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        client_id: Option<String>,
    },
    AgentMessage {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        phase: Option<MessagePhase>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        content_parts: Vec<MessageContentPart>,
    },
    /// Proposed plan content. The completed snapshot is authoritative and may
    /// not equal the concatenation of preceding plan deltas.
    Plan {
        text: String,
        revision_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source: Option<String>,
        #[serde(default)]
        plan: Vec<PlanStep>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        explanation: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_call_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_item_id: Option<String>,
    },
    Reasoning {
        #[serde(default)]
        summary: Vec<String>,
        #[serde(default)]
        content: Vec<String>,
    },
    Tool {
        call_id: String,
        name: String,
        #[serde(default)]
        arguments: Vec<ToolArgument>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output: Option<ToolOutput>,
    },
    DynamicToolCall {
        call_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        namespace: Option<String>,
        tool: String,
        arguments: Value,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        content_items: Vec<DynamicToolCallContentItem>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        success: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
    },
    McpToolCall {
        call_id: String,
        server_name: String,
        tool_name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        app_context: Option<McpToolCallAppContext>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mcp_app_resource_uri: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        plugin_id: Option<String>,
        #[serde(default)]
        arguments: Vec<ToolArgument>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output: Option<ToolOutput>,
    },
    CollabAgentToolCall {
        call_id: String,
        operation: CollabAgentOperation,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        target_thread_id: Option<ThreadId>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output: Option<ToolOutput>,
        #[serde(default, skip_serializing_if = "HashMap::is_empty")]
        agent_states: HashMap<ThreadId, CollabAgentState>,
    },
    Approval {
        request_id: String,
        action: ApprovalAction,
        #[serde(default)]
        scope: ApprovalScope,
        #[serde(default)]
        available_decisions: Vec<ApprovalDecision>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        decision: Option<ApprovalDecision>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        requested_at_ms: Option<i64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        resolved_at_ms: Option<i64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason_code: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        expires_at_ms: Option<i64>,
    },
    Command {
        command: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cwd: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        exit_code: Option<i32>,
    },
    File {
        #[serde(default)]
        changes: Vec<FileChange>,
        status: FileChangeStatus,
    },
    Media {
        uri: String,
        mime_type: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        preview: Option<String>,
    },
    SubAgent {
        child_thread_id: ThreadId,
        activity: SubAgentActivityKind,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    ContextCompaction {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        summary: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        replacement_history: Vec<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        window_number: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        first_window_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        previous_window_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        window_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tail_start_turn_id: Option<String>,
    },
    Unknown {
        upstream_type: String,
        #[serde(default)]
        field_names: Vec<String>,
    },
    Extension {
        name: String,
        data: Value,
    },
}

impl ThreadItemPayload {
    pub fn kind(&self) -> ItemKind {
        match self {
            Self::UserMessage { .. } => ItemKind::UserMessage,
            Self::AgentMessage { .. } => ItemKind::AgentMessage,
            Self::Plan { .. } => ItemKind::Plan,
            Self::Reasoning { .. } => ItemKind::Reasoning,
            Self::Tool { .. } => ItemKind::Tool,
            Self::DynamicToolCall { .. } => ItemKind::DynamicToolCall,
            Self::McpToolCall { .. } => ItemKind::McpToolCall,
            Self::CollabAgentToolCall { .. } => ItemKind::CollabAgentToolCall,
            Self::Approval { .. } => ItemKind::Approval,
            Self::Command { .. } => ItemKind::Command,
            Self::File { .. } => ItemKind::File,
            Self::Media { .. } => ItemKind::Media,
            Self::SubAgent { .. } => ItemKind::SubAgent,
            Self::ContextCompaction { .. } => ItemKind::ContextCompaction,
            Self::Unknown { .. } => ItemKind::Unknown,
            Self::Extension { .. } => ItemKind::Extension,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct Thread {
    pub session_id: SessionId,
    pub thread_id: ThreadId,
    pub status: ThreadStatus,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    pub archived: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recency_at_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_thread_id: Option<ThreadId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_nickname: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_task_message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_state: Option<CollabAgentState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forked_from_id: Option<ThreadId>,
    #[serde(default)]
    pub preview: String,
    #[serde(default)]
    pub model_provider: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub product: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default)]
    pub metadata: Value,
    #[serde(default)]
    pub turns: Vec<Turn>,
    #[serde(default)]
    pub turns_view: ThreadTurnsView,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum ThreadTurnsView {
    NotLoaded,
    Summary,
    #[default]
    Full,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct Turn {
    pub session_id: SessionId,
    pub thread_id: ThreadId,
    pub turn_id: TurnId,
    pub status: TurnStatus,
    #[serde(default)]
    pub admission: TurnAdmissionState,
    #[serde(default)]
    pub queue: TurnQueueState,
    #[serde(default)]
    pub approval: TurnApprovalState,
    #[serde(default)]
    pub items: Vec<ThreadItem>,
    #[serde(default)]
    pub items_view: TurnItemsView,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<TurnError>,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
}

impl Turn {
    pub fn is_terminal(&self) -> bool {
        self.status.is_terminal()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum TurnItemsView {
    NotLoaded,
    Summary,
    #[default]
    Full,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TurnError {
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum SortDirection {
    Asc,
    #[default]
    Desc,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PageCursor {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    #[serde(default)]
    pub sort_direction: SortDirection,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadListParams {
    #[serde(flatten)]
    pub page: PageCursor,
    #[serde(default)]
    pub include_archived: bool,
    #[serde(default)]
    pub turns_view: ThreadTurnsView,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub section: Option<Option<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<Option<String>>,
    #[serde(default)]
    pub sort_by_section_position: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadReadParams {
    pub thread_id: ThreadId,
    #[serde(default)]
    pub turns_view: ThreadTurnsView,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadReadResponse {
    pub thread: Thread,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadListResponse {
    pub data: Vec<Thread>,
    pub next_cursor: Option<String>,
    pub backwards_cursor: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadTurnsListParams {
    pub thread_id: ThreadId,
    #[serde(flatten)]
    pub page: PageCursor,
    #[serde(default)]
    pub items_view: TurnItemsView,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadTurnsListResponse {
    pub data: Vec<Turn>,
    pub next_cursor: Option<String>,
    pub backwards_cursor: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadItemsListParams {
    pub thread_id: ThreadId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<TurnId>,
    #[serde(flatten)]
    pub page: PageCursor,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadItemsListResponse {
    pub data: Vec<ThreadItem>,
    pub next_cursor: Option<String>,
    pub backwards_cursor: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadItem {
    pub session_id: SessionId,
    pub thread_id: ThreadId,
    pub turn_id: TurnId,
    pub item_id: ItemId,
    pub sequence: u64,
    pub ordinal: u64,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at_ms: Option<i64>,
    pub kind: ItemKind,
    pub status: ItemStatus,
    pub payload: ThreadItemPayload,
    #[serde(default)]
    pub metadata: Value,
}

impl ThreadItem {
    pub fn new(
        session_id: SessionId,
        thread_id: ThreadId,
        turn_id: TurnId,
        sequence: u64,
        ordinal: u64,
        payload: ThreadItemPayload,
    ) -> Self {
        let kind = payload.kind();
        Self {
            session_id,
            thread_id,
            turn_id,
            item_id: ItemId::generated(),
            sequence,
            ordinal,
            created_at_ms: 0,
            updated_at_ms: 0,
            completed_at_ms: None,
            kind,
            status: ItemStatus::Pending,
            payload,
            metadata: Value::Null,
        }
    }

    pub fn key(&self) -> (&ThreadId, &TurnId, &ItemId) {
        (&self.thread_id, &self.turn_id, &self.item_id)
    }

    pub fn payload_kind(&self) -> ItemKind {
        self.payload.kind()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum ThreadHistoryChange {
    Create {
        item: ThreadItem,
    },
    Update {
        item: ThreadItem,
    },
    Remove {
        thread_id: ThreadId,
        turn_id: TurnId,
        item_id: ItemId,
        sequence: u64,
    },
}

impl ThreadHistoryChange {
    pub fn sequence(&self) -> u64 {
        match self {
            Self::Create { item } | Self::Update { item } => item.sequence,
            Self::Remove { sequence, .. } => *sequence,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadHistoryChangeSet {
    pub sequence: u64,
    #[serde(default)]
    pub changed_turns: Vec<Turn>,
    #[serde(default)]
    pub changed_items: Vec<ThreadItem>,
    #[serde(default)]
    pub removed_item_ids: Vec<ItemId>,
    #[serde(default)]
    pub removed_turn_ids: Vec<TurnId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollback_to_sequence: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ImageDetail, TextElement};
    use serde_json::json;

    fn item() -> ThreadItem {
        ThreadItem {
            session_id: SessionId::new("session-1"),
            thread_id: ThreadId::new("thread-1"),
            turn_id: TurnId::new("turn-1"),
            item_id: ItemId::new("item-test"),
            sequence: 3,
            ordinal: 1,
            created_at_ms: 100,
            updated_at_ms: 101,
            completed_at_ms: None,
            kind: ItemKind::AgentMessage,
            status: ItemStatus::InProgress,
            payload: ThreadItemPayload::AgentMessage {
                text: "hello".to_string(),
                phase: None,
                content_parts: vec![MessageContentPart::Text {
                    text: "hello".to_string(),
                }],
            },
            metadata: Value::Null,
        }
    }

    #[test]
    fn thread_status_matches_codex_tagged_runtime_contract() {
        let status = ThreadStatus::Active {
            active_flags: vec![ThreadActiveFlag::WaitingOnApproval],
        };
        assert!(status.is_active());
        assert_eq!(
            status.active_flags(),
            &[ThreadActiveFlag::WaitingOnApproval]
        );
        assert_eq!(
            serde_json::to_value(status).expect("serialize status"),
            json!({
                "type": "active",
                "activeFlags": ["waitingOnApproval"]
            })
        );
    }

    #[test]
    fn turn_status_does_not_encode_admission_or_queue_control() {
        assert!(TurnStatus::Completed.is_terminal());
        assert!(!TurnStatus::InProgress.is_terminal());
        assert_eq!(TurnAdmissionState::Accepted, TurnAdmissionState::Accepted);
        assert!(matches!(
            TurnQueueState::Queued { position: Some(2) },
            TurnQueueState::Queued { .. }
        ));
    }

    #[test]
    fn item_payload_is_a_tagged_typed_union() {
        let item = item();
        let encoded = serde_json::to_value(&item).expect("serialize item");
        assert_eq!(encoded["kind"], "agentMessage");
        assert_eq!(encoded["payload"]["type"], "agentMessage");
        assert_eq!(encoded["payload"]["text"], "hello");
        assert_eq!(encoded["payload"]["content_parts"][0]["type"], "text");
        assert!(encoded["payload"].get("contentParts").is_none());
        assert_eq!(encoded["metadata"], Value::Null);
        assert!(serde_json::from_value::<ThreadItem>(encoded).is_ok());
    }

    #[test]
    fn user_message_payload_round_trips_ordered_input_parts() {
        let payload = ThreadItemPayload::UserMessage {
            content: vec![
                AgentInput::Text {
                    text: "inspect".to_string(),
                    text_elements: vec![TextElement::new(0..7, None)],
                },
                AgentInput::Image {
                    uri: "https://example.com/remote.png".to_string(),
                    detail: Some(ImageDetail::High),
                },
                AgentInput::LocalImage {
                    path: "/tmp/local.png".to_string(),
                    detail: Some(ImageDetail::Original),
                },
                AgentInput::Skill {
                    name: "review".to_string(),
                    path: "/skills/review/SKILL.md".to_string(),
                },
                AgentInput::Mention {
                    name: "docs".to_string(),
                    path: "app://docs".to_string(),
                },
            ],
            client_id: Some("client-1".to_string()),
        };

        let encoded = serde_json::to_value(&payload).expect("serialize user message payload");
        assert_eq!(encoded["type"], "userMessage");
        assert_eq!(encoded["content"][0]["type"], "text");
        assert_eq!(
            encoded["content"][0]["text_elements"][0]["byteRange"],
            json!({"start": 0, "end": 7})
        );
        assert_eq!(encoded["content"][1]["type"], "image");
        assert_eq!(encoded["content"][2]["type"], "local_image");
        assert_eq!(encoded["content"][3]["type"], "skill");
        assert_eq!(encoded["content"][4]["type"], "mention");
        assert_eq!(encoded["client_id"], "client-1");
        assert_eq!(
            serde_json::from_value::<ThreadItemPayload>(encoded)
                .expect("deserialize user message payload"),
            payload
        );
        assert!(serde_json::from_value::<ThreadItemPayload>(json!({
            "type": "userMessage",
            "content": "legacy scalar content"
        }))
        .is_err());
    }

    #[test]
    fn item_payload_kind_cannot_be_guessed_from_text() {
        let payload = ThreadItemPayload::Tool {
            call_id: "call-search".to_string(),
            name: "search".to_string(),
            arguments: vec![ToolArgument {
                name: "query".to_string(),
                value: "hello".to_string(),
            }],
            output: None,
        };
        assert_eq!(payload.kind(), ItemKind::Tool);
    }

    #[test]
    fn dynamic_tool_payload_round_trips_typed_ordered_content() {
        let payload = ThreadItemPayload::DynamicToolCall {
            call_id: "call-app-info".to_string(),
            namespace: Some("desktop".to_string()),
            tool: "appInfo".to_string(),
            arguments: json!({"includeLocale": true}),
            content_items: vec![
                DynamicToolCallContentItem::InputText {
                    text: "Lime".to_string(),
                },
                DynamicToolCallContentItem::InputImage {
                    image_url: "data:image/png;base64,AA==".to_string(),
                },
                DynamicToolCallContentItem::InputAudio {
                    audio_url: "data:audio/wav;base64,AA==".to_string(),
                },
            ],
            success: Some(true),
            duration_ms: Some(7),
        };

        assert_eq!(payload.kind(), ItemKind::DynamicToolCall);
        let encoded = serde_json::to_value(&payload).expect("serialize dynamic tool payload");
        assert_eq!(encoded["type"], "dynamicToolCall");
        assert_eq!(encoded["namespace"], "desktop");
        assert_eq!(encoded["tool"], "appInfo");
        assert_eq!(encoded["arguments"], json!({"includeLocale": true}));
        assert_eq!(encoded["content_items"][0]["type"], "inputText");
        assert_eq!(
            encoded["content_items"][1]["imageUrl"],
            "data:image/png;base64,AA=="
        );
        assert_eq!(
            encoded["content_items"][2]["audioUrl"],
            "data:audio/wav;base64,AA=="
        );
        assert_eq!(
            serde_json::from_value::<ThreadItemPayload>(encoded)
                .expect("deserialize dynamic tool payload"),
            payload
        );
    }

    #[test]
    fn plan_payload_keeps_revisioned_completed_snapshot_fields() {
        let payload = ThreadItemPayload::Plan {
            text: "- [x] inspect".to_string(),
            revision_id: "proposed_plan:1".to_string(),
            source: Some("proposed_plan".to_string()),
            plan: vec![PlanStep {
                step: "inspect".to_string(),
                status: PlanStepStatus::Completed,
            }],
            explanation: None,
            tool_call_id: None,
            source_item_id: None,
        };

        assert_eq!(payload.kind(), ItemKind::Plan);
        assert_eq!(
            serde_json::to_value(payload).expect("serialize plan payload"),
            json!({
                "type": "plan",
                "text": "- [x] inspect",
                "revision_id": "proposed_plan:1",
                "source": "proposed_plan",
                "plan": [{"step": "inspect", "status": "completed"}]
            })
        );
    }

    #[test]
    fn collab_agent_state_matches_codex_app_server_wire() {
        let statuses = [
            (CollabAgentStatus::PendingInit, "pendingInit"),
            (CollabAgentStatus::Running, "running"),
            (CollabAgentStatus::Interrupted, "interrupted"),
            (CollabAgentStatus::Completed, "completed"),
            (CollabAgentStatus::Errored, "errored"),
            (CollabAgentStatus::Shutdown, "shutdown"),
            (CollabAgentStatus::NotFound, "notFound"),
        ];
        for (status, wire) in statuses {
            let state = CollabAgentState {
                status,
                message: (status == CollabAgentStatus::Errored)
                    .then(|| "provider failed".to_string()),
            };
            let encoded = serde_json::to_value(&state).expect("serialize agent state");
            assert_eq!(encoded["status"], wire);
            assert_eq!(
                serde_json::from_value::<CollabAgentState>(encoded)
                    .expect("deserialize agent state"),
                state
            );
        }
    }

    #[test]
    fn subagent_activity_accepts_only_codex_current_wire_values() {
        let values = [
            (SubAgentActivityKind::Started, "started"),
            (SubAgentActivityKind::Interacted, "interacted"),
            (SubAgentActivityKind::Interrupted, "interrupted"),
        ];

        for (activity, wire) in values {
            let encoded = serde_json::to_value(activity).expect("serialize subagent activity");
            assert_eq!(encoded, json!(wire));
            assert_eq!(
                serde_json::from_value::<SubAgentActivityKind>(encoded)
                    .expect("deserialize subagent activity"),
                activity
            );
        }

        for retired in [
            "spawned",
            "messageSent",
            "waiting",
            "resumed",
            "completed",
            "failed",
            "closed",
        ] {
            assert!(
                serde_json::from_value::<SubAgentActivityKind>(json!(retired)).is_err(),
                "retired SubAgent activity wire must fail closed: {retired}"
            );
        }
    }

    #[test]
    fn turn_contains_items_view_and_structured_terminal_error() {
        let turn = Turn {
            session_id: SessionId::new("session-1"),
            thread_id: ThreadId::new("thread-1"),
            turn_id: TurnId::new("turn-1"),
            status: TurnStatus::Failed,
            admission: TurnAdmissionState::Accepted,
            queue: TurnQueueState::Running,
            approval: TurnApprovalState::NotRequired,
            items: vec![item()],
            items_view: TurnItemsView::Full,
            error: Some(TurnError {
                message: "provider failed".to_string(),
                code: Some("provider_error".to_string()),
                details: None,
            }),
            created_at_ms: 1,
            updated_at_ms: 2,
            started_at_ms: Some(1),
            completed_at_ms: Some(2),
            duration_ms: Some(1),
        };
        assert!(turn.is_terminal());
        assert_eq!(turn.items.len(), 1);
        assert_eq!(
            turn.error.as_ref().expect("error").code.as_deref(),
            Some("provider_error")
        );
    }

    #[test]
    fn history_change_sequence_and_paging_are_stable() {
        let change = ThreadHistoryChange::Update { item: item() };
        assert_eq!(change.sequence(), 3);
        let params = ThreadItemsListParams {
            thread_id: ThreadId::new("thread-1"),
            turn_id: Some(TurnId::new("turn-1")),
            page: PageCursor {
                cursor: Some("ordinal:1".to_string()),
                limit: Some(50),
                sort_direction: SortDirection::Asc,
            },
        };
        let encoded = serde_json::to_value(params).expect("serialize paging");
        assert_eq!(encoded["cursor"], "ordinal:1");
        assert_eq!(encoded["sortDirection"], "asc");
    }

    #[test]
    fn read_and_list_params_expose_archive_and_view_contracts() {
        let list = ThreadListParams {
            page: PageCursor::default(),
            include_archived: true,
            turns_view: ThreadTurnsView::Summary,
            section: None,
            project: None,
            sort_by_section_position: false,
        };
        let read = ThreadReadParams {
            thread_id: ThreadId::new("thread-1"),
            turns_view: ThreadTurnsView::Full,
        };

        assert_eq!(
            serde_json::to_value(list).expect("serialize thread list params"),
            serde_json::json!({
                "sortDirection": "desc",
                "includeArchived": true,
                "turnsView": "summary",
                "sortBySectionPosition": false
            })
        );
        assert_eq!(
            serde_json::to_value(read).expect("serialize thread read params"),
            serde_json::json!({
                "threadId": "thread-1",
                "turnsView": "full"
            })
        );
    }
}
