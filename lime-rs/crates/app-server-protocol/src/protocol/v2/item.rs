use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use super::{McpToolCallError, McpToolCallResult};

/// Codex v2 assistant message delivery mode.
///
/// The current upstream contract only exposes `async`; keeping this as a
/// closed enum prevents arbitrary presentation strings from crossing the
/// protocol boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum AgentMessageDelivery {
    Async,
}

/// Typed reasoning effort carried by collaboration tool calls.
///
/// Codex accepts model-defined values in addition to its well-known levels.
/// A transparent validated wrapper preserves those values without falling
/// back to an untyped `String` field in the public Item contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
pub struct ReasoningEffort(String);

impl ReasoningEffort {
    pub fn new(value: impl Into<String>) -> Option<Self> {
        let value = value.into();
        (!value.trim().is_empty()).then_some(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for ReasoningEffort {
    fn from(value: String) -> Self {
        Self::new(value).expect("ReasoningEffort must not be empty")
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum ThreadItem {
    UserMessage {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<ThreadItemMetadata>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        client_id: Option<String>,
        content: Vec<super::UserInput>,
    },
    HookPrompt {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<ThreadItemMetadata>,
        fragments: Vec<HookPromptFragment>,
    },
    AgentMessage {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<ThreadItemMetadata>,
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        phase: Option<agent_protocol::response_item::MessagePhase>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        memory_citation: Option<MemoryCitation>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        delivery: Option<AgentMessageDelivery>,
    },
    Plan {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<ThreadItemMetadata>,
        text: String,
    },
    Reasoning {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<ThreadItemMetadata>,
        #[serde(default)]
        summary: Vec<String>,
        #[serde(default)]
        content: Vec<String>,
    },
    CommandExecution {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<ThreadItemMetadata>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        plugin_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        script_path: Option<String>,
        command: String,
        cwd: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        process_id: Option<String>,
        #[serde(default)]
        source: CommandExecutionSource,
        status: CommandExecutionStatus,
        #[serde(default)]
        command_actions: Vec<CommandAction>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        aggregated_output: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        exit_code: Option<i32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        duration_ms: Option<i64>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        terminal_interactions: Vec<CommandExecutionTerminalInteraction>,
    },
    FileChange {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<ThreadItemMetadata>,
        changes: Vec<FileUpdateChange>,
        status: PatchApplyStatus,
    },
    McpToolCall {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<ThreadItemMetadata>,
        server: String,
        tool: String,
        status: McpToolCallStatus,
        arguments: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        app_context: Option<McpToolCallAppContext>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mcp_app_resource_uri: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        plugin_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        read_only_hint: Option<bool>,
        #[schemars(required, extend("type" = ["object", "null"]))]
        result: Option<Box<McpToolCallResult>>,
        #[schemars(required, extend("type" = ["object", "null"]))]
        error: Option<McpToolCallError>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        duration_ms: Option<i64>,
    },
    DynamicToolCall {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<ThreadItemMetadata>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        namespace: Option<String>,
        tool: String,
        arguments: Value,
        status: DynamicToolCallStatus,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content_items: Option<Vec<DynamicToolCallOutputContentItem>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        success: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        duration_ms: Option<i64>,
    },
    CollabAgentToolCall {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<ThreadItemMetadata>,
        tool: CollabAgentTool,
        status: CollabAgentToolCallStatus,
        sender_thread_id: String,
        receiver_thread_ids: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prompt: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning_effort: Option<ReasoningEffort>,
        #[serde(default)]
        agents_states: HashMap<String, CollabAgentState>,
    },
    SubAgentActivity {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<ThreadItemMetadata>,
        kind: SubAgentActivityKind,
        agent_thread_id: String,
        agent_path: String,
    },
    WebSearch(WebSearchItem),
    ImageView {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<ThreadItemMetadata>,
        path: String,
    },
    Sleep(SleepItem),
    ImageGeneration(ImageGenerationItem),
    EnteredReviewMode {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<ThreadItemMetadata>,
        review: String,
    },
    ExitedReviewMode {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<ThreadItemMetadata>,
        review: String,
    },
    ContextCompaction {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<ThreadItemMetadata>,
    },
    UnknownItem {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<ThreadItemMetadata>,
        upstream_type: String,
        field_names: Vec<String>,
    },
}

/// Safe, display-only provenance carried by imported historical items.
///
/// The canonical store keeps richer metadata, but only these stable markers
/// cross the v2 boundary. In particular, raw source payloads and capabilities
/// are intentionally excluded.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadItemMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imported: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imported_read_only: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imported_synthetic: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imported_incomplete: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imported_synthetic_id: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_client: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_thread_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_event_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_event_seq: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_provenance: Option<ThreadItemSourceProvenance>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadItemSourceProvenance {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_client: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_thread_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_event_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_event_seq: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_payload_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_channel: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CommandExecutionTerminalInteraction {
    pub process_id: String,
    pub stdin: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct HookPromptFragment {
    pub text: String,
    pub hook_run_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MemoryCitation {
    pub entries: Vec<MemoryCitationEntry>,
    pub thread_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MemoryCitationEntry {
    pub path: String,
    pub line_start: u32,
    pub line_end: u32,
    pub note: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum CommandAction {
    Read {
        command: String,
        name: String,
        path: String,
    },
    ListFiles {
        command: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<String>,
    },
    Search {
        command: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        query: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<String>,
    },
    Unknown {
        command: String,
    },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum CommandExecutionSource {
    #[default]
    Agent,
    UserShell,
    UnifiedExecStartup,
    UnifiedExecInteraction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum CommandExecutionStatus {
    InProgress,
    Completed,
    Failed,
    Declined,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct FileUpdateChange {
    pub path: String,
    pub kind: PatchChangeKind,
    pub diff: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum PatchChangeKind {
    Add,
    Delete,
    Update {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        move_path: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum PatchApplyStatus {
    InProgress,
    Completed,
    Failed,
    Declined,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
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
    pub template_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action_name: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum McpToolCallStatus {
    InProgress,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum DynamicToolCallStatus {
    InProgress,
    Completed,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DynamicToolCallParams {
    pub thread_id: String,
    pub turn_id: String,
    pub call_id: String,
    pub namespace: Option<String>,
    pub tool: String,
    pub arguments: Value,
    pub phase: DynamicToolCallPhase,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_token: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum DynamicToolCallPhase {
    Preflight,
    ApprovedExecute,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DynamicToolCallResponse {
    pub content_items: Vec<DynamicToolCallOutputContentItem>,
    pub success: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval: Option<DynamicToolCallApproval>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DynamicToolCallApproval {
    pub approval_token: String,
    pub reason: String,
    pub risk_class: String,
    pub action_kind: String,
    pub browser_session_id: String,
    pub tab_id: String,
    pub view_id: String,
    pub web_contents_id: u64,
    pub snapshot_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_node_id: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum DynamicToolCallOutputContentItem {
    InputText { text: String },
    InputImage { image_url: String },
    InputAudio { audio_url: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum CollabAgentTool {
    SpawnAgent,
    SendInput,
    ResumeAgent,
    Wait,
    CloseAgent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum CollabAgentToolCallStatus {
    InProgress,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum SubAgentActivityKind {
    Started,
    Interacted,
    Interrupted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CollabAgentState {
    pub status: CollabAgentStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct WebSearchItem {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ThreadItemMetadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SleepItem {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ThreadItemMetadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ImageGenerationItem {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ThreadItemMetadata>,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
    pub result: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub saved_path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ItemStartedNotification {
    pub item: ThreadItem,
    pub thread_id: String,
    pub turn_id: String,
    pub started_at_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ItemCompletedNotification {
    pub item: ThreadItem,
    pub thread_id: String,
    pub turn_id: String,
    pub completed_at_ms: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum GuardianApprovalReviewStatus {
    InProgress,
    Approved,
    Denied,
    TimedOut,
    Aborted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum AutoReviewDecisionSource {
    Agent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum GuardianRiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum GuardianUserAuthorization {
    Unknown,
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct GuardianCommandReviewAction {
    pub source: String,
    pub command: String,
    pub cwd: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum GuardianApprovalReviewAction {
    Command {
        source: String,
        command: String,
        cwd: String,
    },
    Execve {
        source: String,
        program: String,
        argv: Vec<String>,
        cwd: String,
    },
    ApplyPatch {
        cwd: String,
        files: Vec<String>,
    },
    NetworkAccess {
        target: String,
        host: String,
        protocol: String,
        port: u16,
    },
    McpToolCall {
        server: String,
        tool_name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        connector_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        connector_name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_title: Option<String>,
    },
    RequestPermissions {
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
        permissions: Value,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct GuardianApprovalReview {
    pub status: GuardianApprovalReviewStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub risk_level: Option<GuardianRiskLevel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_authorization: Option<GuardianUserAuthorization>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ItemGuardianApprovalReviewStartedNotification {
    pub thread_id: String,
    pub turn_id: String,
    pub started_at_ms: i64,
    pub review_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_item_id: Option<String>,
    pub review: GuardianApprovalReview,
    pub action: GuardianApprovalReviewAction,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ItemGuardianApprovalReviewCompletedNotification {
    pub thread_id: String,
    pub turn_id: String,
    pub started_at_ms: i64,
    pub completed_at_ms: i64,
    pub review_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_item_id: Option<String>,
    pub decision_source: AutoReviewDecisionSource,
    pub review: GuardianApprovalReview,
    pub action: GuardianApprovalReviewAction,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AgentMessageDeltaNotification {
    pub thread_id: String,
    pub turn_id: String,
    pub item_id: String,
    pub delta: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CommandExecutionOutputDeltaNotification {
    pub thread_id: String,
    pub turn_id: String,
    pub item_id: String,
    pub delta: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CommandExecutionTerminalInteractionNotification {
    pub thread_id: String,
    pub turn_id: String,
    pub item_id: String,
    pub process_id: String,
    pub stdin: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct FileChangePatchUpdatedNotification {
    pub thread_id: String,
    pub turn_id: String,
    pub item_id: String,
    pub changes: Vec<FileUpdateChange>,
}

/// EXPERIMENTAL - proposed plan streaming deltas for plan items. Clients should
/// not assume concatenated deltas match the completed plan item content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PlanDeltaNotification {
    pub thread_id: String,
    pub turn_id: String,
    pub item_id: String,
    pub delta: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ReasoningSummaryTextDeltaNotification {
    pub thread_id: String,
    pub turn_id: String,
    pub item_id: String,
    pub delta: String,
    pub summary_index: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ReasoningSummaryPartAddedNotification {
    pub thread_id: String,
    pub turn_id: String,
    pub item_id: String,
    pub summary_index: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ReasoningTextDeltaNotification {
    pub thread_id: String,
    pub turn_id: String,
    pub item_id: String,
    pub delta: String,
    pub content_index: i64,
}
