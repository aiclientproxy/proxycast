use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use super::{McpToolCallError, McpToolCallResult};

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
        client_id: Option<String>,
        content: Vec<super::UserInput>,
    },
    HookPrompt {
        id: String,
        fragments: Vec<HookPromptFragment>,
    },
    AgentMessage {
        id: String,
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        phase: Option<agent_protocol::response_item::MessagePhase>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        memory_citation: Option<MemoryCitation>,
    },
    Plan {
        id: String,
        text: String,
    },
    Reasoning {
        id: String,
        #[serde(default)]
        summary: Vec<String>,
        #[serde(default)]
        content: Vec<String>,
    },
    CommandExecution {
        id: String,
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
    },
    FileChange {
        id: String,
        changes: Vec<FileUpdateChange>,
        status: PatchApplyStatus,
    },
    McpToolCall {
        id: String,
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
        tool: CollabAgentTool,
        status: CollabAgentToolCallStatus,
        sender_thread_id: String,
        receiver_thread_ids: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prompt: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning_effort: Option<String>,
        #[serde(default)]
        agents_states: HashMap<String, CollabAgentState>,
    },
    SubAgentActivity {
        id: String,
        kind: SubAgentActivityKind,
        agent_thread_id: String,
        agent_path: String,
    },
    WebSearch(WebSearchItem),
    ImageView {
        id: String,
        path: String,
    },
    Sleep(SleepItem),
    ImageGeneration(ImageGenerationItem),
    EnteredReviewMode {
        id: String,
        review: String,
    },
    ExitedReviewMode {
        id: String,
        review: String,
    },
    ContextCompaction {
        id: String,
    },
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
    pub query: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SleepItem {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ImageGenerationItem {
    pub id: String,
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
