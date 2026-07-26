use super::{
    SortDirection, Thread, ThreadHistoryMode, ThreadItem, ThreadListCwdFilter, ThreadSortKey,
    ThreadSourceKind, ThreadStartSource, ThreadStatus, Turn, TurnEnvironmentParams, TurnItemsView,
};
use agent_protocol::MultiAgentMode;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadStartParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_provider: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub allow_provider_model_fallback: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<Option<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_workspace_roots: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_policy: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approvals_reviewer: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub permissions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<HashMap<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub developer_instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub personality: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multi_agent_mode: Option<MultiAgentMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ephemeral: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub history_mode: Option<ThreadHistoryMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_start_source: Option<ThreadStartSource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environments: Option<Vec<TurnEnvironmentParams>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynamic_tools: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_capability_roots: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub experimental_raw_events: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadStartResponse {
    pub thread: Thread,
    pub model: String,
    pub model_provider: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    pub cwd: String,
    #[serde(default)]
    pub runtime_workspace_roots: Vec<String>,
    #[serde(default)]
    pub instruction_sources: Vec<String>,
    pub approval_policy: Value,
    pub approvals_reviewer: Value,
    pub sandbox: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_permission_profile: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub multi_agent_mode: MultiAgentMode,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadForkParams {
    pub thread_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_turn_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before_turn_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<Option<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_workspace_roots: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_policy: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approvals_reviewer: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub permissions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<HashMap<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub developer_instructions: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub ephemeral: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_source: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub exclude_turns: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub defer_goal_continuation: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadForkResponse {
    pub thread: Thread,
    pub model: String,
    pub model_provider: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    pub cwd: String,
    #[serde(default)]
    pub runtime_workspace_roots: Vec<String>,
    #[serde(default)]
    pub instruction_sources: Vec<String>,
    pub approval_policy: Value,
    pub approvals_reviewer: Value,
    pub sandbox: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_permission_profile: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub multi_agent_mode: MultiAgentMode,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadResumeParams {
    pub thread_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub history: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<Option<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_workspace_roots: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_policy: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approvals_reviewer: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub permissions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<HashMap<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub developer_instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub personality: Option<Value>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub exclude_turns: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_turns_page: Option<ThreadResumeInitialTurnsPageParams>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadResumeResponse {
    pub thread: Thread,
    pub model: String,
    pub model_provider: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    pub cwd: String,
    #[serde(default)]
    pub runtime_workspace_roots: Vec<String>,
    #[serde(default)]
    pub instruction_sources: Vec<String>,
    pub approval_policy: Value,
    pub approvals_reviewer: Value,
    pub sandbox: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_permission_profile: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub multi_agent_mode: MultiAgentMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_turns_page: Option<TurnsPage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turns_backwards_cursor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub items_backwards_cursor: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadResumeInitialTurnsPageParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sort_direction: Option<SortDirection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub items_view: Option<TurnItemsView>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TurnsPage {
    pub data: Vec<Turn>,
    pub next_cursor: Option<String>,
    pub backwards_cursor: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadReadParams {
    pub thread_id: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub include_turns: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadReadResponse {
    pub thread: Thread,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadInjectItemsParams {
    pub thread_id: String,
    /// Raw Responses API items appended to model-visible history without a user turn.
    pub items: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadInjectItemsResponse {}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadListParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sort_key: Option<ThreadSortKey>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sort_direction: Option<SortDirection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_providers: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_kinds: Option<Vec<ThreadSourceKind>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub archived: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_pinned: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<ThreadListCwdFilter>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub use_state_db_only: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_term: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_thread_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ancestor_thread_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadListResponse {
    pub data: Vec<Thread>,
    pub next_cursor: Option<String>,
    pub backwards_cursor: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadSearchParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sort_key: Option<ThreadSortKey>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sort_direction: Option<SortDirection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_kinds: Option<Vec<ThreadSourceKind>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub archived: Option<bool>,
    pub search_term: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadSearchResult {
    pub thread: Thread,
    pub snippet: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadSearchResponse {
    pub data: Vec<ThreadSearchResult>,
    pub next_cursor: Option<String>,
    pub backwards_cursor: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadLoadedListParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadLoadedListResponse {
    pub data: Vec<String>,
    pub next_cursor: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadArchiveParams {
    pub thread_id: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadArchiveResponse {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadDeleteParams {
    pub thread_id: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadDeleteResponse {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadUnsubscribeParams {
    pub thread_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadUnsubscribeResponse {
    pub status: ThreadUnsubscribeStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum ThreadUnsubscribeStatus {
    NotLoaded,
    NotSubscribed,
    Unsubscribed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadUnarchiveParams {
    pub thread_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadUnarchiveResponse {
    pub thread: Thread,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadSetNameParams {
    pub thread_id: String,
    pub name: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadSetNameResponse {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadMetadataUpdateParams {
    pub thread_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_info: Option<ThreadMetadataGitInfoUpdateParams>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_pinned: Option<bool>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadMetadataGitInfoUpdateParams {
    #[serde(
        default,
        deserialize_with = "deserialize_double_option_string",
        skip_serializing_if = "Option::is_none"
    )]
    pub sha: Option<Option<String>>,
    #[serde(
        default,
        deserialize_with = "deserialize_double_option_string",
        skip_serializing_if = "Option::is_none"
    )]
    pub branch: Option<Option<String>>,
    #[serde(
        default,
        deserialize_with = "deserialize_double_option_string",
        skip_serializing_if = "Option::is_none"
    )]
    pub origin_url: Option<Option<String>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadMetadataUpdateResponse {
    pub thread: Thread,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadCompactStartParams {
    pub thread_id: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadCompactStartResponse {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadShellCommandParams {
    pub thread_id: String,
    pub command: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadShellCommandResponse {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadBackgroundTerminalsCleanParams {
    pub thread_id: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadBackgroundTerminalsCleanResponse {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadBackgroundTerminalsListParams {
    pub thread_id: String,
    pub cursor: Option<String>,
    pub limit: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadBackgroundTerminal {
    pub item_id: String,
    pub process_id: String,
    pub command: String,
    pub cwd: String,
    pub os_pid: Option<u32>,
    pub cpu_percent: Option<f64>,
    pub rss_kb: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadBackgroundTerminalsListResponse {
    pub data: Vec<ThreadBackgroundTerminal>,
    pub next_cursor: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadBackgroundTerminalsTerminateParams {
    pub thread_id: String,
    pub process_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadBackgroundTerminalsTerminateResponse {
    pub terminated: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadTurnsListParams {
    pub thread_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sort_direction: Option<SortDirection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub items_view: Option<TurnItemsView>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadTurnsListResponse {
    pub data: Vec<Turn>,
    pub next_cursor: Option<String>,
    pub backwards_cursor: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadItemsListParams {
    pub thread_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sort_direction: Option<SortDirection>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadItemEntry {
    pub turn_id: String,
    pub item: ThreadItem,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadItemsListResponse {
    pub data: Vec<ThreadItemEntry>,
    pub next_cursor: Option<String>,
    pub backwards_cursor: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadSearchOccurrencesParams {
    pub thread_id: String,
    pub search_term: String,
    pub cursor: Option<String>,
    pub limit: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadSearchTextRange {
    pub start: u32,
    pub end: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadSearchOccurrence {
    pub turn_id: String,
    pub item_id: String,
    pub snippet: String,
    pub snippet_match_range: ThreadSearchTextRange,
    pub turn_cursor: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadSearchOccurrencesResponse {
    pub data: Vec<ThreadSearchOccurrence>,
    pub next_cursor: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadTokenUsageUpdatedNotification {
    pub thread_id: String,
    pub turn_id: String,
    pub token_usage: ThreadTokenUsage,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadTokenUsage {
    pub total: TokenUsageBreakdown,
    pub last: TokenUsageBreakdown,
    pub model_context_window: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TokenUsageBreakdown {
    pub total_tokens: i64,
    pub input_tokens: i64,
    pub cached_input_tokens: i64,
    #[serde(default)]
    pub cache_write_input_tokens: i64,
    pub output_tokens: i64,
    pub reasoning_output_tokens: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum ThreadGoalStatus {
    Active,
    Paused,
    Blocked,
    UsageLimited,
    BudgetLimited,
    Complete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadGoal {
    pub thread_id: String,
    pub objective: String,
    pub status: ThreadGoalStatus,
    pub token_budget: Option<i64>,
    pub tokens_used: i64,
    pub time_used_seconds: i64,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadGoalSetParams {
    pub thread_id: String,
    #[serde(default)]
    pub objective: Option<String>,
    #[serde(default)]
    pub status: Option<ThreadGoalStatus>,
    #[serde(
        default,
        deserialize_with = "deserialize_double_option_i64",
        skip_serializing_if = "Option::is_none"
    )]
    pub token_budget: Option<Option<i64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadGoalSetResponse {
    pub goal: ThreadGoal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadGoalGetParams {
    pub thread_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadGoalGetResponse {
    pub goal: Option<ThreadGoal>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadGoalClearParams {
    pub thread_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadGoalClearResponse {
    pub cleared: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadStartedNotification {
    pub thread: Thread,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadArchivedNotification {
    pub thread_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadDeletedNotification {
    pub thread_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadUnarchivedNotification {
    pub thread_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadClosedNotification {
    pub thread_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadNameUpdatedNotification {
    pub thread_id: String,
    pub thread_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadStatusChangedNotification {
    pub thread_id: String,
    pub status: ThreadStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadGoalUpdatedNotification {
    pub thread_id: String,
    pub turn_id: Option<String>,
    pub goal: ThreadGoal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadGoalClearedNotification {
    pub thread_id: String,
}

fn deserialize_double_option_i64<'de, D>(deserializer: D) -> Result<Option<Option<i64>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<i64>::deserialize(deserializer).map(Some)
}

fn deserialize_double_option_string<'de, D>(
    deserializer: D,
) -> Result<Option<Option<String>>, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<String>::deserialize(deserializer).map(Some)
}
